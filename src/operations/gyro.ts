import { Vector3, Ray } from "three";
import {
  type Mesh,
  type DCEL,
  type HEFace,
  type HalfEdge,
  outgoingHalfEdges,
} from "../geometry/HalfEdge";
import {
  type Polyhedron,
  faceCentroidHE,
  faceNormalHE,
} from "../geometry/polyhedron";
import { type MorphPlan } from "./types";
import { type InViewTest } from "./truncate";
import { joinHeight } from "./kis";
import { closestLineParam, distancePointToRay } from "../util/lines";

// How far (as a fraction of the centre → edge-midpoint line) the peripheral
// vertices slide at full skew. Kept well short of the midpoint so they stay lifted
// above the face (the centre sits at the raised kis-join apex), giving the relaxer a
// good, non-degenerate starting cap. The exact merged-face planarity is left to the
// post-release solver — the regular target (e.g. the dodecahedron's vertices) does
// not lie on these lines, so there is no closed-form "coplanar" distance to hit.
const GYRO_SLIDE = 0.5;

/** Half-edges around a face, in order. */
function faceHalfEdges(f: HEFace): HalfEdge[] {
  const out: HalfEdge[] = [];
  let h = f.halfedge;
  const start = h;
  do {
    out.push(h);
    h = h.next;
  } while (h !== start);
  return out;
}

function faceLoop(f: HEFace): number[] {
  return faceHalfEdges(f).map((h) => h.origin.id);
}

/** Outward unit normals of the two faces sharing half-edge `h` (centroid-oriented). */
function edgeFaceNormals(h: HalfEdge): Vector3[] {
  const faces = h.twin ? [h.face, h.twin.face] : [h.face];
  return faces.map((f) => {
    const n = faceNormalHE(f);
    if (n.dot(faceCentroidHE(f)) < 0) n.negate();
    return n;
  });
}

/** Are the gyred faces edge-connected to one another? Required so the chirality has
 *  a single coherent twist (one connected patch, no arbitrary per-component choice). */
function selectionConnected(dcel: DCEL, gyred: Set<number>): boolean {
  if (gyred.size <= 1) return true;
  const start = gyred.values().next().value as number;
  const seen = new Set<number>([start]);
  const queue = [start];
  while (queue.length) {
    const f = dcel.faces[queue.shift()!];
    for (const h of faceHalfEdges(f)) {
      const g = h.twin?.face.id;
      if (g !== undefined && gyred.has(g) && !seen.has(g)) {
        seen.add(g);
        queue.push(g);
      }
    }
  }
  return seen.size === gyred.size;
}

/**
 * Proper 2-coloring of the VERTICES (adjacent vertices get opposite colors),
 * restricted to the vertices of the gyred faces and seeded at the dragged face. Only
 * edges that border a gyred face carry the constraint, so the coloring is decided
 * entirely WITHIN the (connected) selection and twists coherently no matter the
 * parity of the rest of the solid. `bipartite` is false exactly when the region has
 * an odd cycle (no 2-coloring exists), in which case the caller refuses the operation
 * — there is no consistent chirality to choose.
 */
function twoColorVerticesInRegion(
  dcel: DCEL,
  gyred: Set<number>,
  seedVid: number,
): { color: Map<number, 0 | 1>; bipartite: boolean } {
  const color = new Map<number, 0 | 1>();
  let bipartite = true;
  const constrains = (h: HalfEdge): boolean =>
    gyred.has(h.face.id) || (h.twin !== null && gyred.has(h.twin.face.id));
  color.set(seedVid, 0);
  const queue = [seedVid];
  while (queue.length) {
    const vid = queue.shift()!;
    const c = color.get(vid)!;
    for (const h of outgoingHalfEdges(dcel.vertices[vid])) {
      if (!constrains(h)) continue;
      const n = h.next.origin.id; // neighbour across this gyred-face edge
      const nc = color.get(n);
      if (nc === undefined) {
        color.set(n, (c ^ 1) as 0 | 1);
        queue.push(n);
      } else if (nc === c) {
        bipartite = false;
      }
    }
  }
  return { color, bipartite };
}

/** Per-face data for one chiral variant. */
interface GFace {
  faceId: number;
  center: number; // apex vertex index, or -1 when n=2 (collapsed to an edge)
  apex: Vector3; // the raised centre = centroid + (baseT·kis-join-height)·normal
  qIdx: number[]; // peripheral vertex indices q_0..q_{n-1}
  qTarget: Vector3[]; // edge midpoints each q heads toward (the snap-line far ends)
  qHe: HalfEdge[]; // the boundary half-edge each q sits over (for in-view tests)
}

/**
 * Gyro, driven by holding Shift WHILE dragging a 2n-gon face (i.e. mid-kis). The
 * dual of Snub. The single kis apex per face is split into a central degree-n vertex
 * C surrounded by n degree-3 peripheral vertices q; with Shift held, each q is
 * dragged out along the invisible line from C to the midpoint of one of the 2n
 * boundary edges, and connects to C and that edge's two endpoints. When n=2 the
 * centre degenerates to "just an edge" (no C vertex).
 *
 * `baseT` is the frozen kis level at the moment Shift was pressed: C sits at the kis
 * apex for that level (centroid + baseT·joinHeight·normal), so at skew=0 (all q's at
 * C) the cap reproduces the current kis exactly — pressing Shift changes nothing
 * until the mouse moves. The skew (the plan's `t`) then slides the q's outward.
 * Whether the result is a "partial gyro" or a full gyro is decided by the caller via
 * the `weld` flag at commit (inherited from whether the base kis reached a full join)
 * — the topology is otherwise independent of how far the q's are dragged.
 *
 * Model: like Kis, BY DEFAULT every face participates (the dragged face is the
 * handle); a multi-select restricts the set, which must be edge-connected (so the
 * chirality has one coherent twist with no arbitrary per-component choice). Each face
 * must have an EVEN number of sides — otherwise it can't be tiled into the
 * alternating pattern — so we throw (the controller makes that a no-op).
 *
 * A 2n-gon becomes n pentagons (meeting at C) + n triangles, alternating so every
 * original edge stays a shared edge; at the welded max each original edge dissolves,
 * merging a pentagon (one face) with a triangle (its neighbour) into a larger face.
 *
 * Chirality is LIVE: both mirror forms are precomputed (the two vertex 2-colorings
 * pick which alternating edges receive a q); `snap` chooses whichever puts a q on the
 * line nearest the cursor, so aiming at an adjacent edge flips the whole twist.
 */
export function buildGyro(
  poly: Polyhedron,
  draggedFid: number,
  selected: Set<number> | null,
  inView: InViewTest | null = null,
  baseT = 1,
): MorphPlan {
  const dcel = poly.dcel;
  const V = dcel.vertices.length;

  const gyred = new Set<number>(
    selected && selected.size > 0 ? selected : dcel.faces.map((f) => f.id),
  );
  gyred.add(draggedFid); // the handle always participates

  // Preconditions (mirrored by `canGyro` for the UI): the selection must be edge-
  // connected, and its vertex region must be 2-colorable. A 2-coloring fails exactly
  // when that region has an ODD CYCLE — which also covers any odd-sided face (its
  // boundary IS an odd cycle), so no separate parity check is needed. Without a
  // coherent coloring there's no consistent chirality, so we refuse outright.
  if (!selectionConnected(dcel, gyred)) {
    throw new Error("Gyro needs the selected faces to be connected.");
  }
  // Chirality is decided within the selected patch, seeded at the dragged face.
  const { color, bipartite } = twoColorVerticesInRegion(
    dcel,
    gyred,
    dcel.faces[draggedFid].halfedge.origin.id,
  );
  if (!bipartite) {
    throw new Error("Gyro needs a 2-colorable selection (the patch has an odd cycle).");
  }

  // ---- The raised centre of each face = the kis apex at the frozen level ----------
  // Height = baseT × the kis-join height (max over gyred neighbours of the
  // coplanarity height, with kis's same fallback), so at baseT=1 the centre is the
  // full join apex and at baseT<1 it tracks a partial kis.
  const apexHeight = new Map<number, number>();
  for (const f of dcel.faces) {
    if (!gyred.has(f.id)) continue;
    const cf = faceCentroidHE(f);
    const nf = faceNormalHE(f);
    let h = 0;
    for (const he of faceHalfEdges(f)) {
      const g = he.twin!.face;
      if (!gyred.has(g.id)) continue;
      const solved = joinHeight(
        he.origin.position,
        he.next.origin.position,
        cf,
        nf,
        faceCentroidHE(g),
        faceNormalHE(g),
      );
      if (solved && solved > 1e-6) h = Math.max(h, solved);
    }
    if (h <= 1e-6) h = 0.5 * cf.distanceTo(f.halfedge.origin.position);
    apexHeight.set(f.id, h);
  }
  const apexOf = (f: HEFace): Vector3 =>
    faceCentroidHE(f)
      .add(faceNormalHE(f).multiplyScalar(baseT * apexHeight.get(f.id)!));

  // ---- Build one chiral variant ---------------------------------------------------
  // `startColor` picks which boundary vertex the alternation starts on (the two
  // colors give the two mirror twists). Vertex indexing (apex + q's, from V upward)
  // is identical across variants — only the q targets and the tiling differ.
  function buildVariant(startColor: 0 | 1): {
    gfaces: GFace[];
    previewFaces: number[][];
    vertexCount: number;
  } {
    const gfaces: GFace[] = [];
    const previewFaces: number[][] = [];
    let idx = V;

    for (const f of dcel.faces) {
      if (!gyred.has(f.id)) {
        previewFaces.push(faceLoop(f)); // untouched face
        continue;
      }
      const bh = faceHalfEdges(f);
      const m = bh.length; // = 2n
      const n = m / 2;

      // Start the boundary order at a `startColor` vertex so the pentagon/triangle
      // assignment stays consistent across shared edges (coherent chirality). The
      // coloring is coherent (guaranteed above), so such a vertex always exists.
      let s = 0;
      for (let i = 0; i < m; i++) {
        if (color.get(bh[i].origin.id) === startColor) {
          s = i;
          break;
        }
      }
      const P: number[] = []; // boundary vertex ids p_0..p_{2n-1}
      for (let i = 0; i < m; i++) P.push(bh[(s + i) % m].origin.id);

      const apex = apexOf(f);
      const hasCenter = n >= 3;
      const center = hasCenter ? idx++ : -1;
      const qIdx: number[] = [];
      const qTarget: Vector3[] = [];
      const qHe: HalfEdge[] = [];
      for (let j = 0; j < n; j++) {
        qIdx.push(idx++);
        // q_j sits over the boundary edge (p_{2j-1}, p_{2j}); head toward its midpoint.
        const he = bh[(s + 2 * j - 1 + m) % m];
        qHe.push(he);
        const a = dcel.vertices[P[(2 * j - 1 + m) % m]].position;
        const b = dcel.vertices[P[2 * j]].position;
        qTarget.push(a.clone().add(b).multiplyScalar(0.5));
      }

      // Tiling: per j, a pentagon (or quad when n=2) meeting at C and a triangle.
      for (let j = 0; j < n; j++) {
        const pent = hasCenter
          ? [center, qIdx[j], P[2 * j], P[2 * j + 1], qIdx[(j + 1) % n]]
          : [qIdx[j], P[2 * j], P[2 * j + 1], qIdx[(j + 1) % n]];
        previewFaces.push(pent);
        previewFaces.push([qIdx[(j + 1) % n], P[2 * j + 1], P[(2 * j + 2) % m]]);
      }

      gfaces.push({ faceId: f.id, center, apex, qIdx, qTarget, qHe });
    }

    return { gfaces, previewFaces, vertexCount: idx };
  }

  const variants = ([0, 1] as const).map((startColor) => {
    const built = buildVariant(startColor);
    const dragged = built.gfaces.find((g) => g.faceId === draggedFid)!;
    return { ...built, dragged };
  });
  let currentVariant = 0;

  function positions(skew: number): Vector3[] {
    const variant = variants[currentVariant];
    const out: Vector3[] = new Array(variant.vertexCount);
    for (let i = 0; i < V; i++) out[i] = dcel.vertices[i].position.clone();
    for (const g of variant.gfaces) {
      if (g.center >= 0) out[g.center] = g.apex.clone(); // centre stays at the apex
      for (let j = 0; j < g.qIdx.length; j++) {
        // q slides from the apex toward its edge midpoint; reaches GYRO_SLIDE at skew=1.
        out[g.qIdx[j]] = g.apex.clone().lerp(g.qTarget[j], skew * GYRO_SLIDE);
      }
    }
    return out;
  }

  // ---- Welded max: dissolve every original edge shared by two gyred faces ---------
  // Each tiling face owns exactly one such boundary edge; across it sit a
  // pentagon/quad and a triangle, which merge into one larger face.
  const edgeKey = (a: number, b: number) => (a < b ? `${a}_${b}` : `${b}_${a}`);
  const dissolve = new Set<string>();
  for (const he of dcel.halfedges) {
    if (!he.twin || he.id >= he.twin.id) continue;
    if (gyred.has(he.face.id) && gyred.has(he.twin.face.id)) {
      dissolve.add(edgeKey(he.origin.id, he.next.origin.id));
    }
  }

  function weldedFaces(faces: number[][]): number[][] {
    // Where does each dissolved edge appear (as a consecutive original-vertex pair)?
    const occ = new Map<string, Array<{ fi: number; i: number }>>();
    for (let fi = 0; fi < faces.length; fi++) {
      const loop = faces[fi];
      for (let i = 0; i < loop.length; i++) {
        const a = loop[i];
        const b = loop[(i + 1) % loop.length];
        if (a < V && b < V && dissolve.has(edgeKey(a, b))) {
          (occ.get(edgeKey(a, b)) ?? occ.set(edgeKey(a, b), []).get(edgeKey(a, b))!).push({ fi, i });
        }
      }
    }
    const consumed = new Set<number>();
    const out: number[][] = [];
    for (const list of occ.values()) {
      if (list.length !== 2) continue; // boundary of a partial selection: leave intact
      const [F, G] = list;
      consumed.add(F.fi);
      consumed.add(G.fi);
      const lf = faces[F.fi];
      const lg = faces[G.fi];
      const a = lf[F.i]; // F: [a, b, ...fRest];  G traverses b->a: [b, a, ...gRest]
      const b = lf[(F.i + 1) % lf.length];
      const fRest: number[] = [];
      for (let k = 2; k < lf.length; k++) fRest.push(lf[(F.i + k) % lf.length]);
      const gRest: number[] = [];
      for (let k = 2; k < lg.length; k++) gRest.push(lg[(G.i + k) % lg.length]);
      out.push([a, ...gRest, b, ...fRest]);
    }
    for (let fi = 0; fi < faces.length; fi++) {
      if (!consumed.has(fi)) out.push(faces[fi].slice());
    }
    return out;
  }

  // ---- Snapping: the dragged q rides the line from the apex to a boundary-edge
  // midpoint. Both variants' q-lines together cover all 2n boundary edges; the
  // nearest one picks the chiral form, the handle q, and how far out it has slid.
  // The mouse begins at the apex (= the kis handle), so skew starts at ~0.
  function snap(ray: Ray): {
    t: number;
    point: Vector3;
    highlight?: { a: Vector3; b: Vector3 };
  } {
    type Cand = { variant: number; point: Vector3; max: Vector3; t: number; dist: number };
    let best: Cand | null = null;
    let visibleBest: Cand | null = null;
    for (let v = 0; v < 2; v++) {
      const dg = variants[v].dragged;
      for (let j = 0; j < dg.qTarget.length; j++) {
        const dir = dg.qTarget[j].clone().sub(dg.apex);
        if (dir.lengthSq() < 1e-12) continue;
        let slide = closestLineParam(dg.apex, dir, ray.origin, ray.direction);
        slide = Math.max(0, Math.min(GYRO_SLIDE, slide));
        const point = dg.apex.clone().addScaledVector(dir, slide);
        const dist = distancePointToRay(point, ray);
        const cand: Cand = {
          variant: v,
          point,
          max: dg.apex.clone().addScaledVector(dir, GYRO_SLIDE),
          t: slide / GYRO_SLIDE,
          dist,
        };
        if (!best || dist < best.dist) best = cand;
        if (!inView || inView(dg.qTarget[j], edgeFaceNormals(dg.qHe[j]))) {
          if (!visibleBest || dist < visibleBest.dist) visibleBest = cand;
        }
      }
    }
    const chosen = visibleBest ?? best;
    if (!chosen) {
      const p = variants[0].dragged.apex.clone();
      return { t: 0, point: p, highlight: { a: p, b: p.clone() } };
    }
    currentVariant = chosen.variant;
    return {
      t: Math.max(0, Math.min(1, chosen.t)),
      point: chosen.point,
      highlight: { a: chosen.point.clone(), b: chosen.max },
    };
  }

  function commit(skew: number, weld: boolean): Mesh {
    const faces = variants[currentVariant].previewFaces;
    return {
      vertices: positions(skew),
      faces: weld ? weldedFaces(faces) : faces.map((f) => f.slice()),
    };
  }

  return {
    kind: "gyro",
    get previewFaces() {
      return variants[currentVariant].previewFaces;
    },
    positions,
    snap,
    commit,
  };
}

/**
 * Whether gyro can be applied COHERENTLY to `selected` faces (or all when null).
 * Mirrors buildGyro's preconditions exactly: the participating faces must be edge-
 * connected and their vertex region must be 2-colorable (no odd cycle — which also
 * rules out odd-sided faces). Cheap; intended for the UI availability hint.
 */
export function canGyro(poly: Polyhedron, selected: Set<number> | null): boolean {
  const dcel = poly.dcel;
  const gyred = new Set<number>(
    selected && selected.size > 0 ? selected : dcel.faces.map((f) => f.id),
  );
  if (gyred.size === 0 || !selectionConnected(dcel, gyred)) return false;
  const seed = gyred.values().next().value as number;
  return twoColorVerticesInRegion(dcel, gyred, dcel.faces[seed].halfedge.origin.id)
    .bipartite;
}

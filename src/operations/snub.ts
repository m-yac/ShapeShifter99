import { Vector3, Ray } from "three";
import {
  type Mesh,
  type DCEL,
  type HalfEdge,
  outgoingHalfEdges,
} from "../geometry/HalfEdge";
import {
  type Polyhedron,
  faceNormalHE,
  faceCentroidHE,
} from "../geometry/polyhedron";
import { type MorphPlan } from "./types";
import { type InViewTest } from "./truncate";
import { weldVertexPairs } from "./weld";
import { closestLineParam, distancePointToRay } from "../util/lines";

// Cut fraction along an edge that an "outer" (triangle-only) cut vertex reaches at
// t=1, and the smaller fraction the "inner" (n-gon) cut vertices reach. The gap is
// what skews the truncated 2n-gon into the snub form. They sum to 1 so that at t=1
// the outer cut vertex from one end of an edge exactly meets the inner cut vertex
// from the other end — the welded max (e.g. snub of the octahedron → icosahedron).
const F_OUT = 0.65;
const F_IN = 0.35;

/** Are the snubbed vertices connected through edges that join two snubbed vertices?
 *  Required so the chirality below has a single coherent twist (one connected patch,
 *  hence no arbitrary per-component mirror choice). */
function verticesConnected(dcel: DCEL, snubbed: Set<number>): boolean {
  if (snubbed.size <= 1) return true;
  const start = snubbed.values().next().value as number;
  const seen = new Set<number>([start]);
  const queue = [start];
  while (queue.length) {
    const v = dcel.vertices[queue.shift()!];
    for (const h of outgoingHalfEdges(v)) {
      const n = h.next.origin.id;
      if (snubbed.has(n) && !seen.has(n)) {
        seen.add(n);
        queue.push(n);
      }
    }
  }
  return seen.size === snubbed.size;
}

/**
 * Proper 2-coloring of the faces (adjacent faces get opposite colors), restricted to
 * the faces touched by the snubbed region and seeded at the dragged vertex. Only
 * edges incident to a snubbed vertex carry the "must differ" constraint — those are
 * the only edges whose two cut vertices must split into an outer/inner pair — so the
 * coloring is decided entirely WITHIN the (connected) selection and twists coherently
 * no matter what parity the rest of the solid has. `coherent` is false exactly when
 * the region has an odd cycle (no 2-coloring exists), in which case the caller refuses
 * the operation — there is no consistent chirality to choose.
 */
function twoColorFacesInRegion(
  dcel: DCEL,
  snubbed: Set<number>,
  seedFaceId: number,
): { color: Map<number, 0 | 1>; coherent: boolean } {
  const color = new Map<number, 0 | 1>();
  let coherent = true;
  const constrains = (h: HalfEdge): boolean =>
    snubbed.has(h.origin.id) || snubbed.has(h.next.origin.id);
  color.set(seedFaceId, 0);
  const queue = [dcel.faces[seedFaceId]];
  while (queue.length) {
    const f = queue.shift()!;
    const c = color.get(f.id)!;
    let h = f.halfedge;
    const start = h;
    do {
      if (h.twin && constrains(h)) {
        const g = h.twin.face;
        const gc = color.get(g.id);
        if (gc === undefined) {
          color.set(g.id, (c ^ 1) as 0 | 1);
          queue.push(g);
        } else if (gc === c) {
          coherent = false;
        }
      }
      h = h.next;
    } while (h !== start);
  }
  return { color, coherent };
}

/** Outward unit normals of the two faces sharing half-edge `h` (centroid-oriented,
 *  matching truncate's edge-in-view test). */
function edgeFaceNormals(h: HalfEdge): Vector3[] {
  const faces = h.twin ? [h.face, h.twin.face] : [h.face];
  return faces.map((f) => {
    const n = faceNormalHE(f);
    if (n.dot(faceCentroidHE(f)) < 0) n.negate();
    return n;
  });
}

/**
 * Snub ↔ (welded max), driven by holding Shift WHILE dragging a degree-2n vertex
 * inward along an edge (i.e. mid-truncation). Same handle as truncate, but instead
 * of cutting symmetrically it skews: the dragged edge's cut vertex slides OUTWARD
 * (so it ends up belonging only to a triangle) while the alternating "inner" cut
 * vertices hang back to form the central n-gon.
 *
 * Model: like Truncate, BY DEFAULT every vertex participates (the dragged vertex is
 * the handle); a multi-select restricts the set, which must be edge-connected (so the
 * chirality has one coherent twist with no arbitrary per-component choice). Each
 * participating vertex must have EVEN degree — otherwise the surrounding cut vertices
 * can't be split into a clean alternation, so we throw (the controller turns that
 * into a no-op).
 *
 * Truncating a degree-2n vertex exposes a 2n-gon of cut vertices. Snub splits that
 * ring into a central n-gon (the "inner" alternating cut vertices) surrounded by n
 * "ear" triangles (around the "outer" cut vertices) — or, when n=2, just the two
 * triangles sharing the diagonal (no central 2-gon). Dragging skews it: outer cut
 * vertices slide outward along their edges, inner ones barely move, puckering the
 * flat truncation into the chiral snub form.
 *
 * `baseT` is the frozen truncation level at the moment Shift was pressed: at skew=0
 * the outer and inner cut vertices both sit at the symmetric cut fraction f0 =
 * baseT·0.5 (i.e. the current truncation), so pressing Shift changes nothing until
 * the mouse moves. Increasing the skew then slides the outer cut vertices out toward
 * F_OUT and the inner ones in toward F_IN. Whether the result welds (partial snub vs
 * snub) is decided by the caller via the `weld` flag at commit — inherited from
 * whether the base truncation reached a full rectify.
 *
 * Chirality is LIVE: the two mirror forms are precomputed (the two face 2-colorings);
 * `snap` picks whichever makes the edge nearest the cursor "outer", so aiming at an
 * adjacent edge flips the whole solid's twist. A coherent face 2-coloring always
 * exists (every vertex is even), keeping the two forms globally consistent.
 */
export function buildSnub(
  poly: Polyhedron,
  draggedVid: number,
  selected: Set<number> | null,
  inView: InViewTest | null = null,
  baseT = 1,
): MorphPlan {
  const dcel = poly.dcel;
  const f0 = baseT * 0.5; // symmetric cut fraction the skew starts from (the truncation)

  const snubbed = new Set<number>(
    selected && selected.size > 0 ? selected : dcel.vertices.map((v) => v.id),
  );
  snubbed.add(draggedVid); // the handle always participates

  // Preconditions (mirrored by `canSnub` for the UI): the selection must be edge-
  // connected, and its incident-face region must be 2-colorable. A 2-coloring fails
  // exactly when that region has an ODD CYCLE — which also covers any odd-degree
  // vertex (its ring of faces IS an odd cycle), so no separate parity check is needed.
  // Without a coherent coloring there's no consistent chirality, so we refuse outright.
  if (!verticesConnected(dcel, snubbed)) {
    throw new Error("Snub needs the selected vertices to be connected.");
  }
  // Chirality is decided within the selected patch, seeded at the dragged vertex.
  const { color, coherent } = twoColorFacesInRegion(
    dcel,
    snubbed,
    dcel.vertices[draggedVid].halfedge.face.id,
  );
  if (!coherent) {
    throw new Error("Snub needs a 2-colorable selection (the patch has an odd cycle).");
  }

  // ---- Index new vertices (one cut vertex per snubbed-origin half-edge) -----------
  const cutIndex = new Map<number, number>(); // halfedge id -> new vertex index
  const keepIndex = new Map<number, number>(); // old vertex id -> new vertex index
  let idx = 0;
  for (const he of dcel.halfedges) {
    if (snubbed.has(he.origin.id)) cutIndex.set(he.id, idx++);
  }
  for (const v of dcel.vertices) {
    if (!snubbed.has(v.id)) keepIndex.set(v.id, idx++);
  }
  const vertexCount = idx;

  const cutEnds: Array<{ index: number; origin: Vector3; dest: Vector3; heId: number }> = [];
  for (const he of dcel.halfedges) {
    const i = cutIndex.get(he.id);
    if (i === undefined) continue;
    cutEnds.push({
      index: i,
      origin: he.origin.position,
      dest: he.next.origin.position,
      heId: he.id,
    });
  }
  const keepEnds: Array<{ index: number; pos: Vector3 }> = [];
  for (const v of dcel.vertices) {
    const i = keepIndex.get(v.id);
    if (i !== undefined) keepEnds.push({ index: i, pos: v.position });
  }

  // ---- Weld pairs for the max end (variant-independent) ---------------------------
  // On every edge with both ends snubbed, one cut vertex is outer and the other
  // inner (the coloring guarantees opposite roles); with F_OUT + F_IN = 1 they
  // coincide at t=1, and welding them collapses the edge — "the long vertices join
  // with the short vertices" (e.g. snub octahedron → icosahedron).
  const weldPairs: Array<[number, number]> = [];
  for (const he of dcel.halfedges) {
    if (!he.twin || he.id >= he.twin.id) continue; // once per undirected edge
    const a = cutIndex.get(he.id);
    const b = cutIndex.get(he.twin.id);
    if (a !== undefined && b !== undefined) weldPairs.push([a, b]);
  }

  // ---- The two chiral variants ----------------------------------------------------
  // Variant v: a cut vertex is "outer" iff its half-edge's face has color v. Around
  // any snubbed vertex the incident faces alternate color (the coloring is coherent —
  // guaranteed above), so the outer cut vertices alternate too; flipping v gives the
  // mirror twist.
  function outerSet(v: 0 | 1): Set<number> {
    const outer = new Set<number>();
    for (const id of snubbed) {
      const H = outgoingHalfEdges(dcel.vertices[id]);
      const phase = color.get(H[0].face.id) === v ? 0 : 1;
      for (let k = 0; k < H.length; k++) {
        if (k % 2 === phase) outer.add(H[k].id);
      }
    }
    return outer;
  }

  function buildFaces(outerHe: Set<number>): number[][] {
    const faces: number[][] = [];

    // (a) one polygon per original face — identical to truncate's truncated face.
    for (const f of dcel.faces) {
      const loop: number[] = [];
      let h = f.halfedge;
      const start = h;
      do {
        const v = h.origin;
        if (snubbed.has(v.id)) {
          loop.push(cutIndex.get(h.prev.twin!.id)!); // incoming cut
          loop.push(cutIndex.get(h.id)!); // outgoing cut
        } else {
          loop.push(keepIndex.get(v.id)!);
        }
        h = h.next;
      } while (h !== start);
      faces.push(loop);
    }

    // (b) per snubbed vertex: central n-gon (inner cut verts) + n ear triangles.
    for (const v of dcel.vertices) {
      if (!snubbed.has(v.id)) continue;
      const H = outgoingHalfEdges(v);
      const m = H.length; // = 2n
      const c = H.map((h) => cutIndex.get(h.id)!);

      const ngon: number[] = [];
      for (let k = 0; k < m; k++) {
        if (!outerHe.has(H[k].id)) ngon.push(c[k]); // inner ones
      }
      if (ngon.length >= 3) faces.push(ngon); // skip the degenerate 2-gon (n=2)

      for (let k = 0; k < m; k++) {
        if (!outerHe.has(H[k].id)) continue; // ear around each outer cut vert
        faces.push([c[(k - 1 + m) % m], c[k], c[(k + 1) % m]]);
      }
    }
    return faces;
  }

  const variants = ([0, 1] as const).map((v) => {
    const outerHe = outerSet(v);
    return { outerHe, previewFaces: buildFaces(outerHe) };
  });
  let currentVariant = 0;

  function positions(skew: number): Vector3[] {
    const out: Vector3[] = new Array(vertexCount);
    const outer = variants[currentVariant].outerHe;
    // skew interpolates from the symmetric truncation (both at f0) toward the snub.
    const fOut = f0 + (F_OUT - f0) * skew;
    const fIn = f0 + (F_IN - f0) * skew;
    for (const c of cutEnds) {
      out[c.index] = c.origin.clone().lerp(c.dest, outer.has(c.heId) ? fOut : fIn);
    }
    for (const k of keepEnds) out[k.index] = k.pos.clone();
    return out;
  }

  // ---- Snapping: drag along the closest incident edge; it becomes "outer" ---------
  const draggedV = dcel.vertices[draggedVid];

  function snap(ray: Ray): {
    t: number;
    point: Vector3;
    highlight?: { a: Vector3; b: Vector3 };
  } {
    let best:
      | { heId: number; frac: number; point: Vector3; max: Vector3; dist: number }
      | null = null;
    for (const h of outgoingHalfEdges(draggedV)) {
      const from = h.origin.position;
      const edge = h.next.origin.position.clone().sub(from);
      const mid = from.clone().add(edge.clone().multiplyScalar(0.5));
      if (inView && !inView(mid, edgeFaceNormals(h))) continue;
      let frac = closestLineParam(from, edge, ray.origin, ray.direction);
      // The outer vertex can't retreat past the frozen truncation (f0); it slides
      // out toward F_OUT. The mouse begins at f0 (the truncate handle), so skew ~0.
      frac = Math.max(f0, Math.min(F_OUT, frac));
      const point = from.clone().add(edge.clone().multiplyScalar(frac));
      const dist = distancePointToRay(point, ray);
      if (!best || dist < best.dist) {
        best = {
          heId: h.id,
          frac,
          point,
          max: from.clone().add(edge.clone().multiplyScalar(F_OUT)),
          dist,
        };
      }
    }
    if (!best) {
      const p = draggedV.position.clone();
      return { t: 0, point: p, highlight: { a: p, b: p.clone() } };
    }
    // Pick the chiral form in which the dragged edge is "outer" (only in a triangle).
    currentVariant = variants[0].outerHe.has(best.heId) ? 0 : 1;
    const span = F_OUT - f0;
    const t = span > 1e-9 ? Math.max(0, Math.min(1, (best.frac - f0) / span)) : 0;
    return { t, point: best.point, highlight: { a: best.point.clone(), b: best.max } };
  }

  function commit(t: number, weld: boolean): Mesh {
    const mesh: Mesh = {
      vertices: positions(t),
      faces: variants[currentVariant].previewFaces.map((f) => f.slice()),
    };
    return weld ? weldVertexPairs(mesh, weldPairs) : mesh;
  }

  return {
    kind: "snub",
    get previewFaces() {
      return variants[currentVariant].previewFaces;
    },
    positions,
    snap,
    commit,
  };
}

/**
 * Whether snub can be applied COHERENTLY to `selected` vertices (or all when null).
 * Mirrors buildSnub's preconditions exactly: the participating vertices must be edge-
 * connected and their incident-face region must be 2-colorable (no odd cycle — which
 * also rules out odd-degree vertices). Cheap; intended for the UI availability hint.
 */
export function canSnub(poly: Polyhedron, selected: Set<number> | null): boolean {
  const dcel = poly.dcel;
  const snubbed = new Set<number>(
    selected && selected.size > 0 ? selected : dcel.vertices.map((v) => v.id),
  );
  if (snubbed.size === 0 || !verticesConnected(dcel, snubbed)) return false;
  const seed = snubbed.values().next().value as number;
  return twoColorFacesInRegion(dcel, snubbed, dcel.vertices[seed].halfedge.face.id)
    .coherent;
}

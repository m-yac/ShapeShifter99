import { Vector3, Ray, Color } from "three";
import {
  type Mesh,
  type HEFace,
  outgoingHalfEdges,
} from "../geometry/HalfEdge";
import {
  type Polyhedron,
  faceCentroidHE,
  faceNormalHE,
} from "../geometry/polyhedron";
import { type ColorSet, edgeKey } from "../geometry/colors";
import { type MorphPlan } from "./types";
import { type InViewTest } from "./truncate";
import { lerpFaceColors } from "./colorUtil";
import { closestLineParam } from "../util/lines";

/** 1 + the maximum geometric color anywhere in the set (the fresh-color base). */
function freshBase(old: ColorSet): number {
  let m = 0;
  for (const c of old.vertex) m = Math.max(m, c);
  for (const c of old.face) m = Math.max(m, c);
  for (const c of old.edge.values()) m = Math.max(m, c);
  return m + 1;
}

/** Outward unit normal of a face (centroid-oriented, like SceneView). */
function outwardNormal(f: HEFace): Vector3 {
  const n = faceNormalHE(f);
  if (n.dot(faceCentroidHE(f)) < 0) n.negate();
  return n;
}

/**
 * Subdivide, driven by dragging an edge midpoint outward along the edge normal
 * (the mean of its two bordering face normals). Like truncate/kis the gesture is
 * global: dragging ONE edge subdivides EVERY edge. A new vertex is placed in the
 * middle of each edge (raised along the normal), every original vertex is kept (as
 * the apex of its vertex figure), each original face becomes the polygon of its
 * edge midpoints, and each original vertex grows a fan of triangles to its
 * surrounding midpoints.
 *
 *   t = 0 → coplanar with the original (midpoints flat, looks unchanged).
 *   0 < t < 1 → the subdivided solid (e.g. cube → 6 quads + 24 triangles).
 *   t = 1 → the raised faces become coplanar with each original face again and
 *           weld back into it (the original solid).
 *
 * @param poly current polyhedron
 * @param edge the dragged undirected edge (vertex-id pair), for the snap axis
 */
export function buildSubdivide(
  poly: Polyhedron,
  edge: [number, number],
  _inView: InViewTest | null = null,
): MorphPlan {
  const dcel = poly.dcel;
  const old = poly.colors;

  // ---- Index a vertex at every edge midpoint, plus an apex at every vertex. ----
  const edgeIndex = new Map<string, number>(); // edgeKey -> midpoint vertex index
  const midData: Array<{ index: number; mid: Vector3; normal: Vector3; key: string }> = [];
  let idx = 0;
  for (const he of dcel.halfedges) {
    if (!he.twin || he.id >= he.twin.id) continue; // once per undirected edge
    const p = he.origin.position;
    const q = he.next.origin.position;
    const key = edgeKey(he.origin.id, he.next.origin.id);
    const normal = outwardNormal(he.face).add(outwardNormal(he.twin.face)).normalize();
    edgeIndex.set(key, idx);
    midData.push({ index: idx, mid: p.clone().add(q).multiplyScalar(0.5), normal, key });
    idx++;
  }
  const E = idx;
  const apexOf = (vid: number) => E + vid;
  const vertexCount = E + dcel.vertices.length;

  // The raised midpoint reaches roughly the level of the surrounding vertices at
  // the weld; a fraction of the average incident edge length is a good, non-
  // degenerate cap height (the post-release solver refines the exact form).
  let avgLen = 0;
  for (const m of midData) {
    // re-derive endpoints from the key for the length estimate
    const [a, b] = m.key.split("_").map(Number);
    avgLen += dcel.vertices[a].position.distanceTo(dcel.vertices[b].position);
  }
  avgLen /= Math.max(1, midData.length);
  const hMax = 0.5 * avgLen;

  function positions(t: number): Vector3[] {
    const out: Vector3[] = new Array(vertexCount);
    for (const m of midData) out[m.index] = m.mid.clone().addScaledVector(m.normal, t * hMax);
    for (const v of dcel.vertices) out[apexOf(v.id)] = v.position.clone();
    return out;
  }

  // ---- Faces: per-face midpoint polygon + per-vertex triangle fan. ------------
  const base = freshBase(old);
  const previewFaces: number[][] = [];
  const faceColor: number[] = [];
  const faceStart: number[] = [];

  const midOf = (vid: number, wid: number) => edgeIndex.get(edgeKey(vid, wid))!;

  // (a) one polygon per original face, through its edge midpoints.
  for (const f of dcel.faces) {
    const loop: number[] = [];
    let h = f.halfedge;
    const start = h;
    do {
      loop.push(midOf(h.origin.id, h.next.origin.id));
      h = h.next;
    } while (h !== start);
    previewFaces.push(loop);
    faceColor.push(old.face[f.id]); // central polygon keeps the original face color
    faceStart.push(old.face[f.id]);
  }
  // (b) one triangle per (vertex, consecutive incident-edge pair): the corner fan.
  const cornerTriangles: number[][] = [];
  for (const v of dcel.vertices) {
    const ring = outgoingHalfEdges(v);
    for (let i = 0; i < ring.length; i++) {
      const mi = midOf(v.id, ring[i].next.origin.id);
      const mj = midOf(v.id, ring[(i + 1) % ring.length].next.origin.id);
      const tri = [mi, apexOf(v.id), mj];
      previewFaces.push(tri);
      cornerTriangles.push(tri);
      faceColor.push(base); // fresh color for the new corner triangles
      faceStart.push(old.vertex[v.id]); // emerge from the vertex color
    }
  }

  // ---- Vertex + edge colors --------------------------------------------------
  const vertexColor: number[] = new Array(vertexCount);
  for (const m of midData) vertexColor[m.index] = base;
  for (const v of dcel.vertices) vertexColor[apexOf(v.id)] = old.vertex[v.id];

  const edgeColor = new Map<string, number>();
  for (const f of dcel.faces) {
    let h = f.halfedge;
    const start = h;
    do {
      const m1 = midOf(h.origin.id, h.next.origin.id);
      const m2 = midOf(h.next.origin.id, h.next.next.origin.id);
      edgeColor.set(edgeKey(m1, m2), base + 1); // central-polygon edges
      h = h.next;
    } while (h !== start);
  }
  for (const tri of cornerTriangles) {
    edgeColor.set(edgeKey(tri[0], tri[1]), base + 1);
    edgeColor.set(edgeKey(tri[1], tri[2]), base + 1);
  }

  function previewFaceColors(t: number): Color[] {
    return lerpFaceColors(faceStart, faceColor, t);
  }

  // ---- Snap: project the cursor onto the edge-normal line from its midpoint. ---
  const eMid = dcel.vertices[edge[0]].position
    .clone()
    .add(dcel.vertices[edge[1]].position)
    .multiplyScalar(0.5);
  const eNormal =
    midData.find((m) => m.key === edgeKey(edge[0], edge[1]))?.normal.clone() ??
    new Vector3(0, 1, 0);
  function snap(ray: Ray): { t: number; point: Vector3; highlight?: { a: Vector3; b: Vector3 } } {
    let s = closestLineParam(eMid, eNormal, ray.origin, ray.direction);
    s = Math.max(0, Math.min(hMax, s));
    const point = eMid.clone().addScaledVector(eNormal, s);
    const t = hMax > 1e-9 ? s / hMax : 0;
    return {
      t,
      point,
      highlight: { a: point.clone(), b: eMid.clone().addScaledVector(eNormal, hMax) },
    };
  }

  function commit(t: number, weld: boolean): { mesh: Mesh; colors: ColorSet } {
    if (weld) {
      // At the coplanar height every original face's pieces weld back into it, so
      // the welded form is the original solid.
      return {
        mesh: {
          vertices: poly.mesh.vertices.map((v) => v.clone()),
          faces: poly.mesh.faces.map((f) => f.slice()),
        },
        colors: {
          vertex: old.vertex.slice(),
          face: old.face.slice(),
          edge: new Map(old.edge),
        },
      };
    }
    return {
      mesh: { vertices: positions(t), faces: previewFaces.map((f) => f.slice()) },
      colors: {
        vertex: vertexColor.slice(),
        face: faceColor.slice(),
        edge: new Map(edgeColor),
      },
    };
  }

  return {
    kind: "subdivide",
    previewFaces,
    positions,
    previewFaceColors,
    previewEdgeColors: edgeColor,
    vanishingEdges: [],
    snap,
    commit,
  };
}

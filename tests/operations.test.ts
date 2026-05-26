import { describe, it, expect } from "vitest";
import { getSeed } from "../src/geometry/seeds";
import { Polyhedron, faceCentroidHE } from "../src/geometry/polyhedron";
import { buildDCEL, counts, outgoingHalfEdges, type HalfEdge } from "../src/geometry/HalfEdge";
import { buildTruncate } from "../src/operations/truncate";
import { buildKis } from "../src/operations/kis";
import { buildSnub, canSnub } from "../src/operations/snub";
import { buildGyro, canGyro } from "../src/operations/gyro";
import { computeSignature } from "../src/identify/configurations";
import { weldVertexPairs } from "../src/operations/weld";
import { Vector3, Ray } from "three";

const cube = () => new Polyhedron(getSeed("cube"));
const octahedron = () => new Polyhedron(getSeed("octahedron"));

/** Multiset of face orders (sides per face) in a mesh. */
const faceOrderCounts = (faces: number[][]): Record<number, number> => {
  const m: Record<number, number> = {};
  for (const f of faces) m[f.length] = (m[f.length] ?? 0) + 1;
  return m;
};

describe("truncate / rectify", () => {
  it("intermediate truncation of the cube is the truncated cube (3.8.8)", () => {
    const mesh = buildTruncate(cube(), 0, null).commit(0.5, false);
    const sig = computeSignature(buildDCEL(mesh));
    expect(sig).toMatchObject({ V: 24, E: 36, F: 14 });
    expect(sig.vertexConfigs).toEqual({ "3.8.8": 24 });
    expect(sig.faceConfigs).toEqual({ "3.3.3": 8, "3.3.3.3.3.3.3.3": 6 });
  });

  it("welded max (rectify) of the cube is the cuboctahedron (3.4.3.4)", () => {
    const mesh = buildTruncate(cube(), 0, null).commit(1, true);
    const sig = computeSignature(buildDCEL(mesh));
    expect(sig).toMatchObject({ V: 12, E: 24, F: 14 });
    expect(sig.vertexConfigs).toEqual({ "3.4.3.4": 12 });
    expect(sig.faceConfigs).toEqual({ "4.4.4": 8, "4.4.4.4": 6 });
  });
});

describe("kis / join", () => {
  it("intermediate kis of the cube is the tetrakis hexahedron", () => {
    const mesh = buildKis(cube(), 0, null).commit(0.5, false);
    const c = counts(buildDCEL(mesh));
    // 8 original + 6 apex vertices; every face a triangle (6 faces × 4 tris)
    expect(c).toEqual({ V: 14, E: 36, F: 24 });
  });

  it("welded max (join) of the cube is the rhombic dodecahedron (3.4.3.4 faces)", () => {
    const mesh = buildKis(cube(), 0, null).commit(1, true);
    const sig = computeSignature(buildDCEL(mesh));
    expect(sig).toMatchObject({ V: 14, E: 24, F: 12 });
    expect(sig.vertexConfigs).toEqual({ "4.4.4": 8, "4.4.4.4": 6 });
    expect(sig.faceConfigs).toEqual({ "3.4.3.4": 12 });
  });
});

describe("snub", () => {
  it("intermediate snub of the octahedron (all degree-4 vertices, n=2)", () => {
    const mesh = buildSnub(octahedron(), 0, null).commit(0.5, false);
    const c = counts(buildDCEL(mesh));
    // 24 cut verts (one per half-edge); 8 hexagons (truncated tris) + 12 ear tris.
    expect(c).toEqual({ V: 24, E: 42, F: 20 });
    expect(faceOrderCounts(mesh.faces)).toEqual({ 3: 12, 6: 8 });
  });

  it("welded max (snub) of the octahedron is the icosahedron (3^5)", () => {
    const mesh = buildSnub(octahedron(), 0, null).commit(1, true);
    const sig = computeSignature(buildDCEL(mesh));
    expect(sig).toMatchObject({ V: 12, E: 30, F: 20 });
    expect(sig.vertexConfigs).toEqual({ "3.3.3.3.3": 12 });
    expect(sig.faceConfigs).toEqual({ "5.5.5": 20 });
  });

  it("throws (no-op) on a solid with odd-degree vertices", () => {
    // cube vertices are degree 3 → snub cannot alternate the cut vertices.
    expect(() => buildSnub(cube(), 0, null)).toThrow();
  });

  it("rejects a disconnected vertex selection (needs one connected patch)", () => {
    const poly = octahedron(); // all degree-4, so parity passes; isolate connectivity
    const adj = new Set(outgoingHalfEdges(poly.dcel.vertices[0]).map((h) => h.next.origin.id));
    const far = poly.dcel.vertices.find((v) => v.id !== 0 && !adj.has(v.id))!.id;
    expect(() => buildSnub(poly, 0, new Set([0, far]))).toThrow();
  });

  it("accepts a connected partial vertex selection", () => {
    const poly = octahedron();
    const near = outgoingHalfEdges(poly.dcel.vertices[0])[0].next.origin.id;
    const mesh = buildSnub(poly, 0, new Set([0, near])).commit(0.5, false);
    const c = counts(buildDCEL(mesh));
    expect(c.V - c.E + c.F).toBe(2); // valid closed manifold
  });

  it("chirality follows the dragged edge: adjacent edges give mirror forms", () => {
    const poly = octahedron();
    const plan = buildSnub(poly, 0, null);
    const H = outgoingHalfEdges(poly.dcel.vertices[0]);
    // A pick ray passing through a point 30% along edge `h` → that edge wins the snap.
    const rayAlong = (h: HalfEdge): Ray => {
      const p = h.origin.position.clone().lerp(h.next.origin.position, 0.3);
      const dir = p.clone().normalize(); // radial: not parallel to the edge
      return new Ray(p.clone().addScaledVector(dir, 5), dir.negate());
    };
    plan.snap(rayAlong(H[0]));
    const formA = JSON.stringify(plan.previewFaces);
    plan.snap(rayAlong(H[1])); // the adjacent edge → opposite parity → mirror twist
    const formB = JSON.stringify(plan.previewFaces);
    expect(formA).not.toEqual(formB);
  });
});

describe("gyro", () => {
  it("intermediate gyro of the cube (all 4-gon faces, n=2)", () => {
    const mesh = buildGyro(cube(), 0, null).commit(0.5, false);
    const c = counts(buildDCEL(mesh));
    // 8 originals + 2 peripherals per face (no apex at n=2); 2 quads + 2 tris / face.
    expect(c).toEqual({ V: 20, E: 42, F: 24 });
    expect(faceOrderCounts(mesh.faces)).toEqual({ 3: 12, 4: 12 });
  });

  it("closes on hexagonal (n=3) faces: gyro of the truncated octahedron", () => {
    // Truncated octahedron: 8 hexagons + 6 squares, all even-sided. Exercises the
    // n>=3 pentagon+triangle tiling (the n=2 cube test only hits the collapsed path).
    const truncOcta = buildTruncate(octahedron(), 0, null).commit(0.5, false);
    const mesh = buildGyro(new Polyhedron(truncOcta), 0, null).commit(0.5, false);
    const c = counts(buildDCEL(mesh)); // valid Euler counts ⇒ a closed manifold
    expect(c).toEqual({ V: 68, E: 138, F: 72 });
    // 24 pentagons (3 per hexagon) + 12 quads (2 per square) + 36 triangles.
    expect(faceOrderCounts(mesh.faces)).toEqual({ 3: 36, 4: 12, 5: 24 });
  });

  it("welded max (gyro) of the cube is the dodecahedron (5^3)", () => {
    const mesh = buildGyro(cube(), 0, null).commit(1, true);
    const sig = computeSignature(buildDCEL(mesh));
    expect(sig).toMatchObject({ V: 20, E: 30, F: 12 });
    expect(sig.vertexConfigs).toEqual({ "5.5.5": 20 });
    expect(sig.faceConfigs).toEqual({ "3.3.3.3.3": 12 });
  });

  it("throws (no-op) on a solid with odd-sided faces", () => {
    // octahedron faces are triangles → gyro cannot alternate the boundary.
    expect(() => buildGyro(octahedron(), 0, null)).toThrow();
  });

  it("rejects a disconnected face selection (needs one connected patch)", () => {
    const poly = cube(); // all 4-gons, so parity passes; isolate connectivity
    const f0 = poly.dcel.faces[0];
    const adj = new Set<number>();
    let h = f0.halfedge;
    do { adj.add(h.twin!.face.id); h = h.next; } while (h !== f0.halfedge);
    const far = poly.dcel.faces.find((f) => f.id !== 0 && !adj.has(f.id))!.id;
    expect(() => buildGyro(poly, 0, new Set([0, far]))).toThrow();
  });

  it("accepts a connected partial face selection", () => {
    const poly = cube();
    const near = poly.dcel.faces[0].halfedge.twin!.face.id;
    const mesh = buildGyro(poly, 0, new Set([0, near])).commit(0.5, false);
    const c = counts(buildDCEL(mesh));
    expect(c.V - c.E + c.F).toBe(2); // valid closed manifold
  });

  it("at skew=0 the cap is degenerate (reproduces the kis apex)", () => {
    // With Shift just pressed (skew=0) every peripheral vertex sits at the centre, so
    // the gyro cap collapses back onto the single kis apex — pressing Shift without
    // moving the mouse changes nothing. Here that shows up as the n peripheral verts
    // of a face all coinciding.
    const verts = buildGyro(cube(), 0, null).positions(0);
    // the 12 peripheral verts are indices 8..19; each face's pair must coincide.
    expect(verts[8].distanceTo(verts[9])).toBeLessThan(1e-9);
  });

  it("chirality follows the dragged edge: adjacent edges give mirror forms", () => {
    // Gyro of the truncated octahedron (n>=3, has real centre vertices). Snapping a
    // ray onto the lines toward two adjacent boundary edges picks opposite twists.
    const base = buildTruncate(octahedron(), 0, null).commit(0.5, false);
    const poly = new Polyhedron(base);
    const plan = buildGyro(poly, 0, null);
    const f = poly.dcel.faces[0];
    const hes: HalfEdge[] = [];
    let h = f.halfedge;
    do { hes.push(h); h = h.next; } while (h !== f.halfedge);
    const apex = faceCentroidHE(f); // skew=0 centre ~ centroid (baseT default lifts it)
    const rayToEdge = (he: HalfEdge): Ray => {
      const mid = he.origin.position.clone().lerp(he.next.origin.position, 0.5);
      const p = apex.clone().lerp(mid, 0.3);
      const dir = p.clone().normalize();
      return new Ray(p.clone().addScaledVector(dir, 5), dir.negate());
    };
    plan.snap(rayToEdge(hes[0]));
    const formA = JSON.stringify(plan.previewFaces);
    plan.snap(rayToEdge(hes[1]));
    const formB = JSON.stringify(plan.previewFaces);
    expect(formA).not.toEqual(formB);
  });
});

describe("operation availability (canSnub / canGyro)", () => {
  it("reflects whole-solid parity when nothing is selected", () => {
    expect(canSnub(octahedron(), null)).toBe(true); // all degree-4
    expect(canSnub(cube(), null)).toBe(false); // all degree-3 → odd cycle
    expect(canGyro(cube(), null)).toBe(true); // all 4-gons
    expect(canGyro(octahedron(), null)).toBe(false); // triangles → odd cycle
  });

  it("is false for a disconnected selection, true for a connected partial one", () => {
    const poly = octahedron();
    const adj = new Set(outgoingHalfEdges(poly.dcel.vertices[0]).map((h) => h.next.origin.id));
    const far = poly.dcel.vertices.find((v) => v.id !== 0 && !adj.has(v.id))!.id;
    const near = outgoingHalfEdges(poly.dcel.vertices[0])[0].next.origin.id;
    expect(canSnub(poly, new Set([0, far]))).toBe(false);
    expect(canSnub(poly, new Set([0, near]))).toBe(true);
  });

  it("is false for a connected, all-even patch that still has an odd cycle", () => {
    // The 4 hexagons of the truncated tetrahedron: connected and all even-sided, but
    // the triangles they share pin the vertex coloring into an odd cycle, so there's
    // no coherent chirality → gyro is refused (and buildGyro throws).
    const tt = buildTruncate(new Polyhedron(getSeed("tetrahedron")), 0, null).commit(0.5, false);
    const poly = new Polyhedron(tt);
    const hexes = new Set<number>();
    poly.faces.forEach((f, i) => {
      if (f.length === 6) hexes.add(i);
    });
    expect(hexes.size).toBe(4);
    expect(canGyro(poly, hexes)).toBe(false);
    expect(() => buildGyro(poly, [...hexes][0], hexes)).toThrow();
  });
});

describe("weldVertexPairs", () => {
  it("merges a pair and drops the collapsed edge", () => {
    // two triangles sharing an edge (a square split by a diagonal)
    const mesh = {
      vertices: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [
        [0, 1, 2],
        [0, 2, 3],
      ],
    };
    const welded = weldVertexPairs(mesh, [[1, 3]]);
    expect(welded.vertices.length).toBe(3);
    // each triangle loses the welded duplicate and stays a triangle
    expect(welded.faces.every((f) => f.length === 3)).toBe(true);
  });
});

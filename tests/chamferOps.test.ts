import { describe, it, expect } from "vitest";
import { getSeed } from "../src/geometry/seeds";
import { Polyhedron } from "../src/geometry/polyhedron";
import { buildDCEL } from "../src/geometry/HalfEdge";
import { buildChamfer } from "../src/operations/chamfer";
import { buildSubdivide } from "../src/operations/subdivide";
import { computeSignature, signaturesEqual, type Signature } from "../src/identify/configurations";
import { NAMED } from "../src/data/namedPolyhedra";

const cube = () => new Polyhedron(getSeed("cube"));
const octahedron = () => new Polyhedron(getSeed("octahedron"));

/** The signature of a named-database entry, by name. */
const namedSig = (name: string): Signature =>
  computeSignature(NAMED.find((n) => n.name === name)!.poly.dcel);

/** A representative undirected edge + one bordering face of a polyhedron. */
const anyEdge = (p: Polyhedron): { edge: [number, number]; face: number } => {
  const he = p.dcel.halfedges[0];
  return { edge: [he.origin.id, he.next.origin.id], face: he.face.id };
};

describe("chamfer", () => {
  it("intermediate chamfer of the cube matches the chamfered cube", () => {
    const { edge, face } = anyEdge(cube());
    const sig = computeSignature(
      buildDCEL(buildChamfer(cube(), edge, face).commit(0.5, false).mesh),
    );
    expect(sig).toMatchObject({ V: 32, E: 48, F: 18 });
    expect(sig.faceConfigs).toEqual({ "3.3.3.3.3.3": 12, "3.3.3.3": 6 });
    expect(signaturesEqual(sig, namedSig("Chamfered cube"))).toBe(true);
  });

  it("intermediate chamfer of the octahedron matches the chamfered octahedron", () => {
    const { edge, face } = anyEdge(octahedron());
    const sig = computeSignature(
      buildDCEL(buildChamfer(octahedron(), edge, face).commit(0.5, false).mesh),
    );
    expect(sig).toMatchObject({ V: 30, E: 48, F: 20 });
    expect(signaturesEqual(sig, namedSig("Chamfered octahedron"))).toBe(true);
  });

  it("welded chamfer of the cube is the rhombic dodecahedron (V=14 E=24 F=12)", () => {
    const { edge, face } = anyEdge(cube());
    const sig = computeSignature(
      buildDCEL(buildChamfer(cube(), edge, face).commit(1, true).mesh),
    );
    expect(sig).toMatchObject({ V: 14, E: 24, F: 12 });
    expect(sig.faceConfigs).toEqual({ "3.4.3.4": 12 });
  });
});

describe("subdivide", () => {
  it("intermediate subdivision of the cube matches the subdivided cube", () => {
    const { edge } = anyEdge(cube());
    const sig = computeSignature(
      buildDCEL(buildSubdivide(cube(), edge).commit(0.5, false).mesh),
    );
    expect(sig).toMatchObject({ V: 20, E: 48, F: 30 });
    expect(sig.faceConfigs).toEqual({ "6.6.6.6": 6, "3.6.6": 24 });
    expect(signaturesEqual(sig, namedSig("Subdivided cube"))).toBe(true);
  });

  it("intermediate subdivision of the octahedron matches the subdivided octahedron", () => {
    const { edge } = anyEdge(octahedron());
    const sig = computeSignature(
      buildDCEL(buildSubdivide(octahedron(), edge).commit(0.5, false).mesh),
    );
    expect(sig).toMatchObject({ V: 18, E: 48, F: 32 });
    expect(signaturesEqual(sig, namedSig("Subdivided octahedron"))).toBe(true);
  });

  it("welded subdivision returns the original solid", () => {
    const { edge } = anyEdge(cube());
    const sig = computeSignature(
      buildDCEL(buildSubdivide(cube(), edge).commit(1, true).mesh),
    );
    expect(sig).toMatchObject({ V: 8, E: 12, F: 6 });
  });
});

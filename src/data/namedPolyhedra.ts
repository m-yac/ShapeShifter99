import { type Mesh } from "../geometry/HalfEdge";
import { Polyhedron } from "../geometry/polyhedron";
import { seedColors, type ColorSet, type SchemeName } from "../geometry/colors";
import { getSeed } from "../geometry/seeds";
import { buildTruncate } from "../operations/truncate";
import { buildSnub } from "../operations/snub";
import { buildKis } from "../operations/kis";
import { buildGyro } from "../operations/gyro";
import { buildChamfer } from "../operations/chamfer";
import { buildSubdivide } from "../operations/subdivide";

/**
 * The named-polyhedron database — the SINGLE source of truth for both
 * identification (`identify/identify.ts`) and the LIBRARY browse diagram
 * (`ui/libraryBrowser.ts` via `libraryShapeFor`).
 *
 * Identification is purely combinatorial (vertex/face configurations + V,E,F), so
 * an entry only needs correct CONNECTIVITY. The LIBRARY, however, also renders each
 * solid in its *default colors*, and those depend on the construction PATH — so we
 * build every solid the way the GAME makes it: rooted at the tetrahedron (the only
 * starting seed), following the same operation tree the player would (e.g. the
 * game's cube is `join(tetrahedron)`, whose faces inherit the tetrahedron's edge
 * color — not a bare cube seed whose faces are color 0). Each entry also carries the
 * symmetry color SCHEME it displays in, mirroring the live app's auto-switch.
 *
 * ── To add your own ──────────────────────────────────────────────────────────
 *   Build it from an existing solid with the recipe helpers below (truncate /
 *   rectify / kis / join / snub / gyro, or the arity-selected truncateVerticesOfDegree
 *   / kisFacesOfSides), then add an `E(name, type, scheme, poly)` entry.
 * ─────────────────────────────────────────────────────────────────────────────
 */
/** The family a named solid belongs to (shown in the discovery popup). */
export type SolidType =
  | "Platonic solid"
  | "Archimedean solid"
  | "Catalan solid"
  | "Chamfered solid"
  | "Subdivided solid"
  | "Johnson solid"
  | "Dihedral solid";

export interface NamedPolyhedron {
  name: string;
  type: SolidType;
  /** A colored embedding built by the recipe. `poly.mesh` is its connectivity
   *  (all identification needs); `poly.colors` carries the geometric colors so
   *  the LIBRARY browse diagram can render each solid in its default colors. */
  poly: Polyhedron;
  /** The symmetry-appropriate color scheme (the one the live app auto-switches to
   *  for this solid's family), so the browse diagram colors each solid the way the
   *  live app does when you make it. */
  scheme: SchemeName;
}

// --- recipe helpers ---------------------------------------------------------
// Colors propagate through a chain of operations exactly as they do during live
// editing (a fresh seed starts with `seedColors`, and every operation layers on its
// c+n rule). Each helper takes and returns a *colored* Polyhedron.
const wrap = (r: { mesh: Mesh; colors: ColorSet }): Polyhedron =>
  new Polyhedron(r.mesh, r.colors);

/** Uniform truncation (intermediate topology). */
const truncate = (p: Polyhedron): Polyhedron =>
  wrap(buildTruncate(p, 0, null).commit(0.5, false));
/** Rectify / ambo (the welded "max" of the truncate drag). */
const rectify = (p: Polyhedron): Polyhedron =>
  wrap(buildTruncate(p, 0, null).commit(1, true));
/** Kis (intermediate topology). */
const kis = (p: Polyhedron): Polyhedron =>
  wrap(buildKis(p, 0, null).commit(0.5, false));
/** Join (the welded "max" of the kis drag). */
const join = (p: Polyhedron): Polyhedron =>
  wrap(buildKis(p, 0, null).commit(1, true));
/** Snub (the welded "max" of the snub drag). */
const snub = (p: Polyhedron): Polyhedron =>
  wrap(buildSnub(p, 0, null).commit(1, true));
/** Gyro (the welded "max" of the gyro drag). */
const gyro = (p: Polyhedron): Polyhedron =>
  wrap(buildGyro(p, 0, null).commit(1, true));

// --- chamfer / subdivide ----------------------------------------------------
// These are built with the actual interactive operations (the same `buildChamfer`
// / `buildSubdivide` the game runs on a dragged edge), rather than reconstructed
// from truncate/kis on a selected arity. That arity trick can't express the
// tetrahedron — its join (the cube) is vertex-uniform and its rectify (the
// octahedron) is face-uniform, so there's no sub-arity to target — whereas the
// real operation chamfers/subdivides EVERY edge and so handles it directly. Any
// edge works as the drag handle (the op is global); we just take the first.

/** Chamfer (intermediate topology) every edge of `p`. */
const chamfer = (p: Polyhedron): Polyhedron => {
  const he = p.dcel.halfedges[0];
  const edge: [number, number] = [he.origin.id, he.next.origin.id];
  return wrap(buildChamfer(p, edge, he.face.id).commit(0.5, false));
};

/** Subdivide (intermediate topology) every edge of `p`. */
const subdivide = (p: Polyhedron): Polyhedron => {
  const he = p.dcel.halfedges[0];
  const edge: [number, number] = [he.origin.id, he.next.origin.id];
  return wrap(buildSubdivide(p, edge).commit(0.5, false));
};

/** Finalize a colored solid into a named-database entry. */
const E = (
  name: string,
  type: SolidType,
  scheme: SchemeName,
  poly: Polyhedron,
): NamedPolyhedron => ({ name, type, poly, scheme });

const P: SolidType = "Platonic solid";
const A: SolidType = "Archimedean solid";
const C: SolidType = "Catalan solid";
const Ch: SolidType = "Chamfered solid";
const Sub: SolidType = "Subdivided solid";

const TE: SchemeName = "tetrahedral";
const OC: SchemeName = "octahedral";
const IC: SchemeName = "icosahedral";

// --- the construction tree, rooted at the tetrahedron -----------------------
// (Identical to what the game produces from the only starting seed.)
const tetMesh = getSeed("tetrahedron");
const tet = new Polyhedron(tetMesh, seedColors(tetMesh));

const oct = rectify(tet); //  rectify(tetra) = octahedron
const cube = join(tet); //    join(tetra)    = cube
const ico = snub(oct); //     snub(octa)     = icosahedron
const dod = gyro(cube); //    gyro(cube)     = dodecahedron

const cuboct = rectify(oct); //  rectify(octa)  = cuboctahedron
const rhDod = join(oct); //      join(octa)     = rhombic dodecahedron
const icosidod = rectify(ico); // rectify(icosa) = icosidodecahedron
const rhTri = join(ico); //      join(icosa)    = rhombic triacontahedron

export const NAMED: NamedPolyhedron[] = [
  // Platonic solids
  E("Tetrahedron", P, TE, tet),
  E("Octahedron", P, OC, oct),
  E("Cube", P, OC, cube),
  E("Icosahedron", P, IC, ico),
  E("Dodecahedron", P, IC, dod),

  // Archimedean solids — truncations
  E("Truncated tetrahedron", A, TE, truncate(tet)),
  E("Truncated octahedron", A, OC, truncate(oct)),
  E("Truncated cube", A, OC, truncate(cube)),
  E("Truncated icosahedron", A, IC, truncate(ico)),
  E("Truncated dodecahedron", A, IC, truncate(dod)),
  // Archimedean solids — rectifications & beyond
  E("Cuboctahedron", A, OC, cuboct),
  E("Icosidodecahedron", A, IC, icosidod),
  E("Truncated Cuboctahedron", A, OC, truncate(cuboct)),
  E("Truncated Icosidodecahedron", A, IC, truncate(icosidod)),
  E("Rhombicuboctahedron", A, OC, rectify(cuboct)),
  E("Rhombicosidodecahedron", A, IC, rectify(icosidod)),
  E("Snub cuboctahedron", A, OC, snub(cuboct)),
  E("Snub Icosidodecahedron", A, IC, snub(icosidod)),

  // Catalan solids — kis
  E("Triakis tetrahedron", C, TE, kis(tet)),
  E("Triakis octahedron", C, OC, kis(oct)),
  E("Tetrakis hexahedron", C, OC, kis(cube)),
  E("Triakis icosahedron", C, IC, kis(ico)),
  E("Pentakis dodecahedron", C, IC, kis(dod)),
  // Catalan solids — joins & beyond
  E("Rhombic dodecahedron", C, OC, rhDod),
  E("Rhombic triacontahedron", C, IC, rhTri),
  E("Disdyakis dodecahedron", C, OC, kis(rhDod)),
  E("Disdyakis triacontahedron", C, IC, kis(rhTri)),
  E("Deltoidal icositetrahedron", C, OC, join(cuboct)),
  E("Deltoidal hexecontahedron", C, IC, join(icosidod)),
  E("Pentagonal icositetrahedron", C, OC, gyro(rhDod)),
  E("Pentagonal hexecontahedron", C, IC, gyro(rhTri)),

  // Chamfered solids — every edge chamfered (the live operation).
  E("Chamfered tetrahedron", Ch, TE, chamfer(tet)),
  E("Chamfered cube", Ch, OC, chamfer(cube)),
  E("Chamfered octahedron", Ch, OC, chamfer(oct)),
  E("Chamfered dodecahedron", Ch, IC, chamfer(dod)),
  E("Chamfered icosahedron", Ch, IC, chamfer(ico)),

  // Subdivided solids — every edge subdivided (the live operation).
  E("Subdivided tetrahedron", Sub, TE, subdivide(tet)),
  E("Subdivided cube", Sub, OC, subdivide(cube)),
  E("Subdivided octahedron", Sub, OC, subdivide(oct)),
  E("Subdivided dodecahedron", Sub, IC, subdivide(dod)),
  E("Subdivided icosahedron", Sub, IC, subdivide(ico)),
];

/** The family ("Platonic solid", …) of a named solid, or null if unknown. */
export function solidTypeFor(name: string): SolidType | null {
  return NAMED.find((n) => n.name === name)?.type ?? null;
}

// Case-insensitive lookup from a display name to its database entry. The
// LIBRARY diagram (config) lists names in Title Case ("Truncated Tetrahedron")
// while the database mixes case ("Truncated tetrahedron"), so normalize both.
const BY_NAME = new Map<string, NamedPolyhedron>();
for (const e of NAMED) BY_NAME.set(e.name.toLowerCase(), e);

/** The database entry (colored Polyhedron + scheme) for a named solid
 *  (case-insensitive), or null. */
export function namedPolyhedronFor(name: string): NamedPolyhedron | null {
  return BY_NAME.get(name.trim().toLowerCase()) ?? null;
}

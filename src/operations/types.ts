import { Vector3, Ray, Color } from "three";
import { type Mesh } from "../geometry/HalfEdge";
import { type ColorSet } from "../geometry/colors";

/** The kind of interactive operation a gesture maps to. */
export type OperationKind =
  | "truncate"
  | "kis"
  | "snub"
  | "gyro"
  | "chamfer"
  | "subdivide";

/**
 * A live, in-progress operation. Built when a drag starts; the topology is fixed
 * for the duration of the drag and only the parameter `t` (in [0, 1]) changes.
 *   t = 0  → geometrically identical to the original (no-op end)
 *   t = 1  → the "max" form: Rectify / Join (welded)
 */
export interface MorphPlan {
  kind: OperationKind;

  /** Topology shown DURING the drag (before any welding). */
  previewFaces: number[][];

  /** Vertex positions for a given parameter t. */
  positions(t: number): Vector3[];

  /**
   * Per-PREVIEW-face RGB color for the live drag, interpolated by t between each
   * face's t=0 appearance and its final ("at the drag limit") rule color. One
   * entry per `previewFaces` face. Used to animate colors while dragging.
   */
  previewFaceColors(t: number): Color[];

  /**
   * Palette indices for the preview topology's edges, keyed by undirected
   * preview-vertex-index pair (`edgeKey`). Used to draw the colored wireframe
   * during a drag (the dark-palette edge colors don't interpolate).
   */
  previewEdgeColors: Map<string, number>;

  /**
   * Edges (as PREVIEW vertex-index pairs) that collapse / dissolve at the weld.
   * When the drag is at the welded max, these are hidden so the about-to-merge
   * faces read as a single face even before the geometry is welded.
   */
  vanishingEdges: Array<[number, number]>;

  /**
   * Snap the camera pick ray to this operation's snap geometry and report the
   * resulting parameter, the exact snapped world point (where the dragged new
   * vertex should sit), and the segment to highlight (the edge / normal line
   * currently being dragged along).
   */
  snap(ray: Ray): {
    t: number;
    point: Vector3;
    highlight?: { a: Vector3; b: Vector3 };
  };

  /**
   * Final mesh + element colors for parameter t. When `weld` is true (t reached
   * the max end) the topology collapses to the Rectify/Join form; otherwise it is
   * the intermediate (truncated / kissed) topology at that t. The returned colors
   * are the NORMAL rule colors (the ico/dodeca special override is applied by the
   * caller).
   */
  commit(t: number, weld: boolean): { mesh: Mesh; colors: ColorSet };

  /**
   * For chiral operations (snub / gyro), the handedness of the currently-selected
   * mirror form — "R" or "L" — so a committed shape's name can distinguish the two
   * enantiomorphs (which are otherwise the same combinatorial signature). Reflects
   * the live `snap` choice; absent on achiral operations (truncate / kis).
   */
  chirality?(): "R" | "L";
}

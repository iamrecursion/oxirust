//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    ArtGalleryResult, ConvexFace3D, ConvexHull3D, DelaunayTri, Line2D, Obstacle2D, Point2,
    VoronoiCell2D,
};
use oxiphysics_core::exact_predicates::{Orientation, incircle, orient2d};

pub use super::delaunay_2d::delaunay_2d;

/// A point in 3D space represented as a plain array.
pub type Point3 = [f64; 3];
/// Subtract two 3D points.
pub fn sub3(a: Point3, b: Point3) -> Point3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
/// Add two 3D vectors.
pub fn add3(a: Point3, b: Point3) -> Point3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
/// Scale a 3D vector.
pub fn scale3(a: Point3, t: f64) -> Point3 {
    [a[0] * t, a[1] * t, a[2] * t]
}
/// Dot product of two 3D vectors.
pub fn dot3(a: Point3, b: Point3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
/// Cross product of two 3D vectors.
pub fn cross3(a: Point3, b: Point3) -> Point3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
/// Squared magnitude of a 3D vector.
pub fn mag3_sq(a: Point3) -> f64 {
    dot3(a, a)
}
/// Magnitude of a 3D vector.
pub fn mag3(a: Point3) -> f64 {
    mag3_sq(a).sqrt()
}
/// Normalize a 3D vector; returns the zero vector if the magnitude is too small.
pub fn normalize3(a: Point3) -> Point3 {
    let m = mag3(a);
    if m < 1e-15 {
        [0.0; 3]
    } else {
        scale3(a, 1.0 / m)
    }
}
/// Index of a half-edge in a `HalfEdgeMesh`.
pub type HalfEdgeId = usize;
/// Index of a vertex in a `HalfEdgeMesh`.
pub type VertexId = usize;
/// Index of a face in a `HalfEdgeMesh`.
pub type FaceId = usize;
/// Check if a point `p` is above (in the direction of the outward normal) a face defined
/// by vertices `a`, `b`, `c` (counter-clockwise order from outside).
pub(super) fn above_face(a: Point3, b: Point3, c: Point3, p: Point3) -> bool {
    let normal = cross3(sub3(b, a), sub3(c, a));
    dot3(normal, sub3(p, a)) > 1e-12
}
/// Build the initial tetrahedron from the first four non-coplanar points.
pub(super) fn initial_tetrahedron(pts: &[Point3]) -> Option<ConvexHull3D> {
    if pts.len() < 4 {
        return None;
    }
    let p0 = pts[0];
    let p1_idx = pts
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, p)| mag3(sub3(**p, p0)) > 1e-10)
        .map(|(i, _)| i)?;
    let p1 = pts[p1_idx];
    let p2_idx = pts
        .iter()
        .enumerate()
        .skip(p1_idx + 1)
        .find(|(_, p)| mag3(cross3(sub3(p1, p0), sub3(**p, p0))) > 1e-10)
        .map(|(i, _)| i)?;
    let p2 = pts[p2_idx];
    let mut p3_idx = None;
    let normal = cross3(sub3(p1, p0), sub3(p2, p0));
    for (i, &pt) in pts.iter().enumerate().skip(p2_idx + 1) {
        if dot3(normal, sub3(pt, p0)).abs() > 1e-10 {
            p3_idx = Some(i);
            break;
        }
    }
    let p3_idx = p3_idx?;
    let p3 = pts[p3_idx];
    let verts = vec![p0, p1, p2, p3];
    let cx = (p0[0] + p1[0] + p2[0] + p3[0]) / 4.0;
    let cy = (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0;
    let cz = (p0[2] + p1[2] + p2[2] + p3[2]) / 4.0;
    let center: Point3 = [cx, cy, cz];
    let face_triples: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    let mut faces = Vec::new();
    for tri in &face_triples {
        let a = verts[tri[0]];
        let b = verts[tri[1]];
        let c = verts[tri[2]];
        let mut normal = normalize3(cross3(sub3(b, a), sub3(c, a)));
        if dot3(normal, sub3(a, center)) < 0.0 {
            normal = scale3(normal, -1.0);
            faces.push(ConvexFace3D {
                verts: [tri[0], tri[2], tri[1]],
                normal,
            });
        } else {
            faces.push(ConvexFace3D {
                verts: *tri,
                normal,
            });
        }
    }
    Some(ConvexHull3D {
        vertices: verts,
        faces,
    })
}
/// Compute the 3D convex hull of a set of points using the incremental algorithm.
///
/// Returns `None` if fewer than 4 non-coplanar points are provided.
///
/// # Algorithm
/// 1. Build an initial tetrahedron from 4 non-coplanar seed points.
/// 2. For each remaining point, find all faces visible from it.
/// 3. Remove visible faces, identify the horizon edge loop, and connect horizon
///    edges to the new point to form new faces.
///
/// Time complexity: O(n²) worst case; O(n log n) expected for random points.
pub fn convex_hull_3d(points: &[Point3]) -> Option<ConvexHull3D> {
    if points.len() < 4 {
        return None;
    }
    let mut hull = initial_tetrahedron(points)?;
    let skip_eps = 1e-10;
    let mut remaining: Vec<Point3> = Vec::new();
    'outer: for &p in points.iter() {
        for v in &hull.vertices {
            if mag3(sub3(p, *v)) < skip_eps {
                continue 'outer;
            }
        }
        remaining.push(p);
    }
    for &pt in &remaining {
        let visible: Vec<bool> = hull
            .faces
            .iter()
            .map(|f| {
                let a = hull.vertices[f.verts[0]];
                above_face(a, hull.vertices[f.verts[1]], hull.vertices[f.verts[2]], pt)
            })
            .collect();
        let any_visible = visible.iter().any(|&v| v);
        if !any_visible {
            continue;
        }
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for (fi, face) in hull.faces.iter().enumerate() {
            if !visible[fi] {
                continue;
            }
            let tri = face.verts;
            let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
            for (ea, eb) in edges {
                let on_horizon = hull.faces.iter().enumerate().any(|(fj, g)| {
                    if visible[fj] {
                        return false;
                    }
                    let t = g.verts;
                    [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
                        .iter()
                        .any(|&(ga, gb)| ga == eb && gb == ea)
                });
                if on_horizon {
                    horizon.push((ea, eb));
                }
            }
        }
        let new_vid = hull.vertices.len();
        hull.vertices.push(pt);
        let n_v = hull.vertices.len() as f64;
        let cx: f64 = hull.vertices.iter().map(|v| v[0]).sum::<f64>() / n_v;
        let cy: f64 = hull.vertices.iter().map(|v| v[1]).sum::<f64>() / n_v;
        let cz: f64 = hull.vertices.iter().map(|v| v[2]).sum::<f64>() / n_v;
        let center = [cx, cy, cz];
        let new_faces: Vec<ConvexFace3D> = hull
            .faces
            .iter()
            .enumerate()
            .filter(|(fi, _)| !visible[*fi])
            .map(|(_, f)| f.clone())
            .collect();
        hull.faces = new_faces;
        for (ha, hb) in horizon {
            let a = hull.vertices[ha];
            let b = hull.vertices[hb];
            let mut normal = normalize3(cross3(sub3(b, a), sub3(pt, a)));
            let verts = if dot3(normal, sub3(a, center)) < 0.0 {
                normal = scale3(normal, -1.0);
                [ha, new_vid, hb]
            } else {
                [ha, hb, new_vid]
            };
            hull.faces.push(ConvexFace3D { verts, normal });
        }
    }
    Some(hull)
}
/// Clip a polygon against an infinite half-plane defined by the directed edge from
/// `edge_a` to `edge_b` (the inside is to the left).
///
/// Returns the clipped polygon vertices.
pub fn clip_polygon_by_edge(polygon: &[Point2], edge_a: Point2, edge_b: Point2) -> Vec<Point2> {
    if polygon.is_empty() {
        return vec![];
    }
    let n = polygon.len();
    let mut output = Vec::with_capacity(n + 1);
    for i in 0..n {
        let cur = polygon[i];
        let next = polygon[(i + 1) % n];
        let d_cur = Point2::cross2(edge_a, edge_b, cur);
        let d_next = Point2::cross2(edge_a, edge_b, next);
        if d_cur >= 0.0 {
            output.push(cur);
            if d_next < 0.0 {
                output.push(edge_intersect(cur, next, edge_a, edge_b));
            }
        } else {
            if d_next >= 0.0 {
                output.push(edge_intersect(cur, next, edge_a, edge_b));
            }
        }
    }
    output
}
/// Compute the intersection of segment (p, q) with the line through (a, b).
pub(super) fn edge_intersect(p: Point2, q: Point2, a: Point2, b: Point2) -> Point2 {
    let pq = q - p;
    let ab = b - a;
    let ap = a - p;
    let denom = pq.cross(ab);
    if denom.abs() < 1e-15 {
        return p;
    }
    let t = ap.cross(ab) / denom;
    p + pq * t
}
/// Clip a subject polygon against a convex clipping polygon using the
/// Sutherland-Hodgman algorithm.
///
/// Both polygons must be given as counter-clockwise vertex lists.
/// Returns the vertices of the clipped (intersection) polygon, which is
/// convex if the clipping polygon is convex.
pub fn sutherland_hodgman(subject: &[Point2], clip: &[Point2]) -> Vec<Point2> {
    if subject.is_empty() || clip.is_empty() {
        return vec![];
    }
    let mut output = subject.to_vec();
    let n = clip.len();
    for i in 0..n {
        if output.is_empty() {
            break;
        }
        let a = clip[i];
        let b = clip[(i + 1) % n];
        output = clip_polygon_by_edge(&output, a, b);
    }
    output
}
/// Compute the area of a simple polygon (positive for CCW).
pub fn polygon_area(poly: &[Point2]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i].x * poly[j].y;
        area -= poly[j].x * poly[i].y;
    }
    area / 2.0
}
/// Compute the Minkowski sum of two convex polygons P and Q.
///
/// Both polygons must be convex and are expected with their vertices in
/// counter-clockwise order (clockwise input is reversed, repeated vertices are
/// dropped).  A single point or a segment (one or two distinct vertices) is
/// accepted as a degenerate polygon.  The result is the convex polygon P ⊕ Q
/// in counter-clockwise order, starting at its bottom-most (then left-most)
/// vertex, and is empty when either input is empty.
///
/// # Algorithm
/// Rotating calipers / edge-angle merge:
/// 1. Orient both polygons counter-clockwise and rotate them so the
///    bottom-most (then left-most) vertex is first.  The edge directions of
///    such a polygon then have increasing polar angle in `[0, 2π)`.
/// 2. Merge the two edge sequences by polar angle, which is the sign of the
///    cross product of the two current edges.  Parallel edges of the same
///    direction are fused into one edge of the sum.  Once the edges of one
///    polygon are used up, the remaining edges of the other are appended.
/// 3. Walk the merged sequence starting at `P[0] + Q[0]` to build the sum.
///
/// Every step consumes at least one edge, so the walk ends after at most
/// `n + m` steps and the result has at most `n + m` vertices.  Collinear
/// vertices of the inputs stay in the result.
pub fn minkowski_sum(p: &[Point2], q: &[Point2]) -> Vec<Point2> {
    if p.is_empty() || q.is_empty() {
        return vec![];
    }
    let p_rot = ccw_from_bottom_vertex(p);
    let q_rot = ccw_from_bottom_vertex(q);
    let edges_p = polygon_edges(&p_rot);
    let edges_q = polygon_edges(&q_rot);
    let mut result = Vec::with_capacity(edges_p.len() + edges_q.len());
    let mut cur = p_rot[0] + q_rot[0];
    let mut i = 0;
    let mut j = 0;
    while i < edges_p.len() || j < edges_q.len() {
        result.push(cur);
        let step = match (edges_p.get(i), edges_q.get(j)) {
            (Some(&ep), Some(&eq)) => {
                let cross = ep.cross(eq);
                if cross > 0.0 {
                    i += 1;
                    ep
                } else if cross < 0.0 {
                    j += 1;
                    eq
                } else {
                    i += 1;
                    j += 1;
                    ep + eq
                }
            }
            (Some(&ep), None) => {
                i += 1;
                ep
            }
            (None, Some(&eq)) => {
                j += 1;
                eq
            }
            (None, None) => break,
        };
        cur = cur + step;
    }
    result
}
/// Prepare a convex polygon for the edge merge of [`minkowski_sum`].
///
/// Drops repeated consecutive vertices (including a closing copy of the first
/// vertex), reverses clockwise input and rotates the vertex list so that it
/// starts at the bottom-most (then left-most) vertex.
fn ccw_from_bottom_vertex(poly: &[Point2]) -> Vec<Point2> {
    let mut pts: Vec<Point2> = Vec::with_capacity(poly.len());
    for &pt in poly {
        if pts.last() != Some(&pt) {
            pts.push(pt);
        }
    }
    while pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    if polygon_area(&pts) < 0.0 {
        pts.reverse();
    }
    let start = bottom_vertex(&pts);
    pts.rotate_left(start);
    pts
}
/// Edge vectors `poly[k + 1] - poly[k]` of a closed polygon; a single vertex
/// has no edges.
fn polygon_edges(poly: &[Point2]) -> Vec<Point2> {
    let n = poly.len();
    if n < 2 {
        return vec![];
    }
    (0..n).map(|k| poly[(k + 1) % n] - poly[k]).collect()
}
/// Find the index of the bottom-most (then left-most) vertex.
pub(super) fn bottom_vertex(poly: &[Point2]) -> usize {
    let mut best = 0;
    for i in 1..poly.len() {
        let b = poly[best];
        let c = poly[i];
        if c.y < b.y || (c.y == b.y && c.x < b.x) {
            best = i;
        }
    }
    best
}
/// Intersection point of two lines (if they are not parallel).
pub fn line_intersect_2d(l1: &Line2D, l2: &Line2D) -> Option<Point2> {
    let det = l1.a * l2.b - l2.a * l1.b;
    if det.abs() < 1e-15 {
        return None;
    }
    let x = (l1.c * l2.b - l2.c * l1.b) / det;
    let y = (l1.a * l2.c - l2.a * l1.c) / det;
    Some(Point2::new(x, y))
}
/// Which parts of an open segment lie strictly inside / strictly outside a
/// simple polygon.  Stretches that run along the polygon boundary (or merely
/// touch it at a vertex) count as neither.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct SegmentExtent {
    /// A part of the open segment lies strictly inside the polygon.
    pub inside: bool,
    /// A part of the open segment lies strictly outside the polygon.
    pub outside: bool,
}
/// Shortest stretch of a segment (as a fraction of its length) that is still
/// classified; shorter stretches between two nearly coincident cut points carry
/// no geometric information.
const MIN_PIECE_FRACTION: f64 = 1e-12;
/// Exact side of `v` relative to the directed line `a -> b`: `1` on the left,
/// `-1` on the right and `0` exactly on the line.
fn side_of_line(a: Point2, b: Point2, v: Point2) -> i8 {
    match orient2d([a.x, a.y], [b.x, b.y], [v.x, v.y]) {
        Orientation::Positive => 1,
        Orientation::Negative => -1,
        Orientation::Degenerate => 0,
    }
}
/// For three exactly collinear points: `true` if `v` lies strictly between
/// `a` and `b`.  Only coordinates are compared, so the answer is exact.
pub(super) fn strictly_between_collinear(a: Point2, v: Point2, b: Point2) -> bool {
    let (lo, mid, hi) = if a.x != b.x {
        (a.x, v.x, b.x)
    } else {
        (a.y, v.y, b.y)
    };
    (lo < mid && mid < hi) || (hi < mid && mid < lo)
}
/// Parameter `t` of the orthogonal projection of `v` onto the line through `p`
/// and `q`, in `p + t (q - p)`.
fn parameter_along(p: Point2, q: Point2, v: Point2) -> f64 {
    let d = q - p;
    (v - p).dot(d) / d.dot(d)
}
/// Even-odd point-in-polygon test whose edge-side decisions use the exact
/// orientation predicate.  A point exactly on the boundary may be reported
/// either way; callers only ask about points known to be off the boundary.
fn point_in_polygon(pt: Point2, polygon: &[Point2]) -> bool {
    let n = polygon.len();
    let mut inside = false;
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        if (a.y > pt.y) != (b.y > pt.y) {
            let side = side_of_line(a, b, pt);
            if (b.y > a.y && side > 0) || (b.y < a.y && side < 0) {
                inside = !inside;
            }
        }
    }
    inside
}
/// Classify the open segment `(p, q)` against a simple polygon.
///
/// The segment is cut at every point where it meets the polygon boundary: a
/// proper crossing of an edge (which already puts a part of the segment on both
/// sides of the polygon), a vertex lying on the segment, and the ends of
/// collinear overlaps with an edge.  Between two consecutive cuts the segment
/// cannot cross the boundary, so each piece is wholly inside, wholly outside or
/// along an edge, and the midpoint of the piece decides.  Orientation tests are
/// exact, hence a segment that merely touches a vertex or runs along an edge is
/// never mistaken for one that enters the polygon, and a segment that passes
/// exactly through vertices (a square's diagonal, say) is not mistaken for one
/// that avoids it.
///
/// A polygon with fewer than three vertices or a zero-length segment has no
/// interior to enter, so the result is empty.
pub(super) fn segment_extent(p: Point2, q: Point2, polygon: &[Point2]) -> SegmentExtent {
    let n = polygon.len();
    if n < 3 || p == q {
        return SegmentExtent::default();
    }
    let sides: Vec<i8> = polygon.iter().map(|&v| side_of_line(p, q, v)).collect();
    // The line through the segment misses the polygon altogether.
    if sides.iter().all(|&s| s > 0) || sides.iter().all(|&s| s < 0) {
        return SegmentExtent {
            inside: false,
            outside: true,
        };
    }
    // A proper crossing of an edge leads from one side of the boundary to the other.
    for i in 0..n {
        let j = (i + 1) % n;
        if sides[i] * sides[j] < 0 && segments_properly_intersect(p, q, polygon[i], polygon[j]) {
            return SegmentExtent {
                inside: true,
                outside: true,
            };
        }
    }
    // Remaining boundary contacts: vertices exactly on the open segment ...
    let mut cuts = vec![0.0, 1.0];
    for (&v, &side) in polygon.iter().zip(sides.iter()) {
        if side == 0 && strictly_between_collinear(p, v, q) {
            cuts.push(parameter_along(p, q, v));
        }
    }
    cuts.sort_by(f64::total_cmp);
    // ... and edges lying along the line of the segment (parameter ranges, which
    // may extend beyond [0, 1]).
    let mut along_edges: Vec<(f64, f64)> = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        if sides[i] == 0 && sides[j] == 0 {
            let ta = parameter_along(p, q, polygon[i]);
            let tb = parameter_along(p, q, polygon[j]);
            along_edges.push((ta.min(tb), ta.max(tb)));
        }
    }
    let mut extent = SegmentExtent::default();
    for pair in cuts.windows(2) {
        let (t0, t1) = (pair[0], pair[1]);
        if t1 - t0 <= MIN_PIECE_FRACTION {
            continue;
        }
        let tm = 0.5 * (t0 + t1);
        if along_edges.iter().any(|&(lo, hi)| lo < tm && tm < hi) {
            continue;
        }
        if point_in_polygon(p + (q - p) * tm, polygon) {
            extent.inside = true;
        } else {
            extent.outside = true;
        }
        if extent.inside && extent.outside {
            break;
        }
    }
    extent
}
/// Test whether the open segment `(p, q)` is visible among polygonal
/// obstacles, i.e. whether it avoids the interior of every obstacle.
///
/// Grazing is allowed: the segment may touch an obstacle at a vertex or run along
/// one of its edges (two vertices of an obstacle that are joined by an edge see
/// each other).  A segment that passes through an obstacle, including through
/// vertices only - the diagonal of a square, say - is blocked.
pub(super) fn segment_visible(p: Point2, q: Point2, obstacles: &[Obstacle2D]) -> bool {
    obstacles
        .iter()
        .all(|obstacle| !segment_extent(p, q, &obstacle.vertices).inside)
}
/// Test whether the open segment `(p, q)` runs inside a simple polygon or along
/// its boundary (a "chord" of the polygon): it must not leave the polygon,
/// whether by crossing an edge or by cutting across a notch between two
/// vertices.
pub(super) fn segment_inside_polygon(p: Point2, q: Point2, polygon: &[Point2]) -> bool {
    !segment_extent(p, q, polygon).outside
}
/// Test whether two open segments properly intersect, i.e. cross at a point
/// that is interior to both.
///
/// Decided with exact orientation predicates: touching at an endpoint, sharing an
/// endpoint, running collinearly and passing exactly through an endpoint of the
/// other segment are not proper intersections.
pub(super) fn segments_properly_intersect(p: Point2, q: Point2, a: Point2, b: Point2) -> bool {
    side_of_line(a, b, p) * side_of_line(a, b, q) < 0
        && side_of_line(p, q, a) * side_of_line(p, q, b) < 0
}
/// Greedy vertex-guard approximation for the art gallery problem.
///
/// Given a simple polygon (vertices in CCW order), iteratively places a guard
/// at the vertex that covers the maximum number of currently uncovered vertices,
/// until all vertices are covered or `max_guards` is reached.  A guard covers a
/// vertex when the straight segment between them lies inside the polygon
/// (touching the boundary is allowed); a segment cutting across a notch of a
/// non-convex polygon does not count.
///
/// This is a set-cover greedy approximation with ratio O(log n).
///
/// Returns a [`ArtGalleryResult`] describing the guards and coverage.
pub fn art_gallery_greedy(poly: &[Point2], max_guards: usize) -> ArtGalleryResult {
    let n = poly.len();
    if n == 0 {
        return ArtGalleryResult {
            guards: vec![],
            covered: vec![],
        };
    }
    // Visibility is symmetric: decide each pair once.
    let mut vis = vec![vec![false; n]; n];
    for i in 0..n {
        vis[i][i] = true;
        for j in (i + 1)..n {
            let visible = segment_inside_polygon(poly[i], poly[j], poly);
            vis[i][j] = visible;
            vis[j][i] = visible;
        }
    }
    let mut covered = vec![false; n];
    let mut guards = Vec::new();
    for _ in 0..max_guards {
        if covered.iter().all(|&c| c) {
            break;
        }
        // Every uncovered vertex sees itself, so the best guard covers at least one.
        let Some(best) =
            (0..n).max_by_key(|&i| (0..n).filter(|&j| !covered[j] && vis[i][j]).count())
        else {
            break;
        };
        guards.push(best);
        for j in 0..n {
            if vis[best][j] {
                covered[j] = true;
            }
        }
    }
    ArtGalleryResult { guards, covered }
}
/// Check if every vertex of a polygon is covered by the given set of guards.
///
/// A guard covers the vertices whose joining segment lies inside the polygon
/// (see [`art_gallery_greedy`]); guard indices outside the polygon cover nothing.
///
/// Returns `true` if full coverage is achieved.
pub fn check_full_coverage(poly: &[Point2], guards: &[usize]) -> bool {
    let n = poly.len();
    (0..n).all(|j| {
        guards
            .iter()
            .any(|&g| g == j || (g < n && segment_inside_polygon(poly[g], poly[j], poly)))
    })
}
/// Circumcircle of three 2D points. Returns (center, radius²) or None if degenerate.
///
/// The triple is degenerate when its points are exactly collinear, which the
/// exact `orient2d` predicate decides without any absolute tolerance, so the
/// answer does not depend on the scale of the coordinates; `None` is also
/// returned when the circle is not representable (a non-finite center).
pub fn circumcircle_2d(pa: Point2, pb: Point2, pc: Point2) -> Option<(Point2, f64)> {
    if orient2d([pa.x, pa.y], [pb.x, pb.y], [pc.x, pc.y]) == Orientation::Degenerate {
        return None;
    }
    let ax = pb.x - pa.x;
    let ay = pb.y - pa.y;
    let bx = pc.x - pa.x;
    let by = pc.y - pa.y;
    let d = 2.0 * (ax * by - ay * bx);
    let ux = (by * (ax * ax + ay * ay) - ay * (bx * bx + by * by)) / d;
    let uy = (ax * (bx * bx + by * by) - bx * (ax * ax + ay * ay)) / d;
    let cx = pa.x + ux;
    let cy = pa.y + uy;
    let center = Point2::new(cx, cy);
    let r2 = center.dist_sq(pa);
    if cx.is_finite() && cy.is_finite() && r2.is_finite() {
        Some((center, r2))
    } else {
        None
    }
}
/// Test if point `p` lies strictly inside the circumcircle of triangle (a, b, c),
/// whichever way the triangle is oriented.
///
/// Decided with the exact `incircle` predicate: a point on the circle is not
/// inside, and a degenerate (collinear) triangle has no circumcircle, so nothing
/// is inside it.
pub fn in_circumcircle_2d(pa: Point2, pb: Point2, pc: Point2, p: Point2) -> bool {
    let (a, b, c, d) = ([pa.x, pa.y], [pb.x, pb.y], [pc.x, pc.y], [p.x, p.y]);
    match orient2d(a, b, c) {
        Orientation::Positive => incircle(a, b, c, d) == Orientation::Positive,
        Orientation::Negative => incircle(a, c, b, d) == Orientation::Positive,
        Orientation::Degenerate => false,
    }
}
/// Compute the dual Voronoi diagram from a Delaunay triangulation.
///
/// For each site point, collects all Delaunay triangles incident to that site
/// and returns their circumcenters as the Voronoi polygon vertices.
pub fn voronoi_from_delaunay(points: &[Point2], tris: &[DelaunayTri]) -> Vec<VoronoiCell2D> {
    let n = points.len();
    let mut cells: Vec<VoronoiCell2D> = (0..n)
        .map(|i| VoronoiCell2D {
            site: i,
            circumcenters: Vec::new(),
        })
        .collect();
    for t in tris {
        if let Some((center, _)) = circumcircle_2d(points[t.a], points[t.b], points[t.c]) {
            cells[t.a].circumcenters.push(center);
            cells[t.b].circumcenters.push(center);
            cells[t.c].circumcenters.push(center);
        }
    }
    for cell in &mut cells {
        let site = points[cell.site];
        cell.circumcenters.sort_by(|a, b| {
            let ta = (a.y - site.y).atan2(a.x - site.x);
            let tb = (b.y - site.y).atan2(b.x - site.x);
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    cells
}
/// Test if three 2D points are in counter-clockwise orientation.
pub fn ccw(a: Point2, b: Point2, c: Point2) -> bool {
    Point2::cross2(a, b, c) > 0.0
}
/// Test if three 2D points are collinear (within tolerance).
pub fn collinear(a: Point2, b: Point2, c: Point2) -> bool {
    Point2::cross2(a, b, c).abs() < 1e-10
}
/// Test if point `p` lies inside the triangle (a, b, c) using barycentric coordinates.
///
/// Returns `true` if strictly inside, `false` on boundary or outside.
pub fn point_in_triangle(p: Point2, a: Point2, b: Point2, c: Point2) -> bool {
    let d1 = Point2::cross2(p, a, b);
    let d2 = Point2::cross2(p, b, c);
    let d3 = Point2::cross2(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}
/// Test if point `p` lies inside a convex polygon (given in CCW order).
pub fn point_in_convex_polygon(p: Point2, poly: &[Point2]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if Point2::cross2(a, b, p) < 0.0 {
            return false;
        }
    }
    true
}
/// Compute the 2D convex hull of a point set (Graham scan).
///
/// Returns vertices in CCW order.
pub fn convex_hull_2d(points: &[Point2]) -> Vec<Point2> {
    let mut pts = points.to_vec();
    if pts.len() < 3 {
        return pts;
    }
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .expect("operation should succeed")
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| a.dist_sq(*b) < 1e-20);
    let n = pts.len();
    let mut hull: Vec<Point2> = Vec::with_capacity(2 * n);
    for &p in &pts {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            if Point2::cross2(a, b, p) <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    let lower_len = hull.len();
    for &p in pts.iter().rev() {
        while hull.len() > lower_len {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            if Point2::cross2(a, b, p) <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

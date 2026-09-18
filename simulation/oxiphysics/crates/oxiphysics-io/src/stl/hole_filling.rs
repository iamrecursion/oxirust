// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hole filling for STL meshes.
//!
//! [`repair_close_holes`] finds the boundary loops of a mesh (chains of edges
//! that belong to a single triangle) and closes each simple loop with a
//! triangulation made of non-degenerate caps that are wound opposite to the
//! faces around the hole, so the repaired surface stays consistently oriented.

use std::collections::HashMap;

use super::functions::{
    DEGENERACY_TOLERANCE, compute_normal, has_finite_corners, is_degenerate_points, quantise_vertex,
};
use super::types::{StlMesh, StlTriangle};

/// Close the holes of `mesh` and return the number of holes closed.
///
/// A hole is a closed loop of boundary edges, i.e. edges used by exactly one
/// triangle.  Coincident vertices are identified through the same 1e-6 grid as
/// [`validate_mesh`](super::validate_mesh).  Every loop is closed with an
/// ear-clipping triangulation in its best-fit plane (Newell normal), which also
/// handles concave outlines.  The caps reuse the original vertex coordinates of
/// the surrounding faces and are wound opposite to the faces next to them, so
/// each edge of a closed loop ends up shared by two triangles that traverse it
/// in opposite directions and the surface stays consistently oriented.
///
/// The triangulation never creates a degenerate triangle: a boundary vertex
/// that lies on a straight line between its neighbours still becomes a corner
/// of the caps (so no T-junction is left behind) but never the tip of a
/// sliver.
///
/// Loops are closed in order of their first appearance in the mesh, so the
/// result is deterministic.  Loops that cannot be closed safely are left
/// untouched:
///
/// * chains that are open, or that meet other chains in a vertex (a vertex
///   with more than one outgoing or incoming boundary edge, as at a pinched or
///   non-manifold place);
/// * loops whose outline is degenerate (all points on a line) or for which no
///   valid ear can be found (a self-overlapping outline).
///
/// Every simple boundary loop is closed whatever its size, including the outer
/// outline of an open surface, which turns a sheet into a zero-thickness
/// closed shell.  Triangles with a NaN or infinite coordinate are ignored.
pub fn repair_close_holes(mesh: &mut StlMesh) -> usize {
    let table = VertexTable::from_mesh(mesh);
    let mut closed = 0usize;
    let mut caps: Vec<StlTriangle> = Vec::new();
    for ring in table.boundary_loops() {
        let points: Vec<[f64; 3]> = ring
            .iter()
            .map(|&vertex| table.positions[vertex].map(f64::from))
            .collect();
        let Some(triangles) = triangulate_polygon(&points) else {
            continue;
        };
        for [a, b, c] in triangles {
            let v0 = table.positions[ring[a]];
            let v1 = table.positions[ring[b]];
            let v2 = table.positions[ring[c]];
            caps.push(StlTriangle {
                normal: compute_normal(v0, v1, v2),
                v0,
                v1,
                v2,
            });
        }
        closed += 1;
    }
    mesh.triangles.extend(caps);
    closed
}

/// The distinct vertices of a mesh and its triangles as indices into them.
struct VertexTable {
    /// Original coordinates of every distinct (grid-quantised) vertex, taken
    /// from its first occurrence in the mesh.
    positions: Vec<[f32; 3]>,
    /// The corners of the triangles with finite coordinates.
    corners: Vec<[usize; 3]>,
}

impl VertexTable {
    fn from_mesh(mesh: &StlMesh) -> Self {
        let mut ids: HashMap<[i64; 3], usize> = HashMap::new();
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut corners: Vec<[usize; 3]> = Vec::with_capacity(mesh.triangles.len());
        for tri in mesh.triangles.iter().filter(|tri| has_finite_corners(tri)) {
            let mut triple = [0usize; 3];
            for (slot, vertex) in triple.iter_mut().zip([tri.v0, tri.v1, tri.v2]) {
                *slot = *ids.entry(quantise_vertex(vertex)).or_insert_with(|| {
                    positions.push(vertex);
                    positions.len() - 1
                });
            }
            corners.push(triple);
        }
        Self { positions, corners }
    }

    /// The simple closed boundary loops of the mesh as vertex indices, each in
    /// the direction in which a cap has to traverse it.
    ///
    /// A boundary edge `a -> b` of its single triangle is traversed `b -> a` by
    /// the cap, opposite to the triangle.  Loops are listed in order of their
    /// first boundary edge in the mesh.
    fn boundary_loops(&self) -> Vec<Vec<usize>> {
        let edges_of = |ids: &[usize; 3]| [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])];
        let mut uses: HashMap<(usize, usize), usize> = HashMap::new();
        for ids in &self.corners {
            for (a, b) in edges_of(ids) {
                if a != b {
                    *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
        }
        let mut cap_edges: Vec<(usize, usize)> = Vec::new();
        for ids in &self.corners {
            for (a, b) in edges_of(ids) {
                if a != b && uses.get(&(a.min(b), a.max(b))) == Some(&1) {
                    cap_edges.push((b, a));
                }
            }
        }
        let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for (index, &(from, to)) in cap_edges.iter().enumerate() {
            outgoing.entry(from).or_default().push(index);
            *in_degree.entry(to).or_insert(0) += 1;
        }

        let mut visited = vec![false; cap_edges.len()];
        let mut loops: Vec<Vec<usize>> = Vec::new();
        let mut next_start = 0usize;
        while next_start < cap_edges.len() {
            let start = next_start;
            next_start += 1;
            if visited[start] {
                continue;
            }
            let mut chain: Vec<usize> = Vec::new();
            let mut edge = start;
            let mut closed = false;
            loop {
                visited[edge] = true;
                chain.push(edge);
                let head = cap_edges[edge].1;
                match outgoing.get(&head).map(Vec::as_slice) {
                    Some(&[next]) if next == start => {
                        closed = true;
                        break;
                    }
                    Some(&[next]) if !visited[next] => edge = next,
                    _ => break,
                }
            }
            let simple = chain
                .iter()
                .all(|&e| in_degree.get(&cap_edges[e].1) == Some(&1));
            if closed && simple && chain.len() >= 3 {
                loops.push(chain.iter().map(|&e| cap_edges[e].0).collect());
            }
        }
        loops
    }
}

type Vec3 = [f64; 3];

fn sub3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn dot3(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Twice the signed area of the planar triangle `a b c` (positive when it
/// turns counter-clockwise).
fn cross2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

/// Orthonormal frame of the best-fit plane of a closed polygon.
///
/// The polygon's Newell normal is the third axis, so a polygon that is
/// counter-clockwise about its Newell normal is counter-clockwise in the
/// `(u, v)` coordinates as well.
struct PlaneFrame {
    origin: Vec3,
    u: Vec3,
    v: Vec3,
}

impl PlaneFrame {
    /// Fit the frame to `points`; `None` when the outline encloses (almost) no
    /// area, i.e. when it is degenerate.
    fn fit(points: &[Vec3]) -> Option<Self> {
        let count = points.len() as f64;
        let sum = points.iter().fold([0.0; 3], |acc, &p| add3(acc, p));
        let origin = [sum[0] / count, sum[1] / count, sum[2] / count];
        let mut normal = [0.0; 3];
        for (i, &p) in points.iter().enumerate() {
            let q = points[(i + 1) % points.len()];
            normal = add3(normal, cross3(sub3(p, origin), sub3(q, origin)));
        }
        let length = dot3(normal, normal).sqrt();
        let extent_sq = points
            .iter()
            .map(|&p| {
                let d = sub3(p, origin);
                dot3(d, d)
            })
            .fold(0.0_f64, f64::max);
        if !length.is_finite() || !extent_sq.is_finite() || length <= 1e-9 * extent_sq {
            return None;
        }
        let unit = [normal[0] / length, normal[1] / length, normal[2] / length];
        // Any axis not parallel to the normal gives a perpendicular direction;
        // the least aligned one is the best conditioned.
        let axis = if unit[0].abs() <= unit[1].abs() && unit[0].abs() <= unit[2].abs() {
            [1.0, 0.0, 0.0]
        } else if unit[1].abs() <= unit[2].abs() {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        let u_raw = cross3(axis, unit);
        let u_len = dot3(u_raw, u_raw).sqrt();
        let u = [u_raw[0] / u_len, u_raw[1] / u_len, u_raw[2] / u_len];
        let v = cross3(unit, u);
        Some(Self { origin, u, v })
    }

    fn project(&self, p: Vec3) -> [f64; 2] {
        let d = sub3(p, self.origin);
        [dot3(d, self.u), dot3(d, self.v)]
    }
}

/// Whether `p` lies in the closed counter-clockwise triangle `a b c`, up to
/// `slack` (a length): a point at most `slack` outside an edge counts as being
/// on it.
fn in_closed_triangle(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2], slack: f64) -> bool {
    [(a, b), (b, c), (c, a)]
        .iter()
        .all(|&(from, to)| cross2(from, to, p) >= -slack * distance(from, to))
}

/// Ear clipping state: a circular doubly linked list of the polygon corners.
struct EarClipper<'a> {
    points: &'a [Vec3],
    flat: Vec<[f64; 2]>,
    prev: Vec<usize>,
    next: Vec<usize>,
    alive: Vec<bool>,
    /// Whether the corner is strictly convex and its triangle with the two
    /// neighbours is not degenerate (i.e. it can be the tip of an ear).
    convex: Vec<bool>,
    /// Corners that are not convex (reflex, straight or sliver): the only
    /// ones that can obstruct an ear.  May contain stale or repeated entries;
    /// `alive` and `convex` give the truth.
    suspects: Vec<usize>,
}

impl<'a> EarClipper<'a> {
    fn new(points: &'a [Vec3], flat: Vec<[f64; 2]>) -> Self {
        let n = points.len();
        let mut clipper = Self {
            points,
            flat,
            prev: (0..n).map(|i| (i + n - 1) % n).collect(),
            next: (0..n).map(|i| (i + 1) % n).collect(),
            alive: vec![true; n],
            convex: vec![false; n],
            suspects: Vec::new(),
        };
        for i in 0..n {
            clipper.convex[i] = clipper.is_proper_corner(i);
            if !clipper.convex[i] {
                clipper.suspects.push(i);
            }
        }
        clipper
    }

    /// Strictly convex corner whose triangle with its neighbours is not
    /// degenerate.
    fn is_proper_corner(&self, i: usize) -> bool {
        let (a, c) = (self.prev[i], self.next[i]);
        cross2(self.flat[a], self.flat[i], self.flat[c]) > 0.0
            && !is_degenerate_points(self.points[a], self.points[i], self.points[c])
    }

    /// Whether clipping corner `i` is valid: it is a proper corner and no
    /// other corner that is not convex lies in (or on) the triangle.
    fn is_ear(&self, i: usize) -> bool {
        if !self.convex[i] {
            return false;
        }
        let (a, c) = (self.prev[i], self.next[i]);
        let (pa, pi, pc) = (self.flat[a], self.flat[i], self.flat[c]);
        let longest = distance(pa, pi).max(distance(pi, pc)).max(distance(pc, pa));
        let slack = DEGENERACY_TOLERANCE * longest;
        !self.suspects.iter().any(|&j| {
            self.alive[j]
                && !self.convex[j]
                && j != a
                && j != i
                && j != c
                && in_closed_triangle(self.flat[j], pa, pi, pc, slack)
        })
    }

    /// Cut corner `i` off; returns the triangle and the corner to continue
    /// the search from.
    fn clip(&mut self, i: usize) -> ([usize; 3], usize) {
        let (a, c) = (self.prev[i], self.next[i]);
        self.alive[i] = false;
        self.next[a] = c;
        self.prev[c] = a;
        for j in [a, c] {
            let proper = self.is_proper_corner(j);
            if self.convex[j] && !proper {
                self.suspects.push(j);
            }
            self.convex[j] = proper;
        }
        ([a, i, c], self.next[c])
    }
}

/// Triangulate a closed polygon in 3D by ear clipping in its best-fit plane.
///
/// `points` is the outline in the order in which the triangles have to
/// traverse it: every returned triangle `[a, b, c]` (indices into `points`)
/// is counter-clockwise about the Newell normal of the outline, and no triangle
/// is degenerate.  `None` when the outline has no area or no ear can be found.
fn triangulate_polygon(points: &[Vec3]) -> Option<Vec<[usize; 3]>> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let frame = PlaneFrame::fit(points)?;
    let flat: Vec<[f64; 2]> = points.iter().map(|&p| frame.project(p)).collect();
    let mut clipper = EarClipper::new(points, flat);
    let mut triangles: Vec<[usize; 3]> = Vec::with_capacity(n - 2);
    let mut remaining = n;
    let mut cursor = 0usize;
    let mut stalled = 0usize;
    while remaining > 3 {
        if clipper.is_ear(cursor) {
            let (triangle, resume) = clipper.clip(cursor);
            triangles.push(triangle);
            remaining -= 1;
            cursor = resume;
            stalled = 0;
        } else {
            cursor = clipper.next[cursor];
            stalled += 1;
            if stalled > remaining {
                return None;
            }
        }
    }
    if !clipper.is_proper_corner(cursor) {
        return None;
    }
    triangles.push([clipper.prev[cursor], cursor, clipper.next[cursor]]);
    Some(triangles)
}

#[cfg(test)]
mod tests {
    use super::super::functions::validate_mesh;
    use super::*;
    use std::collections::HashSet;

    /// Unit cube with every face cut into `n x n` quads (two triangles each),
    /// wound counter-clockwise seen from outside.
    fn box_mesh(n: usize) -> StlMesh {
        // (corner, edge u, edge v) with u x v pointing out of the cube.
        let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        let mut mesh = StlMesh::new("box");
        for (corner, eu, ev) in faces {
            let at = |i: usize, j: usize| -> [f32; 3] {
                let s = i as f32 / n as f32;
                let t = j as f32 / n as f32;
                [
                    corner[0] + s * eu[0] + t * ev[0],
                    corner[1] + s * eu[1] + t * ev[1],
                    corner[2] + s * eu[2] + t * ev[2],
                ]
            };
            for i in 0..n {
                for j in 0..n {
                    mesh.add_triangle(at(i, j), at(i + 1, j), at(i + 1, j + 1));
                    mesh.add_triangle(at(i, j), at(i + 1, j + 1), at(i, j + 1));
                }
            }
        }
        mesh
    }

    /// Remove every triangle that lies entirely in the plane `axis = value`.
    fn remove_face(mesh: &mut StlMesh, axis: usize, value: f32) {
        mesh.triangles
            .retain(|t| ![t.v0, t.v1, t.v2].iter().all(|v| v[axis] == value));
    }

    fn flip(mesh: &mut StlMesh) {
        for t in &mut mesh.triangles {
            std::mem::swap(&mut t.v1, &mut t.v2);
            t.normal = compute_normal(t.v0, t.v1, t.v2);
        }
    }

    /// Signed volume enclosed by the surface (divergence theorem).
    fn signed_volume(mesh: &StlMesh) -> f64 {
        mesh.triangles
            .iter()
            .map(|t| {
                let a = t.v0.map(f64::from);
                let b = t.v1.map(f64::from);
                let c = t.v2.map(f64::from);
                dot3(a, cross3(b, c)) / 6.0
            })
            .sum()
    }

    fn total_area(triangles: &[StlTriangle]) -> f64 {
        triangles
            .iter()
            .map(|t| {
                let e1 = sub3(t.v1.map(f64::from), t.v0.map(f64::from));
                let e2 = sub3(t.v2.map(f64::from), t.v0.map(f64::from));
                let c = cross3(e1, e2);
                0.5 * dot3(c, c).sqrt()
            })
            .sum()
    }

    /// Every directed edge occurs once and its reverse occurs once: the mesh is
    /// a closed, consistently oriented surface.
    fn is_consistently_oriented_closed_surface(mesh: &StlMesh) -> bool {
        let mut directed: HashMap<([i64; 3], [i64; 3]), usize> = HashMap::new();
        for t in &mesh.triangles {
            let k = [
                quantise_vertex(t.v0),
                quantise_vertex(t.v1),
                quantise_vertex(t.v2),
            ];
            for (a, b) in [(k[0], k[1]), (k[1], k[2]), (k[2], k[0])] {
                *directed.entry((a, b)).or_insert(0) += 1;
            }
        }
        directed
            .iter()
            .all(|(&(a, b), &count)| count == 1 && directed.get(&(b, a)) == Some(&1))
    }

    fn assert_closed_and_clean(mesh: &StlMesh) {
        let report = validate_mesh(mesh);
        assert!(report.is_watertight, "not watertight: {report:?}");
        assert!(report.is_manifold, "not manifold: {report:?}");
        assert_eq!(report.degenerate_count, 0, "degenerate caps: {report:?}");
        assert!(mesh.is_watertight(), "bitwise edge matching fails");
        assert!(
            is_consistently_oriented_closed_surface(mesh),
            "caps are wound inconsistently"
        );
    }

    #[test]
    fn the_reference_box_is_a_consistently_oriented_outward_solid() {
        let mesh = box_mesh(1);
        assert_eq!(mesh.triangles.len(), 12);
        assert!(is_consistently_oriented_closed_surface(&mesh));
        assert!((signed_volume(&mesh) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cube_without_a_face_is_closed_by_two_caps() {
        let mut mesh = box_mesh(1);
        remove_face(&mut mesh, 2, 0.0);
        assert_eq!(mesh.triangles.len(), 10);
        assert!(!validate_mesh(&mesh).is_watertight);

        assert_eq!(repair_close_holes(&mut mesh), 1);
        assert_eq!(mesh.triangles.len(), 12);
        assert_closed_and_clean(&mesh);
        assert!((signed_volume(&mesh) - 1.0).abs() < 1e-9);
        // The caps tile the missing unit square.
        assert!((total_area(&mesh.triangles[10..]) - 1.0).abs() < 1e-9);
        // They face outwards (-z) like the face that was removed.
        for cap in &mesh.triangles[10..] {
            assert!(cap.normal[2] < -0.99, "cap normal {:?}", cap.normal);
        }
    }

    #[test]
    fn inward_wound_cube_stays_consistently_inward_after_repair() {
        let mut mesh = box_mesh(1);
        flip(&mut mesh);
        remove_face(&mut mesh, 0, 1.0);
        assert_eq!(repair_close_holes(&mut mesh), 1);
        assert_closed_and_clean(&mesh);
        assert!((signed_volume(&mesh) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn hole_with_collinear_boundary_vertices_gets_no_degenerate_caps() {
        // Every face is a 2 x 2 grid, so the hole is a square outline with a
        // vertex in the middle of each side: eight points, four of them straight.
        let mut mesh = box_mesh(2);
        assert_eq!(mesh.triangles.len(), 48);
        remove_face(&mut mesh, 1, 1.0);
        assert_eq!(mesh.triangles.len(), 40);

        assert_eq!(repair_close_holes(&mut mesh), 1);
        // An 8-gon needs 8 - 2 = 6 triangles.
        assert_eq!(mesh.triangles.len(), 46);
        assert_closed_and_clean(&mesh);
        assert!((signed_volume(&mesh) - 1.0).abs() < 1e-9);
        assert!((total_area(&mesh.triangles[40..]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn caps_reuse_the_exact_coordinates_of_the_surrounding_faces() {
        // Thirds are not representable: a de-quantised cap vertex would differ
        // from its neighbour in the last bits and break bitwise edge matching.
        let mut mesh = box_mesh(3);
        let original = mesh.triangles.len();
        assert_eq!(original, 108);
        remove_face(&mut mesh, 2, 1.0);
        let kept = mesh.triangles.len();
        assert_eq!(kept, original - 18);
        let existing: HashSet<[u32; 3]> = mesh
            .triangles
            .iter()
            .flat_map(|t| [t.v0, t.v1, t.v2])
            .map(|v| v.map(f32::to_bits))
            .collect();
        assert_eq!(repair_close_holes(&mut mesh), 1);
        // The hole outline has 12 vertices, so it needs 10 triangles.
        assert_eq!(mesh.triangles.len(), kept + 10);
        for cap in &mesh.triangles[kept..] {
            for v in [cap.v0, cap.v1, cap.v2] {
                assert!(
                    existing.contains(&v.map(f32::to_bits)),
                    "invented vertex {v:?}"
                );
            }
        }
        assert_closed_and_clean(&mesh);
    }

    #[test]
    fn two_separate_holes_are_both_closed() {
        let mut mesh = box_mesh(1);
        remove_face(&mut mesh, 2, 0.0);
        remove_face(&mut mesh, 2, 1.0);
        assert_eq!(mesh.triangles.len(), 8);
        assert_eq!(repair_close_holes(&mut mesh), 2);
        assert_eq!(mesh.triangles.len(), 12);
        assert_closed_and_clean(&mesh);
        assert!((signed_volume(&mesh) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn closed_mesh_is_left_alone() {
        let mut mesh = box_mesh(2);
        let before = mesh.triangles.len();
        assert_eq!(repair_close_holes(&mut mesh), 0);
        assert_eq!(mesh.triangles.len(), before);
    }

    #[test]
    fn empty_mesh_is_left_alone() {
        let mut mesh = StlMesh::new("empty");
        assert_eq!(repair_close_holes(&mut mesh), 0);
        assert!(mesh.triangles.is_empty());
    }

    /// An L-shaped prism: the outline below (a concave hexagon with one reflex
    /// corner at `(1, 1)`) extruded by one unit, with its top and bottom
    /// triangulated as a fan from the origin corner so that no vertex lies in
    /// the middle of an outline edge.
    fn l_prism() -> StlMesh {
        let outline: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let at = |p: [f32; 2], z: f32| [p[0], p[1], z];
        let mut mesh = StlMesh::new("l_prism");
        for k in 1..outline.len() - 1 {
            let (a, b, c) = (outline[0], outline[k], outline[k + 1]);
            // top, counter-clockwise seen from +z
            mesh.add_triangle(at(a, 1.0), at(b, 1.0), at(c, 1.0));
            // bottom, clockwise seen from +z
            mesh.add_triangle(at(a, 0.0), at(c, 0.0), at(b, 0.0));
        }
        for i in 0..outline.len() {
            let a = outline[i];
            let b = outline[(i + 1) % outline.len()];
            mesh.add_triangle(at(a, 0.0), at(b, 0.0), at(b, 1.0));
            mesh.add_triangle(at(a, 0.0), at(b, 1.0), at(a, 1.0));
        }
        mesh
    }

    #[test]
    fn l_prism_is_a_consistently_oriented_solid_of_volume_three() {
        let mesh = l_prism();
        assert_eq!(mesh.triangles.len(), 8 + 12);
        assert!(is_consistently_oriented_closed_surface(&mesh));
        assert!((signed_volume(&mesh) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn concave_hole_is_closed_without_overlap() {
        let mut mesh = l_prism();
        // Cut away the four triangles of the top face.
        remove_face(&mut mesh, 2, 1.0);
        assert_eq!(mesh.triangles.len(), 16);
        assert!(!validate_mesh(&mesh).is_watertight);

        assert_eq!(repair_close_holes(&mut mesh), 1);
        // The hexagonal outline needs four triangles.
        assert_eq!(mesh.triangles.len(), 20);
        assert_closed_and_clean(&mesh);
        assert!((signed_volume(&mesh) - 3.0).abs() < 1e-9);
        // The caps tile exactly the L-shaped outline (area 3) and face +z.
        let caps = &mesh.triangles[16..];
        assert!(
            (total_area(caps) - 3.0).abs() < 1e-9,
            "cap area {}",
            total_area(caps)
        );
        for cap in caps {
            assert!(cap.normal[2] > 0.99, "cap normal {:?}", cap.normal);
        }
    }

    #[test]
    fn non_planar_hole_across_a_cube_edge_is_closed_consistently() {
        // Removing two adjacent faces leaves one bent, non-planar outline.
        let mut mesh = box_mesh(1);
        remove_face(&mut mesh, 0, 0.0);
        remove_face(&mut mesh, 1, 1.0);
        assert_eq!(mesh.triangles.len(), 8);
        assert_eq!(repair_close_holes(&mut mesh), 1);
        // The outline is a hexagon: four caps.
        assert_eq!(mesh.triangles.len(), 12);
        assert_closed_and_clean(&mesh);
    }

    #[test]
    fn concave_polygon_triangulation_covers_the_polygon_exactly() {
        // L shape counter-clockwise.
        let l: Vec<Vec3> = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let triangles = triangulate_polygon(&l).expect("L shape is triangulable");
        assert_eq!(triangles.len(), 4);
        let area: f64 = triangles
            .iter()
            .map(|&[a, b, c]| {
                0.5 * cross2([l[a][0], l[a][1]], [l[b][0], l[b][1]], [l[c][0], l[c][1]])
            })
            .sum();
        assert!((area - 3.0).abs() < 1e-12);
        for &[a, b, c] in &triangles {
            let signed = cross2([l[a][0], l[a][1]], [l[b][0], l[b][1]], [l[c][0], l[c][1]]);
            assert!(
                signed > 0.0,
                "triangle {a} {b} {c} is not counter-clockwise"
            );
        }
    }

    #[test]
    fn polygon_in_a_tilted_plane_is_triangulated_counter_clockwise_about_its_normal() {
        // A square in the plane x + y + z = 3 (corners listed counter-clockwise
        // about the (1,1,1) direction).
        let square: Vec<Vec3> = vec![
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ];
        let triangles = triangulate_polygon(&square).expect("planar quad");
        assert_eq!(triangles.len(), 2);
        for &[a, b, c] in &triangles {
            let n = cross3(sub3(square[b], square[a]), sub3(square[c], square[a]));
            assert!(dot3(n, [1.0, 1.0, 1.0]) > 0.0);
        }
    }

    #[test]
    fn degenerate_outlines_are_not_triangulated() {
        let line: Vec<Vec3> = vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        assert!(triangulate_polygon(&line).is_none());
        assert!(triangulate_polygon(&[[0.0; 3], [1.0, 0.0, 0.0]]).is_none());
    }

    #[test]
    fn pinched_boundary_is_not_closed() {
        // Two triangles that touch in a single vertex: that vertex has two
        // outgoing and two incoming boundary edges.
        let mut mesh = StlMesh::new("bow_tie");
        mesh.add_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        mesh.add_triangle([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]);
        assert_eq!(repair_close_holes(&mut mesh), 0);
        assert_eq!(mesh.triangles.len(), 2);
    }

    #[test]
    fn lone_triangle_is_closed_by_a_back_face() {
        let mut mesh = StlMesh::new("single");
        mesh.add_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(repair_close_holes(&mut mesh), 1);
        assert_eq!(mesh.triangles.len(), 2);
        assert!(mesh.triangles[0].normal[2] > 0.99);
        assert!(mesh.triangles[1].normal[2] < -0.99);
        assert_closed_and_clean(&mesh);
    }

    #[test]
    fn triangles_with_non_finite_coordinates_are_ignored() {
        let mut mesh = box_mesh(1);
        remove_face(&mut mesh, 2, 0.0);
        mesh.triangles.push(StlTriangle {
            normal: [0.0; 3],
            v0: [f32::NAN, 0.0, 0.0],
            v1: [1.0, 0.0, 0.0],
            v2: [0.0, 1.0, 0.0],
        });
        assert_eq!(repair_close_holes(&mut mesh), 1);
        assert_eq!(mesh.triangles.len(), 10 + 1 + 2);
    }

    #[test]
    fn repair_is_deterministic() {
        let build = || {
            let mut mesh = box_mesh(3);
            remove_face(&mut mesh, 0, 0.0);
            remove_face(&mut mesh, 0, 1.0);
            mesh
        };
        let mut first = build();
        let mut second = build();
        assert_eq!(repair_close_holes(&mut first), 2);
        assert_eq!(repair_close_holes(&mut second), 2);
        let bits = |mesh: &StlMesh| -> Vec<[u32; 9]> {
            mesh.triangles
                .iter()
                .map(|t| {
                    let mut b = [0u32; 9];
                    for (k, f) in t.v0.iter().chain(&t.v1).chain(&t.v2).enumerate() {
                        b[k] = f.to_bits();
                    }
                    b
                })
                .collect()
        };
        assert_eq!(bits(&first), bits(&second));
    }
}

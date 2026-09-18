// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! 3D Convex Hull via QuickHull algorithm.
//!
//! Provides convex hull construction, vertex/face/edge enumeration,
//! volume and surface area computation, point-in-hull test, hull merging,
//! and support queries.

use oxiphysics_core::math::{Mat3, Vec3};

use crate::convex_hull::quickhull_core::{HullTolerances, build_hull, cloud_extent};

/// Relative tolerance of the hull construction.
///
/// A point only sees a hull face when it lies further than
/// `HULL_RELATIVE_TOLERANCE * extent` in front of the face plane, where `extent`
/// is the diagonal of the bounding box of the input. The tolerance scales with
/// the input, so point clouds of any size are treated alike.
const HULL_RELATIVE_TOLERANCE: f64 = 1e-10;

/// 3D Convex Hull computed via QuickHull algorithm (nalgebra Vec3 variant).
pub struct ConvexHull3DVec {
    /// Hull vertices (deduplicated subset of input points).
    pub vertices: Vec<Vec3>,
    /// Triangular faces stored as triples of vertex indices (outward-facing winding).
    pub faces: Vec<[usize; 3]>,
}

impl ConvexHull3DVec {
    /// Compute convex hull from a point cloud using the QuickHull algorithm.
    ///
    /// Returns `None` if the input is degenerate (fewer than 4 non-coplanar
    /// points) or contains a non-finite coordinate.
    ///
    /// This is the Quickhull core shared with
    /// [`ConvexHull3D::build`](crate::convex_hull::ConvexHull3D::build).  The hull
    /// grows from a seed tetrahedron of extreme points by repeatedly adding the
    /// point farthest in front of some face.  Every face keeps the points in
    /// front of it (its outside set); the faces that see the new point are found
    /// over the face adjacency, replaced by a fan of new faces from their
    /// horizon edges to the point (each new face keeps the direction of the
    /// horizon edge it stands on, so all faces stay consistently wound,
    /// counter-clockwise seen from outside, without any geometric guessing), and
    /// the outside points of the removed faces are handed on to the new faces.
    /// Points in front of no face are never revisited, which makes the expected
    /// cost close to `O(n log n)` for `n` points.
    ///
    /// Points closer to a face plane than `1e-10` times the diagonal of the
    /// bounding box count as inside and are not added, which makes the result
    /// independent of the scale of the input; the same distance is the smallest
    /// height of the seed tetrahedron, so clouds that are flatter than that have
    /// no hull.  A face is removed for an added point when the point is in front
    /// of it by more than a hundredth of that distance, and also when keeping it
    /// would leave an edge that is reflex by more than the tolerance; the rounded
    /// planes of thin faces are not trusted and the exact orientation predicate
    /// decides for them.  For input whose noise is of the size of the tolerance
    /// itself the surface stays closed and consistently wound but can be slightly
    /// non-convex.
    pub fn build(points: &[Vec3]) -> Option<Self> {
        let cloud: Vec<[f64; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();
        let tolerance = HULL_RELATIVE_TOLERANCE * cloud_extent(&cloud)?.diagonal;
        let surface = build_hull(
            &cloud,
            HullTolerances {
                visible_distance: tolerance,
                seed_distance: tolerance,
            },
        )?;
        let vertices = surface
            .vertices
            .iter()
            .map(|v| Vec3::new(v[0], v[1], v[2]))
            .collect();
        Some(Self {
            vertices,
            faces: surface.faces,
        })
    }

    /// Support function: returns the farthest point in direction `d`.
    pub fn support(&self, d: Vec3) -> Vec3 {
        self.vertices
            .iter()
            .copied()
            .max_by(|a, b| {
                a.dot(&d)
                    .partial_cmp(&b.dot(&d))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(Vec3::zeros)
    }

    /// Volume of the convex hull via signed tetrahedral decomposition from the centroid.
    pub fn volume(&self) -> f64 {
        if self.faces.is_empty() || self.vertices.is_empty() {
            return 0.0;
        }
        let c = self.centroid();
        let mut vol = 0.0f64;
        for face in &self.faces {
            let a = self.vertices[face[0]] - c;
            let b = self.vertices[face[1]] - c;
            let cc = self.vertices[face[2]] - c;
            vol += a.dot(&b.cross(&cc));
        }
        (vol / 6.0).abs()
    }

    /// Surface area of the convex hull (sum of triangle areas).
    pub fn surface_area(&self) -> f64 {
        let mut area = 0.0f64;
        for face in &self.faces {
            let a = self.vertices[face[0]];
            let b = self.vertices[face[1]];
            let c = self.vertices[face[2]];
            area += (b - a).cross(&(c - a)).norm() * 0.5;
        }
        area
    }

    /// Returns `true` if point `p` is inside (or on the boundary of) the hull.
    ///
    /// A point is inside iff it is on the non-positive side of every face plane.
    pub fn contains_point(&self, p: Vec3) -> bool {
        for face in &self.faces {
            let n = self.face_normal_raw(face);
            let off = n.dot(&self.vertices[face[0]]);
            if point_plane_distance(p, n, off) > 1e-9 {
                return false;
            }
        }
        true
    }

    /// Outward-pointing face normal for face `face_idx`.
    pub fn face_normal(&self, face_idx: usize) -> Vec3 {
        let face = &self.faces[face_idx];
        self.face_normal_raw(face)
    }

    /// Number of hull vertices.
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of triangular faces.
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Get the list of unique edges as pairs of vertex indices.
    ///
    /// Each edge appears once, with the smaller index first.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edge_set = std::collections::BTreeSet::new();
        for face in &self.faces {
            for i in 0..3 {
                let a = face[i];
                let b = face[(i + 1) % 3];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edge_set.insert((lo, hi));
            }
        }
        edge_set.into_iter().collect()
    }

    /// Number of unique edges.
    pub fn n_edges(&self) -> usize {
        self.edges().len()
    }

    /// Enumerate all vertices with their indices.
    pub fn vertex_iter(&self) -> impl Iterator<Item = (usize, &Vec3)> {
        self.vertices.iter().enumerate()
    }

    /// Enumerate all faces with their indices.
    pub fn face_iter(&self) -> impl Iterator<Item = (usize, &[usize; 3])> {
        self.faces.iter().enumerate()
    }

    /// Area of a specific face.
    pub fn face_area(&self, face_idx: usize) -> f64 {
        let face = &self.faces[face_idx];
        let a = self.vertices[face[0]];
        let b = self.vertices[face[1]];
        let c = self.vertices[face[2]];
        (b - a).cross(&(c - a)).norm() * 0.5
    }

    /// Compute the AABB of the hull.
    pub fn aabb(&self) -> (Vec3, Vec3) {
        if self.vertices.is_empty() {
            return (Vec3::zeros(), Vec3::zeros());
        }
        let mut mn = self.vertices[0];
        let mut mx = self.vertices[0];
        for v in &self.vertices[1..] {
            for k in 0..3 {
                if v[k] < mn[k] {
                    mn[k] = v[k];
                }
                if v[k] > mx[k] {
                    mx[k] = v[k];
                }
            }
        }
        (mn, mx)
    }

    /// Compute the diameter of the hull (maximum distance between any two vertices).
    pub fn diameter(&self) -> f64 {
        let mut max_dist = 0.0f64;
        for i in 0..self.vertices.len() {
            for j in (i + 1)..self.vertices.len() {
                let d = (self.vertices[i] - self.vertices[j]).norm();
                if d > max_dist {
                    max_dist = d;
                }
            }
        }
        max_dist
    }

    /// Euler characteristic: V - E + F = 2 for convex polyhedra.
    pub fn euler_characteristic(&self) -> i64 {
        self.n_vertices() as i64 - self.n_edges() as i64 + self.n_faces() as i64
    }

    /// Signed distance from point `p` to the hull surface.
    ///
    /// Negative if inside, positive if outside.
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        let mut max_signed = f64::NEG_INFINITY;
        for face in &self.faces {
            let n = self.face_normal_raw(face);
            let off = n.dot(&self.vertices[face[0]]);
            let d = point_plane_distance(p, n, off);
            if d > max_signed {
                max_signed = d;
            }
        }
        max_signed
    }

    /// Check if two hulls overlap (conservative test using separating axis theorem
    /// on face normals only).
    pub fn overlaps_sat_faces(&self, other: &ConvexHull3DVec) -> bool {
        // Check all face normals of self
        for fi in 0..self.n_faces() {
            let n = self.face_normal(fi);
            let (a_min, a_max) = project_hull_on_axis(&self.vertices, n);
            let (b_min, b_max) = project_hull_on_axis(&other.vertices, n);
            if a_max < b_min || b_max < a_min {
                return false; // separating axis found
            }
        }
        // Check all face normals of other
        for fi in 0..other.n_faces() {
            let n = other.face_normal(fi);
            let (a_min, a_max) = project_hull_on_axis(&self.vertices, n);
            let (b_min, b_max) = project_hull_on_axis(&other.vertices, n);
            if a_max < b_min || b_max < a_min {
                return false;
            }
        }
        true
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn centroid(&self) -> Vec3 {
        let sum: Vec3 = self.vertices.iter().copied().sum();
        sum / self.vertices.len() as f64
    }

    fn face_normal_raw(&self, face: &[usize; 3]) -> Vec3 {
        let a = self.vertices[face[0]];
        let b = self.vertices[face[1]];
        let c = self.vertices[face[2]];
        (b - a).cross(&(c - a)).normalize()
    }
}

/// Project all vertices onto an axis and return (min, max).
fn project_hull_on_axis(verts: &[Vec3], axis: Vec3) -> (f64, f64) {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for v in verts {
        let d = v.dot(&axis);
        if d < mn {
            mn = d;
        }
        if d > mx {
            mx = d;
        }
    }
    (mn, mx)
}

/// Merge two convex hulls into one by computing the hull of the union of vertices.
pub fn merge_hulls(a: &ConvexHull3DVec, b: &ConvexHull3DVec) -> Option<ConvexHull3DVec> {
    let mut all_points: Vec<Vec3> = a.vertices.clone();
    all_points.extend_from_slice(&b.vertices);
    ConvexHull3DVec::build(&all_points)
}

/// Compute convex hull and return both hull and the indices (into original
/// point cloud) of vertices on the hull.
pub fn build_with_indices(points: &[Vec3]) -> Option<(ConvexHull3DVec, Vec<usize>)> {
    let hull = ConvexHull3DVec::build(points)?;
    // Find which original point corresponds to each hull vertex
    let mut indices = Vec::with_capacity(hull.vertices.len());
    for hv in &hull.vertices {
        let idx = points
            .iter()
            .position(|p| (*p - *hv).norm() < 1e-12)
            .unwrap_or(0);
        indices.push(idx);
    }
    Some((hull, indices))
}

/// Check if a point set is convex (i.e., all points lie on the hull).
pub fn is_point_set_convex(points: &[Vec3]) -> bool {
    if points.len() < 4 {
        return true;
    }
    match ConvexHull3DVec::build(points) {
        Some(hull) => hull.n_vertices() == points.len(),
        None => true, // degenerate
    }
}

// ---------------------------------------------------------------------------
// Internal QuickHull helpers
// ---------------------------------------------------------------------------

/// Signed distance from point `p` to the plane `(normal, offset)`.
fn point_plane_distance(p: Vec3, normal: Vec3, offset: f64) -> f64 {
    normal.dot(&p) - offset
}

// ---------------------------------------------------------------------------
// Incremental convex hull
// ---------------------------------------------------------------------------

/// Incremental convex hull that can accept new points one at a time.
///
/// Internally maintains the same triangulated hull structure as `ConvexHull3DVec`.
pub struct IncrementalConvexHull {
    /// All points ever inserted.
    pub all_points: Vec<Vec3>,
    /// Current hull faces.
    pub faces: Vec<[usize; 3]>,
    /// Current hull vertex indices (into `all_points`).
    pub hull_verts: Vec<usize>,
}

impl IncrementalConvexHull {
    /// Build an incremental hull from a slice of initial points.
    pub fn build(points: &[Vec3]) -> Self {
        let mut hull = Self {
            all_points: Vec::new(),
            faces: Vec::new(),
            hull_verts: Vec::new(),
        };
        for &p in points {
            hull.insert(p);
        }
        hull
    }

    /// Insert a new point, updating the hull if necessary.
    pub fn insert(&mut self, p: Vec3) {
        self.all_points.push(p);
        // Rebuild from scratch (simple but correct)
        if let Some(ch) = ConvexHull3DVec::build(&self.all_points) {
            // Remap hull_verts to current all_points indices
            self.hull_verts = (0..self.all_points.len())
                .filter(|&i| {
                    ch.vertices
                        .iter()
                        .any(|&v| (v - self.all_points[i]).norm() < 1e-10)
                })
                .collect();
            // Map from ch.vertices index back to all_points index
            let vert_to_orig: Vec<usize> = ch
                .vertices
                .iter()
                .map(|hv| {
                    self.all_points
                        .iter()
                        .enumerate()
                        .find(|&(_, ap)| (*ap - *hv).norm() < 1e-10)
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect();
            self.faces = ch
                .faces
                .iter()
                .map(|f| [vert_to_orig[f[0]], vert_to_orig[f[1]], vert_to_orig[f[2]]])
                .collect();
        }
    }

    /// Number of vertices on the current hull.
    pub fn n_vertices(&self) -> usize {
        self.hull_verts.len()
    }

    /// Number of triangular faces.
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Number of unique edges.
    pub fn n_edges(&self) -> usize {
        let mut edges = std::collections::BTreeSet::new();
        for f in &self.faces {
            for k in 0..3 {
                let a = f[k];
                let b = f[(k + 1) % 3];
                edges.insert((a.min(b), a.max(b)));
            }
        }
        edges.len()
    }

    /// Euler characteristic V - E + F = 2.
    pub fn euler_characteristic(&self) -> i64 {
        self.n_vertices() as i64 - self.n_edges() as i64 + self.n_faces() as i64
    }

    /// Volume via signed tetrahedral decomposition.
    pub fn volume(&self) -> f64 {
        if self.faces.is_empty() {
            return 0.0;
        }
        let c = {
            let sum: Vec3 = self.hull_verts.iter().map(|&i| self.all_points[i]).sum();
            sum / self.hull_verts.len() as f64
        };
        let mut vol = 0.0f64;
        for f in &self.faces {
            let a = self.all_points[f[0]] - c;
            let b = self.all_points[f[1]] - c;
            let cc = self.all_points[f[2]] - c;
            vol += a.dot(&b.cross(&cc));
        }
        (vol / 6.0).abs()
    }
}

// ---------------------------------------------------------------------------
// Conflict graph
// ---------------------------------------------------------------------------

/// A conflict graph recording which faces can "see" which points outside the hull.
///
/// This is a query structure for analysing a finished hull against a point set:
/// every face lists all the points in front of it, and every point all the faces
/// it sees.  [`ConvexHull3DVec::build`] does not use it; its construction keeps
/// a single outside set per face (each point belongs to the face it is farthest
/// in front of), which is cheaper than the full relation recorded here.
pub struct ConflictGraph {
    /// Number of hull faces.
    pub face_count: usize,
    /// For each face, list of point indices visible from that face.
    pub face_to_points: Vec<Vec<usize>>,
    /// For each point, list of face indices it can see.
    pub point_to_faces: Vec<Vec<usize>>,
}

impl ConflictGraph {
    /// Build a conflict graph for `hull` against a set of `points`.
    pub fn new(hull: &ConvexHull3DVec, points: &[Vec3]) -> Self {
        let n_faces = hull.n_faces();
        let n_points = points.len();
        let mut face_to_points = vec![Vec::new(); n_faces];
        let mut point_to_faces = vec![Vec::new(); n_points];

        for (fi, face) in hull.faces.iter().enumerate() {
            let (n, off) = {
                let a = hull.vertices[face[0]];
                let b = hull.vertices[face[1]];
                let c = hull.vertices[face[2]];
                let nn = (b - a).cross(&(c - a));
                let off = nn.dot(&a);
                (nn, off)
            };
            for (pi, &p) in points.iter().enumerate() {
                if nn_point_plane_distance(p, n, off) > 1e-10 {
                    face_to_points[fi].push(pi);
                    point_to_faces[pi].push(fi);
                }
            }
        }

        Self {
            face_count: n_faces,
            face_to_points,
            point_to_faces,
        }
    }

    /// Returns the faces visible from point `pi`.
    pub fn faces_for_point(&self, pi: usize) -> &[usize] {
        if pi < self.point_to_faces.len() {
            &self.point_to_faces[pi]
        } else {
            &[]
        }
    }

    /// Returns the points visible from face `fi`.
    pub fn points_for_face(&self, fi: usize) -> &[usize] {
        if fi < self.face_to_points.len() {
            &self.face_to_points[fi]
        } else {
            &[]
        }
    }
}

fn nn_point_plane_distance(p: Vec3, n: Vec3, offset: f64) -> f64 {
    n.dot(&p) - offset
}

// ---------------------------------------------------------------------------
// Hull inertia tensor
// ---------------------------------------------------------------------------

/// Compute the inertia tensor of a convex hull treated as a uniform solid.
///
/// The tensor is taken about the centre of mass of the solid and expressed in the
/// world axes. The hull is split into the tetrahedra spanned by each face and a
/// reference point; the volume-weighted centroid of those tetrahedra is the
/// centre of mass. About that point the second moments
/// `S_ij = ∫ x_i x_j dV` of a tetrahedron with one vertex at the origin and the
/// others at `a`, `b`, `c` are integrated exactly as
/// `V / 20 · (a_i a_j + b_i b_j + c_i c_j + s_i s_j)` with `s = a + b + c`, and the
/// inertia tensor is `ρ (tr(S) · 1 − S)` with the uniform density
/// `ρ = mass / volume`.
///
/// Faces that wind towards the inside of the hull are flipped before the
/// integration, so the result does not depend on the winding of the faces.
///
/// # Arguments
/// * `hull`   – convex hull structure.
/// * `mass`   – total mass of the solid.
///
/// # Returns
/// The 3×3 inertia tensor as `[[f64;3\];3]`; all zeros for a hull without volume.
pub fn hull_inertia_tensor(hull: &ConvexHull3DVec, mass: f64) -> [[f64; 3]; 3] {
    const NO_INERTIA: [[f64; 3]; 3] = [[0.0; 3]; 3];
    let Some(&origin) = hull.vertices.first() else {
        return NO_INERTIA;
    };

    // Positions relative to the first vertex keep the arithmetic well scaled for
    // hulls far from the origin. The vertex mean is strictly interior for a hull
    // with volume and tells inward winding from outward winding.
    let vertex_sum: Vec3 = hull.vertices.iter().copied().sum();
    let interior = vertex_sum / hull.vertices.len() as f64 - origin;
    let corners: Vec<[Vec3; 3]> = hull
        .faces
        .iter()
        .map(|face| {
            let a = hull.vertices[face[0]] - origin;
            let mut b = hull.vertices[face[1]] - origin;
            let mut c = hull.vertices[face[2]] - origin;
            if (b - a).cross(&(c - a)).dot(&(interior - a)) > 0.0 {
                std::mem::swap(&mut b, &mut c);
            }
            [a, b, c]
        })
        .collect();

    // Volume and centre of mass from the tetrahedra (origin, a, b, c).
    let mut volume = 0.0f64;
    let mut first_moment = Vec3::zeros();
    for [a, b, c] in &corners {
        let tet_volume = a.dot(&b.cross(c)) / 6.0;
        volume += tet_volume;
        first_moment += (a + b + c) * (tet_volume / 4.0);
    }
    let extent = hull
        .vertices
        .iter()
        .map(|v| (v - origin).norm())
        .fold(0.0f64, f64::max);
    // Below round-off level the centre of mass is meaningless; non-finite input
    // is rejected as well.
    if !volume.is_finite() || volume <= 16.0 * f64::EPSILON * extent.powi(3) {
        return NO_INERTIA;
    }
    let center_of_mass = first_moment / volume;

    // Second moments about the centre of mass, tetrahedron by tetrahedron.
    let mut second_moment = Mat3::zeros();
    for [a, b, c] in &corners {
        let (a, b, c) = (a - center_of_mass, b - center_of_mass, c - center_of_mass);
        let weight = a.dot(&b.cross(&c)) / 120.0;
        let s = a + b + c;
        second_moment +=
            (a * a.transpose() + b * b.transpose() + c * c.transpose() + s * s.transpose())
                * weight;
    }

    let density = mass / volume;
    let inertia = (Mat3::identity() * second_moment.trace() - second_moment) * density;
    std::array::from_fn(|i| std::array::from_fn(|j| inertia[(i, j)]))
}

// ---------------------------------------------------------------------------
// Closest surface point
// ---------------------------------------------------------------------------

impl ConvexHull3DVec {
    /// Find the closest point on the hull surface to query point `q`.
    ///
    /// Iterates over all triangular faces and returns the point that minimizes
    /// distance to `q`.
    pub fn closest_surface_point(&self, q: Vec3) -> Vec3 {
        let mut best_dist = f64::INFINITY;
        let mut best_pt = q;

        for face in &self.faces {
            let a = self.vertices[face[0]];
            let b = self.vertices[face[1]];
            let c = self.vertices[face[2]];
            let cp = closest_point_on_tri(q, a, b, c);
            let d = (cp - q).norm();
            if d < best_dist {
                best_dist = d;
                best_pt = cp;
            }
        }
        best_pt
    }
}

/// Closest point on triangle (a, b, c) to query point p.
fn closest_point_on_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod robustness_tests;

#[cfg(test)]
mod inertia_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn tetrahedron_points() -> Vec<Vec3> {
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
    }

    fn unit_cube_points() -> Vec<Vec3> {
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ]
    }

    #[test]
    fn test_hull_tetrahedron() {
        let pts = tetrahedron_points();
        let hull = ConvexHull3DVec::build(&pts).expect("should build");
        assert_eq!(
            hull.n_faces(),
            4,
            "tetrahedron should have 4 faces, got {}",
            hull.n_faces()
        );
    }

    #[test]
    fn test_hull_cube() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("should build cube hull");
        assert_eq!(
            hull.n_faces(),
            12,
            "unit cube should have 12 triangular faces, got {}",
            hull.n_faces()
        );
    }

    #[test]
    fn test_hull_sphere_points() {
        let n = 50usize;
        let pts: Vec<Vec3> = (0..n)
            .map(|i| {
                let theta = std::f64::consts::PI * (i as f64) / (n as f64 / 2.0);
                let phi = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                )
            })
            .collect();

        let hull = ConvexHull3DVec::build(&pts).expect("should build sphere hull");
        for v in &hull.vertices {
            let r = v.norm();
            assert!(
                (r - 1.0).abs() < 1e-9,
                "hull vertex not on unit sphere: norm={}",
                r
            );
        }
        assert!(hull.n_faces() >= 4, "too few faces: {}", hull.n_faces());
    }

    #[test]
    fn test_hull_support_x() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let sp = hull.support(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            (sp.x - 1.0).abs() < 1e-9,
            "support in +X should have x=1, got x={}",
            sp.x
        );
    }

    #[test]
    fn test_hull_contains_interior() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let centroid: Vec3 = hull.vertices.iter().copied().sum::<Vec3>() / hull.n_vertices() as f64;
        assert!(
            hull.contains_point(centroid),
            "centroid should be inside the hull"
        );
    }

    #[test]
    fn test_hull_exterior_point() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let outside = Vec3::new(5.0, 5.0, 5.0);
        assert!(
            !hull.contains_point(outside),
            "point far outside should not be contained"
        );
    }

    #[test]
    fn test_hull_volume_cube() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let vol = hull.volume();
        assert!(
            (vol - 1.0).abs() < 0.05,
            "unit cube volume should be ~1.0, got {}",
            vol
        );
    }

    #[test]
    fn test_hull_degenerate_coplanar() {
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        match ConvexHull3DVec::build(&pts) {
            None => {}
            Some(hull) => {
                let vol = hull.volume();
                assert!(
                    vol < 1e-6,
                    "coplanar hull should have ~0 volume, got {}",
                    vol
                );
            }
        }
    }

    // --- New tests ---

    #[test]
    fn test_hull_surface_area_cube() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let sa = hull.surface_area();
        // Unit cube surface area = 6 faces * 1.0 = 6.0
        assert!(
            (sa - 6.0).abs() < 0.1,
            "unit cube SA should be ~6.0, got {}",
            sa
        );
    }

    #[test]
    fn test_hull_edges_cube() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let edges = hull.edges();
        // A cube has 12 edges; triangulated cube has 18 edges (12 original + 6 diagonals)
        assert!(
            edges.len() >= 12,
            "cube should have >= 12 edges, got {}",
            edges.len()
        );
    }

    #[test]
    fn test_hull_euler_characteristic() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let chi = hull.euler_characteristic();
        assert_eq!(
            chi, 2,
            "Euler characteristic should be 2 for convex polyhedron, got {}",
            chi
        );
    }

    #[test]
    fn test_hull_euler_characteristic_tetrahedron() {
        let pts = tetrahedron_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        assert_eq!(hull.euler_characteristic(), 2);
    }

    #[test]
    fn test_hull_diameter_cube() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let d = hull.diameter();
        // Diagonal of unit cube = sqrt(3)
        assert!((d - 3.0_f64.sqrt()).abs() < 1e-9, "diameter={}", d);
    }

    #[test]
    fn test_hull_aabb_cube() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let (mn, mx) = hull.aabb();
        for k in 0..3 {
            assert!(mn[k] >= -1e-9, "min[{}] = {}", k, mn[k]);
            assert!((mx[k] - 1.0).abs() < 1e-9, "max[{}] = {}", k, mx[k]);
        }
    }

    #[test]
    fn test_hull_signed_distance_inside() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let sd = hull.signed_distance(Vec3::new(0.5, 0.5, 0.5));
        assert!(
            sd < 1e-9,
            "centroid should have non-positive signed distance, got {}",
            sd
        );
    }

    #[test]
    fn test_hull_signed_distance_outside() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let sd = hull.signed_distance(Vec3::new(5.0, 5.0, 5.0));
        assert!(
            sd > 0.0,
            "point outside should have positive signed distance, got {}",
            sd
        );
    }

    #[test]
    fn test_merge_hulls() {
        // Use two cube-like hulls that share some space to ensure
        // the merged point set is non-degenerate in all axes.
        let pts_a = unit_cube_points();
        let pts_b: Vec<Vec3> = pts_a.iter().map(|p| p + Vec3::new(0.5, 0.0, 0.0)).collect();
        let hull_a = ConvexHull3DVec::build(&pts_a).expect("build A");
        let hull_b = ConvexHull3DVec::build(&pts_b).expect("build B");
        let merged = merge_hulls(&hull_a, &hull_b).expect("merge");
        // Merged hull should contain all vertices of both inputs
        assert!(merged.n_vertices() >= hull_a.n_vertices());
        assert!(merged.volume() >= hull_a.volume() - 1e-6);
    }

    #[test]
    fn test_build_with_indices() {
        let pts = tetrahedron_points();
        let (hull, indices) = build_with_indices(&pts).expect("build");
        assert_eq!(hull.n_vertices(), 4);
        assert_eq!(indices.len(), 4);
        // All indices should be in [0, 3]
        for &idx in &indices {
            assert!(idx < 4, "index {} out of range", idx);
        }
    }

    #[test]
    fn test_is_point_set_convex_cube() {
        let pts = unit_cube_points();
        assert!(is_point_set_convex(&pts));
    }

    #[test]
    fn test_is_point_set_convex_with_interior() {
        let mut pts = unit_cube_points();
        pts.push(Vec3::new(0.5, 0.5, 0.5)); // interior point
        assert!(!is_point_set_convex(&pts));
    }

    #[test]
    fn test_face_area_tetrahedron() {
        let pts = tetrahedron_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        for fi in 0..hull.n_faces() {
            let area = hull.face_area(fi);
            assert!(area > 0.0, "face {} has zero area", fi);
        }
    }

    #[test]
    fn test_vertex_iter() {
        let pts = tetrahedron_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let verts: Vec<_> = hull.vertex_iter().collect();
        assert_eq!(verts.len(), 4);
    }

    #[test]
    fn test_face_iter() {
        let pts = tetrahedron_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let faces: Vec<_> = hull.face_iter().collect();
        assert_eq!(faces.len(), 4);
    }

    #[test]
    fn test_overlaps_sat_faces_overlapping() {
        let pts_a = unit_cube_points();
        let pts_b: Vec<Vec3> = pts_a.iter().map(|p| p + Vec3::new(0.5, 0.0, 0.0)).collect();
        let hull_a = ConvexHull3DVec::build(&pts_a).expect("build A");
        let hull_b = ConvexHull3DVec::build(&pts_b).expect("build B");
        assert!(hull_a.overlaps_sat_faces(&hull_b));
    }

    #[test]
    fn test_overlaps_sat_faces_separated() {
        let pts_a = unit_cube_points();
        let pts_b: Vec<Vec3> = pts_a
            .iter()
            .map(|p| p + Vec3::new(10.0, 0.0, 0.0))
            .collect();
        let hull_a = ConvexHull3DVec::build(&pts_a).expect("build A");
        let hull_b = ConvexHull3DVec::build(&pts_b).expect("build B");
        assert!(!hull_a.overlaps_sat_faces(&hull_b));
    }

    #[test]
    fn test_n_edges_tetrahedron() {
        let pts = tetrahedron_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        assert_eq!(
            hull.n_edges(),
            6,
            "tetrahedron has 6 edges, got {}",
            hull.n_edges()
        );
    }

    #[test]
    fn test_hull_volume_tetrahedron() {
        let pts = tetrahedron_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let vol = hull.volume();
        // Volume of unit tetrahedron = 1/6
        assert!(
            (vol - 1.0 / 6.0).abs() < 0.02,
            "tetrahedron volume = {}",
            vol
        );
    }

    // ── Incremental hull tests ──────────────────────────────────────────────

    #[test]
    fn test_incremental_hull_cube() {
        let pts = unit_cube_points();
        let hull = IncrementalConvexHull::build(&pts);
        assert!(hull.n_vertices() == 8, "cube has 8 vertices");
        assert!(hull.n_faces() >= 12, "cube has >= 12 triangular faces");
    }

    #[test]
    fn test_incremental_hull_volume() {
        let pts = unit_cube_points();
        let hull = IncrementalConvexHull::build(&pts);
        let vol = hull.volume();
        assert!((vol - 1.0).abs() < 0.1, "volume={}", vol);
    }

    #[test]
    fn test_incremental_hull_euler() {
        let pts = unit_cube_points();
        let hull = IncrementalConvexHull::build(&pts);
        let chi = hull.euler_characteristic();
        assert_eq!(chi, 2, "Euler characteristic={}", chi);
    }

    // ── Inertia tensor tests ──────────────────────────────────────────────────

    #[test]
    fn test_hull_inertia_tensor_diagonal_symmetry() {
        // Symmetric hull: cube → off-diagonal inertia should be ~0
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let inertia = hull_inertia_tensor(&hull, 1.0);
        // Off-diagonals should be small for a symmetric body
        assert!(inertia[0][1].abs() < 0.1, "I_xy={}", inertia[0][1]);
        assert!(inertia[0][2].abs() < 0.1, "I_xz={}", inertia[0][2]);
        assert!(inertia[1][2].abs() < 0.1, "I_yz={}", inertia[1][2]);
    }

    #[test]
    fn test_hull_inertia_tensor_positive_diagonal() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let inertia = hull_inertia_tensor(&hull, 1.0);
        assert!(inertia[0][0] > 0.0, "I_xx={}", inertia[0][0]);
        assert!(inertia[1][1] > 0.0, "I_yy={}", inertia[1][1]);
        assert!(inertia[2][2] > 0.0, "I_zz={}", inertia[2][2]);
    }

    // ── Conflict graph tests ─────────────────────────────────────────────────

    #[test]
    fn test_conflict_graph_build() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let cg = ConflictGraph::new(&hull, &pts);
        // For the unit cube (all on hull), no extra point should conflict any face
        assert_eq!(cg.face_count, hull.n_faces());
    }

    // ── Closest point on hull tests ───────────────────────────────────────────

    #[test]
    fn test_closest_point_on_hull_inside() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let q = Vec3::new(0.5, 0.5, 0.5); // inside cube
        let cp = hull.closest_surface_point(q);
        // Closest surface point should be at distance 0.5 from center
        let dist = (cp - q).norm();
        assert!(dist < 0.6, "dist to surface from center={}", dist);
    }

    #[test]
    fn test_closest_point_on_hull_outside() {
        let pts = unit_cube_points();
        let hull = ConvexHull3DVec::build(&pts).expect("build");
        let q = Vec3::new(2.0, 0.5, 0.5); // outside cube
        let cp = hull.closest_surface_point(q);
        // Closest point should be on the x=1 face
        assert!((cp.x - 1.0).abs() < 0.1, "cp.x={}", cp.x);
    }
}

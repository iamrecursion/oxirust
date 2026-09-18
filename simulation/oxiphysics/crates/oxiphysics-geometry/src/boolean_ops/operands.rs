//! Plumbing shared by the mesh boolean front-ends of this module.
//!
//! [`MeshBoolean`](super::MeshBoolean), [`RobustMeshBoolean`](super::RobustMeshBoolean),
//! [`ManifoldBoolean`](super::ManifoldBoolean) and
//! [`CsgWithMaterials`](super::CsgWithMaterials) all run on the exact
//! surface-arrangement pipeline of [`crate::mesh_boolean`]: the two surfaces
//! are intersected, every cut triangle is re-triangulated along the
//! intersection curves, and each resulting piece is kept or dropped by an exact
//! inside/outside decision.  This module converts between the
//! [`TriangleMesh`] used by the front-ends and the [`SimpleMesh`] of the
//! pipeline, validates operands, and provides the tolerance handling and
//! provenance tracking of the robust and the material-tracking variants.

use std::collections::HashMap;

use super::functions::{arr_to_vec3, check_manifold, cross3, dot3, len3, sub3, vec3_to_arr};
use super::types::BooleanOp;
use crate::error::{Error, Result};
use crate::mesh_boolean::{MeshBooleanOp, SimpleMesh, remove_unused_vertices, weld_vertices};
use crate::triangle_mesh::TriangleMesh;

/// Relative tolerance (to the largest coordinate) within which a result
/// triangle counts as lying in an input triangle.  The arrangement snaps to
/// about `4e-13` times the coordinate scale, so this leaves a wide margin
/// without reaching the scale of genuinely different geometry.
const CONTAINMENT_REL: f64 = 1e-10;

/// Largest lattice index (`|coordinate| / epsilon`) that is still rounded.
/// Beyond about `2^52` the lattice is finer than the coordinate's own precision
/// and rounding would only add noise (or overflow for tiny `epsilon`).
const MAX_LATTICE_INDEX: f64 = 4.5e15;

/// The boolean operation of the arrangement pipeline matching `op`.
pub(super) fn mesh_op(op: BooleanOp) -> MeshBooleanOp {
    match op {
        BooleanOp::Union => MeshBooleanOp::Union,
        BooleanOp::Intersection => MeshBooleanOp::Intersection,
        BooleanOp::Difference => MeshBooleanOp::Difference,
    }
}

/// Validate one operand and convert it to the pipeline's mesh type.
///
/// `name` identifies the operand in error messages.
///
/// # Errors
///
/// * [`Error::TooFewPoints`] when the mesh has no vertices.
/// * [`Error::InvalidMesh`] when it has no triangles or a non-finite vertex.
/// * [`Error::IndexOutOfBounds`] when a triangle references a missing vertex.
pub(super) fn to_simple_mesh(name: &str, mesh: &TriangleMesh) -> Result<SimpleMesh> {
    if mesh.vertices.is_empty() {
        return Err(Error::too_few_points(1, 0));
    }
    if mesh.indices.is_empty() {
        return Err(Error::invalid_mesh(format!("{name} has no triangles")));
    }
    let n_vertices = mesh.vertices.len();
    if let Some(&bad) = mesh.indices.iter().flatten().find(|&&i| i >= n_vertices) {
        return Err(Error::index_out_of_bounds(bad, n_vertices));
    }
    if let Some(i) = mesh
        .vertices
        .iter()
        .position(|v| !(v.x.is_finite() && v.y.is_finite() && v.z.is_finite()))
    {
        return Err(Error::invalid_mesh(format!(
            "{name} vertex {i} has a non-finite coordinate"
        )));
    }
    Ok(SimpleMesh::from_data(
        mesh.vertices.iter().map(|&v| vec3_to_arr(v)).collect(),
        mesh.indices.clone(),
    ))
}

/// Convert a pipeline result back to a [`TriangleMesh`].
pub(super) fn from_simple_mesh(mesh: &SimpleMesh) -> TriangleMesh {
    TriangleMesh::new(
        mesh.vertices.iter().map(|&v| arr_to_vec3(v)).collect(),
        mesh.triangles.clone(),
    )
}

/// Require a closed, consistently oriented 2-manifold triangle list.
///
/// `name` identifies the operand in the error message.
///
/// # Errors
///
/// [`Error::InvalidMesh`] when an edge is not shared by exactly two triangles
/// or two triangles traverse their shared edge in the same direction.
pub(super) fn require_closed_oriented(name: &str, tris: &[[usize; 3]]) -> Result<()> {
    let check = check_manifold(tris);
    if !check.is_manifold {
        return Err(Error::invalid_mesh(format!(
            "{name} is non-manifold: {} boundary edges, {} non-manifold edges",
            check.n_boundary_edges, check.n_non_manifold_edges
        )));
    }
    let conflicts = count_orientation_conflicts(tris);
    if conflicts > 0 {
        return Err(Error::invalid_mesh(format!(
            "{name} is not consistently oriented: {conflicts} edges are traversed twice in the same direction"
        )));
    }
    Ok(())
}

/// Number of edges that are used by exactly two triangles *in the same
/// direction*, i.e. the places where a closed surface is not consistently
/// oriented.  Boundary edges and edges shared by more than two triangles are
/// reported by [`check_manifold`] instead.
pub(super) fn count_orientation_conflicts(tris: &[[usize; 3]]) -> usize {
    // (traversals low -> high, traversals high -> low) of every undirected edge.
    let mut uses: HashMap<(usize, usize), (u32, u32)> = HashMap::new();
    for tri in tris {
        for k in 0..3 {
            let (from, to) = (tri[k], tri[(k + 1) % 3]);
            let entry = uses.entry((from.min(to), from.max(to))).or_insert((0, 0));
            if from <= to {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }
    }
    uses.values()
        .filter(|&&(forward, backward)| forward + backward == 2 && forward != backward)
        .count()
}

/// Round `p` to the lattice `epsilon * Z^3` (snap rounding).
///
/// Coordinates that are too large for the lattice (see [`MAX_LATTICE_INDEX`])
/// are left unchanged.
fn snap_to_lattice(p: [f64; 3], epsilon: f64) -> [f64; 3] {
    p.map(|c| {
        let index = c / epsilon;
        if index.is_finite() && index.abs() < MAX_LATTICE_INDEX {
            index.round() * epsilon
        } else {
            c
        }
    })
}

/// Clean the vertices of both operands at the tolerance `epsilon`.
///
/// 1. **Merge.** The vertices of the two meshes are welded together (earlier
///    vertices win, those of `mesh_a` first): a vertex that lies within
///    `epsilon` of an earlier vertex, of either operand, becomes exactly that
///    vertex.  Nearly coincident corners, and the edges and faces spanned by
///    them, thereby turn into exactly coincident ones.
/// 2. **Round.** Every coordinate is rounded to a multiple of `epsilon`.
///    Faces that are nearly coplanar and aligned with a coordinate plane get
///    exactly the same coordinate on all their vertices, so the exact pipeline
///    resolves them as coincident faces instead of as near-degenerate crossings
///    of almost parallel planes (which leave seam artifacts).  Tilted faces
///    that coincide only up to `epsilon` are not made coplanar by this.
///
/// Every vertex moves by less than `2 * epsilon`.  Triangles that collapse are
/// dropped later by the arrangement itself.  With `epsilon <= 0` the operands
/// are returned unchanged.
pub(super) fn snap_operands(
    mesh_a: &SimpleMesh,
    mesh_b: &SimpleMesh,
    epsilon: f64,
) -> (SimpleMesh, SimpleMesh) {
    if epsilon <= 0.0 {
        return (mesh_a.clone(), mesh_b.clone());
    }
    let offset = mesh_a.vertices.len();
    let joint = SimpleMesh::from_data(
        mesh_a
            .vertices
            .iter()
            .chain(mesh_b.vertices.iter())
            .copied()
            .collect(),
        mesh_a
            .triangles
            .iter()
            .copied()
            .chain(
                mesh_b
                    .triangles
                    .iter()
                    .map(|tri| tri.map(|i| i.saturating_add(offset))),
            )
            .collect(),
    );
    let welded = weld_vertices(&joint, epsilon);
    let vertices: Vec<[f64; 3]> = welded
        .vertices
        .iter()
        .map(|&p| snap_to_lattice(p, epsilon))
        .collect();
    let (tris_a, tris_b) = welded.triangles.split_at(mesh_a.triangles.len());
    (
        SimpleMesh::from_data(vertices.clone(), tris_a.to_vec()),
        SimpleMesh::from_data(vertices, tris_b.to_vec()),
    )
}

/// Clean a boolean result at the tolerance `epsilon`: merge vertices closer
/// than `epsilon`, drop the triangles that collapse, cancel back-to-back
/// duplicate pairs and compact the vertex list.
///
/// Slivers thinner than `epsilon` (steps between nearly coplanar faces, for
/// example) collapse in the process, and the surface stays conforming because
/// the merge is one consistent map on the vertices.  With `epsilon <= 0` the
/// mesh is returned unchanged.
pub(super) fn tidy_result(mesh: SimpleMesh, epsilon: f64) -> SimpleMesh {
    if epsilon <= 0.0 || mesh.triangles.is_empty() {
        return mesh;
    }
    let mut welded = weld_vertices(&mesh, epsilon);
    welded
        .triangles
        .retain(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
    cancel_opposite_pairs(&mut welded.triangles);
    remove_unused_vertices(&mut welded);
    welded
}

/// Sorted vertex triple of a triangle (distinct indices) and whether the
/// triangle is an even (cyclic) permutation of it, which tells its orientation.
fn orientation_key(tri: &[usize; 3]) -> ([usize; 3], bool) {
    let mut sorted = *tri;
    sorted.sort_unstable();
    let even =
        *tri == sorted || [tri[1], tri[2], tri[0]] == sorted || [tri[2], tri[0], tri[1]] == sorted;
    (sorted, even)
}

/// Remove pairs of triangles with the same vertices and opposite orientation.
///
/// As 2-chains they sum to zero, so dropping both leaves the boundary of the
/// solid unchanged; such pairs appear when merging vertices collapses two
/// slivers onto each other.
fn cancel_opposite_pairs(tris: &mut Vec<[usize; 3]>) {
    let mut open: HashMap<([usize; 3], bool), Vec<usize>> = HashMap::new();
    let mut remove = vec![false; tris.len()];
    for (i, tri) in tris.iter().enumerate() {
        let (key, even) = orientation_key(tri);
        if let Some(j) = open.get_mut(&(key, !even)).and_then(Vec::pop) {
            remove[i] = true;
            remove[j] = true;
        } else {
            open.entry((key, even)).or_default().push(i);
        }
    }
    let mut flags = remove.into_iter();
    tris.retain(|_| !flags.next().unwrap_or(false));
}

/// A non-degenerate input triangle prepared for containment queries.
struct Face {
    corners: [[f64; 3]; 3],
    normal: [f64; 3],
    lo: [f64; 3],
    hi: [f64; 3],
}

impl Face {
    fn new(corners: [[f64; 3]; 3]) -> Option<Self> {
        let n = cross3(sub3(corners[1], corners[0]), sub3(corners[2], corners[0]));
        let len = len3(n);
        if !(len.is_finite() && len > 0.0) {
            return None;
        }
        let mut lo = corners[0];
        let mut hi = corners[0];
        for p in &corners[1..] {
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        Some(Self {
            corners,
            normal: [n[0] / len, n[1] / len, n[2] / len],
            lo,
            hi,
        })
    }

    /// Whether the triangle `tri` (with the given centroid) lies in this face:
    /// all its corners are within `tol` of the face's plane and its centroid is
    /// inside the face, up to `tol`.
    fn contains(&self, tri: &[[f64; 3]; 3], centroid: [f64; 3], tol: f64) -> bool {
        if (0..3).any(|d| centroid[d] < self.lo[d] - tol || centroid[d] > self.hi[d] + tol) {
            return false;
        }
        if tri
            .iter()
            .any(|&p| dot3(self.normal, sub3(p, self.corners[0])).abs() > tol)
        {
            return false;
        }
        (0..3).all(|k| {
            let edge = sub3(self.corners[(k + 1) % 3], self.corners[k]);
            let inward = cross3(self.normal, edge);
            dot3(inward, sub3(centroid, self.corners[k])) >= -tol * len3(edge)
        })
    }
}

/// For every triangle of `result`: does it lie in a triangle of `first`?
///
/// Every triangle produced by the arrangement pipeline is a piece of exactly
/// one input triangle, and a piece of the second operand that lies in a face of
/// the first one is a coincident face, which the pipeline resolves by keeping
/// the first operand's copy.  A result triangle therefore comes from the first
/// operand exactly when it lies in one of its triangles, and from the second
/// operand otherwise.
pub(super) fn owned_by_first(result: &SimpleMesh, first: &SimpleMesh) -> Vec<bool> {
    let n_first = first.vertices.len();
    let faces: Vec<Face> = first
        .triangles
        .iter()
        .filter(|t| t.iter().all(|&i| i < n_first))
        .filter_map(|t| Face::new(t.map(|i| first.vertices[i])))
        .collect();
    let scale = result
        .vertices
        .iter()
        .chain(first.vertices.iter())
        .flat_map(|v| v.iter())
        .fold(0.0_f64, |m, c| m.max(c.abs()));
    let tol = CONTAINMENT_REL * scale.max(f64::MIN_POSITIVE);
    let n_result = result.vertices.len();
    result
        .triangles
        .iter()
        .map(|t| {
            if t.iter().any(|&i| i >= n_result) {
                return false;
            }
            let corners = t.map(|i| result.vertices[i]);
            let centroid = [
                (corners[0][0] + corners[1][0] + corners[2][0]) / 3.0,
                (corners[0][1] + corners[1][1] + corners[2][1]) / 3.0,
                (corners[0][2] + corners[1][2] + corners[2][2]) / 3.0,
            ];
            faces
                .iter()
                .any(|face| face.contains(&corners, centroid, tol))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(vertices: Vec<[f64; 3]>, triangles: Vec<[usize; 3]>) -> SimpleMesh {
        SimpleMesh::from_data(vertices, triangles)
    }

    #[test]
    fn snap_to_lattice_rounds_to_multiples_and_leaves_huge_indices_alone() {
        let snapped = snap_to_lattice([0.3000000004, -0.7000000004, 1.0], 1e-3);
        for (got, want) in snapped.iter().zip([0.3, -0.7, 1.0]) {
            assert!((got - want).abs() < 1e-12, "{snapped:?}");
        }
        // 1e10 / 1e-9 exceeds the lattice-index limit: the coordinate is kept.
        let kept = snap_to_lattice([1.0e10 + 0.123, 0.0, 5.0], 1e-9);
        assert_eq!(kept[0], 1.0e10 + 0.123);
        // A tiny epsilon must not overflow to infinity.
        let tiny = snap_to_lattice([1.0e300, 1.0, 2.0], 1e-300);
        assert_eq!(tiny[0], 1.0e300);
        assert!(tiny.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn snap_operands_merges_across_operands_and_rounds() {
        let a = mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2]],
        );
        // b's first vertex is within 1e-12 of a's second one; its z is only
        // nearly 0 and rounds onto the lattice together with the others.
        let b = mesh(
            vec![
                [1.0 + 1e-12, 1e-12, 4e-12],
                [2.0, 0.0, 3e-12],
                [1.0, 1.0, 2e-12],
            ],
            vec![[0, 1, 2]],
        );
        let (sa, sb) = snap_operands(&a, &b, 1e-9);
        // The shared corner is exactly one vertex of both operands.
        let shared = sb.triangles[0][0];
        assert_eq!(sb.vertices[shared], sa.vertices[sa.triangles[0][1]]);
        assert_eq!(shared, sa.triangles[0][1]);
        // Every coordinate is now a multiple of the lattice pitch, so the almost
        // coplanar z values of b are exactly zero.
        for v in sa.vertices.iter().chain(sb.vertices.iter()) {
            assert_eq!(v[2], 0.0, "z not snapped: {v:?}");
        }
        // epsilon = 0 changes nothing.
        let (ua, ub) = snap_operands(&a, &b, 0.0);
        assert_eq!(ua.vertices, a.vertices);
        assert_eq!(ub.vertices, b.vertices);
        assert_eq!(ub.triangles, b.triangles);
    }

    #[test]
    fn tidy_result_merges_slivers_and_cancels_back_to_back_pairs() {
        // A tetrahedron whose apex is duplicated 1e-9 away: triangles use either copy.
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0 + 1e-9],
        ];
        let triangles = vec![[0, 2, 1], [0, 1, 3], [1, 2, 4], [2, 0, 4]];
        let tidied = tidy_result(mesh(vertices.clone(), triangles.clone()), 1e-6);
        assert_eq!(tidied.vertices.len(), 4, "the duplicated apex must merge");
        assert_eq!(tidied.triangles.len(), 4);
        let as_mesh = TriangleMesh::new(
            tidied.vertices.iter().map(|&v| arr_to_vec3(v)).collect(),
            tidied.triangles.clone(),
        );
        assert_eq!(count_orientation_conflicts(&as_mesh.indices), 0);
        assert!(check_manifold(&as_mesh.indices).is_manifold);

        // A collapsed triangle and a back-to-back pair disappear.
        let mut with_junk = triangles;
        with_junk.push([0, 3, 4]); // collapses: vertices 3 and 4 merge
        with_junk.push([1, 2, 0]); // reverse of [0, 2, 1]
        let tidied = tidy_result(mesh(vertices, with_junk), 1e-6);
        assert_eq!(tidied.vertices.len(), 4);
        assert_eq!(
            tidied.triangles,
            vec![[0, 1, 3], [1, 2, 3], [2, 0, 3]],
            "the collapsed triangle is dropped and [0, 2, 1] cancels with its reverse"
        );
        // epsilon = 0 leaves the mesh untouched.
        let untouched = tidy_result(mesh(vec![[0.0; 3]; 3], vec![[0, 1, 2]]), 0.0);
        assert_eq!(untouched.triangles, vec![[0, 1, 2]]);
    }

    #[test]
    fn orientation_conflicts_count_same_direction_edges() {
        // Closed tetrahedron, consistently oriented.
        let tetra = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        assert_eq!(count_orientation_conflicts(&tetra), 0);
        // Flip one face: its three edges are now traversed twice in one direction.
        let flipped = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        assert_eq!(count_orientation_conflicts(&flipped), 3);
        // Boundary edges (single use) are not orientation conflicts.
        assert_eq!(count_orientation_conflicts(&[[0, 1, 2]]), 0);
        assert!(require_closed_oriented("m", &tetra).is_ok());
        assert!(matches!(
            require_closed_oriented("m", &flipped),
            Err(Error::InvalidMesh { .. })
        ));
        assert!(matches!(
            require_closed_oriented("m", &[[0, 1, 2]]),
            Err(Error::InvalidMesh { .. })
        ));
    }

    #[test]
    fn owned_by_first_attributes_pieces_to_the_operand_they_lie_in() {
        let first = mesh(
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            vec![[0, 1, 2]],
        );
        let result = mesh(
            vec![
                [0.25, 0.25, 0.0],
                [0.75, 0.25, 0.0],
                [0.25, 0.75, 0.0], // inside the first triangle
                [0.25, 0.25, 0.5],
                [0.75, 0.25, 0.5],
                [0.25, 0.75, 0.5], // parallel to it but lifted: not part of it
                [1.5, 1.5, 0.0],
                [2.5, 1.5, 0.0],
                [1.5, 2.5, 0.0], // coplanar but outside its area
            ],
            vec![[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 99]],
        );
        assert_eq!(
            owned_by_first(&result, &first),
            vec![true, false, false, false],
            "a dangling index is never owned"
        );
    }
}

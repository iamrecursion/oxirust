//! Regression tests for the mesh boolean front-ends of this module: results
//! must be split along the intersection curves, closed with consistent
//! orientation and share their vertices, operands must be validated, and the
//! manifold / robust / material-tracking variants must do what they document.

use super::{
    BooleanOp, CsgMaterial, CsgWithMaterials, ManifoldBoolean, MeshBoolean, RobustMeshBoolean,
};
use crate::{Error, TriangleMesh};
use oxiphysics_core::math::Vec3;
use std::collections::{HashMap, HashSet};

// ── helpers ─────────────────────────────────────────────────────────────────

type P3 = [f64; 3];

/// Axis-aligned box `[lo, hi]` with outward winding and 12 triangles.
fn box_mesh(lo: P3, hi: P3) -> TriangleMesh {
    let vertices = vec![
        Vec3::new(lo[0], lo[1], lo[2]),
        Vec3::new(hi[0], lo[1], lo[2]),
        Vec3::new(hi[0], hi[1], lo[2]),
        Vec3::new(lo[0], hi[1], lo[2]),
        Vec3::new(lo[0], lo[1], hi[2]),
        Vec3::new(hi[0], lo[1], hi[2]),
        Vec3::new(hi[0], hi[1], hi[2]),
        Vec3::new(lo[0], hi[1], hi[2]),
    ];
    let indices = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 6, 2],
        [3, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ];
    TriangleMesh::new(vertices, indices)
}

fn corners(m: &TriangleMesh, t: &[usize; 3]) -> [P3; 3] {
    t.map(|i| [m.vertices[i].x, m.vertices[i].y, m.vertices[i].z])
}

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Signed volume enclosed by the surface (divergence theorem).
fn signed_volume(m: &TriangleMesh) -> f64 {
    m.indices
        .iter()
        .map(|t| {
            let [a, b, c] = corners(m, t);
            dot(a, cross(b, c))
        })
        .sum::<f64>()
        / 6.0
}

fn triangle_area(c: [P3; 3]) -> f64 {
    let n = cross(sub(c[1], c[0]), sub(c[2], c[0]));
    0.5 * dot(n, n).sqrt()
}

fn surface_area(m: &TriangleMesh) -> f64 {
    m.indices.iter().map(|t| triangle_area(corners(m, t))).sum()
}

/// Every directed edge appears exactly once and its reverse exactly once: a
/// closed, consistently oriented 2-manifold surface with shared vertices.
fn is_closed_and_oriented(m: &TriangleMesh) -> bool {
    let mut directed: HashMap<(usize, usize), u32> = HashMap::new();
    for t in &m.indices {
        for k in 0..3 {
            *directed.entry((t[k], t[(k + 1) % 3])).or_insert(0) += 1;
        }
    }
    directed
        .iter()
        .all(|(&(a, b), &count)| count == 1 && directed.get(&(b, a)) == Some(&1))
}

fn assert_closed(m: &TriangleMesh, what: &str) {
    assert!(!m.indices.is_empty(), "{what}: empty result");
    assert!(
        is_closed_and_oriented(m),
        "{what}: not a closed, consistently oriented surface"
    );
    let used: HashSet<usize> = m.indices.iter().flatten().copied().collect();
    assert_eq!(
        used.len(),
        m.vertices.len(),
        "{what}: unused vertices were left in the result"
    );
}

/// Euler characteristic `V - E + F` of the used part of the mesh.
fn euler_characteristic(m: &TriangleMesh) -> i64 {
    let vertices: HashSet<usize> = m.indices.iter().flatten().copied().collect();
    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    for t in &m.indices {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.insert((a.min(b), a.max(b)));
        }
    }
    vertices.len() as i64 - edges.len() as i64 + m.indices.len() as i64
}

/// Triangles whose three vertices all have coordinate `axis` equal to `value`.
fn triangles_in_plane(m: &TriangleMesh, axis: usize, value: f64) -> Vec<[P3; 3]> {
    m.indices
        .iter()
        .map(|t| corners(m, t))
        .filter(|c| c.iter().all(|p| (p[axis] - value).abs() < 1e-9))
        .collect()
}

fn unit_normal(c: [P3; 3]) -> P3 {
    let n = cross(sub(c[1], c[0]), sub(c[2], c[0]));
    let len = dot(n, n).sqrt();
    [n[0] / len, n[1] / len, n[2] / len]
}

fn rotate(v: P3, axis: P3, angle: f64) -> P3 {
    let norm = dot(axis, axis).sqrt();
    let k = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let (s, c) = angle.sin_cos();
    let kxv = cross(k, v);
    let kdv = dot(k, v);
    [
        v[0] * c + kxv[0] * s + k[0] * kdv * (1.0 - c),
        v[1] * c + kxv[1] * s + k[1] * kdv * (1.0 - c),
        v[2] * c + kxv[2] * s + k[2] * kdv * (1.0 - c),
    ]
}

fn rotated_cube(center: P3, half: f64, axis: P3, angle: f64) -> TriangleMesh {
    let mut m = box_mesh([-half; 3], [half; 3]);
    for v in &mut m.vertices {
        let r = rotate([v.x, v.y, v.z], axis, angle);
        *v = Vec3::new(r[0] + center[0], r[1] + center[1], r[2] + center[2]);
    }
    m
}

fn run(op: BooleanOp, a: &TriangleMesh, b: &TriangleMesh) -> TriangleMesh {
    MeshBoolean::execute(op, a, b).unwrap()
}

fn material(id: u32) -> CsgMaterial {
    CsgMaterial {
        id,
        name: format!("material-{id}"),
        density: 1000.0 + f64::from(id),
    }
}

// ── MeshBoolean ─────────────────────────────────────────────────────────────

#[test]
fn offset_unit_cubes_give_exact_closed_booleans() {
    // B = A shifted by 0.5 along x: mid-face cuts on the +-X faces and
    // partially overlapping coplanar +-Y / +-Z faces.
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);

    let union = run(BooleanOp::Union, &a, &b);
    assert_closed(&union, "union");
    assert!((signed_volume(&union) - 1.5).abs() < 1e-9, "union volume");
    assert!((surface_area(&union) - 8.0).abs() < 1e-9, "union area");
    assert_eq!(euler_characteristic(&union), 2);
    // Shared vertices: a closed genus-0 triangulation has V = F / 2 + 2.
    assert_eq!(union.vertices.len(), union.indices.len() / 2 + 2);

    let inter = run(BooleanOp::Intersection, &a, &b);
    assert_closed(&inter, "intersection");
    assert!(
        (signed_volume(&inter) - 0.5).abs() < 1e-9,
        "intersection volume"
    );
    assert!(
        (surface_area(&inter) - 4.0).abs() < 1e-9,
        "intersection area"
    );

    let diff = run(BooleanOp::Difference, &a, &b);
    assert_closed(&diff, "difference");
    assert!(
        (signed_volume(&diff) - 0.5).abs() < 1e-9,
        "difference volume"
    );
    assert!((surface_area(&diff) - 4.0).abs() < 1e-9, "difference area");

    let reverse = run(BooleanOp::Difference, &b, &a);
    assert_closed(&reverse, "reverse difference");
    assert!((signed_volume(&reverse) - 0.5).abs() < 1e-9);
}

#[test]
fn general_position_overlap_obeys_inclusion_exclusion() {
    // Offset (0.3, 0.2, 0.1): the faces of one box cut the faces of the other
    // in the middle, the case a whole-triangle selection cannot represent.
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let b = box_mesh([0.3, 0.2, 0.1], [1.3, 1.2, 1.1]);
    let overlap = 0.7 * 0.8 * 0.9;

    let union = run(BooleanOp::Union, &a, &b);
    let inter = run(BooleanOp::Intersection, &a, &b);
    let diff_ab = run(BooleanOp::Difference, &a, &b);
    let diff_ba = run(BooleanOp::Difference, &b, &a);
    for (what, m) in [
        ("union", &union),
        ("intersection", &inter),
        ("a - b", &diff_ab),
        ("b - a", &diff_ba),
    ] {
        assert_closed(m, what);
    }
    assert!((signed_volume(&inter) - overlap).abs() < 1e-9);
    assert!((signed_volume(&union) - (2.0 - overlap)).abs() < 1e-9);
    assert!((signed_volume(&diff_ab) - (1.0 - overlap)).abs() < 1e-9);
    assert!((signed_volume(&diff_ba) - (1.0 - overlap)).abs() < 1e-9);
    // The intersection is the box [0.3, 1] x [0.2, 1] x [0.1, 1].
    assert!((surface_area(&inter) - 2.0 * (0.7 * 0.8 + 0.7 * 0.9 + 0.8 * 0.9)).abs() < 1e-9);
    // The difference of an intersecting pair is not simply a subset of whole
    // input triangles: it contains vertices that belong to neither operand.
    let on_a_or_b = |p: P3| {
        a.vertices
            .iter()
            .chain(b.vertices.iter())
            .any(|v| (v.x - p[0]).abs() + (v.y - p[1]).abs() + (v.z - p[2]).abs() < 1e-9)
    };
    assert!(
        union.vertices.iter().any(|v| !on_a_or_b([v.x, v.y, v.z])),
        "the union must contain intersection-curve vertices"
    );
}

#[test]
fn boxes_with_a_shared_z_extent_keep_their_coplanar_faces_once() {
    // A = [0, 1]^3, B = [0.2, 1.5]^2 x [0, 1]: the top (z = 1) and bottom (z = 0)
    // faces are coplanar and overlap in [0.2, 1]^2.
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let b = box_mesh([0.2, 0.2, 0.0], [1.5, 1.5, 1.0]);
    let overlap = 0.8 * 0.8;

    let inter = run(BooleanOp::Intersection, &a, &b);
    assert_closed(&inter, "intersection");
    assert!((signed_volume(&inter) - overlap).abs() < 1e-9);
    // A's top-face triangles inside B survive, exactly once, facing up and
    // staying inside the overlap rectangle.
    let top = triangles_in_plane(&inter, 2, 1.0);
    assert!(!top.is_empty(), "the intersection lost its top face");
    let top_area: f64 = top.iter().map(|c| triangle_area(*c)).sum();
    assert!(
        (top_area - overlap).abs() < 1e-9,
        "top face area {top_area}"
    );
    for c in &top {
        assert!(unit_normal(*c)[2] > 0.99, "top face triangle faces down");
        for p in c {
            assert!(p[0] >= 0.2 - 1e-9 && p[0] <= 1.0 + 1e-9, "x = {}", p[0]);
            assert!(p[1] >= 0.2 - 1e-9 && p[1] <= 1.0 + 1e-9, "y = {}", p[1]);
        }
    }
    let bottom = triangles_in_plane(&inter, 2, 0.0);
    let bottom_area: f64 = bottom.iter().map(|c| triangle_area(*c)).sum();
    assert!((bottom_area - overlap).abs() < 1e-9, "bottom face area");
    assert!(bottom.iter().all(|c| unit_normal(*c)[2] < -0.99));

    let union = run(BooleanOp::Union, &a, &b);
    assert_closed(&union, "union");
    assert!((signed_volume(&union) - (1.0 + 1.3 * 1.3 - overlap)).abs() < 1e-9);
    let diff = run(BooleanOp::Difference, &a, &b);
    assert_closed(&diff, "difference");
    assert!((signed_volume(&diff) - (1.0 - overlap)).abs() < 1e-9);

    // The configuration of the audit: B = [0.5, 1.5]^2 x [0, 1].
    let b_half = box_mesh([0.5, 0.5, 0.0], [1.5, 1.5, 1.0]);
    let inter_half = run(BooleanOp::Intersection, &a, &b_half);
    assert_closed(&inter_half, "intersection (audit configuration)");
    assert!((signed_volume(&inter_half) - 0.25).abs() < 1e-9);
    let union_half = run(BooleanOp::Union, &a, &b_half);
    assert_closed(&union_half, "union (audit configuration)");
    assert!((signed_volume(&union_half) - 1.75).abs() < 1e-9);
}

#[test]
fn boxes_touching_face_to_face_merge_without_an_internal_wall() {
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let b = box_mesh([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);

    let union = run(BooleanOp::Union, &a, &b);
    assert_closed(&union, "union");
    assert!((signed_volume(&union) - 2.0).abs() < 1e-9);
    assert!((surface_area(&union) - 10.0).abs() < 1e-9);
    assert!(
        triangles_in_plane(&union, 0, 1.0).is_empty(),
        "the shared interface x = 1 must not remain as an internal wall"
    );
    assert_eq!(union.vertices.len(), 12);

    let inter = run(BooleanOp::Intersection, &a, &b);
    assert!(
        inter.indices.is_empty(),
        "boxes that only touch have no common volume"
    );

    let diff = run(BooleanOp::Difference, &a, &b);
    assert_closed(&diff, "difference");
    assert!((signed_volume(&diff) - 1.0).abs() < 1e-9);
    assert!((surface_area(&diff) - 6.0).abs() < 1e-9);
}

#[test]
fn disjoint_nested_and_identical_operands() {
    let a = box_mesh([0.0; 3], [1.0; 3]);

    // Disjoint.
    let far = box_mesh([5.0; 3], [6.0; 3]);
    let union = run(BooleanOp::Union, &a, &far);
    assert_closed(&union, "union of disjoint boxes");
    assert_eq!(union.indices.len(), 24);
    assert!((signed_volume(&union) - 2.0).abs() < 1e-9);
    assert!(run(BooleanOp::Intersection, &a, &far).indices.is_empty());
    let diff = run(BooleanOp::Difference, &a, &far);
    assert_eq!(diff.indices.len(), 12);
    assert!((signed_volume(&diff) - 1.0).abs() < 1e-9);

    // Nested: the difference is a cube with a cavity whose wall faces inward.
    let small = box_mesh([0.25; 3], [0.75; 3]);
    let union = run(BooleanOp::Union, &a, &small);
    assert_closed(&union, "union with a nested box");
    assert_eq!(union.indices.len(), 12);
    assert!((signed_volume(&union) - 1.0).abs() < 1e-9);
    let inter = run(BooleanOp::Intersection, &a, &small);
    assert_closed(&inter, "intersection with a nested box");
    assert!((signed_volume(&inter) - 0.125).abs() < 1e-9);
    let hollow = run(BooleanOp::Difference, &a, &small);
    assert_closed(&hollow, "hollowed box");
    assert_eq!(hollow.indices.len(), 24);
    assert!((signed_volume(&hollow) - 0.875).abs() < 1e-9);
    assert!((surface_area(&hollow) - (6.0 + 1.5)).abs() < 1e-9);
    assert!(run(BooleanOp::Difference, &small, &a).indices.is_empty());

    // Identical operands.
    let same = run(BooleanOp::Union, &a, &a);
    assert_closed(&same, "A union A");
    assert_eq!(same.indices.len(), 12);
    assert!((signed_volume(&same) - 1.0).abs() < 1e-9);
    let same = run(BooleanOp::Intersection, &a, &a);
    assert_closed(&same, "A intersect A");
    assert!((signed_volume(&same) - 1.0).abs() < 1e-9);
    assert!(run(BooleanOp::Difference, &a, &a).indices.is_empty());
}

#[test]
fn rotated_cube_against_axis_aligned_cube_is_consistent() {
    let a = box_mesh([-0.5; 3], [0.5; 3]);
    let b = rotated_cube([0.2, 0.1, 0.05], 0.5, [1.0, 2.0, 3.0], 0.6);
    let union = run(BooleanOp::Union, &a, &b);
    let inter = run(BooleanOp::Intersection, &a, &b);
    let diff = run(BooleanOp::Difference, &a, &b);
    for (what, m) in [
        ("union", &union),
        ("intersection", &inter),
        ("difference", &diff),
    ] {
        assert_closed(m, what);
    }
    let (vu, vi, vd) = (
        signed_volume(&union),
        signed_volume(&inter),
        signed_volume(&diff),
    );
    assert!(vi > 0.1 && vi < 0.95, "implausible overlap volume {vi}");
    assert!(
        (vu + vi - 2.0).abs() < 1e-9,
        "inclusion-exclusion: {vu} + {vi}"
    );
    assert!((vd - (1.0 - vi)).abs() < 1e-9, "a - b = a - (a and b)");
}

#[test]
fn mesh_boolean_validates_its_operands() {
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let empty = TriangleMesh::new(Vec::new(), Vec::new());
    for op in [
        BooleanOp::Union,
        BooleanOp::Intersection,
        BooleanOp::Difference,
    ] {
        assert!(matches!(
            MeshBoolean::execute(op, &empty, &a),
            Err(Error::TooFewPoints {
                required: 1,
                actual: 0
            })
        ));
        assert!(matches!(
            MeshBoolean::execute(op, &a, &empty),
            Err(Error::TooFewPoints { .. })
        ));
    }
    // Vertices without triangles.
    let no_triangles = TriangleMesh::new(vec![Vec3::new(0.0, 0.0, 0.0)], Vec::new());
    assert!(matches!(
        MeshBoolean::execute(BooleanOp::Union, &a, &no_triangles),
        Err(Error::InvalidMesh { .. })
    ));
    // A triangle that references a missing vertex used to panic.
    let mut dangling = a.clone();
    dangling.indices[3] = [0, 1, 99];
    assert!(matches!(
        MeshBoolean::execute(BooleanOp::Union, &dangling, &a),
        Err(Error::IndexOutOfBounds { index: 99, len: 8 })
    ));
    // Non-finite coordinates.
    let mut nan = a.clone();
    nan.vertices[2] = Vec3::new(f64::NAN, 0.0, 0.0);
    assert!(matches!(
        MeshBoolean::execute(BooleanOp::Union, &a, &nan),
        Err(Error::InvalidMesh { .. })
    ));
}

// ── RobustMeshBoolean ───────────────────────────────────────────────────────

#[test]
fn robust_boolean_matches_the_exact_volumes() {
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let b = box_mesh([0.3, 0.2, 0.1], [1.3, 1.2, 1.1]);
    let overlap = 0.7 * 0.8 * 0.9;
    for epsilon in [0.0, 1e-9, 1e-6] {
        let union = RobustMeshBoolean::execute(BooleanOp::Union, &a, &b, epsilon).unwrap();
        let inter = RobustMeshBoolean::execute(BooleanOp::Intersection, &a, &b, epsilon).unwrap();
        let diff = RobustMeshBoolean::execute(BooleanOp::Difference, &a, &b, epsilon).unwrap();
        for (what, m) in [
            ("union", &union),
            ("intersection", &inter),
            ("difference", &diff),
        ] {
            assert_closed(m, &format!("{what} (epsilon {epsilon})"));
        }
        assert!((signed_volume(&union) - (2.0 - overlap)).abs() < 1e-5);
        assert!((signed_volume(&inter) - overlap).abs() < 1e-5);
        assert!((signed_volume(&diff) - (1.0 - overlap)).abs() < 1e-5);
    }
}

#[test]
fn robust_boolean_keeps_the_coplanar_top_face_of_the_audit_configuration() {
    // Boxes with the same z-extent: every centroid on z = 0 or z = 1 used to be
    // forced "outside" because it lay on the extended plane of some triangle.
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let b = box_mesh([0.2, 0.2, 0.0], [1.5, 1.5, 1.0]);
    let inter = RobustMeshBoolean::execute(BooleanOp::Intersection, &a, &b, 1e-9).unwrap();
    assert_closed(&inter, "intersection");
    assert!((signed_volume(&inter) - 0.64).abs() < 1e-8);
    let top = triangles_in_plane(&inter, 2, 1.0);
    let top_area: f64 = top.iter().map(|c| triangle_area(*c)).sum();
    assert!((top_area - 0.64).abs() < 1e-8, "top face area {top_area}");
    assert!(top.iter().all(|c| unit_normal(*c)[2] > 0.99));
    for c in &top {
        for p in c {
            assert!(p[0] >= 0.2 - 1e-8 && p[0] <= 1.0 + 1e-8);
            assert!(p[1] >= 0.2 - 1e-8 && p[1] <= 1.0 + 1e-8);
        }
    }
    let union = RobustMeshBoolean::execute(BooleanOp::Union, &a, &b, 1e-9).unwrap();
    assert_closed(&union, "union");
    assert!((signed_volume(&union) - (1.0 + 1.69 - 0.64)).abs() < 1e-8);
    let diff = RobustMeshBoolean::execute(BooleanOp::Difference, &a, &b, 1e-9).unwrap();
    assert_closed(&diff, "difference");
    assert!((signed_volume(&diff) - 0.36).abs() < 1e-8);
}

#[test]
fn robust_union_of_face_touching_cubes_has_no_internal_wall() {
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let b = box_mesh([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let union = RobustMeshBoolean::execute(BooleanOp::Union, &a, &b, 1e-9).unwrap();
    assert_closed(&union, "union");
    assert!((signed_volume(&union) - 2.0).abs() < 1e-9);
    assert!(triangles_in_plane(&union, 0, 1.0).is_empty());
    let inter = RobustMeshBoolean::execute(BooleanOp::Intersection, &a, &b, 1e-9).unwrap();
    assert!(inter.indices.is_empty());
}

#[test]
fn robust_boolean_snaps_nearly_coincident_vertices() {
    // The face x = 1 of B is displaced by up to ~5e-13 from the face x = 1 of A
    // and tilted by it: nearly, but not exactly, coplanar.  With epsilon = 1e-9
    // the corners snap onto each other and the shared face cancels.
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let mut b = box_mesh([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let nudges = [
        [4e-13, -2e-13, 3e-13],
        [0.0; 3],
        [0.0; 3],
        [-3e-13, 5e-13, -4e-13],
        [2e-13, 3e-13, -5e-13],
        [0.0; 3],
        [0.0; 3],
        [-5e-13, -3e-13, 2e-13],
    ];
    for (v, nudge) in b.vertices.iter_mut().zip(nudges) {
        *v += Vec3::new(nudge[0], nudge[1], nudge[2]);
    }
    let union = RobustMeshBoolean::execute(BooleanOp::Union, &a, &b, 1e-9).unwrap();
    assert_closed(&union, "union");
    assert!((signed_volume(&union) - 2.0).abs() < 1e-9);
    assert!((surface_area(&union) - 10.0).abs() < 1e-8);
    assert_eq!(
        union.vertices.len(),
        12,
        "the four shared corners must merge"
    );
    assert!(triangles_in_plane(&union, 0, 1.0).is_empty());
    let inter = RobustMeshBoolean::execute(BooleanOp::Intersection, &a, &b, 1e-9).unwrap();
    assert!(inter.indices.is_empty());
}

#[test]
fn robust_boolean_closes_the_seam_of_nearly_coplanar_faces() {
    // B = [0.5, 1.5]^2 x [0, 1] with every vertex displaced by up to about 1e-12,
    // so its top and bottom faces are only nearly coplanar with those of A and
    // their corners coincide with no corner of A.  The exact arrangement alone
    // sees nearly parallel planes there; snap rounding makes the faces coplanar.
    // Displacements are in units of 1e-15, one triple per vertex of B.
    const NUDGES: [[[i32; 3]; 8]; 2] = [
        [
            [-213, -701, 38],
            [-324, -303, -456],
            [-934, 353, 633],
            [-44, 406, -317],
            [901, 323, 674],
            [873, -1, 395],
            [383, -616, 89],
            [925, 330, -306],
        ],
        [
            [-602, -807, -344],
            [-980, 271, -787],
            [16, 751, 777],
            [793, 584, -608],
            [501, -171, -702],
            [-284, -260, 664],
            [519, 407, 792],
            [-337, -614, 984],
        ],
    ];
    let a = box_mesh([0.0; 3], [1.0; 3]);
    for nudges in NUDGES {
        let mut b = box_mesh([0.5, 0.5, 0.0], [1.5, 1.5, 1.0]);
        for (v, n) in b.vertices.iter_mut().zip(nudges) {
            *v += Vec3::new(
                f64::from(n[0]) * 1e-15,
                f64::from(n[1]) * 1e-15,
                f64::from(n[2]) * 1e-15,
            );
        }
        for (op, want) in [
            (BooleanOp::Union, 1.75),
            (BooleanOp::Intersection, 0.25),
            (BooleanOp::Difference, 0.75),
        ] {
            let result = RobustMeshBoolean::execute(op, &a, &b, 1e-9).unwrap();
            assert_closed(&result, &format!("{op:?} of nudged boxes"));
            assert!(
                (signed_volume(&result) - want).abs() < 1e-9,
                "{op:?}: volume {} instead of {want}",
                signed_volume(&result)
            );
        }
        // The top of the union is one flat plane covering the L-shaped footprint.
        let union = RobustMeshBoolean::execute(BooleanOp::Union, &a, &b, 1e-9).unwrap();
        let top_area: f64 = triangles_in_plane(&union, 2, 1.0)
            .iter()
            .map(|c| triangle_area(*c))
            .sum();
        assert!((top_area - 1.75).abs() < 1e-9, "top face area {top_area}");
    }
}

#[test]
fn robust_boolean_removes_steps_thinner_than_epsilon() {
    // B's top face lies 1e-9 above A's top plane: the exact union has a
    // 1e-9-high step wall there; with epsilon = 1e-6 both tops become one plane.
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let b = box_mesh([0.5, 0.5, 0.0], [1.5, 1.5, 1.0 + 1e-9]);
    let epsilon = 1e-6;
    let exact = run(BooleanOp::Union, &a, &b);
    assert!(
        exact.vertices.iter().any(|v| v.z > 1.0 + 1e-10),
        "the exact union keeps B's slightly higher top"
    );
    let robust = RobustMeshBoolean::execute(BooleanOp::Union, &a, &b, epsilon).unwrap();
    assert_closed(&robust, "robust union");
    assert!((signed_volume(&robust) - 1.75).abs() < 1e-6);
    assert!(
        robust.vertices.iter().all(|v| v.z <= 1.0 + 1e-12),
        "nothing may stick out above the common top plane z = 1"
    );
    let top: f64 = triangles_in_plane(&robust, 2, 1.0)
        .iter()
        .map(|c| triangle_area(*c))
        .sum();
    assert!((top - 1.75).abs() < 1e-6, "top face area {top}");
    // No two distinct result vertices are closer than epsilon.
    for (i, p) in robust.vertices.iter().enumerate() {
        for q in &robust.vertices[i + 1..] {
            assert!(
                (p - q).norm() >= epsilon * (1.0 - 1e-9),
                "{p:?} and {q:?} are too close"
            );
        }
    }
}

#[test]
fn robust_boolean_rejects_an_invalid_epsilon() {
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e-9] {
        assert!(
            matches!(
                RobustMeshBoolean::execute(BooleanOp::Union, &a, &b, bad),
                Err(Error::OutOfRange {
                    name: "epsilon",
                    ..
                })
            ),
            "epsilon {bad} must be rejected"
        );
    }
    let empty = TriangleMesh::new(Vec::new(), Vec::new());
    assert!(matches!(
        RobustMeshBoolean::execute(BooleanOp::Union, &empty, &b, 1e-9),
        Err(Error::TooFewPoints { .. })
    ));
}

// ── ManifoldBoolean ─────────────────────────────────────────────────────────

fn error_text(result: crate::Result<TriangleMesh>) -> String {
    result.unwrap_err().to_string()
}

#[test]
fn manifold_boolean_rejects_open_meshes() {
    let closed = box_mesh([0.0; 3], [1.0; 3]);
    let other = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
    // Drop one triangle: the mesh now has boundary edges.
    let mut open = closed.clone();
    open.indices.pop();
    for op in [
        BooleanOp::Union,
        BooleanOp::Intersection,
        BooleanOp::Difference,
    ] {
        let result = ManifoldBoolean::execute(op, &open, &other);
        assert!(
            matches!(&result, Err(Error::InvalidMesh { reason }) if reason.starts_with("mesh_a is non-manifold")),
            "{result:?}"
        );
        let result = ManifoldBoolean::execute_strict(op, &other, &open);
        assert!(
            matches!(&result, Err(Error::InvalidMesh { reason }) if reason.starts_with("mesh_b is non-manifold")),
            "{result:?}"
        );
    }
    let text = error_text(ManifoldBoolean::execute(BooleanOp::Union, &open, &other));
    assert!(
        text.contains("3 boundary edges, 0 non-manifold edges"),
        "{text}"
    );
}

#[test]
fn manifold_boolean_rejects_edges_shared_by_three_triangles() {
    let closed = box_mesh([0.0; 3], [1.0; 3]);
    let other = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
    // A fin on the edge (0, 1): three triangles now use it.
    let mut finned = closed.clone();
    finned.vertices.push(Vec3::new(0.5, -1.0, -1.0));
    finned.indices.push([0, 1, 8]);
    let text = error_text(ManifoldBoolean::execute(BooleanOp::Union, &finned, &other));
    assert!(text.contains("mesh_a is non-manifold"), "{text}");
    assert!(text.contains("1 non-manifold edges"), "{text}");
}

#[test]
fn manifold_boolean_rejects_inconsistently_oriented_meshes() {
    let closed = box_mesh([0.0; 3], [1.0; 3]);
    let other = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
    // Every edge still has exactly two triangles, but one face is flipped.
    let mut flipped = closed.clone();
    flipped.indices[5].swap(0, 1);
    let result = ManifoldBoolean::execute(BooleanOp::Union, &other, &flipped);
    assert!(
        matches!(&result, Err(Error::InvalidMesh { reason }) if reason.starts_with("mesh_b is not consistently oriented")),
        "{result:?}"
    );
}

#[test]
fn manifold_boolean_rejects_empty_meshes_and_accepts_valid_ones() {
    let closed = box_mesh([0.0; 3], [1.0; 3]);
    let other = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
    let empty = TriangleMesh::new(Vec::new(), Vec::new());
    assert!(matches!(
        ManifoldBoolean::execute(BooleanOp::Union, &empty, &closed),
        Err(Error::TooFewPoints {
            required: 1,
            actual: 0
        })
    ));
    let union = ManifoldBoolean::execute(BooleanOp::Union, &closed, &other).unwrap();
    assert_closed(&union, "validated union");
    assert!((signed_volume(&union) - 1.5).abs() < 1e-9);
    let same = ManifoldBoolean::execute(BooleanOp::Union, &closed, &closed).unwrap();
    assert_closed(&same, "validated self-union");
    assert!((signed_volume(&same) - 1.0).abs() < 1e-9);
    assert_eq!(same.indices.len(), 12);
}

// ── CsgWithMaterials ────────────────────────────────────────────────────────

fn area_by_material(m: &TriangleMesh, materials: &[CsgMaterial], id: u32) -> f64 {
    m.indices
        .iter()
        .zip(materials)
        .filter(|(_, mat)| mat.id == id)
        .map(|(t, _)| triangle_area(corners(m, t)))
        .sum()
}

#[test]
fn csg_materials_follow_the_operand_each_triangle_came_from() {
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);

    // Union: A keeps its x = 0 face and the four side faces up to x = 1 (the
    // coplanar overlap resolves to A's copy), B adds x = 1.5 and the four side
    // faces beyond x = 1.
    let (union, mats) =
        CsgWithMaterials::execute(BooleanOp::Union, &a, material(1), &b, material(2)).unwrap();
    assert_closed(&union, "union");
    assert_eq!(mats.len(), union.indices.len());
    assert!((signed_volume(&union) - 1.5).abs() < 1e-9);
    assert!((area_by_material(&union, &mats, 1) - 5.0).abs() < 1e-9);
    assert!((area_by_material(&union, &mats, 2) - 3.0).abs() < 1e-9);
    assert!(mats.iter().all(|m| m.name == format!("material-{}", m.id)));

    // Intersection [0.5, 1] x [0, 1]^2: A contributes x = 1 and the shared side
    // faces, B contributes its x = 0.5 face.
    let (inter, mats) =
        CsgWithMaterials::execute(BooleanOp::Intersection, &a, material(1), &b, material(2))
            .unwrap();
    assert_closed(&inter, "intersection");
    assert_eq!(mats.len(), inter.indices.len());
    assert!((signed_volume(&inter) - 0.5).abs() < 1e-9);
    assert!((area_by_material(&inter, &mats, 1) - 3.0).abs() < 1e-9);
    assert!((area_by_material(&inter, &mats, 2) - 1.0).abs() < 1e-9);

    // Difference A - B = [0, 0.5] x [0, 1]^2: the new wall at x = 0.5 is B's
    // surface, facing +x (out of the remaining solid).
    let (diff, mats) =
        CsgWithMaterials::execute(BooleanOp::Difference, &a, material(1), &b, material(2)).unwrap();
    assert_closed(&diff, "difference");
    assert!((signed_volume(&diff) - 0.5).abs() < 1e-9);
    assert!((area_by_material(&diff, &mats, 1) - 3.0).abs() < 1e-9);
    assert!((area_by_material(&diff, &mats, 2) - 1.0).abs() < 1e-9);
    let wall: Vec<_> = diff
        .indices
        .iter()
        .zip(&mats)
        .filter(|(_, mat)| mat.id == 2)
        .map(|(t, _)| corners(&diff, t))
        .collect();
    assert!(!wall.is_empty());
    for c in &wall {
        assert!(
            c.iter().all(|p| (p[0] - 0.5).abs() < 1e-9),
            "wall not at x = 0.5"
        );
        assert!(unit_normal(*c)[0] > 0.99, "cavity wall must face +x");
    }
}

#[test]
fn csg_materials_for_disjoint_nested_and_invalid_operands() {
    let a = box_mesh([0.0; 3], [1.0; 3]);
    let far = box_mesh([5.0; 3], [6.0; 3]);
    let (union, mats) =
        CsgWithMaterials::execute(BooleanOp::Union, &a, material(10), &far, material(20)).unwrap();
    assert_eq!(union.indices.len(), 24);
    assert_eq!(mats.iter().filter(|m| m.id == 10).count(), 12);
    assert_eq!(mats.iter().filter(|m| m.id == 20).count(), 12);
    // Nested B: the cavity wall (flipped) belongs to B, the shell to A.
    let small = box_mesh([0.25; 3], [0.75; 3]);
    let (hollow, mats) =
        CsgWithMaterials::execute(BooleanOp::Difference, &a, material(1), &small, material(2))
            .unwrap();
    assert_closed(&hollow, "hollowed box");
    assert!((area_by_material(&hollow, &mats, 1) - 6.0).abs() < 1e-9);
    assert!((area_by_material(&hollow, &mats, 2) - 1.5).abs() < 1e-9);
    let empty = TriangleMesh::new(Vec::new(), Vec::new());
    assert!(
        CsgWithMaterials::execute(BooleanOp::Union, &empty, material(1), &a, material(2)).is_err()
    );
}

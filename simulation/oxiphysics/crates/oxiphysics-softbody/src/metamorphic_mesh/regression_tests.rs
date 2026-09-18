// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the rigid-alignment (Kabsch / Procrustes / ICP), ARAP
//! and RBF deformation code of the metamorphic mesh module.
//!
//! The previous alignment code iterated `(R + R⁻¹) / 2` (a rotation collapsed
//! to the identity), used the transposed rotation, and its ARAP global step had
//! the wrong sign; the RBF solve was a Gauss-Seidel sweep that left all
//! thin-plate coefficients at zero.  Every test below recovers a *known*
//! transform from non-trivial data, so none of them is satisfied by the
//! identity.

use super::*;

const TOL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random numbers in `[-1, 1)` (64-bit LCG).
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn next_point(&mut self, half_extent: f64) -> [f64; 3] {
        [
            half_extent * self.next_unit(),
            half_extent * self.next_unit(),
            half_extent * self.next_unit(),
        ]
    }
}

/// Rotation by `angle` (rad) about `axis` (Rodrigues formula).
fn rotation_about(axis: [f64; 3], angle: f64) -> Mat3 {
    let k = v3_normalize(axis);
    let (s, c) = angle.sin_cos();
    let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    let kk = outer3(k, k);
    let mut r = mat3_scale(mat3_identity(), c);
    r = mat3_add(r, mat3_scale(kx, s));
    mat3_add(r, mat3_scale(kk, 1.0 - c))
}

fn rot_z(angle: f64) -> Mat3 {
    rotation_about([0.0, 0.0, 1.0], angle)
}

fn max_abs_diff(a: Mat3, b: Mat3) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((a[i][j] - b[i][j]).abs());
        }
    }
    worst
}

fn det3(m: Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Assert that `r` is a proper rotation: orthonormal with determinant +1.
fn assert_proper_rotation(r: Mat3, what: &str) {
    let rrt = mat3_mul(r, mat3_transpose(r));
    assert!(
        max_abs_diff(rrt, mat3_identity()) < TOL,
        "{what}: R Rᵀ is not the identity: {rrt:?}"
    );
    assert!((det3(r) - 1.0).abs() < TOL, "{what}: det R = {}", det3(r));
}

fn apply_rigid(r: Mat3, t: [f64; 3], pts: &[[f64; 3]]) -> Vec<[f64; 3]> {
    pts.iter().map(|p| v3_add(mat3_mv(r, *p), t)).collect()
}

/// Sum of squared distances `Σ ‖R a_i − b_i‖²` (centred sets).
fn alignment_cost(r: Mat3, a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| v3_norm2(v3_sub(mat3_mv(r, *p), *q)))
        .sum()
}

/// An asymmetric, well-spread cloud (all points at least ~1 apart).
fn asymmetric_cloud() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.3, 0.2, 0.1],
        [0.4, 1.7, 0.3],
        [0.2, 0.5, 1.9],
        [1.1, 1.4, 1.2],
        [2.0, 0.9, 0.5],
        [-0.9, 1.1, 0.8],
    ]
}

// ---------------------------------------------------------------------------
// Kabsch rotation
// ---------------------------------------------------------------------------

/// The old Higham iteration collapsed `Rz(0.5 rad)` to the identity.  Procrustes
/// and the ICP best-fit step must recover rotations of 30° and 170° exactly.
#[test]
fn procrustes_and_best_fit_recover_z_rotations() {
    let src = asymmetric_cloud();
    for angle_deg in [30.0_f64, 170.0, -120.0] {
        let q = rot_z(angle_deg.to_radians());
        let t = [0.7, -1.3, 2.1];
        let dst = apply_rigid(q, t, &src);

        let result = procrustes(&src, &dst);
        assert!(
            max_abs_diff(result.rotation, q) < TOL,
            "procrustes rotation at {angle_deg} deg: {:?}",
            result.rotation
        );
        assert!((result.scale - 1.0).abs() < TOL, "scale={}", result.scale);
        assert!(result.residual < 1e-16, "residual={}", result.residual);
        for (a, b) in result.translation.iter().zip(t.iter()) {
            assert!((a - b).abs() < TOL, "translation {:?}", result.translation);
        }

        let (r, t_fit) = IcpSolver::best_fit_transform(&src, &dst);
        assert!(
            max_abs_diff(r, q) < TOL,
            "best_fit rotation at {angle_deg} deg: {r:?}"
        );
        for (a, b) in t_fit.iter().zip(t.iter()) {
            assert!((a - b).abs() < TOL, "best_fit translation {t_fit:?}");
        }
    }
}

/// Rotations about arbitrary axes (not only z) are recovered, including the
/// uniform scale of a similarity transform.
#[test]
fn procrustes_recovers_similarity_transform() {
    let mut rng = Lcg(0x5eed_1234);
    let src: Vec<[f64; 3]> = (0..12).map(|_| rng.next_point(2.0)).collect();
    for (axis, angle_deg, scale) in [
        ([1.0, 2.0, 3.0], 75.0_f64, 1.0),
        ([-0.3, 0.8, 0.1], 160.0, 2.5),
        ([0.0, 1.0, 0.0], 5.0, 0.4),
    ] {
        let q = rotation_about(axis, angle_deg.to_radians());
        let t = [1.0, 2.0, -3.0];
        let dst: Vec<[f64; 3]> = src
            .iter()
            .map(|p| v3_add(v3_scale(mat3_mv(q, *p), scale), t))
            .collect();
        let result = procrustes(&src, &dst);
        assert!(
            max_abs_diff(result.rotation, q) < TOL,
            "rotation about {axis:?}: {:?}",
            result.rotation
        );
        assert!(
            (result.scale - scale).abs() < TOL,
            "scale {} vs {scale}",
            result.scale
        );
        assert!(result.residual < 1e-14, "residual={}", result.residual);
    }
}

/// A planar (rank-2 covariance) point set rotated by 90° — in plane and out of
/// plane — gives the unique rotation that maps the plane with its orientation.
#[test]
fn kabsch_recovers_rotation_of_planar_point_set() {
    // Triangle in the plane z = 0.
    let src = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 1.5, 0.0]];
    for (axis, angle_deg) in [
        ([0.0, 0.0, 1.0], 90.0_f64),
        ([1.0, 0.0, 0.0], 90.0),
        ([1.0, 1.0, 0.0], 90.0),
        ([0.3, -0.7, 0.4], 123.0),
    ] {
        let q = rotation_about(axis, angle_deg.to_radians());
        let dst = apply_rigid(q, [0.2, 0.1, -0.4], &src);
        let (r, _) = IcpSolver::best_fit_transform(&src, &dst);
        assert_proper_rotation(r, "planar");
        assert!(
            max_abs_diff(r, q) < TOL,
            "planar set, axis {axis:?}, {angle_deg} deg: {r:?}"
        );
    }

    // A tilted plane, many points (rank-2 covariance with rounding noise in
    // the vanishing singular value).
    let mut rng = Lcg(77);
    let normal = v3_normalize([0.4, -0.5, 0.7]);
    let u_axis = v3_normalize(v3_cross(normal, [0.0, 0.0, 1.0]));
    let v_axis = v3_cross(normal, u_axis);
    let tilted: Vec<[f64; 3]> = (0..9)
        .map(|_| {
            let (a, b) = (3.0 * rng.next_unit(), 3.0 * rng.next_unit());
            v3_add(v3_scale(u_axis, a), v3_scale(v_axis, b))
        })
        .collect();
    let q = rotation_about([0.2, 0.9, -0.3], 1.9);
    let dst = apply_rigid(q, [0.0; 3], &tilted);
    let (r, _) = IcpSolver::best_fit_transform(&tilted, &dst);
    assert_proper_rotation(r, "tilted plane");
    assert!(max_abs_diff(r, q) < TOL, "tilted plane: {r:?}");
}

/// Collinear sets (rank-1 covariance) leave the twist about the line free; the
/// result must still be a proper rotation that maps the line direction.
#[test]
fn kabsch_handles_collinear_point_set() {
    let direction = v3_normalize([1.0, 2.0, -0.5]);
    let src: Vec<[f64; 3]> = [-1.0, 0.2, 0.9, 2.0]
        .iter()
        .map(|s| v3_scale(direction, *s))
        .collect();
    for (axis, angle_deg) in [([0.0, 1.0, 0.0], 40.0_f64), ([1.0, 1.0, 1.0], 180.0)] {
        let q = rotation_about(axis, angle_deg.to_radians());
        let dst = apply_rigid(q, [3.0, 0.0, 0.0], &src);
        let (r, _) = IcpSolver::best_fit_transform(&src, &dst);
        assert_proper_rotation(r, "collinear");
        let mapped = mat3_mv(r, direction);
        let expected = mat3_mv(q, direction);
        assert!(
            v3_norm(v3_sub(mapped, expected)) < TOL,
            "line direction maps to {mapped:?}, expected {expected:?}"
        );
        // The alignment is exact although the twist is arbitrary.
        let (r2, t2) = IcpSolver::best_fit_transform(&src, &dst);
        let aligned = apply_rigid(r2, t2, &src);
        for (a, b) in aligned.iter().zip(dst.iter()) {
            assert!(v3_norm(v3_sub(*a, *b)) < TOL);
        }
    }

    // Antiparallel directions: a half turn is required.
    let src = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]];
    let dst = vec![[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-2.5, 0.0, 0.0]];
    let (r, _) = IcpSolver::best_fit_transform(&src, &dst);
    assert_proper_rotation(r, "antiparallel");
    let mapped = mat3_mv(r, [1.0, 0.0, 0.0]);
    assert!(
        v3_norm(v3_sub(mapped, [-1.0, 0.0, 0.0])) < TOL,
        "mapped={mapped:?}"
    );
}

/// A mirrored target cannot be matched by a rotation.  The result must still be
/// a proper rotation (never a reflection) and the best one: no random rotation
/// achieves a lower cost.
#[test]
fn kabsch_never_returns_a_reflection_and_is_optimal() {
    let src = asymmetric_cloud();
    let mirrored: Vec<[f64; 3]> = src.iter().map(|p| [-p[0], p[1], p[2]]).collect();
    let (r, _) = IcpSolver::best_fit_transform(&src, &mirrored);
    assert_proper_rotation(r, "mirrored");

    let c_src = centroid(&src);
    let c_dst = centroid(&mirrored);
    let a: Vec<[f64; 3]> = src.iter().map(|p| v3_sub(*p, c_src)).collect();
    let b: Vec<[f64; 3]> = mirrored.iter().map(|p| v3_sub(*p, c_dst)).collect();
    let best = alignment_cost(r, &a, &b);
    assert!(best > 1e-3, "a mirror image cannot be matched exactly");

    let mut rng = Lcg(2024);
    for _ in 0..400 {
        let axis = rng.next_point(1.0);
        if v3_norm(axis) < 1e-3 {
            continue;
        }
        let angle = std::f64::consts::PI * rng.next_unit();
        let candidate = alignment_cost(rotation_about(axis, angle), &a, &b);
        assert!(
            best <= candidate + 1e-9,
            "random rotation beats Kabsch: {candidate} < {best}"
        );
    }
}

/// Random full-rank clouds and rotations over many trials.
#[test]
fn kabsch_recovers_random_rotations() {
    let mut rng = Lcg(31337);
    for trial in 0..60 {
        let src: Vec<[f64; 3]> = (0..8).map(|_| rng.next_point(3.0)).collect();
        let axis = rng.next_point(1.0);
        if v3_norm(axis) < 1e-2 {
            continue;
        }
        let q = rotation_about(axis, std::f64::consts::PI * rng.next_unit());
        let dst = apply_rigid(q, rng.next_point(5.0), &src);
        let (r, _) = IcpSolver::best_fit_transform(&src, &dst);
        assert_proper_rotation(r, "random");
        assert!(max_abs_diff(r, q) < TOL, "trial {trial}: {r:?} vs {q:?}");
    }
}

/// The rotation does not depend on the length scale of the data (the 3x3 SVD
/// underneath uses absolute thresholds).
#[test]
fn kabsch_is_independent_of_length_scale() {
    let q = rotation_about([0.5, -1.0, 2.0], 2.4);
    for scale in [1e-7, 1e-3, 1.0, 1e5] {
        let src: Vec<[f64; 3]> = asymmetric_cloud()
            .iter()
            .map(|p| v3_scale(*p, scale))
            .collect();
        let dst = apply_rigid(q, [0.0; 3], &src);
        let (r, _) = IcpSolver::best_fit_transform(&src, &dst);
        assert!(max_abs_diff(r, q) < TOL, "scale {scale}: {r:?}");
    }
}

/// With no spread at all there is no orientation information: identity.
#[test]
fn kabsch_of_zero_covariance_is_identity() {
    let r = kabsch_rotation(mat3_zero());
    assert!(max_abs_diff(r, mat3_identity()) < TOL);
    let same = vec![[1.0, 2.0, 3.0]; 4];
    let (r, t) = IcpSolver::best_fit_transform(&same, &same);
    assert!(max_abs_diff(r, mat3_identity()) < TOL);
    assert!(v3_norm(t) < TOL);
}

/// `kabsch(H)` is the transpose of the polar factor of `H`, i.e. a rotation
/// `Q` is recovered from `H = Q⁻¹ S` (`S` symmetric positive definite).
#[test]
fn kabsch_inverts_the_polar_rotation_of_the_covariance() {
    let q = rot_z(90.0_f64.to_radians());
    // H = C Qᵀ with C = Σ e eᵀ symmetric positive definite.
    let c = [[3.0, 0.4, 0.1], [0.4, 2.0, -0.3], [0.1, -0.3, 1.0]];
    let h = mat3_mul(c, mat3_transpose(q));
    let r = kabsch_rotation(h);
    assert!(
        max_abs_diff(r, q) < TOL,
        "Kabsch of C·Qᵀ must be Q, got {r:?}"
    );
    // A 90 degree z-rotation must not be collapsed to diag(0, 0, 1).
    assert!((r[0][1] + 1.0).abs() < TOL && (r[1][0] - 1.0).abs() < TOL);
}

// ---------------------------------------------------------------------------
// ICP
// ---------------------------------------------------------------------------

/// Nearest-neighbour ICP is a local method: for a small misalignment of a
/// well-spread asymmetric cloud it recovers the pose exactly.
#[test]
fn icp_recovers_small_rotation_and_translation() {
    let src = asymmetric_cloud();
    let q = rotation_about([1.0, 2.0, 3.0], 8.0_f64.to_radians());
    let t = [0.05, -0.03, 0.02];
    let dst = apply_rigid(q, t, &src);
    let solver = IcpSolver::new(100, 1e-18);
    let result = solver.register(&src, &dst);
    assert!(result.mse < 1e-18, "mse={}", result.mse);
    assert!(
        max_abs_diff(result.rotation, q) < 1e-7,
        "rotation {:?}",
        result.rotation
    );
    for (a, b) in result.translation.iter().zip(t.iter()) {
        assert!((a - b).abs() < 1e-7, "translation {:?}", result.translation);
    }
}

/// A pure translation is recovered to full precision by the very first step.
#[test]
fn icp_recovers_pure_translation_exactly() {
    let src = asymmetric_cloud();
    let t = [0.3, -0.2, 0.25];
    let dst = apply_rigid(mat3_identity(), t, &src);
    let result = IcpSolver::new(50, 1e-20).register(&src, &dst);
    assert!(result.mse < 1e-20, "mse={}", result.mse);
    assert!(max_abs_diff(result.rotation, mat3_identity()) < 1e-9);
    for (a, b) in result.translation.iter().zip(t.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// ARAP
// ---------------------------------------------------------------------------

/// An asymmetric 1-ring of vertex 0 (vertices 1..=5) in 3-D.
fn asymmetric_ring_rest() -> Vec<[f64; 3]> {
    vec![
        [0.1, 0.05, 0.0],
        [1.0, 0.0, 0.2],
        [0.3, 1.2, -0.1],
        [-0.8, 0.5, 0.3],
        [-0.6, -0.9, 0.1],
        [0.5, -0.7, -0.2],
    ]
}

fn ring_neighbours() -> Vec<usize> {
    vec![1, 2, 3, 4, 5]
}

/// `ArapCell::update_rotation` on a 1-ring that has been rotated by `Q` about
/// the centre returns `Q` — for uniform and for cotangent weights.
#[test]
fn arap_cell_rotation_recovers_rotated_ring() {
    let rest = asymmetric_ring_rest();
    let triangles = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1]];
    let cells = [
        ArapCell::new(0, ring_neighbours(), &rest),
        ArapCell::from_triangles(0, &triangles, &rest),
    ];
    for cell in cells {
        for (axis, angle_deg) in [
            ([0.0, 0.0, 1.0], 30.0_f64),
            ([0.0, 0.0, 1.0], 170.0),
            ([1.0, -2.0, 0.5], 75.0),
        ] {
            let q = rotation_about(axis, angle_deg.to_radians());
            let center = rest[0];
            let deformed: Vec<[f64; 3]> = rest
                .iter()
                .map(|p| v3_add(center, mat3_mv(q, v3_sub(*p, center))))
                .collect();
            let mut cell = cell.clone();
            cell.update_rotation(&deformed);
            assert_proper_rotation(cell.rotation, "arap cell");
            assert!(
                max_abs_diff(cell.rotation, q) < TOL,
                "axis {axis:?}, {angle_deg} deg: {:?}",
                cell.rotation
            );
        }
    }
}

/// At rest the best-fit rotation is the identity and the global step is a
/// fixed point (an asymmetric ring would be moved by the old `+ R e` update).
#[test]
fn arap_rest_pose_is_a_fixed_point() {
    let rest = asymmetric_ring_rest();
    let cell = ArapCell::new(0, ring_neighbours(), &rest);
    let mut deformer = ArapDeformer::new(rest.clone(), vec![cell], vec![1, 2, 3, 4, 5]);
    deformer.solve(5);
    assert!(max_abs_diff(deformer.cells[0].rotation, mat3_identity()) < TOL);
    for (p, q) in deformer.positions.iter().zip(rest.iter()) {
        assert!(v3_norm(v3_sub(*p, *q)) < 1e-12, "{p:?} moved from {q:?}");
    }
}

/// When the pinned ring is rotated rigidly about the centre's rest position
/// the centre must stay where it is: `p_i = Σ w (p_j − R e_j) / W` cancels
/// exactly (with the wrong sign the centre was thrown off by `2 R ē`).
#[test]
fn arap_global_step_has_the_correct_sign() {
    let rest = asymmetric_ring_rest();
    let q = rot_z(90.0_f64.to_radians());
    let center = rest[0];
    let mut moved = rest.clone();
    for p in moved.iter_mut().skip(1) {
        *p = v3_add(center, mat3_mv(q, v3_sub(*p, center)));
    }
    let cell = ArapCell::new(0, ring_neighbours(), &rest);
    let mut deformer = ArapDeformer::new(moved, vec![cell], vec![1, 2, 3, 4, 5]);
    deformer.jacobi_iter = 1;
    deformer.step();
    assert!(max_abs_diff(deformer.cells[0].rotation, q) < TOL);
    let c = deformer.positions[0];
    assert!(
        v3_norm(v3_sub(c, center)) < 1e-12,
        "centre moved to {c:?}, expected {center:?}"
    );
}

/// A rigidly moved pinned ring drags the free centre vertex along: the
/// local-global iteration converges to `Q c⁰ + t` with the cell rotation `Q`.
#[test]
fn arap_free_vertex_follows_rigid_motion_of_its_ring() {
    let rest = asymmetric_ring_rest();
    for (axis, angle_deg) in [([0.0, 0.0, 1.0], 30.0_f64), ([1.0, 2.0, -1.0], 100.0)] {
        let q = rotation_about(axis, angle_deg.to_radians());
        let t = [0.4, -0.2, 0.9];
        let mut moved = apply_rigid(q, t, &rest);
        moved[0] = rest[0]; // the free vertex starts at its rest position
        let cell = ArapCell::new(0, ring_neighbours(), &rest);
        let mut deformer = ArapDeformer::new(moved, vec![cell], vec![1, 2, 3, 4, 5]);
        deformer.solve(200);
        let expected = v3_add(mat3_mv(q, rest[0]), t);
        let err = v3_norm(v3_sub(deformer.positions[0], expected));
        assert!(err < 1e-9, "axis {axis:?}: centre error {err}");
        assert!(max_abs_diff(deformer.cells[0].rotation, q) < 1e-8);
    }
}

/// Triangulated `(n+1) x (n+1)` grid in the plane `z = 0`; every square is
/// split along the same diagonal.
fn grid_mesh(n: usize, spacing: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let idx = |i: usize, j: usize| i * (n + 1) + j;
    let mut pos = Vec::new();
    for i in 0..=n {
        for j in 0..=n {
            pos.push([i as f64 * spacing, j as f64 * spacing, 0.0]);
        }
    }
    let mut tris = Vec::new();
    for i in 0..n {
        for j in 0..n {
            tris.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
            tris.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
        }
    }
    (pos, tris)
}

/// A mesh patch whose boundary is moved rigidly follows the rigid motion
/// everywhere (the energy-free solution); interior vertices have cotangent
/// weights and neighbours with and without cells.
#[test]
fn arap_patch_follows_rigid_motion_of_its_boundary() {
    let n = 4;
    let (rest, tris) = grid_mesh(n, 0.5);
    let on_boundary = |i: usize, j: usize| i == 0 || j == 0 || i == n || j == n;
    let mut pinned = Vec::new();
    let mut cells = Vec::new();
    for i in 0..=n {
        for j in 0..=n {
            let v = i * (n + 1) + j;
            if on_boundary(i, j) {
                pinned.push(v);
            } else {
                cells.push(ArapCell::from_triangles(v, &tris, &rest));
            }
        }
    }
    assert_eq!(cells.len(), (n - 1) * (n - 1));
    // every interior vertex of the grid has six incident triangles
    assert!(cells.iter().all(|c| c.neighbours.len() == 6));

    for (axis, angle_deg) in [([0.0, 0.0, 1.0], 40.0_f64), ([1.0, 0.5, 0.2], 25.0)] {
        let q = rotation_about(axis, angle_deg.to_radians());
        let t = [0.3, 0.1, -0.2];
        let target = apply_rigid(q, t, &rest);
        let mut start = rest.clone();
        for &b in &pinned {
            start[b] = target[b];
        }
        let mut deformer = ArapDeformer::new(start, cells.clone(), pinned.clone());
        deformer.solve(300);
        for cell in &deformer.cells {
            let v = cell.center;
            let err = v3_norm(v3_sub(deformer.positions[v], target[v]));
            assert!(err < 1e-6, "axis {axis:?}: vertex {v} off by {err}");
            assert!(
                max_abs_diff(cell.rotation, q) < 1e-5,
                "axis {axis:?}: cell {v} rotation {:?}",
                cell.rotation
            );
        }
    }
}

/// Pinned vertices never move, also in the presence of free neighbours.
#[test]
fn arap_keeps_pinned_vertices_fixed_while_free_ones_relax() {
    let n = 3;
    let (rest, tris) = grid_mesh(n, 1.0);
    let mut cells = Vec::new();
    let mut pinned = Vec::new();
    for i in 0..=n {
        for j in 0..=n {
            let v = i * (n + 1) + j;
            if i == 0 || j == 0 || i == n || j == n {
                pinned.push(v);
            } else {
                cells.push(ArapCell::from_triangles(v, &tris, &rest));
            }
        }
    }
    let mut start = rest.clone();
    // Lift the boundary vertex (0, 1), which shares a grid edge with the
    // interior vertex (1, 1): the interior has to react, the boundary not.
    let lifted = 1;
    assert!(pinned.contains(&lifted));
    start[lifted][2] = 0.8;
    let mut deformer = ArapDeformer::new(start.clone(), cells, pinned.clone());
    deformer.solve(50);
    for &b in &pinned {
        assert_eq!(deformer.positions[b], start[b], "pinned vertex {b} moved");
    }
    let moved = deformer
        .cells
        .iter()
        .map(|c| v3_norm(v3_sub(deformer.positions[c.center], rest[c.center])))
        .fold(0.0_f64, f64::max);
    assert!(moved > 1e-3, "interior did not react (max move {moved})");
}

// ---------------------------------------------------------------------------
// Cotangent weights
// ---------------------------------------------------------------------------

/// Six equilateral triangles around a vertex: every edge has the weight
/// `½ (cot 60° + cot 60°) = cot 60°`.
#[test]
fn cotangent_weights_of_an_equilateral_fan() {
    let mut pos = vec![[0.0, 0.0, 0.0]];
    for k in 0..6 {
        let a = std::f64::consts::FRAC_PI_3 * k as f64;
        pos.push([a.cos(), a.sin(), 0.0]);
    }
    let tris: Vec<[usize; 3]> = (0..6).map(|k| [0, 1 + k, 1 + (k + 1) % 6]).collect();
    let cell = ArapCell::from_triangles(0, &tris, &pos);
    assert_eq!(cell.neighbours.len(), 6);
    let cot60 = 1.0 / 3.0_f64.sqrt();
    for w in &cell.weights {
        assert!((w - cot60).abs() < TOL, "w={w}");
    }
}

/// One right isosceles triangle: both edges at the right angle see one
/// opposite 45° angle, i.e. `w = ½ cot 45° = ½`.
#[test]
fn cotangent_weights_of_a_single_triangle() {
    let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let cell = ArapCell::from_triangles(0, &[[0, 1, 2]], &pos);
    assert_eq!(cell.neighbours, vec![1, 2]);
    for w in &cell.weights {
        assert!((w - 0.5).abs() < TOL, "w={w}");
    }
    // Triangles that do not contain the centre vertex are ignored.
    let ignored = ArapCell::from_triangles(
        0,
        &[[1, 2, 0], [1, 2, 3]],
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
    );
    assert_eq!(ignored.neighbours.len(), 2);
}

/// The weight of an interior edge is the same seen from both end vertices, and
/// obtuse triangles (negative cotangent sums) are floored at a positive weight.
#[test]
fn cotangent_weights_are_symmetric_and_positive() {
    let (pos, tris) = grid_mesh(2, 1.0);
    let centre = 4; // the single interior vertex of the 3x3 vertex grid
    let cell_c = ArapCell::from_triangles(centre, &tris, &pos);
    for (&nb, &w_c) in cell_c.neighbours.iter().zip(cell_c.weights.iter()) {
        let cell_n = ArapCell::from_triangles(nb, &tris, &pos);
        let k = cell_n
            .neighbours
            .iter()
            .position(|&v| v == centre)
            .expect("neighbour relation is symmetric");
        // Boundary vertices see the shared edge in the same two triangles.
        assert!(
            (cell_n.weights[k] - w_c).abs() < TOL,
            "w({centre},{nb})={w_c} vs w({nb},{centre})={}",
            cell_n.weights[k]
        );
        assert!(w_c >= MIN_EDGE_WEIGHT);
    }

    // Strongly obtuse triangle: the cotangent of the obtuse angle is negative.
    let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.05, 0.0]];
    let cell = ArapCell::from_triangles(1, &[[1, 2, 0]], &pos);
    for w in &cell.weights {
        assert!(*w >= MIN_EDGE_WEIGHT, "weight {w} below the floor");
    }
}

// ---------------------------------------------------------------------------
// RBF
// ---------------------------------------------------------------------------

fn random_non_coplanar_points(rng: &mut Lcg, count: usize, half_extent: f64) -> Vec<[f64; 3]> {
    // Rejection of near-duplicates keeps the systems well conditioned.
    let mut pts: Vec<[f64; 3]> = Vec::new();
    while pts.len() < count {
        let p = rng.next_point(half_extent);
        if pts.iter().all(|q| v3_norm(v3_sub(*q, p)) > 0.6) {
            pts.push(p);
        }
    }
    pts
}

/// Every kernel interpolates the prescribed displacements at the control
/// points.  For the thin-plate kernel the old Gauss-Seidel sweep skipped every
/// row (`φ(0) = 0`) and returned all-zero coefficients.
#[test]
fn rbf_interpolates_control_displacements_for_every_kernel() {
    let mut rng = Lcg(424242);
    let pts = random_non_coplanar_points(&mut rng, 10, 2.0);
    let disp: Vec<[f64; 3]> = (0..10).map(|_| rng.next_point(0.5)).collect();
    for kernel in [
        RbfKernel::ThinPlate,
        RbfKernel::Multiquadric(1.0),
        RbfKernel::Gaussian(0.5),
    ] {
        let rbf = RbfDeformer::new(pts.clone(), disp.clone(), kernel)
            .unwrap_or_else(|e| panic!("{kernel:?}: {e}"));
        for (q, d) in pts.iter().zip(disp.iter()) {
            let got = rbf.evaluate(*q);
            let err = v3_norm(v3_sub(got, *d));
            assert!(err < 1e-9, "{kernel:?}: at {q:?} got {got:?}, want {d:?}");
        }
    }
}

/// The thin-plate deformation is not silently zero away from the controls.
#[test]
fn rbf_thin_plate_deforms_between_control_points() {
    let pts = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 2.0, 2.0],
    ];
    let disp = vec![
        [0.0; 3],
        [0.0, 0.5, 0.0],
        [0.0; 3],
        [0.0; 3],
        [0.0, 0.0, -0.4],
    ];
    let rbf = RbfDeformer::new(pts, disp, RbfKernel::ThinPlate).expect("valid thin-plate system");
    let mid = rbf.evaluate([1.0, 0.2, 0.1]);
    assert!(v3_norm(mid) > 1e-3, "deformation vanished: {mid:?}");
    let deformed = rbf.deform_mesh(&[[2.0, 0.0, 0.0]]);
    assert!((deformed[0][1] - 0.5).abs() < 1e-9, "{:?}", deformed[0]);
}

/// The affine term makes the thin-plate spline reproduce any affine
/// displacement field exactly, also away from the control points.
#[test]
fn rbf_thin_plate_reproduces_affine_fields() {
    let mut rng = Lcg(99);
    let pts = random_non_coplanar_points(&mut rng, 12, 2.0);
    let field = |p: [f64; 3]| -> [f64; 3] {
        [
            0.3 + 0.1 * p[0] - 0.2 * p[1] + 0.05 * p[2],
            -0.4 + 0.25 * p[0] + 0.1 * p[1],
            0.2 - 0.15 * p[2] + 0.3 * p[1],
        ]
    };
    let disp: Vec<[f64; 3]> = pts.iter().map(|p| field(*p)).collect();
    let rbf = RbfDeformer::new(pts, disp, RbfKernel::ThinPlate).expect("valid system");
    for _ in 0..20 {
        let p = rng.next_point(2.5);
        let err = v3_norm(v3_sub(rbf.evaluate(p), field(p)));
        assert!(err < 1e-8, "affine field not reproduced at {p:?}: {err}");
    }
}

/// The multiquadric matrix is not positive definite, so the old Gauss-Seidel
/// iteration was not guaranteed to converge; the direct solve is.
#[test]
fn rbf_multiquadric_with_many_control_points() {
    let mut rng = Lcg(5150);
    let pts = random_non_coplanar_points(&mut rng, 30, 3.0);
    let disp: Vec<[f64; 3]> = (0..30).map(|_| rng.next_point(1.0)).collect();
    let rbf = RbfDeformer::new(pts.clone(), disp.clone(), RbfKernel::Multiquadric(0.8))
        .expect("multiquadric system is nonsingular");
    for (q, d) in pts.iter().zip(disp.iter()) {
        let err = v3_norm(v3_sub(rbf.evaluate(*q), *d));
        assert!(err < 1e-8, "residual {err} at {q:?}");
    }
}

#[test]
fn rbf_reports_singular_systems() {
    // Duplicated control point.
    let pts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let disp = vec![[0.0; 3], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert_eq!(
        RbfDeformer::new(pts, disp, RbfKernel::Gaussian(1.0)).err(),
        Some(RbfError::Singular)
    );

    // Thin-plate with coplanar control points (the affine block is rank deficient).
    let coplanar = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.5, 0.0],
    ];
    let disp = vec![[0.1; 3]; 5];
    assert_eq!(
        RbfDeformer::new(coplanar, disp, RbfKernel::ThinPlate).err(),
        Some(RbfError::Singular)
    );

    // Thin-plate with fewer than four control points.
    let few = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    assert_eq!(
        RbfDeformer::new(few, vec![[0.0; 3]; 3], RbfKernel::ThinPlate).err(),
        Some(RbfError::Singular)
    );
}

#[test]
fn rbf_validates_its_input() {
    let err = RbfDeformer::new(
        vec![[0.0; 3], [1.0, 0.0, 0.0]],
        vec![[0.0; 3]],
        RbfKernel::Multiquadric(1.0),
    )
    .err();
    assert_eq!(
        err,
        Some(RbfError::LengthMismatch {
            control_points: 2,
            displacements: 1
        })
    );
    let err = RbfDeformer::new(
        vec![[f64::NAN, 0.0, 0.0]],
        vec![[0.0; 3]],
        RbfKernel::Multiquadric(1.0),
    )
    .err();
    assert_eq!(err, Some(RbfError::NonFinite));
    // No control points: a valid (identity) deformation.
    let empty = RbfDeformer::new(Vec::new(), Vec::new(), RbfKernel::Gaussian(1.0))
        .expect("empty system is valid");
    assert_eq!(empty.evaluate([1.0, 2.0, 3.0]), [0.0; 3]);
}

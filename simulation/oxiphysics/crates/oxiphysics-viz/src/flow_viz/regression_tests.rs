//! Regression tests for the eigenvalue machinery of the flow-field module:
//! the symmetric 3x3 solver behind the λ2 criterion and the eigenvalue-based
//! classification of critical points.

use super::*;

type Mat3 = [f64; 9];

/// Deterministic xorshift generator returning values in `[-1, 1)`.
struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn matrix(&mut self) -> Mat3 {
        std::array::from_fn(|_| self.next())
    }
}

fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|idx| {
        let (i, j) = (idx / 3, idx % 3);
        (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum()
    })
}

fn transpose(a: &Mat3) -> Mat3 {
    std::array::from_fn(|idx| a[(idx % 3) * 3 + idx / 3])
}

/// `r * m * r^T`.
fn conjugate(r: &Mat3, m: &Mat3) -> Mat3 {
    mat_mul(&mat_mul(r, m), &transpose(r))
}

fn det3(m: &Mat3) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Rotation matrix about the unit `axis` by `angle` (Rodrigues' formula).
fn rotation(axis: [f64; 3], angle: f64) -> Mat3 {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let [x, y, z] = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let (c, s) = (angle.cos(), angle.sin());
    let t = 1.0 - c;
    [
        t * x * x + c,
        t * x * y - s * z,
        t * x * z + s * y,
        t * x * y + s * z,
        t * y * y + c,
        t * y * z - s * x,
        t * x * z - s * y,
        t * y * z + s * x,
        t * z * z + c,
    ]
}

fn diag(a: f64, b: f64, c: f64) -> Mat3 {
    [a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c]
}

fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: {actual} vs {expected} (tolerance {tol})"
    );
}

// ---------------------------------------------------------------------------
// Symmetric 3x3 eigenvalues and the λ2 criterion
// ---------------------------------------------------------------------------

#[test]
fn symmetric_eigenvalues_of_a_block_with_a_distinct_third_axis() {
    // The spectrum of [[1,1,0],[1,1,0],[0,0,5]] is {0, 2, 5}.
    let m = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0];
    let eigs = eigenvalues_3x3_sym(&m);
    for (actual, expected) in eigs.iter().zip([0.0, 2.0, 5.0]) {
        assert_close(*actual, expected, 1e-10, "eigenvalue");
    }
    assert_close(
        middle_eigenvalue_3x3_sym(&m),
        2.0,
        1e-10,
        "middle eigenvalue",
    );
}

#[test]
fn symmetric_eigenvalues_of_the_all_ones_matrix() {
    // Spectrum {0, 0, 3}: the middle eigenvalue is 0, not the reflected -1.
    let m = [1.0; 9];
    let eigs = eigenvalues_3x3_sym(&m);
    for (actual, expected) in eigs.iter().zip([0.0, 0.0, 3.0]) {
        assert_close(*actual, expected, 1e-10, "eigenvalue");
    }
    assert_close(
        middle_eigenvalue_3x3_sym(&m),
        0.0,
        1e-10,
        "middle eigenvalue",
    );
}

#[test]
fn symmetric_eigenvalues_are_invariant_under_rotation() {
    let rot = rotation([1.0, 2.0, 3.0], 0.9);
    let m = conjugate(&rot, &diag(1.0, 3.0, 7.0));
    let eigs = eigenvalues_3x3_sym(&m);
    for (actual, expected) in eigs.iter().zip([1.0, 3.0, 7.0]) {
        assert_close(*actual, expected, 1e-10, "eigenvalue");
    }
    assert_close(
        middle_eigenvalue_3x3_sym(&m),
        3.0,
        1e-10,
        "middle eigenvalue",
    );
}

#[test]
fn symmetric_eigenvalues_of_diagonal_matrices_are_sorted() {
    let eigs = eigenvalues_3x3_sym(&diag(4.0, -2.0, 1.0));
    assert_eq!(eigs, [-2.0, 1.0, 4.0]);
    assert_eq!(eigenvalues_3x3_sym(&[0.0; 9]), [0.0; 3]);
    assert!(
        eigenvalues_3x3_sym(&diag(f64::NAN, 1.0, 2.0))
            .iter()
            .all(|e| e.is_nan())
    );
    assert!(
        eigenvalues_3x3_sym(&diag(f64::INFINITY, 1.0, 2.0))
            .iter()
            .all(|e| e.is_nan())
    );
}

#[test]
fn symmetric_eigenvalues_with_a_double_root() {
    // I + 2 n n^T with |n| = 1 has the spectrum {1, 1, 3}.
    let n = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
    let m: Mat3 = std::array::from_fn(|idx| {
        let (i, j) = (idx / 3, idx % 3);
        2.0 * n[i] * n[j] + if i == j { 1.0 } else { 0.0 }
    });
    let eigs = eigenvalues_3x3_sym(&m);
    for (actual, expected) in eigs.iter().zip([1.0, 1.0, 3.0]) {
        assert_close(*actual, expected, 1e-7, "eigenvalue");
    }
}

#[test]
fn symmetric_eigenvalues_scale_with_the_matrix() {
    let rot = rotation([0.3, -1.0, 0.5], 0.4);
    let base = conjugate(&rot, &diag(-2.0, 0.5, 6.0));
    for factor in [1e-30, 1e-12, 1e12, 1e100] {
        let m: Mat3 = base.map(|v| v * factor);
        let eigs = eigenvalues_3x3_sym(&m);
        for (actual, expected) in eigs.iter().zip([-2.0, 0.5, 6.0]) {
            assert_close(*actual / factor, expected, 1e-9, "scaled eigenvalue");
        }
    }
}

#[test]
fn symmetric_eigenvalues_match_the_characteristic_polynomial() {
    let mut rng = TestRng(0x9E37_79B9_7F4A_7C15);
    for case in 0..300 {
        let raw = rng.matrix();
        // Symmetrise.
        let m: Mat3 = std::array::from_fn(|idx| 0.5 * (raw[idx] + raw[(idx % 3) * 3 + idx / 3]));
        let eigs = eigenvalues_3x3_sym(&m);
        assert!(
            eigs[0] <= eigs[1] && eigs[1] <= eigs[2],
            "case {case}: {eigs:?}"
        );
        let trace = m[0] + m[4] + m[8];
        assert_close(eigs.iter().sum::<f64>(), trace, 1e-12, "trace");
        assert_close(eigs.iter().product::<f64>(), det3(&m), 1e-11, "determinant");
        for lambda in eigs {
            let shifted: Mat3 =
                std::array::from_fn(|idx| m[idx] - if idx % 4 == 0 { lambda } else { 0.0 });
            assert!(
                det3(&shifted).abs() < 1e-11,
                "case {case}: det(M - {lambda} I) = {}",
                det3(&shifted)
            );
        }
    }
}

/// Fill a field with the linear velocity `u = g x` (x measured from the node
/// index times `dx`).
fn linear_field(n: usize, dx: f64, g: &Mat3) -> FlowField3D {
    let mut field = FlowField3D::new(n, n, n, dx);
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let x = [ix as f64 * dx, iy as f64 * dx, iz as f64 * dx];
                let u: [f64; 3] =
                    std::array::from_fn(|c| (0..3).map(|a| g[c * 3 + a] * x[a]).sum());
                field.set_vel(ix as i64, iy as i64, iz as i64, u);
            }
        }
    }
    field
}

#[test]
fn lambda2_of_a_rotation_dominated_gradient_is_negative() {
    // u = (y, z, x): the symmetric matrix S^2 + Omega^2 is
    // 0.5 [[0,1,1],[1,0,1],[1,1,0]] with the spectrum {-1/2, -1/2, 1}, so
    // lambda2 = -1/2 < 0 flags a vortex.
    let g = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    let field = linear_field(5, 0.5, &g);
    let l2 = lambda2_criterion(&field);
    for iz in 1..4 {
        for iy in 1..4 {
            for ix in 1..4 {
                let value = l2[field.idx_clamped(ix, iy, iz)];
                assert_close(value, -0.5, 1e-6, "lambda2");
            }
        }
    }
}

#[test]
fn lambda2_of_pure_strain_is_positive() {
    // u = (x, 0, -z): Omega = 0 and S^2 = diag(1, 0, 1), so lambda2 = 1.
    let field = linear_field(5, 0.5, &diag(1.0, 0.0, -1.0));
    let l2 = lambda2_criterion(&field);
    assert_close(l2[field.idx_clamped(2, 2, 2)], 1.0, 1e-10, "lambda2");
}

#[test]
fn lambda2_of_a_general_gradient_is_the_middle_eigenvalue_of_s2_plus_omega2() {
    // Compare against the eigenvalues of S^2 + Omega^2 computed from the
    // definition for a gradient with all nine components.
    let g = [0.3, -1.2, 0.7, 0.9, -0.4, 1.1, -0.6, 0.5, 0.1];
    let gt = transpose(&g);
    let s: Mat3 = std::array::from_fn(|i| 0.5 * (g[i] + gt[i]));
    let omega: Mat3 = std::array::from_fn(|i| 0.5 * (g[i] - gt[i]));
    let s2 = mat_mul(&s, &s);
    let o2 = mat_mul(&omega, &omega);
    let sum: Mat3 = std::array::from_fn(|i| s2[i] + o2[i]);
    let expected = eigenvalues_3x3_sym(&sum)[1];

    let field = linear_field(5, 0.5, &g);
    let l2 = lambda2_criterion(&field);
    assert_close(l2[field.idx_clamped(2, 2, 2)], expected, 1e-10, "lambda2");
}

// ---------------------------------------------------------------------------
// General 3x3 eigenvalues
// ---------------------------------------------------------------------------

#[test]
fn eigenvalues_of_a_triangular_matrix_are_its_diagonal() {
    let g = [2.0, 5.0, -1.0, 0.0, -3.0, 4.0, 0.0, 0.0, 7.0];
    match eigenvalues_3x3(&g) {
        Eigenvalues3::Real(values) => {
            for (actual, expected) in values.iter().zip([-3.0, 2.0, 7.0]) {
                assert_close(*actual, expected, 1e-10, "eigenvalue");
            }
        }
        other => panic!("expected three real eigenvalues, got {other:?}"),
    }
}

#[test]
fn eigenvalues_of_a_spiral_block_are_a_pair_and_a_real_root() {
    let g = [-1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, -1.0];
    match eigenvalues_3x3(&g) {
        Eigenvalues3::Complex { real, re, im } => {
            assert_close(real, -1.0, 1e-12, "real eigenvalue");
            assert_close(re, -1.0, 1e-12, "pair real part");
            assert_close(im, 2.0, 1e-12, "pair imaginary part");
        }
        other => panic!("expected a complex pair, got {other:?}"),
    }
}

#[test]
fn eigenvalues_of_a_rotation_matrix_lie_on_the_unit_circle() {
    let angle = 0.7;
    let g = rotation([1.0, -2.0, 0.5], angle);
    match eigenvalues_3x3(&g) {
        Eigenvalues3::Complex { real, re, im } => {
            assert_close(real, 1.0, 1e-12, "axis eigenvalue");
            assert_close(re, angle.cos(), 1e-12, "cos");
            assert_close(im, angle.sin(), 1e-12, "sin");
        }
        other => panic!("expected a complex pair, got {other:?}"),
    }
}

#[test]
fn eigenvalues_satisfy_the_characteristic_polynomial_identities() {
    let mut rng = TestRng(0xD1B5_4A32_D192_ED03);
    let (mut real_cases, mut complex_cases) = (0, 0);
    for case in 0..500 {
        let g = rng.matrix();
        let i1 = g[0] + g[4] + g[8];
        let i2 =
            (g[0] * g[4] - g[1] * g[3]) + (g[0] * g[8] - g[2] * g[6]) + (g[4] * g[8] - g[5] * g[7]);
        let i3 = det3(&g);
        let (trace, minors, det) = match eigenvalues_3x3(&g) {
            Eigenvalues3::Real([a, b, c]) => {
                real_cases += 1;
                assert!(a <= b && b <= c, "case {case}: not ascending");
                (a + b + c, a * b + a * c + b * c, a * b * c)
            }
            Eigenvalues3::Complex { real, re, im } => {
                complex_cases += 1;
                assert!(im > 0.0, "case {case}: im = {im}");
                let modulus_sq = re * re + im * im;
                (
                    real + 2.0 * re,
                    2.0 * real * re + modulus_sq,
                    real * modulus_sq,
                )
            }
        };
        assert_close(trace, i1, 1e-10, "trace");
        assert_close(minors, i2, 1e-9, "sum of principal minors");
        assert_close(det, i3, 1e-9, "determinant");
    }
    assert!(
        real_cases > 50 && complex_cases > 50,
        "{real_cases}/{complex_cases}"
    );
}

// ---------------------------------------------------------------------------
// Critical point classification
// ---------------------------------------------------------------------------

#[test]
fn nodes_and_saddles_follow_the_signs_of_the_eigenvalues() {
    assert_eq!(
        classify_from_gradient(&diag(-1.0, -1.0, -1.0)),
        CriticalPointType::AttractingNode,
        "-I is a three-dimensional sink"
    );
    assert_eq!(
        classify_from_gradient(&diag(1.0, 1.0, 1.0)),
        CriticalPointType::RepellingNode
    );
    assert_eq!(
        classify_from_gradient(&diag(1.0, -1.0, -1.0)),
        CriticalPointType::Saddle,
        "diag(1, -1, -1) has mixed signs"
    );
    assert_eq!(
        classify_from_gradient(&diag(-1.0, -2.0, -3.0)),
        CriticalPointType::AttractingNode
    );
    assert_eq!(
        classify_from_gradient(&diag(0.5, 2.0, -3.0)),
        CriticalPointType::Saddle
    );
}

#[test]
fn spirals_follow_the_real_parts_of_the_eigenvalues() {
    let sink = [-1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, -1.0];
    assert_eq!(
        classify_from_gradient(&sink),
        CriticalPointType::AttractingFocus
    );
    let source = [1.0, -2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_eq!(
        classify_from_gradient(&source),
        CriticalPointType::RepellingFocus
    );
    // Outward spiral in the plane, inward along the axis.
    let saddle_focus = [1.0, -2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0];
    assert_eq!(
        classify_from_gradient(&saddle_focus),
        CriticalPointType::Saddle
    );
    // Closed orbits in the plane.
    let center = [0.0, -2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -1.0];
    assert_eq!(classify_from_gradient(&center), CriticalPointType::Center);
}

#[test]
fn zero_eigenvalues_and_invalid_gradients_are_degenerate() {
    assert_eq!(
        classify_from_gradient(&[0.0; 9]),
        CriticalPointType::Degenerate
    );
    // Spiral in the plane with a neutral axis.
    let neutral_axis = [-1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(
        classify_from_gradient(&neutral_axis),
        CriticalPointType::Degenerate
    );
    // Rank-deficient: the spectrum is {0, 0, 3}.
    assert_eq!(
        classify_from_gradient(&[1.0; 9]),
        CriticalPointType::Degenerate
    );
    // Saddle-node: a zero eigenvalue between the signs.
    assert_eq!(
        classify_from_gradient(&diag(-1.0, 0.0, 1.0)),
        CriticalPointType::Degenerate
    );
    assert_eq!(
        classify_from_gradient(&diag(f64::NAN, -1.0, -1.0)),
        CriticalPointType::Degenerate
    );
    assert_eq!(
        classify_from_gradient(&diag(f64::INFINITY, -1.0, -1.0)),
        CriticalPointType::Degenerate
    );
}

#[test]
fn classification_is_invariant_under_rotation_and_scaling() {
    let rot = rotation([1.0, 2.0, 3.0], 0.9);
    let cases: [(Mat3, CriticalPointType); 6] = [
        (diag(-1.0, -2.0, -3.0), CriticalPointType::AttractingNode),
        (diag(1.0, 2.0, 3.0), CriticalPointType::RepellingNode),
        (diag(1.0, -1.0, -1.0), CriticalPointType::Saddle),
        (
            [-1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, -1.0],
            CriticalPointType::AttractingFocus,
        ),
        (
            [1.0, -2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            CriticalPointType::RepellingFocus,
        ),
        (
            [1.0, -2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            CriticalPointType::Saddle,
        ),
    ];
    for (g, expected) in cases {
        for factor in [1e-9, 1.0, 1e9] {
            let rotated: Mat3 = conjugate(&rot, &g).map(|v| v * factor);
            assert_eq!(
                classify_from_gradient(&rotated),
                expected,
                "rotated and scaled by {factor}: {g:?}"
            );
        }
    }
}

#[test]
fn non_symmetric_triangular_gradients_are_classified_by_their_diagonal() {
    let sink = [-1.0, 5.0, 0.0, 0.0, -2.0, 3.0, 0.0, 0.0, -3.0];
    assert_eq!(
        classify_from_gradient(&sink),
        CriticalPointType::AttractingNode
    );
    let saddle = [1.0, 10.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
    assert_eq!(classify_from_gradient(&saddle), CriticalPointType::Saddle);
}

#[test]
fn planar_gradients_are_classified_in_the_phase_plane() {
    let planar = |a: f64, b: f64, c: f64, d: f64| [a, b, 0.0, c, d, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(
        classify_planar_gradient(&planar(-1.0, 0.0, 0.0, -2.0)),
        CriticalPointType::AttractingNode
    );
    assert_eq!(
        classify_planar_gradient(&planar(1.0, 0.0, 0.0, 2.0)),
        CriticalPointType::RepellingNode
    );
    assert_eq!(
        classify_planar_gradient(&planar(1.0, 0.0, 0.0, -1.0)),
        CriticalPointType::Saddle
    );
    assert_eq!(
        classify_planar_gradient(&planar(0.0, -1.0, 1.0, 0.0)),
        CriticalPointType::Center
    );
    assert_eq!(
        classify_planar_gradient(&planar(-0.1, -1.0, 1.0, -0.1)),
        CriticalPointType::AttractingFocus
    );
    assert_eq!(
        classify_planar_gradient(&planar(0.1, -1.0, 1.0, 0.1)),
        CriticalPointType::RepellingFocus
    );
    assert_eq!(
        classify_planar_gradient(&planar(1.0, 1.0, 1.0, 1.0)),
        CriticalPointType::Degenerate
    );
    assert_eq!(
        classify_planar_gradient(&[0.0; 9]),
        CriticalPointType::Degenerate
    );
    assert_eq!(
        classify_planar_gradient(&planar(f64::NAN, 0.0, 0.0, 1.0)),
        CriticalPointType::Degenerate
    );
}

/// Velocity field `u = g (x - c)` on an `n`-cubed grid with the critical point
/// `c` at the centre node.
fn field_with_critical_point(n: usize, nz: usize, g: &Mat3) -> FlowField3D {
    let mut field = FlowField3D::new(n, n, nz, 1.0);
    let centre = [(n / 2) as f64, (n / 2) as f64, (nz / 2) as f64];
    for iz in 0..nz {
        for iy in 0..n {
            for ix in 0..n {
                let x = [
                    ix as f64 - centre[0],
                    iy as f64 - centre[1],
                    iz as f64 - centre[2],
                ];
                let u: [f64; 3] =
                    std::array::from_fn(|c| (0..3).map(|a| g[c * 3 + a] * x[a]).sum());
                field.set_vel(ix as i64, iy as i64, iz as i64, u);
            }
        }
    }
    field
}

#[test]
fn classify_critical_points_finds_and_types_three_dimensional_points() {
    let cases: [(Mat3, CriticalPointType); 3] = [
        (diag(1.0, -1.0, -1.0), CriticalPointType::Saddle),
        (diag(-1.0, -1.0, -1.0), CriticalPointType::AttractingNode),
        (
            [-1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, -1.0],
            CriticalPointType::AttractingFocus,
        ),
    ];
    for (g, expected) in cases {
        let field = field_with_critical_point(5, 5, &g);
        let points = classify_critical_points(&field);
        assert_eq!(points.len(), 1, "{g:?}");
        assert_eq!(points[0].point_type, expected, "{g:?}");
        assert_eq!(points[0].position, [2.0, 2.0, 2.0]);
        for (stored, expected) in points[0].gradient.iter().zip(g) {
            assert_close(*stored, expected, 1e-12, "stored gradient");
        }
    }
}

#[test]
fn classify_critical_points_treats_single_layer_fields_as_planar() {
    let cases: [(Mat3, CriticalPointType); 4] = [
        (
            [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            CriticalPointType::Saddle,
        ),
        (
            [-1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0],
            CriticalPointType::AttractingNode,
        ),
        (
            [-0.5, -1.0, 0.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0],
            CriticalPointType::AttractingFocus,
        ),
        (
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            CriticalPointType::Center,
        ),
    ];
    for (g, expected) in cases {
        let field = field_with_critical_point(5, 1, &g);
        let points = classify_critical_points(&field);
        assert_eq!(points.len(), 1, "{g:?}");
        assert_eq!(points[0].point_type, expected, "{g:?}");
        assert_eq!(points[0].position, [2.0, 2.0, 0.0]);
    }
}

#[test]
fn separatrices_are_traced_from_the_saddle_that_the_classifier_finds() {
    let field = field_with_critical_point(5, 5, &diag(1.0, -1.0, -1.0));
    let points = classify_critical_points(&field);
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].point_type, CriticalPointType::Saddle);
    let pairs = trace_separatrices(&field, &points, 0.1, 3);
    assert_eq!(pairs.len(), 1);

    let sink = field_with_critical_point(5, 5, &diag(-1.0, -1.0, -1.0));
    let sink_points = classify_critical_points(&sink);
    assert!(trace_separatrices(&sink, &sink_points, 0.1, 3).is_empty());
}

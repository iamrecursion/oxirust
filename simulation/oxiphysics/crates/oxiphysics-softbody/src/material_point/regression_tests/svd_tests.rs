//! F4175 / F4174: `mat3_svd` is a true signed SVD and `polar_decomp` returns a
//! proper rotation for every input.

use super::*;

type Svd = ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]);

/// Checks every property of the signed SVD of `m` and returns the factors:
/// proper rotations `U`, `V`, ordered singular values with `σ₂` carrying the sign
/// of `det M`, and `U diag(σ) Vᵀ = M`.
fn assert_valid_svd(m: [[f64; 3]; 3], tol: f64, label: &str) -> Svd {
    let (u, sigma, vt) = mat3_svd(m);
    let v = mat3_transpose(vt);
    assert!(
        orthogonality_error(u) < tol,
        "{label}: UᵀU error {}",
        orthogonality_error(u)
    );
    assert!(
        orthogonality_error(v) < tol,
        "{label}: VᵀV error {}",
        orthogonality_error(v)
    );
    assert!((mat3_det(u) - 1.0).abs() < tol, "{label}: det U");
    assert!((mat3_det(v) - 1.0).abs() < tol, "{label}: det V");
    assert!(
        sigma[0] >= sigma[1] && sigma[1] >= sigma[2].abs() && sigma[1] >= 0.0,
        "{label}: singular values not ordered: {sigma:?}"
    );
    let err = max_diff(reconstruct(u, sigma, vt), m);
    assert!(
        err < tol * (1.0 + frobenius(m)),
        "{label}: |U Σ Vᵀ - M| = {err}"
    );
    let det = mat3_det(m);
    if det.abs() > 1e-9 {
        assert_eq!(
            sigma[2] < 0.0,
            det < 0.0,
            "{label}: sign of σ₂ vs det {det}"
        );
        let product = sigma[0] * sigma[1] * sigma[2];
        assert!(
            (product - det).abs() < 1e-10 * det.abs().max(1.0),
            "{label}: σ₀σ₁σ₂ = {product}, det = {det}"
        );
    }
    (u, sigma, vt)
}

#[test]
fn f4175_svd_of_sorted_diagonals_is_exact_without_rotations() {
    for diag in [
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.005, 1.0, 1.0],
        [1.1, 1.1, 1.1],
        [3.0, 2.0, 0.5],
    ] {
        let (u, sigma, vt) = assert_valid_svd(mat3_diag(diag), 1e-15, "sorted diagonal");
        assert_eq!(u, mat3_identity(), "U for {diag:?}");
        assert_eq!(vt, mat3_identity(), "Vᵀ for {diag:?}");
        assert_eq!(sigma, diag);
    }
}

#[test]
fn f4175_svd_sorts_an_unordered_diagonal_with_proper_rotations() {
    let (u, sigma, vt) = assert_valid_svd(mat3_diag([1.0, 2.0, 3.0]), 1e-15, "diag(1,2,3)");
    assert_eq!(sigma, [3.0, 2.0, 1.0]);
    // The sort is an odd permutation, so a column sign has to move into σ and back.
    assert!((mat3_det(u) - 1.0).abs() < 1e-15);
    assert!((mat3_det(vt) - 1.0).abs() < 1e-15);
}

#[test]
fn f4175_svd_of_zero_and_rank_deficient_matrices_has_orthonormal_factors() {
    // Zero matrix: all singular values vanish and both factors are the identity.
    let (u, sigma, vt) = assert_valid_svd(mat3_zero(), 1e-15, "zero");
    assert_eq!(u, mat3_identity());
    assert_eq!(vt, mat3_identity());
    assert_eq!(sigma, [0.0, 0.0, 0.0]);

    // Exactly zero columns after the Jacobi pass.
    let (_, sigma, _) = assert_valid_svd(mat3_diag([2.0, 0.0, 0.0]), 1e-15, "diag(2,0,0)");
    assert_eq!(sigma, [2.0, 0.0, 0.0]);
    let (_, sigma, _) = assert_valid_svd(mat3_diag([0.0, 0.0, 5.0]), 1e-15, "diag(0,0,5)");
    assert_eq!(sigma, [5.0, 0.0, 0.0]);

    // Rotated rank-1 matrix a bᵀ.
    let a = [1.0, 2.0, 3.0];
    let b = [-2.0, 0.5, 1.0];
    let rank1 = outer3(a, b);
    let (_, sigma, _) = assert_valid_svd(rank1, 1e-13, "rank 1");
    assert!((sigma[0] - len3(a) * len3(b)).abs() < 1e-12);
    assert!(sigma[1].abs() < 1e-12 && sigma[2].abs() < 1e-12);

    // Rotated rank-2 matrix Q₁ diag(3, 1, 0) Q₂.
    let mut rng = TestRng::new(0x7A4E_0002);
    for _ in 0..50 {
        let q1 = random_rotation(&mut rng);
        let q2 = random_rotation(&mut rng);
        let rank2 = mat3_mul(mat3_mul(q1, mat3_diag([3.0, 1.0, 0.0])), q2);
        let (_, sigma, _) = assert_valid_svd(rank2, 1e-13, "rank 2");
        assert!((sigma[0] - 3.0).abs() < 1e-12, "σ₀ = {}", sigma[0]);
        assert!((sigma[1] - 1.0).abs() < 1e-12, "σ₁ = {}", sigma[1]);
        assert!(sigma[2].abs() < 1e-12, "σ₂ = {}", sigma[2]);
    }
}

#[test]
fn f4175_svd_of_an_inverted_matrix_has_a_negative_last_singular_value() {
    let (u, sigma, vt) = assert_valid_svd(mat3_diag([-1.0, 2.0, 3.0]), 1e-15, "diag(-1,2,3)");
    assert_eq!(sigma, [3.0, 2.0, -1.0]);
    assert!(mat3_det(u) > 0.0 && mat3_det(vt) > 0.0);

    let f = mat3_mul(rot_z(0.4), mat3_diag([1.2, -0.9, 1.0]));
    let (_, sigma, _) = assert_valid_svd(f, 1e-13, "inverted");
    assert!(
        sigma[2] < 0.0,
        "inverted gradient must give σ₂ < 0: {sigma:?}"
    );
    assert!((sigma[2].abs() - 0.9).abs() < 1e-12);
}

#[test]
fn f4175_svd_of_random_matrices_reconstructs_and_is_orthonormal() {
    let mut rng = TestRng::new(0x5D0_0F4175);
    for case in 0..1000 {
        let m = rng.mat3(-1.0, 1.0);
        assert_valid_svd(m, 1e-13, &format!("random #{case}"));
    }
}

#[test]
fn f4175_svd_is_valid_for_every_matrix_with_entries_in_minus_one_zero_one() {
    // 3^9 = 19683 matrices: permutations (det -1), zero rows and columns, identical
    // and opposite columns, every rank from 0 to 3 and many repeated singular values.
    let mut checked = 0usize;
    for code in 0..19683usize {
        let mut rest = code;
        let mut m = [[0.0f64; 3]; 3];
        for row in &mut m {
            for entry in row.iter_mut() {
                *entry = (rest % 3) as f64 - 1.0;
                rest /= 3;
            }
        }
        assert_valid_svd(m, 1e-13, &format!("ternary matrix #{code}"));
        checked += 1;
    }
    assert_eq!(checked, 19683);
}

#[test]
fn f4175_svd_is_valid_for_integer_matrices_with_wide_dynamic_range() {
    let mut rng = TestRng::new(0x1A7E_4175);
    for case in 0..20000 {
        let scale = 10f64.powi((rng.range(-6.0, 6.0)).floor() as i32);
        let mut m = [[0.0f64; 3]; 3];
        for row in &mut m {
            for entry in row.iter_mut() {
                *entry = (rng.range(-3.0, 4.0).floor()) * scale;
            }
        }
        let (u, sigma, vt) = mat3_svd(m);
        assert!(orthogonality_error(u) < 1e-13, "#{case}: UᵀU");
        assert!(
            orthogonality_error(mat3_transpose(vt)) < 1e-13,
            "#{case}: VᵀV"
        );
        assert!(
            sigma[0] >= sigma[1] && sigma[1] >= sigma[2].abs(),
            "#{case}: order"
        );
        let err = max_diff(reconstruct(u, sigma, vt), m);
        assert!(
            err <= 1e-13 * (scale * 9.0),
            "#{case}: reconstruction {err} at scale {scale}"
        );
    }
}

#[test]
fn f4175_svd_handles_repeated_singular_values() {
    let mut rng = TestRng::new(0x2E9E_4175);
    for sigma_in in [
        [2.0, 2.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.5, 1.5, 1.5],
        [1.0, 1.0, 0.0],
    ] {
        for _ in 0..20 {
            let q1 = random_rotation(&mut rng);
            let q2 = random_rotation(&mut rng);
            let m = mat3_mul(mat3_mul(q1, mat3_diag(sigma_in)), q2);
            let (_, sigma, _) = assert_valid_svd(m, 1e-13, "repeated");
            for k in 0..3 {
                assert!((sigma[k] - sigma_in[k]).abs() < 1e-12, "σ = {sigma:?}");
            }
        }
    }
}

#[test]
fn f4175_svd_keeps_full_accuracy_for_ill_conditioned_matrices() {
    let mut rng = TestRng::new(0x111C_4175);
    for _ in 0..50 {
        let q1 = random_rotation(&mut rng);
        let q2 = random_rotation(&mut rng);
        let m = mat3_mul(mat3_mul(q1, mat3_diag([1.0, 1e-4, 1e-8])), q2);
        let (u, sigma, vt) = assert_valid_svd(m, 1e-13, "ill-conditioned");
        // Orthogonality does not degrade with the condition number.
        assert!(orthogonality_error(u) < 1e-14);
        assert!(orthogonality_error(mat3_transpose(vt)) < 1e-14);
        assert!((sigma[0] - 1.0).abs() < 1e-14);
        assert!((sigma[1] - 1e-4).abs() < 1e-14);
        assert!((sigma[2] - 1e-8).abs() < 1e-14, "σ₂ = {}", sigma[2]);
    }
}

#[test]
fn f4175_svd_of_non_finite_input_is_nan_and_does_not_panic() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut m = mat3_identity();
        m[1][2] = bad;
        let (u, sigma, vt) = mat3_svd(m);
        assert_eq!(u, mat3_identity());
        assert_eq!(vt, mat3_identity());
        assert!(sigma.iter().all(|s| s.is_nan()));
    }
    // Entries whose squares overflow are rejected the same way.
    let (_, sigma, _) = mat3_svd(mat3_diag([1e300, 1.0, 1.0]));
    assert!(sigma.iter().all(|s| s.is_nan()));
}

#[test]
fn f4175_svd_agrees_with_the_core_based_signed_svd3_on_well_conditioned_input() {
    use crate::fem_soft::{reconstruct_from_svd, signed_svd3};
    let mut rng = TestRng::new(0xC0DE_4175);
    for _ in 0..200 {
        let m = mat3_add(mat3_identity(), mat3_scale(rng.mat3(-1.0, 1.0), 0.3));
        let (u, sigma, vt) = assert_valid_svd(m, 1e-13, "I + 0.3 random");
        let (u_ref, sigma_ref, vt_ref) = signed_svd3(m);
        for k in 0..3 {
            assert!(
                (sigma[k] - sigma_ref[k]).abs() < 1e-10,
                "σ {sigma:?} vs reference {sigma_ref:?}"
            );
        }
        // Singular vectors are only defined up to a common sign per pair, so compare
        // the reconstructions and the rotation U Vᵀ (which is sign independent).
        assert!(
            max_diff(
                reconstruct(u, sigma, vt),
                reconstruct_from_svd(u_ref, sigma_ref, vt_ref)
            ) < 1e-10
        );
        assert!(max_diff(mat3_mul(u, vt), mat3_mul(u_ref, vt_ref)) < 1e-9);
    }
}

#[test]
fn f4174_polar_decomp_of_a_symmetric_stretch_is_the_identity_rotation() {
    // The old Higham iteration stopped once the iterate was symmetric and returned
    // R = diag(1.25, 1, 1) for this input.
    let f = mat3_diag([2.0, 1.0, 1.0]);
    let (r, s) = polar_decomp(f);
    assert!(max_diff(r, mat3_identity()) < 1e-12, "R = {r:?}");
    assert!(max_diff(s, f) < 1e-12, "S = {s:?}");
}

#[test]
fn f4174_polar_decomp_recovers_rotation_and_stretch() {
    let rotation = rot_z(0.3);
    let stretch = mat3_diag([1.2, 0.9, 1.0]);
    let f = mat3_mul(rotation, stretch);
    let (r, s) = polar_decomp(f);
    assert!(orthogonality_error(r) < 1e-12, "RᵀR != I");
    assert!((mat3_det(r) - 1.0).abs() < 1e-12);
    assert!(max_diff(r, rotation) < 1e-10, "R = {r:?}");
    assert!(max_diff(s, stretch) < 1e-10, "S = {s:?}");
    assert!(max_diff(mat3_mul(r, s), f) < 1e-10);
    assert!(symmetry_error(s) < 1e-12);
}

#[test]
fn f4174_polar_decomp_keeps_a_proper_rotation_for_inverted_gradients() {
    let f = mat3_mul(rot_z(0.4), mat3_diag([1.3, -0.8, 1.1]));
    assert!(mat3_det(f) < 0.0);
    let (r, s) = polar_decomp(f);
    assert!((mat3_det(r) - 1.0).abs() < 1e-12, "det R = {}", mat3_det(r));
    assert!(orthogonality_error(r) < 1e-12);
    assert!(max_diff(mat3_mul(r, s), f) < 1e-12);
    assert!(symmetry_error(s) < 1e-12);
    // Exactly one negative eigenvalue: det S < 0.
    assert!(mat3_det(s) < 0.0);
}

#[test]
fn f4174_polar_decomp_of_random_matrices_is_exact() {
    let mut rng = TestRng::new(0x901A_4174);
    for case in 0..500 {
        let f = rng.mat3(-1.5, 1.5);
        let (r, s) = polar_decomp(f);
        assert!(orthogonality_error(r) < 1e-12, "#{case}: RᵀR != I");
        assert!((mat3_det(r) - 1.0).abs() < 1e-12, "#{case}: det R");
        assert!(max_diff(mat3_mul(r, s), f) < 1e-12, "#{case}: R S != F");
        assert!(symmetry_error(s) < 1e-12, "#{case}: S not symmetric");
    }
}

#[test]
fn f4174_fixed_corotated_stress_is_frame_indifferent_for_a_symmetric_stretch() {
    // P(Q F) = Q P(F). For the symmetric F = diag(...) the old polar factor was a
    // skewed R, while Q F (not symmetric) converged correctly, so this broke.
    let (mu, lambda) = (1.0e5, 2.0e5);
    let f = mat3_diag([1.3, 0.8, 1.1]);
    let q = rot_z(0.7);
    let p_f = fixed_corotated_pk1(f, mu, lambda);
    let p_qf = fixed_corotated_pk1(mat3_mul(q, f), mu, lambda);
    assert!(
        max_diff(p_qf, mat3_mul(q, p_f)) < 1e-6 * frobenius(p_f),
        "P(QF) = {p_qf:?}, Q P(F) = {:?}",
        mat3_mul(q, p_f)
    );
    // The symmetric F has R = I, so P_xx = 2 μ (F_xx - 1) + λ (J - 1) J / F_xx.
    let expected_xx = 2.0 * mu * (1.3 - 1.0) + lambda * (1.3 * 0.8 * 1.1 - 1.0) * (0.8 * 1.1);
    assert!((p_f[0][0] - expected_xx).abs() < 1e-6 * expected_xx.abs());
}

#[test]
fn f4174_fixed_corotated_energy_vanishes_for_rigid_rotations() {
    let f = rot_z(1.1);
    assert!(fixed_corotated_energy(f, 1.0e5, 2.0e5).abs() < 1e-18);
    let stretch = mat3_diag([1.2, 1.0, 1.0]);
    assert!(fixed_corotated_energy(stretch, 1.0e5, 2.0e5) > 0.0);
    // Pure rotation of the stretch changes nothing.
    let rotated = mat3_mul(rot_x(0.9), stretch);
    let a = fixed_corotated_energy(stretch, 1.0e5, 2.0e5);
    let b = fixed_corotated_energy(rotated, 1.0e5, 2.0e5);
    assert!((a - b).abs() < 1e-9 * a);
}

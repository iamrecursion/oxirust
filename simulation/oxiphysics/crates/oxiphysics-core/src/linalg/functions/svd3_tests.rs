// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression and property tests for [`svd3`](super::functions_impl::svd3).
//!
//! These cover the degenerate branches that the previous normal-equation
//! implementation got wrong: rank 0/1/2 inputs (missing left singular vectors
//! were filled with duplicated, non-orthogonal columns), repeated singular
//! values, and tiny / huge scales (an absolute tolerance made the eigen solver
//! skip every rotation and the fallback discard non-zero singular values).

use crate::linalg::{det3, polar_decomp3, svd3};
use crate::math::{Mat3, Real, Vec3};
use proptest::prelude::*;

/// Relative tolerance demanded for reconstruction and orthonormality.
const REL_TOL: Real = 1e-12;

/// Deterministic LCG so every randomized test is reproducible.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> Real {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as Real) / ((1_u64 << 53) as Real) * 2.0 - 1.0
    }

    fn matrix(&mut self) -> Mat3 {
        Mat3::from_fn(|_, _| self.next_unit())
    }

    fn unit_vector(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(self.next_unit(), self.next_unit(), self.next_unit());
            let n = v.norm();
            if n > 0.1 {
                return v / n;
            }
        }
    }

    /// Random proper rotation (det = +1) built from a random unit quaternion.
    fn rotation(&mut self) -> Mat3 {
        let (w, x, y, z) = loop {
            let q = [
                self.next_unit(),
                self.next_unit(),
                self.next_unit(),
                self.next_unit(),
            ];
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if n > 0.1 {
                break (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
            }
        };
        Mat3::new(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    }
}

fn max_abs(m: Mat3) -> Real {
    m.abs().max()
}

/// Assert every structural SVD property and return `(U, sigma, V)`.
fn check_svd(m: Mat3, label: &str) -> (Mat3, Vec3, Mat3) {
    let (u, sigma, v) = svd3(m);
    let scale = max_abs(m);
    let ortho_u = max_abs(u.transpose() * u - Mat3::identity());
    let ortho_v = max_abs(v.transpose() * v - Mat3::identity());
    assert!(
        ortho_u < REL_TOL,
        "{label}: U not orthonormal ({ortho_u:e})"
    );
    assert!(
        ortho_v < REL_TOL,
        "{label}: V not orthonormal ({ortho_v:e})"
    );
    assert!(
        (v.determinant() - 1.0).abs() < REL_TOL,
        "{label}: det V = {} (expected +1)",
        v.determinant()
    );
    assert!(
        (u.determinant().abs() - 1.0).abs() < REL_TOL,
        "{label}: |det U| = {}",
        u.determinant().abs()
    );
    assert!(sigma.z >= 0.0, "{label}: negative singular value {sigma:?}");
    assert!(
        sigma.x >= sigma.y && sigma.y >= sigma.z,
        "{label}: singular values not descending {sigma:?}"
    );
    let recon = u * Mat3::from_diagonal(&sigma) * v.transpose();
    let err = max_abs(recon - m);
    assert!(
        err <= REL_TOL * scale,
        "{label}: reconstruction error {err:e} (scale {scale:e})"
    );
    (u, sigma, v)
}

fn assert_sigma(sigma: Vec3, expected: [Real; 3], label: &str) {
    let scale = expected[0].abs().max(Real::MIN_POSITIVE);
    for (i, e) in expected.iter().enumerate() {
        assert!(
            (sigma[i] - e).abs() <= REL_TOL * scale,
            "{label}: sigma[{i}] = {:e}, expected {e:e}",
            sigma[i]
        );
    }
}

#[test]
fn svd3_zero_matrix_is_identity_frames() {
    let (u, sigma, v) = check_svd(Mat3::zeros(), "zero");
    assert_eq!(sigma, Vec3::zeros());
    assert!((u.determinant() - 1.0).abs() < REL_TOL);
    assert!((v.determinant() - 1.0).abs() < REL_TOL);
}

#[test]
fn svd3_rank_one_fills_orthonormal_complement() {
    // Regression: both missing columns of U used to receive the same vector.
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(-0.5, 0.25, 2.0);
    let m = a * b.transpose();
    let (u, sigma, _) = check_svd(m, "rank-1 outer product");
    assert_sigma(sigma, [a.norm() * b.norm(), 0.0, 0.0], "rank-1");
    assert!(
        (u.determinant() - 1.0).abs() < REL_TOL,
        "rank-deficient U must be proper"
    );
}

#[test]
fn svd3_rank_one_axis_aligned() {
    let m = Mat3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0);
    let (_, sigma, _) = check_svd(m, "rank-1 axis");
    assert_sigma(sigma, [5.0, 0.0, 0.0], "rank-1 axis");
}

#[test]
fn svd3_rank_two_exact() {
    let m = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let (u, sigma, _) = check_svd(m, "rank-2 1..9");
    assert!(sigma.z <= 1e-14 * sigma.x, "sigma_min not ~0: {sigma:?}");
    assert!(
        (u.determinant() - 1.0).abs() < REL_TOL,
        "rank-deficient U must be proper"
    );
}

#[test]
fn svd3_repeated_singular_values() {
    let mut rng = Lcg(0x5eed_0001);
    let spectra = [
        [2.0, 2.0, 1.0],
        [3.0, 1.0, 1.0],
        [4.0, 4.0, 4.0],
        [1.0, 1.0, 0.0],
        [7.0, 0.0, 0.0],
    ];
    for spectrum in spectra {
        for _ in 0..50 {
            let p = rng.rotation();
            let q = rng.rotation();
            let s = Mat3::from_diagonal(&Vec3::new(spectrum[0], spectrum[1], spectrum[2]));
            let m = p * s * q.transpose();
            let (_, sigma, _) = check_svd(m, "repeated spectrum");
            assert_sigma(sigma, spectrum, "repeated spectrum");
        }
    }
}

#[test]
fn svd3_reflections_have_negative_det_u() {
    let m = Mat3::from_diagonal(&Vec3::new(1.0, 1.0, -1.0));
    let (u, sigma, _) = check_svd(m, "reflection");
    assert_sigma(sigma, [1.0, 1.0, 1.0], "reflection");
    assert!((u.determinant() + 1.0).abs() < REL_TOL);
    let (u_neg, _, _) = check_svd(-Mat3::identity(), "-I");
    assert!((u_neg.determinant() + 1.0).abs() < REL_TOL);
}

#[test]
fn svd3_random_dense_all_scales() {
    let mut rng = Lcg(0x5eed_0002);
    for &scale in &[1e-200, 1e-150, 1e-30, 1e-8, 1.0, 1e8, 1e30, 1e150, 1e200] {
        for _ in 0..200 {
            let m = rng.matrix() * scale;
            let (u, sigma, _) = check_svd(m, "random dense");
            let det = det3(m / scale);
            if det.abs() > 1e-6 {
                assert_eq!(
                    u.determinant().signum(),
                    det.signum(),
                    "det U must carry sign(det M) at scale {scale:e}"
                );
            }
            let frob = (m / scale).norm();
            let sig_frob = (sigma / scale).norm();
            assert!((frob - sig_frob).abs() <= 1e-12 * frob);
        }
    }
}

#[test]
fn svd3_random_rank_deficient_all_scales() {
    let mut rng = Lcg(0x5eed_0003);
    for &scale in &[1e-150, 1e-8, 1.0, 1e8, 1e150] {
        for rank in 1..=2 {
            for _ in 0..100 {
                let s1 = 0.5 + 0.5 * rng.next_unit().abs();
                let s2 = if rank == 2 {
                    0.1 + 0.4 * rng.next_unit().abs()
                } else {
                    0.0
                };
                let spectrum = [s1 * scale, s2 * scale, 0.0];
                let p = rng.rotation();
                let q = rng.rotation();
                let m = p
                    * Mat3::from_diagonal(&Vec3::new(spectrum[0], spectrum[1], 0.0))
                    * q.transpose();
                let (u, sigma, _) = check_svd(m, "random rank-deficient");
                assert_sigma(sigma, spectrum, "random rank-deficient");
                assert!(
                    (u.determinant() - 1.0).abs() < REL_TOL,
                    "rank-deficient U must be proper (rank {rank}, scale {scale:e})"
                );
            }
        }
    }
}

#[test]
fn svd3_graded_near_degenerate() {
    let mut rng = Lcg(0x5eed_0004);
    for _ in 0..200 {
        let p = rng.rotation();
        let q = rng.rotation();
        let spectrum = [1.0, 1e-7, 1e-11];
        let m = p
            * Mat3::from_diagonal(&Vec3::new(spectrum[0], spectrum[1], spectrum[2]))
            * q.transpose();
        let (_, sigma, _) = check_svd(m, "graded");
        assert_sigma(sigma, spectrum, "graded");
    }
}

#[test]
fn svd3_tiny_diagonal_keeps_singular_values() {
    // Regression: every sigma below the absolute EPS was replaced by a
    // fallback column, so U * S * V^T no longer reproduced the input.
    let m = Mat3::from_diagonal(&Vec3::new(3e-15, 2e-15, 1e-15));
    let (_, sigma, _) = check_svd(m, "tiny diagonal");
    assert_sigma(sigma, [3e-15, 2e-15, 1e-15], "tiny diagonal");
}

#[test]
fn svd3_collinear_columns_and_unit_vector_outer_products() {
    let mut rng = Lcg(0x5eed_0005);
    for _ in 0..200 {
        let a = rng.unit_vector();
        let b = rng.unit_vector();
        let c = rng.unit_vector();
        // Rank 1: three collinear columns.
        let m1 = Mat3::from_columns(&[a * 2.0, a * -3.0, a * 0.5]);
        check_svd(m1, "collinear columns");
        // Rank 2: sum of two outer products.
        let m2 = a * b.transpose() + c * a.transpose() * 0.25;
        check_svd(m2, "two outer products");
    }
}

#[test]
fn svd3_non_finite_input_does_not_hang() {
    let mut m = Mat3::identity();
    m[(1, 2)] = Real::NAN;
    let (_, sigma, _) = svd3(m);
    assert!(sigma.iter().all(|s| s.is_nan()));
    m[(1, 2)] = Real::INFINITY;
    let (_, sigma_inf, _) = svd3(m);
    assert!(sigma_inf.iter().all(|s| s.is_nan()));
}

#[test]
fn polar_decomp3_rank_deficient_rotation_is_orthogonal() {
    let a = Vec3::new(1.0, -1.0, 2.0);
    let b = Vec3::new(0.3, 0.7, -0.2);
    let m = a * b.transpose();
    let (r, s) = polar_decomp3(m);
    assert!(max_abs(r.transpose() * r - Mat3::identity()) < REL_TOL);
    assert!(max_abs(r * s - m) < REL_TOL * max_abs(m));
}

proptest! {
    #[test]
    fn prop_svd3_reconstructs_arbitrary_matrices(
        entries in proptest::array::uniform9(-1.0e3_f64..1.0e3_f64),
        exponent in -100_i32..100_i32,
    ) {
        let scale = (exponent as Real).exp2();
        let m = Mat3::from_row_slice(&entries) * scale;
        check_svd(m, "proptest");
    }

    #[test]
    fn prop_svd3_rank_deficient_columns(
        col in proptest::array::uniform3(-10.0_f64..10.0_f64),
        k1 in -5.0_f64..5.0_f64,
        k2 in -5.0_f64..5.0_f64,
        other in proptest::array::uniform3(-10.0_f64..10.0_f64),
        rank_one in proptest::bool::ANY,
    ) {
        let a = Vec3::new(col[0], col[1], col[2]);
        let o = Vec3::new(other[0], other[1], other[2]);
        let second = if rank_one { a * k1 } else { o };
        let m = Mat3::from_columns(&[a, second, a * k2 + second * 0.5]);
        let (u, sigma, _) = check_svd(m, "proptest rank-deficient");
        prop_assert!(sigma.z <= 1e-13 * sigma.x.max(Real::MIN_POSITIVE));
        prop_assert!((u.determinant() - 1.0).abs() < REL_TOL);
    }
}

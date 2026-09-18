// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the Varga strain energy
//! `W = 2μ (λ̄1 + λ̄2 + λ̄3 - 3) + K/2 (J-1)²` in the principal isochoric stretches
//! `λ̄a = √eig_a(J^{-2/3} FᵀF)`.
//!
//! The previous implementation used the proxy `√(3 Ī1)` for `Σ λ̄a` (exact only for
//! equal stretches) and a deviatoric prefactor `μ`, i.e. half the initial shear
//! modulus of the documented formula.

use super::functions::*;
use super::types::Varga;

const MU: f64 = 1.0e6;
const BULK: f64 = 1.0e9;

fn varga() -> Varga {
    Varga {
        mu: MU,
        bulk_modulus: BULK,
    }
}

fn diagonal(a: f64, b: f64, c: f64) -> [[f64; 3]; 3] {
    [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
}

fn simple_shear(gamma: f64) -> [[f64; 3]; 3] {
    [[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Rotation by `angle` about the (not necessarily unit) `axis` (Rodrigues' formula
/// `R = cos θ I + sin θ [k]× + (1 − cos θ) k kᵀ`).
fn rotation(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let norm = axis.iter().map(|a| a * a).sum::<f64>().sqrt();
    let k = axis.map(|a| a / norm);
    let (s, c) = angle.sin_cos();
    let cross = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let delta = if i == j { 1.0 } else { 0.0 };
            c * delta + (1.0 - c) * k[i] * k[j] + s * cross[i][j]
        })
    })
}

/// A general deformation gradient with all nine entries non-zero (J > 0).
fn general_f() -> [[f64; 3]; 3] {
    [[1.25, 0.30, 0.10], [0.05, 0.90, -0.20], [0.12, 0.15, 1.10]]
}

fn frobenius(a: [[f64; 3]; 3]) -> f64 {
    a.iter().flatten().map(|v| v * v).sum::<f64>().sqrt()
}

#[test]
fn uniaxial_isochoric_energy_is_two_mu_times_stretch_excess() {
    // F = diag(λ, λ^-1/2, λ^-1/2), J = 1: the stretches are the diagonal entries.
    let lambda = 2.0_f64;
    let f = diagonal(lambda, lambda.powf(-0.5), lambda.powf(-0.5));
    let expected = 2.0 * MU * (lambda + 2.0 * lambda.powf(-0.5) - 3.0);
    let w = varga().strain_energy(f);
    assert!(
        (w - expected).abs() <= 1e-10 * expected,
        "W = {w}, expected 2 mu (lambda + 2 lambda^-1/2 - 3) = {expected}"
    );
}

#[test]
fn equibiaxial_isochoric_energy_matches_stretch_sum() {
    // Two equal stretches (a repeated eigenvalue of C-bar) and one distinct.
    let lambda = 1.5_f64;
    let f = diagonal(lambda, lambda, lambda.powi(-2));
    let expected = 2.0 * MU * (2.0 * lambda + lambda.powi(-2) - 3.0);
    let w = varga().strain_energy(f);
    assert!(
        (w - expected).abs() <= 1e-10 * expected,
        "W = {w}, expected {expected}"
    );
    // The old proxy mu * (sqrt(3 I1_bar) - 3) is about 15 % too small in this state
    // (it is exact only for equal stretches, where both sides vanish).
    let (i1_bar, _, _) = invariants(deviatoric_right_cauchy_green(f));
    let proxy = MU * ((3.0 * i1_bar).sqrt() - 3.0);
    assert!(
        (proxy - expected).abs() > 0.1 * expected,
        "proxy {proxy}, expected {expected}"
    );
}

#[test]
fn simple_shear_energy_matches_closed_form() {
    // C-bar of a simple shear has the eigenvalues ((2 + g²) ± g sqrt(g² + 4))/2 and 1,
    // so the stretches are (sqrt(g² + 4) ± g)/2 and 1: sum - 3 = sqrt(g² + 4) - 2.
    // A square root of the diagonal of C (the F1273-type mistake) cannot reproduce it.
    for gamma in [0.2_f64, 0.7, 1.5] {
        let expected = 2.0 * MU * ((gamma * gamma + 4.0).sqrt() - 2.0);
        let w = varga().strain_energy(simple_shear(gamma));
        assert!(
            (w - expected).abs() <= 1e-10 * expected,
            "gamma = {gamma}: W = {w}, expected {expected}"
        );
    }
}

#[test]
fn pure_dilatation_stores_only_volumetric_energy() {
    // F = s I: every isochoric stretch is exactly 1, so W = K/2 (J - 1)².
    let s = 1.1_f64;
    let f = diagonal(s, s, s);
    let j = s * s * s;
    let expected = BULK / 2.0 * (j - 1.0).powi(2);
    let w = varga().strain_energy(f);
    assert!((w - expected).abs() <= 1e-10 * expected, "W = {w}");
}

#[test]
fn deviatoric_principal_stretches_have_unit_product_and_ascend() {
    let stretches = deviatoric_principal_stretches(general_f());
    let product: f64 = stretches.iter().product();
    assert!((product - 1.0).abs() < 1e-12, "product = {product}");
    assert!(stretches[0] <= stretches[1] && stretches[1] <= stretches[2]);
    // For a diagonal F the isochoric stretches are the diagonal entries over J^(1/3).
    let f = diagonal(2.0, 3.0, 0.5);
    let j_cbrt = mat3_det(f).cbrt();
    let stretches = deviatoric_principal_stretches(f);
    let expected = [0.5 / j_cbrt, 2.0 / j_cbrt, 3.0 / j_cbrt];
    for (got, want) in stretches.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-14, "{stretches:?} vs {expected:?}");
    }
}

#[test]
fn energy_is_objective_and_isotropic() {
    let v = varga();
    let f = general_f();
    let w = v.strain_energy(f);
    assert!(w > 0.0);
    let q = rotation([1.0, 2.0, 3.0], 0.7);
    // Objectivity: W(Q F) = W(F) (rotation of the current configuration).
    let w_left = v.strain_energy(mat3_mul(q, f));
    assert!(
        (w_left - w).abs() <= 1e-10 * w,
        "W(QF) = {w_left}, W(F) = {w}"
    );
    // Isotropy: W(F Q) = W(F) (rotation of the reference configuration).
    let w_right = v.strain_energy(mat3_mul(f, q));
    assert!(
        (w_right - w).abs() <= 1e-10 * w,
        "W(FQ) = {w_right}, W(F) = {w}"
    );
}

#[test]
fn reference_state_is_stress_free() {
    let p = varga().pk1_stress(mat3_identity());
    assert!(frobenius(p) < 1e-6 * MU, "P(I) = {p:?}");
}

#[test]
fn small_shear_gives_initial_shear_modulus_mu() {
    // With the deviatoric prefactor 2 mu the shear modulus is mu (the old prefactor
    // mu gave mu/2, and its sqrt(3 I1_bar) proxy a modulus that depends on the proxy).
    let gamma = 1.0e-3;
    let p = varga().pk1_stress(simple_shear(gamma));
    let modulus = p[0][1] / gamma;
    assert!(
        (modulus - MU).abs() < 1e-4 * MU,
        "P12/gamma = {modulus}, expected mu = {MU}"
    );
}

#[test]
fn uniaxial_stress_difference_is_two_mu_times_stretch_difference() {
    // For W = 2 mu (λ1 + λ2 + λ3 - 3) at J = 1 the pressure-independent Cauchy stress
    // difference is σ1 - σ2 = λ1 ∂W/∂λ1 - λ2 ∂W/∂λ2 = 2 mu (λ1 - λ2).
    let lambda = 1.5_f64;
    let f = diagonal(lambda, lambda.powf(-0.5), lambda.powf(-0.5));
    let p = varga().pk1_stress(f);
    let sigma = pk1_to_cauchy(f, p);
    let difference = sigma[0][0] - sigma[1][1];
    let expected = 2.0 * MU * (lambda - lambda.powf(-0.5));
    assert!(
        (difference - expected).abs() < 1e-5 * expected,
        "sigma11 - sigma22 = {difference}, expected {expected}"
    );
    // The two lateral stresses are equal by symmetry.
    assert!((sigma[1][1] - sigma[2][2]).abs() < 1e-5 * expected);
}

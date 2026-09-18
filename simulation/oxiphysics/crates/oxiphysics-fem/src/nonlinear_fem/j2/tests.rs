// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the J2 return mapping (isotropic, Prager and Armstrong–Frederick
//! hardening) and of its consistent tangent.

use super::*;

const YOUNG: f64 = 200e9;
const NU: f64 = 0.3;
const SIGMA_Y0: f64 = 250e6;

fn steel(h_iso: f64, c_kin: f64, gamma_kin: f64) -> J2Material {
    let mut mat = J2Material::new(YOUNG, NU, SIGMA_Y0, h_iso);
    mat.c_kin = c_kin;
    mat.gamma_kin = gamma_kin;
    mat
}

/// Von Mises norm `√(3/2 a:a)` of a deviatoric stress-like tensor.
fn xi(a: &[f64; 6]) -> f64 {
    (1.5 * contract(a, a)).sqrt()
}

/// Equivalent norm of the relative stress `s − α` of a stress/state pair.
fn relative_stress_norm(sigma: &[f64; 6], state: &PlasticState) -> f64 {
    let s = J2Material::deviatoric(sigma);
    let eta: [f64; 6] = std::array::from_fn(|i| s[i] - state.alpha[i]);
    xi(&eta)
}

/// Equivalent plastic strain of the plastic strain increment between two states.
fn equivalent_increment(before: &PlasticState, after: &PlasticState) -> f64 {
    let mut counter = PlasticState::new();
    counter.update_eps_p_eq(&std::array::from_fn(|i| after.eps_p[i] - before.eps_p[i]));
    counter.eps_p_eq
}

fn assert_close(actual: f64, expected: f64, rel: f64, what: &str) {
    let scale = expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        (actual - expected).abs() <= rel * scale,
        "{what}: {actual} differs from {expected} by more than {rel:e} (relative)"
    );
}

#[test]
fn isotropic_return_lands_on_the_hardened_yield_surface() {
    let h = 1e9;
    let mat = steel(h, 0.0, 0.0);
    let g = mat.shear_modulus();
    let trial = mat.elastic_predictor(&[0.0; 6], &[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let (sigma, state, dg) = mat.return_mapping(&trial, &PlasticState::new());

    // A uniaxial strain step of 1% has the trial equivalent stress 2 G * 0.01.
    let xi_tr = 2.0 * g * 0.01;
    assert_close(dg, (xi_tr - SIGMA_Y0) / (3.0 * g + h), 1e-12, "Δγ");
    // The accumulated plastic strain grows by Δγ, not by 1.816 Δγ.
    assert_close(state.eps_p_eq, dg, 1e-14, "ε̄ᵖ");
    assert_close(
        equivalent_increment(&PlasticState::new(), &state),
        dg,
        1e-12,
        "√(2/3 Δεᵖ:Δεᵖ)",
    );
    // The returned stress is exactly on the hardened yield surface.
    let vm = J2Material::von_mises(&sigma);
    assert_close(
        vm,
        mat.yield_stress(state.eps_p_eq),
        1e-8,
        "von Mises stress",
    );
    assert!(state.alpha.iter().all(|a| *a == 0.0), "no back stress");
}

#[test]
fn plastic_flow_is_incompressible_and_leaves_the_pressure_unchanged() {
    let mat = steel(5e8, 0.0, 0.0);
    let trial = [4.0e8, -1.0e8, 2.5e8, 6.0e7, -3.0e7, 2.0e7];
    let (sigma, state, dg) = mat.return_mapping(&trial, &PlasticState::new());
    assert!(dg > 0.0);
    let p_trial = (trial[0] + trial[1] + trial[2]) / 3.0;
    let p = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
    assert_close(p, p_trial, 1e-12, "pressure");
    let trace = state.eps_p[0] + state.eps_p[1] + state.eps_p[2];
    assert!(trace.abs() < 1e-14, "tr εᵖ = {trace}");
}

#[test]
fn equivalent_plastic_strain_increment_matches_the_plastic_strain_tensor() {
    // Shear-dominated trial stress: checks the engineering-shear bookkeeping.
    let trial = [1.0e8, -4.0e7, 6.0e7, 4.5e8, 3.0e8, -2.0e8];
    for mat in [
        steel(1e9, 0.0, 0.0),
        steel(2e9, 15e9, 0.0),
        steel(1e9, 30e9, 80.0),
    ] {
        let (_, state, dg) = mat.return_mapping(&trial, &PlasticState::new());
        assert!(dg > 0.0);
        assert_close(state.eps_p_eq, dg, 1e-14, "ε̄ᵖ");
        assert_close(
            equivalent_increment(&PlasticState::new(), &state),
            dg,
            1e-12,
            "√(2/3 Δεᵖ:Δεᵖ)",
        );
    }
}

#[test]
fn perfect_plasticity_in_pure_shear_returns_to_the_von_mises_shear_yield() {
    let mat = steel(0.0, 0.0, 0.0);
    let g = mat.shear_modulus();
    // Engineering shear strain 1%: the trial shear stress G γ = 7.7e8 is far above yield.
    let trial = mat.elastic_predictor(&[0.0; 6], &[0.0, 0.0, 0.0, 0.01, 0.0, 0.0]);
    assert_close(trial[3], g * 0.01, 1e-14, "trial shear stress");
    let (sigma, state, _) = mat.return_mapping(&trial, &PlasticState::new());
    assert_close(
        sigma[3],
        SIGMA_Y0 / 3.0_f64.sqrt(),
        1e-12,
        "shear yield stress",
    );
    assert!(sigma[0].abs() + sigma[1].abs() + sigma[2].abs() < 1e-3);
    assert_close(J2Material::von_mises(&sigma), SIGMA_Y0, 1e-12, "von Mises");
    // The plastic strain is pure shear and carries the full overstress.
    assert!(state.eps_p[3] > 0.0);
    assert!(state.eps_p[0].abs() < 1e-15 && state.eps_p[4].abs() < 1e-15);
}

#[test]
fn prager_back_stress_is_finite_and_the_return_is_radial() {
    let c = 20e9;
    let mat = steel(0.0, c, 0.0);
    let g = mat.shear_modulus();
    let trial = mat.elastic_predictor(&[0.0; 6], &[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let (sigma, state, dg) = mat.return_mapping(&trial, &PlasticState::new());

    let xi_tr = 2.0 * g * 0.01;
    assert_close(dg, (xi_tr - SIGMA_Y0) / (3.0 * g + c), 1e-12, "Δγ");
    // With γ = 0 the back stress grows by C Δγ along the (radial) flow direction.
    assert!(
        state.alpha.iter().all(|a| a.is_finite()),
        "{:?}",
        state.alpha
    );
    assert_close(xi(&state.alpha), c * dg, 1e-12, "‖α‖");
    let s = J2Material::deviatoric(&sigma);
    let dir_err: f64 = (0..6)
        .map(|i| (s[i] / xi(&s) - state.alpha[i] / xi(&state.alpha)).abs())
        .sum();
    assert!(dir_err < 1e-12, "α is not parallel to s: {dir_err}");
    // The relative stress sits on the (non-hardening) yield surface.
    assert_close(
        relative_stress_norm(&sigma, &state),
        SIGMA_Y0,
        1e-9,
        "‖s − α‖",
    );
}

#[test]
fn combined_isotropic_and_prager_hardening_share_the_overstress() {
    let (h, c) = (3e9, 12e9);
    let mat = steel(h, c, 0.0);
    let g = mat.shear_modulus();
    let trial = mat.elastic_predictor(&[0.0; 6], &[0.0, 0.0, 0.0, 0.01, 0.004, 0.0]);
    let xi_tr = xi(&J2Material::deviatoric(&trial));
    let (sigma, state, dg) = mat.return_mapping(&trial, &PlasticState::new());
    assert_close(dg, (xi_tr - SIGMA_Y0) / (3.0 * g + h + c), 1e-12, "Δγ");
    assert_close(
        relative_stress_norm(&sigma, &state),
        mat.yield_stress(state.eps_p_eq),
        1e-9,
        "‖s − α‖",
    );
}

#[test]
fn armstrong_frederick_with_a_noncollinear_back_stress_stays_on_the_yield_surface() {
    // The recovery term rotates the flow direction away from the trial direction, so the
    // return is not radial; a radial (trial-direction) update would leave the stress off
    // the yield surface.
    let mat = steel(1e9, 30e9, 120.0);
    let state_n = PlasticState {
        eps_p: [0.0; 6],
        eps_p_eq: 0.002,
        alpha: [1.0e8, -3.0e7, -7.0e7, 2.0e7, -1.0e7, 5.0e6],
    };
    let sigma_n = [1.0e8, 2.0e8, -5.0e7, 3.0e7, 1.0e7, -2.0e7];
    let trial = mat.elastic_predictor(&sigma_n, &[0.004, -0.001, 0.0005, 0.003, -0.002, 0.001]);
    let (sigma, state, dg) = mat.return_mapping(&trial, &state_n);

    assert!(dg > 0.0);
    assert_close(state.eps_p_eq, state_n.eps_p_eq + dg, 1e-14, "ε̄ᵖ");
    assert_close(
        relative_stress_norm(&sigma, &state),
        mat.yield_stress(state.eps_p_eq),
        1e-10,
        "‖s − α‖ against σ_y",
    );
    assert_close(
        equivalent_increment(&state_n, &state),
        dg,
        1e-12,
        "√(2/3 Δεᵖ:Δεᵖ)",
    );
    // The back stress remains a deviator.
    let trace = state.alpha[0] + state.alpha[1] + state.alpha[2];
    assert!(trace.abs() < 1e-3, "tr α = {trace}");
}

#[test]
fn armstrong_frederick_reduces_to_prager_for_vanishing_recovery() {
    let prager = steel(1e9, 25e9, 0.0);
    let af = steel(1e9, 25e9, 1e-10);
    let trial = prager.elastic_predictor(&[0.0; 6], &[0.008, -0.003, 0.001, 0.004, 0.0, 0.0]);
    let (sigma_p, state_p, dg_p) = prager.return_mapping(&trial, &PlasticState::new());
    let (sigma_a, state_a, dg_a) = af.return_mapping(&trial, &PlasticState::new());
    assert_close(dg_a, dg_p, 1e-8, "Δγ");
    for i in 0..6 {
        assert!((sigma_a[i] - sigma_p[i]).abs() <= 1e-8 * sigma_p[0].abs());
        assert!((state_a.alpha[i] - state_p.alpha[i]).abs() <= 1e-8 * state_p.alpha[0].abs());
    }
}

#[test]
fn armstrong_frederick_back_stress_saturates_at_c_over_gamma() {
    let (c, gamma) = (30e9, 150.0);
    let mat = steel(0.0, c, gamma);
    let d_eps = [1e-4, -0.5e-4, -0.5e-4, 0.0, 0.0, 0.0];
    let mut sigma = [0.0; 6];
    let mut state = PlasticState::new();
    let mut checked_midway = false;
    for step in 0..3000 {
        let trial = mat.elastic_predictor(&sigma, &d_eps);
        let (s, st, _) = mat.return_mapping(&trial, &state);
        sigma = s;
        state = st;
        if step == 99 {
            // Monotonic proportional loading: X(ε̄ᵖ) = (C/γ)(1 − exp(−γ ε̄ᵖ)).
            let exact = c / gamma * (1.0 - (-gamma * state.eps_p_eq).exp());
            assert_close(xi(&state.alpha), exact, 2e-2, "back stress along the path");
            checked_midway = true;
        }
    }
    assert!(checked_midway);
    assert_close(xi(&state.alpha), c / gamma, 1e-3, "saturated back stress");
    // At saturation the flow stress stays bounded by σ_y0 + C/γ.
    assert!(J2Material::von_mises(&sigma) <= SIGMA_Y0 + c / gamma + 1e3);
}

#[test]
fn elastic_steps_are_returned_unchanged() {
    let mat = steel(1e9, 20e9, 100.0);
    let state_n = PlasticState {
        eps_p: [1e-4, -5e-5, -5e-5, 0.0, 0.0, 0.0],
        eps_p_eq: 1e-4,
        alpha: [1e6, -5e5, -5e5, 0.0, 0.0, 0.0],
    };
    let trial = mat.elastic_predictor(&[0.0; 6], &[1e-5, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let (sigma, state, dg) = mat.return_mapping(&trial, &state_n);
    assert_eq!(dg, 0.0);
    assert_eq!(sigma, trial);
    assert_eq!(state.eps_p, state_n.eps_p);
    assert_eq!(state.eps_p_eq, state_n.eps_p_eq);
    assert_eq!(state.alpha, state_n.alpha);
}

/// Central finite-difference tangent of the stress update with respect to the strain
/// increment (engineering shear).
fn finite_difference_tangent(
    mat: &J2Material,
    sigma_n: &[f64; 6],
    d_eps: &[f64; 6],
    state_n: &PlasticState,
) -> [[f64; 6]; 6] {
    let h = 1e-7;
    let mut tangent = [[0.0; 6]; 6];
    for j in 0..6 {
        let stress_at = |shift: f64| -> [f64; 6] {
            let mut de = *d_eps;
            de[j] += shift;
            let trial = mat.elastic_predictor(sigma_n, &de);
            mat.return_mapping(&trial, state_n).0
        };
        let plus = stress_at(h);
        let minus = stress_at(-h);
        for i in 0..6 {
            tangent[i][j] = (plus[i] - minus[i]) / (2.0 * h);
        }
    }
    tangent
}

fn assert_tangent_matches_finite_differences(
    mat: &J2Material,
    sigma_n: &[f64; 6],
    d_eps: &[f64; 6],
    state_n: &PlasticState,
) {
    let trial = mat.elastic_predictor(sigma_n, d_eps);
    let (_, _, dg) = mat.return_mapping(&trial, state_n);
    assert!(dg > 0.0, "the test point must be plastic");
    let analytic = mat.consistent_tangent(dg, &trial, state_n);
    let numeric = finite_difference_tangent(mat, sigma_n, d_eps, state_n);
    let scale = numeric
        .iter()
        .flatten()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    for i in 0..6 {
        for j in 0..6 {
            assert!(
                (analytic[i][j] - numeric[i][j]).abs() <= 1e-6 * scale,
                "C[{i}][{j}]: analytic {} vs finite difference {}",
                analytic[i][j],
                numeric[i][j]
            );
        }
    }
}

#[test]
fn consistent_tangent_matches_finite_differences_for_isotropic_hardening() {
    let mat = steel(1e9, 0.0, 0.0);
    assert_tangent_matches_finite_differences(
        &mat,
        &[0.0; 6],
        &[0.01, -0.002, 0.001, 0.003, 0.0, -0.001],
        &PlasticState::new(),
    );
}

#[test]
fn consistent_tangent_matches_finite_differences_for_prager_hardening() {
    let mat = steel(2e9, 20e9, 0.0);
    let state_n = PlasticState {
        eps_p: [0.0; 6],
        eps_p_eq: 0.001,
        alpha: [5.0e7, -2.0e7, -3.0e7, 1.0e7, 0.0, -4.0e6],
    };
    assert_tangent_matches_finite_differences(
        &mat,
        &[1.0e8, 2.0e8, -5.0e7, 3.0e7, 1.0e7, -2.0e7],
        &[0.004, -0.001, 0.0005, 0.003, -0.002, 0.001],
        &state_n,
    );
}

#[test]
fn consistent_tangent_matches_finite_differences_for_armstrong_frederick_hardening() {
    let mat = steel(1e9, 30e9, 120.0);
    let state_n = PlasticState {
        eps_p: [0.0; 6],
        eps_p_eq: 0.002,
        alpha: [1.0e8, -3.0e7, -7.0e7, 2.0e7, -1.0e7, 5.0e6],
    };
    assert_tangent_matches_finite_differences(
        &mat,
        &[1.0e8, 2.0e8, -5.0e7, 3.0e7, 1.0e7, -2.0e7],
        &[0.004, -0.001, 0.0005, 0.003, -0.002, 0.001],
        &state_n,
    );
}

#[test]
fn consistent_tangent_is_symmetric_without_dynamic_recovery() {
    for mat in [steel(1e9, 0.0, 0.0), steel(2e9, 20e9, 0.0)] {
        let state_n = PlasticState {
            alpha: [4.0e7, -1.5e7, -2.5e7, 0.0, 0.0, 0.0],
            ..PlasticState::new()
        };
        let trial = mat.elastic_predictor(&[0.0; 6], &[0.006, -0.002, 0.001, 0.002, 0.0, 0.0]);
        let (_, _, dg) = mat.return_mapping(&trial, &state_n);
        assert!(dg > 0.0);
        let c = mat.consistent_tangent(dg, &trial, &state_n);
        let scale = c.iter().flatten().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        for (i, row) in c.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert!(
                    (value - c[j][i]).abs() <= 1e-10 * scale,
                    "C[{i}][{j}] = {value} vs C[{j}][{i}] = {}",
                    c[j][i]
                );
            }
        }
    }
}

#[test]
fn consistent_tangent_softens_in_the_flow_direction() {
    // After yielding the shear stiffness of a pure-shear state drops far below G.
    let mat = steel(1e9, 0.0, 0.0);
    let g = mat.shear_modulus();
    let trial = mat.elastic_predictor(&[0.0; 6], &[0.0, 0.0, 0.0, 0.01, 0.0, 0.0]);
    let (_, _, dg) = mat.return_mapping(&trial, &PlasticState::new());
    let c = mat.consistent_tangent(dg, &trial, &PlasticState::new());
    assert!(c[3][3] < 0.2 * g, "C44 = {} vs G = {g}", c[3][3]);
    // The untouched shear directions keep a reduced but positive stiffness.
    assert!(c[4][4] > 0.0 && c[4][4] <= g);
}

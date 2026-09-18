//! Regression tests for the damage-model defects found in the production audit
//! (F1118 Lemaitre energy release rate, F1119 damage-plasticity return map).

use super::*;

/// Voigt stress \[xx, yy, zz, xy, yz, xz\] with von-Mises stress `sigma_eq` and mean
/// stress `sigma_h` (uniaxial deviator plus hydrostatic part).
fn stress_with(sigma_eq: f64, sigma_h: f64) -> [f64; 6] {
    [
        sigma_h + 2.0 * sigma_eq / 3.0,
        sigma_h - sigma_eq / 3.0,
        sigma_h - sigma_eq / 3.0,
        0.0,
        0.0,
        0.0,
    ]
}

fn deviator(s: &[f64; 6]) -> [f64; 6] {
    let mean = (s[0] + s[1] + s[2]) / 3.0;
    [s[0] - mean, s[1] - mean, s[2] - mean, s[3], s[4], s[5]]
}

fn von_mises_of(s: &[f64; 6]) -> f64 {
    (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2))
        + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
        .sqrt()
}

fn max_abs_diff(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// F1118 — LemaitreCDM energy release rate
// ---------------------------------------------------------------------------

fn lemaitre_cdm(d: f64, s: f64, s0: f64, young_modulus: f64, nu: f64) -> LemaitreCDM {
    LemaitreCDM {
        d,
        s,
        s0,
        young_modulus,
        nu,
    }
}

#[test]
fn lemaitre_cdm_rate_equals_thermodynamic_force_law() {
    // Identical inputs must give (Y / S)^s0 · Δp with the Y of LemaitreDamage.
    let cases = [
        (0.0, 300e6, 100e6, 200e9, 0.3, 2.0e6, 1.0),
        (0.25, 450e6, -60e6, 210e9, 0.28, 1.5e6, 2.0),
        (0.6, 120e6, 250e6, 70e9, 0.33, 8.0e5, 1.5),
        (0.0, 80e6, 0.0, 200e9, 0.5, 1.0e6, 1.0),
    ];
    let eps_p = 0.013;
    for (d, sigma_eq, sigma_h, e, nu, s, s0) in cases {
        let model = lemaitre_cdm(d, s, s0, e, nu);
        let stress = stress_with(sigma_eq, sigma_h);
        let y = LemaitreDamage::thermodynamic_force(&stress, d, e, nu);
        let expected = (y / s).powf(s0) * eps_p;
        let rate = model.evolution_rate(eps_p, sigma_eq, sigma_h);
        assert!(
            (rate - expected).abs() <= 1e-12 * expected.abs(),
            "rate {rate:e} != {expected:e} for D={d}, σ_eq={sigma_eq}, σ_h={sigma_h}"
        );
        assert!(
            (model.energy_release_rate(sigma_eq, sigma_h) - y).abs() <= 1e-12 * y,
            "Y mismatch"
        );
    }
}

#[test]
fn lemaitre_cdm_energy_release_rate_closed_forms() {
    // Uniaxial stress: R_ν = 1 for every ν, so Y = σ² / (2E (1−D)²).
    let sigma = 400e6;
    let model = lemaitre_cdm(0.0, 1.0e6, 1.0, 200e9, 0.3);
    let y_uniaxial = model.energy_release_rate(sigma, sigma / 3.0);
    assert!((y_uniaxial - sigma * sigma / (2.0 * 200e9)).abs() <= 1e-12 * y_uniaxial);
    // Damage amplifies Y by (1−D)⁻².
    let damaged = lemaitre_cdm(0.5, 1.0e6, 1.0, 200e9, 0.3);
    let y_damaged = damaged.energy_release_rate(sigma, sigma / 3.0);
    assert!((y_damaged - 4.0 * y_uniaxial).abs() <= 1e-12 * y_damaged);
    // High triaxiality T = σ_h/σ_eq = 2, ν = 0.25: R_ν = ⅔·1.25 + 3·0.5·4.
    let nu = 0.25;
    let triaxial = lemaitre_cdm(0.0, 1.0e6, 1.0, 200e9, nu);
    let r_nu = 2.0 / 3.0 * (1.0 + nu) + 3.0 * (1.0 - 2.0 * nu) * 4.0;
    let y_triaxial = triaxial.energy_release_rate(sigma, 2.0 * sigma);
    assert!((y_triaxial - sigma * sigma * r_nu / (2.0 * 200e9)).abs() <= 1e-12 * y_triaxial);
    assert!(y_triaxial > 6.0 * y_uniaxial, "triaxiality must raise Y");
}

#[test]
fn lemaitre_cdm_rate_uses_modulus_not_strength() {
    // The former code divided by S inside Y; now Y is S-independent and the
    // rate scales exactly as (1/S)^s0 and as E^(−s0).
    let (sigma_eq, sigma_h, eps_p) = (350e6, 90e6, 0.02);
    let s0 = 2.0;
    let base = lemaitre_cdm(0.1, 1.0e6, s0, 200e9, 0.3).evolution_rate(eps_p, sigma_eq, sigma_h);
    let stronger =
        lemaitre_cdm(0.1, 2.0e6, s0, 200e9, 0.3).evolution_rate(eps_p, sigma_eq, sigma_h);
    assert!((stronger - base / 4.0).abs() <= 1e-12 * base, "S scaling");
    let stiffer = lemaitre_cdm(0.1, 1.0e6, s0, 400e9, 0.3).evolution_rate(eps_p, sigma_eq, sigma_h);
    assert!((stiffer - base / 4.0).abs() <= 1e-12 * base, "E scaling");
}

#[test]
fn lemaitre_cdm_guards_degenerate_inputs() {
    let model = lemaitre_cdm(0.0, 1.0e6, 1.0, 200e9, 0.3);
    assert_eq!(model.evolution_rate(0.01, 0.0, 0.0), 0.0);
    assert_eq!(
        lemaitre_cdm(0.0, 0.0, 1.0, 200e9, 0.3).evolution_rate(0.01, 1e8, 0.0),
        0.0
    );
    assert_eq!(
        lemaitre_cdm(0.0, 1.0e6, 1.0, 0.0, 0.3).evolution_rate(0.01, 1e8, 0.0),
        0.0
    );
}

// ---------------------------------------------------------------------------
// F1119 — damage-plasticity return map with back stress
// ---------------------------------------------------------------------------

fn dp_params(gamma_kin: f64) -> CoupledDamagePlasticityParams {
    CoupledDamagePlasticityParams {
        e: 200e9,
        nu: 0.3,
        sigma_y0: 250e6,
        h: 1e9,
        q: 100e6,
        b_hard: 5.0,
        c_kin: 30e9,
        gamma_kin,
        s_dmg: 1.0e6,
        s_dmg_exp: 1.0,
        d_c: 0.8,
    }
}

/// Back stress with a shear component, i.e. not coaxial with a uniaxial trial deviator.
const BACK_STRESS: [f64; 6] = [40e6, -15e6, -25e6, 30e6, -12e6, 8e6];

fn state_with_back_stress(d: f64, p_bar: f64) -> DamagePlasticityState {
    DamagePlasticityState {
        d,
        p_bar,
        r: 0.0,
        back_stress: BACK_STRESS,
        effective_stress: [0.0; 6],
    }
}

#[test]
fn return_map_lands_on_yield_surface_with_back_stress_prager() {
    // γ = 0 (Prager kinematic hardening) with X ≠ 0 and a non-coaxial trial stress:
    // the old map used the unshifted deviator and left the surface by O(|X|).
    let model = CoupledDamagePlasticity::new(dp_params(0.0));
    let state = state_with_back_stress(0.0, 0.002);
    let sigma_trial = [520e6, 20e6, -10e6, 25e6, 0.0, 0.0];
    let f_trial = model.yield_function(&sigma_trial, &state.back_stress, state.p_bar);
    assert!(
        f_trial > 0.0,
        "test needs a plastic step, f_trial = {f_trial}"
    );
    let (new_state, converged) = model.return_mapping(&state, &sigma_trial);
    assert!(converged);
    assert!(new_state.p_bar > state.p_bar);
    let f_new = model.yield_function(
        &new_state.effective_stress,
        &new_state.back_stress,
        new_state.p_bar,
    );
    assert!(
        f_new.abs() <= 1e-8 * model.sigma_y0,
        "state left the yield surface: f = {f_new:e}"
    );
}

#[test]
fn return_map_prager_kinematic_modulus_enters_the_update() {
    // For γ = 0: X⁺ − X = ⅔C·Δp·N and σ̃_trial − σ̃⁺ = 2G·Δp·N, hence
    // 3G·(X⁺ − X) = C·(σ̃_trial − σ̃⁺) componentwise.
    let model = CoupledDamagePlasticity::new(dp_params(0.0));
    let state = state_with_back_stress(0.0, 0.0);
    let sigma_trial = [480e6, -30e6, 10e6, 18e6, 5e6, -9e6];
    let (new_state, converged) = model.return_mapping(&state, &sigma_trial);
    assert!(converged);
    let g = model.shear_modulus();
    for (i, &trial) in sigma_trial.iter().enumerate() {
        let lhs = 3.0 * g * (new_state.back_stress[i] - state.back_stress[i]);
        let rhs = model.c_kin * (trial - new_state.effective_stress[i]);
        assert!(
            (lhs - rhs).abs() <= 1e-6 * (lhs.abs() + rhs.abs()).max(1e3),
            "component {i}: 3G·ΔX = {lhs:e}, C·Δσ = {rhs:e}"
        );
    }
}

#[test]
fn return_map_linear_hardening_matches_closed_form_multiplier() {
    // Q = 0, γ = 0: Δp̄ = f_trial / (3G + C + H) exactly (linear return map).
    let mut params = dp_params(0.0);
    params.q = 0.0;
    let model = CoupledDamagePlasticity::new(params);
    let state = state_with_back_stress(0.0, 0.0);
    let sigma_trial = [610e6, 0.0, 0.0, 40e6, 0.0, 0.0];
    let f_trial = model.yield_function(&sigma_trial, &state.back_stress, 0.0);
    let (new_state, converged) = model.return_mapping(&state, &sigma_trial);
    assert!(converged);
    let expected = f_trial / (3.0 * model.shear_modulus() + model.c_kin + model.h);
    assert!(
        (new_state.p_bar - expected).abs() <= 1e-9 * expected,
        "Δp̄ = {:e}, closed form {expected:e}",
        new_state.p_bar
    );
}

#[test]
fn return_map_with_recall_satisfies_consistency_and_backward_euler_identities() {
    // γ > 0: the implicit Armstrong-Frederick update must satisfy
    //   f(σ̃⁺, X⁺, p̄⁺) = 0,
    //   (1 + γΔp̄)·X⁺ = X + ⅔C·Δp̄·N,   σ̃_trial − σ̃⁺ = 2G·Δp̄·N
    // with N = 3/2 dev(σ̃⁺ − X⁺) / σ_y(p̄⁺).
    for gamma_kin in [50.0, 400.0] {
        let model = CoupledDamagePlasticity::new(dp_params(gamma_kin));
        let state = state_with_back_stress(0.0, 0.004);
        let sigma_trial = [560e6, 35e6, -20e6, 45e6, -15e6, 10e6];
        let (new_state, converged) = model.return_mapping(&state, &sigma_trial);
        assert!(converged, "γ = {gamma_kin}");
        let dp = new_state.p_bar - state.p_bar;
        assert!(dp > 0.0);
        let f_new = model.yield_function(
            &new_state.effective_stress,
            &new_state.back_stress,
            new_state.p_bar,
        );
        assert!(
            f_new.abs() <= 1e-8 * model.sigma_y0,
            "γ = {gamma_kin}: f = {f_new:e}"
        );
        let mut xi = [0.0; 6];
        for (k, x) in xi.iter_mut().enumerate() {
            *x = new_state.effective_stress[k] - new_state.back_stress[k];
        }
        let dev_xi = deviator(&xi);
        let sigma_y = model.yield_stress(new_state.p_bar);
        let n_dir: Vec<f64> = dev_xi.iter().map(|v| 1.5 * v / sigma_y).collect();
        let g = model.shear_modulus();
        for i in 0..6 {
            let plastic_flow = 2.0 * g * dp * n_dir[i];
            let stress_drop = sigma_trial[i] - new_state.effective_stress[i];
            // The trial stress is the nominal one and D = 0, so σ̃_trial = σ_trial.
            assert!(
                (stress_drop - plastic_flow).abs() <= 1e-6 * stress_drop.abs().max(1e6),
                "γ = {gamma_kin}, stress component {i}: {stress_drop:e} vs {plastic_flow:e}"
            );
            let lhs = (1.0 + gamma_kin * dp) * new_state.back_stress[i];
            let rhs = state.back_stress[i] + 2.0 / 3.0 * model.c_kin * dp * n_dir[i];
            assert!(
                (lhs - rhs).abs() <= 1e-6 * lhs.abs().max(1e6),
                "γ = {gamma_kin}, back-stress component {i}: {lhs:e} vs {rhs:e}"
            );
        }
        // The recall term must reduce the back stress relative to the Prager update.
        let prager = CoupledDamagePlasticity::new(dp_params(0.0));
        let (prager_state, _) = prager.return_mapping(&state, &sigma_trial);
        assert!(
            von_mises_of(&new_state.back_stress) < von_mises_of(&prager_state.back_stress),
            "γ = {gamma_kin} must limit the back stress"
        );
    }
}

#[test]
fn return_map_is_isochoric_and_keeps_back_stress_deviatoric() {
    let model = CoupledDamagePlasticity::new(dp_params(120.0));
    let state = state_with_back_stress(0.0, 0.0);
    let sigma_trial = [530e6, 60e6, 15e6, 22e6, 0.0, -7e6];
    let (new_state, converged) = model.return_mapping(&state, &sigma_trial);
    assert!(converged);
    let trace_trial = sigma_trial[0] + sigma_trial[1] + sigma_trial[2];
    let trace_new: f64 = new_state.effective_stress[..3].iter().sum();
    assert!((trace_new - trace_trial).abs() <= 1e-6 * trace_trial.abs());
    let trace_x: f64 = new_state.back_stress[..3].iter().sum();
    assert!(trace_x.abs() <= 1e-6 * 40e6);
}

#[test]
fn return_map_isotropic_variable_follows_the_voce_law() {
    let model = CoupledDamagePlasticity::new(dp_params(0.0));
    let state = DamagePlasticityState::default();
    let sigma_trial = [700e6, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (new_state, converged) = model.return_mapping(&state, &sigma_trial);
    assert!(converged);
    let p = new_state.p_bar;
    let expected = model.h * p + model.q * (1.0 - (-model.b_hard * p).exp());
    assert!(
        (new_state.r - expected).abs() <= 1e-9 * expected,
        "R = {:e}, expected {expected:e}",
        new_state.r
    );
    assert!(
        new_state.r > model.h * p * 1.01,
        "the Voce saturation term was ignored: R = {:e}",
        new_state.r
    );
}

#[test]
fn return_map_damage_uses_effective_stress_energy_release_rate() {
    // With D > 0 the energy release rate must be Y = σ_eq²R_ν/(2E(1−D)²) of the
    // *nominal* stress, i.e. σ̃²R_ν/(2E) of the returned effective stress; the former
    // code applied the (1−D)⁻² factor a second time.
    let model = CoupledDamagePlasticity::new(dp_params(80.0));
    let d0 = 0.3;
    let state = state_with_back_stress(d0, 0.001);
    let sigma_trial = [330e6, 12e6, -8e6, 14e6, 0.0, 0.0];
    let (new_state, converged) = model.return_mapping(&state, &sigma_trial);
    assert!(converged);
    let dp = new_state.p_bar - state.p_bar;
    assert!(dp > 0.0);
    let nominal_new: [f64; 6] = std::array::from_fn(|i| (1.0 - d0) * new_state.effective_stress[i]);
    let y = LemaitreDamage::thermodynamic_force(&nominal_new, d0, model.e, model.nu);
    let expected_dd = (y / model.s_dmg).powf(model.s_dmg_exp) * dp;
    assert!(expected_dd > 0.0 && d0 + expected_dd < model.d_c);
    assert!(
        (new_state.d - d0 - expected_dd).abs() <= 1e-9 * expected_dd,
        "ΔD = {:e}, expected {expected_dd:e}",
        new_state.d - d0
    );
}

#[test]
fn return_map_elastic_step_keeps_state() {
    let model = CoupledDamagePlasticity::new(dp_params(100.0));
    let state = state_with_back_stress(0.2, 0.01);
    let sigma_trial = [60e6, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (new_state, converged) = model.return_mapping(&state, &sigma_trial);
    assert!(converged);
    assert_eq!(new_state.p_bar, state.p_bar);
    assert_eq!(new_state.d, state.d);
    assert_eq!(new_state.back_stress, state.back_stress);
    let factor = 1.0 / (1.0 - state.d);
    let expected: [f64; 6] = std::array::from_fn(|i| sigma_trial[i] * factor);
    assert!(max_abs_diff(&new_state.effective_stress, &expected) < 1e-6);
}

#[test]
fn return_map_reports_non_convergence_instead_of_claiming_success() {
    // Softening steeper than the elastic + kinematic stiffness (H < −(3G + C)) has no
    // consistent plastic multiplier; the map must say so and keep the trial state.
    let mut params = dp_params(0.0);
    params.h = -1.0e12;
    let model = CoupledDamagePlasticity::new(params);
    let state = DamagePlasticityState::default();
    let sigma_trial = [400e6, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (new_state, converged) = model.return_mapping(&state, &sigma_trial);
    assert!(
        !converged,
        "an unsolvable return map must report converged = false"
    );
    assert_eq!(new_state.p_bar, 0.0);
    assert!(max_abs_diff(&new_state.effective_stress, &sigma_trial) < 1e-6);
}

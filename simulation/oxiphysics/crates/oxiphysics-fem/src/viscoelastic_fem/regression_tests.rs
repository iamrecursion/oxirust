//! Regression tests for the viscoelastic FEM module.
//!
//! Covers the Standard Linear Solid time constants and complex moduli (checked
//! against the complex compliance of the series spring + Kelvin-Voigt model),
//! the incremental time stepping of [`ViscoelasticBeam1D`] (creep, equilibrium,
//! unloading and time-temperature superposition) and the total-strain
//! bookkeeping of [`PronyFemElement`].

use super::*;

// ── helpers ──────────────────────────────────────────────────────────────────

fn sls() -> StandardLinearSolid {
    StandardLinearSolid::new(2.0e9, 3.0e9, 1.0e7)
}

/// Four-element rubber bar of length 1 m and area 1e-4 m².
fn rubber_beam() -> ViscoelasticBeam1D {
    ViscoelasticBeam1D::new(4, 1.0, 1.0e-4, ViscoelasticMaterial::rubber())
}

/// External force vector with a single tip load.
fn tip_load(n_nodes: usize, force: f64) -> Vec<f64> {
    let mut forces = vec![0.0; n_nodes];
    forces[n_nodes - 1] = force;
    forces
}

/// Zener-type material: E_inf in parallel with one Maxwell arm.
fn zener_material() -> ViscoelasticMaterial {
    ViscoelasticMaterial {
        e_inf: 2.0e6,
        prony_terms: vec![PronyTerm {
            modulus: 6.0e6,
            relaxation_time: 2.0,
        }],
        nu: 0.3,
    }
}

/// Analytic creep compliance of [`zener_material`]:
/// J(t) = 1/E_inf - (1/E_inf - 1/E_0) exp(-t / tau_c) with tau_c = tau E_0 / E_inf.
fn zener_creep_compliance(t: f64) -> f64 {
    let e_inf = 2.0e6;
    let e0 = 8.0e6;
    let tau_c = 2.0 * e0 / e_inf;
    1.0 / e_inf - (1.0 / e_inf - 1.0 / e0) * (-t / tau_c).exp()
}

/// Relative tip-displacement error of the beam creep response against the
/// analytic Zener creep curve at the given times, for a step size `dt`.
fn zener_creep_errors(dt: f64, times: &[f64]) -> Vec<f64> {
    let (length, area, force) = (1.0, 1.0e-4, 10.0);
    let mut beam = ViscoelasticBeam1D::new(4, length, area, zener_material());
    let forces = tip_load(beam.n_nodes, force);
    let last_step = (times.iter().cloned().fold(0.0_f64, f64::max) / dt).round() as usize;
    let mut tip = Vec::with_capacity(last_step);
    for _ in 0..last_step {
        let u = beam.solve_step(&forces, dt);
        tip.push(u[beam.n_nodes - 1]);
    }
    times
        .iter()
        .map(|&t| {
            let idx = (t / dt).round() as usize - 1;
            let exact = length * force / area * zener_creep_compliance(t);
            (tip[idx] - exact).abs() / exact
        })
        .collect()
}

// ── F1407: Standard Linear Solid time constants ─────────────────────────────

#[test]
fn sls_relaxation_time_is_eta_over_sum_of_moduli() {
    let tau_r = sls().relaxation_time();
    assert!((tau_r - 2.0e-3).abs() < 1e-15, "tau_r = {tau_r}");
}

#[test]
fn sls_relaxation_time_is_infinite_without_stiffness() {
    let model = StandardLinearSolid::new(1.0e9, -1.0e9, 1.0e7);
    assert!(model.relaxation_time().is_infinite());
}

#[test]
fn sls_retardation_to_relaxation_ratio_equals_modulus_ratio() {
    for &(e1, e2, eta) in &[
        (2.0e9, 3.0e9, 1.0e7),
        (1.0e6, 4.0e6, 50.0),
        (7.5e10, 2.0e9, 3.0e5),
    ] {
        let model = StandardLinearSolid::new(e1, e2, eta);
        let ratio = model.retardation_time() / model.relaxation_time();
        let expected = model.instantaneous_modulus() / model.equilibrium_modulus();
        assert!(
            (ratio / expected - 1.0).abs() < 1e-12,
            "tau_c/tau_r = {ratio}, E_0/E_inf = {expected}"
        );
        assert!(
            model.retardation_time() > model.relaxation_time(),
            "creep must be slower than relaxation"
        );
    }
}

#[test]
fn sls_relaxation_modulus_obeys_series_spring_kelvin_voigt_ode() {
    // Under a held strain eps0 the model obeys
    //   d(sigma)/dt = (E1 E2 eps0 - (E1 + E2) sigma) / eta.
    let model = sls();
    let eps0 = 1.0e-3;
    let tau_r = model.relaxation_time();
    let sigma = |t: f64| eps0 * model.relaxation_modulus(t);
    for &fraction in &[0.1, 0.5, 1.0, 3.0] {
        let t = fraction * tau_r;
        let h = 1.0e-6 * tau_r;
        let d_sigma = (sigma(t + h) - sigma(t - h)) / (2.0 * h);
        let rhs = (model.e1 * model.e2 * eps0 - (model.e1 + model.e2) * sigma(t)) / model.eta;
        assert!(
            ((d_sigma - rhs) / rhs).abs() < 1e-6,
            "t = {fraction} tau_r: d_sigma = {d_sigma:e}, rhs = {rhs:e}"
        );
    }
}

#[test]
fn sls_complex_modulus_matches_series_compliance() {
    // E*(w) = 1 / J*(w) with J*(w) = 1/E1 + 1/(E2 + i w eta).
    let model = sls();
    for &omega in &[1.0, 100.0, 500.0, 2.0e3, 1.0e5] {
        let den = model.e2 * model.e2 + (omega * model.eta).powi(2);
        let j_re = 1.0 / model.e1 + model.e2 / den;
        let j_im = -omega * model.eta / den;
        let mag2 = j_re * j_re + j_im * j_im;
        let storage = j_re / mag2;
        let loss = -j_im / mag2;
        assert!(
            (model.storage_modulus(omega) / storage - 1.0).abs() < 1e-10,
            "E'({omega}) = {} vs {storage}",
            model.storage_modulus(omega)
        );
        assert!(
            (model.loss_modulus(omega) / loss - 1.0).abs() < 1e-10,
            "E''({omega}) = {} vs {loss}",
            model.loss_modulus(omega)
        );
        assert!(
            (model.loss_tangent(omega) / (loss / storage) - 1.0).abs() < 1e-10,
            "tan(delta)({omega})"
        );
    }
}

// ── F1408: incremental beam stepping ────────────────────────────────────────

#[test]
fn beam_constant_load_creeps_monotonically_to_equilibrium() {
    let material = ViscoelasticMaterial::rubber();
    let (length, area, force) = (1.0, 1.0e-4, 10.0);
    let mut beam = ViscoelasticBeam1D::new(4, length, area, material.clone());
    let forces = tip_load(beam.n_nodes, force);
    let u_eq = force * length / (material.e_inf * area);
    let u_inst = force * length / (material.instantaneous_modulus() * area);
    let mut previous = 0.0;
    for step in 0..1000 {
        let u = beam.solve_step(&forces, 1.0);
        let tip = u[beam.n_nodes - 1];
        assert!(
            tip >= previous,
            "creep must not decrease (step {step}: {tip} < {previous})"
        );
        assert!(tip <= u_eq * (1.0 + 1e-9), "step {step}: {tip} > {u_eq}");
        previous = tip;
    }
    assert!(
        previous > u_inst * 5.0,
        "the bar must have crept far beyond the instantaneous response"
    );
    assert!(
        (previous / u_eq - 1.0).abs() < 1e-8,
        "tip displacement {previous} must converge to F L / (E_inf A) = {u_eq}"
    );
    assert_eq!(beam.displacement().len(), beam.n_nodes);
    assert!((beam.displacement()[beam.n_nodes - 1] - previous).abs() < 1e-15);
}

#[test]
fn beam_internal_forces_balance_external_load_at_every_step() {
    let mut beam = rubber_beam();
    let force = 25.0;
    let forces = tip_load(beam.n_nodes, force);
    for step in 0..40 {
        beam.solve_step(&forces, 0.5);
        let f_int = beam.assemble_internal_forces();
        let n = f_int.len();
        assert!(
            (f_int[n - 1] - force).abs() < 1e-6 * force,
            "step {step}: tip force {} != {force}",
            f_int[n - 1]
        );
        assert!(
            (f_int[0] + force).abs() < 1e-6 * force,
            "step {step}: reaction {} != {}",
            f_int[0],
            -force
        );
        for (i, &fi) in f_int[1..n - 1].iter().enumerate() {
            assert!(
                fi.abs() < 1e-6 * force,
                "step {step}: interior node {} out of balance: {fi}",
                i + 1
            );
        }
    }
}

#[test]
fn beam_first_step_with_tiny_dt_is_instantaneously_elastic() {
    let material = ViscoelasticMaterial::rubber();
    let (length, area, force) = (1.0, 1.0e-4, 10.0);
    let mut beam = ViscoelasticBeam1D::new(4, length, area, material.clone());
    let u = beam.solve_step(&tip_load(beam.n_nodes, force), 1.0e-9);
    let u_inst = force * length / (material.instantaneous_modulus() * area);
    assert!(
        (u[4] / u_inst - 1.0).abs() < 1e-6,
        "tip {} vs glassy response {u_inst}",
        u[4]
    );
    assert!(u[0].abs() < 1e-12 * u[4], "clamped end moved: {}", u[0]);
}

#[test]
fn beam_creep_follows_zener_compliance_with_first_order_convergence() {
    let times = [0.5, 1.0, 4.0, 8.0];
    let coarse = zener_creep_errors(0.01, &times);
    let fine = zener_creep_errors(0.005, &times);
    for (i, &t) in times.iter().enumerate() {
        assert!(
            coarse[i] < 3.0e-3,
            "t = {t}: relative error {} at dt = 0.01",
            coarse[i]
        );
        assert!(
            fine[i] < 0.65 * coarse[i],
            "t = {t}: error must shrink with dt ({} -> {})",
            coarse[i],
            fine[i]
        );
    }
}

#[test]
fn beam_unloading_recovers_viscoelastically() {
    let mut beam = rubber_beam();
    let force = 10.0;
    let load = tip_load(beam.n_nodes, force);
    let unload = vec![0.0; beam.n_nodes];
    for _ in 0..300 {
        beam.solve_step(&load, 1.0);
    }
    let peak = beam.displacement()[4];
    let after_unload = beam.solve_step(&unload, 1.0)[4];
    assert!(
        after_unload < peak && after_unload > 0.0,
        "elastic recovery is partial: {after_unload} vs peak {peak}"
    );
    for (i, fi) in beam.assemble_internal_forces().iter().enumerate() {
        assert!(
            fi.abs() < 1e-6 * force,
            "node {i} out of balance after unloading: {fi}"
        );
    }
    for _ in 0..2000 {
        beam.solve_step(&unload, 1.0);
    }
    assert!(
        beam.displacement()[4].abs() < 1e-3 * peak,
        "bar must recover fully: {} vs peak {peak}",
        beam.displacement()[4]
    );
}

#[test]
fn beam_tolerates_short_and_long_force_vectors() {
    let mut beam = rubber_beam();
    let u = beam.solve_step(&[], 1.0);
    assert!(u.iter().all(|&v| v == 0.0), "no load, no displacement");
    let mut padded = tip_load(beam.n_nodes, 10.0);
    padded.extend_from_slice(&[1.0e6, 2.0e6]);
    let mut reference = rubber_beam();
    let expected = reference.solve_step(&tip_load(reference.n_nodes, 10.0), 1.0);
    let got = beam.solve_step(&padded, 1.0);
    for (a, b) in got.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-15, "extra force entries must be ignored");
    }
}

#[test]
fn beam_temperature_step_at_reference_temperature_matches_isothermal_step() {
    let wlf = WlfParameters::typical_polymer();
    let mut isothermal = rubber_beam();
    let mut shifted = rubber_beam();
    let forces = tip_load(isothermal.n_nodes, 10.0);
    for _ in 0..8 {
        let ua = isothermal.solve_step(&forces, 0.7);
        let ub = shifted.solve_step_with_temperature(&forces, 0.7, &wlf, wlf.t_ref);
        for (a, b) in ua.iter().zip(ub.iter()) {
            assert!((a - b).abs() <= 1e-12 * a.abs().max(1e-30), "{a} vs {b}");
        }
    }
}

#[test]
fn beam_temperature_step_accelerates_creep_and_stays_in_equilibrium() {
    let wlf = WlfParameters::typical_polymer();
    let force = 10.0;
    let mut cold = rubber_beam();
    let mut hot = rubber_beam();
    let forces = tip_load(cold.n_nodes, force);
    let u_eq = force / (ViscoelasticMaterial::rubber().e_inf * 1.0e-4);
    for _ in 0..6 {
        cold.solve_step_with_temperature(&forces, 0.5, &wlf, wlf.t_ref);
        hot.solve_step_with_temperature(&forces, 0.5, &wlf, wlf.t_ref + 20.0);
        let f_int = hot.assemble_internal_forces();
        let n = f_int.len();
        assert!((f_int[n - 1] - force).abs() < 1e-6 * force, "tip force");
        assert!((f_int[0] + force).abs() < 1e-6 * force, "reaction");
    }
    let cold_tip = cold.displacement()[4];
    let hot_tip = hot.displacement()[4];
    assert!(
        hot_tip > cold_tip,
        "a hotter bar creeps faster: {hot_tip} vs {cold_tip}"
    );
    assert!(
        (hot_tip / u_eq - 1.0).abs() < 1e-6,
        "reduced time is far beyond every relaxation time: {hot_tip} vs {u_eq}"
    );
}

#[test]
fn element_history_stress_plus_tangent_reproduces_stress_update() {
    let mut element = ViscoelasticElement1D::new(1.0, 1.0e-4, ViscoelasticMaterial::polymer());
    let dt = 0.8;
    for &d_eps in &[1.0e-3, 4.0e-4, 0.0, -2.0e-4, 3.0e-4] {
        let predicted = element.history_stress(dt) + element.algorithmic_modulus(dt) * d_eps;
        let updated = element.stress_update(d_eps, dt);
        assert!(
            (predicted - updated).abs() <= 1e-12 * updated.abs().max(1.0),
            "{predicted} vs {updated}"
        );
    }
}

// ── F1409: PronyFemElement total strain ─────────────────────────────────────

fn two_arm_prony() -> PronyRelaxation {
    PronyRelaxation {
        e_inf: 1.0e9,
        elements: vec![
            PronyElement {
                e_i: 0.5e9,
                tau_i: 1.0,
            },
            PronyElement {
                e_i: 0.3e9,
                tau_i: 10.0,
            },
        ],
    }
}

#[test]
fn prony_fem_first_step_stress_matches_closed_form() {
    let prony = two_arm_prony();
    let dt = 0.1;
    let eps0 = 1.0e-3;
    let mut element = PronyFemElement::new([0, 1, 2, 3], prony.clone(), dt);
    let stress = element.update_stresses(&[eps0]);
    let tangent: f64 = prony.e_inf
        + prony
            .elements
            .iter()
            .map(|arm| {
                let xi = (-dt / arm.tau_i).exp();
                arm.e_i * (1.0 - xi) * arm.tau_i / dt
            })
            .sum::<f64>();
    assert!(
        (stress[0] / (tangent * eps0) - 1.0).abs() < 1e-12,
        "first-step stress {} vs {}",
        stress[0],
        tangent * eps0
    );
    assert!((element.state.strain[0] - eps0).abs() < 1e-18);
}

#[test]
fn prony_fem_held_strain_relaxes_to_long_term_elastic_stress() {
    let prony = two_arm_prony();
    let eps0 = 1.0e-3;
    let mut element = PronyFemElement::new([0, 1, 2, 3], prony.clone(), 1.0);
    let first = element.update_stresses(&[eps0])[0];
    let mut stress = first;
    let mut previous = first;
    for _ in 0..300 {
        stress = element.update_stresses(&[0.0])[0];
        assert!(stress <= previous + 1e-9, "stress must not grow");
        previous = stress;
    }
    let long_term = prony.e_inf * eps0;
    assert!(
        first > long_term,
        "the glassy response exceeds the long-term one"
    );
    assert!(
        (stress / long_term - 1.0).abs() < 1e-9,
        "held strain: stress {stress} must tend to E_inf eps0 = {long_term}"
    );
}

#[test]
fn prony_fem_stress_accumulates_total_strain() {
    let prony = PronyRelaxation {
        e_inf: 2.0e9,
        elements: Vec::new(),
    };
    let mut element = PronyFemElement::new([0, 1, 2, 3], prony, 0.1);
    let mut total = 0.0;
    for _ in 0..10 {
        let stress = element.update_stresses(&[1.0e-4])[0];
        total += 1.0e-4;
        assert!(
            (stress / (2.0e9 * total) - 1.0).abs() < 1e-12,
            "elastic stress must follow the total strain {total}, got {stress}"
        );
    }
}

#[test]
fn prony_fem_matches_the_one_dimensional_bar_element() {
    let material = ViscoelasticMaterial::polymer();
    let prony = PronyRelaxation {
        e_inf: material.e_inf,
        elements: material
            .prony_terms
            .iter()
            .map(|term| PronyElement {
                e_i: term.modulus,
                tau_i: term.relaxation_time,
            })
            .collect(),
    };
    let dt = 0.5;
    let mut fem = PronyFemElement::new([0, 1, 2, 3], prony, dt);
    let mut bar = ViscoelasticElement1D::new(1.0, 1.0e-4, material);
    for &d_eps in &[1.0e-3, 5.0e-4, 0.0, -2.0e-4, 3.0e-4, 0.0, 0.0, 1.0e-4] {
        let from_fem = fem.update_stresses(&[d_eps])[0];
        let from_bar = bar.stress_update(d_eps, dt);
        assert!(
            (from_fem - from_bar).abs() <= 1e-12 * from_bar.abs().max(1.0),
            "{from_fem} vs {from_bar}"
        );
    }
}

#[test]
fn prony_fem_components_are_independent_and_state_resizes() {
    let prony = two_arm_prony();
    let mut multi = PronyFemElement::new([0, 1, 2, 3], prony.clone(), 0.2);
    let mut single_a = PronyFemElement::new([0, 1, 2, 3], prony.clone(), 0.2);
    let mut single_b = PronyFemElement::new([0, 1, 2, 3], prony, 0.2);
    for &(a, b) in &[(1.0e-3, -2.0e-4), (0.0, 5.0e-4), (2.0e-4, 0.0)] {
        let stress = multi.update_stresses(&[a, b]);
        let sa = single_a.update_stresses(&[a])[0];
        let sb = single_b.update_stresses(&[b])[0];
        assert!((stress[0] - sa).abs() <= 1e-12 * sa.abs().max(1.0));
        assert!((stress[1] - sb).abs() <= 1e-12 * sb.abs().max(1.0));
    }
    assert_eq!(multi.state.strain.len(), 2);
    // A different component count restarts the state from zero strain.
    let stress = multi.update_stresses(&[1.0e-3, 0.0, 0.0]);
    assert_eq!(stress.len(), 3);
    assert_eq!(multi.state.strain, vec![1.0e-3, 0.0, 0.0]);
    assert_eq!(multi.state.internal_vars.len(), 2);
    assert!(multi.state.internal_vars.iter().all(|iv| iv.len() == 3));
}

#[test]
fn viscoelastic_state_starts_with_zero_strain() {
    let state = ViscoelasticState::new(6, 3);
    assert_eq!(state.stress, vec![0.0; 6]);
    assert_eq!(state.strain, vec![0.0; 6]);
    assert_eq!(state.internal_vars.len(), 3);
}

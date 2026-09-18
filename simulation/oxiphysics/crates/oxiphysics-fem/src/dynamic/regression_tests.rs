//! Regression tests for the time integrators and modal solvers of the `dynamic` module.
//!
//! Every test compares against a closed-form result or a known order of accuracy (exact step
//! response, period elongation, convergence rate, high-frequency annihilation, static limit),
//! so that a wrong integration coefficient cannot pass merely by returning finite numbers.

use super::functions::{modal_newmark_1dof, response_spectrum};
use super::types::{BatheIntegrator, DynamicSolver, GeneralizedAlphaState, WilsonThetaState};
use crate::sparse::CsrMatrix;
use std::f64::consts::PI;

/// Closed-form displacement of an underdamped unit-mass oscillator released from rest under a
/// constant unit force.
fn exact_step_response(omega: f64, zeta: f64, t: f64) -> f64 {
    let omega_d = omega * (1.0 - zeta * zeta).sqrt();
    let envelope = (-zeta * omega * t).exp();
    let oscillation = (omega_d * t).cos() + zeta * omega / omega_d * (omega_d * t).sin();
    (1.0 - envelope * oscillation) / (omega * omega)
}

/// Closed-form displacement of an underdamped unit-mass oscillator with `u(0) = 1`, `v(0) = 0`.
fn exact_free_vibration(omega: f64, zeta: f64, t: f64) -> f64 {
    let omega_d = omega * (1.0 - zeta * zeta).sqrt();
    (-zeta * omega * t).exp() * ((omega_d * t).cos() + zeta * omega / omega_d * (omega_d * t).sin())
}

/// Time of the first downward zero crossing of the velocity after `after`, found by linear
/// interpolation of a `[u, v, a]` trajectory sampled every `dt`.
fn first_velocity_downcrossing(trajectory: &[[f64; 3]], dt: f64, after: f64) -> Option<f64> {
    trajectory.windows(2).enumerate().find_map(|(i, pair)| {
        let t0 = i as f64 * dt;
        (t0 > after && pair[0][1] > 0.0 && pair[1][1] <= 0.0)
            .then(|| t0 + dt * pair[0][1] / (pair[0][1] - pair[1][1]))
    })
}

fn scalar_matrix(value: f64) -> CsrMatrix {
    CsrMatrix::from_triplets(1, 1, &[(0, 0, value)])
}

// ── modal_newmark_1dof / response_spectrum ──────────────────────────────────────────────────

/// Largest deviation of `modal_newmark_1dof` from the closed-form step response on `[0, t_end]`.
fn modal_newmark_step_error(omega: f64, zeta: f64, dt: f64, t_end: f64) -> f64 {
    let n_steps = (t_end / dt).round() as usize;
    let force = vec![1.0; n_steps + 1];
    let q = modal_newmark_1dof(omega, zeta, 1.0, &force, dt);
    q.iter()
        .enumerate()
        .map(|(i, &qi)| (qi - exact_step_response(omega, zeta, i as f64 * dt)).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn modal_newmark_damped_constant_load_settles_at_static_deflection() {
    let (omega, zeta, dt) = (10.0, 0.05, 0.01);
    let force = vec![1.0; 5000];
    let q = modal_newmark_1dof(omega, zeta, 1.0, &force, dt);
    let q_static = 1.0 / (omega * omega);
    assert_eq!(q.len(), 5000);
    assert!(
        q.iter().all(|x| x.is_finite()),
        "damped response must stay finite"
    );
    assert!(
        (q[4999] - q_static).abs() < 1.0e-6,
        "q_end = {}, expected the static deflection {q_static}",
        q[4999]
    );
    let peak = q.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        peak < 2.0 * q_static,
        "a step load can at most double the static deflection, peak = {peak}"
    );
}

#[test]
fn modal_newmark_matches_closed_form_step_response() {
    let (omega, zeta, dt) = (10.0, 0.05, 1.0e-3);
    let err = modal_newmark_step_error(omega, zeta, dt, 2.0);
    assert!(
        err < 2.0e-6,
        "max deviation from the exact response = {err}"
    );

    let force = vec![1.0; 2001];
    let q = modal_newmark_1dof(omega, zeta, 1.0, &force, dt);
    let peak = q.iter().cloned().fold(0.0_f64, f64::max);
    let overshoot = (-PI * zeta / (1.0 - zeta * zeta).sqrt()).exp();
    let expected_peak = (1.0 + overshoot) / (omega * omega);
    assert!(
        (peak - expected_peak).abs() < 2.0e-6,
        "peak = {peak}, expected {expected_peak}"
    );
}

#[test]
fn modal_newmark_error_is_second_order_in_dt() {
    let coarse = modal_newmark_step_error(10.0, 0.05, 0.01, 1.0);
    let fine = modal_newmark_step_error(10.0, 0.05, 0.005, 1.0);
    let ratio = coarse / fine;
    assert!(
        (3.8..4.2).contains(&ratio),
        "halving dt must cut the error by 4, got {coarse} -> {fine} (ratio {ratio})"
    );
}

#[test]
fn modal_newmark_response_depends_only_on_force_per_modal_mass() {
    let force: Vec<f64> = (0..300).map(|i| (i as f64 * 0.05).sin()).collect();
    let scaled_force: Vec<f64> = force.iter().map(|f| 3.0 * f).collect();
    let unit_mass = modal_newmark_1dof(7.0, 0.03, 1.0, &force, 0.01);
    let heavy_mass = modal_newmark_1dof(7.0, 0.03, 3.0, &scaled_force, 0.01);
    for (i, (a, b)) in unit_mass.iter().zip(&heavy_mass).enumerate() {
        assert!((a - b).abs() < 1.0e-12, "q[{i}]: {a} vs {b}");
    }
}

/// A 2000-sample (10 s at 200 Hz) decaying two-tone ground acceleration record.
fn synthetic_ground_motion() -> (Vec<f64>, f64) {
    let dt = 0.005;
    let record = (0..2000)
        .map(|i| {
            let t = i as f64 * dt;
            2.0 * (2.0 * PI * 1.5 * t).sin() * (-0.2 * t).exp()
                + (2.0 * PI * 4.0 * t + 0.7).sin() * (-0.1 * t).exp()
        })
        .collect();
    (record, dt)
}

#[test]
fn response_spectrum_long_record_is_finite_and_rigid_limit_matches_pga() {
    let (record, dt) = synthetic_ground_motion();
    let pga = record.iter().fold(0.0_f64, |peak, a| peak.max(a.abs()));
    let periods = [1.0e-9, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0];
    let spectrum = response_spectrum(&record, dt, &periods, 0.05);
    assert_eq!(spectrum.len(), periods.len());
    for &(period, sa) in &spectrum {
        assert!(
            sa.is_finite() && sa > 0.0,
            "Sa({period}) must be finite and positive, got {sa}"
        );
    }
    assert!((spectrum[0].1 - pga).abs() < 1.0e-12, "rigid branch = PGA");
    for &(period, sa) in &spectrum[1..3] {
        assert!(
            (sa / pga - 1.0).abs() < 0.01,
            "a stiff oscillator (T = {period} s) follows the ground: Sa = {sa}, PGA = {pga}"
        );
    }
    let sa_long_period = spectrum[8].1;
    assert!(
        sa_long_period < 0.5 * pga,
        "a flexible oscillator is isolated from the ground: Sa(3 s) = {sa_long_period}"
    );
}

#[test]
fn response_spectrum_resonance_amplification_matches_closed_form() {
    let dt = 0.005;
    let omega_0 = 2.0 * PI;
    let record: Vec<f64> = (0..2000).map(|i| (omega_0 * i as f64 * dt).sin()).collect();
    let t_end = 1999.0 * dt;
    for zeta in [0.05, 0.1] {
        let spectrum = response_spectrum(&record, dt, &[1.0], zeta);
        // Resonant build-up: the response envelope grows as (1 - exp(-zeta*omega*t)) / (2 zeta).
        let expected = (1.0 - (-zeta * omega_0 * t_end).exp()) / (2.0 * zeta);
        let sa = spectrum[0].1;
        assert!(
            (sa - expected).abs() / expected < 5.0e-3,
            "zeta = {zeta}: Sa = {sa}, expected {expected}"
        );
    }
}

// ── BatheIntegrator ─────────────────────────────────────────────────────────────────────────

/// Integrate a unit-mass oscillator with the Bathe scheme and return the `[u, v, a]` trajectory
/// including the initial state.  `load(t)` is sampled at the mid-point and at the end of every
/// step; the initial acceleration is made consistent with the load at `t = 0`.
fn bathe_sdof_trajectory(
    omega: f64,
    zeta: f64,
    initial: (f64, f64),
    load: &dyn Fn(f64) -> f64,
    dt: f64,
    n_steps: usize,
) -> Vec<[f64; 3]> {
    let (mass, damping, stiffness) = ([1.0], [2.0 * zeta * omega], [omega * omega]);
    let mut bathe = BatheIntegrator::new(1);
    bathe.u[0] = initial.0;
    bathe.v[0] = initial.1;
    bathe.a[0] = load(0.0) - damping[0] * initial.1 - stiffness[0] * initial.0;
    let mut trajectory = vec![[bathe.u[0], bathe.v[0], bathe.a[0]]];
    for step in 0..n_steps {
        let t = step as f64 * dt;
        bathe.step_diagonal_with_loads(
            &mass,
            &stiffness,
            &damping,
            &[load(t + 0.5 * dt)],
            &[load(t + dt)],
            dt,
        );
        trajectory.push([bathe.u[0], bathe.v[0], bathe.a[0]]);
    }
    trajectory
}

/// Distance from the exact state `u = 1, v = 0` after one period of an undamped oscillator
/// integrated with `n_steps` Bathe steps (velocity measured in units of `omega`).
fn bathe_free_vibration_error(n_steps: usize) -> f64 {
    let omega = 2.0 * PI;
    let trajectory = bathe_sdof_trajectory(
        omega,
        0.0,
        (1.0, 0.0),
        &|_| 0.0,
        1.0 / n_steps as f64,
        n_steps,
    );
    let end = trajectory[n_steps];
    (end[0] - 1.0).hypot(end[1] / omega)
}

#[test]
fn bathe_annihilates_high_frequency_content() {
    // One step with omega*dt = 100 and 1000 from unit amplitude: the second sub-step is the
    // three-point backward Euler scheme, so the spectral radius tends to zero.  The former
    // trapezoidal-in-both-sub-steps code left the amplitude at 1.003.
    let dt = 1.0;
    for (omega, bound) in [(100.0, 0.05), (1000.0, 0.006)] {
        let trajectory = bathe_sdof_trajectory(omega, 0.0, (1.0, 0.0), &|_| 0.0, dt, 3);
        let amplitude = |state: &[f64; 3]| state[0].hypot(state[1] / omega);
        let first = amplitude(&trajectory[1]);
        assert!(
            first < bound,
            "omega*dt = {omega}: amplitude after one step = {first}, bound {bound}"
        );
        assert!(
            amplitude(&trajectory[3]) < first,
            "omega*dt = {omega}: the amplitude must keep decreasing"
        );
    }
}

#[test]
fn bathe_free_vibration_error_is_second_order() {
    let errors = [40, 80, 160].map(bathe_free_vibration_error);
    for pair in errors.windows(2) {
        let ratio = pair[0] / pair[1];
        assert!(
            (3.8..4.2).contains(&ratio),
            "halving dt must cut the error by 4, got {errors:?}"
        );
    }
    assert!(errors[2] < 1.0e-3, "error with 160 steps = {}", errors[2]);
}

#[test]
fn bathe_damped_step_response_matches_closed_form() {
    let (omega, zeta, n_steps) = (10.0, 0.05, 200);
    let dt = 1.0 / n_steps as f64;
    let trajectory = bathe_sdof_trajectory(omega, zeta, (0.0, 0.0), &|_| 1.0, dt, n_steps);
    let worst = trajectory
        .iter()
        .enumerate()
        .map(|(i, state)| (state[0] - exact_step_response(omega, zeta, i as f64 * dt)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        worst < 1.0e-5,
        "max deviation from the exact response = {worst}"
    );
}

#[test]
fn bathe_free_mass_translates_and_accelerates_exactly() {
    // m = 2, no stiffness or damping, v0 = 1: pure translation u = t.
    let mut coasting = BatheIntegrator::new(1);
    coasting.v[0] = 1.0;
    for _ in 0..10 {
        coasting.step_diagonal(&[2.0], &[0.0], &[0.0], &[0.0], 0.1);
    }
    assert!(
        (coasting.u[0] - 1.0).abs() < 1.0e-12,
        "u = {}",
        coasting.u[0]
    );
    assert!(
        (coasting.v[0] - 1.0).abs() < 1.0e-12,
        "v = {}",
        coasting.v[0]
    );
    assert!(coasting.a[0].abs() < 1.0e-12, "a = {}", coasting.a[0]);

    // Constant force 4 on m = 2: a = 2, v = 2 t, u = t^2.
    let mut pushed = BatheIntegrator::new(1);
    pushed.a[0] = 2.0;
    for _ in 0..10 {
        pushed.step_diagonal(&[2.0], &[0.0], &[0.0], &[4.0], 0.1);
    }
    assert!((pushed.u[0] - 1.0).abs() < 1.0e-12, "u = {}", pushed.u[0]);
    assert!((pushed.v[0] - 2.0).abs() < 1.0e-12, "v = {}", pushed.v[0]);
    assert!((pushed.a[0] - 2.0).abs() < 1.0e-12, "a = {}", pushed.a[0]);
}

#[test]
fn bathe_mid_step_load_integrates_a_linear_ramp() {
    // Unit free mass under f(t) = t: a = t, v = t^2 / 2, u = t^3 / 6.
    let (n_steps, dt) = (20, 0.05);
    let ramp = |t: f64| t;
    let sampled = bathe_sdof_trajectory(0.0, 0.0, (0.0, 0.0), &ramp, dt, n_steps);
    let end = sampled[n_steps];
    assert!((end[1] - 0.5).abs() < 1.0e-9, "v = {}", end[1]);
    assert!((end[2] - 1.0).abs() < 1.0e-9, "a = {}", end[2]);
    assert!((end[0] - 1.0 / 6.0).abs() < 2.0e-4, "u = {}", end[0]);

    // Holding the end-of-step load over the whole step is only first-order accurate.
    let mut held = BatheIntegrator::new(1);
    for step in 0..n_steps {
        let f_end = ramp((step + 1) as f64 * dt);
        held.step_diagonal(&[1.0], &[0.0], &[0.0], &[f_end], dt);
    }
    assert!(
        (held.v[0] - 0.5).abs() > 5.0e-3,
        "the held-load velocity error should expose the missing mid-step load, v = {}",
        held.v[0]
    );
}

#[test]
fn bathe_step_diagonal_holds_the_load_constant() {
    let (mass, stiffness, damping) = ([1.0, 2.0], [50.0, 80.0], [0.3, 0.1]);
    let force = [1.5, -0.5];
    let mut held = BatheIntegrator::new(2);
    let mut explicit = BatheIntegrator::new(2);
    for _ in 0..25 {
        held.step_diagonal(&mass, &stiffness, &damping, &force, 0.01);
        explicit.step_diagonal_with_loads(&mass, &stiffness, &damping, &force, &force, 0.01);
    }
    assert_eq!(held.u, explicit.u);
    assert_eq!(held.v, explicit.v);
    assert_eq!(held.a, explicit.a);
}

#[test]
fn bathe_dofs_are_independent_of_each_other() {
    let omegas = [3.0, 11.0];
    let damping = [0.2, 0.7];
    let stiffness = [omegas[0] * omegas[0], omegas[1] * omegas[1]];
    let initial_u = [1.0, -0.5];
    let initial_a = [-stiffness[0] * initial_u[0], -stiffness[1] * initial_u[1]];
    let mut coupled = BatheIntegrator::new(2);
    coupled.u = initial_u.to_vec();
    coupled.a = initial_a.to_vec();
    for _ in 0..40 {
        coupled.step_diagonal(&[1.0, 1.0], &stiffness, &damping, &[0.0, 0.0], 0.02);
    }
    for dof in 0..2 {
        let mut alone = BatheIntegrator::new(1);
        alone.u[0] = initial_u[dof];
        alone.a[0] = initial_a[dof];
        for _ in 0..40 {
            alone.step_diagonal(&[1.0], &[stiffness[dof]], &[damping[dof]], &[0.0], 0.02);
        }
        assert_eq!(alone.u[0], coupled.u[dof], "u of DOF {dof}");
        assert_eq!(alone.v[0], coupled.v[dof], "v of DOF {dof}");
        assert_eq!(alone.a[0], coupled.a[dof], "a of DOF {dof}");
    }
}

#[test]
fn bathe_dof_without_inertia_damping_and_stiffness_keeps_its_state() {
    let mut bathe = BatheIntegrator::new(2);
    bathe.u = vec![0.25, 1.0];
    bathe.v = vec![-0.5, 0.0];
    bathe.a = vec![0.125, -4.0];
    bathe.step_diagonal(&[0.0, 1.0], &[0.0, 4.0], &[0.0, 0.0], &[3.0, 0.0], 0.1);
    assert_eq!(
        (bathe.u[0], bathe.v[0], bathe.a[0]),
        (0.25, -0.5, 0.125),
        "an equation-free DOF must not be advanced"
    );
    assert!(bathe.u[1] != 1.0, "the regular DOF must be advanced");
}

// ── WilsonThetaState ────────────────────────────────────────────────────────────────────────

/// Integrate a unit-mass oscillator released at `u0` from rest with the Wilson-θ method (θ = 1.4)
/// and return the `[u, v, a]` trajectory including the initial state.
fn wilson_sdof_trajectory(
    omega: f64,
    zeta: f64,
    u0: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<[f64; 3]> {
    let (mass, damping, stiffness) = ([1.0], [2.0 * zeta * omega], [omega * omega]);
    let mut wilson = WilsonThetaState::new(1);
    wilson.u[0] = u0;
    wilson.a[0] = -omega * omega * u0;
    let mut trajectory = vec![[wilson.u[0], wilson.v[0], wilson.a[0]]];
    for _ in 0..n_steps {
        wilson.step_diagonal(&mass, &stiffness, &damping, &[0.0], &[0.0], dt);
        trajectory.push([wilson.u[0], wilson.v[0], wilson.a[0]]);
    }
    trajectory
}

/// Relative error of the numerical period of an undamped oscillator (exact period 1 s) with
/// `steps_per_period` Wilson-θ steps per period.
fn wilson_period_error(steps_per_period: usize) -> f64 {
    let omega = 2.0 * PI;
    let dt = 1.0 / steps_per_period as f64;
    let trajectory = wilson_sdof_trajectory(omega, 0.0, 1.0, dt, 2 * steps_per_period);
    let period = first_velocity_downcrossing(&trajectory, dt, 0.5);
    period.map_or(f64::INFINITY, |p| (p - 1.0).abs())
}

#[test]
fn wilson_theta_undamped_oscillator_returns_to_initial_state_after_one_period() {
    // m = 1, k = (2 pi)^2, u0 = 1, theta = 1.4, dt = T / 100.  The former 3*a1 velocity
    // coefficient left u(T) at 0.87.
    let omega = 2.0 * PI;
    let n_steps = 100;
    let trajectory = wilson_sdof_trajectory(omega, 0.0, 1.0, 1.0 / n_steps as f64, n_steps);
    let [u, v, a] = trajectory[n_steps];
    assert!((u - 1.0).abs() < 1.0e-3, "u(T) = {u}");
    assert!((v / omega).abs() < 1.0e-2, "v(T) / omega = {}", v / omega);
    assert!(
        (a + omega * omega).abs() / (omega * omega) < 5.0e-3,
        "a(T) = {a}, expected {}",
        -omega * omega
    );
}

#[test]
fn wilson_theta_period_error_is_small_and_second_order() {
    let errors = [50, 100, 200].map(wilson_period_error);
    assert!(
        errors[1] < 1.0e-2,
        "period error with 100 steps = {}",
        errors[1]
    );
    for pair in errors.windows(2) {
        let ratio = pair[0] / pair[1];
        assert!(
            (3.5..4.5).contains(&ratio),
            "halving dt must cut the period error by 4, got {errors:?}"
        );
    }
}

/// Largest deviation from the closed-form damped free vibration over three periods.
fn wilson_damped_error(steps_per_period: usize) -> f64 {
    let (omega, zeta) = (2.0 * PI, 0.05);
    let dt = 1.0 / steps_per_period as f64;
    let trajectory = wilson_sdof_trajectory(omega, zeta, 1.0, dt, 3 * steps_per_period);
    trajectory
        .iter()
        .enumerate()
        .map(|(i, state)| (state[0] - exact_free_vibration(omega, zeta, i as f64 * dt)).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn wilson_theta_damped_free_vibration_matches_closed_form() {
    let coarse = wilson_damped_error(100);
    let fine = wilson_damped_error(200);
    assert!(
        fine < 2.0e-3,
        "max deviation with 200 steps/period = {fine}"
    );
    let ratio = coarse / fine;
    assert!(
        (3.5..4.5).contains(&ratio),
        "halving dt must cut the error by 4, got {coarse} -> {fine}"
    );
}

#[test]
fn wilson_theta_free_mass_translates_and_accelerates_exactly() {
    let mut coasting = WilsonThetaState::new(1);
    coasting.v[0] = 1.0;
    for _ in 0..10 {
        coasting.step_diagonal(&[2.0], &[0.0], &[0.0], &[0.0], &[0.0], 0.1);
    }
    assert!(
        (coasting.u[0] - 1.0).abs() < 1.0e-12,
        "u = {}",
        coasting.u[0]
    );
    assert!(
        (coasting.v[0] - 1.0).abs() < 1.0e-12,
        "v = {}",
        coasting.v[0]
    );

    let mut pushed = WilsonThetaState::new(1);
    pushed.a[0] = 2.0;
    for _ in 0..10 {
        pushed.step_diagonal(&[2.0], &[0.0], &[0.0], &[4.0], &[4.0], 0.1);
    }
    assert!((pushed.u[0] - 1.0).abs() < 1.0e-12, "u = {}", pushed.u[0]);
    assert!((pushed.v[0] - 2.0).abs() < 1.0e-12, "v = {}", pushed.v[0]);
    assert!((pushed.a[0] - 2.0).abs() < 1.0e-12, "a = {}", pushed.a[0]);
}

// ── DynamicSolver::generalized_alpha ────────────────────────────────────────────────────────

/// Trajectory `[u, v, a]` of the `m = k = 1` oscillator (omega = 1, T = 2 pi) released from
/// `u0 = 1` and advanced with `DynamicSolver::generalized_alpha`.
fn generalized_alpha_sdof_trajectory(
    rho_inf: f64,
    steps_per_period: usize,
    n_periods: usize,
) -> Vec<[f64; 3]> {
    let (mass, stiffness) = (scalar_matrix(1.0), scalar_matrix(1.0));
    let dt = 2.0 * PI / steps_per_period as f64;
    let mut solver = DynamicSolver::new(1);
    solver.u[0] = 1.0;
    solver.a[0] = -1.0;
    let mut trajectory = vec![[1.0, 0.0, -1.0]];
    for _ in 0..steps_per_period * n_periods {
        solver.generalized_alpha(&mass, &stiffness, &[0.0], dt, rho_inf);
        trajectory.push([solver.u[0], solver.v[0], solver.a[0]]);
    }
    trajectory
}

#[test]
fn generalized_alpha_free_mass_translates_exactly() {
    // The former diagonal-only solve gave u = 4 instead of 1 for m = 1, v0 = 1, dt = 1.
    let free_mass = scalar_matrix(3.0);
    let no_stiffness = CsrMatrix::new(1, 1);
    for rho_inf in [1.0, 0.5, 0.0] {
        let mut solver = DynamicSolver::new(1);
        solver.v[0] = 1.0;
        solver.generalized_alpha(&free_mass, &no_stiffness, &[0.0], 1.0, rho_inf);
        assert!(
            (solver.u[0] - 1.0).abs() < 1.0e-12,
            "rho_inf = {rho_inf}: u after one step = {}",
            solver.u[0]
        );
        for _ in 1..10 {
            solver.generalized_alpha(&free_mass, &no_stiffness, &[0.0], 1.0, rho_inf);
        }
        assert!(
            (solver.u[0] - 10.0).abs() < 1.0e-9,
            "rho_inf = {rho_inf}: u after ten steps = {}",
            solver.u[0]
        );
        assert!((solver.v[0] - 1.0).abs() < 1.0e-9, "v = {}", solver.v[0]);
        assert!(solver.a[0].abs() < 1.0e-9, "a = {}", solver.a[0]);
    }
}

#[test]
fn generalized_alpha_free_mass_under_constant_force_is_exact() {
    // F = 6 on m = 3: a = 2, v = 2 t, u = t^2.
    let free_mass = scalar_matrix(3.0);
    let no_stiffness = CsrMatrix::new(1, 1);
    for rho_inf in [1.0, 0.5, 0.0] {
        let mut solver = DynamicSolver::new(1);
        solver.a[0] = 2.0;
        for _ in 0..10 {
            solver.generalized_alpha(&free_mass, &no_stiffness, &[6.0], 0.1, rho_inf);
        }
        assert!(
            (solver.u[0] - 1.0).abs() < 1.0e-9,
            "rho {rho_inf}: u = {}",
            solver.u[0]
        );
        assert!(
            (solver.v[0] - 2.0).abs() < 1.0e-9,
            "rho {rho_inf}: v = {}",
            solver.v[0]
        );
        assert!(
            (solver.a[0] - 2.0).abs() < 1.0e-9,
            "rho {rho_inf}: a = {}",
            solver.a[0]
        );
    }
}

#[test]
fn generalized_alpha_sdof_period_error_is_small() {
    let steps_per_period = 40;
    let period = 2.0 * PI;
    let dt = period / steps_per_period as f64;
    for (rho_inf, bound) in [(1.0, 0.005), (0.9, 0.005), (0.5, 0.005), (0.0, 0.015)] {
        let trajectory = generalized_alpha_sdof_trajectory(rho_inf, steps_per_period, 3);
        let measured = first_velocity_downcrossing(&trajectory, dt, 0.5 * period);
        let measured = measured.unwrap_or(f64::NAN);
        let relative_error = (measured / period - 1.0).abs();
        assert!(
            relative_error < bound,
            "rho_inf = {rho_inf}: period {measured} vs {period} (relative error {relative_error})"
        );
    }
}

#[test]
fn generalized_alpha_conserves_energy_at_rho_one_and_dissipates_below() {
    let steps_per_period = 40;
    let energy_after_five_periods = |rho_inf: f64| {
        let trajectory = generalized_alpha_sdof_trajectory(rho_inf, steps_per_period, 5);
        let end = trajectory[5 * steps_per_period];
        0.5 * (end[0] * end[0] + end[1] * end[1])
    };
    let energies = [1.0, 0.9, 0.5, 0.0].map(energy_after_five_periods);
    assert!(
        (energies[0] - 0.5).abs() < 1.0e-8,
        "rho_inf = 1 must conserve the energy, got {}",
        energies[0]
    );
    for pair in energies.windows(2) {
        assert!(
            pair[1] < pair[0],
            "a smaller rho_inf must dissipate more, got {energies:?}"
        );
    }
    assert!(
        energies[3] < 0.475,
        "rho_inf = 0 loses about 10 % of the energy in five periods, got {}",
        energies[3]
    );
}

#[test]
fn generalized_alpha_static_limit_with_full_dissipation() {
    // k = m = 1, F = 1 from rest, dt = T: with rho_inf = 0 the transient is annihilated and the
    // solution settles at the static deflection F / k = 1.  The former solve (without the
    // alpha_f*K*u_n terms and with double-counted predictor terms) could not reach it.
    let (mass, stiffness) = (scalar_matrix(1.0), scalar_matrix(1.0));
    let mut solver = DynamicSolver::new(1);
    solver.a[0] = 1.0;
    for _ in 0..40 {
        solver.generalized_alpha(&mass, &stiffness, &[1.0], 2.0 * PI, 0.0);
    }
    assert!((solver.u[0] - 1.0).abs() < 1.0e-8, "u = {}", solver.u[0]);
    assert!(solver.v[0].abs() < 1.0e-8, "v = {}", solver.v[0]);
    assert!(solver.a[0].abs() < 1.0e-8, "a = {}", solver.a[0]);
}

/// `M = I`, `K = [[2, -1], [-1, 2]]` (modes `(1, 1)` with omega = 1 and `(1, -1)` with
/// omega = sqrt 3).
fn two_dof_system() -> (CsrMatrix, CsrMatrix) {
    let mass = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
    let stiffness = CsrMatrix::from_triplets(
        2,
        2,
        &[(0, 0, 2.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0)],
    );
    (mass, stiffness)
}

/// Largest nodal deviation from the modal solution at `t = 2 pi` after `n_steps` steps of the
/// free vibration released from `u0 = (1, 0)`.
fn generalized_alpha_two_dof_error(rho_inf: f64, n_steps: usize) -> f64 {
    let (mass, stiffness) = two_dof_system();
    let t_end = 2.0 * PI;
    let mut solver = DynamicSolver::new(2);
    solver.u = vec![1.0, 0.0];
    solver.a = vec![-2.0, 1.0];
    for _ in 0..n_steps {
        solver.generalized_alpha(
            &mass,
            &stiffness,
            &[0.0, 0.0],
            t_end / n_steps as f64,
            rho_inf,
        );
    }
    let slow = 0.5 * t_end.cos();
    let fast = 0.5 * (3.0_f64.sqrt() * t_end).cos();
    (solver.u[0] - (slow + fast))
        .abs()
        .max((solver.u[1] - (slow - fast)).abs())
}

#[test]
fn generalized_alpha_two_dof_free_vibration_converges_to_the_modal_solution() {
    for rho_inf in [1.0, 0.9] {
        let coarse = generalized_alpha_two_dof_error(rho_inf, 100);
        let fine = generalized_alpha_two_dof_error(rho_inf, 200);
        assert!(
            fine < 2.0e-3,
            "rho_inf = {rho_inf}: error with 200 steps = {fine}"
        );
        let ratio = coarse / fine;
        assert!(
            (3.5..4.5).contains(&ratio),
            "rho_inf = {rho_inf}: halving dt must cut the error by 4, got {coarse} -> {fine}"
        );
    }
}

/// One Chung-Hulbert step of the two-DOF test system (`M = I`, `K` as in [`two_dof_system`]),
/// written directly from the published equations and solved with Cramer's rule, independently
/// of `GeneralizedAlphaState`.
fn reference_two_dof_step(state: &mut [[f64; 2]; 3], force: [f64; 2], dt: f64, rho_inf: f64) {
    let alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0);
    let alpha_f = rho_inf / (rho_inf + 1.0);
    let gamma = 0.5 - alpha_m + alpha_f;
    let beta = 0.25 * (1.0 - alpha_m + alpha_f).powi(2);
    let [u, v, a] = *state;
    let k_u = [2.0 * u[0] - u[1], -u[0] + 2.0 * u[1]];
    let inertia_scale = (1.0 - alpha_m) / (beta * dt * dt);
    let rhs: Vec<f64> = (0..2)
        .map(|i| {
            force[i]
                + (1.0 - alpha_m)
                    * ((u[i] + dt * v[i]) / (beta * dt * dt) + (0.5 / beta - 1.0) * a[i])
                - alpha_m * a[i]
                - alpha_f * k_u[i]
        })
        .collect();
    let (k11, k12) = ((1.0 - alpha_f) * 2.0 + inertia_scale, -(1.0 - alpha_f));
    let det = k11 * k11 - k12 * k12;
    let u_new = [
        (rhs[0] * k11 - k12 * rhs[1]) / det,
        (k11 * rhs[1] - k12 * rhs[0]) / det,
    ];
    let mut a_new = [0.0; 2];
    let mut v_new = [0.0; 2];
    for i in 0..2 {
        a_new[i] = (u_new[i] - u[i] - dt * v[i]) / (beta * dt * dt) - (0.5 / beta - 1.0) * a[i];
        v_new[i] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a_new[i]);
    }
    *state = [u_new, v_new, a_new];
}

#[test]
fn generalized_alpha_two_dof_forced_run_matches_independent_reference() {
    let (mass, stiffness) = two_dof_system();
    let force = [0.5, -0.25];
    let (dt, rho_inf) = (0.15, 0.7);
    let mut solver = DynamicSolver::new(2);
    solver.a = force.to_vec();
    let mut reference = [[0.0; 2], [0.0; 2], force];
    let deviation = |actual: &[f64], expected: &[f64; 2]| {
        actual
            .iter()
            .zip(expected)
            .map(|(got, want)| (got - want).abs())
            .fold(0.0_f64, f64::max)
    };
    for step in 0..60 {
        solver.generalized_alpha(&mass, &stiffness, &force, dt, rho_inf);
        reference_two_dof_step(&mut reference, force, dt, rho_inf);
        assert!(
            deviation(&solver.u, &reference[0]) < 1.0e-8,
            "step {step}: u = {:?} vs reference {:?}",
            solver.u,
            reference[0]
        );
        assert!(
            deviation(&solver.v, &reference[1]) < 1.0e-8,
            "step {step}: v = {:?} vs reference {:?}",
            solver.v,
            reference[1]
        );
        assert!(
            deviation(&solver.a, &reference[2]) < 1.0e-7,
            "step {step}: a = {:?} vs reference {:?}",
            solver.a,
            reference[2]
        );
    }
}

#[test]
fn generalized_alpha_solver_delegates_to_generalized_alpha_state() {
    let (mass, stiffness) = two_dof_system();
    let force = [0.3, 0.1];
    let mut solver = DynamicSolver::new(2);
    let mut state = GeneralizedAlphaState::new(2, 0.6);
    solver.u = vec![0.2, -0.1];
    solver.v = vec![0.0, 0.4];
    solver.a = vec![-0.9, 0.7];
    state.u = solver.u.clone();
    state.v = solver.v.clone();
    state.a = solver.a.clone();
    for _ in 0..20 {
        solver.generalized_alpha(&mass, &stiffness, &force, 0.05, 0.6);
        state.step(&mass, &stiffness, &force, 0.05, &[]);
    }
    assert_eq!(solver.u, state.u);
    assert_eq!(solver.v, state.v);
    assert_eq!(solver.a, state.a);
}

#[test]
fn generalized_alpha_solver_clamps_rho_inf_to_the_unit_interval() {
    let (mass, stiffness) = two_dof_system();
    let run = |rho_inf: f64| {
        let mut solver = DynamicSolver::new(2);
        solver.u = vec![1.0, 0.0];
        solver.a = vec![-2.0, 1.0];
        for _ in 0..10 {
            solver.generalized_alpha(&mass, &stiffness, &[0.0, 0.0], 0.1, rho_inf);
        }
        solver.u
    };
    assert_eq!(run(1.0), run(2.5));
    assert_eq!(run(0.0), run(-1.0));
}

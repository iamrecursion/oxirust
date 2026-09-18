// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the Lagrange-top integrator.
//!
//! The tilt equation of motion used to carry the wrong sign on both the
//! gravity and the gyroscopic term, so a non-spinning top "righted itself"
//! and the fast-top precession was reversed.  The tests below check the
//! equation against closed-form limits (inverted pendulum, steady
//! precession) and against the conserved energy and momenta.

use super::*;

/// Heavy top used throughout: `I₃ = 0.1`, `I₁ = 0.05`, `m = 0.5 kg`,
/// `l = 0.1 m`, `g = 9.81 m/s²`.
fn top(theta0: f64, spin_rate: f64) -> LagrangeTopState {
    LagrangeTopState::new(0.1, 0.05, 0.5, 0.1, 9.81, theta0, spin_rate)
}

fn mgl(top: &LagrangeTopState) -> f64 {
    top.mass * top.gravity * top.cm_distance
}

#[test]
fn non_spinning_top_falls_away_from_the_vertical() {
    let mut t = top(0.3, 0.0);
    let mut previous = t.theta;
    for step in 0..100 {
        integrate_lagrange_top(&mut t, 1.0e-3);
        assert!(
            t.theta > previous,
            "step {step}: the tilt must keep growing, {previous} -> {}",
            t.theta
        );
        previous = t.theta;
    }
    assert!(t.theta_dot > 0.0, "theta_dot = {}", t.theta_dot);
    assert!(t.theta > 0.3 + 1e-3, "theta = {}", t.theta);
    // Without spin there is no precession.
    assert!(t.phi.abs() < 1e-12 && t.phi_dot.abs() < 1e-12);
}

#[test]
fn non_spinning_top_is_a_conservative_inverted_pendulum() {
    // p_phi = p_psi = 0: I1·theta'' = m g l sin(theta), E = I1 theta'^2 / 2 + m g l cos(theta).
    let mut t = top(0.3, 0.0);
    let e0 = t.total_energy();
    for _ in 0..1000 {
        integrate_lagrange_top(&mut t, 1.0e-3);
        let e = t.total_energy();
        assert!((e - e0).abs() < 1e-9 * e0.abs(), "E = {e}, E0 = {e0}");
    }
    assert!(
        t.theta > 1.0,
        "the top must have fallen far from the vertical"
    );
}

#[test]
fn small_tilt_grows_exponentially_like_the_linearised_inverted_pendulum() {
    // theta(t) = theta0 · cosh(lambda t) with lambda² = m g l / I1.
    let theta0 = 0.01;
    let mut t = top(theta0, 0.0);
    let lambda = (mgl(&t) / t.i_transverse).sqrt();
    let t_end = 5.0_f64.acosh() / lambda;
    let steps = 2000;
    let dt = t_end / steps as f64;
    for _ in 0..steps {
        integrate_lagrange_top(&mut t, dt);
    }
    // The nonlinear correction at theta = 0.05 is below 0.1 %.
    assert!(
        (t.theta - 5.0 * theta0).abs() < 0.01 * 5.0 * theta0,
        "theta = {}, linear prediction {}",
        t.theta,
        5.0 * theta0
    );
}

#[test]
fn fast_top_conserves_the_total_energy() {
    let mut t = top(0.3, 100.0);
    let e0 = t.total_energy();
    let mut worst: f64 = 0.0;
    for _ in 0..1000 {
        integrate_lagrange_top(&mut t, 1.0e-4);
        worst = worst.max(((t.total_energy() - e0) / e0).abs());
    }
    assert!(worst < 1e-6, "relative energy drift {worst}");
}

#[test]
fn slowly_spinning_top_conserves_energy_and_both_momenta() {
    // Spin energy comparable to the gravitational energy: a sensitive check
    // of the relative sign and size of the gravity and gyroscopic terms.
    let mut t = top(0.3, 8.0);
    let e0 = t.total_energy();
    let p_psi0 = t.spin_angular_momentum();
    let p_phi0 = t.vertical_angular_momentum();
    let mut theta_min = t.theta;
    let mut theta_max = t.theta;
    for _ in 0..3000 {
        integrate_lagrange_top(&mut t, 1.0e-4);
        assert!((t.total_energy() - e0).abs() < 1e-9 * e0.abs().max(1.0));
        assert!((t.spin_angular_momentum() - p_psi0).abs() < 1e-9 * p_psi0.abs());
        assert!((t.vertical_angular_momentum() - p_phi0).abs() < 1e-9 * p_phi0.abs());
        theta_min = theta_min.min(t.theta);
        theta_max = theta_max.max(t.theta);
    }
    // The released top starts at rest in theta, so theta never drops below
    // the release angle (the gravity torque tips it away from the vertical).
    assert!(theta_min >= 0.3 - 1e-9, "theta_min = {theta_min}");
    assert!(theta_max > 0.3 + 1e-3, "theta_max = {theta_max}");
}

#[test]
fn steady_precession_keeps_the_tilt_and_advances_at_the_slow_precession_rate() {
    let theta0: f64 = 0.3;
    let spin_rate = 100.0;
    let mut t = top(theta0, spin_rate);
    // Steady precession (theta'' = 0) requires
    // (I1 − I3)·cosθ·φ̇² − I3·ψ̇·φ̇ + m g l = 0; take the slow root.
    let a = (t.i_transverse - t.i_spin) * theta0.cos();
    let b = -t.i_spin * spin_rate;
    let c = mgl(&t);
    let disc = (b * b - 4.0 * a * c).sqrt();
    let roots = [(-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a)];
    let slow = if roots[0].abs() < roots[1].abs() {
        roots[0]
    } else {
        roots[1]
    };
    // The gyroscopic approximation m g l / (I3·ψ̇) of a fast top.
    let gyroscopic = c / (t.i_spin * spin_rate);
    assert!(
        (slow - gyroscopic).abs() < 1e-3 * gyroscopic,
        "slow root {slow}, gyroscopic estimate {gyroscopic}"
    );
    assert!(slow > 0.0, "a positively spinning top precesses positively");

    t.phi_dot = slow;
    let dt = 1.0e-3;
    let steps = 1000;
    for _ in 0..steps {
        integrate_lagrange_top(&mut t, dt);
    }
    assert!((t.theta - theta0).abs() < 1e-9, "theta = {}", t.theta);
    assert!(t.theta_dot.abs() < 1e-9, "theta_dot = {}", t.theta_dot);
    let elapsed = dt * steps as f64;
    assert!(
        (t.phi - slow * elapsed).abs() < 1e-6 * slow * elapsed,
        "phi = {}, expected {}",
        t.phi,
        slow * elapsed
    );
}

#[test]
fn fast_top_released_at_rest_precesses_in_the_direction_of_the_gravity_torque() {
    // Starting with φ̇ = θ̇ = 0 the top nutates: the tilt oscillates above its
    // release value while the precession angle advances in the positive sense.
    let mut t = top(0.3, 100.0);
    for _ in 0..2000 {
        integrate_lagrange_top(&mut t, 1.0e-4);
        assert!(t.theta >= 0.3 - 1e-9, "theta = {}", t.theta);
        assert!(t.phi_dot >= -1e-9, "phi_dot = {}", t.phi_dot);
    }
    assert!(t.phi > 0.0);
}

#[test]
fn integration_is_reversible_in_time() {
    let mut t = top(0.4, 12.0);
    let start = t.clone();
    for _ in 0..500 {
        integrate_lagrange_top(&mut t, 2.0e-4);
    }
    assert!(
        (t.theta - start.theta).abs() > 1e-3,
        "the top must have moved"
    );
    for _ in 0..500 {
        integrate_lagrange_top(&mut t, -2.0e-4);
    }
    assert!((t.theta - start.theta).abs() < 1e-9, "theta = {}", t.theta);
    assert!((t.theta_dot - start.theta_dot).abs() < 1e-8);
    assert!((t.phi - start.phi).abs() < 1e-9, "phi = {}", t.phi);
    assert!((t.phi_dot - start.phi_dot).abs() < 1e-8);
}

#[test]
fn zero_step_keeps_a_consistent_state_unchanged() {
    let mut t = top(0.7, 25.0);
    t.phi_dot = 0.4;
    t.theta_dot = -0.2;
    let before = t.clone();
    integrate_lagrange_top(&mut t, 0.0);
    assert!((t.theta - before.theta).abs() < 1e-15);
    assert!((t.theta_dot - before.theta_dot).abs() < 1e-15);
    assert!((t.phi_dot - before.phi_dot).abs() < 1e-12);
    assert!((t.psi_dot - before.psi_dot).abs() < 1e-12);
}

#[test]
fn upright_top_is_clamped_away_from_the_euler_angle_singularity() {
    let mut t = top(0.0, 50.0);
    integrate_lagrange_top(&mut t, 1.0e-3);
    assert!(t.theta >= LAGRANGE_TOP_THETA_MARGIN);
    assert!(t.theta.is_finite() && t.phi_dot.is_finite() && t.psi_dot.is_finite());
    // The physical spin about the vertical axis is preserved by the re-gauging.
    let total_spin = t.phi_dot + t.psi_dot;
    assert!(
        (total_spin - 50.0).abs() < 1e-6,
        "phi_dot + psi_dot = {total_spin}"
    );
}

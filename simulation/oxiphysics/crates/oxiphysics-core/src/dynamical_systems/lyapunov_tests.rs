// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the tangent (variational) flow and the Lyapunov
//! routines of `dynamical_systems`: the tangent vectors are advanced by RK4
//! with the Jacobian of every stage, `renorm == 0` and empty spectra do not
//! panic.

use super::*;

/// `ẋ = A x` for a constant 2 × 2 matrix `A = [[a, b], [c, d]]`.
struct Linear2 {
    a: [f64; 4],
}

impl OdeSystem for Linear2 {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&self, _t: f64, x: &[f64]) -> Vec<f64> {
        vec![
            self.a[0] * x[0] + self.a[1] * x[1],
            self.a[2] * x[0] + self.a[3] * x[1],
        ]
    }
}

/// `ẋ = -t x`: the Jacobian `J = -t` depends on time only.
struct TimeDecay;

impl OdeSystem for TimeDecay {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&self, t: f64, x: &[f64]) -> Vec<f64> {
        vec![-t * x[0]]
    }
}

fn harmonic_oscillator() -> Linear2 {
    Linear2 {
        a: [0.0, 1.0, -1.0, 0.0],
    }
}

/// Integrate `n` RK4 steps of size `h` to get onto the attractor.
fn settle(sys: &dyn OdeSystem, x0: &[f64], h: f64, n: usize) -> Vec<f64> {
    let mut x = x0.to_vec();
    for i in 0..n {
        x = rk4_step(sys, i as f64 * h, &x, h);
    }
    x
}

fn identity_columns(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

// ── rk4_step_with_tangents ───────────────────────────────────────────────────

#[test]
fn test_tangent_step_is_the_derivative_of_the_rk4_flow_map() {
    // Applying RK4 to the augmented system (x, Φ) with the Jacobian of every
    // stage is exactly the derivative of the RK4 flow map, for any step size.
    let lorenz = LorenzSystem::classic();
    let x0 = [3.0, -2.0, 21.0];
    for h in [0.01, 0.1] {
        let (next, phi) = rk4_step_with_tangents(&lorenz, 0.0, &x0, &identity_columns(3), h, 1e-6);
        assert_eq!(
            next,
            rk4_step(&lorenz, 0.0, &x0, h),
            "base trajectory, h = {h}"
        );
        let eps = 1e-6;
        for (j, column) in phi.iter().enumerate() {
            let mut plus = x0;
            let mut minus = x0;
            plus[j] += eps;
            minus[j] -= eps;
            let flow_plus = rk4_step(&lorenz, 0.0, &plus, h);
            let flow_minus = rk4_step(&lorenz, 0.0, &minus, h);
            for (i, entry) in column.iter().enumerate() {
                let derivative = (flow_plus[i] - flow_minus[i]) / (2.0 * eps);
                assert!(
                    (entry - derivative).abs() < 1e-6,
                    "h = {h}: dΦ[{i}][{j}] = {entry}, d(flow)/dx = {derivative}"
                );
            }
        }
    }
}

#[test]
fn test_tangent_step_evaluates_the_jacobian_at_the_stage_times() {
    // q' = -t q, so q(t0 + h) = q0 exp(-((t0 + h)² - t0²) / 2).  Evaluating
    // J at t0 (or at the end of the step) instead of at the stage times
    // t0, t0 + h/2, t0 + h/2, t0 + h is wrong at order h²/2.
    let t0 = 1.0;
    let h = 0.1;
    let (_, tangents) = rk4_step_with_tangents(&TimeDecay, t0, &[2.0], &[vec![1.0]], h, 1e-6);
    let exact = (-((t0 + h) * (t0 + h) - t0 * t0) / 2.0).exp();
    assert!(
        (tangents[0][0] - exact).abs() < 3e-6,
        "q = {}, exact {exact}",
        tangents[0][0]
    );
    // Freezing J at the start of the step would be off by ~4.5e-3.
    let frozen = (-t0 * h).exp();
    assert!((frozen - exact).abs() > 1e-3);
    // An Euler step with J at the end of the step is only first-order accurate.
    let euler_end = 1.0 + h * (-(t0 + h));
    assert!(
        (euler_end - exact).abs() > 1e-3,
        "the old scheme is off by {}",
        (euler_end - exact).abs()
    );
}

#[test]
fn test_tangent_step_without_tangents_is_plain_rk4() {
    let lorenz = LorenzSystem::classic();
    let x0 = [1.0, 1.0, 1.0];
    let (next, tangents) = rk4_step_with_tangents(&lorenz, 0.5, &x0, &[], 0.02, 1e-6);
    assert!(tangents.is_empty());
    assert_eq!(next, rk4_step(&lorenz, 0.5, &x0, 0.02));
}

// ── lyapunov_exponents_qr ────────────────────────────────────────────────────

#[test]
fn test_lyapunov_qr_harmonic_oscillator_is_conservative() {
    // The Euler tangent flow with J at x_{n+1} gave Σλ = ln(1 + h²ω²)/h ≈ 0.0995
    // at h = 0.1; RK4 gives ln of the RK4 amplification factor, ~1e-7.
    let exponents = lyapunov_exponents_qr(&harmonic_oscillator(), &[1.0, 0.0], 0.1, 200);
    assert_eq!(exponents.len(), 2);
    let sum: f64 = exponents.iter().sum();
    assert!(sum.abs() < 1e-6, "Σλ = {sum}");
    for (k, lambda) in exponents.iter().enumerate() {
        assert!(lambda.abs() < 1e-5, "λ[{k}] = {lambda}");
    }
}

#[test]
fn test_lyapunov_qr_linear_systems_match_the_eigenvalues() {
    // Real eigenvalues -1 and -3.
    let diagonal = Linear2 {
        a: [-1.0, 0.0, 0.0, -3.0],
    };
    let exponents = lyapunov_exponents_qr(&diagonal, &[1.0, 1.0], 0.01, 2000);
    assert!((exponents[0] + 1.0).abs() < 1e-6, "λ1 = {}", exponents[0]);
    assert!((exponents[1] + 3.0).abs() < 1e-6, "λ2 = {}", exponents[1]);
    // Damped rotation: eigenvalues -0.5 ± 2i, both exponents -0.5.
    let spiral = Linear2 {
        a: [-0.5, 2.0, -2.0, -0.5],
    };
    let exponents = lyapunov_exponents_qr(&spiral, &[1.0, 0.5], 0.01, 4000);
    for (k, lambda) in exponents.iter().enumerate() {
        assert!((lambda + 0.5).abs() < 1e-5, "λ[{k}] = {lambda}");
    }
}

#[test]
fn test_lyapunov_qr_lorenz_spectrum() {
    let lorenz = LorenzSystem::classic();
    let h = 0.01;
    // The classic exponents: λ1 ≈ 0.906, λ2 = 0, λ3 ≈ -14.572, and the
    // exponents add up to the trace of the Jacobian, -(σ + 1 + β) = -13.667.
    let start = settle(&lorenz, &[1.0, 1.0, 1.0], h, 2000);
    let exponents = lyapunov_exponents_qr(&lorenz, &start, h, 100_000);
    assert_eq!(exponents.len(), 3);
    let sum: f64 = exponents.iter().sum();
    let trace = -(10.0 + 1.0 + 8.0 / 3.0);
    assert!((sum - trace).abs() < 0.02, "Σλ = {sum}, expected {trace}");
    assert!(
        (exponents[0] - 0.906).abs() < 0.03,
        "λ1 = {} should be ≈ 0.906",
        exponents[0]
    );
    assert!(
        exponents[1].abs() < 0.03,
        "λ2 = {} should be ≈ 0",
        exponents[1]
    );
    assert!(
        (exponents[2] + 14.572).abs() < 0.1,
        "λ3 = {} should be ≈ -14.572",
        exponents[2]
    );
    // Kaplan–Yorke dimension of the Lorenz attractor ≈ 2.06.
    let dimension = kaplan_yorke_dim(&exponents);
    assert!((dimension - 2.06).abs() < 0.02, "D_KY = {dimension}");
}

#[test]
fn test_lyapunov_qr_degenerate_arguments() {
    let lorenz = LorenzSystem::classic();
    assert_eq!(
        lyapunov_exponents_qr(&lorenz, &[1.0, 1.0, 1.0], 0.01, 0),
        vec![0.0; 3]
    );
    assert_eq!(
        lyapunov_exponents_qr(&lorenz, &[1.0, 1.0, 1.0], 0.0, 100),
        vec![0.0; 3]
    );
    assert_eq!(
        lyapunov_exponents_qr(&lorenz, &[1.0, 1.0, 1.0], f64::NAN, 100),
        vec![0.0; 3]
    );
}

// ── maximal_lyapunov_exponent ────────────────────────────────────────────────

#[test]
fn test_mle_of_a_contracting_linear_flow_is_exact() {
    // Every vector decays like e^{-t}: the exponent is -1.  Euler would give
    // ln(1 - h)/h = -1.005 at h = 0.01, RK4 is exact to ~1e-10.  The base
    // state sits at the fixed point (the origin), so the finite-difference
    // Jacobian is exact and only the tangent flow is tested.
    let sys = Linear2 {
        a: [-1.0, 0.0, 0.0, -3.0],
    };
    let mle = maximal_lyapunov_exponent(&sys, &[0.0, 0.0], 0.01, 2000, 10);
    assert!((mle + 1.0).abs() < 1e-8, "MLE = {mle}");
    // A growing direction: eigenvalues 0.7 and -2.
    let saddle = Linear2 {
        a: [0.7, 0.0, 0.0, -2.0],
    };
    let mle = maximal_lyapunov_exponent(&saddle, &[0.0, 0.0], 0.01, 2000, 5);
    assert!((mle - 0.7).abs() < 1e-8, "MLE = {mle}");
}

#[test]
fn test_mle_lorenz_matches_the_known_exponent() {
    let lorenz = LorenzSystem::classic();
    let h = 0.01;
    // Discard the transient, then average over 1000 time units.
    let start = settle(&lorenz, &[1.0, 1.0, 1.0], h, 2000);
    let mle = maximal_lyapunov_exponent(&lorenz, &start, h, 100_000, 10);
    assert!(
        (mle - 0.906).abs() < 0.03,
        "MLE = {mle}, expected 0.906 ± 0.03"
    );
}

#[test]
fn test_mle_with_zero_renorm_does_not_panic() {
    let lorenz = LorenzSystem::classic();
    let x0 = [1.0, 1.0, 1.0];
    // `renorm == 0` used to be a remainder-by-zero panic; it now means
    // "renormalise after every step".
    let zero = maximal_lyapunov_exponent(&lorenz, &x0, 0.01, 500, 0);
    let one = maximal_lyapunov_exponent(&lorenz, &x0, 0.01, 500, 1);
    assert!(zero.is_finite());
    assert_eq!(zero, one);
}

#[test]
fn test_mle_degenerate_arguments() {
    let lorenz = LorenzSystem::classic();
    let x0 = [1.0, 1.0, 1.0];
    // No renormalisation takes place, so there is nothing to average.
    assert_eq!(maximal_lyapunov_exponent(&lorenz, &x0, 0.01, 0, 10), 0.0);
    assert_eq!(maximal_lyapunov_exponent(&lorenz, &x0, 0.01, 9, 10), 0.0);
    assert_eq!(maximal_lyapunov_exponent(&lorenz, &x0, 0.0, 100, 10), 0.0);
    assert_eq!(
        maximal_lyapunov_exponent(&lorenz, &x0, f64::INFINITY, 100, 10),
        0.0
    );
}

// ── kaplan_yorke_dim ─────────────────────────────────────────────────────────

#[test]
fn test_kaplan_yorke_dim_of_an_empty_spectrum_is_zero() {
    // `exponents[0]` used to be read unconditionally.
    assert_eq!(kaplan_yorke_dim(&[]), 0.0);
}

#[test]
fn test_kaplan_yorke_dim_known_spectra() {
    // Lorenz: 2 + (0.9056 + 0) / 14.5723.
    let lorenz = kaplan_yorke_dim(&[0.9056, 0.0, -14.5723]);
    assert!(
        (lorenz - (2.0 + 0.9056 / 14.5723)).abs() < 1e-12,
        "D = {lorenz}"
    );
    // A single non-negative exponent: every exponent sum stays non-negative.
    assert_eq!(kaplan_yorke_dim(&[0.5]), 1.0);
    // All exponents negative: a fixed point.
    assert_eq!(kaplan_yorke_dim(&[-1.0, -2.0]), 0.0);
    // The partial sums never become negative: full-dimensional.
    assert_eq!(kaplan_yorke_dim(&[1.0, 0.5, -0.25]), 3.0);
    // Hénon-like: j = 1, D = 1 + 0.42 / 1.62.
    let henon = kaplan_yorke_dim(&[0.42, -1.62]);
    assert!((henon - (1.0 + 0.42 / 1.62)).abs() < 1e-12, "D = {henon}");
}

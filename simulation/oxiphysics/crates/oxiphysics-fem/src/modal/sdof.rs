// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Exact time stepping of one damped modal oscillator.

/// Rigid-body modes have `ω` below this value; they are integrated as a free mass.
const RIGID_BODY_OMEGA: f64 = 1e-30;

/// Damping ratios closer to `1` than this are integrated with the critically damped
/// formulas.  The general formulas are exact and numerically stable right up to this
/// band (`ζ² − 1` is formed as `(ζ − 1)(ζ + 1)` and the factors `sin(ω_d t)/ω_d` and
/// `(1 − e^{−2ω̂t})/(2ω̂)` are evaluated without cancellation); inside it
/// `ω_d, ω̂ < 1.5e-7 ω` and the factors differ from their `ω_d → 0` limits by a
/// relative `(ω_d t)²/6 ≤ 4e-15 (ω t)²`.
const CRITICAL_DAMPING_BAND: f64 = 1e-14;

/// Advance the modal oscillator `q̈ + 2ζω q̇ + ω² q = f` by `dt`, with the modal
/// force `f` held constant over the step, and return the new `(q, q̇)`.
///
/// The step is the exact solution.  Writing `q_s = f/ω²` for the static response
/// and `q_f = q − q_s`, every regime has the form
///
/// ```text
/// q_f(dt) = E · [ c · q_f + s · (q̇ + ζω q_f) ]
/// q̇(dt)   = E · [ c · q̇  − s · (ω² q_f + ζω q̇) ]
/// ```
///
/// with
///
/// | regime              | `E`                | `c`                 | `s`                         |
/// |---------------------|--------------------|---------------------|-----------------------------|
/// | `ζ < 1` (under)     | `e^{−ζω dt}`       | `cos(ω_d dt)`       | `sin(ω_d dt)/ω_d`           |
/// | `ζ = 1` (critical)  | `e^{−ω dt}`        | `1`                 | `dt`                        |
/// | `ζ > 1` (over)      | `e^{−(ζω−ω̂) dt}`   | `(1 + e^{−2ω̂ dt})/2` | `(1 − e^{−2ω̂ dt})/(2ω̂)`   |
///
/// where `ω_d = ω√(1−ζ²)` and `ω̂ = ω√(ζ²−1)`.  For `ζ > 1` this is the exact
/// `c₁ e^{s₁ dt} + c₂ e^{s₂ dt}` solution with `s₁,₂ = −ω(ζ ∓ √(ζ²−1))`, written
/// so that the slow decay rate `ζω − ω̂ = ω/(ζ + √(ζ²−1))` has no cancellation, no
/// exponential can overflow and the limit `ζ → 1` is smooth.
///
/// A rigid-body mode (`ω ≈ 0`) is a free mass, `q̈ = f`.
pub(super) fn advance_sdof(
    q: f64,
    q_dot: f64,
    omega: f64,
    zeta: f64,
    modal_force: f64,
    dt: f64,
) -> (f64, f64) {
    if omega < RIGID_BODY_OMEGA {
        return (
            q + q_dot * dt + 0.5 * modal_force * dt * dt,
            q_dot + modal_force * dt,
        );
    }
    let q_static = modal_force / (omega * omega);
    let q_free = q - q_static;
    let zeta_omega = zeta * omega;

    let (decay, c, s) = if (zeta - 1.0).abs() <= CRITICAL_DAMPING_BAND {
        ((-zeta_omega * dt).exp(), 1.0, dt)
    } else if zeta < 1.0 {
        let omega_d = omega * ((1.0 - zeta) * (1.0 + zeta)).sqrt();
        (
            (-zeta_omega * dt).exp(),
            (omega_d * dt).cos(),
            (omega_d * dt).sin() / omega_d,
        )
    } else {
        let root = ((zeta - 1.0) * (zeta + 1.0)).sqrt();
        let omega_hat = omega * root;
        let slow_rate = omega / (zeta + root);
        let fast = (-2.0 * omega_hat * dt).exp();
        (
            (-slow_rate * dt).exp(),
            0.5 * (1.0 + fast),
            -(-2.0 * omega_hat * dt).exp_m1() / (2.0 * omega_hat),
        )
    };

    let drive = q_dot + zeta_omega * q_free;
    let q_free_new = decay * (c * q_free + s * drive);
    let q_dot_new = decay * (c * q_dot - s * (omega * omega * q_free + zeta_omega * q_dot));
    (q_static + q_free_new, q_dot_new)
}

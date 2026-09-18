//! Regression tests for the Rayleigh-Lamb residuals and the S0 phase velocity.
//!
//! The residuals are validated against independent facts instead of against
//! themselves: the thickness resonances at `k = 0`, the textbook `tan` form of
//! the equations, the thin-plate and Kirchhoff-plate limits, the Rayleigh speed
//! at high frequency and the `tanh` form of the symmetric equation at the
//! returned S0 root.

use super::*;
use crate::error::Error;
use std::f64::consts::PI;

/// Aluminium longitudinal bulk speed \[m/s\].
const C_L: f64 = 6320.0;
/// Aluminium shear bulk speed \[m/s\].
const C_S: f64 = 3130.0;
/// Plate thickness \[m\].
const THICKNESS: f64 = 1.0e-3;

/// Thin-plate (extensional) velocity `2 c_S sqrt(1 - c_S^2 / c_L^2)`.
fn plate_velocity(c_l: f64, c_s: f64) -> f64 {
    2.0 * c_s * (1.0 - c_s * c_s / (c_l * c_l)).sqrt()
}

/// Rayleigh speed from the unsquared Rayleigh equation by bisection, which is
/// independent of the cubic used by the implementation.
fn rayleigh_speed_reference(c_l: f64, c_s: f64) -> f64 {
    let equation = |c: f64| {
        (2.0 - c * c / (c_s * c_s)).powi(2)
            - 4.0 * (1.0 - c * c / (c_l * c_l)).sqrt() * (1.0 - c * c / (c_s * c_s)).sqrt()
    };
    let (mut lo, mut hi) = (0.5 * c_s, 0.999_999_9 * c_s);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if equation(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// `tanh` form of the symmetric Rayleigh-Lamb equation for a wave slower than
/// the longitudinal speed (`p` imaginary):
/// `(k² − q²)² tan(q h)/q − 4 k² γ tanh(γ h) = 0`.  Returns the residual and
/// the sum of the magnitudes of its two terms.
fn symmetric_tanh_form(omega: f64, k: f64, c_l: f64, c_s: f64) -> (f64, f64) {
    let h = 0.5 * THICKNESS;
    let gamma = (k * k - (omega / c_l).powi(2)).sqrt();
    let q2 = (omega / c_s).powi(2) - k * k;
    let shear_term = if q2 > 0.0 {
        let q = q2.sqrt();
        (k * k - q2).powi(2) * (q * h).tan() / q
    } else {
        let kappa = (-q2).sqrt();
        (k * k - q2).powi(2) * (kappa * h).tanh() / kappa
    };
    let flexural_term = 4.0 * k * k * gamma * (gamma * h).tanh();
    (
        shear_term - flexural_term,
        shear_term.abs() + flexural_term.abs(),
    )
}

// ── residuals: thickness resonances ──────────────────────────────────────────

#[test]
fn lamb_symmetric_vanishes_at_symmetric_thickness_resonances() {
    for n in 1..=3 {
        let n = n as f64;
        // omega d / (2 c_L) = (n - 1/2) pi  and  omega d / (2 c_S) = n pi
        for omega in [
            2.0 * PI * (n - 0.5) * C_L / THICKNESS,
            2.0 * PI * n * C_S / THICKNESS,
        ] {
            let scale = (omega / C_S).powi(3);
            let value = lamb_wave_symmetric(omega, 0.0, THICKNESS, C_L, C_S);
            assert!(
                value.abs() < 1e-9 * scale,
                "n = {n}, omega = {omega:e}: F_S = {value:e} (scale {scale:e})"
            );
        }
    }
}

#[test]
fn lamb_symmetric_is_clearly_nonzero_at_antisymmetric_thickness_resonances() {
    for n in 1..=3 {
        let omega = 2.0 * PI * n as f64 * C_L / THICKNESS;
        let scale = (omega / C_S).powi(3);
        let value = lamb_wave_symmetric(omega, 0.0, THICKNESS, C_L, C_S);
        assert!(
            value.abs() > 1e-2 * scale,
            "n = {n}: F_S = {value:e} must not vanish (scale {scale:e})"
        );
    }
}

#[test]
fn lamb_antisymmetric_vanishes_at_antisymmetric_thickness_resonances() {
    for n in 1..=3 {
        let n = n as f64;
        // omega d / (2 c_L) = n pi  and  omega d / (2 c_S) = (n - 1/2) pi
        for omega in [
            2.0 * PI * n * C_L / THICKNESS,
            2.0 * PI * (n - 0.5) * C_S / THICKNESS,
        ] {
            let scale = (omega / C_S).powi(4) * 0.5 * THICKNESS;
            let value = lamb_wave_antisymmetric(omega, 0.0, THICKNESS, C_L, C_S);
            assert!(
                value.abs() < 1e-9 * scale,
                "n = {n}, omega = {omega:e}: F_A = {value:e} (scale {scale:e})"
            );
        }
    }
}

#[test]
fn lamb_antisymmetric_is_clearly_nonzero_at_symmetric_thickness_resonances() {
    for n in 1..=3 {
        let omega = 2.0 * PI * (n as f64 - 0.5) * C_L / THICKNESS;
        let scale = (omega / C_S).powi(4) * 0.5 * THICKNESS;
        let value = lamb_wave_antisymmetric(omega, 0.0, THICKNESS, C_L, C_S);
        assert!(
            value.abs() > 1e-2 * scale,
            "n = {n}: F_A = {value:e} must not vanish (scale {scale:e})"
        );
    }
}

// ── residuals: equivalence with the textbook tan form ───────────────────────

#[test]
fn lamb_residuals_match_textbook_tangent_form() {
    let omega = 2.0 * PI * 1.5e6;
    let h = 0.5 * THICKNESS;
    for fraction in [0.2, 0.5, 0.8] {
        let k = fraction * omega / C_L;
        let p = ((omega / C_L).powi(2) - k * k).sqrt();
        let q = ((omega / C_S).powi(2) - k * k).sqrt();
        let k2_minus_q2 = k * k - q * q;
        let common = (p * h).cos() * (q * h).cos();
        let symmetric_tan =
            (k2_minus_q2.powi(2) * (q * h).tan() + 4.0 * k * k * p * q * (p * h).tan()) * common
                / q;
        let antisymmetric_tan =
            (4.0 * k * k * p * q * (q * h).tan() + k2_minus_q2.powi(2) * (p * h).tan()) * common
                / p;
        let symmetric = lamb_wave_symmetric(omega, k, THICKNESS, C_L, C_S);
        let antisymmetric = lamb_wave_antisymmetric(omega, k, THICKNESS, C_L, C_S);
        assert!(
            (symmetric / symmetric_tan - 1.0).abs() < 1e-10,
            "k/k_L = {fraction}: F_S = {symmetric:e} vs tan form {symmetric_tan:e}"
        );
        assert!(
            (antisymmetric / antisymmetric_tan - 1.0).abs() < 1e-10,
            "k/k_L = {fraction}: F_A = {antisymmetric:e} vs tan form {antisymmetric_tan:e}"
        );
        assert!(
            (symmetric / antisymmetric - 1.0).abs() > 1e-3,
            "the symmetric and antisymmetric residuals must differ"
        );
    }
}

#[test]
fn lamb_residuals_are_continuous_across_the_branch_points() {
    let omega = 2.0 * PI * 3.0e6;
    let scale = (omega / C_S).powi(3);
    for branch_k in [omega / C_L, omega / C_S] {
        let below = branch_k * (1.0 - 1e-10);
        let above = branch_k * (1.0 + 1e-10);
        let s_below = lamb_wave_symmetric(omega, below, THICKNESS, C_L, C_S);
        let s_above = lamb_wave_symmetric(omega, above, THICKNESS, C_L, C_S);
        let a_below = lamb_wave_antisymmetric(omega, below, THICKNESS, C_L, C_S);
        let a_above = lamb_wave_antisymmetric(omega, above, THICKNESS, C_L, C_S);
        assert!(
            (s_below - s_above).abs() < 1e-6 * scale,
            "F_S jumps at k = {branch_k}: {s_below:e} vs {s_above:e}"
        );
        assert!(
            (a_below - a_above).abs() < 1e-6 * scale * THICKNESS * omega / C_S,
            "F_A jumps at k = {branch_k}: {a_below:e} vs {a_above:e}"
        );
    }
}

#[test]
fn lamb_residuals_overflow_to_infinity_but_never_to_nan() {
    let omega = 2.0 * PI * 1.0e6;
    let k = 1.0e6;
    let symmetric = lamb_wave_symmetric(omega, k, THICKNESS, C_L, C_S);
    let antisymmetric = lamb_wave_antisymmetric(omega, k, THICKNESS, C_L, C_S);
    assert!(!symmetric.is_nan() && !antisymmetric.is_nan());
    assert!(symmetric.is_infinite() || symmetric.abs() > 1e300);
    assert!(antisymmetric.is_infinite() || antisymmetric.abs() > 1e300);
    // The scaled form used for root finding stays finite: exponent ~ (|p| + |q|) h.
    let (scaled, exponent) = lamb_symmetric_scaled(omega, k, THICKNESS, C_L, C_S);
    assert!(scaled.is_finite() && scaled != 0.0);
    assert!((exponent - 1000.0).abs() < 1.0, "exponent = {exponent}");
    let (scaled, exponent) = lamb_antisymmetric_scaled(omega, k, THICKNESS, C_L, C_S);
    assert!(scaled.is_finite() && scaled != 0.0);
    assert!((exponent - 1000.0).abs() < 1.0, "exponent = {exponent}");
}

#[test]
fn lamb_residuals_reject_invalid_arguments() {
    let valid = (2.0 * PI * 1.0e6, 2.0e3, THICKNESS, C_L, C_S);
    assert!(lamb_wave_symmetric(valid.0, valid.1, valid.2, valid.3, valid.4).is_finite());
    assert!(lamb_wave_antisymmetric(valid.0, valid.1, valid.2, valid.3, valid.4).is_finite());
    for (omega, k, d, c_l, c_s) in [
        (f64::NAN, valid.1, valid.2, valid.3, valid.4),
        (valid.0, f64::INFINITY, valid.2, valid.3, valid.4),
        (valid.0, valid.1, 0.0, valid.3, valid.4),
        (valid.0, valid.1, -1.0e-3, valid.3, valid.4),
        (valid.0, valid.1, valid.2, 0.0, valid.4),
        (valid.0, valid.1, valid.2, valid.3, -3130.0),
    ] {
        assert!(lamb_wave_symmetric(omega, k, d, c_l, c_s).is_nan());
        assert!(lamb_wave_antisymmetric(omega, k, d, c_l, c_s).is_nan());
    }
}

// ── residuals: low-frequency plate theories ─────────────────────────────────

#[test]
fn lamb_symmetric_root_brackets_the_thin_plate_velocity() {
    let omega = 0.04 * C_S / THICKNESS;
    let c_plate = plate_velocity(C_L, C_S);
    let slow = lamb_wave_symmetric(omega, omega / (0.99 * c_plate), THICKNESS, C_L, C_S);
    let fast = lamb_wave_symmetric(omega, omega / (1.01 * c_plate), THICKNESS, C_L, C_S);
    assert!(
        slow < 0.0 && fast > 0.0,
        "F_S must change sign around the plate velocity: {slow:e}, {fast:e}"
    );
}

#[test]
fn lamb_antisymmetric_root_brackets_the_kirchhoff_flexural_velocity() {
    // E and nu from the bulk speeds (density cancels in the flexural speed).
    let rho = 2700.0;
    let nu = (C_L * C_L - 2.0 * C_S * C_S) / (2.0 * (C_L * C_L - C_S * C_S));
    let e_modulus = rho * C_S * C_S * (3.0 * C_L * C_L - 4.0 * C_S * C_S) / (C_L * C_L - C_S * C_S);
    let omega = 0.04 * C_S / THICKNESS;
    // Kirchhoff plate: c = sqrt(omega) * (D / (rho d))^(1/4), D = E d^3 / (12 (1 - nu^2)).
    let c_flexural = omega.sqrt()
        * (e_modulus * THICKNESS * THICKNESS / (12.0 * rho * (1.0 - nu * nu))).powf(0.25);
    let slow = lamb_wave_antisymmetric(omega, omega / (0.95 * c_flexural), THICKNESS, C_L, C_S);
    let fast = lamb_wave_antisymmetric(omega, omega / (1.05 * c_flexural), THICKNESS, C_L, C_S);
    assert!(
        slow < 0.0 && fast > 0.0,
        "F_A must change sign around the Kirchhoff velocity: {slow:e}, {fast:e}"
    );
}

// ── S0 phase velocity ────────────────────────────────────────────────────────

#[test]
fn rayleigh_cubic_matches_the_rayleigh_equation() {
    for (c_l, c_s) in [
        (C_L, C_S),
        (5900.0, 3200.0),
        (3000.0 * 2.0_f64.sqrt(), 3000.0),
    ] {
        let from_cubic = rayleigh_speed_from_cubic(c_l, c_s);
        let reference = rayleigh_speed_reference(c_l, c_s);
        assert!(
            (from_cubic / reference - 1.0).abs() < 1e-9,
            "c_R = {from_cubic} vs {reference}"
        );
    }
}

#[test]
fn lamb_s0_low_frequency_limit_is_the_thin_plate_velocity() {
    for (c_l, c_s) in [(C_L, C_S), (5900.0, 3200.0), (2500.0, 1000.0)] {
        let omega = 0.02 * c_s / THICKNESS;
        let c = lamb_s0_phase_velocity(omega, THICKNESS, c_l, c_s).expect("S0 root");
        let c_plate = plate_velocity(c_l, c_s);
        assert!(
            (c / c_plate - 1.0).abs() < 1e-4 && c <= c_plate * (1.0 + 1e-9),
            "S0 = {c} vs c_plate = {c_plate} (c_l = {c_l}, c_s = {c_s})"
        );
    }
}

#[test]
fn lamb_s0_high_frequency_limit_is_the_rayleigh_speed() {
    let c_rayleigh = rayleigh_speed_reference(C_L, C_S);
    // 40 MHz mm, and a frequency-thickness so large that cosh/sinh overflow.
    for omega in [2.0 * PI * 40.0e6, 2.0 * PI * 1.0e9] {
        let c = lamb_s0_phase_velocity(omega, THICKNESS, C_L, C_S).expect("S0 root");
        assert!(
            (c / c_rayleigh - 1.0).abs() < 1e-6 && c >= c_rayleigh * (1.0 - 1e-9),
            "omega = {omega:e}: S0 = {c} vs c_R = {c_rayleigh}"
        );
    }
}

#[test]
fn lamb_s0_phase_velocity_decreases_monotonically_between_the_two_limits() {
    let c_plate = plate_velocity(C_L, C_S);
    let c_rayleigh = rayleigh_speed_reference(C_L, C_S);
    let mut previous = f64::INFINITY;
    for i in 0..60 {
        // omega d / c_S from 0.01 to 40 on a logarithmic grid.
        let omega_d_over_c_s = 0.01 * (4000.0_f64).powf(i as f64 / 59.0);
        let omega = omega_d_over_c_s * C_S / THICKNESS;
        let c = lamb_s0_phase_velocity(omega, THICKNESS, C_L, C_S).expect("S0 root");
        assert!(
            c <= previous * (1.0 + 1e-9),
            "S0 must not speed up: {previous} -> {c} at omega d / c_S = {omega_d_over_c_s}"
        );
        assert!(
            c <= c_plate * (1.0 + 1e-9) && c >= c_rayleigh * (1.0 - 1e-9),
            "c = {c}"
        );
        previous = c;
    }
    assert!(
        previous < 1.001 * c_rayleigh,
        "the highest sampled frequency is close to the Rayleigh speed"
    );
}

#[test]
fn lamb_s0_root_satisfies_the_tanh_form_and_residual_is_nonzero_elsewhere() {
    // 0.3 .. 12 MHz mm, covering q real (c > c_S) and q imaginary (c < c_S).
    for frequency_thickness in [0.3e3, 1.0e3, 2.5e3, 3.5e3, 6.0e3, 12.0e3] {
        let omega = 2.0 * PI * frequency_thickness / THICKNESS;
        let c = lamb_s0_phase_velocity(omega, THICKNESS, C_L, C_S).expect("S0 root");
        let k = omega / c;
        let (residual, magnitude) = symmetric_tanh_form(omega, k, C_L, C_S);
        assert!(
            residual.abs() < 1e-9 * magnitude,
            "f d = {frequency_thickness}: tanh-form residual {residual:e} (terms {magnitude:e})"
        );
        let (value, exponent) = lamb_symmetric_scaled(omega, k, THICKNESS, C_L, C_S);
        let mut scale = 0.0_f64;
        for factor in [0.9, 1.1] {
            let (off_root, off_exponent) =
                lamb_symmetric_scaled(omega, factor * k, THICKNESS, C_L, C_S);
            scale = scale.max(off_root.abs() * (off_exponent - exponent).exp());
        }
        assert!(
            value.abs() < 1e-8 * scale,
            "f d = {frequency_thickness}: root residual {value:e} vs off-root {scale:e}"
        );
        let slower = lamb_wave_symmetric(omega, k * 1.05, THICKNESS, C_L, C_S);
        let faster = lamb_wave_symmetric(omega, k * 0.95, THICKNESS, C_L, C_S);
        assert!(
            slower.is_finite() && faster.is_finite() && slower * faster < 0.0,
            "f d = {frequency_thickness}: the residual must change sign across the root: \
             {slower:e}, {faster:e}"
        );
    }
}

#[test]
fn lamb_s0_is_non_dispersive_for_zero_poisson_ratio() {
    // c_L = sqrt(2) c_S  <=>  nu = 0, where S0 is the pure P-wave c_L for low d f.
    let c_s = 3000.0;
    let c_l = c_s * 2.0_f64.sqrt();
    for ratio in [0.05, 0.5, 1.0] {
        let omega = ratio * c_s / THICKNESS;
        let c = lamb_s0_phase_velocity(omega, THICKNESS, c_l, c_s).expect("S0 root");
        assert!(
            (c / c_l - 1.0).abs() < 1e-6,
            "omega d / c_S = {ratio}: S0 = {c} vs c_L = {c_l}"
        );
    }
}

#[test]
fn lamb_s0_rejects_invalid_input() {
    let omega = 2.0 * PI * 1.0e6;
    let invalid = [
        lamb_s0_phase_velocity(0.0, THICKNESS, C_L, C_S),
        lamb_s0_phase_velocity(f64::NAN, THICKNESS, C_L, C_S),
        lamb_s0_phase_velocity(omega, 0.0, C_L, C_S),
        lamb_s0_phase_velocity(omega, -THICKNESS, C_L, C_S),
        lamb_s0_phase_velocity(omega, THICKNESS, C_L, 0.0),
        lamb_s0_phase_velocity(omega, THICKNESS, f64::INFINITY, C_S),
        // A Poisson ratio of -1 corresponds to c_L = 2 c_S / sqrt(3) = 1.1547 c_S.
        lamb_s0_phase_velocity(omega, THICKNESS, 1.15 * C_S, C_S),
        lamb_s0_phase_velocity(omega, THICKNESS, 0.5 * C_S, C_S),
    ];
    for result in invalid {
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }
}

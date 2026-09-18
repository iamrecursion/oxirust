// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the short-term-encounter collision probability.
//!
//! The reference values were computed with 50-digit arithmetic (numerical
//! integration of the defining integral with Bessel `I₀` from `mpmath`, cross
//! checked against the non-central chi-square distribution of `scipy`).  Before
//! the fix, `collision_probability` ignored the miss distance entirely.

use super::collision::{bessel_i0_scaled, gauss_legendre_8};
use super::collision_probability_with_miss_distance as collision_probability;
use std::f64::consts::PI;

/// `1 − exp(−R²/(2σ²))`, the probability of a direct hit, accurate for small `R/σ`.
fn direct_hit(radius: f64, sigma: f64) -> f64 {
    -(-(radius * radius) / (2.0 * sigma * sigma)).exp_m1()
}

/// Largest tolerated relative error of a probability above `TAIL`.
const BODY_TOL: f64 = 1e-13;

/// Probabilities below this are far-tail values; the quadrature resolves them
/// only to about 1e-8 relative (their absolute error is still ~1e-28).
const TAIL: f64 = 1e-6;

/// Largest tolerated relative error of a probability below `TAIL`.
const TAIL_TOL: f64 = 1e-7;

fn assert_close(label: &str, got: f64, expected: f64) {
    let tolerance = if expected.abs() >= TAIL {
        BODY_TOL
    } else {
        TAIL_TOL
    };
    assert!(
        (got - expected).abs() <= tolerance * expected.abs(),
        "{label}: got {got:e}, expected {expected:e}"
    );
}

// ── the scaled Bessel function ──────────────────────────────────────────────

#[test]
fn scaled_bessel_matches_reference_values() {
    // (x, exp(−x)·I₀(x)); 19.9, 20 and 20.1 straddle the switch from the power
    // series to the asymptotic expansion.
    let table = [
        (0.0, 1.0),
        (0.5, 0.64503527044915007),
        (1.0, 0.46575960759364044),
        (2.0, 0.30850832255367104),
        (5.0, 0.18354081260932835),
        (10.0, 0.12783333716342861),
        (19.9, 0.090008588864389597),
        (20.0, 0.089780311884826022),
        (20.1, 0.089553763620613444),
        (50.0, 0.056561626647454193),
        (100.0, 0.039944379299096683),
        (1000.0, 0.012617240455891257),
        (1.0e4, 0.0039894726746047321),
        (1.0e6, 0.00039894233026924578),
    ];
    for (x, expected) in table {
        let got = bessel_i0_scaled(x);
        assert!(
            (got - expected).abs() <= 2e-15 * expected,
            "x = {x}: got {got:e}, expected {expected:e}"
        );
    }
}

#[test]
fn scaled_bessel_stays_finite_where_the_bessel_function_overflows() {
    // I₀(x) itself exceeds f64::MAX for x > 713.
    for x in [700.0, 800.0, 1.0e5, 1.0e12] {
        let value = bessel_i0_scaled(x);
        let leading = 1.0 / (2.0 * PI * x).sqrt();
        assert!(value.is_finite() && value > 0.0, "x = {x}: {value}");
        // 1 + 1/(8x) + 9/(2 (8x)²) are the first terms of the correction factor;
        // the next one is below 3e-10 for x ≥ 700.
        let factor = 1.0 + 1.0 / (8.0 * x) + 9.0 / (2.0 * 64.0 * x * x);
        assert!(
            (value - leading * factor).abs() < 1e-9 * leading,
            "x = {x}: {value:e} vs {:e}",
            leading * factor
        );
    }
}

#[test]
fn scaled_bessel_decreases_monotonically() {
    // exp(−x)·I₀(x) is strictly decreasing from 1 at x = 0.
    let mut previous = bessel_i0_scaled(0.0);
    assert_eq!(previous, 1.0);
    for step in 1..=400 {
        let value = bessel_i0_scaled(0.25 * step as f64);
        assert!(
            value < previous,
            "not decreasing at x = {}",
            0.25 * step as f64
        );
        assert!(value > 0.0);
        previous = value;
    }
}

// ── the quadrature rule ─────────────────────────────────────────────────────

#[test]
fn gauss_legendre_integrates_polynomials_up_to_degree_fifteen_exactly() {
    for degree in 0..=15_i32 {
        let exact = if degree % 2 == 1 {
            0.0
        } else {
            2.0 / f64::from(degree + 1)
        };
        let got = gauss_legendre_8(|x| x.powi(degree), -1.0, 1.0, 1);
        assert!(
            (got - exact).abs() < 1e-15,
            "degree {degree}: {got} vs {exact}"
        );
    }
    // Three panels over [0, 2]: ∫ x¹⁵ dx = 2¹⁶/16.
    let got = gauss_legendre_8(|x| x.powi(15), 0.0, 2.0, 3);
    assert!((got - 4096.0).abs() < 1e-11 * 4096.0, "{got}");
}

#[test]
fn gauss_legendre_resolves_smooth_functions_and_degenerate_panels() {
    let got = gauss_legendre_8(f64::sin, 0.0, PI, 4);
    assert!((got - 2.0).abs() < 1e-14, "{got}");
    // An empty interval integrates to nothing; zero panels count as one.
    assert_eq!(gauss_legendre_8(|x| x, 1.5, 1.5, 4), 0.0);
    let got = gauss_legendre_8(|x| x * x, 0.0, 3.0, 0);
    assert!((got - 9.0).abs() < 1e-14, "{got}");
}

// ── the probability ─────────────────────────────────────────────────────────

#[test]
fn a_direct_hit_matches_the_closed_form() {
    // d = 0: Pc = 1 − exp(−R²/(2σ²)).
    let cases = [
        (10.0, 100.0),
        (1.0, 1.0),
        (5.0, 1.0),
        (3.0, 0.7),
        (7.5, 3.0),
        (1e-3, 1.0),
        (10.0, 1e6),
        (15.0, 1.0),
        (50.0, 1.0),
    ];
    for (radius, sigma) in cases {
        let got = collision_probability(radius, 0.0, sigma);
        let expected = direct_hit(radius, sigma);
        assert!(
            (got - expected).abs() <= 1e-13 * expected.max(1e-300),
            "R = {radius}, sigma = {sigma}: {got:e} vs {expected:e}"
        );
    }
}

#[test]
fn matches_reference_values_of_the_defining_integral() {
    // (R, d, sigma, Pc)
    let table = [
        (10.0, 0.0, 100.0, 0.0049875208073176866),
        (5.0, 3.0, 2.0, 0.76787438743848434),
        (1.0, 1.0, 1.0, 0.26712019620317978),
        (2.0, 2.0, 1.0, 0.39649903938800665),
        (10.0, 8.0, 1.0, 0.97405833704558427),
        (3.0, 0.2, 0.7, 0.99985591495663215),
        (20.0, 150.0, 40.0, 0.00015550996700306331),
        (25.0, 300.0, 120.0, 0.00097544989096773569),
        (0.01, 0.0, 1.0, 4.9998750020833075e-5),
        (2.0, 4.5, 1.5, 0.027239990407689495),
        (7.5, 12.0, 3.0, 0.048499589519292369),
        (0.001, 0.5, 1.0, 4.4124835476921293e-7),
        // The disc boundary passes through the mean of a narrow distribution:
        // a bit less than half of the probability mass is inside.
        (50.0, 50.0, 1.0, 0.49601037767994575),
        (100.0, 100.0, 1.0, 0.49800526366269765),
        // The disc covers the whole distribution.
        (100.0, 60.0, 2.0, 1.0),
        // Far tails: the target is many sigmas away from the mean.
        (1.0, 10.0, 1.0, 3.4136489462303752e-20),
        (0.5, 10.0, 1.0, 2.1478490807689061e-22),
        (30.0, 40.0, 1.0, 6.588868980557369e-24),
    ];
    for (radius, distance, sigma, expected) in table {
        let got = collision_probability(radius, distance, sigma);
        assert_close(
            &format!("R={radius} d={distance} sigma={sigma}"),
            got,
            expected,
        );
    }
}

#[test]
fn the_result_depends_on_the_miss_distance() {
    // The old model returned the same value for every conjunction.
    let radius = 1.0;
    let sigma = 100.0;
    let direct = collision_probability(radius, 0.0, sigma);
    let near = collision_probability(radius, 100.0, sigma);
    let far = collision_probability(radius, 300.0, sigma);
    assert!(direct > near && near > far && far > 0.0);
    // For a small target Pc ≈ (R²/2σ²) exp(−d²/2σ²), so the ratios are
    // exp(−1/2) and exp(−9/2) up to O(R²/σ²) = 1e-4.
    assert!((near / direct - (-0.5_f64).exp()).abs() < 2e-5);
    assert!((far / direct - (-4.5_f64).exp()).abs() < 2e-6);
}

#[test]
fn decreases_monotonically_with_the_miss_distance() {
    for (radius, sigma) in [(3.0, 1.5), (0.1, 1.0), (40.0, 4.0), (1.0, 1e-1)] {
        let first = collision_probability(radius, 0.0, sigma);
        let mut previous = first;
        assert!(previous > 0.0);
        for step in 1..=240 {
            let distance = 0.25 * sigma * f64::from(step);
            let probability = collision_probability(radius, distance, sigma);
            // A probability within round-off of 1 may wobble in its last digits.
            assert!(
                probability <= previous * (1.0 + 1e-12),
                "R = {radius}, sigma = {sigma}: Pc({distance}) = {probability:e} > {previous:e}"
            );
            if previous > 0.0 && previous < 0.999 {
                assert!(
                    probability < previous,
                    "R = {radius}, sigma = {sigma}: Pc stalls at d = {distance}"
                );
            }
            previous = probability;
        }
        assert_eq!(
            previous, 0.0,
            "the far end is beyond the reach of the density"
        );
        assert!(first > 0.0);
    }
}

#[test]
fn a_distant_small_target_has_a_negligible_probability() {
    // d = 10 sigma, R = sigma / 2.
    let probability = collision_probability(0.5, 10.0, 1.0);
    assert!(probability > 0.0 && probability < 1e-20, "{probability:e}");
    // Beyond the integration window the probability is below 1e-31 and reported as 0.
    assert_eq!(collision_probability(1.0, 100.0, 1.0), 0.0);
}

#[test]
fn a_small_target_follows_the_density_at_the_miss_distance() {
    // R ≪ σ: Pc ≈ πR² · pdf(d) = (R²/2σ²) · exp(−d²/2σ²), independent of the
    // quadrature, with a relative error of order R²/σ².
    let radius = 1e-2;
    for distance in [0.0, 1.0, 2.0, 3.0, 4.0] {
        let got = collision_probability(radius, distance, 1.0);
        let approximation = 0.5 * radius * radius * (-0.5 * distance * distance).exp();
        assert!(
            (got - approximation).abs() < 5e-4 * approximation,
            "d = {distance}: {got:e} vs {approximation:e}"
        );
    }
}

#[test]
fn depends_only_on_the_ratios_to_the_uncertainty() {
    for (radius, distance, sigma) in [(2.0, 3.0, 1.5), (10.0, 0.0, 100.0), (1.0, 8.0, 2.0)] {
        let base = collision_probability(radius, distance, sigma);
        for scale in [1e-3, 7.0, 1e3] {
            let scaled = collision_probability(radius * scale, distance * scale, sigma * scale);
            assert!(
                (scaled - base).abs() <= 1e-12 * base,
                "scale {scale}: {scaled:e} vs {base:e}"
            );
        }
    }
}

#[test]
fn stays_a_probability_over_a_wide_parameter_range() {
    for radius in [1e-6, 1e-2, 1.0, 10.0, 1e3] {
        for distance in [0.0, 0.1, 1.0, 10.0, 100.0, 1e4] {
            for sigma in [1e-3, 1.0, 100.0, 1e6] {
                let probability = collision_probability(radius, distance, sigma);
                assert!(
                    (0.0..=1.0).contains(&probability),
                    "R = {radius}, d = {distance}, sigma = {sigma}: {probability}"
                );
            }
        }
    }
}

#[test]
fn a_zero_radius_never_collides() {
    assert_eq!(collision_probability(0.0, 0.0, 100.0), 0.0);
    assert_eq!(collision_probability(0.0, 5.0, 1.0), 0.0);
    assert_eq!(collision_probability(0.0, 0.0, 0.0), 0.0);
}

#[test]
fn exactly_known_positions_make_the_encounter_deterministic() {
    assert_eq!(collision_probability(5.0, 4.9, 0.0), 1.0);
    assert_eq!(collision_probability(5.0, 5.0, 0.0), 1.0);
    assert_eq!(collision_probability(5.0, 5.1, 0.0), 0.0);
    // An uncertainty so small that the ratios overflow behaves the same way.
    let tiny = f64::MIN_POSITIVE / 1e10;
    assert!(tiny > 0.0 && (1.0 / tiny).is_infinite());
    assert_eq!(collision_probability(1.0, 0.5, tiny), 1.0);
    assert_eq!(collision_probability(1.0, 2.0, tiny), 0.0);
}

#[test]
fn invalid_arguments_give_nan() {
    let invalid = [
        (-1.0, 0.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 0.0, -1.0),
        (f64::NAN, 0.0, 1.0),
        (1.0, f64::NAN, 1.0),
        (1.0, 0.0, f64::NAN),
        (f64::INFINITY, 0.0, 1.0),
        (1.0, f64::INFINITY, 1.0),
        (1.0, 0.0, f64::INFINITY),
    ];
    for (radius, distance, sigma) in invalid {
        let probability = collision_probability(radius, distance, sigma);
        assert!(
            probability.is_nan(),
            "R = {radius}, d = {distance}, sigma = {sigma}: {probability}"
        );
    }
}

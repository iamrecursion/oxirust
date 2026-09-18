// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Collision probability of a conjunction between two space objects.
//!
//! The probability is evaluated with the short-term-encounter model of Foster
//! and Estes (1992).  The two objects pass each other so fast that their
//! relative motion is a straight line at constant speed and the position
//! uncertainty can be frozen.  Projecting the relative position onto the plane
//! perpendicular to the relative velocity (the *encounter plane*) then reduces
//! the conjunction to a two-dimensional problem: the probability that a
//! Gaussian-distributed point falls inside the hard-body disc of radius
//! `R = r₁ + r₂`.  Neither the relative speed nor the duration of the
//! conjunction enters the result; the miss distance does.
//!
//! For an isotropic Gaussian of standard deviation `σ` whose mean lies at the
//! distance `d` from the centre of the disc,
//!
//! ```text
//! Pc = ∫₀ᴿ (r/σ²) · exp(−(r² + d²)/(2σ²)) · I₀(r·d/σ²) dr
//! ```
//!
//! where `I₀` is the modified Bessel function of the first kind and order 0.
//! (This is the cumulative distribution of a Rice variable, `1 − Q₁(d/σ, R/σ)`
//! with Marcum's `Q` function.)

use std::f64::consts::PI;

/// Half-width, in standard deviations, of the window around the miss distance
/// that [`collision_probability_with_miss_distance`] integrates over.
///
/// The integrand is a Gaussian bump `exp(−(r − d)²/(2σ²))` centred on the miss
/// distance, so the part of the integral outside `d ± 12σ` is below
/// `exp(−72) ≈ 5e-32`.
const WINDOW_HALF_WIDTH: f64 = 12.0;

/// First argument of [`bessel_i0_scaled`] for which the asymptotic expansion is
/// used instead of the power series.
///
/// The asymptotic series can be summed down to a term of about `exp(−2x)`, so
/// from `x = 20` on it reaches the accuracy of a double (`exp(−40) ≈ 4e-18`),
/// while the power series, whose terms are all positive, is still free of
/// cancellation and overflow below it (`I₀(20) ≈ 4.3e7`).
const BESSEL_ASYMPTOTIC_START: f64 = 20.0;

/// Cap on the number of terms summed by [`bessel_i0_scaled`].
const BESSEL_MAX_TERMS: usize = 100;

/// Terms smaller than this fraction of the running sum are negligible (a
/// quarter of the double-precision unit round-off).
const BESSEL_TERM_TOL: f64 = 1e-17;

/// Non-negative abscissae of the 8-point Gauss–Legendre rule on `[−1, 1]`.
const GAUSS_LEGENDRE_8_NODES: [f64; 4] = [
    0.18343464249564984,
    0.525532409916329,
    0.7966664774136267,
    0.9602898564975362,
];

/// Weights belonging to [`GAUSS_LEGENDRE_8_NODES`] (the rule is symmetric).
const GAUSS_LEGENDRE_8_WEIGHTS: [f64; 4] = [
    0.3626837833783618,
    0.31370664587788716,
    0.22238103445337445,
    0.10122853629037652,
];

/// The scaled modified Bessel function `exp(−x)·I₀(x)` for `x ≥ 0`.
///
/// `I₀` overflows a double for `x > 713`, the scaled function stays in
/// `(0, 1]`.  Up to [`BESSEL_ASYMPTOTIC_START`] the power series
/// `I₀(x) = Σ (x²/4)ᵏ/(k!)²` is summed, beyond it the asymptotic expansion
/// `exp(−x)·I₀(x) ~ (2πx)^(−1/2) · Σ ((2k−1)!!)² / (k!·(8x)ᵏ)`.  Both are
/// accurate to a few units of the last place.
pub(super) fn bessel_i0_scaled(x: f64) -> f64 {
    if x < BESSEL_ASYMPTOTIC_START {
        let quarter_square = 0.25 * x * x;
        let mut term = 1.0_f64;
        let mut sum = 1.0_f64;
        for k in 1..=BESSEL_MAX_TERMS {
            let order = k as f64;
            term *= quarter_square / (order * order);
            sum += term;
            if term <= BESSEL_TERM_TOL * sum {
                break;
            }
        }
        sum * (-x).exp()
    } else {
        let mut term = 1.0_f64;
        let mut sum = 1.0_f64;
        for k in 1..=BESSEL_MAX_TERMS {
            let order = k as f64;
            let odd = 2.0 * order - 1.0;
            let next = term * odd * odd / (order * 8.0 * x);
            if next >= term {
                // The asymptotic series has reached its smallest term and starts
                // to diverge: stop, the error is of that size.
                break;
            }
            term = next;
            sum += term;
            if term <= BESSEL_TERM_TOL * sum {
                break;
            }
        }
        sum / (2.0 * PI * x).sqrt()
    }
}

/// Integral of `f` over `[lo, hi]` by the 8-point Gauss–Legendre rule applied
/// to `panels` equal sub-intervals.
///
/// The rule integrates polynomials up to degree 15 exactly on every panel, so
/// an analytic integrand is resolved to double precision as soon as a panel is
/// no wider than the scale on which the integrand varies.
pub(super) fn gauss_legendre_8<F>(f: F, lo: f64, hi: f64, panels: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let panels = panels.max(1);
    let width = (hi - lo) / panels as f64;
    let half_width = 0.5 * width;
    let mut total = 0.0;
    for panel in 0..panels {
        let centre = lo + (panel as f64 + 0.5) * width;
        let panel_sum: f64 = GAUSS_LEGENDRE_8_NODES
            .iter()
            .zip(GAUSS_LEGENDRE_8_WEIGHTS.iter())
            .map(|(node, weight)| {
                weight * (f(centre + half_width * node) + f(centre - half_width * node))
            })
            .sum();
        total += half_width * panel_sum;
    }
    total
}

/// Probability that two space objects collide in a conjunction, given the
/// miss distance of their predicted closest approach.
///
/// This is the short-term-encounter probability of Foster and Estes (1992):
/// the position uncertainty at the time of closest approach is an isotropic
/// Gaussian of standard deviation `sigma_pos` in the encounter plane
/// (perpendicular to the relative velocity), the hard bodies are the disc of
/// radius `combined_radius` around the primary, and
///
/// ```text
/// Pc = ∫₀ᴿ (r/σ²) · exp(−(r² + d²)/(2σ²)) · I₀(r·d/σ²) dr
/// ```
///
/// with `R = combined_radius`, `d = miss_distance` and `σ = sigma_pos`.  The
/// result is largest for a direct hit, `Pc(d = 0) = 1 − exp(−R²/(2σ²))`, and
/// falls monotonically with the miss distance; for a small target
/// (`R ≪ σ`) it is `≈ (R²/2σ²)·exp(−d²/(2σ²))`.  The relative speed and the
/// duration of the encounter do not enter the model.
///
/// The integral is evaluated by composite 8-point Gauss–Legendre quadrature
/// over the window `d ± 12σ` in which the integrand is not negligible, with
/// panels no wider than `σ`; the absolute error is at the level of the double
/// precision round-off.
///
/// # Arguments
/// * `combined_radius` — Combined hard-body radius `r₁ + r₂` \[m\].
/// * `miss_distance` — Distance between the objects at closest approach,
///   e.g. from [`miss_distance`](super::miss_distance) \[m\].
/// * `sigma_pos` — Combined position uncertainty (1-sigma, isotropic) in the
///   encounter plane \[m\].
///
/// # Returns
/// The collision probability in `[0, 1]`.  A zero `combined_radius` gives `0`.
/// With exact positions (`sigma_pos == 0`) the encounter is deterministic: the
/// probability is `1` if `miss_distance <= combined_radius` and `0` otherwise.
/// NaN is returned if an argument is negative, NaN or infinite.
pub fn collision_probability_with_miss_distance(
    combined_radius: f64,
    miss_distance: f64,
    sigma_pos: f64,
) -> f64 {
    let is_valid = |value: f64| value.is_finite() && value >= 0.0;
    if !(is_valid(combined_radius) && is_valid(miss_distance) && is_valid(sigma_pos)) {
        return f64::NAN;
    }
    if combined_radius == 0.0 {
        return 0.0;
    }
    let deterministic_hit = if miss_distance <= combined_radius {
        1.0
    } else {
        0.0
    };
    if sigma_pos == 0.0 {
        return deterministic_hit;
    }

    // Work in units of the standard deviation: radius b, miss distance a.
    let radius = combined_radius / sigma_pos;
    let offset = miss_distance / sigma_pos;
    if !(radius.is_finite() && offset.is_finite()) {
        // The uncertainty is so small that the ratios overflow: exact encounter.
        return deterministic_hit;
    }
    let lower = (offset - WINDOW_HALF_WIDTH).max(0.0);
    let upper = radius.min(offset + WINDOW_HALF_WIDTH);
    if lower >= upper {
        // The disc lies entirely beyond the reach of the distribution (or has
        // underflowed to nothing): the probability is below 1e-31.
        return 0.0;
    }
    let panels = (upper - lower).ceil() as usize;
    let probability = gauss_legendre_8(
        |u| {
            // u · exp(−(u² + a²)/2) · I₀(a·u), the exponential folded into the
            // scaled Bessel function so that nothing overflows.
            u * (-0.5 * (u - offset) * (u - offset)).exp() * bessel_i0_scaled(offset * u)
        },
        lower,
        upper,
        panels,
    );
    probability.clamp(0.0, 1.0)
}

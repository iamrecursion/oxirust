//! Poisson random variates: Knuth's product method for small means and
//! Hörmann's PTRS transformed rejection for large ones.
//!
//! Knuth's method multiplies uniforms until the product drops below `exp(-λ)`:
//! its cost grows linearly with `λ`, and for `λ` above about 745 the threshold
//! underflows to zero, so the loop only stops when the product itself
//! underflows and the result saturates near 745 whatever `λ` is.  PTRS needs
//! between 2.3 and 2.7 uniforms per variate for every `λ >= 10`.

use super::functions::log_factorial;
use std::f64::consts::TAU;

/// Smallest mean for which PTRS is valid (W. Hörmann, 1993); below it the
/// product method is both exact and cheap.
const PTRS_MIN_LAMBDA: f64 = 1.0e1;

/// Mean from which the normal approximation replaces PTRS.
///
/// The PTRS acceptance test compares `-λ + k ln λ` with `ln k!`, quantities of
/// size `λ ln λ` that cancel to order one; their rounding error grows as
/// `λ ln λ` times the unit roundoff and reaches about `1e-6` here.  The Poisson
/// law itself differs from the rounded normal `N(λ, λ)` by only about
/// `1 / sqrt(λ)` times a small constant at this mean, so the approximation is
/// the more accurate of the two beyond it.
const NORMAL_APPROXIMATION_MIN_LAMBDA: f64 = 1.0e9;

/// Draws one Poisson variate with mean `lambda`, taking uniform samples of
/// `[0, 1)` from `uniform`.
///
/// * `lambda < 10`: Knuth's product method, exact.
/// * `10 <= lambda < 1e9`: PTRS transformed rejection with squeeze (Hörmann,
///   "The transformed rejection method for generating Poisson random
///   variables", 1993), exact and `O(1)` in expectation (2.3 to 2.7 uniforms per
///   variate, decreasing with `lambda`).
/// * `lambda >= 1e9`: the normal approximation `round(λ + sqrt(λ) Z)`, whose
///   error in total variation is of the order of `1e-5` at the threshold (it is
///   proportional to the skewness `1 / sqrt(λ)`) and smaller beyond.  Variates
///   beyond `u64::MAX` saturate.
///
/// Returns `0` for `lambda <= 0` and for a `lambda` that is `NaN` or infinite,
/// without drawing any uniform.
pub(crate) fn poisson_ptrs(lambda: f64, uniform: &mut impl FnMut() -> f64) -> u64 {
    if !lambda.is_finite() || lambda <= 0.0 {
        return 0;
    }
    if lambda < PTRS_MIN_LAMBDA {
        poisson_knuth(lambda, uniform)
    } else if lambda < NORMAL_APPROXIMATION_MIN_LAMBDA {
        poisson_transformed_rejection(lambda, uniform)
    } else {
        poisson_normal_approximation(lambda, uniform)
    }
}

/// Knuth's product method: the number of uniforms whose running product stays
/// above `exp(-lambda)`.
fn poisson_knuth(lambda: f64, uniform: &mut impl FnMut() -> f64) -> u64 {
    let threshold = (-lambda).exp();
    let mut count = 0_u64;
    let mut product = uniform();
    while product > threshold {
        count += 1;
        product *= uniform();
    }
    count
}

/// Hörmann's PTRS (Poisson, transformed rejection with squeeze) for
/// `lambda >= 10`.
fn poisson_transformed_rejection(lambda: f64, uniform: &mut impl FnMut() -> f64) -> u64 {
    let sqrt_lambda = lambda.sqrt();
    let ln_lambda = lambda.ln();
    let b = 0.931 + 2.53 * sqrt_lambda;
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let v_r = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = uniform() - 0.5;
        let v = uniform();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
        // Squeeze: accepted without evaluating the probability mass function.
        if us >= 0.07 && v <= v_r {
            return k as u64;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        let ln_accept = v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln();
        let ln_mass = -lambda + k * ln_lambda - log_factorial(k as u64);
        if ln_accept <= ln_mass {
            return k as u64;
        }
    }
}

/// `round(lambda + sqrt(lambda) Z)` with a standard normal `Z` (Box-Muller),
/// clamped at zero; casting saturates at `u64::MAX`.
fn poisson_normal_approximation(lambda: f64, uniform: &mut impl FnMut() -> f64) -> u64 {
    let u1 = loop {
        let u = uniform();
        if u > 0.0 {
            break u;
        }
    };
    let u2 = uniform();
    let z = (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos();
    (lambda + lambda.sqrt() * z + 0.5).floor().max(0.0) as u64
}

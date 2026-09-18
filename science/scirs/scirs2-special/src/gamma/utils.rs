//! Utility functions for gamma function computation

use crate::error::{SpecialError, SpecialResult};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::approximations::stirling_approximation_ln;
use super::digamma::digamma;

/// Asymptotic gamma function for large negative values to avoid overflow
#[allow(dead_code)]
pub(super) fn asymptotic_gamma_large_negative<F: Float + FromPrimitive + std::ops::AddAssign>(
    x: F,
) -> F {
    // For very large negative x, use the reflection formula with asymptotic expansions
    // to avoid catastrophic cancellation
    let x_f64 = x.to_f64().expect("Operation failed");
    let n = (-x_f64).floor() as i32;
    let _z = x + F::from(n).expect("Failed to convert to float"); // z is the fractional part in [0,1)

    // Use asymptotic expansion for large negative arguments
    // Γ(x) = π / (sin(πx) * Γ(1-x))
    // For large |x|, Γ(1-x) ≈ Stirling's approximation

    let pi = F::from(std::f64::consts::PI).expect("Failed to convert to float");
    let oneminus_x = F::one() - x;

    // Use Stirling for the positive large argument
    let log_gamma_pos = stirling_approximation_ln(oneminus_x);
    let log_sin_pi_x = enhanced_log_sin_pi_x(x);
    let log_pi = pi.ln();

    let sign: F = enhanced_reflection_sign(x_f64);
    let log_result = log_pi - log_sin_pi_x - log_gamma_pos;

    if log_result < F::from(std::f64::MAX.ln() * 0.9).expect("Operation failed") {
        sign * log_result.exp()
    } else if sign > F::zero() {
        F::infinity()
    } else {
        F::neg_infinity()
    }
}

/// Stable computation for gamma near large negative integers
#[allow(dead_code)]
pub(super) fn stable_gamma_near_large_negative_integer<
    F: Float + FromPrimitive + std::ops::AddAssign,
>(
    x: F,
    n: i32,
) -> F {
    let epsilon = x + F::from(n).expect("Failed to convert to float");

    // For large n, use logarithmic computation to avoid overflow
    // Γ(x) ≈ (-1)^n / (n! * ε) where ε = x + n

    // Use Stirling's approximation for log(n!)
    let n_f = F::from(n as f64).expect("Failed to convert to float");
    let log_n_factorial = stirling_approximation_ln(n_f + F::one());

    let sign = if n % 2 == 0 { F::one() } else { -F::one() };
    let log_epsilon = epsilon.abs().ln();

    let log_result = -log_n_factorial - log_epsilon;

    if log_result < F::from(std::f64::MAX.ln() * 0.9).expect("Operation failed") {
        sign / epsilon * log_result.exp()
    } else if epsilon > F::zero() {
        if sign > F::zero() {
            F::infinity()
        } else {
            F::neg_infinity()
        }
    } else if sign > F::zero() {
        F::neg_infinity()
    } else {
        F::infinity()
    }
}

/// Enhanced computation of log(|sin(πx)|) for better numerical stability
#[allow(dead_code)]
pub(super) fn enhanced_log_sin_pi_x<F: Float + FromPrimitive>(x: F) -> F {
    let pi = F::from(std::f64::consts::PI).expect("Failed to convert to float");
    let x_f64 = x.to_f64().expect("Operation failed");

    // Reduce x to the fundamental period to improve accuracy
    let x_reduced = x_f64 - x_f64.floor();
    let x_red = F::from(x_reduced).expect("Failed to convert to float");

    // Use different approaches based on the reduced value
    if x_reduced < 0.5 {
        // For x in [0, 0.5), use sin(πx) directly
        (pi * x_red).sin().abs().ln()
    } else {
        // For x in [0.5, 1), use sin(π(1-x)) = sin(πx)
        let complement = F::one() - x_red;
        (pi * complement).sin().abs().ln()
    }
}

/// Enhanced sign computation for reflection formula with extreme values
#[allow(dead_code)]
pub(super) fn enhanced_reflection_sign<F: Float + FromPrimitive>(xf64: f64) -> F {
    // For the reflection formula Γ(x) = π / (sin(πx) * Γ(1-x))
    // The sign depends on both sin(πx) and the parity

    let x_floor = xf64.floor();
    let _n = x_floor as i32;

    // sin(πx) has the same sign as sin(π(x - floor(x)))
    let fractional_part = xf64 - x_floor;

    if fractional_part == 0.0 {
        // x is an integer, sin(πx) = 0, return NaN indicator
        return F::nan();
    }

    // For negative integers n, the sign alternates
    // sin(π(x - n)) > 0 when fractional_part ∈ (0, 1)
    let sin_sign = if fractional_part > 0.0 && fractional_part < 1.0 {
        F::one()
    } else {
        -F::one()
    };

    // The reflection formula includes division by sin(πx)
    // So we need 1/sin_sign
    if sin_sign > F::zero() {
        F::one()
    } else {
        -F::one()
    }
}

/// Enhanced numerical validation for extreme gamma function values
#[allow(dead_code)]
pub(super) fn validate_gamma_computation<
    F: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
>(
    x: F,
    result: F,
) -> SpecialResult<F> {
    let x_f64 = x.to_f64().expect("Operation failed");

    // Check for obvious invalid inputs
    if x.is_nan() {
        return Err(SpecialError::DomainError("Input x is NaN".to_string()));
    }

    // Check for negative integers (poles)
    if x < F::zero() {
        let nearest_int = x_f64.round() as i32;
        if nearest_int <= 0 && (x_f64 - nearest_int as f64).abs() < 1e-14 {
            return Err(SpecialError::DomainError(format!(
                "Gamma function has a pole at x = {x_f64}"
            )));
        }
    }

    // Enhanced result validation with condition number estimation
    if result.is_nan() && !x.is_nan() {
        return Err(SpecialError::ComputationError(format!(
            "Gamma computation failed for x = {x_f64}, result is NaN"
        )));
    }

    // Check for potential overflow/underflow issues
    if result.is_infinite() {
        if x_f64 > 171.5 {
            // Expected overflow for large positive x
            return Ok(result);
        } else if x_f64 < 0.0 && (x_f64 - x_f64.round()).abs() < 1e-12 {
            // Expected overflow near negative integers
            return Ok(result);
        } else {
            return Err(SpecialError::ComputationError(format!(
                "Unexpected overflow in gamma computation for x = {x_f64}"
            )));
        }
    }

    // Check for potential underflow
    if result.is_zero() && x_f64 > 0.0 && x_f64 < 171.0 {
        return Err(SpecialError::ComputationError(format!(
            "Unexpected underflow in gamma computation for x = {x_f64}"
        )));
    }

    // Estimate condition number for numerical stability assessment
    let condition_estimate = estimate_gamma_condition_number(x);
    if condition_estimate > 1e12 {
        #[cfg(feature = "gpu")]
        log::warn!(
            "High condition number ({:.2e}) for gamma({}), result may be inaccurate",
            condition_estimate,
            x_f64
        );
    }

    Ok(result)
}

/// Estimate condition number for gamma function to assess numerical stability
#[allow(dead_code)]
pub(super) fn estimate_gamma_condition_number<
    F: Float
        + FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
>(
    x: F,
) -> f64 {
    let x_f64 = x.to_f64().expect("Operation failed");
    let h = 1e-8;

    // For condition number estimation: κ = |x * Γ'(x) / Γ(x)|
    // Use finite differences to approximate Γ'(x)
    if x_f64 > 0.0 && x_f64 < 100.0 {
        // Use the digamma function: Γ'(x)/Γ(x) = ψ(x)
        let psi_x = digamma(x).to_f64().unwrap_or(0.0);
        (x_f64 * psi_x).abs()
    } else {
        // For extreme values, use a simplified estimate
        if x_f64 > 100.0 {
            x_f64.ln() // Large x: condition number ~ log(x)
        } else {
            1.0 / x_f64.abs() // Small x: condition number ~ 1/|x|
        }
    }
}

/// Polygamma function - the nth derivative of the digamma function.
///
/// This function computes the polygamma function ψ^(n)(x), which is defined as:
///
/// ```text
/// ψ^(n)(x) = d^(n+1)/dx^(n+1) ln Γ(x) = d^n/dx^n ψ(x)
/// ```
///
/// where ψ(x) = digamma(x) is the digamma function (ψ^(0)(x)).
///
/// **Mathematical Properties**:
///
/// 1. **Special cases**:
///    - ψ^(0)(x) = digamma(x)
///    - ψ^(1)(x) = trigamma(x) = π²/6 - Σ[k=0..∞] 1/(x+k)²
///    - ψ^(2)(x) = tetragamma(x) = 2 Σ[k=0..∞] 1/(x+k)³
///
/// 2. **Recurrence relation**: ψ^(n)(x+1) = ψ^(n)(x) + (-1)^n n!/x^(n+1)
///
/// 3. **Asymptotic behavior**: For large x, ψ^(n)(x) ~ (-1)^(n+1) n!/x^(n+1)
///
/// **Physical Applications**:
/// - Statistical mechanics (correlation functions)
/// - Quantum field theory (loop calculations)
/// - Number theory (special values of zeta functions)
///
/// # Arguments
///
/// * `n` - Order of the derivative (non-negative integer)
/// * `x` - Input value (must be positive for real result)
///
/// # Returns
///
/// * ψ^(n)(x) Polygamma function value
///
/// # Examples
///
/// ```
/// use scirs2_special::gamma::polygamma;
///
/// // ψ^(0)(1) = digamma(1) = -γ ≈ -0.5772156649
/// let psi0_1 = polygamma(0, 1.0f64);
/// assert!((psi0_1 + 0.5772156649).abs() < 1e-8);
///
/// // ψ^(1)(1) = trigamma(1) = π²/6 ≈ 1.6449340668
/// let psi1_1 = polygamma(1, 1.0f64);
/// assert!((psi1_1 - 1.6449340668).abs() < 1e-8);
/// ```
#[allow(dead_code)]
pub fn polygamma<
    F: Float
        + FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
>(
    n: u32,
    x: F,
) -> F {
    // Handle special cases
    if x <= F::zero() {
        return F::nan();
    }

    // For n = 0, return digamma
    if n == 0 {
        return digamma(x);
    }

    // For x = 1, use exact values based on Riemann zeta function
    // ψ^(n)(1) = (-1)^(n+1) n! ζ(n+1)
    if (x - F::one()).abs() < F::from(1e-10).expect("Failed to convert constant to float") {
        // Compute using known values of ζ(n+1)
        let zeta_value = match n {
            1 => std::f64::consts::PI.powi(2) / 6.0, // ζ(2) = π²/6
            2 => {
                // ζ(3) ≈ 1.2020569031595942 (Apéry's constant)
                // ψ^(2)(1) = (-1)^3 * 2! * ζ(3) = -2 * 1.202... = -2.404...
                1.2020569031595942
            }
            3 => std::f64::consts::PI.powi(4) / 90.0, // ζ(4) = π⁴/90
            _ => {
                // For other n, fall through to series computation
                0.0
            }
        };

        if zeta_value != 0.0 {
            let sign = if n.is_multiple_of(2) {
                -F::one()
            } else {
                F::one()
            };
            let n_factorial = factorial_f(n);
            return sign * F::from(n_factorial * zeta_value).expect("Failed to convert to float");
        }
    }

    // For large x, use asymptotic expansion
    if x > F::from(20.0).expect("Failed to convert constant to float") {
        // Asymptotic series: ψ^(n)(x) ~ (-1)^(n+1) n!/x^(n+1) * [1 + (n+1)/(2x) + ...]
        // Sign convention: (-1)^(n+1), so for n=1 (trigamma) we get +1, for n=2 we get -1, etc.
        let sign = if n.is_multiple_of(2) {
            -F::one() // n even → n+1 odd → (-1)^(n+1) = -1
        } else {
            F::one() // n odd → n+1 even → (-1)^(n+1) = +1
        };
        let n_factorial = factorial_f(n);
        let x_power = x.powi(n as i32 + 1);

        let leading_term =
            sign * F::from(n_factorial).expect("Failed to convert to float") / x_power;

        // Add first correction term
        let correction = F::from(n + 1).expect("Failed to convert to float")
            / (F::from(2.0).expect("Failed to convert constant to float") * x);

        return leading_term * (F::one() + correction);
    }

    // For moderate x, use the series representation
    // ψ^(n)(x) = (-1)^(n+1) n! Σ[k=0..∞] 1/(x+k)^(n+1)
    // Sign convention: (-1)^(n+1), so for n=1 (trigamma) we get +1, for n=2 we get -1, etc.
    let sign = if n.is_multiple_of(2) {
        -F::one() // n even → n+1 odd → (-1)^(n+1) = -1
    } else {
        F::one() // n odd → n+1 even → (-1)^(n+1) = +1
    };
    let n_factorial = factorial_f(n);

    let mut sum = F::zero();
    let n_plus_1 = n + 1;

    // Sum the series with improved convergence check
    for k in 0..10000 {
        let term = (x + F::from(k).expect("Failed to convert to float")).powi(-(n_plus_1 as i32));
        sum += term;

        // Check for convergence - use absolute value comparison
        // Need term.abs() < eps * sum.abs() for proper convergence
        if k > 10
            && term.abs() < F::from(1e-16).expect("Failed to convert constant to float") * sum.abs()
        {
            break;
        }
    }

    sign * F::from(n_factorial).expect("Failed to convert to float") * sum
}

/// Helper function to compute factorial as f64
#[allow(dead_code)]
fn factorial_f(n: u32) -> f64 {
    match n {
        0 | 1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        4 => 24.0,
        5 => 120.0,
        6 => 720.0,
        7 => 5040.0,
        8 => 40320.0,
        9 => 362880.0,
        10 => 3628800.0,
        _ => {
            // For larger n, compute iteratively
            let mut result = 1.0f64;
            for i in 1..=n {
                result *= i as f64;
            }
            result
        }
    }
}

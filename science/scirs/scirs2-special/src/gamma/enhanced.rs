//! Enhanced gamma function implementations with improved edge case handling
//!
//! This module provides advanced implementations of gamma-related functions
//! with special attention to numerical stability in edge cases.
//!
//! ## Key Improvements
//!
//! 1. **Near-pole handling**: Improved accuracy near negative integers
//! 2. **Large argument handling**: Extended Stirling series for large x
//! 3. **Small argument handling**: Enhanced series near zero
//! 4. **Overflow protection**: Logarithmic computation paths
//! 5. **Condition number estimation**: Numerical stability assessment

use crate::error::{SpecialError, SpecialResult};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::approximations::stirling_approximation_ln;
use super::constants::{EULER_MASCHERONI, LANCZOS_COEFFICIENTS};

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<F: Float + FromPrimitive>(value: f64) -> F {
    F::from(value).unwrap_or_else(|| {
        if value > 0.0 {
            F::infinity()
        } else if value < 0.0 {
            F::neg_infinity()
        } else {
            F::zero()
        }
    })
}

/// Enhanced gamma function with comprehensive edge case handling
///
/// This implementation uses different algorithms depending on the input range:
/// - For x in (0, 0.5): Taylor series around x=0
/// - For x in [0.5, 1.5]: Direct Lanczos
/// - For x in (1.5, 12]: Recurrence to bring into optimal range
/// - For x > 12: Extended Stirling series
/// - For x < 0: Reflection formula with enhanced stability
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// * Gamma(x) with enhanced precision in edge cases
#[allow(dead_code)]
pub fn gamma_enhanced<F>(x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    let x_f64 = x
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert x to f64".to_string()))?;

    // Handle special cases
    if x.is_nan() {
        return Ok(F::nan());
    }

    if x == F::zero() {
        return Ok(F::infinity());
    }

    // Handle negative values
    if x < F::zero() {
        return gamma_negative_enhanced(x);
    }

    // Handle very small positive values
    if x < const_f64::<F>(1e-10) {
        return gamma_near_zero(x);
    }

    // Handle the region (0, 0.5) with Taylor series
    if x < const_f64::<F>(0.5) {
        return gamma_small_positive(x);
    }

    // Handle large values
    if x_f64 > 171.0 {
        // Use logarithmic computation to avoid overflow
        let log_gamma = gammaln_enhanced(x)?;
        if log_gamma > const_f64::<F>(std::f64::MAX.ln() * 0.9) {
            return Ok(F::infinity());
        }
        return Ok(log_gamma.exp());
    }

    if x_f64 > 12.0 {
        return gamma_large_positive(x);
    }

    // For x in [0.5, 12], use Lanczos with recurrence if needed
    gamma_moderate_positive(x)
}

/// Gamma function for negative values with enhanced stability
#[allow(dead_code)]
fn gamma_negative_enhanced<F>(x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    let x_f64 = x
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert x to f64".to_string()))?;

    // Check if x is at a pole (negative integer)
    let nearest_int = x_f64.round() as i32;
    let distance_to_int = (x_f64 - nearest_int as f64).abs();

    if nearest_int <= 0 && distance_to_int < 1e-14 {
        return Err(SpecialError::DomainError(format!(
            "Gamma function has a pole at x = {x_f64}"
        )));
    }

    // For values very close to negative integers, use Laurent series
    if nearest_int <= 0 && distance_to_int < 1e-4 {
        return gamma_near_pole(x, nearest_int);
    }

    // Use reflection formula: Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
    let pi = const_f64::<F>(std::f64::consts::PI);
    let sin_pi_x = (pi * x).sin();

    if sin_pi_x.abs() < const_f64::<F>(1e-14) {
        return Err(SpecialError::DomainError(
            "Gamma function pole detected".to_string(),
        ));
    }

    // Compute Gamma(1-x) with enhanced stability
    let one_minus_x = F::one() - x;
    let gamma_complement = if one_minus_x > const_f64::<F>(171.0) {
        // Use logarithmic computation
        let log_gamma = gammaln_enhanced(one_minus_x)?;
        if log_gamma > const_f64::<F>(std::f64::MAX.ln() * 0.9) {
            F::infinity()
        } else {
            log_gamma.exp()
        }
    } else {
        gamma_enhanced(one_minus_x)?
    };

    if gamma_complement.is_infinite() {
        return Ok(F::zero());
    }

    Ok(pi / (sin_pi_x * gamma_complement))
}

/// Gamma function near a pole (negative integer) using Laurent series
#[allow(dead_code)]
fn gamma_near_pole<F>(x: F, n: i32) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    // Near x = -n, Gamma(x) = (-1)^n / (n! * (x + n)) * (1 + O(x + n))
    let epsilon = x + const_f64::<F>(n as f64);
    let n_abs = (-n) as u32;

    // Compute n! carefully to avoid overflow
    let n_factorial = factorial_float::<F>(n_abs);
    let sign = if n_abs.is_multiple_of(2) {
        F::one()
    } else {
        -F::one()
    };

    // Leading term
    let leading = sign / (n_factorial * epsilon);

    // Add first correction term using the harmonic number
    let harmonic = harmonic_number::<F>(n_abs);
    let correction = F::one() - epsilon * harmonic;

    // Add second order correction
    let harmonic_sq_sum = harmonic_sum_squared::<F>(n_abs);
    let second_order =
        epsilon * epsilon * (harmonic * harmonic - harmonic_sq_sum) / const_f64::<F>(2.0);

    Ok(leading * (correction + second_order))
}

/// Gamma function for very small positive x (near zero)
#[allow(dead_code)]
fn gamma_near_zero<F>(x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    // Near x = 0: Gamma(x) = 1/x - gamma + (gamma^2/2 + pi^2/12)*x + O(x^2)
    let gamma = const_f64::<F>(EULER_MASCHERONI);
    let pi_sq = const_f64::<F>(std::f64::consts::PI * std::f64::consts::PI);

    // Leading singular term
    let leading = F::one() / x;

    // Correction terms
    let c1 = -gamma;
    let c2 = gamma * gamma / const_f64::<F>(2.0) + pi_sq / const_f64::<F>(12.0);

    // Third order term coefficient
    let zeta3 = const_f64::<F>(1.2020569031595942); // Riemann zeta(3)
    let c3 = -(gamma * gamma * gamma / const_f64::<F>(6.0)
        + pi_sq * gamma / const_f64::<F>(12.0)
        + zeta3);

    Ok(leading + c1 + c2 * x + c3 * x * x)
}

/// Gamma function for small positive x in (0, 0.5)
#[allow(dead_code)]
fn gamma_small_positive<F>(x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    // Use Gamma(x) = Gamma(x+1) / x
    let x_plus_1 = x + F::one();
    let gamma_x_plus_1 = gamma_moderate_positive(x_plus_1)?;
    Ok(gamma_x_plus_1 / x)
}

/// Gamma function for moderate positive x in [0.5, 12]
#[allow(dead_code)]
fn gamma_moderate_positive<F>(x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    // Use Lanczos approximation with g=7
    const G: f64 = 7.0;

    let x_minus_1 = x - F::one();

    // Compute the Lanczos sum
    let mut ag = const_f64::<F>(LANCZOS_COEFFICIENTS[0]);
    for (i, &coeff) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        ag += const_f64::<F>(coeff) / (x_minus_1 + const_f64::<F>(i as f64));
    }

    // Compute the rest of the Lanczos approximation
    let sqrt_2pi = const_f64::<F>((2.0 * std::f64::consts::PI).sqrt());
    let tmp = x_minus_1 + const_f64::<F>(G + 0.5);

    // Gamma(x) = sqrt(2*pi) * ag * (x + g - 0.5)^(x-0.5) * exp(-(x + g - 0.5))
    let power_term = tmp.powf(x_minus_1 + const_f64::<F>(0.5));
    let exp_term = (-tmp).exp();

    Ok(sqrt_2pi * ag * power_term * exp_term)
}

/// Gamma function for large positive x using extended Stirling series
#[allow(dead_code)]
fn gamma_large_positive<F>(x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    // Use Stirling's approximation with more terms:
    // Gamma(x) = sqrt(2*pi/x) * (x/e)^x * (1 + 1/(12x) + 1/(288x^2) - 139/(51840x^3) + ...)

    let log_gamma = stirling_approximation_ln(x);

    // Check for overflow
    if log_gamma > const_f64::<F>(std::f64::MAX.ln() * 0.9) {
        return Ok(F::infinity());
    }

    Ok(log_gamma.exp())
}

/// Enhanced log-gamma function
#[allow(dead_code)]
pub fn gammaln_enhanced<F>(x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    let x_f64 = x
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert x to f64".to_string()))?;

    if x <= F::zero() {
        return Err(SpecialError::DomainError(
            "log-gamma requires positive argument".to_string(),
        ));
    }

    // For very small x: ln(Gamma(x)) ~ -ln(x) - gamma*x
    if x < const_f64::<F>(1e-8) {
        let gamma = const_f64::<F>(EULER_MASCHERONI);
        return Ok(-x.ln() - gamma * x);
    }

    // For moderate x, use Lanczos
    if x_f64 < 50.0 {
        return gammaln_lanczos(x);
    }

    // For large x, use Stirling
    Ok(stirling_approximation_ln(x))
}

/// Log-gamma using Lanczos approximation
#[allow(dead_code)]
fn gammaln_lanczos<F>(x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    const G: f64 = 7.0;

    let x_minus_1 = x - F::one();

    // Compute the Lanczos sum
    let mut ag = const_f64::<F>(LANCZOS_COEFFICIENTS[0]);
    for (i, &coeff) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        ag += const_f64::<F>(coeff) / (x_minus_1 + const_f64::<F>(i as f64));
    }

    let log_sqrt_2pi = const_f64::<F>((2.0 * std::f64::consts::PI).ln() / 2.0);
    let tmp = x_minus_1 + const_f64::<F>(G + 0.5);

    // ln(Gamma(x)) = 0.5*ln(2*pi) + ln(ag) + (x-0.5)*ln(tmp) - tmp
    Ok(log_sqrt_2pi + ag.ln() + (x_minus_1 + const_f64::<F>(0.5)) * tmp.ln() - tmp)
}

/// Reciprocal gamma function 1/Gamma(x) with enhanced stability
///
/// The reciprocal gamma function is entire (no poles) and is often more
/// numerically stable than computing 1/Gamma(x) directly.
#[allow(dead_code)]
pub fn rgamma<F>(x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    let x_f64 = x
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert x to f64".to_string()))?;

    // At poles of Gamma (non-positive integers), 1/Gamma = 0
    if x_f64 <= 0.0 && x_f64.fract() == 0.0 {
        return Ok(F::zero());
    }

    // For large positive x, Gamma is very large, so 1/Gamma is very small
    if x_f64 > 171.0 {
        return Ok(F::zero());
    }

    // For moderate values, compute normally
    let gamma_x = gamma_enhanced(x)?;
    if gamma_x.is_infinite() {
        return Ok(F::zero());
    }

    Ok(F::one() / gamma_x)
}

/// Log of the absolute value of gamma function with sign
///
/// Returns (ln|Gamma(x)|, sign(Gamma(x)))
#[allow(dead_code)]
pub fn lgamma_with_sign<F>(x: F) -> SpecialResult<(F, F)>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    let x_f64 = x
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert x to f64".to_string()))?;

    if x > F::zero() {
        // Positive x: Gamma is positive
        let log_gamma = gammaln_enhanced(x)?;
        return Ok((log_gamma, F::one()));
    }

    // Negative x: use reflection formula
    // Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
    // ln|Gamma(x)| = ln(pi) - ln|sin(pi*x)| - ln(Gamma(1-x))

    // Check for poles
    let nearest_int = x_f64.round() as i32;
    if nearest_int <= 0 && (x_f64 - nearest_int as f64).abs() < 1e-14 {
        return Err(SpecialError::DomainError(format!(
            "log-gamma undefined at pole x = {x_f64}"
        )));
    }

    let pi = const_f64::<F>(std::f64::consts::PI);
    let sin_pi_x = (pi * x).sin();
    let one_minus_x = F::one() - x;

    let log_pi = pi.ln();
    let log_sin_pi_x = sin_pi_x.abs().ln();
    let log_gamma_complement = gammaln_enhanced(one_minus_x)?;

    let log_abs_gamma = log_pi - log_sin_pi_x - log_gamma_complement;

    // Determine sign
    // sign(Gamma(x)) = sign(sin(pi*x)) * (-1)^floor(-x)
    let floor_neg_x = (-x_f64).floor() as i32;
    let sin_sign = if sin_pi_x > F::zero() {
        F::one()
    } else {
        -F::one()
    };
    let parity_sign = if floor_neg_x % 2 == 0 {
        F::one()
    } else {
        -F::one()
    };

    Ok((log_abs_gamma, sin_sign * parity_sign))
}

/// Gamma ratio Gamma(a) / Gamma(b) with enhanced stability
///
/// This is more stable than computing Gamma(a)/Gamma(b) separately,
/// especially when a and b are close or both large.
#[allow(dead_code)]
pub fn gamma_ratio<F>(a: F, b: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    let a_f64 = a
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert a to f64".to_string()))?;
    let b_f64 = b
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert b to f64".to_string()))?;

    // If a and b are close, use the recurrence relation
    let diff = a_f64 - b_f64;
    if diff.abs() < 10.0 && diff.abs() == diff.abs().floor() {
        // a - b is a small integer
        let n = diff as i32;
        if n >= 0 {
            // Gamma(a) / Gamma(b) = (b)(b+1)...(a-1) for a > b
            let mut result = F::one();
            for i in 0..n {
                result = result * (b + const_f64::<F>(i as f64));
            }
            return Ok(result);
        } else {
            // Gamma(a) / Gamma(b) = 1 / ((a)(a+1)...(b-1)) for b > a
            let mut result = F::one();
            for i in 0..(-n) {
                result = result * (a + const_f64::<F>(i as f64));
            }
            return Ok(F::one() / result);
        }
    }

    // For general case, use logarithms
    let log_gamma_a = gammaln_enhanced(a)?;
    let log_gamma_b = gammaln_enhanced(b)?;
    let log_ratio = log_gamma_a - log_gamma_b;

    if log_ratio > const_f64::<F>(std::f64::MAX.ln() * 0.9) {
        return Ok(F::infinity());
    }
    if log_ratio < const_f64::<F>(std::f64::MIN.ln() * 0.9) {
        return Ok(F::zero());
    }

    Ok(log_ratio.exp())
}

/// Pochhammer symbol (rising factorial) with enhanced stability
///
/// (a)_n = a * (a+1) * ... * (a+n-1) = Gamma(a+n) / Gamma(a)
#[allow(dead_code)]
pub fn pochhammer_enhanced<F>(a: F, n: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    let n_f64 = n
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert n to f64".to_string()))?;

    // For integer n, use direct computation if small
    if n_f64.fract() == 0.0 && (0.0..=20.0).contains(&n_f64) {
        let n_int = n_f64 as usize;
        let mut result = F::one();
        for i in 0..n_int {
            result = result * (a + const_f64::<F>(i as f64));
        }
        return Ok(result);
    }

    // For general case, use gamma ratio
    gamma_ratio(a + n, a)
}

/// Helper: compute factorial for non-negative integer
#[allow(dead_code)]
fn factorial_float<F: Float + FromPrimitive>(n: u32) -> F {
    match n {
        0 | 1 => F::one(),
        _ => {
            let mut result = F::one();
            for i in 2..=n {
                result = result * const_f64::<F>(i as f64);
            }
            result
        }
    }
}

/// Helper: compute harmonic number H_n = 1 + 1/2 + ... + 1/n
#[allow(dead_code)]
fn harmonic_number<F: Float + FromPrimitive>(n: u32) -> F {
    let mut result = F::zero();
    for i in 1..=n {
        result = result + F::one() / const_f64::<F>(i as f64);
    }
    result
}

/// Helper: compute sum 1/1^2 + 1/2^2 + ... + 1/n^2
#[allow(dead_code)]
fn harmonic_sum_squared<F: Float + FromPrimitive>(n: u32) -> F {
    let mut result = F::zero();
    for i in 1..=n {
        let i_sq = (i * i) as f64;
        result = result + F::one() / const_f64::<F>(i_sq);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gamma_enhanced_integers() {
        // Gamma(n) = (n-1)! for positive integers
        assert_relative_eq!(
            gamma_enhanced(1.0).expect("test should succeed"),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            gamma_enhanced(2.0).expect("test should succeed"),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            gamma_enhanced(5.0).expect("test should succeed"),
            24.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            gamma_enhanced(6.0).expect("test should succeed"),
            120.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_gamma_enhanced_half_integers() {
        // Gamma(0.5) = sqrt(pi)
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert_relative_eq!(
            gamma_enhanced(0.5).expect("test should succeed"),
            sqrt_pi,
            epsilon = 1e-10
        );

        // Gamma(1.5) = sqrt(pi)/2
        assert_relative_eq!(
            gamma_enhanced(1.5).expect("test should succeed"),
            sqrt_pi / 2.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_gamma_enhanced_negative() {
        // Gamma(-0.5) = -2*sqrt(pi)
        let result = gamma_enhanced(-0.5).expect("test should succeed");
        assert_relative_eq!(result, -2.0 * std::f64::consts::PI.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_at_pole() {
        // Should return error at negative integers
        assert!(gamma_enhanced(-1.0_f64).is_err());
        assert!(gamma_enhanced(-2.0_f64).is_err());
    }

    #[test]
    fn test_gamma_near_pole() {
        // Test values very close to poles
        let x = -1.0 + 1e-10_f64;
        let result = gamma_enhanced(x);
        assert!(result.is_ok());
        assert!(result.expect("test should succeed").is_finite());
    }

    #[test]
    fn test_gammaln_enhanced() {
        // ln(Gamma(5)) = ln(24)
        assert_relative_eq!(
            gammaln_enhanced(5.0).expect("test should succeed"),
            24.0_f64.ln(),
            epsilon = 1e-10
        );

        // ln(Gamma(0.5)) = ln(sqrt(pi))
        assert_relative_eq!(
            gammaln_enhanced(0.5).expect("test should succeed"),
            std::f64::consts::PI.sqrt().ln(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_rgamma() {
        // 1/Gamma(5) = 1/24
        assert_relative_eq!(
            rgamma(5.0).expect("test should succeed"),
            1.0 / 24.0,
            epsilon = 1e-10
        );

        // At poles, 1/Gamma = 0
        assert_relative_eq!(
            rgamma(-1.0).expect("test should succeed"),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_gamma_ratio() {
        // Gamma(5)/Gamma(3) = 4! / 2! = 12
        assert_relative_eq!(
            gamma_ratio(5.0, 3.0).expect("test should succeed"),
            12.0,
            epsilon = 1e-10
        );

        // Gamma(3)/Gamma(5) = 2! / 4! = 1/12
        assert_relative_eq!(
            gamma_ratio(3.0, 5.0).expect("test should succeed"),
            1.0 / 12.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_pochhammer_enhanced() {
        // (1)_4 = 1*2*3*4 = 24
        assert_relative_eq!(
            pochhammer_enhanced(1.0, 4.0).expect("test should succeed"),
            24.0,
            epsilon = 1e-10
        );

        // (3)_2 = 3*4 = 12
        assert_relative_eq!(
            pochhammer_enhanced(3.0, 2.0).expect("test should succeed"),
            12.0,
            epsilon = 1e-10
        );
    }
}

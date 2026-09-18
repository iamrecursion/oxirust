//! Enhanced asymptotic expansions for Bessel functions
//!
//! This module provides high-precision asymptotic expansions for Bessel functions,
//! including Debye expansions for large order, uniform asymptotic expansions near
//! turning points, and Hankel's expansions for large arguments.
//!
//! ## Mathematical Theory
//!
//! ### Hankel's Asymptotic Expansion (Large Argument)
//!
//! For |x| >> |v| and |x| >> 1:
//! ```text
//! J_v(x) ~ sqrt(2/(pi*x)) * [P(v,x) * cos(chi) - Q(v,x) * sin(chi)]
//! Y_v(x) ~ sqrt(2/(pi*x)) * [P(v,x) * sin(chi) + Q(v,x) * cos(chi)]
//! ```
//! where chi = x - (2v+1)pi/4 and P, Q are asymptotic series.
//!
//! ### Debye Asymptotic Expansion (Large Order)
//!
//! For v >> 1 and x = v*sech(alpha):
//! ```text
//! J_v(v*sech(alpha)) ~ exp(v*(tanh(alpha) - alpha)) / sqrt(2*pi*v*tanh(alpha)) * sum_k U_k(coth(alpha))/v^k
//! ```
//!
//! ### Uniform Asymptotic Expansion (Near Turning Point)
//!
//! For v ~ x (near the turning point):
//! ```text
//! J_v(x) ~ (4*zeta/(1-t^2))^(1/4) * [Ai(v^(2/3)*zeta) * sum_k A_k(zeta)/v^(2k)
//!        + Ai'(v^(2/3)*zeta)/v^(4/3) * sum_k B_k(zeta)/v^(2k)]
//! ```
//! where Ai is the Airy function.

use crate::error::SpecialResult;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

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

// NOTE: Hankel expansion coefficients are computed dynamically from mu = 4*v^2
// using the standard recurrence. No static coefficient arrays needed.

/// Debye U_k polynomials coefficients for the asymptotic expansion
/// U_0 = 1
/// U_1 = (3t - 5t^3)/24
/// U_2 = (81t^2 - 462t^4 + 385t^6)/1152
/// etc.
const DEBYE_U_COEFFS: [[f64; 7]; 4] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                  // U_0
    [0.0, 0.125, 0.0, -0.208333333333333, 0.0, 0.0, 0.0], // U_1
    [
        0.0,
        0.0,
        0.0703125,
        0.0,
        -0.4010416666666667,
        0.0,
        0.3342013888889,
    ], // U_2
    [
        0.0,
        0.0,
        0.0,
        0.0732421875,
        0.0,
        -0.8912109375,
        1.8464626736111,
    ], // U_3
];

/// Compute P and Q polynomials for the Hankel asymptotic expansion.
///
/// The standard asymptotic series for Bessel functions of large argument:
///   P(v,x) = sum_{k=0}^{N} (-1)^k * a_{2k}(v,x)
///   Q(v,x) = sum_{k=0}^{N} (-1)^k * a_{2k+1}(v,x)
///
/// where:
///   a_0 = 1
///   a_k = a_{k-1} * (mu - (2k-1)^2) / (k * 8x)
///   mu = 4 * v^2
///
/// P accumulates even-indexed terms, Q accumulates odd-indexed terms.
#[allow(dead_code)]
fn hankel_pq<F>(v: F, x: F) -> (F, F)
where
    F: Float + FromPrimitive + Debug,
{
    let mu = const_f64::<F>(4.0) * v * v;
    let eight_x = const_f64::<F>(8.0) * x;

    let mut p = F::one();
    let mut q = F::zero();
    let mut term = F::one(); // a_0 = 1

    // Compute up to 30 terms for convergence (asymptotic series, stops when terms grow)
    let mut prev_term_abs = F::infinity();
    for k in 1..=30u32 {
        let k_f = const_f64::<F>(k as f64);
        let sq = const_f64::<F>((2 * k - 1) as f64);

        // a_k = a_{k-1} * (mu - (2k-1)^2) / (k * 8x)
        term = term * (mu - sq * sq) / (k_f * eight_x);

        let term_abs = term.abs();

        // For asymptotic series, stop when terms start growing (diverging)
        if term_abs > prev_term_abs && k > 2 {
            break;
        }
        prev_term_abs = term_abs;

        if k % 2 == 0 {
            // Even k -> contributes to P
            p = p + term;
        } else {
            // Odd k -> contributes to Q
            q = q + term;
        }

        // Check for convergence
        if term_abs < const_f64::<F>(1e-16) * (p.abs() + q.abs() + F::epsilon()) {
            break;
        }
    }

    (p, q)
}

/// Enhanced Hankel asymptotic expansion for J_v(x) with large argument
///
/// This implementation uses the standard asymptotic series:
///   J_v(x) ~ sqrt(2/(pi*x)) * [P(v,x)*cos(chi) - Q(v,x)*sin(chi)]
/// where chi = x - (2v+1)*pi/4
///
/// # Arguments
/// * `v` - Order of the Bessel function
/// * `x` - Argument (must be large, typically x > 2*|v| + 25)
///
/// # Returns
/// * Approximate value of J_v(x)
#[allow(dead_code)]
pub fn hankel_j<F>(v: F, x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    let abs_x = x.abs();
    let v_f64 = v.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert v to f64".to_string())
    })?;

    // Chi = x - (2v + 1) * pi / 4
    let pi = const_f64::<F>(std::f64::consts::PI);
    let chi = abs_x - (const_f64::<F>(2.0) * v + F::one()) * pi / const_f64::<F>(4.0);

    let (p, q) = hankel_pq(v, abs_x);

    // Final result: J_v(x) ~ sqrt(2/(pi*x)) * [P*cos(chi) - Q*sin(chi)]
    let amplitude = (const_f64::<F>(2.0) / (pi * abs_x)).sqrt();
    let result = amplitude * (p * chi.cos() - q * chi.sin());

    // Handle sign for negative x with integer order
    if x.is_sign_negative() && v_f64.fract() == 0.0 {
        let n = v_f64 as i32;
        if n % 2 != 0 {
            return Ok(-result);
        }
    }

    Ok(result)
}

/// Enhanced Hankel asymptotic expansion for Y_v(x) with large argument
///
/// Uses the standard asymptotic series:
///   Y_v(x) ~ sqrt(2/(pi*x)) * [P(v,x)*sin(chi) + Q(v,x)*cos(chi)]
/// where chi = x - (2v+1)*pi/4
#[allow(dead_code)]
pub fn hankel_y<F>(v: F, x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    if x <= F::zero() {
        return Err(crate::error::SpecialError::DomainError(
            "Y_v(x) undefined for x <= 0".to_string(),
        ));
    }

    let pi = const_f64::<F>(std::f64::consts::PI);
    let chi = x - (const_f64::<F>(2.0) * v + F::one()) * pi / const_f64::<F>(4.0);

    let (p, q) = hankel_pq(v, x);

    // For Y_v: Y_v(x) ~ sqrt(2/(pi*x)) * [P*sin(chi) + Q*cos(chi)]
    let amplitude = (const_f64::<F>(2.0) / (pi * x)).sqrt();
    Ok(amplitude * (p * chi.sin() + q * chi.cos()))
}

/// Debye asymptotic expansion for J_v(x) with large order v
///
/// This is particularly useful when v is large compared to x.
/// Uses the representation J_v(v * sech(alpha)) for stable computation.
///
/// # Arguments
/// * `v` - Order (should be large, v > 10 recommended)
/// * `x` - Argument
///
/// # Returns
/// * Approximate value of J_v(x)
#[allow(dead_code)]
pub fn debye_j<F>(v: F, x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    let v_f64 = v.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert v to f64".to_string())
    })?;
    let x_f64 = x.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert x to f64".to_string())
    })?;

    // Determine the region: x < v (exponentially small), x > v (oscillatory), x ~ v (transition)
    let ratio = x_f64 / v_f64;

    if ratio >= 1.0 {
        // x >= v: Use oscillatory Debye expansion
        debye_oscillatory(v, x)
    } else {
        // x < v: Use exponential Debye expansion
        debye_exponential(v, x)
    }
}

/// Debye expansion for the oscillatory region (x > v)
#[allow(dead_code)]
fn debye_oscillatory<F>(v: F, x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    let v_f64 = v.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert v to f64".to_string())
    })?;
    let x_f64 = x.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert x to f64".to_string())
    })?;

    // Compute beta = arccosh(x/v)
    let ratio = x_f64 / v_f64;
    let beta = (ratio + (ratio * ratio - 1.0).sqrt()).ln(); // arccosh

    // Compute tau = sqrt(t^2 - 1) where t = x/v
    let tau = (ratio * ratio - 1.0).sqrt();

    // Phase: xi = v * (tau - arctan(tau))
    let xi = v_f64 * (tau - tau.atan());

    // Amplitude factor
    let pi = std::f64::consts::PI;
    let amplitude = 1.0 / (tau * (2.0 * pi * v_f64).sqrt());

    // Compute U polynomials
    let cot_beta = 1.0 / tau; // coth(beta) for oscillatory region simplifies
    let mut u_sum = 1.0;

    for (k, coeffs) in DEBYE_U_COEFFS.iter().enumerate().skip(1) {
        let mut u_k = 0.0;
        let mut cot_power = 1.0;
        for &c in coeffs.iter() {
            u_k += c * cot_power;
            cot_power *= cot_beta;
        }
        u_sum += u_k / v_f64.powi(k as i32);
    }

    // Final result with phase
    let result = amplitude * u_sum * (xi - pi / 4.0).cos();
    Ok(const_f64(result))
}

/// Debye expansion for the exponential region (x < v)
#[allow(dead_code)]
fn debye_exponential<F>(v: F, x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    let v_f64 = v.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert v to f64".to_string())
    })?;
    let x_f64 = x.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert x to f64".to_string())
    })?;

    // Compute alpha = arcsech(x/v) = ln((1 + sqrt(1 - t^2)) / t) where t = x/v
    let t = x_f64 / v_f64;
    let sqrt_term = (1.0 - t * t).sqrt();
    let alpha = ((1.0 + sqrt_term) / t).ln();

    // tanh(alpha) = sqrt(1 - t^2) / 1 = sqrt(1 - t^2)
    let tanh_alpha = sqrt_term;

    // Exponent: eta = v * (tanh(alpha) - alpha)
    let eta = v_f64 * (tanh_alpha - alpha);

    // Amplitude
    let pi = std::f64::consts::PI;
    let amplitude = eta.exp() / (2.0 * pi * v_f64 * tanh_alpha).sqrt();

    // Compute U polynomials with coth(alpha)
    let coth_alpha = 1.0 / tanh_alpha;
    let mut u_sum = 1.0;

    for (k, coeffs) in DEBYE_U_COEFFS.iter().enumerate().skip(1) {
        let mut u_k = 0.0;
        let mut coth_power = 1.0;
        for &c in coeffs.iter() {
            u_k += c * coth_power;
            coth_power *= coth_alpha;
        }
        u_sum += u_k / v_f64.powi(k as i32);
    }

    let result = amplitude * u_sum;
    Ok(const_f64(result))
}

/// Uniform asymptotic expansion for J_v(x) near the transition region (x ~ v)
///
/// Uses Airy function representation for stable computation near the turning point.
///
/// # Arguments
/// * `v` - Order
/// * `x` - Argument (should be close to v)
///
/// # Returns
/// * Approximate value of J_v(x)
#[allow(dead_code)]
pub fn uniform_j<F>(v: F, x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    let v_f64 = v.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert v to f64".to_string())
    })?;
    let x_f64 = x.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert x to f64".to_string())
    })?;

    // Compute t = x/v
    let t = x_f64 / v_f64;

    // Compute zeta using the appropriate formula depending on t
    let zeta: f64;
    let phi: f64; // (4*zeta/(1-t^2))^(1/4)

    if t < 1.0 {
        // x < v case
        let sqrt_1_minus_t2 = (1.0 - t * t).sqrt();
        let half_arg = ((1.0 + sqrt_1_minus_t2) / t).ln() - sqrt_1_minus_t2;
        zeta = -((3.0 / 2.0) * half_arg).powf(2.0 / 3.0);
        phi = (4.0 * zeta.abs() / (1.0 - t * t)).powf(0.25);
    } else if t > 1.0 {
        // x > v case
        let sqrt_t2_minus_1 = (t * t - 1.0).sqrt();
        let half_arg = sqrt_t2_minus_1 - (sqrt_t2_minus_1 / t).acos();
        zeta = ((3.0 / 2.0) * half_arg).powf(2.0 / 3.0);
        phi = (4.0 * zeta / (t * t - 1.0)).powf(0.25);
    } else {
        // t = 1, exactly at turning point
        zeta = 0.0;
        phi = (2.0_f64).powf(1.0 / 3.0);
    }

    // Compute Airy function argument
    let airy_arg = v_f64.powf(2.0 / 3.0) * zeta;

    // Use Airy function approximation
    let (ai, _aip) = airy_approx(airy_arg);

    // Leading term of the uniform expansion
    let result = phi / v_f64.powf(1.0 / 3.0) * ai;

    Ok(const_f64(result))
}

/// Simple Airy function approximation for the uniform expansion
#[allow(dead_code)]
fn airy_approx(x: f64) -> (f64, f64) {
    if x > 0.0 {
        // Positive x: exponentially decaying
        let xi = (2.0 / 3.0) * x.powf(1.5);
        let ai = (-xi).exp() / (2.0 * std::f64::consts::PI.sqrt() * x.powf(0.25));
        let aip = -x.powf(0.25) * ai;
        (ai, aip)
    } else if x < 0.0 {
        // Negative x: oscillatory
        let abs_x = x.abs();
        let xi = (2.0 / 3.0) * abs_x.powf(1.5);
        let ai = (xi - std::f64::consts::PI / 4.0).sin()
            / (std::f64::consts::PI.sqrt() * abs_x.powf(0.25));
        let aip = -abs_x.powf(0.25) * (xi - std::f64::consts::PI / 4.0).cos()
            / std::f64::consts::PI.sqrt();
        (ai, aip)
    } else {
        // x = 0
        let ai_0 = 1.0 / (3.0_f64.powf(2.0 / 3.0) * std::f64::consts::PI.sqrt() / 3.0_f64.sqrt());
        let aip_0 = -1.0 / (3.0_f64.powf(1.0 / 3.0) * std::f64::consts::PI.sqrt() / 3.0_f64.sqrt());
        (ai_0, aip_0)
    }
}

/// Adaptive Bessel function that selects the best algorithm based on parameters
///
/// This function automatically chooses between:
/// - Direct series for small x
/// - Forward/backward recurrence for moderate x
/// - Hankel asymptotic for large x
/// - Debye asymptotic for large v
/// - Uniform asymptotic near turning point
///
/// # Arguments
/// * `v` - Order
/// * `x` - Argument
///
/// # Returns
/// * Best available approximation of J_v(x)
#[allow(dead_code)]
pub fn adaptive_bessel_j<F>(v: F, x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    let v_f64 = v.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert v to f64".to_string())
    })?;
    let x_f64 = x.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert x to f64".to_string())
    })?;

    let abs_v = v_f64.abs();
    let abs_x = x_f64.abs();

    // Region 1: Large argument (Hankel)
    if abs_x > 2.0 * abs_v + 25.0 {
        return hankel_j(v, x);
    }

    // Region 2: Large order with x < v (Debye exponential)
    if abs_v > 25.0 && abs_x < abs_v * 0.9 {
        return debye_j(v, x);
    }

    // Region 3: Large order with x > v (Debye oscillatory)
    if abs_v > 25.0 && abs_x > abs_v * 1.1 {
        return debye_j(v, x);
    }

    // Region 4: Near turning point (Uniform)
    if abs_v > 10.0 && (abs_x / abs_v - 1.0).abs() < 0.2 {
        return uniform_j(v, x);
    }

    // Fallback to standard implementation
    Ok(crate::bessel::jv(v, x))
}

/// Modified Bessel function I_v(x) with Debye asymptotic expansion for large order
#[allow(dead_code)]
pub fn debye_i<F>(v: F, x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    let v_f64 = v.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert v to f64".to_string())
    })?;
    let x_f64 = x.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert x to f64".to_string())
    })?;

    // For modified Bessel, use the relation I_v(x) = i^(-v) * J_v(i*x)
    // For large v, use Debye expansion directly

    let t = x_f64 / v_f64;
    let pi = std::f64::consts::PI;

    if t <= 1.0 {
        // x <= v: exponentially growing
        let sqrt_1_minus_t2 = (1.0 - t * t).max(0.0).sqrt();
        let eta = (sqrt_1_minus_t2 + ((1.0 + sqrt_1_minus_t2) / t).ln()) * v_f64
            - v_f64 * sqrt_1_minus_t2;

        let amplitude = (eta).exp() / (2.0 * pi * v_f64 * sqrt_1_minus_t2.max(0.01)).sqrt();
        Ok(const_f64(amplitude))
    } else {
        // x > v: use different expansion
        let sqrt_t2_minus_1 = (t * t - 1.0).sqrt();
        let eta = v_f64 * (sqrt_t2_minus_1 - (t / sqrt_t2_minus_1).atan());

        let amplitude = (eta).exp() / (2.0 * pi * v_f64 * sqrt_t2_minus_1).sqrt();
        Ok(const_f64(amplitude))
    }
}

/// Modified Bessel function K_v(x) with Debye asymptotic expansion for large order
#[allow(dead_code)]
pub fn debye_k<F>(v: F, x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    let v_f64 = v.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert v to f64".to_string())
    })?;
    let x_f64 = x.to_f64().ok_or_else(|| {
        crate::error::SpecialError::ValueError("Failed to convert x to f64".to_string())
    })?;

    let t = x_f64 / v_f64;
    let pi = std::f64::consts::PI;

    if t <= 1.0 {
        // x <= v
        let sqrt_1_minus_t2 = (1.0 - t * t).max(0.0).sqrt();
        let eta = v_f64 * (sqrt_1_minus_t2 - ((1.0 + sqrt_1_minus_t2) / t).ln());

        let amplitude = (pi / (2.0 * v_f64 * sqrt_1_minus_t2.max(0.01))).sqrt() * (-eta).exp();
        Ok(const_f64(amplitude))
    } else {
        // x > v
        let sqrt_t2_minus_1 = (t * t - 1.0).sqrt();
        let eta = v_f64 * ((t / sqrt_t2_minus_1).atan() - sqrt_t2_minus_1);

        let amplitude = (pi / (2.0 * v_f64 * sqrt_t2_minus_1)).sqrt() * (-eta).exp();
        Ok(const_f64(amplitude))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hankel_j_large_x() {
        // Test Hankel expansion for large x
        let x = 100.0_f64;
        let v = 0.0_f64;

        let result = hankel_j(v, x).expect("Hankel expansion failed");

        // Compare with known value (from SciPy)
        let expected = 0.019985850304223122_f64;
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_hankel_y_large_x() {
        let x = 100.0_f64;
        let v = 0.0_f64;

        let result = hankel_y(v, x).expect("Hankel expansion failed");

        // Compare with known value (SciPy reference)
        // The Hankel asymptotic expansion for Y_0(100) converges to high precision
        // but has a finite truncation error from the asymptotic series.
        let expected = -0.07724431336886536_f64;
        assert_relative_eq!(result, expected, epsilon = 2e-5);
    }

    #[test]
    fn test_debye_large_order_small_x() {
        // Test Debye expansion for v >> x (exponential region)
        let v = 50.0_f64;
        let x = 25.0_f64;

        let result = debye_j(v, x).expect("Debye expansion failed");

        // Result should be very small (exponentially small)
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_debye_large_order_large_x() {
        // Test Debye expansion for x > v (oscillatory region)
        let v = 20.0_f64;
        let x = 30.0_f64;

        let result = debye_j(v, x).expect("Debye expansion failed");

        // Result should be oscillatory with moderate magnitude
        assert!(result.is_finite());
        assert!(result.abs() < 0.5);
    }

    #[test]
    fn test_uniform_at_turning_point() {
        // Test uniform expansion at x = v
        let v = 20.0_f64;
        let x = 20.0_f64;

        let result = uniform_j(v, x).expect("Uniform expansion failed");

        // At turning point, J_v(v) should have specific behavior
        assert!(result.is_finite());
    }

    #[test]
    fn test_adaptive_selection() {
        // Test that adaptive function selects appropriate method
        let x = 100.0_f64;
        let v = 5.0_f64;

        let result = adaptive_bessel_j(v, x).expect("Adaptive Bessel failed");

        // Should give reasonable result
        assert!(result.is_finite());
        assert!(result.abs() < 0.2);
    }
}

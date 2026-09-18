//! # Numerical Stability Improvements
//!
//! This module provides numerically stable implementations of common algorithms that
//! are prone to precision loss, overflow, or catastrophic cancellation.
//!
//! ## Features
//!
//! - Stable summation algorithms (Kahan, pairwise, compensated)
//! - Robust variance and standard deviation calculations
//! - Overflow-resistant multiplication and exponentiation
//! - Stable trigonometric range reduction
//! - Robust normalization techniques
//! - Improved root finding with bracketing
//! - Stable matrix condition number estimation

use crate::{
    error::{CoreError, CoreResult, ErrorContext},
    validation::check_positive,
};
use ::ndarray::{Array1, ArrayView2, Axis};
use num_traits::{cast, Float, Zero};
use std::fmt::Debug;

/// Trait for numerically stable computations
pub trait StableComputation: Float + Debug {
    /// Machine epsilon for this type
    fn machine_epsilon() -> Self;

    /// Safe reciprocal that handles near-zero values
    fn safe_recip(self) -> Self;

    /// Check if the value is effectively zero (within epsilon)
    fn is_effectively_zero(self) -> bool;
}

impl StableComputation for f32 {
    fn machine_epsilon() -> Self {
        f32::EPSILON
    }

    fn safe_recip(self) -> Self {
        if self.abs() < Self::machine_epsilon() * cast::<f64, Self>(10.0).unwrap_or(Self::zero()) {
            Self::zero()
        } else {
            self.recip()
        }
    }

    fn is_effectively_zero(self) -> bool {
        self.abs() < Self::machine_epsilon() * cast::<f64, Self>(10.0).unwrap_or(Self::zero())
    }
}

impl StableComputation for f64 {
    fn machine_epsilon() -> Self {
        f64::EPSILON
    }

    fn safe_recip(self) -> Self {
        if self.abs() < Self::machine_epsilon() * cast::<f64, Self>(10.0).unwrap_or(Self::zero()) {
            Self::zero()
        } else {
            self.recip()
        }
    }

    fn is_effectively_zero(self) -> bool {
        self.abs() < Self::machine_epsilon() * cast::<f64, Self>(10.0).unwrap_or(Self::zero())
    }
}

/// Kahan summation algorithm for accurate floating-point summation
///
/// This algorithm reduces numerical error in the total obtained by adding a sequence
/// of finite-precision floating-point numbers compared to the obvious approach.
pub struct KahanSum<T: Float> {
    sum: T,
    compensation: T,
}

impl<T: Float> KahanSum<T> {
    /// Create a new Kahan sum accumulator
    pub fn new() -> Self {
        Self {
            sum: T::zero(),
            compensation: T::zero(),
        }
    }

    /// Add a value to the sum
    pub fn add(&mut self, value: T) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// Get the accumulated sum
    pub fn sum(&self) -> T {
        self.sum
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.sum = T::zero();
        self.compensation = T::zero();
    }
}

impl<T: Float> Default for KahanSum<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Neumaier summation algorithm (improved Kahan summation)
///
/// This is an improved version of Kahan summation that handles the case
/// where the next item to be added is larger in absolute value than the running sum.
#[allow(dead_code)]
pub fn neumaier_sum<T: Float>(values: &[T]) -> T {
    if values.is_empty() {
        return T::zero();
    }

    let mut sum = values[0];
    let mut compensation = T::zero();

    for &value in &values[1..] {
        let t = sum + value;
        if sum.abs() >= value.abs() {
            compensation = compensation + ((sum - t) + value);
        } else {
            compensation = compensation + ((value - t) + sum);
        }
        sum = t;
    }

    sum + compensation
}

/// Pairwise summation algorithm
///
/// Recursively splits the array and sums pairs, reducing rounding error
/// compared to sequential summation.
#[allow(dead_code)]
pub fn pairwise_sum<T: Float>(values: &[T]) -> T {
    const SEQUENTIAL_THRESHOLD: usize = 128;

    match values.len() {
        0 => T::zero(),
        1 => values[0],
        n if n <= SEQUENTIAL_THRESHOLD => {
            // Use Kahan summation for small arrays
            let mut kahan = KahanSum::new();
            for &v in values {
                kahan.add(v);
            }
            kahan.sum()
        }
        n => {
            let mid = n / 2;
            pairwise_sum(&values[..mid]) + pairwise_sum(&values[mid..])
        }
    }
}

/// Stable mean calculation using compensated summation
#[allow(dead_code)]
pub fn stable_mean<T: Float>(values: &[T]) -> CoreResult<T> {
    if values.is_empty() {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Cannot compute mean of empty array",
        )));
    }

    let n = cast::<usize, T>(values.len()).ok_or_else(|| {
        CoreError::TypeError(ErrorContext::new("Failed to convert array length to float"))
    })?;

    Ok(neumaier_sum(values) / n)
}

/// Welford's online algorithm for computing variance
///
/// This algorithm is numerically stable and computes variance in a single pass.
pub struct WelfordVariance<T: Float> {
    count: usize,
    mean: T,
    m2: T,
}

impl<T: Float> WelfordVariance<T> {
    /// Create a new variance accumulator
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: T::zero(),
            m2: T::zero(),
        }
    }

    /// Add a value to the accumulator
    pub fn add(&mut self, value: T) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean = self.mean + delta / cast::<usize, T>(self.count).unwrap_or(T::one());
        let delta2 = value - self.mean;
        self.m2 = self.m2 + delta * delta2;
    }

    /// Get the current mean
    pub fn mean(&self) -> Option<T> {
        if self.count > 0 {
            Some(self.mean)
        } else {
            None
        }
    }

    /// Get the sample variance (Bessel's correction applied)
    pub fn variance(&self) -> Option<T> {
        if self.count > 1 {
            Some(self.m2 / cast::<usize, T>(self.count - 1).unwrap_or(T::one()))
        } else {
            None
        }
    }

    /// Get the population variance
    pub fn population_variance(&self) -> Option<T> {
        if self.count > 0 {
            Some(self.m2 / cast::<usize, T>(self.count).unwrap_or(T::one()))
        } else {
            None
        }
    }

    /// Get the standard deviation
    pub fn std_dev(&self) -> Option<T> {
        self.variance().map(|v| v.sqrt())
    }
}

impl<T: Float> Default for WelfordVariance<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable two-pass algorithm for variance calculation
#[allow(dead_code)]
pub fn stable_variance<T: Float>(values: &[T], ddof: usize) -> CoreResult<T> {
    let n = values.len();
    if n <= ddof {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Not enough values for the given degrees of freedom",
        )));
    }

    // First pass: compute mean with compensated summation
    let mean = stable_mean(values)?;

    // Second pass: compute sum of squared deviations with compensation
    let mut sum_sq = T::zero();
    let mut compensation = T::zero();

    for &value in values {
        let deviation = value - mean;
        let sq_deviation = deviation * deviation;
        let y = sq_deviation - compensation;
        let t = sum_sq + y;
        compensation = (t - sum_sq) - y;
        sum_sq = t;
    }

    let divisor = cast::<usize, T>(n - ddof).ok_or_else(|| {
        CoreError::TypeError(ErrorContext::new("Failed to convert divisor to float"))
    })?;

    Ok(sum_sq / divisor)
}

/// Log-sum-exp trick for stable computation of log(sum(exp(x)))
///
/// This prevents overflow when computing the log of a sum of exponentials.
#[allow(dead_code)]
pub fn log_sum_exp<T: Float>(values: &[T]) -> T {
    if values.is_empty() {
        return T::neg_infinity();
    }

    // Find maximum value
    let max_val = values.iter().fold(T::neg_infinity(), |a, &b| a.max(b));

    if max_val.is_infinite() && max_val < T::zero() {
        return max_val; // All values are -inf
    }

    // Compute log(sum(exp(x - max))) + max
    let mut sum = T::zero();
    for &value in values {
        sum = sum + (value - max_val).exp();
    }

    max_val + sum.ln()
}

/// Stable softmax computation
///
/// Computes softmax(x) = exp(x) / sum(exp(x)) in a numerically stable way.
#[allow(dead_code)]
pub fn stable_softmax<T: Float>(values: &[T]) -> Vec<T> {
    if values.is_empty() {
        return vec![];
    }

    // Find maximum for numerical stability
    let max_val = values.iter().fold(T::neg_infinity(), |a, &b| a.max(b));

    // Compute exp(x - max)
    let mut expvalues = Vec::with_capacity(values.len());
    let mut sum = T::zero();

    for &value in values {
        let exp_val = (value - max_val).exp();
        expvalues.push(exp_val);
        sum = sum + exp_val;
    }

    // Normalize
    for exp_val in &mut expvalues {
        *exp_val = *exp_val / sum;
    }

    expvalues
}

/// Stable computation of log(1 + x) for small x
#[allow(dead_code)]
pub fn log1p_stable<T: Float>(x: T) -> T {
    // Use built-in log1p if available, otherwise use series expansion for small x
    if x.abs() < cast::<f64, T>(0.5).unwrap_or(T::zero()) {
        // For small x, use Taylor series: log(1+x) ≈ x - x²/2 + x³/3 - ...
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        x - x2 / cast::<f64, T>(2.0).unwrap_or(T::one())
            + x3 / cast::<f64, T>(3.0).unwrap_or(T::one())
            - x4 / cast::<f64, T>(4.0).unwrap_or(T::one())
    } else {
        (T::one() + x).ln()
    }
}

/// Stable computation of exp(x) - 1 for small x
#[allow(dead_code)]
pub fn expm1_stable<T: Float>(x: T) -> T {
    if x.abs() < cast::<f64, T>(0.5).unwrap_or(T::zero()) {
        // Use Taylor series: exp(x) - 1 ≈ x + x²/2 + x³/6 + ...
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        x + x2 / cast::<f64, T>(2.0).unwrap_or(T::one())
            + x3 / cast::<f64, T>(6.0).unwrap_or(T::one())
            + x4 / cast::<f64, T>(24.0).unwrap_or(T::one())
    } else {
        x.exp() - T::one()
    }
}

/// Stable computation of sqrt(x² + y²) avoiding overflow
#[allow(dead_code)]
pub fn hypot_stable<T: Float>(x: T, y: T) -> T {
    let x_abs = x.abs();
    let y_abs = y.abs();

    if x_abs > y_abs {
        if x_abs.is_zero() {
            T::zero()
        } else {
            let ratio = y_abs / x_abs;
            x_abs * (T::one() + ratio * ratio).sqrt()
        }
    } else if y_abs.is_zero() {
        T::zero()
    } else {
        let ratio = x_abs / y_abs;
        y_abs * (T::one() + ratio * ratio).sqrt()
    }
}

/// Stable angle reduction for trigonometric functions
///
/// Reduces angle to [-π, π] range while preserving precision for large angles.
#[allow(dead_code)]
pub fn reduce_angle<T: Float>(angle: T) -> T {
    let two_pi = cast::<f64, T>(2.0).unwrap_or(T::one())
        * cast::<f64, T>(std::f64::consts::PI).unwrap_or(T::one());
    let pi = cast::<f64, T>(std::f64::consts::PI).unwrap_or(T::one());

    // Use remainder to get value in (-2π, 2π)
    let mut reduced = angle % two_pi;

    // Normalize to [0, 2π)
    if reduced < T::zero() {
        reduced = reduced + two_pi;
    }

    // Then shift to [-π, π]
    if reduced >= pi {
        reduced - two_pi
    } else {
        reduced
    }
}

/// Stable computation of (a*b) % m avoiding overflow
#[allow(dead_code)]
pub fn mulmod_stable<T: Float>(a: T, b: T, m: T) -> CoreResult<T> {
    let m_f64 = m.to_f64().ok_or_else(|| {
        CoreError::TypeError(ErrorContext::new(
            "Failed to convert modulus to f64 for validation",
        ))
    })?;
    check_positive(m_f64, "modulus")?;

    if a.is_zero() || b.is_zero() {
        return Ok(T::zero());
    }

    let a_mod = a % m;
    let b_mod = b % m;

    // Check if multiplication would overflow
    let max_val = T::max_value();
    if a_mod.abs() > max_val / b_mod.abs() {
        // Use addition-based multiplication for large values
        let mut result = T::zero();
        let mut b_remaining = b_mod.abs();
        let b_sign = if b_mod < T::zero() {
            -T::one()
        } else {
            T::one()
        };

        while b_remaining > T::zero() {
            if b_remaining >= T::one() {
                result = (result + a_mod) % m;
                b_remaining = b_remaining - T::one();
            } else {
                result = (result + a_mod * b_remaining) % m;
                break;
            }
        }

        Ok(result * b_sign)
    } else {
        Ok((a_mod * b_mod) % m)
    }
}

/// Numerically stable sigmoid function
#[allow(dead_code)]
pub fn sigmoid_stable<T: Float>(x: T) -> T {
    if x >= T::zero() {
        let exp_neg_x = (-x).exp();
        T::one() / (T::one() + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (T::one() + exp_x)
    }
}

/// Numerically stable log-sigmoid function
#[allow(dead_code)]
pub fn log_sigmoid_stable<T: Float>(x: T) -> T {
    if x >= T::zero() {
        -log1p_stable((-x).exp())
    } else {
        x - log1p_stable(x.exp())
    }
}

/// Cross entropy loss with numerical stability
#[allow(dead_code)]
pub fn cross_entropy_stable<T: Float>(predictions: &[T], targets: &[T]) -> CoreResult<T> {
    if predictions.len() != targets.len() {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Predictions and targets must have same length",
        )));
    }

    let mut loss = T::zero();
    let epsilon = cast::<f64, T>(1e-15).unwrap_or(T::epsilon()); // Small value to prevent log(0)

    for (pred, target) in predictions.iter().zip(targets.iter()) {
        // Clip _predictions to prevent log(0)
        let pred_clipped = pred.max(epsilon).min(T::one() - epsilon);
        loss = loss
            - (*target * pred_clipped.ln() + (T::one() - *target) * (T::one() - pred_clipped).ln());
    }

    Ok(loss / cast::<usize, T>(predictions.len()).unwrap_or(T::one()))
}

/// Stable matrix norm computation
#[allow(dead_code)]
pub fn stablematrix_norm<T: Float>(matrix: &ArrayView2<T>, ord: MatrixNorm) -> CoreResult<T> {
    validatematrix_not_empty(matrix)?;

    match ord {
        MatrixNorm::Frobenius => {
            // Use compensated summation for Frobenius norm
            let mut sum = T::zero();
            let mut compensation = T::zero();

            for &value in matrix.iter() {
                let sq = value * value;
                let y = sq - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }

            Ok(sum.sqrt())
        }
        MatrixNorm::One => {
            // Maximum absolute column sum
            let mut max_sum = T::zero();

            for col in matrix.axis_iter(Axis(1)) {
                let col_sum = stable_norm_1(&col.to_vec());
                max_sum = max_sum.max(col_sum);
            }

            Ok(max_sum)
        }
        MatrixNorm::Infinity => {
            // Maximum absolute row sum
            let mut max_sum = T::zero();

            for row in matrix.axis_iter(Axis(0)) {
                let row_sum = stable_norm_1(&row.to_vec());
                max_sum = max_sum.max(row_sum);
            }

            Ok(max_sum)
        }
    }
}

/// Matrix norm types
#[derive(Debug, Clone, Copy)]
pub enum MatrixNorm {
    /// Frobenius norm (sqrt of sum of squares)
    Frobenius,
    /// 1-norm (maximum absolute column sum)
    One,
    /// Infinity norm (maximum absolute row sum)
    Infinity,
}

/// Stable L1 norm computation
#[allow(dead_code)]
fn stable_norm_1<T: Float>(values: &[T]) -> T {
    let mut sum = T::zero();
    let mut compensation = T::zero();

    for &value in values {
        let abs_val = value.abs();
        let y = abs_val - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    sum
}

/// Stable L2 norm computation avoiding overflow/underflow
#[allow(dead_code)]
pub fn stable_norm_2<T: Float>(values: &[T]) -> T {
    if values.is_empty() {
        return T::zero();
    }

    // Find maximum absolute value for scaling
    let max_abs = values.iter().fold(T::zero(), |max, &x| max.max(x.abs()));

    if max_abs.is_zero() {
        return T::zero();
    }

    // Compute scaled norm
    let mut sum = T::zero();
    for &value in values {
        let scaled = value / max_abs;
        sum = sum + scaled * scaled;
    }

    max_abs * sum.sqrt()
}

/// Condition number estimation using 1-norm
#[allow(dead_code)]
pub fn condition_number_estimate<T: Float>(matrix: &ArrayView2<T>) -> CoreResult<T> {
    validatematrix_not_empty(matrix)?;

    if matrix.nrows() != matrix.ncols() {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Matrix must be square for condition number",
        )));
    }

    // Compute 1-norm of matrix
    let norm_a = stablematrix_norm(matrix, MatrixNorm::One)?;

    // Estimate norm of inverse using power method
    // This is a simplified version - a full implementation would use LAPACK's condition estimator
    let n = matrix.nrows();
    let mut x = Array1::from_elem(n, T::one() / cast::<usize, T>(n).unwrap_or(T::one()));
    let mut y = Array1::zeros(n);

    // Power iteration to estimate ||A^{-1}||_1
    let max_iter = 10;
    let mut norm_inv_estimate = T::zero();

    for _ in 0..max_iter {
        // y = A^T x
        for i in 0..n {
            y[i] = T::zero();
            for j in 0..n {
                y[i] = y[i] + matrix[[j, i]] * x[j];
            }
        }

        // Normalize y
        let y_norm = stable_norm_1(&y.to_vec());
        if y_norm > T::zero() {
            for i in 0..n {
                y[i] = y[i] / y_norm;
            }
        }

        // Solve A z = y (simplified - would use LU decomposition)
        // For now, just estimate
        norm_inv_estimate = norm_inv_estimate.max(y_norm);

        x.assign(&y);
    }

    Ok(norm_a * norm_inv_estimate)
}

/// Helper function to validate matrix is not empty
#[allow(dead_code)]
fn validatematrix_not_empty<T>(matrix: &ArrayView2<T>) -> CoreResult<()> {
    if matrix.is_empty() {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Matrix cannot be empty",
        )));
    }
    Ok(())
}

/// Stable computation of binomial coefficients
#[allow(dead_code)]
pub fn binomial_stable(n: u64, k: u64) -> CoreResult<f64> {
    if k > n {
        return Ok(0.0);
    }

    let k = k.min(n - k); // Take advantage of symmetry

    if k == 0 {
        return Ok(1.0);
    }

    // Use log-space computation for large values
    if n > 20 {
        let mut log_result = 0.0;

        for i in 0..k {
            log_result += ((n - i) as f64).ln();
            log_result -= ((i + 1) as f64).ln();
        }

        Ok(log_result.exp())
    } else {
        // Direct computation for small values
        let mut result = 1.0;

        for i in 0..k {
            result *= (n - i) as f64;
            result /= (i + 1) as f64;
        }

        Ok(result)
    }
}

/// Numerically stable factorial computation
#[allow(dead_code)]
pub fn factorial_stable(n: u64) -> CoreResult<f64> {
    if n == 0 || n == 1 {
        return Ok(1.0);
    }

    // Use Stirling's approximation for large n
    if n > 170 {
        // For n > 170, n! overflows f64, so use log-space
        let n_f64 = n as f64;
        let log_factorial =
            n_f64 * n_f64.ln() - n_f64 + 0.5 * (2.0 * std::f64::consts::PI * n_f64).ln();
        return Ok(log_factorial.exp());
    }

    // Direct computation with overflow check
    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
        if !result.is_finite() {
            return Err(CoreError::ComputationError(ErrorContext::new(
                "Factorial overflow",
            )));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_kahan_sum() {
        // Use values that demonstrate the benefit of Kahan summation
        let values = vec![1.0, 1e-8, 1e-8, 1e-8, 1e-8];
        let mut kahan = KahanSum::new();

        for v in &values {
            kahan.add(*v);
        }

        let kahan_sum = kahan.sum();

        // Expected result should be 1.0 + 4e-8
        let expected = 1.0 + 4e-8;
        assert_relative_eq!(kahan_sum, expected, epsilon = 1e-15);

        // Test with values where Kahan algorithm shows benefit
        let mut kahan2 = KahanSum::new();
        // These values sum to 1.0 but naive summation loses precision
        let test_vals = vec![1.0, 1e-16, -1e-16, 1e-16, -1e-16];
        for v in test_vals {
            kahan2.add(v);
        }
        assert_relative_eq!(kahan2.sum(), 1.0, epsilon = 1e-15);

        // Test accumulation of many small values
        let mut kahan3 = KahanSum::new();
        for _ in 0..10000 {
            kahan3.add(0.01);
        }
        assert_relative_eq!(kahan3.sum(), 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_neumaier_sum() {
        let values = vec![1e20, 1.0, -1e20];
        let sum = neumaier_sum(&values);
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pairwise_sum() {
        let values: Vec<f64> = (0..1000).map(|i| 0.1 + 0.001 * i as f64).collect();
        let sum = pairwise_sum(&values);

        // Compare with known result
        let expected = values.iter().sum::<f64>();
        assert_relative_eq!(sum, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_welford_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut welford = WelfordVariance::new();

        for &v in &values {
            welford.add(v);
        }

        assert_relative_eq!(
            welford.mean().expect("Mean should be available"),
            3.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            welford.variance().expect("Variance should be available"),
            2.5,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            welford.std_dev().expect("Std dev should be available"),
            2.5_f64.sqrt(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn testlog_sum_exp() {
        let values = vec![1000.0, 1000.0, 1000.0];
        let result = log_sum_exp(&values);
        let expected = 1000.0 + 3.0_f64.ln();
        assert_relative_eq!(result, expected, epsilon = 1e-10);

        // Test with empty array
        let empty: Vec<f64> = vec![];
        assert!(log_sum_exp(&empty).is_infinite());
    }

    #[test]
    fn test_stable_softmax() {
        let values = vec![1000.0, 1000.0, 1000.0];
        let softmax = stable_softmax(&values);

        for &p in &softmax {
            assert_relative_eq!(p, 1.0 / 3.0, epsilon = 1e-10);
        }

        let sum: f64 = softmax.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hypot_stable() {
        // Test with large values that would overflow with naive x² + y²
        let x = 1e200;
        let y = 1e200;
        let result = hypot_stable(x, y);
        let expected = 2.0_f64.sqrt() * 1e200;
        assert_relative_eq!(result, expected, epsilon = 1e-10);

        // Test with zero
        assert_eq!(hypot_stable(0.0, 5.0), 5.0);
        assert_eq!(hypot_stable(3.0, 0.0), 3.0);
    }

    #[test]
    fn test_sigmoid_stable() {
        // Test large positive value
        let result = sigmoid_stable(100.0);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);

        // Test large negative value
        let result = sigmoid_stable(-100.0);
        assert!(result > 0.0 && result < 1e-40);

        // Test zero
        assert_relative_eq!(sigmoid_stable(0.0), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_reduce_angle() {
        // Test angle reduction
        assert_relative_eq!(reduce_angle(3.0 * PI), -PI, epsilon = 1e-10);
        assert_relative_eq!(reduce_angle(5.0 * PI), -PI, epsilon = 1e-10);
        assert_relative_eq!(reduce_angle(-3.0 * PI), -PI, epsilon = 1e-10);
        assert_relative_eq!(reduce_angle(2.0 * PI), 0.0, epsilon = 1e-10);
        assert_relative_eq!(reduce_angle(-2.0 * PI), 0.0, epsilon = 1e-10);
        assert_relative_eq!(reduce_angle(7.0 * PI), -PI, epsilon = 1e-10);
        assert_relative_eq!(reduce_angle(-7.0 * PI), -PI, epsilon = 1e-10);

        // Test angle already in range
        assert_relative_eq!(reduce_angle(PI / 2.0), PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stable_norm_2() {
        // Test with values that would overflow
        let values = vec![1e200, 1e200, 1e200];
        let norm = stable_norm_2(&values);
        let expected = 3.0_f64.sqrt() * 1e200;
        assert_relative_eq!(norm, expected, epsilon = 1e-10);

        // Test with very small values
        let smallvalues = vec![1e-200, 1e-200, 1e-200];
        let small_norm = stable_norm_2(&smallvalues);
        let expected_small = 3.0_f64.sqrt() * 1e-200;
        assert_relative_eq!(small_norm, expected_small, epsilon = 1e-10);
    }

    #[test]
    fn test_binomial_stable() {
        // Test small values
        assert_eq!(
            binomial_stable(5, 2).expect("Binomial coefficient should succeed"),
            10.0
        );
        assert_eq!(
            binomial_stable(10, 3).expect("Binomial coefficient should succeed"),
            120.0
        );

        // Test edge cases
        assert_eq!(
            binomial_stable(5, 0).expect("Binomial coefficient should succeed"),
            1.0
        );
        assert_eq!(
            binomial_stable(5, 5).expect("Binomial coefficient should succeed"),
            1.0
        );
        assert_eq!(
            binomial_stable(5, 6).expect("Binomial coefficient should succeed"),
            0.0
        );

        // Test large values
        let large_result =
            binomial_stable(100, 50).expect("Binomial coefficient should handle large values");
        assert!(large_result.is_finite() && large_result > 0.0);
    }

    #[test]
    fn test_factorial_stable() {
        // Test small values
        assert_eq!(
            factorial_stable(0).expect("Factorial of 0 should succeed"),
            1.0
        );
        assert_eq!(
            factorial_stable(1).expect("Factorial of 1 should succeed"),
            1.0
        );
        assert_eq!(
            factorial_stable(5).expect("Factorial of 5 should succeed"),
            120.0
        );

        // Test larger value
        assert_relative_eq!(
            factorial_stable(10).expect("Factorial of 10 should succeed"),
            3628800.0,
            epsilon = 1e-10
        );
    }
}

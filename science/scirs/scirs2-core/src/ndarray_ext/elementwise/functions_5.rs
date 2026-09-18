//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::numeric::Float;
use crate::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use ::ndarray::{Array1, ArrayView1};

/// SIMD-accelerated inverse square root: 1/sqrt(x)
///
/// Computes the reciprocal of the square root for each element.
/// This operation is fundamental in graphics and physics simulations.
///
/// # Arguments
/// * `a` - Input values (should be positive)
///
/// # Returns
/// Array of 1/sqrt(x) values
/// - Returns INFINITY for x = 0
/// - Returns NaN for x < 0
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::rsqrt_simd;
///
/// let x = array![1.0_f64, 4.0, 9.0, 16.0];
///
/// let result = rsqrt_simd::<f64>(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-14);     // 1/sqrt(1) = 1
/// assert!((result[1] - 0.5).abs() < 1e-14);     // 1/sqrt(4) = 0.5
/// assert!((result[2] - 1.0/3.0).abs() < 1e-14); // 1/sqrt(9) = 1/3
/// assert!((result[3] - 0.25).abs() < 1e-14);    // 1/sqrt(16) = 0.25
/// ```
///
/// # Use Cases
/// - Vector normalization: v_normalized = v * rsqrt(dot(v, v))
/// - Quaternion normalization in 3D graphics
/// - Inverse distance weighting
/// - Fast approximate physics simulations
pub fn rsqrt_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_rsqrt(a);
    }
    F::simd_rsqrt(a)
}
/// SIMD-accelerated simultaneous sin and cos computation
///
/// Computes both sine and cosine of each element in a single pass.
/// This is more efficient than calling sin and cos separately because
/// many implementations can compute both with minimal additional overhead.
///
/// # Arguments
/// * `a` - Input angles in radians
///
/// # Returns
/// Tuple of (sin_array, cos_array)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::sincos_simd;
/// use std::f64::consts::PI;
///
/// let x = array![0.0_f64, PI/6.0, PI/4.0, PI/2.0];
///
/// let (sin_result, cos_result) = sincos_simd::<f64>(&x.view());
/// assert!((sin_result[0] - 0.0).abs() < 1e-14);        // sin(0) = 0
/// assert!((cos_result[0] - 1.0).abs() < 1e-14);        // cos(0) = 1
/// assert!((sin_result[1] - 0.5).abs() < 1e-14);        // sin(π/6) = 0.5
/// assert!((sin_result[3] - 1.0).abs() < 1e-14);        // sin(π/2) = 1
/// assert!(cos_result[3].abs() < 1e-14);                // cos(π/2) ≈ 0
/// ```
///
/// # Use Cases
/// - Rotation matrices (need both sin and cos)
/// - Complex number operations: e^(iθ) = cos(θ) + i·sin(θ)
/// - Fourier transform calculations
/// - Wave simulations
/// - Animation and interpolation (circular motion)
pub fn sincos_simd<F>(a: &ArrayView1<F>) -> (Array1<F>, Array1<F>)
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return (Array1::zeros(0), Array1::zeros(0));
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_sincos(a);
    }
    F::simd_sincos(a)
}
/// SIMD-accelerated numerically stable exp(x) - 1
///
/// Computes exp(x) - 1 accurately for small x values where the direct
/// calculation `exp(x) - 1` would suffer from catastrophic cancellation.
/// For |x| < 1e-10, the result is approximately x (Taylor expansion).
///
/// # Arguments
/// * `a` - Input values
///
/// # Returns
/// Array of exp(x) - 1 values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::expm1_simd;
///
/// let x = array![0.0_f64, 1e-15, 1.0, -1.0];
///
/// let result = expm1_simd::<f64>(&x.view());
/// // exp(0) - 1 = 0
/// assert!((result[0] - 0.0).abs() < 1e-14);
/// // For small x: exp(x) - 1 ≈ x
/// assert!((result[1] - 1e-15).abs() < 1e-29);
/// // exp(1) - 1 ≈ 1.718
/// assert!((result[2] - (1.0_f64.exp() - 1.0)).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Financial calculations (compound interest for small rates)
/// - Numerical integration (avoiding cancellation errors)
/// - Statistical distributions (Poisson, exponential)
/// - Machine learning (softplus: log(1 + exp(x)))
pub fn expm1_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_expm1(a);
    }
    F::simd_expm1(a)
}
/// SIMD-accelerated numerically stable ln(1 + x)
///
/// Computes ln(1 + x) accurately for small x values where the direct
/// calculation `(1 + x).ln()` would suffer from catastrophic cancellation.
/// For |x| < 1e-10, the result is approximately x - x²/2 (Taylor expansion).
///
/// # Arguments
/// * `a` - Input values (should be > -1)
///
/// # Returns
/// Array of ln(1 + x) values
/// - Returns -∞ for x = -1
/// - Returns NaN for x < -1
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::log1p_simd;
///
/// let x = array![0.0_f64, 1e-15, 1.0, -0.5];
///
/// let result = log1p_simd::<f64>(&x.view());
/// // ln(1 + 0) = 0
/// assert!((result[0] - 0.0).abs() < 1e-14);
/// // For small x: ln(1 + x) ≈ x
/// assert!((result[1] - 1e-15).abs() < 1e-29);
/// // ln(2) ≈ 0.693
/// assert!((result[2] - 2.0_f64.ln()).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Log-probability calculations (log(1 - p) for small p)
/// - Numerical integration
/// - Statistical distributions
/// - Machine learning (binary cross-entropy: -y·log(p) - (1-y)·log(1-p))
pub fn log1p_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_log1p(a);
    }
    F::simd_log1p(a)
}
/// SIMD-accelerated element-wise clipping (clamping)
///
/// Clips (limits) each element to be within [min_val, max_val].
/// Values below min_val become min_val, values above max_val become max_val.
///
/// # Arguments
/// * `a` - Input array
/// * `min_val` - Minimum value (lower bound)
/// * `max_val` - Maximum value (upper bound)
///
/// # Returns
/// Array with values clipped to [min_val, max_val]
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::clip_simd;
///
/// let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0];
///
/// let result = clip_simd::<f64>(&x.view(), 0.0, 1.0);
/// assert!((result[0] - 0.0).abs() < 1e-14);  // -2 clipped to 0
/// assert!((result[2] - 0.0).abs() < 1e-14);  // 0 unchanged
/// assert!((result[3] - 1.0).abs() < 1e-14);  // 1 unchanged
/// assert!((result[5] - 1.0).abs() < 1e-14);  // 3 clipped to 1
/// ```
///
/// # Use Cases
/// - Gradient clipping in neural networks
/// - Image processing (pixel value clamping)
/// - Bounded optimization
/// - Data normalization preprocessing
pub fn clip_simd<F>(a: &ArrayView1<F>, min_val: F, max_val: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_clip(a, min_val, max_val);
    }
    F::simd_clip(a, min_val, max_val)
}
/// SIMD-accelerated cumulative sum (prefix sum)
///
/// Computes the cumulative sum of elements. `result[i] = sum(a[0..=i])`
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Array of cumulative sums
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::cumsum_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
///
/// let result = cumsum_simd::<f64>(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-14);   // 1
/// assert!((result[1] - 3.0).abs() < 1e-14);   // 1+2
/// assert!((result[2] - 6.0).abs() < 1e-14);   // 1+2+3
/// assert!((result[3] - 10.0).abs() < 1e-14);  // 1+2+3+4
/// assert!((result[4] - 15.0).abs() < 1e-14);  // 1+2+3+4+5
/// ```
///
/// # Use Cases
/// - Computing CDF from PDF
/// - Running totals and statistics
/// - Parallel prefix algorithms
/// - Integral image computation
pub fn cumsum_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_cumsum(a);
    }
    F::simd_cumsum(a)
}
/// SIMD-accelerated cumulative product
///
/// Computes the cumulative product of elements. `result[i] = product(a[0..=i])`
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Array of cumulative products
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::cumprod_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0];
///
/// let result = cumprod_simd::<f64>(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-14);   // 1
/// assert!((result[1] - 2.0).abs() < 1e-14);   // 1*2
/// assert!((result[2] - 6.0).abs() < 1e-14);   // 1*2*3
/// assert!((result[3] - 24.0).abs() < 1e-14);  // 1*2*3*4 = 4!
/// ```
///
/// # Use Cases
/// - Computing factorials and permutations
/// - Product of survival probabilities
/// - Chain rule in automatic differentiation
/// - Compound interest calculations
pub fn cumprod_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_cumprod(a);
    }
    F::simd_cumprod(a)
}
/// SIMD-accelerated first-order difference
///
/// Computes the first-order finite difference: `result[i] = a[i+1] - a[i]`
/// The output has length n-1 for input of length n.
///
/// # Arguments
/// * `a` - Input array (length >= 2)
///
/// # Returns
/// Array of differences (length n-1)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::diff_simd;
///
/// let x = array![1.0_f64, 3.0, 6.0, 10.0, 15.0];
///
/// let result = diff_simd::<f64>(&x.view());
/// assert_eq!(result.len(), 4);  // n-1 elements
/// assert!((result[0] - 2.0).abs() < 1e-14);  // 3-1
/// assert!((result[1] - 3.0).abs() < 1e-14);  // 6-3
/// assert!((result[2] - 4.0).abs() < 1e-14);  // 10-6
/// assert!((result[3] - 5.0).abs() < 1e-14);  // 15-10
/// ```
///
/// # Use Cases
/// - Numerical differentiation
/// - Detecting changes in time series
/// - Computing gradients
/// - Edge detection in signal processing
pub fn diff_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.len() < 2 {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_diff(a);
    }
    F::simd_diff(a)
}
/// SIMD-accelerated variance computation
///
/// Computes the sample variance: Var(x) = sum((x - mean)²) / (n-1)
/// Uses Bessel's correction for unbiased estimation.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Sample variance of the array
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::variance_simd;
///
/// let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let var = variance_simd::<f64>(&x.view());
/// // Sample variance: sum of squared deviations = 32, n = 8
/// // var = 32 / (8-1) = 32/7 ≈ 4.571
/// let expected = 32.0 / 7.0;
/// assert!((var - expected).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Statistical analysis
/// - Quality control (process capability)
/// - Risk assessment (financial variance)
/// - Feature scaling (standardization)
pub fn variance_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }
    F::simd_variance(a)
}
/// SIMD-accelerated standard deviation computation
///
/// Computes the sample standard deviation: std = sqrt(sample_variance)
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Sample standard deviation
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::std_simd;
///
/// let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let s = std_simd::<f64>(&x.view());
/// // Sample std = sqrt(sample variance) = sqrt(32/7) ≈ 2.138
/// let expected = (32.0_f64 / 7.0).sqrt();
/// assert!((s - expected).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Volatility measurement in finance
/// - Error analysis in experiments
/// - Gaussian distribution parameters
/// - Confidence interval calculations
pub fn std_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }
    F::simd_std(a)
}
/// SIMD-accelerated array sum
///
/// Computes the sum of all elements in the array.
/// Uses Kahan summation for improved numerical accuracy.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Sum of all elements
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sum_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let s = sum_simd::<f64>(&x.view());
/// assert!((s - 15.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Computing totals
/// - Mean calculation component
/// - Probability mass functions
/// - Integration approximation
pub fn sum_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }
    F::simd_sum(a)
}
/// SIMD-accelerated array mean
///
/// Computes the arithmetic mean: mean = sum(x) / n
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Arithmetic mean of elements
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::mean_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let m = mean_simd::<f64>(&x.view());
/// assert!((m - 3.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Central tendency measurement
/// - Batch normalization (compute running mean)
/// - Signal averaging
/// - Expected value estimation
pub fn mean_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }
    F::simd_mean(a)
}
/// SIMD-accelerated weighted sum
///
/// Computes sum(values * weights).
///
/// # Arguments
/// * `values` - Values array
/// * `weights` - Weights array (same length as values)
///
/// # Returns
/// Weighted sum
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::weighted_sum_simd;
///
/// let values = array![10.0_f64, 20.0, 30.0];
/// let weights = array![0.2_f64, 0.3, 0.5];
/// let ws = weighted_sum_simd::<f64>(&values.view(), &weights.view());
/// // 10*0.2 + 20*0.3 + 30*0.5 = 2 + 6 + 15 = 23
/// assert!((ws - 23.0).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Portfolio valuation
/// - Weighted regression
/// - Attention mechanism scores
/// - Signal filtering
pub fn weighted_sum_simd<F>(values: &ArrayView1<F>, weights: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if values.is_empty() || weights.is_empty() {
        return F::zero();
    }
    F::simd_weighted_sum(values, weights)
}
/// SIMD-accelerated weighted mean
///
/// Computes sum(values * weights) / sum(weights).
///
/// # Arguments
/// * `values` - Values array
/// * `weights` - Weights array (same length as values)
///
/// # Returns
/// Weighted mean
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::weighted_mean_simd;
///
/// let values = array![10.0_f64, 20.0, 30.0];
/// let weights = array![1.0_f64, 2.0, 2.0];
/// let wm = weighted_mean_simd::<f64>(&values.view(), &weights.view());
/// // (10*1 + 20*2 + 30*2) / (1+2+2) = (10+40+60)/5 = 22
/// assert!((wm - 22.0).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Grade point average calculation
/// - Portfolio weighted return
/// - Weighted sentiment analysis
/// - Attention-weighted representations
pub fn weighted_mean_simd<F>(values: &ArrayView1<F>, weights: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if values.is_empty() || weights.is_empty() {
        return F::zero();
    }
    F::simd_weighted_mean(values, weights)
}
/// SIMD-accelerated maximum element
///
/// Finds the maximum value in the array.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Maximum element value
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::max_element_simd;
///
/// let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
/// let max = max_element_simd::<f64>(&x.view());
/// assert!((max - 9.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Finding peak values
/// - Range calculation
/// - Normalization (min-max scaling)
/// - Softmax numerator stability
pub fn max_element_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::neg_infinity();
    }
    F::simd_max_element(a)
}
/// SIMD-accelerated minimum element
///
/// Finds the minimum value in the array.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Minimum element value
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::min_element_simd;
///
/// let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
/// let min = min_element_simd::<f64>(&x.view());
/// assert!((min - 1.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Finding minimum values
/// - Range calculation
/// - Clipping lower bounds
/// - Outlier detection
pub fn min_element_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::infinity();
    }
    F::simd_min_element(a)
}
/// SIMD-accelerated argmax (index of maximum)
///
/// Returns the index of the maximum element.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Some(index) of maximum element, or None if array is empty
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::argmax_simd;
///
/// let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
/// let idx = argmax_simd::<f64>(&x.view());
/// assert_eq!(idx, Some(5));  // x[5] = 9.0 is the maximum
/// ```
///
/// # Use Cases
/// - Classification prediction (class with highest probability)
/// - Finding optimal parameters
/// - Greedy selection algorithms
/// - Winner-take-all networks
pub fn argmax_simd<F>(a: &ArrayView1<F>) -> Option<usize>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return None;
    }
    F::simd_argmax(a)
}
/// SIMD-accelerated argmin (index of minimum)
///
/// Returns the index of the minimum element.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Some(index) of minimum element, or None if array is empty
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::argmin_simd;
///
/// let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
/// let idx = argmin_simd::<f64>(&x.view());
/// assert_eq!(idx, Some(1));  // x[1] = 1.0 is the minimum (first occurrence)
/// ```
///
/// # Use Cases
/// - Finding nearest neighbors
/// - Minimum loss selection
/// - Optimal path finding
/// - Resource allocation
pub fn argmin_simd<F>(a: &ArrayView1<F>) -> Option<usize>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return None;
    }
    F::simd_argmin(a)
}
/// SIMD-accelerated sum of squares
///
/// Computes sum(x²) efficiently.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Sum of squared elements
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sum_squares_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0];
/// let ss = sum_squares_simd::<f64>(&x.view());
/// // 1 + 4 + 9 + 16 = 30
/// assert!((ss - 30.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - L2 norm calculation (sqrt of sum of squares)
/// - Mean squared error
/// - Variance computation
/// - Energy of signal
pub fn sum_squares_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }
    F::simd_sum_squares(a)
}
/// SIMD-accelerated sum of cubes
///
/// Computes sum(x³) efficiently.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Sum of cubed elements
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sum_cubes_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0];
/// let sc = sum_cubes_simd::<f64>(&x.view());
/// // 1 + 8 + 27 = 36
/// assert!((sc - 36.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Higher-order moment calculation
/// - Skewness estimation
/// - Cubic interpolation
/// - Power sum computation
pub fn sum_cubes_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }
    F::simd_sum_cubes(a)
}
/// SIMD-accelerated log-sum-exp
///
/// Computes log(sum(exp(x))) in a numerically stable way.
/// Uses the identity: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Log of sum of exponentials
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::log_sum_exp_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0];
/// let lse = log_sum_exp_simd::<f64>(&x.view());
/// // log(e^1 + e^2 + e^3) ≈ 3.407
/// let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
/// assert!((lse - expected).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Softmax denominator
/// - Log-probability computations
/// - Partition function in statistical mechanics
/// - Evidence lower bound (ELBO) in VAEs
pub fn log_sum_exp_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::neg_infinity();
    }
    F::simd_log_sum_exp(a)
}
/// SIMD-accelerated L2 norm (Euclidean norm)
///
/// Computes ||x||₂ = sqrt(sum(x²))
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// L2 (Euclidean) norm
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::norm_simd;
///
/// let x = array![3.0_f64, 4.0];
/// let n = norm_simd::<f64>(&x.view());
/// // sqrt(9 + 16) = 5
/// assert!((n - 5.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Vector magnitude
/// - Normalization preprocessing
/// - Regularization penalty
/// - Gradient clipping
pub fn norm_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }
    F::simd_norm(a)
}
/// SIMD-accelerated L1 norm (Manhattan norm)
///
/// Computes ||x||₁ = sum(|x|)
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// L1 (Manhattan) norm
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::norm_l1_simd;
///
/// let x = array![1.0_f64, -2.0, 3.0, -4.0];
/// let n = norm_l1_simd::<f64>(&x.view());
/// // |1| + |-2| + |3| + |-4| = 10
/// assert!((n - 10.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Sparse regularization (LASSO)
/// - Robust statistics
/// - Compressed sensing
/// - Feature selection
pub fn norm_l1_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }
    F::simd_norm_l1(a)
}
/// SIMD-accelerated L-infinity norm (Chebyshev/max norm)
///
/// Computes ||x||∞ = max(|x|)
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// L-infinity (max) norm
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::norm_linf_simd;
///
/// let x = array![1.0_f64, -5.0, 3.0, -2.0];
/// let n = norm_linf_simd::<f64>(&x.view());
/// // max(|1|, |-5|, |3|, |-2|) = 5
/// assert!((n - 5.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Gradient clipping by max norm
/// - Adversarial robustness (L∞ perturbations)
/// - Convergence criteria
/// - Worst-case analysis
pub fn norm_linf_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }
    F::simd_norm_linf(a)
}
/// SIMD-accelerated Euclidean distance
///
/// Computes ||a - b||₂ = sqrt(sum((a - b)²))
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Euclidean distance between vectors
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::distance_euclidean_simd;
///
/// let a = array![0.0_f64, 0.0];
/// let b = array![3.0_f64, 4.0];
/// let d = distance_euclidean_simd::<f64>(&a.view(), &b.view());
/// assert!((d - 5.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - K-means clustering
/// - Nearest neighbor search
/// - Physical distance measurement
/// - Embedding similarity
pub fn distance_euclidean_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }
    F::simd_distance_euclidean(a, b)
}
/// SIMD-accelerated Manhattan distance
///
/// Computes sum(|a - b|)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Manhattan (L1) distance between vectors
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::distance_manhattan_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![4.0_f64, 0.0, 3.0];
/// let d = distance_manhattan_simd::<f64>(&a.view(), &b.view());
/// // |1-4| + |2-0| + |3-3| = 3 + 2 + 0 = 5
/// assert!((d - 5.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Grid-based path finding
/// - Robust distance metric
/// - Taxicab geometry
/// - Feature comparison
pub fn distance_manhattan_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }
    F::simd_distance_manhattan(a, b)
}

//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::numeric::Float;
use crate::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use ::ndarray::{Array1, ArrayView1};

/// Compute the base-2 logarithm of each element (SIMD-accelerated).
///
/// Computes log₂(x) for each element, where log₂(x) = ln(x) / ln(2).
///
/// # Arguments
///
/// * `x` - Input 1D array with positive values
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the base-2
/// logarithm. Returns NaN for x ≤ 0, and -∞ for x = 0.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Domain: (0, ∞)
/// - Range: (-∞, ∞)
/// - log₂(1) = 0
/// - log₂(2) = 1
/// - log₂(2ⁿ) = n
/// - log₂(x * y) = log₂(x) + log₂(y)
/// - log₂(x / y) = log₂(x) - log₂(y)
/// - log₂(xⁿ) = n * log₂(x)
/// - Returns NaN for x ≤ 0
/// - Returns -∞ for x = 0
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::log2_simd;
///
/// let x = array![1.0_f64, 2.0, 4.0, 8.0, 16.0];
/// let result = log2_simd(&x.view());
/// // Result: [0.0, 1.0, 2.0, 3.0, 4.0]
/// ```
///
/// # Applications
///
/// - **Information Theory**: Shannon entropy (bits), channel capacity
/// - **Computer Science**: Binary tree depth, algorithm complexity analysis
/// - **Machine Learning**: Decision trees, information gain
/// - **Signal Processing**: Octave scales, frequency resolution
/// - **Data Compression**: Optimal coding, Huffman coding
/// - **Cryptography**: Key size representation, security levels
/// - **Scientific Computing**: Binary logarithmic plots
/// - **Digital Systems**: Bit depth calculations
///
/// # See Also
///
/// - `ln_simd`: Natural logarithm (base e)
/// - `log10_simd`: Base-10 logarithm
/// - `exp_simd`: Exponential function (inverse of ln)
pub fn log2_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_log2(x);
    }
    x.mapv(|val| val.log2())
}
/// Clamp each element to a specified range [min, max] (SIMD-accelerated).
///
/// Constrains each element x to satisfy min ≤ x ≤ max. Values below min are set
/// to min, values above max are set to max, and values within range are unchanged.
///
/// # Arguments
///
/// * `x` - Input 1D array
/// * `min` - Minimum value (lower bound)
/// * `max` - Maximum value (upper bound)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is clamped to [min, max].
///
/// # Panics
///
/// Panics if min > max (invalid range).
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - clamp(x, min, max) = max(min, min(x, max))
/// - clamp(x, min, max) ∈ [min, max] for all x
/// - clamp(min, min, max) = min
/// - clamp(max, min, max) = max
/// - clamp is idempotent: clamp(clamp(x, a, b), a, b) = clamp(x, a, b)
/// - clamp preserves monotonicity: if x₁ ≤ x₂, then clamp(x₁) ≤ clamp(x₂)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::clamp_simd;
///
/// let x = array![-1.0_f64, 0.5, 1.5, 2.5, 3.5];
/// let result = clamp_simd(&x.view(), 0.0, 2.0);
/// // Result: [0.0, 0.5, 1.5, 2.0, 2.0]
/// ```
///
/// # Applications
///
/// - **Image Processing**: Pixel value normalization (0-255, 0.0-1.0)
/// - **Neural Networks**: Gradient clipping, activation bounding
/// - **Computer Vision**: Color space conversions, contrast limiting
/// - **Signal Processing**: Dynamic range compression, amplitude limiting
/// - **Data Normalization**: Feature scaling, outlier handling
/// - **Numerical Stability**: Preventing overflow/underflow in computations
/// - **Game Development**: Velocity limiting, position constraints
/// - **Robotics**: Joint angle limits, actuator bounds
/// - **Audio Processing**: Volume limiting, signal clipping prevention
/// - **Machine Learning**: Learning rate bounds, weight constraints
///
/// # See Also
///
/// - `abs_simd`: Absolute value (clamping negative values to positive)
/// - `floor_simd`: Lower bound only (ceiling at integers)
/// - `ceil_simd`: Upper bound only (floor at integers)
pub fn clamp_simd<F>(x: &ArrayView1<F>, min: F, max: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps + std::fmt::Debug,
{
    assert!(
        min <= max,
        "clamp_simd: min ({:?}) must be <= max ({:?})",
        min,
        max
    );
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_clamp(x, min, max);
    }
    x.mapv(|val| val.clamp(min, max))
}
/// Compute 2^x for each element (SIMD-accelerated).
///
/// Computes the base-2 exponential function 2^x for each element.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is 2 raised to the
/// corresponding input power.
///
/// # Performance
///
/// Uses the identity 2^x = exp(x * ln(2)) with SIMD-accelerated exp and scalar multiply.
///
/// # Mathematical Properties
///
/// - 2^0 = 1
/// - 2^1 = 2
/// - 2^(-1) = 0.5
/// - 2^n for integer n is exact for small n
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::exp2_simd;
///
/// let x = array![0.0_f64, 1.0, 2.0, 3.0, -1.0];
/// let result = exp2_simd(&x.view());
/// // Result: [1.0, 2.0, 4.0, 8.0, 0.5]
/// ```
///
/// # Applications
///
/// - **Audio Processing**: Octave/semitone calculations
/// - **Computer Graphics**: Level of detail (LOD) calculations
/// - **Floating-Point**: Exponent manipulation
pub fn exp2_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_exp2(x);
    }
    F::simd_exp2(x)
}
/// Compute the cube root of each element (SIMD-accelerated).
///
/// Computes cbrt(x) = x^(1/3) for each element, correctly handling negative values.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the cube root.
///
/// # Mathematical Properties
///
/// - cbrt(x^3) = x (exact inverse of cubing)
/// - cbrt(-x) = -cbrt(x) (handles negative numbers)
/// - cbrt(0) = 0
/// - cbrt(1) = 1, cbrt(8) = 2, cbrt(27) = 3
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::cbrt_simd;
///
/// let x = array![0.0_f64, 1.0, 8.0, 27.0, -8.0];
/// let result = cbrt_simd(&x.view());
/// // Result: [0.0, 1.0, 2.0, 3.0, -2.0]
/// ```
///
/// # Applications
///
/// - **Statistics**: Transforming skewed data
/// - **Physics**: Volume calculations from cubic dimensions
/// - **Numerical Analysis**: Root-finding algorithms
pub fn cbrt_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_cbrt(x);
    }
    x.mapv(|val| val.cbrt())
}
/// Compute ln(1+x) for each element (SIMD-accelerated, numerically stable).
///
/// Computes the natural logarithm of (1+x) with improved accuracy for small x values
/// where direct computation of ln(1+x) would lose precision.
///
/// # Arguments
///
/// * `x` - Input 1D array with values > -1
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is ln(1+x).
///
/// # Mathematical Properties
///
/// - ln_1p(0) = 0
/// - ln_1p(x) ≈ x for |x| << 1 (Taylor series first term)
/// - More accurate than ln(1+x) when |x| is small
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::ln_1p_simd;
///
/// let x = array![0.0_f64, 1.0, 1e-15, -0.5];
/// let result = ln_1p_simd(&x.view());
/// // Result: [0.0, ln(2), ≈1e-15, -ln(2)]
/// ```
///
/// # Applications
///
/// - **Finance**: Continuous compound interest rates
/// - **Statistics**: Log-likelihood computations
/// - **Numerical Analysis**: Avoiding catastrophic cancellation
pub fn ln_1p_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_ln_1p(x);
    }
    x.mapv(|val| val.ln_1p())
}
/// Compute exp(x)-1 for each element (SIMD-accelerated, numerically stable).
///
/// Computes e^x - 1 with improved accuracy for small x values where direct
/// computation would lose precision.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is exp(x)-1.
///
/// # Mathematical Properties
///
/// - exp_m1(0) = 0
/// - exp_m1(x) ≈ x for |x| << 1 (Taylor series first term)
/// - exp_m1(-x) = -exp_m1(x) / (1 + exp_m1(x))
/// - More accurate than exp(x)-1 when |x| is small
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::exp_m1_simd;
///
/// let x = array![0.0_f64, 1.0, 1e-15, -1.0];
/// let result = exp_m1_simd(&x.view());
/// // Result: [0.0, e-1, ≈1e-15, 1/e - 1]
/// ```
///
/// # Applications
///
/// - **Finance**: Continuous compounding calculations
/// - **Physics**: Small perturbation calculations
/// - **Numerical Analysis**: Avoiding catastrophic cancellation
pub fn exp_m1_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_exp_m1(x);
    }
    x.mapv(|val| val.exp_m1())
}
/// Convert degrees to radians for each element (SIMD-accelerated).
///
/// Computes x * π / 180 for each element, converting angle measurements
/// from degrees to radians.
///
/// # Arguments
///
/// * `x` - Input 1D array of angles in degrees
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the
/// corresponding angle in radians.
///
/// # Performance
///
/// Uses SIMD scalar multiplication for optimal performance.
///
/// # Mathematical Properties
///
/// - to_radians(0) = 0
/// - to_radians(90) = π/2
/// - to_radians(180) = π
/// - to_radians(360) = 2π
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::to_radians_simd;
///
/// let degrees = array![0.0, 90.0, 180.0, 360.0];
/// let radians = to_radians_simd(&degrees.view());
/// // Result: [0.0, π/2, π, 2π]
/// ```
///
/// # Applications
///
/// - **Trigonometry**: Converting human-readable angles to radian form
/// - **Graphics**: Rotation transformations
/// - **Physics**: Angular velocity and position calculations
/// - **Navigation**: Bearing and heading conversions
pub fn to_radians_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_to_radians(x);
    }
    F::simd_to_radians(x)
}
/// Convert radians to degrees for each element (SIMD-accelerated).
///
/// Computes x * 180 / π for each element, converting angle measurements
/// from radians to degrees.
///
/// # Arguments
///
/// * `x` - Input 1D array of angles in radians
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the
/// corresponding angle in degrees.
///
/// # Performance
///
/// Uses SIMD scalar multiplication for optimal performance.
///
/// # Mathematical Properties
///
/// - to_degrees(0) = 0
/// - to_degrees(π/2) = 90
/// - to_degrees(π) = 180
/// - to_degrees(2π) = 360
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::to_degrees_simd;
/// use std::f64::consts::PI;
///
/// let radians = array![0.0, PI/2.0, PI, 2.0*PI];
/// let degrees = to_degrees_simd(&radians.view());
/// // Result: [0.0, 90.0, 180.0, 360.0]
/// ```
///
/// # Applications
///
/// - **Trigonometry**: Converting calculation results to human-readable form
/// - **Graphics**: Displaying rotation angles
/// - **Physics**: Reporting angular measurements
/// - **Navigation**: Displaying compass headings
pub fn to_degrees_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_to_degrees(x);
    }
    F::simd_to_degrees(x)
}
/// Computes the element-wise digamma function ψ(x) = d/dx ln(Γ(x)) using SIMD acceleration.
///
/// The digamma function is the logarithmic derivative of the gamma function.
/// It is essential for:
/// - Bayesian variational inference (Dirichlet, Beta priors)
/// - Maximum likelihood estimation for gamma distributions
/// - Statistical special function computations
///
/// # Implementation Details
///
/// The implementation uses three techniques for high accuracy:
/// 1. **Reflection formula**: For x < 0.5, uses ψ(1-x) - π/tan(πx)
/// 2. **Recurrence relation**: For small x, uses ψ(x+1) = ψ(x) + 1/x
/// 3. **Asymptotic expansion**: For large x, uses Bernoulli number series
///
/// # Arguments
///
/// * `x` - Input array of values. Should avoid non-positive integers where digamma has poles.
///
/// # Returns
///
/// Array of ψ(x) values, same shape as input.
///
/// # Mathematical Properties
///
/// - ψ(1) = -γ (Euler-Mascheroni constant ≈ -0.5772)
/// - ψ(n) = -γ + Σ(k=1 to n-1) 1/k for positive integers
/// - ψ(x+1) = ψ(x) + 1/x
/// - ψ(1-x) - ψ(x) = π·cot(πx)
///
/// # Example
///
/// ```ignore
/// use scirs2_core::ndarray_ext::elementwise::digamma_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0];
/// let result = digamma_simd(&x.view());
/// // ψ(1) ≈ -0.5772, ψ(2) ≈ 0.4228, ψ(3) ≈ 0.9228, ψ(4) ≈ 1.2561
/// ```
pub fn digamma_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_digamma(x);
    }
    F::simd_digamma(x)
}
/// Computes the element-wise trigamma function ψ'(x) = d²/dx² ln(Γ(x)) using SIMD acceleration.
///
/// The trigamma function is the second derivative of the log-gamma function (first derivative
/// of digamma). It is critical for:
/// - Fisher information computation in Bayesian inference
/// - Variance of gamma and beta distribution parameters
/// - Maximum likelihood estimation for gamma distributions
/// - Sensitivity analysis in Bayesian variational inference
///
/// # Implementation Details
///
/// The implementation uses three techniques for high accuracy:
/// 1. **Reflection formula**: For x < 0, uses ψ'(1-x) + ψ'(x) = π²/sin²(πx)
/// 2. **Recurrence relation**: For small x, uses ψ'(x+1) = ψ'(x) - 1/x²
/// 3. **Asymptotic expansion**: For large x, uses ψ'(x) ≈ 1/x + 1/(2x²) + B₂/x³ - B₄/x⁵ + ...
///
/// # Arguments
///
/// * `x` - Input array of values. Should avoid non-positive integers where trigamma has poles.
///
/// # Returns
///
/// Array of ψ'(x) values, same shape as input.
///
/// # Mathematical Properties
///
/// - ψ'(1) = π²/6 ≈ 1.6449340668 (Basel problem)
/// - ψ'(n) = π²/6 - Σ(k=1 to n-1) 1/k² for positive integers
/// - ψ'(x+1) = ψ'(x) - 1/x²
/// - For large x: ψ'(x) → 1/x
///
/// # Example
///
/// ```ignore
/// use scirs2_core::ndarray_ext::elementwise::trigamma_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0];
/// let result = trigamma_simd(&x.view());
/// // ψ'(1) ≈ 1.6449, ψ'(2) ≈ 0.6449, ψ'(3) ≈ 0.3949, ψ'(4) ≈ 0.2838
/// ```
pub fn trigamma_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_trigamma(x);
    }
    F::simd_trigamma(x)
}
/// Computes the element-wise log-gamma function ln(Γ(x)) using SIMD acceleration.
///
/// The log-gamma function is more numerically stable than computing `gamma(x).ln()`
/// since it avoids overflow for large arguments and handles the full range of inputs.
/// This function is extensively used in:
/// - Statistical distributions (gamma, beta, binomial, Poisson)
/// - Bayesian inference (prior/posterior computations)
/// - Maximum likelihood estimation
/// - Combinatorics (log of binomial coefficients)
///
/// # Implementation Details
///
/// Uses the Lanczos approximation with reflection formula:
/// - For x >= 0.5: ln(Γ(z)) = ln(√(2π)) + (z-0.5)·ln(t) - t + ln(sum) where t = z + g - 0.5
/// - For x < 0.5: ln(Γ(z)) = ln(π) - ln(|sin(πz)|) - ln(Γ(1-z))
///
/// # Arguments
///
/// * `x` - Input array of values. Poles at non-positive integers (returns +∞).
///
/// # Returns
///
/// Array of ln(Γ(x)) values, same shape as input.
///
/// # Mathematical Properties
///
/// - ln(Γ(1)) = ln(Γ(2)) = 0
/// - ln(Γ(n)) = ln((n-1)!) for positive integers
/// - ln(Γ(1/2)) = ln(√π) ≈ 0.5724
/// - For large x: ln(Γ(x)) ≈ (x-0.5)·ln(x) - x + ln(√(2π))
///
/// # Example
///
/// ```ignore
/// use scirs2_core::ndarray_ext::elementwise::ln_gamma_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
/// let result = ln_gamma_simd(&x.view());
/// // ln(Γ(1)) = 0, ln(Γ(2)) = 0, ln(Γ(3)) = ln(2!) ≈ 0.693
/// // ln(Γ(4)) = ln(3!) ≈ 1.792, ln(Γ(5)) = ln(4!) ≈ 3.178
/// ```
pub fn ln_gamma_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_ln_gamma(x);
    }
    F::simd_ln_gamma(x)
}
/// Element-wise error function erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
///
/// Uses SIMD acceleration when available for optimal performance.
/// Critical for normal distribution CDF: Φ(x) = 0.5 * (1 + erf(x/√2))
///
/// # Properties
/// - erf(0) = 0
/// - erf(∞) = 1, erf(-∞) = -1
/// - erf(-x) = -erf(x) (odd function)
/// - erf(1) ≈ 0.8427007929497148
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::erf_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0_f64, 1.0, 2.0, -1.0];
/// let result = erf_simd(&x.view());
/// assert!((result[0] - 0.0_f64).abs() < 1e-10);  // erf(0) = 0
/// assert!((result[1] - 0.8427007929497148_f64).abs() < 1e-6);  // erf(1)
/// assert!((result[3] + 0.8427007929497148_f64).abs() < 1e-6);  // erf(-1) = -erf(1)
/// ```
pub fn erf_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_erf(x);
    }
    F::simd_erf(x)
}
/// Element-wise complementary error function erfc(x) = 1 - erf(x)
///
/// More numerically stable than computing 1 - erf(x) directly for large x.
/// Uses SIMD acceleration when available for optimal performance.
///
/// # Properties
/// - erfc(0) = 1
/// - erfc(∞) = 0, erfc(-∞) = 2
/// - erfc(x) = 1 - erf(x)
/// - erfc(-x) = 2 - erfc(x)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::erfc_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0];
/// let result = erfc_simd(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-10);  // erfc(0) = 1
/// assert!((result[1] - 0.1572992070502852).abs() < 1e-6);  // erfc(1)
/// ```
pub fn erfc_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_erfc(x);
    }
    F::simd_erfc(x)
}
/// Element-wise inverse error function erfinv(y) = x such that erf(x) = y
///
/// Uses SIMD acceleration when available for optimal performance.
/// Critical for inverse normal CDF (probit function): Φ⁻¹(p) = √2 * erfinv(2p - 1)
///
/// # Domain and Range
/// - Domain: (-1, 1)
/// - Range: (-∞, ∞)
/// - erfinv(-1) = -∞, erfinv(1) = ∞
///
/// # Properties
/// - erfinv(0) = 0
/// - erfinv(-y) = -erfinv(y) (odd function)
/// - erf(erfinv(y)) = y
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::erfinv_simd;
/// use scirs2_core::ndarray::array;
///
/// let y = array![0.0f64, 0.5, -0.5, 0.9];
/// let result = erfinv_simd(&y.view());
/// assert!((result[0] - 0.0).abs() < 1e-10);  // erfinv(0) = 0
/// ```
pub fn erfinv_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_erfinv(x);
    }
    F::simd_erfinv(x)
}
/// Element-wise inverse complementary error function erfcinv(y) = x such that erfc(x) = y
///
/// More numerically stable than erfinv(1 - y) for y close to 0.
/// Uses SIMD acceleration when available for optimal performance.
///
/// # Domain and Range
/// - Domain: (0, 2)
/// - Range: (-∞, ∞)
/// - erfcinv(0) = ∞, erfcinv(2) = -∞
///
/// # Properties
/// - erfcinv(1) = 0
/// - erfcinv(y) = erfinv(1 - y)
/// - erfc(erfcinv(y)) = y
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::erfcinv_simd;
/// use scirs2_core::ndarray::array;
///
/// let y = array![1.0f64, 0.5, 1.5];
/// let result = erfcinv_simd(&y.view());
/// assert!((result[0] - 0.0).abs() < 1e-10);  // erfcinv(1) = 0
/// ```
pub fn erfcinv_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_erfcinv(x);
    }
    F::simd_erfcinv(x)
}
/// Compute the element-wise sigmoid (logistic) function of an array.
///
/// The sigmoid function is defined as:
/// σ(x) = 1 / (1 + exp(-x))
///
/// This is critical for neural networks, logistic regression, and probability modeling.
/// The implementation is numerically stable, avoiding overflow for large |x|.
///
/// # Properties
///
/// - Range: (0, 1)
/// - σ(0) = 0.5
/// - σ(-x) = 1 - σ(x)
/// - Derivative: σ'(x) = σ(x)(1 - σ(x))
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::sigmoid_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, -1.0];
/// let result = sigmoid_simd(&x.view());
/// assert!((result[0] - 0.5).abs() < 1e-10);  // sigmoid(0) = 0.5
/// ```
pub fn sigmoid_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sigmoid(x);
    }
    F::simd_sigmoid(x)
}
/// Compute the element-wise GELU (Gaussian Error Linear Unit) of an array.
///
/// The GELU function is defined as:
/// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
///
/// Where Φ(x) is the cumulative distribution function of the standard normal distribution.
/// GELU is critical for Transformer models (BERT, GPT, etc.) and provides a smooth
/// approximation of ReLU.
///
/// # Properties
///
/// - GELU(0) = 0
/// - GELU(x) ≈ x for large positive x
/// - GELU(x) ≈ 0 for large negative x
/// - Smooth and differentiable everywhere
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::gelu_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, -1.0];
/// let result = gelu_simd(&x.view());
/// assert!(result[0].abs() < 1e-10);  // GELU(0) = 0
/// ```
pub fn gelu_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_gelu(x);
    }
    F::simd_gelu(x)
}
/// SIMD-accelerated Swish (SiLU - Sigmoid Linear Unit) activation function
///
/// Computes the Swish activation function element-wise:
/// `Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))`
///
/// Swish is a self-gated activation function discovered through neural architecture
/// search that has become a popular choice for modern neural networks.
///
/// # Key Properties
///
/// - Swish(0) = 0
/// - Swish is smooth and non-monotonic
/// - Has a small negative region (unlike ReLU)
/// - Self-gating: x modulates its own activation via sigmoid
/// - Unbounded above, bounded below (minimum ≈ -0.278 at x ≈ -1.278)
///
/// # Usage in Deep Learning
///
/// Swish is used in:
/// - EfficientNet and EfficientNetV2
/// - GPT-NeoX and other large language models
/// - MobileNetV3
/// - Many modern vision and NLP architectures
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::swish_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, -1.0];
/// let result = swish_simd(&x.view());
/// assert!(result[0].abs() < 1e-10);  // Swish(0) = 0
/// // Swish(1) ≈ 0.7311
/// assert!((result[1] - 0.7310585786).abs() < 1e-6);
/// // Swish(-1) ≈ -0.2689
/// assert!((result[2] - (-0.2689414214)).abs() < 1e-6);
/// ```
pub fn swish_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_swish(x);
    }
    F::simd_swish(x)
}
/// SIMD-accelerated Softplus activation function
///
/// Computes the Softplus activation function element-wise:
/// `Softplus(x) = ln(1 + exp(x))`
///
/// Softplus is a smooth approximation of ReLU and is commonly used in
/// probabilistic models, Bayesian deep learning, and smooth counting.
///
/// # Key Properties
///
/// - Softplus(0) = ln(2) ≈ 0.693
/// - Always positive (> 0 for all x)
/// - Derivative: softplus'(x) = sigmoid(x)
/// - Approaches ReLU for x → +∞: softplus(x) ≈ x
/// - Approaches 0 for x → -∞: softplus(x) ≈ exp(x) ≈ 0
///
/// # Usage
///
/// Softplus is used in:
/// - Probabilistic models (output layer for positive quantities)
/// - Bayesian deep learning
/// - Smooth counting applications
/// - As a smooth ReLU replacement
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::softplus_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, -1.0];
/// let result = softplus_simd(&x.view());
/// // Softplus(0) = ln(2) ≈ 0.693
/// assert!((result[0] - 0.6931471805599453).abs() < 1e-10);
/// // Softplus(1) = ln(1 + e) ≈ 1.3133
/// assert!((result[1] - 1.3132616875182228).abs() < 1e-10);
/// // Softplus(-1) = ln(1 + 1/e) ≈ 0.3133
/// assert!((result[2] - 0.31326168751822286).abs() < 1e-10);
/// ```
pub fn softplus_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_softplus(x);
    }
    F::simd_softplus(x)
}
/// SIMD-accelerated Mish activation function
///
/// Computes the Mish activation function element-wise:
/// `Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`
///
/// Mish is a self-regularized non-monotonic activation function that combines
/// the benefits of ReLU, Swish, and other modern activations.
///
/// # Key Properties
///
/// - Mish(0) = 0
/// - Smooth and non-monotonic
/// - Self-regularizing (bounded negative region)
/// - Unbounded above, bounded below (minimum ≈ -0.31 at x ≈ -1.2)
/// - Derivative: complex but well-behaved
///
/// # Usage in Deep Learning
///
/// Mish is used in:
/// - YOLOv4 and YOLOv5 object detection
/// - Modern convolutional neural networks
/// - Image classification and segmentation tasks
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::mish_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, -1.0];
/// let result = mish_simd(&x.view());
/// // Mish(0) = 0
/// assert!(result[0].abs() < 1e-10);
/// // Mish(1) = 1 * tanh(ln(1 + e)) ≈ 0.8651
/// assert!((result[1] - 0.8650983882673103).abs() < 1e-10);
/// ```
pub fn mish_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_mish(x);
    }
    F::simd_mish(x)
}

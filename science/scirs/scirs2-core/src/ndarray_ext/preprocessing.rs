//! SIMD-accelerated preprocessing operations for array normalization and standardization
//!
//! This module provides high-performance implementations of common data preprocessing
//! operations that are critical for machine learning pipelines, statistical analysis,
//! and scientific computing.
//!
//! # Operations
//!
//! - **L2 Normalization** (`normalize_simd`): Converts vectors to unit length
//! - **Z-Score Standardization** (`standardize_simd`): Zero mean, unit variance
//! - **Value Clipping** (`clip_simd`): Bounds values to a specified range
//!
//! # Performance
//!
//! All operations automatically use SIMD acceleration when:
//! - Platform supports AVX2 (x86_64) or NEON (ARM)
//! - Array size is large enough to benefit from vectorization
//! - Array memory layout is contiguous
//!
//! Falls back to scalar implementations for small arrays or unsupported platforms.
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_core::ndarray_ext::preprocessing::{normalize_simd, standardize_simd, clip_simd};
//!
//! // L2 normalization - convert to unit vector
//! let x = array![3.0, 4.0];  // norm = 5
//! let normalized = normalize_simd(&x.view());
//! // Result: [0.6, 0.8]
//!
//! // Z-score standardization
//! let data = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
//! let standardized = standardize_simd(&data.view());
//! // Result: mean ≈ 0, std ≈ 1
//!
//! // Value clipping
//! let values = array![-10.0, -5.0, 0.0, 5.0, 10.0];
//! let clipped = clip_simd(&values.view(), -3.0, 7.0);
//! // Result: [-3.0, -3.0, 0.0, 5.0, 7.0]
//! ```

use crate::numeric::Float;
use crate::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use ::ndarray::{Array1, ArrayView1};

/// Normalize a 1D array to unit length using L2 norm (SIMD-accelerated).
///
/// Computes x / ||x||₂ where ||x||₂ is the Euclidean (L2) norm.
/// The resulting vector will have unit length (norm = 1).
///
/// # Arguments
///
/// * `x` - Input 1D array to normalize
///
/// # Returns
///
/// `Array1<F>` with the same length as input, normalized to unit length.
/// Returns zero array if input norm is zero or NaN.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD unavailable
/// - **Speedup**: 2-4x for large f32 arrays on AVX2 systems
///
/// # Mathematical Definition
///
/// ```text
/// normalize(x) = x / ||x||₂
/// where ||x||₂ = sqrt(Σ xᵢ²)
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::preprocessing::normalize_simd;
///
/// // 3-4-5 triangle
/// let x = array![3.0_f64, 4.0];
/// let result = normalize_simd(&x.view());
///
/// // norm = sqrt(9 + 16) = 5
/// // result = [3/5, 4/5] = [0.6, 0.8]
/// assert!((result[0] - 0.6_f64).abs() < 1e-10);
/// assert!((result[1] - 0.8_f64).abs() < 1e-10);
///
/// // Verify unit norm
/// let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
/// assert!((norm - 1.0_f64).abs() < 1e-10);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero vector**: Returns zero array (undefined normalization)
/// - **Single element**: Returns array with single element ±1
/// - **NaN values**: Returns zero array
///
/// # Applications
///
/// - **Machine Learning**: Feature scaling for algorithms sensitive to magnitude
/// - **Cosine Similarity**: Prerequisite for efficient similarity computation
/// - **Direction Vectors**: Unit direction vectors in physics/graphics
/// - **Text Processing**: TF-IDF normalization in NLP
/// - **Signal Processing**: Normalized cross-correlation
pub fn normalize_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_normalize(x);
    }

    // Scalar fallback for small arrays
    let norm = x
        .iter()
        .map(|&val| val * val)
        .fold(F::zero(), |acc, val| acc + val)
        .sqrt();

    if norm == F::zero() || norm.is_nan() {
        return Array1::zeros(x.len());
    }

    x.mapv(|val| val / norm)
}

/// Standardize a 1D array to zero mean and unit variance (SIMD-accelerated).
///
/// Computes (x - μ) / σ where μ is the mean and σ is the sample standard deviation.
/// The resulting array will have mean ≈ 0 and standard deviation ≈ 1.
///
/// # Arguments
///
/// * `x` - Input 1D array to standardize
///
/// # Returns
///
/// `Array1<F>` with the same length as input, standardized to zero mean and unit variance.
/// Returns zero array if input has <= 1 element or zero standard deviation.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD unavailable
/// - **Speedup**: 2-4x for large f32 arrays on AVX2 systems
///
/// # Mathematical Definition
///
/// ```text
/// standardize(x) = (x - μ) / σ
/// where:
///   μ = (1/n) Σ xᵢ         (sample mean)
///   σ = sqrt((1/(n-1)) Σ (xᵢ - μ)²)  (sample std, ddof=1)
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::preprocessing::standardize_simd;
///
/// let x = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let result = standardize_simd(&x.view());
///
/// // Verify mean ≈ 0
/// let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
/// assert!(mean.abs() < 1e-10);
///
/// // Verify std ≈ 1 (sample std with ddof=1)
/// let variance: f64 = result.iter()
///     .map(|&x| x * x)
///     .sum::<f64>() / (result.len() - 1) as f64;
/// let std = variance.sqrt();
/// assert!((std - 1.0).abs() < 1e-10);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Single element**: Returns zero array (std undefined for n=1)
/// - **Constant array**: Returns zero array (std = 0)
/// - **NaN values**: Returns zero array
///
/// # Applications
///
/// - **Machine Learning**: Feature preprocessing for models assuming normally distributed features
/// - **Statistical Analysis**: Z-score computation for outlier detection
/// - **Time Series**: Detrending and variance normalization
/// - **Data Science**: Preparing features for PCA, clustering, regression
/// - **Neural Networks**: Input normalization for faster convergence
///
/// # Implementation Notes
///
/// Uses **sample standard deviation** (ddof=1, Bessel's correction) rather than
/// population standard deviation (ddof=0). This is consistent with NumPy, SciPy,
/// and pandas default behavior.
pub fn standardize_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    if x.len() <= 1 {
        return Array1::zeros(x.len());
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_standardize(x);
    }

    // Scalar fallback for small arrays
    let n = F::from(x.len()).expect("Operation failed");
    let mean = x.iter().fold(F::zero(), |acc, &val| acc + val) / n;

    let n_minus_1 = F::from(x.len() - 1).expect("Operation failed");
    let variance = x
        .iter()
        .map(|&val| {
            let diff = val - mean;
            diff * diff
        })
        .fold(F::zero(), |acc, val| acc + val)
        / n_minus_1;

    let std = variance.sqrt();

    if std == F::zero() || std.is_nan() {
        return Array1::zeros(x.len());
    }

    x.mapv(|val| (val - mean) / std)
}

/// Clip (clamp) array values to a specified range (SIMD-accelerated).
///
/// Constrains each element to be within [min_val, max_val]:
/// - Values < min_val are set to min_val
/// - Values > max_val are set to max_val
/// - Values within range are unchanged
///
/// # Arguments
///
/// * `x` - Input 1D array to clip
/// * `min_val` - Minimum allowed value
/// * `max_val` - Maximum allowed value
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with values clipped to [min_val, max_val].
///
/// # Panics
///
/// Panics if `min_val > max_val`.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD unavailable
/// - **Speedup**: 2-4x for large f32 arrays on AVX2 systems
///
/// # Mathematical Definition
///
/// ```text
/// clip(x, min, max) = {
///     min     if x < min
///     max     if x > max
///     x       otherwise
/// }
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::preprocessing::clip_simd;
///
/// let x = array![-10.0, -5.0, 0.0, 5.0, 10.0];
/// let result = clip_simd(&x.view(), -3.0, 7.0);
///
/// // Values clipped to [-3.0, 7.0]
/// assert_eq!(result[0], -3.0);  // -10 -> -3
/// assert_eq!(result[1], -3.0);  // -5 -> -3
/// assert_eq!(result[2], 0.0);   // 0 unchanged
/// assert_eq!(result[3], 5.0);   // 5 unchanged
/// assert_eq!(result[4], 7.0);   // 10 -> 7
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **min_val == max_val**: All values set to that value
/// - **NaN values**: Behavior is platform-dependent (typically become min_val)
/// - **Infinite values**: Clipped to min/max as appropriate
///
/// # Applications
///
/// - **Gradient Clipping**: Prevent exploding gradients in neural network training
/// - **Outlier Handling**: Remove or bound extreme values in statistical analysis
/// - **Data Preprocessing**: Enforce valid ranges for features
/// - **Numerical Stability**: Prevent overflow/underflow in computations
/// - **Image Processing**: Clamp pixel values to valid ranges
/// - **Audio Processing**: Prevent clipping distortion
///
/// # Example: Gradient Clipping
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::preprocessing::clip_simd;
///
/// // Clip gradients to prevent exploding gradients
/// let gradients = array![10.5, -8.2, 0.3, 15.7, -12.1];
/// let clipped = clip_simd(&gradients.view(), -5.0, 5.0);
///
/// // All gradients now in [-5.0, 5.0]
/// assert_eq!(clipped[0], 5.0);   // 10.5 -> 5.0
/// assert_eq!(clipped[1], -5.0);  // -8.2 -> -5.0
/// assert_eq!(clipped[2], 0.3);   // unchanged
/// assert_eq!(clipped[3], 5.0);   // 15.7 -> 5.0
/// assert_eq!(clipped[4], -5.0);  // -12.1 -> -5.0
/// ```
pub fn clip_simd<F>(x: &ArrayView1<F>, min_val: F, max_val: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    assert!(min_val <= max_val, "min_val must be <= max_val");

    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_clip(x, min_val, max_val);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| {
        if val < min_val {
            min_val
        } else if val > max_val {
            max_val
        } else {
            val
        }
    })
}

/// Compute softmax activation function with SIMD acceleration (Phase 33).
///
/// The softmax function converts a vector of real numbers into a probability distribution
/// where all values are in (0, 1) and sum to 1. It is numerically stable using the
/// max-subtraction trick.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` containing the softmax probabilities, where:
/// - All values are in the range (0, 1)
/// - Sum of all values equals 1.0
/// - Empty input returns empty array
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD unavailable
/// - **Speedup**: 4-8x for large arrays on AVX2/NEON systems
/// - Uses newly implemented min_simd (Phase 29) and sum_simd (Phase 30)
///
/// # Mathematical Definition
///
/// ```text
/// softmax(x)ᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
/// ```
///
/// The max-subtraction trick (xᵢ - max(x)) ensures numerical stability by preventing
/// overflow in the exponential function.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::preprocessing::softmax_simd;
///
/// let x = array![1.0, 2.0, 3.0];
/// let result = softmax_simd(&x.view());
///
/// // Verify probabilities sum to 1
/// let sum: f64 = result.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-10);
///
/// // Verify all values in (0, 1)
/// for &val in result.iter() {
///     assert!(val > 0.0 && val < 1.0);
/// }
/// ```
///
/// # Applications
///
/// - **Attention Mechanisms**: Compute attention weights in Transformers (PRIMARY USE CASE)
/// - **Multi-class Classification**: Convert logits to class probabilities
/// - **Neural Networks**: Final layer activation for multi-class problems
/// - **Reinforcement Learning**: Action probability distributions
/// - **Natural Language Processing**: Token probability distributions
///
/// # Numerical Stability
///
/// The implementation subtracts the maximum value before exponentiation to prevent
/// overflow. This is mathematically equivalent to the standard softmax but numerically
/// stable even for large input values.
///
/// # Example: Attention Weights
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::preprocessing::softmax_simd;
///
/// // Attention scores (before softmax)
/// let scores = array![2.0, 4.0, 1.0, 3.0];
/// let attention_weights = softmax_simd(&scores.view());
///
/// // Highest score (4.0) gets highest probability
/// let max_idx = attention_weights
///     .iter()
///     .enumerate()
///     .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
///     .map(|(idx, _)| idx)
///     .expect("Operation failed");
/// assert_eq!(max_idx, 1); // Index of score 4.0
///
/// // All weights sum to 1
/// let sum: f64 = attention_weights.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn softmax_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(x.len()) {
        // SIMD fast path using Phase 29-30 operations
        // 1. Find max for numerical stability (Phase 29: max_simd)
        let max_val = F::simd_max_element(x);

        // 2. Subtract max and exponentiate (SIMD exp)
        let shifted = x.mapv(|val| val - max_val);
        let exp_vals = F::simd_exp(&shifted.view());

        // 3. Sum the exponents (Phase 30: sum_simd)
        let sum = F::simd_sum(&exp_vals.view());

        // 4. Normalize by dividing by sum (scalar multiplication with 1/sum)
        if sum > F::zero() {
            let inv_sum = F::one() / sum;
            F::simd_scalar_mul(&exp_vals.view(), inv_sum)
        } else {
            Array1::zeros(x.len())
        }
    } else {
        // Scalar fallback for small arrays
        // Find max for numerical stability
        let max_val = x.fold(F::neg_infinity(), |max, &val| max.max(val));

        // Subtract max and exponentiate
        let mut exp_vals = x.mapv(|val| (val - max_val).exp());

        // Sum the exponents
        let sum = exp_vals.iter().fold(F::zero(), |acc, &val| acc + val);

        // Normalize
        if sum > F::zero() {
            exp_vals.mapv_inplace(|val| val / sum);
        }

        exp_vals
    }
}

// ============================================================================
// Activation Functions (Phase 33.5)
// ============================================================================

/// Compute ReLU (Rectified Linear Unit) activation with SIMD acceleration.
///
/// ReLU is one of the most common activation functions in deep learning, defined as:
/// ReLU(x) = max(0, x)
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` containing the ReLU activation output, where negative values are zeroed
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD unavailable
/// - **Speedup**: 3-5x for large arrays on AVX2/NEON systems
/// - Most common activation in CNNs, ResNets, and modern architectures
///
/// # Mathematical Definition
///
/// ```text
/// ReLU(x) = { x  if x > 0
///           { 0  if x ≤ 0
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::preprocessing::relu_simd;
///
/// let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
/// let result = relu_simd(&x.view());
///
/// assert_eq!(result[0], 0.0); // -2.0 -> 0.0
/// assert_eq!(result[1], 0.0); // -1.0 -> 0.0
/// assert_eq!(result[2], 0.0); //  0.0 -> 0.0
/// assert_eq!(result[3], 1.0); //  1.0 -> 1.0
/// assert_eq!(result[4], 2.0); //  2.0 -> 2.0
/// ```
///
/// # Applications
///
/// - **Convolutional Neural Networks**: Primary activation in CNN layers
/// - **ResNet/DenseNet**: Core activation in residual blocks
/// - **Fully Connected Layers**: Standard activation for hidden layers
/// - **Feature Extraction**: Non-linear transformation in deep networks
/// - **Modern Architectures**: Default choice for most deep learning models
///
/// # Advantages of ReLU
///
/// - **Computational Efficiency**: Very fast to compute (simple max operation)
/// - **No Gradient Vanishing**: Gradients don't saturate for positive values
/// - **Sparse Activation**: Promotes sparsity (zeros out negative values)
/// - **SIMD-Friendly**: Perfect for vectorization
#[allow(dead_code)]
pub fn relu_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(x.len()) {
        F::simd_relu(x)
    } else {
        // Scalar fallback: max(0, x)
        x.mapv(|val| if val > F::zero() { val } else { F::zero() })
    }
}

/// Compute Leaky ReLU activation with SIMD acceleration.
///
/// Leaky ReLU is a variant of ReLU that allows small negative values:
/// LeakyReLU(x) = max(αx, x) where α is a small constant (typically 0.01)
///
/// # Arguments
///
/// * `x` - Input 1D array
/// * `alpha` - Negative slope coefficient (typically 0.01)
///
/// # Returns
///
/// `Array1<F>` containing the Leaky ReLU activation output
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Speedup**: 3-5x for large arrays
///
/// # Mathematical Definition
///
/// ```text
/// LeakyReLU(x) = { x     if x > 0
///                { αx    if x ≤ 0
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::preprocessing::leaky_relu_simd;
///
/// let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
/// let result = leaky_relu_simd(&x.view(), 0.01);
///
/// assert_eq!(result[0], -0.02); // -2.0 * 0.01
/// assert_eq!(result[1], -0.01); // -1.0 * 0.01
/// assert_eq!(result[2], 0.0);   //  0.0 * 0.01
/// assert_eq!(result[3], 1.0);   //  1.0 (unchanged)
/// assert_eq!(result[4], 2.0);   //  2.0 (unchanged)
/// ```
///
/// # Applications
///
/// - **Addressing Dying ReLU Problem**: Allows gradient flow for negative values
/// - **GANs**: Common in discriminator networks
/// - **ResNet Variants**: Alternative to standard ReLU
#[allow(dead_code)]
pub fn leaky_relu_simd<F>(x: &ArrayView1<F>, alpha: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(x.len()) {
        F::simd_leaky_relu(x, alpha)
    } else {
        // Scalar fallback: max(alpha * x, x)
        x.mapv(|val| if val > F::zero() { val } else { val * alpha })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::array;

    #[test]
    fn test_normalize_simd_basic_f64() {
        let x = array![3.0, 4.0];
        let result = normalize_simd(&x.view());

        // 3-4-5 triangle: norm = 5, so [0.6, 0.8]
        assert!((result[0] - 0.6).abs() < 1e-10);
        assert!((result[1] - 0.8).abs() < 1e-10);

        // Verify unit norm
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_simd_basic_f32() {
        let x = array![3.0f32, 4.0];
        let result = normalize_simd(&x.view());

        assert!((result[0] - 0.6).abs() < 1e-6);
        assert!((result[1] - 0.8).abs() < 1e-6);

        // Verify unit norm
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_simd_empty() {
        let x: Array1<f64> = array![];
        let result = normalize_simd(&x.view());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_normalize_simd_zero_vector() {
        let x = array![0.0, 0.0, 0.0];
        let result = normalize_simd(&x.view());

        // Zero norm should return zero array
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
    }

    #[test]
    fn test_standardize_simd_basic_f64() {
        let x = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = standardize_simd(&x.view());

        // Verify mean ≈ 0
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        assert!(mean.abs() < 1e-10, "Mean should be ~0, got {}", mean);

        // Verify std ≈ 1 (sample std with ddof=1)
        let variance: f64 = result.iter().map(|&x| x * x).sum::<f64>() / (result.len() - 1) as f64;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 1e-10, "Std should be ~1, got {}", std);
    }

    #[test]
    fn test_standardize_simd_basic_f32() {
        let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = standardize_simd(&x.view());

        // Verify mean ≈ 0
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);

        // Verify std ≈ 1
        let variance: f32 = result.iter().map(|&x| x * x).sum::<f32>() / (result.len() - 1) as f32;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 1e-5, "Std should be ~1, got {}", std);
    }

    #[test]
    fn test_standardize_simd_empty() {
        let x: Array1<f64> = array![];
        let result = standardize_simd(&x.view());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_standardize_simd_single_element() {
        let x = array![5.0];
        let result = standardize_simd(&x.view());

        // Single element has undefined std, should return zero
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_standardize_simd_constant() {
        let x = array![5.0, 5.0, 5.0, 5.0];
        let result = standardize_simd(&x.view());

        // Constant array has zero std, should return zero array
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 0.0);
    }

    // ========================================================================
    // Tests for Phase 33: softmax_simd
    // ========================================================================

    #[test]
    fn test_softmax_simd_f64_basic() {
        let x = array![1.0f64, 2.0, 3.0];
        let result = softmax_simd(&x.view());

        // Verify probabilities sum to 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Verify all values in (0, 1)
        for &val in result.iter() {
            assert!(val > 0.0 && val < 1.0);
        }

        // Verify highest input gets highest probability
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
            .map(|(idx, _)| idx)
            .expect("Operation failed");
        assert_eq!(max_idx, 2); // Index of value 3.0
    }

    #[test]
    fn test_softmax_simd_f32_basic() {
        let x = array![1.0f32, 2.0, 3.0, 4.0];
        let result = softmax_simd(&x.view());

        // Verify probabilities sum to 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Verify all values in (0, 1)
        for &val in result.iter() {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_softmax_simd_empty() {
        let x: Array1<f64> = array![];
        let result = softmax_simd(&x.view());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_softmax_simd_single() {
        let x = array![42.0f64];
        let result = softmax_simd(&x.view());

        // Single element softmax should be 1.0
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_simd_uniform() {
        let x = array![2.0f64, 2.0, 2.0, 2.0];
        let result = softmax_simd(&x.view());

        // Uniform input should give uniform probabilities (0.25 each)
        for &val in result.iter() {
            assert!((val - 0.25).abs() < 1e-10);
        }

        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_simd_large_values() {
        // Test numerical stability with large values
        let x = array![1000.0f64, 1001.0, 1002.0];
        let result = softmax_simd(&x.view());

        // Should not overflow
        assert!(!result.iter().any(|&v| v.is_infinite() || v.is_nan()));

        // Should still sum to 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Highest value should get highest probability
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
            .map(|(idx, _)| idx)
            .expect("Operation failed");
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn test_softmax_simd_negative_values() {
        let x = array![-10.0f64, -5.0, -2.0, -8.0];
        let result = softmax_simd(&x.view());

        // Should sum to 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Highest value (-2.0) should get highest probability
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
            .map(|(idx, _)| idx)
            .expect("Operation failed");
        assert_eq!(max_idx, 2); // Index of -2.0
    }

    #[test]
    fn test_softmax_simd_attention_scores() {
        // Realistic attention scores scenario
        let scores = array![2.0f64, 4.0, 1.0, 3.0];
        let attention_weights = softmax_simd(&scores.view());

        // Verify it's a valid probability distribution
        let sum: f64 = attention_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Highest score (4.0) should get highest weight
        let max_idx = attention_weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
            .map(|(idx, _)| idx)
            .expect("Operation failed");
        assert_eq!(max_idx, 1); // Index of score 4.0

        // Verify all weights are positive
        for &weight in attention_weights.iter() {
            assert!(weight > 0.0);
        }
    }

    #[test]
    fn test_softmax_simd_large_array() {
        // Test SIMD path with large array (>1000 elements)
        let x: Array1<f64> = Array1::from_vec((0..5000).map(|i| (i as f64) * 0.01).collect());
        let result = softmax_simd(&x.view());

        // Should sum to 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        // All values should be positive
        assert!(result.iter().all(|&v| v > 0.0));

        // Highest input should give highest probability
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
            .map(|(idx, _)| idx)
            .expect("Operation failed");
        assert_eq!(max_idx, 4999); // Last element has highest value
    }

    #[test]
    fn test_softmax_simd_temperature_scaling() {
        // Test temperature scaling (dividing logits by temperature)
        let logits = array![1.0f64, 2.0, 3.0];

        // High temperature (T=2.0) -> more uniform distribution
        let high_temp = softmax_simd(&logits.mapv(|x| x / 2.0).view());

        // Low temperature (T=0.5) -> more peaked distribution
        let low_temp = softmax_simd(&logits.mapv(|x| x * 2.0).view());

        // Both should sum to 1
        let sum_high: f64 = high_temp.iter().sum();
        let sum_low: f64 = low_temp.iter().sum();
        assert!((sum_high - 1.0).abs() < 1e-10);
        assert!((sum_low - 1.0).abs() < 1e-10);

        // Low temperature should have higher max probability (more peaked)
        let max_high = high_temp.iter().cloned().fold(0.0f64, f64::max);
        let max_low = low_temp.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_low > max_high);
    }

    // ========================================================================
    // Tests for Phase 33.5: relu_simd and leaky_relu_simd
    // ========================================================================

    #[test]
    fn test_relu_simd_f64_basic() {
        let x = array![-2.0f64, -1.0, 0.0, 1.0, 2.0];
        let result = relu_simd(&x.view());

        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 2.0);
    }

    #[test]
    fn test_relu_simd_f32_basic() {
        let x = array![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let result = relu_simd(&x.view());

        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 2.0);
    }

    #[test]
    fn test_relu_simd_all_positive() {
        let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let result = relu_simd(&x.view());

        // All values should remain unchanged
        for i in 0..5 {
            assert_eq!(result[i], x[i]);
        }
    }

    #[test]
    fn test_relu_simd_all_negative() {
        let x = array![-5.0f64, -4.0, -3.0, -2.0, -1.0];
        let result = relu_simd(&x.view());

        // All values should be zero
        for i in 0..5 {
            assert_eq!(result[i], 0.0);
        }
    }

    #[test]
    fn test_relu_simd_empty() {
        let x: Array1<f64> = array![];
        let result = relu_simd(&x.view());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_relu_simd_large_array() {
        // Test SIMD path with large array (>1000 elements)
        let x: Array1<f64> = Array1::from_vec((0..5000).map(|i| (i as f64) - 2500.0).collect());
        let result = relu_simd(&x.view());

        // First 2500 should be zero (negative)
        for i in 0..2500 {
            assert_eq!(result[i], 0.0);
        }

        // Element at 2500 is exactly 0 (2500 - 2500 = 0)
        assert_eq!(result[2500], 0.0);

        // Last 2499 should be positive (2501-2500=1, ..., 4999-2500=2499)
        for i in 2501..5000 {
            assert!(result[i] > 0.0);
        }
    }

    #[test]
    fn test_leaky_relu_simd_f64_basic() {
        let x = array![-2.0f64, -1.0, 0.0, 1.0, 2.0];
        let result = leaky_relu_simd(&x.view(), 0.01);

        assert_eq!(result[0], -0.02);
        assert_eq!(result[1], -0.01);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 2.0);
    }

    #[test]
    fn test_leaky_relu_simd_f32_basic() {
        let x = array![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let result = leaky_relu_simd(&x.view(), 0.01);

        assert!((result[0] - (-0.02)).abs() < 1e-6);
        assert!((result[1] - (-0.01)).abs() < 1e-6);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 2.0);
    }

    #[test]
    fn test_leaky_relu_simd_different_alpha() {
        let x = array![-10.0f64, -5.0, 0.0, 5.0, 10.0];

        // Test with alpha = 0.1
        let result_01 = leaky_relu_simd(&x.view(), 0.1);
        assert_eq!(result_01[0], -1.0); // -10.0 * 0.1
        assert_eq!(result_01[1], -0.5); // -5.0 * 0.1

        // Test with alpha = 0.2
        let result_02 = leaky_relu_simd(&x.view(), 0.2);
        assert_eq!(result_02[0], -2.0); // -10.0 * 0.2
        assert_eq!(result_02[1], -1.0); // -5.0 * 0.2
    }

    #[test]
    fn test_leaky_relu_simd_preserves_positive() {
        let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let result = leaky_relu_simd(&x.view(), 0.01);

        // All positive values should remain unchanged
        for i in 0..5 {
            assert_eq!(result[i], x[i]);
        }
    }

    #[test]
    fn test_leaky_relu_simd_empty() {
        let x: Array1<f64> = array![];
        let result = leaky_relu_simd(&x.view(), 0.01);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_relu_vs_leaky_relu() {
        let x = array![-2.0f64, -1.0, 0.0, 1.0, 2.0];

        let relu_result = relu_simd(&x.view());
        let leaky_result = leaky_relu_simd(&x.view(), 0.01);

        // Positive values should be the same
        assert_eq!(relu_result[3], leaky_result[3]);
        assert_eq!(relu_result[4], leaky_result[4]);

        // Negative values should differ
        assert!(relu_result[0] != leaky_result[0]);
        assert!(relu_result[1] != leaky_result[1]);

        // ReLU zeros negatives, Leaky ReLU scales them
        assert_eq!(relu_result[0], 0.0);
        assert_eq!(leaky_result[0], -0.02);
    }
}

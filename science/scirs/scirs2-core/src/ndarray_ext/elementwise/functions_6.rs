//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::numeric::Float;
use crate::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use ::ndarray::{Array1, ArrayView1};

/// SIMD-accelerated Chebyshev distance
///
/// Computes max(|a - b|)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Chebyshev (L∞) distance between vectors
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::distance_chebyshev_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![4.0_f64, 0.0, 3.0];
/// let d = distance_chebyshev_simd::<f64>(&a.view(), &b.view());
/// // max(|1-4|, |2-0|, |3-3|) = max(3, 2, 0) = 3
/// assert!((d - 3.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Chess king distance
/// - Maximum deviation
/// - Robust outlier detection
/// - Image processing
pub fn distance_chebyshev_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }
    F::simd_distance_chebyshev(a, b)
}
/// SIMD-accelerated cosine distance
///
/// Computes 1 - cosine_similarity(a, b)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine distance (0 = identical direction, 2 = opposite direction)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::distance_cosine_simd;
///
/// let a = array![1.0_f64, 0.0, 0.0];
/// let b = array![0.0_f64, 1.0, 0.0];
/// let d = distance_cosine_simd::<f64>(&a.view(), &b.view());
/// // Orthogonal vectors have cosine similarity 0, distance 1
/// assert!((d - 1.0).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Text similarity (TF-IDF vectors)
/// - Recommendation systems
/// - Document clustering
/// - Image retrieval
pub fn distance_cosine_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::one();
    }
    F::simd_distance_cosine(a, b)
}
/// SIMD-accelerated cosine similarity
///
/// Computes (a · b) / (||a|| * ||b||)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity (-1 to 1, where 1 = identical direction)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::cosine_similarity_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![2.0_f64, 4.0, 6.0];
/// let sim = cosine_similarity_simd::<f64>(&a.view(), &b.view());
/// // Parallel vectors have cosine similarity 1
/// assert!((sim - 1.0).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Word embedding similarity
/// - Document comparison
/// - Recommendation scoring
/// - Semantic search
pub fn cosine_similarity_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }
    F::simd_cosine_similarity(a, b)
}
/// SIMD-accelerated element-wise addition
///
/// Computes a + b element-wise.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
/// Element-wise sum
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::add_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![4.0_f64, 5.0, 6.0];
/// let c = add_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 5.0).abs() < 1e-14);
/// assert!((c[1] - 7.0).abs() < 1e-14);
/// assert!((c[2] - 9.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Vector arithmetic
/// - Residual connections
/// - Bias addition
/// - Signal combination
pub fn add_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_add(a, b)
}
/// SIMD-accelerated element-wise subtraction
///
/// Computes a - b element-wise.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
/// Element-wise difference
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sub_simd;
///
/// let a = array![5.0_f64, 7.0, 9.0];
/// let b = array![1.0_f64, 2.0, 3.0];
/// let c = sub_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 4.0).abs() < 1e-14);
/// assert!((c[1] - 5.0).abs() < 1e-14);
/// assert!((c[2] - 6.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Gradient computation
/// - Error calculation
/// - Differencing signals
/// - Relative positioning
pub fn sub_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_sub(a, b)
}
/// SIMD-accelerated element-wise multiplication
///
/// Computes a * b element-wise (Hadamard product).
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
/// Element-wise product
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::mul_simd;
///
/// let a = array![2.0_f64, 3.0, 4.0];
/// let b = array![5.0_f64, 6.0, 7.0];
/// let c = mul_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 10.0).abs() < 1e-14);
/// assert!((c[1] - 18.0).abs() < 1e-14);
/// assert!((c[2] - 28.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Attention weights application
/// - Feature gating
/// - Gradient scaling
/// - Signal modulation
pub fn mul_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_mul(a, b)
}
/// SIMD-accelerated element-wise division
///
/// Computes a / b element-wise.
///
/// # Arguments
/// * `a` - Numerator array
/// * `b` - Denominator array
///
/// # Returns
/// Element-wise quotient
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::div_simd;
///
/// let a = array![10.0_f64, 20.0, 30.0];
/// let b = array![2.0_f64, 4.0, 5.0];
/// let c = div_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 5.0).abs() < 1e-14);
/// assert!((c[1] - 5.0).abs() < 1e-14);
/// assert!((c[2] - 6.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Normalization
/// - Ratio computation
/// - Scaling by variable factors
/// - Probability normalization
pub fn div_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_div(a, b)
}
/// SIMD-accelerated element-wise maximum
///
/// Computes max(a, b) element-wise.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
/// Element-wise maximum
///
/// # Examples
/// ```ignore
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::max_simd;
///
/// let a = array![1.0_f64, 5.0, 3.0];
/// let b = array![4.0_f64, 2.0, 6.0];
/// let c = max_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 4.0_f64).abs() < 1e-14);
/// assert!((c[1] - 5.0_f64).abs() < 1e-14);
/// assert!((c[2] - 6.0_f64).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - ReLU activation: max(0, x)
/// - Soft clipping
/// - Envelope detection
/// - Upper bound enforcement
pub fn max_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_max(a, b)
}
/// SIMD-accelerated element-wise minimum
///
/// Computes min(a, b) element-wise.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
/// Element-wise minimum
///
/// # Examples
/// ```ignore
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::min_simd;
///
/// let a = array![1.0_f64, 5.0, 3.0];
/// let b = array![4.0_f64, 2.0, 6.0];
/// let c = min_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 1.0_f64).abs() < 1e-14);
/// assert!((c[1] - 2.0_f64).abs() < 1e-14);
/// assert!((c[2] - 3.0_f64).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Gradient clipping
/// - Soft clipping
/// - Lower bound enforcement
/// - Minimum filter in signal processing
pub fn min_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_min(a, b)
}
/// SIMD-accelerated scalar multiplication
///
/// Computes a * scalar for all elements.
///
/// # Arguments
/// * `a` - Input array
/// * `scalar` - Scalar multiplier
///
/// # Returns
/// Scaled array
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::scalar_mul_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let c = scalar_mul_simd::<f64>(&a.view(), 2.5);
/// assert!((c[0] - 2.5).abs() < 1e-14);
/// assert!((c[1] - 5.0).abs() < 1e-14);
/// assert!((c[2] - 7.5).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Learning rate scaling
/// - Normalization
/// - Unit conversion
/// - Signal amplification
pub fn scalar_mul_simd<F>(a: &ArrayView1<F>, scalar: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_scalar_mul(a, scalar)
}
/// SIMD-accelerated fused multiply-add
///
/// Computes a * b + c element-wise in a single operation.
/// More accurate and efficient than separate multiply and add.
///
/// # Arguments
/// * `a` - First multiplicand
/// * `b` - Second multiplicand
/// * `c` - Addend
///
/// # Returns
/// Element-wise fused multiply-add result
///
/// # Examples
/// ```ignore
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::fma_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![4.0_f64, 5.0, 6.0];
/// let c = array![7.0_f64, 8.0, 9.0];
/// let result = fma_simd::<f64>(&a.view(), &b.view(), &c.view());
/// // 1*4+7=11, 2*5+8=18, 3*6+9=27
/// assert!((result[0] - 11.0_f64).abs() < 1e-14);
/// assert!((result[1] - 18.0_f64).abs() < 1e-14);
/// assert!((result[2] - 27.0_f64).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Matrix multiplication inner loop
/// - Polynomial evaluation (Horner's method)
/// - Linear combinations
/// - Neural network forward pass
pub fn fma_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>, c: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() || c.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_fma(a, b, c)
}
/// SIMD-accelerated dot product
///
/// Computes sum(a * b).
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Dot product (scalar)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::dot_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![4.0_f64, 5.0, 6.0];
/// let d = dot_simd::<f64>(&a.view(), &b.view());
/// // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// assert!((d - 32.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Projection calculations
/// - Cosine similarity numerator
/// - Attention scores
/// - Linear layer computation
pub fn dot_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }
    F::simd_dot(a, b)
}
/// SIMD-accelerated ReLU (Rectified Linear Unit)
///
/// Computes max(0, x) element-wise.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// ReLU-activated array
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::relu_simd;
///
/// let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
/// let result = relu_simd::<f64>(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-14);
/// assert!((result[1] - 0.0).abs() < 1e-14);
/// assert!((result[2] - 0.0).abs() < 1e-14);
/// assert!((result[3] - 1.0).abs() < 1e-14);
/// assert!((result[4] - 2.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Neural network activation
/// - Sparse representations
/// - Thresholding signals
/// - Feature rectification
pub fn relu_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_relu(a)
}
/// SIMD-accelerated L2 normalization
///
/// Normalizes the vector to unit length: x / ||x||₂
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Unit-normalized array (or zero if input is zero)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::normalize_simd;
///
/// let x = array![3.0_f64, 4.0];
/// let result = normalize_simd::<f64>(&x.view());
/// // ||result|| = 1
/// let norm = (result[0]*result[0] + result[1]*result[1]).sqrt();
/// assert!((norm - 1.0).abs() < 1e-10);
/// // x/5 = [0.6, 0.8]
/// assert!((result[0] - 0.6).abs() < 1e-10);
/// assert!((result[1] - 0.8).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Unit vector computation
/// - Cosine similarity preparation
/// - Gradient normalization
/// - Direction extraction
pub fn normalize_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_normalize(a)
}
/// SIMD-accelerated standardization (z-score normalization)
///
/// Transforms to zero mean and unit variance: (x - mean) / std
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Standardized array
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::standardize_simd;
///
/// let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let result = standardize_simd::<f64>(&x.view());
/// // Mean should be ~0, std should be ~1
/// let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
/// assert!(mean.abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Feature scaling for ML
/// - Batch normalization
/// - Statistical preprocessing
/// - Anomaly scoring
pub fn standardize_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_standardize(a)
}
/// SIMD-accelerated softmax
///
/// Computes exp(x - max(x)) / sum(exp(x - max(x))) for numerical stability.
/// Output probabilities sum to 1.
///
/// # Arguments
/// * `a` - Input array (logits)
///
/// # Returns
/// Probability distribution (sums to 1)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::softmax_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0];
/// let result = softmax_simd::<f64>(&x.view());
/// // Probabilities should sum to 1
/// let sum: f64 = result.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-10);
/// // Higher logits should have higher probabilities
/// assert!(result[2] > result[1]);
/// assert!(result[1] > result[0]);
/// ```
///
/// # Use Cases
/// - Classification output layer
/// - Attention weights
/// - Policy probabilities in RL
/// - Mixture model weights
pub fn softmax_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_softmax(a)
}
/// SIMD-accelerated truncation (round towards zero)
///
/// Removes the fractional part, rounding towards zero.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Truncated array
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::trunc_simd;
///
/// let x = array![2.7_f64, -2.7, 0.9, -0.9];
/// let result = trunc_simd::<f64>(&x.view());
/// assert!((result[0] - 2.0).abs() < 1e-14);
/// assert!((result[1] - (-2.0)).abs() < 1e-14);
/// assert!((result[2] - 0.0).abs() < 1e-14);
/// assert!((result[3] - 0.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Integer conversion
/// - Discretization
/// - Quantization
/// - Index calculation
pub fn trunc_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    F::simd_trunc(a)
}

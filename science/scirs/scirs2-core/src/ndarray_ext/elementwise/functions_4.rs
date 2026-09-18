//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::numeric::Float;
use crate::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use ::ndarray::{Array1, ArrayView1};

/// Apply ELU (Exponential Linear Unit) activation using SIMD operations
///
/// ELU is defined as:
/// - f(x) = x, if x >= 0
/// - f(x) = α * (exp(x) - 1), if x < 0
///
/// ELU is used in deep neural networks to:
/// - Push mean activations closer to zero (faster learning)
/// - Have negative values (unlike ReLU) for better gradient flow
/// - Have a smooth curve everywhere (unlike Leaky ReLU)
///
/// # Arguments
/// * `x` - Input array
/// * `alpha` - Scaling factor for negative inputs (commonly 1.0)
///
/// # Returns
/// * Array with ELU applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::elu_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![1.0_f32, 0.0, -1.0, -2.0];
/// let result = elu_simd(&x.view(), 1.0);
/// assert!((result[0] - 1.0).abs() < 1e-6);  // Positive: unchanged
/// assert!((result[1] - 0.0).abs() < 1e-6);  // Zero: unchanged
/// assert!(result[2] < 0.0);  // Negative: α * (exp(x) - 1) < 0
/// ```
pub fn elu_simd<F>(x: &ArrayView1<F>, alpha: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_elu(x, alpha);
    }
    F::simd_elu(x, alpha)
}
/// Apply Leaky ReLU / PReLU activation using SIMD operations
///
/// Leaky ReLU (Parametric ReLU when alpha is learned) is defined as:
/// - f(x) = x, if x >= 0
/// - f(x) = alpha * x, if x < 0
///
/// Leaky ReLU addresses the "dying ReLU" problem by allowing
/// a small gradient for negative inputs, preventing neurons from
/// becoming permanently inactive.
///
/// # Arguments
/// * `x` - Input array
/// * `alpha` - Slope for negative inputs (commonly 0.01 for Leaky ReLU, learned for PReLU)
///
/// # Returns
/// * Array with Leaky ReLU applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::leaky_relu_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![1.0_f32, 0.0, -1.0, -2.0];
/// let result = leaky_relu_simd(&x.view(), 0.01);
/// assert!((result[0] - 1.0).abs() < 1e-6);    // Positive: unchanged
/// assert!((result[1] - 0.0).abs() < 1e-6);    // Zero: unchanged
/// assert!((result[2] - (-0.01)).abs() < 1e-6); // Negative: alpha * x
/// ```
pub fn leaky_relu_simd<F>(x: &ArrayView1<F>, alpha: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_leaky_relu(x, alpha);
    }
    F::simd_leaky_relu(x, alpha)
}
/// Alias for leaky_relu_simd - PReLU (Parametric ReLU)
///
/// PReLU is mathematically identical to Leaky ReLU, but the alpha
/// parameter is learned during training rather than being fixed.
/// This function provides a convenience alias for neural network code
/// that uses PReLU terminology.
#[inline]
pub fn prelu_simd<F>(x: &ArrayView1<F>, alpha: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    leaky_relu_simd(x, alpha)
}
/// Apply SELU (Scaled Exponential Linear Unit) activation using SIMD operations
///
/// SELU is defined as:
/// - f(x) = λ * x, if x > 0
/// - f(x) = λ * α * (exp(x) - 1), if x <= 0
///
/// where λ ≈ 1.0507 and α ≈ 1.6733 are fixed constants.
///
/// SELU is the key activation for Self-Normalizing Neural Networks (SNNs):
/// - Automatically maintains mean ≈ 0 and variance ≈ 1 through layers
/// - Eliminates the need for Batch Normalization
/// - Requires LeCun Normal initialization for weights
/// - Works best with fully-connected networks
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * Array with SELU applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::selu_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![1.0_f32, 0.0, -1.0, -2.0];
/// let result = selu_simd(&x.view());
/// assert!(result[0] > 1.0);  // Positive: scaled by λ ≈ 1.0507
/// assert!((result[1] - 0.0).abs() < 1e-6);  // Zero: unchanged
/// assert!(result[2] < 0.0);  // Negative: λ * α * (exp(x) - 1) < 0
/// ```
pub fn selu_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_selu(x);
    }
    F::simd_selu(x)
}
/// Apply Hardsigmoid activation using SIMD operations
///
/// Hardsigmoid is defined as:
/// - f(x) = 0, if x <= -3
/// - f(x) = 1, if x >= 3
/// - f(x) = (x + 3) / 6, otherwise
///
/// Hardsigmoid is a piecewise linear approximation of sigmoid:
/// - Used in MobileNetV3 for efficient mobile inference
/// - Avoids expensive exp() computation
/// - Output always in range [0, 1]
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * Array with Hardsigmoid applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::hardsigmoid_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![-4.0_f32, -3.0, 0.0, 3.0, 4.0];
/// let result = hardsigmoid_simd(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-6);   // Saturated at 0
/// assert!((result[1] - 0.0).abs() < 1e-6);   // Boundary
/// assert!((result[2] - 0.5).abs() < 1e-6);   // Linear region: (0+3)/6 = 0.5
/// assert!((result[3] - 1.0).abs() < 1e-6);   // Boundary
/// assert!((result[4] - 1.0).abs() < 1e-6);   // Saturated at 1
/// ```
pub fn hardsigmoid_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_hardsigmoid(x);
    }
    F::simd_hardsigmoid(x)
}
/// Apply Hardswish activation using SIMD operations
///
/// Hardswish is defined as:
/// - f(x) = 0, if x <= -3
/// - f(x) = x, if x >= 3
/// - f(x) = x * (x + 3) / 6, otherwise
///
/// Hardswish is a piecewise linear approximation of Swish:
/// - Used in MobileNetV3 for efficient mobile inference
/// - Self-gating like Swish but without exp() computation
/// - Smooth at boundaries despite being piecewise
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * Array with Hardswish applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::hardswish_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![-4.0_f32, -3.0, 0.0, 3.0, 4.0];
/// let result = hardswish_simd(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-6);   // Saturated at 0
/// assert!((result[1] - 0.0).abs() < 1e-6);   // Boundary: -3 * 0 / 6 = 0
/// assert!((result[2] - 0.0).abs() < 1e-6);   // Linear: 0 * 3 / 6 = 0
/// assert!((result[3] - 3.0).abs() < 1e-6);   // Boundary: identity
/// assert!((result[4] - 4.0).abs() < 1e-6);   // Identity region
/// ```
pub fn hardswish_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_hardswish(x);
    }
    F::simd_hardswish(x)
}
/// Apply Sinc function using SIMD operations
///
/// The normalized sinc function is defined as:
/// - sinc(x) = sin(πx) / (πx) for x ≠ 0
/// - sinc(0) = 1 (by L'Hôpital's rule)
///
/// The sinc function is fundamental in signal processing:
/// - Ideal low-pass filter impulse response
/// - Whittaker-Shannon interpolation formula
/// - Windowing functions (Lanczos kernel)
/// - Sampling theory and Nyquist theorem
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * Array with sinc applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::sinc_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![0.0_f64, 0.5, 1.0, 2.0];
/// let result = sinc_simd(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-10);   // sinc(0) = 1
/// assert!((result[2] - 0.0).abs() < 1e-10);   // sinc(1) = 0
/// assert!((result[3] - 0.0).abs() < 1e-10);   // sinc(2) = 0
/// ```
pub fn sinc_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sinc(x);
    }
    F::simd_sinc(x)
}
/// Apply Log-Softmax function using SIMD operations
///
/// The log-softmax function is defined as:
/// log_softmax(x_i) = x_i - log(Σ_j exp(x_j))
///
/// This is more numerically stable than computing log(softmax(x)) directly,
/// and is critical for neural network training:
/// - Cross-entropy loss computation
/// - Classification networks (output layer)
/// - Transformer language models
/// - Large language models (LLMs)
///
/// # Mathematical Properties
///
/// - log_softmax(x) = x - logsumexp(x)
/// - Σ exp(log_softmax(x)) = 1 (outputs are log-probabilities)
/// - log_softmax(x + c) = log_softmax(x) (invariant to constant shift)
/// - Maximum output element approaches 0 for peaked distributions
///
/// # Arguments
/// * `x` - Input array of logits (unnormalized log-probabilities)
///
/// # Returns
/// * Array with log-softmax applied elementwise (log-probabilities that sum to 0)
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::log_softmax_simd;
/// use ndarray::{array, ArrayView1};
///
/// let logits = array![1.0_f64, 2.0, 3.0];
/// let log_probs = log_softmax_simd(&logits.view());
///
/// // Verify: exp(log_probs) sums to 1
/// let probs: f64 = log_probs.mapv(|lp| lp.exp()).sum();
/// assert!((probs - 1.0).abs() < 1e-10);
///
/// // log_softmax values are always <= 0
/// for &lp in log_probs.iter() {
///     assert!(lp <= 0.0);
/// }
/// ```
///
/// # Applications
///
/// - **Cross-Entropy Loss**: -Σ target * log_softmax(logits)
/// - **Classification**: Converting logits to log-probabilities
/// - **Transformers**: Final output layer for token prediction
/// - **Language Models**: Computing perplexity and token probabilities
/// - **Numerical Stability**: Avoiding underflow in softmax computation
pub fn log_softmax_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_log_softmax(x);
    }
    F::simd_log_softmax(x)
}
/// Apply inverse hyperbolic sine (asinh) using SIMD operations
///
/// The inverse hyperbolic sine is defined as:
/// asinh(x) = ln(x + √(x² + 1))
///
/// Domain: (-∞, +∞), Range: (-∞, +∞)
/// This is the inverse function of sinh.
///
/// # Mathematical Properties
///
/// - asinh(0) = 0
/// - asinh(-x) = -asinh(x) (odd function)
/// - asinh'(x) = 1/√(x² + 1)
/// - For large x: asinh(x) ≈ ln(2x)
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * Array with asinh applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::asinh_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![0.0_f64, 1.0, -1.0, 10.0];
/// let result = asinh_simd(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-10);
/// assert!((result[1] - 0.881373587).abs() < 1e-6);  // asinh(1)
/// assert!((result[1] + result[2]).abs() < 1e-10);  // odd function
/// ```
///
/// # Applications
///
/// - **Hyperbolic Geometry**: Distance calculations in hyperbolic space
/// - **Special Relativity**: Rapidity in Lorentz transformations
/// - **Signal Processing**: Parametric representation of signals
/// - **Statistics**: Transformations for variance stabilization
pub fn asinh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_asinh(x);
    }
    F::simd_asinh(x)
}
/// Apply inverse hyperbolic cosine (acosh) using SIMD operations
///
/// The inverse hyperbolic cosine is defined as:
/// acosh(x) = ln(x + √(x² - 1))
///
/// Domain: [1, +∞), Range: [0, +∞)
/// Returns NaN for x < 1.
/// This is the inverse function of cosh.
///
/// # Mathematical Properties
///
/// - acosh(1) = 0
/// - acosh(x) is monotonically increasing for x ≥ 1
/// - acosh'(x) = 1/√(x² - 1)
/// - For large x: acosh(x) ≈ ln(2x)
///
/// # Arguments
/// * `x` - Input array (values should be ≥ 1 for valid results)
///
/// # Returns
/// * Array with acosh applied elementwise (NaN for values < 1)
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::acosh_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![1.0_f64, 2.0, 10.0];
/// let result = acosh_simd(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-10);         // acosh(1) = 0
/// assert!((result[1] - 1.316957897).abs() < 1e-6);  // acosh(2)
///
/// // Out of domain returns NaN
/// let x_invalid = array![0.5_f64];
/// let result_invalid = acosh_simd(&x_invalid.view());
/// assert!(result_invalid[0].is_nan());
/// ```
///
/// # Applications
///
/// - **Hyperbolic Geometry**: Distance in Poincaré disk model
/// - **Physics**: Catenary curves, suspension bridges
/// - **Electronics**: Transmission line analysis
/// - **Computer Graphics**: Hyperbolic tessellations
pub fn acosh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_acosh(x);
    }
    F::simd_acosh(x)
}
/// Apply inverse hyperbolic tangent (atanh) using SIMD operations
///
/// The inverse hyperbolic tangent is defined as:
/// atanh(x) = 0.5 * ln((1+x)/(1-x))
///
/// Domain: (-1, 1), Range: (-∞, +∞)
/// Returns ±∞ at x = ±1, NaN for |x| > 1.
/// This is the inverse function of tanh.
///
/// # Mathematical Properties
///
/// - atanh(0) = 0
/// - atanh(-x) = -atanh(x) (odd function)
/// - atanh(±1) = ±∞
/// - atanh'(x) = 1/(1 - x²)
///
/// # Arguments
/// * `x` - Input array (values should be in (-1, 1) for finite results)
///
/// # Returns
/// * Array with atanh applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::atanh_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![0.0_f64, 0.5, -0.5, 0.99];
/// let result = atanh_simd(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-10);          // atanh(0) = 0
/// assert!((result[1] - 0.5493061).abs() < 1e-6);     // atanh(0.5)
/// assert!((result[1] + result[2]).abs() < 1e-10);   // odd function
///
/// // Boundaries
/// let x_boundary = array![1.0_f64, -1.0];
/// let result_boundary = atanh_simd(&x_boundary.view());
/// assert!(result_boundary[0].is_infinite() && result_boundary[0] > 0.0);
/// assert!(result_boundary[1].is_infinite() && result_boundary[1] < 0.0);
/// ```
///
/// # Applications
///
/// - **Statistics**: Fisher's z-transformation for correlation coefficients
/// - **Signal Processing**: Parametric signal representation
/// - **Probability**: Logit function relationship (atanh(x) = 0.5*logit((1+x)/2))
/// - **Machine Learning**: Activation function transformations
pub fn atanh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_atanh(x);
    }
    F::simd_atanh(x)
}
/// Compute the Beta function B(a, b) using SIMD operations
///
/// The Beta function is defined as:
/// B(a, b) = Γ(a)Γ(b) / Γ(a+b)
///
/// This function computes `B(a[i], b[i])` for each pair of elements.
/// The Beta function is fundamental in:
/// - Beta distribution (Bayesian priors for probabilities)
/// - Binomial coefficients: C(n,k) = 1/((n+1)·B(n-k+1, k+1))
/// - Statistical hypothesis testing
/// - Machine learning (Dirichlet processes)
///
/// # Mathematical Properties
///
/// - B(a, b) = B(b, a) (symmetric)
/// - B(1, 1) = 1
/// - B(a, 1) = 1/a
/// - B(1, b) = 1/b
/// - B(a, b) = B(a+1, b) + B(a, b+1)
///
/// # Arguments
/// * `a` - First parameter array (must be > 0)
/// * `b` - Second parameter array (must be > 0, same length as `a`)
///
/// # Returns
/// * Array with `B(a[i], b[i])` for each pair
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::beta_simd;
/// use ndarray::{array, ArrayView1};
///
/// let a = array![1.0_f64, 2.0, 0.5];
/// let b = array![1.0_f64, 2.0, 0.5];
/// let result = beta_simd(&a.view(), &b.view());
/// assert!((result[0] - 1.0).abs() < 1e-10);       // B(1,1) = 1
/// assert!((result[1] - 1.0/6.0).abs() < 1e-10);  // B(2,2) = 1/6
/// assert!((result[2] - std::f64::consts::PI).abs() < 1e-8);  // B(0.5,0.5) = π
/// ```
///
/// # Applications
///
/// - **Beta Distribution**: PDF = x^(a-1)(1-x)^(b-1) / B(a,b)
/// - **Bayesian Statistics**: Prior/posterior for probability parameters
/// - **A/B Testing**: Conversion rate analysis
/// - **Machine Learning**: Dirichlet processes, topic modeling
pub fn beta_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_beta(a, b);
    }
    F::simd_beta(a, b)
}
/// Compute the Log-Beta function ln(B(a, b)) using SIMD operations
///
/// The Log-Beta function is defined as:
/// ln(B(a, b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
///
/// This is more numerically stable than computing B(a,b) directly,
/// especially for large arguments where Γ would overflow.
///
/// # Mathematical Properties
///
/// - ln(B(a, b)) = ln(B(b, a)) (symmetric)
/// - ln(B(1, 1)) = 0
/// - ln(B(a, 1)) = -ln(a)
/// - For large a,b: ln(B(a,b)) ≈ 0.5*ln(2π) - (a+b-0.5)*ln(a+b) + (a-0.5)*ln(a) + (b-0.5)*ln(b)
///
/// # Arguments
/// * `a` - First parameter array (must be > 0)
/// * `b` - Second parameter array (must be > 0, same length as `a`)
///
/// # Returns
/// * Array with `ln(B(a[i], b[i]))` for each pair
///
/// # Example
/// ```ignore
/// use scirs2_core::ndarray_ext::elementwise::ln_beta_simd;
/// use scirs2_core::ndarray::{array, ArrayView1};
///
/// let a = array![1.0_f64, 2.0, 10.0];
/// let b = array![1.0_f64, 2.0, 10.0];
/// let result = ln_beta_simd(&a.view(), &b.view());
/// assert!((result[0] - 0.0_f64).abs() < 1e-10);  // ln(B(1,1)) = ln(1) = 0
/// assert!((result[1] - (-6.0_f64).ln()).abs() < 1e-10);  // ln(B(2,2)) = ln(1/6)
/// ```
///
/// # Applications
///
/// - **Numerical Stability**: Avoiding overflow in Beta distribution computations
/// - **Log-likelihood**: Direct computation of log-probabilities
/// - **Monte Carlo Methods**: Log-probability computations
/// - **Variational Inference**: KL divergence between Beta distributions
pub fn ln_beta_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_ln_beta(a, b);
    }
    F::simd_ln_beta(a, b)
}
/// SIMD-accelerated linear interpolation
///
/// Computes element-wise linear interpolation: lerp(a, b, t) = a + t * (b - a)
/// When t=0, returns a; when t=1, returns b.
///
/// # Arguments
/// * `a` - Start values
/// * `b` - End values
/// * `t` - Interpolation parameter (typically in [0, 1])
///
/// # Returns
/// Array of interpolated values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::lerp_simd;
///
/// let a = array![0.0_f32, 0.0, 0.0];
/// let b = array![10.0_f32, 20.0, 30.0];
///
/// // t = 0: returns a
/// let result = lerp_simd::<f32>(&a.view(), &b.view(), 0.0);
/// assert!((result[0] - 0.0).abs() < 1e-6);
///
/// // t = 1: returns b
/// let result = lerp_simd::<f32>(&a.view(), &b.view(), 1.0);
/// assert!((result[0] - 10.0).abs() < 1e-6);
///
/// // t = 0.5: midpoint
/// let result = lerp_simd::<f32>(&a.view(), &b.view(), 0.5);
/// assert!((result[0] - 5.0).abs() < 1e-6);
/// ```
///
/// # Use Cases
/// - Animation blending
/// - Color interpolation
/// - Smooth parameter transitions
/// - Gradient computation in neural networks
pub fn lerp_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>, t: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_lerp(a, b, t);
    }
    F::simd_lerp(a, b, t)
}
/// SIMD-accelerated smoothstep interpolation
///
/// Returns smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1:
/// - Returns 0 if x <= edge0
/// - Returns 1 if x >= edge1
/// - Returns smooth curve: 3t² - 2t³ where t = (x - edge0) / (edge1 - edge0)
///
/// # Arguments
/// * `edge0` - Lower edge of the transition
/// * `edge1` - Upper edge of the transition
/// * `x` - Input values
///
/// # Returns
/// Array of smoothstep values in [0, 1]
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::smoothstep_simd;
///
/// let x = array![0.0_f32, 0.25, 0.5, 0.75, 1.0];
///
/// let result = smoothstep_simd::<f32>(0.0, 1.0, &x.view());
/// assert!((result[0] - 0.0).abs() < 1e-6);  // x=0 -> 0
/// assert!((result[2] - 0.5).abs() < 1e-6);  // x=0.5 -> 0.5
/// assert!((result[4] - 1.0).abs() < 1e-6);  // x=1 -> 1
/// ```
///
/// # Use Cases
/// - Shader programming (smooth transitions)
/// - Activation function variants
/// - Anti-aliasing
/// - Soft thresholding
pub fn smoothstep_simd<F>(edge0: F, edge1: F, x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_smoothstep(edge0, edge1, x);
    }
    F::simd_smoothstep(edge0, edge1, x)
}
/// SIMD-accelerated hypotenuse calculation
///
/// Computes element-wise hypotenuse: hypot(x, y) = sqrt(x² + y²)
/// Uses the standard library implementation which handles overflow/underflow correctly.
///
/// # Arguments
/// * `x` - First coordinate values
/// * `y` - Second coordinate values
///
/// # Returns
/// Array of hypotenuse values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::hypot_simd;
///
/// let x = array![3.0_f64, 5.0, 8.0];
/// let y = array![4.0_f64, 12.0, 15.0];
///
/// let result = hypot_simd::<f64>(&x.view(), &y.view());
/// assert!((result[0] - 5.0).abs() < 1e-14);   // 3-4-5 triangle
/// assert!((result[1] - 13.0).abs() < 1e-14);  // 5-12-13 triangle
/// assert!((result[2] - 17.0).abs() < 1e-14);  // 8-15-17 triangle
/// ```
///
/// # Use Cases
/// - Distance calculations in 2D
/// - Computing vector magnitudes
/// - Complex number modulus: |a+bi| = hypot(a, b)
/// - Graphics and physics simulations
pub fn hypot_simd<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() || y.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_hypot(x, y);
    }
    F::simd_hypot(x, y)
}
/// SIMD-accelerated copysign operation
///
/// Returns element-wise magnitude of x with the sign of y.
///
/// # Arguments
/// * `x` - Magnitude source values
/// * `y` - Sign source values
///
/// # Returns
/// Array where each element has the magnitude of x and sign of y
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::copysign_simd;
///
/// let x = array![1.0_f64, -2.0, 3.0, -4.0];
/// let y = array![-1.0_f64, 1.0, 1.0, -1.0];
///
/// let result = copysign_simd::<f64>(&x.view(), &y.view());
/// assert!((result[0] - (-1.0)).abs() < 1e-14);  // 1 with sign of -1 = -1
/// assert!((result[1] - 2.0).abs() < 1e-14);     // -2 with sign of 1 = 2
/// assert!((result[2] - 3.0).abs() < 1e-14);     // 3 with sign of 1 = 3
/// assert!((result[3] - (-4.0)).abs() < 1e-14); // -4 with sign of -1 = -4
/// ```
///
/// # Use Cases
/// - Sign manipulation in numerical algorithms
/// - Implementing special functions (reflection formula)
/// - Gradient sign propagation
pub fn copysign_simd<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() || y.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_copysign(x, y);
    }
    F::simd_copysign(x, y)
}
/// SIMD-accelerated smootherstep interpolation (Ken Perlin's improved version)
///
/// Returns smooth Hermite interpolation with second-order continuity.
/// Formula: 6t⁵ - 15t⁴ + 10t³ where t = (x - edge0) / (edge1 - edge0)
///
/// Unlike smoothstep, both the first AND second derivatives are zero at the boundaries,
/// making it ideal for procedural generation and high-quality animations.
///
/// # Arguments
/// * `edge0` - Lower edge of the transition
/// * `edge1` - Upper edge of the transition
/// * `x` - Input values
///
/// # Returns
/// Array of smootherstep values in [0, 1]
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::smootherstep_simd;
///
/// let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];
///
/// let result = smootherstep_simd::<f64>(0.0, 1.0, &x.view());
/// assert!((result[0] - 0.0).abs() < 1e-14);  // x=0 -> 0
/// assert!((result[2] - 0.5).abs() < 1e-14);  // x=0.5 -> 0.5
/// assert!((result[4] - 1.0).abs() < 1e-14);  // x=1 -> 1
/// ```
///
/// # Use Cases
/// - Perlin noise and procedural generation
/// - High-quality animation easing
/// - Shader programming (smooth lighting transitions)
/// - Gradient-based optimization (smoother loss landscapes)
pub fn smootherstep_simd<F>(edge0: F, edge1: F, x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_smootherstep(edge0, edge1, x);
    }
    F::simd_smootherstep(edge0, edge1, x)
}
/// SIMD-accelerated logaddexp: log(exp(a) + exp(b))
///
/// Computes the logarithm of the sum of exponentials in a numerically stable way.
/// Uses the identity: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
///
/// # Arguments
/// * `a` - First input values
/// * `b` - Second input values
///
/// # Returns
/// Array of log(exp(a) + exp(b)) values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::logaddexp_simd;
///
/// let a = array![0.0_f64, 1.0, 2.0];
/// let b = array![0.0_f64, 1.0, 2.0];
///
/// let result = logaddexp_simd::<f64>(&a.view(), &b.view());
/// // log(exp(0) + exp(0)) = log(2) ≈ 0.693
/// assert!((result[0] - 2.0_f64.ln()).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Log-probability computations (Bayesian inference)
/// - Log-likelihood calculations in ML
/// - Hidden Markov Model algorithms
/// - Cross-entropy loss functions
pub fn logaddexp_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_logaddexp(a, b);
    }
    F::simd_logaddexp(a, b)
}
/// SIMD-accelerated logit function: log(p / (1-p))
///
/// The logit function maps probabilities in (0, 1) to log-odds in (-∞, +∞).
/// It is the inverse of the sigmoid (logistic) function.
///
/// # Arguments
/// * `a` - Probability values in (0, 1)
///
/// # Returns
/// Array of log-odds values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::logit_simd;
///
/// let p = array![0.5_f64, 0.1, 0.9];
///
/// let result = logit_simd::<f64>(&p.view());
/// // logit(0.5) = log(1) = 0
/// assert!((result[0] - 0.0).abs() < 1e-14);
/// // logit(0.1) = log(0.1/0.9) ≈ -2.197
/// assert!((result[1] - (0.1_f64 / 0.9).ln()).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Logistic regression
/// - Probability calibration
/// - Converting probabilities to unbounded space
/// - Statistical modeling (link functions)
pub fn logit_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_logit(a);
    }
    F::simd_logit(a)
}
/// SIMD-accelerated element-wise square: x²
///
/// Computes the square of each element. This is more efficient than `pow(x, 2)`
/// since it avoids the overhead of general exponentiation.
///
/// # Arguments
/// * `a` - Input values
///
/// # Returns
/// Array of squared values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::square_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, -4.0];
///
/// let result = square_simd::<f64>(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-14);   // 1² = 1
/// assert!((result[1] - 4.0).abs() < 1e-14);   // 2² = 4
/// assert!((result[2] - 9.0).abs() < 1e-14);   // 3² = 9
/// assert!((result[3] - 16.0).abs() < 1e-14);  // (-4)² = 16
/// ```
///
/// # Use Cases
/// - Computing squared distances
/// - Variance calculations (sum of squared deviations)
/// - Loss functions (MSE, L2 regularization)
/// - Polynomial evaluation
pub fn square_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_square(a);
    }
    F::simd_square(a)
}

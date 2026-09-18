//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::numeric::Float;
use crate::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use ::ndarray::{Array1, ArrayView1};

/// Compute the absolute value of each element (SIMD-accelerated).
///
/// Computes |x| for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with absolute values.
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
/// abs(x) = |x| = {
///     x   if x >= 0
///     -x  if x < 0
/// }
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::abs_simd;
///
/// let x = array![-3.0, -1.5, 0.0, 1.5, 3.0];
/// let result = abs_simd(&x.view());
///
/// assert_eq!(result[0], 3.0);
/// assert_eq!(result[1], 1.5);
/// assert_eq!(result[2], 0.0);
/// assert_eq!(result[3], 1.5);
/// assert_eq!(result[4], 3.0);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: Returns zero
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Infinity**: Returns positive infinity
///
/// # Applications
///
/// - **Statistics**: Absolute deviation, MAE (Mean Absolute Error)
/// - **Signal Processing**: Envelope detection, magnitude calculation
/// - **Optimization**: L1 norm computation, absolute loss functions
/// - **Numerical Analysis**: Error analysis, residual computation
/// - **Image Processing**: Gradient magnitude, difference images
pub fn abs_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_abs(x);
    }
    x.mapv(|val| val.abs())
}
/// Compute the sign (signum) of each element (SIMD-accelerated).
///
/// Returns +1.0 for positive values, -1.0 for negative values, and 0.0 for zero.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the sign of the input.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD unavailable
/// - **Speedup**: 2-4x for large arrays on AVX2/NEON systems
///
/// # Mathematical Definition
///
/// ```text
/// sign(x) = { +1.0  if x > 0
///           {  0.0  if x = 0
///           { -1.0  if x < 0
/// ```
///
/// # Properties
///
/// - sign(-x) = -sign(x) (odd function)
/// - sign(0) = 0
/// - sign(x) * |x| = x
/// - sign(x) * sign(y) = sign(x * y)
/// - |sign(x)| ≤ 1 for all x
///
/// # Applications
///
/// - **Numerical Analysis**: Gradient descent direction, optimization algorithms
/// - **Signal Processing**: Phase detection, zero-crossing analysis
/// - **Physics Simulations**: Force direction, velocity direction
/// - **Machine Learning**: Feature engineering, binary classification features
/// - **Control Systems**: Error sign for PID controllers
/// - **Game Development**: Direction vectors, collision normals
/// - **Statistics**: Wilcoxon signed-rank test, sign test
/// - **Finance**: Market trend indicators (bull/bear)
/// - **Geometry**: Surface normal orientation, vector direction
/// - **Image Processing**: Edge direction, gradient orientation
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sign_simd;
///
/// let x = array![-3.0_f64, -1.5, 0.0, 1.5, 3.0];
/// let result = sign_simd(&x.view());
/// assert_eq!(result[0], -1.0_f64); // negative → -1
/// assert_eq!(result[1], -1.0_f64); // negative → -1
/// assert_eq!(result[2],  0.0_f64); // zero → 0
/// assert_eq!(result[3],  1.0_f64); // positive → +1
/// assert_eq!(result[4],  1.0_f64); // positive → +1
///
/// // Property: sign(x) * |x| = x
/// let values = array![-5.0_f64, -2.0, 0.0, 2.0, 5.0];
/// let signs = sign_simd(&values.view());
/// let abs_values = values.mapv(|x: f64| x.abs());
/// let reconstructed = signs * abs_values;
/// for i in 0..values.len() {
///     assert!((reconstructed[i] - values[i]).abs() < 1e-10);
/// }
/// ```
///
/// # See Also
///
/// - `abs_simd`: Magnitude without sign
/// - `clamp_simd`: Constrain values to a range
pub fn sign_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sign(x);
    }
    x.mapv(|val| {
        if val > F::zero() {
            F::one()
        } else if val < F::zero() {
            -F::one()
        } else {
            F::zero()
        }
    })
}
/// Compute the square root of each element (SIMD-accelerated).
///
/// Computes √x for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array (must contain non-negative values)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with square roots.
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
/// sqrt(x) = √x = y such that y² = x
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sqrt_simd;
///
/// let x = array![1.0_f64, 4.0, 9.0, 16.0, 25.0];
/// let result = sqrt_simd(&x.view());
///
/// assert_eq!(result[0], 1.0);
/// assert_eq!(result[1], 2.0);
/// assert_eq!(result[2], 3.0);
/// assert_eq!(result[3], 4.0);
/// assert_eq!(result[4], 5.0);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: Returns zero
/// - **Negative numbers**: Returns NaN (undefined in real numbers)
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Positive infinity**: Returns positive infinity
///
/// # Applications
///
/// - **Statistics**: Standard deviation, RMS (Root Mean Square)
/// - **Geometry**: Distance calculations, Euclidean metrics
/// - **Physics**: Velocity from kinetic energy, magnitude calculations
/// - **Machine Learning**: Gradient scaling, learning rate schedules
/// - **Image Processing**: Gaussian blur, distance transforms
///
/// # Note on Negative Values
///
/// Square root of negative numbers is undefined in real arithmetic.
/// The function will return NaN for negative inputs, following IEEE 754 standard.
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sqrt_simd;
///
/// let x = array![-1.0_f64];
/// let result = sqrt_simd(&x.view());
/// assert!(result[0].is_nan());
/// ```
pub fn sqrt_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sqrt(x);
    }
    x.mapv(|val| val.sqrt())
}
/// Compute the exponential (e^x) of each element (SIMD-accelerated).
///
/// Computes e^x for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with exponential values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// exp(x) = e^x where e ≈ 2.71828...
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::exp_simd;
///
/// let x = array![0.0_f64, 1.0, 2.0];
/// let result = exp_simd(&x.view());
///
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// assert!((result[1] - 2.718281828).abs() < 1e-9);
/// assert!((result[2] - 7.389056099).abs() < 1e-9);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: exp(0) = 1
/// - **Large positive**: May overflow to infinity
/// - **Large negative**: Approaches zero
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Machine Learning**: Softmax, sigmoid activation
/// - **Optimization**: Exponential decay, learning rate schedules
/// - **Probability**: Exponential distribution, Gaussian PDF
/// - **Neural Networks**: Attention mechanisms, transformer models
/// - **Reinforcement Learning**: Policy gradients, Q-learning
pub fn exp_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_exp(x);
    }
    x.mapv(|val| val.exp())
}
/// Compute the natural logarithm (ln(x)) of each element (SIMD-accelerated).
///
/// Computes ln(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array (must contain positive values)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with natural logarithm values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// ln(x) = log_e(x) = y such that e^y = x
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::ln_simd;
///
/// let x = array![1.0_f64, 2.718281828, 7.389056099];
/// let result = ln_simd(&x.view());
///
/// assert!((result[0] - 0.0).abs() < 1e-10);
/// assert!((result[1] - 1.0).abs() < 1e-9);
/// assert!((result[2] - 2.0).abs() < 1e-9);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **One**: ln(1) = 0
/// - **Zero**: Returns negative infinity
/// - **Negative numbers**: Returns NaN (undefined in reals)
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Positive infinity**: Returns positive infinity
///
/// # Applications
///
/// - **Machine Learning**: Log-likelihood, cross-entropy loss
/// - **Statistics**: Log-normal distribution, Shannon entropy
/// - **Optimization**: Log-barrier methods, logarithmic objectives
/// - **Automatic Differentiation**: Logarithmic derivatives
/// - **Information Theory**: Mutual information, KL divergence
///
/// # Note on Negative Values
///
/// Natural logarithm of negative numbers is undefined in real arithmetic.
/// The function will return NaN for negative inputs, following IEEE 754 standard.
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::ln_simd;
///
/// let x = array![-1.0_f64];
/// let result = ln_simd(&x.view());
/// assert!(result[0].is_nan());
/// ```
pub fn ln_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_ln(x);
    }
    x.mapv(|val| val.ln())
}
/// Compute the sine of each element (SIMD-accelerated).
///
/// Computes sin(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array (angles in radians)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with sine values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// sin(x) = opposite/hypotenuse in a right triangle
/// Periodic with period 2π: sin(x + 2π) = sin(x)
/// Range: [-1, 1]
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sin_simd;
/// use std::f64::consts::PI;
///
/// let x = array![0.0_f64, PI/6.0, PI/4.0, PI/2.0, PI];
/// let result = sin_simd(&x.view());
///
/// // sin(0) = 0, sin(π/2) = 1, sin(π) ≈ 0
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: sin(0) = 0
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Infinity**: Returns NaN (undefined for infinity)
///
/// # Applications
///
/// - **Signal Processing**: Wave generation, modulation, Fourier analysis
/// - **Computer Graphics**: Rotation matrices, smooth animations
/// - **Physics Simulations**: Oscillations, wave propagation
/// - **Audio**: Synthesizers, tone generation
/// - **Robotics**: Trajectory planning, inverse kinematics
pub fn sin_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sin(x);
    }
    x.mapv(|val| val.sin())
}
/// Compute the cosine of each element (SIMD-accelerated).
///
/// Computes cos(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array (angles in radians)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with cosine values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// cos(x) = adjacent/hypotenuse in a right triangle
/// Periodic with period 2π: cos(x + 2π) = cos(x)
/// Range: [-1, 1]
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::cos_simd;
/// use std::f64::consts::PI;
///
/// let x = array![0.0_f64, PI/3.0, PI/2.0, PI];
/// let result = cos_simd(&x.view());
///
/// // cos(0) = 1, cos(π/2) ≈ 0, cos(π) = -1
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: cos(0) = 1
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Infinity**: Returns NaN (undefined for infinity)
///
/// # Applications
///
/// - **Computer Vision**: Rotation matrices, coordinate transformations
/// - **Signal Processing**: Fourier transforms, filtering
/// - **Path Planning**: Dubins curves, trajectory optimization
/// - **Geometry**: Circle parametrization, spherical coordinates
/// - **Neural Networks**: Positional encoding (transformers)
pub fn cos_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_cos(x);
    }
    x.mapv(|val| val.cos())
}
/// Compute the tangent of each element (SIMD-accelerated).
///
/// Computes tan(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array (angles in radians)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with tangent values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// tan(x) = sin(x) / cos(x)
/// Periodic with period π: tan(x + π) = tan(x)
/// Range: (-∞, ∞)
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::tan_simd;
/// use std::f64::consts::PI;
///
/// let x = array![0.0_f64, PI/4.0, PI/6.0];
/// let result = tan_simd(&x.view());
///
/// // tan(0) = 0, tan(π/4) = 1
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: tan(0) = 0
/// - **π/2, 3π/2, ...**: Returns ±infinity (undefined at cos(x)=0)
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Infinity**: Returns NaN (undefined for infinity)
///
/// # Applications
///
/// - **Computer Graphics**: Perspective projection, field of view
/// - **Navigation**: Bearing calculations, angle determination
/// - **Physics**: Slope calculations, inclined planes
/// - **Image Processing**: Gradient direction, edge angles
/// - **Surveying**: Distance and height measurements
///
/// # Note on Singularities
///
/// Tangent has singularities at x = π/2 + nπ where n is any integer.
/// At these points, cos(x) = 0 and tan(x) approaches ±infinity.
pub fn tan_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_tan(x);
    }
    x.mapv(|val| val.tan())
}
/// Compute the power of each element with a scalar exponent (SIMD-accelerated).
///
/// Computes base^exp for each element in the base array.
///
/// # Arguments
///
/// * `base` - Input 1D array of base values
/// * `exp` - Scalar exponent value
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with power values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// powf(base, exp) = base^exp
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::powf_simd;
///
/// let base = array![2.0_f64, 3.0, 4.0, 5.0];
/// let result = powf_simd(&base.view(), 2.0);
///
/// // [4.0, 9.0, 16.0, 25.0]
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **x^0**: Returns 1 for any x (including 0^0 = 1 by convention)
/// - **x^1**: Returns x
/// - **0^negative**: Returns infinity
/// - **negative^non-integer**: Returns NaN
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Statistics**: Variance, standard deviation, moment calculations
/// - **Machine Learning**: Polynomial features, L2 regularization
/// - **Signal Processing**: Power spectral density, energy calculations
/// - **Physics**: Kinetic energy, gravitational potential
/// - **Finance**: Compound interest, exponential growth models
pub fn powf_simd<F>(base: &ArrayView1<F>, exp: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if base.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(base.len()) {
        return F::simd_powf(base, exp);
    }
    base.mapv(|val| val.powf(exp))
}
/// Compute the element-wise power with array exponents (SIMD-accelerated).
///
/// Computes `base[i]^exp[i]` for each pair of elements.
///
/// # Arguments
///
/// * `base` - Input 1D array of base values
/// * `exp` - Input 1D array of exponent values (must match base length)
///
/// # Returns
///
/// `Array1<F>` with the same length as inputs, with power values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// pow(base, exp)[i] = base[i]^exp[i]
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::pow_simd;
///
/// let base = array![2.0_f64, 3.0, 4.0, 5.0];
/// let exp = array![2.0, 3.0, 2.0, 1.0];
/// let result = pow_simd(&base.view(), &exp.view());
///
/// // [4.0, 27.0, 16.0, 5.0]
/// ```
///
/// # Edge Cases
///
/// - **Empty arrays**: Returns empty array
/// - **Mismatched lengths**: Panics (arrays must have same length)
/// - **x^0**: Returns 1 for any x
/// - **0^0**: Returns 1 by convention
/// - **0^negative**: Returns infinity
/// - **negative^non-integer**: Returns NaN
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Machine Learning**: Custom activation functions, attention mechanisms
/// - **Statistics**: Generalized power transformations
/// - **Optimization**: Power-law scaling, Pareto distributions
/// - **Signal Processing**: Non-linear transformations, compression
/// - **Physics**: Variable exponent models, fractal dimensions
pub fn pow_simd<F>(base: &ArrayView1<F>, exp: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    assert_eq!(
        base.len(),
        exp.len(),
        "Base and exponent arrays must have the same length"
    );
    if base.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(base.len()) {
        return F::simd_pow(base, exp);
    }
    base.iter()
        .zip(exp.iter())
        .map(|(&b, &e)| b.powf(e))
        .collect::<Vec<_>>()
        .into()
}
/// Compute the hyperbolic sine of each element (SIMD-accelerated).
///
/// Computes sinh(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with hyperbolic sine values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// sinh(x) = (e^x - e^(-x)) / 2
/// Range: (-∞, ∞)
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sinh_simd;
///
/// let x = array![0.0_f64, 1.0, -1.0];
/// let result = sinh_simd(&x.view());
///
/// // sinh(0) = 0, sinh(1) ≈ 1.175, sinh(-1) ≈ -1.175
/// assert!((result[0] - 0.0).abs() < 1e-10);
/// assert!((result[1] - 1.1752011936).abs() < 1e-9);
/// assert!((result[2] + 1.1752011936).abs() < 1e-9);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: sinh(0) = 0
/// - **Large positive**: May overflow to infinity
/// - **Large negative**: May overflow to negative infinity
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Neural Networks**: Tanh activation (related via tanh = sinh/cosh)
/// - **Numerical Integration**: Tanh-sinh quadrature
/// - **Physics**: Catenary curves, special relativity
/// - **Engineering**: Transmission line theory, heat transfer
/// - **Mathematics**: Hyperbolic geometry, complex analysis
pub fn sinh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sinh(x);
    }
    x.mapv(|val| val.sinh())
}
/// Compute the hyperbolic cosine of each element (SIMD-accelerated).
///
/// Computes cosh(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with hyperbolic cosine values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// cosh(x) = (e^x + e^(-x)) / 2
/// Range: [1, ∞) (always >= 1)
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::cosh_simd;
///
/// let x = array![0.0_f64, 1.0, -1.0];
/// let result = cosh_simd(&x.view());
///
/// // cosh(0) = 1, cosh(1) ≈ 1.543, cosh(-1) ≈ 1.543
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// assert!((result[1] - 1.5430806348).abs() < 1e-9);
/// assert!((result[2] - 1.5430806348).abs() < 1e-9);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: cosh(0) = 1
/// - **Symmetric**: cosh(-x) = cosh(x)
/// - **Large values**: May overflow to infinity
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Neural Networks**: Activation functions, normalization
/// - **Physics**: Wave propagation, special relativity
/// - **Engineering**: Cable suspension, arch design
/// - **Mathematics**: Hyperbolic identities (cosh² - sinh² = 1)
/// - **Numerical Methods**: Stability analysis
pub fn cosh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_cosh(x);
    }
    x.mapv(|val| val.cosh())
}
/// Compute the hyperbolic tangent of each element (SIMD-accelerated).
///
/// Computes tanh(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with hyperbolic tangent values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
/// Range: (-1, 1)
/// ```
///
/// # Examples
///
/// ```ignore
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::tanh_simd;
///
/// let x = array![0.0_f64, 1.0, -1.0, 10.0];
/// let result = tanh_simd(&x.view());
///
/// // tanh(0) = 0, tanh(1) ≈ 0.762, tanh(-1) ≈ -0.762, tanh(∞) → 1
/// assert!((result[0] - 0.0_f64).abs() < 1e-10);
/// assert!((result[1] - 0.7615941559_f64).abs() < 1e-9);
/// assert!((result[2] + 0.7615941559_f64).abs() < 1e-9);
/// assert!((result[3] - 1.0_f64).abs() < 1e-9); // tanh(10) ≈ 1
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: tanh(0) = 0
/// - **Asymptotic**: tanh(x) → ±1 as x → ±∞
/// - **Anti-symmetric**: tanh(-x) = -tanh(x)
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Neural Networks**: Tanh activation function (classic activation)
/// - **Machine Learning**: Gradient clipping, normalization layers
/// - **Reinforcement Learning**: Policy networks, value functions
/// - **Signal Processing**: Soft limiting, saturation
/// - **Optimization**: Smooth approximation to sign function
/// - **Physics**: Relativistic velocity addition
///
/// # Note on Neural Networks
///
/// Tanh was historically the most popular activation function before ReLU.
/// It's still widely used in RNNs, LSTMs, and GRUs for gating mechanisms.
/// Gradient: d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
pub fn tanh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_tanh(x);
    }
    x.mapv(|val| val.tanh())
}
/// Compute the floor (round down) of each element (SIMD-accelerated).
///
/// Computes the largest integer less than or equal to x for each element.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is rounded down
/// to the nearest integer.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - For any x: floor(x) <= x
/// - floor(x) is the largest integer <= x
/// - floor(-x) = -ceil(x)
/// - floor(x) = x if x is already an integer
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::floor_simd;
///
/// let x = array![1.2, 2.7, -1.3, -2.9, 3.0];
/// let result = floor_simd(&x.view());
/// // Result: [1.0, 2.0, -2.0, -3.0, 3.0]
/// ```
///
/// # Applications
///
/// - **Binning**: Discretizing continuous values into bins
/// - **Indexing**: Converting continuous coordinates to discrete indices
/// - **Quantization**: Reducing precision for data compression
/// - **Digital Signal Processing**: Sample rate conversion, downsampling
/// - **Computer Graphics**: Pixel coordinate calculations
/// - **Financial**: Rounding down monetary amounts
///
/// # See Also
///
/// - `ceil_simd`: Round up to smallest integer >= x
/// - `round_simd`: Round to nearest integer
pub fn floor_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_floor(x);
    }
    x.mapv(|val| val.floor())
}

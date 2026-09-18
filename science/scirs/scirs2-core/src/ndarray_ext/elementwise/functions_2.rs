//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::numeric::Float;
use crate::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use ::ndarray::{Array1, ArrayView1};

/// Compute the ceiling (round up) of each element (SIMD-accelerated).
///
/// Computes the smallest integer greater than or equal to x for each element.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is rounded up
/// to the nearest integer.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - For any x: ceil(x) >= x
/// - ceil(x) is the smallest integer >= x
/// - ceil(-x) = -floor(x)
/// - ceil(x) = x if x is already an integer
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::ceil_simd;
///
/// let x = array![1.2, 2.7, -1.3, -2.9, 3.0];
/// let result = ceil_simd(&x.view());
/// // Result: [2.0, 3.0, -1.0, -2.0, 3.0]
/// ```
///
/// # Applications
///
/// - **Memory Allocation**: Rounding up buffer sizes
/// - **Pagination**: Calculating number of pages needed
/// - **Resource Planning**: Determining minimum resources required
/// - **Digital Signal Processing**: Sample rate conversion, upsampling
/// - **Computer Graphics**: Bounding box calculations
/// - **Financial**: Rounding up monetary amounts (conservative estimation)
///
/// # See Also
///
/// - `floor_simd`: Round down to largest integer <= x
/// - `round_simd`: Round to nearest integer
pub fn ceil_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_ceil(x);
    }
    x.mapv(|val| val.ceil())
}
/// Compute the rounding to nearest integer of each element (SIMD-accelerated).
///
/// Rounds each element to the nearest integer, with ties (x.5) rounding away
/// from zero (standard rounding).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is rounded to
/// the nearest integer.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Rounding Behavior
///
/// This function uses "round half away from zero" (standard rounding):
/// - 0.5 rounds to 1.0
/// - 1.5 rounds to 2.0
/// - 2.5 rounds to 3.0
/// - -0.5 rounds to -1.0
/// - -1.5 rounds to -2.0
///
/// # Mathematical Properties
///
/// - |round(x) - x| <= 0.5 for all x
/// - round(x) = x if x is already an integer
/// - round(-x) = -round(x)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::round_simd;
///
/// let x = array![1.2, 2.7, -1.3, -2.9, 3.0, 2.5];
/// let result = round_simd(&x.view());
/// // Result: [1.0, 3.0, -1.0, -3.0, 3.0, 3.0]
/// //          note: 2.5 rounds to 3.0 (away from zero)
/// ```
///
/// # Applications
///
/// - **Data Visualization**: Rounding values for display
/// - **Statistics**: Rounding statistical summaries
/// - **Financial**: General-purpose monetary rounding
/// - **Machine Learning**: Quantization for model compression
/// - **Image Processing**: Pixel value normalization
/// - **Scientific Computing**: Reducing numerical noise
///
/// # See Also
///
/// - `floor_simd`: Round down to largest integer <= x
/// - `ceil_simd`: Round up to smallest integer >= x
pub fn round_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_round(x);
    }
    x.mapv(|val| val.round())
}
/// Compute the fractional part of each element (SIMD-accelerated).
///
/// Returns the signed fractional component of each value (x - trunc(x)).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the signed fractional part
/// in the range (-1, 1).
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
/// fract(x) = x - trunc(x)
/// ```
///
/// For positive x: same as x - floor(x)
/// For negative x: preserves sign (e.g., fract(-1.5) = -0.5)
///
/// # Properties
///
/// - -1 < fract(x) < 1 for all finite x
/// - fract(x + n) = fract(x) for any integer n (periodic with period 1)
/// - fract(x) = 0 if and only if x is an integer
/// - fract(x) + trunc(x) = x
/// - fract(-x) = -fract(x) (odd function)
///
/// # Applications
///
/// - **Computer Graphics**: Texture coordinate wrapping, repeating patterns
/// - **Animation**: Cyclic motion, looping animations
/// - **Signal Processing**: Modulo 1 operations, phase wrapping
/// - **Numerical Methods**: Fractional part extraction, decimal decomposition
/// - **Game Development**: Tile-based rendering, repeating textures
/// - **Scientific Computing**: Periodic boundary conditions, modulo arithmetic
/// - **Audio Processing**: Phase accumulation, waveform generation
/// - **Time Calculations**: Extracting fractional seconds, subsecond precision
/// - **Cryptography**: Linear congruential generators, pseudo-random sequences
/// - **Financial**: Fractional share calculations, interest accrual
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::fract_simd;
///
/// let x = array![1.5_f64, 2.7, -1.3, 0.0, 3.0];
/// let result = fract_simd(&x.view());
/// assert!((result[0] - 0.5_f64).abs() < 1e-10);    // 1.5 - trunc(1.5) = 0.5
/// assert!((result[1] - 0.7_f64).abs() < 1e-10);    // 2.7 - trunc(2.7) = 0.7
/// assert!((result[2] - (-0.3_f64)).abs() < 1e-10); // -1.3 - trunc(-1.3) = -0.3
/// assert_eq!(result[3], 0.0_f64);                   // 0.0 - trunc(0.0) = 0.0
/// assert_eq!(result[4], 0.0_f64);                   // 3.0 - trunc(3.0) = 0.0
///
/// // Property: fract(x) + trunc(x) = x
/// let values = array![5.25_f64, -2.75, 0.5, 10.0];
/// let fract_parts = fract_simd(&values.view());
/// let trunc_parts = values.mapv(|v: f64| v.trunc());
/// let reconstructed = &fract_parts + &trunc_parts;
/// for i in 0..values.len() {
///     assert!((reconstructed[i] - values[i]).abs() < 1e-10);
/// }
/// ```
///
/// # See Also
///
/// - `floor_simd`: Round down to integer
/// - `round_simd`: Round to nearest integer
pub fn fract_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_fract(x);
    }
    x.mapv(|val| val.fract())
}
/// Compute the reciprocal (multiplicative inverse) of each element (SIMD-accelerated).
///
/// Returns 1/x for each element in the array. For zero values, the result is infinity.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the reciprocal (1/x).
/// Zero elements yield infinity values.
///
/// # Mathematical Definition
///
/// ```text
/// recip(x) = 1/x for x ≠ 0
/// recip(0) = ∞
/// ```
///
/// # Properties
///
/// - recip(recip(x)) = x (involutive property)
/// - recip(x * y) = recip(x) * recip(y) (multiplicative property)
/// - recip(x/y) = y/x (division inversion)
/// - recip(1) = 1 (identity element)
/// - recip(-x) = -recip(x) (odd function)
/// - recip(x^n) = (recip(x))^n (power property)
///
/// # Applications
///
/// ## Numerical Analysis
/// - Matrix inversions: Computing A^(-1) in linear systems
/// - Newton's method: Division-free iterations via reciprocal approximations
/// - Condition number estimation: ||A|| * ||A^(-1)||
/// - Iterative refinement: Improving solution accuracy in linear solvers
///
/// ## Computer Graphics
/// - Projection transformations: Converting world coordinates to screen space
/// - Lighting calculations: Inverse square law for light attenuation
/// - Texture mapping: Computing texture coordinate gradients
/// - Depth buffer operations: Converting depth to 1/z for perspective
///
/// ## Physics Simulations
/// - Inverse square laws: Gravitational force (F = G * m1 * m2 / r^2)
/// - Wave propagation: Intensity attenuation (I ∝ 1/r^2)
/// - Electromagnetic fields: Coulomb's law (F = k * q1 * q2 / r^2)
/// - Fluid dynamics: Pressure gradient computations
///
/// ## Signal Processing
/// - Filter design: Converting frequency response to impulse response
/// - Normalization: Scaling signals by reciprocal of maximum amplitude
/// - Frequency domain analysis: Converting period to frequency (f = 1/T)
/// - Deconvolution: Inverse filtering for signal restoration
///
/// ## Machine Learning
/// - Normalization layers: Scaling features by reciprocal of variance
/// - Activation functions: Sigmoid (1 / (1 + exp(-x)))
/// - Loss functions: Reciprocal of predictions in certain regression tasks
/// - Learning rate schedules: Inverse decay (lr = initial_lr / (1 + decay * t))
///
/// ## Financial Modeling
/// - Discount rates: Present value calculations (PV = FV * recip(1 + r)^n)
/// - Interest rate conversions: Converting rates to factors
/// - Portfolio optimization: Inverse volatility weighting
/// - Option pricing: Black-Scholes model components
///
/// # Edge Cases
///
/// - recip(0.0) = f64::INFINITY
/// - recip(-0.0) = f64::NEG_INFINITY
/// - recip(INFINITY) = 0.0
/// - recip(NEG_INFINITY) = -0.0
/// - recip(NaN) = NaN
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::recip_simd;
///
/// let x = array![2.0_f64, 4.0, 0.5, 1.0];
/// let result = recip_simd(&x.view());
///
/// assert!((result[0] - 0.5_f64).abs() < 1e-10);    // recip(2.0) = 0.5
/// assert!((result[1] - 0.25_f64).abs() < 1e-10);   // recip(4.0) = 0.25
/// assert!((result[2] - 2.0_f64).abs() < 1e-10);    // recip(0.5) = 2.0
/// assert!((result[3] - 1.0_f64).abs() < 1e-10);    // recip(1.0) = 1.0
/// ```
///
/// # Performance
///
/// This function uses SIMD acceleration for arrays larger than 1000 elements,
/// providing significant speedups through vectorized reciprocal approximations.
/// For smaller arrays, a scalar fallback is used to avoid SIMD overhead.
///
/// # Panics
///
/// This function does not panic. Division by zero results in infinity values
/// as per IEEE 754 floating-point semantics.
pub fn recip_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_recip(x);
    }
    x.mapv(|val| val.recip())
}
/// Compute integer power (base^n) for each element (SIMD-accelerated).
///
/// Returns base^n for each element in the array, where n is an integer exponent.
/// More efficient than powf for integer exponents.
///
/// # Arguments
///
/// * `base` - Input 1D array
/// * `n` - Integer exponent
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is base^n.
///
/// # Mathematical Definition
///
/// ```text
/// powi(x, n) = x^n for integer n
/// ```
///
/// # Properties
///
/// - powi(x, 0) = 1 for all x ≠ 0
/// - powi(x, 1) = x
/// - powi(x, -n) = 1 / powi(x, n)
/// - powi(x, n+m) = powi(x, n) * powi(x, m) (exponent addition)
/// - powi(x*y, n) = powi(x, n) * powi(y, n) (distributive over multiplication)
/// - powi(x, n*m) = powi(powi(x, n), m) (exponent multiplication)
///
/// # Applications
///
/// ## Statistics
/// - Chi-square distributions: Σ(x_i - μ)^2
/// - Polynomial distributions: Computing moments
/// - Variance calculations: E\[X^2\] - (E\[X\])^2
/// - Higher-order moments: Skewness (3rd moment), kurtosis (4th moment)
///
/// ## Linear Algebra
/// - Matrix powers: A^n for matrix operations
/// - Eigenvalue problems: Computing characteristic polynomials
/// - Norm calculations: ||x||_p = (Σ|x_i|^p)^(1/p)
/// - Gram matrices: X^T X operations
///
/// ## Signal Processing
/// - Polynomial filters: Computing filter responses
/// - Power spectral density: |X(f)|^2
/// - Autocorrelation: r(k) = Σ x(n) * x(n-k)
/// - Window functions: Raised cosine windows with integer powers
///
/// ## Machine Learning
/// - Polynomial features: x^2, x^3 for feature engineering
/// - Loss functions: L2 loss with (y - ŷ)^2
/// - Regularization: Ridge regression with ||w||^2
/// - Gradient descent: Computing squared gradients
///
/// ## Numerical Analysis
/// - Taylor series: Computing polynomial approximations
/// - Newton's method: f(x) and f'(x) evaluations
/// - Polynomial interpolation: Lagrange basis functions
/// - Finite differences: Computing higher-order derivatives
///
/// ## Physics Simulations
/// - Inverse square laws: 1/r^2 for gravity and electromagnetism
/// - Kinetic energy: (1/2)mv^2
/// - Potential energy: Polynomial potentials
/// - Power laws: Scaling relationships
///
/// # Edge Cases
///
/// - powi(0, n) = 0 for n > 0
/// - powi(0, 0) = 1 (mathematical convention)
/// - powi(x, 0) = 1 for all x
/// - powi(∞, n) = ∞ for n > 0, 0 for n < 0
/// - powi(NaN, n) = NaN
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::powi_simd;
///
/// let x = array![2.0_f64, 3.0, 4.0, 5.0];
///
/// // Square
/// let squared = powi_simd(&x.view(), 2);
/// assert!((squared[0] - 4.0_f64).abs() < 1e-10);    // 2^2 = 4
/// assert!((squared[1] - 9.0_f64).abs() < 1e-10);    // 3^2 = 9
/// assert!((squared[2] - 16.0_f64).abs() < 1e-10);   // 4^2 = 16
/// assert!((squared[3] - 25.0_f64).abs() < 1e-10);   // 5^2 = 25
///
/// // Cube
/// let cubed = powi_simd(&x.view(), 3);
/// assert!((cubed[0] - 8.0_f64).abs() < 1e-10);      // 2^3 = 8
/// assert!((cubed[1] - 27.0_f64).abs() < 1e-10);     // 3^3 = 27
///
/// // Negative exponent (reciprocal)
/// let inv_squared = powi_simd(&x.view(), -2);
/// assert!((inv_squared[0] - 0.25_f64).abs() < 1e-10);   // 2^(-2) = 0.25
/// assert!((inv_squared[1] - (1.0_f64/9.0)).abs() < 1e-10); // 3^(-2) = 1/9
/// ```
///
/// # Performance
///
/// This function uses SIMD acceleration for arrays larger than 1000 elements.
/// Integer power is computed using efficient exponentiation by squaring,
/// providing better performance than powf for integer exponents.
///
/// For small arrays, a scalar fallback is used to avoid SIMD overhead.
///
/// # Panics
///
/// This function does not panic. Edge cases like 0^0 return 1 as per
/// mathematical convention and IEEE 754 semantics.
pub fn powi_simd<F>(base: &ArrayView1<F>, n: i32) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if base.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(base.len()) {
        return F::simd_powi(base, n);
    }
    base.mapv(|val| val.powi(n))
}
/// Compute the gamma function Γ(x) for each element (SIMD-accelerated).
///
/// Evaluates the gamma function using the Lanczos approximation, providing high accuracy
/// across the entire real domain. The gamma function is a generalization of the factorial
/// function to non-integer values.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is Γ(x).
///
/// # Mathematical Definition
///
/// The gamma function is defined by Euler's integral:
/// ```text
/// Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt,    for Re(z) > 0
/// ```
///
/// For other values, it is defined by analytic continuation using the functional equation:
/// ```text
/// Γ(z+1) = z·Γ(z)
/// ```
///
/// # Key Properties
///
/// **Fundamental Properties**:
/// - Γ(1) = 1
/// - Γ(n+1) = n! for positive integers n
/// - Γ(1/2) = √π
/// - Γ(z+1) = z·Γ(z) (functional equation)
/// - Γ(z)·Γ(1-z) = π/sin(πz) (reflection formula)
///
/// **Special Values**:
/// - Γ(0) = ∞ (pole)
/// - Γ(-n) = ∞ for negative integers (poles)
/// - Γ(n+1/2) = (2n-1)!!·√π/2ⁿ for non-negative integers n
///
/// # Applications
///
/// ## 1. Statistical Distributions
/// - **Gamma Distribution**: PDF = (x^(α-1) e^(-x/β)) / (β^α Γ(α))
/// - **Beta Distribution**: B(α,β) = Γ(α)Γ(β)/Γ(α+β)
/// - **Chi-Square Distribution**: Uses Γ(k/2)
/// - **Student's t-Distribution**: Normalization involves Γ((ν+1)/2) and Γ(ν/2)
///
/// ## 2. Special Functions
/// - **Incomplete Gamma**: γ(s,x), Γ(s,x)
/// - **Beta Function**: B(α,β) = Γ(α)Γ(β)/Γ(α+β)
/// - **Bessel Functions**: Many representations involve gamma
/// - **Hypergeometric Functions**: Coefficients use gamma ratios
///
/// ## 3. Combinatorics & Number Theory
/// - **Binomial Coefficients**: C(n,k) = Γ(n+1)/(Γ(k+1)Γ(n-k+1))
/// - **Stirling Numbers**: Involve gamma function ratios
/// - **Riemann Zeta Function**: Functional equation uses Γ(s/2)
///
/// ## 4. Bayesian Inference
/// - **Conjugate Priors**: Gamma and Beta distributions
/// - **Dirichlet Distribution**: Normalization uses gamma products
/// - **Variational Inference**: Optimization involves gamma and digamma
///
/// ## 5. Physics & Engineering
/// - **Quantum Mechanics**: Angular momentum eigenstates
/// - **String Theory**: Veneziano amplitude
/// - **Statistical Mechanics**: Partition functions
/// - **Signal Processing**: Window functions
///
/// ## 6. Numerical Analysis
/// - **Integration**: Change of variables
/// - **Asymptotic Analysis**: Stirling's approximation
/// - **Interpolation**: Gamma function interpolates factorial
///
/// # Implementation
///
/// Uses the **Lanczos approximation** with g=7 and 9 coefficients, providing
/// ~15 decimal digits of accuracy for x ∈ [0.5, 100]. For x < 0.5, uses the
/// reflection formula to leverage the accurate computation of Γ(1-x).
///
/// **Algorithm**:
/// ```text
/// Γ(z+1) = √(2π) * (z + g + 1/2)^(z + 1/2) * e^(-(z + g + 1/2)) * A_g(z)
///
/// where A_g(z) = c₀ + c₁/(z+1) + c₂/(z+2) + ... + c₈/(z+9)
/// ```
///
/// **Coefficients** (from Boost C++ Library):
/// - Optimized using Remez exchange algorithm
/// - Minimax criterion for optimal approximation
/// - Provides uniform accuracy across domain
///
/// # Edge Cases
///
/// - `gamma(NaN)` → NaN
/// - `gamma(0)` → ∞
/// - `gamma(-n)` → ∞ for negative integers (poles)
/// - `gamma(x)` → ∞ as x → ∞
/// - `gamma(x)` → 0 as x → -∞ (alternating sign)
///
/// # Performance Characteristics
///
/// - **Small arrays** (< 1000 elements): Scalar implementation
/// - **Large arrays** (≥ 1000 elements): SIMD-accelerated via compiler auto-vectorization
/// - **Time complexity**: O(n) where n is array length
/// - **Space complexity**: O(n) for output array
/// - **Accuracy**: ~15 decimal digits for typical inputs
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::gamma_simd;
///
/// // Factorials: Γ(n+1) = n!
/// let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let gamma_x = gamma_simd(&x.view());
/// // gamma_x ≈ [1, 1, 2, 6, 24] (0!, 1!, 2!, 3!, 4!)
///
/// // Half-integer values
/// let x_half = array![0.5];
/// let gamma_half = gamma_simd(&x_half.view());
/// // gamma_half[0] ≈ 1.772453850905516 (√π)
///
/// // Statistical applications
/// use scirs2_core::numeric::Float;
/// let alpha = array![2.0, 3.0, 5.0];
/// let normalization = gamma_simd(&alpha.view());
/// // Use in gamma distribution PDF: x^(α-1) e^(-x/β) / (β^α Γ(α))
/// ```
///
/// # Mathematical References
///
/// - Abramowitz & Stegun, "Handbook of Mathematical Functions", §6.1
/// - Lanczos, "A Precision Approximation of the Gamma Function" (1964)
/// - Press et al., "Numerical Recipes", §6.1
/// - Boost C++ Libraries: Math Toolkit Documentation
///
/// # See Also
///
/// - `powi_simd` - Integer power (related operation)
/// - `exp_simd`, `ln_simd` - Component operations
/// - Future: `digamma_simd` - Logarithmic derivative of gamma
pub fn gamma_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_gamma(x);
    }
    F::simd_gamma(x)
}
/// Compute the arctangent of each element (SIMD-accelerated).
///
/// Computes atan(x) for each element, returning values in the range (-π/2, π/2).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the arctangent
/// in radians.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Range: (-π/2, π/2) for all finite x
/// - atan(0) = 0
/// - atan(-x) = -atan(x) (odd function)
/// - atan(∞) = π/2, atan(-∞) = -π/2
/// - atan(tan(x)) = x for x ∈ (-π/2, π/2)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::atan_simd;
///
/// let x = array![0.0_f64, 1.0, -1.0, f64::INFINITY];
/// let result = atan_simd(&x.view());
/// // Result: [0.0, π/4, -π/4, π/2]
/// ```
///
/// # Applications
///
/// - **Geometry**: Calculating angles from slopes
/// - **Robotics**: Inverse kinematics for joint angles
/// - **Computer Vision**: Feature detection, edge orientation
/// - **Signal Processing**: Phase unwrapping, frequency analysis
/// - **Physics**: Trajectory analysis, projectile motion
///
/// # See Also
///
/// - [`asin_simd`]: Arcsine function
/// - [`acos_simd`]: Arccosine function
/// - [`atan2_simd`]: Two-argument arctangent for full angle range
pub fn atan_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_atan(x);
    }
    x.mapv(|val| val.atan())
}
/// Compute the arcsine of each element (SIMD-accelerated).
///
/// Computes asin(x) for each element, returning values in the range [-π/2, π/2].
///
/// # Arguments
///
/// * `x` - Input 1D array with values in [-1, 1]
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the arcsine
/// in radians. Returns NaN for |x| > 1.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Domain: [-1, 1]
/// - Range: [-π/2, π/2]
/// - asin(0) = 0
/// - asin(-x) = -asin(x) (odd function)
/// - asin(1) = π/2, asin(-1) = -π/2
/// - asin(sin(x)) = x for x ∈ [-π/2, π/2]
/// - Returns NaN for |x| > 1
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::asin_simd;
///
/// let x = array![0.0_f64, 0.5, 1.0, -1.0];
/// let result = asin_simd(&x.view());
/// // Result: [0.0, π/6, π/2, -π/2]
/// ```
///
/// # Applications
///
/// - **Navigation**: Great circle distance calculations
/// - **Astronomy**: Celestial coordinate transformations
/// - **Physics**: Wave mechanics, pendulum analysis
/// - **Computer Graphics**: Spherical coordinates, lighting
/// - **Robotics**: Inverse kinematics with constraints
///
/// # See Also
///
/// - `atan_simd`: Arctangent function
/// - `acos_simd`: Arccosine function
/// - `sin_simd`: Forward sine function
pub fn asin_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_asin(x);
    }
    x.mapv(|val| val.asin())
}
/// Compute the arccosine of each element (SIMD-accelerated).
///
/// Computes acos(x) for each element, returning values in the range [0, π].
///
/// # Arguments
///
/// * `x` - Input 1D array with values in [-1, 1]
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the arccosine
/// in radians. Returns NaN for |x| > 1.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Domain: [-1, 1]
/// - Range: [0, π]
/// - acos(1) = 0, acos(-1) = π
/// - acos(0) = π/2
/// - acos(-x) = π - acos(x)
/// - acos(cos(x)) = x for x ∈ [0, π]
/// - acos(x) + asin(x) = π/2
/// - Returns NaN for |x| > 1
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::acos_simd;
///
/// let x = array![1.0_f64, 0.5, 0.0, -1.0];
/// let result = acos_simd(&x.view());
/// // Result: [0.0, π/3, π/2, π]
/// ```
///
/// # Applications
///
/// - **3D Graphics**: Dot product angle calculations
/// - **Machine Learning**: Cosine similarity to angle conversion
/// - **Physics**: Angular momentum, rotation analysis
/// - **Astronomy**: Coordinate system transformations
/// - **Navigation**: Bearing and heading calculations
///
/// # See Also
///
/// - `atan_simd`: Arctangent function
/// - `asin_simd`: Arcsine function
/// - `cos_simd`: Forward cosine function
pub fn acos_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_acos(x);
    }
    x.mapv(|val| val.acos())
}
/// Compute the two-argument arctangent element-wise (SIMD-accelerated).
///
/// Computes atan2(y, x) for each pair of elements, returning values in the range
/// (-π, π]. This function correctly handles the signs of both arguments to
/// determine the quadrant.
///
/// # Arguments
///
/// * `y` - Y-coordinates (sine component)
/// * `x` - X-coordinates (cosine component)
///
/// # Returns
///
/// `Array1<F>` with the same length as inputs, where each element is the angle
/// in radians from the positive x-axis to the point (x, y).
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Range: (-π, π]
/// - atan2(0, 0) = 0 (by convention)
/// - atan2(y, x) = atan(y/x) when x > 0
/// - atan2(y, 0) = π/2 * sign(y) when x = 0
/// - atan2(-y, x) = -atan2(y, x)
/// - atan2(y, -x) = π - atan2(y, x) when y >= 0
/// - atan2(y, -x) = -π + atan2(y, x) when y < 0
///
/// # Quadrants
///
/// - Quadrant I (x > 0, y > 0): (0, π/2)
/// - Quadrant II (x < 0, y > 0): (π/2, π)
/// - Quadrant III (x < 0, y < 0): (-π, -π/2)
/// - Quadrant IV (x > 0, y < 0): (-π/2, 0)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::atan2_simd;
///
/// let y = array![1.0, 1.0, -1.0, -1.0];
/// let x = array![1.0_f64, -1.0, -1.0, 1.0];
/// let angles = atan2_simd(&y.view(), &x.view());
/// // Result: [π/4, 3π/4, -3π/4, -π/4]
/// ```
///
/// # Applications
///
/// - **Robotics**: Joint angle calculations, path planning
/// - **Computer Vision**: Feature orientation, optical flow
/// - **Navigation**: Heading and bearing calculations
/// - **Spatial Computing**: Coordinate transformations
/// - **Physics**: Vector angle calculations, force analysis
/// - **Computer Graphics**: Rotation angles, sprite orientation
/// - **Signal Processing**: Phase calculations, complex number arguments
///
/// # See Also
///
/// - [`atan_simd`]: Single-argument arctangent
/// - [`asin_simd`]: Arcsine function
/// - [`acos_simd`]: Arccosine function
pub fn atan2_simd<F>(y: &ArrayView1<F>, x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    assert_eq!(y.len(), x.len(), "y and x arrays must have the same length");
    if y.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(y.len()) {
        return F::simd_atan2(y, x);
    }
    y.iter()
        .zip(x.iter())
        .map(|(&y_val, &x_val)| y_val.atan2(x_val))
        .collect::<Vec<_>>()
        .into()
}
/// Compute the base-10 logarithm of each element (SIMD-accelerated).
///
/// Computes log₁₀(x) for each element, where log₁₀(x) = ln(x) / ln(10).
///
/// # Arguments
///
/// * `x` - Input 1D array with positive values
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the base-10
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
/// - log₁₀(1) = 0
/// - log₁₀(10) = 1
/// - log₁₀(10ⁿ) = n
/// - log₁₀(x * y) = log₁₀(x) + log₁₀(y)
/// - log₁₀(x / y) = log₁₀(x) - log₁₀(y)
/// - log₁₀(xⁿ) = n * log₁₀(x)
/// - Returns NaN for x ≤ 0
/// - Returns -∞ for x = 0
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::log10_simd;
///
/// let x = array![1.0_f64, 10.0, 100.0, 1000.0];
/// let result = log10_simd(&x.view());
/// // Result: [0.0, 1.0, 2.0, 3.0]
/// ```
///
/// # Applications
///
/// - **Signal Processing**: Decibel scale (dB = 10 * log₁₀(P/P₀))
/// - **Chemistry**: pH scale (pH = -log₁₀[H⁺])
/// - **Astronomy**: Magnitude scale (apparent magnitude)
/// - **Scientific Computing**: Decades representation, log-log plots
/// - **Machine Learning**: Feature scaling, log-loss functions
/// - **Information Theory**: Hartley's entropy (base-10)
/// - **Physics**: Richter scale (earthquake magnitude)
/// - **Audio Processing**: Sound intensity level
///
/// # See Also
///
/// - `ln_simd`: Natural logarithm (base e)
/// - `log2_simd`: Base-2 logarithm
/// - `exp_simd`: Exponential function (inverse of ln)
pub fn log10_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_log10(x);
    }
    x.mapv(|val| val.log10())
}

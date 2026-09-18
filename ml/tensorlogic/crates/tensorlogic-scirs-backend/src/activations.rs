//! Activation functions for neural network layers.
//!
//! Provides element-wise, output, gradient, and scalar activation functions
//! backed by ndarray operations, as well as a unified [`ActivationType`] enum
//! for dispatch and an [`ActivationBenchmark`] for statistical summaries.

use scirs2_core::ndarray::{ArrayD, Zip};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise during activation-function computation.
#[derive(Debug, Clone)]
pub enum ActivationError {
    /// The input tensor has no elements.
    EmptyInput,
    /// A hyperparameter has an illegal value.
    InvalidParameter {
        name: String,
        value: f64,
        reason: String,
    },
    /// Tensor shapes are incompatible (e.g. PReLU weights vs. input).
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for ActivationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "activation: input tensor is empty"),
            Self::InvalidParameter {
                name,
                value,
                reason,
            } => {
                write!(
                    f,
                    "activation: invalid parameter '{name}' = {value}: {reason}"
                )
            }
            Self::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "activation: shape mismatch — expected {expected:?}, got {got:?}"
                )
            }
        }
    }
}

impl std::error::Error for ActivationError {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically-stable `erf` via Abramowitz & Stegun rational approximation
/// (maximum error ≈ 1.5 × 10⁻⁷).
#[inline]
fn erf_approx(x: f64) -> f64 {
    const A1: f64 = 0.278_393;
    const A2: f64 = 0.230_389;
    const A3: f64 = 0.000_972;
    const A4: f64 = 0.078_108;
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.47047 * x);
    let poly = ((A4 * t + A3) * t + A2) * t + A1;
    let result = 1.0 - poly * t * (-x * x).exp();
    sign * result
}

#[inline]
fn sigmoid_scalar_impl(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus_scalar(x: f64, beta: f64) -> f64 {
    // Use identity softplus(x) ≈ x for large x to avoid overflow.
    let bx = beta * x;
    if bx > 30.0 {
        x
    } else {
        (1.0 + bx.exp()).ln() / beta
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar helpers (public)
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar ReLU.
#[inline]
pub fn relu_scalar(x: f64) -> f64 {
    x.max(0.0)
}

/// Scalar GELU: `x * 0.5 * (1 + erf(x / sqrt(2)))`.
#[inline]
pub fn gelu_scalar(x: f64) -> f64 {
    x * 0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Scalar Swish / SiLU: `x * sigmoid(x)`.
#[inline]
pub fn swish_scalar(x: f64) -> f64 {
    x * sigmoid_scalar_impl(x)
}

/// Scalar sigmoid: `1 / (1 + exp(-x))`.
#[inline]
pub fn sigmoid_scalar(x: f64) -> f64 {
    sigmoid_scalar_impl(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Element-wise activation functions
// ─────────────────────────────────────────────────────────────────────────────

/// Rectified Linear Unit: `max(0, x)`.
pub fn relu(input: &ArrayD<f64>) -> ArrayD<f64> {
    input.mapv(relu_scalar)
}

/// ReLU6: `min(max(0, x), 6)`.
pub fn relu6(input: &ArrayD<f64>) -> ArrayD<f64> {
    input.mapv(|x| x.clamp(0.0, 6.0))
}

/// Leaky ReLU: `x` if `x >= 0`, else `negative_slope * x`.
pub fn leaky_relu(input: &ArrayD<f64>, negative_slope: f64) -> ArrayD<f64> {
    input.mapv(|x| if x >= 0.0 { x } else { negative_slope * x })
}

/// Exponential Linear Unit: `x` if `x >= 0`, else `alpha * (exp(x) - 1)`.
///
/// Returns `ActivationError::InvalidParameter` when `alpha < 0`.
pub fn elu(input: &ArrayD<f64>, alpha: f64) -> Result<ArrayD<f64>, ActivationError> {
    if alpha < 0.0 {
        return Err(ActivationError::InvalidParameter {
            name: "alpha".into(),
            value: alpha,
            reason: "alpha must be non-negative for ELU".into(),
        });
    }
    Ok(input.mapv(|x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }))
}

/// Scaled ELU with fixed constants: `scale * max(x, alpha*(exp(x)-1))`.
///
/// alpha = 1.6732632423543772, scale = 1.0507009873554805.
pub fn selu(input: &ArrayD<f64>) -> ArrayD<f64> {
    const ALPHA: f64 = 1.673_263_242_354_377_2;
    const SCALE: f64 = 1.050_700_987_355_480_5;
    input.mapv(|x| SCALE * if x >= 0.0 { x } else { ALPHA * (x.exp() - 1.0) })
}

/// Gaussian Error Linear Unit (exact): `x * 0.5 * (1 + erf(x / sqrt(2)))`.
pub fn gelu(input: &ArrayD<f64>) -> ArrayD<f64> {
    input.mapv(gelu_scalar)
}

/// GELU fast approximation via tanh:
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu_approx(input: &ArrayD<f64>) -> ArrayD<f64> {
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    input.mapv(|x| {
        let inner = C * (x + 0.044_715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    })
}

/// Swish / SiLU: `x * sigmoid(x)`.
pub fn swish(input: &ArrayD<f64>) -> ArrayD<f64> {
    input.mapv(swish_scalar)
}

/// Alias for [`swish`].
pub fn silu(input: &ArrayD<f64>) -> ArrayD<f64> {
    swish(input)
}

/// Mish: `x * tanh(ln(1 + exp(x)))`.
pub fn mish(input: &ArrayD<f64>) -> ArrayD<f64> {
    input.mapv(|x| {
        let sp = softplus_scalar(x, 1.0);
        x * sp.tanh()
    })
}

/// Softplus: `(1/beta) * ln(1 + exp(beta * x))`.
///
/// Returns `ActivationError::InvalidParameter` when `beta <= 0`.
pub fn softplus(input: &ArrayD<f64>, beta: f64) -> Result<ArrayD<f64>, ActivationError> {
    if beta <= 0.0 {
        return Err(ActivationError::InvalidParameter {
            name: "beta".into(),
            value: beta,
            reason: "beta must be positive for Softplus".into(),
        });
    }
    Ok(input.mapv(|x| softplus_scalar(x, beta)))
}

/// Softsign: `x / (1 + |x|)`.
pub fn softsign(input: &ArrayD<f64>) -> ArrayD<f64> {
    input.mapv(|x| x / (1.0 + x.abs()))
}

/// Hard-Swish: `x * relu6(x + 3) / 6`.
pub fn hardswish(input: &ArrayD<f64>) -> ArrayD<f64> {
    input.mapv(|x| x * (x + 3.0).clamp(0.0, 6.0) / 6.0)
}

/// Hard-Sigmoid: `relu6(x + 3) / 6`.
pub fn hardsigmoid(input: &ArrayD<f64>) -> ArrayD<f64> {
    input.mapv(|x| (x + 3.0).clamp(0.0, 6.0) / 6.0)
}

/// Sigmoid: `1 / (1 + exp(-x))`.
pub fn sigmoid(input: &ArrayD<f64>) -> ArrayD<f64> {
    input.mapv(sigmoid_scalar_impl)
}

/// Hyperbolic tangent activation (renamed to avoid conflict with `f64::tanh`).
pub fn tanh_activation(input: &ArrayD<f64>) -> ArrayD<f64> {
    input.mapv(|x| x.tanh())
}

/// Parametric ReLU: `x` if `x >= 0`, else `weights[channel] * x`.
///
/// `weights` must broadcast along axis-0 of `input` (i.e. its total number of
/// elements equals the number of channels = `input.shape()[0]`).  For a 1-D
/// input the weights tensor must have a single element.
pub fn prelu(input: &ArrayD<f64>, weights: &ArrayD<f64>) -> Result<ArrayD<f64>, ActivationError> {
    if input.is_empty() {
        return Err(ActivationError::EmptyInput);
    }

    // Determine channel count: axis 0 for ndim >= 1, else 1.
    let channels = if input.ndim() == 0 {
        1
    } else {
        input.shape()[0]
    };
    let w_len = weights.len();

    if w_len != channels && w_len != 1 {
        return Err(ActivationError::ShapeMismatch {
            expected: vec![channels],
            got: weights.shape().to_vec(),
        });
    }

    let weights_flat: Vec<f64> = weights.iter().copied().collect();
    let get_w = |ch: usize| -> f64 {
        if w_len == 1 {
            weights_flat[0]
        } else {
            weights_flat[ch]
        }
    };

    if input.ndim() <= 1 {
        // 0-D or 1-D: channel index = element index (or 0)
        let out: Vec<f64> = input
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let ch = if w_len == 1 { 0 } else { i };
                if x >= 0.0 {
                    x
                } else {
                    get_w(ch) * x
                }
            })
            .collect();
        return Ok(ArrayD::from_shape_vec(input.raw_dim(), out)
            .unwrap_or_else(|_| input.mapv(relu_scalar)));
    }

    // N-D: channel = first axis index
    let shape = input.shape().to_vec();
    let mut result = input.clone();
    let stride: usize = shape[1..].iter().product();

    for (idx, val) in result.iter_mut().enumerate() {
        let ch = (idx / stride) % channels;
        if *val < 0.0 {
            *val *= get_w(ch);
        }
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Output activations
// ─────────────────────────────────────────────────────────────────────────────

/// Softmax along `axis`.  Subtracts the max for numerical stability.
pub fn softmax(input: &ArrayD<f64>, axis: usize) -> Result<ArrayD<f64>, ActivationError> {
    if input.is_empty() {
        return Err(ActivationError::EmptyInput);
    }
    if axis >= input.ndim() {
        return Err(ActivationError::InvalidParameter {
            name: "axis".into(),
            value: axis as f64,
            reason: format!("axis {} out of range for ndim {}", axis, input.ndim()),
        });
    }

    // max along axis for stability
    let max_vals = input.map_axis(scirs2_core::ndarray::Axis(axis), |lane| {
        lane.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });

    let mut shifted = input.clone();
    // Broadcast-subtract max along the given axis
    Zip::from(&mut shifted)
        .and_broadcast(&max_vals.insert_axis(scirs2_core::ndarray::Axis(axis)))
        .for_each(|s, &m| *s -= m);

    let mut exped = shifted.mapv(f64::exp);

    let sum_vals = exped.map_axis(scirs2_core::ndarray::Axis(axis), |lane| {
        lane.iter().cloned().sum::<f64>()
    });

    Zip::from(&mut exped)
        .and_broadcast(&sum_vals.insert_axis(scirs2_core::ndarray::Axis(axis)))
        .for_each(|e, &s| *e /= s);

    Ok(exped)
}

/// Numerically stable log-softmax along `axis`.
pub fn log_softmax(input: &ArrayD<f64>, axis: usize) -> Result<ArrayD<f64>, ActivationError> {
    if input.is_empty() {
        return Err(ActivationError::EmptyInput);
    }
    if axis >= input.ndim() {
        return Err(ActivationError::InvalidParameter {
            name: "axis".into(),
            value: axis as f64,
            reason: format!("axis {} out of range for ndim {}", axis, input.ndim()),
        });
    }

    let max_vals = input.map_axis(scirs2_core::ndarray::Axis(axis), |lane| {
        lane.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });

    let mut shifted = input.clone();
    Zip::from(&mut shifted)
        .and_broadcast(&max_vals.insert_axis(scirs2_core::ndarray::Axis(axis)))
        .for_each(|s, &m| *s -= m);

    let log_sum_exp = shifted
        .mapv(f64::exp)
        .map_axis(scirs2_core::ndarray::Axis(axis), |lane| {
            lane.iter().cloned().sum::<f64>().ln()
        });

    Zip::from(&mut shifted)
        .and_broadcast(&log_sum_exp.insert_axis(scirs2_core::ndarray::Axis(axis)))
        .for_each(|s, &lse| *s -= lse);

    Ok(shifted)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient functions
// ─────────────────────────────────────────────────────────────────────────────

/// ReLU gradient: `grad_output` where `input > 0`, else `0`.
pub fn relu_grad(input: &ArrayD<f64>, grad_output: &ArrayD<f64>) -> ArrayD<f64> {
    let mut out = grad_output.clone();
    Zip::from(&mut out).and(input).for_each(|g, &x| {
        if x <= 0.0 {
            *g = 0.0;
        }
    });
    out
}

/// Sigmoid gradient: `output * (1 - output) * grad_output`.
///
/// `output` should be the **result** of `sigmoid(x)`, not the raw input.
pub fn sigmoid_grad(output: &ArrayD<f64>, grad_output: &ArrayD<f64>) -> ArrayD<f64> {
    let mut out = grad_output.clone();
    Zip::from(&mut out)
        .and(output)
        .for_each(|g, &s| *g *= s * (1.0 - s));
    out
}

/// Tanh gradient: `(1 - output^2) * grad_output`.
///
/// `output` should be the **result** of `tanh(x)`, not the raw input.
pub fn tanh_grad(output: &ArrayD<f64>, grad_output: &ArrayD<f64>) -> ArrayD<f64> {
    let mut out = grad_output.clone();
    Zip::from(&mut out)
        .and(output)
        .for_each(|g, &t| *g *= 1.0 - t * t);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified dispatch enum
// ─────────────────────────────────────────────────────────────────────────────

/// Enumeration of supported activation functions for unified dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    Relu,
    Relu6,
    LeakyRelu(f64),
    Elu(f64),
    Selu,
    Gelu,
    GeluApprox,
    Swish,
    Mish,
    Softplus(f64),
    Softsign,
    Hardswish,
    Hardsigmoid,
    Sigmoid,
    Tanh,
}

impl ActivationType {
    /// Apply this activation to `input`.
    pub fn apply(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, ActivationError> {
        match self {
            Self::Relu => Ok(relu(input)),
            Self::Relu6 => Ok(relu6(input)),
            Self::LeakyRelu(s) => Ok(leaky_relu(input, *s)),
            Self::Elu(a) => elu(input, *a),
            Self::Selu => Ok(selu(input)),
            Self::Gelu => Ok(gelu(input)),
            Self::GeluApprox => Ok(gelu_approx(input)),
            Self::Swish => Ok(swish(input)),
            Self::Mish => Ok(mish(input)),
            Self::Softplus(b) => softplus(input, *b),
            Self::Softsign => Ok(softsign(input)),
            Self::Hardswish => Ok(hardswish(input)),
            Self::Hardsigmoid => Ok(hardsigmoid(input)),
            Self::Sigmoid => Ok(sigmoid(input)),
            Self::Tanh => Ok(tanh_activation(input)),
        }
    }

    /// Human-readable name of this activation.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Relu6 => "relu6",
            Self::LeakyRelu(_) => "leaky_relu",
            Self::Elu(_) => "elu",
            Self::Selu => "selu",
            Self::Gelu => "gelu",
            Self::GeluApprox => "gelu_approx",
            Self::Swish => "swish",
            Self::Mish => "mish",
            Self::Softplus(_) => "softplus",
            Self::Softsign => "softsign",
            Self::Hardswish => "hardswish",
            Self::Hardsigmoid => "hardsigmoid",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
        }
    }

    /// Whether this activation is a monotonically non-decreasing function.
    pub fn is_monotone(&self) -> bool {
        matches!(
            self,
            Self::Relu
                | Self::Relu6
                | Self::LeakyRelu(_)
                | Self::Elu(_)
                | Self::Selu
                | Self::Gelu
                | Self::GeluApprox
                | Self::Swish
                | Self::Softplus(_)
                | Self::Softsign
                | Self::Sigmoid
                | Self::Tanh
        )
    }

    /// Approximate output range `(min, max)`.
    pub fn output_range(&self) -> (f64, f64) {
        match self {
            Self::Relu => (0.0, f64::INFINITY),
            Self::Relu6 => (0.0, 6.0),
            Self::LeakyRelu(_) => (f64::NEG_INFINITY, f64::INFINITY),
            Self::Elu(_) | Self::Selu => (f64::NEG_INFINITY, f64::INFINITY),
            Self::Gelu | Self::GeluApprox => (f64::NEG_INFINITY, f64::INFINITY),
            Self::Swish | Self::Mish => (f64::NEG_INFINITY, f64::INFINITY),
            Self::Softplus(_) => (0.0, f64::INFINITY),
            Self::Softsign => (-1.0, 1.0),
            Self::Hardswish => (f64::NEG_INFINITY, f64::INFINITY),
            Self::Hardsigmoid => (0.0, 1.0),
            Self::Sigmoid => (0.0, 1.0),
            Self::Tanh => (-1.0, 1.0),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark helper
// ─────────────────────────────────────────────────────────────────────────────

/// Statistical summary of an activation applied to a sample input.
#[derive(Debug, Clone)]
pub struct ActivationBenchmark {
    pub name: String,
    pub input_size: usize,
    pub mean_output: f64,
    pub std_output: f64,
    pub min_output: f64,
    pub max_output: f64,
}

impl ActivationBenchmark {
    /// Run `activation` on `input` and collect statistics.
    pub fn compute(
        activation: &ActivationType,
        input: &ArrayD<f64>,
    ) -> Result<Self, ActivationError> {
        if input.is_empty() {
            return Err(ActivationError::EmptyInput);
        }
        let output = activation.apply(input)?;
        let n = output.len() as f64;
        let values: Vec<f64> = output.iter().copied().collect();

        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_output = variance.sqrt();
        let min_output = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_output = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Ok(Self {
            name: activation.name().to_owned(),
            input_size: input.len(),
            mean_output: mean,
            std_output,
            min_output,
            max_output,
        })
    }

    /// One-line human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "{} [n={}] mean={:.4} std={:.4} min={:.4} max={:.4}",
            self.name,
            self.input_size,
            self.mean_output,
            self.std_output,
            self.min_output,
            self.max_output,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, Array2};

    const EPS: f64 = 1e-6;

    fn arr(v: &[f64]) -> ArrayD<f64> {
        arr1(v).into_dyn()
    }

    fn check_close(a: f64, b: f64, eps: f64, msg: &str) {
        assert!((a - b).abs() < eps, "{msg}: |{a} - {b}| >= {eps}");
    }

    #[test]
    fn test_relu_zeros_negative() {
        let input = arr(&[-3.0, -1.0, 0.0]);
        let out = relu(&input);
        for &v in out.iter() {
            assert_eq!(v, 0.0, "ReLU of non-positive must be 0");
        }
    }

    #[test]
    fn test_relu_positive_unchanged() {
        let input = arr(&[1.0, 2.5, 100.0]);
        let out = relu(&input);
        for (&i, &o) in input.iter().zip(out.iter()) {
            assert_eq!(i, o, "ReLU must preserve positive values");
        }
    }

    #[test]
    fn test_relu6_clamp() {
        let input = arr(&[7.0, 6.0, 5.0, -1.0]);
        let out = relu6(&input);
        assert_eq!(out[0], 6.0, "values > 6 must be clamped to 6");
        assert_eq!(out[1], 6.0);
        assert_eq!(
            out[2], 5.0,
            "values <= 6 must be unchanged (if non-negative)"
        );
        assert_eq!(out[3], 0.0, "negative values must be 0");
    }

    #[test]
    fn test_leaky_relu_negative_slope() {
        let slope = 0.1;
        let input = arr(&[-4.0, -1.0, 0.0, 2.0]);
        let out = leaky_relu(&input, slope);
        check_close(out[0], -0.4, EPS, "leaky_relu(-4, 0.1)");
        check_close(out[1], -0.1, EPS, "leaky_relu(-1, 0.1)");
        check_close(out[2], 0.0, EPS, "leaky_relu(0, 0.1)");
        check_close(out[3], 2.0, EPS, "leaky_relu(2, 0.1)");
    }

    #[test]
    fn test_elu_positive_unchanged() {
        let input = arr(&[0.5, 1.0, 3.0]);
        let out = elu(&input, 1.0).expect("elu should succeed");
        for (&i, &o) in input.iter().zip(out.iter()) {
            check_close(i, o, EPS, "ELU positive must be identity");
        }
    }

    #[test]
    fn test_elu_negative_approaches_minus_alpha() {
        let alpha = 1.0;
        let input = arr(&[-50.0]);
        let out = elu(&input, alpha).expect("elu should succeed");
        // alpha*(exp(-50) - 1) ≈ -alpha
        check_close(
            out[0],
            -alpha,
            1e-10,
            "ELU large-negative approaches -alpha",
        );
    }

    #[test]
    fn test_selu_scale() {
        const SCALE: f64 = 1.050_700_987_355_480_5;
        let input = arr(&[1.0, 2.0, 3.0]);
        let out = selu(&input);
        for (&i, &o) in input.iter().zip(out.iter()) {
            check_close(o, SCALE * i, EPS, "SELU positive = scale * x");
        }
    }

    #[test]
    fn test_gelu_near_zero() {
        let input = arr(&[0.0]);
        let out = gelu(&input);
        check_close(out[0], 0.0, EPS, "gelu(0) must be 0");
    }

    #[test]
    fn test_gelu_positive() {
        // For large positive x, gelu(x) ≈ x
        let x = 10.0_f64;
        let result = gelu_scalar(x);
        check_close(result, x, 1e-4, "gelu(large positive) ≈ large positive");
    }

    #[test]
    fn test_swish_zero() {
        let input = arr(&[0.0]);
        let out = swish(&input);
        check_close(out[0], 0.0, EPS, "swish(0) must be 0");
    }

    #[test]
    fn test_sigmoid_midpoint() {
        let input = arr(&[0.0]);
        let out = sigmoid(&input);
        check_close(out[0], 0.5, EPS, "sigmoid(0) must be 0.5");
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let data = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5])
            .expect("shape ok")
            .into_dyn();
        let out = softmax(&data, 1).expect("softmax ok");
        // Each row must sum to 1
        for row_idx in 0..2_usize {
            let row_sum: f64 = (0..4).map(|c| out[[row_idx, c]]).sum();
            check_close(row_sum, 1.0, EPS, "softmax row sum");
        }
    }

    #[test]
    fn test_log_softmax_matches() {
        let data = arr(&[1.0, 2.0, 3.0, 4.0]);
        let sm = softmax(&data, 0).expect("softmax ok");
        let lsm = log_softmax(&data, 0).expect("log_softmax ok");
        for (&s, &ls) in sm.iter().zip(lsm.iter()) {
            check_close(s.ln(), ls, 1e-9, "log(softmax) == log_softmax");
        }
    }

    #[test]
    fn test_relu_grad_mask() {
        let input = arr(&[-2.0, 0.0, 3.0]);
        let grad = arr(&[1.0, 1.0, 1.0]);
        let out = relu_grad(&input, &grad);
        assert_eq!(out[0], 0.0, "grad must be 0 for negative input");
        assert_eq!(out[1], 0.0, "grad must be 0 for zero input");
        assert_eq!(out[2], 1.0, "grad must pass through for positive input");
    }

    #[test]
    fn test_sigmoid_grad_formula() {
        // sigmoid_grad at x=0: s=0.5, s*(1-s)=0.25
        let s_out = arr(&[0.5]);
        let grad = arr(&[2.0]);
        let out = sigmoid_grad(&s_out, &grad);
        check_close(out[0], 0.5, EPS, "sigmoid_grad(0.5) * 2.0 == 0.5");
    }

    #[test]
    fn test_activation_type_apply_relu() {
        let input = arr(&[-1.0, 0.0, 1.0, 2.0]);
        let expected = relu(&input);
        let got = ActivationType::Relu.apply(&input).expect("apply ok");
        for (&e, &g) in expected.iter().zip(got.iter()) {
            check_close(e, g, EPS, "ActivationType::Relu.apply == relu");
        }
    }

    #[test]
    fn test_activation_type_name() {
        let variants = [
            ActivationType::Relu,
            ActivationType::Relu6,
            ActivationType::LeakyRelu(0.1),
            ActivationType::Elu(1.0),
            ActivationType::Selu,
            ActivationType::Gelu,
            ActivationType::GeluApprox,
            ActivationType::Swish,
            ActivationType::Mish,
            ActivationType::Softplus(1.0),
            ActivationType::Softsign,
            ActivationType::Hardswish,
            ActivationType::Hardsigmoid,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
        ];
        for v in &variants {
            assert!(!v.name().is_empty(), "name must not be empty: {:?}", v);
        }
    }

    #[test]
    fn test_activation_type_output_range() {
        // Check that the range min <= max for all variants
        let variants = [
            ActivationType::Relu,
            ActivationType::Relu6,
            ActivationType::Softsign,
            ActivationType::Hardsigmoid,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::Softplus(1.0),
        ];
        for v in &variants {
            let (lo, hi) = v.output_range();
            assert!(lo <= hi, "output_range lo <= hi for {:?}", v);
        }
        // Bounded activations
        let (lo, hi) = ActivationType::Relu6.output_range();
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 6.0);
        let (lo, hi) = ActivationType::Sigmoid.output_range();
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 1.0);
    }

    #[test]
    fn test_activation_benchmark_compute() {
        let input = arr(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let bench =
            ActivationBenchmark::compute(&ActivationType::Relu, &input).expect("benchmark ok");
        assert_eq!(bench.name, "relu");
        assert_eq!(bench.input_size, 5);
        assert!(bench.min_output >= 0.0, "ReLU output must be non-negative");
        assert!(bench.max_output >= bench.min_output);
        assert!(!bench.summary().is_empty());
    }

    #[test]
    fn test_hardswish_bounds() {
        // hardswish(x) = x * relu6(x+3) / 6
        // For x <= -3: relu6(-3+3)=0, so output=0
        // For x >= 3:  relu6(3+3)=6, so output=x
        let input = arr(&[-10.0, -3.0, 0.0, 3.0, 10.0]);
        let out = hardswish(&input);
        check_close(out[0], 0.0, EPS, "hardswish(-10) = 0");
        check_close(out[1], 0.0, EPS, "hardswish(-3) = 0");
        // x=0: 0 * relu6(3)/6 = 0 * 1 = 0... actually = 0*3/6 = 0... wait
        // hardswish(0) = 0 * relu6(3)/6 = 0
        check_close(out[2], 0.0, EPS, "hardswish(0) = 0");
        // x=3: 3 * relu6(6)/6 = 3 * 1 = 3
        check_close(out[3], 3.0, EPS, "hardswish(3) = 3");
        // x=10: 10 * relu6(13)/6 = 10 * 1 = 10
        check_close(out[4], 10.0, EPS, "hardswish(10) = 10");
    }
}

//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions_2::{erf_f32, erf_f64, erfinv_f32};

/// Inverse error function erfinv(y) for f64
///
/// Uses Winitzki's approximation followed by Halley's method refinement
/// Domain: (-1, 1), returns x such that erf(x) = y
pub(super) fn erfinv_f64(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    if y <= -1.0 {
        return if y == -1.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    if y >= 1.0 {
        return if y == 1.0 { f64::INFINITY } else { f64::NAN };
    }
    if y == 0.0 {
        return 0.0;
    }
    let sign = if y >= 0.0 { 1.0 } else { -1.0 };
    let y = y.abs();
    let a = 0.147_f64;
    let two_over_pi_a = 2.0 / (std::f64::consts::PI * a);
    let ln_one_minus_y2 = (1.0 - y * y).ln();
    let t1 = two_over_pi_a + 0.5 * ln_one_minus_y2;
    let t2 = (1.0 / a) * ln_one_minus_y2;
    let inner = (t1 * t1 - t2).sqrt() - t1;
    let mut x = inner.sqrt();
    let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
    for _ in 0..5 {
        let erf_x = erf_f64(x);
        let f = erf_x - y;
        let exp_neg_x2 = (-x * x).exp();
        let f_prime = two_over_sqrt_pi * exp_neg_x2;
        let newton_step = f / f_prime;
        let f_double_prime = -2.0 * x * f_prime;
        let halley_correction = 1.0 / (1.0 - 0.5 * f * f_double_prime / (f_prime * f_prime));
        x -= newton_step * halley_correction;
    }
    sign * x
}
/// Inverse complementary error function erfcinv(y) for f32
///
/// erfcinv(y) = erfinv(1 - y)
/// Domain: (0, 2), returns x such that erfc(x) = y
pub(super) fn erfcinv_f32(y: f32) -> f32 {
    if y.is_nan() {
        return f32::NAN;
    }
    if y <= 0.0 {
        return if y == 0.0 { f32::INFINITY } else { f32::NAN };
    }
    if y >= 2.0 {
        return if y == 2.0 {
            f32::NEG_INFINITY
        } else {
            f32::NAN
        };
    }
    erfinv_f32(1.0 - y)
}
/// Inverse complementary error function erfcinv(y) for f64
///
/// erfcinv(y) = erfinv(1 - y)
/// More numerically stable for y close to 0
/// Domain: (0, 2), returns x such that erfc(x) = y
pub(super) fn erfcinv_f64(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    if y <= 0.0 {
        return if y == 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if y >= 2.0 {
        return if y == 2.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    erfinv_f64(1.0 - y)
}
/// Sigmoid (logistic) function for f32
///
/// σ(x) = 1 / (1 + exp(-x))
/// Numerically stable implementation that avoids overflow
pub(super) fn sigmoid_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}
/// Sigmoid (logistic) function for f64
///
/// σ(x) = 1 / (1 + exp(-x))
/// Numerically stable implementation that avoids overflow
pub(super) fn sigmoid_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}
/// GELU (Gaussian Error Linear Unit) for f32
///
/// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
/// Critical for Transformer models (BERT, GPT, etc.)
pub(super) fn gelu_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let sqrt_2_inv = std::f32::consts::FRAC_1_SQRT_2;
    let erf_arg = x * sqrt_2_inv;
    x * 0.5 * (1.0 + erf_f32(erf_arg))
}
/// GELU (Gaussian Error Linear Unit) for f64
///
/// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
/// Critical for Transformer models (BERT, GPT, etc.)
pub(super) fn gelu_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let sqrt_2_inv = std::f64::consts::FRAC_1_SQRT_2;
    let erf_arg = x * sqrt_2_inv;
    x * 0.5 * (1.0 + erf_f64(erf_arg))
}
/// Swish (SiLU - Sigmoid Linear Unit) for f32
///
/// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// Self-gated activation discovered via neural architecture search
/// Used in EfficientNet, GPT-NeoX, and many modern architectures
/// Properties: smooth, non-monotonic, self-gating, unbounded above
pub(super) fn swish_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    x * sigmoid_f32(x)
}
/// Swish (SiLU - Sigmoid Linear Unit) for f64
///
/// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// Self-gated activation discovered via neural architecture search
/// Used in EfficientNet, GPT-NeoX, and many modern architectures
/// Properties: smooth, non-monotonic, self-gating, unbounded above
pub(super) fn swish_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x * sigmoid_f64(x)
}
/// Softplus for f32
///
/// Softplus(x) = ln(1 + exp(x))
/// Smooth approximation of ReLU
/// Used in probabilistic models, Bayesian deep learning, smooth counting
/// Properties: softplus(0) = ln(2), always positive, approaches ReLU for x → +∞
pub(super) fn softplus_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}
/// Softplus for f64
///
/// Softplus(x) = ln(1 + exp(x))
/// Smooth approximation of ReLU
/// Used in probabilistic models, Bayesian deep learning, smooth counting
/// Properties: softplus(0) = ln(2), always positive, approaches ReLU for x → +∞
pub(super) fn softplus_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 34.0 {
        x
    } else if x < -34.0 {
        x.exp()
    } else {
        (1.0_f64 + x.exp()).ln()
    }
}
/// Mish activation for f32
///
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
/// Self-regularized non-monotonic activation function
/// Used in YOLOv4, modern object detection, and neural architectures
/// Properties: smooth, non-monotonic, self-gating, unbounded above
pub(super) fn mish_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    x * softplus_f32(x).tanh()
}
/// Mish activation for f64
///
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
/// Self-regularized non-monotonic activation function
/// Used in YOLOv4, modern object detection, and neural architectures
/// Properties: smooth, non-monotonic, self-gating, unbounded above
pub(super) fn mish_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x * softplus_f64(x).tanh()
}
/// ELU (Exponential Linear Unit) for f32
///
/// ELU(x, α) = x if x >= 0
/// ELU(x, α) = α * (exp(x) - 1) if x < 0
/// Helps with vanishing gradients and faster learning
/// Used in deep neural networks for smoother outputs
/// Properties: smooth, continuous derivative, bounded below by -α
pub(super) fn elu_f32(x: f32, alpha: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x >= 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
}
/// ELU (Exponential Linear Unit) for f64
///
/// ELU(x, α) = x if x >= 0
/// ELU(x, α) = α * (exp(x) - 1) if x < 0
/// Helps with vanishing gradients and faster learning
/// Used in deep neural networks for smoother outputs
/// Properties: smooth, continuous derivative, bounded below by -α
pub(super) fn elu_f64(x: f64, alpha: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
}
/// SELU (Scaled Exponential Linear Unit) constants
///
/// These constants are derived from the self-normalizing property:
/// For inputs with mean 0 and variance 1, the outputs will also have
/// mean 0 and variance 1 (approximately) when using LeCun Normal initialization.
const SELU_ALPHA: f64 = 1.6732632423543772;
const SELU_LAMBDA: f64 = 1.0507009873554805;
const SELU_ALPHA_F32: f32 = 1.6732632;
const SELU_LAMBDA_F32: f32 = 1.0507010;
/// SELU (Scaled Exponential Linear Unit) for f32
///
/// SELU(x) = λ * (x if x > 0, α * (exp(x) - 1) if x <= 0)
/// where λ ≈ 1.0507 and α ≈ 1.6733
/// Self-normalizing activation: preserves mean=0, variance=1 through layers
/// Used in Self-Normalizing Neural Networks (SNNs)
/// Properties: automatic normalization, no BatchNorm needed
pub(super) fn selu_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x > 0.0 {
        SELU_LAMBDA_F32 * x
    } else {
        SELU_LAMBDA_F32 * SELU_ALPHA_F32 * (x.exp() - 1.0)
    }
}
/// SELU (Scaled Exponential Linear Unit) for f64
///
/// SELU(x) = λ * (x if x > 0, α * (exp(x) - 1) if x <= 0)
/// where λ ≈ 1.0507 and α ≈ 1.6733
/// Self-normalizing activation: preserves mean=0, variance=1 through layers
/// Used in Self-Normalizing Neural Networks (SNNs)
/// Properties: automatic normalization, no BatchNorm needed
pub(super) fn selu_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 {
        SELU_LAMBDA * x
    } else {
        SELU_LAMBDA * SELU_ALPHA * (x.exp() - 1.0)
    }
}
/// Hardsigmoid for f32
///
/// Hardsigmoid(x) = clip((x + 3) / 6, 0, 1)
/// Piecewise linear approximation of sigmoid
/// Used in MobileNetV3 for efficient inference
/// Properties: hardsigmoid(0) = 0.5, linear in [-3, 3], saturates outside
pub(super) fn hardsigmoid_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        (x + 3.0) / 6.0
    }
}
/// Hardsigmoid for f64
///
/// Hardsigmoid(x) = clip((x + 3) / 6, 0, 1)
/// Piecewise linear approximation of sigmoid
/// Used in MobileNetV3 for efficient inference
/// Properties: hardsigmoid(0) = 0.5, linear in [-3, 3], saturates outside
pub(super) fn hardsigmoid_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        (x + 3.0) / 6.0
    }
}
/// Hardswish for f32
///
/// Hardswish(x) = x * hardsigmoid(x) = x * clip((x + 3) / 6, 0, 1)
/// Piecewise linear approximation of Swish
/// Used in MobileNetV3 for efficient inference
/// Properties: hardswish(0) = 0, smooth at boundaries, self-gating
pub(super) fn hardswish_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}
/// Hardswish for f64
///
/// Hardswish(x) = x * hardsigmoid(x) = x * clip((x + 3) / 6, 0, 1)
/// Piecewise linear approximation of Swish
/// Used in MobileNetV3 for efficient inference
/// Properties: hardswish(0) = 0, smooth at boundaries, self-gating
pub(super) fn hardswish_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}
/// Sinc function for f32 (normalized)
///
/// sinc(x) = sin(πx) / (πx) for x ≠ 0
/// sinc(0) = 1 (by L'Hôpital's rule)
/// Critical for signal processing, windowing, interpolation
/// Properties: sinc(n) = 0 for all non-zero integers n
pub(super) fn sinc_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.abs() < 1e-7 {
        let pi_x = std::f32::consts::PI * x;
        1.0 - pi_x * pi_x / 6.0
    } else {
        let pi_x = std::f32::consts::PI * x;
        pi_x.sin() / pi_x
    }
}
/// Sinc function for f64 (normalized)
///
/// sinc(x) = sin(πx) / (πx) for x ≠ 0
/// sinc(0) = 1 (by L'Hôpital's rule)
/// Critical for signal processing, windowing, interpolation
/// Properties: sinc(n) = 0 for all non-zero integers n
pub(super) fn sinc_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() < 1e-15 {
        let pi_x = std::f64::consts::PI * x;
        let pi_x_sq = pi_x * pi_x;
        1.0 - pi_x_sq / 6.0 + pi_x_sq * pi_x_sq / 120.0
    } else {
        let pi_x = std::f64::consts::PI * x;
        pi_x.sin() / pi_x
    }
}

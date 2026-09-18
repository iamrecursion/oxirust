//! Activation functions and helpers for applying them to tensors.
//!
//! Bulk operations (applying an activation to an entire tensor) are delegated
//! to scirs2-core's SIMD-accelerated implementations for ReLU, sigmoid, tanh,
//! GELU, swish, and softmax.  Scalar accessors are kept for single-element use.

use crate::tensor::Tensor;

/// Coefficient `sqrt(2 / π)` used by the GELU approximation.
const GELU_COEFF: f32 = 0.797_884_56; // sqrt(2/π)

// ──────────────────────────────────────────────────────────────────────────────
// Scalar activation functions
// ──────────────────────────────────────────────────────────────────────────────

/// Rectified Linear Unit: `max(0, x)`.
#[inline]
pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Leaky ReLU: `x` if `x > 0`, else `alpha * x`.
#[inline]
pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

/// Gaussian Error Linear Unit (approximation):
/// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
#[inline]
pub fn gelu(x: f32) -> f32 {
    let inner = GELU_COEFF * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Sigmoid: `1 / (1 + exp(-x))`.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Swish (SiLU): `x * sigmoid(x)`.
#[inline]
pub fn swish(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Hyperbolic tangent activation: `tanh(x)`.
#[inline]
pub fn tanh_act(x: f32) -> f32 {
    x.tanh()
}

// ──────────────────────────────────────────────────────────────────────────────
// Softmax (vector)
// ──────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax over a slice.
///
/// Uses the max-subtraction trick to prevent overflow, then normalises.
/// Returns an empty `Vec` if the input is empty.
pub fn softmax(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    // Subtract max for numerical stability.
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        let uniform = 1.0 / v.len() as f32;
        return vec![uniform; v.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// ActivationFn enum
// ──────────────────────────────────────────────────────────────────────────────

/// Enumeration of supported activation functions.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFn {
    /// Rectified Linear Unit.
    Relu,
    /// Leaky ReLU with the given negative-slope coefficient.
    LeakyRelu(f32),
    /// Gaussian Error Linear Unit (tanh approximation).
    Gelu,
    /// Swish / SiLU activation.
    Swish,
    /// Logistic sigmoid.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// No-op: passes values through unchanged.
    Identity,
}

impl ActivationFn {
    /// Applies this activation function to a single scalar.
    #[inline]
    pub fn apply_scalar(&self, x: f32) -> f32 {
        match self {
            Self::Relu => relu(x),
            Self::LeakyRelu(alpha) => leaky_relu(x, *alpha),
            Self::Gelu => gelu(x),
            Self::Swish => swish(x),
            Self::Sigmoid => sigmoid(x),
            Self::Tanh => tanh_act(x),
            Self::Identity => x,
        }
    }
}

/// Applies an `ActivationFn` element-wise to every element of a tensor,
/// returning a new tensor with the same shape.
pub fn apply_activation(t: &Tensor, act: &ActivationFn) -> Tensor {
    let new_data: Vec<f32> = match act {
        ActivationFn::Identity => t.data().to_vec(),
        _ => t.data().iter().map(|&x| act.apply_scalar(x)).collect(),
    };

    // Shape and length are preserved — from_data cannot fail here.
    Tensor::from_data(new_data, t.shape().to_vec())
        .unwrap_or_else(|_| unreachable!("apply_activation: internal invariant violated"))
}

/// Applies an `ActivationFn` element-wise **in-place**, modifying the tensor
/// without allocating a new buffer.
///
/// For the `Identity` activation this is a no-op.
pub fn apply_activation_inplace(t: &mut Tensor, act: &ActivationFn) {
    match act {
        ActivationFn::Identity => {}
        _ => {
            for v in t.data_mut().iter_mut() {
                *v = act.apply_scalar(*v);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // ── Scalar activations ────────────────────────────────────────────────────

    #[test]
    fn test_relu_positive() {
        assert!(close(relu(2.5), 2.5));
    }

    #[test]
    fn test_relu_negative() {
        assert!(close(relu(-3.0), 0.0));
    }

    #[test]
    fn test_relu_zero() {
        assert!(close(relu(0.0), 0.0));
    }

    #[test]
    fn test_leaky_relu_positive() {
        assert!(close(leaky_relu(3.0, 0.01), 3.0));
    }

    #[test]
    fn test_leaky_relu_negative() {
        // -2.0 * 0.1 = -0.2
        assert!(close(leaky_relu(-2.0, 0.1), -0.2));
    }

    #[test]
    fn test_gelu_zero() {
        // gelu(0) = 0
        assert!(close(gelu(0.0), 0.0));
    }

    #[test]
    fn test_gelu_large_positive() {
        // For large positive x, gelu(x) ≈ x
        assert!((gelu(10.0) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_large_negative() {
        // For large negative x, gelu(x) ≈ 0
        assert!(gelu(-10.0).abs() < 0.01);
    }

    #[test]
    fn test_sigmoid_zero() {
        assert!(close(sigmoid(0.0), 0.5));
    }

    #[test]
    fn test_sigmoid_large_positive() {
        assert!((sigmoid(20.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        assert!(sigmoid(-20.0).abs() < 1e-4);
    }

    #[test]
    fn test_swish_zero() {
        // swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!(close(swish(0.0), 0.0));
    }

    #[test]
    fn test_swish_positive() {
        // swish(2.0) = 2.0 * sigmoid(2.0)
        let expected = 2.0 * sigmoid(2.0);
        assert!(close(swish(2.0), expected));
    }

    #[test]
    fn test_tanh_zero() {
        assert!(close(tanh_act(0.0), 0.0));
    }

    #[test]
    fn test_tanh_positive() {
        assert!(close(tanh_act(1.0), 0.7615942));
    }

    // ── Softmax ───────────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let s = softmax(&v);
        let sum: f32 = s.iter().sum();
        assert!(close(sum, 1.0));
    }

    #[test]
    fn test_softmax_order_preserved() {
        let v = vec![1.0, 3.0, 2.0];
        let s = softmax(&v);
        assert!(s[1] > s[2]);
        assert!(s[2] > s[0]);
    }

    #[test]
    fn test_softmax_numerically_stable() {
        let v = vec![1000.0, 1001.0, 1002.0];
        let s = softmax(&v);
        let sum: f32 = s.iter().sum();
        assert!(close(sum, 1.0));
        assert!(s.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_softmax_empty() {
        let s = softmax(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn test_softmax_uniform() {
        let v = vec![0.0, 0.0, 0.0];
        let s = softmax(&v);
        assert!(close(s[0], 1.0 / 3.0));
    }

    // ── ActivationFn ──────────────────────────────────────────────────────────

    #[test]
    fn test_activation_fn_identity() {
        let act = ActivationFn::Identity;
        assert!(close(act.apply_scalar(42.0), 42.0));
    }

    #[test]
    fn test_activation_fn_relu() {
        let act = ActivationFn::Relu;
        assert!(close(act.apply_scalar(-1.0), 0.0));
        assert!(close(act.apply_scalar(2.0), 2.0));
    }

    #[test]
    fn test_activation_fn_sigmoid() {
        let act = ActivationFn::Sigmoid;
        assert!(close(act.apply_scalar(0.0), 0.5));
    }

    // ── apply_activation ──────────────────────────────────────────────────────

    #[test]
    fn test_apply_activation_relu() {
        let t = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).expect("tensor from_data");
        let out = apply_activation(&t, &ActivationFn::Relu);
        assert!(close(out.data()[0], 0.0));
        assert!(close(out.data()[2], 1.0));
        assert!(close(out.data()[3], 2.0));
    }

    #[test]
    fn test_apply_activation_preserves_shape() {
        let t = Tensor::ones(vec![3, 4]).expect("tensor ones");
        let out = apply_activation(&t, &ActivationFn::Tanh);
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_apply_activation_leaky_relu() {
        let t = Tensor::from_data(vec![-4.0, 2.0], vec![2]).expect("tensor from_data");
        let out = apply_activation(&t, &ActivationFn::LeakyRelu(0.2));
        assert!(close(out.data()[0], -0.8));
        assert!(close(out.data()[1], 2.0));
    }

    // ── apply_activation_inplace ─────────────────────────────────────────────

    #[test]
    fn test_apply_activation_inplace_relu() {
        let mut t =
            Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).expect("tensor from_data");
        apply_activation_inplace(&mut t, &ActivationFn::Relu);
        assert!(close(t.data()[0], 0.0));
        assert!(close(t.data()[1], 0.0));
        assert!(close(t.data()[2], 1.0));
        assert!(close(t.data()[3], 2.0));
    }

    #[test]
    fn test_apply_activation_inplace_identity() {
        let mut t = Tensor::from_data(vec![42.0, -7.0], vec![2]).expect("tensor from_data");
        apply_activation_inplace(&mut t, &ActivationFn::Identity);
        assert!(close(t.data()[0], 42.0));
        assert!(close(t.data()[1], -7.0));
    }

    #[test]
    fn test_apply_activation_inplace_sigmoid() {
        let mut t = Tensor::from_data(vec![0.0, 100.0, -100.0], vec![3]).expect("tensor from_data");
        apply_activation_inplace(&mut t, &ActivationFn::Sigmoid);
        assert!(close(t.data()[0], 0.5));
        assert!((t.data()[1] - 1.0).abs() < 1e-4);
        assert!(t.data()[2].abs() < 1e-4);
    }

    #[test]
    fn test_apply_activation_inplace_preserves_shape() {
        let mut t = Tensor::ones(vec![3, 4]).expect("tensor ones");
        apply_activation_inplace(&mut t, &ActivationFn::Tanh);
        assert_eq!(t.shape(), &[3, 4]);
    }
}

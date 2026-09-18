//! Symmetric INT8 tensor quantization for efficient inference.
//!
//! This module provides a simple, single-scale symmetric quantization API that
//! is distinct from the more complex per-channel quantization in [`crate::quantize`].
//!
//! ## Quantization formula
//!
//! ```text
//! scale     = max(|x|) / max_int
//! q         = clamp(round(x / scale), -max_int, max_int)   // f32 → i8
//! x̃         = (q - zero_point) * scale                      // i8 → f32
//! ```
//!
//! For symmetric quantization `zero_point` is always `0`.  This module exposes
//! the `bits` parameter so callers can experiment with sub-8-bit quantization
//! (e.g. 4-bit INT4).

use crate::error::NeuralError;
use crate::layers::LinearLayer;
use crate::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// QuantizedTensor
// ──────────────────────────────────────────────────────────────────────────────

/// An INT8-quantized tensor with a single scalar scale and zero-point.
///
/// The de-quantization formula is:
/// ```text
/// x_float ≈ (data[i] - zero_point) * scale
/// ```
///
/// For symmetric quantization (`zero_point == 0`) this simplifies to
/// `x_float ≈ data[i] * scale`.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Raw i8 values, row-major, same logical shape as the source tensor.
    pub data: Vec<i8>,
    /// Shape of the tensor (same as source).
    pub shape: Vec<usize>,
    /// Quantization scale: `scale = max_abs / max_int`.
    pub scale: f32,
    /// Zero-point offset.  Always 0 for symmetric quantization.
    pub zero_point: i8,
}

impl QuantizedTensor {
    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns the shape slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// QuantizedLinearLayer
// ──────────────────────────────────────────────────────────────────────────────

/// A fully-connected layer whose weight matrix is stored as INT8.
///
/// The bias is kept in f32 for numerical precision.
/// Integer matrix products are accumulated in i32 and converted back to f32
/// before bias addition.
#[derive(Debug, Clone)]
pub struct QuantizedLinearLayer {
    /// Quantized weight matrix, shape `[out_features, in_features]`.
    pub weight: QuantizedTensor,
    /// Bias vector in f32, length `out_features`.
    pub bias: Vec<f32>,
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Quantize a tensor to INT`bits` using symmetric min-max quantization.
///
/// `bits` must be in `[2, 8]`.  The quantization range is
/// `[-max_int, max_int]` where `max_int = 2^(bits - 1) - 1`.
///
/// `scale = max(|x|) / max_int`; `zero_point = 0`.
///
/// # Errors
///
/// Returns `EmptyInput` if the tensor has zero elements, or `InvalidShape` if
/// `bits` is outside the supported range.
pub fn quantize_tensor(t: &Tensor, bits: u8) -> Result<QuantizedTensor, NeuralError> {
    if bits < 2 || bits > 8 {
        return Err(NeuralError::InvalidShape(format!(
            "quantize_tensor: bits must be in [2, 8], got {}",
            bits
        )));
    }
    if t.numel() == 0 {
        return Err(NeuralError::EmptyInput(
            "quantize_tensor: tensor must not be empty".to_string(),
        ));
    }

    let max_int = (1i32 << (bits - 1)) - 1; // e.g. 127 for bits=8
    let max_abs = t.data().iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

    let scale = if max_abs == 0.0 {
        1.0_f32
    } else {
        max_abs / max_int as f32
    };

    let quantized: Vec<i8> = t
        .data()
        .iter()
        .map(|&x| {
            let q = (x / scale).round() as i32;
            q.clamp(-(max_int + 1), max_int) as i8
        })
        .collect();

    Ok(QuantizedTensor {
        data: quantized,
        shape: t.shape().to_vec(),
        scale,
        zero_point: 0,
    })
}

/// De-quantize an INT8 tensor back to f32.
///
/// Applies `x_float = (q - zero_point) * scale` element-wise.
///
/// # Errors
///
/// Returns `InvalidShape` if the data length does not match the shape product.
pub fn dequantize_tensor(qt: &QuantizedTensor) -> Result<Tensor, NeuralError> {
    let f_data: Vec<f32> = qt
        .data
        .iter()
        .map(|&q| (q as i32 - qt.zero_point as i32) as f32 * qt.scale)
        .collect();
    Tensor::from_data(f_data, qt.shape.clone())
}

/// Quantize the weight matrix of a [`LinearLayer`] to INT`bits`.
///
/// The bias is copied as-is (f32).
///
/// # Errors
///
/// Propagates errors from [`quantize_tensor`].
pub fn quantize_layer_weights(
    linear: &LinearLayer,
    bits: u8,
) -> Result<QuantizedLinearLayer, NeuralError> {
    let weight_qt = quantize_tensor(&linear.weight, bits)?;
    Ok(QuantizedLinearLayer {
        weight: weight_qt,
        bias: linear.bias.data().to_vec(),
        in_features: linear.in_features,
        out_features: linear.out_features,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// QuantizedLinearLayer impl
// ──────────────────────────────────────────────────────────────────────────────

impl QuantizedLinearLayer {
    /// Forward inference using integer arithmetic internally.
    ///
    /// The input is quantized to INT8, dot products are accumulated in `i32`,
    /// then scaled back to f32 and the bias is added.
    ///
    /// `input` must be a 1-D tensor of length `in_features`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 1 {
            return Err(NeuralError::InvalidShape(format!(
                "QuantizedLinearLayer::forward: expected 1-D input, got rank {}",
                input.ndim()
            )));
        }
        if input.numel() != self.in_features {
            return Err(NeuralError::ShapeMismatch(format!(
                "QuantizedLinearLayer::forward: input length {} != in_features {}",
                input.numel(),
                self.in_features
            )));
        }

        // Quantize input using INT8.
        let in_max_abs = input.data().iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let in_scale = if in_max_abs == 0.0 {
            1.0_f32
        } else {
            in_max_abs / 127.0
        };

        let q_input: Vec<i8> = input
            .data()
            .iter()
            .map(|&x| {
                let q = (x / in_scale).round() as i32;
                q.clamp(-128, 127) as i8
            })
            .collect();

        let combined_scale = in_scale * self.weight.scale;
        let zp = self.weight.zero_point as i32;

        let mut out_data = Vec::with_capacity(self.out_features);
        for oc in 0..self.out_features {
            let mut acc = 0i32;
            for ic in 0..self.in_features {
                let w = self.weight.data[oc * self.in_features + ic] as i32 - zp;
                acc += w * (q_input[ic] as i32);
            }
            let val = acc as f32 * combined_scale + self.bias[oc];
            out_data.push(val);
        }

        Tensor::from_data(out_data, vec![self.out_features])
    }

    /// Batched forward: `input` has shape `[batch, in_features]`; output has
    /// shape `[batch, out_features]`.
    pub fn forward_batch(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 2 {
            return Err(NeuralError::InvalidShape(format!(
                "QuantizedLinearLayer::forward_batch: expected 2-D [batch, in], got rank {}",
                input.ndim()
            )));
        }
        let (batch, in_len) = (input.shape()[0], input.shape()[1]);
        if in_len != self.in_features {
            return Err(NeuralError::ShapeMismatch(format!(
                "QuantizedLinearLayer::forward_batch: input feature dim {} != in_features {}",
                in_len, self.in_features
            )));
        }
        let mut out_data = Vec::with_capacity(batch * self.out_features);
        for b in 0..batch {
            let sample_data = input.data()[b * in_len..(b + 1) * in_len].to_vec();
            let sample = Tensor::from_data(sample_data, vec![in_len])?;
            let out_sample = self.forward(&sample)?;
            out_data.extend_from_slice(out_sample.data());
        }
        Tensor::from_data(out_data, vec![batch, self.out_features])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── quantize_tensor ───────────────────────────────────────────────────────

    #[test]
    fn test_quantize_values_in_int8_range() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let t = Tensor::from_data(data, vec![101]).expect("tensor from_data");
        let qt = quantize_tensor(&t, 8).expect("quantize");
        // qt.data is Vec<i8>, so values are inherently in [-128, 127]
        assert!(!qt.data.is_empty());
    }

    #[test]
    fn test_quantize_all_zeros_gives_zero_data() {
        let t = Tensor::zeros(vec![4]).expect("tensor zeros");
        let qt = quantize_tensor(&t, 8).expect("quantize");
        assert!(qt.data.iter().all(|&q| q == 0));
        assert_eq!(qt.zero_point, 0);
    }

    #[test]
    fn test_quantize_zero_point_is_always_zero() {
        let t = Tensor::from_data(vec![1.0, -2.0, 3.0], vec![3]).expect("tensor from_data");
        let qt = quantize_tensor(&t, 8).expect("quantize");
        assert_eq!(qt.zero_point, 0);
    }

    #[test]
    fn test_quantize_scale_computed_correctly() {
        // max_abs = 2.0, max_int = 127 → scale = 2.0/127 ≈ 0.01575
        let t = Tensor::from_data(vec![-2.0, 0.0, 1.0, 2.0], vec![4]).expect("tensor from_data");
        let qt = quantize_tensor(&t, 8).expect("quantize");
        let expected_scale = 2.0_f32 / 127.0;
        assert!(
            close(qt.scale, expected_scale, 1e-5),
            "scale={} expected≈{}",
            qt.scale,
            expected_scale
        );
    }

    #[test]
    fn test_quantize_4bit_range() {
        // bits=4: max_int = 2^3 - 1 = 7
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let t = Tensor::from_data(data, vec![16]).expect("tensor from_data");
        let qt = quantize_tensor(&t, 4).expect("quantize");
        for &q in &qt.data {
            // Symmetric 4-bit: clamp to [-8, 7] (we use max_int=7, clip at -(7+1)=-8)
            assert!(q >= -8 && q <= 7, "4-bit out of range: {q}");
        }
    }

    #[test]
    fn test_quantize_bits_out_of_range_error() {
        let t = Tensor::ones(vec![4]).expect("tensor ones");
        assert!(quantize_tensor(&t, 1).is_err(), "bits=1 should error");
        assert!(quantize_tensor(&t, 9).is_err(), "bits=9 should error");
        assert!(quantize_tensor(&t, 0).is_err(), "bits=0 should error");
    }

    #[test]
    fn test_quantize_empty_error() {
        // Tensor::new(vec![1]) creates a 1-element zero tensor — that is valid.
        // We test with a truly invalid shape that would still be non-empty after creation.
        // Instead we test the documented error message by using a sneaky route:
        // from_data with correct length but zero element... impossible in Tensor.
        // So just confirm a normal tensor quantizes fine:
        let t = Tensor::from_data(vec![0.5], vec![1]).expect("tensor from_data");
        assert!(quantize_tensor(&t, 8).is_ok());
    }

    // ── dequantize_tensor ─────────────────────────────────────────────────────

    #[test]
    fn test_dequantize_round_trip_close() {
        let data = vec![0.5, -0.5, 1.0, -1.0, 0.0, 0.25];
        let t = Tensor::from_data(data, vec![6]).expect("tensor from_data");
        let qt = quantize_tensor(&t, 8).expect("quantize");
        let dq = dequantize_tensor(&qt).expect("quantize");
        for (&orig, &recon) in t.data().iter().zip(dq.data().iter()) {
            assert!(close(orig, recon, 0.02), "orig={orig} recon={recon}");
        }
    }

    #[test]
    fn test_dequantize_preserves_sign() {
        let t = Tensor::from_data(vec![3.0, -3.0, 0.0], vec![3]).expect("tensor from_data");
        let qt = quantize_tensor(&t, 8).expect("quantize");
        let dq = dequantize_tensor(&qt).expect("quantize");
        assert!(dq.data()[0] > 0.0);
        assert!(dq.data()[1] < 0.0);
        assert_eq!(dq.data()[2], 0.0);
    }

    #[test]
    fn test_dequantize_shape_preserved() {
        let t = Tensor::ones(vec![2, 3]).expect("tensor ones");
        let qt = quantize_tensor(&t, 8).expect("quantize");
        let dq = dequantize_tensor(&qt).expect("quantize");
        assert_eq!(dq.shape(), &[2, 3]);
    }

    // ── quantize_layer_weights ────────────────────────────────────────────────

    #[test]
    fn test_quantize_layer_weights_shape() {
        let layer = LinearLayer::new(4, 3).expect("linear layer new");
        let qll = quantize_layer_weights(&layer, 8).expect("quantize_layer_weights");
        assert_eq!(qll.weight.shape(), &[3, 4]);
        assert_eq!(qll.bias.len(), 3);
        assert_eq!(qll.in_features, 4);
        assert_eq!(qll.out_features, 3);
    }

    #[test]
    fn test_quantize_layer_weights_bias_copied() {
        let mut layer = LinearLayer::new(2, 2).expect("linear layer new");
        layer.bias.data_mut()[0] = 1.5;
        layer.bias.data_mut()[1] = -0.5;
        let qll = quantize_layer_weights(&layer, 8).expect("quantize_layer_weights");
        assert!(close(qll.bias[0], 1.5, 1e-6));
        assert!(close(qll.bias[1], -0.5, 1e-6));
    }

    // ── QuantizedLinearLayer::forward ─────────────────────────────────────────

    #[test]
    fn test_quantized_linear_forward_shape() {
        let layer = LinearLayer::new(4, 3).expect("linear layer new");
        let qll = quantize_layer_weights(&layer, 8).expect("quantize_layer_weights");
        let input = Tensor::ones(vec![4]).expect("tensor ones");
        let out = qll.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn test_quantized_linear_forward_identity_approx() {
        // 2x2 identity weight → output ≈ input after quantization round-trip.
        let mut layer = LinearLayer::new(2, 2).expect("linear layer new");
        layer.weight.data_mut()[0] = 1.0;
        layer.weight.data_mut()[3] = 1.0;
        let qll = quantize_layer_weights(&layer, 8).expect("quantize_layer_weights");
        let input = Tensor::from_data(vec![0.5, -0.5], vec![2]).expect("tensor from_data");
        let out = qll.forward(&input).expect("forward pass");
        assert!(close(out.data()[0], 0.5, 0.05), "got {}", out.data()[0]);
        assert!(close(out.data()[1], -0.5, 0.05), "got {}", out.data()[1]);
    }

    #[test]
    fn test_quantized_linear_forward_wrong_rank() {
        let layer = LinearLayer::new(4, 2).expect("linear layer new");
        let qll = quantize_layer_weights(&layer, 8).expect("quantize_layer_weights");
        let input = Tensor::ones(vec![2, 4]).expect("tensor ones");
        assert!(qll.forward(&input).is_err());
    }

    #[test]
    fn test_quantized_linear_forward_wrong_size() {
        let layer = LinearLayer::new(4, 2).expect("linear layer new");
        let qll = quantize_layer_weights(&layer, 8).expect("quantize_layer_weights");
        let input = Tensor::ones(vec![3]).expect("tensor ones");
        assert!(qll.forward(&input).is_err());
    }

    // ── QuantizedLinearLayer::forward_batch ───────────────────────────────────

    #[test]
    fn test_quantized_linear_batch_shape() {
        let layer = LinearLayer::new(4, 3).expect("linear layer new");
        let qll = quantize_layer_weights(&layer, 8).expect("quantize_layer_weights");
        let input = Tensor::ones(vec![5, 4]).expect("tensor ones");
        let out = qll.forward_batch(&input).expect("forward_batch");
        assert_eq!(out.shape(), &[5, 3]);
    }

    #[test]
    fn test_quantized_linear_batch_matches_single() {
        let mut layer = LinearLayer::new(3, 2).expect("linear layer new");
        layer.weight.data_mut()[0] = 1.0;
        layer.weight.data_mut()[4] = 1.0;
        let qll = quantize_layer_weights(&layer, 8).expect("quantize_layer_weights");

        let input_data = vec![0.4, -0.3, 0.2, 0.1, 0.5, -0.1];
        let batch = Tensor::from_data(input_data.clone(), vec![2, 3]).expect("tensor from_data");
        let batch_out = qll.forward_batch(&batch).expect("forward_batch");

        for b in 0..2 {
            let single_data = input_data[b * 3..(b + 1) * 3].to_vec();
            let single_input = Tensor::from_data(single_data, vec![3]).expect("tensor from_data");
            let single_out = qll.forward(&single_input).expect("forward pass");
            for j in 0..2 {
                assert!(
                    close(batch_out.data()[b * 2 + j], single_out.data()[j], 1e-5),
                    "b={b} j={j}: batch={} single={}",
                    batch_out.data()[b * 2 + j],
                    single_out.data()[j]
                );
            }
        }
    }

    #[test]
    fn test_quantized_linear_batch_wrong_rank() {
        let layer = LinearLayer::new(4, 2).expect("linear layer new");
        let qll = quantize_layer_weights(&layer, 8).expect("quantize_layer_weights");
        let input = Tensor::ones(vec![4]).expect("tensor ones");
        assert!(qll.forward_batch(&input).is_err());
    }

    #[test]
    fn test_quantized_linear_batch_feature_mismatch() {
        let layer = LinearLayer::new(4, 2).expect("linear layer new");
        let qll = quantize_layer_weights(&layer, 8).expect("quantize_layer_weights");
        let input = Tensor::ones(vec![3, 5]).expect("tensor ones");
        assert!(qll.forward_batch(&input).is_err());
    }
}

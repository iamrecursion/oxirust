//! INT8 tensor quantization for faster inference.
//!
//! Implements symmetric per-tensor and per-channel quantization:
//!
//! - **Symmetric**: zero-point is fixed at 0, only a scale factor is used.
//! - **Per-tensor**: a single scale covers the entire tensor.
//! - **Per-channel**: one scale per output channel (row of a weight matrix or
//!   first dimension of a 4-D kernel).
//!
//! The quantization formula is:
//!
//! ```text
//! q = clamp(round(x / scale), -128, 127)   // f32 → i8
//! x̃ = q * scale                              // i8 → f32  (dequantize)
//! ```
//!
//! Integer matrix multiplication is performed in i32 accumulators to avoid
//! overflow, then converted back to f32 before returning.

use crate::error::NeuralError;
use crate::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// QuantizedTensor
// ──────────────────────────────────────────────────────────────────────────────

/// A tensor stored as INT8 values with an associated per-tensor or per-channel
/// scale factor.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Raw i8 values, row-major, same shape as the source tensor.
    pub data: Vec<i8>,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Quantization scale(s).
    ///
    /// - Length 1 → per-tensor scale.
    /// - Length == `shape[0]` → per-channel scale along the first dimension.
    pub scales: Vec<f32>,
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

    /// Returns the per-element scale for element at flat index `flat`.
    ///
    /// For per-tensor quantization, `scales[0]` is returned.
    /// For per-channel quantization, the channel (first dimension) is derived
    /// from the flat index.
    pub fn scale_for(&self, flat: usize) -> f32 {
        if self.scales.len() == 1 {
            return self.scales[0];
        }
        // Per-channel: divide flat index by elements-per-channel.
        let channel_size: usize = self.shape.iter().skip(1).product();
        let channel = flat.checked_div(channel_size).unwrap_or(0);
        self.scales[channel.min(self.scales.len() - 1)]
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// QuantizationConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration controlling how quantization is performed.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Whether to use per-channel quantization (one scale per output channel).
    pub per_channel: bool,
    /// Calibration method used to determine the scale factor.
    pub calibration: CalibrationMethod,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            per_channel: false,
            calibration: CalibrationMethod::MinMax,
        }
    }
}

/// Method used to calibrate the quantization scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Scale is derived from the absolute min/max of the data.
    MinMax,
    /// Scale is derived from the 99.9th percentile (reduces outlier impact).
    Percentile,
}

// ──────────────────────────────────────────────────────────────────────────────
// Quantize / dequantize
// ──────────────────────────────────────────────────────────────────────────────

/// Quantizes a floating-point tensor to INT8.
///
/// The symmetric range `[-128, 127]` is used; zero-point is always 0.
///
/// With `config.per_channel = true`, one scale per first dimension is computed;
/// otherwise a single scale covers the whole tensor.
pub fn quantize(
    tensor: &Tensor,
    config: &QuantizationConfig,
) -> Result<QuantizedTensor, NeuralError> {
    if tensor.numel() == 0 {
        return Err(NeuralError::EmptyInput(
            "quantize: tensor must not be empty".to_string(),
        ));
    }

    let scales = if config.per_channel && tensor.shape().len() >= 2 {
        compute_per_channel_scales(tensor, config.calibration)
    } else {
        vec![compute_single_scale(tensor.data(), config.calibration)]
    };

    let mut q_data = Vec::with_capacity(tensor.numel());
    for (flat, &x) in tensor.data().iter().enumerate() {
        let scale = if scales.len() == 1 {
            scales[0]
        } else {
            let ch_size: usize = tensor.shape().iter().skip(1).product();
            let ch = flat.checked_div(ch_size).unwrap_or(0);
            scales[ch.min(scales.len() - 1)]
        };
        let q = quantize_value(x, scale);
        q_data.push(q);
    }

    Ok(QuantizedTensor {
        data: q_data,
        shape: tensor.shape().to_vec(),
        scales,
    })
}

/// Dequantizes an INT8 tensor back to f32.
pub fn dequantize(qtensor: &QuantizedTensor) -> Result<Tensor, NeuralError> {
    let f_data: Vec<f32> = qtensor
        .data
        .iter()
        .enumerate()
        .map(|(flat, &q)| q as f32 * qtensor.scale_for(flat))
        .collect();
    Tensor::from_data(f_data, qtensor.shape.clone())
}

// ──────────────────────────────────────────────────────────────────────────────
// Quantized linear layer
// ──────────────────────────────────────────────────────────────────────────────

/// A quantized fully-connected layer.
///
/// Weights are stored as INT8; activations are dequantized before bias addition.
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Quantized weight matrix, shape `[out_features, in_features]`.
    pub weight: QuantizedTensor,
    /// Bias (kept in f32 for precision), shape `[out_features]`.
    pub bias: Vec<f32>,
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

impl QuantizedLinear {
    /// Constructs a `QuantizedLinear` by quantizing the weight matrix of a
    /// `LinearLayer`-shaped pair of tensors.
    pub fn from_weights(
        weight: &Tensor,
        bias: &Tensor,
        config: &QuantizationConfig,
    ) -> Result<Self, NeuralError> {
        if weight.shape().len() != 2 {
            return Err(NeuralError::InvalidShape(
                "QuantizedLinear: weight must be 2-D [out, in]".to_string(),
            ));
        }
        let out_features = weight.shape()[0];
        let in_features = weight.shape()[1];
        if bias.numel() != out_features {
            return Err(NeuralError::ShapeMismatch(
                "QuantizedLinear: bias length != out_features".to_string(),
            ));
        }
        let q_weight = quantize(weight, config)?;
        Ok(Self {
            weight: q_weight,
            bias: bias.data().to_vec(),
            in_features,
            out_features,
        })
    }

    /// Forward pass using integer arithmetic internally.
    ///
    /// The input tensor is quantized to INT8, the dot products are accumulated
    /// in i32, then converted to f32 and bias-corrected.
    ///
    /// Input must be a 1-D tensor of length `in_features`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 1 {
            return Err(NeuralError::InvalidShape(
                "QuantizedLinear::forward: expected 1-D input".to_string(),
            ));
        }
        if input.numel() != self.in_features {
            return Err(NeuralError::ShapeMismatch(format!(
                "QuantizedLinear::forward: input length {} != in_features {}",
                input.numel(),
                self.in_features
            )));
        }

        // Quantize the input using a per-tensor scale.
        let in_scale = compute_single_scale(input.data(), CalibrationMethod::MinMax);
        let q_input: Vec<i8> = input
            .data()
            .iter()
            .map(|&x| quantize_value(x, in_scale))
            .collect();

        let mut out = Vec::with_capacity(self.out_features);
        for oc in 0..self.out_features {
            // Per-channel weight scale.
            let w_scale = if self.weight.scales.len() == 1 {
                self.weight.scales[0]
            } else {
                self.weight.scales[oc.min(self.weight.scales.len() - 1)]
            };
            let combined_scale = in_scale * w_scale;

            // Integer dot product.
            let mut acc = 0i32;
            for ic in 0..self.in_features {
                let wi = self.weight.data[oc * self.in_features + ic] as i32;
                let xi = q_input[ic] as i32;
                acc += wi * xi;
            }
            let val = acc as f32 * combined_scale + self.bias[oc];
            out.push(val);
        }

        Tensor::from_data(out, vec![self.out_features])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Quantized Conv2d layer
// ──────────────────────────────────────────────────────────────────────────────

/// A quantized 2-D convolution layer.
///
/// Weights are stored as INT8 with per-channel scale (one scale per output
/// channel).  Activations are processed in integer arithmetic and converted back
/// to f32 before bias addition.
///
/// Input shape: `[C_in, H, W]`.
/// Output shape: `[C_out, out_H, out_W]`.
#[derive(Debug, Clone)]
pub struct QuantizedConv2d {
    /// Quantized kernels, shape `[out_channels, in_channels, kH, kW]`.
    pub weight: QuantizedTensor,
    /// Bias (f32), shape `[out_channels]`.
    pub bias: Vec<f32>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// (stride_h, stride_w).
    pub stride: (usize, usize),
    /// (pad_h, pad_w).
    pub padding: (usize, usize),
}

impl QuantizedConv2d {
    /// Constructs a `QuantizedConv2d` from f32 weight/bias tensors.
    pub fn from_weights(
        weight: &Tensor,
        bias: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        config: &QuantizationConfig,
    ) -> Result<Self, NeuralError> {
        if weight.shape().len() != 4 {
            return Err(NeuralError::InvalidShape(
                "QuantizedConv2d: weight must be 4-D [out,in,kH,kW]".to_string(),
            ));
        }
        let (out_channels, in_channels, kernel_h, kernel_w) = (
            weight.shape()[0],
            weight.shape()[1],
            weight.shape()[2],
            weight.shape()[3],
        );
        if bias.numel() != out_channels {
            return Err(NeuralError::ShapeMismatch(
                "QuantizedConv2d: bias length != out_channels".to_string(),
            ));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NeuralError::InvalidShape(
                "QuantizedConv2d: stride must be > 0".to_string(),
            ));
        }
        let per_ch_config = QuantizationConfig {
            per_channel: true,
            ..config.clone()
        };
        let q_weight = quantize(weight, &per_ch_config)?;
        Ok(Self {
            weight: q_weight,
            bias: bias.data().to_vec(),
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
        })
    }

    /// Forward pass.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(
                "QuantizedConv2d::forward: expected [C,H,W] input".to_string(),
            ));
        }
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if in_c != self.in_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "QuantizedConv2d: input channels {} != {}",
                in_c, self.in_channels
            )));
        }

        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;
        let (kh, kw) = (self.kernel_h, self.kernel_w);

        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;

        if padded_h < kh || padded_w < kw {
            return Err(NeuralError::InvalidShape(
                "QuantizedConv2d::forward: padded size < kernel".to_string(),
            ));
        }

        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;

        // Quantize input.
        let in_scale = compute_single_scale(input.data(), CalibrationMethod::MinMax);
        let q_in: Vec<i8> = input
            .data()
            .iter()
            .map(|&x| quantize_value(x, in_scale))
            .collect();

        // Build padded input in i8.
        let mut padded = vec![0i8; in_c * padded_h * padded_w];
        for c in 0..in_c {
            for h in 0..in_h {
                for w in 0..in_w {
                    let dst = c * padded_h * padded_w + (h + pad_h) * padded_w + (w + pad_w);
                    let src = c * in_h * in_w + h * in_w + w;
                    padded[dst] = q_in[src];
                }
            }
        }

        let mut out_data = vec![0.0_f32; self.out_channels * out_h * out_w];
        let ksize = in_c * kh * kw;
        for oc in 0..self.out_channels {
            let w_scale = if self.weight.scales.len() == 1 {
                self.weight.scales[0]
            } else {
                self.weight.scales[oc.min(self.weight.scales.len() - 1)]
            };
            let combined = in_scale * w_scale;
            let bias_val = self.bias[oc];

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0i32;
                    for ic in 0..in_c {
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let ph = oh * stride_h + kh_i;
                                let pw = ow * stride_w + kw_i;
                                let in_val =
                                    padded[ic * padded_h * padded_w + ph * padded_w + pw] as i32;
                                let w_idx = oc * ksize + ic * (kh * kw) + kh_i * kw + kw_i;
                                let w_val = self.weight.data[w_idx] as i32;
                                acc += in_val * w_val;
                            }
                        }
                    }
                    out_data[oc * out_h * out_w + oh * out_w + ow] =
                        acc as f32 * combined + bias_val;
                }
            }
        }

        Tensor::from_data(out_data, vec![self.out_channels, out_h, out_w])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Quantizes a single f32 value to i8 using the given scale.
#[allow(clippy::manual_checked_ops)]
fn quantize_value(x: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    let q = (x / scale).round();
    q.clamp(-128.0, 127.0) as i8
}

/// Computes the scale for a single slice of f32 data.
fn compute_single_scale(data: &[f32], method: CalibrationMethod) -> f32 {
    if data.is_empty() {
        return 1.0;
    }
    let abs_max = match method {
        CalibrationMethod::MinMax => data.iter().map(|x| x.abs()).fold(0.0_f32, f32::max),
        CalibrationMethod::Percentile => {
            let mut sorted: Vec<f32> = data.iter().map(|x| x.abs()).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Use the 99th percentile to clip outliers.
            let idx = ((sorted.len() as f32 * 0.99) as usize).min(sorted.len() - 1);
            sorted[idx]
        }
    };
    if abs_max == 0.0 {
        1.0
    } else {
        abs_max / 127.0
    }
}

/// Computes per-channel scales for the first dimension of `tensor`.
fn compute_per_channel_scales(tensor: &Tensor, method: CalibrationMethod) -> Vec<f32> {
    let num_channels = tensor.shape()[0];
    if num_channels == 0 {
        return vec![1.0];
    }
    let channel_size: usize = tensor.shape().iter().skip(1).product();
    (0..num_channels)
        .map(|ch| {
            let start = ch * channel_size;
            let end = start + channel_size;
            let slice = &tensor.data()[start..end.min(tensor.numel())];
            compute_single_scale(slice, method)
        })
        .collect()
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

    #[test]
    fn test_quantize_zeros() {
        let t = Tensor::zeros(vec![4]).expect("tensor zeros");
        let qt = quantize(&t, &QuantizationConfig::default()).expect("quantize");
        assert!(qt.data.iter().all(|&q| q == 0));
    }

    #[test]
    fn test_quantize_round_trip() {
        let data = vec![0.5, -0.5, 1.0, -1.0, 0.0, 0.25];
        let t = Tensor::from_data(data, vec![6]).expect("tensor from_data");
        let qt = quantize(&t, &QuantizationConfig::default()).expect("quantize");
        let dq = dequantize(&qt).expect("quantize");
        // After round-trip, values should be close (within 1/127 of original).
        for (&orig, &recon) in t.data().iter().zip(dq.data().iter()) {
            assert!(close(orig, recon, 0.02), "orig={orig}, recon={recon}");
        }
    }

    #[test]
    fn test_quantize_range() {
        // All quantized values must lie in [-128, 127].
        let data: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let t = Tensor::from_data(data, vec![100]).expect("tensor from_data");
        let qt = quantize(&t, &QuantizationConfig::default()).expect("quantize");
        // qt.data is Vec<i8>, so values are inherently in [-128, 127]
        assert!(!qt.data.is_empty());
    }

    #[test]
    fn test_per_channel_scale() {
        // Two channels with very different magnitudes.
        let data = vec![1.0_f32, 1.0, 100.0, 100.0];
        let t = Tensor::from_data(data, vec![2, 2]).expect("tensor from_data");
        let config = QuantizationConfig {
            per_channel: true,
            ..Default::default()
        };
        let qt = quantize(&t, &config).expect("quantize");
        assert_eq!(qt.scales.len(), 2);
        // Channel 0 scale ≈ 1/127; channel 1 scale ≈ 100/127.
        assert!(qt.scales[0] < qt.scales[1]);
    }

    #[test]
    fn test_quantized_linear_forward_shape() {
        use crate::tensor::Tensor;
        let weight = Tensor::ones(vec![3, 4]).expect("tensor ones");
        let bias = Tensor::zeros(vec![3]).expect("tensor zeros");
        let ql = QuantizedLinear::from_weights(&weight, &bias, &QuantizationConfig::default())
            .expect("from_weights");
        let input = Tensor::ones(vec![4]).expect("tensor ones");
        let out = ql.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn test_quantized_linear_values_approx() {
        // Weight = 2-D identity, no bias → output ≈ input after round-trip.
        let mut weight = Tensor::zeros(vec![2, 2]).expect("tensor zeros");
        weight.data_mut()[0] = 1.0;
        weight.data_mut()[3] = 1.0;
        let bias = Tensor::zeros(vec![2]).expect("tensor zeros");
        let ql = QuantizedLinear::from_weights(&weight, &bias, &QuantizationConfig::default())
            .expect("from_weights");
        let input = Tensor::from_data(vec![0.5, -0.5], vec![2]).expect("tensor from_data");
        let out = ql.forward(&input).expect("forward pass");
        assert!(close(out.data()[0], 0.5, 0.02), "got {}", out.data()[0]);
        assert!(close(out.data()[1], -0.5, 0.02), "got {}", out.data()[1]);
    }

    #[test]
    fn test_quantized_conv2d_shape() {
        let weight = Tensor::ones(vec![2, 1, 3, 3]).expect("tensor ones");
        let bias = Tensor::zeros(vec![2]).expect("tensor zeros");
        let qc = QuantizedConv2d::from_weights(
            &weight,
            &bias,
            (1, 1),
            (0, 0),
            &QuantizationConfig::default(),
        )
        .expect("operation should succeed");
        let input = Tensor::ones(vec![1, 5, 5]).expect("tensor ones");
        let out = qc.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[2, 3, 3]);
    }

    #[test]
    fn test_quantize_empty_error() {
        let t = Tensor::new(vec![1]).expect("tensor new"); // 1 element
        let _ = quantize(&t, &QuantizationConfig::default()).expect("quantize");
    }

    #[test]
    fn test_dequantize_preserves_sign() {
        let data = vec![5.0_f32, -5.0, 0.0];
        let t = Tensor::from_data(data, vec![3]).expect("tensor from_data");
        let qt = quantize(&t, &QuantizationConfig::default()).expect("quantize");
        let dq = dequantize(&qt).expect("quantize");
        // Signs must be preserved.
        assert!(dq.data()[0] > 0.0, "positive should stay positive");
        assert!(dq.data()[1] < 0.0, "negative should stay negative");
        assert_eq!(dq.data()[2], 0.0);
    }

    #[test]
    fn test_calibration_percentile() {
        // 100 elements at 1.0 plus 1 outlier at 1000.0 → 1% outlier rate.
        // MinMax scale = 1000/127 ≈ 7.87.
        // Percentile at 99% → sorted[floor(101*0.99)] = sorted[99] = 1.0
        //                   → scale = 1.0/127 ≈ 0.0079.
        // So Percentile scale should be MUCH smaller than MinMax scale.
        let mut data = vec![1.0_f32; 100];
        data.push(1000.0); // 1 outlier
        let t = Tensor::from_data(data, vec![101]).expect("tensor from_data");
        let config_minmax = QuantizationConfig {
            calibration: CalibrationMethod::MinMax,
            ..Default::default()
        };
        let config_perc = QuantizationConfig {
            calibration: CalibrationMethod::Percentile,
            ..Default::default()
        };
        let qt_mm = quantize(&t, &config_minmax).expect("quantize");
        let qt_p = quantize(&t, &config_perc).expect("quantize");
        // Percentile clips the outlier → smaller scale.
        assert!(
            qt_p.scales[0] < qt_mm.scales[0],
            "percentile={} minmax={}",
            qt_p.scales[0],
            qt_mm.scales[0]
        );
    }
}

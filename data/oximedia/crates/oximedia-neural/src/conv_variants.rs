//! Advanced convolution variants: depthwise separable, group convolution,
//! and multi-threaded Conv2d.
//!
//! These are separated from the core `layers` module to keep individual
//! source files under the 2 000-line limit.

use crate::error::NeuralError;
use crate::im2col::im2col as im2col_transform;
use crate::tensor::{matmul_simd_runtime, Tensor};
use rayon::prelude::*;

// ──────────────────────────────────────────────────────────────────────────────
// ParallelConv2d — multi-threaded convolution using rayon
// ──────────────────────────────────────────────────────────────────────────────

/// A 2-D convolution layer that parallelises computation over output channels
/// using rayon.
///
/// Internally it performs im2col once, then dispatches independent GEMM
/// rows (one per output channel) to a rayon thread pool.
///
/// Weight shape: `[out_channels, in_channels, kH, kW]`.
/// Bias shape: `[out_channels]`.
/// Input shape: `[C, H, W]`.
#[derive(Debug, Clone)]
pub struct ParallelConv2d {
    /// Convolution kernels, shape `[out_channels, in_channels, kH, kW]`.
    pub weight: Tensor,
    /// Per-output-channel bias, shape `[out_channels]`.
    pub bias: Tensor,
    /// (stride_h, stride_w).
    pub stride: (usize, usize),
    /// (pad_h, pad_w).
    pub padding: (usize, usize),
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
}

impl ParallelConv2d {
    /// Creates a zero-initialized `ParallelConv2d`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, NeuralError> {
        if in_channels == 0 || out_channels == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NeuralError::InvalidShape(
                "ParallelConv2d: all sizes must be > 0".to_string(),
            ));
        }
        let weight = Tensor::zeros(vec![out_channels, in_channels, kernel_h, kernel_w])?;
        let bias = Tensor::zeros(vec![out_channels])?;
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
        })
    }

    /// Forward pass with rayon-parallel output channels.
    ///
    /// Input shape: `[C, H, W]`.
    /// Output shape: `[out_C, out_H, out_W]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "ParallelConv2d::forward: expected [C,H,W] input, got rank {}",
                input.ndim()
            )));
        }
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if in_c != self.in_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "ParallelConv2d::forward: input channels {} != {}",
                in_c, self.in_channels
            )));
        }

        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;
        let (kh, kw) = (self.kernel_h, self.kernel_w);

        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;

        if padded_h < kh || padded_w < kw {
            return Err(NeuralError::InvalidShape(format!(
                "ParallelConv2d::forward: padded ({},{}) < kernel ({},{})",
                padded_h, padded_w, kh, kw
            )));
        }

        // im2col: [col_rows, col_cols] where col_rows = in_c*kh*kw
        let (col, col_rows, col_cols) = im2col_transform(
            input.data(),
            in_c,
            in_h,
            in_w,
            kh,
            kw,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        );

        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;
        let out_hw = out_h * out_w;

        debug_assert_eq!(col_cols, out_hw);

        // Parallel over output channels: each channel computes one row of the
        // output by doing a vector-matrix multiply:
        //   weight_row[oc] [1, col_rows] @ col [col_rows, col_cols] → [1, col_cols]
        let weight_data = self.weight.data();
        let bias_data = self.bias.data();

        let channel_results: Vec<Vec<f32>> = (0..self.out_channels)
            .into_par_iter()
            .map(|oc| {
                let w_start = oc * col_rows;
                let w_end = w_start + col_rows;
                let w_row = &weight_data[w_start..w_end];
                let bias_val = bias_data[oc];

                // Single-row GEMM: [1, col_rows] @ [col_rows, col_cols] → [1, col_cols]
                let row_out = matmul_simd_runtime(w_row, &col, 1, col_rows, col_cols);

                // Add bias.
                let mut result = row_out;
                for v in result.iter_mut() {
                    *v += bias_val;
                }
                result
            })
            .collect();

        // Assemble into flat output.
        let mut out_data = Vec::with_capacity(self.out_channels * out_hw);
        for ch_data in &channel_results {
            out_data.extend_from_slice(ch_data);
        }

        Tensor::from_data(out_data, vec![self.out_channels, out_h, out_w])
    }

    /// Batched forward: `[N, C, H, W]` → `[N, out_C, out_H, out_W]`.
    pub fn forward_batch(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        match input.ndim() {
            3 => self.forward(input),
            4 => {
                let n = input.shape()[0];
                let (in_c, in_h, in_w) = (input.shape()[1], input.shape()[2], input.shape()[3]);
                let in_spatial = in_c * in_h * in_w;

                let first_sample =
                    Tensor::from_data(input.data()[..in_spatial].to_vec(), vec![in_c, in_h, in_w])?;
                let first_out = self.forward(&first_sample)?;
                let out_shape_single = first_out.shape().to_vec();
                let out_spatial: usize = out_shape_single.iter().product();

                let mut out_data = Vec::with_capacity(n * out_spatial);
                out_data.extend_from_slice(first_out.data());

                for bi in 1..n {
                    let start = bi * in_spatial;
                    let sample = Tensor::from_data(
                        input.data()[start..start + in_spatial].to_vec(),
                        vec![in_c, in_h, in_w],
                    )?;
                    let sample_out = self.forward(&sample)?;
                    out_data.extend_from_slice(sample_out.data());
                }

                let mut batch_shape = vec![n];
                batch_shape.extend_from_slice(&out_shape_single);
                Tensor::from_data(out_data, batch_shape)
            }
            other => Err(NeuralError::InvalidShape(format!(
                "ParallelConv2d::forward_batch: expected rank 3 or 4, got {}",
                other
            ))),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DepthwiseSeparableConv2d — depthwise + pointwise
// ──────────────────────────────────────────────────────────────────────────────

/// Depthwise separable convolution: a depthwise convolution (one filter per
/// input channel) followed by a 1x1 pointwise convolution that mixes channels.
///
/// This factorisation reduces parameters from `O(C_in * C_out * k^2)` to
/// `O(C_in * k^2 + C_in * C_out)`.
///
/// Input shape: `[C_in, H, W]`.
/// Output shape: `[C_out, out_H, out_W]`.
#[derive(Debug, Clone)]
pub struct DepthwiseSeparableConv2d {
    /// Depthwise kernels, shape `[in_channels, 1, kH, kW]`.
    pub dw_weight: Tensor,
    /// Depthwise bias, shape `[in_channels]`.
    pub dw_bias: Tensor,
    /// Pointwise (1x1) weight, shape `[out_channels, in_channels]`.
    pub pw_weight: Tensor,
    /// Pointwise bias, shape `[out_channels]`.
    pub pw_bias: Tensor,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel height (depthwise).
    pub kernel_h: usize,
    /// Kernel width (depthwise).
    pub kernel_w: usize,
    /// (stride_h, stride_w) for depthwise.
    pub stride: (usize, usize),
    /// (pad_h, pad_w) for depthwise.
    pub padding: (usize, usize),
}

impl DepthwiseSeparableConv2d {
    /// Creates a zero-initialized depthwise separable convolution.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, NeuralError> {
        if in_channels == 0 || out_channels == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NeuralError::InvalidShape(
                "DepthwiseSeparableConv2d: all sizes must be > 0".to_string(),
            ));
        }
        let dw_weight = Tensor::zeros(vec![in_channels, 1, kernel_h, kernel_w])?;
        let dw_bias = Tensor::zeros(vec![in_channels])?;
        let pw_weight = Tensor::zeros(vec![out_channels, in_channels])?;
        let pw_bias = Tensor::zeros(vec![out_channels])?;

        Ok(Self {
            dw_weight,
            dw_bias,
            pw_weight,
            pw_bias,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
        })
    }

    /// Forward pass: depthwise convolution followed by 1x1 pointwise.
    ///
    /// Input: `[C_in, H, W]` → Output: `[C_out, out_H, out_W]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "DepthwiseSeparableConv2d::forward: expected [C,H,W], got rank {}",
                input.ndim()
            )));
        }
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if in_c != self.in_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "DepthwiseSeparableConv2d::forward: channels {} != {}",
                in_c, self.in_channels
            )));
        }

        // Step 1: Depthwise convolution.
        let dw_out = self.depthwise_forward(input, in_c, in_h, in_w)?;
        let out_h = dw_out.shape()[1];
        let out_w = dw_out.shape()[2];

        // Step 2: Pointwise (1x1) convolution.
        // dw_out shape: [in_channels, out_h, out_w]
        // Reshape to [in_channels, out_h*out_w], then:
        //   pw_weight [out_channels, in_channels] @ dw_reshaped [in_channels, out_h*out_w]
        //   = [out_channels, out_h*out_w]
        let spatial = out_h * out_w;
        let pw_out = matmul_simd_runtime(
            self.pw_weight.data(),
            dw_out.data(),
            self.out_channels,
            self.in_channels,
            spatial,
        );

        // Add pointwise bias.
        let mut out_data = pw_out;
        for oc in 0..self.out_channels {
            let bias_val = self.pw_bias.data()[oc];
            let base = oc * spatial;
            for i in 0..spatial {
                out_data[base + i] += bias_val;
            }
        }

        Tensor::from_data(out_data, vec![self.out_channels, out_h, out_w])
    }

    /// Internal depthwise convolution step.
    fn depthwise_forward(
        &self,
        input: &Tensor,
        in_c: usize,
        in_h: usize,
        in_w: usize,
    ) -> Result<Tensor, NeuralError> {
        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;
        let (kh, kw) = (self.kernel_h, self.kernel_w);

        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;

        if padded_h < kh || padded_w < kw {
            return Err(NeuralError::InvalidShape(
                "DepthwiseSeparableConv2d: padded size < kernel".to_string(),
            ));
        }

        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;

        // Build padded input.
        let mut padded = vec![0.0_f32; in_c * padded_h * padded_w];
        for c in 0..in_c {
            for h in 0..in_h {
                for w in 0..in_w {
                    let dst = c * padded_h * padded_w + (h + pad_h) * padded_w + (w + pad_w);
                    let src = c * in_h * in_w + h * in_w + w;
                    padded[dst] = input.data()[src];
                }
            }
        }

        let mut out_data = vec![0.0_f32; in_c * out_h * out_w];
        for c in 0..in_c {
            let bias_val = self.dw_bias.data()[c];
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias_val;
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ph = oh * stride_h + kh_i;
                            let pw = ow * stride_w + kw_i;
                            let in_val = padded[c * padded_h * padded_w + ph * padded_w + pw];
                            let w_idx = c * (kh * kw) + kh_i * kw + kw_i;
                            acc += in_val * self.dw_weight.data()[w_idx];
                        }
                    }
                    out_data[c * out_h * out_w + oh * out_w + ow] = acc;
                }
            }
        }

        Tensor::from_data(out_data, vec![in_c, out_h, out_w])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GroupConv2d — group convolution
// ──────────────────────────────────────────────────────────────────────────────

/// Group convolution: input and output channels are divided into `groups`
/// equal partitions. Each group's channels are convolved independently.
///
/// When `groups == in_channels == out_channels`, this is a depthwise convolution.
/// When `groups == 1`, this is a standard convolution.
///
/// Weight shape: `[out_channels, in_channels / groups, kH, kW]`.
/// Bias shape: `[out_channels]`.
/// Input shape: `[C_in, H, W]`.
#[derive(Debug, Clone)]
pub struct GroupConv2d {
    /// Convolution kernels, shape `[out_channels, in_channels/groups, kH, kW]`.
    pub weight: Tensor,
    /// Per-output-channel bias, shape `[out_channels]`.
    pub bias: Tensor,
    /// Number of groups.
    pub groups: usize,
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

impl GroupConv2d {
    /// Creates a zero-initialized `GroupConv2d`.
    ///
    /// Both `in_channels` and `out_channels` must be divisible by `groups`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        groups: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, NeuralError> {
        if in_channels == 0 || out_channels == 0 || groups == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NeuralError::InvalidShape(
                "GroupConv2d: all sizes must be > 0".to_string(),
            ));
        }
        if in_channels % groups != 0 {
            return Err(NeuralError::InvalidShape(format!(
                "GroupConv2d: in_channels {} must be divisible by groups {}",
                in_channels, groups
            )));
        }
        if out_channels % groups != 0 {
            return Err(NeuralError::InvalidShape(format!(
                "GroupConv2d: out_channels {} must be divisible by groups {}",
                out_channels, groups
            )));
        }
        let ic_per_group = in_channels / groups;
        let weight = Tensor::zeros(vec![out_channels, ic_per_group, kernel_h, kernel_w])?;
        let bias = Tensor::zeros(vec![out_channels])?;
        Ok(Self {
            weight,
            bias,
            groups,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
        })
    }

    /// Forward pass: applies group convolution.
    ///
    /// Input: `[C_in, H, W]` → Output: `[C_out, out_H, out_W]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "GroupConv2d::forward: expected [C,H,W], got rank {}",
                input.ndim()
            )));
        }
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if in_c != self.in_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "GroupConv2d::forward: channels {} != {}",
                in_c, self.in_channels
            )));
        }

        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;
        let (kh, kw) = (self.kernel_h, self.kernel_w);

        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;
        if padded_h < kh || padded_w < kw {
            return Err(NeuralError::InvalidShape(format!(
                "GroupConv2d::forward: padded ({},{}) < kernel ({},{})",
                padded_h, padded_w, kh, kw
            )));
        }

        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;
        let out_hw = out_h * out_w;

        let ic_per_group = self.in_channels / self.groups;
        let oc_per_group = self.out_channels / self.groups;

        let mut out_data = vec![0.0_f32; self.out_channels * out_hw];

        for g in 0..self.groups {
            let ic_start = g * ic_per_group;

            // Extract group's input channels and do im2col.
            let mut group_input = vec![0.0_f32; ic_per_group * in_h * in_w];
            for c in 0..ic_per_group {
                let src_base = (ic_start + c) * in_h * in_w;
                let dst_base = c * in_h * in_w;
                group_input[dst_base..dst_base + in_h * in_w]
                    .copy_from_slice(&input.data()[src_base..src_base + in_h * in_w]);
            }

            let (col, col_rows, col_cols) = im2col_transform(
                &group_input,
                ic_per_group,
                in_h,
                in_w,
                kh,
                kw,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
            );

            debug_assert_eq!(col_rows, ic_per_group * kh * kw);
            debug_assert_eq!(col_cols, out_hw);

            // Extract group's weight: [oc_per_group, ic_per_group*kh*kw]
            let oc_start = g * oc_per_group;
            let w_row_len = ic_per_group * kh * kw;
            let group_w_start = oc_start * w_row_len;
            let group_w_end = group_w_start + oc_per_group * w_row_len;
            let group_weight = &self.weight.data()[group_w_start..group_w_end];

            // GEMM: [oc_per_group, col_rows] @ [col_rows, col_cols]
            let gemm_out =
                matmul_simd_runtime(group_weight, &col, oc_per_group, col_rows, col_cols);

            // Copy results and add bias.
            for oc_local in 0..oc_per_group {
                let oc_global = oc_start + oc_local;
                let bias_val = self.bias.data()[oc_global];
                let src_base = oc_local * out_hw;
                let dst_base = oc_global * out_hw;
                for i in 0..out_hw {
                    out_data[dst_base + i] = gemm_out[src_base + i] + bias_val;
                }
            }
        }

        Tensor::from_data(out_data, vec![self.out_channels, out_h, out_w])
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

    // ── ParallelConv2d ──────────────────────────────────────────────────────

    #[test]
    fn test_parallel_conv2d_output_shape() {
        let layer = ParallelConv2d::new(1, 2, 3, 3, (1, 1), (1, 1)).expect("ok");
        let input = Tensor::ones(vec![1, 5, 5]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        assert_eq!(out.shape(), &[2, 5, 5]);
    }

    #[test]
    fn test_parallel_conv2d_zero_weight_is_bias() {
        let mut layer = ParallelConv2d::new(1, 1, 3, 3, (1, 1), (1, 1)).expect("ok");
        layer.bias.data_mut()[0] = 3.0;
        let input = Tensor::ones(vec![1, 4, 4]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        assert_eq!(out.shape(), &[1, 4, 4]);
        assert!(out.data().iter().all(|&v| close(v, 3.0)));
    }

    #[test]
    fn test_parallel_conv2d_batch() {
        let layer = ParallelConv2d::new(1, 2, 3, 3, (1, 1), (1, 1)).expect("ok");
        let input = Tensor::ones(vec![2, 1, 4, 4]).expect("ok");
        let out = layer.forward_batch(&input).expect("ok");
        assert_eq!(out.shape(), &[2, 2, 4, 4]);
    }

    #[test]
    fn test_parallel_conv2d_stride2() {
        let layer = ParallelConv2d::new(1, 1, 2, 2, (2, 2), (0, 0)).expect("ok");
        let input = Tensor::ones(vec![1, 6, 6]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        assert_eq!(out.shape(), &[1, 3, 3]);
    }

    #[test]
    fn test_parallel_conv2d_channel_mismatch() {
        let layer = ParallelConv2d::new(3, 1, 3, 3, (1, 1), (0, 0)).expect("ok");
        let input = Tensor::ones(vec![1, 5, 5]).expect("ok");
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_parallel_conv2d_matches_sequential() {
        // ParallelConv2d with known weights should produce the same result
        // as a sequential computation.
        let mut layer = ParallelConv2d::new(1, 2, 1, 1, (1, 1), (0, 0)).expect("ok");
        // Set weights: oc0 = 2.0, oc1 = 3.0
        layer.weight.data_mut()[0] = 2.0;
        layer.weight.data_mut()[1] = 3.0;
        layer.bias.data_mut()[0] = 0.5;
        layer.bias.data_mut()[1] = -0.5;

        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        assert_eq!(out.shape(), &[2, 2, 2]);
        // oc0: 2*x + 0.5
        assert!(close(out.data()[0], 2.5));
        assert!(close(out.data()[1], 4.5));
        assert!(close(out.data()[2], 6.5));
        assert!(close(out.data()[3], 8.5));
        // oc1: 3*x - 0.5
        assert!(close(out.data()[4], 2.5));
        assert!(close(out.data()[5], 5.5));
        assert!(close(out.data()[6], 8.5));
        assert!(close(out.data()[7], 11.5));
    }

    // ── DepthwiseSeparableConv2d ────────────────────────────────────────────

    #[test]
    fn test_dw_sep_output_shape() {
        let layer = DepthwiseSeparableConv2d::new(3, 8, 3, 3, (1, 1), (1, 1)).expect("ok");
        let input = Tensor::ones(vec![3, 8, 8]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        assert_eq!(out.shape(), &[8, 8, 8]);
    }

    #[test]
    fn test_dw_sep_zero_weight_bias_only() {
        let mut layer = DepthwiseSeparableConv2d::new(2, 3, 1, 1, (1, 1), (0, 0)).expect("ok");
        // dw_weight all zero, dw_bias = [1.0, 2.0]
        layer.dw_bias.data_mut()[0] = 1.0;
        layer.dw_bias.data_mut()[1] = 2.0;
        // pw_weight all zero, pw_bias = [10.0, 20.0, 30.0]
        layer.pw_bias.data_mut()[0] = 10.0;
        layer.pw_bias.data_mut()[1] = 20.0;
        layer.pw_bias.data_mut()[2] = 30.0;
        let input = Tensor::ones(vec![2, 3, 3]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        assert_eq!(out.shape(), &[3, 3, 3]);
        // All outputs should be pw_bias since dw produces only bias, and pw_weight is zero.
        let spatial = 9;
        assert!(out.data()[..spatial].iter().all(|&v| close(v, 10.0)));
        assert!(out.data()[spatial..2 * spatial]
            .iter()
            .all(|&v| close(v, 20.0)));
        assert!(out.data()[2 * spatial..].iter().all(|&v| close(v, 30.0)));
    }

    #[test]
    fn test_dw_sep_channel_mismatch() {
        let layer = DepthwiseSeparableConv2d::new(3, 8, 3, 3, (1, 1), (1, 1)).expect("ok");
        let input = Tensor::ones(vec![2, 8, 8]).expect("ok");
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_dw_sep_stride() {
        let layer = DepthwiseSeparableConv2d::new(2, 4, 3, 3, (2, 2), (0, 0)).expect("ok");
        let input = Tensor::ones(vec![2, 7, 7]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        // out_h = (7 - 3) / 2 + 1 = 3
        assert_eq!(out.shape(), &[4, 3, 3]);
    }

    // ── GroupConv2d ─────────────────────────────────────────────────────────

    #[test]
    fn test_group_conv_output_shape() {
        let layer = GroupConv2d::new(4, 8, 2, 3, 3, (1, 1), (1, 1)).expect("ok");
        let input = Tensor::ones(vec![4, 6, 6]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        assert_eq!(out.shape(), &[8, 6, 6]);
    }

    #[test]
    fn test_group_conv_groups1_like_standard() {
        // groups=1 should behave like standard conv2d.
        let mut layer = GroupConv2d::new(1, 2, 1, 1, 1, (1, 1), (0, 0)).expect("ok");
        layer.weight.data_mut()[0] = 1.0;
        layer.weight.data_mut()[1] = 2.0;
        layer.bias.data_mut()[0] = 0.5;
        layer.bias.data_mut()[1] = -0.5;
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        assert_eq!(out.shape(), &[2, 2, 2]);
        // oc0: 1*x + 0.5
        assert!(close(out.data()[0], 1.5));
        assert!(close(out.data()[1], 2.5));
        // oc1: 2*x - 0.5
        assert!(close(out.data()[4], 1.5));
        assert!(close(out.data()[5], 3.5));
    }

    #[test]
    fn test_group_conv_zero_weight_is_bias() {
        let mut layer = GroupConv2d::new(4, 4, 4, 3, 3, (1, 1), (1, 1)).expect("ok");
        for (i, v) in layer.bias.data_mut().iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
        let input = Tensor::ones(vec![4, 5, 5]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        assert_eq!(out.shape(), &[4, 5, 5]);
        let spatial = 25;
        for oc in 0..4 {
            let expected = (oc + 1) as f32;
            let base = oc * spatial;
            assert!(
                out.data()[base..base + spatial]
                    .iter()
                    .all(|&v| close(v, expected)),
                "oc {} failed",
                oc
            );
        }
    }

    #[test]
    fn test_group_conv_indivisible_in_channels() {
        assert!(GroupConv2d::new(5, 10, 3, 3, 3, (1, 1), (0, 0)).is_err());
    }

    #[test]
    fn test_group_conv_indivisible_out_channels() {
        assert!(GroupConv2d::new(6, 7, 3, 3, 3, (1, 1), (0, 0)).is_err());
    }

    #[test]
    fn test_group_conv_channel_mismatch() {
        let layer = GroupConv2d::new(4, 8, 2, 3, 3, (1, 1), (0, 0)).expect("ok");
        let input = Tensor::ones(vec![3, 6, 6]).expect("ok");
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_group_conv_depthwise_case() {
        // groups == in_channels == out_channels → depthwise
        let mut layer = GroupConv2d::new(3, 3, 3, 1, 1, (1, 1), (0, 0)).expect("ok");
        // Each group has 1 in_channel, 1 out_channel, 1x1 kernel
        // weight shape: [3, 1, 1, 1]
        layer.weight.data_mut()[0] = 1.0;
        layer.weight.data_mut()[1] = 2.0;
        layer.weight.data_mut()[2] = 3.0;
        let input = Tensor::ones(vec![3, 2, 2]).expect("ok");
        let out = layer.forward(&input).expect("ok");
        assert_eq!(out.shape(), &[3, 2, 2]);
        // ch0: 1*1 = 1, ch1: 2*1 = 2, ch2: 3*1 = 3
        assert!(out.data()[..4].iter().all(|&v| close(v, 1.0)));
        assert!(out.data()[4..8].iter().all(|&v| close(v, 2.0)));
        assert!(out.data()[8..12].iter().all(|&v| close(v, 3.0)));
    }
}

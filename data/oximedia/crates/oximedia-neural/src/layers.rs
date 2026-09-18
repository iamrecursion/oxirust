//! Neural network layer implementations.
//!
//! All layers are stateless with respect to training; they only perform
//! forward inference passes.

use crate::error::NeuralError;
use crate::im2col::im2col as im2col_transform;
use crate::tensor::{add, matmul, matmul_simd_runtime, Tensor};

// ──────────────────────────────────────────────────────────────────────────────
// LinearLayer
// ──────────────────────────────────────────────────────────────────────────────

/// A fully-connected (dense) linear layer: `output = W * input + b`.
///
/// Weight matrix has shape `[out_features, in_features]`;
/// bias vector has shape `[out_features]`.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Weight matrix, shape `[out_features, in_features]`.
    pub weight: Tensor,
    /// Bias vector, shape `[out_features]`.
    pub bias: Tensor,
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

impl LinearLayer {
    /// Creates a zero-initialized `LinearLayer`.
    pub fn new(in_features: usize, out_features: usize) -> Result<Self, NeuralError> {
        if in_features == 0 || out_features == 0 {
            return Err(NeuralError::InvalidShape(
                "LinearLayer: feature sizes must be > 0".to_string(),
            ));
        }
        let weight = Tensor::zeros(vec![out_features, in_features])?;
        let bias = Tensor::zeros(vec![out_features])?;
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Forward pass: computes `W * x + b`.
    ///
    /// `input` must be a 1-D tensor of length `in_features`.
    /// Returns a 1-D tensor of length `out_features`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 1 {
            return Err(NeuralError::InvalidShape(format!(
                "LinearLayer::forward: expected 1-D input, got rank {}",
                input.ndim()
            )));
        }
        if input.numel() != self.in_features {
            return Err(NeuralError::ShapeMismatch(format!(
                "LinearLayer::forward: input length {} does not match in_features {}",
                input.numel(),
                self.in_features
            )));
        }
        // Reshape input to [in_features, 1] for matmul.
        let x_col = input.reshape(vec![self.in_features, 1])?;
        // W: [out, in] @ x_col: [in, 1] → out_col: [out, 1]
        let out_col = matmul(&self.weight, &x_col)?;
        // Reshape [out, 1] → [out]
        let out_flat = out_col.reshape(vec![self.out_features])?;
        // Add bias (both 1-D with same length).
        add(&out_flat, &self.bias)
    }

    /// Batched forward pass: processes all samples in a batch independently.
    ///
    /// * `input` — 2-D tensor of shape `[batch_size, in_features]`.
    ///
    /// Returns a 2-D tensor of shape `[batch_size, out_features]`.
    ///
    /// Each row is passed through [`forward`](Self::forward) individually and
    /// the results are concatenated along the batch dimension, preserving
    /// row-major ordering.
    pub fn forward_batch(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 2 {
            return Err(NeuralError::InvalidShape(format!(
                "LinearLayer::forward_batch: expected 2-D input [B, in_features], got rank {}",
                input.ndim()
            )));
        }
        let batch_size = input.shape()[0];
        let in_len = input.shape()[1];
        if in_len != self.in_features {
            return Err(NeuralError::ShapeMismatch(format!(
                "LinearLayer::forward_batch: input feature dim {} != in_features {}",
                in_len, self.in_features
            )));
        }

        // Process each sample independently and collect into a flat output buffer.
        let mut out_data = Vec::with_capacity(batch_size * self.out_features);
        for b in 0..batch_size {
            let sample_data = input.data()[b * in_len..(b + 1) * in_len].to_vec();
            let sample = Tensor::from_data(sample_data, vec![in_len])?;
            let out_sample = self.forward(&sample)?;
            out_data.extend_from_slice(out_sample.data());
        }

        Tensor::from_data(out_data, vec![batch_size, self.out_features])
    }

    /// Batched forward pass accepting an N-D batch tensor.
    ///
    /// * For a 2-D input `[N, in_features]`, delegates to [`forward_batch`](Self::forward_batch).
    /// * For higher-dimensional input (e.g. `[N, T, in_features]`), the leading
    ///   dimensions are merged into a single batch axis, linear transformation is
    ///   applied, then the shape is restored with `out_features` replacing the
    ///   last dimension.
    ///
    /// Returns a tensor with the same leading shape as `batch` but the last
    /// dimension replaced by `out_features`.
    pub fn forward_batch_tensor(&self, batch: &Tensor) -> Result<Tensor, NeuralError> {
        if batch.ndim() < 2 {
            return Err(NeuralError::InvalidShape(format!(
                "LinearLayer::forward_batch_tensor: expected at least 2-D input, got rank {}",
                batch.ndim()
            )));
        }
        let last_dim = batch.shape()[batch.ndim() - 1];
        if last_dim != self.in_features {
            return Err(NeuralError::ShapeMismatch(format!(
                "LinearLayer::forward_batch_tensor: last dim {} != in_features {}",
                last_dim, self.in_features
            )));
        }
        if batch.ndim() == 2 {
            return self.forward_batch(batch);
        }
        // Flatten all leading dims into one batch axis.
        let leading: usize = batch.shape()[..batch.ndim() - 1].iter().product();
        let flat = batch.reshape(vec![leading, self.in_features])?;
        let out_flat = self.forward_batch(&flat)?; // [leading, out_features]

        // Restore original leading shape with out_features at the end.
        let mut out_shape = batch.shape()[..batch.ndim() - 1].to_vec();
        out_shape.push(self.out_features);
        out_flat.reshape(out_shape)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Conv2dLayer
// ──────────────────────────────────────────────────────────────────────────────

/// A 2-D convolution layer.
///
/// Weight tensor has shape `[out_channels, in_channels, kH, kW]`.
/// Bias tensor has shape `[out_channels]`.
/// Input tensor is expected in `[C, H, W]` (channels-first) format.
#[derive(Debug, Clone)]
pub struct Conv2dLayer {
    /// Convolution kernels, shape `[out_channels, in_channels, kH, kW]`.
    pub weight: Tensor,
    /// Per-output-channel bias, shape `[out_channels]`.
    pub bias: Tensor,
    /// (stride_h, stride_w).
    pub stride: (usize, usize),
    /// (pad_h, pad_w) — zero-padding added to each side.
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

impl Conv2dLayer {
    /// Creates a zero-initialized `Conv2dLayer`.
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
                "Conv2dLayer: all sizes must be > 0".to_string(),
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

    /// Forward pass via im2col + GEMM.
    ///
    /// Input shape: `[C, H, W]`.
    /// Output shape: `[out_C, out_H, out_W]` where:
    /// - `out_H = (H + 2*pad_h - kH) / stride_h + 1`
    /// - `out_W = (W + 2*pad_w - kW) / stride_w + 1`
    ///
    /// The convolution is computed as a single matrix multiplication:
    ///   `weight_mat [out_C, in_C*kH*kW] @ col_mat [in_C*kH*kW, out_H*out_W]`
    /// followed by bias addition, using [`matmul_simd_runtime`] for the GEMM.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "Conv2dLayer::forward: expected [C,H,W] input, got rank {}",
                input.ndim()
            )));
        }
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if in_c != self.in_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "Conv2dLayer::forward: input channels {} != layer in_channels {}",
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
                "Conv2dLayer::forward: padded input ({},{}) is smaller than kernel ({},{})",
                padded_h, padded_w, kh, kw
            )));
        }

        // ── Step 1: im2col ────────────────────────────────────────────────────
        // col shape: [col_rows = in_c*kh*kw,  col_cols = out_h*out_w]
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

        debug_assert_eq!(col_rows, in_c * kh * kw);
        debug_assert_eq!(col_cols, out_hw);

        // ── Step 2: GEMM ──────────────────────────────────────────────────────
        // weight_mat shape: [out_channels, col_rows]  (weights stored row-major,
        //                    each row = one output filter flattened over in_c*kh*kw)
        // col        shape: [col_rows, col_cols]
        // gemm_out   shape: [out_channels, col_cols]  = [out_C, out_h*out_w]
        let gemm_out = matmul_simd_runtime(
            self.weight.data(),
            &col,
            self.out_channels,
            col_rows,
            col_cols,
        );

        // ── Step 3: add bias ──────────────────────────────────────────────────
        // bias shape: [out_channels]; broadcast over spatial positions.
        let bias_data = self.bias.data();
        let mut out_data = gemm_out;
        for oc in 0..self.out_channels {
            let bias_val = bias_data[oc];
            let base = oc * out_hw;
            for pos in 0..out_hw {
                out_data[base + pos] += bias_val;
            }
        }

        Tensor::from_data(out_data, vec![self.out_channels, out_h, out_w])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BatchNorm1d
// ──────────────────────────────────────────────────────────────────────────────

/// 1-D Batch Normalization layer (inference mode, uses running statistics).
///
/// Normalizes a 1-D feature vector using pre-computed running mean and variance.
#[derive(Debug, Clone)]
pub struct BatchNorm1d {
    /// Number of features (channels).
    pub num_features: usize,
    /// Learnable scale parameter `γ`, one per feature.
    pub weight: Vec<f32>,
    /// Learnable shift parameter `β`, one per feature.
    pub bias: Vec<f32>,
    /// Running mean (updated during training, fixed at inference).
    pub running_mean: Vec<f32>,
    /// Running variance (updated during training, fixed at inference).
    pub running_var: Vec<f32>,
    /// Small constant added to variance to prevent division by zero.
    pub eps: f32,
}

impl BatchNorm1d {
    /// Creates a `BatchNorm1d` with identity scale, zero shift, zero mean, and
    /// unit variance — effectively a no-op at initialization.
    pub fn new(num_features: usize) -> Result<Self, NeuralError> {
        if num_features == 0 {
            return Err(NeuralError::InvalidShape(
                "BatchNorm1d: num_features must be > 0".to_string(),
            ));
        }
        Ok(Self {
            num_features,
            weight: vec![1.0_f32; num_features],
            bias: vec![0.0_f32; num_features],
            running_mean: vec![0.0_f32; num_features],
            running_var: vec![1.0_f32; num_features],
            eps: 1e-5,
        })
    }

    /// Forward pass (inference).
    ///
    /// `input` must be a 1-D tensor of length `num_features`.
    /// Computes `γ * (x - mean) / sqrt(var + eps) + β`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 1 {
            return Err(NeuralError::InvalidShape(format!(
                "BatchNorm1d::forward: expected 1-D input, got rank {}",
                input.ndim()
            )));
        }
        if input.numel() != self.num_features {
            return Err(NeuralError::ShapeMismatch(format!(
                "BatchNorm1d::forward: input length {} != num_features {}",
                input.numel(),
                self.num_features
            )));
        }
        let out_data: Vec<f32> = (0..self.num_features)
            .map(|i| {
                let x_norm = (input.data()[i] - self.running_mean[i])
                    / (self.running_var[i] + self.eps).sqrt();
                self.weight[i] * x_norm + self.bias[i]
            })
            .collect();
        Tensor::from_data(out_data, vec![self.num_features])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BatchNorm2d
// ──────────────────────────────────────────────────────────────────────────────

/// 2-D Batch Normalization layer (inference mode, uses running statistics).
///
/// Normalises a `[C, H, W]` (or `[N, C, H, W]`) feature map channel-by-channel
/// using pre-computed running mean and variance.  The per-channel learnable
/// scale `γ` and shift `β` are applied after normalisation.
#[derive(Debug, Clone)]
pub struct BatchNorm2d {
    /// Number of channels.
    pub num_features: usize,
    /// Learnable scale parameter `γ`, shape `[num_features]`.
    pub weight: Vec<f32>,
    /// Learnable shift parameter `β`, shape `[num_features]`.
    pub bias: Vec<f32>,
    /// Running mean, shape `[num_features]`.
    pub running_mean: Vec<f32>,
    /// Running variance, shape `[num_features]`.
    pub running_var: Vec<f32>,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

impl BatchNorm2d {
    /// Creates a `BatchNorm2d` with identity scale, zero shift, zero mean, and
    /// unit variance — effectively a no-op at initialization.
    pub fn new(num_features: usize) -> Result<Self, NeuralError> {
        if num_features == 0 {
            return Err(NeuralError::InvalidShape(
                "BatchNorm2d: num_features must be > 0".to_string(),
            ));
        }
        Ok(Self {
            num_features,
            weight: vec![1.0_f32; num_features],
            bias: vec![0.0_f32; num_features],
            running_mean: vec![0.0_f32; num_features],
            running_var: vec![1.0_f32; num_features],
            eps: 1e-5,
        })
    }

    /// Forward pass (inference).
    ///
    /// Accepts `[C, H, W]` or `[N, C, H, W]` tensors.
    /// For each channel `c`, computes:
    ///   `γ[c] * (x - mean[c]) / sqrt(var[c] + eps) + β[c]`
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        match input.ndim() {
            3 => {
                let (c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
                if c != self.num_features {
                    return Err(NeuralError::ShapeMismatch(format!(
                        "BatchNorm2d::forward: input channels {} != num_features {}",
                        c, self.num_features
                    )));
                }
                let spatial = h * w;
                let mut out_data = vec![0.0_f32; c * spatial];
                for ch in 0..c {
                    let gamma = self.weight[ch];
                    let beta = self.bias[ch];
                    let mean = self.running_mean[ch];
                    let inv_std = 1.0 / (self.running_var[ch] + self.eps).sqrt();
                    let base = ch * spatial;
                    for i in 0..spatial {
                        out_data[base + i] =
                            gamma * (input.data()[base + i] - mean) * inv_std + beta;
                    }
                }
                Tensor::from_data(out_data, vec![c, h, w])
            }
            4 => {
                let (n, c, h, w) = (
                    input.shape()[0],
                    input.shape()[1],
                    input.shape()[2],
                    input.shape()[3],
                );
                if c != self.num_features {
                    return Err(NeuralError::ShapeMismatch(format!(
                        "BatchNorm2d::forward: input channels {} != num_features {}",
                        c, self.num_features
                    )));
                }
                let spatial = h * w;
                let sample_size = c * spatial;
                let mut out_data = vec![0.0_f32; n * sample_size];
                for bi in 0..n {
                    for ch in 0..c {
                        let gamma = self.weight[ch];
                        let beta = self.bias[ch];
                        let mean = self.running_mean[ch];
                        let inv_std = 1.0 / (self.running_var[ch] + self.eps).sqrt();
                        let in_base = bi * sample_size + ch * spatial;
                        let out_base = bi * sample_size + ch * spatial;
                        for i in 0..spatial {
                            out_data[out_base + i] =
                                gamma * (input.data()[in_base + i] - mean) * inv_std + beta;
                        }
                    }
                }
                Tensor::from_data(out_data, vec![n, c, h, w])
            }
            other => Err(NeuralError::InvalidShape(format!(
                "BatchNorm2d::forward: expected 3-D [C,H,W] or 4-D [N,C,H,W] input, got rank {}",
                other
            ))),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DepthwiseConv2d
// ──────────────────────────────────────────────────────────────────────────────

/// Depthwise 2-D convolution: each input channel is convolved with its own
/// single-channel kernel (groups = in_channels).
///
/// Weight shape: `[in_channels, 1, kH, kW]` (one kernel per channel).
/// Bias shape: `[in_channels]`.
/// Input / output channel count is identical: `in_channels = out_channels`.
#[derive(Debug, Clone)]
pub struct DepthwiseConv2d {
    /// Per-channel kernels, shape `[channels, 1, kH, kW]`.
    pub weight: Tensor,
    /// Per-channel bias, shape `[channels]`.
    pub bias: Tensor,
    /// Number of channels (= groups).
    pub channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// (stride_h, stride_w).
    pub stride: (usize, usize),
    /// (pad_h, pad_w).
    pub padding: (usize, usize),
}

impl DepthwiseConv2d {
    /// Creates a zero-initialized `DepthwiseConv2d`.
    pub fn new(
        channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, NeuralError> {
        if channels == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NeuralError::InvalidShape(
                "DepthwiseConv2d: all sizes must be > 0".to_string(),
            ));
        }
        let weight = Tensor::zeros(vec![channels, 1, kernel_h, kernel_w])?;
        let bias = Tensor::zeros(vec![channels])?;
        Ok(Self {
            weight,
            bias,
            channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
        })
    }

    /// Forward pass.
    ///
    /// Input shape: `[C, H, W]`.
    /// Output shape: `[C, out_H, out_W]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "DepthwiseConv2d::forward: expected [C,H,W] input, got rank {}",
                input.ndim()
            )));
        }
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if in_c != self.channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "DepthwiseConv2d::forward: input channels {} != layer channels {}",
                in_c, self.channels
            )));
        }

        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;
        let (kh, kw) = (self.kernel_h, self.kernel_w);

        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;

        if padded_h < kh || padded_w < kw {
            return Err(NeuralError::InvalidShape(
                "DepthwiseConv2d::forward: padded size smaller than kernel".to_string(),
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

        let mut out_data = vec![0.0_f32; self.channels * out_h * out_w];
        for c in 0..self.channels {
            let bias_val = self.bias.data()[c];
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias_val;
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ph = oh * stride_h + kh_i;
                            let pw = ow * stride_w + kw_i;
                            let in_val = padded[c * padded_h * padded_w + ph * padded_w + pw];
                            let w_idx = c * (kh * kw) + kh_i * kw + kw_i;
                            acc += in_val * self.weight.data()[w_idx];
                        }
                    }
                    out_data[c * out_h * out_w + oh * out_w + ow] = acc;
                }
            }
        }
        Tensor::from_data(out_data, vec![self.channels, out_h, out_w])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MaxPool2d
// ──────────────────────────────────────────────────────────────────────────────

/// 2-D max-pooling layer.
///
/// Input shape: `[C, H, W]`.
/// Output shape: `[C, out_H, out_W]`.
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    /// (kernel_h, kernel_w).
    pub kernel: (usize, usize),
    /// (stride_h, stride_w).
    pub stride: (usize, usize),
}

impl MaxPool2d {
    /// Creates a `MaxPool2d` layer.
    pub fn new(kernel: (usize, usize), stride: (usize, usize)) -> Result<Self, NeuralError> {
        if kernel.0 == 0 || kernel.1 == 0 || stride.0 == 0 || stride.1 == 0 {
            return Err(NeuralError::InvalidShape(
                "MaxPool2d: kernel and stride sizes must be > 0".to_string(),
            ));
        }
        Ok(Self { kernel, stride })
    }

    /// Forward pass.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "MaxPool2d::forward: expected [C,H,W] input, got rank {}",
                input.ndim()
            )));
        }
        let (c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let (kh, kw) = self.kernel;
        let (sh, sw) = self.stride;

        if in_h < kh || in_w < kw {
            return Err(NeuralError::InvalidShape(format!(
                "MaxPool2d::forward: input ({},{}) smaller than kernel ({},{})",
                in_h, in_w, kh, kw
            )));
        }

        let out_h = (in_h - kh) / sh + 1;
        let out_w = (in_w - kw) / sw + 1;
        let mut out_data = vec![f32::NEG_INFINITY; c * out_h * out_w];

        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ih = oh * sh + kh_i;
                            let iw = ow * sw + kw_i;
                            let val = input.data()[ch * in_h * in_w + ih * in_w + iw];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                    out_data[ch * out_h * out_w + oh * out_w + ow] = max_val;
                }
            }
        }
        Tensor::from_data(out_data, vec![c, out_h, out_w])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GlobalAvgPool
// ──────────────────────────────────────────────────────────────────────────────

/// Global average pooling: reduces the spatial dimensions `H` and `W` to `1×1`
/// by averaging all values in each channel.
///
/// Input shape: `[C, H, W]`.
/// Output shape: `[C, 1, 1]`.
#[derive(Debug, Clone, Default)]
pub struct GlobalAvgPool;

impl GlobalAvgPool {
    /// Creates a `GlobalAvgPool` layer.
    pub fn new() -> Self {
        Self
    }

    /// Forward pass.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "GlobalAvgPool::forward: expected [C,H,W] input, got rank {}",
                input.ndim()
            )));
        }
        let (c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let spatial = h * w;
        if spatial == 0 {
            return Err(NeuralError::EmptyInput(
                "GlobalAvgPool::forward: spatial size is zero".to_string(),
            ));
        }
        let mut out_data = vec![0.0_f32; c];
        for ch in 0..c {
            let mut sum = 0.0_f32;
            for idx in 0..spatial {
                sum += input.data()[ch * spatial + idx];
            }
            out_data[ch] = sum / spatial as f32;
        }
        Tensor::from_data(out_data, vec![c, 1, 1])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AvgPool2d
// ──────────────────────────────────────────────────────────────────────────────

/// 2-D average pooling layer.
///
/// Input shape: `[C, H, W]` or `[N, C, H, W]` (batch mode).
/// Output shape: `[C, out_H, out_W]` or `[N, C, out_H, out_W]`.
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    /// (kernel_h, kernel_w).
    pub kernel: (usize, usize),
    /// (stride_h, stride_w).
    pub stride: (usize, usize),
}

impl AvgPool2d {
    /// Creates an `AvgPool2d` layer.
    pub fn new(kernel: (usize, usize), stride: (usize, usize)) -> Result<Self, NeuralError> {
        if kernel.0 == 0 || kernel.1 == 0 || stride.0 == 0 || stride.1 == 0 {
            return Err(NeuralError::InvalidShape(
                "AvgPool2d: kernel and stride sizes must be > 0".to_string(),
            ));
        }
        Ok(Self { kernel, stride })
    }

    /// Forward pass.
    ///
    /// Accepts both `[C, H, W]` and `[N, C, H, W]` inputs.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        match input.ndim() {
            3 => self.forward_single(input),
            4 => self.forward_batch(input),
            other => Err(NeuralError::InvalidShape(format!(
                "AvgPool2d::forward: expected rank 3 or 4, got {}",
                other
            ))),
        }
    }

    /// Forward for a single `[C, H, W]` tensor.
    fn forward_single(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        let (c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let (kh, kw) = self.kernel;
        let (sh, sw) = self.stride;

        if in_h < kh || in_w < kw {
            return Err(NeuralError::InvalidShape(format!(
                "AvgPool2d::forward: input ({},{}) smaller than kernel ({},{})",
                in_h, in_w, kh, kw
            )));
        }

        let out_h = (in_h - kh) / sh + 1;
        let out_w = (in_w - kw) / sw + 1;
        let pool_size = (kh * kw) as f32;
        let mut out_data = vec![0.0_f32; c * out_h * out_w];

        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0_f32;
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ih = oh * sh + kh_i;
                            let iw = ow * sw + kw_i;
                            sum += input.data()[ch * in_h * in_w + ih * in_w + iw];
                        }
                    }
                    out_data[ch * out_h * out_w + oh * out_w + ow] = sum / pool_size;
                }
            }
        }
        Tensor::from_data(out_data, vec![c, out_h, out_w])
    }

    /// Forward for a batch `[N, C, H, W]` tensor.
    fn forward_batch(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        let (n, c, in_h, in_w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (kh, kw) = self.kernel;
        let (sh, sw) = self.stride;

        if in_h < kh || in_w < kw {
            return Err(NeuralError::InvalidShape(format!(
                "AvgPool2d::forward: input ({},{}) smaller than kernel ({},{})",
                in_h, in_w, kh, kw
            )));
        }

        let out_h = (in_h - kh) / sh + 1;
        let out_w = (in_w - kw) / sw + 1;
        let pool_size = (kh * kw) as f32;
        let in_spatial = in_h * in_w;
        let out_spatial = out_h * out_w;
        let mut out_data = vec![0.0_f32; n * c * out_spatial];

        for bi in 0..n {
            for ch in 0..c {
                let in_base = bi * c * in_spatial + ch * in_spatial;
                let out_base = bi * c * out_spatial + ch * out_spatial;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0_f32;
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let ih = oh * sh + kh_i;
                                let iw = ow * sw + kw_i;
                                sum += input.data()[in_base + ih * in_w + iw];
                            }
                        }
                        out_data[out_base + oh * out_w + ow] = sum / pool_size;
                    }
                }
            }
        }
        Tensor::from_data(out_data, vec![n, c, out_h, out_w])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ConvTranspose2d
// ──────────────────────────────────────────────────────────────────────────────

/// 2-D transposed convolution (fractionally-strided convolution) layer.
///
/// Weight tensor has shape `[in_channels, out_channels, kH, kW]`.
/// Note: unlike regular conv2d, the weight indexing is `[in, out, kH, kW]`.
/// Bias tensor has shape `[out_channels]`.
///
/// Input shape: `[C_in, H, W]` or `[N, C_in, H, W]` (batch mode).
/// Output shape: `[C_out, out_H, out_W]` or `[N, C_out, out_H, out_W]` where:
/// - `out_H = (H - 1) * stride_h - 2 * pad_h + kH + output_pad_h`
/// - `out_W = (W - 1) * stride_w - 2 * pad_w + kW + output_pad_w`
#[derive(Debug, Clone)]
pub struct ConvTranspose2d {
    /// Convolution kernels, shape `[in_channels, out_channels, kH, kW]`.
    pub weight: Tensor,
    /// Per-output-channel bias, shape `[out_channels]`.
    pub bias: Tensor,
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
    /// (pad_h, pad_w) — zero-padding removed from each side of the output.
    pub padding: (usize, usize),
    /// (output_pad_h, output_pad_w) — extra size added to output.
    pub output_padding: (usize, usize),
}

impl ConvTranspose2d {
    /// Creates a zero-initialized `ConvTranspose2d`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    ) -> Result<Self, NeuralError> {
        if in_channels == 0 || out_channels == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NeuralError::InvalidShape(
                "ConvTranspose2d: all sizes must be > 0".to_string(),
            ));
        }
        if output_padding.0 >= stride.0 || output_padding.1 >= stride.1 {
            return Err(NeuralError::InvalidShape(
                "ConvTranspose2d: output_padding must be < stride".to_string(),
            ));
        }
        let weight = Tensor::zeros(vec![in_channels, out_channels, kernel_h, kernel_w])?;
        let bias = Tensor::zeros(vec![out_channels])?;
        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            output_padding,
        })
    }

    /// Forward pass.
    ///
    /// Accepts both `[C, H, W]` and `[N, C, H, W]` inputs.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        match input.ndim() {
            3 => self.forward_single(input),
            4 => self.forward_batch(input),
            other => Err(NeuralError::InvalidShape(format!(
                "ConvTranspose2d::forward: expected rank 3 or 4, got {}",
                other
            ))),
        }
    }

    /// Forward for a single `[C_in, H, W]` tensor.
    fn forward_single(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if in_c != self.in_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "ConvTranspose2d::forward: input channels {} != in_channels {}",
                in_c, self.in_channels
            )));
        }

        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (oph, opw) = self.output_padding;
        let (kh, kw) = (self.kernel_h, self.kernel_w);

        let out_h = (in_h - 1) * sh + kh + oph - 2 * ph;
        let out_w = (in_w - 1) * sw + kw + opw - 2 * pw;

        let mut out_data = vec![0.0_f32; self.out_channels * out_h * out_w];

        // Add bias.
        for oc in 0..self.out_channels {
            let bias_val = self.bias.data()[oc];
            for oh in 0..out_h {
                for ow in 0..out_w {
                    out_data[oc * out_h * out_w + oh * out_w + ow] = bias_val;
                }
            }
        }

        // Scatter each input element across the output.
        for ic in 0..in_c {
            for ih in 0..in_h {
                for iw in 0..in_w {
                    let in_val = input.data()[ic * in_h * in_w + ih * in_w + iw];
                    for oc in 0..self.out_channels {
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let oh_raw =
                                    ih as isize * sh as isize + kh_i as isize - ph as isize;
                                let ow_raw =
                                    iw as isize * sw as isize + kw_i as isize - pw as isize;
                                if oh_raw >= 0
                                    && (oh_raw as usize) < out_h
                                    && ow_raw >= 0
                                    && (ow_raw as usize) < out_w
                                {
                                    let oh = oh_raw as usize;
                                    let ow = ow_raw as usize;
                                    let w_idx = ic * (self.out_channels * kh * kw)
                                        + oc * (kh * kw)
                                        + kh_i * kw
                                        + kw_i;
                                    out_data[oc * out_h * out_w + oh * out_w + ow] +=
                                        in_val * self.weight.data()[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_data(out_data, vec![self.out_channels, out_h, out_w])
    }

    /// Forward for a batch `[N, C_in, H, W]` tensor.
    fn forward_batch(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        let n = input.shape()[0];
        let in_c = input.shape()[1];
        let in_h = input.shape()[2];
        let in_w = input.shape()[3];

        if in_c != self.in_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "ConvTranspose2d::forward: input channels {} != in_channels {}",
                in_c, self.in_channels
            )));
        }

        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (oph, opw) = self.output_padding;
        let (kh, kw) = (self.kernel_h, self.kernel_w);

        let out_h = (in_h - 1) * sh + kh + oph - 2 * ph;
        let out_w = (in_w - 1) * sw + kw + opw - 2 * pw;
        let in_spatial = in_h * in_w;
        let out_spatial = out_h * out_w;
        let mut out_data = vec![0.0_f32; n * self.out_channels * out_spatial];

        for bi in 0..n {
            // Add bias.
            for oc in 0..self.out_channels {
                let bias_val = self.bias.data()[oc];
                let out_base = bi * self.out_channels * out_spatial + oc * out_spatial;
                for idx in 0..out_spatial {
                    out_data[out_base + idx] = bias_val;
                }
            }

            // Scatter.
            for ic in 0..in_c {
                let in_base = bi * in_c * in_spatial + ic * in_spatial;
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let in_val = input.data()[in_base + ih * in_w + iw];
                        for oc in 0..self.out_channels {
                            let out_base = bi * self.out_channels * out_spatial + oc * out_spatial;
                            for kh_i in 0..kh {
                                for kw_i in 0..kw {
                                    let oh_raw =
                                        ih as isize * sh as isize + kh_i as isize - ph as isize;
                                    let ow_raw =
                                        iw as isize * sw as isize + kw_i as isize - pw as isize;
                                    if oh_raw >= 0
                                        && (oh_raw as usize) < out_h
                                        && ow_raw >= 0
                                        && (ow_raw as usize) < out_w
                                    {
                                        let oh = oh_raw as usize;
                                        let ow = ow_raw as usize;
                                        let w_idx = ic * (self.out_channels * kh * kw)
                                            + oc * (kh * kw)
                                            + kh_i * kw
                                            + kw_i;
                                        out_data[out_base + oh * out_w + ow] +=
                                            in_val * self.weight.data()[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_data(out_data, vec![n, self.out_channels, out_h, out_w])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Batch-aware Conv2d forward
// ──────────────────────────────────────────────────────────────────────────────

impl Conv2dLayer {
    /// Forward pass supporting both `[C, H, W]` and `[N, C, H, W]` inputs.
    ///
    /// For 3-D input, delegates to the existing single-sample forward.
    /// For 4-D input, applies the convolution independently to each batch sample.
    pub fn forward_batch(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        match input.ndim() {
            3 => self.forward(input),
            4 => {
                let n = input.shape()[0];
                let (in_c, in_h, in_w) = (input.shape()[1], input.shape()[2], input.shape()[3]);
                let in_spatial = in_c * in_h * in_w;

                // Compute output for first sample to determine output shape.
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
                "Conv2dLayer::forward_batch: expected rank 3 or 4, got {}",
                other
            ))),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Batch-aware MaxPool2d forward
// ──────────────────────────────────────────────────────────────────────────────

impl MaxPool2d {
    /// Forward pass supporting both `[C, H, W]` and `[N, C, H, W]` inputs.
    pub fn forward_batch(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        match input.ndim() {
            3 => self.forward(input),
            4 => {
                let n = input.shape()[0];
                let (c, in_h, in_w) = (input.shape()[1], input.shape()[2], input.shape()[3]);
                let (kh, kw) = self.kernel;
                let (sh, sw) = self.stride;

                if in_h < kh || in_w < kw {
                    return Err(NeuralError::InvalidShape(format!(
                        "MaxPool2d::forward: input ({},{}) smaller than kernel ({},{})",
                        in_h, in_w, kh, kw
                    )));
                }

                let out_h = (in_h - kh) / sh + 1;
                let out_w = (in_w - kw) / sw + 1;
                let in_spatial = c * in_h * in_w;
                let out_spatial = c * out_h * out_w;
                let mut out_data = vec![f32::NEG_INFINITY; n * out_spatial];

                for bi in 0..n {
                    for ch in 0..c {
                        let in_base = bi * in_spatial + ch * in_h * in_w;
                        let out_base = bi * out_spatial + ch * out_h * out_w;
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let mut max_val = f32::NEG_INFINITY;
                                for kh_i in 0..kh {
                                    for kw_i in 0..kw {
                                        let ih = oh * sh + kh_i;
                                        let iw = ow * sw + kw_i;
                                        let val = input.data()[in_base + ih * in_w + iw];
                                        if val > max_val {
                                            max_val = val;
                                        }
                                    }
                                }
                                out_data[out_base + oh * out_w + ow] = max_val;
                            }
                        }
                    }
                }
                Tensor::from_data(out_data, vec![n, c, out_h, out_w])
            }
            other => Err(NeuralError::InvalidShape(format!(
                "MaxPool2d::forward_batch: expected rank 3 or 4, got {}",
                other
            ))),
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

    // ── LinearLayer ───────────────────────────────────────────────────────────

    #[test]
    fn test_linear_zero_init_output() {
        let layer = LinearLayer::new(4, 3).expect("linear layer new");
        let input = Tensor::ones(vec![4]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[3]);
        assert!(out.data().iter().all(|&x| close(x, 0.0)));
    }

    #[test]
    fn test_linear_identity_weight() {
        // 2×2 identity matrix as weight, zero bias → output equals input.
        let mut layer = LinearLayer::new(2, 2).expect("linear layer new");
        layer.weight.data_mut()[0] = 1.0;
        layer.weight.data_mut()[3] = 1.0;
        let input = Tensor::from_data(vec![3.0, 5.0], vec![2]).expect("tensor from_data");
        let out = layer.forward(&input).expect("forward pass");
        assert!(close(out.data()[0], 3.0));
        assert!(close(out.data()[1], 5.0));
    }

    #[test]
    fn test_linear_with_bias() {
        let mut layer = LinearLayer::new(2, 1).expect("linear layer new");
        // weight = [[1, 1]], bias = [0.5]
        layer.weight.data_mut()[0] = 1.0;
        layer.weight.data_mut()[1] = 1.0;
        layer.bias.data_mut()[0] = 0.5;
        let input = Tensor::from_data(vec![2.0, 3.0], vec![2]).expect("tensor from_data");
        let out = layer.forward(&input).expect("forward pass");
        assert!(close(out.data()[0], 5.5)); // 2+3+0.5
    }

    #[test]
    fn test_linear_wrong_input_size() {
        let layer = LinearLayer::new(4, 3).expect("linear layer new");
        let input = Tensor::ones(vec![5]).expect("tensor ones");
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_linear_zero_features_error() {
        assert!(LinearLayer::new(0, 3).is_err());
        assert!(LinearLayer::new(3, 0).is_err());
    }

    // ── LinearLayer::forward_batch ────────────────────────────────────────────

    #[test]
    fn test_linear_forward_batch_shape() {
        let layer = LinearLayer::new(4, 3).expect("linear layer new");
        // batch of 5 samples, each with 4 features
        let input = Tensor::ones(vec![5, 4]).expect("tensor ones");
        let out = layer.forward_batch(&input).expect("forward_batch");
        assert_eq!(out.shape(), &[5, 3]);
    }

    #[test]
    fn test_linear_forward_batch_matches_single() {
        // forward_batch on [B, in] must produce the same rows as B calls to forward.
        let mut layer = LinearLayer::new(3, 2).expect("linear layer new");
        // Set a non-trivial weight and bias.
        let w_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.5 - 0.75).collect();
        let b_data = vec![0.1_f32, -0.1];
        layer.weight = Tensor::from_data(w_data, vec![2, 3]).expect("tensor from_data");
        layer.bias = Tensor::from_data(b_data, vec![2]).expect("tensor from_data");

        let batch_data: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect();
        let batch = Tensor::from_data(batch_data.clone(), vec![3, 3]).expect("tensor from_data");

        let batched_out = layer.forward_batch(&batch).expect("forward_batch");

        for b in 0..3usize {
            let sample_data = batch_data[b * 3..(b + 1) * 3].to_vec();
            let sample = Tensor::from_data(sample_data, vec![3]).expect("tensor from_data");
            let single_out = layer.forward(&sample).expect("forward pass");
            for f in 0..2usize {
                let diff = (batched_out.data()[b * 2 + f] - single_out.data()[f]).abs();
                assert!(diff < 1e-5, "batch b={b} feature f={f}: diff={diff}");
            }
        }
    }

    #[test]
    fn test_linear_forward_batch_zero_weights_all_bias() {
        // Zero weights: every output row should equal the bias.
        let mut layer = LinearLayer::new(4, 2).expect("linear layer new");
        layer.bias = Tensor::from_data(vec![3.0_f32, -1.5], vec![2]).expect("tensor from_data");
        let input = Tensor::ones(vec![6, 4]).expect("tensor ones");
        let out = layer.forward_batch(&input).expect("forward_batch");
        assert_eq!(out.shape(), &[6, 2]);
        for b in 0..6usize {
            assert!((out.data()[b * 2] - 3.0).abs() < 1e-5);
            assert!((out.data()[b * 2 + 1] + 1.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_linear_forward_batch_wrong_rank_error() {
        let layer = LinearLayer::new(4, 3).expect("linear layer new");
        let input = Tensor::ones(vec![4]).expect("tensor ones"); // 1-D, should be 2-D
        assert!(layer.forward_batch(&input).is_err());
    }

    #[test]
    fn test_linear_forward_batch_feature_mismatch_error() {
        let layer = LinearLayer::new(4, 3).expect("linear layer new");
        let input = Tensor::ones(vec![2, 5]).expect("tensor ones"); // 5 != 4
        assert!(layer.forward_batch(&input).is_err());
    }

    // ── Conv2dLayer ───────────────────────────────────────────────────────────

    #[test]
    fn test_conv2d_zero_weight_output_is_bias() {
        let mut layer = Conv2dLayer::new(1, 1, 3, 3, (1, 1), (1, 1)).expect("conv2d layer new");
        layer.bias.data_mut()[0] = 2.0;
        // 1-channel 5×5 input of all ones.
        let input = Tensor::ones(vec![1, 5, 5]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        // Output should be 5×5 (stride=1, pad=1 preserves size with 3×3 kernel).
        assert_eq!(out.shape(), &[1, 5, 5]);
        // All values equal the bias because weight is zero.
        assert!(out.data().iter().all(|&v| close(v, 2.0)));
    }

    #[test]
    fn test_conv2d_output_size_no_pad() {
        // 1-channel 7×7, 3×3 kernel, stride 1, no pad → (7-3)/1+1 = 5
        let layer = Conv2dLayer::new(1, 2, 3, 3, (1, 1), (0, 0)).expect("conv2d layer new");
        let input = Tensor::ones(vec![1, 7, 7]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[2, 5, 5]);
    }

    #[test]
    fn test_conv2d_stride2() {
        // 1-channel 6×6, 2×2 kernel, stride 2, no pad → (6-2)/2+1 = 3
        let layer = Conv2dLayer::new(1, 1, 2, 2, (2, 2), (0, 0)).expect("conv2d layer new");
        let input = Tensor::ones(vec![1, 6, 6]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[1, 3, 3]);
    }

    #[test]
    fn test_conv2d_channel_mismatch_error() {
        let layer = Conv2dLayer::new(3, 1, 3, 3, (1, 1), (0, 0)).expect("conv2d layer new");
        let input = Tensor::ones(vec![1, 5, 5]).expect("tensor ones");
        assert!(layer.forward(&input).is_err());
    }

    // ── BatchNorm1d ───────────────────────────────────────────────────────────

    #[test]
    fn test_batchnorm1d_identity() {
        // Default: mean=0, var=1, w=1, b=0 → output equals input.
        let bn = BatchNorm1d::new(4).expect("batch norm new");
        let input =
            Tensor::from_data(vec![1.0, -1.0, 2.0, -2.0], vec![4]).expect("tensor from_data");
        let out = bn.forward(&input).expect("forward pass");
        assert!(close(out.data()[0], 1.0));
        assert!(close(out.data()[1], -1.0));
    }

    #[test]
    fn test_batchnorm1d_scale_shift() {
        let mut bn = BatchNorm1d::new(2).expect("batch norm new");
        bn.weight = vec![2.0, 3.0];
        bn.bias = vec![1.0, -1.0];
        // running_mean = 0, running_var = 1, eps = 1e-5
        // output[0] = 2*(x-0)/1 + 1, output[1] = 3*(x-0)/1 - 1
        let input = Tensor::from_data(vec![1.0, 2.0], vec![2]).expect("tensor from_data");
        let out = bn.forward(&input).expect("forward pass");
        assert!(close(out.data()[0], 3.0)); // 2*1 + 1
        assert!(close(out.data()[1], 5.0)); // 3*2 - 1
    }

    #[test]
    fn test_batchnorm1d_wrong_size() {
        let bn = BatchNorm1d::new(4).expect("batch norm new");
        let input = Tensor::ones(vec![5]).expect("tensor ones");
        assert!(bn.forward(&input).is_err());
    }

    // ── DepthwiseConv2d ───────────────────────────────────────────────────────

    #[test]
    fn test_depthwise_output_size() {
        // 3-channel 8×8, 3×3 kernel, stride 1, pad 1 → 8×8 output
        let layer = DepthwiseConv2d::new(3, 3, 3, (1, 1), (1, 1)).expect("depthwise conv2d new");
        let input = Tensor::ones(vec![3, 8, 8]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[3, 8, 8]);
    }

    #[test]
    fn test_depthwise_channel_isolation() {
        // Only channel 0 has non-zero weight; channels 1 and 2 should produce zero.
        let mut layer =
            DepthwiseConv2d::new(3, 1, 1, (1, 1), (0, 0)).expect("depthwise conv2d new");
        // Weight layout: [channels, 1, kh, kw] = [3, 1, 1, 1]
        layer.weight.data_mut()[0] = 1.0; // channel 0 kernel = 1
                                          // channels 1 and 2 remain 0
        let input = Tensor::ones(vec![3, 4, 4]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        // channel 0: all 1.0, channels 1-2: all 0.0
        for i in 0..16 {
            assert!(close(out.data()[i], 1.0));
        }
        for i in 16..48 {
            assert!(close(out.data()[i], 0.0));
        }
    }

    // ── MaxPool2d ─────────────────────────────────────────────────────────────

    #[test]
    fn test_maxpool_selects_max() {
        // 1-channel 2×2, kernel 2×2, stride 2 → 1×1 output = max of all values.
        let layer = MaxPool2d::new((2, 2), (2, 2)).expect("maxpool2d new");
        let input =
            Tensor::from_data(vec![1.0, 3.0, 2.0, 4.0], vec![1, 2, 2]).expect("tensor from_data");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[1, 1, 1]);
        assert!(close(out.data()[0], 4.0));
    }

    #[test]
    fn test_maxpool_output_size() {
        // 2-channel 4×4, kernel 2×2, stride 2 → 2-channel 2×2
        let layer = MaxPool2d::new((2, 2), (2, 2)).expect("maxpool2d new");
        let input = Tensor::ones(vec![2, 4, 4]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_maxpool_zero_stride_error() {
        assert!(MaxPool2d::new((2, 2), (0, 1)).is_err());
    }

    // ── GlobalAvgPool ─────────────────────────────────────────────────────────

    #[test]
    fn test_global_avg_pool_shape() {
        let layer = GlobalAvgPool::new();
        let input = Tensor::ones(vec![4, 8, 8]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[4, 1, 1]);
    }

    #[test]
    fn test_global_avg_pool_value() {
        let layer = GlobalAvgPool::new();
        // 2-channel 2×2 input: channel 0 all 2.0, channel 1 all 6.0
        let data = vec![2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0];
        let input = Tensor::from_data(data, vec![2, 2, 2]).expect("tensor from_data");
        let out = layer.forward(&input).expect("forward pass");
        assert!(close(out.data()[0], 2.0));
        assert!(close(out.data()[1], 6.0));
    }

    #[test]
    fn test_global_avg_pool_wrong_rank() {
        let layer = GlobalAvgPool::new();
        let input = Tensor::ones(vec![4, 8]).expect("tensor ones");
        assert!(layer.forward(&input).is_err());
    }

    // ── AvgPool2d ────────────────────────────────────────────────────────────

    #[test]
    fn test_avgpool_value() {
        let layer = AvgPool2d::new((2, 2), (2, 2)).expect("avgpool2d new");
        // 1-channel 2×2: [1,3,2,4] → avg = 2.5
        let input =
            Tensor::from_data(vec![1.0, 3.0, 2.0, 4.0], vec![1, 2, 2]).expect("tensor from_data");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[1, 1, 1]);
        assert!(close(out.data()[0], 2.5));
    }

    #[test]
    fn test_avgpool_output_size() {
        let layer = AvgPool2d::new((2, 2), (2, 2)).expect("avgpool2d new");
        let input = Tensor::ones(vec![3, 4, 4]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[3, 2, 2]);
    }

    #[test]
    fn test_avgpool_batch() {
        let layer = AvgPool2d::new((2, 2), (2, 2)).expect("avgpool2d new");
        let input = Tensor::ones(vec![2, 3, 4, 4]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[2, 3, 2, 2]);
        // All ones → average is 1.0
        assert!(out.data().iter().all(|&v| close(v, 1.0)));
    }

    #[test]
    fn test_avgpool_zero_stride_error() {
        assert!(AvgPool2d::new((2, 2), (0, 1)).is_err());
    }

    // ── ConvTranspose2d ──────────────────────────────────────────────────────

    #[test]
    fn test_conv_transpose_output_size() {
        // Input: 1ch 2×2, kernel 3×3, stride 2, pad 0, output_pad 0
        // out_h = (2-1)*2 + 3 + 0 - 0 = 5
        let layer = ConvTranspose2d::new(1, 1, 3, 3, (2, 2), (0, 0), (0, 0))
            .expect("conv transpose 2d new");
        let input = Tensor::ones(vec![1, 2, 2]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[1, 5, 5]);
    }

    #[test]
    fn test_conv_transpose_with_padding() {
        // Input: 1ch 3×3, kernel 3×3, stride 1, pad 1, output_pad 0
        // out_h = (3-1)*1 + 3 + 0 - 2 = 4
        let layer = ConvTranspose2d::new(1, 1, 3, 3, (1, 1), (1, 1), (0, 0))
            .expect("conv transpose 2d new");
        let input = Tensor::ones(vec![1, 3, 3]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[1, 3, 3]); // preserves size with pad=1
    }

    #[test]
    fn test_conv_transpose_bias_only() {
        // Zero weights → output is just the bias replicated.
        let mut layer = ConvTranspose2d::new(1, 2, 2, 2, (1, 1), (0, 0), (0, 0))
            .expect("conv transpose 2d new");
        layer.bias.data_mut()[0] = 5.0;
        layer.bias.data_mut()[1] = 7.0;
        let input = Tensor::ones(vec![1, 3, 3]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        // out_h = (3-1)*1 + 2 = 4
        assert_eq!(out.shape(), &[2, 4, 4]);
        // channel 0 all 5.0, channel 1 all 7.0
        let out_spatial = 16;
        assert!(out.data()[..out_spatial].iter().all(|&v| close(v, 5.0)));
        assert!(out.data()[out_spatial..].iter().all(|&v| close(v, 7.0)));
    }

    #[test]
    fn test_conv_transpose_batch() {
        let layer = ConvTranspose2d::new(1, 1, 2, 2, (2, 2), (0, 0), (0, 0))
            .expect("conv transpose 2d new");
        let input = Tensor::ones(vec![2, 1, 3, 3]).expect("tensor ones");
        let out = layer.forward(&input).expect("forward pass");
        // out_h = (3-1)*2 + 2 = 6
        assert_eq!(out.shape(), &[2, 1, 6, 6]);
    }

    #[test]
    fn test_conv_transpose_output_padding_error() {
        // output_padding must be < stride
        assert!(ConvTranspose2d::new(1, 1, 3, 3, (2, 2), (0, 0), (2, 0)).is_err());
    }

    #[test]
    fn test_conv_transpose_channel_mismatch() {
        let layer = ConvTranspose2d::new(3, 1, 3, 3, (1, 1), (0, 0), (0, 0))
            .expect("conv transpose 2d new");
        let input = Tensor::ones(vec![1, 5, 5]).expect("tensor ones");
        assert!(layer.forward(&input).is_err());
    }

    // ── Batch-aware Conv2d ───────────────────────────────────────────────────

    #[test]
    fn test_conv2d_batch_forward() {
        let mut layer = Conv2dLayer::new(1, 1, 3, 3, (1, 1), (1, 1)).expect("conv2d layer new");
        layer.bias.data_mut()[0] = 1.0;
        // Batch of 2, 1-channel 4×4
        let input = Tensor::ones(vec![2, 1, 4, 4]).expect("tensor ones");
        let out = layer.forward_batch(&input).expect("forward_batch");
        assert_eq!(out.shape(), &[2, 1, 4, 4]);
        // Zero weight + bias=1 → all 1.0
        assert!(out.data().iter().all(|&v| close(v, 1.0)));
    }

    // ── Batch-aware MaxPool2d ────────────────────────────────────────────────

    #[test]
    fn test_maxpool_batch_forward() {
        let layer = MaxPool2d::new((2, 2), (2, 2)).expect("maxpool2d new");
        let input = Tensor::ones(vec![2, 1, 4, 4]).expect("tensor ones");
        let out = layer.forward_batch(&input).expect("forward_batch");
        assert_eq!(out.shape(), &[2, 1, 2, 2]);
        assert!(out.data().iter().all(|&v| close(v, 1.0)));
    }

    #[test]
    fn test_maxpool_batch_selects_max() {
        let layer = MaxPool2d::new((2, 2), (2, 2)).expect("maxpool2d new");
        // 2 samples, 1 channel, 2×2
        let data = vec![
            1.0, 3.0, 2.0, 4.0, // sample 0
            5.0, 7.0, 6.0, 8.0, // sample 1
        ];
        let input = Tensor::from_data(data, vec![2, 1, 2, 2]).expect("tensor from_data");
        let out = layer.forward_batch(&input).expect("forward_batch");
        assert_eq!(out.shape(), &[2, 1, 1, 1]);
        assert!(close(out.data()[0], 4.0));
        assert!(close(out.data()[1], 8.0));
    }

    // ── Tensor::from_batch / unbatch / batch_size ─────────────────────────────

    #[test]
    fn test_from_batch_stacks_along_dim0() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor from_data");
        let b = Tensor::from_data(vec![4.0, 5.0, 6.0], vec![3]).expect("tensor from_data");
        let batched = Tensor::from_batch(vec![a, b]).expect("from_batch");
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(batched.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_from_batch_single_item() {
        let t = Tensor::ones(vec![2, 3]).expect("tensor ones");
        let batched = Tensor::from_batch(vec![t]).expect("from_batch");
        assert_eq!(batched.shape(), &[1, 2, 3]);
    }

    #[test]
    fn test_from_batch_empty_error() {
        let result = Tensor::from_batch(vec![]);
        assert!(result.is_err(), "empty items should error");
    }

    #[test]
    fn test_from_batch_shape_mismatch_error() {
        let a = Tensor::ones(vec![3]).expect("tensor ones");
        let b = Tensor::ones(vec![4]).expect("tensor ones");
        let result = Tensor::from_batch(vec![a, b]);
        assert!(result.is_err(), "mismatched shapes should error");
    }

    #[test]
    fn test_from_batch_3d_items() {
        let a = Tensor::ones(vec![2, 3, 4]).expect("tensor ones");
        let b = Tensor::zeros(vec![2, 3, 4]).expect("tensor zeros");
        let batched = Tensor::from_batch(vec![a, b]).expect("from_batch");
        assert_eq!(batched.shape(), &[2, 2, 3, 4]);
    }

    #[test]
    fn test_batch_size_returns_some_for_4d() {
        let t = Tensor::ones(vec![5, 3, 8, 8]).expect("tensor ones");
        assert_eq!(t.batch_size(), Some(5));
    }

    #[test]
    fn test_batch_size_returns_none_for_3d() {
        let t = Tensor::ones(vec![3, 8, 8]).expect("tensor ones");
        assert_eq!(t.batch_size(), None);
    }

    #[test]
    fn test_batch_size_returns_none_for_1d() {
        let t = Tensor::ones(vec![10]).expect("tensor ones");
        assert_eq!(t.batch_size(), None);
    }

    #[test]
    fn test_batch_size_returns_some_for_5d() {
        let t = Tensor::ones(vec![2, 3, 4, 5, 6]).expect("tensor ones");
        assert_eq!(t.batch_size(), Some(2));
    }

    #[test]
    fn test_unbatch_splits_correctly() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let batched = Tensor::from_data(data, vec![2, 3]).expect("tensor from_data");
        let items = batched.unbatch().expect("unbatch");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].shape(), &[3]);
        assert_eq!(items[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(items[1].data(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unbatch_1d_error() {
        let t = Tensor::ones(vec![5]).expect("tensor ones");
        assert!(t.unbatch().is_err(), "1-D tensor cannot be unbatched");
    }

    #[test]
    fn test_from_batch_unbatch_roundtrip() {
        let a = Tensor::from_data(vec![1.0, 2.0], vec![2]).expect("tensor from_data");
        let b = Tensor::from_data(vec![3.0, 4.0], vec![2]).expect("tensor from_data");
        let c = Tensor::from_data(vec![5.0, 6.0], vec![2]).expect("tensor from_data");
        let batched =
            Tensor::from_batch(vec![a.clone(), b.clone(), c.clone()]).expect("from_batch");
        let items = batched.unbatch().expect("unbatch");
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].data(), a.data());
        assert_eq!(items[1].data(), b.data());
        assert_eq!(items[2].data(), c.data());
    }

    // ── LinearLayer::forward_batch_tensor ────────────────────────────────────

    #[test]
    fn test_forward_batch_tensor_2d_delegates_to_forward_batch() {
        let mut layer = LinearLayer::new(3, 2).expect("linear layer new");
        // Set weight to identity-like (3 in, 2 out: take first 2 inputs)
        layer.weight.data_mut()[0] = 1.0; // out0 += in0
        layer.weight.data_mut()[4] = 1.0; // out1 += in1
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .expect("tensor from_data");
        let out_batch = layer.forward_batch(&input).expect("forward_batch");
        let out_batch_tensor = layer
            .forward_batch_tensor(&input)
            .expect("forward_batch_tensor");
        assert_eq!(out_batch.shape(), out_batch_tensor.shape());
        assert_eq!(out_batch.data(), out_batch_tensor.data());
    }

    #[test]
    fn test_forward_batch_tensor_3d_input() {
        // Input shape [2, 3, 4] → treat as batch of 2*3=6 vectors of dim 4
        let mut layer = LinearLayer::new(4, 2).expect("linear layer new");
        layer.weight.data_mut()[0] = 1.0; // out0 += in0
        layer.weight.data_mut()[5] = 1.0; // out1 += in1
        let input = Tensor::ones(vec![2, 3, 4]).expect("tensor ones");
        let out = layer
            .forward_batch_tensor(&input)
            .expect("forward_batch_tensor");
        // Output should be [2, 3, 2]
        assert_eq!(out.shape(), &[2, 3, 2]);
    }

    #[test]
    fn test_forward_batch_tensor_wrong_rank_error() {
        let layer = LinearLayer::new(4, 2).expect("linear layer new");
        let input = Tensor::ones(vec![4]).expect("tensor ones");
        assert!(layer.forward_batch_tensor(&input).is_err());
    }

    #[test]
    fn test_forward_batch_tensor_feature_mismatch_error() {
        let layer = LinearLayer::new(4, 2).expect("linear layer new");
        let input = Tensor::ones(vec![3, 5]).expect("tensor ones"); // 5 != 4
        assert!(layer.forward_batch_tensor(&input).is_err());
    }

    #[test]
    fn test_conv2d_forward_batch_uses_from_batch() {
        // Test the Conv2dLayer::forward_batch produces 4-D output
        let layer = Conv2dLayer::new(1, 2, 3, 3, (1, 1), (1, 1)).expect("conv2d layer new");
        let input = Tensor::ones(vec![3, 1, 4, 4]).expect("tensor ones");
        let out = layer.forward_batch(&input).expect("forward_batch");
        assert_eq!(out.shape(), &[3, 2, 4, 4]);
    }

    #[test]
    fn test_conv2d_forward_batch_matches_individual() {
        let mut layer = Conv2dLayer::new(1, 1, 3, 3, (1, 1), (1, 1)).expect("conv2d layer new");
        layer.bias.data_mut()[0] = 2.0;
        let s1 = Tensor::ones(vec![1, 4, 4]).expect("tensor ones");
        let s2 = Tensor::from_data(vec![0.5_f32; 16], vec![1, 4, 4]).expect("tensor from_data");
        let batched = Tensor::from_batch(vec![s1.clone(), s2.clone()]).expect("from_batch");
        let batch_out = layer.forward_batch(&batched).expect("forward_batch");
        let out1 = layer.forward(&s1).expect("forward pass");
        let out2 = layer.forward(&s2).expect("forward pass");
        // batch_out[0] should match out1, batch_out[1] should match out2
        let item_numel = out1.numel();
        assert_eq!(&batch_out.data()[..item_numel], out1.data());
        assert_eq!(&batch_out.data()[item_numel..], out2.data());
    }
}

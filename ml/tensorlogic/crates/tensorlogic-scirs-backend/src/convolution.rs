//! Convolution operations for neural network tensor processing.
//!
//! Provides 1D convolution, 2D convolution, transposed convolution, depthwise convolution,
//! and im2col/col2im utilities for efficient convolution via matrix multiplication.

use scirs2_core::ndarray::{ArrayD, IxDyn};

/// Errors that can occur during convolution operations.
#[derive(Debug, Clone)]
pub enum ConvError {
    /// Kernel size contains a zero or is otherwise invalid.
    InvalidKernelSize(String),
    /// Stride contains a zero or is otherwise invalid.
    InvalidStride(String),
    /// Padding value is invalid.
    InvalidPadding(String),
    /// Dilation value is invalid.
    InvalidDilation(String),
    /// Shape mismatch between expected and actual tensors.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Input tensor does not have enough dimensions.
    InsufficientDimensions { ndim: usize, required: usize },
    /// Groups parameter is invalid for the given channel counts.
    InvalidGroups {
        groups: usize,
        in_channels: usize,
        out_channels: usize,
    },
    /// Input tensor is empty.
    EmptyInput,
}

impl std::fmt::Display for ConvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKernelSize(msg) => write!(f, "Invalid kernel size: {msg}"),
            Self::InvalidStride(msg) => write!(f, "Invalid stride: {msg}"),
            Self::InvalidPadding(msg) => write!(f, "Invalid padding: {msg}"),
            Self::InvalidDilation(msg) => write!(f, "Invalid dilation: {msg}"),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {expected:?}, got {got:?}")
            }
            Self::InsufficientDimensions { ndim, required } => {
                write!(
                    f,
                    "Insufficient dimensions: got {ndim}, need at least {required}"
                )
            }
            Self::InvalidGroups {
                groups,
                in_channels,
                out_channels,
            } => write!(
                f,
                "Invalid groups={groups}: in_channels={in_channels} and \
                 out_channels={out_channels} must both be divisible by groups"
            ),
            Self::EmptyInput => write!(f, "Empty input tensor"),
        }
    }
}

impl std::error::Error for ConvError {}

/// Convolution configuration specifying kernel size, stride, padding, dilation, and groups.
#[derive(Debug, Clone)]
pub struct ConvConfig {
    /// Kernel size for each spatial dimension (e.g. `[3, 3]`).
    pub kernel_size: Vec<usize>,
    /// Stride for each spatial dimension (e.g. `[1, 1]`).
    pub stride: Vec<usize>,
    /// Zero-padding on each side for each spatial dimension (e.g. `[1, 1]`).
    pub padding: Vec<usize>,
    /// Dilation factor for each spatial dimension (e.g. `[1, 1]`).
    pub dilation: Vec<usize>,
    /// Number of groups for grouped convolution (1 = standard convolution).
    pub groups: usize,
}

impl ConvConfig {
    /// Create a new convolution config with the given kernel size.
    /// Defaults: stride=1, padding=0, dilation=1, groups=1.
    pub fn new(kernel_size: Vec<usize>) -> Self {
        let ndim = kernel_size.len();
        Self {
            kernel_size,
            stride: vec![1; ndim],
            padding: vec![0; ndim],
            dilation: vec![1; ndim],
            groups: 1,
        }
    }

    /// Set the stride (builder pattern).
    pub fn with_stride(mut self, stride: Vec<usize>) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding (builder pattern).
    pub fn with_padding(mut self, padding: Vec<usize>) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation (builder pattern).
    pub fn with_dilation(mut self, dilation: Vec<usize>) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set the number of groups (builder pattern).
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Compute the output size for one spatial dimension.
    ///
    /// Formula: `(input_size + 2*padding - dilation*(kernel-1) - 1) / stride + 1`
    pub fn output_size(&self, input_size: usize, dim: usize) -> usize {
        let k = self.kernel_size[dim];
        let s = self.stride[dim];
        let p = self.padding[dim];
        let d = self.dilation[dim];
        let effective_k = d * (k - 1) + 1;
        (input_size + 2 * p - effective_k) / s + 1
    }

    /// Validate the configuration, returning an error if any parameter is invalid.
    pub fn validate(&self) -> Result<(), ConvError> {
        let ndim = self.kernel_size.len();

        // All spatial parameter vectors must have the same length
        if self.stride.len() != ndim {
            return Err(ConvError::InvalidStride(format!(
                "stride length {} != kernel_size length {ndim}",
                self.stride.len()
            )));
        }
        if self.padding.len() != ndim {
            return Err(ConvError::InvalidPadding(format!(
                "padding length {} != kernel_size length {ndim}",
                self.padding.len()
            )));
        }
        if self.dilation.len() != ndim {
            return Err(ConvError::InvalidDilation(format!(
                "dilation length {} != kernel_size length {ndim}",
                self.dilation.len()
            )));
        }

        for i in 0..ndim {
            if self.kernel_size[i] == 0 {
                return Err(ConvError::InvalidKernelSize(format!(
                    "kernel_size[{i}] must be > 0"
                )));
            }
            if self.stride[i] == 0 {
                return Err(ConvError::InvalidStride(format!("stride[{i}] must be > 0")));
            }
            if self.dilation[i] == 0 {
                return Err(ConvError::InvalidDilation(format!(
                    "dilation[{i}] must be > 0"
                )));
            }
        }

        if self.groups == 0 {
            return Err(ConvError::InvalidGroups {
                groups: 0,
                in_channels: 0,
                out_channels: 0,
            });
        }

        Ok(())
    }

    /// Number of spatial dimensions (length of kernel_size).
    pub fn num_spatial_dims(&self) -> usize {
        self.kernel_size.len()
    }
}

/// 1D convolution.
///
/// - Input shape: `[batch, in_channels, length]`
/// - Weight shape: `[out_channels, in_channels/groups, kernel_length]`
/// - Output shape: `[batch, out_channels, output_length]`
pub fn conv1d(
    input: &ArrayD<f64>,
    weight: &ArrayD<f64>,
    bias: Option<&ArrayD<f64>>,
    config: &ConvConfig,
) -> Result<ArrayD<f64>, ConvError> {
    config.validate()?;

    let in_shape = input.shape();
    if in_shape.is_empty() || input.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    if in_shape.len() != 3 {
        return Err(ConvError::InsufficientDimensions {
            ndim: in_shape.len(),
            required: 3,
        });
    }

    let w_shape = weight.shape();
    if w_shape.len() != 3 {
        return Err(ConvError::InsufficientDimensions {
            ndim: w_shape.len(),
            required: 3,
        });
    }

    let batch = in_shape[0];
    let in_channels = in_shape[1];
    let in_len = in_shape[2];
    let out_channels = w_shape[0];
    let kernel_len = config.kernel_size[0];
    let groups = config.groups;

    // Validate groups
    if !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
        return Err(ConvError::InvalidGroups {
            groups,
            in_channels,
            out_channels,
        });
    }

    let out_len = config.output_size(in_len, 0);
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    let mut output = ArrayD::zeros(IxDyn(&[batch, out_channels, out_len]));

    let stride = config.stride[0];
    let padding = config.padding[0];
    let dilation = config.dilation[0];

    for b in 0..batch {
        for g in 0..groups {
            let oc_start = g * out_channels_per_group;
            let ic_start = g * in_channels_per_group;

            for oc in 0..out_channels_per_group {
                for ol in 0..out_len {
                    let mut sum = 0.0_f64;
                    for ic in 0..in_channels_per_group {
                        for kl in 0..kernel_len {
                            let il_raw = ol as isize * stride as isize
                                + kl as isize * dilation as isize
                                - padding as isize;
                            if il_raw >= 0 && (il_raw as usize) < in_len {
                                let il = il_raw as usize;
                                sum += input[[b, ic_start + ic, il].as_ref()]
                                    * weight[[oc_start + oc, ic, kl].as_ref()];
                            }
                        }
                    }
                    output[[b, oc_start + oc, ol].as_ref()] = sum;
                }
            }
        }
    }

    // Apply bias
    if let Some(bias_arr) = bias {
        for b in 0..batch {
            for oc in 0..out_channels {
                let bias_val = bias_arr[IxDyn(&[oc])];
                for ol in 0..out_len {
                    output[[b, oc, ol].as_ref()] += bias_val;
                }
            }
        }
    }

    Ok(output)
}

/// 2D convolution.
///
/// - Input shape: `[batch, in_channels, height, width]`
/// - Weight shape: `[out_channels, in_channels/groups, kH, kW]`
/// - Output shape: `[batch, out_channels, outH, outW]`
pub fn conv2d(
    input: &ArrayD<f64>,
    weight: &ArrayD<f64>,
    bias: Option<&ArrayD<f64>>,
    config: &ConvConfig,
) -> Result<ArrayD<f64>, ConvError> {
    config.validate()?;

    let in_shape = input.shape();
    if in_shape.is_empty() || input.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    if in_shape.len() != 4 {
        return Err(ConvError::InsufficientDimensions {
            ndim: in_shape.len(),
            required: 4,
        });
    }

    let w_shape = weight.shape();
    if w_shape.len() != 4 {
        return Err(ConvError::InsufficientDimensions {
            ndim: w_shape.len(),
            required: 4,
        });
    }

    let batch = in_shape[0];
    let in_channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];
    let out_channels = w_shape[0];
    let groups = config.groups;

    if !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
        return Err(ConvError::InvalidGroups {
            groups,
            in_channels,
            out_channels,
        });
    }

    let out_h = config.output_size(in_h, 0);
    let out_w = config.output_size(in_w, 1);
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    let k_h = config.kernel_size[0];
    let k_w = config.kernel_size[1];
    let stride_h = config.stride[0];
    let stride_w = config.stride[1];
    let pad_h = config.padding[0];
    let pad_w = config.padding[1];
    let dil_h = config.dilation[0];
    let dil_w = config.dilation[1];

    let mut output = ArrayD::zeros(IxDyn(&[batch, out_channels, out_h, out_w]));

    for b in 0..batch {
        for g in 0..groups {
            let oc_start = g * out_channels_per_group;
            let ic_start = g * in_channels_per_group;

            for oc in 0..out_channels_per_group {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0_f64;
                        for ic in 0..in_channels_per_group {
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih_raw = oh as isize * stride_h as isize
                                        + kh as isize * dil_h as isize
                                        - pad_h as isize;
                                    let iw_raw = ow as isize * stride_w as isize
                                        + kw as isize * dil_w as isize
                                        - pad_w as isize;
                                    if ih_raw >= 0
                                        && (ih_raw as usize) < in_h
                                        && iw_raw >= 0
                                        && (iw_raw as usize) < in_w
                                    {
                                        let ih = ih_raw as usize;
                                        let iw = iw_raw as usize;
                                        sum += input[IxDyn(&[b, ic_start + ic, ih, iw])]
                                            * weight[IxDyn(&[oc_start + oc, ic, kh, kw])];
                                    }
                                }
                            }
                        }
                        output[IxDyn(&[b, oc_start + oc, oh, ow])] = sum;
                    }
                }
            }
        }
    }

    // Apply bias
    if let Some(bias_arr) = bias {
        for b in 0..batch {
            for oc in 0..out_channels {
                let bias_val = bias_arr[IxDyn(&[oc])];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        output[IxDyn(&[b, oc, oh, ow])] += bias_val;
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Transposed 2D convolution (deconvolution / fractionally-strided convolution).
///
/// - Input shape: `[batch, in_channels, height, width]`
/// - Weight shape: `[in_channels, out_channels/groups, kH, kW]`
/// - Output shape: `[batch, out_channels, outH, outW]`
///
/// Output size formula per dimension:
/// `(input - 1) * stride - 2*padding + dilation*(kernel - 1) + output_padding + 1`
pub fn conv_transpose2d(
    input: &ArrayD<f64>,
    weight: &ArrayD<f64>,
    bias: Option<&ArrayD<f64>>,
    config: &ConvConfig,
    output_padding: &[usize],
) -> Result<ArrayD<f64>, ConvError> {
    config.validate()?;

    let in_shape = input.shape();
    if in_shape.is_empty() || input.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    if in_shape.len() != 4 {
        return Err(ConvError::InsufficientDimensions {
            ndim: in_shape.len(),
            required: 4,
        });
    }

    let w_shape = weight.shape();
    if w_shape.len() != 4 {
        return Err(ConvError::InsufficientDimensions {
            ndim: w_shape.len(),
            required: 4,
        });
    }

    let batch = in_shape[0];
    let in_channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];
    let groups = config.groups;

    // For transposed conv, weight is [in_channels, out_channels/groups, kH, kW]
    let out_channels_per_group = w_shape[1];
    let out_channels = out_channels_per_group * groups;

    if !in_channels.is_multiple_of(groups) {
        return Err(ConvError::InvalidGroups {
            groups,
            in_channels,
            out_channels,
        });
    }

    let in_channels_per_group = in_channels / groups;
    let k_h = config.kernel_size[0];
    let k_w = config.kernel_size[1];
    let stride_h = config.stride[0];
    let stride_w = config.stride[1];
    let pad_h = config.padding[0];
    let pad_w = config.padding[1];
    let dil_h = config.dilation[0];
    let dil_w = config.dilation[1];

    let out_pad_h = if output_padding.is_empty() {
        0
    } else {
        output_padding[0]
    };
    let out_pad_w = if output_padding.len() < 2 {
        0
    } else {
        output_padding[1]
    };

    let out_h = (in_h - 1) * stride_h + dil_h * (k_h - 1) + 1 + out_pad_h - 2 * pad_h;
    let out_w = (in_w - 1) * stride_w + dil_w * (k_w - 1) + 1 + out_pad_w - 2 * pad_w;

    let mut output = ArrayD::zeros(IxDyn(&[batch, out_channels, out_h, out_w]));

    // Transposed convolution: for each input position, scatter-add weighted kernel to output
    for b in 0..batch {
        for g in 0..groups {
            let ic_start = g * in_channels_per_group;
            let oc_start = g * out_channels_per_group;

            for ic in 0..in_channels_per_group {
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let input_val = input[IxDyn(&[b, ic_start + ic, ih, iw])];
                        for oc in 0..out_channels_per_group {
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let oh_raw = ih as isize * stride_h as isize
                                        + kh as isize * dil_h as isize
                                        - pad_h as isize;
                                    let ow_raw = iw as isize * stride_w as isize
                                        + kw as isize * dil_w as isize
                                        - pad_w as isize;
                                    if oh_raw >= 0
                                        && (oh_raw as usize) < out_h
                                        && ow_raw >= 0
                                        && (ow_raw as usize) < out_w
                                    {
                                        let oh = oh_raw as usize;
                                        let ow = ow_raw as usize;
                                        output[IxDyn(&[b, oc_start + oc, oh, ow])] +=
                                            input_val * weight[IxDyn(&[ic_start + ic, oc, kh, kw])];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply bias
    if let Some(bias_arr) = bias {
        for b in 0..batch {
            for oc in 0..out_channels {
                let bias_val = bias_arr[IxDyn(&[oc])];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        output[IxDyn(&[b, oc, oh, ow])] += bias_val;
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Depthwise 2D convolution: groups == in_channels == out_channels.
///
/// Convenience wrapper around [`conv2d`] that sets `groups = in_channels`.
/// Weight shape: `[in_channels, 1, kH, kW]` (one filter per channel).
pub fn depthwise_conv2d(
    input: &ArrayD<f64>,
    weight: &ArrayD<f64>,
    bias: Option<&ArrayD<f64>>,
    config: &ConvConfig,
) -> Result<ArrayD<f64>, ConvError> {
    let in_shape = input.shape();
    if in_shape.len() < 4 {
        return Err(ConvError::InsufficientDimensions {
            ndim: in_shape.len(),
            required: 4,
        });
    }

    let in_channels = in_shape[1];
    let mut dw_config = config.clone();
    dw_config.groups = in_channels;

    conv2d(input, weight, bias, &dw_config)
}

/// im2col: unfold input patches into columns for efficient convolution via GEMM.
///
/// - Input shape: `[batch, channels, H, W]`
/// - Output shape: `[batch, channels * kH * kW, outH * outW]`
pub fn im2col(
    input: &ArrayD<f64>,
    kernel_size: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) -> Result<ArrayD<f64>, ConvError> {
    let in_shape = input.shape();
    if in_shape.is_empty() || input.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    if in_shape.len() != 4 {
        return Err(ConvError::InsufficientDimensions {
            ndim: in_shape.len(),
            required: 4,
        });
    }
    if kernel_size.len() != 2 || stride.len() != 2 || padding.len() != 2 || dilation.len() != 2 {
        return Err(ConvError::InvalidKernelSize(
            "im2col requires exactly 2 spatial dimensions".to_string(),
        ));
    }

    let batch = in_shape[0];
    let channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];
    let k_h = kernel_size[0];
    let k_w = kernel_size[1];
    let s_h = stride[0];
    let s_w = stride[1];
    let p_h = padding[0];
    let p_w = padding[1];
    let d_h = dilation[0];
    let d_w = dilation[1];

    let eff_k_h = d_h * (k_h - 1) + 1;
    let eff_k_w = d_w * (k_w - 1) + 1;
    let out_h = (in_h + 2 * p_h - eff_k_h) / s_h + 1;
    let out_w = (in_w + 2 * p_w - eff_k_w) / s_w + 1;

    let col_rows = channels * k_h * k_w;
    let col_cols = out_h * out_w;
    let mut cols = ArrayD::zeros(IxDyn(&[batch, col_rows, col_cols]));

    for b in 0..batch {
        let mut col_idx = 0;
        for c in 0..channels {
            for kh in 0..k_h {
                for kw in 0..k_w {
                    let mut spatial_idx = 0;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let ih_raw = oh as isize * s_h as isize + kh as isize * d_h as isize
                                - p_h as isize;
                            let iw_raw = ow as isize * s_w as isize + kw as isize * d_w as isize
                                - p_w as isize;
                            let val = if ih_raw >= 0
                                && (ih_raw as usize) < in_h
                                && iw_raw >= 0
                                && (iw_raw as usize) < in_w
                            {
                                input[IxDyn(&[b, c, ih_raw as usize, iw_raw as usize])]
                            } else {
                                0.0
                            };
                            cols[IxDyn(&[b, col_idx, spatial_idx])] = val;
                            spatial_idx += 1;
                        }
                    }
                    col_idx += 1;
                }
            }
        }
    }

    Ok(cols)
}

/// col2im: fold columns back into image form (inverse of im2col).
///
/// - Cols shape: `[batch, channels * kH * kW, outH * outW]`
/// - Output shape: `[batch, channels, H, W]` (specified via `output_size`)
///
/// Where overlapping patches are summed (accumulated).
pub fn col2im(
    cols: &ArrayD<f64>,
    output_size: &[usize],
    kernel_size: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) -> Result<ArrayD<f64>, ConvError> {
    let col_shape = cols.shape();
    if col_shape.is_empty() || cols.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    if col_shape.len() != 3 {
        return Err(ConvError::InsufficientDimensions {
            ndim: col_shape.len(),
            required: 3,
        });
    }
    if output_size.len() != 4 {
        return Err(ConvError::InvalidKernelSize(
            "output_size must have 4 elements [batch, channels, H, W]".to_string(),
        ));
    }

    let batch = output_size[0];
    let channels = output_size[1];
    let out_h_img = output_size[2];
    let out_w_img = output_size[3];

    let k_h = kernel_size[0];
    let k_w = kernel_size[1];
    let s_h = stride[0];
    let s_w = stride[1];
    let p_h = padding[0];
    let p_w = padding[1];
    let d_h = dilation[0];
    let d_w = dilation[1];

    let eff_k_h = d_h * (k_h - 1) + 1;
    let eff_k_w = d_w * (k_w - 1) + 1;
    let col_out_h = (out_h_img + 2 * p_h - eff_k_h) / s_h + 1;
    let col_out_w = (out_w_img + 2 * p_w - eff_k_w) / s_w + 1;

    let mut output = ArrayD::zeros(IxDyn(&[batch, channels, out_h_img, out_w_img]));

    for b in 0..batch {
        let mut col_idx = 0;
        for c in 0..channels {
            for kh in 0..k_h {
                for kw in 0..k_w {
                    let mut spatial_idx = 0;
                    for oh in 0..col_out_h {
                        for ow in 0..col_out_w {
                            let ih_raw = oh as isize * s_h as isize + kh as isize * d_h as isize
                                - p_h as isize;
                            let iw_raw = ow as isize * s_w as isize + kw as isize * d_w as isize
                                - p_w as isize;
                            if ih_raw >= 0
                                && (ih_raw as usize) < out_h_img
                                && iw_raw >= 0
                                && (iw_raw as usize) < out_w_img
                            {
                                output[IxDyn(&[b, c, ih_raw as usize, iw_raw as usize])] +=
                                    cols[IxDyn(&[b, col_idx, spatial_idx])];
                            }
                            spatial_idx += 1;
                        }
                    }
                    col_idx += 1;
                }
            }
        }
    }

    Ok(output)
}

/// Statistics about a convolution operation (parameter count, FLOPs, receptive field).
#[derive(Debug, Clone)]
pub struct ConvStats {
    /// Shape of the input tensor.
    pub input_shape: Vec<usize>,
    /// Shape of the output tensor.
    pub output_shape: Vec<usize>,
    /// Shape of the kernel/weight tensor.
    pub kernel_shape: Vec<usize>,
    /// Total number of learnable parameters (weights + bias if present).
    pub num_parameters: usize,
    /// Estimated floating-point operations (multiply-accumulate counted as 2).
    pub flops: u64,
    /// Receptive field size in each spatial dimension.
    pub receptive_field: Vec<usize>,
}

impl ConvStats {
    /// Compute convolution statistics from input/weight shapes and config.
    ///
    /// Input shape: `[batch, in_channels, spatial...]`
    /// Weight shape: `[out_channels, in_channels/groups, kernel_spatial...]`
    pub fn compute(
        input_shape: &[usize],
        weight_shape: &[usize],
        config: &ConvConfig,
    ) -> Result<Self, ConvError> {
        config.validate()?;

        if input_shape.len() < 3 {
            return Err(ConvError::InsufficientDimensions {
                ndim: input_shape.len(),
                required: 3,
            });
        }
        if weight_shape.len() < 3 {
            return Err(ConvError::InsufficientDimensions {
                ndim: weight_shape.len(),
                required: 3,
            });
        }

        let batch = input_shape[0];
        let out_channels = weight_shape[0];
        let ndim = config.num_spatial_dims();

        // Compute output spatial dimensions
        let mut output_spatial = Vec::with_capacity(ndim);
        for d in 0..ndim {
            let in_size = input_shape[2 + d];
            output_spatial.push(config.output_size(in_size, d));
        }

        let mut output_shape = vec![batch, out_channels];
        output_shape.extend_from_slice(&output_spatial);

        // Number of parameters: weight elements + bias (out_channels)
        let weight_params: usize = weight_shape.iter().product();
        let num_parameters = weight_params + out_channels; // assume bias present

        // FLOPs: for each output element, we do kernel_volume * in_channels_per_group
        // multiply-accumulates. Each MAC = 2 ops (mul + add).
        let kernel_volume: usize = config.kernel_size.iter().product();
        let in_channels_per_group = if config.groups > 0 {
            weight_shape[1]
        } else {
            return Err(ConvError::InvalidGroups {
                groups: 0,
                in_channels: 0,
                out_channels: 0,
            });
        };
        let output_elements: u64 = output_shape.iter().map(|&s| s as u64).product();
        let macs_per_element = (kernel_volume * in_channels_per_group) as u64;
        let flops = output_elements * macs_per_element * 2;

        // Receptive field per spatial dim: dilation * (kernel - 1) + 1
        let receptive_field: Vec<usize> = (0..ndim)
            .map(|d| config.dilation[d] * (config.kernel_size[d] - 1) + 1)
            .collect();

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_shape,
            kernel_shape: weight_shape.to_vec(),
            num_parameters,
            flops,
            receptive_field,
        })
    }

    /// Human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "ConvStats {{ input: {:?}, output: {:?}, kernel: {:?}, \
             params: {}, flops: {}, receptive_field: {:?} }}",
            self.input_shape,
            self.output_shape,
            self.kernel_shape,
            self.num_parameters,
            self.flops,
            self.receptive_field,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_conv_config_output_size() {
        // kernel=3, stride=1, pad=1, dilation=1: same size
        let cfg = ConvConfig::new(vec![3, 3]).with_padding(vec![1, 1]);
        assert_eq!(cfg.output_size(8, 0), 8);
        assert_eq!(cfg.output_size(8, 1), 8);
    }

    #[test]
    fn test_conv_config_validate_valid() {
        let cfg = ConvConfig::new(vec![3, 3])
            .with_stride(vec![1, 1])
            .with_padding(vec![1, 1])
            .with_dilation(vec![1, 1])
            .with_groups(1);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_conv_config_validate_zero_kernel() {
        let cfg = ConvConfig::new(vec![0, 3]);
        let err = cfg.validate();
        assert!(err.is_err());
        let msg = format!("{}", err.expect_err("expected error"));
        assert!(msg.contains("kernel_size"));
    }

    #[test]
    fn test_conv1d_basic() {
        // Input: [1, 1, 5], Kernel: [1, 1, 3], no padding, stride=1
        // Output length = (5 - 3) / 1 + 1 = 3
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("input shape");
        let weight =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0, 1.0, 1.0]).expect("weight shape");
        let cfg = ConvConfig::new(vec![3]);

        let out = conv1d(&input, &weight, None, &cfg).expect("conv1d");
        assert_eq!(out.shape(), &[1, 1, 3]);
        // [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
        assert!((out[IxDyn(&[0, 0, 0])] - 6.0).abs() < 1e-10);
        assert!((out[IxDyn(&[0, 0, 1])] - 9.0).abs() < 1e-10);
        assert!((out[IxDyn(&[0, 0, 2])] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv1d_with_bias() {
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0, 2.0, 3.0]).expect("input shape");
        let weight = ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .expect("weight shape");
        let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).expect("bias shape");
        let cfg = ConvConfig::new(vec![3]);

        let out = conv1d(&input, &weight, Some(&bias), &cfg).expect("conv1d");
        assert_eq!(out.shape(), &[1, 2, 1]);
        // channel 0: 1*1 + 0*2 + 0*3 + 10 = 11
        assert!((out[IxDyn(&[0, 0, 0])] - 11.0).abs() < 1e-10);
        // channel 1: 0*1 + 0*2 + 1*3 + 20 = 23
        assert!((out[IxDyn(&[0, 1, 0])] - 23.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_identity_kernel() {
        // 1x1 kernel = channel mixing only, spatial preserved
        let input = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .expect("input shape");
        // Weight [1, 2, 1, 1]: output channel 0 = 1*ch0 + 1*ch1
        let weight =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 1, 1]), vec![1.0, 1.0]).expect("weight shape");
        let cfg = ConvConfig::new(vec![1, 1]);

        let out = conv2d(&input, &weight, None, &cfg).expect("conv2d");
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        // (0,0): 1+5=6, (0,1): 2+6=8, (1,0): 3+7=10, (1,1): 4+8=12
        assert!((out[IxDyn(&[0, 0, 0, 0])] - 6.0).abs() < 1e-10);
        assert!((out[IxDyn(&[0, 0, 0, 1])] - 8.0).abs() < 1e-10);
        assert!((out[IxDyn(&[0, 0, 1, 0])] - 10.0).abs() < 1e-10);
        assert!((out[IxDyn(&[0, 0, 1, 1])] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_basic() {
        // [1,1,4,4] input, [1,1,3,3] kernel, no padding → [1,1,2,2]
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 4, 4]), (1..=16).map(|x| x as f64).collect())
                .expect("input shape");
        let weight = ArrayD::ones(IxDyn(&[1, 1, 3, 3]));
        let cfg = ConvConfig::new(vec![3, 3]);

        let out = conv2d(&input, &weight, None, &cfg).expect("conv2d");
        assert_eq!(out.shape(), &[1, 1, 2, 2]);

        // Top-left 3x3: 1+2+3+5+6+7+9+10+11 = 54
        assert!((out[IxDyn(&[0, 0, 0, 0])] - 54.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_with_padding() {
        // 3x3 kernel, padding=1 → same spatial size
        let input = ArrayD::ones(IxDyn(&[1, 1, 4, 4]));
        let weight = ArrayD::ones(IxDyn(&[1, 1, 3, 3]));
        let cfg = ConvConfig::new(vec![3, 3]).with_padding(vec![1, 1]);

        let out = conv2d(&input, &weight, None, &cfg).expect("conv2d");
        assert_eq!(out.shape(), &[1, 1, 4, 4]);

        // Center pixel: all 9 neighbors present → 9.0
        assert!((out[IxDyn(&[0, 0, 1, 1])] - 9.0).abs() < 1e-10);
        // Corner: 4 neighbors present → 4.0
        assert!((out[IxDyn(&[0, 0, 0, 0])] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_stride2() {
        // stride=2 → output halved
        let input = ArrayD::ones(IxDyn(&[1, 1, 4, 4]));
        let weight = ArrayD::ones(IxDyn(&[1, 1, 3, 3]));
        let cfg = ConvConfig::new(vec![3, 3])
            .with_stride(vec![2, 2])
            .with_padding(vec![1, 1]);

        let out = conv2d(&input, &weight, None, &cfg).expect("conv2d");
        // output_size = (4 + 2 - 3) / 2 + 1 = 2
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_conv2d_groups() {
        // 2 input channels, 2 output channels, groups=2 → each group has 1 in/out channel
        let input = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 3, 3]),
            vec![
                // ch0: all 1s
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // ch1: all 2s
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            ],
        )
        .expect("input shape");
        // Weight: [2, 1, 3, 3] — 2 output channels, 1 input channel per group
        let weight = ArrayD::ones(IxDyn(&[2, 1, 3, 3]));
        let cfg = ConvConfig::new(vec![3, 3]).with_groups(2);

        let out = conv2d(&input, &weight, None, &cfg).expect("conv2d");
        assert_eq!(out.shape(), &[1, 2, 1, 1]);
        // Group 0: sum of 1s in 3x3 = 9
        assert!((out[IxDyn(&[0, 0, 0, 0])] - 9.0).abs() < 1e-10);
        // Group 1: sum of 2s in 3x3 = 18
        assert!((out[IxDyn(&[0, 1, 0, 0])] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_dilation() {
        // dilation=2 with 3x3 kernel: effective kernel = 5x5
        let input = ArrayD::ones(IxDyn(&[1, 1, 7, 7]));
        let weight = ArrayD::ones(IxDyn(&[1, 1, 3, 3]));
        let cfg = ConvConfig::new(vec![3, 3]).with_dilation(vec![2, 2]);

        let out = conv2d(&input, &weight, None, &cfg).expect("conv2d");
        // output_size = (7 - 2*(3-1) - 1) / 1 + 1 = (7 - 5) / 1 + 1 = 3
        assert_eq!(out.shape(), &[1, 1, 3, 3]);
        // All 9 sampled positions are within bounds and input=1 → sum=9
        assert!((out[IxDyn(&[0, 0, 1, 1])] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv_transpose2d_basic() {
        // Input [1,1,2,2], weight [1,1,3,3], stride=2 → upsamples
        // output_size = (2-1)*2 + 3 + 0 - 0 = 5 per dim
        let input = ArrayD::ones(IxDyn(&[1, 1, 2, 2]));
        let weight = ArrayD::ones(IxDyn(&[1, 1, 3, 3]));
        let cfg = ConvConfig::new(vec![3, 3]).with_stride(vec![2, 2]);

        let out = conv_transpose2d(&input, &weight, None, &cfg, &[]).expect("conv_transpose2d");
        assert_eq!(out.shape(), &[1, 1, 5, 5]);
        // Center (2,2): overlapped by all 4 input positions → 4
        assert!((out[IxDyn(&[0, 0, 2, 2])] - 4.0).abs() < 1e-10);
        // Corner (0,0): only 1 input contributes → 1
        assert!((out[IxDyn(&[0, 0, 0, 0])] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_depthwise_conv2d() {
        // 2 channels, depthwise: each channel convolved independently
        let input = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 3, 3]),
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // ch0
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, // ch1
            ],
        )
        .expect("input shape");
        // Weight: [2, 1, 3, 3] — each channel has its own filter
        let weight = ArrayD::ones(IxDyn(&[2, 1, 3, 3]));
        let cfg = ConvConfig::new(vec![3, 3]);

        let out = depthwise_conv2d(&input, &weight, None, &cfg).expect("depthwise");
        assert_eq!(out.shape(), &[1, 2, 1, 1]);
        assert!((out[IxDyn(&[0, 0, 0, 0])] - 9.0).abs() < 1e-10);
        assert!((out[IxDyn(&[0, 1, 0, 0])] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_im2col_shape() {
        // [1, 2, 4, 4], kernel 3x3, stride 1, pad 0 → cols: [1, 2*3*3=18, 2*2=4]
        let input = ArrayD::ones(IxDyn(&[1, 2, 4, 4]));
        let cols = im2col(&input, &[3, 3], &[1, 1], &[0, 0], &[1, 1]).expect("im2col");
        assert_eq!(cols.shape(), &[1, 18, 4]);
    }

    #[test]
    fn test_im2col_values() {
        // [1, 1, 3, 3] input with values 1..9, kernel=2x2
        let input = ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .expect("input shape");

        let cols = im2col(&input, &[2, 2], &[1, 1], &[0, 0], &[1, 1]).expect("im2col");
        // output spatial: (3-2)/1+1 = 2 per dim → 4 columns
        assert_eq!(cols.shape(), &[1, 4, 4]);

        // First column (oh=0,ow=0): patch at (0,0)→(1,1) = [1,2,4,5]
        assert!((cols[IxDyn(&[0, 0, 0])] - 1.0).abs() < 1e-10);
        assert!((cols[IxDyn(&[0, 1, 0])] - 2.0).abs() < 1e-10);
        assert!((cols[IxDyn(&[0, 2, 0])] - 4.0).abs() < 1e-10);
        assert!((cols[IxDyn(&[0, 3, 0])] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_col2im_roundtrip_no_overlap() {
        // With stride >= kernel, patches don't overlap → roundtrip is exact
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 4, 4]), (1..=16).map(|x| x as f64).collect())
                .expect("input shape");

        let kernel = [2, 2];
        let stride = [2, 2];
        let padding = [0, 0];
        let dilation = [1, 1];

        let cols = im2col(&input, &kernel, &stride, &padding, &dilation).expect("im2col");
        let reconstructed =
            col2im(&cols, &[1, 1, 4, 4], &kernel, &stride, &padding, &dilation).expect("col2im");

        assert_eq!(reconstructed.shape(), input.shape());
        for (a, b) in input.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_conv_stats_flops() {
        let cfg = ConvConfig::new(vec![3, 3]);
        let stats = ConvStats::compute(&[1, 3, 32, 32], &[16, 3, 3, 3], &cfg).expect("conv stats");
        assert!(stats.flops > 0);
    }

    #[test]
    fn test_conv_stats_parameters() {
        // Weight [16, 3, 3, 3] = 432 + 16 bias = 448
        let cfg = ConvConfig::new(vec![3, 3]);
        let stats = ConvStats::compute(&[1, 3, 32, 32], &[16, 3, 3, 3], &cfg).expect("conv stats");
        assert_eq!(stats.num_parameters, 432 + 16);
    }

    #[test]
    fn test_conv_stats_summary_nonempty() {
        let cfg = ConvConfig::new(vec![3, 3]);
        let stats = ConvStats::compute(&[1, 3, 32, 32], &[16, 3, 3, 3], &cfg).expect("conv stats");
        let s = stats.summary();
        assert!(!s.is_empty());
        assert!(s.contains("ConvStats"));
    }

    #[test]
    fn test_conv_error_display() {
        let errors: Vec<ConvError> = vec![
            ConvError::InvalidKernelSize("zero".to_string()),
            ConvError::InvalidStride("zero".to_string()),
            ConvError::InvalidPadding("negative".to_string()),
            ConvError::InvalidDilation("zero".to_string()),
            ConvError::ShapeMismatch {
                expected: vec![1, 2],
                got: vec![3, 4],
            },
            ConvError::InsufficientDimensions {
                ndim: 2,
                required: 4,
            },
            ConvError::InvalidGroups {
                groups: 3,
                in_channels: 4,
                out_channels: 6,
            },
            ConvError::EmptyInput,
        ];
        for err in &errors {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "error display should be non-empty");
        }
    }
}

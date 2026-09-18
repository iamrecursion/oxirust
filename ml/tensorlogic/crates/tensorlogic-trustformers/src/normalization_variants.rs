//! Advanced normalization variants for transformer architectures.
//!
//! This module provides numerical (ndarray-based) implementations of normalization
//! techniques beyond standard LayerNorm, including:
//!
//! - **RmsNorm**: Root Mean Square Layer Normalization (no mean centering)
//! - **GroupNorm**: Group Normalization (divides channels into groups)
//! - **InstanceNorm**: Instance Normalization (per-instance, per-channel)
//! - **BatchNorm**: Batch Normalization (across-batch statistics)
//! - **WeightNorm**: Weight Normalization (weight reparametrization)
//! - **NormStats**: Normalization statistics for debugging/monitoring

use ndarray::{ArrayD, Axis, IxDyn};

/// Errors that can occur during normalization operations.
#[derive(Debug, Clone)]
pub enum NormalizationError {
    /// Shape mismatch between expected and actual tensor shapes.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Invalid axis for the given tensor dimensionality.
    InvalidAxis { axis: usize, ndim: usize },
    /// Number of groups does not evenly divide number of channels.
    InvalidNumGroups { groups: usize, channels: usize },
    /// Encountered zero variance during normalization.
    ZeroVariance,
    /// Input tensor is empty.
    EmptyInput,
    /// Tensor does not have enough dimensions for the operation.
    InsufficientDimensions { ndim: usize, required: usize },
}

impl std::fmt::Display for NormalizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::InvalidAxis { axis, ndim } => {
                write!(
                    f,
                    "Invalid axis {} for tensor with {} dimensions",
                    axis, ndim
                )
            }
            Self::InvalidNumGroups { groups, channels } => {
                write!(
                    f,
                    "Invalid number of groups {}: does not evenly divide {} channels",
                    groups, channels
                )
            }
            Self::ZeroVariance => write!(f, "Zero variance encountered during normalization"),
            Self::EmptyInput => write!(f, "Empty input tensor"),
            Self::InsufficientDimensions { ndim, required } => {
                write!(
                    f,
                    "Insufficient dimensions: tensor has {} dims, but {} required",
                    ndim, required
                )
            }
        }
    }
}

impl std::error::Error for NormalizationError {}

// ---------------------------------------------------------------------------
// Helper: compute mean and variance along a set of axes
// ---------------------------------------------------------------------------

/// Compute element count for given axes.
fn axis_element_count(shape: &[usize], axes: &[usize]) -> f64 {
    axes.iter().map(|&a| shape[a] as f64).product()
}

/// Compute mean along specified axes, keeping dimensions.
fn mean_along_axes(input: &ArrayD<f64>, axes: &[usize]) -> ArrayD<f64> {
    let mut result = input.clone();
    // Process axes in descending order so indices remain valid
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort_unstable();
    sorted_axes.reverse();

    let count = axis_element_count(input.shape(), axes);

    for &ax in &sorted_axes {
        result = result.sum_axis(Axis(ax)).insert_axis(Axis(ax));
    }
    result / count
}

/// Compute variance along specified axes, keeping dimensions.
fn var_along_axes(input: &ArrayD<f64>, axes: &[usize]) -> ArrayD<f64> {
    let mean = mean_along_axes(input, axes);
    let diff = input - &mean;
    let sq = &diff * &diff;
    mean_along_axes(&sq, axes)
}

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization (no mean centering).
///
/// Normalizes by: `x / sqrt(mean(x^2) + eps) * gamma`
///
/// Used in LLaMA and other modern transformer architectures.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Shape of the dimensions to normalize over (typically the last N dims).
    pub normalized_shape: Vec<usize>,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Learnable scale parameter.
    pub gamma: ArrayD<f64>,
}

impl RmsNorm {
    /// Create a new RmsNorm layer.
    ///
    /// `normalized_shape` specifies the trailing dimensions to normalize over.
    /// `gamma` is initialized to ones.
    pub fn new(normalized_shape: Vec<usize>, eps: f64) -> Result<Self, NormalizationError> {
        if normalized_shape.is_empty() {
            return Err(NormalizationError::EmptyInput);
        }
        let gamma = ArrayD::ones(IxDyn(&normalized_shape));
        Ok(Self {
            normalized_shape,
            eps,
            gamma,
        })
    }

    /// Forward pass: normalize the input tensor.
    ///
    /// The last `normalized_shape.len()` dimensions of the input must match
    /// `normalized_shape`.
    pub fn forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, NormalizationError> {
        let ndim = input.ndim();
        let norm_ndim = self.normalized_shape.len();
        if ndim < norm_ndim {
            return Err(NormalizationError::InsufficientDimensions {
                ndim,
                required: norm_ndim,
            });
        }
        if input.is_empty() {
            return Err(NormalizationError::EmptyInput);
        }

        // Verify trailing shape matches
        let trailing: Vec<usize> = input.shape()[(ndim - norm_ndim)..].to_vec();
        if trailing != self.normalized_shape {
            return Err(NormalizationError::ShapeMismatch {
                expected: self.normalized_shape.clone(),
                got: trailing,
            });
        }

        // Axes to reduce over (trailing dimensions)
        let axes: Vec<usize> = ((ndim - norm_ndim)..ndim).collect();
        let rms = Self::rms(input, &axes);

        // x / (rms + eps) * gamma
        let rms_inv = rms.mapv(|v| 1.0 / (v + self.eps));
        let normalized = input * &rms_inv;
        Ok(normalized * &self.gamma)
    }

    /// Compute Root Mean Square along specified axes (keeping dims).
    pub fn rms(input: &ArrayD<f64>, axes: &[usize]) -> ArrayD<f64> {
        let sq = input.mapv(|x| x * x);
        let mean_sq = mean_along_axes(&sq, axes);
        mean_sq.mapv(f64::sqrt)
    }

    /// Update the learnable scale parameter gamma.
    pub fn update_gamma(&mut self, new_gamma: ArrayD<f64>) -> Result<(), NormalizationError> {
        let expected: Vec<usize> = self.normalized_shape.clone();
        let got: Vec<usize> = new_gamma.shape().to_vec();
        if expected != got {
            return Err(NormalizationError::ShapeMismatch { expected, got });
        }
        self.gamma = new_gamma;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

/// Group Normalization: divides channels into groups and normalizes within each group.
///
/// Input shape: `[batch, channels, ...spatial_dims...]`
#[derive(Debug, Clone)]
pub struct GroupNorm {
    /// Number of groups to divide channels into.
    pub num_groups: usize,
    /// Total number of channels.
    pub num_channels: usize,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Learnable scale parameter `[channels]`.
    pub gamma: ArrayD<f64>,
    /// Learnable shift parameter `[channels]`.
    pub beta: ArrayD<f64>,
    /// Whether to apply learnable affine transformation.
    pub affine: bool,
}

impl GroupNorm {
    /// Create a new GroupNorm layer.
    ///
    /// `num_channels` must be evenly divisible by `num_groups`.
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: f64,
        affine: bool,
    ) -> Result<Self, NormalizationError> {
        if num_groups == 0 || num_channels == 0 {
            return Err(NormalizationError::EmptyInput);
        }
        if !num_channels.is_multiple_of(num_groups) {
            return Err(NormalizationError::InvalidNumGroups {
                groups: num_groups,
                channels: num_channels,
            });
        }
        let gamma = ArrayD::ones(IxDyn(&[num_channels]));
        let beta = ArrayD::zeros(IxDyn(&[num_channels]));
        Ok(Self {
            num_groups,
            num_channels,
            eps,
            gamma,
            beta,
            affine,
        })
    }

    /// Forward pass: normalize within each group.
    ///
    /// Input shape: `[batch, channels, ...spatial_dims...]` (at least 2-D).
    pub fn forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, NormalizationError> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(NormalizationError::InsufficientDimensions { ndim, required: 2 });
        }
        if input.is_empty() {
            return Err(NormalizationError::EmptyInput);
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[1];

        if channels != self.num_channels {
            return Err(NormalizationError::ShapeMismatch {
                expected: vec![batch_size, self.num_channels],
                got: vec![batch_size, channels],
            });
        }

        let cpg = self.channels_per_group();
        let spatial: Vec<usize> = shape[2..].to_vec();

        // Build reshaped dims: [B, G, C/G, *spatial]
        let mut reshaped_dims = vec![batch_size, self.num_groups, cpg];
        reshaped_dims.extend_from_slice(&spatial);

        let reshaped = input
            .clone()
            .into_shape_with_order(IxDyn(&reshaped_dims))
            .map_err(|_| NormalizationError::ShapeMismatch {
                expected: reshaped_dims.clone(),
                got: shape.to_vec(),
            })?;

        // Normalize over axes 2.. (C/G and spatial dims)
        let norm_axes: Vec<usize> = (2..reshaped.ndim()).collect();
        let mean = mean_along_axes(&reshaped, &norm_axes);
        let var = var_along_axes(&reshaped, &norm_axes);

        let inv_std = var.mapv(|v| 1.0 / (v + self.eps).sqrt());
        let normalized = (&reshaped - &mean) * &inv_std;

        // Reshape back to [B, C, *spatial]
        let mut out_shape = vec![batch_size, channels];
        out_shape.extend_from_slice(&spatial);
        let mut output = normalized
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|_| NormalizationError::ShapeMismatch {
                expected: out_shape.clone(),
                got: vec![],
            })?;

        // Apply affine: broadcast gamma/beta over batch and spatial dims
        if self.affine {
            // Build broadcast shape for gamma/beta: [1, C, 1, 1, ...]
            let mut broadcast_shape = vec![1usize; ndim];
            broadcast_shape[1] = channels;

            let gamma_bc = self
                .gamma
                .clone()
                .into_shape_with_order(IxDyn(&broadcast_shape))
                .map_err(|_| NormalizationError::ShapeMismatch {
                    expected: broadcast_shape.clone(),
                    got: self.gamma.shape().to_vec(),
                })?;
            let beta_bc = self
                .beta
                .clone()
                .into_shape_with_order(IxDyn(&broadcast_shape))
                .map_err(|_| NormalizationError::ShapeMismatch {
                    expected: broadcast_shape.clone(),
                    got: self.beta.shape().to_vec(),
                })?;

            output = output * &gamma_bc + &beta_bc;
        }

        Ok(output)
    }

    /// Number of channels per group.
    pub fn channels_per_group(&self) -> usize {
        self.num_channels / self.num_groups
    }
}

// ---------------------------------------------------------------------------
// InstanceNorm
// ---------------------------------------------------------------------------

/// Instance Normalization: normalizes each (batch, channel) independently.
///
/// Equivalent to GroupNorm with `num_groups == num_channels`.
/// Input shape: `[batch, channels, ...spatial_dims...]`
#[derive(Debug, Clone)]
pub struct InstanceNorm {
    /// Number of channels.
    pub num_channels: usize,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Learnable scale parameter.
    pub gamma: ArrayD<f64>,
    /// Learnable shift parameter.
    pub beta: ArrayD<f64>,
    /// Whether to apply learnable affine transformation.
    pub affine: bool,
}

impl InstanceNorm {
    /// Create a new InstanceNorm layer.
    pub fn new(num_channels: usize, eps: f64, affine: bool) -> Result<Self, NormalizationError> {
        if num_channels == 0 {
            return Err(NormalizationError::EmptyInput);
        }
        let gamma = ArrayD::ones(IxDyn(&[num_channels]));
        let beta = ArrayD::zeros(IxDyn(&[num_channels]));
        Ok(Self {
            num_channels,
            eps,
            gamma,
            beta,
            affine,
        })
    }

    /// Forward pass: normalize each (batch, channel) slice independently.
    ///
    /// Delegates to GroupNorm with `num_groups == num_channels`.
    pub fn forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, NormalizationError> {
        let mut gn = GroupNorm::new(self.num_channels, self.num_channels, self.eps, self.affine)?;
        if self.affine {
            gn.gamma = self.gamma.clone();
            gn.beta = self.beta.clone();
        }
        gn.forward(input)
    }
}

// ---------------------------------------------------------------------------
// BatchNorm
// ---------------------------------------------------------------------------

/// Batch Normalization: normalizes across the batch dimension.
///
/// Tracks running mean/variance for evaluation mode.
/// Input shape: `[batch, channels, ...spatial_dims...]`
#[derive(Debug, Clone)]
pub struct BatchNorm {
    /// Number of channels (features).
    pub num_channels: usize,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Momentum for running statistics update (EMA coefficient).
    pub momentum: f64,
    /// Learnable scale parameter.
    pub gamma: ArrayD<f64>,
    /// Learnable shift parameter.
    pub beta: ArrayD<f64>,
    /// Whether to apply learnable affine transformation.
    pub affine: bool,
    /// Running mean used during evaluation.
    pub running_mean: ArrayD<f64>,
    /// Running variance used during evaluation.
    pub running_var: ArrayD<f64>,
    /// Whether the module is in training mode.
    pub training: bool,
    /// Number of mini-batches tracked.
    pub num_batches_tracked: u64,
}

impl BatchNorm {
    /// Create a new BatchNorm layer.
    pub fn new(
        num_channels: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
    ) -> Result<Self, NormalizationError> {
        if num_channels == 0 {
            return Err(NormalizationError::EmptyInput);
        }
        let gamma = ArrayD::ones(IxDyn(&[num_channels]));
        let beta = ArrayD::zeros(IxDyn(&[num_channels]));
        let running_mean = ArrayD::zeros(IxDyn(&[num_channels]));
        let running_var = ArrayD::ones(IxDyn(&[num_channels]));
        Ok(Self {
            num_channels,
            eps,
            momentum,
            gamma,
            beta,
            affine,
            running_mean,
            running_var,
            training: true,
            num_batches_tracked: 0,
        })
    }

    /// Forward pass: normalize across batch (and spatial) dimensions per channel.
    ///
    /// In training mode: compute batch statistics and update running stats via EMA.
    /// In eval mode: use running statistics.
    pub fn forward(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, NormalizationError> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(NormalizationError::InsufficientDimensions { ndim, required: 2 });
        }
        if input.is_empty() {
            return Err(NormalizationError::EmptyInput);
        }

        let shape = input.shape();
        let channels = shape[1];
        if channels != self.num_channels {
            return Err(NormalizationError::ShapeMismatch {
                expected: vec![shape[0], self.num_channels],
                got: vec![shape[0], channels],
            });
        }

        // Axes to reduce: batch (0) and all spatial dims (2..)
        // We keep the channel axis (1).
        let reduce_axes: Vec<usize> = std::iter::once(0).chain(2..ndim).collect();

        let (mean, var) = if self.training {
            let batch_mean = mean_along_axes(input, &reduce_axes);
            let batch_var = var_along_axes(input, &reduce_axes);

            // Squeeze to [C] for running stat update
            let mean_1d = batch_mean
                .clone()
                .into_shape_with_order(IxDyn(&[channels]))
                .map_err(|_| NormalizationError::ShapeMismatch {
                    expected: vec![channels],
                    got: batch_mean.shape().to_vec(),
                })?;
            let var_1d = batch_var
                .clone()
                .into_shape_with_order(IxDyn(&[channels]))
                .map_err(|_| NormalizationError::ShapeMismatch {
                    expected: vec![channels],
                    got: batch_var.shape().to_vec(),
                })?;

            // EMA update: running = (1 - momentum) * running + momentum * batch
            let mom = self.momentum;
            self.running_mean =
                self.running_mean.mapv(|r| r * (1.0 - mom)) + mean_1d.mapv(|m| m * mom);
            self.running_var =
                self.running_var.mapv(|r| r * (1.0 - mom)) + var_1d.mapv(|v| v * mom);
            self.num_batches_tracked += 1;

            (batch_mean, batch_var)
        } else {
            // Build broadcast shape: [1, C, 1, 1, ...]
            let mut bc_shape = vec![1usize; ndim];
            bc_shape[1] = channels;

            let mean = self
                .running_mean
                .clone()
                .into_shape_with_order(IxDyn(&bc_shape))
                .map_err(|_| NormalizationError::ShapeMismatch {
                    expected: bc_shape.clone(),
                    got: self.running_mean.shape().to_vec(),
                })?;
            let var = self
                .running_var
                .clone()
                .into_shape_with_order(IxDyn(&bc_shape))
                .map_err(|_| NormalizationError::ShapeMismatch {
                    expected: bc_shape.clone(),
                    got: self.running_var.shape().to_vec(),
                })?;
            (mean, var)
        };

        let inv_std = var.mapv(|v| 1.0 / (v + self.eps).sqrt());
        let mut output = (input - &mean) * &inv_std;

        if self.affine {
            let mut bc_shape = vec![1usize; ndim];
            bc_shape[1] = channels;

            let gamma_bc = self
                .gamma
                .clone()
                .into_shape_with_order(IxDyn(&bc_shape))
                .map_err(|_| NormalizationError::ShapeMismatch {
                    expected: bc_shape.clone(),
                    got: self.gamma.shape().to_vec(),
                })?;
            let beta_bc = self
                .beta
                .clone()
                .into_shape_with_order(IxDyn(&bc_shape))
                .map_err(|_| NormalizationError::ShapeMismatch {
                    expected: bc_shape.clone(),
                    got: self.beta.shape().to_vec(),
                })?;
            output = output * &gamma_bc + &beta_bc;
        }

        Ok(output)
    }

    /// Switch to evaluation mode (use running statistics).
    pub fn eval_mode(&mut self) {
        self.training = false;
    }

    /// Switch to training mode (compute batch statistics).
    pub fn train_mode(&mut self) {
        self.training = true;
    }

    /// Check whether the module is in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Reset running statistics to initial values.
    pub fn reset_running_stats(&mut self) {
        self.running_mean = ArrayD::zeros(IxDyn(&[self.num_channels]));
        self.running_var = ArrayD::ones(IxDyn(&[self.num_channels]));
        self.num_batches_tracked = 0;
    }
}

// ---------------------------------------------------------------------------
// WeightNorm
// ---------------------------------------------------------------------------

/// Weight Normalization: reparametrizes weight as `w = g * v / ||v||`.
///
/// This is a weight reparametrization technique, not a layer normalization.
#[derive(Debug, Clone)]
pub struct WeightNorm {
    /// Dimension along which to compute the norm.
    pub dim: usize,
}

impl WeightNorm {
    /// Create a new WeightNorm reparametrization.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Decompose a weight tensor into `(g, v)` where `g = ||w||` per slice
    /// along `self.dim` and `v = w / ||w||`.
    pub fn apply(
        &self,
        weight: &ArrayD<f64>,
    ) -> Result<(ArrayD<f64>, ArrayD<f64>), NormalizationError> {
        let ndim = weight.ndim();
        if ndim == 0 {
            return Err(NormalizationError::EmptyInput);
        }
        if self.dim >= ndim {
            return Err(NormalizationError::InvalidAxis {
                axis: self.dim,
                ndim,
            });
        }

        // Compute ||w|| along all axes except self.dim
        let reduce_axes: Vec<usize> = (0..ndim).filter(|&a| a != self.dim).collect();

        // Compute squared sum along those axes
        let sq = weight.mapv(|x| x * x);
        let mut sum_sq = sq;
        // Reduce in descending order to keep indices valid
        let mut sorted_axes = reduce_axes.clone();
        sorted_axes.sort_unstable();
        sorted_axes.reverse();
        for &ax in &sorted_axes {
            sum_sq = sum_sq.sum_axis(Axis(ax));
        }
        // g shape: [dim_size]
        let g = sum_sq.mapv(f64::sqrt);

        // Build broadcast shape for g: all 1s except self.dim
        let mut bc_shape = vec![1usize; ndim];
        bc_shape[self.dim] = weight.shape()[self.dim];
        let g_bc = g
            .clone()
            .into_shape_with_order(IxDyn(&bc_shape))
            .map_err(|_| NormalizationError::ShapeMismatch {
                expected: bc_shape.clone(),
                got: g.shape().to_vec(),
            })?;

        // v = w / ||w||, avoiding division by zero
        let v = weight / &g_bc.mapv(|val| if val.abs() < 1e-12 { 1e-12 } else { val });

        Ok((g, v))
    }

    /// Reparametrize: given `(g, v)` compute `g * v / ||v||`.
    pub fn reparametrize(
        g: &ArrayD<f64>,
        v: &ArrayD<f64>,
        dim: usize,
    ) -> Result<ArrayD<f64>, NormalizationError> {
        let ndim = v.ndim();
        if ndim == 0 {
            return Err(NormalizationError::EmptyInput);
        }
        if dim >= ndim {
            return Err(NormalizationError::InvalidAxis { axis: dim, ndim });
        }

        // Compute ||v|| along all axes except dim
        let reduce_axes: Vec<usize> = (0..ndim).filter(|&a| a != dim).collect();
        let sq = v.mapv(|x| x * x);
        let mut sum_sq = sq;
        let mut sorted_axes = reduce_axes;
        sorted_axes.sort_unstable();
        sorted_axes.reverse();
        for &ax in &sorted_axes {
            sum_sq = sum_sq.sum_axis(Axis(ax));
        }
        let v_norm = sum_sq.mapv(f64::sqrt);

        // Broadcast g and v_norm to weight shape
        let mut bc_shape = vec![1usize; ndim];
        bc_shape[dim] = v.shape()[dim];

        let g_bc = g
            .clone()
            .into_shape_with_order(IxDyn(&bc_shape))
            .map_err(|_| NormalizationError::ShapeMismatch {
                expected: bc_shape.clone(),
                got: g.shape().to_vec(),
            })?;
        let v_norm_bc = v_norm
            .into_shape_with_order(IxDyn(&bc_shape))
            .map_err(|_| NormalizationError::ShapeMismatch {
                expected: bc_shape.clone(),
                got: vec![],
            })?;

        let v_norm_safe = v_norm_bc.mapv(|val| if val.abs() < 1e-12 { 1e-12 } else { val });
        Ok(v * &g_bc / &v_norm_safe)
    }
}

// ---------------------------------------------------------------------------
// NormStats
// ---------------------------------------------------------------------------

/// Normalization statistics for debugging and monitoring.
#[derive(Debug, Clone)]
pub struct NormStats {
    /// Mean of the input tensor.
    pub input_mean: f64,
    /// Standard deviation of the input tensor.
    pub input_std: f64,
    /// Mean of the output tensor.
    pub output_mean: f64,
    /// Standard deviation of the output tensor.
    pub output_std: f64,
    /// Mean of the gamma (scale) parameter.
    pub gamma_mean: f64,
    /// Mean of the beta (shift) parameter.
    pub beta_mean: f64,
}

impl NormStats {
    /// Compute normalization statistics from input, output, gamma, and beta tensors.
    pub fn compute(
        input: &ArrayD<f64>,
        output: &ArrayD<f64>,
        gamma: &ArrayD<f64>,
        beta: &ArrayD<f64>,
    ) -> Self {
        let input_mean = Self::array_mean(input);
        let input_std = Self::array_std(input, input_mean);
        let output_mean = Self::array_mean(output);
        let output_std = Self::array_std(output, output_mean);
        let gamma_mean = Self::array_mean(gamma);
        let beta_mean = Self::array_mean(beta);

        Self {
            input_mean,
            input_std,
            output_mean,
            output_std,
            gamma_mean,
            beta_mean,
        }
    }

    /// Produce a human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "NormStats {{ input: mean={:.6}, std={:.6} | output: mean={:.6}, std={:.6} | gamma_mean={:.6}, beta_mean={:.6} }}",
            self.input_mean, self.input_std,
            self.output_mean, self.output_std,
            self.gamma_mean, self.beta_mean,
        )
    }

    // -- private helpers --

    fn array_mean(arr: &ArrayD<f64>) -> f64 {
        if arr.is_empty() {
            return 0.0;
        }
        arr.sum() / arr.len() as f64
    }

    fn array_std(arr: &ArrayD<f64>, mean: f64) -> f64 {
        if arr.len() <= 1 {
            return 0.0;
        }
        let var = arr.mapv(|x| (x - mean).powi(2)).sum() / arr.len() as f64;
        var.sqrt()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    fn make_input_4d(batch: usize, channels: usize, h: usize, w: usize) -> ArrayD<f64> {
        let total = batch * channels * h * w;
        ArrayD::from_shape_vec(
            IxDyn(&[batch, channels, h, w]),
            (0..total).map(|i| (i as f64) * 0.01 + 0.1).collect(),
        )
        .expect("test helper: shape matches element count")
    }

    fn make_input_2d(rows: usize, cols: usize) -> ArrayD<f64> {
        let total = rows * cols;
        ArrayD::from_shape_vec(
            IxDyn(&[rows, cols]),
            (0..total).map(|i| (i as f64) * 0.05 + 0.5).collect(),
        )
        .expect("test helper: shape matches element count")
    }

    // -----------------------------------------------------------------------
    // RmsNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rmsnorm_new_valid() {
        let rms = RmsNorm::new(vec![64], 1e-5);
        assert!(rms.is_ok());
        let rms = rms.expect("already checked");
        assert_eq!(rms.normalized_shape, vec![64]);
    }

    #[test]
    fn test_rmsnorm_forward_shape_preserved() {
        let rms = RmsNorm::new(vec![8], 1e-5).expect("valid config");
        let input = make_input_2d(4, 8);
        let output = rms.forward(&input).expect("forward should succeed");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_rmsnorm_output_scale() {
        // After RMSNorm with gamma=1, the RMS of the output along the last dim
        // should be close to 1 for each row.
        let rms = RmsNorm::new(vec![16], 1e-8).expect("valid config");
        let input = ArrayD::from_shape_vec(
            IxDyn(&[2, 16]),
            (0..32).map(|i| (i as f64) * 0.1 + 1.0).collect(),
        )
        .expect("test data");

        let output = rms.forward(&input).expect("forward");
        // Check RMS of each row is close to 1
        for row_idx in 0..2 {
            let mut sum_sq = 0.0;
            for col_idx in 0..16 {
                let v = output[[row_idx, col_idx]];
                sum_sq += v * v;
            }
            let row_rms = (sum_sq / 16.0).sqrt();
            assert!(
                (row_rms - 1.0).abs() < 0.1,
                "RMS should be close to 1, got {}",
                row_rms
            );
        }
    }

    #[test]
    fn test_rmsnorm_update_gamma() {
        let mut rms = RmsNorm::new(vec![4], 1e-5).expect("valid");
        let new_gamma =
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0, 2.0, 2.0, 2.0]).expect("test data");
        assert!(rms.update_gamma(new_gamma).is_ok());
        assert!((rms.gamma[[0]] - 2.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // GroupNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_groupnorm_new_valid() {
        let gn = GroupNorm::new(4, 16, 1e-5, true);
        assert!(gn.is_ok());
    }

    #[test]
    fn test_groupnorm_invalid_groups() {
        let gn = GroupNorm::new(5, 16, 1e-5, true);
        assert!(gn.is_err());
        match gn {
            Err(NormalizationError::InvalidNumGroups { groups, channels }) => {
                assert_eq!(groups, 5);
                assert_eq!(channels, 16);
            }
            _ => panic!("Expected InvalidNumGroups error"),
        }
    }

    #[test]
    fn test_groupnorm_forward_shape_preserved() {
        let gn = GroupNorm::new(4, 8, 1e-5, true).expect("valid");
        let input = make_input_4d(2, 8, 4, 4);
        let output = gn.forward(&input).expect("forward");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_groupnorm_channels_per_group() {
        let gn = GroupNorm::new(4, 16, 1e-5, true).expect("valid");
        assert_eq!(gn.channels_per_group(), 4);
    }

    // -----------------------------------------------------------------------
    // InstanceNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_instancenorm_new_valid() {
        let ins = InstanceNorm::new(8, 1e-5, true);
        assert!(ins.is_ok());
    }

    #[test]
    fn test_instancenorm_forward_shape_preserved() {
        let ins = InstanceNorm::new(4, 1e-5, true).expect("valid");
        let input = make_input_4d(2, 4, 3, 3);
        let output = ins.forward(&input).expect("forward");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_instancenorm_normalizes_per_instance() {
        let ins = InstanceNorm::new(2, 1e-8, false).expect("valid");
        let input = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 4]),
            (0..16).map(|i| (i as f64) * 0.5 + 1.0).collect(),
        )
        .expect("test data");

        let output = ins.forward(&input).expect("forward");
        // Each (batch, channel) slice should have mean close to 0
        for b in 0..2 {
            for c in 0..2 {
                let mut sum = 0.0;
                for s in 0..4 {
                    sum += output[[b, c, s]];
                }
                let slice_mean = sum / 4.0;
                assert!(
                    slice_mean.abs() < 0.01,
                    "Expected ~0 mean, got {} at b={}, c={}",
                    slice_mean,
                    b,
                    c
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // BatchNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batchnorm_new_valid() {
        let bn = BatchNorm::new(16, 1e-5, 0.1, true);
        assert!(bn.is_ok());
        let bn = bn.expect("valid");
        assert!(bn.is_training());
    }

    #[test]
    fn test_batchnorm_forward_training() {
        let mut bn = BatchNorm::new(4, 1e-5, 0.1, true).expect("valid");
        let input = make_input_4d(2, 4, 3, 3);
        let output = bn.forward(&input).expect("forward");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_batchnorm_running_stats_update() {
        let mut bn = BatchNorm::new(4, 1e-5, 0.1, true).expect("valid");
        let initial_mean = bn.running_mean.clone();
        let input = make_input_4d(2, 4, 3, 3);
        let _output = bn.forward(&input).expect("forward");
        // Running mean should have changed
        assert_ne!(bn.running_mean, initial_mean);
        assert_eq!(bn.num_batches_tracked, 1);
    }

    #[test]
    fn test_batchnorm_eval_mode() {
        let mut bn = BatchNorm::new(4, 1e-5, 0.1, true).expect("valid");
        // Train once to populate running stats
        let input = make_input_4d(2, 4, 3, 3);
        let _output = bn.forward(&input).expect("forward training");

        // Switch to eval and run again -- should use running stats
        bn.eval_mode();
        assert!(!bn.is_training());
        let batches_before = bn.num_batches_tracked;
        let output = bn.forward(&input).expect("forward eval");
        assert_eq!(output.shape(), input.shape());
        // num_batches_tracked should not change in eval mode
        assert_eq!(bn.num_batches_tracked, batches_before);
    }

    #[test]
    fn test_batchnorm_train_eval_toggle() {
        let mut bn = BatchNorm::new(4, 1e-5, 0.1, true).expect("valid");
        assert!(bn.is_training());
        bn.eval_mode();
        assert!(!bn.is_training());
        bn.train_mode();
        assert!(bn.is_training());
    }

    // -----------------------------------------------------------------------
    // WeightNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_weightnorm_apply() {
        let wn = WeightNorm::new(0);
        let weight = ArrayD::from_shape_vec(
            IxDyn(&[3, 4]),
            (0..12).map(|i| (i as f64) * 0.1 + 0.1).collect(),
        )
        .expect("test data");

        let (g, v) = wn.apply(&weight).expect("apply");
        // g should have shape [3] (one norm per row)
        assert_eq!(g.shape(), &[3]);
        // v should have same shape as weight
        assert_eq!(v.shape(), weight.shape());
    }

    #[test]
    fn test_weightnorm_reparametrize() {
        let wn = WeightNorm::new(0);
        let weight = ArrayD::from_shape_vec(
            IxDyn(&[3, 4]),
            (0..12).map(|i| (i as f64) * 0.3 + 0.5).collect(),
        )
        .expect("test data");

        let (g, v) = wn.apply(&weight).expect("apply");
        let reconstructed = WeightNorm::reparametrize(&g, &v, 0).expect("reparametrize");

        assert_eq!(reconstructed.shape(), weight.shape());
        // Reconstructed weight should be close to original
        for (orig, recon) in weight.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - recon).abs() < 1e-8,
                "Mismatch: orig={}, recon={}",
                orig,
                recon
            );
        }
    }

    // -----------------------------------------------------------------------
    // NormStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_norm_stats_compute() {
        let input = make_input_2d(4, 8);
        let output = make_input_2d(4, 8);
        let gamma = ArrayD::ones(IxDyn(&[8]));
        let beta = ArrayD::zeros(IxDyn(&[8]));

        let stats = NormStats::compute(&input, &output, &gamma, &beta);
        // All fields should be populated (finite)
        assert!(stats.input_mean.is_finite());
        assert!(stats.input_std.is_finite());
        assert!(stats.output_mean.is_finite());
        assert!(stats.output_std.is_finite());
        assert!(stats.gamma_mean.is_finite());
        assert!(stats.beta_mean.is_finite());
    }

    #[test]
    fn test_norm_stats_summary_nonempty() {
        let input = make_input_2d(2, 4);
        let output = make_input_2d(2, 4);
        let gamma = ArrayD::ones(IxDyn(&[4]));
        let beta = ArrayD::zeros(IxDyn(&[4]));

        let stats = NormStats::compute(&input, &output, &gamma, &beta);
        let summary = stats.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("NormStats"));
    }
}

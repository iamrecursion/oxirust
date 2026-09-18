//! Log-space scoring aggregation and weighted quantifiers.
//!
//! This module provides numerically stable log-space operations for probabilistic
//! inference, along with weighted soft quantifiers (exists/forall) and their
//! gradients for end-to-end training of logical models.
//!
//! ## Log-Space Operations
//!
//! All operations in this module are designed to work in log-probability space
//! to avoid numerical underflow when dealing with very small probabilities.
//!
//! ## Weighted Quantifiers
//!
//! Weighted versions of soft-exists and soft-forall quantifiers that allow
//! assigning importance weights to individual elements before aggregation.

use scirs2_core::ndarray::{Array, ArrayD, Axis, IxDyn};

/// Error type for scoring operations.
#[derive(Debug, thiserror::Error)]
pub enum ScoringError {
    /// Input and weights have incompatible shapes.
    #[error("Shape mismatch: input {input:?}, weights {weights:?}")]
    ShapeMismatch {
        /// Shape of the input tensor.
        input: Vec<usize>,
        /// Shape of the weights tensor.
        weights: Vec<usize>,
    },
    /// Requested axis is out of bounds for the tensor.
    #[error("Axis {axis} out of range for {ndim}D tensor")]
    AxisOutOfRange {
        /// Requested axis.
        axis: usize,
        /// Number of dimensions in the tensor.
        ndim: usize,
    },
    /// All weights sum to zero, cannot normalize.
    #[error("Division by zero in weight normalization")]
    ZeroWeightSum,
    /// A probability value outside [0, 1] was provided.
    #[error("Invalid probability value {value}: must be in [0, 1]")]
    InvalidProbability {
        /// The offending value.
        value: f64,
    },
    /// Reduction attempted on an empty tensor.
    #[error("Empty input tensor")]
    EmptyInput,
}

/// Scoring mode that controls the domain of input values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoringMode {
    /// Standard probability space: values in [0, 1].
    Standard,
    /// Log-probability space: values in (-∞, 0].
    LogProbability,
    /// Log-odds space: values in ℝ (logit scale).
    LogOdds,
}

/// Configuration for scoring operations.
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    /// The domain/mode of the scoring values.
    pub mode: ScoringMode,
    /// Floor value for log-space to avoid -∞.
    /// Default: `f64::MIN_POSITIVE.ln() ≈ -708`.
    pub log_floor: f64,
    /// Temperature parameter for softmax-style operations.
    /// Default: 1.0.
    pub temperature: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            mode: ScoringMode::Standard,
            log_floor: f64::MIN_POSITIVE.ln(), // ≈ -708
            temperature: 1.0,
        }
    }
}

impl ScoringConfig {
    /// Create a log-probability scoring configuration.
    pub fn log_probability() -> Self {
        Self {
            mode: ScoringMode::LogProbability,
            ..Self::default()
        }
    }

    /// Create a log-odds scoring configuration.
    pub fn log_odds() -> Self {
        Self {
            mode: ScoringMode::LogOdds,
            ..Self::default()
        }
    }

    /// Override the temperature parameter (builder pattern).
    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }
}

// ============================================================================
// Internal stable helpers
// ============================================================================

/// Numerically stable log-sum-exp over a flat slice.
///
/// Implements: log Σ exp(x_i) via max subtraction.
fn log_sum_exp_slice(slice: &[f64], log_floor: f64) -> f64 {
    if slice.is_empty() {
        return log_floor;
    }
    let max = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return log_floor;
    }
    let sum_exp: f64 = slice.iter().map(|&x| (x - max).exp()).sum();
    max + sum_exp.ln()
}

/// Compute log-sum-exp along a specific axis, returning the reduced array.
///
/// The output has the reduction axis removed.
fn log_sum_exp_along_axis(
    input: &ArrayD<f64>,
    axis: usize,
    log_floor: f64,
) -> Result<ArrayD<f64>, ScoringError> {
    if axis >= input.ndim() {
        return Err(ScoringError::AxisOutOfRange {
            axis,
            ndim: input.ndim(),
        });
    }
    if input.is_empty() {
        return Err(ScoringError::EmptyInput);
    }
    Ok(input.map_axis(Axis(axis), |lane| {
        let s: Vec<f64> = lane.iter().cloned().collect();
        log_sum_exp_slice(&s, log_floor)
    }))
}

/// Compute log-product (sum) along a specific axis, returning the reduced array.
fn log_product_along_axis(input: &ArrayD<f64>, axis: usize) -> Result<ArrayD<f64>, ScoringError> {
    if axis >= input.ndim() {
        return Err(ScoringError::AxisOutOfRange {
            axis,
            ndim: input.ndim(),
        });
    }
    if input.is_empty() {
        return Err(ScoringError::EmptyInput);
    }
    Ok(input.map_axis(Axis(axis), |lane| lane.iter().sum::<f64>()))
}

// ============================================================================
// LogSpaceAggregator
// ============================================================================

/// Numerically stable aggregation operations in log-probability space.
///
/// All reduction operations are implemented with maximum-subtraction tricks
/// to avoid overflow and underflow when working with log-probabilities.
pub struct LogSpaceAggregator {
    config: ScoringConfig,
}

impl LogSpaceAggregator {
    /// Create a new aggregator with the given configuration.
    pub fn new(config: ScoringConfig) -> Self {
        Self { config }
    }

    /// Compute log-sum-exp: `log Σ_i exp(x_i)`.
    ///
    /// Numerically stable via max subtraction:
    /// `log Σ exp(x_i) = max + log Σ exp(x_i - max)`.
    ///
    /// # Arguments
    /// * `input` - Input tensor (values in any domain).
    /// * `axis` - `None` → reduce all elements; `Some(k)` → reduce along axis `k`.
    ///
    /// # Returns
    /// Reduced tensor.  For `axis=None` this is a 0-D scalar tensor.
    pub fn log_sum_exp(
        &self,
        input: &ArrayD<f64>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        if input.is_empty() {
            return Err(ScoringError::EmptyInput);
        }
        match axis {
            None => {
                let flat: Vec<f64> = input.iter().cloned().collect();
                let result = log_sum_exp_slice(&flat, self.config.log_floor);
                Ok(ArrayD::from_elem(IxDyn(&[]), result))
            }
            Some(ax) => log_sum_exp_along_axis(input, ax, self.config.log_floor),
        }
    }

    /// Compute log-product: `Σ_i x_i` (addition in log space = product in probability space).
    ///
    /// # Arguments
    /// * `input` - Input tensor (values assumed to be in log-probability space).
    /// * `axis` - `None` → sum all; `Some(k)` → sum along axis `k`.
    pub fn log_product(
        &self,
        input: &ArrayD<f64>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        if input.is_empty() {
            return Err(ScoringError::EmptyInput);
        }
        match axis {
            None => {
                let result: f64 = input.iter().sum();
                // Clamp to log_floor
                let result = result.max(self.config.log_floor);
                Ok(ArrayD::from_elem(IxDyn(&[]), result))
            }
            Some(ax) => {
                let out = log_product_along_axis(input, ax)?;
                Ok(out.mapv(|v| v.max(self.config.log_floor)))
            }
        }
    }

    /// Element-wise binary log-add-exp: `log(exp(a) + exp(b))`.
    ///
    /// Numerically stable: `max + log(1 + exp(min - max))`.
    ///
    /// # Arguments
    /// * `a` - First operand (must have same shape as `b`).
    /// * `b` - Second operand.
    pub fn log_add_exp(
        &self,
        a: &ArrayD<f64>,
        b: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        if a.shape() != b.shape() {
            return Err(ScoringError::ShapeMismatch {
                input: a.shape().to_vec(),
                weights: b.shape().to_vec(),
            });
        }
        // log_add_exp(a, b) = max(a,b) + log1p(exp(-|a-b|))
        let result = a.mapv(|_| 0.0_f64); // same shape placeholder
        let result = scirs2_core::ndarray::Zip::from(&result)
            .and(a)
            .and(b)
            .map_collect(|_, &ai, &bi| {
                let max = ai.max(bi);
                let min = ai.min(bi);
                if max == f64::NEG_INFINITY {
                    self.config.log_floor
                } else {
                    max + (1.0_f64 + (min - max).exp()).ln()
                }
            });
        Ok(result)
    }

    /// Convert probabilities to log-space, clamping to `log_floor`.
    ///
    /// # Arguments
    /// * `probs` - Input probabilities (must be in [0, 1]).
    pub fn to_log_space(&self, probs: &ArrayD<f64>) -> Result<ArrayD<f64>, ScoringError> {
        // Validate that all values are in [0, 1]
        for &v in probs.iter() {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(ScoringError::InvalidProbability { value: v });
            }
        }
        let floor = self.config.log_floor;
        Ok(probs.mapv(|p| if p <= 0.0 { floor } else { p.ln().max(floor) }))
    }

    /// Convert log-probabilities back to probability space via `exp`.
    ///
    /// # Arguments
    /// * `log_probs` - Input log-probabilities (values in (-∞, 0]).
    pub fn from_log_space(&self, log_probs: &ArrayD<f64>) -> Result<ArrayD<f64>, ScoringError> {
        Ok(log_probs.mapv(|lp| lp.exp()))
    }
}

// ============================================================================
// WeightedQuantifier
// ============================================================================

/// Validate that input and weights share compatible shapes for a given axis.
///
/// Returns `Ok(weight_sum)` – the total sum of weights – so callers can
/// check for zero-sum and avoid a second pass.
fn validate_weights_for_axis(
    input: &ArrayD<f64>,
    weights: &ArrayD<f64>,
    axis: Option<usize>,
) -> Result<(), ScoringError> {
    match axis {
        None => {
            // Weights must be 1-D with length = total number of elements,
            // OR have the same shape as input.
            if weights.shape() != input.shape() && weights.len() != input.len() {
                return Err(ScoringError::ShapeMismatch {
                    input: input.shape().to_vec(),
                    weights: weights.shape().to_vec(),
                });
            }
        }
        Some(ax) => {
            if ax >= input.ndim() {
                return Err(ScoringError::AxisOutOfRange {
                    axis: ax,
                    ndim: input.ndim(),
                });
            }
            let expected_len = input.shape()[ax];
            // Weights should either match input shape exactly, or be 1-D with length = axis_size.
            let compatible = weights.shape() == input.shape()
                || (weights.ndim() == 1 && weights.len() == expected_len);
            if !compatible {
                return Err(ScoringError::ShapeMismatch {
                    input: input.shape().to_vec(),
                    weights: weights.shape().to_vec(),
                });
            }
        }
    }
    Ok(())
}

/// Weighted soft-quantifier operations.
///
/// Provides differentiable, weight-aware approximations of logical quantifiers
/// that can be used in end-to-end training pipelines.
pub struct WeightedQuantifier {
    config: ScoringConfig,
}

impl WeightedQuantifier {
    /// Create a new quantifier with the given configuration.
    pub fn new(config: ScoringConfig) -> Self {
        Self { config }
    }

    /// Soft-exists: weighted mean (standard) or log-sum-exp with log-weights (log mode).
    ///
    /// **Standard mode**: `Σ w_i * x_i / Σ w_i`
    ///
    /// **LogProbability / LogOdds mode**: `log-sum-exp(log(w) + x) - log(Σ w_i)`
    ///
    /// # Arguments
    /// * `input`   - Input values.
    /// * `weights` - Non-negative importance weights (must broadcast with input along `axis`).
    /// * `axis`    - `None` → over all elements; `Some(k)` → along axis `k`.
    pub fn weighted_exists(
        &self,
        input: &ArrayD<f64>,
        weights: &ArrayD<f64>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        if input.is_empty() {
            return Err(ScoringError::EmptyInput);
        }
        validate_weights_for_axis(input, weights, axis)?;

        match self.config.mode {
            ScoringMode::Standard => self.weighted_exists_standard(input, weights, axis),
            ScoringMode::LogProbability | ScoringMode::LogOdds => {
                self.weighted_exists_log(input, weights, axis)
            }
        }
    }

    fn weighted_exists_standard(
        &self,
        input: &ArrayD<f64>,
        weights: &ArrayD<f64>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        // Broadcast weights to input shape if needed
        let w = broadcast_weights(weights, input, axis)?;

        let weight_sum: f64 = w.iter().sum();
        if weight_sum == 0.0 {
            return Err(ScoringError::ZeroWeightSum);
        }

        match axis {
            None => {
                let numerator: f64 = input.iter().zip(w.iter()).map(|(&x, &wi)| wi * x).sum();
                let result = numerator / weight_sum;
                Ok(ArrayD::from_elem(IxDyn(&[]), result))
            }
            Some(ax) => {
                let weighted = input * &w;
                let num = weighted.sum_axis(Axis(ax));
                // Per-slice weight sums
                let w_sum = w.sum_axis(Axis(ax));
                // Avoid division by zero per element
                let result = scirs2_core::ndarray::Zip::from(&num)
                    .and(&w_sum)
                    .map_collect(|&n, &ws| if ws == 0.0 { 0.0 } else { n / ws });
                Ok(result)
            }
        }
    }

    fn weighted_exists_log(
        &self,
        input: &ArrayD<f64>,
        weights: &ArrayD<f64>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        // Broadcast weights to input shape
        let w = broadcast_weights(weights, input, axis)?;
        let weight_sum: f64 = w.iter().sum();
        if weight_sum == 0.0 {
            return Err(ScoringError::ZeroWeightSum);
        }
        let log_norm = weight_sum.ln();
        let floor = self.config.log_floor;

        // log(w_i) + x_i, then log-sum-exp, minus log(Σ w_i)
        let log_w_plus_x =
            scirs2_core::ndarray::Zip::from(&w)
                .and(input)
                .map_collect(|&wi, &xi| {
                    if wi <= 0.0 {
                        floor
                    } else {
                        (wi.ln() + xi).max(floor)
                    }
                });

        let agg = LogSpaceAggregator::new(self.config.clone());
        let lse = agg.log_sum_exp(&log_w_plus_x, axis)?;
        Ok(lse.mapv(|v| v - log_norm))
    }

    /// Soft-forall: weighted geometric mean (standard) or log-mean (log mode).
    ///
    /// **Standard mode**: `∏ x_i^(w_i/Σw_i)` = geometric weighted mean.
    ///
    /// **LogProbability / LogOdds mode**: `Σ (w_i/Σw_i) * x_i` (weighted arithmetic mean in log space).
    ///
    /// # Arguments
    /// * `input`   - Input values.
    /// * `weights` - Non-negative importance weights.
    /// * `axis`    - Reduction axis.
    pub fn weighted_forall(
        &self,
        input: &ArrayD<f64>,
        weights: &ArrayD<f64>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        if input.is_empty() {
            return Err(ScoringError::EmptyInput);
        }
        validate_weights_for_axis(input, weights, axis)?;

        match self.config.mode {
            ScoringMode::Standard => self.weighted_forall_standard(input, weights, axis),
            ScoringMode::LogProbability | ScoringMode::LogOdds => {
                self.weighted_forall_log(input, weights, axis)
            }
        }
    }

    fn weighted_forall_standard(
        &self,
        input: &ArrayD<f64>,
        weights: &ArrayD<f64>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        let w = broadcast_weights(weights, input, axis)?;
        let weight_sum: f64 = w.iter().sum();
        if weight_sum == 0.0 {
            return Err(ScoringError::ZeroWeightSum);
        }

        // Compute log(x_i) * (w_i / Σw), sum along axis, then exp
        let log_input = input.mapv(|x| {
            if x <= 0.0 {
                self.config.log_floor
            } else {
                x.ln()
            }
        });

        match axis {
            None => {
                let log_geo: f64 = log_input
                    .iter()
                    .zip(w.iter())
                    .map(|(&lx, &wi)| lx * wi / weight_sum)
                    .sum();
                Ok(ArrayD::from_elem(IxDyn(&[]), log_geo.exp()))
            }
            Some(ax) => {
                let w_sum_ax = w.sum_axis(Axis(ax));
                let weighted_log = &log_input * &w;
                let num = weighted_log.sum_axis(Axis(ax));
                let result = scirs2_core::ndarray::Zip::from(&num)
                    .and(&w_sum_ax)
                    .map_collect(|&n, &ws| {
                        if ws == 0.0 {
                            1.0 // neutral element for geometric mean
                        } else {
                            (n / ws).exp()
                        }
                    });
                Ok(result)
            }
        }
    }

    fn weighted_forall_log(
        &self,
        input: &ArrayD<f64>,
        weights: &ArrayD<f64>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        // In log space, forall is weighted arithmetic mean of log values.
        let w = broadcast_weights(weights, input, axis)?;
        let weight_sum: f64 = w.iter().sum();
        if weight_sum == 0.0 {
            return Err(ScoringError::ZeroWeightSum);
        }

        match axis {
            None => {
                let result: f64 = input
                    .iter()
                    .zip(w.iter())
                    .map(|(&xi, &wi)| xi * wi / weight_sum)
                    .sum();
                Ok(ArrayD::from_elem(IxDyn(&[]), result))
            }
            Some(ax) => {
                let w_sum_ax = w.sum_axis(Axis(ax));
                let weighted = input * &w;
                let num = weighted.sum_axis(Axis(ax));
                let result = scirs2_core::ndarray::Zip::from(&num)
                    .and(&w_sum_ax)
                    .map_collect(|&n, &ws| if ws == 0.0 { 0.0 } else { n / ws });
                Ok(result)
            }
        }
    }

    /// Gradient of [`WeightedQuantifier::weighted_exists`] with respect to input.
    ///
    /// **Standard mode**: `∂/∂x_i (Σ w_j x_j / Σ w_j) = w_i / Σ w_j`
    ///
    /// # Arguments
    /// * `grad`    - Upstream gradient (same shape as the forward output).
    /// * `input`   - Forward input (used for shape in log mode).
    /// * `weights` - Same weights used in the forward pass.
    /// * `axis`    - Same axis used in the forward pass.
    pub fn weighted_exists_grad(
        &self,
        grad: &ArrayD<f64>,
        input: &ArrayD<f64>,
        weights: &ArrayD<f64>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        if input.is_empty() {
            return Err(ScoringError::EmptyInput);
        }
        validate_weights_for_axis(input, weights, axis)?;

        let w = broadcast_weights(weights, input, axis)?;
        let weight_sum: f64 = w.iter().sum();
        if weight_sum == 0.0 {
            return Err(ScoringError::ZeroWeightSum);
        }

        // Normalized weights: w_i / Σ w
        let w_norm = w.mapv(|wi| wi / weight_sum);

        match axis {
            None => {
                // grad is scalar (0-D), broadcast to input shape
                let g_scalar = grad.iter().next().copied().unwrap_or(0.0);
                Ok(w_norm.mapv(|wn| wn * g_scalar))
            }
            Some(ax) => {
                // grad has axis `ax` removed; need to reinsert for broadcasting
                let grad_expanded = grad.view().insert_axis(Axis(ax));
                Ok(&w_norm * &grad_expanded)
            }
        }
    }

    /// Gradient of [`WeightedQuantifier::weighted_forall`] with respect to input.
    ///
    /// **Standard mode**: geometric mean gradient.
    /// `∂/∂x_i ∏ x_j^(w_j/W) = (w_i/W) * ∏ x_j^(w_j/W) / x_i`
    ///
    /// # Arguments
    /// * `grad`    - Upstream gradient (same shape as the forward output).
    /// * `input`   - Forward input (needed for ∏ x^w computation).
    /// * `weights` - Same weights used in the forward pass.
    /// * `axis`    - Same axis used in the forward pass.
    pub fn weighted_forall_grad(
        &self,
        grad: &ArrayD<f64>,
        input: &ArrayD<f64>,
        weights: &ArrayD<f64>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f64>, ScoringError> {
        if input.is_empty() {
            return Err(ScoringError::EmptyInput);
        }
        validate_weights_for_axis(input, weights, axis)?;

        let w = broadcast_weights(weights, input, axis)?;
        let weight_sum: f64 = w.iter().sum();
        if weight_sum == 0.0 {
            return Err(ScoringError::ZeroWeightSum);
        }

        match self.config.mode {
            ScoringMode::Standard => {
                // Geometric mean gradient:
                // ∂out/∂x_i = (w_i/W) * out / x_i
                // out = ∏ x_j^(w_j/W)
                let log_input = input.mapv(|x| {
                    if x <= 0.0 {
                        self.config.log_floor
                    } else {
                        x.ln()
                    }
                });

                let forall_out = match axis {
                    None => {
                        let log_geo: f64 = log_input
                            .iter()
                            .zip(w.iter())
                            .map(|(&lx, &wi)| lx * wi / weight_sum)
                            .sum();
                        ArrayD::from_elem(input.raw_dim(), log_geo.exp())
                    }
                    Some(ax) => {
                        let w_sum_ax = w.sum_axis(Axis(ax));
                        let weighted_log = &log_input * &w;
                        let num = weighted_log.sum_axis(Axis(ax));
                        let out_no_ax = scirs2_core::ndarray::Zip::from(&num)
                            .and(&w_sum_ax)
                            .map_collect(|&n, &ws| if ws == 0.0 { 1.0 } else { (n / ws).exp() });
                        // broadcast back
                        out_no_ax
                            .insert_axis(Axis(ax))
                            .broadcast(input.raw_dim())
                            .map_or_else(|| Array::zeros(input.raw_dim()), |v| v.to_owned())
                    }
                };

                // ∂out/∂x_i = (w_i/W) * out / x_i
                let w_norm = w.mapv(|wi| wi / weight_sum);
                let scale = scirs2_core::ndarray::Zip::from(&w_norm)
                    .and(&forall_out)
                    .and(input)
                    .map_collect(
                        |&wn, &out_v, &xi| {
                            if xi == 0.0 {
                                0.0
                            } else {
                                wn * out_v / xi
                            }
                        },
                    );

                match axis {
                    None => {
                        let g_scalar = grad.iter().next().copied().unwrap_or(0.0);
                        Ok(scale.mapv(|s| s * g_scalar))
                    }
                    Some(ax) => {
                        let grad_expanded = grad.view().insert_axis(Axis(ax));
                        Ok(&scale * &grad_expanded)
                    }
                }
            }
            ScoringMode::LogProbability | ScoringMode::LogOdds => {
                // In log mode forall is weighted mean: ∂/∂x_i = w_i/W
                let w_norm = w.mapv(|wi| wi / weight_sum);
                match axis {
                    None => {
                        let g_scalar = grad.iter().next().copied().unwrap_or(0.0);
                        Ok(w_norm.mapv(|wn| wn * g_scalar))
                    }
                    Some(ax) => {
                        let grad_expanded = grad.view().insert_axis(Axis(ax));
                        Ok(&w_norm * &grad_expanded)
                    }
                }
            }
        }
    }
}

// ============================================================================
// Broadcast helpers
// ============================================================================

/// Broadcast `weights` to the full shape of `input`.
///
/// Supported cases:
/// - `weights.shape() == input.shape()` → clone
/// - 1-D weights along `axis` → expand dimensions
/// - flat weights with same length as input → reshape
fn broadcast_weights(
    weights: &ArrayD<f64>,
    input: &ArrayD<f64>,
    axis: Option<usize>,
) -> Result<ArrayD<f64>, ScoringError> {
    if weights.shape() == input.shape() {
        return Ok(weights.clone());
    }

    match axis {
        None => {
            // Flat weights: must have same total length as input
            if weights.len() != input.len() {
                return Err(ScoringError::ShapeMismatch {
                    input: input.shape().to_vec(),
                    weights: weights.shape().to_vec(),
                });
            }
            // Reshape to input shape
            weights
                .clone()
                .into_shape_with_order(input.raw_dim())
                .map_err(|_| ScoringError::ShapeMismatch {
                    input: input.shape().to_vec(),
                    weights: weights.shape().to_vec(),
                })
        }
        Some(ax) => {
            if weights.ndim() == 1 && weights.len() == input.shape()[ax] {
                // Build broadcast shape: 1 everywhere except `ax`
                let mut shape = vec![1usize; input.ndim()];
                shape[ax] = input.shape()[ax];
                let reshaped = weights
                    .clone()
                    .into_shape_with_order(IxDyn(&shape))
                    .map_err(|_| ScoringError::ShapeMismatch {
                        input: input.shape().to_vec(),
                        weights: weights.shape().to_vec(),
                    })?;
                reshaped
                    .broadcast(input.raw_dim())
                    .map(|v| v.to_owned())
                    .ok_or_else(|| ScoringError::ShapeMismatch {
                        input: input.shape().to_vec(),
                        weights: weights.shape().to_vec(),
                    })
            } else if weights.shape() == input.shape() {
                Ok(weights.clone())
            } else {
                Err(ScoringError::ShapeMismatch {
                    input: input.shape().to_vec(),
                    weights: weights.shape().to_vec(),
                })
            }
        }
    }
}

// ============================================================================
// Free functions (gradient_ops.rs style)
// ============================================================================

/// Compute log-sum-exp with an explicit `ScoringConfig`.
///
/// Convenience wrapper around [`LogSpaceAggregator::log_sum_exp`].
pub fn log_sum_exp(
    input: &ArrayD<f64>,
    axis: Option<usize>,
    config: ScoringConfig,
) -> Result<ArrayD<f64>, ScoringError> {
    LogSpaceAggregator::new(config).log_sum_exp(input, axis)
}

/// Compute weighted soft-exists with an explicit `ScoringConfig`.
///
/// Convenience wrapper around [`WeightedQuantifier::weighted_exists`].
pub fn weighted_soft_exists(
    input: &ArrayD<f64>,
    weights: &ArrayD<f64>,
    axis: Option<usize>,
    config: ScoringConfig,
) -> Result<ArrayD<f64>, ScoringError> {
    WeightedQuantifier::new(config).weighted_exists(input, weights, axis)
}

/// Compute weighted soft-forall with an explicit `ScoringConfig`.
///
/// Convenience wrapper around [`WeightedQuantifier::weighted_forall`].
pub fn weighted_soft_forall(
    input: &ArrayD<f64>,
    weights: &ArrayD<f64>,
    axis: Option<usize>,
    config: ScoringConfig,
) -> Result<ArrayD<f64>, ScoringError> {
    WeightedQuantifier::new(config).weighted_forall(input, weights, axis)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    const EPS: f64 = 1e-9;

    fn config() -> ScoringConfig {
        ScoringConfig::default()
    }

    fn agg() -> LogSpaceAggregator {
        LogSpaceAggregator::new(config())
    }

    fn make_1d(data: Vec<f64>) -> ArrayD<f64> {
        Array::from_vec(data).into_dyn()
    }

    fn make_2d(data: Vec<Vec<f64>>) -> ArrayD<f64> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        Array2::from_shape_vec((rows, cols), flat)
            .expect("valid shape")
            .into_dyn()
    }

    // -------------------------------------------------------------------------
    // 1. test_log_sum_exp_scalar
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_sum_exp_scalar() {
        let input = make_1d(vec![3.0]);
        let result = agg().log_sum_exp(&input, None).expect("log_sum_exp scalar");
        // log(exp(3)) == 3
        assert!(
            (result[[]] - 3.0).abs() < EPS,
            "expected 3.0, got {}",
            result[[]]
        );
    }

    // -------------------------------------------------------------------------
    // 2. test_log_sum_exp_zeros
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_sum_exp_zeros() {
        // log-sum-exp([0,0,0,0]) = log(4)
        let n = 4usize;
        let input = make_1d(vec![0.0; n]);
        let result = agg().log_sum_exp(&input, None).expect("log_sum_exp zeros");
        let expected = (n as f64).ln();
        assert!(
            (result[[]] - expected).abs() < EPS,
            "expected log({}), got {}",
            n,
            result[[]]
        );
    }

    // -------------------------------------------------------------------------
    // 3. test_log_sum_exp_vs_naive
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_sum_exp_vs_naive() {
        let vals = vec![1.0, 2.0, 3.0];
        let input = make_1d(vals.clone());
        let result = agg().log_sum_exp(&input, None).expect("vs naive");
        let naive = vals.iter().map(|&x| x.exp()).sum::<f64>().ln();
        assert!(
            (result[[]] - naive).abs() < 1e-10,
            "stable != naive: {} vs {}",
            result[[]],
            naive
        );
    }

    // -------------------------------------------------------------------------
    // 4. test_log_sum_exp_numerical_stability
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_sum_exp_numerical_stability() {
        // Naive exp(300) overflows; stable version should not.
        let input = make_1d(vec![300.0, 299.0, 298.0]);
        let result = agg()
            .log_sum_exp(&input, None)
            .expect("numerical stability");
        assert!(
            result[[]].is_finite(),
            "result should be finite, got {}",
            result[[]]
        );
        // Should be close to 300 + log(1 + exp(-1) + exp(-2))
        let expected = 300.0 + (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp()).ln();
        assert!((result[[]] - expected).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // 5. test_log_sum_exp_axis_0
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_sum_exp_axis_0() {
        // 2x3 matrix, reduce along axis 0 → shape [3]
        let input = make_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let result = agg().log_sum_exp(&input, Some(0)).expect("axis 0");
        assert_eq!(result.shape(), &[3]);
        for col in 0..3 {
            let a = (col + 1) as f64;
            let b = (col + 4) as f64;
            let expected = a.max(b) + (1.0 + (a.min(b) - a.max(b)).exp()).ln();
            assert!((result[[col]] - expected).abs() < 1e-10);
        }
    }

    // -------------------------------------------------------------------------
    // 6. test_log_sum_exp_axis_1
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_sum_exp_axis_1() {
        // 2x3 matrix, reduce along axis 1 → shape [2]
        let input = make_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let result = agg().log_sum_exp(&input, Some(1)).expect("axis 1");
        assert_eq!(result.shape(), &[2]);
        for row in 0..2 {
            let vals: Vec<f64> = (1..=3).map(|c| (row * 3 + c) as f64).collect();
            let expected_v = vals.iter().map(|&v| v.exp()).sum::<f64>().ln();
            assert!(
                (result[[row]] - expected_v).abs() < 1e-8,
                "row {}: {} vs {}",
                row,
                result[[row]],
                expected_v
            );
        }
    }

    // -------------------------------------------------------------------------
    // 7. test_log_sum_exp_full_reduction
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_sum_exp_full_reduction() {
        let input = make_2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = agg().log_sum_exp(&input, None).expect("full reduction");
        assert_eq!(result.shape(), &[] as &[usize]);
        let naive = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp() + 4.0_f64.exp()).ln();
        assert!((result[[]] - naive).abs() < 1e-8);
    }

    // -------------------------------------------------------------------------
    // 8. test_log_product_basic
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_product_basic() {
        // log(0.5) + log(0.25) = log(0.125)
        let input = make_1d(vec![0.5_f64.ln(), 0.25_f64.ln()]);
        let result = agg().log_product(&input, None).expect("log_product basic");
        let expected = 0.125_f64.ln();
        assert!((result[[]] - expected).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // 9. test_log_add_exp_symmetry
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_add_exp_symmetry() {
        let a = make_1d(vec![1.0, 2.0, 3.0]);
        let b = make_1d(vec![3.0, 1.0, 2.0]);
        let ab = agg().log_add_exp(&a, &b).expect("log_add_exp ab");
        let ba = agg().log_add_exp(&b, &a).expect("log_add_exp ba");
        for i in 0..3 {
            assert!(
                (ab[[i]] - ba[[i]]).abs() < EPS,
                "symmetry violated at {}",
                i
            );
        }
    }

    // -------------------------------------------------------------------------
    // 10. test_to_log_space_range
    // -------------------------------------------------------------------------
    #[test]
    fn test_to_log_space_range() {
        let probs = make_1d(vec![0.0, 0.1, 0.5, 0.9, 1.0]);
        let result = agg().to_log_space(&probs).expect("to_log_space");
        for &v in result.iter() {
            assert!(v <= 0.0, "log-probability must be <= 0, got {}", v);
        }
    }

    // -------------------------------------------------------------------------
    // 11. test_from_log_space_roundtrip
    // -------------------------------------------------------------------------
    #[test]
    fn test_from_log_space_roundtrip() {
        let probs = make_1d(vec![0.1, 0.5, 0.9]);
        let log_p = agg().to_log_space(&probs).expect("to_log_space");
        let recovered = agg().from_log_space(&log_p).expect("from_log_space");
        for i in 0..3 {
            assert!(
                (probs[[i]] - recovered[[i]]).abs() < 1e-12,
                "roundtrip failed at {}: {} != {}",
                i,
                probs[[i]],
                recovered[[i]]
            );
        }
    }

    // -------------------------------------------------------------------------
    // 12. test_log_floor_prevents_neg_inf
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_floor_prevents_neg_inf() {
        let probs = make_1d(vec![0.0, 0.5, 1.0]); // p=0 would give -inf
        let result = agg().to_log_space(&probs).expect("log_floor");
        for &v in result.iter() {
            assert!(v.is_finite(), "value should be finite, got {}", v);
        }
        assert!(result[[0]] <= 0.0, "floor should be <= 0");
    }

    // -------------------------------------------------------------------------
    // 13. test_weighted_exists_uniform_weights
    // -------------------------------------------------------------------------
    #[test]
    fn test_weighted_exists_uniform_weights() {
        // Uniform weights → weighted mean = simple mean
        let input = make_1d(vec![0.2, 0.4, 0.6, 0.8]);
        let weights = make_1d(vec![1.0, 1.0, 1.0, 1.0]);
        let q = WeightedQuantifier::new(config());
        let result = q
            .weighted_exists(&input, &weights, None)
            .expect("uniform weights");
        let expected = 0.5; // (0.2+0.4+0.6+0.8)/4
        assert!(
            (result[[]] - expected).abs() < EPS,
            "expected {}, got {}",
            expected,
            result[[]]
        );
    }

    // -------------------------------------------------------------------------
    // 14. test_weighted_exists_zero_weight_error
    // -------------------------------------------------------------------------
    #[test]
    fn test_weighted_exists_zero_weight_error() {
        let input = make_1d(vec![0.5, 0.5]);
        let weights = make_1d(vec![0.0, 0.0]);
        let q = WeightedQuantifier::new(config());
        let result = q.weighted_exists(&input, &weights, None);
        assert!(
            matches!(result, Err(ScoringError::ZeroWeightSum)),
            "expected ZeroWeightSum error"
        );
    }

    // -------------------------------------------------------------------------
    // 15. test_weighted_exists_concentrated_weight
    // -------------------------------------------------------------------------
    #[test]
    fn test_weighted_exists_concentrated_weight() {
        // All weight on the third element → result ≈ x[2]
        let input = make_1d(vec![0.1, 0.3, 0.7, 0.9]);
        let weights = make_1d(vec![0.0, 0.0, 1.0, 0.0]);
        let q = WeightedQuantifier::new(config());
        let result = q
            .weighted_exists(&input, &weights, None)
            .expect("concentrated weight");
        assert!(
            (result[[]] - 0.7).abs() < EPS,
            "expected 0.7, got {}",
            result[[]]
        );
    }

    // -------------------------------------------------------------------------
    // 16. test_weighted_forall_uniform
    // -------------------------------------------------------------------------
    #[test]
    fn test_weighted_forall_uniform() {
        // Uniform weights → geometric mean
        let vals = vec![0.5, 0.25, 1.0, 0.5];
        let input = make_1d(vals.clone());
        let weights = make_1d(vec![1.0; 4]);
        let q = WeightedQuantifier::new(config());
        let result = q
            .weighted_forall(&input, &weights, None)
            .expect("forall uniform");
        // geometric mean = (0.5 * 0.25 * 1.0 * 0.5)^(1/4)
        let geo: f64 = vals.iter().product::<f64>().powf(0.25);
        assert!(
            (result[[]] - geo).abs() < 1e-10,
            "expected {}, got {}",
            geo,
            result[[]]
        );
    }

    // -------------------------------------------------------------------------
    // 17. test_weighted_exists_gradient_shape
    // -------------------------------------------------------------------------
    #[test]
    fn test_weighted_exists_gradient_shape() {
        let input = make_2d(vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]);
        let weights = make_2d(vec![vec![1.0, 2.0, 1.0], vec![1.0, 2.0, 1.0]]);
        let q = WeightedQuantifier::new(config());
        // Forward along axis 1 → shape [2]
        let out = q
            .weighted_exists(&input, &weights, Some(1))
            .expect("forward");
        assert_eq!(out.shape(), &[2]);
        let grad = Array::ones(out.raw_dim());
        let d_input = q
            .weighted_exists_grad(&grad, &input, &weights, Some(1))
            .expect("grad");
        assert_eq!(
            d_input.shape(),
            input.shape(),
            "gradient should match input shape"
        );
    }

    // -------------------------------------------------------------------------
    // 18. test_weighted_exists_gradient_finite
    // -------------------------------------------------------------------------
    #[test]
    fn test_weighted_exists_gradient_finite() {
        let input = make_1d(vec![0.2, 0.5, 0.8]);
        let weights = make_1d(vec![1.0, 3.0, 1.0]);
        let q = WeightedQuantifier::new(config());
        let out = q.weighted_exists(&input, &weights, None).expect("forward");
        let grad = Array::ones(out.raw_dim());
        let d_input = q
            .weighted_exists_grad(&grad, &input, &weights, None)
            .expect("grad");
        for &v in d_input.iter() {
            assert!(v.is_finite(), "gradient must be finite, got {}", v);
        }
    }

    // -------------------------------------------------------------------------
    // 19. test_scoring_config_default
    // -------------------------------------------------------------------------
    #[test]
    fn test_scoring_config_default() {
        let cfg = ScoringConfig::default();
        assert_eq!(cfg.mode, ScoringMode::Standard);
        assert!((cfg.temperature - 1.0).abs() < EPS);
        assert!(cfg.log_floor < -100.0, "log_floor should be very negative");
        assert!(cfg.log_floor.is_finite(), "log_floor must be finite");
    }

    // -------------------------------------------------------------------------
    // 20. test_scoring_config_builders
    // -------------------------------------------------------------------------
    #[test]
    fn test_scoring_config_builders() {
        let lp = ScoringConfig::log_probability();
        assert_eq!(lp.mode, ScoringMode::LogProbability);

        let lo = ScoringConfig::log_odds();
        assert_eq!(lo.mode, ScoringMode::LogOdds);

        let with_t = ScoringConfig::default().with_temperature(0.5);
        assert!((with_t.temperature - 0.5).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // 21. test_free_function_log_sum_exp
    // -------------------------------------------------------------------------
    #[test]
    fn test_free_function_log_sum_exp() {
        let input = make_1d(vec![0.0, 0.0, 0.0]);
        let result = log_sum_exp(&input, None, config()).expect("free fn log_sum_exp");
        let expected = (3.0_f64).ln();
        assert!((result[[]] - expected).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // 22. test_log_space_quantifier_mode_via_gradient_ops
    // -------------------------------------------------------------------------
    #[test]
    fn test_log_space_quantifier_mode_via_gradient_ops() {
        use crate::gradient_ops::{soft_exists, QuantifierMode};

        // LogSpace mode should delegate to LogSpaceAggregator::log_sum_exp
        let input = make_1d(vec![0.0, 0.0, 0.0]);
        let scoring_cfg = ScoringConfig::log_probability();
        let mode = QuantifierMode::LogSpace(scoring_cfg);
        let result = soft_exists(&input, None, mode).expect("log_space quantifier");
        let expected = (3.0_f64).ln(); // log-sum-exp([0,0,0]) = log(3)
        assert!(
            (result[[]] - expected).abs() < 1e-10,
            "expected log(3)={}, got {}",
            expected,
            result[[]]
        );
    }
}

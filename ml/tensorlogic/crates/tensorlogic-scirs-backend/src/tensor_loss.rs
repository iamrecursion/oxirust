//! Tensor-level loss functions operating on `ArrayD<f64>` with optional gradient output.
//!
//! This module provides production-ready implementations of common loss functions used
//! in machine learning, each operating on N-dimensional tensors. Unlike scalar-level
//! losses (see `tensorlogic-train`), these functions accept and return `ArrayD<f64>`
//! and support configurable reductions and gradient computation.

use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};
use std::collections::HashMap;

// ───────────────────────────────────────────────────────────────────────────────
// Error type
// ───────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during tensor-level loss computation.
#[derive(Debug, Clone)]
pub enum TensorLossError {
    /// The prediction and target tensors have different shapes.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// The target tensor contains an invalid value (e.g. out of `[0,1]` for BCE).
    InvalidTarget(String),
    /// A division-by-zero was encountered (e.g. zero-norm vector in cosine loss).
    DivisionByZero,
    /// The input tensor has no elements.
    EmptyInput,
    /// The loss was configured with an invalid parameter value.
    InvalidConfig(String),
}

impl std::fmt::Display for TensorLossError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::InvalidTarget(msg) => write!(f, "invalid target: {}", msg),
            Self::DivisionByZero => write!(f, "division by zero encountered"),
            Self::EmptyInput => write!(f, "input tensor is empty"),
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {}", msg),
        }
    }
}

impl std::error::Error for TensorLossError {}

// ───────────────────────────────────────────────────────────────────────────────
// Reduction modes
// ───────────────────────────────────────────────────────────────────────────────

/// How to aggregate element-wise losses into a scalar.
#[derive(Debug, Clone, PartialEq)]
pub enum LossReduction {
    /// Divide the summed loss by the number of elements.
    Mean,
    /// Sum all element-wise losses.
    Sum,
    /// Return the element-wise loss tensor without any aggregation.
    None,
}

// ───────────────────────────────────────────────────────────────────────────────
// Output type
// ───────────────────────────────────────────────────────────────────────────────

/// The result of computing a tensor-level loss.
#[derive(Debug, Clone)]
pub struct TensorLossOutput {
    /// Scalar loss value. When `reduction == None` this is `0.0`.
    pub loss: f64,
    /// Element-wise loss tensor. Present only when `reduction == None`.
    pub loss_tensor: Option<ArrayD<f64>>,
    /// Gradient of the loss with respect to `pred`. Present when `compute_grad == true`.
    pub grad: Option<ArrayD<f64>>,
}

// ───────────────────────────────────────────────────────────────────────────────
// Trait
// ───────────────────────────────────────────────────────────────────────────────

/// Trait implemented by all tensor-level loss functions.
pub trait TensorLoss: std::fmt::Debug {
    /// Compute the loss (and optionally the gradient) between `pred` and `target`.
    fn compute(
        &self,
        pred: &ArrayD<f64>,
        target: &ArrayD<f64>,
    ) -> Result<TensorLossOutput, TensorLossError>;

    /// Human-readable name used by the registry.
    fn name(&self) -> &'static str;
}

// ───────────────────────────────────────────────────────────────────────────────
// Shared configuration
// ───────────────────────────────────────────────────────────────────────────────

/// Configuration options shared by all built-in loss functions.
#[derive(Debug, Clone)]
pub struct TensorLossConfig {
    /// How to reduce the element-wise losses to a scalar.
    pub reduction: LossReduction,
    /// Whether to compute and return the gradient w.r.t. predictions.
    pub compute_grad: bool,
    /// Small constant for numerical stability (default `1e-8`).
    pub epsilon: f64,
}

impl Default for TensorLossConfig {
    fn default() -> Self {
        Self {
            reduction: LossReduction::Mean,
            compute_grad: true,
            epsilon: 1e-8,
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ───────────────────────────────────────────────────────────────────────────────

/// Validate that `pred` and `target` have identical shapes and are non-empty.
fn validate_shapes(pred: &ArrayD<f64>, target: &ArrayD<f64>) -> Result<usize, TensorLossError> {
    let n = pred.len();
    if n == 0 {
        return Err(TensorLossError::EmptyInput);
    }
    if pred.shape() != target.shape() {
        return Err(TensorLossError::ShapeMismatch {
            expected: pred.shape().to_vec(),
            got: target.shape().to_vec(),
        });
    }
    Ok(n)
}

/// Apply a reduction to an element-wise loss tensor and an element-wise gradient.
fn apply_reduction(
    loss_elem: ArrayD<f64>,
    grad_elem: Option<ArrayD<f64>>,
    reduction: &LossReduction,
    n: usize,
) -> TensorLossOutput {
    match reduction {
        LossReduction::None => TensorLossOutput {
            loss: 0.0,
            loss_tensor: Some(loss_elem),
            grad: grad_elem,
        },
        LossReduction::Sum => {
            let loss = loss_elem.sum();
            TensorLossOutput {
                loss,
                loss_tensor: None,
                grad: grad_elem,
            }
        }
        LossReduction::Mean => {
            let loss = loss_elem.sum() / n as f64;
            TensorLossOutput {
                loss,
                loss_tensor: None,
                grad: grad_elem,
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// MSE Loss
// ───────────────────────────────────────────────────────────────────────────────

/// Mean Squared Error loss: `mean((pred - target)^2)`.
///
/// Gradient: `2 * (pred - target) / N` (for Mean reduction).
#[derive(Debug, Clone)]
pub struct TensorMseLoss {
    pub config: TensorLossConfig,
}

impl TensorMseLoss {
    /// Create with default configuration (Mean reduction, gradient enabled).
    pub fn new() -> Self {
        Self {
            config: TensorLossConfig::default(),
        }
    }

    /// Create with a custom configuration.
    pub fn with_config(config: TensorLossConfig) -> Self {
        Self { config }
    }
}

impl Default for TensorMseLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorLoss for TensorMseLoss {
    fn name(&self) -> &'static str {
        "mse"
    }

    fn compute(
        &self,
        pred: &ArrayD<f64>,
        target: &ArrayD<f64>,
    ) -> Result<TensorLossOutput, TensorLossError> {
        let n = validate_shapes(pred, target)?;

        let diff = pred - target;
        let loss_elem = diff.mapv(|x| x * x);

        let grad = if self.config.compute_grad {
            let scale = match self.config.reduction {
                LossReduction::Mean => 2.0 / n as f64,
                LossReduction::Sum | LossReduction::None => 2.0,
            };
            Some(diff.mapv(|x| x * scale))
        } else {
            None
        };

        Ok(apply_reduction(loss_elem, grad, &self.config.reduction, n))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Binary Cross-Entropy Loss
// ───────────────────────────────────────────────────────────────────────────────

/// Binary Cross-Entropy loss: `-[t*log(p) + (1-t)*log(1-p)]`.
///
/// Predictions are clamped to `[eps, 1-eps]` for numerical stability.
/// Gradient: `-(t/p - (1-t)/(1-p))`.
#[derive(Debug, Clone)]
pub struct TensorBCELoss {
    pub config: TensorLossConfig,
}

impl TensorBCELoss {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: TensorLossConfig::default(),
        }
    }
}

impl Default for TensorBCELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorLoss for TensorBCELoss {
    fn name(&self) -> &'static str {
        "bce"
    }

    fn compute(
        &self,
        pred: &ArrayD<f64>,
        target: &ArrayD<f64>,
    ) -> Result<TensorLossOutput, TensorLossError> {
        let n = validate_shapes(pred, target)?;
        let eps = self.config.epsilon;

        // Clamp predictions for numerical stability
        let p = pred.mapv(|x| x.clamp(eps, 1.0 - eps));

        let mut loss_elem = ArrayD::zeros(IxDyn(pred.shape()));
        let mut grad_elem = if self.config.compute_grad {
            Some(ArrayD::zeros(IxDyn(pred.shape())))
        } else {
            None
        };

        Zip::from(&mut loss_elem)
            .and(&p)
            .and(target)
            .for_each(|l, &pi, &ti| {
                *l = -(ti * pi.ln() + (1.0 - ti) * (1.0 - pi).ln());
            });

        if let Some(ref mut g) = grad_elem {
            Zip::from(g).and(&p).and(target).for_each(|gi, &pi, &ti| {
                *gi = -(ti / pi - (1.0 - ti) / (1.0 - pi));
            });
        }

        Ok(apply_reduction(
            loss_elem,
            grad_elem,
            &self.config.reduction,
            n,
        ))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Categorical Cross-Entropy Loss
// ───────────────────────────────────────────────────────────────────────────────

/// Categorical Cross-Entropy loss: `-sum(t * log(p + eps))`.
///
/// Optionally applies softmax to predictions before computing the loss,
/// and supports label smoothing.
#[derive(Debug, Clone)]
pub struct TensorCrossEntropyLoss {
    pub config: TensorLossConfig,
    /// Label smoothing coefficient in `[0, 1)`. `0.0` means no smoothing.
    pub label_smoothing: f64,
    /// If `true`, apply a numerically stable softmax to predictions first.
    pub apply_softmax: bool,
}

impl TensorCrossEntropyLoss {
    /// Create with default configuration, no label smoothing, no softmax.
    pub fn new() -> Self {
        Self {
            config: TensorLossConfig::default(),
            label_smoothing: 0.0,
            apply_softmax: false,
        }
    }
}

impl Default for TensorCrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Numerically stable softmax along the last dimension of a flat tensor.
fn softmax_flat(logits: &ArrayD<f64>) -> ArrayD<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let shifted = logits.mapv(|x| (x - max_val).exp());
    let sum = shifted.sum();
    if sum == 0.0 {
        shifted
    } else {
        shifted.mapv(|x| x / sum)
    }
}

impl TensorLoss for TensorCrossEntropyLoss {
    fn name(&self) -> &'static str {
        "cross_entropy"
    }

    fn compute(
        &self,
        pred: &ArrayD<f64>,
        target: &ArrayD<f64>,
    ) -> Result<TensorLossOutput, TensorLossError> {
        let n = validate_shapes(pred, target)?;
        let eps = self.config.epsilon;
        let k = n as f64;

        // Optional softmax on predictions
        let p = if self.apply_softmax {
            softmax_flat(pred)
        } else {
            pred.clone()
        };

        // Label smoothing
        let t_smooth = if self.label_smoothing > 0.0 {
            let ls = self.label_smoothing;
            target.mapv(|ti| ti * (1.0 - ls) + ls / k)
        } else {
            target.clone()
        };

        let mut loss_elem = ArrayD::zeros(IxDyn(pred.shape()));
        Zip::from(&mut loss_elem)
            .and(&p)
            .and(&t_smooth)
            .for_each(|l, &pi, &ti| {
                *l = -(ti * (pi + eps).ln());
            });

        let grad = if self.config.compute_grad {
            // Gradient of -sum(t * log(p + eps)) w.r.t. p is -t/(p+eps)
            let mut g = ArrayD::zeros(IxDyn(pred.shape()));
            Zip::from(&mut g)
                .and(&p)
                .and(&t_smooth)
                .for_each(|gi, &pi, &ti| {
                    *gi = -ti / (pi + eps);
                });
            // If softmax was applied, chain through softmax Jacobian: g = p - t_smooth
            if self.apply_softmax {
                Some((&p) - &t_smooth)
            } else {
                Some(g)
            }
        } else {
            None
        };

        Ok(apply_reduction(loss_elem, grad, &self.config.reduction, n))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Focal Loss
// ───────────────────────────────────────────────────────────────────────────────

/// Focal loss for binary classification: `-(1 - p_t)^gamma * log(p_t + eps)`.
///
/// Downweights easy examples so the model focuses on hard ones.
#[derive(Debug, Clone)]
pub struct TensorFocalLoss {
    pub config: TensorLossConfig,
    /// Focusing parameter (default `2.0`). Higher values increase focus on hard examples.
    pub gamma: f64,
    /// Optional class-balance weight applied to the positive class.
    pub alpha: Option<f64>,
}

impl TensorFocalLoss {
    /// Create with default configuration and `gamma = 2.0`.
    pub fn new() -> Self {
        Self {
            config: TensorLossConfig::default(),
            gamma: 2.0,
            alpha: None,
        }
    }

    /// Create with a custom `gamma` value.
    pub fn with_gamma(gamma: f64) -> Self {
        Self {
            config: TensorLossConfig::default(),
            gamma,
            alpha: None,
        }
    }
}

impl Default for TensorFocalLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorLoss for TensorFocalLoss {
    fn name(&self) -> &'static str {
        "focal"
    }

    fn compute(
        &self,
        pred: &ArrayD<f64>,
        target: &ArrayD<f64>,
    ) -> Result<TensorLossOutput, TensorLossError> {
        let n = validate_shapes(pred, target)?;
        let eps = self.config.epsilon;
        let gamma = self.gamma;

        // p clamped for safety
        let p = pred.mapv(|x| x.clamp(eps, 1.0 - eps));

        let mut loss_elem = ArrayD::zeros(IxDyn(pred.shape()));
        let mut grad_elem = if self.config.compute_grad {
            Some(ArrayD::zeros(IxDyn(pred.shape())))
        } else {
            None
        };

        Zip::from(&mut loss_elem)
            .and(&p)
            .and(target)
            .for_each(|l, &pi, &ti| {
                // p_t = p if target == 1, else (1 - p)
                let p_t = if ti > 0.5 { pi } else { 1.0 - pi };
                let modulator = (1.0 - p_t).powf(gamma);
                let weight = match self.alpha {
                    Some(a) => {
                        if ti > 0.5 {
                            a
                        } else {
                            1.0 - a
                        }
                    }
                    None => 1.0,
                };
                *l = -weight * modulator * (p_t + eps).ln();
            });

        if let Some(ref mut g) = grad_elem {
            Zip::from(g).and(&p).and(target).for_each(|gi, &pi, &ti| {
                let p_t = if ti > 0.5 { pi } else { 1.0 - pi };
                let sign = if ti > 0.5 { 1.0_f64 } else { -1.0_f64 };
                let modulator = (1.0 - p_t).powf(gamma);
                let weight = match self.alpha {
                    Some(a) => {
                        if ti > 0.5 {
                            a
                        } else {
                            1.0 - a
                        }
                    }
                    None => 1.0,
                };
                // d/dp_t [ -(1-p_t)^g * ln(p_t) ]
                //   = gamma*(1-p_t)^(g-1)*ln(p_t) - (1-p_t)^g / p_t
                let term1 = if gamma > 0.0 {
                    gamma * (1.0 - p_t).powf(gamma - 1.0) * (p_t + eps).ln()
                } else {
                    0.0
                };
                let term2 = modulator / (p_t + eps);
                // chain: dp_t/dp = sign
                *gi = -weight * (term1 - term2) * sign;
            });
        }

        Ok(apply_reduction(
            loss_elem,
            grad_elem,
            &self.config.reduction,
            n,
        ))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Huber Loss
// ───────────────────────────────────────────────────────────────────────────────

/// Huber (Smooth L1) loss.
///
/// For element-wise absolute error `|x|`:
/// - If `|x| < delta`: `0.5 * x^2 / delta`
/// - Otherwise: `|x| - 0.5 * delta`
///
/// Gradient: `sign(x) * min(|x|/delta, 1)`.
#[derive(Debug, Clone)]
pub struct TensorHuberLoss {
    pub config: TensorLossConfig,
    /// Threshold between quadratic and linear regime (default `1.0`).
    pub delta: f64,
}

impl TensorHuberLoss {
    /// Create with default configuration and `delta = 1.0`.
    pub fn new() -> Self {
        Self {
            config: TensorLossConfig::default(),
            delta: 1.0,
        }
    }

    /// Create with a custom `delta` value.
    pub fn with_delta(delta: f64) -> Self {
        Self {
            config: TensorLossConfig::default(),
            delta,
        }
    }
}

impl Default for TensorHuberLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorLoss for TensorHuberLoss {
    fn name(&self) -> &'static str {
        "huber"
    }

    fn compute(
        &self,
        pred: &ArrayD<f64>,
        target: &ArrayD<f64>,
    ) -> Result<TensorLossOutput, TensorLossError> {
        let n = validate_shapes(pred, target)?;
        let delta = self.delta;

        if delta <= 0.0 {
            return Err(TensorLossError::InvalidConfig(format!(
                "delta must be positive, got {}",
                delta
            )));
        }

        let diff = pred - target;
        let mut loss_elem = ArrayD::zeros(IxDyn(pred.shape()));
        let mut grad_elem = if self.config.compute_grad {
            Some(ArrayD::zeros(IxDyn(pred.shape())))
        } else {
            None
        };

        Zip::from(&mut loss_elem).and(&diff).for_each(|l, &d| {
            let abs_d = d.abs();
            if abs_d < delta {
                *l = 0.5 * d * d / delta;
            } else {
                *l = abs_d - 0.5 * delta;
            }
        });

        if let Some(ref mut g) = grad_elem {
            Zip::from(g).and(&diff).for_each(|gi, &d| {
                let abs_d = d.abs();
                let sign = if d > 0.0 {
                    1.0
                } else if d < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                *gi = sign * (abs_d / delta).min(1.0);
            });
        }

        Ok(apply_reduction(
            loss_elem,
            grad_elem,
            &self.config.reduction,
            n,
        ))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// KL Divergence Loss
// ───────────────────────────────────────────────────────────────────────────────

/// Kullback-Leibler Divergence: `sum(target * log(target / (pred + eps)))`.
///
/// Elements where `target ≈ 0` contribute zero (following the convention `0 * log(0) = 0`).
#[derive(Debug, Clone)]
pub struct TensorKLDivLoss {
    pub config: TensorLossConfig,
}

impl TensorKLDivLoss {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: TensorLossConfig::default(),
        }
    }
}

impl Default for TensorKLDivLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorLoss for TensorKLDivLoss {
    fn name(&self) -> &'static str {
        "kl_div"
    }

    fn compute(
        &self,
        pred: &ArrayD<f64>,
        target: &ArrayD<f64>,
    ) -> Result<TensorLossOutput, TensorLossError> {
        let n = validate_shapes(pred, target)?;
        let eps = self.config.epsilon;

        let mut loss_elem = ArrayD::zeros(IxDyn(pred.shape()));
        let mut grad_elem = if self.config.compute_grad {
            Some(ArrayD::zeros(IxDyn(pred.shape())))
        } else {
            None
        };

        Zip::from(&mut loss_elem)
            .and(pred)
            .and(target)
            .for_each(|l, &pi, &ti| {
                if ti > eps {
                    let p_safe = pi.max(eps);
                    // KL(T || P) = T * (ln T - ln P)
                    *l = ti * (ti.ln() - p_safe.ln());
                }
                // else: 0 * log(0) = 0, leave as 0
            });

        if let Some(ref mut g) = grad_elem {
            // d KL / d p_i = -t_i / (p_i + eps)
            Zip::from(g).and(pred).and(target).for_each(|gi, &pi, &ti| {
                if ti > eps {
                    *gi = -ti / (pi + eps);
                }
            });
        }

        Ok(apply_reduction(
            loss_elem,
            grad_elem,
            &self.config.reduction,
            n,
        ))
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Cosine Embedding Loss
// ───────────────────────────────────────────────────────────────────────────────

/// Cosine Embedding loss: `1 - cosine_similarity(pred, target)`.
///
/// Treats inputs as flat vectors (all dimensions collapsed).
#[derive(Debug, Clone)]
pub struct TensorCosineEmbeddingLoss {
    pub config: TensorLossConfig,
}

impl TensorCosineEmbeddingLoss {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: TensorLossConfig::default(),
        }
    }
}

impl Default for TensorCosineEmbeddingLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorLoss for TensorCosineEmbeddingLoss {
    fn name(&self) -> &'static str {
        "cosine_embedding"
    }

    fn compute(
        &self,
        pred: &ArrayD<f64>,
        target: &ArrayD<f64>,
    ) -> Result<TensorLossOutput, TensorLossError> {
        let n = validate_shapes(pred, target)?;
        let eps = self.config.epsilon;

        let dot: f64 = pred.iter().zip(target.iter()).map(|(p, t)| p * t).sum();
        let norm_p: f64 = pred.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_t: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
        let denom = norm_p * norm_t + eps;

        let similarity = dot / denom;
        let scalar_loss = 1.0 - similarity;

        // For the gradient: d(1 - cos) / d(pred_i)
        //   = -(d cos / d pred_i)
        //   = -(target_i / denom - dot * pred_i / (norm_p^2 * denom + eps))
        let grad = if self.config.compute_grad {
            let mut g = ArrayD::zeros(IxDyn(pred.shape()));
            let norm_p_sq = norm_p * norm_p + eps;
            Zip::from(&mut g)
                .and(pred)
                .and(target)
                .for_each(|gi, &pi, &ti| {
                    let d_sim = ti / denom - dot * pi / (norm_p_sq * denom);
                    *gi = -d_sim;
                });
            Some(g)
        } else {
            None
        };

        // Cosine loss is inherently a single scalar; build a uniform tensor for consistency.
        match self.config.reduction {
            LossReduction::None => {
                // Return an element-wise tensor filled with scalar_loss / n
                // (so it sums to scalar_loss).
                let loss_tensor = ArrayD::from_elem(IxDyn(pred.shape()), scalar_loss / n as f64);
                Ok(TensorLossOutput {
                    loss: 0.0,
                    loss_tensor: Some(loss_tensor),
                    grad,
                })
            }
            LossReduction::Mean | LossReduction::Sum => Ok(TensorLossOutput {
                loss: scalar_loss,
                loss_tensor: None,
                grad,
            }),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Registry
// ───────────────────────────────────────────────────────────────────────────────

/// Dynamic registry for named tensor-level loss functions.
///
/// Use [`TensorLossRegistry::with_all_defaults`] to get a registry pre-populated
/// with all seven built-in losses.
#[derive(Debug)]
pub struct TensorLossRegistry {
    losses: HashMap<String, Box<dyn TensorLoss>>,
}

impl TensorLossRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            losses: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with all seven built-in losses:
    /// `"mse"`, `"bce"`, `"cross_entropy"`, `"focal"`, `"huber"`, `"kl_div"`,
    /// `"cosine_embedding"`.
    pub fn with_all_defaults() -> Self {
        let mut reg = Self::new();
        reg.register("mse", Box::new(TensorMseLoss::new()));
        reg.register("bce", Box::new(TensorBCELoss::new()));
        reg.register("cross_entropy", Box::new(TensorCrossEntropyLoss::new()));
        reg.register("focal", Box::new(TensorFocalLoss::new()));
        reg.register("huber", Box::new(TensorHuberLoss::new()));
        reg.register("kl_div", Box::new(TensorKLDivLoss::new()));
        reg.register(
            "cosine_embedding",
            Box::new(TensorCosineEmbeddingLoss::new()),
        );
        reg
    }

    /// Register a loss under a name. Overwrites any previous entry with the same name.
    pub fn register(&mut self, name: impl Into<String>, loss: Box<dyn TensorLoss>) {
        self.losses.insert(name.into(), loss);
    }

    /// Compute a named loss.
    ///
    /// Returns [`TensorLossError::InvalidConfig`] if the name is not registered.
    pub fn compute(
        &self,
        name: &str,
        pred: &ArrayD<f64>,
        target: &ArrayD<f64>,
    ) -> Result<TensorLossOutput, TensorLossError> {
        let loss = self.losses.get(name).ok_or_else(|| {
            TensorLossError::InvalidConfig(format!("no loss registered under name '{}'", name))
        })?;
        loss.compute(pred, target)
    }

    /// Return all registered loss names (order is not guaranteed).
    pub fn names(&self) -> Vec<&str> {
        self.losses.keys().map(|s| s.as_str()).collect()
    }

    /// Return `true` if a loss is registered under `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.losses.contains_key(name)
    }
}

impl Default for TensorLossRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    fn to_arrayd(v: Vec<f64>) -> ArrayD<f64> {
        arr1(&v).into_dyn()
    }

    // ── MSE ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mse_zero_loss_identical_arrays() {
        let a = to_arrayd(vec![1.0, 2.0, 3.0]);
        let loss = TensorMseLoss::new().compute(&a, &a).unwrap();
        assert!(
            (loss.loss).abs() < 1e-10,
            "identical arrays should yield zero loss"
        );
    }

    #[test]
    fn test_mse_loss_value_correct() {
        // pred = [1, 2], target = [0, 0], mse = mean([1, 4]) = 2.5
        let pred = to_arrayd(vec![1.0, 2.0]);
        let target = to_arrayd(vec![0.0, 0.0]);
        let out = TensorMseLoss::new().compute(&pred, &target).unwrap();
        assert!((out.loss - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_mse_gradient_shape() {
        let pred = to_arrayd(vec![1.0, 2.0, 3.0]);
        let target = to_arrayd(vec![0.0, 0.0, 0.0]);
        let out = TensorMseLoss::new().compute(&pred, &target).unwrap();
        let grad = out.grad.unwrap();
        assert_eq!(grad.shape(), pred.shape());
    }

    #[test]
    fn test_mse_gradient_direction() {
        // When pred > target, gradient should be positive.
        let pred = to_arrayd(vec![3.0, 2.0]);
        let target = to_arrayd(vec![1.0, 1.0]);
        let out = TensorMseLoss::new().compute(&pred, &target).unwrap();
        let grad = out.grad.unwrap();
        for &g in grad.iter() {
            assert!(g > 0.0, "gradient should be positive when pred > target");
        }
    }

    // ── BCE ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_bce_perfect_prediction_near_zero() {
        // Perfect binary predictions → very small loss
        let pred = to_arrayd(vec![0.9999, 0.0001]);
        let target = to_arrayd(vec![1.0, 0.0]);
        let out = TensorBCELoss::new().compute(&pred, &target).unwrap();
        assert!(out.loss < 1e-3, "near-perfect predictions → near-zero loss");
    }

    #[test]
    fn test_bce_gradient_shape() {
        let pred = to_arrayd(vec![0.5, 0.7]);
        let target = to_arrayd(vec![1.0, 0.0]);
        let out = TensorBCELoss::new().compute(&pred, &target).unwrap();
        let grad = out.grad.unwrap();
        assert_eq!(grad.shape(), pred.shape());
    }

    // ── Cross-Entropy ────────────────────────────────────────────────────────────

    #[test]
    fn test_cross_entropy_uniform_target() {
        // Uniform prediction and target 1/3 each.
        // element loss = -(1/3) * ln(1/3 + eps), 3 elements, mean reduction.
        let eps = 1e-8_f64;
        let p = 1.0_f64 / 3.0;
        let pred = to_arrayd(vec![p; 3]);
        let target = to_arrayd(vec![p; 3]);
        let out = TensorCrossEntropyLoss::new()
            .compute(&pred, &target)
            .unwrap();
        // mean of 3 identical elements: -(p * ln(p + eps))
        let expected = -(p * (p + eps).ln());
        assert!(
            (out.loss - expected).abs() < 1e-6,
            "expected {}, got {}",
            expected,
            out.loss
        );
    }

    #[test]
    fn test_cross_entropy_label_smoothing() {
        // With label smoothing, loss should differ from no-smoothing version
        let pred = to_arrayd(vec![0.9, 0.05, 0.05]);
        let target = to_arrayd(vec![1.0, 0.0, 0.0]);

        let no_smooth = TensorCrossEntropyLoss::new()
            .compute(&pred, &target)
            .unwrap();

        let with_smooth = TensorCrossEntropyLoss {
            label_smoothing: 0.1,
            ..TensorCrossEntropyLoss::new()
        }
        .compute(&pred, &target)
        .unwrap();

        assert!(
            (no_smooth.loss - with_smooth.loss).abs() > 1e-6,
            "label smoothing should change the loss"
        );
    }

    // ── Focal ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_focal_gamma_zero_equals_bce() {
        // focal loss with gamma=0 should approximate BCE
        let pred = to_arrayd(vec![0.7, 0.3, 0.8]);
        let target = to_arrayd(vec![1.0, 0.0, 1.0]);

        let focal = TensorFocalLoss::with_gamma(0.0)
            .compute(&pred, &target)
            .unwrap();
        let bce = TensorBCELoss::new().compute(&pred, &target).unwrap();

        assert!(
            (focal.loss - bce.loss).abs() < 1e-6,
            "focal(gamma=0) ≈ BCE, got focal={} bce={}",
            focal.loss,
            bce.loss
        );
    }

    #[test]
    fn test_focal_high_confidence_downweighted() {
        // High-confidence correct prediction should have less focal loss than BCE contribution
        let pred_high = to_arrayd(vec![0.99]);
        let pred_low = to_arrayd(vec![0.6]);
        let target = to_arrayd(vec![1.0]);

        let focal = TensorFocalLoss::new(); // gamma=2
        let out_high = focal.compute(&pred_high, &target).unwrap();
        let out_low = focal.compute(&pred_low, &target).unwrap();
        assert!(
            out_high.loss < out_low.loss,
            "high-confidence correct prediction should be downweighted"
        );
    }

    // ── Huber ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_huber_small_error_quadratic() {
        // |x| = 0.5 < delta=1 → quadratic: 0.5 * x^2 / delta = 0.5 * 0.25 / 1 = 0.125
        let pred = to_arrayd(vec![0.5]);
        let target = to_arrayd(vec![0.0]);
        let out = TensorHuberLoss::new().compute(&pred, &target).unwrap();
        assert!((out.loss - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_huber_large_error_linear() {
        // |x| = 2.0 > delta=1 → linear: |x| - 0.5*delta = 2 - 0.5 = 1.5
        let pred = to_arrayd(vec![2.0]);
        let target = to_arrayd(vec![0.0]);
        let out = TensorHuberLoss::new().compute(&pred, &target).unwrap();
        assert!((out.loss - 1.5).abs() < 1e-10);
    }

    // ── KL Divergence ────────────────────────────────────────────────────────────

    #[test]
    fn test_kl_div_identical_distributions_zero() {
        let p = to_arrayd(vec![0.3, 0.5, 0.2]);
        let out = TensorKLDivLoss::new().compute(&p, &p).unwrap();
        // KL(P||P) should be ≈ 0 (small due to eps)
        assert!(out.loss.abs() < 1e-6);
    }

    #[test]
    fn test_kl_div_gradient_shape() {
        let pred = to_arrayd(vec![0.3, 0.5, 0.2]);
        let target = to_arrayd(vec![0.4, 0.4, 0.2]);
        let out = TensorKLDivLoss::new().compute(&pred, &target).unwrap();
        let grad = out.grad.unwrap();
        assert_eq!(grad.shape(), pred.shape());
    }

    // ── Cosine Embedding ─────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_parallel_loss_zero() {
        // Same direction → cosine similarity = 1 → loss = 0
        let pred = to_arrayd(vec![1.0, 0.0, 0.0]);
        let target = to_arrayd(vec![2.0, 0.0, 0.0]); // parallel, different magnitude
        let out = TensorCosineEmbeddingLoss::new()
            .compute(&pred, &target)
            .unwrap();
        assert!(out.loss.abs() < 1e-6, "parallel vectors → loss ≈ 0");
    }

    #[test]
    fn test_cosine_orthogonal_loss_one() {
        // Orthogonal vectors → cosine similarity = 0 → loss = 1
        let pred = to_arrayd(vec![1.0, 0.0]);
        let target = to_arrayd(vec![0.0, 1.0]);
        let out = TensorCosineEmbeddingLoss::new()
            .compute(&pred, &target)
            .unwrap();
        assert!(
            (out.loss - 1.0).abs() < 1e-6,
            "orthogonal vectors → loss ≈ 1"
        );
    }

    // ── Reduction ────────────────────────────────────────────────────────────────

    #[test]
    fn test_reduction_sum_vs_mean() {
        let pred = to_arrayd(vec![1.0, 2.0, 3.0]);
        let target = to_arrayd(vec![0.0, 0.0, 0.0]);

        let mean_loss = TensorMseLoss::with_config(TensorLossConfig {
            reduction: LossReduction::Mean,
            ..Default::default()
        })
        .compute(&pred, &target)
        .unwrap();

        let sum_loss = TensorMseLoss::with_config(TensorLossConfig {
            reduction: LossReduction::Sum,
            ..Default::default()
        })
        .compute(&pred, &target)
        .unwrap();

        assert!(
            (sum_loss.loss - mean_loss.loss).abs() > 1e-6,
            "sum != mean for non-unit arrays"
        );
    }

    #[test]
    fn test_reduction_none_returns_tensor() {
        let pred = to_arrayd(vec![1.0, 2.0]);
        let target = to_arrayd(vec![0.0, 0.0]);

        let out = TensorMseLoss::with_config(TensorLossConfig {
            reduction: LossReduction::None,
            ..Default::default()
        })
        .compute(&pred, &target)
        .unwrap();

        assert!(
            out.loss_tensor.is_some(),
            "None reduction should return a loss tensor"
        );
        let lt = out.loss_tensor.unwrap();
        assert_eq!(lt.shape(), pred.shape());
    }

    // ── Registry ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_registry_with_all_defaults() {
        let reg = TensorLossRegistry::with_all_defaults();
        assert_eq!(
            reg.names().len(),
            7,
            "registry should contain 7 built-in losses"
        );
        for name in &[
            "mse",
            "bce",
            "cross_entropy",
            "focal",
            "huber",
            "kl_div",
            "cosine_embedding",
        ] {
            assert!(reg.contains(name), "missing: {}", name);
        }
    }

    #[test]
    fn test_registry_compute_by_name() {
        let reg = TensorLossRegistry::with_all_defaults();
        let pred = to_arrayd(vec![0.5, 0.5]);
        let target = to_arrayd(vec![1.0, 0.0]);
        let out = reg.compute("bce", &pred, &target).unwrap();
        assert!(
            out.loss > 0.0,
            "BCE of non-perfect prediction should be positive"
        );
    }
}

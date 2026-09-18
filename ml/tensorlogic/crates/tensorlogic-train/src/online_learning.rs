//! Online learning algorithms: Perceptron, Passive-Aggressive, OGD, and FTRL.
//!
//! All algorithms process one sample at a time with O(d) memory where d is
//! the number of features, making them suitable for streaming and large-scale
//! applications where the full dataset cannot be held in memory.
//!
//! # Algorithms
//!
//! - [`Perceptron`]: Classic binary classifier (Rosenblatt 1958)
//! - [`PassiveAggressive`]: PA, PA-I, PA-II variants (Crammer et al. 2006)
//! - [`OnlineGradientDescent`]: OGD with squared/hinge/logistic losses
//! - [`Ftrl`]: Follow the Regularized Leader-Proximal (McMahan et al. 2013)

use std::fmt;

// ---------------------------------------------------------------------------
// Core trait and result types
// ---------------------------------------------------------------------------

/// Trait for online learners that update one sample at a time.
pub trait OnlineLearner {
    /// Update model on a single (features, label) pair. Returns update stats.
    fn update(&mut self, features: &[f64], label: f64) -> Result<OnlineUpdateResult, OnlineError>;

    /// Predict class label or regression value for features.
    fn predict(&self, features: &[f64]) -> Result<f64, OnlineError>;

    /// Number of updates seen so far.
    fn n_updates(&self) -> usize;

    /// Current weight vector (without bias).
    fn weights(&self) -> &[f64];
}

/// Result of a single online update step.
#[derive(Debug, Clone)]
pub struct OnlineUpdateResult {
    /// Loss on this sample computed *before* the update.
    pub loss: f64,
    /// L2 norm of the weight change vector (||Δw||).
    pub weight_delta_norm: f64,
    /// For classifiers: whether the prediction was incorrect before the update.
    pub was_mistake: bool,
}

/// Errors that can arise in online learning routines.
#[derive(Debug)]
pub enum OnlineError {
    /// Feature dimensionality does not match model dimensionality.
    DimensionMismatch { expected: usize, got: usize },
    /// A hyperparameter has an invalid value.
    InvalidHyperparameter(String),
    /// Prediction attempted before the model has received any data.
    NotFitted,
}

impl fmt::Display for OnlineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnlineError::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} features, got {got}"
            ),
            OnlineError::InvalidHyperparameter(msg) => {
                write!(f, "invalid hyperparameter: {msg}")
            }
            OnlineError::NotFitted => write!(f, "model has not been fitted yet"),
        }
    }
}

impl std::error::Error for OnlineError {}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for an online learning session.
#[derive(Debug, Clone, Default)]
pub struct OnlineStats {
    /// Total number of update calls.
    pub n_updates: usize,
    /// Total number of incorrect predictions (classification only).
    pub n_mistakes: usize,
    /// Sum of per-sample losses.
    pub cumulative_loss: f64,
    /// Running mean loss: cumulative_loss / n_updates.
    pub mean_loss: f64,
    /// ||w|| after the most recent update.
    pub last_weight_norm: f64,
}

impl OnlineStats {
    /// Fraction of updates that resulted in a mistake (classification).
    ///
    /// Returns 0.0 when no updates have been performed.
    pub fn mistake_rate(&self) -> f64 {
        if self.n_updates == 0 {
            0.0
        } else {
            self.n_mistakes as f64 / self.n_updates as f64
        }
    }

    /// Incorporate the result of one update step into running statistics.
    pub fn update(&mut self, result: &OnlineUpdateResult) {
        self.n_updates += 1;
        if result.was_mistake {
            self.n_mistakes += 1;
        }
        self.cumulative_loss += result.loss;
        self.mean_loss = self.cumulative_loss / self.n_updates as f64;
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the squared L2 norm of a slice.
#[inline]
fn l2_norm_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Compute the L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    l2_norm_sq(v).sqrt()
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Sign function returning -1.0, 0.0, or +1.0.
#[inline]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Perceptron
// ---------------------------------------------------------------------------

/// Binary Perceptron classifier (Rosenblatt 1958).
///
/// Labels must be in {−1, +1}. The update rule fires only when a prediction
/// is wrong: `w ← w + η·y·x` and `bias ← bias + η·y`.
#[derive(Debug, Clone)]
pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    n_updates: usize,
    stats: OnlineStats,
    learning_rate: f64,
}

impl Perceptron {
    /// Create a new Perceptron with `n_features` dimensions and default `η = 1.0`.
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            n_updates: 0,
            stats: OnlineStats::default(),
            learning_rate: 1.0,
        }
    }

    /// Set the per-mistake learning rate (η).
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Current bias term.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Reference to running statistics.
    pub fn stats(&self) -> &OnlineStats {
        &self.stats
    }

    /// Raw score w·x + bias.
    fn score(&self, features: &[f64]) -> f64 {
        dot(&self.weights, features) + self.bias
    }
}

impl OnlineLearner for Perceptron {
    fn update(&mut self, features: &[f64], label: f64) -> Result<OnlineUpdateResult, OnlineError> {
        let n = self.weights.len();
        if features.len() != n {
            return Err(OnlineError::DimensionMismatch {
                expected: n,
                got: features.len(),
            });
        }

        let score = self.score(features);
        let predicted_sign = sign(score);
        let true_sign = sign(label);

        // Hinge-like loss: max(0, -y * score).
        let margin = true_sign * score;
        let loss = if margin <= 0.0 { -margin } else { 0.0 };
        let was_mistake = predicted_sign != true_sign;

        let mut delta_sq = 0.0_f64;

        if was_mistake {
            let eta_y = self.learning_rate * true_sign;
            for (w, x) in self.weights.iter_mut().zip(features.iter()) {
                let delta = eta_y * x;
                delta_sq += delta * delta;
                *w += delta;
            }
            let bias_delta = self.learning_rate * true_sign;
            delta_sq += bias_delta * bias_delta;
            self.bias += bias_delta;
        }

        self.n_updates += 1;

        // Update last_weight_norm in stats.
        let weight_delta_norm = delta_sq.sqrt();
        let result = OnlineUpdateResult {
            loss,
            weight_delta_norm,
            was_mistake,
        };
        self.stats.update(&result);
        self.stats.last_weight_norm = l2_norm(&self.weights);

        Ok(result)
    }

    fn predict(&self, features: &[f64]) -> Result<f64, OnlineError> {
        let n = self.weights.len();
        if features.len() != n {
            return Err(OnlineError::DimensionMismatch {
                expected: n,
                got: features.len(),
            });
        }
        Ok(sign(self.score(features)))
    }

    fn n_updates(&self) -> usize {
        self.n_updates
    }

    fn weights(&self) -> &[f64] {
        &self.weights
    }
}

// ---------------------------------------------------------------------------
// Passive-Aggressive
// ---------------------------------------------------------------------------

/// Selects the PA update variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PAVariant {
    /// Unconstrained PA: τ = loss / ||x||².
    PA,
    /// PA-I: τ = min(C, loss / ||x||²).
    PAI,
    /// PA-II: τ = loss / (||x||² + 1 / (2C)).
    PAII,
}

/// Passive-Aggressive classifier (Crammer et al. 2006).
///
/// Labels must be in {−1, +1}. The PA family uses the hinge loss
/// `ℓ = max(0, 1 − y(w·x + b))` and then computes a closed-form
/// minimal weight update.
#[derive(Debug, Clone)]
pub struct PassiveAggressive {
    weights: Vec<f64>,
    bias: f64,
    n_updates: usize,
    stats: OnlineStats,
    aggressiveness: f64,
    variant: PAVariant,
}

impl PassiveAggressive {
    /// Create a new PA classifier. `variant` controls the update rule.
    pub fn new(n_features: usize, variant: PAVariant) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            n_updates: 0,
            stats: OnlineStats::default(),
            aggressiveness: 1.0,
            variant,
        }
    }

    /// Set the aggressiveness parameter C (must be positive).
    pub fn with_aggressiveness(mut self, c: f64) -> Result<Self, OnlineError> {
        if c <= 0.0 {
            return Err(OnlineError::InvalidHyperparameter(format!(
                "aggressiveness C must be > 0, got {c}"
            )));
        }
        self.aggressiveness = c;
        Ok(self)
    }

    /// Reference to running statistics.
    pub fn stats(&self) -> &OnlineStats {
        &self.stats
    }

    /// Compute τ (step size) for the current sample.
    fn compute_tau(&self, loss: f64, x_norm_sq: f64) -> f64 {
        match self.variant {
            PAVariant::PA => {
                if x_norm_sq == 0.0 {
                    0.0
                } else {
                    loss / x_norm_sq
                }
            }
            PAVariant::PAI => {
                let tau_unconstrained = if x_norm_sq == 0.0 {
                    0.0
                } else {
                    loss / x_norm_sq
                };
                tau_unconstrained.min(self.aggressiveness)
            }
            PAVariant::PAII => {
                let denom = x_norm_sq + 1.0 / (2.0 * self.aggressiveness);
                if denom == 0.0 {
                    0.0
                } else {
                    loss / denom
                }
            }
        }
    }
}

impl OnlineLearner for PassiveAggressive {
    fn update(&mut self, features: &[f64], label: f64) -> Result<OnlineUpdateResult, OnlineError> {
        let n = self.weights.len();
        if features.len() != n {
            return Err(OnlineError::DimensionMismatch {
                expected: n,
                got: features.len(),
            });
        }

        let score = dot(&self.weights, features) + self.bias;
        let y = sign(label);

        // Hinge loss: max(0, 1 - y * score).
        let margin = y * score;
        let loss = (1.0 - margin).max(0.0);
        let was_mistake = sign(score) != y;

        let x_norm_sq = l2_norm_sq(features);
        let tau = self.compute_tau(loss, x_norm_sq);

        let mut delta_sq = 0.0_f64;
        if tau > 0.0 {
            let tau_y = tau * y;
            for (w, x) in self.weights.iter_mut().zip(features.iter()) {
                let delta = tau_y * x;
                delta_sq += delta * delta;
                *w += delta;
            }
            let bias_delta = tau * y;
            delta_sq += bias_delta * bias_delta;
            self.bias += bias_delta;
        }

        self.n_updates += 1;

        let result = OnlineUpdateResult {
            loss,
            weight_delta_norm: delta_sq.sqrt(),
            was_mistake,
        };
        self.stats.update(&result);
        self.stats.last_weight_norm = l2_norm(&self.weights);

        Ok(result)
    }

    fn predict(&self, features: &[f64]) -> Result<f64, OnlineError> {
        let n = self.weights.len();
        if features.len() != n {
            return Err(OnlineError::DimensionMismatch {
                expected: n,
                got: features.len(),
            });
        }
        Ok(sign(dot(&self.weights, features) + self.bias))
    }

    fn n_updates(&self) -> usize {
        self.n_updates
    }

    fn weights(&self) -> &[f64] {
        &self.weights
    }
}

// ---------------------------------------------------------------------------
// Online Gradient Descent
// ---------------------------------------------------------------------------

/// Loss function for [`OnlineGradientDescent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OGDLoss {
    /// Squared loss: ℓ = ½(pred − y)². Gradient: (pred − y)·x.
    Squared,
    /// Hinge loss: ℓ = max(0, 1 − y·score). Gradient: −y·x when margin < 1.
    Hinge,
    /// Logistic loss: ℓ = log(1 + exp(−y·score)). Gradient: −y·σ(−y·score)·x.
    Logistic,
}

/// Online Gradient Descent for convex losses.
///
/// The learning rate schedule follows η_t = η_0 / √(t + 1) when `lr_decay > 0`,
/// otherwise a constant η_0 is used. Optional L2 regularisation applies weight
/// decay at each step.
#[derive(Debug, Clone)]
pub struct OnlineGradientDescent {
    weights: Vec<f64>,
    bias: f64,
    n_updates: usize,
    stats: OnlineStats,
    initial_lr: f64,
    lr_decay: f64,
    l2_reg: f64,
    loss: OGDLoss,
}

impl OnlineGradientDescent {
    /// Create a new OGD learner for the given loss function.
    pub fn new(n_features: usize, loss: OGDLoss) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            n_updates: 0,
            stats: OnlineStats::default(),
            initial_lr: 0.1,
            lr_decay: 0.0,
            l2_reg: 0.0,
            loss,
        }
    }

    /// Set the initial learning rate η_0.
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.initial_lr = lr;
        self
    }

    /// Set the L2 regularisation coefficient λ.
    pub fn with_l2(mut self, lambda: f64) -> Self {
        self.l2_reg = lambda;
        self
    }

    /// Enable learning rate decay. When `decay > 0`, η_t = η_0 / √(t + 1).
    pub fn with_lr_decay(mut self, decay: f64) -> Self {
        self.lr_decay = decay;
        self
    }

    /// Reference to running statistics.
    pub fn stats(&self) -> &OnlineStats {
        &self.stats
    }

    /// Effective learning rate at the current step.
    fn current_lr(&self) -> f64 {
        if self.lr_decay > 0.0 {
            self.initial_lr / ((self.n_updates as f64 + 1.0).sqrt())
        } else {
            self.initial_lr
        }
    }

    /// Compute loss and gradient coefficient `g` such that `∂ℓ/∂w = g·x`.
    /// Returns `(loss, grad_coeff, bias_grad)`.
    fn compute_loss_and_grad(&self, features: &[f64], label: f64) -> (f64, f64, f64) {
        let score = dot(&self.weights, features) + self.bias;
        match self.loss {
            OGDLoss::Squared => {
                let diff = score - label;
                let loss = 0.5 * diff * diff;
                (loss, diff, diff)
            }
            OGDLoss::Hinge => {
                let y = sign(label);
                let margin = y * score;
                if margin < 1.0 {
                    let loss = 1.0 - margin;
                    (loss, -y, -y)
                } else {
                    (0.0, 0.0, 0.0)
                }
            }
            OGDLoss::Logistic => {
                let y = sign(label);
                // σ(-y·s) = 1 / (1 + exp(y·s))
                let ys = y * score;
                let sigma_neg = 1.0 / (1.0 + ys.exp()); // σ(-y·s)
                let loss = (1.0 + (-ys).exp()).ln();
                let grad_coeff = -y * sigma_neg;
                (loss, grad_coeff, grad_coeff)
            }
        }
    }
}

impl OnlineLearner for OnlineGradientDescent {
    fn update(&mut self, features: &[f64], label: f64) -> Result<OnlineUpdateResult, OnlineError> {
        let n = self.weights.len();
        if features.len() != n {
            return Err(OnlineError::DimensionMismatch {
                expected: n,
                got: features.len(),
            });
        }

        let (loss, grad_coeff, bias_grad) = self.compute_loss_and_grad(features, label);
        let eta = self.current_lr();

        let was_mistake = match self.loss {
            OGDLoss::Squared => false, // regression — no concept of "mistake"
            OGDLoss::Hinge | OGDLoss::Logistic => {
                let score = dot(&self.weights, features) + self.bias;
                sign(score) != sign(label)
            }
        };

        let mut delta_sq = 0.0_f64;

        // Gradient step + L2 regularisation (weight decay).
        for (w, x) in self.weights.iter_mut().zip(features.iter()) {
            let grad = grad_coeff * x + self.l2_reg * (*w);
            let delta = -eta * grad;
            delta_sq += delta * delta;
            *w += delta;
        }
        // Bias is not regularised.
        let bias_delta = -eta * bias_grad;
        delta_sq += bias_delta * bias_delta;
        self.bias += bias_delta;

        self.n_updates += 1;

        let result = OnlineUpdateResult {
            loss,
            weight_delta_norm: delta_sq.sqrt(),
            was_mistake,
        };
        self.stats.update(&result);
        self.stats.last_weight_norm = l2_norm(&self.weights);

        Ok(result)
    }

    fn predict(&self, features: &[f64]) -> Result<f64, OnlineError> {
        let n = self.weights.len();
        if features.len() != n {
            return Err(OnlineError::DimensionMismatch {
                expected: n,
                got: features.len(),
            });
        }
        let score = dot(&self.weights, features) + self.bias;
        let prediction = match self.loss {
            OGDLoss::Squared => score,
            OGDLoss::Hinge | OGDLoss::Logistic => sign(score),
        };
        Ok(prediction)
    }

    fn n_updates(&self) -> usize {
        self.n_updates
    }

    fn weights(&self) -> &[f64] {
        &self.weights
    }
}

// ---------------------------------------------------------------------------
// FTRL-Proximal
// ---------------------------------------------------------------------------

/// Follow the Regularized Leader — Proximal (McMahan et al. 2013).
///
/// FTRL-Proximal maintains per-feature adaptive learning rates and supports
/// L1 + L2 regularization. L1 induces sparsity: features with |z_i| ≤ l1
/// are zeroed out, which makes FTRL popular for large-scale sparse models.
///
/// Update equations (per coordinate i):
/// ```text
/// g_i       = gradient of logistic loss on this sample
/// z_i      += g_i − (√(n_i + g_i²) − √n_i) / α · w_i
/// n_i      += g_i²
/// if |z_i| ≤ l1:
///     w_i = 0
/// else:
///     w_i = −(z_i − sign(z_i)·l1) / ((β + √n_i) / α + l2)
/// ```
#[derive(Debug, Clone)]
pub struct Ftrl {
    weights: Vec<f64>,
    /// Accumulated gradient vector (z in the FTRL paper).
    z: Vec<f64>,
    /// Accumulated squared gradient per feature (n in the FTRL paper).
    n_vec: Vec<f64>,
    n_updates: usize,
    stats: OnlineStats,
    alpha: f64,
    beta: f64,
    l1: f64,
    l2: f64,
}

impl Ftrl {
    /// Create a new FTRL learner with `n_features` dimensions.
    ///
    /// Defaults: α = 0.1, β = 1.0, l1 = 0.0, l2 = 0.0.
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            z: vec![0.0; n_features],
            n_vec: vec![0.0; n_features],
            n_updates: 0,
            stats: OnlineStats::default(),
            alpha: 0.1,
            beta: 1.0,
            l1: 0.0,
            l2: 0.0,
        }
    }

    /// Set the learning rate α.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set L1 and L2 regularization coefficients.
    pub fn with_l1_l2(mut self, l1: f64, l2: f64) -> Self {
        self.l1 = l1;
        self.l2 = l2;
        self
    }

    /// Reference to running statistics.
    pub fn stats(&self) -> &OnlineStats {
        &self.stats
    }

    /// Recompute weight from accumulated z and n for coordinate i.
    #[inline]
    fn compute_weight(&self, i: usize) -> f64 {
        let z_i = self.z[i];
        let n_i = self.n_vec[i];
        if z_i.abs() <= self.l1 {
            0.0
        } else {
            let numerator = -(z_i - sign(z_i) * self.l1);
            let denominator = (self.beta + n_i.sqrt()) / self.alpha + self.l2;
            if denominator == 0.0 {
                0.0
            } else {
                numerator / denominator
            }
        }
    }

    /// Compute raw score w·x using on-the-fly weight computation.
    fn score(&self, features: &[f64]) -> f64 {
        features
            .iter()
            .enumerate()
            .map(|(i, x)| self.compute_weight(i) * x)
            .sum::<f64>()
    }

    /// Logistic probability σ(s) = 1 / (1 + e^{-s}).
    #[inline]
    fn sigmoid(s: f64) -> f64 {
        1.0 / (1.0 + (-s).exp())
    }
}

impl OnlineLearner for Ftrl {
    fn update(&mut self, features: &[f64], label: f64) -> Result<OnlineUpdateResult, OnlineError> {
        let n = self.weights.len();
        if features.len() != n {
            return Err(OnlineError::DimensionMismatch {
                expected: n,
                got: features.len(),
            });
        }

        // Sync weights from z/n before computing score.
        for i in 0..n {
            self.weights[i] = self.compute_weight(i);
        }

        let score = dot(&self.weights, features);
        let p = Self::sigmoid(score);

        // FTRL uses logistic loss; label is mapped to {0, 1} for gradient.
        // y_01 = 1 if label > 0 else 0.
        let y_01 = if label > 0.0 { 1.0_f64 } else { 0.0_f64 };
        let grad_scale = p - y_01; // ∂ℓ/∂score = p − y

        // Logistic loss: -y log p - (1-y) log(1-p).
        let loss = if y_01 > 0.0 {
            -p.ln().max(-1e15)
        } else {
            -(1.0 - p).ln().max(-1e15)
        };

        let was_mistake = sign(score) != sign(label - 0.5); // compare against 0.5 threshold

        let old_weights = self.weights.clone();

        // FTRL update per coordinate.
        for (i, &feat_i) in features.iter().enumerate().take(n) {
            let g_i = grad_scale * feat_i;
            let n_i_old = self.n_vec[i];
            let n_i_new = n_i_old + g_i * g_i;

            // σ_i = (√n_i_new − √n_i_old) / α
            let sigma_i = (n_i_new.sqrt() - n_i_old.sqrt()) / self.alpha;

            self.z[i] += g_i - sigma_i * self.weights[i];
            self.n_vec[i] = n_i_new;
            self.weights[i] = self.compute_weight(i);
        }

        let delta_norm = {
            let sq: f64 = self
                .weights
                .iter()
                .zip(old_weights.iter())
                .map(|(w_new, w_old)| {
                    let d = w_new - w_old;
                    d * d
                })
                .sum();
            sq.sqrt()
        };

        self.n_updates += 1;

        let result = OnlineUpdateResult {
            loss,
            weight_delta_norm: delta_norm,
            was_mistake,
        };
        self.stats.update(&result);
        self.stats.last_weight_norm = l2_norm(&self.weights);

        Ok(result)
    }

    fn predict(&self, features: &[f64]) -> Result<f64, OnlineError> {
        let n = self.weights.len();
        if features.len() != n {
            return Err(OnlineError::DimensionMismatch {
                expected: n,
                got: features.len(),
            });
        }
        let score = self.score(features);
        Ok(sign(score))
    }

    fn n_updates(&self) -> usize {
        self.n_updates
    }

    fn weights(&self) -> &[f64] {
        &self.weights
    }
}

// ---------------------------------------------------------------------------
// Batch evaluation helper
// ---------------------------------------------------------------------------

/// Evaluate an online learner sequentially on a dataset.
///
/// When `train = true`, each sample is used to update the model (prequential
/// evaluation: predict first, then learn). When `train = false`, only
/// predictions are made and the model is not updated.
///
/// Returns `(predictions, stats)` where `predictions[i]` is the prediction
/// made *before* any update on sample `i`.
pub fn online_evaluate(
    learner: &mut dyn OnlineLearner,
    data: &[(Vec<f64>, f64)],
    train: bool,
) -> Result<(Vec<f64>, OnlineStats), OnlineError> {
    let mut predictions = Vec::with_capacity(data.len());
    let mut stats = OnlineStats::default();

    for (features, label) in data {
        let pred = learner.predict(features)?;
        predictions.push(pred);

        if train {
            let result = learner.update(features, *label)?;
            stats.update(&result);
        } else {
            // Still track prediction quality without updating.
            let was_mistake = sign(pred) != sign(*label);
            let pseudo_result = OnlineUpdateResult {
                loss: 0.0,
                weight_delta_norm: 0.0,
                was_mistake,
            };
            stats.update(&pseudo_result);
        }
    }

    Ok((predictions, stats))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper utilities
    // -----------------------------------------------------------------------

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -----------------------------------------------------------------------
    // Perceptron tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_perceptron_zero_init() {
        let p = Perceptron::new(4);
        assert_eq!(p.weights(), &[0.0_f64; 4]);
        assert_eq!(p.bias(), 0.0);
        assert_eq!(p.n_updates(), 0);
    }

    #[test]
    fn test_perceptron_update_on_mistake_positive() {
        // y = +1, w·x = 0 → score=0 → sign=0 → mistake
        let mut p = Perceptron::new(2).with_learning_rate(1.0);
        let x = vec![1.0, 0.5];
        let result = p.update(&x, 1.0).expect("update failed");
        assert!(result.was_mistake);
        // w should now be [1.0, 0.5]
        assert!(approx_eq(p.weights()[0], 1.0, 1e-10));
        assert!(approx_eq(p.weights()[1], 0.5, 1e-10));
        assert!(approx_eq(p.bias(), 1.0, 1e-10));
    }

    #[test]
    fn test_perceptron_no_update_on_correct() {
        // Initialise with weights already correct for +1.
        let mut p = Perceptron::new(2);
        // Manually nudge weights so that x=[1,0] gets score > 0.
        let x = vec![1.0, 0.0];
        // First update creates a weight for y=+1.
        p.update(&x, 1.0).expect("update");
        let w_after_first = p.weights().to_vec();
        // Second update: w·x = 1 > 0, so sign = +1 = y → no update.
        p.update(&x, 1.0).expect("update");
        assert_eq!(p.weights(), w_after_first.as_slice());
    }

    #[test]
    fn test_perceptron_linearly_separable_2d() {
        // 2D points: (+1 when x[0]>0, else -1).
        let data: Vec<(Vec<f64>, f64)> = vec![
            (vec![1.0, 0.2], 1.0),
            (vec![-1.0, 0.3], -1.0),
            (vec![2.0, -0.5], 1.0),
            (vec![-2.0, 0.1], -1.0),
            (vec![0.5, 0.5], 1.0),
            (vec![-0.5, -0.5], -1.0),
            (vec![1.5, -0.1], 1.0),
            (vec![-1.5, 0.4], -1.0),
            (vec![0.8, 0.0], 1.0),
            (vec![-0.8, 0.2], -1.0),
        ];
        let mut p = Perceptron::new(2);
        for _ in 0..20 {
            for (x, y) in &data {
                p.update(x, *y).expect("update");
            }
        }
        // After convergence every point must be correct.
        for (x, y) in &data {
            let pred = p.predict(x).expect("predict");
            assert_eq!(pred, *y, "misclassified {:?} (label {})", x, y);
        }
    }

    #[test]
    fn test_perceptron_n_updates_increments() {
        let mut p = Perceptron::new(2);
        for i in 0..5 {
            p.update(&[1.0, -1.0], 1.0).expect("update");
            assert_eq!(p.n_updates(), i + 1);
        }
    }

    #[test]
    fn test_perceptron_dimension_mismatch() {
        let mut p = Perceptron::new(3);
        let err = p.update(&[1.0, 2.0], 1.0);
        assert!(matches!(
            err,
            Err(OnlineError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    // -----------------------------------------------------------------------
    // Passive-Aggressive tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pa_tau_basic() {
        // PA variant: τ = loss / ||x||²
        // Start from w=0, x=[1,0], y=+1 → score=0, loss = max(0,1-0)=1, ||x||²=1 → τ=1
        let mut pa = PassiveAggressive::new(2, PAVariant::PA);
        let result = pa.update(&[1.0, 0.0], 1.0).expect("update");
        assert!(approx_eq(result.loss, 1.0, 1e-10));
        // w = τ·y·x = 1·1·[1,0] = [1,0]
        assert!(approx_eq(pa.weights()[0], 1.0, 1e-10));
    }

    #[test]
    fn test_pa1_tau_clamped() {
        // PA-I: τ = min(C, loss/||x||²).  Set C=0.3 so τ should be clamped.
        let mut pa = PassiveAggressive::new(2, PAVariant::PAI)
            .with_aggressiveness(0.3)
            .expect("valid C");
        // w=0, x=[1,0], y=+1 → loss=1, ||x||²=1 → unclamped τ=1.0 > C=0.3 → τ=0.3
        let _r = pa.update(&[1.0, 0.0], 1.0).expect("update");
        assert!(approx_eq(pa.weights()[0], 0.3, 1e-10));
    }

    #[test]
    fn test_pa2_tau_formula() {
        // PA-II: τ = loss / (||x||² + 1/(2C)), C=1.0 → denom = 1 + 0.5 = 1.5 → τ=1/1.5
        let mut pa = PassiveAggressive::new(2, PAVariant::PAII)
            .with_aggressiveness(1.0)
            .expect("valid C");
        let _r = pa.update(&[1.0, 0.0], 1.0).expect("update");
        let expected_tau = 1.0 / 1.5;
        assert!(
            approx_eq(pa.weights()[0], expected_tau, 1e-10),
            "expected {expected_tau}, got {}",
            pa.weights()[0]
        );
    }

    #[test]
    fn test_pa_negative_c_returns_err() {
        let res = PassiveAggressive::new(2, PAVariant::PA).with_aggressiveness(-1.0);
        assert!(res.is_err());
    }

    #[test]
    fn test_pa_dimension_mismatch() {
        let mut pa = PassiveAggressive::new(3, PAVariant::PA);
        let err = pa.update(&[1.0], 1.0);
        assert!(matches!(
            err,
            Err(OnlineError::DimensionMismatch {
                expected: 3,
                got: 1
            })
        ));
    }

    // -----------------------------------------------------------------------
    // OGD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ogd_squared_loss_gradient() {
        // w=0, b=0, x=[2.0,0.0], y=3.0 → pred=0, loss=½·9=4.5, grad=(-3)·x → delta = +η·3·x
        let mut ogd = OnlineGradientDescent::new(2, OGDLoss::Squared).with_lr(0.1);
        let result = ogd.update(&[2.0, 0.0], 3.0).expect("update");
        // loss = ½(0-3)² = 4.5
        assert!(approx_eq(result.loss, 4.5, 1e-10));
        // weight[0] += -η * (0 - 3) * 2 = +0.6
        assert!(approx_eq(ogd.weights()[0], 0.6, 1e-10));
    }

    #[test]
    fn test_ogd_hinge_no_update_when_margin_ok() {
        // Manually set up a learner where y*score ≥ 1 → no gradient.
        let mut ogd = OnlineGradientDescent::new(2, OGDLoss::Hinge).with_lr(1.0);
        // After one y=+1 update on x=[1,0], w=[0.1, 0], score for x=[1,0]=0.1 < 1 → updates happen.
        // Set up with large weights directly: use multiple updates.
        for _ in 0..20 {
            ogd.update(&[10.0, 0.0], 1.0).expect("update");
        }
        let w_before = ogd.weights().to_vec();
        // Now y·score >> 1, no gradient.
        let result = ogd.update(&[10.0, 0.0], 1.0).expect("update");
        assert_eq!(result.loss, 0.0, "expected zero hinge loss");
        assert_eq!(result.weight_delta_norm, 0.0);
        assert_eq!(ogd.weights(), w_before.as_slice());
    }

    #[test]
    fn test_ogd_lr_decay_reduces_lr() {
        // Verify that the effective learning rate decreases over time.
        // current_lr() = initial_lr / sqrt(n_updates + 1) when lr_decay > 0.
        // At t=0: lr = 1.0/√1 = 1.0
        // At t=5: lr = 1.0/√6 ≈ 0.408
        // We measure this by observing the step on a constant gradient.
        // Use a large single-feature example where the gradient is always 1.
        // Step size for the bias (not regularised) = eta * bias_grad.
        // For squared loss on x=[], label=1, score=0: grad=0-1=-1, delta_bias=eta*1.
        // We compare delta_bias at t=0 vs t=5.

        let mut ogd_decay = OnlineGradientDescent::new(1, OGDLoss::Squared)
            .with_lr(1.0)
            .with_lr_decay(1.0);

        let mut ogd_nodecay = OnlineGradientDescent::new(1, OGDLoss::Squared).with_lr(1.0);

        // Both start at w=0, b=0; use x=[0] so weight gets no gradient, only bias does.
        for _ in 0..5 {
            ogd_decay.update(&[0.0], 1.0).expect("update");
            ogd_nodecay.update(&[0.0], 1.0).expect("update");
        }
        // After 5 steps with same gradient, lr_decay model should have made smaller total progress.
        // bias_nodecay converges faster (constant lr=1.0 vs decaying).
        // Actually both converge; check that decayed model has lower bias after same # of steps.
        // For no-decay: bias → 1.0 quickly. For decay: slower convergence.
        assert!(
            ogd_decay.bias.abs() <= ogd_nodecay.bias.abs() + 1e-9,
            "decaying lr should not exceed constant lr convergence; decay_bias={}, nodecay_bias={}",
            ogd_decay.bias,
            ogd_nodecay.bias
        );

        // Verify n_updates is used to compute lr: at t=10 the lr should be < lr at t=0.
        let mut ogd = OnlineGradientDescent::new(1, OGDLoss::Squared)
            .with_lr(1.0)
            .with_lr_decay(1.0);
        // Drive n_updates to 9 (lr at t=9: 1/√10 ≈ 0.316).
        for _ in 0..9 {
            ogd.update(&[0.0], 0.0).expect("update"); // zero gradient, just increments counter
        }
        let lr_at_t9 = ogd.current_lr();
        assert!(
            lr_at_t9 < 0.5,
            "lr at t=9 should be 1/√10 ≈ 0.316, got {lr_at_t9}"
        );
        assert!(
            approx_eq(lr_at_t9, 1.0 / 10_f64.sqrt(), 1e-10),
            "expected 1/√10, got {lr_at_t9}"
        );
    }

    #[test]
    fn test_ogd_l2_penalises_large_weights() {
        // With l2_reg, the weight should be smaller after many updates than without.
        let mut ogd_no_reg = OnlineGradientDescent::new(1, OGDLoss::Squared).with_lr(0.5);
        let mut ogd_l2 = OnlineGradientDescent::new(1, OGDLoss::Squared)
            .with_lr(0.5)
            .with_l2(0.5);

        for _ in 0..30 {
            ogd_no_reg.update(&[1.0], 1.0).expect("update");
            ogd_l2.update(&[1.0], 1.0).expect("update");
        }
        // The regularised model should have smaller weights.
        assert!(
            ogd_l2.weights()[0].abs() < ogd_no_reg.weights()[0].abs(),
            "l2 reg should shrink weights; no_reg={}, l2={}",
            ogd_no_reg.weights()[0],
            ogd_l2.weights()[0]
        );
    }

    #[test]
    fn test_ogd_dimension_mismatch() {
        let mut ogd = OnlineGradientDescent::new(3, OGDLoss::Squared);
        let err = ogd.update(&[1.0, 2.0], 0.0);
        assert!(matches!(
            err,
            Err(OnlineError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    // -----------------------------------------------------------------------
    // FTRL tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ftrl_l1_sparsity() {
        // With l1 > 0, features whose accumulated gradient |z_i| ≤ l1 are zeroed.
        let mut ftrl = Ftrl::new(2).with_alpha(0.1).with_l1_l2(10.0, 0.0);
        // After one small update, z[i] should be well within l1 → w should be 0.
        ftrl.update(&[1.0, 0.0], 1.0).expect("update");
        // After just a few steps against a large l1, weights should remain zero.
        assert_eq!(ftrl.weights()[0], 0.0, "weight should be zero due to L1");
    }

    #[test]
    fn test_ftrl_adaptive_per_feature() {
        // Feature 0 appears frequently, feature 1 rarely → n_vec[0] >> n_vec[1]
        // meaning feature 0's effective lr should be smaller.
        let mut ftrl = Ftrl::new(2).with_alpha(0.1);
        for _ in 0..50 {
            ftrl.update(&[1.0, 0.0], 1.0).expect("update");
        }
        // n_vec[0] should be much larger than n_vec[1] (which stays at 0).
        assert!(ftrl.n_vec[0] > ftrl.n_vec[1]);
    }

    #[test]
    fn test_ftrl_l1_zero_l2_zero_adagrad_like() {
        // With l1=0, l2=0, FTRL reduces to a form of AdaGrad.
        // The weight update should be non-zero after enough iterations.
        let mut ftrl = Ftrl::new(1).with_alpha(1.0).with_l1_l2(0.0, 0.0);
        for _ in 0..10 {
            ftrl.update(&[1.0], 1.0).expect("update");
        }
        // With consistent positive label signals, weight should be positive.
        assert!(
            ftrl.weights()[0] > 0.0,
            "weight should be positive; got {}",
            ftrl.weights()[0]
        );
    }

    #[test]
    fn test_ftrl_dimension_mismatch() {
        let mut ftrl = Ftrl::new(3);
        let err = ftrl.update(&[1.0, 2.0], 1.0);
        assert!(matches!(
            err,
            Err(OnlineError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn test_ftrl_predict_dimension_mismatch() {
        let ftrl = Ftrl::new(3);
        let err = ftrl.predict(&[1.0]);
        assert!(matches!(
            err,
            Err(OnlineError::DimensionMismatch {
                expected: 3,
                got: 1
            })
        ));
    }

    // -----------------------------------------------------------------------
    // OnlineStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_online_stats_mistake_rate_zero_updates() {
        let stats = OnlineStats::default();
        assert_eq!(stats.mistake_rate(), 0.0);
    }

    #[test]
    fn test_online_stats_mistake_rate_computation() {
        let mut stats = OnlineStats::default();
        let mistake = OnlineUpdateResult {
            loss: 1.0,
            weight_delta_norm: 0.5,
            was_mistake: true,
        };
        let correct = OnlineUpdateResult {
            loss: 0.0,
            weight_delta_norm: 0.0,
            was_mistake: false,
        };
        stats.update(&mistake);
        stats.update(&correct);
        stats.update(&mistake);
        // 2 mistakes out of 3.
        assert!(approx_eq(stats.mistake_rate(), 2.0 / 3.0, 1e-10));
    }

    #[test]
    fn test_online_stats_cumulative_loss() {
        let mut stats = OnlineStats::default();
        for loss_val in [0.5, 1.0, 1.5] {
            let r = OnlineUpdateResult {
                loss: loss_val,
                weight_delta_norm: 0.0,
                was_mistake: false,
            };
            stats.update(&r);
        }
        assert!(approx_eq(stats.cumulative_loss, 3.0, 1e-10));
        assert!(approx_eq(stats.mean_loss, 1.0, 1e-10));
    }

    // -----------------------------------------------------------------------
    // online_evaluate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_online_evaluate_train_true_updates_model() {
        let mut p = Perceptron::new(2);
        let data = vec![(vec![1.0, 0.0], 1.0), (vec![-1.0, 0.0], -1.0)];
        let (preds, _stats) = online_evaluate(&mut p, &data, true).expect("evaluate");
        assert_eq!(preds.len(), 2);
        // After processing both samples, n_updates should be 2.
        assert_eq!(p.n_updates(), 2);
    }

    #[test]
    fn test_online_evaluate_train_false_no_update() {
        let mut p = Perceptron::new(2);
        let data = vec![(vec![1.0, 0.0], 1.0), (vec![-1.0, 0.0], -1.0)];
        let (preds, _stats) = online_evaluate(&mut p, &data, false).expect("evaluate");
        assert_eq!(preds.len(), 2);
        // No updates when train=false.
        assert_eq!(p.n_updates(), 0);
    }

    // -----------------------------------------------------------------------
    // Convergence test
    // -----------------------------------------------------------------------

    #[test]
    fn test_perceptron_converges_linearly_separable_10_samples() {
        let data: Vec<(Vec<f64>, f64)> = vec![
            (vec![2.0, 1.0], 1.0),
            (vec![1.5, 0.8], 1.0),
            (vec![1.0, 0.5], 1.0),
            (vec![0.5, 0.2], 1.0),
            (vec![0.2, 0.1], 1.0),
            (vec![-0.2, -0.1], -1.0),
            (vec![-0.5, -0.3], -1.0),
            (vec![-1.0, -0.5], -1.0),
            (vec![-1.5, -0.7], -1.0),
            (vec![-2.0, -1.0], -1.0),
        ];
        let mut p = Perceptron::new(2);
        // Run multiple passes.
        for _ in 0..50 {
            for (x, y) in &data {
                p.update(x, *y).expect("update");
            }
        }
        let mut correct = 0;
        for (x, y) in &data {
            let pred = p.predict(x).expect("predict");
            if pred == *y {
                correct += 1;
            }
        }
        assert_eq!(
            correct, 10,
            "Perceptron should converge on linearly separable data"
        );
    }

    #[test]
    fn test_pa_converges_linearly_separable() {
        let data: Vec<(Vec<f64>, f64)> = vec![
            (vec![1.0, 0.5], 1.0),
            (vec![-1.0, -0.5], -1.0),
            (vec![2.0, 1.0], 1.0),
            (vec![-2.0, -1.0], -1.0),
        ];
        let mut pa = PassiveAggressive::new(2, PAVariant::PAI)
            .with_aggressiveness(1.0)
            .expect("valid C");
        for _ in 0..30 {
            for (x, y) in &data {
                pa.update(x, *y).expect("update");
            }
        }
        for (x, y) in &data {
            let pred = pa.predict(x).expect("predict");
            assert_eq!(pred, *y);
        }
    }

    #[test]
    fn test_ogd_squared_converges_to_constant() {
        // All labels are 2.0 — OGD squared loss should drive w·x toward 2.
        let mut ogd = OnlineGradientDescent::new(1, OGDLoss::Squared).with_lr(0.3);
        let x = vec![1.0];
        for _ in 0..200 {
            ogd.update(&x, 2.0).expect("update");
        }
        let pred = ogd.predict(&x).expect("predict");
        assert!(
            approx_eq(pred, 2.0, 0.1),
            "OGD should converge near 2.0, got {pred}"
        );
    }

    #[test]
    fn test_ftrl_n_updates_increments() {
        let mut ftrl = Ftrl::new(2);
        for i in 0..7 {
            ftrl.update(&[1.0, 0.5], 1.0).expect("update");
            assert_eq!(ftrl.n_updates(), i + 1);
        }
    }

    #[test]
    fn test_online_error_display() {
        let e = OnlineError::DimensionMismatch {
            expected: 5,
            got: 3,
        };
        let s = e.to_string();
        assert!(s.contains("5") && s.contains("3"));

        let e2 = OnlineError::InvalidHyperparameter("C must be positive".to_string());
        assert!(e2.to_string().contains("C must be positive"));

        let e3 = OnlineError::NotFitted;
        assert!(e3.to_string().contains("fitted"));
    }
}

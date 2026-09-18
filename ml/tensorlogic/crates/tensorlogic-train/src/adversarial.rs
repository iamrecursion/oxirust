//! Adversarial training utilities for TensorLogic.
//!
//! Provides FGSM (Fast Gradient Sign Method), PGD (Projected Gradient Descent),
//! adversarial example generation, adversarial training loss, and robustness evaluation.
//!
//! # References
//! - Goodfellow et al. (2014): "Explaining and Harnessing Adversarial Examples" (FGSM)
//! - Madry et al. (2017): "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise during adversarial attack construction or execution.
#[derive(Debug)]
pub enum AdversarialError {
    /// Input and label dimensions did not match what the model expects.
    DimensionMismatch { expected: usize, got: usize },
    /// The epsilon (perturbation budget) is not strictly positive.
    InvalidEpsilon(f64),
    /// The per-step step-size is not strictly positive.
    InvalidStepSize(f64),
    /// The number of PGD iterations must be at least 1.
    InvalidIterations(usize),
    /// Gradient computation produced a non-finite value or other failure.
    GradientComputationFailed(String),
}

impl fmt::Display for AdversarialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdversarialError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected} but got {got}")
            }
            AdversarialError::InvalidEpsilon(e) => {
                write!(f, "epsilon must be strictly positive, got {e}")
            }
            AdversarialError::InvalidStepSize(s) => {
                write!(f, "step_size must be strictly positive, got {s}")
            }
            AdversarialError::InvalidIterations(n) => write!(f, "n_steps must be >= 1, got {n}"),
            AdversarialError::GradientComputationFailed(msg) => {
                write!(f, "gradient computation failed: {msg}")
            }
        }
    }
}

impl std::error::Error for AdversarialError {}

// ─────────────────────────────────────────────────────────────────────────────
// Norm type
// ─────────────────────────────────────────────────────────────────────────────

/// The norm used to measure and project the adversarial perturbation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerturbNorm {
    /// L∞ constraint: max |δᵢ| ≤ ε.
    LInf,
    /// L2 constraint: ‖δ‖₂ ≤ ε.
    L2,
    /// L1 constraint: ‖δ‖₁ ≤ ε.
    L1,
}

// ─────────────────────────────────────────────────────────────────────────────
// AdversarialExample
// ─────────────────────────────────────────────────────────────────────────────

/// The result of running an adversarial attack on a single input.
#[derive(Debug, Clone)]
pub struct AdversarialExample {
    /// The clean (unperturbed) input.
    pub original: Vec<f64>,
    /// The perturbed input `original + perturbation`.
    pub perturbed: Vec<f64>,
    /// The additive perturbation δ = perturbed − original.
    pub perturbation: Vec<f64>,
    /// The actual norm of the perturbation (measured in the configured norm).
    pub perturbation_norm: f64,
    /// Number of attack iterations performed (1 for FGSM).
    pub n_iterations: usize,
}

impl AdversarialExample {
    /// L∞ norm of the perturbation: max |δᵢ|.
    pub fn perturbation_linf(&self) -> f64 {
        self.perturbation
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
    }

    /// L2 norm of the perturbation: √(Σ δᵢ²).
    pub fn perturbation_l2(&self) -> f64 {
        self.perturbation.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// L1 norm of the perturbation: Σ |δᵢ|.
    pub fn perturbation_l1(&self) -> f64 {
        self.perturbation.iter().map(|v| v.abs()).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AttackLoss trait
// ─────────────────────────────────────────────────────────────────────────────

/// A differentiable loss function used by attack algorithms.
///
/// Both `loss` and `grad` receive raw model outputs (logits or probabilities)
/// and target labels, and must be thread-safe.
pub trait AttackLoss: Send + Sync {
    /// Compute the scalar loss value.
    fn loss(&self, predictions: &[f64], labels: &[f64]) -> f64;

    /// Compute the gradient of the loss with respect to `predictions`.
    fn grad(&self, predictions: &[f64], labels: &[f64]) -> Vec<f64>;
}

// ─────────────────────────────────────────────────────────────────────────────
// CrossEntropyAttackLoss
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-entropy loss for multi-class classification attacks.
///
/// Applies softmax internally:
/// - loss = −Σ yᵢ · log(softmax(zᵢ) + ε)
/// - grad = softmax(z) − y
pub struct CrossEntropyAttackLoss;

impl CrossEntropyAttackLoss {
    fn softmax(logits: &[f64]) -> Vec<f64> {
        let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = logits.iter().map(|&z| (z - max_val).exp()).collect();
        let sum: f64 = exp.iter().sum();
        if sum == 0.0 {
            vec![1.0 / logits.len() as f64; logits.len()]
        } else {
            exp.iter().map(|&e| e / sum).collect()
        }
    }
}

impl AttackLoss for CrossEntropyAttackLoss {
    fn loss(&self, predictions: &[f64], labels: &[f64]) -> f64 {
        let probs = Self::softmax(predictions);
        const EPS: f64 = 1e-12;
        -probs
            .iter()
            .zip(labels.iter())
            .map(|(&p, &y)| y * (p + EPS).ln())
            .sum::<f64>()
    }

    fn grad(&self, predictions: &[f64], labels: &[f64]) -> Vec<f64> {
        let probs = Self::softmax(predictions);
        probs
            .iter()
            .zip(labels.iter())
            .map(|(&p, &y)| p - y)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MseAttackLoss
// ─────────────────────────────────────────────────────────────────────────────

/// Mean-squared-error loss for regression attacks.
///
/// - loss = mean((predictions − labels)²)
/// - grad = 2 · (predictions − labels) / n
pub struct MseAttackLoss;

impl AttackLoss for MseAttackLoss {
    fn loss(&self, predictions: &[f64], labels: &[f64]) -> f64 {
        let n = predictions.len() as f64;
        predictions
            .iter()
            .zip(labels.iter())
            .map(|(&p, &y)| (p - y).powi(2))
            .sum::<f64>()
            / n
    }

    fn grad(&self, predictions: &[f64], labels: &[f64]) -> Vec<f64> {
        let n = predictions.len() as f64;
        predictions
            .iter()
            .zip(labels.iter())
            .map(|(&p, &y)| 2.0 * (p - y) / n)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AttackModel trait
// ─────────────────────────────────────────────────────────────────────────────

/// A model that can be attacked.
///
/// Implementors provide a forward pass; `input_gradient` has a default finite-
/// difference implementation that can be overridden for efficiency.
pub trait AttackModel: Send + Sync {
    /// Forward pass: given an input slice, return predictions (logits or probs).
    fn forward(&self, input: &[f64]) -> Vec<f64>;

    /// Gradient of the scalar `output_grad · forward(input)` w.r.t. input,
    /// via reverse-mode chain rule if available, otherwise via finite differences.
    ///
    /// `output_grad` has the same length as `forward(input)`.
    fn input_gradient(&self, input: &[f64], output_grad: &[f64]) -> Vec<f64> {
        // Default: forward-mode finite differences
        const H: f64 = 1e-5;
        let mut grad_in = vec![0.0_f64; input.len()];
        let mut x_plus = input.to_vec();
        let mut x_minus = input.to_vec();
        for i in 0..input.len() {
            x_plus[i] = input[i] + H;
            x_minus[i] = input[i] - H;
            let f_plus = self.forward(&x_plus);
            let f_minus = self.forward(&x_minus);
            grad_in[i] = f_plus
                .iter()
                .zip(f_minus.iter())
                .zip(output_grad.iter())
                .map(|((&fp, &fm), &g)| g * (fp - fm) / (2.0 * H))
                .sum::<f64>();
            x_plus[i] = input[i];
            x_minus[i] = input[i];
        }
        grad_in
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LinearAttackModel
// ─────────────────────────────────────────────────────────────────────────────

/// A simple linear model `f(x) = W·x + b` used primarily for testing attacks.
pub struct LinearAttackModel {
    /// Weight matrix: `weights[i]` is the i-th output row (length = n_inputs).
    pub weights: Vec<Vec<f64>>,
    /// Bias vector (length = n_outputs).
    pub bias: Vec<f64>,
}

impl LinearAttackModel {
    /// Construct a new linear model, validating that all rows have the same length.
    pub fn new(weights: Vec<Vec<f64>>, bias: Vec<f64>) -> Result<Self, AdversarialError> {
        if weights.is_empty() || bias.is_empty() {
            return Err(AdversarialError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if weights.len() != bias.len() {
            return Err(AdversarialError::DimensionMismatch {
                expected: weights.len(),
                got: bias.len(),
            });
        }
        let n_inputs = weights[0].len();
        for (i, row) in weights.iter().enumerate() {
            if row.len() != n_inputs {
                return Err(AdversarialError::DimensionMismatch {
                    expected: n_inputs,
                    got: row.len(),
                });
            }
            let _ = i; // suppress unused warning
        }
        Ok(Self { weights, bias })
    }
}

impl AttackModel for LinearAttackModel {
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| {
                row.iter()
                    .zip(input.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>()
                    + b
            })
            .collect()
    }

    /// Exact analytical gradient for a linear model: ∂(g·Wx)/∂x = Wᵀ·g.
    fn input_gradient(&self, _input: &[f64], output_grad: &[f64]) -> Vec<f64> {
        let n_inputs = self.weights[0].len();
        let mut grad = vec![0.0_f64; n_inputs];
        for (row, &g) in self.weights.iter().zip(output_grad.iter()) {
            for (j, &w) in row.iter().enumerate() {
                grad[j] += w * g;
            }
        }
        grad
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AttackConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an adversarial attack.
#[derive(Debug, Clone)]
pub struct AttackConfig {
    /// Maximum allowed perturbation magnitude (ε > 0).
    pub epsilon: f64,
    /// Norm used to constrain the perturbation.
    pub norm: PerturbNorm,
    /// Number of iterative steps (used by PGD; FGSM uses 1).
    pub n_steps: usize,
    /// Per-step size α.  Defaults to `epsilon / 4.0`.
    pub step_size: f64,
    /// If true (PGD), initialise with a random perturbation inside the ε-ball.
    pub random_start: bool,
    /// Minimum allowed value for the perturbed input.
    pub clip_min: f64,
    /// Maximum allowed value for the perturbed input.
    pub clip_max: f64,
}

impl AttackConfig {
    /// Create a new config with `epsilon` as the perturbation budget.
    ///
    /// Defaults: L∞ norm, 40 PGD steps, step_size = ε/4, no random start,
    /// no input clipping.
    pub fn new(epsilon: f64) -> Result<Self, AdversarialError> {
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(AdversarialError::InvalidEpsilon(epsilon));
        }
        Ok(Self {
            epsilon,
            norm: PerturbNorm::LInf,
            n_steps: 40,
            step_size: epsilon / 4.0,
            random_start: false,
            clip_min: f64::NEG_INFINITY,
            clip_max: f64::INFINITY,
        })
    }

    /// Override the perturbation norm.
    pub fn with_norm(mut self, norm: PerturbNorm) -> Self {
        self.norm = norm;
        self
    }

    /// Override the number of PGD steps.  Must be ≥ 1.
    pub fn with_steps(mut self, n: usize) -> Result<Self, AdversarialError> {
        if n == 0 {
            return Err(AdversarialError::InvalidIterations(n));
        }
        self.n_steps = n;
        Ok(self)
    }

    /// Override the per-step size.  Must be strictly positive.
    pub fn with_step_size(mut self, s: f64) -> Result<Self, AdversarialError> {
        if s <= 0.0 || !s.is_finite() {
            return Err(AdversarialError::InvalidStepSize(s));
        }
        self.step_size = s;
        Ok(self)
    }

    /// Enable or disable random initialisation of the perturbation.
    pub fn with_random_start(mut self, b: bool) -> Self {
        self.random_start = b;
        self
    }

    /// Set the input clipping range [min, max].
    pub fn with_clip(mut self, min: f64, max: f64) -> Self {
        self.clip_min = min;
        self.clip_max = max;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdversarialTrainStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics collected during adversarial training over a batch.
#[derive(Debug, Default, Clone)]
pub struct AdversarialTrainStats {
    /// Number of samples processed.
    pub n_samples: usize,
    /// Average L∞ (or configured-norm) magnitude of the adversarial perturbations.
    pub mean_perturbation_norm: f64,
    /// Mean clean loss across the batch.
    pub clean_loss: f64,
    /// Mean adversarial loss across the batch.
    pub adversarial_loss: f64,
    /// Combined loss: α · clean + (1−α) · adversarial.
    pub combined_loss: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Project `perturbation` onto the L∞ ball of radius `epsilon`.
///
/// Each component is clamped independently to [−ε, ε].
pub fn project_linf(perturbation: &[f64], epsilon: f64) -> Vec<f64> {
    perturbation
        .iter()
        .map(|&d| d.clamp(-epsilon, epsilon))
        .collect()
}

/// Project `perturbation` onto the L2 ball of radius `epsilon`.
///
/// If ‖δ‖₂ > ε, the vector is scaled down to have norm exactly ε.
pub fn project_l2(perturbation: &[f64], epsilon: f64) -> Vec<f64> {
    let norm: f64 = perturbation.iter().map(|&d| d * d).sum::<f64>().sqrt();
    if norm <= epsilon || norm == 0.0 {
        perturbation.to_vec()
    } else {
        perturbation.iter().map(|&d| d * epsilon / norm).collect()
    }
}

/// Project `perturbation` onto the L1 ball of radius `epsilon`.
///
/// Uses the classic Duchi et al. (2008) algorithm via sorting of absolute values.
pub fn project_l1(perturbation: &[f64], epsilon: f64) -> Vec<f64> {
    let l1: f64 = perturbation.iter().map(|&d| d.abs()).sum();
    if l1 <= epsilon {
        return perturbation.to_vec();
    }
    // Compute the soft-threshold via the simplex projection on |δ|/l1.
    let n = perturbation.len();
    let mut abs_sorted: Vec<f64> = perturbation.iter().map(|&d| d.abs()).collect();
    abs_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0_f64;
    let mut rho = 0_usize;
    for (i, &v) in abs_sorted.iter().enumerate() {
        cumsum += v;
        if v > (cumsum - epsilon) / (i as f64 + 1.0) {
            rho = i;
        }
    }
    let cumsum_rho: f64 = abs_sorted[..=rho].iter().sum();
    let theta = (cumsum_rho - epsilon) / (rho as f64 + 1.0);

    (0..n)
        .map(|i| {
            let sign = if perturbation[i] >= 0.0 { 1.0 } else { -1.0 };
            sign * (perturbation[i].abs() - theta).max(0.0)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute ∇_x L(f(x), y) = J_x^T · ∇_z L(f(x), y).
fn loss_input_gradient(
    model: &dyn AttackModel,
    loss: &dyn AttackLoss,
    input: &[f64],
    labels: &[f64],
) -> Result<Vec<f64>, AdversarialError> {
    let predictions = model.forward(input);
    let loss_grad = loss.grad(&predictions, labels); // ∂L/∂z
    let input_grad = model.input_gradient(input, &loss_grad); // ∂L/∂x

    // Validate that all values are finite.
    for &g in &input_grad {
        if !g.is_finite() {
            return Err(AdversarialError::GradientComputationFailed(
                "non-finite value in input gradient".to_string(),
            ));
        }
    }
    Ok(input_grad)
}

/// Clip `x` component-wise to the configured [clip_min, clip_max] range.
#[inline]
fn clip_input(x: &[f64], config: &AttackConfig) -> Vec<f64> {
    x.iter()
        .map(|&v| v.clamp(config.clip_min, config.clip_max))
        .collect()
}

/// Project a perturbation δ onto the ε-ball determined by the configured norm.
fn project(perturbation: &[f64], config: &AttackConfig) -> Vec<f64> {
    match config.norm {
        PerturbNorm::LInf => project_linf(perturbation, config.epsilon),
        PerturbNorm::L2 => project_l2(perturbation, config.epsilon),
        PerturbNorm::L1 => project_l1(perturbation, config.epsilon),
    }
}

/// Measure the norm of `perturbation` under the configured `norm`.
fn measure_norm(perturbation: &[f64], norm: PerturbNorm) -> f64 {
    match norm {
        PerturbNorm::LInf => perturbation
            .iter()
            .map(|&d| d.abs())
            .fold(0.0_f64, f64::max),
        PerturbNorm::L2 => perturbation.iter().map(|&d| d * d).sum::<f64>().sqrt(),
        PerturbNorm::L1 => perturbation.iter().map(|&d| d.abs()).sum(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal LCG PRNG (no external rand dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// A simple 64-bit LCG (Knuth's constants) used only for `random_start`.
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state.
        Self {
            state: if seed == 0 { 0xdeadbeef_cafebabe } else { seed },
        }
    }

    /// Advance and return the next u64.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a uniform f64 in (−1, 1).
    fn next_f64_signed(&mut self) -> f64 {
        // Map u64 to [0, 1) then shift to (−1, 1).
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        u * 2.0 - 1.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FGSM
// ─────────────────────────────────────────────────────────────────────────────

/// Fast Gradient Sign Method (Goodfellow et al., 2014).
///
/// Computes a single-step adversarial perturbation:
///
/// - L∞: δ = ε · sign(∇_x L)
/// - L2:  δ = ε · ∇_x L / ‖∇_x L‖₂
/// - L1:  δ = ε · e_k  where k = argmax |∂L/∂xᵢ|
pub fn fgsm(
    model: &dyn AttackModel,
    loss: &dyn AttackLoss,
    input: &[f64],
    labels: &[f64],
    config: &AttackConfig,
) -> Result<AdversarialExample, AdversarialError> {
    let grad = loss_input_gradient(model, loss, input, labels)?;

    let raw_delta: Vec<f64> = match config.norm {
        PerturbNorm::LInf => grad
            .iter()
            .map(|&g| {
                if g == 0.0 {
                    0.0
                } else {
                    config.epsilon * g.signum()
                }
            })
            .collect(),
        PerturbNorm::L2 => {
            let norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
            if norm < 1e-12 {
                vec![0.0; grad.len()]
            } else {
                grad.iter().map(|&g| config.epsilon * g / norm).collect()
            }
        }
        PerturbNorm::L1 => {
            // Largest-coordinate attack: unit vector in the direction of max |gradient|.
            let argmax = grad
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.abs()
                        .partial_cmp(&b.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut delta = vec![0.0_f64; grad.len()];
            delta[argmax] = config.epsilon * grad[argmax].signum();
            delta
        }
    };

    let perturbed_raw: Vec<f64> = input
        .iter()
        .zip(raw_delta.iter())
        .map(|(&x, &d)| x + d)
        .collect();
    let perturbed = clip_input(&perturbed_raw, config);

    let perturbation: Vec<f64> = perturbed
        .iter()
        .zip(input.iter())
        .map(|(&p, &x)| p - x)
        .collect();

    let perturbation_norm = measure_norm(&perturbation, config.norm);

    Ok(AdversarialExample {
        original: input.to_vec(),
        perturbed,
        perturbation,
        perturbation_norm,
        n_iterations: 1,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// PGD
// ─────────────────────────────────────────────────────────────────────────────

/// Projected Gradient Descent (Madry et al., 2017).
///
/// Iterative attack with optional random initialisation:
///
/// ```text
/// x₀ = x + Uniform(−ε, ε)  [if random_start]
/// xₜ₊₁ = Proj_{Bε(x)}( clip( xₜ + α · step_direction ) )
/// ```
///
/// Step direction:
/// - L∞: sign(∇_x L)
/// - L2:  ∇_x L / ‖∇_x L‖₂
/// - L1:  argmax-coordinate (greedy Frank-Wolfe step)
///
/// `seed` is used only when `config.random_start = true`.
pub fn pgd(
    model: &dyn AttackModel,
    loss: &dyn AttackLoss,
    input: &[f64],
    labels: &[f64],
    config: &AttackConfig,
    seed: u64,
) -> Result<AdversarialExample, AdversarialError> {
    let n = input.len();
    let mut rng = Lcg64::new(seed);

    // Initialise δ.
    let mut delta: Vec<f64> = if config.random_start {
        let raw: Vec<f64> = (0..n)
            .map(|_| rng.next_f64_signed() * config.epsilon)
            .collect();
        project(&raw, config)
    } else {
        vec![0.0_f64; n]
    };

    for _ in 0..config.n_steps {
        // Construct current adversarial input.
        let x_adv: Vec<f64> = input
            .iter()
            .zip(delta.iter())
            .map(|(&x, &d)| x + d)
            .collect();
        let x_adv = clip_input(&x_adv, config);

        let grad = loss_input_gradient(model, loss, &x_adv, labels)?;

        // Compute step direction.
        let step: Vec<f64> = match config.norm {
            PerturbNorm::LInf => grad
                .iter()
                .map(|&g| {
                    if g == 0.0 {
                        0.0
                    } else {
                        config.step_size * g.signum()
                    }
                })
                .collect(),
            PerturbNorm::L2 => {
                let norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
                if norm < 1e-12 {
                    vec![0.0; n]
                } else {
                    grad.iter().map(|&g| config.step_size * g / norm).collect()
                }
            }
            PerturbNorm::L1 => {
                let argmax = grad
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.abs()
                            .partial_cmp(&b.abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let mut s = vec![0.0_f64; n];
                s[argmax] = config.step_size * grad[argmax].signum();
                s
            }
        };

        // Update δ and project back onto the ε-ball.
        let new_x_adv: Vec<f64> = input
            .iter()
            .zip(delta.iter())
            .zip(step.iter())
            .map(|((&x, &d), &s)| x + d + s)
            .collect();
        let new_x_adv = clip_input(&new_x_adv, config);

        delta = new_x_adv
            .iter()
            .zip(input.iter())
            .map(|(&xa, &x)| xa - x)
            .collect();
        delta = project(&delta, config);
    }

    let perturbed: Vec<f64> = input
        .iter()
        .zip(delta.iter())
        .map(|(&x, &d)| (x + d).clamp(config.clip_min, config.clip_max))
        .collect();

    let perturbation: Vec<f64> = perturbed
        .iter()
        .zip(input.iter())
        .map(|(&p, &x)| p - x)
        .collect();

    let perturbation_norm = measure_norm(&perturbation, config.norm);

    Ok(AdversarialExample {
        original: input.to_vec(),
        perturbed,
        perturbation,
        perturbation_norm,
        n_iterations: config.n_steps,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Adversarial training loss
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the combined adversarial training loss over a batch:
///
/// ```text
/// L = α · L_clean(x, y)  +  (1−α) · L_adv(x+δ*, y)
/// ```
///
/// where δ* is the PGD adversarial perturbation for each sample.
///
/// Returns the combined scalar loss and per-batch statistics.
pub fn adversarial_training_loss(
    model: &dyn AttackModel,
    loss: &dyn AttackLoss,
    inputs: &[Vec<f64>],
    labels: &[Vec<f64>],
    config: &AttackConfig,
    alpha: f64,
    seed: u64,
) -> Result<(f64, AdversarialTrainStats), AdversarialError> {
    if inputs.is_empty() {
        return Ok((0.0, AdversarialTrainStats::default()));
    }
    if inputs.len() != labels.len() {
        return Err(AdversarialError::DimensionMismatch {
            expected: inputs.len(),
            got: labels.len(),
        });
    }

    let mut total_clean = 0.0_f64;
    let mut total_adv = 0.0_f64;
    let mut total_norm = 0.0_f64;
    let n = inputs.len();

    for (i, (x, y)) in inputs.iter().zip(labels.iter()).enumerate() {
        // Clean loss.
        let preds_clean = model.forward(x);
        total_clean += loss.loss(&preds_clean, y);

        // PGD adversarial example — vary seed per sample to avoid correlation.
        let sample_seed = seed.wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let adv_ex = pgd(model, loss, x, y, config, sample_seed)?;
        let preds_adv = model.forward(&adv_ex.perturbed);
        total_adv += loss.loss(&preds_adv, y);
        total_norm += adv_ex.perturbation_norm;
    }

    let mean_clean = total_clean / n as f64;
    let mean_adv = total_adv / n as f64;
    let combined = alpha * mean_clean + (1.0 - alpha) * mean_adv;

    let stats = AdversarialTrainStats {
        n_samples: n,
        mean_perturbation_norm: total_norm / n as f64,
        clean_loss: mean_clean,
        adversarial_loss: mean_adv,
        combined_loss: combined,
    };

    Ok((combined, stats))
}

// ─────────────────────────────────────────────────────────────────────────────
// Robustness evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the model's adversarial robustness on a set of samples.
///
/// For each sample the PGD attack is run; a sample is considered "robust" if
/// the argmax prediction does not change after the attack (for classification),
/// or equivalently if the adversarial loss is not greater than the clean loss
/// (for regression).
///
/// Returns the fraction of samples that remain correctly classified (robust),
/// in the range \[0, 1\].
pub fn robustness_eval(
    model: &dyn AttackModel,
    inputs: &[Vec<f64>],
    labels: &[Vec<f64>],
    config: &AttackConfig,
    seed: u64,
) -> Result<f64, AdversarialError> {
    if inputs.is_empty() {
        return Ok(1.0);
    }
    if inputs.len() != labels.len() {
        return Err(AdversarialError::DimensionMismatch {
            expected: inputs.len(),
            got: labels.len(),
        });
    }

    let mut robust_count = 0_usize;
    let n = inputs.len();

    for (i, (x, y)) in inputs.iter().zip(labels.iter()).enumerate() {
        let clean_preds = model.forward(x);
        let clean_argmax = argmax_vec(&clean_preds);
        let label_argmax = argmax_vec(y);

        // Only count samples that are correctly classified before the attack.
        if clean_argmax != label_argmax {
            // Misclassified even on clean input — not robust by definition.
            continue;
        }

        let sample_seed = seed.wrapping_add((i as u64).wrapping_mul(0x6c62272e07bb0142));
        let adv_ex = pgd(model, loss_for_eval(), x, y, config, sample_seed)?;
        let adv_preds = model.forward(&adv_ex.perturbed);
        let adv_argmax = argmax_vec(&adv_preds);

        if adv_argmax == clean_argmax {
            robust_count += 1;
        }
    }

    Ok(robust_count as f64 / n as f64)
}

/// Internal: build a cross-entropy loss instance for robustness evaluation.
fn loss_for_eval() -> &'static CrossEntropyAttackLoss {
    static LOSS: CrossEntropyAttackLoss = CrossEntropyAttackLoss;
    &LOSS
}

/// Return the index of the maximum element in `v`.
fn argmax_vec(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── Helpers ────────────────────────────────────────────────────────────────

    /// 2-class linear model with weights [[1,0],[0,1]] and zero bias.
    fn identity_model_2x2() -> LinearAttackModel {
        LinearAttackModel::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0.0, 0.0])
            .expect("valid model")
    }

    fn default_config() -> AttackConfig {
        AttackConfig::new(0.1).expect("valid epsilon")
    }

    // ── FGSM tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_fgsm_linf_norm_le_epsilon() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.5, 0.5];
        let labels = vec![1.0, 0.0];
        let config = default_config();
        let ex = fgsm(&model, &loss, &input, &labels, &config).expect("fgsm ok");
        assert!(ex.perturbation_linf() <= config.epsilon + 1e-10);
    }

    #[test]
    fn test_fgsm_changes_input_when_gradient_nonzero() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.5, 0.3];
        let labels = vec![1.0, 0.0]; // gradient is non-zero
        let config = default_config();
        let ex = fgsm(&model, &loss, &input, &labels, &config).expect("fgsm ok");
        let norm: f64 = ex.perturbation.iter().map(|&d| d * d).sum::<f64>().sqrt();
        assert!(norm > 1e-10, "perturbation should be non-zero");
    }

    #[test]
    fn test_fgsm_zero_gradient_produces_zero_perturbation() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        // labels == predictions → MSE grad = 0
        let input = vec![0.5, 0.5];
        let labels = vec![0.5, 0.5];
        let config = default_config();
        let ex = fgsm(&model, &loss, &input, &labels, &config).expect("fgsm ok");
        assert_abs_diff_eq!(ex.perturbation_linf(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_fgsm_all_dims_within_epsilon() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.2, 0.8];
        let labels = vec![0.0, 1.0];
        let config = AttackConfig::new(0.05).expect("ok");
        let ex = fgsm(&model, &loss, &input, &labels, &config).expect("fgsm ok");
        for &d in &ex.perturbation {
            assert!(d.abs() <= 0.05 + 1e-10, "component {d} exceeds epsilon");
        }
    }

    // ── PGD tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_pgd_linf_norm_le_epsilon() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.5, 0.5];
        let labels = vec![1.0, 0.0];
        let config = default_config();
        let ex = pgd(&model, &loss, &input, &labels, &config, 42).expect("pgd ok");
        assert!(ex.perturbation_linf() <= config.epsilon + 1e-10);
    }

    #[test]
    fn test_pgd_n_steps_1_matches_fgsm_linf() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.3, 0.7];
        let labels = vec![1.0, 0.0];
        let eps = 0.1_f64;
        // Both should produce the same perturbation for a linear model (one step).
        let config_fgsm = AttackConfig::new(eps)
            .expect("ok")
            .with_step_size(eps)
            .expect("ok")
            .with_steps(1)
            .expect("ok");
        let config_pgd = config_fgsm.clone();
        let ex_fgsm = fgsm(&model, &loss, &input, &labels, &config_fgsm).expect("ok");
        let ex_pgd = pgd(&model, &loss, &input, &labels, &config_pgd, 0).expect("ok");
        for (df, dp) in ex_fgsm.perturbation.iter().zip(ex_pgd.perturbation.iter()) {
            assert_abs_diff_eq!(df, dp, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_pgd_iterates_more_than_one() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.5, 0.5];
        let labels = vec![1.0, 0.0];
        let config = AttackConfig::new(0.1)
            .expect("ok")
            .with_steps(10)
            .expect("ok");
        let ex = pgd(&model, &loss, &input, &labels, &config, 7).expect("ok");
        assert_eq!(ex.n_iterations, 10);
    }

    // ── Projection tests ───────────────────────────────────────────────────────

    #[test]
    fn test_project_linf_clamps_each_dim() {
        let delta = vec![0.2, -0.3, 0.05, -0.01];
        let eps = 0.1;
        let proj = project_linf(&delta, eps);
        for &d in &proj {
            assert!(d >= -eps - 1e-12 && d <= eps + 1e-12);
        }
        assert_abs_diff_eq!(proj[0], 0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(proj[1], -0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(proj[2], 0.05, epsilon = 1e-10);
    }

    #[test]
    fn test_project_l2_result_within_epsilon() {
        let delta = vec![0.3, 0.4]; // norm = 0.5
        let eps = 0.1;
        let proj = project_l2(&delta, eps);
        let norm: f64 = proj.iter().map(|&d| d * d).sum::<f64>().sqrt();
        assert!(norm <= eps + 1e-10, "L2 norm {norm} exceeds epsilon {eps}");
    }

    #[test]
    fn test_project_l2_identity_when_within_ball() {
        let delta = vec![0.03, 0.04]; // norm = 0.05 < 0.1
        let eps = 0.1;
        let proj = project_l2(&delta, eps);
        assert_abs_diff_eq!(proj[0], 0.03, epsilon = 1e-10);
        assert_abs_diff_eq!(proj[1], 0.04, epsilon = 1e-10);
    }

    // ── CrossEntropyAttackLoss tests ───────────────────────────────────────────

    #[test]
    fn test_cross_entropy_grad_finite_difference() {
        let ce = CrossEntropyAttackLoss;
        let preds = vec![1.0, 0.5, -0.5];
        let labels = vec![1.0, 0.0, 0.0];
        let grad = ce.grad(&preds, &labels);
        let h = 1e-5_f64;
        for i in 0..preds.len() {
            let mut p_plus = preds.clone();
            let mut p_minus = preds.clone();
            p_plus[i] += h;
            p_minus[i] -= h;
            let fd = (ce.loss(&p_plus, &labels) - ce.loss(&p_minus, &labels)) / (2.0 * h);
            assert_abs_diff_eq!(grad[i], fd, epsilon = 1e-5);
        }
    }

    // ── MseAttackLoss tests ────────────────────────────────────────────────────

    #[test]
    fn test_mse_loss_zero_for_equal_predictions_and_labels() {
        let mse = MseAttackLoss;
        let v = vec![0.1, 0.5, -0.3];
        assert_abs_diff_eq!(mse.loss(&v, &v), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_mse_grad_zero_for_equal_predictions_and_labels() {
        let mse = MseAttackLoss;
        let v = vec![0.1, 0.5, -0.3];
        let grad = mse.grad(&v, &v);
        for &g in &grad {
            assert_abs_diff_eq!(g, 0.0, epsilon = 1e-12);
        }
    }

    // ── LinearAttackModel tests ────────────────────────────────────────────────

    #[test]
    fn test_linear_model_forward_correct_dimension() {
        let model = identity_model_2x2();
        let preds = model.forward(&[0.3, 0.7]);
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_linear_model_forward_correct_values() {
        let model = identity_model_2x2();
        let preds = model.forward(&[0.3, 0.7]);
        assert_abs_diff_eq!(preds[0], 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(preds[1], 0.7, epsilon = 1e-12);
    }

    #[test]
    fn test_linear_model_input_gradient_finite_difference() {
        // 3-output × 2-input model.
        let model = LinearAttackModel::new(
            vec![vec![2.0, -1.0], vec![0.5, 3.0], vec![-1.0, 1.0]],
            vec![0.0, 0.0, 0.0],
        )
        .expect("ok");
        let input = vec![0.4, 0.6];
        let out_grad = vec![1.0, 0.0, 0.0]; // select first output
        let analytic = model.input_gradient(&input, &out_grad);
        // Verify against numerical FD (default impl).
        let h = 1e-5_f64;
        for j in 0..input.len() {
            let mut ip = input.clone();
            let mut im = input.clone();
            ip[j] += h;
            im[j] -= h;
            let fd: f64 = model
                .forward(&ip)
                .iter()
                .zip(model.forward(&im).iter())
                .zip(out_grad.iter())
                .map(|((&fp, &fm), &g)| g * (fp - fm) / (2.0 * h))
                .sum();
            assert_abs_diff_eq!(analytic[j], fd, epsilon = 1e-6);
        }
    }

    // ── AdversarialExample tests ───────────────────────────────────────────────

    #[test]
    fn test_adversarial_example_perturbation_equals_diff() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.3, 0.7];
        let labels = vec![1.0, 0.0];
        let config = default_config();
        let ex = fgsm(&model, &loss, &input, &labels, &config).expect("ok");
        for (i, (&p, &o)) in ex.perturbed.iter().zip(ex.original.iter()).enumerate() {
            assert_abs_diff_eq!(ex.perturbation[i], p - o, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_adversarial_example_linf_le_epsilon() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.3, 0.7];
        let labels = vec![1.0, 0.0];
        let config = AttackConfig::new(0.05).expect("ok");
        let ex = fgsm(&model, &loss, &input, &labels, &config).expect("ok");
        assert!(ex.perturbation_linf() <= 0.05 + 1e-10);
    }

    // ── AttackConfig validation tests ─────────────────────────────────────────

    #[test]
    fn test_attack_config_negative_epsilon_is_error() {
        let result = AttackConfig::new(-0.1);
        assert!(
            matches!(result, Err(AdversarialError::InvalidEpsilon(_))),
            "expected InvalidEpsilon"
        );
    }

    #[test]
    fn test_attack_config_zero_epsilon_is_error() {
        let result = AttackConfig::new(0.0);
        assert!(matches!(result, Err(AdversarialError::InvalidEpsilon(_))));
    }

    #[test]
    fn test_attack_config_zero_steps_is_error() {
        let result = AttackConfig::new(0.1).expect("ok").with_steps(0);
        assert!(matches!(
            result,
            Err(AdversarialError::InvalidIterations(0))
        ));
    }

    // ── adversarial_training_loss tests ───────────────────────────────────────

    #[test]
    fn test_adversarial_training_loss_alpha_one_equals_clean_loss() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let inputs = vec![vec![0.5_f64, 0.5_f64]];
        let labels = vec![vec![1.0_f64, 0.0_f64]];
        // 1 step PGD = FGSM-like; but alpha=1 should zero-out adv contribution.
        let config = AttackConfig::new(0.1)
            .expect("ok")
            .with_steps(1)
            .expect("ok");
        let (combined, stats) =
            adversarial_training_loss(&model, &loss, &inputs, &labels, &config, 1.0, 0)
                .expect("ok");
        assert_abs_diff_eq!(combined, stats.clean_loss, epsilon = 1e-10);
    }

    #[test]
    fn test_adversarial_training_loss_alpha_zero_equals_adversarial_loss() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let inputs = vec![vec![0.5_f64, 0.5_f64]];
        let labels = vec![vec![1.0_f64, 0.0_f64]];
        let config = AttackConfig::new(0.1)
            .expect("ok")
            .with_steps(1)
            .expect("ok");
        let (combined, stats) =
            adversarial_training_loss(&model, &loss, &inputs, &labels, &config, 0.0, 0)
                .expect("ok");
        assert_abs_diff_eq!(combined, stats.adversarial_loss, epsilon = 1e-10);
    }

    // ── robustness_eval test ───────────────────────────────────────────────────

    #[test]
    fn test_robustness_eval_result_in_0_1() {
        let model = identity_model_2x2();
        let inputs = vec![
            vec![0.9_f64, 0.1_f64], // predicts class 0
            vec![0.1_f64, 0.9_f64], // predicts class 1
        ];
        let labels = vec![vec![1.0_f64, 0.0_f64], vec![0.0_f64, 1.0_f64]];
        let config = AttackConfig::new(0.05)
            .expect("ok")
            .with_steps(5)
            .expect("ok");
        let frac = robustness_eval(&model, &inputs, &labels, &config, 42).expect("ok");
        assert!(
            (0.0..=1.0).contains(&frac),
            "robustness fraction {frac} out of range"
        );
    }

    #[test]
    fn test_robustness_eval_empty_inputs() {
        let model = identity_model_2x2();
        let config = default_config();
        let frac = robustness_eval(&model, &[], &[], &config, 0).expect("ok");
        assert_abs_diff_eq!(frac, 1.0, epsilon = 1e-12);
    }

    // ── AdversarialTrainStats tests ────────────────────────────────────────────

    #[test]
    fn test_adversarial_train_stats_n_samples() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let inputs = vec![
            vec![0.5_f64, 0.5_f64],
            vec![0.2_f64, 0.8_f64],
            vec![0.7_f64, 0.3_f64],
        ];
        let labels = vec![
            vec![1.0_f64, 0.0_f64],
            vec![0.0_f64, 1.0_f64],
            vec![1.0_f64, 0.0_f64],
        ];
        let config = AttackConfig::new(0.1)
            .expect("ok")
            .with_steps(2)
            .expect("ok");
        let (_, stats) =
            adversarial_training_loss(&model, &loss, &inputs, &labels, &config, 0.5, 1)
                .expect("ok");
        assert_eq!(stats.n_samples, 3);
        assert!(stats.mean_perturbation_norm >= 0.0);
    }

    #[test]
    fn test_adversarial_train_stats_combined_loss_between_clean_and_adv() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let inputs = vec![vec![0.5_f64, 0.5_f64]];
        let labels = vec![vec![1.0_f64, 0.0_f64]];
        let config = AttackConfig::new(0.1)
            .expect("ok")
            .with_steps(3)
            .expect("ok");
        let alpha = 0.5;
        let (combined, stats) =
            adversarial_training_loss(&model, &loss, &inputs, &labels, &config, alpha, 99)
                .expect("ok");
        let expected = alpha * stats.clean_loss + (1.0 - alpha) * stats.adversarial_loss;
        assert_abs_diff_eq!(combined, expected, epsilon = 1e-10);
    }

    // ── Additional coverage ────────────────────────────────────────────────────

    #[test]
    fn test_pgd_random_start_stays_within_epsilon() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.5_f64, 0.5_f64];
        let labels = vec![1.0_f64, 0.0_f64];
        let config = AttackConfig::new(0.1)
            .expect("ok")
            .with_steps(5)
            .expect("ok")
            .with_random_start(true);
        let ex = pgd(&model, &loss, &input, &labels, &config, 12345).expect("ok");
        assert!(ex.perturbation_linf() <= 0.1 + 1e-10);
    }

    #[test]
    fn test_fgsm_l2_norm_attack() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.3, 0.7];
        let labels = vec![0.0, 1.0];
        let config = AttackConfig::new(0.1)
            .expect("ok")
            .with_norm(PerturbNorm::L2);
        let ex = fgsm(&model, &loss, &input, &labels, &config).expect("ok");
        assert!(ex.perturbation_l2() <= 0.1 + 1e-10);
    }

    #[test]
    fn test_fgsm_l1_norm_attack_single_nonzero_component() {
        let model = identity_model_2x2();
        let loss = MseAttackLoss;
        let input = vec![0.3, 0.7];
        let labels = vec![1.0, 0.0];
        let config = AttackConfig::new(0.1)
            .expect("ok")
            .with_norm(PerturbNorm::L1);
        let ex = fgsm(&model, &loss, &input, &labels, &config).expect("ok");
        // L1 FGSM puts all budget on one coordinate.
        let nonzero: Vec<f64> = ex
            .perturbation
            .iter()
            .cloned()
            .filter(|&d| d.abs() > 1e-12)
            .collect();
        assert_eq!(
            nonzero.len(),
            1,
            "L1 FGSM should perturb exactly one dimension"
        );
    }

    #[test]
    fn test_linear_model_construction_invalid_bias_len() {
        let result = LinearAttackModel::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0.0], // wrong length
        );
        assert!(result.is_err());
    }
}

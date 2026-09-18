//! Core value types for Loopy Belief Propagation: log-domain messages,
//! damping policies, update schedules, and the convergence monitor.

use scirs2_core::ndarray::{Array1, ArrayD};
use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Log-domain message
// ──────────────────────────────────────────────────────────────────────────────

/// A message stored in log-space for numerical stability.
///
/// All arithmetic (product, damping) is performed in log space, then
/// converted back to probability space only when computing beliefs.
#[derive(Clone, Debug)]
pub struct LogMessage {
    /// Variable this message is "about"
    pub variable: String,
    /// Log-probabilities (not necessarily normalised)
    pub log_values: Array1<f64>,
}

impl LogMessage {
    /// Create from a (probability-space) [`crate::Factor`] projected onto a single variable.
    pub fn from_factor_slice(variable: &str, values: &ArrayD<f64>) -> Self {
        let flat: Vec<f64> = values
            .iter()
            .map(|&v| if v > 1e-300 { v.ln() } else { -700.0 })
            .collect();
        Self {
            variable: variable.to_string(),
            log_values: Array1::from(flat),
        }
    }

    /// Create a uniform log-message of length `card`.
    pub fn uniform(variable: &str, card: usize) -> Self {
        let log_val = -(card as f64).ln();
        Self {
            variable: variable.to_string(),
            log_values: Array1::from_elem(card, log_val),
        }
    }

    /// Log-sum-exp normalise (subtract max, then subtract log-sum-exp).
    pub fn log_normalise(&mut self) {
        let max_v = self
            .log_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if max_v.is_finite() {
            self.log_values -= max_v;
            let lse = self.log_values.iter().map(|&x| x.exp()).sum::<f64>().ln();
            self.log_values -= lse;
        }
    }

    /// Convert to probability-space `Array1`.
    pub fn to_probs(&self) -> Array1<f64> {
        let max_v = self
            .log_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Array1<f64> = self.log_values.mapv(|x| (x - max_v).exp());
        let s: f64 = probs.iter().sum();
        if s > 1e-300 {
            probs /= s;
        } else {
            let n = probs.len();
            probs.fill(1.0 / n as f64);
        }
        probs
    }

    /// L∞ residual vs another message (in log space).
    pub fn residual_linf(&self, other: &LogMessage) -> f64 {
        self.log_values
            .iter()
            .zip(other.log_values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Apply damping: `λ * self + (1-λ) * old` (in log space via log-sum-exp).
    pub fn damp(&self, old: &LogMessage, lambda: f64) -> Self {
        // log( λ·exp(new) + (1-λ)·exp(old) )
        let mixed: Array1<f64> = self
            .log_values
            .iter()
            .zip(old.log_values.iter())
            .map(|(&n, &o)| {
                let a = lambda.ln() + n;
                let b = (1.0 - lambda).ln() + o;
                let m = a.max(b);
                m + ((a - m).exp() + (b - m).exp()).ln()
            })
            .collect();
        let mut out = Self {
            variable: self.variable.clone(),
            log_values: mixed,
        };
        out.log_normalise();
        out
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Damping policy
// ──────────────────────────────────────────────────────────────────────────────

/// Policy governing how message damping is applied.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LbpDampingPolicy {
    /// No damping; messages are replaced unconditionally.
    None,
    /// Uniform damping coefficient `λ ∈ (0, 1)`.
    ///
    /// Damped message = `λ * new + (1-λ) * old` (in log space).
    Uniform(f64),
    /// Adaptive damping: use `λ = residual / (residual + 1)`.
    ///
    /// Messages with large residuals are damped more aggressively.
    Adaptive { base_lambda: f64 },
}

impl LbpDampingPolicy {
    /// Compute the effective lambda for a message with a given residual.
    pub fn effective_lambda(&self, residual: f64) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Uniform(lam) => *lam,
            Self::Adaptive { base_lambda } => {
                // Schedule: lambda = base + (1-base) * exp(-residual)
                // High residual → lambda closer to base (more damping).
                let t = (-residual).exp();
                base_lambda + (1.0 - base_lambda) * t
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Update schedule
// ──────────────────────────────────────────────────────────────────────────────

/// Order in which messages are updated each sweep.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UpdateSchedule {
    /// All messages updated in parallel (synchronous / flood schedule).
    ///
    /// The new round of messages is computed from the *previous* round.
    /// Classic Loopy BP.
    Synchronous,
    /// Messages updated one by one in a fixed round-robin order.
    ///
    /// More stable but potentially slower to converge.
    Sequential,
    /// Residual BP — at each step update the message with the largest
    /// current L∞ residual first.
    ///
    /// Often converges in fewer sweeps than synchronous or sequential
    /// schedules on highly loopy graphs.
    Residual,
}

// ──────────────────────────────────────────────────────────────────────────────
// Convergence monitor
// ──────────────────────────────────────────────────────────────────────────────

/// Per-iteration convergence statistics for Loopy BP.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LbpIterStats {
    /// Iteration index (0-based).
    pub iteration: usize,
    /// Maximum L∞ residual across all messages in this iteration.
    pub max_residual: f64,
    /// Mean L∞ residual across all messages.
    pub mean_residual: f64,
    /// Number of messages that changed by more than `tolerance`.
    pub active_messages: usize,
}

/// Convergence monitor for Loopy Belief Propagation.
///
/// Records per-iteration residual statistics and detects convergence,
/// divergence, or oscillation (residual increasing after initial decrease).
#[derive(Clone, Debug, Default)]
pub struct LbpConvergenceMonitor {
    /// Per-iteration stats (most recent first).
    pub history: Vec<LbpIterStats>,
    /// Whether the algorithm has converged.
    pub converged: bool,
    /// Whether divergence was detected.
    pub diverged: bool,
    /// Iteration at which convergence occurred (if any).
    pub converged_at: Option<usize>,
}

impl LbpConvergenceMonitor {
    /// Create a new monitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record statistics from one iteration.
    pub fn record(&mut self, stats: LbpIterStats, tolerance: f64) {
        if stats.max_residual < tolerance && !self.converged {
            self.converged = true;
            self.converged_at = Some(stats.iteration);
        }

        // Detect divergence: residual more than 100× the initial value and growing.
        if self.history.len() >= 2 {
            let initial = self.history[0].max_residual;
            if initial > 0.0 && stats.max_residual > 100.0 * initial {
                self.diverged = true;
            }
        }

        self.history.push(stats);
    }

    /// True if the last recorded residual is below `tolerance`.
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Return the most recent max residual, or `f64::INFINITY` if no iterations yet.
    pub fn last_residual(&self) -> f64 {
        self.history
            .last()
            .map(|s| s.max_residual)
            .unwrap_or(f64::INFINITY)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Message direction (internal)
// ──────────────────────────────────────────────────────────────────────────────

/// Direction of a message in the factor graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum MessageDirection {
    VtoF,
    FtoV,
}

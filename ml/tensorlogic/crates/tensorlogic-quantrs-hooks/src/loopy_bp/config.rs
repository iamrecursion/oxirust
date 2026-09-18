//! Configuration and result types for Loopy Belief Propagation.

use scirs2_core::ndarray::{Array1, ArrayD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::cycle::CycleAnalysis;
use super::energy::BetheFreeEnergy;
use super::types::{LbpConvergenceMonitor, LbpDampingPolicy, UpdateSchedule};

// ──────────────────────────────────────────────────────────────────────────────
// Loopy BP configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for Loopy Belief Propagation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopyBpConfig {
    /// Maximum number of sweeps (full passes over all messages).
    pub max_iterations: usize,
    /// Convergence tolerance (L∞ norm of message residuals).
    pub tolerance: f64,
    /// Message damping policy.
    pub damping: LbpDampingPolicy,
    /// Message update schedule.
    pub schedule: UpdateSchedule,
    /// Whether to compute Bethe free energy at the end.
    pub compute_bethe: bool,
    /// Random seed for the residual-BP priority queue tie-breaking.
    pub seed: u64,
}

impl Default for LoopyBpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-6,
            damping: LbpDampingPolicy::Uniform(0.5),
            schedule: UpdateSchedule::Synchronous,
            compute_bethe: true,
            seed: 42,
        }
    }
}

impl LoopyBpConfig {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set maximum iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Builder: set tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Builder: set damping policy.
    pub fn with_damping(mut self, d: LbpDampingPolicy) -> Self {
        self.damping = d;
        self
    }

    /// Builder: set update schedule.
    pub fn with_schedule(mut self, s: UpdateSchedule) -> Self {
        self.schedule = s;
        self
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Result type
// ──────────────────────────────────────────────────────────────────────────────

/// Full output from a Loopy BP run.
#[derive(Clone, Debug)]
pub struct LoopyBpResult {
    /// Variable marginal beliefs: var_name → probability vector.
    pub beliefs: HashMap<String, Array1<f64>>,
    /// Factor joint beliefs: factor_id → joint probability tensor.
    pub factor_beliefs: HashMap<String, ArrayD<f64>>,
    /// Convergence monitor with full iteration history.
    pub convergence: LbpConvergenceMonitor,
    /// Bethe free energy (if `config.compute_bethe` was set).
    pub bethe: Option<BetheFreeEnergy>,
    /// Cycle analysis of the input factor graph.
    pub cycle_analysis: CycleAnalysis,
}

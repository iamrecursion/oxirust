//! Core Loopy BP engine: message store, belief computation, and the
//! [`MessagePassingAlgorithm`] trait implementation.

use scirs2_core::ndarray::{Array1, ArrayD};
use std::collections::HashMap;

use crate::error::{PgmError, Result};
use crate::graph::FactorGraph;
use crate::message_passing::MessagePassingAlgorithm;

use super::config::{LoopyBpConfig, LoopyBpResult};
use super::cycle::CycleDetector;
use super::energy::bethe_free_energy;
use super::types::{LbpConvergenceMonitor, LbpIterStats, LogMessage, UpdateSchedule};

// ──────────────────────────────────────────────────────────────────────────────
// Message store (log-domain)
// ──────────────────────────────────────────────────────────────────────────────

/// Log-domain message store for LBP.
#[derive(Clone, Debug, Default)]
pub(super) struct LogMessageStore {
    /// (variable, factor) → log-message
    pub(super) var_to_factor: HashMap<(String, String), LogMessage>,
    /// (factor, variable) → log-message
    pub(super) factor_to_var: HashMap<(String, String), LogMessage>,
}

impl LogMessageStore {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn get_vtf(&self, var: &str, fac: &str) -> Option<&LogMessage> {
        self.var_to_factor.get(&(var.to_string(), fac.to_string()))
    }

    pub(super) fn set_vtf(&mut self, var: String, fac: String, msg: LogMessage) {
        self.var_to_factor.insert((var, fac), msg);
    }

    pub(super) fn get_ftv(&self, fac: &str, var: &str) -> Option<&LogMessage> {
        self.factor_to_var.get(&(fac.to_string(), var.to_string()))
    }

    pub(super) fn set_ftv(&mut self, fac: String, var: String, msg: LogMessage) {
        self.factor_to_var.insert((fac, var), msg);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core Loopy BP engine
// ──────────────────────────────────────────────────────────────────────────────

/// Loopy Belief Propagation on general (cyclic) factor graphs.
///
/// # Example
/// ```rust,ignore
/// use tensorlogic_quantrs_hooks::loopy_bp::{LoopyBpConfig, LoopyBeliefPropagation, LbpDampingPolicy, UpdateSchedule};
/// use tensorlogic_quantrs_hooks::graph::FactorGraph;
///
/// let mut graph = FactorGraph::new();
/// graph.add_variable("x".to_string(), "Binary".to_string());
/// graph.add_variable("y".to_string(), "Binary".to_string());
///
/// let config = LoopyBpConfig::default()
///     .with_damping(LbpDampingPolicy::Uniform(0.5))
///     .with_schedule(UpdateSchedule::Synchronous);
///
/// let lbp = LoopyBeliefPropagation::new(config);
/// let result = lbp.run_full(&graph).expect("LBP failed");
/// println!("Converged: {}", result.convergence.is_converged());
/// ```
pub struct LoopyBeliefPropagation {
    /// Configuration.
    pub config: LoopyBpConfig,
}

impl LoopyBeliefPropagation {
    /// Create a new LBP engine with the given configuration.
    pub fn new(config: LoopyBpConfig) -> Self {
        Self { config }
    }

    /// Run Loopy BP and return the full [`LoopyBpResult`].
    pub fn run_full(&self, graph: &FactorGraph) -> Result<LoopyBpResult> {
        // Analyse cycles first.
        let cycle_analysis = CycleDetector::new(graph).analyse();

        // Initialise message store with uniform messages.
        let mut messages = self.initialise_messages(graph);

        let mut monitor = LbpConvergenceMonitor::new();

        match self.config.schedule {
            UpdateSchedule::Synchronous => {
                self.run_synchronous(graph, &mut messages, &mut monitor)?;
            }
            UpdateSchedule::Sequential => {
                self.run_sequential(graph, &mut messages, &mut monitor)?;
            }
            UpdateSchedule::Residual => {
                self.run_residual(graph, &mut messages, &mut monitor)?;
            }
        }

        // Compute variable beliefs.
        let beliefs = self.compute_variable_beliefs(graph, &messages)?;
        let factor_beliefs = self.compute_factor_beliefs(graph, &messages)?;

        // Optionally compute Bethe free energy.
        let bethe = if self.config.compute_bethe {
            Some(bethe_free_energy(graph, &beliefs, &factor_beliefs))
        } else {
            None
        };

        Ok(LoopyBpResult {
            beliefs,
            factor_beliefs,
            convergence: monitor,
            bethe,
            cycle_analysis,
        })
    }

    // ── Initialisation ──────────────────────────────────────────────────────

    pub(super) fn initialise_messages(&self, graph: &FactorGraph) -> LogMessageStore {
        let mut store = LogMessageStore::new();

        for var_name in graph.variable_names() {
            let card = graph
                .get_variable(var_name)
                .map(|v| v.cardinality)
                .unwrap_or(2);

            if let Some(fac_ids) = graph.get_adjacent_factors(var_name) {
                for fac_id in fac_ids {
                    // var → factor  (uniform)
                    store.set_vtf(
                        var_name.clone(),
                        fac_id.clone(),
                        LogMessage::uniform(var_name, card),
                    );

                    // factor → var  (uniform, but seeded from factor values if available)
                    let ftv_msg = if let Some(factor) = graph.get_factor(fac_id) {
                        // Marginalise factor over all variables except this one.
                        let marginal = self.marginalise_factor_to_var(factor, var_name);
                        marginal.unwrap_or_else(|_| LogMessage::uniform(var_name, card))
                    } else {
                        LogMessage::uniform(var_name, card)
                    };
                    store.set_ftv(fac_id.clone(), var_name.clone(), ftv_msg);
                }
            }
        }

        store
    }

    // ── Message computation ───────────────────────────────────────────────────

    /// Compute a variable→factor log-message:
    /// `log μ(x→f, xᵢ) = ∑_{g ∈ N(x) \ f} log μ(g→x, xᵢ)`
    pub(super) fn compute_vtf_message(
        &self,
        graph: &FactorGraph,
        messages: &LogMessageStore,
        var: &str,
        target_fac: &str,
    ) -> Result<LogMessage> {
        let card = graph
            .get_variable(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?
            .cardinality;

        let mut log_msg = Array1::<f64>::zeros(card);

        if let Some(fac_ids) = graph.get_adjacent_factors(var) {
            for fac_id in fac_ids {
                if fac_id == target_fac {
                    continue;
                }
                if let Some(ftv) = messages.get_ftv(fac_id, var) {
                    // Sum log-messages (= product in probability space).
                    log_msg += &ftv.log_values;
                }
            }
        }

        let mut msg = LogMessage {
            variable: var.to_string(),
            log_values: log_msg,
        };
        msg.log_normalise();
        Ok(msg)
    }

    /// Compute a factor→variable log-message (log-domain sum-product):
    /// `log μ(f→x, xᵢ) = log ∑_{~xᵢ} [φ_f(x) ∏_{y∈N(f)\xᵢ} μ(y→f, y_j)]`
    pub(super) fn compute_ftv_message(
        &self,
        graph: &FactorGraph,
        messages: &LogMessageStore,
        fac_id: &str,
        target_var: &str,
    ) -> Result<LogMessage> {
        let factor = graph
            .get_factor(fac_id)
            .ok_or_else(|| PgmError::FactorNotFound(fac_id.to_string()))?;

        let target_idx = factor
            .variables
            .iter()
            .position(|v| v == target_var)
            .ok_or_else(|| {
                PgmError::VariableNotFound(format!(
                    "Variable '{}' not in factor '{}'",
                    target_var, fac_id
                ))
            })?;

        let target_card = factor.values.shape()[target_idx];

        // Compute log(φ_f(x) * ∏_{y≠target} μ(y→f, y_j)) for every joint assignment,
        // then marginalise (log-sum-exp) over all dimensions except target.
        let total_size: usize = factor.values.shape().iter().product();
        let mut log_joint = Vec::with_capacity(total_size);

        for lin_idx in 0..total_size {
            let assignment = linear_to_assignment(lin_idx, factor.values.shape());
            let mut log_val = {
                let phi = factor.values[assignment.as_slice()];
                if phi > 1e-300 {
                    phi.ln()
                } else {
                    -700.0
                }
            };
            // Multiply in incoming var→factor messages.
            for (dim, var_name) in factor.variables.iter().enumerate() {
                if var_name == target_var {
                    continue;
                }
                if let Some(vtf) = messages.get_vtf(var_name, fac_id) {
                    let val_idx = assignment[dim];
                    let lv = vtf.log_values.get(val_idx).copied().unwrap_or(-700.0);
                    log_val += lv;
                }
            }
            log_joint.push((assignment[target_idx], log_val));
        }

        // Log-sum-exp over all assignments sharing the same target value.
        let mut result = vec![f64::NEG_INFINITY; target_card];
        for (t_val, lv) in log_joint {
            // log-sum-exp accumulate: log(exp(a) + exp(b)) = max + log(1 + exp(min-max))
            let cur = result[t_val];
            if cur == f64::NEG_INFINITY {
                result[t_val] = lv;
            } else {
                let m = cur.max(lv);
                result[t_val] = m + ((cur - m).exp() + (lv - m).exp()).ln();
            }
        }

        let mut msg = LogMessage {
            variable: target_var.to_string(),
            log_values: Array1::from(result),
        };
        msg.log_normalise();
        Ok(msg)
    }

    /// Apply all updated messages with damping and track residuals.
    pub(super) fn apply_updates_and_track(
        &self,
        messages: &mut LogMessageStore,
        new_vtf: HashMap<(String, String), LogMessage>,
        new_ftv: HashMap<(String, String), LogMessage>,
        iteration: usize,
    ) -> LbpIterStats {
        let mut max_residual = 0.0_f64;
        let mut sum_residual = 0.0_f64;
        let mut count = 0usize;
        let mut active = 0usize;

        for ((var, fac), new_msg) in new_vtf {
            let old = messages.get_vtf(&var, &fac).cloned();
            let residual = old
                .as_ref()
                .map(|o| new_msg.residual_linf(o))
                .unwrap_or(1.0);
            let lambda = self.config.damping.effective_lambda(residual);
            let final_msg = if let Some(o) = &old {
                new_msg.damp(o, lambda)
            } else {
                new_msg
            };
            max_residual = max_residual.max(residual);
            sum_residual += residual;
            count += 1;
            if residual >= self.config.tolerance {
                active += 1;
            }
            messages.set_vtf(var, fac, final_msg);
        }

        for ((fac, var), new_msg) in new_ftv {
            let old = messages.get_ftv(&fac, &var).cloned();
            let residual = old
                .as_ref()
                .map(|o| new_msg.residual_linf(o))
                .unwrap_or(1.0);
            let lambda = self.config.damping.effective_lambda(residual);
            let final_msg = if let Some(o) = &old {
                new_msg.damp(o, lambda)
            } else {
                new_msg
            };
            max_residual = max_residual.max(residual);
            sum_residual += residual;
            count += 1;
            if residual >= self.config.tolerance {
                active += 1;
            }
            messages.set_ftv(fac, var, final_msg);
        }

        let mean_residual = if count > 0 {
            sum_residual / count as f64
        } else {
            0.0
        };

        LbpIterStats {
            iteration,
            max_residual,
            mean_residual,
            active_messages: active,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MessagePassingAlgorithm impl (integrates with the existing trait)
// ──────────────────────────────────────────────────────────────────────────────

impl MessagePassingAlgorithm for LoopyBeliefPropagation {
    fn run(
        &self,
        graph: &FactorGraph,
    ) -> std::result::Result<HashMap<String, ArrayD<f64>>, crate::error::PgmError> {
        let result = self.run_full(graph)?;
        // Convert Array1 → ArrayD for compatibility with the trait.
        let beliefs_dyn: HashMap<String, ArrayD<f64>> = result
            .beliefs
            .into_iter()
            .map(|(k, v)| (k, v.into_dyn()))
            .collect();
        Ok(beliefs_dyn)
    }

    fn name(&self) -> &str {
        "LoopyBeliefPropagation"
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Utility
// ──────────────────────────────────────────────────────────────────────────────

/// Convert a linear index to a multi-dimensional index for a given shape.
pub(super) fn linear_to_assignment(mut lin: usize, shape: &[usize]) -> Vec<usize> {
    let mut assignment = vec![0usize; shape.len()];
    for (i, &dim) in shape.iter().enumerate().rev() {
        assignment[i] = lin % dim;
        lin /= dim;
    }
    assignment
}

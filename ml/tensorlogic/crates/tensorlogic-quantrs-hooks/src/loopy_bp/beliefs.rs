//! Variable and factor belief computation from the LBP message store,
//! plus the helper that marginalises a factor to bootstrap messages.

use scirs2_core::ndarray::{Array1, ArrayD};
use std::collections::HashMap;

use crate::error::{PgmError, Result};
use crate::factor::Factor;
use crate::graph::FactorGraph;

use super::engine::{linear_to_assignment, LogMessageStore, LoopyBeliefPropagation};
use super::types::LogMessage;

impl LoopyBeliefPropagation {
    /// Compute per-variable marginal beliefs from the current message store.
    pub(super) fn compute_variable_beliefs(
        &self,
        graph: &FactorGraph,
        messages: &LogMessageStore,
    ) -> Result<HashMap<String, Array1<f64>>> {
        let mut beliefs = HashMap::new();

        for var_name in graph.variable_names() {
            let card = graph
                .get_variable(var_name)
                .map(|v| v.cardinality)
                .unwrap_or(2);
            let mut log_belief = Array1::<f64>::zeros(card);

            if let Some(fac_ids) = graph.get_adjacent_factors(var_name) {
                for fac_id in fac_ids {
                    if let Some(ftv) = messages.get_ftv(fac_id, var_name) {
                        log_belief += &ftv.log_values;
                    }
                }
            }

            let mut belief_msg = LogMessage {
                variable: var_name.clone(),
                log_values: log_belief,
            };
            belief_msg.log_normalise();
            beliefs.insert(var_name.clone(), belief_msg.to_probs());
        }

        Ok(beliefs)
    }

    /// Compute per-factor joint beliefs from the current message store.
    pub(super) fn compute_factor_beliefs(
        &self,
        graph: &FactorGraph,
        messages: &LogMessageStore,
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut factor_beliefs = HashMap::new();

        for fac_id in graph.factor_ids() {
            if let Some(factor) = graph.get_factor(fac_id) {
                // log belief_f(x_a) = log φ_f(x_a) + ∑_{y∈N(f)} log μ(y→f, y_j)
                let shape = factor.values.shape().to_vec();
                let total: usize = shape.iter().product();
                let mut log_belief = Vec::with_capacity(total);

                for lin_idx in 0..total {
                    let assignment = linear_to_assignment(lin_idx, &shape);
                    let phi = factor.values[assignment.as_slice()];
                    let mut lv = if phi > 1e-300 { phi.ln() } else { -700.0 };
                    for (dim, var_name) in factor.variables.iter().enumerate() {
                        if let Some(vtf) = messages.get_vtf(var_name, fac_id) {
                            let val_idx = assignment[dim];
                            lv += vtf.log_values.get(val_idx).copied().unwrap_or(-700.0);
                        }
                    }
                    log_belief.push(lv);
                }

                // Normalise (subtract log-sum-exp).
                let max_lv = log_belief.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut probs: Vec<f64> = log_belief.iter().map(|&x| (x - max_lv).exp()).collect();
                let s: f64 = probs.iter().sum();
                if s > 1e-300 {
                    for p in &mut probs {
                        *p /= s;
                    }
                } else {
                    let n = probs.len();
                    probs.fill(1.0 / n as f64);
                }

                let arr = ArrayD::from_shape_vec(shape, probs).map_err(|_| {
                    PgmError::InvalidGraph(format!(
                        "Could not reshape factor belief for '{}'",
                        fac_id
                    ))
                })?;
                factor_beliefs.insert(fac_id.clone(), arr);
            }
        }

        Ok(factor_beliefs)
    }

    /// Marginalise a factor over all variables except `target_var` to produce
    /// the initial factor→variable message.
    pub(super) fn marginalise_factor_to_var(
        &self,
        factor: &Factor,
        target_var: &str,
    ) -> Result<LogMessage> {
        let target_idx = factor
            .variables
            .iter()
            .position(|v| v == target_var)
            .ok_or_else(|| PgmError::VariableNotFound(target_var.to_string()))?;
        let target_card = factor.values.shape()[target_idx];
        let total: usize = factor.values.shape().iter().product();

        let mut result = vec![f64::NEG_INFINITY; target_card];

        for lin_idx in 0..total {
            let assignment = linear_to_assignment(lin_idx, factor.values.shape());
            let phi = factor.values[assignment.as_slice()];
            let lv = if phi > 1e-300 { phi.ln() } else { -700.0 };
            let t_val = assignment[target_idx];
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
}

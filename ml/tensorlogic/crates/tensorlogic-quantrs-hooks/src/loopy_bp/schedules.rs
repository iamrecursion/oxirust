//! Update-schedule implementations for Loopy BP: synchronous, sequential,
//! and residual (priority-queue) schedules.

use std::collections::{BinaryHeap, HashMap};

use crate::error::Result;
use crate::graph::FactorGraph;

use super::engine::{LogMessageStore, LoopyBeliefPropagation};
use super::types::{LbpConvergenceMonitor, LbpIterStats, LogMessage, MessageDirection};

impl LoopyBeliefPropagation {
    // ── Synchronous schedule ─────────────────────────────────────────────────

    pub(super) fn run_synchronous(
        &self,
        graph: &FactorGraph,
        messages: &mut LogMessageStore,
        monitor: &mut LbpConvergenceMonitor,
    ) -> Result<()> {
        for iteration in 0..self.config.max_iterations {
            // Compute all new messages from the *current* (old) messages.
            let mut new_vtf: HashMap<(String, String), LogMessage> = HashMap::new();
            let mut new_ftv: HashMap<(String, String), LogMessage> = HashMap::new();

            // Variable → Factor messages.
            for var_name in graph.variable_names() {
                if let Some(fac_ids) = graph.get_adjacent_factors(var_name) {
                    for fac_id in fac_ids {
                        let new_msg =
                            self.compute_vtf_message(graph, messages, var_name, fac_id)?;
                        new_vtf.insert((var_name.clone(), fac_id.clone()), new_msg);
                    }
                }
            }

            // Factor → Variable messages.
            for fac_id in graph.factor_ids() {
                if let Some(vars) = graph.get_adjacent_variables(fac_id) {
                    for var_name in vars {
                        let new_msg =
                            self.compute_ftv_message(graph, messages, fac_id, var_name)?;
                        new_ftv.insert((fac_id.clone(), var_name.clone()), new_msg);
                    }
                }
            }

            // Compute residuals and apply damping.
            let stats = self.apply_updates_and_track(messages, new_vtf, new_ftv, iteration);

            monitor.record(stats, self.config.tolerance);

            if monitor.is_converged() {
                break;
            }
        }

        Ok(())
    }

    // ── Sequential schedule ───────────────────────────────────────────────────

    pub(super) fn run_sequential(
        &self,
        graph: &FactorGraph,
        messages: &mut LogMessageStore,
        monitor: &mut LbpConvergenceMonitor,
    ) -> Result<()> {
        // Build a deterministic ordering of all (directed) messages.
        let mut all_pairs: Vec<(MessageDirection, String, String)> = Vec::new();
        for var_name in graph.variable_names() {
            if let Some(fac_ids) = graph.get_adjacent_factors(var_name) {
                for fac_id in fac_ids {
                    all_pairs.push((MessageDirection::VtoF, var_name.clone(), fac_id.clone()));
                    all_pairs.push((MessageDirection::FtoV, fac_id.clone(), var_name.clone()));
                }
            }
        }

        for iteration in 0..self.config.max_iterations {
            let mut max_residual = 0.0_f64;
            let mut sum_residual = 0.0_f64;
            let mut active = 0usize;

            for (dir, a, b) in &all_pairs {
                match dir {
                    MessageDirection::VtoF => {
                        let new_msg = self.compute_vtf_message(graph, messages, a, b)?;
                        let old = messages.get_vtf(a, b).cloned();
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
                        if residual >= self.config.tolerance {
                            active += 1;
                        }
                        messages.set_vtf(a.clone(), b.clone(), final_msg);
                    }
                    MessageDirection::FtoV => {
                        let new_msg = self.compute_ftv_message(graph, messages, a, b)?;
                        let old = messages.get_ftv(a, b).cloned();
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
                        if residual >= self.config.tolerance {
                            active += 1;
                        }
                        messages.set_ftv(a.clone(), b.clone(), final_msg);
                    }
                }
            }

            let mean_residual = if all_pairs.is_empty() {
                0.0
            } else {
                sum_residual / all_pairs.len() as f64
            };

            let stats = LbpIterStats {
                iteration,
                max_residual,
                mean_residual,
                active_messages: active,
            };
            monitor.record(stats, self.config.tolerance);

            if monitor.is_converged() {
                break;
            }
        }

        Ok(())
    }

    // ── Residual BP schedule ─────────────────────────────────────────────────

    pub(super) fn run_residual(
        &self,
        graph: &FactorGraph,
        messages: &mut LogMessageStore,
        monitor: &mut LbpConvergenceMonitor,
    ) -> Result<()> {
        // Use a max-heap keyed by residual.  We use ordered floats via a wrapper.
        #[derive(PartialEq)]
        struct PQEntry {
            residual: f64,
            dir: MessageDirection,
            from: String,
            to: String,
        }

        impl Eq for PQEntry {}

        impl PartialOrd for PQEntry {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for PQEntry {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.residual
                    .partial_cmp(&other.residual)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| self.from.cmp(&other.from))
                    .then_with(|| self.to.cmp(&other.to))
            }
        }

        // Initialise heap with all messages at residual 1.0.
        let mut heap: BinaryHeap<PQEntry> = BinaryHeap::new();
        for var_name in graph.variable_names() {
            if let Some(fac_ids) = graph.get_adjacent_factors(var_name) {
                for fac_id in fac_ids {
                    heap.push(PQEntry {
                        residual: 1.0,
                        dir: MessageDirection::VtoF,
                        from: var_name.clone(),
                        to: fac_id.clone(),
                    });
                    heap.push(PQEntry {
                        residual: 1.0,
                        dir: MessageDirection::FtoV,
                        from: fac_id.clone(),
                        to: var_name.clone(),
                    });
                }
            }
        }

        let total_messages = heap.len().max(1);
        let mut global_iter = 0usize;
        let mut steps_since_report = 0usize;
        let mut max_residual = 1.0_f64;
        let mut sum_residual = 0.0_f64;
        let mut active = total_messages;

        // Total budget in "message updates" mapped to sweeps.
        let budget = self.config.max_iterations * total_messages;
        let mut steps = 0usize;

        while let Some(entry) = heap.pop() {
            if steps >= budget {
                break;
            }
            steps += 1;
            steps_since_report += 1;

            let (residual, new_neighbors) = match entry.dir {
                MessageDirection::VtoF => {
                    let new_msg =
                        self.compute_vtf_message(graph, messages, &entry.from, &entry.to)?;
                    let old = messages.get_vtf(&entry.from, &entry.to).cloned();
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
                    messages.set_vtf(entry.from.clone(), entry.to.clone(), final_msg);

                    // Neighbours to re-schedule: factor→var messages from entry.to.
                    let neighbors = graph
                        .get_adjacent_variables(&entry.to)
                        .cloned()
                        .unwrap_or_default();
                    (
                        residual,
                        neighbors
                            .into_iter()
                            .map(|v| (entry.to.clone(), v))
                            .collect::<Vec<_>>(),
                    )
                }
                MessageDirection::FtoV => {
                    let new_msg =
                        self.compute_ftv_message(graph, messages, &entry.from, &entry.to)?;
                    let old = messages.get_ftv(&entry.from, &entry.to).cloned();
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
                    messages.set_ftv(entry.from.clone(), entry.to.clone(), final_msg);

                    // Neighbours to re-schedule: var→factor messages from entry.to.
                    let neighbors = graph
                        .get_adjacent_factors(&entry.to)
                        .cloned()
                        .unwrap_or_default();
                    (
                        residual,
                        neighbors
                            .into_iter()
                            .map(|f| (entry.to.clone(), f))
                            .collect::<Vec<_>>(),
                    )
                }
            };

            // Re-add affected messages to the priority queue.
            for (from, to) in new_neighbors {
                // Compute prospective residual (from→to, FtoV direction since
                // after a FtoV update, we perturb VtoF).
                let dir = match entry.dir {
                    MessageDirection::VtoF => MessageDirection::FtoV,
                    MessageDirection::FtoV => MessageDirection::VtoF,
                };
                heap.push(PQEntry {
                    residual,
                    dir,
                    from,
                    to,
                });
            }

            // Emit convergence statistics every `total_messages` steps.
            if steps_since_report >= total_messages || heap.is_empty() {
                steps_since_report = 0;
                let stats = LbpIterStats {
                    iteration: global_iter,
                    max_residual,
                    mean_residual: sum_residual / total_messages as f64,
                    active_messages: active,
                };
                monitor.record(stats, self.config.tolerance);
                global_iter += 1;
                max_residual = 0.0;
                sum_residual = 0.0;
                active = 0;
                if monitor.is_converged() {
                    break;
                }
            } else {
                max_residual = max_residual.max(residual);
                sum_residual += residual;
                if residual >= self.config.tolerance {
                    active += 1;
                }
            }
        }

        Ok(())
    }
}

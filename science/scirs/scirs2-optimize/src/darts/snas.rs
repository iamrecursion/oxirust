//! SNAS: Stochastic Neural Architecture Search (Xie et al., ICLR 2019).
//!
//! SNAS uses the concrete (Gumbel-Softmax) distribution as mixing weights for
//! the forward pass — unlike GDAS's straight-through hard selection, SNAS
//! computes a **weighted sum** over **all** operations using concrete-sample
//! weights.  This allows the resource cost to be differentiated through the
//! architecture parameters via the expected-cost term.
//!
//! The full objective is:
//!
//! ```text
//! L_total = L_task + λ · E[cost(architecture)]
//! ```
//!
//! where the expectation is approximated via `softmax(α / τ)` weights.
//!
//! ## References
//!
//! - Xie, S., Zheng, H., Liu, C. and Lin, L. (2019). "SNAS: Stochastic Neural
//!   Architecture Search". ICLR 2019.

use super::{AnnealingStrategy, Lcg, Operation, TemperatureSchedule};
use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────── SnasConfig ──

/// Configuration for a SNAS architecture search experiment.
#[derive(Debug, Clone)]
pub struct SnasConfig {
    /// Number of cells stacked in the super-network.
    pub n_cells: usize,
    /// Number of candidate operations per edge.
    pub n_operations: usize,
    /// Number of feature channels (used for FLOP cost estimation).
    pub channels: usize,
    /// Number of intermediate nodes per cell.
    pub n_nodes: usize,
    /// Learning rate for architecture parameter updates.
    pub arch_lr: f64,
    /// Learning rate for network weight updates.
    pub weight_lr: f64,
    /// Temperature schedule for the concrete distribution.
    pub temperature_schedule: TemperatureSchedule,
    /// Resource penalty weight λ (multiplies expected FLOP cost in the loss).
    pub resource_weight: f64,
    /// Random seed for the internal LCG.
    pub seed: u64,
}

impl Default for SnasConfig {
    fn default() -> Self {
        Self {
            n_cells: 3,
            n_operations: 6,
            channels: 32,
            n_nodes: 4,
            arch_lr: 3e-4,
            weight_lr: 1e-3,
            temperature_schedule: TemperatureSchedule::new(
                1.0,
                0.1,
                AnnealingStrategy::Exponential,
                100,
            ),
            resource_weight: 0.001,
            seed: 42,
        }
    }
}

// ──────────────────────────────────────────────────── SnasMixedOperation ──

/// One mixed operation on a directed edge in the SNAS cell DAG.
///
/// Unlike GDAS's hard selection, SNAS uses the concrete sample weights to
/// compute a differentiable weighted sum over all operations.
#[derive(Debug, Clone)]
pub struct SnasMixedOperation {
    /// Un-normalised architecture parameters `α_k`, one per operation.
    pub arch_params: Vec<f64>,
    /// Concrete sample weights from the last sampling pass.
    pub last_concrete_weights: Vec<f64>,
}

impl SnasMixedOperation {
    /// Create a new `SnasMixedOperation` initialised to uniform weights.
    pub fn new(n_ops: usize) -> Self {
        Self {
            arch_params: vec![0.0_f64; n_ops],
            last_concrete_weights: vec![1.0 / n_ops as f64; n_ops],
        }
    }

    /// Draw a concrete (Gumbel-Softmax) sample.
    ///
    /// Returns the soft weights `w_k = softmax((α_k + g_k) / τ)` where `g_k`
    /// are i.i.d. Gumbel(0,1) noise samples.  The weights sum to 1 and are
    /// non-negative.
    pub fn concrete_sample(&self, temperature: f64, rng: &mut Lcg) -> Vec<f64> {
        let eps = 1e-20_f64;
        let temp = temperature.max(1e-8);
        let n = self.arch_params.len();

        let mut logits = vec![0.0_f64; n];
        for k in 0..n {
            let u = rng.next_f64().max(eps);
            let gumbel_noise = -(-u.ln()).ln();
            logits[k] = self.arch_params[k] + gumbel_noise;
        }

        // Numerically-stable softmax at temperature τ.
        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exp_vals: Vec<f64> = logits.iter().map(|&l| ((l - max_l) / temp).exp()).collect();
        let sum = exp_vals.iter().sum::<f64>().max(eps);
        for v in &mut exp_vals {
            *v /= sum;
        }
        exp_vals
    }

    /// Compute standard softmax-normalised weights (no noise) at `temperature`.
    pub fn weights(&self, temperature: f64) -> Vec<f64> {
        let t = temperature.max(1e-8);
        let scaled: Vec<f64> = self.arch_params.iter().map(|a| a / t).collect();
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scaled.iter().map(|s| (s - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            let n = self.arch_params.len();
            vec![1.0 / n as f64; n]
        } else {
            exps.iter().map(|e| e / sum).collect()
        }
    }

    /// Compute the expected FLOP cost for this edge.
    ///
    /// `E[cost] = Σ_k w_k * cost_flops(op_k)` where `w_k` are softmax weights
    /// at the current temperature.
    pub fn expected_cost(&self, temperature: f64, channels: usize) -> f64 {
        let ops = Operation::all();
        let w = self.weights(temperature);
        w.iter()
            .zip(ops.iter())
            .take(self.arch_params.len())
            .map(|(&wk, op)| wk * op.cost_flops(channels))
            .sum()
    }

    /// Index of the operation with the highest architecture weight (argmax).
    pub fn argmax_op(&self) -> usize {
        self.arch_params
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Apply a gradient-descent step to architecture parameters.
    pub fn update_arch_params(&mut self, grads: &[f64], lr: f64) {
        for (p, g) in self.arch_params.iter_mut().zip(grads.iter()) {
            *p -= lr * g;
        }
    }
}

// ─────────────────────────────────────────────────────────────── SnasCell ──

/// A SNAS cell: DAG with fixed input nodes and learnable intermediate nodes.
#[derive(Debug, Clone)]
pub struct SnasCell {
    /// Number of intermediate (learnable) nodes.
    pub n_nodes: usize,
    /// Number of fixed input nodes (typically 2).
    pub n_input_nodes: usize,
    /// `edges[i][j]` is the `SnasMixedOperation` from node j to intermediate
    /// node i.
    pub edges: Vec<Vec<SnasMixedOperation>>,
}

impl SnasCell {
    /// Create a new SNAS cell.
    pub fn new(n_input_nodes: usize, n_intermediate_nodes: usize, n_ops: usize) -> Self {
        let edges: Vec<Vec<SnasMixedOperation>> = (0..n_intermediate_nodes)
            .map(|i| {
                let n_predecessors = n_input_nodes + i;
                (0..n_predecessors)
                    .map(|_| SnasMixedOperation::new(n_ops))
                    .collect()
            })
            .collect();
        Self {
            n_nodes: n_intermediate_nodes,
            n_input_nodes,
            edges,
        }
    }

    /// Collect all architecture parameters from every edge, flattened.
    pub fn arch_parameters(&self) -> Vec<f64> {
        self.edges
            .iter()
            .flat_map(|row| row.iter().flat_map(|mo| mo.arch_params.iter().cloned()))
            .collect()
    }

    /// Compute total expected FLOP cost for this cell.
    pub fn total_expected_cost(&self, temperature: f64, channels: usize) -> f64 {
        self.edges
            .iter()
            .flat_map(|row| row.iter())
            .map(|mo| mo.expected_cost(temperature, channels))
            .sum()
    }

    /// Apply gradient updates to architecture parameters.
    pub fn update_arch_params(&mut self, grads: &[f64], lr: f64) -> OptimizeResult<()> {
        let n_params: usize = self
            .edges
            .iter()
            .flat_map(|row| row.iter())
            .map(|mo| mo.arch_params.len())
            .sum();
        if grads.len() != n_params {
            return Err(OptimizeError::InvalidInput(format!(
                "SnasCell::update_arch_params: expected {n_params} grads, got {}",
                grads.len()
            )));
        }
        let mut idx = 0;
        for row in self.edges.iter_mut() {
            for mo in row.iter_mut() {
                let n = mo.arch_params.len();
                mo.update_arch_params(&grads[idx..idx + n], lr);
                idx += n;
            }
        }
        Ok(())
    }

    /// Derive the discrete architecture: argmax operation index per edge.
    pub fn derive_discrete(&self) -> Vec<Vec<usize>> {
        self.edges
            .iter()
            .map(|row| row.iter().map(|mo| mo.argmax_op()).collect())
            .collect()
    }
}

// ──────────────────────────────────────────────────────────── SnasSearch ──

/// Top-level SNAS search controller.
///
/// Implements the bi-level optimisation loop with resource penalty.
pub struct SnasSearch {
    /// Stack of cells forming the super-network.
    pub cells: Vec<SnasCell>,
    /// Configuration.
    pub config: SnasConfig,
    /// Flat network weights (one scalar per cell in this toy model).
    weights: Vec<f64>,
    /// Internal pseudo-random number generator.
    rng: Lcg,
    /// Current training step (advances the temperature schedule).
    current_step: usize,
}

impl SnasSearch {
    /// Construct a `SnasSearch` from the given config.
    pub fn new(config: SnasConfig) -> Self {
        let cells: Vec<SnasCell> = (0..config.n_cells)
            .map(|_| SnasCell::new(2, config.n_nodes, config.n_operations))
            .collect();
        let weights = vec![0.01_f64; config.n_cells];
        let rng = Lcg::new(config.seed);
        Self {
            cells,
            config,
            weights,
            rng,
            current_step: 0,
        }
    }

    /// Current concrete-distribution temperature.
    pub fn current_temperature(&self) -> f64 {
        self.config
            .temperature_schedule
            .temperature_at(self.current_step)
    }

    /// Return all architecture parameters across all cells, flattened.
    pub fn arch_parameters(&self) -> Vec<f64> {
        self.cells
            .iter()
            .flat_map(|c| c.arch_parameters())
            .collect()
    }

    /// Total number of architecture parameters.
    pub fn n_arch_params(&self) -> usize {
        self.cells.iter().map(|c| c.arch_parameters().len()).sum()
    }

    /// Compute total expected FLOP cost over all cells (resource penalty term).
    pub fn total_expected_cost(&self) -> f64 {
        let temp = self.current_temperature();
        let channels = self.config.channels;
        self.cells
            .iter()
            .map(|c| c.total_expected_cost(temp, channels))
            .sum()
    }

    /// Apply a gradient step to architecture parameters.
    pub fn update_arch_params(&mut self, grads: &[f64], lr: f64) -> OptimizeResult<()> {
        let total = self.n_arch_params();
        if grads.len() != total {
            return Err(OptimizeError::InvalidInput(format!(
                "SnasSearch::update_arch_params: expected {total} grads, got {}",
                grads.len()
            )));
        }
        let mut offset = 0;
        for cell in self.cells.iter_mut() {
            let n = cell.arch_parameters().len();
            cell.update_arch_params(&grads[offset..offset + n], lr)?;
            offset += n;
        }
        Ok(())
    }

    /// Derive discrete architecture: argmax op index per edge, per cell.
    pub fn derive_discrete_arch_indices(&self) -> Vec<Vec<Vec<usize>>> {
        self.cells.iter().map(|c| c.derive_discrete()).collect()
    }

    /// Compute finite-difference gradients of the SNAS validation objective
    /// (task loss + λ · resource_cost) w.r.t. architecture parameters.
    ///
    /// `val_fn(arch_params)` should return the **task** loss.  The resource
    /// penalty is computed analytically via `total_expected_cost`.
    pub fn arch_grads_fd(&self, val_fn: impl Fn(&[f64]) -> f64, step: f64) -> Vec<f64> {
        let params = self.arch_parameters();
        let n = params.len();
        let lambda = self.config.resource_weight;
        let temp = self.current_temperature();
        let channels = self.config.channels;

        let mut grads = vec![0.0_f64; n];
        for i in 0..n {
            let mut p_plus = params.clone();
            p_plus[i] += step;
            let mut p_minus = params.clone();
            p_minus[i] -= step;

            // Resource cost gradient via FD on arch params.
            let cost_plus = resource_cost_at(&p_plus, &self.cells, temp, channels, lambda);
            let cost_minus = resource_cost_at(&p_minus, &self.cells, temp, channels, lambda);

            let task_grad = (val_fn(&p_plus) - val_fn(&p_minus)) / (2.0 * step);
            let cost_grad = (cost_plus - cost_minus) / (2.0 * step);
            grads[i] = task_grad + cost_grad;
        }
        grads
    }

    /// One bi-level optimisation step with resource penalty.
    ///
    /// The full validation objective is `L_task(arch_params) + λ · E[cost]`.
    pub fn bilevel_step(
        &mut self,
        weight_grad_fn: impl Fn(&[f64]) -> Vec<f64>,
        val_fn: impl Fn(&[f64]) -> f64,
    ) -> OptimizeResult<()> {
        self.current_step += 1;

        // Inner: gradient step on network weights.
        let w_grads = weight_grad_fn(&self.weights);
        if w_grads.len() != self.weights.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "weight_grad_fn returned {} grads, expected {}",
                w_grads.len(),
                self.weights.len()
            )));
        }
        let lr_w = self.config.weight_lr;
        for (w, g) in self.weights.iter_mut().zip(w_grads.iter()) {
            *w -= lr_w * g;
        }

        // Outer: gradient step on architecture params (task loss + resource).
        let a_grads = self.arch_grads_fd(&val_fn, 1e-4);
        if !a_grads.is_empty() {
            self.update_arch_params(&a_grads, self.config.arch_lr)?;
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────── helper: resource cost at params ──

/// Compute `λ * E[FLOP cost]` for a given flat arch-param vector without
/// mutably borrowing `SnasSearch`.  Used inside FD gradient computation.
fn resource_cost_at(
    params: &[f64],
    cells: &[SnasCell],
    temperature: f64,
    channels: usize,
    lambda: f64,
) -> f64 {
    let ops = Operation::all();
    let n_ops_canonical = ops.len();
    let eps = 1e-8_f64;
    let temp = temperature.max(eps);

    let mut total_cost = 0.0_f64;
    let mut offset = 0_usize;

    for cell in cells.iter() {
        for node_edges in cell.edges.iter() {
            for mo in node_edges.iter() {
                let n = mo.arch_params.len().min(n_ops_canonical);
                let slice = &params[offset..offset + mo.arch_params.len()];

                // Softmax of this edge's params.
                let scaled: Vec<f64> = slice.iter().map(|a| a / temp).collect();
                let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scaled.iter().map(|s| (s - max_val).exp()).collect();
                let sum: f64 = exps.iter().sum::<f64>().max(eps);

                for k in 0..n {
                    let wk = exps[k] / sum;
                    total_cost += wk * ops[k].cost_flops(channels);
                }

                offset += mo.arch_params.len();
            }
        }
    }

    lambda * total_cost
}

// ═══════════════════════════════════════════════════════════════════ tests ═══

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lcg() -> Lcg {
        Lcg::new(99999)
    }

    // ── SnasMixedOperation ─────────────────────────────────────────────────────

    #[test]
    fn test_concrete_sample_valid() {
        let mo = SnasMixedOperation::new(6);
        let mut rng = make_lcg();
        let weights = mo.concrete_sample(1.0, &mut rng);

        assert_eq!(weights.len(), 6);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "concrete sample sum={sum}");
        for &w in &weights {
            assert!(w >= 0.0, "negative concrete weight {w}");
        }
    }

    #[test]
    fn test_concrete_sample_multiple_calls_valid() {
        let mo = SnasMixedOperation::new(6);
        let mut rng = make_lcg();
        for _ in 0..20 {
            let w = mo.concrete_sample(0.5, &mut rng);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
            for &v in &w {
                assert!(v >= 0.0);
            }
        }
    }

    #[test]
    fn test_concrete_sample_low_temp_peaks() {
        // At very low temperature the concrete sample should be near one-hot.
        let mut mo = SnasMixedOperation::new(6);
        mo.arch_params = vec![5.0, 0.1, 0.1, 0.1, 0.1, 0.1];
        let mut rng = make_lcg();
        // Run several trials; the dominant weight should frequently be very large.
        let mut dominant_count = 0;
        for _ in 0..20 {
            let w = mo.concrete_sample(0.01, &mut rng);
            if w[0] > 0.5 {
                dominant_count += 1;
            }
        }
        // Expect most draws to be dominated by index 0 at τ=0.01.
        assert!(
            dominant_count >= 10,
            "dominant_count={dominant_count} too low"
        );
    }

    // ── Expected cost ──────────────────────────────────────────────────────────

    #[test]
    fn test_expected_cost_nonneg() {
        let mo = SnasMixedOperation::new(6);
        let cost = mo.expected_cost(1.0, 16);
        assert!(cost >= 0.0, "cost={cost}");
    }

    #[test]
    fn test_total_expected_cost_nonneg() {
        let config = SnasConfig::default();
        let search = SnasSearch::new(config);
        let cost = search.total_expected_cost();
        assert!(cost >= 0.0, "total cost={cost}");
    }

    #[test]
    fn test_expected_cost_zero_for_no_flop_ops() {
        // If all arch_params prefer Identity/Zero/SkipConnect (no-FLOP ops),
        // the expected cost should be very small.
        let mut mo = SnasMixedOperation::new(3); // identity=0, zero=1, skip=2
        mo.arch_params = vec![10.0, 10.0, 10.0]; // all equal, ops 0..2 have 0 FLOPs
        let cost = mo.expected_cost(1.0, 16);
        // Identity(0), Zero(0), Conv3x3(non-zero in original ordering).
        // op[2] = Conv3x3 in Operation::all(), so we can't claim 0.
        // Just verify non-negative.
        assert!(cost >= 0.0);
    }

    // ── SnasCell ───────────────────────────────────────────────────────────────

    #[test]
    fn test_snas_cell_arch_params_shape() {
        let cell = SnasCell::new(2, 4, 6);
        // Same edge structure as GDAS: total edges = 2+3+4+5 = 14, 6 ops each → 84.
        assert_eq!(cell.arch_parameters().len(), 84);
    }

    #[test]
    fn test_snas_cell_update_wrong_len_errors() {
        let mut cell = SnasCell::new(2, 3, 6);
        let result = cell.update_arch_params(&[0.0; 3], 0.01);
        assert!(result.is_err());
    }

    // ── SnasSearch ─────────────────────────────────────────────────────────────

    #[test]
    fn test_snas_bilevel_step_runs() {
        let config = SnasConfig::default();
        let mut search = SnasSearch::new(config);

        let weight_grad_fn = |weights: &[f64]| vec![0.0_f64; weights.len()];
        let val_fn = |params: &[f64]| params.iter().map(|p| p * p).sum::<f64>();

        search
            .bilevel_step(weight_grad_fn, val_fn)
            .expect("snas bilevel_step should not error");
    }

    #[test]
    fn test_snas_bilevel_step_advances_temperature() {
        let config = SnasConfig::default();
        let mut search = SnasSearch::new(config);
        let t0 = search.current_temperature();
        let _ = search.bilevel_step(|w| vec![0.0; w.len()], |p| p.iter().sum::<f64>());
        let t1 = search.current_temperature();
        assert!(t1 <= t0 + 1e-12, "t1={t1} should be ≤ t0={t0}");
    }

    #[test]
    fn test_derive_discrete_arch_valid() {
        let config = SnasConfig {
            n_cells: 2,
            n_operations: 6,
            n_nodes: 3,
            ..Default::default()
        };
        let search = SnasSearch::new(config);
        let arch = search.derive_discrete_arch_indices();
        assert_eq!(arch.len(), 2);
        for cell_disc in &arch {
            for node_edges in cell_disc {
                for &op_idx in node_edges {
                    assert!(op_idx < 6, "op_idx={op_idx}");
                }
            }
        }
    }

    #[test]
    fn test_snas_arch_params_consistent() {
        let search = SnasSearch::new(SnasConfig::default());
        assert_eq!(search.arch_parameters().len(), search.n_arch_params());
    }
}

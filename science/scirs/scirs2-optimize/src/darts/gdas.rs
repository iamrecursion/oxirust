//! GDAS: Searching for A Robust Neural Architecture in Four GPU Hours
//! (Dong & Yang, CVPR 2019).
//!
//! GDAS replaces the continuous softmax mixing of DARTS with Gumbel-Softmax
//! sampling: each forward pass through a cell selects **one** operation per
//! edge via a Gumbel-Softmax draw.  The hard argmax is used for the forward
//! computation (straight-through estimator), while the soft Gumbel-Softmax
//! weights are retained for gradient flow through the architecture parameters.
//!
//! ## References
//!
//! - Dong, X. and Yang, Y. (2019). "Searching for A Robust Neural Architecture
//!   in Four GPU Hours: A Practical Neural Architecture Search Approach".
//!   CVPR 2019.

use super::{AnnealingStrategy, Lcg, TemperatureSchedule};
use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────── GdasConfig ──

/// Configuration for a GDAS architecture search experiment.
#[derive(Debug, Clone)]
pub struct GdasConfig {
    /// Number of cells stacked in the super-network.
    pub n_cells: usize,
    /// Number of candidate operations per edge (matches `Operation::all().len()`).
    pub n_operations: usize,
    /// Number of feature channels (used for FLOP estimation).
    pub channels: usize,
    /// Number of intermediate nodes per cell.
    pub n_nodes: usize,
    /// Learning rate for architecture parameter updates.
    pub arch_lr: f64,
    /// Learning rate for network weight updates.
    pub weight_lr: f64,
    /// Temperature schedule for Gumbel-Softmax.
    pub temperature_schedule: TemperatureSchedule,
    /// Random seed for the internal LCG.
    pub seed: u64,
}

impl Default for GdasConfig {
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
            seed: 42,
        }
    }
}

// ──────────────────────────────────────────────────── GdasMixedOperation ──

/// One mixed operation on a directed edge in the GDAS cell DAG.
///
/// Architecture parameters are stored as un-normalised log-weights (`α_k`).
/// During search a Gumbel-Softmax sample decides which single operation is
/// executed (straight-through hard selection), but the soft weights are kept
/// for gradient computation.
#[derive(Debug, Clone)]
pub struct GdasMixedOperation {
    /// Un-normalised architecture parameters `α_k`, one per operation.
    pub arch_params: Vec<f64>,
    /// Operation index selected in the last forward pass.
    pub last_selected: usize,
    /// Soft Gumbel-Softmax weights from the last forward pass.
    pub last_soft_weights: Vec<f64>,
}

impl GdasMixedOperation {
    /// Create a new `GdasMixedOperation` with `n_ops` operations initialised
    /// to uniform architecture weights (all log-weights = 0).
    pub fn new(n_ops: usize) -> Self {
        Self {
            arch_params: vec![0.0_f64; n_ops],
            last_selected: 0,
            last_soft_weights: vec![1.0 / n_ops as f64; n_ops],
        }
    }

    /// Draw a Gumbel-Softmax sample.
    ///
    /// Returns `(selected_index, soft_weights)` where `selected_index` is the
    /// argmax of `(α_k + gumbel_noise_k) / temperature` and `soft_weights` is
    /// the corresponding softmax distribution (used for gradient flow).
    pub fn gumbel_softmax_sample(&self, temperature: f64, rng: &mut Lcg) -> (usize, Vec<f64>) {
        let eps = 1e-20_f64;
        let temp = temperature.max(1e-8);
        let n = self.arch_params.len();

        // Perturb arch_params with Gumbel noise: g_k = -log(-log(u_k)).
        let mut logits = vec![0.0_f64; n];
        for k in 0..n {
            let u = rng.next_f64().max(eps);
            let gumbel_noise = -(-u.ln()).ln();
            logits[k] = self.arch_params[k] + gumbel_noise;
        }

        // Scale by temperature, then softmax (numerically stable).
        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exp_vals: Vec<f64> = logits.iter().map(|&l| ((l - max_l) / temp).exp()).collect();
        let sum = exp_vals.iter().sum::<f64>().max(eps);
        for v in &mut exp_vals {
            *v /= sum;
        }

        // Hard selection: argmax of exp_vals (equivalent to argmax of logits).
        let selected = exp_vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        (selected, exp_vals)
    }

    /// Compute standard softmax-normalised operation weights at `temperature`.
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

    /// Index of the operation with the highest architecture weight.
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

// ─────────────────────────────────────────────────────────────── GdasCell ──

/// A GDAS cell: DAG with fixed input nodes and learnable intermediate nodes.
/// Each edge carries a `GdasMixedOperation`.
#[derive(Debug, Clone)]
pub struct GdasCell {
    /// Number of intermediate (learnable) nodes.
    pub n_nodes: usize,
    /// Number of fixed input nodes (typically 2).
    pub n_input_nodes: usize,
    /// `edges[i][j]` is the `GdasMixedOperation` from node j to intermediate
    /// node i.  Node indices: 0..n_input_nodes are inputs.
    pub edges: Vec<Vec<GdasMixedOperation>>,
}

impl GdasCell {
    /// Create a new GDAS cell.
    pub fn new(n_input_nodes: usize, n_intermediate_nodes: usize, n_ops: usize) -> Self {
        let edges: Vec<Vec<GdasMixedOperation>> = (0..n_intermediate_nodes)
            .map(|i| {
                let n_predecessors = n_input_nodes + i;
                (0..n_predecessors)
                    .map(|_| GdasMixedOperation::new(n_ops))
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

    /// Apply gradient updates to architecture parameters.
    ///
    /// `grads` must have the same length as `arch_parameters()`.
    pub fn update_arch_params(&mut self, grads: &[f64], lr: f64) -> OptimizeResult<()> {
        let n_params: usize = self
            .edges
            .iter()
            .flat_map(|row| row.iter())
            .map(|mo| mo.arch_params.len())
            .sum();
        if grads.len() != n_params {
            return Err(OptimizeError::InvalidInput(format!(
                "GdasCell::update_arch_params: expected {n_params} grads, got {}",
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

// ──────────────────────────────────────────────────────────── GdasSearch ──

/// Top-level GDAS search controller.
///
/// Manages a stack of `GdasCell`s, an internal LCG, and the bi-level
/// Gumbel-Softmax optimisation loop.
pub struct GdasSearch {
    /// Stack of cells forming the super-network.
    pub cells: Vec<GdasCell>,
    /// Configuration.
    pub config: GdasConfig,
    /// Flat network weights (one scalar per cell in this toy model).
    weights: Vec<f64>,
    /// Internal pseudo-random number generator.
    rng: Lcg,
    /// Current training step (used for temperature scheduling).
    current_step: usize,
}

impl GdasSearch {
    /// Construct a `GdasSearch` from the given config.
    pub fn new(config: GdasConfig) -> Self {
        let cells: Vec<GdasCell> = (0..config.n_cells)
            .map(|_| GdasCell::new(2, config.n_nodes, config.n_operations))
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

    /// Current Gumbel-Softmax temperature according to the schedule.
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

    /// Apply a gradient step to architecture parameters.
    pub fn update_arch_params(&mut self, grads: &[f64], lr: f64) -> OptimizeResult<()> {
        let total = self.n_arch_params();
        if grads.len() != total {
            return Err(OptimizeError::InvalidInput(format!(
                "GdasSearch::update_arch_params: expected {total} grads, got {}",
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

    /// Derive discrete architecture: for each cell, argmax op index per edge.
    ///
    /// Returns `Vec<Vec<Vec<usize>>>` — `[cell][intermediate_node][predecessor]`.
    pub fn derive_discrete_arch_indices(&self) -> Vec<Vec<Vec<usize>>> {
        self.cells.iter().map(|c| c.derive_discrete()).collect()
    }

    /// Compute finite-difference gradients of `val_fn` with respect to the
    /// architecture parameters.  Uses central differences with step `step`.
    pub fn arch_grads_fd(&self, val_fn: impl Fn(&[f64]) -> f64, step: f64) -> Vec<f64> {
        let params = self.arch_parameters();
        let n = params.len();
        let mut grads = vec![0.0_f64; n];
        for i in 0..n {
            let mut p_plus = params.clone();
            p_plus[i] += step;
            let mut p_minus = params.clone();
            p_minus[i] -= step;
            grads[i] = (val_fn(&p_plus) - val_fn(&p_minus)) / (2.0 * step);
        }
        grads
    }

    /// One bi-level optimisation step.
    ///
    /// 1. Increment internal step counter (advances temperature schedule).
    /// 2. Inner step: call `weight_grad_fn` on current weights, update weights.
    /// 3. Outer step: compute FD gradients via `val_fn`, update arch params.
    ///
    /// # Arguments
    /// - `weight_grad_fn`: Returns gradient of train loss w.r.t. `self.weights`.
    /// - `val_fn`: Returns validation loss as a function of flat arch params.
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

        // Outer: FD gradient step on architecture params.
        let a_grads = self.arch_grads_fd(&val_fn, 1e-4);
        if !a_grads.is_empty() {
            self.update_arch_params(&a_grads, self.config.arch_lr)?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════ tests ═══

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lcg() -> Lcg {
        Lcg::new(12345)
    }

    // ── GdasMixedOperation ─────────────────────────────────────────────────────

    #[test]
    fn test_gumbel_softmax_valid_distribution() {
        let mo = GdasMixedOperation::new(6);
        let mut rng = make_lcg();
        let (selected, soft) = mo.gumbel_softmax_sample(1.0, &mut rng);

        assert!(selected < 6, "selected={selected} out of range");
        assert_eq!(soft.len(), 6);

        let sum: f64 = soft.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "soft sum={sum}");
        for &w in &soft {
            assert!(w >= 0.0, "negative weight {w}");
        }
    }

    #[test]
    fn test_temperature_annealing_sharpens() {
        // With high temperature, Gumbel-Softmax is more uniform.
        // With low temperature, the winning logit dominates.
        let mut mo = GdasMixedOperation::new(6);
        mo.arch_params = vec![2.0, 0.5, 0.3, 0.1, 0.1, 0.0];
        // Use deterministic arch params for softmax (no noise).
        let w_hot = mo.weights(10.0);
        let w_cold = mo.weights(0.01);
        // Index 0 has the highest α; should dominate more at low temp.
        assert!(
            w_cold[0] > w_hot[0],
            "cold w[0]={} should > hot w[0]={}",
            w_cold[0],
            w_hot[0]
        );
        // Hot distribution should be closer to uniform.
        let entropy_hot: f64 = w_hot
            .iter()
            .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
            .sum();
        let entropy_cold: f64 = w_cold
            .iter()
            .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
            .sum();
        assert!(
            entropy_hot > entropy_cold,
            "hot entropy={entropy_hot} should > cold entropy={entropy_cold}"
        );
    }

    #[test]
    fn test_gumbel_softmax_selected_in_range() {
        let mo = GdasMixedOperation::new(7);
        let mut rng = make_lcg();
        for _ in 0..50 {
            let (sel, _) = mo.gumbel_softmax_sample(0.5, &mut rng);
            assert!(sel < 7, "sel={sel}");
        }
    }

    // ── GdasCell ───────────────────────────────────────────────────────────────

    #[test]
    fn test_gdas_cell_arch_params_shape() {
        let cell = GdasCell::new(2, 4, 6);
        let params = cell.arch_parameters();
        // Edges: node i has (2+i) predecessors, i in 0..4.
        // Total edges = 2+3+4+5 = 14; each has 6 params → 84.
        assert_eq!(params.len(), 84, "params.len()={}", params.len());
    }

    #[test]
    fn test_gdas_cell_update_wrong_len_errors() {
        let mut cell = GdasCell::new(2, 3, 6);
        let result = cell.update_arch_params(&[1.0, 2.0], 0.01);
        assert!(result.is_err());
    }

    // ── GdasSearch ─────────────────────────────────────────────────────────────

    #[test]
    fn test_gdas_bilevel_step_runs() {
        let config = GdasConfig::default();
        let mut search = GdasSearch::new(config);

        // Toy weight gradient function: just return zeros.
        let weight_grad_fn = |weights: &[f64]| vec![0.0_f64; weights.len()];
        // Toy validation loss: sum of arch params squared.
        let val_fn = |params: &[f64]| params.iter().map(|p| p * p).sum::<f64>();

        search
            .bilevel_step(weight_grad_fn, val_fn)
            .expect("bilevel_step should not error");
    }

    #[test]
    fn test_gdas_bilevel_step_advances_temperature() {
        let config = GdasConfig::default();
        let mut search = GdasSearch::new(config);
        let t0 = search.current_temperature();
        let _ = search.bilevel_step(|w| vec![0.0; w.len()], |p| p.iter().sum::<f64>());
        // Step count has advanced by 1; for exponential decay temp should decrease.
        let t1 = search.current_temperature();
        // t1 <= t0 for an exponential decay schedule (or very close).
        assert!(t1 <= t0 + 1e-12, "t1={t1} should be ≤ t0={t0}");
    }

    #[test]
    fn test_derive_discrete_arch_valid() {
        let config = GdasConfig {
            n_cells: 2,
            n_operations: 6,
            n_nodes: 3,
            ..Default::default()
        };
        let search = GdasSearch::new(config);
        let arch = search.derive_discrete_arch_indices();

        assert_eq!(arch.len(), 2);
        for cell_disc in &arch {
            for node_edges in cell_disc {
                for &op_idx in node_edges {
                    assert!(op_idx < 6, "op_idx={op_idx} >= n_operations=6");
                }
            }
        }
    }

    #[test]
    fn test_gdas_arch_parameters_length_consistent() {
        let config = GdasConfig::default();
        let search = GdasSearch::new(config);
        assert_eq!(search.arch_parameters().len(), search.n_arch_params());
    }

    #[test]
    fn test_gdas_update_arch_params_wrong_length_errors() {
        let mut search = GdasSearch::new(GdasConfig::default());
        let result = search.update_arch_params(&[1.0, 2.0], 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_gdas_weight_gradient_wrong_length_errors() {
        let mut search = GdasSearch::new(GdasConfig::default());
        // Gradient function returns wrong size.
        let bad_grad_fn = |_: &[f64]| vec![0.0_f64; 9999];
        let result = search.bilevel_step(bad_grad_fn, |p| p.iter().sum::<f64>());
        assert!(result.is_err());
    }
}

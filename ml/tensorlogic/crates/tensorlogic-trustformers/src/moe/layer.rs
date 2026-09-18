//! Numerical Mixture-of-Experts layer: [`TopKGate`] + `Vec<Box<dyn Expert>>`.
//!
//! Forward pass: for input `x`, obtain `top_k_indices` and
//! `top_k_softmax_weights` from the gate, dispatch `x` to each selected
//! expert, and combine outputs as
//! `y = Σ_{i in top_k} weight_i · expert_i.forward(x)`.
//!
//! The optional Switch-Transformer **capacity factor** (Fedus et al.,
//! 2022) caps each expert's per-batch token count at
//! `C = ceil(capacity_factor * batch_size / num_experts)`. Tokens routed
//! beyond capacity have their contribution from the overflowing expert
//! zeroed out rather than overloading the expert. Set
//! [`MoELayer::with_capacity_factor`] to `None` to disable the cap.

use ndarray::{Array1, Array2, ArrayView1};

use super::error::{MoeError, MoeResult};
use super::expert::Expert;
use super::gate::{GatingDecision, TopKGate};
use super::load_balance::BatchGatingStats;

/// Default capacity factor recommended by Fedus et al. (2022), §2.2.
pub const DEFAULT_CAPACITY_FACTOR: f64 = 1.25;

/// Mixture-of-Experts layer: gate + experts + optional capacity cap.
///
/// The struct contains `Box<dyn Expert>` which is not auto-`Debug`, so
/// a manual implementation is provided below.
pub struct MoELayer {
    gate: TopKGate,
    experts: Vec<Box<dyn Expert>>,
    capacity_factor: Option<f64>,
    d_in: usize,
    d_out: usize,
}

impl std::fmt::Debug for MoELayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoELayer")
            .field("num_experts", &self.experts.len())
            .field("d_in", &self.d_in)
            .field("d_out", &self.d_out)
            .field("capacity_factor", &self.capacity_factor)
            .finish()
    }
}

impl MoELayer {
    /// Construct from a gate and a list of experts.
    ///
    /// All experts must share the same input / output dimensions, and
    /// the gate's `num_experts` must match `experts.len()`. The
    /// capacity factor defaults to `None` (no capping); use
    /// [`Self::with_capacity_factor`] to enable Switch-style overflow
    /// handling.
    ///
    /// # Errors
    ///
    /// * [`MoeError::EmptyExpertPool`] if `experts` is empty.
    /// * [`MoeError::ShapeMismatch`] if experts disagree on dimensions,
    ///   or if `gate.num_experts()` disagrees with `experts.len()`, or if
    ///   `gate.d_model()` disagrees with the experts' input dim.
    pub fn new(gate: TopKGate, experts: Vec<Box<dyn Expert>>) -> MoeResult<Self> {
        if experts.is_empty() {
            return Err(MoeError::EmptyExpertPool);
        }
        if gate.num_experts() != experts.len() {
            return Err(MoeError::ShapeMismatch {
                expected: gate.num_experts(),
                got: experts.len(),
            });
        }
        let d_in = experts[0].input_dim();
        let d_out = experts[0].output_dim();
        if gate.d_model() != d_in {
            return Err(MoeError::ShapeMismatch {
                expected: gate.d_model(),
                got: d_in,
            });
        }
        for expert in experts.iter().skip(1) {
            if expert.input_dim() != d_in {
                return Err(MoeError::ShapeMismatch {
                    expected: d_in,
                    got: expert.input_dim(),
                });
            }
            if expert.output_dim() != d_out {
                return Err(MoeError::ShapeMismatch {
                    expected: d_out,
                    got: expert.output_dim(),
                });
            }
        }
        Ok(Self {
            gate,
            experts,
            capacity_factor: None,
            d_in,
            d_out,
        })
    }

    /// Builder: enable Switch-Transformer capacity-factor dropping with
    /// the supplied factor (must be strictly positive and finite).
    ///
    /// # Errors
    ///
    /// [`MoeError::InvalidCapacityFactor`] if the value is not finite
    /// or not strictly positive.
    pub fn with_capacity_factor(mut self, factor: f64) -> MoeResult<Self> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(MoeError::InvalidCapacityFactor { value: factor });
        }
        self.capacity_factor = Some(factor);
        Ok(self)
    }

    /// Builder: explicitly disable the capacity cap.
    pub fn without_capacity_factor(mut self) -> Self {
        self.capacity_factor = None;
        self
    }

    /// Access to the underlying gate.
    pub fn gate(&self) -> &TopKGate {
        &self.gate
    }

    /// Mutable access to the gate (for parameter updates).
    pub fn gate_mut(&mut self) -> &mut TopKGate {
        &mut self.gate
    }

    /// Number of experts in the pool.
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Current capacity factor, if enabled.
    pub fn capacity_factor(&self) -> Option<f64> {
        self.capacity_factor
    }

    /// Input feature dimension.
    pub fn input_dim(&self) -> usize {
        self.d_in
    }

    /// Output feature dimension.
    pub fn output_dim(&self) -> usize {
        self.d_out
    }

    /// Forward pass for a single input vector.
    ///
    /// Returns `(y, decision)` where `y` is the combined expert output
    /// and `decision` is the gate's top-k choice. No capacity capping
    /// is applied in this single-item path — capacity is a batch
    /// concept, handled by [`Self::forward_batch`].
    pub fn forward(&self, x: &ArrayView1<f64>) -> MoeResult<(Array1<f64>, GatingDecision)> {
        if x.len() != self.d_in {
            return Err(MoeError::ShapeMismatch {
                expected: self.d_in,
                got: x.len(),
            });
        }
        let decision = self.gate.forward(x)?;
        let mut output = Array1::<f64>::zeros(self.d_out);
        for (slot, &expert_idx) in decision.top_k_indices.iter().enumerate() {
            let weight = decision.top_k_softmax_weights[slot];
            let expert_out = self.experts[expert_idx].forward(x)?;
            // y += weight * expert_out
            output.scaled_add(weight, &expert_out);
        }
        Ok((output, decision))
    }

    /// Forward pass for a batch, applying capacity-factor dropping if
    /// enabled.
    ///
    /// `batch` has shape `(batch_size, d_in)`. Returns `(outputs, stats)`:
    ///
    /// * `outputs` of shape `(batch_size, d_out)`: combined expert
    ///   outputs. Tokens that overflow an expert's capacity have that
    ///   expert's contribution zeroed out (other top-k experts still
    ///   contribute normally).
    /// * `stats`: [`BatchGatingStats`] with the full softmax gate scores
    ///   per token and the top-1 routed expert per token (pre-capacity
    ///   adjustment — the auxiliary losses use the *intended* routing,
    ///   matching the Shazeer convention).
    pub fn forward_batch(
        &self,
        batch: &ndarray::ArrayView2<f64>,
    ) -> MoeResult<(Array2<f64>, BatchGatingStats)> {
        if batch.ncols() != self.d_in {
            return Err(MoeError::ShapeMismatch {
                expected: self.d_in,
                got: batch.ncols(),
            });
        }
        let batch_size = batch.nrows();
        let num_experts = self.experts.len();

        // Compute every gate decision up-front so we can plan capacity
        // before dispatching to experts. (The alternative — streaming
        // capacity tracking — yields identical behaviour but would be
        // harder to test deterministically.)
        let mut decisions: Vec<GatingDecision> = Vec::with_capacity(batch_size);
        let mut stats = BatchGatingStats::empty(batch_size, num_experts);
        for (t, row) in batch.rows().into_iter().enumerate() {
            let decision = self.gate.forward(&row)?;
            // Populate the stats row with the full softmax.
            let full = decision.full_softmax();
            for (i, v) in full.iter().enumerate() {
                stats.gate_scores_per_token[(t, i)] = *v;
            }
            // Primary (top-1) expert for the hard-count load loss.
            let primary = decision.top_k_indices[0];
            stats.routed_expert_per_token.push(primary);
            decisions.push(decision);
        }

        // Plan per-expert capacity. If disabled, use `usize::MAX` so
        // the "has room" check is always true.
        let capacity: usize = match self.capacity_factor {
            Some(factor) => {
                let cap = (factor * batch_size as f64 / num_experts as f64).ceil() as usize;
                cap.max(1)
            }
            None => usize::MAX,
        };
        let mut assigned_counts = vec![0_usize; num_experts];

        let mut outputs = Array2::<f64>::zeros((batch_size, self.d_out));
        for (t, decision) in decisions.iter().enumerate() {
            let row = batch.row(t);
            for (slot, &expert_idx) in decision.top_k_indices.iter().enumerate() {
                if assigned_counts[expert_idx] >= capacity {
                    // Capacity-factor drop: this expert's contribution
                    // to this token is zero. The rest of the top-k for
                    // this token may still contribute.
                    continue;
                }
                assigned_counts[expert_idx] += 1;
                let weight = decision.top_k_softmax_weights[slot];
                let expert_out = self.experts[expert_idx].forward(&row)?;
                let mut row_view = outputs.row_mut(t);
                // row += weight * expert_out
                for (o, e) in row_view.iter_mut().zip(expert_out.iter()) {
                    *o += weight * *e;
                }
            }
        }

        Ok((outputs, stats))
    }
}

#[cfg(test)]
mod local_tests {
    use super::*;
    use crate::moe::expert::LinearExpert;
    use ndarray::{array, Array2};

    fn linear_identity(dim: usize) -> LinearExpert {
        let weights = Array2::<f64>::eye(dim);
        let bias = Array1::<f64>::zeros(dim);
        LinearExpert::from_arrays(weights, bias).expect("construct identity expert")
    }

    #[test]
    fn new_rejects_empty_pool() {
        let gate = TopKGate::xavier_init(2, 2, 1, 0).expect("gate");
        let err = MoELayer::new(gate, Vec::new()).expect_err("must fail");
        assert_eq!(err, MoeError::EmptyExpertPool);
    }

    #[test]
    fn new_rejects_gate_expert_count_mismatch() {
        let gate = TopKGate::xavier_init(2, 3, 1, 0).expect("gate");
        let experts: Vec<Box<dyn Expert>> = vec![Box::new(linear_identity(2))];
        let err = MoELayer::new(gate, experts).expect_err("must fail");
        assert!(matches!(err, MoeError::ShapeMismatch { .. }));
    }

    #[test]
    fn single_forward_matches_hand_computation() {
        // Two identity experts, k=2, so output = (w0 + w1) * x = x.
        let gate = TopKGate::xavier_init(2, 2, 2, 7).expect("gate");
        let experts: Vec<Box<dyn Expert>> =
            vec![Box::new(linear_identity(2)), Box::new(linear_identity(2))];
        let layer = MoELayer::new(gate, experts).expect("layer");

        let x = array![1.5_f64, -2.5];
        let (y, decision) = layer.forward(&x.view()).expect("forward");
        assert_eq!(decision.top_k_indices.len(), 2);
        for (a, b) in y.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-12, "identity pass-through failed");
        }
    }
}

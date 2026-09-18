//! Top-K gating network for the research-preview MoE layer.
//!
//! Given an input vector `x ∈ R^d`, the gate computes `g = W_g x ∈ R^E`
//! (where `E` is the number of experts), selects the top-`k` indices by
//! raw logit value, and applies softmax **only over the selected `k`
//! logits** (not over the full `E`). This matches the Shazeer-style
//! top-k gating of Shazeer et al. (2017) §2.1 and the Switch variant
//! (`k = 1`) of Fedus et al. (2022) §2.
//!
//! The full `raw_logits` vector is also returned inside the
//! [`GatingDecision`] so downstream callers can feed it to the
//! importance- and load-balancing losses in [`super::load_balance`].

use ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{Normal, SeedableRng, StdRng};
use smallvec::SmallVec;

use super::error::{MoeError, MoeResult};

/// Output of a single gating decision.
///
/// The `top_k_indices` and `top_k_softmax_weights` vectors are parallel:
/// `top_k_softmax_weights[i]` is the gate weight for the expert at
/// `top_k_indices[i]`. Softmax is computed over the selected `k` logits
/// only, so the weights sum to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct GatingDecision {
    /// Indices of the selected top-k experts (length `k`).
    pub top_k_indices: SmallVec<[usize; 8]>,
    /// Softmax weights over the selected top-k logits (length `k`, sums to 1).
    pub top_k_softmax_weights: SmallVec<[f64; 8]>,
    /// Raw gate logits over *all* experts (length `num_experts`).
    ///
    /// Exposed so auxiliary losses (importance / load balance) can
    /// examine the full distribution, not just the selected subset.
    pub raw_logits: Vec<f64>,
}

impl GatingDecision {
    /// Number of experts this decision routes to.
    pub fn k(&self) -> usize {
        self.top_k_indices.len()
    }

    /// Total number of experts in the pool (equal to `raw_logits.len()`).
    pub fn num_experts(&self) -> usize {
        self.raw_logits.len()
    }

    /// Full softmax distribution over *all* experts (length `num_experts`,
    /// sums to 1). Useful for the importance loss in
    /// [`super::load_balance`].
    pub fn full_softmax(&self) -> Vec<f64> {
        let n = self.raw_logits.len();
        if n == 0 {
            return Vec::new();
        }
        let max_logit = self
            .raw_logits
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut out = Vec::with_capacity(n);
        let mut sum = 0.0_f64;
        for &logit in &self.raw_logits {
            let e = (logit - max_logit).exp();
            sum += e;
            out.push(e);
        }
        if sum > 0.0 {
            for v in &mut out {
                *v /= sum;
            }
        } else {
            // Degenerate (all -inf); fall back to uniform so downstream
            // consumers never see NaNs.
            let uniform = 1.0_f64 / n as f64;
            out.fill(uniform);
        }
        out
    }
}

/// Top-K gating network: `y = softmax(top_k(W_g x))`.
#[derive(Debug, Clone)]
pub struct TopKGate {
    /// Gate projection matrix of shape `(num_experts, d_model)`.
    weights: Array2<f64>,
    /// `k` in top-k.
    k: usize,
}

impl TopKGate {
    /// Build a gate directly from a `(num_experts, d_model)` weight matrix.
    ///
    /// # Errors
    ///
    /// * [`MoeError::EmptyExpertPool`] if `weights.nrows() == 0`.
    /// * [`MoeError::InvalidTopK`] if `k == 0` or `k > num_experts`.
    /// * [`MoeError::ShapeMismatch`] if `weights.ncols() == 0`.
    pub fn from_weights(weights: Array2<f64>, k: usize) -> MoeResult<Self> {
        let num_experts = weights.nrows();
        let d_model = weights.ncols();
        if num_experts == 0 {
            return Err(MoeError::EmptyExpertPool);
        }
        if d_model == 0 {
            return Err(MoeError::ShapeMismatch {
                expected: 1,
                got: 0,
            });
        }
        if k == 0 || k > num_experts {
            return Err(MoeError::InvalidTopK { k, num_experts });
        }
        Ok(Self { weights, k })
    }

    /// Xavier / Glorot-normal initialisation for the gate weights,
    /// seeded by a `scirs2_core::random::StdRng`. Default `k = 2`.
    ///
    /// Pass `k = 1` for Switch-Transformer style routing.
    ///
    /// # Errors
    ///
    /// See [`Self::from_weights`].
    pub fn xavier_init(d_model: usize, num_experts: usize, k: usize, seed: u64) -> MoeResult<Self> {
        if num_experts == 0 {
            return Err(MoeError::EmptyExpertPool);
        }
        if d_model == 0 {
            return Err(MoeError::ShapeMismatch {
                expected: 1,
                got: 0,
            });
        }
        if k == 0 || k > num_experts {
            return Err(MoeError::InvalidTopK { k, num_experts });
        }
        let std = (2.0_f64 / (d_model + num_experts) as f64).sqrt();
        let dist = Normal::new(0.0, std).map_err(|_| MoeError::ShapeMismatch {
            expected: 1,
            got: 0,
        })?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = Array2::<f64>::zeros((num_experts, d_model));
        for value in weights.iter_mut() {
            *value = rng.sample(dist);
        }
        Ok(Self { weights, k })
    }

    /// Number of experts this gate routes over.
    pub fn num_experts(&self) -> usize {
        self.weights.nrows()
    }

    /// Input feature dimension.
    pub fn d_model(&self) -> usize {
        self.weights.ncols()
    }

    /// `k` parameter (top-k routing).
    pub fn k(&self) -> usize {
        self.k
    }

    /// Mutate `k` in place. Useful when toggling between top-k and
    /// Switch (`k = 1`) routing without rebuilding the gate.
    ///
    /// # Errors
    ///
    /// [`MoeError::InvalidTopK`] if `k == 0` or `k > num_experts`.
    pub fn set_k(&mut self, k: usize) -> MoeResult<()> {
        if k == 0 || k > self.num_experts() {
            return Err(MoeError::InvalidTopK {
                k,
                num_experts: self.num_experts(),
            });
        }
        self.k = k;
        Ok(())
    }

    /// Gate weight matrix view.
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Compute raw logits `g = W_g x` for the full expert pool.
    pub fn logits(&self, x: &ArrayView1<f64>) -> MoeResult<Array1<f64>> {
        if x.len() != self.d_model() {
            return Err(MoeError::ShapeMismatch {
                expected: self.d_model(),
                got: x.len(),
            });
        }
        Ok(self.weights.dot(x))
    }

    /// Full forward: compute logits, pick top-k, softmax over the
    /// selected `k` logits only.
    pub fn forward(&self, x: &ArrayView1<f64>) -> MoeResult<GatingDecision> {
        let logits = self.logits(x)?;
        let raw_logits_vec: Vec<f64> = logits.to_vec();

        // Argsort descending by logit. Ties broken by ascending index
        // (stable) so the result is fully deterministic given equal
        // logits.
        let mut order: Vec<usize> = (0..raw_logits_vec.len()).collect();
        order.sort_by(|&a, &b| {
            raw_logits_vec[b]
                .partial_cmp(&raw_logits_vec[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        let mut top_k_indices: SmallVec<[usize; 8]> = SmallVec::new();
        let mut top_k_logits: SmallVec<[f64; 8]> = SmallVec::new();
        for &idx in order.iter().take(self.k) {
            top_k_indices.push(idx);
            top_k_logits.push(raw_logits_vec[idx]);
        }

        // Softmax over the selected k logits only (numerically stable).
        let max_logit = top_k_logits
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut exp_values: SmallVec<[f64; 8]> = SmallVec::new();
        let mut sum = 0.0_f64;
        for &lg in top_k_logits.iter() {
            let e = (lg - max_logit).exp();
            sum += e;
            exp_values.push(e);
        }
        let mut top_k_softmax_weights: SmallVec<[f64; 8]> = SmallVec::new();
        if sum > 0.0 {
            for e in exp_values.iter() {
                top_k_softmax_weights.push(*e / sum);
            }
        } else {
            // Degenerate — fall back to uniform over the top-k to keep
            // the weights well-defined.
            let uniform = 1.0_f64 / self.k as f64;
            for _ in 0..self.k {
                top_k_softmax_weights.push(uniform);
            }
        }

        Ok(GatingDecision {
            top_k_indices,
            top_k_softmax_weights,
            raw_logits: raw_logits_vec,
        })
    }
}

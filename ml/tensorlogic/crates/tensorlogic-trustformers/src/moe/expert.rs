//! The pluggable [`Expert`] trait and a reference [`LinearExpert`].
//!
//! `Expert` is deliberately narrow — a single-vector forward pass — so
//! downstream users can plug in arbitrary feed-forward blocks (dense,
//! gated, MLP-with-hidden-activation, etc.) without fighting the MoE
//! dispatch machinery in [`super::layer`].
//!
//! The reference [`LinearExpert`] is enough for unit-testing the MoE
//! contract: it computes `y = W x + b`. Initialisation uses a Xavier /
//! Glorot normal distribution seeded by a `scirs2_core::random::StdRng`
//! — *no* `rand` / `rand_distr`, per the workspace RNG policy.

use ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{Normal, SeedableRng, StdRng};

use super::error::{MoeError, MoeResult};

/// Pluggable expert contract.
///
/// An expert is a deterministic, parameterised map from one input vector
/// (`ArrayView1<f64>` of shape `d_in`) to one output vector
/// (`Array1<f64>` of shape `d_out`). The output dimension is reported
/// by [`Expert::output_dim`]; the input dimension by
/// [`Expert::input_dim`].
///
/// Implementors must be `Send + Sync` so MoE layers can be shared across
/// threads.
pub trait Expert: Send + Sync {
    /// Input feature dimension consumed by [`Expert::forward`].
    fn input_dim(&self) -> usize;

    /// Output feature dimension produced by [`Expert::forward`].
    fn output_dim(&self) -> usize;

    /// Forward-propagate a single input vector.
    ///
    /// # Errors
    ///
    /// Returns [`MoeError::ShapeMismatch`] if `x.len() != self.input_dim()`.
    fn forward(&self, x: &ArrayView1<f64>) -> MoeResult<Array1<f64>>;
}

/// Reference implementation: a single-layer affine expert `y = W x + b`.
///
/// `weights` has shape `(d_out, d_in)`; `bias` has shape `(d_out,)`.
#[derive(Debug, Clone)]
pub struct LinearExpert {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl LinearExpert {
    /// Construct directly from a `(d_out, d_in)` weight matrix and a
    /// `d_out`-long bias vector.
    ///
    /// # Errors
    ///
    /// * [`MoeError::ShapeMismatch`] if `bias.len() != weights.nrows()`.
    pub fn from_arrays(weights: Array2<f64>, bias: Array1<f64>) -> MoeResult<Self> {
        if bias.len() != weights.nrows() {
            return Err(MoeError::ShapeMismatch {
                expected: weights.nrows(),
                got: bias.len(),
            });
        }
        Ok(Self { weights, bias })
    }

    /// Construct an expert with all-zero weights and bias.
    ///
    /// Useful as a baseline in tests where we want to distinguish the
    /// contribution of a specific expert from its neighbours.
    pub fn zeros(d_in: usize, d_out: usize) -> MoeResult<Self> {
        if d_in == 0 || d_out == 0 {
            return Err(MoeError::ShapeMismatch {
                expected: d_in.max(1),
                got: 0,
            });
        }
        Ok(Self {
            weights: Array2::zeros((d_out, d_in)),
            bias: Array1::zeros(d_out),
        })
    }

    /// Construct with Xavier / Glorot-normal weights (`std = sqrt(2 / (d_in + d_out))`)
    /// and zero bias, using a seeded `scirs2_core::random::StdRng`.
    ///
    /// # Errors
    ///
    /// * [`MoeError::ShapeMismatch`] if either dimension is zero.
    /// * [`MoeError::ShapeMismatch`] (with `expected == 1`, `got == 0`) if
    ///   the Xavier standard deviation collapses to zero — effectively
    ///   impossible for positive dimensions but reported defensively.
    pub fn xavier_init(d_in: usize, d_out: usize, seed: u64) -> MoeResult<Self> {
        if d_in == 0 || d_out == 0 {
            return Err(MoeError::ShapeMismatch {
                expected: d_in.max(1),
                got: 0,
            });
        }
        let std = (2.0_f64 / (d_in + d_out) as f64).sqrt();
        if !(std.is_finite() && std > 0.0) {
            return Err(MoeError::ShapeMismatch {
                expected: 1,
                got: 0,
            });
        }
        let dist = Normal::new(0.0, std).map_err(|e| MoeError::ShapeMismatch {
            // Re-use ShapeMismatch as a catch-all for "bad distribution" —
            // we never expect this to fire given a positive finite std.
            expected: 1,
            got: format!("{e}").len(),
        })?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = Array2::<f64>::zeros((d_out, d_in));
        for value in weights.iter_mut() {
            *value = rng.sample(dist);
        }
        let bias = Array1::<f64>::zeros(d_out);
        Ok(Self { weights, bias })
    }

    /// Immutable view of the weight matrix.
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Immutable view of the bias vector.
    pub fn bias(&self) -> &Array1<f64> {
        &self.bias
    }
}

impl Expert for LinearExpert {
    fn input_dim(&self) -> usize {
        self.weights.ncols()
    }

    fn output_dim(&self) -> usize {
        self.weights.nrows()
    }

    fn forward(&self, x: &ArrayView1<f64>) -> MoeResult<Array1<f64>> {
        if x.len() != self.input_dim() {
            return Err(MoeError::ShapeMismatch {
                expected: self.input_dim(),
                got: x.len(),
            });
        }
        // y = W x + b
        let mut out = self.weights.dot(x);
        out += &self.bias;
        Ok(out)
    }
}

/// Identity "expert" useful for round-trip tests in [`super::tests`].
///
/// Not exported at crate level — only constructible internally.
#[cfg(test)]
#[allow(dead_code)]
pub(super) struct IdentityExpert {
    dim: usize,
}

#[cfg(test)]
impl IdentityExpert {
    #[allow(dead_code)]
    pub(super) fn new(dim: usize) -> Self {
        Self { dim }
    }
}

#[cfg(test)]
impl Expert for IdentityExpert {
    fn input_dim(&self) -> usize {
        self.dim
    }

    fn output_dim(&self) -> usize {
        self.dim
    }

    fn forward(&self, x: &ArrayView1<f64>) -> MoeResult<Array1<f64>> {
        if x.len() != self.dim {
            return Err(MoeError::ShapeMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        Ok(x.to_owned())
    }
}

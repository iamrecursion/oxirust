//! Core LoRA layer: low-rank A/B decomposition of a frozen base weight.

use scirs2_core::random::{RngExt, SeedableRng, StdRng};
use std::f64::consts::PI;

use super::config::LoraConfig;
use super::error::{LoraError, LoraResult};

/// A single LoRA-augmented weight matrix.
///
/// Holds the frozen base weight `W` (d x k) and the low-rank factors
/// `B` (d x r, init zeros) and `A` (r x k, init Gaussian).
/// Forward: `output = input @ (W + scaling * B @ A)^T`.
pub struct LoraLayer {
    /// A matrix (r x k), initialised with random Gaussian N(0, 1/r).
    pub weight_a: Vec<Vec<f64>>,
    /// B matrix (d x r), initialised to zeros.
    pub weight_b: Vec<Vec<f64>>,
    /// Original frozen weight (d x k).
    pub base_weight: Vec<Vec<f64>>,
    /// Configuration.
    pub config: LoraConfig,
    /// Whether delta W has been merged into base_weight.
    pub merged: bool,
    /// RNG for dropout during forward pass.
    rng: StdRng,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn next_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> LoraResult<Vec<Vec<f64>>> {
    if a.is_empty() || b.is_empty() {
        return Err(LoraError::DimensionMismatch {
            expected: "non-empty matrices".into(),
            got: format!("a rows={}, b rows={}", a.len(), b.len()),
        });
    }
    let a_cols = a[0].len();
    let b_rows = b.len();
    if a_cols != b_rows {
        return Err(LoraError::DimensionMismatch {
            expected: format!("a_cols ({a_cols}) == b_rows"),
            got: format!("{b_rows}"),
        });
    }
    let b_cols = b[0].len();
    let mut out = vec![vec![0.0; b_cols]; a.len()];
    for i in 0..a.len() {
        for k in 0..a_cols {
            let a_ik = a[i][k];
            for j in 0..b_cols {
                out[i][j] += a_ik * b[k][j];
            }
        }
    }
    Ok(out)
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() {
        return Vec::new();
    }
    let rows = m.len();
    let cols = m[0].len();
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = m[i][j];
        }
    }
    t
}

fn add_matrices(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| x + y).collect())
        .collect()
}

fn sub_matrices(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| x - y).collect())
        .collect()
}

fn scale_matrix(s: f64, m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    m.iter()
        .map(|row| row.iter().map(|v| v * s).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// LoraLayer impl
// ---------------------------------------------------------------------------

impl LoraLayer {
    /// Create a new LoRA layer wrapping `base_weight` (d x k).
    ///
    /// `weight_a` is initialised from N(0, 1/r) and `weight_b` from zeros,
    /// so the initial delta W is the zero matrix.
    pub fn new(base_weight: Vec<Vec<f64>>, config: LoraConfig) -> LoraResult<Self> {
        let d = base_weight.len();
        if d == 0 {
            return Err(LoraError::DimensionMismatch {
                expected: "d > 0".into(),
                got: "0".into(),
            });
        }
        let k = base_weight[0].len();
        if k == 0 {
            return Err(LoraError::DimensionMismatch {
                expected: "k > 0".into(),
                got: "0".into(),
            });
        }

        let rank = config.rank;
        if rank == 0 || rank > d.min(k) {
            return Err(LoraError::InvalidRank(rank));
        }

        let mut rng = StdRng::seed_from_u64(config.seed);
        let stddev = 1.0 / (rank as f64).sqrt();

        // A: r x k, Gaussian N(0, 1/r)
        let weight_a: Vec<Vec<f64>> = (0..rank)
            .map(|_| (0..k).map(|_| next_normal(&mut rng) * stddev).collect())
            .collect();

        // B: d x r, zeros
        let weight_b: Vec<Vec<f64>> = vec![vec![0.0; rank]; d];

        Ok(Self {
            weight_a,
            weight_b,
            base_weight,
            config,
            merged: false,
            rng,
        })
    }

    fn scaling(&self) -> f64 {
        self.config.alpha / self.config.rank as f64
    }

    /// Compute `B @ A` (d x k).
    fn delta_weight(&self) -> LoraResult<Vec<Vec<f64>>> {
        matmul(&self.weight_b, &self.weight_a)
    }

    /// Compute the effective weight `W + scaling * B @ A` without mutating state.
    /// When merged, returns `base_weight` (the delta is already folded in).
    pub fn effective_weight(&self) -> LoraResult<Vec<Vec<f64>>> {
        if self.merged {
            return Ok(self.base_weight.clone());
        }
        let dw = self.delta_weight()?;
        Ok(add_matrices(
            &self.base_weight,
            &scale_matrix(self.scaling(), &dw),
        ))
    }

    /// Forward pass: `output = input @ effective_weight^T`.
    ///
    /// `input` has shape `(n, k)` and the result has shape `(n, d)`.
    /// When not merged, applies optional dropout on the LoRA branch.
    pub fn forward(&mut self, input: &[Vec<f64>]) -> LoraResult<Vec<Vec<f64>>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let k = self.base_weight[0].len();
        if input[0].len() != k {
            return Err(LoraError::DimensionMismatch {
                expected: format!("input cols = {k}"),
                got: format!("{}", input[0].len()),
            });
        }

        if self.merged {
            // base_weight already contains the delta
            let wt = transpose(&self.base_weight);
            return matmul(input, &wt);
        }

        // base contribution: input @ W^T
        let wt = transpose(&self.base_weight);
        let base_out = matmul(input, &wt)?;

        // LoRA branch: input @ A^T @ B^T * scaling
        let at = transpose(&self.weight_a);
        let mut lora_hidden = matmul(input, &at)?;

        // Apply dropout on the hidden activations if p > 0
        if self.config.dropout > 0.0 && self.config.dropout < 1.0 {
            let inv_keep = 1.0 / (1.0 - self.config.dropout);
            for row in &mut lora_hidden {
                for v in row.iter_mut() {
                    if self.rng.random::<f64>() < self.config.dropout {
                        *v = 0.0;
                    } else {
                        *v *= inv_keep;
                    }
                }
            }
        }

        let bt = transpose(&self.weight_b);
        let lora_out = matmul(&lora_hidden, &bt)?;
        let scaled = scale_matrix(self.scaling(), &lora_out);
        Ok(add_matrices(&base_out, &scaled))
    }

    /// Merge `scaling * B @ A` into `base_weight`.
    pub fn merge(&mut self) -> LoraResult<()> {
        if self.merged {
            return Err(LoraError::MergeError("already merged".into()));
        }
        let dw = self.delta_weight()?;
        self.base_weight = add_matrices(&self.base_weight, &scale_matrix(self.scaling(), &dw));
        self.merged = true;
        Ok(())
    }

    /// Remove `scaling * B @ A` from `base_weight`.
    pub fn unmerge(&mut self) -> LoraResult<()> {
        if !self.merged {
            return Err(LoraError::MergeError("not merged".into()));
        }
        let dw = self.delta_weight()?;
        self.base_weight = sub_matrices(&self.base_weight, &scale_matrix(self.scaling(), &dw));
        self.merged = false;
        Ok(())
    }

    /// Number of trainable parameters: `r * (d + k)`.
    pub fn trainable_params(&self) -> usize {
        let d = self.base_weight.len();
        let k = self.base_weight[0].len();
        self.config.rank * (d + k)
    }

    /// Total parameter count: `d * k + r * (d + k)`.
    pub fn total_params(&self) -> usize {
        let d = self.base_weight.len();
        let k = self.base_weight[0].len();
        d * k + self.trainable_params()
    }

    /// Fraction of trainable vs total parameters.
    pub fn compression_ratio(&self) -> f64 {
        self.trainable_params() as f64 / self.total_params() as f64
    }
}

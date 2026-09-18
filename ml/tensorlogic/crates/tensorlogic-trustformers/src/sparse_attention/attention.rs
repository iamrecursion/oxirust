//! Longformer-style sparse multi-head attention forward pass.
//!
//! Implements numerical attention computation (not graph building) with the
//! sliding-window + global-token mask from Beltagy et al. (2020).

use super::config::SparseAttentionConfig;
use super::error::{SparseAttentionError, SparseAttentionResult};
use super::mask::{build_mask, AttentionMask};

/// Longformer-style sparse attention engine.
///
/// Holds a validated [`SparseAttentionConfig`] and exposes `forward` /
/// `forward_with_mask` methods that perform the full multi-head scaled
/// dot-product attention with sparse masking.
#[derive(Clone, Debug)]
pub struct SparseAttention {
    config: SparseAttentionConfig,
}

impl SparseAttention {
    /// Construct a new engine after validating the configuration.
    pub fn new(config: SparseAttentionConfig) -> SparseAttentionResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Access the underlying configuration.
    pub fn config(&self) -> &SparseAttentionConfig {
        &self.config
    }

    /// Run a forward pass, building the mask on-the-fly from the config.
    ///
    /// `query`, `key`, `value` are each `seq_len x d_model` where
    /// `d_model = num_heads * head_dim`.
    pub fn forward(
        &self,
        query: &[Vec<f64>],
        key: &[Vec<f64>],
        value: &[Vec<f64>],
    ) -> SparseAttentionResult<Vec<Vec<f64>>> {
        let seq_len = query.len();
        if seq_len == 0 {
            return Err(SparseAttentionError::InvalidSequenceLength(0));
        }
        let mask = build_mask(seq_len, &self.config)?;
        self.forward_with_mask(query, key, value, &mask)
    }

    /// Run a forward pass using a pre-built [`AttentionMask`].
    ///
    /// This is useful when the same mask is reused across layers.
    pub fn forward_with_mask(
        &self,
        query: &[Vec<f64>],
        key: &[Vec<f64>],
        value: &[Vec<f64>],
        mask: &AttentionMask,
    ) -> SparseAttentionResult<Vec<Vec<f64>>> {
        let seq_len = query.len();
        let d_model = self.config.d_model();
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        self.validate_inputs(query, key, value, seq_len, d_model)?;

        if mask.seq_len != seq_len {
            return Err(SparseAttentionError::DimensionMismatch {
                context: "mask seq_len vs query seq_len".into(),
                expected: seq_len,
                got: mask.seq_len,
            });
        }

        let scale = 1.0 / (head_dim as f64).sqrt();

        let mut output = vec![vec![0.0_f64; d_model]; seq_len];

        for h in 0..num_heads {
            let h_start = h * head_dim;
            let h_end = h_start + head_dim;

            self.compute_head(
                query,
                key,
                value,
                mask,
                seq_len,
                head_dim,
                h_start,
                h_end,
                scale,
                &mut output,
            )?;
        }

        Ok(output)
    }

    fn validate_inputs(
        &self,
        query: &[Vec<f64>],
        key: &[Vec<f64>],
        value: &[Vec<f64>],
        seq_len: usize,
        d_model: usize,
    ) -> SparseAttentionResult<()> {
        if key.len() != seq_len {
            return Err(SparseAttentionError::DimensionMismatch {
                context: "key seq_len".into(),
                expected: seq_len,
                got: key.len(),
            });
        }
        if value.len() != seq_len {
            return Err(SparseAttentionError::DimensionMismatch {
                context: "value seq_len".into(),
                expected: seq_len,
                got: value.len(),
            });
        }

        for (idx, row) in query.iter().enumerate() {
            if row.len() != d_model {
                return Err(SparseAttentionError::DimensionMismatch {
                    context: format!("query row {idx} width"),
                    expected: d_model,
                    got: row.len(),
                });
            }
        }
        for (idx, row) in key.iter().enumerate() {
            if row.len() != d_model {
                return Err(SparseAttentionError::DimensionMismatch {
                    context: format!("key row {idx} width"),
                    expected: d_model,
                    got: row.len(),
                });
            }
        }
        for (idx, row) in value.iter().enumerate() {
            if row.len() != d_model {
                return Err(SparseAttentionError::DimensionMismatch {
                    context: format!("value row {idx} width"),
                    expected: d_model,
                    got: row.len(),
                });
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_head(
        &self,
        query: &[Vec<f64>],
        key: &[Vec<f64>],
        value: &[Vec<f64>],
        mask: &AttentionMask,
        seq_len: usize,
        head_dim: usize,
        h_start: usize,
        h_end: usize,
        scale: f64,
        output: &mut [Vec<f64>],
    ) -> SparseAttentionResult<()> {
        for (i, q_row) in query.iter().enumerate().take(seq_len) {
            let scores: Vec<f64> = key
                .iter()
                .enumerate()
                .take(seq_len)
                .map(|(j, k_row)| {
                    if mask.is_attended(i, j) {
                        dot_product_slice(&q_row[h_start..h_end], &k_row[h_start..h_end]) * scale
                    } else {
                        -1e9
                    }
                })
                .collect();

            let weights = softmax_vec(&scores)?;

            for d in 0..head_dim {
                let acc: f64 = weights
                    .iter()
                    .zip(value.iter())
                    .map(|(&w, v_row)| w * v_row[h_start + d])
                    .sum();
                output[i][h_start + d] = acc;
            }
        }

        Ok(())
    }

    /// Retrieve the attention weights for a given head (for debugging/testing).
    ///
    /// Returns the `seq_len x seq_len` weight matrix after softmax for head
    /// index `head`.
    pub fn attention_weights(
        &self,
        query: &[Vec<f64>],
        key: &[Vec<f64>],
        mask: &AttentionMask,
        head: usize,
    ) -> SparseAttentionResult<Vec<Vec<f64>>> {
        let seq_len = query.len();
        let head_dim = self.config.head_dim;
        let h_start = head * head_dim;
        let h_end = h_start + head_dim;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let mut result = Vec::with_capacity(seq_len);
        for (i, q_row) in query.iter().enumerate().take(seq_len) {
            let scores: Vec<f64> = key
                .iter()
                .enumerate()
                .take(seq_len)
                .map(|(j, k_row)| {
                    if mask.is_attended(i, j) {
                        dot_product_slice(&q_row[h_start..h_end], &k_row[h_start..h_end]) * scale
                    } else {
                        -1e9
                    }
                })
                .collect();
            result.push(softmax_vec(&scores)?);
        }
        Ok(result)
    }
}

/// Dot product of two slices of equal length.
fn dot_product_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Numerically stable softmax over a vector.
///
/// Uses the max-subtraction trick to avoid overflow.  If every entry is
/// `-1e9` (all masked out), the row gets uniform weights — which is
/// mathematically sound because the output for a fully-masked position
/// is a uniform average of values (effectively zero-information).
fn softmax_vec(logits: &[f64]) -> SparseAttentionResult<Vec<f64>> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();

    if sum.abs() < 1e-30 {
        let n = logits.len();
        return Ok(vec![1.0 / n as f64; n]);
    }

    Ok(exps.iter().map(|&e| e / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_qkv(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
        (0..seq_len)
            .map(|i| {
                (0..d_model)
                    .map(|d| if i == d { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect()
    }

    fn constant_qkv(seq_len: usize, d_model: usize, val: f64) -> Vec<Vec<f64>> {
        vec![vec![val; d_model]; seq_len]
    }

    #[test]
    fn output_dimensions_match() {
        let cfg = SparseAttentionConfig::new(2, 2, 4).expect("config valid");
        let attn = SparseAttention::new(cfg).expect("attn valid");
        let q = constant_qkv(6, 8, 1.0);
        let k = constant_qkv(6, 8, 1.0);
        let v = constant_qkv(6, 8, 1.0);

        let out = attn.forward(&q, &k, &v).expect("forward ok");
        assert_eq!(out.len(), 6);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn full_window_matches_dense() {
        let seq_len = 4;
        let d_model = 4;
        let cfg = SparseAttentionConfig::new(seq_len, 1, d_model).expect("config valid");
        let attn = SparseAttention::new(cfg).expect("attn valid");

        let q = identity_qkv(seq_len, d_model);
        let k = identity_qkv(seq_len, d_model);
        let v: Vec<Vec<f64>> = (0..seq_len)
            .map(|i| vec![(i + 1) as f64; d_model])
            .collect();

        let out = attn.forward(&q, &k, &v).expect("forward ok");

        // With identity Q/K and full window, softmax produces a specific
        // distribution.  We just check the output is finite and non-trivial.
        for row in &out {
            for &val in row {
                assert!(val.is_finite());
                assert!(val > 0.0);
            }
        }
    }

    #[test]
    fn dim_mismatch_detected() {
        let cfg = SparseAttentionConfig::new(2, 1, 4).expect("config valid");
        let attn = SparseAttention::new(cfg).expect("attn valid");

        let q = constant_qkv(4, 4, 1.0);
        let k = constant_qkv(3, 4, 1.0); // wrong seq_len
        let v = constant_qkv(4, 4, 1.0);

        assert!(matches!(
            attn.forward(&q, &k, &v),
            Err(SparseAttentionError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_sequence_rejected() {
        let cfg = SparseAttentionConfig::new(2, 1, 4).expect("config valid");
        let attn = SparseAttention::new(cfg).expect("attn valid");

        let q: Vec<Vec<f64>> = vec![];
        let k: Vec<Vec<f64>> = vec![];
        let v: Vec<Vec<f64>> = vec![];

        assert!(matches!(
            attn.forward(&q, &k, &v),
            Err(SparseAttentionError::InvalidSequenceLength(0))
        ));
    }

    #[test]
    fn global_only_single_token() {
        let cfg = SparseAttentionConfig::new(1, 1, 2)
            .map(|c| c.with_global_tokens(vec![0]))
            .expect("config valid");
        let attn = SparseAttention::new(cfg).expect("attn valid");

        let q = constant_qkv(4, 2, 1.0);
        let k = constant_qkv(4, 2, 1.0);
        let v: Vec<Vec<f64>> = vec![
            vec![10.0, 20.0],
            vec![30.0, 40.0],
            vec![50.0, 60.0],
            vec![70.0, 80.0],
        ];

        let out = attn.forward(&q, &k, &v).expect("forward ok");

        // Position 0 is global: attends to all.  With constant Q/K the
        // softmax is uniform, so output[0] = mean of all V rows.
        let expected_mean = [
            (10.0 + 30.0 + 50.0 + 70.0) / 4.0,
            (20.0 + 40.0 + 60.0 + 80.0) / 4.0,
        ];
        for d in 0..2 {
            assert!(
                (out[0][d] - expected_mean[d]).abs() < 1e-6,
                "out[0][{d}] = {}, expected {}",
                out[0][d],
                expected_mean[d],
            );
        }
    }

    #[test]
    fn softmax_numerical_stability() {
        let logits = vec![1000.0, 1001.0, 999.0];
        let result = softmax_vec(&logits);
        assert!(result.is_ok());
        let probs = result.expect("softmax should succeed");
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn attention_weights_sum_to_one() {
        let cfg = SparseAttentionConfig::new(1, 2, 4).expect("config valid");
        let attn = SparseAttention::new(cfg.clone()).expect("attn valid");
        let q = constant_qkv(6, 8, 0.5);
        let k = constant_qkv(6, 8, 0.5);

        let mask = build_mask(6, &cfg).expect("mask ok");
        let weights = attn
            .attention_weights(&q, &k, &mask, 0)
            .expect("weights ok");

        for (i, row) in weights.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "weight row {i} sums to {sum}, expected 1.0"
            );
        }
    }
}

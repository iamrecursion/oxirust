//! Numerically stable attention operations for TensorLogic.
//!
//! Implements scaled dot-product attention using the max-subtraction trick
//! for numerically stable softmax, with optional chunked computation to
//! control peak memory usage.

use scirs2_core::ndarray::{Array2, Array3, Array4, ArrayD, Axis};
use thiserror::Error;

/// Configuration for multi-head attention computation.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Chunk size for chunked attention (None = full sequence).
    pub chunk_size: Option<usize>,
    /// Scale factor (default: 1/sqrt(head_dim)).
    pub scale: Option<f64>,
    /// Whether to apply causal (autoregressive) masking.
    pub causal: bool,
}

impl AttentionConfig {
    /// Create a new AttentionConfig with the given number of heads and head dimension.
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        AttentionConfig {
            n_heads,
            head_dim,
            chunk_size: None,
            scale: None,
            causal: false,
        }
    }

    /// Set the chunk size for chunked attention.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = Some(chunk_size);
        self
    }

    /// Override the default scale factor.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }

    /// Set causal (autoregressive) masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Returns the effective scale: custom value or 1/sqrt(head_dim).
    pub fn effective_scale(&self) -> f64 {
        self.scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt())
    }
}

/// Errors from attention computation.
#[derive(Debug, Error)]
pub enum AttentionError {
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Result of attention computation.
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Output tensor [batch, seq_len, n_heads * head_dim].
    pub output: ArrayD<f64>,
    /// Attention weights [batch, n_heads, seq_len, seq_len] (optional, for debugging).
    pub attention_weights: Option<Array4<f64>>,
}

/// Numerically stable softmax using the max-subtraction trick.
///
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
///
/// Each row is softmaxed independently.
pub fn stable_softmax(x: &Array2<f64>) -> Array2<f64> {
    // For each row: subtract row max, exp, normalize
    let max_vals = x.map_axis(Axis(1), |row| {
        row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });
    let mut result = x.clone();
    for (i, mut row) in result.rows_mut().into_iter().enumerate() {
        let m = max_vals[i];
        row.mapv_inplace(|v| (v - m).exp());
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            row.mapv_inplace(|v| v / s);
        }
    }
    result
}

/// Scaled dot-product attention for a single head.
///
/// Computes softmax(Q K^T / scale) V.
/// Uses max-subtraction trick for numerical stability.
///
/// Inputs:
/// - q: [seq_q, head_dim]
/// - k: [seq_k, head_dim]
/// - v: [seq_k, head_dim]
///
/// Returns `(output [seq_q, d_v], weights [seq_q, seq_k])`.
pub fn scaled_dot_product_attention(
    q: &Array2<f64>,
    k: &Array2<f64>,
    v: &Array2<f64>,
    scale: f64,
    causal: bool,
) -> Result<(Array2<f64>, Array2<f64>), AttentionError> {
    // Validate shapes
    let (seq_q, d_q) = (q.nrows(), q.ncols());
    let (seq_k, d_k) = (k.nrows(), k.ncols());
    let (seq_v, d_v) = (v.nrows(), v.ncols());
    if d_q != d_k {
        return Err(AttentionError::ShapeMismatch(format!(
            "Q head_dim {} != K head_dim {}",
            d_q, d_k
        )));
    }
    if seq_k != seq_v {
        return Err(AttentionError::ShapeMismatch(format!(
            "K seq {} != V seq {}",
            seq_k, seq_v
        )));
    }

    // Compute attention scores: Q K^T / scale  [seq_q, seq_k]
    let mut scores = Array2::<f64>::zeros((seq_q, seq_k));
    for i in 0..seq_q {
        for j in 0..seq_k {
            let dot: f64 = (0..d_q).map(|d| q[[i, d]] * k[[j, d]]).sum();
            scores[[i, j]] = dot / scale;
        }
    }

    // Apply causal mask: set upper triangle to -inf
    if causal {
        for i in 0..seq_q {
            for j in (i + 1)..seq_k {
                scores[[i, j]] = f64::NEG_INFINITY;
            }
        }
    }

    // Stable softmax
    let weights = stable_softmax(&scores);

    // Output: weights @ V  [seq_q, d_v]
    let mut out = Array2::<f64>::zeros((seq_q, d_v));
    for i in 0..seq_q {
        for d in 0..d_v {
            out[[i, d]] = (0..seq_k).map(|j| weights[[i, j]] * v[[j, d]]).sum();
        }
    }

    Ok((out, weights))
}

/// Chunked scaled dot-product attention.
///
/// Processes query sequence in chunks to reduce peak memory from O(seq^2) to O(seq * chunk).
/// Does not apply causal masking (uses full KV context for each query chunk).
///
/// Inputs:
/// - q: [seq_q, head_dim]
/// - k: [seq_k, head_dim]
/// - v: [seq_k, head_dim]
pub fn chunked_attention(
    q: &Array2<f64>,
    k: &Array2<f64>,
    v: &Array2<f64>,
    scale: f64,
    chunk_size: usize,
) -> Result<Array2<f64>, AttentionError> {
    if chunk_size == 0 {
        return Err(AttentionError::InvalidConfig(
            "chunk_size must be > 0".to_string(),
        ));
    }

    let seq_q = q.nrows();
    let d_v = v.ncols();
    let mut out = Array2::<f64>::zeros((seq_q, d_v));

    let mut start = 0;
    while start < seq_q {
        let end = (start + chunk_size).min(seq_q);
        let q_chunk = q.slice(scirs2_core::ndarray::s![start..end, ..]).to_owned();
        // No causal mask in chunked (full kv context)
        let (chunk_out, _) = scaled_dot_product_attention(&q_chunk, k, v, scale, false)?;
        out.slice_mut(scirs2_core::ndarray::s![start..end, ..])
            .assign(&chunk_out);
        start = end;
    }

    Ok(out)
}

/// Multi-head attention computation.
///
/// Inputs:
/// - query: [batch, seq_q, n_heads * head_dim]
/// - key:   [batch, seq_k, n_heads * head_dim]
/// - value: [batch, seq_k, n_heads * head_dim]
pub struct MultiHeadAttention {
    config: AttentionConfig,
}

impl MultiHeadAttention {
    /// Create a new MultiHeadAttention with the given configuration.
    pub fn new(config: AttentionConfig) -> Self {
        MultiHeadAttention { config }
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Compute multi-head attention (output only, no weight matrices returned).
    pub fn forward(
        &self,
        query: &Array3<f64>,
        key: &Array3<f64>,
        value: &Array3<f64>,
    ) -> Result<AttentionOutput, AttentionError> {
        let (batch, seq_q, model_dim) = (query.shape()[0], query.shape()[1], query.shape()[2]);
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim;
        let expected_dim = n_heads * head_dim;

        if model_dim != expected_dim {
            return Err(AttentionError::ShapeMismatch(format!(
                "model_dim {} != n_heads*head_dim {}",
                model_dim, expected_dim
            )));
        }

        // Validate key/value shapes
        if key.shape()[0] != batch || value.shape()[0] != batch {
            return Err(AttentionError::ShapeMismatch(format!(
                "batch size mismatch: query={}, key={}, value={}",
                batch,
                key.shape()[0],
                value.shape()[0]
            )));
        }
        if key.shape()[2] != expected_dim || value.shape()[2] != expected_dim {
            return Err(AttentionError::ShapeMismatch(format!(
                "key/value model_dim {} != expected {}",
                key.shape()[2],
                expected_dim
            )));
        }

        let scale = self.config.effective_scale();
        let mut out = Array3::<f64>::zeros((batch, seq_q, model_dim));

        for b in 0..batch {
            for h in 0..n_heads {
                let h_start = h * head_dim;
                let h_end = h_start + head_dim;

                // Extract head slices [seq, head_dim]
                let q_h = query
                    .slice(scirs2_core::ndarray::s![b, .., h_start..h_end])
                    .to_owned();
                let k_h = key
                    .slice(scirs2_core::ndarray::s![b, .., h_start..h_end])
                    .to_owned();
                let v_h = value
                    .slice(scirs2_core::ndarray::s![b, .., h_start..h_end])
                    .to_owned();

                let head_out = if let Some(cs) = self.config.chunk_size {
                    chunked_attention(&q_h, &k_h, &v_h, scale, cs)?
                } else {
                    let (o, _) =
                        scaled_dot_product_attention(&q_h, &k_h, &v_h, scale, self.config.causal)?;
                    o
                };

                out.slice_mut(scirs2_core::ndarray::s![b, .., h_start..h_end])
                    .assign(&head_out);
            }
        }

        Ok(AttentionOutput {
            output: out.into_dyn(),
            attention_weights: None,
        })
    }

    /// Compute attention and also return attention weight matrices
    /// (for visualization/debugging).
    ///
    /// Note: this always uses full (non-chunked) attention so that weight
    /// matrices can be captured per head.
    pub fn forward_with_weights(
        &self,
        query: &Array3<f64>,
        key: &Array3<f64>,
        value: &Array3<f64>,
    ) -> Result<AttentionOutput, AttentionError> {
        let (batch, seq_q, model_dim) = (query.shape()[0], query.shape()[1], query.shape()[2]);
        let seq_k = key.shape()[1];
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim;
        let expected_dim = n_heads * head_dim;

        if model_dim != expected_dim {
            return Err(AttentionError::ShapeMismatch(format!(
                "model_dim {} != n_heads*head_dim {}",
                model_dim, expected_dim
            )));
        }
        if key.shape()[0] != batch || value.shape()[0] != batch {
            return Err(AttentionError::ShapeMismatch(format!(
                "batch size mismatch: query={}, key={}, value={}",
                batch,
                key.shape()[0],
                value.shape()[0]
            )));
        }
        if key.shape()[2] != expected_dim || value.shape()[2] != expected_dim {
            return Err(AttentionError::ShapeMismatch(format!(
                "key/value model_dim {} != expected {}",
                key.shape()[2],
                expected_dim
            )));
        }

        let scale = self.config.effective_scale();
        let mut out = Array3::<f64>::zeros((batch, seq_q, model_dim));
        let mut all_weights = Array4::<f64>::zeros((batch, n_heads, seq_q, seq_k));

        for b in 0..batch {
            for h in 0..n_heads {
                let h_start = h * head_dim;
                let h_end = h_start + head_dim;

                let q_h = query
                    .slice(scirs2_core::ndarray::s![b, .., h_start..h_end])
                    .to_owned();
                let k_h = key
                    .slice(scirs2_core::ndarray::s![b, .., h_start..h_end])
                    .to_owned();
                let v_h = value
                    .slice(scirs2_core::ndarray::s![b, .., h_start..h_end])
                    .to_owned();

                let (head_out, weights) =
                    scaled_dot_product_attention(&q_h, &k_h, &v_h, scale, self.config.causal)?;

                out.slice_mut(scirs2_core::ndarray::s![b, .., h_start..h_end])
                    .assign(&head_out);
                all_weights
                    .slice_mut(scirs2_core::ndarray::s![b, h, .., ..])
                    .assign(&weights);
            }
        }

        Ok(AttentionOutput {
            output: out.into_dyn(),
            attention_weights: Some(all_weights),
        })
    }
}

/// Compute the attention entropy (measure of how diffuse/concentrated attention is).
///
/// Higher entropy = more uniform attention, lower entropy = more focused.
///
/// Returns one entropy value per query position (row).
pub fn attention_entropy(weights: &Array2<f64>) -> Vec<f64> {
    weights
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .filter(|&&w| w > 0.0)
                .map(|&w| -w * w.ln())
                .sum::<f64>()
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, Array3};

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a deterministic Array2 filled with f(i, j).
    fn make_array2(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Array2<f64> {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Array2::from_shape_vec((rows, cols), data).expect("shape ok")
    }

    /// Build a deterministic Array3 filled with f(b, i, j).
    fn make_array3(
        batch: usize,
        seq: usize,
        dim: usize,
        f: impl Fn(usize, usize, usize) -> f64,
    ) -> Array3<f64> {
        let mut data = Vec::with_capacity(batch * seq * dim);
        for b in 0..batch {
            for i in 0..seq {
                for j in 0..dim {
                    data.push(f(b, i, j));
                }
            }
        }
        Array3::from_shape_vec((batch, seq, dim), data).expect("shape ok")
    }

    // ── stable_softmax ───────────────────────────────────────────────────────

    #[test]
    fn test_stable_softmax_basic() {
        let x = make_array2(1, 3, |_, j| j as f64); // [[0, 1, 2]]
        let out = stable_softmax(&x);
        let row = out.row(0);
        // Expected (hand-computed): [e^0, e^1, e^2] / sum
        let e0 = 1.0_f64;
        let e1 = 1.0_f64.exp();
        let e2 = 2.0_f64.exp();
        let s = e0 + e1 + e2;
        let eps = 1e-6;
        assert!((row[0] - e0 / s).abs() < eps, "p0 ~ 0.09");
        assert!((row[1] - e1 / s).abs() < eps, "p1 ~ 0.245");
        assert!((row[2] - e2 / s).abs() < eps, "p2 ~ 0.665");
    }

    #[test]
    fn test_stable_softmax_uniform() {
        let x = make_array2(1, 4, |_, _| 3.7); // same value → uniform
        let out = stable_softmax(&x);
        let row = out.row(0);
        for &v in row.iter() {
            assert!((v - 0.25).abs() < 1e-10, "uniform expected 0.25, got {}", v);
        }
    }

    #[test]
    fn test_stable_softmax_large_values() {
        // Without max-subtraction, exp(1000) overflows to Inf.
        let x = make_array2(1, 2, |_, j| if j == 0 { 1000.0 } else { 1001.0 });
        let out = stable_softmax(&x);
        let row = out.row(0);
        // Should be finite and sum to 1
        for &v in row.iter() {
            assert!(v.is_finite(), "value must be finite, got {}", v);
        }
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stable_softmax_rows_sum_to_one() {
        let x = make_array2(5, 6, |i, j| ((i * 6 + j) as f64) * 0.3 - 1.0);
        let out = stable_softmax(&x);
        for row in out.rows() {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "row sum = {}", s);
        }
    }

    // ── scaled_dot_product_attention ─────────────────────────────────────────

    #[test]
    fn test_sdp_attention_basic() {
        let seq = 4;
        let dim = 8;
        let q = make_array2(seq, dim, |i, j| (i + j) as f64 * 0.1);
        let k = make_array2(seq, dim, |i, j| (i * j) as f64 * 0.05);
        let v = make_array2(seq, dim, |i, j| (i + j * 2) as f64 * 0.1);
        let scale = 1.0 / (dim as f64).sqrt();
        let (out, _weights) = scaled_dot_product_attention(&q, &k, &v, scale, false).expect("ok");
        assert_eq!(out.shape(), &[seq, dim]);
    }

    #[test]
    fn test_sdp_attention_shape_mismatch() {
        let q = make_array2(4, 8, |i, j| (i + j) as f64);
        let k = make_array2(4, 6, |i, j| (i + j) as f64); // wrong dim
        let v = make_array2(4, 8, |i, j| (i + j) as f64);
        let result = scaled_dot_product_attention(&q, &k, &v, 1.0, false);
        assert!(result.is_err());
        if let Err(AttentionError::ShapeMismatch(msg)) = result {
            assert!(msg.contains("head_dim"), "msg = {}", msg);
        } else {
            panic!("expected ShapeMismatch");
        }
    }

    #[test]
    fn test_sdp_attention_causal_mask() {
        // With causal masking, the upper triangle of the weight matrix must be 0.
        let seq = 5;
        let dim = 4;
        let q = make_array2(seq, dim, |i, j| (i + j) as f64 * 0.1 + 0.01);
        let k = make_array2(seq, dim, |i, j| (i + j) as f64 * 0.1 + 0.01);
        let v = make_array2(seq, dim, |i, j| (i + j) as f64 * 0.1);
        let scale = 1.0 / (dim as f64).sqrt();
        let (_out, weights) = scaled_dot_product_attention(&q, &k, &v, scale, true).expect("ok");
        for i in 0..seq {
            for j in (i + 1)..seq {
                assert!(
                    weights[[i, j]] < 1e-14,
                    "upper triangle weight[{},{}] = {} must be ~0",
                    i,
                    j,
                    weights[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_sdp_attention_weights_sum_to_one() {
        let seq = 6;
        let dim = 4;
        let q = make_array2(seq, dim, |i, j| (i + j) as f64 * 0.2);
        let k = make_array2(seq, dim, |i, j| (i * 2 + j) as f64 * 0.15);
        let v = make_array2(seq, dim, |i, j| i as f64 + j as f64 * 0.5);
        let scale = 1.0 / (dim as f64).sqrt();
        let (_out, weights) = scaled_dot_product_attention(&q, &k, &v, scale, false).expect("ok");
        for row in weights.rows() {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "weight row sum = {}", s);
        }
    }

    // ── chunked_attention ────────────────────────────────────────────────────

    #[test]
    fn test_chunked_attention_matches_full() {
        let seq = 8;
        let dim = 4;
        let q = make_array2(seq, dim, |i, j| (i + j) as f64 * 0.1 + 0.05);
        let k = make_array2(seq, dim, |i, j| (i * 2 + j) as f64 * 0.07);
        let v = make_array2(seq, dim, |i, j| (j + 1) as f64 + i as f64 * 0.1);
        let scale = 1.0 / (dim as f64).sqrt();

        let (full_out, _) =
            scaled_dot_product_attention(&q, &k, &v, scale, false).expect("full ok");
        let chunked_out = chunked_attention(&q, &k, &v, scale, 2).expect("chunked ok");

        for i in 0..seq {
            for d in 0..dim {
                let diff = (full_out[[i, d]] - chunked_out[[i, d]]).abs();
                assert!(
                    diff < 1e-10,
                    "mismatch at [{},{}]: full={}, chunked={}",
                    i,
                    d,
                    full_out[[i, d]],
                    chunked_out[[i, d]]
                );
            }
        }
    }

    #[test]
    fn test_chunked_attention_single_chunk() {
        // chunk_size >= seq_len should give same result as full attention
        let seq = 5;
        let dim = 4;
        let q = make_array2(seq, dim, |i, j| (i + j) as f64 * 0.1 + 0.1);
        let k = make_array2(seq, dim, |i, j| (i + j) as f64 * 0.1 + 0.1);
        let v = make_array2(seq, dim, |i, j| (i + j) as f64 * 0.2);
        let scale = 1.0 / (dim as f64).sqrt();

        let (full_out, _) =
            scaled_dot_product_attention(&q, &k, &v, scale, false).expect("full ok");
        let chunked_out = chunked_attention(&q, &k, &v, scale, 100).expect("chunked ok");

        for i in 0..seq {
            for d in 0..dim {
                let diff = (full_out[[i, d]] - chunked_out[[i, d]]).abs();
                assert!(diff < 1e-10, "single-chunk mismatch at [{},{}]", i, d);
            }
        }
    }

    // ── MultiHeadAttention ───────────────────────────────────────────────────

    #[test]
    fn test_multihead_basic() {
        let batch = 1;
        let seq = 4;
        let n_heads = 2;
        let head_dim = 4;
        let model_dim = n_heads * head_dim;

        let query = make_array3(batch, seq, model_dim, |b, i, j| (b + i + j) as f64 * 0.1);
        let key = make_array3(batch, seq, model_dim, |b, i, j| {
            (b + i * 2 + j) as f64 * 0.1
        });
        let value = make_array3(batch, seq, model_dim, |b, _i, j| (b + j + 1) as f64 * 0.2);

        let cfg = AttentionConfig::new(n_heads, head_dim);
        let mha = MultiHeadAttention::new(cfg);
        let out = mha.forward(&query, &key, &value).expect("forward ok");
        assert_eq!(out.output.shape(), &[batch, seq, model_dim]);
        assert!(out.attention_weights.is_none());
    }

    #[test]
    fn test_multihead_wrong_dim() {
        let batch = 1;
        let seq = 4;
        // model_dim=7 but n_heads=2, head_dim=4 → expected 8
        let query = make_array3(batch, seq, 7, |_, _, _| 1.0);
        let key = make_array3(batch, seq, 7, |_, _, _| 1.0);
        let value = make_array3(batch, seq, 7, |_, _, _| 1.0);

        let cfg = AttentionConfig::new(2, 4);
        let mha = MultiHeadAttention::new(cfg);
        let result = mha.forward(&query, &key, &value);
        assert!(result.is_err());
        if let Err(AttentionError::ShapeMismatch(msg)) = result {
            assert!(msg.contains("model_dim"), "msg = {}", msg);
        } else {
            panic!("expected ShapeMismatch");
        }
    }

    #[test]
    fn test_multihead_batch() {
        let batch = 3;
        let seq = 5;
        let n_heads = 2;
        let head_dim = 3;
        let model_dim = n_heads * head_dim;

        // Each batch item has slightly different values to confirm independent processing
        let query = make_array3(batch, seq, model_dim, |b, i, j| {
            (b * 10 + i + j) as f64 * 0.1
        });
        let key = make_array3(batch, seq, model_dim, |b, i, j| {
            (b * 5 + i + j) as f64 * 0.1
        });
        let value = make_array3(batch, seq, model_dim, |b, i, j| {
            (b + i + j + 1) as f64 * 0.15
        });

        let cfg = AttentionConfig::new(n_heads, head_dim);
        let mha = MultiHeadAttention::new(cfg);
        let out = mha.forward(&query, &key, &value).expect("forward ok");
        assert_eq!(out.output.shape(), &[batch, seq, model_dim]);
    }

    #[test]
    fn test_multihead_with_weights() {
        let batch = 1;
        let seq = 4;
        let n_heads = 2;
        let head_dim = 4;
        let model_dim = n_heads * head_dim;

        let query = make_array3(batch, seq, model_dim, |_, i, j| (i + j) as f64 * 0.1);
        let key = make_array3(batch, seq, model_dim, |_, i, j| (i + j) as f64 * 0.1);
        let value = make_array3(batch, seq, model_dim, |_, i, j| (i + j) as f64 * 0.1);

        let cfg = AttentionConfig::new(n_heads, head_dim);
        let mha = MultiHeadAttention::new(cfg);
        let out = mha
            .forward_with_weights(&query, &key, &value)
            .expect("forward_with_weights ok");

        assert!(out.attention_weights.is_some());
        let w = out.attention_weights.as_ref().expect("weights present");
        assert_eq!(w.shape(), &[batch, n_heads, seq, seq]);
    }

    // ── AttentionConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_attention_config_scale() {
        let cfg = AttentionConfig::new(4, 16);
        let expected = 1.0 / (16.0_f64).sqrt();
        assert!(
            (cfg.effective_scale() - expected).abs() < 1e-12,
            "scale = {}, expected {}",
            cfg.effective_scale(),
            expected
        );
    }

    #[test]
    fn test_attention_config_custom_scale() {
        let cfg = AttentionConfig::new(4, 16).with_scale(0.5);
        assert!((cfg.effective_scale() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_attention_config_causal() {
        let cfg = AttentionConfig::new(2, 8).with_causal(true);
        assert!(cfg.causal);
        let cfg2 = AttentionConfig::new(2, 8);
        assert!(!cfg2.causal);
    }

    // ── attention_entropy ────────────────────────────────────────────────────

    #[test]
    fn test_attention_entropy_uniform() {
        // Uniform distribution over n positions → maximum entropy = ln(n)
        let n = 8;
        let x = make_array2(1, n, |_, _| 0.0); // softmax of zeros = uniform
        let weights = stable_softmax(&x);
        let entropy = attention_entropy(&weights);
        let max_entropy = (n as f64).ln();
        assert!(
            (entropy[0] - max_entropy).abs() < 1e-10,
            "entropy = {}, expected {}",
            entropy[0],
            max_entropy
        );
    }

    #[test]
    fn test_attention_entropy_focused() {
        // Peaked distribution: one position gets weight ≈ 1, rest ≈ 0
        // Use very large logit gap to create near-deterministic distribution
        let n = 8;
        let x = make_array2(1, n, |_, j| if j == 0 { 1000.0 } else { 0.0 });
        let weights = stable_softmax(&x);
        let entropy = attention_entropy(&weights);
        // Near-zero entropy for a peaked distribution
        assert!(
            entropy[0] < 1e-6,
            "entropy should be ~0 for peaked attention, got {}",
            entropy[0]
        );
    }

    // ── numerical sanity ─────────────────────────────────────────────────────

    #[test]
    fn test_attention_output_bounded() {
        // Verify output has no NaN or Inf, even with diverse inputs
        let batch = 2;
        let seq = 8;
        let n_heads = 4;
        let head_dim = 8;
        let model_dim = n_heads * head_dim;

        let query = make_array3(batch, seq, model_dim, |b, i, j| {
            ((b + 1) * (i + 1) * (j + 1)) as f64 * 0.05 - 1.0
        });
        let key = make_array3(batch, seq, model_dim, |b, i, j| {
            (b as f64 * 0.3 + i as f64 * 0.7 - j as f64 * 0.1) * 0.5
        });
        let value = make_array3(batch, seq, model_dim, |b, i, j| {
            ((b * 2 + i + j) as f64).sin()
        });

        let cfg = AttentionConfig::new(n_heads, head_dim);
        let mha = MultiHeadAttention::new(cfg);
        let out = mha.forward(&query, &key, &value).expect("forward ok");

        for &v in out.output.iter() {
            assert!(v.is_finite(), "output contains non-finite value: {}", v);
        }
    }
}

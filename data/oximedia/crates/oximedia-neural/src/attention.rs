//! Attention mechanisms: scaled dot-product, multi-head self-attention,
//! and cross-attention.
//!
//! All operations follow the standard Transformer convention
//! (Vaswani et al., 2017):
//!
//! ```text
//! Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
//! ```
//!
//! ## Shapes
//!
//! | Symbol | Description |
//! |--------|-------------|
//! | `B` | Batch size (optional outer dimension) |
//! | `T_q` | Query sequence length |
//! | `T_k` | Key/value sequence length |
//! | `D` | Model (embedding) dimension |
//! | `H` | Number of attention heads |
//! | `d_k` | Per-head dimension = `D / H` |
//!
//! The input sequences are `[T, D]` (single) or `[B, T, D]` (batched).

use crate::error::NeuralError;
use crate::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// Softmax over last dimension
// ──────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax over a slice in-place.
fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Scaled dot-product attention
// ──────────────────────────────────────────────────────────────────────────────

/// Computes scaled dot-product attention for a single head.
///
/// * `q` — query matrix `[T_q, d_k]`.
/// * `k` — key matrix `[T_k, d_k]`.
/// * `v` — value matrix `[T_k, d_v]`.
/// * `mask` — optional additive mask `[T_q, T_k]` (e.g. `-inf` for future
///   positions); `None` means no masking.
///
/// Returns an output tensor `[T_q, d_v]`.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor, NeuralError> {
    if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
        return Err(NeuralError::InvalidShape(
            "scaled_dot_product_attention: Q, K, V must all be 2-D [T, D]".to_string(),
        ));
    }
    let (t_q, d_k_q) = (q.shape()[0], q.shape()[1]);
    let (t_k, d_k_k) = (k.shape()[0], k.shape()[1]);
    let (t_k_v, d_v) = (v.shape()[0], v.shape()[1]);

    if d_k_q != d_k_k {
        return Err(NeuralError::ShapeMismatch(format!(
            "scaled_dot_product_attention: Q d_k={d_k_q} != K d_k={d_k_k}"
        )));
    }
    if t_k != t_k_v {
        return Err(NeuralError::ShapeMismatch(format!(
            "scaled_dot_product_attention: K T_k={t_k} != V T_k={t_k_v}"
        )));
    }

    let scale = 1.0 / (d_k_q as f32).sqrt();
    let d_k = d_k_q;

    // Compute scores = Q K^T / sqrt(d_k): [T_q, T_k]
    let mut scores = vec![0.0_f32; t_q * t_k];
    for iq in 0..t_q {
        for ik in 0..t_k {
            let mut dot = 0.0_f32;
            for dk in 0..d_k {
                dot += q.data()[iq * d_k + dk] * k.data()[ik * d_k + dk];
            }
            scores[iq * t_k + ik] = dot * scale;
        }
    }

    // Apply optional additive mask.
    if let Some(m) = mask {
        if m.ndim() != 2 || m.shape()[0] != t_q || m.shape()[1] != t_k {
            return Err(NeuralError::ShapeMismatch(format!(
                "scaled_dot_product_attention: mask shape {:?} incompatible with [{t_q},{t_k}]",
                m.shape()
            )));
        }
        for i in 0..t_q * t_k {
            scores[i] += m.data()[i];
        }
    }

    // Softmax over key dimension for each query position.
    for iq in 0..t_q {
        softmax_inplace(&mut scores[iq * t_k..(iq + 1) * t_k]);
    }

    // Output = scores @ V: [T_q, T_k] @ [T_k, d_v] → [T_q, d_v]
    let mut out = vec![0.0_f32; t_q * d_v];
    for iq in 0..t_q {
        for dv in 0..d_v {
            let mut acc = 0.0_f32;
            for ik in 0..t_k {
                acc += scores[iq * t_k + ik] * v.data()[ik * d_v + dv];
            }
            out[iq * d_v + dv] = acc;
        }
    }

    Tensor::from_data(out, vec![t_q, d_v])
}

// ──────────────────────────────────────────────────────────────────────────────
// MultiHeadAttention
// ──────────────────────────────────────────────────────────────────────────────

/// Multi-head attention layer (inference mode).
///
/// Projects queries, keys, and values into `num_heads` subspaces, applies
/// scaled dot-product attention in each, concatenates, and projects back.
///
/// All projection weight matrices follow `[out_dim, in_dim]` (row-major):
/// * `w_q`, `w_k`, `w_v` — each `[D, D]`
/// * `w_o` — `[D, D]`
///
/// where `D = embed_dim` and each head dimension is `D / num_heads`.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Query projection weight `[embed_dim, embed_dim]`.
    pub w_q: Vec<f32>,
    /// Key projection weight `[embed_dim, embed_dim]`.
    pub w_k: Vec<f32>,
    /// Value projection weight `[embed_dim, embed_dim]`.
    pub w_v: Vec<f32>,
    /// Output projection weight `[embed_dim, embed_dim]`.
    pub w_o: Vec<f32>,
    /// Query projection bias `[embed_dim]`.
    pub b_q: Vec<f32>,
    /// Key projection bias `[embed_dim]`.
    pub b_k: Vec<f32>,
    /// Value projection bias `[embed_dim]`.
    pub b_v: Vec<f32>,
    /// Output projection bias `[embed_dim]`.
    pub b_o: Vec<f32>,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head key/value dimension (`embed_dim / num_heads`).
    pub head_dim: usize,
}

impl MultiHeadAttention {
    /// Creates a zero-initialised `MultiHeadAttention`.
    ///
    /// `embed_dim` must be divisible by `num_heads`.
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self, NeuralError> {
        if embed_dim == 0 || num_heads == 0 {
            return Err(NeuralError::InvalidShape(
                "MultiHeadAttention: embed_dim and num_heads must be > 0".to_string(),
            ));
        }
        if embed_dim % num_heads != 0 {
            return Err(NeuralError::InvalidShape(format!(
                "MultiHeadAttention: embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )));
        }
        let head_dim = embed_dim / num_heads;
        let d2 = embed_dim * embed_dim;
        Ok(Self {
            w_q: vec![0.0_f32; d2],
            w_k: vec![0.0_f32; d2],
            w_v: vec![0.0_f32; d2],
            w_o: vec![0.0_f32; d2],
            b_q: vec![0.0_f32; embed_dim],
            b_k: vec![0.0_f32; embed_dim],
            b_v: vec![0.0_f32; embed_dim],
            b_o: vec![0.0_f32; embed_dim],
            embed_dim,
            num_heads,
            head_dim,
        })
    }

    /// Self-attention forward pass.
    ///
    /// * `x` — input `[T, D]` tensor (query = key = value = x).
    /// * `mask` — optional causal/padding mask `[T, T]`.
    ///
    /// Returns `[T, D]`.
    pub fn self_attention(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor, NeuralError> {
        self.forward_internal(x, x, x, mask)
    }

    /// Cross-attention forward pass.
    ///
    /// * `query` — query sequence `[T_q, D]`.
    /// * `context` — context (key/value) sequence `[T_k, D]`.
    /// * `mask` — optional mask `[T_q, T_k]`.
    ///
    /// Returns `[T_q, D]`.
    pub fn cross_attention(
        &self,
        query: &Tensor,
        context: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, NeuralError> {
        self.forward_internal(query, context, context, mask)
    }

    fn forward_internal(
        &self,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, NeuralError> {
        let d = self.embed_dim;
        let h = self.num_heads;
        let dh = self.head_dim;

        // Validate input ranks.
        if q_in.ndim() != 2 || k_in.ndim() != 2 || v_in.ndim() != 2 {
            return Err(NeuralError::InvalidShape(
                "MultiHeadAttention: inputs must be 2-D [T, D]".to_string(),
            ));
        }
        let t_q = q_in.shape()[0];
        let t_k = k_in.shape()[0];
        for (name, t, feat) in [("Q", q_in, d), ("K", k_in, d), ("V", v_in, d)] {
            if t.shape()[1] != feat {
                return Err(NeuralError::ShapeMismatch(format!(
                    "MultiHeadAttention: {name} feature dim {} != embed_dim {d}",
                    t.shape()[1]
                )));
            }
        }

        // Linear projections: [T, D] → [T, D]
        let q_proj = linear_proj(q_in.data(), &self.w_q, &self.b_q, t_q, d, d);
        let k_proj = linear_proj(k_in.data(), &self.w_k, &self.b_k, t_k, d, d);
        let v_proj = linear_proj(v_in.data(), &self.w_v, &self.b_v, t_k, d, d);

        // Multi-head attention: for each head, slice the d_h columns,
        // compute attention, concatenate.
        let scale = 1.0 / (dh as f32).sqrt();
        let mut concat_out = vec![0.0_f32; t_q * d];

        for head in 0..h {
            let col_start = head * dh;
            // Extract head slice from projected Q/K/V: [T, d_h]
            let q_h = extract_head_slice(&q_proj, t_q, d, col_start, dh);
            let k_h = extract_head_slice(&k_proj, t_k, d, col_start, dh);
            let v_h = extract_head_slice(&v_proj, t_k, d, col_start, dh);

            // Compute scores [T_q, T_k]
            let mut scores = vec![0.0_f32; t_q * t_k];
            for iq in 0..t_q {
                for ik in 0..t_k {
                    let mut dot = 0.0_f32;
                    for dd in 0..dh {
                        dot += q_h[iq * dh + dd] * k_h[ik * dh + dd];
                    }
                    scores[iq * t_k + ik] = dot * scale;
                }
            }

            // Apply mask if provided.
            if let Some(m) = mask {
                if m.shape()[0] != t_q || m.shape()[1] != t_k {
                    return Err(NeuralError::ShapeMismatch(format!(
                        "MultiHeadAttention: mask shape {:?} incompatible with [{t_q},{t_k}]",
                        m.shape()
                    )));
                }
                for i in 0..t_q * t_k {
                    scores[i] += m.data()[i];
                }
            }

            // Softmax over key dim.
            for iq in 0..t_q {
                softmax_inplace(&mut scores[iq * t_k..(iq + 1) * t_k]);
            }

            // Weighted sum of values: [T_q, T_k] @ [T_k, d_h] → [T_q, d_h]
            for iq in 0..t_q {
                for dd in 0..dh {
                    let mut acc = 0.0_f32;
                    for ik in 0..t_k {
                        acc += scores[iq * t_k + ik] * v_h[ik * dh + dd];
                    }
                    // Write into the correct head column of concat_out [T_q, D].
                    concat_out[iq * d + col_start + dd] += acc;
                }
            }
        }

        // Output projection: [T_q, D] → [T_q, D]
        let out_proj = linear_proj(&concat_out, &self.w_o, &self.b_o, t_q, d, d);
        Tensor::from_data(out_proj, vec![t_q, d])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Dense matmul `x [T, in_dim] @ W^T [out_dim, in_dim] + b [out_dim]` → `[T, out_dim]`.
fn linear_proj(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    t: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; t * out_dim];
    for row in 0..t {
        for col in 0..out_dim {
            let mut acc = b[col];
            for k in 0..in_dim {
                // w layout [out_dim, in_dim]
                acc += w[col * in_dim + k] * x[row * in_dim + k];
            }
            out[row * out_dim + col] = acc;
        }
    }
    out
}

/// Extract columns `[col_start..col_start+dh]` from a `[T, d]` slice → `[T, dh]`.
fn extract_head_slice(src: &[f32], t: usize, d: usize, col_start: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; t * dh];
    for row in 0..t {
        for c in 0..dh {
            out[row * dh + c] = src[row * d + col_start + c];
        }
    }
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// CausalMask helper
// ──────────────────────────────────────────────────────────────────────────────

/// Creates a causal (upper-triangular) attention mask of shape `[T, T]`.
///
/// Future positions receive a large negative value (`-1e9`) so that after
/// softmax they become effectively zero.  Past and current positions receive
/// `0.0` (no masking).
pub fn causal_mask(seq_len: usize) -> Result<Tensor, NeuralError> {
    if seq_len == 0 {
        return Err(NeuralError::InvalidShape(
            "causal_mask: seq_len must be > 0".to_string(),
        ));
    }
    let mut data = vec![0.0_f32; seq_len * seq_len];
    for q in 0..seq_len {
        for k in 0..seq_len {
            if k > q {
                data[q * seq_len + k] = -1e9;
            }
        }
    }
    Tensor::from_data(data, vec![seq_len, seq_len])
}

// ──────────────────────────────────────────────────────────────────────────────
// Sinusoidal Positional Encoding
// ──────────────────────────────────────────────────────────────────────────────

/// Computes the classic sinusoidal positional encoding from Vaswani et al. (2017).
///
/// For each position `pos` in `[0, seq_len)` and each dimension `i` in `[0, d_model)`:
///
/// ```text
/// PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
/// PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
/// ```
///
/// Returns a flat `Vec<f32>` of length `seq_len * d_model` in row-major order,
/// i.e. `output[pos * d_model + dim]`.
///
/// ## Errors
///
/// Returns an error if `seq_len` or `d_model` is zero, or if `d_model` is odd
/// (embeddings must have even dimensionality for paired sin/cos encoding).
pub fn sinusoidal_positional_encoding(
    seq_len: usize,
    d_model: usize,
) -> Result<Vec<f32>, NeuralError> {
    if seq_len == 0 {
        return Err(NeuralError::InvalidShape(
            "sinusoidal_positional_encoding: seq_len must be > 0".to_string(),
        ));
    }
    if d_model == 0 {
        return Err(NeuralError::InvalidShape(
            "sinusoidal_positional_encoding: d_model must be > 0".to_string(),
        ));
    }
    if d_model % 2 != 0 {
        return Err(NeuralError::InvalidShape(format!(
            "sinusoidal_positional_encoding: d_model={d_model} must be even \
             (paired sin/cos dimensions required)"
        )));
    }

    let mut out = vec![0.0_f32; seq_len * d_model];

    // Number of sin/cos pairs.
    let half = d_model / 2;

    for pos in 0..seq_len {
        for i in 0..half {
            // frequency denominator: 10000^(2i / d_model)
            let exponent = (2 * i) as f32 / d_model as f32;
            let denom = 10_000.0_f32.powf(exponent);
            let angle = pos as f32 / denom;

            out[pos * d_model + 2 * i] = angle.sin();
            out[pos * d_model + 2 * i + 1] = angle.cos();
        }
    }

    Ok(out)
}

/// Adds sinusoidal positional encoding to an existing embedding tensor.
///
/// * `embeddings` — flat slice of length `seq_len * d_model` (row-major `[T, D]`).
/// * `seq_len`, `d_model` — sequence length and model dimension.
///
/// Returns a new `Vec<f32>` with positional encoding added element-wise.
pub fn add_positional_encoding(
    embeddings: &[f32],
    seq_len: usize,
    d_model: usize,
) -> Result<Vec<f32>, NeuralError> {
    let expected = seq_len * d_model;
    if embeddings.len() != expected {
        return Err(NeuralError::ShapeMismatch(format!(
            "add_positional_encoding: embeddings length {} != seq_len*d_model {}",
            embeddings.len(),
            expected
        )));
    }

    let pe = sinusoidal_positional_encoding(seq_len, d_model)?;
    let out = embeddings
        .iter()
        .zip(pe.iter())
        .map(|(&e, &p)| e + p)
        .collect();
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tiled matrix multiplication for cache locality
// ──────────────────────────────────────────────────────────────────────────────

/// Tiled (blocked) matrix multiplication `C = A @ B^T` with configurable
/// tile size for improved cache locality.
///
/// * `a` — row-major `[M, K]`.
/// * `b` — row-major `[N, K]` (**not** transposed; the function transposes
///   internally by iterating in `b[j, :]` order).
/// * `block_size` — tile width/height for the blocking strategy.
///
/// Returns a flat `Vec<f32>` of length `M * N` in row-major order.
pub fn tiled_matmul_t(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Vec<f32> {
    let bs = if block_size == 0 { 32 } else { block_size };
    let mut c = vec![0.0_f32; m * n];

    let mut ii = 0;
    while ii < m {
        let i_end = (ii + bs).min(m);
        let mut jj = 0;
        while jj < n {
            let j_end = (jj + bs).min(n);
            let mut kk = 0;
            while kk < k {
                let k_end = (kk + bs).min(k);
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut acc = 0.0_f32;
                        for p in kk..k_end {
                            acc += a[i * k + p] * b[j * k + p];
                        }
                        c[i * n + j] += acc;
                    }
                }
                kk += bs;
            }
            jj += bs;
        }
        ii += bs;
    }
    c
}

// ──────────────────────────────────────────────────────────────────────────────
// Flash attention approximation (tiled softmax with running max)
// ──────────────────────────────────────────────────────────────────────────────

/// Flash-attention–inspired tiled computation of scaled dot-product attention.
///
/// Instead of materialising the full `[T_q, T_k]` score matrix, this
/// processes keys/values in tiles of `block_size` columns, maintaining a
/// running max and normalisation factor for numerical stability.
///
/// * `q` — queries `[T_q, d_k]`.
/// * `k` — keys `[T_k, d_k]`.
/// * `v` — values `[T_k, d_v]`.
/// * `block_size` — number of key columns processed per tile (0 → auto 32).
/// * `mask` — optional additive mask `[T_q, T_k]`.
///
/// Returns `[T_q, d_v]`.
pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    block_size: usize,
    mask: Option<&Tensor>,
) -> Result<Tensor, NeuralError> {
    if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
        return Err(NeuralError::InvalidShape(
            "flash_attention: Q, K, V must all be 2-D [T, D]".to_string(),
        ));
    }
    let (t_q, d_k) = (q.shape()[0], q.shape()[1]);
    let (t_k, d_k_k) = (k.shape()[0], k.shape()[1]);
    let (t_k_v, d_v) = (v.shape()[0], v.shape()[1]);

    if d_k != d_k_k {
        return Err(NeuralError::ShapeMismatch(format!(
            "flash_attention: Q d_k={d_k} != K d_k={d_k_k}"
        )));
    }
    if t_k != t_k_v {
        return Err(NeuralError::ShapeMismatch(format!(
            "flash_attention: K T_k={t_k} != V T_k={t_k_v}"
        )));
    }
    if let Some(m) = mask {
        if m.ndim() != 2 || m.shape()[0] != t_q || m.shape()[1] != t_k {
            return Err(NeuralError::ShapeMismatch(format!(
                "flash_attention: mask shape {:?} incompatible with [{t_q},{t_k}]",
                m.shape()
            )));
        }
    }

    let bs = if block_size == 0 { 32 } else { block_size };
    let scale = 1.0 / (d_k as f32).sqrt();

    // Output accumulators.
    let mut out = vec![0.0_f32; t_q * d_v];
    // Per-query running max and sum-of-exp.
    let mut row_max = vec![f32::NEG_INFINITY; t_q];
    let mut row_sum = vec![0.0_f32; t_q];

    let mut j_start = 0;
    while j_start < t_k {
        let j_end = (j_start + bs).min(t_k);
        let tile_len = j_end - j_start;

        for iq in 0..t_q {
            // Compute scores for this tile: [tile_len].
            let mut tile_scores = vec![0.0_f32; tile_len];
            for (jj, tj) in (j_start..j_end).enumerate() {
                let mut dot = 0.0_f32;
                for dd in 0..d_k {
                    dot += q.data()[iq * d_k + dd] * k.data()[tj * d_k + dd];
                }
                tile_scores[jj] = dot * scale;
                if let Some(m) = mask {
                    tile_scores[jj] += m.data()[iq * t_k + tj];
                }
            }

            // Find tile max.
            let tile_max = tile_scores
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            // Update running max.
            let prev_max = row_max[iq];
            let new_max = prev_max.max(tile_max);

            // Rescale previous accumulator.
            let correction = (prev_max - new_max).exp();
            let prev_sum = row_sum[iq] * correction;
            for dv in 0..d_v {
                out[iq * d_v + dv] *= correction;
            }

            // Accumulate this tile.
            let mut tile_sum = 0.0_f32;
            for (jj, tj) in (j_start..j_end).enumerate() {
                let w = (tile_scores[jj] - new_max).exp();
                tile_sum += w;
                for dv in 0..d_v {
                    out[iq * d_v + dv] += w * v.data()[tj * d_v + dv];
                }
            }

            row_max[iq] = new_max;
            row_sum[iq] = prev_sum + tile_sum;
        }

        j_start += bs;
    }

    // Final normalisation: divide each row by its sum.
    for iq in 0..t_q {
        let inv_sum = if row_sum[iq] > 0.0 {
            1.0 / row_sum[iq]
        } else {
            0.0
        };
        for dv in 0..d_v {
            out[iq * d_v + dv] *= inv_sum;
        }
    }

    Tensor::from_data(out, vec![t_q, d_v])
}

// ──────────────────────────────────────────────────────────────────────────────
// Relative position encoding (Shaw et al., 2018)
// ──────────────────────────────────────────────────────────────────────────────

/// Computes Shaw-style relative position bias for attention scores.
///
/// For each query position `q` and key position `k`, the relative distance is
/// `clip(k - q, -max_dist, max_dist)` and the corresponding bias is looked up
/// from a learnable table of shape `[2 * max_dist + 1, d_k]`.
///
/// * `seq_len` — sequence length.
/// * `d_k` — per-head key dimension.
/// * `max_dist` — maximum clipping distance.
/// * `rel_embeddings` — flat `[2*max_dist+1, d_k]` table (row-major).
///
/// Returns `[seq_len, seq_len]` additive bias matrix for attention scores.
pub fn relative_position_shaw(
    seq_len: usize,
    d_k: usize,
    max_dist: usize,
    rel_embeddings: &[f32],
) -> Result<Tensor, NeuralError> {
    let table_rows = 2 * max_dist + 1;
    let expected_len = table_rows * d_k;
    if rel_embeddings.len() != expected_len {
        return Err(NeuralError::ShapeMismatch(format!(
            "relative_position_shaw: embeddings len {} != {} (table_rows={} * d_k={})",
            rel_embeddings.len(),
            expected_len,
            table_rows,
            d_k
        )));
    }
    if seq_len == 0 || d_k == 0 {
        return Err(NeuralError::InvalidShape(
            "relative_position_shaw: seq_len and d_k must be > 0".to_string(),
        ));
    }

    let mut bias = vec![0.0_f32; seq_len * seq_len];

    for q_pos in 0..seq_len {
        for k_pos in 0..seq_len {
            let rel = k_pos as isize - q_pos as isize;
            let clipped = rel.max(-(max_dist as isize)).min(max_dist as isize);
            let idx = (clipped + max_dist as isize) as usize;

            // Dot product between query-position embedding and relative embedding.
            // This produces a scalar bias; for simplicity we sum across d_k.
            let row_start = idx * d_k;
            let mut dot = 0.0_f32;
            for dd in 0..d_k {
                dot += rel_embeddings[row_start + dd];
            }
            bias[q_pos * seq_len + k_pos] = dot;
        }
    }

    Tensor::from_data(bias, vec![seq_len, seq_len])
}

// ──────────────────────────────────────────────────────────────────────────────
// Rotary Position Embedding (RoPE)
// ──────────────────────────────────────────────────────────────────────────────

/// Applies Rotary Position Embedding (RoPE) in-place to a `[seq_len, d]`
/// tensor.
///
/// RoPE rotates pairs of dimensions `(2i, 2i+1)` by an angle proportional
/// to the position:
///
/// ```text
/// θ_i   = base^{-2i / d}
/// angle = pos * θ_i
/// x'[2i]   = x[2i]   * cos(angle) - x[2i+1] * sin(angle)
/// x'[2i+1] = x[2i]   * sin(angle) + x[2i+1] * cos(angle)
/// ```
///
/// * `x` — mutable tensor of shape `[seq_len, d]` where `d` is even.
/// * `base` — frequency base (typically 10000.0).
///
/// Modifies `x` in-place.
pub fn apply_rope(x: &mut Tensor, base: f32) -> Result<(), NeuralError> {
    if x.ndim() != 2 {
        return Err(NeuralError::InvalidShape(format!(
            "apply_rope: expected 2-D [T, D], got rank {}",
            x.ndim()
        )));
    }
    let seq_len = x.shape()[0];
    let d = x.shape()[1];
    if d % 2 != 0 {
        return Err(NeuralError::InvalidShape(format!(
            "apply_rope: dimension {} must be even",
            d
        )));
    }
    let half = d / 2;

    let data = x.data_mut();
    for pos in 0..seq_len {
        for i in 0..half {
            let exponent = (2 * i) as f32 / d as f32;
            let theta = base.powf(-exponent);
            let angle = pos as f32 * theta;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let idx0 = pos * d + 2 * i;
            let idx1 = idx0 + 1;
            let x0 = data[idx0];
            let x1 = data[idx1];
            data[idx0] = x0 * cos_a - x1 * sin_a;
            data[idx1] = x0 * sin_a + x1 * cos_a;
        }
    }

    Ok(())
}

/// Generates the RoPE frequency table for given `seq_len` and `d` (even).
///
/// Returns a tensor of shape `[seq_len, d]` where each element is the
/// rotated version of a unit input. Useful for pre-computing embeddings.
pub fn rope_frequencies(seq_len: usize, d: usize, base: f32) -> Result<Tensor, NeuralError> {
    if seq_len == 0 || d == 0 {
        return Err(NeuralError::InvalidShape(
            "rope_frequencies: seq_len and d must be > 0".to_string(),
        ));
    }
    if d % 2 != 0 {
        return Err(NeuralError::InvalidShape(format!(
            "rope_frequencies: d={} must be even",
            d
        )));
    }

    let half = d / 2;
    let mut data = vec![0.0_f32; seq_len * d];

    for pos in 0..seq_len {
        for i in 0..half {
            let exponent = (2 * i) as f32 / d as f32;
            let theta = base.powf(-exponent);
            let angle = pos as f32 * theta;
            data[pos * d + 2 * i] = angle.cos();
            data[pos * d + 2 * i + 1] = angle.sin();
        }
    }

    Tensor::from_data(data, vec![seq_len, d])
}

// ──────────────────────────────────────────────────────────────────────────────
// Causal attention mask (enhanced)
// ──────────────────────────────────────────────────────────────────────────────

/// Creates a rectangular causal mask for cross-attention where the query
/// sequence has length `t_q` and the key sequence has length `t_k`.
///
/// Position `(q, k)` is masked (set to `-1e9`) when `k > q`.
/// This is useful for autoregressive decoding where the decoder attends to
/// the encoder with causal constraints.
pub fn causal_mask_rect(t_q: usize, t_k: usize) -> Result<Tensor, NeuralError> {
    if t_q == 0 || t_k == 0 {
        return Err(NeuralError::InvalidShape(
            "causal_mask_rect: t_q and t_k must be > 0".to_string(),
        ));
    }
    let mut data = vec![0.0_f32; t_q * t_k];
    for q in 0..t_q {
        for k in 0..t_k {
            if k > q {
                data[q * t_k + k] = -1e9;
            }
        }
    }
    Tensor::from_data(data, vec![t_q, t_k])
}

/// Creates a sliding-window causal mask where each query can attend to at most
/// `window_size` past keys (inclusive of itself).
///
/// Position `(q, k)` is masked when `k > q` (future) or `k < q - window_size + 1`
/// (too far in the past).
pub fn sliding_window_mask(seq_len: usize, window_size: usize) -> Result<Tensor, NeuralError> {
    if seq_len == 0 {
        return Err(NeuralError::InvalidShape(
            "sliding_window_mask: seq_len must be > 0".to_string(),
        ));
    }
    if window_size == 0 {
        return Err(NeuralError::InvalidShape(
            "sliding_window_mask: window_size must be > 0".to_string(),
        ));
    }
    let mut data = vec![0.0_f32; seq_len * seq_len];
    for q in 0..seq_len {
        for k in 0..seq_len {
            let too_far_past = if window_size <= q {
                k < q - window_size + 1
            } else {
                false
            };
            if k > q || too_far_past {
                data[q * seq_len + k] = -1e9;
            }
        }
    }
    Tensor::from_data(data, vec![seq_len, seq_len])
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // ── scaled_dot_product_attention ─────────────────────────────────────────

    #[test]
    fn test_sdp_attention_shape() {
        let q = Tensor::ones(vec![3, 4]).expect("tensor ones");
        let k = Tensor::ones(vec![5, 4]).expect("tensor ones");
        let v = Tensor::ones(vec![5, 6]).expect("tensor ones");
        let out =
            scaled_dot_product_attention(&q, &k, &v, None).expect("scaled dot product attention");
        assert_eq!(out.shape(), &[3, 6]);
    }

    #[test]
    fn test_sdp_attention_uniform_scores() {
        // Uniform scores (all Q,K identical) → each output row = mean of V rows.
        let q = Tensor::ones(vec![2, 2]).expect("tensor ones");
        let k = Tensor::ones(vec![3, 2]).expect("tensor ones");
        // V rows: [1,0], [0,1], [1,1] → mean = [2/3, 2/3]
        let v = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2])
            .expect("tensor from_data");
        let out =
            scaled_dot_product_attention(&q, &k, &v, None).expect("scaled dot product attention");
        assert_eq!(out.shape(), &[2, 2]);
        // All rows should be approximately [2/3, 2/3]
        for &val in out.data() {
            assert!((val - 2.0 / 3.0).abs() < 0.01, "expected ~0.667, got {val}");
        }
    }

    #[test]
    fn test_sdp_attention_with_causal_mask() {
        // With a strict causal mask, token 0 can only attend to itself.
        let t = 3;
        let d = 2;
        let q = Tensor::ones(vec![t, d]).expect("tensor ones");
        let k = Tensor::ones(vec![t, d]).expect("tensor ones");
        let v_data: Vec<f32> = (0..t * d).map(|i| i as f32).collect();
        let v = Tensor::from_data(v_data, vec![t, d]).expect("tensor from_data");
        let mask = causal_mask(t).expect("causal mask");
        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask))
            .expect("scaled dot product attention");
        // Row 0 (position 0) can only attend to position 0 → output[0] = v[0]
        assert!(close(out.data()[0], 0.0));
        assert!(close(out.data()[1], 1.0));
    }

    #[test]
    fn test_sdp_attention_dk_mismatch_error() {
        let q = Tensor::ones(vec![3, 4]).expect("tensor ones");
        let k = Tensor::ones(vec![5, 3]).expect("tensor ones"); // d_k mismatch
        let v = Tensor::ones(vec![5, 4]).expect("tensor ones");
        assert!(scaled_dot_product_attention(&q, &k, &v, None).is_err());
    }

    #[test]
    fn test_sdp_attention_tk_mismatch_error() {
        let q = Tensor::ones(vec![3, 4]).expect("tensor ones");
        let k = Tensor::ones(vec![5, 4]).expect("tensor ones");
        let v = Tensor::ones(vec![4, 4]).expect("tensor ones"); // T_k mismatch
        assert!(scaled_dot_product_attention(&q, &k, &v, None).is_err());
    }

    #[test]
    fn test_sdp_attention_wrong_rank_error() {
        let q = Tensor::ones(vec![3, 4, 1]).expect("tensor ones"); // 3-D
        let k = Tensor::ones(vec![5, 4]).expect("tensor ones");
        let v = Tensor::ones(vec![5, 4]).expect("tensor ones");
        assert!(scaled_dot_product_attention(&q, &k, &v, None).is_err());
    }

    // ── causal_mask ──────────────────────────────────────────────────────────

    #[test]
    fn test_causal_mask_shape() {
        let m = causal_mask(4).expect("causal mask");
        assert_eq!(m.shape(), &[4, 4]);
    }

    #[test]
    fn test_causal_mask_values() {
        let m = causal_mask(3).expect("causal mask");
        // lower triangle (k <= q) = 0.0, upper triangle = -1e9
        assert!(close(m.data()[0 * 3 + 0], 0.0)); // (0,0)
        assert!(close(m.data()[0 * 3 + 1], -1e9)); // (0,1)
        assert!(close(m.data()[0 * 3 + 2], -1e9)); // (0,2)
        assert!(close(m.data()[1 * 3 + 0], 0.0)); // (1,0)
        assert!(close(m.data()[1 * 3 + 1], 0.0)); // (1,1)
        assert!(close(m.data()[1 * 3 + 2], -1e9)); // (1,2)
        assert!(close(m.data()[2 * 3 + 2], 0.0)); // (2,2)
    }

    #[test]
    fn test_causal_mask_zero_error() {
        assert!(causal_mask(0).is_err());
    }

    // ── MultiHeadAttention ────────────────────────────────────────────────────

    #[test]
    fn test_mha_output_shape_self_attn() {
        let mha = MultiHeadAttention::new(8, 2).expect("multi head attention new");
        let x = Tensor::ones(vec![5, 8]).expect("tensor ones");
        let out = mha.self_attention(&x, None).expect("self_attention");
        assert_eq!(out.shape(), &[5, 8]);
    }

    #[test]
    fn test_mha_output_shape_cross_attn() {
        let mha = MultiHeadAttention::new(8, 2).expect("multi head attention new");
        let query = Tensor::ones(vec![3, 8]).expect("tensor ones");
        let context = Tensor::ones(vec![7, 8]).expect("tensor ones");
        let out = mha
            .cross_attention(&query, &context, None)
            .expect("cross_attention");
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_mha_zero_weights_finite_output() {
        let mha = MultiHeadAttention::new(4, 2).expect("multi head attention new");
        let x = Tensor::ones(vec![3, 4]).expect("tensor ones");
        let out = mha.self_attention(&x, None).expect("self_attention");
        assert!(out.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mha_with_causal_mask() {
        let mha = MultiHeadAttention::new(4, 2).expect("multi head attention new");
        let x = Tensor::ones(vec![4, 4]).expect("tensor ones");
        let mask = causal_mask(4).expect("causal mask");
        let out = mha.self_attention(&x, Some(&mask)).expect("self_attention");
        assert_eq!(out.shape(), &[4, 4]);
        assert!(out.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mha_embed_not_divisible_error() {
        assert!(MultiHeadAttention::new(5, 2).is_err());
    }

    #[test]
    fn test_mha_zero_heads_error() {
        assert!(MultiHeadAttention::new(8, 0).is_err());
    }

    #[test]
    fn test_mha_wrong_input_dim_error() {
        let mha = MultiHeadAttention::new(8, 2).expect("multi head attention new");
        let x = Tensor::ones(vec![5, 4]).expect("tensor ones"); // 4 != 8
        assert!(mha.self_attention(&x, None).is_err());
    }

    #[test]
    fn test_mha_deterministic() {
        // Same input twice should give identical output.
        let mut mha = MultiHeadAttention::new(4, 2).expect("multi head attention new");
        for (i, w) in mha.w_q.iter_mut().enumerate() {
            *w = (i as f32 * 0.1) - 0.8;
        }
        let x = Tensor::from_data((0..3 * 4).map(|i| i as f32 * 0.1).collect(), vec![3, 4])
            .expect("tensor from_data");
        let out1 = mha.self_attention(&x, None).expect("self_attention");
        let out2 = mha.self_attention(&x, None).expect("self_attention");
        for (a, b) in out1.data().iter().zip(out2.data().iter()) {
            assert!(close(*a, *b));
        }
    }

    // ── sinusoidal_positional_encoding ────────────────────────────────────────

    #[test]
    fn test_sinusoidal_pe_shape() {
        let pe = sinusoidal_positional_encoding(10, 8).expect("sinusoidal positional encoding");
        assert_eq!(pe.len(), 10 * 8);
    }

    #[test]
    fn test_sinusoidal_pe_position_zero() {
        // At position 0, sin(0/denom) = 0 and cos(0/denom) = 1 for all dims.
        let pe = sinusoidal_positional_encoding(4, 4).expect("sinusoidal positional encoding");
        // pos=0: dims 0,1,2,3 → sin(0)=0, cos(0)=1, sin(0)=0, cos(0)=1
        assert!(close(pe[0], 0.0)); // sin
        assert!(close(pe[1], 1.0)); // cos
        assert!(close(pe[2], 0.0)); // sin
        assert!(close(pe[3], 1.0)); // cos
    }

    #[test]
    fn test_sinusoidal_pe_sin_cos_interleaved() {
        // Verify the interleaving pattern: even dims are sin, odd dims are cos.
        let pe = sinusoidal_positional_encoding(3, 8).expect("sinusoidal positional encoding");
        let d_model = 8usize;
        let half = d_model / 2;
        for pos in 0..3usize {
            for i in 0..half {
                let exponent = (2 * i) as f32 / d_model as f32;
                let denom = 10_000.0_f32.powf(exponent);
                let angle = pos as f32 / denom;
                let expected_sin = angle.sin();
                let expected_cos = angle.cos();
                assert!(
                    close(pe[pos * d_model + 2 * i], expected_sin),
                    "pos={pos} i={i} sin mismatch"
                );
                assert!(
                    close(pe[pos * d_model + 2 * i + 1], expected_cos),
                    "pos={pos} i={i} cos mismatch"
                );
            }
        }
    }

    #[test]
    fn test_sinusoidal_pe_all_finite() {
        let pe = sinusoidal_positional_encoding(64, 512).expect("sinusoidal positional encoding");
        assert!(pe.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sinusoidal_pe_zero_seq_len_error() {
        assert!(sinusoidal_positional_encoding(0, 8).is_err());
    }

    #[test]
    fn test_sinusoidal_pe_zero_d_model_error() {
        assert!(sinusoidal_positional_encoding(4, 0).is_err());
    }

    #[test]
    fn test_sinusoidal_pe_odd_d_model_error() {
        assert!(sinusoidal_positional_encoding(4, 5).is_err());
    }

    #[test]
    fn test_add_positional_encoding_roundtrip() {
        let seq_len = 3;
        let d_model = 4;
        let embeddings = vec![1.0_f32; seq_len * d_model];
        let out = add_positional_encoding(&embeddings, seq_len, d_model)
            .expect("add positional encoding");
        let pe = sinusoidal_positional_encoding(seq_len, d_model)
            .expect("sinusoidal positional encoding");
        for (i, (&o, &p)) in out.iter().zip(pe.iter()).enumerate() {
            assert!(
                close(o, 1.0 + p),
                "index {i}: expected {}, got {o}",
                1.0 + p
            );
        }
    }

    #[test]
    fn test_add_positional_encoding_length_mismatch_error() {
        let embeddings = vec![1.0_f32; 10]; // wrong length
        assert!(add_positional_encoding(&embeddings, 3, 4).is_err());
    }

    // ── tiled_matmul_t ──────────────────────────────────────────────────────

    #[test]
    fn test_tiled_matmul_t_basic() {
        // A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity)
        // A @ B^T = A @ I^T = A @ I = [[1,2],[3,4]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let c = tiled_matmul_t(&a, &b, 2, 2, 2, 2);
        assert!(close(c[0], 1.0));
        assert!(close(c[1], 2.0));
        assert!(close(c[2], 3.0));
        assert!(close(c[3], 4.0));
    }

    #[test]
    fn test_tiled_matmul_t_small_block() {
        // Use block_size=1 to exercise all tiling paths.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2x3, B^T = [[1,0],[0,1],[0,0]]
        let c = tiled_matmul_t(&a, &b, 2, 2, 3, 1);
        // Row 0: dot([1,2,3],[1,0,0])=1, dot([1,2,3],[0,1,0])=2
        assert!(close(c[0], 1.0));
        assert!(close(c[1], 2.0));
    }

    #[test]
    fn test_tiled_matmul_t_agrees_with_naive() {
        let m = 5;
        let n = 4;
        let k = 6;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.2).collect();
        let tiled = tiled_matmul_t(&a, &b, m, n, k, 2);
        // Naive A @ B^T
        let mut naive = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    naive[i * n + j] += a[i * k + p] * b[j * k + p];
                }
            }
        }
        for idx in 0..m * n {
            assert!(
                (tiled[idx] - naive[idx]).abs() < 1e-3,
                "mismatch at {idx}: tiled={}, naive={}",
                tiled[idx],
                naive[idx]
            );
        }
    }

    // ── flash_attention ─────────────────────────────────────────────────────

    #[test]
    fn test_flash_attention_matches_standard() {
        let q = Tensor::ones(vec![3, 4]).expect("ok");
        let k = Tensor::ones(vec![5, 4]).expect("ok");
        let v = Tensor::ones(vec![5, 6]).expect("ok");
        let std_out = scaled_dot_product_attention(&q, &k, &v, None).expect("ok");
        let flash_out = flash_attention(&q, &k, &v, 2, None).expect("ok");
        assert_eq!(flash_out.shape(), std_out.shape());
        for (a, b) in flash_out.data().iter().zip(std_out.data().iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "flash vs std mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_flash_attention_with_mask() {
        let t = 4;
        let d = 2;
        let q = Tensor::ones(vec![t, d]).expect("ok");
        let k = Tensor::ones(vec![t, d]).expect("ok");
        let v_data: Vec<f32> = (0..t * d).map(|i| i as f32).collect();
        let v = Tensor::from_data(v_data, vec![t, d]).expect("ok");
        let mask = causal_mask(t).expect("ok");
        let std_out = scaled_dot_product_attention(&q, &k, &v, Some(&mask)).expect("ok");
        let flash_out = flash_attention(&q, &k, &v, 2, Some(&mask)).expect("ok");
        for (a, b) in flash_out.data().iter().zip(std_out.data().iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "flash masked mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_flash_attention_large_block() {
        // block_size larger than T_k → single tile, should still work.
        let q = Tensor::ones(vec![2, 4]).expect("ok");
        let k = Tensor::ones(vec![3, 4]).expect("ok");
        let v = Tensor::ones(vec![3, 2]).expect("ok");
        let out = flash_attention(&q, &k, &v, 100, None).expect("ok");
        assert_eq!(out.shape(), &[2, 2]);
        assert!(out.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_flash_attention_shape_error() {
        let q = Tensor::ones(vec![3, 4, 1]).expect("ok");
        let k = Tensor::ones(vec![5, 4]).expect("ok");
        let v = Tensor::ones(vec![5, 4]).expect("ok");
        assert!(flash_attention(&q, &k, &v, 2, None).is_err());
    }

    // ── relative_position_shaw ──────────────────────────────────────────────

    #[test]
    fn test_shaw_basic_shape() {
        let seq_len = 4;
        let d_k = 2;
        let max_dist = 2;
        let table = vec![0.1_f32; (2 * max_dist + 1) * d_k];
        let bias = relative_position_shaw(seq_len, d_k, max_dist, &table).expect("ok");
        assert_eq!(bias.shape(), &[4, 4]);
    }

    #[test]
    fn test_shaw_symmetry_at_same_position() {
        // When rel_embeddings are uniform, all biases at distance 0 should be equal.
        let seq_len = 3;
        let d_k = 4;
        let max_dist = 3;
        let table = vec![1.0_f32; (2 * max_dist + 1) * d_k];
        let bias = relative_position_shaw(seq_len, d_k, max_dist, &table).expect("ok");
        // diagonal: same position → same relative distance (0)
        let diag_val = bias.data()[0]; // (0,0)
        for q in 1..seq_len {
            assert!(close(bias.data()[q * seq_len + q], diag_val));
        }
    }

    #[test]
    fn test_shaw_embeddings_length_error() {
        assert!(relative_position_shaw(4, 2, 2, &[0.0; 5]).is_err());
    }

    #[test]
    fn test_shaw_zero_seq_len_error() {
        assert!(relative_position_shaw(0, 2, 2, &[0.0; 10]).is_err());
    }

    // ── RoPE ────────────────────────────────────────────────────────────────

    #[test]
    fn test_rope_preserves_shape() {
        let mut x = Tensor::ones(vec![4, 8]).expect("ok");
        apply_rope(&mut x, 10000.0).expect("ok");
        assert_eq!(x.shape(), &[4, 8]);
    }

    #[test]
    fn test_rope_position_zero_unchanged() {
        // At position 0, angle=0, cos=1, sin=0 → x unchanged.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = Tensor::from_data(data.clone(), vec![2, 4]).expect("ok");
        apply_rope(&mut x, 10000.0).expect("ok");
        // Only position 0 row should be unchanged.
        for i in 0..4 {
            assert!(
                close(x.data()[i], data[i]),
                "pos0 dim{i}: {} != {}",
                x.data()[i],
                data[i]
            );
        }
    }

    #[test]
    fn test_rope_all_finite() {
        let mut x = Tensor::ones(vec![32, 64]).expect("ok");
        apply_rope(&mut x, 10000.0).expect("ok");
        assert!(x.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rope_odd_dim_error() {
        let mut x = Tensor::ones(vec![3, 5]).expect("ok");
        assert!(apply_rope(&mut x, 10000.0).is_err());
    }

    #[test]
    fn test_rope_frequencies_shape() {
        let freq = rope_frequencies(8, 16, 10000.0).expect("ok");
        assert_eq!(freq.shape(), &[8, 16]);
    }

    #[test]
    fn test_rope_frequencies_pos0() {
        let freq = rope_frequencies(4, 4, 10000.0).expect("ok");
        // pos=0: all cos=1, sin=0 → [1,0,1,0]
        assert!(close(freq.data()[0], 1.0));
        assert!(close(freq.data()[1], 0.0));
        assert!(close(freq.data()[2], 1.0));
        assert!(close(freq.data()[3], 0.0));
    }

    // ── causal_mask_rect ────────────────────────────────────────────────────

    #[test]
    fn test_causal_mask_rect_shape() {
        let m = causal_mask_rect(3, 5).expect("ok");
        assert_eq!(m.shape(), &[3, 5]);
    }

    #[test]
    fn test_causal_mask_rect_values() {
        let m = causal_mask_rect(2, 4).expect("ok");
        // Row 0: k=0 ok, k=1..3 masked
        assert!(close(m.data()[0], 0.0));
        assert!(close(m.data()[1], -1e9));
        // Row 1: k=0..1 ok, k=2..3 masked
        assert!(close(m.data()[4], 0.0));
        assert!(close(m.data()[5], 0.0));
        assert!(close(m.data()[6], -1e9));
    }

    #[test]
    fn test_causal_mask_rect_zero_error() {
        assert!(causal_mask_rect(0, 3).is_err());
        assert!(causal_mask_rect(3, 0).is_err());
    }

    // ── sliding_window_mask ─────────────────────────────────────────────────

    #[test]
    fn test_sliding_window_mask_full_window() {
        // window = seq_len → equivalent to causal mask.
        let sw = sliding_window_mask(3, 3).expect("ok");
        let cm = causal_mask(3).expect("ok");
        assert_eq!(sw.data(), cm.data());
    }

    #[test]
    fn test_sliding_window_mask_window1() {
        // window = 1 → each position can only attend to itself.
        let m = sliding_window_mask(3, 1).expect("ok");
        // Only diagonal should be 0.0
        for q in 0..3 {
            for k in 0..3 {
                if q == k {
                    assert!(close(m.data()[q * 3 + k], 0.0));
                } else {
                    assert!(close(m.data()[q * 3 + k], -1e9));
                }
            }
        }
    }

    #[test]
    fn test_sliding_window_mask_window2() {
        let m = sliding_window_mask(4, 2).expect("ok");
        // q=0: attend k=0 only
        assert!(close(m.data()[0], 0.0));
        assert!(close(m.data()[1], -1e9));
        // q=1: attend k=0,1
        assert!(close(m.data()[4], 0.0));
        assert!(close(m.data()[5], 0.0));
        assert!(close(m.data()[6], -1e9));
        // q=2: attend k=1,2
        assert!(close(m.data()[8], -1e9));
        assert!(close(m.data()[9], 0.0));
        assert!(close(m.data()[10], 0.0));
    }

    #[test]
    fn test_sliding_window_zero_error() {
        assert!(sliding_window_mask(0, 3).is_err());
        assert!(sliding_window_mask(3, 0).is_err());
    }
}

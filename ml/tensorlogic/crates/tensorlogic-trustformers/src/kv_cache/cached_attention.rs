use ndarray::{s, Array2, Array3, ArrayD, IxDyn};
use std::fmt;

use super::position::{PositionError, RotaryPositionEmbedding};
use super::simple_cache::{KvCache, KvCacheError};

/// Errors that can occur during cached attention forward passes.
#[derive(Debug, Clone)]
pub enum CachedAttentionError {
    /// Wrapped KV-cache error.
    KvCacheError(KvCacheError),
    /// Wrapped position encoding error.
    PositionError(PositionError),
    /// General shape or configuration error.
    InvalidShape(String),
}

impl fmt::Display for CachedAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KvCacheError(e) => write!(f, "KV-cache error: {}", e),
            Self::PositionError(e) => write!(f, "Position encoding error: {}", e),
            Self::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
        }
    }
}

impl std::error::Error for CachedAttentionError {}

impl From<KvCacheError> for CachedAttentionError {
    fn from(e: KvCacheError) -> Self {
        Self::KvCacheError(e)
    }
}

impl From<PositionError> for CachedAttentionError {
    fn from(e: PositionError) -> Self {
        Self::PositionError(e)
    }
}

/// Scaled dot-product multi-head attention with optional KV-cache and RoPE.
///
/// Inputs are assumed to have shape `[batch, seq_len, num_heads * head_dim]` and
/// are internally reshaped to `[batch, seq_len, num_heads, head_dim]`.
#[derive(Debug, Clone)]
pub struct CachedAttention {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each head.
    pub head_dim: usize,
    /// Attention scale factor (defaults to `1 / sqrt(head_dim)`).
    pub scale: f64,
    /// Optional Rotary Position Embedding applied to Q and K.
    pub rope: Option<RotaryPositionEmbedding>,
    /// If `true`, apply a causal mask to prevent attending to future positions.
    pub use_causal_mask: bool,
}

impl CachedAttention {
    /// Create a new `CachedAttention`.
    ///
    /// When `use_rope` is `true`, a `RotaryPositionEmbedding` is pre-built for
    /// `max_seq_len` positions using the standard base of 10000.
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        use_rope: bool,
        max_seq_len: usize,
    ) -> std::result::Result<Self, CachedAttentionError> {
        let scale = 1.0 / (head_dim as f64).sqrt();
        let rope = if use_rope {
            Some(
                RotaryPositionEmbedding::new(head_dim, max_seq_len, 10000.0)
                    .map_err(CachedAttentionError::PositionError)?,
            )
        } else {
            None
        };
        Ok(Self {
            num_heads,
            head_dim,
            scale,
            rope,
            use_causal_mask: true,
        })
    }

    /// Run the forward pass.
    ///
    /// * `query`, `key`, `value` — shape `[batch, seq_len, num_heads * head_dim]`
    /// * `cache` — optional mutable KV-cache; keys/values from previous steps are
    ///   prepended before computing attention.
    /// * `layer_idx` — index used when reading/writing the cache.
    ///
    /// Returns output of shape `[batch, seq_len, num_heads * head_dim]`.
    pub fn forward(
        &self,
        query: &ArrayD<f64>,
        key: &ArrayD<f64>,
        value: &ArrayD<f64>,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
    ) -> std::result::Result<ArrayD<f64>, CachedAttentionError> {
        let q_shape = query.shape();
        if q_shape.len() != 3 {
            return Err(CachedAttentionError::InvalidShape(format!(
                "query must be 3-D [batch, seq, d], got {} dims",
                q_shape.len()
            )));
        }
        let batch = q_shape[0];
        let seq_len = q_shape[1];
        let d = q_shape[2];

        if d != self.num_heads * self.head_dim {
            return Err(CachedAttentionError::InvalidShape(format!(
                "last dim {} != num_heads * head_dim = {}",
                d,
                self.num_heads * self.head_dim
            )));
        }

        // Reshape Q, K, V to [batch * seq, num_heads, head_dim] for easier ops.
        let q = query
            .view()
            .into_shape_with_order(IxDyn(&[batch * seq_len, self.num_heads, self.head_dim]))
            .map_err(|e| CachedAttentionError::InvalidShape(e.to_string()))?
            .to_owned();

        let mut k = key
            .view()
            .into_shape_with_order(IxDyn(&[batch * seq_len, self.num_heads, self.head_dim]))
            .map_err(|e| CachedAttentionError::InvalidShape(e.to_string()))?
            .to_owned();

        let v = value
            .view()
            .into_shape_with_order(IxDyn(&[batch * seq_len, self.num_heads, self.head_dim]))
            .map_err(|e| CachedAttentionError::InvalidShape(e.to_string()))?
            .to_owned();

        // Apply RoPE to Q and K if configured.
        let seq_offset = cache.as_ref().map(|c| c.seq_len).unwrap_or(0);

        let (q_rope, k_rope) = if let Some(rope) = &self.rope {
            let q_r = rope
                .apply(&q, seq_offset)
                .map_err(CachedAttentionError::PositionError)?;
            let k_r = rope
                .apply(&k, seq_offset)
                .map_err(CachedAttentionError::PositionError)?;
            (q_r, k_r)
        } else {
            (q, k.clone())
        };

        // Append current K, V to cache (if present), then read full K, V.
        let (full_k, full_v) = if let Some(cache_ref) = cache {
            cache_ref
                .append_kv(layer_idx, k_rope.clone(), v.clone())
                .map_err(CachedAttentionError::KvCacheError)?;
            let (ck, cv) = cache_ref.get_kv(layer_idx).ok_or({
                CachedAttentionError::KvCacheError(KvCacheError::LayerOutOfBounds {
                    layer: layer_idx,
                    num_layers: cache_ref.num_layers,
                })
            })?;
            (ck.to_owned(), cv.to_owned())
        } else {
            k = k_rope;
            (k, v)
        };

        let cache_len = full_k.shape()[0] / batch.max(1);

        // Build optional causal mask.
        let mask = if self.use_causal_mask {
            Some(Self::causal_mask(seq_len, cache_len))
        } else {
            None
        };

        // Reshape Q to [seq_len, num_heads, head_dim] (single batch for simplicity).
        // Full attention: Q [seq, heads, d], K [cache+seq, heads, d], V [cache+seq, heads, d].
        self.scaled_dot_product(&q_rope, &full_k, &full_v, mask.as_ref())
            .map(|out| {
                // Reshape output back to [batch, seq_len, num_heads * head_dim].
                out.into_shape_with_order(IxDyn(&[batch, seq_len, self.num_heads * self.head_dim]))
                    .unwrap_or_else(|_| {
                        ArrayD::zeros(IxDyn(&[batch, seq_len, self.num_heads * self.head_dim]))
                    })
            })
    }

    /// Build a lower-triangular causal mask of shape `[seq_len, cache_len + seq_len]`.
    ///
    /// Positions where attention is allowed have value `0.0`; masked positions
    /// have value `-1e9` (a large negative additive bias).
    pub fn causal_mask(seq_len: usize, cache_len: usize) -> Array2<f64> {
        let total_k = cache_len + seq_len;
        let mut mask = Array2::<f64>::zeros((seq_len, total_k));
        for q in 0..seq_len {
            // Query position relative to the full key sequence is (cache_len + q).
            // Allow attention to positions <= cache_len + q.
            for k in 0..total_k {
                if k > cache_len + q {
                    mask[[q, k]] = -1.0e9;
                }
            }
        }
        mask
    }

    /// Compute scaled dot-product attention.
    ///
    /// * `q` — shape `[total_q, num_heads, head_dim]`
    /// * `k` — shape `[total_k, num_heads, head_dim]`
    /// * `v` — shape `[total_k, num_heads, head_dim]`
    /// * `mask` — optional additive mask of shape `[total_q / num_heads, total_k / num_heads]`
    ///   or `[seq_q, seq_k]` that is broadcast across heads.
    pub fn scaled_dot_product(
        &self,
        q: &ArrayD<f64>,
        k: &ArrayD<f64>,
        v: &ArrayD<f64>,
        mask: Option<&Array2<f64>>,
    ) -> std::result::Result<ArrayD<f64>, CachedAttentionError> {
        let q_shape = q.shape();
        let k_shape = k.shape();

        if q_shape.len() != 3 || k_shape.len() != 3 {
            return Err(CachedAttentionError::InvalidShape(
                "q, k, v must be 3-D [tokens, heads, head_dim]".to_string(),
            ));
        }

        let total_q = q_shape[0];
        let total_k = k_shape[0];
        let num_heads = q_shape[1];
        let head_dim = q_shape[2];

        if head_dim == 0 || num_heads == 0 {
            return Err(CachedAttentionError::InvalidShape(
                "head_dim and num_heads must be > 0".to_string(),
            ));
        }

        // Compute attention scores: [total_q, num_heads, total_k]
        // scores[i, h, j] = sum_d q[i, h, d] * k[j, h, d] * scale
        let mut scores = Array3::<f64>::zeros((total_q, num_heads, total_k));

        let q3 = q
            .view()
            .into_shape_with_order((total_q, num_heads, head_dim))
            .map_err(|e| CachedAttentionError::InvalidShape(e.to_string()))?;

        let k3 = k
            .view()
            .into_shape_with_order((total_k, num_heads, head_dim))
            .map_err(|e| CachedAttentionError::InvalidShape(e.to_string()))?;

        for i in 0..total_q {
            for h in 0..num_heads {
                for j in 0..total_k {
                    let mut dot = 0.0_f64;
                    for d in 0..head_dim {
                        dot += q3[[i, h, d]] * k3[[j, h, d]];
                    }
                    scores[[i, h, j]] = dot * self.scale;
                }
            }
        }

        // Apply mask if provided.
        if let Some(m) = mask {
            let mask_q = m.shape()[0];
            let mask_k = m.shape()[1];
            for i in 0..total_q.min(mask_q) {
                for h in 0..num_heads {
                    for j in 0..total_k.min(mask_k) {
                        scores[[i, h, j]] += m[[i, j]];
                    }
                }
            }
        }

        // Softmax over key dimension (axis 2).
        for i in 0..total_q {
            for h in 0..num_heads {
                let row_max = scores
                    .slice(s![i, h, ..])
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0_f64;
                for j in 0..total_k {
                    scores[[i, h, j]] = (scores[[i, h, j]] - row_max).exp();
                    sum += scores[[i, h, j]];
                }
                let safe_sum = if sum == 0.0 { 1.0 } else { sum };
                for j in 0..total_k {
                    scores[[i, h, j]] /= safe_sum;
                }
            }
        }

        // Weighted sum over values: output[i, h, d] = sum_j scores[i, h, j] * v[j, h, d]
        let v_shape = v.shape();
        let v3 = v
            .view()
            .into_shape_with_order((v_shape[0], num_heads, head_dim))
            .map_err(|e| CachedAttentionError::InvalidShape(e.to_string()))?;

        let mut output = Array3::<f64>::zeros((total_q, num_heads, head_dim));

        for i in 0..total_q {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let mut val = 0.0_f64;
                    for j in 0..total_k {
                        val += scores[[i, h, j]] * v3[[j, h, d]];
                    }
                    output[[i, h, d]] = val;
                }
            }
        }

        Ok(output.into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(shape: &[usize], fill: f64) -> ArrayD<f64> {
        ArrayD::from_elem(IxDyn(shape), fill)
    }

    #[test]
    fn test_cached_attention_forward_no_cache() {
        let attn = CachedAttention::new(2, 4, false, 32).expect("valid config");
        // [batch=1, seq=3, d=8]
        let q = make_tensor(&[1, 3, 8], 0.5);
        let k = make_tensor(&[1, 3, 8], 0.5);
        let v = make_tensor(&[1, 3, 8], 0.5);
        let out = attn
            .forward(&q, &k, &v, None, 0)
            .expect("forward should succeed");
        assert_eq!(
            out.shape(),
            &[1, 3, 8],
            "output shape must be [batch, seq, d]"
        );
    }

    #[test]
    fn test_cached_attention_causal_mask_shape() {
        let mask = CachedAttention::causal_mask(4, 0);
        assert_eq!(mask.shape(), &[4, 4], "causal mask must be [seq, seq]");
        // Lower triangular: mask[0,1] should be large negative.
        assert!(mask[[0, 1]] < -1e8, "future positions should be masked");
        // mask[1,0] should be zero (allowed to attend to past).
        assert!(
            (mask[[1, 0]]).abs() < 1e-9,
            "past positions should not be masked"
        );
    }

    #[test]
    fn test_cached_attention_with_cache_extends_seq() {
        let attn = CachedAttention::new(2, 4, false, 64).expect("valid");
        let mut cache = KvCache::new(1, 2, 4, 64);
        let q = make_tensor(&[1, 2, 8], 0.1);
        let k = make_tensor(&[1, 2, 8], 0.1);
        let v = make_tensor(&[1, 2, 8], 0.1);
        attn.forward(&q, &k, &v, Some(&mut cache), 0)
            .expect("forward with cache");
        assert!(cache.seq_len > 0, "cache seq_len should grow after forward");
    }

    #[test]
    fn test_cached_attention_error_display() {
        let err = CachedAttentionError::InvalidShape("bad shape".to_string());
        let s = err.to_string();
        assert!(
            s.contains("bad shape"),
            "Display impl should include the message"
        );
    }
}

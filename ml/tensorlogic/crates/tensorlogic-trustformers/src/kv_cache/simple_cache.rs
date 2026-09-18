use ndarray::{ArrayD, Axis, IxDyn};
use std::fmt;

/// Error for ndarray-based KV-cache operations.
#[derive(Debug, Clone)]
pub enum KvCacheError {
    /// Layer index is out of bounds.
    LayerOutOfBounds { layer: usize, num_layers: usize },
    /// Cache has reached maximum sequence length.
    CacheFull { max_seq_len: usize },
    /// Tensor shape does not match expected shape.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayerOutOfBounds { layer, num_layers } => write!(
                f,
                "Layer index {} is out of bounds (num_layers = {})",
                layer, num_layers
            ),
            Self::CacheFull { max_seq_len } => {
                write!(f, "KV cache is full (max_seq_len = {})", max_seq_len)
            }
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
        }
    }
}

impl std::error::Error for KvCacheError {}

/// Cached key-value pairs for autoregressive inference using ndarray tensors.
///
/// Stores per-layer key and value tensors as dynamic-rank `ArrayD<f64>`.
/// Tensors are concatenated along the sequence dimension on each `append_kv` call.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Cached K tensors per layer (shape: `[seq_len, num_heads, head_dim]` after appends).
    pub keys: Vec<ArrayD<f64>>,
    /// Cached V tensors per layer (shape: `[seq_len, num_heads, head_dim]` after appends).
    pub values: Vec<ArrayD<f64>>,
    /// Current cached sequence length.
    pub seq_len: usize,
    /// Maximum allowed sequence length.
    pub max_seq_len: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
}

impl KvCache {
    /// Create a new, empty KV-cache.
    ///
    /// Initially all per-layer tensors are zero-sized along the sequence dimension.
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let empty = ArrayD::<f64>::zeros(IxDyn(&[0, num_heads, head_dim]));
        Self {
            keys: vec![empty.clone(); num_layers],
            values: vec![empty; num_layers],
            seq_len: 0,
            max_seq_len,
            num_layers,
            num_heads,
        }
    }

    /// Append new key and value tensors for the given layer.
    ///
    /// `new_k` and `new_v` must have shape `[new_tokens, num_heads, head_dim]`.
    pub fn append_kv(
        &mut self,
        layer: usize,
        new_k: ArrayD<f64>,
        new_v: ArrayD<f64>,
    ) -> std::result::Result<(), KvCacheError> {
        if layer >= self.num_layers {
            return Err(KvCacheError::LayerOutOfBounds {
                layer,
                num_layers: self.num_layers,
            });
        }

        if self.seq_len >= self.max_seq_len {
            return Err(KvCacheError::CacheFull {
                max_seq_len: self.max_seq_len,
            });
        }

        // Validate shapes match existing cache shape (except axis 0 which is seq).
        let expected_tail = &self.keys[layer].shape()[1..];
        let got_tail = &new_k.shape()[1..];
        if expected_tail != got_tail && !self.keys[layer].shape()[0] == 0 {
            return Err(KvCacheError::ShapeMismatch {
                expected: expected_tail.to_vec(),
                got: got_tail.to_vec(),
            });
        }

        let new_tokens = new_k.shape()[0];
        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(KvCacheError::CacheFull {
                max_seq_len: self.max_seq_len,
            });
        }

        // Concatenate along axis 0 (sequence dimension).
        let concat_k = if self.keys[layer].shape()[0] == 0 {
            new_k
        } else {
            let views_k = vec![self.keys[layer].view(), new_k.view()];
            ndarray::concatenate(Axis(0), &views_k).map_err(|e| KvCacheError::ShapeMismatch {
                expected: self.keys[layer].shape().to_vec(),
                got: vec![e.to_string().len()], // encode error into shape slot
            })?
        };

        let concat_v = if self.values[layer].shape()[0] == 0 {
            new_v
        } else {
            let views_v = vec![self.values[layer].view(), new_v.view()];
            ndarray::concatenate(Axis(0), &views_v).map_err(|e| KvCacheError::ShapeMismatch {
                expected: self.values[layer].shape().to_vec(),
                got: vec![e.to_string().len()],
            })?
        };

        // Only update seq_len for layer 0 to keep a single global counter.
        // For multi-layer caches, seq_len tracks the common sequence length.
        if layer == 0 {
            self.seq_len += new_tokens;
        } else {
            // Update seq_len from the actual key length of layer 0.
            self.seq_len = self.keys[0].shape()[0];
        }

        self.keys[layer] = concat_k;
        self.values[layer] = concat_v;

        // Recompute seq_len as the max over all layers.
        self.seq_len = self.keys.iter().map(|k| k.shape()[0]).max().unwrap_or(0);

        Ok(())
    }

    /// Retrieve cached keys and values for the given layer.
    ///
    /// Returns `None` if the layer index is out of bounds.
    pub fn get_kv(&self, layer: usize) -> Option<(&ArrayD<f64>, &ArrayD<f64>)> {
        if layer >= self.num_layers {
            return None;
        }
        Some((&self.keys[layer], &self.values[layer]))
    }

    /// Reset the cache to empty (seq_len == 0).
    pub fn reset(&mut self) {
        let head_dim = if self.num_layers > 0 && !self.keys[0].shape().is_empty() {
            *self.keys[0].shape().last().unwrap_or(&0)
        } else {
            0
        };
        let empty = ArrayD::<f64>::zeros(IxDyn(&[0, self.num_heads, head_dim]));
        for k in &mut self.keys {
            *k = empty.clone();
        }
        for v in &mut self.values {
            *v = empty.clone();
        }
        self.seq_len = 0;
    }

    /// Current cached sequence length.
    pub fn current_len(&self) -> usize {
        self.seq_len
    }

    /// Returns `true` if the cache is at maximum capacity.
    pub fn is_full(&self) -> bool {
        self.seq_len >= self.max_seq_len
    }

    /// Approximate memory usage in bytes (f64 = 8 bytes per element).
    pub fn memory_usage_bytes(&self) -> usize {
        let key_bytes: usize = self.keys.iter().map(|k| k.len() * 8).sum();
        let val_bytes: usize = self.values.iter().map(|v| v.len() * 8).sum();
        key_bytes + val_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(shape: &[usize], fill: f64) -> ArrayD<f64> {
        ArrayD::from_elem(IxDyn(shape), fill)
    }

    #[test]
    fn test_kv_cache_new_and_append() {
        let mut cache = KvCache::new(2, 4, 8, 16);
        let new_k = make_tensor(&[3, 4, 8], 1.0);
        let new_v = make_tensor(&[3, 4, 8], 2.0);
        cache
            .append_kv(0, new_k, new_v)
            .expect("append should succeed");
        assert_eq!(cache.seq_len, 3, "seq_len should increment");
    }

    #[test]
    fn test_kv_cache_full_returns_error() {
        let mut cache = KvCache::new(1, 2, 4, 3);
        // Fill up the cache completely (3 tokens).
        let k = make_tensor(&[3, 2, 4], 1.0);
        let v = make_tensor(&[3, 2, 4], 1.0);
        cache.append_kv(0, k, v).expect("initial fill");
        // Next append should fail.
        let k2 = make_tensor(&[1, 2, 4], 1.0);
        let v2 = make_tensor(&[1, 2, 4], 1.0);
        let result = cache.append_kv(0, k2, v2);
        assert!(
            matches!(result, Err(KvCacheError::CacheFull { .. })),
            "expected CacheFull error"
        );
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KvCache::new(1, 2, 4, 16);
        let k = make_tensor(&[4, 2, 4], 1.0);
        let v = make_tensor(&[4, 2, 4], 1.0);
        cache.append_kv(0, k, v).expect("append");
        assert!(cache.seq_len > 0);
        cache.reset();
        assert_eq!(cache.seq_len, 0, "seq_len must be 0 after reset");
    }

    #[test]
    fn test_kv_cache_memory_usage() {
        let mut cache = KvCache::new(1, 2, 4, 16);
        let k = make_tensor(&[2, 2, 4], 1.0);
        let v = make_tensor(&[2, 2, 4], 1.0);
        cache.append_kv(0, k, v).expect("append");
        assert!(
            cache.memory_usage_bytes() > 0,
            "memory should be non-zero after append"
        );
    }

    #[test]
    fn test_kv_cache_get_kv_valid_layer() {
        let mut cache = KvCache::new(2, 2, 4, 16);
        let k = make_tensor(&[2, 2, 4], 1.0);
        let v = make_tensor(&[2, 2, 4], 2.0);
        cache.append_kv(0, k, v).expect("append");
        let result = cache.get_kv(0);
        assert!(result.is_some(), "should return Some for valid layer");
    }

    #[test]
    fn test_kv_cache_get_kv_invalid_layer() {
        let cache = KvCache::new(2, 2, 4, 16);
        let result = cache.get_kv(99);
        assert!(
            result.is_none(),
            "should return None for out-of-range layer"
        );
    }

    #[test]
    fn test_kv_cache_layer_out_of_bounds_error() {
        let mut cache = KvCache::new(2, 2, 4, 16);
        let k = make_tensor(&[1, 2, 4], 1.0);
        let v = make_tensor(&[1, 2, 4], 1.0);
        let result = cache.append_kv(5, k, v);
        assert!(
            matches!(result, Err(KvCacheError::LayerOutOfBounds { .. })),
            "layer >= num_layers should return LayerOutOfBounds"
        );
    }
}

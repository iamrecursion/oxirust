//! Longformer-style sparse attention configuration.
//!
//! Implements the sliding-window + global-token pattern from
//! Beltagy et al. (2020), "Longformer: The Long-Document Transformer".

use serde::{Deserialize, Serialize};

use super::error::{SparseAttentionError, SparseAttentionResult};

/// Configuration for Longformer-style sparse attention.
///
/// Each token attends to its local neighbourhood (sliding window of
/// `2 * window_size + 1` positions) plus a set of designated *global*
/// tokens that attend to — and are attended by — every position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseAttentionConfig {
    /// Half-window size for local attention.
    /// Each token attends to positions within `[i - window_size, i + window_size]`.
    pub window_size: usize,

    /// Positions of global tokens that attend to (and are attended by) all
    /// positions.  Typically `[0]` (CLS) or start/end markers.
    pub global_token_indices: Vec<usize>,

    /// Number of attention heads.
    pub num_heads: usize,

    /// Dimension per head.
    pub head_dim: usize,

    /// When `true`, future positions are masked (`j > i` is never attended).
    pub causal: bool,

    /// Attention dropout probability (`0.0` = none).
    /// Stored for configuration completeness; actual dropout sampling is not
    /// applied in this research-preview implementation.
    pub dropout: f64,
}

impl SparseAttentionConfig {
    /// Create a minimal configuration with required fields.
    pub fn new(
        window_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> SparseAttentionResult<Self> {
        let cfg = Self {
            window_size,
            global_token_indices: Vec::new(),
            num_heads,
            head_dim,
            causal: false,
            dropout: 0.0,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Builder: set global token indices.
    #[must_use]
    pub fn with_global_tokens(mut self, indices: Vec<usize>) -> Self {
        self.global_token_indices = indices;
        self
    }

    /// Builder: enable or disable causal masking.
    #[must_use]
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Builder: set dropout probability.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Validate configuration invariants.
    pub fn validate(&self) -> SparseAttentionResult<()> {
        if self.window_size == 0 {
            return Err(SparseAttentionError::InvalidWindowSize(0));
        }
        if self.num_heads == 0 {
            return Err(SparseAttentionError::DimensionMismatch {
                context: "num_heads".into(),
                expected: 1,
                got: 0,
            });
        }
        if self.head_dim == 0 {
            return Err(SparseAttentionError::DimensionMismatch {
                context: "head_dim".into(),
                expected: 1,
                got: 0,
            });
        }
        Ok(())
    }

    /// Total model dimension (`num_heads * head_dim`).
    pub fn d_model(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Validate global indices against a concrete sequence length.
    pub fn validate_globals_for_seq_len(&self, seq_len: usize) -> SparseAttentionResult<()> {
        for &idx in &self.global_token_indices {
            if idx >= seq_len {
                return Err(SparseAttentionError::InvalidGlobalIndices {
                    index: idx,
                    seq_len,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_config() {
        let cfg = SparseAttentionConfig::new(4, 2, 8);
        assert!(cfg.is_ok());
        let cfg = cfg.ok();
        assert!(cfg.is_some());
        let cfg = cfg.as_ref();
        assert!(cfg.is_some());
        let cfg_ref = cfg.map(|c| c.d_model());
        assert_eq!(cfg_ref, Some(16));
    }

    #[test]
    fn zero_window_rejected() {
        let result = SparseAttentionConfig::new(0, 2, 8);
        assert!(matches!(
            result,
            Err(SparseAttentionError::InvalidWindowSize(0))
        ));
    }

    #[test]
    fn zero_heads_rejected() {
        let result = SparseAttentionConfig::new(4, 0, 8);
        assert!(result.is_err());
    }

    #[test]
    fn global_oob_detected() {
        let cfg = SparseAttentionConfig::new(4, 2, 8).map(|c| c.with_global_tokens(vec![0, 99]));
        assert!(cfg.is_ok());
        let cfg = cfg.as_ref().ok();
        assert!(cfg.is_some());
        let result = cfg.map(|c| c.validate_globals_for_seq_len(16));
        assert!(matches!(result, Some(Err(_))));
    }

    #[test]
    fn builder_chain() {
        let cfg = SparseAttentionConfig::new(4, 2, 8).map(|c| {
            c.with_global_tokens(vec![0])
                .with_causal(true)
                .with_dropout(0.1)
        });
        assert!(cfg.is_ok());
        let cfg = cfg.as_ref().ok();
        assert!(cfg.is_some());
        let cfg = cfg.map(|c| (c.causal, c.dropout, c.global_token_indices.len()));
        assert_eq!(cfg, Some((true, 0.1, 1)));
    }
}

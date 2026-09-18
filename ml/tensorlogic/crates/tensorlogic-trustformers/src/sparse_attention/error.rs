//! Error taxonomy for the Longformer-style sparse attention module.
//!
//! Kept local (not merged into [`crate::error::TrustformerError`]) so that
//! attention-internal diagnostics do not pollute the public transformer error
//! enum.  A [`From`] bridge forwards errors to the crate-wide type when
//! invoked from higher-level code.

use thiserror::Error;

/// Errors raised during sparse attention mask generation or forward pass.
#[derive(Debug, Clone, Error, PartialEq)]
pub enum SparseAttentionError {
    /// Window size must be strictly positive.
    #[error("invalid window_size: must be > 0, got {0}")]
    InvalidWindowSize(usize),

    /// Sequence length must be strictly positive.
    #[error("invalid sequence length: must be > 0, got {0}")]
    InvalidSequenceLength(usize),

    /// One or more global token indices exceed the sequence length.
    #[error("global token index {index} is out of bounds for sequence length {seq_len}")]
    InvalidGlobalIndices { index: usize, seq_len: usize },

    /// Query, key, or value tensors have incompatible shapes.
    #[error("dimension mismatch: {context} — expected {expected}, got {got}")]
    DimensionMismatch {
        context: String,
        expected: usize,
        got: usize,
    },

    /// A softmax row collapsed to zero mass after masking.
    #[error("numerical instability: softmax denominator is zero at position {position}")]
    NumericalInstability { position: usize },
}

/// Result alias used across the sparse-attention module.
pub type SparseAttentionResult<T> = Result<T, SparseAttentionError>;

impl From<SparseAttentionError> for crate::error::TrustformerError {
    fn from(err: SparseAttentionError) -> Self {
        crate::error::TrustformerError::CompilationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_contains_context() {
        let err = SparseAttentionError::InvalidWindowSize(0);
        let msg = err.to_string();
        assert!(msg.contains("window_size"));
        assert!(msg.contains("0"));
    }

    #[test]
    fn bridges_into_trustformer_error() {
        let err = SparseAttentionError::InvalidWindowSize(0);
        let bridged: crate::error::TrustformerError = err.into();
        assert!(bridged.to_string().contains("window_size"));
    }

    #[test]
    fn global_index_error_message() {
        let err = SparseAttentionError::InvalidGlobalIndices {
            index: 42,
            seq_len: 16,
        };
        assert!(err.to_string().contains("42"));
        assert!(err.to_string().contains("16"));
    }

    #[test]
    fn dimension_mismatch_message() {
        let err = SparseAttentionError::DimensionMismatch {
            context: "query rows".into(),
            expected: 32,
            got: 16,
        };
        let msg = err.to_string();
        assert!(msg.contains("query rows"));
        assert!(msg.contains("32"));
        assert!(msg.contains("16"));
    }
}

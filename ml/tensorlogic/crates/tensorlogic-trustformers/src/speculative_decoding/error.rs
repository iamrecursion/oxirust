//! Error taxonomy for the speculative decoder.
//!
//! Kept local (not merged into [`crate::error::TrustformerError`]) so that
//! decoder-internal diagnostics do not pollute the public transformer error
//! enum.  A [`From`] bridge forwards errors to the crate-wide type when the
//! decoder is invoked from higher-level code.

use thiserror::Error;

/// Errors that can be raised during speculative decoding.
#[derive(Debug, Clone, Error, PartialEq)]
pub enum SpeculativeDecodingError {
    /// A draft and target model disagreed on `vocab_size`.
    #[error(
        "vocab size mismatch between draft ({draft}) and target ({target}) \
         speculative-decoding models"
    )]
    VocabMismatch { draft: usize, target: usize },

    /// A distribution row had the wrong width.
    #[error("distribution row width mismatch: expected {expected}, got {got}")]
    DistributionWidthMismatch { expected: usize, got: usize },

    /// The draft model returned a proposal whose `tokens` / `token_logprobs`
    /// / `distributions` vectors disagreed in length.
    #[error(
        "draft proposal shape mismatch: tokens={tokens}, token_logprobs={logprobs}, \
         distributions={distributions}"
    )]
    DraftShapeMismatch {
        tokens: usize,
        logprobs: usize,
        distributions: usize,
    },

    /// The target model returned the wrong number of distribution rows.
    #[error("target verification shape mismatch: expected {expected} rows (k+1), got {got}")]
    TargetShapeMismatch { expected: usize, got: usize },

    /// The caller asked for `k == 0` draft tokens (or similar degenerate
    /// configuration).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// The caller supplied an empty prefix to `generate` even though the
    /// configured models require at least one bos/sos token.
    #[error("speculative decoding was invoked with an empty prefix")]
    EmptyPrefix,

    /// A token id produced by a model was outside the configured vocabulary.
    #[error("token id {token} is out of range for vocabulary size {vocab_size}")]
    TokenOutOfRange { token: usize, vocab_size: usize },

    /// A probability row collapsed to zero mass — typically because
    /// `max(0, p_target - p_draft)` was identically zero on every index, which
    /// happens iff `p_target == p_draft` everywhere.  We fall back to the raw
    /// target distribution in that case; this variant is reserved for cases
    /// where even that is degenerate.
    #[error("no mass left in adjusted distribution and target fallback is also zero")]
    DegenerateDistribution,

    /// A model implementation returned a descriptive error.
    #[error("model error: {0}")]
    ModelError(String),
}

/// Result alias used across the speculative-decoding module.
pub type SpeculativeDecodingResult<T> = Result<T, SpeculativeDecodingError>;

impl From<SpeculativeDecodingError> for crate::error::TrustformerError {
    fn from(err: SpeculativeDecodingError) -> Self {
        crate::error::TrustformerError::CompilationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_contains_context() {
        let err = SpeculativeDecodingError::VocabMismatch {
            draft: 10,
            target: 20,
        };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("20"));
        assert!(msg.contains("vocab"));
    }

    #[test]
    fn bridges_into_trustformer_error() {
        let err = SpeculativeDecodingError::InvalidConfig("k must be > 0".into());
        let bridged: crate::error::TrustformerError = err.into();
        assert!(bridged.to_string().contains("k must be > 0"));
    }

    #[test]
    fn empty_prefix_is_distinct() {
        let err = SpeculativeDecodingError::EmptyPrefix;
        assert!(err.to_string().contains("empty prefix"));
    }
}

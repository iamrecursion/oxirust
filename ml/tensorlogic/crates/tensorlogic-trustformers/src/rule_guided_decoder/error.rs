//! Error types for the rule-guided decoder.
//!
//! We keep these types local to the module to avoid polluting the crate-wide
//! [`crate::error::TrustformerError`] with decoder-specific variants.  A
//! [`From`] impl bridges the two when the decoder is invoked from higher-level
//! code.

use thiserror::Error;

use tensorlogic_infer::beam_search::BeamSearchError;

/// Error cases specific to rule-guided decoding.
#[derive(Debug, Error)]
pub enum RuleGuidedError {
    /// Compiling a `TLExpr` into a runtime representation failed.
    #[error("rule-guided decoder compilation error: {0}")]
    CompilationError(String),

    /// The caller supplied an invalid configuration (e.g. lambda < 0).
    #[error("rule-guided decoder configuration error: {0}")]
    InvalidConfig(String),

    /// The caller's scoring function returned an error (forwarded from
    /// `BeamSearchDecoder`).
    #[error("beam search failure: {0}")]
    BeamSearch(#[from] BeamSearchError),

    /// The wrapped score function returned a logits row whose length did not
    /// match the configured vocabulary size.
    #[error("logits row width mismatch: expected {expected}, got {got}")]
    LogitsWidthMismatch { expected: usize, got: usize },
}

/// Result alias for rule-guided decoder operations.
pub type RuleGuidedResult<T> = Result<T, RuleGuidedError>;

impl From<RuleGuidedError> for crate::error::TrustformerError {
    fn from(err: RuleGuidedError) -> Self {
        crate::error::TrustformerError::CompilationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_contains_context() {
        let err = RuleGuidedError::InvalidConfig("lambda must be non-negative".into());
        let msg = err.to_string();
        assert!(msg.contains("lambda"));
        assert!(msg.contains("configuration"));
    }

    #[test]
    fn bridges_into_trustformer_error() {
        let err = RuleGuidedError::CompilationError("oops".into());
        let bridged: crate::error::TrustformerError = err.into();
        assert!(bridged.to_string().contains("oops"));
    }
}

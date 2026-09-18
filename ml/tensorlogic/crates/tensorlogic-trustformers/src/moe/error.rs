//! Error taxonomy for the research-preview numerical MoE layer.
//!
//! Kept local (not merged into [`crate::error::TrustformerError`]) so that
//! MoE-internal diagnostics do not pollute the public transformer error
//! enum. A [`From`] bridge forwards errors to the crate-wide type when the
//! layer is invoked from higher-level code.

use thiserror::Error;

/// Errors that can be raised by the numerical Mixture-of-Experts layer.
#[derive(Debug, Clone, Error, PartialEq)]
pub enum MoeError {
    /// The caller tried to build a [`super::MoELayer`] with an empty
    /// `Vec<Box<dyn Expert>>`.
    #[error("MoE layer constructed with zero experts")]
    EmptyExpertPool,

    /// The requested top-k value is outside `1..=num_experts`.
    #[error("invalid top-k: k={k} must be in 1..={num_experts}")]
    InvalidTopK {
        /// Requested `k`.
        k: usize,
        /// Size of the expert pool.
        num_experts: usize,
    },

    /// An input vector had the wrong length for the gate or expert pool.
    #[error("shape mismatch: expected feature length {expected}, got {got}")]
    ShapeMismatch {
        /// Expected length.
        expected: usize,
        /// Observed length.
        got: usize,
    },

    /// `capacity_factor` must be strictly positive and finite.
    #[error("invalid capacity factor: {value} (must be strictly positive and finite)")]
    InvalidCapacityFactor {
        /// The offending value.
        value: f64,
    },
}

/// Result alias used inside the `moe` research preview.
pub type MoeResult<T> = Result<T, MoeError>;

impl From<MoeError> for crate::error::TrustformerError {
    fn from(err: MoeError) -> Self {
        crate::error::TrustformerError::CompilationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_topk_display_contains_context() {
        let err = MoeError::InvalidTopK {
            k: 5,
            num_experts: 2,
        };
        let msg = err.to_string();
        assert!(msg.contains('5'));
        assert!(msg.contains('2'));
    }

    #[test]
    fn capacity_factor_error_carries_value() {
        let err = MoeError::InvalidCapacityFactor { value: -1.0 };
        assert!(err.to_string().contains("-1"));
    }

    #[test]
    fn bridges_into_trustformer_error() {
        let err = MoeError::EmptyExpertPool;
        let bridged: crate::error::TrustformerError = err.into();
        assert!(bridged.to_string().contains("MoE"));
    }
}

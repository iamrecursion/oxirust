//! Error type for the Kernel PCA research-preview module.
//!
//! This is intentionally separate from the crate-wide
//! [`crate::error::KernelError`]: the Kernel PCA API exposes a narrow set
//! of failure modes (fit-before-transform, bad dimensions, degenerate
//! eigenproblems, ...) that are easier to pattern-match on via a
//! dedicated enum.
//!
//! The error implements [`std::error::Error`] via `thiserror`, so it
//! converts cleanly into `anyhow::Error` or any other boxed-error type a
//! caller might already be using.

use thiserror::Error;

/// Result alias used throughout the [`crate::kernel_pca`] module.
pub type KernelPcaResult<T> = std::result::Result<T, KernelPcaError>;

/// Failure modes for [`crate::kernel_pca::KernelPCA`] and friends.
///
/// Variants are deliberately coarse — one variant per *kind* of failure
/// — and each carries a `String` message that names the offending
/// argument. The five variants listed below match the
/// v0.2.0 research-preview contract.
#[derive(Debug, Error)]
pub enum KernelPcaError {
    /// Operation was requested against a model that has not been fitted
    /// yet (e.g. `transform` before `fit`).
    #[error("Kernel PCA model has not been fitted yet: {0}")]
    NotFitted(String),

    /// Input violates an invariant of the Kernel PCA API — empty
    /// training set, zero-dimensional feature vector, `n_components = 0`
    /// requested, etc.
    #[error("Invalid input to Kernel PCA: {0}")]
    InvalidInput(String),

    /// The underlying symmetric eigensolver refused the problem or
    /// failed numerically. The inner `String` is the message reported
    /// by the solver.
    #[error("Eigendecomposition of the centered Gram matrix failed: {0}")]
    EigendecompositionFailed(String),

    /// Vector or matrix supplied to `transform` has a different feature
    /// dimension than the data used at `fit` time.
    #[error(
        "Dimension mismatch: expected feature dimension {expected}, got {got} (context: {context})"
    )]
    DimensionMismatch {
        /// Feature dimension that was recorded at `fit` time.
        expected: usize,
        /// Feature dimension that was supplied at `transform` time.
        got: usize,
        /// Human-readable description of where the mismatch occurred.
        context: String,
    },

    /// The eigendecomposition produced fewer positive eigenvalues than
    /// the number of components requested by the caller — i.e. the
    /// kernel matrix is effectively low-rank and the requested KPCA
    /// embedding dimension is unreachable.
    #[error(
        "Requested {requested} components but only {available} positive eigenvalues are available"
    )]
    InsufficientComponents {
        /// `n_components` passed to the config.
        requested: usize,
        /// Number of eigenvalues above the positivity threshold.
        available: usize,
    },
}

impl KernelPcaError {
    /// Convenience constructor used throughout the module to wrap a
    /// [`crate::error::KernelError`] produced by a kernel evaluation
    /// into a [`KernelPcaError::InvalidInput`] with the kernel's error
    /// message preserved. Kernel failures during KPCA are almost always
    /// caused by bad input (wrong feature dimension, NaN, ...), not by
    /// internal KPCA bugs, so this mapping is usually the right one.
    pub(crate) fn from_kernel(err: crate::error::KernelError) -> Self {
        KernelPcaError::InvalidInput(format!("kernel evaluation failed: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_fitted_display_contains_context() {
        let err = KernelPcaError::NotFitted("transform".to_string());
        let msg = err.to_string();
        assert!(msg.contains("not been fitted"));
        assert!(msg.contains("transform"));
    }

    #[test]
    fn dimension_mismatch_display_mentions_both_sides() {
        let err = KernelPcaError::DimensionMismatch {
            expected: 3,
            got: 5,
            context: "transform input".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("3"));
        assert!(msg.contains("5"));
        assert!(msg.contains("transform input"));
    }

    #[test]
    fn insufficient_components_display_is_informative() {
        let err = KernelPcaError::InsufficientComponents {
            requested: 10,
            available: 3,
        };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn from_kernel_wraps_message() {
        let kernel_err = crate::error::KernelError::ComputationError("boom".to_string());
        let pca_err = KernelPcaError::from_kernel(kernel_err);
        let msg = pca_err.to_string();
        assert!(msg.contains("boom"));
    }
}

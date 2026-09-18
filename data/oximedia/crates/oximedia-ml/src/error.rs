//! Error types for `oximedia-ml`.
//!
//! Every fallible operation in the crate returns [`MlResult<T>`], which
//! is an alias for `Result<T, MlError>`. The variants are designed so
//! call sites never need to reach into backend-specific error types:
//! backend failures are flattened into [`MlError::OnnxRuntime`] or
//! [`MlError::ModelLoad`], and user-supplied data problems surface as
//! [`MlError::InvalidInput`] / [`MlError::Preprocess`] /
//! [`MlError::Postprocess`].
//!
//! ## Pattern
//!
//! ```no_run
//! use oximedia_ml::{MlError, MlResult};
//!
//! fn check_top_k(k: usize) -> MlResult<()> {
//!     if k == 0 {
//!         return Err(MlError::invalid_input("top_k must be >= 1"));
//!     }
//!     Ok(())
//! }
//! ```
//!
//! The [`MlError::pipeline`], [`MlError::invalid_input`],
//! [`MlError::preprocess`], and [`MlError::postprocess`] constructors
//! accept anything `Into<String>` so integrators can propagate rich
//! messages without allocating at call sites that already have owned
//! strings.

use std::path::PathBuf;
use thiserror::Error;

/// Result alias for ML operations.
///
/// All fallible APIs in this crate return `MlResult<T>`; pattern-match
/// on [`MlError`] to handle the different failure categories.
pub type MlResult<T> = Result<T, MlError>;

/// Errors surfaced by oximedia-ml.
///
/// Construct user-facing failures with [`MlError::invalid_input`],
/// [`MlError::preprocess`], [`MlError::postprocess`], or
/// [`MlError::pipeline`]; the remaining variants are usually produced by
/// the crate itself and matched on at call sites.
#[derive(Debug, Error)]
pub enum MlError {
    /// The requested device is either unknown, unavailable, or built without its feature flag.
    #[error("device '{0}' is not available in this build")]
    DeviceUnavailable(String),

    /// A feature required to perform the operation is disabled.
    #[error("required feature '{0}' is not enabled (re-build oximedia-ml with --features {0})")]
    FeatureDisabled(&'static str),

    /// The model file could not be loaded from disk.
    #[error("failed to load model at {path}: {reason}")]
    ModelLoad {
        /// Model path that failed to load.
        path: PathBuf,
        /// Human-readable reason provided by the backend.
        reason: String,
    },

    /// Input tensor(s) did not match the model's contract.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Pre-processing failed (e.g. image dimensions don't fit).
    #[error("preprocess error: {0}")]
    Preprocess(String),

    /// Post-processing failed (e.g. reading an output tensor).
    #[error("postprocess error: {0}")]
    Postprocess(String),

    /// A pipeline sub-component failed to execute.
    #[error("pipeline error ({stage}): {message}")]
    Pipeline {
        /// Pipeline stage that raised the error.
        stage: &'static str,
        /// Descriptive error message.
        message: String,
    },

    /// The cache capacity was zero, which is invalid.
    #[error("model cache capacity must be at least 1")]
    CacheCapacityZero,

    /// Underlying ONNX runtime error (feature-gated).
    #[error("onnx runtime error: {0}")]
    OnnxRuntime(String),
}

impl MlError {
    /// Convenience constructor for pipeline errors.
    #[must_use]
    pub fn pipeline(stage: &'static str, message: impl Into<String>) -> Self {
        Self::Pipeline {
            stage,
            message: message.into(),
        }
    }

    /// Convenience constructor for invalid input errors.
    #[must_use]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Convenience constructor for preprocess errors.
    #[must_use]
    pub fn preprocess(message: impl Into<String>) -> Self {
        Self::Preprocess(message.into())
    }

    /// Convenience constructor for postprocess errors.
    #[must_use]
    pub fn postprocess(message: impl Into<String>) -> Self {
        Self::Postprocess(message.into())
    }
}

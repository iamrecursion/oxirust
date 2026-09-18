//! Error types for `oximedia-compat-cv2`.

use thiserror::Error;

/// Errors that can occur when using the cv2 compatibility layer.
#[derive(Debug, Error)]
pub enum Cv2Error {
    /// I/O error when reading or writing files.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Image codec error (decode or encode failure).
    #[error("image codec error: {0}")]
    Codec(String),

    /// An operation was called with a flag value it does not support.
    #[error("unsupported flag for `{name}`: {value}")]
    UnsupportedFlag { name: &'static str, value: i32 },

    /// Operation does not support the given `MatType`.
    #[error("unsupported dtype: {mat_type:?}")]
    UnsupportedDtype { mat_type: crate::MatType },

    /// Feature is planned but not yet implemented in this slice.
    #[error("feature not implemented: {name} — see {refinement}")]
    FeatureNotImplemented {
        name: &'static str,
        refinement: &'static str,
    },

    /// Feature requires a compile-time cargo feature that is not enabled.
    #[error("feature not enabled: compile with cargo feature \"{feature}\"")]
    FeatureNotEnabled { feature: &'static str },

    /// Size of two operands does not match.
    #[error("size mismatch: expected {expected:?}, got {actual:?}")]
    SizeMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
    },

    /// File extension could not be mapped to a known image format.
    #[error("unknown file extension: {ext}")]
    UnknownExtension { ext: String },

    /// Error raised by the `dnn` (deep-neural-network) compatibility layer.
    ///
    /// Wraps failures originating from the underlying ONNX runtime, model
    /// loading, tensor reshaping, or operations on `Mat` blobs.
    #[error("dnn error: {0}")]
    Dnn(String),
}

/// Result alias for `Cv2Error`.
pub type Cv2Result<T> = Result<T, Cv2Error>;

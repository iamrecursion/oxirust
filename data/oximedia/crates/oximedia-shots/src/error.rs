//! Error types for shot detection and classification.

use thiserror::Error;

/// Result type for shot operations.
pub type ShotResult<T> = Result<T, ShotError>;

/// Errors that can occur during shot detection and classification.
#[derive(Debug, Error)]
pub enum ShotError {
    /// Invalid frame data.
    #[error("Invalid frame data: {0}")]
    InvalidFrame(String),

    /// Invalid shot parameters.
    #[error("Invalid shot parameters: {0}")]
    InvalidParameters(String),

    /// Shot detection failed.
    #[error("Shot detection failed: {0}")]
    DetectionFailed(String),

    /// Shot classification failed.
    #[error("Shot classification failed: {0}")]
    ClassificationFailed(String),

    /// Camera movement detection failed.
    #[error("Camera movement detection failed: {0}")]
    MovementDetectionFailed(String),

    /// Scene grouping failed.
    #[error("Scene grouping failed: {0}")]
    SceneGroupingFailed(String),

    /// Export failed.
    #[error("Export failed: {0}")]
    ExportFailed(String),

    /// Computer vision error.
    #[error("CV error: {0}")]
    CvError(String),

    /// Scene analysis error.
    #[error("Scene error: {0}")]
    SceneError(String),

    /// Core error.
    #[error("Core error: {0}")]
    CoreError(String),

    /// EDL error.
    #[error("EDL error: {0}")]
    EdlError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid metadata value or key.
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),

    /// Item not found.
    #[error("Not found: {0}")]
    NotFound(String),

    /// ML pipeline error (only available when the `onnx` feature is enabled).
    #[cfg(feature = "onnx")]
    #[error("ML error: {0}")]
    MlError(#[from] oximedia_ml::MlError),
}

impl From<oximedia_cv::error::CvError> for ShotError {
    fn from(err: oximedia_cv::error::CvError) -> Self {
        Self::CvError(err.to_string())
    }
}

impl From<oximedia_scene::error::SceneError> for ShotError {
    fn from(err: oximedia_scene::error::SceneError) -> Self {
        Self::SceneError(err.to_string())
    }
}

impl From<oximedia_core::error::OxiError> for ShotError {
    fn from(err: oximedia_core::error::OxiError) -> Self {
        Self::CoreError(err.to_string())
    }
}

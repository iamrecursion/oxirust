//! Error types for scene understanding.

use thiserror::Error;

/// Result type for scene operations.
pub type SceneResult<T> = Result<T, SceneError>;

/// Error types for scene understanding operations.
#[derive(Error, Debug)]
pub enum SceneError {
    /// Invalid input dimensions.
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Feature extraction failed.
    #[error("Feature extraction failed: {0}")]
    FeatureExtractionFailed(String),

    /// Classification failed.
    #[error("Classification failed: {0}")]
    ClassificationFailed(String),

    /// Detection failed.
    #[error("Detection failed: {0}")]
    DetectionFailed(String),

    /// Segmentation failed.
    #[error("Segmentation failed: {0}")]
    SegmentationFailed(String),

    /// Model not loaded.
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    /// Insufficient data for analysis.
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Computer vision error.
    #[error("CV error: {0}")]
    CvError(#[from] oximedia_cv::error::CvError),

    /// Core error.
    #[error("Core error: {0}")]
    CoreError(#[from] oximedia_core::error::OxiError),

    /// ML pipeline error (only available when the `onnx` feature is enabled).
    #[cfg(feature = "onnx")]
    #[error("ML error: {0}")]
    MlError(#[from] oximedia_ml::MlError),
}

//! Error types for automated video editing operations.
//!
//! This module provides the [`AutoError`] type which represents all errors
//! that can occur during automated editing operations, and the [`AutoResult`]
//! type alias for convenient use.

use oximedia_core::OxiError;
use oximedia_cv::CvError;
use oximedia_edit::EditError;

/// Error type for automated video editing operations.
///
/// This enum covers all possible errors that can occur during automated
/// editing, highlight detection, and scene scoring operations.
///
/// # Examples
///
/// ```
/// use oximedia_auto::error::{AutoError, AutoResult};
///
/// fn validate_duration(duration_ms: i64) -> AutoResult<()> {
///     if duration_ms <= 0 {
///         return Err(AutoError::InvalidDuration { duration_ms });
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, thiserror::Error)]
pub enum AutoError {
    /// Invalid duration value.
    #[error("Invalid duration: {duration_ms}ms (must be positive)")]
    InvalidDuration {
        /// Duration in milliseconds.
        duration_ms: i64,
    },

    /// Invalid parameter value.
    #[error("Invalid parameter '{name}': {value}")]
    InvalidParameter {
        /// Parameter name.
        name: String,
        /// Invalid value description.
        value: String,
    },

    /// Invalid threshold value.
    #[error("Invalid threshold: {threshold} (must be between {min} and {max})")]
    InvalidThreshold {
        /// Threshold value.
        threshold: f64,
        /// Minimum allowed value.
        min: f64,
        /// Maximum allowed value.
        max: f64,
    },

    /// Insufficient data for analysis.
    #[error("Insufficient data: {message}")]
    InsufficientData {
        /// Description of the missing data.
        message: String,
    },

    /// No highlights detected.
    #[error("No highlights detected in video")]
    NoHighlights,

    /// Invalid clip selection.
    #[error("Invalid clip selection: {message}")]
    InvalidClipSelection {
        /// Description of the invalid selection.
        message: String,
    },

    /// Scene detection failed.
    #[error("Scene detection failed: {message}")]
    SceneDetectionFailed {
        /// Error message.
        message: String,
    },

    /// Audio analysis failed.
    #[error("Audio analysis failed: {message}")]
    AudioAnalysisFailed {
        /// Error message.
        message: String,
    },

    /// Motion analysis failed.
    #[error("Motion analysis failed: {message}")]
    MotionAnalysisFailed {
        /// Error message.
        message: String,
    },

    /// Face detection failed.
    #[error("Face detection failed: {message}")]
    FaceDetectionFailed {
        /// Error message.
        message: String,
    },

    /// Object detection failed.
    #[error("Object detection failed: {message}")]
    ObjectDetectionFailed {
        /// Error message.
        message: String,
    },

    /// Assembly failed.
    #[error("Video assembly failed: {message}")]
    AssemblyFailed {
        /// Error message.
        message: String,
    },

    /// Rules validation failed.
    #[error("Rules validation failed: {message}")]
    RulesValidationFailed {
        /// Error message.
        message: String,
    },

    /// Configuration error.
    #[error("Configuration error: {message}")]
    ConfigurationError {
        /// Error message.
        message: String,
    },

    /// Timeline error.
    #[error("Timeline error: {message}")]
    TimelineError {
        /// Error message.
        message: String,
    },

    /// Rendering error.
    #[error("Rendering error: {message}")]
    RenderingError {
        /// Error message.
        message: String,
    },

    /// Core library error.
    #[error("OxiMedia core error: {0}")]
    Core(#[from] OxiError),

    /// Computer vision error.
    #[error("Computer vision error: {0}")]
    Cv(#[from] CvError),

    /// Edit library error.
    #[error("Edit library error: {0}")]
    Edit(#[from] EditError),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Other error.
    #[error("{0}")]
    Other(String),
}

impl AutoError {
    /// Create an invalid parameter error.
    #[must_use]
    pub fn invalid_parameter<S: Into<String>, V: Into<String>>(name: S, value: V) -> Self {
        Self::InvalidParameter {
            name: name.into(),
            value: value.into(),
        }
    }

    /// Create an insufficient data error.
    #[must_use]
    pub fn insufficient_data<S: Into<String>>(message: S) -> Self {
        Self::InsufficientData {
            message: message.into(),
        }
    }

    /// Create a scene detection failed error.
    #[must_use]
    pub fn scene_detection_failed<S: Into<String>>(message: S) -> Self {
        Self::SceneDetectionFailed {
            message: message.into(),
        }
    }

    /// Create an audio analysis failed error.
    #[must_use]
    pub fn audio_analysis_failed<S: Into<String>>(message: S) -> Self {
        Self::AudioAnalysisFailed {
            message: message.into(),
        }
    }

    /// Create a motion analysis failed error.
    #[must_use]
    pub fn motion_analysis_failed<S: Into<String>>(message: S) -> Self {
        Self::MotionAnalysisFailed {
            message: message.into(),
        }
    }

    /// Create a configuration error.
    #[must_use]
    pub fn configuration_error<S: Into<String>>(message: S) -> Self {
        Self::ConfigurationError {
            message: message.into(),
        }
    }

    /// Create an assembly failed error.
    #[must_use]
    pub fn assembly_failed<S: Into<String>>(message: S) -> Self {
        Self::AssemblyFailed {
            message: message.into(),
        }
    }
}

/// Result type for automated video editing operations.
pub type AutoResult<T> = Result<T, AutoError>;

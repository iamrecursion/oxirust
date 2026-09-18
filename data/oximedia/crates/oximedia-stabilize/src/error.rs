//! Error types for video stabilization.

use thiserror::Error;

/// Result type for stabilization operations.
pub type StabilizeResult<T> = Result<T, StabilizeError>;

/// Errors that can occur during video stabilization.
#[derive(Debug, Error)]
pub enum StabilizeError {
    /// Invalid configuration parameter
    #[error("Invalid parameter '{name}': {value}")]
    InvalidParameter {
        /// Parameter name
        name: String,
        /// Parameter value
        value: String,
    },

    /// Empty frame sequence
    #[error("Frame sequence is empty")]
    EmptyFrameSequence,

    /// Motion tracking failed
    #[error("Motion tracking failed: {0}")]
    MotionTrackingFailed(String),

    /// Motion estimation failed
    #[error("Motion estimation failed: {0}")]
    MotionEstimationFailed(String),

    /// Transform calculation failed
    #[error("Transform calculation failed: {0}")]
    TransformCalculationFailed(String),

    /// Frame warping failed
    #[error("Frame warping failed: {0}")]
    WarpingFailed(String),

    /// Insufficient features for tracking
    #[error("Insufficient features: found {found}, need at least {required}")]
    InsufficientFeatures {
        /// Number of features found
        found: usize,
        /// Minimum required features
        required: usize,
    },

    /// Matrix operation failed
    #[error("Matrix operation failed: {0}")]
    MatrixError(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimensions
        expected: String,
        /// Actual dimensions
        actual: String,
    },

    /// Rolling shutter correction failed
    #[error("Rolling shutter correction failed: {0}")]
    RollingShutterError(String),

    /// 3D stabilization failed
    #[error("3D stabilization failed: {0}")]
    ThreeDStabilizationError(String),

    /// Horizon detection failed
    #[error("Horizon detection failed: {0}")]
    HorizonDetectionError(String),

    /// Zoom optimization failed
    #[error("Zoom optimization failed: {0}")]
    ZoomOptimizationError(String),

    /// Multi-pass analysis failed
    #[error("Multi-pass analysis failed: {0}")]
    MultipassAnalysisError(String),

    /// Interpolation failed
    #[error("Interpolation failed: {0}")]
    InterpolationError(String),

    /// Invalid transform
    #[error("Invalid transform: {0}")]
    InvalidTransform(String),

    /// Feature tracking lost
    #[error("Feature tracking lost at frame {frame_index}")]
    TrackingLost {
        /// Frame index where tracking was lost
        frame_index: usize,
    },

    /// General error
    #[error("{0}")]
    General(String),
}

impl StabilizeError {
    /// Create an invalid parameter error.
    #[must_use]
    pub fn invalid_parameter(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name: name.into(),
            value: value.into(),
        }
    }

    /// Create a motion tracking error.
    #[must_use]
    pub fn motion_tracking(msg: impl Into<String>) -> Self {
        Self::MotionTrackingFailed(msg.into())
    }

    /// Create a motion estimation error.
    #[must_use]
    pub fn motion_estimation(msg: impl Into<String>) -> Self {
        Self::MotionEstimationFailed(msg.into())
    }

    /// Create a transform calculation error.
    #[must_use]
    pub fn transform_calculation(msg: impl Into<String>) -> Self {
        Self::TransformCalculationFailed(msg.into())
    }

    /// Create a warping error.
    #[must_use]
    pub fn warping(msg: impl Into<String>) -> Self {
        Self::WarpingFailed(msg.into())
    }

    /// Create an insufficient features error.
    #[must_use]
    pub const fn insufficient_features(found: usize, required: usize) -> Self {
        Self::InsufficientFeatures { found, required }
    }

    /// Create a matrix error.
    #[must_use]
    pub fn matrix(msg: impl Into<String>) -> Self {
        Self::MatrixError(msg.into())
    }

    /// Create a dimension mismatch error.
    #[must_use]
    pub fn dimension_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::DimensionMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = StabilizeError::invalid_parameter("smoothing_strength", "2.0");
        assert!(matches!(err, StabilizeError::InvalidParameter { .. }));

        let err = StabilizeError::insufficient_features(10, 50);
        assert!(matches!(err, StabilizeError::InsufficientFeatures { .. }));
    }

    #[test]
    fn test_error_display() {
        let err = StabilizeError::EmptyFrameSequence;
        assert_eq!(err.to_string(), "Frame sequence is empty");

        let err = StabilizeError::invalid_parameter("test", "value");
        assert!(err.to_string().contains("Invalid parameter"));
    }
}

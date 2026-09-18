//! Error types for computer vision operations.
//!
//! This module provides the [`CvError`] type which represents all errors
//! that can occur during computer vision operations, and the [`CvResult`]
//! type alias for convenient use.

use oximedia_core::OxiError;

/// Error type for computer vision operations.
///
/// This enum covers all possible errors that can occur during image processing,
/// detection, and transformation operations.
///
/// # Examples
///
/// ```
/// use oximedia_cv::error::{CvError, CvResult};
///
/// fn process_image(width: u32, height: u32) -> CvResult<()> {
///     if width == 0 || height == 0 {
///         return Err(CvError::InvalidDimensions { width, height });
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, thiserror::Error)]
pub enum CvError {
    /// Invalid image dimensions (width or height is zero).
    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },

    /// Invalid region of interest.
    #[error("Invalid ROI: ({x}, {y}, {width}, {height})")]
    InvalidRoi {
        /// ROI x coordinate.
        x: u32,
        /// ROI y coordinate.
        y: u32,
        /// ROI width.
        width: u32,
        /// ROI height.
        height: u32,
    },

    /// Invalid kernel size for filter operations.
    #[error("Invalid kernel size: {size} (must be odd and >= 3)")]
    InvalidKernelSize {
        /// The invalid kernel size.
        size: usize,
    },

    /// Color space conversion error.
    #[error("Color conversion error: {message}")]
    ColorConversion {
        /// Description of the error.
        message: String,
    },

    /// Unsupported pixel format.
    #[error("Unsupported pixel format: {format}")]
    UnsupportedFormat {
        /// The unsupported format name.
        format: String,
    },

    /// Detection failed.
    #[error("Detection failed: {message}")]
    DetectionFailed {
        /// Description of the failure.
        message: String,
    },

    /// Transform computation failed.
    #[error("Transform error: {message}")]
    TransformError {
        /// Description of the error.
        message: String,
    },

    /// Tracking operation failed.
    #[error("Tracking error: {message}")]
    TrackingError {
        /// Description of the error.
        message: String,
    },

    /// Insufficient data for operation.
    #[error("Insufficient data: expected {expected} bytes, got {actual}")]
    InsufficientData {
        /// Expected number of bytes.
        expected: usize,
        /// Actual number of bytes.
        actual: usize,
    },

    /// Matrix operation error.
    #[error("Matrix error: {message}")]
    MatrixError {
        /// Description of the error.
        message: String,
    },

    /// Invalid parameter value.
    #[error("Invalid parameter: {name} = {value}")]
    InvalidParameter {
        /// Parameter name.
        name: String,
        /// Invalid value description.
        value: String,
    },

    /// Core error from oximedia-core.
    #[error("Core error: {0}")]
    Core(#[from] OxiError),

    /// ONNX Runtime error.
    #[error("ONNX Runtime error: {message}")]
    OnnxRuntime {
        /// Description of the error.
        message: String,
    },

    /// Model loading error.
    #[error("Model load error: {message}")]
    ModelLoad {
        /// Description of the error.
        message: String,
    },

    /// Tensor operation error.
    #[error("Tensor error: {message}")]
    TensorError {
        /// Description of the error.
        message: String,
    },

    /// Shape mismatch error.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl CvError {
    /// Creates a new invalid dimensions error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::invalid_dimensions(0, 100);
    /// assert!(matches!(err, CvError::InvalidDimensions { width: 0, height: 100 }));
    /// ```
    #[must_use]
    pub const fn invalid_dimensions(width: u32, height: u32) -> Self {
        Self::InvalidDimensions { width, height }
    }

    /// Creates a new invalid ROI error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::invalid_roi(10, 20, 0, 50);
    /// assert!(matches!(err, CvError::InvalidRoi { .. }));
    /// ```
    #[must_use]
    pub const fn invalid_roi(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self::InvalidRoi {
            x,
            y,
            width,
            height,
        }
    }

    /// Creates a new invalid kernel size error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::invalid_kernel_size(4);
    /// assert!(matches!(err, CvError::InvalidKernelSize { size: 4 }));
    /// ```
    #[must_use]
    pub const fn invalid_kernel_size(size: usize) -> Self {
        Self::InvalidKernelSize { size }
    }

    /// Creates a new color conversion error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::color_conversion("Invalid color values");
    /// assert!(matches!(err, CvError::ColorConversion { .. }));
    /// ```
    #[must_use]
    pub fn color_conversion(message: impl Into<String>) -> Self {
        Self::ColorConversion {
            message: message.into(),
        }
    }

    /// Creates a new unsupported format error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::unsupported_format("YUV444");
    /// assert!(matches!(err, CvError::UnsupportedFormat { .. }));
    /// ```
    #[must_use]
    pub fn unsupported_format(format: impl Into<String>) -> Self {
        Self::UnsupportedFormat {
            format: format.into(),
        }
    }

    /// Creates a new detection failed error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::detection_failed("No faces found");
    /// assert!(matches!(err, CvError::DetectionFailed { .. }));
    /// ```
    #[must_use]
    pub fn detection_failed(message: impl Into<String>) -> Self {
        Self::DetectionFailed {
            message: message.into(),
        }
    }

    /// Creates a new transform error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::transform_error("Matrix is singular");
    /// assert!(matches!(err, CvError::TransformError { .. }));
    /// ```
    #[must_use]
    pub fn transform_error(message: impl Into<String>) -> Self {
        Self::TransformError {
            message: message.into(),
        }
    }

    /// Creates a new tracking error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::tracking_error("Tracker not initialized");
    /// assert!(matches!(err, CvError::TrackingError { .. }));
    /// ```
    #[must_use]
    pub fn tracking_error(message: impl Into<String>) -> Self {
        Self::TrackingError {
            message: message.into(),
        }
    }

    /// Creates a new insufficient data error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::insufficient_data(1024, 512);
    /// assert!(matches!(err, CvError::InsufficientData { expected: 1024, actual: 512 }));
    /// ```
    #[must_use]
    pub const fn insufficient_data(expected: usize, actual: usize) -> Self {
        Self::InsufficientData { expected, actual }
    }

    /// Creates a new matrix error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::matrix_error("Matrix dimensions mismatch");
    /// assert!(matches!(err, CvError::MatrixError { .. }));
    /// ```
    #[must_use]
    pub fn matrix_error(message: impl Into<String>) -> Self {
        Self::MatrixError {
            message: message.into(),
        }
    }

    /// Creates a new invalid parameter error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::invalid_parameter("sigma", "-1.0");
    /// assert!(matches!(err, CvError::InvalidParameter { .. }));
    /// ```
    #[must_use]
    pub fn invalid_parameter(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name: name.into(),
            value: value.into(),
        }
    }

    /// Creates a new ONNX Runtime error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::onnx_runtime("Session initialization failed");
    /// assert!(matches!(err, CvError::OnnxRuntime { .. }));
    /// ```
    #[must_use]
    pub fn onnx_runtime(message: impl Into<String>) -> Self {
        Self::OnnxRuntime {
            message: message.into(),
        }
    }

    /// Creates a new model load error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::model_load("File not found");
    /// assert!(matches!(err, CvError::ModelLoad { .. }));
    /// ```
    #[must_use]
    pub fn model_load(message: impl Into<String>) -> Self {
        Self::ModelLoad {
            message: message.into(),
        }
    }

    /// Creates a new tensor error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::tensor_error("Invalid data type");
    /// assert!(matches!(err, CvError::TensorError { .. }));
    /// ```
    #[must_use]
    pub fn tensor_error(message: impl Into<String>) -> Self {
        Self::TensorError {
            message: message.into(),
        }
    }

    /// Creates a new shape mismatch error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::error::CvError;
    ///
    /// let err = CvError::shape_mismatch(vec![1, 3, 224, 224], vec![1, 3, 256, 256]);
    /// assert!(matches!(err, CvError::ShapeMismatch { .. }));
    /// ```
    #[must_use]
    pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::ShapeMismatch { expected, actual }
    }
}

/// Result type alias for computer vision operations.
///
/// This is a convenience alias for `Result<T, CvError>`.
///
/// # Examples
///
/// ```
/// use oximedia_cv::error::CvResult;
///
/// fn process() -> CvResult<u32> {
///     Ok(42)
/// }
/// ```
pub type CvResult<T> = Result<T, CvError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_dimensions() {
        let err = CvError::invalid_dimensions(0, 100);
        assert!(matches!(
            err,
            CvError::InvalidDimensions {
                width: 0,
                height: 100
            }
        ));
        let msg = format!("{err}");
        assert!(msg.contains("0x100"));
    }

    #[test]
    fn test_invalid_roi() {
        let err = CvError::invalid_roi(10, 20, 0, 50);
        assert!(matches!(
            err,
            CvError::InvalidRoi {
                x: 10,
                y: 20,
                width: 0,
                height: 50
            }
        ));
    }

    #[test]
    fn test_invalid_kernel_size() {
        let err = CvError::invalid_kernel_size(4);
        assert!(matches!(err, CvError::InvalidKernelSize { size: 4 }));
        let msg = format!("{err}");
        assert!(msg.contains('4'));
    }

    #[test]
    fn test_color_conversion_error() {
        let err = CvError::color_conversion("Invalid color");
        assert!(matches!(err, CvError::ColorConversion { .. }));
        let msg = format!("{err}");
        assert!(msg.contains("Invalid color"));
    }

    #[test]
    fn test_unsupported_format() {
        let err = CvError::unsupported_format("YUV444");
        assert!(format!("{err}").contains("YUV444"));
    }

    #[test]
    fn test_detection_failed() {
        let err = CvError::detection_failed("No faces");
        assert!(format!("{err}").contains("No faces"));
    }

    #[test]
    fn test_transform_error() {
        let err = CvError::transform_error("Singular matrix");
        assert!(format!("{err}").contains("Singular matrix"));
    }

    #[test]
    fn test_insufficient_data() {
        let err = CvError::insufficient_data(1024, 512);
        let msg = format!("{err}");
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_matrix_error() {
        let err = CvError::matrix_error("Dimension mismatch");
        assert!(format!("{err}").contains("Dimension mismatch"));
    }

    #[test]
    fn test_invalid_parameter() {
        let err = CvError::invalid_parameter("sigma", "-1.0");
        let msg = format!("{err}");
        assert!(msg.contains("sigma"));
        assert!(msg.contains("-1.0"));
    }
}

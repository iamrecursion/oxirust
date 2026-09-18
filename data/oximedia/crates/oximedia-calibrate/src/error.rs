//! Error types for color calibration operations.

use thiserror::Error;

/// Result type for calibration operations.
pub type CalibrationResult<T> = Result<T, CalibrationError>;

/// Errors that can occur during color calibration operations.
#[derive(Error, Debug)]
pub enum CalibrationError {
    /// `ColorChecker` detection failed.
    #[error("ColorChecker detection failed: {0}")]
    ColorCheckerNotFound(String),

    /// Patch extraction failed.
    #[error("Patch extraction failed: {0}")]
    PatchExtractionFailed(String),

    /// Invalid patch count.
    #[error("Invalid patch count: expected {expected}, got {actual}")]
    InvalidPatchCount {
        /// Expected patch count.
        expected: usize,
        /// Actual patch count.
        actual: usize,
    },

    /// Profile generation failed.
    #[error("Profile generation failed: {0}")]
    ProfileGenerationFailed(String),

    /// ICC profile parsing failed.
    #[error("ICC profile parsing failed: {0}")]
    IccParseError(String),

    /// ICC profile invalid.
    #[error("ICC profile invalid: {0}")]
    IccInvalidProfile(String),

    /// LUT generation failed.
    #[error("LUT generation failed: {0}")]
    LutGenerationFailed(String),

    /// Measurement data invalid.
    #[error("Measurement data invalid: {0}")]
    InvalidMeasurementData(String),

    /// Display calibration failed.
    #[error("Display calibration failed: {0}")]
    DisplayCalibrationFailed(String),

    /// White balance failed.
    #[error("White balance failed: {0}")]
    WhiteBalanceFailed(String),

    /// Color temperature estimation failed.
    #[error("Color temperature estimation failed: {0}")]
    TemperatureEstimationFailed(String),

    /// Gamut mapping failed.
    #[error("Gamut mapping failed: {0}")]
    GamutMappingFailed(String),

    /// Chromatic adaptation failed.
    #[error("Chromatic adaptation failed: {0}")]
    ChromaticAdaptationFailed(String),

    /// Color matching failed.
    #[error("Color matching failed: {0}")]
    ColorMatchingFailed(String),

    /// Invalid color space.
    #[error("Invalid color space: {0}")]
    InvalidColorSpace(String),

    /// Invalid image dimensions.
    #[error("Invalid image dimensions: {0}")]
    InvalidImageDimensions(String),

    /// Image too small for calibration.
    #[error(
        "Image too small for calibration: minimum {min_width}x{min_height}, got {width}x{height}"
    )]
    ImageTooSmall {
        /// Actual width.
        width: usize,
        /// Actual height.
        height: usize,
        /// Minimum width.
        min_width: usize,
        /// Minimum height.
        min_height: usize,
    },

    /// Insufficient data for calibration.
    #[error("Insufficient data for calibration: {0}")]
    InsufficientData(String),

    /// Calibration verification failed.
    #[error("Calibration verification failed: {0}")]
    VerificationFailed(String),

    /// Invalid measurement value.
    #[error("Invalid measurement: {0}")]
    InvalidMeasurement(String),

    /// Numerical instability during calibration.
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// LUT error.
    #[error("LUT error: {0}")]
    Lut(#[from] oximedia_lut::LutError),

    /// CV error.
    #[error("CV error: {0}")]
    Cv(#[from] oximedia_cv::CvError),

    /// Core error.
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::OxiError),
}

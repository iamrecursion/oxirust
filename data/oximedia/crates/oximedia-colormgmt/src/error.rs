//! Error types for color management operations.

use thiserror::Error;

/// Result type for color management operations.
pub type Result<T> = std::result::Result<T, ColorError>;

/// Errors that can occur during color management operations.
#[derive(Debug, Error)]
pub enum ColorError {
    /// Invalid color value (e.g., out of range).
    #[error("Invalid color value: {0}")]
    InvalidColor(String),

    /// Invalid color space definition.
    #[error("Invalid color space: {0}")]
    InvalidColorSpace(String),

    /// ICC profile parsing error.
    #[error("ICC profile error: {0}")]
    IccProfile(String),

    /// LUT error.
    #[error("LUT error: {0}")]
    Lut(String),

    /// Matrix operation error.
    #[error("Matrix error: {0}")]
    Matrix(String),

    /// Gamut mapping error.
    #[error("Gamut mapping error: {0}")]
    GamutMapping(String),

    /// ACES transform error.
    #[error("ACES transform error: {0}")]
    AcesTransform(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Parse error.
    #[error("Parse error: {0}")]
    Parse(String),

    /// Unsupported feature.
    #[error("Unsupported feature: {0}")]
    Unsupported(String),
}

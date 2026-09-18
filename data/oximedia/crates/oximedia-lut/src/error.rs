//! Error types for LUT operations.

use std::io;
use thiserror::Error;

/// Result type for LUT operations.
pub type LutResult<T> = Result<T, LutError>;

/// Errors that can occur during LUT operations.
#[derive(Error, Debug)]
pub enum LutError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Parse error.
    #[error("Parse error: {0}")]
    Parse(String),

    /// Invalid LUT size.
    #[error("Invalid LUT size: expected {expected}, got {actual}")]
    InvalidSize {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },

    /// Invalid LUT data.
    #[error("Invalid LUT data: {0}")]
    InvalidData(String),

    /// Unsupported format.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Color space error.
    #[error("Color space error: {0}")]
    ColorSpace(String),

    /// Gamut error.
    #[error("Gamut error: {0}")]
    Gamut(String),

    /// Invalid color value.
    #[error("Invalid color value: {0}")]
    InvalidColor(String),

    /// LUT operation error.
    #[error("LUT operation error: {0}")]
    Operation(String),

    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Invalid interpolation method.
    #[error("Invalid interpolation method: {0}")]
    InvalidInterpolation(String),
}

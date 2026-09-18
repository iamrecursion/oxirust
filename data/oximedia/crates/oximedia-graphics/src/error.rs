//! Error types for the graphics engine

use thiserror::Error;

/// Result type for graphics operations
pub type Result<T> = std::result::Result<T, GraphicsError>;

/// Graphics engine errors
#[derive(Debug, Error)]
pub enum GraphicsError {
    /// Invalid dimensions
    #[error("Invalid dimensions: {0}x{1}")]
    InvalidDimensions(u32, u32),

    /// Invalid color value
    #[error("Invalid color value: {0}")]
    InvalidColor(String),

    /// Font loading error
    #[error("Font error: {0}")]
    FontError(String),

    /// Template parsing error
    #[error("Template error: {0}")]
    TemplateError(String),

    /// Rendering error
    #[error("Rendering error: {0}")]
    RenderError(String),

    /// Animation error
    #[error("Animation error: {0}")]
    AnimationError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Resource not found
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    /// GPU error
    #[cfg(feature = "gpu")]
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Server error
    #[cfg(feature = "server")]
    #[error("Server error: {0}")]
    ServerError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GraphicsError::InvalidDimensions(0, 0);
        assert_eq!(err.to_string(), "Invalid dimensions: 0x0");

        let err = GraphicsError::InvalidColor("invalid".to_string());
        assert_eq!(err.to_string(), "Invalid color value: invalid");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: GraphicsError = io_err.into();
        assert!(matches!(err, GraphicsError::IoError(_)));
    }
}

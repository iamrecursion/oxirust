//! Error types for the datasets module

use std::io;
use thiserror::Error;

/// Error type for datasets operations
#[derive(Error, Debug)]
pub enum DatasetsError {
    /// Invalid data format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Data loading error
    #[error("Loading error: {0}")]
    LoadingError(String),

    /// Format error
    #[error("Format error: {0}")]
    FormatError(String),

    /// Not found error
    #[error("Not found: {0}")]
    NotFound(String),

    /// Authentication error
    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    /// Download error
    #[error("Download error: {0}")]
    DownloadError(String),

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    /// Serialization/Deserialization error
    #[error("Serialization error: {0}")]
    SerdeError(String),

    /// GPU acceleration error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Processing error
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

impl PartialEq for DatasetsError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DatasetsError::InvalidFormat(a), DatasetsError::InvalidFormat(b)) => a == b,
            (DatasetsError::LoadingError(a), DatasetsError::LoadingError(b)) => a == b,
            (DatasetsError::FormatError(a), DatasetsError::FormatError(b)) => a == b,
            (DatasetsError::NotFound(a), DatasetsError::NotFound(b)) => a == b,
            (DatasetsError::AuthenticationError(a), DatasetsError::AuthenticationError(b)) => {
                a == b
            }
            (DatasetsError::DownloadError(a), DatasetsError::DownloadError(b)) => a == b,
            (DatasetsError::CacheError(a), DatasetsError::CacheError(b)) => a == b,
            (DatasetsError::IoError(a), DatasetsError::IoError(b)) => {
                // Compare io::Error by kind and message
                a.kind() == b.kind() && a.to_string() == b.to_string()
            }
            (DatasetsError::SerdeError(a), DatasetsError::SerdeError(b)) => a == b,
            (DatasetsError::GpuError(a), DatasetsError::GpuError(b)) => a == b,
            (DatasetsError::ComputationError(a), DatasetsError::ComputationError(b)) => a == b,
            (DatasetsError::ValidationError(a), DatasetsError::ValidationError(b)) => a == b,
            (DatasetsError::ProcessingError(a), DatasetsError::ProcessingError(b)) => a == b,
            (DatasetsError::Other(a), DatasetsError::Other(b)) => a == b,
            _ => false,
        }
    }
}

/// Result type for datasets operations
pub type Result<T> = std::result::Result<T, DatasetsError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_invalid_format_error() {
        let error = DatasetsError::InvalidFormat("test format".to_string());
        assert_eq!(error.to_string(), "Invalid format: test format");
    }

    #[test]
    fn test_loading_error() {
        let error = DatasetsError::LoadingError("test loading".to_string());
        assert_eq!(error.to_string(), "Loading error: test loading");
    }

    #[test]
    fn test_download_error() {
        let error = DatasetsError::DownloadError("test download".to_string());
        assert_eq!(error.to_string(), "Download error: test download");
    }

    #[test]
    fn test_cache_error() {
        let error = DatasetsError::CacheError("test cache".to_string());
        assert_eq!(error.to_string(), "Cache error: test cache");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let datasets_error: DatasetsError = io_error.into();

        match datasets_error {
            DatasetsError::IoError(_) => {
                assert!(datasets_error.to_string().contains("file not found"));
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_serde_error() {
        let error = DatasetsError::SerdeError("serialization failed".to_string());
        assert_eq!(
            error.to_string(),
            "Serialization error: serialization failed"
        );
    }

    #[test]
    fn test_gpu_error() {
        let error = DatasetsError::GpuError("CUDA initialization failed".to_string());
        assert_eq!(error.to_string(), "GPU error: CUDA initialization failed");
    }

    #[test]
    fn test_other_error() {
        let error = DatasetsError::Other("generic error".to_string());
        assert_eq!(error.to_string(), "Error: generic error");
    }

    #[test]
    fn test_error_debug_format() {
        let error = DatasetsError::InvalidFormat("debug test".to_string());
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("InvalidFormat"));
        assert!(debug_str.contains("debug test"));
    }

    #[test]
    fn test_result_type() {
        // Test Ok case
        let ok_result: Result<i32> = Ok(42);
        assert_eq!(ok_result, Ok(42));

        // Test Err case
        let err_result: Result<i32> = Err(DatasetsError::Other("test".to_string()));
        assert!(err_result.is_err());
    }

    #[test]
    fn test_error_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let datasets_err = DatasetsError::from(io_err);

        if let DatasetsError::IoError(ref inner) = datasets_err {
            assert_eq!(inner.kind(), io::ErrorKind::PermissionDenied);
        } else {
            panic!("Expected IoError variant");
        }
    }

    #[test]
    fn test_error_chain() {
        // Test that error displays work correctly in error chains
        let error = DatasetsError::LoadingError("failed to parse CSV".to_string());
        let result: Result<()> = Err(error);

        match result {
            Ok(_) => panic!("Expected error"),
            Err(e) => {
                assert_eq!(e.to_string(), "Loading error: failed to parse CSV");
            }
        }
    }
}

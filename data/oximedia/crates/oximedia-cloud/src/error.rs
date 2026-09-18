//! Error types for cloud operations

/// Result type alias for cloud operations
pub type Result<T> = std::result::Result<T, CloudError>;

/// Errors that can occur during cloud operations
#[derive(Debug, thiserror::Error)]
pub enum CloudError {
    /// Storage operation failed
    #[error("Storage error: {0}")]
    Storage(String),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),

    /// Authentication failed
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Authorization failed
    #[error("Authorization error: {0}")]
    Authorization(String),

    /// Object not found
    #[error("Object not found: {0}")]
    NotFound(String),

    /// Object already exists
    #[error("Object already exists: {0}")]
    AlreadyExists(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Transfer error
    #[error("Transfer error: {0}")]
    Transfer(String),

    /// Checksum mismatch
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    /// Quota exceeded
    #[error("Quota exceeded: {0}")]
    QuotaExceeded(String),

    /// Service unavailable
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Timeout error
    #[error("Operation timeout: {0}")]
    Timeout(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Media service error
    #[error("Media service error: {0}")]
    MediaService(String),

    /// Encryption error
    #[error("Encryption error: {0}")]
    Encryption(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Generic error
    #[error("Cloud error: {0}")]
    Other(String),
}

impl CloudError {
    /// Check if the error is retryable
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            CloudError::Network(_)
                | CloudError::ServiceUnavailable(_)
                | CloudError::Timeout(_)
                | CloudError::RateLimitExceeded(_)
        )
    }

    /// Check if the error is a client error (4xx)
    #[must_use]
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            CloudError::Authentication(_)
                | CloudError::Authorization(_)
                | CloudError::NotFound(_)
                | CloudError::AlreadyExists(_)
                | CloudError::InvalidConfig(_)
                | CloudError::InvalidParameter(_)
        )
    }

    /// Check if the error is a server error (5xx)
    #[must_use]
    pub fn is_server_error(&self) -> bool {
        matches!(
            self,
            CloudError::ServiceUnavailable(_) | CloudError::Storage(_)
        )
    }
}

// Conversion implementations for common error types
impl From<reqwest::Error> for CloudError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            CloudError::Timeout(err.to_string())
        } else {
            CloudError::Network(err.to_string())
        }
    }
}

impl From<serde_json::Error> for CloudError {
    fn from(err: serde_json::Error) -> Self {
        CloudError::Serialization(err.to_string())
    }
}

impl From<std::io::Error> for CloudError {
    fn from(err: std::io::Error) -> Self {
        CloudError::Storage(err.to_string())
    }
}

impl From<url::ParseError> for CloudError {
    fn from(err: url::ParseError) -> Self {
        CloudError::InvalidConfig(err.to_string())
    }
}

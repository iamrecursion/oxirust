//! Error types for network streaming operations.
//!
//! This module provides the [`NetError`] type which represents all errors
//! that can occur during network streaming operations.

use std::io;
use thiserror::Error;

/// Error type for network streaming operations.
///
/// This enum covers all possible errors that can occur during network
/// streaming, including connection errors, protocol errors, and timeouts.
#[derive(Debug, Error)]
pub enum NetError {
    /// I/O error during network operations.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Connection error.
    #[error("Connection error: {0}")]
    Connection(String),

    /// Protocol error with description.
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Timeout during network operation.
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Invalid URL format.
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    /// HTTP error with status code.
    #[error("HTTP error: status {status}, {message}")]
    Http {
        /// HTTP status code.
        status: u16,
        /// Error message.
        message: String,
    },

    /// Parse error in protocol data.
    #[error("Parse error at offset {offset}: {message}")]
    Parse {
        /// Byte offset where the error occurred.
        offset: u64,
        /// Description of the parse error.
        message: String,
    },

    /// Invalid state for the requested operation.
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Handshake failed.
    #[error("Handshake failed: {0}")]
    Handshake(String),

    /// Authentication failed.
    #[error("Authentication failed: {0}")]
    Authentication(String),

    /// Resource not found.
    #[error("Not found: {0}")]
    NotFound(String),

    /// Segment error in streaming protocols.
    #[error("Segment error: {0}")]
    Segment(String),

    /// Playlist error in HLS/DASH.
    #[error("Playlist error: {0}")]
    Playlist(String),

    /// Encoding/decoding error.
    #[error("Encoding error: {0}")]
    Encoding(String),

    /// Buffer overflow or underflow.
    #[error("Buffer error: {0}")]
    Buffer(String),

    /// End of stream reached.
    #[error("End of stream")]
    Eof,

    /// Core library error.
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::OxiError),
}

impl NetError {
    /// Creates a new connection error.
    #[must_use]
    pub fn connection(message: impl Into<String>) -> Self {
        Self::Connection(message.into())
    }

    /// Creates a new protocol error.
    #[must_use]
    pub fn protocol(message: impl Into<String>) -> Self {
        Self::Protocol(message.into())
    }

    /// Creates a new timeout error.
    #[must_use]
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::Timeout(message.into())
    }

    /// Creates a new invalid URL error.
    #[must_use]
    pub fn invalid_url(message: impl Into<String>) -> Self {
        Self::InvalidUrl(message.into())
    }

    /// Creates a new HTTP error.
    #[must_use]
    pub fn http(status: u16, message: impl Into<String>) -> Self {
        Self::Http {
            status,
            message: message.into(),
        }
    }

    /// Creates a new parse error.
    #[must_use]
    pub fn parse(offset: u64, message: impl Into<String>) -> Self {
        Self::Parse {
            offset,
            message: message.into(),
        }
    }

    /// Creates a new invalid state error.
    #[must_use]
    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::InvalidState(message.into())
    }

    /// Creates a new handshake error.
    #[must_use]
    pub fn handshake(message: impl Into<String>) -> Self {
        Self::Handshake(message.into())
    }

    /// Creates a new authentication error.
    #[must_use]
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::Authentication(message.into())
    }

    /// Creates a new not found error.
    #[must_use]
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::NotFound(message.into())
    }

    /// Creates a new segment error.
    #[must_use]
    pub fn segment(message: impl Into<String>) -> Self {
        Self::Segment(message.into())
    }

    /// Creates a new playlist error.
    #[must_use]
    pub fn playlist(message: impl Into<String>) -> Self {
        Self::Playlist(message.into())
    }

    /// Creates a new encoding error.
    #[must_use]
    pub fn encoding(message: impl Into<String>) -> Self {
        Self::Encoding(message.into())
    }

    /// Creates a new buffer error.
    #[must_use]
    pub fn buffer(message: impl Into<String>) -> Self {
        Self::Buffer(message.into())
    }

    /// Returns true if this is an end-of-stream error.
    #[must_use]
    pub const fn is_eof(&self) -> bool {
        matches!(self, Self::Eof)
    }

    /// Returns true if this is a timeout error.
    #[must_use]
    pub const fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout(_))
    }

    /// Returns true if this is a connection error.
    #[must_use]
    pub const fn is_connection(&self) -> bool {
        matches!(self, Self::Connection(_))
    }
}

/// Result type alias for network streaming operations.
pub type NetResult<T> = Result<T, NetError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_error() {
        let err = NetError::connection("Failed to connect");
        assert!(err.is_connection());
        assert!(format!("{err}").contains("Failed to connect"));
    }

    #[test]
    fn test_protocol_error() {
        let err = NetError::protocol("Invalid message format");
        assert!(format!("{err}").contains("Invalid message format"));
    }

    #[test]
    fn test_timeout_error() {
        let err = NetError::timeout("Connection timed out");
        assert!(err.is_timeout());
        assert!(format!("{err}").contains("Connection timed out"));
    }

    #[test]
    fn test_http_error() {
        let err = NetError::http(404, "Not Found");
        assert!(format!("{err}").contains("404"));
        assert!(format!("{err}").contains("Not Found"));
    }

    #[test]
    fn test_parse_error() {
        let err = NetError::parse(100, "Invalid header");
        if let NetError::Parse { offset, message } = err {
            assert_eq!(offset, 100);
            assert_eq!(message, "Invalid header");
        } else {
            panic!("Expected Parse error");
        }
    }

    #[test]
    fn test_eof_error() {
        let err = NetError::Eof;
        assert!(err.is_eof());
    }

    #[test]
    fn test_io_error_from() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: NetError = io_err.into();
        assert!(matches!(err, NetError::Io(_)));
    }
}

//! Error types for `OxiMedia`.
//!
//! This module provides the [`OxiError`] type which represents all errors
//! that can occur during multimedia processing, and the [`OxiResult`] type
//! alias for convenient use.
//!
//! # Patent Protection
//!
//! The [`PatentViolation`](OxiError::PatentViolation) error is returned when
//! attempting to use patent-encumbered codecs (H.264, H.265, AAC, etc.).
//! `OxiMedia` only supports Green List codecs.

use std::io;

/// Error type for `OxiMedia` operations.
///
/// This enum covers all possible errors that can occur during multimedia
/// processing, including I/O errors, parsing errors, codec errors, and
/// patent violations.
///
/// # Examples
///
/// ```
/// use oximedia_core::error::{OxiError, OxiResult};
///
/// fn parse_data(data: &[u8]) -> OxiResult<()> {
///     if data.is_empty() {
///         return Err(OxiError::Parse {
///             offset: 0,
///             message: "Empty data".to_string(),
///         });
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, thiserror::Error)]
pub enum OxiError {
    /// I/O error during file or stream operations.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Parse error at a specific offset in the data.
    #[error("Parse error at offset {offset}: {message}")]
    Parse {
        /// Byte offset where the error occurred.
        offset: u64,
        /// Description of the parse error.
        message: String,
    },

    /// Codec-related error.
    #[error("Codec error: {0}")]
    Codec(String),

    /// Unsupported format or feature.
    #[error("Unsupported format: {0}")]
    Unsupported(String),

    /// Attempted to use a patent-encumbered codec.
    ///
    /// `OxiMedia` only supports patent-free (Green List) codecs.
    /// This error is returned when attempting to use codecs like
    /// H.264, H.265, AAC, etc.
    #[error("Patent-encumbered codec detected: {0}")]
    PatentViolation(String),

    /// End of stream reached.
    #[error("End of stream")]
    Eof,

    /// Buffer is too small for the requested operation.
    #[error("Buffer too small: need {needed}, have {have}")]
    BufferTooSmall {
        /// Required buffer size in bytes.
        needed: usize,
        /// Available buffer size in bytes.
        have: usize,
    },

    /// Unexpected end of file during read operation.
    ///
    /// This is returned when attempting to read beyond the end of available data.
    #[error("Unexpected end of file")]
    UnexpectedEof,

    /// Invalid data encountered during parsing.
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Format could not be recognized.
    #[error("Unknown format")]
    UnknownFormat,
}

impl OxiError {
    /// Creates a new parse error at the given offset.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::error::OxiError;
    ///
    /// let err = OxiError::parse(42, "Invalid header");
    /// assert!(matches!(err, OxiError::Parse { offset: 42, .. }));
    /// ```
    #[must_use]
    pub fn parse(offset: u64, message: impl Into<String>) -> Self {
        Self::Parse {
            offset,
            message: message.into(),
        }
    }

    /// Creates a new codec error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::error::OxiError;
    ///
    /// let err = OxiError::codec("Invalid frame data");
    /// assert!(matches!(err, OxiError::Codec(_)));
    /// ```
    #[must_use]
    pub fn codec(message: impl Into<String>) -> Self {
        Self::Codec(message.into())
    }

    /// Creates a new unsupported format error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::error::OxiError;
    ///
    /// let err = OxiError::unsupported("H.265 is not supported");
    /// assert!(matches!(err, OxiError::Unsupported(_)));
    /// ```
    #[must_use]
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    /// Creates a new patent violation error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::error::OxiError;
    ///
    /// let err = OxiError::patent_violation("H.264");
    /// assert!(matches!(err, OxiError::PatentViolation(_)));
    /// ```
    #[must_use]
    pub fn patent_violation(codec_name: impl Into<String>) -> Self {
        Self::PatentViolation(codec_name.into())
    }

    /// Creates a buffer too small error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::error::OxiError;
    ///
    /// let err = OxiError::buffer_too_small(1024, 512);
    /// assert!(matches!(err, OxiError::BufferTooSmall { needed: 1024, have: 512 }));
    /// ```
    #[must_use]
    pub fn buffer_too_small(needed: usize, have: usize) -> Self {
        Self::BufferTooSmall { needed, have }
    }

    /// Returns true if this is an end-of-stream error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::error::OxiError;
    ///
    /// assert!(OxiError::Eof.is_eof());
    /// assert!(!OxiError::codec("test").is_eof());
    /// ```
    #[must_use]
    pub const fn is_eof(&self) -> bool {
        matches!(self, Self::Eof)
    }

    /// Returns true if this is a patent violation error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::error::OxiError;
    ///
    /// assert!(OxiError::patent_violation("H.264").is_patent_violation());
    /// assert!(!OxiError::Eof.is_patent_violation());
    /// ```
    #[must_use]
    pub const fn is_patent_violation(&self) -> bool {
        matches!(self, Self::PatentViolation(_))
    }

    /// Creates an invalid data error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::error::OxiError;
    ///
    /// let err = OxiError::invalid_data("Malformed header");
    /// assert!(matches!(err, OxiError::InvalidData(_)));
    /// ```
    #[must_use]
    pub fn invalid_data(message: impl Into<String>) -> Self {
        Self::InvalidData(message.into())
    }

    /// Returns true if this is an unexpected EOF error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::error::OxiError;
    ///
    /// assert!(OxiError::UnexpectedEof.is_unexpected_eof());
    /// assert!(!OxiError::Eof.is_unexpected_eof());
    /// ```
    #[must_use]
    pub const fn is_unexpected_eof(&self) -> bool {
        matches!(self, Self::UnexpectedEof)
    }
}

/// Result type alias for `OxiMedia` operations.
///
/// This is a convenience alias for `Result<T, OxiError>`.
///
/// # Examples
///
/// ```
/// use oximedia_core::error::OxiResult;
///
/// fn process() -> OxiResult<u32> {
///     Ok(42)
/// }
/// ```
pub type OxiResult<T> = Result<T, OxiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error() {
        let err = OxiError::parse(100, "Invalid magic bytes");
        assert!(matches!(err, OxiError::Parse { offset: 100, .. }));
        let msg = format!("{err}");
        assert!(msg.contains("100"));
        assert!(msg.contains("Invalid magic bytes"));
    }

    #[test]
    fn test_codec_error() {
        let err = OxiError::codec("Frame decode failed");
        assert!(matches!(err, OxiError::Codec(_)));
        assert!(format!("{err}").contains("Frame decode failed"));
    }

    #[test]
    fn test_unsupported_error() {
        let err = OxiError::unsupported("H.265 codec");
        assert!(format!("{err}").contains("H.265 codec"));
    }

    #[test]
    fn test_patent_violation() {
        let err = OxiError::patent_violation("H.264");
        assert!(err.is_patent_violation());
        assert!(format!("{err}").contains("H.264"));
    }

    #[test]
    fn test_buffer_too_small() {
        let err = OxiError::buffer_too_small(1024, 512);
        assert!(format!("{err}").contains("1024"));
        assert!(format!("{err}").contains("512"));
    }

    #[test]
    fn test_eof() {
        let err = OxiError::Eof;
        assert!(err.is_eof());
        assert!(!OxiError::codec("test").is_eof());
    }

    #[test]
    fn test_io_error_from() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: OxiError = io_err.into();
        assert!(matches!(err, OxiError::Io(_)));
    }
}

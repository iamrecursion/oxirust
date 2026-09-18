#![allow(dead_code)]
//! # Python Error Bindings
//!
//! Typed error codes and conversion utilities for OxiMedia's Python surface.
//! Provides a structured [`PyError`] type, a flat [`ErrorCode`] enum, and a
//! [`ErrorConverter`] that bridges Rust errors to Python-compatible strings.

/// Classified error codes surfaced to Python callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// No error; operation succeeded.
    Ok,
    /// An input argument was invalid.
    InvalidArgument,
    /// The requested resource was not found.
    NotFound,
    /// The operation is not supported in this configuration.
    Unsupported,
    /// Codec decode or encode error.
    CodecError,
    /// Container mux / demux error.
    ContainerError,
    /// I/O or network error.
    IoError,
    /// Out-of-memory or allocation failure.
    OutOfMemory,
    /// Timeout waiting for a resource or event.
    Timeout,
    /// Concurrent access conflict.
    Concurrency,
    /// Internal logic error (should not occur).
    Internal,
}

impl ErrorCode {
    /// Return the numeric code sent across the Python boundary.
    pub fn code(&self) -> i32 {
        match self {
            ErrorCode::Ok => 0,
            ErrorCode::InvalidArgument => 1,
            ErrorCode::NotFound => 2,
            ErrorCode::Unsupported => 3,
            ErrorCode::CodecError => 4,
            ErrorCode::ContainerError => 5,
            ErrorCode::IoError => 6,
            ErrorCode::OutOfMemory => 7,
            ErrorCode::Timeout => 8,
            ErrorCode::Concurrency => 9,
            ErrorCode::Internal => 10,
        }
    }

    /// Reconstruct an `ErrorCode` from a numeric code.
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => ErrorCode::Ok,
            1 => ErrorCode::InvalidArgument,
            2 => ErrorCode::NotFound,
            3 => ErrorCode::Unsupported,
            4 => ErrorCode::CodecError,
            5 => ErrorCode::ContainerError,
            6 => ErrorCode::IoError,
            7 => ErrorCode::OutOfMemory,
            8 => ErrorCode::Timeout,
            9 => ErrorCode::Concurrency,
            _ => ErrorCode::Internal,
        }
    }

    /// Return a short machine-readable name (used as Python exception name).
    pub fn name(&self) -> &'static str {
        match self {
            ErrorCode::Ok => "Ok",
            ErrorCode::InvalidArgument => "InvalidArgument",
            ErrorCode::NotFound => "NotFound",
            ErrorCode::Unsupported => "Unsupported",
            ErrorCode::CodecError => "CodecError",
            ErrorCode::ContainerError => "ContainerError",
            ErrorCode::IoError => "IoError",
            ErrorCode::OutOfMemory => "OutOfMemory",
            ErrorCode::Timeout => "Timeout",
            ErrorCode::Concurrency => "Concurrency",
            ErrorCode::Internal => "Internal",
        }
    }

    /// Return `true` if this code represents a successful outcome.
    pub fn is_ok(&self) -> bool {
        *self == ErrorCode::Ok
    }

    /// Return `true` if this code represents a recoverable error.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            ErrorCode::Timeout | ErrorCode::Concurrency | ErrorCode::NotFound
        )
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Severity level of a [`PyError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational — not truly an error.
    Info,
    /// Warning — operation completed with degradation.
    Warning,
    /// Error — operation failed but process continues.
    Error,
    /// Fatal — process cannot continue safely.
    Fatal,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Severity::Info => "INFO",
            Severity::Warning => "WARNING",
            Severity::Error => "ERROR",
            Severity::Fatal => "FATAL",
        };
        write!(f, "{s}")
    }
}

/// A structured error suitable for surfacing through Python bindings.
#[derive(Debug, Clone)]
pub struct PyError {
    /// Classified error code.
    pub code: ErrorCode,
    /// Severity level.
    pub severity: Severity,
    /// Human-readable description.
    pub message: String,
    /// Optional underlying cause chain.
    pub cause: Option<Box<PyError>>,
}

impl PyError {
    /// Create a new error with `Error` severity.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            severity: Severity::Error,
            message: message.into(),
            cause: None,
        }
    }

    /// Create a fatal error.
    pub fn fatal(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            severity: Severity::Fatal,
            message: message.into(),
            cause: None,
        }
    }

    /// Attach a cause to this error.
    pub fn with_cause(mut self, cause: PyError) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }

    /// Return `true` if this is a fatal error.
    pub fn is_fatal(&self) -> bool {
        self.severity == Severity::Fatal
    }

    /// Return `true` if the underlying code is recoverable.
    pub fn is_recoverable(&self) -> bool {
        self.code.is_recoverable()
    }

    /// Render the full error chain as a string (most useful in tracebacks).
    pub fn to_string(&self) -> String {
        let mut s = format!("[{}] {}: {}", self.severity, self.code, self.message);
        if let Some(cause) = &self.cause {
            s.push_str(&format!(" caused by: {}", cause.to_string()));
        }
        s
    }

    /// Return the numeric error code.
    pub fn numeric_code(&self) -> i32 {
        self.code.code()
    }
}

impl std::fmt::Display for PyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// Converts between [`PyError`] and flat representations suitable for Python.
pub struct ErrorConverter;

impl ErrorConverter {
    /// Render a `PyError` to the format Python exceptions expect:
    /// `"ERROR_NAME: message"`.
    pub fn to_python_str(err: &PyError) -> String {
        format!("{}: {}", err.code.name(), err.message)
    }

    /// Build a `PyError` from a simple numeric code and message string.
    pub fn from_parts(code: i32, message: impl Into<String>) -> PyError {
        PyError::new(ErrorCode::from_code(code), message)
    }

    /// Map a standard `std::io::Error` to a `PyError`.
    pub fn from_io(err: std::io::Error) -> PyError {
        PyError::new(ErrorCode::IoError, err.to_string())
    }

    /// Wrap an arbitrary string as an `Internal` error.
    pub fn internal(msg: impl Into<String>) -> PyError {
        PyError::new(ErrorCode::Internal, msg)
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_numeric() {
        assert_eq!(ErrorCode::Ok.code(), 0);
        assert_eq!(ErrorCode::Internal.code(), 10);
    }

    #[test]
    fn test_error_code_from_code_roundtrip() {
        for i in 0..=10_i32 {
            let ec = ErrorCode::from_code(i);
            assert_eq!(ec.code(), i);
        }
    }

    #[test]
    fn test_error_code_from_code_unknown() {
        assert_eq!(ErrorCode::from_code(999), ErrorCode::Internal);
    }

    #[test]
    fn test_error_code_name() {
        assert_eq!(ErrorCode::CodecError.name(), "CodecError");
        assert_eq!(ErrorCode::Ok.name(), "Ok");
    }

    #[test]
    fn test_error_code_is_ok() {
        assert!(ErrorCode::Ok.is_ok());
        assert!(!ErrorCode::IoError.is_ok());
    }

    #[test]
    fn test_error_code_is_recoverable() {
        assert!(ErrorCode::Timeout.is_recoverable());
        assert!(ErrorCode::Concurrency.is_recoverable());
        assert!(!ErrorCode::Internal.is_recoverable());
        assert!(!ErrorCode::OutOfMemory.is_recoverable());
    }

    #[test]
    fn test_error_code_display() {
        assert_eq!(format!("{}", ErrorCode::NotFound), "NotFound");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Fatal);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Fatal), "FATAL");
    }

    #[test]
    fn test_py_error_new() {
        let e = PyError::new(ErrorCode::NotFound, "file missing");
        assert_eq!(e.code, ErrorCode::NotFound);
        assert_eq!(e.severity, Severity::Error);
        assert!(!e.is_fatal());
    }

    #[test]
    fn test_py_error_fatal() {
        let e = PyError::fatal(ErrorCode::OutOfMemory, "OOM");
        assert!(e.is_fatal());
    }

    #[test]
    fn test_py_error_to_string() {
        let e = PyError::new(ErrorCode::CodecError, "decode failed");
        let s = e.to_string();
        assert!(s.contains("CodecError"));
        assert!(s.contains("decode failed"));
    }

    #[test]
    fn test_py_error_with_cause() {
        let cause = PyError::new(ErrorCode::IoError, "read error");
        let e = PyError::new(ErrorCode::CodecError, "codec").with_cause(cause);
        let s = e.to_string();
        assert!(s.contains("caused by"));
        assert!(s.contains("read error"));
    }

    #[test]
    fn test_py_error_numeric_code() {
        let e = PyError::new(ErrorCode::Timeout, "timed out");
        assert_eq!(e.numeric_code(), 8);
    }

    #[test]
    fn test_converter_to_python_str() {
        let e = PyError::new(ErrorCode::Unsupported, "no GPU");
        let s = ErrorConverter::to_python_str(&e);
        assert_eq!(s, "Unsupported: no GPU");
    }

    #[test]
    fn test_converter_from_parts() {
        let e = ErrorConverter::from_parts(4, "bad packet");
        assert_eq!(e.code, ErrorCode::CodecError);
    }

    #[test]
    fn test_converter_internal() {
        let e = ErrorConverter::internal("unreachable branch");
        assert_eq!(e.code, ErrorCode::Internal);
    }
}

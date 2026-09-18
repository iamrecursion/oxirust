//! Structured exit codes for the OxiMedia CLI.
//!
//! Maps runtime errors to well-known exit code values so callers can
//! distinguish IO failures from usage mistakes in scripts.

use std::process::ExitCode;

/// Exit codes produced by the OxiMedia CLI binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum OxiExitCode {
    /// Command succeeded.
    Ok = 0,
    /// Unclassified runtime error.
    GenericError = 1,
    /// Bad command-line usage (clap error).
    UsageError = 2,
    /// IO / file-system error.
    IoError = 3,
    /// Input validation error.
    ValidationError = 4,
}

impl From<OxiExitCode> for ExitCode {
    fn from(code: OxiExitCode) -> ExitCode {
        ExitCode::from(code as u8)
    }
}

/// Inspect `err` and return the most specific exit code.
pub(crate) fn classify_error(err: &anyhow::Error) -> OxiExitCode {
    for cause in err.chain() {
        if cause.downcast_ref::<std::io::Error>().is_some() {
            return OxiExitCode::IoError;
        }
        if cause.downcast_ref::<clap::Error>().is_some() {
            return OxiExitCode::UsageError;
        }
        let msg = cause.to_string();
        if msg.contains("validation error:") || msg.contains("ValidationError:") {
            return OxiExitCode::ValidationError;
        }
    }
    OxiExitCode::GenericError
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let anyhow_err = anyhow::Error::from(io_err);
        assert_eq!(classify_error(&anyhow_err), OxiExitCode::IoError);
    }

    #[test]
    fn test_classify_generic_error() {
        let err = anyhow::anyhow!("something went wrong");
        assert_eq!(classify_error(&err), OxiExitCode::GenericError);
    }

    #[test]
    fn test_classify_validation_error() {
        let err = anyhow::anyhow!("validation error: field X is invalid");
        assert_eq!(classify_error(&err), OxiExitCode::ValidationError);
    }

    #[test]
    fn test_exit_code_values() {
        assert_eq!(OxiExitCode::Ok as u8, 0);
        assert_eq!(OxiExitCode::GenericError as u8, 1);
        assert_eq!(OxiExitCode::UsageError as u8, 2);
        assert_eq!(OxiExitCode::IoError as u8, 3);
        assert_eq!(OxiExitCode::ValidationError as u8, 4);
    }
}

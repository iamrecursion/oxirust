//! [`ErrorHandler`], reporting parse warnings and errors with position
//! information.

use crate::error::SaxError;
use crate::locator::SaxParseException;

/// Receives notification of parse warnings and errors.
///
/// Mirrors `org.xml.sax.ErrorHandler`. If no [`ErrorHandler`] is installed,
/// [`XMLReader::parse`](crate::XMLReader::parse) still stops on a fatal error and
/// returns `Err`, it simply skips the callback.
///
/// # `error` versus `fatal_error`
///
/// XML 1.0 §1.2 makes *every* well-formedness violation a fatal error, and says
/// a processor "MUST NOT continue normal processing" past one, so
/// [`XMLReader`](crate::XMLReader) reports all of them — a syntax error, a
/// mismatched or missing end tag, a duplicated attribute, character data outside
/// the document element, an I/O error, an encoding error — through
/// [`fatal_error`](Self::fatal_error). Parsing always stops immediately after
/// that call, and [`XMLReader::parse`](crate::XMLReader::parse) returns `Err`.
///
/// [`error`](Self::error) and [`warning`](Self::warning) are never called by
/// [`XMLReader`](crate::XMLReader) itself (this crate does not perform DTD
/// validation, the usual source of recoverable SAX2 errors and warnings), but
/// are available for handlers layered on top.
pub trait ErrorHandler {
    /// Called for a non-fatal condition that does not affect the ability to keep
    /// parsing. Never called by [`XMLReader`](crate::XMLReader) itself; see the
    /// trait-level documentation.
    fn warning(&mut self, _exception: &SaxParseException) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called for a recoverable problem; parsing continues afterwards. Never
    /// called by [`XMLReader`](crate::XMLReader) itself; see the trait-level
    /// documentation.
    fn error(&mut self, _exception: &SaxParseException) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called for an unrecoverable problem; parsing stops immediately afterwards.
    fn fatal_error(&mut self, _exception: &SaxParseException) -> Result<(), SaxError> {
        Ok(())
    }
}

/// An [`ErrorHandler`] that ignores every event (the default when none is
/// installed).
#[derive(Debug, Clone, Copy, Default)]
pub struct NullErrorHandler;

impl ErrorHandler for NullErrorHandler {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::locator::Locator;

    #[test]
    fn null_handler_accepts_every_call() {
        let mut handler = NullErrorHandler;
        let exception = SaxParseException::new("boom".to_string(), Locator::start());
        assert!(handler.warning(&exception).is_ok());
        assert!(handler.error(&exception).is_ok());
        assert!(handler.fatal_error(&exception).is_ok());
    }
}

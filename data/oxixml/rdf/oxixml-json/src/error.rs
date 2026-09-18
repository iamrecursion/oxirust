//! Parse error types: [`JsonSyntaxError`] and [`JsonParseError`].

use oxixml_support::pos::TextPosition;
use std::error::Error;
use std::fmt;
use std::io;
use std::ops::Range;

/// An error in the syntax of a parsed JSON document.
///
/// It carries a human-readable message plus the half-open [`TextPosition`] range in the input
/// that the error was detected at, so callers can produce editor-quality diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonSyntaxError {
    location: Range<TextPosition>,
    message: String,
}

impl JsonSyntaxError {
    /// Builds a new syntax error at `location` with the given `message`.
    #[inline]
    #[must_use]
    pub(crate) fn new(location: Range<TextPosition>, message: impl Into<String>) -> Self {
        Self {
            location,
            message: message.into(),
        }
    }

    /// The location of the error inside of the document: a half-open range of [`TextPosition`]s.
    #[inline]
    #[must_use]
    pub fn location(&self) -> Range<TextPosition> {
        self.location.clone()
    }

    /// The human-readable error message. Not guaranteed to be stable across releases.
    #[inline]
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for JsonSyntaxError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `TextPosition` stores 0-based line/column; messages use the human-facing 1-based form.
        let start = self.location.start;
        let end = self.location.end;
        if start.offset + 1 >= end.offset {
            write!(
                f,
                "Parser error at line {} column {}: {}",
                start.line + 1,
                start.column + 1,
                self.message
            )
        } else if start.line == end.line {
            write!(
                f,
                "Parser error at line {} between columns {} and column {}: {}",
                start.line + 1,
                start.column + 1,
                end.column + 1,
                self.message
            )
        } else {
            write!(
                f,
                "Parser error between line {} column {} and line {} column {}: {}",
                start.line + 1,
                start.column + 1,
                end.line + 1,
                end.column + 1,
                self.message
            )
        }
    }
}

impl Error for JsonSyntaxError {}

impl From<JsonSyntaxError> for io::Error {
    #[inline]
    fn from(error: JsonSyntaxError) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, error)
    }
}

/// An error while parsing a JSON document from a [`std::io::Read`] source.
///
/// It is the union of a [`JsonSyntaxError`] (malformed JSON) and a [`std::io::Error`] (the
/// underlying byte source failed).
#[derive(Debug)]
pub enum JsonParseError {
    /// An I/O error while reading from the underlying source (file not found, broken pipe, ...).
    Io(io::Error),
    /// An error in the syntax of the document itself.
    Syntax(JsonSyntaxError),
}

impl fmt::Display for JsonParseError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Syntax(e) => e.fmt(f),
        }
    }
}

impl Error for JsonParseError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(match self {
            Self::Io(e) => e,
            Self::Syntax(e) => e,
        })
    }
}

impl From<JsonSyntaxError> for JsonParseError {
    #[inline]
    fn from(error: JsonSyntaxError) -> Self {
        Self::Syntax(error)
    }
}

impl From<io::Error> for JsonParseError {
    #[inline]
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<JsonParseError> for io::Error {
    #[inline]
    fn from(error: JsonParseError) -> Self {
        match error {
            JsonParseError::Syntax(e) => e.into(),
            JsonParseError::Io(e) => e,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a 0-based [`TextPosition`] (see [`oxixml_support::pos`] for the numbering base).
    fn pos(line: u64, column: u64, offset: u64) -> TextPosition {
        TextPosition::new(line, column, offset)
    }

    #[test]
    fn display_single_point() -> Result<(), Box<dyn Error>> {
        let err = JsonSyntaxError::new(pos(0, 0, 0)..pos(0, 1, 1), "boom");
        assert_eq!(err.to_string(), "Parser error at line 1 column 1: boom");
        assert_eq!(err.message(), "boom");
        // Storage stays 0-based even though the rendering above is 1-based.
        assert_eq!(err.location(), pos(0, 0, 0)..pos(0, 1, 1));
        assert_eq!(err.location().start.line, 0);
        assert_eq!(err.location().start.column, 0);
        Ok(())
    }

    #[test]
    fn display_same_line_range() {
        let err = JsonSyntaxError::new(pos(1, 2, 10)..pos(1, 5, 13), "bad token");
        assert_eq!(
            err.to_string(),
            "Parser error at line 2 between columns 3 and column 6: bad token"
        );
    }

    #[test]
    fn display_multi_line_range() {
        let err = JsonSyntaxError::new(pos(0, 2, 2)..pos(2, 0, 20), "unterminated");
        assert_eq!(
            err.to_string(),
            "Parser error between line 1 column 3 and line 3 column 1: unterminated"
        );
    }

    #[test]
    fn conversions_round_trip() {
        let syntax = JsonSyntaxError::new(pos(0, 0, 0)..pos(0, 0, 0), "e");
        let io_err: io::Error = syntax.clone().into();
        assert_eq!(io_err.kind(), io::ErrorKind::InvalidData);

        let parse_err: JsonParseError = syntax.into();
        assert!(matches!(parse_err, JsonParseError::Syntax(_)));
        assert!(parse_err.source().is_some());

        let io_err = io::Error::new(io::ErrorKind::UnexpectedEof, "eof");
        let parse_err: JsonParseError = io_err.into();
        assert!(matches!(parse_err, JsonParseError::Io(_)));
        let _: io::Error = parse_err.into();
    }
}

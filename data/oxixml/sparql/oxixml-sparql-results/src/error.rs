//! Error types returned while parsing a SPARQL query results serialization.

use oxixml_model::TermParseError;
use std::fmt;
use std::io;
use std::ops::Range;
use std::sync::Arc;

/// A position in a text, re-exported from [`oxixml_support::pos`] so that callers of
/// [`QueryResultsSyntaxError::location`] never need to depend on `oxixml-support` directly.
///
/// It is a `line` number, a `column` number (in Unicode scalar values), and a byte `offset`,
/// **all 0-based**: the first character of a document is line `0`, column `0`, offset `0`. This
/// matches `sparesults`' numbering. Only [`Display`](std::fmt::Display) renders the 1-based form.
pub use oxixml_support::pos::TextPosition;

/// Error returned during SPARQL result formats parsing.
#[derive(Debug)]
pub enum QueryResultsParseError {
    /// I/O error during parsing (file not found...).
    Io(io::Error),
    /// An error in the file syntax.
    Syntax(QueryResultsSyntaxError),
}

impl fmt::Display for QueryResultsParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Syntax(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for QueryResultsParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Io(e) => e,
            Self::Syntax(e) => e,
        })
    }
}

impl From<io::Error> for QueryResultsParseError {
    #[inline]
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<QueryResultsSyntaxError> for QueryResultsParseError {
    #[inline]
    fn from(error: QueryResultsSyntaxError) -> Self {
        Self::Syntax(error)
    }
}

impl From<QueryResultsParseError> for io::Error {
    #[inline]
    fn from(error: QueryResultsParseError) -> Self {
        match error {
            QueryResultsParseError::Io(error) => error,
            QueryResultsParseError::Syntax(error) => error.into(),
        }
    }
}

#[doc(hidden)]
impl From<oxixml_json::JsonParseError> for QueryResultsParseError {
    fn from(error: oxixml_json::JsonParseError) -> Self {
        match error {
            oxixml_json::JsonParseError::Syntax(error) => {
                QueryResultsSyntaxError::from(error).into()
            }
            oxixml_json::JsonParseError::Io(error) => error.into(),
        }
    }
}

#[doc(hidden)]
impl From<oxixml_xml::errors::Error> for QueryResultsParseError {
    #[inline]
    fn from(error: oxixml_xml::errors::Error) -> Self {
        match error {
            oxixml_xml::errors::Error::Io(error) => {
                Self::Io(Arc::try_unwrap(error).unwrap_or_else(|e| io::Error::new(e.kind(), e)))
            }
            other => Self::Syntax(QueryResultsSyntaxError(SyntaxErrorKind::Xml(other))),
        }
    }
}

#[doc(hidden)]
impl From<oxixml_xml::escape::EscapeError> for QueryResultsParseError {
    #[inline]
    fn from(error: oxixml_xml::escape::EscapeError) -> Self {
        oxixml_xml::errors::Error::from(error).into()
    }
}

#[doc(hidden)]
impl From<oxixml_xml::encoding::EncodingError> for QueryResultsParseError {
    #[inline]
    fn from(error: oxixml_xml::encoding::EncodingError) -> Self {
        oxixml_xml::errors::Error::from(error).into()
    }
}

/// An error in the syntax of the parsed file.
#[derive(Debug)]
pub struct QueryResultsSyntaxError(SyntaxErrorKind);

#[derive(Debug)]
enum SyntaxErrorKind {
    Json(oxixml_json::JsonSyntaxError),
    Xml(oxixml_xml::errors::Error),
    Term {
        error: TermParseError,
        term: String,
        location: Range<TextPosition>,
    },
    Msg {
        msg: String,
        location: Option<Range<TextPosition>>,
    },
}

impl QueryResultsSyntaxError {
    /// Builds an error from a printable error message.
    pub(crate) fn msg(msg: impl Into<String>) -> Self {
        Self(SyntaxErrorKind::Msg {
            msg: msg.into(),
            location: None,
        })
    }

    /// Builds an error from a [`TermParseError`] encountered while parsing the given `term`
    /// serialization at `location`.
    pub(crate) fn term(error: TermParseError, term: String, location: Range<TextPosition>) -> Self {
        Self(SyntaxErrorKind::Term {
            error,
            term,
            location,
        })
    }

    /// Builds an error from a printable error message and a location.
    #[inline]
    pub(crate) fn located_message(msg: impl Into<String>, location: Range<TextPosition>) -> Self {
        Self(SyntaxErrorKind::Msg {
            msg: msg.into(),
            location: Some(location),
        })
    }

    /// The location of the error inside of the file.
    #[inline]
    #[must_use]
    pub fn location(&self) -> Option<Range<TextPosition>> {
        match &self.0 {
            SyntaxErrorKind::Json(e) => Some(e.location()),
            SyntaxErrorKind::Term { location, .. } => Some(location.clone()),
            SyntaxErrorKind::Msg { location, .. } => location.clone(),
            SyntaxErrorKind::Xml(_) => None,
        }
    }
}

impl fmt::Display for QueryResultsSyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            SyntaxErrorKind::Json(e) => e.fmt(f),
            SyntaxErrorKind::Xml(e) => e.fmt(f),
            SyntaxErrorKind::Term {
                error,
                term,
                location,
            } => write!(
                f,
                "Error {error} on '{term}' in line {}",
                location.start.line + 1
            ),
            SyntaxErrorKind::Msg { msg, .. } => f.write_str(msg),
        }
    }
}

impl std::error::Error for QueryResultsSyntaxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0 {
            SyntaxErrorKind::Json(e) => Some(e),
            SyntaxErrorKind::Xml(e) => Some(e),
            SyntaxErrorKind::Term { error, .. } => Some(error),
            SyntaxErrorKind::Msg { .. } => None,
        }
    }
}

impl From<QueryResultsSyntaxError> for io::Error {
    #[inline]
    fn from(error: QueryResultsSyntaxError) -> Self {
        match error.0 {
            SyntaxErrorKind::Json(json_error) => Self::new(io::ErrorKind::InvalidData, json_error),
            SyntaxErrorKind::Xml(xml_error) => match xml_error {
                oxixml_xml::errors::Error::Io(io_error) => {
                    Arc::try_unwrap(io_error).unwrap_or_else(|e| Self::new(e.kind(), e))
                }
                other => Self::new(io::ErrorKind::InvalidData, other),
            },
            SyntaxErrorKind::Term { error, term, .. } => Self::new(
                io::ErrorKind::InvalidData,
                format!("Error {error} on '{term}'"),
            ),
            SyntaxErrorKind::Msg { msg, .. } => Self::new(io::ErrorKind::InvalidData, msg),
        }
    }
}

#[doc(hidden)]
impl From<oxixml_json::JsonSyntaxError> for QueryResultsSyntaxError {
    fn from(error: oxixml_json::JsonSyntaxError) -> Self {
        Self(SyntaxErrorKind::Json(error))
    }
}

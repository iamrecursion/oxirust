//! Error types returned while parsing [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) documents.

use std::fmt;
use std::io;

use oxixml_model::{IriParseError, LanguageTagParseError};

/// An error in the syntax (or semantics) of a parsed TriX document.
///
/// It carries a byte offset into the input, pointing at the XML markup that triggered the
/// error, together with a human-readable message.
#[derive(Debug)]
pub struct TriXSyntaxError {
    position: u64,
    kind: TriXSyntaxErrorKind,
}

#[derive(Debug)]
enum TriXSyntaxErrorKind {
    /// The input is not well-formed XML (a mismatched tag, an unclosed comment...).
    Xml(String),
    /// A `<uri>` (or `datatype` attribute) content is not a valid IRI.
    InvalidIri { iri: String, error: IriParseError },
    /// A `plainLiteral` `xml:lang` attribute is not a valid BCP47 language tag.
    InvalidLanguageTag {
        tag: String,
        error: LanguageTagParseError,
    },
    /// A free-form structural or semantic error, described by `message`.
    Message(String),
}

impl TriXSyntaxError {
    pub(crate) fn xml(position: u64, message: impl Into<String>) -> Self {
        Self {
            position,
            kind: TriXSyntaxErrorKind::Xml(message.into()),
        }
    }

    pub(crate) fn invalid_iri(position: u64, iri: impl Into<String>, error: IriParseError) -> Self {
        Self {
            position,
            kind: TriXSyntaxErrorKind::InvalidIri {
                iri: iri.into(),
                error,
            },
        }
    }

    pub(crate) fn invalid_language_tag(
        position: u64,
        tag: impl Into<String>,
        error: LanguageTagParseError,
    ) -> Self {
        Self {
            position,
            kind: TriXSyntaxErrorKind::InvalidLanguageTag {
                tag: tag.into(),
                error,
            },
        }
    }

    pub(crate) fn msg(position: u64, message: impl Into<String>) -> Self {
        Self {
            position,
            kind: TriXSyntaxErrorKind::Message(message.into()),
        }
    }

    /// The byte offset into the input at which the error was detected.
    #[inline]
    #[must_use]
    pub fn position(&self) -> u64 {
        self.position
    }

    /// The human-readable error message, without the position prefix.
    #[must_use]
    pub fn message(&self) -> String {
        match &self.kind {
            TriXSyntaxErrorKind::Xml(msg) | TriXSyntaxErrorKind::Message(msg) => msg.clone(),
            TriXSyntaxErrorKind::InvalidIri { iri, error } => {
                format!("invalid IRI '{iri}': {error}")
            }
            TriXSyntaxErrorKind::InvalidLanguageTag { tag, error } => {
                format!("invalid language tag '{tag}': {error}")
            }
        }
    }
}

impl fmt::Display for TriXSyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.position, self.message())
    }
}

impl std::error::Error for TriXSyntaxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            TriXSyntaxErrorKind::InvalidIri { error, .. } => Some(error),
            TriXSyntaxErrorKind::InvalidLanguageTag { error, .. } => Some(error),
            TriXSyntaxErrorKind::Xml(_) | TriXSyntaxErrorKind::Message(_) => None,
        }
    }
}

impl From<TriXSyntaxError> for io::Error {
    #[inline]
    fn from(error: TriXSyntaxError) -> Self {
        Self::new(io::ErrorKind::InvalidData, error)
    }
}

/// An error that can occur while parsing a TriX document: either an I/O failure, or a
/// [`TriXSyntaxError`].
#[derive(Debug)]
pub enum TriXParseError {
    /// An I/O error occurred while reading the underlying stream.
    Io(io::Error),
    /// The document is not a well-formed and valid TriX document.
    Syntax(TriXSyntaxError),
}

impl fmt::Display for TriXParseError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Syntax(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for TriXParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Syntax(e) => Some(e),
        }
    }
}

impl From<io::Error> for TriXParseError {
    #[inline]
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<TriXSyntaxError> for TriXParseError {
    #[inline]
    fn from(error: TriXSyntaxError) -> Self {
        Self::Syntax(error)
    }
}

impl From<TriXParseError> for io::Error {
    #[inline]
    fn from(error: TriXParseError) -> Self {
        match error {
            TriXParseError::Io(e) => e,
            TriXParseError::Syntax(e) => e.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax_error_display_and_source() {
        let err = TriXSyntaxError::msg(12, "unexpected element `<foo>`");
        assert_eq!(err.position(), 12);
        assert_eq!(err.to_string(), "at byte 12: unexpected element `<foo>`");
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn parse_error_conversions() {
        let syntax = TriXSyntaxError::msg(0, "boom");
        let parse: TriXParseError = syntax.into();
        assert!(matches!(parse, TriXParseError::Syntax(_)));
        let io_err: io::Error = parse.into();
        assert_eq!(io_err.kind(), io::ErrorKind::InvalidData);

        let parse_io: TriXParseError = io::Error::new(io::ErrorKind::NotFound, "nope").into();
        assert!(matches!(parse_io, TriXParseError::Io(_)));
        assert!(std::error::Error::source(&parse_io).is_some());
    }
}

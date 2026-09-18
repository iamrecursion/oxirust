//! Error types returned while parsing [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/).
//!
//! The hierarchy mirrors the rest of the OxiXML RDF parsers: a top-level
//! [`RdfXmlParseError`] separates I/O failures from data failures, and the data failures are
//! carried by [`RdfXmlSyntaxError`].

use oxixml_iri::IriParseError;
use oxixml_langtag::LanguageTagParseError;
use oxixml_model::OxString;
use oxixml_xml::encoding::EncodingError;
use oxixml_xml::escape::EscapeError;
use oxixml_xml::events::attributes::AttrError;
use std::error::Error;
use std::fmt;
use std::io;

/// Error returned during RDF/XML parsing.
///
/// # Example
///
/// ```
/// use oxixml_rdfxml::{RdfXmlParseError, RdfXmlParser};
///
/// let error = RdfXmlParser::new()
///     .for_reader("<not-xml".as_bytes())
///     .next()
///     .expect("an event")
///     .expect_err("invalid document");
/// assert!(matches!(error, RdfXmlParseError::Syntax(_)));
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum RdfXmlParseError {
    /// I/O error during parsing (file not found...).
    Io(io::Error),
    /// An error in the file syntax.
    Syntax(RdfXmlSyntaxError),
}

impl fmt::Display for RdfXmlParseError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::Syntax(error) => error.fmt(f),
        }
    }
}

impl Error for RdfXmlParseError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Syntax(error) => Some(error),
        }
    }
}

impl From<io::Error> for RdfXmlParseError {
    #[inline]
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<RdfXmlSyntaxError> for RdfXmlParseError {
    #[inline]
    fn from(error: RdfXmlSyntaxError) -> Self {
        Self::Syntax(error)
    }
}

impl From<RdfXmlParseError> for io::Error {
    #[inline]
    fn from(error: RdfXmlParseError) -> Self {
        match error {
            RdfXmlParseError::Io(error) => error,
            RdfXmlParseError::Syntax(error) => error.into(),
        }
    }
}

#[doc(hidden)]
impl From<oxixml_xml::Error> for RdfXmlParseError {
    #[inline]
    fn from(error: oxixml_xml::Error) -> Self {
        match error {
            oxixml_xml::Error::Io(error) => Self::Io(
                std::sync::Arc::try_unwrap(error)
                    .unwrap_or_else(|error| io::Error::new(error.kind(), error)),
            ),
            _ => Self::Syntax(RdfXmlSyntaxError(SyntaxErrorKind::Xml(error))),
        }
    }
}

#[doc(hidden)]
impl From<EncodingError> for RdfXmlParseError {
    #[inline]
    fn from(error: EncodingError) -> Self {
        oxixml_xml::Error::from(error).into()
    }
}

#[doc(hidden)]
impl From<EscapeError> for RdfXmlParseError {
    #[inline]
    fn from(error: EscapeError) -> Self {
        oxixml_xml::Error::from(error).into()
    }
}

#[doc(hidden)]
impl From<AttrError> for RdfXmlParseError {
    #[inline]
    fn from(error: AttrError) -> Self {
        oxixml_xml::Error::from(error).into()
    }
}

/// An error in the syntax of the parsed file.
///
/// This covers both malformed XML and documents that are well-formed XML but do not follow the
/// [RDF/XML grammar](https://www.w3.org/TR/rdf-syntax-grammar/#section-Infoset-Grammar).
#[derive(Debug)]
pub struct RdfXmlSyntaxError(SyntaxErrorKind);

#[derive(Debug)]
enum SyntaxErrorKind {
    Xml(oxixml_xml::Error),
    InvalidIri {
        iri: OxString,
        error: IriParseError,
    },
    InvalidLanguageTag {
        tag: OxString,
        error: LanguageTagParseError,
    },
    Msg(String),
}

impl RdfXmlSyntaxError {
    /// Builds an error from a printable error message.
    pub(crate) fn msg(msg: impl Into<String>) -> Self {
        Self(SyntaxErrorKind::Msg(msg.into()))
    }

    pub(crate) fn invalid_iri(iri: OxString, error: IriParseError) -> Self {
        Self(SyntaxErrorKind::InvalidIri { iri, error })
    }

    pub(crate) fn invalid_language_tag(tag: OxString, error: LanguageTagParseError) -> Self {
        Self(SyntaxErrorKind::InvalidLanguageTag { tag, error })
    }
}

impl fmt::Display for RdfXmlSyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            SyntaxErrorKind::Xml(error) => error.fmt(f),
            SyntaxErrorKind::InvalidIri { iri, error } => {
                write!(f, "error while parsing IRI '{iri}': {error}")
            }
            SyntaxErrorKind::InvalidLanguageTag { tag, error } => {
                write!(f, "error while parsing language tag '{tag}': {error}")
            }
            SyntaxErrorKind::Msg(msg) => f.write_str(msg),
        }
    }
}

impl Error for RdfXmlSyntaxError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.0 {
            SyntaxErrorKind::Xml(error) => Some(error),
            SyntaxErrorKind::InvalidIri { error, .. } => Some(error),
            SyntaxErrorKind::InvalidLanguageTag { error, .. } => Some(error),
            SyntaxErrorKind::Msg(_) => None,
        }
    }
}

impl From<RdfXmlSyntaxError> for io::Error {
    #[inline]
    fn from(error: RdfXmlSyntaxError) -> Self {
        match error.0 {
            SyntaxErrorKind::Xml(error) => match error {
                oxixml_xml::Error::Io(error) => std::sync::Arc::try_unwrap(error)
                    .unwrap_or_else(|error| Self::new(error.kind(), error)),
                _ => Self::new(io::ErrorKind::InvalidData, error),
            },
            SyntaxErrorKind::Msg(msg) => Self::new(io::ErrorKind::InvalidData, msg),
            _ => Self::new(io::ErrorKind::InvalidData, error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax_error_display_and_io_conversion() {
        let error = RdfXmlSyntaxError::msg("boom");
        assert_eq!(error.to_string(), "boom");
        let io_error = io::Error::from(error);
        assert_eq!(io_error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn parse_error_wraps_io() {
        let error = RdfXmlParseError::from(io::Error::other("nope"));
        assert!(matches!(error, RdfXmlParseError::Io(_)));
        assert_eq!(io::Error::from(error).to_string(), "nope");
    }
}

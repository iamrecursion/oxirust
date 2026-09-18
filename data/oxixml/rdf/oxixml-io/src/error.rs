//! Unified error types: [`RdfParseError`], [`RdfSyntaxError`], and [`TextPosition`].

use std::io;
use std::ops::Range;

/// A position in a text: a `line` number, a `column` number (in Unicode code points), and a
/// global byte `offset`, each measured from the start of the document.
///
/// All three are **0-based** for every backend that reports a position — the first character of a
/// document is line `0`, column `0`, offset `0` — matching `oxrdfio`'s
/// [`TextPosition`](https://docs.rs/oxrdfio/latest/oxrdfio/struct.TextPosition.html). This holds
/// both for the Turtle family (N-Triples/N-Quads/Turtle/TriG/N3) and for JSON-LD. TriX only
/// reports a byte `offset` (`line` and `column` are always `0`). RDF/XML does not report a
/// location at all, so [`RdfSyntaxError::location`] returns `None` for it.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub struct TextPosition {
    /// The 0-based line number.
    pub line: u64,
    /// The 0-based column number, in Unicode code points.
    pub column: u64,
    /// The 0-based global byte offset in the input.
    pub offset: u64,
}

/// An error in the syntax (or semantics) of a parsed RDF document.
///
/// Wraps the format-specific syntax error of whichever backend
/// ([`oxixml-turtle`](oxixml_turtle), [`oxixml-rdfxml`](oxixml_rdfxml),
/// [`oxixml-jsonld`](oxixml_jsonld), or [`oxixml-trix`](oxixml_trix)) raised it, and exposes a
/// best-effort [`location`](Self::location) when the backend tracks one.
#[derive(Debug)]
pub struct RdfSyntaxError(SyntaxErrorKind);

#[derive(Debug)]
enum SyntaxErrorKind {
    JsonLd(oxixml_jsonld::JsonLdSyntaxError),
    Turtle(oxixml_turtle::TurtleSyntaxError),
    RdfXml(oxixml_rdfxml::RdfXmlSyntaxError),
    TriX(oxixml_trix::TriXSyntaxError),
    Msg(&'static str),
}

impl RdfSyntaxError {
    /// The location of the error inside of the file, when the backend format tracks one.
    ///
    /// Always `None` for RDF/XML (its parser does not track positions) and for errors raised
    /// directly by `oxixml-io` itself (for example
    /// [`without_named_graphs`](crate::RdfParser::without_named_graphs) rejecting a named graph).
    #[must_use]
    pub fn location(&self) -> Option<Range<TextPosition>> {
        match &self.0 {
            SyntaxErrorKind::JsonLd(error) => {
                let location = error.location()?;
                Some(
                    TextPosition {
                        line: location.start.line,
                        column: location.start.column,
                        offset: location.start.offset,
                    }..TextPosition {
                        line: location.end.line,
                        column: location.end.column,
                        offset: location.end.offset,
                    },
                )
            }
            SyntaxErrorKind::Turtle(error) => {
                let location = error.location();
                Some(
                    TextPosition {
                        line: location.start.line,
                        column: location.start.column,
                        offset: location.start.offset,
                    }..TextPosition {
                        line: location.end.line,
                        column: location.end.column,
                        offset: location.end.offset,
                    },
                )
            }
            SyntaxErrorKind::TriX(error) => {
                let point = TextPosition {
                    line: 0,
                    column: 0,
                    offset: error.position(),
                };
                Some(point..point)
            }
            SyntaxErrorKind::RdfXml(_) | SyntaxErrorKind::Msg(_) => None,
        }
    }

    /// Builds an error from a printable, `oxixml-io`-level error message (used when this crate
    /// itself rejects a document, independently of any backend parser).
    pub(crate) fn msg(msg: &'static str) -> Self {
        Self(SyntaxErrorKind::Msg(msg))
    }
}

impl std::fmt::Display for RdfSyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            SyntaxErrorKind::JsonLd(error) => error.fmt(f),
            SyntaxErrorKind::Turtle(error) => error.fmt(f),
            SyntaxErrorKind::RdfXml(error) => error.fmt(f),
            SyntaxErrorKind::TriX(error) => error.fmt(f),
            SyntaxErrorKind::Msg(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for RdfSyntaxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0 {
            SyntaxErrorKind::JsonLd(error) => Some(error),
            SyntaxErrorKind::Turtle(error) => Some(error),
            SyntaxErrorKind::RdfXml(error) => Some(error),
            SyntaxErrorKind::TriX(error) => Some(error),
            SyntaxErrorKind::Msg(_) => None,
        }
    }
}

impl From<oxixml_jsonld::JsonLdSyntaxError> for RdfSyntaxError {
    #[inline]
    fn from(error: oxixml_jsonld::JsonLdSyntaxError) -> Self {
        Self(SyntaxErrorKind::JsonLd(error))
    }
}

impl From<oxixml_turtle::TurtleSyntaxError> for RdfSyntaxError {
    #[inline]
    fn from(error: oxixml_turtle::TurtleSyntaxError) -> Self {
        Self(SyntaxErrorKind::Turtle(error))
    }
}

impl From<oxixml_rdfxml::RdfXmlSyntaxError> for RdfSyntaxError {
    #[inline]
    fn from(error: oxixml_rdfxml::RdfXmlSyntaxError) -> Self {
        Self(SyntaxErrorKind::RdfXml(error))
    }
}

impl From<oxixml_trix::TriXSyntaxError> for RdfSyntaxError {
    #[inline]
    fn from(error: oxixml_trix::TriXSyntaxError) -> Self {
        Self(SyntaxErrorKind::TriX(error))
    }
}

impl From<RdfSyntaxError> for io::Error {
    #[inline]
    fn from(error: RdfSyntaxError) -> Self {
        match error.0 {
            SyntaxErrorKind::JsonLd(error) => error.into(),
            SyntaxErrorKind::Turtle(error) => error.into(),
            SyntaxErrorKind::RdfXml(error) => error.into(),
            SyntaxErrorKind::TriX(error) => error.into(),
            SyntaxErrorKind::Msg(msg) => Self::new(io::ErrorKind::InvalidData, msg),
        }
    }
}

/// An error returned while parsing an RDF document: either an I/O failure, or an
/// [`RdfSyntaxError`].
#[derive(Debug)]
pub enum RdfParseError {
    /// An I/O error from the underlying byte source (file not found, broken pipe, ...).
    Io(io::Error),
    /// An error in the file syntax.
    Syntax(RdfSyntaxError),
}

impl std::fmt::Display for RdfParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::Syntax(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for RdfParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Syntax(error) => Some(error),
        }
    }
}

impl From<io::Error> for RdfParseError {
    #[inline]
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<RdfSyntaxError> for RdfParseError {
    #[inline]
    fn from(error: RdfSyntaxError) -> Self {
        Self::Syntax(error)
    }
}

impl From<RdfParseError> for io::Error {
    #[inline]
    fn from(error: RdfParseError) -> Self {
        match error {
            RdfParseError::Io(error) => error,
            RdfParseError::Syntax(error) => error.into(),
        }
    }
}

impl From<oxixml_jsonld::JsonLdParseError> for RdfParseError {
    #[inline]
    fn from(error: oxixml_jsonld::JsonLdParseError) -> Self {
        match error {
            oxixml_jsonld::JsonLdParseError::Syntax(error) => Self::Syntax(error.into()),
            oxixml_jsonld::JsonLdParseError::Io(error) => Self::Io(error),
        }
    }
}

impl From<oxixml_turtle::TurtleParseError> for RdfParseError {
    #[inline]
    fn from(error: oxixml_turtle::TurtleParseError) -> Self {
        match error {
            oxixml_turtle::TurtleParseError::Syntax(error) => Self::Syntax(error.into()),
            oxixml_turtle::TurtleParseError::Io(error) => Self::Io(error),
        }
    }
}

impl From<oxixml_rdfxml::RdfXmlParseError> for RdfParseError {
    #[inline]
    fn from(error: oxixml_rdfxml::RdfXmlParseError) -> Self {
        match error {
            oxixml_rdfxml::RdfXmlParseError::Syntax(error) => Self::Syntax(error.into()),
            oxixml_rdfxml::RdfXmlParseError::Io(error) => Self::Io(error),
            // `RdfXmlParseError` is `#[non_exhaustive]`: fall back to an I/O error carrying the
            // message rather than failing to compile against a future added variant.
            other => Self::Io(io::Error::other(other.to_string())),
        }
    }
}

impl From<oxixml_trix::TriXParseError> for RdfParseError {
    #[inline]
    fn from(error: oxixml_trix::TriXParseError) -> Self {
        match error {
            oxixml_trix::TriXParseError::Syntax(error) => Self::Syntax(error.into()),
            oxixml_trix::TriXParseError::Io(error) => Self::Io(error),
        }
    }
}

// The backends' *slice* parsers (Turtle family, RDF/XML) yield a pure syntax-error type rather
// than the full I/O-union error, since parsing a slice can never perform I/O. These direct
// conversions let `SliceQuadParser` map either error shape into `RdfParseError` with one `.into()`.

impl From<oxixml_jsonld::JsonLdSyntaxError> for RdfParseError {
    #[inline]
    fn from(error: oxixml_jsonld::JsonLdSyntaxError) -> Self {
        Self::Syntax(error.into())
    }
}

impl From<oxixml_turtle::TurtleSyntaxError> for RdfParseError {
    #[inline]
    fn from(error: oxixml_turtle::TurtleSyntaxError) -> Self {
        Self::Syntax(error.into())
    }
}

impl From<oxixml_rdfxml::RdfXmlSyntaxError> for RdfParseError {
    #[inline]
    fn from(error: oxixml_rdfxml::RdfXmlSyntaxError) -> Self {
        Self::Syntax(error.into())
    }
}

impl From<oxixml_trix::TriXSyntaxError> for RdfParseError {
    #[inline]
    fn from(error: oxixml_trix::TriXSyntaxError) -> Self {
        Self::Syntax(error.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msg_error_round_trips_through_io() {
        let error = RdfSyntaxError::msg("named graphs are not allowed");
        assert_eq!(error.to_string(), "named graphs are not allowed");
        assert!(error.location().is_none());
        let io_error: io::Error = error.into();
        assert_eq!(io_error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn parse_error_wraps_io() {
        let error = RdfParseError::from(io::Error::other("nope"));
        assert!(matches!(error, RdfParseError::Io(_)));
        assert_eq!(io::Error::from(error).to_string(), "nope");
    }

    #[test]
    fn parse_error_wraps_syntax() {
        let error = RdfParseError::from(RdfSyntaxError::msg("boom"));
        assert!(matches!(error, RdfParseError::Syntax(_)));
        assert_eq!(error.to_string(), "boom");
    }
}

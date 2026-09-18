//! The error types of the RDFa parser.

use std::error::Error;
use std::fmt;
use std::io;

/// An error raised while extracting RDFa from a document.
///
/// RDFa processing is deliberately forgiving: [RDFa Core 1.1
/// §7.6](https://www.w3.org/TR/rdfa-core/#processor-status) requires a processor to keep going
/// after an unresolvable term, prefix or CURIE rather than to stop, so an unresolvable value never
/// becomes an error here — it simply produces no triple. What *can* fail is getting the document
/// into the processor in the first place (reading bytes, establishing a usable base IRI), and a
/// document nested deeper than [`RdfaParser::max_depth`](crate::RdfaParser::max_depth), which
/// stops extraction outright rather than walking it with no bound on native stack usage.
#[derive(Debug)]
pub enum RdfaParseError {
    /// The input could not be read.
    Io(io::Error),
    /// The document, or the base IRI supplied for it, is unusable, or it is nested deeper than
    /// the configured limit.
    Syntax(RdfaSyntaxError),
}

impl fmt::Display for RdfaParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::Syntax(error) => error.fmt(f),
        }
    }
}

impl Error for RdfaParseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Syntax(error) => Some(error),
        }
    }
}

impl From<io::Error> for RdfaParseError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<RdfaSyntaxError> for RdfaParseError {
    fn from(error: RdfaSyntaxError) -> Self {
        Self::Syntax(error)
    }
}

impl From<RdfaParseError> for io::Error {
    fn from(error: RdfaParseError) -> Self {
        match error {
            RdfaParseError::Io(error) => error,
            RdfaParseError::Syntax(error) => Self::new(io::ErrorKind::InvalidData, error),
        }
    }
}

/// A document-level problem that keeps RDFa extraction from completing: an unusable base IRI, or
/// a document nested deeper than the configured limit.
#[derive(Debug)]
pub struct RdfaSyntaxError {
    message: String,
}

impl RdfaSyntaxError {
    pub(crate) fn msg(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The human-readable description of the problem.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for RdfaSyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for RdfaSyntaxError {}

//! Errors of the RDF layer. Nothing here panics.

use core::fmt;

/// An error building or serializing an `OxiEphemeris` RDF graph.
#[non_exhaustive]
#[derive(Debug)]
pub enum RdfError {
    /// The caller-supplied instance base IRI is not a valid absolute IRI.
    InvalidBaseIri(String),
    /// A caller-supplied chart IRI is not a valid absolute IRI.
    InvalidChartIri(String),
    /// Registering a namespace prefix on the serializer failed.
    InvalidPrefix(String),
    /// Writing the serialized graph failed.
    Io(std::io::Error),
}

impl fmt::Display for RdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBaseIri(iri) => {
                write!(f, "invalid base IRI '{iri}': must be an absolute IRI")
            }
            Self::InvalidChartIri(iri) => {
                write!(f, "invalid chart IRI '{iri}': must be an absolute IRI")
            }
            Self::InvalidPrefix(p) => write!(f, "invalid namespace prefix binding '{p}'"),
            Self::Io(e) => write!(f, "writing RDF failed: {e}"),
        }
    }
}

impl std::error::Error for RdfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RdfError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

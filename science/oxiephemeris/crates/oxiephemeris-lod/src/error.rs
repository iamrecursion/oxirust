//! The error type for the LOD store and endpoint.
//!
//! # Why one enum, not `Box<dyn Error>`
//!
//! The endpoint needs to map failures to HTTP status codes, and a *syntax*
//! error (the client sent a malformed query) is a `400` while an
//! *evaluation* error (the query is well-formed but something went wrong
//! executing it) is a `500`. Those two cases originate in different
//! oxigraph types ([`SparqlSyntaxError`] vs [`QueryEvaluationError`] /
//! [`UpdateEvaluationError`]), so we keep them as distinct variants rather
//! than erasing the distinction behind a trait object. Everything else
//! (storage, I/O, serialization, RDF construction) collapses into a
//! server error and carries the underlying message for logging.
//!
//! Nothing in this module panics.

use std::fmt;

use oxiephemeris_rdf::RdfError;
use oxigraph::sparql::{QueryEvaluationError, SparqlSyntaxError, UpdateEvaluationError};
use oxigraph::store::{LoaderError, SerializerError, StorageError};

/// A failure in the LOD store or SPARQL endpoint.
///
/// The variant determines how the endpoint reports the failure over HTTP:
/// [`LodError::Syntax`] is a client error (`400`), everything else is a
/// server error (`500`).
#[non_exhaustive]
#[derive(Debug)]
pub enum LodError {
    /// A SPARQL query or update failed to parse. Reported as `400`.
    Syntax(SparqlSyntaxError),
    /// A well-formed query failed during evaluation.
    QueryEval(QueryEvaluationError),
    /// A well-formed update failed during evaluation.
    UpdateEval(UpdateEvaluationError),
    /// The underlying triple store reported an error.
    Storage(StorageError),
    /// Loading RDF into the store failed (parse or storage).
    Loader(LoaderError),
    /// Serializing the store to RDF failed.
    Serializer(SerializerError),
    /// Building or serializing an in-memory RDF graph failed.
    Rdf(RdfError),
    /// A filesystem or network I/O operation failed.
    Io(std::io::Error),
    /// A configuration or command-line argument was invalid.
    Config(String),
}

impl fmt::Display for LodError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(e) => write!(f, "malformed SPARQL: {e}"),
            Self::QueryEval(e) => write!(f, "query evaluation failed: {e}"),
            Self::UpdateEval(e) => write!(f, "update evaluation failed: {e}"),
            Self::Storage(e) => write!(f, "storage error: {e}"),
            Self::Loader(e) => write!(f, "RDF load error: {e}"),
            Self::Serializer(e) => write!(f, "RDF serialization error: {e}"),
            Self::Rdf(e) => write!(f, "RDF graph error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Config(msg) => write!(f, "configuration error: {msg}"),
        }
    }
}

impl std::error::Error for LodError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Syntax(e) => Some(e),
            Self::QueryEval(e) => Some(e),
            Self::UpdateEval(e) => Some(e),
            Self::Storage(e) => Some(e),
            Self::Loader(e) => Some(e),
            Self::Serializer(e) => Some(e),
            Self::Rdf(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Config(_) => None,
        }
    }
}

impl From<SparqlSyntaxError> for LodError {
    fn from(e: SparqlSyntaxError) -> Self {
        Self::Syntax(e)
    }
}

impl From<QueryEvaluationError> for LodError {
    fn from(e: QueryEvaluationError) -> Self {
        Self::QueryEval(e)
    }
}

impl From<UpdateEvaluationError> for LodError {
    fn from(e: UpdateEvaluationError) -> Self {
        Self::UpdateEval(e)
    }
}

impl From<StorageError> for LodError {
    fn from(e: StorageError) -> Self {
        Self::Storage(e)
    }
}

impl From<LoaderError> for LodError {
    fn from(e: LoaderError) -> Self {
        Self::Loader(e)
    }
}

impl From<SerializerError> for LodError {
    fn from(e: SerializerError) -> Self {
        Self::Serializer(e)
    }
}

impl From<RdfError> for LodError {
    fn from(e: RdfError) -> Self {
        Self::Rdf(e)
    }
}

impl From<std::io::Error> for LodError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

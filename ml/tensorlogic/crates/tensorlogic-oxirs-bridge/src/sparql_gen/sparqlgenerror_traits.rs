//! # SparqlGenError - Trait Implementations
//!
//! This module contains trait implementations for `SparqlGenError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::SparqlGenError;

impl fmt::Display for SparqlGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparqlGenError::UnsupportedExpr(msg) => {
                write!(f, "SPARQL gen: unsupported expression: {msg}")
            }
            SparqlGenError::AmbiguousVariable(var) => {
                write!(f, "SPARQL gen: ambiguous variable '{var}'")
            }
            SparqlGenError::EmptyQuery => {
                write!(f, "SPARQL gen: expression produces an empty query body")
            }
        }
    }
}

impl std::error::Error for SparqlGenError {}

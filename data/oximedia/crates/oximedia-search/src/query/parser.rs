//! Query language parser.

use crate::error::SearchResult;
use serde::{Deserialize, Serialize};

/// Parsed query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParsedQuery {
    /// Term query
    Term(String),
    /// Phrase query
    Phrase(String),
    /// Boolean query
    Boolean(BooleanQuery),
    /// Range query
    Range(RangeQuery),
    /// Fuzzy query
    Fuzzy(String, u8),
}

/// Boolean query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BooleanQuery {
    /// Must clauses (AND)
    pub must: Vec<ParsedQuery>,
    /// Should clauses (OR)
    pub should: Vec<ParsedQuery>,
    /// Must not clauses (NOT)
    pub must_not: Vec<ParsedQuery>,
}

/// Range query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeQuery {
    /// Field name
    pub field: String,
    /// Minimum value
    pub min: Option<i64>,
    /// Maximum value
    pub max: Option<i64>,
}

/// Query parser
pub struct QueryParser;

impl QueryParser {
    /// Create a new query parser
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Parse a query string
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails
    pub fn parse(&self, query: &str) -> SearchResult<ParsedQuery> {
        // Simple implementation: treat as term query
        Ok(ParsedQuery::Term(query.to_string()))
    }
}

impl Default for QueryParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_term() {
        let parser = QueryParser::new();
        let result = parser.parse("hello").expect("should succeed in test");
        assert!(matches!(result, ParsedQuery::Term(_)));
    }
}

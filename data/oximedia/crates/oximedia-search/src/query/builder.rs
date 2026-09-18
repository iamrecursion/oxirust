//! Query builder API.

use crate::query::parser::ParsedQuery;

/// Query builder for constructing complex queries
pub struct QueryBuilder {
    queries: Vec<ParsedQuery>,
}

impl QueryBuilder {
    /// Create a new query builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            queries: Vec::new(),
        }
    }

    /// Add a term query
    #[must_use]
    pub fn term(mut self, term: &str) -> Self {
        self.queries.push(ParsedQuery::Term(term.to_string()));
        self
    }

    /// Add a phrase query
    #[must_use]
    pub fn phrase(mut self, phrase: &str) -> Self {
        self.queries.push(ParsedQuery::Phrase(phrase.to_string()));
        self
    }

    /// Build the final query
    #[must_use]
    pub fn build(self) -> Vec<ParsedQuery> {
        self.queries
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() {
        let builder = QueryBuilder::new().term("hello").phrase("world peace");

        let queries = builder.build();
        assert_eq!(queries.len(), 2);
    }
}

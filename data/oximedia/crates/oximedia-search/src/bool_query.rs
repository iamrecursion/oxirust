//! Boolean query parsing and evaluation for structured search.
//!
//! Provides a simple boolean query AST and an evaluator that can test whether
//! a document (represented as a set of terms) satisfies a boolean expression.
//! Supports AND, OR, NOT, and phrase matching.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Query AST
// ---------------------------------------------------------------------------

/// A node in a boolean query expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum BoolQuery {
    /// A single term that must appear in the document.
    Term(String),
    /// All sub-queries must be satisfied (conjunction).
    And(Vec<BoolQuery>),
    /// At least one sub-query must be satisfied (disjunction).
    Or(Vec<BoolQuery>),
    /// The sub-query must NOT be satisfied.
    Not(Box<BoolQuery>),
    /// A phrase where all words must appear adjacent (simplified: all terms
    /// must be present; ordering is not verified at this layer).
    Phrase(Vec<String>),
}

impl BoolQuery {
    /// Convenience constructor for a `Term` query.
    #[must_use]
    pub fn term(s: &str) -> Self {
        Self::Term(s.to_lowercase())
    }

    /// Convenience constructor for an `And` query.
    #[must_use]
    pub fn and(clauses: Vec<Self>) -> Self {
        Self::And(clauses)
    }

    /// Convenience constructor for an `Or` query.
    #[must_use]
    pub fn or(clauses: Vec<Self>) -> Self {
        Self::Or(clauses)
    }

    /// Convenience constructor for a `Not` query.
    #[must_use]
    pub fn not(inner: Self) -> Self {
        Self::Not(Box::new(inner))
    }

    /// Convenience constructor for a `Phrase` query.
    #[must_use]
    pub fn phrase(words: &[&str]) -> Self {
        Self::Phrase(words.iter().map(|w| w.to_lowercase()).collect())
    }

    /// Evaluate whether this query matches `doc_terms`.
    #[must_use]
    pub fn matches(&self, doc_terms: &HashSet<String>) -> bool {
        match self {
            Self::Term(t) => doc_terms.contains(t),
            Self::And(clauses) => clauses.iter().all(|c| c.matches(doc_terms)),
            Self::Or(clauses) => clauses.iter().any(|c| c.matches(doc_terms)),
            Self::Not(inner) => !inner.matches(doc_terms),
            Self::Phrase(words) => words.iter().all(|w| doc_terms.contains(w)),
        }
    }
}

// ---------------------------------------------------------------------------
// Simple query parser
// ---------------------------------------------------------------------------

/// Parse a simple query string into a `BoolQuery`.
///
/// Syntax (case-insensitive):
/// - `term` → `Term("term")`
/// - `a AND b` → `And([Term("a"), Term("b")])`
/// - `a OR b` → `Or([Term("a"), Term("b")])`
/// - `NOT a` → `Not(Term("a"))`
/// - Multiple space-separated tokens without an operator are treated as AND.
///
/// This is intentionally simplified and does not handle nested parentheses.
#[must_use]
pub fn parse_query(input: &str) -> BoolQuery {
    let input = input.trim();
    if input.is_empty() {
        return BoolQuery::And(vec![]);
    }

    let tokens: Vec<&str> = input.split_whitespace().collect();

    // Check for NOT prefix.
    if tokens[0].eq_ignore_ascii_case("NOT") && tokens.len() > 1 {
        let rest = tokens[1..].join(" ");
        return BoolQuery::not(parse_query(&rest));
    }

    // Look for top-level AND / OR operators.
    if let Some(and_pos) = tokens.iter().position(|t| t.eq_ignore_ascii_case("AND")) {
        let left = tokens[..and_pos].join(" ");
        let right = tokens[and_pos + 1..].join(" ");
        return BoolQuery::and(vec![parse_query(&left), parse_query(&right)]);
    }

    if let Some(or_pos) = tokens.iter().position(|t| t.eq_ignore_ascii_case("OR")) {
        let left = tokens[..or_pos].join(" ");
        let right = tokens[or_pos + 1..].join(" ");
        return BoolQuery::or(vec![parse_query(&left), parse_query(&right)]);
    }

    // Multiple terms without operator → implicit AND.
    if tokens.len() == 1 {
        BoolQuery::term(tokens[0])
    } else {
        let clauses = tokens.iter().map(|t| BoolQuery::term(t)).collect();
        BoolQuery::And(clauses)
    }
}

// ---------------------------------------------------------------------------
// Fuzzy matching
// ---------------------------------------------------------------------------

/// Compute the **Levenshtein (edit) distance** between two strings.
///
/// Returns the minimum number of single-character insertions, deletions, or
/// substitutions required to transform `a` into `b`.
#[must_use]
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1])
            };
        }
    }
    dp[m][n]
}

/// Return all terms from `candidates` within `max_distance` edits of `query`.
#[must_use]
pub fn fuzzy_match<'a>(
    query: &str,
    candidates: &[&'a str],
    max_distance: usize,
) -> Vec<(&'a str, usize)> {
    let q_lower = query.to_lowercase();
    let mut matches: Vec<(&str, usize)> = candidates
        .iter()
        .filter_map(|c| {
            let dist = edit_distance(&q_lower, &c.to_lowercase());
            if dist <= max_distance {
                Some((*c, dist))
            } else {
                None
            }
        })
        .collect();
    matches.sort_by_key(|(_, d)| *d);
    matches
}

// ---------------------------------------------------------------------------
// Query expansion
// ---------------------------------------------------------------------------

/// Expand `query` by adding fuzzy variants found in `vocab` within
/// `max_distance` edits.
///
/// Returns an `Or` query containing the original term plus any fuzzy matches.
#[must_use]
pub fn expand_query(query: &str, vocab: &[&str], max_distance: usize) -> BoolQuery {
    let q_lower = query.to_lowercase();
    let mut clauses: Vec<BoolQuery> = vec![BoolQuery::term(&q_lower)];
    for (term, _) in fuzzy_match(&q_lower, vocab, max_distance) {
        if term.to_lowercase() != q_lower {
            clauses.push(BoolQuery::term(term));
        }
    }
    if clauses.len() == 1 {
        clauses.remove(0)
    } else {
        BoolQuery::Or(clauses)
    }
}

// ---------------------------------------------------------------------------
// Term set helpers
// ---------------------------------------------------------------------------

/// Build a `HashSet<String>` of lowercase tokens from `text`.
#[must_use]
pub fn terms_from_text(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|t| {
            t.to_lowercase()
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_owned()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(text: &str) -> HashSet<String> {
        terms_from_text(text)
    }

    // --- BoolQuery::matches ---

    #[test]
    fn test_term_match() {
        let q = BoolQuery::term("cat");
        assert!(q.matches(&doc("a cat sat on the mat")));
        assert!(!q.matches(&doc("a dog sat on the mat")));
    }

    #[test]
    fn test_and_match() {
        let q = BoolQuery::and(vec![BoolQuery::term("cat"), BoolQuery::term("mat")]);
        assert!(q.matches(&doc("cat sat on mat")));
        assert!(!q.matches(&doc("cat sat here")));
    }

    #[test]
    fn test_or_match() {
        let q = BoolQuery::or(vec![BoolQuery::term("cat"), BoolQuery::term("dog")]);
        assert!(q.matches(&doc("dog is here")));
        assert!(!q.matches(&doc("fish are here")));
    }

    #[test]
    fn test_not_match() {
        let q = BoolQuery::not(BoolQuery::term("cat"));
        assert!(q.matches(&doc("a dog")));
        assert!(!q.matches(&doc("a cat")));
    }

    #[test]
    fn test_phrase_match_all_terms_present() {
        let q = BoolQuery::phrase(&["quick", "brown", "fox"]);
        assert!(q.matches(&doc("the quick brown fox jumps")));
        assert!(!q.matches(&doc("the quick fox")));
    }

    // --- parse_query ---

    #[test]
    fn test_parse_single_term() {
        let q = parse_query("hello");
        assert!(q.matches(&doc("hello world")));
        assert!(!q.matches(&doc("goodbye world")));
    }

    #[test]
    fn test_parse_and() {
        let q = parse_query("hello AND world");
        assert!(q.matches(&doc("hello world")));
        assert!(!q.matches(&doc("hello there")));
    }

    #[test]
    fn test_parse_or() {
        let q = parse_query("cat OR dog");
        assert!(q.matches(&doc("I love my dog")));
        assert!(q.matches(&doc("a cat is here")));
        assert!(!q.matches(&doc("fish swim")));
    }

    #[test]
    fn test_parse_not() {
        let q = parse_query("NOT spam");
        assert!(q.matches(&doc("legitimate email")));
        assert!(!q.matches(&doc("buy cheap spam pills")));
    }

    #[test]
    fn test_parse_implicit_and() {
        let q = parse_query("video audio codec");
        assert!(q.matches(&doc("video audio codec quality")));
        assert!(!q.matches(&doc("video audio quality")));
    }

    // --- edit_distance ---

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance("kitten", "kitten"), 0);
    }

    #[test]
    fn test_edit_distance_one_edit() {
        assert_eq!(edit_distance("cat", "bat"), 1);
    }

    #[test]
    fn test_edit_distance_empty_strings() {
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("abc", ""), 3);
    }

    // --- fuzzy_match ---

    #[test]
    fn test_fuzzy_match_exact() {
        let candidates = ["colour", "color", "collar"];
        let result = fuzzy_match("color", &candidates, 2);
        // "color" exact match should be first
        assert!(!result.is_empty());
        assert_eq!(result[0].1, 0);
    }

    #[test]
    fn test_fuzzy_match_no_results_beyond_distance() {
        let candidates = ["completely", "different"];
        let result = fuzzy_match("cat", &candidates, 1);
        assert!(result.is_empty());
    }

    // --- expand_query ---

    #[test]
    fn test_expand_query_adds_variants() {
        let vocab = ["colour", "colors", "collar"];
        let expanded = expand_query("color", &vocab, 2);
        // Should be an Or that includes at least "color" and "colour".
        let doc_colour = doc("colour film");
        assert!(expanded.matches(&doc_colour));
    }

    // --- terms_from_text ---

    #[test]
    fn test_terms_from_text_lowercase() {
        let terms = terms_from_text("Hello World");
        assert!(terms.contains("hello"));
        assert!(terms.contains("world"));
    }
}

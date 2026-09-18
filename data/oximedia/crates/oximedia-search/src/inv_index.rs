//! Inverted index and term-frequency utilities for full-text search.
//!
//! This module provides a lightweight, in-process inverted index distinct from
//! the Tantivy-backed index in the `index` directory module.  It is useful for
//! unit testing, small document sets, and situations where a full-blown search
//! engine is not warranted.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

/// Tokenise `text` into lowercase alphabetic tokens.
///
/// Punctuation, digits, and whitespace are used as delimiters and are dropped.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Remove common English stop-words from a token list.
#[must_use]
pub fn remove_stopwords(tokens: Vec<String>) -> Vec<String> {
    const STOP: &[&str] = &[
        "a", "an", "the", "is", "in", "on", "at", "to", "for", "of", "and", "or", "but", "be",
        "as", "it", "its", "by", "from", "with", "this", "that", "are", "was",
    ];
    let stop_set: HashSet<&str> = STOP.iter().copied().collect();
    tokens
        .into_iter()
        .filter(|t| !stop_set.contains(t.as_str()))
        .collect()
}

// ---------------------------------------------------------------------------
// Term frequency / TF-IDF
// ---------------------------------------------------------------------------

/// Compute the **term frequency** (TF) of each token in `tokens`.
///
/// Uses the raw count definition: TF(t) = count(t) / len(tokens).
#[must_use]
pub fn term_frequency(tokens: &[String]) -> HashMap<String, f32> {
    if tokens.is_empty() {
        return HashMap::new();
    }
    let mut counts: HashMap<String, f32> = HashMap::new();
    for token in tokens {
        *counts.entry(token.clone()).or_insert(0.0) += 1.0;
    }
    let total = tokens.len() as f32;
    counts.values_mut().for_each(|v| *v /= total);
    counts
}

/// Compute **inverse document frequency** for a term given `doc_count`
/// documents, `df` of which contain the term.
///
/// Uses the smoothed IDF formula: log((1 + `doc_count`) / (1 + df)) + 1.
#[must_use]
pub fn inverse_document_frequency(doc_count: usize, df: usize) -> f32 {
    let n = doc_count as f32;
    let d = df as f32;
    ((1.0 + n) / (1.0 + d)).ln() + 1.0
}

// ---------------------------------------------------------------------------
// Posting list
// ---------------------------------------------------------------------------

/// One entry in a posting list: the document that contains the term and the
/// within-document term frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct Posting {
    /// Document identifier.
    pub doc_id: Uuid,
    /// Term frequency within this document.
    pub tf: f32,
}

impl Posting {
    /// Create a new posting.
    #[must_use]
    pub fn new(doc_id: Uuid, tf: f32) -> Self {
        Self { doc_id, tf }
    }
}

// ---------------------------------------------------------------------------
// Inverted index
// ---------------------------------------------------------------------------

/// In-memory inverted index mapping terms to posting lists.
///
/// Documents are stored as raw text; they are tokenised and indexed on
/// insertion.
#[derive(Debug, Clone, Default)]
pub struct InvertedIndex {
    /// term → list of postings
    postings: HashMap<String, Vec<Posting>>,
    /// `doc_id` → original text (for retrieval)
    docs: HashMap<Uuid, String>,
    /// Total number of documents
    doc_count: usize,
    /// term → document frequency
    df: HashMap<String, usize>,
}

impl InvertedIndex {
    /// Create an empty inverted index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a document to the index.
    pub fn add_document(&mut self, doc_id: Uuid, text: &str) {
        let tokens = remove_stopwords(tokenize(text));
        let tf_map = term_frequency(&tokens);

        for (term, tf) in &tf_map {
            // Update document frequency.
            *self.df.entry(term.clone()).or_insert(0) += 1;

            // Insert posting.
            self.postings
                .entry(term.clone())
                .or_default()
                .push(Posting::new(doc_id, *tf));
        }

        self.docs.insert(doc_id, text.to_owned());
        self.doc_count += 1;
    }

    /// Remove a document from the index.
    pub fn remove_document(&mut self, doc_id: Uuid) {
        if self.docs.remove(&doc_id).is_none() {
            return;
        }
        self.doc_count = self.doc_count.saturating_sub(1);

        // Remove postings and update df.
        for postings in self.postings.values_mut() {
            if let Some(pos) = postings.iter().position(|p| p.doc_id == doc_id) {
                postings.swap_remove(pos);
            }
        }
        for (term, postings) in &self.postings {
            let count = postings.len();
            self.df.insert(term.clone(), count);
        }
    }

    /// Return the posting list for `term`, or an empty slice.
    pub fn postings_for(&self, term: &str) -> &[Posting] {
        self.postings.get(term).map_or(&[], Vec::as_slice)
    }

    /// Compute the TF-IDF score for `term` in `doc_id`.
    #[must_use]
    pub fn tfidf(&self, term: &str, doc_id: Uuid) -> f32 {
        let tf = self
            .postings_for(term)
            .iter()
            .find(|p| p.doc_id == doc_id)
            .map_or(0.0, |p| p.tf);
        let df = self.df.get(term).copied().unwrap_or(0);
        let idf = inverse_document_frequency(self.doc_count, df);
        tf * idf
    }

    /// Search for documents containing `term` ranked by TF-IDF score.
    #[must_use]
    pub fn search(&self, term: &str) -> Vec<(Uuid, f32)> {
        let query_term = term.to_lowercase();
        let mut results: Vec<(Uuid, f32)> = self
            .postings_for(&query_term)
            .iter()
            .map(|p| (p.doc_id, self.tfidf(&query_term, p.doc_id)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    /// Number of distinct terms.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.postings.len()
    }

    /// Retrieve the original text of a document.
    pub fn get_document(&self, doc_id: Uuid) -> Option<&str> {
        self.docs.get(&doc_id).map(String::as_str)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uid() -> Uuid {
        Uuid::new_v4()
    }

    // --- tokenize ---

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn test_tokenize_numbers_dropped() {
        let tokens = tokenize("abc 123 def");
        // digits are delimiters; "abc" and "def" survive
        assert!(tokens.contains(&"abc".to_owned()));
        assert!(tokens.contains(&"def".to_owned()));
    }

    // --- remove_stopwords ---

    #[test]
    fn test_remove_stopwords_filters_the() {
        let tokens = vec!["the".to_owned(), "quick".to_owned(), "fox".to_owned()];
        let filtered = remove_stopwords(tokens);
        assert!(!filtered.contains(&"the".to_owned()));
        assert!(filtered.contains(&"quick".to_owned()));
    }

    #[test]
    fn test_remove_stopwords_empty() {
        assert!(remove_stopwords(vec![]).is_empty());
    }

    // --- term_frequency ---

    #[test]
    fn test_term_frequency_basic() {
        let tokens = vec!["cat".to_owned(), "cat".to_owned(), "dog".to_owned()];
        let tf = term_frequency(&tokens);
        assert!((tf["cat"] - 2.0 / 3.0).abs() < 1e-5);
        assert!((tf["dog"] - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_term_frequency_empty() {
        assert!(term_frequency(&[]).is_empty());
    }

    // --- IDF ---

    #[test]
    fn test_idf_zero_df_gives_large_value() {
        // idf should be large when df is small relative to N
        let idf = inverse_document_frequency(1000, 1);
        assert!(idf > 1.0);
    }

    #[test]
    fn test_idf_all_docs_contain_term() {
        let idf = inverse_document_frequency(100, 100);
        // log((101)/(101)) + 1 = 0 + 1 = 1
        assert!((idf - 1.0).abs() < 1e-5);
    }

    // --- Posting ---

    #[test]
    fn test_posting_new() {
        let id = uid();
        let p = Posting::new(id, 0.33);
        assert_eq!(p.doc_id, id);
        assert!((p.tf - 0.33).abs() < 1e-5);
    }

    // --- InvertedIndex ---

    #[test]
    fn test_index_add_and_search() {
        let mut idx = InvertedIndex::new();
        let id = uid();
        idx.add_document(id, "quick brown fox jumps");
        let results = idx.search("fox");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
    }

    #[test]
    fn test_index_doc_count() {
        let mut idx = InvertedIndex::new();
        assert_eq!(idx.doc_count(), 0);
        idx.add_document(uid(), "hello world");
        idx.add_document(uid(), "another document");
        assert_eq!(idx.doc_count(), 2);
    }

    #[test]
    fn test_index_missing_term_returns_empty() {
        let idx = InvertedIndex::new();
        assert!(idx.search("nonexistent").is_empty());
    }

    #[test]
    fn test_index_remove_document() {
        let mut idx = InvertedIndex::new();
        let id = uid();
        idx.add_document(id, "unique rainbow term");
        assert_eq!(idx.search("rainbow").len(), 1);
        idx.remove_document(id);
        assert_eq!(idx.doc_count(), 0);
        assert!(idx.search("rainbow").is_empty());
    }

    #[test]
    fn test_index_vocab_size() {
        let mut idx = InvertedIndex::new();
        idx.add_document(uid(), "apple banana cherry");
        // After stop-word removal, should have 3 terms.
        assert!(idx.vocab_size() >= 1);
    }

    #[test]
    fn test_index_get_document() {
        let mut idx = InvertedIndex::new();
        let id = uid();
        idx.add_document(id, "stored text");
        assert_eq!(idx.get_document(id), Some("stored text"));
    }

    #[test]
    fn test_tfidf_positive() {
        let mut idx = InvertedIndex::new();
        let id = uid();
        idx.add_document(id, "cat cat dog");
        let score = idx.tfidf("cat", id);
        assert!(score > 0.0);
    }

    #[test]
    fn test_search_ranks_higher_tf_first() {
        let mut idx = InvertedIndex::new();
        let id_many = uid();
        let id_few = uid();
        // id_many has "music" twice, id_few once
        idx.add_document(id_many, "music music beats");
        idx.add_document(id_few, "music art dance");
        let results = idx.search("music");
        assert_eq!(results[0].0, id_many);
    }
}

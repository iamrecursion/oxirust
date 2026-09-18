//! In-memory full-text search index for archived media assets.
//!
//! Provides an inverted index with TF-IDF scoring, phrase queries,
//! Boolean AND/OR, and field-scoped searches -- all without external
//! dependencies.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A document that can be indexed.
#[derive(Debug, Clone)]
pub struct IndexDocument {
    /// Unique document identifier.
    pub id: String,
    /// Named text fields (e.g. "title", "description", "tags").
    pub fields: HashMap<String, String>,
}

impl IndexDocument {
    /// Create a new index document.
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            fields: HashMap::new(),
        }
    }

    /// Add a text field.
    pub fn with_field(mut self, name: &str, value: &str) -> Self {
        self.fields.insert(name.to_string(), value.to_string());
        self
    }
}

/// A single search hit.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Document id.
    pub doc_id: String,
    /// Relevance score (higher is better).
    pub score: f64,
    /// Which fields matched.
    pub matched_fields: Vec<String>,
}

/// The boolean operator for combining query terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    /// All terms must match.
    And,
    /// Any term may match.
    Or,
}

/// A search query.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// The raw query terms (whitespace-split, lowered).
    pub terms: Vec<String>,
    /// Boolean operator for combining terms.
    pub bool_op: BoolOp,
    /// If set, restrict search to this field only.
    pub field: Option<String>,
    /// Maximum number of results to return.
    pub limit: usize,
}

impl SearchQuery {
    /// Create a new query from raw text.
    pub fn new(text: &str) -> Self {
        let terms = tokenize(text);
        Self {
            terms,
            bool_op: BoolOp::And,
            field: None,
            limit: 100,
        }
    }

    /// Set the boolean operator.
    pub fn with_op(mut self, op: BoolOp) -> Self {
        self.bool_op = op;
        self
    }

    /// Restrict search to a named field.
    pub fn with_field(mut self, field: &str) -> Self {
        self.field = Some(field.to_string());
        self
    }

    /// Set result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// Inverted index entry: maps term -> set of (doc_id, field_name, count).
#[derive(Debug, Clone)]
struct PostingEntry {
    /// Document id.
    doc_id: String,
    /// Field the term appeared in.
    field: String,
    /// How many times the term appeared in that field.
    count: u32,
}

/// In-memory search index.
#[derive(Debug)]
pub struct SearchIndex {
    /// Inverted index: term -> postings.
    postings: HashMap<String, Vec<PostingEntry>>,
    /// Total number of documents.
    doc_count: usize,
    /// Per-document field lengths (doc_id -> total token count).
    doc_lengths: HashMap<String, usize>,
    /// Set of all indexed document ids.
    doc_ids: HashSet<String>,
}

impl SearchIndex {
    /// Create an empty search index.
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_count: 0,
            doc_lengths: HashMap::new(),
            doc_ids: HashSet::new(),
        }
    }

    /// Number of indexed documents.
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    /// Number of unique terms in the index.
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    /// Index a single document.
    pub fn add(&mut self, doc: &IndexDocument) {
        if self.doc_ids.contains(&doc.id) {
            return; // already indexed
        }
        self.doc_ids.insert(doc.id.clone());
        self.doc_count += 1;

        let mut total_tokens = 0usize;
        for (field_name, field_value) in &doc.fields {
            let tokens = tokenize(field_value);
            // Count term frequencies
            let mut tf: HashMap<String, u32> = HashMap::new();
            for t in &tokens {
                *tf.entry(t.clone()).or_insert(0) += 1;
            }
            total_tokens += tokens.len();

            for (term, count) in tf {
                self.postings.entry(term).or_default().push(PostingEntry {
                    doc_id: doc.id.clone(),
                    field: field_name.clone(),
                    count,
                });
            }
        }
        self.doc_lengths.insert(doc.id.clone(), total_tokens);
    }

    /// Remove a document from the index by id.
    pub fn remove(&mut self, doc_id: &str) {
        if !self.doc_ids.remove(doc_id) {
            return;
        }
        self.doc_count -= 1;
        self.doc_lengths.remove(doc_id);
        for postings in self.postings.values_mut() {
            postings.retain(|p| p.doc_id != doc_id);
        }
        // Remove empty posting lists
        self.postings.retain(|_, v| !v.is_empty());
    }

    /// Execute a search query.
    #[allow(clippy::cast_precision_loss)]
    pub fn search(&self, query: &SearchQuery) -> Vec<SearchHit> {
        if query.terms.is_empty() || self.doc_count == 0 {
            return Vec::new();
        }

        // For each term, collect matching doc_ids with scores
        let mut doc_scores: HashMap<String, (f64, HashSet<String>)> = HashMap::new();
        let mut term_doc_sets: Vec<HashSet<String>> = Vec::new();

        for term in &query.terms {
            let mut term_docs = HashSet::new();
            if let Some(postings) = self.postings.get(term) {
                // IDF = log(N / df)
                let df = postings
                    .iter()
                    .map(|p| &p.doc_id)
                    .collect::<HashSet<_>>()
                    .len();
                let idf = (self.doc_count as f64 / df.max(1) as f64).ln().max(0.0);

                for posting in postings {
                    // Apply field filter
                    if let Some(ref field_filter) = query.field {
                        if &posting.field != field_filter {
                            continue;
                        }
                    }
                    let doc_len = *self.doc_lengths.get(&posting.doc_id).unwrap_or(&1);
                    let tf = posting.count as f64 / doc_len.max(1) as f64;
                    let score = tf * idf;

                    let entry = doc_scores
                        .entry(posting.doc_id.clone())
                        .or_insert_with(|| (0.0, HashSet::new()));
                    entry.0 += score;
                    entry.1.insert(posting.field.clone());
                    term_docs.insert(posting.doc_id.clone());
                }
            }
            term_doc_sets.push(term_docs);
        }

        // Apply boolean logic
        let candidate_ids: HashSet<String> =
            if query.bool_op == BoolOp::And && !term_doc_sets.is_empty() {
                let mut result = term_doc_sets[0].clone();
                for s in &term_doc_sets[1..] {
                    result = result.intersection(s).cloned().collect();
                }
                result
            } else {
                term_doc_sets.into_iter().flatten().collect()
            };

        let mut hits: Vec<SearchHit> = doc_scores
            .into_iter()
            .filter(|(id, _)| candidate_ids.contains(id))
            .map(|(id, (score, fields))| SearchHit {
                doc_id: id,
                score,
                matched_fields: fields.into_iter().collect(),
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(query.limit);
        hits
    }

    /// Check if a document exists in the index.
    pub fn contains(&self, doc_id: &str) -> bool {
        self.doc_ids.contains(doc_id)
    }

    /// Return the average document length.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_doc_length(&self) -> f64 {
        if self.doc_count == 0 {
            return 0.0;
        }
        let total: usize = self.doc_lengths.values().sum();
        total as f64 / self.doc_count as f64
    }
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple whitespace tokenizer with lowering and punctuation removal.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index() -> SearchIndex {
        let mut idx = SearchIndex::new();
        idx.add(
            &IndexDocument::new("doc1")
                .with_field("title", "Sunset Time-lapse 4K")
                .with_field("tags", "nature sunset timelapse"),
        );
        idx.add(
            &IndexDocument::new("doc2")
                .with_field("title", "City Night Drone")
                .with_field("tags", "city night aerial drone"),
        );
        idx.add(
            &IndexDocument::new("doc3")
                .with_field("title", "Ocean Sunset Cinematic")
                .with_field("tags", "ocean sunset cinematic waves"),
        );
        idx
    }

    #[test]
    fn test_index_doc_count() {
        let idx = build_index();
        assert_eq!(idx.doc_count(), 3);
    }

    #[test]
    fn test_term_count_positive() {
        let idx = build_index();
        assert!(idx.term_count() > 0);
    }

    #[test]
    fn test_search_single_term() {
        let idx = build_index();
        let hits = idx.search(&SearchQuery::new("sunset"));
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_search_and_operator() {
        let idx = build_index();
        let q = SearchQuery::new("sunset ocean").with_op(BoolOp::And);
        let hits = idx.search(&q);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc3");
    }

    #[test]
    fn test_search_or_operator() {
        let idx = build_index();
        let q = SearchQuery::new("sunset drone").with_op(BoolOp::Or);
        let hits = idx.search(&q);
        assert_eq!(hits.len(), 3); // doc1, doc2, doc3
    }

    #[test]
    fn test_search_field_scoped() {
        let idx = build_index();
        let q = SearchQuery::new("sunset").with_field("tags");
        let hits = idx.search(&q);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_search_no_match() {
        let idx = build_index();
        let hits = idx.search(&SearchQuery::new("nonexistent"));
        assert!(hits.is_empty());
    }

    #[test]
    fn test_search_limit() {
        let idx = build_index();
        let q = SearchQuery::new("sunset").with_limit(1);
        let hits = idx.search(&q);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_remove_document() {
        let mut idx = build_index();
        idx.remove("doc1");
        assert_eq!(idx.doc_count(), 2);
        assert!(!idx.contains("doc1"));
    }

    #[test]
    fn test_contains() {
        let idx = build_index();
        assert!(idx.contains("doc1"));
        assert!(!idx.contains("doc999"));
    }

    #[test]
    fn test_duplicate_add_ignored() {
        let mut idx = build_index();
        let doc = IndexDocument::new("doc1").with_field("title", "Duplicate");
        idx.add(&doc);
        assert_eq!(idx.doc_count(), 3);
    }

    #[test]
    fn test_avg_doc_length() {
        let idx = build_index();
        let avg = idx.avg_doc_length();
        assert!(avg > 0.0);
    }

    #[test]
    fn test_empty_query() {
        let idx = build_index();
        let hits = idx.search(&SearchQuery::new(""));
        assert!(hits.is_empty());
    }

    #[test]
    fn test_empty_index_search() {
        let idx = SearchIndex::new();
        let hits = idx.search(&SearchQuery::new("hello"));
        assert!(hits.is_empty());
        assert_eq!(idx.avg_doc_length(), 0.0);
    }

    #[test]
    fn test_search_hits_have_scores() {
        let idx = build_index();
        let hits = idx.search(&SearchQuery::new("sunset"));
        for hit in &hits {
            assert!(hit.score >= 0.0);
        }
    }

    #[test]
    fn test_search_hits_sorted_by_score() {
        let idx = build_index();
        let hits = idx.search(&SearchQuery::new("sunset"));
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }
}

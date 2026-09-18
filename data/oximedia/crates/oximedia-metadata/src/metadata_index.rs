#![allow(dead_code)]
//! Metadata indexing and full-text search index building.
//!
//! Provides in-memory inverted index construction over metadata fields,
//! enabling fast lookup by keyword, field name, or combined queries.

use std::collections::{BTreeMap, HashMap, HashSet};

/// Unique identifier for an indexed document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DocId(pub u64);

/// A single term extracted from metadata text.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Term {
    /// The normalized (lowercased, trimmed) term text.
    pub text: String,
    /// The metadata field this term was extracted from.
    pub field: String,
}

impl Term {
    /// Create a new term.
    pub fn new(text: &str, field: &str) -> Self {
        Self {
            text: text.to_lowercase().trim().to_string(),
            field: field.to_string(),
        }
    }
}

/// A posting entry in the inverted index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Posting {
    /// The document containing this term.
    pub doc_id: DocId,
    /// Position(s) of the term within the field value.
    pub positions: Vec<u32>,
    /// Term frequency in this document.
    pub term_frequency: u32,
}

impl Posting {
    /// Create a new posting entry.
    pub fn new(doc_id: DocId, positions: Vec<u32>) -> Self {
        let term_frequency = positions.len() as u32;
        Self {
            doc_id,
            positions,
            term_frequency,
        }
    }
}

/// Configuration for the metadata index.
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Minimum term length to index.
    pub min_term_length: usize,
    /// Maximum term length to index.
    pub max_term_length: usize,
    /// Whether to apply case folding.
    pub case_fold: bool,
    /// Stop words to exclude from indexing.
    pub stop_words: HashSet<String>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        let stop_words: HashSet<String> = ["the", "a", "an", "is", "at", "on", "in", "of", "to"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        Self {
            min_term_length: 2,
            max_term_length: 128,
            case_fold: true,
            stop_words,
        }
    }
}

/// An in-memory inverted index for metadata.
#[derive(Debug, Clone)]
pub struct MetadataIndex {
    /// Configuration.
    config: IndexConfig,
    /// Inverted index: term text -> list of postings.
    postings: HashMap<String, Vec<Posting>>,
    /// Forward index: doc_id -> field -> raw text.
    documents: BTreeMap<DocId, HashMap<String, String>>,
    /// Total number of indexed documents.
    doc_count: u64,
    /// Next document id to assign.
    next_doc_id: u64,
}

impl MetadataIndex {
    /// Create a new metadata index with default configuration.
    pub fn new() -> Self {
        Self::with_config(IndexConfig::default())
    }

    /// Create a new metadata index with the given configuration.
    pub fn with_config(config: IndexConfig) -> Self {
        Self {
            config,
            postings: HashMap::new(),
            documents: BTreeMap::new(),
            doc_count: 0,
            next_doc_id: 0,
        }
    }

    /// Index a document with the given fields.
    ///
    /// Returns the assigned document identifier.
    pub fn add_document(&mut self, fields: HashMap<String, String>) -> DocId {
        let doc_id = DocId(self.next_doc_id);
        self.next_doc_id += 1;

        for (field, value) in &fields {
            let terms = self.tokenize(value);
            for (pos, term_text) in terms.iter().enumerate() {
                if self.should_index(term_text) {
                    let normalized = if self.config.case_fold {
                        term_text.to_lowercase()
                    } else {
                        term_text.clone()
                    };
                    let entry = self.postings.entry(normalized).or_default();
                    // Check if we already have a posting for this doc in this term
                    if let Some(posting) = entry.iter_mut().find(|p| p.doc_id == doc_id) {
                        posting.positions.push(pos as u32);
                        posting.term_frequency += 1;
                    } else {
                        entry.push(Posting::new(doc_id, vec![pos as u32]));
                    }
                    let _ = field; // field tracked in documents map
                }
            }
        }

        self.documents.insert(doc_id, fields);
        self.doc_count += 1;
        doc_id
    }

    /// Search for documents containing the given query term.
    ///
    /// Returns a list of matching document IDs sorted by relevance (term frequency).
    pub fn search(&self, query: &str) -> Vec<DocId> {
        let normalized = if self.config.case_fold {
            query.to_lowercase().trim().to_string()
        } else {
            query.trim().to_string()
        };

        if let Some(postings) = self.postings.get(&normalized) {
            let mut results: Vec<(DocId, u32)> = postings
                .iter()
                .map(|p| (p.doc_id, p.term_frequency))
                .collect();
            results.sort_by(|a, b| b.1.cmp(&a.1).then(a.0 .0.cmp(&b.0 .0)));
            results.into_iter().map(|(id, _)| id).collect()
        } else {
            Vec::new()
        }
    }

    /// Search for documents matching all query terms (AND query).
    pub fn search_all(&self, terms: &[&str]) -> Vec<DocId> {
        if terms.is_empty() {
            return Vec::new();
        }
        let mut result_sets: Vec<HashSet<DocId>> = Vec::new();
        for term in terms {
            let docs: HashSet<DocId> = self.search(term).into_iter().collect();
            result_sets.push(docs);
        }
        let mut intersection = result_sets[0].clone();
        for set in &result_sets[1..] {
            intersection = intersection.intersection(set).copied().collect();
        }
        let mut result: Vec<DocId> = intersection.into_iter().collect();
        result.sort_by_key(|d| d.0);
        result
    }

    /// Get the stored document fields for a document ID.
    pub fn get_document(&self, doc_id: DocId) -> Option<&HashMap<String, String>> {
        self.documents.get(&doc_id)
    }

    /// Return the total number of indexed documents.
    pub fn doc_count(&self) -> u64 {
        self.doc_count
    }

    /// Return the total number of unique terms in the index.
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    /// Get the document frequency (number of docs containing the term).
    pub fn doc_frequency(&self, term: &str) -> usize {
        let normalized = if self.config.case_fold {
            term.to_lowercase()
        } else {
            term.to_string()
        };
        self.postings.get(&normalized).map_or(0, |p| p.len())
    }

    /// Remove a document from the index.
    pub fn remove_document(&mut self, doc_id: DocId) -> bool {
        if self.documents.remove(&doc_id).is_some() {
            // Remove all postings for this doc
            self.postings.retain(|_, postings| {
                postings.retain(|p| p.doc_id != doc_id);
                !postings.is_empty()
            });
            self.doc_count -= 1;
            true
        } else {
            false
        }
    }

    /// Clear the entire index.
    pub fn clear(&mut self) {
        self.postings.clear();
        self.documents.clear();
        self.doc_count = 0;
    }

    /// Tokenize a text value into individual terms.
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Check whether a term should be indexed based on configuration.
    fn should_index(&self, term: &str) -> bool {
        let len = term.len();
        if len < self.config.min_term_length || len > self.config.max_term_length {
            return false;
        }
        let lower = term.to_lowercase();
        !self.config.stop_words.contains(&lower)
    }
}

impl Default for MetadataIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fields(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_add_and_search_single_term() {
        let mut idx = MetadataIndex::new();
        idx.add_document(make_fields(&[("title", "Summer Breeze")]));
        let results = idx.search("summer");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], DocId(0));
    }

    #[test]
    fn test_search_returns_empty_for_missing_term() {
        let mut idx = MetadataIndex::new();
        idx.add_document(make_fields(&[("title", "Hello World")]));
        let results = idx.search("missing");
        assert!(results.is_empty());
    }

    #[test]
    fn test_case_insensitive_search() {
        let mut idx = MetadataIndex::new();
        idx.add_document(make_fields(&[("title", "UPPERCASE title")]));
        let results = idx.search("uppercase");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_multiple_documents() {
        let mut idx = MetadataIndex::new();
        idx.add_document(make_fields(&[("title", "Song Alpha")]));
        idx.add_document(make_fields(&[("title", "Song Beta")]));
        idx.add_document(make_fields(&[("title", "Alpha Beta")]));
        let results = idx.search("alpha");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_all_and_query() {
        let mut idx = MetadataIndex::new();
        idx.add_document(make_fields(&[("title", "Song Alpha Beta")]));
        idx.add_document(make_fields(&[("title", "Song Alpha")]));
        idx.add_document(make_fields(&[("title", "Song Beta")]));
        let results = idx.search_all(&["alpha", "beta"]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], DocId(0));
    }

    #[test]
    fn test_doc_count_and_term_count() {
        let mut idx = MetadataIndex::new();
        assert_eq!(idx.doc_count(), 0);
        assert_eq!(idx.term_count(), 0);
        idx.add_document(make_fields(&[("title", "Word1 Word2")]));
        assert_eq!(idx.doc_count(), 1);
        assert!(idx.term_count() >= 2);
    }

    #[test]
    fn test_doc_frequency() {
        let mut idx = MetadataIndex::new();
        idx.add_document(make_fields(&[("title", "Rust programming")]));
        idx.add_document(make_fields(&[("title", "Rust language")]));
        idx.add_document(make_fields(&[("title", "Python language")]));
        assert_eq!(idx.doc_frequency("rust"), 2);
        assert_eq!(idx.doc_frequency("language"), 2);
        assert_eq!(idx.doc_frequency("python"), 1);
    }

    #[test]
    fn test_remove_document() {
        let mut idx = MetadataIndex::new();
        let id = idx.add_document(make_fields(&[("title", "Remove me")]));
        assert_eq!(idx.doc_count(), 1);
        assert!(idx.remove_document(id));
        assert_eq!(idx.doc_count(), 0);
        assert!(idx.search("remove").is_empty());
    }

    #[test]
    fn test_remove_nonexistent_document() {
        let mut idx = MetadataIndex::new();
        assert!(!idx.remove_document(DocId(999)));
    }

    #[test]
    fn test_clear_index() {
        let mut idx = MetadataIndex::new();
        idx.add_document(make_fields(&[("title", "First")]));
        idx.add_document(make_fields(&[("title", "Second")]));
        idx.clear();
        assert_eq!(idx.doc_count(), 0);
        assert_eq!(idx.term_count(), 0);
    }

    #[test]
    fn test_get_document() {
        let mut idx = MetadataIndex::new();
        let id = idx.add_document(make_fields(&[("title", "My Song")]));
        let doc = idx.get_document(id).expect("should succeed in test");
        assert_eq!(doc.get("title").expect("should succeed in test"), "My Song");
    }

    #[test]
    fn test_stop_words_excluded() {
        let mut idx = MetadataIndex::new();
        idx.add_document(make_fields(&[("title", "The quick brown fox")]));
        // "the" is a stop word and should not be indexed
        assert!(idx.search("the").is_empty());
        // "quick" should be found
        assert_eq!(idx.search("quick").len(), 1);
    }

    #[test]
    fn test_min_term_length_filter() {
        let mut idx = MetadataIndex::new();
        idx.add_document(make_fields(&[("title", "I am here")]));
        // "I" is length 1, below min_term_length of 2
        assert!(idx.search("i").is_empty());
        // "am" is length 2, exactly at min
        assert_eq!(idx.search("am").len(), 1);
    }

    #[test]
    fn test_term_frequency_ranking() {
        let mut idx = MetadataIndex::new();
        // Doc0: "rust" appears once
        idx.add_document(make_fields(&[("title", "rust programming")]));
        // Doc1: "rust" appears three times
        idx.add_document(make_fields(&[("title", "rust rust rust")]));
        let results = idx.search("rust");
        assert_eq!(results.len(), 2);
        // Higher TF document should come first
        assert_eq!(results[0], DocId(1));
    }

    #[test]
    fn test_default_config_values() {
        let config = IndexConfig::default();
        assert_eq!(config.min_term_length, 2);
        assert_eq!(config.max_term_length, 128);
        assert!(config.case_fold);
        assert!(config.stop_words.contains("the"));
    }
}

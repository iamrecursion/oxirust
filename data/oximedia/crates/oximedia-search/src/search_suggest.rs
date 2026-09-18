#![allow(dead_code)]
//! Autocomplete and search suggestion engine for `oximedia-search`.
//!
//! Provides prefix-based and context-aware query suggestions to help
//! users find media assets faster by completing partial queries.
//!
//! # Architecture
//!
//! The [`SearchSuggestor`] combines four suggestion sources:
//! - **Personal query history** (user-specific, highest priority)
//! - **Global popular queries** (corpus-wide frequency)
//! - **Indexed term trie** (frequency-weighted prefix trie from indexed documents)
//! - **Metadata-derived terms** (titles, tags, descriptions)
//!
//! The [`TermTrie`] provides efficient O(k) prefix lookups where k is the
//! prefix length, and collects completions ranked by document frequency.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Suggestion types
// ---------------------------------------------------------------------------

/// The kind of suggestion being offered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestType {
    /// A query the user has previously typed.
    HistoryCompletion,
    /// A popular query from the global corpus.
    PopularQuery,
    /// A term derived from indexed media metadata (title, tag, etc.).
    MetadataCompletion,
    /// A term from the frequency-weighted indexed term trie.
    IndexedTermCompletion,
    /// A spelling-corrected alternative.
    SpellCorrection,
}

/// A single autocomplete suggestion.
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// The full suggested query string.
    pub text: String,
    /// How the suggestion was generated.
    pub suggest_type: SuggestType,
    /// Confidence / relevance score in `[0.0, 1.0]`.
    pub score: f32,
    /// Number of times this query has been observed (for ranking).
    pub frequency: u32,
}

impl Suggestion {
    /// Create a new suggestion.
    pub fn new(
        text: impl Into<String>,
        suggest_type: SuggestType,
        score: f32,
        frequency: u32,
    ) -> Self {
        Self {
            text: text.into(),
            suggest_type,
            score: score.clamp(0.0, 1.0),
            frequency,
        }
    }

    /// Returns `true` if this suggestion came from query history.
    #[must_use]
    pub fn is_history(&self) -> bool {
        self.suggest_type == SuggestType::HistoryCompletion
    }

    /// Returns `true` if this suggestion was generated from metadata.
    #[must_use]
    pub fn is_metadata(&self) -> bool {
        self.suggest_type == SuggestType::MetadataCompletion
    }

    /// Returns `true` if this suggestion came from the indexed term trie.
    #[must_use]
    pub fn is_indexed_term(&self) -> bool {
        self.suggest_type == SuggestType::IndexedTermCompletion
    }
}

// ---------------------------------------------------------------------------
// Term Trie - frequency-weighted prefix tree
// ---------------------------------------------------------------------------

/// A node in the frequency-weighted prefix trie.
#[derive(Debug, Clone, Default)]
struct TrieNode {
    /// Children keyed by character.
    children: HashMap<char, TrieNode>,
    /// If this node represents a complete term, the document frequency.
    /// `None` means this is just a prefix node.
    frequency: Option<u32>,
    /// The complete term stored at this node (only set when `frequency.is_some()`).
    term: Option<String>,
}

/// A frequency-weighted prefix trie for efficient autocomplete over indexed terms.
///
/// Each leaf stores the document frequency of the term so that completions
/// can be ranked by how common they are in the index.
#[derive(Debug, Clone, Default)]
pub struct TermTrie {
    root: TrieNode,
    /// Total number of distinct terms in the trie.
    term_count: usize,
}

impl TermTrie {
    /// Create an empty trie.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a term with its document frequency.
    ///
    /// If the term already exists, the frequency is updated to the higher value.
    pub fn insert(&mut self, term: &str, frequency: u32) {
        let lower = term.to_lowercase();
        let mut node = &mut self.root;
        for ch in lower.chars() {
            node = node.children.entry(ch).or_default();
        }
        let is_new = node.frequency.is_none();
        let current_freq = node.frequency.get_or_insert(0);
        if frequency > *current_freq {
            *current_freq = frequency;
        }
        node.term = Some(lower);
        if is_new {
            self.term_count += 1;
        }
    }

    /// Increment the frequency of an existing term by `delta`, or insert with `delta`.
    pub fn increment(&mut self, term: &str, delta: u32) {
        let lower = term.to_lowercase();
        let mut node = &mut self.root;
        for ch in lower.chars() {
            node = node.children.entry(ch).or_default();
        }
        let is_new = node.frequency.is_none();
        *node.frequency.get_or_insert(0) += delta;
        node.term = Some(lower);
        if is_new {
            self.term_count += 1;
        }
    }

    /// Find all completions for the given prefix, sorted by frequency descending.
    ///
    /// Returns up to `limit` results as `(term, frequency)` pairs.
    #[must_use]
    pub fn completions(&self, prefix: &str, limit: usize) -> Vec<(String, u32)> {
        let lower = prefix.to_lowercase();
        let mut node = &self.root;
        for ch in lower.chars() {
            match node.children.get(&ch) {
                Some(child) => node = child,
                None => return Vec::new(),
            }
        }

        // Collect all terminal nodes under this prefix.
        let mut results: Vec<(String, u32)> = Vec::new();
        Self::collect_terms(node, &mut results);

        // Sort by frequency descending, then alphabetically.
        results.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        results.truncate(limit);
        results
    }

    /// Recursively collect all terminal terms from a node.
    fn collect_terms(node: &TrieNode, results: &mut Vec<(String, u32)>) {
        if let (Some(ref term), Some(freq)) = (&node.term, node.frequency) {
            results.push((term.clone(), freq));
        }
        for child in node.children.values() {
            Self::collect_terms(child, results);
        }
    }

    /// Return the frequency of an exact term, or `None` if not present.
    #[must_use]
    pub fn get_frequency(&self, term: &str) -> Option<u32> {
        let lower = term.to_lowercase();
        let mut node = &self.root;
        for ch in lower.chars() {
            match node.children.get(&ch) {
                Some(child) => node = child,
                None => return None,
            }
        }
        node.frequency
    }

    /// Number of distinct terms in the trie.
    #[must_use]
    pub fn len(&self) -> usize {
        self.term_count
    }

    /// Whether the trie is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.term_count == 0
    }

    /// Remove all terms from the trie.
    pub fn clear(&mut self) {
        self.root = TrieNode::default();
        self.term_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Suggestor configuration
// ---------------------------------------------------------------------------

/// Configuration for the `SearchSuggestor`.
#[derive(Debug, Clone)]
pub struct SuggestorConfig {
    /// Maximum number of suggestions to return.
    pub max_results: usize,
    /// Minimum prefix length before suggestions are generated.
    pub min_prefix_len: usize,
    /// Whether to include history-based suggestions.
    pub include_history: bool,
    /// Whether to include metadata-derived suggestions.
    pub include_metadata: bool,
    /// Whether to include indexed-term trie suggestions.
    pub include_indexed_terms: bool,
}

impl Default for SuggestorConfig {
    fn default() -> Self {
        Self {
            max_results: 8,
            min_prefix_len: 2,
            include_history: true,
            include_metadata: true,
            include_indexed_terms: true,
        }
    }
}

// ---------------------------------------------------------------------------
// SearchSuggestor
// ---------------------------------------------------------------------------

/// Autocomplete and query suggestion engine.
///
/// Combines personal query history, global popular queries, indexed term trie,
/// and metadata terms to produce ranked suggestions for a given prefix.
#[derive(Debug)]
pub struct SearchSuggestor {
    /// User query history: query -> occurrence count.
    history: HashMap<String, u32>,
    /// Global popular queries: query -> occurrence count.
    popular: HashMap<String, u32>,
    /// Metadata-derived completion terms.
    metadata_terms: Vec<String>,
    /// Frequency-weighted trie built from indexed document terms.
    indexed_trie: TermTrie,
    /// Configuration.
    config: SuggestorConfig,
}

impl SearchSuggestor {
    /// Create a new, empty suggestor with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
            popular: HashMap::new(),
            metadata_terms: Vec::new(),
            indexed_trie: TermTrie::new(),
            config: SuggestorConfig::default(),
        }
    }

    /// Create a suggestor with a custom configuration.
    #[must_use]
    pub fn with_config(config: SuggestorConfig) -> Self {
        Self {
            history: HashMap::new(),
            popular: HashMap::new(),
            metadata_terms: Vec::new(),
            indexed_trie: TermTrie::new(),
            config,
        }
    }

    /// Record a user query in the personal history.
    pub fn record_query(&mut self, query: impl Into<String>) {
        *self.history.entry(query.into()).or_insert(0) += 1;
    }

    /// Add or update a global popular query.
    pub fn add_popular(&mut self, query: impl Into<String>, count: u32) {
        self.popular.insert(query.into(), count);
    }

    /// Add a metadata-derived completion term (e.g. a title or tag).
    pub fn add_metadata_term(&mut self, term: impl Into<String>) {
        self.metadata_terms.push(term.into());
    }

    /// Index a term from a document field with its document frequency.
    ///
    /// This feeds the frequency-weighted trie for fast prefix completion.
    pub fn index_term(&mut self, term: &str, doc_frequency: u32) {
        self.indexed_trie.insert(term, doc_frequency);
    }

    /// Index all words from a text string, each with frequency 1.
    ///
    /// Useful for bulk-indexing document titles, descriptions, and tags.
    pub fn index_text(&mut self, text: &str) {
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            if word.len() >= 3 {
                self.indexed_trie.increment(word, 1);
            }
        }
    }

    /// Return the indexed trie for inspection.
    #[must_use]
    pub fn indexed_trie(&self) -> &TermTrie {
        &self.indexed_trie
    }

    /// Return up to `config.max_results` suggestions for the given prefix.
    ///
    /// Suggestions are sorted by score descending, then alphabetically.
    /// Sources are merged with priority: history > popular > indexed > metadata.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn suggest(&self, prefix: &str) -> Vec<Suggestion> {
        let prefix_lower = prefix.to_lowercase();

        if prefix_lower.len() < self.config.min_prefix_len {
            return Vec::new();
        }

        let mut suggestions: Vec<Suggestion> = Vec::new();

        // History-based completions (highest priority base=1.0).
        if self.config.include_history {
            for (query, &freq) in &self.history {
                if query.to_lowercase().starts_with(&prefix_lower) {
                    let score = Self::score_completion(query.len(), freq, 1.0);
                    suggestions.push(Suggestion::new(
                        query.clone(),
                        SuggestType::HistoryCompletion,
                        score,
                        freq,
                    ));
                }
            }
        }

        // Popular-query completions (base=0.85).
        for (query, &freq) in &self.popular {
            if query.to_lowercase().starts_with(&prefix_lower) {
                let score = Self::score_completion(query.len(), freq, 0.85);
                suggestions.push(Suggestion::new(
                    query.clone(),
                    SuggestType::PopularQuery,
                    score,
                    freq,
                ));
            }
        }

        // Indexed term trie completions (base=0.75).
        if self.config.include_indexed_terms {
            let trie_results = self.indexed_trie.completions(
                &prefix_lower,
                self.config.max_results * 2, // fetch more, then merge
            );
            for (term, freq) in trie_results {
                let score = Self::score_completion(term.len(), freq, 0.75);
                suggestions.push(Suggestion::new(
                    term,
                    SuggestType::IndexedTermCompletion,
                    score,
                    freq,
                ));
            }
        }

        // Metadata-term completions (base=0.7).
        if self.config.include_metadata {
            for term in &self.metadata_terms {
                if term.to_lowercase().starts_with(&prefix_lower) {
                    let score = Self::score_completion(term.len(), 1, 0.7);
                    suggestions.push(Suggestion::new(
                        term.clone(),
                        SuggestType::MetadataCompletion,
                        score,
                        1,
                    ));
                }
            }
        }

        // Sort: score descending, then text ascending.
        suggestions.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.text.cmp(&b.text))
        });
        suggestions.dedup_by(|a, b| a.text.eq_ignore_ascii_case(&b.text));
        suggestions.truncate(self.config.max_results);
        suggestions
    }

    /// Score a completion candidate.
    ///
    /// Higher frequency and shorter completions (closer to the prefix) rank higher.
    #[allow(clippy::cast_precision_loss)]
    fn score_completion(query_len: usize, frequency: u32, base: f32) -> f32 {
        let length_factor = 1.0_f32 / (1.0 + (query_len as f32) * 0.05);
        let freq_factor = (frequency as f32).ln_1p() / 10.0;
        (base + freq_factor + length_factor).clamp(0.0, 1.0)
    }

    /// Returns the number of history entries.
    #[must_use]
    pub fn history_size(&self) -> usize {
        self.history.len()
    }

    /// Clear personal query history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Clear the indexed term trie.
    pub fn clear_indexed_terms(&mut self) {
        self.indexed_trie.clear();
    }
}

impl Default for SearchSuggestor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn base_suggestor() -> SearchSuggestor {
        let mut s = SearchSuggestor::new();
        s.record_query("video editor");
        s.record_query("video encoder");
        s.record_query("video encoder");
        s.add_popular("video stream", 500);
        s.add_popular("audio codec", 300);
        s.add_metadata_term("Video Production 2024");
        s
    }

    #[test]
    fn test_suggest_returns_vec() {
        let s = base_suggestor();
        let results = s.suggest("vid");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_suggest_respects_min_prefix_len() {
        let s = base_suggestor();
        let results = s.suggest("v"); // len 1, below default min of 2
        assert!(results.is_empty());
    }

    #[test]
    fn test_suggest_prefix_filtering() {
        let s = base_suggestor();
        let results = s.suggest("aud");
        for r in &results {
            assert!(r.text.to_lowercase().starts_with("aud"));
        }
    }

    #[test]
    fn test_suggest_history_type() {
        let s = base_suggestor();
        let results = s.suggest("video e");
        let has_history = results
            .iter()
            .any(|r| r.suggest_type == SuggestType::HistoryCompletion);
        assert!(has_history);
    }

    #[test]
    fn test_suggest_popular_type() {
        let s = base_suggestor();
        let results = s.suggest("video s");
        let has_popular = results
            .iter()
            .any(|r| r.suggest_type == SuggestType::PopularQuery);
        assert!(has_popular);
    }

    #[test]
    fn test_suggest_metadata_type() {
        let s = base_suggestor();
        let results = s.suggest("Video P");
        let has_meta = results
            .iter()
            .any(|r| r.suggest_type == SuggestType::MetadataCompletion);
        assert!(has_meta);
    }

    #[test]
    fn test_suggest_max_results_respected() {
        let mut s = SearchSuggestor::with_config(SuggestorConfig {
            max_results: 3,
            min_prefix_len: 1,
            include_history: true,
            include_metadata: true,
            include_indexed_terms: true,
        });
        for i in 0..20 {
            s.add_popular(format!("video clip {}", i), 100);
        }
        let results = s.suggest("v");
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_suggest_sorted_by_score_desc() {
        let s = base_suggestor();
        let results = s.suggest("video");
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_record_query_increments_history() {
        let mut s = SearchSuggestor::new();
        s.record_query("codec");
        s.record_query("codec");
        assert_eq!(s.history_size(), 1);
        assert_eq!(*s.history.get("codec").expect("should succeed in test"), 2);
    }

    #[test]
    fn test_clear_history() {
        let mut s = SearchSuggestor::new();
        s.record_query("something");
        s.clear_history();
        assert_eq!(s.history_size(), 0);
    }

    #[test]
    fn test_suggestion_is_history() {
        let sug = Suggestion::new("video editor", SuggestType::HistoryCompletion, 0.9, 5);
        assert!(sug.is_history());
        assert!(!sug.is_metadata());
    }

    #[test]
    fn test_suggestion_is_metadata() {
        let sug = Suggestion::new("4K Footage", SuggestType::MetadataCompletion, 0.7, 1);
        assert!(sug.is_metadata());
        assert!(!sug.is_history());
    }

    #[test]
    fn test_suggestion_score_clamped() {
        let sug = Suggestion::new("x", SuggestType::PopularQuery, 5.0, 1);
        assert!(sug.score <= 1.0);
    }

    #[test]
    fn test_no_metadata_when_disabled() {
        let mut s = SearchSuggestor::with_config(SuggestorConfig {
            max_results: 10,
            min_prefix_len: 1,
            include_history: false,
            include_metadata: false,
            include_indexed_terms: false,
        });
        s.add_metadata_term("special clip");
        s.add_popular("special", 100);
        let results = s.suggest("spec");
        let has_meta = results
            .iter()
            .any(|r| r.suggest_type == SuggestType::MetadataCompletion);
        assert!(!has_meta);
    }

    #[test]
    fn test_empty_prefix_returns_nothing_when_below_min() {
        let s = SearchSuggestor::new();
        assert!(s.suggest("").is_empty());
    }

    #[test]
    fn test_no_results_for_unmatched_prefix() {
        let s = base_suggestor();
        let results = s.suggest("zzzzzz");
        assert!(results.is_empty());
    }

    // ── TermTrie tests ──

    #[test]
    fn test_trie_empty() {
        let trie = TermTrie::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn test_trie_insert_and_lookup() {
        let mut trie = TermTrie::new();
        trie.insert("video", 42);
        assert_eq!(trie.len(), 1);
        assert_eq!(trie.get_frequency("video"), Some(42));
        assert_eq!(trie.get_frequency("audio"), None);
    }

    #[test]
    fn test_trie_case_insensitive() {
        let mut trie = TermTrie::new();
        trie.insert("Video", 10);
        assert_eq!(trie.get_frequency("VIDEO"), Some(10));
        assert_eq!(trie.get_frequency("video"), Some(10));
    }

    #[test]
    fn test_trie_insert_updates_higher_freq() {
        let mut trie = TermTrie::new();
        trie.insert("codec", 5);
        trie.insert("codec", 20);
        assert_eq!(trie.get_frequency("codec"), Some(20));
        // Lower frequency does not overwrite
        trie.insert("codec", 3);
        assert_eq!(trie.get_frequency("codec"), Some(20));
        assert_eq!(trie.len(), 1); // still just one term
    }

    #[test]
    fn test_trie_increment() {
        let mut trie = TermTrie::new();
        trie.increment("frame", 1);
        trie.increment("frame", 1);
        trie.increment("frame", 3);
        assert_eq!(trie.get_frequency("frame"), Some(5));
    }

    #[test]
    fn test_trie_completions_basic() {
        let mut trie = TermTrie::new();
        trie.insert("video", 100);
        trie.insert("visual", 50);
        trie.insert("vignette", 10);
        trie.insert("audio", 200);

        let results = trie.completions("vi", 10);
        assert_eq!(results.len(), 3);
        // Sorted by frequency descending
        assert_eq!(results[0].0, "video");
        assert_eq!(results[0].1, 100);
        assert_eq!(results[1].0, "visual");
        assert_eq!(results[2].0, "vignette");
    }

    #[test]
    fn test_trie_completions_limit() {
        let mut trie = TermTrie::new();
        for i in 0..20 {
            trie.insert(&format!("term{i:02}"), i as u32);
        }
        let results = trie.completions("term", 5);
        assert_eq!(results.len(), 5);
        // Should be the top-5 by frequency
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn test_trie_completions_no_match() {
        let mut trie = TermTrie::new();
        trie.insert("hello", 1);
        let results = trie.completions("xyz", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_trie_clear() {
        let mut trie = TermTrie::new();
        trie.insert("a", 1);
        trie.insert("b", 2);
        trie.clear();
        assert!(trie.is_empty());
        assert!(trie.completions("a", 10).is_empty());
    }

    // ── Indexed term integration tests ──

    #[test]
    fn test_suggest_indexed_terms() {
        let mut s = SearchSuggestor::new();
        s.index_term("transcoding", 500);
        s.index_term("transparency", 200);
        s.index_term("transcode", 300);
        s.index_term("audio", 100);

        let results = s.suggest("trans");
        assert!(!results.is_empty());
        let has_indexed = results
            .iter()
            .any(|r| r.suggest_type == SuggestType::IndexedTermCompletion);
        assert!(has_indexed);
        // All results should start with "trans"
        for r in &results {
            assert!(r.text.to_lowercase().starts_with("trans"));
        }
    }

    #[test]
    fn test_suggest_indexed_terms_ranked_by_frequency() {
        let mut s = SearchSuggestor::with_config(SuggestorConfig {
            max_results: 10,
            min_prefix_len: 1,
            include_history: false,
            include_metadata: false,
            include_indexed_terms: true,
        });
        s.index_term("codec_avc", 10);
        s.index_term("codec_av1", 500);
        s.index_term("codec_vp9", 100);

        let results = s.suggest("codec");
        assert_eq!(results.len(), 3);
        // Highest frequency first
        assert!(results[0].text.contains("av1"));
    }

    #[test]
    fn test_index_text_tokenizes_and_indexes() {
        let mut s = SearchSuggestor::new();
        s.index_text("4K Video Editing Software for Professionals");
        // Words with 3+ chars should be in the trie
        assert!(s.indexed_trie().get_frequency("video").is_some());
        assert!(s.indexed_trie().get_frequency("editing").is_some());
        assert!(s.indexed_trie().get_frequency("software").is_some());
        assert!(s.indexed_trie().get_frequency("professionals").is_some());
        // Short words like "for" (3 chars) should be indexed
        assert!(s.indexed_trie().get_frequency("for").is_some());
    }

    #[test]
    fn test_index_text_increments_frequency() {
        let mut s = SearchSuggestor::new();
        s.index_text("video editing video production video");
        assert_eq!(s.indexed_trie().get_frequency("video"), Some(3));
        assert_eq!(s.indexed_trie().get_frequency("editing"), Some(1));
    }

    #[test]
    fn test_suggest_merges_all_sources() {
        let mut s = SearchSuggestor::new();
        s.record_query("video editor");
        s.add_popular("video stream", 500);
        s.index_term("video_encoding", 200);
        s.add_metadata_term("Video Production 2024");

        let results = s.suggest("video");
        assert!(results.len() >= 3); // at least from history, popular, indexed

        let types: Vec<&SuggestType> = results.iter().map(|r| &r.suggest_type).collect();
        assert!(types.contains(&&SuggestType::HistoryCompletion));
        assert!(types.contains(&&SuggestType::PopularQuery));
    }

    #[test]
    fn test_clear_indexed_terms() {
        let mut s = SearchSuggestor::new();
        s.index_term("something", 10);
        s.clear_indexed_terms();
        assert!(s.indexed_trie().is_empty());
    }

    #[test]
    fn test_suggestion_is_indexed_term() {
        let sug = Suggestion::new("codec", SuggestType::IndexedTermCompletion, 0.8, 100);
        assert!(sug.is_indexed_term());
        assert!(!sug.is_history());
        assert!(!sug.is_metadata());
    }

    #[test]
    fn test_no_indexed_terms_when_disabled() {
        let mut s = SearchSuggestor::with_config(SuggestorConfig {
            max_results: 10,
            min_prefix_len: 1,
            include_history: false,
            include_metadata: false,
            include_indexed_terms: false,
        });
        s.index_term("disabled_term", 1000);
        let results = s.suggest("dis");
        let has_indexed = results
            .iter()
            .any(|r| r.suggest_type == SuggestType::IndexedTermCompletion);
        assert!(!has_indexed);
    }

    #[test]
    fn test_trie_multiple_prefixes() {
        let mut trie = TermTrie::new();
        trie.insert("abc", 10);
        trie.insert("abd", 20);
        trie.insert("xyz", 30);

        let ab_results = trie.completions("ab", 10);
        assert_eq!(ab_results.len(), 2);
        // "abd" has higher frequency
        assert_eq!(ab_results[0].0, "abd");

        let x_results = trie.completions("x", 10);
        assert_eq!(x_results.len(), 1);
        assert_eq!(x_results[0].0, "xyz");
    }

    #[test]
    fn test_trie_exact_match_completion() {
        let mut trie = TermTrie::new();
        trie.insert("video", 100);
        trie.insert("video_editor", 50);

        // Exact match for "video" should also appear
        let results = trie.completions("video", 10);
        assert_eq!(results.len(), 2);
    }
}

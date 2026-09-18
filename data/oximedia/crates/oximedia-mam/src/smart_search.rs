//! Enhanced asset search with BM25-inspired scoring, faceted filtering, and
//! Jaccard-similarity recommendations.
//!
//! # Overview
//!
//! `SmartSearchIndex` maintains an in-memory inverted index over
//! `IndexedAsset` records.  The `SmartSearchIndex::search` method
//! implements a BM25-inspired ranking (TF × IDF approximation) over the
//! tokenised text of an asset's title, description, and tags.
//!
//! `SmartSearchIndex::similar_assets` computes pairwise Jaccard similarity
//! on the union of manual and auto tags.
//!
//! No C/Fortran dependencies — everything is pure Rust.

use std::collections::{HashMap, HashSet};

use crate::ai_tagging::AutoTag;

// ---------------------------------------------------------------------------
// Indexed asset record
// ---------------------------------------------------------------------------

/// A fully indexed asset document.
#[derive(Debug, Clone)]
pub struct IndexedAsset {
    /// Unique asset identifier.
    pub id: String,
    /// Manually curated tags.
    pub tags: Vec<String>,
    /// Automatically generated tags (with confidence scores).
    pub auto_tags: Vec<AutoTag>,
    /// Human-readable title.
    pub title: String,
    /// Free-text description.
    pub description: String,
    /// Duration in seconds, if applicable.
    pub duration_secs: Option<f64>,
    /// Frame width in pixels.
    pub width: Option<u32>,
    /// Container/format identifier.
    pub format: Option<String>,
    /// Collection memberships.
    pub collection_ids: Vec<String>,
    /// Unix timestamp of ingest.
    pub ingested_at: u64,
}

// ---------------------------------------------------------------------------
// Search filter
// ---------------------------------------------------------------------------

/// Structured filter applied *before* scoring during a search.
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    /// Asset must have **all** of these tags (manual or auto).
    pub tags: Vec<String>,
    /// Asset must have **at least one** of these tags.
    pub tags_any: Vec<String>,
    /// Asset must **not** have any of these tags.
    pub exclude_tags: Vec<String>,
    /// Minimum duration in seconds.
    pub min_duration_secs: Option<f64>,
    /// Maximum duration in seconds.
    pub max_duration_secs: Option<f64>,
    /// Minimum frame width in pixels.
    pub min_width: Option<u32>,
    /// Asset format must be one of these values (empty = accept all).
    pub formats: Vec<String>,
    /// Asset must belong to at least one of these collections (empty = accept all).
    pub collections: Vec<String>,
    /// Asset must have been ingested after this Unix timestamp (inclusive).
    pub ingested_after: Option<u64>,
    /// Asset must have been ingested before this Unix timestamp (exclusive).
    pub ingested_before: Option<u64>,
    /// For auto-tags: only consider tags at or above this confidence level.
    pub min_confidence: Option<f32>,
    /// Enable fuzzy matching with typo tolerance (Levenshtein distance).
    /// When `Some(n)`, query terms will also match tokens within edit distance
    /// `n`. Typical values: 1 (strict) or 2 (tolerant). `None` disables fuzzy
    /// matching (exact token match only).
    pub max_edit_distance: Option<usize>,
}

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------

/// A single result from [`SmartSearchIndex::search`] or
/// [`SmartSearchIndex::similar_assets`].
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Asset identifier.
    pub asset_id: String,
    /// Relevance / similarity score.
    pub score: f32,
    /// Tags that matched the query terms or similarity computation.
    pub matched_tags: Vec<String>,
    /// Metadata field names that contained query terms.
    pub matched_fields: Vec<String>,
}

// ---------------------------------------------------------------------------
// Internal scoring helpers
// ---------------------------------------------------------------------------

/// Tokenise a string into lower-case alphanumeric tokens.
fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Compute term frequency: count of `term` in `tokens` / total tokens.
fn tf(term: &str, tokens: &[String]) -> f32 {
    if tokens.is_empty() {
        return 0.0;
    }
    let count = tokens.iter().filter(|t| t.as_str() == term).count();
    count as f32 / tokens.len() as f32
}

/// IDF approximation: `log(1 + N / (1 + df))` where `df` is the document
/// frequency of the term.
fn idf(n: usize, df: usize) -> f32 {
    let n = n.max(1) as f32;
    let df = df as f32;
    (1.0 + n / (1.0 + df)).ln()
}

// ---------------------------------------------------------------------------
// Levenshtein distance and fuzzy helpers
// ---------------------------------------------------------------------------

/// Compute the Levenshtein edit distance between two strings.
///
/// Uses the classic Wagner-Fischer dynamic-programming algorithm with a
/// single-row optimisation (O(min(m,n)) space).
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    // Optimise so that the shorter string forms the column dimension.
    let (short, long) = if a_chars.len() <= b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };

    let short_len = short.len();
    let long_len = long.len();

    if short_len == 0 {
        return long_len;
    }

    let mut prev_row: Vec<usize> = (0..=short_len).collect();
    let mut curr_row: Vec<usize> = vec![0; short_len + 1];

    for i in 1..=long_len {
        curr_row[0] = i;
        for j in 1..=short_len {
            let cost = if long[i - 1] == short[j - 1] { 0 } else { 1 };
            curr_row[j] = (prev_row[j] + 1)
                .min(curr_row[j - 1] + 1)
                .min(prev_row[j - 1] + cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[short_len]
}

/// Default maximum edit distance for fuzzy matching.
const DEFAULT_MAX_EDIT_DISTANCE: usize = 2;

/// Return `true` if `candidate` is within `max_dist` edits of `query_term`
/// (both assumed lower-case).
fn is_fuzzy_match(query_term: &str, candidate: &str, max_dist: usize) -> bool {
    // Quick length-difference pruning.
    let len_diff = if query_term.len() > candidate.len() {
        query_term.len() - candidate.len()
    } else {
        candidate.len() - query_term.len()
    };
    if len_diff > max_dist {
        return false;
    }
    levenshtein_distance(query_term, candidate) <= max_dist
}

/// Compute fuzzy term frequency: for each token in `tokens`, if it is within
/// `max_dist` edits of `term`, it counts as a (discounted) match.
fn fuzzy_tf(term: &str, tokens: &[String], max_dist: usize) -> f32 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut score = 0.0_f32;
    for tok in tokens {
        let dist = levenshtein_distance(term, tok.as_str());
        if dist == 0 {
            score += 1.0;
        } else if dist <= max_dist {
            // Discount proportional to distance: 1 edit => 0.7, 2 edits => 0.4
            score += 1.0 - (dist as f32 * 0.3).min(0.9);
        }
    }
    score / tokens.len() as f32
}

// ---------------------------------------------------------------------------
// Internal per-asset document representation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct IndexedDocument {
    /// The stored asset.
    asset: IndexedAsset,
    /// Pre-tokenised title tokens.
    title_tokens: Vec<String>,
    /// Pre-tokenised description tokens.
    desc_tokens: Vec<String>,
    /// All tag strings (manual + auto above filter threshold).
    all_tag_strings: Vec<String>,
}

impl IndexedDocument {
    fn from_asset(asset: IndexedAsset) -> Self {
        let title_tokens = tokenise(&asset.title);
        let desc_tokens = tokenise(&asset.description);
        let all_tag_strings = {
            let mut v: Vec<String> = asset.tags.iter().map(|t| t.to_lowercase()).collect();
            for at in &asset.auto_tags {
                v.push(at.tag.to_lowercase());
            }
            v.sort();
            v.dedup();
            v
        };
        Self {
            asset,
            title_tokens,
            desc_tokens,
            all_tag_strings,
        }
    }

    /// Combined tag set for Jaccard similarity.
    fn tag_set(&self) -> HashSet<&str> {
        self.all_tag_strings.iter().map(|s| s.as_str()).collect()
    }

    /// Effective auto-tag labels after applying `min_confidence`.
    fn auto_tag_labels(&self, min_confidence: Option<f32>) -> Vec<String> {
        self.asset
            .auto_tags
            .iter()
            .filter(|at| min_confidence.map(|mc| at.confidence >= mc).unwrap_or(true))
            .map(|at| at.tag.to_lowercase())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SmartSearchIndex
// ---------------------------------------------------------------------------

/// In-memory search index with BM25-inspired scoring and tag-based similarity.
pub struct SmartSearchIndex {
    /// asset_id → document.
    assets: HashMap<String, IndexedDocument>,
}

impl SmartSearchIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            assets: HashMap::new(),
        }
    }

    /// Add or replace an asset in the index.
    pub fn index_asset(&mut self, asset: IndexedAsset) {
        let id = asset.id.clone();
        self.assets.insert(id, IndexedDocument::from_asset(asset));
    }

    /// Remove an asset from the index.  Returns `true` if it existed.
    pub fn remove_asset(&mut self, id: &str) -> bool {
        self.assets.remove(id).is_some()
    }

    /// Number of assets currently in the index.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// The `n` most frequent tags across all indexed assets, with counts.
    #[must_use]
    pub fn top_tags(&self, n: usize) -> Vec<(String, usize)> {
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for doc in self.assets.values() {
            for tag in &doc.all_tag_strings {
                *freq.entry(tag.as_str()).or_insert(0) += 1;
            }
        }
        let mut pairs: Vec<(String, usize)> =
            freq.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(n);
        pairs
    }

    // -----------------------------------------------------------------------
    // Filter predicate
    // -----------------------------------------------------------------------

    /// Returns `true` if the document passes every condition in `filter`.
    fn passes_filter(doc: &IndexedDocument, filter: &SearchFilter) -> bool {
        // Build effective tag set respecting min_confidence.
        let manual_tags: HashSet<String> =
            doc.asset.tags.iter().map(|t| t.to_lowercase()).collect();
        let auto_labels: HashSet<String> = doc
            .auto_tag_labels(filter.min_confidence)
            .into_iter()
            .collect();
        let effective_tags: HashSet<&str> = manual_tags
            .iter()
            .map(|s| s.as_str())
            .chain(auto_labels.iter().map(|s| s.as_str()))
            .collect();

        // Must have ALL of filter.tags.
        for required in &filter.tags {
            if !effective_tags.contains(required.to_lowercase().as_str()) {
                return false;
            }
        }

        // Must have ANY of filter.tags_any (if non-empty).
        if !filter.tags_any.is_empty() {
            let any_match = filter
                .tags_any
                .iter()
                .any(|t| effective_tags.contains(t.to_lowercase().as_str()));
            if !any_match {
                return false;
            }
        }

        // Must NOT have excluded tags.
        for excluded in &filter.exclude_tags {
            if effective_tags.contains(excluded.to_lowercase().as_str()) {
                return false;
            }
        }

        // Duration bounds.
        if let Some(min_d) = filter.min_duration_secs {
            match doc.asset.duration_secs {
                Some(d) if d >= min_d => {}
                _ => return false,
            }
        }
        if let Some(max_d) = filter.max_duration_secs {
            match doc.asset.duration_secs {
                Some(d) if d <= max_d => {}
                _ => return false,
            }
        }

        // Width bound.
        if let Some(min_w) = filter.min_width {
            match doc.asset.width {
                Some(w) if w >= min_w => {}
                _ => return false,
            }
        }

        // Format allowlist.
        if !filter.formats.is_empty() {
            let fmt = doc.asset.format.as_deref().unwrap_or("").to_lowercase();
            if !filter.formats.iter().any(|f| f.to_lowercase() == fmt) {
                return false;
            }
        }

        // Collection membership.
        if !filter.collections.is_empty() {
            let in_collection = filter
                .collections
                .iter()
                .any(|c| doc.asset.collection_ids.contains(c));
            if !in_collection {
                return false;
            }
        }

        // Ingest timestamp range.
        if let Some(after) = filter.ingested_after {
            if doc.asset.ingested_at < after {
                return false;
            }
        }
        if let Some(before) = filter.ingested_before {
            if doc.asset.ingested_at >= before {
                return false;
            }
        }

        true
    }

    // -----------------------------------------------------------------------
    // BM25-inspired search
    // -----------------------------------------------------------------------

    /// Search the index with a free-text `query` and structured `filter`.
    ///
    /// Scoring uses a TF × IDF approximation over title (weight ×3),
    /// description (weight ×1), and tags (weight ×2).
    ///
    /// Returns up to `limit` results sorted by score descending.
    #[must_use]
    pub fn search(&self, query: &str, filter: &SearchFilter, limit: usize) -> Vec<SearchResult> {
        let query_terms: Vec<String> = tokenise(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let n = self.assets.len();

        // Precompute document frequencies for each query term.
        let df: HashMap<&str, usize> = {
            let mut map = HashMap::new();
            for term in &query_terms {
                let count = self
                    .assets
                    .values()
                    .filter(|doc| {
                        doc.title_tokens.iter().any(|t| t == term)
                            || doc.desc_tokens.iter().any(|t| t == term)
                            || doc.all_tag_strings.iter().any(|t| t == term)
                    })
                    .count();
                map.insert(term.as_str(), count);
            }
            map
        };

        let mut results: Vec<SearchResult> = self
            .assets
            .values()
            .filter(|doc| Self::passes_filter(doc, filter))
            .filter_map(|doc| {
                let mut score = 0.0_f32;
                let mut matched_fields: Vec<String> = Vec::new();
                let mut matched_tags: Vec<String> = Vec::new();

                // Build the effective tag set for scoring, respecting min_confidence
                // so that low-confidence auto-tags don't contribute to term matches.
                let effective_scoring_tags: Vec<String> = {
                    let mut v: Vec<String> =
                        doc.asset.tags.iter().map(|t| t.to_lowercase()).collect();
                    for at in &doc.asset.auto_tags {
                        let meets_threshold = filter
                            .min_confidence
                            .map(|mc| at.confidence >= mc)
                            .unwrap_or(true);
                        if meets_threshold {
                            v.push(at.tag.to_lowercase());
                        }
                    }
                    v.sort();
                    v.dedup();
                    v
                };

                for term in &query_terms {
                    let idf_val = idf(n, *df.get(term.as_str()).unwrap_or(&0));

                    // Title (weight 3).
                    let tf_title = tf(term, &doc.title_tokens);
                    if tf_title > 0.0 {
                        score += tf_title * idf_val * 3.0;
                        if !matched_fields.contains(&"title".to_string()) {
                            matched_fields.push("title".to_string());
                        }
                    }

                    // Description (weight 1).
                    let tf_desc = tf(term, &doc.desc_tokens);
                    if tf_desc > 0.0 {
                        score += tf_desc * idf_val;
                        if !matched_fields.contains(&"description".to_string()) {
                            matched_fields.push("description".to_string());
                        }
                    }

                    // Tags (weight 2) — only tags meeting min_confidence threshold.
                    for tag in &effective_scoring_tags {
                        let tag_tokens = tokenise(tag);
                        let tf_tag = tf(term, &tag_tokens);
                        if tf_tag > 0.0 {
                            score += tf_tag * idf_val * 2.0;
                            if !matched_tags.contains(tag) {
                                matched_tags.push(tag.clone());
                            }
                            if !matched_fields.contains(&"tags".to_string()) {
                                matched_fields.push("tags".to_string());
                            }
                        }
                    }
                }

                if score > 0.0 {
                    Some(SearchResult {
                        asset_id: doc.asset.id.clone(),
                        score,
                        matched_tags,
                        matched_fields,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.asset_id.cmp(&b.asset_id))
        });
        results.truncate(limit);
        results
    }

    // -----------------------------------------------------------------------
    // Fuzzy search (Levenshtein-tolerant)
    // -----------------------------------------------------------------------

    /// Search with fuzzy matching: query terms are matched against document
    /// tokens using Levenshtein distance, so minor typos still produce
    /// results.
    ///
    /// `max_edit_distance` controls how many character edits are tolerated
    /// (default 2 if `None`).  The rest of the parameters behave identically
    /// to [`Self::search`].
    #[must_use]
    pub fn fuzzy_search(
        &self,
        query: &str,
        filter: &SearchFilter,
        limit: usize,
        max_edit_distance: Option<usize>,
    ) -> Vec<SearchResult> {
        let query_terms: Vec<String> = tokenise(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let max_dist = max_edit_distance.unwrap_or(DEFAULT_MAX_EDIT_DISTANCE);
        let n = self.assets.len();

        // Precompute fuzzy document frequencies.
        let df: HashMap<&str, usize> = {
            let mut map = HashMap::new();
            for term in &query_terms {
                let count = self
                    .assets
                    .values()
                    .filter(|doc| {
                        doc.title_tokens
                            .iter()
                            .any(|t| is_fuzzy_match(term, t, max_dist))
                            || doc
                                .desc_tokens
                                .iter()
                                .any(|t| is_fuzzy_match(term, t, max_dist))
                            || doc
                                .all_tag_strings
                                .iter()
                                .any(|t| is_fuzzy_match(term, t, max_dist))
                    })
                    .count();
                map.insert(term.as_str(), count);
            }
            map
        };

        let mut results: Vec<SearchResult> = self
            .assets
            .values()
            .filter(|doc| Self::passes_filter(doc, filter))
            .filter_map(|doc| {
                let mut score = 0.0_f32;
                let mut matched_fields: Vec<String> = Vec::new();
                let mut matched_tags: Vec<String> = Vec::new();

                let effective_scoring_tags: Vec<String> = {
                    let mut v: Vec<String> =
                        doc.asset.tags.iter().map(|t| t.to_lowercase()).collect();
                    for at in &doc.asset.auto_tags {
                        let meets = filter
                            .min_confidence
                            .map(|mc| at.confidence >= mc)
                            .unwrap_or(true);
                        if meets {
                            v.push(at.tag.to_lowercase());
                        }
                    }
                    v.sort();
                    v.dedup();
                    v
                };

                for term in &query_terms {
                    let idf_val = idf(n, *df.get(term.as_str()).unwrap_or(&0));

                    // Title (weight 3) -- fuzzy.
                    let tf_title = fuzzy_tf(term, &doc.title_tokens, max_dist);
                    if tf_title > 0.0 {
                        score += tf_title * idf_val * 3.0;
                        if !matched_fields.contains(&"title".to_string()) {
                            matched_fields.push("title".to_string());
                        }
                    }

                    // Description (weight 1) -- fuzzy.
                    let tf_desc = fuzzy_tf(term, &doc.desc_tokens, max_dist);
                    if tf_desc > 0.0 {
                        score += tf_desc * idf_val;
                        if !matched_fields.contains(&"description".to_string()) {
                            matched_fields.push("description".to_string());
                        }
                    }

                    // Tags (weight 2) -- fuzzy.
                    for tag in &effective_scoring_tags {
                        let tag_tokens = tokenise(tag);
                        let tf_tag = fuzzy_tf(term, &tag_tokens, max_dist);
                        if tf_tag > 0.0 {
                            score += tf_tag * idf_val * 2.0;
                            if !matched_tags.contains(tag) {
                                matched_tags.push(tag.clone());
                            }
                            if !matched_fields.contains(&"tags".to_string()) {
                                matched_fields.push("tags".to_string());
                            }
                        }
                    }
                }

                if score > 0.0 {
                    Some(SearchResult {
                        asset_id: doc.asset.id.clone(),
                        score,
                        matched_tags,
                        matched_fields,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.asset_id.cmp(&b.asset_id))
        });
        results.truncate(limit);
        results
    }

    // -----------------------------------------------------------------------
    // Similarity
    // -----------------------------------------------------------------------

    /// Find assets most similar to `asset_id` using Jaccard similarity on
    /// the union of manual and auto tags.
    ///
    /// Returns up to `limit` results (excluding the query asset itself),
    /// sorted by score descending.
    #[must_use]
    pub fn similar_assets(&self, asset_id: &str, limit: usize) -> Vec<SearchResult> {
        let query_doc = match self.assets.get(asset_id) {
            Some(d) => d,
            None => return Vec::new(),
        };
        let query_tags = query_doc.tag_set();

        let mut results: Vec<SearchResult> = self
            .assets
            .values()
            .filter(|doc| doc.asset.id != asset_id)
            .filter_map(|doc| {
                let candidate_tags = doc.tag_set();
                let intersection: Vec<&str> = query_tags
                    .iter()
                    .filter(|&&t| candidate_tags.contains(t))
                    .copied()
                    .collect();
                let union_size = query_tags.len() + candidate_tags.len() - intersection.len();
                if union_size == 0 {
                    return None;
                }
                let jaccard = intersection.len() as f32 / union_size as f32;
                if jaccard == 0.0 {
                    return None;
                }
                Some(SearchResult {
                    asset_id: doc.asset.id.clone(),
                    score: jaccard,
                    matched_tags: intersection.iter().map(|s| s.to_string()).collect(),
                    matched_fields: vec!["tags".to_string()],
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.asset_id.cmp(&b.asset_id))
        });
        results.truncate(limit);
        results
    }
}

impl Default for SmartSearchIndex {
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
    use crate::ai_tagging::{AutoTag, TagCategory};

    fn make_asset(id: &str, title: &str, tags: &[&str]) -> IndexedAsset {
        IndexedAsset {
            id: id.to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            auto_tags: Vec::new(),
            title: title.to_string(),
            description: String::new(),
            duration_secs: None,
            width: None,
            format: None,
            collection_ids: Vec::new(),
            ingested_at: 1_000_000,
        }
    }

    fn make_auto_tag(tag: &str, confidence: f32) -> AutoTag {
        AutoTag {
            tag: tag.to_string(),
            confidence,
            category: TagCategory::Technical,
            source: "rule-based".to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // Indexing
    // -----------------------------------------------------------------------

    #[test]
    fn test_index_and_count() {
        let mut idx = SmartSearchIndex::new();
        assert_eq!(idx.asset_count(), 0);
        idx.index_asset(make_asset("a1", "Alpha video", &["av1", "hd"]));
        assert_eq!(idx.asset_count(), 1);
        idx.index_asset(make_asset("a2", "Beta video", &["opus"]));
        assert_eq!(idx.asset_count(), 2);
    }

    #[test]
    fn test_remove_asset() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Alpha", &[]));
        assert!(idx.remove_asset("a1"));
        assert!(!idx.remove_asset("a1")); // already gone
        assert_eq!(idx.asset_count(), 0);
    }

    #[test]
    fn test_replace_asset() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Original title", &["old-tag"]));
        idx.index_asset(make_asset("a1", "Updated title", &["new-tag"]));
        assert_eq!(idx.asset_count(), 1);
        let results = idx.search("updated", &SearchFilter::default(), 10);
        assert!(!results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Search — basic query matching
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_title_match() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "The quick brown fox", &[]));
        idx.index_asset(make_asset("a2", "Lazy dog video", &[]));
        let results = idx.search("quick", &SearchFilter::default(), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "a1");
        assert!(results[0].matched_fields.contains(&"title".to_string()));
    }

    #[test]
    fn test_search_tag_match() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset(
            "a1",
            "Nature documentary",
            &["outdoors", "wildlife"],
        ));
        idx.index_asset(make_asset("a2", "City tour", &["urban", "cityscape"]));
        let results = idx.search("wildlife", &SearchFilter::default(), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "a1");
    }

    #[test]
    fn test_search_no_match_returns_empty() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Some video", &[]));
        let results = idx.search("nonexistent_xyzzy", &SearchFilter::default(), 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_respects_limit() {
        let mut idx = SmartSearchIndex::new();
        for i in 0..10 {
            idx.index_asset(make_asset(&format!("a{i}"), "common keyword here", &[]));
        }
        let results = idx.search("keyword", &SearchFilter::default(), 3);
        assert!(results.len() <= 3);
    }

    // -----------------------------------------------------------------------
    // Search — filters
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_tags_all() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Both tags", &["hd", "av1"]));
        idx.index_asset(make_asset("a2", "One tag", &["hd"]));
        let filter = SearchFilter {
            tags: vec!["hd".into(), "av1".into()],
            ..Default::default()
        };
        let results = idx.search("tags", &filter, 10);
        // Only a1 passes the AND filter.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "a1");
    }

    #[test]
    fn test_filter_tags_any() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "AV1 video", &["av1"]));
        idx.index_asset(make_asset("a2", "VP9 video", &["vp9"]));
        idx.index_asset(make_asset("a3", "Other video", &["other"]));
        let filter = SearchFilter {
            tags_any: vec!["av1".into(), "vp9".into()],
            ..Default::default()
        };
        let results = idx.search("video", &filter, 10);
        let ids: Vec<_> = results.iter().map(|r| r.asset_id.as_str()).collect();
        assert!(ids.contains(&"a1"));
        assert!(ids.contains(&"a2"));
        assert!(!ids.contains(&"a3"));
    }

    #[test]
    fn test_filter_exclude_tags() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Good video", &["approved"]));
        idx.index_asset(make_asset("a2", "Rejected video", &["rejected"]));
        let filter = SearchFilter {
            exclude_tags: vec!["rejected".into()],
            ..Default::default()
        };
        let results = idx.search("video", &filter, 10);
        assert!(results.iter().all(|r| r.asset_id != "a2"));
    }

    #[test]
    fn test_filter_duration() {
        let mut idx = SmartSearchIndex::new();
        let mut asset_short = make_asset("a1", "Short clip", &[]);
        asset_short.duration_secs = Some(10.0);
        let mut asset_long = make_asset("a2", "Long clip", &[]);
        asset_long.duration_secs = Some(3600.0);
        idx.index_asset(asset_short);
        idx.index_asset(asset_long);

        let filter = SearchFilter {
            min_duration_secs: Some(60.0),
            ..Default::default()
        };
        let results = idx.search("clip", &filter, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "a2");
    }

    #[test]
    fn test_filter_format() {
        let mut idx = SmartSearchIndex::new();
        let mut a1 = make_asset("a1", "MP4 video", &[]);
        a1.format = Some("mp4".to_string());
        let mut a2 = make_asset("a2", "MKV video", &[]);
        a2.format = Some("mkv".to_string());
        idx.index_asset(a1);
        idx.index_asset(a2);

        let filter = SearchFilter {
            formats: vec!["mp4".into()],
            ..Default::default()
        };
        let results = idx.search("video", &filter, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "a1");
    }

    #[test]
    fn test_filter_ingested_range() {
        let mut idx = SmartSearchIndex::new();
        let mut a1 = make_asset("a1", "Old video", &[]);
        a1.ingested_at = 500;
        let mut a2 = make_asset("a2", "New video", &[]);
        a2.ingested_at = 2000;
        idx.index_asset(a1);
        idx.index_asset(a2);

        let filter = SearchFilter {
            ingested_after: Some(1000),
            ..Default::default()
        };
        let results = idx.search("video", &filter, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "a2");
    }

    #[test]
    fn test_filter_min_confidence_auto_tags() {
        let mut idx = SmartSearchIndex::new();
        let mut a1 = make_asset("a1", "Alpha footage", &[]);
        a1.auto_tags = vec![make_auto_tag("wildlife", 0.9)];
        let mut a2 = make_asset("a2", "Beta footage", &[]);
        a2.auto_tags = vec![make_auto_tag("wildlife", 0.2)];
        idx.index_asset(a1);
        idx.index_asset(a2);

        let filter = SearchFilter {
            min_confidence: Some(0.5),
            ..Default::default()
        };
        let results = idx.search("wildlife", &filter, 10);
        // a2's auto tag doesn't meet min_confidence so "wildlife" won't appear
        // in its effective tag set, meaning it won't match the query.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "a1");
    }

    // -----------------------------------------------------------------------
    // Similarity
    // -----------------------------------------------------------------------

    #[test]
    fn test_similar_assets_jaccard() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Ref", &["a", "b", "c"]));
        idx.index_asset(make_asset("a2", "Similar", &["a", "b", "d"]));
        idx.index_asset(make_asset("a3", "Different", &["x", "y", "z"]));

        let results = idx.similar_assets("a1", 10);
        assert!(!results.is_empty());
        // a2 should have higher score than a3 (or a3 score == 0 = not returned).
        let a2_score = results.iter().find(|r| r.asset_id == "a2").map(|r| r.score);
        let a3_score = results.iter().find(|r| r.asset_id == "a3").map(|r| r.score);
        assert!(a2_score.unwrap_or(0.0) > a3_score.unwrap_or(0.0));
    }

    #[test]
    fn test_similar_assets_excludes_self() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Asset", &["tag1"]));
        let results = idx.similar_assets("a1", 10);
        assert!(results.iter().all(|r| r.asset_id != "a1"));
    }

    #[test]
    fn test_similar_assets_unknown_id() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Asset", &["tag1"]));
        let results = idx.similar_assets("nonexistent", 10);
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Top tags
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_tags() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "A", &["common", "rare"]));
        idx.index_asset(make_asset("a2", "B", &["common"]));
        idx.index_asset(make_asset("a3", "C", &["common"]));

        let top = idx.top_tags(2);
        assert_eq!(top[0].0, "common");
        assert_eq!(top[0].1, 3);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_top_tags_empty_index() {
        let idx = SmartSearchIndex::new();
        assert!(idx.top_tags(5).is_empty());
    }

    #[test]
    fn test_top_tags_limit() {
        let mut idx = SmartSearchIndex::new();
        for i in 0..20 {
            idx.index_asset(make_asset(&format!("a{i}"), "T", &[&format!("tag{i}")]));
        }
        let top = idx.top_tags(5);
        assert_eq!(top.len(), 5);
    }

    // -----------------------------------------------------------------------
    // Levenshtein distance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_empty_strings() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "xyz"), 3);
    }

    #[test]
    fn test_levenshtein_single_edit() {
        assert_eq!(levenshtein_distance("cat", "bat"), 1); // substitution
        assert_eq!(levenshtein_distance("cat", "ca"), 1); // deletion
        assert_eq!(levenshtein_distance("cat", "cart"), 1); // insertion
    }

    #[test]
    fn test_levenshtein_two_edits() {
        assert_eq!(levenshtein_distance("kitten", "sittin"), 2);
    }

    #[test]
    fn test_levenshtein_classic() {
        // kitten -> sitting = 3 edits
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_levenshtein_symmetric() {
        assert_eq!(
            levenshtein_distance("abc", "xyz"),
            levenshtein_distance("xyz", "abc"),
        );
    }

    #[test]
    fn test_levenshtein_unicode() {
        assert_eq!(levenshtein_distance("cafe", "café"), 1);
    }

    // -----------------------------------------------------------------------
    // Fuzzy helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_fuzzy_match_exact() {
        assert!(is_fuzzy_match("video", "video", 2));
    }

    #[test]
    fn test_is_fuzzy_match_one_edit() {
        // "vido" is 1 deletion from "video"
        assert!(is_fuzzy_match("vido", "video", 1));
        // "bideo" is 1 substitution from "video"
        assert!(is_fuzzy_match("bideo", "video", 1));
        // "vidoe" is a transposition = 2 Levenshtein edits
        assert!(!is_fuzzy_match("vidoe", "video", 1));
        assert!(is_fuzzy_match("vidoe", "video", 2));
    }

    #[test]
    fn test_is_fuzzy_match_too_distant() {
        assert!(!is_fuzzy_match("abcde", "xyz", 2));
    }

    #[test]
    fn test_is_fuzzy_match_length_pruning() {
        // Length difference > max_dist => fast reject
        assert!(!is_fuzzy_match("a", "abcde", 2));
    }

    #[test]
    fn test_fuzzy_tf_exact_and_close() {
        let tokens = vec!["video".to_string(), "clip".to_string()];
        // Exact match
        let exact = fuzzy_tf("video", &tokens, 2);
        assert!(exact > 0.0);
        // Typo: "vdieo" is 2 edits from "video"
        let fuzzy = fuzzy_tf("vidoe", &tokens, 2);
        assert!(fuzzy > 0.0);
        // Exact should score higher
        assert!(exact > fuzzy);
    }

    #[test]
    fn test_fuzzy_tf_no_match() {
        let tokens = vec!["alpha".to_string()];
        assert_eq!(fuzzy_tf("zzzzz", &tokens, 1), 0.0);
    }

    #[test]
    fn test_fuzzy_tf_empty_tokens() {
        assert_eq!(fuzzy_tf("anything", &[], 2), 0.0);
    }

    // -----------------------------------------------------------------------
    // Fuzzy search integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fuzzy_search_finds_typo() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Wildlife documentary", &["nature"]));
        idx.index_asset(make_asset("a2", "City tour", &["urban"]));

        // "wildife" is 1 edit away from "wildlife"
        let results = idx.fuzzy_search("wildife", &SearchFilter::default(), 10, Some(1));
        assert!(!results.is_empty());
        assert_eq!(results[0].asset_id, "a1");
    }

    #[test]
    fn test_fuzzy_search_two_edits() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Documentary about nature", &[]));
        // "documentry" has 2 edits from "documentary"
        let results = idx.fuzzy_search("documentry", &SearchFilter::default(), 10, Some(2));
        assert!(!results.is_empty());
        assert_eq!(results[0].asset_id, "a1");
    }

    #[test]
    fn test_fuzzy_search_no_match_strict() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Ocean waves video", &[]));
        // "oxean" is 1 edit from "ocean", but with max_dist=0 => exact only
        let results = idx.fuzzy_search("oxean", &SearchFilter::default(), 10, Some(0));
        assert!(results.is_empty());
    }

    #[test]
    fn test_fuzzy_search_tag_typo() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Some video", &["landscape"]));
        // "landsape" is 1 edit from "landscape"
        let results = idx.fuzzy_search("landsape", &SearchFilter::default(), 10, Some(1));
        assert!(!results.is_empty());
        assert!(results[0].matched_tags.contains(&"landscape".to_string()));
    }

    #[test]
    fn test_fuzzy_search_respects_filter() {
        let mut idx = SmartSearchIndex::new();
        let mut a1 = make_asset("a1", "Summer trip", &["outdoors"]);
        a1.format = Some("mp4".to_string());
        let mut a2 = make_asset("a2", "Summer camp", &["outdoors"]);
        a2.format = Some("mkv".to_string());
        idx.index_asset(a1);
        idx.index_asset(a2);

        let filter = SearchFilter {
            formats: vec!["mp4".into()],
            ..Default::default()
        };
        // "sumer" is 1 edit from "summer"
        let results = idx.fuzzy_search("sumer", &filter, 10, Some(1));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "a1");
    }

    #[test]
    fn test_fuzzy_search_default_distance() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Architecture overview", &[]));
        // "architecure" is 1 edit from "architecture", default max_dist=2
        let results = idx.fuzzy_search("architecure", &SearchFilter::default(), 10, None);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_fuzzy_search_empty_query() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Test", &[]));
        let results = idx.fuzzy_search("", &SearchFilter::default(), 10, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_fuzzy_search_exact_ranks_higher_than_fuzzy() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_asset("a1", "Nature photography", &[]));
        idx.index_asset(make_asset("a2", "Natur film", &[]));
        // "nature" exact-matches a1's title token; fuzzy-matches a2
        let results = idx.fuzzy_search("nature", &SearchFilter::default(), 10, Some(2));
        assert!(results.len() >= 1);
        assert_eq!(results[0].asset_id, "a1");
    }

    #[test]
    fn test_fuzzy_search_description_typo() {
        let mut idx = SmartSearchIndex::new();
        let mut asset = make_asset("a1", "Video", &[]);
        asset.description = "Comprehensive documentary about coral reefs".to_string();
        idx.index_asset(asset);
        // "documetary" -> 2 edits from "documentary"
        let results = idx.fuzzy_search("documetary", &SearchFilter::default(), 10, Some(2));
        assert!(!results.is_empty());
        assert!(results[0]
            .matched_fields
            .contains(&"description".to_string()));
    }

    // -----------------------------------------------------------------------
    // BM25 scoring accuracy tests — known-relevance corpus
    //
    // These tests pin the score ordering and magnitude properties against a
    // small but fully-specified corpus where the expected relevance ordering
    // is unambiguous.  They act as regression guards for the TF×IDF weighting
    // coefficients (title ×3, tags ×2, description ×1) and IDF semantics.
    // -----------------------------------------------------------------------

    /// Helper: build a full asset record (title, description, tags).
    fn make_full_asset(id: &str, title: &str, description: &str, tags: &[&str]) -> IndexedAsset {
        IndexedAsset {
            id: id.to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            auto_tags: Vec::new(),
            title: title.to_string(),
            description: description.to_string(),
            duration_secs: None,
            width: None,
            format: None,
            collection_ids: Vec::new(),
            ingested_at: 1_000_000,
        }
    }

    /// **IDF property**: a term that appears in only one document should score
    /// strictly higher than a term that appears in every document, assuming
    /// the same TF and field weight.
    ///
    /// Corpus:
    ///   a1: title = "rare exclusive footage"
    ///   a2: title = "common everyday footage"
    ///   a3: title = "common daily footage"
    ///
    /// Query "rare" matches only a1 → IDF is high.
    /// Query "common" matches a2 and a3 → IDF is lower.
    /// Query "footage" matches all 3 → IDF is lowest.
    ///
    /// We verify: score(a1 | "rare") > score(a2 | "footage").
    #[test]
    fn test_bm25_idf_rare_term_scores_higher_than_common_term() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_full_asset("a1", "rare exclusive footage", "", &[]));
        idx.index_asset(make_full_asset("a2", "common everyday footage", "", &[]));
        idx.index_asset(make_full_asset("a3", "common daily footage", "", &[]));

        // Score of the rare-term match.
        let rare_results = idx.search("rare", &SearchFilter::default(), 10);
        assert_eq!(rare_results.len(), 1, "only a1 should match 'rare'");
        let score_rare = rare_results[0].score;

        // Score of the ubiquitous-term match (footage appears in all 3 docs).
        let footage_results = idx.search("footage", &SearchFilter::default(), 10);
        // All three docs match "footage", but a1's score should be its per-doc score.
        let score_footage_a1 = footage_results
            .iter()
            .find(|r| r.asset_id == "a1")
            .expect("a1 must appear in footage results")
            .score;

        assert!(
            score_rare > score_footage_a1,
            "IDF: rare term (df=1) must score higher than ubiquitous term (df=3); \
             score_rare={score_rare:.4} score_footage_a1={score_footage_a1:.4}"
        );
    }

    /// **TF property**: more occurrences of the query term in the same field
    /// yield a higher score.
    ///
    /// Corpus:
    ///   a1: title = "ocean ocean ocean deep dive"   ← 3 occurrences of "ocean"
    ///   a2: title = "ocean surface tour"            ← 1 occurrence of "ocean"
    ///
    /// Both docs are the only ones with "ocean", so IDF is identical.
    /// a1 has TF = 3/5, a2 has TF = 1/3 → a1 must rank first.
    #[test]
    fn test_bm25_tf_higher_frequency_ranks_first() {
        let mut idx = SmartSearchIndex::new();
        // a1: 3 occurrences out of 5 tokens → TF = 0.60
        idx.index_asset(make_full_asset(
            "a1",
            "ocean ocean ocean deep dive",
            "",
            &[],
        ));
        // a2: 1 occurrence out of 3 tokens → TF = 0.33
        idx.index_asset(make_full_asset("a2", "ocean surface tour", "", &[]));

        let results = idx.search("ocean", &SearchFilter::default(), 10);
        assert_eq!(results.len(), 2, "both docs must match 'ocean'");
        // a1 should rank first due to higher TF.
        assert_eq!(
            results[0].asset_id,
            "a1",
            "higher TF document must rank first; order: {:?}",
            results.iter().map(|r| &r.asset_id).collect::<Vec<_>>()
        );
        assert!(
            results[0].score > results[1].score,
            "a1 score ({}) must exceed a2 score ({})",
            results[0].score,
            results[1].score
        );
    }

    /// **Field weight property**: the same query term in the title (weight ×3)
    /// must produce a strictly higher score than in the description (weight ×1),
    /// all else being equal.
    ///
    /// Corpus:
    ///   a1: title = "volcano documentary"         ← term in title
    ///   a2: title = "nature film",
    ///       description = "volcano documentary"   ← same term in description only
    ///
    /// Both have identical token count and IDF. a1 uses weight 3; a2 uses weight 1.
    #[test]
    fn test_bm25_title_weight_beats_description_weight() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_full_asset("a1", "volcano documentary", "", &[]));
        idx.index_asset(make_full_asset(
            "a2",
            "nature film",
            "volcano documentary",
            &[],
        ));

        let results = idx.search("volcano", &SearchFilter::default(), 10);
        assert_eq!(results.len(), 2, "both docs contain 'volcano'");

        let score_a1 = results
            .iter()
            .find(|r| r.asset_id == "a1")
            .expect("a1 must be in results")
            .score;
        let score_a2 = results
            .iter()
            .find(|r| r.asset_id == "a2")
            .expect("a2 must be in results")
            .score;

        assert!(
            score_a1 > score_a2,
            "title field weight (×3) must exceed description weight (×1); \
             score_a1={score_a1:.4} score_a2={score_a2:.4}"
        );
    }

    /// **Tag weight property**: the same query term in a tag (weight ×2)
    /// must score above the description (weight ×1) but below the title (weight ×3).
    ///
    /// Corpus:
    ///   a_title: title = "glacier"          ← weight 3
    ///   a_tag:   tags  = ["glacier"]        ← weight 2
    ///   a_desc:  description = "glacier"    ← weight 1
    ///
    /// All three are the only holders of "glacier" so IDF is identical.
    #[test]
    fn test_bm25_field_weight_ordering_title_tag_description() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_full_asset("a_title", "glacier", "", &[]));
        idx.index_asset(make_full_asset("a_tag", "video", "", &["glacier"]));
        idx.index_asset(make_full_asset("a_desc", "video", "glacier", &[]));

        let results = idx.search("glacier", &SearchFilter::default(), 10);
        assert_eq!(results.len(), 3, "all three docs must match 'glacier'");

        // Extract scores by asset_id.
        let score = |id: &str| -> f32 {
            results
                .iter()
                .find(|r| r.asset_id == id)
                .map(|r| r.score)
                .unwrap_or_else(|| panic!("{id} not found in results"))
        };

        let s_title = score("a_title");
        let s_tag = score("a_tag");
        let s_desc = score("a_desc");

        assert!(
            s_title > s_tag,
            "title weight (×3) > tag weight (×2); s_title={s_title:.4} s_tag={s_tag:.4}"
        );
        assert!(
            s_tag > s_desc,
            "tag weight (×2) > description weight (×1); s_tag={s_tag:.4} s_desc={s_desc:.4}"
        );
    }

    /// **Multi-term query scoring**: a document matching both query terms must
    /// outscore a document matching only one, assuming the same IDF and field.
    ///
    /// Corpus:
    ///   a1: title = "arctic expedition footage"   ← matches both "arctic" and "footage"
    ///   a2: title = "tropical footage only"       ← matches only "footage"
    ///
    /// Query: "arctic footage".
    #[test]
    fn test_bm25_multi_term_both_matches_outscores_single_match() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_full_asset("a1", "arctic expedition footage", "", &[]));
        idx.index_asset(make_full_asset("a2", "tropical footage only", "", &[]));

        let results = idx.search("arctic footage", &SearchFilter::default(), 10);
        assert!(results.len() >= 1, "at least a1 must match");

        let score_a1 = results
            .iter()
            .find(|r| r.asset_id == "a1")
            .expect("a1 must appear")
            .score;

        if let Some(res_a2) = results.iter().find(|r| r.asset_id == "a2") {
            assert!(
                score_a1 > res_a2.score,
                "two-term match must beat single-term match; \
                 score_a1={score_a1:.4} score_a2={:.4}",
                res_a2.score
            );
        }
        // a1 must be the top result.
        assert_eq!(results[0].asset_id, "a1", "two-term match must rank first");
    }

    /// **Precision@K test with known relevance judgments**.
    ///
    /// Corpus of 6 assets; ground truth for query "nature wildlife":
    ///   Relevant (grade ≥ 1): nature_doc, wildlife_cam, ecology_study
    ///   Non-relevant:         city_tour, sports_reel, cooking_show
    ///
    /// We require precision@3 (top 3 results are all relevant) = 1.0.
    #[test]
    fn test_bm25_precision_at_k_known_relevance_corpus() {
        let mut idx = SmartSearchIndex::new();

        // Relevant docs — all contain the query terms in prominent positions.
        idx.index_asset(make_full_asset(
            "nature_doc",
            "nature wildlife documentary",
            "Explores wild habitats",
            &["nature", "wildlife", "documentary"],
        ));
        idx.index_asset(make_full_asset(
            "wildlife_cam",
            "wildlife camera trapping nature study",
            "Camera traps capture wildlife in nature",
            &["wildlife", "nature", "camera-trap"],
        ));
        idx.index_asset(make_full_asset(
            "ecology_study",
            "ecology and nature wildlife survey",
            "A scientific survey of wildlife populations",
            &["ecology", "nature", "wildlife"],
        ));

        // Non-relevant docs — no "nature" or "wildlife" in title/tags/description.
        idx.index_asset(make_full_asset(
            "city_tour",
            "city skyline tour",
            "Urban architecture and streets",
            &["city", "urban", "architecture"],
        ));
        idx.index_asset(make_full_asset(
            "sports_reel",
            "highlight sports reel",
            "Football match highlights",
            &["sports", "football", "highlights"],
        ));
        idx.index_asset(make_full_asset(
            "cooking_show",
            "cooking show episode",
            "Chef prepares Mediterranean dishes",
            &["cooking", "food", "chef"],
        ));

        // Ground truth — the three relevant IDs.
        let relevant: std::collections::HashSet<&str> =
            ["nature_doc", "wildlife_cam", "ecology_study"]
                .iter()
                .copied()
                .collect();

        let results = idx.search("nature wildlife", &SearchFilter::default(), 3);
        assert_eq!(
            results.len(),
            3,
            "must return 3 results for precision@3 test"
        );

        // Precision@3 must be exactly 1.0 (all top-3 are relevant).
        let hits: usize = results
            .iter()
            .filter(|r| relevant.contains(r.asset_id.as_str()))
            .count();
        assert_eq!(
            hits,
            3,
            "precision@3 must be 3/3=1.0; top-3 result IDs: {:?}",
            results.iter().map(|r| &r.asset_id).collect::<Vec<_>>()
        );
    }

    /// **nDCG-inspired ordering test**.
    ///
    /// A document that matches the query in the title AND has a matching tag
    /// (total signal: weight 3 + weight 2 = 5) must rank above one that only
    /// matches in the description (weight 1).
    ///
    /// We validate this as a proxy for the nDCG ordering property: the highest-
    /// relevance document (strongest signal) appears at rank 1.
    #[test]
    fn test_bm25_ndcg_title_plus_tag_beats_description_only() {
        let mut idx = SmartSearchIndex::new();

        // a1: "summit" appears in title (×3) AND as a tag (×2) → total weight 5
        idx.index_asset(make_full_asset(
            "a1",
            "summit expedition film",
            "an adventure documentary",
            &["summit", "adventure"],
        ));
        // a2: "summit" appears only in description (×1)
        idx.index_asset(make_full_asset(
            "a2",
            "mountain adventure",
            "a summit climbing expedition",
            &["mountain", "adventure"],
        ));
        // a3: no "summit" at all — must not appear in results.
        idx.index_asset(make_full_asset("a3", "ocean dive", "", &["underwater"]));

        let results = idx.search("summit", &SearchFilter::default(), 10);

        assert_eq!(
            results.len(),
            2,
            "only a1 and a2 contain 'summit'; results={:?}",
            results.iter().map(|r| &r.asset_id).collect::<Vec<_>>()
        );
        assert_eq!(
            results[0].asset_id,
            "a1",
            "title+tag combined weight must rank a1 first; actual ranking: {:?}",
            results.iter().map(|r| &r.asset_id).collect::<Vec<_>>()
        );
        assert!(
            results[0].score > results[1].score,
            "score gap must be positive: a1={:.4} a2={:.4}",
            results[0].score,
            results[1].score
        );
    }

    /// **Matched-fields metadata correctness**.
    ///
    /// Verify that `matched_fields` accurately records which field(s) contributed
    /// to the score — no phantom fields and no missed fields.
    #[test]
    fn test_bm25_matched_fields_completeness_and_accuracy() {
        let mut idx = SmartSearchIndex::new();

        // "rain" only in description.
        let mut a1 = make_asset("a1", "weather film", &[]);
        a1.description = "heavy rain and thunder".to_string();
        idx.index_asset(a1);

        // "storm" only in title.
        idx.index_asset(make_full_asset("a2", "storm chaser documentary", "", &[]));

        // "wind" only in tag.
        idx.index_asset(make_full_asset("a3", "outdoor video", "", &["wind"]));

        // "rain" → must report description only.
        let rain = idx.search("rain", &SearchFilter::default(), 10);
        assert_eq!(rain.len(), 1);
        assert!(rain[0].matched_fields.contains(&"description".to_string()));
        assert!(!rain[0].matched_fields.contains(&"title".to_string()));
        assert!(!rain[0].matched_fields.contains(&"tags".to_string()));

        // "storm" → must report title only.
        let storm = idx.search("storm", &SearchFilter::default(), 10);
        assert_eq!(storm.len(), 1);
        assert!(storm[0].matched_fields.contains(&"title".to_string()));
        assert!(!storm[0].matched_fields.contains(&"description".to_string()));

        // "wind" → must report tags only.
        let wind = idx.search("wind", &SearchFilter::default(), 10);
        assert_eq!(wind.len(), 1);
        assert!(wind[0].matched_fields.contains(&"tags".to_string()));
        assert!(!wind[0].matched_fields.contains(&"title".to_string()));
    }

    /// **Score monotonicity**: adding a relevant document to an otherwise
    /// irrelevant corpus must not reduce the score of an existing relevant
    /// document (IDF denominator grows but the existing doc's df doesn't).
    ///
    /// Adding a non-matching doc for "forest" increases N (corpus size) while
    /// keeping df("forest") = 1, which strictly increases IDF and therefore
    /// increases the score of the relevant document.
    #[test]
    fn test_bm25_score_increases_as_corpus_grows_with_irrelevant_docs() {
        let mut idx = SmartSearchIndex::new();
        idx.index_asset(make_full_asset("a1", "forest trail footage", "", &[]));

        let results_small = idx.search("forest", &SearchFilter::default(), 10);
        let score_small = results_small[0].score;

        // Add 5 irrelevant docs (no "forest").
        for i in 0..5_u32 {
            idx.index_asset(make_full_asset(
                &format!("irr{i}"),
                "urban landscape tour",
                "",
                &[],
            ));
        }

        let results_large = idx.search("forest", &SearchFilter::default(), 10);
        let score_large = results_large
            .iter()
            .find(|r| r.asset_id == "a1")
            .expect("a1 must still be found after corpus expansion")
            .score;

        assert!(
            score_large >= score_small,
            "IDF must increase (or stay equal) as irrelevant docs are added; \
             score_small={score_small:.4} score_large={score_large:.4}"
        );
    }
}

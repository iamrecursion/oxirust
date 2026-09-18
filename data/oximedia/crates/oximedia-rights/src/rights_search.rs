//! Full-text search across rights holders, territories, and license terms.
//!
//! Provides a lightweight in-memory inverted-index over [`SearchDocument`]
//! records.  The index tokenises text fields (case-insensitive ASCII) and
//! scores results using a simple TF (term frequency) ranking model.
//!
//! # Search fields indexed
//!
//! - `holder` — rights holder name / organisation
//! - `territory` — territory codes and names
//! - `license_terms` — free-text description of the license terms
//! - `asset_id` — asset identifier (exact + prefix)
//! - `notes` — miscellaneous notes
//!
//! # Query syntax
//!
//! Queries are whitespace-tokenised; all tokens must match (AND semantics).
//! Single-character tokens are ignored.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

// ── SearchDocument ────────────────────────────────────────────────────────────

/// A rights record prepared for search indexing.
#[derive(Debug, Clone)]
pub struct SearchDocument {
    /// Unique record ID (also the primary search key).
    pub record_id: String,
    /// Asset this record applies to.
    pub asset_id: String,
    /// Rights holder name or organisation.
    pub holder: String,
    /// Territory codes or names (space-separated for multi-territory).
    pub territory: String,
    /// Free-text description of the license terms.
    pub license_terms: String,
    /// Miscellaneous notes.
    pub notes: String,
    /// Whether the record is active.
    pub active: bool,
    /// Optional expiry (Unix seconds).
    pub expires_at: Option<u64>,
}

impl SearchDocument {
    /// Create a minimal document.
    #[must_use]
    pub fn new(
        record_id: impl Into<String>,
        asset_id: impl Into<String>,
        holder: impl Into<String>,
    ) -> Self {
        Self {
            record_id: record_id.into(),
            asset_id: asset_id.into(),
            holder: holder.into(),
            territory: String::new(),
            license_terms: String::new(),
            notes: String::new(),
            active: true,
            expires_at: None,
        }
    }

    /// Builder: set territory string.
    #[must_use]
    pub fn with_territory(mut self, territory: impl Into<String>) -> Self {
        self.territory = territory.into();
        self
    }

    /// Builder: set license terms.
    #[must_use]
    pub fn with_license_terms(mut self, terms: impl Into<String>) -> Self {
        self.license_terms = terms.into();
        self
    }

    /// Builder: set notes.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    /// Builder: set active flag.
    #[must_use]
    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Builder: set expiry.
    #[must_use]
    pub fn with_expires_at(mut self, ts: u64) -> Self {
        self.expires_at = Some(ts);
        self
    }

    /// Return all searchable text concatenated (lowercase).
    fn searchable_text(&self) -> String {
        format!(
            "{} {} {} {} {} {}",
            self.record_id,
            self.asset_id,
            self.holder,
            self.territory,
            self.license_terms,
            self.notes
        )
        .to_lowercase()
    }
}

// ── SearchResult ──────────────────────────────────────────────────────────────

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matching document.
    pub document: SearchDocument,
    /// Relevance score (higher = more relevant).
    pub score: f64,
}

// ── SearchFilter ──────────────────────────────────────────────────────────────

/// Optional filters applied after text matching.
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    /// If `Some(true)`, only return active records; `Some(false)` = inactive only.
    pub active_only: Option<bool>,
    /// If set, only return records for this asset ID.
    pub asset_id: Option<String>,
    /// If set, only return records that expire before this Unix timestamp.
    pub expires_before: Option<u64>,
    /// If set, only return records that expire after this Unix timestamp.
    pub expires_after: Option<u64>,
}

impl SearchFilter {
    /// Create a filter that accepts only active records.
    #[must_use]
    pub fn active() -> Self {
        Self {
            active_only: Some(true),
            ..Default::default()
        }
    }

    /// Apply the filter to a document. Returns `true` if the document passes.
    #[must_use]
    pub fn matches(&self, doc: &SearchDocument) -> bool {
        if let Some(active) = self.active_only {
            if doc.active != active {
                return false;
            }
        }
        if let Some(ref asset) = self.asset_id {
            if &doc.asset_id != asset {
                return false;
            }
        }
        if let Some(before) = self.expires_before {
            match doc.expires_at {
                Some(exp) if exp < before => {}
                _ => return false,
            }
        }
        if let Some(after) = self.expires_after {
            match doc.expires_at {
                Some(exp) if exp > after => {}
                _ => return false,
            }
        }
        true
    }
}

// ── RightsSearchIndex ─────────────────────────────────────────────────────────

/// In-memory inverted-index over rights [`SearchDocument`]s.
#[derive(Debug, Default)]
pub struct RightsSearchIndex {
    /// All documents keyed by record_id.
    documents: HashMap<String, SearchDocument>,
    /// Inverted index: token → set of record_ids.
    index: HashMap<String, HashSet<String>>,
}

impl RightsSearchIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace a document in the index.
    pub fn add(&mut self, doc: SearchDocument) {
        // Remove old index entries for this record_id.
        self.remove(&doc.record_id);

        let tokens = tokenise(&doc.searchable_text());
        for token in &tokens {
            self.index
                .entry(token.clone())
                .or_default()
                .insert(doc.record_id.clone());
        }
        self.documents.insert(doc.record_id.clone(), doc);
    }

    /// Remove a document by record_id.
    pub fn remove(&mut self, record_id: &str) {
        if let Some(old_doc) = self.documents.remove(record_id) {
            let tokens = tokenise(&old_doc.searchable_text());
            for token in &tokens {
                if let Some(set) = self.index.get_mut(token) {
                    set.remove(record_id);
                }
            }
        }
    }

    /// Total number of indexed documents.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Search for documents matching a query string.
    ///
    /// Returns results sorted by descending relevance score.
    /// Empty query returns all documents (subject to `filter`).
    #[must_use]
    pub fn search(&self, query: &str, filter: Option<&SearchFilter>) -> Vec<SearchResult> {
        let query_tokens = tokenise(&query.to_lowercase());

        let candidates: Vec<&SearchDocument> = if query_tokens.is_empty() {
            self.documents.values().collect()
        } else {
            // Intersect posting lists for AND semantics.
            let mut candidate_ids: Option<HashSet<&str>> = None;
            for token in &query_tokens {
                let posting: HashSet<&str> = self
                    .index
                    .get(token)
                    .map(|s| s.iter().map(String::as_str).collect())
                    .unwrap_or_default();
                candidate_ids = Some(match candidate_ids {
                    None => posting,
                    Some(existing) => existing.intersection(&posting).copied().collect(),
                });
            }
            candidate_ids
                .unwrap_or_default()
                .into_iter()
                .filter_map(|id| self.documents.get(id))
                .collect()
        };

        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|doc| filter.map_or(true, |f| f.matches(doc)))
            .map(|doc| {
                let score = if query_tokens.is_empty() {
                    1.0
                } else {
                    score_document(doc, &query_tokens)
                };
                SearchResult {
                    document: doc.clone(),
                    score,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Find all documents matching a specific holder (exact, case-insensitive).
    #[must_use]
    pub fn find_by_holder(&self, holder: &str) -> Vec<&SearchDocument> {
        let lower = holder.to_lowercase();
        self.documents
            .values()
            .filter(|d| d.holder.to_lowercase() == lower)
            .collect()
    }

    /// Find all documents for a specific territory code (case-insensitive).
    #[must_use]
    pub fn find_by_territory(&self, code: &str) -> Vec<&SearchDocument> {
        let lower = code.to_lowercase();
        self.documents
            .values()
            .filter(|d| {
                d.territory
                    .to_lowercase()
                    .split_whitespace()
                    .any(|t| t == lower)
            })
            .collect()
    }

    /// Suggest completions for a prefix (returns matching unique tokens).
    #[must_use]
    pub fn suggest(&self, prefix: &str) -> Vec<String> {
        let lower = prefix.to_lowercase();
        let mut matches: Vec<String> = self
            .index
            .keys()
            .filter(|k| k.starts_with(&lower) && k.len() > 2)
            .cloned()
            .collect();
        matches.sort();
        matches.dedup();
        matches
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Tokenise a string into lowercase alphanumeric tokens of length > 2
/// (i.e., at least 3 characters, filtering out stop-word-length tokens).
fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|t| t.len() > 2)
        .collect()
}

/// Score a document against a set of query tokens using term frequency.
fn score_document(doc: &SearchDocument, tokens: &[String]) -> f64 {
    let text = doc.searchable_text();
    let mut score = 0.0_f64;
    for token in tokens {
        let count = text.matches(token.as_str()).count() as f64;
        score += count;
        // Bonus: exact match in holder or asset_id (high importance fields)
        if doc.holder.to_lowercase().contains(token.as_str()) {
            score += 2.0;
        }
        if doc.asset_id.to_lowercase().contains(token.as_str()) {
            score += 1.5;
        }
    }
    score
}

// ── CachedRightsCheck ─────────────────────────────────────────────────────────
//
// Task 12: Query caching in rights_check for frequently accessed assets.
// We add a thin caching wrapper here (separate from the index) that stores
// the last N check results in an LRU-like structure.

use crate::rights_check::{CheckRequest, CheckResult, RightsChecker};

/// A cache entry for a rights check result.
#[derive(Debug, Clone)]
struct CacheEntry {
    result: CheckResult,
    /// Unix timestamp when this entry was inserted.
    inserted_at: u64,
    /// Number of times this entry has been served from cache.
    hit_count: u64,
}

/// LRU-bounded cache key for a rights check request.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    asset_id: String,
    action: String,
    territory: String,
    platform: String,
}

impl CacheKey {
    fn from_request(req: &CheckRequest) -> Self {
        Self {
            asset_id: req.asset_id.clone(),
            action: format!("{:?}", req.action),
            territory: req.territory.clone(),
            platform: req.platform.clone(),
        }
    }
}

/// A rights checker with an in-memory result cache.
///
/// Identical requests within the TTL window return cached results.
/// The cache is invalidated when new grants are added via
/// [`add_grant`](CachedRightsChecker::add_grant).
#[derive(Debug)]
pub struct CachedRightsChecker {
    inner: RightsChecker,
    cache: HashMap<CacheKey, CacheEntry>,
    /// Time-to-live in seconds for cached results.
    ttl_secs: u64,
    /// Maximum number of cached entries (oldest evicted when exceeded).
    max_entries: usize,
    /// Insertion order tracker for eviction (queue of keys).
    insertion_order: Vec<CacheKey>,
}

impl Default for CachedRightsChecker {
    fn default() -> Self {
        Self::new(300, 1024)
    }
}

impl CachedRightsChecker {
    /// Create a caching checker with a specific TTL and max entries.
    #[must_use]
    pub fn new(ttl_secs: u64, max_entries: usize) -> Self {
        Self {
            inner: RightsChecker::new(),
            cache: HashMap::new(),
            ttl_secs,
            max_entries,
            insertion_order: Vec::new(),
        }
    }

    /// Add a rights grant (also invalidates the cache).
    pub fn add_grant(&mut self, grant: crate::rights_check::RightsGrant) {
        self.inner.add_grant(grant);
        self.invalidate();
    }

    /// Invalidate the entire cache.
    pub fn invalidate(&mut self) {
        self.cache.clear();
        self.insertion_order.clear();
    }

    /// Perform a rights check, using cached results when available.
    #[must_use]
    pub fn check(&mut self, req: &CheckRequest) -> CheckResult {
        let key = CacheKey::from_request(req);
        let now = req.now;

        // Check cache hit.
        if let Some(entry) = self.cache.get_mut(&key) {
            if now.saturating_sub(entry.inserted_at) < self.ttl_secs {
                entry.hit_count += 1;
                return entry.result.clone();
            }
            // Expired entry — remove it.
            self.cache.remove(&key);
            self.insertion_order.retain(|k| k != &key);
        }

        // Cache miss: perform the actual check.
        let result = self.inner.check(req);

        // Evict oldest if at capacity.
        if self.cache.len() >= self.max_entries && !self.insertion_order.is_empty() {
            let oldest = self.insertion_order.remove(0);
            self.cache.remove(&oldest);
        }

        self.cache.insert(
            key.clone(),
            CacheEntry {
                result: result.clone(),
                inserted_at: now,
                hit_count: 0,
            },
        );
        self.insertion_order.push(key);
        result
    }

    /// Number of entries currently in the cache.
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Total cache hits across all entries.
    #[must_use]
    pub fn total_hits(&self) -> u64 {
        self.cache.values().map(|e| e.hit_count).sum()
    }

    /// Underlying grant count.
    #[must_use]
    pub fn grant_count(&self) -> usize {
        self.inner.grant_count()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rights_check::{ActionKind, CheckRequest, RightsGrant};

    // ── tokenise ──

    #[test]
    fn test_tokenise_basic() {
        let tokens = tokenise("Hello World");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_tokenise_filters_short_tokens() {
        let tokens = tokenise("a to the testing");
        // 1-char "a" → filtered (len 1, not > 2)
        assert!(!tokens.contains(&"a".to_string()));
        // 2-char "to" → filtered (len 2, not > 2)
        assert!(!tokens.contains(&"to".to_string()));
        // 4-char "test" → included
        assert!(tokens.contains(&"testing".to_string()));
    }

    // ── RightsSearchIndex ──

    fn sample_index() -> RightsSearchIndex {
        let mut idx = RightsSearchIndex::new();
        idx.add(
            SearchDocument::new("r1", "asset-A", "Alice Music Publishing")
                .with_territory("US GB")
                .with_license_terms("royalty-free unlimited broadcast")
                .with_notes("Film score rights"),
        );
        idx.add(
            SearchDocument::new("r2", "asset-B", "Bob Video Rights")
                .with_territory("DE FR")
                .with_license_terms("rights-managed single use only")
                .with_notes("Documentary footage"),
        );
        idx.add(
            SearchDocument::new("r3", "asset-C", "Alice Music Publishing")
                .with_territory("US")
                .with_license_terms("editorial only no commercial use")
                .with_active(false),
        );
        idx
    }

    #[test]
    fn test_document_count() {
        assert_eq!(sample_index().document_count(), 3);
    }

    #[test]
    fn test_search_holder_name() {
        let idx = sample_index();
        let results = idx.search("alice", None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_territory() {
        let idx = sample_index();
        let results = idx.search("documentary", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.record_id, "r2");
    }

    #[test]
    fn test_search_empty_query_returns_all() {
        let idx = sample_index();
        let results = idx.search("", None);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_with_active_filter() {
        let idx = sample_index();
        let filter = SearchFilter::active();
        let results = idx.search("alice", Some(&filter));
        // r3 is inactive → only r1
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.record_id, "r1");
    }

    #[test]
    fn test_search_no_match() {
        let idx = sample_index();
        let results = idx.search("nonexistent_term_xyz", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_and_semantics() {
        let idx = sample_index();
        // Both terms must match: "alice" AND "broadcast"
        let results = idx.search("alice broadcast", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.record_id, "r1");
    }

    #[test]
    fn test_search_sorted_by_score() {
        let idx = sample_index();
        // Search for "alice" — both r1 and r3 match; r1 has more mentions
        let results = idx.search("alice", None);
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn test_find_by_holder_exact() {
        let idx = sample_index();
        let docs = idx.find_by_holder("Alice Music Publishing");
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_find_by_holder_case_insensitive() {
        let idx = sample_index();
        let docs = idx.find_by_holder("alice music publishing");
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_find_by_territory() {
        let idx = sample_index();
        let docs = idx.find_by_territory("US");
        // r1 (US GB) and r3 (US)
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_remove_document() {
        let mut idx = sample_index();
        idx.remove("r1");
        assert_eq!(idx.document_count(), 2);
        let results = idx.search("film score", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_suggest_prefix() {
        let idx = sample_index();
        let suggestions = idx.suggest("royal");
        assert!(suggestions.iter().any(|s| s.starts_with("royal")));
    }

    #[test]
    fn test_search_filter_asset_id() {
        let idx = sample_index();
        let filter = SearchFilter {
            asset_id: Some("asset-A".to_string()),
            ..Default::default()
        };
        let results = idx.search("", Some(&filter));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.asset_id, "asset-A");
    }

    // ── CachedRightsChecker ──

    fn stream_grant(id: &str, asset: &str) -> RightsGrant {
        RightsGrant::new(id, asset)
            .with_action(ActionKind::Stream)
            .with_window(0, u64::MAX)
    }

    #[test]
    fn test_cached_checker_miss_then_hit() {
        let mut checker = CachedRightsChecker::new(600, 100);
        checker.add_grant(stream_grant("g1", "asset-A"));

        let req = CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 1000);
        let r1 = checker.check(&req);
        assert!(r1.is_allowed());
        assert_eq!(checker.cache_size(), 1);
        assert_eq!(checker.total_hits(), 0);

        // Second call → cache hit
        let r2 = checker.check(&req);
        assert!(r2.is_allowed());
        assert_eq!(checker.total_hits(), 1);
    }

    #[test]
    fn test_cached_checker_expired_entry() {
        let mut checker = CachedRightsChecker::new(100, 100); // TTL = 100s
        checker.add_grant(stream_grant("g1", "asset-A"));

        let req1 = CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 1000);
        checker.check(&req1);
        assert_eq!(checker.cache_size(), 1);

        // Request at ts=1200 (200s later, past TTL of 100s) → cache miss
        let req2 = CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 1200);
        checker.check(&req2);
        // Total hits should still be 0 (both were misses / re-computed)
        assert_eq!(checker.total_hits(), 0);
    }

    #[test]
    fn test_cached_checker_invalidate_on_add_grant() {
        let mut checker = CachedRightsChecker::new(600, 100);
        checker.add_grant(stream_grant("g1", "asset-A"));

        let req = CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 1000);
        checker.check(&req);
        assert_eq!(checker.cache_size(), 1);

        // Adding a grant invalidates cache
        checker.add_grant(stream_grant("g2", "asset-B"));
        assert_eq!(checker.cache_size(), 0);
    }

    #[test]
    fn test_cached_checker_evicts_oldest_at_capacity() {
        let mut checker = CachedRightsChecker::new(600, 2); // max 2 entries
        checker.add_grant(stream_grant("g1", "asset-A"));
        checker.add_grant(stream_grant("g2", "asset-B"));
        checker.add_grant(stream_grant("g3", "asset-C"));

        // Re-enable cache after add_grant invalidations
        checker.cache.clear();
        checker.insertion_order.clear();

        let r1 = CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 1000);
        let r2 = CheckRequest::new("asset-B", ActionKind::Stream, "US", "web", 1000);
        let r3 = CheckRequest::new("asset-C", ActionKind::Stream, "US", "web", 1000);

        let _ = checker.check(&r1);
        let _ = checker.check(&r2);
        assert_eq!(checker.cache_size(), 2);

        // Third entry should evict the oldest (r1)
        let _ = checker.check(&r3);
        assert_eq!(checker.cache_size(), 2);
    }

    #[test]
    fn test_cached_checker_denied_result_cached() {
        let mut checker = CachedRightsChecker::new(600, 100);
        let req = CheckRequest::new("no-asset", ActionKind::Stream, "US", "web", 1000);
        let r1 = checker.check(&req);
        assert!(r1.is_denied());
        assert_eq!(checker.cache_size(), 1);

        let r2 = checker.check(&req);
        assert!(r2.is_denied());
        assert_eq!(checker.total_hits(), 1);
    }
}

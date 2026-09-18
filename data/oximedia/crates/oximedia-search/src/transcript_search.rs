#![allow(dead_code)]
//! Speech-to-text transcript indexing and search with timestamp resolution.
//!
//! This module indexes time-aligned transcripts (e.g. from ASR engines) and
//! supports full-text search returning exact timecodes so that media players
//! can jump directly to the spoken phrase.
//!
//! # Architecture
//!
//! A [`TranscriptIndex`] stores a sequence of [`TranscriptSegment`]s per
//! asset. Each segment carries a start/end timestamp and the spoken text.
//! An inverted index maps lowercase tokens to the (asset, segment) positions
//! where they appear, enabling sub-linear full-text lookup.
//!
//! Search results include the matched segment's timestamps, making it
//! possible to deep-link into a video or audio file at the exact moment
//! a phrase was spoken.
//!
//! # Patent-free
//!
//! Only standard TF-IDF and BM25-style scoring are used — no patented
//! audio-fingerprint or licensed speech APIs are required.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{SearchError, SearchResult};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single time-aligned transcript segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// Start time in milliseconds from the beginning of the media asset.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Spoken text for this segment.
    pub text: String,
    /// Confidence score from the ASR engine, in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Speaker identifier (if diarization was performed).
    pub speaker_id: Option<String>,
}

impl TranscriptSegment {
    /// Create a new transcript segment.
    pub fn new(start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
            confidence: 1.0,
            speaker_id: None,
        }
    }

    /// Create a segment with confidence and optional speaker.
    pub fn with_meta(
        start_ms: u64,
        end_ms: u64,
        text: impl Into<String>,
        confidence: f32,
        speaker_id: Option<String>,
    ) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
            confidence: confidence.clamp(0.0, 1.0),
            speaker_id,
        }
    }

    /// Duration of the segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// A match returned by transcript search.
#[derive(Debug, Clone)]
pub struct TranscriptMatch {
    /// Asset ID that contains the match.
    pub asset_id: Uuid,
    /// The segment containing the matched text.
    pub segment: TranscriptSegment,
    /// TF-IDF relevance score.
    pub score: f32,
    /// The specific tokens that were matched (for highlighting).
    pub matched_tokens: Vec<String>,
}

/// Configuration for the transcript index.
#[derive(Debug, Clone)]
pub struct TranscriptIndexConfig {
    /// Minimum token length (shorter tokens are not indexed).
    pub min_token_len: usize,
    /// Whether to strip punctuation from tokens.
    pub strip_punctuation: bool,
    /// Maximum number of results to return per search.
    pub default_limit: usize,
    /// Stop words to exclude from the index.
    pub stop_words: Vec<String>,
}

impl Default for TranscriptIndexConfig {
    fn default() -> Self {
        Self {
            min_token_len: 2,
            strip_punctuation: true,
            default_limit: 20,
            stop_words: vec![
                "a".into(),
                "an".into(),
                "the".into(),
                "is".into(),
                "in".into(),
                "on".into(),
                "at".into(),
                "to".into(),
                "for".into(),
                "of".into(),
                "and".into(),
                "or".into(),
                "but".into(),
                "i".into(),
                "we".into(),
                "you".into(),
                "it".into(),
                "he".into(),
                "she".into(),
                "they".into(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Posting: (asset_id, segment_index) with term frequency
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SegmentPosting {
    asset_id: Uuid,
    segment_index: usize,
    tf: f32,
}

// ---------------------------------------------------------------------------
// TranscriptIndex
// ---------------------------------------------------------------------------

/// Indexes speech-to-text transcripts with millisecond-resolution timestamps.
///
/// Supports full-text search returning the asset ID and exact segment
/// timestamps so media players can seek directly to the spoken phrase.
#[derive(Debug)]
pub struct TranscriptIndex {
    /// Configuration.
    config: TranscriptIndexConfig,
    /// Per-asset segments: asset_id -> `Vec<TranscriptSegment>`
    segments: HashMap<Uuid, Vec<TranscriptSegment>>,
    /// Inverted index: token -> list of (asset_id, segment_index, tf)
    postings: HashMap<String, Vec<SegmentPosting>>,
    /// Document frequency per token (number of assets containing the token).
    df: HashMap<String, usize>,
    /// Total number of indexed assets.
    asset_count: usize,
}

impl TranscriptIndex {
    /// Create a new, empty transcript index.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(TranscriptIndexConfig::default())
    }

    /// Create a transcript index with a custom configuration.
    #[must_use]
    pub fn with_config(config: TranscriptIndexConfig) -> Self {
        Self {
            config,
            segments: HashMap::new(),
            postings: HashMap::new(),
            df: HashMap::new(),
            asset_count: 0,
        }
    }

    /// Index the transcript for an asset.
    ///
    /// If the asset was previously indexed, its previous transcript is
    /// replaced entirely.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidQuery` if any segment has `start_ms > end_ms`.
    pub fn index_asset(
        &mut self,
        asset_id: Uuid,
        segments: Vec<TranscriptSegment>,
    ) -> SearchResult<()> {
        for seg in &segments {
            if seg.start_ms > seg.end_ms {
                return Err(SearchError::InvalidQuery(format!(
                    "Segment start_ms {} > end_ms {} for asset {}",
                    seg.start_ms, seg.end_ms, asset_id
                )));
            }
        }

        // Remove old postings if the asset was previously indexed.
        if self.segments.contains_key(&asset_id) {
            self.remove_asset_postings(asset_id);
        } else {
            self.asset_count += 1;
        }

        // Build per-segment token maps.
        let mut asset_df_tokens: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for (seg_idx, seg) in segments.iter().enumerate() {
            let tokens = self.tokenize(&seg.text);
            let tf_map = compute_tf(&tokens);

            for (token, tf) in &tf_map {
                self.postings
                    .entry(token.clone())
                    .or_default()
                    .push(SegmentPosting {
                        asset_id,
                        segment_index: seg_idx,
                        tf: *tf,
                    });
                asset_df_tokens.insert(token.clone());
            }
        }

        // Update document frequencies.
        for token in asset_df_tokens {
            *self.df.entry(token).or_insert(0) += 1;
        }

        self.segments.insert(asset_id, segments);
        Ok(())
    }

    /// Remove all indexed data for an asset.
    pub fn remove_asset(&mut self, asset_id: Uuid) {
        if self.segments.remove(&asset_id).is_none() {
            return;
        }
        self.remove_asset_postings(asset_id);
        self.asset_count = self.asset_count.saturating_sub(1);
    }

    /// Remove postings for an asset without decrementing asset_count.
    fn remove_asset_postings(&mut self, asset_id: Uuid) {
        // Collect df tokens to update.
        let mut removed_tokens: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for (token, postings) in self.postings.iter_mut() {
            let before = postings.len();
            postings.retain(|p| p.asset_id != asset_id);
            if postings.len() < before {
                removed_tokens.insert(token.clone());
            }
        }

        // Update df.
        for token in &removed_tokens {
            if let Some(count) = self.df.get_mut(token.as_str()) {
                *count = count.saturating_sub(1);
            }
        }

        // Remove empty posting lists.
        self.postings.retain(|_, v| !v.is_empty());
    }

    /// Search for a query phrase in the indexed transcripts.
    ///
    /// Returns matches sorted by descending TF-IDF score.
    /// Each match includes the exact segment timestamps for deep-linking.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<TranscriptMatch> {
        self.search_with_limit(query, self.config.default_limit)
    }

    /// Search with an explicit result limit.
    #[must_use]
    pub fn search_with_limit(&self, query: &str, limit: usize) -> Vec<TranscriptMatch> {
        let query_tokens = self.tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        // Collect candidate (asset_id, segment_index) pairs and their scores.
        // Use a HashMap keyed by (asset_id, segment_index) for accumulation.
        let mut candidate_scores: HashMap<(Uuid, usize), (f32, Vec<String>)> = HashMap::new();

        for token in &query_tokens {
            let Some(postings) = self.postings.get(token.as_str()) else {
                continue;
            };
            let df = self.df.get(token.as_str()).copied().unwrap_or(1);
            let idf = compute_idf(self.asset_count, df);

            for posting in postings {
                let tfidf = posting.tf * idf;
                let key = (posting.asset_id, posting.segment_index);
                let entry = candidate_scores.entry(key).or_insert((0.0, Vec::new()));
                entry.0 += tfidf;
                if !entry.1.contains(token) {
                    entry.1.push(token.clone());
                }
            }
        }

        // Convert to TranscriptMatch objects.
        let mut matches: Vec<TranscriptMatch> = candidate_scores
            .into_iter()
            .filter_map(|((asset_id, seg_idx), (score, matched_tokens))| {
                let segs = self.segments.get(&asset_id)?;
                let segment = segs.get(seg_idx)?.clone();
                Some(TranscriptMatch {
                    asset_id,
                    segment,
                    score,
                    matched_tokens,
                })
            })
            .collect();

        // Sort by score descending, then by start_ms for tie-breaking.
        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.segment.start_ms.cmp(&b.segment.start_ms))
        });
        matches.truncate(limit);
        matches
    }

    /// Search for an exact phrase (all tokens must appear in the same segment).
    #[must_use]
    pub fn search_phrase(&self, phrase: &str) -> Vec<TranscriptMatch> {
        let tokens = self.tokenize(phrase);
        if tokens.is_empty() {
            return Vec::new();
        }

        // Find segments that contain ALL tokens.
        let mut candidate_sets: Option<std::collections::HashSet<(Uuid, usize)>> = None;

        for token in &tokens {
            let Some(postings) = self.postings.get(token.as_str()) else {
                // If any token is missing, no phrase matches are possible.
                return Vec::new();
            };
            let current_set: std::collections::HashSet<(Uuid, usize)> = postings
                .iter()
                .map(|p| (p.asset_id, p.segment_index))
                .collect();

            candidate_sets = Some(match candidate_sets {
                None => current_set,
                Some(prev) => prev.intersection(&current_set).copied().collect(),
            });
        }

        let candidates = match candidate_sets {
            Some(s) if !s.is_empty() => s,
            _ => return Vec::new(),
        };

        let mut matches: Vec<TranscriptMatch> = candidates
            .into_iter()
            .filter_map(|(asset_id, seg_idx)| {
                let segs = self.segments.get(&asset_id)?;
                let segment = segs.get(seg_idx)?.clone();
                // Score by segment confidence as proxy.
                Some(TranscriptMatch {
                    asset_id,
                    segment: segment.clone(),
                    score: segment.confidence,
                    matched_tokens: tokens.clone(),
                })
            })
            .collect();

        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.segment.start_ms.cmp(&b.segment.start_ms))
        });
        matches
    }

    /// Retrieve all segments for an asset.
    #[must_use]
    pub fn get_segments(&self, asset_id: Uuid) -> Option<&[TranscriptSegment]> {
        self.segments.get(&asset_id).map(Vec::as_slice)
    }

    /// Return the total number of indexed assets.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.asset_count
    }

    /// Return the total number of distinct tokens in the index.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.postings.len()
    }

    /// Total number of segments across all assets.
    #[must_use]
    pub fn total_segment_count(&self) -> usize {
        self.segments.values().map(Vec::len).sum()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn tokenize(&self, text: &str) -> Vec<String> {
        let cleaned = if self.config.strip_punctuation {
            text.chars()
                .map(|c| {
                    if c.is_alphanumeric() || c.is_whitespace() {
                        c
                    } else {
                        ' '
                    }
                })
                .collect::<String>()
        } else {
            text.to_string()
        };

        cleaned
            .split_whitespace()
            .map(str::to_lowercase)
            .filter(|t| t.len() >= self.config.min_token_len && !self.config.stop_words.contains(t))
            .collect()
    }
}

impl Default for TranscriptIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TF / IDF helpers
// ---------------------------------------------------------------------------

fn compute_tf(tokens: &[String]) -> HashMap<String, f32> {
    if tokens.is_empty() {
        return HashMap::new();
    }
    let mut counts: HashMap<String, f32> = HashMap::new();
    for t in tokens {
        *counts.entry(t.clone()).or_insert(0.0) += 1.0;
    }
    let total = tokens.len() as f32;
    counts.values_mut().for_each(|v| *v /= total);
    counts
}

fn compute_idf(asset_count: usize, df: usize) -> f32 {
    let n = (asset_count as f32).max(1.0);
    let d = (df as f32).max(1.0);
    ((1.0 + n) / (1.0 + d)).ln() + 1.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_segments() -> Vec<TranscriptSegment> {
        vec![
            TranscriptSegment::new(0, 3000, "Welcome to the nature documentary"),
            TranscriptSegment::new(3000, 6000, "Today we explore the rainforest"),
            TranscriptSegment::new(6000, 9000, "The rainforest is home to many animals"),
            TranscriptSegment::new(9000, 12000, "Birds and insects fill the canopy"),
        ]
    }

    #[test]
    fn test_transcript_segment_new() {
        let seg = TranscriptSegment::new(0, 5000, "hello world");
        assert_eq!(seg.start_ms, 0);
        assert_eq!(seg.end_ms, 5000);
        assert_eq!(seg.text, "hello world");
        assert!((seg.confidence - 1.0).abs() < f32::EPSILON);
        assert!(seg.speaker_id.is_none());
    }

    #[test]
    fn test_transcript_segment_duration() {
        let seg = TranscriptSegment::new(1000, 4500, "test");
        assert_eq!(seg.duration_ms(), 3500);
    }

    #[test]
    fn test_transcript_segment_with_meta() {
        let seg = TranscriptSegment::with_meta(0, 2000, "hello", 0.95, Some("speaker1".into()));
        assert!((seg.confidence - 0.95).abs() < 1e-5);
        assert_eq!(seg.speaker_id.as_deref(), Some("speaker1"));
    }

    #[test]
    fn test_index_asset_and_search() {
        let mut idx = TranscriptIndex::new();
        let id = Uuid::new_v4();
        idx.index_asset(id, sample_segments())
            .expect("should index");
        assert_eq!(idx.asset_count(), 1);
        assert!(idx.vocab_size() > 0);

        let results = idx.search("rainforest");
        assert!(!results.is_empty());
        assert_eq!(results[0].asset_id, id);
        assert!(results[0]
            .matched_tokens
            .contains(&"rainforest".to_string()));
    }

    #[test]
    fn test_search_returns_correct_timestamps() {
        let mut idx = TranscriptIndex::new();
        let id = Uuid::new_v4();
        idx.index_asset(id, sample_segments())
            .expect("should index");

        // "rainforest" appears in segments at 3000-6000 and 6000-9000
        let results = idx.search("rainforest");
        assert!(!results.is_empty());
        // All results should contain valid timestamps from the sample.
        for r in &results {
            assert!(r.segment.end_ms > r.segment.start_ms);
        }
    }

    #[test]
    fn test_search_multiple_assets() {
        let mut idx = TranscriptIndex::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        idx.index_asset(
            id1,
            vec![TranscriptSegment::new(0, 3000, "ocean creatures swim")],
        )
        .expect("should index");
        idx.index_asset(
            id2,
            vec![
                TranscriptSegment::new(0, 3000, "forest creatures run"),
                TranscriptSegment::new(3000, 6000, "creatures are everywhere"),
            ],
        )
        .expect("should index");

        assert_eq!(idx.asset_count(), 2);

        let results = idx.search("creatures");
        // Both assets should appear; id2 has higher TF so should rank first.
        let ids: Vec<Uuid> = results.iter().map(|r| r.asset_id).collect();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn test_search_no_results() {
        let mut idx = TranscriptIndex::new();
        let id = Uuid::new_v4();
        idx.index_asset(id, sample_segments())
            .expect("should index");
        let results = idx.search("zyrglgmx");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_phrase_all_tokens_present() {
        let mut idx = TranscriptIndex::new();
        let id = Uuid::new_v4();
        idx.index_asset(id, sample_segments())
            .expect("should index");

        // "rainforest animals" — both tokens appear in segment index 2
        let results = idx.search_phrase("rainforest animals");
        assert!(!results.is_empty());
        assert_eq!(results[0].asset_id, id);
    }

    #[test]
    fn test_search_phrase_missing_token_returns_empty() {
        let mut idx = TranscriptIndex::new();
        let id = Uuid::new_v4();
        idx.index_asset(id, sample_segments())
            .expect("should index");

        // "rainforest zyrglgmx" — second token absent
        let results = idx.search_phrase("rainforest zyrglgmx");
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_asset() {
        let mut idx = TranscriptIndex::new();
        let id = Uuid::new_v4();
        idx.index_asset(id, sample_segments())
            .expect("should index");
        assert_eq!(idx.asset_count(), 1);

        idx.remove_asset(id);
        assert_eq!(idx.asset_count(), 0);
        let results = idx.search("rainforest");
        assert!(results.is_empty());
    }

    #[test]
    fn test_replace_asset() {
        let mut idx = TranscriptIndex::new();
        let id = Uuid::new_v4();
        idx.index_asset(id, sample_segments())
            .expect("should index");

        // Replace with entirely different transcript.
        let new_segs = vec![TranscriptSegment::new(0, 5000, "cooking pasta recipe")];
        idx.index_asset(id, new_segs).expect("should replace");

        // Old content gone.
        assert!(idx.search("rainforest").is_empty());
        // New content present.
        assert!(!idx.search("pasta").is_empty());
        // Asset count unchanged.
        assert_eq!(idx.asset_count(), 1);
    }

    #[test]
    fn test_invalid_segment_rejected() {
        let mut idx = TranscriptIndex::new();
        let id = Uuid::new_v4();
        let bad = vec![TranscriptSegment::new(5000, 1000, "backwards")];
        assert!(idx.index_asset(id, bad).is_err());
    }

    #[test]
    fn test_get_segments() {
        let mut idx = TranscriptIndex::new();
        let id = Uuid::new_v4();
        idx.index_asset(id, sample_segments())
            .expect("should index");
        let segs = idx.get_segments(id);
        assert!(segs.is_some());
        assert_eq!(segs.map(|s| s.len()), Some(4));
        assert!(idx.get_segments(Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_total_segment_count() {
        let mut idx = TranscriptIndex::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        idx.index_asset(id1, sample_segments())
            .expect("should index");
        idx.index_asset(id2, vec![TranscriptSegment::new(0, 1000, "single segment")])
            .expect("should index");
        assert_eq!(idx.total_segment_count(), 5);
    }

    #[test]
    fn test_search_with_limit() {
        let mut idx = TranscriptIndex::new();
        for _ in 0..5 {
            let id = Uuid::new_v4();
            idx.index_asset(
                id,
                vec![TranscriptSegment::new(0, 1000, "common term everywhere")],
            )
            .expect("should index");
        }
        let results = idx.search_with_limit("common", 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_stop_words_not_indexed() {
        let mut idx = TranscriptIndex::new();
        let id = Uuid::new_v4();
        idx.index_asset(
            id,
            vec![TranscriptSegment::new(0, 1000, "the quick brown fox")],
        )
        .expect("should index");
        // "the" is a stop word; searching for it should find nothing.
        let results = idx.search("the");
        assert!(results.is_empty());
    }

    #[test]
    fn test_idf_boosts_rare_terms() {
        let mut idx = TranscriptIndex::new();
        // Add 10 assets all containing "common", only 1 containing "rare".
        for _ in 0..10 {
            let id = Uuid::new_v4();
            idx.index_asset(
                id,
                vec![TranscriptSegment::new(
                    0,
                    1000,
                    "common word appears everywhere",
                )],
            )
            .expect("should index");
        }
        let rare_id = Uuid::new_v4();
        idx.index_asset(
            rare_id,
            vec![TranscriptSegment::new(
                0,
                1000,
                "unique xylophone rare term common",
            )],
        )
        .expect("should index");

        // When searching for "rare", the only result should be rare_id.
        let results = idx.search("rare");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, rare_id);
    }

    #[test]
    fn test_empty_index_search() {
        let idx = TranscriptIndex::new();
        assert!(idx.search("hello").is_empty());
        assert!(idx.search_phrase("hello world").is_empty());
    }

    #[test]
    fn test_serialization_roundtrip_segment() {
        let seg = TranscriptSegment::with_meta(0, 5000, "test text", 0.88, Some("A".into()));
        let json = serde_json::to_string(&seg).expect("serialize");
        let back: TranscriptSegment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.start_ms, 0);
        assert_eq!(back.end_ms, 5000);
        assert_eq!(back.text, "test text");
        assert!((back.confidence - 0.88).abs() < 1e-5);
        assert_eq!(back.speaker_id.as_deref(), Some("A"));
    }

    #[test]
    fn test_config_custom_stop_words() {
        let config = TranscriptIndexConfig {
            min_token_len: 2,
            strip_punctuation: true,
            default_limit: 10,
            stop_words: vec!["custom".into()],
        };
        let mut idx = TranscriptIndex::with_config(config);
        let id = Uuid::new_v4();
        idx.index_asset(
            id,
            vec![TranscriptSegment::new(0, 1000, "custom word present")],
        )
        .expect("should index");
        // "custom" is a stop word; should not be indexed.
        assert!(idx.search("custom").is_empty());
        // "word" and "present" should be indexed.
        assert!(!idx.search("word").is_empty());
    }
}

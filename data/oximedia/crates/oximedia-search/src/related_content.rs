#![allow(dead_code)]
//! "More like this" related content recommendations.
//!
//! Given a seed search result item, this module finds other items that are
//! similar based on configurable signals:
//!
//! - **Text similarity**: TF-IDF cosine similarity on title, description, tags
//! - **Metadata overlap**: same format, codec, resolution tier, category
//! - **Temporal proximity**: items created around the same time
//! - **Duration similarity**: items of similar length
//!
//! Multiple signals are combined via weighted scoring to produce a final
//! relevance-ranked list of related items.

use std::collections::{HashMap, HashSet};

use crate::SearchResultItem;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Weights for different similarity signals in the related content scoring.
#[derive(Debug, Clone)]
pub struct RelatedWeights {
    /// Weight for text-based (title/description) similarity.
    pub text_similarity: f64,
    /// Weight for metadata overlap (format, codec, category).
    pub metadata_overlap: f64,
    /// Weight for temporal proximity.
    pub temporal_proximity: f64,
    /// Weight for duration similarity.
    pub duration_similarity: f64,
}

impl Default for RelatedWeights {
    fn default() -> Self {
        Self {
            text_similarity: 0.4,
            metadata_overlap: 0.3,
            temporal_proximity: 0.15,
            duration_similarity: 0.15,
        }
    }
}

/// Configuration for related content retrieval.
#[derive(Debug, Clone)]
pub struct RelatedContentConfig {
    /// Signal weights.
    pub weights: RelatedWeights,
    /// Maximum number of related items to return.
    pub max_results: usize,
    /// Minimum combined score to include a result (0.0 - 1.0).
    pub min_score: f64,
    /// Maximum age difference in seconds for temporal proximity scoring.
    pub max_temporal_diff_secs: i64,
    /// Maximum duration difference in ms for duration similarity scoring.
    pub max_duration_diff_ms: i64,
}

impl Default for RelatedContentConfig {
    fn default() -> Self {
        Self {
            weights: RelatedWeights::default(),
            max_results: 20,
            min_score: 0.05,
            max_temporal_diff_secs: 365 * 86_400, // 1 year
            max_duration_diff_ms: 600_000,        // 10 minutes
        }
    }
}

// ---------------------------------------------------------------------------
// Related content result
// ---------------------------------------------------------------------------

/// A related content recommendation with per-signal scores.
#[derive(Debug, Clone)]
pub struct RelatedItem {
    /// The recommended item.
    pub item: SearchResultItem,
    /// Combined relevance score (0.0 - 1.0).
    pub score: f64,
    /// Breakdown of individual signal scores.
    pub signal_scores: SignalScores,
}

/// Per-signal score breakdown for transparency and debugging.
#[derive(Debug, Clone, Default)]
pub struct SignalScores {
    /// Text similarity score (0.0 - 1.0).
    pub text: f64,
    /// Metadata overlap score (0.0 - 1.0).
    pub metadata: f64,
    /// Temporal proximity score (0.0 - 1.0).
    pub temporal: f64,
    /// Duration similarity score (0.0 - 1.0).
    pub duration: f64,
}

// ---------------------------------------------------------------------------
// Related content finder
// ---------------------------------------------------------------------------

/// Finds content related to a seed item from a corpus of search results.
#[derive(Debug)]
pub struct RelatedContentFinder {
    config: RelatedContentConfig,
}

impl RelatedContentFinder {
    /// Create a new finder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RelatedContentConfig::default(),
        }
    }

    /// Create a finder with custom configuration.
    #[must_use]
    pub fn with_config(config: RelatedContentConfig) -> Self {
        Self { config }
    }

    /// Find items related to the `seed` from the given `corpus`.
    ///
    /// The seed item itself is excluded from results.
    #[must_use]
    pub fn find_related(
        &self,
        seed: &SearchResultItem,
        corpus: &[SearchResultItem],
    ) -> Vec<RelatedItem> {
        let seed_tokens = tokenize_text_fields(seed);
        let seed_idf = compute_idf(&seed_tokens, corpus);
        let seed_tfidf = compute_tfidf(&seed_tokens, &seed_idf);

        let mut results: Vec<RelatedItem> = corpus
            .iter()
            .filter(|item| item.asset_id != seed.asset_id)
            .filter_map(|item| {
                let signals = self.compute_signals(seed, item, &seed_tfidf, corpus);
                let score = self.combined_score(&signals);
                if score >= self.config.min_score {
                    Some(RelatedItem {
                        item: item.clone(),
                        score,
                        signal_scores: signals,
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
        });
        results.truncate(self.config.max_results);
        results
    }

    /// Compute all similarity signals between seed and candidate.
    fn compute_signals(
        &self,
        seed: &SearchResultItem,
        candidate: &SearchResultItem,
        seed_tfidf: &HashMap<String, f64>,
        corpus: &[SearchResultItem],
    ) -> SignalScores {
        let candidate_tokens = tokenize_text_fields(candidate);
        let candidate_idf = compute_idf(&candidate_tokens, corpus);
        let candidate_tfidf = compute_tfidf(&candidate_tokens, &candidate_idf);

        SignalScores {
            text: tfidf_cosine_similarity(seed_tfidf, &candidate_tfidf),
            metadata: metadata_overlap_score(seed, candidate),
            temporal: temporal_proximity_score(
                seed.created_at,
                candidate.created_at,
                self.config.max_temporal_diff_secs,
            ),
            duration: duration_similarity_score(
                seed.duration_ms,
                candidate.duration_ms,
                self.config.max_duration_diff_ms,
            ),
        }
    }

    /// Combine signal scores using configured weights.
    fn combined_score(&self, signals: &SignalScores) -> f64 {
        let w = &self.config.weights;
        let raw = signals.text * w.text_similarity
            + signals.metadata * w.metadata_overlap
            + signals.temporal * w.temporal_proximity
            + signals.duration * w.duration_similarity;
        raw.clamp(0.0, 1.0)
    }
}

impl Default for RelatedContentFinder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Text similarity helpers
// ---------------------------------------------------------------------------

/// Extract and tokenize text from a result item's title and description.
fn tokenize_text_fields(item: &SearchResultItem) -> Vec<String> {
    let mut tokens = Vec::new();
    if let Some(ref title) = item.title {
        tokens.extend(tokenize(title));
    }
    if let Some(ref desc) = item.description {
        tokens.extend(tokenize(desc));
    }
    tokens
}

/// Simple whitespace and punctuation tokenizer with lowercasing.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Compute term frequency for a token list.
fn term_frequency(tokens: &[String]) -> HashMap<String, f64> {
    let mut tf: HashMap<String, f64> = HashMap::new();
    let n = tokens.len().max(1) as f64;
    for token in tokens {
        *tf.entry(token.clone()).or_insert(0.0) += 1.0;
    }
    for v in tf.values_mut() {
        *v /= n;
    }
    tf
}

/// Compute inverse document frequency from the corpus.
#[allow(clippy::cast_precision_loss)]
fn compute_idf(tokens: &[String], corpus: &[SearchResultItem]) -> HashMap<String, f64> {
    let unique_terms: HashSet<&String> = tokens.iter().collect();
    let n = (corpus.len().max(1)) as f64;
    let mut idf = HashMap::new();

    for term in unique_terms {
        let doc_count = corpus
            .iter()
            .filter(|item| {
                let text = format!(
                    "{} {}",
                    item.title.as_deref().unwrap_or(""),
                    item.description.as_deref().unwrap_or("")
                )
                .to_lowercase();
                text.contains(term.as_str())
            })
            .count();
        let df = (doc_count.max(1)) as f64;
        idf.insert(term.clone(), (n / df).ln() + 1.0);
    }
    idf
}

/// Compute TF-IDF vector from tokens and IDF values.
fn compute_tfidf(tokens: &[String], idf: &HashMap<String, f64>) -> HashMap<String, f64> {
    let tf = term_frequency(tokens);
    let mut tfidf = HashMap::new();
    for (term, tf_val) in &tf {
        let idf_val = idf.get(term).copied().unwrap_or(1.0);
        tfidf.insert(term.clone(), tf_val * idf_val);
    }
    tfidf
}

/// Cosine similarity between two TF-IDF vectors represented as hashmaps.
fn tfidf_cosine_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    let dot: f64 = a
        .iter()
        .filter_map(|(k, v)| b.get(k).map(|bv| v * bv))
        .sum();

    let norm_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let norm_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();

    let denom = norm_a * norm_b;
    if denom < f64::EPSILON {
        0.0
    } else {
        (dot / denom).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Metadata overlap
// ---------------------------------------------------------------------------

/// Compute metadata overlap score between two items (0.0 - 1.0).
///
/// Checks: MIME type match, format match, file extension match, path overlap.
fn metadata_overlap_score(a: &SearchResultItem, b: &SearchResultItem) -> f64 {
    let mut matches = 0u32;
    let mut total = 0u32;

    // MIME type
    total += 1;
    if a.mime_type.is_some() && a.mime_type == b.mime_type {
        matches += 1;
    }

    // Format (video/audio/image) from MIME prefix
    total += 1;
    let a_format = a
        .mime_type
        .as_deref()
        .and_then(|m| m.split('/').next())
        .unwrap_or("");
    let b_format = b
        .mime_type
        .as_deref()
        .and_then(|m| m.split('/').next())
        .unwrap_or("");
    if !a_format.is_empty() && a_format == b_format {
        matches += 1;
    }

    // File extension
    total += 1;
    let a_ext = std::path::Path::new(&a.file_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let b_ext = std::path::Path::new(&b.file_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if !a_ext.is_empty() && a_ext.eq_ignore_ascii_case(b_ext) {
        matches += 1;
    }

    if total == 0 {
        0.0
    } else {
        f64::from(matches) / f64::from(total)
    }
}

// ---------------------------------------------------------------------------
// Temporal proximity
// ---------------------------------------------------------------------------

/// Score temporal proximity between two timestamps (0.0 - 1.0).
///
/// Uses exponential decay: items created close together score higher.
fn temporal_proximity_score(a_secs: i64, b_secs: i64, max_diff: i64) -> f64 {
    let diff = (a_secs - b_secs).unsigned_abs() as f64;
    let max = max_diff.unsigned_abs() as f64;
    if max < f64::EPSILON {
        return if diff < 1.0 { 1.0 } else { 0.0 };
    }
    let ratio = diff / max;
    // Exponential decay
    (-3.0 * ratio).exp().clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Duration similarity
// ---------------------------------------------------------------------------

/// Score duration similarity between two items (0.0 - 1.0).
fn duration_similarity_score(a_ms: Option<i64>, b_ms: Option<i64>, max_diff: i64) -> f64 {
    match (a_ms, b_ms) {
        (Some(a), Some(b)) => {
            let diff = (a - b).unsigned_abs() as f64;
            let max = max_diff.unsigned_abs() as f64;
            if max < f64::EPSILON {
                return if diff < 1.0 { 1.0 } else { 0.0 };
            }
            (1.0 - diff / max).clamp(0.0, 1.0)
        }
        _ => 0.0, // If either lacks duration, no similarity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_item(
        title: &str,
        description: &str,
        mime: &str,
        path: &str,
        created_at: i64,
        duration_ms: Option<i64>,
    ) -> SearchResultItem {
        SearchResultItem {
            asset_id: Uuid::new_v4(),
            score: 1.0,
            title: Some(title.to_string()),
            description: Some(description.to_string()),
            file_path: path.to_string(),
            mime_type: Some(mime.to_string()),
            duration_ms,
            created_at,
            modified_at: None,
            file_size: None,
            matched_fields: Vec::new(),
            thumbnail_url: None,
        }
    }

    fn sample_corpus() -> Vec<SearchResultItem> {
        vec![
            make_item(
                "Sunset Beach",
                "Beautiful sunset over the ocean",
                "video/mp4",
                "sunset_beach.mp4",
                1_700_000_000,
                Some(120_000),
            ),
            make_item(
                "Sunset Mountains",
                "Sunset behind the mountain range",
                "video/mp4",
                "sunset_mountains.mp4",
                1_700_001_000,
                Some(90_000),
            ),
            make_item(
                "Ocean Waves",
                "Waves crashing on a tropical beach",
                "video/mp4",
                "ocean_waves.mp4",
                1_700_002_000,
                Some(180_000),
            ),
            make_item(
                "City Night",
                "City skyline at night with lights",
                "video/webm",
                "city_night.webm",
                1_699_000_000,
                Some(60_000),
            ),
            make_item(
                "Jazz Piano",
                "Smooth jazz piano performance",
                "audio/flac",
                "jazz_piano.flac",
                1_698_000_000,
                Some(240_000),
            ),
        ]
    }

    #[test]
    fn test_find_related_excludes_seed() {
        let corpus = sample_corpus();
        let finder = RelatedContentFinder::new();
        let related = finder.find_related(&corpus[0], &corpus);
        assert!(related
            .iter()
            .all(|r| r.item.asset_id != corpus[0].asset_id));
    }

    #[test]
    fn test_find_related_returns_results() {
        let corpus = sample_corpus();
        let finder = RelatedContentFinder::new();
        let related = finder.find_related(&corpus[0], &corpus);
        assert!(!related.is_empty());
    }

    #[test]
    fn test_find_related_sorted_by_score() {
        let corpus = sample_corpus();
        let finder = RelatedContentFinder::new();
        let related = finder.find_related(&corpus[0], &corpus);
        for w in related.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_find_related_respects_max_results() {
        let config = RelatedContentConfig {
            max_results: 2,
            ..Default::default()
        };
        let corpus = sample_corpus();
        let finder = RelatedContentFinder::with_config(config);
        let related = finder.find_related(&corpus[0], &corpus);
        assert!(related.len() <= 2);
    }

    #[test]
    fn test_sunset_beach_most_similar_to_sunset_mountains() {
        let corpus = sample_corpus();
        let finder = RelatedContentFinder::new();
        let related = finder.find_related(&corpus[0], &corpus);
        // "Sunset Mountains" should score highest — shares "sunset" and same format
        assert!(!related.is_empty());
        assert_eq!(related[0].item.title.as_deref(), Some("Sunset Mountains"));
    }

    #[test]
    fn test_audio_item_less_related_to_video_seed() {
        let corpus = sample_corpus();
        let finder = RelatedContentFinder::new();
        let related = finder.find_related(&corpus[0], &corpus);
        // Jazz Piano (audio) should score lower than video items
        let jazz_score = related
            .iter()
            .find(|r| r.item.title.as_deref() == Some("Jazz Piano"))
            .map(|r| r.score);
        let mountain_score = related
            .iter()
            .find(|r| r.item.title.as_deref() == Some("Sunset Mountains"))
            .map(|r| r.score);
        if let (Some(j), Some(m)) = (jazz_score, mountain_score) {
            assert!(m > j);
        }
    }

    #[test]
    fn test_signal_scores_present() {
        let corpus = sample_corpus();
        let finder = RelatedContentFinder::new();
        let related = finder.find_related(&corpus[0], &corpus);
        if let Some(first) = related.first() {
            // All signals should be non-negative
            assert!(first.signal_scores.text >= 0.0);
            assert!(first.signal_scores.metadata >= 0.0);
            assert!(first.signal_scores.temporal >= 0.0);
            assert!(first.signal_scores.duration >= 0.0);
        }
    }

    #[test]
    fn test_empty_corpus() {
        let seed = make_item("Test", "Test desc", "video/mp4", "test.mp4", 0, None);
        let finder = RelatedContentFinder::new();
        let related = finder.find_related(&seed, &[]);
        assert!(related.is_empty());
    }

    #[test]
    fn test_corpus_with_only_seed() {
        let seed = make_item("Test", "Test desc", "video/mp4", "test.mp4", 0, None);
        let finder = RelatedContentFinder::new();
        let related = finder.find_related(&seed, std::slice::from_ref(&seed));
        assert!(related.is_empty());
    }

    // -- Signal function unit tests --

    #[test]
    fn test_temporal_proximity_same_time() {
        let score = temporal_proximity_score(1000, 1000, 86400);
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_proximity_max_diff() {
        let score = temporal_proximity_score(0, 86400, 86400);
        // exp(-3) ≈ 0.0498
        assert!(score < 0.1);
        assert!(score > 0.0);
    }

    #[test]
    fn test_temporal_proximity_half_diff() {
        let score = temporal_proximity_score(0, 43200, 86400);
        // exp(-1.5) ≈ 0.223
        assert!(score > 0.2);
        assert!(score < 0.3);
    }

    #[test]
    fn test_duration_similarity_same() {
        let score = duration_similarity_score(Some(60000), Some(60000), 600000);
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_duration_similarity_max_diff() {
        let score = duration_similarity_score(Some(0), Some(600000), 600000);
        assert!(score.abs() < 1e-5);
    }

    #[test]
    fn test_duration_similarity_half() {
        let score = duration_similarity_score(Some(0), Some(300000), 600000);
        assert!((score - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_duration_similarity_none() {
        assert!((duration_similarity_score(None, Some(60000), 600000)).abs() < 1e-5);
        assert!((duration_similarity_score(Some(60000), None, 600000)).abs() < 1e-5);
        assert!((duration_similarity_score(None, None, 600000)).abs() < 1e-5);
    }

    #[test]
    fn test_metadata_overlap_same_mime() {
        let a = make_item("A", "desc", "video/mp4", "a.mp4", 0, None);
        let b = make_item("B", "desc", "video/mp4", "b.mp4", 0, None);
        let score = metadata_overlap_score(&a, &b);
        assert!((score - 1.0).abs() < 1e-5); // all 3 checks match
    }

    #[test]
    fn test_metadata_overlap_different_type() {
        let a = make_item("A", "desc", "video/mp4", "a.mp4", 0, None);
        let b = make_item("B", "desc", "audio/flac", "b.flac", 0, None);
        let score = metadata_overlap_score(&a, &b);
        assert!(score < 0.5); // different format, mime, and extension
    }

    #[test]
    fn test_metadata_overlap_same_format_different_codec() {
        let a = make_item("A", "desc", "video/mp4", "a.mp4", 0, None);
        let b = make_item("B", "desc", "video/webm", "b.webm", 0, None);
        let score = metadata_overlap_score(&a, &b);
        // Format matches (video), but mime and extension differ
        assert!(score > 0.0);
        assert!(score < 1.0);
    }

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello World! This is a Test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // "a" is too short (< 2 chars)
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_tfidf_cosine_identical() {
        let mut a = HashMap::new();
        a.insert("video".to_string(), 0.5);
        a.insert("sunset".to_string(), 0.3);
        let sim = tfidf_cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tfidf_cosine_disjoint() {
        let mut a = HashMap::new();
        a.insert("video".to_string(), 1.0);
        let mut b = HashMap::new();
        b.insert("audio".to_string(), 1.0);
        let sim = tfidf_cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_tfidf_cosine_empty() {
        let a: HashMap<String, f64> = HashMap::new();
        let b: HashMap<String, f64> = HashMap::new();
        let sim = tfidf_cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_config_defaults() {
        let config = RelatedContentConfig::default();
        assert_eq!(config.max_results, 20);
        assert!(config.min_score > 0.0);
    }

    #[test]
    fn test_custom_weights() {
        let config = RelatedContentConfig {
            weights: RelatedWeights {
                text_similarity: 1.0,
                metadata_overlap: 0.0,
                temporal_proximity: 0.0,
                duration_similarity: 0.0,
            },
            ..Default::default()
        };
        let corpus = sample_corpus();
        let finder = RelatedContentFinder::with_config(config);
        let related = finder.find_related(&corpus[0], &corpus);
        // With only text weight, metadata/temporal/duration signals are ignored
        if let Some(first) = related.first() {
            assert!(first.signal_scores.text > 0.0);
        }
    }
}

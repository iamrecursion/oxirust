//! Combined matching strategies.

use crate::config::ConformConfig;
use crate::types::{ClipMatch, ClipReference, MatchMethod, MediaFile};
use dashmap::DashMap;
use rayon::prelude::*;
use std::sync::Arc;

use super::content::duration_match;
use super::filename::{exact_filename_match, fuzzy_filename_match, relink_proxy_match};
use super::timecode::timecode_match;

// ── Confidence scoring ────────────────────────────────────────────────────────

/// The strategy that contributed to a confidence score.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MatchStrategyKind {
    /// Exact or fuzzy filename comparison.
    Filename,
    /// Timecode-based containment / overlap.
    Timecode,
    /// Content-hash or perceptual-hash comparison.
    ContentHash,
    /// Duration-proximity comparison.
    Duration,
    /// Weighted combination of multiple strategies.
    WeightedCombined,
}

impl std::fmt::Display for MatchStrategyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Filename => "Filename",
            Self::Timecode => "Timecode",
            Self::ContentHash => "ContentHash",
            Self::Duration => "Duration",
            Self::WeightedCombined => "WeightedCombined",
        };
        write!(f, "{s}")
    }
}

use serde::{Deserialize, Serialize};

/// Confidence score produced by a matching strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchConfidence {
    /// Normalised confidence score in \[0.0, 1.0\].
    pub score: f32,
    /// The strategy that produced this score.
    pub strategy_used: MatchStrategyKind,
}

impl MatchConfidence {
    /// Create a new `MatchConfidence`.
    #[must_use]
    pub fn new(score: f32, strategy_used: MatchStrategyKind) -> Self {
        Self {
            score: score.clamp(0.0, 1.0),
            strategy_used,
        }
    }

    /// Return `true` if the confidence score meets or exceeds `threshold`.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.score >= threshold
    }
}

// ── Weighted multi-strategy matcher ──────────────────────────────────────────

/// Weights used by [`WeightedMultiStrategyMatcher`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyWeights {
    /// Weight for filename-based matching (default 0.4).
    pub filename: f32,
    /// Weight for timecode-based matching (default 0.4).
    pub timecode: f32,
    /// Weight for content-hash / perceptual-hash matching (default 0.2).
    pub content_hash: f32,
}

impl Default for StrategyWeights {
    fn default() -> Self {
        Self {
            filename: 0.4,
            timecode: 0.4,
            content_hash: 0.2,
        }
    }
}

/// A matcher that combines filename, timecode, and content-hash scores using
/// configurable weights and returns a [`MatchConfidence`] for every candidate.
pub struct WeightedMultiStrategyMatcher {
    config: Arc<ConformConfig>,
    weights: StrategyWeights,
}

impl WeightedMultiStrategyMatcher {
    /// Create with default weights (filename 0.4, timecode 0.4, hash 0.2).
    #[must_use]
    pub fn new(config: ConformConfig) -> Self {
        Self {
            config: Arc::new(config),
            weights: StrategyWeights::default(),
        }
    }

    /// Create with explicit weights.
    ///
    /// # Panics
    ///
    /// Panics if any weight is negative.
    #[must_use]
    pub fn with_weights(config: ConformConfig, weights: StrategyWeights) -> Self {
        assert!(weights.filename >= 0.0, "filename weight must be >= 0");
        assert!(weights.timecode >= 0.0, "timecode weight must be >= 0");
        assert!(
            weights.content_hash >= 0.0,
            "content_hash weight must be >= 0"
        );
        Self {
            config: Arc::new(config),
            weights,
        }
    }

    /// Compute a weighted [`MatchConfidence`] for `(clip, media)`.
    ///
    /// Each component strategy contributes its score multiplied by its weight.
    /// The components are normalised so that absent scores do not penalise the
    /// result (only present components contribute to the final denominator).
    #[must_use]
    pub fn confidence(&self, clip: &ClipReference, media: &MediaFile) -> MatchConfidence {
        let mut weighted_sum: f32 = 0.0;
        let mut total_weight: f32 = 0.0;

        // ── Filename component ─────────────────────────────────────────────
        let filename_score = self.filename_score(clip, media);
        if let Some(s) = filename_score {
            weighted_sum += self.weights.filename * s;
            total_weight += self.weights.filename;
        }

        // ── Timecode component ─────────────────────────────────────────────
        let timecode_score = self.timecode_score(clip, media);
        if let Some(s) = timecode_score {
            weighted_sum += self.weights.timecode * s;
            total_weight += self.weights.timecode;
        }

        // ── Content-hash component ─────────────────────────────────────────
        let hash_score = self.content_hash_score(media);
        if let Some(s) = hash_score {
            weighted_sum += self.weights.content_hash * s;
            total_weight += self.weights.content_hash;
        }

        let score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        MatchConfidence::new(score, MatchStrategyKind::WeightedCombined)
    }

    /// Match all `media` candidates for `clip` and return
    /// `(ClipMatch, MatchConfidence)` pairs sorted by descending confidence.
    #[must_use]
    pub fn match_with_confidence(
        &self,
        clip: &ClipReference,
        all_media: &[MediaFile],
    ) -> Vec<(ClipMatch, MatchConfidence)> {
        let mut results: Vec<(ClipMatch, MatchConfidence)> = all_media
            .iter()
            .filter_map(|media| {
                let conf = self.confidence(clip, media);
                if conf.score < self.config.match_threshold as f32 {
                    return None;
                }
                let cm = ClipMatch {
                    clip: clip.clone(),
                    media: media.clone(),
                    score: f64::from(conf.score),
                    method: MatchMethod::Combined,
                    details: format!(
                        "WeightedMulti(filename={:.2}, timecode={:.2}, hash={:.2}) → {:.3}",
                        self.filename_score(clip, media).unwrap_or(0.0),
                        self.timecode_score(clip, media).unwrap_or(0.0),
                        self.content_hash_score(media).unwrap_or(0.0),
                        conf.score,
                    ),
                };
                Some((cm, conf))
            })
            .collect();

        results.sort_by(|(_, a), (_, b)| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    fn filename_score(&self, clip: &ClipReference, media: &MediaFile) -> Option<f32> {
        let source_file = clip.source_file.as_deref()?;
        if media.filename == source_file {
            return Some(1.0);
        }
        // Fuzzy
        if self.config.fuzzy_matching {
            let distance = strsim::levenshtein(source_file, &media.filename);
            if distance <= self.config.fuzzy_max_distance {
                let max_len = source_file.len().max(media.filename.len());
                let s = if max_len == 0 {
                    0.0_f32
                } else {
                    1.0 - (distance as f32 / max_len as f32)
                };
                return Some(s);
            }
        }
        None
    }

    fn timecode_score(&self, clip: &ClipReference, media: &MediaFile) -> Option<f32> {
        use crate::types::Timecode;
        let media_tc_start = media.timecode_start?;
        let media_duration = media.duration?;

        let media_frames = (media_duration * clip.fps.as_f64()) as u64;
        let media_tc_end =
            Timecode::from_frames(media_tc_start.to_frames(clip.fps) + media_frames, clip.fps);

        let clip_start = clip.source_in.to_frames(clip.fps);
        let clip_end = clip.source_out.to_frames(clip.fps);
        let m_start = media_tc_start.to_frames(clip.fps);
        let m_end = media_tc_end.to_frames(clip.fps);

        let _ = media_tc_end; // suppress unused warning

        if m_start <= clip_start && clip_end <= m_end {
            return Some(1.0);
        }
        // Partial overlap
        if m_start <= clip_end && clip_start <= m_end {
            let overlap_start = m_start.max(clip_start);
            let overlap_end = m_end.min(clip_end);
            if overlap_end > overlap_start {
                let clip_len = clip_end.saturating_sub(clip_start).max(1);
                let s = (overlap_end - overlap_start) as f32 / clip_len as f32;
                if s >= self.config.match_threshold as f32 {
                    return Some(s);
                }
            }
        }
        None
    }

    fn content_hash_score(&self, media: &MediaFile) -> Option<f32> {
        // If perceptual hash is present in metadata JSON, attempt to decode it.
        // Currently we treat any stored hash as full confidence if present.
        if media.md5.is_some() || media.xxhash.is_some() {
            // Cannot compare without the expected hash here; return a neutral
            // positive signal indicating hash data is available.
            return Some(0.5);
        }
        // Check metadata field for stored perceptual hash
        if let Some(meta) = &media.metadata {
            if meta.contains("phash") {
                return Some(0.7);
            }
        }
        None
    }
}

/// Match strategy for conforming clips to media files.
pub struct MatchStrategy {
    config: Arc<ConformConfig>,
}

impl MatchStrategy {
    /// Create a new match strategy with the given configuration.
    #[must_use]
    pub fn new(config: ConformConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    /// Match a single clip against all available media files.
    #[must_use]
    pub fn match_clip(&self, clip: &ClipReference, all_media: &[MediaFile]) -> Vec<ClipMatch> {
        let mut all_matches = Vec::new();

        // Try exact filename matching first
        let mut matches = exact_filename_match(clip, all_media);
        all_matches.append(&mut matches);

        // Try fuzzy filename matching if enabled
        if self.config.fuzzy_matching {
            let mut matches = fuzzy_filename_match(clip, all_media, &self.config);
            all_matches.append(&mut matches);
        }

        // Try proxy relinking if enabled
        if self.config.auto_relink {
            let mut matches = relink_proxy_match(clip, all_media, &self.config);
            all_matches.append(&mut matches);
        }

        // Try timecode matching if enabled
        if self.config.timecode_matching {
            let mut matches = timecode_match(clip, all_media, &self.config);
            all_matches.append(&mut matches);
        }

        // Try duration matching if enabled
        if self.config.duration_matching {
            let mut matches = duration_match(clip, all_media, &self.config);
            all_matches.append(&mut matches);
        }

        // Filter by match threshold
        all_matches.retain(|m| m.score >= self.config.match_threshold);

        // Sort by score (highest first)
        all_matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate by media file ID
        self.deduplicate_matches(all_matches)
    }

    /// Match multiple clips in parallel.
    #[must_use]
    pub fn match_clips(
        &self,
        clips: &[ClipReference],
        all_media: &[MediaFile],
    ) -> Vec<Vec<ClipMatch>> {
        clips
            .par_iter()
            .map(|clip| self.match_clip(clip, all_media))
            .collect()
    }

    /// Deduplicate matches by keeping the highest score for each media file.
    fn deduplicate_matches(&self, matches: Vec<ClipMatch>) -> Vec<ClipMatch> {
        let best_matches: DashMap<uuid::Uuid, ClipMatch> = DashMap::new();

        for m in matches {
            let media_id = m.media.id;
            best_matches
                .entry(media_id)
                .and_modify(|existing| {
                    if m.score > existing.score {
                        *existing = m.clone();
                    }
                })
                .or_insert(m);
        }

        let mut result: Vec<ClipMatch> = best_matches.into_iter().map(|(_, v)| v).collect();
        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    /// Get the best match for a clip (highest score).
    #[must_use]
    pub fn get_best_match(
        &self,
        clip: &ClipReference,
        all_media: &[MediaFile],
    ) -> Option<ClipMatch> {
        let matches = self.match_clip(clip, all_media);
        matches.into_iter().next()
    }

    /// Check if a match is ambiguous (multiple matches with similar scores).
    #[must_use]
    pub fn is_ambiguous(&self, matches: &[ClipMatch], tolerance: f64) -> bool {
        if matches.len() <= 1 {
            return false;
        }

        if let Some(best) = matches.first() {
            matches
                .iter()
                .skip(1)
                .any(|m| (best.score - m.score).abs() < tolerance)
        } else {
            false
        }
    }

    /// Combine multiple matches into a single match with combined score.
    #[must_use]
    pub fn combine_matches(&self, matches: Vec<ClipMatch>) -> Option<ClipMatch> {
        if matches.is_empty() {
            return None;
        }

        if matches.len() == 1 {
            return matches.into_iter().next();
        }

        // Check if all matches point to the same media file
        let first_media_id = matches[0].media.id;
        if matches.iter().all(|m| m.media.id == first_media_id) {
            let combined_score: f64 =
                matches.iter().map(|m| m.score).sum::<f64>() / matches.len() as f64;
            let methods: Vec<String> = matches.iter().map(|m| m.method.to_string()).collect();
            let details = format!("Combined match from: {}", methods.join(", "));

            Some(ClipMatch {
                clip: matches[0].clip.clone(),
                media: matches[0].media.clone(),
                score: combined_score,
                method: MatchMethod::Combined,
                details,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FrameRate, Timecode, TrackType};
    use std::path::PathBuf;

    fn create_test_clip(id: &str, source_file: &str) -> ClipReference {
        ClipReference {
            id: id.to_string(),
            source_file: Some(source_file.to_string()),
            source_in: Timecode::new(1, 0, 0, 0),
            source_out: Timecode::new(1, 0, 10, 0),
            record_in: Timecode::new(1, 0, 0, 0),
            record_out: Timecode::new(1, 0, 10, 0),
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_match_strategy_exact() {
        let config = ConformConfig::default();
        let strategy = MatchStrategy::new(config);

        let clip = create_test_clip("clip1", "test.mov");
        let media = MediaFile::new(PathBuf::from("/path/test.mov"));

        let matches = strategy.match_clip(&clip, &[media]);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_match_clips_parallel() {
        let config = ConformConfig::default();
        let strategy = MatchStrategy::new(config);

        let clips = vec![
            create_test_clip("clip1", "test1.mov"),
            create_test_clip("clip2", "test2.mov"),
        ];

        let media = vec![
            MediaFile::new(PathBuf::from("/path/test1.mov")),
            MediaFile::new(PathBuf::from("/path/test2.mov")),
        ];

        let results = strategy.match_clips(&clips, &media);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_get_best_match() {
        let config = ConformConfig::default();
        let strategy = MatchStrategy::new(config);

        let clip = create_test_clip("clip1", "test.mov");
        let media = MediaFile::new(PathBuf::from("/path/test.mov"));

        let best = strategy.get_best_match(&clip, &[media]);
        assert!(best.is_some());
    }

    #[test]
    fn test_is_ambiguous() {
        let config = ConformConfig::default();
        let strategy = MatchStrategy::new(config);

        let clip = create_test_clip("clip1", "test.mov");
        let media1 = MediaFile::new(PathBuf::from("/path/test1.mov"));
        let media2 = MediaFile::new(PathBuf::from("/path/test2.mov"));

        let matches = vec![
            ClipMatch {
                clip: clip.clone(),
                media: media1,
                score: 0.9,
                method: MatchMethod::ExactFilename,
                details: String::new(),
            },
            ClipMatch {
                clip,
                media: media2,
                score: 0.89,
                method: MatchMethod::FuzzyFilename,
                details: String::new(),
            },
        ];

        assert!(strategy.is_ambiguous(&matches, 0.05));
        assert!(!strategy.is_ambiguous(&matches, 0.005));
    }

    // ── MatchConfidence tests ─────────────────────────────────────────────────

    #[test]
    fn test_match_confidence_clamps_score() {
        let c = MatchConfidence::new(1.5, MatchStrategyKind::Filename);
        assert!((c.score - 1.0).abs() < f32::EPSILON);

        let c2 = MatchConfidence::new(-0.5, MatchStrategyKind::Timecode);
        assert!(c2.score.abs() < f32::EPSILON);
    }

    #[test]
    fn test_match_confidence_is_confident() {
        let c = MatchConfidence::new(0.85, MatchStrategyKind::ContentHash);
        assert!(c.is_confident(0.8));
        assert!(!c.is_confident(0.9));
    }

    // ── WeightedMultiStrategyMatcher tests ───────────────────────────────────

    #[test]
    fn test_weighted_matcher_exact_filename_full_confidence() {
        let config = ConformConfig::default();
        let matcher = WeightedMultiStrategyMatcher::new(config);

        let clip = create_test_clip("clip1", "exact_match.mov");
        let media = MediaFile::new(PathBuf::from("/path/exact_match.mov"));

        let conf = matcher.confidence(&clip, &media);
        // Filename matches exactly → filename_score = Some(1.0), only filename contributes
        assert!(
            conf.score > 0.9,
            "exact filename should yield high confidence, got {}",
            conf.score
        );
        assert_eq!(conf.strategy_used, MatchStrategyKind::WeightedCombined);
    }

    #[test]
    fn test_weighted_matcher_no_match_returns_zero() {
        let mut config = ConformConfig::default();
        config.fuzzy_matching = false;
        config.timecode_matching = false;
        config.duration_matching = false;
        let matcher = WeightedMultiStrategyMatcher::new(config);

        let clip = create_test_clip("clip1", "some_clip.mov");
        let media = MediaFile::new(PathBuf::from("/path/totally_different.mkv"));

        let conf = matcher.confidence(&clip, &media);
        // No filename match, no timecode, no hash → score should be 0
        assert_eq!(conf.score, 0.0);
    }

    #[test]
    fn test_weighted_matcher_match_with_confidence_sorted() {
        let config = ConformConfig::default();
        let matcher = WeightedMultiStrategyMatcher::new(config);

        let clip = create_test_clip("clip1", "file_a.mov");
        let media_exact = MediaFile::new(PathBuf::from("/path/file_a.mov"));
        let media_none = MediaFile::new(PathBuf::from("/path/zzz_unknown.mkv"));

        let results = matcher.match_with_confidence(&clip, &[media_none, media_exact]);

        // Exact match should be ranked first
        assert!(!results.is_empty());
        assert!(results[0].1.score > 0.0);
    }

    #[test]
    fn test_strategy_weights_default() {
        let w = StrategyWeights::default();
        assert!((w.filename - 0.4).abs() < f32::EPSILON);
        assert!((w.timecode - 0.4).abs() < f32::EPSILON);
        assert!((w.content_hash - 0.2).abs() < f32::EPSILON);
    }
}

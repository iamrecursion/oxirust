//! Automatic chapter point generation from scene analysis.
//!
//! Generates chapter markers by analysing scored scenes and detecting
//! structural boundaries in video content:
//!
//! - **Scene clustering**: Groups consecutive similar scenes into chapters
//! - **Content-type transitions**: Detects shifts between action/dialogue/etc.
//! - **Energy-based segmentation**: Finds natural breaks via interest curve valleys
//! - **Temporal constraints**: Enforces minimum/maximum chapter duration
//! - **Title generation**: Produces human-readable chapter titles
//!
//! # Example
//!
//! ```
//! use oximedia_auto::auto_chaptering::{ChapterGenerator, ChapterConfig};
//!
//! let config = ChapterConfig::default();
//! let generator = ChapterGenerator::new(config);
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use crate::scoring::{ContentType, InterestCurve, ScoredScene, Sentiment};
use oximedia_core::Timestamp;

// ---------------------------------------------------------------------------
// Chapter
// ---------------------------------------------------------------------------

/// A generated chapter marker.
#[derive(Debug, Clone)]
pub struct Chapter {
    /// Zero-based chapter index.
    pub index: usize,
    /// Start timestamp (inclusive).
    pub start: Timestamp,
    /// End timestamp (exclusive).
    pub end: Timestamp,
    /// Generated title.
    pub title: String,
    /// Dominant content type within this chapter.
    pub content_type: ContentType,
    /// Dominant sentiment within this chapter.
    pub sentiment: Sentiment,
    /// Average importance score of scenes in this chapter.
    pub average_importance: f64,
    /// Number of scenes aggregated into this chapter.
    pub scene_count: usize,
}

impl Chapter {
    /// Duration of this chapter in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }

    /// Check if this chapter contains the given timestamp.
    #[must_use]
    pub fn contains(&self, ts: Timestamp) -> bool {
        ts.pts >= self.start.pts && ts.pts < self.end.pts
    }

    /// Format as a simple `HH:MM:SS - Title` line.
    #[must_use]
    pub fn format_timestamp_line(&self) -> String {
        let total_secs = self.start.pts / 1000;
        let h = total_secs / 3600;
        let m = (total_secs % 3600) / 60;
        let s = total_secs % 60;
        format!("{h:02}:{m:02}:{s:02} - {}", self.title)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for automatic chapter generation.
#[derive(Debug, Clone)]
pub struct ChapterConfig {
    /// Minimum chapter duration in milliseconds.
    pub min_chapter_duration_ms: i64,
    /// Maximum chapter duration in milliseconds.
    pub max_chapter_duration_ms: i64,
    /// Target number of chapters (0 = auto-determine).
    pub target_chapter_count: usize,
    /// Minimum importance drop between scenes to trigger a chapter boundary.
    pub boundary_importance_drop: f64,
    /// Whether to detect content-type transitions as chapter boundaries.
    pub use_content_type_transitions: bool,
    /// Whether to use interest curve valleys as chapter boundaries.
    pub use_interest_valleys: bool,
    /// Interest curve valley threshold (lower = more aggressive splitting).
    pub valley_threshold: f64,
    /// Whether to generate chapter titles automatically.
    pub auto_title: bool,
    /// Prefix to prepend to generated chapter titles.
    pub title_prefix: String,
}

impl Default for ChapterConfig {
    fn default() -> Self {
        Self {
            min_chapter_duration_ms: 30_000,  // 30 seconds
            max_chapter_duration_ms: 600_000, // 10 minutes
            target_chapter_count: 0,
            boundary_importance_drop: 0.15,
            use_content_type_transitions: true,
            use_interest_valleys: true,
            valley_threshold: 0.35,
            auto_title: true,
            title_prefix: String::new(),
        }
    }
}

impl ChapterConfig {
    /// Create default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum chapter duration.
    #[must_use]
    pub const fn with_min_duration_ms(mut self, ms: i64) -> Self {
        self.min_chapter_duration_ms = ms;
        self
    }

    /// Set the maximum chapter duration.
    #[must_use]
    pub const fn with_max_duration_ms(mut self, ms: i64) -> Self {
        self.max_chapter_duration_ms = ms;
        self
    }

    /// Set the target chapter count (0 = auto).
    #[must_use]
    pub const fn with_target_count(mut self, count: usize) -> Self {
        self.target_chapter_count = count;
        self
    }

    /// Set the valley threshold for interest-based splitting.
    #[must_use]
    pub fn with_valley_threshold(mut self, t: f64) -> Self {
        self.valley_threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if self.min_chapter_duration_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.min_chapter_duration_ms,
            });
        }
        if self.max_chapter_duration_ms <= self.min_chapter_duration_ms {
            return Err(AutoError::invalid_parameter(
                "max_chapter_duration_ms",
                "must be greater than min_chapter_duration_ms",
            ));
        }
        if !(0.0..=1.0).contains(&self.valley_threshold) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.valley_threshold,
                min: 0.0,
                max: 1.0,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Boundary detection
// ---------------------------------------------------------------------------

/// Reason a chapter boundary was placed at a specific position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryReason {
    /// Content type changed (e.g. action -> dialogue).
    ContentTypeChange,
    /// Interest curve valley (natural energy dip).
    InterestValley,
    /// Large importance score drop between consecutive scenes.
    ImportanceDrop,
    /// Forced split to honour maximum chapter duration.
    MaxDurationSplit,
}

/// A candidate chapter boundary between two consecutive scenes.
#[derive(Debug, Clone)]
struct BoundaryCandidate {
    /// Index of the scene *after* this boundary in the input slice.
    scene_index: usize,
    /// Timestamp of the boundary.
    timestamp: Timestamp,
    /// Strength of this boundary (higher = more confident).
    strength: f64,
    /// Why this boundary was detected.
    reason: BoundaryReason,
}

/// Detect boundaries from content-type transitions.
fn detect_content_type_boundaries(scenes: &[ScoredScene]) -> Vec<BoundaryCandidate> {
    let mut boundaries = Vec::new();
    for i in 1..scenes.len() {
        if scenes[i].content_type != scenes[i - 1].content_type {
            boundaries.push(BoundaryCandidate {
                scene_index: i,
                timestamp: scenes[i].start,
                strength: 0.6,
                reason: BoundaryReason::ContentTypeChange,
            });
        }
    }
    boundaries
}

/// Detect boundaries from importance score drops.
fn detect_importance_drop_boundaries(
    scenes: &[ScoredScene],
    min_drop: f64,
) -> Vec<BoundaryCandidate> {
    let mut boundaries = Vec::new();
    for i in 1..scenes.len() {
        let drop = scenes[i - 1].adjusted_score() - scenes[i].adjusted_score();
        if drop >= min_drop {
            boundaries.push(BoundaryCandidate {
                scene_index: i,
                timestamp: scenes[i].start,
                strength: drop.min(1.0),
                reason: BoundaryReason::ImportanceDrop,
            });
        }
    }
    boundaries
}

/// Detect boundaries from interest curve valleys.
fn detect_interest_valley_boundaries(
    scenes: &[ScoredScene],
    curve: &InterestCurve,
    threshold: f64,
) -> Vec<BoundaryCandidate> {
    let valleys = curve.find_valleys(threshold);
    let mut boundaries = Vec::new();

    for (valley_ts, valley_score) in &valleys {
        // Find the scene closest to this valley
        if let Some((idx, _)) = scenes
            .iter()
            .enumerate()
            .skip(1)
            .min_by_key(|(_, s)| (s.start.pts - valley_ts.pts).abs())
        {
            boundaries.push(BoundaryCandidate {
                scene_index: idx,
                timestamp: *valley_ts,
                strength: 1.0 - valley_score, // deeper valley = stronger boundary
                reason: BoundaryReason::InterestValley,
            });
        }
    }

    boundaries
}

// ---------------------------------------------------------------------------
// Title generation
// ---------------------------------------------------------------------------

/// Generate a chapter title from the dominant characteristics of its scenes.
fn generate_chapter_title(chapter_index: usize, scenes: &[ScoredScene], prefix: &str) -> String {
    if scenes.is_empty() {
        let base = format!("Chapter {}", chapter_index + 1);
        return if prefix.is_empty() {
            base
        } else {
            format!("{prefix} {base}")
        };
    }

    // Count content types
    let mut type_counts = std::collections::HashMap::new();
    for s in scenes {
        *type_counts.entry(s.content_type).or_insert(0usize) += 1;
    }
    let dominant_type = type_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(ct, _)| ct)
        .unwrap_or(ContentType::Unknown);

    // Count sentiments
    let mut sent_counts = std::collections::HashMap::new();
    for s in scenes {
        *sent_counts
            .entry(format!("{:?}", s.sentiment))
            .or_insert(0usize) += 1;
    }
    let dominant_sentiment_label = sent_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(label, _)| label)
        .unwrap_or_else(|| "Neutral".to_string());

    let type_label = match dominant_type {
        ContentType::Action => "Action Sequence",
        ContentType::Dialogue => "Conversation",
        ContentType::CloseUp => "Close-up",
        ContentType::Group => "Group Scene",
        ContentType::Establishing => "Establishing",
        ContentType::Static => "Static Scene",
        ContentType::Transition => "Transition",
        ContentType::Unknown => "Scene",
    };

    let sentiment_suffix = match dominant_sentiment_label.as_str() {
        "Positive" => " (Uplifting)",
        "Negative" => " (Somber)",
        "Tense" => " (Intense)",
        "Calm" => " (Peaceful)",
        _ => "",
    };

    let base = format!("{type_label}{sentiment_suffix}");
    if prefix.is_empty() {
        base
    } else {
        format!("{prefix} {base}")
    }
}

/// Determine the dominant content type among a set of scenes.
fn dominant_content_type(scenes: &[ScoredScene]) -> ContentType {
    if scenes.is_empty() {
        return ContentType::Unknown;
    }
    let mut counts = std::collections::HashMap::new();
    for s in scenes {
        *counts.entry(s.content_type).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(ct, _)| ct)
        .unwrap_or(ContentType::Unknown)
}

/// Determine the dominant sentiment among a set of scenes.
fn dominant_sentiment(scenes: &[ScoredScene]) -> Sentiment {
    if scenes.is_empty() {
        return Sentiment::Neutral;
    }
    let mut counts = std::collections::HashMap::new();
    for s in scenes {
        let key = format!("{:?}", s.sentiment);
        *counts.entry(key).or_insert(0usize) += 1;
    }
    let label = counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(l, _)| l)
        .unwrap_or_else(|| "Neutral".to_string());

    match label.as_str() {
        "Positive" => Sentiment::Positive,
        "Negative" => Sentiment::Negative,
        "Tense" => Sentiment::Tense,
        "Calm" => Sentiment::Calm,
        _ => Sentiment::Neutral,
    }
}

// ---------------------------------------------------------------------------
// ChapterGenerator
// ---------------------------------------------------------------------------

/// Generates chapter points from scored scene data.
pub struct ChapterGenerator {
    config: ChapterConfig,
}

impl ChapterGenerator {
    /// Create a new chapter generator.
    #[must_use]
    pub fn new(config: ChapterConfig) -> Self {
        Self { config }
    }

    /// Generate chapters from scored scenes.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid or insufficient data is
    /// provided.
    pub fn generate(&self, scenes: &[ScoredScene]) -> AutoResult<Vec<Chapter>> {
        self.config.validate()?;

        if scenes.is_empty() {
            return Err(AutoError::insufficient_data(
                "No scenes provided for chapter generation",
            ));
        }

        // Collect candidate boundaries
        let mut candidates: Vec<BoundaryCandidate> = Vec::new();

        if self.config.use_content_type_transitions {
            candidates.extend(detect_content_type_boundaries(scenes));
        }

        candidates.extend(detect_importance_drop_boundaries(
            scenes,
            self.config.boundary_importance_drop,
        ));

        // Sort and deduplicate by scene_index (keep strongest)
        candidates.sort_by_key(|b| b.scene_index);
        candidates.dedup_by(|a, b| {
            if a.scene_index == b.scene_index {
                // keep the one with higher strength in b
                if a.strength > b.strength {
                    std::mem::swap(a, b);
                }
                true
            } else {
                false
            }
        });

        // Filter boundaries by min/max chapter duration
        let boundary_indices = self.filter_boundaries_by_duration(scenes, &candidates);

        // Build chapters from boundary indices
        let chapters = self.build_chapters(scenes, &boundary_indices);

        Ok(chapters)
    }

    /// Generate chapters using the interest curve in addition to scene data.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid or insufficient data.
    pub fn generate_with_curve(
        &self,
        scenes: &[ScoredScene],
        curve: &InterestCurve,
    ) -> AutoResult<Vec<Chapter>> {
        self.config.validate()?;

        if scenes.is_empty() {
            return Err(AutoError::insufficient_data(
                "No scenes provided for chapter generation",
            ));
        }

        let mut candidates: Vec<BoundaryCandidate> = Vec::new();

        if self.config.use_content_type_transitions {
            candidates.extend(detect_content_type_boundaries(scenes));
        }

        candidates.extend(detect_importance_drop_boundaries(
            scenes,
            self.config.boundary_importance_drop,
        ));

        if self.config.use_interest_valleys {
            candidates.extend(detect_interest_valley_boundaries(
                scenes,
                curve,
                self.config.valley_threshold,
            ));
        }

        // Sort and deduplicate
        candidates.sort_by_key(|b| b.scene_index);
        candidates.dedup_by(|a, b| {
            if a.scene_index == b.scene_index {
                if a.strength > b.strength {
                    std::mem::swap(a, b);
                }
                true
            } else {
                false
            }
        });

        let boundary_indices = self.filter_boundaries_by_duration(scenes, &candidates);
        let chapters = self.build_chapters(scenes, &boundary_indices);

        Ok(chapters)
    }

    /// Filter boundary candidates to respect duration constraints, returning
    /// the scene indices where chapter breaks occur.
    fn filter_boundaries_by_duration(
        &self,
        scenes: &[ScoredScene],
        candidates: &[BoundaryCandidate],
    ) -> Vec<usize> {
        if scenes.is_empty() {
            return Vec::new();
        }

        let total_start = scenes[0].start.pts;

        // Sort candidates by strength descending for greedy selection
        let mut sorted: Vec<&BoundaryCandidate> = candidates.iter().collect();
        sorted.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut accepted: Vec<usize> = Vec::new();

        for candidate in &sorted {
            let idx = candidate.scene_index;
            if idx == 0 || idx >= scenes.len() {
                continue;
            }

            let boundary_ts = scenes[idx].start.pts;

            // Check that adding this boundary doesn't create too-short chapters
            let mut test_boundaries: Vec<i64> =
                accepted.iter().map(|&i| scenes[i].start.pts).collect();
            test_boundaries.push(boundary_ts);
            test_boundaries.sort();

            let mut valid = true;
            let mut prev = total_start;
            for &bp in &test_boundaries {
                let gap = bp - prev;
                if gap < self.config.min_chapter_duration_ms {
                    valid = false;
                    break;
                }
                prev = bp;
            }

            // Also check last chapter
            if valid {
                let last_end = scenes.last().map(|s| s.end.pts).unwrap_or(prev);
                let final_gap = last_end - prev;
                if final_gap < self.config.min_chapter_duration_ms {
                    valid = false;
                }
            }

            if valid {
                accepted.push(idx);
            }

            // If we have a target count, stop when reached
            if self.config.target_chapter_count > 0
                && accepted.len() + 1 >= self.config.target_chapter_count
            {
                break;
            }
        }

        accepted.sort();

        // Force-split overly long chapters
        self.force_split_long_chapters(scenes, &mut accepted);

        accepted
    }

    /// Insert additional boundaries where chapters exceed the max duration.
    fn force_split_long_chapters(&self, scenes: &[ScoredScene], boundaries: &mut Vec<usize>) {
        if scenes.is_empty() {
            return;
        }

        let mut additional = Vec::new();

        // Build the chapter ranges
        let mut starts: Vec<usize> = vec![0];
        starts.extend(boundaries.iter().copied());

        for window in starts.windows(2) {
            let chapter_start_idx = window[0];
            let chapter_end_idx = window[1];
            let start_ts = scenes[chapter_start_idx].start.pts;
            let end_ts = scenes[chapter_end_idx].start.pts;
            let duration = end_ts - start_ts;

            if duration > self.config.max_chapter_duration_ms {
                // Split at the midpoint scene
                let mid_idx = (chapter_start_idx + chapter_end_idx) / 2;
                if mid_idx > chapter_start_idx && mid_idx < chapter_end_idx {
                    additional.push(mid_idx);
                }
            }
        }

        // Check the last chapter too
        if let Some(&last_boundary) = boundaries.last() {
            let last_end = scenes.last().map(|s| s.end.pts).unwrap_or(0);
            let start_ts = scenes[last_boundary].start.pts;
            if last_end - start_ts > self.config.max_chapter_duration_ms {
                let mid = (last_boundary + scenes.len()) / 2;
                if mid > last_boundary && mid < scenes.len() {
                    additional.push(mid);
                }
            }
        }

        boundaries.extend(additional);
        boundaries.sort();
        boundaries.dedup();
    }

    /// Build `Chapter` structs from scene data and boundary indices.
    fn build_chapters(&self, scenes: &[ScoredScene], boundary_indices: &[usize]) -> Vec<Chapter> {
        if scenes.is_empty() {
            return Vec::new();
        }

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut prev = 0;
        for &bi in boundary_indices {
            if bi > prev && bi < scenes.len() {
                ranges.push((prev, bi));
                prev = bi;
            }
        }
        ranges.push((prev, scenes.len()));

        let mut chapters = Vec::new();

        for (chapter_idx, (start_idx, end_idx)) in ranges.iter().enumerate() {
            let chapter_scenes = &scenes[*start_idx..*end_idx];
            if chapter_scenes.is_empty() {
                continue;
            }

            let start = chapter_scenes[0].start;
            let end = chapter_scenes.last().map(|s| s.end).unwrap_or(start);

            let avg_importance = if chapter_scenes.is_empty() {
                0.0
            } else {
                let sum: f64 = chapter_scenes.iter().map(|s| s.adjusted_score()).sum();
                sum / chapter_scenes.len() as f64
            };

            let title = if self.config.auto_title {
                generate_chapter_title(chapter_idx, chapter_scenes, &self.config.title_prefix)
            } else {
                format!("Chapter {}", chapter_idx + 1)
            };

            chapters.push(Chapter {
                index: chapter_idx,
                start,
                end,
                title,
                content_type: dominant_content_type(chapter_scenes),
                sentiment: dominant_sentiment(chapter_scenes),
                average_importance: avg_importance,
                scene_count: chapter_scenes.len(),
            });
        }

        chapters
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &ChapterConfig {
        &self.config
    }
}

impl Default for ChapterGenerator {
    fn default() -> Self {
        Self::new(ChapterConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Convenience: format chapters as YouTube-style description timestamps
// ---------------------------------------------------------------------------

/// Format a list of chapters as YouTube-style description timestamps.
pub fn format_youtube_chapters(chapters: &[Chapter]) -> String {
    chapters
        .iter()
        .map(|c| c.format_timestamp_line())
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::{ContentType, SceneFeatures, Sentiment};
    use oximedia_core::Rational;

    fn ts(ms: i64) -> Timestamp {
        Timestamp::new(ms, Rational::new(1, 1000))
    }

    fn make_scene(
        start_ms: i64,
        end_ms: i64,
        score: f64,
        ct: ContentType,
        sentiment: Sentiment,
    ) -> ScoredScene {
        let mut s = ScoredScene::new(ts(start_ms), ts(end_ms), score, ct, sentiment);
        s.features = SceneFeatures::default();
        s
    }

    fn default_scenes() -> Vec<ScoredScene> {
        vec![
            make_scene(0, 30_000, 0.7, ContentType::Establishing, Sentiment::Calm),
            make_scene(30_000, 60_000, 0.8, ContentType::Action, Sentiment::Tense),
            make_scene(
                60_000,
                90_000,
                0.5,
                ContentType::Dialogue,
                Sentiment::Neutral,
            ),
            make_scene(90_000, 120_000, 0.9, ContentType::Action, Sentiment::Tense),
            make_scene(120_000, 150_000, 0.3, ContentType::Static, Sentiment::Calm),
        ]
    }

    #[test]
    fn test_config_default_valid() {
        let cfg = ChapterConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_invalid_min_duration() {
        let cfg = ChapterConfig::default().with_min_duration_ms(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_max_less_than_min() {
        let cfg = ChapterConfig::default()
            .with_min_duration_ms(100_000)
            .with_max_duration_ms(50_000);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let cfg = ChapterConfig::default()
            .with_min_duration_ms(10_000)
            .with_max_duration_ms(300_000)
            .with_target_count(5)
            .with_valley_threshold(0.4);
        assert_eq!(cfg.min_chapter_duration_ms, 10_000);
        assert_eq!(cfg.max_chapter_duration_ms, 300_000);
        assert_eq!(cfg.target_chapter_count, 5);
        assert!((cfg.valley_threshold - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_generate_empty_scenes_error() {
        let gen = ChapterGenerator::default();
        let result = gen.generate(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_single_scene() {
        let scenes = vec![make_scene(
            0,
            60_000,
            0.7,
            ContentType::Action,
            Sentiment::Tense,
        )];
        let gen = ChapterGenerator::new(ChapterConfig::default().with_min_duration_ms(1000));
        let chapters = gen.generate(&scenes).expect("should succeed");
        assert_eq!(chapters.len(), 1);
        assert_eq!(chapters[0].index, 0);
        assert_eq!(chapters[0].scene_count, 1);
    }

    #[test]
    fn test_generate_detects_content_type_change() {
        let scenes = vec![
            make_scene(0, 40_000, 0.7, ContentType::Action, Sentiment::Tense),
            make_scene(
                40_000,
                80_000,
                0.6,
                ContentType::Dialogue,
                Sentiment::Neutral,
            ),
        ];
        let cfg = ChapterConfig {
            min_chapter_duration_ms: 10_000,
            max_chapter_duration_ms: 600_000,
            use_content_type_transitions: true,
            ..Default::default()
        };
        let gen = ChapterGenerator::new(cfg);
        let chapters = gen.generate(&scenes).expect("should succeed");
        assert_eq!(chapters.len(), 2, "should split at content type change");
        assert_eq!(chapters[0].content_type, ContentType::Action);
        assert_eq!(chapters[1].content_type, ContentType::Dialogue);
    }

    #[test]
    fn test_generate_detects_importance_drop() {
        let scenes = vec![
            make_scene(0, 40_000, 0.9, ContentType::Action, Sentiment::Tense),
            make_scene(40_000, 80_000, 0.3, ContentType::Action, Sentiment::Neutral),
        ];
        let cfg = ChapterConfig {
            min_chapter_duration_ms: 10_000,
            max_chapter_duration_ms: 600_000,
            boundary_importance_drop: 0.15,
            use_content_type_transitions: false,
            ..Default::default()
        };
        let gen = ChapterGenerator::new(cfg);
        let chapters = gen.generate(&scenes).expect("should succeed");
        assert_eq!(chapters.len(), 2, "should split at importance drop");
    }

    #[test]
    fn test_generate_respects_min_duration() {
        // Two very short scenes: should not split
        let scenes = vec![
            make_scene(0, 5_000, 0.9, ContentType::Action, Sentiment::Tense),
            make_scene(5_000, 10_000, 0.3, ContentType::Dialogue, Sentiment::Calm),
        ];
        let cfg = ChapterConfig {
            min_chapter_duration_ms: 30_000,
            max_chapter_duration_ms: 600_000,
            ..Default::default()
        };
        let gen = ChapterGenerator::new(cfg);
        let chapters = gen.generate(&scenes).expect("should succeed");
        assert_eq!(chapters.len(), 1, "should not split tiny chapters");
    }

    #[test]
    fn test_generate_with_default_scenes() {
        let scenes = default_scenes();
        let cfg = ChapterConfig {
            min_chapter_duration_ms: 20_000,
            max_chapter_duration_ms: 600_000,
            ..Default::default()
        };
        let gen = ChapterGenerator::new(cfg);
        let chapters = gen.generate(&scenes).expect("should succeed");
        assert!(!chapters.is_empty());
        // Chapters should cover the full duration
        assert_eq!(chapters[0].start.pts, 0);
        let last = chapters.last().expect("non-empty");
        assert_eq!(last.end.pts, 150_000);
    }

    #[test]
    fn test_chapter_duration_ms() {
        let ch = Chapter {
            index: 0,
            start: ts(10_000),
            end: ts(70_000),
            title: "Test".into(),
            content_type: ContentType::Action,
            sentiment: Sentiment::Tense,
            average_importance: 0.8,
            scene_count: 3,
        };
        assert_eq!(ch.duration_ms(), 60_000);
    }

    #[test]
    fn test_chapter_contains() {
        let ch = Chapter {
            index: 0,
            start: ts(10_000),
            end: ts(70_000),
            title: "Test".into(),
            content_type: ContentType::Action,
            sentiment: Sentiment::Neutral,
            average_importance: 0.5,
            scene_count: 1,
        };
        assert!(ch.contains(ts(10_000)));
        assert!(ch.contains(ts(50_000)));
        assert!(!ch.contains(ts(70_000))); // exclusive end
        assert!(!ch.contains(ts(5_000)));
    }

    #[test]
    fn test_format_timestamp_line() {
        let ch = Chapter {
            index: 0,
            start: ts(3_661_000), // 1h 1m 1s
            end: ts(7_200_000),
            title: "Grand Finale".into(),
            content_type: ContentType::Action,
            sentiment: Sentiment::Tense,
            average_importance: 0.9,
            scene_count: 5,
        };
        let line = ch.format_timestamp_line();
        assert_eq!(line, "01:01:01 - Grand Finale");
    }

    #[test]
    fn test_format_youtube_chapters() {
        let chapters = vec![
            Chapter {
                index: 0,
                start: ts(0),
                end: ts(60_000),
                title: "Intro".into(),
                content_type: ContentType::Establishing,
                sentiment: Sentiment::Calm,
                average_importance: 0.5,
                scene_count: 1,
            },
            Chapter {
                index: 1,
                start: ts(60_000),
                end: ts(120_000),
                title: "Main Event".into(),
                content_type: ContentType::Action,
                sentiment: Sentiment::Tense,
                average_importance: 0.9,
                scene_count: 2,
            },
        ];
        let text = format_youtube_chapters(&chapters);
        assert!(text.contains("00:00:00 - Intro"));
        assert!(text.contains("00:01:00 - Main Event"));
    }

    #[test]
    fn test_generate_chapter_title_action_tense() {
        let scenes = vec![make_scene(
            0,
            30_000,
            0.9,
            ContentType::Action,
            Sentiment::Tense,
        )];
        let title = generate_chapter_title(0, &scenes, "");
        assert!(title.contains("Action"));
        assert!(title.contains("Intense"));
    }

    #[test]
    fn test_generate_chapter_title_with_prefix() {
        let scenes = vec![make_scene(
            0,
            30_000,
            0.5,
            ContentType::Dialogue,
            Sentiment::Neutral,
        )];
        let title = generate_chapter_title(0, &scenes, "Part");
        assert!(title.starts_with("Part "));
    }

    #[test]
    fn test_generate_chapter_title_empty_scenes() {
        let title = generate_chapter_title(2, &[], "");
        assert_eq!(title, "Chapter 3");
    }

    #[test]
    fn test_dominant_content_type() {
        let scenes = vec![
            make_scene(0, 30_000, 0.7, ContentType::Action, Sentiment::Tense),
            make_scene(30_000, 60_000, 0.6, ContentType::Action, Sentiment::Neutral),
            make_scene(60_000, 90_000, 0.5, ContentType::Dialogue, Sentiment::Calm),
        ];
        let ct = dominant_content_type(&scenes);
        assert_eq!(ct, ContentType::Action);
    }

    #[test]
    fn test_dominant_sentiment() {
        let scenes = vec![
            make_scene(0, 30_000, 0.7, ContentType::Action, Sentiment::Tense),
            make_scene(30_000, 60_000, 0.6, ContentType::Action, Sentiment::Tense),
            make_scene(60_000, 90_000, 0.5, ContentType::Dialogue, Sentiment::Calm),
        ];
        let s = dominant_sentiment(&scenes);
        assert_eq!(s, Sentiment::Tense);
    }

    #[test]
    fn test_generate_with_curve() {
        let scenes = default_scenes();
        let mut curve = InterestCurve::new(3);
        for s in &scenes {
            curve.add_point(s.start, s.adjusted_score());
            curve.add_point(s.end, s.adjusted_score());
        }
        let cfg = ChapterConfig {
            min_chapter_duration_ms: 20_000,
            max_chapter_duration_ms: 600_000,
            use_interest_valleys: true,
            valley_threshold: 0.5,
            ..Default::default()
        };
        let gen = ChapterGenerator::new(cfg);
        let chapters = gen
            .generate_with_curve(&scenes, &curve)
            .expect("should succeed");
        assert!(!chapters.is_empty());
    }

    #[test]
    fn test_target_chapter_count() {
        let scenes: Vec<ScoredScene> = (0..10)
            .map(|i| {
                let ct = if i % 2 == 0 {
                    ContentType::Action
                } else {
                    ContentType::Dialogue
                };
                make_scene(
                    i as i64 * 30_000,
                    (i + 1) as i64 * 30_000,
                    0.5 + (i as f64 * 0.05),
                    ct,
                    Sentiment::Neutral,
                )
            })
            .collect();
        let cfg = ChapterConfig {
            min_chapter_duration_ms: 10_000,
            max_chapter_duration_ms: 600_000,
            target_chapter_count: 3,
            ..Default::default()
        };
        let gen = ChapterGenerator::new(cfg);
        let chapters = gen.generate(&scenes).expect("should succeed");
        assert!(
            chapters.len() <= 3,
            "should respect target count: got {}",
            chapters.len()
        );
    }

    #[test]
    fn test_chapters_cover_full_duration() {
        let scenes = default_scenes();
        let cfg = ChapterConfig {
            min_chapter_duration_ms: 10_000,
            max_chapter_duration_ms: 600_000,
            ..Default::default()
        };
        let gen = ChapterGenerator::new(cfg);
        let chapters = gen.generate(&scenes).expect("should succeed");
        // First chapter starts at video start
        assert_eq!(chapters[0].start.pts, scenes[0].start.pts);
        // Last chapter ends at video end
        let last = chapters.last().expect("non-empty");
        let video_end = scenes.last().expect("non-empty").end.pts;
        assert_eq!(last.end.pts, video_end);
    }

    #[test]
    fn test_chapter_scene_counts_sum() {
        let scenes = default_scenes();
        let cfg = ChapterConfig {
            min_chapter_duration_ms: 10_000,
            max_chapter_duration_ms: 600_000,
            ..Default::default()
        };
        let gen = ChapterGenerator::new(cfg);
        let chapters = gen.generate(&scenes).expect("should succeed");
        let total: usize = chapters.iter().map(|c| c.scene_count).sum();
        assert_eq!(total, scenes.len());
    }
}

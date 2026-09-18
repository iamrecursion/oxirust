//! Automatic B-roll insertion suggestion for video editing.
//!
//! Analyses the A-roll (primary footage) and dialogue transcript to suggest
//! where B-roll (supplementary footage) should be inserted, and what type of
//! B-roll would best support each moment.
//!
//! The module provides:
//!
//! - **Dialogue gap detection**: Finds pauses in dialogue where B-roll covers
//!   otherwise static footage.
//! - **Visual monotony detection**: Identifies sequences of low-variety scenes
//!   that would benefit from cutaways.
//! - **Keyword-driven suggestion**: Matches dialogue keywords to suggested
//!   B-roll categories (e.g. "city" → exterior urban shots).
//! - **Pacing-aware insertion**: Ensures B-roll segments respect shot duration
//!   constraints.
//! - **Priority scoring**: Ranks suggestions by expected editorial benefit.
//!
//! # Example
//!
//! ```
//! use oximedia_auto::a_b_roll::{BRollSuggester, BRollConfig};
//!
//! let config = BRollConfig::default();
//! let suggester = BRollSuggester::new(config);
//! ```

#![allow(dead_code)]

use crate::cuts::DialogueSegment;
use crate::error::{AutoError, AutoResult};
use crate::scoring::ScoredScene;
use oximedia_core::Timestamp;
use std::collections::HashMap;

/// A timestamped text segment with optional content (used for keyword-based B-roll matching).
///
/// Unlike [`DialogueSegment`], which contains only timing metadata, a
/// `TextSegment` carries the raw transcript text for keyword extraction.
/// These can be created from subtitle exports or ASR results.
#[derive(Debug, Clone)]
pub struct TextSegment {
    /// Start timestamp.
    pub start: Timestamp,
    /// End timestamp.
    pub end: Timestamp,
    /// Transcript or subtitle text for this segment.
    pub text: String,
}

impl TextSegment {
    /// Create a new text segment.
    #[must_use]
    pub fn new(start: Timestamp, end: Timestamp, text: impl Into<String>) -> Self {
        Self {
            start,
            end,
            text: text.into(),
        }
    }
}

// ─── B-roll category ─────────────────────────────────────────────────────────

/// Thematic category of suggested B-roll footage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BRollCategory {
    /// Exterior urban environment (streets, buildings).
    UrbanExterior,
    /// Interior room or office setting.
    Interior,
    /// Nature or outdoor landscape.
    Nature,
    /// Close-up detail shots (hands, objects).
    DetailCloseUp,
    /// Abstract or graphical imagery.
    Abstract,
    /// Archival or historical material.
    Archival,
    /// Reaction or emotion close-up.
    ReactionShot,
    /// Aerial or wide establishing shot.
    AerialEstablishing,
    /// Product or object demonstration.
    ProductDemo,
    /// Data / chart / infographic overlay.
    DataVisualization,
    /// Generic cutaway (category inferred from context).
    Generic,
    /// Custom category with a user-supplied tag.
    Custom(String),
}

impl BRollCategory {
    /// Human-readable label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::UrbanExterior => "Urban Exterior",
            Self::Interior => "Interior",
            Self::Nature => "Nature",
            Self::DetailCloseUp => "Detail Close-Up",
            Self::Abstract => "Abstract",
            Self::Archival => "Archival",
            Self::ReactionShot => "Reaction Shot",
            Self::AerialEstablishing => "Aerial/Establishing",
            Self::ProductDemo => "Product Demo",
            Self::DataVisualization => "Data Visualization",
            Self::Generic => "Generic Cutaway",
            Self::Custom(tag) => tag.as_str(),
        }
    }

    /// Map a keyword to the most relevant category.
    #[must_use]
    pub fn from_keyword(keyword: &str) -> Self {
        let kw = keyword.to_lowercase();
        if ["city", "street", "building", "downtown", "urban", "traffic"]
            .iter()
            .any(|&k| kw.contains(k))
        {
            Self::UrbanExterior
        } else if [
            "nature", "forest", "mountain", "ocean", "river", "tree", "field",
        ]
        .iter()
        .any(|&k| kw.contains(k))
        {
            Self::Nature
        } else if ["office", "room", "indoors", "studio", "interior"]
            .iter()
            .any(|&k| kw.contains(k))
        {
            Self::Interior
        } else if ["data", "chart", "graph", "statistic", "figure", "number"]
            .iter()
            .any(|&k| kw.contains(k))
        {
            Self::DataVisualization
        } else if ["product", "device", "demo", "demonstrate", "show"]
            .iter()
            .any(|&k| kw.contains(k))
        {
            Self::ProductDemo
        } else if ["archive", "historical", "old", "vintage", "past"]
            .iter()
            .any(|&k| kw.contains(k))
        {
            Self::Archival
        } else if ["aerial", "drone", "skyline", "overview", "establishing"]
            .iter()
            .any(|&k| kw.contains(k))
        {
            Self::AerialEstablishing
        } else if ["close", "detail", "hands", "texture", "object"]
            .iter()
            .any(|&k| kw.contains(k))
        {
            Self::DetailCloseUp
        } else {
            Self::Generic
        }
    }
}

// ─── B-roll suggestion ────────────────────────────────────────────────────────

/// A single B-roll insertion suggestion.
#[derive(Debug, Clone)]
pub struct BRollSuggestion {
    /// Where to start the B-roll in the A-roll timeline.
    pub start: Timestamp,
    /// Where to end the B-roll in the A-roll timeline.
    pub end: Timestamp,
    /// Suggested B-roll category.
    pub category: BRollCategory,
    /// Priority score (0.0 – 1.0; higher = more beneficial).
    pub priority: f64,
    /// Reason for the suggestion.
    pub reason: String,
    /// Optional keywords extracted from dialogue at this position.
    pub dialogue_keywords: Vec<String>,
    /// Whether this is a mandatory insert (e.g. A-roll has no coverage here).
    pub is_mandatory: bool,
    /// Suggested minimum duration for the B-roll (ms).
    pub suggested_min_duration_ms: i64,
}

impl BRollSuggestion {
    /// Duration of the suggested B-roll slot in milliseconds.
    #[must_use]
    pub fn slot_duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }

    /// Check if this suggestion overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start.pts < other.end.pts && self.end.pts > other.start.pts
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Keyword → B-roll category mapping entry.
#[derive(Debug, Clone)]
pub struct KeywordMapping {
    /// Trigger keyword (case-insensitive substring match).
    pub keyword: String,
    /// B-roll category to suggest when the keyword is detected.
    pub category: BRollCategory,
    /// Priority boost for this mapping (added to the base priority).
    pub priority_boost: f64,
}

impl KeywordMapping {
    /// Create a new keyword mapping.
    #[must_use]
    pub fn new(keyword: impl Into<String>, category: BRollCategory, priority_boost: f64) -> Self {
        Self {
            keyword: keyword.into(),
            category,
            priority_boost: priority_boost.clamp(0.0, 1.0),
        }
    }
}

/// Configuration for the B-roll suggester.
#[derive(Debug, Clone)]
pub struct BRollConfig {
    /// Minimum dialogue gap duration to insert B-roll (ms).
    pub min_dialogue_gap_ms: i64,
    /// Minimum number of consecutive low-variety scenes for a monotony suggestion.
    pub monotony_scene_count: usize,
    /// Scene feature variance threshold below which a scene is "low variety".
    pub low_variety_threshold: f64,
    /// Minimum slot duration for a B-roll suggestion (ms).
    pub min_slot_duration_ms: i64,
    /// Maximum slot duration for a single B-roll suggestion (ms).
    pub max_slot_duration_ms: i64,
    /// Custom keyword mappings (in addition to built-in heuristics).
    pub keyword_mappings: Vec<KeywordMapping>,
    /// Maximum number of suggestions to return (0 = unlimited).
    pub max_suggestions: usize,
    /// Minimum priority threshold to include a suggestion.
    pub min_priority: f64,
}

impl Default for BRollConfig {
    fn default() -> Self {
        Self {
            min_dialogue_gap_ms: 2000,
            monotony_scene_count: 3,
            low_variety_threshold: 0.25,
            min_slot_duration_ms: 1000,
            max_slot_duration_ms: 8000,
            keyword_mappings: Vec::new(),
            max_suggestions: 0,
            min_priority: 0.20,
        }
    }
}

impl BRollConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if self.min_dialogue_gap_ms < 0 {
            return Err(AutoError::InvalidParameter {
                name: "min_dialogue_gap_ms".into(),
                value: "must be non-negative".into(),
            });
        }
        if self.monotony_scene_count == 0 {
            return Err(AutoError::InvalidParameter {
                name: "monotony_scene_count".into(),
                value: "must be at least 1".into(),
            });
        }
        if self.min_slot_duration_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.min_slot_duration_ms,
            });
        }
        if self.max_slot_duration_ms <= self.min_slot_duration_ms {
            return Err(AutoError::InvalidParameter {
                name: "max_slot_duration_ms".into(),
                value: "must be greater than min_slot_duration_ms".into(),
            });
        }
        if !(0.0..=1.0).contains(&self.min_priority) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.min_priority,
                min: 0.0,
                max: 1.0,
            });
        }
        Ok(())
    }

    /// Add a custom keyword mapping.
    #[must_use]
    pub fn with_keyword(mut self, mapping: KeywordMapping) -> Self {
        self.keyword_mappings.push(mapping);
        self
    }
}

// ─── Suggester ────────────────────────────────────────────────────────────────

/// B-roll insertion suggester.
pub struct BRollSuggester {
    config: BRollConfig,
}

impl BRollSuggester {
    /// Create a new suggester with the given config.
    #[must_use]
    pub fn new(config: BRollConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn default_suggester() -> Self {
        Self::new(BRollConfig::default())
    }

    /// Generate B-roll insertion suggestions.
    ///
    /// Analyses both scored scenes and dialogue timing to produce a ranked
    /// list of insertion points. For keyword-driven suggestions, use
    /// `suggest_with_text` which accepts [`TextSegment`] slices carrying
    /// transcript content.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn suggest(
        &self,
        scenes: &[ScoredScene],
        dialogue: &[DialogueSegment],
    ) -> AutoResult<Vec<BRollSuggestion>> {
        self.config.validate()?;

        let mut suggestions = Vec::new();

        // 1. Dialogue gap-based suggestions
        suggestions.extend(self.suggestions_from_dialogue_gaps(dialogue));

        // 2. Visual monotony-based suggestions
        suggestions.extend(self.suggestions_from_monotony(scenes));

        // 3. Low-importance scene coverage
        suggestions.extend(self.suggestions_from_low_importance(scenes));

        // Deduplicate and rank
        let mut unique = self.deduplicate(suggestions);
        unique.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply min priority filter
        unique.retain(|s| s.priority >= self.config.min_priority);

        // Apply max suggestions limit
        if self.config.max_suggestions > 0 {
            unique.truncate(self.config.max_suggestions);
        }

        Ok(unique)
    }

    /// Generate B-roll suggestions with full text transcript for keyword extraction.
    ///
    /// In addition to the gap/monotony/importance analysis performed by
    /// `suggest`, this method scans `text_segments` for subject keywords
    /// and maps them to appropriate B-roll categories.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn suggest_with_text(
        &self,
        scenes: &[ScoredScene],
        dialogue: &[DialogueSegment],
        text_segments: &[TextSegment],
    ) -> AutoResult<Vec<BRollSuggestion>> {
        self.config.validate()?;

        let mut suggestions = Vec::new();

        // 1. Dialogue gap-based suggestions
        suggestions.extend(self.suggestions_from_dialogue_gaps(dialogue));

        // 2. Visual monotony-based suggestions
        suggestions.extend(self.suggestions_from_monotony(scenes));

        // 3. Keyword-driven suggestions from transcript text
        suggestions.extend(self.suggestions_from_keywords(scenes, text_segments));

        // 4. Low-importance scene coverage
        suggestions.extend(self.suggestions_from_low_importance(scenes));

        // Deduplicate and rank
        let mut unique = self.deduplicate(suggestions);
        unique.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        unique.retain(|s| s.priority >= self.config.min_priority);

        if self.config.max_suggestions > 0 {
            unique.truncate(self.config.max_suggestions);
        }

        Ok(unique)
    }

    // ─── Private suggestion generators ───────────────────────────────────────

    /// Find gaps between dialogue segments where B-roll can cover.
    fn suggestions_from_dialogue_gaps(&self, dialogue: &[DialogueSegment]) -> Vec<BRollSuggestion> {
        if dialogue.len() < 2 {
            return Vec::new();
        }

        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut suggestions = Vec::new();

        for window in dialogue.windows(2) {
            let prev = &window[0];
            let next = &window[1];
            let gap_ms = next.start.pts - prev.end.pts;

            if gap_ms < self.config.min_dialogue_gap_ms {
                continue;
            }

            let slot_ms = gap_ms.min(self.config.max_slot_duration_ms);
            if slot_ms < self.config.min_slot_duration_ms {
                continue;
            }

            let priority = (gap_ms as f64 / 10_000.0).clamp(0.20, 0.90);

            suggestions.push(BRollSuggestion {
                start: Timestamp::new(prev.end.pts, timebase),
                end: Timestamp::new(prev.end.pts + slot_ms, timebase),
                category: BRollCategory::Generic,
                priority,
                reason: format!("Dialogue gap of {gap_ms}ms"),
                dialogue_keywords: Vec::new(),
                is_mandatory: gap_ms > 4000,
                suggested_min_duration_ms: self.config.min_slot_duration_ms,
            });
        }

        suggestions
    }

    /// Detect runs of low-variety scenes and suggest B-roll.
    fn suggestions_from_monotony(&self, scenes: &[ScoredScene]) -> Vec<BRollSuggestion> {
        if scenes.len() < self.config.monotony_scene_count {
            return Vec::new();
        }

        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut suggestions = Vec::new();
        let mut run_start: Option<usize> = None;

        for (i, scene) in scenes.iter().enumerate() {
            let variety = self.scene_variety_score(scene);
            if variety < self.config.low_variety_threshold {
                if run_start.is_none() {
                    run_start = Some(i);
                }
            } else if let Some(start_idx) = run_start {
                let run_len = i - start_idx;
                if run_len >= self.config.monotony_scene_count {
                    let first = &scenes[start_idx];
                    let last = &scenes[i - 1];
                    let slot_ms =
                        (last.end.pts - first.start.pts).min(self.config.max_slot_duration_ms);

                    if slot_ms >= self.config.min_slot_duration_ms {
                        let severity = (run_len as f64 / 10.0).clamp(0.0, 1.0);
                        suggestions.push(BRollSuggestion {
                            start: Timestamp::new(first.start.pts, timebase),
                            end: Timestamp::new(first.start.pts + slot_ms, timebase),
                            category: BRollCategory::Generic,
                            priority: (0.30 + severity * 0.40).clamp(0.0, 1.0),
                            reason: format!(
                                "{run_len} consecutive low-variety scenes (variety < {:.2})",
                                self.config.low_variety_threshold
                            ),
                            dialogue_keywords: Vec::new(),
                            is_mandatory: false,
                            suggested_min_duration_ms: self.config.min_slot_duration_ms,
                        });
                    }
                }
                run_start = None;
            }
        }

        // Handle a run that ends at the last scene
        if let Some(start_idx) = run_start {
            let run_len = scenes.len() - start_idx;
            if run_len >= self.config.monotony_scene_count {
                let first = &scenes[start_idx];
                let last = &scenes[scenes.len() - 1];
                let slot_ms =
                    (last.end.pts - first.start.pts).min(self.config.max_slot_duration_ms);
                if slot_ms >= self.config.min_slot_duration_ms {
                    suggestions.push(BRollSuggestion {
                        start: Timestamp::new(first.start.pts, timebase),
                        end: Timestamp::new(first.start.pts + slot_ms, timebase),
                        category: BRollCategory::Generic,
                        priority: (0.30 + (run_len as f64 / 10.0) * 0.40).clamp(0.0, 1.0),
                        reason: format!("{run_len} low-variety scenes at tail"),
                        dialogue_keywords: Vec::new(),
                        is_mandatory: false,
                        suggested_min_duration_ms: self.config.min_slot_duration_ms,
                    });
                }
            }
        }

        suggestions
    }

    /// Extract keywords from text segments and map to B-roll categories.
    ///
    /// Uses [`TextSegment`] which carries actual transcript text, unlike
    /// [`DialogueSegment`] which only has timing metadata.
    fn suggestions_from_keywords(
        &self,
        scenes: &[ScoredScene],
        text_segments: &[TextSegment],
    ) -> Vec<BRollSuggestion> {
        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut suggestions = Vec::new();

        // Built-in keyword heuristics
        let builtins: &[(&str, BRollCategory, f64)] = &[
            ("city", BRollCategory::UrbanExterior, 0.10),
            ("street", BRollCategory::UrbanExterior, 0.10),
            ("building", BRollCategory::UrbanExterior, 0.10),
            ("traffic", BRollCategory::UrbanExterior, 0.05),
            ("forest", BRollCategory::Nature, 0.10),
            ("mountain", BRollCategory::Nature, 0.10),
            ("ocean", BRollCategory::Nature, 0.10),
            ("nature", BRollCategory::Nature, 0.10),
            ("chart", BRollCategory::DataVisualization, 0.15),
            ("graph", BRollCategory::DataVisualization, 0.15),
            ("data", BRollCategory::DataVisualization, 0.10),
            ("statistic", BRollCategory::DataVisualization, 0.10),
            ("product", BRollCategory::ProductDemo, 0.15),
            ("device", BRollCategory::ProductDemo, 0.10),
            ("archive", BRollCategory::Archival, 0.10),
            ("historical", BRollCategory::Archival, 0.10),
            ("aerial", BRollCategory::AerialEstablishing, 0.15),
            ("skyline", BRollCategory::AerialEstablishing, 0.10),
        ];

        for seg in text_segments {
            let text_lower = seg.text.to_lowercase();

            // Find matching built-in keywords
            let mut matched: HashMap<String, (BRollCategory, f64)> = HashMap::new();
            for &(kw, ref cat, boost) in builtins {
                if text_lower.contains(kw) {
                    let entry = matched
                        .entry(kw.to_string())
                        .or_insert_with(|| (cat.clone(), boost));
                    if boost > entry.1 {
                        *entry = (cat.clone(), boost);
                    }
                }
            }
            // Also check custom mappings
            for mapping in &self.config.keyword_mappings {
                let kw_lower = mapping.keyword.to_lowercase();
                if text_lower.contains(&kw_lower) {
                    let entry = matched
                        .entry(mapping.keyword.clone())
                        .or_insert_with(|| (mapping.category.clone(), mapping.priority_boost));
                    if mapping.priority_boost > entry.1 {
                        *entry = (mapping.category.clone(), mapping.priority_boost);
                    }
                }
            }

            if matched.is_empty() {
                continue;
            }

            // Find the scene that overlaps this text segment
            let scene_opt = scenes
                .iter()
                .find(|s| s.start.pts <= seg.start.pts && s.end.pts >= seg.start.pts);

            let (start_pts, end_pts) = if let Some(scene) = scene_opt {
                let slot = (scene.end.pts - seg.start.pts).min(self.config.max_slot_duration_ms);
                (
                    seg.start.pts,
                    seg.start.pts + slot.max(self.config.min_slot_duration_ms),
                )
            } else {
                let slot_end = seg
                    .end
                    .pts
                    .min(seg.start.pts + self.config.max_slot_duration_ms);
                (seg.start.pts, slot_end)
            };

            if end_pts - start_pts < self.config.min_slot_duration_ms {
                continue;
            }

            for (kw, (cat, boost)) in &matched {
                let priority = (0.40 + boost).clamp(0.0, 1.0);
                suggestions.push(BRollSuggestion {
                    start: Timestamp::new(start_pts, timebase),
                    end: Timestamp::new(end_pts, timebase),
                    category: cat.clone(),
                    priority,
                    reason: format!("Transcript keyword \"{kw}\" suggests {}", cat.as_str()),
                    dialogue_keywords: matched.keys().cloned().collect(),
                    is_mandatory: false,
                    suggested_min_duration_ms: self.config.min_slot_duration_ms,
                });
            }
        }

        suggestions
    }

    /// Suggest B-roll for low-importance scenes to improve engagement.
    fn suggestions_from_low_importance(&self, scenes: &[ScoredScene]) -> Vec<BRollSuggestion> {
        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut suggestions = Vec::new();

        for scene in scenes {
            if scene.adjusted_score() >= 0.40 {
                continue;
            }

            let slot_ms = scene.duration().min(self.config.max_slot_duration_ms);
            if slot_ms < self.config.min_slot_duration_ms {
                continue;
            }

            let priority = (0.30 + (0.40 - scene.adjusted_score()) * 0.50).clamp(0.0, 1.0);

            suggestions.push(BRollSuggestion {
                start: Timestamp::new(scene.start.pts, timebase),
                end: Timestamp::new(scene.start.pts + slot_ms, timebase),
                category: BRollCategory::Generic,
                priority,
                reason: format!(
                    "Low-importance scene (score {:.2}); B-roll improves engagement",
                    scene.adjusted_score()
                ),
                dialogue_keywords: Vec::new(),
                is_mandatory: false,
                suggested_min_duration_ms: self.config.min_slot_duration_ms,
            });
        }

        suggestions
    }

    /// Compute a variety score for a scene based on its feature diversity.
    fn scene_variety_score(&self, scene: &ScoredScene) -> f64 {
        let f = &scene.features;
        // Variety = combination of motion, colour diversity, edge density, sharpness
        (f.motion_intensity * 0.30
            + f.color_diversity * 0.30
            + f.edge_density * 0.20
            + f.sharpness * 0.20)
            .clamp(0.0, 1.0)
    }

    /// Remove overlapping suggestions, keeping the one with the highest priority.
    fn deduplicate(&self, mut suggestions: Vec<BRollSuggestion>) -> Vec<BRollSuggestion> {
        if suggestions.is_empty() {
            return suggestions;
        }

        // Sort by start time
        suggestions.sort_by_key(|s| s.start.pts);

        let mut result: Vec<BRollSuggestion> = Vec::new();
        for sug in suggestions {
            if let Some(last) = result.last_mut() {
                if sug.overlaps(last) {
                    if sug.priority > last.priority {
                        *last = sug;
                    }
                    continue;
                }
            }
            result.push(sug);
        }
        result
    }
}

impl Default for BRollSuggester {
    fn default() -> Self {
        Self::default_suggester()
    }
}

// ─── Public helpers ───────────────────────────────────────────────────────────

/// Format suggestions as a simple editorial report string.
#[must_use]
pub fn format_suggestions(suggestions: &[BRollSuggestion]) -> String {
    if suggestions.is_empty() {
        return "No B-roll suggestions generated.".to_string();
    }

    let lines: Vec<String> = suggestions
        .iter()
        .enumerate()
        .map(|(i, s)| {
            format!(
                "{}. [{}-{}ms] {} (priority {:.2}): {}",
                i + 1,
                s.start.pts,
                s.end.pts,
                s.category.as_str(),
                s.priority,
                s.reason,
            )
        })
        .collect();
    lines.join("\n")
}

/// Filter suggestions by category.
#[must_use]
pub fn filter_by_category<'a>(
    suggestions: &'a [BRollSuggestion],
    category: &BRollCategory,
) -> Vec<&'a BRollSuggestion> {
    suggestions
        .iter()
        .filter(|s| &s.category == category)
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuts::DialogueSegment;
    use crate::scoring::{ContentType, SceneFeatures, ScoredScene, Sentiment};
    use oximedia_core::{Rational, Timestamp};

    fn ts(ms: i64) -> Timestamp {
        Timestamp::new(ms, Rational::new(1, 1000))
    }

    fn make_scene(start_ms: i64, end_ms: i64, score: f64) -> ScoredScene {
        ScoredScene::new(
            ts(start_ms),
            ts(end_ms),
            score,
            ContentType::Dialogue,
            Sentiment::Neutral,
        )
    }

    fn make_scene_with_variety(
        start_ms: i64,
        end_ms: i64,
        score: f64,
        variety: f64,
    ) -> ScoredScene {
        let mut s = make_scene(start_ms, end_ms, score);
        s.features = SceneFeatures {
            motion_intensity: variety * 0.5,
            color_diversity: variety * 0.5,
            edge_density: variety,
            sharpness: variety,
            ..SceneFeatures::default()
        };
        s
    }

    fn make_dialogue(start_ms: i64, end_ms: i64) -> DialogueSegment {
        DialogueSegment::new(ts(start_ms), ts(end_ms), 0.9)
    }

    fn make_text_segment(start_ms: i64, end_ms: i64, text: &str) -> TextSegment {
        TextSegment::new(ts(start_ms), ts(end_ms), text)
    }

    #[test]
    fn test_default_config_valid() {
        assert!(BRollConfig::default().validate().is_ok());
    }

    #[test]
    fn test_invalid_min_slot_duration() {
        let mut cfg = BRollConfig::default();
        cfg.min_slot_duration_ms = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_max_less_than_min_slot() {
        let mut cfg = BRollConfig::default();
        cfg.max_slot_duration_ms = 500;
        cfg.min_slot_duration_ms = 1000;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_empty_inputs_returns_empty() {
        let suggester = BRollSuggester::default();
        let suggestions = suggester.suggest(&[], &[]).expect("suggest should succeed");
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_dialogue_gap_creates_suggestion() {
        let suggester = BRollSuggester::new(BRollConfig {
            min_dialogue_gap_ms: 1000,
            min_priority: 0.0,
            ..BRollConfig::default()
        });

        let dialogue = vec![
            make_dialogue(0, 3000),
            make_dialogue(8000, 12_000), // 5000ms gap → should trigger
        ];

        let suggestions = suggester
            .suggest(&[], &dialogue)
            .expect("suggest should succeed");
        assert!(!suggestions.is_empty(), "Expected dialogue-gap suggestion");
        assert!(suggestions[0].start.pts == 3000);
    }

    #[test]
    fn test_small_dialogue_gap_ignored() {
        let suggester = BRollSuggester::new(BRollConfig {
            min_dialogue_gap_ms: 5000,
            min_priority: 0.0,
            ..BRollConfig::default()
        });

        let dialogue = vec![
            make_dialogue(0, 3000),
            make_dialogue(4000, 8000), // only 1000ms gap → below threshold
        ];

        let gap_suggestions: Vec<_> = suggester
            .suggest(&[], &dialogue)
            .expect("value should be present should succeed")
            .into_iter()
            .filter(|s| s.reason.contains("gap"))
            .collect();
        assert!(
            gap_suggestions.is_empty(),
            "Small gap should not trigger suggestion"
        );
    }

    #[test]
    fn test_monotony_detection() {
        let suggester = BRollSuggester::new(BRollConfig {
            monotony_scene_count: 3,
            low_variety_threshold: 0.50,
            min_priority: 0.0,
            ..BRollConfig::default()
        });

        // 5 consecutive low-variety scenes
        let scenes: Vec<ScoredScene> = (0..5)
            .map(|i| make_scene_with_variety(i * 3000, (i + 1) * 3000, 0.50, 0.10))
            .collect();

        let suggestions = suggester
            .suggest(&scenes, &[])
            .expect("suggest should succeed");
        assert!(
            !suggestions.is_empty(),
            "Expected monotony-based B-roll suggestion"
        );
    }

    #[test]
    fn test_keyword_city_maps_to_urban_exterior() {
        let cat = BRollCategory::from_keyword("city streets");
        assert_eq!(cat, BRollCategory::UrbanExterior);
    }

    #[test]
    fn test_keyword_forest_maps_to_nature() {
        let cat = BRollCategory::from_keyword("deep forest");
        assert_eq!(cat, BRollCategory::Nature);
    }

    #[test]
    fn test_keyword_data_maps_to_data_viz() {
        let cat = BRollCategory::from_keyword("the data shows");
        assert_eq!(cat, BRollCategory::DataVisualization);
    }

    #[test]
    fn test_keyword_suggestion_from_text_segment() {
        let suggester = BRollSuggester::new(BRollConfig {
            min_priority: 0.0,
            ..BRollConfig::default()
        });

        let scenes = vec![make_scene(0, 10_000, 0.50)];
        let text = vec![make_text_segment(
            0,
            5000,
            "We walked through the city streets",
        )];

        let suggestions = suggester
            .suggest_with_text(&scenes, &[], &text)
            .expect("suggest with text should succeed");
        let urban: Vec<_> = suggestions
            .iter()
            .filter(|s| s.category == BRollCategory::UrbanExterior)
            .collect();
        assert!(
            !urban.is_empty(),
            "Expected UrbanExterior suggestion from transcript keyword"
        );
    }

    #[test]
    fn test_low_importance_generates_suggestion() {
        let suggester = BRollSuggester::new(BRollConfig {
            min_priority: 0.0,
            ..BRollConfig::default()
        });

        let scenes = vec![make_scene(0, 5000, 0.10)];
        let suggestions = suggester
            .suggest(&scenes, &[])
            .expect("suggest should succeed");
        assert!(
            !suggestions.is_empty(),
            "Low-importance scene should generate a B-roll suggestion"
        );
    }

    #[test]
    fn test_max_suggestions_limit() {
        let suggester = BRollSuggester::new(BRollConfig {
            max_suggestions: 2,
            min_priority: 0.0,
            ..BRollConfig::default()
        });

        let scenes: Vec<ScoredScene> = (0..20)
            .map(|i| make_scene(i * 2000, (i + 1) * 2000, 0.05))
            .collect();

        let suggestions = suggester
            .suggest(&scenes, &[])
            .expect("suggest should succeed");
        assert!(
            suggestions.len() <= 2,
            "Should respect max_suggestions limit"
        );
    }

    #[test]
    fn test_custom_keyword_mapping() {
        let cfg = BRollConfig::default().with_keyword(KeywordMapping::new(
            "volcano",
            BRollCategory::Nature,
            0.40,
        ));
        let suggester = BRollSuggester::new(BRollConfig {
            min_priority: 0.0,
            ..cfg
        });

        let scenes = vec![make_scene(0, 10_000, 0.50)];
        let text = vec![make_text_segment(0, 5000, "erupting volcano today")];

        let suggestions = suggester
            .suggest_with_text(&scenes, &[], &text)
            .expect("suggest with text should succeed");
        let nature: Vec<_> = suggestions
            .iter()
            .filter(|s| s.category == BRollCategory::Nature)
            .collect();
        assert!(
            !nature.is_empty(),
            "Custom keyword 'volcano' should map to Nature"
        );
    }

    #[test]
    fn test_suggestions_sorted_by_priority() {
        let suggester = BRollSuggester::new(BRollConfig {
            min_priority: 0.0,
            ..BRollConfig::default()
        });

        let scenes: Vec<ScoredScene> = (0..5)
            .map(|i| make_scene(i * 3000, (i + 1) * 3000, 0.05 + i as f64 * 0.05))
            .collect();

        let suggestions = suggester
            .suggest(&scenes, &[])
            .expect("suggest should succeed");
        for window in suggestions.windows(2) {
            assert!(
                window[0].priority >= window[1].priority,
                "Suggestions should be sorted by priority (desc)"
            );
        }
    }

    #[test]
    fn test_slot_duration() {
        let tb = Rational::new(1, 1000);
        let s = BRollSuggestion {
            start: Timestamp::new(0, tb),
            end: Timestamp::new(3000, tb),
            category: BRollCategory::Generic,
            priority: 0.5,
            reason: "test".into(),
            dialogue_keywords: Vec::new(),
            is_mandatory: false,
            suggested_min_duration_ms: 1000,
        };
        assert_eq!(s.slot_duration_ms(), 3000);
    }

    #[test]
    fn test_overlapping_deduplication() {
        let suggester = BRollSuggester::new(BRollConfig {
            min_priority: 0.0,
            min_dialogue_gap_ms: 100,
            ..BRollConfig::default()
        });

        // Two dialogue gaps that result in overlapping suggestions
        let dialogue = vec![
            make_dialogue(0, 1000),
            make_dialogue(1200, 2000), // tiny gap → low priority
            make_dialogue(1300, 5000), // overlapping gap
        ];

        let suggestions = suggester
            .suggest(&[], &dialogue)
            .expect("suggest should succeed");
        // After deduplication, overlapping suggestions should be removed/merged
        for window in suggestions.windows(2) {
            assert!(
                !window[0].overlaps(&window[1]),
                "No overlapping suggestions should remain after deduplication"
            );
        }
    }

    #[test]
    fn test_format_suggestions_empty() {
        assert_eq!(format_suggestions(&[]), "No B-roll suggestions generated.");
    }

    #[test]
    fn test_format_suggestions_non_empty() {
        let tb = Rational::new(1, 1000);
        let sug = BRollSuggestion {
            start: Timestamp::new(0, tb),
            end: Timestamp::new(2000, tb),
            category: BRollCategory::Nature,
            priority: 0.75,
            reason: "Test".into(),
            dialogue_keywords: vec!["forest".into()],
            is_mandatory: false,
            suggested_min_duration_ms: 1000,
        };
        let output = format_suggestions(&[sug]);
        assert!(output.contains("Nature"));
        assert!(output.contains("0.75") || output.contains("0.7"));
    }

    #[test]
    fn test_filter_by_category() {
        let tb = Rational::new(1, 1000);
        let suggestions = vec![
            BRollSuggestion {
                start: Timestamp::new(0, tb),
                end: Timestamp::new(2000, tb),
                category: BRollCategory::Nature,
                priority: 0.60,
                reason: "A".into(),
                dialogue_keywords: Vec::new(),
                is_mandatory: false,
                suggested_min_duration_ms: 1000,
            },
            BRollSuggestion {
                start: Timestamp::new(3000, tb),
                end: Timestamp::new(5000, tb),
                category: BRollCategory::UrbanExterior,
                priority: 0.50,
                reason: "B".into(),
                dialogue_keywords: Vec::new(),
                is_mandatory: false,
                suggested_min_duration_ms: 1000,
            },
        ];

        let nature = filter_by_category(&suggestions, &BRollCategory::Nature);
        assert_eq!(nature.len(), 1);
        assert_eq!(nature[0].category, BRollCategory::Nature);
    }
}

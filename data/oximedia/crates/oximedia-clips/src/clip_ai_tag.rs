//! Automatic keyword suggestion for clips based on visual content analysis.
//!
//! This module provides a rule-based and statistical keyword suggestion system
//! that analyses clip metadata (file name, duration, camera metadata, existing
//! keywords) to propose relevant tags.  In a full production implementation
//! the visual-content analysis step would invoke a neural-network scene
//! classifier (e.g. from `oximedia-neural`); this module exposes the same
//! interface while providing a deterministic rule-based backend suitable for
//! testing and offline environments.

use crate::clip::Clip;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Confidence threshold ────────────────────────────────────────────────────

/// Default minimum confidence to include a suggestion.
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.5;

// ─── Data structures ─────────────────────────────────────────────────────────

/// A single keyword suggestion with an associated confidence score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KeywordSuggestion {
    /// The suggested keyword.
    pub keyword: String,
    /// Confidence in the range `[0.0, 1.0]`.
    pub confidence: f32,
    /// Human-readable reason for the suggestion.
    pub reason: String,
}

impl KeywordSuggestion {
    /// Creates a new suggestion.
    #[must_use]
    pub fn new(keyword: impl Into<String>, confidence: f32, reason: impl Into<String>) -> Self {
        Self {
            keyword: keyword.into(),
            confidence: confidence.clamp(0.0, 1.0),
            reason: reason.into(),
        }
    }
}

/// Result of running the AI tagger on a single clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiTagResult {
    /// The ID of the clip that was analysed.
    pub clip_id: crate::clip::ClipId,
    /// All suggested keywords above the configured threshold.
    pub suggestions: Vec<KeywordSuggestion>,
}

impl AiTagResult {
    /// Returns only suggestions with confidence `>= min_confidence`.
    #[must_use]
    pub fn filtered(&self, min_confidence: f32) -> Vec<&KeywordSuggestion> {
        self.suggestions
            .iter()
            .filter(|s| s.confidence >= min_confidence)
            .collect()
    }

    /// Returns the top-N suggestions ordered by descending confidence.
    #[must_use]
    pub fn top_n(&self, n: usize) -> Vec<&KeywordSuggestion> {
        let mut sorted: Vec<&KeywordSuggestion> = self.suggestions.iter().collect();
        sorted.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }
}

// ─── Tagger ──────────────────────────────────────────────────────────────────

/// Configuration for the `ClipAiTagger`.
#[derive(Debug, Clone)]
pub struct AiTaggerConfig {
    /// Minimum confidence score to include a suggestion.
    pub min_confidence: f32,
    /// Maximum number of suggestions to return per clip.
    pub max_suggestions: usize,
    /// Custom keyword → weight overrides.  Entries here are boosted by the
    /// specified multiplier before filtering.
    pub keyword_weights: HashMap<String, f32>,
}

impl Default for AiTaggerConfig {
    fn default() -> Self {
        Self {
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            max_suggestions: 10,
            keyword_weights: HashMap::new(),
        }
    }
}

/// Automatic keyword suggestion engine for clips.
///
/// The engine combines multiple heuristic signals:
///
/// 1. **File name tokens** – words in the file name (e.g. `_int_` → "interior")
/// 2. **Duration bucketing** – very short clips are tagged "clip-short",
///    long ones "clip-long"
/// 3. **Camera metadata** – ISO range → "low-light", frame rate → "slow-motion"
/// 4. **Existing keywords** – co-occurrence rules suggest related tags
#[derive(Debug, Clone)]
pub struct ClipAiTagger {
    config: AiTaggerConfig,
}

impl Default for ClipAiTagger {
    fn default() -> Self {
        Self::new(AiTaggerConfig::default())
    }
}

impl ClipAiTagger {
    /// Creates a new tagger with the given configuration.
    #[must_use]
    pub fn new(config: AiTaggerConfig) -> Self {
        Self { config }
    }

    /// Analyses a single clip and returns keyword suggestions.
    #[must_use]
    pub fn tag_clip(&self, clip: &Clip) -> AiTagResult {
        let mut raw: Vec<KeywordSuggestion> = Vec::new();

        self.analyse_file_name(clip, &mut raw);
        self.analyse_duration(clip, &mut raw);
        self.analyse_camera_metadata(clip, &mut raw);
        self.analyse_existing_keywords(clip, &mut raw);

        // Apply custom weight overrides.
        for s in &mut raw {
            if let Some(&w) = self.config.keyword_weights.get(&s.keyword) {
                s.confidence = (s.confidence * w).clamp(0.0, 1.0);
            }
        }

        // Filter and deduplicate.
        let mut seen: HashMap<String, f32> = HashMap::new();
        for s in raw {
            let entry = seen.entry(s.keyword.clone()).or_insert(0.0_f32);
            if s.confidence > *entry {
                *entry = s.confidence;
            }
        }

        let mut suggestions: Vec<KeywordSuggestion> = seen
            .into_iter()
            .filter(|(_, conf)| *conf >= self.config.min_confidence)
            .map(|(kw, conf)| KeywordSuggestion {
                keyword: kw.clone(),
                confidence: conf,
                reason: "combined-signal".to_string(),
            })
            .collect();

        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.truncate(self.config.max_suggestions);

        AiTagResult {
            clip_id: clip.id,
            suggestions,
        }
    }

    /// Analyses a batch of clips, returning one result per clip.
    #[must_use]
    pub fn tag_clips(&self, clips: &[Clip]) -> Vec<AiTagResult> {
        clips.iter().map(|c| self.tag_clip(c)).collect()
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn analyse_file_name(&self, clip: &Clip, out: &mut Vec<KeywordSuggestion>) {
        let stem = clip
            .file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Token-to-keyword mapping.
        let token_map: &[(&str, &str, f32)] = &[
            ("int", "interior", 0.75),
            ("interior", "interior", 0.85),
            ("ext", "exterior", 0.75),
            ("exterior", "exterior", 0.85),
            ("interview", "interview", 0.90),
            ("broll", "b-roll", 0.85),
            ("b_roll", "b-roll", 0.85),
            ("vox", "vox-pop", 0.80),
            ("aerial", "aerial", 0.85),
            ("drone", "aerial", 0.85),
            ("timelapse", "time-lapse", 0.90),
            ("slowmo", "slow-motion", 0.85),
            ("slo_mo", "slow-motion", 0.85),
            ("night", "night", 0.80),
            ("day", "day", 0.60),
            ("wide", "wide-shot", 0.75),
            ("closeup", "close-up", 0.80),
            ("close_up", "close-up", 0.80),
            ("cutaway", "cutaway", 0.85),
        ];

        for (token, keyword, conf) in token_map {
            if stem.contains(token) {
                out.push(KeywordSuggestion::new(
                    *keyword,
                    *conf,
                    format!("file-name token '{token}'"),
                ));
            }
        }
    }

    fn analyse_duration(&self, clip: &Clip, out: &mut Vec<KeywordSuggestion>) {
        if let Some(dur) = clip.effective_duration() {
            // Assume 24 fps if no frame rate is available.
            let fps = clip.frame_rate.map_or(24.0, |fr| fr.to_f64());
            let seconds = dur as f64 / fps;

            if seconds < 5.0 {
                out.push(KeywordSuggestion::new("clip-short", 0.70, "duration < 5 s"));
            } else if seconds > 120.0 {
                out.push(KeywordSuggestion::new(
                    "clip-long",
                    0.65,
                    "duration > 2 min",
                ));
            }
        }
    }

    fn analyse_camera_metadata(&self, clip: &Clip, out: &mut Vec<KeywordSuggestion>) {
        // ISO-based low-light detection from camera metadata.
        if let Some(cam) = &clip.camera {
            if let Some(iso) = cam.iso {
                if iso >= 3200 {
                    out.push(KeywordSuggestion::new(
                        "low-light",
                        0.80,
                        format!("ISO {iso}"),
                    ));
                }
            }
        }

        // Frame-rate-based slow-motion detection from clip's frame_rate.
        if let Some(fr) = clip.frame_rate {
            let fps = fr.to_f64();
            if fps > 60.0 {
                out.push(KeywordSuggestion::new(
                    "slow-motion",
                    0.85,
                    format!("{fps:.0} fps"),
                ));
            }
        }
    }

    fn analyse_existing_keywords(&self, clip: &Clip, out: &mut Vec<KeywordSuggestion>) {
        // Co-occurrence rules: if clip already has keyword A, suggest keyword B.
        let co_occur: &[(&str, &str, f32)] = &[
            ("interview", "talking-head", 0.70),
            ("interview", "dialogue", 0.65),
            ("b-roll", "cutaway", 0.65),
            ("aerial", "establishing-shot", 0.70),
            ("slow-motion", "action", 0.60),
            ("exterior", "establishing-shot", 0.55),
            ("low-light", "cinematic", 0.60),
        ];

        for (trigger, suggestion, conf) in co_occur {
            if clip.keywords.iter().any(|k| k == trigger) {
                out.push(KeywordSuggestion::new(
                    *suggestion,
                    *conf,
                    format!("co-occurrence with '{trigger}'"),
                ));
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_metadata::CameraMetadata;
    use std::path::PathBuf;

    fn make_clip(name: &str) -> Clip {
        Clip::new(PathBuf::from(format!("/media/{name}.mov")))
    }

    #[test]
    fn test_tag_clip_empty_no_suggestions_below_threshold() {
        let tagger = ClipAiTagger::default();
        let clip = make_clip("generic_clip");
        let result = tagger.tag_clip(&clip);
        // Should not panic; result may be empty.
        assert!(result.clip_id == clip.id);
    }

    #[test]
    fn test_tag_clip_file_name_interview() {
        let tagger = ClipAiTagger::default();
        let clip = make_clip("interview_001");
        let result = tagger.tag_clip(&clip);
        let has_interview = result.suggestions.iter().any(|s| s.keyword == "interview");
        assert!(has_interview, "Expected 'interview' suggestion");
    }

    #[test]
    fn test_tag_clip_file_name_broll() {
        let tagger = ClipAiTagger::default();
        let clip = make_clip("broll_city");
        let result = tagger.tag_clip(&clip);
        let has_broll = result.suggestions.iter().any(|s| s.keyword == "b-roll");
        assert!(has_broll, "Expected 'b-roll' suggestion");
    }

    #[test]
    fn test_tag_clip_duration_short() {
        let tagger = ClipAiTagger::default();
        let mut clip = make_clip("stinger");
        clip.set_duration(60); // 60 frames @ 24fps = 2.5s
        let result = tagger.tag_clip(&clip);
        let has_short = result.suggestions.iter().any(|s| s.keyword == "clip-short");
        assert!(has_short, "Expected 'clip-short' suggestion");
    }

    #[test]
    fn test_tag_clip_camera_low_light() {
        let tagger = ClipAiTagger::default();
        let mut clip = make_clip("night_scene");
        let mut cam = CameraMetadata::default();
        cam.iso = Some(6400);
        clip.set_camera_metadata(cam);
        let result = tagger.tag_clip(&clip);
        let has_low_light = result.suggestions.iter().any(|s| s.keyword == "low-light");
        assert!(has_low_light, "Expected 'low-light' suggestion from ISO");
    }

    #[test]
    fn test_tag_clip_camera_slow_motion() {
        use oximedia_core::types::Rational;
        let tagger = ClipAiTagger::default();
        let mut clip = make_clip("sports");
        // 120 fps → slow-motion
        clip.set_frame_rate(Rational::new(120, 1));
        let result = tagger.tag_clip(&clip);
        let has_slo = result
            .suggestions
            .iter()
            .any(|s| s.keyword == "slow-motion");
        assert!(has_slo, "Expected 'slow-motion' from high fps");
    }

    #[test]
    fn test_tag_clip_co_occurrence() {
        let tagger = ClipAiTagger::default();
        let mut clip = make_clip("clip");
        clip.add_keyword("interview");
        let result = tagger.tag_clip(&clip);
        let has_th = result
            .suggestions
            .iter()
            .any(|s| s.keyword == "talking-head");
        assert!(has_th, "Expected 'talking-head' co-occurrence suggestion");
    }

    #[test]
    fn test_top_n_returns_sorted_descending() {
        let tagger = ClipAiTagger::default();
        let clip = make_clip("broll_interview_aerial");
        let result = tagger.tag_clip(&clip);
        let top = result.top_n(3);
        if top.len() >= 2 {
            assert!(top[0].confidence >= top[1].confidence);
        }
    }

    #[test]
    fn test_filtered_min_confidence() {
        let tagger = ClipAiTagger::default();
        let clip = make_clip("interview_clip");
        let result = tagger.tag_clip(&clip);
        let high = result.filtered(0.9);
        for s in high {
            assert!(s.confidence >= 0.9);
        }
    }

    #[test]
    fn test_tag_clips_batch() {
        let tagger = ClipAiTagger::default();
        let clips = vec![make_clip("interview_01"), make_clip("broll_outdoor")];
        let results = tagger.tag_clips(&clips);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_max_suggestions_respected() {
        let config = AiTaggerConfig {
            max_suggestions: 2,
            min_confidence: 0.0,
            ..AiTaggerConfig::default()
        };
        let tagger = ClipAiTagger::new(config);
        // A clip that would trigger many signals.
        let clip = make_clip("broll_interview_aerial_night_drone");
        let result = tagger.tag_clip(&clip);
        assert!(result.suggestions.len() <= 2);
    }
}

//! Forced subtitle detection and flagging.
//!
//! "Forced subtitles" are subtitle cues that should be shown even when
//! subtitles are turned off — typically used for foreign-language dialogue
//! in an otherwise native-language film (e.g., Klingon in a Star Trek movie,
//! on-screen text, signs, etc.).
//!
//! This module provides heuristics to detect forced subtitle candidates and
//! flag them accordingly.

#![allow(dead_code)]

use crate::{Subtitle, SubtitleResult};

// ============================================================================
// Types
// ============================================================================

/// Classification of a subtitle cue for forced-subtitle analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedSubtitleClass {
    /// Normal subtitle — full dialogue translation.
    Normal,
    /// Forced subtitle — foreign-language/sign that should always show.
    Forced,
    /// Uncertain — heuristic confidence is below threshold.
    Uncertain,
}

/// A single detection result for a subtitle cue.
#[derive(Debug, Clone)]
pub struct ForcedDetectionResult {
    /// Index of the cue in the original track.
    pub cue_index: usize,
    /// Classification.
    pub classification: ForcedSubtitleClass,
    /// Confidence score (0.0 – 1.0) for the forced classification.
    pub confidence: f64,
    /// Which heuristic signals were triggered.
    pub signals: Vec<ForcedSignal>,
}

/// Individual signals that contribute to forced subtitle detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedSignal {
    /// Cue is short relative to surrounding cues (isolated foreign phrase).
    ShortDuration,
    /// Cue text is significantly shorter than the track average.
    ShortText,
    /// Cue appears in a sparse region (large gap before and after).
    Sparse,
    /// Cue text contains non-Latin script characters (likely foreign).
    NonLatinScript,
    /// Cue text is surrounded by brackets, parentheses, or similar markers.
    Bracketed,
    /// Cue has an italic style hint (common convention for foreign dialogue).
    ItalicStyled,
    /// Cue occurs during a gap in the main dialogue track.
    GapInDialogue,
    /// Text density (chars / duration) is much lower than track average.
    LowDensity,
}

/// Configuration for forced subtitle detection heuristics.
#[derive(Debug, Clone)]
pub struct ForcedDetectionConfig {
    /// Minimum confidence to classify as `Forced` (default 0.6).
    pub forced_threshold: f64,
    /// Below this confidence and above `uncertain_floor`, classify as `Uncertain`.
    pub uncertain_floor: f64,
    /// Maximum fraction of cues that can be forced in a normal track.
    /// If the fraction exceeds this, it is likely a full translation track
    /// and nothing should be flagged forced (default 0.3 = 30%).
    pub max_forced_fraction: f64,
    /// Short-duration threshold in milliseconds (default 3000).
    pub short_duration_ms: i64,
    /// Short-text threshold in characters (default 40).
    pub short_text_chars: usize,
    /// Sparse gap threshold in milliseconds (default 5000).
    pub sparse_gap_ms: i64,
}

impl Default for ForcedDetectionConfig {
    fn default() -> Self {
        Self {
            forced_threshold: 0.6,
            uncertain_floor: 0.3,
            max_forced_fraction: 0.3,
            short_duration_ms: 3000,
            short_text_chars: 40,
            sparse_gap_ms: 5000,
        }
    }
}

/// Report summarising forced subtitle detection across a track.
#[derive(Debug, Clone)]
pub struct ForcedDetectionReport {
    /// Per-cue results.
    pub results: Vec<ForcedDetectionResult>,
    /// Total number of cues analysed.
    pub total_cues: usize,
    /// Number of cues classified as forced.
    pub forced_count: usize,
    /// Number of cues classified as uncertain.
    pub uncertain_count: usize,
    /// Average confidence across all cues.
    pub avg_confidence: f64,
    /// Whether the entire track looks like a full translation (not a forced track).
    pub is_full_translation: bool,
}

impl ForcedDetectionReport {
    /// Indices of cues classified as forced.
    #[must_use]
    pub fn forced_indices(&self) -> Vec<usize> {
        self.results
            .iter()
            .filter(|r| r.classification == ForcedSubtitleClass::Forced)
            .map(|r| r.cue_index)
            .collect()
    }

    /// Fraction of cues classified as forced (0.0 – 1.0).
    #[must_use]
    pub fn forced_fraction(&self) -> f64 {
        if self.total_cues == 0 {
            return 0.0;
        }
        self.forced_count as f64 / self.total_cues as f64
    }
}

// ============================================================================
// Detector
// ============================================================================

/// Detects forced subtitles using multiple heuristic signals.
#[derive(Debug, Clone)]
pub struct ForcedSubtitleDetector {
    /// Configuration.
    pub config: ForcedDetectionConfig,
}

impl ForcedSubtitleDetector {
    /// Create a detector with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ForcedDetectionConfig::default(),
        }
    }

    /// Create a detector with custom configuration.
    #[must_use]
    pub fn with_config(config: ForcedDetectionConfig) -> Self {
        Self { config }
    }

    /// Analyse a subtitle track and produce a detection report.
    #[must_use]
    pub fn detect(&self, subtitles: &[Subtitle]) -> ForcedDetectionReport {
        if subtitles.is_empty() {
            return ForcedDetectionReport {
                results: Vec::new(),
                total_cues: 0,
                forced_count: 0,
                uncertain_count: 0,
                avg_confidence: 0.0,
                is_full_translation: false,
            };
        }

        // Compute track-level statistics
        let stats = TrackStats::compute(subtitles);

        // Classify each cue
        let mut results: Vec<ForcedDetectionResult> = subtitles
            .iter()
            .enumerate()
            .map(|(i, sub)| self.classify_cue(i, sub, subtitles, &stats))
            .collect();

        // Check if the track looks like a full translation
        let forced_count_raw = results
            .iter()
            .filter(|r| r.classification == ForcedSubtitleClass::Forced)
            .count();
        let fraction = forced_count_raw as f64 / subtitles.len() as f64;
        let is_full_translation = fraction > self.config.max_forced_fraction;

        // If the track is a full translation, downgrade all forced → Normal
        if is_full_translation {
            for r in &mut results {
                if r.classification == ForcedSubtitleClass::Forced {
                    r.classification = ForcedSubtitleClass::Normal;
                }
            }
        }

        let forced_count = results
            .iter()
            .filter(|r| r.classification == ForcedSubtitleClass::Forced)
            .count();
        let uncertain_count = results
            .iter()
            .filter(|r| r.classification == ForcedSubtitleClass::Uncertain)
            .count();
        let avg_confidence = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64
        };

        ForcedDetectionReport {
            total_cues: subtitles.len(),
            forced_count,
            uncertain_count,
            avg_confidence,
            is_full_translation,
            results,
        }
    }

    /// Extract only the forced subtitles from a track.
    #[must_use]
    pub fn extract_forced(&self, subtitles: &[Subtitle]) -> Vec<Subtitle> {
        let report = self.detect(subtitles);
        report
            .forced_indices()
            .into_iter()
            .filter_map(|i| subtitles.get(i).cloned())
            .collect()
    }

    /// Flag forced subtitles in-place by setting their `id` to include `FORCED`.
    pub fn flag_forced(&self, subtitles: &mut [Subtitle]) {
        let report = self.detect(subtitles);
        for idx in report.forced_indices() {
            if let Some(sub) = subtitles.get_mut(idx) {
                let current_id = sub.id.clone().unwrap_or_default();
                sub.id = Some(format!("[FORCED] {current_id}").trim().to_string());
            }
        }
    }

    /// Classify a single cue.
    fn classify_cue(
        &self,
        index: usize,
        sub: &Subtitle,
        all: &[Subtitle],
        stats: &TrackStats,
    ) -> ForcedDetectionResult {
        let mut signals = Vec::new();
        let mut score = 0.0f64;

        let duration = sub.duration();
        let text_len = sub.text.chars().filter(|c| !c.is_whitespace()).count();

        // Signal: short duration
        if duration < self.config.short_duration_ms && duration > 0 {
            if stats.avg_duration > 0.0 && (duration as f64) < stats.avg_duration * 0.5 {
                signals.push(ForcedSignal::ShortDuration);
                score += 0.15;
            }
        }

        // Signal: short text
        if text_len < self.config.short_text_chars && text_len > 0 {
            if stats.avg_text_len > 0.0 && (text_len as f64) < stats.avg_text_len * 0.4 {
                signals.push(ForcedSignal::ShortText);
                score += 0.10;
            }
        }

        // Signal: sparse (large gaps before and after)
        let gap_before = if index > 0 {
            sub.start_time - all[index - 1].end_time
        } else {
            sub.start_time // gap from start of media
        };
        let gap_after = if index + 1 < all.len() {
            all[index + 1].start_time - sub.end_time
        } else {
            self.config.sparse_gap_ms + 1 // treat end-of-track as sparse
        };

        if gap_before > self.config.sparse_gap_ms || gap_after > self.config.sparse_gap_ms {
            signals.push(ForcedSignal::Sparse);
            score += 0.15;
        }

        // Signal: non-Latin script (CJK, Cyrillic, Arabic, Hebrew, Devanagari, etc.)
        let non_latin_ratio = non_latin_ratio(&sub.text);
        if non_latin_ratio > 0.3 {
            signals.push(ForcedSignal::NonLatinScript);
            score += 0.20 * non_latin_ratio;
        }

        // Signal: bracketed text [text], (text), *text*
        if is_bracketed(&sub.text) {
            signals.push(ForcedSignal::Bracketed);
            score += 0.20;
        }

        // Signal: italic style hint
        if let Some(ref style) = sub.style {
            if style.font_style == crate::style::FontStyle::Italic {
                signals.push(ForcedSignal::ItalicStyled);
                score += 0.15;
            }
        }
        // Also check if text contains <i> tags (common in SRT)
        if sub.text.contains("<i>") || sub.text.contains("</i>") {
            if !signals.contains(&ForcedSignal::ItalicStyled) {
                signals.push(ForcedSignal::ItalicStyled);
                score += 0.10;
            }
        }

        // Signal: low density
        if duration > 0 {
            let density = text_len as f64 / (duration as f64 / 1000.0);
            if stats.avg_density > 0.0 && density < stats.avg_density * 0.3 {
                signals.push(ForcedSignal::LowDensity);
                score += 0.10;
            }
        }

        let confidence = score.min(1.0);
        let classification = if confidence >= self.config.forced_threshold {
            ForcedSubtitleClass::Forced
        } else if confidence >= self.config.uncertain_floor {
            ForcedSubtitleClass::Uncertain
        } else {
            ForcedSubtitleClass::Normal
        };

        ForcedDetectionResult {
            cue_index: index,
            classification,
            confidence,
            signals,
        }
    }
}

impl Default for ForcedSubtitleDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Track-level Statistics
// ============================================================================

/// Pre-computed statistics for the subtitle track.
struct TrackStats {
    avg_duration: f64,
    avg_text_len: f64,
    avg_density: f64,
}

impl TrackStats {
    fn compute(subtitles: &[Subtitle]) -> Self {
        if subtitles.is_empty() {
            return Self {
                avg_duration: 0.0,
                avg_text_len: 0.0,
                avg_density: 0.0,
            };
        }

        let n = subtitles.len() as f64;
        let total_duration: i64 = subtitles.iter().map(|s| s.duration().max(0)).sum();
        let total_text: usize = subtitles
            .iter()
            .map(|s| s.text.chars().filter(|c| !c.is_whitespace()).count())
            .sum();
        let avg_duration = total_duration as f64 / n;
        let avg_text_len = total_text as f64 / n;

        let densities: Vec<f64> = subtitles
            .iter()
            .filter_map(|s| {
                let dur = s.duration();
                if dur > 0 {
                    let chars = s.text.chars().filter(|c| !c.is_whitespace()).count();
                    Some(chars as f64 / (dur as f64 / 1000.0))
                } else {
                    None
                }
            })
            .collect();
        let avg_density = if densities.is_empty() {
            0.0
        } else {
            densities.iter().sum::<f64>() / densities.len() as f64
        };

        Self {
            avg_duration,
            avg_text_len,
            avg_density,
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Calculate the fraction of non-Latin characters in the text.
fn non_latin_ratio(text: &str) -> f64 {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() {
        return 0.0;
    }

    let non_latin = chars
        .iter()
        .filter(|&&c| {
            !c.is_ascii_alphanumeric()
                && !c.is_ascii_punctuation()
                && c != '\''
                && c != '"'
                && c != '-'
        })
        .count();

    non_latin as f64 / chars.len() as f64
}

/// Check if the text is enclosed in brackets or similar markers.
fn is_bracketed(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.len() < 2 {
        return false;
    }

    let first = trimmed.chars().next();
    let last = trimmed.chars().last();

    matches!(
        (first, last),
        (Some('['), Some(']'))
            | (Some('('), Some(')'))
            | (Some('*'), Some('*'))
            | (Some('♪'), Some('♪'))
            | (Some('♫'), Some('♫'))
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::FontStyle;

    fn make_sub(start: i64, end: i64, text: &str) -> Subtitle {
        Subtitle::new(start, end, text.to_string())
    }

    // Build a track with many normal dialogue subs and a few sparse short ones
    fn make_mixed_track() -> Vec<Subtitle> {
        let mut subs = Vec::new();
        // Normal dialogue: 10 cues, 3 seconds each, 50+ chars
        for i in 0..10 {
            let start = i * 4000;
            let end = start + 3000;
            subs.push(make_sub(
                start,
                end,
                "This is a normal length dialogue subtitle that has many words",
            ));
        }
        // Gap, then a short forced-like cue
        let forced_start = 50000;
        let forced_end = 51500;
        subs.push(make_sub(forced_start, forced_end, "[外国語]"));
        subs
    }

    #[test]
    fn test_forced_detection_empty_track() {
        let detector = ForcedSubtitleDetector::new();
        let report = detector.detect(&[]);
        assert_eq!(report.total_cues, 0);
        assert_eq!(report.forced_count, 0);
        assert!(!report.is_full_translation);
    }

    #[test]
    fn test_forced_detection_normal_dialogue() {
        let subs: Vec<Subtitle> = (0..5)
            .map(|i| make_sub(i * 4000, i * 4000 + 3000, "Normal dialogue subtitle here"))
            .collect();
        let detector = ForcedSubtitleDetector::new();
        let report = detector.detect(&subs);
        assert_eq!(report.forced_count, 0);
    }

    #[test]
    fn test_forced_detection_bracketed() {
        let subs = make_mixed_track();
        // The last cue "[外国語]" is bracketed + non-Latin + sparse + short
        let detector = ForcedSubtitleDetector::new();
        let report = detector.detect(&subs);
        let last_idx = subs.len() - 1;
        let last_result = &report.results[last_idx];
        // Should have multiple signals
        assert!(
            last_result.signals.contains(&ForcedSignal::Bracketed),
            "signals: {:?}",
            last_result.signals
        );
        assert!(
            last_result.signals.contains(&ForcedSignal::NonLatinScript),
            "signals: {:?}",
            last_result.signals
        );
    }

    #[test]
    fn test_forced_detection_non_latin() {
        let ratio = non_latin_ratio("これは日本語です");
        assert!(ratio > 0.9, "ratio={ratio}");
    }

    #[test]
    fn test_non_latin_ratio_latin() {
        let ratio = non_latin_ratio("Hello world!");
        assert!(ratio < 0.01, "ratio={ratio}");
    }

    #[test]
    fn test_non_latin_ratio_empty() {
        assert!((non_latin_ratio("") - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_is_bracketed_square() {
        assert!(is_bracketed("[Foreign language]"));
    }

    #[test]
    fn test_is_bracketed_parens() {
        assert!(is_bracketed("(speaking Japanese)"));
    }

    #[test]
    fn test_is_bracketed_music() {
        assert!(is_bracketed("♪ La la la ♪"));
    }

    #[test]
    fn test_is_bracketed_none() {
        assert!(!is_bracketed("Normal text"));
    }

    #[test]
    fn test_is_bracketed_short() {
        assert!(!is_bracketed("x"));
    }

    #[test]
    fn test_forced_classification_enum() {
        assert_ne!(ForcedSubtitleClass::Forced, ForcedSubtitleClass::Normal);
        assert_ne!(ForcedSubtitleClass::Uncertain, ForcedSubtitleClass::Normal);
    }

    #[test]
    fn test_forced_signal_enum() {
        let s = ForcedSignal::ShortDuration;
        assert_eq!(s, ForcedSignal::ShortDuration);
    }

    #[test]
    fn test_full_translation_detection() {
        // Use a very low threshold so that most cues are classified forced,
        // triggering the max_forced_fraction guard.
        let config = ForcedDetectionConfig {
            forced_threshold: 0.05,
            uncertain_floor: 0.02,
            max_forced_fraction: 0.3,
            short_duration_ms: 3000,
            short_text_chars: 100,
            sparse_gap_ms: 2000,
        };
        // All cues are bracketed non-Latin text with large gaps
        let subs: Vec<Subtitle> = (0..10)
            .map(|i| {
                let start = i * 10000;
                let end = start + 1500;
                make_sub(start, end, "[日本語テスト]")
            })
            .collect();
        let detector = ForcedSubtitleDetector::with_config(config);
        let report = detector.detect(&subs);
        assert!(
            report.is_full_translation,
            "Should detect full translation track, forced_count_raw would exceed 30% with low threshold"
        );
        assert_eq!(
            report.forced_count, 0,
            "Full translation should have 0 forced after downgrade"
        );
    }

    #[test]
    fn test_extract_forced() {
        let subs = make_mixed_track();
        let detector = ForcedSubtitleDetector::new();
        let forced = detector.extract_forced(&subs);
        // The exact number depends on heuristic scores, but we can check it doesn't crash
        // and returns a subset
        assert!(forced.len() <= subs.len());
    }

    #[test]
    fn test_flag_forced() {
        let mut subs = make_mixed_track();
        let detector = ForcedSubtitleDetector::new();
        detector.flag_forced(&mut subs);
        // Check that any flagged cues have "[FORCED]" in their id
        for sub in &subs {
            if let Some(ref id) = sub.id {
                if id.contains("[FORCED]") {
                    // Confirm it was actually classified as forced
                    assert!(id.starts_with("[FORCED]"));
                }
            }
        }
    }

    #[test]
    fn test_forced_report_fraction() {
        let report = ForcedDetectionReport {
            results: Vec::new(),
            total_cues: 10,
            forced_count: 3,
            uncertain_count: 1,
            avg_confidence: 0.5,
            is_full_translation: false,
        };
        assert!((report.forced_fraction() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_forced_report_fraction_empty() {
        let report = ForcedDetectionReport {
            results: Vec::new(),
            total_cues: 0,
            forced_count: 0,
            uncertain_count: 0,
            avg_confidence: 0.0,
            is_full_translation: false,
        };
        assert!((report.forced_fraction() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_forced_config_default() {
        let config = ForcedDetectionConfig::default();
        assert!((config.forced_threshold - 0.6).abs() < 0.01);
        assert!((config.uncertain_floor - 0.3).abs() < 0.01);
        assert_eq!(config.short_duration_ms, 3000);
    }

    #[test]
    fn test_forced_italic_signal() {
        let mut subs = Vec::new();
        // Normal dialogue
        for i in 0..10 {
            subs.push(make_sub(
                i * 4000,
                i * 4000 + 3000,
                "Normal dialogue text that is quite long indeed for testing",
            ));
        }
        // Italic foreign subtitle far from others
        let mut foreign = make_sub(60000, 61500, "Sayonara");
        let mut style = crate::SubtitleStyle::default();
        style.font_style = FontStyle::Italic;
        foreign.style = Some(style);
        subs.push(foreign);

        let detector = ForcedSubtitleDetector::new();
        let report = detector.detect(&subs);
        let last = &report.results[subs.len() - 1];
        assert!(
            last.signals.contains(&ForcedSignal::ItalicStyled),
            "signals: {:?}",
            last.signals
        );
    }

    #[test]
    fn test_custom_config() {
        let config = ForcedDetectionConfig {
            forced_threshold: 0.9,
            uncertain_floor: 0.5,
            max_forced_fraction: 0.1,
            short_duration_ms: 2000,
            short_text_chars: 20,
            sparse_gap_ms: 3000,
        };
        let detector = ForcedSubtitleDetector::with_config(config);
        let report = detector.detect(&[make_sub(0, 1000, "Hi")]);
        assert_eq!(report.total_cues, 1);
    }

    #[test]
    fn test_forced_detector_default_trait() {
        let d = ForcedSubtitleDetector::default();
        assert!((d.config.forced_threshold - 0.6).abs() < 0.01);
    }
}

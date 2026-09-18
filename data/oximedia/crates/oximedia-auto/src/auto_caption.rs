//! Automatic caption generation for video content.
//!
//! Provides `CaptionLanguage`, `AutoCaptionConfig`, `CaptionSegment`, and
//! `AutoCaptionResult` to drive speech-to-text caption workflows.

#![allow(dead_code)]

/// Language tag for captions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CaptionLanguage {
    /// BCP-47 language tag (e.g. `"en-US"`, `"ar"`, `"he"`).
    pub tag: String,
    /// Human-readable name (e.g. `"English"`).
    pub name: String,
}

impl CaptionLanguage {
    /// Create a new `CaptionLanguage`.
    #[must_use]
    pub fn new(tag: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            name: name.into(),
        }
    }

    /// Return `true` when the language is written right-to-left.
    ///
    /// Currently recognises Arabic (`ar`), Hebrew (`he`), Persian (`fa`),
    /// and Urdu (`ur`).
    #[must_use]
    pub fn is_rtl(&self) -> bool {
        let prefix = self.tag.split('-').next().unwrap_or("");
        matches!(prefix, "ar" | "he" | "fa" | "ur")
    }
}

/// Configuration for automatic caption generation.
#[derive(Debug, Clone)]
pub struct AutoCaptionConfig {
    /// Minimum word confidence score (0.0–1.0) to include a word.
    pub min_word_confidence: f64,
    /// Maximum number of words per caption line.
    pub max_words_per_line: usize,
    /// Maximum line duration in milliseconds.
    pub max_line_duration_ms: u64,
    /// Target language.
    pub language: CaptionLanguage,
    /// Whether to emit speaker labels.
    pub speaker_labels: bool,
}

impl Default for AutoCaptionConfig {
    fn default() -> Self {
        Self {
            min_word_confidence: 0.70,
            max_words_per_line: 10,
            max_line_duration_ms: 5_000,
            language: CaptionLanguage::new("en-US", "English"),
            speaker_labels: false,
        }
    }
}

impl AutoCaptionConfig {
    /// Return `true` when `confidence` meets or exceeds the configured threshold.
    #[must_use]
    pub fn confidence_threshold_ok(&self, confidence: f64) -> bool {
        confidence >= self.min_word_confidence
    }
}

/// A single timed caption segment.
#[derive(Debug, Clone)]
pub struct CaptionSegment {
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Caption text.
    pub text: String,
    /// Average word confidence for this segment (0.0–1.0).
    pub confidence: f64,
    /// Optional speaker identifier.
    pub speaker: Option<String>,
}

impl CaptionSegment {
    /// Create a new `CaptionSegment`.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, text: impl Into<String>, confidence: f64) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
            confidence,
            speaker: None,
        }
    }

    /// Return the number of whitespace-delimited words in the caption text.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Return the duration of this segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Result of an automatic captioning run.
#[derive(Debug, Clone, Default)]
pub struct AutoCaptionResult {
    /// All generated segments (may include low-confidence ones).
    pub segments: Vec<CaptionSegment>,
    /// Overall confidence for the run (average of all segments).
    pub overall_confidence: f64,
    /// Language detected or used.
    pub language: Option<CaptionLanguage>,
}

impl AutoCaptionResult {
    /// Create a new result from a list of segments.
    #[must_use]
    pub fn from_segments(segments: Vec<CaptionSegment>) -> Self {
        let overall_confidence = if segments.is_empty() {
            0.0
        } else {
            segments.iter().map(|s| s.confidence).sum::<f64>() / segments.len() as f64
        };
        Self {
            segments,
            overall_confidence,
            language: None,
        }
    }

    /// Return only segments whose confidence is at least `min_confidence`.
    #[must_use]
    pub fn filter_by_confidence(&self, min_confidence: f64) -> Vec<&CaptionSegment> {
        self.segments
            .iter()
            .filter(|s| s.confidence >= min_confidence)
            .collect()
    }

    /// Return the total number of words across all segments.
    #[must_use]
    pub fn total_word_count(&self) -> usize {
        self.segments.iter().map(CaptionSegment::word_count).sum()
    }

    /// Return the total duration covered by all segments (end of last – start of first).
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        match (self.segments.first(), self.segments.last()) {
            (Some(first), Some(last)) => last.end_ms.saturating_sub(first.start_ms),
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caption_language_is_rtl_arabic() {
        let lang = CaptionLanguage::new("ar", "Arabic");
        assert!(lang.is_rtl());
    }

    #[test]
    fn test_caption_language_is_rtl_hebrew() {
        let lang = CaptionLanguage::new("he", "Hebrew");
        assert!(lang.is_rtl());
    }

    #[test]
    fn test_caption_language_not_rtl_english() {
        let lang = CaptionLanguage::new("en-US", "English");
        assert!(!lang.is_rtl());
    }

    #[test]
    fn test_caption_language_not_rtl_japanese() {
        let lang = CaptionLanguage::new("ja", "Japanese");
        assert!(!lang.is_rtl());
    }

    #[test]
    fn test_config_confidence_threshold_ok() {
        let cfg = AutoCaptionConfig::default();
        assert!(cfg.confidence_threshold_ok(0.75));
        assert!(!cfg.confidence_threshold_ok(0.60));
        assert!(cfg.confidence_threshold_ok(0.70));
    }

    #[test]
    fn test_segment_word_count() {
        let seg = CaptionSegment::new(0, 1000, "hello world foo bar", 0.9);
        assert_eq!(seg.word_count(), 4);
    }

    #[test]
    fn test_segment_word_count_empty() {
        let seg = CaptionSegment::new(0, 0, "", 0.0);
        assert_eq!(seg.word_count(), 0);
    }

    #[test]
    fn test_segment_duration_ms() {
        let seg = CaptionSegment::new(1000, 3500, "test", 0.8);
        assert_eq!(seg.duration_ms(), 2500);
    }

    #[test]
    fn test_result_from_segments_overall_confidence() {
        let segs = vec![
            CaptionSegment::new(0, 1000, "one", 0.8),
            CaptionSegment::new(1000, 2000, "two", 0.6),
        ];
        let result = AutoCaptionResult::from_segments(segs);
        assert!((result.overall_confidence - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_result_from_segments_empty() {
        let result = AutoCaptionResult::from_segments(vec![]);
        assert_eq!(result.overall_confidence, 0.0);
    }

    #[test]
    fn test_result_filter_by_confidence() {
        let segs = vec![
            CaptionSegment::new(0, 1000, "high", 0.9),
            CaptionSegment::new(1000, 2000, "low", 0.5),
        ];
        let result = AutoCaptionResult::from_segments(segs);
        let filtered = result.filter_by_confidence(0.8);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].text, "high");
    }

    #[test]
    fn test_result_total_word_count() {
        let segs = vec![
            CaptionSegment::new(0, 1000, "one two", 0.9),
            CaptionSegment::new(1000, 2000, "three four five", 0.9),
        ];
        let result = AutoCaptionResult::from_segments(segs);
        assert_eq!(result.total_word_count(), 5);
    }

    #[test]
    fn test_result_total_duration_ms() {
        let segs = vec![
            CaptionSegment::new(500, 1500, "a", 0.9),
            CaptionSegment::new(2000, 4000, "b", 0.9),
        ];
        let result = AutoCaptionResult::from_segments(segs);
        assert_eq!(result.total_duration_ms(), 3500);
    }

    #[test]
    fn test_caption_language_farsi_is_rtl() {
        let lang = CaptionLanguage::new("fa-IR", "Persian");
        assert!(lang.is_rtl());
    }
}

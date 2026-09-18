//! Automatic subtitle synchronization module.
//!
//! Provides tools for aligning subtitle lines to audio events, detecting
//! sync offsets, and validating subtitle timing constraints.

#![allow(dead_code)]

/// A single subtitle entry with timing and text.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleLine {
    /// Displayed text.
    pub text: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Confidence that this line is correctly aligned (0.0–1.0).
    pub confidence: f64,
}

impl SubtitleLine {
    /// Create a new subtitle line.
    #[must_use]
    pub fn new(text: impl Into<String>, start_ms: u64, end_ms: u64, confidence: f64) -> Self {
        Self {
            text: text.into(),
            start_ms,
            end_ms,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Duration of the subtitle in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Number of whitespace-delimited words in the text.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Reading speed in characters per second.
    ///
    /// Returns `0.0` if the duration is zero.
    #[must_use]
    pub fn reading_speed_cps(&self) -> f64 {
        let dur_s = self.duration_ms() as f64 / 1000.0;
        if dur_s <= 0.0 {
            return 0.0;
        }
        self.text.chars().count() as f64 / dur_s
    }
}

/// Align subtitle lines to detected audio events (word/phrase onset times).
///
/// Each subtitle is shifted so its start time matches the nearest audio event
/// that falls after the subtitle's current start.  If no suitable event is
/// found the line is left unchanged.
#[must_use]
pub fn align_subtitle_to_audio(
    subtitle: &[SubtitleLine],
    audio_events_ms: &[u64],
) -> Vec<SubtitleLine> {
    subtitle
        .iter()
        .map(|line| {
            // Find the closest audio event >= line.start_ms.
            let nearest = audio_events_ms
                .iter()
                .filter(|&&t| t >= line.start_ms)
                .min_by_key(|&&t| t.saturating_sub(line.start_ms));

            if let Some(&event_ms) = nearest {
                let shift = event_ms as i64 - line.start_ms as i64;
                let new_start = (line.start_ms as i64 + shift).max(0) as u64;
                let new_end = (line.end_ms as i64 + shift).max(0) as u64;
                SubtitleLine {
                    text: line.text.clone(),
                    start_ms: new_start,
                    end_ms: new_end,
                    confidence: line.confidence,
                }
            } else {
                line.clone()
            }
        })
        .collect()
}

/// Detect the global sync offset between subtitle timings and reference audio events.
///
/// Uses the median of the differences between subtitle start times and the nearest
/// reference timestamp.  Returns the offset (in ms) to add to all subtitle timestamps.
#[must_use]
pub fn detect_sync_offset(subtitles: &[SubtitleLine], reference_ms: &[u64]) -> i64 {
    if subtitles.is_empty() || reference_ms.is_empty() {
        return 0;
    }

    let mut diffs: Vec<i64> = subtitles
        .iter()
        .filter_map(|line| {
            // Nearest reference point to line.start_ms.
            reference_ms
                .iter()
                .min_by_key(|&&r| {
                    let diff = r as i64 - line.start_ms as i64;
                    diff.unsigned_abs()
                })
                .map(|&r| r as i64 - line.start_ms as i64)
        })
        .collect();

    if diffs.is_empty() {
        return 0;
    }

    diffs.sort_unstable();
    let mid = diffs.len() / 2;
    if diffs.len() % 2 == 0 {
        (diffs[mid - 1] + diffs[mid]) / 2
    } else {
        diffs[mid]
    }
}

/// Apply a fixed millisecond offset to every subtitle line in-place.
pub fn fix_sync_offset(subtitles: &mut Vec<SubtitleLine>, offset_ms: i64) {
    for line in subtitles.iter_mut() {
        let new_start = (line.start_ms as i64 + offset_ms).max(0) as u64;
        let new_end = (line.end_ms as i64 + offset_ms).max(0) as u64;
        line.start_ms = new_start;
        line.end_ms = new_end;
    }
}

/// Validates subtitle timing to ensure readability standards are met.
#[derive(Debug, Clone)]
pub struct SubtitleTimingValidator {
    /// Minimum acceptable subtitle duration in milliseconds.
    pub min_duration_ms: u64,
    /// Maximum acceptable characters per second.
    pub max_chars_per_second: f64,
}

impl SubtitleTimingValidator {
    /// Create a validator with broadcast-standard defaults.
    ///
    /// Defaults: 1 000 ms minimum, 25 cps maximum.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_duration_ms: 1_000,
            max_chars_per_second: 25.0,
        }
    }

    /// Validate all lines, returning a list of `(line_index, error_description)` pairs.
    #[must_use]
    pub fn validate(&self, lines: &[SubtitleLine]) -> Vec<(usize, String)> {
        let mut errors = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            let dur = line.duration_ms();
            if dur < self.min_duration_ms {
                errors.push((
                    i,
                    format!(
                        "Duration {dur} ms is below minimum {} ms",
                        self.min_duration_ms
                    ),
                ));
            }

            let cps = line.reading_speed_cps();
            if cps > self.max_chars_per_second {
                errors.push((
                    i,
                    format!(
                        "Reading speed {cps:.1} cps exceeds maximum {} cps",
                        self.max_chars_per_second
                    ),
                ));
            }

            if line.end_ms <= line.start_ms {
                errors.push((i, "end_ms must be greater than start_ms".to_string()));
            }
        }

        errors
    }
}

impl Default for SubtitleTimingValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line(text: &str, start: u64, end: u64) -> SubtitleLine {
        SubtitleLine::new(text, start, end, 1.0)
    }

    #[test]
    fn test_duration_ms() {
        let line = make_line("hello", 1000, 3000);
        assert_eq!(line.duration_ms(), 2000);
    }

    #[test]
    fn test_duration_ms_zero_when_inverted() {
        let line = make_line("hello", 3000, 1000);
        assert_eq!(line.duration_ms(), 0);
    }

    #[test]
    fn test_word_count() {
        let line = make_line("Hello world foo", 0, 1000);
        assert_eq!(line.word_count(), 3);
    }

    #[test]
    fn test_word_count_empty() {
        let line = make_line("", 0, 1000);
        assert_eq!(line.word_count(), 0);
    }

    #[test]
    fn test_reading_speed_cps() {
        // 10 chars over 2 seconds = 5 cps
        let line = make_line("1234567890", 0, 2000);
        assert!((line.reading_speed_cps() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_reading_speed_cps_zero_duration() {
        let line = make_line("hello", 1000, 1000);
        assert!((line.reading_speed_cps()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_align_subtitle_to_audio_shifts_forward() {
        let lines = vec![make_line("hello", 100, 500)];
        let events = vec![200u64];
        let aligned = align_subtitle_to_audio(&lines, &events);
        assert_eq!(aligned[0].start_ms, 200);
        assert_eq!(aligned[0].end_ms, 600);
    }

    #[test]
    fn test_align_subtitle_no_later_event_unchanged() {
        let lines = vec![make_line("hello", 5000, 6000)];
        let events = vec![100u64, 200]; // all before start_ms
        let aligned = align_subtitle_to_audio(&lines, &events);
        // No event >= 5000, so unchanged
        assert_eq!(aligned[0].start_ms, 5000);
    }

    #[test]
    fn test_detect_sync_offset_positive() {
        let lines = vec![make_line("a", 1000, 2000), make_line("b", 3000, 4000)];
        let refs = vec![1500u64, 3500];
        let offset = detect_sync_offset(&lines, &refs);
        assert_eq!(offset, 500);
    }

    #[test]
    fn test_detect_sync_offset_negative() {
        let lines = vec![make_line("a", 2000, 3000)];
        let refs = vec![1500u64];
        let offset = detect_sync_offset(&lines, &refs);
        assert_eq!(offset, -500);
    }

    #[test]
    fn test_detect_sync_offset_empty() {
        assert_eq!(detect_sync_offset(&[], &[100]), 0);
        assert_eq!(detect_sync_offset(&[make_line("a", 0, 100)], &[]), 0);
    }

    #[test]
    fn test_fix_sync_offset_positive() {
        let mut lines = vec![make_line("a", 1000, 2000)];
        fix_sync_offset(&mut lines, 500);
        assert_eq!(lines[0].start_ms, 1500);
        assert_eq!(lines[0].end_ms, 2500);
    }

    #[test]
    fn test_fix_sync_offset_negative_clamp() {
        let mut lines = vec![make_line("a", 200, 500)];
        fix_sync_offset(&mut lines, -1000); // would go negative
        assert_eq!(lines[0].start_ms, 0);
        assert_eq!(lines[0].end_ms, 0);
    }

    #[test]
    fn test_validator_ok() {
        let validator = SubtitleTimingValidator::new();
        let lines = vec![make_line("Hello world", 0, 2000)];
        assert!(validator.validate(&lines).is_empty());
    }

    #[test]
    fn test_validator_too_short() {
        let validator = SubtitleTimingValidator::new();
        let lines = vec![make_line("Hi", 0, 100)]; // 100 ms < 1000 ms minimum
        let errors = validator.validate(&lines);
        assert!(!errors.is_empty());
        assert_eq!(errors[0].0, 0);
    }

    #[test]
    fn test_validator_too_fast() {
        let validator = SubtitleTimingValidator::new();
        // 100 chars in 1 second = 100 cps, way above 25 limit
        let text = "a".repeat(100);
        let lines = vec![SubtitleLine::new(&text, 0, 1000, 1.0)];
        let errors = validator.validate(&lines);
        assert!(errors.iter().any(|(_, msg)| msg.contains("cps")));
    }

    #[test]
    fn test_validator_end_before_start() {
        let validator = SubtitleTimingValidator::new();
        let lines = vec![make_line("hello", 5000, 1000)];
        let errors = validator.validate(&lines);
        assert!(errors.iter().any(|(_, msg)| msg.contains("end_ms")));
    }

    #[test]
    fn test_subtitle_confidence_clamped() {
        let line = SubtitleLine::new("test", 0, 1000, 1.5);
        assert!((line.confidence - 1.0).abs() < f64::EPSILON);
        let line2 = SubtitleLine::new("test", 0, 1000, -0.5);
        assert!((line2.confidence).abs() < f64::EPSILON);
    }
}

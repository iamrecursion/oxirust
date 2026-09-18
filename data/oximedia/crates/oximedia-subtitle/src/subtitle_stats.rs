#![allow(dead_code)]
//! Statistics computation for subtitle tracks.
//!
//! This module provides functions to compute aggregate metrics about subtitle
//! data: word counts, character counts, average durations, reading speed
//! analysis (words per minute / characters per second), gap analysis, and
//! overall track summaries.

/// Summary statistics for a subtitle track.
#[derive(Clone, Debug, Default)]
pub struct TrackStats {
    /// Total number of subtitle cues.
    pub cue_count: usize,
    /// Total word count across all cues.
    pub total_words: usize,
    /// Total character count (excluding whitespace) across all cues.
    pub total_chars: usize,
    /// Total duration of all cues in milliseconds.
    pub total_duration_ms: i64,
    /// Average cue duration in milliseconds.
    pub avg_duration_ms: f64,
    /// Minimum cue duration in milliseconds.
    pub min_duration_ms: i64,
    /// Maximum cue duration in milliseconds.
    pub max_duration_ms: i64,
    /// Average words per cue.
    pub avg_words_per_cue: f64,
    /// Average characters per second across all cues.
    pub avg_chars_per_second: f64,
    /// Average words per minute across all cues.
    pub avg_words_per_minute: f64,
}

/// A single subtitle entry for statistics computation.
#[derive(Clone, Debug)]
pub struct StatSubtitle {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Subtitle text.
    pub text: String,
}

impl StatSubtitle {
    /// Create a new statistics subtitle entry.
    #[must_use]
    pub fn new(start_ms: i64, end_ms: i64, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }
}

/// Count words in a string (splitting on whitespace).
#[must_use]
pub fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Count non-whitespace characters in a string.
#[must_use]
pub fn char_count(text: &str) -> usize {
    text.chars().filter(|c| !c.is_whitespace()).count()
}

/// Compute characters per second for a single cue.
///
/// Returns 0.0 if the duration is zero or negative.
#[must_use]
pub fn chars_per_second(text: &str, duration_ms: i64) -> f64 {
    if duration_ms <= 0 {
        return 0.0;
    }
    let chars = char_count(text);
    chars as f64 / (duration_ms as f64 / 1000.0)
}

/// Compute words per minute for a single cue.
///
/// Returns 0.0 if the duration is zero or negative.
#[must_use]
pub fn words_per_minute(text: &str, duration_ms: i64) -> f64 {
    if duration_ms <= 0 {
        return 0.0;
    }
    let words = word_count(text);
    words as f64 / (duration_ms as f64 / 60_000.0)
}

/// Compute full track statistics for a slice of subtitle entries.
#[must_use]
pub fn compute_track_stats(subtitles: &[StatSubtitle]) -> TrackStats {
    if subtitles.is_empty() {
        return TrackStats::default();
    }

    let cue_count = subtitles.len();
    let mut total_words = 0usize;
    let mut total_chars = 0usize;
    let mut total_duration_ms = 0i64;
    let mut min_dur = i64::MAX;
    let mut max_dur = i64::MIN;
    let mut weighted_cps_sum = 0.0_f64;
    let mut weighted_wpm_sum = 0.0_f64;

    for sub in subtitles {
        let w = word_count(&sub.text);
        let c = char_count(&sub.text);
        let d = sub.duration_ms();
        total_words += w;
        total_chars += c;
        total_duration_ms += d;
        if d < min_dur {
            min_dur = d;
        }
        if d > max_dur {
            max_dur = d;
        }
        weighted_cps_sum += chars_per_second(&sub.text, d);
        weighted_wpm_sum += words_per_minute(&sub.text, d);
    }

    let avg_duration_ms = total_duration_ms as f64 / cue_count as f64;
    let avg_words_per_cue = total_words as f64 / cue_count as f64;
    let avg_cps = weighted_cps_sum / cue_count as f64;
    let avg_wpm = weighted_wpm_sum / cue_count as f64;

    TrackStats {
        cue_count,
        total_words,
        total_chars,
        total_duration_ms,
        avg_duration_ms,
        min_duration_ms: min_dur,
        max_duration_ms: max_dur,
        avg_words_per_cue,
        avg_chars_per_second: avg_cps,
        avg_words_per_minute: avg_wpm,
    }
}

/// Gap information between consecutive subtitles.
#[derive(Clone, Debug)]
pub struct GapInfo {
    /// Index of the first subtitle (gap is between index and index+1).
    pub before_index: usize,
    /// Gap duration in milliseconds.
    pub gap_ms: i64,
}

/// Compute gaps between consecutive subtitles.
///
/// Returns a list of gaps, including negative gaps (overlaps).
#[must_use]
pub fn compute_gaps(subtitles: &[StatSubtitle]) -> Vec<GapInfo> {
    if subtitles.len() < 2 {
        return Vec::new();
    }
    let mut gaps = Vec::with_capacity(subtitles.len() - 1);
    for i in 0..subtitles.len() - 1 {
        let gap = subtitles[i + 1].start_ms - subtitles[i].end_ms;
        gaps.push(GapInfo {
            before_index: i,
            gap_ms: gap,
        });
    }
    gaps
}

/// Compute the average gap duration between consecutive subtitles.
///
/// Returns 0.0 if there are fewer than 2 subtitles.
#[must_use]
pub fn average_gap(subtitles: &[StatSubtitle]) -> f64 {
    let gaps = compute_gaps(subtitles);
    if gaps.is_empty() {
        return 0.0;
    }
    let total: i64 = gaps.iter().map(|g| g.gap_ms).sum();
    total as f64 / gaps.len() as f64
}

/// Count subtitles whose reading speed exceeds a threshold (CPS).
#[must_use]
pub fn count_fast_cues(subtitles: &[StatSubtitle], max_cps: f64) -> usize {
    subtitles
        .iter()
        .filter(|sub| chars_per_second(&sub.text, sub.duration_ms()) > max_cps)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_subs() -> Vec<StatSubtitle> {
        vec![
            StatSubtitle::new(0, 2000, "Hello world"),
            StatSubtitle::new(2500, 5000, "The quick brown fox jumps"),
            StatSubtitle::new(5500, 7000, "Over the lazy dog"),
        ]
    }

    #[test]
    fn test_word_count_simple() {
        assert_eq!(word_count("Hello world"), 2);
        assert_eq!(word_count("one"), 1);
        assert_eq!(word_count(""), 0);
    }

    #[test]
    fn test_char_count_excludes_whitespace() {
        assert_eq!(char_count("Hello world"), 10);
        assert_eq!(char_count("  a  b  "), 2);
    }

    #[test]
    fn test_chars_per_second() {
        let cps = chars_per_second("Hello", 1000); // 5 chars, 1 second
        assert!((cps - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_chars_per_second_zero_duration() {
        let cps = chars_per_second("Hello", 0);
        assert!((cps - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_words_per_minute() {
        let wpm = words_per_minute("one two three", 60000); // 3 words in 1 minute
        assert!((wpm - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_track_stats_basic() {
        let subs = sample_subs();
        let stats = compute_track_stats(&subs);
        assert_eq!(stats.cue_count, 3);
        assert_eq!(stats.total_words, 11);
        assert_eq!(stats.total_duration_ms, 6000);
        assert_eq!(stats.min_duration_ms, 1500);
        assert_eq!(stats.max_duration_ms, 2500);
    }

    #[test]
    fn test_track_stats_empty() {
        let stats = compute_track_stats(&[]);
        assert_eq!(stats.cue_count, 0);
        assert_eq!(stats.total_words, 0);
    }

    #[test]
    fn test_compute_gaps() {
        let subs = sample_subs();
        let gaps = compute_gaps(&subs);
        assert_eq!(gaps.len(), 2);
        assert_eq!(gaps[0].gap_ms, 500); // 2500 - 2000
        assert_eq!(gaps[1].gap_ms, 500); // 5500 - 5000
    }

    #[test]
    fn test_average_gap() {
        let subs = sample_subs();
        let avg = average_gap(&subs);
        assert!((avg - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_gap_single() {
        let subs = vec![StatSubtitle::new(0, 1000, "only one")];
        let avg = average_gap(&subs);
        assert!((avg - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_count_fast_cues() {
        let subs = vec![
            StatSubtitle::new(0, 1000, "Short"), // 5 chars / 1s = 5 CPS
            StatSubtitle::new(1000, 2000, "This is a longer subtitle"), // 22 chars / 1s = 22 CPS
        ];
        let count = count_fast_cues(&subs, 10.0);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_stat_subtitle_duration() {
        let sub = StatSubtitle::new(1000, 3500, "test");
        assert_eq!(sub.duration_ms(), 2500);
    }

    #[test]
    fn test_track_stats_avg_words_per_cue() {
        let subs = sample_subs();
        let stats = compute_track_stats(&subs);
        // 11 words / 3 cues
        assert!((stats.avg_words_per_cue - 11.0 / 3.0).abs() < 1e-10);
    }
}

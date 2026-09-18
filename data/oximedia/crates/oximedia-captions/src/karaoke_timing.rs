//! Karaoke word-by-word highlight timing for WebVTT cue settings.
//!
//! This module provides structures and utilities for creating karaoke-style
//! captions where each word is highlighted in turn as the audio progresses.
//! The output conforms to WebVTT `<timestamp>` cue notation.

use std::fmt;
use std::time::Duration;

use crate::error::{CaptionError, Result};

// ── Core data types ──────────────────────────────────────────────────────────

/// A single word entry in a karaoke track with precise start/end timing.
#[derive(Debug, Clone, PartialEq)]
pub struct KaraokeEntry {
    /// The word text (may include punctuation attached to word).
    pub word: String,
    /// Time at which this word begins to be highlighted.
    pub start_time: Duration,
    /// Time at which highlighting of this word ends.
    pub end_time: Duration,
}

impl KaraokeEntry {
    /// Create a new karaoke entry.
    #[must_use]
    pub fn new(word: impl Into<String>, start_time: Duration, end_time: Duration) -> Self {
        Self {
            word: word.into(),
            start_time,
            end_time,
        }
    }

    /// Duration of this word's highlight window.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.end_time.saturating_sub(self.start_time)
    }

    /// Format the start time as a WebVTT timestamp string `HH:MM:SS.mmm`.
    #[must_use]
    pub fn start_timestamp(&self) -> String {
        format_vtt_timestamp(self.start_time)
    }

    /// Format the end time as a WebVTT timestamp string.
    #[must_use]
    pub fn end_timestamp(&self) -> String {
        format_vtt_timestamp(self.end_time)
    }
}

/// Format a `Duration` as a WebVTT timestamp `HH:MM:SS.mmm`.
#[must_use]
pub fn format_vtt_timestamp(d: Duration) -> String {
    let total_ms = d.as_millis();
    let ms = total_ms % 1000;
    let total_sec = total_ms / 1000;
    let sec = total_sec % 60;
    let total_min = total_sec / 60;
    let min = total_min % 60;
    let hr = total_min / 60;
    format!("{hr:02}:{min:02}:{sec:02}.{ms:03}")
}

/// Parse a WebVTT timestamp string (`HH:MM:SS.mmm` or `MM:SS.mmm`) into a `Duration`.
///
/// # Errors
/// Returns [`CaptionError::Parse`] if the timestamp format is invalid.
pub fn parse_vtt_timestamp(s: &str) -> Result<Duration> {
    let s = s.trim();
    // Support both HH:MM:SS.mmm and MM:SS.mmm
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    match parts.as_slice() {
        [hh, mm, ss_ms] => {
            let h: u64 = hh
                .parse()
                .map_err(|_| CaptionError::Parse(format!("invalid hours in timestamp: {s}")))?;
            let m: u64 = mm
                .parse()
                .map_err(|_| CaptionError::Parse(format!("invalid minutes in timestamp: {s}")))?;
            let (secs_str, ms_str) = ss_ms.split_once('.').unwrap_or((ss_ms, "0"));
            let sec: u64 = secs_str
                .parse()
                .map_err(|_| CaptionError::Parse(format!("invalid seconds in timestamp: {s}")))?;
            let ms_raw: &str = if ms_str.len() > 3 {
                &ms_str[..3]
            } else {
                ms_str
            };
            let ms_pad = format!("{ms_raw:0<3}"); // left-pad with zeros to 3 digits
            let ms: u64 = ms_pad
                .parse()
                .map_err(|_| CaptionError::Parse(format!("invalid ms in timestamp: {s}")))?;
            let total_ms = h * 3_600_000 + m * 60_000 + sec * 1_000 + ms;
            Ok(Duration::from_millis(total_ms))
        }
        [mm, ss_ms] => {
            let m: u64 = mm
                .parse()
                .map_err(|_| CaptionError::Parse(format!("invalid minutes in timestamp: {s}")))?;
            let (secs_str, ms_str) = ss_ms.split_once('.').unwrap_or((ss_ms, "0"));
            let sec: u64 = secs_str
                .parse()
                .map_err(|_| CaptionError::Parse(format!("invalid seconds in timestamp: {s}")))?;
            let ms_raw: &str = if ms_str.len() > 3 {
                &ms_str[..3]
            } else {
                ms_str
            };
            let ms_pad = format!("{ms_raw:0<3}");
            let ms: u64 = ms_pad
                .parse()
                .map_err(|_| CaptionError::Parse(format!("invalid ms in timestamp: {s}")))?;
            let total_ms = m * 60_000 + sec * 1_000 + ms;
            Ok(Duration::from_millis(total_ms))
        }
        _ => Err(CaptionError::Parse(format!(
            "unrecognized timestamp format: {s}"
        ))),
    }
}

// ── KaraokeTrack ─────────────────────────────────────────────────────────────

/// A full karaoke track containing ordered word entries.
#[derive(Debug, Clone, Default)]
pub struct KaraokeTrack {
    /// Ordered list of word entries.
    pub entries: Vec<KaraokeEntry>,
}

impl KaraokeTrack {
    /// Create an empty karaoke track.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a word entry to the track.
    pub fn push(&mut self, entry: KaraokeEntry) {
        self.entries.push(entry);
    }

    /// Number of word entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the track has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Overall start time (start of first entry), or zero if empty.
    #[must_use]
    pub fn start_time(&self) -> Duration {
        self.entries
            .first()
            .map(|e| e.start_time)
            .unwrap_or(Duration::ZERO)
    }

    /// Overall end time (end of last entry), or zero if empty.
    #[must_use]
    pub fn end_time(&self) -> Duration {
        self.entries
            .last()
            .map(|e| e.end_time)
            .unwrap_or(Duration::ZERO)
    }

    /// Generate a WebVTT cue body with inline `<HH:MM:SS.mmm><word>` notation.
    ///
    /// Each word is preceded by a timestamp tag marking when it should be highlighted.
    /// This conforms to the WebVTT karaoke timing extension (Section 6.1 of the spec).
    ///
    /// # Example output
    /// ```text
    /// <00:00:00.500>Hello <00:00:01.000>world
    /// ```
    #[must_use]
    pub fn to_webvtt_cue_body(&self) -> String {
        self.entries
            .iter()
            .map(|e| format!("<{}>{}", e.start_timestamp(), e.word))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Generate complete WebVTT cue settings header line.
    ///
    /// Returns a string suitable for use as the cue settings line in a `.vtt` file,
    /// formatted as `<start> --> <end>`.
    #[must_use]
    pub fn to_webvtt_cue_settings(&self) -> String {
        format!(
            "{} --> {}",
            format_vtt_timestamp(self.start_time()),
            format_vtt_timestamp(self.end_time())
        )
    }

    /// Render a complete WebVTT cue block (header + karaoke body).
    ///
    /// # Parameters
    /// * `cue_id` – optional cue identifier (e.g. `"1"`)
    #[must_use]
    pub fn to_webvtt_cue(&self, cue_id: Option<&str>) -> String {
        let mut lines = Vec::new();
        if let Some(id) = cue_id {
            lines.push(id.to_string());
        }
        lines.push(self.to_webvtt_cue_settings());
        lines.push(self.to_webvtt_cue_body());
        lines.join("\n")
    }

    /// Merge adjacent entries whose gap is less than `threshold` into the same word token.
    ///
    /// This is useful when automated alignment produces very short silence breaks between
    /// phonemes of the same word.
    pub fn merge_short_gaps(&mut self, threshold: Duration) {
        if self.entries.len() < 2 {
            return;
        }
        let mut merged: Vec<KaraokeEntry> = Vec::with_capacity(self.entries.len());
        for entry in self.entries.drain(..) {
            match merged.last_mut() {
                Some(last) => {
                    let gap = entry.start_time.saturating_sub(last.end_time);
                    if gap < threshold {
                        last.word.push(' ');
                        last.word.push_str(&entry.word);
                        last.end_time = entry.end_time;
                    } else {
                        merged.push(entry);
                    }
                }
                None => merged.push(entry),
            }
        }
        self.entries = merged;
    }

    /// Validate that all entries have non-zero duration and non-overlapping windows.
    ///
    /// # Errors
    /// Returns a list of error messages describing any validation failures.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let mut prev_end = Duration::ZERO;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.end_time <= entry.start_time {
                errors.push(format!(
                    "entry {i} ('{}') has zero or negative duration",
                    entry.word
                ));
            }
            if entry.start_time < prev_end {
                errors.push(format!(
                    "entry {i} ('{}') overlaps with previous entry (prev_end={:?}, start={:?})",
                    entry.word, prev_end, entry.start_time
                ));
            }
            prev_end = entry.end_time;
        }
        errors
    }
}

// ── KaraokeAligner ───────────────────────────────────────────────────────────

/// Strategy for distributing timing across words when exact timestamps are unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentStrategy {
    /// Distribute time uniformly: each word receives the same duration.
    Uniform,
    /// Proportional: longer words (by character count) receive proportionally more time.
    Proportional,
    /// Syllable-weighted: estimate syllable count and weight by syllables.
    SyllableWeighted,
}

impl Default for AlignmentStrategy {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Aligns a list of words to a given time window using a chosen strategy.
///
/// # Example
/// ```rust
/// use std::time::Duration;
/// use oximedia_captions::karaoke_timing::{KaraokeAligner, AlignmentStrategy};
///
/// let aligner = KaraokeAligner::new(AlignmentStrategy::Uniform);
/// let words = vec!["Hello".to_string(), "world".to_string()];
/// let track = aligner.align(
///     &words,
///     Duration::from_secs(0),
///     Duration::from_secs(2),
/// ).unwrap();
/// assert_eq!(track.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct KaraokeAligner {
    /// Alignment strategy to use.
    pub strategy: AlignmentStrategy,
    /// Minimum duration per word in milliseconds (prevents zero-duration entries).
    pub min_word_ms: u64,
}

impl Default for KaraokeAligner {
    fn default() -> Self {
        Self {
            strategy: AlignmentStrategy::Uniform,
            min_word_ms: 100,
        }
    }
}

impl KaraokeAligner {
    /// Create a new aligner with the specified strategy.
    #[must_use]
    pub fn new(strategy: AlignmentStrategy) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }

    /// Align `words` across `[start, end)` using the chosen strategy.
    ///
    /// # Errors
    /// Returns [`CaptionError`] if the word list is empty or the time range is invalid.
    pub fn align(&self, words: &[String], start: Duration, end: Duration) -> Result<KaraokeTrack> {
        if words.is_empty() {
            return Err(CaptionError::Parse(
                "cannot align empty word list".to_string(),
            ));
        }
        if end <= start {
            return Err(CaptionError::Parse(format!(
                "invalid time range: start={start:?} >= end={end:?}"
            )));
        }

        let weights = self.compute_weights(words);
        let total_ms = end.as_millis().saturating_sub(start.as_millis()) as u64;
        let weight_sum: f64 = weights.iter().sum();

        let mut track = KaraokeTrack::new();
        let mut cursor_ms = start.as_millis() as u64;

        for (i, word) in words.iter().enumerate() {
            let frac = if weight_sum > 0.0 {
                weights[i] / weight_sum
            } else {
                1.0 / words.len() as f64
            };
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let word_ms = ((total_ms as f64 * frac).round() as u64).max(self.min_word_ms);

            let word_start = Duration::from_millis(cursor_ms);
            let word_end = Duration::from_millis((cursor_ms + word_ms).min(end.as_millis() as u64));
            track.push(KaraokeEntry::new(word, word_start, word_end));
            cursor_ms += word_ms;
        }

        // Clamp the last entry end time to track end
        if let Some(last) = track.entries.last_mut() {
            if last.end_time > end {
                last.end_time = end;
            }
        }

        Ok(track)
    }

    /// Compute per-word weights according to the chosen strategy.
    fn compute_weights(&self, words: &[String]) -> Vec<f64> {
        match self.strategy {
            AlignmentStrategy::Uniform => vec![1.0; words.len()],
            AlignmentStrategy::Proportional => words
                .iter()
                .map(|w| (w.chars().count() as f64).max(1.0))
                .collect(),
            AlignmentStrategy::SyllableWeighted => words
                .iter()
                .map(|w| (estimate_syllables(w) as f64).max(1.0))
                .collect(),
        }
    }
}

/// Estimate syllable count for English text using vowel-cluster heuristic.
///
/// This is a simple approximation — not a linguistic model.
#[must_use]
fn estimate_syllables(word: &str) -> usize {
    if word.is_empty() {
        return 1;
    }
    let lower = word.to_lowercase();
    let vowels = ['a', 'e', 'i', 'o', 'u'];
    let mut count = 0usize;
    let mut prev_vowel = false;
    for ch in lower.chars() {
        let is_vowel = vowels.contains(&ch);
        if is_vowel && !prev_vowel {
            count += 1;
        }
        prev_vowel = is_vowel;
    }
    // Trailing silent 'e' adjustment
    if lower.ends_with('e') && count > 1 {
        count -= 1;
    }
    count.max(1)
}

// ── Display ──────────────────────────────────────────────────────────────────

impl fmt::Display for KaraokeTrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_webvtt_cue_body())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    // ── KaraokeEntry tests ───────────────────────────────────────────────────

    #[test]
    fn test_karaoke_entry_duration() {
        let e = KaraokeEntry::new("hello", ms(500), ms(1500));
        assert_eq!(e.duration(), ms(1000));
    }

    #[test]
    fn test_karaoke_entry_start_timestamp_format() {
        let e = KaraokeEntry::new("hi", ms(500), ms(1000));
        assert_eq!(e.start_timestamp(), "00:00:00.500");
    }

    #[test]
    fn test_karaoke_entry_end_timestamp_format() {
        let e = KaraokeEntry::new("hi", ms(0), ms(3_723_456));
        // 3_723_456 ms = 1h 2m 3s 456ms
        assert_eq!(e.end_timestamp(), "01:02:03.456");
    }

    // ── format / parse round-trip ────────────────────────────────────────────

    #[test]
    fn test_format_parse_vtt_timestamp_roundtrip() {
        let original = ms(12_345_678);
        let formatted = format_vtt_timestamp(original);
        let parsed = parse_vtt_timestamp(&formatted).expect("should parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_parse_vtt_timestamp_mm_ss() {
        let d = parse_vtt_timestamp("02:30.500").expect("should parse MM:SS.mmm");
        assert_eq!(d.as_millis(), 150_500);
    }

    #[test]
    fn test_parse_vtt_timestamp_invalid() {
        assert!(parse_vtt_timestamp("not_a_timestamp").is_err());
    }

    // ── KaraokeTrack tests ───────────────────────────────────────────────────

    #[test]
    fn test_karaoke_track_cue_body() {
        let mut track = KaraokeTrack::new();
        track.push(KaraokeEntry::new("Hello", ms(0), ms(500)));
        track.push(KaraokeEntry::new("world", ms(500), ms(1000)));
        let body = track.to_webvtt_cue_body();
        assert!(body.contains("<00:00:00.000>Hello"));
        assert!(body.contains("<00:00:00.500>world"));
    }

    #[test]
    fn test_karaoke_track_cue_settings_header() {
        let mut track = KaraokeTrack::new();
        track.push(KaraokeEntry::new("A", ms(1000), ms(2000)));
        track.push(KaraokeEntry::new("B", ms(2000), ms(3000)));
        let header = track.to_webvtt_cue_settings();
        assert_eq!(header, "00:00:01.000 --> 00:00:03.000");
    }

    #[test]
    fn test_karaoke_track_cue_with_id() {
        let mut track = KaraokeTrack::new();
        track.push(KaraokeEntry::new("word", ms(0), ms(1000)));
        let cue = track.to_webvtt_cue(Some("cue-1"));
        assert!(cue.starts_with("cue-1\n"));
    }

    #[test]
    fn test_karaoke_track_validation_overlap() {
        let mut track = KaraokeTrack::new();
        track.push(KaraokeEntry::new("A", ms(0), ms(1000)));
        track.push(KaraokeEntry::new("B", ms(500), ms(1500))); // overlaps with A
        let errors = track.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("overlaps"));
    }

    #[test]
    fn test_karaoke_track_validation_zero_duration() {
        let mut track = KaraokeTrack::new();
        track.push(KaraokeEntry::new("A", ms(500), ms(500))); // zero duration
        let errors = track.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("zero"));
    }

    #[test]
    fn test_karaoke_track_validation_ok() {
        let mut track = KaraokeTrack::new();
        track.push(KaraokeEntry::new("Hello", ms(0), ms(500)));
        track.push(KaraokeEntry::new("world", ms(500), ms(1000)));
        assert!(track.validate().is_empty());
    }

    // ── KaraokeAligner tests ─────────────────────────────────────────────────

    #[test]
    fn test_aligner_uniform_word_count() {
        let aligner = KaraokeAligner::new(AlignmentStrategy::Uniform);
        let words = vec![
            "Hello".to_string(),
            "beautiful".to_string(),
            "world".to_string(),
        ];
        let track = aligner.align(&words, ms(0), ms(3000)).expect("align ok");
        assert_eq!(track.len(), 3);
    }

    #[test]
    fn test_aligner_uniform_covers_range() {
        let aligner = KaraokeAligner::new(AlignmentStrategy::Uniform);
        let words: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let track = aligner.align(&words, ms(1000), ms(4000)).expect("align ok");
        assert_eq!(track.start_time(), ms(1000));
        // Last entry should end at or before end
        assert!(track.end_time() <= ms(4000) + ms(10)); // allow tiny rounding
    }

    #[test]
    fn test_aligner_proportional() {
        let aligner = KaraokeAligner::new(AlignmentStrategy::Proportional);
        let words = vec!["I".to_string(), "extraordinary".to_string()];
        let track = aligner.align(&words, ms(0), ms(1400)).expect("align ok");
        // "extraordinary" is much longer, should get more time
        let w0_dur = track.entries[0].duration().as_millis();
        let w1_dur = track.entries[1].duration().as_millis();
        assert!(
            w1_dur > w0_dur,
            "longer word should get more time in proportional mode"
        );
    }

    #[test]
    fn test_aligner_empty_words_error() {
        let aligner = KaraokeAligner::default();
        let result = aligner.align(&[], ms(0), ms(1000));
        assert!(result.is_err());
    }

    #[test]
    fn test_aligner_invalid_range_error() {
        let aligner = KaraokeAligner::default();
        let words = vec!["hello".to_string()];
        let result = aligner.align(&words, ms(2000), ms(1000));
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_short_gaps() {
        let mut track = KaraokeTrack::new();
        // gap of 10ms between "Don't" and "know" — less than threshold 50ms
        track.push(KaraokeEntry::new("Don't", ms(0), ms(490)));
        track.push(KaraokeEntry::new("know", ms(500), ms(1000)));
        track.merge_short_gaps(ms(50));
        // Should be merged into one entry
        assert_eq!(track.len(), 1);
        assert!(track.entries[0].word.contains("Don't"));
    }

    #[test]
    fn test_estimate_syllables_heuristic() {
        // "hello" → h-e-ll-o → 2 syllable clusters: 'e' and 'o'
        assert!(estimate_syllables("hello") >= 2);
        // "a" → 1 syllable
        assert_eq!(estimate_syllables("a"), 1);
        // "extraordinary" → many vowels
        assert!(estimate_syllables("extraordinary") >= 4);
    }
}

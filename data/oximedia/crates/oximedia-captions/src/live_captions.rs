//! Live captioning pipeline.
//!
//! Provides types and utilities for ingesting real-time speech recognition
//! output and assembling caption segments with latency monitoring.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Configuration for a live captioning pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveCaptionConfig {
    /// Target maximum end-to-end latency, in milliseconds.
    pub latency_target_ms: u32,
    /// Minimum confidence threshold (0.0–1.0) for accepting words.
    pub confidence_threshold: f32,
    /// Maximum number of words to buffer before flushing.
    pub max_buffer_words: u32,
    /// BCP-47 language tag (e.g. `"en-US"`).
    pub language: String,
}

impl LiveCaptionConfig {
    /// Create a config with sensible defaults for English live captioning.
    #[must_use]
    pub fn default_english() -> Self {
        Self {
            latency_target_ms: 3000,
            confidence_threshold: 0.75,
            max_buffer_words: 20,
            language: "en-US".to_string(),
        }
    }
}

impl Default for LiveCaptionConfig {
    fn default() -> Self {
        Self::default_english()
    }
}

/// A single recognised word from an ASR engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionWord {
    /// The word text.
    pub word: String,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Start time of the word, in milliseconds since stream start.
    pub start_ms: u64,
    /// End time of the word, in milliseconds since stream start.
    pub end_ms: u64,
    /// Optional speaker identifier for diarised streams.
    pub speaker_id: Option<u32>,
}

impl CaptionWord {
    /// Create a new caption word.
    #[must_use]
    pub fn new(
        word: impl Into<String>,
        confidence: f32,
        start_ms: u64,
        end_ms: u64,
        speaker_id: Option<u32>,
    ) -> Self {
        Self {
            word: word.into(),
            confidence: confidence.clamp(0.0, 1.0),
            start_ms,
            end_ms,
            speaker_id,
        }
    }
}

/// A segment of caption text, formed from one or more words.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionSegment {
    /// The words that make up this segment.
    pub words: Vec<CaptionWord>,
    /// Full text of the segment (words joined with spaces).
    pub text: String,
    /// Start time of the segment, in milliseconds.
    pub start_ms: u64,
    /// End time of the segment, in milliseconds.
    pub end_ms: u64,
    /// Whether the ASR engine has confirmed this segment is final.
    pub is_final: bool,
}

impl CaptionSegment {
    /// Construct a segment from a list of words.
    ///
    /// The `text`, `start_ms`, and `end_ms` fields are derived automatically.
    /// Returns `None` if `words` is empty.
    #[must_use]
    pub fn from_words(words: Vec<CaptionWord>) -> Option<Self> {
        if words.is_empty() {
            return None;
        }
        let start_ms = words.first().map_or(0, |w| w.start_ms);
        let end_ms = words.last().map_or(0, |w| w.end_ms);
        let text = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Some(Self {
            words,
            text,
            start_ms,
            end_ms,
            is_final: false,
        })
    }

    /// Mark this segment as final.
    #[must_use]
    pub fn finalized(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Duration of this segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// A rolling buffer that accumulates words and flushes caption segments when a
/// sentence boundary is detected.
#[derive(Debug, Default)]
pub struct LiveCaptionBuffer {
    pending: Vec<CaptionWord>,
}

impl LiveCaptionBuffer {
    /// Create a new empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a word to the buffer.
    pub fn add_word(&mut self, word: CaptionWord) {
        self.pending.push(word);
    }

    /// Attempt to flush a ready segment.
    ///
    /// A segment is considered ready when the last word in the buffer meets
    /// `confidence_min` AND the last word ends a sentence (detected by
    /// `SentenceBoundaryDetector`).
    ///
    /// Returns `Some(CaptionSegment)` and clears the buffer on success.
    pub fn flush_ready(&mut self, confidence_min: f32) -> Option<CaptionSegment> {
        if self.pending.is_empty() {
            return None;
        }

        // Check the last word
        let last = self.pending.last()?;
        let prev = if self.pending.len() >= 2 {
            &self.pending[self.pending.len() - 2].word.clone()
        } else {
            ""
        };

        let at_boundary = SentenceBoundaryDetector::is_boundary(&last.word, prev);
        let high_confidence = last.confidence >= confidence_min;

        if at_boundary && high_confidence {
            let words: Vec<CaptionWord> = self.pending.drain(..).collect();
            CaptionSegment::from_words(words).map(CaptionSegment::finalized)
        } else {
            None
        }
    }

    /// Return the current text of all buffered words joined with spaces.
    #[must_use]
    pub fn current_text(&self) -> String {
        self.pending
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Number of words currently in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Returns `true` if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Force-flush all pending words as a (possibly non-final) segment,
    /// regardless of whether a sentence boundary has been detected.
    pub fn flush_all(&mut self) -> Option<CaptionSegment> {
        if self.pending.is_empty() {
            return None;
        }
        let words: Vec<CaptionWord> = self.pending.drain(..).collect();
        CaptionSegment::from_words(words)
    }
}

/// Detects sentence boundaries based on punctuation.
pub struct SentenceBoundaryDetector;

impl SentenceBoundaryDetector {
    /// Returns `true` if `word` (after `prev_word`) represents a sentence
    /// boundary.
    ///
    /// A boundary is detected when `word` ends with `.`, `!`, or `?`, or when
    /// `prev_word` ends with those characters.
    #[must_use]
    pub fn is_boundary(word: &str, prev_word: &str) -> bool {
        let ends_sentence = |w: &str| {
            w.ends_with('.') || w.ends_with('!') || w.ends_with('?') || w.ends_with("...")
        };
        ends_sentence(word) || ends_sentence(prev_word)
    }
}

/// Tracks end-to-end captioning latency and computes statistics.
#[derive(Debug, Clone)]
pub struct LatencyMonitor {
    /// Target latency in milliseconds.
    pub target_ms: u32,
    /// Recorded latency measurements.
    pub measurements: Vec<u64>,
}

impl LatencyMonitor {
    /// Create a new monitor with the given target.
    #[must_use]
    pub fn new(target_ms: u32) -> Self {
        Self {
            target_ms,
            measurements: Vec::new(),
        }
    }

    /// Record a latency measurement.
    pub fn add(&mut self, latency_ms: u64) {
        self.measurements.push(latency_ms);
    }

    /// Return the 95th-percentile latency across all recorded measurements.
    ///
    /// Uses the nearest-rank method: returns the value at rank
    /// `ceil(0.95 * N)` (1-based), i.e. index `ceil(0.95 * N) - 1`.
    ///
    /// Returns `0` if no measurements have been recorded.
    #[must_use]
    pub fn p95_ms(&self) -> u64 {
        if self.measurements.is_empty() {
            return 0;
        }
        let mut sorted = self.measurements.clone();
        sorted.sort_unstable();
        let n = sorted.len() as f64;
        // Nearest-rank: rank = ceil(0.95 * N), index = rank - 1
        let rank = (0.95 * n).ceil() as usize;
        let idx = rank.saturating_sub(1).min(sorted.len() - 1);
        sorted[idx]
    }

    /// Returns `true` if the p95 latency is at or below the target.
    #[must_use]
    pub fn is_meeting_target(&self) -> bool {
        self.p95_ms() <= u64::from(self.target_ms)
    }

    /// Returns the number of recorded measurements.
    #[must_use]
    pub fn count(&self) -> usize {
        self.measurements.len()
    }
}

// ── Types merged from live_caption module ────────────────────────────────────

/// Indicates the stability level of a recognised word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveCaptionMode {
    /// A word-granularity hypothesis that may still change.
    Word,
    /// An in-progress utterance that is not yet final.
    Partial,
    /// A finalised, committed recognition result.
    Final,
}

impl LiveCaptionMode {
    /// Returns `true` for results that are committed and will not be revised.
    #[must_use]
    pub fn is_committed(&self) -> bool {
        matches!(self, Self::Final)
    }
}

/// A single recognised word from a live caption stream.
///
/// This is a lightweight word representation carrying a [`LiveCaptionMode`]
/// stability indicator, complementing [`CaptionWord`] which carries richer
/// timing and speaker metadata.
#[derive(Debug, Clone)]
pub struct LiveWord {
    /// The recognised text of the word.
    pub text: String,
    /// Timestamp when recognition started for this word, in milliseconds.
    pub start_ms: u64,
    /// Confidence score in the range `[0.0, 1.0]`.
    pub confidence: f32,
    /// Stability mode of this word.
    pub mode: LiveCaptionMode,
}

impl LiveWord {
    /// Create a new live word.
    pub fn new(
        text: impl Into<String>,
        start_ms: u64,
        confidence: f32,
        mode: LiveCaptionMode,
    ) -> Self {
        Self {
            text: text.into(),
            start_ms,
            confidence,
            mode,
        }
    }

    /// Returns `true` when the confidence score meets or exceeds `threshold`.
    #[must_use]
    pub fn is_reliable(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// A buffer that accumulates incoming [`LiveWord`]s and manages committed text.
///
/// Words older than `max_latency_ms` from a reference time are flushed to
/// keep the buffer from growing indefinitely.  This is a simpler, latency-aware
/// alternative to [`LiveCaptionBuffer`] which uses sentence-boundary detection.
#[derive(Debug)]
pub struct StreamingCaptionBuffer {
    /// Words currently held in the buffer (partial or word-mode).
    pub words: Vec<LiveWord>,
    /// Text that has been committed (finalised) and will not change.
    pub committed_text: String,
    /// Maximum age (in ms) that a word may remain in the buffer before flush.
    pub max_latency_ms: u32,
}

impl StreamingCaptionBuffer {
    /// Create a new, empty buffer.
    #[must_use]
    pub fn new(max_latency_ms: u32) -> Self {
        Self {
            words: Vec::new(),
            committed_text: String::new(),
            max_latency_ms,
        }
    }

    /// Add a word to the buffer.
    pub fn add_word(&mut self, word: LiveWord) {
        self.words.push(word);
    }

    /// Commit all words currently in the buffer to `committed_text` and clear
    /// the word buffer.
    pub fn commit(&mut self) {
        for word in self.words.drain(..) {
            if !self.committed_text.is_empty() {
                self.committed_text.push(' ');
            }
            self.committed_text.push_str(&word.text);
        }
    }

    /// Returns the partial (in-progress) text from unbuffered words only,
    /// without modifying the buffer.
    #[must_use]
    pub fn partial_text(&self) -> String {
        self.words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Number of words that have been committed to `committed_text`.
    ///
    /// Counts space-separated tokens in `committed_text`, returning 0 for an
    /// empty string.
    #[must_use]
    pub fn total_committed_words(&self) -> usize {
        if self.committed_text.is_empty() {
            0
        } else {
            self.committed_text.split_whitespace().count()
        }
    }

    /// Remove words whose `start_ms` is older than `now_ms - max_latency_ms`.
    ///
    /// Removed words are committed to `committed_text` in arrival order before
    /// being dropped, so no recognised content is lost.
    pub fn flush_old(&mut self, now_ms: u64) {
        let cutoff = now_ms.saturating_sub(u64::from(self.max_latency_ms));
        let mut to_commit: Vec<LiveWord> = Vec::new();
        self.words.retain(|w| {
            if w.start_ms < cutoff {
                to_commit.push(w.clone());
                false
            } else {
                true
            }
        });
        // Commit in the order they were retained (stable by insertion order).
        for word in to_commit {
            if !self.committed_text.is_empty() {
                self.committed_text.push(' ');
            }
            self.committed_text.push_str(&word.text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_word(text: &str, confidence: f32, start_ms: u64, end_ms: u64) -> CaptionWord {
        CaptionWord::new(text, confidence, start_ms, end_ms, None)
    }

    fn live_word(text: &str, start_ms: u64, conf: f32, mode: LiveCaptionMode) -> LiveWord {
        LiveWord::new(text, start_ms, conf, mode)
    }

    #[test]
    fn test_caption_config_defaults() {
        let config = LiveCaptionConfig::default_english();
        assert_eq!(config.language, "en-US");
        assert!(config.confidence_threshold > 0.0);
    }

    #[test]
    fn test_caption_word_confidence_clamped() {
        let w = CaptionWord::new("hello", 1.5, 0, 100, None);
        assert_eq!(w.confidence, 1.0);
        let w2 = CaptionWord::new("world", -0.5, 0, 100, None);
        assert_eq!(w2.confidence, 0.0);
    }

    #[test]
    fn test_segment_from_words() {
        let words = vec![
            make_word("Hello", 0.95, 0, 500),
            make_word("world.", 0.90, 500, 1000),
        ];
        let seg =
            CaptionSegment::from_words(words).expect("caption segment creation should succeed");
        assert_eq!(seg.text, "Hello world.");
        assert_eq!(seg.start_ms, 0);
        assert_eq!(seg.end_ms, 1000);
    }

    #[test]
    fn test_segment_from_empty_words() {
        let seg = CaptionSegment::from_words(vec![]);
        assert!(seg.is_none());
    }

    #[test]
    fn test_segment_duration() {
        let words = vec![make_word("Test.", 0.9, 1000, 2500)];
        let seg =
            CaptionSegment::from_words(words).expect("caption segment creation should succeed");
        assert_eq!(seg.duration_ms(), 1500);
    }

    #[test]
    fn test_buffer_add_and_current_text() {
        let mut buf = LiveCaptionBuffer::new();
        buf.add_word(make_word("Hello", 0.9, 0, 300));
        buf.add_word(make_word("there.", 0.85, 300, 600));
        assert_eq!(buf.current_text(), "Hello there.");
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_buffer_flush_on_sentence_boundary() {
        let mut buf = LiveCaptionBuffer::new();
        buf.add_word(make_word("Hello", 0.95, 0, 300));
        buf.add_word(make_word("world.", 0.92, 300, 600));
        let seg = buf.flush_ready(0.80);
        assert!(seg.is_some());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_no_flush_without_boundary() {
        let mut buf = LiveCaptionBuffer::new();
        buf.add_word(make_word("Hello", 0.95, 0, 300));
        buf.add_word(make_word("world", 0.92, 300, 600)); // no period
        let seg = buf.flush_ready(0.80);
        assert!(seg.is_none());
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_flush_all() {
        let mut buf = LiveCaptionBuffer::new();
        buf.add_word(make_word("Testing", 0.9, 0, 400));
        let seg = buf.flush_all();
        assert!(seg.is_some());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_sentence_boundary_period() {
        assert!(SentenceBoundaryDetector::is_boundary("world.", "hello"));
        assert!(!SentenceBoundaryDetector::is_boundary("world", "hello"));
    }

    #[test]
    fn test_sentence_boundary_exclamation() {
        assert!(SentenceBoundaryDetector::is_boundary("stop!", "please"));
    }

    #[test]
    fn test_sentence_boundary_question() {
        assert!(SentenceBoundaryDetector::is_boundary("ready?", "are"));
    }

    #[test]
    fn test_latency_monitor_p95() {
        let mut monitor = LatencyMonitor::new(3000);
        for i in 1..=20u64 {
            monitor.add(i * 100);
        }
        // p95 of [100,200,...,2000] → index 19 (95% of 20 = 19) → 1900
        assert_eq!(monitor.p95_ms(), 1900);
    }

    #[test]
    fn test_latency_monitor_empty() {
        let monitor = LatencyMonitor::new(3000);
        assert_eq!(monitor.p95_ms(), 0);
        assert!(monitor.is_meeting_target());
    }

    #[test]
    fn test_latency_monitor_meeting_target() {
        let mut monitor = LatencyMonitor::new(3000);
        for _ in 0..100 {
            monitor.add(1500);
        }
        assert!(monitor.is_meeting_target());
    }

    #[test]
    fn test_latency_monitor_failing_target() {
        let mut monitor = LatencyMonitor::new(1000);
        for _ in 0..100 {
            monitor.add(5000);
        }
        assert!(!monitor.is_meeting_target());
    }

    // ── LiveCaptionMode tests (merged from live_caption) ──

    #[test]
    fn test_mode_final_is_committed() {
        assert!(LiveCaptionMode::Final.is_committed());
    }

    #[test]
    fn test_mode_partial_not_committed() {
        assert!(!LiveCaptionMode::Partial.is_committed());
    }

    #[test]
    fn test_mode_word_not_committed() {
        assert!(!LiveCaptionMode::Word.is_committed());
    }

    // ── LiveWord tests (merged from live_caption) ──

    #[test]
    fn test_live_word_is_reliable_above_threshold() {
        let w = live_word("hello", 0, 0.9, LiveCaptionMode::Final);
        assert!(w.is_reliable(0.8));
    }

    #[test]
    fn test_live_word_is_reliable_below_threshold() {
        let w = live_word("hmm", 0, 0.4, LiveCaptionMode::Partial);
        assert!(!w.is_reliable(0.7));
    }

    #[test]
    fn test_live_word_is_reliable_at_exact_threshold() {
        let w = live_word("ok", 0, 0.5, LiveCaptionMode::Word);
        assert!(w.is_reliable(0.5));
    }

    // ── StreamingCaptionBuffer tests (merged from live_caption) ──

    #[test]
    fn test_streaming_buffer_add_and_partial_text() {
        let mut buf = StreamingCaptionBuffer::new(1000);
        buf.add_word(live_word("Hello", 100, 0.9, LiveCaptionMode::Word));
        buf.add_word(live_word("world", 200, 0.85, LiveCaptionMode::Word));
        assert_eq!(buf.partial_text(), "Hello world");
    }

    #[test]
    fn test_streaming_buffer_commit_clears_words() {
        let mut buf = StreamingCaptionBuffer::new(1000);
        buf.add_word(live_word("Hello", 0, 1.0, LiveCaptionMode::Final));
        buf.commit();
        assert!(buf.words.is_empty());
        assert_eq!(buf.committed_text, "Hello");
    }

    #[test]
    fn test_streaming_buffer_total_committed_words_empty() {
        let buf = StreamingCaptionBuffer::new(500);
        assert_eq!(buf.total_committed_words(), 0);
    }

    #[test]
    fn test_streaming_buffer_total_committed_words_after_commit() {
        let mut buf = StreamingCaptionBuffer::new(500);
        buf.add_word(live_word("one", 0, 0.9, LiveCaptionMode::Final));
        buf.add_word(live_word("two", 100, 0.9, LiveCaptionMode::Final));
        buf.commit();
        assert_eq!(buf.total_committed_words(), 2);
    }

    #[test]
    fn test_streaming_buffer_flush_old_commits_stale_words() {
        let mut buf = StreamingCaptionBuffer::new(500);
        buf.add_word(live_word("stale", 0, 0.9, LiveCaptionMode::Word));
        buf.add_word(live_word("fresh", 600, 0.9, LiveCaptionMode::Word));
        buf.flush_old(1000);
        assert_eq!(buf.committed_text, "stale");
        assert_eq!(buf.words.len(), 1);
        assert_eq!(buf.words[0].text, "fresh");
    }

    #[test]
    fn test_streaming_buffer_flush_old_keeps_recent_words() {
        let mut buf = StreamingCaptionBuffer::new(2000);
        buf.add_word(live_word("new", 900, 1.0, LiveCaptionMode::Word));
        buf.flush_old(1000);
        assert_eq!(buf.words.len(), 1);
        assert!(buf.committed_text.is_empty());
    }

    #[test]
    fn test_streaming_buffer_multiple_commits_accumulate() {
        let mut buf = StreamingCaptionBuffer::new(1000);
        buf.add_word(live_word("first", 0, 1.0, LiveCaptionMode::Final));
        buf.commit();
        buf.add_word(live_word("second", 100, 1.0, LiveCaptionMode::Final));
        buf.commit();
        assert_eq!(buf.committed_text, "first second");
        assert_eq!(buf.total_committed_words(), 2);
    }

    #[test]
    fn test_streaming_buffer_partial_text_empty_when_no_words() {
        let buf = StreamingCaptionBuffer::new(1000);
        assert_eq!(buf.partial_text(), "");
    }
}

/// Stress tests for live captioning under high-frequency update loads.
///
/// These tests verify that the streaming infrastructure does not panic,
/// corrupt state, or lose words when ingesting 100+ caption updates per second.
#[cfg(test)]
mod live_caption_stress_tests {
    use super::*;

    /// Feed 100 words at 10 ms intervals (100 Hz) into a `StreamingCaptionBuffer`
    /// and verify that no words are lost and no panics occur.
    #[test]
    fn test_streaming_buffer_high_frequency_updates() {
        let max_latency_ms = 500;
        let mut buf = StreamingCaptionBuffer::new(max_latency_ms);

        let total_updates = 100usize;
        for i in 0..total_updates {
            let word_text = format!("word{i}");
            let start_ms = (i as u64) * 10; // 10 ms apart = 100 Hz
            buf.add_word(LiveWord::new(
                word_text,
                start_ms,
                0.95_f32,
                LiveCaptionMode::Word,
            ));

            // Every 20 words, flush old entries as if 200 ms has elapsed.
            if i % 20 == 19 {
                buf.flush_old(start_ms + u64::from(max_latency_ms) + 1);
            }
        }

        // Drain remaining buffered words.
        buf.commit();

        let committed_words = buf.total_committed_words();
        assert_eq!(
            committed_words, total_updates,
            "all {total_updates} words must be committed after high-frequency updates"
        );
    }

    /// Feed 200 sentence-boundary words into a `LiveCaptionBuffer` at 5 ms
    /// intervals and verify that all sentence-boundary flushes succeed without
    /// panicking or producing empty segments.
    #[test]
    fn test_live_caption_buffer_high_frequency_sentences() {
        let mut buf = LiveCaptionBuffer::new();
        let confidence_min = 0.8_f32;
        let mut segment_count = 0usize;

        let total_words = 200usize;
        for i in 0..total_words {
            // Every 5th word ends a sentence; others do not.
            let word_text = if i % 5 == 4 {
                format!("end{}.", i)
            } else {
                format!("word{i}")
            };
            let start_ms = (i as u64) * 5;
            let end_ms = start_ms + 4;
            buf.add_word(CaptionWord::new(
                word_text, 0.95_f32, start_ms, end_ms, None,
            ));

            if let Some(seg) = buf.flush_ready(confidence_min) {
                assert!(
                    !seg.text.is_empty(),
                    "flushed segment at word {i} must not be empty"
                );
                assert!(seg.is_final, "flushed segment must be marked final");
                segment_count += 1;
            }
        }

        // Flush any remainder.
        if let Some(seg) = buf.flush_all() {
            assert!(!seg.text.is_empty());
            segment_count += 1;
        }

        assert!(
            segment_count > 0,
            "at least one segment should have been flushed during 200-word stress run"
        );
    }

    /// `LatencyMonitor` must remain accurate and not panic under a large
    /// number of recorded measurements.
    #[test]
    fn test_latency_monitor_high_frequency_recordings() {
        let mut monitor = LatencyMonitor::new(3000);
        // Simulate 1000 back-to-back latency samples between 10 ms and 2000 ms.
        for i in 0..1000u64 {
            monitor.add(10 + (i % 2000));
        }
        assert_eq!(monitor.count(), 1000);
        // p95 must be a sensible value (≤ max sample, ≥ min sample).
        let p95 = monitor.p95_ms();
        assert!(
            p95 >= 10 && p95 <= 2009,
            "p95 {p95} ms must lie within the sample range [10, 2009]"
        );
    }
}

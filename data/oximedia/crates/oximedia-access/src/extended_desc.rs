//! Extended audio description management.
//!
//! Extended audio description pauses the main programme when dialogue gaps are
//! insufficient to fit a full description.  This module provides types for
//! scheduling such pauses and the descriptions that fill them.

use serde::{Deserialize, Serialize};

/// A single extended audio description segment.
///
/// When the main video is paused, this description fills the pause window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedDesc {
    /// Unique identifier for this segment.
    pub segment_id: u64,
    /// The textual description to be read aloud.
    pub description: String,
    /// Optional path to a pre-rendered audio file for this description.
    pub audio_path: Option<String>,
    /// Duration of the description audio in milliseconds.
    pub duration_ms: u64,
}

impl ExtendedDesc {
    /// Create a new extended description segment.
    #[must_use]
    pub fn new(segment_id: u64, description: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            segment_id,
            description: description.into(),
            audio_path: None,
            duration_ms,
        }
    }

    /// Create a segment with a pre-rendered audio file.
    #[must_use]
    pub fn with_audio(
        segment_id: u64,
        description: impl Into<String>,
        audio_path: impl Into<String>,
        duration_ms: u64,
    ) -> Self {
        Self {
            segment_id,
            description: description.into(),
            audio_path: Some(audio_path.into()),
            duration_ms,
        }
    }

    /// Estimated word count derived from the description text.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.description.split_whitespace().count()
    }

    /// Estimated reading speed in words per minute given the duration.
    ///
    /// Returns `None` if the duration is zero.
    #[must_use]
    pub fn words_per_minute(&self) -> Option<f64> {
        if self.duration_ms == 0 {
            return None;
        }
        let minutes = self.duration_ms as f64 / 60_000.0;
        Some(self.word_count() as f64 / minutes)
    }
}

/// An ordered schedule of extended audio description segments.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtendedDescSchedule {
    /// Ordered list of description segments.
    pub segments: Vec<ExtendedDesc>,
    /// Cumulative pause time introduced by all segments (milliseconds).
    pub total_pause_ms: u64,
}

impl ExtendedDescSchedule {
    /// Create an empty schedule.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a description segment to the schedule.
    pub fn add(&mut self, desc: ExtendedDesc) {
        self.total_pause_ms += desc.duration_ms;
        self.segments.push(desc);
    }

    /// Total pause time introduced by all segments, in milliseconds.
    #[must_use]
    pub fn total_pause_time_ms(&self) -> u64 {
        self.total_pause_ms
    }

    /// Number of description segments in the schedule.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns `true` when no segments have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Iterate over all segments.
    pub fn iter(&self) -> std::slice::Iter<'_, ExtendedDesc> {
        self.segments.iter()
    }
}

/// Schedule extended descriptions against a timeline, returning each description
/// paired with the timestamp (in milliseconds from the original timeline start)
/// at which the pause should begin.
///
/// The caller provides the description segments in the order they should appear
/// and the original timeline length.  The function distributes the pauses
/// evenly so that they do not cluster at the start.
///
/// # Arguments
///
/// * `pauses` – Slice of descriptions to schedule.
/// * `timeline_ms` – Total duration of the original programme in milliseconds.
///
/// # Returns
///
/// A `Vec` of `(pause_start_ms, &ExtendedDesc)` tuples in chronological order.
#[must_use]
pub fn schedule_extended_descriptions(
    pauses: &[ExtendedDesc],
    timeline_ms: u64,
) -> Vec<(u64, &ExtendedDesc)> {
    if pauses.is_empty() || timeline_ms == 0 {
        return Vec::new();
    }

    let n = pauses.len() as u64;
    let step = timeline_ms / (n + 1);

    pauses
        .iter()
        .enumerate()
        .map(|(i, desc)| {
            let pause_start = step * (i as u64 + 1);
            (pause_start, desc)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ExtendedDesc ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_creates_desc() {
        let desc = ExtendedDesc::new(1, "A dramatic scene unfolds.", 3000);
        assert_eq!(desc.segment_id, 1);
        assert_eq!(desc.duration_ms, 3000);
        assert!(desc.audio_path.is_none());
    }

    #[test]
    fn test_with_audio_sets_path() {
        let desc = ExtendedDesc::with_audio(2, "Explosion.", "/audio/explosion.wav", 2000);
        assert_eq!(desc.audio_path.as_deref(), Some("/audio/explosion.wav"));
    }

    #[test]
    fn test_word_count() {
        let desc = ExtendedDesc::new(1, "A man runs across the street.", 3000);
        assert_eq!(desc.word_count(), 6);
    }

    #[test]
    fn test_word_count_empty() {
        let desc = ExtendedDesc::new(1, "", 3000);
        assert_eq!(desc.word_count(), 0);
    }

    #[test]
    fn test_words_per_minute_reasonable() {
        // 12 words in 3000 ms = 0.05 min → 240 wpm
        let desc = ExtendedDesc::new(
            1,
            "one two three four five six seven eight nine ten eleven twelve",
            3000,
        );
        let wpm = desc.words_per_minute().expect("non-zero duration");
        assert!((wpm - 240.0).abs() < 1.0);
    }

    #[test]
    fn test_words_per_minute_zero_duration() {
        let desc = ExtendedDesc::new(1, "hello world", 0);
        assert!(desc.words_per_minute().is_none());
    }

    // ── ExtendedDescSchedule ──────────────────────────────────────────────────

    #[test]
    fn test_schedule_new_is_empty() {
        let sched = ExtendedDescSchedule::new();
        assert!(sched.is_empty());
        assert_eq!(sched.segment_count(), 0);
        assert_eq!(sched.total_pause_time_ms(), 0);
    }

    #[test]
    fn test_schedule_add_accumulates_pause() {
        let mut sched = ExtendedDescSchedule::new();
        sched.add(ExtendedDesc::new(1, "First.", 2000));
        sched.add(ExtendedDesc::new(2, "Second.", 3000));
        assert_eq!(sched.segment_count(), 2);
        assert_eq!(sched.total_pause_time_ms(), 5000);
    }

    #[test]
    fn test_schedule_iter() {
        let mut sched = ExtendedDescSchedule::new();
        sched.add(ExtendedDesc::new(10, "Desc A", 1000));
        sched.add(ExtendedDesc::new(20, "Desc B", 2000));
        let ids: Vec<u64> = sched.iter().map(|d| d.segment_id).collect();
        assert_eq!(ids, vec![10, 20]);
    }

    // ── schedule_extended_descriptions ────────────────────────────────────────

    #[test]
    fn test_schedule_fn_empty_input() {
        let result = schedule_extended_descriptions(&[], 60_000);
        assert!(result.is_empty());
    }

    #[test]
    fn test_schedule_fn_zero_timeline() {
        let pauses = vec![ExtendedDesc::new(1, "Desc", 2000)];
        let result = schedule_extended_descriptions(&pauses, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_schedule_fn_single_desc_placed_at_midpoint() {
        let pauses = vec![ExtendedDesc::new(1, "Single description.", 3000)];
        let result = schedule_extended_descriptions(&pauses, 60_000);
        assert_eq!(result.len(), 1);
        // With 1 pause the step is 60_000 / 2 = 30_000
        assert_eq!(result[0].0, 30_000);
        assert_eq!(result[0].1.segment_id, 1);
    }

    #[test]
    fn test_schedule_fn_multiple_descs_evenly_distributed() {
        let pauses = vec![
            ExtendedDesc::new(1, "First.", 1000),
            ExtendedDesc::new(2, "Second.", 1000),
            ExtendedDesc::new(3, "Third.", 1000),
        ];
        let result = schedule_extended_descriptions(&pauses, 120_000);
        // step = 120_000 / 4 = 30_000
        assert_eq!(result[0].0, 30_000);
        assert_eq!(result[1].0, 60_000);
        assert_eq!(result[2].0, 90_000);
    }

    #[test]
    fn test_schedule_fn_preserves_order() {
        let pauses = vec![
            ExtendedDesc::new(10, "Alpha.", 500),
            ExtendedDesc::new(20, "Beta.", 500),
        ];
        let result = schedule_extended_descriptions(&pauses, 90_000);
        assert_eq!(result[0].1.segment_id, 10);
        assert_eq!(result[1].1.segment_id, 20);
        assert!(result[0].0 < result[1].0);
    }
}

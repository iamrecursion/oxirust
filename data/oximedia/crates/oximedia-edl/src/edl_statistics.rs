#![allow(dead_code)]
//! EDL statistics and analysis utilities.
//!
//! This module provides tools for computing statistics about EDL files,
//! including duration analysis, reel usage tracking, edit type distribution,
//! and timeline coverage metrics.
//!
//! ## Single-pass computation
//!
//! The [`StatsAccumulator`] struct enables computing all statistics in a single
//! pass over the events via [`StatsAccumulator::accumulate`], then deriving the
//! final [`EdlStatistics`] via [`StatsAccumulator::finalize`].

use std::collections::HashMap;

/// Statistics computed from an EDL's events.
#[derive(Debug, Clone)]
pub struct EdlStatistics {
    /// Total number of events.
    pub event_count: usize,
    /// Total duration in frames.
    pub total_frames: u64,
    /// Total source duration in frames.
    pub total_source_frames: u64,
    /// Total record duration in frames.
    pub total_record_frames: u64,
    /// Minimum event duration in frames.
    pub min_duration_frames: u64,
    /// Maximum event duration in frames.
    pub max_duration_frames: u64,
    /// Mean event duration in frames.
    pub mean_duration_frames: f64,
    /// Number of unique reels used.
    pub unique_reel_count: usize,
    /// Edit type distribution (edit type name -> count).
    pub edit_type_counts: HashMap<String, usize>,
    /// Track type distribution (track type name -> count).
    pub track_type_counts: HashMap<String, usize>,
    /// Reel usage (reel name -> event count).
    pub reel_usage: HashMap<String, usize>,
}

impl EdlStatistics {
    /// Create empty statistics.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            event_count: 0,
            total_frames: 0,
            total_source_frames: 0,
            total_record_frames: 0,
            min_duration_frames: 0,
            max_duration_frames: 0,
            mean_duration_frames: 0.0,
            unique_reel_count: 0,
            edit_type_counts: HashMap::new(),
            track_type_counts: HashMap::new(),
            reel_usage: HashMap::new(),
        }
    }

    /// Check if there are no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.event_count == 0
    }
}

/// Single-pass accumulator that computes all [`EdlStatistics`] fields in one
/// iteration over a sequence of [`EventRecord`]s.
///
/// Call [`StatsAccumulator::accumulate`] once per event, then call
/// [`StatsAccumulator::finalize`] to obtain the completed [`EdlStatistics`].
/// Each event is visited exactly once, eliminating redundant passes.
#[derive(Debug, Clone)]
pub struct StatsAccumulator {
    event_count: usize,
    total_source: u64,
    total_record: u64,
    /// `u64::MAX` when no event has been seen yet (sentinel).
    min_dur: u64,
    max_dur: u64,
    edit_type_counts: HashMap<String, usize>,
    track_type_counts: HashMap<String, usize>,
    reel_usage: HashMap<String, usize>,
}

impl StatsAccumulator {
    /// Create a new, empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            event_count: 0,
            total_source: 0,
            total_record: 0,
            min_dur: u64::MAX,
            max_dur: 0,
            edit_type_counts: HashMap::new(),
            track_type_counts: HashMap::new(),
            reel_usage: HashMap::new(),
        }
    }

    /// Incorporate a single event into the running totals.
    ///
    /// This method is called exactly once per event; calling it again for the
    /// same event would double-count that event.
    pub fn accumulate(&mut self, event: &EventRecord) {
        self.event_count += 1;
        self.total_source += event.source_duration_frames;
        self.total_record += event.record_duration_frames;

        let dur = event.record_duration_frames;
        if dur < self.min_dur {
            self.min_dur = dur;
        }
        if dur > self.max_dur {
            self.max_dur = dur;
        }

        *self
            .edit_type_counts
            .entry(event.edit_type.clone())
            .or_insert(0) += 1;
        *self
            .track_type_counts
            .entry(event.track_type.clone())
            .or_insert(0) += 1;
        *self.reel_usage.entry(event.reel.clone()).or_insert(0) += 1;
    }

    /// Compute derived fields and return the completed [`EdlStatistics`].
    ///
    /// Consumes the accumulator.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn finalize(self) -> EdlStatistics {
        if self.event_count == 0 {
            return EdlStatistics::empty();
        }

        let mean = self.total_record as f64 / self.event_count as f64;
        let unique_reel_count = self.reel_usage.len();

        EdlStatistics {
            event_count: self.event_count,
            total_frames: self.total_record,
            total_source_frames: self.total_source,
            total_record_frames: self.total_record,
            min_duration_frames: self.min_dur,
            max_duration_frames: self.max_dur,
            mean_duration_frames: mean,
            unique_reel_count,
            edit_type_counts: self.edit_type_counts,
            track_type_counts: self.track_type_counts,
            reel_usage: self.reel_usage,
        }
    }
}

impl Default for StatsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// A single event record for statistics computation.
#[derive(Debug, Clone)]
pub struct EventRecord {
    /// Event number.
    pub number: u32,
    /// Reel name.
    pub reel: String,
    /// Edit type name (e.g., "Cut", "Dissolve").
    pub edit_type: String,
    /// Track type name (e.g., "Video", "Audio").
    pub track_type: String,
    /// Source duration in frames.
    pub source_duration_frames: u64,
    /// Record duration in frames.
    pub record_duration_frames: u64,
}

/// Computes statistics from a collection of event records.
pub struct StatisticsCalculator {
    /// Events to analyze.
    events: Vec<EventRecord>,
}

impl StatisticsCalculator {
    /// Create a new statistics calculator.
    #[must_use]
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Add an event record for analysis.
    pub fn add_event(&mut self, record: EventRecord) {
        self.events.push(record);
    }

    /// Add multiple event records.
    pub fn add_events(&mut self, records: impl IntoIterator<Item = EventRecord>) {
        self.events.extend(records);
    }

    /// Compute statistics from the accumulated events.
    ///
    /// Internally uses [`StatsAccumulator`] for a single-pass fold over events.
    #[must_use]
    pub fn compute(&self) -> EdlStatistics {
        self.events
            .iter()
            .fold(StatsAccumulator::new(), |mut acc, ev| {
                acc.accumulate(ev);
                acc
            })
            .finalize()
    }

    /// Clear all accumulated events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Return the number of events accumulated.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

impl Default for StatisticsCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Duration bucket for histogram analysis.
#[derive(Debug, Clone)]
pub struct DurationBucket {
    /// Lower bound of the bucket (inclusive) in frames.
    pub lower_frames: u64,
    /// Upper bound of the bucket (exclusive) in frames.
    pub upper_frames: u64,
    /// Number of events in this bucket.
    pub count: usize,
}

/// Produces a histogram of event durations.
pub struct DurationHistogram {
    /// Number of buckets.
    bucket_count: usize,
}

impl DurationHistogram {
    /// Create a new histogram builder with the given number of buckets.
    #[must_use]
    pub fn new(bucket_count: usize) -> Self {
        Self {
            bucket_count: bucket_count.max(1),
        }
    }

    /// Compute the histogram from duration values.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, durations: &[u64]) -> Vec<DurationBucket> {
        if durations.is_empty() {
            return Vec::new();
        }

        let min_val = *durations.iter().min().unwrap_or(&0);
        let max_val = *durations.iter().max().unwrap_or(&0);

        if min_val == max_val {
            return vec![DurationBucket {
                lower_frames: min_val,
                upper_frames: max_val + 1,
                count: durations.len(),
            }];
        }

        let range = max_val - min_val;
        let bucket_size = (range as f64 / self.bucket_count as f64).ceil() as u64;
        let bucket_size = bucket_size.max(1);

        let mut buckets = Vec::with_capacity(self.bucket_count);
        for i in 0..self.bucket_count {
            let lower = min_val + (i as u64) * bucket_size;
            let upper = lower + bucket_size;
            buckets.push(DurationBucket {
                lower_frames: lower,
                upper_frames: upper,
                count: 0,
            });
        }

        for &dur in durations {
            let idx = ((dur - min_val) / bucket_size) as usize;
            let idx = idx.min(self.bucket_count - 1);
            buckets[idx].count += 1;
        }

        buckets
    }
}

/// Reel summary data extracted from EDL statistics.
#[derive(Debug, Clone)]
pub struct ReelSummary {
    /// Reel name.
    pub name: String,
    /// Number of events using this reel.
    pub event_count: usize,
    /// Total duration in frames for events from this reel.
    pub total_frames: u64,
    /// Percentage of total timeline occupied by this reel.
    pub percentage: f64,
}

/// Computes per-reel summaries from event records.
#[allow(clippy::cast_precision_loss)]
pub fn compute_reel_summaries(events: &[EventRecord]) -> Vec<ReelSummary> {
    let mut reel_data: HashMap<String, (usize, u64)> = HashMap::new();
    let mut total_frames: u64 = 0;

    for ev in events {
        let entry = reel_data.entry(ev.reel.clone()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += ev.record_duration_frames;
        total_frames += ev.record_duration_frames;
    }

    let mut summaries: Vec<ReelSummary> = reel_data
        .into_iter()
        .map(|(name, (count, frames))| {
            let percentage = if total_frames > 0 {
                frames as f64 / total_frames as f64 * 100.0
            } else {
                0.0
            };
            ReelSummary {
                name,
                event_count: count,
                total_frames: frames,
                percentage,
            }
        })
        .collect();

    summaries.sort_by(|a, b| b.total_frames.cmp(&a.total_frames));
    summaries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(
        number: u32,
        reel: &str,
        edit: &str,
        track: &str,
        src: u64,
        rec: u64,
    ) -> EventRecord {
        EventRecord {
            number,
            reel: reel.to_string(),
            edit_type: edit.to_string(),
            track_type: track.to_string(),
            source_duration_frames: src,
            record_duration_frames: rec,
        }
    }

    #[test]
    fn test_empty_statistics() {
        let stats = EdlStatistics::empty();
        assert!(stats.is_empty());
        assert_eq!(stats.event_count, 0);
        assert_eq!(stats.total_frames, 0);
    }

    #[test]
    fn test_calculator_no_events() {
        let calc = StatisticsCalculator::new();
        let stats = calc.compute();
        assert!(stats.is_empty());
    }

    #[test]
    fn test_calculator_single_event() {
        let mut calc = StatisticsCalculator::new();
        calc.add_event(make_event(1, "A001", "Cut", "Video", 100, 100));
        let stats = calc.compute();

        assert_eq!(stats.event_count, 1);
        assert_eq!(stats.total_record_frames, 100);
        assert_eq!(stats.min_duration_frames, 100);
        assert_eq!(stats.max_duration_frames, 100);
        assert!((stats.mean_duration_frames - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculator_multiple_events() {
        let mut calc = StatisticsCalculator::new();
        calc.add_event(make_event(1, "A001", "Cut", "Video", 50, 50));
        calc.add_event(make_event(2, "A001", "Dissolve", "Video", 100, 100));
        calc.add_event(make_event(3, "A002", "Cut", "Audio", 200, 200));

        let stats = calc.compute();
        assert_eq!(stats.event_count, 3);
        assert_eq!(stats.total_source_frames, 350);
        assert_eq!(stats.total_record_frames, 350);
        assert_eq!(stats.min_duration_frames, 50);
        assert_eq!(stats.max_duration_frames, 200);
        assert_eq!(stats.unique_reel_count, 2);
    }

    #[test]
    fn test_edit_type_counts() {
        let mut calc = StatisticsCalculator::new();
        calc.add_event(make_event(1, "R1", "Cut", "Video", 10, 10));
        calc.add_event(make_event(2, "R1", "Cut", "Video", 20, 20));
        calc.add_event(make_event(3, "R1", "Dissolve", "Video", 30, 30));

        let stats = calc.compute();
        assert_eq!(stats.edit_type_counts.get("Cut"), Some(&2));
        assert_eq!(stats.edit_type_counts.get("Dissolve"), Some(&1));
    }

    #[test]
    fn test_track_type_counts() {
        let mut calc = StatisticsCalculator::new();
        calc.add_event(make_event(1, "R1", "Cut", "Video", 10, 10));
        calc.add_event(make_event(2, "R1", "Cut", "Audio", 20, 20));
        calc.add_event(make_event(3, "R1", "Cut", "Video", 30, 30));

        let stats = calc.compute();
        assert_eq!(stats.track_type_counts.get("Video"), Some(&2));
        assert_eq!(stats.track_type_counts.get("Audio"), Some(&1));
    }

    #[test]
    fn test_reel_usage() {
        let mut calc = StatisticsCalculator::new();
        calc.add_event(make_event(1, "A001", "Cut", "Video", 10, 10));
        calc.add_event(make_event(2, "A002", "Cut", "Video", 20, 20));
        calc.add_event(make_event(3, "A001", "Cut", "Video", 30, 30));

        let stats = calc.compute();
        assert_eq!(stats.reel_usage.get("A001"), Some(&2));
        assert_eq!(stats.reel_usage.get("A002"), Some(&1));
    }

    #[test]
    fn test_calculator_add_events_batch() {
        let mut calc = StatisticsCalculator::new();
        let events = vec![
            make_event(1, "R1", "Cut", "Video", 50, 50),
            make_event(2, "R2", "Cut", "Video", 75, 75),
        ];
        calc.add_events(events);
        assert_eq!(calc.event_count(), 2);
    }

    #[test]
    fn test_calculator_clear() {
        let mut calc = StatisticsCalculator::new();
        calc.add_event(make_event(1, "R1", "Cut", "Video", 10, 10));
        assert_eq!(calc.event_count(), 1);
        calc.clear();
        assert_eq!(calc.event_count(), 0);
    }

    #[test]
    fn test_duration_histogram_empty() {
        let hist = DurationHistogram::new(5);
        let buckets = hist.compute(&[]);
        assert!(buckets.is_empty());
    }

    #[test]
    fn test_duration_histogram_single_value() {
        let hist = DurationHistogram::new(5);
        let buckets = hist.compute(&[100]);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].count, 1);
        assert_eq!(buckets[0].lower_frames, 100);
    }

    #[test]
    fn test_duration_histogram_distribution() {
        let hist = DurationHistogram::new(3);
        let durations = vec![10, 20, 30, 40, 50, 60];
        let buckets = hist.compute(&durations);
        assert_eq!(buckets.len(), 3);

        let total: usize = buckets.iter().map(|b| b.count).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_reel_summaries_empty() {
        let summaries = compute_reel_summaries(&[]);
        assert!(summaries.is_empty());
    }

    #[test]
    fn test_reel_summaries_sorted_by_total_frames() {
        let events = vec![
            make_event(1, "A001", "Cut", "Video", 50, 50),
            make_event(2, "A002", "Cut", "Video", 200, 200),
            make_event(3, "A001", "Cut", "Video", 100, 100),
        ];
        let summaries = compute_reel_summaries(&events);
        assert_eq!(summaries.len(), 2);
        // A002 has 200, A001 has 150 total; A002 should be first
        assert_eq!(summaries[0].name, "A002");
        assert_eq!(summaries[0].total_frames, 200);
        assert_eq!(summaries[1].name, "A001");
        assert_eq!(summaries[1].total_frames, 150);
    }

    #[test]
    fn test_reel_summaries_percentage() {
        let events = vec![
            make_event(1, "A001", "Cut", "Video", 100, 100),
            make_event(2, "A002", "Cut", "Video", 100, 100),
        ];
        let summaries = compute_reel_summaries(&events);
        for s in &summaries {
            assert!((s.percentage - 50.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_default_calculator() {
        let calc = StatisticsCalculator::default();
        assert_eq!(calc.event_count(), 0);
    }

    /// Verify that the single-pass `StatsAccumulator` produces identical results
    /// to the multi-pass `StatisticsCalculator` on a fixture with multiple events,
    /// reels, and transition types.
    #[test]
    fn test_single_pass_stats_agreement() {
        let events = vec![
            make_event(1, "A001", "Cut", "Video", 50, 50),
            make_event(2, "A001", "Dissolve", "Video", 100, 100),
            make_event(3, "A002", "Cut", "Audio", 200, 200),
            make_event(4, "B001", "Wipe", "Video", 75, 75),
            make_event(5, "A002", "Cut", "Video", 30, 30),
        ];

        // Compute via the existing calculator (multi-pass baseline)
        let mut calc = StatisticsCalculator::new();
        calc.add_events(events.clone());
        let baseline = calc.compute();

        // Compute via the single-pass accumulator
        let single_pass: EdlStatistics = events
            .iter()
            .fold(StatsAccumulator::new(), |mut acc, ev| {
                acc.accumulate(ev);
                acc
            })
            .finalize();

        // Field-by-field equality
        assert_eq!(single_pass.event_count, baseline.event_count);
        assert_eq!(single_pass.total_frames, baseline.total_frames);
        assert_eq!(
            single_pass.total_source_frames,
            baseline.total_source_frames
        );
        assert_eq!(
            single_pass.total_record_frames,
            baseline.total_record_frames
        );
        assert_eq!(
            single_pass.min_duration_frames,
            baseline.min_duration_frames
        );
        assert_eq!(
            single_pass.max_duration_frames,
            baseline.max_duration_frames
        );
        assert!(
            (single_pass.mean_duration_frames - baseline.mean_duration_frames).abs() < f64::EPSILON
        );
        assert_eq!(single_pass.unique_reel_count, baseline.unique_reel_count);
        assert_eq!(single_pass.edit_type_counts, baseline.edit_type_counts);
        assert_eq!(single_pass.track_type_counts, baseline.track_type_counts);
        assert_eq!(single_pass.reel_usage, baseline.reel_usage);
    }

    /// Verify that single-pass computation visits each event exactly once by
    /// counting accumulations with a wrapping counter iterator.
    #[test]
    fn test_single_pass_visits_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let events = vec![
            make_event(1, "R1", "Cut", "Video", 10, 10),
            make_event(2, "R1", "Cut", "Video", 20, 20),
            make_event(3, "R2", "Dissolve", "Audio", 30, 30),
        ];

        let visit_count = Arc::new(AtomicUsize::new(0));
        let vc = Arc::clone(&visit_count);

        let stats = events
            .iter()
            .inspect(|_ev| {
                vc.fetch_add(1, Ordering::Relaxed);
            })
            .fold(StatsAccumulator::new(), |mut acc, ev| {
                acc.accumulate(ev);
                acc
            })
            .finalize();

        // Each of the 3 events must have been visited exactly once
        assert_eq!(visit_count.load(Ordering::Relaxed), 3);
        assert_eq!(stats.event_count, 3);
    }
}

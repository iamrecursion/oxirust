//! Beat tracking and tempo hypothesis scoring.
//!
//! Provides a streaming `BeatTracker` that accumulates onset positions and
//! produces tempo hypotheses with ranked confidence scores.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single tempo hypothesis with an associated confidence score.
#[derive(Debug, Clone)]
pub struct TempoHypothesis {
    /// Estimated tempo in beats-per-minute.
    pub bpm: f32,
    /// Confidence in [0, 1].
    pub confidence: f32,
    /// Period in seconds corresponding to this tempo.
    pub period_s: f32,
}

impl TempoHypothesis {
    /// Create a new tempo hypothesis.
    #[must_use]
    pub fn new(bpm: f32, confidence: f32) -> Self {
        let period_s = 60.0 / bpm;
        Self {
            bpm,
            confidence,
            period_s,
        }
    }

    /// Return the normalised score (same as `confidence`).
    #[must_use]
    pub fn score(&self) -> f32 {
        self.confidence
    }

    /// Return the period in milliseconds.
    #[must_use]
    pub fn period_ms(&self) -> f32 {
        self.period_s * 1000.0
    }
}

/// A tempo interval computed from two consecutive beat positions.
#[derive(Debug, Clone, Copy)]
pub struct TempoInterval {
    /// Start time in seconds.
    pub start_s: f32,
    /// End time in seconds.
    pub end_s: f32,
}

impl TempoInterval {
    /// Create a new tempo interval.
    #[must_use]
    pub fn new(start_s: f32, end_s: f32) -> Self {
        Self { start_s, end_s }
    }

    /// Compute BPM from interval length.
    #[must_use]
    pub fn bpm(&self) -> f32 {
        let dur = self.end_s - self.start_s;
        if dur > 0.0 {
            60.0 / dur
        } else {
            0.0
        }
    }

    /// Interval duration in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f32 {
        self.end_s - self.start_s
    }
}

/// Accumulates onset times and tracks beats via inter-onset-interval analysis.
#[derive(Debug, Clone)]
pub struct BeatTracker {
    sample_rate: f32,
    hop_size: usize,
    min_bpm: f32,
    max_bpm: f32,
    /// Onset times in seconds that have been fed so far.
    onset_times: Vec<f32>,
    /// Beat positions detected on the last `detect_tempo` call.
    beat_positions: Vec<f32>,
}

impl BeatTracker {
    /// Create a new beat tracker.
    ///
    /// * `sample_rate` – audio sample rate in Hz
    /// * `hop_size`    – hop size used to compute onset times from frame indices
    /// * `min_bpm`     – lower bound for tempo search (default 60)
    /// * `max_bpm`     – upper bound for tempo search (default 200)
    #[must_use]
    pub fn new(sample_rate: f32, hop_size: usize, min_bpm: f32, max_bpm: f32) -> Self {
        Self {
            sample_rate,
            hop_size,
            min_bpm,
            max_bpm,
            onset_times: Vec::new(),
            beat_positions: Vec::new(),
        }
    }

    /// Add an onset at the given frame index.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_onset(&mut self, frame_index: usize) {
        let time_s = (frame_index * self.hop_size) as f32 / self.sample_rate;
        self.onset_times.push(time_s);
    }

    /// Add an onset at the given time in seconds directly.
    pub fn add_onset_time(&mut self, time_s: f32) {
        self.onset_times.push(time_s);
    }

    /// Detect the dominant tempo from accumulated onsets using IOI histogram.
    ///
    /// Returns up to `n_hypotheses` ranked `TempoHypothesis` values.
    #[must_use]
    pub fn detect_tempo(&mut self, n_hypotheses: usize) -> Vec<TempoHypothesis> {
        if self.onset_times.len() < 2 {
            return Vec::new();
        }

        let min_period = 60.0 / self.max_bpm;
        let max_period = 60.0 / self.min_bpm;

        // Build IOI histogram with fixed-width bins of 10 ms
        let bin_width = 0.010_f32; // 10 ms
        let n_bins = ((max_period - min_period) / bin_width).ceil() as usize + 1;
        let mut histogram = vec![0_u32; n_bins];

        for i in 0..self.onset_times.len() {
            for j in (i + 1)..self.onset_times.len().min(i + 8) {
                let ioi = self.onset_times[j] - self.onset_times[i];
                if ioi >= min_period && ioi <= max_period {
                    let bin = ((ioi - min_period) / bin_width) as usize;
                    if bin < n_bins {
                        histogram[bin] += 1;
                    }
                }
            }
        }

        let total: u32 = histogram.iter().sum();
        if total == 0 {
            return Vec::new();
        }

        // Collect peaks
        let mut peaks: Vec<(usize, u32)> =
            histogram.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        peaks.sort_by(|a, b| b.1.cmp(&a.1));

        let hypotheses: Vec<TempoHypothesis> = peaks
            .iter()
            .take(n_hypotheses)
            .filter(|(_, count)| *count > 0)
            .map(|(bin, count)| {
                let period = min_period + (*bin as f32 + 0.5) * bin_width;
                let bpm = 60.0 / period;
                let confidence = *count as f32 / total as f32;
                TempoHypothesis::new(bpm, confidence)
            })
            .collect();

        // Generate beat grid from top hypothesis
        if let Some(top) = hypotheses.first() {
            self.beat_positions = self.generate_beat_grid(top.period_s);
        }

        hypotheses
    }

    /// Return the beat positions (in seconds) from the most recent `detect_tempo` call.
    #[must_use]
    pub fn beat_positions(&self) -> &[f32] {
        &self.beat_positions
    }

    /// Return all onset times accumulated so far.
    #[must_use]
    pub fn onset_times(&self) -> &[f32] {
        &self.onset_times
    }

    /// Return intervals between consecutive detected beats.
    #[must_use]
    pub fn beat_intervals(&self) -> Vec<TempoInterval> {
        self.beat_positions
            .windows(2)
            .map(|w| TempoInterval::new(w[0], w[1]))
            .collect()
    }

    // ---- internal helpers ----

    fn generate_beat_grid(&self, period_s: f32) -> Vec<f32> {
        if self.onset_times.is_empty() || period_s <= 0.0 {
            return Vec::new();
        }
        let start = *self.onset_times.first().unwrap_or(&0.0);
        let end = *self.onset_times.last().unwrap_or(&0.0);

        let mut beats = Vec::new();
        let mut t = start;
        while t <= end + period_s * 0.5 {
            beats.push(t);
            t += period_s;
        }
        beats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TempoHypothesis tests

    #[test]
    fn test_hypothesis_bpm_roundtrip() {
        let h = TempoHypothesis::new(120.0, 0.9);
        assert!((h.bpm - 120.0).abs() < 1e-4);
    }

    #[test]
    fn test_hypothesis_period_correct() {
        let h = TempoHypothesis::new(60.0, 1.0);
        assert!((h.period_s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hypothesis_period_ms() {
        let h = TempoHypothesis::new(120.0, 1.0);
        assert!((h.period_ms() - 500.0).abs() < 0.1);
    }

    #[test]
    fn test_hypothesis_score_equals_confidence() {
        let h = TempoHypothesis::new(100.0, 0.7);
        assert!((h.score() - 0.7).abs() < 1e-6);
    }

    // TempoInterval tests

    #[test]
    fn test_interval_bpm_120() {
        let iv = TempoInterval::new(0.0, 0.5);
        assert!((iv.bpm() - 120.0).abs() < 0.1);
    }

    #[test]
    fn test_interval_duration() {
        let iv = TempoInterval::new(1.0, 1.75);
        assert!((iv.duration_s() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_interval_zero_duration() {
        let iv = TempoInterval::new(1.0, 1.0);
        assert_eq!(iv.bpm(), 0.0);
    }

    // BeatTracker tests

    #[test]
    fn test_tracker_no_onsets_returns_empty() {
        let mut tracker = BeatTracker::new(44100.0, 512, 60.0, 200.0);
        let hypotheses = tracker.detect_tempo(3);
        assert!(hypotheses.is_empty());
    }

    #[test]
    fn test_tracker_one_onset_returns_empty() {
        let mut tracker = BeatTracker::new(44100.0, 512, 60.0, 200.0);
        tracker.add_onset_time(0.0);
        assert!(tracker.detect_tempo(3).is_empty());
    }

    #[test]
    fn test_tracker_detects_120bpm() {
        let mut tracker = BeatTracker::new(44100.0, 512, 60.0, 200.0);
        // Add onsets at 0, 0.5, 1.0, 1.5, 2.0 seconds (120 BPM)
        for i in 0..=8 {
            tracker.add_onset_time(i as f32 * 0.5);
        }
        let hyps = tracker.detect_tempo(3);
        assert!(!hyps.is_empty());
        // Top hypothesis should be near 120 BPM
        let top_bpm = hyps[0].bpm;
        assert!(
            top_bpm > 110.0 && top_bpm < 130.0,
            "Expected ~120 BPM, got {top_bpm}"
        );
    }

    #[test]
    fn test_beat_positions_populated_after_detect() {
        let mut tracker = BeatTracker::new(44100.0, 512, 60.0, 200.0);
        for i in 0..=8 {
            tracker.add_onset_time(i as f32 * 0.5);
        }
        let _ = tracker.detect_tempo(1);
        assert!(!tracker.beat_positions().is_empty());
    }

    #[test]
    fn test_add_onset_by_frame() {
        let mut tracker = BeatTracker::new(44100.0, 512, 60.0, 200.0);
        tracker.add_onset(0);
        tracker.add_onset(3969); // ~0.046s per frame * 3969 frames ≈ 46s
        assert_eq!(tracker.onset_times().len(), 2);
    }

    #[test]
    fn test_onset_times_sorted_by_insertion() {
        let mut tracker = BeatTracker::new(44100.0, 512, 60.0, 200.0);
        tracker.add_onset_time(0.0);
        tracker.add_onset_time(0.5);
        tracker.add_onset_time(1.0);
        let times = tracker.onset_times();
        assert_eq!(times.len(), 3);
        assert!(times[0] < times[1] && times[1] < times[2]);
    }

    #[test]
    fn test_beat_intervals_count() {
        let mut tracker = BeatTracker::new(44100.0, 512, 60.0, 200.0);
        for i in 0..=10 {
            tracker.add_onset_time(i as f32 * 0.5);
        }
        let _ = tracker.detect_tempo(1);
        let ivs = tracker.beat_intervals();
        // intervals count = beats - 1
        assert!(!ivs.is_empty());
    }

    #[test]
    fn test_hypothesis_confidence_sum_not_exceeds_one() {
        let mut tracker = BeatTracker::new(44100.0, 512, 60.0, 200.0);
        for i in 0..=16 {
            tracker.add_onset_time(i as f32 * 0.5);
        }
        let hyps = tracker.detect_tempo(5);
        let total: f32 = hyps.iter().map(|h| h.confidence).sum();
        assert!(total <= 1.0 + 1e-4);
    }
}

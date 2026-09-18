#![allow(dead_code)]
//! Tempo map for music information retrieval.
//!
//! Represents tempo changes across a musical timeline, enabling accurate
//! beat-position to time conversion (and vice-versa) when tempo is not constant.

use std::fmt;

/// A tempo change event at a specific beat position.
#[derive(Debug, Clone, PartialEq)]
pub struct TempoChange {
    /// Beat position where this tempo takes effect (0-based).
    pub beat: f64,
    /// Tempo in BPM.
    pub bpm: f64,
    /// Time in seconds at this beat (computed).
    pub time_s: f64,
}

impl TempoChange {
    /// Create a tempo change at a given beat with a given BPM.
    #[must_use]
    pub fn new(beat: f64, bpm: f64) -> Self {
        Self {
            beat,
            bpm,
            time_s: 0.0,
        }
    }

    /// Seconds per beat at this tempo.
    #[must_use]
    pub fn seconds_per_beat(&self) -> f64 {
        if self.bpm <= 0.0 {
            return f64::INFINITY;
        }
        60.0 / self.bpm
    }
}

impl fmt::Display for TempoChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "beat={:.2} bpm={:.1} t={:.3}s",
            self.beat, self.bpm, self.time_s
        )
    }
}

/// A complete tempo map with one or more tempo regions.
#[derive(Debug, Clone)]
pub struct TempoMap {
    /// Tempo changes sorted by beat position.
    regions: Vec<TempoChange>,
}

impl TempoMap {
    /// Create a tempo map with a single initial tempo.
    #[must_use]
    pub fn new(initial_bpm: f64) -> Self {
        let mut map = Self {
            regions: Vec::new(),
        };
        map.regions.push(TempoChange {
            beat: 0.0,
            bpm: initial_bpm,
            time_s: 0.0,
        });
        map
    }

    /// Number of tempo regions.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Add a tempo change region. Recomputes all time stamps.
    pub fn add_region(&mut self, beat: f64, bpm: f64) {
        // Avoid duplicate at same beat
        self.regions.retain(|r| (r.beat - beat).abs() > 1e-9);
        self.regions.push(TempoChange::new(beat, bpm));
        self.regions.sort_by(|a, b| {
            a.beat
                .partial_cmp(&b.beat)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.recompute_times();
    }

    /// Remove all tempo changes except the first.
    pub fn clear(&mut self) {
        if let Some(first) = self.regions.first().cloned() {
            self.regions.clear();
            self.regions.push(first);
        }
    }

    /// Get tempo (BPM) at a given beat position.
    #[must_use]
    pub fn tempo_at_beat(&self, beat: f64) -> f64 {
        let mut bpm = self.regions[0].bpm;
        for r in &self.regions {
            if r.beat <= beat {
                bpm = r.bpm;
            } else {
                break;
            }
        }
        bpm
    }

    /// Convert a beat position to time in seconds.
    #[must_use]
    pub fn beat_to_time(&self, beat: f64) -> f64 {
        let mut time = 0.0;
        let mut prev_beat = 0.0;
        let mut prev_bpm = self.regions[0].bpm;

        for r in &self.regions {
            if r.beat >= beat {
                break;
            }
            // Accumulate time from previous region up to this change
            let delta_beats = r.beat - prev_beat;
            if prev_bpm > 0.0 {
                time += delta_beats * 60.0 / prev_bpm;
            }
            prev_beat = r.beat;
            prev_bpm = r.bpm;
        }

        // Remaining beats after last change before `beat`
        let remaining = beat - prev_beat;
        if prev_bpm > 0.0 {
            time += remaining * 60.0 / prev_bpm;
        }
        time
    }

    /// Convert time in seconds to beat position.
    #[must_use]
    pub fn time_to_beat(&self, time_s: f64) -> f64 {
        let mut elapsed = 0.0;
        let mut beat = 0.0;
        let mut prev_time = 0.0;
        let mut bpm = self.regions[0].bpm;

        for i in 1..self.regions.len() {
            let r = &self.regions[i];
            let region_beats = r.beat - beat;
            let region_time = if bpm > 0.0 {
                region_beats * 60.0 / bpm
            } else {
                0.0
            };

            if elapsed + region_time > time_s {
                break;
            }
            elapsed += region_time;
            beat = r.beat;
            prev_time = elapsed;
            bpm = r.bpm;
        }

        // Remaining time within current region
        let remaining_time = time_s - prev_time;
        if bpm > 0.0 {
            beat += remaining_time * bpm / 60.0;
        }
        beat
    }

    /// Total time in seconds for a given number of beats from the start.
    #[must_use]
    pub fn duration_for_beats(&self, total_beats: f64) -> f64 {
        self.beat_to_time(total_beats)
    }

    /// Return all regions.
    #[must_use]
    pub fn regions(&self) -> &[TempoChange] {
        &self.regions
    }

    /// Average tempo across a beat range.
    #[must_use]
    pub fn average_tempo(&self, start_beat: f64, end_beat: f64) -> f64 {
        if end_beat <= start_beat {
            return self.tempo_at_beat(start_beat);
        }
        let duration = self.beat_to_time(end_beat) - self.beat_to_time(start_beat);
        if duration <= 0.0 {
            return self.tempo_at_beat(start_beat);
        }
        (end_beat - start_beat) * 60.0 / duration
    }

    /// Recompute time stamps for all regions.
    fn recompute_times(&mut self) {
        if self.regions.is_empty() {
            return;
        }
        self.regions[0].time_s = 0.0;
        for i in 1..self.regions.len() {
            let prev_beat = self.regions[i - 1].beat;
            let prev_bpm = self.regions[i - 1].bpm;
            let prev_time = self.regions[i - 1].time_s;
            let delta = self.regions[i].beat - prev_beat;
            self.regions[i].time_s = prev_time
                + if prev_bpm > 0.0 {
                    delta * 60.0 / prev_bpm
                } else {
                    0.0
                };
        }
    }
}

/// Builder for constructing a `TempoMap` step-by-step.
#[derive(Debug)]
pub struct TempoMapBuilder {
    /// Initial BPM.
    initial_bpm: f64,
    /// Pending regions.
    regions: Vec<(f64, f64)>,
}

impl TempoMapBuilder {
    /// Start building with an initial tempo.
    #[must_use]
    pub fn new(initial_bpm: f64) -> Self {
        Self {
            initial_bpm,
            regions: Vec::new(),
        }
    }

    /// Add a tempo region at the given beat.
    #[must_use]
    pub fn add_region(mut self, beat: f64, bpm: f64) -> Self {
        self.regions.push((beat, bpm));
        self
    }

    /// Build the final `TempoMap`.
    #[must_use]
    pub fn build(self) -> TempoMap {
        let mut map = TempoMap::new(self.initial_bpm);
        for (beat, bpm) in self.regions {
            map.add_region(beat, bpm);
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tempo_change_creation() {
        let tc = TempoChange::new(0.0, 120.0);
        assert!((tc.bpm - 120.0).abs() < f64::EPSILON);
        assert!((tc.seconds_per_beat() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tempo_change_display() {
        let tc = TempoChange::new(4.0, 140.0);
        let s = format!("{tc}");
        assert!(s.contains("140.0"));
    }

    #[test]
    fn test_tempo_map_single_tempo() {
        let map = TempoMap::new(120.0);
        assert_eq!(map.region_count(), 1);
        assert!((map.tempo_at_beat(0.0) - 120.0).abs() < f64::EPSILON);
        assert!((map.tempo_at_beat(100.0) - 120.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_beat_to_time_constant() {
        let map = TempoMap::new(120.0);
        // 4 beats at 120 BPM = 2 seconds
        let t = map.beat_to_time(4.0);
        assert!((t - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_time_to_beat_constant() {
        let map = TempoMap::new(120.0);
        let beat = map.time_to_beat(2.0);
        assert!((beat - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_region() {
        let mut map = TempoMap::new(120.0);
        map.add_region(8.0, 140.0);
        assert_eq!(map.region_count(), 2);
        assert!((map.tempo_at_beat(0.0) - 120.0).abs() < f64::EPSILON);
        assert!((map.tempo_at_beat(10.0) - 140.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_beat_to_time_two_tempos() {
        let mut map = TempoMap::new(120.0);
        map.add_region(4.0, 60.0);
        // 0..4 beats at 120 BPM = 2.0s
        // 4..6 beats at 60 BPM = 2.0s
        let t = map.beat_to_time(6.0);
        assert!((t - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_time_to_beat_two_tempos() {
        let mut map = TempoMap::new(120.0);
        map.add_region(4.0, 60.0);
        // 2.0s => beat 4 (end of first region at 120 BPM)
        let beat = map.time_to_beat(2.0);
        assert!((beat - 4.0).abs() < 1e-9);
        // 4.0s => beat 6 (2s more at 60 BPM => 2 beats)
        let beat2 = map.time_to_beat(4.0);
        assert!((beat2 - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_clear() {
        let mut map = TempoMap::new(120.0);
        map.add_region(4.0, 140.0);
        map.add_region(8.0, 100.0);
        assert_eq!(map.region_count(), 3);
        map.clear();
        assert_eq!(map.region_count(), 1);
    }

    #[test]
    fn test_duration_for_beats() {
        let map = TempoMap::new(60.0);
        // 60 BPM => 1 beat/sec
        let dur = map.duration_for_beats(10.0);
        assert!((dur - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_average_tempo_constant() {
        let map = TempoMap::new(120.0);
        let avg = map.average_tempo(0.0, 8.0);
        assert!((avg - 120.0).abs() < 0.01);
    }

    #[test]
    fn test_average_tempo_varying() {
        let mut map = TempoMap::new(120.0);
        map.add_region(4.0, 60.0);
        // 0..4 at 120BPM => 2s, 4..8 at 60BPM => 4s, total=6s for 8 beats
        let avg = map.average_tempo(0.0, 8.0);
        // 8 beats / (6s/60) = 80 BPM
        assert!((avg - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_builder() {
        let map = TempoMapBuilder::new(100.0)
            .add_region(8.0, 120.0)
            .add_region(16.0, 140.0)
            .build();
        assert_eq!(map.region_count(), 3);
        assert!((map.tempo_at_beat(0.0) - 100.0).abs() < f64::EPSILON);
        assert!((map.tempo_at_beat(10.0) - 120.0).abs() < f64::EPSILON);
        assert!((map.tempo_at_beat(20.0) - 140.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_regions_accessor() {
        let map = TempoMap::new(120.0);
        assert_eq!(map.regions().len(), 1);
        assert!((map.regions()[0].bpm - 120.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recomputed_timestamps() {
        let mut map = TempoMap::new(120.0);
        map.add_region(4.0, 60.0);
        // Region at beat 4: time should be 2.0s (4 beats at 120 BPM)
        assert!((map.regions()[1].time_s - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_seconds_per_beat_zero_bpm() {
        let tc = TempoChange::new(0.0, 0.0);
        assert!(tc.seconds_per_beat().is_infinite());
    }
}

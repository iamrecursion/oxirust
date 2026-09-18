//! Beat tracking and tempo estimation for audio analysis.
//!
//! This module provides beat detection, tempo estimation, and meter analysis
//! using onset event times as input.

/// A single detected beat event.
#[derive(Debug, Clone)]
pub struct BeatEvent {
    /// Time of the beat in milliseconds from the start of the signal.
    pub time_ms: u64,
    /// Confidence of this beat detection (0.0 – 1.0).
    pub confidence: f32,
    /// Sequential beat number starting from 1.
    pub beat_number: u32,
}

impl BeatEvent {
    /// Returns `true` if this beat falls on a downbeat for the given meter.
    ///
    /// Beat 1, `1 + meter`, `1 + 2*meter`, … are considered downbeats.
    #[must_use]
    pub fn is_downbeat(&self, meter: u8) -> bool {
        if meter == 0 {
            return false;
        }
        self.beat_number % u32::from(meter) == 1
    }
}

/// Tempo estimate derived from onset data.
#[derive(Debug, Clone)]
pub struct TempoEstimate {
    /// Estimated tempo in beats per minute.
    pub bpm: f32,
    /// Confidence of this estimate (0.0 – 1.0).
    pub confidence: f32,
    /// Period between beats in milliseconds.
    pub period_ms: f32,
}

impl TempoEstimate {
    /// Returns `true` if confidence exceeds 0.5 and the tempo is musically
    /// plausible (30 – 300 BPM).
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.confidence > 0.5 && self.bpm >= 30.0 && self.bpm <= 300.0
    }
}

/// Estimate the tempo from a list of onset times in milliseconds.
///
/// Uses inter-onset interval (IOI) histogram peak to find the most common IOI,
/// then converts to BPM. Returns a [`TempoEstimate`] with low confidence when
/// insufficient onsets are available.
#[must_use]
pub fn estimate_tempo_from_onsets(onsets_ms: &[u64]) -> TempoEstimate {
    // Build a histogram with 10 ms bins from 50 ms to 2000 ms
    const BIN_SIZE: f32 = 10.0;
    const MIN_IOI: f32 = 50.0;
    const MAX_IOI: f32 = 2000.0;

    if onsets_ms.len() < 2 {
        return TempoEstimate {
            bpm: 120.0,
            confidence: 0.0,
            period_ms: 500.0,
        };
    }

    // Compute inter-onset intervals
    let iois: Vec<f32> = onsets_ms
        .windows(2)
        .map(|w| (w[1] - w[0]) as f32)
        .filter(|&d| d > 50.0 && d < 2000.0) // Plausible IOI range
        .collect();

    if iois.is_empty() {
        return TempoEstimate {
            bpm: 120.0,
            confidence: 0.0,
            period_ms: 500.0,
        };
    }

    let n_bins = ((MAX_IOI - MIN_IOI) / BIN_SIZE) as usize;
    let mut histogram = vec![0u32; n_bins];

    for &ioi in &iois {
        let bin = ((ioi - MIN_IOI) / BIN_SIZE) as usize;
        if bin < n_bins {
            histogram[bin] += 1;
        }
    }

    // Find peak bin
    let (peak_bin, &peak_count) = histogram
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .unwrap_or((0, &0));

    let period_ms = MIN_IOI + (peak_bin as f32 + 0.5) * BIN_SIZE;
    let bpm = 60_000.0 / period_ms;
    let confidence = if iois.is_empty() {
        0.0
    } else {
        (peak_count as f32 / iois.len() as f32).min(1.0)
    };

    TempoEstimate {
        bpm,
        confidence,
        period_ms,
    }
}

/// Beat tracker that projects evenly-spaced beats from onset data.
pub struct BeatTracker {
    /// Tempo estimate used for beat projection.
    pub tempo_estimate: TempoEstimate,
}

impl BeatTracker {
    /// Create a new `BeatTracker` with the given tempo estimate.
    #[must_use]
    pub fn new(tempo_estimate: TempoEstimate) -> Self {
        Self { tempo_estimate }
    }

    /// Project beat positions from the onset list using the stored tempo.
    ///
    /// Beats are evenly spaced starting from the first onset.
    #[must_use]
    pub fn track(&self, onsets_ms: &[u64]) -> Vec<BeatEvent> {
        if onsets_ms.is_empty() || self.tempo_estimate.period_ms <= 0.0 {
            return Vec::new();
        }

        let first_ms = onsets_ms[0] as f32;
        // Find the last onset to bound our projection
        let last_ms = onsets_ms[onsets_ms.len() - 1] as f32;

        let mut beats = Vec::new();
        let mut t = first_ms;
        let mut beat_number: u32 = 1;

        while t <= last_ms + self.tempo_estimate.period_ms {
            beats.push(BeatEvent {
                time_ms: t as u64,
                confidence: self.tempo_estimate.confidence,
                beat_number,
            });
            t += self.tempo_estimate.period_ms;
            beat_number += 1;
        }

        beats
    }
}

/// Analyzes accent patterns in a beat sequence to estimate musical meter.
pub struct MeterAnalyzer;

impl MeterAnalyzer {
    /// Estimate the meter (3 for triple, 4 for quadruple) from beat accents.
    ///
    /// Uses the confidence values as proxy for accent strength. If the signal is
    /// insufficient, defaults to 4.
    #[must_use]
    pub fn estimate_meter(beats: &[BeatEvent]) -> u8 {
        if beats.len() < 6 {
            return 4;
        }

        // Score for 3/4 and 4/4 by checking periodically stronger beats
        let score_3: f32 = (0..beats.len())
            .filter(|&i| i % 3 == 0)
            .map(|i| beats[i].confidence)
            .sum();

        let score_4: f32 = (0..beats.len())
            .filter(|&i| i % 4 == 0)
            .map(|i| beats[i].confidence)
            .sum();

        // Normalize by number of candidates
        let count_3 = beats.len().div_ceil(3).max(1) as f32;
        let count_4 = beats.len().div_ceil(4).max(1) as f32;

        let avg_3 = score_3 / count_3;
        let avg_4 = score_4 / count_4;

        if avg_3 > avg_4 {
            3
        } else {
            4
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beat_event_is_downbeat_4_4() {
        let beat1 = BeatEvent {
            time_ms: 0,
            confidence: 1.0,
            beat_number: 1,
        };
        let beat2 = BeatEvent {
            time_ms: 500,
            confidence: 0.8,
            beat_number: 2,
        };
        let beat5 = BeatEvent {
            time_ms: 2000,
            confidence: 1.0,
            beat_number: 5,
        };
        assert!(beat1.is_downbeat(4));
        assert!(!beat2.is_downbeat(4));
        assert!(beat5.is_downbeat(4));
    }

    #[test]
    fn test_beat_event_is_downbeat_3_4() {
        let beat1 = BeatEvent {
            time_ms: 0,
            confidence: 1.0,
            beat_number: 1,
        };
        let beat4 = BeatEvent {
            time_ms: 1500,
            confidence: 1.0,
            beat_number: 4,
        };
        let beat3 = BeatEvent {
            time_ms: 1000,
            confidence: 0.5,
            beat_number: 3,
        };
        assert!(beat1.is_downbeat(3));
        assert!(beat4.is_downbeat(3));
        assert!(!beat3.is_downbeat(3));
    }

    #[test]
    fn test_beat_event_zero_meter() {
        let beat = BeatEvent {
            time_ms: 0,
            confidence: 1.0,
            beat_number: 1,
        };
        assert!(!beat.is_downbeat(0));
    }

    #[test]
    fn test_tempo_estimate_is_reliable() {
        let reliable = TempoEstimate {
            bpm: 120.0,
            confidence: 0.8,
            period_ms: 500.0,
        };
        assert!(reliable.is_reliable());

        let low_conf = TempoEstimate {
            bpm: 120.0,
            confidence: 0.4,
            period_ms: 500.0,
        };
        assert!(!low_conf.is_reliable());

        let too_slow = TempoEstimate {
            bpm: 10.0,
            confidence: 0.9,
            period_ms: 6000.0,
        };
        assert!(!too_slow.is_reliable());

        let too_fast = TempoEstimate {
            bpm: 400.0,
            confidence: 0.9,
            period_ms: 150.0,
        };
        assert!(!too_fast.is_reliable());
    }

    #[test]
    fn test_tempo_estimate_is_reliable_boundary() {
        let boundary = TempoEstimate {
            bpm: 120.0,
            confidence: 0.5,
            period_ms: 500.0,
        };
        // Exactly 0.5 is NOT reliable (> 0.5 required)
        assert!(!boundary.is_reliable());
    }

    #[test]
    fn test_estimate_tempo_insufficient_onsets() {
        let result = estimate_tempo_from_onsets(&[100]);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_estimate_tempo_regular_onsets() {
        // 120 BPM = 500 ms period
        let onsets: Vec<u64> = (0..10).map(|i| i * 500).collect();
        let result = estimate_tempo_from_onsets(&onsets);
        // Should be close to 120 BPM
        assert!(result.bpm > 100.0 && result.bpm < 140.0);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_estimate_tempo_empty() {
        let result = estimate_tempo_from_onsets(&[]);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_beat_tracker_empty_onsets() {
        let tempo = TempoEstimate {
            bpm: 120.0,
            confidence: 0.9,
            period_ms: 500.0,
        };
        let tracker = BeatTracker::new(tempo);
        assert!(tracker.track(&[]).is_empty());
    }

    #[test]
    fn test_beat_tracker_projects_beats() {
        let tempo = TempoEstimate {
            bpm: 120.0,
            confidence: 0.9,
            period_ms: 500.0,
        };
        let tracker = BeatTracker::new(tempo);
        let onsets = vec![0u64, 500, 1000, 1500, 2000];
        let beats = tracker.track(&onsets);
        assert!(!beats.is_empty());
        assert_eq!(beats[0].time_ms, 0);
        // Second beat should be ~500 ms
        assert!(beats[1].time_ms >= 490 && beats[1].time_ms <= 510);
    }

    #[test]
    fn test_beat_tracker_beat_numbers() {
        let tempo = TempoEstimate {
            bpm: 120.0,
            confidence: 0.8,
            period_ms: 500.0,
        };
        let tracker = BeatTracker::new(tempo);
        let onsets = vec![0u64, 500, 1000, 1500];
        let beats = tracker.track(&onsets);
        for (i, beat) in beats.iter().enumerate() {
            assert_eq!(beat.beat_number, (i + 1) as u32);
        }
    }

    #[test]
    fn test_meter_analyzer_defaults_to_4() {
        let beats: Vec<BeatEvent> = (1..=4)
            .map(|i| BeatEvent {
                time_ms: i * 500,
                confidence: 0.7,
                beat_number: i as u32,
            })
            .collect();
        // Fewer than 6 beats defaults to 4
        let meter = MeterAnalyzer::estimate_meter(&beats);
        assert_eq!(meter, 4);
    }

    #[test]
    fn test_meter_analyzer_returns_3_or_4() {
        let beats: Vec<BeatEvent> = (1..=12)
            .map(|i| BeatEvent {
                time_ms: i * 500,
                confidence: 0.7,
                beat_number: i as u32,
            })
            .collect();
        let meter = MeterAnalyzer::estimate_meter(&beats);
        assert!(meter == 3 || meter == 4);
    }
}

#![allow(dead_code)]
//! Automated tempo and rhythm detection for music-driven video editing.
//!
//! This module analyses audio energy to detect tempo (BPM), beat positions,
//! and rhythmic structure so that automated edits can be synchronised to the
//! musical pulse of the soundtrack.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Strategy for computing onset-detection energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnsetStrategy {
    /// Use the raw energy envelope.
    Energy,
    /// Use spectral-flux (difference between successive spectra).
    SpectralFlux,
    /// Use high-frequency content weighting.
    HighFrequencyContent,
}

/// Configuration for the tempo detector.
#[derive(Debug, Clone)]
pub struct TempoDetectConfig {
    /// Minimum BPM to consider.
    pub min_bpm: f64,
    /// Maximum BPM to consider.
    pub max_bpm: f64,
    /// Hop size in samples for the onset function.
    pub hop_size: usize,
    /// Onset-detection strategy.
    pub onset_strategy: OnsetStrategy,
    /// Smoothing window length for the onset function.
    pub smooth_window: usize,
    /// Peak-picking threshold (0..1).
    pub peak_threshold: f64,
}

impl Default for TempoDetectConfig {
    fn default() -> Self {
        Self {
            min_bpm: 60.0,
            max_bpm: 200.0,
            hop_size: 512,
            onset_strategy: OnsetStrategy::Energy,
            smooth_window: 5,
            peak_threshold: 0.35,
        }
    }
}

impl TempoDetectConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the BPM range.
    pub fn with_bpm_range(mut self, min: f64, max: f64) -> Self {
        self.min_bpm = min;
        self.max_bpm = max;
        self
    }

    /// Set the hop size.
    pub fn with_hop_size(mut self, hop: usize) -> Self {
        self.hop_size = hop;
        self
    }

    /// Set the onset strategy.
    pub fn with_onset_strategy(mut self, s: OnsetStrategy) -> Self {
        self.onset_strategy = s;
        self
    }

    /// Set the peak threshold.
    pub fn with_peak_threshold(mut self, t: f64) -> Self {
        self.peak_threshold = t;
        self
    }
}

/// A single detected beat event.
#[derive(Debug, Clone, PartialEq)]
pub struct BeatEvent {
    /// Sample offset where the beat occurs.
    pub sample_offset: usize,
    /// Time in seconds.
    pub time_secs: f64,
    /// Strength / confidence of the beat (0..1).
    pub strength: f64,
}

/// Result of a full tempo analysis.
#[derive(Debug, Clone)]
pub struct TempoAnalysis {
    /// Estimated BPM.
    pub bpm: f64,
    /// Confidence of the BPM estimate (0..1).
    pub confidence: f64,
    /// Detected beat positions.
    pub beats: Vec<BeatEvent>,
    /// Onset-detection function values.
    pub onset_values: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Automated tempo / beat detector.
#[derive(Debug, Clone)]
pub struct TempoDetector {
    /// Configuration.
    config: TempoDetectConfig,
}

impl TempoDetector {
    /// Create a new detector with the given config.
    pub fn new(config: TempoDetectConfig) -> Self {
        Self { config }
    }

    /// Analyse mono audio samples at the given sample rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyse(&self, samples: &[f32], sample_rate: u32) -> TempoAnalysis {
        let onset = self.compute_onset(samples);
        let smoothed = smooth(&onset, self.config.smooth_window);
        let peaks = self.pick_peaks(&smoothed);
        let beat_events = self.peaks_to_beats(&peaks, sample_rate);
        let bpm = self.estimate_bpm(&beat_events);
        let confidence = self.estimate_confidence(&beat_events, bpm);

        TempoAnalysis {
            bpm,
            confidence,
            beats: beat_events,
            onset_values: smoothed,
        }
    }

    // -- internal helpers ---------------------------------------------------

    #[allow(clippy::cast_precision_loss)]
    fn compute_onset(&self, samples: &[f32]) -> Vec<f64> {
        let hop = self.config.hop_size.max(1);
        let n_frames = samples.len() / hop;
        let mut onset = Vec::with_capacity(n_frames);

        match self.config.onset_strategy {
            OnsetStrategy::Energy => {
                for i in 0..n_frames {
                    let start = i * hop;
                    let end = (start + hop).min(samples.len());
                    let energy: f64 = samples[start..end]
                        .iter()
                        .map(|&s| (s as f64) * (s as f64))
                        .sum();
                    onset.push(energy / hop as f64);
                }
            }
            OnsetStrategy::SpectralFlux => {
                let mut prev_energy = 0.0_f64;
                for i in 0..n_frames {
                    let start = i * hop;
                    let end = (start + hop).min(samples.len());
                    let energy: f64 = samples[start..end]
                        .iter()
                        .map(|&s| (s as f64) * (s as f64))
                        .sum();
                    let flux = (energy - prev_energy).max(0.0);
                    onset.push(flux / hop as f64);
                    prev_energy = energy;
                }
            }
            OnsetStrategy::HighFrequencyContent => {
                for i in 0..n_frames {
                    let start = i * hop;
                    let end = (start + hop).min(samples.len());
                    let hfc: f64 = samples[start..end]
                        .iter()
                        .enumerate()
                        .map(|(k, &s)| (k as f64 + 1.0) * (s as f64).abs())
                        .sum();
                    onset.push(hfc / hop as f64);
                }
            }
        }
        onset
    }

    fn pick_peaks(&self, onset: &[f64]) -> Vec<(usize, f64)> {
        if onset.is_empty() {
            return Vec::new();
        }
        let max_val = onset.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_val <= 0.0 {
            return Vec::new();
        }
        let threshold = self.config.peak_threshold * max_val;
        let mut peaks = Vec::new();
        for i in 1..onset.len().saturating_sub(1) {
            if onset[i] > onset[i - 1] && onset[i] >= onset[i + 1] && onset[i] >= threshold {
                peaks.push((i, onset[i] / max_val));
            }
        }
        peaks
    }

    #[allow(clippy::cast_precision_loss)]
    fn peaks_to_beats(&self, peaks: &[(usize, f64)], sample_rate: u32) -> Vec<BeatEvent> {
        let hop = self.config.hop_size.max(1);
        peaks
            .iter()
            .map(|&(idx, strength)| {
                let sample_offset = idx * hop;
                let time_secs = sample_offset as f64 / sample_rate as f64;
                BeatEvent {
                    sample_offset,
                    time_secs,
                    strength,
                }
            })
            .collect()
    }

    #[allow(clippy::cast_precision_loss)]
    fn estimate_bpm(&self, beats: &[BeatEvent]) -> f64 {
        if beats.len() < 2 {
            return self.config.min_bpm;
        }
        let mut intervals: Vec<f64> = Vec::new();
        for w in beats.windows(2) {
            let dt = w[1].time_secs - w[0].time_secs;
            if dt > 0.0 {
                intervals.push(dt);
            }
        }
        if intervals.is_empty() {
            return self.config.min_bpm;
        }
        intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = intervals[intervals.len() / 2];
        if median <= 0.0 {
            return self.config.min_bpm;
        }
        let bpm = 60.0 / median;
        bpm.clamp(self.config.min_bpm, self.config.max_bpm)
    }

    fn estimate_confidence(&self, beats: &[BeatEvent], bpm: f64) -> f64 {
        if beats.len() < 4 || bpm <= 0.0 {
            return 0.0;
        }
        let expected_interval = 60.0 / bpm;
        let mut errors: Vec<f64> = Vec::new();
        for w in beats.windows(2) {
            let dt = w[1].time_secs - w[0].time_secs;
            let err = ((dt - expected_interval) / expected_interval).abs();
            errors.push(err);
        }
        let mean_err: f64 = errors.iter().sum::<f64>() / errors.len() as f64;
        (1.0 - mean_err).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Simple box-car smoothing.
fn smooth(data: &[f64], window: usize) -> Vec<f64> {
    if window <= 1 || data.is_empty() {
        return data.to_vec();
    }
    let mut result = Vec::with_capacity(data.len());
    let mut ring = VecDeque::with_capacity(window);
    let mut sum = 0.0_f64;
    for &v in data {
        ring.push_back(v);
        sum += v;
        if ring.len() > window {
            sum -= ring.pop_front().unwrap_or(0.0);
        }
        result.push(sum / ring.len() as f64);
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

    fn make_clicks(bpm: f64, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        let interval = (60.0 / bpm * sample_rate as f64) as usize;
        let click_len = 256; // wider click so it dominates its hop
        let mut buf = vec![0.0f32; n];
        let mut pos = 0;
        while pos < n {
            let end = (pos + click_len).min(n);
            for s in &mut buf[pos..end] {
                *s = 1.0;
            }
            pos += interval;
        }
        buf
    }

    #[test]
    fn test_default_config() {
        let cfg = TempoDetectConfig::default();
        assert!((cfg.min_bpm - 60.0).abs() < f64::EPSILON);
        assert!((cfg.max_bpm - 200.0).abs() < f64::EPSILON);
        assert_eq!(cfg.hop_size, 512);
    }

    #[test]
    fn test_config_builder() {
        let cfg = TempoDetectConfig::new()
            .with_bpm_range(80.0, 180.0)
            .with_hop_size(256)
            .with_onset_strategy(OnsetStrategy::SpectralFlux)
            .with_peak_threshold(0.5);
        assert!((cfg.min_bpm - 80.0).abs() < f64::EPSILON);
        assert!((cfg.max_bpm - 180.0).abs() < f64::EPSILON);
        assert_eq!(cfg.hop_size, 256);
        assert_eq!(cfg.onset_strategy, OnsetStrategy::SpectralFlux);
        assert!((cfg.peak_threshold - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyse_silence() {
        let det = TempoDetector::new(TempoDetectConfig::default());
        let samples = vec![0.0f32; 44100];
        let result = det.analyse(&samples, 44100);
        assert!(result.beats.is_empty());
    }

    #[test]
    fn test_analyse_clicks() {
        let clicks = make_clicks(120.0, 44100, 4.0);
        let cfg = TempoDetectConfig::default()
            .with_peak_threshold(0.15)
            .with_onset_strategy(OnsetStrategy::SpectralFlux);
        let det = TempoDetector::new(cfg);
        let result = det.analyse(&clicks, 44100);
        assert!(result.bpm >= 60.0);
        assert!(result.bpm <= 200.0);
        assert!(!result.beats.is_empty());
    }

    #[test]
    fn test_beat_event_fields() {
        let ev = BeatEvent {
            sample_offset: 22050,
            time_secs: 0.5,
            strength: 0.9,
        };
        assert_eq!(ev.sample_offset, 22050);
        assert!((ev.time_secs - 0.5).abs() < f64::EPSILON);
        assert!((ev.strength - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_smooth_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let out = smooth(&data, 1);
        assert_eq!(out, data);
    }

    #[test]
    fn test_smooth_window() {
        let data = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let out = smooth(&data, 3);
        assert_eq!(out.len(), 5);
        // Middle value should be averaged with neighbours
        assert!(out[2] > 0.0);
        assert!(out[2] < 10.0);
    }

    #[test]
    fn test_onset_strategies() {
        let sine = make_sine(440.0, 44100, 1.0);
        for strategy in &[
            OnsetStrategy::Energy,
            OnsetStrategy::SpectralFlux,
            OnsetStrategy::HighFrequencyContent,
        ] {
            let cfg = TempoDetectConfig::default().with_onset_strategy(*strategy);
            let det = TempoDetector::new(cfg);
            let result = det.analyse(&sine, 44100);
            assert!(!result.onset_values.is_empty());
        }
    }

    #[test]
    fn test_empty_input() {
        let det = TempoDetector::new(TempoDetectConfig::default());
        let result = det.analyse(&[], 44100);
        assert!(result.beats.is_empty());
        assert!(result.onset_values.is_empty());
    }

    #[test]
    fn test_confidence_range() {
        let clicks = make_clicks(120.0, 44100, 4.0);
        let det = TempoDetector::new(TempoDetectConfig::default());
        let result = det.analyse(&clicks, 44100);
        assert!(result.confidence >= 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_bpm_clamped_to_range() {
        let cfg = TempoDetectConfig::default().with_bpm_range(100.0, 150.0);
        let det = TempoDetector::new(cfg);
        let clicks = make_clicks(60.0, 44100, 4.0);
        let result = det.analyse(&clicks, 44100);
        assert!(result.bpm >= 100.0);
        assert!(result.bpm <= 150.0);
    }

    #[test]
    fn test_tempo_analysis_clone() {
        let analysis = TempoAnalysis {
            bpm: 120.0,
            confidence: 0.8,
            beats: vec![BeatEvent {
                sample_offset: 0,
                time_secs: 0.0,
                strength: 1.0,
            }],
            onset_values: vec![0.5],
        };
        let cloned = analysis.clone();
        assert!((cloned.bpm - 120.0).abs() < f64::EPSILON);
        assert_eq!(cloned.beats.len(), 1);
    }
}

#![allow(dead_code)]

//! Onset peak detection with adaptive thresholds for music information retrieval.
//!
//! Detects note onsets in audio by analysing spectral flux with an adaptive
//! threshold derived from a sliding median and a configurable sensitivity.

/// Default hop size in samples.
const DEFAULT_HOP: usize = 512;

/// Default window length for adaptive threshold (in frames).
const DEFAULT_MEDIAN_WINDOW: usize = 11;

/// Minimum peak distance in frames to suppress duplicates.
const DEFAULT_MIN_PEAK_DISTANCE: usize = 4;

/// A detected onset peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnsetPeak {
    /// Frame index where the onset was detected.
    pub frame: usize,
    /// Time in seconds of the onset.
    pub time_secs: f64,
    /// Strength of the onset (spectral flux value).
    pub strength: f64,
}

impl OnsetPeak {
    /// Create a new onset peak.
    #[must_use]
    pub fn new(frame: usize, time_secs: f64, strength: f64) -> Self {
        Self {
            frame,
            time_secs,
            strength,
        }
    }
}

/// Configuration for onset peak detection.
#[derive(Debug, Clone)]
pub struct OnsetPeakConfig {
    /// Hop size in samples.
    pub hop_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Sensitivity multiplier above the adaptive threshold (larger = fewer detections).
    pub sensitivity: f64,
    /// Sliding median window length in frames.
    pub median_window: usize,
    /// Minimum distance between peaks in frames.
    pub min_peak_distance: usize,
}

impl Default for OnsetPeakConfig {
    fn default() -> Self {
        Self {
            hop_size: DEFAULT_HOP,
            sample_rate: 44100.0,
            sensitivity: 1.5,
            median_window: DEFAULT_MEDIAN_WINDOW,
            min_peak_distance: DEFAULT_MIN_PEAK_DISTANCE,
        }
    }
}

/// Onset peak detector.
#[derive(Debug)]
pub struct OnsetPeakDetector {
    config: OnsetPeakConfig,
}

impl OnsetPeakDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: OnsetPeakConfig) -> Self {
        Self { config }
    }

    /// Create a detector with default configuration and a given sample rate.
    #[must_use]
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self {
            config: OnsetPeakConfig {
                sample_rate,
                ..OnsetPeakConfig::default()
            },
        }
    }

    /// Compute spectral flux from raw audio samples.
    ///
    /// Spectral flux is approximated here as the sum of positive differences
    /// between consecutive frame energies computed via a simple windowed RMS.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn spectral_flux(&self, samples: &[f32]) -> Vec<f64> {
        let hop = self.config.hop_size.max(1);
        let n_frames = if samples.len() >= hop {
            (samples.len() - hop) / hop + 1
        } else if samples.is_empty() {
            0
        } else {
            1
        };

        // Compute per-frame energy
        let mut energies = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let start = i * hop;
            let end = (start + hop).min(samples.len());
            let energy: f64 = samples[start..end]
                .iter()
                .map(|&s| f64::from(s) * f64::from(s))
                .sum();
            energies.push(energy / (end - start) as f64);
        }

        // Spectral flux: positive first-order difference
        let mut flux = Vec::with_capacity(n_frames);
        if !energies.is_empty() {
            flux.push(energies[0]);
        }
        for i in 1..energies.len() {
            let diff = energies[i] - energies[i - 1];
            flux.push(if diff > 0.0 { diff } else { 0.0 });
        }
        flux
    }

    /// Compute adaptive threshold using a sliding median.
    #[must_use]
    pub fn adaptive_threshold(&self, flux: &[f64]) -> Vec<f64> {
        let half_w = self.config.median_window / 2;
        let mut threshold = Vec::with_capacity(flux.len());
        for i in 0..flux.len() {
            let start = if i >= half_w { i - half_w } else { 0 };
            let end = (i + half_w + 1).min(flux.len());
            let mut window: Vec<f64> = flux[start..end].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if window.is_empty() {
                0.0
            } else {
                window[window.len() / 2]
            };
            threshold.push(median * self.config.sensitivity);
        }
        threshold
    }

    /// Detect onset peaks from raw audio samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, samples: &[f32]) -> Vec<OnsetPeak> {
        let flux = self.spectral_flux(samples);
        let threshold = self.adaptive_threshold(&flux);
        let hop = self.config.hop_size.max(1);
        let sr = f64::from(self.config.sample_rate);

        let mut peaks = Vec::new();
        let mut last_peak_frame: Option<usize> = None;

        for i in 1..flux.len().saturating_sub(1) {
            // Local maximum check
            if flux[i] > flux[i - 1] && flux[i] >= flux[i + 1] && flux[i] > threshold[i] {
                // Minimum distance check
                if let Some(last) = last_peak_frame {
                    if i - last < self.config.min_peak_distance {
                        continue;
                    }
                }
                let time = (i * hop) as f64 / sr;
                peaks.push(OnsetPeak::new(i, time, flux[i]));
                last_peak_frame = Some(i);
            }
        }
        peaks
    }

    /// Return the number of frames for a given sample count.
    #[must_use]
    pub fn frame_count(&self, sample_count: usize) -> usize {
        let hop = self.config.hop_size.max(1);
        if sample_count >= hop {
            (sample_count - hop) / hop + 1
        } else if sample_count == 0 {
            0
        } else {
            1
        }
    }

    /// Convert a frame index to time in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_to_time(&self, frame: usize) -> f64 {
        let hop = self.config.hop_size.max(1);
        (frame * hop) as f64 / f64::from(self.config.sample_rate)
    }
}

/// Utility: generate a simple sinusoidal test signal.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn generate_sine(freq: f32, sample_rate: f32, duration_secs: f32) -> Vec<f32> {
    let n = (sample_rate * duration_secs) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Utility: generate a click train (impulse at regular intervals).
#[must_use]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn generate_clicks(interval_secs: f32, sample_rate: f32, duration_secs: f32) -> Vec<f32> {
    let n = (sample_rate * duration_secs) as usize;
    let interval_samples = (sample_rate * interval_secs) as usize;
    let mut out = vec![0.0f32; n];
    if interval_samples == 0 {
        return out;
    }
    let mut pos = 0;
    while pos < n {
        out[pos] = 1.0;
        pos += interval_samples;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onset_peak_creation() {
        let peak = OnsetPeak::new(10, 0.116, 0.42);
        assert_eq!(peak.frame, 10);
        assert!((peak.time_secs - 0.116).abs() < 1e-9);
        assert!((peak.strength - 0.42).abs() < 1e-9);
    }

    #[test]
    fn test_config_default() {
        let cfg = OnsetPeakConfig::default();
        assert_eq!(cfg.hop_size, 512);
        assert!((cfg.sample_rate - 44100.0).abs() < f32::EPSILON);
        assert!((cfg.sensitivity - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spectral_flux_silence() {
        let det = OnsetPeakDetector::with_sample_rate(44100.0);
        let silence = vec![0.0f32; 44100];
        let flux = det.spectral_flux(&silence);
        assert!(!flux.is_empty());
        for &v in &flux {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_spectral_flux_impulse() {
        let det = OnsetPeakDetector::with_sample_rate(44100.0);
        let mut signal = vec![0.0f32; 4096];
        signal[512] = 1.0; // impulse at frame 1
        let flux = det.spectral_flux(&signal);
        // There should be a spike in flux
        let max_flux = flux.iter().copied().fold(0.0f64, f64::max);
        assert!(max_flux > 0.0);
    }

    #[test]
    fn test_adaptive_threshold_length() {
        let det = OnsetPeakDetector::with_sample_rate(44100.0);
        let flux = vec![0.1, 0.2, 0.5, 0.3, 0.1, 0.05, 0.6, 0.2];
        let thresh = det.adaptive_threshold(&flux);
        assert_eq!(thresh.len(), flux.len());
    }

    #[test]
    fn test_detect_silence_no_peaks() {
        let det = OnsetPeakDetector::with_sample_rate(44100.0);
        let silence = vec![0.0f32; 44100];
        let peaks = det.detect(&silence);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_detect_clicks() {
        let det = OnsetPeakDetector::new(OnsetPeakConfig {
            hop_size: 256,
            sample_rate: 44100.0,
            sensitivity: 1.2,
            median_window: 7,
            min_peak_distance: 4,
        });
        let clicks = generate_clicks(0.25, 44100.0, 2.0);
        let peaks = det.detect(&clicks);
        // Should detect multiple onsets
        assert!(!peaks.is_empty());
    }

    #[test]
    fn test_frame_count() {
        let det = OnsetPeakDetector::with_sample_rate(44100.0);
        let fc = det.frame_count(44100);
        assert!(fc > 0);
        assert_eq!(det.frame_count(0), 0);
    }

    #[test]
    fn test_frame_to_time() {
        let det = OnsetPeakDetector::with_sample_rate(44100.0);
        let t = det.frame_to_time(0);
        assert!((t - 0.0).abs() < 1e-9);
        let t1 = det.frame_to_time(1);
        assert!(t1 > 0.0);
    }

    #[test]
    fn test_generate_sine() {
        let sig = generate_sine(440.0, 44100.0, 0.1);
        assert_eq!(sig.len(), 4410);
        // Check bounded
        for &s in &sig {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_generate_clicks_interval() {
        let clicks = generate_clicks(0.5, 44100.0, 1.0);
        assert_eq!(clicks.len(), 44100);
        assert!((clicks[0] - 1.0).abs() < f32::EPSILON);
        assert!((clicks[22050] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_min_peak_distance_respected() {
        let det = OnsetPeakDetector::new(OnsetPeakConfig {
            hop_size: 64,
            sample_rate: 8000.0,
            sensitivity: 0.5,
            median_window: 5,
            min_peak_distance: 10,
        });
        let clicks = generate_clicks(0.01, 8000.0, 1.0);
        let peaks = det.detect(&clicks);
        for w in peaks.windows(2) {
            assert!(w[1].frame - w[0].frame >= 10);
        }
    }

    #[test]
    fn test_empty_input() {
        let det = OnsetPeakDetector::with_sample_rate(44100.0);
        let peaks = det.detect(&[]);
        assert!(peaks.is_empty());
    }
}

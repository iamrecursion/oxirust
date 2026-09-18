#![allow(dead_code)]
//! Frequency-domain vibration isolation and removal for video stabilization.
//!
//! Many stabilization scenarios involve high-frequency mechanical vibrations
//! (e.g., helicopter mounts, vehicle-mounted cameras, industrial machinery).
//! Standard smoothing filters can struggle with these periodic disturbances.
//! This module provides frequency-domain analysis and filtering to isolate
//! and remove specific vibration frequencies from motion trajectories.
//!
//! # Features
//!
//! - **Vibration detection**: Identify dominant vibration frequencies
//! - **Notch filtering**: Remove specific frequency bands from trajectories
//! - **Band-pass isolation**: Extract vibration components for analysis
//! - **Adaptive filtering**: Automatically detect and suppress vibrations
//! - **Multi-axis support**: Independent filtering for X, Y, and rotation

use std::f64::consts::PI;

/// Configuration for vibration isolation.
#[derive(Debug, Clone)]
pub struct VibrationConfig {
    /// Sample rate of the motion trajectory (typically = video FPS).
    pub sample_rate: f64,
    /// Minimum vibration frequency to detect (Hz).
    pub min_freq: f64,
    /// Maximum vibration frequency to detect (Hz).
    pub max_freq: f64,
    /// Notch filter bandwidth (Hz).
    pub notch_bandwidth: f64,
    /// Detection threshold for peak identification in frequency domain.
    pub detection_threshold: f64,
    /// Maximum number of vibration frequencies to suppress.
    pub max_notches: usize,
    /// Enable adaptive filtering.
    pub adaptive: bool,
}

impl Default for VibrationConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            min_freq: 2.0,
            max_freq: 14.0,
            notch_bandwidth: 0.5,
            detection_threshold: 3.0,
            max_notches: 5,
            adaptive: true,
        }
    }
}

impl VibrationConfig {
    /// Create a new vibration configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sample rate (video FPS).
    #[must_use]
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.max(1.0);
        self
    }

    /// Set the frequency range for vibration detection.
    #[must_use]
    pub fn with_freq_range(mut self, min: f64, max: f64) -> Self {
        self.min_freq = min.max(0.0);
        self.max_freq = max.max(min);
        self
    }

    /// Set the notch bandwidth.
    #[must_use]
    pub fn with_notch_bandwidth(mut self, bw: f64) -> Self {
        self.notch_bandwidth = bw.max(0.01);
        self
    }

    /// Set the maximum number of notch filters.
    #[must_use]
    pub const fn with_max_notches(mut self, n: usize) -> Self {
        self.max_notches = n;
        self
    }

    /// Enable or disable adaptive filtering.
    #[must_use]
    pub const fn with_adaptive(mut self, enable: bool) -> Self {
        self.adaptive = enable;
        self
    }
}

/// A detected vibration frequency component.
#[derive(Debug, Clone)]
pub struct VibrationPeak {
    /// Frequency in Hz.
    pub frequency: f64,
    /// Magnitude (power) at this frequency.
    pub magnitude: f64,
    /// Phase offset in radians.
    pub phase: f64,
    /// Signal-to-noise ratio of this peak.
    pub snr: f64,
}

/// Result of vibration analysis on a motion trajectory.
#[derive(Debug, Clone)]
pub struct VibrationAnalysis {
    /// Detected vibration peaks sorted by magnitude (descending).
    pub peaks: Vec<VibrationPeak>,
    /// Power spectral density of the input signal.
    pub psd: Vec<f64>,
    /// Frequency bins corresponding to PSD values.
    pub freq_bins: Vec<f64>,
    /// Total vibration power as a fraction of total signal power.
    pub vibration_ratio: f64,
}

/// Compute the Discrete Fourier Transform of a real-valued signal.
///
/// Returns (real, imaginary) component vectors.
#[allow(clippy::cast_precision_loss)]
fn dft(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    let mut real = vec![0.0; n];
    let mut imag = vec![0.0; n];
    for k in 0..n {
        for (t, &val) in signal.iter().enumerate() {
            let angle = 2.0 * PI * k as f64 * t as f64 / n as f64;
            real[k] += val * angle.cos();
            imag[k] -= val * angle.sin();
        }
    }
    (real, imag)
}

/// Compute the inverse DFT to reconstruct a signal.
#[allow(clippy::cast_precision_loss)]
fn idft(real: &[f64], imag: &[f64]) -> Vec<f64> {
    let n = real.len();
    let mut signal = vec![0.0; n];
    for t in 0..n {
        for k in 0..n {
            let angle = 2.0 * PI * k as f64 * t as f64 / n as f64;
            signal[t] += real[k] * angle.cos() - imag[k] * angle.sin();
        }
        signal[t] /= n as f64;
    }
    signal
}

/// Compute the power spectral density from DFT components.
fn power_spectrum(real: &[f64], imag: &[f64]) -> Vec<f64> {
    real.iter()
        .zip(imag.iter())
        .map(|(&r, &i)| r * r + i * i)
        .collect()
}

/// Vibration isolator for motion trajectory filtering.
#[derive(Debug)]
pub struct VibrationIsolator {
    /// Configuration.
    config: VibrationConfig,
}

impl VibrationIsolator {
    /// Create a new vibration isolator.
    #[must_use]
    pub fn new(config: VibrationConfig) -> Self {
        Self { config }
    }

    /// Create an isolator with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(VibrationConfig::default())
    }

    /// Analyze a motion trajectory for vibration components.
    ///
    /// The input signal is a 1D trajectory (e.g., x-axis camera motion over time).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn analyze(&self, signal: &[f64]) -> VibrationAnalysis {
        let n = signal.len();
        if n < 4 {
            return VibrationAnalysis {
                peaks: Vec::new(),
                psd: Vec::new(),
                freq_bins: Vec::new(),
                vibration_ratio: 0.0,
            };
        }

        let (real, imag) = dft(signal);
        let psd = power_spectrum(&real, &imag);

        // Frequency bins
        let freq_bins: Vec<f64> = (0..n)
            .map(|k| k as f64 * self.config.sample_rate / n as f64)
            .collect();

        // Only look at positive frequencies up to Nyquist
        let nyquist = n / 2;
        let total_power: f64 = psd[1..nyquist].iter().sum();
        let mean_power = if nyquist > 1 {
            total_power / (nyquist - 1) as f64
        } else {
            0.0
        };

        // Find peaks above threshold
        let mut peaks = Vec::new();
        // Skip peak detection when total power is negligible (e.g. constant signal)
        let min_total_power = 1e-10 * n as f64;
        for k in 1..nyquist {
            if total_power < min_total_power {
                break;
            }
            let freq = freq_bins[k];
            if freq < self.config.min_freq || freq > self.config.max_freq {
                continue;
            }
            // Simple peak detection: local maximum above threshold * mean
            let is_peak =
                (k == 1 || psd[k] > psd[k - 1]) && (k == nyquist - 1 || psd[k] > psd[k + 1]);
            let snr = if mean_power > 0.0 {
                psd[k] / mean_power
            } else {
                0.0
            };
            if is_peak && snr > self.config.detection_threshold {
                let phase = imag[k].atan2(real[k]);
                peaks.push(VibrationPeak {
                    frequency: freq,
                    magnitude: psd[k].sqrt(),
                    phase,
                    snr,
                });
            }
        }

        // Sort by magnitude descending, limit to max_notches
        peaks.sort_by(|a, b| {
            b.magnitude
                .partial_cmp(&a.magnitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        peaks.truncate(self.config.max_notches);

        // Compute vibration ratio
        let vibration_power: f64 = peaks.iter().map(|p| p.magnitude * p.magnitude).sum();
        let vibration_ratio = if total_power > 0.0 {
            vibration_power / total_power
        } else {
            0.0
        };

        VibrationAnalysis {
            peaks,
            psd,
            freq_bins,
            vibration_ratio,
        }
    }

    /// Apply a notch filter to remove a specific frequency from the signal.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn notch_filter(&self, signal: &[f64], center_freq: f64) -> Vec<f64> {
        let n = signal.len();
        if n < 4 {
            return signal.to_vec();
        }
        let (mut real, mut imag) = dft(signal);
        let bw = self.config.notch_bandwidth;

        for k in 0..n {
            let freq = k as f64 * self.config.sample_rate / n as f64;
            // Also handle the mirror frequency
            let mirror_freq = self.config.sample_rate - freq;
            let in_band = (freq - center_freq).abs() < bw || (mirror_freq - center_freq).abs() < bw;
            if in_band {
                real[k] = 0.0;
                imag[k] = 0.0;
            }
        }

        idft(&real, &imag)
    }

    /// Apply notch filters for all detected vibration frequencies.
    #[must_use]
    pub fn remove_vibrations(&self, signal: &[f64]) -> Vec<f64> {
        let analysis = self.analyze(signal);
        let mut filtered = signal.to_vec();
        for peak in &analysis.peaks {
            filtered = self.notch_filter(&filtered, peak.frequency);
        }
        filtered
    }

    /// Apply a band-pass filter to extract only the vibration components.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn extract_vibrations(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        if n < 4 {
            return vec![0.0; n];
        }
        let (mut real, mut imag) = dft(signal);

        for k in 0..n {
            let freq = k as f64 * self.config.sample_rate / n as f64;
            let mirror_freq = self.config.sample_rate - freq;
            let in_range = (freq >= self.config.min_freq && freq <= self.config.max_freq)
                || (mirror_freq >= self.config.min_freq && mirror_freq <= self.config.max_freq);
            if !in_range {
                real[k] = 0.0;
                imag[k] = 0.0;
            }
        }

        idft(&real, &imag)
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &VibrationConfig {
        &self.config
    }
}

/// Generate a synthetic vibration signal for testing.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn generate_test_signal(
    sample_rate: f64,
    duration_secs: f64,
    frequencies: &[(f64, f64)],
) -> Vec<f64> {
    let n = (sample_rate * duration_secs) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            frequencies
                .iter()
                .map(|&(freq, amp)| amp * (2.0 * PI * freq * t).sin())
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = VibrationConfig::default();
        assert!((cfg.sample_rate - 30.0).abs() < 1e-10);
        assert!((cfg.min_freq - 2.0).abs() < 1e-10);
        assert!((cfg.max_freq - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_builder() {
        let cfg = VibrationConfig::new()
            .with_sample_rate(60.0)
            .with_freq_range(1.0, 20.0)
            .with_notch_bandwidth(1.0)
            .with_max_notches(3)
            .with_adaptive(false);
        assert!((cfg.sample_rate - 60.0).abs() < 1e-10);
        assert!((cfg.min_freq - 1.0).abs() < 1e-10);
        assert!((cfg.max_freq - 20.0).abs() < 1e-10);
        assert!((cfg.notch_bandwidth - 1.0).abs() < 1e-10);
        assert_eq!(cfg.max_notches, 3);
        assert!(!cfg.adaptive);
    }

    #[test]
    fn test_dft_and_idft_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let (real, imag) = dft(&signal);
        let reconstructed = idft(&real, &imag);
        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 1e-8);
        }
    }

    #[test]
    fn test_generate_test_signal() {
        let signal = generate_test_signal(30.0, 1.0, &[(5.0, 1.0)]);
        assert_eq!(signal.len(), 30);
        // Signal should oscillate around zero
        let sum: f64 = signal.iter().sum();
        assert!((sum / signal.len() as f64).abs() < 0.2);
    }

    #[test]
    fn test_analyze_pure_sine() {
        let cfg = VibrationConfig::new()
            .with_sample_rate(100.0)
            .with_freq_range(1.0, 40.0);
        let isolator = VibrationIsolator::new(cfg);
        // Generate a 10Hz sine wave
        let signal = generate_test_signal(100.0, 1.0, &[(10.0, 5.0)]);
        let analysis = isolator.analyze(&signal);
        // Should find a peak near 10Hz
        assert!(!analysis.peaks.is_empty());
        let first_peak = &analysis.peaks[0];
        assert!((first_peak.frequency - 10.0).abs() < 2.0);
    }

    #[test]
    fn test_analyze_empty_signal() {
        let isolator = VibrationIsolator::with_defaults();
        let analysis = isolator.analyze(&[]);
        assert!(analysis.peaks.is_empty());
        assert!((analysis.vibration_ratio - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_notch_filter_removes_frequency() {
        let cfg = VibrationConfig::new()
            .with_sample_rate(100.0)
            .with_notch_bandwidth(2.0);
        let isolator = VibrationIsolator::new(cfg);
        // Mixed signal: 5Hz + 10Hz
        let signal = generate_test_signal(100.0, 1.0, &[(5.0, 3.0), (10.0, 3.0)]);
        let filtered = isolator.notch_filter(&signal, 10.0);
        // Filtered signal should have less power than original
        let orig_power: f64 = signal.iter().map(|x| x * x).sum();
        let filt_power: f64 = filtered.iter().map(|x| x * x).sum();
        assert!(filt_power < orig_power);
    }

    #[test]
    fn test_remove_vibrations() {
        let cfg = VibrationConfig::new()
            .with_sample_rate(100.0)
            .with_freq_range(8.0, 15.0)
            .with_notch_bandwidth(2.0);
        let isolator = VibrationIsolator::new(cfg);
        // Low-frequency camera motion + high-freq vibration
        let signal = generate_test_signal(100.0, 1.0, &[(2.0, 5.0), (12.0, 3.0)]);
        let filtered = isolator.remove_vibrations(&signal);
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_extract_vibrations() {
        let cfg = VibrationConfig::new()
            .with_sample_rate(100.0)
            .with_freq_range(8.0, 15.0);
        let isolator = VibrationIsolator::new(cfg);
        let signal = generate_test_signal(100.0, 1.0, &[(2.0, 5.0), (12.0, 3.0)]);
        let vibrations = isolator.extract_vibrations(&signal);
        assert_eq!(vibrations.len(), signal.len());
        // Extracted vibrations should have less power than original
        let orig_power: f64 = signal.iter().map(|x| x * x).sum();
        let vib_power: f64 = vibrations.iter().map(|x| x * x).sum();
        assert!(vib_power < orig_power);
    }

    #[test]
    fn test_power_spectrum() {
        let real = vec![1.0, 0.0, -1.0, 0.0];
        let imag = vec![0.0, 1.0, 0.0, -1.0];
        let ps = power_spectrum(&real, &imag);
        assert!((ps[0] - 1.0).abs() < 1e-10);
        assert!((ps[1] - 1.0).abs() < 1e-10);
        assert!((ps[2] - 1.0).abs() < 1e-10);
        assert!((ps[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_short_signal_analysis() {
        let isolator = VibrationIsolator::with_defaults();
        let signal = vec![1.0, 2.0];
        let analysis = isolator.analyze(&signal);
        assert!(analysis.peaks.is_empty());
    }

    #[test]
    fn test_constant_signal() {
        let cfg = VibrationConfig::new().with_sample_rate(30.0);
        let isolator = VibrationIsolator::new(cfg);
        let signal = vec![5.0; 30];
        let analysis = isolator.analyze(&signal);
        // A constant signal has no vibrations
        assert!(analysis.peaks.is_empty());
    }
}

#![allow(dead_code)]
//! Harmonic enhancement and reconstruction for degraded audio.
//!
//! This module provides tools for recovering lost harmonics in audio that has
//! been degraded by bandwidth limiting, heavy compression, or analog signal
//! loss. Techniques include harmonic exciter synthesis, spectral envelope
//! matching, and overtone reconstruction.

use std::f64::consts::PI;

/// Type of harmonic enhancement algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnhanceMode {
    /// Generate new harmonics via non-linear waveshaping.
    Waveshaping,
    /// Reconstruct missing harmonics from detected fundamental.
    OvertoneReconstruction,
    /// Apply spectral tilt correction to brighten dull recordings.
    SpectralTilt,
    /// Subtle exciter that adds presence via even harmonics.
    EvenHarmonicExciter,
}

/// Configuration for harmonic enhancement.
#[derive(Debug, Clone)]
pub struct HarmonicEnhanceConfig {
    /// Enhancement mode to use.
    pub mode: EnhanceMode,
    /// Amount of enhancement (0.0..1.0).
    pub amount: f32,
    /// High-pass frequency for the exciter band in Hz.
    pub exciter_freq_hz: f32,
    /// Number of overtones to reconstruct.
    pub overtone_count: usize,
    /// Dry/wet mix (0.0 = fully dry, 1.0 = fully wet).
    pub mix: f32,
    /// Maximum number of harmonics to generate.
    pub max_harmonics: usize,
}

impl Default for HarmonicEnhanceConfig {
    fn default() -> Self {
        Self {
            mode: EnhanceMode::Waveshaping,
            amount: 0.3,
            exciter_freq_hz: 3000.0,
            overtone_count: 4,
            mix: 0.5,
            max_harmonics: 8,
        }
    }
}

/// Result of harmonic analysis on a signal segment.
#[derive(Debug, Clone)]
pub struct HarmonicAnalysis {
    /// Detected fundamental frequency in Hz (0 if undetected).
    pub fundamental_hz: f32,
    /// Relative strength of each detected harmonic (index 0 = fundamental).
    pub harmonic_strengths: Vec<f32>,
    /// Spectral centroid of the analyzed segment.
    pub spectral_centroid: f32,
    /// Spectral rolloff frequency.
    pub spectral_rolloff: f32,
}

/// Harmonic enhancer processor.
#[derive(Debug, Clone)]
pub struct HarmonicEnhancer {
    /// Configuration.
    config: HarmonicEnhanceConfig,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// High-pass filter coefficient for exciter band.
    hp_coeff: f32,
}

impl HarmonicEnhancer {
    /// Create a new harmonic enhancer.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(config: HarmonicEnhanceConfig, sample_rate: u32) -> Self {
        let hp_coeff = Self::compute_hp_coeff(config.exciter_freq_hz, sample_rate);
        Self {
            config,
            sample_rate,
            hp_coeff,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(sample_rate: u32) -> Self {
        Self::new(HarmonicEnhanceConfig::default(), sample_rate)
    }

    /// Compute single-pole high-pass filter coefficient.
    #[allow(clippy::cast_precision_loss)]
    fn compute_hp_coeff(freq_hz: f32, sample_rate: u32) -> f32 {
        let omega = 2.0 * PI * freq_hz as f64 / sample_rate as f64;
        (1.0 / (1.0 + omega)) as f32
    }

    /// Analyze harmonics in a signal segment.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, samples: &[f32]) -> HarmonicAnalysis {
        if samples.is_empty() {
            return HarmonicAnalysis {
                fundamental_hz: 0.0,
                harmonic_strengths: Vec::new(),
                spectral_centroid: 0.0,
                spectral_rolloff: 0.0,
            };
        }

        // Simple autocorrelation-based pitch detection
        let fundamental_hz = self.detect_pitch(samples);

        // Estimate harmonic strengths
        let harmonic_strengths = self.estimate_harmonics(samples, fundamental_hz);

        // Compute spectral centroid (energy-weighted average frequency)
        let spectral_centroid = self.compute_spectral_centroid(samples);
        let spectral_rolloff = spectral_centroid * 2.5; // rough approximation

        HarmonicAnalysis {
            fundamental_hz,
            harmonic_strengths,
            spectral_centroid,
            spectral_rolloff,
        }
    }

    /// Detect pitch using zero-crossing rate (simplified).
    #[allow(clippy::cast_precision_loss)]
    fn detect_pitch(&self, samples: &[f32]) -> f32 {
        if samples.len() < 4 {
            return 0.0;
        }
        let mut zero_crossings = 0_usize;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }
        let duration_s = samples.len() as f64 / self.sample_rate as f64;
        if duration_s > 0.0 {
            (zero_crossings as f64 / (2.0 * duration_s)) as f32
        } else {
            0.0
        }
    }

    /// Estimate harmonic strengths relative to fundamental.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_harmonics(&self, samples: &[f32], fundamental: f32) -> Vec<f32> {
        if fundamental <= 0.0 || samples.is_empty() {
            return Vec::new();
        }
        let n_harmonics = self.config.max_harmonics.min(8);
        let mut strengths = Vec::with_capacity(n_harmonics);

        for h in 1..=n_harmonics {
            let freq = fundamental * h as f32;
            if freq > self.sample_rate as f32 / 2.0 {
                break;
            }
            // Compute power at this frequency using Goertzel-like calculation
            let omega = 2.0 * PI * freq as f64 / self.sample_rate as f64;
            let mut real_sum = 0.0_f64;
            let mut imag_sum = 0.0_f64;
            for (n, &s) in samples.iter().enumerate() {
                real_sum += s as f64 * (omega * n as f64).cos();
                imag_sum += s as f64 * (omega * n as f64).sin();
            }
            let power = ((real_sum * real_sum + imag_sum * imag_sum) / samples.len() as f64).sqrt();
            strengths.push(power as f32);
        }

        // Normalize to the fundamental
        if let Some(&first) = strengths.first() {
            if first > 0.0 {
                for s in &mut strengths {
                    *s /= first;
                }
            }
        }
        strengths
    }

    /// Compute spectral centroid of the signal.
    #[allow(clippy::cast_precision_loss)]
    fn compute_spectral_centroid(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let energy: f32 = samples.iter().map(|s| s * s).sum();
        if energy < 1e-10 {
            return 0.0;
        }
        // Weighted average based on sample differences (approximation)
        let mut weighted_sum = 0.0_f64;
        let mut total_weight = 0.0_f64;
        for i in 1..samples.len() {
            let diff = (samples[i] - samples[i - 1]).abs() as f64;
            weighted_sum += diff * i as f64;
            total_weight += diff;
        }
        if total_weight > 0.0 {
            let normalized = weighted_sum / total_weight;
            (normalized * self.sample_rate as f64 / samples.len() as f64) as f32
        } else {
            0.0
        }
    }

    /// Process the signal with harmonic enhancement.
    pub fn process(&self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        match self.config.mode {
            EnhanceMode::Waveshaping => self.waveshape(samples),
            EnhanceMode::OvertoneReconstruction => self.reconstruct_overtones(samples),
            EnhanceMode::SpectralTilt => self.correct_spectral_tilt(samples),
            EnhanceMode::EvenHarmonicExciter => self.even_harmonic_excite(samples),
        }
    }

    /// Apply non-linear waveshaping for harmonic generation.
    fn waveshape(&self, samples: &[f32]) -> Vec<f32> {
        let amount = self.config.amount;
        let mix = self.config.mix;
        samples
            .iter()
            .map(|&s| {
                // Soft clipping waveshaper: tanh-like
                let shaped = (s * (1.0 + amount * 4.0)).tanh();
                let wet = shaped;
                let mixed = s * (1.0 - mix) + wet * mix;
                mixed.clamp(-1.0, 1.0)
            })
            .collect()
    }

    /// Reconstruct missing overtones from detected fundamental.
    #[allow(clippy::cast_precision_loss)]
    fn reconstruct_overtones(&self, samples: &[f32]) -> Vec<f32> {
        let fundamental = self.detect_pitch(samples);
        if fundamental <= 0.0 {
            return samples.to_vec();
        }

        let mix = self.config.mix * self.config.amount;
        let mut output = samples.to_vec();
        let sr = self.sample_rate as f64;

        for h in 2..=(self.config.overtone_count + 1) {
            let freq = fundamental as f64 * h as f64;
            if freq > sr / 2.0 {
                break;
            }
            let omega = 2.0 * PI * freq / sr;
            // Amplitude decreases with harmonic number
            let amp = mix / h as f32;
            for (i, sample) in output.iter_mut().enumerate() {
                let harmonic = amp * (omega * i as f64).sin() as f32;
                *sample = (*sample + harmonic).clamp(-1.0, 1.0);
            }
        }
        output
    }

    /// Apply spectral tilt correction to brighten the signal.
    fn correct_spectral_tilt(&self, samples: &[f32]) -> Vec<f32> {
        let amount = self.config.amount;
        let mix = self.config.mix;
        let mut output = Vec::with_capacity(samples.len());
        let mut prev = 0.0_f32;
        let mut prev_hp = 0.0_f32;

        for &s in samples {
            // Simple high-pass to extract high-frequency content
            let hp = self.hp_coeff * (prev_hp + s - prev);
            prev = s;
            prev_hp = hp;

            // Add boosted high-frequency content
            let enhanced = s + hp * amount * 2.0;
            let mixed = s * (1.0 - mix) + enhanced * mix;
            output.push(mixed.clamp(-1.0, 1.0));
        }
        output
    }

    /// Apply even harmonic exciter.
    fn even_harmonic_excite(&self, samples: &[f32]) -> Vec<f32> {
        let amount = self.config.amount;
        let mix = self.config.mix;
        samples
            .iter()
            .map(|&s| {
                // Even harmonics via half-wave rectification + soft clip
                let rectified = if s > 0.0 { s } else { 0.0 };
                let excited = (rectified * (1.0 + amount * 3.0)).tanh();
                let mixed = s * (1.0 - mix) + (s + excited * 0.3) * mix;
                mixed.clamp(-1.0, 1.0)
            })
            .collect()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &HarmonicEnhanceConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: HarmonicEnhanceConfig) {
        self.hp_coeff = Self::compute_hp_coeff(config.exciter_freq_hz, self.sample_rate);
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq: f32, sample_rate: u32, len: usize) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        (0..len)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * PI * freq as f64 * t).sin() as f32
            })
            .collect()
    }

    #[test]
    fn test_default_config() {
        let cfg = HarmonicEnhanceConfig::default();
        assert_eq!(cfg.mode, EnhanceMode::Waveshaping);
        assert!((cfg.amount - 0.3).abs() < f32::EPSILON);
        assert!((cfg.mix - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.overtone_count, 4);
    }

    #[test]
    fn test_create_enhancer() {
        let enhancer = HarmonicEnhancer::with_defaults(44100);
        assert_eq!(enhancer.sample_rate, 44100);
        assert!(enhancer.hp_coeff > 0.0 && enhancer.hp_coeff < 1.0);
    }

    #[test]
    fn test_analyze_silence() {
        let enhancer = HarmonicEnhancer::with_defaults(44100);
        let silence = vec![0.0_f32; 1024];
        let analysis = enhancer.analyze(&silence);
        assert!(analysis.fundamental_hz.abs() < f32::EPSILON);
        assert!(analysis.spectral_centroid.abs() < f32::EPSILON);
    }

    #[test]
    fn test_analyze_sine() {
        let enhancer = HarmonicEnhancer::with_defaults(44100);
        let sine = make_sine(440.0, 44100, 4096);
        let analysis = enhancer.analyze(&sine);
        // Pitch detection is approximate; just verify it returns something reasonable
        assert!(analysis.fundamental_hz > 100.0);
        assert!(!analysis.harmonic_strengths.is_empty());
    }

    #[test]
    fn test_waveshaping_preserves_length() {
        let enhancer = HarmonicEnhancer::with_defaults(44100);
        let sine = make_sine(440.0, 44100, 2048);
        let result = enhancer.process(&sine);
        assert_eq!(result.len(), 2048);
    }

    #[test]
    fn test_waveshaping_clamps() {
        let config = HarmonicEnhanceConfig {
            amount: 1.0,
            mix: 1.0,
            ..HarmonicEnhanceConfig::default()
        };
        let enhancer = HarmonicEnhancer::new(config, 44100);
        let signal = vec![0.9, -0.9, 0.95, -0.95];
        let result = enhancer.process(&signal);
        for &s in &result {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_overtone_reconstruction() {
        let config = HarmonicEnhanceConfig {
            mode: EnhanceMode::OvertoneReconstruction,
            amount: 0.5,
            mix: 0.5,
            overtone_count: 3,
            ..HarmonicEnhanceConfig::default()
        };
        let enhancer = HarmonicEnhancer::new(config, 44100);
        let sine = make_sine(440.0, 44100, 4096);
        let result = enhancer.process(&sine);
        assert_eq!(result.len(), 4096);
        for &s in &result {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_spectral_tilt_correction() {
        let config = HarmonicEnhanceConfig {
            mode: EnhanceMode::SpectralTilt,
            ..HarmonicEnhanceConfig::default()
        };
        let enhancer = HarmonicEnhancer::new(config, 44100);
        let sine = make_sine(200.0, 44100, 2048);
        let result = enhancer.process(&sine);
        assert_eq!(result.len(), 2048);
    }

    #[test]
    fn test_even_harmonic_exciter() {
        let config = HarmonicEnhanceConfig {
            mode: EnhanceMode::EvenHarmonicExciter,
            ..HarmonicEnhanceConfig::default()
        };
        let enhancer = HarmonicEnhancer::new(config, 44100);
        let sine = make_sine(440.0, 44100, 2048);
        let result = enhancer.process(&sine);
        assert_eq!(result.len(), 2048);
        for &s in &result {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_process_empty() {
        let enhancer = HarmonicEnhancer::with_defaults(44100);
        let result = enhancer.process(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_set_config() {
        let mut enhancer = HarmonicEnhancer::with_defaults(44100);
        let old_coeff = enhancer.hp_coeff;
        let new_config = HarmonicEnhanceConfig {
            exciter_freq_hz: 5000.0,
            ..HarmonicEnhanceConfig::default()
        };
        enhancer.set_config(new_config);
        assert!((enhancer.hp_coeff - old_coeff).abs() > 1e-6);
    }

    #[test]
    fn test_analyze_empty() {
        let enhancer = HarmonicEnhancer::with_defaults(44100);
        let analysis = enhancer.analyze(&[]);
        assert!(analysis.fundamental_hz.abs() < f32::EPSILON);
        assert!(analysis.harmonic_strengths.is_empty());
    }
}

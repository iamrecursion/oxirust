//! Time-stretching algorithms for `OxiMedia` effects.
//!
//! Allows changing audio playback tempo without affecting pitch, using
//! an overlap-add (OLA) approach.

#![allow(dead_code)]

/// Algorithm variant for time stretching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeStretchAlgorithm {
    /// Simple overlap-add (OLA) — fastest, noticeable artefacts on transients.
    Ola,
    /// Waveform similarity OLA (WSOLA) — improved quality.
    Wsola,
    /// Phase vocoder — best quality, higher CPU cost.
    PhaseVocoder,
}

impl TimeStretchAlgorithm {
    /// Quality score from 1 (lowest) to 3 (highest).
    #[must_use]
    pub fn quality_score(&self) -> u8 {
        match self {
            Self::Ola => 1,
            Self::Wsola => 2,
            Self::PhaseVocoder => 3,
        }
    }

    /// Whether the algorithm is phase-coherent.
    #[must_use]
    pub fn is_phase_coherent(&self) -> bool {
        matches!(self, Self::PhaseVocoder)
    }
}

/// Configuration for a [`TimeStretcher`].
#[derive(Debug, Clone)]
pub struct TimeStretchConfig {
    /// Algorithm variant.
    pub algorithm: TimeStretchAlgorithm,
    /// Stretch ratio (e.g. 0.5 = half speed, 2.0 = double speed).
    pub ratio: f32,
    /// Synthesis hop size in samples.
    pub hop_size: usize,
    /// Analysis window size in samples (must be >= `hop_size` * 2).
    pub window_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: f32,
}

impl Default for TimeStretchConfig {
    fn default() -> Self {
        Self {
            algorithm: TimeStretchAlgorithm::Wsola,
            ratio: 1.0,
            hop_size: 512,
            window_size: 2048,
            sample_rate: 48000.0,
        }
    }
}

impl TimeStretchConfig {
    /// Create a config for a given ratio.
    #[must_use]
    pub fn with_ratio(ratio: f32) -> Self {
        Self {
            ratio: ratio.max(0.05),
            ..Default::default()
        }
    }

    /// Returns `true` when all parameters are in sensible ranges.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.ratio > 0.0
            && self.hop_size > 0
            && self.window_size >= self.hop_size * 2
            && self.sample_rate > 0.0
    }

    /// Calculate the analysis hop size from the synthesis hop and ratio.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn analysis_hop(&self) -> usize {
        ((self.hop_size as f32 * self.ratio) as usize).max(1)
    }
}

/// Hanning window of length `n`.
#[allow(clippy::cast_precision_loss)]
fn hanning(n: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos()))
        .collect()
}

/// Time-stretcher using overlap-add.
pub struct TimeStretcher {
    config: TimeStretchConfig,
    window: Vec<f32>,
    /// Internal output accumulation buffer.
    output_buf: Vec<f32>,
}

impl TimeStretcher {
    /// Create a new time-stretcher.
    #[must_use]
    pub fn new(config: TimeStretchConfig) -> Self {
        let window = hanning(config.window_size);
        Self {
            window,
            output_buf: Vec::new(),
            config,
        }
    }

    /// Number of output samples produced for `input_len` input samples.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn output_length(&self, input_len: usize) -> usize {
        (input_len as f32 / self.config.ratio).round() as usize
    }

    /// Stretch `input` by the configured ratio, returning the output buffer.
    ///
    /// Uses OLA: analysis frames are taken at `analysis_hop` intervals,
    /// windowed, and added into the output at `hop_size` intervals.
    #[allow(clippy::cast_precision_loss)]
    pub fn stretch_buffer(&mut self, input: &[f32]) -> Vec<f32> {
        let win_size = self.config.window_size;
        let hop_s = self.config.hop_size;
        let hop_a = self.config.analysis_hop();
        let out_len = self.output_length(input.len());

        let mut output = vec![0.0_f32; out_len + win_size];
        let mut normalize = vec![0.0_f32; out_len + win_size];

        let mut ana_pos = 0usize;
        let mut syn_pos = 0usize;

        while ana_pos + win_size <= input.len() {
            // Extract and window the analysis frame
            for i in 0..win_size {
                let sample = input[ana_pos + i] * self.window[i];
                let out_idx = syn_pos + i;
                if out_idx < output.len() {
                    output[out_idx] += sample;
                    normalize[out_idx] += self.window[i];
                }
            }
            ana_pos += hop_a;
            syn_pos += hop_s;
        }

        // Normalize by overlap weight
        for (s, n) in output.iter_mut().zip(normalize.iter()) {
            if *n > 1e-6 {
                *s /= n;
            }
        }

        output.truncate(out_len);
        output
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_quality_order() {
        assert!(
            TimeStretchAlgorithm::Ola.quality_score() < TimeStretchAlgorithm::Wsola.quality_score()
        );
        assert!(
            TimeStretchAlgorithm::Wsola.quality_score()
                < TimeStretchAlgorithm::PhaseVocoder.quality_score()
        );
    }

    #[test]
    fn test_algorithm_phase_coherence() {
        assert!(!TimeStretchAlgorithm::Ola.is_phase_coherent());
        assert!(!TimeStretchAlgorithm::Wsola.is_phase_coherent());
        assert!(TimeStretchAlgorithm::PhaseVocoder.is_phase_coherent());
    }

    #[test]
    fn test_config_default_is_valid() {
        assert!(TimeStretchConfig::default().is_valid());
    }

    #[test]
    fn test_config_invalid_ratio_zero() {
        let mut cfg = TimeStretchConfig::default();
        cfg.ratio = 0.0;
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_invalid_window_too_small() {
        let mut cfg = TimeStretchConfig::default();
        cfg.window_size = cfg.hop_size; // must be >= 2 * hop_size
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_analysis_hop_ratio_1() {
        let cfg = TimeStretchConfig {
            ratio: 1.0,
            hop_size: 512,
            ..Default::default()
        };
        assert_eq!(cfg.analysis_hop(), 512);
    }

    #[test]
    fn test_config_analysis_hop_double_speed() {
        let cfg = TimeStretchConfig {
            ratio: 2.0,
            hop_size: 512,
            ..Default::default()
        };
        assert_eq!(cfg.analysis_hop(), 1024);
    }

    #[test]
    fn test_output_length_unity() {
        let stretcher = TimeStretcher::new(TimeStretchConfig::default());
        assert_eq!(stretcher.output_length(4800), 4800);
    }

    #[test]
    fn test_output_length_double_speed() {
        let cfg = TimeStretchConfig::with_ratio(2.0);
        let stretcher = TimeStretcher::new(cfg);
        assert_eq!(stretcher.output_length(4800), 2400);
    }

    #[test]
    fn test_output_length_half_speed() {
        let cfg = TimeStretchConfig::with_ratio(0.5);
        let stretcher = TimeStretcher::new(cfg);
        assert_eq!(stretcher.output_length(4800), 9600);
    }

    #[test]
    fn test_stretch_buffer_length_correct() {
        let cfg = TimeStretchConfig {
            ratio: 1.0,
            hop_size: 256,
            window_size: 1024,
            ..Default::default()
        };
        let mut stretcher = TimeStretcher::new(cfg.clone());
        let input = vec![0.5_f32; 8192];
        let output = stretcher.stretch_buffer(&input);
        let expected = stretcher.output_length(input.len());
        assert_eq!(output.len(), expected);
    }

    #[test]
    fn test_stretch_silent_remains_silent() {
        let mut stretcher = TimeStretcher::new(TimeStretchConfig::default());
        let input = vec![0.0_f32; 4096];
        let output = stretcher.stretch_buffer(&input);
        for s in &output {
            assert!(s.abs() < 1e-6, "expected silence, got {s}");
        }
    }
}

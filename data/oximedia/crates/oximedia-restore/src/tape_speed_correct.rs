#![allow(dead_code)]
//! Tape speed correction and calibration for analog audio restoration.
//!
//! This module provides tools for detecting and correcting tape speed errors
//! in digitized analog recordings. It handles constant speed offsets (e.g.,
//! a recording played back at the wrong speed), gradual drift over the
//! duration of a tape, and periodic fluctuations from capstan irregularities.

/// Standard tape speeds in inches per second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StandardTapeSpeed {
    /// 1-7/8 IPS (compact cassette standard).
    Ips1_875,
    /// 3-3/4 IPS (compact cassette high-speed dub).
    Ips3_75,
    /// 7-1/2 IPS (reel-to-reel standard).
    Ips7_5,
    /// 15 IPS (professional reel-to-reel).
    Ips15,
    /// 30 IPS (studio mastering).
    Ips30,
}

impl StandardTapeSpeed {
    /// Get the speed value in inches per second.
    pub fn ips(&self) -> f64 {
        match self {
            Self::Ips1_875 => 1.875,
            Self::Ips3_75 => 3.75,
            Self::Ips7_5 => 7.5,
            Self::Ips15 => 15.0,
            Self::Ips30 => 30.0,
        }
    }

    /// Get all standard speeds.
    pub fn all() -> &'static [StandardTapeSpeed] {
        &[
            Self::Ips1_875,
            Self::Ips3_75,
            Self::Ips7_5,
            Self::Ips15,
            Self::Ips30,
        ]
    }
}

/// Type of speed correction to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionType {
    /// Constant speed ratio correction (uniform across entire recording).
    ConstantRatio,
    /// Linear drift correction (speed changes linearly over time).
    LinearDrift,
    /// Piecewise correction using detected reference points.
    PiecewiseReference,
}

/// Configuration for tape speed correction.
#[derive(Debug, Clone)]
pub struct TapeSpeedConfig {
    /// Type of correction to apply.
    pub correction_type: CorrectionType,
    /// Speed ratio (1.0 = no change, >1.0 = speed up, <1.0 = slow down).
    pub speed_ratio: f64,
    /// For linear drift: ratio at the start of the recording.
    pub drift_start_ratio: f64,
    /// For linear drift: ratio at the end of the recording.
    pub drift_end_ratio: f64,
    /// Reference frequency in Hz for pilot-tone-based correction.
    pub reference_freq_hz: f64,
    /// Whether to apply anti-aliasing filtering during resampling.
    pub anti_alias: bool,
    /// Interpolation quality (1..8, higher = better quality, slower).
    pub interpolation_quality: u8,
}

impl Default for TapeSpeedConfig {
    fn default() -> Self {
        Self {
            correction_type: CorrectionType::ConstantRatio,
            speed_ratio: 1.0,
            drift_start_ratio: 1.0,
            drift_end_ratio: 1.0,
            reference_freq_hz: 440.0,
            anti_alias: true,
            interpolation_quality: 4,
        }
    }
}

/// Result of tape speed analysis.
#[derive(Debug, Clone)]
pub struct SpeedAnalysis {
    /// Detected speed ratio relative to expected.
    pub detected_ratio: f64,
    /// Confidence in the detection (0.0..1.0).
    pub confidence: f32,
    /// Speed variation over time (ratio per segment).
    pub variation_profile: Vec<f64>,
    /// Suggested correction type based on analysis.
    pub suggested_correction: CorrectionType,
    /// If a pilot tone was detected, its frequency.
    pub pilot_tone_hz: Option<f64>,
}

/// Tape speed correction processor.
#[derive(Debug, Clone)]
pub struct TapeSpeedCorrector {
    /// Configuration for correction.
    config: TapeSpeedConfig,
    /// Source sample rate.
    sample_rate: u32,
}

impl TapeSpeedCorrector {
    /// Create a new tape speed corrector.
    pub fn new(config: TapeSpeedConfig, sample_rate: u32) -> Self {
        Self {
            config,
            sample_rate,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(sample_rate: u32) -> Self {
        Self::new(TapeSpeedConfig::default(), sample_rate)
    }

    /// Analyze the signal to detect tape speed errors.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, samples: &[f32]) -> SpeedAnalysis {
        if samples.is_empty() {
            return SpeedAnalysis {
                detected_ratio: 1.0,
                confidence: 0.0,
                variation_profile: Vec::new(),
                suggested_correction: CorrectionType::ConstantRatio,
                pilot_tone_hz: None,
            };
        }

        // Detect fundamental frequency using zero-crossing rate
        let freq = self.detect_frequency(samples);
        let expected = self.config.reference_freq_hz;
        let ratio = if freq > 0.0 { expected / freq } else { 1.0 };

        // Analyze speed variation across segments
        let segment_len = (self.sample_rate as usize).max(1024);
        let mut variation_profile = Vec::new();
        let mut i = 0;
        while i + segment_len <= samples.len() {
            let seg_freq = self.detect_frequency(&samples[i..i + segment_len]);
            let seg_ratio = if seg_freq > 0.0 {
                expected / seg_freq
            } else {
                1.0
            };
            variation_profile.push(seg_ratio);
            i += segment_len;
        }

        // Determine if drift is present
        let suggested = if variation_profile.len() >= 2 {
            let first = variation_profile[0];
            let last = *variation_profile.last().unwrap_or(&1.0);
            let drift = (last - first).abs();
            if drift > 0.02 {
                CorrectionType::LinearDrift
            } else {
                CorrectionType::ConstantRatio
            }
        } else {
            CorrectionType::ConstantRatio
        };

        let confidence = if freq > 20.0 && freq < 20000.0 {
            0.8_f32
        } else {
            0.2_f32
        };

        SpeedAnalysis {
            detected_ratio: ratio,
            confidence,
            variation_profile,
            suggested_correction: suggested,
            pilot_tone_hz: if freq > 0.0 { Some(freq) } else { None },
        }
    }

    /// Detect dominant frequency using zero-crossing rate.
    #[allow(clippy::cast_precision_loss)]
    fn detect_frequency(&self, samples: &[f32]) -> f64 {
        if samples.len() < 4 {
            return 0.0;
        }
        let mut crossings = 0_usize;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        let duration_s = samples.len() as f64 / self.sample_rate as f64;
        if duration_s > 0.0 {
            crossings as f64 / (2.0 * duration_s)
        } else {
            0.0
        }
    }

    /// Apply tape speed correction to the signal.
    pub fn correct(&self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        match self.config.correction_type {
            CorrectionType::ConstantRatio => self.apply_constant_ratio(samples),
            CorrectionType::LinearDrift => self.apply_linear_drift(samples),
            CorrectionType::PiecewiseReference => self.apply_piecewise(samples),
        }
    }

    /// Apply constant speed ratio correction via resampling.
    #[allow(clippy::cast_precision_loss)]
    fn apply_constant_ratio(&self, samples: &[f32]) -> Vec<f32> {
        let ratio = self.config.speed_ratio;
        if (ratio - 1.0).abs() < 1e-6 {
            return samples.to_vec();
        }
        self.resample(samples, ratio)
    }

    /// Apply linear drift correction.
    #[allow(clippy::cast_precision_loss)]
    fn apply_linear_drift(&self, samples: &[f32]) -> Vec<f32> {
        let start_ratio = self.config.drift_start_ratio;
        let end_ratio = self.config.drift_end_ratio;

        if samples.is_empty() {
            return Vec::new();
        }

        // Estimate output length from average ratio
        let avg_ratio = (start_ratio + end_ratio) / 2.0;
        let out_len = (samples.len() as f64 / avg_ratio).round() as usize;
        let mut output = Vec::with_capacity(out_len.max(1));

        let mut read_pos = 0.0_f64;
        for i in 0..out_len {
            let t = if out_len > 1 {
                i as f64 / (out_len - 1) as f64
            } else {
                0.0
            };
            let local_ratio = start_ratio + (end_ratio - start_ratio) * t;

            let idx = read_pos as usize;
            if idx + 1 >= samples.len() {
                break;
            }
            let frac = (read_pos - idx as f64) as f32;
            let interp = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac;
            output.push(interp);

            read_pos += local_ratio;
        }
        output
    }

    /// Apply piecewise correction (uses constant ratio as fallback).
    fn apply_piecewise(&self, samples: &[f32]) -> Vec<f32> {
        // For piecewise, we'd normally have reference points.
        // Fallback to constant ratio correction.
        self.apply_constant_ratio(samples)
    }

    /// Resample signal by a given ratio using linear interpolation.
    #[allow(clippy::cast_precision_loss)]
    fn resample(&self, samples: &[f32], ratio: f64) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        let out_len = (samples.len() as f64 / ratio).round().max(1.0) as usize;
        let mut output = Vec::with_capacity(out_len);

        for i in 0..out_len {
            let src_pos = i as f64 * ratio;
            let idx = src_pos as usize;
            if idx + 1 >= samples.len() {
                if idx < samples.len() {
                    output.push(samples[idx]);
                }
                break;
            }
            let frac = (src_pos - idx as f64) as f32;
            let interp = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac;
            output.push(interp);
        }
        output
    }

    /// Compute the semitone offset for a given speed ratio.
    pub fn ratio_to_semitones(ratio: f64) -> f64 {
        12.0 * (ratio).log2()
    }

    /// Compute the speed ratio for a given semitone offset.
    pub fn semitones_to_ratio(semitones: f64) -> f64 {
        2.0_f64.powf(semitones / 12.0)
    }

    /// Get the current configuration.
    pub fn config(&self) -> &TapeSpeedConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: TapeSpeedConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_sine_f64(freq: f64, sample_rate: u32, len: usize) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        (0..len)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * PI * freq * t).sin() as f32
            })
            .collect()
    }

    #[test]
    fn test_standard_tape_speeds() {
        assert!((StandardTapeSpeed::Ips1_875.ips() - 1.875).abs() < 1e-6);
        assert!((StandardTapeSpeed::Ips7_5.ips() - 7.5).abs() < 1e-6);
        assert!((StandardTapeSpeed::Ips15.ips() - 15.0).abs() < 1e-6);
        assert!((StandardTapeSpeed::Ips30.ips() - 30.0).abs() < 1e-6);
        assert_eq!(StandardTapeSpeed::all().len(), 5);
    }

    #[test]
    fn test_default_config() {
        let cfg = TapeSpeedConfig::default();
        assert_eq!(cfg.correction_type, CorrectionType::ConstantRatio);
        assert!((cfg.speed_ratio - 1.0).abs() < 1e-6);
        assert!(cfg.anti_alias);
        assert_eq!(cfg.interpolation_quality, 4);
    }

    #[test]
    fn test_create_corrector() {
        let corrector = TapeSpeedCorrector::with_defaults(44100);
        assert_eq!(corrector.sample_rate, 44100);
    }

    #[test]
    fn test_analyze_silence() {
        let corrector = TapeSpeedCorrector::with_defaults(44100);
        let silence = vec![0.0_f32; 4096];
        let analysis = corrector.analyze(&silence);
        assert!((analysis.detected_ratio - 1.0).abs() < 0.1 || analysis.confidence < 0.5);
    }

    #[test]
    fn test_analyze_sine() {
        let corrector = TapeSpeedCorrector::with_defaults(44100);
        let sine = make_sine_f64(440.0, 44100, 44100); // 1 second
        let analysis = corrector.analyze(&sine);
        assert!(analysis.pilot_tone_hz.is_some());
    }

    #[test]
    fn test_correct_no_change() {
        let corrector = TapeSpeedCorrector::with_defaults(44100);
        let signal = vec![0.5_f32; 1024];
        let corrected = corrector.correct(&signal);
        // ratio=1.0, so output should be same length
        assert_eq!(corrected.len(), signal.len());
    }

    #[test]
    fn test_correct_speed_up() {
        let config = TapeSpeedConfig {
            speed_ratio: 2.0,
            ..TapeSpeedConfig::default()
        };
        let corrector = TapeSpeedCorrector::new(config, 44100);
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let corrected = corrector.correct(&signal);
        // Speeding up by 2x should roughly halve the length
        assert!(corrected.len() < signal.len());
    }

    #[test]
    fn test_correct_slow_down() {
        let config = TapeSpeedConfig {
            speed_ratio: 0.5,
            ..TapeSpeedConfig::default()
        };
        let corrector = TapeSpeedCorrector::new(config, 44100);
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let corrected = corrector.correct(&signal);
        // Slowing down by 0.5x should roughly double the length
        assert!(corrected.len() > signal.len());
    }

    #[test]
    fn test_linear_drift_correction() {
        let config = TapeSpeedConfig {
            correction_type: CorrectionType::LinearDrift,
            drift_start_ratio: 0.98,
            drift_end_ratio: 1.02,
            ..TapeSpeedConfig::default()
        };
        let corrector = TapeSpeedCorrector::new(config, 44100);
        let signal = vec![0.3_f32; 4096];
        let corrected = corrector.correct(&signal);
        assert!(!corrected.is_empty());
    }

    #[test]
    fn test_piecewise_correction() {
        let config = TapeSpeedConfig {
            correction_type: CorrectionType::PiecewiseReference,
            speed_ratio: 1.05,
            ..TapeSpeedConfig::default()
        };
        let corrector = TapeSpeedCorrector::new(config, 44100);
        let signal = vec![0.2_f32; 2048];
        let corrected = corrector.correct(&signal);
        assert!(!corrected.is_empty());
    }

    #[test]
    fn test_semitone_conversions() {
        // One octave up = 12 semitones = 2.0 ratio
        let ratio = TapeSpeedCorrector::semitones_to_ratio(12.0);
        assert!((ratio - 2.0).abs() < 1e-6);

        let semitones = TapeSpeedCorrector::ratio_to_semitones(2.0);
        assert!((semitones - 12.0).abs() < 1e-6);

        // Round-trip
        let rt =
            TapeSpeedCorrector::ratio_to_semitones(TapeSpeedCorrector::semitones_to_ratio(7.0));
        assert!((rt - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_correct_empty() {
        let corrector = TapeSpeedCorrector::with_defaults(44100);
        let result = corrector.correct(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_set_config() {
        let mut corrector = TapeSpeedCorrector::with_defaults(44100);
        let new_cfg = TapeSpeedConfig {
            speed_ratio: 1.1,
            ..TapeSpeedConfig::default()
        };
        corrector.set_config(new_cfg);
        assert!((corrector.config().speed_ratio - 1.1).abs() < 1e-6);
    }
}

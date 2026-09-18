//! Singing voice detection and singing quality assessment.
//!
//! Distinguishes singing from speech using pitch stability, vibrato presence,
//! sustained voicing, and harmonic richness. Assesses singing quality via
//! intonation accuracy, vibrato regularity, and timbre consistency.

use crate::pitch::{PitchResult, PitchTracker, VibratoResult};
use crate::{AnalysisConfig, AnalysisError, Result};

/// Result of singing voice detection.
#[derive(Debug, Clone)]
pub struct SingingDetectionResult {
    /// Whether singing is present (true) or this is likely speech/silence
    pub is_singing: bool,
    /// Confidence in the detection (0.0–1.0)
    pub confidence: f32,
    /// Pitch stability score (0.0–1.0; higher = more stable = more singing-like)
    pub pitch_stability: f32,
    /// Voicing continuity (fraction of frames that are voiced)
    pub voicing_continuity: f32,
    /// Whether vibrato is detected
    pub has_vibrato: bool,
    /// Vibrato analysis result
    pub vibrato: VibratoResult,
    /// Mean F0 in Hz (0 if unvoiced)
    pub mean_f0: f32,
}

/// Quality assessment of a singing performance.
#[derive(Debug, Clone)]
pub struct SingingQuality {
    /// Overall quality score (0.0–1.0)
    pub overall: f32,
    /// Intonation accuracy (0.0–1.0; how close to equal-tempered pitches)
    pub intonation: f32,
    /// Vibrato regularity (0.0–1.0)
    pub vibrato_regularity: f32,
    /// Pitch stability (0.0–1.0; lower jitter = better)
    pub pitch_stability: f32,
    /// Tonal clarity (0.0–1.0; based on HNR)
    pub tonal_clarity: f32,
    /// Breath support / dynamic consistency (0.0–1.0)
    pub dynamic_consistency: f32,
}

/// Singing analyzer for detection and quality assessment.
pub struct SingingAnalyzer {
    config: AnalysisConfig,
    pitch_tracker: PitchTracker,
}

impl SingingAnalyzer {
    /// Create a new singing analyzer.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let pitch_tracker = PitchTracker::new(config.clone());
        Self {
            config,
            pitch_tracker,
        }
    }

    /// Detect singing voice in audio and return detection result.
    pub fn detect(&self, samples: &[f32], sample_rate: f32) -> Result<SingingDetectionResult> {
        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        let pitch_result = self.pitch_tracker.track(samples, sample_rate)?;
        let vibrato =
            crate::pitch::detect_vibrato(&pitch_result, self.config.hop_size, sample_rate);

        let pitch_stability = compute_pitch_stability(&pitch_result);
        let voicing_continuity = pitch_result.voicing_rate;

        // Singing signature: high voicing rate, stable pitch, sustained notes
        // Speech has more variable pitch and lower voicing rate.
        let is_singing = voicing_continuity > 0.55 && pitch_stability > 0.45;

        let confidence = {
            let vc_score = (voicing_continuity - 0.5).max(0.0) / 0.5;
            let ps_score = (pitch_stability - 0.3).max(0.0) / 0.7;
            let vib_bonus = if vibrato.present { 0.1_f32 } else { 0.0 };
            ((vc_score * 0.5 + ps_score * 0.4 + vib_bonus).min(1.0)).max(0.0)
        };

        Ok(SingingDetectionResult {
            is_singing,
            confidence,
            pitch_stability,
            voicing_continuity,
            has_vibrato: vibrato.present,
            vibrato,
            mean_f0: pitch_result.mean_f0,
        })
    }

    /// Assess the quality of singing in the provided audio.
    pub fn assess_quality(&self, samples: &[f32], sample_rate: f32) -> Result<SingingQuality> {
        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        let pitch_result = self.pitch_tracker.track(samples, sample_rate)?;
        let vibrato =
            crate::pitch::detect_vibrato(&pitch_result, self.config.hop_size, sample_rate);

        let pitch_stability = compute_pitch_stability(&pitch_result);
        let intonation = compute_intonation_score(&pitch_result);
        let vibrato_regularity = compute_vibrato_regularity(&vibrato, &pitch_result);
        let tonal_clarity = compute_tonal_clarity(samples, sample_rate, pitch_result.mean_f0);
        let dynamic_consistency = compute_dynamic_consistency(samples, self.config.hop_size);

        let overall = (intonation * 0.3
            + vibrato_regularity * 0.2
            + pitch_stability * 0.25
            + tonal_clarity * 0.15
            + dynamic_consistency * 0.1)
            .min(1.0)
            .max(0.0);

        Ok(SingingQuality {
            overall,
            intonation,
            vibrato_regularity,
            pitch_stability,
            tonal_clarity,
            dynamic_consistency,
        })
    }
}

// ── internal helpers ────────────────────────────────────────────────────────

/// Compute pitch stability as 1 - normalized standard deviation of voiced frames.
fn compute_pitch_stability(pitch_result: &PitchResult) -> f32 {
    let voiced: Vec<f32> = pitch_result
        .estimates
        .iter()
        .zip(&pitch_result.confidences)
        .filter(|(_, &c)| c > 0.5)
        .map(|(&f, _)| f)
        .collect();

    if voiced.len() < 2 {
        return 0.0;
    }

    let mean = voiced.iter().sum::<f32>() / voiced.len() as f32;
    if mean <= 0.0 {
        return 0.0;
    }

    let variance = voiced.iter().map(|&f| (f - mean).powi(2)).sum::<f32>() / voiced.len() as f32;
    let std_dev = variance.sqrt();
    let cv = std_dev / mean; // coefficient of variation

    // Stability: low CV → high stability
    (1.0 - (cv * 5.0).min(1.0)).max(0.0)
}

/// Intonation: fraction of voiced frames within ±50 cents of nearest equal-tempered pitch.
fn compute_intonation_score(pitch_result: &PitchResult) -> f32 {
    let voiced: Vec<f32> = pitch_result
        .estimates
        .iter()
        .zip(&pitch_result.confidences)
        .filter(|(_, &c)| c > 0.5)
        .map(|(&f, _)| f)
        .collect();

    if voiced.is_empty() {
        return 0.0;
    }

    // A4 = 440 Hz, MIDI 69
    let in_tune: usize = voiced
        .iter()
        .filter(|&&f| {
            if f <= 0.0 {
                return false;
            }
            // Cents from nearest semitone
            let midi_float = 69.0 + 12.0 * (f / 440.0_f32).log2();
            let nearest = midi_float.round();
            let cents = (midi_float - nearest).abs() * 100.0;
            cents <= 50.0
        })
        .count();

    in_tune as f32 / voiced.len() as f32
}

/// Vibrato regularity: confidence of vibrato detection × extent fit.
fn compute_vibrato_regularity(vibrato: &VibratoResult, pitch_result: &PitchResult) -> f32 {
    if !vibrato.present {
        // No vibrato is acceptable; neutral score
        return 0.5;
    }

    // Rate should be 5–7 Hz, extent 50–150 cents for classical/pop singing
    let rate_score = if (5.0..=7.5).contains(&vibrato.rate) {
        1.0_f32
    } else if (4.0..=9.0).contains(&vibrato.rate) {
        0.7
    } else {
        0.3
    };

    let extent_score = if (50.0..=150.0).contains(&vibrato.extent) {
        1.0_f32
    } else if (30.0..=200.0).contains(&vibrato.extent) {
        0.7
    } else {
        0.3
    };

    let consistency = compute_pitch_stability(pitch_result);

    (rate_score * 0.4 + extent_score * 0.4 + consistency * 0.2).min(1.0)
}

/// Tonal clarity via approximated HNR from autocorrelation.
fn compute_tonal_clarity(samples: &[f32], sample_rate: f32, f0: f32) -> f32 {
    if f0 <= 0.0 || samples.is_empty() {
        return 0.0;
    }

    let period = (sample_rate / f0) as usize;
    if period == 0 || samples.len() < period * 2 {
        return 0.0;
    }

    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    for i in 0..(samples.len() - period) {
        sum += samples[i] * samples[i + period];
        norm += samples[i] * samples[i];
    }

    if norm <= 0.0 {
        return 0.0;
    }

    let r = (sum / norm).clamp(0.0, 0.9999);
    // Convert HNR to a 0–1 score: HNR = 10*log10(r/(1-r)), cap at 30 dB
    let hnr_db = 10.0 * (r / (1.0 - r)).log10();
    (hnr_db / 30.0).clamp(0.0, 1.0)
}

/// Dynamic consistency: inverse of amplitude variation across hop-size frames.
fn compute_dynamic_consistency(samples: &[f32], hop_size: usize) -> f32 {
    if samples.len() < hop_size * 2 {
        return 0.0;
    }

    let hop = hop_size.max(1);
    let rms_values: Vec<f32> = samples
        .chunks(hop)
        .map(|chunk| {
            let sq: f32 = chunk.iter().map(|&x| x * x).sum();
            (sq / chunk.len() as f32).sqrt()
        })
        .collect();

    if rms_values.len() < 2 {
        return 0.0;
    }

    let mean = rms_values.iter().sum::<f32>() / rms_values.len() as f32;
    if mean <= 0.0 {
        return 0.0;
    }

    let variance =
        rms_values.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / rms_values.len() as f32;
    let cv = variance.sqrt() / mean;

    (1.0 - (cv * 3.0).min(1.0)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_sustained_sine(freq: f32, duration_s: f32, sr: f32) -> Vec<f32> {
        (0..(sr * duration_s) as usize)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin() * 0.8)
            .collect()
    }

    #[test]
    fn test_singing_detector_on_sine_wave() {
        let config = AnalysisConfig::default();
        let analyzer = SingingAnalyzer::new(config);
        let samples = make_sustained_sine(220.0, 2.0, 44100.0);
        let result = analyzer.detect(&samples, 44100.0);
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        assert!(r.confidence >= 0.0 && r.confidence <= 1.0);
        assert!(r.pitch_stability >= 0.0 && r.pitch_stability <= 1.0);
    }

    #[test]
    fn test_quality_on_pure_sine() {
        let config = AnalysisConfig::default();
        let analyzer = SingingAnalyzer::new(config);
        // A4 = 440 Hz is a perfect equal-tempered pitch → high intonation
        let samples = make_sustained_sine(440.0, 2.0, 44100.0);
        let quality = analyzer.assess_quality(&samples, 44100.0);
        assert!(quality.is_ok());
        let q = quality.expect("should succeed");
        assert!(q.overall >= 0.0 && q.overall <= 1.0);
        assert!(q.intonation >= 0.0 && q.intonation <= 1.0);
        assert!(q.dynamic_consistency >= 0.0 && q.dynamic_consistency <= 1.0);
    }

    #[test]
    fn test_singing_detector_insufficient_samples() {
        let config = AnalysisConfig::default();
        let analyzer = SingingAnalyzer::new(config.clone());
        let result = analyzer.detect(&[0.0; 10], 44100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_pitch_stability_constant_pitch() {
        let estimates = vec![440.0_f32; 50];
        let confidences = vec![0.9_f32; 50];
        let pr = crate::pitch::PitchResult {
            estimates,
            confidences,
            mean_f0: 440.0,
            voicing_rate: 1.0,
        };
        let stab = compute_pitch_stability(&pr);
        assert!(
            stab > 0.9,
            "Constant pitch should have very high stability: {stab}"
        );
    }

    #[test]
    fn test_intonation_a440() {
        // A4 = 440 Hz is exactly on equal temperament
        let estimates = vec![440.0_f32; 30];
        let confidences = vec![0.9_f32; 30];
        let pr = crate::pitch::PitchResult {
            estimates,
            confidences,
            mean_f0: 440.0,
            voicing_rate: 1.0,
        };
        let score = compute_intonation_score(&pr);
        assert!(
            score > 0.9,
            "440 Hz should have near-perfect intonation: {score}"
        );
    }

    #[test]
    fn test_dynamic_consistency_constant_amplitude() {
        let samples: Vec<f32> = vec![0.5; 4096];
        let score = compute_dynamic_consistency(&samples, 512);
        assert!(
            score > 0.8,
            "Constant amplitude should have high dynamic consistency: {score}"
        );
    }
}

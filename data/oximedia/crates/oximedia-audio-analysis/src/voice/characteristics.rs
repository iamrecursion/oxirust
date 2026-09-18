//! Voice characteristics analysis.

use crate::formant::FormantAnalyzer;
use crate::pitch::PitchTracker;
use crate::{AnalysisConfig, AnalysisError, Result};

/// Voice analyzer for extracting voice characteristics.
pub struct VoiceAnalyzer {
    config: AnalysisConfig,
    pitch_tracker: PitchTracker,
    formant_analyzer: FormantAnalyzer,
}

impl VoiceAnalyzer {
    /// Create a new voice analyzer.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            pitch_tracker: PitchTracker::new(config.clone()),
            formant_analyzer: FormantAnalyzer::new(config.clone()),
            config,
        }
    }

    /// Analyze voice characteristics from audio samples.
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<VoiceCharacteristics> {
        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        // Extract pitch (F0)
        let pitch_result = self.pitch_tracker.track(samples, sample_rate)?;
        let f0 = pitch_result.mean_f0;

        // Extract formants
        let formant_result = self.formant_analyzer.analyze(samples, sample_rate)?;

        // Compute jitter (pitch variation)
        let jitter = self.compute_jitter(samples, sample_rate, f0)?;

        // Compute shimmer (amplitude variation)
        let shimmer = self.compute_shimmer(samples)?;

        // Compute harmonics-to-noise ratio (HNR)
        let hnr = self.compute_hnr(samples, sample_rate, f0)?;

        // Detect gender based on F0 and formants
        let gender = super::gender::detect_gender(f0, &formant_result.formants);

        // Estimate age
        let age_group = super::age::estimate_age(f0, &formant_result.formants, jitter, shimmer);

        // Detect emotion
        let emotion = super::emotion::detect_emotion(f0, jitter, shimmer, &formant_result.formants);

        Ok(VoiceCharacteristics {
            f0,
            formants: formant_result.formants,
            jitter,
            shimmer,
            hnr,
            gender,
            age_group,
            emotion,
        })
    }

    /// Compute jitter (pitch period variation).
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
    fn compute_jitter(&self, samples: &[f32], sample_rate: f32, f0: f32) -> Result<f32> {
        if f0 <= 0.0 {
            return Ok(0.0);
        }

        // Detect pitch periods
        let period_samples = (sample_rate / f0) as usize;
        if period_samples == 0 || samples.len() < period_samples * 3 {
            return Ok(0.0);
        }

        // Find zero crossings to detect periods
        let mut periods = Vec::new();
        let mut last_crossing = 0;

        for i in 1..samples.len() {
            if samples[i] >= 0.0 && samples[i - 1] < 0.0 {
                if last_crossing > 0 {
                    periods.push((i - last_crossing) as f32);
                }
                last_crossing = i;
            }
        }

        if periods.len() < 2 {
            return Ok(0.0);
        }

        // Compute jitter as average absolute difference between consecutive periods
        let mut jitter_sum = 0.0;
        for i in 1..periods.len() {
            jitter_sum += (periods[i] - periods[i - 1]).abs();
        }

        let mean_period: f32 = periods.iter().sum::<f32>() / periods.len() as f32;
        if mean_period > 0.0 {
            Ok(jitter_sum / (periods.len() - 1) as f32 / mean_period)
        } else {
            Ok(0.0)
        }
    }

    /// Compute shimmer (amplitude variation).
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
    fn compute_shimmer(&self, samples: &[f32]) -> Result<f32> {
        // Compute peak amplitudes in consecutive frames
        let frame_size = 512;
        if samples.len() < frame_size * 2 {
            return Ok(0.0);
        }

        let mut peaks = Vec::new();
        for chunk in samples.chunks(frame_size) {
            let peak = chunk.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
            peaks.push(peak);
        }

        if peaks.len() < 2 {
            return Ok(0.0);
        }

        // Compute shimmer as average absolute difference between consecutive peaks
        let mut shimmer_sum = 0.0;
        for i in 1..peaks.len() {
            shimmer_sum += (peaks[i] - peaks[i - 1]).abs();
        }

        let mean_peak: f32 = peaks.iter().sum::<f32>() / peaks.len() as f32;
        if mean_peak > 0.0 {
            Ok(shimmer_sum / (peaks.len() - 1) as f32 / mean_peak)
        } else {
            Ok(0.0)
        }
    }

    /// Compute harmonics-to-noise ratio (HNR).
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
    fn compute_hnr(&self, samples: &[f32], sample_rate: f32, f0: f32) -> Result<f32> {
        if f0 <= 0.0 || samples.is_empty() {
            return Ok(0.0);
        }

        let period_samples = (sample_rate / f0) as usize;
        if period_samples == 0 || samples.len() < period_samples * 2 {
            return Ok(0.0);
        }

        // Compute autocorrelation at fundamental period
        let autocorr = self.autocorrelation(samples, period_samples);

        // HNR = 10 * log10(r / (1 - r))
        let r = autocorr.clamp(0.0, 0.9999);
        Ok(10.0 * (r / (1.0 - r)).log10())
    }

    /// Compute autocorrelation at a specific lag.
    #[allow(clippy::unused_self)]
    fn autocorrelation(&self, samples: &[f32], lag: usize) -> f32 {
        if lag >= samples.len() {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut norm = 0.0;

        for i in 0..(samples.len() - lag) {
            sum += samples[i] * samples[i + lag];
            norm += samples[i] * samples[i];
        }

        if norm > 0.0 {
            sum / norm
        } else {
            0.0
        }
    }
}

/// Voice characteristics extracted from audio.
#[derive(Debug, Clone)]
pub struct VoiceCharacteristics {
    /// Fundamental frequency (F0) in Hz
    pub f0: f32,
    /// Formant frequencies [F1, F2, F3, F4] in Hz
    pub formants: Vec<f32>,
    /// Jitter (pitch period variation, 0-1)
    pub jitter: f32,
    /// Shimmer (amplitude variation, 0-1)
    pub shimmer: f32,
    /// Harmonics-to-noise ratio in dB
    pub hnr: f32,
    /// Detected gender
    pub gender: super::gender::Gender,
    /// Estimated age group
    pub age_group: super::age::AgeGroup,
    /// Detected emotion
    pub emotion: super::emotion::Emotion,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_analyzer() {
        let config = AnalysisConfig::default();
        let analyzer = VoiceAnalyzer::new(config);

        // Generate test signal (440 Hz sine wave)
        let sample_rate = 44100.0;
        let duration = 1.0;
        let frequency = 440.0;
        let samples: Vec<f32> = (0..(sample_rate * duration) as usize)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        let result = analyzer.analyze(&samples, sample_rate);
        assert!(result.is_ok());
    }
}

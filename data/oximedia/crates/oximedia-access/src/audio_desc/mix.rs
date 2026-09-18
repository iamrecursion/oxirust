//! Mix audio descriptions into main audio.

use crate::audio_desc::generator::AudioSegment;
use crate::error::{AccessError, AccessResult};
use bytes::Bytes;
use oximedia_audio::frame::AudioBuffer;
use serde::{Deserialize, Serialize};

/// Strategy for mixing audio description into main audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixStrategy {
    /// Replace main audio with description.
    Replace,
    /// Mix description over main audio (description louder).
    Mix,
    /// Duck main audio and mix description.
    Duck,
    /// Pause main audio during description.
    Pause,
}

impl MixStrategy {
    /// Get description of the strategy.
    #[must_use]
    pub const fn description(&self) -> &str {
        match self {
            Self::Replace => "Replace main audio with audio description",
            Self::Mix => "Mix description over main audio",
            Self::Duck => "Lower main audio volume during description",
            Self::Pause => "Pause main audio during description",
        }
    }

    /// Check if this strategy modifies main audio timing.
    #[must_use]
    pub const fn alters_timing(&self) -> bool {
        matches!(self, Self::Pause)
    }
}

/// Configuration for audio description mixing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixConfig {
    /// Mixing strategy.
    pub strategy: MixStrategy,
    /// Description volume (0.0 to 1.0).
    pub description_volume: f32,
    /// Main audio volume during description (0.0 to 1.0).
    pub main_volume: f32,
    /// Ducking attack time in milliseconds (for Duck strategy).
    pub duck_attack_ms: f32,
    /// Ducking release time in milliseconds (for Duck strategy).
    pub duck_release_ms: f32,
    /// Crossfade duration in milliseconds.
    pub crossfade_ms: i64,
}

impl Default for MixConfig {
    fn default() -> Self {
        Self {
            strategy: MixStrategy::Duck,
            description_volume: 1.0,
            main_volume: 0.3,
            duck_attack_ms: 100.0,
            duck_release_ms: 200.0,
            crossfade_ms: 50,
        }
    }
}

impl MixConfig {
    /// Create configuration for a specific strategy.
    #[must_use]
    pub fn for_strategy(strategy: MixStrategy) -> Self {
        let mut config = Self::default();
        config.strategy = strategy;

        match strategy {
            MixStrategy::Replace => {
                config.description_volume = 1.0;
                config.main_volume = 0.0;
            }
            MixStrategy::Mix => {
                config.description_volume = 0.8;
                config.main_volume = 0.6;
            }
            MixStrategy::Duck => {
                config.description_volume = 1.0;
                config.main_volume = 0.3;
            }
            MixStrategy::Pause => {
                config.description_volume = 1.0;
                config.main_volume = 0.0;
            }
        }

        config
    }

    /// Validate configuration.
    pub fn validate(&self) -> AccessResult<()> {
        if !(0.0..=1.0).contains(&self.description_volume) {
            return Err(AccessError::AudioDescriptionFailed(
                "Description volume must be between 0.0 and 1.0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.main_volume) {
            return Err(AccessError::AudioDescriptionFailed(
                "Main volume must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.duck_attack_ms < 0.0 || self.duck_attack_ms > 1000.0 {
            return Err(AccessError::AudioDescriptionFailed(
                "Duck attack time must be between 0 and 1000ms".to_string(),
            ));
        }

        if self.duck_release_ms < 0.0 || self.duck_release_ms > 1000.0 {
            return Err(AccessError::AudioDescriptionFailed(
                "Duck release time must be between 0 and 1000ms".to_string(),
            ));
        }

        Ok(())
    }
}

/// Extract interleaved f32 samples from an `AudioBuffer`.
fn buffer_to_f32_samples(buf: &AudioBuffer) -> Vec<f32> {
    let bytes: Vec<u8> = match buf {
        AudioBuffer::Interleaved(data) => data.to_vec(),
        AudioBuffer::Planar(planes) => planes.iter().flat_map(|p| p.iter().copied()).collect(),
    };
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Encode f32 samples back into an interleaved `AudioBuffer`.
fn f32_samples_to_buffer(samples: &[f32]) -> AudioBuffer {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    AudioBuffer::Interleaved(Bytes::from(bytes))
}

/// Mixes audio descriptions into main audio.
pub struct AudioDescriptionMixer {
    config: MixConfig,
}

impl AudioDescriptionMixer {
    /// Create a new mixer.
    #[must_use]
    pub fn new(config: MixConfig) -> Self {
        Self { config }
    }

    /// Create mixer with default configuration.
    #[must_use]
    pub fn default() -> Self {
        Self::new(MixConfig::default())
    }

    /// Mix audio description segments into main audio.
    ///
    /// `sample_rate` and `channels` describe the layout of `main_audio`.
    pub fn mix(
        &self,
        main_audio: &AudioBuffer,
        segments: &[AudioSegment],
        sample_rate: u32,
        channels: u16,
    ) -> AccessResult<AudioBuffer> {
        self.config.validate()?;

        match self.config.strategy {
            MixStrategy::Replace => self.mix_replace(main_audio, segments, sample_rate, channels),
            MixStrategy::Mix => self.mix_blend(main_audio, segments, sample_rate, channels),
            MixStrategy::Duck => self.mix_duck(main_audio, segments, sample_rate, channels),
            MixStrategy::Pause => self.mix_pause(main_audio, segments, sample_rate, channels),
        }
    }

    /// Replace strategy: Replace main audio with description.
    fn mix_replace(
        &self,
        main_audio: &AudioBuffer,
        segments: &[AudioSegment],
        sample_rate: u32,
        channels: u16,
    ) -> AccessResult<AudioBuffer> {
        let mut samples = buffer_to_f32_samples(main_audio);

        for segment in segments {
            let start_sample = self.time_to_sample(segment.start_time_ms, sample_rate);
            let end_sample = self.time_to_sample(segment.end_time_ms, sample_rate);
            let seg_samples = buffer_to_f32_samples(&segment.audio);

            self.apply_segment(
                &mut samples,
                &seg_samples,
                start_sample,
                end_sample,
                self.config.description_volume,
                0.0,
                channels,
            )?;
        }

        Ok(f32_samples_to_buffer(&samples))
    }

    /// Mix strategy: Blend description with main audio.
    fn mix_blend(
        &self,
        main_audio: &AudioBuffer,
        segments: &[AudioSegment],
        sample_rate: u32,
        channels: u16,
    ) -> AccessResult<AudioBuffer> {
        let mut samples = buffer_to_f32_samples(main_audio);

        for segment in segments {
            let start_sample = self.time_to_sample(segment.start_time_ms, sample_rate);
            let end_sample = self.time_to_sample(segment.end_time_ms, sample_rate);
            let seg_samples = buffer_to_f32_samples(&segment.audio);

            self.apply_segment(
                &mut samples,
                &seg_samples,
                start_sample,
                end_sample,
                self.config.description_volume,
                self.config.main_volume,
                channels,
            )?;
        }

        Ok(f32_samples_to_buffer(&samples))
    }

    /// Duck strategy: Lower main audio volume during description.
    fn mix_duck(
        &self,
        main_audio: &AudioBuffer,
        segments: &[AudioSegment],
        sample_rate: u32,
        channels: u16,
    ) -> AccessResult<AudioBuffer> {
        let mut samples = buffer_to_f32_samples(main_audio);

        let attack_samples = (self.config.duck_attack_ms * sample_rate as f32 / 1000.0) as usize;
        let release_samples = (self.config.duck_release_ms * sample_rate as f32 / 1000.0) as usize;

        for segment in segments {
            let start_sample = self.time_to_sample(segment.start_time_ms, sample_rate);
            let end_sample = self.time_to_sample(segment.end_time_ms, sample_rate);
            let seg_samples = buffer_to_f32_samples(&segment.audio);

            // Apply ducking envelope
            self.apply_ducking(
                &mut samples,
                start_sample,
                end_sample,
                attack_samples,
                release_samples,
                self.config.main_volume,
                channels,
            );

            // Mix in description
            self.apply_segment(
                &mut samples,
                &seg_samples,
                start_sample,
                end_sample,
                self.config.description_volume,
                1.0,
                channels,
            )?;
        }

        Ok(f32_samples_to_buffer(&samples))
    }

    /// Pause strategy: Pause main audio during description.
    fn mix_pause(
        &self,
        main_audio: &AudioBuffer,
        segments: &[AudioSegment],
        sample_rate: u32,
        channels: u16,
    ) -> AccessResult<AudioBuffer> {
        // This would require time stretching or frame dropping
        // For now, just replace like Replace strategy
        self.mix_replace(main_audio, segments, sample_rate, channels)
    }

    /// Apply a segment to the output sample buffer.
    #[allow(clippy::too_many_arguments)]
    fn apply_segment(
        &self,
        output: &mut Vec<f32>,
        seg_samples: &[f32],
        start_sample: usize,
        end_sample: usize,
        desc_volume: f32,
        main_volume: f32,
        channels: u16,
    ) -> AccessResult<()> {
        let duration_samples =
            (end_sample - start_sample).min(seg_samples.len() / channels as usize);

        for i in 0..duration_samples {
            for ch in 0..channels as usize {
                let output_idx = (start_sample + i) * channels as usize + ch;
                let seg_idx = i * channels as usize + ch;

                if output_idx < output.len() && seg_idx < seg_samples.len() {
                    output[output_idx] =
                        output[output_idx] * main_volume + seg_samples[seg_idx] * desc_volume;
                }
            }
        }

        Ok(())
    }

    /// Apply ducking envelope to main audio sample buffer.
    #[allow(clippy::too_many_arguments)]
    fn apply_ducking(
        &self,
        samples: &mut Vec<f32>,
        start_sample: usize,
        end_sample: usize,
        attack_samples: usize,
        release_samples: usize,
        duck_level: f32,
        channels: u16,
    ) {
        for i in start_sample..end_sample {
            let envelope = if i < start_sample + attack_samples {
                // Attack phase
                let t = (i - start_sample) as f32 / attack_samples as f32;
                1.0 - (1.0 - duck_level) * t
            } else if i > end_sample - release_samples {
                // Release phase
                let t = (end_sample - i) as f32 / release_samples as f32;
                duck_level + (1.0 - duck_level) * (1.0 - t)
            } else {
                // Sustain phase
                duck_level
            };

            for ch in 0..channels as usize {
                let idx = i * channels as usize + ch;
                if idx < samples.len() {
                    samples[idx] *= envelope;
                }
            }
        }
    }

    /// Convert time in milliseconds to sample index.
    fn time_to_sample(&self, time_ms: i64, sample_rate: u32) -> usize {
        ((time_ms as f64 / 1000.0) * f64::from(sample_rate)) as usize
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &MixConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mix_strategy_description() {
        assert!(!MixStrategy::Replace.description().is_empty());
        assert!(!MixStrategy::Mix.description().is_empty());
    }

    #[test]
    fn test_mix_strategy_timing() {
        assert!(!MixStrategy::Replace.alters_timing());
        assert!(!MixStrategy::Duck.alters_timing());
        assert!(MixStrategy::Pause.alters_timing());
    }

    #[test]
    fn test_mix_config_default() {
        let config = MixConfig::default();
        assert_eq!(config.strategy, MixStrategy::Duck);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_mix_config_for_strategy() {
        let config = MixConfig::for_strategy(MixStrategy::Replace);
        assert_eq!(config.strategy, MixStrategy::Replace);
        assert!((config.main_volume - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_validation() {
        let mut config = MixConfig::default();
        assert!(config.validate().is_ok());

        config.description_volume = 1.5;
        assert!(config.validate().is_err());

        config.description_volume = 0.8;
        config.duck_attack_ms = 2000.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mixer_creation() {
        let mixer = AudioDescriptionMixer::default();
        assert_eq!(mixer.config().strategy, MixStrategy::Duck);
    }

    #[test]
    fn test_time_to_sample() {
        let mixer = AudioDescriptionMixer::default();
        let sample = mixer.time_to_sample(1000, 48000);
        assert_eq!(sample, 48000);

        let sample = mixer.time_to_sample(500, 48000);
        assert_eq!(sample, 24000);
    }
}

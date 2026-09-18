//! Chorus effect implementation.
//!
//! A chorus effect creates a thicker, richer sound by combining the original
//! signal with multiple delayed and modulated copies (voices). Each voice
//! has a slightly different delay time that varies with an LFO.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_arguments)]

use super::delay::{FractionalDelayLine, InterpolationMode};
use super::lfo::{Lfo, LfoWaveform, ParameterSmoother};

/// Maximum number of chorus voices.
pub const MAX_CHORUS_VOICES: usize = 8;

/// Minimum number of chorus voices.
pub const MIN_CHORUS_VOICES: usize = 2;

/// Default chorus delay in milliseconds.
const DEFAULT_DELAY_MS: f64 = 25.0;

/// Default chorus depth in milliseconds.
const DEFAULT_DEPTH_MS: f64 = 5.0;

/// Configuration for the chorus effect.
#[derive(Clone, Debug)]
pub struct ChorusConfig {
    /// Number of voices (2-8).
    pub voices: usize,
    /// LFO rate in Hz (0.1 to 10.0).
    pub rate: f64,
    /// Modulation depth in milliseconds (0.0 to 20.0).
    pub depth: f64,
    /// Base delay in milliseconds (10.0 to 50.0).
    pub delay: f64,
    /// Dry/wet mix (0.0 = dry only, 1.0 = wet only).
    pub mix: f64,
    /// Feedback amount (-0.9 to 0.9).
    pub feedback: f64,
    /// Stereo spread (0.0 = mono, 1.0 = maximum spread).
    pub spread: f64,
    /// LFO waveform.
    pub waveform: LfoWaveform,
}

impl Default for ChorusConfig {
    fn default() -> Self {
        Self {
            voices: 4,
            rate: 1.5,
            depth: DEFAULT_DEPTH_MS,
            delay: DEFAULT_DELAY_MS,
            mix: 0.5,
            feedback: 0.0,
            spread: 0.7,
            waveform: LfoWaveform::Sine,
        }
    }
}

impl ChorusConfig {
    /// Create a new chorus configuration.
    ///
    /// # Arguments
    ///
    /// * `voices` - Number of voices (2-8)
    /// * `rate` - LFO rate in Hz
    /// * `depth` - Modulation depth in milliseconds
    #[must_use]
    pub fn new(voices: usize, rate: f64, depth: f64) -> Self {
        Self {
            voices: voices.clamp(MIN_CHORUS_VOICES, MAX_CHORUS_VOICES),
            rate: rate.clamp(0.1, 10.0),
            depth: depth.clamp(0.0, 20.0),
            ..Default::default()
        }
    }

    /// Set the number of voices.
    #[must_use]
    pub fn with_voices(mut self, voices: usize) -> Self {
        self.voices = voices.clamp(MIN_CHORUS_VOICES, MAX_CHORUS_VOICES);
        self
    }

    /// Set the LFO rate.
    #[must_use]
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate = rate.clamp(0.1, 10.0);
        self
    }

    /// Set the modulation depth.
    #[must_use]
    pub fn with_depth(mut self, depth: f64) -> Self {
        self.depth = depth.clamp(0.0, 20.0);
        self
    }

    /// Set the base delay.
    #[must_use]
    pub fn with_delay(mut self, delay: f64) -> Self {
        self.delay = delay.clamp(10.0, 50.0);
        self
    }

    /// Set the dry/wet mix.
    #[must_use]
    pub fn with_mix(mut self, mix: f64) -> Self {
        self.mix = mix.clamp(0.0, 1.0);
        self
    }

    /// Set the feedback amount.
    #[must_use]
    pub fn with_feedback(mut self, feedback: f64) -> Self {
        self.feedback = feedback.clamp(-0.9, 0.9);
        self
    }

    /// Set the stereo spread.
    #[must_use]
    pub fn with_spread(mut self, spread: f64) -> Self {
        self.spread = spread.clamp(0.0, 1.0);
        self
    }

    /// Set the LFO waveform.
    #[must_use]
    pub fn with_waveform(mut self, waveform: LfoWaveform) -> Self {
        self.waveform = waveform;
        self
    }

    /// Create a subtle chorus preset.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            voices: 2,
            rate: 0.8,
            depth: 3.0,
            delay: 20.0,
            mix: 0.3,
            feedback: 0.1,
            spread: 0.5,
            waveform: LfoWaveform::Sine,
        }
    }

    /// Create a lush chorus preset.
    #[must_use]
    pub fn lush() -> Self {
        Self {
            voices: 6,
            rate: 1.2,
            depth: 7.0,
            delay: 25.0,
            mix: 0.6,
            feedback: 0.2,
            spread: 0.8,
            waveform: LfoWaveform::Sine,
        }
    }

    /// Create a vibrato preset (deep modulation, no dry signal).
    #[must_use]
    pub fn vibrato() -> Self {
        Self {
            voices: 1,
            rate: 5.0,
            depth: 10.0,
            delay: 15.0,
            mix: 1.0,
            feedback: 0.0,
            spread: 0.0,
            waveform: LfoWaveform::Sine,
        }
    }
}

/// Single chorus voice.
struct ChorusVoice {
    /// Fractional delay line.
    delay_line: FractionalDelayLine,
    /// LFO for modulation.
    lfo: Lfo,
}

impl ChorusVoice {
    /// Create a new chorus voice.
    fn new(sample_rate: f64, rate: f64, phase_offset: f64, waveform: LfoWaveform) -> Self {
        let max_delay_ms = 100.0;
        let mut lfo = Lfo::new(rate, sample_rate, waveform);
        lfo.set_phase(phase_offset);

        Self {
            delay_line: FractionalDelayLine::new(
                max_delay_ms,
                sample_rate,
                InterpolationMode::Linear,
            ),
            lfo,
        }
    }

    /// Process one sample through the voice.
    fn process(&mut self, input: f64, base_delay: f64, depth: f64, feedback: f64) -> f64 {
        let lfo_value = self.lfo.next();
        let modulated_delay = base_delay + lfo_value * depth;
        self.delay_line
            .process_with_feedback(input, modulated_delay, feedback)
    }

    /// Reset the voice.
    fn reset(&mut self) {
        self.delay_line.reset();
        self.lfo.reset();
    }
}

/// Mono chorus effect processor.
pub struct MonoChorus {
    /// Configuration.
    config: ChorusConfig,
    /// Chorus voices.
    voices: Vec<ChorusVoice>,
    /// Sample rate.
    sample_rate: f64,
    /// Base delay in samples.
    base_delay_samples: f64,
    /// Depth in samples.
    depth_samples: f64,
    /// Mix smoother.
    mix_smoother: ParameterSmoother,
}

impl MonoChorus {
    /// Create a new mono chorus effect.
    ///
    /// # Arguments
    ///
    /// * `config` - Chorus configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: ChorusConfig, sample_rate: f64) -> Self {
        let mut voices = Vec::with_capacity(config.voices);

        for i in 0..config.voices {
            let phase_offset = i as f64 / config.voices as f64;
            voices.push(ChorusVoice::new(
                sample_rate,
                config.rate,
                phase_offset,
                config.waveform,
            ));
        }

        let base_delay_samples = config.delay * 0.001 * sample_rate;
        let depth_samples = config.depth * 0.001 * sample_rate;

        Self {
            voices,
            base_delay_samples,
            depth_samples,
            mix_smoother: ParameterSmoother::new(config.mix, 10.0, sample_rate),
            sample_rate,
            config,
        }
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: ChorusConfig) {
        if config.voices != self.config.voices {
            self.voices.clear();
            for i in 0..config.voices {
                let phase_offset = i as f64 / config.voices as f64;
                self.voices.push(ChorusVoice::new(
                    self.sample_rate,
                    config.rate,
                    phase_offset,
                    config.waveform,
                ));
            }
        } else {
            for voice in &mut self.voices {
                voice.lfo.set_rate(config.rate);
                voice.lfo.set_waveform(config.waveform);
            }
        }

        self.base_delay_samples = config.delay * 0.001 * self.sample_rate;
        self.depth_samples = config.depth * 0.001 * self.sample_rate;
        self.mix_smoother.set_target(config.mix);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ChorusConfig {
        &self.config
    }

    /// Process a single sample.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let mut wet = 0.0;
        let num_voices = self.voices.len() as f64;

        for voice in &mut self.voices {
            wet += voice.process(
                input,
                self.base_delay_samples,
                self.depth_samples,
                self.config.feedback,
            ) / num_voices;
        }

        let mix = self.mix_smoother.next();
        input * (1.0 - mix) + wet * mix
    }

    /// Process a buffer of samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input/output sample buffer
    pub fn process(&mut self, samples: &mut [f64]) {
        for sample in samples.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Set the wet/dry mix (0.0 = fully dry, 1.0 = fully wet).
    ///
    /// This provides a convenient way to adjust the mix at runtime
    /// without rebuilding the entire configuration.
    pub fn set_mix(&mut self, mix: f64) {
        let clamped = mix.clamp(0.0, 1.0);
        self.config.mix = clamped;
        self.mix_smoother.set_target(clamped);
    }

    /// Get the current wet/dry mix value.
    #[must_use]
    pub fn mix(&self) -> f64 {
        self.config.mix
    }

    /// Reset the effect state.
    pub fn reset(&mut self) {
        for voice in &mut self.voices {
            voice.reset();
        }
        self.mix_smoother.reset(self.config.mix);
    }
}

/// Stereo chorus effect processor.
pub struct StereoChorus {
    /// Configuration.
    config: ChorusConfig,
    /// Left channel voices.
    left_voices: Vec<ChorusVoice>,
    /// Right channel voices.
    right_voices: Vec<ChorusVoice>,
    /// Sample rate.
    sample_rate: f64,
    /// Base delay in samples.
    base_delay_samples: f64,
    /// Depth in samples.
    depth_samples: f64,
    /// Mix smoother.
    mix_smoother: ParameterSmoother,
}

impl StereoChorus {
    /// Create a new stereo chorus effect.
    ///
    /// # Arguments
    ///
    /// * `config` - Chorus configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: ChorusConfig, sample_rate: f64) -> Self {
        let mut left_voices = Vec::with_capacity(config.voices);
        let mut right_voices = Vec::with_capacity(config.voices);

        for i in 0..config.voices {
            let phase_offset = i as f64 / config.voices as f64;

            left_voices.push(ChorusVoice::new(
                sample_rate,
                config.rate,
                phase_offset,
                config.waveform,
            ));

            right_voices.push(ChorusVoice::new(
                sample_rate,
                config.rate,
                phase_offset + 0.5,
                config.waveform,
            ));
        }

        let base_delay_samples = config.delay * 0.001 * sample_rate;
        let depth_samples = config.depth * 0.001 * sample_rate;

        Self {
            left_voices,
            right_voices,
            base_delay_samples,
            depth_samples,
            mix_smoother: ParameterSmoother::new(config.mix, 10.0, sample_rate),
            sample_rate,
            config,
        }
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: ChorusConfig) {
        if config.voices != self.config.voices {
            self.left_voices.clear();
            self.right_voices.clear();

            for i in 0..config.voices {
                let phase_offset = i as f64 / config.voices as f64;

                self.left_voices.push(ChorusVoice::new(
                    self.sample_rate,
                    config.rate,
                    phase_offset,
                    config.waveform,
                ));

                self.right_voices.push(ChorusVoice::new(
                    self.sample_rate,
                    config.rate,
                    phase_offset + 0.5,
                    config.waveform,
                ));
            }
        } else {
            for voice in self
                .left_voices
                .iter_mut()
                .chain(self.right_voices.iter_mut())
            {
                voice.lfo.set_rate(config.rate);
                voice.lfo.set_waveform(config.waveform);
            }
        }

        self.base_delay_samples = config.delay * 0.001 * self.sample_rate;
        self.depth_samples = config.depth * 0.001 * self.sample_rate;
        self.mix_smoother.set_target(config.mix);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ChorusConfig {
        &self.config
    }

    /// Process stereo samples (interleaved).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved stereo input/output buffer
    /// * `num_frames` - Number of sample frames
    pub fn process_interleaved(&mut self, samples: &mut [f64], num_frames: usize) {
        for i in 0..num_frames {
            let left_idx = i * 2;
            let right_idx = i * 2 + 1;

            if left_idx >= samples.len() || right_idx >= samples.len() {
                break;
            }

            let (left_out, right_out) = self.process_frame(samples[left_idx], samples[right_idx]);

            samples[left_idx] = left_out;
            samples[right_idx] = right_out;
        }
    }

    /// Process stereo samples (planar).
    ///
    /// # Arguments
    ///
    /// * `left` - Left channel input/output buffer
    /// * `right` - Right channel input/output buffer
    pub fn process_planar(&mut self, left: &mut [f64], right: &mut [f64]) {
        let num_samples = left.len().min(right.len());

        for i in 0..num_samples {
            let (left_out, right_out) = self.process_frame(left[i], right[i]);
            left[i] = left_out;
            right[i] = right_out;
        }
    }

    /// Process a single stereo frame.
    fn process_frame(&mut self, left_in: f64, right_in: f64) -> (f64, f64) {
        let mut left_wet = 0.0;
        let mut right_wet = 0.0;
        let num_voices = self.left_voices.len() as f64;

        for voice in &mut self.left_voices {
            left_wet += voice.process(
                left_in,
                self.base_delay_samples,
                self.depth_samples,
                self.config.feedback,
            ) / num_voices;
        }

        for voice in &mut self.right_voices {
            right_wet += voice.process(
                right_in,
                self.base_delay_samples,
                self.depth_samples,
                self.config.feedback,
            ) / num_voices;
        }

        let mix = self.mix_smoother.next();
        let left_out = left_in * (1.0 - mix) + left_wet * mix;
        let right_out = right_in * (1.0 - mix) + right_wet * mix;

        (left_out, right_out)
    }

    /// Set the wet/dry mix (0.0 = fully dry, 1.0 = fully wet).
    pub fn set_mix(&mut self, mix: f64) {
        let clamped = mix.clamp(0.0, 1.0);
        self.config.mix = clamped;
        self.mix_smoother.set_target(clamped);
    }

    /// Get the current wet/dry mix value.
    #[must_use]
    pub fn mix(&self) -> f64 {
        self.config.mix
    }

    /// Reset the effect state.
    pub fn reset(&mut self) {
        for voice in self
            .left_voices
            .iter_mut()
            .chain(self.right_voices.iter_mut())
        {
            voice.reset();
        }
        self.mix_smoother.reset(self.config.mix);
    }
}

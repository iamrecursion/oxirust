//! Flanger effect implementation.
//!
//! A flanger creates a sweeping, jet-like sound by mixing the input signal
//! with a very short delayed copy that is modulated by an LFO. The delay
//! time is typically 0.5-10ms with feedback to enhance the effect.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use super::delay::{FractionalDelayLine, InterpolationMode};
use super::lfo::{Lfo, LfoWaveform, ParameterSmoother};

/// Minimum flanger delay in milliseconds.
const MIN_DELAY_MS: f64 = 0.5;

/// Maximum flanger delay in milliseconds.
const MAX_DELAY_MS: f64 = 10.0;

/// Default flanger delay in milliseconds.
const DEFAULT_DELAY_MS: f64 = 2.0;

/// Default flanger depth in milliseconds.
const DEFAULT_DEPTH_MS: f64 = 1.5;

/// Feedback mode for the flanger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FeedbackMode {
    /// Positive feedback (standard flanging).
    #[default]
    Positive,
    /// Negative feedback (inverted, more metallic sound).
    Negative,
}

/// Configuration for the flanger effect.
#[derive(Clone, Debug)]
pub struct FlangerConfig {
    /// LFO rate in Hz (0.05 to 10.0).
    pub rate: f64,
    /// Modulation depth in milliseconds (0.1 to 5.0).
    pub depth: f64,
    /// Base delay in milliseconds (0.5 to 10.0).
    pub delay: f64,
    /// Feedback amount (-0.95 to 0.95).
    pub feedback: f64,
    /// Feedback mode.
    pub feedback_mode: FeedbackMode,
    /// Dry/wet mix (0.0 = dry only, 1.0 = wet only).
    pub mix: f64,
    /// LFO waveform.
    pub waveform: LfoWaveform,
    /// Stereo phase offset (0.0 to 1.0).
    pub stereo_phase: f64,
}

impl Default for FlangerConfig {
    fn default() -> Self {
        Self {
            rate: 0.5,
            depth: DEFAULT_DEPTH_MS,
            delay: DEFAULT_DELAY_MS,
            feedback: 0.6,
            feedback_mode: FeedbackMode::Positive,
            mix: 0.5,
            waveform: LfoWaveform::Sine,
            stereo_phase: 0.5,
        }
    }
}

impl FlangerConfig {
    /// Create a new flanger configuration.
    ///
    /// # Arguments
    ///
    /// * `rate` - LFO rate in Hz
    /// * `depth` - Modulation depth in milliseconds
    /// * `feedback` - Feedback amount
    #[must_use]
    pub fn new(rate: f64, depth: f64, feedback: f64) -> Self {
        Self {
            rate: rate.clamp(0.05, 10.0),
            depth: depth.clamp(0.1, 5.0),
            feedback: feedback.clamp(-0.95, 0.95),
            ..Default::default()
        }
    }

    /// Set the LFO rate.
    #[must_use]
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate = rate.clamp(0.05, 10.0);
        self
    }

    /// Set the modulation depth.
    #[must_use]
    pub fn with_depth(mut self, depth: f64) -> Self {
        self.depth = depth.clamp(0.1, 5.0);
        self
    }

    /// Set the base delay.
    #[must_use]
    pub fn with_delay(mut self, delay: f64) -> Self {
        self.delay = delay.clamp(MIN_DELAY_MS, MAX_DELAY_MS);
        self
    }

    /// Set the feedback amount.
    #[must_use]
    pub fn with_feedback(mut self, feedback: f64) -> Self {
        self.feedback = feedback.clamp(-0.95, 0.95);
        self
    }

    /// Set the feedback mode.
    #[must_use]
    pub fn with_feedback_mode(mut self, mode: FeedbackMode) -> Self {
        self.feedback_mode = mode;
        self
    }

    /// Set the dry/wet mix.
    #[must_use]
    pub fn with_mix(mut self, mix: f64) -> Self {
        self.mix = mix.clamp(0.0, 1.0);
        self
    }

    /// Set the LFO waveform.
    #[must_use]
    pub fn with_waveform(mut self, waveform: LfoWaveform) -> Self {
        self.waveform = waveform;
        self
    }

    /// Set the stereo phase offset.
    #[must_use]
    pub fn with_stereo_phase(mut self, phase: f64) -> Self {
        self.stereo_phase = phase.clamp(0.0, 1.0);
        self
    }

    /// Create a classic flanger preset.
    #[must_use]
    pub fn classic() -> Self {
        Self {
            rate: 0.5,
            depth: 1.5,
            delay: 2.0,
            feedback: 0.7,
            feedback_mode: FeedbackMode::Positive,
            mix: 0.5,
            waveform: LfoWaveform::Sine,
            stereo_phase: 0.5,
        }
    }

    /// Create a jet flanger preset (fast, high feedback).
    #[must_use]
    pub fn jet() -> Self {
        Self {
            rate: 2.0,
            depth: 2.5,
            delay: 1.5,
            feedback: 0.8,
            feedback_mode: FeedbackMode::Positive,
            mix: 0.6,
            waveform: LfoWaveform::Triangle,
            stereo_phase: 0.5,
        }
    }

    /// Create a metallic flanger preset (negative feedback).
    #[must_use]
    pub fn metallic() -> Self {
        Self {
            rate: 0.8,
            depth: 2.0,
            delay: 2.5,
            feedback: 0.7,
            feedback_mode: FeedbackMode::Negative,
            mix: 0.5,
            waveform: LfoWaveform::Sine,
            stereo_phase: 0.5,
        }
    }

    /// Create a through-zero flanger preset.
    #[must_use]
    pub fn through_zero() -> Self {
        Self {
            rate: 0.3,
            depth: 4.0,
            delay: 5.0,
            feedback: 0.5,
            feedback_mode: FeedbackMode::Positive,
            mix: 0.5,
            waveform: LfoWaveform::Triangle,
            stereo_phase: 0.5,
        }
    }
}

/// Mono flanger effect processor.
pub struct MonoFlanger {
    /// Configuration.
    config: FlangerConfig,
    /// Fractional delay line.
    delay_line: FractionalDelayLine,
    /// LFO for modulation.
    lfo: Lfo,
    /// Sample rate.
    sample_rate: f64,
    /// Base delay in samples.
    base_delay_samples: f64,
    /// Depth in samples.
    depth_samples: f64,
    /// Mix smoother.
    mix_smoother: ParameterSmoother,
    /// Feedback smoother.
    feedback_smoother: ParameterSmoother,
}

impl MonoFlanger {
    /// Create a new mono flanger effect.
    ///
    /// # Arguments
    ///
    /// * `config` - Flanger configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: FlangerConfig, sample_rate: f64) -> Self {
        let max_delay_ms = 20.0;
        let base_delay_samples = config.delay * 0.001 * sample_rate;
        let depth_samples = config.depth * 0.001 * sample_rate;

        Self {
            delay_line: FractionalDelayLine::new(
                max_delay_ms,
                sample_rate,
                InterpolationMode::Linear,
            ),
            lfo: Lfo::new(config.rate, sample_rate, config.waveform),
            base_delay_samples,
            depth_samples,
            mix_smoother: ParameterSmoother::new(config.mix, 10.0, sample_rate),
            feedback_smoother: ParameterSmoother::new(config.feedback, 10.0, sample_rate),
            sample_rate,
            config,
        }
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: FlangerConfig) {
        self.lfo.set_rate(config.rate);
        self.lfo.set_waveform(config.waveform);
        self.base_delay_samples = config.delay * 0.001 * self.sample_rate;
        self.depth_samples = config.depth * 0.001 * self.sample_rate;
        self.mix_smoother.set_target(config.mix);
        self.feedback_smoother.set_target(config.feedback);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &FlangerConfig {
        &self.config
    }

    /// Process a single sample.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let lfo_value = self.lfo.next();
        let modulated_delay = self.base_delay_samples + lfo_value * self.depth_samples;

        let feedback = self.feedback_smoother.next();
        let feedback_adjusted = match self.config.feedback_mode {
            FeedbackMode::Positive => feedback,
            FeedbackMode::Negative => -feedback,
        };

        let wet = self
            .delay_line
            .process_with_feedback(input, modulated_delay, feedback_adjusted);

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
        self.delay_line.reset();
        self.lfo.reset();
        self.mix_smoother.reset(self.config.mix);
        self.feedback_smoother.reset(self.config.feedback);
    }
}

/// Stereo flanger effect processor.
pub struct StereoFlanger {
    /// Configuration.
    config: FlangerConfig,
    /// Left channel delay line.
    left_delay: FractionalDelayLine,
    /// Right channel delay line.
    right_delay: FractionalDelayLine,
    /// Left channel LFO.
    left_lfo: Lfo,
    /// Right channel LFO.
    right_lfo: Lfo,
    /// Sample rate.
    sample_rate: f64,
    /// Base delay in samples.
    base_delay_samples: f64,
    /// Depth in samples.
    depth_samples: f64,
    /// Mix smoother.
    mix_smoother: ParameterSmoother,
    /// Feedback smoother.
    feedback_smoother: ParameterSmoother,
}

impl StereoFlanger {
    /// Create a new stereo flanger effect.
    ///
    /// # Arguments
    ///
    /// * `config` - Flanger configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: FlangerConfig, sample_rate: f64) -> Self {
        let max_delay_ms = 20.0;
        let base_delay_samples = config.delay * 0.001 * sample_rate;
        let depth_samples = config.depth * 0.001 * sample_rate;

        let left_lfo = Lfo::new(config.rate, sample_rate, config.waveform);
        let mut right_lfo = Lfo::new(config.rate, sample_rate, config.waveform);
        right_lfo.set_phase(config.stereo_phase);

        Self {
            left_delay: FractionalDelayLine::new(
                max_delay_ms,
                sample_rate,
                InterpolationMode::Linear,
            ),
            right_delay: FractionalDelayLine::new(
                max_delay_ms,
                sample_rate,
                InterpolationMode::Linear,
            ),
            left_lfo,
            right_lfo,
            base_delay_samples,
            depth_samples,
            mix_smoother: ParameterSmoother::new(config.mix, 10.0, sample_rate),
            feedback_smoother: ParameterSmoother::new(config.feedback, 10.0, sample_rate),
            sample_rate,
            config,
        }
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: FlangerConfig) {
        self.left_lfo.set_rate(config.rate);
        self.left_lfo.set_waveform(config.waveform);
        self.right_lfo.set_rate(config.rate);
        self.right_lfo.set_waveform(config.waveform);

        let left_phase = self.left_lfo.phase();
        self.right_lfo
            .set_phase((left_phase + config.stereo_phase).fract());

        self.base_delay_samples = config.delay * 0.001 * self.sample_rate;
        self.depth_samples = config.depth * 0.001 * self.sample_rate;
        self.mix_smoother.set_target(config.mix);
        self.feedback_smoother.set_target(config.feedback);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &FlangerConfig {
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
        let left_lfo = self.left_lfo.next();
        let right_lfo = self.right_lfo.next();

        let left_delay = self.base_delay_samples + left_lfo * self.depth_samples;
        let right_delay = self.base_delay_samples + right_lfo * self.depth_samples;

        let feedback = self.feedback_smoother.next();
        let feedback_adjusted = match self.config.feedback_mode {
            FeedbackMode::Positive => feedback,
            FeedbackMode::Negative => -feedback,
        };

        let left_wet =
            self.left_delay
                .process_with_feedback(left_in, left_delay, feedback_adjusted);
        let right_wet =
            self.right_delay
                .process_with_feedback(right_in, right_delay, feedback_adjusted);

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
        self.left_delay.reset();
        self.right_delay.reset();
        self.left_lfo.reset();
        self.right_lfo.reset();
        self.mix_smoother.reset(self.config.mix);
        self.feedback_smoother.reset(self.config.feedback);
    }
}

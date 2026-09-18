//! Phaser effect implementation.
//!
//! A phaser creates a sweeping sound by using a series of all-pass filters
//! whose cutoff frequencies are modulated by an LFO. The phase-shifted
//! signal is mixed with the original, creating notches in the frequency spectrum.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use super::delay::FirstOrderAllPass;
use super::lfo::{Lfo, LfoWaveform, ParameterSmoother};

/// Minimum number of phaser stages.
pub const MIN_PHASER_STAGES: usize = 2;

/// Maximum number of phaser stages.
pub const MAX_PHASER_STAGES: usize = 12;

/// Default number of phaser stages.
const DEFAULT_STAGES: usize = 6;

/// Default center frequency in Hz.
const DEFAULT_CENTER_FREQ: f64 = 1000.0;

/// Default frequency range in Hz.
const DEFAULT_FREQ_RANGE: f64 = 800.0;

/// Phaser feedback mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PhaserFeedbackMode {
    /// Positive feedback (standard phasing).
    #[default]
    Positive,
    /// Negative feedback (inverted, more pronounced notches).
    Negative,
}

/// Configuration for the phaser effect.
#[derive(Clone, Debug)]
pub struct PhaserConfig {
    /// Number of all-pass filter stages (2-12, must be even).
    pub stages: usize,
    /// LFO rate in Hz (0.05 to 10.0).
    pub rate: f64,
    /// Modulation depth (0.0 to 1.0).
    pub depth: f64,
    /// Center frequency in Hz (200.0 to 5000.0).
    pub center_frequency: f64,
    /// Frequency range in Hz (100.0 to 3000.0).
    pub frequency_range: f64,
    /// Resonance/feedback amount (0.0 to 0.95).
    pub resonance: f64,
    /// Feedback mode.
    pub feedback_mode: PhaserFeedbackMode,
    /// Dry/wet mix (0.0 = dry only, 1.0 = wet only).
    pub mix: f64,
    /// LFO waveform.
    pub waveform: LfoWaveform,
    /// Stereo phase offset (0.0 to 1.0).
    pub stereo_phase: f64,
}

impl Default for PhaserConfig {
    fn default() -> Self {
        Self {
            stages: DEFAULT_STAGES,
            rate: 0.5,
            depth: 1.0,
            center_frequency: DEFAULT_CENTER_FREQ,
            frequency_range: DEFAULT_FREQ_RANGE,
            resonance: 0.5,
            feedback_mode: PhaserFeedbackMode::Positive,
            mix: 0.5,
            waveform: LfoWaveform::Sine,
            stereo_phase: 0.5,
        }
    }
}

impl PhaserConfig {
    /// Create a new phaser configuration.
    ///
    /// # Arguments
    ///
    /// * `stages` - Number of all-pass filter stages (2-12, must be even)
    /// * `rate` - LFO rate in Hz
    /// * `depth` - Modulation depth (0.0 to 1.0)
    #[must_use]
    pub fn new(stages: usize, rate: f64, depth: f64) -> Self {
        let stages_clamped = stages.clamp(MIN_PHASER_STAGES, MAX_PHASER_STAGES);
        let stages_even = if stages_clamped % 2 == 0 {
            stages_clamped
        } else {
            stages_clamped + 1
        };

        Self {
            stages: stages_even,
            rate: rate.clamp(0.05, 10.0),
            depth: depth.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Set the number of stages.
    #[must_use]
    pub fn with_stages(mut self, stages: usize) -> Self {
        let stages_clamped = stages.clamp(MIN_PHASER_STAGES, MAX_PHASER_STAGES);
        self.stages = if stages_clamped % 2 == 0 {
            stages_clamped
        } else {
            stages_clamped + 1
        };
        self
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
        self.depth = depth.clamp(0.0, 1.0);
        self
    }

    /// Set the center frequency.
    #[must_use]
    pub fn with_center_frequency(mut self, freq: f64) -> Self {
        self.center_frequency = freq.clamp(200.0, 5000.0);
        self
    }

    /// Set the frequency range.
    #[must_use]
    pub fn with_frequency_range(mut self, range: f64) -> Self {
        self.frequency_range = range.clamp(100.0, 3000.0);
        self
    }

    /// Set the resonance amount.
    #[must_use]
    pub fn with_resonance(mut self, resonance: f64) -> Self {
        self.resonance = resonance.clamp(0.0, 0.95);
        self
    }

    /// Set the feedback mode.
    #[must_use]
    pub fn with_feedback_mode(mut self, mode: PhaserFeedbackMode) -> Self {
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

    /// Create a classic phaser preset.
    #[must_use]
    pub fn classic() -> Self {
        Self {
            stages: 4,
            rate: 0.5,
            depth: 1.0,
            center_frequency: 1000.0,
            frequency_range: 800.0,
            resonance: 0.6,
            feedback_mode: PhaserFeedbackMode::Positive,
            mix: 0.5,
            waveform: LfoWaveform::Sine,
            stereo_phase: 0.5,
        }
    }

    /// Create a deep phaser preset (more stages, higher resonance).
    #[must_use]
    pub fn deep() -> Self {
        Self {
            stages: 8,
            rate: 0.3,
            depth: 1.0,
            center_frequency: 800.0,
            frequency_range: 1000.0,
            resonance: 0.7,
            feedback_mode: PhaserFeedbackMode::Positive,
            mix: 0.6,
            waveform: LfoWaveform::Triangle,
            stereo_phase: 0.5,
        }
    }

    /// Create a fast phaser preset.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            stages: 6,
            rate: 2.0,
            depth: 0.8,
            center_frequency: 1200.0,
            frequency_range: 600.0,
            resonance: 0.5,
            feedback_mode: PhaserFeedbackMode::Positive,
            mix: 0.5,
            waveform: LfoWaveform::Sine,
            stereo_phase: 0.5,
        }
    }

    /// Create a vibrato phaser preset (high frequency modulation).
    #[must_use]
    pub fn vibrato() -> Self {
        Self {
            stages: 4,
            rate: 5.0,
            depth: 0.5,
            center_frequency: 1500.0,
            frequency_range: 500.0,
            resonance: 0.3,
            feedback_mode: PhaserFeedbackMode::Positive,
            mix: 0.4,
            waveform: LfoWaveform::Sine,
            stereo_phase: 0.5,
        }
    }

    /// Create a barber pole phaser preset (continuously rising).
    #[must_use]
    pub fn barber_pole() -> Self {
        Self {
            stages: 10,
            rate: 0.2,
            depth: 1.0,
            center_frequency: 1000.0,
            frequency_range: 1200.0,
            resonance: 0.8,
            feedback_mode: PhaserFeedbackMode::Positive,
            mix: 0.7,
            waveform: LfoWaveform::Sawtooth,
            stereo_phase: 0.25,
        }
    }
}

/// Mono phaser effect processor.
pub struct MonoPhaser {
    /// Configuration.
    config: PhaserConfig,
    /// All-pass filter stages.
    all_pass_filters: Vec<FirstOrderAllPass>,
    /// LFO for modulation.
    lfo: Lfo,
    /// Sample rate.
    sample_rate: f64,
    /// Mix smoother.
    mix_smoother: ParameterSmoother,
    /// Resonance smoother.
    resonance_smoother: ParameterSmoother,
    /// Feedback buffer.
    feedback_sample: f64,
}

impl MonoPhaser {
    /// Create a new mono phaser effect.
    ///
    /// # Arguments
    ///
    /// * `config` - Phaser configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: PhaserConfig, sample_rate: f64) -> Self {
        let all_pass_filters: Vec<FirstOrderAllPass> = (0..config.stages)
            .map(|_| FirstOrderAllPass::new(config.center_frequency, sample_rate))
            .collect();

        Self {
            all_pass_filters,
            lfo: Lfo::new(config.rate, sample_rate, config.waveform),
            mix_smoother: ParameterSmoother::new(config.mix, 10.0, sample_rate),
            resonance_smoother: ParameterSmoother::new(config.resonance, 10.0, sample_rate),
            feedback_sample: 0.0,
            sample_rate,
            config,
        }
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: PhaserConfig) {
        if config.stages != self.config.stages {
            self.all_pass_filters = (0..config.stages)
                .map(|_| FirstOrderAllPass::new(config.center_frequency, self.sample_rate))
                .collect();
        }

        self.lfo.set_rate(config.rate);
        self.lfo.set_waveform(config.waveform);
        self.mix_smoother.set_target(config.mix);
        self.resonance_smoother.set_target(config.resonance);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &PhaserConfig {
        &self.config
    }

    /// Calculate modulated frequency from LFO value.
    fn calculate_frequency(&self, lfo_value: f64) -> f64 {
        let normalized = (lfo_value + 1.0) * 0.5;
        let depth_scaled = normalized * self.config.depth;
        self.config.center_frequency + (depth_scaled - 0.5) * self.config.frequency_range
    }

    /// Process a single sample.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let lfo_value = self.lfo.next();
        let frequency = self.calculate_frequency(lfo_value);

        for filter in &mut self.all_pass_filters {
            filter.set_frequency(frequency, self.sample_rate);
        }

        let resonance = self.resonance_smoother.next();
        let feedback_adjusted = match self.config.feedback_mode {
            PhaserFeedbackMode::Positive => resonance,
            PhaserFeedbackMode::Negative => -resonance,
        };

        let mut signal = input + self.feedback_sample * feedback_adjusted;

        for filter in &mut self.all_pass_filters {
            signal = filter.process(signal);
        }

        self.feedback_sample = signal;

        let mix = self.mix_smoother.next();
        input * (1.0 - mix) + signal * mix
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
        for filter in &mut self.all_pass_filters {
            filter.reset();
        }
        self.lfo.reset();
        self.mix_smoother.reset(self.config.mix);
        self.resonance_smoother.reset(self.config.resonance);
        self.feedback_sample = 0.0;
    }
}

/// Stereo phaser effect processor.
pub struct StereoPhaser {
    /// Configuration.
    config: PhaserConfig,
    /// Left channel all-pass filters.
    left_filters: Vec<FirstOrderAllPass>,
    /// Right channel all-pass filters.
    right_filters: Vec<FirstOrderAllPass>,
    /// Left channel LFO.
    left_lfo: Lfo,
    /// Right channel LFO.
    right_lfo: Lfo,
    /// Sample rate.
    sample_rate: f64,
    /// Mix smoother.
    mix_smoother: ParameterSmoother,
    /// Resonance smoother.
    resonance_smoother: ParameterSmoother,
    /// Left channel feedback buffer.
    left_feedback: f64,
    /// Right channel feedback buffer.
    right_feedback: f64,
}

impl StereoPhaser {
    /// Create a new stereo phaser effect.
    ///
    /// # Arguments
    ///
    /// * `config` - Phaser configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: PhaserConfig, sample_rate: f64) -> Self {
        let left_filters: Vec<FirstOrderAllPass> = (0..config.stages)
            .map(|_| FirstOrderAllPass::new(config.center_frequency, sample_rate))
            .collect();

        let right_filters: Vec<FirstOrderAllPass> = (0..config.stages)
            .map(|_| FirstOrderAllPass::new(config.center_frequency, sample_rate))
            .collect();

        let left_lfo = Lfo::new(config.rate, sample_rate, config.waveform);
        let mut right_lfo = Lfo::new(config.rate, sample_rate, config.waveform);
        right_lfo.set_phase(config.stereo_phase);

        Self {
            left_filters,
            right_filters,
            left_lfo,
            right_lfo,
            mix_smoother: ParameterSmoother::new(config.mix, 10.0, sample_rate),
            resonance_smoother: ParameterSmoother::new(config.resonance, 10.0, sample_rate),
            left_feedback: 0.0,
            right_feedback: 0.0,
            sample_rate,
            config,
        }
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: PhaserConfig) {
        if config.stages != self.config.stages {
            self.left_filters = (0..config.stages)
                .map(|_| FirstOrderAllPass::new(config.center_frequency, self.sample_rate))
                .collect();
            self.right_filters = (0..config.stages)
                .map(|_| FirstOrderAllPass::new(config.center_frequency, self.sample_rate))
                .collect();
        }

        self.left_lfo.set_rate(config.rate);
        self.left_lfo.set_waveform(config.waveform);
        self.right_lfo.set_rate(config.rate);
        self.right_lfo.set_waveform(config.waveform);

        let left_phase = self.left_lfo.phase();
        self.right_lfo
            .set_phase((left_phase + config.stereo_phase).fract());

        self.mix_smoother.set_target(config.mix);
        self.resonance_smoother.set_target(config.resonance);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &PhaserConfig {
        &self.config
    }

    /// Calculate modulated frequency from LFO value.
    fn calculate_frequency(&self, lfo_value: f64) -> f64 {
        let normalized = (lfo_value + 1.0) * 0.5;
        let depth_scaled = normalized * self.config.depth;
        self.config.center_frequency + (depth_scaled - 0.5) * self.config.frequency_range
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

        let left_freq = self.calculate_frequency(left_lfo);
        let right_freq = self.calculate_frequency(right_lfo);

        for filter in &mut self.left_filters {
            filter.set_frequency(left_freq, self.sample_rate);
        }

        for filter in &mut self.right_filters {
            filter.set_frequency(right_freq, self.sample_rate);
        }

        let resonance = self.resonance_smoother.next();
        let feedback_adjusted = match self.config.feedback_mode {
            PhaserFeedbackMode::Positive => resonance,
            PhaserFeedbackMode::Negative => -resonance,
        };

        let mut left_signal = left_in + self.left_feedback * feedback_adjusted;
        let mut right_signal = right_in + self.right_feedback * feedback_adjusted;

        for filter in &mut self.left_filters {
            left_signal = filter.process(left_signal);
        }

        for filter in &mut self.right_filters {
            right_signal = filter.process(right_signal);
        }

        self.left_feedback = left_signal;
        self.right_feedback = right_signal;

        let mix = self.mix_smoother.next();
        let left_out = left_in * (1.0 - mix) + left_signal * mix;
        let right_out = right_in * (1.0 - mix) + right_signal * mix;

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
        for filter in self
            .left_filters
            .iter_mut()
            .chain(self.right_filters.iter_mut())
        {
            filter.reset();
        }
        self.left_lfo.reset();
        self.right_lfo.reset();
        self.mix_smoother.reset(self.config.mix);
        self.resonance_smoother.reset(self.config.resonance);
        self.left_feedback = 0.0;
        self.right_feedback = 0.0;
    }
}

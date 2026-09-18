//! Automatic ducking (reducing main audio) for audio description.
//!
//! This module provides automatic gain reduction of the main audio track
//! when audio description is active, with smooth transitions, envelope
//! following, and voice activity detection.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;

/// Configuration for automatic ducking.
#[derive(Clone, Debug)]
pub struct DuckingConfig {
    /// Target gain reduction in dB when ducking is active.
    pub target_gain_db: f64,
    /// Attack time in milliseconds (how fast to duck).
    pub attack_ms: f64,
    /// Release time in milliseconds (how fast to return to normal).
    pub release_ms: f64,
    /// Threshold for voice activity detection in dB.
    pub vad_threshold_db: f64,
    /// Enable frequency-dependent ducking.
    pub frequency_dependent: bool,
    /// Frequency bands for frequency-dependent ducking.
    pub frequency_bands: Vec<FrequencyBand>,
    /// Lookahead time in milliseconds.
    pub lookahead_ms: f64,
    /// Hold time after AD ends in milliseconds.
    pub hold_ms: f64,
}

impl Default for DuckingConfig {
    fn default() -> Self {
        Self {
            target_gain_db: -12.0,
            attack_ms: 20.0,
            release_ms: 200.0,
            vad_threshold_db: -40.0,
            frequency_dependent: false,
            frequency_bands: vec![
                FrequencyBand::new(200.0, 3000.0, -6.0),
                FrequencyBand::new(3000.0, 8000.0, -12.0),
            ],
            lookahead_ms: 10.0,
            hold_ms: 50.0,
        }
    }
}

impl DuckingConfig {
    /// Create a new ducking configuration.
    #[must_use]
    pub fn new(target_gain_db: f64) -> Self {
        Self {
            target_gain_db,
            ..Default::default()
        }
    }

    /// Set attack and release times.
    #[must_use]
    pub fn with_timing(mut self, attack_ms: f64, release_ms: f64) -> Self {
        self.attack_ms = attack_ms.max(0.1);
        self.release_ms = release_ms.max(0.1);
        self
    }

    /// Set voice activity detection threshold.
    #[must_use]
    pub fn with_vad_threshold(mut self, threshold_db: f64) -> Self {
        self.vad_threshold_db = threshold_db;
        self
    }

    /// Enable frequency-dependent ducking.
    #[must_use]
    pub fn with_frequency_dependent(mut self, enabled: bool) -> Self {
        self.frequency_dependent = enabled;
        self
    }

    /// Set lookahead time.
    #[must_use]
    pub fn with_lookahead(mut self, lookahead_ms: f64) -> Self {
        self.lookahead_ms = lookahead_ms.max(0.0);
        self
    }

    /// Set hold time.
    #[must_use]
    pub fn with_hold(mut self, hold_ms: f64) -> Self {
        self.hold_ms = hold_ms.max(0.0);
        self
    }

    /// Convert dB to linear gain.
    #[must_use]
    pub fn db_to_linear(db: f64) -> f64 {
        10.0_f64.powf(db / 20.0)
    }

    /// Convert linear gain to dB.
    #[must_use]
    pub fn linear_to_db(linear: f64) -> f64 {
        if linear <= 0.0 {
            f64::NEG_INFINITY
        } else {
            20.0 * linear.log10()
        }
    }

    /// Create a broadcast ducking preset.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            target_gain_db: -15.0,
            attack_ms: 10.0,
            release_ms: 150.0,
            vad_threshold_db: -35.0,
            frequency_dependent: true,
            frequency_bands: vec![
                FrequencyBand::new(200.0, 3000.0, -8.0),
                FrequencyBand::new(3000.0, 8000.0, -15.0),
            ],
            lookahead_ms: 15.0,
            hold_ms: 100.0,
        }
    }

    /// Create a gentle ducking preset.
    #[must_use]
    pub fn gentle() -> Self {
        Self {
            target_gain_db: -6.0,
            attack_ms: 50.0,
            release_ms: 300.0,
            vad_threshold_db: -45.0,
            frequency_dependent: false,
            frequency_bands: Vec::new(),
            lookahead_ms: 5.0,
            hold_ms: 50.0,
        }
    }
}

/// Frequency band for frequency-dependent ducking.
#[derive(Clone, Debug)]
pub struct FrequencyBand {
    /// Lower frequency bound in Hz.
    pub freq_low: f64,
    /// Upper frequency bound in Hz.
    pub freq_high: f64,
    /// Target gain reduction for this band in dB.
    pub target_gain_db: f64,
}

impl FrequencyBand {
    /// Create a new frequency band.
    #[must_use]
    pub fn new(freq_low: f64, freq_high: f64, target_gain_db: f64) -> Self {
        Self {
            freq_low,
            freq_high,
            target_gain_db,
        }
    }
}

/// Envelope follower for gain reduction.
struct EnvelopeFollower {
    /// Current envelope level (0.0-1.0, where 1.0 is no reduction).
    envelope: f64,
    /// Attack coefficient.
    attack_coeff: f64,
    /// Release coefficient.
    release_coeff: f64,
    /// Hold counter in samples.
    hold_counter: usize,
    /// Hold duration in samples.
    hold_samples: usize,
}

impl EnvelopeFollower {
    /// Create a new envelope follower.
    fn new(attack_ms: f64, release_ms: f64, hold_ms: f64, sample_rate: f64) -> Self {
        let attack_coeff = if attack_ms > 0.0 {
            (-1.0 / (attack_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        let release_coeff = if release_ms > 0.0 {
            (-1.0 / (release_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        let hold_samples = (hold_ms * 0.001 * sample_rate) as usize;

        Self {
            envelope: 1.0,
            attack_coeff,
            release_coeff,
            hold_counter: 0,
            hold_samples,
        }
    }

    /// Update envelope with new target gain.
    fn update(&mut self, target_gain: f64, ad_active: bool) {
        if ad_active {
            self.hold_counter = self.hold_samples;
        }

        if self.hold_counter > 0 {
            self.hold_counter = self.hold_counter.saturating_sub(1);

            if target_gain < self.envelope {
                self.envelope =
                    self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * target_gain;
            }
        } else if target_gain > self.envelope {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * target_gain;
        } else {
            self.envelope =
                self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * target_gain;
        }
    }

    /// Get current envelope level.
    fn level(&self) -> f64 {
        self.envelope
    }

    /// Reset envelope.
    fn reset(&mut self) {
        self.envelope = 1.0;
        self.hold_counter = 0;
    }
}

/// Voice activity detector for the audio description track.
pub struct VoiceActivityDetector {
    /// Threshold in dB.
    threshold_db: f64,
    /// Minimum duration for voice activity in samples.
    min_duration_samples: usize,
    /// Current activity counter.
    activity_counter: usize,
    /// Is voice currently active?
    is_active: bool,
    /// Energy history for smoothing.
    energy_history: VecDeque<f64>,
    /// History size.
    history_size: usize,
}

impl VoiceActivityDetector {
    /// Create a new voice activity detector.
    #[must_use]
    pub fn new(threshold_db: f64, min_duration_ms: f64, sample_rate: f64) -> Self {
        let min_duration_samples = (min_duration_ms * 0.001 * sample_rate) as usize;
        let history_size = (50.0 * 0.001 * sample_rate) as usize;

        Self {
            threshold_db,
            min_duration_samples,
            activity_counter: 0,
            is_active: false,
            energy_history: VecDeque::with_capacity(history_size),
            history_size,
        }
    }

    /// Update VAD with new audio sample.
    pub fn update(&mut self, sample: f64) -> bool {
        let energy = sample * sample;

        self.energy_history.push_back(energy);
        if self.energy_history.len() > self.history_size {
            self.energy_history.pop_front();
        }

        let avg_energy: f64 =
            self.energy_history.iter().sum::<f64>() / self.energy_history.len() as f64;
        let rms = avg_energy.sqrt();
        let db = DuckingConfig::linear_to_db(rms);

        if db > self.threshold_db {
            self.activity_counter = self.activity_counter.saturating_add(1);
            if self.activity_counter >= self.min_duration_samples {
                self.is_active = true;
            }
        } else {
            self.activity_counter = 0;
            self.is_active = false;
        }

        self.is_active
    }

    /// Check if voice is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Reset VAD state.
    pub fn reset(&mut self) {
        self.activity_counter = 0;
        self.is_active = false;
        self.energy_history.clear();
    }
}

/// Lookahead buffer for smooth ducking transitions.
struct LookaheadBuffer {
    /// Circular buffer.
    buffer: VecDeque<f64>,
    /// Delay in samples.
    delay_samples: usize,
}

impl LookaheadBuffer {
    /// Create a new lookahead buffer.
    fn new(lookahead_ms: f64, sample_rate: f64) -> Self {
        let delay_samples = (lookahead_ms * 0.001 * sample_rate) as usize;

        Self {
            buffer: VecDeque::with_capacity(delay_samples + 1),
            delay_samples,
        }
    }

    /// Process one sample through the buffer.
    fn process(&mut self, input: f64) -> f64 {
        if self.delay_samples == 0 {
            return input;
        }

        self.buffer.push_back(input);

        if self.buffer.len() > self.delay_samples {
            self.buffer.pop_front().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Reset buffer.
    fn reset(&mut self) {
        self.buffer.clear();
    }
}

/// Automatic ducking processor.
pub struct AutomaticDucker {
    /// Configuration.
    config: DuckingConfig,
    /// Envelope follower.
    envelope: EnvelopeFollower,
    /// Voice activity detector.
    vad: VoiceActivityDetector,
    /// Lookahead buffers per channel.
    lookahead_buffers: Vec<LookaheadBuffer>,
    /// Target gain (linear).
    target_gain_linear: f64,
    /// Current gain reduction in dB (for metering).
    gain_reduction_db: f64,
    /// Sample rate.
    sample_rate: f64,
    /// Number of channels.
    channels: usize,
    /// Is audio description currently active?
    ad_active: bool,
}

impl AutomaticDucker {
    /// Create a new automatic ducker.
    ///
    /// # Arguments
    ///
    /// * `config` - Ducking configuration
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(config: DuckingConfig, sample_rate: f64, channels: usize) -> Self {
        let envelope = EnvelopeFollower::new(
            config.attack_ms,
            config.release_ms,
            config.hold_ms,
            sample_rate,
        );

        let vad = VoiceActivityDetector::new(config.vad_threshold_db, 20.0, sample_rate);

        let lookahead_buffers: Vec<_> = (0..channels)
            .map(|_| LookaheadBuffer::new(config.lookahead_ms, sample_rate))
            .collect();

        let target_gain_linear = DuckingConfig::db_to_linear(config.target_gain_db);

        Self {
            config,
            envelope,
            vad,
            lookahead_buffers,
            target_gain_linear,
            gain_reduction_db: 0.0,
            sample_rate,
            channels,
            ad_active: false,
        }
    }

    /// Set the ducking configuration.
    pub fn set_config(&mut self, config: DuckingConfig) {
        self.envelope = EnvelopeFollower::new(
            config.attack_ms,
            config.release_ms,
            config.hold_ms,
            self.sample_rate,
        );
        self.target_gain_linear = DuckingConfig::db_to_linear(config.target_gain_db);
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DuckingConfig {
        &self.config
    }

    /// Get current gain reduction in dB (for metering).
    #[must_use]
    pub fn gain_reduction_db(&self) -> f64 {
        self.gain_reduction_db
    }

    /// Set audio description active state.
    pub fn set_ad_active(&mut self, active: bool) {
        self.ad_active = active;
    }

    /// Check if audio description is active.
    #[must_use]
    pub fn is_ad_active(&self) -> bool {
        self.ad_active
    }

    /// Process interleaved main audio samples with ducking.
    ///
    /// # Arguments
    ///
    /// * `main_audio` - Main audio buffer (will be modified in place)
    /// * `ad_audio` - Audio description buffer (used for VAD)
    /// * `num_samples` - Number of samples per channel
    pub fn process_interleaved(
        &mut self,
        main_audio: &mut [f64],
        ad_audio: &[f64],
        num_samples: usize,
    ) {
        for i in 0..num_samples {
            let ad_sample_idx = i * self.channels;
            let ad_sample = if ad_sample_idx < ad_audio.len() {
                ad_audio[ad_sample_idx]
            } else {
                0.0
            };

            let voice_active = self.vad.update(ad_sample);
            let should_duck = self.ad_active && voice_active;

            let target_gain = if should_duck {
                self.target_gain_linear
            } else {
                1.0
            };

            self.envelope.update(target_gain, should_duck);
            let gain = self.envelope.level();

            self.gain_reduction_db = DuckingConfig::linear_to_db(gain);

            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if idx < main_audio.len() && ch < self.lookahead_buffers.len() {
                    let delayed = self.lookahead_buffers[ch].process(main_audio[idx]);
                    main_audio[idx] = delayed * gain;
                }
            }
        }
    }

    /// Process planar main audio samples with ducking.
    ///
    /// # Arguments
    ///
    /// * `main_channels` - Main audio channels (will be modified in place)
    /// * `ad_channel` - Audio description channel (used for VAD)
    pub fn process_planar(&mut self, main_channels: &mut [Vec<f64>], ad_channel: &[f64]) {
        if main_channels.is_empty() {
            return;
        }

        let num_samples = main_channels[0].len().min(ad_channel.len());

        for i in 0..num_samples {
            let ad_sample = ad_channel[i];
            let voice_active = self.vad.update(ad_sample);
            let should_duck = self.ad_active && voice_active;

            let target_gain = if should_duck {
                self.target_gain_linear
            } else {
                1.0
            };

            self.envelope.update(target_gain, should_duck);
            let gain = self.envelope.level();

            self.gain_reduction_db = DuckingConfig::linear_to_db(gain);

            for (ch, channel) in main_channels.iter_mut().enumerate() {
                if i < channel.len() && ch < self.lookahead_buffers.len() {
                    let delayed = self.lookahead_buffers[ch].process(channel[i]);
                    channel[i] = delayed * gain;
                }
            }
        }
    }

    /// Process without audio description input (manual control).
    ///
    /// # Arguments
    ///
    /// * `main_channels` - Main audio channels (will be modified in place)
    /// * `force_duck` - Force ducking active
    pub fn process_manual(&mut self, main_channels: &mut [Vec<f64>], force_duck: bool) {
        if main_channels.is_empty() {
            return;
        }

        let num_samples = main_channels[0].len();

        for i in 0..num_samples {
            let target_gain = if force_duck {
                self.target_gain_linear
            } else {
                1.0
            };

            self.envelope.update(target_gain, force_duck);
            let gain = self.envelope.level();

            self.gain_reduction_db = DuckingConfig::linear_to_db(gain);

            for (ch, channel) in main_channels.iter_mut().enumerate() {
                if i < channel.len() && ch < self.lookahead_buffers.len() {
                    let delayed = self.lookahead_buffers[ch].process(channel[i]);
                    channel[i] = delayed * gain;
                }
            }
        }
    }

    /// Reset all ducking state.
    pub fn reset(&mut self) {
        self.envelope.reset();
        self.vad.reset();
        for buffer in &mut self.lookahead_buffers {
            buffer.reset();
        }
        self.gain_reduction_db = 0.0;
        self.ad_active = false;
    }
}

/// Smooth gain fader for crossfades.
pub struct GainFader {
    /// Current gain (linear).
    current_gain: f64,
    /// Target gain (linear).
    target_gain: f64,
    /// Fade rate per sample.
    fade_rate: f64,
}

impl GainFader {
    /// Create a new gain fader.
    #[must_use]
    pub fn new(initial_gain: f64) -> Self {
        Self {
            current_gain: initial_gain,
            target_gain: initial_gain,
            fade_rate: 0.0,
        }
    }

    /// Start fading to target gain over duration.
    pub fn fade_to(&mut self, target_gain: f64, duration_samples: usize) {
        self.target_gain = target_gain;

        if duration_samples > 0 {
            self.fade_rate = (target_gain - self.current_gain) / duration_samples as f64;
        } else {
            self.current_gain = target_gain;
            self.fade_rate = 0.0;
        }
    }

    /// Process one sample and return current gain.
    pub fn process(&mut self) -> f64 {
        if (self.current_gain - self.target_gain).abs() > f64::EPSILON {
            self.current_gain += self.fade_rate;

            if (self.fade_rate > 0.0 && self.current_gain >= self.target_gain)
                || (self.fade_rate < 0.0 && self.current_gain <= self.target_gain)
            {
                self.current_gain = self.target_gain;
                self.fade_rate = 0.0;
            }
        }

        self.current_gain
    }

    /// Get current gain.
    #[must_use]
    pub fn current_gain(&self) -> f64 {
        self.current_gain
    }

    /// Check if fade is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        (self.current_gain - self.target_gain).abs() < f64::EPSILON
    }

    /// Reset to specified gain.
    pub fn reset(&mut self, gain: f64) {
        self.current_gain = gain;
        self.target_gain = gain;
        self.fade_rate = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ducking_config() {
        let config = DuckingConfig::new(-12.0);
        assert!((config.target_gain_db + 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_envelope_follower() {
        let mut envelope = EnvelopeFollower::new(10.0, 100.0, 50.0, 48000.0);
        assert!((envelope.level() - 1.0).abs() < f64::EPSILON);

        envelope.update(0.5, true);
        assert!(envelope.level() < 1.0);
    }

    #[test]
    fn test_vad() {
        let mut vad = VoiceActivityDetector::new(-40.0, 20.0, 48000.0);
        assert!(!vad.is_active());

        for _ in 0..1000 {
            vad.update(0.1);
        }
        assert!(vad.is_active());
    }

    #[test]
    fn test_gain_fader() {
        let mut fader = GainFader::new(1.0);
        fader.fade_to(0.0, 100);

        assert!(!fader.is_complete());
        assert!(fader.current_gain() > 0.0);

        for _ in 0..100 {
            fader.process();
        }

        assert!(fader.is_complete());
        assert!((fader.current_gain()).abs() < 0.01);
    }

    #[test]
    fn test_automatic_ducker() {
        let config = DuckingConfig::default();
        let mut ducker = AutomaticDucker::new(config, 48000.0, 2);

        let mut main_audio = vec![1.0; 200];
        let ad_audio = vec![0.1; 200];

        ducker.set_ad_active(true);
        ducker.process_interleaved(&mut main_audio, &ad_audio, 100);

        assert!(main_audio.iter().any(|&s| s < 1.0));
    }
}

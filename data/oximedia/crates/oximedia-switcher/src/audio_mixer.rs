//! Built-in audio mixer for video switchers.
//!
//! Provides a multi-channel audio mixer with per-input gain, pan, mute, and solo,
//! plus a flexible output routing system.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ─── Audio source ──────────────────────────────────────────────────────────────

/// Available audio sources for output routing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AudioSource {
    /// Mixed program output
    Program,
    /// Preview bus audio
    Preview,
    /// A specific input by ID
    Input(u32),
    /// An internal audio bus by ID
    Bus(u32),
    /// Silence generator
    SilenceGen,
}

// ─── Input configuration ───────────────────────────────────────────────────────

/// Configuration for a single audio input channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInputConfig {
    /// Input ID (matches video input)
    pub input_id: u32,
    /// Human-readable label
    pub label: String,
    /// When true, this input follows its companion video input on/off air state
    pub follow_video: bool,
    /// Channel gain in dBFS
    pub gain_db: f32,
    /// Synchronisation delay in video frames
    pub delay_frames: u32,
}

impl AudioInputConfig {
    /// Create a new audio input configuration.
    #[must_use]
    pub fn new(input_id: u32, label: impl Into<String>) -> Self {
        Self {
            input_id,
            label: label.into(),
            follow_video: false,
            gain_db: 0.0,
            delay_frames: 0,
        }
    }

    /// Enable audio-follow-video.
    #[must_use]
    pub fn with_follow_video(mut self) -> Self {
        self.follow_video = true;
        self
    }
}

// ─── Fader ────────────────────────────────────────────────────────────────────

/// Per-channel fader state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFader {
    /// Channel gain in dBFS (typically -inf … +10)
    pub gain_db: f32,
    /// Pan position in [-1.0 (full left) … +1.0 (full right)]
    pub pan: f32,
    /// Whether the channel is muted
    pub muted: bool,
    /// Whether the channel is soloed (only soloed channels play)
    pub solo: bool,
}

impl AudioFader {
    /// Create a unity-gain, centre-panned, unmuted fader.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gain_db: 0.0,
            pan: 0.0,
            muted: false,
            solo: false,
        }
    }

    /// Compute the effective linear gain (accounting for mute).
    ///
    /// Returns 0.0 when muted; otherwise converts dBFS to linear amplitude.
    #[must_use]
    pub fn effective_gain_linear(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            db_to_linear(self.gain_db)
        }
    }

    /// Left channel gain (linear, pan-adjusted).
    #[must_use]
    pub fn left_gain(&self) -> f32 {
        // Constant-power panning
        let angle = (self.pan + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2;
        self.effective_gain_linear() * angle.cos()
    }

    /// Right channel gain (linear, pan-adjusted).
    #[must_use]
    pub fn right_gain(&self) -> f32 {
        let angle = (self.pan + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2;
        self.effective_gain_linear() * angle.sin()
    }
}

impl Default for AudioFader {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Output routing ────────────────────────────────────────────────────────────

/// Routing configuration for audio outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOutputRouting {
    /// Headphone monitor source
    pub headphone_source: AudioSource,
    /// Broadcast monitor source
    pub monitor_source: AudioSource,
    /// Program (on-air) output source
    pub program_source: AudioSource,
}

impl AudioOutputRouting {
    /// Create a default routing (all outputs to Program).
    #[must_use]
    pub fn default_routing() -> Self {
        Self {
            headphone_source: AudioSource::Program,
            monitor_source: AudioSource::Program,
            program_source: AudioSource::Program,
        }
    }
}

impl Default for AudioOutputRouting {
    fn default() -> Self {
        Self::default_routing()
    }
}

// ─── Mixer state ──────────────────────────────────────────────────────────────

/// State of the entire audio mixer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMixerState {
    /// Per-input faders: (input_id, AudioFader)
    pub inputs: Vec<(u32, AudioFader)>,
    /// Master output gain in dBFS
    pub master_gain_db: f32,
    /// Monitor output gain in dBFS
    pub monitor_gain_db: f32,
    /// Output routing
    pub routing: AudioOutputRouting,
}

impl AudioMixerState {
    /// Create a new mixer state with the given input IDs.
    #[must_use]
    pub fn new(input_ids: &[u32]) -> Self {
        let inputs = input_ids
            .iter()
            .map(|&id| (id, AudioFader::new()))
            .collect();

        Self {
            inputs,
            master_gain_db: 0.0,
            monitor_gain_db: 0.0,
            routing: AudioOutputRouting::default_routing(),
        }
    }

    /// Get a reference to the fader for a given input ID.
    #[must_use]
    pub fn fader(&self, input_id: u32) -> Option<&AudioFader> {
        self.inputs
            .iter()
            .find(|(id, _)| *id == input_id)
            .map(|(_, f)| f)
    }

    /// Get a mutable reference to the fader for a given input ID.
    pub fn fader_mut(&mut self, input_id: u32) -> Option<&mut AudioFader> {
        self.inputs
            .iter_mut()
            .find(|(id, _)| *id == input_id)
            .map(|(_, f)| f)
    }

    /// Add a new input channel.
    pub fn add_input(&mut self, input_id: u32) {
        if self.fader(input_id).is_none() {
            self.inputs.push((input_id, AudioFader::new()));
        }
    }

    /// Remove an input channel.
    pub fn remove_input(&mut self, input_id: u32) {
        self.inputs.retain(|(id, _)| *id != input_id);
    }

    /// Check whether any channel has solo enabled.
    #[must_use]
    pub fn any_solo(&self) -> bool {
        self.inputs.iter().any(|(_, f)| f.solo)
    }

    /// Mix all active input samples to a mono output vector.
    ///
    /// `input_samples` is a slice of `(input_id, samples)` pairs.  Channels
    /// that are muted (or non-solo when any channel is soloed) are excluded.
    /// The master gain is applied to the final mix.
    #[must_use]
    pub fn mix(&self, input_samples: &[(u32, Vec<f32>)]) -> Vec<f32> {
        if input_samples.is_empty() {
            return Vec::new();
        }

        let frame_len = input_samples[0].1.len();
        let mut output = vec![0.0f32; frame_len];

        let has_solo = self.any_solo();
        let master_linear = db_to_linear(self.master_gain_db);

        for (input_id, samples) in input_samples {
            // Look up fader for this input; skip if unknown
            let Some(fader) = self.fader(*input_id) else {
                continue;
            };

            // Solo logic: if any channel is soloed, skip non-solo channels
            if has_solo && !fader.solo {
                continue;
            }

            let gain = fader.effective_gain_linear();
            for (out, &samp) in output.iter_mut().zip(samples.iter()) {
                *out += samp * gain;
            }
        }

        // Apply master gain
        for sample in &mut output {
            *sample *= master_linear;
        }

        output
    }
}

// ─── Delay buffer ─────────────────────────────────────────────────────────────

/// A fixed-latency delay buffer.
///
/// Pushing a value returns the oldest value once the buffer is full.
#[derive(Debug, Clone)]
pub struct DelayBuffer<T> {
    buffer: VecDeque<T>,
    delay: usize,
}

impl<T: Clone + Default> DelayBuffer<T> {
    /// Create a new delay buffer with the given latency in samples/frames.
    #[must_use]
    pub fn new(delay: usize) -> Self {
        let buffer = VecDeque::from(vec![T::default(); delay]);
        Self { buffer, delay }
    }

    /// Push a new value and pop the oldest.
    ///
    /// Returns `Some(delayed_value)` once the buffer has accumulated `delay`
    /// samples, `None` for an empty delay (pass-through).
    pub fn push(&mut self, val: T) -> Option<T> {
        if self.delay == 0 {
            return Some(val);
        }
        self.buffer.push_back(val);
        self.buffer.pop_front()
    }

    /// Current configured delay.
    #[must_use]
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// Change the delay, resetting the buffer contents to the default value.
    pub fn set_delay(&mut self, delay: usize) {
        self.delay = delay;
        self.buffer.clear();
        self.buffer.extend(vec![T::default(); delay]);
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Convert dBFS to linear amplitude.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_fader_unity_gain() {
        let fader = AudioFader::new();
        // 0 dB → linear 1.0
        assert!((fader.effective_gain_linear() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_audio_fader_muted() {
        let mut fader = AudioFader::new();
        fader.muted = true;
        assert_eq!(fader.effective_gain_linear(), 0.0);
    }

    #[test]
    fn test_audio_fader_gain_db() {
        let mut fader = AudioFader::new();
        fader.gain_db = 20.0; // +20 dB → 10.0 linear
        let gain = fader.effective_gain_linear();
        assert!((gain - 10.0).abs() < 1e-3);
    }

    #[test]
    fn test_audio_fader_pan_centre() {
        let fader = AudioFader::new(); // pan = 0
        let l = fader.left_gain();
        let r = fader.right_gain();
        // At centre, left ≈ right ≈ 1/√2
        let expected = 1.0 / std::f32::consts::SQRT_2;
        assert!((l - expected).abs() < 1e-5, "left={l}");
        assert!((r - expected).abs() < 1e-5, "right={r}");
    }

    #[test]
    fn test_audio_mixer_state_creation() {
        let state = AudioMixerState::new(&[0, 1, 2]);
        assert_eq!(state.inputs.len(), 3);
        assert!(state.fader(0).is_some());
        assert!(state.fader(3).is_none());
    }

    #[test]
    fn test_audio_mixer_mix_basic() {
        let state = AudioMixerState::new(&[0, 1]);
        let samples: Vec<(u32, Vec<f32>)> =
            vec![(0, vec![0.5, 0.5, 0.5]), (1, vec![0.5, 0.5, 0.5])];
        let out = state.mix(&samples);
        // Both at unity gain, sum = 1.0
        assert_eq!(out.len(), 3);
        for &s in &out {
            assert!((s - 1.0).abs() < 1e-5, "sample={s}");
        }
    }

    #[test]
    fn test_audio_mixer_mute() {
        let mut state = AudioMixerState::new(&[0, 1]);
        state.fader_mut(1).expect("should succeed in test").muted = true;
        let samples: Vec<(u32, Vec<f32>)> = vec![(0, vec![0.5]), (1, vec![0.5])];
        let out = state.mix(&samples);
        assert!((out[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_audio_mixer_solo() {
        let mut state = AudioMixerState::new(&[0, 1]);
        state.fader_mut(0).expect("should succeed in test").solo = true;
        let samples: Vec<(u32, Vec<f32>)> = vec![(0, vec![1.0]), (1, vec![1.0])];
        let out = state.mix(&samples);
        // Only channel 0 (solo) contributes
        assert!((out[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_audio_mixer_add_remove_input() {
        let mut state = AudioMixerState::new(&[0]);
        state.add_input(5);
        assert!(state.fader(5).is_some());
        state.remove_input(5);
        assert!(state.fader(5).is_none());
    }

    #[test]
    fn test_delay_buffer_passthrough() {
        let mut buf: DelayBuffer<f32> = DelayBuffer::new(0);
        assert_eq!(buf.push(42.0), Some(42.0));
    }

    #[test]
    fn test_delay_buffer_delay_1() {
        let mut buf: DelayBuffer<f32> = DelayBuffer::new(1);
        // First push returns the initial default (0.0)
        assert_eq!(buf.push(1.0), Some(0.0));
        // Second push returns the value we pushed first
        assert_eq!(buf.push(2.0), Some(1.0));
    }

    #[test]
    fn test_delay_buffer_delay_3() {
        let mut buf: DelayBuffer<i32> = DelayBuffer::new(3);
        // First 3 pushes drain the initial zero-buffer
        assert_eq!(buf.push(10), Some(0));
        assert_eq!(buf.push(20), Some(0));
        assert_eq!(buf.push(30), Some(0));
        // Now we start getting back the values we pushed
        assert_eq!(buf.push(40), Some(10));
        assert_eq!(buf.push(50), Some(20));
    }

    #[test]
    fn test_delay_buffer_set_delay() {
        let mut buf: DelayBuffer<f32> = DelayBuffer::new(5);
        buf.set_delay(2);
        assert_eq!(buf.delay(), 2);
        // Buffer should be reset
        assert_eq!(buf.push(1.0), Some(0.0));
    }

    #[test]
    fn test_audio_output_routing_default() {
        let routing = AudioOutputRouting::default();
        assert_eq!(routing.program_source, AudioSource::Program);
        assert_eq!(routing.monitor_source, AudioSource::Program);
        assert_eq!(routing.headphone_source, AudioSource::Program);
    }

    #[test]
    fn test_audio_input_config() {
        let cfg = AudioInputConfig::new(3, "Microphone").with_follow_video();
        assert_eq!(cfg.input_id, 3);
        assert_eq!(cfg.label, "Microphone");
        assert!(cfg.follow_video);
        assert_eq!(cfg.gain_db, 0.0);
    }
}

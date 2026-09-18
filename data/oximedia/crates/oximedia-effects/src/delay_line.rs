//! Multi-tap delay line effect with ping-pong and stereo modes.
//!
//! Provides a circular-buffer delay line with independent read taps,
//! feedback support, and dry/wet mixing.

#![allow(dead_code)]

/// Stereo routing mode for the delay effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayMode {
    /// Mono delay — both channels receive the same delayed signal.
    Mono,
    /// Stereo delay — left and right have independent delay times.
    Stereo,
    /// Ping-pong — signal alternates left and right on each repeat.
    Ping,
    /// Pong — identical to ping-pong but phase-inverted on alternating taps.
    Pong,
}

impl DelayMode {
    /// Number of independent delay channels implied by this mode.
    #[must_use]
    pub fn channels(self) -> usize {
        match self {
            DelayMode::Mono => 1,
            DelayMode::Stereo | DelayMode::Ping | DelayMode::Pong => 2,
        }
    }

    /// Returns `true` for modes that route signal between channels.
    #[must_use]
    pub fn is_ping_pong(self) -> bool {
        matches!(self, DelayMode::Ping | DelayMode::Pong)
    }
}

/// A single read tap in the delay line.
#[derive(Debug, Clone)]
pub struct DelayTap {
    /// Delay in samples for this tap.
    pub delay_samples: usize,
    /// Gain applied to this tap's output (0.0–1.0 recommended).
    pub gain: f32,
    /// Feedback fraction fed back into the write pointer (0.0–0.99).
    pub feedback: f32,
    /// Whether this tap is enabled.
    pub enabled: bool,
}

impl DelayTap {
    /// Create a tap with the given delay and gain, no feedback.
    #[must_use]
    pub fn new(delay_samples: usize, gain: f32) -> Self {
        Self {
            delay_samples,
            gain,
            feedback: 0.0,
            enabled: true,
        }
    }

    /// Builder: set feedback amount.
    #[must_use]
    pub fn with_feedback(mut self, feedback: f32) -> Self {
        self.feedback = feedback.clamp(0.0, 0.99);
        self
    }

    /// Returns `true` if this tap has any feedback component.
    #[must_use]
    pub fn is_feedback(&self) -> bool {
        self.feedback > 0.0
    }
}

/// Circular buffer delay line supporting multiple read taps.
pub struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
    /// Active taps.
    taps: Vec<DelayTap>,
    /// Current delay mode.
    pub mode: DelayMode,
    /// Dry signal mix level (0.0–1.0).
    pub dry_mix: f32,
    /// Wet signal mix level (0.0–1.0).
    pub wet_mix: f32,
}

impl DelayLine {
    /// Create a delay line capable of holding `capacity_samples` samples.
    #[must_use]
    pub fn new(capacity_samples: usize, mode: DelayMode) -> Self {
        Self {
            buffer: vec![0.0; capacity_samples.max(1)],
            write_pos: 0,
            taps: Vec::new(),
            mode,
            dry_mix: 1.0,
            wet_mix: 0.5,
        }
    }

    /// Write a new sample into the circular buffer and advance the write pointer.
    pub fn write_sample(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
    }

    /// Read the delayed sample for a specific tap index.
    ///
    /// Returns 0.0 if the tap index is out of range or disabled.
    #[must_use]
    pub fn read_tap(&self, tap_index: usize) -> f32 {
        let tap = match self.taps.get(tap_index) {
            Some(t) if t.enabled => t,
            _ => return 0.0,
        };
        let buf_len = self.buffer.len();
        let read_pos = (self.write_pos + buf_len - tap.delay_samples.min(buf_len - 1)) % buf_len;
        self.buffer[read_pos] * tap.gain
    }

    /// Mix all enabled taps into a single output sample.
    #[must_use]
    pub fn mix_taps(&self) -> f32 {
        (0..self.taps.len()).map(|i| self.read_tap(i)).sum()
    }

    /// Process one input sample: write it, apply feedback, return dry+wet mix.
    pub fn process(&mut self, input: f32) -> f32 {
        let wet = self.mix_taps();
        // Apply feedback from all feedback taps back into write position
        let feedback_sum: f32 = self
            .taps
            .iter()
            .filter(|t| t.is_feedback() && t.enabled)
            .map(|t| {
                let buf_len = self.buffer.len();
                let pos = (self.write_pos + buf_len - t.delay_samples.min(buf_len - 1)) % buf_len;
                self.buffer[pos] * t.feedback
            })
            .sum();
        self.write_sample(input + feedback_sum);
        input * self.dry_mix + wet * self.wet_mix
    }

    /// Add a tap to the delay line.
    pub fn add_tap(&mut self, tap: DelayTap) {
        self.taps.push(tap);
    }

    /// Number of taps registered.
    #[must_use]
    pub fn tap_count(&self) -> usize {
        self.taps.len()
    }

    /// Capacity of the delay buffer in samples.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Reset the buffer and write pointer.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_mode_channels() {
        assert_eq!(DelayMode::Mono.channels(), 1);
        assert_eq!(DelayMode::Stereo.channels(), 2);
        assert_eq!(DelayMode::Ping.channels(), 2);
        assert_eq!(DelayMode::Pong.channels(), 2);
    }

    #[test]
    fn test_delay_mode_is_ping_pong() {
        assert!(DelayMode::Ping.is_ping_pong());
        assert!(DelayMode::Pong.is_ping_pong());
        assert!(!DelayMode::Mono.is_ping_pong());
        assert!(!DelayMode::Stereo.is_ping_pong());
    }

    #[test]
    fn test_delay_tap_is_feedback_false() {
        let tap = DelayTap::new(100, 0.8);
        assert!(!tap.is_feedback());
    }

    #[test]
    fn test_delay_tap_is_feedback_true() {
        let tap = DelayTap::new(100, 0.8).with_feedback(0.5);
        assert!(tap.is_feedback());
        assert!((tap.feedback - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_delay_tap_feedback_clamped() {
        let tap = DelayTap::new(10, 1.0).with_feedback(2.0);
        assert!(tap.feedback <= 0.99);
    }

    #[test]
    fn test_delay_line_capacity() {
        let dl = DelayLine::new(512, DelayMode::Mono);
        assert_eq!(dl.capacity(), 512);
    }

    #[test]
    fn test_delay_line_empty_read() {
        let dl = DelayLine::new(128, DelayMode::Mono);
        assert_eq!(dl.read_tap(0), 0.0);
    }

    #[test]
    fn test_delay_line_write_and_read() {
        let mut dl = DelayLine::new(64, DelayMode::Mono);
        dl.add_tap(DelayTap::new(2, 1.0));
        dl.write_sample(0.5);
        dl.write_sample(0.0);
        // After writing 2 samples, read_tap(0) with delay=2 should see 0.5
        let out = dl.read_tap(0);
        assert!((out - 0.5).abs() < 1e-6, "got {out}");
    }

    #[test]
    fn test_delay_line_mix_taps_empty() {
        let dl = DelayLine::new(64, DelayMode::Stereo);
        assert_eq!(dl.mix_taps(), 0.0);
    }

    #[test]
    fn test_delay_line_tap_count() {
        let mut dl = DelayLine::new(128, DelayMode::Mono);
        assert_eq!(dl.tap_count(), 0);
        dl.add_tap(DelayTap::new(10, 0.5));
        dl.add_tap(DelayTap::new(20, 0.3));
        assert_eq!(dl.tap_count(), 2);
    }

    #[test]
    fn test_delay_line_reset() {
        let mut dl = DelayLine::new(32, DelayMode::Mono);
        dl.write_sample(1.0);
        dl.reset();
        // After reset buffer is all zeros
        dl.add_tap(DelayTap::new(1, 1.0));
        assert_eq!(dl.read_tap(0), 0.0);
    }

    #[test]
    fn test_delay_line_disabled_tap() {
        let mut dl = DelayLine::new(64, DelayMode::Mono);
        let mut tap = DelayTap::new(1, 1.0);
        tap.enabled = false;
        dl.add_tap(tap);
        dl.write_sample(1.0);
        assert_eq!(dl.read_tap(0), 0.0);
    }

    #[test]
    fn test_delay_line_process_dry() {
        let mut dl = DelayLine::new(64, DelayMode::Mono);
        dl.wet_mix = 0.0;
        dl.dry_mix = 1.0;
        let out = dl.process(0.7);
        assert!((out - 0.7).abs() < 1e-5, "got {out}");
    }
}

//! Audio warp effects: tape-style warping, pitch bend, and formant shifting.
//!
//! Provides time-domain manipulation techniques that distort audio playback
//! speed or pitch without changing the other dimension permanently.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::f32::consts::PI;

/// Tape-flutter LFO that modulates playback speed in real time.
///
/// The LFO outputs a rate multiplier centred around 1.0.
#[derive(Debug, Clone)]
pub struct TapeFlutter {
    /// Flutter rate in Hz.
    pub rate_hz: f32,
    /// Flutter depth (0–1).
    pub depth: f32,
    /// Current LFO phase.
    phase: f32,
    /// Sample rate in Hz.
    sample_rate: f32,
}

impl TapeFlutter {
    /// Create a new [`TapeFlutter`].
    #[must_use]
    pub fn new(rate_hz: f32, depth: f32, sample_rate: f32) -> Self {
        Self {
            rate_hz: rate_hz.max(0.01),
            depth: depth.clamp(0.0, 1.0),
            phase: 0.0,
            sample_rate: sample_rate.max(1.0),
        }
    }

    /// Advance LFO by one sample and return the current rate multiplier.
    ///
    /// Returns a value centred on 1.0 ± `depth / 10`.
    pub fn tick(&mut self) -> f32 {
        let lfo = (2.0 * PI * self.phase).sin();
        self.phase += self.rate_hz / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        1.0 + lfo * self.depth * 0.1
    }

    /// Reset phase to zero.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

/// Pitch bend via linear interpolation in a circular buffer.
///
/// Reads from a delay line at a sub-sample position that drifts over time,
/// creating a pitch shift without an external clock.
#[derive(Debug)]
pub struct PitchBend {
    buffer: Vec<f32>,
    write_pos: usize,
    read_pos: f32,
    /// Pitch ratio (1.0 = no shift, 2.0 = one octave up, 0.5 = one octave down).
    pub ratio: f32,
}

impl PitchBend {
    /// Create a new [`PitchBend`] with the given buffer size and pitch ratio.
    #[must_use]
    pub fn new(buffer_size: usize, ratio: f32) -> Self {
        let size = buffer_size.max(4);
        Self {
            buffer: vec![0.0; size],
            write_pos: 0,
            read_pos: 0.0,
            ratio: ratio.max(0.0625),
        }
    }

    /// Process one sample through the pitch bender.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let len = self.buffer.len();
        self.buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % len;

        // Advance read position at `ratio` speed
        self.read_pos += self.ratio;
        if self.read_pos >= len as f32 {
            self.read_pos -= len as f32;
        }

        // Linear interpolation
        let idx = self.read_pos as usize % len;
        let frac = self.read_pos - idx as f32;
        let next = (idx + 1) % len;
        self.buffer[idx] * (1.0 - frac) + self.buffer[next] * frac
    }

    /// Reset buffer and positions.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.read_pos = 0.0;
    }
}

/// Formant-preserving pitch shift approximation via two overlapping windows.
///
/// This is a simplified single-band implementation; production use requires
/// a full PSOLA or phase-vocoder engine.
#[derive(Debug, Clone)]
pub struct FormantShifter {
    /// Pitch ratio.
    pub ratio: f32,
    window_size: usize,
    hop_size: usize,
    input_buf: Vec<f32>,
    output_buf: Vec<f32>,
    write_head: usize,
    read_head: f32,
}

impl FormantShifter {
    /// Create a new [`FormantShifter`].
    #[must_use]
    pub fn new(window_size: usize, ratio: f32) -> Self {
        let window_size = window_size.max(64);
        let hop_size = window_size / 4;
        Self {
            ratio: ratio.clamp(0.5, 4.0),
            window_size,
            hop_size,
            input_buf: vec![0.0; window_size * 2],
            output_buf: vec![0.0; window_size * 2],
            write_head: 0,
            read_head: 0.0,
        }
    }

    /// Process a single sample.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let ilen = self.input_buf.len();
        let olen = self.output_buf.len();

        self.input_buf[self.write_head % ilen] = input;
        self.write_head += 1;

        // Simple linear interpolated read at scaled position
        self.read_head += self.ratio;
        if self.read_head >= olen as f32 {
            self.read_head -= olen as f32;
        }

        let idx = self.read_head as usize % ilen;
        let frac = self.read_head - idx as f32;
        let next = (idx + 1) % ilen;

        // Apply Hann window weight
        let win_pos = (idx % self.window_size) as f32 / self.window_size as f32;
        let hann = 0.5 * (1.0 - (2.0 * PI * win_pos).cos());

        let out = self.input_buf[idx] * (1.0 - frac) + self.input_buf[next] * frac;
        out * hann
    }

    /// Get window size.
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get hop size.
    #[must_use]
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.input_buf.fill(0.0);
        self.output_buf.fill(0.0);
        self.write_head = 0;
        self.read_head = 0.0;
    }
}

/// Apply a Hann window to a slice of samples in-place.
pub fn apply_hann_window(buffer: &mut [f32]) {
    let n = buffer.len();
    if n == 0 {
        return;
    }
    for (i, sample) in buffer.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos());
        *sample *= w;
    }
}

/// Apply a Blackman window to a slice of samples in-place.
pub fn apply_blackman_window(buffer: &mut [f32]) {
    let n = buffer.len();
    if n == 0 {
        return;
    }
    let n1 = (n as f32) - 1.0;
    for (i, sample) in buffer.iter_mut().enumerate() {
        let w =
            0.42 - 0.5 * (2.0 * PI * i as f32 / n1).cos() + 0.08 * (4.0 * PI * i as f32 / n1).cos();
        *sample *= w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_flutter_rate_centred_at_one() {
        let mut flutter = TapeFlutter::new(2.0, 0.5, 48000.0);
        // Average over a full cycle should be close to 1.0
        let samples = 48000;
        let sum: f32 = (0..samples).map(|_| flutter.tick()).sum();
        let mean = sum / samples as f32;
        assert!((mean - 1.0).abs() < 0.01, "mean={mean}");
    }

    #[test]
    fn test_tape_flutter_reset() {
        let mut flutter = TapeFlutter::new(1.0, 0.5, 48000.0);
        for _ in 0..100 {
            flutter.tick();
        }
        flutter.reset();
        assert_eq!(flutter.phase, 0.0);
    }

    #[test]
    fn test_tape_flutter_depth_bounds() {
        let flutter = TapeFlutter::new(1.0, 1.5, 48000.0);
        // depth should be clamped to 1.0
        assert!(flutter.depth <= 1.0);
    }

    #[test]
    fn test_pitch_bend_unity_ratio_passthrough() {
        let mut bender = PitchBend::new(256, 1.0);
        // At ratio 1.0 the read head advances at the same rate as write,
        // so output is a delayed version of the input.
        let input = vec![0.1, 0.2, 0.3, 0.4];
        for s in &input {
            bender.process_sample(*s);
        }
        // Just check it doesn't panic and returns finite values
        let out = bender.process_sample(0.5);
        assert!(out.is_finite());
    }

    #[test]
    fn test_pitch_bend_reset_zeroes_buffer() {
        let mut bender = PitchBend::new(64, 1.5);
        for _ in 0..32 {
            bender.process_sample(0.9);
        }
        bender.reset();
        for v in &bender.buffer {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_pitch_bend_output_finite() {
        let mut bender = PitchBend::new(512, 2.0);
        for i in 0..1000 {
            let input = (i as f32 * 0.01).sin();
            let out = bender.process_sample(input);
            assert!(out.is_finite(), "Non-finite at step {i}");
        }
    }

    #[test]
    fn test_formant_shifter_output_finite() {
        let mut shifter = FormantShifter::new(256, 1.5);
        for i in 0..512 {
            let input = (i as f32 * 0.02).sin();
            let out = shifter.process_sample(input);
            assert!(out.is_finite(), "Non-finite at step {i}");
        }
    }

    #[test]
    fn test_formant_shifter_reset() {
        let mut shifter = FormantShifter::new(128, 1.2);
        for _ in 0..64 {
            shifter.process_sample(0.5);
        }
        shifter.reset();
        assert_eq!(shifter.write_head, 0);
        for v in &shifter.input_buf {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_formant_shifter_window_and_hop() {
        let shifter = FormantShifter::new(256, 1.0);
        assert_eq!(shifter.window_size(), 256);
        assert_eq!(shifter.hop_size(), 64);
    }

    #[test]
    fn test_hann_window_endpoints_near_zero() {
        let mut buf = vec![1.0; 64];
        apply_hann_window(&mut buf);
        // First and last samples should be near 0
        assert!(buf[0] < 0.01, "First sample: {}", buf[0]);
        assert!(buf[63] < 0.01, "Last sample: {}", buf[63]);
    }

    #[test]
    fn test_hann_window_peak_at_center() {
        let mut buf = vec![1.0; 64];
        apply_hann_window(&mut buf);
        let peak = buf[32];
        assert!(peak > 0.9, "Peak: {peak}");
    }

    #[test]
    fn test_blackman_window_endpoints_near_zero() {
        let mut buf = vec![1.0; 64];
        apply_blackman_window(&mut buf);
        assert!(buf[0].abs() < 0.01);
    }

    #[test]
    fn test_hann_window_empty_buffer() {
        let mut buf: Vec<f32> = vec![];
        apply_hann_window(&mut buf); // should not panic
    }

    #[test]
    fn test_blackman_window_empty_buffer() {
        let mut buf: Vec<f32> = vec![];
        apply_blackman_window(&mut buf); // should not panic
    }
}

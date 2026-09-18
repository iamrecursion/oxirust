//! A power-of-two ring buffer delay line for audio DSP.
//!
//! [`DelayLine`] is a fixed-capacity circular buffer used internally by
//! reverb, chorus, flanger, and comb-filter algorithms. Samples are pushed
//! at the write head and tapped at arbitrary fractional offsets using linear
//! interpolation.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::delay_line::DelayLine;
//!
//! let mut dl = DelayLine::new(1024);
//! for i in 0..16 {
//!     dl.push(i as f32 * 0.1);
//! }
//! // Tap 4 samples ago
//! let val = dl.tap(4);
//! assert!((val - 1.1).abs() < 1e-5);
//! ```

#![allow(dead_code)]

/// A circular ring-buffer delay line.
///
/// The capacity is rounded up to the next power of two so that wrap-around
/// can be performed with a bitwise AND.
pub struct DelayLine {
    buffer: Vec<f32>,
    mask: usize,
    write_pos: usize,
    sample_rate: f32,
}

impl DelayLine {
    /// Create a new delay line with at least `min_capacity` samples.
    ///
    /// Actual capacity is rounded up to the next power of two.
    #[must_use]
    pub fn new(min_capacity: usize) -> Self {
        let capacity = min_capacity.next_power_of_two().max(2);
        Self {
            buffer: vec![0.0; capacity],
            mask: capacity - 1,
            write_pos: 0,
            sample_rate: 48_000.0,
        }
    }

    /// Create a delay line sized to hold at least `max_delay_ms` milliseconds
    /// at `sample_rate`.
    #[must_use]
    pub fn with_max_delay_ms(max_delay_ms: f32, sample_rate: f32) -> Self {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let samples = (max_delay_ms * 0.001 * sample_rate).ceil() as usize;
        let mut dl = Self::new(samples + 1);
        dl.sample_rate = sample_rate;
        dl
    }

    /// Return the capacity of the delay line in samples.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Return the maximum delay expressible with this buffer, in milliseconds.
    #[must_use]
    pub fn max_delay_ms(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let cap = self.buffer.len() as f32;
        cap / self.sample_rate * 1000.0
    }

    /// Push a single sample into the delay line, advancing the write head.
    pub fn push(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) & self.mask;
    }

    /// Read a sample `delay_samples` into the past (integer tap).
    ///
    /// `delay_samples` is clamped to `[0, capacity - 1]`.
    #[must_use]
    pub fn tap(&self, delay_samples: usize) -> f32 {
        let d = delay_samples.min(self.buffer.len() - 1);
        let idx = self.write_pos.wrapping_sub(d).wrapping_sub(1) & self.mask;
        self.buffer[idx]
    }

    /// Read a sample at a fractional delay using linear interpolation.
    ///
    /// `delay_samples` can be any positive value up to `capacity - 1`.
    #[must_use]
    pub fn tap_frac(&self, delay_samples: f32) -> f32 {
        let capped = delay_samples.clamp(0.0, (self.buffer.len() - 1) as f32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let d_int = capped as usize;
        let d_frac = capped - d_int as f32;

        let s0 = self.tap(d_int);
        let s1 = self.tap(d_int + 1);
        s0 + d_frac * (s1 - s0)
    }

    /// Clear the delay line (fill with zeros) and reset the write head.
    pub fn clear(&mut self) {
        for s in self.buffer.iter_mut() {
            *s = 0.0;
        }
        self.write_pos = 0;
    }

    /// Fill the delay line with `value` (useful for initialising with DC).
    pub fn fill(&mut self, value: f32) {
        for s in self.buffer.iter_mut() {
            *s = value;
        }
    }

    /// Return a read-only view of the internal ring buffer.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.buffer
    }

    /// Convert a time in milliseconds to a fractional number of samples.
    #[must_use]
    pub fn ms_to_samples(&self, ms: f32) -> f32 {
        ms * 0.001 * self.sample_rate
    }
}

/// A stereo delay line (two independent channels sharing the same parameters).
pub struct StereoDelayLine {
    /// Left channel delay.
    pub left: DelayLine,
    /// Right channel delay.
    pub right: DelayLine,
}

impl StereoDelayLine {
    /// Create a stereo delay line sized for at least `min_capacity` samples.
    #[must_use]
    pub fn new(min_capacity: usize) -> Self {
        Self {
            left: DelayLine::new(min_capacity),
            right: DelayLine::new(min_capacity),
        }
    }

    /// Push a stereo sample pair.
    pub fn push(&mut self, left: f32, right: f32) {
        self.left.push(left);
        self.right.push(right);
    }

    /// Tap both channels at the same integer delay.
    #[must_use]
    pub fn tap(&self, delay_samples: usize) -> (f32, f32) {
        (self.left.tap(delay_samples), self.right.tap(delay_samples))
    }

    /// Tap both channels at the same fractional delay.
    #[must_use]
    pub fn tap_frac(&self, delay_samples: f32) -> (f32, f32) {
        (
            self.left.tap_frac(delay_samples),
            self.right.tap_frac(delay_samples),
        )
    }

    /// Clear both channels.
    pub fn clear(&mut self) {
        self.left.clear();
        self.right.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_line_capacity_power_of_two() {
        let dl = DelayLine::new(100);
        assert!(dl.capacity().is_power_of_two());
        assert!(dl.capacity() >= 100);
    }

    #[test]
    fn test_delay_line_tap_zero_on_empty() {
        let dl = DelayLine::new(64);
        assert_eq!(dl.tap(0), 0.0);
        assert_eq!(dl.tap(10), 0.0);
    }

    #[test]
    fn test_delay_line_push_and_tap() {
        let mut dl = DelayLine::new(64);
        dl.push(0.5);
        // Immediately after push, tap(0) retrieves the last written sample
        assert!((dl.tap(0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_delay_line_integer_tap_history() {
        let mut dl = DelayLine::new(64);
        // Push samples 0.0, 0.1, 0.2, …, 1.5
        for i in 0..=15 {
            dl.push(i as f32 * 0.1);
        }
        // The last pushed sample (0.15=1.5) is at tap(0)
        let v0 = dl.tap(0);
        assert!((v0 - 1.5).abs() < 1e-5);
        // One sample before (0.14=1.4) is at tap(1)
        let v1 = dl.tap(1);
        assert!((v1 - 1.4).abs() < 1e-5);
    }

    #[test]
    fn test_delay_line_fractional_tap_midpoint() {
        let mut dl = DelayLine::new(64);
        dl.push(0.0);
        dl.push(1.0);
        // Fractional tap 0.5 between sample 0 (0.0) and sample 1 (1.0)
        // Since tap(0)=1.0 and tap(1)=0.0, tap_frac(0.5) should be ~0.5
        let v = dl.tap_frac(0.5);
        assert!((v - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_delay_line_clear() {
        let mut dl = DelayLine::new(64);
        dl.push(0.9);
        dl.clear();
        assert_eq!(dl.tap(0), 0.0);
        assert_eq!(dl.write_pos, 0);
    }

    #[test]
    fn test_delay_line_fill() {
        let mut dl = DelayLine::new(16);
        dl.fill(0.5);
        for &v in dl.as_slice() {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_delay_line_max_delay_ms() {
        let dl = DelayLine::with_max_delay_ms(10.0, 48_000.0);
        assert!(dl.max_delay_ms() >= 10.0);
    }

    #[test]
    fn test_delay_line_ms_to_samples() {
        let dl = DelayLine::with_max_delay_ms(100.0, 48_000.0);
        let samples = dl.ms_to_samples(1.0);
        assert!((samples - 48.0).abs() < 1e-3);
    }

    #[test]
    fn test_delay_line_wrap_around() {
        // Use a tiny buffer to force wrap-around quickly
        let mut dl = DelayLine::new(4);
        for i in 0..8 {
            dl.push(i as f32);
        }
        // After 8 pushes into a buffer of 4, the last 4 samples are 4,5,6,7
        assert!((dl.tap(0) - 7.0).abs() < 1e-6);
        assert!((dl.tap(3) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_delay_line_tap_clamp() {
        let mut dl = DelayLine::new(4);
        dl.push(1.0);
        // Asking for a delay beyond capacity should not panic
        let _ = dl.tap(1000);
    }

    #[test]
    fn test_stereo_delay_line_push_tap() {
        let mut sdl = StereoDelayLine::new(64);
        sdl.push(0.3, -0.3);
        let (l, r) = sdl.tap(0);
        assert!((l - 0.3).abs() < 1e-6);
        assert!((r - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn test_stereo_delay_line_clear() {
        let mut sdl = StereoDelayLine::new(64);
        sdl.push(0.8, 0.6);
        sdl.clear();
        let (l, r) = sdl.tap(0);
        assert_eq!(l, 0.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_stereo_delay_line_frac_tap() {
        let mut sdl = StereoDelayLine::new(64);
        sdl.push(0.0, 0.0);
        sdl.push(1.0, -1.0);
        let (l, r) = sdl.tap_frac(0.5);
        assert!((l - 0.5).abs() < 1e-5);
        assert!((r - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_delay_line_capacity_minimum() {
        // Even if min_capacity is 0 or 1, we get at least 2
        let dl = DelayLine::new(0);
        assert!(dl.capacity() >= 2);
    }
}

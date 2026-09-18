//! Delay line implementations for audio effects.
//!
//! Provides efficient circular buffer-based delay lines with support for
//! fractional delays using interpolation.

#![allow(clippy::cast_precision_loss)]

/// Simple delay line using a circular buffer.
#[derive(Debug, Clone)]
pub struct DelayLine {
    /// Circular buffer.
    buffer: Vec<f32>,
    /// Write position.
    write_pos: usize,
    /// Buffer size.
    size: usize,
}

impl DelayLine {
    /// Create a new delay line.
    ///
    /// # Arguments
    ///
    /// * `max_delay_samples` - Maximum delay in samples
    #[must_use]
    pub fn new(max_delay_samples: usize) -> Self {
        let size = max_delay_samples.max(1);
        Self {
            buffer: vec![0.0; size],
            write_pos: 0,
            size,
        }
    }

    /// Write a sample and read delayed sample.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `delay_samples` - Delay in samples (integer)
    pub fn process(&mut self, input: f32, delay_samples: usize) -> f32 {
        let delay = delay_samples.min(self.size - 1);

        // Calculate read position
        let read_pos = if self.write_pos >= delay {
            self.write_pos - delay
        } else {
            self.size - (delay - self.write_pos)
        };

        // Read delayed sample
        let output = self.buffer[read_pos];

        // Write new sample
        self.buffer[self.write_pos] = input;

        // Advance write position
        self.write_pos = (self.write_pos + 1) % self.size;

        output
    }

    /// Write a sample without reading.
    pub fn write(&mut self, input: f32) {
        self.buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % self.size;
    }

    /// Read delayed sample without writing.
    #[must_use]
    pub fn read(&self, delay_samples: usize) -> f32 {
        let delay = delay_samples.min(self.size - 1);

        let read_pos = if self.write_pos >= delay {
            self.write_pos - delay
        } else {
            self.size - (delay - self.write_pos)
        };

        self.buffer[read_pos]
    }

    /// Read with fractional delay using linear interpolation.
    #[must_use]
    pub fn read_fractional(&self, delay_samples: f32) -> f32 {
        let delay = delay_samples.min((self.size - 1) as f32);

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let delay_int = delay as usize;
        let delay_frac = delay - delay_int as f32;

        let read_pos1 = if self.write_pos >= delay_int {
            self.write_pos - delay_int
        } else {
            self.size - (delay_int - self.write_pos)
        };

        let read_pos2 = if read_pos1 == 0 {
            self.size - 1
        } else {
            read_pos1 - 1
        };

        // Linear interpolation
        let sample1 = self.buffer[read_pos1];
        let sample2 = self.buffer[read_pos2];

        sample1 + delay_frac * (sample2 - sample1)
    }

    /// Clear the delay line.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Get maximum delay in samples.
    #[must_use]
    pub fn max_delay(&self) -> usize {
        self.size - 1
    }
}

/// Interpolation mode for fractional delay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// No interpolation (nearest sample).
    None,
    /// Linear interpolation.
    Linear,
    /// Cubic Hermite interpolation (higher quality).
    Cubic,
}

/// Fractional delay line with configurable interpolation.
#[derive(Debug, Clone)]
pub struct FractionalDelayLine {
    /// Circular buffer.
    buffer: Vec<f32>,
    /// Write position.
    write_pos: usize,
    /// Buffer size.
    size: usize,
    /// Interpolation mode.
    interpolation: InterpolationMode,
}

impl FractionalDelayLine {
    /// Create a new fractional delay line.
    #[must_use]
    pub fn new(max_delay_samples: usize, interpolation: InterpolationMode) -> Self {
        let size = max_delay_samples.max(4); // Need at least 4 samples for cubic
        Self {
            buffer: vec![0.0; size],
            write_pos: 0,
            size,
            interpolation,
        }
    }

    /// Process a sample with fractional delay.
    pub fn process(&mut self, input: f32, delay_samples: f32) -> f32 {
        let output = self.read(delay_samples);
        self.write(input);
        output
    }

    /// Write a sample.
    pub fn write(&mut self, input: f32) {
        self.buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % self.size;
    }

    /// Read with fractional delay.
    #[must_use]
    pub fn read(&self, delay_samples: f32) -> f32 {
        let delay = delay_samples.clamp(0.0, (self.size - 1) as f32);

        match self.interpolation {
            InterpolationMode::None => self.read_none(delay),
            InterpolationMode::Linear => self.read_linear(delay),
            InterpolationMode::Cubic => self.read_cubic(delay),
        }
    }

    fn read_none(&self, delay: f32) -> f32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let delay_int = delay.round() as usize;

        let read_pos = if self.write_pos >= delay_int {
            self.write_pos - delay_int
        } else {
            self.size - (delay_int - self.write_pos)
        };

        self.buffer[read_pos]
    }

    fn read_linear(&self, delay: f32) -> f32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let delay_int = delay as usize;
        let frac = delay - delay_int as f32;

        let pos1 = if self.write_pos >= delay_int {
            self.write_pos - delay_int
        } else {
            self.size - (delay_int - self.write_pos)
        };

        let pos2 = if pos1 == 0 { self.size - 1 } else { pos1 - 1 };

        let s1 = self.buffer[pos1];
        let s2 = self.buffer[pos2];

        s1 + frac * (s2 - s1)
    }

    fn read_cubic(&self, delay: f32) -> f32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let delay_int = delay as usize;
        let frac = delay - delay_int as f32;

        // Get 4 samples for cubic interpolation
        let pos1 = if self.write_pos >= delay_int {
            self.write_pos - delay_int
        } else {
            self.size - (delay_int - self.write_pos)
        };

        let pos0 = if pos1 == self.size - 1 { 0 } else { pos1 + 1 };
        let pos2 = if pos1 == 0 { self.size - 1 } else { pos1 - 1 };
        let pos3 = if pos2 == 0 { self.size - 1 } else { pos2 - 1 };

        let y0 = self.buffer[pos0];
        let y1 = self.buffer[pos1];
        let y2 = self.buffer[pos2];
        let y3 = self.buffer[pos3];

        // Hermite interpolation
        let c0 = y1;
        let c1 = 0.5 * (y2 - y0);
        let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
        let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);

        ((c3 * frac + c2) * frac + c1) * frac + c0
    }

    /// Clear the delay line.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Set interpolation mode.
    pub fn set_interpolation(&mut self, interpolation: InterpolationMode) {
        self.interpolation = interpolation;
    }
}

/// All-pass filter for phase shifting.
///
/// Used in phaser effects and reverbs.
#[derive(Debug, Clone)]
pub struct AllPassFilter {
    /// Delay line.
    buffer: f32,
    /// Coefficient.
    coefficient: f32,
}

impl AllPassFilter {
    /// Create a new all-pass filter.
    #[must_use]
    pub fn new(coefficient: f32) -> Self {
        Self {
            buffer: 0.0,
            coefficient: coefficient.clamp(-0.999, 0.999),
        }
    }

    /// Process a sample.
    pub fn process(&mut self, input: f32) -> f32 {
        let output = -input + self.buffer;
        self.buffer = input + self.coefficient * output;
        output
    }

    /// Set coefficient.
    pub fn set_coefficient(&mut self, coefficient: f32) {
        self.coefficient = coefficient.clamp(-0.999, 0.999);
    }

    /// Reset the filter.
    pub fn reset(&mut self) {
        self.buffer = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_line_basic() {
        let mut delay = DelayLine::new(10);

        // Write some samples
        delay.write(1.0);
        delay.write(2.0);
        delay.write(3.0);

        // Read with 2 sample delay (should get sample from 2 positions back)
        let output = delay.read(2);
        assert_eq!(output, 2.0); // After writing 3 samples, read_pos 2 back gives us sample[1] = 2.0
    }

    #[test]
    fn test_delay_line_process() {
        let mut delay = DelayLine::new(5);

        // Fill delay line
        for _ in 0..5 {
            delay.process(0.0, 4);
        }

        // Process with delay
        let out1 = delay.process(1.0, 4);
        assert_eq!(out1, 0.0);

        for _ in 0..3 {
            delay.process(0.0, 4);
        }

        let out2 = delay.process(0.0, 4);
        assert_eq!(out2, 1.0);
    }

    #[test]
    fn test_fractional_delay_linear() {
        let mut delay = FractionalDelayLine::new(10, InterpolationMode::Linear);

        // Write samples
        delay.write(1.0);
        delay.write(0.0);

        // Read at fractional delay (should interpolate)
        let output = delay.read(1.5);
        assert!((output - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fractional_delay_modes() {
        let mut delay_none = FractionalDelayLine::new(10, InterpolationMode::None);
        let mut delay_linear = FractionalDelayLine::new(10, InterpolationMode::Linear);
        let mut delay_cubic = FractionalDelayLine::new(10, InterpolationMode::Cubic);

        // Write samples
        for i in 0..10 {
            #[allow(clippy::cast_precision_loss)]
            let sample = i as f32 / 10.0;
            delay_none.write(sample);
            delay_linear.write(sample);
            delay_cubic.write(sample);
        }

        // Read at fractional delay
        let out_none = delay_none.read(2.7);
        let out_linear = delay_linear.read(2.7);
        let out_cubic = delay_cubic.read(2.7);

        // None should round to nearest
        assert!((out_none - 0.7).abs() < 0.01);

        // Linear and cubic should be different
        assert!((out_linear - out_none).abs() > 0.01);
        assert!((out_cubic - out_none).abs() > 0.01);
    }

    #[test]
    fn test_allpass_filter() {
        let mut ap = AllPassFilter::new(0.5);

        // All-pass should have unity gain but shift phase
        let output1 = ap.process(1.0);
        let output2 = ap.process(0.0);

        // Output should be non-zero
        assert!(output1 != 0.0);
        assert!(output2 != 0.0);
    }

    #[test]
    fn test_delay_clear() {
        let mut delay = DelayLine::new(10);

        delay.write(1.0);
        delay.write(2.0);
        delay.write(3.0);

        delay.clear();

        let output = delay.read(1);
        assert_eq!(output, 0.0);
    }
}

//! Rate buffer management for HRD compliance.
//!
//! This module implements buffer models used for bitrate control:
//! - Leaky bucket model for CBR/VBR
//! - CPB (Coded Picture Buffer) management
//! - HRD (Hypothetical Reference Decoder) compliance

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![forbid(unsafe_code)]

/// Rate buffer using leaky bucket model.
///
/// The buffer simulates an encoder's output buffer:
/// - Bits are added when frames are encoded
/// - Bits are removed at a constant rate (target bitrate)
#[derive(Clone, Debug)]
pub struct RateBuffer {
    /// Buffer capacity in bits.
    capacity: u64,
    /// Current buffer level in bits.
    level: u64,
    /// Initial buffer fullness ratio.
    initial_fullness: f32,
}

impl RateBuffer {
    /// Create a new rate buffer.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Buffer size in bits
    /// * `initial_fullness` - Initial fill level as fraction (0.0-1.0)
    #[must_use]
    pub fn new(capacity: u64, initial_fullness: f32) -> Self {
        let fullness = initial_fullness.clamp(0.0, 1.0);
        let level = (capacity as f64 * fullness as f64) as u64;
        Self {
            capacity,
            level,
            initial_fullness: fullness,
        }
    }

    /// Add bits to the buffer (frame encoded).
    ///
    /// Returns the number of bits that overflowed (couldn't fit).
    pub fn add_bits(&mut self, bits: u64) -> u64 {
        let new_level = self.level.saturating_add(bits);
        if new_level > self.capacity {
            let overflow = new_level - self.capacity;
            self.level = self.capacity;
            overflow
        } else {
            self.level = new_level;
            0
        }
    }

    /// Remove bits from the buffer (drain at target bitrate).
    ///
    /// Returns the number of bits that underflowed (buffer went negative).
    pub fn remove_bits(&mut self, bits: u64) -> u64 {
        if bits > self.level {
            let underflow = bits - self.level;
            self.level = 0;
            underflow
        } else {
            self.level -= bits;
            0
        }
    }

    /// Get current buffer level in bits.
    #[must_use]
    pub fn level(&self) -> u64 {
        self.level
    }

    /// Get buffer capacity in bits.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Get buffer fullness as fraction (0.0-1.0).
    #[must_use]
    pub fn fullness(&self) -> f32 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.level as f32 / self.capacity as f32
    }

    /// Get available space in bits.
    #[must_use]
    pub fn available_space(&self) -> u64 {
        self.capacity.saturating_sub(self.level)
    }

    /// Check if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.level == 0
    }

    /// Check if buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.level >= self.capacity
    }

    /// Reset buffer to initial state.
    pub fn reset(&mut self) {
        self.level = (self.capacity as f64 * self.initial_fullness as f64) as u64;
    }

    /// Set buffer level directly (use with caution).
    pub fn set_level(&mut self, level: u64) {
        self.level = level.min(self.capacity);
    }
}

impl Default for RateBuffer {
    fn default() -> Self {
        Self::new(5_000_000, 0.5)
    }
}

/// Buffer model for HRD (Hypothetical Reference Decoder) compliance.
///
/// Implements the CPB (Coded Picture Buffer) model as defined in
/// video coding standards for streaming compliance.
#[derive(Clone, Debug)]
pub struct BufferModel {
    /// CPB size in bits.
    cpb_size: u64,
    /// Current CPB occupancy in bits.
    cpb_level: u64,
    /// Target bitrate in bits per second.
    bitrate: u64,
    /// Frame rate (frames per second).
    framerate: f64,
    /// Initial CPB removal delay in seconds.
    initial_delay: f64,
    /// Time since last removal (for drift tracking).
    time_since_removal: f64,
    /// Overflow events count.
    overflow_count: u64,
    /// Underflow events count.
    underflow_count: u64,
}

impl BufferModel {
    /// Create a new buffer model.
    #[must_use]
    pub fn new(cpb_size: u64, bitrate: u64, framerate: f64, initial_delay: f64) -> Self {
        // Initial CPB level based on delay
        let initial_level = (bitrate as f64 * initial_delay) as u64;

        Self {
            cpb_size,
            cpb_level: initial_level.min(cpb_size),
            bitrate,
            framerate,
            initial_delay,
            time_since_removal: 0.0,
            overflow_count: 0,
            underflow_count: 0,
        }
    }

    /// Fill buffer at constant bitrate for one frame duration.
    pub fn fill_for_frame(&mut self) {
        let frame_duration = 1.0 / self.framerate;
        let bits_to_add = (self.bitrate as f64 * frame_duration) as u64;

        let new_level = self.cpb_level.saturating_add(bits_to_add);
        if new_level > self.cpb_size {
            self.overflow_count += 1;
            self.cpb_level = self.cpb_size;
        } else {
            self.cpb_level = new_level;
        }

        self.time_since_removal += frame_duration;
    }

    /// Remove bits for an encoded frame.
    pub fn remove_frame_bits(&mut self, bits: u64) {
        if bits > self.cpb_level {
            self.underflow_count += 1;
            self.cpb_level = 0;
        } else {
            self.cpb_level -= bits;
        }
        self.time_since_removal = 0.0;
    }

    /// Simulate encoding a frame with the given bits.
    ///
    /// Fills the buffer, then removes the frame bits.
    pub fn encode_frame(&mut self, bits: u64) {
        self.fill_for_frame();
        self.remove_frame_bits(bits);
    }

    /// Get maximum bits allowed for next frame (before overflow).
    #[must_use]
    pub fn max_frame_bits(&self) -> u64 {
        // After filling for one frame, how many bits can we remove?
        let frame_duration = 1.0 / self.framerate;
        let bits_added = (self.bitrate as f64 * frame_duration) as u64;

        // We can remove all bits in the buffer
        self.cpb_level.saturating_add(bits_added).min(self.cpb_size)
    }

    /// Get minimum bits required for next frame (before underflow).
    #[must_use]
    pub fn min_frame_bits(&self) -> u64 {
        // After filling for one frame, what's the minimum we must remove?
        let frame_duration = 1.0 / self.framerate;
        let bits_added = (self.bitrate as f64 * frame_duration) as u64;
        let future_level = self.cpb_level.saturating_add(bits_added);

        // If would overflow, we must remove at least this much
        future_level.saturating_sub(self.cpb_size)
    }

    /// Check if the buffer is in a healthy state.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        let fullness = self.fullness();
        (0.2..=0.8).contains(&fullness)
    }

    /// Get buffer fullness as fraction.
    #[must_use]
    pub fn fullness(&self) -> f32 {
        if self.cpb_size == 0 {
            return 0.0;
        }
        self.cpb_level as f32 / self.cpb_size as f32
    }

    /// Get CPB level in bits.
    #[must_use]
    pub fn level(&self) -> u64 {
        self.cpb_level
    }

    /// Get CPB size in bits.
    #[must_use]
    pub fn cpb_size(&self) -> u64 {
        self.cpb_size
    }

    /// Get overflow count.
    #[must_use]
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    /// Get underflow count.
    #[must_use]
    pub fn underflow_count(&self) -> u64 {
        self.underflow_count
    }

    /// Reset the buffer model.
    pub fn reset(&mut self) {
        let initial_level = (self.bitrate as f64 * self.initial_delay) as u64;
        self.cpb_level = initial_level.min(self.cpb_size);
        self.time_since_removal = 0.0;
        self.overflow_count = 0;
        self.underflow_count = 0;
    }
}

impl Default for BufferModel {
    fn default() -> Self {
        Self::new(5_000_000, 5_000_000, 30.0, 0.5)
    }
}

/// VBV (Video Buffering Verifier) parameters.
#[derive(Clone, Debug)]
pub struct VbvParams {
    /// Maximum buffer size in bits.
    pub max_buffer_size: u64,
    /// Maximum bitrate in bits per second.
    pub max_bitrate: u64,
    /// Initial buffer delay in seconds.
    pub initial_delay: f64,
    /// Whether VBV is enabled.
    pub enabled: bool,
}

impl VbvParams {
    /// Create VBV parameters from bitrate.
    #[must_use]
    pub fn from_bitrate(bitrate: u64) -> Self {
        Self {
            max_buffer_size: bitrate * 2, // 2 seconds of buffer
            max_bitrate: bitrate,
            initial_delay: 0.5,
            enabled: true,
        }
    }

    /// Create VBV parameters for streaming.
    #[must_use]
    pub fn for_streaming(bitrate: u64, buffer_seconds: f64) -> Self {
        Self {
            max_buffer_size: (bitrate as f64 * buffer_seconds) as u64,
            max_bitrate: bitrate,
            initial_delay: buffer_seconds / 2.0,
            enabled: true,
        }
    }

    /// Disable VBV.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            max_buffer_size: u64::MAX,
            max_bitrate: u64::MAX,
            initial_delay: 0.0,
            enabled: false,
        }
    }
}

impl Default for VbvParams {
    fn default() -> Self {
        Self::from_bitrate(5_000_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_buffer_creation() {
        let buffer = RateBuffer::new(1_000_000, 0.5);
        assert_eq!(buffer.capacity(), 1_000_000);
        assert_eq!(buffer.level(), 500_000);
        assert!((buffer.fullness() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rate_buffer_add_bits() {
        let mut buffer = RateBuffer::new(1_000_000, 0.0);
        assert_eq!(buffer.level(), 0);

        buffer.add_bits(500_000);
        assert_eq!(buffer.level(), 500_000);

        // Add more bits, causing overflow
        let overflow = buffer.add_bits(600_000);
        assert_eq!(overflow, 100_000);
        assert_eq!(buffer.level(), 1_000_000);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_rate_buffer_remove_bits() {
        let mut buffer = RateBuffer::new(1_000_000, 0.5);

        buffer.remove_bits(200_000);
        assert_eq!(buffer.level(), 300_000);

        // Remove more than available, causing underflow
        let underflow = buffer.remove_bits(400_000);
        assert_eq!(underflow, 100_000);
        assert_eq!(buffer.level(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_rate_buffer_available_space() {
        let buffer = RateBuffer::new(1_000_000, 0.5);
        assert_eq!(buffer.available_space(), 500_000);
    }

    #[test]
    fn test_rate_buffer_reset() {
        let mut buffer = RateBuffer::new(1_000_000, 0.5);
        buffer.add_bits(500_000);
        buffer.reset();
        assert_eq!(buffer.level(), 500_000);
    }

    #[test]
    fn test_buffer_model_creation() {
        let model = BufferModel::new(5_000_000, 5_000_000, 30.0, 0.5);
        assert_eq!(model.cpb_size(), 5_000_000);
        // Initial level should be 0.5 seconds worth = 2,500,000 bits
        assert_eq!(model.level(), 2_500_000);
    }

    #[test]
    fn test_buffer_model_encode_frame() {
        let mut model = BufferModel::new(5_000_000, 5_000_000, 30.0, 0.5);
        let initial_level = model.level();

        // Encode a frame with target bits (bitrate / fps)
        let target_bits = 5_000_000 / 30; // ~166,666 bits
        model.encode_frame(target_bits);

        // Level should be approximately the same (filled then removed)
        let diff = (model.level() as i64 - initial_level as i64).abs();
        assert!(diff < 100); // Allow small rounding errors
    }

    #[test]
    fn test_buffer_model_max_min_bits() {
        let model = BufferModel::new(5_000_000, 5_000_000, 30.0, 0.5);

        let max_bits = model.max_frame_bits();
        let min_bits = model.min_frame_bits();

        assert!(max_bits > 0);
        assert!(max_bits >= min_bits);
    }

    #[test]
    fn test_buffer_model_overflow() {
        let mut model = BufferModel::new(1_000_000, 5_000_000, 30.0, 0.9);

        // Keep filling without removing enough bits
        for _ in 0..60 {
            model.fill_for_frame();
            model.remove_frame_bits(1000); // Remove very little
        }

        assert!(model.overflow_count() > 0);
    }

    #[test]
    fn test_buffer_model_underflow() {
        let mut model = BufferModel::new(1_000_000, 5_000_000, 30.0, 0.1);

        // Remove lots of bits
        model.remove_frame_bits(500_000);

        assert!(model.underflow_count() > 0 || model.level() == 0);
    }

    #[test]
    fn test_vbv_params() {
        let vbv = VbvParams::from_bitrate(5_000_000);
        assert!(vbv.enabled);
        assert_eq!(vbv.max_bitrate, 5_000_000);
        assert_eq!(vbv.max_buffer_size, 10_000_000);

        let vbv = VbvParams::for_streaming(5_000_000, 2.0);
        assert_eq!(vbv.max_buffer_size, 10_000_000);
        assert!((vbv.initial_delay - 1.0).abs() < f64::EPSILON);

        let vbv = VbvParams::disabled();
        assert!(!vbv.enabled);
    }
}

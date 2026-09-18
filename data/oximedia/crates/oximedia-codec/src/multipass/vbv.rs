//! VBV (Video Buffering Verifier) buffer model implementation.
//!
//! The VBV model ensures that encoded video can be decoded without buffer
//! underflow or overflow, which is critical for streaming applications.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use crate::frame::FrameType;

/// VBV buffer configuration.
#[derive(Clone, Debug)]
pub struct VbvConfig {
    /// Buffer size in bits.
    pub buffer_size: u64,
    /// Maximum bitrate in bits per second.
    pub max_bitrate: u64,
    /// Initial buffer fullness (0.0-1.0).
    pub initial_fullness: f64,
    /// Frame rate numerator.
    pub framerate_num: u32,
    /// Frame rate denominator.
    pub framerate_den: u32,
}

impl VbvConfig {
    /// Create a new VBV configuration.
    #[must_use]
    pub fn new(buffer_size: u64, max_bitrate: u64, framerate_num: u32, framerate_den: u32) -> Self {
        Self {
            buffer_size,
            max_bitrate,
            initial_fullness: 0.75,
            framerate_num,
            framerate_den,
        }
    }

    /// Set initial buffer fullness.
    #[must_use]
    pub fn with_initial_fullness(mut self, fullness: f64) -> Self {
        self.initial_fullness = fullness.clamp(0.0, 1.0);
        self
    }

    /// Calculate frame period in seconds.
    #[must_use]
    pub fn frame_period(&self) -> f64 {
        self.framerate_den as f64 / self.framerate_num as f64
    }

    /// Calculate bits added per frame period.
    #[must_use]
    pub fn bits_per_frame(&self) -> f64 {
        self.max_bitrate as f64 * self.frame_period()
    }
}

/// VBV buffer state and management.
pub struct VbvBuffer {
    config: VbvConfig,
    /// Current buffer fullness in bits.
    current_fullness: f64,
    /// Minimum observed fullness (for debugging).
    min_fullness: f64,
    /// Maximum observed fullness (for debugging).
    max_fullness: f64,
    /// Number of frames processed.
    frame_count: u64,
    /// Total underflows detected.
    underflow_count: u64,
    /// Total overflows detected.
    overflow_count: u64,
}

impl VbvBuffer {
    /// Create a new VBV buffer.
    #[must_use]
    pub fn new(config: VbvConfig) -> Self {
        let current_fullness = config.buffer_size as f64 * config.initial_fullness;

        Self {
            config,
            current_fullness,
            min_fullness: current_fullness,
            max_fullness: current_fullness,
            frame_count: 0,
            underflow_count: 0,
            overflow_count: 0,
        }
    }

    /// Check if we can send a frame of given size without overflow.
    #[must_use]
    pub fn can_send_frame(&self, frame_bits: u64) -> bool {
        let new_fullness = self.current_fullness - frame_bits as f64;
        new_fullness >= 0.0
    }

    /// Get maximum frame size that can be sent without underflow.
    #[must_use]
    pub fn max_frame_size(&self) -> u64 {
        self.current_fullness as u64
    }

    /// Get minimum frame size to avoid overflow after refill.
    #[must_use]
    pub fn min_frame_size(&self) -> u64 {
        let bits_added = self.config.bits_per_frame();
        let max_fullness = self.config.buffer_size as f64;
        let available_after_refill = self.current_fullness + bits_added;

        if available_after_refill > max_fullness {
            (available_after_refill - max_fullness) as u64
        } else {
            0
        }
    }

    /// Update buffer state after encoding a frame.
    pub fn update(&mut self, frame_bits: u64) -> VbvUpdateResult {
        // Remove bits for encoded frame
        self.current_fullness -= frame_bits as f64;

        let mut result = VbvUpdateResult {
            underflow: false,
            overflow: false,
            fullness_ratio: 0.0,
            available_bits: 0,
        };

        // Check for underflow
        if self.current_fullness < 0.0 {
            result.underflow = true;
            self.underflow_count += 1;
            self.current_fullness = 0.0;
        }

        // Add bits for next frame period
        let bits_added = self.config.bits_per_frame();
        self.current_fullness += bits_added;

        // Check for overflow
        if self.current_fullness > self.config.buffer_size as f64 {
            result.overflow = true;
            self.overflow_count += 1;
            self.current_fullness = self.config.buffer_size as f64;
        }

        // Update min/max tracking
        self.min_fullness = self.min_fullness.min(self.current_fullness);
        self.max_fullness = self.max_fullness.max(self.current_fullness);

        result.fullness_ratio = self.current_fullness / self.config.buffer_size as f64;
        result.available_bits = self.current_fullness as u64;

        self.frame_count += 1;
        result
    }

    /// Calculate target frame size based on buffer state.
    #[must_use]
    pub fn target_frame_size(&self, frame_type: FrameType, base_size: f64) -> u64 {
        let fullness_ratio = self.current_fullness / self.config.buffer_size as f64;

        // Adjust based on buffer fullness
        // If buffer is full, we can use more bits
        // If buffer is empty, we need to use fewer bits
        let buffer_factor = if fullness_ratio > 0.75 {
            1.2 // Buffer is full, can be generous
        } else if fullness_ratio > 0.5 {
            1.0 // Normal operation
        } else if fullness_ratio > 0.25 {
            0.8 // Buffer getting low, be conservative
        } else {
            0.6 // Buffer critically low, minimize size
        };

        // Adjust for frame type
        let type_factor = match frame_type {
            FrameType::Key => 3.0,    // Keyframes need more bits
            FrameType::Inter => 1.0,  // Inter frames are baseline
            FrameType::BiDir => 0.5,  // B-frames use fewer bits
            FrameType::Switch => 2.0, // Switch frames need extra bits
        };

        let target = base_size * buffer_factor * type_factor;

        // Ensure we don't exceed buffer capacity
        target.min(self.max_frame_size() as f64) as u64
    }

    /// Get current buffer fullness ratio (0.0-1.0).
    #[must_use]
    pub fn fullness_ratio(&self) -> f64 {
        self.current_fullness / self.config.buffer_size as f64
    }

    /// Get current buffer fullness in bits.
    #[must_use]
    pub fn current_fullness(&self) -> u64 {
        self.current_fullness as u64
    }

    /// Get VBV statistics.
    #[must_use]
    pub fn statistics(&self) -> VbvStatistics {
        VbvStatistics {
            frame_count: self.frame_count,
            underflow_count: self.underflow_count,
            overflow_count: self.overflow_count,
            current_fullness: self.current_fullness,
            min_fullness: self.min_fullness,
            max_fullness: self.max_fullness,
            buffer_size: self.config.buffer_size,
        }
    }

    /// Reset buffer state.
    pub fn reset(&mut self) {
        self.current_fullness = self.config.buffer_size as f64 * self.config.initial_fullness;
        self.min_fullness = self.current_fullness;
        self.max_fullness = self.current_fullness;
        self.frame_count = 0;
        self.underflow_count = 0;
        self.overflow_count = 0;
    }
}

/// Result of a VBV buffer update.
#[derive(Clone, Debug)]
pub struct VbvUpdateResult {
    /// Buffer underflow occurred.
    pub underflow: bool,
    /// Buffer overflow occurred.
    pub overflow: bool,
    /// Current buffer fullness ratio (0.0-1.0).
    pub fullness_ratio: f64,
    /// Available bits in buffer.
    pub available_bits: u64,
}

/// VBV buffer statistics.
#[derive(Clone, Debug)]
pub struct VbvStatistics {
    /// Total frames processed.
    pub frame_count: u64,
    /// Number of underflows.
    pub underflow_count: u64,
    /// Number of overflows.
    pub overflow_count: u64,
    /// Current buffer fullness in bits.
    pub current_fullness: f64,
    /// Minimum observed fullness.
    pub min_fullness: f64,
    /// Maximum observed fullness.
    pub max_fullness: f64,
    /// Total buffer size.
    pub buffer_size: u64,
}

impl VbvStatistics {
    /// Check if VBV compliance is good (no underflows/overflows).
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.underflow_count == 0 && self.overflow_count == 0
    }

    /// Get buffer utilization statistics.
    #[must_use]
    pub fn utilization(&self) -> BufferUtilization {
        BufferUtilization {
            min_ratio: self.min_fullness / self.buffer_size as f64,
            max_ratio: self.max_fullness / self.buffer_size as f64,
            current_ratio: self.current_fullness / self.buffer_size as f64,
        }
    }
}

/// Buffer utilization metrics.
#[derive(Clone, Debug)]
pub struct BufferUtilization {
    /// Minimum fullness ratio observed.
    pub min_ratio: f64,
    /// Maximum fullness ratio observed.
    pub max_ratio: f64,
    /// Current fullness ratio.
    pub current_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vbv_config_new() {
        let config = VbvConfig::new(1_000_000, 5_000_000, 30, 1);
        assert_eq!(config.buffer_size, 1_000_000);
        assert_eq!(config.max_bitrate, 5_000_000);
        assert_eq!(config.initial_fullness, 0.75);
    }

    #[test]
    fn test_vbv_config_frame_period() {
        let config = VbvConfig::new(1_000_000, 5_000_000, 30, 1);
        let period = config.frame_period();
        assert!((period - 1.0 / 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_vbv_buffer_new() {
        let config = VbvConfig::new(1_000_000, 5_000_000, 30, 1);
        let buffer = VbvBuffer::new(config);
        assert_eq!(buffer.current_fullness, 750_000.0);
    }

    #[test]
    fn test_vbv_buffer_can_send() {
        let config = VbvConfig::new(1_000_000, 5_000_000, 30, 1);
        let buffer = VbvBuffer::new(config);

        assert!(buffer.can_send_frame(100_000));
        assert!(buffer.can_send_frame(750_000));
        assert!(!buffer.can_send_frame(800_000));
    }

    #[test]
    fn test_vbv_buffer_update() {
        let config = VbvConfig::new(1_000_000, 5_000_000, 30, 1);
        let mut buffer = VbvBuffer::new(config);

        let result = buffer.update(100_000);
        assert!(!result.underflow);
        assert!(!result.overflow);
        assert!(result.fullness_ratio > 0.0);
    }

    #[test]
    fn test_vbv_buffer_underflow() {
        let config = VbvConfig::new(1_000_000, 5_000_000, 30, 1);
        let mut buffer = VbvBuffer::new(config);

        // Try to send more than available
        let result = buffer.update(1_000_000);
        assert!(result.underflow);
        assert_eq!(buffer.underflow_count, 1);
    }

    #[test]
    fn test_vbv_target_frame_size() {
        let config = VbvConfig::new(1_000_000, 5_000_000, 30, 1);
        let buffer = VbvBuffer::new(config);

        let target = buffer.target_frame_size(FrameType::Inter, 50_000.0);
        assert!(target > 0);

        let key_target = buffer.target_frame_size(FrameType::Key, 50_000.0);
        assert!(key_target > target); // Keyframes should be larger
    }

    #[test]
    fn test_vbv_statistics() {
        let config = VbvConfig::new(1_000_000, 5_000_000, 30, 1);
        let mut buffer = VbvBuffer::new(config);

        buffer.update(100_000);
        buffer.update(100_000);

        let stats = buffer.statistics();
        assert_eq!(stats.frame_count, 2);
        assert!(stats.is_compliant());
    }

    #[test]
    fn test_vbv_reset() {
        let config = VbvConfig::new(1_000_000, 5_000_000, 30, 1);
        let mut buffer = VbvBuffer::new(config);

        buffer.update(100_000);
        assert_eq!(buffer.frame_count, 1);

        buffer.reset();
        assert_eq!(buffer.frame_count, 0);
        assert_eq!(buffer.current_fullness, 750_000.0);
    }
}

//! Constant QP rate control.
//!
//! CQP (Constant Quantization Parameter) is the simplest rate control mode.
//! It uses a fixed QP for each frame type, with optional offsets for
//! P-frames and B-frames relative to I-frames.
//!
//! # Usage
//!
//! ```ignore
//! use oximedia_codec::rate_control::{CqpController, RcConfig};
//!
//! let config = RcConfig::cqp(28);
//! let mut controller = CqpController::new(&config);
//!
//! // Get QP for I-frame
//! let output = controller.get_qp(FrameType::Key);
//! ```

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![forbid(unsafe_code)]

use crate::frame::FrameType;

use super::types::{FrameStats, RcConfig, RcOutput};

/// Constant QP rate controller.
///
/// Provides fixed QP values for each frame type with configurable offsets.
#[derive(Clone, Debug)]
pub struct CqpController {
    /// Base QP value (for I-frames).
    base_qp: u8,
    /// QP offset for P-frames.
    p_offset: i8,
    /// QP offset for B-frames.
    b_offset: i8,
    /// Minimum allowed QP.
    min_qp: u8,
    /// Maximum allowed QP.
    max_qp: u8,
    /// Frame counter.
    frame_count: u64,
    /// Total bits encoded.
    total_bits: u64,
}

impl CqpController {
    /// Create a new CQP controller from configuration.
    #[must_use]
    pub fn new(config: &RcConfig) -> Self {
        Self {
            base_qp: config.initial_qp.clamp(config.min_qp, config.max_qp),
            p_offset: 0,
            b_offset: config.b_qp_offset,
            min_qp: config.min_qp,
            max_qp: config.max_qp,
            frame_count: 0,
            total_bits: 0,
        }
    }

    /// Create a CQP controller with explicit QP and offsets.
    #[must_use]
    pub fn with_offsets(base_qp: u8, p_offset: i8, b_offset: i8) -> Self {
        Self {
            base_qp: base_qp.clamp(1, 63),
            p_offset,
            b_offset,
            min_qp: 1,
            max_qp: 63,
            frame_count: 0,
            total_bits: 0,
        }
    }

    /// Set the base QP value.
    pub fn set_base_qp(&mut self, qp: u8) {
        self.base_qp = qp.clamp(self.min_qp, self.max_qp);
    }

    /// Set frame type offsets.
    pub fn set_offsets(&mut self, p_offset: i8, b_offset: i8) {
        self.p_offset = p_offset;
        self.b_offset = b_offset;
    }

    /// Get the QP for a given frame type.
    #[must_use]
    pub fn get_qp(&self, frame_type: FrameType) -> RcOutput {
        let offset = match frame_type {
            FrameType::Key => 0,
            FrameType::Inter | FrameType::Switch => self.p_offset,
            FrameType::BiDir => self.b_offset,
        };

        let qp = self.apply_offset(self.base_qp, offset);
        let qp_f = qp as f32;

        let mut output = RcOutput {
            qp,
            qp_f,
            ..Default::default()
        };
        output.compute_lambda();
        output
    }

    /// Apply an offset to a QP value, clamping to valid range.
    fn apply_offset(&self, qp: u8, offset: i8) -> u8 {
        let result = i16::from(qp) + i16::from(offset);
        result.clamp(i16::from(self.min_qp), i16::from(self.max_qp)) as u8
    }

    /// Update controller with frame statistics.
    pub fn update(&mut self, stats: &FrameStats) {
        self.frame_count += 1;
        self.total_bits += stats.bits;
    }

    /// Get the current base QP.
    #[must_use]
    pub fn base_qp(&self) -> u8 {
        self.base_qp
    }

    /// Get the QP for I-frames.
    #[must_use]
    pub fn i_qp(&self) -> u8 {
        self.base_qp
    }

    /// Get the QP for P-frames.
    #[must_use]
    pub fn p_qp(&self) -> u8 {
        self.apply_offset(self.base_qp, self.p_offset)
    }

    /// Get the QP for B-frames.
    #[must_use]
    pub fn b_qp(&self) -> u8 {
        self.apply_offset(self.base_qp, self.b_offset)
    }

    /// Get total frames processed.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get total bits produced.
    #[must_use]
    pub fn total_bits(&self) -> u64 {
        self.total_bits
    }

    /// Get average bits per frame.
    #[must_use]
    pub fn average_bits_per_frame(&self) -> f64 {
        if self.frame_count == 0 {
            0.0
        } else {
            self.total_bits as f64 / self.frame_count as f64
        }
    }

    /// Reset the controller state.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.total_bits = 0;
    }
}

impl Default for CqpController {
    fn default() -> Self {
        Self {
            base_qp: 28,
            p_offset: 0,
            b_offset: 2,
            min_qp: 1,
            max_qp: 63,
            frame_count: 0,
            total_bits: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cqp_creation() {
        let config = RcConfig::cqp(28);
        let controller = CqpController::new(&config);
        assert_eq!(controller.base_qp(), 28);
    }

    #[test]
    fn test_cqp_with_offsets() {
        let controller = CqpController::with_offsets(28, 2, 4);
        assert_eq!(controller.i_qp(), 28);
        assert_eq!(controller.p_qp(), 30);
        assert_eq!(controller.b_qp(), 32);
    }

    #[test]
    fn test_get_qp_by_frame_type() {
        let controller = CqpController::with_offsets(28, 2, 4);

        let output = controller.get_qp(FrameType::Key);
        assert_eq!(output.qp, 28);

        let output = controller.get_qp(FrameType::Inter);
        assert_eq!(output.qp, 30);

        let output = controller.get_qp(FrameType::BiDir);
        assert_eq!(output.qp, 32);
    }

    #[test]
    fn test_qp_clamping() {
        let mut controller = CqpController::with_offsets(5, -10, 10);
        controller.min_qp = 1;
        controller.max_qp = 10;

        // I-frame: 5, P-frame: 5-10=clamped to 1, B-frame: 5+10=clamped to 10
        assert_eq!(controller.i_qp(), 5);
        assert_eq!(controller.p_qp(), 1);
        assert_eq!(controller.b_qp(), 10);
    }

    #[test]
    fn test_statistics_tracking() {
        let mut controller = CqpController::default();

        let mut stats = FrameStats::new(0, FrameType::Key);
        stats.bits = 100_000;
        controller.update(&stats);

        let mut stats = FrameStats::new(1, FrameType::Inter);
        stats.bits = 50_000;
        controller.update(&stats);

        assert_eq!(controller.frame_count(), 2);
        assert_eq!(controller.total_bits(), 150_000);
        assert!((controller.average_bits_per_frame() - 75_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset() {
        let mut controller = CqpController::default();

        let mut stats = FrameStats::new(0, FrameType::Key);
        stats.bits = 100_000;
        controller.update(&stats);

        controller.reset();

        assert_eq!(controller.frame_count(), 0);
        assert_eq!(controller.total_bits(), 0);
    }

    #[test]
    fn test_lambda_calculation() {
        let controller = CqpController::default();
        let output = controller.get_qp(FrameType::Key);

        assert!(output.lambda > 0.0);
        assert!(output.lambda_me > 0.0);
        assert!((output.lambda_me - output.lambda.sqrt()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_set_base_qp() {
        let mut controller = CqpController::default();
        controller.set_base_qp(35);
        assert_eq!(controller.base_qp(), 35);

        // Test clamping
        controller.min_qp = 10;
        controller.max_qp = 40;
        controller.set_base_qp(5);
        assert_eq!(controller.base_qp(), 10);

        controller.set_base_qp(50);
        assert_eq!(controller.base_qp(), 40);
    }
}

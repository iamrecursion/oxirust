//! Advanced bitrate control for video encoding.
//!
//! Provides HRD buffer management, frame-type-aware bit allocation, and
//! a moving-average bitrate history for adaptive rate control.

#![allow(dead_code)]

use std::collections::VecDeque;

/// Bitrate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitrateControlMode {
    /// CBR – constant bitrate.
    ConstantBitrate,
    /// VBR – variable bitrate, within min/max bounds.
    VariableBitrate,
    /// CQP/CRF – target constant quality, bitrate floats freely.
    ConstantQuality,
    /// Constrained VBR – variable but within an HRD buffer constraint.
    ConstrainedVbr,
}

/// Bitrate model parameters.
#[derive(Debug, Clone)]
pub struct BitrateModel {
    /// Target bitrate in kbps.
    pub target_kbps: u32,
    /// Minimum bitrate in kbps.
    pub min_kbps: u32,
    /// Maximum bitrate in kbps (peak).
    pub max_kbps: u32,
    /// HRD buffer size in milliseconds.
    pub buffer_size_ms: u32,
}

impl BitrateModel {
    /// Create a new bitrate model.
    #[must_use]
    pub fn new(target_kbps: u32, min_kbps: u32, max_kbps: u32, buffer_size_ms: u32) -> Self {
        Self {
            target_kbps,
            min_kbps,
            max_kbps,
            buffer_size_ms,
        }
    }

    /// Average bits per frame at the target rate for a given fps.
    #[must_use]
    pub fn bits_per_frame(&self, fps: f32) -> u32 {
        if fps <= 0.0 {
            return 0;
        }
        ((self.target_kbps as f64 * 1000.0) / fps as f64) as u32
    }
}

/// HRD (Hypothetical Reference Decoder) buffer model.
///
/// Simulates the decoder buffer to detect under/overflow.
#[derive(Debug, Clone)]
pub struct HrdBuffer {
    /// Buffer size in bits.
    pub size_bits: u64,
    /// Current fullness in bits.
    pub fullness_bits: u64,
    /// Constant drain rate in bits-per-second.
    pub target_rate_bps: u64,
}

impl HrdBuffer {
    /// Create a new HRD buffer.
    #[must_use]
    pub fn new(size_bits: u64, target_rate_bps: u64) -> Self {
        Self {
            size_bits,
            fullness_bits: size_bits / 2, // start half-full
            target_rate_bps,
        }
    }

    /// Add bits (frame data enters decoder buffer).
    pub fn add_bits(&mut self, n: u64) {
        self.fullness_bits = self.fullness_bits.saturating_add(n).min(self.size_bits);
    }

    /// Drain bits corresponding to `ms` milliseconds of playback.
    pub fn drain_bits(&mut self, ms: u32) {
        let drained = (self.target_rate_bps * ms as u64) / 1000;
        self.fullness_bits = self.fullness_bits.saturating_sub(drained);
    }

    /// Fill level as a percentage [0.0, 100.0].
    #[must_use]
    pub fn fill_level_pct(&self) -> f32 {
        if self.size_bits == 0 {
            return 0.0;
        }
        (self.fullness_bits as f64 / self.size_bits as f64 * 100.0) as f32
    }

    /// Returns `true` if the buffer has overflowed.
    #[must_use]
    pub fn is_overflow(&self) -> bool {
        self.fullness_bits >= self.size_bits
    }

    /// Returns `true` if the buffer has underflowed (empty).
    #[must_use]
    pub fn is_underflow(&self) -> bool {
        self.fullness_bits == 0
    }
}

/// Frame type enum used for bit allocation decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Intra-only key frame.
    I,
    /// Predicted frame.
    P,
    /// Bi-directionally predicted frame.
    B,
}

/// Advanced bitrate controller.
///
/// Combines mode, model, and HRD buffer to compute per-frame bit budgets.
#[derive(Debug, Clone)]
pub struct BitrateController {
    /// Control mode.
    pub mode: BitrateControlMode,
    /// Bitrate model.
    pub model: BitrateModel,
    /// HRD buffer state.
    pub hrd: HrdBuffer,
    /// Current QP estimate (used in CQ mode).
    pub quality_qp: u8,
}

impl BitrateController {
    /// Create a new bitrate controller.
    #[must_use]
    pub fn new(mode: BitrateControlMode, model: BitrateModel) -> Self {
        let size_bits = (model.target_kbps as u64 * 1000 * model.buffer_size_ms as u64) / 1000;
        let target_rate_bps = model.target_kbps as u64 * 1000;
        let hrd = HrdBuffer::new(size_bits, target_rate_bps);
        Self {
            mode,
            model,
            hrd,
            quality_qp: 26,
        }
    }

    /// Allocate bits for a single frame.
    ///
    /// `complexity` is a normalised score [0, 1]; `frame_type` drives the
    /// I/P/B multipliers.
    ///
    /// Returns a bit budget in bits.
    #[must_use]
    pub fn allocate_bits(&self, complexity: f32, frame_type: FrameType) -> u32 {
        let avg_bits = self.model.bits_per_frame(30.0);

        // Frame-type multipliers
        let type_multiplier: f32 = match frame_type {
            FrameType::I => 3.0,
            FrameType::P => 1.0,
            FrameType::B => 0.5,
        };

        // Complexity multiplier: 0.5× at complexity=0, 1.5× at complexity=1
        let complexity_multiplier = 0.5 + complexity;

        let raw_bits = (avg_bits as f32 * type_multiplier * complexity_multiplier) as u32;

        // Clamp to min/max bitrate bounds (per-frame equivalent)
        let min_bits = self
            .model
            .bits_per_frame(30.0)
            .saturating_mul(self.model.min_kbps)
            / self.model.target_kbps.max(1);
        let max_bits = self
            .model
            .bits_per_frame(30.0)
            .saturating_mul(self.model.max_kbps)
            / self.model.target_kbps.max(1);

        // In ConstantQuality mode we do not clamp
        if self.mode == BitrateControlMode::ConstantQuality {
            return raw_bits;
        }

        raw_bits.clamp(min_bits.min(raw_bits), max_bits.max(raw_bits / 2))
    }
}

/// Rolling bitrate history with per-frame accounting.
#[derive(Debug)]
pub struct BitrateHistory {
    /// Ring buffer of (frame_type, bits).
    pub frames: VecDeque<(FrameType, u32)>,
    /// Maximum history length.
    capacity: usize,
}

impl BitrateHistory {
    /// Create with a given maximum capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a new frame record.
    pub fn push(&mut self, frame_type: FrameType, bits: u32) {
        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back((frame_type, bits));
    }

    /// Moving average over the last `n` frames (all frame types).
    #[must_use]
    pub fn moving_average(&self, n: usize) -> f32 {
        let skip = self.frames.len().saturating_sub(n);
        let window: Vec<u32> = self.frames.iter().skip(skip).map(|(_, b)| *b).collect();
        if window.is_empty() {
            return 0.0;
        }
        window.iter().sum::<u32>() as f32 / window.len() as f32
    }

    /// Moving average over the last `n` frames, limited to a specific type.
    #[must_use]
    pub fn moving_average_for_type(&self, n: usize, frame_type: FrameType) -> f32 {
        let skip = self.frames.len().saturating_sub(n);
        let window: Vec<u32> = self
            .frames
            .iter()
            .skip(skip)
            .filter(|(ft, _)| *ft == frame_type)
            .map(|(_, b)| *b)
            .collect();
        if window.is_empty() {
            return 0.0;
        }
        window.iter().sum::<u32>() as f32 / window.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_model() -> BitrateModel {
        BitrateModel::new(5000, 1000, 15000, 2000)
    }

    #[test]
    fn test_bitrate_model_bits_per_frame() {
        let model = BitrateModel::new(6000, 1000, 12000, 2000);
        let bpf = model.bits_per_frame(30.0);
        assert_eq!(bpf, 200_000);
    }

    #[test]
    fn test_hrd_buffer_add_and_drain() {
        let mut hrd = HrdBuffer::new(1_000_000, 500_000);
        hrd.add_bits(100_000);
        assert!(hrd.fullness_bits > 500_000);
        let fill_before = hrd.fullness_bits;
        hrd.drain_bits(500); // 250 000 bits at 500kbps
        assert!(hrd.fullness_bits < fill_before);
    }

    #[test]
    fn test_hrd_fill_level_pct() {
        let hrd = HrdBuffer {
            size_bits: 1_000_000,
            fullness_bits: 500_000,
            target_rate_bps: 500_000,
        };
        assert!((hrd.fill_level_pct() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_hrd_overflow() {
        let mut hrd = HrdBuffer::new(1000, 500);
        hrd.add_bits(2000);
        assert!(hrd.is_overflow());
    }

    #[test]
    fn test_hrd_underflow() {
        let mut hrd = HrdBuffer::new(1_000_000, 1_000_000);
        hrd.fullness_bits = 0;
        assert!(hrd.is_underflow());
    }

    #[test]
    fn test_allocate_bits_i_frame_larger() {
        let ctrl = BitrateController::new(BitrateControlMode::VariableBitrate, default_model());
        let i_bits = ctrl.allocate_bits(0.5, FrameType::I);
        let p_bits = ctrl.allocate_bits(0.5, FrameType::P);
        let b_bits = ctrl.allocate_bits(0.5, FrameType::B);
        assert!(i_bits > p_bits, "I should get more bits than P");
        assert!(p_bits > b_bits, "P should get more bits than B");
    }

    #[test]
    fn test_allocate_bits_complexity_scales() {
        let ctrl = BitrateController::new(BitrateControlMode::VariableBitrate, default_model());
        let low = ctrl.allocate_bits(0.0, FrameType::P);
        let high = ctrl.allocate_bits(1.0, FrameType::P);
        assert!(high > low);
    }

    #[test]
    fn test_bitrate_history_moving_average() {
        let mut hist = BitrateHistory::new(100);
        hist.push(FrameType::I, 600_000);
        hist.push(FrameType::P, 200_000);
        hist.push(FrameType::P, 200_000);
        let avg = hist.moving_average(3);
        assert!((avg - 333_333.33).abs() < 1.0, "avg={avg}");
    }

    #[test]
    fn test_bitrate_history_type_filter() {
        let mut hist = BitrateHistory::new(100);
        hist.push(FrameType::I, 600_000);
        hist.push(FrameType::P, 200_000);
        hist.push(FrameType::B, 100_000);
        let avg_p = hist.moving_average_for_type(10, FrameType::P);
        assert_eq!(avg_p, 200_000.0);
    }

    #[test]
    fn test_bitrate_history_capacity() {
        let mut hist = BitrateHistory::new(3);
        for i in 0..10u32 {
            hist.push(FrameType::P, i * 1000);
        }
        assert_eq!(hist.frames.len(), 3);
    }

    #[test]
    fn test_moving_average_empty() {
        let hist = BitrateHistory::new(10);
        assert_eq!(hist.moving_average(5), 0.0);
    }

    #[test]
    fn test_controller_cbr_mode() {
        let model = BitrateModel::new(5000, 5000, 5000, 1000);
        let ctrl = BitrateController::new(BitrateControlMode::ConstantBitrate, model);
        let bits = ctrl.allocate_bits(0.5, FrameType::P);
        assert!(bits > 0);
    }

    #[test]
    fn test_hrd_drain_does_not_go_negative() {
        let mut hrd = HrdBuffer::new(1_000_000, 500_000);
        hrd.fullness_bits = 100;
        hrd.drain_bits(10_000); // much more than available
        assert_eq!(hrd.fullness_bits, 0);
    }
}

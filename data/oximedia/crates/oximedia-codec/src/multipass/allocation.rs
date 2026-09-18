//! Bitrate allocation algorithms for multi-pass encoding.
//!
//! This module implements sophisticated bitrate allocation strategies that
//! distribute bits optimally across frames based on their complexity and
//! importance.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::too_many_arguments)]

use crate::frame::FrameType;
use crate::multipass::stats::{FrameStatistics, PassStatistics};
use crate::multipass::vbv::VbvBuffer;

/// Bitrate allocation strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Uniform allocation (equal bits per frame).
    Uniform,
    /// Complexity-based allocation.
    Complexity,
    /// Perceptual optimization (allocate more bits to important frames).
    Perceptual,
    /// Two-pass optimal allocation.
    TwoPass,
}

/// Configuration for bitrate allocation.
#[derive(Clone, Debug)]
pub struct AllocationConfig {
    /// Allocation strategy.
    pub strategy: AllocationStrategy,
    /// Target bitrate in bits per second.
    pub target_bitrate: u64,
    /// Frame rate numerator.
    pub framerate_num: u32,
    /// Frame rate denominator.
    pub framerate_den: u32,
    /// I-frame bit boost factor (1.0-10.0).
    pub i_frame_boost: f64,
    /// P-frame bit factor (relative to average).
    pub p_frame_factor: f64,
    /// B-frame bit factor (relative to average).
    pub b_frame_factor: f64,
    /// Complexity weight (0.0-1.0, how much to favor complex frames).
    pub complexity_weight: f64,
    /// Temporal weight (0.0-1.0, how much to favor temporally important frames).
    pub temporal_weight: f64,
}

impl Default for AllocationConfig {
    fn default() -> Self {
        Self {
            strategy: AllocationStrategy::Complexity,
            target_bitrate: 5_000_000,
            framerate_num: 30,
            framerate_den: 1,
            i_frame_boost: 3.0,
            p_frame_factor: 1.0,
            b_frame_factor: 0.5,
            complexity_weight: 0.7,
            temporal_weight: 0.3,
        }
    }
}

impl AllocationConfig {
    /// Create a new allocation configuration.
    #[must_use]
    pub fn new(strategy: AllocationStrategy, target_bitrate: u64) -> Self {
        Self {
            strategy,
            target_bitrate,
            ..Default::default()
        }
    }

    /// Set frame rate.
    #[must_use]
    pub fn with_framerate(mut self, num: u32, den: u32) -> Self {
        self.framerate_num = num;
        self.framerate_den = den;
        self
    }

    /// Calculate average bits per frame.
    #[must_use]
    pub fn bits_per_frame(&self) -> f64 {
        let fps = self.framerate_num as f64 / self.framerate_den as f64;
        self.target_bitrate as f64 / fps
    }
}

/// Bitrate allocator for video encoding.
pub struct BitrateAllocator {
    config: AllocationConfig,
    first_pass_stats: Option<PassStatistics>,
    total_allocated: u64,
    frames_allocated: u64,
}

impl BitrateAllocator {
    /// Create a new bitrate allocator.
    #[must_use]
    pub fn new(config: AllocationConfig) -> Self {
        Self {
            config,
            first_pass_stats: None,
            total_allocated: 0,
            frames_allocated: 0,
        }
    }

    /// Set first-pass statistics for two-pass encoding.
    pub fn set_first_pass_stats(&mut self, stats: PassStatistics) {
        self.first_pass_stats = Some(stats);
    }

    /// Allocate bits for a frame.
    #[must_use]
    pub fn allocate(
        &mut self,
        frame_index: u64,
        frame_type: FrameType,
        complexity: f64,
    ) -> FrameAllocation {
        let allocation = match self.config.strategy {
            AllocationStrategy::Uniform => self.allocate_uniform(frame_type),
            AllocationStrategy::Complexity => self.allocate_complexity(frame_type, complexity),
            AllocationStrategy::Perceptual => self.allocate_perceptual(frame_type, complexity),
            AllocationStrategy::TwoPass => {
                if let Some(ref stats) = self.first_pass_stats {
                    self.allocate_two_pass(frame_index, frame_type, stats)
                } else {
                    self.allocate_complexity(frame_type, complexity)
                }
            }
        };

        self.total_allocated += allocation.target_bits;
        self.frames_allocated += 1;

        allocation
    }

    /// Uniform allocation (equal bits per frame, adjusted for frame type).
    fn allocate_uniform(&self, frame_type: FrameType) -> FrameAllocation {
        let base_bits = self.config.bits_per_frame();
        let type_factor = self.get_frame_type_factor(frame_type);
        let target_bits = (base_bits * type_factor) as u64;

        FrameAllocation {
            target_bits,
            min_bits: (target_bits as f64 * 0.5) as u64,
            max_bits: (target_bits as f64 * 2.0) as u64,
            qp_adjustment: 0.0,
        }
    }

    /// Complexity-based allocation.
    fn allocate_complexity(&self, frame_type: FrameType, complexity: f64) -> FrameAllocation {
        let base_bits = self.config.bits_per_frame();
        let type_factor = self.get_frame_type_factor(frame_type);

        // Adjust based on complexity (higher complexity = more bits)
        let complexity_factor = 0.5 + complexity;
        let target_bits = (base_bits * type_factor * complexity_factor) as u64;

        FrameAllocation {
            target_bits,
            min_bits: (target_bits as f64 * 0.5) as u64,
            max_bits: (target_bits as f64 * 2.5) as u64,
            qp_adjustment: -((complexity - 0.5) * 10.0), // More complex = lower QP
        }
    }

    /// Perceptual allocation (favor important frames).
    fn allocate_perceptual(&self, frame_type: FrameType, complexity: f64) -> FrameAllocation {
        let base_bits = self.config.bits_per_frame();
        let type_factor = self.get_frame_type_factor(frame_type);

        // Perceptual importance (higher for keyframes and complex frames)
        let perceptual_importance = match frame_type {
            FrameType::Key => 2.0,
            FrameType::Inter => 1.0 + complexity * 0.5,
            FrameType::BiDir => 0.7,
            FrameType::Switch => 1.5,
        };

        let target_bits = (base_bits * type_factor * perceptual_importance) as u64;

        FrameAllocation {
            target_bits,
            min_bits: (target_bits as f64 * 0.6) as u64,
            max_bits: (target_bits as f64 * 2.0) as u64,
            qp_adjustment: -(perceptual_importance - 1.0) * 5.0,
        }
    }

    /// Two-pass optimal allocation using first-pass statistics.
    fn allocate_two_pass(
        &self,
        frame_index: u64,
        frame_type: FrameType,
        stats: &PassStatistics,
    ) -> FrameAllocation {
        // Get first-pass statistics for this frame
        let frame_stats = stats.get_frame(frame_index);

        if let Some(frame_stats) = frame_stats {
            // Calculate target bits based on first-pass results
            let total_bits_available = self.calculate_remaining_budget(stats);
            let frames_remaining = (stats.total_frames - frame_index) as f64;

            // Weight based on first-pass complexity and bits used
            let weight = self.calculate_frame_weight(frame_stats, stats);
            let total_weight = self.calculate_total_remaining_weight(frame_index, stats);

            let target_bits = if total_weight > 0.0 {
                ((total_bits_available as f64) * weight / total_weight) as u64
            } else {
                (total_bits_available as f64 / frames_remaining) as u64
            };

            // QP adjustment based on first-pass QP
            let qp_adjustment = self.calculate_qp_adjustment(frame_stats, stats);

            FrameAllocation {
                target_bits,
                min_bits: (target_bits as f64 * 0.5) as u64,
                max_bits: (target_bits as f64 * 3.0) as u64,
                qp_adjustment,
            }
        } else {
            // Fallback if frame stats not found
            self.allocate_complexity(frame_type, 0.5)
        }
    }

    /// Calculate remaining bitrate budget.
    fn calculate_remaining_budget(&self, stats: &PassStatistics) -> u64 {
        let total_duration = stats.total_frames as f64
            * (self.config.framerate_den as f64 / self.config.framerate_num as f64);
        let total_bits = (self.config.target_bitrate as f64 * total_duration) as u64;

        total_bits.saturating_sub(self.total_allocated)
    }

    /// Calculate weight for a frame based on first-pass data.
    fn calculate_frame_weight(&self, frame_stats: &FrameStatistics, stats: &PassStatistics) -> f64 {
        // Normalize complexity
        let complexity_dist = stats.complexity_distribution();
        let normalized_complexity = if complexity_dist.mean > 0.0 {
            frame_stats.complexity.combined_complexity / complexity_dist.mean
        } else {
            1.0
        };

        // Weight by frame type
        let type_weight = match frame_stats.frame_type {
            FrameType::Key => self.config.i_frame_boost,
            FrameType::Inter => self.config.p_frame_factor,
            FrameType::BiDir => self.config.b_frame_factor,
            FrameType::Switch => 1.5,
        };

        // Combine weights
        self.config.complexity_weight * normalized_complexity
            + (1.0 - self.config.complexity_weight) * type_weight
    }

    /// Calculate total weight of remaining frames.
    fn calculate_total_remaining_weight(&self, current_index: u64, stats: &PassStatistics) -> f64 {
        stats
            .frames
            .iter()
            .filter(|f| f.frame_index >= current_index)
            .map(|f| self.calculate_frame_weight(f, stats))
            .sum()
    }

    /// Calculate QP adjustment for second pass.
    fn calculate_qp_adjustment(
        &self,
        frame_stats: &FrameStatistics,
        stats: &PassStatistics,
    ) -> f64 {
        let avg_qp = stats.avg_qp;
        let frame_qp = frame_stats.qp;

        // Adjust QP relative to first-pass average
        (avg_qp - frame_qp) * 0.5 // Dampen the adjustment
    }

    /// Get frame type factor for bit allocation.
    fn get_frame_type_factor(&self, frame_type: FrameType) -> f64 {
        match frame_type {
            FrameType::Key => self.config.i_frame_boost,
            FrameType::Inter => self.config.p_frame_factor,
            FrameType::BiDir => self.config.b_frame_factor,
            FrameType::Switch => 1.5,
        }
    }

    /// Get total bits allocated.
    #[must_use]
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated
    }

    /// Get number of frames allocated.
    #[must_use]
    pub fn frames_allocated(&self) -> u64 {
        self.frames_allocated
    }

    /// Reset allocator state.
    pub fn reset(&mut self) {
        self.total_allocated = 0;
        self.frames_allocated = 0;
    }
}

/// Bit allocation result for a single frame.
#[derive(Clone, Debug)]
pub struct FrameAllocation {
    /// Target bits for the frame.
    pub target_bits: u64,
    /// Minimum acceptable bits.
    pub min_bits: u64,
    /// Maximum allowed bits.
    pub max_bits: u64,
    /// Suggested QP adjustment from base.
    pub qp_adjustment: f64,
}

impl FrameAllocation {
    /// Clamp actual bits to acceptable range.
    #[must_use]
    pub fn clamp_bits(&self, actual_bits: u64) -> u64 {
        actual_bits.clamp(self.min_bits, self.max_bits)
    }

    /// Check if actual bits are within acceptable range.
    #[must_use]
    pub fn is_within_range(&self, actual_bits: u64) -> bool {
        actual_bits >= self.min_bits && actual_bits <= self.max_bits
    }

    /// Calculate error from target (positive = over, negative = under).
    #[must_use]
    pub fn error(&self, actual_bits: u64) -> i64 {
        actual_bits as i64 - self.target_bits as i64
    }
}

/// VBV-aware bitrate allocator.
pub struct VbvAwareAllocator {
    allocator: BitrateAllocator,
    vbv_buffer: Option<VbvBuffer>,
}

impl VbvAwareAllocator {
    /// Create a new VBV-aware allocator.
    #[must_use]
    pub fn new(config: AllocationConfig) -> Self {
        Self {
            allocator: BitrateAllocator::new(config),
            vbv_buffer: None,
        }
    }

    /// Set VBV buffer for allocation constraints.
    pub fn set_vbv_buffer(&mut self, vbv_buffer: VbvBuffer) {
        self.vbv_buffer = Some(vbv_buffer);
    }

    /// Allocate bits with VBV constraints.
    #[must_use]
    pub fn allocate(
        &mut self,
        frame_index: u64,
        frame_type: FrameType,
        complexity: f64,
    ) -> FrameAllocation {
        let mut allocation = self.allocator.allocate(frame_index, frame_type, complexity);

        // Apply VBV constraints if buffer is set
        if let Some(ref vbv) = self.vbv_buffer {
            let max_allowed = vbv.max_frame_size();
            allocation.max_bits = allocation.max_bits.min(max_allowed);
            allocation.target_bits = allocation.target_bits.min(max_allowed);
            allocation.min_bits = allocation.min_bits.min(allocation.target_bits);
        }

        allocation
    }

    /// Update VBV buffer after encoding.
    pub fn update_vbv(&mut self, frame_bits: u64) {
        if let Some(ref mut vbv) = self.vbv_buffer {
            vbv.update(frame_bits);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_config_new() {
        let config = AllocationConfig::new(AllocationStrategy::Complexity, 5_000_000);
        assert_eq!(config.strategy, AllocationStrategy::Complexity);
        assert_eq!(config.target_bitrate, 5_000_000);
    }

    #[test]
    fn test_allocation_config_bits_per_frame() {
        let config =
            AllocationConfig::new(AllocationStrategy::Uniform, 3_000_000).with_framerate(30, 1);
        let bpf = config.bits_per_frame();
        assert!((bpf - 100_000.0).abs() < 1.0); // 3M / 30 = 100k
    }

    #[test]
    fn test_allocator_uniform() {
        let config =
            AllocationConfig::new(AllocationStrategy::Uniform, 3_000_000).with_framerate(30, 1);
        let mut allocator = BitrateAllocator::new(config);

        let alloc = allocator.allocate(0, FrameType::Inter, 0.5);
        assert!(alloc.target_bits > 0);
        assert!(alloc.min_bits < alloc.target_bits);
        assert!(alloc.max_bits > alloc.target_bits);
    }

    #[test]
    fn test_allocator_complexity() {
        let config =
            AllocationConfig::new(AllocationStrategy::Complexity, 3_000_000).with_framerate(30, 1);
        let mut allocator = BitrateAllocator::new(config);

        let low_complexity = allocator.allocate(0, FrameType::Inter, 0.2);
        let high_complexity = allocator.allocate(1, FrameType::Inter, 0.8);

        // High complexity should get more bits
        assert!(high_complexity.target_bits > low_complexity.target_bits);
    }

    #[test]
    fn test_allocator_keyframe_boost() {
        let config =
            AllocationConfig::new(AllocationStrategy::Uniform, 3_000_000).with_framerate(30, 1);
        let mut allocator = BitrateAllocator::new(config);

        let keyframe_alloc = allocator.allocate(0, FrameType::Key, 0.5);
        let inter_alloc = allocator.allocate(1, FrameType::Inter, 0.5);

        // Keyframes should get more bits
        assert!(keyframe_alloc.target_bits > inter_alloc.target_bits);
    }

    #[test]
    fn test_frame_allocation_clamp() {
        let alloc = FrameAllocation {
            target_bits: 100_000,
            min_bits: 50_000,
            max_bits: 200_000,
            qp_adjustment: 0.0,
        };

        assert_eq!(alloc.clamp_bits(30_000), 50_000); // Below min
        assert_eq!(alloc.clamp_bits(100_000), 100_000); // Within range
        assert_eq!(alloc.clamp_bits(250_000), 200_000); // Above max
    }

    #[test]
    fn test_frame_allocation_within_range() {
        let alloc = FrameAllocation {
            target_bits: 100_000,
            min_bits: 50_000,
            max_bits: 200_000,
            qp_adjustment: 0.0,
        };

        assert!(!alloc.is_within_range(30_000));
        assert!(alloc.is_within_range(100_000));
        assert!(!alloc.is_within_range(250_000));
    }
}

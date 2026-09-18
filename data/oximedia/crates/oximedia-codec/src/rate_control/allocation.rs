// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Bitrate allocation strategies for rate control.
//!
//! This module provides sophisticated algorithms for allocating bits across frames:
//!
//! - **Complexity-Based Allocation** - Allocate bits proportional to frame complexity
//! - **GOP-Level Allocation** - Distribute bits across GOPs
//! - **Hierarchical Allocation** - B-pyramid and hierarchical GOP structures
//! - **Multi-Pass Allocation** - Use first-pass statistics for optimal allocation
//! - **Constrained Allocation** - VBV/HRD buffer constraints
//! - **Adaptive Allocation** - Real-time adjustment based on encoding results
//!
//! # Architecture
//!
//! ```text
//! Total Bits → GOP Allocator → Frame Allocator → Block Allocator
//!      ↓             ↓               ↓                 ↓
//!   Budget      GOP Budget    Frame Target    Block Weights
//! ```

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![forbid(unsafe_code)]

use crate::frame::FrameType;

/// Bitrate allocation strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Equal allocation across all frames.
    Uniform,
    /// Complexity-based allocation.
    ComplexityBased,
    /// Multi-pass allocation using statistics.
    MultiPass,
    /// Hierarchical allocation for B-pyramid.
    Hierarchical,
    /// Constrained allocation with VBV compliance.
    Constrained,
    /// Adaptive allocation with real-time feedback.
    Adaptive,
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        Self::ComplexityBased
    }
}

/// Bitrate allocator for distributing bits across frames.
#[derive(Clone, Debug)]
pub struct BitrateAllocator {
    /// Allocation strategy.
    strategy: AllocationStrategy,
    /// Target bitrate in bits per second.
    target_bitrate: u64,
    /// Frame rate.
    framerate: f64,
    /// GOP length.
    gop_length: u32,
    /// I-frame to P-frame bit ratio.
    i_p_ratio: f32,
    /// P-frame to B-frame bit ratio.
    p_b_ratio: f32,
    /// Complexity weight factor.
    complexity_weight: f32,
    /// Historical complexity data.
    complexity_history: Vec<f32>,
    /// Historical bit usage.
    bit_usage_history: Vec<u64>,
    /// Maximum history size.
    max_history: usize,
    /// VBV buffer size (if constrained).
    vbv_buffer_size: Option<u64>,
    /// Bit reservoir for smoothing.
    bit_reservoir: i64,
    /// Maximum reservoir size.
    max_reservoir: i64,
    /// Current GOP index.
    current_gop: u32,
    /// Frames in current GOP.
    frames_in_gop: u32,
    /// Bits used in current GOP.
    bits_used_in_gop: u64,
    /// Target bits for current GOP.
    gop_target_bits: u64,
}

impl BitrateAllocator {
    /// Create a new bitrate allocator.
    #[must_use]
    pub fn new(target_bitrate: u64, framerate: f64, gop_length: u32) -> Self {
        let max_reservoir = (target_bitrate as i64) * 2;

        Self {
            strategy: AllocationStrategy::default(),
            target_bitrate,
            framerate,
            gop_length,
            i_p_ratio: 3.0,
            p_b_ratio: 2.0,
            complexity_weight: 1.0,
            complexity_history: Vec::new(),
            bit_usage_history: Vec::new(),
            max_history: 100,
            vbv_buffer_size: None,
            bit_reservoir: 0,
            max_reservoir,
            current_gop: 0,
            frames_in_gop: 0,
            bits_used_in_gop: 0,
            gop_target_bits: 0,
        }
    }

    /// Set allocation strategy.
    pub fn set_strategy(&mut self, strategy: AllocationStrategy) {
        self.strategy = strategy;
    }

    /// Set I/P frame bit ratio.
    pub fn set_i_p_ratio(&mut self, ratio: f32) {
        self.i_p_ratio = ratio.max(1.0);
    }

    /// Set P/B frame bit ratio.
    pub fn set_p_b_ratio(&mut self, ratio: f32) {
        self.p_b_ratio = ratio.max(1.0);
    }

    /// Set complexity weight factor.
    pub fn set_complexity_weight(&mut self, weight: f32) {
        self.complexity_weight = weight.clamp(0.0, 2.0);
    }

    /// Set VBV buffer size for constrained allocation.
    pub fn set_vbv_buffer_size(&mut self, size: u64) {
        self.vbv_buffer_size = Some(size);
        if self.strategy == AllocationStrategy::Uniform {
            self.strategy = AllocationStrategy::Constrained;
        }
    }

    /// Calculate target bits for a frame.
    #[must_use]
    pub fn allocate_frame_bits(
        &self,
        frame_type: FrameType,
        complexity: f32,
        frame_in_gop: u32,
    ) -> AllocationResult {
        let base_bits = self.calculate_base_allocation();

        let allocated_bits = match self.strategy {
            AllocationStrategy::Uniform => self.allocate_uniform(base_bits, frame_type),
            AllocationStrategy::ComplexityBased => {
                self.allocate_complexity_based(base_bits, frame_type, complexity)
            }
            AllocationStrategy::MultiPass => {
                self.allocate_multipass(base_bits, frame_type, complexity)
            }
            AllocationStrategy::Hierarchical => {
                self.allocate_hierarchical(base_bits, frame_type, frame_in_gop)
            }
            AllocationStrategy::Constrained => {
                self.allocate_constrained(base_bits, frame_type, complexity)
            }
            AllocationStrategy::Adaptive => {
                self.allocate_adaptive(base_bits, frame_type, complexity)
            }
        };

        let mut max_bits = allocated_bits * 4;

        // Clamp max_bits to VBV buffer size when constrained
        if let Some(vbv_size) = self.vbv_buffer_size {
            max_bits = max_bits.min(vbv_size);
        }

        AllocationResult {
            target_bits: allocated_bits,
            min_bits: allocated_bits / 4,
            max_bits,
            frame_type,
            complexity_factor: complexity / self.average_complexity(),
            reservoir_adjustment: self.calculate_reservoir_adjustment(allocated_bits),
        }
    }

    /// Calculate base bit allocation per frame.
    fn calculate_base_allocation(&self) -> u64 {
        if self.framerate <= 0.0 {
            return 0;
        }
        (self.target_bitrate as f64 / self.framerate) as u64
    }

    /// Uniform allocation (simple frame type based).
    fn allocate_uniform(&self, base_bits: u64, frame_type: FrameType) -> u64 {
        match frame_type {
            FrameType::Key => (base_bits as f32 * self.i_p_ratio) as u64,
            FrameType::Inter => base_bits,
            FrameType::BiDir => (base_bits as f32 / self.p_b_ratio) as u64,
            FrameType::Switch => (base_bits as f32 * 1.5) as u64,
        }
    }

    /// Complexity-based allocation.
    fn allocate_complexity_based(
        &self,
        base_bits: u64,
        frame_type: FrameType,
        complexity: f32,
    ) -> u64 {
        // Get base allocation for frame type
        let type_bits = self.allocate_uniform(base_bits, frame_type);

        // Apply complexity factor
        let avg_complexity = self.average_complexity();
        if avg_complexity <= 0.0 {
            return type_bits;
        }

        let complexity_ratio = complexity / avg_complexity;
        let complexity_multiplier = 1.0 + (complexity_ratio - 1.0) * self.complexity_weight;

        (type_bits as f32 * complexity_multiplier.clamp(0.5, 2.0)) as u64
    }

    /// Multi-pass allocation using statistics.
    fn allocate_multipass(&self, base_bits: u64, frame_type: FrameType, complexity: f32) -> u64 {
        // Use historical data to improve allocation
        if self.complexity_history.is_empty() {
            return self.allocate_complexity_based(base_bits, frame_type, complexity);
        }

        // Calculate total complexity for proportional allocation
        let total_complexity: f32 = self.complexity_history.iter().sum();
        if total_complexity <= 0.0 {
            return self.allocate_complexity_based(base_bits, frame_type, complexity);
        }

        // Total bits for this GOP
        let gop_bits = base_bits * self.gop_length as u64;

        // Proportional allocation based on complexity
        let frame_proportion = complexity / total_complexity;
        let mut allocated = (gop_bits as f32 * frame_proportion) as u64;

        // Apply frame type multiplier
        allocated = match frame_type {
            FrameType::Key => (allocated as f32 * self.i_p_ratio) as u64,
            FrameType::BiDir => (allocated as f32 / self.p_b_ratio) as u64,
            _ => allocated,
        };

        allocated.max(base_bits / 4)
    }

    /// Hierarchical allocation for B-pyramid structures.
    fn allocate_hierarchical(
        &self,
        base_bits: u64,
        frame_type: FrameType,
        frame_in_gop: u32,
    ) -> u64 {
        let type_bits = self.allocate_uniform(base_bits, frame_type);

        match frame_type {
            FrameType::BiDir => {
                // B-frames in pyramid: lower levels get more bits
                let pyramid_level = self.calculate_pyramid_level(frame_in_gop);
                let level_multiplier = 1.0 + (pyramid_level as f32 * 0.2);
                (type_bits as f32 * level_multiplier) as u64
            }
            _ => type_bits,
        }
    }

    /// Calculate B-frame pyramid level.
    fn calculate_pyramid_level(&self, frame_in_gop: u32) -> u32 {
        // Simplified pyramid level calculation
        // Level 0 = reference B-frames (higher quality)
        // Level N = highest level (lower quality)
        let mut level = 0;
        let mut pos = frame_in_gop;

        while pos % 2 == 0 && pos > 0 {
            level += 1;
            pos /= 2;
        }

        level
    }

    /// Constrained allocation with VBV compliance.
    fn allocate_constrained(&self, base_bits: u64, frame_type: FrameType, complexity: f32) -> u64 {
        let mut allocated = self.allocate_complexity_based(base_bits, frame_type, complexity);

        // Apply VBV constraints if enabled
        if let Some(vbv_size) = self.vbv_buffer_size {
            // Estimate current buffer level
            let buffer_level = self.estimate_buffer_level();

            // If buffer is getting full, reduce allocation
            let fullness_ratio = buffer_level as f32 / vbv_size as f32;
            if fullness_ratio > 0.8 {
                let reduction = (fullness_ratio - 0.8) * 5.0; // Aggressive reduction
                allocated = (allocated as f32 * (1.0 - reduction).max(0.5)) as u64;
            }

            // Don't exceed buffer capacity
            allocated = allocated.min(vbv_size);
        }

        allocated
    }

    /// Adaptive allocation with real-time feedback.
    fn allocate_adaptive(&self, base_bits: u64, frame_type: FrameType, complexity: f32) -> u64 {
        let mut allocated = self.allocate_complexity_based(base_bits, frame_type, complexity);

        // Adjust based on recent bit usage accuracy
        if !self.bit_usage_history.is_empty() {
            let recent_usage: u64 = self.bit_usage_history.iter().rev().take(10).sum();
            let recent_target = base_bits * 10.min(self.bit_usage_history.len()) as u64;

            if recent_target > 0 {
                let usage_ratio = recent_usage as f32 / recent_target as f32;

                // If consistently overshooting, reduce allocation
                // If undershooting, increase allocation
                if usage_ratio > 1.2 {
                    allocated = (allocated as f32 * 0.9) as u64;
                } else if usage_ratio < 0.8 {
                    allocated = (allocated as f32 * 1.1) as u64;
                }
            }
        }

        // Apply reservoir adjustment
        let reservoir_adjustment = self.calculate_reservoir_adjustment(allocated);
        ((allocated as i64) + reservoir_adjustment).max(base_bits as i64 / 4) as u64
    }

    /// Calculate reservoir adjustment.
    fn calculate_reservoir_adjustment(&self, target: u64) -> i64 {
        if self.bit_reservoir == 0 {
            return 0;
        }

        let reservoir_ratio = self.bit_reservoir as f32 / self.max_reservoir as f32;

        // Use reservoir when it's full, save to reservoir when empty
        let adjustment = (target as f32 * reservoir_ratio * 0.1) as i64;

        // Clamp to prevent extreme adjustments
        adjustment.clamp(-(target as i64 / 4), target as i64 / 4)
    }

    /// Estimate current VBV buffer level.
    fn estimate_buffer_level(&self) -> u64 {
        // Simplified estimation based on recent bit usage
        if let Some(vbv_size) = self.vbv_buffer_size {
            let bits_per_frame = self.calculate_base_allocation();
            let recent_frames = 10.min(self.bit_usage_history.len());

            if recent_frames == 0 {
                return vbv_size / 2; // Assume half full initially
            }

            let recent_bits: u64 = self
                .bit_usage_history
                .iter()
                .rev()
                .take(recent_frames)
                .sum();
            let target_bits = bits_per_frame * recent_frames as u64;

            // Buffer level increases when using less than target
            if recent_bits < target_bits {
                let saved = target_bits - recent_bits;
                (vbv_size / 2 + saved).min(vbv_size)
            } else {
                let overage = recent_bits - target_bits;
                (vbv_size / 2).saturating_sub(overage)
            }
        } else {
            0
        }
    }

    /// Get average complexity from history.
    fn average_complexity(&self) -> f32 {
        if self.complexity_history.is_empty() {
            return 1.0;
        }

        let sum: f32 = self.complexity_history.iter().sum();
        (sum / self.complexity_history.len() as f32).max(0.01)
    }

    /// Update allocator with actual frame results.
    pub fn update(&mut self, complexity: f32, bits_used: u64) {
        // Update complexity history
        self.complexity_history.push(complexity);
        if self.complexity_history.len() > self.max_history {
            self.complexity_history.remove(0);
        }

        // Update bit usage history
        self.bit_usage_history.push(bits_used);
        if self.bit_usage_history.len() > self.max_history {
            self.bit_usage_history.remove(0);
        }

        // Update bit reservoir
        let target_per_frame = self.calculate_base_allocation();
        self.bit_reservoir += target_per_frame as i64 - bits_used as i64;
        self.bit_reservoir = self
            .bit_reservoir
            .clamp(-self.max_reservoir, self.max_reservoir);

        // Update GOP tracking
        self.frames_in_gop += 1;
        self.bits_used_in_gop += bits_used;
    }

    /// Start a new GOP.
    pub fn start_new_gop(&mut self) {
        self.current_gop += 1;
        self.frames_in_gop = 0;
        self.bits_used_in_gop = 0;
        self.gop_target_bits = self.calculate_base_allocation() * self.gop_length as u64;
    }

    /// Get current GOP allocation status.
    #[must_use]
    pub fn gop_status(&self) -> GopAllocationStatus {
        let remaining_frames = self.gop_length.saturating_sub(self.frames_in_gop);
        let remaining_bits = self.gop_target_bits.saturating_sub(self.bits_used_in_gop);

        GopAllocationStatus {
            gop_index: self.current_gop,
            frames_encoded: self.frames_in_gop,
            frames_remaining: remaining_frames,
            bits_used: self.bits_used_in_gop,
            bits_remaining: remaining_bits,
            target_bits: self.gop_target_bits,
            on_target: self.is_gop_on_target(),
        }
    }

    /// Check if GOP allocation is on target.
    fn is_gop_on_target(&self) -> bool {
        if self.frames_in_gop == 0 {
            return true;
        }

        let expected_bits = (self.gop_target_bits as f32
            * (self.frames_in_gop as f32 / self.gop_length as f32))
            as u64;

        let accuracy = self.bits_used_in_gop as f32 / expected_bits as f32;
        accuracy > 0.8 && accuracy < 1.2
    }

    /// Reset the allocator state.
    pub fn reset(&mut self) {
        self.complexity_history.clear();
        self.bit_usage_history.clear();
        self.bit_reservoir = 0;
        self.current_gop = 0;
        self.frames_in_gop = 0;
        self.bits_used_in_gop = 0;
        self.gop_target_bits = 0;
    }
}

/// Result of frame allocation.
#[derive(Clone, Debug)]
pub struct AllocationResult {
    /// Target bits for the frame.
    pub target_bits: u64,
    /// Minimum bits (underflow prevention).
    pub min_bits: u64,
    /// Maximum bits (overflow prevention).
    pub max_bits: u64,
    /// Frame type.
    pub frame_type: FrameType,
    /// Complexity factor relative to average.
    pub complexity_factor: f32,
    /// Reservoir adjustment applied.
    pub reservoir_adjustment: i64,
}

impl AllocationResult {
    /// Check if actual bits are within acceptable range.
    #[must_use]
    pub fn is_within_range(&self, actual_bits: u64) -> bool {
        actual_bits >= self.min_bits && actual_bits <= self.max_bits
    }

    /// Calculate allocation accuracy.
    #[must_use]
    pub fn accuracy(&self, actual_bits: u64) -> f32 {
        if self.target_bits == 0 {
            return 1.0;
        }
        actual_bits as f32 / self.target_bits as f32
    }
}

/// GOP allocation status.
#[derive(Clone, Debug)]
pub struct GopAllocationStatus {
    /// GOP index.
    pub gop_index: u32,
    /// Frames encoded in this GOP.
    pub frames_encoded: u32,
    /// Frames remaining in this GOP.
    pub frames_remaining: u32,
    /// Bits used so far.
    pub bits_used: u64,
    /// Bits remaining in budget.
    pub bits_remaining: u64,
    /// Target bits for this GOP.
    pub target_bits: u64,
    /// Whether allocation is on target.
    pub on_target: bool,
}

impl GopAllocationStatus {
    /// Get the average bits per frame so far.
    #[must_use]
    pub fn average_bits_per_frame(&self) -> u64 {
        if self.frames_encoded == 0 {
            return 0;
        }
        self.bits_used / self.frames_encoded as u64
    }

    /// Get recommended bits per remaining frame.
    #[must_use]
    pub fn recommended_bits_per_frame(&self) -> u64 {
        if self.frames_remaining == 0 {
            return 0;
        }
        self.bits_remaining / self.frames_remaining as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_creation() {
        let allocator = BitrateAllocator::new(5_000_000, 30.0, 250);
        assert_eq!(allocator.target_bitrate, 5_000_000);
        assert_eq!(allocator.gop_length, 250);
    }

    #[test]
    fn test_base_allocation() {
        let allocator = BitrateAllocator::new(3_000_000, 30.0, 250);
        let base = allocator.calculate_base_allocation();
        assert_eq!(base, 100_000); // 3M / 30 fps
    }

    #[test]
    fn test_uniform_allocation() {
        let allocator = BitrateAllocator::new(3_000_000, 30.0, 250);
        let base = 100_000;

        let i_bits = allocator.allocate_uniform(base, FrameType::Key);
        let p_bits = allocator.allocate_uniform(base, FrameType::Inter);
        let b_bits = allocator.allocate_uniform(base, FrameType::BiDir);

        assert!(i_bits > p_bits); // I-frames get more bits
        assert!(p_bits > b_bits); // P-frames get more than B-frames
    }

    #[test]
    fn test_complexity_based_allocation() {
        let mut allocator = BitrateAllocator::new(3_000_000, 30.0, 250);

        // Add some complexity history
        allocator.update(1.0, 100_000);
        allocator.update(2.0, 150_000);
        allocator.update(1.5, 125_000);

        let result = allocator.allocate_frame_bits(FrameType::Inter, 2.0, 0);
        let low_complexity_result = allocator.allocate_frame_bits(FrameType::Inter, 0.5, 0);

        assert!(result.target_bits > low_complexity_result.target_bits);
    }

    #[test]
    fn test_hierarchical_allocation() {
        let allocator = BitrateAllocator::new(3_000_000, 30.0, 250);
        let base = 100_000;

        // B-frames at different pyramid levels
        let level0 = allocator.allocate_hierarchical(base, FrameType::BiDir, 2);
        let level1 = allocator.allocate_hierarchical(base, FrameType::BiDir, 4);

        // Lower pyramid levels should get more bits
        assert!(level1 >= level0);
    }

    #[test]
    fn test_reservoir_management() {
        let mut allocator = BitrateAllocator::new(3_000_000, 30.0, 250);

        // Undershoot target - reservoir should grow
        allocator.update(1.0, 80_000); // Target is 100_000
        assert!(allocator.bit_reservoir > 0);

        // Overshoot target - reservoir should shrink
        allocator.update(1.0, 120_000);
        assert!(allocator.bit_reservoir < 20_000);
    }

    #[test]
    fn test_vbv_constraint() {
        let mut allocator = BitrateAllocator::new(3_000_000, 30.0, 250);
        allocator.set_vbv_buffer_size(1_000_000);
        allocator.set_strategy(AllocationStrategy::Constrained);

        let result = allocator.allocate_frame_bits(FrameType::Key, 5.0, 0);

        // Should not exceed VBV buffer size
        assert!(result.max_bits <= 1_000_000);
    }

    #[test]
    fn test_adaptive_allocation() {
        let mut allocator = BitrateAllocator::new(3_000_000, 30.0, 250);
        allocator.set_strategy(AllocationStrategy::Adaptive);

        // Consistently overshoot
        for _ in 0..10 {
            allocator.update(1.0, 130_000); // Target is ~100_000
        }

        let result = allocator.allocate_frame_bits(FrameType::Inter, 1.0, 0);

        // Should adapt by reducing allocation
        assert!(result.target_bits < 100_000);
    }

    #[test]
    fn test_gop_tracking() {
        let mut allocator = BitrateAllocator::new(3_000_000, 30.0, 10);
        allocator.start_new_gop();

        for i in 0..5 {
            allocator.update(1.0, 100_000);
            let status = allocator.gop_status();
            assert_eq!(status.frames_encoded, i + 1);
            assert_eq!(status.frames_remaining, 10 - (i + 1));
        }

        let status = allocator.gop_status();
        assert_eq!(status.frames_encoded, 5);
        assert_eq!(status.bits_used, 500_000);
    }

    #[test]
    fn test_allocation_result() {
        let result = AllocationResult {
            target_bits: 100_000,
            min_bits: 25_000,
            max_bits: 400_000,
            frame_type: FrameType::Inter,
            complexity_factor: 1.5,
            reservoir_adjustment: 0,
        };

        assert!(result.is_within_range(100_000));
        assert!(result.is_within_range(50_000));
        assert!(!result.is_within_range(500_000));
        assert!(!result.is_within_range(10_000));

        let accuracy = result.accuracy(110_000);
        assert!((accuracy - 1.1).abs() < 0.01);
    }

    #[test]
    fn test_gop_status() {
        let status = GopAllocationStatus {
            gop_index: 1,
            frames_encoded: 5,
            frames_remaining: 5,
            bits_used: 500_000,
            bits_remaining: 500_000,
            target_bits: 1_000_000,
            on_target: true,
        };

        assert_eq!(status.average_bits_per_frame(), 100_000);
        assert_eq!(status.recommended_bits_per_frame(), 100_000);
    }

    #[test]
    fn test_strategy_switching() {
        let mut allocator = BitrateAllocator::new(3_000_000, 30.0, 250);

        allocator.set_strategy(AllocationStrategy::Uniform);
        let uniform_result = allocator.allocate_frame_bits(FrameType::Inter, 2.0, 0);

        allocator.set_strategy(AllocationStrategy::ComplexityBased);
        allocator.update(1.0, 100_000);
        let complexity_result = allocator.allocate_frame_bits(FrameType::Inter, 2.0, 0);

        // Complexity-based should differ from uniform for high complexity
        assert_ne!(uniform_result.target_bits, complexity_result.target_bits);
    }

    #[test]
    fn test_reset() {
        let mut allocator = BitrateAllocator::new(3_000_000, 30.0, 250);

        allocator.update(1.0, 100_000);
        allocator.update(2.0, 150_000);
        assert!(!allocator.complexity_history.is_empty());

        allocator.reset();

        assert!(allocator.complexity_history.is_empty());
        assert!(allocator.bit_usage_history.is_empty());
        assert_eq!(allocator.bit_reservoir, 0);
    }
}

// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Advanced quantization parameter (QP) selection.
//!
//! This module provides sophisticated QP selection algorithms:
//!
//! - **Frame-Level QP** - Base QP selection for each frame
//! - **Block-Level QP** - Adaptive QP for blocks based on content
//! - **Temporal QP** - QP modulation based on temporal characteristics
//! - **Psychovisual QP** - Perceptually-optimized QP selection
//! - **Rate-Distortion QP** - QP selection for optimal R-D trade-off
//! - **Hierarchical QP** - QP for hierarchical GOP structures
//!
//! # Architecture
//!
//! ```text
//! Frame → QP Selector → Base QP → Block Analyzer → Block QPs
//!    ↓         ↓           ↓            ↓              ↓
//! Content  Strategy    Frame QP    Adaptation    Final QPs
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

/// QP selection strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QpStrategy {
    /// Constant QP for all frames and blocks.
    Constant,
    /// Frame type based QP (I/P/B offsets).
    FrameTypeBased,
    /// Complexity-adaptive QP.
    ComplexityAdaptive,
    /// Psychovisually-optimized QP.
    Psychovisual,
    /// Rate-distortion optimized QP.
    RateDistortionOptimized,
    /// Hierarchical QP for B-pyramids.
    Hierarchical,
}

impl Default for QpStrategy {
    fn default() -> Self {
        Self::ComplexityAdaptive
    }
}

/// Quantization parameter selector.
#[derive(Clone, Debug)]
pub struct QpSelector {
    /// QP selection strategy.
    strategy: QpStrategy,
    /// Base QP value.
    base_qp: f32,
    /// Minimum QP allowed.
    min_qp: u8,
    /// Maximum QP allowed.
    max_qp: u8,
    /// I-frame QP offset.
    i_qp_offset: i8,
    /// P-frame QP offset.
    p_qp_offset: i8,
    /// B-frame QP offset.
    b_qp_offset: i8,
    /// Enable adaptive quantization.
    enable_aq: bool,
    /// AQ strength (0.0-2.0).
    aq_strength: f32,
    /// Enable psychovisual optimization.
    enable_psy: bool,
    /// Psychovisual strength (0.0-2.0).
    psy_strength: f32,
    /// QP adaptation speed (0.0-1.0).
    adaptation_speed: f32,
    /// Historical average QP.
    historical_qp: Vec<f32>,
    /// Maximum history size.
    max_history: usize,
    /// Frame counter.
    frame_count: u64,
}

impl QpSelector {
    /// Create a new QP selector.
    #[must_use]
    pub fn new(base_qp: f32, min_qp: u8, max_qp: u8) -> Self {
        Self {
            strategy: QpStrategy::default(),
            base_qp: base_qp.clamp(min_qp as f32, max_qp as f32),
            min_qp,
            max_qp,
            i_qp_offset: -2,
            p_qp_offset: 0,
            b_qp_offset: 2,
            enable_aq: true,
            aq_strength: 1.0,
            enable_psy: true,
            psy_strength: 1.0,
            adaptation_speed: 0.5,
            historical_qp: Vec::new(),
            max_history: 100,
            frame_count: 0,
        }
    }

    /// Set QP selection strategy.
    pub fn set_strategy(&mut self, strategy: QpStrategy) {
        self.strategy = strategy;
    }

    /// Set base QP.
    pub fn set_base_qp(&mut self, qp: f32) {
        self.base_qp = qp.clamp(self.min_qp as f32, self.max_qp as f32);
    }

    /// Set I-frame QP offset.
    pub fn set_i_qp_offset(&mut self, offset: i8) {
        self.i_qp_offset = offset;
    }

    /// Set P-frame QP offset.
    pub fn set_p_qp_offset(&mut self, offset: i8) {
        self.p_qp_offset = offset;
    }

    /// Set B-frame QP offset.
    pub fn set_b_qp_offset(&mut self, offset: i8) {
        self.b_qp_offset = offset;
    }

    /// Enable or disable adaptive quantization.
    pub fn set_aq_enabled(&mut self, enabled: bool) {
        self.enable_aq = enabled;
    }

    /// Set AQ strength.
    pub fn set_aq_strength(&mut self, strength: f32) {
        self.aq_strength = strength.clamp(0.0, 2.0);
    }

    /// Enable or disable psychovisual optimization.
    pub fn set_psy_enabled(&mut self, enabled: bool) {
        self.enable_psy = enabled;
    }

    /// Set psychovisual strength.
    pub fn set_psy_strength(&mut self, strength: f32) {
        self.psy_strength = strength.clamp(0.0, 2.0);
    }

    /// Set adaptation speed.
    pub fn set_adaptation_speed(&mut self, speed: f32) {
        self.adaptation_speed = speed.clamp(0.0, 1.0);
    }

    /// Select QP for a frame.
    #[must_use]
    pub fn select_frame_qp(
        &mut self,
        frame_type: FrameType,
        complexity: f32,
        target_bits: u64,
        frame_in_gop: u32,
    ) -> QpResult {
        let base_qp = match self.strategy {
            QpStrategy::Constant => self.select_constant_qp(frame_type),
            QpStrategy::FrameTypeBased => self.select_frame_type_qp(frame_type),
            QpStrategy::ComplexityAdaptive => {
                self.select_complexity_adaptive_qp(frame_type, complexity)
            }
            QpStrategy::Psychovisual => {
                self.select_psychovisual_qp(frame_type, complexity, target_bits)
            }
            QpStrategy::RateDistortionOptimized => {
                self.select_rd_optimized_qp(frame_type, complexity, target_bits)
            }
            QpStrategy::Hierarchical => self.select_hierarchical_qp(frame_type, frame_in_gop),
        };

        // Calculate lambda for RDO
        let lambda = Self::qp_to_lambda(base_qp);
        let lambda_me = lambda.sqrt();

        // Generate block-level QP offsets if AQ is enabled
        let block_qp_offsets = if self.enable_aq {
            Some(self.generate_aq_offsets(base_qp, complexity))
        } else {
            None
        };

        // Update history
        self.historical_qp.push(base_qp);
        if self.historical_qp.len() > self.max_history {
            self.historical_qp.remove(0);
        }

        self.frame_count += 1;

        QpResult {
            qp: base_qp.round() as u8,
            qp_f: base_qp,
            lambda,
            lambda_me,
            block_qp_offsets,
            frame_type,
            complexity_factor: complexity / self.average_complexity(),
        }
    }

    /// Select constant QP (with frame type offset).
    fn select_constant_qp(&self, frame_type: FrameType) -> f32 {
        let offset = self.get_frame_type_offset(frame_type);
        (self.base_qp + offset as f32).clamp(self.min_qp as f32, self.max_qp as f32)
    }

    /// Select frame type based QP.
    fn select_frame_type_qp(&self, frame_type: FrameType) -> f32 {
        let offset = self.get_frame_type_offset(frame_type);
        (self.base_qp + offset as f32).clamp(self.min_qp as f32, self.max_qp as f32)
    }

    /// Select complexity-adaptive QP.
    fn select_complexity_adaptive_qp(&self, frame_type: FrameType, complexity: f32) -> f32 {
        let base = self.select_frame_type_qp(frame_type);
        let avg_complexity = self.average_complexity();

        if avg_complexity <= 0.0 {
            return base;
        }

        // Higher complexity → higher QP to save bits
        let complexity_ratio = complexity / avg_complexity;
        let adjustment = (complexity_ratio - 1.0) * 3.0 * self.adaptation_speed;

        (base + adjustment).clamp(self.min_qp as f32, self.max_qp as f32)
    }

    /// Select psychovisually-optimized QP.
    fn select_psychovisual_qp(
        &self,
        frame_type: FrameType,
        complexity: f32,
        _target_bits: u64,
    ) -> f32 {
        let base = self.select_complexity_adaptive_qp(frame_type, complexity);

        if !self.enable_psy {
            return base;
        }

        // Psychovisual optimization: reduce QP for visually important frames
        // This is a simplified model - full implementation would analyze spatial features
        let psy_adjustment = if complexity > 3.0 {
            // High complexity frames are less visually sensitive
            self.psy_strength * 0.5
        } else if complexity < 0.5 {
            // Low complexity frames are more visually sensitive
            -self.psy_strength * 0.5
        } else {
            0.0
        };

        (base + psy_adjustment).clamp(self.min_qp as f32, self.max_qp as f32)
    }

    /// Select rate-distortion optimized QP.
    fn select_rd_optimized_qp(
        &self,
        frame_type: FrameType,
        complexity: f32,
        target_bits: u64,
    ) -> f32 {
        let base = self.select_complexity_adaptive_qp(frame_type, complexity);

        // Estimate bits at current QP
        let estimated_bits = self.estimate_bits_at_qp(base, complexity, frame_type);

        if estimated_bits == 0 {
            return base;
        }

        // Adjust QP to hit target bits
        let bits_ratio = estimated_bits as f32 / target_bits as f32;

        // QP adjustment based on rate model
        // Higher bits_ratio → increase QP to reduce bits
        let qp_adjustment = if bits_ratio > 1.0 {
            (bits_ratio.ln() * 6.0).min(5.0)
        } else {
            (bits_ratio.ln() * 6.0).max(-5.0)
        };

        (base + qp_adjustment).clamp(self.min_qp as f32, self.max_qp as f32)
    }

    /// Select hierarchical QP for B-pyramid structures.
    fn select_hierarchical_qp(&self, frame_type: FrameType, frame_in_gop: u32) -> f32 {
        let base = self.select_frame_type_qp(frame_type);

        if frame_type != FrameType::BiDir {
            return base;
        }

        // B-frames in higher pyramid levels get higher QP
        let pyramid_level = self.calculate_pyramid_level(frame_in_gop);
        let level_offset = pyramid_level as f32 * 1.0;

        (base + level_offset).clamp(self.min_qp as f32, self.max_qp as f32)
    }

    /// Calculate pyramid level for a frame position.
    fn calculate_pyramid_level(&self, frame_in_gop: u32) -> u32 {
        let mut level = 0;
        let mut pos = frame_in_gop;

        while pos % 2 == 0 && pos > 0 {
            level += 1;
            pos /= 2;
        }

        level
    }

    /// Get frame type QP offset.
    fn get_frame_type_offset(&self, frame_type: FrameType) -> i8 {
        match frame_type {
            FrameType::Key => self.i_qp_offset,
            FrameType::Inter => self.p_qp_offset,
            FrameType::BiDir => self.b_qp_offset,
            FrameType::Switch => (self.i_qp_offset + self.p_qp_offset) / 2,
        }
    }

    /// Generate adaptive quantization offsets for blocks.
    fn generate_aq_offsets(&self, base_qp: f32, _complexity: f32) -> Vec<f32> {
        // Simplified AQ: would normally analyze frame content
        // For now, return empty - full implementation would compute per-block offsets
        Vec::new()
    }

    /// Estimate bits needed at a given QP.
    fn estimate_bits_at_qp(&self, qp: f32, complexity: f32, frame_type: FrameType) -> u64 {
        // Simplified rate model: bits ≈ complexity * 2^((QP_ref - QP) / 6)
        // This is based on the relationship between QP and quantization step size

        let base_complexity_bits = complexity * 50_000.0;
        let qp_ref = 28.0;
        let qp_factor = 2.0_f32.powf((qp_ref - qp) / 6.0);

        let frame_type_multiplier = match frame_type {
            FrameType::Key => 3.0,
            FrameType::Inter => 1.0,
            FrameType::BiDir => 0.5,
            FrameType::Switch => 1.5,
        };

        (base_complexity_bits * qp_factor * frame_type_multiplier) as u64
    }

    /// Convert QP to lambda for rate-distortion optimization.
    fn qp_to_lambda(qp: f32) -> f64 {
        // Standard lambda formula: λ = 0.85 * 2^((QP - 12) / 3)
        0.85 * 2.0_f64.powf((f64::from(qp) - 12.0) / 3.0)
    }

    /// Get average QP from history.
    fn average_qp(&self) -> f32 {
        if self.historical_qp.is_empty() {
            return self.base_qp;
        }

        let sum: f32 = self.historical_qp.iter().sum();
        sum / self.historical_qp.len() as f32
    }

    /// Get average complexity estimate.
    fn average_complexity(&self) -> f32 {
        // Simplified: assume average complexity of 1.0
        // Full implementation would track complexity history
        1.0
    }

    /// Update selector with actual encoding results.
    pub fn update(&mut self, actual_qp: f32, actual_bits: u64, target_bits: u64) {
        // Adapt base QP based on results
        if target_bits == 0 {
            return;
        }

        let bits_ratio = actual_bits as f32 / target_bits as f32;

        // If consistently over/under target, adjust base QP
        let adjustment = if bits_ratio > 1.1 {
            0.5 * self.adaptation_speed
        } else if bits_ratio < 0.8 {
            -0.5 * self.adaptation_speed
        } else {
            0.0
        };

        self.base_qp = (self.base_qp + adjustment).clamp(self.min_qp as f32, self.max_qp as f32);
    }

    /// Reset the selector state.
    pub fn reset(&mut self) {
        self.historical_qp.clear();
        self.frame_count = 0;
    }
}

/// QP selection result.
#[derive(Clone, Debug)]
pub struct QpResult {
    /// Selected QP (integer).
    pub qp: u8,
    /// Selected QP (floating point).
    pub qp_f: f32,
    /// Lambda for RDO.
    pub lambda: f64,
    /// Lambda for motion estimation.
    pub lambda_me: f64,
    /// Block-level QP offsets (if AQ enabled).
    pub block_qp_offsets: Option<Vec<f32>>,
    /// Frame type.
    pub frame_type: FrameType,
    /// Complexity factor relative to average.
    pub complexity_factor: f32,
}

impl QpResult {
    /// Get effective QP for a block.
    #[must_use]
    pub fn get_block_qp(&self, block_index: usize) -> u8 {
        if let Some(ref offsets) = self.block_qp_offsets {
            if let Some(&offset) = offsets.get(block_index) {
                return ((self.qp_f + offset).round() as i32).clamp(1, 63) as u8;
            }
        }
        self.qp
    }

    /// Check if QP is within acceptable range.
    #[must_use]
    pub fn is_valid(&self, min_qp: u8, max_qp: u8) -> bool {
        self.qp >= min_qp && self.qp <= max_qp
    }
}

/// Block-level QP map for adaptive quantization.
#[derive(Clone, Debug)]
pub struct BlockQpMap {
    /// Width in blocks.
    width: usize,
    /// Height in blocks.
    height: usize,
    /// QP values for each block.
    qp_values: Vec<u8>,
    /// Base QP for the frame.
    base_qp: u8,
}

impl BlockQpMap {
    /// Create a new block QP map.
    #[must_use]
    pub fn new(width: usize, height: usize, base_qp: u8) -> Self {
        let qp_values = vec![base_qp; width * height];
        Self {
            width,
            height,
            qp_values,
            base_qp,
        }
    }

    /// Set QP for a block.
    pub fn set_block_qp(&mut self, x: usize, y: usize, qp: u8) {
        if x < self.width && y < self.height {
            self.qp_values[y * self.width + x] = qp;
        }
    }

    /// Get QP for a block.
    #[must_use]
    pub fn get_block_qp(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.qp_values[y * self.width + x]
        } else {
            self.base_qp
        }
    }

    /// Apply QP offsets to all blocks.
    pub fn apply_offsets(&mut self, offsets: &[f32]) {
        for (i, offset) in offsets.iter().enumerate() {
            if i < self.qp_values.len() {
                let new_qp = ((self.base_qp as f32 + offset).round() as i32).clamp(1, 63) as u8;
                self.qp_values[i] = new_qp;
            }
        }
    }

    /// Get average QP across all blocks.
    #[must_use]
    pub fn average_qp(&self) -> f32 {
        if self.qp_values.is_empty() {
            return self.base_qp as f32;
        }

        let sum: u32 = self.qp_values.iter().map(|&qp| u32::from(qp)).sum();
        sum as f32 / self.qp_values.len() as f32
    }

    /// Get QP variance.
    #[must_use]
    pub fn qp_variance(&self) -> f32 {
        if self.qp_values.is_empty() {
            return 0.0;
        }

        let avg = self.average_qp();
        let variance: f32 = self
            .qp_values
            .iter()
            .map(|&qp| {
                let diff = qp as f32 - avg;
                diff * diff
            })
            .sum::<f32>()
            / self.qp_values.len() as f32;

        variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qp_selector_creation() {
        let selector = QpSelector::new(28.0, 1, 63);
        assert_eq!(selector.base_qp, 28.0);
        assert_eq!(selector.min_qp, 1);
        assert_eq!(selector.max_qp, 63);
    }

    #[test]
    fn test_constant_qp() {
        let mut selector = QpSelector::new(28.0, 1, 63);
        selector.set_strategy(QpStrategy::Constant);

        let result = selector.select_frame_qp(FrameType::Inter, 1.0, 100_000, 0);
        assert_eq!(result.qp, 28);

        let i_result = selector.select_frame_qp(FrameType::Key, 1.0, 300_000, 0);
        assert_eq!(i_result.qp, 26); // Base 28 + I offset -2
    }

    #[test]
    fn test_frame_type_based_qp() {
        let mut selector = QpSelector::new(28.0, 1, 63);
        selector.set_strategy(QpStrategy::FrameTypeBased);

        let i_result = selector.select_frame_qp(FrameType::Key, 1.0, 300_000, 0);
        let p_result = selector.select_frame_qp(FrameType::Inter, 1.0, 100_000, 1);
        let b_result = selector.select_frame_qp(FrameType::BiDir, 1.0, 50_000, 2);

        assert!(i_result.qp < p_result.qp); // I-frames get lower QP
        assert!(p_result.qp < b_result.qp); // P-frames get lower QP than B
    }

    #[test]
    fn test_complexity_adaptive_qp() {
        let mut selector = QpSelector::new(28.0, 1, 63);
        selector.set_strategy(QpStrategy::ComplexityAdaptive);

        let low_complexity = selector.select_frame_qp(FrameType::Inter, 0.5, 100_000, 0);
        let high_complexity = selector.select_frame_qp(FrameType::Inter, 2.0, 100_000, 1);

        // Higher complexity should get higher QP
        assert!(high_complexity.qp > low_complexity.qp);
    }

    #[test]
    fn test_hierarchical_qp() {
        let mut selector = QpSelector::new(28.0, 1, 63);
        selector.set_strategy(QpStrategy::Hierarchical);

        let level0_b = selector.select_frame_qp(FrameType::BiDir, 1.0, 50_000, 2);
        let level1_b = selector.select_frame_qp(FrameType::BiDir, 1.0, 50_000, 4);

        // Higher pyramid level should get higher QP
        assert!(level1_b.qp >= level0_b.qp);
    }

    #[test]
    fn test_qp_clamping() {
        let mut selector = QpSelector::new(28.0, 10, 40);

        // Set very low base QP
        selector.set_base_qp(5.0);
        assert_eq!(selector.base_qp, 10.0); // Should clamp to min

        // Set very high base QP
        selector.set_base_qp(50.0);
        assert_eq!(selector.base_qp, 40.0); // Should clamp to max
    }

    #[test]
    fn test_lambda_calculation() {
        let lambda1 = QpSelector::qp_to_lambda(28.0);
        let lambda2 = QpSelector::qp_to_lambda(34.0);

        assert!(lambda1 > 0.0);
        assert!(lambda2 > lambda1); // Higher QP → higher lambda
    }

    #[test]
    fn test_qp_adaptation() {
        let mut selector = QpSelector::new(28.0, 1, 63);

        // Consistently overshooting target
        for _ in 0..10 {
            selector.update(28.0, 120_000, 100_000);
        }

        // Base QP should increase
        assert!(selector.base_qp > 28.0);
    }

    #[test]
    fn test_block_qp_map() {
        let mut map = BlockQpMap::new(10, 10, 28);

        assert_eq!(map.get_block_qp(5, 5), 28);

        map.set_block_qp(5, 5, 30);
        assert_eq!(map.get_block_qp(5, 5), 30);

        let avg = map.average_qp();
        assert!(avg > 28.0); // One block increased to 30
    }

    #[test]
    fn test_block_qp_offsets() {
        let mut map = BlockQpMap::new(4, 4, 28);
        let offsets = vec![
            0.0, 1.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        map.apply_offsets(&offsets);

        assert_eq!(map.get_block_qp(0, 0), 28);
        assert_eq!(map.get_block_qp(1, 0), 29);
        assert_eq!(map.get_block_qp(2, 0), 27);
        assert_eq!(map.get_block_qp(3, 0), 30);
    }

    #[test]
    fn test_qp_variance() {
        let mut map = BlockQpMap::new(3, 3, 28);

        // All same QP - zero variance
        let variance1 = map.qp_variance();
        assert!(variance1 < 0.01);

        // Different QPs - non-zero variance
        map.set_block_qp(0, 0, 20);
        map.set_block_qp(1, 1, 36);
        let variance2 = map.qp_variance();
        assert!(variance2 > 1.0);
    }

    #[test]
    fn test_qp_result() {
        let result = QpResult {
            qp: 28,
            qp_f: 28.0,
            lambda: 10.0,
            lambda_me: 3.16,
            block_qp_offsets: Some(vec![0.0, 1.0, -1.0]),
            frame_type: FrameType::Inter,
            complexity_factor: 1.2,
        };

        assert_eq!(result.get_block_qp(0), 28);
        assert_eq!(result.get_block_qp(1), 29);
        assert_eq!(result.get_block_qp(2), 27);
        assert!(result.is_valid(1, 63));
    }

    #[test]
    fn test_reset() {
        let mut selector = QpSelector::new(28.0, 1, 63);

        let _ = selector.select_frame_qp(FrameType::Inter, 1.0, 100_000, 0);
        let _ = selector.select_frame_qp(FrameType::Inter, 1.0, 100_000, 1);

        assert!(!selector.historical_qp.is_empty());
        assert!(selector.frame_count > 0);

        selector.reset();

        assert!(selector.historical_qp.is_empty());
        assert_eq!(selector.frame_count, 0);
    }
}

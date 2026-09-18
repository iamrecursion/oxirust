// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rate control for Theora encoding.
//!
//! Implements bitrate control algorithms to maintain target bitrate
//! while maximizing visual quality.

use crate::error::CodecResult;
use std::collections::VecDeque;

/// Rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant Quality (CQ) - fixed quality parameter.
    ConstantQuality,
    /// Constant Bitrate (CBR) - target specific bitrate.
    ConstantBitrate,
    /// Variable Bitrate (VBR) - average bitrate with quality priority.
    VariableBitrate,
    /// Two-pass encoding.
    TwoPass,
}

/// Rate controller for Theora encoding.
pub struct RateController {
    /// Control mode.
    mode: RateControlMode,
    /// Target bitrate (bits per second).
    target_bitrate: u64,
    /// Target quality (0-63).
    target_quality: u8,
    /// Frame rate (frames per second).
    framerate: f64,
    /// Buffer size for CBR (in bits).
    buffer_size: u64,
    /// Current buffer fullness (in bits).
    buffer_fullness: i64,
    /// Frame statistics history.
    frame_history: VecDeque<FrameStats>,
    /// Maximum history size.
    history_size: usize,
    /// Current frame number.
    frame_number: u64,
}

/// Statistics for a single frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    /// Frame number.
    pub frame_number: u64,
    /// Frame type (true = keyframe).
    pub is_keyframe: bool,
    /// Actual size in bits.
    pub size_bits: u32,
    /// Quality parameter used.
    pub quality: u8,
    /// Complexity metric.
    pub complexity: f64,
}

impl RateController {
    /// Create a new rate controller.
    ///
    /// # Arguments
    ///
    /// * `mode` - Rate control mode
    /// * `target_bitrate` - Target bitrate in bits per second
    /// * `target_quality` - Target quality (0-63)
    /// * `framerate` - Frame rate in frames per second
    #[must_use]
    pub fn new(
        mode: RateControlMode,
        target_bitrate: u64,
        target_quality: u8,
        framerate: f64,
    ) -> Self {
        // Buffer size = 2 seconds of video
        let buffer_size = target_bitrate * 2;

        Self {
            mode,
            target_bitrate,
            target_quality: target_quality.min(63),
            framerate,
            buffer_size,
            buffer_fullness: (buffer_size / 2) as i64,
            frame_history: VecDeque::new(),
            history_size: 100,
            frame_number: 0,
        }
    }

    /// Get quality parameter for the next frame.
    ///
    /// # Arguments
    ///
    /// * `is_keyframe` - Whether this is a keyframe
    /// * `complexity` - Frame complexity estimate
    #[must_use]
    pub fn get_frame_quality(&mut self, is_keyframe: bool, complexity: f64) -> u8 {
        match self.mode {
            RateControlMode::ConstantQuality => self.target_quality,
            RateControlMode::ConstantBitrate => self.calculate_cbr_quality(is_keyframe, complexity),
            RateControlMode::VariableBitrate => self.calculate_vbr_quality(is_keyframe, complexity),
            RateControlMode::TwoPass => self.calculate_twopass_quality(is_keyframe),
        }
    }

    /// Update statistics after encoding a frame.
    ///
    /// # Arguments
    ///
    /// * `is_keyframe` - Whether this was a keyframe
    /// * `size_bits` - Actual frame size in bits
    /// * `quality` - Quality parameter used
    /// * `complexity` - Frame complexity
    pub fn update_stats(
        &mut self,
        is_keyframe: bool,
        size_bits: u32,
        quality: u8,
        complexity: f64,
    ) {
        let stats = FrameStats {
            frame_number: self.frame_number,
            is_keyframe,
            size_bits,
            quality,
            complexity,
        };

        self.frame_history.push_back(stats);
        if self.frame_history.len() > self.history_size {
            self.frame_history.pop_front();
        }

        // Update buffer fullness for CBR
        if self.mode == RateControlMode::ConstantBitrate {
            let target_frame_bits = (self.target_bitrate as f64 / self.framerate) as i64;
            self.buffer_fullness += target_frame_bits - i64::from(size_bits);
            self.buffer_fullness = self.buffer_fullness.clamp(0, self.buffer_size as i64);
        }

        self.frame_number += 1;
    }

    /// Calculate quality for CBR mode.
    fn calculate_cbr_quality(&self, is_keyframe: bool, complexity: f64) -> u8 {
        let target_frame_bits = self.target_bitrate as f64 / self.framerate;

        // Buffer-based adjustment
        let buffer_ratio = self.buffer_fullness as f64 / self.buffer_size as f64;
        let buffer_adjustment = if buffer_ratio > 0.75 {
            // Buffer filling up, reduce quality to reduce bitrate
            5
        } else if buffer_ratio < 0.25 {
            // Buffer emptying, increase quality
            -5i8
        } else {
            0
        };

        // Complexity adjustment
        let avg_complexity = self.get_average_complexity();
        let complexity_ratio = if avg_complexity > 0.0 {
            complexity / avg_complexity
        } else {
            1.0
        };

        let complexity_adjustment = if complexity_ratio > 1.5 {
            5 // More complex than average, reduce quality
        } else if complexity_ratio < 0.5 {
            -5i8 // Less complex than average, increase quality
        } else {
            0
        };

        // Keyframe adjustment
        let keyframe_adjustment = if is_keyframe { 5i8 } else { 0 };

        let quality = self.target_quality as i8
            + buffer_adjustment
            + complexity_adjustment
            + keyframe_adjustment;

        quality.clamp(0, 63) as u8
    }

    /// Calculate quality for VBR mode.
    fn calculate_vbr_quality(&self, is_keyframe: bool, complexity: f64) -> u8 {
        // In VBR, we prioritize quality but adjust based on complexity
        let avg_complexity = self.get_average_complexity();
        let complexity_ratio = if avg_complexity > 0.0 {
            complexity / avg_complexity
        } else {
            1.0
        };

        let adjustment = if complexity_ratio > 1.5 {
            3 // Slightly reduce quality for complex frames
        } else if complexity_ratio < 0.5 {
            -3i8 // Slightly increase quality for simple frames
        } else {
            0
        };

        let keyframe_adjustment = if is_keyframe { 3i8 } else { 0 };

        let quality = self.target_quality as i8 + adjustment + keyframe_adjustment;
        quality.clamp(0, 63) as u8
    }

    /// Calculate quality for two-pass mode (first pass).
    fn calculate_twopass_quality(&self, _is_keyframe: bool) -> u8 {
        // In first pass, use constant quality
        self.target_quality
    }

    /// Get average complexity from history.
    fn get_average_complexity(&self) -> f64 {
        if self.frame_history.is_empty() {
            return 1.0;
        }

        let sum: f64 = self.frame_history.iter().map(|s| s.complexity).sum();
        sum / self.frame_history.len() as f64
    }

    /// Get average bitrate over recent history.
    #[must_use]
    pub fn get_average_bitrate(&self) -> f64 {
        if self.frame_history.is_empty() {
            return 0.0;
        }

        let total_bits: u64 = self
            .frame_history
            .iter()
            .map(|s| u64::from(s.size_bits))
            .sum();
        let duration = self.frame_history.len() as f64 / self.framerate;

        if duration > 0.0 {
            total_bits as f64 / duration
        } else {
            0.0
        }
    }

    /// Get current buffer fullness ratio (0.0 to 1.0).
    #[must_use]
    pub fn buffer_fullness_ratio(&self) -> f64 {
        self.buffer_fullness as f64 / self.buffer_size as f64
    }

    /// Reset the rate controller state.
    pub fn reset(&mut self) {
        self.frame_history.clear();
        self.buffer_fullness = (self.buffer_size / 2) as i64;
        self.frame_number = 0;
    }
}

/// Estimate frame complexity for rate control.
///
/// # Arguments
///
/// * `y_plane` - Y plane data
/// * `width` - Frame width
/// * `height` - Frame height
/// * `stride` - Plane stride
#[must_use]
pub fn estimate_frame_complexity(
    y_plane: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> f64 {
    if width == 0 || height == 0 {
        return 0.0;
    }

    let mut variance_sum = 0u64;
    let block_size = 16;

    // Calculate variance for each 16x16 macroblock
    for by in (0..height).step_by(block_size) {
        for bx in (0..width).step_by(block_size) {
            let variance = calculate_block_variance(y_plane, stride, bx, by, block_size);
            variance_sum += u64::from(variance);
        }
    }

    let num_blocks =
        ((width + block_size - 1) / block_size) * ((height + block_size - 1) / block_size);
    if num_blocks > 0 {
        (variance_sum / num_blocks as u64) as f64
    } else {
        0.0
    }
}

/// Calculate variance for a block.
fn calculate_block_variance(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    block_size: usize,
) -> u32 {
    let mut sum = 0u32;
    let mut sum_sq = 0u32;
    let mut count = 0u32;

    for dy in 0..block_size {
        if y + dy >= plane.len() / stride {
            break;
        }
        for dx in 0..block_size {
            if x + dx >= stride {
                break;
            }
            let offset = (y + dy) * stride + x + dx;
            if offset >= plane.len() {
                break;
            }

            let pixel = u32::from(plane[offset]);
            sum += pixel;
            sum_sq += pixel * pixel;
            count += 1;
        }
    }

    if count == 0 {
        return 0;
    }

    let mean = sum / count;
    let variance = (sum_sq / count).saturating_sub(mean * mean);
    variance
}

/// Quantization parameter mapper.
///
/// Maps quality values to quantization parameters with perceptual weighting.
pub struct QuantMapper {
    /// Quality to QP lookup table.
    quality_to_qp: [u8; 64],
}

impl QuantMapper {
    /// Create a new quantization mapper.
    #[must_use]
    pub fn new() -> Self {
        let mut quality_to_qp = [0u8; 64];

        // Map quality (0-63) to QP with non-linear scaling
        for q in 0..64 {
            let qp = if q < 20 {
                // Very high quality range
                10 + q / 2
            } else if q < 40 {
                // Medium quality range
                20 + (q - 20)
            } else {
                // Lower quality range
                40 + (q - 40) * 2
            };
            quality_to_qp[q] = qp.min(255) as u8;
        }

        Self { quality_to_qp }
    }

    /// Get quantization parameter for a quality value.
    #[must_use]
    pub fn quality_to_qp(&self, quality: u8) -> u8 {
        let quality = quality.min(63);
        self.quality_to_qp[quality as usize]
    }

    /// Get quality value from quantization parameter.
    #[must_use]
    pub fn qp_to_quality(&self, qp: u8) -> u8 {
        // Find closest quality value
        for q in 0..64 {
            if self.quality_to_qp[q] >= qp {
                return q as u8;
            }
        }
        63
    }
}

impl Default for QuantMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive quantization for macroblocks.
///
/// Adjusts quantization based on local image characteristics.
pub struct AdaptiveQuantization {
    /// Base quality parameter.
    base_quality: u8,
    /// Strength of adaptation (0.0 to 1.0).
    strength: f32,
}

impl AdaptiveQuantization {
    /// Create a new adaptive quantization controller.
    #[must_use]
    pub const fn new(base_quality: u8, strength: f32) -> Self {
        Self {
            base_quality,
            strength,
        }
    }

    /// Get adjusted quality for a macroblock.
    ///
    /// # Arguments
    ///
    /// * `plane` - Y plane data
    /// * `stride` - Plane stride
    /// * `mb_x` - Macroblock X coordinate
    /// * `mb_y` - Macroblock Y coordinate
    #[must_use]
    pub fn get_mb_quality(&self, plane: &[u8], stride: usize, mb_x: usize, mb_y: usize) -> u8 {
        let x = mb_x * 16;
        let y = mb_y * 16;

        // Calculate macroblock variance
        let variance = calculate_block_variance(plane, stride, x, y, 16);

        // Adjust quality based on variance
        let adjustment = if variance < 100 {
            // Low variance (smooth): increase quality (lower QP)
            -(self.strength * 5.0) as i8
        } else if variance > 1000 {
            // High variance (textured): decrease quality (higher QP)
            (self.strength * 5.0) as i8
        } else {
            0
        };

        let quality = self.base_quality as i8 + adjustment;
        quality.clamp(0, 63) as u8
    }
}

/// Bitrate allocation for different frame types.
#[derive(Debug, Clone, Copy)]
pub struct BitrateAllocation {
    /// Intra frame multiplier.
    pub intra_factor: f32,
    /// Inter frame multiplier.
    pub inter_factor: f32,
}

impl Default for BitrateAllocation {
    fn default() -> Self {
        Self {
            intra_factor: 3.0, // Keyframes get 3x bitrate
            inter_factor: 1.0,
        }
    }
}

impl BitrateAllocation {
    /// Get target bits for a frame type.
    #[must_use]
    pub fn get_target_bits(&self, base_bits: f64, is_keyframe: bool) -> u32 {
        let factor = if is_keyframe {
            self.intra_factor
        } else {
            self.inter_factor
        };
        (base_bits * f64::from(factor)) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_controller_creation() {
        let rc = RateController::new(RateControlMode::ConstantBitrate, 2_000_000, 30, 30.0);
        assert_eq!(rc.target_bitrate, 2_000_000);
        assert_eq!(rc.target_quality, 30);
    }

    #[test]
    fn test_constant_quality_mode() {
        let mut rc = RateController::new(RateControlMode::ConstantQuality, 2_000_000, 30, 30.0);
        let quality = rc.get_frame_quality(false, 100.0);
        assert_eq!(quality, 30);
    }

    #[test]
    fn test_frame_stats_update() {
        let mut rc = RateController::new(RateControlMode::ConstantBitrate, 2_000_000, 30, 30.0);

        rc.update_stats(true, 10000, 30, 100.0);
        assert_eq!(rc.frame_history.len(), 1);
        assert_eq!(rc.frame_number, 1);
    }

    #[test]
    fn test_complexity_estimation() {
        let plane = vec![128u8; 640 * 480];
        let complexity = estimate_frame_complexity(&plane, 640, 480, 640);
        assert_eq!(complexity, 0.0); // Uniform plane has zero variance
    }

    #[test]
    fn test_quant_mapper() {
        let mapper = QuantMapper::new();
        let qp = mapper.quality_to_qp(30);
        let quality = mapper.qp_to_quality(qp);
        assert!((quality as i16 - 30).abs() <= 2); // Should be close
    }

    #[test]
    fn test_adaptive_quantization() {
        let aq = AdaptiveQuantization::new(30, 0.5);
        let plane = vec![128u8; 640 * 480];
        let quality = aq.get_mb_quality(&plane, 640, 0, 0);
        assert!(quality < 30); // Smooth area should get better quality
    }

    #[test]
    fn test_bitrate_allocation() {
        let alloc = BitrateAllocation::default();
        let base_bits = 10000.0;
        let intra_bits = alloc.get_target_bits(base_bits, true);
        let inter_bits = alloc.get_target_bits(base_bits, false);
        assert!(intra_bits > inter_bits);
    }
}

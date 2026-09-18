//! Sub-pixel motion estimation refinement.
//!
//! This module provides:
//! - Half-pel and quarter-pel interpolation filters
//! - Sub-pixel refinement search
//! - SATD (Sum of Absolute Transformed Differences) computation
//! - Hadamard transform for SATD
//!
//! Sub-pixel motion estimation significantly improves prediction quality
//! at the cost of additional computation for interpolation.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::similar_names)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::let_and_return)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::unused_self)]

use super::types::{BlockMatch, BlockSize, MotionVector, MvCost, MvPrecision};

/// 6-tap filter coefficients for half-pel interpolation.
/// Used in H.264/AVC and similar codecs.
pub const HALF_PEL_FILTER_6TAP: [i16; 6] = [1, -5, 20, 20, -5, 1];

/// 8-tap filter coefficients for quarter-pel interpolation.
pub const QUARTER_PEL_FILTER_8TAP: [i16; 8] = [-1, 4, -10, 58, 17, -5, 1, 0];

/// Bilinear filter for simple half-pel.
pub const BILINEAR_HALF: [i16; 2] = [1, 1];

/// Configuration for sub-pixel refinement.
#[derive(Clone, Debug)]
pub struct SubpelConfig {
    /// Target precision.
    pub precision: MvPrecision,
    /// Use SATD instead of SAD.
    pub use_satd: bool,
    /// MV cost for RD optimization.
    pub mv_cost: MvCost,
    /// Filter type for half-pel.
    pub half_pel_filter: HalfPelFilter,
    /// Filter type for quarter-pel.
    pub quarter_pel_filter: QuarterPelFilter,
}

impl Default for SubpelConfig {
    fn default() -> Self {
        Self {
            precision: MvPrecision::QuarterPel,
            use_satd: true,
            mv_cost: MvCost::default(),
            half_pel_filter: HalfPelFilter::Sixtap,
            quarter_pel_filter: QuarterPelFilter::Bilinear,
        }
    }
}

/// Half-pel interpolation filter type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HalfPelFilter {
    /// Bilinear (2-tap) filter.
    Bilinear,
    /// 6-tap filter (H.264 style).
    #[default]
    Sixtap,
}

/// Quarter-pel interpolation filter type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuarterPelFilter {
    /// Bilinear interpolation.
    #[default]
    Bilinear,
    /// 8-tap filter.
    Eighttap,
}

/// Half-pel interpolation at a single position.
#[derive(Clone, Copy, Debug, Default)]
pub struct HalfPelInterpolator;

impl HalfPelInterpolator {
    /// Creates a new interpolator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Interpolates a half-pel position using bilinear filter.
    #[must_use]
    pub fn bilinear(a: u8, b: u8) -> u8 {
        (u16::from(a) + u16::from(b)).div_ceil(2) as u8
    }

    /// Interpolates a half-pel position using 6-tap filter.
    ///
    /// Input: 6 samples centered on the interpolation position.
    #[must_use]
    pub fn sixtap(samples: &[u8; 6]) -> u8 {
        let mut sum: i32 = 0;
        for (i, &coef) in HALF_PEL_FILTER_6TAP.iter().enumerate() {
            sum += i32::from(coef) * i32::from(samples[i]);
        }
        // Round and normalize (divide by 32)
        ((sum + 16) >> 5).clamp(0, 255) as u8
    }

    /// Interpolates horizontal half-pel for a row.
    pub fn interpolate_h(src: &[u8], stride: usize, dst: &mut [u8], width: usize, height: usize) {
        for y in 0..height {
            let row_offset = y * stride;
            for x in 0..width {
                let src_x = row_offset + x;
                if src_x + 1 < src.len() {
                    dst[y * width + x] = Self::bilinear(src[src_x], src[src_x + 1]);
                }
            }
        }
    }

    /// Interpolates vertical half-pel for a column.
    pub fn interpolate_v(src: &[u8], stride: usize, dst: &mut [u8], width: usize, height: usize) {
        for y in 0..height {
            for x in 0..width {
                let src_idx = y * stride + x;
                let src_idx_next = (y + 1) * stride + x;
                if src_idx_next < src.len() {
                    dst[y * width + x] = Self::bilinear(src[src_idx], src[src_idx_next]);
                }
            }
        }
    }

    /// Interpolates diagonal (HV) half-pel.
    pub fn interpolate_hv(src: &[u8], stride: usize, dst: &mut [u8], width: usize, height: usize) {
        for y in 0..height {
            for x in 0..width {
                let p00 = src.get(y * stride + x).copied().unwrap_or(0);
                let p01 = src.get(y * stride + x + 1).copied().unwrap_or(0);
                let p10 = src.get((y + 1) * stride + x).copied().unwrap_or(0);
                let p11 = src.get((y + 1) * stride + x + 1).copied().unwrap_or(0);

                let sum = u16::from(p00) + u16::from(p01) + u16::from(p10) + u16::from(p11);
                dst[y * width + x] = ((sum + 2) / 4) as u8;
            }
        }
    }
}

/// Quarter-pel interpolation.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuarterPelInterpolator;

impl QuarterPelInterpolator {
    /// Creates a new interpolator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Bilinear interpolation for quarter-pel.
    #[must_use]
    pub fn bilinear(a: u8, b: u8, weight_a: u8, weight_b: u8) -> u8 {
        let wa = u16::from(weight_a);
        let wb = u16::from(weight_b);
        let total = wa + wb;
        ((u16::from(a) * wa + u16::from(b) * wb + total / 2) / total) as u8
    }

    /// Interpolates quarter-pel from full-pel and half-pel samples.
    #[must_use]
    pub fn interpolate_qpel(full: u8, half: u8) -> u8 {
        Self::bilinear(full, half, 1, 1)
    }
}

/// Hadamard transform for SATD computation.
#[derive(Clone, Copy, Debug, Default)]
pub struct HadamardTransform;

impl HadamardTransform {
    /// 4x4 Hadamard transform (in-place).
    pub fn hadamard_4x4(block: &mut [[i16; 4]; 4]) {
        // Horizontal transform
        for row in block.iter_mut() {
            let a = row[0] + row[1];
            let b = row[2] + row[3];
            let c = row[0] - row[1];
            let d = row[2] - row[3];

            row[0] = a + b;
            row[1] = c + d;
            row[2] = a - b;
            row[3] = c - d;
        }

        // Vertical transform
        for col in 0..4 {
            let a = block[0][col] + block[1][col];
            let b = block[2][col] + block[3][col];
            let c = block[0][col] - block[1][col];
            let d = block[2][col] - block[3][col];

            block[0][col] = a + b;
            block[1][col] = c + d;
            block[2][col] = a - b;
            block[3][col] = c - d;
        }
    }

    /// 8x8 Hadamard transform using two 4x4 transforms.
    pub fn hadamard_8x8(block: &mut [[i16; 8]; 8]) {
        // Process as four 4x4 blocks
        let mut sub = [[0i16; 4]; 4];

        // Top-left 4x4
        for i in 0..4 {
            for j in 0..4 {
                sub[i][j] = block[i][j];
            }
        }
        Self::hadamard_4x4(&mut sub);
        for i in 0..4 {
            for j in 0..4 {
                block[i][j] = sub[i][j];
            }
        }

        // Top-right 4x4
        for i in 0..4 {
            for j in 0..4 {
                sub[i][j] = block[i][j + 4];
            }
        }
        Self::hadamard_4x4(&mut sub);
        for i in 0..4 {
            for j in 0..4 {
                block[i][j + 4] = sub[i][j];
            }
        }

        // Bottom-left 4x4
        for i in 0..4 {
            for j in 0..4 {
                sub[i][j] = block[i + 4][j];
            }
        }
        Self::hadamard_4x4(&mut sub);
        for i in 0..4 {
            for j in 0..4 {
                block[i + 4][j] = sub[i][j];
            }
        }

        // Bottom-right 4x4
        for i in 0..4 {
            for j in 0..4 {
                sub[i][j] = block[i + 4][j + 4];
            }
        }
        Self::hadamard_4x4(&mut sub);
        for i in 0..4 {
            for j in 0..4 {
                block[i + 4][j + 4] = sub[i][j];
            }
        }
    }
}

/// SATD (Sum of Absolute Transformed Differences) calculator.
#[derive(Clone, Copy, Debug, Default)]
pub struct SatdCalculator;

impl SatdCalculator {
    /// Creates a new calculator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Calculates SATD for a 4x4 block.
    #[must_use]
    pub fn satd_4x4(src: &[u8], src_stride: usize, ref_block: &[u8], ref_stride: usize) -> u32 {
        // Calculate differences
        let mut diff = [[0i16; 4]; 4];
        for row in 0..4 {
            let src_offset = row * src_stride;
            let ref_offset = row * ref_stride;
            for col in 0..4 {
                if src_offset + col < src.len() && ref_offset + col < ref_block.len() {
                    diff[row][col] =
                        i16::from(src[src_offset + col]) - i16::from(ref_block[ref_offset + col]);
                }
            }
        }

        // Apply Hadamard transform
        HadamardTransform::hadamard_4x4(&mut diff);

        // Sum absolute values
        let mut sum = 0u32;
        for row in &diff {
            for &val in row {
                sum += u32::from(val.unsigned_abs());
            }
        }

        // Normalize (divide by 2 as Hadamard doubles values)
        (sum + 1) >> 1
    }

    /// Calculates SATD for an 8x8 block.
    #[must_use]
    pub fn satd_8x8(src: &[u8], src_stride: usize, ref_block: &[u8], ref_stride: usize) -> u32 {
        let mut total = 0u32;

        // Process as four 4x4 blocks
        for block_row in 0..2 {
            for block_col in 0..2 {
                let src_offset = block_row * 4 * src_stride + block_col * 4;
                let ref_offset = block_row * 4 * ref_stride + block_col * 4;

                if src_offset < src.len() && ref_offset < ref_block.len() {
                    total += Self::satd_4x4(
                        &src[src_offset..],
                        src_stride,
                        &ref_block[ref_offset..],
                        ref_stride,
                    );
                }
            }
        }

        total
    }

    /// Calculates SATD for a 16x16 block.
    #[must_use]
    pub fn satd_16x16(src: &[u8], src_stride: usize, ref_block: &[u8], ref_stride: usize) -> u32 {
        let mut total = 0u32;

        // Process as four 8x8 blocks
        for block_row in 0..2 {
            for block_col in 0..2 {
                let src_offset = block_row * 8 * src_stride + block_col * 8;
                let ref_offset = block_row * 8 * ref_stride + block_col * 8;

                if src_offset < src.len() && ref_offset < ref_block.len() {
                    total += Self::satd_8x8(
                        &src[src_offset..],
                        src_stride,
                        &ref_block[ref_offset..],
                        ref_stride,
                    );
                }
            }
        }

        total
    }

    /// Calculates SATD for arbitrary block size.
    #[must_use]
    pub fn satd(
        src: &[u8],
        src_stride: usize,
        ref_block: &[u8],
        ref_stride: usize,
        block_size: BlockSize,
    ) -> u32 {
        match block_size {
            BlockSize::Block4x4 => Self::satd_4x4(src, src_stride, ref_block, ref_stride),
            BlockSize::Block8x8 => Self::satd_8x8(src, src_stride, ref_block, ref_stride),
            BlockSize::Block16x16 => Self::satd_16x16(src, src_stride, ref_block, ref_stride),
            _ => {
                // For other sizes, use 4x4 SATD blocks
                let width = block_size.width();
                let height = block_size.height();
                let mut total = 0u32;

                for by in (0..height).step_by(4) {
                    for bx in (0..width).step_by(4) {
                        let src_offset = by * src_stride + bx;
                        let ref_offset = by * ref_stride + bx;

                        if src_offset < src.len() && ref_offset < ref_block.len() {
                            total += Self::satd_4x4(
                                &src[src_offset..],
                                src_stride,
                                &ref_block[ref_offset..],
                                ref_stride,
                            );
                        }
                    }
                }

                total
            }
        }
    }
}

/// Sub-pixel refinement search.
#[derive(Clone, Debug)]
pub struct SubpelRefiner {
    /// Configuration.
    config: SubpelConfig,
    /// Interpolation buffer.
    interp_buffer: Vec<u8>,
    /// SATD calculator.
    satd: SatdCalculator,
}

impl Default for SubpelRefiner {
    fn default() -> Self {
        Self::new()
    }
}

impl SubpelRefiner {
    /// Half-pel search pattern offsets (in 1/8 pel units).
    const HALF_PEL_PATTERN: [(i32, i32); 8] = [
        (0, -4),  // Top
        (-4, 0),  // Left
        (4, 0),   // Right
        (0, 4),   // Bottom
        (-4, -4), // Top-left
        (4, -4),  // Top-right
        (-4, 4),  // Bottom-left
        (4, 4),   // Bottom-right
    ];

    /// Quarter-pel search pattern offsets (in 1/8 pel units).
    const QUARTER_PEL_PATTERN: [(i32, i32); 8] = [
        (0, -2),  // Top
        (-2, 0),  // Left
        (2, 0),   // Right
        (0, 2),   // Bottom
        (-2, -2), // Top-left
        (2, -2),  // Top-right
        (-2, 2),  // Bottom-left
        (2, 2),   // Bottom-right
    ];

    /// Creates a new sub-pixel refiner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SubpelConfig::default(),
            interp_buffer: vec![0u8; 256 * 256],
            satd: SatdCalculator::new(),
        }
    }

    /// Sets the configuration.
    #[must_use]
    pub fn with_config(mut self, config: SubpelConfig) -> Self {
        self.config = config;
        self
    }

    /// Refines a full-pel motion vector to sub-pixel precision.
    pub fn refine(
        &mut self,
        src: &[u8],
        src_stride: usize,
        reference: &[u8],
        ref_stride: usize,
        block_size: BlockSize,
        ref_x: usize,
        ref_y: usize,
        ref_width: usize,
        ref_height: usize,
        mv: MotionVector,
    ) -> BlockMatch {
        let mut best_mv = mv;
        let mut best_cost = self.calculate_cost(
            src, src_stride, reference, ref_stride, block_size, ref_x, ref_y, ref_width,
            ref_height, &mv,
        );

        // Half-pel refinement
        if self.config.precision as u8 >= MvPrecision::HalfPel as u8 {
            let (new_mv, new_cost) = self.search_subpel(
                src,
                src_stride,
                reference,
                ref_stride,
                block_size,
                ref_x,
                ref_y,
                ref_width,
                ref_height,
                best_mv,
                &Self::HALF_PEL_PATTERN,
            );
            if new_cost < best_cost {
                best_mv = new_mv;
                best_cost = new_cost;
            }
        }

        // Quarter-pel refinement
        if self.config.precision as u8 >= MvPrecision::QuarterPel as u8 {
            let (new_mv, new_cost) = self.search_subpel(
                src,
                src_stride,
                reference,
                ref_stride,
                block_size,
                ref_x,
                ref_y,
                ref_width,
                ref_height,
                best_mv,
                &Self::QUARTER_PEL_PATTERN,
            );
            if new_cost < best_cost {
                best_mv = new_mv;
                best_cost = new_cost;
            }
        }

        // Calculate final SAD for the best MV
        let sad = self.calculate_distortion(
            src, src_stride, reference, ref_stride, block_size, ref_x, ref_y, ref_width,
            ref_height, &best_mv,
        );

        BlockMatch::new(best_mv, sad, best_cost)
    }

    /// Searches sub-pixel positions around a center.
    fn search_subpel(
        &mut self,
        src: &[u8],
        src_stride: usize,
        reference: &[u8],
        ref_stride: usize,
        block_size: BlockSize,
        ref_x: usize,
        ref_y: usize,
        ref_width: usize,
        ref_height: usize,
        center: MotionVector,
        pattern: &[(i32, i32)],
    ) -> (MotionVector, u32) {
        let mut best_mv = center;
        let mut best_cost = self.calculate_cost(
            src, src_stride, reference, ref_stride, block_size, ref_x, ref_y, ref_width,
            ref_height, &center,
        );

        for &(dx, dy) in pattern {
            let candidate = MotionVector::new(center.dx + dx, center.dy + dy);

            let cost = self.calculate_cost(
                src, src_stride, reference, ref_stride, block_size, ref_x, ref_y, ref_width,
                ref_height, &candidate,
            );

            if cost < best_cost {
                best_mv = candidate;
                best_cost = cost;
            }
        }

        (best_mv, best_cost)
    }

    /// Calculates RD cost for a motion vector.
    fn calculate_cost(
        &mut self,
        src: &[u8],
        src_stride: usize,
        reference: &[u8],
        ref_stride: usize,
        block_size: BlockSize,
        ref_x: usize,
        ref_y: usize,
        ref_width: usize,
        ref_height: usize,
        mv: &MotionVector,
    ) -> u32 {
        let distortion = self.calculate_distortion(
            src, src_stride, reference, ref_stride, block_size, ref_x, ref_y, ref_width,
            ref_height, mv,
        );

        self.config.mv_cost.rd_cost(mv, distortion)
    }

    /// Calculates distortion (SAD or SATD) for a motion vector.
    #[allow(clippy::too_many_arguments)]
    fn calculate_distortion(
        &mut self,
        src: &[u8],
        src_stride: usize,
        reference: &[u8],
        ref_stride: usize,
        block_size: BlockSize,
        ref_x: usize,
        ref_y: usize,
        ref_width: usize,
        ref_height: usize,
        mv: &MotionVector,
    ) -> u32 {
        let width = block_size.width();
        let height = block_size.height();

        // Get interpolated reference block
        if !self.interpolate_block(
            reference, ref_stride, ref_x, ref_y, ref_width, ref_height, mv, width, height,
        ) {
            return u32::MAX;
        }

        // Calculate distortion
        if self.config.use_satd {
            SatdCalculator::satd(src, src_stride, &self.interp_buffer, width, block_size)
        } else {
            self.calculate_sad(src, src_stride, width, height)
        }
    }

    /// Interpolates a block at sub-pixel position.
    #[allow(clippy::too_many_arguments)]
    fn interpolate_block(
        &mut self,
        reference: &[u8],
        ref_stride: usize,
        ref_x: usize,
        ref_y: usize,
        ref_width: usize,
        ref_height: usize,
        mv: &MotionVector,
        width: usize,
        height: usize,
    ) -> bool {
        let full_x = ref_x as i32 + mv.full_pel_x();
        let full_y = ref_y as i32 + mv.full_pel_y();
        let frac_x = mv.frac_x();
        let frac_y = mv.frac_y();

        // Bounds check
        if full_x < 0 || full_y < 0 {
            return false;
        }
        let full_x = full_x as usize;
        let full_y = full_y as usize;

        if full_x + width > ref_width || full_y + height > ref_height {
            return false;
        }

        // No sub-pixel interpolation needed
        if frac_x == 0 && frac_y == 0 {
            for row in 0..height {
                let src_offset = (full_y + row) * ref_stride + full_x;
                let dst_offset = row * width;
                if src_offset + width <= reference.len() {
                    self.interp_buffer[dst_offset..dst_offset + width]
                        .copy_from_slice(&reference[src_offset..src_offset + width]);
                }
            }
            return true;
        }

        // Half-pel interpolation
        let hx = usize::from(frac_x >= 4);
        let hy = usize::from(frac_y >= 4);

        for row in 0..height {
            for col in 0..width {
                let x0 = full_x + col;
                let y0 = full_y + row;
                let x1 = (x0 + hx).min(ref_width - 1);
                let y1 = (y0 + hy).min(ref_height - 1);

                let p00 = reference[y0 * ref_stride + x0];
                let p01 = reference[y0 * ref_stride + x1];
                let p10 = reference[y1 * ref_stride + x0];
                let p11 = reference[y1 * ref_stride + x1];

                // Bilinear interpolation weights
                let wx = (frac_x & 3) as u16;
                let wy = (frac_y & 3) as u16;
                let wx_inv = 4 - wx;
                let wy_inv = 4 - wy;

                let val = (u16::from(p00) * wx_inv * wy_inv
                    + u16::from(p01) * wx * wy_inv
                    + u16::from(p10) * wx_inv * wy
                    + u16::from(p11) * wx * wy
                    + 8)
                    / 16;

                self.interp_buffer[row * width + col] = val as u8;
            }
        }

        true
    }

    /// Calculates SAD from interpolation buffer.
    fn calculate_sad(&self, src: &[u8], src_stride: usize, width: usize, height: usize) -> u32 {
        let mut sad = 0u32;
        for row in 0..height {
            let src_offset = row * src_stride;
            let ref_offset = row * width;
            for col in 0..width {
                if src_offset + col < src.len() {
                    let diff = i32::from(src[src_offset + col])
                        - i32::from(self.interp_buffer[ref_offset + col]);
                    sad += diff.unsigned_abs();
                }
            }
        }
        sad
    }
}

/// Sub-pixel search patterns.
pub struct SubpelPatterns;

impl SubpelPatterns {
    /// Square pattern for exhaustive sub-pel search.
    pub const SQUARE_9: [(i32, i32); 9] = [
        (0, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    /// Diamond pattern for fast sub-pel search.
    pub const DIAMOND_5: [(i32, i32); 5] = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)];

    /// Extended pattern for thorough search.
    pub const EXTENDED_25: [(i32, i32); 25] = [
        (0, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (-2, -2),
        (-1, -2),
        (0, -2),
        (1, -2),
        (2, -2),
        (-2, -1),
        (2, -1),
        (-2, 0),
        (2, 0),
        (-2, 1),
        (2, 1),
        (-2, 2),
        (-1, 2),
        (0, 2),
        (1, 2),
        (2, 2),
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_pel_bilinear() {
        assert_eq!(HalfPelInterpolator::bilinear(100, 200), 150);
        assert_eq!(HalfPelInterpolator::bilinear(0, 255), 128);
        assert_eq!(HalfPelInterpolator::bilinear(100, 100), 100);
    }

    #[test]
    fn test_half_pel_sixtap() {
        let samples = [100u8, 100, 100, 100, 100, 100];
        let result = HalfPelInterpolator::sixtap(&samples);
        // Constant input should give approximately the same output
        assert!((result as i32 - 100).abs() < 5);
    }

    #[test]
    fn test_quarter_pel_bilinear() {
        assert_eq!(QuarterPelInterpolator::bilinear(100, 200, 3, 1), 125);
        assert_eq!(QuarterPelInterpolator::bilinear(100, 200, 1, 3), 175);
        assert_eq!(QuarterPelInterpolator::bilinear(100, 200, 1, 1), 150);
    }

    #[test]
    fn test_hadamard_4x4() {
        let mut block = [[0i16; 4]; 4];
        block[0] = [1, 2, 3, 4];
        block[1] = [5, 6, 7, 8];
        block[2] = [9, 10, 11, 12];
        block[3] = [13, 14, 15, 16];

        HadamardTransform::hadamard_4x4(&mut block);

        // DC coefficient should be sum of all values
        assert_eq!(block[0][0], 136); // Sum of 1..16 = 136
    }

    #[test]
    fn test_satd_4x4_identical() {
        let block = vec![100u8; 16];
        let satd = SatdCalculator::satd_4x4(&block, 4, &block, 4);
        assert_eq!(satd, 0);
    }

    #[test]
    fn test_satd_4x4_constant_diff() {
        let src = vec![100u8; 16];
        let ref_block = vec![110u8; 16];
        let satd = SatdCalculator::satd_4x4(&src, 4, &ref_block, 4);
        // SATD of constant difference is special case
        assert!(satd > 0);
    }

    #[test]
    fn test_satd_8x8() {
        let src = vec![100u8; 64];
        let ref_block = vec![100u8; 64];
        let satd = SatdCalculator::satd_8x8(&src, 8, &ref_block, 8);
        assert_eq!(satd, 0);
    }

    #[test]
    fn test_satd_16x16() {
        let src = vec![100u8; 256];
        let ref_block = vec![100u8; 256];
        let satd = SatdCalculator::satd_16x16(&src, 16, &ref_block, 16);
        assert_eq!(satd, 0);
    }

    #[test]
    fn test_subpel_refiner_creation() {
        let refiner = SubpelRefiner::new();
        assert_eq!(refiner.config.precision, MvPrecision::QuarterPel);
        assert!(refiner.config.use_satd);
    }

    #[test]
    fn test_subpel_config() {
        let config = SubpelConfig {
            precision: MvPrecision::HalfPel,
            use_satd: false,
            ..Default::default()
        };
        assert_eq!(config.precision, MvPrecision::HalfPel);
        assert!(!config.use_satd);
    }

    #[test]
    fn test_subpel_refiner_no_motion() {
        let src = vec![100u8; 64];
        let reference = vec![100u8; 256];

        let mut refiner = SubpelRefiner::new();
        let mv = MotionVector::zero();

        let result = refiner.refine(
            &src,
            8,
            &reference,
            16,
            BlockSize::Block8x8,
            0,
            0,
            16,
            16,
            mv,
        );

        // Perfect match, no motion
        assert_eq!(result.mv.dx, 0);
        assert_eq!(result.mv.dy, 0);
    }

    #[test]
    fn test_interpolation_full_pel() {
        let mut refiner = SubpelRefiner::new();
        let reference = vec![128u8; 256];
        let mv = MotionVector::zero();

        let success = refiner.interpolate_block(&reference, 16, 0, 0, 16, 16, &mv, 8, 8);

        assert!(success);
        // Full-pel position should copy exactly
        assert_eq!(refiner.interp_buffer[0], 128);
    }

    #[test]
    fn test_interpolation_half_pel() {
        let mut refiner = SubpelRefiner::new();
        let mut reference = vec![100u8; 256];
        // Create gradient
        for i in 0..256 {
            reference[i] = (i % 256) as u8;
        }

        let mv = MotionVector::new(4, 0); // Half-pel in x

        let success = refiner.interpolate_block(&reference, 16, 0, 0, 16, 16, &mv, 8, 8);

        assert!(success);
    }

    #[test]
    fn test_subpel_patterns() {
        assert_eq!(SubpelPatterns::DIAMOND_5.len(), 5);
        assert_eq!(SubpelPatterns::SQUARE_9.len(), 9);
        assert_eq!(SubpelPatterns::EXTENDED_25.len(), 25);
    }

    #[test]
    fn test_half_pel_interpolate_h() {
        let src = vec![100u8, 200u8, 100u8, 200u8, 100u8, 200u8, 100u8, 200u8];
        let mut dst = vec![0u8; 7];

        HalfPelInterpolator::interpolate_h(&src, 8, &mut dst, 7, 1);

        // Interpolated values should be between neighbors
        assert_eq!(dst[0], 150);
        assert_eq!(dst[1], 150);
    }

    #[test]
    fn test_satd_block_size() {
        let src = vec![100u8; 128];
        let ref_block = vec![100u8; 128];

        let satd = SatdCalculator::satd(&src, 8, &ref_block, 8, BlockSize::Block8x16);
        assert_eq!(satd, 0);
    }
}

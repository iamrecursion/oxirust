// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Motion compensation for Theora.
//!
//! Implements block-based motion compensation with half-pixel precision
//! for inter frame prediction.

use crate::error::CodecResult;

/// Motion vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVector {
    /// Horizontal component (in half-pixels).
    pub x: i16,
    /// Vertical component (in half-pixels).
    pub y: i16,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub const fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// Check if this is a zero motion vector.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.x == 0 && self.y == 0
    }

    /// Add another motion vector.
    #[must_use]
    pub const fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    /// Clamp motion vector to valid range.
    #[must_use]
    pub fn clamp(&self, min_x: i16, max_x: i16, min_y: i16, max_y: i16) -> Self {
        Self {
            x: self.x.clamp(min_x, max_x),
            y: self.y.clamp(min_y, max_y),
        }
    }
}

/// Motion compensate an 8x8 block with half-pixel precision.
///
/// # Arguments
///
/// * `reference` - Reference frame plane
/// * `ref_stride` - Stride of reference plane
/// * `x` - Block X position in reference frame (pixels)
/// * `y` - Block Y position in reference frame (pixels)
/// * `mv` - Motion vector (half-pixel precision)
/// * `output` - Output 8x8 block
pub fn motion_compensate_8x8(
    reference: &[u8],
    ref_stride: usize,
    x: usize,
    y: usize,
    mv: MotionVector,
    output: &mut [u8; 64],
) {
    // Convert half-pixel motion vector to full pixels and fractional part
    let full_x = (x as i32 + (mv.x >> 1) as i32) as usize;
    let full_y = (y as i32 + (mv.y >> 1) as i32) as usize;
    let frac_x = (mv.x & 1) != 0;
    let frac_y = (mv.y & 1) != 0;

    match (frac_x, frac_y) {
        (false, false) => {
            // Full-pixel: simple copy
            copy_block_fullpel(reference, ref_stride, full_x, full_y, output);
        }
        (true, false) => {
            // Half-pixel horizontal
            interpolate_horizontal(reference, ref_stride, full_x, full_y, output);
        }
        (false, true) => {
            // Half-pixel vertical
            interpolate_vertical(reference, ref_stride, full_x, full_y, output);
        }
        (true, true) => {
            // Half-pixel both directions
            interpolate_both(reference, ref_stride, full_x, full_y, output);
        }
    }
}

/// Copy a full-pixel aligned 8x8 block.
fn copy_block_fullpel(
    reference: &[u8],
    ref_stride: usize,
    x: usize,
    y: usize,
    output: &mut [u8; 64],
) {
    for i in 0..8 {
        let src_offset = (y + i) * ref_stride + x;
        let dst_offset = i * 8;
        if src_offset + 8 <= reference.len() {
            output[dst_offset..dst_offset + 8]
                .copy_from_slice(&reference[src_offset..src_offset + 8]);
        } else {
            output[dst_offset..dst_offset + 8].fill(128);
        }
    }
}

/// Interpolate horizontally for half-pixel motion compensation.
fn interpolate_horizontal(
    reference: &[u8],
    ref_stride: usize,
    x: usize,
    y: usize,
    output: &mut [u8; 64],
) {
    for i in 0..8 {
        let row_offset = (y + i) * ref_stride + x;
        if row_offset + 9 > reference.len() {
            output[i * 8..(i + 1) * 8].fill(128);
            continue;
        }

        for j in 0..8 {
            let p0 = u16::from(reference[row_offset + j]);
            let p1 = u16::from(reference[row_offset + j + 1]);
            output[i * 8 + j] = ((p0 + p1 + 1) >> 1) as u8;
        }
    }
}

/// Interpolate vertically for half-pixel motion compensation.
fn interpolate_vertical(
    reference: &[u8],
    ref_stride: usize,
    x: usize,
    y: usize,
    output: &mut [u8; 64],
) {
    for i in 0..8 {
        let row_offset = (y + i) * ref_stride + x;
        let next_row_offset = (y + i + 1) * ref_stride + x;

        if next_row_offset + 8 > reference.len() {
            output[i * 8..(i + 1) * 8].fill(128);
            continue;
        }

        for j in 0..8 {
            let p0 = u16::from(reference[row_offset + j]);
            let p1 = u16::from(reference[next_row_offset + j]);
            output[i * 8 + j] = ((p0 + p1 + 1) >> 1) as u8;
        }
    }
}

/// Interpolate both horizontally and vertically for half-pixel motion compensation.
fn interpolate_both(
    reference: &[u8],
    ref_stride: usize,
    x: usize,
    y: usize,
    output: &mut [u8; 64],
) {
    for i in 0..8 {
        let row_offset = (y + i) * ref_stride + x;
        let next_row_offset = (y + i + 1) * ref_stride + x;

        if next_row_offset + 9 > reference.len() {
            output[i * 8..(i + 1) * 8].fill(128);
            continue;
        }

        for j in 0..8 {
            let p00 = u16::from(reference[row_offset + j]);
            let p01 = u16::from(reference[row_offset + j + 1]);
            let p10 = u16::from(reference[next_row_offset + j]);
            let p11 = u16::from(reference[next_row_offset + j + 1]);
            output[i * 8 + j] = ((p00 + p01 + p10 + p11 + 2) >> 2) as u8;
        }
    }
}

/// Motion estimation using Sum of Absolute Differences (SAD).
///
/// Finds the best matching block in the reference frame.
///
/// # Arguments
///
/// * `current` - Current block (8x8)
/// * `reference` - Reference frame plane
/// * `ref_stride` - Stride of reference plane
/// * `search_x` - X position to start search
/// * `search_y` - Y position to start search
/// * `search_range` - Search range in pixels (in each direction)
///
/// # Returns
///
/// Best motion vector and SAD cost.
pub fn motion_estimation_sad(
    current: &[u8; 64],
    reference: &[u8],
    ref_stride: usize,
    search_x: usize,
    search_y: usize,
    search_range: i16,
) -> (MotionVector, u32) {
    let mut best_mv = MotionVector::new(0, 0);
    let mut best_sad = u32::MAX;

    let min_x = (search_x as i16).saturating_sub(search_range).max(0);
    let max_x = (search_x as i16) + search_range;
    let min_y = (search_y as i16).saturating_sub(search_range).max(0);
    let max_y = (search_y as i16) + search_range;

    // Full-pixel search
    for dy in min_y..=max_y {
        for dx in min_x..=max_x {
            let mut block = [0u8; 64];
            copy_block_fullpel(reference, ref_stride, dx as usize, dy as usize, &mut block);

            let sad = calculate_sad(current, &block);
            if sad < best_sad {
                best_sad = sad;
                best_mv = MotionVector::new(
                    ((dx - search_x as i16) * 2) as i16,
                    ((dy - search_y as i16) * 2) as i16,
                );
            }
        }
    }

    // Half-pixel refinement around best full-pixel position
    let center_x = search_x as i16 + (best_mv.x >> 1);
    let center_y = search_y as i16 + (best_mv.y >> 1);

    for hy in -1..=1 {
        for hx in -1..=1 {
            let test_mv = MotionVector::new(best_mv.x + hx, best_mv.y + hy);
            let mut block = [0u8; 64];
            motion_compensate_8x8(
                reference, ref_stride, search_x, search_y, test_mv, &mut block,
            );

            let sad = calculate_sad(current, &block);
            if sad < best_sad {
                best_sad = sad;
                best_mv = test_mv;
            }
        }
    }

    (best_mv, best_sad)
}

/// Calculate Sum of Absolute Differences between two 8x8 blocks.
fn calculate_sad(block1: &[u8; 64], block2: &[u8; 64]) -> u32 {
    let mut sad = 0u32;
    for i in 0..64 {
        sad += (i32::from(block1[i]) - i32::from(block2[i])).unsigned_abs();
    }
    sad
}

/// Diamond search pattern for motion estimation.
///
/// More efficient than full search for larger search ranges.
pub fn motion_estimation_diamond(
    current: &[u8; 64],
    reference: &[u8],
    ref_stride: usize,
    search_x: usize,
    search_y: usize,
    search_range: i16,
) -> (MotionVector, u32) {
    let diamond_pattern = [
        (0i16, -2i16),
        (-1, -1),
        (0, -1),
        (1, -1),
        (-2, 0),
        (-1, 0),
        (1, 0),
        (2, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (0, 2),
    ];

    let mut best_mv = MotionVector::new(0, 0);
    let mut block = [0u8; 64];
    copy_block_fullpel(reference, ref_stride, search_x, search_y, &mut block);
    let mut best_sad = calculate_sad(current, &block);

    let mut step = search_range;
    while step >= 1 {
        let mut improved = false;

        for &(dx, dy) in &diamond_pattern {
            let test_x = (search_x as i16 + (best_mv.x >> 1) + dx * step) as usize;
            let test_y = (search_y as i16 + (best_mv.y >> 1) + dy * step) as usize;

            copy_block_fullpel(reference, ref_stride, test_x, test_y, &mut block);
            let sad = calculate_sad(current, &block);

            if sad < best_sad {
                best_sad = sad;
                best_mv = MotionVector::new(
                    ((test_x as i16 - search_x as i16) * 2) as i16,
                    ((test_y as i16 - search_y as i16) * 2) as i16,
                );
                improved = true;
            }
        }

        if !improved {
            step /= 2;
        }
    }

    // Half-pixel refinement
    for hy in -1..=1 {
        for hx in -1..=1 {
            let test_mv = MotionVector::new(best_mv.x + hx, best_mv.y + hy);
            motion_compensate_8x8(
                reference, ref_stride, search_x, search_y, test_mv, &mut block,
            );

            let sad = calculate_sad(current, &block);
            if sad < best_sad {
                best_sad = sad;
                best_mv = test_mv;
            }
        }
    }

    (best_mv, best_sad)
}

/// Median motion vector predictor.
///
/// Predicts the motion vector based on neighboring blocks.
#[must_use]
pub fn predict_motion_vector(
    left_mv: Option<MotionVector>,
    top_mv: Option<MotionVector>,
    top_right_mv: Option<MotionVector>,
) -> MotionVector {
    let left = left_mv.unwrap_or_default();
    let top = top_mv.unwrap_or_default();
    let top_right = top_right_mv.unwrap_or_default();

    // Median of three components
    let x = median3(left.x, top.x, top_right.x);
    let y = median3(left.y, top.y, top_right.y);

    MotionVector::new(x, y)
}

/// Calculate median of three values.
fn median3(a: i16, b: i16, c: i16) -> i16 {
    if a > b {
        if b > c {
            b
        } else if a > c {
            c
        } else {
            a
        }
    } else if a > c {
        a
    } else if b > c {
        c
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector() {
        let mv1 = MotionVector::new(4, 6);
        let mv2 = MotionVector::new(2, -3);
        let sum = mv1.add(&mv2);

        assert_eq!(sum.x, 6);
        assert_eq!(sum.y, 3);
        assert!(!sum.is_zero());
    }

    #[test]
    fn test_full_pixel_compensation() {
        let reference = vec![100u8; 16 * 16];
        let mut output = [0u8; 64];

        motion_compensate_8x8(&reference, 16, 4, 4, MotionVector::new(0, 0), &mut output);

        assert_eq!(output[0], 100);
        assert_eq!(output[63], 100);
    }

    #[test]
    fn test_half_pixel_compensation() {
        let mut reference = vec![0u8; 16 * 16];
        for i in 0..16 {
            for j in 0..16 {
                reference[i * 16 + j] = ((i + j) * 10) as u8;
            }
        }

        let mut output = [0u8; 64];
        motion_compensate_8x8(&reference, 16, 4, 4, MotionVector::new(1, 0), &mut output);

        // Half-pixel should be average of adjacent pixels
        let expected =
            ((reference[4 * 16 + 4] as u16 + reference[4 * 16 + 5] as u16 + 1) >> 1) as u8;
        assert_eq!(output[0], expected);
    }

    #[test]
    fn test_motion_predictor() {
        let left = Some(MotionVector::new(4, 6));
        let top = Some(MotionVector::new(2, 8));
        let top_right = Some(MotionVector::new(6, 4));

        let pred = predict_motion_vector(left, top, top_right);
        assert_eq!(pred.x, 4); // median(4, 2, 6)
        assert_eq!(pred.y, 6); // median(6, 8, 4)
    }

    #[test]
    fn test_median() {
        assert_eq!(median3(1, 2, 3), 2);
        assert_eq!(median3(3, 2, 1), 2);
        assert_eq!(median3(2, 1, 3), 2);
        assert_eq!(median3(5, 5, 5), 5);
    }
}

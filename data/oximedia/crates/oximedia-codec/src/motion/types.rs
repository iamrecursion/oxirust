//! Core types for motion estimation.
//!
//! This module provides fundamental types used throughout the motion
//! estimation pipeline, including motion vectors, search ranges, and
//! block matching results.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::trivially_copy_pass_by_ref)]

use std::ops::{Add, Neg, Sub};

/// Maximum motion vector component magnitude (in sub-pixel units).
pub const MV_MAX: i32 = 16383 * 8; // 1/8 pel precision

/// Minimum motion vector component magnitude (in sub-pixel units).
pub const MV_MIN: i32 = -16384 * 8;

/// Default search range in pixels.
pub const DEFAULT_SEARCH_RANGE: i32 = 64;

/// Motion vector precision levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum MvPrecision {
    /// Full pixel precision (integer pel).
    FullPel = 0,
    /// Half pixel precision (1/2 pel).
    HalfPel = 1,
    /// Quarter pixel precision (1/4 pel).
    #[default]
    QuarterPel = 2,
    /// Eighth pixel precision (1/8 pel).
    EighthPel = 3,
}

impl MvPrecision {
    /// Returns the number of fractional bits for this precision.
    #[must_use]
    pub const fn fractional_bits(self) -> u8 {
        match self {
            Self::FullPel => 0,
            Self::HalfPel => 1,
            Self::QuarterPel => 2,
            Self::EighthPel => 3,
        }
    }

    /// Returns the scale factor for sub-pixel units.
    #[must_use]
    pub const fn scale(self) -> i32 {
        1 << self.fractional_bits()
    }

    /// Returns the mask for extracting fractional part.
    #[must_use]
    pub const fn frac_mask(self) -> i32 {
        self.scale() - 1
    }

    /// Converts a value from this precision to another.
    #[must_use]
    pub const fn convert(self, value: i32, target: Self) -> i32 {
        let src_bits = self.fractional_bits() as i32;
        let dst_bits = target.fractional_bits() as i32;
        let shift = dst_bits - src_bits;
        if shift > 0 {
            value << shift
        } else {
            value >> (-shift)
        }
    }
}

/// A motion vector with sub-pixel precision.
///
/// Components are stored in 1/8 pixel (eighth-pel) precision internally.
/// This allows conversion to any lower precision without loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub struct MotionVector {
    /// Horizontal displacement (dx) in 1/8 pixel units.
    pub dx: i32,
    /// Vertical displacement (dy) in 1/8 pixel units.
    pub dy: i32,
}

impl MotionVector {
    /// Creates a zero motion vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self { dx: 0, dy: 0 }
    }

    /// Creates a motion vector with the given components (in 1/8 pel).
    #[must_use]
    pub const fn new(dx: i32, dy: i32) -> Self {
        Self { dx, dy }
    }

    /// Creates a motion vector from full-pixel coordinates.
    #[must_use]
    pub const fn from_full_pel(dx: i32, dy: i32) -> Self {
        Self {
            dx: dx << 3,
            dy: dy << 3,
        }
    }

    /// Creates a motion vector at the given precision.
    #[must_use]
    pub const fn from_precision(dx: i32, dy: i32, precision: MvPrecision) -> Self {
        let shift = 3 - precision.fractional_bits() as i32;
        Self {
            dx: dx << shift,
            dy: dy << shift,
        }
    }

    /// Returns true if this is a zero motion vector.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.dx == 0 && self.dy == 0
    }

    /// Returns the full-pixel horizontal component.
    #[must_use]
    pub const fn full_pel_x(&self) -> i32 {
        self.dx >> 3
    }

    /// Returns the full-pixel vertical component.
    #[must_use]
    pub const fn full_pel_y(&self) -> i32 {
        self.dy >> 3
    }

    /// Returns the fractional horizontal component (0-7).
    #[must_use]
    pub const fn frac_x(&self) -> i32 {
        self.dx & 7
    }

    /// Returns the fractional vertical component (0-7).
    #[must_use]
    pub const fn frac_y(&self) -> i32 {
        self.dy & 7
    }

    /// Returns the half-pel x component (0-1).
    #[must_use]
    pub const fn half_pel_x(&self) -> i32 {
        (self.dx >> 2) & 1
    }

    /// Returns the half-pel y component (0-1).
    #[must_use]
    pub const fn half_pel_y(&self) -> i32 {
        (self.dy >> 2) & 1
    }

    /// Returns the quarter-pel x component (0-3).
    #[must_use]
    pub const fn quarter_pel_x(&self) -> i32 {
        (self.dx >> 1) & 3
    }

    /// Returns the quarter-pel y component (0-3).
    #[must_use]
    pub const fn quarter_pel_y(&self) -> i32 {
        (self.dy >> 1) & 3
    }

    /// Converts to the specified precision (may lose fractional bits).
    #[must_use]
    pub const fn to_precision(&self, precision: MvPrecision) -> Self {
        let shift = 3 - precision.fractional_bits() as i32;
        let mask = !((1 << shift) - 1);
        Self {
            dx: self.dx & mask,
            dy: self.dy & mask,
        }
    }

    /// Rounds to the specified precision.
    #[must_use]
    pub const fn round_to_precision(&self, precision: MvPrecision) -> Self {
        let shift = 3 - precision.fractional_bits() as i32;
        let round = 1 << (shift - 1);
        if shift > 0 {
            Self {
                dx: ((self.dx + round) >> shift) << shift,
                dy: ((self.dy + round) >> shift) << shift,
            }
        } else {
            *self
        }
    }

    /// Clamps the motion vector to valid range.
    #[must_use]
    pub fn clamp(&self) -> Self {
        Self {
            dx: self.dx.clamp(MV_MIN, MV_MAX),
            dy: self.dy.clamp(MV_MIN, MV_MAX),
        }
    }

    /// Clamps to the specified search range (in full pixels).
    #[must_use]
    pub fn clamp_to_range(&self, range: &SearchRange) -> Self {
        Self {
            dx: self.dx.clamp(-range.horizontal << 3, range.horizontal << 3),
            dy: self.dy.clamp(-range.vertical << 3, range.vertical << 3),
        }
    }

    /// Returns the squared magnitude.
    #[must_use]
    pub const fn magnitude_squared(&self) -> i64 {
        (self.dx as i64) * (self.dx as i64) + (self.dy as i64) * (self.dy as i64)
    }

    /// Returns the L1 norm (Manhattan distance).
    #[must_use]
    pub const fn l1_norm(&self) -> i32 {
        self.dx.abs() + self.dy.abs()
    }

    /// Returns the L-infinity norm (Chebyshev distance).
    #[must_use]
    pub fn linf_norm(&self) -> i32 {
        self.dx.abs().max(self.dy.abs())
    }

    /// Scales the motion vector.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn scale(&self, num: i32, den: i32) -> Self {
        if den == 0 {
            return *self;
        }
        Self {
            dx: ((i64::from(self.dx) * i64::from(num)) / i64::from(den)) as i32,
            dy: ((i64::from(self.dy) * i64::from(num)) / i64::from(den)) as i32,
        }
    }
}

impl Add for MotionVector {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            dx: self.dx.saturating_add(other.dx),
            dy: self.dy.saturating_add(other.dy),
        }
    }
}

impl Sub for MotionVector {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            dx: self.dx.saturating_sub(other.dx),
            dy: self.dy.saturating_sub(other.dy),
        }
    }
}

impl Neg for MotionVector {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            dx: self.dx.saturating_neg(),
            dy: self.dy.saturating_neg(),
        }
    }
}

/// Search range for motion estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchRange {
    /// Horizontal search range in full pixels.
    pub horizontal: i32,
    /// Vertical search range in full pixels.
    pub vertical: i32,
}

impl Default for SearchRange {
    fn default() -> Self {
        Self::new(DEFAULT_SEARCH_RANGE, DEFAULT_SEARCH_RANGE)
    }
}

impl SearchRange {
    /// Creates a new search range.
    #[must_use]
    pub const fn new(horizontal: i32, vertical: i32) -> Self {
        Self {
            horizontal,
            vertical,
        }
    }

    /// Creates a symmetric search range.
    #[must_use]
    pub const fn symmetric(range: i32) -> Self {
        Self::new(range, range)
    }

    /// Returns the total number of search positions.
    #[must_use]
    pub const fn num_positions(&self) -> u64 {
        let w = (2 * self.horizontal + 1) as u64;
        let h = (2 * self.vertical + 1) as u64;
        w * h
    }

    /// Checks if a position is within the search range.
    #[must_use]
    pub const fn contains(&self, dx: i32, dy: i32) -> bool {
        dx >= -self.horizontal
            && dx <= self.horizontal
            && dy >= -self.vertical
            && dy <= self.vertical
    }

    /// Returns a scaled search range.
    #[must_use]
    pub const fn scale(&self, factor: i32) -> Self {
        Self {
            horizontal: self.horizontal * factor,
            vertical: self.vertical * factor,
        }
    }

    /// Returns a reduced search range (for refinement).
    #[must_use]
    pub const fn reduce(&self, factor: i32) -> Self {
        if factor == 0 {
            *self
        } else {
            Self {
                horizontal: self.horizontal / factor,
                vertical: self.vertical / factor,
            }
        }
    }
}

/// Result of a block matching operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockMatch {
    /// Motion vector.
    pub mv: MotionVector,
    /// Sum of Absolute Differences (distortion).
    pub sad: u32,
    /// Rate-distortion cost (if computed).
    pub cost: u32,
}

impl Default for BlockMatch {
    fn default() -> Self {
        Self::worst()
    }
}

impl BlockMatch {
    /// Creates a new block match result.
    #[must_use]
    pub const fn new(mv: MotionVector, sad: u32, cost: u32) -> Self {
        Self { mv, sad, cost }
    }

    /// Creates a zero motion vector match.
    #[must_use]
    pub const fn zero_mv(sad: u32) -> Self {
        Self {
            mv: MotionVector::zero(),
            sad,
            cost: sad,
        }
    }

    /// Creates the worst possible match (for initialization).
    #[must_use]
    pub const fn worst() -> Self {
        Self {
            mv: MotionVector::zero(),
            sad: u32::MAX,
            cost: u32::MAX,
        }
    }

    /// Returns true if this match is better than another.
    #[must_use]
    pub const fn is_better_than(&self, other: &Self) -> bool {
        self.cost < other.cost
    }

    /// Updates with a better match if found.
    pub fn update_if_better(&mut self, other: &Self) {
        if other.is_better_than(self) {
            *self = *other;
        }
    }
}

/// Motion vector cost calculator for rate-distortion optimization.
#[derive(Clone, Copy, Debug)]
pub struct MvCost {
    /// Lambda for rate-distortion tradeoff.
    pub lambda: f32,
    /// Weight for MV bits.
    pub mv_weight: f32,
    /// Reference motion vector for differential coding.
    pub ref_mv: MotionVector,
}

impl Default for MvCost {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl MvCost {
    /// Creates a new MV cost calculator.
    #[must_use]
    pub const fn new(lambda: f32) -> Self {
        Self {
            lambda,
            mv_weight: 1.0,
            ref_mv: MotionVector::zero(),
        }
    }

    /// Creates with a reference motion vector.
    #[must_use]
    pub const fn with_ref_mv(lambda: f32, ref_mv: MotionVector) -> Self {
        Self {
            lambda,
            mv_weight: 1.0,
            ref_mv,
        }
    }

    /// Estimates the bit cost of a motion vector.
    #[must_use]
    pub fn estimate_bits(&self, mv: &MotionVector) -> f32 {
        let diff = *mv - self.ref_mv;
        let dx_bits = Self::component_bits(diff.dx);
        let dy_bits = Self::component_bits(diff.dy);
        (dx_bits + dy_bits) * self.mv_weight
    }

    /// Estimates bits for a single component.
    #[must_use]
    fn component_bits(value: i32) -> f32 {
        if value == 0 {
            return 1.0;
        }
        let abs_val = value.unsigned_abs();
        // Approximate: 2 * log2(abs) + constant overhead
        let log2_approx = 32 - abs_val.leading_zeros();
        (2 * log2_approx + 2) as f32
    }

    /// Calculates the rate-distortion cost.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn rd_cost(&self, mv: &MotionVector, sad: u32) -> u32 {
        let bits = self.estimate_bits(mv);
        let rate_cost = (bits * self.lambda) as u32;
        sad.saturating_add(rate_cost)
    }

    /// Updates the reference motion vector.
    pub fn set_ref_mv(&mut self, ref_mv: MotionVector) {
        self.ref_mv = ref_mv;
    }
}

/// Block size for motion estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum BlockSize {
    /// 4x4 block.
    Block4x4 = 0,
    /// 4x8 block.
    Block4x8 = 1,
    /// 8x4 block.
    Block8x4 = 2,
    /// 8x8 block.
    #[default]
    Block8x8 = 3,
    /// 8x16 block.
    Block8x16 = 4,
    /// 16x8 block.
    Block16x8 = 5,
    /// 16x16 block.
    Block16x16 = 6,
    /// 16x32 block.
    Block16x32 = 7,
    /// 32x16 block.
    Block32x16 = 8,
    /// 32x32 block.
    Block32x32 = 9,
    /// 32x64 block.
    Block32x64 = 10,
    /// 64x32 block.
    Block64x32 = 11,
    /// 64x64 block.
    Block64x64 = 12,
    /// 64x128 block.
    Block64x128 = 13,
    /// 128x64 block.
    Block128x64 = 14,
    /// 128x128 block.
    Block128x128 = 15,
}

impl BlockSize {
    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(&self) -> usize {
        match self {
            Self::Block4x4 | Self::Block4x8 => 4,
            Self::Block8x4 | Self::Block8x8 | Self::Block8x16 => 8,
            Self::Block16x8 | Self::Block16x16 | Self::Block16x32 => 16,
            Self::Block32x16 | Self::Block32x32 | Self::Block32x64 => 32,
            Self::Block64x32 | Self::Block64x64 | Self::Block64x128 => 64,
            Self::Block128x64 | Self::Block128x128 => 128,
        }
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(&self) -> usize {
        match self {
            Self::Block4x4 | Self::Block8x4 => 4,
            Self::Block4x8 | Self::Block8x8 | Self::Block16x8 => 8,
            Self::Block8x16 | Self::Block16x16 | Self::Block32x16 => 16,
            Self::Block16x32 | Self::Block32x32 | Self::Block64x32 => 32,
            Self::Block32x64 | Self::Block64x64 | Self::Block128x64 => 64,
            Self::Block64x128 | Self::Block128x128 => 128,
        }
    }

    /// Returns the number of pixels in the block.
    #[must_use]
    pub const fn num_pixels(&self) -> usize {
        self.width() * self.height()
    }

    /// Returns true if the block is square.
    #[must_use]
    pub const fn is_square(&self) -> bool {
        matches!(
            self,
            Self::Block4x4
                | Self::Block8x8
                | Self::Block16x16
                | Self::Block32x32
                | Self::Block64x64
                | Self::Block128x128
        )
    }

    /// Returns the log2 of width.
    #[must_use]
    pub const fn width_log2(&self) -> u8 {
        match self.width() {
            4 => 2,
            8 => 3,
            16 => 4,
            32 => 5,
            64 => 6,
            128 => 7,
            _ => 0,
        }
    }

    /// Returns the log2 of height.
    #[must_use]
    pub const fn height_log2(&self) -> u8 {
        match self.height() {
            4 => 2,
            8 => 3,
            16 => 4,
            32 => 5,
            64 => 6,
            128 => 7,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mv_precision() {
        assert_eq!(MvPrecision::FullPel.fractional_bits(), 0);
        assert_eq!(MvPrecision::HalfPel.fractional_bits(), 1);
        assert_eq!(MvPrecision::QuarterPel.fractional_bits(), 2);
        assert_eq!(MvPrecision::EighthPel.fractional_bits(), 3);

        assert_eq!(MvPrecision::FullPel.scale(), 1);
        assert_eq!(MvPrecision::QuarterPel.scale(), 4);
        assert_eq!(MvPrecision::EighthPel.scale(), 8);
    }

    #[test]
    fn test_mv_precision_convert() {
        // Full pel to quarter pel
        assert_eq!(MvPrecision::FullPel.convert(2, MvPrecision::QuarterPel), 8);
        // Quarter pel to full pel
        assert_eq!(MvPrecision::QuarterPel.convert(8, MvPrecision::FullPel), 2);
    }

    #[test]
    fn test_motion_vector_creation() {
        let mv = MotionVector::new(16, -24);
        assert_eq!(mv.dx, 16);
        assert_eq!(mv.dy, -24);

        let mv_fp = MotionVector::from_full_pel(2, -3);
        assert_eq!(mv_fp.dx, 16);
        assert_eq!(mv_fp.dy, -24);
    }

    #[test]
    fn test_motion_vector_components() {
        let mv = MotionVector::new(27, -19); // 3.375, -2.375 in full pixels

        assert_eq!(mv.full_pel_x(), 3);
        assert_eq!(mv.full_pel_y(), -3); // -19 >> 3 = -3
        assert_eq!(mv.frac_x(), 3);
        assert_eq!(mv.frac_y(), -19 & 7);
    }

    #[test]
    fn test_motion_vector_zero() {
        let mv = MotionVector::zero();
        assert!(mv.is_zero());
        assert_eq!(mv.magnitude_squared(), 0);
    }

    #[test]
    fn test_motion_vector_arithmetic() {
        let mv1 = MotionVector::new(10, 20);
        let mv2 = MotionVector::new(5, -10);

        let sum = mv1 + mv2;
        assert_eq!(sum.dx, 15);
        assert_eq!(sum.dy, 10);

        let diff = mv1 - mv2;
        assert_eq!(diff.dx, 5);
        assert_eq!(diff.dy, 30);

        let neg = -mv1;
        assert_eq!(neg.dx, -10);
        assert_eq!(neg.dy, -20);
    }

    #[test]
    fn test_motion_vector_magnitude() {
        let mv = MotionVector::new(3, 4);
        assert_eq!(mv.magnitude_squared(), 25);
        assert_eq!(mv.l1_norm(), 7);
        assert_eq!(mv.linf_norm(), 4);
    }

    #[test]
    fn test_motion_vector_precision_conversion() {
        let mv = MotionVector::new(27, 19); // 3 + 3/8, 2 + 3/8

        let qpel = mv.to_precision(MvPrecision::QuarterPel);
        assert_eq!(qpel.dx & 1, 0); // Should be even
        assert_eq!(qpel.dy & 1, 0);

        let fpel = mv.to_precision(MvPrecision::FullPel);
        assert_eq!(fpel.dx & 7, 0); // Should be multiple of 8
        assert_eq!(fpel.dy & 7, 0);
    }

    #[test]
    fn test_search_range() {
        let range = SearchRange::symmetric(32);
        assert_eq!(range.horizontal, 32);
        assert_eq!(range.vertical, 32);

        assert!(range.contains(0, 0));
        assert!(range.contains(32, 32));
        assert!(range.contains(-32, -32));
        assert!(!range.contains(33, 0));
    }

    #[test]
    fn test_search_range_positions() {
        let range = SearchRange::symmetric(2);
        // (-2..2) x (-2..2) = 5 x 5 = 25 positions
        assert_eq!(range.num_positions(), 25);
    }

    #[test]
    fn test_block_match() {
        let best = BlockMatch::new(MotionVector::new(8, 16), 100, 120);
        let worst = BlockMatch::worst();

        assert!(best.is_better_than(&worst));
        assert!(!worst.is_better_than(&best));
    }

    #[test]
    fn test_block_match_update() {
        let mut current = BlockMatch::worst();
        let better = BlockMatch::new(MotionVector::new(8, 16), 100, 120);

        current.update_if_better(&better);
        assert_eq!(current.sad, 100);
    }

    #[test]
    fn test_mv_cost() {
        let cost = MvCost::new(1.0);
        let mv = MotionVector::new(16, 16);

        let bits = cost.estimate_bits(&mv);
        assert!(bits > 0.0);

        let rd = cost.rd_cost(&mv, 100);
        assert!(rd >= 100);
    }

    #[test]
    fn test_mv_cost_with_ref() {
        let ref_mv = MotionVector::new(16, 16);
        let cost = MvCost::with_ref_mv(1.0, ref_mv);

        // Same MV as reference should have low cost
        let same_bits = cost.estimate_bits(&ref_mv);

        // Different MV should have higher cost
        let diff_mv = MotionVector::new(32, 32);
        let diff_bits = cost.estimate_bits(&diff_mv);

        assert!(same_bits < diff_bits);
    }

    #[test]
    fn test_block_size() {
        assert_eq!(BlockSize::Block8x8.width(), 8);
        assert_eq!(BlockSize::Block8x8.height(), 8);
        assert_eq!(BlockSize::Block8x8.num_pixels(), 64);
        assert!(BlockSize::Block8x8.is_square());

        assert_eq!(BlockSize::Block16x8.width(), 16);
        assert_eq!(BlockSize::Block16x8.height(), 8);
        assert!(!BlockSize::Block16x8.is_square());
    }

    #[test]
    fn test_block_size_log2() {
        assert_eq!(BlockSize::Block4x4.width_log2(), 2);
        assert_eq!(BlockSize::Block8x8.width_log2(), 3);
        assert_eq!(BlockSize::Block16x16.width_log2(), 4);
        assert_eq!(BlockSize::Block64x64.width_log2(), 6);
    }
}

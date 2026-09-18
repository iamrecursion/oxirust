//! AV1 prediction modes.
//!
//! AV1 supports both intra prediction (using samples from the current
//! frame) and inter prediction (using samples from reference frames).
//!
//! # Intra Prediction
//!
//! AV1 has 13 directional intra modes plus special modes:
//! - DC prediction (average of neighbors)
//! - Smooth modes (interpolation)
//! - Paeth prediction (adaptive)
//! - Filter intra (for small blocks)
//!
//! # Inter Prediction
//!
//! Inter prediction uses motion vectors to reference previous frames:
//! - Single reference
//! - Compound reference (two references)
//! - Warped motion
//! - OBMC (Overlapped Block Motion Compensation)
//!
//! # Block Sizes
//!
//! AV1 supports blocks from 4x4 to 128x128 with various rectangular shapes.

#![allow(dead_code)]

/// Intra prediction modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IntraMode {
    /// DC prediction (average of neighbors).
    #[default]
    Dc = 0,
    /// Vertical prediction.
    V = 1,
    /// Horizontal prediction.
    H = 2,
    /// Diagonal 45 degrees.
    D45 = 3,
    /// Diagonal 135 degrees.
    D135 = 4,
    /// Diagonal 113 degrees.
    D113 = 5,
    /// Diagonal 157 degrees.
    D157 = 6,
    /// Diagonal 203 degrees.
    D203 = 7,
    /// Diagonal 67 degrees.
    D67 = 8,
    /// Smooth prediction.
    Smooth = 9,
    /// Smooth vertical prediction.
    SmoothV = 10,
    /// Smooth horizontal prediction.
    SmoothH = 11,
    /// Paeth prediction.
    Paeth = 12,
}

impl IntraMode {
    /// Check if this is a directional mode.
    #[must_use]
    pub const fn is_directional(self) -> bool {
        matches!(
            self,
            Self::V | Self::H | Self::D45 | Self::D135 | Self::D113 | Self::D157 | Self::D203 | Self::D67
        )
    }

    /// Get the angle for directional modes (in degrees).
    #[must_use]
    pub const fn angle(self) -> Option<u16> {
        match self {
            Self::V => Some(90),
            Self::H => Some(180),
            Self::D45 => Some(45),
            Self::D135 => Some(135),
            Self::D113 => Some(113),
            Self::D157 => Some(157),
            Self::D203 => Some(203),
            Self::D67 => Some(67),
            _ => None,
        }
    }
}

/// Inter prediction mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InterMode {
    /// Nearest MV.
    #[default]
    Nearestmv = 0,
    /// Near MV.
    Nearmv = 1,
    /// Global MV.
    Globalmv = 2,
    /// New MV.
    Newmv = 3,
    /// Nearest-Nearest compound.
    NearestNearestmv = 4,
    /// Near-Near compound.
    NearNearmv = 5,
    /// Nearest-New compound.
    NearestNewmv = 6,
    /// New-Nearest compound.
    NewNearestmv = 7,
    /// Near-New compound.
    NearNewmv = 8,
    /// New-Near compound.
    NewNearmv = 9,
    /// Global-Global compound.
    GlobalGlobalmv = 10,
    /// New-New compound.
    NewNewmv = 11,
}

impl InterMode {
    /// Check if this is a compound mode.
    #[must_use]
    pub const fn is_compound(self) -> bool {
        matches!(
            self,
            Self::NearestNearestmv
                | Self::NearNearmv
                | Self::NearestNewmv
                | Self::NewNearestmv
                | Self::NearNewmv
                | Self::NewNearmv
                | Self::GlobalGlobalmv
                | Self::NewNewmv
        )
    }

    /// Check if this mode requires a new motion vector.
    #[must_use]
    pub const fn has_new_mv(self) -> bool {
        matches!(
            self,
            Self::Newmv
                | Self::NearestNewmv
                | Self::NewNearestmv
                | Self::NearNewmv
                | Self::NewNearmv
                | Self::NewNewmv
        )
    }
}

/// Block size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlockSize {
    /// 4x4 block.
    #[default]
    Block4x4 = 0,
    /// 4x8 block.
    Block4x8 = 1,
    /// 8x4 block.
    Block8x4 = 2,
    /// 8x8 block.
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
    /// 4x16 block.
    Block4x16 = 16,
    /// 16x4 block.
    Block16x4 = 17,
    /// 8x32 block.
    Block8x32 = 18,
    /// 32x8 block.
    Block32x8 = 19,
    /// 16x64 block.
    Block16x64 = 20,
    /// 64x16 block.
    Block64x16 = 21,
}

impl BlockSize {
    /// Get width in samples.
    #[must_use]
    pub const fn width(self) -> u32 {
        match self {
            Self::Block4x4 | Self::Block4x8 | Self::Block4x16 => 4,
            Self::Block8x4 | Self::Block8x8 | Self::Block8x16 | Self::Block8x32 => 8,
            Self::Block16x4 | Self::Block16x8 | Self::Block16x16 | Self::Block16x32 | Self::Block16x64 => 16,
            Self::Block32x8 | Self::Block32x16 | Self::Block32x32 | Self::Block32x64 => 32,
            Self::Block64x16 | Self::Block64x32 | Self::Block64x64 | Self::Block64x128 => 64,
            Self::Block128x64 | Self::Block128x128 => 128,
        }
    }

    /// Get height in samples.
    #[must_use]
    pub const fn height(self) -> u32 {
        match self {
            Self::Block4x4 | Self::Block8x4 | Self::Block16x4 => 4,
            Self::Block4x8 | Self::Block8x8 | Self::Block16x8 | Self::Block32x8 => 8,
            Self::Block4x16 | Self::Block8x16 | Self::Block16x16 | Self::Block32x16 | Self::Block64x16 => 16,
            Self::Block8x32 | Self::Block16x32 | Self::Block32x32 | Self::Block64x32 => 32,
            Self::Block16x64 | Self::Block32x64 | Self::Block64x64 | Self::Block128x64 => 64,
            Self::Block64x128 | Self::Block128x128 => 128,
        }
    }
}

/// Motion vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionVector {
    /// Horizontal component (1/8 pel).
    pub row: i16,
    /// Vertical component (1/8 pel).
    pub col: i16,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub const fn new(row: i16, col: i16) -> Self {
        Self { row, col }
    }

    /// Zero motion vector.
    pub const ZERO: Self = Self { row: 0, col: 0 };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intra_mode_directional() {
        assert!(IntraMode::V.is_directional());
        assert!(IntraMode::H.is_directional());
        assert!(IntraMode::D45.is_directional());
        assert!(!IntraMode::Dc.is_directional());
        assert!(!IntraMode::Smooth.is_directional());
    }

    #[test]
    fn test_inter_mode_compound() {
        assert!(!InterMode::Nearestmv.is_compound());
        assert!(!InterMode::Newmv.is_compound());
        assert!(InterMode::NearestNearestmv.is_compound());
        assert!(InterMode::NewNewmv.is_compound());
    }

    #[test]
    fn test_block_size_dimensions() {
        assert_eq!(BlockSize::Block4x4.width(), 4);
        assert_eq!(BlockSize::Block4x4.height(), 4);
        assert_eq!(BlockSize::Block128x128.width(), 128);
        assert_eq!(BlockSize::Block128x128.height(), 128);
        assert_eq!(BlockSize::Block4x16.width(), 4);
        assert_eq!(BlockSize::Block4x16.height(), 16);
    }

    #[test]
    fn test_motion_vector() {
        let mv = MotionVector::new(10, -5);
        assert_eq!(mv.row, 10);
        assert_eq!(mv.col, -5);
        assert_eq!(MotionVector::ZERO.row, 0);
    }
}

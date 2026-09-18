//! VP9 Motion vector types and structures.
//!
//! This module provides motion vector types used for inter-prediction
//! in VP9 decoding. Motion vectors describe the displacement between
//! a block in the current frame and its reference in a previous frame.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use std::ops::{Add, Neg, Sub};

/// Maximum motion vector component magnitude.
pub const MV_MAX: i16 = 16383;

/// Minimum motion vector component magnitude.
pub const MV_MIN: i16 = -16384;

/// Number of motion vector reference types.
pub const MV_REF_TYPES: usize = 4;

/// Number of motion vector joints.
pub const MV_JOINTS: usize = 4;

/// Number of motion vector classes.
pub const MV_CLASSES: usize = 11;

/// Number of motion vector class0 values.
pub const MV_CLASS0_SIZE: usize = 2;

/// Number of motion vector offset bits.
pub const MV_OFFSET_BITS: usize = 10;

/// Number of fractional precision bits.
pub const MV_FP_SIZE: usize = 4;

/// Motion vector reference type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum MvRefType {
    /// Intra prediction (no motion vector).
    #[default]
    Intra = 0,
    /// Last reference frame.
    Last = 1,
    /// Golden reference frame.
    Golden = 2,
    /// Alternate reference frame.
    AltRef = 3,
}

impl MvRefType {
    /// Returns true if this is an inter reference type.
    #[must_use]
    pub const fn is_inter(&self) -> bool {
        !matches!(self, Self::Intra)
    }

    /// Returns true if this is an intra reference type.
    #[must_use]
    pub const fn is_intra(&self) -> bool {
        matches!(self, Self::Intra)
    }

    /// Converts from u8 value to `MvRefType`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Intra),
            1 => Some(Self::Last),
            2 => Some(Self::Golden),
            3 => Some(Self::AltRef),
            _ => None,
        }
    }

    /// Returns the index of this reference type.
    #[must_use]
    pub const fn index(&self) -> usize {
        *self as usize
    }
}

impl From<MvRefType> for u8 {
    fn from(value: MvRefType) -> Self {
        value as u8
    }
}

/// Motion vector joint types.
///
/// Describes which components of a motion vector are non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum MvJoint {
    /// Both components are zero.
    #[default]
    Zero = 0,
    /// Only horizontal component is non-zero.
    HnzVz = 1,
    /// Only vertical component is non-zero.
    HzVnz = 2,
    /// Both components are non-zero.
    HnzVnz = 3,
}

impl MvJoint {
    /// Returns true if the horizontal component is non-zero.
    #[must_use]
    pub const fn has_horizontal(&self) -> bool {
        matches!(self, Self::HnzVz | Self::HnzVnz)
    }

    /// Returns true if the vertical component is non-zero.
    #[must_use]
    pub const fn has_vertical(&self) -> bool {
        matches!(self, Self::HzVnz | Self::HnzVnz)
    }

    /// Converts from u8 value to `MvJoint`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Zero),
            1 => Some(Self::HnzVz),
            2 => Some(Self::HzVnz),
            3 => Some(Self::HnzVnz),
            _ => None,
        }
    }
}

impl From<MvJoint> for u8 {
    fn from(value: MvJoint) -> Self {
        value as u8
    }
}

/// Motion vector class for magnitude encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum MvClass {
    /// Class 0: magnitude 2-3.
    #[default]
    Class0 = 0,
    /// Class 1: magnitude 4-7.
    Class1 = 1,
    /// Class 2: magnitude 8-15.
    Class2 = 2,
    /// Class 3: magnitude 16-31.
    Class3 = 3,
    /// Class 4: magnitude 32-63.
    Class4 = 4,
    /// Class 5: magnitude 64-127.
    Class5 = 5,
    /// Class 6: magnitude 128-255.
    Class6 = 6,
    /// Class 7: magnitude 256-511.
    Class7 = 7,
    /// Class 8: magnitude 512-1023.
    Class8 = 8,
    /// Class 9: magnitude 1024-2047.
    Class9 = 9,
    /// Class 10: magnitude 2048+.
    Class10 = 10,
}

impl MvClass {
    /// Returns the number of bits needed to encode the offset for this class.
    #[must_use]
    pub const fn offset_bits(&self) -> u8 {
        match self {
            Self::Class0 => 0,
            Self::Class1 => 1,
            Self::Class2 => 2,
            Self::Class3 => 3,
            Self::Class4 => 4,
            Self::Class5 => 5,
            Self::Class6 => 6,
            Self::Class7 => 7,
            Self::Class8 => 8,
            Self::Class9 => 9,
            Self::Class10 => 10,
        }
    }

    /// Returns the base magnitude for this class.
    #[must_use]
    pub const fn base_magnitude(&self) -> i16 {
        match self {
            Self::Class0 => 0,
            Self::Class1 => 4,
            Self::Class2 => 8,
            Self::Class3 => 16,
            Self::Class4 => 32,
            Self::Class5 => 64,
            Self::Class6 => 128,
            Self::Class7 => 256,
            Self::Class8 => 512,
            Self::Class9 => 1024,
            Self::Class10 => 2048,
        }
    }

    /// Converts from u8 value to `MvClass`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Class0),
            1 => Some(Self::Class1),
            2 => Some(Self::Class2),
            3 => Some(Self::Class3),
            4 => Some(Self::Class4),
            5 => Some(Self::Class5),
            6 => Some(Self::Class6),
            7 => Some(Self::Class7),
            8 => Some(Self::Class8),
            9 => Some(Self::Class9),
            10 => Some(Self::Class10),
            _ => None,
        }
    }
}

impl From<MvClass> for u8 {
    fn from(value: MvClass) -> Self {
        value as u8
    }
}

/// A motion vector with row and column components.
///
/// Components are in 1/8 pixel precision for VP9.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub struct MotionVector {
    /// Row (vertical) component.
    pub row: i16,
    /// Column (horizontal) component.
    pub col: i16,
}

impl MotionVector {
    /// Creates a new zero motion vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self { row: 0, col: 0 }
    }

    /// Creates a new motion vector with the given components.
    #[must_use]
    pub const fn new(row: i16, col: i16) -> Self {
        Self { row, col }
    }

    /// Returns true if this is a zero motion vector.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.row == 0 && self.col == 0
    }

    /// Returns the joint type for this motion vector.
    #[must_use]
    pub const fn joint(&self) -> MvJoint {
        match (self.col != 0, self.row != 0) {
            (false, false) => MvJoint::Zero,
            (true, false) => MvJoint::HnzVz,
            (false, true) => MvJoint::HzVnz,
            (true, true) => MvJoint::HnzVnz,
        }
    }

    /// Clamps the motion vector components to valid range.
    #[must_use]
    pub fn clamp(&self) -> Self {
        Self {
            row: self.row.clamp(MV_MIN, MV_MAX),
            col: self.col.clamp(MV_MIN, MV_MAX),
        }
    }

    /// Returns the absolute value of the motion vector components.
    #[must_use]
    pub fn abs(&self) -> Self {
        Self {
            row: self.row.abs(),
            col: self.col.abs(),
        }
    }

    /// Returns the squared magnitude of the motion vector.
    #[must_use]
    pub fn magnitude_squared(&self) -> i32 {
        i32::from(self.row) * i32::from(self.row) + i32::from(self.col) * i32::from(self.col)
    }

    /// Converts to full pixel precision (divides by 8).
    #[must_use]
    pub const fn to_full_pixel(&self) -> Self {
        Self {
            row: self.row >> 3,
            col: self.col >> 3,
        }
    }

    /// Returns the row component as full pixel.
    #[must_use]
    pub const fn full_pixel_row(&self) -> i16 {
        self.row >> 3
    }

    /// Returns the column component as full pixel.
    #[must_use]
    pub const fn full_pixel_col(&self) -> i16 {
        self.col >> 3
    }

    /// Returns the fractional row component (0-7).
    #[must_use]
    pub const fn fractional_row(&self) -> i16 {
        self.row & 7
    }

    /// Returns the fractional column component (0-7).
    #[must_use]
    pub const fn fractional_col(&self) -> i16 {
        self.col & 7
    }

    /// Returns true if the motion vector uses quarter-pixel precision.
    #[must_use]
    pub const fn is_quarter_pixel(&self) -> bool {
        (self.row & 1) == 0 && (self.col & 1) == 0
    }

    /// Scales the motion vector by a factor.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn scale(&self, num: i32, den: i32) -> Self {
        if den == 0 {
            return *self;
        }
        Self {
            row: ((i32::from(self.row) * num / den) as i16).clamp(MV_MIN, MV_MAX),
            col: ((i32::from(self.col) * num / den) as i16).clamp(MV_MIN, MV_MAX),
        }
    }
}

impl Add for MotionVector {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            row: self.row.saturating_add(other.row),
            col: self.col.saturating_add(other.col),
        }
    }
}

impl Sub for MotionVector {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            row: self.row.saturating_sub(other.row),
            col: self.col.saturating_sub(other.col),
        }
    }
}

impl Neg for MotionVector {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            row: self.row.saturating_neg(),
            col: self.col.saturating_neg(),
        }
    }
}

/// Motion vector candidate for prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MvCandidate {
    /// The motion vector.
    pub mv: MotionVector,
    /// Reference frame type.
    pub ref_type: MvRefType,
    /// Weight for averaging (higher = more important).
    pub weight: u8,
}

impl MvCandidate {
    /// Creates a new motion vector candidate.
    #[must_use]
    pub const fn new(mv: MotionVector, ref_type: MvRefType, weight: u8) -> Self {
        Self {
            mv,
            ref_type,
            weight,
        }
    }

    /// Creates a zero candidate.
    #[must_use]
    pub const fn zero(ref_type: MvRefType) -> Self {
        Self {
            mv: MotionVector::zero(),
            ref_type,
            weight: 0,
        }
    }
}

/// Motion vector pair for compound prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MvPair {
    /// Motion vector for the first reference.
    pub mv0: MotionVector,
    /// Motion vector for the second reference.
    pub mv1: MotionVector,
}

impl MvPair {
    /// Creates a new motion vector pair.
    #[must_use]
    pub const fn new(mv0: MotionVector, mv1: MotionVector) -> Self {
        Self { mv0, mv1 }
    }

    /// Creates a zero motion vector pair.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            mv0: MotionVector::zero(),
            mv1: MotionVector::zero(),
        }
    }

    /// Returns the motion vector for the given index.
    #[must_use]
    pub const fn get(&self, index: usize) -> MotionVector {
        if index == 0 {
            self.mv0
        } else {
            self.mv1
        }
    }
}

/// Reference frame pair for compound prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RefPair {
    /// First reference frame.
    pub ref0: MvRefType,
    /// Second reference frame.
    pub ref1: MvRefType,
}

impl RefPair {
    /// Creates a single reference pair.
    #[must_use]
    pub const fn single(ref_type: MvRefType) -> Self {
        Self {
            ref0: ref_type,
            ref1: MvRefType::Intra,
        }
    }

    /// Creates a compound reference pair.
    #[must_use]
    pub const fn compound(ref0: MvRefType, ref1: MvRefType) -> Self {
        Self { ref0, ref1 }
    }

    /// Returns true if this is a compound reference.
    #[must_use]
    pub const fn is_compound(&self) -> bool {
        self.ref1.is_inter()
    }

    /// Returns true if this is a single reference.
    #[must_use]
    pub const fn is_single(&self) -> bool {
        !self.is_compound()
    }
}

/// Motion vector context for entropy coding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MvContext {
    /// Number of same-reference neighbors.
    pub same_ref_count: u8,
    /// Number of different-reference neighbors.
    pub diff_ref_count: u8,
    /// Number of new motion vector neighbors.
    pub new_mv_count: u8,
    /// Number of zero motion vector neighbors.
    pub zero_mv_count: u8,
    /// Reference motion vector count.
    pub ref_mv_count: u8,
}

impl MvContext {
    /// Creates a new motion vector context.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            same_ref_count: 0,
            diff_ref_count: 0,
            new_mv_count: 0,
            zero_mv_count: 0,
            ref_mv_count: 0,
        }
    }

    /// Returns the context index for new motion vector mode.
    #[must_use]
    pub const fn new_mv_context(&self) -> usize {
        match self.new_mv_count {
            0 => 0,
            1 => 1,
            _ => 2,
        }
    }

    /// Returns the context index for zero motion vector mode.
    #[must_use]
    pub const fn zero_mv_context(&self) -> usize {
        match self.zero_mv_count {
            0 => 0,
            1 => 1,
            _ => 2,
        }
    }

    /// Returns the context index for reference motion vector mode.
    #[must_use]
    pub const fn ref_mv_context(&self) -> usize {
        match self.ref_mv_count {
            0 => 0,
            1 => 1,
            _ => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector_zero() {
        let mv = MotionVector::zero();
        assert!(mv.is_zero());
        assert_eq!(mv.row, 0);
        assert_eq!(mv.col, 0);
    }

    #[test]
    fn test_motion_vector_new() {
        let mv = MotionVector::new(10, -20);
        assert!(!mv.is_zero());
        assert_eq!(mv.row, 10);
        assert_eq!(mv.col, -20);
    }

    #[test]
    fn test_motion_vector_joint() {
        assert_eq!(MotionVector::zero().joint(), MvJoint::Zero);
        assert_eq!(MotionVector::new(0, 5).joint(), MvJoint::HnzVz);
        assert_eq!(MotionVector::new(5, 0).joint(), MvJoint::HzVnz);
        assert_eq!(MotionVector::new(5, 5).joint(), MvJoint::HnzVnz);
    }

    #[test]
    fn test_motion_vector_add() {
        let mv1 = MotionVector::new(10, 20);
        let mv2 = MotionVector::new(5, -10);
        let result = mv1 + mv2;
        assert_eq!(result.row, 15);
        assert_eq!(result.col, 10);
    }

    #[test]
    fn test_motion_vector_sub() {
        let mv1 = MotionVector::new(10, 20);
        let mv2 = MotionVector::new(5, 10);
        let result = mv1 - mv2;
        assert_eq!(result.row, 5);
        assert_eq!(result.col, 10);
    }

    #[test]
    fn test_motion_vector_neg() {
        let mv = MotionVector::new(10, -20);
        let neg = -mv;
        assert_eq!(neg.row, -10);
        assert_eq!(neg.col, 20);
    }

    #[test]
    fn test_motion_vector_clamp() {
        let mv = MotionVector::new(20000, -20000);
        let clamped = mv.clamp();
        assert_eq!(clamped.row, MV_MAX);
        assert_eq!(clamped.col, MV_MIN);
    }

    #[test]
    fn test_motion_vector_full_pixel() {
        let mv = MotionVector::new(24, 16);
        let full = mv.to_full_pixel();
        assert_eq!(full.row, 3);
        assert_eq!(full.col, 2);
    }

    #[test]
    fn test_motion_vector_fractional() {
        let mv = MotionVector::new(27, 19);
        assert_eq!(mv.fractional_row(), 3);
        assert_eq!(mv.fractional_col(), 3);
    }

    #[test]
    fn test_motion_vector_magnitude_squared() {
        let mv = MotionVector::new(3, 4);
        assert_eq!(mv.magnitude_squared(), 25);
    }

    #[test]
    fn test_motion_vector_scale() {
        let mv = MotionVector::new(100, 200);
        let scaled = mv.scale(1, 2);
        assert_eq!(scaled.row, 50);
        assert_eq!(scaled.col, 100);
    }

    #[test]
    fn test_mv_ref_type() {
        assert!(MvRefType::Intra.is_intra());
        assert!(!MvRefType::Intra.is_inter());
        assert!(MvRefType::Last.is_inter());
        assert!(MvRefType::Golden.is_inter());
        assert!(MvRefType::AltRef.is_inter());
    }

    #[test]
    fn test_mv_joint() {
        assert!(!MvJoint::Zero.has_horizontal());
        assert!(!MvJoint::Zero.has_vertical());
        assert!(MvJoint::HnzVz.has_horizontal());
        assert!(!MvJoint::HnzVz.has_vertical());
        assert!(!MvJoint::HzVnz.has_horizontal());
        assert!(MvJoint::HzVnz.has_vertical());
        assert!(MvJoint::HnzVnz.has_horizontal());
        assert!(MvJoint::HnzVnz.has_vertical());
    }

    #[test]
    fn test_mv_class() {
        assert_eq!(MvClass::Class0.offset_bits(), 0);
        assert_eq!(MvClass::Class5.offset_bits(), 5);
        assert_eq!(MvClass::Class10.offset_bits(), 10);
        assert_eq!(MvClass::Class0.base_magnitude(), 0);
        assert_eq!(MvClass::Class5.base_magnitude(), 64);
    }

    #[test]
    fn test_mv_pair() {
        let pair = MvPair::new(MotionVector::new(10, 20), MotionVector::new(30, 40));
        assert_eq!(pair.get(0), MotionVector::new(10, 20));
        assert_eq!(pair.get(1), MotionVector::new(30, 40));
    }

    #[test]
    fn test_ref_pair() {
        let single = RefPair::single(MvRefType::Last);
        assert!(single.is_single());
        assert!(!single.is_compound());

        let compound = RefPair::compound(MvRefType::Last, MvRefType::Golden);
        assert!(!compound.is_single());
        assert!(compound.is_compound());
    }

    #[test]
    fn test_mv_context() {
        let ctx = MvContext::new();
        assert_eq!(ctx.new_mv_context(), 0);
        assert_eq!(ctx.zero_mv_context(), 0);
        assert_eq!(ctx.ref_mv_context(), 0);
    }
}

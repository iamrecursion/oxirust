//! Intra prediction modes and traits.
//!
//! This module defines the prediction mode types and the common trait
//! that all intra predictors implement.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]

use super::{BlockDimensions, IntraPredContext};

/// Trait for intra prediction implementations.
///
/// All prediction modes implement this trait to provide a unified interface
/// for generating prediction samples.
pub trait IntraPredictor {
    /// Generate prediction samples into the output buffer.
    ///
    /// # Arguments
    /// * `ctx` - Prediction context with neighbor samples
    /// * `output` - Output buffer to fill with predicted samples
    /// * `stride` - Stride (row width) of the output buffer
    /// * `dims` - Block dimensions (width x height)
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    );
}

/// Intra prediction mode enumeration (shared by AV1 and VP9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum IntraMode {
    /// DC prediction (average of neighbors).
    #[default]
    Dc = 0,
    /// Vertical prediction (project top samples down).
    Vertical = 1,
    /// Horizontal prediction (project left samples right).
    Horizontal = 2,
    /// Diagonal 45 degrees (down-right).
    D45 = 3,
    /// Diagonal 135 degrees (up-left).
    D135 = 4,
    /// Diagonal 113 degrees (AV1 naming).
    D113 = 5,
    /// Diagonal 157 degrees (AV1 naming).
    D157 = 6,
    /// Diagonal 203 degrees (AV1 naming).
    D203 = 7,
    /// Diagonal 67 degrees (AV1 naming).
    D67 = 8,
    /// Smooth prediction (AV1).
    Smooth = 9,
    /// Smooth vertical prediction (AV1).
    SmoothV = 10,
    /// Smooth horizontal prediction (AV1).
    SmoothH = 11,
    /// Paeth prediction (AV1).
    Paeth = 12,
    /// Filter intra (AV1 small blocks).
    FilterIntra = 13,
}

impl IntraMode {
    /// Total number of intra modes.
    pub const COUNT: usize = 14;

    /// Check if this is a directional mode.
    #[must_use]
    pub const fn is_directional(self) -> bool {
        matches!(
            self,
            Self::Vertical
                | Self::Horizontal
                | Self::D45
                | Self::D135
                | Self::D113
                | Self::D157
                | Self::D203
                | Self::D67
        )
    }

    /// Check if this is a smooth mode.
    #[must_use]
    pub const fn is_smooth(self) -> bool {
        matches!(self, Self::Smooth | Self::SmoothV | Self::SmoothH)
    }

    /// Get the nominal angle for directional modes (in degrees * 2).
    /// Returns None for non-directional modes.
    #[must_use]
    pub const fn nominal_angle(self) -> Option<u16> {
        match self {
            Self::Vertical => Some(90),
            Self::Horizontal => Some(180),
            Self::D45 => Some(45),
            Self::D135 => Some(135),
            Self::D113 => Some(113),
            Self::D157 => Some(157),
            Self::D203 => Some(203),
            Self::D67 => Some(67),
            _ => None,
        }
    }

    /// Convert from u8.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Dc),
            1 => Some(Self::Vertical),
            2 => Some(Self::Horizontal),
            3 => Some(Self::D45),
            4 => Some(Self::D135),
            5 => Some(Self::D113),
            6 => Some(Self::D157),
            7 => Some(Self::D203),
            8 => Some(Self::D67),
            9 => Some(Self::Smooth),
            10 => Some(Self::SmoothV),
            11 => Some(Self::SmoothH),
            12 => Some(Self::Paeth),
            13 => Some(Self::FilterIntra),
            _ => None,
        }
    }
}

/// Directional mode with angle information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectionalMode {
    /// Base angle in degrees.
    pub angle: u16,
    /// Angle delta (AV1 only, -3 to +3 steps of 3 degrees).
    pub delta: AngleDelta,
}

impl DirectionalMode {
    /// Create a new directional mode.
    #[must_use]
    pub const fn new(angle: u16) -> Self {
        Self {
            angle,
            delta: AngleDelta::Zero,
        }
    }

    /// Create a new directional mode with delta.
    #[must_use]
    pub const fn with_delta(angle: u16, delta: AngleDelta) -> Self {
        Self { angle, delta }
    }

    /// Get the effective angle including delta adjustment.
    #[must_use]
    pub const fn effective_angle(self) -> i16 {
        self.angle as i16 + self.delta.degrees()
    }

    /// Check if this is a vertical-ish direction (45-135 degrees).
    #[must_use]
    pub const fn is_vertical_ish(self) -> bool {
        let angle = self.effective_angle();
        angle > 45 && angle < 135
    }

    /// Check if this is a horizontal-ish direction (135-225 degrees).
    #[must_use]
    pub const fn is_horizontal_ish(self) -> bool {
        let angle = self.effective_angle();
        angle > 135 && angle < 225
    }
}

impl Default for DirectionalMode {
    fn default() -> Self {
        Self::new(90) // Vertical
    }
}

/// Angle delta for AV1 directional modes.
///
/// AV1 allows fine-tuning directional prediction angles by -3 to +3 steps
/// of 3 degrees each, giving a range of -9 to +9 degrees.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(i8)]
pub enum AngleDelta {
    /// -3 steps (-9 degrees)
    Minus3 = -3,
    /// -2 steps (-6 degrees)
    Minus2 = -2,
    /// -1 step (-3 degrees)
    Minus1 = -1,
    /// No delta (0 degrees)
    #[default]
    Zero = 0,
    /// +1 step (+3 degrees)
    Plus1 = 1,
    /// +2 steps (+6 degrees)
    Plus2 = 2,
    /// +3 steps (+9 degrees)
    Plus3 = 3,
}

impl AngleDelta {
    /// Angle step size in degrees.
    pub const STEP_DEGREES: i16 = 3;

    /// Convert to degrees offset.
    #[must_use]
    pub const fn degrees(self) -> i16 {
        (self as i8 as i16) * Self::STEP_DEGREES
    }

    /// Create from step count (-3 to +3).
    #[must_use]
    pub const fn from_steps(steps: i8) -> Option<Self> {
        match steps {
            -3 => Some(Self::Minus3),
            -2 => Some(Self::Minus2),
            -1 => Some(Self::Minus1),
            0 => Some(Self::Zero),
            1 => Some(Self::Plus1),
            2 => Some(Self::Plus2),
            3 => Some(Self::Plus3),
            _ => None,
        }
    }

    /// Convert to step count.
    #[must_use]
    pub const fn steps(self) -> i8 {
        self as i8
    }
}

/// Non-directional prediction modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NonDirectionalMode {
    /// DC prediction.
    #[default]
    Dc,
    /// Smooth prediction (bilinear interpolation).
    Smooth,
    /// Smooth vertical only.
    SmoothV,
    /// Smooth horizontal only.
    SmoothH,
    /// Paeth prediction (adaptive).
    Paeth,
}

impl NonDirectionalMode {
    /// Convert from IntraMode.
    #[must_use]
    pub const fn from_intra_mode(mode: IntraMode) -> Option<Self> {
        match mode {
            IntraMode::Dc => Some(Self::Dc),
            IntraMode::Smooth => Some(Self::Smooth),
            IntraMode::SmoothV => Some(Self::SmoothV),
            IntraMode::SmoothH => Some(Self::SmoothH),
            IntraMode::Paeth => Some(Self::Paeth),
            _ => None,
        }
    }
}

/// Mode angle table for directional prediction.
///
/// Maps mode indices to base angles. Used for VP9 compatibility.
pub const MODE_ANGLE_TABLE: [u16; 8] = [
    90,  // V (Vertical)
    180, // H (Horizontal)
    45,  // D45
    135, // D135
    113, // D113 / D117 in VP9
    157, // D157 / D153 in VP9
    203, // D203 / D207 in VP9
    67,  // D67 / D63 in VP9
];

/// VP9 directional mode mapping (VP9 uses different naming).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vp9DirectionalMode {
    /// Vertical (V_PRED)
    Vertical = 0,
    /// Horizontal (H_PRED)
    Horizontal = 1,
    /// D45 prediction
    D45 = 2,
    /// D135 prediction
    D135 = 3,
    /// D117 prediction (similar to AV1 D113)
    D117 = 4,
    /// D153 prediction (similar to AV1 D157)
    D153 = 5,
    /// D207 prediction (similar to AV1 D203)
    D207 = 6,
    /// D63 prediction (similar to AV1 D67)
    D63 = 7,
}

impl Vp9DirectionalMode {
    /// Get angle in degrees.
    #[must_use]
    pub const fn angle(self) -> u16 {
        match self {
            Self::Vertical => 90,
            Self::Horizontal => 180,
            Self::D45 => 45,
            Self::D135 => 135,
            Self::D117 => 117,
            Self::D153 => 153,
            Self::D207 => 207,
            Self::D63 => 63,
        }
    }

    /// Convert to generic DirectionalMode.
    #[must_use]
    pub const fn to_directional(self) -> DirectionalMode {
        DirectionalMode::new(self.angle())
    }
}

/// Extended angle table for AV1 with deltas.
/// Format: base_angle + delta * 3
/// Range: 0 to 270 degrees with 3-degree granularity.
pub const EXTENDED_ANGLES: [i16; 7] = [
    -9, // delta = -3
    -6, // delta = -2
    -3, // delta = -1
    0,  // delta = 0
    3,  // delta = +1
    6,  // delta = +2
    9,  // delta = +3
];

/// Calculate the effective angle given base angle and delta.
#[must_use]
pub const fn calculate_effective_angle(base_angle: u16, delta: AngleDelta) -> i16 {
    base_angle as i16 + delta.degrees()
}

/// Get the dx/dy values for directional prediction.
///
/// Returns (dx, dy) where each value is in 1/256ths of a pixel.
/// Used for sub-pixel interpolation in directional modes.
#[must_use]
pub fn get_direction_delta(angle: i16) -> (i32, i32) {
    // Angle 0 = top-left diagonal
    // Angle 90 = vertical (straight down)
    // Angle 180 = horizontal (straight right)
    // Angle 270 = top-right diagonal

    // Convert angle to radians for calculation
    let radians = (angle as f64) * std::f64::consts::PI / 180.0;

    // Calculate dx and dy in 1/256ths
    // Note: In video codec coordinates, y increases downward
    let dx = (radians.sin() * 256.0).round() as i32;
    let dy = (radians.cos() * 256.0).round() as i32;

    (dx, dy)
}

/// Precomputed dx values for angles 0-90 (in 1/256ths of a pixel).
/// Index by angle.
pub const DX_TABLE: [i32; 91] = {
    let mut table = [0i32; 91];
    let mut i = 0;
    while i < 91 {
        // sin(angle) * 256
        // Using integer approximation for const fn
        table[i] = match i {
            0 => 0,
            15 => 66,
            30 => 128,
            45 => 181,
            60 => 222,
            75 => 247,
            90 => 256,
            _ => {
                // Linear interpolation between known values
                let base = (i / 15) * 15;
                let next = base + 15;
                if next > 90 {
                    256
                } else {
                    let base_val = match base {
                        0 => 0,
                        15 => 66,
                        30 => 128,
                        45 => 181,
                        60 => 222,
                        75 => 247,
                        _ => 0,
                    };
                    let next_val = match next {
                        15 => 66,
                        30 => 128,
                        45 => 181,
                        60 => 222,
                        75 => 247,
                        90 => 256,
                        _ => 256,
                    };
                    base_val + ((next_val - base_val) * ((i - base) as i32)) / 15
                }
            }
        };
        i += 1;
    }
    table
};

/// Precomputed dy values for angles 0-90 (in 1/256ths of a pixel).
pub const DY_TABLE: [i32; 91] = {
    let mut table = [0i32; 91];
    let mut i = 0;
    while i < 91 {
        // cos(angle) * 256
        table[i] = match i {
            0 => 256,
            15 => 247,
            30 => 222,
            45 => 181,
            60 => 128,
            75 => 66,
            90 => 0,
            _ => {
                let base = (i / 15) * 15;
                let next = base + 15;
                if next > 90 {
                    0
                } else {
                    let base_val = match base {
                        0 => 256,
                        15 => 247,
                        30 => 222,
                        45 => 181,
                        60 => 128,
                        75 => 66,
                        _ => 0,
                    };
                    let next_val = match next {
                        15 => 247,
                        30 => 222,
                        45 => 181,
                        60 => 128,
                        75 => 66,
                        90 => 0,
                        _ => 0,
                    };
                    base_val + ((next_val - base_val) * ((i - base) as i32)) / 15
                }
            }
        };
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intra_mode_directional() {
        assert!(IntraMode::Vertical.is_directional());
        assert!(IntraMode::Horizontal.is_directional());
        assert!(IntraMode::D45.is_directional());
        assert!(IntraMode::D135.is_directional());
        assert!(!IntraMode::Dc.is_directional());
        assert!(!IntraMode::Smooth.is_directional());
        assert!(!IntraMode::Paeth.is_directional());
    }

    #[test]
    fn test_intra_mode_smooth() {
        assert!(IntraMode::Smooth.is_smooth());
        assert!(IntraMode::SmoothV.is_smooth());
        assert!(IntraMode::SmoothH.is_smooth());
        assert!(!IntraMode::Dc.is_smooth());
        assert!(!IntraMode::Vertical.is_smooth());
    }

    #[test]
    fn test_angle_delta() {
        assert_eq!(AngleDelta::Zero.degrees(), 0);
        assert_eq!(AngleDelta::Plus1.degrees(), 3);
        assert_eq!(AngleDelta::Plus3.degrees(), 9);
        assert_eq!(AngleDelta::Minus1.degrees(), -3);
        assert_eq!(AngleDelta::Minus3.degrees(), -9);
    }

    #[test]
    fn test_directional_mode() {
        let mode = DirectionalMode::new(90);
        assert_eq!(mode.effective_angle(), 90);
        assert!(mode.is_vertical_ish());
        assert!(!mode.is_horizontal_ish());

        let mode_with_delta = DirectionalMode::with_delta(90, AngleDelta::Plus2);
        assert_eq!(mode_with_delta.effective_angle(), 96);
    }

    #[test]
    fn test_vp9_directional_mode() {
        assert_eq!(Vp9DirectionalMode::Vertical.angle(), 90);
        assert_eq!(Vp9DirectionalMode::Horizontal.angle(), 180);
        assert_eq!(Vp9DirectionalMode::D45.angle(), 45);
        assert_eq!(Vp9DirectionalMode::D117.angle(), 117);
    }

    #[test]
    fn test_intra_mode_from_u8() {
        assert_eq!(IntraMode::from_u8(0), Some(IntraMode::Dc));
        assert_eq!(IntraMode::from_u8(1), Some(IntraMode::Vertical));
        assert_eq!(IntraMode::from_u8(12), Some(IntraMode::Paeth));
        assert_eq!(IntraMode::from_u8(100), None);
    }

    #[test]
    fn test_nominal_angles() {
        assert_eq!(IntraMode::Vertical.nominal_angle(), Some(90));
        assert_eq!(IntraMode::Horizontal.nominal_angle(), Some(180));
        assert_eq!(IntraMode::D45.nominal_angle(), Some(45));
        assert_eq!(IntraMode::Dc.nominal_angle(), None);
    }

    #[test]
    fn test_angle_delta_from_steps() {
        assert_eq!(AngleDelta::from_steps(0), Some(AngleDelta::Zero));
        assert_eq!(AngleDelta::from_steps(3), Some(AngleDelta::Plus3));
        assert_eq!(AngleDelta::from_steps(-3), Some(AngleDelta::Minus3));
        assert_eq!(AngleDelta::from_steps(4), None);
        assert_eq!(AngleDelta::from_steps(-4), None);
    }
}

//! Mathematical utilities for color management.
//!
//! This module provides matrix operations, interpolation, and other
//! mathematical functions needed for color transformations.

pub mod chromatic_adaptation;
pub mod interpolation;
pub mod matrix;

pub use chromatic_adaptation::{bradford_transform, von_kries_transform, ChromaticAdaptation};
pub use interpolation::{lerp, tetrahedral_interpolate, trilinear_interpolate};
pub use matrix::{invert_matrix_3x3, multiply_matrix_vector, Matrix3, Matrix3x3};

/// Linear interpolation between two values.
///
/// # Arguments
///
/// * `a` - Start value
/// * `b` - End value
/// * `t` - Interpolation factor [0, 1]
#[must_use]
#[inline]
pub fn lerp_f64(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Clamps a value between 0 and 1.
#[must_use]
#[inline]
pub fn clamp_01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// Clamps an RGB triplet to [0, 1] range.
#[must_use]
#[inline]
pub fn clamp_rgb(rgb: [f64; 3]) -> [f64; 3] {
    [clamp_01(rgb[0]), clamp_01(rgb[1]), clamp_01(rgb[2])]
}

/// Converts degrees to radians.
#[must_use]
#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

/// Converts radians to degrees.
#[must_use]
#[inline]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp_f64() {
        assert!((lerp_f64(0.0, 1.0, 0.5) - 0.5).abs() < 1e-10);
        assert!((lerp_f64(0.0, 1.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((lerp_f64(0.0, 1.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clamp_01() {
        assert_eq!(clamp_01(-0.5), 0.0);
        assert_eq!(clamp_01(0.5), 0.5);
        assert_eq!(clamp_01(1.5), 1.0);
    }

    #[test]
    fn test_clamp_rgb() {
        let rgb = clamp_rgb([-0.1, 0.5, 1.2]);
        assert_eq!(rgb, [0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_angle_conversions() {
        assert!((deg_to_rad(180.0) - std::f64::consts::PI).abs() < 1e-10);
        assert!((rad_to_deg(std::f64::consts::PI) - 180.0).abs() < 1e-10);
    }
}

//! sRGB → CIE Lab colour-space conversion helpers for camera calibration.
//!
//! This module provides the end-to-end perceptual conversion pipeline used to
//! evaluate camera calibration accuracy:
//!
//! 1. **sRGB → linear**: piecewise gamma decode (IEC 61966-2-1).
//! 2. **linear RGB → CIE XYZ**: ITU-R BT.709 / sRGB primaries, D65 white.
//! 3. **CIE XYZ → CIE L\*a\*b\***: D65 reference white.
//!
//! These helpers are deliberately small and dependency-free so that perceptual
//! ΔE error metrics can be computed from the gamma-encoded RGB patch values
//! exposed by [`crate::camera::ColorChecker`].

use crate::{Lab, Rgb, Xyz};

/// sRGB → linear-light transfer function (IEC 61966-2-1, single channel).
///
/// `c` is a gamma-encoded sRGB component in `[0, 1]`. Returns the corresponding
/// linear-light value. The piecewise threshold is `0.04045`.
#[must_use]
#[inline]
pub fn srgb_to_linear_channel(c: f64) -> f64 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Decode a gamma-encoded sRGB triplet to linear-light RGB.
#[must_use]
pub fn srgb_to_linear(rgb: Rgb) -> Rgb {
    [
        srgb_to_linear_channel(rgb[0]),
        srgb_to_linear_channel(rgb[1]),
        srgb_to_linear_channel(rgb[2]),
    ]
}

/// Convert linear-light sRGB / BT.709 RGB to CIE XYZ (D65 reference white).
///
/// Uses the standard sRGB → XYZ matrix derived from the BT.709 primaries and
/// the D65 white point.
#[must_use]
pub fn linear_rgb_to_xyz(rgb: Rgb) -> Xyz {
    // sRGB (BT.709) → XYZ, D65 white point.
    let x = rgb[0] * 0.412_456_4 + rgb[1] * 0.357_576_1 + rgb[2] * 0.180_437_5;
    let y = rgb[0] * 0.212_672_9 + rgb[1] * 0.715_152_2 + rgb[2] * 0.072_175_0;
    let z = rgb[0] * 0.019_333_9 + rgb[1] * 0.119_192_0 + rgb[2] * 0.950_304_1;
    [x, y, z]
}

/// CIE XYZ → CIE L\*a\*b\* conversion (D65 reference white).
///
/// Reference white: D65 (`Xn = 0.95047`, `Yn = 1.0`, `Zn = 1.08883`).
#[must_use]
pub fn xyz_to_lab(xyz: Xyz) -> Lab {
    const XN: f64 = 0.950_47;
    const YN: f64 = 1.000_00;
    const ZN: f64 = 1.088_83;

    let f = |t: f64| -> f64 {
        const DELTA: f64 = 6.0 / 29.0;
        const DELTA2: f64 = DELTA * DELTA;
        const DELTA3: f64 = DELTA * DELTA * DELTA;
        if t > DELTA3 {
            t.cbrt()
        } else {
            t / (3.0 * DELTA2) + 4.0 / 29.0
        }
    };

    let fx = f(xyz[0] / XN);
    let fy = f(xyz[1] / YN);
    let fz = f(xyz[2] / ZN);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    [l, a, b]
}

/// Full pipeline: gamma-encoded sRGB → CIE L\*a\*b\* (D65 white).
///
/// Equivalent to [`srgb_to_linear`] → [`linear_rgb_to_xyz`] → [`xyz_to_lab`].
#[must_use]
pub fn srgb_to_lab(rgb: Rgb) -> Lab {
    xyz_to_lab(linear_rgb_to_xyz(srgb_to_linear(rgb)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_to_linear_endpoints() {
        assert!((srgb_to_linear_channel(0.0) - 0.0).abs() < 1e-12);
        assert!((srgb_to_linear_channel(1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_srgb_to_linear_low_segment_is_linear() {
        // Below the 0.04045 threshold the transfer is a pure division by 12.92.
        let c = 0.02_f64;
        assert!((srgb_to_linear_channel(c) - c / 12.92).abs() < 1e-12);
    }

    #[test]
    fn test_srgb_to_linear_monotonic() {
        let lo = srgb_to_linear_channel(0.25);
        let hi = srgb_to_linear_channel(0.75);
        assert!(hi > lo, "transfer must be monotonically increasing");
    }

    #[test]
    fn test_linear_white_maps_to_d65() {
        // Linear white (1,1,1) → XYZ should equal the D65 white point.
        let xyz = linear_rgb_to_xyz([1.0, 1.0, 1.0]);
        assert!((xyz[0] - 0.950_47).abs() < 1e-3, "X={}", xyz[0]);
        assert!((xyz[1] - 1.0).abs() < 1e-3, "Y={}", xyz[1]);
        assert!((xyz[2] - 1.088_83).abs() < 1e-3, "Z={}", xyz[2]);
    }

    #[test]
    fn test_srgb_white_is_l100_neutral() {
        // sRGB white → Lab should be L*≈100, a*≈0, b*≈0.
        let lab = srgb_to_lab([1.0, 1.0, 1.0]);
        assert!((lab[0] - 100.0).abs() < 0.1, "L*={}", lab[0]);
        assert!(lab[1].abs() < 0.5, "a*={}", lab[1]);
        assert!(lab[2].abs() < 0.5, "b*={}", lab[2]);
    }

    #[test]
    fn test_srgb_black_is_l0() {
        let lab = srgb_to_lab([0.0, 0.0, 0.0]);
        assert!(
            lab[0].abs() < 1e-6,
            "L* of black should be 0, got {}",
            lab[0]
        );
    }

    #[test]
    fn test_neutral_gray_is_achromatic() {
        // A neutral gray patch should have near-zero a* and b*.
        let lab = srgb_to_lab([0.5, 0.5, 0.5]);
        assert!(lab[1].abs() < 0.5, "a*={} should be ~0 for neutral", lab[1]);
        assert!(lab[2].abs() < 0.5, "b*={} should be ~0 for neutral", lab[2]);
        assert!(lab[0] > 0.0 && lab[0] < 100.0, "L*={} mid-range", lab[0]);
    }
}

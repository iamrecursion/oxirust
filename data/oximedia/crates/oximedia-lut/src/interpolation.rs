//! Interpolation methods for LUT lookups.
//!
//! This module provides various interpolation algorithms used for LUT application:
//!
//! - **Linear**: Simple linear interpolation (1D)
//! - **Cubic**: Cubic spline interpolation (1D)
//! - **Trilinear**: 3D linear interpolation
//! - **Tetrahedral**: 4-point tetrahedral interpolation (highest quality for 3D)
//!
//! Tetrahedral interpolation is generally preferred for 3D LUTs as it provides
//! better quality than trilinear with similar performance.

use crate::Rgb;

/// Interpolation method for LUT lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LutInterpolation {
    /// Nearest neighbor (no interpolation).
    Nearest,
    /// Linear interpolation.
    Linear,
    /// Trilinear interpolation (3D only).
    #[default]
    Trilinear,
    /// Tetrahedral interpolation (3D only, highest quality).
    Tetrahedral,
    /// Cubic interpolation (1D only).
    Cubic,
}

/// Linear interpolation between two values.
#[must_use]
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Linear interpolation between two RGB colors.
#[must_use]
#[inline]
pub fn lerp_rgb(a: &Rgb, b: &Rgb, t: f64) -> Rgb {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

/// Bilinear interpolation (2D).
#[must_use]
#[inline]
pub fn bilerp(c00: f64, c10: f64, c01: f64, c11: f64, tx: f64, ty: f64) -> f64 {
    let c0 = lerp(c00, c10, tx);
    let c1 = lerp(c01, c11, tx);
    lerp(c0, c1, ty)
}

/// Trilinear interpolation (3D).
///
/// Interpolates between 8 corner values of a cube.
#[must_use]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::similar_names)]
pub fn trilerp(
    c000: f64,
    c100: f64,
    c010: f64,
    c110: f64,
    c001: f64,
    c101: f64,
    c011: f64,
    c111: f64,
    tx: f64,
    ty: f64,
    tz: f64,
) -> f64 {
    let c00 = lerp(c000, c100, tx);
    let c10 = lerp(c010, c110, tx);
    let c01 = lerp(c001, c101, tx);
    let c11 = lerp(c011, c111, tx);

    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);

    lerp(c0, c1, tz)
}

/// Trilinear interpolation for RGB colors.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn trilerp_rgb(
    c000: &Rgb,
    c100: &Rgb,
    c010: &Rgb,
    c110: &Rgb,
    c001: &Rgb,
    c101: &Rgb,
    c011: &Rgb,
    c111: &Rgb,
    tx: f64,
    ty: f64,
    tz: f64,
) -> Rgb {
    [
        trilerp(
            c000[0], c100[0], c010[0], c110[0], c001[0], c101[0], c011[0], c111[0], tx, ty, tz,
        ),
        trilerp(
            c000[1], c100[1], c010[1], c110[1], c001[1], c101[1], c011[1], c111[1], tx, ty, tz,
        ),
        trilerp(
            c000[2], c100[2], c010[2], c110[2], c001[2], c101[2], c011[2], c111[2], tx, ty, tz,
        ),
    ]
}

/// Tetrahedral interpolation for RGB colors.
///
/// This method provides higher quality than trilinear interpolation by using
/// only 4 points instead of 8, which better preserves color relationships.
///
/// The 3D cube is divided into 6 tetrahedra, and we interpolate within the
/// appropriate tetrahedron based on the position.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn tetrahedral_interp(
    c000: &Rgb,
    c100: &Rgb,
    c010: &Rgb,
    c110: &Rgb,
    c001: &Rgb,
    c101: &Rgb,
    c011: &Rgb,
    c111: &Rgb,
    tx: f64,
    ty: f64,
    tz: f64,
) -> Rgb {
    // Determine which tetrahedron we're in based on coordinate order

    if tx >= ty {
        if ty >= tz {
            // Tetrahedron 0: tx >= ty >= tz
            tetrahedral_interp_impl(c000, c100, c110, c111, tx, ty, tz)
        } else if tx >= tz {
            // Tetrahedron 1: tx >= tz >= ty
            tetrahedral_interp_impl(c000, c100, c101, c111, tx, tz, ty)
        } else {
            // Tetrahedron 2: tz >= tx >= ty
            tetrahedral_interp_impl(c000, c001, c101, c111, tz, tx, ty)
        }
    } else if ty >= tz {
        if tx >= tz {
            // Tetrahedron 3: ty >= tx >= tz
            tetrahedral_interp_impl(c000, c010, c110, c111, ty, tx, tz)
        } else {
            // Tetrahedron 4: ty >= tz >= tx
            tetrahedral_interp_impl(c000, c010, c011, c111, ty, tz, tx)
        }
    } else {
        // Tetrahedron 5: tz >= ty >= tx
        tetrahedral_interp_impl(c000, c001, c011, c111, tz, ty, tx)
    }
}

/// Helper function for tetrahedral interpolation.
///
/// Interpolates within a single tetrahedron defined by 4 vertices.
/// The coordinates are assumed to be ordered (a >= b >= c).
#[must_use]
#[inline]
fn tetrahedral_interp_impl(v0: &Rgb, v1: &Rgb, v2: &Rgb, v3: &Rgb, a: f64, b: f64, c: f64) -> Rgb {
    let w0 = 1.0 - a;
    let w1 = a - b;
    let w2 = b - c;
    let w3 = c;

    [
        v0[0] * w0 + v1[0] * w1 + v2[0] * w2 + v3[0] * w3,
        v0[1] * w0 + v1[1] * w1 + v2[1] * w2 + v3[1] * w3,
        v0[2] * w0 + v1[2] * w1 + v2[2] * w2 + v3[2] * w3,
    ]
}

/// Cubic hermite spline interpolation.
///
/// Uses 4 control points for smooth interpolation.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn cubic_interp(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;

    let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let c = -0.5 * p0 + 0.5 * p2;
    let d = p1;

    a * t3 + b * t2 + c * t + d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 1.0, 0.0) - 0.0).abs() < f64::EPSILON);
        assert!((lerp(0.0, 1.0, 1.0) - 1.0).abs() < f64::EPSILON);
        assert!((lerp(0.0, 1.0, 0.5) - 0.5).abs() < f64::EPSILON);
        assert!((lerp(10.0, 20.0, 0.25) - 12.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lerp_rgb() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];
        let result = lerp_rgb(&a, &b, 0.5);
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 0.5).abs() < f64::EPSILON);
        assert!((result[2] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bilerp() {
        // Test corners
        assert!((bilerp(0.0, 1.0, 2.0, 3.0, 0.0, 0.0) - 0.0).abs() < f64::EPSILON);
        assert!((bilerp(0.0, 1.0, 2.0, 3.0, 1.0, 0.0) - 1.0).abs() < f64::EPSILON);
        assert!((bilerp(0.0, 1.0, 2.0, 3.0, 0.0, 1.0) - 2.0).abs() < f64::EPSILON);
        assert!((bilerp(0.0, 1.0, 2.0, 3.0, 1.0, 1.0) - 3.0).abs() < f64::EPSILON);
        // Test center
        assert!((bilerp(0.0, 1.0, 2.0, 3.0, 0.5, 0.5) - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trilerp() {
        // Test corners
        assert!(
            (trilerp(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0) - 0.0).abs()
                < f64::EPSILON
        );
        assert!(
            (trilerp(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 1.0, 1.0) - 7.0).abs()
                < f64::EPSILON
        );
        // Test center
        assert!(
            (trilerp(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.5, 0.5, 0.5) - 3.5).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_tetrahedral_vs_trilinear() {
        let c000 = [0.0, 0.0, 0.0];
        let c100 = [1.0, 0.0, 0.0];
        let c010 = [0.0, 1.0, 0.0];
        let c110 = [1.0, 1.0, 0.0];
        let c001 = [0.0, 0.0, 1.0];
        let c101 = [1.0, 0.0, 1.0];
        let c011 = [0.0, 1.0, 1.0];
        let c111 = [1.0, 1.0, 1.0];

        // At corners, both methods should give the same result
        let tri = trilerp_rgb(
            &c000, &c100, &c010, &c110, &c001, &c101, &c011, &c111, 0.0, 0.0, 0.0,
        );
        let tet = tetrahedral_interp(
            &c000, &c100, &c010, &c110, &c001, &c101, &c011, &c111, 0.0, 0.0, 0.0,
        );
        assert!((tri[0] - tet[0]).abs() < 1e-10);
        assert!((tri[1] - tet[1]).abs() < 1e-10);
        assert!((tri[2] - tet[2]).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_interp() {
        // At t=0, should return p1
        assert!((cubic_interp(0.0, 1.0, 2.0, 3.0, 0.0) - 1.0).abs() < 1e-10);
        // At t=1, should return p2
        assert!((cubic_interp(0.0, 1.0, 2.0, 3.0, 1.0) - 2.0).abs() < 1e-10);
    }
}

//! Tetrahedral 3D LUT interpolation for calibration workflows.
//!
//! This module wraps the `oximedia-lut` tetrahedral interpolation primitive
//! with a convenient, calibration-oriented API.  The `TetrahedralInterpolator`
//! struct provides a stable interface for comparing tetrahedral against
//! trilinear interpolation and for use in downstream calibration checks.
//!
//! # Algorithm
//!
//! The unit cube around each sample is decomposed into one of six tetrahedra
//! based on the relative ordering of the fractional offsets `(dr, dg, db)`.
//! Barycentric interpolation within the chosen tetrahedron uses four LUT
//! lattice vertices, avoiding the over-smoothing artefacts that can appear
//! with trilinear interpolation near chromatic boundaries.
//!
//! Reference:
//! Kennel, G. (2006). *Color and Mastering for Digital Cinema*.
//! Focal Press. Chapter on Look-Up Tables.

use oximedia_lut::{Lut3d, LutInterpolation};

// ---------------------------------------------------------------------------
// TetrahedralInterpolator
// ---------------------------------------------------------------------------

/// High-quality tetrahedral interpolation wrapper for 3D calibration LUTs.
///
/// Provides a stable, calibration-oriented API on top of the underlying
/// `oximedia-lut` tetrahedral implementation.
///
/// # Example
///
/// ```
/// use oximedia_calibrate::lut::tetrahedral::TetrahedralInterpolator;
/// use oximedia_lut::{Lut3d, LutSize};
///
/// let lut = Lut3d::identity(LutSize::Size33);
/// let out = TetrahedralInterpolator::interpolate(&lut, 0.5, 0.3, 0.7);
/// ```
pub struct TetrahedralInterpolator;

impl TetrahedralInterpolator {
    /// Interpolate a colour through a 3D LUT using tetrahedral interpolation.
    ///
    /// Input `r`, `g`, `b` values are clamped to `[0.0, 1.0]` before
    /// the LUT is applied.
    ///
    /// # Arguments
    ///
    /// * `lut` - The `Lut3d` to sample.
    /// * `r`   - Red channel in `[0.0, 1.0]`.
    /// * `g`   - Green channel in `[0.0, 1.0]`.
    /// * `b`   - Blue channel in `[0.0, 1.0]`.
    ///
    /// # Returns
    ///
    /// `[f32; 3]` output colour in the LUT's output domain.
    #[must_use]
    pub fn interpolate(lut: &Lut3d, r: f32, g: f32, b: f32) -> [f32; 3] {
        let rgb = [
            f64::from(r.clamp(0.0, 1.0)),
            f64::from(g.clamp(0.0, 1.0)),
            f64::from(b.clamp(0.0, 1.0)),
        ];
        let out = lut.apply(&rgb, LutInterpolation::Tetrahedral);
        [out[0] as f32, out[1] as f32, out[2] as f32]
    }

    /// Interpolate using `f64` precision throughout.
    ///
    /// # Arguments
    ///
    /// * `lut` - The `Lut3d` to sample.
    /// * `r`   - Red channel in `[0.0, 1.0]`.
    /// * `g`   - Green channel in `[0.0, 1.0]`.
    /// * `b`   - Blue channel in `[0.0, 1.0]`.
    ///
    /// # Returns
    ///
    /// `[f64; 3]` output colour.
    #[must_use]
    pub fn interpolate_f64(lut: &Lut3d, r: f64, g: f64, b: f64) -> [f64; 3] {
        let rgb = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
        lut.apply(&rgb, LutInterpolation::Tetrahedral)
    }

    /// Trilinear interpolation for baseline comparison.
    ///
    /// Useful for verifying accuracy of tetrahedral against the simpler
    /// trilinear method.
    ///
    /// # Arguments
    ///
    /// * `lut` - The `Lut3d` to sample.
    /// * `r`   - Red channel in `[0.0, 1.0]`.
    /// * `g`   - Green channel in `[0.0, 1.0]`.
    /// * `b`   - Blue channel in `[0.0, 1.0]`.
    ///
    /// # Returns
    ///
    /// `[f32; 3]` output colour using trilinear interpolation.
    #[must_use]
    pub fn interpolate_trilinear(lut: &Lut3d, r: f32, g: f32, b: f32) -> [f32; 3] {
        let rgb = [
            f64::from(r.clamp(0.0, 1.0)),
            f64::from(g.clamp(0.0, 1.0)),
            f64::from(b.clamp(0.0, 1.0)),
        ];
        let out = lut.apply(&rgb, LutInterpolation::Trilinear);
        [out[0] as f32, out[1] as f32, out[2] as f32]
    }

    /// Apply tetrahedral interpolation to a batch of pixels in-place.
    ///
    /// # Arguments
    ///
    /// * `lut`    - The `Lut3d` to sample.
    /// * `pixels` - Mutable slice of `[f32; 3]` pixels.
    pub fn apply_batch(lut: &Lut3d, pixels: &mut [[f32; 3]]) {
        for px in pixels.iter_mut() {
            *px = Self::interpolate(lut, px[0], px[1], px[2]);
        }
    }

    /// Compute the maximum channel error between tetrahedral and trilinear
    /// interpolation across a set of test points.
    ///
    /// This is useful for verifying that the two methods agree within an
    /// acceptable tolerance for a given LUT.
    ///
    /// # Arguments
    ///
    /// * `lut`    - The `Lut3d` to test.
    /// * `points` - Slice of `(r, g, b)` test points in `[0.0, 1.0]³`.
    ///
    /// # Returns
    ///
    /// The maximum absolute channel error across all test points and channels.
    #[must_use]
    pub fn max_error_vs_trilinear(lut: &Lut3d, points: &[(f32, f32, f32)]) -> f32 {
        let mut max_err = 0.0_f32;
        for &(r, g, b) in points {
            let tet = Self::interpolate(lut, r, g, b);
            let tri = Self::interpolate_trilinear(lut, r, g, b);
            for ch in 0..3 {
                let err = (tet[ch] - tri[ch]).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        max_err
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_lut::{Lut3d, LutSize};

    fn approx_f32(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn rgb_approx_f32(a: &[f32; 3], b: &[f32; 3], tol: f32) -> bool {
        approx_f32(a[0], b[0], tol) && approx_f32(a[1], b[1], tol) && approx_f32(a[2], b[2], tol)
    }

    // ── 1. Identity LUT – tetrahedral preserves value ────────────────────

    #[test]
    fn test_identity_lut_tetrahedral_midpoint() {
        let lut = Lut3d::identity(LutSize::Size33);
        let out = TetrahedralInterpolator::interpolate(&lut, 0.5, 0.3, 0.7);
        assert!(
            rgb_approx_f32(&out, &[0.5, 0.3, 0.7], 1e-5),
            "Identity LUT should preserve colour: {out:?}"
        );
    }

    // ── 2. Identity LUT – corners ────────────────────────────────────────

    #[test]
    fn test_identity_lut_corners() {
        let lut = Lut3d::identity(LutSize::Size17);
        let corners: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        for c in corners {
            let out = TetrahedralInterpolator::interpolate(&lut, c[0], c[1], c[2]);
            assert!(
                rgb_approx_f32(&out, c, 1e-5),
                "Corner {c:?} should be preserved: {out:?}"
            );
        }
    }

    // ── 3. Clamp out-of-range input ──────────────────────────────────────

    #[test]
    fn test_clamp_below_zero() {
        let lut = Lut3d::identity(LutSize::Size17);
        // Negative inputs should clamp to 0.
        let out = TetrahedralInterpolator::interpolate(&lut, -0.5, 0.5, 0.5);
        assert!(
            out[0] >= 0.0 && out[0] <= 1.0,
            "Clamped R channel out of range: {}",
            out[0]
        );
    }

    #[test]
    fn test_clamp_above_one() {
        let lut = Lut3d::identity(LutSize::Size17);
        let out = TetrahedralInterpolator::interpolate(&lut, 1.5, 0.5, 0.5);
        assert!(
            out[0] <= 1.0 + 1e-5,
            "Clamped R channel should be ≤ 1.0: {}",
            out[0]
        );
    }

    // ── 4. Tetrahedral vs trilinear for identity ─────────────────────────

    #[test]
    fn test_tetrahedral_vs_trilinear_identity_lut() {
        let lut = Lut3d::identity(LutSize::Size33);
        let test_points: &[(f32, f32, f32)] = &[
            (0.1, 0.2, 0.3),
            (0.7, 0.5, 0.9),
            (0.33, 0.66, 0.12),
            (0.5, 0.5, 0.5),
            (0.0, 1.0, 0.5),
        ];
        for &(r, g, b) in test_points {
            let tet = TetrahedralInterpolator::interpolate(&lut, r, g, b);
            let tri = TetrahedralInterpolator::interpolate_trilinear(&lut, r, g, b);
            // On an identity LUT both should agree to high precision.
            assert!(
                rgb_approx_f32(&tet, &tri, 1e-5),
                "tet={tet:?} vs tri={tri:?} for ({r},{g},{b})"
            );
        }
    }

    // ── 5. Non-identity LUT – tetrahedral and trilinear ──────────────────

    #[test]
    fn test_tetrahedral_vs_trilinear_scale_lut() {
        // LUT that doubles R and halves G.
        let lut = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [(rgb[0] * 2.0).min(1.0), rgb[1] * 0.5, rgb[2]]
        });
        // For lattice-aligned points both methods should be exact.
        let out_tet = TetrahedralInterpolator::interpolate(&lut, 0.5, 0.5, 0.5);
        let out_tri = TetrahedralInterpolator::interpolate_trilinear(&lut, 0.5, 0.5, 0.5);
        // Trilinear and tetrahedral may differ slightly at off-grid points;
        // at grid midpoints they should be close.
        assert!(
            (out_tet[0] - out_tri[0]).abs() < 0.05,
            "R: tet={} tri={}",
            out_tet[0],
            out_tri[0]
        );
    }

    // ── 6. max_error_vs_trilinear returns 0 for identity ─────────────────

    #[test]
    fn test_max_error_identity_lut() {
        let lut = Lut3d::identity(LutSize::Size33);
        let points: Vec<(f32, f32, f32)> = vec![(0.1, 0.2, 0.3), (0.5, 0.5, 0.5), (0.9, 0.8, 0.7)];
        let err = TetrahedralInterpolator::max_error_vs_trilinear(&lut, &points);
        assert!(
            err < 1e-5,
            "Max error on identity LUT should be tiny: {err}"
        );
    }

    // ── 7. Batch application consistency ─────────────────────────────────

    #[test]
    fn test_apply_batch_consistent_with_single() {
        let lut = Lut3d::identity(LutSize::Size17);
        let pixel = [0.3_f32, 0.5, 0.2];
        let expected = TetrahedralInterpolator::interpolate(&lut, pixel[0], pixel[1], pixel[2]);
        let mut batch = vec![pixel; 5];
        TetrahedralInterpolator::apply_batch(&lut, &mut batch);
        for out in &batch {
            assert!(
                rgb_approx_f32(out, &expected, 1e-6),
                "Batch output {out:?} != single {expected:?}"
            );
        }
    }

    // ── 8. interpolate_f64 precision ─────────────────────────────────────

    #[test]
    fn test_interpolate_f64_identity() {
        let lut = Lut3d::identity(LutSize::Size33);
        let out = TetrahedralInterpolator::interpolate_f64(&lut, 0.5, 0.3, 0.7);
        assert!((out[0] - 0.5).abs() < 1e-9, "R: {}", out[0]);
        assert!((out[1] - 0.3).abs() < 1e-9, "G: {}", out[1]);
        assert!((out[2] - 0.7).abs() < 1e-9, "B: {}", out[2]);
    }
}

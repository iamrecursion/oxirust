//! Barrel and pincushion lens distortion correction / simulation.
//!
//! Implements the Brown-Conrady radial distortion model which captures
//! barrel (`k1 < 0`) and pincushion (`k1 > 0`) distortion, as well as
//! tangential distortion (`p1`, `p2`).

#![allow(dead_code)]
#![allow(missing_docs)]

/// Radial + tangential lens distortion coefficients (Brown-Conrady model).
#[derive(Debug, Clone)]
pub struct LensDistortionCoeffs {
    /// First radial distortion coefficient.
    pub k1: f32,
    /// Second radial distortion coefficient.
    pub k2: f32,
    /// First tangential distortion coefficient.
    pub p1: f32,
    /// Second tangential distortion coefficient.
    pub p2: f32,
}

impl LensDistortionCoeffs {
    /// Barrel distortion preset (negative `k1`).
    #[must_use]
    pub fn barrel(k1: f32) -> Self {
        Self {
            k1: -k1.abs(),
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Pincushion distortion preset (positive `k1`).
    #[must_use]
    pub fn pincushion(k1: f32) -> Self {
        Self {
            k1: k1.abs(),
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// No distortion (identity coefficients).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}

/// Apply Brown-Conrady radial + tangential distortion to a normalised UV coordinate.
///
/// Input `(u, v)` are in normalised image coordinates centred at (0, 0), with
/// the image extending to ±1 on the shorter axis.  Output is the distorted position.
#[must_use]
pub fn distort_uv(u: f32, v: f32, coeffs: &LensDistortionCoeffs) -> (f32, f32) {
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let radial = 1.0 + coeffs.k1 * r2 + coeffs.k2 * r4;
    let u_d = u * radial + 2.0 * coeffs.p1 * u * v + coeffs.p2 * (r2 + 2.0 * u * u);
    let v_d = v * radial + coeffs.p1 * (r2 + 2.0 * v * v) + 2.0 * coeffs.p2 * u * v;
    (u_d, v_d)
}

/// Undistort a single UV coordinate using Newton-Raphson iteration.
///
/// Approximates the inverse of `distort_uv` to within `tolerance` in 2-norm.
/// Returns the undistorted coordinate or the best estimate after `max_iters`.
#[must_use]
pub fn undistort_uv(
    u_d: f32,
    v_d: f32,
    coeffs: &LensDistortionCoeffs,
    max_iters: usize,
    tolerance: f32,
) -> (f32, f32) {
    let mut u = u_d;
    let mut v = v_d;
    for _ in 0..max_iters {
        let (fu, fv) = distort_uv(u, v, coeffs);
        let eu = fu - u_d;
        let ev = fv - v_d;
        if eu * eu + ev * ev < tolerance * tolerance {
            break;
        }
        // Simple fixed-point iteration: u_next = u_d / radial(u, v)
        let r2 = u * u + v * v;
        let r4 = r2 * r2;
        let radial = 1.0 + coeffs.k1 * r2 + coeffs.k2 * r4;
        let radial = if radial.abs() < 1e-8 { 1e-8 } else { radial };
        u = (u_d - (2.0 * coeffs.p1 * u * v + coeffs.p2 * (r2 + 2.0 * u * u))) / radial;
        v = (v_d - (coeffs.p1 * (r2 + 2.0 * v * v) + 2.0 * coeffs.p2 * u * v)) / radial;
    }
    (u, v)
}

/// Maximum distortion magnitude at the corner of the image for a given coefficient set.
///
/// Checks the four corners `(±1, ±1)` and returns the largest displacement.
#[must_use]
pub fn max_corner_distortion(coeffs: &LensDistortionCoeffs) -> f32 {
    let corners = [(1.0_f32, 1.0_f32), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)];
    corners
        .iter()
        .map(|&(u, v)| {
            let (ud, vd) = distort_uv(u, v, coeffs);
            let du = ud - u;
            let dv = vd - v;
            (du * du + dv * dv).sqrt()
        })
        .fold(0.0_f32, f32::max)
}

/// Apply lens distortion correction to an entire RGBA image (modifies in-place).
///
/// Pixels that map outside `[0, width) × [0, height)` are filled with black.
///
/// # Panics
///
/// Panics if `src` or `dst` does not have length `width * height * channels`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn apply_distortion_correction(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    coeffs: &LensDistortionCoeffs,
    channels: usize,
) {
    assert_eq!(src.len(), width * height * channels);
    assert_eq!(dst.len(), width * height * channels);

    let w = width as f32;
    let h = height as f32;
    let cx = w * 0.5;
    let cy = h * 0.5;
    let scale = cx.min(cy);

    for row in 0..height {
        for col in 0..width {
            // Normalise
            let u_d = (col as f32 - cx) / scale;
            let v_d = (row as f32 - cy) / scale;
            // Undistort to find source pixel
            let (u_s, v_s) = undistort_uv(u_d, v_d, coeffs, 10, 1e-5);
            // Back to pixel coords
            let sx = u_s * scale + cx;
            let sy = v_s * scale + cy;
            let src_col = sx.round() as isize;
            let src_row = sy.round() as isize;
            let dst_idx = (row * width + col) * channels;
            if src_col >= 0 && src_col < width as isize && src_row >= 0 && src_row < height as isize
            {
                let src_idx = (src_row as usize * width + src_col as usize) * channels;
                dst[dst_idx..dst_idx + channels].copy_from_slice(&src[src_idx..src_idx + channels]);
            } else {
                for c in 0..channels {
                    dst[dst_idx + c] = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_distort_uv() {
        let c = LensDistortionCoeffs::identity();
        let (ud, vd) = distort_uv(0.5, 0.3, &c);
        assert!((ud - 0.5).abs() < 1e-6);
        assert!((vd - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_barrel_distort_shrinks_outer() {
        let c = LensDistortionCoeffs::barrel(0.2);
        let (ud, vd) = distort_uv(0.8, 0.6, &c);
        // barrel should pull corners inward
        let orig_r = (0.8_f32 * 0.8 + 0.6 * 0.6).sqrt();
        let dist_r = (ud * ud + vd * vd).sqrt();
        assert!(dist_r < orig_r);
    }

    #[test]
    fn test_pincushion_distort_pushes_outer() {
        let c = LensDistortionCoeffs::pincushion(0.2);
        let (ud, vd) = distort_uv(0.8, 0.6, &c);
        let orig_r = (0.8_f32 * 0.8 + 0.6 * 0.6).sqrt();
        let dist_r = (ud * ud + vd * vd).sqrt();
        assert!(dist_r > orig_r);
    }

    #[test]
    fn test_undistort_roundtrip() {
        let c = LensDistortionCoeffs {
            k1: -0.1,
            k2: 0.01,
            p1: 0.0,
            p2: 0.0,
        };
        let (ud, vd) = distort_uv(0.4, 0.3, &c);
        let (ur, vr) = undistort_uv(ud, vd, &c, 20, 1e-5);
        assert!((ur - 0.4).abs() < 1e-3);
        assert!((vr - 0.3).abs() < 1e-3);
    }

    #[test]
    fn test_identity_undistort() {
        let c = LensDistortionCoeffs::identity();
        let (ur, vr) = undistort_uv(0.5, -0.3, &c, 5, 1e-6);
        assert!((ur - 0.5).abs() < 1e-5);
        assert!((vr + 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_max_corner_distortion_identity() {
        let c = LensDistortionCoeffs::identity();
        let d = max_corner_distortion(&c);
        assert!(d < 1e-5);
    }

    #[test]
    fn test_max_corner_distortion_barrel() {
        let c = LensDistortionCoeffs::barrel(0.2);
        let d = max_corner_distortion(&c);
        assert!(d > 0.0);
    }

    #[test]
    fn test_apply_distortion_correction_identity() {
        let width = 4;
        let height = 4;
        let channels = 4;
        let src: Vec<u8> = (0..(width * height * channels) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        let c = LensDistortionCoeffs::identity();
        apply_distortion_correction(&src, &mut dst, width, height, &c, channels);
        // With identity, central pixels should be preserved
        // (we just check it doesn't panic and dst is not all zeros)
        let non_zero = dst.iter().any(|&b| b > 0);
        assert!(non_zero);
    }

    #[test]
    fn test_barrel_preset_negative_k1() {
        let c = LensDistortionCoeffs::barrel(0.3);
        assert!(c.k1 < 0.0);
    }

    #[test]
    fn test_pincushion_preset_positive_k1() {
        let c = LensDistortionCoeffs::pincushion(0.3);
        assert!(c.k1 > 0.0);
    }

    #[test]
    fn test_distort_origin_unchanged() {
        let c = LensDistortionCoeffs::barrel(0.4);
        let (ud, vd) = distort_uv(0.0, 0.0, &c);
        assert!(ud.abs() < 1e-6);
        assert!(vd.abs() < 1e-6);
    }

    #[test]
    fn test_undistort_zero_origin() {
        let c = LensDistortionCoeffs::barrel(0.4);
        let (ur, vr) = undistort_uv(0.0, 0.0, &c, 5, 1e-6);
        assert!(ur.abs() < 1e-5);
        assert!(vr.abs() < 1e-5);
    }
}

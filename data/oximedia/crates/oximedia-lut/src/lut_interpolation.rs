//! LUT interpolation algorithms: 1D and 3D sampling strategies.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

/// Available interpolation methods for LUT sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Round to the nearest lattice point (lowest quality, fastest).
    Nearest,
    /// Linear interpolation between adjacent samples (1D only).
    Linear,
    /// Trilinear interpolation for 3D LUTs (good balance of quality/speed).
    Trilinear,
    /// Tetrahedral interpolation for 3D LUTs (higher quality than trilinear).
    Tetrahedral,
    /// Prismatic interpolation for 3D LUTs (seldom used).
    Prismatic,
}

impl InterpolationMethod {
    /// Relative quality score (higher = better quality output).
    #[must_use]
    pub fn quality_score(&self) -> u32 {
        match self {
            Self::Nearest => 1,
            Self::Linear => 2,
            Self::Trilinear => 3,
            Self::Tetrahedral => 4,
            Self::Prismatic => 3,
        }
    }
}

/// Sample a 1D LUT using nearest-neighbour interpolation.
///
/// `t` is clamped to `[0, 1]`.  The LUT must not be empty.
///
/// # Panics
///
/// Panics if `lut` is empty.
#[must_use]
pub fn sample_1d_nearest(lut: &[f32], t: f32) -> f32 {
    assert!(!lut.is_empty(), "LUT must not be empty");
    let t = t.clamp(0.0, 1.0);
    let n = lut.len();
    let idx = (t * (n - 1) as f32).round() as usize;
    lut[idx.min(n - 1)]
}

/// Sample a 1D LUT using linear interpolation.
///
/// `t` is clamped to `[0, 1]`.  The LUT must have at least two entries.
///
/// # Panics
///
/// Panics if `lut` has fewer than 2 entries.
#[must_use]
pub fn sample_1d_linear(lut: &[f32], t: f32) -> f32 {
    assert!(lut.len() >= 2, "LUT must have at least 2 entries");
    let t = t.clamp(0.0, 1.0);
    let n = lut.len();
    let pos = t * (n - 1) as f32;
    let lo = (pos as usize).min(n - 2);
    let hi = lo + 1;
    let frac = pos - lo as f32;
    lut[lo] * (1.0 - frac) + lut[hi] * frac
}

// Internal helpers -----------------------------------------------------------------

/// Index into a flattened 3D LUT array stored as R-major (B outer, G mid, R inner).
///
/// `data` has `size * size * size * 3` elements in the order
/// `[b][g][r][channel]`.
#[inline]
fn idx3(size: usize, r: usize, g: usize, b: usize, ch: usize) -> usize {
    ((b * size + g) * size + r) * 3 + ch
}

/// Clamp a float coordinate to `[0, size-1]` and split into floor index + fraction.
#[inline]
fn split_coord(v: f32, size: usize) -> (usize, f32) {
    let scaled = v.clamp(0.0, 1.0) * (size - 1) as f32;
    let lo = (scaled as usize).min(size - 2);
    (lo, scaled - lo as f32)
}

// ---------------------------------------------------------------------------------

/// Sample a 3D LUT using the specified method.
///
/// `data` is a flattened array of `size * size * size * 3` f32 values stored in
/// `[b][g][r]` lattice order, where each triplet is `(R, G, B)` output values.
///
/// `r`, `g`, `b` are input coordinates in `[0, 1]`.
///
/// Returns `(out_r, out_g, out_b)`.
///
/// # Panics
///
/// Panics if `data.len() != size * size * size * 3` or `size < 2`.
#[must_use]
pub fn sample_3d_lut(
    data: &[f32],
    size: usize,
    r: f32,
    g: f32,
    b: f32,
    method: InterpolationMethod,
) -> (f32, f32, f32) {
    assert!(size >= 2, "LUT size must be at least 2");
    assert_eq!(data.len(), size * size * size * 3, "data length mismatch");

    match method {
        InterpolationMethod::Nearest => sample_3d_nearest(data, size, r, g, b),
        InterpolationMethod::Trilinear | InterpolationMethod::Prismatic => {
            sample_3d_trilinear(data, size, r, g, b)
        }
        InterpolationMethod::Tetrahedral => sample_3d_tetrahedral(data, size, r, g, b),
        InterpolationMethod::Linear => sample_3d_trilinear(data, size, r, g, b),
    }
}

fn sample_3d_nearest(data: &[f32], size: usize, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let ri = ((r.clamp(0.0, 1.0) * (size - 1) as f32).round() as usize).min(size - 1);
    let gi = ((g.clamp(0.0, 1.0) * (size - 1) as f32).round() as usize).min(size - 1);
    let bi = ((b.clamp(0.0, 1.0) * (size - 1) as f32).round() as usize).min(size - 1);
    let base = idx3(size, ri, gi, bi, 0);
    (data[base], data[base + 1], data[base + 2])
}

fn sample_3d_trilinear(data: &[f32], size: usize, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (r0, fr) = split_coord(r, size);
    let (g0, fg) = split_coord(g, size);
    let (b0, fb) = split_coord(b, size);
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    let mut out = [0.0_f32; 3];
    for ch in 0..3 {
        let c000 = data[idx3(size, r0, g0, b0, ch)];
        let c100 = data[idx3(size, r1, g0, b0, ch)];
        let c010 = data[idx3(size, r0, g1, b0, ch)];
        let c110 = data[idx3(size, r1, g1, b0, ch)];
        let c001 = data[idx3(size, r0, g0, b1, ch)];
        let c101 = data[idx3(size, r1, g0, b1, ch)];
        let c011 = data[idx3(size, r0, g1, b1, ch)];
        let c111 = data[idx3(size, r1, g1, b1, ch)];

        let c00 = c000 * (1.0 - fr) + c100 * fr;
        let c01 = c001 * (1.0 - fr) + c101 * fr;
        let c10 = c010 * (1.0 - fr) + c110 * fr;
        let c11 = c011 * (1.0 - fr) + c111 * fr;

        let c0 = c00 * (1.0 - fg) + c10 * fg;
        let c1 = c01 * (1.0 - fg) + c11 * fg;

        out[ch] = c0 * (1.0 - fb) + c1 * fb;
    }
    (out[0], out[1], out[2])
}

fn sample_3d_tetrahedral(data: &[f32], size: usize, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (r0, fr) = split_coord(r, size);
    let (g0, fg) = split_coord(g, size);
    let (b0, fb) = split_coord(b, size);
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    // Tetrahedral decomposition of the unit cube.
    let mut out = [0.0_f32; 3];
    for ch in 0..3 {
        let c000 = data[idx3(size, r0, g0, b0, ch)];
        let c111 = data[idx3(size, r1, g1, b1, ch)];

        let v = if fr >= fg && fg >= fb {
            // Tetrahedron 1: fr >= fg >= fb
            let c100 = data[idx3(size, r1, g0, b0, ch)];
            let c110 = data[idx3(size, r1, g1, b0, ch)];
            (1.0 - fr) * c000 + (fr - fg) * c100 + (fg - fb) * c110 + fb * c111
        } else if fr >= fb && fb >= fg {
            // Tetrahedron 2: fr >= fb >= fg
            let c100 = data[idx3(size, r1, g0, b0, ch)];
            let c101 = data[idx3(size, r1, g0, b1, ch)];
            (1.0 - fr) * c000 + (fr - fb) * c100 + (fb - fg) * c101 + fg * c111
        } else if fg >= fr && fr >= fb {
            // Tetrahedron 3: fg >= fr >= fb
            let c010 = data[idx3(size, r0, g1, b0, ch)];
            let c110 = data[idx3(size, r1, g1, b0, ch)];
            (1.0 - fg) * c000 + (fg - fr) * c010 + (fr - fb) * c110 + fb * c111
        } else if fg >= fb && fb >= fr {
            // Tetrahedron 4: fg >= fb >= fr
            let c010 = data[idx3(size, r0, g1, b0, ch)];
            let c011 = data[idx3(size, r0, g1, b1, ch)];
            (1.0 - fg) * c000 + (fg - fb) * c010 + (fb - fr) * c011 + fr * c111
        } else if fb >= fr && fr >= fg {
            // Tetrahedron 5: fb >= fr >= fg
            let c001 = data[idx3(size, r0, g0, b1, ch)];
            let c101 = data[idx3(size, r1, g0, b1, ch)];
            (1.0 - fb) * c000 + (fb - fr) * c001 + (fr - fg) * c101 + fg * c111
        } else {
            // Tetrahedron 6: fb >= fg >= fr
            let c001 = data[idx3(size, r0, g0, b1, ch)];
            let c011 = data[idx3(size, r0, g1, b1, ch)];
            (1.0 - fb) * c000 + (fb - fg) * c001 + (fg - fr) * c011 + fr * c111
        };
        out[ch] = v;
    }
    (out[0], out[1], out[2])
}

/// Build a simple identity 3D LUT: `output == input` for each lattice point.
fn identity_lut(size: usize) -> Vec<f32> {
    let mut data = vec![0.0_f32; size * size * size * 3];
    for bi in 0..size {
        for gi in 0..size {
            for ri in 0..size {
                let base = idx3(size, ri, gi, bi, 0);
                data[base] = ri as f32 / (size - 1) as f32;
                data[base + 1] = gi as f32 / (size - 1) as f32;
                data[base + 2] = bi as f32 / (size - 1) as f32;
            }
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- InterpolationMethod ---

    #[test]
    fn test_quality_score_ordering() {
        assert!(
            InterpolationMethod::Nearest.quality_score()
                < InterpolationMethod::Linear.quality_score()
        );
        assert!(
            InterpolationMethod::Trilinear.quality_score()
                < InterpolationMethod::Tetrahedral.quality_score()
        );
    }

    #[test]
    fn test_quality_scores_values() {
        assert_eq!(InterpolationMethod::Nearest.quality_score(), 1);
        assert_eq!(InterpolationMethod::Linear.quality_score(), 2);
        assert_eq!(InterpolationMethod::Trilinear.quality_score(), 3);
        assert_eq!(InterpolationMethod::Tetrahedral.quality_score(), 4);
        assert_eq!(InterpolationMethod::Prismatic.quality_score(), 3);
    }

    // --- sample_1d_nearest ---

    #[test]
    fn test_sample_1d_nearest_at_boundaries() {
        let lut = vec![0.0_f32, 0.5, 1.0];
        assert!((sample_1d_nearest(&lut, 0.0) - 0.0).abs() < 1e-6);
        assert!((sample_1d_nearest(&lut, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_1d_nearest_clamps_out_of_range() {
        let lut = vec![0.0_f32, 0.5, 1.0];
        assert!((sample_1d_nearest(&lut, -0.5) - 0.0).abs() < 1e-6);
        assert!((sample_1d_nearest(&lut, 1.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_1d_nearest_midpoint_snaps() {
        let lut = vec![0.0_f32, 1.0];
        // t=0.49 → 0.49 * 1 = 0.49 → round → 0 → lut[0] = 0.0
        assert!((sample_1d_nearest(&lut, 0.49) - 0.0).abs() < 1e-6);
        // t=0.51 → 0.51 * 1 = 0.51 → round → 1 → lut[1] = 1.0
        assert!((sample_1d_nearest(&lut, 0.51) - 1.0).abs() < 1e-6);
    }

    // --- sample_1d_linear ---

    #[test]
    fn test_sample_1d_linear_at_boundaries() {
        let lut: Vec<f32> = (0..=4).map(|i| i as f32 / 4.0).collect();
        assert!((sample_1d_linear(&lut, 0.0) - 0.0).abs() < 1e-6);
        assert!((sample_1d_linear(&lut, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_1d_linear_midpoint() {
        let lut = vec![0.0_f32, 1.0];
        assert!((sample_1d_linear(&lut, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sample_1d_linear_clamps() {
        let lut = vec![0.0_f32, 1.0];
        assert!((sample_1d_linear(&lut, -1.0) - 0.0).abs() < 1e-6);
        assert!((sample_1d_linear(&lut, 2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_1d_linear_quarter_point() {
        let lut = vec![0.0_f32, 1.0];
        assert!((sample_1d_linear(&lut, 0.25) - 0.25).abs() < 1e-6);
    }

    // --- sample_3d_lut (identity LUT) ---

    fn check_identity(method: InterpolationMethod, r: f32, g: f32, b: f32, tol: f32) {
        let size = 4;
        let data = identity_lut(size);
        let (or_, og, ob) = sample_3d_lut(&data, size, r, g, b, method);
        assert!((or_ - r).abs() < tol, "R mismatch: got {or_}, expected {r}");
        assert!((og - g).abs() < tol, "G mismatch: got {og}, expected {g}");
        assert!((ob - b).abs() < tol, "B mismatch: got {ob}, expected {b}");
    }

    #[test]
    fn test_identity_nearest_corner() {
        check_identity(InterpolationMethod::Nearest, 0.0, 0.0, 0.0, 1e-5);
        check_identity(InterpolationMethod::Nearest, 1.0, 1.0, 1.0, 1e-5);
    }

    #[test]
    fn test_identity_trilinear_corner() {
        check_identity(InterpolationMethod::Trilinear, 0.0, 0.0, 0.0, 1e-5);
        check_identity(InterpolationMethod::Trilinear, 1.0, 1.0, 1.0, 1e-5);
    }

    #[test]
    fn test_identity_trilinear_midpoint() {
        check_identity(InterpolationMethod::Trilinear, 0.5, 0.5, 0.5, 1e-4);
    }

    #[test]
    fn test_identity_tetrahedral_corner() {
        check_identity(InterpolationMethod::Tetrahedral, 0.0, 0.0, 0.0, 1e-5);
        check_identity(InterpolationMethod::Tetrahedral, 1.0, 1.0, 1.0, 1e-5);
    }

    #[test]
    fn test_identity_tetrahedral_midpoint() {
        check_identity(InterpolationMethod::Tetrahedral, 0.5, 0.5, 0.5, 1e-4);
    }

    #[test]
    fn test_identity_nearest_mid() {
        // Nearest won't be exact at non-lattice points, but should be close for small size.
        let size = 4;
        let data = identity_lut(size);
        let (or_, og, ob) = sample_3d_lut(&data, size, 0.5, 0.5, 0.5, InterpolationMethod::Nearest);
        // lattice step = 1/3, nearest point is at 1/3 or 2/3
        assert!((or_ - og).abs() < 1e-5);
        assert!((og - ob).abs() < 1e-5);
    }

    #[test]
    fn test_sample_3d_asymmetric_trilinear() {
        let size = 2;
        let data = identity_lut(size);
        let (or_, og, ob) =
            sample_3d_lut(&data, size, 0.25, 0.5, 0.75, InterpolationMethod::Trilinear);
        assert!((or_ - 0.25).abs() < 1e-5);
        assert!((og - 0.5).abs() < 1e-5);
        assert!((ob - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_sample_3d_asymmetric_tetrahedral() {
        let size = 2;
        let data = identity_lut(size);
        // For an identity LUT, tetrahedral should also give exact results at lattice-aligned points.
        let (or_, og, ob) =
            sample_3d_lut(&data, size, 0.0, 1.0, 0.0, InterpolationMethod::Tetrahedral);
        assert!((or_ - 0.0).abs() < 1e-5);
        assert!((og - 1.0).abs() < 1e-5);
        assert!((ob - 0.0).abs() < 1e-5);
    }
}

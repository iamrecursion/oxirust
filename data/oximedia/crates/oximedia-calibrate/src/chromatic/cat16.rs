//! CAT16 chromatic adaptation transform from CIECAM16.
//!
//! CAT16 is the chromatic adaptation transform used in CIECAM16, offering
//! improved performance over CAT02 for large colour difference predictions and
//! better hue linearity under changing illumination conditions.
//!
//! Reference:
//! Li, C., Li, Z., Wang, Z., Xu, Y., Luo, M. R., Cui, G., ... & Melgosa, M.
//! (2017). Comprehensive colour appearance model (CIECAM16).
//! *Color Research & Application*, 42(6), 703–718.

use crate::{Matrix3x3, Xyz};

// ---------------------------------------------------------------------------
// CAT16 forward matrix  (XYZ → LMS-like space)
// ---------------------------------------------------------------------------
//
// Values taken directly from Table 2 of Li et al. (2017).

/// CAT16 forward transform matrix (XYZ→RGB_CAT16).
pub const CAT16_MATRIX: Matrix3x3 = [
    [0.401_288, 0.650_173, -0.051_461],
    [-0.250_268, 1.204_414, 0.045_854],
    [-0.002_079, 0.048_952, 0.953_127],
];

// ---------------------------------------------------------------------------
// CAT16 inverse matrix  (LMS-like space → XYZ)
// ---------------------------------------------------------------------------
//
// Analytically computed inverse of CAT16_MATRIX.
// Verified by checking M * M_inv ≈ I_3.

/// CAT16 inverse transform matrix (RGB_CAT16→XYZ).
///
/// Computed as the analytical inverse of `CAT16_MATRIX`.
pub const CAT16_INVERSE: Matrix3x3 = [
    [1.862_067_855_087, -1.011_254_630_532, 0.149_186_775_444],
    [0.387_526_543_236, 0.621_447_441_931, -0.008_973_985_168],
    [-0.015_841_498_849, -0.034_122_938_029, 1.049_964_436_878],
];

// ---------------------------------------------------------------------------
// Helper: 3×3 matrix × 3-vector
// ---------------------------------------------------------------------------

/// Apply a 3×3 matrix to a 3-element column vector.
#[inline]
#[must_use]
fn mat_vec(m: &Matrix3x3, v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ---------------------------------------------------------------------------
// Cat16Adapter
// ---------------------------------------------------------------------------

/// Chromatic adaptation using the CIECAM16 CAT16 transform.
///
/// # Example
///
/// ```
/// use oximedia_calibrate::chromatic::cat16::Cat16Adapter;
/// use oximedia_calibrate::Illuminant;
///
/// let src = Illuminant::D65.xyz();
/// let dst = Illuminant::D50.xyz();
/// let adapted = Cat16Adapter::adapt(&[0.95, 1.00, 1.09], &src, &dst);
/// ```
pub struct Cat16Adapter;

impl Cat16Adapter {
    /// Adapt an XYZ colour from `src_white` to `dst_white` using CAT16.
    ///
    /// This applies the full von-Kries-style diagonal model:
    ///
    /// 1. Convert source and destination white-points to CAT16 cone space.
    /// 2. Compute the per-channel gain that maps source to destination white.
    /// 3. Apply that gain to the sample colour in CAT16 space.
    /// 4. Convert back to XYZ.
    ///
    /// All white-point components must be strictly positive; a degenerate
    /// white-point (any component ≤ 0) results in an identity pass-through
    /// for that channel.
    ///
    /// # Arguments
    ///
    /// * `xyz` - Input XYZ colour to adapt (any scale, must match `src_white` scale).
    /// * `src_white` - XYZ white-point of the source illuminant.
    /// * `dst_white` - XYZ white-point of the destination illuminant.
    ///
    /// # Returns
    ///
    /// The adapted XYZ colour under `dst_white`.
    #[must_use]
    pub fn adapt(xyz: &[f64; 3], src_white: &[f64; 3], dst_white: &[f64; 3]) -> [f64; 3] {
        // Forward-transform white-points into CAT16 cone space.
        let src_lms = mat_vec(&CAT16_MATRIX, src_white);
        let dst_lms = mat_vec(&CAT16_MATRIX, dst_white);

        // Per-channel von-Kries gain.
        let gain = [
            if src_lms[0].abs() > f64::EPSILON {
                dst_lms[0] / src_lms[0]
            } else {
                1.0
            },
            if src_lms[1].abs() > f64::EPSILON {
                dst_lms[1] / src_lms[1]
            } else {
                1.0
            },
            if src_lms[2].abs() > f64::EPSILON {
                dst_lms[2] / src_lms[2]
            } else {
                1.0
            },
        ];

        // Forward-transform sample into CAT16 space.
        let lms = mat_vec(&CAT16_MATRIX, xyz);

        // Apply diagonal chromatic adaptation.
        let lms_adapted = [lms[0] * gain[0], lms[1] * gain[1], lms[2] * gain[2]];

        // Back-transform to XYZ.
        mat_vec(&CAT16_INVERSE, &lms_adapted)
    }

    /// Convert an XYZ colour to the CAT16 cone-like response space.
    ///
    /// # Arguments
    ///
    /// * `xyz` - Input XYZ colour.
    ///
    /// # Returns
    ///
    /// The colour in CAT16 (LMS-like) space.
    #[must_use]
    pub fn xyz_to_cat16(xyz: &Xyz) -> [f64; 3] {
        mat_vec(&CAT16_MATRIX, xyz)
    }

    /// Convert a CAT16 (LMS-like) colour back to XYZ.
    ///
    /// # Arguments
    ///
    /// * `lms` - Input CAT16 (LMS-like) colour.
    ///
    /// # Returns
    ///
    /// The colour in XYZ space.
    #[must_use]
    pub fn cat16_to_xyz(lms: &[f64; 3]) -> Xyz {
        mat_vec(&CAT16_INVERSE, lms)
    }

    /// Compute the full CAT16 adaptation matrix for the given source/destination
    /// white-point pair.
    ///
    /// The returned 3×3 matrix `M` satisfies `M * xyz_src_white ≈ xyz_dst_white`
    /// and can be applied to any XYZ colour under `src_white` to obtain its
    /// appearance under `dst_white`.
    ///
    /// # Arguments
    ///
    /// * `src_white` - XYZ of source illuminant.
    /// * `dst_white` - XYZ of destination illuminant.
    ///
    /// # Returns
    ///
    /// A 3×3 adaptation matrix.
    #[must_use]
    pub fn compute_adaptation_matrix(src_white: &[f64; 3], dst_white: &[f64; 3]) -> Matrix3x3 {
        let src_lms = mat_vec(&CAT16_MATRIX, src_white);
        let dst_lms = mat_vec(&CAT16_MATRIX, dst_white);

        let gain = [
            if src_lms[0].abs() > f64::EPSILON {
                dst_lms[0] / src_lms[0]
            } else {
                1.0
            },
            if src_lms[1].abs() > f64::EPSILON {
                dst_lms[1] / src_lms[1]
            } else {
                1.0
            },
            if src_lms[2].abs() > f64::EPSILON {
                dst_lms[2] / src_lms[2]
            } else {
                1.0
            },
        ];

        // Diagonal scaling matrix in CAT16 space.
        let diag: Matrix3x3 = [
            [gain[0], 0.0, 0.0],
            [0.0, gain[1], 0.0],
            [0.0, 0.0, gain[2]],
        ];

        // M = CAT16_INVERSE * diag * CAT16_MATRIX
        let tmp = mat3x3_mul(&diag, &CAT16_MATRIX);
        mat3x3_mul(&CAT16_INVERSE, &tmp)
    }
}

// ---------------------------------------------------------------------------
// 3×3 matrix multiplication
// ---------------------------------------------------------------------------

/// Multiply two 3×3 matrices: result = a × b.
#[must_use]
fn mat3x3_mul(a: &Matrix3x3, b: &Matrix3x3) -> Matrix3x3 {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Illuminant;

    // ── Tolerance helpers ────────────────────────────────────────────────

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn xyz_approx(a: &[f64; 3], b: &[f64; 3], tol: f64) -> bool {
        approx(a[0], b[0], tol) && approx(a[1], b[1], tol) && approx(a[2], b[2], tol)
    }

    // ── 1. White point maps to white point ───────────────────────────────

    #[test]
    fn test_adapt_white_d65_to_d50_maps_white() {
        let d65 = Illuminant::D65.xyz();
        let d50 = Illuminant::D50.xyz();
        let result = Cat16Adapter::adapt(&d65, &d65, &d50);
        // The source white should map to the destination white.
        assert!(
            xyz_approx(&result, &d50, 1e-6),
            "result={result:?}, expected={d50:?}"
        );
    }

    #[test]
    fn test_adapt_white_d50_to_d65_maps_white() {
        let d50 = Illuminant::D50.xyz();
        let d65 = Illuminant::D65.xyz();
        let result = Cat16Adapter::adapt(&d50, &d50, &d65);
        assert!(
            xyz_approx(&result, &d65, 1e-6),
            "result={result:?}, expected={d65:?}"
        );
    }

    // ── 2. Identity: same source and destination ─────────────────────────

    #[test]
    fn test_adapt_same_illuminant_is_identity() {
        let d65 = Illuminant::D65.xyz();
        let color = [0.3127, 0.3290, 0.3583];
        let result = Cat16Adapter::adapt(&color, &d65, &d65);
        // Same-illuminant adaptation should be near-identity; 1e-6 tolerates
        // floating-point accumulation through the forward+inverse matrices.
        assert!(
            xyz_approx(&result, &color, 1e-6),
            "result={result:?}, expected={color:?}"
        );
    }

    // ── 3. Round-trip D65 → D50 → D65 ───────────────────────────────────

    #[test]
    fn test_round_trip_d65_d50_d65() {
        let d65 = Illuminant::D65.xyz();
        let d50 = Illuminant::D50.xyz();
        let original = [0.5, 0.4, 0.3];
        let forward = Cat16Adapter::adapt(&original, &d65, &d50);
        let back = Cat16Adapter::adapt(&forward, &d50, &d65);
        assert!(
            xyz_approx(&back, &original, 1e-9),
            "back={back:?}, original={original:?}"
        );
    }

    // ── 4. Forward/inverse CAT16 round-trip ─────────────────────────────

    #[test]
    fn test_xyz_to_cat16_round_trip() {
        let xyz = [0.4, 0.3, 0.6];
        let lms = Cat16Adapter::xyz_to_cat16(&xyz);
        let back = Cat16Adapter::cat16_to_xyz(&lms);
        assert!(
            xyz_approx(&back, &xyz, 1e-9),
            "back={back:?}, original={xyz:?}"
        );
    }

    // ── 5. CAT16 of equal-energy illuminant ─────────────────────────────

    #[test]
    fn test_xyz_to_cat16_equal_energy() {
        // For equal-energy white E (XYZ = [1,1,1]), all LMS should be positive.
        let e = Illuminant::E.xyz();
        let lms = Cat16Adapter::xyz_to_cat16(&e);
        assert!(lms[0] > 0.0 && lms[1] > 0.0 && lms[2] > 0.0, "lms={lms:?}");
    }

    // ── 6. Non-negativity for typical colours ────────────────────────────

    #[test]
    fn test_adapt_non_negative_output() {
        let d65 = Illuminant::D65.xyz();
        let d50 = Illuminant::D50.xyz();
        // Neutral grey under D65.
        let grey = [0.5 * d65[0], 0.5 * d65[1], 0.5 * d65[2]];
        let result = Cat16Adapter::adapt(&grey, &d65, &d50);
        assert!(
            result[1] >= 0.0,
            "Luminance should be non-negative: {result:?}"
        );
    }

    // ── 7. Adaptation matrix maps white ─────────────────────────────────

    #[test]
    fn test_adaptation_matrix_maps_white() {
        let d65 = Illuminant::D65.xyz();
        let d50 = Illuminant::D50.xyz();
        let m = Cat16Adapter::compute_adaptation_matrix(&d65, &d50);
        // Apply matrix to D65 white.
        let result = [
            m[0][0] * d65[0] + m[0][1] * d65[1] + m[0][2] * d65[2],
            m[1][0] * d65[0] + m[1][1] * d65[1] + m[1][2] * d65[2],
            m[2][0] * d65[0] + m[2][1] * d65[1] + m[2][2] * d65[2],
        ];
        assert!(
            xyz_approx(&result, &d50, 1e-6),
            "result={result:?}, expected={d50:?}"
        );
    }

    // ── 8. D65 → A illuminant round-trip ────────────────────────────────

    #[test]
    fn test_round_trip_d65_a_d65() {
        let d65 = Illuminant::D65.xyz();
        let a = Illuminant::A.xyz();
        let original = [0.2, 0.5, 0.8];
        let forward = Cat16Adapter::adapt(&original, &d65, &a);
        let back = Cat16Adapter::adapt(&forward, &a, &d65);
        assert!(
            xyz_approx(&back, &original, 1e-9),
            "back={back:?}, original={original:?}"
        );
    }

    // ── 9. CAT16 matrix * inverse ≈ identity ────────────────────────────

    #[test]
    fn test_cat16_matrix_times_inverse_is_identity() {
        let m = &CAT16_MATRIX;
        let inv = &CAT16_INVERSE;
        // Compute M * M_inv
        for i in 0..3 {
            for j in 0..3 {
                let val: f64 = (0..3).map(|k| m[i][k] * inv[k][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-4,
                    "M*M_inv[{i}][{j}] = {val}, expected {expected}"
                );
            }
        }
    }

    // ── 10. Zero colour maps to zero ─────────────────────────────────────

    #[test]
    fn test_adapt_zero_colour() {
        let d65 = Illuminant::D65.xyz();
        let d50 = Illuminant::D50.xyz();
        let black = [0.0_f64, 0.0, 0.0];
        let result = Cat16Adapter::adapt(&black, &d65, &d50);
        assert!(
            xyz_approx(&result, &black, 1e-12),
            "Black should map to black: {result:?}"
        );
    }

    // ── 11. Luminance preservation under neutral adaptation ──────────────

    #[test]
    fn test_adapt_d65_d65_preserves_all_channels() {
        let d65 = Illuminant::D65.xyz();
        let test_colors: &[[f64; 3]] = &[
            [0.1, 0.05, 0.15],
            [0.9, 0.85, 0.95],
            [0.5, 0.5, 0.5],
            [0.2, 0.4, 0.6],
        ];
        for &c in test_colors {
            let result = Cat16Adapter::adapt(&c, &d65, &d65);
            assert!(
                xyz_approx(&result, &c, 1e-9),
                "Identity failed for {c:?}: got {result:?}"
            );
        }
    }
}

//! ACES (Academy Color Encoding System) Input Device Transform (IDT) generation.
//!
//! This module provides tools for generating ACES IDTs that transform camera-native
//! RGB values into the ACEScg (AP1) working colour space via a Bradford-adapted
//! 3×3 matrix.
//!
//! # Background
//!
//! The ACES IDT pipeline:
//!
//! 1. Camera-native RGB (linearised) →
//! 2. XYZ (D65) via camera colour matrix →
//! 3. XYZ (D60) via Bradford chromatic adaptation →
//! 4. ACEScg (AP1) via the XYZ→AP1 matrix.
//!
//! References:
//! - SMPTE ST 2065-1:2021 (ACES specification)
//! - S-2014-006 – Derivation of the ACES White Point CIE Chromaticity Coordinates
//! - ACEScg is defined in SMPTE ST 2065-1:2021, Annex B

#![allow(dead_code)]

use crate::Matrix3x3;

// ---------------------------------------------------------------------------
// Constants — AP1 (ACEScg) primaries
// ---------------------------------------------------------------------------

/// XYZ → AP1 (ACEScg) matrix.
///
/// Derived from ACES AP1 primary chromaticities defined in S-2014-004.
/// Columns are AP1 red, green, blue primaries in XYZ; rows are X, Y, Z.
pub const XYZ_TO_AP1: Matrix3x3 = [
    [1.641_023_4, -0.324_803_3, -0.236_408_6],
    [-0.663_662_4, 1.615_300_8, 0.016_756_7],
    [0.011_721_9, -0.008_284_4, 0.988_606_0],
];

/// AP1 → XYZ matrix (analytically computed inverse of `XYZ_TO_AP1`).
pub const AP1_TO_XYZ: Matrix3x3 = [
    [0.662_455_496_937, 0.134_006_798_567, 0.156_143_766_952],
    [0.272_234_247_337, 0.674_095_649_956, 0.053_674_465_568],
    [-0.005_573_443_506, 0.004_059_922_467, 1.010_123_708_657],
];

// ---------------------------------------------------------------------------
// Constants — ACES D60 white point (CIE XYZ, Y=1)
// ---------------------------------------------------------------------------

/// ACES D60 white point (XYZ, Y=1).
///
/// Chromaticity x=0.32168, y=0.33767 → XYZ = [0.95265, 1.0, 1.00882].
pub const ACES_D60_WHITE: [f64; 3] = [0.952_65, 1.0, 1.008_82];

// ---------------------------------------------------------------------------
// Bradford chromatic adaptation (used internally)
// ---------------------------------------------------------------------------

/// Bradford forward matrix (XYZ → LMS).
const BRADFORD: Matrix3x3 = [
    [0.895_1, 0.266_4, -0.161_4],
    [-0.750_2, 1.713_5, 0.036_7],
    [0.038_9, -0.068_5, 1.029_6],
];

/// Bradford inverse matrix (LMS → XYZ).
const BRADFORD_INV: Matrix3x3 = [
    [0.986_993, -0.147_054, 0.159_963],
    [0.432_305, 0.518_360, 0.049_291],
    [-0.008_529, 0.040_043, 0.968_487],
];

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

/// Apply a 3×3 matrix to a 3-vector.
#[inline]
#[must_use]
fn mat_vec(m: &Matrix3x3, v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

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

/// Compute the 3×3 matrix inverse using Cramer's rule.
///
/// Returns `None` if the matrix is singular (determinant ≈ 0).
#[must_use]
fn mat3x3_inv(m: &Matrix3x3) -> Option<Matrix3x3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-15 {
        return None;
    }

    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

// ---------------------------------------------------------------------------
// Bradford chromatic adaptation: src_white → dst_white
// ---------------------------------------------------------------------------

/// Compute a Bradford chromatic adaptation matrix mapping `src_white` to
/// `dst_white`.  Both white-points should be in XYZ (Y=1 normalisation
/// is recommended but not enforced).
///
/// Returns the identity matrix if `src_white` has any component ≤ 0.
#[must_use]
pub fn bradford_cat(src_white: &[f64; 3], dst_white: &[f64; 3]) -> Matrix3x3 {
    let src_lms = mat_vec(&BRADFORD, src_white);
    let dst_lms = mat_vec(&BRADFORD, dst_white);

    if src_lms[0].abs() <= f64::EPSILON
        || src_lms[1].abs() <= f64::EPSILON
        || src_lms[2].abs() <= f64::EPSILON
    {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let scale: Matrix3x3 = [
        [dst_lms[0] / src_lms[0], 0.0, 0.0],
        [0.0, dst_lms[1] / src_lms[1], 0.0],
        [0.0, 0.0, dst_lms[2] / src_lms[2]],
    ];

    let tmp = mat3x3_mul(&scale, &BRADFORD);
    mat3x3_mul(&BRADFORD_INV, &tmp)
}

// ---------------------------------------------------------------------------
// AcesIdtConfig
// ---------------------------------------------------------------------------

/// Configuration for ACES IDT generation.
///
/// `color_matrix` converts linearised camera-native RGB to XYZ under
/// `camera_white` (usually D65 for modern digital cameras).
#[derive(Debug, Clone)]
pub struct AcesIdtConfig {
    /// Camera manufacturer name (e.g. `"Canon"`, `"Sony"`).
    pub camera_make: String,
    /// Camera model name (e.g. `"EOS R5"`, `"α7 IV"`).
    pub camera_model: String,
    /// 3×3 matrix: camera-native RGB → XYZ (row-major, Y-normalised).
    ///
    /// Each row corresponds to one XYZ output channel; each column to one
    /// RGB input channel.  The matrix should be computed under the same
    /// illuminant as `camera_white`.
    pub color_matrix: Matrix3x3,
    /// XYZ white-point of the camera calibration illuminant.
    ///
    /// Defaults to D65: `[0.950_47, 1.0, 1.088_83]`.
    pub camera_white: [f64; 3],
}

impl AcesIdtConfig {
    /// Create a config assuming D65 camera white-point.
    #[must_use]
    pub fn new_d65(
        camera_make: impl Into<String>,
        camera_model: impl Into<String>,
        color_matrix: Matrix3x3,
    ) -> Self {
        Self {
            camera_make: camera_make.into(),
            camera_model: camera_model.into(),
            color_matrix,
            camera_white: [0.950_47, 1.0, 1.088_83], // D65
        }
    }

    /// Create a config with a custom camera white-point.
    #[must_use]
    pub fn new(
        camera_make: impl Into<String>,
        camera_model: impl Into<String>,
        color_matrix: Matrix3x3,
        camera_white: [f64; 3],
    ) -> Self {
        Self {
            camera_make: camera_make.into(),
            camera_model: camera_model.into(),
            color_matrix,
            camera_white,
        }
    }
}

// ---------------------------------------------------------------------------
// AcesIdt — the generated transform
// ---------------------------------------------------------------------------

/// A generated ACES Input Device Transform.
///
/// Encapsulates the combined 3×3 matrix that converts linearised camera-native
/// RGB directly into ACEScg (AP1) values.
///
/// # Pipeline
///
/// ```text
/// camera_rgb ──[combined_matrix]──► ACEScg (AP1)
/// ```
///
/// The `combined_matrix` is:
///
/// ```text
/// XYZ_TO_AP1 × Bradford(cam_white→D60) × color_matrix
/// ```
#[derive(Debug, Clone)]
pub struct AcesIdt {
    /// Descriptive name derived from camera make/model.
    pub name: String,
    /// Combined 3×3 IDT matrix (camera RGB → AP1).
    pub combined_matrix: Matrix3x3,
    /// Intermediate Bradford adaptation matrix (XYZ@cam_white → XYZ@D60).
    pub cat_matrix: Matrix3x3,
}

impl AcesIdt {
    /// Apply the IDT to a single linearised camera-native RGB triplet.
    ///
    /// Input and output values are **not** clamped; negative values are
    /// permitted (they represent out-of-gamut colours in ACEScg).
    ///
    /// # Arguments
    ///
    /// * `rgb` - Linearised camera-native RGB in the range typically [0.0, 1.0].
    ///
    /// # Returns
    ///
    /// The colour in ACEScg (AP1) space.
    #[must_use]
    pub fn apply(&self, rgb: &[f32; 3]) -> [f32; 3] {
        let r = f64::from(rgb[0]);
        let g = f64::from(rgb[1]);
        let b = f64::from(rgb[2]);
        let m = &self.combined_matrix;
        let out_r = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        let out_g = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        let out_b = m[2][0] * r + m[2][1] * g + m[2][2] * b;
        [out_r as f32, out_g as f32, out_b as f32]
    }

    /// Apply the IDT to a single linearised camera-native RGB triplet using
    /// `f64` precision.
    ///
    /// # Arguments
    ///
    /// * `rgb` - Linearised camera-native RGB.
    ///
    /// # Returns
    ///
    /// The colour in ACEScg (AP1) space.
    #[must_use]
    pub fn apply_f64(&self, rgb: &[f64; 3]) -> [f64; 3] {
        mat_vec(&self.combined_matrix, rgb)
    }

    /// Apply the IDT to a batch of pixels in-place.
    ///
    /// # Arguments
    ///
    /// * `pixels` - Slice of `[f32; 3]` linearised camera-native RGB pixels.
    pub fn apply_batch(&self, pixels: &mut [[f32; 3]]) {
        for px in pixels.iter_mut() {
            *px = self.apply(px);
        }
    }
}

// ---------------------------------------------------------------------------
// AcesIdtGenerator
// ---------------------------------------------------------------------------

/// Generator for ACES IDTs.
///
/// # Example
///
/// ```
/// use oximedia_calibrate::aces_calibration::{AcesIdtConfig, AcesIdtGenerator};
///
/// // Identity camera matrix (camera already in XYZ).
/// let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let config = AcesIdtConfig::new_d65("Test", "Camera", identity);
/// let idt = AcesIdtGenerator::generate(&config);
/// // Apply to a neutral grey.
/// let out = idt.apply(&[0.18, 0.18, 0.18]);
/// assert!(out[0] > 0.0);
/// ```
pub struct AcesIdtGenerator;

impl AcesIdtGenerator {
    /// Generate an `AcesIdt` from the given configuration.
    ///
    /// The generation pipeline:
    ///
    /// 1. Bradford CAT: `camera_white` → ACES D60 white.
    /// 2. Compose: `XYZ_TO_AP1 × cat × color_matrix`.
    ///
    /// # Arguments
    ///
    /// * `config` - IDT generation configuration.
    ///
    /// # Returns
    ///
    /// A ready-to-use `AcesIdt`.
    #[must_use]
    pub fn generate(config: &AcesIdtConfig) -> AcesIdt {
        // Step 1: Bradford CAT from camera white to ACES D60.
        let cat = bradford_cat(&config.camera_white, &ACES_D60_WHITE);

        // Step 2: chain matrices: XYZ→AP1 * CAT * (cam_rgb→XYZ)
        let xyz_adapted = mat3x3_mul(&cat, &config.color_matrix);
        let combined = mat3x3_mul(&XYZ_TO_AP1, &xyz_adapted);

        AcesIdt {
            name: format!("{} {}", config.camera_make, config.camera_model),
            combined_matrix: combined,
            cat_matrix: cat,
        }
    }

    /// Generate an `AcesIdt` and also return the intermediate XYZ (after CAT)
    /// for diagnostic purposes.
    ///
    /// # Arguments
    ///
    /// * `config` - IDT generation configuration.
    ///
    /// # Returns
    ///
    /// `(AcesIdt, xyz_matrix)` where `xyz_matrix` is the camera → XYZ @ D60
    /// intermediate matrix.
    #[must_use]
    pub fn generate_with_diagnostics(config: &AcesIdtConfig) -> (AcesIdt, Matrix3x3) {
        let cat = bradford_cat(&config.camera_white, &ACES_D60_WHITE);
        let xyz_d60 = mat3x3_mul(&cat, &config.color_matrix);
        let combined = mat3x3_mul(&XYZ_TO_AP1, &xyz_d60);

        let idt = AcesIdt {
            name: format!("{} {}", config.camera_make, config.camera_model),
            combined_matrix: combined,
            cat_matrix: cat,
        };

        (idt, xyz_d60)
    }

    /// Verify that an `AcesIdt` maps the camera white-point to the ACES D60
    /// white.
    ///
    /// Returns the maximum channel error between the adapted white and D60.
    ///
    /// # Arguments
    ///
    /// * `idt` - The IDT to verify.
    /// * `config` - The config used to generate the IDT.
    ///
    /// # Returns
    ///
    /// Maximum channel absolute error.
    #[must_use]
    pub fn verify_white_point(idt: &AcesIdt, config: &AcesIdtConfig) -> f64 {
        // Camera white in camera-native space.
        // If the camera colour matrix maps cam_white_xyz → actual XYZ white,
        // we need to find the camera-native RGB that corresponds to cam_white.
        // We compute the inverse of `color_matrix` to map XYZ→cam_rgb.
        let cam_rgb_white = match mat3x3_inv(&config.color_matrix) {
            Some(inv) => mat_vec(&inv, &config.camera_white),
            None => return f64::INFINITY,
        };

        // Apply IDT.
        let aces_white = idt.apply_f64(&cam_rgb_white);

        // Expected: AP1 (ACEScg) representation of ACES D60 white.
        // By definition, the ACES D60 white maps to (1, 1, 1) in ACEScg.
        let expected = mat_vec(&XYZ_TO_AP1, &ACES_D60_WHITE);

        (0..3)
            .map(|i| (aces_white[i] - expected[i]).abs())
            .fold(0.0_f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_matrix() -> Matrix3x3 {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── 1. Identity camera matrix (camera already in XYZ) ─────────────

    #[test]
    fn test_generate_identity_camera() {
        let config = AcesIdtConfig::new_d65("Test", "Camera", identity_matrix());
        let idt = AcesIdtGenerator::generate(&config);
        assert_eq!(idt.name, "Test Camera");
        // The combined matrix should not be all zeros.
        let sum: f64 = idt.combined_matrix.iter().flat_map(|r| r.iter()).sum();
        assert!(sum.abs() > 0.1, "combined_matrix should not be trivial");
    }

    // ── 2. White point neutrality ────────────────────────────────────────

    #[test]
    fn test_aces_d60_white_luminance() {
        // D60 Y channel should be 1.0.
        assert!(approx(ACES_D60_WHITE[1], 1.0, 1e-6));
    }

    // ── 3. Apply neutral grey ────────────────────────────────────────────

    #[test]
    fn test_apply_neutral_grey_positive() {
        let config = AcesIdtConfig::new_d65("Test", "Camera", identity_matrix());
        let idt = AcesIdtGenerator::generate(&config);
        let out = idt.apply(&[0.18, 0.18, 0.18]);
        // Neutral grey must have positive luminance.
        assert!(
            out[1] > 0.0,
            "Green channel (luminance proxy) must be positive: {out:?}"
        );
    }

    // ── 4. Apply black → black ───────────────────────────────────────────

    #[test]
    fn test_apply_black_maps_to_zero() {
        let config = AcesIdtConfig::new_d65("Test", "Camera", identity_matrix());
        let idt = AcesIdtGenerator::generate(&config);
        let out = idt.apply(&[0.0, 0.0, 0.0]);
        assert!(
            out[0].abs() < 1e-7 && out[1].abs() < 1e-7 && out[2].abs() < 1e-7,
            "Black should map to near-zero: {out:?}"
        );
    }

    // ── 5. apply_f64 consistency with apply ─────────────────────────────

    #[test]
    fn test_apply_f64_consistent_with_apply_f32() {
        let config = AcesIdtConfig::new_d65("Canon", "EOS R5", identity_matrix());
        let idt = AcesIdtGenerator::generate(&config);
        let rgb = [0.5_f32, 0.4, 0.6];
        let out_f32 = idt.apply(&rgb);
        let out_f64 = idt.apply_f64(&[0.5, 0.4, 0.6]);
        for ch in 0..3 {
            assert!(
                (f64::from(out_f32[ch]) - out_f64[ch]).abs() < 1e-5,
                "ch={ch} mismatch: f32={}, f64={}",
                out_f32[ch],
                out_f64[ch]
            );
        }
    }

    // ── 6. Batch application consistency ─────────────────────────────────

    #[test]
    fn test_apply_batch_consistent() {
        let config = AcesIdtConfig::new_d65("Sony", "α7 IV", identity_matrix());
        let idt = AcesIdtGenerator::generate(&config);
        let pixel = [0.3_f32, 0.5, 0.2];
        let expected = idt.apply(&pixel);
        let mut batch = vec![pixel; 4];
        idt.apply_batch(&mut batch);
        for out in &batch {
            for ch in 0..3 {
                assert!(
                    (out[ch] - expected[ch]).abs() < 1e-6,
                    "Batch mismatch at ch={ch}"
                );
            }
        }
    }

    // ── 7. bradford_cat identity (same white) ────────────────────────────

    #[test]
    fn test_bradford_cat_identity_same_white() {
        let d65 = [0.950_47_f64, 1.0, 1.088_83];
        let cat = bradford_cat(&d65, &d65);
        // Should be near-identity.
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (cat[i][j] - expected).abs() < 1e-6,
                    "cat[{i}][{j}]={} expected {expected}",
                    cat[i][j]
                );
            }
        }
    }

    // ── 8. XYZ_TO_AP1 * AP1_TO_XYZ ≈ identity ──────────────────────────

    #[test]
    fn test_xyz_ap1_round_trip() {
        let prod = mat3x3_mul(&XYZ_TO_AP1, &AP1_TO_XYZ);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                // 1e-6 tolerates floating-point accumulation in 3×3 multiply.
                assert!(
                    (prod[i][j] - expected).abs() < 1e-6,
                    "XYZ_TO_AP1*AP1_TO_XYZ[{i}][{j}]={} expected {expected}",
                    prod[i][j]
                );
            }
        }
    }

    // ── 9. Diagnostics returns intermediate matrix ───────────────────────

    #[test]
    fn test_generate_with_diagnostics_returns_xyz_matrix() {
        let config = AcesIdtConfig::new_d65("Nikon", "Z9", identity_matrix());
        let (idt, xyz_mat) = AcesIdtGenerator::generate_with_diagnostics(&config);
        // Verify the product XYZ_TO_AP1 * xyz_mat ≈ combined_matrix.
        let recon = mat3x3_mul(&XYZ_TO_AP1, &xyz_mat);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recon[i][j] - idt.combined_matrix[i][j]).abs() < 1e-10,
                    "Reconstruction mismatch [{i}][{j}]"
                );
            }
        }
    }

    // ── 10. Custom white point config ────────────────────────────────────

    #[test]
    fn test_custom_white_point_d50() {
        let d50 = [0.964_22_f64, 1.0, 0.825_21];
        let config = AcesIdtConfig::new("Arri", "Alexa Mini LF", identity_matrix(), d50);
        let idt = AcesIdtGenerator::generate(&config);
        // Verify CAT matrix is not identity (D50 ≠ D60).
        let is_identity = (0..3).all(|i| {
            (0..3).all(|j| {
                let expected = if i == j { 1.0 } else { 0.0 };
                (idt.cat_matrix[i][j] - expected).abs() < 1e-4
            })
        });
        assert!(!is_identity, "D50 → D60 CAT should not be identity");
    }
}

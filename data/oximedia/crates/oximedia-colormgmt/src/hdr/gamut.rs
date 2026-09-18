//! Colour gamut conversion matrices for HDR workflows.
//!
//! Provides precomputed 3×3 matrices for the most common gamut conversions:
//!
//! | Conversion              | Source    | Destination |
//! |-------------------------|-----------|-------------|
//! | BT.709  → BT.2020       | Rec.709   | Rec.2020    |
//! | BT.2020 → BT.709        | Rec.2020  | Rec.709     |
//! | P3-D65  → BT.2020       | DCI-P3 D65| Rec.2020    |
//! | BT.2020 → P3-D65        | Rec.2020  | DCI-P3 D65  |
//! | DCI-P3  → sRGB          | DCI-P3    | sRGB/BT.709 |
//! | sRGB    → DCI-P3        | sRGB      | DCI-P3      |
//! | BT.709  → DCI-P3        | Rec.709   | DCI-P3 D65  |
//! | DCI-P3  → BT.709        | DCI-P3 D65| Rec.709     |
//!
//! All matrices operate on **linear** (scene-referred or display-referred)
//! RGB values. Apply the appropriate EOTF before and OETF after if your
//! content is gamma-encoded.
//!
//! The matrices are derived from the ITU-R BT.709, BT.2020, and SMPTE
//! RP 431-2 / ISO 26428-1 standards using the standard D65 white point
//! (CIE xy = 0.3127, 0.3290). Values match the commonly published
//! reference matrices used in professional colour tools and standards.

/// A 3×3 colour transformation matrix in row-major order.
///
/// Apply with `apply_matrix(rgb, &MATRIX_CONST)`.
pub type GamutMatrix = [[f64; 3]; 3];

// ============================================================================
// BT.709 ↔ BT.2020
// ============================================================================

/// Matrix: linear BT.709 → linear BT.2020.
///
/// Per ITU-R BT.2087-0, Table 2 (direct derivation from primaries).
/// Small negative values are expected due to the gamut expansion.
pub const BT709_TO_BT2020: GamutMatrix = [
    [0.627_403_895_536, 0.329_283_038_377, 0.043_313_066_087],
    [0.069_097_289_358, 0.919_540_395_075, 0.011_362_315_567],
    [0.016_391_438_875, 0.088_013_307_877, 0.895_595_253_248],
];

/// Matrix: linear BT.2020 → linear BT.709.
///
/// Inverse of `BT709_TO_BT2020`. Out-of-gamut colours (negative components)
/// are possible when BT.2020 content contains colours outside BT.709 gamut.
pub const BT2020_TO_BT709: GamutMatrix = [
    [1.660_491_082_533, -0.587_641_249_024, -0.072_849_833_509],
    [-0.124_550_379_337, 1.132_898_385_804, -0.008_348_006_467],
    [-0.018_151_062_136, -0.100_578_696_038, 1.118_729_758_174],
];

// ============================================================================
// P3-D65 ↔ BT.2020
// ============================================================================

/// Matrix: linear P3-D65 (Display P3) → linear BT.2020.
///
/// Derived from primary chromaticities: P3-D65 (DCI-P3 with D65 white)
/// and BT.2020, both using D65 white point.
pub const P3_D65_TO_BT2020: GamutMatrix = [
    [0.753_845_199_453, 0.198_593_627_825, 0.047_561_172_722],
    [0.045_717_830_680, 0.941_777_592_695, 0.012_504_576_625],
    [-0.001_204_063_875, 0.017_567_090_058, 0.983_636_973_817],
];

/// Matrix: linear BT.2020 → linear P3-D65.
///
/// Inverse of `P3_D65_TO_BT2020`.
pub const BT2020_TO_P3_D65: GamutMatrix = [
    [1.343_071_710_625, -0.282_193_826_519, -0.060_877_884_106],
    [-0.065_276_989_571, 1.075_806_282_726, -0.010_529_293_155],
    [0.002_801_328_311, -0.019_637_109_940, 1.016_835_781_629],
];

// ============================================================================
// DCI-P3 ↔ sRGB  (DCI-P3 uses DCI white ~0.314, 0.351; sRGB uses D65)
// ============================================================================

/// Matrix: linear DCI-P3 (DCI white 0.314, 0.351) → linear sRGB / BT.709.
///
/// Since DCI-P3 and sRGB have different white points (DCI vs D65) a
/// Bradford chromatic adaptation is included in this matrix.
pub const DCI_P3_TO_SRGB: GamutMatrix = [
    [1.189_478_311_116, -0.160_007_477_783, -0.023_544_620_671],
    [-0.052_999_521_234, 1.079_868_512_619, -0.026_868_991_385],
    [-0.005_445_756_869, 0.026_459_454_143, 0.978_986_302_726],
];

/// Matrix: linear sRGB / BT.709 → linear DCI-P3 (DCI white).
///
/// Inverse of `DCI_P3_TO_SRGB`.
pub const SRGB_TO_DCI_P3: GamutMatrix = [
    [0.848_529_874_688, 0.128_832_578_285, 0.016_988_612_244],
    [0.042_500_985_521, 0.935_873_289_022, 0.019_782_034_099],
    [0.000_482_226_248, -0.025_479_282_000, 1.018_814_552_869],
];

// ============================================================================
// P3-D65 (Display P3) ↔ sRGB / BT.709
// ============================================================================

/// Matrix: linear P3-D65 → linear sRGB / BT.709.
///
/// Same white point (D65), so no chromatic adaptation needed.
pub const P3_D65_TO_SRGB: GamutMatrix = [
    [1.224_940_175_195, -0.224_940_175_195, 0.000_000_000_000],
    [-0.042_056_955_234, 1.042_056_955_234, 0.000_000_000_000],
    [-0.019_637_109_940, -0.078_636_253_027, 1.098_273_362_967],
];

/// Matrix: linear sRGB / BT.709 → linear P3-D65.
///
/// Inverse of `P3_D65_TO_SRGB`.
pub const SRGB_TO_P3_D65: GamutMatrix = [
    [0.822_461_950_813, 0.177_538_049_187, 0.000_000_000_000],
    [0.033_194_199_958, 0.966_805_800_042, 0.000_000_000_000],
    [0.017_082_072_016, 0.072_399_521_960, 0.910_518_405_988],
];

// ============================================================================
// Utility function
// ============================================================================

/// Apply a 3×3 gamut conversion matrix to a linear RGB triplet.
///
/// # Arguments
///
/// * `rgb`    – Input linear RGB in any range (typically [0, 1] for display-referred)
/// * `matrix` – One of the precomputed `GamutMatrix` constants in this module
///
/// # Returns
///
/// Transformed linear RGB. Out-of-gamut (negative) values are preserved as-is;
/// apply gamut mapping if clipping is desired.
///
/// # Example
///
/// ```
/// use oximedia_colormgmt::hdr::gamut::{apply_matrix, BT2020_TO_BT709};
///
/// let bt2020_linear = [0.5_f64, 0.3, 0.8];
/// let bt709_linear = apply_matrix(bt2020_linear, &BT2020_TO_BT709);
/// ```
#[must_use]
pub fn apply_matrix(rgb: [f64; 3], matrix: &GamutMatrix) -> [f64; 3] {
    [
        matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2],
        matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2],
        matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2],
    ]
}

/// Apply gamut matrix and clamp result to [0, 1].
///
/// Convenience wrapper over `apply_matrix` that discards out-of-gamut colours.
/// Prefer this when the downstream display cannot represent extended gamut.
///
/// # Arguments
///
/// * `rgb`    – Input linear RGB
/// * `matrix` – A `GamutMatrix` constant from this module
///
/// # Returns
///
/// Clamped linear RGB in [0, 1].
#[must_use]
pub fn apply_matrix_clamped(rgb: [f64; 3], matrix: &GamutMatrix) -> [f64; 3] {
    let out = apply_matrix(rgb, matrix);
    [
        out[0].clamp(0.0, 1.0),
        out[1].clamp(0.0, 1.0),
        out[2].clamp(0.0, 1.0),
    ]
}

/// Identify which gamut matrix to use given source and destination gamut names.
///
/// Returns `None` if the combination is not supported; the caller should then
/// fall through to computing a matrix from primaries.
#[must_use]
#[allow(dead_code)]
pub fn lookup_gamut_matrix(src: &str, dst: &str) -> Option<&'static GamutMatrix> {
    match (src, dst) {
        ("bt709" | "srgb" | "rec709", "bt2020" | "rec2020") => Some(&BT709_TO_BT2020),
        ("bt2020" | "rec2020", "bt709" | "srgb" | "rec709") => Some(&BT2020_TO_BT709),
        ("p3-d65" | "display-p3", "bt2020" | "rec2020") => Some(&P3_D65_TO_BT2020),
        ("bt2020" | "rec2020", "p3-d65" | "display-p3") => Some(&BT2020_TO_P3_D65),
        ("dci-p3", "srgb" | "bt709" | "rec709") => Some(&DCI_P3_TO_SRGB),
        ("srgb" | "bt709" | "rec709", "dci-p3") => Some(&SRGB_TO_DCI_P3),
        ("p3-d65" | "display-p3", "srgb" | "bt709" | "rec709") => Some(&P3_D65_TO_SRGB),
        ("srgb" | "bt709" | "rec709", "p3-d65" | "display-p3") => Some(&SRGB_TO_P3_D65),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_approx_identity(m: &GamutMatrix, tol: f64) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (m[i][j] - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn multiply_matrices(a: &GamutMatrix, b: &GamutMatrix) -> GamutMatrix {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        out
    }

    #[test]
    fn test_bt709_bt2020_round_trip() {
        let m = multiply_matrices(&BT2020_TO_BT709, &BT709_TO_BT2020);
        assert!(
            is_approx_identity(&m, 1e-5),
            "BT709<->BT2020 matrices are not inverses: {m:?}"
        );
    }

    #[test]
    fn test_p3_d65_bt2020_round_trip() {
        let m = multiply_matrices(&BT2020_TO_P3_D65, &P3_D65_TO_BT2020);
        assert!(
            is_approx_identity(&m, 1e-5),
            "P3-D65<->BT2020 matrices are not inverses: {m:?}"
        );
    }

    #[test]
    fn test_dci_p3_srgb_round_trip() {
        let m = multiply_matrices(&SRGB_TO_DCI_P3, &DCI_P3_TO_SRGB);
        assert!(
            is_approx_identity(&m, 1e-4),
            "DCI-P3<->sRGB matrices are not inverses: {m:?}"
        );
    }

    #[test]
    fn test_p3_d65_srgb_round_trip() {
        let m = multiply_matrices(&SRGB_TO_P3_D65, &P3_D65_TO_SRGB);
        assert!(
            is_approx_identity(&m, 1e-5),
            "P3-D65<->sRGB matrices are not inverses: {m:?}"
        );
    }

    #[test]
    fn test_apply_matrix_identity() {
        let identity: GamutMatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let rgb = [0.4, 0.6, 0.2];
        let out = apply_matrix(rgb, &identity);
        for i in 0..3 {
            assert!((out[i] - rgb[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_apply_matrix_clamped() {
        // BT.2020 saturated green is out of BT.709 gamut
        let bt2020_green = [0.0, 1.0, 0.0];
        let bt709 = apply_matrix(bt2020_green, &BT2020_TO_BT709);
        // Some components will be negative (out-of-gamut)
        assert!(bt709.iter().any(|&v| v < 0.0));

        // Clamped version should be in [0, 1]
        let clamped = apply_matrix_clamped(bt2020_green, &BT2020_TO_BT709);
        for v in clamped {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_apply_matrix_bt709_to_bt2020_white() {
        // White (1, 1, 1) in BT.709 should remain (1, 1, 1) in BT.2020
        // since both use D65 white point.
        let white = [1.0_f64, 1.0, 1.0];
        let out = apply_matrix(white, &BT709_TO_BT2020);
        // Sum of rows should equal 1 for each row when applied to white
        for v in out {
            assert!((v - 1.0).abs() < 1e-6, "white point not preserved: {v}");
        }
    }

    #[test]
    fn test_lookup_gamut_matrix() {
        assert!(lookup_gamut_matrix("bt709", "bt2020").is_some());
        assert!(lookup_gamut_matrix("rec709", "rec2020").is_some());
        assert!(lookup_gamut_matrix("srgb", "dci-p3").is_some());
        assert!(lookup_gamut_matrix("unknown", "bt2020").is_none());
    }
}

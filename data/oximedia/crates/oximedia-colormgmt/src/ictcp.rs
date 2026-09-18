//! ICtCp perceptual color space for HDR content analysis.
//!
//! ICtCp is a color space defined by Dolby (SMPTE ST 2100) designed specifically
//! for HDR and wide color gamut imagery. It provides superior perceptual uniformity
//! compared to IPT, Jzazbz, and CIELab for high luminance levels, making it
//! ideal for HDR content quality analysis and color difference computation.
//!
//! # Overview
//!
//! - **I**: Intensity (lightness), correlates well with perceived brightness
//! - **Ct**: Chroma along the tritan (blue-yellow) axis
//! - **Cp**: Chroma along the protan (red-green) axis
//!
//! The color space uses the PQ (SMPTE ST 2084) transfer function and is defined
//! for absolute luminance levels (in nits / cd/m²).
//!
//! # Reference
//!
//! Luo, M.R. & Pointer, M.R. (2017). "ICtCp Colour Space and Its Compression
//! Performance for High Dynamic Range and Wide Colour Gamut Imagery."
//! SMPTE Motion Imaging Journal, 126(7), 1–10.
//!
//! Dolby Laboratories (2016). "ICtCp – A New Colour Encoding for High Dynamic
//! Range and Wide Colour Gamut Content". SMPTE 2016-01.

// ── PQ transfer function constants (SMPTE ST 2084) ───────────────────────────

/// PQ m1 exponent.
const PQ_M1: f64 = 0.159_301_758_125_0; // 2610 / 16384
/// PQ m2 exponent.
const PQ_M2: f64 = 78.843_75; // 2523 * 128 / 4096
/// PQ c1 offset.
const PQ_C1: f64 = 0.835_937_5; // 3424 / 4096
/// PQ c2 linear term.
const PQ_C2: f64 = 18.851_5625; // 2413 * 128 / 4096
/// PQ c3 correction term.
const PQ_C3: f64 = 18.686_718_75; // 2392 * 128 / 4096

// ── LMS conversion matrices ───────────────────────────────────────────────────

/// Matrix from ICtCp spec: absolute XYZ (D65) → LMS (ICtCp-specific).
///
/// This matrix transforms from XYZ to a modified LMS that works with the PQ EOTF.
/// Reference: SMPTE ST 2100 Table 3.
#[rustfmt::skip]
const M1_XYZ_TO_LMS: [[f64; 3]; 3] = [
    [ 0.3592,  0.6976, -0.0358],
    [-0.1922,  1.1004,  0.0755],
    [ 0.0070,  0.0749,  0.8434],
];

/// Matrix from ICtCp spec: LMS′ (PQ-encoded) → ICtCp.
#[rustfmt::skip]
const M2_LMS_TO_ICTCP: [[f64; 3]; 3] = [
    [0.5,       0.5,       0.0     ],
    [1.613769,  -3.323486, 1.709716],
    [4.378174,  -4.245605, -0.132568],
];

/// Inverse of M2: ICtCp → LMS′ (PQ-encoded).
#[rustfmt::skip]
const M2_INV_ICTCP_TO_LMS: [[f64; 3]; 3] = [
    [1.0,  0.008609037037932,  0.111029625003026],
    [1.0, -0.008609037037932, -0.111029625003026],
    [1.0,  0.560031335710679, -0.320627174625105],
];

/// Inverse of M1: LMS → XYZ (D65).
#[rustfmt::skip]
const M1_INV_LMS_TO_XYZ: [[f64; 3]; 3] = [
    [ 2.070180056695613,  -1.326456876103021,  0.206616006143232],
    [ 0.364988250032238,   0.680467362852235, -0.045421753098554],
    [-0.049595542238981,  -0.049590301901773,  1.187391058932107],
];

// ── Helper functions ──────────────────────────────────────────────────────────

#[inline]
fn mat3_mul_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// PQ OETF (linear nit-luminance → PQ-encoded signal).
#[inline]
fn pq_forward(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let xm1 = x.powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * xm1) / (1.0 + PQ_C3 * xm1)).powf(PQ_M2)
}

/// PQ EOTF (PQ-encoded signal → linear nit-luminance).
#[inline]
fn pq_inverse(y: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }
    let ym2 = y.powf(1.0 / PQ_M2);
    let num = (ym2 - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * ym2;
    if den.abs() < 1e-30 {
        return 0.0;
    }
    (num / den).powf(1.0 / PQ_M1)
}

// ── ICtCp color value ─────────────────────────────────────────────────────────

/// An ICtCp color value.
///
/// - `i`:  Intensity (correlates to perceptual brightness; ≈ 0 for black, ≈ 0.58 at 100 nit white)
/// - `ct`: Tritan (blue-yellow) chroma component
/// - `cp`: Protan (red-green) chroma component
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ICtCp {
    /// Intensity (perceptual brightness)
    pub i: f64,
    /// Tritan chroma axis (blue-yellow)
    pub ct: f64,
    /// Protan chroma axis (red-green)
    pub cp: f64,
}

impl ICtCp {
    /// Creates a new ICtCp color.
    #[must_use]
    pub const fn new(i: f64, ct: f64, cp: f64) -> Self {
        Self { i, ct, cp }
    }

    /// Converts from **absolute** XYZ (D65, in cd/m² units) to ICtCp.
    ///
    /// Input XYZ values should be in absolute luminance (cd/m²), not normalised.
    /// For sRGB white at 100 nit: X ≈ 95.047, Y = 100.0, Z ≈ 108.883.
    ///
    /// # Arguments
    ///
    /// * `x`, `y`, `z` - XYZ tristimulus values in cd/m²
    #[must_use]
    pub fn from_xyz_abs(x: f64, y: f64, z: f64) -> Self {
        // Step 1: XYZ → LMS (absolute)
        // ICtCp uses a specific XYZ-to-LMS matrix where the input luminance
        // is normalised by dividing by 203.0 (reference display white luminance)
        let norm = 203.0_f64;
        let xyz_norm = [x / norm, y / norm, z / norm];
        let lms = mat3_mul_vec(&M1_XYZ_TO_LMS, xyz_norm);

        // Step 2: LMS → LMS′ via PQ OETF (each channel independently)
        let lms_pq = [
            pq_forward(lms[0].max(0.0)),
            pq_forward(lms[1].max(0.0)),
            pq_forward(lms[2].max(0.0)),
        ];

        // Step 3: LMS′ → ICtCp via M2
        let ictcp = mat3_mul_vec(&M2_LMS_TO_ICTCP, lms_pq);

        Self {
            i: ictcp[0],
            ct: ictcp[1],
            cp: ictcp[2],
        }
    }

    /// Converts from **normalised** XYZ (Y = 1.0 for reference white) to ICtCp,
    /// given the peak luminance of the content in nits.
    ///
    /// # Arguments
    ///
    /// * `x`, `y`, `z` - Normalised XYZ (Y = 1.0 for white)
    /// * `peak_nits` - Peak luminance in cd/m² (e.g. 100.0 for SDR, 10000.0 for HDR)
    #[must_use]
    pub fn from_xyz_normalized(x: f64, y: f64, z: f64, peak_nits: f64) -> Self {
        Self::from_xyz_abs(x * peak_nits, y * peak_nits, z * peak_nits)
    }

    /// Converts ICtCp back to absolute XYZ (cd/m²).
    ///
    /// The inverse of [`ICtCp::from_xyz_abs`].
    #[must_use]
    pub fn to_xyz_abs(&self) -> (f64, f64, f64) {
        // Step 1: ICtCp → LMS′ via M2_INV
        let lms_pq = mat3_mul_vec(&M2_INV_ICTCP_TO_LMS, [self.i, self.ct, self.cp]);

        // Step 2: LMS′ → LMS via PQ EOTF
        let lms = [
            pq_inverse(lms_pq[0].clamp(0.0, 1.0)),
            pq_inverse(lms_pq[1].clamp(0.0, 1.0)),
            pq_inverse(lms_pq[2].clamp(0.0, 1.0)),
        ];

        // Step 3: LMS → XYZ (un-normalise by 203)
        let xyz_norm = mat3_mul_vec(&M1_INV_LMS_TO_XYZ, lms);
        let norm = 203.0_f64;
        (xyz_norm[0] * norm, xyz_norm[1] * norm, xyz_norm[2] * norm)
    }

    /// Returns the chroma (colorfulness magnitude).
    #[must_use]
    pub fn chroma(&self) -> f64 {
        (self.ct * self.ct + self.cp * self.cp).sqrt()
    }

    /// Returns the hue angle in degrees [0, 360).
    #[must_use]
    pub fn hue_degrees(&self) -> f64 {
        let mut h = self.cp.atan2(self.ct).to_degrees();
        if h < 0.0 {
            h += 360.0;
        }
        h
    }

    /// Computes the ΔICtCp Euclidean color difference.
    ///
    /// The Euclidean distance in ICtCp space is a perceptually uniform color
    /// difference metric for HDR content. A value < 1 is typically imperceptible.
    #[must_use]
    pub fn delta_ictcp(&self, other: &Self) -> f64 {
        let di = self.i - other.i;
        let dct = self.ct - other.ct;
        let dcp = self.cp - other.cp;
        (di * di + dct * dct + dcp * dcp).sqrt()
    }

    /// Linearly interpolates between two ICtCp colors.
    ///
    /// `t = 0` returns `self`, `t = 1` returns `other`.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            i: self.i + (other.i - self.i) * t,
            ct: self.ct + (other.ct - self.ct) * t,
            cp: self.cp + (other.cp - self.cp) * t,
        }
    }

    /// Returns the cylindrical representation (I, C, h).
    ///
    /// - `i`: Intensity (same as `self.i`)
    /// - `c`: Chroma (magnitude of the ct–cp vector)
    /// - `h_deg`: Hue angle in degrees [0, 360)
    #[must_use]
    pub fn to_ich(&self) -> (f64, f64, f64) {
        (self.i, self.chroma(), self.hue_degrees())
    }
}

// ── ICtCp signal-domain (0–1) convenience API ────────────────────────────────

/// Converts linear light sRGB (D65, Y = 1.0 for 100-nit white) to ICtCp.
///
/// This is a convenience wrapper that applies the linearisation-free path:
/// multiply by 100 nits, convert XYZ via the sRGB matrix, then convert to ICtCp.
///
/// # Arguments
///
/// * `r`, `g`, `b` - Linear light sRGB values in [0, 1] (1.0 = 100 nit white)
#[must_use]
pub fn srgb_linear_to_ictcp(r: f64, g: f64, b: f64) -> ICtCp {
    // sRGB linear → XYZ (D65) matrix (IEC 61966-2-1)
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    // Multiply by 100 nits (SDR reference) for absolute luminance
    ICtCp::from_xyz_abs(x * 100.0, y * 100.0, z * 100.0)
}

/// Converts linear light RGB (BT.2020 primaries, D65, absolute 1.0 = 203 nit reference)
/// to ICtCp per BT.2100.
///
/// The input is expected to be linear scene-referred values normalised so that
/// 1.0 corresponds to the BT.2100 reference display white (203 cd/m²).
///
/// # Arguments
///
/// * `rgb` - `[R, G, B]` linear light BT.2020 or sRGB signal (1.0 = reference white)
///
/// # Returns
///
/// `[I, Ct, Cp]` in ICtCp space
#[must_use]
pub fn linear_rgb_to_ictcp(rgb: [f64; 3]) -> [f64; 3] {
    // BT.2020 linear RGB → XYZ (D65) matrix (ITU-R BT.2020)
    let x = 0.636_958_05 * rgb[0] + 0.144_616_9 * rgb[1] + 0.168_880_97 * rgb[2];
    let y = 0.262_700_21 * rgb[0] + 0.677_998_07 * rgb[1] + 0.059_301_72 * rgb[2];
    let z = 0.000_000_00 * rgb[0] + 0.028_072_69 * rgb[1] + 1.060_985_06 * rgb[2];
    // Multiply by 203 to obtain absolute nit luminance (BT.2100 reference)
    let norm = 203.0;
    let c = ICtCp::from_xyz_abs(x * norm, y * norm, z * norm);
    [c.i, c.ct, c.cp]
}

/// Converts ICtCp back to linear light BT.2020 RGB (1.0 = 203 nit reference).
///
/// # Arguments
///
/// * `ictcp` - `[I, Ct, Cp]` in ICtCp space
///
/// # Returns
///
/// `[R, G, B]` linear light BT.2020 signal (1.0 = reference white)
#[must_use]
pub fn ictcp_to_linear_rgb(ictcp: [f64; 3]) -> [f64; 3] {
    let c = ICtCp::new(ictcp[0], ictcp[1], ictcp[2]);
    let (x_abs, y_abs, z_abs) = c.to_xyz_abs();
    // Divide by 203 to normalise back to reference-white units
    let norm = 203.0;
    let x = x_abs / norm;
    let y = y_abs / norm;
    let z = z_abs / norm;
    // XYZ (D65) → BT.2020 linear RGB (inverse of BT.2020 matrix)
    let r = 1.716_651_19 * x - 0.355_670_78 * y - 0.253_366_79 * z;
    let g = -0.666_684_35 * x + 1.616_481_24 * y + 0.015_768_55 * z;
    let b = 0.017_639_86 * x - 0.042_770_61 * y + 0.942_103_07 * z;
    [r, g, b]
}

/// Computes the ΔICtCp color difference between two linear sRGB colors.
///
/// Both colors should be normalised linear sRGB (1.0 = 100-nit white).
/// Returns a perceptually-scaled difference in ICtCp space.
#[must_use]
pub fn delta_ictcp_srgb(r1: f64, g1: f64, b1: f64, r2: f64, g2: f64, b2: f64) -> f64 {
    let c1 = srgb_linear_to_ictcp(r1, g1, b1);
    let c2 = srgb_linear_to_ictcp(r2, g2, b2);
    c1.delta_ictcp(&c2)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(a: f64, b: f64, tol: f64, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{label}: {a} vs {b}, diff={}",
            (a - b).abs()
        );
    }

    // ── Basic construction ────────────────────────────────────────────────────

    #[test]
    fn test_ictcp_creation() {
        let c = ICtCp::new(0.5, 0.01, -0.02);
        assert_eq!(c.i, 0.5);
        assert_eq!(c.ct, 0.01);
        assert_eq!(c.cp, -0.02);
    }

    // ── Black point ───────────────────────────────────────────────────────────

    #[test]
    fn test_ictcp_black_is_zero() {
        let black = ICtCp::from_xyz_abs(0.0, 0.0, 0.0);
        assert_near(black.i, 0.0, 1e-6, "Black I");
        assert_near(black.ct, 0.0, 1e-6, "Black Ct");
        assert_near(black.cp, 0.0, 1e-6, "Black Cp");
    }

    // ── White point intensity is positive ─────────────────────────────────────

    #[test]
    fn test_ictcp_white_positive_i() {
        // D65 white at 100 nits
        let white = ICtCp::from_xyz_abs(95.047, 100.0, 108.883);
        assert!(white.i > 0.0, "White I should be positive: {}", white.i);
        assert!(
            white.i < 1.0,
            "White I should be < 1 at 100 nits: {}",
            white.i
        );
    }

    // ── HDR white point ───────────────────────────────────────────────────────

    #[test]
    fn test_ictcp_hdr_white() {
        // D65 at 10000 nits. ICtCp I at 10000 nits is near the PQ ceiling.
        // With the normalisation factor of 203 cd/m², 10000/203 ≈ 49.3 which PQ
        // maps to approximately 1.0 in PQ signal, yielding I ≈ 0.5 * (PQ + PQ).
        // Values above 1.0 are expected for super-white inputs.
        let hdr = ICtCp::from_xyz_abs(9504.7, 10000.0, 10888.3);
        assert!(hdr.i > 0.5, "10000-nit I should be > 0.5: {}", hdr.i);
        assert!(hdr.i < 2.0, "10000-nit I should be < 2.0: {}", hdr.i);
    }

    // ── Monotonic intensity with luminance ────────────────────────────────────

    #[test]
    fn test_ictcp_monotonic_i() {
        let dim = ICtCp::from_xyz_abs(9.5047, 10.0, 10.888);
        let mid = ICtCp::from_xyz_abs(47.524, 50.0, 54.442);
        let bright = ICtCp::from_xyz_abs(95.047, 100.0, 108.883);
        assert!(dim.i < mid.i, "dim.i {} < mid.i {}", dim.i, mid.i);
        assert!(mid.i < bright.i, "mid.i {} < bright.i {}", mid.i, bright.i);
    }

    // ── Round-trip accuracy ───────────────────────────────────────────────────

    #[test]
    fn test_ictcp_round_trip_white() {
        let x = 95.047;
        let y = 100.0;
        let z = 108.883;
        let ictcp = ICtCp::from_xyz_abs(x, y, z);
        let (x2, y2, z2) = ictcp.to_xyz_abs();
        assert_near(x2, x, 0.5, "White X round-trip");
        assert_near(y2, y, 0.5, "White Y round-trip");
        assert_near(z2, z, 0.5, "White Z round-trip");
    }

    #[test]
    fn test_ictcp_round_trip_red() {
        let x = 41.24;
        let y = 21.26;
        let z = 1.93;
        let ictcp = ICtCp::from_xyz_abs(x, y, z);
        let (x2, y2, z2) = ictcp.to_xyz_abs();
        assert_near(x2, x, 1.0, "Red X round-trip");
        assert_near(y2, y, 1.0, "Red Y round-trip");
        assert_near(z2, z, 1.0, "Red Z round-trip");
    }

    #[test]
    fn test_ictcp_round_trip_blue() {
        let x = 18.05;
        let y = 7.22;
        let z = 95.05;
        let ictcp = ICtCp::from_xyz_abs(x, y, z);
        let (x2, y2, z2) = ictcp.to_xyz_abs();
        assert_near(x2, x, 1.0, "Blue X round-trip");
        assert_near(y2, y, 1.0, "Blue Y round-trip");
        assert_near(z2, z, 1.0, "Blue Z round-trip");
    }

    // ── Achromatic colors have near-zero chroma ────────────────────────────────

    #[test]
    fn test_ictcp_achromatic_near_zero_chroma() {
        // D65 white should be achromatic
        let white = ICtCp::from_xyz_abs(95.047, 100.0, 108.883);
        assert!(
            white.chroma() < 0.05,
            "White chroma should be near 0: {}",
            white.chroma()
        );
    }

    // ── Chroma and hue ────────────────────────────────────────────────────────

    #[test]
    fn test_ictcp_chroma_positive_for_saturated() {
        // Red (high X, low Z) should have positive chroma
        let red = ICtCp::from_xyz_abs(41.24, 21.26, 1.93);
        assert!(
            red.chroma() > 0.0,
            "Red chroma should be positive: {}",
            red.chroma()
        );
    }

    #[test]
    fn test_ictcp_hue_in_range() {
        for (x, y, z) in [
            (41.24, 21.26, 1.93),
            (35.76, 71.52, 11.92),
            (18.05, 7.22, 95.05),
            (77.0, 92.78, 13.85),
        ] {
            let c = ICtCp::from_xyz_abs(x, y, z);
            let h = c.hue_degrees();
            assert!(
                (0.0..360.0).contains(&h),
                "Hue should be in [0,360): {} for ({},{},{})",
                h,
                x,
                y,
                z
            );
        }
    }

    // ── Delta ICtCp ──────────────────────────────────────────────────────────

    #[test]
    fn test_delta_ictcp_same_color() {
        let c = ICtCp::new(0.5, 0.01, -0.02);
        assert_near(c.delta_ictcp(&c), 0.0, 1e-10, "Same color ΔICtCp");
    }

    #[test]
    fn test_delta_ictcp_symmetry() {
        let c1 = ICtCp::new(0.4, 0.02, -0.01);
        let c2 = ICtCp::new(0.6, -0.01, 0.03);
        assert_near(
            c1.delta_ictcp(&c2),
            c2.delta_ictcp(&c1),
            1e-10,
            "ΔICtCp symmetry",
        );
    }

    #[test]
    fn test_delta_ictcp_positive() {
        let c1 = ICtCp::new(0.4, 0.02, -0.01);
        let c2 = ICtCp::new(0.6, -0.01, 0.03);
        assert!(
            c1.delta_ictcp(&c2) > 0.0,
            "ΔICtCp should be positive for different colors"
        );
    }

    // ── Lerp ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_ictcp_lerp_endpoints() {
        let c1 = ICtCp::new(0.3, 0.05, 0.02);
        let c2 = ICtCp::new(0.7, -0.03, 0.08);
        let at0 = c1.lerp(&c2, 0.0);
        let at1 = c1.lerp(&c2, 1.0);
        assert_near(at0.i, c1.i, 1e-10, "lerp t=0 I");
        assert_near(at1.i, c2.i, 1e-10, "lerp t=1 I");
    }

    #[test]
    fn test_ictcp_lerp_midpoint() {
        let c1 = ICtCp::new(0.2, 0.0, 0.0);
        let c2 = ICtCp::new(0.6, 0.04, 0.0);
        let mid = c1.lerp(&c2, 0.5);
        assert_near(mid.i, 0.4, 1e-10, "lerp midpoint I");
        assert_near(mid.ct, 0.02, 1e-10, "lerp midpoint Ct");
    }

    #[test]
    fn test_ictcp_lerp_clamps_t() {
        let c1 = ICtCp::new(0.3, 0.0, 0.0);
        let c2 = ICtCp::new(0.7, 0.0, 0.0);
        let below = c1.lerp(&c2, -0.5);
        let above = c1.lerp(&c2, 1.5);
        assert_near(below.i, c1.i, 1e-10, "lerp t<0 clamped");
        assert_near(above.i, c2.i, 1e-10, "lerp t>1 clamped");
    }

    // ── Cylindrical ICh ───────────────────────────────────────────────────────

    #[test]
    fn test_ictcp_to_ich() {
        let c = ICtCp::new(0.5, 0.03, 0.04);
        let (i, ch, _h) = c.to_ich();
        assert_near(i, 0.5, 1e-10, "to_ich I");
        assert_near(ch, 0.05, 1e-10, "to_ich C"); // √(0.03²+0.04²) = 0.05
    }

    // ── from_xyz_normalized ───────────────────────────────────────────────────

    #[test]
    fn test_ictcp_from_xyz_normalized_100_nit_white() {
        // D65 normalised, 100 nits
        let c_norm = ICtCp::from_xyz_normalized(0.95047, 1.0, 1.08883, 100.0);
        let c_abs = ICtCp::from_xyz_abs(95.047, 100.0, 108.883);
        assert_near(c_norm.i, c_abs.i, 1e-6, "normalised vs absolute I");
        assert_near(c_norm.ct, c_abs.ct, 1e-6, "normalised vs absolute Ct");
        assert_near(c_norm.cp, c_abs.cp, 1e-6, "normalised vs absolute Cp");
    }

    // ── sRGB linear convenience functions ────────────────────────────────────

    #[test]
    fn test_srgb_linear_black_to_ictcp() {
        let c = srgb_linear_to_ictcp(0.0, 0.0, 0.0);
        assert_near(c.i, 0.0, 1e-6, "sRGB black I");
    }

    #[test]
    fn test_srgb_linear_white_to_ictcp_positive_i() {
        let c = srgb_linear_to_ictcp(1.0, 1.0, 1.0);
        assert!(c.i > 0.0, "sRGB white I should be positive: {}", c.i);
    }

    #[test]
    fn test_delta_ictcp_srgb_identical() {
        let de = delta_ictcp_srgb(0.5, 0.3, 0.2, 0.5, 0.3, 0.2);
        assert_near(de, 0.0, 1e-10, "Identical sRGB ΔICtCp");
    }

    #[test]
    fn test_delta_ictcp_srgb_different() {
        let de = delta_ictcp_srgb(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert!(de > 0.0, "Red vs Blue ΔICtCp should be positive: {de}");
    }

    // ── PQ round-trip ─────────────────────────────────────────────────────────

    #[test]
    fn test_pq_forward_inverse_round_trip() {
        for x in [0.001, 0.01, 0.1, 0.5, 1.0] {
            let y = pq_forward(x);
            let x2 = pq_inverse(y);
            assert_near(x2, x, 1e-8, &format!("PQ round-trip at {x}"));
        }
    }

    #[test]
    fn test_pq_monotonic() {
        let v1 = pq_forward(0.01);
        let v2 = pq_forward(0.1);
        let v3 = pq_forward(0.5);
        assert!(v1 < v2 && v2 < v3, "PQ should be monotonically increasing");
    }
}

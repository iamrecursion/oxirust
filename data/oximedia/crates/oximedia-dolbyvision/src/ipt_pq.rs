//! IPT-PQ color space transforms used in Dolby Vision Profile 5.
//!
//! Implements the full pipeline from BT.2020 RGB to IPT-PQ and back,
//! following the Dolby Vision specification for perceptually uniform
//! color encoding using the ST.2084 PQ transfer function.

// ── PQ Transfer Function Constants ───────────────────────────────────────────

/// M1 exponent for PQ (2610/16384 ≈ 0.1593017578125)
pub const M1: f32 = 2610.0 / 16384.0;

/// M2 exponent for PQ (2523/32 ≈ 78.84375)
pub const M2: f32 = 2523.0 / 32.0;

/// C1 constant for PQ (3424/4096 ≈ 0.8359375)
pub const C1: f32 = 3424.0 / 4096.0;

/// C2 constant for PQ (2413/128 ≈ 18.8515625)
pub const C2: f32 = 2413.0 / 128.0;

/// C3 constant for PQ (2392/128 ≈ 18.6875)
pub const C3: f32 = 2392.0 / 128.0;

// ── PQ Transfer Functions ─────────────────────────────────────────────────────

/// PQ EOTF: convert PQ signal (0–1) to linear light (relative, 0–1 where 1 = 10 000 nits).
///
/// Implements ST.2084 EOTF: `((max(x^(1/M2) - C1, 0)) / (C2 - C3*x^(1/M2)))^(1/M1)`
#[must_use]
#[inline]
pub fn pq_eotf(x: f32) -> f32 {
    let x_clamped = x.clamp(0.0, 1.0);
    let xp = x_clamped.powf(1.0 / M2);
    let numerator = (xp - C1).max(0.0);
    let denominator = (C2 - C3 * xp).max(f32::EPSILON);
    (numerator / denominator).powf(1.0 / M1)
}

/// PQ OETF: convert linear light (relative, 0–1 where 1 = 10 000 nits) to PQ signal (0–1).
///
/// Implements ST.2084 OETF (inverse of EOTF):
/// `((C1 + C2*x^M1) / (1 + C3*x^M1))^M2`
#[must_use]
#[inline]
pub fn pq_oetf(x: f32) -> f32 {
    let x_clamped = x.max(0.0);
    let xm = x_clamped.powf(M1);
    let numerator = C1 + C2 * xm;
    let denominator = 1.0 + C3 * xm;
    (numerator / denominator).powf(M2)
}

// ── RGB to LMS (BT.2020 primaries) ───────────────────────────────────────────

/// Convert BT.2020 linear RGB to LMS (IPT color model matrix), then apply PQ OETF.
///
/// The matrix converts linear BT.2020 RGB to the Hunt-Pointer-Estevez LMS
/// cone response adapted for the Dolby Vision IPT-PQ color space.
///
/// Returns `(l_pq, m_pq, s_pq)` — each channel encoded with PQ OETF.
#[must_use]
pub fn rgb_to_lms_pq(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // BT.2020 RGB → LMS linear (Dolby Vision IPT matrix)
    let l_lin = 0.412_109 * r + 0.523_926 * g + 0.063_964_8 * b;
    let m_lin = 0.166_748 * r + 0.720_459 * g + 0.112_793 * b;
    let s_lin = 0.024_194 * r + 0.075_439 * g + 0.900_366 * b;

    // Apply PQ OETF to each channel
    let l_pq = pq_oetf(l_lin.max(0.0));
    let m_pq = pq_oetf(m_lin.max(0.0));
    let s_pq = pq_oetf(s_lin.max(0.0));

    (l_pq, m_pq, s_pq)
}

/// Convert PQ-encoded LMS back to BT.2020 linear RGB (inverse of `rgb_to_lms_pq`).
///
/// Returns `(r, g, b)` in BT.2020 linear light.
#[must_use]
pub fn lms_pq_to_rgb(l_pq: f32, m_pq: f32, s_pq: f32) -> (f32, f32, f32) {
    // Undo PQ OETF
    let l = pq_eotf(l_pq);
    let m = pq_eotf(m_pq);
    let s = pq_eotf(s_pq);

    // Exact analytical inverse of the Dolby Vision IPT BT.2020 RGB→LMS matrix:
    //   [0.412109, 0.523926, 0.063965]
    //   [0.166748, 0.720459, 0.112793]
    //   [0.024194, 0.075439, 0.900366]
    let r = 3.436_605_4 * l - 2.506_452_4 * m + 0.069_847_7 * s;
    let g = -0.791_314_3 * l + 1.983_589_9 * m - 0.192_276_0 * s;
    let b = -0.026_044_2 * l - 0.098_847_5 * m + 1.124_892_8 * s;

    (r, g, b)
}

// ── LMS PQ → IPT ──────────────────────────────────────────────────────────────

/// Convert PQ-encoded LMS to IPT color space.
///
/// Applies the Dolby Vision IPT matrix:
/// ```text
/// I =  0.4000*l + 0.4000*m + 0.2000*s
/// P =  4.4550*l - 4.8510*m + 0.3960*s
/// T =  0.8056*l + 0.3572*m - 1.1628*s
/// ```
#[must_use]
#[inline]
pub fn lms_pq_to_ipt(l: f32, m: f32, s: f32) -> (f32, f32, f32) {
    let i = 0.4000 * l + 0.4000 * m + 0.2000 * s;
    let p = 4.4550 * l - 4.8510 * m + 0.3960 * s;
    let t = 0.8056 * l + 0.3572 * m - 1.1628 * s;
    (i, p, t)
}

/// Convert IPT back to PQ-encoded LMS (inverse of `lms_pq_to_ipt`).
///
/// Uses the analytically-inverted IPT matrix.
#[must_use]
#[inline]
pub fn ipt_to_lms_pq(i: f32, p: f32, t: f32) -> (f32, f32, f32) {
    // Analytical inverse of the 3x3 IPT matrix above
    let l = 1.000_000 * i + 0.097_569 * p + 0.205_226 * t;
    let m = 1.000_000 * i - 0.113_876 * p + 0.133_218 * t;
    let s = 1.000_000 * i + 0.032_615 * p - 0.676_890 * t;
    (l, m, s)
}

// ── Full Pipeline ─────────────────────────────────────────────────────────────

/// Full forward pipeline: BT.2020 linear RGB → IPT-PQ.
///
/// Steps:
/// 1. BT.2020 RGB → LMS (linear)
/// 2. Apply PQ OETF to each LMS channel
/// 3. LMS-PQ → IPT
///
/// Returns `(I, P, T)` in IPT-PQ color space.
#[must_use]
pub fn rgb_bt2020_to_ipt_pq(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (l_pq, m_pq, s_pq) = rgb_to_lms_pq(r, g, b);
    lms_pq_to_ipt(l_pq, m_pq, s_pq)
}

/// Full inverse pipeline: IPT-PQ → BT.2020 linear RGB.
///
/// Steps:
/// 1. IPT → LMS-PQ
/// 2. Apply PQ EOTF to recover linear LMS
/// 3. LMS → BT.2020 RGB (linear)
///
/// Returns `(r, g, b)` in BT.2020 linear light.
#[must_use]
pub fn ipt_pq_to_rgb_bt2020(i: f32, p: f32, t: f32) -> (f32, f32, f32) {
    let (l_pq, m_pq, s_pq) = ipt_to_lms_pq(i, p, t);
    lms_pq_to_rgb(l_pq, m_pq, s_pq)
}

// ── IptPqPixel ────────────────────────────────────────────────────────────────

/// A single pixel encoded in IPT-PQ color space.
///
/// - `i` (Intensity): achromatic luminance channel.
/// - `p` (Protan): red–green opponent channel.
/// - `t` (Tritan): blue–yellow opponent channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IptPqPixel {
    /// Intensity (achromatic)
    pub i: f32,
    /// Protan (red-green opponent)
    pub p: f32,
    /// Tritan (blue-yellow opponent)
    pub t: f32,
}

impl IptPqPixel {
    /// Create a new IPT-PQ pixel.
    #[must_use]
    pub fn new(i: f32, p: f32, t: f32) -> Self {
        Self { i, p, t }
    }

    /// Chroma magnitude: `sqrt(p^2 + t^2)`.
    #[must_use]
    pub fn chroma_magnitude(&self) -> f32 {
        (self.p * self.p + self.t * self.t).sqrt()
    }

    /// Hue angle in radians: `atan2(t, p)`.
    #[must_use]
    pub fn hue_angle_rad(&self) -> f32 {
        self.t.atan2(self.p)
    }

    /// Hue angle in degrees (0–360).
    #[must_use]
    pub fn hue_angle_deg(&self) -> f32 {
        let deg = self.hue_angle_rad().to_degrees();
        if deg < 0.0 {
            deg + 360.0
        } else {
            deg
        }
    }

    /// Create an `IptPqPixel` from BT.2020 linear RGB.
    #[must_use]
    pub fn from_rgb_bt2020(r: f32, g: f32, b: f32) -> Self {
        let (i, p, t) = rgb_bt2020_to_ipt_pq(r, g, b);
        Self { i, p, t }
    }

    /// Convert back to BT.2020 linear RGB.
    #[must_use]
    pub fn to_rgb_bt2020(&self) -> (f32, f32, f32) {
        ipt_pq_to_rgb_bt2020(self.i, self.p, self.t)
    }

    /// Returns `true` if this is an achromatic pixel (chroma below threshold).
    #[must_use]
    pub fn is_achromatic(&self, threshold: f32) -> bool {
        self.chroma_magnitude() < threshold
    }
}

impl Default for IptPqPixel {
    fn default() -> Self {
        Self {
            i: 0.0,
            p: 0.0,
            t: 0.0,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for roundtrip accuracy tests
    const ROUNDTRIP_TOL: f32 = 1e-4;

    // ── PQ EOTF / OETF ───────────────────────────────────────────────────────

    #[test]
    fn test_pq_oetf_zero_input() {
        let pq = pq_oetf(0.0);
        // At 0 linear light, PQ signal should be C1^M2 (black level)
        assert!(pq >= 0.0 && pq < 0.1, "pq={pq}");
    }

    #[test]
    fn test_pq_oetf_full_white() {
        let pq = pq_oetf(1.0);
        // Linear 1.0 (= 10000 nits) should map to PQ ≈ 1.0
        assert!((pq - 1.0).abs() < 0.001, "pq={pq}");
    }

    #[test]
    fn test_pq_eotf_zero_input() {
        let lin = pq_eotf(0.0);
        assert!(lin >= 0.0 && lin < 0.001, "lin={lin}");
    }

    #[test]
    fn test_pq_eotf_full_white() {
        let lin = pq_eotf(1.0);
        assert!((lin - 1.0).abs() < 0.001, "lin={lin}");
    }

    #[test]
    fn test_pq_roundtrip_midtone() {
        let original = 0.01_f32; // ~100 nits relative
        let pq = pq_oetf(original);
        let recovered = pq_eotf(pq);
        assert!(
            (recovered - original).abs() < ROUNDTRIP_TOL,
            "diff={}",
            (recovered - original).abs()
        );
    }

    #[test]
    fn test_pq_roundtrip_highlight() {
        let original = 0.5_f32;
        let pq = pq_oetf(original);
        let recovered = pq_eotf(pq);
        assert!(
            (recovered - original).abs() < ROUNDTRIP_TOL,
            "diff={}",
            (recovered - original).abs()
        );
    }

    #[test]
    fn test_pq_oetf_monotonic() {
        let values = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0];
        for w in values.windows(2) {
            let a = pq_oetf(w[0]);
            let b = pq_oetf(w[1]);
            assert!(
                b >= a,
                "PQ OETF should be monotonic: pq({})={a} >= pq({})={b}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_pq_constants_values() {
        assert!((M1 - 0.159_301_757_812_5).abs() < 1e-10);
        assert!((M2 - 78.843_75).abs() < 1e-6);
        assert!((C1 - 0.835_937_5).abs() < 1e-8);
        assert!((C2 - 18.851_562_5).abs() < 1e-6);
        assert!((C3 - 18.687_5).abs() < 1e-6);
    }

    #[test]
    fn test_pq_oetf_negative_clamped() {
        let pq = pq_oetf(-0.5);
        // Should give same result as pq_oetf(0.0)
        let pq_zero = pq_oetf(0.0);
        assert!((pq - pq_zero).abs() < 1e-6, "pq={pq}");
    }

    // ── RGB → LMS PQ ─────────────────────────────────────────────────────────

    #[test]
    fn test_rgb_to_lms_pq_neutral_grey() {
        // Neutral grey: r=g=b should yield L≈M≈S (near-neutral chromaticity)
        let (l, m, s) = rgb_to_lms_pq(0.5, 0.5, 0.5);
        // L, M, S should be similar but not identical (small spread expected)
        assert!((l - m).abs() < 0.05, "L={l}, M={m}");
        assert!((l - s).abs() < 0.15, "L={l}, S={s}");
    }

    #[test]
    fn test_rgb_to_lms_pq_zero_black() {
        let (l, m, s) = rgb_to_lms_pq(0.0, 0.0, 0.0);
        // All zero input → all PQ OETF of near-zero → near min PQ
        let min_pq = pq_oetf(0.0);
        assert!((l - min_pq).abs() < 1e-5, "l={l}");
        assert!((m - min_pq).abs() < 1e-5, "m={m}");
        assert!((s - min_pq).abs() < 1e-5, "s={s}");
    }

    #[test]
    fn test_lms_pq_to_rgb_roundtrip_neutral() {
        let (l_pq, m_pq, s_pq) = rgb_to_lms_pq(0.18, 0.18, 0.18);
        let (r, g, b) = lms_pq_to_rgb(l_pq, m_pq, s_pq);
        assert!((r - 0.18).abs() < ROUNDTRIP_TOL, "r={r}");
        assert!((g - 0.18).abs() < ROUNDTRIP_TOL, "g={g}");
        assert!((b - 0.18).abs() < ROUNDTRIP_TOL, "b={b}");
    }

    // ── LMS PQ → IPT ─────────────────────────────────────────────────────────

    #[test]
    fn test_lms_pq_to_ipt_achromatic() {
        // If l == m == s, then P and T should be ~0 (achromatic)
        let (i, p, t) = lms_pq_to_ipt(0.5, 0.5, 0.5);
        assert!(p.abs() < 1e-5, "p={p}");
        assert!(t.abs() < 1e-5, "t={t}");
        assert!((i - 0.5).abs() < 1e-5, "i={i}");
    }

    #[test]
    fn test_ipt_to_lms_roundtrip() {
        let (l0, m0, s0) = (0.3, 0.4, 0.5);
        let (i, p, t) = lms_pq_to_ipt(l0, m0, s0);
        let (l1, m1, s1) = ipt_to_lms_pq(i, p, t);
        assert!((l1 - l0).abs() < 1e-4, "l diff={}", (l1 - l0).abs());
        assert!((m1 - m0).abs() < 1e-4, "m diff={}", (m1 - m0).abs());
        assert!((s1 - s0).abs() < 1e-4, "s diff={}", (s1 - s0).abs());
    }

    // ── Full Pipeline Roundtrip ───────────────────────────────────────────────

    // Full-pipeline tests allow 5e-4 tolerance due to accumulated f32 rounding
    // across three nonlinear transforms (two PQ OETF/EOTF + two 3x3 matrix muls).
    const PIPELINE_TOL: f32 = 5e-4;

    #[test]
    fn test_full_pipeline_roundtrip_neutral_grey() {
        let (r0, g0, b0) = (0.18, 0.18, 0.18);
        let (i, p, t) = rgb_bt2020_to_ipt_pq(r0, g0, b0);
        let (r1, g1, b1) = ipt_pq_to_rgb_bt2020(i, p, t);
        assert!((r1 - r0).abs() < PIPELINE_TOL, "r diff={}", (r1 - r0).abs());
        assert!((g1 - g0).abs() < PIPELINE_TOL, "g diff={}", (g1 - g0).abs());
        assert!((b1 - b0).abs() < PIPELINE_TOL, "b diff={}", (b1 - b0).abs());
    }

    #[test]
    fn test_full_pipeline_roundtrip_bright_red() {
        let (r0, g0, b0) = (0.8, 0.05, 0.05);
        let (i, p, t) = rgb_bt2020_to_ipt_pq(r0, g0, b0);
        let (r1, g1, b1) = ipt_pq_to_rgb_bt2020(i, p, t);
        assert!((r1 - r0).abs() < PIPELINE_TOL, "r diff={}", (r1 - r0).abs());
        assert!((g1 - g0).abs() < PIPELINE_TOL, "g diff={}", (g1 - g0).abs());
        assert!((b1 - b0).abs() < PIPELINE_TOL, "b diff={}", (b1 - b0).abs());
    }

    #[test]
    fn test_full_pipeline_roundtrip_bright_blue() {
        let (r0, g0, b0) = (0.05, 0.1, 0.9);
        let (i, p, t) = rgb_bt2020_to_ipt_pq(r0, g0, b0);
        let (r1, g1, b1) = ipt_pq_to_rgb_bt2020(i, p, t);
        assert!((r1 - r0).abs() < PIPELINE_TOL, "r diff={}", (r1 - r0).abs());
        assert!((g1 - g0).abs() < PIPELINE_TOL, "g diff={}", (g1 - g0).abs());
        assert!((b1 - b0).abs() < PIPELINE_TOL, "b diff={}", (b1 - b0).abs());
    }

    #[test]
    fn test_full_pipeline_roundtrip_black() {
        let (r0, g0, b0) = (0.0, 0.0, 0.0);
        let (i, p, t) = rgb_bt2020_to_ipt_pq(r0, g0, b0);
        let (r1, g1, b1) = ipt_pq_to_rgb_bt2020(i, p, t);
        // Black may not roundtrip exactly due to PQ floor, allow wider tolerance
        assert!((r1 - r0).abs() < 0.01, "r diff={}", (r1 - r0).abs());
        assert!((g1 - g0).abs() < 0.01, "g diff={}", (g1 - g0).abs());
        assert!((b1 - b0).abs() < 0.01, "b diff={}", (b1 - b0).abs());
    }

    #[test]
    fn test_full_pipeline_roundtrip_green() {
        let (r0, g0, b0) = (0.05, 0.7, 0.1);
        let (i, p, t) = rgb_bt2020_to_ipt_pq(r0, g0, b0);
        let (r1, g1, b1) = ipt_pq_to_rgb_bt2020(i, p, t);
        assert!((r1 - r0).abs() < PIPELINE_TOL, "r diff={}", (r1 - r0).abs());
        assert!((g1 - g0).abs() < PIPELINE_TOL, "g diff={}", (g1 - g0).abs());
        assert!((b1 - b0).abs() < PIPELINE_TOL, "b diff={}", (b1 - b0).abs());
    }

    // ── IptPqPixel ────────────────────────────────────────────────────────────

    #[test]
    fn test_ipt_pq_pixel_chroma_magnitude_achromatic() {
        let px = IptPqPixel::new(0.5, 0.0, 0.0);
        assert!((px.chroma_magnitude()).abs() < 1e-6);
    }

    #[test]
    fn test_ipt_pq_pixel_chroma_magnitude_chromatic() {
        let px = IptPqPixel::new(0.5, 3.0, 4.0);
        assert!((px.chroma_magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_ipt_pq_pixel_hue_angle_zero() {
        let px = IptPqPixel::new(0.5, 1.0, 0.0);
        assert!(
            (px.hue_angle_rad()).abs() < 1e-5,
            "hue={}",
            px.hue_angle_rad()
        );
    }

    #[test]
    fn test_ipt_pq_pixel_hue_angle_90deg() {
        let px = IptPqPixel::new(0.5, 0.0, 1.0);
        let expected = std::f32::consts::FRAC_PI_2;
        assert!(
            (px.hue_angle_rad() - expected).abs() < 1e-5,
            "hue={}",
            px.hue_angle_rad()
        );
    }

    #[test]
    fn test_ipt_pq_pixel_hue_angle_deg_positive() {
        let px = IptPqPixel::new(0.5, -1.0, 0.0);
        let deg = px.hue_angle_deg();
        assert!((deg - 180.0).abs() < 0.01, "deg={deg}");
    }

    #[test]
    fn test_ipt_pq_pixel_is_achromatic_true() {
        let px = IptPqPixel::new(0.5, 0.0001, 0.0001);
        assert!(px.is_achromatic(0.001));
    }

    #[test]
    fn test_ipt_pq_pixel_is_achromatic_false() {
        let px = IptPqPixel::new(0.5, 0.5, 0.0);
        assert!(!px.is_achromatic(0.001));
    }

    #[test]
    fn test_ipt_pq_pixel_from_rgb_roundtrip() {
        let (r0, g0, b0) = (0.3, 0.5, 0.7);
        let px = IptPqPixel::from_rgb_bt2020(r0, g0, b0);
        let (r1, g1, b1) = px.to_rgb_bt2020();
        assert!((r1 - r0).abs() < PIPELINE_TOL, "r diff={}", (r1 - r0).abs());
        assert!((g1 - g0).abs() < PIPELINE_TOL, "g diff={}", (g1 - g0).abs());
        assert!((b1 - b0).abs() < PIPELINE_TOL, "b diff={}", (b1 - b0).abs());
    }

    #[test]
    fn test_ipt_pq_pixel_default_is_black() {
        let px = IptPqPixel::default();
        assert_eq!(px.i, 0.0);
        assert_eq!(px.p, 0.0);
        assert_eq!(px.t, 0.0);
    }
}

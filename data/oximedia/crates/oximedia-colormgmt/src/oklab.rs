//! Oklab and Oklch perceptual color spaces.
//!
//! Oklab is a modern perceptual color space by Bjorn Ottosson (2020) designed
//! for image processing and uniform perceptual blending. It provides better
//! perceptual uniformity than CIELAB with simpler math.
//!
//! Oklch is the cylindrical (polar) form, offering intuitive hue/chroma
//! manipulation similar to HSL but with perceptual correctness.
//!
//! Reference: Ottosson, B. (2020). "A perceptual color space for image processing."
//! <https://bottosson.github.io/posts/oklab/>
//!
//! Both spaces operate on linear sRGB (D65) input by default. Use XYZ
//! conversions for other illuminants.

use crate::xyz::Xyz;

// ── Matrices ─────────────────────────────────────────────────────────────────
// Oklab defines two matrices: XYZ→LMS (M1) and cbrt(LMS)→Lab (M2).

/// M1: XYZ (D65) → approximate cone response (LMS).
#[rustfmt::skip]
const M1_XYZ_TO_LMS: [[f64; 3]; 3] = [
    [ 0.8189330101, 0.3618667424, -0.1288597137],
    [ 0.0329845436, 0.9293118715,  0.0361456387],
    [ 0.0482003018, 0.2643662691,  0.6338517070],
];

/// M1 inverse: LMS → XYZ (D65).
#[rustfmt::skip]
const M1_INV_LMS_TO_XYZ: [[f64; 3]; 3] = [
    [ 1.2270138511035211, -0.5577999806518222,  0.2812561489664678],
    [-0.0405801784232806,  1.1122568696168302, -0.0716766786656012],
    [-0.0763812845057069, -0.4214819784180127,  1.5861632204407947],
];

/// M2: LMS^(1/3) → Oklab (L, a, b).
#[rustfmt::skip]
const M2_LMS_TO_LAB: [[f64; 3]; 3] = [
    [ 0.2104542553, 0.7936177850, -0.0040720468],
    [ 1.9779984951,-2.4285922050,  0.4505937099],
    [ 0.0259040371, 0.7827717662, -0.8086757660],
];

/// M2 inverse: Oklab → LMS^(1/3).
#[rustfmt::skip]
const M2_INV_LAB_TO_LMS: [[f64; 3]; 3] = [
    [ 1.0,  0.3963377774,  0.2158037573],
    [ 1.0, -0.1055613458, -0.0638541728],
    [ 1.0, -0.0894841775, -1.2914855480],
];

// ── Linear sRGB ↔ XYZ matrices (D65, IEC 61966-2-1) ────────────────────────

/// sRGB → XYZ (D65).
#[rustfmt::skip]
const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [ 0.4124564, 0.3575761, 0.1804375],
    [ 0.2126729, 0.7151522, 0.0721750],
    [ 0.0193339, 0.1191920, 0.9503041],
];

/// XYZ (D65) → sRGB.
#[rustfmt::skip]
const XYZ_TO_SRGB: [[f64; 3]; 3] = [
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
];

// ── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn mat3_mul_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// sRGB gamma-to-linear (EOTF).
#[inline]
fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear-to-sRGB gamma (OETF).
#[inline]
fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Cube root that preserves sign.
#[inline]
fn cbrt_signed(x: f64) -> f64 {
    if x >= 0.0 {
        x.cbrt()
    } else {
        -((-x).cbrt())
    }
}

// ── Oklab color ──────────────────────────────────────────────────────────────

/// Oklab perceptual color.
///
/// - `l`: Lightness (0 = black, 1 = white for sRGB white)
/// - `a`: Green-red axis (negative = green, positive = red)
/// - `b`: Blue-yellow axis (negative = blue, positive = yellow)
///
/// Provides better perceptual uniformity than CIELAB with simpler math.
/// Ideal for image processing, blending, and color interpolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Oklab {
    /// Lightness [0, 1] for SDR content
    pub l: f64,
    /// Green-red axis
    pub a: f64,
    /// Blue-yellow axis
    pub b: f64,
}

impl Oklab {
    /// Creates a new Oklab color.
    #[must_use]
    pub const fn new(l: f64, a: f64, b: f64) -> Self {
        Self { l, a, b }
    }

    /// Converts from CIE XYZ (D65, Y=1 for reference white) to Oklab.
    #[must_use]
    pub fn from_xyz(xyz: &Xyz) -> Self {
        let lms = mat3_mul_vec(&M1_XYZ_TO_LMS, [xyz.x, xyz.y, xyz.z]);
        let lms_g = [
            cbrt_signed(lms[0]),
            cbrt_signed(lms[1]),
            cbrt_signed(lms[2]),
        ];
        let lab = mat3_mul_vec(&M2_LMS_TO_LAB, lms_g);
        Self {
            l: lab[0],
            a: lab[1],
            b: lab[2],
        }
    }

    /// Converts from sRGB (gamma-encoded, 0-1 range) to Oklab.
    #[must_use]
    pub fn from_srgb(r: f64, g: f64, b: f64) -> Self {
        let lr = srgb_to_linear(r);
        let lg = srgb_to_linear(g);
        let lb = srgb_to_linear(b);
        Self::from_linear_srgb(lr, lg, lb)
    }

    /// Converts from linear sRGB to Oklab.
    #[must_use]
    pub fn from_linear_srgb(r: f64, g: f64, b: f64) -> Self {
        let xyz = mat3_mul_vec(&SRGB_TO_XYZ, [r, g, b]);
        let xyz_struct = Xyz::new(xyz[0], xyz[1], xyz[2]);
        Self::from_xyz(&xyz_struct)
    }

    /// Converts Oklab back to CIE XYZ (D65).
    #[must_use]
    pub fn to_xyz(&self) -> Xyz {
        let lms_g = mat3_mul_vec(&M2_INV_LAB_TO_LMS, [self.l, self.a, self.b]);
        let lms = [
            lms_g[0] * lms_g[0] * lms_g[0],
            lms_g[1] * lms_g[1] * lms_g[1],
            lms_g[2] * lms_g[2] * lms_g[2],
        ];
        let xyz = mat3_mul_vec(&M1_INV_LMS_TO_XYZ, lms);
        Xyz::new(xyz[0], xyz[1], xyz[2])
    }

    /// Converts Oklab to linear sRGB.
    #[must_use]
    pub fn to_linear_srgb(&self) -> (f64, f64, f64) {
        let xyz = self.to_xyz();
        let rgb = mat3_mul_vec(&XYZ_TO_SRGB, [xyz.x, xyz.y, xyz.z]);
        (rgb[0], rgb[1], rgb[2])
    }

    /// Converts Oklab to gamma-encoded sRGB (0-1 range).
    ///
    /// Note: values may be outside \[0,1\] for out-of-gamut colors.
    #[must_use]
    pub fn to_srgb(&self) -> (f64, f64, f64) {
        let (r, g, b) = self.to_linear_srgb();
        (linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b))
    }

    /// Returns the chroma (distance from the achromatic axis).
    #[must_use]
    pub fn chroma(&self) -> f64 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    /// Returns the hue angle in degrees [0, 360).
    #[must_use]
    pub fn hue(&self) -> f64 {
        let mut h = self.b.atan2(self.a).to_degrees();
        if h < 0.0 {
            h += 360.0;
        }
        h
    }

    /// Converts to the cylindrical Oklch form.
    #[must_use]
    pub fn to_oklch(&self) -> Oklch {
        Oklch::from_oklab(self)
    }

    /// Linearly interpolates between two Oklab colors.
    ///
    /// Interpolation in Oklab space produces perceptually smooth gradients
    /// without the hue shift issues of sRGB or HSL blending.
    ///
    /// `t` in [0, 1] where 0 returns `self` and 1 returns `other`.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            l: self.l + (other.l - self.l) * t,
            a: self.a + (other.a - self.a) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }

    /// Computes Euclidean color difference in Oklab space.
    ///
    /// Since Oklab is perceptually uniform, this gives a good approximation
    /// of perceived color difference.
    #[must_use]
    pub fn delta_eok(&self, other: &Self) -> f64 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        (dl * dl + da * da + db * db).sqrt()
    }
}

// ── Oklch color (cylindrical Oklab) ──────────────────────────────────────────

/// Oklch cylindrical color (polar form of Oklab).
///
/// - `l`: Lightness [0, 1] (same as Oklab L)
/// - `c`: Chroma (distance from achromatic axis, 0 = gray)
/// - `h`: Hue angle in degrees [0, 360)
///
/// Oklch is the most practical choice for CSS Color Level 4 and modern
/// design tools, providing intuitive hue and saturation control with
/// perceptual uniformity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Oklch {
    /// Lightness [0, 1]
    pub l: f64,
    /// Chroma (colorfulness, >= 0)
    pub c: f64,
    /// Hue angle in degrees [0, 360)
    pub h: f64,
}

impl Oklch {
    /// Creates a new Oklch color.
    #[must_use]
    pub const fn new(l: f64, c: f64, h: f64) -> Self {
        Self { l, c, h }
    }

    /// Converts from Oklab to Oklch.
    #[must_use]
    pub fn from_oklab(oklab: &Oklab) -> Self {
        Self {
            l: oklab.l,
            c: oklab.chroma(),
            h: oklab.hue(),
        }
    }

    /// Converts from CIE XYZ (D65) to Oklch.
    #[must_use]
    pub fn from_xyz(xyz: &Xyz) -> Self {
        Oklab::from_xyz(xyz).to_oklch()
    }

    /// Converts from sRGB (gamma-encoded) to Oklch.
    #[must_use]
    pub fn from_srgb(r: f64, g: f64, b: f64) -> Self {
        Oklab::from_srgb(r, g, b).to_oklch()
    }

    /// Converts back to Oklab.
    #[must_use]
    pub fn to_oklab(&self) -> Oklab {
        let h_rad = self.h.to_radians();
        Oklab {
            l: self.l,
            a: self.c * h_rad.cos(),
            b: self.c * h_rad.sin(),
        }
    }

    /// Converts to CIE XYZ (D65).
    #[must_use]
    pub fn to_xyz(&self) -> Xyz {
        self.to_oklab().to_xyz()
    }

    /// Converts to gamma-encoded sRGB.
    #[must_use]
    pub fn to_srgb(&self) -> (f64, f64, f64) {
        self.to_oklab().to_srgb()
    }

    /// Linearly interpolates between two Oklch colors.
    ///
    /// Hue interpolation uses the shortest angular path.
    /// `t` in [0, 1].
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        let l = self.l + (other.l - self.l) * t;
        let c = self.c + (other.c - self.c) * t;

        // Shortest hue interpolation
        let mut dh = other.h - self.h;
        if dh > 180.0 {
            dh -= 360.0;
        } else if dh < -180.0 {
            dh += 360.0;
        }
        let mut h = self.h + dh * t;
        if h < 0.0 {
            h += 360.0;
        }
        if h >= 360.0 {
            h -= 360.0;
        }

        Self { l, c, h }
    }

    /// Computes color difference using Oklch coordinates.
    #[must_use]
    pub fn delta_eok(&self, other: &Self) -> f64 {
        let dl = self.l - other.l;
        let dc = self.c - other.c;
        let dh_rad = (other.h - self.h).to_radians();
        let dhab = 2.0 * (self.c * other.c).sqrt() * (dh_rad / 2.0).sin();
        (dl * dl + dc * dc + dhab * dhab).sqrt()
    }

    /// Adjusts the chroma while keeping lightness and hue constant.
    #[must_use]
    pub fn with_chroma(&self, c: f64) -> Self {
        Self {
            l: self.l,
            c: c.max(0.0),
            h: self.h,
        }
    }

    /// Adjusts the hue while keeping lightness and chroma constant.
    #[must_use]
    pub fn with_hue(&self, h: f64) -> Self {
        let mut h = h % 360.0;
        if h < 0.0 {
            h += 360.0;
        }
        Self {
            l: self.l,
            c: self.c,
            h,
        }
    }

    /// Adjusts the lightness while keeping chroma and hue constant.
    #[must_use]
    pub fn with_lightness(&self, l: f64) -> Self {
        Self {
            l: l.clamp(0.0, 1.0),
            c: self.c,
            h: self.h,
        }
    }
}

// ── Array-based free functions (ergonomic API) ────────────────────────────────

/// Converts gamma-encoded sRGB (values in [0, 1]) to Oklab.
///
/// Uses the Oklab matrix from Björn Ottosson's specification.
///
/// # Arguments
///
/// * `rgb` - `[R, G, B]` in gamma-encoded sRGB, typically [0, 1]
///
/// # Returns
///
/// `[L, a, b]` in Oklab space
#[must_use]
pub fn srgb_to_oklab(rgb: [f64; 3]) -> [f64; 3] {
    let ok = Oklab::from_srgb(rgb[0], rgb[1], rgb[2]);
    [ok.l, ok.a, ok.b]
}

/// Converts Oklab back to gamma-encoded sRGB.
///
/// Note: returned values may be outside [0, 1] for out-of-gamut colors.
///
/// # Arguments
///
/// * `lab` - `[L, a, b]` in Oklab space
///
/// # Returns
///
/// `[R, G, B]` in gamma-encoded sRGB
#[must_use]
pub fn oklab_to_srgb(lab: [f64; 3]) -> [f64; 3] {
    let ok = Oklab::new(lab[0], lab[1], lab[2]);
    let (r, g, b) = ok.to_srgb();
    [r, g, b]
}

/// Converts Oklab to cylindrical Oklch.
///
/// # Arguments
///
/// * `lab` - `[L, a, b]` in Oklab space
///
/// # Returns
///
/// `[L, C, h]` where h is the hue angle in degrees [0, 360)
#[must_use]
pub fn oklab_to_oklch(lab: [f64; 3]) -> [f64; 3] {
    let ok = Oklab::new(lab[0], lab[1], lab[2]);
    let lch = ok.to_oklch();
    [lch.l, lch.c, lch.h]
}

/// Converts cylindrical Oklch back to Oklab.
///
/// # Arguments
///
/// * `lch` - `[L, C, h]` where h is the hue angle in degrees
///
/// # Returns
///
/// `[L, a, b]` in Oklab space
#[must_use]
pub fn oklch_to_oklab(lch: [f64; 3]) -> [f64; 3] {
    let ok = Oklch::new(lch[0], lch[1], lch[2]);
    let lab = ok.to_oklab();
    [lab.l, lab.a, lab.b]
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-4;

    // ── Oklab basic tests ────────────────────────────────────────────────────

    #[test]
    fn test_oklab_creation() {
        let c = Oklab::new(0.5, 0.1, -0.1);
        assert_eq!(c.l, 0.5);
        assert_eq!(c.a, 0.1);
        assert_eq!(c.b, -0.1);
    }

    // ── White/black ──────────────────────────────────────────────────────────

    #[test]
    fn test_oklab_white() {
        let white = Oklab::from_srgb(1.0, 1.0, 1.0);
        assert!(
            (white.l - 1.0).abs() < 0.01,
            "White L should be ~1.0, got {}",
            white.l
        );
        assert!(white.a.abs() < 0.01, "White a={}", white.a);
        assert!(white.b.abs() < 0.01, "White b={}", white.b);
    }

    #[test]
    fn test_oklab_black() {
        let black = Oklab::from_srgb(0.0, 0.0, 0.0);
        assert!(
            black.l.abs() < 0.01,
            "Black L should be ~0.0, got {}",
            black.l
        );
    }

    // ── sRGB roundtrip ──────────────────────────────────────────────────────

    #[test]
    fn test_oklab_srgb_roundtrip() {
        let test_colors = [
            (1.0, 0.0, 0.0), // red
            (0.0, 1.0, 0.0), // green
            (0.0, 0.0, 1.0), // blue
            (1.0, 1.0, 0.0), // yellow
            (0.5, 0.5, 0.5), // mid gray
            (0.2, 0.6, 0.8), // teal
        ];

        for (r, g, b) in test_colors {
            let oklab = Oklab::from_srgb(r, g, b);
            let (r2, g2, b2) = oklab.to_srgb();
            assert!(
                (r2 - r).abs() < 0.001,
                "R roundtrip for ({},{},{}): {} vs {}",
                r,
                g,
                b,
                r2,
                r
            );
            assert!(
                (g2 - g).abs() < 0.001,
                "G roundtrip for ({},{},{}): {} vs {}",
                r,
                g,
                b,
                g2,
                g
            );
            assert!(
                (b2 - b).abs() < 0.001,
                "B roundtrip for ({},{},{}): {} vs {}",
                r,
                g,
                b,
                b2,
                b
            );
        }
    }

    // ── XYZ roundtrip ───────────────────────────────────────────────────────

    #[test]
    fn test_oklab_xyz_roundtrip() {
        let xyz = Xyz::d65();
        let oklab = Oklab::from_xyz(&xyz);
        let xyz2 = oklab.to_xyz();
        assert!(
            (xyz2.x - xyz.x).abs() < TOLERANCE,
            "X: {} vs {}",
            xyz2.x,
            xyz.x
        );
        assert!(
            (xyz2.y - xyz.y).abs() < TOLERANCE,
            "Y: {} vs {}",
            xyz2.y,
            xyz.y
        );
        assert!(
            (xyz2.z - xyz.z).abs() < TOLERANCE,
            "Z: {} vs {}",
            xyz2.z,
            xyz.z
        );
    }

    // ── Perceptual properties ───────────────────────────────────────────────

    #[test]
    fn test_oklab_lightness_ordering() {
        let dark = Oklab::from_srgb(0.1, 0.1, 0.1);
        let mid = Oklab::from_srgb(0.5, 0.5, 0.5);
        let bright = Oklab::from_srgb(0.9, 0.9, 0.9);
        assert!(dark.l < mid.l);
        assert!(mid.l < bright.l);
    }

    #[test]
    fn test_oklab_grays_achromatic() {
        for gray_level in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let oklab = Oklab::from_srgb(gray_level, gray_level, gray_level);
            assert!(
                oklab.chroma() < 0.01,
                "Gray {} should be achromatic, chroma={}",
                gray_level,
                oklab.chroma()
            );
        }
    }

    #[test]
    fn test_oklab_chroma_of_saturated_colors() {
        let red = Oklab::from_srgb(1.0, 0.0, 0.0);
        let green = Oklab::from_srgb(0.0, 1.0, 0.0);
        let blue = Oklab::from_srgb(0.0, 0.0, 1.0);
        // All saturated colors should have significant chroma
        assert!(red.chroma() > 0.1, "Red chroma={}", red.chroma());
        assert!(green.chroma() > 0.1, "Green chroma={}", green.chroma());
        assert!(blue.chroma() > 0.1, "Blue chroma={}", blue.chroma());
    }

    // ── Hue ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_oklab_hue_different_for_primaries() {
        let red = Oklab::from_srgb(1.0, 0.0, 0.0);
        let green = Oklab::from_srgb(0.0, 1.0, 0.0);
        let blue = Oklab::from_srgb(0.0, 0.0, 1.0);
        // Hues should be distinct
        let hr = red.hue();
        let hg = green.hue();
        let hb = blue.hue();
        assert!((hr - hg).abs() > 10.0, "Red and green hues too similar");
        assert!((hg - hb).abs() > 10.0, "Green and blue hues too similar");
    }

    // ── Delta Eok ───────────────────────────────────────────────────────────

    #[test]
    fn test_delta_eok_same() {
        let c = Oklab::new(0.5, 0.1, -0.05);
        assert!(c.delta_eok(&c) < 1e-15);
    }

    #[test]
    fn test_delta_eok_symmetry() {
        let c1 = Oklab::from_srgb(0.8, 0.2, 0.3);
        let c2 = Oklab::from_srgb(0.3, 0.7, 0.1);
        assert!((c1.delta_eok(&c2) - c2.delta_eok(&c1)).abs() < 1e-15);
    }

    #[test]
    fn test_delta_eok_positive() {
        let c1 = Oklab::from_srgb(0.0, 0.0, 0.0);
        let c2 = Oklab::from_srgb(1.0, 1.0, 1.0);
        assert!(
            c1.delta_eok(&c2) > 0.5,
            "Black-white dE should be significant"
        );
    }

    // ── Lerp ────────────────────────────────────────────────────────────────

    #[test]
    fn test_oklab_lerp_endpoints() {
        let c1 = Oklab::new(0.3, 0.1, -0.05);
        let c2 = Oklab::new(0.8, -0.05, 0.1);

        let at0 = c1.lerp(&c2, 0.0);
        assert!((at0.l - c1.l).abs() < 1e-10);
        assert!((at0.a - c1.a).abs() < 1e-10);
        assert!((at0.b - c1.b).abs() < 1e-10);

        let at1 = c1.lerp(&c2, 1.0);
        assert!((at1.l - c2.l).abs() < 1e-10);
        assert!((at1.a - c2.a).abs() < 1e-10);
        assert!((at1.b - c2.b).abs() < 1e-10);
    }

    #[test]
    fn test_oklab_lerp_midpoint() {
        let c1 = Oklab::new(0.2, 0.1, 0.0);
        let c2 = Oklab::new(0.8, 0.0, 0.1);
        let mid = c1.lerp(&c2, 0.5);
        assert!((mid.l - 0.5).abs() < 1e-10);
        assert!((mid.a - 0.05).abs() < 1e-10);
        assert!((mid.b - 0.05).abs() < 1e-10);
    }

    // ── Oklch tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_oklch_creation() {
        let c = Oklch::new(0.5, 0.15, 120.0);
        assert_eq!(c.l, 0.5);
        assert_eq!(c.c, 0.15);
        assert_eq!(c.h, 120.0);
    }

    #[test]
    fn test_oklch_from_oklab_roundtrip() {
        let oklab = Oklab::new(0.6, 0.08, 0.06);
        let oklch = oklab.to_oklch();
        let back = oklch.to_oklab();
        assert!((back.l - oklab.l).abs() < 1e-10);
        assert!((back.a - oklab.a).abs() < 1e-10);
        assert!((back.b - oklab.b).abs() < 1e-10);
    }

    #[test]
    fn test_oklch_chroma_and_hue() {
        let oklab = Oklab::new(0.5, 0.0, 0.1);
        let oklch = oklab.to_oklch();
        assert!((oklch.c - 0.1).abs() < 1e-10);
        assert!((oklch.h - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_oklch_xyz_roundtrip() {
        let xyz = Xyz::new(0.4, 0.5, 0.6);
        let oklch = Oklch::from_xyz(&xyz);
        let xyz2 = oklch.to_xyz();
        assert!(
            (xyz2.x - xyz.x).abs() < TOLERANCE,
            "X: {} vs {}",
            xyz2.x,
            xyz.x
        );
        assert!(
            (xyz2.y - xyz.y).abs() < TOLERANCE,
            "Y: {} vs {}",
            xyz2.y,
            xyz.y
        );
        assert!(
            (xyz2.z - xyz.z).abs() < TOLERANCE,
            "Z: {} vs {}",
            xyz2.z,
            xyz.z
        );
    }

    #[test]
    fn test_oklch_srgb_roundtrip() {
        let oklch = Oklch::from_srgb(0.8, 0.3, 0.5);
        let (r, g, b) = oklch.to_srgb();
        assert!((r - 0.8).abs() < 0.001);
        assert!((g - 0.3).abs() < 0.001);
        assert!((b - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_oklch_lerp_shortest_hue() {
        let c1 = Oklch::new(0.5, 0.1, 10.0);
        let c2 = Oklch::new(0.5, 0.1, 350.0);
        let mid = c1.lerp(&c2, 0.5);
        // Should go through 0/360 not through 180
        assert!(
            mid.h < 10.0 || mid.h > 350.0,
            "Lerp hue should take shortest path, got h={}",
            mid.h
        );
    }

    #[test]
    fn test_oklch_with_chroma() {
        let c = Oklch::new(0.5, 0.1, 120.0);
        let c2 = c.with_chroma(0.2);
        assert_eq!(c2.l, 0.5);
        assert_eq!(c2.c, 0.2);
        assert_eq!(c2.h, 120.0);
    }

    #[test]
    fn test_oklch_with_hue() {
        let c = Oklch::new(0.5, 0.1, 120.0);
        let c2 = c.with_hue(240.0);
        assert_eq!(c2.l, 0.5);
        assert_eq!(c2.c, 0.1);
        assert_eq!(c2.h, 240.0);
    }

    #[test]
    fn test_oklch_with_lightness() {
        let c = Oklch::new(0.5, 0.1, 120.0);
        let c2 = c.with_lightness(0.8);
        assert_eq!(c2.l, 0.8);
        assert_eq!(c2.c, 0.1);
        assert_eq!(c2.h, 120.0);
    }

    #[test]
    fn test_oklch_delta_eok() {
        let c1 = Oklch::new(0.5, 0.1, 30.0);
        let c2 = Oklch::new(0.6, 0.15, 60.0);
        let de = c1.delta_eok(&c2);
        assert!(de > 0.0);
    }

    #[test]
    fn test_oklch_with_negative_hue_wraps() {
        let c = Oklch::new(0.5, 0.1, 30.0);
        let c2 = c.with_hue(-30.0);
        assert!((c2.h - 330.0).abs() < 1e-10, "h={}", c2.h);
    }

    #[test]
    fn test_oklch_with_chroma_clamps_negative() {
        let c = Oklch::new(0.5, 0.1, 30.0);
        let c2 = c.with_chroma(-0.1);
        assert_eq!(c2.c, 0.0, "Negative chroma should clamp to 0");
    }

    #[test]
    fn test_oklch_with_lightness_clamps() {
        let c = Oklch::new(0.5, 0.1, 30.0);
        assert_eq!(c.with_lightness(-0.5).l, 0.0);
        assert_eq!(c.with_lightness(1.5).l, 1.0);
    }

    // ── sRGB transfer function tests ────────────────────────────────────────

    #[test]
    fn test_srgb_transfer_roundtrip() {
        for v in [0.0, 0.01, 0.03, 0.1, 0.5, 1.0] {
            let linear = srgb_to_linear(v);
            let back = linear_to_srgb(linear);
            assert!(
                (back - v).abs() < 1e-8,
                "sRGB transfer roundtrip for {}: got {}",
                v,
                back
            );
        }
    }

    // ── Oklab linear sRGB roundtrip ─────────────────────────────────────────

    #[test]
    fn test_oklab_linear_srgb_roundtrip() {
        let oklab = Oklab::from_linear_srgb(0.5, 0.3, 0.2);
        let (r, g, b) = oklab.to_linear_srgb();
        assert!((r - 0.5).abs() < 0.001);
        assert!((g - 0.3).abs() < 0.001);
        assert!((b - 0.2).abs() < 0.001);
    }
}

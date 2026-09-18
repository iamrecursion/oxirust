//! Colour space calibration primitives.
//!
//! Provides colour-space primaries definitions, 3×3 colour matrix arithmetic,
//! CIE ΔE 76 colour difference, XYZ↔Lab conversion, and RGB→XYZ
//! transformation.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Primaries
// ---------------------------------------------------------------------------

/// Colour space chromaticity primaries and white point, expressed in CIE
/// 1931 xy coordinates.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Primaries {
    /// Red primary x.
    pub rx: f64,
    /// Red primary y.
    pub ry: f64,
    /// Green primary x.
    pub gx: f64,
    /// Green primary y.
    pub gy: f64,
    /// Blue primary x.
    pub bx: f64,
    /// Blue primary y.
    pub by: f64,
    /// White point x.
    pub wx: f64,
    /// White point y.
    pub wy: f64,
}

impl Primaries {
    /// ITU-R BT.709 (sRGB) primaries.
    #[must_use]
    pub const fn rec709() -> Self {
        Self {
            rx: 0.640,
            ry: 0.330,
            gx: 0.300,
            gy: 0.600,
            bx: 0.150,
            by: 0.060,
            wx: 0.3127,
            wy: 0.3290, // D65
        }
    }

    /// ITU-R BT.2020 (Rec. 2020) primaries.
    #[must_use]
    pub const fn rec2020() -> Self {
        Self {
            rx: 0.708,
            ry: 0.292,
            gx: 0.170,
            gy: 0.797,
            bx: 0.131,
            by: 0.046,
            wx: 0.3127,
            wy: 0.3290, // D65
        }
    }

    /// DCI-P3 (Digital Cinema Initiative) primaries.
    #[must_use]
    pub const fn dci_p3() -> Self {
        Self {
            rx: 0.680,
            ry: 0.320,
            gx: 0.265,
            gy: 0.690,
            bx: 0.150,
            by: 0.060,
            wx: 0.3140,
            wy: 0.3510, // DCI white point
        }
    }
}

// ---------------------------------------------------------------------------
// ColorMatrix3x3
// ---------------------------------------------------------------------------

/// Row-major 3×3 linear colour matrix.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ColorMatrix3x3 {
    /// Row-major matrix coefficients.
    pub m: [[f64; 3]; 3],
}

impl ColorMatrix3x3 {
    /// Identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Multiply the matrix by a column vector `v` and return `M * v`.
    #[must_use]
    pub fn multiply(&self, v: [f64; 3]) -> [f64; 3] {
        let m = &self.m;
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    /// Compose (multiply) this matrix with `other`: result = `self * other`.
    #[must_use]
    pub fn compose(&self, other: &ColorMatrix3x3) -> ColorMatrix3x3 {
        let a = &self.m;
        let b = &other.m;
        let mut out = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        ColorMatrix3x3 { m: out }
    }
}

// ---------------------------------------------------------------------------
// Colour-difference helpers
// ---------------------------------------------------------------------------

/// CIE ΔE 76 colour difference between two CIE L\*a\*b\* colours.
///
/// ΔE 76 is the Euclidean distance in CIELAB space.
#[must_use]
pub fn delta_e_76(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    let dl = lab1[0] - lab2[0];
    let da = lab1[1] - lab2[1];
    let db = lab1[2] - lab2[2];
    (dl * dl + da * da + db * db).sqrt()
}

/// Convert CIE XYZ (D65 reference white) to CIE L\*a\*b\*.
///
/// Reference white: D65 (`Xn = 0.95047`, `Yn = 1.0`, `Zn = 1.08883`).
#[must_use]
pub fn xyz_to_lab(xyz: [f64; 3]) -> [f64; 3] {
    // D65 white point
    const XN: f64 = 0.950_47;
    const YN: f64 = 1.000_00;
    const ZN: f64 = 1.088_83;

    let f = |t: f64| -> f64 {
        const DELTA: f64 = 6.0 / 29.0;
        const DELTA2: f64 = DELTA * DELTA;
        const DELTA3: f64 = DELTA * DELTA * DELTA;
        if t > DELTA3 {
            t.cbrt()
        } else {
            t / (3.0 * DELTA2) + 4.0 / 29.0
        }
    };

    let fx = f(xyz[0] / XN);
    let fy = f(xyz[1] / YN);
    let fz = f(xyz[2] / ZN);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    [l, a, b]
}

/// Transform linear RGB to CIE XYZ using the supplied colour matrix.
///
/// `matrix` should be a 3×3 RGB→XYZ transform (e.g. generated from
/// [`Primaries`]).
#[must_use]
pub fn rgb_to_xyz(rgb: [f64; 3], matrix: &ColorMatrix3x3) -> [f64; 3] {
    matrix.multiply(rgb)
}

// ---------------------------------------------------------------------------
// ColorSpace enum
// ---------------------------------------------------------------------------

/// Named working colour space with associated primaries and white point.
///
/// Each variant exposes its [`Primaries`] (CIE xy chromaticities) and a
/// reference-white XYZ triplet via [`ColorSpace::primaries`] and
/// [`ColorSpace::white_point_xyz`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ColorSpace {
    /// ITU-R BT.709 / sRGB (D65 white).
    Rec709,
    /// ITU-R BT.2020 (D65 white).
    Rec2020,
    /// DCI-P3 digital cinema (DCI white ~6300 K).
    DciP3,
    /// DCI-P3 with D65 white point (Display P3 used by Apple/HDR monitors).
    DciP3D65,
    /// CIE equal-energy illuminant E.
    EqualEnergy,
}

impl ColorSpace {
    /// Return the [`Primaries`] for this colour space.
    #[must_use]
    pub fn primaries(self) -> Primaries {
        match self {
            Self::Rec709 => Primaries::rec709(),
            Self::Rec2020 => Primaries::rec2020(),
            Self::DciP3 => Primaries::dci_p3(),
            Self::DciP3D65 => Primaries {
                rx: 0.680,
                ry: 0.320,
                gx: 0.265,
                gy: 0.690,
                bx: 0.150,
                by: 0.060,
                wx: 0.3127,
                wy: 0.3290, // D65
            },
            Self::EqualEnergy => Primaries {
                rx: 0.700,
                ry: 0.300,
                gx: 0.200,
                gy: 0.600,
                bx: 0.150,
                by: 0.060,
                wx: 1.0 / 3.0,
                wy: 1.0 / 3.0, // equal energy E
            },
        }
    }

    /// Return the approximate D65-normalised XYZ white point `[X, Y, Z]`.
    ///
    /// All Rec.709, Rec.2020, and Display-P3 spaces use D65; original DCI-P3
    /// uses the DCI white (approximately D63).
    #[must_use]
    pub fn white_point_xyz(self) -> [f64; 3] {
        match self {
            Self::Rec709 | Self::Rec2020 | Self::DciP3D65 => {
                [0.950_47, 1.000_00, 1.088_83] // D65
            }
            Self::DciP3 => {
                // DCI white point: x=0.3140, y=0.3510
                // XYZ from xyY with Y=1: X=x/y, Z=(1-x-y)/y
                let x = 0.3140_f64;
                let y = 0.3510_f64;
                [x / y, 1.0, (1.0 - x - y) / y]
            }
            Self::EqualEnergy => [1.0, 1.0, 1.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Primaries ─────────────────────────────────────────────────────────

    #[test]
    fn test_rec709_white_point_d65() {
        let p = Primaries::rec709();
        assert!((p.wx - 0.3127).abs() < 1e-4, "wx={}", p.wx);
        assert!((p.wy - 0.3290).abs() < 1e-4, "wy={}", p.wy);
    }

    #[test]
    fn test_rec2020_wider_gamut_red() {
        let r709 = Primaries::rec709();
        let r2020 = Primaries::rec2020();
        // Rec. 2020 red primary has higher x than Rec. 709
        assert!(
            r2020.rx > r709.rx,
            "2020 red x={}, 709 red x={}",
            r2020.rx,
            r709.rx
        );
    }

    #[test]
    fn test_dci_p3_primaries_values() {
        let p = Primaries::dci_p3();
        assert!((p.rx - 0.680).abs() < 1e-4);
        assert!((p.gy - 0.690).abs() < 1e-4);
    }

    // ── ColorMatrix3x3 ────────────────────────────────────────────────────

    #[test]
    fn test_identity_multiply_unchanged() {
        let m = ColorMatrix3x3::identity();
        let v = [0.2, 0.5, 0.8];
        let out = m.multiply(v);
        assert!((out[0] - 0.2).abs() < 1e-12);
        assert!((out[1] - 0.5).abs() < 1e-12);
        assert!((out[2] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_multiply_scales_red_channel() {
        let mut m = ColorMatrix3x3::identity();
        m.m[0][0] = 2.0; // scale red by 2
        let out = m.multiply([1.0, 1.0, 1.0]);
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
        assert!((out[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compose_identity_with_self() {
        let m = ColorMatrix3x3::identity();
        let composed = m.compose(&m);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((composed.m[i][j] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_compose_two_scales() {
        let mut a = ColorMatrix3x3::identity();
        a.m[0][0] = 2.0;
        let mut b = ColorMatrix3x3::identity();
        b.m[0][0] = 3.0;
        let c = a.compose(&b);
        // (2*I) * (3*I) applied to [1,0,0] = 6
        assert!((c.m[0][0] - 6.0).abs() < 1e-12);
    }

    // ── delta_e_76 ────────────────────────────────────────────────────────

    #[test]
    fn test_delta_e_76_identical_colours() {
        let de = delta_e_76([50.0, 20.0, -10.0], [50.0, 20.0, -10.0]);
        assert!(
            de.abs() < 1e-12,
            "ΔE should be 0 for identical colours, got {de}"
        );
    }

    #[test]
    fn test_delta_e_76_known_distance() {
        // Simple 3-4-5 right triangle in Lab
        let de = delta_e_76([0.0, 3.0, 0.0], [0.0, 0.0, 4.0]);
        assert!((de - 5.0).abs() < 1e-9, "ΔE={de}");
    }

    #[test]
    fn test_delta_e_76_white_vs_black() {
        // L* difference of 100 (black vs white)
        let de = delta_e_76([100.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!((de - 100.0).abs() < 1e-9, "ΔE={de}");
    }

    // ── xyz_to_lab ────────────────────────────────────────────────────────

    #[test]
    fn test_xyz_to_lab_d65_white_is_l100() {
        // D65 white XYZ normalised
        let lab = xyz_to_lab([0.950_47, 1.000_00, 1.088_83]);
        assert!((lab[0] - 100.0).abs() < 0.01, "L*={}", lab[0]);
        assert!(lab[1].abs() < 0.01, "a*={}", lab[1]);
        assert!(lab[2].abs() < 0.01, "b*={}", lab[2]);
    }

    #[test]
    fn test_xyz_to_lab_black_is_l0() {
        let lab = xyz_to_lab([0.0, 0.0, 0.0]);
        assert!(
            lab[0].abs() < 0.01,
            "L* of black should be 0, got {}",
            lab[0]
        );
    }

    #[test]
    fn test_xyz_to_lab_l_positive_for_nonzero_y() {
        let lab = xyz_to_lab([0.2, 0.2, 0.2]);
        assert!(lab[0] > 0.0, "L* should be positive for nonzero XYZ");
    }

    // ── rgb_to_xyz ────────────────────────────────────────────────────────

    #[test]
    fn test_rgb_to_xyz_black_stays_black() {
        let m = ColorMatrix3x3::identity();
        let xyz = rgb_to_xyz([0.0, 0.0, 0.0], &m);
        for c in xyz {
            assert!(c.abs() < 1e-12, "XYZ of black should be [0,0,0], got {c}");
        }
    }

    #[test]
    fn test_rgb_to_xyz_identity_passes_through() {
        let m = ColorMatrix3x3::identity();
        let rgb = [0.3, 0.6, 0.9];
        let xyz = rgb_to_xyz(rgb, &m);
        assert!((xyz[0] - 0.3).abs() < 1e-12);
        assert!((xyz[1] - 0.6).abs() < 1e-12);
        assert!((xyz[2] - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_rgb_to_xyz_then_lab_roundtrip_lightness() {
        // Pure green (linear) through identity matrix – just checks pipeline runs
        let m = ColorMatrix3x3::identity();
        let xyz = rgb_to_xyz([0.0, 1.0, 0.0], &m);
        let lab = xyz_to_lab(xyz);
        // L* of [0, Y, 0] with Y=1 and Yn=1 → L* ≈ 100
        assert!(lab[0] > 90.0, "L* should be high for Y=1, got {}", lab[0]);
    }

    // ── ColorSpace ────────────────────────────────────────────────────────

    #[test]
    fn test_color_space_rec709_white_point_d65() {
        let cs = ColorSpace::Rec709;
        let p = cs.primaries();
        assert!((p.wx - 0.3127).abs() < 1e-4);
        assert!((p.wy - 0.3290).abs() < 1e-4);
    }

    #[test]
    fn test_color_space_rec2020_wider_red_primary() {
        let p709 = ColorSpace::Rec709.primaries();
        let p2020 = ColorSpace::Rec2020.primaries();
        assert!(
            p2020.rx > p709.rx,
            "Rec.2020 red x should exceed Rec.709: {} vs {}",
            p2020.rx,
            p709.rx
        );
    }

    #[test]
    fn test_color_space_dci_p3_primaries() {
        let p = ColorSpace::DciP3.primaries();
        assert!((p.rx - 0.680).abs() < 1e-4, "rx={}", p.rx);
        assert!((p.gy - 0.690).abs() < 1e-4, "gy={}", p.gy);
    }

    #[test]
    fn test_color_space_white_point_xyz_d65_rec709() {
        let wp = ColorSpace::Rec709.white_point_xyz();
        // D65: X≈0.9505, Y=1, Z≈1.0888
        assert!((wp[0] - 0.9505).abs() < 0.001, "X={}", wp[0]);
        assert!((wp[1] - 1.000).abs() < 0.001, "Y={}", wp[1]);
    }

    #[test]
    fn test_color_space_equal_energy_white_point() {
        let wp = ColorSpace::EqualEnergy.white_point_xyz();
        assert!((wp[0] - 1.0).abs() < 1e-9);
        assert!((wp[1] - 1.0).abs() < 1e-9);
        assert!((wp[2] - 1.0).abs() < 1e-9);
    }
}

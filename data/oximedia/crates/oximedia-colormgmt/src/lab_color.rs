#![allow(dead_code)]
//! CIE L*a*b* color representation, gamut checking, and RGB conversion.

use std::f64::consts::PI;

// ── CIE illuminant D65 reference white (XYZ) ──────────────────────────────
const D65_X: f64 = 0.950_456;
const D65_Y: f64 = 1.000_000;
const D65_Z: f64 = 1.089_058;

// ── sRGB ↔ XYZ matrices (IEC 61966-2-1) ──────────────────────────────────
#[rustfmt::skip]
const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.412_391, 0.357_584, 0.180_481],
    [0.212_639, 0.715_169, 0.072_192],
    [0.019_331, 0.119_195, 0.950_532],
];

#[rustfmt::skip]
const XYZ_TO_SRGB: [[f64; 3]; 3] = [
    [ 3.240_970, -1.537_383, -0.498_611],
    [-0.969_244,  1.875_968,  0.041_555],
    [ 0.055_630, -0.203_977,  1.056_972],
];

fn mat3_mul_vec3(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// CIE f(t) helper.
fn f_lab(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// Inverse CIE f(t).
fn f_lab_inv(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// A color in CIE L*a*b* space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabColor {
    /// Lightness [0, 100].
    pub l: f64,
    /// Green-red axis (typically −128 to +127).
    pub a: f64,
    /// Blue-yellow axis (typically −128 to +127).
    pub b: f64,
}

impl LabColor {
    /// Create a new Lab color.
    pub fn new(l: f64, a: f64, b: f64) -> Self {
        Self { l, a, b }
    }

    /// ΔE 1976 (simple Euclidean distance in Lab space).
    pub fn delta_e_76(&self, other: &LabColor) -> f64 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        (dl * dl + da * da + db * db).sqrt()
    }

    /// ΔE CIE 2000 (CIEDE2000).
    #[allow(clippy::too_many_lines)]
    pub fn delta_e_cie2000(&self, other: &LabColor) -> f64 {
        let l1 = self.l;
        let a1 = self.a;
        let b1 = self.b;
        let l2 = other.l;
        let a2 = other.a;
        let b2 = other.b;

        let c1 = (a1 * a1 + b1 * b1).sqrt();
        let c2 = (a2 * a2 + b2 * b2).sqrt();
        let c_bar = (c1 + c2) / 2.0;
        let g = 0.5 * (1.0 - ((c_bar.powi(7)) / (c_bar.powi(7) + 25.0_f64.powi(7))).sqrt());

        let a1p = a1 * (1.0 + g);
        let a2p = a2 * (1.0 + g);

        let c1p = (a1p * a1p + b1 * b1).sqrt();
        let c2p = (a2p * a2p + b2 * b2).sqrt();

        let h_prime = |bp: f64, ap: f64| -> f64 {
            if bp == 0.0 && ap == 0.0 {
                0.0
            } else {
                let h = bp.atan2(ap).to_degrees();
                if h < 0.0 {
                    h + 360.0
                } else {
                    h
                }
            }
        };
        let h1p = h_prime(b1, a1p);
        let h2p = h_prime(b2, a2p);

        let delta_lp = l2 - l1;
        let delta_cp = c2p - c1p;
        let delta_hp = if c1p * c2p == 0.0 {
            0.0
        } else if (h2p - h1p).abs() <= 180.0 {
            h2p - h1p
        } else if h2p - h1p > 180.0 {
            h2p - h1p - 360.0
        } else {
            h2p - h1p + 360.0
        };
        let delta_bhp = 2.0 * (c1p * c2p).sqrt() * (delta_hp / 2.0 * PI / 180.0).sin();

        let l_bar_p = (l1 + l2) / 2.0;
        let c_bar_p = (c1p + c2p) / 2.0;
        let h_bar_p = if c1p * c2p == 0.0 {
            h1p + h2p
        } else if (h1p - h2p).abs() <= 180.0 {
            (h1p + h2p) / 2.0
        } else if h1p + h2p < 360.0 {
            (h1p + h2p + 360.0) / 2.0
        } else {
            (h1p + h2p - 360.0) / 2.0
        };

        let t = 1.0 - 0.17 * ((h_bar_p - 30.0) * PI / 180.0).cos()
            + 0.24 * (2.0 * h_bar_p * PI / 180.0).cos()
            + 0.32 * ((3.0 * h_bar_p + 6.0) * PI / 180.0).cos()
            - 0.20 * ((4.0 * h_bar_p - 63.0) * PI / 180.0).cos();

        let s_l = 1.0 + 0.015 * (l_bar_p - 50.0).powi(2) / (20.0 + (l_bar_p - 50.0).powi(2)).sqrt();
        let s_c = 1.0 + 0.045 * c_bar_p;
        let s_h = 1.0 + 0.015 * c_bar_p * t;

        let delta_theta = 30.0 * (-((h_bar_p - 275.0) / 25.0).powi(2)).exp();
        let r_c = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + 25.0_f64.powi(7))).sqrt();
        let r_t = -r_c * (2.0 * delta_theta * PI / 180.0).sin();

        ((delta_lp / s_l).powi(2)
            + (delta_cp / s_c).powi(2)
            + (delta_bhp / s_h).powi(2)
            + r_t * (delta_cp / s_c) * (delta_bhp / s_h))
            .sqrt()
    }
}

/// A simple axis-aligned gamut in Lab space.
#[derive(Debug, Clone)]
pub struct LabGamut {
    /// Minimum L* value (lightness).
    pub l_min: f64,
    /// Maximum L* value (lightness).
    pub l_max: f64,
    /// Minimum a* value (green–red axis).
    pub a_min: f64,
    /// Maximum a* value (green–red axis).
    pub a_max: f64,
    /// Minimum b* value (blue–yellow axis).
    pub b_min: f64,
    /// Maximum b* value (blue–yellow axis).
    pub b_max: f64,
}

impl LabGamut {
    /// Create a new Lab gamut bounding box.
    pub fn new(l_min: f64, l_max: f64, a_min: f64, a_max: f64, b_min: f64, b_max: f64) -> Self {
        Self {
            l_min,
            l_max,
            a_min,
            a_max,
            b_min,
            b_max,
        }
    }

    /// Return `true` if the given Lab color lies within this gamut.
    pub fn contains(&self, color: &LabColor) -> bool {
        color.l >= self.l_min
            && color.l <= self.l_max
            && color.a >= self.a_min
            && color.a <= self.a_max
            && color.b >= self.b_min
            && color.b <= self.b_max
    }
}

/// Converts between linear sRGB and CIE L*a*b*.
pub struct LabColorConverter;

impl LabColorConverter {
    /// Convert linear sRGB [0, 1] to CIE L*a*b*.
    pub fn rgb_to_lab(rgb: [f64; 3]) -> LabColor {
        let xyz = mat3_mul_vec3(&SRGB_TO_XYZ, rgb);
        let fx = f_lab(xyz[0] / D65_X);
        let fy = f_lab(xyz[1] / D65_Y);
        let fz = f_lab(xyz[2] / D65_Z);
        LabColor {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
        }
    }

    /// Convert CIE L*a*b* back to linear sRGB [0, 1].
    pub fn lab_to_rgb(lab: &LabColor) -> [f64; 3] {
        let fy = (lab.l + 16.0) / 116.0;
        let fx = lab.a / 500.0 + fy;
        let fz = fy - lab.b / 200.0;
        let xyz = [
            D65_X * f_lab_inv(fx),
            D65_Y * f_lab_inv(fy),
            D65_Z * f_lab_inv(fz),
        ];
        let rgb = mat3_mul_vec3(&XYZ_TO_SRGB, xyz);
        [
            rgb[0].clamp(0.0, 1.0),
            rgb[1].clamp(0.0, 1.0),
            rgb[2].clamp(0.0, 1.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_e_76_same_color() {
        let c = LabColor::new(50.0, 0.0, 0.0);
        assert!((c.delta_e_76(&c) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_delta_e_76_known() {
        let c1 = LabColor::new(50.0, 0.0, 0.0);
        let c2 = LabColor::new(53.0, 4.0, 0.0);
        let de = c1.delta_e_76(&c2);
        assert!((de - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_delta_e_76_symmetric() {
        let c1 = LabColor::new(40.0, 10.0, -5.0);
        let c2 = LabColor::new(60.0, -10.0, 20.0);
        assert!((c1.delta_e_76(&c2) - c2.delta_e_76(&c1)).abs() < 1e-10);
    }

    #[test]
    fn test_delta_e_cie2000_same_color() {
        let c = LabColor::new(50.0, 25.0, -10.0);
        assert!((c.delta_e_cie2000(&c) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_delta_e_cie2000_positive() {
        let c1 = LabColor::new(50.0, 0.0, 0.0);
        let c2 = LabColor::new(55.0, 5.0, 5.0);
        let de = c1.delta_e_cie2000(&c2);
        assert!(de > 0.0);
    }

    #[test]
    fn test_lab_gamut_contains_true() {
        let gamut = LabGamut::new(0.0, 100.0, -128.0, 128.0, -128.0, 128.0);
        assert!(gamut.contains(&LabColor::new(50.0, 0.0, 0.0)));
    }

    #[test]
    fn test_lab_gamut_contains_false_l() {
        let gamut = LabGamut::new(0.0, 80.0, -128.0, 128.0, -128.0, 128.0);
        assert!(!gamut.contains(&LabColor::new(90.0, 0.0, 0.0)));
    }

    #[test]
    fn test_lab_gamut_contains_false_a() {
        let gamut = LabGamut::new(0.0, 100.0, -50.0, 50.0, -128.0, 128.0);
        assert!(!gamut.contains(&LabColor::new(50.0, 60.0, 0.0)));
    }

    #[test]
    fn test_rgb_to_lab_black() {
        let lab = LabColorConverter::rgb_to_lab([0.0, 0.0, 0.0]);
        assert!(lab.l.abs() < 1e-6);
    }

    #[test]
    fn test_rgb_to_lab_white() {
        let lab = LabColorConverter::rgb_to_lab([1.0, 1.0, 1.0]);
        assert!((lab.l - 100.0).abs() < 1e-4);
    }

    #[test]
    fn test_lab_to_rgb_roundtrip() {
        let original = [0.5, 0.3, 0.7_f64];
        let lab = LabColorConverter::rgb_to_lab(original);
        let back = LabColorConverter::lab_to_rgb(&lab);
        assert!((original[0] - back[0]).abs() < 1e-6);
        assert!((original[1] - back[1]).abs() < 1e-6);
        assert!((original[2] - back[2]).abs() < 1e-6);
    }

    #[test]
    fn test_lab_to_rgb_white() {
        let lab = LabColor::new(100.0, 0.0, 0.0);
        let rgb = LabColorConverter::lab_to_rgb(&lab);
        assert!((rgb[0] - 1.0).abs() < 1e-4);
        assert!((rgb[1] - 1.0).abs() < 1e-4);
        assert!((rgb[2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_lab_color_copy() {
        let c = LabColor::new(50.0, 10.0, -5.0);
        let d = c;
        assert_eq!(c, d);
    }
}

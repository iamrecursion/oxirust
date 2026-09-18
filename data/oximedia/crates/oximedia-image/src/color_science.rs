//! Color science calculations for professional imaging workflows.
//!
//! Provides CIE XYZ, L*a*b*, color primaries, chromatic adaptation, and delta-E metrics.

#![allow(dead_code)]

/// CIE XYZ color space representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xyz {
    /// X tristimulus value.
    pub x: f64,
    /// Y tristimulus value (luminance).
    pub y: f64,
    /// Z tristimulus value.
    pub z: f64,
}

/// CIE L*a*b* color space representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lab {
    /// Lightness (0–100).
    pub l: f64,
    /// Green–red chromatic component.
    pub a: f64,
    /// Blue–yellow chromatic component.
    pub b: f64,
}

/// Linear RGB color representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb {
    /// Red component (0.0–1.0).
    pub r: f64,
    /// Green component (0.0–1.0).
    pub g: f64,
    /// Blue component (0.0–1.0).
    pub b: f64,
}

/// Color primaries definition (CIE xy chromaticity coordinates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorPrimaries {
    /// Red primary xy chromaticity.
    pub r: (f64, f64),
    /// Green primary xy chromaticity.
    pub g: (f64, f64),
    /// Blue primary xy chromaticity.
    pub b: (f64, f64),
    /// White point xy chromaticity.
    pub white: (f64, f64),
}

impl ColorPrimaries {
    /// ITU-R BT.709 (HDTV) color primaries.
    #[must_use]
    pub const fn bt709() -> Self {
        Self {
            r: (0.640, 0.330),
            g: (0.300, 0.600),
            b: (0.150, 0.060),
            white: (0.3127, 0.3290),
        }
    }

    /// ITU-R BT.2020 (UHDTV) color primaries.
    #[must_use]
    pub const fn bt2020() -> Self {
        Self {
            r: (0.708, 0.292),
            g: (0.170, 0.797),
            b: (0.131, 0.046),
            white: (0.3127, 0.3290),
        }
    }

    /// DCI-P3 with D65 white point color primaries.
    #[must_use]
    pub const fn p3_d65() -> Self {
        Self {
            r: (0.680, 0.320),
            g: (0.265, 0.690),
            b: (0.150, 0.060),
            white: (0.3127, 0.3290),
        }
    }

    /// ACES AP0 color primaries.
    #[must_use]
    pub const fn aces_ap0() -> Self {
        Self {
            r: (0.7347, 0.2653),
            g: (0.0000, 1.0000),
            b: (0.0001, -0.0770),
            white: (0.32168, 0.33767),
        }
    }

    /// ACES AP1 color primaries.
    #[must_use]
    pub const fn aces_ap1() -> Self {
        Self {
            r: (0.713, 0.293),
            g: (0.165, 0.830),
            b: (0.128, 0.044),
            white: (0.32168, 0.33767),
        }
    }
}

impl Rgb {
    /// Create a new RGB color.
    #[must_use]
    pub const fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Convert RGB to CIE XYZ using the given color primaries.
    ///
    /// Builds a 3x3 RGB-to-XYZ matrix from the primaries and white point,
    /// then applies it to the RGB values.
    #[must_use]
    pub fn to_xyz(&self, primaries: &ColorPrimaries) -> Xyz {
        let mat = rgb_to_xyz_matrix(primaries);
        let x = mat[0][0] * self.r + mat[0][1] * self.g + mat[0][2] * self.b;
        let y = mat[1][0] * self.r + mat[1][1] * self.g + mat[1][2] * self.b;
        let z = mat[2][0] * self.r + mat[2][1] * self.g + mat[2][2] * self.b;
        Xyz { x, y, z }
    }
}

impl Xyz {
    /// Create a new XYZ color.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// D65 reference white in XYZ (normalised so Y = 1).
    #[must_use]
    pub const fn d65() -> Self {
        Self {
            x: 0.950489,
            y: 1.000000,
            z: 1.088840,
        }
    }

    /// Convert XYZ to CIE L*a*b* relative to the given reference white.
    #[must_use]
    pub fn to_lab(&self, reference: &Xyz) -> Lab {
        let fx = lab_f(self.x / reference.x);
        let fy = lab_f(self.y / reference.y);
        let fz = lab_f(self.z / reference.z);

        Lab {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
        }
    }
}

impl Lab {
    /// Create a new Lab color.
    #[must_use]
    pub const fn new(l: f64, a: f64, b: f64) -> Self {
        Self { l, a, b }
    }

    /// CIE ΔE 1976 (Euclidean distance in L*a*b* space).
    #[must_use]
    pub fn delta_e_1976(&self, other: &Lab) -> f64 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        (dl * dl + da * da + db * db).sqrt()
    }

    /// CIE ΔE 2000 (CIEDE2000) with kL = kC = kH = 1.
    ///
    /// Reference: Luo, Cui, Rigg (2001).
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn delta_e_2000(&self, other: &Lab) -> f64 {
        // Step 1: compute C*ab and h*ab
        let c1 = (self.a * self.a + self.b * self.b).sqrt();
        let c2 = (other.a * other.a + other.b * other.b).sqrt();
        let c_avg = (c1 + c2) / 2.0;
        let c_avg7 = c_avg.powi(7);
        let g = 0.5 * (1.0 - (c_avg7 / (c_avg7 + 25_f64.powi(7))).sqrt());

        let a1p = self.a * (1.0 + g);
        let a2p = other.a * (1.0 + g);

        let c1p = (a1p * a1p + self.b * self.b).sqrt();
        let c2p = (a2p * a2p + other.b * other.b).sqrt();

        let h1p = atan2_deg(self.b, a1p);
        let h2p = atan2_deg(other.b, a2p);

        // Step 2: delta L', delta C', delta H'
        let dl_p = other.l - self.l;
        let dc_p = c2p - c1p;

        let dh_p = if c1p * c2p == 0.0 {
            0.0
        } else if (h2p - h1p).abs() <= 180.0 {
            h2p - h1p
        } else if h2p - h1p > 180.0 {
            h2p - h1p - 360.0
        } else {
            h2p - h1p + 360.0
        };

        let dh_p_big = 2.0 * (c1p * c2p).sqrt() * (dh_p.to_radians() / 2.0).sin();

        // Step 3: CIEDE2000
        let l_avg = (self.l + other.l) / 2.0;
        let c_avgp = (c1p + c2p) / 2.0;

        let h_avgp = if c1p * c2p == 0.0 {
            h1p + h2p
        } else if (h1p - h2p).abs() <= 180.0 {
            (h1p + h2p) / 2.0
        } else if h1p + h2p < 360.0 {
            (h1p + h2p + 360.0) / 2.0
        } else {
            (h1p + h2p - 360.0) / 2.0
        };

        let t = 1.0 - 0.17 * (h_avgp - 30.0).to_radians().cos()
            + 0.24 * (2.0 * h_avgp).to_radians().cos()
            + 0.32 * (3.0 * h_avgp + 6.0).to_radians().cos()
            - 0.20 * (4.0 * h_avgp - 63.0).to_radians().cos();

        let sl = 1.0 + 0.015 * (l_avg - 50.0).powi(2) / (20.0 + (l_avg - 50.0).powi(2)).sqrt();
        let sc = 1.0 + 0.045 * c_avgp;
        let sh = 1.0 + 0.015 * c_avgp * t;

        let c_avgp7 = c_avgp.powi(7);
        let rc = 2.0 * (c_avgp7 / (c_avgp7 + 25_f64.powi(7))).sqrt();
        let d_theta = 30.0 * (-(((h_avgp - 275.0) / 25.0).powi(2))).exp();
        let rt = -(rc * (2.0 * d_theta).to_radians().sin());

        ((dl_p / sl).powi(2)
            + (dc_p / sc).powi(2)
            + (dh_p_big / sh).powi(2)
            + rt * (dc_p / sc) * (dh_p_big / sh))
            .sqrt()
    }
}

/// Chromatic adaptation transforms.
pub struct ChromaticAdaptation;

impl ChromaticAdaptation {
    /// Compute a Bradford chromatic adaptation matrix from `src_white` to `dst_white`.
    ///
    /// Both whites are given as CIE xy chromaticity coordinates.
    #[must_use]
    pub fn bradford_matrix(src_white: (f64, f64), dst_white: (f64, f64)) -> [[f64; 3]; 3] {
        // Bradford matrix (forward cone-response transform)
        let m_bradford: [[f64; 3]; 3] = [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ];
        let m_bradford_inv: [[f64; 3]; 3] = [
            [0.9869929, -0.1470543, 0.1599627],
            [0.4323053, 0.5183603, 0.0492912],
            [-0.0085287, 0.0400428, 0.9684867],
        ];

        let src_xyz = xy_to_xyz(src_white);
        let dst_xyz = xy_to_xyz(dst_white);

        // Transform whites into Bradford cone space
        let src_cone = mat3_mul_vec(&m_bradford, src_xyz);
        let dst_cone = mat3_mul_vec(&m_bradford, dst_xyz);

        // Scale matrix
        let scale: [[f64; 3]; 3] = [
            [dst_cone[0] / src_cone[0], 0.0, 0.0],
            [0.0, dst_cone[1] / src_cone[1], 0.0],
            [0.0, 0.0, dst_cone[2] / src_cone[2]],
        ];

        // Full matrix: M_inv * Scale * M
        let tmp = mat3_mul(&scale, &m_bradford);
        mat3_mul(&m_bradford_inv, &tmp)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// CIE L*a*b* `f` function.
fn lab_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// atan2 in degrees, remapped to [0, 360).
fn atan2_deg(y: f64, x: f64) -> f64 {
    let deg = y.atan2(x).to_degrees();
    if deg < 0.0 {
        deg + 360.0
    } else {
        deg
    }
}

/// Convert CIE xy chromaticity to normalised XYZ (Y = 1).
fn xy_to_xyz(xy: (f64, f64)) -> [f64; 3] {
    let (x, y) = xy;
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Build an RGB-to-XYZ matrix from color primaries.
fn rgb_to_xyz_matrix(p: &ColorPrimaries) -> [[f64; 3]; 3] {
    // Convert primaries and white to XYZ
    let xr = p.r.0 / p.r.1;
    let yr = 1.0;
    let zr = (1.0 - p.r.0 - p.r.1) / p.r.1;

    let xg = p.g.0 / p.g.1;
    let yg = 1.0;
    let zg = (1.0 - p.g.0 - p.g.1) / p.g.1;

    let xb = p.b.0 / p.b.1;
    let yb = 1.0;
    let zb = (1.0 - p.b.0 - p.b.1) / p.b.1;

    let white = xy_to_xyz(p.white);

    // M = [Xr Xg Xb; Yr Yg Yb; Zr Zg Zb]
    let m: [[f64; 3]; 3] = [[xr, xg, xb], [yr, yg, yb], [zr, zg, zb]];

    // S = M^-1 * W
    let m_inv = mat3_inv(&m);
    let s = mat3_mul_vec(&m_inv, white);

    // Final matrix: columns of M scaled by S
    [
        [s[0] * xr, s[1] * xg, s[2] * xb],
        [s[0] * yr, s[1] * yg, s[2] * yb],
        [s[0] * zr, s[1] * zg, s[2] * zb],
    ]
}

/// 3×3 matrix inverse (Cramer's rule).
fn mat3_inv(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    let inv_det = 1.0 / det;

    [
        [
            inv_det * (m[1][1] * m[2][2] - m[1][2] * m[2][1]),
            inv_det * (m[0][2] * m[2][1] - m[0][1] * m[2][2]),
            inv_det * (m[0][1] * m[1][2] - m[0][2] * m[1][1]),
        ],
        [
            inv_det * (m[1][2] * m[2][0] - m[1][0] * m[2][2]),
            inv_det * (m[0][0] * m[2][2] - m[0][2] * m[2][0]),
            inv_det * (m[0][2] * m[1][0] - m[0][0] * m[1][2]),
        ],
        [
            inv_det * (m[1][0] * m[2][1] - m[1][1] * m[2][0]),
            inv_det * (m[0][1] * m[2][0] - m[0][0] * m[2][1]),
            inv_det * (m[0][0] * m[1][1] - m[0][1] * m[1][0]),
        ],
    ]
}

/// 3×3 matrix multiply.
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

/// 3×3 matrix × 3-vector.
fn mat3_mul_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lab_f_above_delta_cubed() {
        // (6/29)^3 ≈ 0.008856
        let val = lab_f(0.5);
        assert!((val - 0.5f64.cbrt()).abs() < 1e-12);
    }

    #[test]
    fn test_lab_f_below_delta_cubed() {
        let val = lab_f(0.001);
        // linear part
        let delta = 6.0 / 29.0;
        let expected = 0.001 / (3.0 * delta * delta) + 4.0 / 29.0;
        assert!((val - expected).abs() < 1e-12);
    }

    #[test]
    fn test_xyz_to_lab_d65() {
        // D65 white should produce L=100, a≈0, b≈0
        let white = Xyz::d65();
        let lab = white.to_lab(&white);
        assert!((lab.l - 100.0).abs() < 1e-8);
        assert!(lab.a.abs() < 1e-6);
        assert!(lab.b.abs() < 1e-6);
    }

    #[test]
    fn test_delta_e_1976_zero() {
        let lab = Lab::new(50.0, 10.0, -10.0);
        assert_eq!(lab.delta_e_1976(&lab), 0.0);
    }

    #[test]
    fn test_delta_e_1976_nonzero() {
        let a = Lab::new(50.0, 0.0, 0.0);
        let b = Lab::new(53.0, 4.0, 0.0);
        let de = a.delta_e_1976(&b);
        assert!((de - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_delta_e_2000_zero() {
        let lab = Lab::new(50.0, 25.0, -25.0);
        let de = lab.delta_e_2000(&lab);
        assert!(de.abs() < 1e-10);
    }

    #[test]
    fn test_delta_e_2000_approximately_correct() {
        // Pair 1 from the CIEDE2000 test spreadsheet (Sharma 2005):
        // L1=50.0000, a1=2.6772, b1=-79.7751
        // L2=50.0000, a2=0.0000, b2=-82.7485
        // Expected ΔE = 2.0425
        let l1 = Lab::new(50.0, 2.6772, -79.7751);
        let l2 = Lab::new(50.0, 0.0, -82.7485);
        let de = l1.delta_e_2000(&l2);
        assert!((de - 2.0425).abs() < 0.01, "de = {}", de);
    }

    #[test]
    fn test_rgb_to_xyz_bt709_white() {
        // Pure white (1,1,1) in BT.709 should be close to D65 XYZ
        let rgb = Rgb::new(1.0, 1.0, 1.0);
        let xyz = rgb.to_xyz(&ColorPrimaries::bt709());
        // Y ≈ 1.0
        assert!((xyz.y - 1.0).abs() < 1e-6, "Y = {}", xyz.y);
    }

    #[test]
    fn test_color_primaries_bt709() {
        let p = ColorPrimaries::bt709();
        assert_eq!(p.r, (0.640, 0.330));
        assert_eq!(p.white, (0.3127, 0.3290));
    }

    #[test]
    fn test_color_primaries_bt2020() {
        let p = ColorPrimaries::bt2020();
        assert_eq!(p.r, (0.708, 0.292));
    }

    #[test]
    fn test_color_primaries_p3_d65() {
        let p = ColorPrimaries::p3_d65();
        assert_eq!(p.r, (0.680, 0.320));
    }

    #[test]
    fn test_color_primaries_aces_ap0() {
        let p = ColorPrimaries::aces_ap0();
        assert_eq!(p.r, (0.7347, 0.2653));
    }

    #[test]
    fn test_color_primaries_aces_ap1() {
        let p = ColorPrimaries::aces_ap1();
        assert_eq!(p.r, (0.713, 0.293));
    }

    #[test]
    fn test_bradford_matrix_identity() {
        // Same src/dst white → adaptation matrix should be close to identity
        let d65 = (0.3127, 0.3290);
        let m = ChromaticAdaptation::bradford_matrix(d65, d65);
        assert!((m[0][0] - 1.0).abs() < 1e-4);
        assert!((m[1][1] - 1.0).abs() < 1e-4);
        assert!((m[2][2] - 1.0).abs() < 1e-4);
        assert!(m[0][1].abs() < 1e-4);
    }

    #[test]
    fn test_bradford_d65_to_d50() {
        // D65 → D50 adaptation should produce a non-identity matrix
        let d65 = (0.3127, 0.3290);
        let d50 = (0.3457, 0.3585);
        let m = ChromaticAdaptation::bradford_matrix(d65, d50);
        // Just sanity-check that something changed
        assert!((m[0][0] - 1.0).abs() > 1e-4);
    }
}

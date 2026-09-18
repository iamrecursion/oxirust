//! Jzazbz and JzCzhz perceptual color spaces.
//!
//! Jzazbz is a perceptually uniform color space designed for HDR and wide color gamut
//! content. Developed by Safdar et al. (2017), it provides excellent perceptual
//! uniformity, especially for high dynamic range content.
//!
//! JzCzhz is the cylindrical (polar) form of Jzazbz, analogous to LCH for Lab.
//!
//! Reference: Safdar, M., Cui, G., Kim, Y.J. & Luo, M.R. (2017).
//! "Perceptually uniform color space for image signals including high dynamic range
//! and wide gamut." Optics Express, 25(13), 15131-15151.

use crate::xyz::Xyz;

// ── PQ transfer function constants (SMPTE ST 2084) ──────────────────────────

const PQ_M1: f64 = 0.159_301_758_125_0; // = 2610 / 16384
const PQ_M2: f64 = 78.843_75; // = 2523 / 32 × 128
const PQ_C1: f64 = 0.835_937_5; // = 3424 / 4096
const PQ_C2: f64 = 18.851_5625; // = 2413 / 128
const PQ_C3: f64 = 18.686_718_75; // = 2392 / 128

// ── Jzazbz constants ─────────────────────────────────────────────────────────

const JZ_B: f64 = 1.15;
const JZ_G: f64 = 0.66;
const JZ_D: f64 = -0.56;
const JZ_D0: f64 = 1.6295499532821566e-11;

/// Matrix M1: XYZ(D65) → LMS (Jzazbz-specific)
#[rustfmt::skip]
const M1: [[f64; 3]; 3] = [
    [ 0.41478972, 0.579999,  0.0146480],
    [-0.20151000, 1.120649,  0.0531008],
    [-0.01660080, 0.264800,  0.6684799],
];

/// Matrix M2: LMS′ → Izazbz
#[rustfmt::skip]
const M2: [[f64; 3]; 3] = [
    [ 0.5,       0.5,       0.0      ],
    [ 3.524000, -4.066708,  0.542708 ],
    [ 0.199076,  1.096799, -1.295875 ],
];

/// Inverse of M1
#[rustfmt::skip]
const M1_INV: [[f64; 3]; 3] = [
    [ 1.9242264357876067,  -1.0047923125953657,  0.037651404030618  ],
    [ 0.35031676209499907,  0.7264811939316552, -0.06538442294808501],
    [-0.09098281098284752, -0.3127282905230739,  1.5227665613052603 ],
];

/// Inverse of M2
#[rustfmt::skip]
const M2_INV: [[f64; 3]; 3] = [
    [ 1.0,                 0.1386050432715393,   0.05804731615611886 ],
    [ 1.0,                -0.1386050432715393,  -0.05804731615611886 ],
    [ 1.0,                -0.09601924202631895, -0.8118918960560388  ],
];

// ── Helper functions ─────────────────────────────────────────────────────────

#[inline]
fn mat3_mul_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// PQ forward transfer (OETF): linear luminance → PQ encoded value.
#[inline]
fn pq_forward(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let xp = x.powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * xp) / (1.0 + PQ_C3 * xp)).powf(PQ_M2)
}

/// PQ inverse transfer (EOTF): PQ encoded value → linear luminance.
#[inline]
fn pq_inverse(y: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }
    let yp = y.powf(1.0 / PQ_M2);
    let num = (yp - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * yp;
    if den.abs() < 1e-30 {
        return 0.0;
    }
    (num / den).powf(1.0 / PQ_M1)
}

// ── Jzazbz color ─────────────────────────────────────────────────────────────

/// Jzazbz perceptually uniform color.
///
/// - `jz`: Lightness (0 = black, approximately 1.0 for white at 10000 nits)
/// - `az`: Red-green axis
/// - `bz`: Yellow-blue axis
///
/// Designed for HDR content with excellent perceptual uniformity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jzazbz {
    /// Lightness (typically 0.0–1.0 for HDR content)
    pub jz: f64,
    /// Red-green opponent dimension
    pub az: f64,
    /// Yellow-blue opponent dimension
    pub bz: f64,
}

impl Jzazbz {
    /// Creates a new Jzazbz color.
    #[must_use]
    pub const fn new(jz: f64, az: f64, bz: f64) -> Self {
        Self { jz, az, bz }
    }

    /// Converts from CIE XYZ (D65, absolute luminance in cd/m²) to Jzazbz.
    ///
    /// Input XYZ values should be in absolute luminance (cd/m²), not normalized.
    /// For normalized XYZ (Y=1 for white), multiply by the peak luminance in nits
    /// before calling this function.
    ///
    /// # Arguments
    ///
    /// * `xyz` - XYZ color in absolute luminance (cd/m²)
    #[must_use]
    pub fn from_xyz(xyz: &Xyz) -> Self {
        Self::from_xyz_abs(xyz.x, xyz.y, xyz.z)
    }

    /// Converts from absolute XYZ (cd/m²) to Jzazbz.
    #[must_use]
    pub fn from_xyz_abs(x: f64, y: f64, z: f64) -> Self {
        // Step 1: Modify XYZ for Jzazbz
        let xp = JZ_B * x - (JZ_B - 1.0) * z;
        let yp = JZ_G * y - (JZ_G - 1.0) * x;

        // Step 2: XYZ → LMS via M1
        let lms = mat3_mul_vec(&M1, [xp, yp, z]);

        // Step 3: PQ compress each LMS channel
        // Normalize by 10000 for PQ (which expects nits / 10000)
        let lms_pq = [
            pq_forward(lms[0] / 10000.0),
            pq_forward(lms[1] / 10000.0),
            pq_forward(lms[2] / 10000.0),
        ];

        // Step 4: LMS′ → Izazbz via M2
        let izazbz = mat3_mul_vec(&M2, lms_pq);

        // Step 5: Iz → Jz
        let jz = (1.0 + JZ_D) * izazbz[0] / (1.0 + JZ_D * izazbz[0]) - JZ_D0;

        Self {
            jz,
            az: izazbz[1],
            bz: izazbz[2],
        }
    }

    /// Converts from normalized XYZ (Y=1 for reference white) to Jzazbz,
    /// given the peak luminance of the content in nits.
    ///
    /// # Arguments
    ///
    /// * `xyz` - Normalized XYZ (Y=1 for white)
    /// * `peak_nits` - Peak luminance in nits (e.g., 10000.0 for full HDR)
    #[must_use]
    pub fn from_xyz_normalized(xyz: &Xyz, peak_nits: f64) -> Self {
        Self::from_xyz_abs(xyz.x * peak_nits, xyz.y * peak_nits, xyz.z * peak_nits)
    }

    /// Converts Jzazbz back to absolute XYZ (cd/m²).
    #[must_use]
    pub fn to_xyz(&self) -> Xyz {
        let (x, y, z) = self.to_xyz_abs();
        Xyz::new(x, y, z)
    }

    /// Converts Jzazbz back to absolute XYZ (cd/m²), returning a tuple.
    #[must_use]
    pub fn to_xyz_abs(&self) -> (f64, f64, f64) {
        // Step 1: Jz → Iz
        let iz = (self.jz + JZ_D0) / (1.0 + JZ_D - JZ_D * (self.jz + JZ_D0));

        // Step 2: Izazbz → LMS′ via M2_INV
        let lms_pq = mat3_mul_vec(&M2_INV, [iz, self.az, self.bz]);

        // Step 3: PQ inverse → LMS (absolute)
        let lms = [
            pq_inverse(lms_pq[0]) * 10000.0,
            pq_inverse(lms_pq[1]) * 10000.0,
            pq_inverse(lms_pq[2]) * 10000.0,
        ];

        // Step 4: LMS → modified XYZ via M1_INV
        let xyz_p = mat3_mul_vec(&M1_INV, lms);
        let xp = xyz_p[0];
        let yp = xyz_p[1];
        let z = xyz_p[2];

        // Step 5: Undo the XYZ modification
        // xp = B*x - (B-1)*z  => x = (xp + (B-1)*z) / B
        // yp = G*y - (G-1)*x  => y = (yp + (G-1)*x) / G
        let x = (xp + (JZ_B - 1.0) * z) / JZ_B;
        let y = (yp + (JZ_G - 1.0) * x) / JZ_G;

        (x, y, z)
    }

    /// Converts to cylindrical JzCzhz form.
    #[must_use]
    pub fn to_jzczhz(&self) -> JzCzhz {
        JzCzhz::from_jzazbz(self)
    }

    /// Returns the chroma (colorfulness).
    #[must_use]
    pub fn chroma(&self) -> f64 {
        (self.az * self.az + self.bz * self.bz).sqrt()
    }

    /// Returns the hue angle in degrees [0, 360).
    #[must_use]
    pub fn hue(&self) -> f64 {
        let mut h = self.bz.atan2(self.az).to_degrees();
        if h < 0.0 {
            h += 360.0;
        }
        h
    }

    /// Computes the ΔEz color difference between two Jzazbz colors.
    ///
    /// This is a simple Euclidean distance in Jzazbz space, which is perceptually
    /// uniform for HDR content.
    #[must_use]
    pub fn delta_ez(&self, other: &Self) -> f64 {
        let djz = self.jz - other.jz;
        let daz = self.az - other.az;
        let dbz = self.bz - other.bz;
        (djz * djz + daz * daz + dbz * dbz).sqrt()
    }
}

// ── JzCzhz color (cylindrical Jzazbz) ───────────────────────────────────────

/// JzCzhz cylindrical color (polar form of Jzazbz).
///
/// - `jz`: Lightness
/// - `cz`: Chroma (colorfulness / saturation)
/// - `hz`: Hue angle in degrees [0, 360)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JzCzhz {
    /// Lightness
    pub jz: f64,
    /// Chroma (distance from the achromatic axis)
    pub cz: f64,
    /// Hue angle in degrees [0, 360)
    pub hz: f64,
}

impl JzCzhz {
    /// Creates a new JzCzhz color.
    #[must_use]
    pub const fn new(jz: f64, cz: f64, hz: f64) -> Self {
        Self { jz, cz, hz }
    }

    /// Converts from Jzazbz to JzCzhz.
    #[must_use]
    pub fn from_jzazbz(jzazbz: &Jzazbz) -> Self {
        let cz = jzazbz.chroma();
        let hz = jzazbz.hue();
        Self {
            jz: jzazbz.jz,
            cz,
            hz,
        }
    }

    /// Converts from XYZ (absolute luminance) to JzCzhz.
    #[must_use]
    pub fn from_xyz(xyz: &Xyz) -> Self {
        Jzazbz::from_xyz(xyz).to_jzczhz()
    }

    /// Converts back to Jzazbz.
    #[must_use]
    pub fn to_jzazbz(&self) -> Jzazbz {
        let hz_rad = self.hz.to_radians();
        Jzazbz {
            jz: self.jz,
            az: self.cz * hz_rad.cos(),
            bz: self.cz * hz_rad.sin(),
        }
    }

    /// Converts back to XYZ.
    #[must_use]
    pub fn to_xyz(&self) -> Xyz {
        self.to_jzazbz().to_xyz()
    }

    /// Computes ΔEz color difference using JzCzhz coordinates.
    #[must_use]
    pub fn delta_ez(&self, other: &Self) -> f64 {
        let djz = self.jz - other.jz;
        let dcz = self.cz - other.cz;
        // Hue difference: use angle subtraction
        let dh_deg = other.hz - self.hz;
        // Convert hue difference to Cartesian distance
        let dh_rad = dh_deg.to_radians();
        let dhz = 2.0 * (self.cz * other.cz).sqrt() * (dh_rad / 2.0).sin();
        (djz * djz + dcz * dcz + dhz * dhz).sqrt()
    }

    /// Linearly interpolates between two JzCzhz colors.
    ///
    /// `t` should be in [0, 1] where 0 returns `self` and 1 returns `other`.
    /// Hue interpolation uses the shortest angular path.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        let jz = self.jz + (other.jz - self.jz) * t;
        let cz = self.cz + (other.cz - self.cz) * t;

        // Shortest path hue interpolation
        let mut dh = other.hz - self.hz;
        if dh > 180.0 {
            dh -= 360.0;
        } else if dh < -180.0 {
            dh += 360.0;
        }
        let mut hz = self.hz + dh * t;
        if hz < 0.0 {
            hz += 360.0;
        }
        if hz >= 360.0 {
            hz -= 360.0;
        }

        Self { jz, cz, hz }
    }
}

// ── Array-based free functions (ergonomic API) ────────────────────────────────

/// Converts CIE XYZ (D65, absolute luminance in cd/m²) to Jzazbz.
///
/// Returns `[Jz, az, bz]`.
///
/// # Arguments
///
/// * `xyz` - `[X, Y, Z]` in absolute luminance (cd/m²)
#[must_use]
pub fn xyz_to_jzazbz(xyz: [f64; 3]) -> [f64; 3] {
    let j = Jzazbz::from_xyz_abs(xyz[0], xyz[1], xyz[2]);
    [j.jz, j.az, j.bz]
}

/// Converts Jzazbz back to CIE XYZ (D65, absolute luminance in cd/m²).
///
/// Returns `[X, Y, Z]` in absolute luminance (cd/m²).
///
/// # Arguments
///
/// * `jab` - `[Jz, az, bz]`
#[must_use]
pub fn jzazbz_to_xyz(jab: [f64; 3]) -> [f64; 3] {
    let j = Jzazbz::new(jab[0], jab[1], jab[2]);
    let (x, y, z) = j.to_xyz_abs();
    [x, y, z]
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Jzazbz basic tests ───────────────────────────────────────────────────

    #[test]
    fn test_jzazbz_creation() {
        let c = Jzazbz::new(0.5, 0.1, -0.05);
        assert_eq!(c.jz, 0.5);
        assert_eq!(c.az, 0.1);
        assert_eq!(c.bz, -0.05);
    }

    #[test]
    fn test_jzazbz_black() {
        let black = Jzazbz::from_xyz_abs(0.0, 0.0, 0.0);
        assert!(
            black.jz.abs() < 1e-6,
            "Black Jz={} should be near 0",
            black.jz
        );
        assert!(
            black.az.abs() < 1e-6,
            "Black az={} should be near 0",
            black.az
        );
        assert!(
            black.bz.abs() < 1e-6,
            "Black bz={} should be near 0",
            black.bz
        );
    }

    #[test]
    fn test_jzazbz_white_positive_jz() {
        // D65 at 100 nits
        let white = Jzazbz::from_xyz_abs(95.047, 100.0, 108.883);
        assert!(white.jz > 0.0, "White Jz={} should be positive", white.jz);
    }

    #[test]
    fn test_jzazbz_roundtrip_white() {
        let x = 95.047;
        let y = 100.0;
        let z = 108.883;
        let jzazbz = Jzazbz::from_xyz_abs(x, y, z);
        let (x2, y2, z2) = jzazbz.to_xyz_abs();
        assert!((x2 - x).abs() < 0.5, "X roundtrip: {} vs {}", x2, x);
        assert!((y2 - y).abs() < 0.5, "Y roundtrip: {} vs {}", y2, y);
        assert!((z2 - z).abs() < 0.5, "Z roundtrip: {} vs {}", z2, z);
    }

    #[test]
    fn test_jzazbz_roundtrip_red() {
        // A reddish color
        let x = 41.24;
        let y = 21.26;
        let z = 1.93;
        let jzazbz = Jzazbz::from_xyz_abs(x, y, z);
        let (x2, y2, z2) = jzazbz.to_xyz_abs();
        assert!((x2 - x).abs() < 0.5, "X roundtrip: {} vs {}", x2, x);
        assert!((y2 - y).abs() < 0.5, "Y roundtrip: {} vs {}", y2, y);
        assert!((z2 - z).abs() < 0.5, "Z roundtrip: {} vs {}", z2, z);
    }

    #[test]
    fn test_jzazbz_roundtrip_blue() {
        let x = 18.05;
        let y = 7.22;
        let z = 95.05;
        let jzazbz = Jzazbz::from_xyz_abs(x, y, z);
        let (x2, y2, z2) = jzazbz.to_xyz_abs();
        assert!((x2 - x).abs() < 0.5, "X roundtrip: {} vs {}", x2, x);
        assert!((y2 - y).abs() < 0.5, "Y roundtrip: {} vs {}", y2, y);
        assert!((z2 - z).abs() < 0.5, "Z roundtrip: {} vs {}", z2, z);
    }

    #[test]
    fn test_jzazbz_monotonic_lightness() {
        // Brighter colors should have higher Jz
        let dim = Jzazbz::from_xyz_abs(9.5, 10.0, 10.9);
        let mid = Jzazbz::from_xyz_abs(47.5, 50.0, 54.4);
        let bright = Jzazbz::from_xyz_abs(95.0, 100.0, 108.9);
        assert!(dim.jz < mid.jz, "dim.jz={} < mid.jz={}", dim.jz, mid.jz);
        assert!(
            mid.jz < bright.jz,
            "mid.jz={} < bright.jz={}",
            mid.jz,
            bright.jz
        );
    }

    #[test]
    fn test_jzazbz_hdr_range() {
        // 10000 nit HDR content
        let hdr_white = Jzazbz::from_xyz_abs(9504.7, 10000.0, 10888.3);
        assert!(
            hdr_white.jz > 0.0,
            "HDR Jz should be positive: {}",
            hdr_white.jz
        );
        // Jz should be in a reasonable range for 10000 nits
        assert!(hdr_white.jz < 2.0, "HDR Jz should be < 2: {}", hdr_white.jz);
    }

    #[test]
    fn test_jzazbz_from_xyz_struct() {
        let xyz = Xyz::new(50.0, 50.0, 50.0);
        let jz = Jzazbz::from_xyz(&xyz);
        assert!(jz.jz > 0.0);
    }

    #[test]
    fn test_jzazbz_from_xyz_normalized() {
        let xyz = Xyz::new(0.95047, 1.0, 1.08883);
        let jz = Jzazbz::from_xyz_normalized(&xyz, 100.0);
        assert!(jz.jz > 0.0, "Normalized Jz should be positive: {}", jz.jz);
    }

    #[test]
    fn test_jzazbz_chroma() {
        let c = Jzazbz::new(0.5, 0.03, 0.04);
        let chroma = c.chroma();
        assert!((chroma - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_jzazbz_hue() {
        let c = Jzazbz::new(0.5, 0.0, 0.05);
        let hue = c.hue();
        assert!((hue - 90.0).abs() < 1e-6, "Hue should be 90, got {}", hue);
    }

    // ── Delta Ez tests ───────────────────────────────────────────────────────

    #[test]
    fn test_delta_ez_same_color() {
        let c = Jzazbz::new(0.5, 0.1, -0.05);
        assert!(c.delta_ez(&c) < 1e-10);
    }

    #[test]
    fn test_delta_ez_different_colors() {
        let c1 = Jzazbz::new(0.5, 0.1, -0.05);
        let c2 = Jzazbz::new(0.6, 0.12, -0.03);
        let de = c1.delta_ez(&c2);
        assert!(de > 0.0, "ΔEz should be positive for different colors");
    }

    #[test]
    fn test_delta_ez_symmetry() {
        let c1 = Jzazbz::new(0.3, 0.05, 0.02);
        let c2 = Jzazbz::new(0.7, -0.02, 0.08);
        assert!((c1.delta_ez(&c2) - c2.delta_ez(&c1)).abs() < 1e-10);
    }

    // ── JzCzhz tests ────────────────────────────────────────────────────────

    #[test]
    fn test_jzczhz_creation() {
        let c = JzCzhz::new(0.5, 0.1, 45.0);
        assert_eq!(c.jz, 0.5);
        assert_eq!(c.cz, 0.1);
        assert_eq!(c.hz, 45.0);
    }

    #[test]
    fn test_jzczhz_from_jzazbz_roundtrip() {
        let original = Jzazbz::new(0.5, 0.06, 0.08);
        let czhz = original.to_jzczhz();
        let back = czhz.to_jzazbz();
        assert!((back.jz - original.jz).abs() < 1e-10);
        assert!((back.az - original.az).abs() < 1e-10);
        assert!((back.bz - original.bz).abs() < 1e-10);
    }

    #[test]
    fn test_jzczhz_chroma_matches() {
        let jz = Jzazbz::new(0.5, 0.03, 0.04);
        let czhz = jz.to_jzczhz();
        assert!((czhz.cz - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_jzczhz_hue_matches() {
        let jz = Jzazbz::new(0.5, 0.0, 0.05);
        let czhz = jz.to_jzczhz();
        assert!((czhz.hz - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_jzczhz_xyz_roundtrip() {
        let xyz = Xyz::new(50.0, 40.0, 30.0);
        let czhz = JzCzhz::from_xyz(&xyz);
        let xyz2 = czhz.to_xyz();
        assert!((xyz2.x - xyz.x).abs() < 0.5, "X: {} vs {}", xyz2.x, xyz.x);
        assert!((xyz2.y - xyz.y).abs() < 0.5, "Y: {} vs {}", xyz2.y, xyz.y);
        assert!((xyz2.z - xyz.z).abs() < 0.5, "Z: {} vs {}", xyz2.z, xyz.z);
    }

    #[test]
    fn test_jzczhz_delta_ez() {
        let c1 = JzCzhz::new(0.5, 0.1, 30.0);
        let c2 = JzCzhz::new(0.6, 0.12, 35.0);
        let de = c1.delta_ez(&c2);
        assert!(de > 0.0);
    }

    #[test]
    fn test_jzczhz_lerp_endpoints() {
        let c1 = JzCzhz::new(0.3, 0.05, 30.0);
        let c2 = JzCzhz::new(0.7, 0.15, 270.0);

        let at0 = c1.lerp(&c2, 0.0);
        assert!((at0.jz - c1.jz).abs() < 1e-10);
        assert!((at0.cz - c1.cz).abs() < 1e-10);
        assert!((at0.hz - c1.hz).abs() < 1e-10);

        let at1 = c1.lerp(&c2, 1.0);
        assert!((at1.jz - c2.jz).abs() < 1e-10);
        assert!((at1.cz - c2.cz).abs() < 1e-10);
        assert!((at1.hz - c2.hz).abs() < 1e-10);
    }

    #[test]
    fn test_jzczhz_lerp_midpoint() {
        let c1 = JzCzhz::new(0.3, 0.1, 0.0);
        let c2 = JzCzhz::new(0.7, 0.2, 60.0);
        let mid = c1.lerp(&c2, 0.5);
        assert!((mid.jz - 0.5).abs() < 1e-10);
        assert!((mid.cz - 0.15).abs() < 1e-10);
        assert!((mid.hz - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_jzczhz_lerp_shortest_path() {
        // 10° to 350° should go backwards (through 0°), not forward (through 180°)
        let c1 = JzCzhz::new(0.5, 0.1, 10.0);
        let c2 = JzCzhz::new(0.5, 0.1, 350.0);
        let mid = c1.lerp(&c2, 0.5);
        // Midpoint should be near 0°/360°
        assert!(
            mid.hz < 10.0 || mid.hz > 350.0,
            "Lerp should take shortest hue path, got hz={}",
            mid.hz
        );
    }

    #[test]
    fn test_jzazbz_achromatic_near_zero_chroma() {
        // D65 white should be near-achromatic
        let white = Jzazbz::from_xyz_abs(95.047, 100.0, 108.883);
        assert!(
            white.chroma() < 0.01,
            "White chroma should be near 0: {}",
            white.chroma()
        );
    }

    #[test]
    fn test_pq_forward_inverse_roundtrip() {
        for x in [0.0, 0.001, 0.01, 0.1, 0.5, 1.0] {
            let y = pq_forward(x);
            let x2 = pq_inverse(y);
            assert!(
                (x2 - x).abs() < 1e-8,
                "PQ roundtrip failed for {}: {} vs {}",
                x,
                x2,
                x
            );
        }
    }

    #[test]
    fn test_pq_monotonic() {
        let v1 = pq_forward(0.01);
        let v2 = pq_forward(0.1);
        let v3 = pq_forward(0.5);
        assert!(v1 < v2 && v2 < v3, "PQ should be monotonic");
    }

    #[test]
    fn test_jzazbz_multiple_roundtrips() {
        // Test various colors for roundtrip accuracy
        let test_colors = [
            (50.0, 50.0, 50.0),    // neutral grey
            (41.24, 21.26, 1.93),  // red-ish
            (35.76, 71.52, 11.92), // green-ish
            (18.05, 7.22, 95.05),  // blue-ish
            (77.0, 92.78, 13.85),  // yellow-ish
            (1.0, 1.0, 1.0),       // very dim
            (500.0, 500.0, 500.0), // bright
        ];

        for (x, y, z) in test_colors {
            let jz = Jzazbz::from_xyz_abs(x, y, z);
            let (x2, y2, z2) = jz.to_xyz_abs();
            let tol = x.abs().max(y.abs()).max(z.abs()) * 0.01 + 0.1;
            assert!(
                (x2 - x).abs() < tol,
                "X roundtrip for ({},{},{}): {} vs {} (tol={})",
                x,
                y,
                z,
                x2,
                x,
                tol
            );
            assert!(
                (y2 - y).abs() < tol,
                "Y roundtrip for ({},{},{}): {} vs {} (tol={})",
                x,
                y,
                z,
                y2,
                y,
                tol
            );
            assert!(
                (z2 - z).abs() < tol,
                "Z roundtrip for ({},{},{}): {} vs {} (tol={})",
                x,
                y,
                z,
                z2,
                z,
                tol
            );
        }
    }
}

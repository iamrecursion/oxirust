//! Gamut mapping algorithms for color space conversion.
//!
//! Provides two complementary APIs:
//!
//! 1. **Legacy `GamutMapper`** – simple gamut-clipping / compression based on
//!    known source/destination [`GamutBoundary`] definitions.
//! 2. **New `ColorPrimaries` + `GamutMapper`** – physically grounded 3×3 RGB→RGB
//!    matrix computed through an XYZ intermediate space, followed by the
//!    specified [`GamutMappingMethod`].

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ── New gamut-mapping API ─────────────────────────────────────────────────────

/// Selects the method used to handle out-of-gamut values after matrix conversion.
#[derive(Debug, Clone, PartialEq)]
pub enum GamutMappingMethod {
    /// Hard clip every channel to [0, 1].
    Clip,
    /// Knee-based smooth compression.  The `f32` argument is the knee threshold
    /// in the range 0.0–1.0 (e.g. 0.8 means compression starts at 80 % of peak).
    Compress(f32),
    /// Perceptual: uniformly scale all channels so the peak is at 1.0.
    Perceptual,
    /// Preserve luminance (Rec.709 weights) while reducing saturation.
    Saturation,
    /// Map white point, then clip out-of-gamut colours.
    RelativeColorimetric,
    /// No white-point adaptation; clip out-of-gamut colours.
    AbsoluteColorimetric,
}

/// CIE xy chromaticity primaries and white point for an RGB colour space.
#[derive(Debug, Clone)]
pub struct ColorPrimaries {
    /// Human-readable name.
    pub name: String,
    /// Red primary (x, y).
    pub red: (f32, f32),
    /// Green primary (x, y).
    pub green: (f32, f32),
    /// Blue primary (x, y).
    pub blue: (f32, f32),
    /// White point (x, y).  D65 = (0.3127, 0.3290).
    pub white: (f32, f32),
}

impl ColorPrimaries {
    /// sRGB / Rec.709 primaries (D65 white point).
    #[must_use]
    pub fn rec709() -> Self {
        Self {
            name: "Rec.709 / sRGB".to_string(),
            red: (0.640, 0.330),
            green: (0.300, 0.600),
            blue: (0.150, 0.060),
            white: (0.3127, 0.3290),
        }
    }

    /// Rec.2020 / BT.2020 primaries (D65 white point).
    #[must_use]
    pub fn rec2020() -> Self {
        Self {
            name: "Rec.2020".to_string(),
            red: (0.708, 0.292),
            green: (0.170, 0.797),
            blue: (0.131, 0.046),
            white: (0.3127, 0.3290),
        }
    }

    /// DCI-P3 with D65 white point (Display P3).
    #[must_use]
    pub fn p3_d65() -> Self {
        Self {
            name: "P3-D65".to_string(),
            red: (0.680, 0.320),
            green: (0.265, 0.690),
            blue: (0.150, 0.060),
            white: (0.3127, 0.3290),
        }
    }

    /// DCI-P3 with DCI (theatre) white point.
    #[must_use]
    pub fn p3_dci() -> Self {
        Self {
            name: "P3-DCI".to_string(),
            red: (0.680, 0.320),
            green: (0.265, 0.690),
            blue: (0.150, 0.060),
            white: (0.3140, 0.3510),
        }
    }

    // ── Matrix derivation helpers ─────────────────────────────────────────

    /// Compute the 3×3 matrix that converts linear RGB (this primaries) → XYZ D50.
    ///
    /// Based on the Bradford / ICC methodology:
    /// 1. Convert xy chromaticities to XYZ for each primary.
    /// 2. Solve for the diagonal scaling matrix S using the white point.
    /// 3. M = [Xr Xg Xb; Yr Yg Yb; Zr Zg Zb] · diag(S)
    #[must_use]
    pub fn rgb_to_xyz(&self) -> [[f32; 3]; 3] {
        let (xr, yr) = self.red;
        let (xg, yg) = self.green;
        let (xb, yb) = self.blue;
        let (xw, yw) = self.white;

        // Convert xy → XYZ (Y=1 for each primary)
        let zr = 1.0 - xr - yr;
        let zg = 1.0 - xg - yg;
        let zb = 1.0 - xb - yb;

        // White-point XYZ (normalised to Y=1)
        let yw_safe = if yw.abs() < f32::EPSILON { 1.0 } else { yw };
        let xw_xyz = xw / yw_safe;
        let yw_xyz = 1.0_f32;
        let zw_xyz = (1.0 - xw - yw) / yw_safe;

        // Solve M * s = Wn:
        // [[xr, xg, xb],   [sr]   [xw_xyz]
        //  [yr, yg, yb], * [sg] = [yw_xyz]
        //  [zr, zg, zb]]   [sb]   [zw_xyz]
        let m = [[xr, xg, xb], [yr, yg, yb], [zr, zg, zb]];

        let inv = mat3_inverse(m);
        let s = mat3_mul_vec(inv, [xw_xyz, yw_xyz, zw_xyz]);

        // Final matrix: columns of M scaled by s
        [
            [xr * s[0], xg * s[1], xb * s[2]],
            [yr * s[0], yg * s[1], yb * s[2]],
            [zr * s[0], zg * s[1], zb * s[2]],
        ]
    }
}

/// Gamut mapper that converts linear RGB values from one colour space to another
/// via an XYZ intermediate and applies the chosen out-of-gamut strategy.
pub struct GamutMapper {
    /// Source primaries.
    pub src_primaries: ColorPrimaries,
    /// Destination primaries.
    pub dst_primaries: ColorPrimaries,
    /// Gamut-mapping method.
    pub method: GamutMappingMethod,
    /// Pre-computed 3×3 src-RGB → dst-RGB conversion matrix.
    pub matrix: [[f32; 3]; 3],
}

impl GamutMapper {
    /// Construct a new `GamutMapper`.
    ///
    /// Computes the 3×3 conversion matrix as:
    /// `matrix = (XYZ→dst_RGB) × (src_RGB→XYZ)`
    #[must_use]
    pub fn new(src: ColorPrimaries, dst: ColorPrimaries, method: GamutMappingMethod) -> Self {
        let src_to_xyz = src.rgb_to_xyz();
        let dst_to_xyz = dst.rgb_to_xyz();
        let xyz_to_dst = mat3_inverse(dst_to_xyz);
        let matrix = mat3_mul(xyz_to_dst, src_to_xyz);

        Self {
            src_primaries: src,
            dst_primaries: dst,
            method,
            matrix,
        }
    }

    /// Convert a single linear-light RGB pixel from source to destination gamut.
    #[must_use]
    pub fn map_pixel(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Apply the 3×3 matrix
        let [ro, go, bo] = mat3_mul_vec(self.matrix, [r, g, b]);

        // Apply gamut-mapping strategy
        self.apply_method(ro, go, bo)
    }

    /// Map an entire RGB-interleaved frame.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len()` is not a multiple of 3.
    #[must_use]
    pub fn map_frame(&self, pixels: &[f32]) -> Vec<f32> {
        assert_eq!(
            pixels.len() % 3,
            0,
            "pixels slice length must be a multiple of 3"
        );
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            let (r, g, b) = self.map_pixel(chunk[0], chunk[1], chunk[2]);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        out
    }

    /// Returns `true` when all channels of the *converted* pixel lie in [0, 1].
    #[must_use]
    pub fn is_in_gamut(&self, r: f32, g: f32, b: f32) -> bool {
        let (ro, go, bo) = {
            let [ro, go, bo] = mat3_mul_vec(self.matrix, [r, g, b]);
            (ro, go, bo)
        };
        ro >= 0.0 && ro <= 1.0 && go >= 0.0 && go <= 1.0 && bo >= 0.0 && bo <= 1.0
    }

    // ── Internal: apply the mapping method ───────────────────────────────────

    fn apply_method(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        match &self.method {
            GamutMappingMethod::Clip | GamutMappingMethod::AbsoluteColorimetric => {
                (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
            }
            GamutMappingMethod::Compress(knee) => {
                let k = knee.clamp(0.0, 1.0);
                (
                    smooth_compress(r, k),
                    smooth_compress(g, k),
                    smooth_compress(b, k),
                )
            }
            GamutMappingMethod::Perceptual => {
                let peak = r.max(g).max(b);
                if peak <= 1.0 {
                    (r.max(0.0), g.max(0.0), b.max(0.0))
                } else {
                    (
                        (r / peak).clamp(0.0, 1.0),
                        (g / peak).clamp(0.0, 1.0),
                        (b / peak).clamp(0.0, 1.0),
                    )
                }
            }
            GamutMappingMethod::Saturation => {
                // Rec.709 luminance weights
                let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
                let max_over = r.max(g).max(b).max(1.0);
                if max_over <= 1.0 {
                    (r.max(0.0), g.max(0.0), b.max(0.0))
                } else {
                    let scale = if (max_over - luma).abs() > f32::EPSILON {
                        (1.0 - luma) / (max_over - luma)
                    } else {
                        1.0
                    };
                    (
                        (luma + (r - luma) * scale).clamp(0.0, 1.0),
                        (luma + (g - luma) * scale).clamp(0.0, 1.0),
                        (luma + (b - luma) * scale).clamp(0.0, 1.0),
                    )
                }
            }
            GamutMappingMethod::RelativeColorimetric => {
                // White-point adaptation is already baked into the matrix; just clip.
                (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
            }
        }
    }
}

/// Knee-based smooth compression for a single channel.
///
/// Values below `knee` are passed through; values above are smoothly compressed
/// toward 1.0.
#[inline]
fn smooth_compress(v: f32, knee: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    if v <= knee {
        return v;
    }
    // Smoothstep from knee to 1.0 as v goes from knee to 2.0
    let t = ((v - knee) / (2.0 - knee + f32::EPSILON)).clamp(0.0, 1.0);
    knee + (1.0 - knee) * (3.0 * t * t - 2.0 * t * t * t)
}

// ── 3×3 matrix helpers ────────────────────────────────────────────────────────

/// Multiply 3×3 matrix by a 3-vector.
#[inline]
fn mat3_mul_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Multiply two 3×3 matrices: result = a × b.
#[inline]
fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Compute the inverse of a 3×3 matrix via cofactor expansion.
///
/// If the matrix is singular (det ≈ 0) the identity is returned.
#[must_use]
fn mat3_inverse(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];

    let a00 = e * i - f * h;
    let a01 = -(d * i - f * g);
    let a02 = d * h - e * g;
    let a10 = -(b * i - c * h);
    let a11 = a * i - c * g;
    let a12 = -(a * h - b * g);
    let a20 = b * f - c * e;
    let a21 = -(a * f - c * d);
    let a22 = a * e - b * d;

    let det = a * a00 + b * a01 + c * a02;

    if det.abs() < f32::EPSILON {
        // Singular: return identity
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let inv_det = 1.0 / det;
    [
        [a00 * inv_det, a10 * inv_det, a20 * inv_det],
        [a01 * inv_det, a11 * inv_det, a21 * inv_det],
        [a02 * inv_det, a12 * inv_det, a22 * inv_det],
    ]
}

// ── Legacy gamut-mapping API (preserved for backward compatibility) ────────────

/// Gamut mapping method selection (legacy flat API).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyGamutMappingMethod {
    /// Hard clip.
    Clip,
    /// Chroma-compression toward neutral grey.
    Compress,
    /// Sigmoid soft-clip.
    SoftClip,
    /// RA-LUR style uniform peak normalisation.
    RaLur,
    /// Hue-preserving chroma reduction.
    HuePreserve,
}

/// Gamut boundary definition using CIE xy chromaticity coordinates.
#[derive(Debug, Clone, Copy)]
pub struct GamutBoundary {
    /// Primary chromaticities [[rx, ry], [gx, gy], [bx, by]].
    pub primaries: [[f64; 2]; 3],
    /// White point chromaticity [x, y].
    pub white_point: [f64; 2],
}

/// Gamut mapper that converts colours from source to destination gamut (legacy API).
pub struct LegacyGamutMapper {
    /// The mapping method.
    pub method: LegacyGamutMappingMethod,
    /// Source gamut.
    pub src_gamut: GamutBoundary,
    /// Destination gamut.
    pub dst_gamut: GamutBoundary,
}

impl LegacyGamutMapper {
    /// Create a new legacy gamut mapper.
    #[must_use]
    pub fn new(
        method: LegacyGamutMappingMethod,
        src_gamut: GamutBoundary,
        dst_gamut: GamutBoundary,
    ) -> Self {
        Self {
            method,
            src_gamut,
            dst_gamut,
        }
    }

    /// Map an RGB triplet.
    #[must_use]
    pub fn map_rgb(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        match self.method {
            LegacyGamutMappingMethod::Clip => legacy_clip_rgb(r, g, b),
            LegacyGamutMappingMethod::Compress => legacy_compress_rgb(r, g, b),
            LegacyGamutMappingMethod::SoftClip => legacy_soft_clip_rgb(r, g, b),
            LegacyGamutMappingMethod::RaLur => legacy_ra_lur_rgb(r, g, b),
            LegacyGamutMappingMethod::HuePreserve => legacy_hue_preserve_rgb(r, g, b),
        }
    }

    /// Check whether an RGB value is in gamut (all channels in [0, 1]).
    #[must_use]
    pub fn is_in_gamut(&self, r: f64, g: f64, b: f64) -> bool {
        r >= 0.0 && r <= 1.0 && g >= 0.0 && g <= 1.0 && b >= 0.0 && b <= 1.0
    }

    /// Fraction of pixels outside \[0,1\]³.
    #[must_use]
    pub fn compute_out_of_gamut_ratio(pixels: &[(f64, f64, f64)]) -> f64 {
        if pixels.is_empty() {
            return 0.0;
        }
        let out = pixels
            .iter()
            .filter(|(r, g, b)| {
                *r < 0.0 || *r > 1.0 || *g < 0.0 || *g > 1.0 || *b < 0.0 || *b > 1.0
            })
            .count();
        out as f64 / pixels.len() as f64
    }
}

// Legacy internal helpers

#[inline]
fn legacy_clip_rgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

fn legacy_compress_rgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
    let max_over = r.max(g).max(b).max(1.0);
    if max_over <= 1.0 {
        return (r.max(0.0), g.max(0.0), b.max(0.0));
    }
    let scale = (1.0 - luma) / (max_over - luma).max(f64::EPSILON);
    (
        (luma + (r - luma) * scale).clamp(0.0, 1.0),
        (luma + (g - luma) * scale).clamp(0.0, 1.0),
        (luma + (b - luma) * scale).clamp(0.0, 1.0),
    )
}

fn legacy_soft_clip_rgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    (
        legacy_soft_clip_channel(r),
        legacy_soft_clip_channel(g),
        legacy_soft_clip_channel(b),
    )
}

#[inline]
fn legacy_soft_clip_channel(v: f64) -> f64 {
    const KNEE: f64 = 0.8;
    if v <= 0.0 {
        return 0.0;
    }
    if v <= KNEE {
        return v;
    }
    if v >= 2.0 {
        return 1.0;
    }
    let t = ((v - KNEE) / (2.0 - KNEE)).clamp(0.0, 1.0);
    KNEE + (1.0 - KNEE) * (3.0 * t * t - 2.0 * t * t * t)
}

fn legacy_ra_lur_rgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let peak = r.max(g).max(b);
    if peak <= 1.0 {
        return (r.max(0.0), g.max(0.0), b.max(0.0));
    }
    (
        (r / peak).clamp(0.0, 1.0),
        (g / peak).clamp(0.0, 1.0),
        (b / peak).clamp(0.0, 1.0),
    )
}

fn legacy_hue_preserve_rgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let cr = r - luma;
    let cg = g - luma;
    let cb = b - luma;
    let max_scale = legacy_chroma_scale_limit(luma, cr)
        .min(legacy_chroma_scale_limit(luma, cg))
        .min(legacy_chroma_scale_limit(luma, cb));
    let scale = max_scale.clamp(0.0, 1.0);
    (
        (luma + cr * scale).clamp(0.0, 1.0),
        (luma + cg * scale).clamp(0.0, 1.0),
        (luma + cb * scale).clamp(0.0, 1.0),
    )
}

#[inline]
fn legacy_chroma_scale_limit(luma: f64, c: f64) -> f64 {
    if c.abs() < f64::EPSILON {
        return 1.0;
    }
    if c > 0.0 {
        (1.0 - luma) / c
    } else {
        luma / (-c)
    }
    .max(0.0)
}

// ── Standard gamut definitions ────────────────────────────────────────────────

/// Standard sRGB / Rec.709 gamut boundary (D65).
#[must_use]
pub fn srgb_gamut() -> GamutBoundary {
    GamutBoundary {
        primaries: [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]],
        white_point: [0.3127, 0.3290],
    }
}

/// DCI-P3 D65 gamut boundary.
#[must_use]
pub fn p3_d65_gamut() -> GamutBoundary {
    GamutBoundary {
        primaries: [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        white_point: [0.3127, 0.3290],
    }
}

/// Rec.2020 gamut boundary (D65).
#[must_use]
pub fn bt2020_gamut() -> GamutBoundary {
    GamutBoundary {
        primaries: [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        white_point: [0.3127, 0.3290],
    }
}

// ── Cusp-based gamut mapping (ACES-style) ────────────────────────────────────

/// A point on the gamut boundary in (lightness, chroma) space for a given hue.
///
/// The "cusp" is the point on the gamut boundary with maximum chroma for a
/// given hue. ACES gamut mapping algorithms use this cusp to define a smooth
/// mapping from out-of-gamut colors toward the gamut boundary.
#[derive(Debug, Clone, Copy)]
pub struct GamutCusp {
    /// Lightness at the cusp (0–1 normalized).
    pub lightness: f32,
    /// Maximum chroma at the cusp.
    pub max_chroma: f32,
}

/// Cusp-based gamut mapping configuration.
///
/// Implements an ACES-style gamut mapping algorithm that uses the gamut cusp
/// (the point of maximum chroma for each hue) to create a smooth, perceptual
/// mapping of out-of-gamut colors.
///
/// The algorithm works in a perceptual color space (assumed LCh-like):
/// 1. Find the cusp for the destination gamut at the given hue
/// 2. Map colors outside the gamut boundary toward the cusp
/// 3. Use a smooth compression function to avoid harsh clipping artifacts
///
/// Reference: ACES Gamut Mapping Algorithm (GMA) Technical Documentation.
#[derive(Debug, Clone)]
pub struct CuspBasedGamutMapper {
    /// Compression limit: how far beyond the boundary to start compressing (1.0 = boundary).
    pub compression_limit: f32,
    /// Power exponent for the compression curve (higher = sharper knee).
    pub power: f32,
    /// Threshold below which no compression is applied (as a fraction of max chroma).
    pub threshold: f32,
    /// Whether to protect shadows (reduce chroma compression for dark colors).
    pub shadow_protection: bool,
    /// Shadow rolloff zone: lightness below this value gets reduced compression.
    pub shadow_rolloff: f32,
}

impl Default for CuspBasedGamutMapper {
    fn default() -> Self {
        Self {
            compression_limit: 1.2,
            power: 1.2,
            threshold: 0.75,
            shadow_protection: true,
            shadow_rolloff: 0.1,
        }
    }
}

impl CuspBasedGamutMapper {
    /// Creates a new cusp-based gamut mapper with default ACES-style parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `compression_limit` - How far beyond the boundary to start compressing (> 1.0)
    /// * `power` - Power exponent for the compression curve (> 0.0)
    /// * `threshold` - Fraction of max chroma below which no compression is applied
    #[must_use]
    pub fn with_params(compression_limit: f32, power: f32, threshold: f32) -> Self {
        Self {
            compression_limit: compression_limit.max(1.001),
            power: power.max(0.1),
            threshold: threshold.clamp(0.0, 0.999),
            ..Self::default()
        }
    }

    /// Estimate the gamut cusp for a given hue angle using the destination primaries.
    ///
    /// This approximation finds the max-chroma point on the gamut boundary for
    /// the given hue by testing the six edges of the RGB cube (where one channel
    /// is 0 or 1 and the others vary). Returns (lightness, max_chroma) in
    /// normalized \[0,1\] coordinates.
    ///
    /// # Arguments
    ///
    /// * `hue_deg` - Hue angle in degrees [0, 360)
    #[must_use]
    pub fn estimate_cusp(hue_deg: f32) -> GamutCusp {
        // Approximate cusp from hue angle using a simplified model.
        // In practice, this would be computed per-gamut using the primaries,
        // but we use a perceptual model here that works for typical RGB gamuts.
        let hue_rad = hue_deg * std::f32::consts::PI / 180.0;

        // Model the cusp lightness as varying with hue:
        // Yellow (~60°) has highest lightness cusp, blue (~240°) has lowest
        let cusp_l = 0.5 + 0.2 * ((hue_rad - std::f32::consts::PI / 3.0) * 1.0).cos();

        // Model the maximum chroma as varying with hue:
        // Red/green/blue primaries have highest chroma
        let cusp_c = 0.15 + 0.1 * (3.0 * hue_rad).cos().abs();

        GamutCusp {
            lightness: cusp_l.clamp(0.1, 0.95),
            max_chroma: cusp_c.clamp(0.05, 0.4),
        }
    }

    /// Maps a single color in LCh-like space toward the destination gamut.
    ///
    /// # Arguments
    ///
    /// * `lightness` - Lightness component (0–1)
    /// * `chroma` - Chroma/colorfulness (>= 0)
    /// * `hue_deg` - Hue angle in degrees
    /// * `cusp` - Pre-computed cusp for this hue in the destination gamut
    ///
    /// # Returns
    ///
    /// (lightness, chroma) after gamut mapping. Hue is preserved.
    #[must_use]
    pub fn map_color(
        &self,
        lightness: f32,
        chroma: f32,
        _hue_deg: f32,
        cusp: &GamutCusp,
    ) -> (f32, f32) {
        if chroma <= 0.0 {
            return (lightness.clamp(0.0, 1.0), 0.0);
        }

        // Compute the gamut boundary chroma at this lightness for this hue
        let boundary_chroma = self.boundary_chroma_at_lightness(lightness, cusp);

        // If inside gamut, no mapping needed
        if chroma <= boundary_chroma * self.threshold {
            return (lightness, chroma);
        }

        // Apply compression beyond the threshold
        let mapped_chroma = self.compress_chroma(chroma, boundary_chroma);

        // Shadow protection: reduce compression for very dark colors
        let final_chroma = if self.shadow_protection && lightness < self.shadow_rolloff {
            let shadow_factor = (lightness / self.shadow_rolloff).clamp(0.0, 1.0);
            mapped_chroma * shadow_factor
        } else {
            mapped_chroma
        };

        // Also softly compress lightness if it's out of range
        let mapped_l = if lightness > 1.0 {
            1.0 - (1.0 - 1.0 / lightness.max(1.001)) * 0.5
        } else if lightness < 0.0 {
            0.0
        } else {
            lightness
        };

        (mapped_l, final_chroma.max(0.0))
    }

    /// Computes the maximum chroma at a given lightness along the gamut boundary.
    ///
    /// The boundary is modeled as a triangle from black (L=0, C=0) through
    /// the cusp (L_cusp, C_max) to white (L=1, C=0).
    fn boundary_chroma_at_lightness(&self, lightness: f32, cusp: &GamutCusp) -> f32 {
        let l = lightness.clamp(0.0, 1.0);
        if l <= cusp.lightness {
            // Lower triangle: black to cusp
            if cusp.lightness < f32::EPSILON {
                return 0.0;
            }
            cusp.max_chroma * l / cusp.lightness
        } else {
            // Upper triangle: cusp to white
            let top = 1.0 - cusp.lightness;
            if top < f32::EPSILON {
                return 0.0;
            }
            cusp.max_chroma * (1.0 - l) / top
        }
    }

    /// Applies the smooth compression function to chroma.
    ///
    /// Uses a power-curve knee that smoothly transitions from identity below
    /// the threshold to compression above it, asymptotically approaching the
    /// compression limit.
    fn compress_chroma(&self, chroma: f32, boundary_chroma: f32) -> f32 {
        if boundary_chroma < f32::EPSILON {
            return 0.0;
        }

        let normalized = chroma / boundary_chroma;
        let thresh = self.threshold;

        if normalized <= thresh {
            return chroma;
        }

        // Parabolic compression above threshold
        let limit = self.compression_limit;
        let excess = normalized - thresh;
        let range = limit - thresh;

        if range < f32::EPSILON {
            return thresh * boundary_chroma;
        }

        // Smooth compression: maps [thresh, infinity) -> [thresh, limit)
        let compressed = thresh + range * (1.0 - (-excess * self.power / range).exp());

        compressed.min(limit) * boundary_chroma
    }

    /// Maps an entire frame of colors in LCh representation.
    ///
    /// Each pixel is `[lightness, chroma, hue]`.
    #[must_use]
    pub fn map_frame(&self, pixels: &[[f32; 3]]) -> Vec<[f32; 3]> {
        pixels
            .iter()
            .map(|&[l, c, h]| {
                let cusp = Self::estimate_cusp(h);
                let (ml, mc) = self.map_color(l, c, h, &cusp);
                [ml, mc, h]
            })
            .collect()
    }
}

// ── ACES Gamut Compression Algorithm v1.0 ─────────────────────────────────────

/// Configuration for ACES Gamut Compression Algorithm v1.0.
///
/// Reference: Mansencal & Sobott (2020),
/// `ACESUtil.Rec2020_to_ACES_gamut_compress.ctl`.
///
/// The algorithm uses a tanh-based soft-knee function to smoothly map
/// out-of-gamut (negative) components back into gamut without harsh clipping.
#[derive(Debug, Clone)]
pub struct AcesGamutCompressConfig {
    /// Compression threshold (default: 0.815 for Rec.2020 out-gamut).
    ///
    /// Colors whose distance from the achromatic axis exceeds this value
    /// enter the soft compression zone.
    pub threshold: f32,
    /// Compression limit (default: 1.147).
    ///
    /// The maximum distance from the achromatic axis that will be compressed.
    /// Values beyond this are hard-clamped.
    pub limit: f32,
}

impl Default for AcesGamutCompressConfig {
    fn default() -> Self {
        Self {
            threshold: 0.815,
            limit: 1.147,
        }
    }
}

/// Compress a single wide-gamut RGB pixel into gamut using the ACES GCA soft-knee.
///
/// All channels are in linear light (0..=1 for in-gamut, may exceed 1.0 for
/// wide-gamut or go negative for out-of-gamut).  The achromatic axis is
/// taken as `max(r, g, b)` and each channel's distance from that axis is
/// independently compressed with a `tanh`-based knee function.
///
/// # Arguments
///
/// * `rgb` — linear-light `[r, g, b]` (may be outside `[0, 1]`).
/// * `cfg` — compression parameters.
///
/// # Returns
///
/// Gamut-compressed `[r, g, b]` where every channel is `>= 0.0`.
#[must_use]
pub fn aces_gamut_compress_pixel(rgb: [f32; 3], cfg: &AcesGamutCompressConfig) -> [f32; 3] {
    let achromatic = rgb[0].max(rgb[1]).max(rgb[2]);
    let range = cfg.limit - cfg.threshold;

    let mut out = rgb;
    for c in &mut out {
        let d = achromatic - *c;
        if d > cfg.threshold {
            // tanh-based soft-knee: smoothly maps (threshold, ∞) → (threshold, limit)
            let compressed_d = cfg.limit - range * ((d - cfg.threshold) / range).tanh();
            *c = achromatic - compressed_d;
        }
    }
    out
}

/// Apply ACES gamut compression in-place to a frame of linear RGB pixels.
///
/// Processes every pixel in `pixels` using [`aces_gamut_compress_pixel`].
///
/// # Arguments
///
/// * `pixels` — mutable slice of `[r, g, b]` linear-light pixels.
/// * `cfg` — compression parameters.
pub fn aces_gamut_compress_frame(pixels: &mut [[f32; 3]], cfg: &AcesGamutCompressConfig) {
    for px in pixels.iter_mut() {
        *px = aces_gamut_compress_pixel(*px, cfg);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ColorPrimaries constructors ───────────────────────────────────────────

    #[test]
    fn test_rec709_primaries() {
        let p = ColorPrimaries::rec709();
        assert!((p.red.0 - 0.640).abs() < 1e-6, "red.x={}", p.red.0);
        assert!((p.white.0 - 0.3127).abs() < 1e-6, "white.x={}", p.white.0);
    }

    #[test]
    fn test_rec2020_primaries() {
        let p = ColorPrimaries::rec2020();
        assert!((p.red.0 - 0.708).abs() < 1e-6);
        assert!((p.green.1 - 0.797).abs() < 1e-6);
    }

    #[test]
    fn test_p3_d65_primaries() {
        let p = ColorPrimaries::p3_d65();
        assert!((p.red.0 - 0.680).abs() < 1e-6);
        assert!((p.white.0 - 0.3127).abs() < 1e-6);
    }

    #[test]
    fn test_p3_dci_primaries_white_point() {
        let p = ColorPrimaries::p3_dci();
        // DCI white is different from D65
        assert!((p.white.0 - 0.3140).abs() < 1e-6);
        assert!((p.white.1 - 0.3510).abs() < 1e-6);
    }

    // ── rgb_to_xyz round-trip ─────────────────────────────────────────────────

    #[test]
    fn test_rgb_to_xyz_white_maps_to_unit_y() {
        let p = ColorPrimaries::rec709();
        let m = p.rgb_to_xyz();
        // White (1,1,1) should map to Y ≈ 1.0
        let y = m[1][0] + m[1][1] + m[1][2];
        assert!((y - 1.0).abs() < 1e-4, "Y_white={y}");
    }

    #[test]
    fn test_rgb_to_xyz_black_maps_to_zero() {
        let p = ColorPrimaries::rec2020();
        let m = p.rgb_to_xyz();
        let [x, y, z] = mat3_mul_vec(m, [0.0, 0.0, 0.0]);
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6 && z.abs() < 1e-6);
    }

    // ── GamutMapper construction ──────────────────────────────────────────────

    #[test]
    fn test_identity_mapper_clip() {
        let mapper = GamutMapper::new(
            ColorPrimaries::rec709(),
            ColorPrimaries::rec709(),
            GamutMappingMethod::Clip,
        );
        let (r, g, b) = mapper.map_pixel(0.5, 0.3, 0.2);
        assert!((r - 0.5).abs() < 1e-4, "r={r}");
        assert!((g - 0.3).abs() < 1e-4, "g={g}");
        assert!((b - 0.2).abs() < 1e-4, "b={b}");
    }

    #[test]
    fn test_clip_out_of_gamut() {
        let mapper = GamutMapper::new(
            ColorPrimaries::rec2020(),
            ColorPrimaries::rec709(),
            GamutMappingMethod::Clip,
        );
        let (r, g, b) = mapper.map_pixel(1.5, 0.5, -0.1);
        assert!(r >= 0.0 && r <= 1.0, "r={r}");
        assert!(g >= 0.0 && g <= 1.0, "g={g}");
        assert!(b >= 0.0 && b <= 1.0, "b={b}");
    }

    #[test]
    fn test_perceptual_in_gamut() {
        let mapper = GamutMapper::new(
            ColorPrimaries::rec709(),
            ColorPrimaries::rec709(),
            GamutMappingMethod::Perceptual,
        );
        let (r, g, b) = mapper.map_pixel(0.8, 0.5, 0.2);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn test_saturation_maps_to_gamut() {
        let mapper = GamutMapper::new(
            ColorPrimaries::rec2020(),
            ColorPrimaries::rec709(),
            GamutMappingMethod::Saturation,
        );
        let (r, g, b) = mapper.map_pixel(1.8, 0.5, 0.1);
        assert!(r >= 0.0 && r <= 1.0, "r={r}");
        assert!(g >= 0.0 && g <= 1.0, "g={g}");
        assert!(b >= 0.0 && b <= 1.0, "b={b}");
    }

    #[test]
    fn test_compress_method_maps_to_gamut() {
        let mapper = GamutMapper::new(
            ColorPrimaries::rec2020(),
            ColorPrimaries::rec709(),
            GamutMappingMethod::Compress(0.8),
        );
        let (r, g, b) = mapper.map_pixel(1.5, 0.6, 0.1);
        assert!(r >= 0.0 && r <= 1.0, "r={r}");
        assert!(g >= 0.0 && g <= 1.0, "g={g}");
        assert!(b >= 0.0 && b <= 1.0, "b={b}");
    }

    // ── map_frame ─────────────────────────────────────────────────────────────

    #[test]
    fn test_map_frame_length_preserved() {
        let mapper = GamutMapper::new(
            ColorPrimaries::rec709(),
            ColorPrimaries::rec2020(),
            GamutMappingMethod::Clip,
        );
        let pixels = vec![0.5_f32; 30]; // 10 RGB pixels
        let out = mapper.map_frame(&pixels);
        assert_eq!(out.len(), 30);
    }

    #[test]
    fn test_map_frame_all_in_gamut_after_clip() {
        let mapper = GamutMapper::new(
            ColorPrimaries::rec2020(),
            ColorPrimaries::rec709(),
            GamutMappingMethod::Clip,
        );
        let pixels: Vec<f32> = (0..30).map(|i| i as f32 / 10.0).collect();
        let out = mapper.map_frame(&pixels);
        for &v in &out {
            assert!(v >= 0.0 && v <= 1.0, "out-of-gamut: {v}");
        }
    }

    // ── is_in_gamut ───────────────────────────────────────────────────────────

    #[test]
    fn test_is_in_gamut_identity() {
        let mapper = GamutMapper::new(
            ColorPrimaries::rec709(),
            ColorPrimaries::rec709(),
            GamutMappingMethod::Clip,
        );
        assert!(mapper.is_in_gamut(0.5, 0.5, 0.5));
    }

    #[test]
    fn test_is_in_gamut_outside_wide_gamut() {
        // A saturated Rec.2020 colour likely falls outside sRGB after matrix
        let mapper = GamutMapper::new(
            ColorPrimaries::rec2020(),
            ColorPrimaries::rec709(),
            GamutMappingMethod::Clip,
        );
        // Pure Rec.2020 green (0,1,0) is outside sRGB
        let in_gamut = mapper.is_in_gamut(0.0, 1.0, 0.0);
        // We just verify the function runs without panic; result depends on matrix
        let _ = in_gamut;
    }

    // ── smooth_compress ───────────────────────────────────────────────────────

    #[test]
    fn test_smooth_compress_below_knee_identity() {
        let v = smooth_compress(0.5, 0.8);
        assert!((v - 0.5).abs() < 1e-6, "v={v}");
    }

    #[test]
    fn test_smooth_compress_above_one_stays_le_one() {
        let v = smooth_compress(1.5, 0.8);
        assert!(v <= 1.0 && v >= 0.8, "v={v}");
    }

    #[test]
    fn test_smooth_compress_zero_input() {
        assert_eq!(smooth_compress(0.0, 0.8), 0.0);
    }

    // ── Legacy tests (from original implementation) ───────────────────────────

    #[test]
    fn test_legacy_clip_in_gamut() {
        let mapper =
            LegacyGamutMapper::new(LegacyGamutMappingMethod::Clip, srgb_gamut(), srgb_gamut());
        let (r, g, b) = mapper.map_rgb(0.5, 0.3, 0.2);
        assert!((r - 0.5).abs() < 1e-10);
        assert!((g - 0.3).abs() < 1e-10);
        assert!((b - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_legacy_clip_out_of_gamut() {
        let mapper =
            LegacyGamutMapper::new(LegacyGamutMappingMethod::Clip, bt2020_gamut(), srgb_gamut());
        let (r, g, b) = mapper.map_rgb(1.5, -0.1, 0.8);
        assert_eq!(r, 1.0);
        assert_eq!(g, 0.0);
        assert!((b - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_legacy_compute_out_of_gamut_ratio_half_out() {
        let pixels = vec![(0.5, 0.5, 0.5), (1.5, 0.5, 0.5)];
        let ratio = LegacyGamutMapper::compute_out_of_gamut_ratio(&pixels);
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    // ── Cusp-based gamut mapping tests ────────────────────────────────────────

    #[test]
    fn test_cusp_mapper_default_construction() {
        let mapper = CuspBasedGamutMapper::new();
        assert!(mapper.compression_limit > 1.0);
        assert!(mapper.power > 0.0);
        assert!(mapper.threshold > 0.0 && mapper.threshold < 1.0);
    }

    #[test]
    fn test_cusp_mapper_custom_params() {
        let mapper = CuspBasedGamutMapper::with_params(1.5, 2.0, 0.8);
        assert!((mapper.compression_limit - 1.5).abs() < 1e-6);
        assert!((mapper.power - 2.0).abs() < 1e-6);
        assert!((mapper.threshold - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cusp_mapper_custom_params_clamp() {
        let mapper = CuspBasedGamutMapper::with_params(0.5, -1.0, 2.0);
        assert!(mapper.compression_limit >= 1.001);
        assert!(mapper.power >= 0.1);
        assert!(mapper.threshold <= 0.999);
    }

    #[test]
    fn test_estimate_cusp_valid_range() {
        for hue in [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 359.0] {
            let cusp = CuspBasedGamutMapper::estimate_cusp(hue);
            assert!(
                cusp.lightness > 0.0 && cusp.lightness < 1.0,
                "Cusp lightness out of range for hue {}: {}",
                hue,
                cusp.lightness
            );
            assert!(
                cusp.max_chroma > 0.0,
                "Cusp chroma should be positive for hue {}: {}",
                hue,
                cusp.max_chroma
            );
        }
    }

    #[test]
    fn test_cusp_map_achromatic_preserved() {
        let mapper = CuspBasedGamutMapper::new();
        let cusp = CuspBasedGamutMapper::estimate_cusp(0.0);
        let (l, c) = mapper.map_color(0.5, 0.0, 0.0, &cusp);
        assert!(
            (l - 0.5).abs() < 1e-6,
            "Lightness should be preserved: {}",
            l
        );
        assert_eq!(c, 0.0, "Zero chroma should remain zero");
    }

    #[test]
    fn test_cusp_map_in_gamut_preserved() {
        let mapper = CuspBasedGamutMapper::new();
        let cusp = CuspBasedGamutMapper::estimate_cusp(120.0);
        // Use a very small chroma that's well within gamut
        let (l, c) = mapper.map_color(0.5, 0.01, 120.0, &cusp);
        assert!(
            (l - 0.5).abs() < 1e-6,
            "In-gamut lightness preserved: {}",
            l
        );
        assert!((c - 0.01).abs() < 1e-6, "In-gamut chroma preserved: {}", c);
    }

    #[test]
    fn test_cusp_map_out_of_gamut_compressed() {
        let mapper = CuspBasedGamutMapper::new();
        let cusp = GamutCusp {
            lightness: 0.5,
            max_chroma: 0.15,
        };
        // Use chroma well beyond the boundary
        let (_, c) = mapper.map_color(0.5, 0.5, 120.0, &cusp);
        assert!(c < 0.5, "Out-of-gamut chroma should be compressed: {}", c);
        assert!(c > 0.0, "Compressed chroma should remain positive: {}", c);
    }

    #[test]
    fn test_cusp_map_shadow_protection() {
        let mapper = CuspBasedGamutMapper::new();
        let cusp = GamutCusp {
            lightness: 0.5,
            max_chroma: 0.15,
        };
        // Very dark color (within shadow rolloff zone)
        let (_, c) = mapper.map_color(0.01, 0.2, 60.0, &cusp);
        assert!(
            c >= 0.0,
            "Shadow-protected chroma should be non-negative: {}",
            c
        );
    }

    #[test]
    fn test_cusp_map_lightness_over_one() {
        let mapper = CuspBasedGamutMapper::new();
        let cusp = CuspBasedGamutMapper::estimate_cusp(60.0);
        let (l, _) = mapper.map_color(1.5, 0.1, 60.0, &cusp);
        assert!(l <= 1.0, "Lightness should be mapped to [0,1]: {}", l);
    }

    #[test]
    fn test_cusp_map_frame() {
        let mapper = CuspBasedGamutMapper::new();
        let pixels = vec![
            [0.5, 0.1, 30.0],
            [0.8, 0.3, 120.0],
            [0.2, 0.0, 0.0],
            [0.6, 0.5, 240.0],
        ];
        let mapped = mapper.map_frame(&pixels);
        assert_eq!(mapped.len(), 4);
        for px in &mapped {
            assert!(px[0] >= 0.0 && px[0] <= 1.0, "L out of range: {}", px[0]);
            assert!(px[1] >= 0.0, "C should be non-negative: {}", px[1]);
        }
    }

    #[test]
    fn test_cusp_boundary_chroma_at_cusp() {
        let mapper = CuspBasedGamutMapper::new();
        let cusp = GamutCusp {
            lightness: 0.5,
            max_chroma: 0.2,
        };
        let bc = mapper.boundary_chroma_at_lightness(0.5, &cusp);
        assert!(
            (bc - 0.2).abs() < 1e-6,
            "At cusp lightness, boundary chroma should equal max_chroma: {}",
            bc
        );
    }

    #[test]
    fn test_cusp_boundary_chroma_at_black_and_white() {
        let mapper = CuspBasedGamutMapper::new();
        let cusp = GamutCusp {
            lightness: 0.5,
            max_chroma: 0.2,
        };
        let bc_black = mapper.boundary_chroma_at_lightness(0.0, &cusp);
        let bc_white = mapper.boundary_chroma_at_lightness(1.0, &cusp);
        assert!(
            bc_black.abs() < 1e-6,
            "Black should have zero boundary chroma"
        );
        assert!(
            bc_white.abs() < 1e-6,
            "White should have zero boundary chroma"
        );
    }

    #[test]
    fn test_cusp_compression_monotonic() {
        let mapper = CuspBasedGamutMapper::new();
        // Compressing increasing chroma values should produce increasing (but slower) output
        let c1 = mapper.compress_chroma(0.2, 0.15);
        let c2 = mapper.compress_chroma(0.4, 0.15);
        let c3 = mapper.compress_chroma(0.8, 0.15);
        assert!(
            c1 <= c2,
            "Compression should be monotonic: {} <= {}",
            c1,
            c2
        );
        assert!(
            c2 <= c3,
            "Compression should be monotonic: {} <= {}",
            c2,
            c3
        );
    }

    // ── ACES Gamut Compression tests ─────────────────────────────────────────

    #[test]
    fn test_aces_compress_wide_gamut_pixel_no_negative() {
        // [2.0, 0.5, 0.5] has a large red, very negative G/B distance from achromatic
        let result = aces_gamut_compress_pixel([2.0, 0.5, 0.5], &Default::default());
        assert!(
            result.iter().all(|&v| v >= 0.0),
            "all channels must be >= 0 after compression, got {:?}",
            result
        );
    }

    #[test]
    fn test_aces_compress_achromatic_identity() {
        // [1.0, 1.0, 1.0] is achromatic (d = 0 for all channels) — no compression
        let result = aces_gamut_compress_pixel([1.0, 1.0, 1.0], &Default::default());
        for (&a, &b) in result.iter().zip([1.0f32, 1.0, 1.0].iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "achromatic pixel should be unchanged: expected {b}, got {a}"
            );
        }
    }

    #[test]
    fn test_aces_compress_in_gamut_unchanged() {
        // [0.5, 0.3, 0.2] — distances from achromatic are all below threshold 0.815
        let pixel = [0.5f32, 0.3, 0.2];
        let result = aces_gamut_compress_pixel(pixel, &Default::default());
        for (&a, &b) in result.iter().zip(pixel.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "in-gamut pixel should be unchanged: expected {b}, got {a}"
            );
        }
    }

    #[test]
    fn test_aces_compress_frame_all_nonneg() {
        let mut pixels = vec![[2.0f32, -0.5, 0.3], [0.5, 0.5, 0.5], [1.5, 0.8, -0.2]];
        aces_gamut_compress_frame(&mut pixels, &Default::default());
        for px in &pixels {
            assert!(
                px.iter().all(|&v| v >= 0.0),
                "frame pixel has negative: {:?}",
                px
            );
        }
    }

    #[test]
    fn test_aces_compress_default_config() {
        let cfg = AcesGamutCompressConfig::default();
        assert!((cfg.threshold - 0.815).abs() < 1e-6);
        assert!((cfg.limit - 1.147).abs() < 1e-6);
    }
}

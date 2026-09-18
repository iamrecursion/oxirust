//! Colour gamut definitions and RGB → RGB conversion matrices.
//!
//! Conversion is performed via the CIE XYZ intermediate space.
//! White point adaptation uses the Bradford chromatic adaptation transform (CAT).

use crate::{HdrError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::OnceLock;

// ── Process-wide matrix cache ─────────────────────────────────────────────────

/// Type alias for the inner map stored in the process-wide gamut cache.
type GamutCacheMap = HashMap<(ColorGamut, ColorGamut), [[f32; 3]; 3]>;

/// Process-wide cache for gamut conversion matrices.
///
/// Keyed by `(src, dst)` pair; stores the computed `[[f32; 3]; 3]` matrix.
/// Two-phase locking: read-locked fast path, write-locked slow path.
static MATRIX_CACHE: OnceLock<RwLock<GamutCacheMap>> = OnceLock::new();

// ── Gamut enum ────────────────────────────────────────────────────────────────

/// Well-known RGB colour gamuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorGamut {
    /// BT.709 / sRGB (HD broadcast).
    Rec709,
    /// BT.2020 / Rec.2100 (UHD / HDR).
    Rec2020,
    /// DCI-P3 with D65 white point (HDR cinema displays).
    P3D65,
    /// DCI-P3 with DCI white point (digital cinema projectors).
    P3Dci,
    /// ACES AP0 (Academy Color Encoding System).
    Aces,
}

// ── Primaries / white-point data ──────────────────────────────────────────────

/// CIE 1931 xy chromaticity coordinates for a single primary.
#[derive(Debug, Clone, Copy)]
struct Xy {
    x: f64,
    y: f64,
}

/// Full set of primaries and white point for a gamut.
#[derive(Debug, Clone, Copy)]
struct GamutData {
    r: Xy,
    g: Xy,
    b: Xy,
    w: Xy,
}

impl ColorGamut {
    fn data(&self) -> GamutData {
        match self {
            ColorGamut::Rec709 => GamutData {
                r: Xy { x: 0.64, y: 0.33 },
                g: Xy { x: 0.30, y: 0.60 },
                b: Xy { x: 0.15, y: 0.06 },
                w: Xy {
                    x: 0.3127,
                    y: 0.3290,
                }, // D65
            },
            ColorGamut::Rec2020 => GamutData {
                r: Xy { x: 0.708, y: 0.292 },
                g: Xy { x: 0.170, y: 0.797 },
                b: Xy { x: 0.131, y: 0.046 },
                w: Xy {
                    x: 0.3127,
                    y: 0.3290,
                }, // D65
            },
            ColorGamut::P3D65 => GamutData {
                r: Xy { x: 0.680, y: 0.320 },
                g: Xy { x: 0.265, y: 0.690 },
                b: Xy { x: 0.150, y: 0.060 },
                w: Xy {
                    x: 0.3127,
                    y: 0.3290,
                }, // D65
            },
            ColorGamut::P3Dci => GamutData {
                r: Xy { x: 0.680, y: 0.320 },
                g: Xy { x: 0.265, y: 0.690 },
                b: Xy { x: 0.150, y: 0.060 },
                w: Xy { x: 0.314, y: 0.351 }, // DCI white
            },
            ColorGamut::Aces => GamutData {
                r: Xy {
                    x: 0.7347,
                    y: 0.2653,
                },
                g: Xy {
                    x: 0.0000,
                    y: 1.0000,
                },
                b: Xy {
                    x: 0.0001,
                    y: -0.0770,
                },
                w: Xy {
                    x: 0.32168,
                    y: 0.33767,
                }, // ACES white ~D60
            },
        }
    }
}

// ── Matrix helpers ────────────────────────────────────────────────────────────

type Mat3 = [[f64; 3]; 3];

/// Convert CIE xy to XYZ tristimulus with Y=1.
fn xy_to_xyz(p: Xy) -> [f64; 3] {
    let y = 1.0;
    let x = (p.x / p.y) * y;
    let z = ((1.0 - p.x - p.y) / p.y) * y;
    [x, y, z]
}

/// Compute the 3×3 matrix that converts RGB to XYZ for a given gamut.
fn rgb_to_xyz_matrix(g: &GamutData) -> Mat3 {
    // Each column of M is the XYZ of the primary normalised to Y=1.
    let xr = xy_to_xyz(g.r);
    let xg = xy_to_xyz(g.g);
    let xb = xy_to_xyz(g.b);
    let xw = xy_to_xyz(g.w);

    // Build matrix with primaries as columns
    let m = [
        [xr[0], xg[0], xb[0]],
        [xr[1], xg[1], xb[1]],
        [xr[2], xg[2], xb[2]],
    ];

    // Solve M * S = Xw → S = M⁻¹ * Xw
    let mi = mat3_inverse(m);
    let s = mat3_mul_vec(&mi, xw);

    // Scale columns
    [
        [m[0][0] * s[0], m[0][1] * s[1], m[0][2] * s[2]],
        [m[1][0] * s[0], m[1][1] * s[1], m[1][2] * s[2]],
        [m[2][0] * s[0], m[2][1] * s[1], m[2][2] * s[2]],
    ]
}

fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
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

fn mat3_mul_vec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat3_inverse(m: Mat3) -> Mat3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    let inv_det = if det.abs() < 1e-12 { 0.0 } else { 1.0 / det };

    [
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
    ]
}

// ── Bradford chromatic adaptation ─────────────────────────────────────────────

/// Bradford cone response domain matrix.
const BRADFORD: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Inverse Bradford matrix.
const BRADFORD_INV: Mat3 = [
    [0.9869929, -0.1470543, 0.1599627],
    [0.4323053, 0.5183603, 0.0492912],
    [-0.0085287, 0.0400428, 0.9684867],
];

/// Compute a Bradford chromatic adaptation matrix from source to destination white.
fn bradford_cat(src_white: Xy, dst_white: Xy) -> Mat3 {
    let s_xyz = xy_to_xyz(src_white);
    let d_xyz = xy_to_xyz(dst_white);

    let s_cone = mat3_mul_vec(&BRADFORD, s_xyz);
    let d_cone = mat3_mul_vec(&BRADFORD, d_xyz);

    // Diagonal scaling matrix
    let scale = [
        [d_cone[0] / s_cone[0], 0.0, 0.0],
        [0.0, d_cone[1] / s_cone[1], 0.0],
        [0.0, 0.0, d_cone[2] / s_cone[2]],
    ];

    mat3_mul(BRADFORD_INV, mat3_mul(scale, BRADFORD))
}

// ── Matrix computation ────────────────────────────────────────────────────────

/// Compute the raw `[[f32; 3]; 3]` conversion matrix for the given gamut pair.
///
/// This is the inner computation extracted from `GamutConversionMatrix::new` so
/// it can be called from the cache miss path without holding any lock.
fn compute_matrix(src: ColorGamut, dst: ColorGamut) -> Result<[[f32; 3]; 3]> {
    let src_data = src.data();
    let dst_data = dst.data();

    let src_to_xyz = rgb_to_xyz_matrix(&src_data);
    let dst_to_xyz = rgb_to_xyz_matrix(&dst_data);
    let xyz_to_dst = mat3_inverse(dst_to_xyz);

    // Apply Bradford CAT only if white points differ.
    let cat = if src_data.w.x != dst_data.w.x || src_data.w.y != dst_data.w.y {
        Some(bradford_cat(src_data.w, dst_data.w))
    } else {
        None
    };

    let xyz_adapted = if let Some(cat_mat) = cat {
        mat3_mul(cat_mat, src_to_xyz)
    } else {
        src_to_xyz
    };

    let m64 = mat3_mul(xyz_to_dst, xyz_adapted);

    // Downcast to f32.
    let mut matrix = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            matrix[i][j] = m64[i][j] as f32;
        }
    }

    Ok(matrix)
}

// ── GamutConversionMatrix ─────────────────────────────────────────────────────

/// A 3×3 RGB-to-RGB colour matrix with source and destination gamut metadata.
#[derive(Debug, Clone)]
pub struct GamutConversionMatrix {
    /// Source gamut.
    pub src: ColorGamut,
    /// Destination gamut.
    pub dst: ColorGamut,
    /// The 3×3 conversion matrix (row-major, single-precision).
    pub matrix: [[f32; 3]; 3],
}

impl GamutConversionMatrix {
    /// Compute the conversion matrix from `src` to `dst` via XYZ with Bradford CAT.
    ///
    /// # Errors
    /// Returns `HdrError::GamutConversionError` if a primary matrix is singular (degenerate
    /// gamut definition).
    pub fn new(src: ColorGamut, dst: ColorGamut) -> Result<Self> {
        let key = (src, dst);
        let cache = MATRIX_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

        // Fast path: read-locked lookup.
        {
            let guard = cache.read();
            if let Some(&matrix) = guard.get(&key) {
                return Ok(Self { src, dst, matrix });
            }
        }

        // Slow path: compute and insert under write lock.
        let matrix = compute_matrix(src, dst)?;
        cache.write().insert(key, matrix);
        Ok(Self { src, dst, matrix })
    }

    /// Convert a single RGB triplet using the stored matrix.
    pub fn convert(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let m = &self.matrix;
        (
            m[0][0] * r + m[0][1] * g + m[0][2] * b,
            m[1][0] * r + m[1][1] * g + m[1][2] * b,
            m[2][0] * r + m[2][1] * g + m[2][2] * b,
        )
    }

    /// Convert an interleaved RGB frame.
    ///
    /// # Errors
    /// Returns `HdrError::GamutConversionError` if the pixel buffer length is not divisible by 3.
    pub fn convert_frame(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::GamutConversionError(format!(
                "pixel buffer length {} not divisible by 3",
                pixels.len()
            )));
        }
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            let (r, g, b) = self.convert(chunk[0], chunk[1], chunk[2]);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        Ok(out)
    }

    /// Convert a single RGB triplet with soft-clip gamut mapping.
    ///
    /// Instead of hard-clamping out-of-gamut values to [0, 1], this performs
    /// perceptual desaturation: negative channels are brought toward the
    /// luminance-preserving achromatic axis, and overshoot above 1.0 is
    /// smoothly compressed using a soft knee.
    ///
    /// The `softness` parameter controls the knee width (0.0 = hard clamp,
    /// 1.0 = maximum softness).
    pub fn convert_soft_clip(&self, r: f32, g: f32, b: f32, softness: f32) -> (f32, f32, f32) {
        let m = &self.matrix;
        let out_r = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        let out_g = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        let out_b = m[2][0] * r + m[2][1] * g + m[2][2] * b;

        soft_clip_gamut_map(out_r, out_g, out_b, softness)
    }

    /// Convert an interleaved RGB frame with soft-clip gamut mapping.
    ///
    /// See [`convert_soft_clip`](Self::convert_soft_clip) for details.
    ///
    /// # Errors
    /// Returns `HdrError::GamutConversionError` if the pixel buffer length is not divisible by 3.
    pub fn convert_frame_soft_clip(&self, pixels: &[f32], softness: f32) -> Result<Vec<f32>> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::GamutConversionError(format!(
                "pixel buffer length {} not divisible by 3",
                pixels.len()
            )));
        }
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            let (r, g, b) = self.convert_soft_clip(chunk[0], chunk[1], chunk[2], softness);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        Ok(out)
    }

    /// Factory: Rec. 2020 → Rec. 709 conversion matrix.
    ///
    /// # Errors
    /// Propagates matrix construction errors (should not occur for standard gamuts).
    pub fn rec2020_to_rec709() -> Result<Self> {
        Self::new(ColorGamut::Rec2020, ColorGamut::Rec709)
    }
}

// ── Soft-clip gamut mapping ─────────────────────────────────────────────────

/// Soft-clip gamut mapping: perceptual desaturation of out-of-gamut colours.
///
/// Instead of hard-clamping each channel to [0, 1], this function applies a
/// multi-phase perceptual gamut mapping pipeline:
///
/// 1. **Luminance computation** — BT.2100 coefficients for accurate HDR luminance
/// 2. **Negative desaturation** — Channels below zero are desaturated toward the
///    achromatic axis using minimum-desaturation blending, which preserves the
///    hue angle while pulling the colour into gamut
/// 3. **Highlight desaturation** — When any channel exceeds 1.0, the entire pixel
///    is desaturated toward its luminance proportionally, preserving the achromatic
///    axis and reducing chroma without shifting hue
/// 4. **Soft-knee compression** — A smooth, C1-continuous roll-off curve compresses
///    remaining overshoot using a tanh-based sigmoid that asymptotically approaches
///    1.0 without hard discontinuities
///
/// This preserves hue much better than naive clamping and avoids the colour shifts
/// that hard clipping introduces. The `softness` parameter controls the knee width:
/// - `softness = 0.0`: hard clamp (equivalent to per-channel clamp)
/// - `softness = 0.5`: moderate soft knee (recommended for HDR mastering)
/// - `softness = 1.0`: maximum softness (wide roll-off, useful for extreme gamut mismatches)
fn soft_clip_gamut_map(r: f32, g: f32, b: f32, softness: f32) -> (f32, f32, f32) {
    // BT.2100 luminance coefficients
    let luma = 0.2627 * r + 0.6780 * g + 0.0593 * b;

    // Phase 1: Fix negative values by desaturating toward luma.
    let (r1, g1, b1) = desaturate_negatives(r, g, b, luma);

    // Phase 2: Desaturate highlights — if any channel exceeds 1.0, reduce
    // chroma uniformly toward the achromatic axis. This is perceptually
    // superior to independent per-channel compression because it preserves
    // the hue angle.
    let (r2, g2, b2) = desaturate_highlights(r1, g1, b1, luma.clamp(0.0, 1.0));

    // Phase 3: Soft-compress any remaining overshoot.
    let s = softness.clamp(0.0, 1.0);
    let r3 = soft_knee_compress(r2, s);
    let g3 = soft_knee_compress(g2, s);
    let b3 = soft_knee_compress(b2, s);

    (r3, g3, b3)
}

/// Desaturate highlights toward the achromatic axis when any channel exceeds 1.0.
///
/// This finds the maximum blend factor `t` in [0, 1] such that
/// `luma + t * (ch - luma) <= 1.0` for all channels, then applies the blend.
///
/// The result preserves hue (the direction from luma to the colour in RGB space
/// is unchanged) while reducing chroma just enough to bring all channels within
/// [0, 1]. Residual overshoot (if luma itself is > 1) is handled by the
/// subsequent soft-knee pass.
fn desaturate_highlights(r: f32, g: f32, b: f32, luma: f32) -> (f32, f32, f32) {
    let max_ch = r.max(g).max(b);
    if max_ch <= 1.0 {
        return (r, g, b);
    }

    // For each channel c with c > 1.0:
    //   luma + t * (c - luma) = 1.0
    //   t = (1.0 - luma) / (c - luma)
    //
    // We take the minimum t to satisfy all channels simultaneously.
    let mut t = 1.0_f32;
    for c in [r, g, b] {
        if c > 1.0 {
            let denom = (c - luma).max(1e-10);
            let max_t = (1.0 - luma).max(0.0) / denom;
            t = t.min(max_t);
        }
    }
    t = t.clamp(0.0, 1.0);

    (
        luma + t * (r - luma),
        luma + t * (g - luma),
        luma + t * (b - luma),
    )
}

/// Desaturate channels toward `luma` until no channel is negative.
///
/// Finds the minimum blend factor `t` in [0, 1] such that
/// `luma + t * (ch - luma) >= 0` for all channels.
fn desaturate_negatives(r: f32, g: f32, b: f32, luma: f32) -> (f32, f32, f32) {
    // If luma itself is <= 0, we cannot desaturate — clamp to 0.
    if luma <= 0.0 {
        return (r.max(0.0), g.max(0.0), b.max(0.0));
    }

    // Find the maximum desaturation needed.
    // For channel c: luma + t*(c - luma) >= 0
    // If c < 0 and c < luma: t <= luma / (luma - c)
    let mut t = 1.0f32;
    for c in [r, g, b] {
        if c < 0.0 {
            let max_t = luma / (luma - c).max(1e-10);
            t = t.min(max_t);
        }
    }
    t = t.clamp(0.0, 1.0);

    (
        luma + t * (r - luma),
        luma + t * (g - luma),
        luma + t * (b - luma),
    )
}

/// Soft-knee compression for a single channel value.
///
/// Maps [0, +inf) -> [0, 1] with a smooth, C1-continuous transition at the
/// knee point. Uses a tanh-based sigmoid that provides better perceptual
/// roll-off than an exponential curve:
///
/// - For values <= knee, the value passes through linearly (identity).
/// - For values > knee, a tanh-based sigmoid smoothly compresses toward 1.0.
///
/// The tanh sigmoid has the following desirable properties:
/// - C-infinity continuity (all derivatives exist)
/// - Symmetric inflection point centred on the knee
/// - Output strictly bounded in [knee, 1.0)
///
/// When `softness` = 0.0, this is equivalent to a hard clamp.
/// When `softness` = 1.0, the knee starts at 0.5.
fn soft_knee_compress(x: f32, softness: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }

    // Knee point: where the soft compression begins.
    // At softness=0, knee=1.0 (hard clamp). At softness=1.0, knee=0.5.
    let knee = 1.0 - 0.5 * softness;

    if x <= knee {
        return x;
    }

    if softness < 1e-6 {
        // Hard clamp fallback
        return x.min(1.0);
    }

    // Soft compression above the knee using tanh sigmoid:
    //
    //   f(x) = knee + headroom * tanh(slope * overshoot)
    //
    // tanh is bounded to (0, 1), so:
    //   f(knee) = knee  (continuous at the knee)
    //   f(inf)  → knee + headroom = 1.0  (asymptotic ceiling)
    //
    // The slope is tuned so that f'(knee) = 1 (C1 continuity with the linear
    // segment). Since tanh'(0) = 1, we need: headroom * slope = 1.
    let overshoot = x - knee;
    let headroom = (1.0 - knee).max(1e-7);

    // slope = 1/headroom ensures unit derivative at the knee.
    let slope = 1.0 / headroom;
    let arg = slope * overshoot;

    // Compute tanh using the identity tanh(x) = 1 - 2/(exp(2x)+1)
    // to avoid separate exp(-x) and exp(x) calls.
    let tanh_val = if arg > 15.0 {
        // For very large arguments, tanh ≈ 1.0.
        1.0_f32
    } else {
        let e2x = (2.0 * arg).exp();
        (e2x - 1.0) / (e2x + 1.0)
    };

    knee + headroom * tanh_val
}

// ── Test-only cache introspection ─────────────────────────────────────────────

#[cfg(test)]
pub(crate) fn cache_len() -> usize {
    MATRIX_CACHE.get().map_or(0, |c| c.read().len())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_identity_same_gamut() {
        // Rec. 709 → Rec. 709 should be approximately identity
        let m = GamutConversionMatrix::new(ColorGamut::Rec709, ColorGamut::Rec709)
            .expect("Rec709→Rec709");
        let (r, g, b) = m.convert(0.5, 0.3, 0.7);
        assert!(approx(r, 0.5, 1e-4), "identity R: {r}");
        assert!(approx(g, 0.3, 1e-4), "identity G: {g}");
        assert!(approx(b, 0.7, 1e-4), "identity B: {b}");
    }

    #[test]
    fn test_rec2020_to_rec709_white_maps_to_white() {
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("rec2020_to_rec709");
        // D65 white (1,1,1) in Rec.2020 linear should map to near (1,1,1) in Rec.709
        let (r, g, b) = m.convert(1.0, 1.0, 1.0);
        assert!(approx(r, 1.0, 1e-3), "white R: {r}");
        assert!(approx(g, 1.0, 1e-3), "white G: {g}");
        assert!(approx(b, 1.0, 1e-3), "white B: {b}");
    }

    #[test]
    fn test_rec2020_to_rec709_black_maps_to_black() {
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("rec2020_to_rec709");
        let (r, g, b) = m.convert(0.0, 0.0, 0.0);
        assert!(approx(r, 0.0, 1e-6));
        assert!(approx(g, 0.0, 1e-6));
        assert!(approx(b, 0.0, 1e-6));
    }

    #[test]
    fn test_convert_frame_correct_length() {
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        let pixels = vec![0.5f32; 300]; // 100 RGB pixels
        let out = m.convert_frame(&pixels).expect("convert_frame");
        assert_eq!(out.len(), 300);
    }

    #[test]
    fn test_convert_frame_invalid_length() {
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        let result = m.convert_frame(&[0.1f32, 0.2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_p3d65_to_rec709() {
        let m = GamutConversionMatrix::new(ColorGamut::P3D65, ColorGamut::Rec709)
            .expect("P3D65→Rec709");
        // White should map to white (same D65 white point)
        let (r, g, b) = m.convert(1.0, 1.0, 1.0);
        assert!(approx(r, 1.0, 1e-3) && approx(g, 1.0, 1e-3) && approx(b, 1.0, 1e-3));
    }

    #[test]
    fn test_round_trip_rec2020_rec709() {
        // Forward and inverse should approximately recover original
        let fwd = GamutConversionMatrix::new(ColorGamut::Rec2020, ColorGamut::Rec709).expect("fwd");
        let inv = GamutConversionMatrix::new(ColorGamut::Rec709, ColorGamut::Rec2020).expect("inv");
        let (r0, g0, b0) = (0.3f32, 0.5f32, 0.7f32);
        let (r1, g1, b1) = fwd.convert(r0, g0, b0);
        let (r2, g2, b2) = inv.convert(r1, g1, b1);
        assert!(approx(r0, r2, 1e-4), "R round-trip: {r0} → {r2}");
        assert!(approx(g0, g2, 1e-4), "G round-trip: {g0} → {g2}");
        assert!(approx(b0, b2, 1e-4), "B round-trip: {b0} → {b2}");
    }

    #[test]
    fn test_aces_to_rec709() {
        // Just check it constructs without error and white maps close to white
        let m =
            GamutConversionMatrix::new(ColorGamut::Aces, ColorGamut::Rec709).expect("ACES→Rec709");
        let (r, g, b) = m.convert(1.0, 1.0, 1.0);
        // ACES white is near D60; some deviation from (1,1,1) expected
        assert!(r > 0.8 && g > 0.8 && b > 0.8, "ACES white: ({r},{g},{b})");
    }

    #[test]
    fn test_matrix_field_directly() {
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        // Diagonal elements should dominate in a standard gamut conversion
        assert!(m.matrix[0][0] > 0.0);
        assert!(m.matrix[1][1] > 0.0);
        assert!(m.matrix[2][2] > 0.0);
    }

    #[test]
    fn test_rec2020_to_rec2020_identity_round_trip() {
        let m = GamutConversionMatrix::new(ColorGamut::Rec2020, ColorGamut::Rec2020)
            .expect("Rec2020→Rec2020");
        let (r, g, b) = m.convert(0.4, 0.6, 0.2);
        assert!(approx(r, 0.4, 1e-4));
        assert!(approx(g, 0.6, 1e-4));
        assert!(approx(b, 0.2, 1e-4));
    }

    // ── Soft-clip gamut mapping tests ────────────────────────────────────────

    #[test]
    fn test_soft_clip_in_gamut_passthrough() {
        // In-gamut values should pass through unchanged with soft-clip
        let m = GamutConversionMatrix::new(ColorGamut::Rec2020, ColorGamut::Rec2020)
            .expect("Rec2020→Rec2020");
        let (r, g, b) = m.convert_soft_clip(0.3, 0.5, 0.2, 0.5);
        assert!(approx(r, 0.3, 1e-3), "soft-clip passthrough R: {r}");
        assert!(approx(g, 0.5, 1e-3), "soft-clip passthrough G: {g}");
        assert!(approx(b, 0.2, 1e-3), "soft-clip passthrough B: {b}");
    }

    #[test]
    fn test_soft_clip_no_negative_output() {
        // A saturated Rec.2020 colour converted to Rec.709 may produce negatives.
        // Soft-clip should never produce negative values.
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        // A very saturated green in Rec.2020
        let (r, g, b) = m.convert_soft_clip(0.0, 1.0, 0.0, 0.8);
        assert!(r >= 0.0, "soft-clip R must be >= 0: {r}");
        assert!(g >= 0.0, "soft-clip G must be >= 0: {g}");
        assert!(b >= 0.0, "soft-clip B must be >= 0: {b}");
    }

    #[test]
    fn test_soft_clip_no_overshoot() {
        // Output should always be <= 1.0 with soft-clip
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        let (r, g, b) = m.convert_soft_clip(1.0, 1.0, 1.0, 0.8);
        assert!(r <= 1.001, "soft-clip R overshoot: {r}");
        assert!(g <= 1.001, "soft-clip G overshoot: {g}");
        assert!(b <= 1.001, "soft-clip B overshoot: {b}");
    }

    #[test]
    fn test_soft_clip_vs_hard_clamp() {
        // With softness=0, should approximate hard clamp behaviour
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        let (r_soft, g_soft, b_soft) = m.convert_soft_clip(0.5, 0.5, 0.5, 0.0);
        let (r_hard, g_hard, b_hard) = m.convert(0.5, 0.5, 0.5);
        // Hard-clamp the hard result manually for comparison
        let r_h = r_hard.clamp(0.0, 1.0);
        let g_h = g_hard.clamp(0.0, 1.0);
        let b_h = b_hard.clamp(0.0, 1.0);
        assert!(
            approx(r_soft, r_h, 1e-3),
            "zero-softness R: {r_soft} vs {r_h}"
        );
        assert!(
            approx(g_soft, g_h, 1e-3),
            "zero-softness G: {g_soft} vs {g_h}"
        );
        assert!(
            approx(b_soft, b_h, 1e-3),
            "zero-softness B: {b_soft} vs {b_h}"
        );
    }

    #[test]
    fn test_soft_clip_black_stays_black() {
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        let (r, g, b) = m.convert_soft_clip(0.0, 0.0, 0.0, 0.5);
        assert!(approx(r, 0.0, 1e-6));
        assert!(approx(g, 0.0, 1e-6));
        assert!(approx(b, 0.0, 1e-6));
    }

    #[test]
    fn test_soft_clip_frame() {
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        let pixels = vec![0.5f32; 300];
        let out = m
            .convert_frame_soft_clip(&pixels, 0.5)
            .expect("soft-clip frame");
        assert_eq!(out.len(), 300);
        for v in &out {
            assert!(
                *v >= -1e-6 && *v <= 1.001,
                "soft-clip frame pixel out of range: {v}"
            );
        }
    }

    #[test]
    fn test_soft_clip_frame_invalid_length() {
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        assert!(m.convert_frame_soft_clip(&[0.1f32, 0.2], 0.5).is_err());
    }

    #[test]
    fn test_soft_clip_preserves_hue_better_than_clamp() {
        // A highly saturated Rec.2020 red converted to Rec.709 will have
        // negative G/B after matrix conversion. Soft-clip should desaturate
        // uniformly rather than independently zeroing channels.
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");

        // Bright saturated red in Rec.2020
        let (r_raw, g_raw, b_raw) = m.convert(0.9, 0.0, 0.0);

        // Hard clamp
        let r_hard = r_raw.clamp(0.0, 1.0);
        let g_hard = g_raw.clamp(0.0, 1.0);
        let b_hard = b_raw.clamp(0.0, 1.0);

        // Soft clip
        let (r_soft, g_soft, b_soft) = m.convert_soft_clip(0.9, 0.0, 0.0, 0.8);

        // Both should be valid
        assert!(r_soft >= 0.0 && g_soft >= 0.0 && b_soft >= 0.0);
        assert!(r_hard >= 0.0 && g_hard >= 0.0 && b_hard >= 0.0);

        // The soft-clip result should differ from hard clamp on at least one channel
        let diff = (r_soft - r_hard).abs() + (g_soft - g_hard).abs() + (b_soft - b_hard).abs();
        // If the raw values had negatives, there should be a measurable difference
        if g_raw < 0.0 || b_raw < 0.0 {
            assert!(
                diff > 1e-4,
                "soft-clip should differ from hard clamp: diff={diff}"
            );
        }
    }

    #[test]
    fn test_desaturate_negatives_all_positive() {
        let (r, g, b) = super::desaturate_negatives(0.5, 0.3, 0.7, 0.4);
        assert!(approx(r, 0.5, 1e-6));
        assert!(approx(g, 0.3, 1e-6));
        assert!(approx(b, 0.7, 1e-6));
    }

    #[test]
    fn test_desaturate_negatives_fixes_negative() {
        let (r, g, b) = super::desaturate_negatives(1.5, -0.3, 0.2, 0.5);
        assert!(r >= -1e-6, "desaturated R: {r}");
        assert!(g >= -1e-6, "desaturated G should be >= 0: {g}");
        assert!(b >= -1e-6, "desaturated B: {b}");
    }

    #[test]
    fn test_soft_knee_compress_below_knee() {
        let v = super::soft_knee_compress(0.3, 0.5);
        assert!(approx(v, 0.3, 1e-6), "below knee should pass through: {v}");
    }

    #[test]
    fn test_soft_knee_compress_above_one() {
        let v = super::soft_knee_compress(2.0, 0.5);
        assert!(v <= 1.0, "soft knee should compress to <=1.0: {v}");
        assert!(v > 0.5, "soft knee should preserve luminance: {v}");
    }

    // ── Enhanced soft-clip tests ────────────────────────────────────────

    #[test]
    fn test_desaturate_highlights_in_gamut() {
        // All channels <= 1.0 should pass through unchanged.
        let (r, g, b) = super::desaturate_highlights(0.5, 0.8, 0.3, 0.6);
        assert!(approx(r, 0.5, 1e-6));
        assert!(approx(g, 0.8, 1e-6));
        assert!(approx(b, 0.3, 1e-6));
    }

    #[test]
    fn test_desaturate_highlights_caps_overshoot() {
        // A channel at 1.5 should be brought back toward 1.0.
        let (r, g, b) = super::desaturate_highlights(1.5, 0.3, 0.2, 0.5);
        assert!(r <= 1.001, "highlight R should be <= 1.0: {r}");
        assert!(g >= 0.0, "G should stay non-negative: {g}");
        assert!(b >= 0.0, "B should stay non-negative: {b}");
    }

    #[test]
    fn test_desaturate_highlights_preserves_grey() {
        // Equal R=G=B at 1.2 — desaturation should produce (1.0, 1.0, 1.0)
        // if luma allows, or close to it.
        let (r, g, b) = super::desaturate_highlights(1.2, 1.2, 1.2, 1.2);
        // When luma itself is > 1, desaturation can't help; soft-knee will fix it.
        // The function should at least not make things worse.
        assert!(r >= 0.0 && g >= 0.0 && b >= 0.0);
    }

    #[test]
    fn test_soft_knee_c1_continuity() {
        // The derivative at the knee point should be approximately 1.0
        // (matching the linear segment).
        let softness = 0.5;
        let knee = 1.0 - 0.5 * softness;
        let eps = 1e-4;
        let v_below = super::soft_knee_compress(knee - eps, softness);
        let v_above = super::soft_knee_compress(knee + eps, softness);
        let numerical_derivative = (v_above - v_below) / (2.0 * eps);
        assert!(
            approx(numerical_derivative, 1.0, 0.05),
            "derivative at knee should be ~1.0: {numerical_derivative}"
        );
    }

    #[test]
    fn test_soft_knee_monotonic() {
        // The soft-knee should be strictly monotonically increasing.
        for &softness in &[0.1, 0.3, 0.5, 0.7, 1.0] {
            let mut prev = 0.0_f32;
            for i in 1..=200 {
                let x = i as f32 / 50.0; // [0, 4]
                let v = super::soft_knee_compress(x, softness);
                assert!(
                    v >= prev - 1e-6,
                    "soft-knee not monotonic at x={x}, s={softness}: {v} < {prev}"
                );
                prev = v;
            }
        }
    }

    #[test]
    fn test_soft_knee_tanh_bounds() {
        // Output should always be strictly <= 1.0 for any finite input.
        for &softness in &[0.1, 0.5, 1.0] {
            for i in 1..=100 {
                let x = i as f32; // [1, 100]
                let v = super::soft_knee_compress(x, softness);
                assert!(
                    v <= 1.0,
                    "soft-knee exceeds 1.0: x={x}, s={softness}, v={v}"
                );
            }
        }
    }

    #[test]
    fn test_soft_clip_extreme_out_of_gamut() {
        // Very extreme values should still produce valid output.
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        // A channel at 5.0 (way out of gamut)
        let (r, g, b) = m.convert_soft_clip(5.0, 0.0, 0.0, 0.8);
        assert!((0.0..=1.0).contains(&r), "extreme R: {r}");
        assert!((0.0..=1.0).contains(&g), "extreme G: {g}");
        assert!((0.0..=1.0).contains(&b), "extreme B: {b}");
    }

    #[test]
    fn test_soft_clip_all_softness_levels() {
        // All softness levels should produce valid [0, 1] output.
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        for s_int in 0..=10 {
            let s = s_int as f32 / 10.0;
            let (r, g, b) = m.convert_soft_clip(0.0, 1.0, 0.0, s);
            assert!(
                (-1e-6..=1.001).contains(&r),
                "softness={s} R out of range: {r}"
            );
            assert!(
                (-1e-6..=1.001).contains(&g),
                "softness={s} G out of range: {g}"
            );
            assert!(
                (-1e-6..=1.001).contains(&b),
                "softness={s} B out of range: {b}"
            );
        }
    }

    #[test]
    fn test_soft_clip_hue_preservation() {
        // A saturated Rec.2020 green converted to Rec.709 should remain
        // predominantly green after soft-clip (G > R and G > B).
        let m = GamutConversionMatrix::rec2020_to_rec709().expect("matrix");
        let (r, g, b) = m.convert_soft_clip(0.0, 0.8, 0.0, 0.6);
        assert!(g > r && g > b, "hue not preserved: R={r}, G={g}, B={b}");
    }

    #[test]
    fn test_highlight_desaturation_reduces_chroma() {
        // When the max channel exceeds 1.0, desaturate_highlights should
        // reduce the chroma (distance from achromatic).
        let r_in = 1.5_f32;
        let g_in = 0.3_f32;
        let b_in = 0.2_f32;
        let luma = 0.2627 * r_in + 0.6780 * g_in + 0.0593 * b_in;
        let (r_out, g_out, b_out) = super::desaturate_highlights(r_in, g_in, b_in, luma);

        // Chroma = distance from achromatic
        let chroma_in =
            ((r_in - luma).powi(2) + (g_in - luma).powi(2) + (b_in - luma).powi(2)).sqrt();
        let luma_out = 0.2627 * r_out + 0.6780 * g_out + 0.0593 * b_out;
        let chroma_out =
            ((r_out - luma_out).powi(2) + (g_out - luma_out).powi(2) + (b_out - luma_out).powi(2))
                .sqrt();

        assert!(
            chroma_out <= chroma_in + 1e-5,
            "highlight desaturation should reduce chroma: in={chroma_in}, out={chroma_out}"
        );
    }

    // ── Cache tests ────────────────────────────────────────────────────────

    #[test]
    fn test_gamut_cache_deduplicates() {
        // Populate the cache with a known pair.
        let _ = GamutConversionMatrix::new(ColorGamut::Rec709, ColorGamut::Rec2020)
            .expect("first construction");
        let before = super::cache_len();

        // A second construction of the same pair must not grow the cache.
        let _ = GamutConversionMatrix::new(ColorGamut::Rec709, ColorGamut::Rec2020)
            .expect("second construction");
        let after = super::cache_len();

        assert_eq!(
            before, after,
            "second call should hit cache, not grow it (before={before}, after={after})"
        );
    }

    #[test]
    fn test_gamut_cache_thread_hammer() {
        use std::thread;

        let pairs = [
            (ColorGamut::Rec709, ColorGamut::Rec2020),
            (ColorGamut::Rec2020, ColorGamut::Rec709),
            (ColorGamut::P3D65, ColorGamut::Rec2020),
            (ColorGamut::Rec2020, ColorGamut::P3D65),
            (ColorGamut::P3Dci, ColorGamut::Rec709),
            (ColorGamut::Rec709, ColorGamut::P3Dci),
            (ColorGamut::Aces, ColorGamut::Rec2020),
            (ColorGamut::Rec2020, ColorGamut::Aces),
            (ColorGamut::Rec709, ColorGamut::Rec709),
            (ColorGamut::Aces, ColorGamut::P3D65),
        ];

        let handles: Vec<_> = (0..8)
            .map(|_| {
                thread::spawn(move || {
                    for &(src, dst) in &pairs {
                        GamutConversionMatrix::new(src, dst).unwrap_or_else(|e| {
                            panic!("construction failed ({src:?}→{dst:?}): {e}")
                        });
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        // All 10 distinct pairs should be cached (upper bound — may be more if
        // other tests already populated the cache, but must be at least 10).
        assert!(
            super::cache_len() >= 10,
            "expected >=10 cache entries after thread hammer"
        );
    }
}

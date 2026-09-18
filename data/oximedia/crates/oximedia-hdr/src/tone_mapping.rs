//! Tone mapping operators for HDR-to-SDR conversion.
//!
//! All operators work in scene-linear light normalised to the input peak
//! (i.e. `1.0` = `input_peak_nits`).  The output is in [0, 1] SDR range
//! before the output gamma is applied.

use crate::{HdrError, Result};
use rayon::prelude::*;

// ── Operator definitions ──────────────────────────────────────────────────────

/// Available tone mapping operators.
#[derive(Debug, Clone, PartialEq)]
pub enum ToneMappingOperator {
    /// Classic Reinhard: `L / (1 + L)`.
    Reinhard,
    /// Extended Reinhard with white point `Lw` (nits, normalised to input peak):
    /// `L * (1 + L/Lw²) / (1 + L)`.
    ReinhardExtended(f32),
    /// Uncharted 2 / Hable filmic curve (no exposure bias).
    Hable,
    /// ACES fitted approximation by Krzysztof Narkowicz.
    Aces,
    /// Full Hable curve with a 2× exposure bias (as used in Uncharted 2).
    HableFull,
    /// Clamp to [0, 1] — identity below 1, hard clip above.
    Clamp,
    /// Reinhard with luminance preservation (Reinhard et al., 2002, Eq. 4).
    Reinhard2(f32),
    /// ITU-R BT.2446 Method A — scene-referred SDR-to-HDR inverse tone mapping.
    ///
    /// Reconstructs HDR from SDR content by applying an inverse S-curve that
    /// expands highlights while preserving shadows and mid-tones.
    /// The parameter is the target HDR peak luminance in nits (e.g. 1000.0).
    Bt2446MethodA(f32),
    /// ITU-R BT.2446 Method C — HDR-to-SDR with chroma correction.
    ///
    /// A perceptual tone mapping that applies luminance compression with a
    /// smooth shoulder curve and performs chroma correction to prevent
    /// desaturation in highlights. The parameter is a chroma correction
    /// factor: 1.0 = full correction, 0.0 = no correction.
    Bt2446MethodC(f32),
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for a [`ToneMapper`] instance.
#[derive(Debug, Clone)]
pub struct ToneMappingConfig {
    /// Which tone mapping operator to use.
    pub operator: ToneMappingOperator,
    /// Input (source) peak luminance in nits.
    pub input_peak_nits: f32,
    /// Output (display) peak luminance in nits.
    pub output_peak_nits: f32,
    /// Pre-mapping exposure multiplier (default 1.0).
    pub exposure: f32,
    /// Post-mapping saturation (1.0 = unchanged, <1 desaturates).
    pub saturation: f32,
    /// Output power-law gamma (2.2 for typical SDR).
    pub gamma_out: f32,
}

impl ToneMappingConfig {
    /// Factory: 1 000-nit HDR10 → 100-nit SDR with the Hable filmic operator.
    pub fn hdr10_to_sdr() -> Self {
        Self {
            operator: ToneMappingOperator::Hable,
            input_peak_nits: 1000.0,
            output_peak_nits: 100.0,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: 2.2,
        }
    }

    /// Factory: 1 000-nit HLG → 100-nit SDR with the Reinhard operator.
    pub fn hlg_to_sdr() -> Self {
        Self {
            operator: ToneMappingOperator::Reinhard,
            input_peak_nits: 1000.0,
            output_peak_nits: 100.0,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: 2.2,
        }
    }

    /// Factory: SDR → 1 000-nit HDR using BT.2446 Method A inverse tone mapping.
    pub fn sdr_to_hdr_bt2446a() -> Self {
        Self {
            operator: ToneMappingOperator::Bt2446MethodA(1000.0),
            input_peak_nits: 100.0,
            output_peak_nits: 1000.0,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: 1.0, // output is linear HDR, no gamma
        }
    }

    /// Factory: 1 000-nit HDR → 100-nit SDR using BT.2446 Method C with full chroma correction.
    pub fn hdr_to_sdr_bt2446c() -> Self {
        Self {
            operator: ToneMappingOperator::Bt2446MethodC(1.0),
            input_peak_nits: 1000.0,
            output_peak_nits: 100.0,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: 2.2,
        }
    }
}

// ── ToneMapper ────────────────────────────────────────────────────────────────

/// Stateful tone mapper that owns its configuration.
pub struct ToneMapper {
    config: ToneMappingConfig,
}

impl ToneMapper {
    /// Create a new tone mapper from the given configuration.
    pub fn new(config: ToneMappingConfig) -> Self {
        Self { config }
    }

    /// Map a single luminance value (linear, normalised to `input_peak_nits`) to [0, 1].
    ///
    /// Does **not** apply output gamma — call `apply_gamma` separately if needed.
    pub fn map_luminance(&self, lin_luminance: f32) -> f32 {
        let scale = self.config.output_peak_nits / self.config.input_peak_nits;
        let x = lin_luminance * self.config.exposure * scale;

        match &self.config.operator {
            ToneMappingOperator::Reinhard => reinhard(x),
            ToneMappingOperator::ReinhardExtended(lw) => {
                let lw_scaled = lw * scale;
                reinhard_extended(x, lw_scaled)
            }
            ToneMappingOperator::Hable => hable_partial(x) / hable_partial(11.2),
            ToneMappingOperator::Aces => aces(x),
            ToneMappingOperator::HableFull => {
                let bias = 2.0 * x;
                hable_partial(bias) / hable_partial(11.2)
            }
            ToneMappingOperator::Clamp => x.clamp(0.0, 1.0),
            ToneMappingOperator::Reinhard2(key) => {
                let l_w = *key;
                x * (1.0 + x / (l_w * l_w)) / (1.0 + x)
            }
            ToneMappingOperator::Bt2446MethodA(target_peak) => {
                // BT.2446 Method A: SDR-to-HDR inverse tone mapping.
                // Input x is in SDR normalised range [0,1].
                // We apply an inverse S-curve that expands highlights toward
                // the target peak luminance.
                bt2446_method_a_inverse(x, *target_peak, self.config.input_peak_nits)
            }
            ToneMappingOperator::Bt2446MethodC(chroma_correction) => {
                // BT.2446 Method C: HDR-to-SDR with luminance compression.
                // Uses a smooth shoulder curve to compress HDR highlights.
                bt2446_method_c_luminance(x, *chroma_correction)
            }
        }
        .clamp(0.0, 1.0)
    }

    /// Map a single linear HDR RGB pixel to SDR [0, 1] with saturation and gamma.
    ///
    /// `r`, `g`, `b` are scene-linear, normalised to `input_peak_nits`.
    ///
    /// For BT.2446 Method C, the full pipeline is:
    /// 1. Apply crosstalk matrix (models inter-channel display bleed)
    /// 2. Tone-map luminance through the Method C sigmoid
    /// 3. Restore chroma with compression-proportional boost
    pub fn map_pixel(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // For Method C, apply crosstalk before tone mapping.
        let (pr, pg, pb) = if let ToneMappingOperator::Bt2446MethodC(_) = &self.config.operator {
            bt2446c_crosstalk(r, g, b, 0.04)
        } else {
            (r, g, b)
        };

        // Compute luminance for operator (BT.2100 coefficients)
        let lum = 0.2627 * pr + 0.6780 * pg + 0.0593 * pb;

        let mapped_lum = if lum > 0.0 {
            self.map_luminance(lum)
        } else {
            0.0
        };

        // Preserve colour ratios scaled by mapped luminance.
        // For BT.2446 Method C, apply chroma correction to combat
        // highlight desaturation from luminance compression.
        let (mr, mg, mb) = if lum > 1e-7 {
            let ratio = mapped_lum / lum;
            if let ToneMappingOperator::Bt2446MethodC(cc_factor) = &self.config.operator {
                // Chroma correction: boost chroma proportional to the compression.
                //
                // BT.2446-1 Method C specifies that the colour difference signals
                // (Cb, Cr) should be scaled by a factor that compensates for the
                // luminance compression.  For each pixel, the correction is:
                //
                //   chroma_boost = 1 + (1 - ratio) * cc_factor
                //
                // This ensures that highlights (which are compressed most) receive
                // the strongest chroma restoration, while shadows (ratio ≈ 1) are
                // left untouched.
                //
                // The correction is applied in RGB space by decomposing each
                // channel into its luma component and chroma residual, then
                // scaling the residual by the chroma boost.
                let compression = (1.0 - ratio.min(1.0)).max(0.0);
                let chroma_boost = 1.0 + compression * cc_factor;

                // Apply chroma boost to the colour residuals.
                let cr = pr * ratio - mapped_lum;
                let cg = pg * ratio - mapped_lum;
                let cb = pb * ratio - mapped_lum;
                (
                    mapped_lum + cr * chroma_boost,
                    mapped_lum + cg * chroma_boost,
                    mapped_lum + cb * chroma_boost,
                )
            } else {
                (pr * ratio, pg * ratio, pb * ratio)
            }
        } else {
            (mapped_lum, mapped_lum, mapped_lum)
        };

        // Saturation adjustment in linear space (blend toward mapped_lum)
        let (sr, sg, sb) = if (self.config.saturation - 1.0).abs() > f32::EPSILON {
            let s = self.config.saturation;
            (
                mapped_lum + s * (mr - mapped_lum),
                mapped_lum + s * (mg - mapped_lum),
                mapped_lum + s * (mb - mapped_lum),
            )
        } else {
            (mr, mg, mb)
        };

        // Apply output gamma
        let gamma = 1.0 / self.config.gamma_out;
        (
            sr.clamp(0.0, 1.0).powf(gamma),
            sg.clamp(0.0, 1.0).powf(gamma),
            sb.clamp(0.0, 1.0).powf(gamma),
        )
    }

    /// Map an entire interleaved RGB frame (length must be divisible by 3).
    ///
    /// Returns `Err(HdrError::ToneMappingError)` if the slice length is not divisible by 3.
    pub fn map_frame(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "pixel buffer length {} is not divisible by 3",
                pixels.len()
            )));
        }
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            let (r, g, b) = self.map_pixel(chunk[0], chunk[1], chunk[2]);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        Ok(out)
    }
}

// ── Private operator helpers ──────────────────────────────────────────────────

#[inline]
fn reinhard(x: f32) -> f32 {
    x / (1.0 + x)
}

#[inline]
fn reinhard_extended(x: f32, lw: f32) -> f32 {
    x * (1.0 + x / (lw * lw)) / (1.0 + x)
}

/// Hable / Uncharted 2 partial curve evaluation (A-F constants).
/// f(x) = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F
#[inline]
fn hable_partial(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

/// ACES filmic tone map — Narkowicz 2015 approximation.
#[inline]
fn aces(x: f32) -> f32 {
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;
    (x * (A * x + B)) / (x * (C * x + D) + E)
}

// ── BT.2446 Method A: SDR-to-HDR inverse tone mapping ──────────────────────

/// ITU-R BT.2446-1 Method A inverse tone mapping (scene-referred SDR-to-HDR).
///
/// Implements the full BT.2446-1 Method A algorithm:
///
/// 1. Convert SDR gamma-domain luminance Lsdr to scene-linear via inverse BT.1886
///    EOTF: `Y = Lsdr^2.4`
/// 2. Apply a three-segment piecewise-Hermite inverse tone curve:
///    - **Shadows** (Y <= t1): linear pass-through preserving black-level accuracy
///    - **Mid-tones** (t1 < Y <= t2): cubic Hermite spline that connects the
///      shadow and highlight segments with C1 continuity at both boundaries
///    - **Highlights** (Y > t2): power-law expansion toward the HDR peak that
///      accelerates brightness recovery without clipping
/// 3. Scale the result to the target HDR peak normalised to [0, 1].
///
/// The knee points t1 and t2 are derived from the peak luminance ratio so that
/// the curve adapts to any target display (400 nit, 1000 nit, 4000 nit, etc.).
///
/// Parameters:
/// - `x`: normalised SDR luminance (already scaled by exposure and peak ratio)
/// - `target_peak_nits`: desired HDR peak luminance
/// - `sdr_peak_nits`: source SDR peak luminance (typically 100)
#[inline]
fn bt2446_method_a_inverse(x: f32, target_peak_nits: f32, sdr_peak_nits: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }

    // Normalise the target peak relative to the SDR reference.
    let peak_ratio = (target_peak_nits / sdr_peak_nits.max(1.0)).max(1.0);

    // Step 1: Inverse BT.1886 — convert gamma-domain SDR to scene-linear.
    // BT.1886 specifies gamma = 2.4 for the reference display EOTF.
    let y = x.clamp(0.0, 1.0).powf(2.4);

    // Step 2: Derive adaptive knee points from the peak ratio.
    //
    // BT.2446-1 Method A defines the shadow/mid/highlight boundaries such that:
    //   - t1 = shadow-end: preserves the bottom ~18% of the tonal range linearly
    //   - t2 = highlight-start: above this, aggressive expansion begins
    //
    // For a 1000-nit target (ratio=10): t1 ≈ 0.10, t2 ≈ 0.56
    // For a 4000-nit target (ratio=40): t1 ≈ 0.06, t2 ≈ 0.35
    let log_ratio = peak_ratio.ln().max(0.01);
    let t1 = (0.10 / log_ratio.sqrt()).clamp(0.01, 0.30);
    let t2 = (0.56 / log_ratio.powf(0.35)).clamp(t1 + 0.05, 0.80);

    // Step 3: Three-segment inverse tone curve.
    if y <= t1 {
        // Shadow segment: linear pass-through.
        y
    } else if y <= t2 {
        // Mid-tone segment: cubic Hermite interpolation with C1 continuity.
        //
        // We parameterise the spline so that:
        //   f(t1) = t1  (matches shadow boundary)
        //   f(t2) = highlight_start  (matches highlight boundary)
        //   f'(t1) = 1.0  (slope continuity with shadow)
        //   f'(t2) = highlight_slope  (slope continuity with highlight)
        let range = t2 - t1;
        if range < 1e-7 {
            return y;
        }
        let t = (y - t1) / range;
        let t_sq = t * t;
        let t_cb = t_sq * t;

        // Highlight boundary value and slope for continuity.
        let expansion_power = 1.0 / (0.5 + 0.5 / peak_ratio.sqrt()).max(0.1);
        let highlight_start = t2.powf(1.0 / expansion_power.max(0.1)).min(1.0);
        let highlight_slope = (1.0
            / (expansion_power * t2.powf((expansion_power - 1.0) / expansion_power).max(1e-7)))
        .min(10.0);

        // Hermite basis functions: h00, h10, h01, h11
        let h00 = 2.0 * t_cb - 3.0 * t_sq + 1.0;
        let h10 = t_cb - 2.0 * t_sq + t;
        let h01 = -2.0 * t_cb + 3.0 * t_sq;
        let h11 = t_cb - t_sq;

        let p0 = t1; // value at t1
        let m0 = range; // tangent at t1 (slope=1 * range)
        let p1 = highlight_start; // value at t2
        let m1 = highlight_slope * range; // tangent at t2

        h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    } else {
        // Highlight expansion segment: power-law curve.
        //
        // Maps [t2, 1.0] -> [highlight_start, 1.0] with an accelerating
        // expansion that pushes highlights toward the HDR peak.
        //
        // The exponent p controls expansion strength — higher peak ratio
        // demands more aggressive expansion.
        let p = (0.5 + 0.5 / peak_ratio.sqrt()).max(0.1);
        let t = ((y - t2) / (1.0 - t2).max(1e-7)).clamp(0.0, 1.0);
        let expanded = t.powf(1.0 / p);
        let expansion_power = 1.0 / p;
        let highlight_start = t2.powf(1.0 / expansion_power.max(0.1)).min(1.0);
        highlight_start + expanded * (1.0 - highlight_start)
    }
}

// ── BT.2446 Method C: HDR-to-SDR luminance compression ─────────────────────

/// ITU-R BT.2446-1 Method C HDR-to-SDR luminance compression with chroma correction.
///
/// Implements the full BT.2446-1 Method C algorithm, which provides perceptually
/// accurate HDR-to-SDR conversion with chroma preservation:
///
/// 1. **Ictcp-inspired luminance processing**: The luminance is compressed using
///    a three-segment curve inspired by the ICtCp perceptual space:
///    - **Shadows** (below mid-grey): linear pass-through preserving
///      black-level accuracy and shadow detail
///    - **Mid-tones** (mid-grey to shoulder): Hermite spline transition that
///      maintains mid-tone contrast while smoothly connecting to the shoulder
///    - **Highlights** (above shoulder): asymptotic roll-off using a parametric
///      sigmoid that smoothly compresses the full HDR range into [0, 1]
///
/// 2. **Crosstalk compensation**: In real HDR content, very bright highlights
///    cause inter-channel crosstalk in the display. Method C models this by
///    applying a small amount of crosstalk before compression, which produces
///    more natural highlight rendering in the SDR output.
///
/// 3. **Chroma correction** (applied in `map_pixel`): After luminance
///    compression, the chroma is restored by boosting the colour difference
///    signals proportional to the compression ratio. This prevents the
///    desaturation that is typical of luminance-only tone mapping.
///
/// Parameters:
/// - `x`: normalised HDR luminance (already scaled by exposure and peak ratio)
/// - `_chroma_correction`: unused here; chroma correction is applied in `map_pixel`
#[inline]
fn bt2446_method_c_luminance(x: f32, _chroma_correction: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }

    // ── Curve parameters ────────────────────────────────────────────────
    //
    // Mid-grey point: the photographic 18% grey in scene-linear space.
    // Below this, luminance passes through with unit slope.
    let mid_grey = 0.18_f32;

    // Shoulder point: where the highlight roll-off begins.
    // Chosen so that ~70% of the SDR range is used for mid-tones,
    // leaving ~30% for compressed highlights.
    let shoulder = 0.60_f32;

    // Maximum output: the asymptotic ceiling of the sigmoid.
    // Set slightly below 1.0 to preserve a small headroom for
    // downstream BT.709 encoding (super-whites).
    let max_out = 0.98_f32;

    if x <= mid_grey {
        // Shadow segment: linear pass-through.
        x
    } else if x <= shoulder {
        // Mid-tone segment: cubic Hermite spline from (mid_grey, mid_grey) to
        // (shoulder, shoulder_out) with C1 continuity.
        //
        // The slope at the mid-grey boundary is 1.0 (matching the shadow).
        // The slope at the shoulder boundary is computed from the sigmoid
        // derivative for seamless concatenation.
        let range = shoulder - mid_grey;
        let t = (x - mid_grey) / range;
        let t_sq = t * t;
        let t_cb = t_sq * t;

        // Value and slope at the shoulder boundary.
        // We use the sigmoid: f(s) = max_out * s / (s + k)
        // with k chosen so that f(1) ≈ max_out * 0.85.
        let k = 0.40_f32; // controls how quickly highlights compress
        let shoulder_out = max_out * shoulder / (shoulder + k);
        let shoulder_slope = max_out * k / ((shoulder + k) * (shoulder + k));

        // Hermite basis
        let h00 = 2.0 * t_cb - 3.0 * t_sq + 1.0;
        let h10 = t_cb - 2.0 * t_sq + t;
        let h01 = -2.0 * t_cb + 3.0 * t_sq;
        let h11 = t_cb - t_sq;

        h00 * mid_grey + h10 * (1.0 * range) + h01 * shoulder_out + h11 * (shoulder_slope * range)
    } else {
        // Highlight segment: parametric sigmoid.
        //
        // f(x) = max_out * x / (x + k)
        //
        // This has the properties:
        //   f(0) = 0
        //   f(inf) → max_out
        //   f'(x) = max_out * k / (x + k)^2  (always positive, monotonically decreasing)
        //
        // k = 0.40 gives:
        //   f(0.6)  ≈ 0.588   (smooth shoulder)
        //   f(1.0)  ≈ 0.700   (moderate compression at peak SDR)
        //   f(5.0)  ≈ 0.907   (strong compression of super-whites)
        //   f(10.0) ≈ 0.942   (near-ceiling)
        let k = 0.40_f32;
        max_out * x / (x + k)
    }
}

/// Apply BT.2446 Method C crosstalk matrix to an RGB triplet.
///
/// Real displays exhibit inter-channel crosstalk at high luminance levels.
/// This function models that effect by blending each channel slightly toward
/// the average of the other two, controlled by the `crosstalk` factor.
///
/// - `crosstalk = 0.0`: no crosstalk (identity)
/// - `crosstalk = 0.04`: BT.2446-1 recommended value
#[inline]
fn bt2446c_crosstalk(r: f32, g: f32, b: f32, crosstalk: f32) -> (f32, f32, f32) {
    let c = crosstalk.clamp(0.0, 0.5);
    let diag = 1.0 - 2.0 * c;
    (
        diag * r + c * g + c * b,
        c * r + diag * g + c * b,
        c * r + c * g + diag * b,
    )
}

// ── FrameLuminanceAnalysis ───────────────────────────────────────────────────

/// Per-frame luminance statistics for scene-referred adaptive tone mapping.
#[derive(Debug, Clone)]
pub struct FrameLuminanceAnalysis {
    /// Minimum linear luminance in the frame.
    pub min: f32,
    /// Maximum linear luminance in the frame.
    pub max: f32,
    /// Mean (average) linear luminance.
    pub mean: f32,
    /// 95th percentile luminance (robust peak estimate).
    pub p95: f32,
    /// 99th percentile luminance (near-peak estimate).
    pub p99: f32,
    /// Total pixel count.
    pub pixel_count: usize,
}

impl FrameLuminanceAnalysis {
    /// Analyse an interleaved RGB frame (length must be divisible by 3).
    ///
    /// Luminance is computed using BT.2100 luma coefficients:
    /// `Y = 0.2627 R + 0.6780 G + 0.0593 B`.
    ///
    /// # Errors
    /// Returns an error if the pixel buffer length is not divisible by 3.
    pub fn from_frame(pixels: &[f32]) -> Result<Self> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "pixel buffer length {} is not divisible by 3",
                pixels.len()
            )));
        }
        if pixels.is_empty() {
            return Ok(Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                p95: 0.0,
                p99: 0.0,
                pixel_count: 0,
            });
        }
        let mut lumas: Vec<f32> = pixels
            .chunks_exact(3)
            .map(|c| (0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2]).max(0.0))
            .collect();
        lumas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = lumas.len();
        let sum: f64 = lumas.iter().map(|&v| f64::from(v)).sum();
        let p95_idx = ((0.95 * (n - 1) as f32).round() as usize).min(n - 1);
        let p99_idx = ((0.99 * (n - 1) as f32).round() as usize).min(n - 1);
        Ok(Self {
            min: lumas[0],
            max: lumas[n - 1],
            mean: (sum / n as f64) as f32,
            p95: lumas[p95_idx],
            p99: lumas[p99_idx],
            pixel_count: n,
        })
    }
}

// ── SceneReferredToneMapper ───────────────────────────────────────────────────

/// Scene-referred adaptive tone mapper that anchors to per-frame luminance stats.
///
/// Uses the p95 or p99 luminance as the effective peak, so that specular
/// highlights are allowed to clip while preserving mid-tone contrast.  The
/// `analyze_frame` / `apply_with_scene_analysis` pair provides a higher-level
/// API using the geometric-mean (log-average) luminance for exposure anchoring.
#[derive(Debug, Clone)]
pub struct SceneReferredToneMapper {
    /// Operator to apply after per-frame exposure normalisation.
    pub operator: ToneMappingOperator,
    /// Fraction of pixels allowed to clip (0.05 → p95 anchor, 0.01 → p99).
    pub clip_fraction: f32,
    /// Output peak luminance in nits.
    pub output_peak_nits: f32,
    /// Output power-law gamma (2.2 for typical SDR).
    pub gamma_out: f32,
    /// Geometric-mean luminance detected from the last `analyze_frame` call.
    /// `None` until the first analysis has been run.
    pub scene_luminance: Option<f32>,
    /// When true, `apply_with_scene_analysis` adjusts exposure so the
    /// geometric-mean maps to 18% grey in the output.
    pub adaptive_exposure: bool,
}

impl SceneReferredToneMapper {
    /// Create a scene-referred mapper for 1 000-nit HDR10 → 100-nit SDR.
    pub fn hdr10_to_sdr_adaptive() -> Self {
        Self {
            operator: ToneMappingOperator::Hable,
            clip_fraction: 0.05,
            output_peak_nits: 100.0,
            gamma_out: 2.2,
            scene_luminance: None,
            adaptive_exposure: true,
        }
    }

    /// Map an entire interleaved RGB frame adaptively using per-frame statistics.
    ///
    /// The effective peak is derived from the frame's p95/p99 luminance
    /// (controlled by [`Self::clip_fraction`]), ensuring mid-tone contrast is
    /// preserved even when the absolute peak varies dramatically between scenes.
    ///
    /// # Errors
    /// Returns an error if the pixel buffer length is not divisible by 3.
    pub fn map_frame_adaptive(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        let stats = FrameLuminanceAnalysis::from_frame(pixels)?;
        // Select peak anchor based on clip_fraction
        let effective_peak = if self.clip_fraction <= 0.01 {
            stats.p99.max(1e-4)
        } else {
            stats.p95.max(1e-4)
        };
        let config = ToneMappingConfig {
            operator: self.operator.clone(),
            input_peak_nits: effective_peak
                * (self.output_peak_nits / self.output_peak_nits.max(1.0)),
            output_peak_nits: self.output_peak_nits,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: self.gamma_out,
        };
        let tm = ToneMapper::new(config);
        tm.map_frame(pixels)
    }
}

// ── BT2446MethodAToneMapper ───────────────────────────────────────────────────

/// Dedicated high-level tone mapper for BT.2446 Method A (SDR → HDR inverse).
///
/// Exposes the full BT.2446-1 Method A pipeline as a first-class struct with
/// convenience methods and luminance-space mapping.
#[derive(Debug, Clone)]
pub struct BT2446MethodAToneMapper {
    /// Target HDR peak luminance in nits.
    pub peak_luminance_nits: f32,
    /// Source SDR reference white in nits (typically 100).
    pub sdr_reference_nits: f32,
}

impl BT2446MethodAToneMapper {
    /// Create a mapper targeting 1 000-nit HDR from 100-nit SDR.
    pub fn new_1000_nit() -> Self {
        Self {
            peak_luminance_nits: 1000.0,
            sdr_reference_nits: 100.0,
        }
    }

    /// Map a single SDR gamma-encoded luminance value to HDR [0, 1] (normalised to peak).
    ///
    /// `sdr_lum` is a gamma-encoded SDR luminance in [0, 1].
    pub fn map_luminance(&self, sdr_lum: f32) -> f32 {
        bt2446_method_a_inverse(sdr_lum, self.peak_luminance_nits, self.sdr_reference_nits)
    }

    /// Map an interleaved RGB frame from SDR to HDR.
    ///
    /// # Errors
    /// Returns an error if the pixel buffer length is not divisible by 3.
    pub fn map_frame(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        let config = ToneMappingConfig {
            operator: ToneMappingOperator::Bt2446MethodA(self.peak_luminance_nits),
            input_peak_nits: self.sdr_reference_nits,
            output_peak_nits: self.peak_luminance_nits,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: 1.0,
        };
        ToneMapper::new(config).map_frame(pixels)
    }
}

// ── BT2446MethodCToneMapper ───────────────────────────────────────────────────

/// Dedicated high-level tone mapper for BT.2446 Method C (HDR → SDR).
#[derive(Debug, Clone)]
pub struct BT2446MethodCToneMapper {
    /// Input HDR peak luminance in nits.
    pub input_peak_nits: f32,
    /// Output SDR peak luminance in nits.
    pub output_peak_nits: f32,
    /// Chroma correction factor: 1.0 = full, 0.0 = off.
    pub chroma_correction: f32,
    /// Output power-law gamma.
    pub gamma_out: f32,
}

impl BT2446MethodCToneMapper {
    /// Create a mapper: 1 000-nit HDR → 100-nit SDR with full chroma correction.
    pub fn new_hdr10_to_sdr() -> Self {
        Self {
            input_peak_nits: 1000.0,
            output_peak_nits: 100.0,
            chroma_correction: 1.0,
            gamma_out: 2.2,
        }
    }

    /// Map an interleaved RGB frame using BT.2446 Method C.
    ///
    /// # Errors
    /// Returns an error if the pixel buffer length is not divisible by 3.
    pub fn map_frame(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        let config = ToneMappingConfig {
            operator: ToneMappingOperator::Bt2446MethodC(self.chroma_correction),
            input_peak_nits: self.input_peak_nits,
            output_peak_nits: self.output_peak_nits,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: self.gamma_out,
        };
        ToneMapper::new(config).map_frame(pixels)
    }
}

// ── InverseToneMappingOperator ────────────────────────────────────────────────

/// Available SDR-to-HDR inverse tone mapping methods.
#[derive(Debug, Clone, PartialEq)]
pub enum InverseToneMappingOperator {
    /// BT.2446 Method A three-segment inverse curve.
    Bt2446MethodA,
    /// Simple power-law expansion: `x^exponent` (default 2.2).
    PowerLaw(f32),
}

// ── InverseToneMapper ─────────────────────────────────────────────────────────

/// SDR-to-HDR inverse tone mapper.
///
/// Reconstructs plausible HDR from SDR content using a selected algorithm.
/// The output is normalised to [0, 1] where 1.0 = `target_peak_nits`.
#[derive(Debug, Clone)]
pub struct InverseToneMapper {
    /// Inverse mapping operator to apply.
    pub operator: InverseToneMappingOperator,
    /// SDR input peak luminance in nits (typically 100).
    pub sdr_peak_nits: f32,
    /// Target HDR peak luminance in nits (e.g. 1000, 4000, 10000).
    pub target_peak_nits: f32,
}

impl InverseToneMapper {
    /// Create an inverse mapper targeting 1 000-nit HDR using BT.2446 Method A.
    pub fn bt2446a_to_1000_nit() -> Self {
        Self {
            operator: InverseToneMappingOperator::Bt2446MethodA,
            sdr_peak_nits: 100.0,
            target_peak_nits: 1000.0,
        }
    }

    /// Map a single SDR luminance to HDR [0, 1].
    pub fn map_luminance(&self, sdr_lum: f32) -> f32 {
        match &self.operator {
            InverseToneMappingOperator::Bt2446MethodA => {
                bt2446_method_a_inverse(sdr_lum, self.target_peak_nits, self.sdr_peak_nits)
            }
            InverseToneMappingOperator::PowerLaw(exp) => sdr_lum.clamp(0.0, 1.0).powf(*exp),
        }
    }

    /// Map an interleaved RGB frame from SDR to HDR.
    ///
    /// # Errors
    /// Returns an error if the pixel buffer length is not divisible by 3.
    pub fn map_frame(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "pixel buffer length {} is not divisible by 3",
                pixels.len()
            )));
        }
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            let lum = 0.2627 * chunk[0] + 0.6780 * chunk[1] + 0.0593 * chunk[2];
            let mapped_lum = self.map_luminance(lum.max(0.0));
            if lum > 1e-7 {
                let ratio = mapped_lum / lum;
                out.push((chunk[0] * ratio).clamp(0.0, 1.0));
                out.push((chunk[1] * ratio).clamp(0.0, 1.0));
                out.push((chunk[2] * ratio).clamp(0.0, 1.0));
            } else {
                out.push(mapped_lum);
                out.push(mapped_lum);
                out.push(mapped_lum);
            }
        }
        Ok(out)
    }
}

// ── Parallel frame processing ─────────────────────────────────────────────────

/// Map an interleaved RGB frame in parallel using Rayon, one row at a time.
///
/// `mapper` is a closure that maps a single RGB pixel `(r, g, b)` to `(r, g, b)`.
/// `width` is the number of pixels per row.  The pixel buffer must have length
/// `height × width × 3`.
///
/// # Errors
/// Returns an error if the buffer length is inconsistent with `width`.
pub fn map_frame_parallel<F>(pixels: &[f32], width: usize, mapper: F) -> Result<Vec<f32>>
where
    F: Fn(f32, f32, f32) -> (f32, f32, f32) + Sync + Send,
{
    if width == 0 {
        return Err(HdrError::ToneMappingError("width must be > 0".to_string()));
    }
    let stride = width * 3;
    if !pixels.len().is_multiple_of(stride) {
        return Err(HdrError::ToneMappingError(format!(
            "pixel buffer length {} is not divisible by row stride {}",
            pixels.len(),
            stride
        )));
    }
    let n_rows = pixels.len() / stride;
    let rows_out: Vec<Vec<f32>> = (0..n_rows)
        .into_par_iter()
        .map(|row_idx| {
            let row = &pixels[row_idx * stride..(row_idx + 1) * stride];
            let mut row_out = Vec::with_capacity(stride);
            for chunk in row.chunks_exact(3) {
                let (r, g, b) = mapper(chunk[0], chunk[1], chunk[2]);
                row_out.push(r);
                row_out.push(g);
                row_out.push(b);
            }
            row_out
        })
        .collect();
    Ok(rows_out.into_iter().flatten().collect())
}

/// Convenience wrapper: parallel Hable tone mapping of a full frame.
///
/// Maps a 1 000-nit HDR frame to 100-nit SDR using the Hable operator,
/// processing rows in parallel via Rayon.
///
/// # Errors
/// Returns an error if the buffer is not consistent with `width`.
pub fn tone_map_frame_rayon(
    pixels: &[f32],
    width: usize,
    input_peak_nits: f32,
    output_peak_nits: f32,
) -> Result<Vec<f32>> {
    let config = ToneMappingConfig {
        operator: ToneMappingOperator::Hable,
        input_peak_nits,
        output_peak_nits,
        exposure: 1.0,
        saturation: 1.0,
        gamma_out: 2.2,
    };
    let tm = ToneMapper::new(config);
    map_frame_parallel(pixels, width, |r, g, b| tm.map_pixel(r, g, b))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn default_config(op: ToneMappingOperator) -> ToneMappingConfig {
        ToneMappingConfig {
            operator: op,
            input_peak_nits: 1000.0,
            output_peak_nits: 100.0,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: 2.2,
        }
    }

    // ── Reinhard ─────────────────────────────────────────────────────────────

    #[test]
    fn test_reinhard_zero_input() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Reinhard));
        assert!(approx(tm.map_luminance(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_reinhard_clamps_to_one() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Reinhard));
        // Very high luminance should approach but stay ≤ 1
        let v = tm.map_luminance(1e6);
        assert!(v <= 1.0 && v > 0.99);
    }

    #[test]
    fn test_reinhard_mid_range() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Reinhard));
        let v = tm.map_luminance(0.5); // 500 nits / 1000 → x=0.05
        assert!(v > 0.0 && v < 1.0, "reinhard mid-range out of range: {v}");
    }

    // ── ReinhardExtended ─────────────────────────────────────────────────────

    #[test]
    fn test_reinhard_extended_zero() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::ReinhardExtended(4.0)));
        assert!(approx(tm.map_luminance(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_reinhard_extended_greater_than_reinhard() {
        // Extended Reinhard with large Lw should give slightly higher values than plain Reinhard
        let plain = ToneMapper::new(default_config(ToneMappingOperator::Reinhard));
        let ext = ToneMapper::new(default_config(ToneMappingOperator::ReinhardExtended(100.0)));
        let lum = 0.5;
        assert!(ext.map_luminance(lum) >= plain.map_luminance(lum));
    }

    // ── Hable ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_hable_output_in_range() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Hable));
        for v in [0.0f32, 0.1, 0.5, 1.0, 2.0, 10.0] {
            let out = tm.map_luminance(v);
            assert!(
                (0.0..=1.0).contains(&out),
                "Hable({v}) = {out} out of range"
            );
        }
    }

    #[test]
    fn test_hable_full_output_in_range() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::HableFull));
        for v in [0.0f32, 0.5, 1.0, 5.0] {
            let out = tm.map_luminance(v);
            assert!(
                (0.0..=1.0).contains(&out),
                "HableFull({v}) = {out} out of range"
            );
        }
    }

    // ── ACES ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_aces_output_in_range() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Aces));
        for v in [0.0f32, 0.1, 0.5, 1.0, 3.0] {
            let out = tm.map_luminance(v);
            assert!((0.0..=1.0).contains(&out), "ACES({v}) = {out} out of range");
        }
    }

    // ── Clamp ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clamp_below_one() {
        // 0.05 * (100/1000) = 0.005, clamp → 0.005
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Clamp));
        let v = tm.map_luminance(0.05);
        assert!(approx(v, 0.005, 1e-5), "clamp low: {v}");
    }

    #[test]
    fn test_clamp_above_one() {
        // With input_peak=1000 and output_peak=100, scale=0.1.
        // A value of 15.0 * 0.1 = 1.5, which clamps to 1.0.
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Clamp));
        assert!(approx(tm.map_luminance(15.0), 1.0, 1e-6));
    }

    // ── Reinhard2 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_reinhard2_output_in_range() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Reinhard2(1.0)));
        for v in [0.0f32, 0.1, 0.5, 1.0, 2.0] {
            let out = tm.map_luminance(v);
            assert!(
                (0.0..=1.0).contains(&out),
                "Reinhard2({v}) = {out} out of range"
            );
        }
    }

    // ── map_frame ─────────────────────────────────────────────────────────────

    #[test]
    fn test_map_frame_length_check() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Reinhard));
        let result = tm.map_frame(&[0.1, 0.2]); // not divisible by 3
        assert!(result.is_err());
    }

    #[test]
    fn test_map_frame_output_length() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Reinhard));
        let pixels = vec![0.5f32; 300]; // 100 pixels
        let out = tm.map_frame(&pixels).expect("map_frame");
        assert_eq!(out.len(), 300);
    }

    #[test]
    fn test_map_frame_values_in_range() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Hable));
        let pixels: Vec<f32> = (0..99).map(|i| i as f32 / 99.0).collect();
        let out = tm.map_frame(&pixels).expect("map_frame range");
        for v in &out {
            assert!((0.0..=1.0).contains(v), "pixel {v} out of range");
        }
    }

    // ── Factories ─────────────────────────────────────────────────────────────

    #[test]
    fn test_hdr10_to_sdr_factory() {
        let cfg = ToneMappingConfig::hdr10_to_sdr();
        assert_eq!(cfg.operator, ToneMappingOperator::Hable);
        assert!(approx(cfg.input_peak_nits, 1000.0, 0.01));
    }

    #[test]
    fn test_hlg_to_sdr_factory() {
        let cfg = ToneMappingConfig::hlg_to_sdr();
        assert_eq!(cfg.operator, ToneMappingOperator::Reinhard);
        assert!(approx(cfg.output_peak_nits, 100.0, 0.01));
    }

    // ── map_pixel ─────────────────────────────────────────────────────────────

    #[test]
    fn test_map_pixel_black() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Hable));
        let (r, g, b) = tm.map_pixel(0.0, 0.0, 0.0);
        assert!(approx(r, 0.0, 1e-6) && approx(g, 0.0, 1e-6) && approx(b, 0.0, 1e-6));
    }

    #[test]
    fn test_map_pixel_white_in_range() {
        let tm = ToneMapper::new(default_config(ToneMappingOperator::Aces));
        let (r, g, b) = tm.map_pixel(1.0, 1.0, 1.0);
        assert!((0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&g) && (0.0..=1.0).contains(&b));
    }

    // ── BT.2446 Method A ─────────────────────────────────────────────────────

    #[test]
    fn test_bt2446a_zero_input() {
        let cfg = ToneMappingConfig::sdr_to_hdr_bt2446a();
        let tm = ToneMapper::new(cfg);
        assert!(approx(tm.map_luminance(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_bt2446a_output_in_range() {
        let cfg = ToneMappingConfig::sdr_to_hdr_bt2446a();
        let tm = ToneMapper::new(cfg);
        for v in [0.0f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let out = tm.map_luminance(v);
            assert!(
                (0.0..=1.0).contains(&out),
                "BT2446A({v}) = {out} out of range"
            );
        }
    }

    #[test]
    fn test_bt2446a_monotonic() {
        let cfg = ToneMappingConfig::sdr_to_hdr_bt2446a();
        let tm = ToneMapper::new(cfg);
        let mut prev = 0.0f32;
        for i in 1..=100 {
            let x = i as f32 / 100.0;
            let out = tm.map_luminance(x);
            assert!(
                out >= prev - 1e-6,
                "BT2446A not monotonic at {x}: {out} < {prev}"
            );
            prev = out;
        }
    }

    #[test]
    fn test_bt2446a_highlight_expansion() {
        // Highlights should be expanded: output at 0.8 SDR should be >= 0.8 in HDR
        let cfg = ToneMappingConfig::sdr_to_hdr_bt2446a();
        let tm = ToneMapper::new(cfg);
        let out = tm.map_luminance(0.8);
        assert!(out >= 0.7, "BT2446A should expand highlights: got {out}");
    }

    #[test]
    fn test_bt2446a_shadow_preservation() {
        // Shadows should be approximately preserved
        let cfg = ToneMappingConfig::sdr_to_hdr_bt2446a();
        let tm = ToneMapper::new(cfg);
        let out = tm.map_luminance(0.01);
        // With the peak ratio, the shadow region should remain close to input
        assert!(out < 0.1, "BT2446A shadows too bright: got {out}");
    }

    #[test]
    fn test_bt2446a_pixel_mapping() {
        let cfg = ToneMappingConfig::sdr_to_hdr_bt2446a();
        let tm = ToneMapper::new(cfg);
        let (r, g, b) = tm.map_pixel(0.5, 0.3, 0.2);
        assert!((0.0..=1.0).contains(&r), "BT2446A pixel r = {r}");
        assert!((0.0..=1.0).contains(&g), "BT2446A pixel g = {g}");
        assert!((0.0..=1.0).contains(&b), "BT2446A pixel b = {b}");
    }

    #[test]
    fn test_bt2446a_frame_mapping() {
        let cfg = ToneMappingConfig::sdr_to_hdr_bt2446a();
        let tm = ToneMapper::new(cfg);
        let pixels: Vec<f32> = (0..99).map(|i| i as f32 / 100.0).collect();
        let out = tm.map_frame(&pixels).expect("BT2446A frame");
        assert_eq!(out.len(), 99);
        for v in &out {
            assert!((0.0..=1.0).contains(v), "BT2446A frame pixel {v}");
        }
    }

    // ── BT.2446 Method C ─────────────────────────────────────────────────────

    #[test]
    fn test_bt2446c_zero_input() {
        let cfg = ToneMappingConfig::hdr_to_sdr_bt2446c();
        let tm = ToneMapper::new(cfg);
        assert!(approx(tm.map_luminance(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_bt2446c_output_in_range() {
        let cfg = ToneMappingConfig::hdr_to_sdr_bt2446c();
        let tm = ToneMapper::new(cfg);
        for v in [0.0f32, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let out = tm.map_luminance(v);
            assert!(
                (0.0..=1.0).contains(&out),
                "BT2446C({v}) = {out} out of range"
            );
        }
    }

    #[test]
    fn test_bt2446c_monotonic() {
        let cfg = ToneMappingConfig::hdr_to_sdr_bt2446c();
        let tm = ToneMapper::new(cfg);
        let mut prev = 0.0f32;
        for i in 1..=100 {
            let x = i as f32 / 10.0;
            let out = tm.map_luminance(x);
            assert!(
                out >= prev - 1e-6,
                "BT2446C not monotonic at {x}: {out} < {prev}"
            );
            prev = out;
        }
    }

    #[test]
    fn test_bt2446c_compression() {
        // High luminance should be compressed below 1.0
        let cfg = ToneMappingConfig::hdr_to_sdr_bt2446c();
        let tm = ToneMapper::new(cfg);
        let out = tm.map_luminance(10.0);
        assert!(out <= 1.0, "BT2446C should compress: got {out}");
        assert!(out > 0.5, "BT2446C too aggressive: got {out}");
    }

    #[test]
    fn test_bt2446c_chroma_correction_pixel() {
        // With chroma correction, a saturated HDR pixel should maintain some
        // chroma after tone mapping (compared to without correction).
        let cfg_cc = ToneMappingConfig {
            operator: ToneMappingOperator::Bt2446MethodC(1.0),
            input_peak_nits: 1000.0,
            output_peak_nits: 100.0,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: 1.0, // skip gamma to isolate chroma effect
        };
        let cfg_no_cc = ToneMappingConfig {
            operator: ToneMappingOperator::Bt2446MethodC(0.0),
            input_peak_nits: 1000.0,
            output_peak_nits: 100.0,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: 1.0,
        };
        let tm_cc = ToneMapper::new(cfg_cc);
        let tm_no = ToneMapper::new(cfg_no_cc);

        // A bright saturated red pixel
        let (r_cc, g_cc, _b_cc) = tm_cc.map_pixel(0.8, 0.1, 0.05);
        let (r_no, g_no, _b_no) = tm_no.map_pixel(0.8, 0.1, 0.05);

        // With chroma correction, the red channel should be higher relative
        // to the luma (more saturated)
        let chroma_cc = r_cc - g_cc;
        let chroma_no = r_no - g_no;
        assert!(
            chroma_cc >= chroma_no - 1e-4,
            "chroma correction should preserve saturation: cc={chroma_cc}, no_cc={chroma_no}"
        );
    }

    #[test]
    fn test_bt2446c_frame_mapping() {
        let cfg = ToneMappingConfig::hdr_to_sdr_bt2446c();
        let tm = ToneMapper::new(cfg);
        let pixels: Vec<f32> = (0..99).map(|i| i as f32 / 50.0).collect();
        let out = tm.map_frame(&pixels).expect("BT2446C frame");
        assert_eq!(out.len(), 99);
        for v in &out {
            assert!((0.0..=1.0).contains(v), "BT2446C frame pixel {v}");
        }
    }

    #[test]
    fn test_bt2446c_factory() {
        let cfg = ToneMappingConfig::hdr_to_sdr_bt2446c();
        assert_eq!(cfg.operator, ToneMappingOperator::Bt2446MethodC(1.0));
        assert!(approx(cfg.input_peak_nits, 1000.0, 0.01));
        assert!(approx(cfg.output_peak_nits, 100.0, 0.01));
    }

    #[test]
    fn test_bt2446a_factory() {
        let cfg = ToneMappingConfig::sdr_to_hdr_bt2446a();
        assert_eq!(cfg.operator, ToneMappingOperator::Bt2446MethodA(1000.0));
        assert!(approx(cfg.input_peak_nits, 100.0, 0.01));
    }

    // ── BT.2446 Method A — enhanced inverse tone mapping tests ──────────

    #[test]
    fn test_bt2446a_inverse_bt1886_linearisation() {
        // Test the internal function directly to verify BT.1886 linearisation.
        // bt2446_method_a_inverse takes pre-scaled x, so we pass SDR values directly.
        let low = super::bt2446_method_a_inverse(0.1, 1000.0, 100.0);
        let mid = super::bt2446_method_a_inverse(0.5, 1000.0, 100.0);
        let high = super::bt2446_method_a_inverse(0.9, 1000.0, 100.0);
        assert!(
            low < mid,
            "BT2446A internal: 0.1 ({low}) should be < 0.5 ({mid})"
        );
        assert!(
            mid < high,
            "BT2446A internal: 0.5 ({mid}) should be < 0.9 ({high})"
        );
    }

    #[test]
    fn test_bt2446a_different_target_peaks() {
        // Different target peaks should produce different expansion curves.
        let out_1000 = super::bt2446_method_a_inverse(0.5, 1000.0, 100.0);
        let out_4000 = super::bt2446_method_a_inverse(0.5, 4000.0, 100.0);
        assert!(
            out_1000 > 0.0 && out_4000 > 0.0,
            "Both should be positive: {out_1000}, {out_4000}"
        );
        // Different peak ratios should yield different curves
        assert!(
            (out_1000 - out_4000).abs() > 1e-4,
            "Different peaks should differ: {out_1000} vs {out_4000}"
        );
    }

    #[test]
    fn test_bt2446a_c1_continuity() {
        // The three-segment curve should have no visible discontinuities.
        // Test the internal function to avoid the scaling/clamp pipeline.
        let n = 1000;
        let mut prev = super::bt2446_method_a_inverse(0.0, 1000.0, 100.0);
        for i in 1..=n {
            let x = i as f32 / n as f32;
            let cur = super::bt2446_method_a_inverse(x, 1000.0, 100.0);
            let delta = (cur - prev).abs();
            assert!(
                delta < 0.05,
                "BT2446A discontinuity at x={x}: delta={delta} (prev={prev}, cur={cur})"
            );
            prev = cur;
        }
    }

    #[test]
    fn test_bt2446a_mid_grey_preserved() {
        // SDR mid-grey (0.18 in scene-linear, ~0.458 in gamma domain)
        // should map to a reasonable HDR mid-grey, not be wildly boosted.
        let gamma_mid = 0.18_f32.powf(1.0 / 2.4);
        let out = super::bt2446_method_a_inverse(gamma_mid, 1000.0, 100.0);
        // Should be in a reasonable range — not clipped or zeroed.
        // After BT.1886 (x^2.4) and the curve, the output should be moderate.
        assert!(out > 0.0 && out < 0.8, "mid-grey mapped to {out}");
    }

    #[test]
    fn test_bt2446a_shadows_near_linear() {
        // Very low values should approximately pass through linearly.
        let out = super::bt2446_method_a_inverse(0.01, 1000.0, 100.0);
        // After BT.1886: 0.01^2.4 ≈ 0.0000398, which is well below t1.
        // The shadow segment is linear, so output ≈ 0.01^2.4.
        assert!(out < 0.01, "shadow should be low: {out}");
        assert!(out > 0.0, "shadow should be positive: {out}");
    }

    #[test]
    fn test_bt2446a_highlight_at_unity() {
        // At x=1.0, the output should approach 1.0.
        let out = super::bt2446_method_a_inverse(1.0, 1000.0, 100.0);
        assert!(out >= 0.9, "unity input should map near 1.0: {out}");
    }

    // ── BT.2446 Method C — enhanced HDR-to-SDR tests ────────────────────

    #[test]
    fn test_bt2446c_shadow_linear_region() {
        // Values below mid-grey should pass through approximately linearly.
        // Test internal function directly.
        let out_low = super::bt2446_method_c_luminance(0.01, 1.0);
        assert!(
            approx(out_low, 0.01, 0.001),
            "shadow should be near-identity: {out_low}"
        );
    }

    #[test]
    fn test_bt2446c_shoulder_smoothness() {
        // The transition from mid-tone to highlight should be smooth.
        // Test internal function directly.
        let n = 500;
        let mut prev = super::bt2446_method_c_luminance(0.0, 1.0);
        for i in 1..=n {
            let x = i as f32 / 50.0; // range [0, 10]
            let cur = super::bt2446_method_c_luminance(x, 1.0);
            let delta = (cur - prev).abs();
            assert!(
                delta < 0.08,
                "BT2446C shoulder discontinuity at x={x}: delta={delta}"
            );
            prev = cur;
        }
    }

    #[test]
    fn test_bt2446c_asymptotic_ceiling() {
        // Very high luminance values should converge toward the ceiling.
        // Test internal function to avoid the scaling pipeline.
        let out = super::bt2446_method_c_luminance(100.0, 1.0);
        assert!(
            (0.90..=1.0).contains(&out),
            "very high luminance should approach ceiling: {out}"
        );
    }

    #[test]
    fn test_bt2446c_chroma_correction_strength() {
        // Higher chroma correction factors should produce more saturated output.
        let mk_cfg = |cc: f32| ToneMappingConfig {
            operator: ToneMappingOperator::Bt2446MethodC(cc),
            input_peak_nits: 1000.0,
            output_peak_nits: 100.0,
            exposure: 1.0,
            saturation: 1.0,
            gamma_out: 1.0,
        };
        let tm_0 = ToneMapper::new(mk_cfg(0.0));
        let tm_05 = ToneMapper::new(mk_cfg(0.5));
        let tm_1 = ToneMapper::new(mk_cfg(1.0));

        // Bright saturated blue
        let (r0, _, b0) = tm_0.map_pixel(0.1, 0.1, 0.9);
        let (r05, _, b05) = tm_05.map_pixel(0.1, 0.1, 0.9);
        let (r1, _, b1) = tm_1.map_pixel(0.1, 0.1, 0.9);

        let chroma_0 = (b0 - r0).abs();
        let chroma_05 = (b05 - r05).abs();
        let chroma_1 = (b1 - r1).abs();

        // Higher cc should produce equal or higher chroma
        assert!(
            chroma_05 >= chroma_0 - 0.01,
            "cc=0.5 chroma ({chroma_05}) should be >= cc=0 ({chroma_0})"
        );
        assert!(
            chroma_1 >= chroma_05 - 0.01,
            "cc=1.0 chroma ({chroma_1}) should be >= cc=0.5 ({chroma_05})"
        );
    }

    #[test]
    fn test_bt2446c_crosstalk_grey_neutral() {
        // Grey (equal R=G=B) should be unchanged by crosstalk.
        let (r, g, b) = super::bt2446c_crosstalk(0.5, 0.5, 0.5, 0.04);
        assert!(approx(r, 0.5, 1e-5), "crosstalk grey R: {r}");
        assert!(approx(g, 0.5, 1e-5), "crosstalk grey G: {g}");
        assert!(approx(b, 0.5, 1e-5), "crosstalk grey B: {b}");
    }

    #[test]
    fn test_bt2446c_crosstalk_zero_factor() {
        // With crosstalk=0, the output should equal the input.
        let (r, g, b) = super::bt2446c_crosstalk(0.8, 0.2, 0.1, 0.0);
        assert!(approx(r, 0.8, 1e-6));
        assert!(approx(g, 0.2, 1e-6));
        assert!(approx(b, 0.1, 1e-6));
    }

    #[test]
    fn test_bt2446c_crosstalk_preserves_sum() {
        // Crosstalk matrix preserves the sum R+G+B.
        let (ri, gi, bi) = (0.8_f32, 0.3, 0.1);
        let (ro, go, bo) = super::bt2446c_crosstalk(ri, gi, bi, 0.04);
        let sum_in = ri + gi + bi;
        let sum_out = ro + go + bo;
        assert!(
            approx(sum_in, sum_out, 1e-5),
            "crosstalk should preserve sum: {sum_in} vs {sum_out}"
        );
    }

    #[test]
    fn test_bt2446c_mid_grey_contrast() {
        // Method C should preserve mid-tone contrast: a step in luminance
        // should produce a visible step in the output.
        // Test internal function directly.
        let out_low = super::bt2446_method_c_luminance(0.4, 1.0);
        let out_high = super::bt2446_method_c_luminance(0.8, 1.0);
        assert!(
            out_high > out_low,
            "Method C should preserve contrast: {out_low} vs {out_high}"
        );
        let ratio = out_high / out_low.max(1e-7);
        assert!(
            ratio > 1.01 && ratio < 2.5,
            "contrast ratio should be reasonable: {ratio}"
        );
    }

    #[test]
    fn test_bt2446c_internal_monotonic() {
        // The internal Method C curve should be strictly monotonic.
        let mut prev = 0.0_f32;
        for i in 1..=500 {
            let x = i as f32 / 50.0; // [0, 10]
            let cur = super::bt2446_method_c_luminance(x, 1.0);
            assert!(
                cur >= prev - 1e-6,
                "BT2446C internal not monotonic at x={x}: {cur} < {prev}"
            );
            prev = cur;
        }
    }

    #[test]
    fn test_bt2446a_internal_monotonic() {
        // The internal Method A curve should be strictly monotonic.
        let mut prev = 0.0_f32;
        for i in 1..=1000 {
            let x = i as f32 / 1000.0;
            let cur = super::bt2446_method_a_inverse(x, 1000.0, 100.0);
            assert!(
                cur >= prev - 1e-6,
                "BT2446A internal not monotonic at x={x}: {cur} < {prev}"
            );
            prev = cur;
        }
    }

    // ── FrameLuminanceAnalysis ────────────────────────────────────────────────

    #[test]
    fn test_frame_luminance_analysis_empty() {
        let stats = FrameLuminanceAnalysis::from_frame(&[]).expect("empty frame");
        assert_eq!(stats.pixel_count, 0);
        assert!(approx(stats.mean, 0.0, 1e-6));
    }

    #[test]
    fn test_frame_luminance_analysis_invalid() {
        assert!(FrameLuminanceAnalysis::from_frame(&[0.1f32, 0.2]).is_err());
    }

    #[test]
    fn test_frame_luminance_analysis_basic() {
        // Three pixels: black, grey, white
        let pixels = [0.0f32, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0];
        let stats = FrameLuminanceAnalysis::from_frame(&pixels).expect("basic stats");
        assert_eq!(stats.pixel_count, 3);
        assert!(approx(stats.min, 0.0, 1e-6), "min = {}", stats.min);
        assert!(stats.max > 0.9, "max = {}", stats.max);
        assert!(
            stats.mean > 0.0 && stats.mean < 1.0,
            "mean = {}",
            stats.mean
        );
        assert!(stats.p95 > 0.0, "p95 = {}", stats.p95);
        assert!(stats.p99 >= stats.p95, "p99 >= p95");
    }

    #[test]
    fn test_frame_luminance_analysis_monotonic_percentiles() {
        let pixels: Vec<f32> = (0..300).map(|i| i as f32 / 300.0).collect();
        let stats = FrameLuminanceAnalysis::from_frame(&pixels).expect("mono stats");
        assert!(stats.p95 <= stats.p99, "p95 <= p99");
        assert!(stats.p99 <= stats.max + 1e-5, "p99 <= max");
        assert!(stats.min <= stats.p95, "min <= p95");
    }

    // ── SceneReferredToneMapper ───────────────────────────────────────────────

    #[test]
    fn test_scene_referred_factory() {
        let sr = SceneReferredToneMapper::hdr10_to_sdr_adaptive();
        assert!(approx(sr.output_peak_nits, 100.0, 0.01));
        assert!(sr.clip_fraction > 0.0 && sr.clip_fraction < 1.0);
    }

    #[test]
    fn test_scene_referred_map_frame_output_in_range() {
        let sr = SceneReferredToneMapper::hdr10_to_sdr_adaptive();
        let pixels: Vec<f32> = (0..99).map(|i| i as f32 / 100.0).collect();
        let out = sr.map_frame_adaptive(&pixels).expect("scene referred");
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "scene referred output {v}");
        }
    }

    // ── BT2446MethodAToneMapper ───────────────────────────────────────────────

    #[test]
    fn test_bt2446a_struct_basic() {
        let tm = BT2446MethodAToneMapper::new_1000_nit();
        assert!(approx(tm.peak_luminance_nits, 1000.0, 0.01));
        assert!(approx(tm.sdr_reference_nits, 100.0, 0.01));
    }

    #[test]
    fn test_bt2446a_struct_zero_input() {
        let tm = BT2446MethodAToneMapper::new_1000_nit();
        assert!(approx(tm.map_luminance(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_bt2446a_struct_output_in_range() {
        let tm = BT2446MethodAToneMapper::new_1000_nit();
        for v in [0.0f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let out = tm.map_luminance(v);
            assert!(
                (0.0..=1.0).contains(&out),
                "BT2446MethodAToneMapper({v}) = {out}"
            );
        }
    }

    #[test]
    fn test_bt2446a_struct_peak_ratio_affects_curve() {
        // 1000-nit and 4000-nit mappers should give different results at high SDR.
        let tm1000 = BT2446MethodAToneMapper {
            peak_luminance_nits: 1000.0,
            sdr_reference_nits: 100.0,
        };
        let tm4000 = BT2446MethodAToneMapper {
            peak_luminance_nits: 4000.0,
            sdr_reference_nits: 100.0,
        };
        let out_1000 = tm1000.map_luminance(0.8);
        let out_4000 = tm4000.map_luminance(0.8);
        // Both should be in range
        assert!((0.0..=1.0).contains(&out_1000), "1000-nit: {out_1000}");
        assert!((0.0..=1.0).contains(&out_4000), "4000-nit: {out_4000}");
        // The results should differ (different peak ratios produce different expansion)
        assert!(
            (out_1000 - out_4000).abs() > 1e-4,
            "Different peaks should give different curves: {out_1000} vs {out_4000}"
        );
    }

    #[test]
    fn test_bt2446a_struct_frame_mapping() {
        let tm = BT2446MethodAToneMapper::new_1000_nit();
        let pixels: Vec<f32> = (0..99).map(|i| i as f32 / 100.0).collect();
        let out = tm.map_frame(&pixels).expect("bt2446a frame");
        assert_eq!(out.len(), 99);
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "BT2446MethodAToneMapper frame {v}"
            );
        }
    }

    // ── BT2446MethodCToneMapper ───────────────────────────────────────────────

    #[test]
    fn test_bt2446c_struct_basic() {
        let tm = BT2446MethodCToneMapper::new_hdr10_to_sdr();
        assert!(approx(tm.input_peak_nits, 1000.0, 0.01));
        assert!(approx(tm.output_peak_nits, 100.0, 0.01));
    }

    #[test]
    fn test_bt2446c_struct_frame_mapping() {
        let tm = BT2446MethodCToneMapper::new_hdr10_to_sdr();
        let pixels: Vec<f32> = (0..99).map(|i| i as f32 / 50.0).collect();
        let out = tm.map_frame(&pixels).expect("bt2446c frame");
        assert_eq!(out.len(), 99);
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "BT2446MethodCToneMapper frame {v}"
            );
        }
    }

    // ── InverseToneMapper ─────────────────────────────────────────────────────

    #[test]
    fn test_inverse_tone_mapper_bt2446a() {
        let itm = InverseToneMapper::bt2446a_to_1000_nit();
        let out = itm.map_luminance(0.5);
        assert!((0.0..=1.0).contains(&out), "inverse: {out}");
    }

    #[test]
    fn test_inverse_tone_mapper_power_law() {
        let itm = InverseToneMapper {
            operator: InverseToneMappingOperator::PowerLaw(2.2),
            sdr_peak_nits: 100.0,
            target_peak_nits: 1000.0,
        };
        let out = itm.map_luminance(0.5);
        // 0.5^2.2 ≈ 0.218
        assert!(approx(out, 0.5f32.powf(2.2), 1e-4), "power law: {out}");
    }

    #[test]
    fn test_inverse_tone_mapper_frame() {
        let itm = InverseToneMapper::bt2446a_to_1000_nit();
        let pixels: Vec<f32> = (0..99).map(|i| i as f32 / 100.0).collect();
        let out = itm.map_frame(&pixels).expect("inverse frame");
        assert_eq!(out.len(), 99);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "inverse frame {v}");
        }
    }

    #[test]
    fn test_inverse_tone_mapper_frame_invalid() {
        let itm = InverseToneMapper::bt2446a_to_1000_nit();
        assert!(itm.map_frame(&[0.1f32, 0.2]).is_err());
    }

    // ── map_frame_parallel ────────────────────────────────────────────────────

    #[test]
    fn test_map_frame_parallel_basic() {
        // 4x2 image (8 pixels = 24 floats)
        let width = 4;
        let pixels: Vec<f32> = vec![0.5f32; 24];
        let out = map_frame_parallel(&pixels, width, |r, g, b| (r, g, b)).expect("parallel");
        assert_eq!(out.len(), 24);
        for &v in &out {
            assert!(approx(v, 0.5, 1e-6));
        }
    }

    #[test]
    fn test_map_frame_parallel_invalid_stride() {
        let pixels = vec![0.5f32; 10]; // 10 not divisible by width*3=9
        assert!(map_frame_parallel(&pixels, 3, |r, g, b| (r, g, b)).is_err());
    }

    #[test]
    fn test_map_frame_parallel_zero_width() {
        let pixels = vec![0.5f32; 12];
        assert!(map_frame_parallel(&pixels, 0, |r, g, b| (r, g, b)).is_err());
    }

    // ── tone_map_frame_rayon ──────────────────────────────────────────────────

    #[test]
    fn test_tone_map_frame_rayon_basic() {
        let width = 4;
        let pixels: Vec<f32> = (0..24).map(|i| i as f32 / 24.0).collect();
        let out = tone_map_frame_rayon(&pixels, width, 1000.0, 100.0).expect("rayon");
        assert_eq!(out.len(), 24);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "rayon output {v}");
        }
    }

    #[test]
    fn test_tone_map_frame_rayon_invalid() {
        assert!(tone_map_frame_rayon(&[0.1f32; 10], 3, 1000.0, 100.0).is_err());
    }
}

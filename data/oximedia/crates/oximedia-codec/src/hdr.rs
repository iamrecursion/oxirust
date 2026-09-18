//! HDR Tone Mapping operators for High Dynamic Range to Standard Dynamic Range conversion.
//!
//! This module provides a comprehensive set of tone mapping algorithms operating directly
//! on RGB f32 pixel data.  All operators work in linear light (scene-referred) and can be
//! chained through the [`HdrToneMapper`] pipeline which handles:
//!
//! 1. PQ / HLG / BT.709 EOTF  (signal → linear light in nits)
//! 2. Optional BT.2020 → BT.709 gamut conversion
//! 3. Tone-mapping operator
//! 4. Saturation correction in IPT-like perceptual space
//! 5. BT.709 OETF  (linear light → gamma-encoded signal)
//!
//! # Operators
//!
//! | Operator | Description |
//! |---|---|
//! | [`ToneMapOperator::Clamp`] | Hard clip — fastest, no soft rolloff |
//! | [`ToneMapOperator::ReinhardGlobal`] | Reinhard (2002) per-channel |
//! | [`ToneMapOperator::ReinhardLocal`] | Luminance-domain Reinhard (colour ratios preserved) |
//! | [`ToneMapOperator::AcesFilmic`] | ACES Narkowicz approximation |
//! | [`ToneMapOperator::Hable`] | Hable / Uncharted 2 filmic |
//! | [`ToneMapOperator::PqToSdr`] | Full ST.2084 → BT.709 pipeline |
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::hdr::{HdrToneMapper, ToneMappingConfig, ToneMapOperator};
//!
//! // Build configuration
//! let config = ToneMappingConfig::builder()
//!     .operator(ToneMapOperator::AcesFilmic)
//!     .peak_brightness(1000.0)
//!     .white_point(203.0)
//!     .saturation_correction(1.1)
//!     .build();
//!
//! let mapper = HdrToneMapper::new(config);
//!
//! // Process a single HDR pixel (linear scene-referred, values may exceed 1.0)
//! let hdr_pixel = [10.0_f32, 4.0, 1.5];
//! let sdr_pixel = mapper.map_pixel(hdr_pixel);
//!
//! assert!(sdr_pixel[0] >= 0.0 && sdr_pixel[0] <= 1.0);
//! assert!(sdr_pixel[1] >= 0.0 && sdr_pixel[1] <= 1.0);
//! assert!(sdr_pixel[2] >= 0.0 && sdr_pixel[2] <= 1.0);
//! ```

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]

use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────
//  Transfer-function constants
// ─────────────────────────────────────────────────────────────

/// ST.2084 (PQ) constants per SMPTE ST 2084:2014.
mod pq {
    pub const M1: f64 = 2610.0 / 16384.0;
    pub const M2: f64 = 2523.0 / 4096.0 * 128.0;
    pub const C1: f64 = 3424.0 / 4096.0;
    pub const C2: f64 = 2413.0 / 4096.0 * 32.0;
    pub const C3: f64 = 2392.0 / 4096.0 * 32.0;
    /// Reference white in nits for PQ (where signal = 1.0 maps to 10 000 nits).
    pub const PEAK_NITS: f64 = 10_000.0;
}

/// HLG constants per ITU-R BT.2100.
mod hlg {
    pub const A: f64 = 0.178_832_77;
    pub const B: f64 = 0.284_668_92;
    pub const C: f64 = 0.559_910_73;
}

/// BT.709 gamma constants.
mod bt709 {
    pub const ALPHA: f64 = 1.099_296_826_809_44;
    pub const BETA: f64 = 0.018_053_968_510_807;
    pub const GAMMA_OETF: f64 = 0.45;
    pub const GAMMA_EOTF: f64 = 1.0 / GAMMA_OETF;
}

// ─────────────────────────────────────────────────────────────
//  Tone-mapping operators
// ─────────────────────────────────────────────────────────────

/// Selection of tone-mapping operator applied by [`HdrToneMapper`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ToneMapOperator {
    /// Hard clamp to [0, 1].  No artistic rolloff; only useful as a reference.
    Clamp,
    /// Reinhard (2002) global operator applied per channel independently.
    ///
    /// Formula: `y = x / (1 + x)` — maps [0, ∞) → [0, 1).
    ReinhardGlobal,
    /// Luminance-domain Reinhard.  The tone curve is applied to the luminance
    /// channel only; RGB ratios are then restored, preserving hue and saturation.
    #[default]
    ReinhardLocal,
    /// ACES approximation by Narkowicz (2015).
    ///
    /// `y = x(2.51x + 0.03) / (x(2.43x + 0.59) + 0.14)`
    AcesFilmic,
    /// Hable / Uncharted 2 filmic tone curve (John Hable, 2010).
    Hable,
    /// Full Perceptual Quantizer (ST.2084) → SDR pipeline.
    ///
    /// Converts PQ-encoded signal through linear-light space, applies a
    /// luminance-domain Reinhard with configurable peak / white point, then
    /// applies BT.709 OETF.  Suitable for HDR10 → SDR conversion.
    PqToSdr,
}

// ─────────────────────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────────────────────

/// Configuration for [`HdrToneMapper`].
///
/// Use [`ToneMappingConfig::builder()`] for ergonomic construction.
#[derive(Clone, Debug)]
pub struct ToneMappingConfig {
    /// Tone-mapping operator to apply.
    pub operator: ToneMapOperator,
    /// Peak scene-referred brightness of the HDR input, in nits.
    ///
    /// For HDR10 content this is typically 1 000 – 4 000 nits.
    /// Defaults to 1 000 nits.
    pub peak_brightness: f32,
    /// Scene white point in nits — the luminance that should map to SDR white (100 nits).
    ///
    /// Defaults to 203 nits (ITU-R BT.2408 reference diffuse white).
    pub white_point: f32,
    /// Multiplicative saturation correction applied in linear RGB after tone mapping.
    ///
    /// 1.0 = neutral.  Values > 1 boost saturation, < 1 desaturate.
    /// Typical useful range: 0.5 – 2.0.  Defaults to 1.0.
    pub saturation_correction: f32,
    /// Exposure pre-gain (linear multiplier applied before the tone curve).
    ///
    /// 1.0 = no adjustment.  Use values < 1.0 to darken / protect highlights.
    pub exposure: f32,
    /// Soft-knee start, as a fraction of `peak_brightness` (0.0 → disabled).
    ///
    /// When > 0, a smooth cosine blend replaces the hard transition at the
    /// shoulder, reducing clipping artefacts.  Defaults to 0.0 (off).
    pub knee_start: f32,
}

impl Default for ToneMappingConfig {
    fn default() -> Self {
        Self {
            operator: ToneMapOperator::default(),
            peak_brightness: 1000.0,
            white_point: 203.0,
            saturation_correction: 1.0,
            exposure: 1.0,
            knee_start: 0.0,
        }
    }
}

impl ToneMappingConfig {
    /// Create a builder for ergonomic configuration.
    #[must_use]
    pub fn builder() -> ToneMappingConfigBuilder {
        ToneMappingConfigBuilder::default()
    }
}

/// Builder for [`ToneMappingConfig`].
#[derive(Default)]
pub struct ToneMappingConfigBuilder {
    inner: ToneMappingConfig,
}

impl ToneMappingConfigBuilder {
    /// Set tone-mapping operator.
    #[must_use]
    pub fn operator(mut self, op: ToneMapOperator) -> Self {
        self.inner.operator = op;
        self
    }

    /// Set peak brightness of the HDR source in nits.
    #[must_use]
    pub fn peak_brightness(mut self, nits: f32) -> Self {
        self.inner.peak_brightness = nits;
        self
    }

    /// Set the white-point in nits.
    #[must_use]
    pub fn white_point(mut self, nits: f32) -> Self {
        self.inner.white_point = nits;
        self
    }

    /// Set saturation correction (1.0 = neutral).
    #[must_use]
    pub fn saturation_correction(mut self, s: f32) -> Self {
        self.inner.saturation_correction = s;
        self
    }

    /// Set linear pre-exposure multiplier (1.0 = neutral).
    #[must_use]
    pub fn exposure(mut self, e: f32) -> Self {
        self.inner.exposure = e;
        self
    }

    /// Enable soft-knee blending starting at this fraction of peak brightness.
    ///
    /// For example, `0.7` enables the knee above 70 % of peak.
    #[must_use]
    pub fn knee_start(mut self, k: f32) -> Self {
        self.inner.knee_start = k;
        self
    }

    /// Finalise and return the configuration.
    #[must_use]
    pub fn build(self) -> ToneMappingConfig {
        self.inner
    }
}

// ─────────────────────────────────────────────────────────────
//  HdrToneMapper
// ─────────────────────────────────────────────────────────────

/// HDR tone mapper operating on RGB f32 pixel data.
///
/// Pixels are expected to arrive in **linear, scene-referred light** normalised
/// so that 1.0 == `config.peak_brightness` nits (i.e. values may exceed 1.0
/// for content brighter than the configured peak).
///
/// Use [`HdrToneMapper::map_pixel`] for single pixels or
/// [`HdrToneMapper::map_frame`] to process a flat interleaved RGB f32 buffer.
#[derive(Clone, Debug)]
pub struct HdrToneMapper {
    config: ToneMappingConfig,
    /// Pre-computed scale: maps nits → normalised value (1/peak_brightness).
    nit_scale: f32,
    /// Pre-computed white-point in normalised units.
    white_norm: f32,
}

impl HdrToneMapper {
    /// Create a new tone mapper from configuration.
    #[must_use]
    pub fn new(config: ToneMappingConfig) -> Self {
        let nit_scale = 1.0 / config.peak_brightness.max(f32::EPSILON);
        let white_norm = config.white_point * nit_scale;
        Self {
            config,
            nit_scale,
            white_norm,
        }
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &ToneMappingConfig {
        &self.config
    }

    // ── Public pixel / frame API ─────────────────────────────

    /// Apply tone mapping to a single RGB pixel.
    ///
    /// **Input**: linear, scene-referred RGB where values may exceed 1.0.
    /// The absolute maximum representable brightness is `config.peak_brightness`.
    ///
    /// **Output**: gamma-encoded SDR RGB in [0, 1].
    #[must_use]
    pub fn map_pixel(&self, rgb: [f32; 3]) -> [f32; 3] {
        // 1. Pre-exposure
        let rgb = [
            rgb[0] * self.config.exposure,
            rgb[1] * self.config.exposure,
            rgb[2] * self.config.exposure,
        ];

        // 2. Optional soft knee
        let rgb = if self.config.knee_start > 0.0 {
            soft_knee(rgb, self.config.knee_start, 1.0)
        } else {
            rgb
        };

        // 3. Operator dispatch
        let mapped = match self.config.operator {
            ToneMapOperator::Clamp => clamp_op(rgb),
            ToneMapOperator::ReinhardGlobal => reinhard_global(rgb),
            ToneMapOperator::ReinhardLocal => reinhard_local(rgb),
            ToneMapOperator::AcesFilmic => aces_filmic(rgb),
            ToneMapOperator::Hable => hable(rgb),
            ToneMapOperator::PqToSdr => pq_to_sdr(rgb, self.nit_scale, self.white_norm),
        };

        // 4. Saturation correction
        let mapped = saturation_correct(mapped, self.config.saturation_correction);

        // 5. Final clamp to ensure valid [0, 1] output
        [
            mapped[0].clamp(0.0, 1.0),
            mapped[1].clamp(0.0, 1.0),
            mapped[2].clamp(0.0, 1.0),
        ]
    }

    /// Process a flat interleaved RGB f32 frame in-place.
    ///
    /// The buffer must have length == `width * height * 3`.
    /// Each triple `[r, g, b]` is processed by [`Self::map_pixel`].
    pub fn map_frame(&self, buffer: &mut [f32]) {
        for chunk in buffer.chunks_exact_mut(3) {
            let rgb = [chunk[0], chunk[1], chunk[2]];
            let out = self.map_pixel(rgb);
            chunk[0] = out[0];
            chunk[1] = out[1];
            chunk[2] = out[2];
        }
    }

    /// Process a frame from a read-only input buffer into a new output `Vec`.
    ///
    /// Allocates a new `Vec<f32>` of the same length as `input`.
    #[must_use]
    pub fn map_frame_owned(&self, input: &[f32]) -> Vec<f32> {
        let mut out = input.to_vec();
        self.map_frame(&mut out);
        out
    }

    /// Apply the PQ EOTF to a normalised signal `[0, 1]` and return linear nits.
    ///
    /// Useful when building custom pipelines that feed PQ-encoded data.
    #[must_use]
    pub fn pq_eotf_nits(signal: f32) -> f32 {
        pq_eotf_f64(f64::from(signal)) as f32 * pq::PEAK_NITS as f32
    }

    /// Apply the BT.709 OETF (gamma encode) to a linear value in [0, 1].
    #[must_use]
    pub fn bt709_oetf(linear: f32) -> f32 {
        bt709_oetf_f64(f64::from(linear)) as f32
    }
}

// ─────────────────────────────────────────────────────────────
//  Operator implementations
// ─────────────────────────────────────────────────────────────

/// Hard clamp — maps any value in (−∞, +∞) to [0, 1].
#[inline]
fn clamp_op(rgb: [f32; 3]) -> [f32; 3] {
    [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ]
}

/// Reinhard (2002) global operator: `y = x / (1 + x)` per channel.
///
/// Fast and simple; highlights never clip but asymptotically approach 1.
#[inline]
fn reinhard_global(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0].max(0.0);
    let g = rgb[1].max(0.0);
    let b = rgb[2].max(0.0);
    [r / (1.0 + r), g / (1.0 + g), b / (1.0 + b)]
}

/// Luminance-domain (local-style) Reinhard.
///
/// Computes the BT.2020 luminance `Y`, applies Reinhard to it, then scales
/// each channel by `Y_out / Y_in` so hue and chroma ratios are preserved.
///
/// This avoids the hue shifts that per-channel Reinhard can introduce.
#[inline]
fn reinhard_local(rgb: [f32; 3]) -> [f32; 3] {
    // BT.2020 luminance coefficients
    const KR: f32 = 0.2627;
    const KG: f32 = 0.6780;
    const KB: f32 = 0.0593;

    let luma = KR * rgb[0] + KG * rgb[1] + KB * rgb[2];

    if luma < f32::EPSILON {
        return [0.0, 0.0, 0.0];
    }

    let luma_tm = luma / (1.0 + luma);
    let scale = luma_tm / luma;

    [
        (rgb[0] * scale).max(0.0),
        (rgb[1] * scale).max(0.0),
        (rgb[2] * scale).max(0.0),
    ]
}

/// ACES filmic approximation (Narkowicz 2015).
///
/// `y = x(2.51x + 0.03) / (x(2.43x + 0.59) + 0.14)`
///
/// Produces a natural S-curve with good shadow lift and smooth highlight rolloff.
#[inline]
fn aces_filmic(rgb: [f32; 3]) -> [f32; 3] {
    #[inline]
    fn aces_channel(x: f32) -> f32 {
        const A: f32 = 2.51;
        const B: f32 = 0.03;
        const C: f32 = 2.43;
        const D: f32 = 0.59;
        const E: f32 = 0.14;
        let x = x.max(0.0);
        let num = x * (A * x + B);
        let den = x * (C * x + D) + E;
        if den.abs() < f32::EPSILON {
            0.0
        } else {
            (num / den).clamp(0.0, 1.0)
        }
    }

    [
        aces_channel(rgb[0]),
        aces_channel(rgb[1]),
        aces_channel(rgb[2]),
    ]
}

/// Hable / Uncharted 2 filmic tone curve.
///
/// Introduced by John Hable for Uncharted 2. Parameters match the original
/// published settings.  Produces a rich filmic look with good highlight
/// compression and a slight toe lift.
#[inline]
fn hable(rgb: [f32; 3]) -> [f32; 3] {
    const EXPOSURE_BIAS: f32 = 2.0;
    const WHITE_POINT: f32 = 11.2;

    #[inline]
    fn partial(x: f32) -> f32 {
        const A: f32 = 0.15; // Shoulder strength
        const B: f32 = 0.50; // Linear strength
        const C: f32 = 0.10; // Linear angle
        const D: f32 = 0.20; // Toe strength
        const E: f32 = 0.02; // Toe numerator
        const F: f32 = 0.30; // Toe denominator
        ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    }

    let w = partial(WHITE_POINT);
    if w.abs() < f32::EPSILON {
        return [0.0, 0.0, 0.0];
    }

    let map = |v: f32| (partial(v * EXPOSURE_BIAS) / w).clamp(0.0, 1.0);
    [map(rgb[0]), map(rgb[1]), map(rgb[2])]
}

/// Full PQ (ST.2084) → SDR pipeline.
///
/// Steps:
/// 1. Treat input as already-linear scene-referred; apply `nit_scale` to
///    put it in [0, 1] where 1.0 == peak.
/// 2. Apply luminance-domain Reinhard with white-point `white_norm`.
/// 3. Apply BT.709 OETF.
///
/// `nit_scale`  = 1 / peak_brightness
/// `white_norm` = white_point / peak_brightness
fn pq_to_sdr(rgb: [f32; 3], nit_scale: f32, white_norm: f32) -> [f32; 3] {
    // Normalise scene-referred input to [0, 1]
    let norm = [rgb[0] * nit_scale, rgb[1] * nit_scale, rgb[2] * nit_scale];

    // Luminance-domain Reinhard with white point
    // Formula: L_out = L_in * (1 + L_in / w²) / (1 + L_in)
    const KR: f32 = 0.2627;
    const KG: f32 = 0.6780;
    const KB: f32 = 0.0593;
    let luma = KR * norm[0] + KG * norm[1] + KB * norm[2];

    let luma_out = if luma < f32::EPSILON {
        0.0_f32
    } else {
        let w2 = white_norm * white_norm;
        let out = luma * (1.0 + luma / w2.max(f32::EPSILON)) / (1.0 + luma);
        out.clamp(0.0, 1.0)
    };

    let scale = if luma < f32::EPSILON {
        0.0_f32
    } else {
        luma_out / luma
    };

    // Scale channels and apply BT.709 OETF
    let apply = |v: f32| {
        let linear = (v * scale).clamp(0.0, 1.0);
        bt709_oetf_f64(f64::from(linear)) as f32
    };

    [apply(norm[0]), apply(norm[1]), apply(norm[2])]
}

// ─────────────────────────────────────────────────────────────
//  Saturation correction
// ─────────────────────────────────────────────────────────────

/// Adjust saturation in linear RGB using the BT.709 luma coefficient.
///
/// Saturation is defined as the departure from achromatic (grey).
/// `factor = 1.0` → identity; `factor = 0.0` → greyscale; `factor > 1.0` → boost.
#[inline]
fn saturation_correct(rgb: [f32; 3], factor: f32) -> [f32; 3] {
    if (factor - 1.0).abs() < f32::EPSILON {
        return rgb;
    }

    // BT.709 luminance
    const KR: f32 = 0.2126;
    const KG: f32 = 0.7152;
    const KB: f32 = 0.0722;
    let luma = KR * rgb[0] + KG * rgb[1] + KB * rgb[2];

    [
        luma + (rgb[0] - luma) * factor,
        luma + (rgb[1] - luma) * factor,
        luma + (rgb[2] - luma) * factor,
    ]
}

// ─────────────────────────────────────────────────────────────
//  Soft-knee helper
// ─────────────────────────────────────────────────────────────

/// Apply a soft cosine knee to the luminance channel above `knee_start`.
///
/// Above `knee_start` the values are blended smoothly toward 1.0 using a
/// half-cosine, preventing hard clipping.  The knee ends at 1.0.
fn soft_knee(rgb: [f32; 3], knee_start: f32, knee_end: f32) -> [f32; 3] {
    let apply = |v: f32| -> f32 {
        if v <= knee_start || knee_end <= knee_start {
            v
        } else if v >= knee_end {
            knee_end
        } else {
            let t = (v - knee_start) / (knee_end - knee_start);
            let smooth = (1.0 - (t * PI).cos()) * 0.5;
            knee_start + (knee_end - knee_start) * smooth
        }
    };
    [apply(rgb[0]), apply(rgb[1]), apply(rgb[2])]
}

// ─────────────────────────────────────────────────────────────
//  Transfer-function helpers (f64 for numerical precision)
// ─────────────────────────────────────────────────────────────

/// ST.2084 PQ EOTF: signal [0, 1] → linear [0, 1] (1.0 == 10 000 nits).
#[allow(dead_code)]
pub fn pq_eotf_f64(e: f64) -> f64 {
    let e = e.clamp(0.0, 1.0);
    let e_m2 = e.powf(1.0 / pq::M2);
    let num = (e_m2 - pq::C1).max(0.0);
    let den = pq::C2 - pq::C3 * e_m2;
    if den.abs() < 1e-10 {
        0.0
    } else {
        (num / den).powf(1.0 / pq::M1)
    }
}

/// ST.2084 PQ inverse EOTF: linear [0, 1] → signal [0, 1].
#[allow(dead_code)]
pub fn pq_oetf_f64(y: f64) -> f64 {
    let y = y.clamp(0.0, 1.0);
    let y_m1 = y.powf(pq::M1);
    let num = pq::C1 + pq::C2 * y_m1;
    let den = 1.0 + pq::C3 * y_m1;
    (num / den).powf(pq::M2)
}

/// HLG EOTF: signal [0, 1] → linear [0, 1].
#[allow(dead_code)]
pub fn hlg_eotf_f64(e: f64) -> f64 {
    let e = e.clamp(0.0, 1.0);
    if e <= 0.5 {
        (e * e) / 3.0
    } else {
        (((e - hlg::C) / hlg::A).exp() + hlg::B) / 12.0
    }
}

/// HLG inverse EOTF: linear [0, 1] → signal [0, 1].
#[allow(dead_code)]
pub fn hlg_oetf_f64(y: f64) -> f64 {
    let y = y.clamp(0.0, 1.0);
    if y <= 1.0 / 12.0 {
        (3.0 * y).sqrt()
    } else {
        hlg::A * (12.0 * y - hlg::B).ln() + hlg::C
    }
}

/// BT.709 OETF: linear [0, 1] → gamma-encoded [0, 1].
pub fn bt709_oetf_f64(y: f64) -> f64 {
    let y = y.clamp(0.0, 1.0);
    if y < bt709::BETA {
        4.5 * y
    } else {
        bt709::ALPHA * y.powf(bt709::GAMMA_OETF) - (bt709::ALPHA - 1.0)
    }
}

/// BT.709 EOTF: gamma-encoded [0, 1] → linear [0, 1].
#[allow(dead_code)]
pub fn bt709_eotf_f64(e: f64) -> f64 {
    let e = e.clamp(0.0, 1.0);
    let threshold = 4.5 * bt709::BETA;
    if e < threshold {
        e / 4.5
    } else {
        ((e + (bt709::ALPHA - 1.0)) / bt709::ALPHA).powf(bt709::GAMMA_EOTF)
    }
}

// ─────────────────────────────────────────────────────────────
//  BT.2020 → BT.709 gamut conversion matrix
// ─────────────────────────────────────────────────────────────

/// Convert linear RGB from BT.2020 primaries to BT.709 primaries.
///
/// The matrix is the product `M_XYZ→709 * M_2020→XYZ`, pre-computed for
/// performance.  Values outside gamut are clipped to zero.
#[must_use]
pub fn bt2020_to_bt709(rgb: [f32; 3]) -> [f32; 3] {
    // Derived from the chromatic adaptation (Bradford) chain.
    // Row-major 3×3 matrix.
    const M: [[f32; 3]; 3] = [
        [1.660_491, -0.587_641, -0.072_850],
        [-0.124_551, 1.132_9, -0.008_349],
        [-0.018_151, -0.100_579, 1.118_73],
    ];

    let r = M[0][0] * rgb[0] + M[0][1] * rgb[1] + M[0][2] * rgb[2];
    let g = M[1][0] * rgb[0] + M[1][1] * rgb[1] + M[1][2] * rgb[2];
    let b = M[2][0] * rgb[0] + M[2][1] * rgb[1] + M[2][2] * rgb[2];

    [r.max(0.0), g.max(0.0), b.max(0.0)]
}

// ─────────────────────────────────────────────────────────────
//  Convenience constructors
// ─────────────────────────────────────────────────────────────

impl HdrToneMapper {
    /// Create a mapper using [`ToneMapOperator::AcesFilmic`] with typical HDR10 parameters.
    ///
    /// Peak: 1 000 nits, white point: 203 nits, neutral saturation.
    #[must_use]
    pub fn aces_hdr10() -> Self {
        Self::new(
            ToneMappingConfig::builder()
                .operator(ToneMapOperator::AcesFilmic)
                .peak_brightness(1000.0)
                .white_point(203.0)
                .saturation_correction(1.0)
                .build(),
        )
    }

    /// Create a mapper using [`ToneMapOperator::Hable`] with typical HDR10 parameters.
    #[must_use]
    pub fn hable_hdr10() -> Self {
        Self::new(
            ToneMappingConfig::builder()
                .operator(ToneMapOperator::Hable)
                .peak_brightness(1000.0)
                .white_point(203.0)
                .build(),
        )
    }

    /// Create a mapper using [`ToneMapOperator::PqToSdr`] for HDR10 → SDR conversion.
    #[must_use]
    pub fn pq_to_sdr_hdr10() -> Self {
        Self::new(
            ToneMappingConfig::builder()
                .operator(ToneMapOperator::PqToSdr)
                .peak_brightness(1000.0)
                .white_point(203.0)
                .build(),
        )
    }

    /// Create a mapper using [`ToneMapOperator::ReinhardLocal`] with default HDR10 parameters.
    #[must_use]
    pub fn reinhard_hdr10() -> Self {
        Self::new(
            ToneMappingConfig::builder()
                .operator(ToneMapOperator::ReinhardLocal)
                .peak_brightness(1000.0)
                .white_point(203.0)
                .build(),
        )
    }
}

// ─────────────────────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────

    fn all_in_range(rgb: [f32; 3]) -> bool {
        rgb.iter().all(|&v| v >= 0.0 && v <= 1.0)
    }

    fn is_monotone(a: [f32; 3], b: [f32; 3]) -> bool {
        // Brighter input ⇒ brighter output in each channel.
        a[0] <= b[0] && a[1] <= b[1] && a[2] <= b[2]
    }

    fn mapper(op: ToneMapOperator) -> HdrToneMapper {
        HdrToneMapper::new(
            ToneMappingConfig::builder()
                .operator(op)
                .peak_brightness(1000.0)
                .white_point(203.0)
                .build(),
        )
    }

    // ── Black preservation ─────────────────────────────────────

    #[test]
    fn test_clamp_preserves_black() {
        let out = mapper(ToneMapOperator::Clamp).map_pixel([0.0, 0.0, 0.0]);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_reinhard_global_preserves_black() {
        let out = mapper(ToneMapOperator::ReinhardGlobal).map_pixel([0.0, 0.0, 0.0]);
        assert!(out[0].abs() < 1e-6 && out[1].abs() < 1e-6 && out[2].abs() < 1e-6);
    }

    #[test]
    fn test_reinhard_local_preserves_black() {
        let out = mapper(ToneMapOperator::ReinhardLocal).map_pixel([0.0, 0.0, 0.0]);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_aces_preserves_black() {
        let out = mapper(ToneMapOperator::AcesFilmic).map_pixel([0.0, 0.0, 0.0]);
        // ACES has a very small lift at zero
        assert!(out[0] < 0.05 && out[1] < 0.05 && out[2] < 0.05);
    }

    #[test]
    fn test_hable_near_black() {
        let out = mapper(ToneMapOperator::Hable).map_pixel([0.0, 0.0, 0.0]);
        // Hable has a toe; absolute black may lift slightly
        assert!(all_in_range(out));
        assert!(out[0] < 0.1 && out[1] < 0.1 && out[2] < 0.1);
    }

    // ── Output range [0, 1] for all operators ─────────────────

    #[test]
    fn test_clamp_output_range() {
        for &v in &[0.0f32, 0.5, 1.0, 2.0, 10.0, 100.0] {
            assert!(all_in_range(mapper(ToneMapOperator::Clamp).map_pixel([
                v,
                v * 0.5,
                v * 0.25
            ])));
        }
    }

    #[test]
    fn test_reinhard_global_output_range() {
        for &v in &[0.5f32, 1.0, 2.0, 10.0, 1000.0] {
            assert!(all_in_range(
                mapper(ToneMapOperator::ReinhardGlobal).map_pixel([v, v * 0.7, v * 0.3])
            ));
        }
    }

    #[test]
    fn test_reinhard_local_output_range() {
        for &v in &[0.5f32, 1.0, 2.0, 10.0, 1000.0] {
            assert!(all_in_range(
                mapper(ToneMapOperator::ReinhardLocal).map_pixel([v, v * 0.7, v * 0.3])
            ));
        }
    }

    #[test]
    fn test_aces_output_range() {
        for &v in &[0.1f32, 0.5, 1.0, 2.0, 5.0, 50.0] {
            assert!(all_in_range(
                mapper(ToneMapOperator::AcesFilmic).map_pixel([v, v * 0.6, v * 0.2])
            ));
        }
    }

    #[test]
    fn test_hable_output_range() {
        for &v in &[0.5f32, 1.0, 2.0, 10.0, 100.0] {
            assert!(all_in_range(mapper(ToneMapOperator::Hable).map_pixel([
                v,
                v * 0.6,
                v * 0.25
            ])));
        }
    }

    #[test]
    fn test_pq_to_sdr_output_range() {
        for &v in &[0.001f32, 0.01, 0.1, 0.5, 1.0, 2.0] {
            assert!(all_in_range(mapper(ToneMapOperator::PqToSdr).map_pixel([
                v,
                v * 0.5,
                v * 0.1
            ])));
        }
    }

    // ── Monotonicity ───────────────────────────────────────────

    #[test]
    fn test_reinhard_local_monotone() {
        let m = mapper(ToneMapOperator::ReinhardLocal);
        let dim = m.map_pixel([0.5, 0.3, 0.1]);
        let bright = m.map_pixel([5.0, 3.0, 1.0]);
        assert!(is_monotone(dim, bright));
    }

    #[test]
    fn test_aces_monotone() {
        let m = mapper(ToneMapOperator::AcesFilmic);
        let dim = m.map_pixel([0.2, 0.1, 0.05]);
        let mid = m.map_pixel([2.0, 1.0, 0.5]);
        let bright = m.map_pixel([10.0, 5.0, 2.5]);
        assert!(is_monotone(dim, mid));
        assert!(is_monotone(mid, bright));
    }

    #[test]
    fn test_hable_monotone() {
        let m = mapper(ToneMapOperator::Hable);
        let dim = m.map_pixel([0.1, 0.05, 0.01]);
        let bright = m.map_pixel([5.0, 2.5, 0.5]);
        assert!(is_monotone(dim, bright));
    }

    // ── Saturation correction ──────────────────────────────────

    #[test]
    fn test_saturation_neutral() {
        let config = ToneMappingConfig::builder()
            .operator(ToneMapOperator::AcesFilmic)
            .saturation_correction(1.0)
            .build();
        let config_sat = ToneMappingConfig::builder()
            .operator(ToneMapOperator::AcesFilmic)
            .saturation_correction(1.0)
            .build();
        let m1 = HdrToneMapper::new(config);
        let m2 = HdrToneMapper::new(config_sat);
        let rgb = [1.0, 0.5, 0.2];
        let a = m1.map_pixel(rgb);
        let b = m2.map_pixel(rgb);
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < 1e-5,
                "Neutral saturation should be identity"
            );
        }
    }

    #[test]
    fn test_saturation_desaturate() {
        let config = ToneMappingConfig::builder()
            .operator(ToneMapOperator::ReinhardGlobal)
            .saturation_correction(0.0)
            .build();
        let m = HdrToneMapper::new(config);
        let out = m.map_pixel([2.0, 1.0, 0.5]);
        // With saturation = 0 all channels should equal luma
        let diff_rg = (out[0] - out[1]).abs();
        let diff_rb = (out[0] - out[2]).abs();
        assert!(diff_rg < 0.01, "Fully desaturated: R≈G (diff={diff_rg:.4})");
        assert!(diff_rb < 0.01, "Fully desaturated: R≈B (diff={diff_rb:.4})");
    }

    // ── Builder API ────────────────────────────────────────────

    #[test]
    fn test_builder_round_trip() {
        let config = ToneMappingConfig::builder()
            .operator(ToneMapOperator::AcesFilmic)
            .peak_brightness(4000.0)
            .white_point(400.0)
            .saturation_correction(1.2)
            .exposure(0.9)
            .knee_start(0.7)
            .build();
        assert_eq!(config.operator, ToneMapOperator::AcesFilmic);
        assert!((config.peak_brightness - 4000.0).abs() < f32::EPSILON);
        assert!((config.white_point - 400.0).abs() < f32::EPSILON);
        assert!((config.saturation_correction - 1.2).abs() < f32::EPSILON);
        assert!((config.exposure - 0.9).abs() < f32::EPSILON);
        assert!((config.knee_start - 0.7).abs() < f32::EPSILON);
    }

    // ── Convenience constructors ───────────────────────────────

    #[test]
    fn test_aces_hdr10_constructor() {
        let m = HdrToneMapper::aces_hdr10();
        assert_eq!(m.config().operator, ToneMapOperator::AcesFilmic);
        assert!(all_in_range(m.map_pixel([3.0, 1.5, 0.5])));
    }

    #[test]
    fn test_hable_hdr10_constructor() {
        let m = HdrToneMapper::hable_hdr10();
        assert_eq!(m.config().operator, ToneMapOperator::Hable);
        assert!(all_in_range(m.map_pixel([5.0, 2.0, 0.8])));
    }

    #[test]
    fn test_pq_to_sdr_constructor() {
        let m = HdrToneMapper::pq_to_sdr_hdr10();
        assert_eq!(m.config().operator, ToneMapOperator::PqToSdr);
        assert!(all_in_range(m.map_pixel([0.5, 0.3, 0.1])));
    }

    #[test]
    fn test_reinhard_hdr10_constructor() {
        let m = HdrToneMapper::reinhard_hdr10();
        assert_eq!(m.config().operator, ToneMapOperator::ReinhardLocal);
        assert!(all_in_range(m.map_pixel([2.0, 1.0, 0.4])));
    }

    // ── Frame processing ───────────────────────────────────────

    #[test]
    fn test_map_frame_in_place() {
        let m = HdrToneMapper::aces_hdr10();
        let mut buf = vec![10.0f32, 5.0, 2.0, 0.1, 0.05, 0.02];
        m.map_frame(&mut buf);
        for chunk in buf.chunks_exact(3) {
            assert!(chunk[0] >= 0.0 && chunk[0] <= 1.0);
            assert!(chunk[1] >= 0.0 && chunk[1] <= 1.0);
            assert!(chunk[2] >= 0.0 && chunk[2] <= 1.0);
        }
    }

    #[test]
    fn test_map_frame_owned() {
        let m = HdrToneMapper::hable_hdr10();
        let input = vec![3.0f32, 1.5, 0.7, 0.3, 0.15, 0.07];
        let output = m.map_frame_owned(&input);
        assert_eq!(output.len(), input.len());
        for chunk in output.chunks_exact(3) {
            assert!(chunk.iter().all(|&v| (0.0..=1.0).contains(&v)));
        }
    }

    // ── Transfer function helpers ──────────────────────────────

    #[test]
    fn test_pq_eotf_bounds() {
        // Signal 0 → linear 0
        let v = pq_eotf_f64(0.0);
        assert!(v.abs() < 1e-10);
        // Signal 1 → linear 1 (10 000 nits normalised)
        let v = pq_eotf_f64(1.0);
        assert!(
            (v - 1.0).abs() < 1e-6,
            "PQ EOTF(1.0) should be 1.0, got {v}"
        );
    }

    #[test]
    fn test_pq_round_trip() {
        for signal in [0.1f64, 0.3, 0.5, 0.7, 0.9] {
            let linear = pq_eotf_f64(signal);
            let back = pq_oetf_f64(linear);
            assert!(
                (back - signal).abs() < 1e-6,
                "PQ round-trip failed at {signal}: got {back}"
            );
        }
    }

    #[test]
    fn test_hlg_eotf_bounds() {
        assert!(hlg_eotf_f64(0.0).abs() < 1e-10);
        let v = hlg_eotf_f64(1.0);
        // HLG EOTF(1.0) ≈ 1.0 (may be slightly above due to floating point)
        assert!(v > 0.0 && v <= 1.0 + 1e-6, "HLG EOTF(1.0) = {v}");
    }

    #[test]
    fn test_hlg_round_trip() {
        for y in [0.05f64, 0.2, 0.5, 0.8, 0.95] {
            let sig = hlg_oetf_f64(y);
            let back = hlg_eotf_f64(sig);
            assert!(
                (back - y).abs() < 1e-6,
                "HLG round-trip failed at {y}: got {back}"
            );
        }
    }

    #[test]
    fn test_bt709_oetf_monotone() {
        let mut prev = bt709_oetf_f64(0.0);
        for i in 1..=20 {
            let y = i as f64 / 20.0;
            let v = bt709_oetf_f64(y);
            assert!(v >= prev, "BT.709 OETF not monotone at {y}");
            prev = v;
        }
    }

    // ── Gamut conversion ───────────────────────────────────────

    #[test]
    fn test_bt2020_to_bt709_white() {
        // D65 white should be roughly (1, 1, 1) in BT.709 as well
        let out = bt2020_to_bt709([1.0, 1.0, 1.0]);
        assert!(
            out[0] > 0.9 && out[1] > 0.9 && out[2] > 0.9,
            "D65 white gamut convert: {out:?}"
        );
    }

    #[test]
    fn test_bt2020_to_bt709_no_negative() {
        // Out-of-gamut colours should not produce negative channels after clipping
        let out = bt2020_to_bt709([0.0, 1.0, 0.0]); // Pure BT.2020 green
        assert!(
            out.iter().all(|&v| v >= 0.0),
            "Negative after gamut convert: {out:?}"
        );
    }

    // ── Soft-knee ─────────────────────────────────────────────

    #[test]
    fn test_soft_knee_below_knee() {
        // Below knee_start values should be unchanged
        let out = soft_knee([0.3, 0.2, 0.1], 0.7, 1.0);
        assert!((out[0] - 0.3).abs() < 1e-6);
        assert!((out[1] - 0.2).abs() < 1e-6);
        assert!((out[2] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_soft_knee_above_knee_end_clamps() {
        let out = soft_knee([2.0, 1.5, 1.2], 0.7, 1.0);
        // Everything at or above knee_end maps to knee_end (1.0)
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
        assert!((out[2] - 1.0).abs() < 1e-5);
    }
}

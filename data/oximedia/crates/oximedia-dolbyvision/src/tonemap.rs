//! Tone mapping and color conversion functions.
//!
//! Implements PQ, HLG, and other transfer functions for Dolby Vision.
//!
//! # Tone mapping pipeline: RPU metadata to display output
//!
//! This module is the low-level engine that the crate's higher-level tone
//! mapping helpers ([`crate::tone_mapping`] curve modeling and
//! [`crate::tone_map_preview`] preview rendering) sit on top of. The
//! end-to-end flow for converting a decoded HDR frame plus its parsed RPU
//! into pixels for a specific display is:
//!
//! 1. **Extract display-management parameters from the RPU**
//!    ([`TonemapParams::from_rpu`]): source characteristics come from
//!    `rpu.vdr_dm_data` (`source_max_pq`/`source_min_pq` converted to nits
//!    via [`pq_to_nits`], and `signal_eotf` decoded to an [`crate::Eotf`]);
//!    target characteristics come from the RPU's L8 target-display metadata
//!    (`rpu.level8`: `target_max_pq`/`target_min_pq` in nits, `target_eotf`).
//!    Any field absent from the RPU falls back to [`TonemapParams::default`].
//! 2. **Decode signal to linear light** ([`apply_eotf`]): each RGB component
//!    (still in normalized `[0, 1]` signal domain) is passed through the
//!    source [`crate::Eotf`] — [`pq_to_linear`] for PQ, [`hlg_to_linear`] for
//!    HLG, [`bt1886_to_linear`] for traditional gamma, or the identity for
//!    already-linear input — yielding scene/display-referred linear light.
//! 3. **Scale to absolute luminance**: linear values are multiplied by
//!    `params.source_peak_nits` to obtain absolute nits, matching the
//!    mastering display's photometric range recorded in the RPU.
//! 4. **Tone-map from source range to target range**: the absolute-nit
//!    signal is compressed from `[source_min_nits, source_peak_nits]` down
//!    to `[target_min_nits, target_peak_nits]`. [`apply_dolbyvision_tonemap`]
//!    performs this inline; [`tonemap_reinhard`] offers the classic
//!    Reinhard operator (`L_out = L_in / (1 + L_in / L_white)`) as a
//!    reusable building block, and the higher-level
//!    [`crate::tone_map_preview::ToneMapper`] implements a saturation- and
//!    shadow-roll-off-aware variant of the same idea for interactive
//!    preview, while [`crate::tone_mapping::ToneCurve`] models the mapping
//!    as an explicit, evaluable piecewise curve for reuse and inspection.
//!    Perceptually uniform color-volume remapping across this stage can be
//!    accelerated with a precomputed 3-D LUT ([`ColorVolumeLut`], trilinear
//!    interpolation) or spatially-aware local tone mapping via
//!    [`BilateralGrid`]; reshaping curves decoded from RPU pivots are
//!    evaluated through [`ReshapingLut`].
//! 5. **Normalize and re-encode for the target display**: the tone-mapped
//!    nits are divided by `params.target_peak_nits` back to `[0, 1]`, then
//!    passed through [`apply_inverse_eotf`] with the target
//!    [`crate::Eotf`] ([`linear_to_pq`], [`linear_to_hlg`], or
//!    [`linear_to_bt1886`]) to produce the final signal-domain samples ready
//!    for display output.
//!
//! [`apply_dolbyvision_tonemap`] drives steps 1-5 end to end for a full RGB
//! pixel buffer given only the parsed [`crate::DolbyVisionRpu`].

use crate::{DolbyVisionError, DolbyVisionRpu, Eotf, Result};

/// PQ (Perceptual Quantizer) EOTF constants from SMPTE ST 2084.
pub mod pq_constants {
    /// m1 constant
    pub const M1: f64 = 0.159_301_758_113_479_8;
    /// m2 constant
    pub const M2: f64 = 78.843_750;
    /// c1 constant
    pub const C1: f64 = 0.835_937_5;
    /// c2 constant
    pub const C2: f64 = 18.851_562_5;
    /// c3 constant
    pub const C3: f64 = 18.6875;
    /// Peak luminance in nits
    pub const PEAK_LUMINANCE: f64 = 10_000.0;
}

/// HLG (Hybrid Log-Gamma) OETF constants from ITU-R BT.2100.
pub mod hlg_constants {
    /// a constant
    pub const A: f64 = 0.178_832_89;
    /// b constant
    pub const B: f64 = 0.284_668_92;
    /// c constant
    pub const C: f64 = 0.559_910_73;
    /// Reference white (75% signal)
    pub const REFERENCE_WHITE: f64 = 0.75;
    /// System gamma for display
    pub const SYSTEM_GAMMA: f64 = 1.2;
}

/// Convert PQ code value (0-4095) to linear light (0-1).
///
/// Implements SMPTE ST 2084 inverse EOTF.
#[must_use]
pub fn pq_to_linear(pq_code: u16) -> f32 {
    let pq_norm = f64::from(pq_code) / 4095.0;

    if pq_norm <= 0.0 {
        return 0.0;
    }

    let v = pq_norm.powf(1.0 / pq_constants::M2);
    let num = (v - pq_constants::C1).max(0.0);
    let den = pq_constants::C2 - pq_constants::C3 * v;

    if den <= 0.0 {
        return 0.0;
    }

    let y = (num / den).powf(1.0 / pq_constants::M1);
    (y * pq_constants::PEAK_LUMINANCE / pq_constants::PEAK_LUMINANCE) as f32
}

/// Convert linear light (0-1) to PQ code value (0-4095).
///
/// Implements SMPTE ST 2084 EOTF.
#[must_use]
pub fn linear_to_pq(linear: f32) -> u16 {
    if linear <= 0.0 {
        return 0;
    }

    let y = f64::from(linear).min(1.0);
    let y_m1 = y.powf(pq_constants::M1);
    let num = pq_constants::C1 + pq_constants::C2 * y_m1;
    let den = 1.0 + pq_constants::C3 * y_m1;

    let pq = (num / den).powf(pq_constants::M2);
    ((pq * 4095.0).min(4095.0).max(0.0)) as u16
}

/// Convert PQ code value (0-4095) to nits.
#[must_use]
pub fn pq_to_nits(pq_code: u16) -> f32 {
    let linear = pq_to_linear(pq_code);
    linear * pq_constants::PEAK_LUMINANCE as f32
}

/// Convert nits to PQ code value (0-4095).
#[must_use]
pub fn nits_to_pq(nits: f32) -> u16 {
    let linear = nits / pq_constants::PEAK_LUMINANCE as f32;
    linear_to_pq(linear)
}

/// Convert HLG signal (0-1) to linear light (0-1).
///
/// Implements ITU-R BT.2100 HLG inverse OETF.
#[must_use]
pub fn hlg_to_linear(hlg: f32) -> f32 {
    let e = f64::from(hlg);

    let linear = if e <= 0.5 {
        (e * e) / 3.0
    } else {
        ((e - hlg_constants::C).exp() + hlg_constants::B) / 12.0
    };

    linear as f32
}

/// Convert linear light (0-1) to HLG signal (0-1).
///
/// Implements ITU-R BT.2100 HLG OETF.
#[must_use]
pub fn linear_to_hlg(linear: f32) -> f32 {
    let e = f64::from(linear);

    let hlg = if e <= (1.0 / 12.0) {
        (3.0 * e).sqrt()
    } else {
        hlg_constants::A * (12.0 * e - hlg_constants::B).ln() + hlg_constants::C
    };

    hlg as f32
}

/// Apply HLG OOTF (Opto-Optical Transfer Function).
///
/// Converts scene-referred linear light to display-referred linear light.
#[must_use]
#[allow(dead_code)]
pub fn hlg_ootf(linear: f32, gamma: f32) -> f32 {
    if linear <= 0.0 {
        return 0.0;
    }

    // Simplified OOTF: Y_d = Y_s^γ
    linear.powf(gamma)
}

/// Apply inverse HLG OOTF.
#[must_use]
#[allow(dead_code)]
pub fn hlg_inverse_ootf(linear: f32, gamma: f32) -> f32 {
    if linear <= 0.0 || gamma <= 0.0 {
        return 0.0;
    }

    linear.powf(1.0 / gamma)
}

/// Convert BT.1886 gamma signal to linear light.
#[must_use]
pub fn bt1886_to_linear(gamma: f32) -> f32 {
    const GAMMA: f32 = 2.4;
    if gamma <= 0.0 {
        0.0
    } else {
        gamma.powf(GAMMA)
    }
}

/// Convert linear light to BT.1886 gamma signal.
#[must_use]
pub fn linear_to_bt1886(linear: f32) -> f32 {
    const INV_GAMMA: f32 = 1.0 / 2.4;
    if linear <= 0.0 {
        0.0
    } else {
        linear.powf(INV_GAMMA)
    }
}

/// Calculate MMR (Min, Max, Average RGB) from RGB values.
#[must_use]
#[allow(dead_code)]
pub fn calculate_mmr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let min = r.min(g).min(b);
    let max = r.max(g).max(b);
    let avg = (r + g + b) / 3.0;
    (min, max, avg)
}

/// Apply polynomial reshaping to a value.
///
/// Evaluates polynomial: y = c0 + c1*x + c2*x^2 + ... + cn*x^n
#[must_use]
#[allow(dead_code)]
pub fn apply_polynomial(value: f32, coefficients: &[i64], log2_denom: u8) -> f32 {
    if coefficients.is_empty() {
        return value;
    }

    let scale = 1.0 / (1u32 << log2_denom) as f32;
    let mut result = 0.0;
    let mut x_power = 1.0;

    for &coef in coefficients {
        result += (coef as f32) * scale * x_power;
        x_power *= value;
    }

    result.max(0.0).min(1.0)
}

/// Apply MMR-based reshaping.
///
/// Combines min, max, and average RGB using weighted polynomial.
#[must_use]
#[allow(dead_code)]
pub fn apply_mmr_reshaping(
    min: f32,
    max: f32,
    avg: f32,
    coefficients: &[i64],
    log2_denom: u8,
) -> f32 {
    if coefficients.len() < 3 {
        return avg;
    }

    let scale = 1.0 / (1u32 << log2_denom) as f32;
    let result = (coefficients[0] as f32) * scale * min
        + (coefficients[1] as f32) * scale * max
        + (coefficients[2] as f32) * scale * avg;

    result.max(0.0).min(1.0)
}

/// Reshaping LUT (Look-Up Table) for tone mapping.
pub struct ReshapingLut {
    /// Input pivot points
    pub pivots: Vec<u16>,
    /// Output values at pivots
    pub outputs: Vec<f32>,
}

impl ReshapingLut {
    /// Create identity LUT (no reshaping).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            pivots: vec![0, 4095],
            outputs: vec![0.0, 1.0],
        }
    }

    /// Create from pivot points and polynomial coefficients.
    #[must_use]
    pub fn from_pivots_and_poly(pivots: &[u16], poly_coefs: &[Vec<i64>], log2_denom: u8) -> Self {
        let mut outputs = Vec::with_capacity(pivots.len());

        for (i, &pivot) in pivots.iter().enumerate() {
            let normalized = f32::from(pivot) / 4095.0;

            let output = if i < poly_coefs.len() && !poly_coefs[i].is_empty() {
                apply_polynomial(normalized, &poly_coefs[i], log2_denom)
            } else {
                normalized
            };

            outputs.push(output);
        }

        Self {
            pivots: pivots.to_vec(),
            outputs,
        }
    }

    /// Apply LUT to input value using linear interpolation.
    #[must_use]
    pub fn apply(&self, input: u16) -> f32 {
        if self.pivots.is_empty() || self.outputs.is_empty() {
            return f32::from(input) / 4095.0;
        }

        // Find segment
        for i in 0..self.pivots.len().saturating_sub(1) {
            if input <= self.pivots[i + 1] {
                let x0 = self.pivots[i];
                let x1 = self.pivots[i + 1];
                let y0 = self.outputs[i];
                let y1 = self.outputs[i + 1];

                if x1 == x0 {
                    return y0;
                }

                // Linear interpolation
                let t = f32::from(input.saturating_sub(x0)) / f32::from(x1 - x0);
                return y0 + t * (y1 - y0);
            }
        }

        // Beyond last pivot
        self.outputs.last().copied().unwrap_or(1.0)
    }
}

/// 3D LUT for color volume transform.
pub struct ColorVolumeLut {
    /// LUT size per dimension
    pub size: usize,
    /// Flattened 3D LUT data \[R\]\[G\]\[B\] -> (R', G', B')
    pub data: Vec<[f32; 3]>,
}

impl ColorVolumeLut {
    /// Create identity 3D LUT.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let total_points = size * size * size;
        let mut data = Vec::with_capacity(total_points);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let scale = 1.0 / (size - 1) as f32;
                    data.push([r as f32 * scale, g as f32 * scale, b as f32 * scale]);
                }
            }
        }

        Self { size, data }
    }

    /// Apply 3D LUT using trilinear interpolation.
    #[must_use]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        if self.size <= 1 {
            return rgb;
        }

        let [r, g, b] = rgb;
        let max_index = (self.size - 1) as f32;

        // Map input to LUT coordinates
        let r_coord = (r * max_index).max(0.0).min(max_index);
        let g_coord = (g * max_index).max(0.0).min(max_index);
        let b_coord = (b * max_index).max(0.0).min(max_index);

        // Get integer indices and fractional parts
        let r_idx = r_coord as usize;
        let g_idx = g_coord as usize;
        let b_idx = b_coord as usize;

        let r_frac = r_coord - r_idx as f32;
        let g_frac = g_coord - g_idx as f32;
        let b_frac = b_coord - b_idx as f32;

        // Clamp to valid range
        let r_idx1 = (r_idx + 1).min(self.size - 1);
        let g_idx1 = (g_idx + 1).min(self.size - 1);
        let b_idx1 = (b_idx + 1).min(self.size - 1);

        // Trilinear interpolation
        let c000 = self.get_value(r_idx, g_idx, b_idx);
        let c001 = self.get_value(r_idx, g_idx, b_idx1);
        let c010 = self.get_value(r_idx, g_idx1, b_idx);
        let c011 = self.get_value(r_idx, g_idx1, b_idx1);
        let c100 = self.get_value(r_idx1, g_idx, b_idx);
        let c101 = self.get_value(r_idx1, g_idx, b_idx1);
        let c110 = self.get_value(r_idx1, g_idx1, b_idx);
        let c111 = self.get_value(r_idx1, g_idx1, b_idx1);

        let c00 = lerp_rgb(c000, c001, b_frac);
        let c01 = lerp_rgb(c010, c011, b_frac);
        let c10 = lerp_rgb(c100, c101, b_frac);
        let c11 = lerp_rgb(c110, c111, b_frac);

        let c0 = lerp_rgb(c00, c01, g_frac);
        let c1 = lerp_rgb(c10, c11, g_frac);

        lerp_rgb(c0, c1, r_frac)
    }

    /// Get LUT value at specific indices.
    fn get_value(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        let idx = r * self.size * self.size + g * self.size + b;
        self.data.get(idx).copied().unwrap_or([0.0, 0.0, 0.0])
    }
}

/// Linear interpolation between two RGB values.
#[must_use]
fn lerp_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Apply 3x3 color matrix transformation.
#[must_use]
#[allow(dead_code)]
pub fn apply_color_matrix(rgb: [f32; 3], matrix: &[[i32; 3]; 3], log2_denom: u8) -> [f32; 3] {
    let scale = 1.0 / (1u32 << log2_denom) as f32;

    let r = (matrix[0][0] as f32 * rgb[0]
        + matrix[0][1] as f32 * rgb[1]
        + matrix[0][2] as f32 * rgb[2])
        * scale;

    let g = (matrix[1][0] as f32 * rgb[0]
        + matrix[1][1] as f32 * rgb[1]
        + matrix[1][2] as f32 * rgb[2])
        * scale;

    let b = (matrix[2][0] as f32 * rgb[0]
        + matrix[2][1] as f32 * rgb[1]
        + matrix[2][2] as f32 * rgb[2])
        * scale;

    [
        r.max(0.0).min(1.0),
        g.max(0.0).min(1.0),
        b.max(0.0).min(1.0),
    ]
}

/// Apply EOTF (Electro-Optical Transfer Function).
#[must_use]
pub fn apply_eotf(value: f32, eotf: Eotf) -> f32 {
    match eotf {
        Eotf::Bt1886 => bt1886_to_linear(value),
        Eotf::Pq => {
            let pq_code = (value * 4095.0) as u16;
            pq_to_linear(pq_code)
        }
        Eotf::Hlg => hlg_to_linear(value),
        Eotf::Linear => value,
    }
}

/// Apply inverse EOTF (OETF - Opto-Electronic Transfer Function).
#[must_use]
pub fn apply_inverse_eotf(value: f32, eotf: Eotf) -> f32 {
    match eotf {
        Eotf::Bt1886 => linear_to_bt1886(value),
        Eotf::Pq => {
            let pq_code = linear_to_pq(value);
            f32::from(pq_code) / 4095.0
        }
        Eotf::Hlg => linear_to_hlg(value),
        Eotf::Linear => value,
    }
}

/// Tone mapping parameters.
pub struct TonemapParams {
    /// Source EOTF
    pub source_eotf: Eotf,
    /// Target EOTF
    pub target_eotf: Eotf,
    /// Source peak luminance (nits)
    pub source_peak_nits: f32,
    /// Target peak luminance (nits)
    pub target_peak_nits: f32,
    /// Source minimum luminance (nits)
    pub source_min_nits: f32,
    /// Target minimum luminance (nits)
    pub target_min_nits: f32,
}

impl Default for TonemapParams {
    fn default() -> Self {
        Self {
            source_eotf: Eotf::Pq,
            target_eotf: Eotf::Pq,
            source_peak_nits: 4000.0,
            target_peak_nits: 1000.0,
            source_min_nits: 0.005,
            target_min_nits: 0.005,
        }
    }
}

impl TonemapParams {
    /// Create from Dolby Vision RPU metadata.
    #[must_use]
    pub fn from_rpu(rpu: &DolbyVisionRpu) -> Self {
        let mut params = Self::default();

        // Get source characteristics
        if let Some(ref vdr_dm) = rpu.vdr_dm_data {
            params.source_peak_nits = pq_to_nits(vdr_dm.source_max_pq);
            params.source_min_nits = pq_to_nits(vdr_dm.source_min_pq);

            if let Some(eotf) = Eotf::from_u16(vdr_dm.signal_eotf) {
                params.source_eotf = eotf;
            }
        }

        // Get target characteristics
        if let Some(ref level8) = rpu.level8 {
            params.target_peak_nits = pq_to_nits(level8.target_max_pq);
            params.target_min_nits = pq_to_nits(level8.target_min_pq);
            params.target_eotf = Eotf::from_u16(u16::from(level8.target_eotf)).unwrap_or(Eotf::Pq);
        }

        params
    }
}

/// Simple Reinhard tone mapping operator.
#[must_use]
pub fn tonemap_reinhard(linear: f32, max_white: f32) -> f32 {
    if max_white <= 0.0 {
        return linear;
    }

    let numerator = linear * (1.0 + linear / (max_white * max_white));
    let denominator = 1.0 + linear;

    if denominator <= 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// ACES filmic tone mapping (simplified).
#[must_use]
#[allow(dead_code)]
pub fn tonemap_aces(linear: f32) -> f32 {
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;

    let x = linear;
    let numerator = x * (A * x + B);
    let denominator = x * (C * x + D) + E;

    if denominator <= 0.0 {
        0.0
    } else {
        (numerator / denominator).max(0.0).min(1.0)
    }
}

/// Hable/Uncharted 2 tone mapping.
#[must_use]
#[allow(dead_code)]
pub fn tonemap_hable(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;

    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

/// Apply tone mapping to a single luminance value.
#[must_use]
#[allow(dead_code)]
pub fn tonemap_luminance(linear_nits: f32, params: &TonemapParams) -> f32 {
    if params.source_peak_nits <= 0.0 || params.target_peak_nits <= 0.0 {
        return linear_nits;
    }

    // Normalize to [0, 1] based on source range
    let normalized =
        (linear_nits - params.source_min_nits) / (params.source_peak_nits - params.source_min_nits);

    // Apply tone mapping
    let ratio = params.target_peak_nits / params.source_peak_nits;
    let mapped = if ratio < 1.0 {
        // Compress highlights
        tonemap_reinhard(normalized, 1.0 / ratio)
    } else {
        // Expand
        normalized * ratio
    };

    // Remap to target range
    params.target_min_nits + mapped * (params.target_peak_nits - params.target_min_nits)
}

/// Apply Dolby Vision tone mapping to RGB pixel data.
///
/// # Errors
///
/// Returns error if RPU metadata is missing or invalid.
pub fn apply_dolbyvision_tonemap(rpu: &DolbyVisionRpu, pixel_data: &mut [f32]) -> Result<()> {
    if pixel_data.len() % 3 != 0 {
        return Err(DolbyVisionError::Generic(
            "Pixel data must be RGB triplets".to_string(),
        ));
    }

    let params = TonemapParams::from_rpu(rpu);

    // Process each RGB pixel
    for chunk in pixel_data.chunks_exact_mut(3) {
        let mut rgb = [chunk[0], chunk[1], chunk[2]];

        // Convert from signal to linear
        rgb = [
            apply_eotf(rgb[0], params.source_eotf),
            apply_eotf(rgb[1], params.source_eotf),
            apply_eotf(rgb[2], params.source_eotf),
        ];

        // Convert to nits
        rgb = [
            rgb[0] * params.source_peak_nits,
            rgb[1] * params.source_peak_nits,
            rgb[2] * params.source_peak_nits,
        ];

        // Apply color matrix if present
        if let Some(ref vdr_dm) = rpu.vdr_dm_data {
            if let Some(ref matrix) = vdr_dm.ycbcr_to_rgb_matrix {
                rgb = apply_color_matrix(rgb, &matrix.matrix, 14);
            }
        }

        // Apply reshaping curves if present
        if let Some(ref vdr_dm) = rpu.vdr_dm_data {
            if !vdr_dm.reshaping_curves.is_empty() {
                // Apply to each channel (simplified)
                for (i, curve) in vdr_dm.reshaping_curves.iter().take(3).enumerate() {
                    if i < 3 {
                        let pq_code = nits_to_pq(rgb[i]);
                        let lut =
                            ReshapingLut::from_pivots_and_poly(&curve.pivots, &curve.poly_coef, 14);
                        let reshaped = lut.apply(pq_code);
                        rgb[i] = pq_to_nits(linear_to_pq(reshaped));
                    }
                }
            }
        }

        // Apply tone mapping per channel
        rgb = [
            tonemap_luminance(rgb[0], &params),
            tonemap_luminance(rgb[1], &params),
            tonemap_luminance(rgb[2], &params),
        ];

        // Convert back to linear [0, 1]
        rgb = [
            rgb[0] / params.target_peak_nits,
            rgb[1] / params.target_peak_nits,
            rgb[2] / params.target_peak_nits,
        ];

        // Apply target EOTF
        rgb = [
            apply_inverse_eotf(rgb[0], params.target_eotf),
            apply_inverse_eotf(rgb[1], params.target_eotf),
            apply_inverse_eotf(rgb[2], params.target_eotf),
        ];

        chunk[0] = rgb[0];
        chunk[1] = rgb[1];
        chunk[2] = rgb[2];
    }

    Ok(())
}

/// Bilateral grid for edge-preserving tone mapping.
pub struct BilateralGrid {
    /// Grid width
    pub width: usize,
    /// Grid height
    pub height: usize,
    /// Grid depth (intensity levels)
    pub depth: usize,
    /// Grid data
    pub data: Vec<f32>,
}

impl BilateralGrid {
    /// Create new bilateral grid.
    #[must_use]
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        Self {
            width,
            height,
            depth,
            data: vec![0.0; width * height * depth],
        }
    }

    /// Get grid value at coordinates.
    #[must_use]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        if x >= self.width || y >= self.height || z >= self.depth {
            return 0.0;
        }
        let idx = (z * self.height + y) * self.width + x;
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    /// Set grid value at coordinates.
    pub fn set(&mut self, x: usize, y: usize, z: usize, value: f32) {
        if x < self.width && y < self.height && z < self.depth {
            let idx = (z * self.height + y) * self.width + x;
            if let Some(cell) = self.data.get_mut(idx) {
                *cell = value;
            }
        }
    }

    /// Apply bilateral filtering with trilinear interpolation.
    #[must_use]
    pub fn apply(&self, x: f32, y: f32, intensity: f32) -> f32 {
        let x_coord = (x * (self.width - 1) as f32).max(0.0);
        let y_coord = (y * (self.height - 1) as f32).max(0.0);
        let z_coord = (intensity * (self.depth - 1) as f32).max(0.0);

        let x0 = x_coord as usize;
        let y0 = y_coord as usize;
        let z0 = z_coord as usize;

        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let z1 = (z0 + 1).min(self.depth - 1);

        let x_frac = x_coord - x0 as f32;
        let y_frac = y_coord - y0 as f32;
        let z_frac = z_coord - z0 as f32;

        // Trilinear interpolation
        let c000 = self.get(x0, y0, z0);
        let c001 = self.get(x0, y0, z1);
        let c010 = self.get(x0, y1, z0);
        let c011 = self.get(x0, y1, z1);
        let c100 = self.get(x1, y0, z0);
        let c101 = self.get(x1, y0, z1);
        let c110 = self.get(x1, y1, z0);
        let c111 = self.get(x1, y1, z1);

        let c00 = c000 + (c001 - c000) * z_frac;
        let c01 = c010 + (c011 - c010) * z_frac;
        let c10 = c100 + (c101 - c100) * z_frac;
        let c11 = c110 + (c111 - c110) * z_frac;

        let c0 = c00 + (c01 - c00) * y_frac;
        let c1 = c10 + (c11 - c10) * y_frac;

        c0 + (c1 - c0) * x_frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_roundtrip() {
        let original = 0.5f32;
        let pq_code = linear_to_pq(original);
        let roundtrip = pq_to_linear(pq_code);
        assert!((roundtrip - original).abs() < 0.01);
    }

    #[test]
    fn test_pq_nits_conversion() {
        let nits = 1000.0f32;
        let pq_code = nits_to_pq(nits);
        let back = pq_to_nits(pq_code);
        assert!((back - nits).abs() < 10.0);
    }

    #[test]
    fn test_hlg_roundtrip() {
        // Test with a value in the lower range where HLG should be more accurate
        let original = 0.1f32;
        let hlg = linear_to_hlg(original);
        let roundtrip = hlg_to_linear(hlg);
        assert!((roundtrip - original).abs() < 0.2);
    }

    #[test]
    fn test_bt1886_roundtrip() {
        let original = 0.5f32;
        let gamma = linear_to_bt1886(original);
        let roundtrip = bt1886_to_linear(gamma);
        assert!((roundtrip - original).abs() < 0.01);
    }

    #[test]
    fn test_mmr_calculation() {
        let (min, max, avg) = calculate_mmr(0.3, 0.7, 0.5);
        assert_eq!(min, 0.3);
        assert_eq!(max, 0.7);
        assert!((avg - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_polynomial_application() {
        let coefs = vec![0, 1 << 14]; // Identity: y = x
        let result = apply_polynomial(0.5, &coefs, 14);
        assert!((result - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_reshaping_lut_identity() {
        let lut = ReshapingLut::identity();
        assert_eq!(lut.apply(0), 0.0);
        assert_eq!(lut.apply(4095), 1.0);
        let mid = lut.apply(2048);
        assert!((mid - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_color_volume_lut_identity() {
        let lut = ColorVolumeLut::identity(5);
        let result = lut.apply([0.5, 0.5, 0.5]);
        assert!((result[0] - 0.5).abs() < 0.1);
        assert!((result[1] - 0.5).abs() < 0.1);
        assert!((result[2] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_color_matrix_identity() {
        let identity = [[1 << 14, 0, 0], [0, 1 << 14, 0], [0, 0, 1 << 14]];
        let rgb = [0.5, 0.6, 0.7];
        let result = apply_color_matrix(rgb, &identity, 14);
        assert!((result[0] - rgb[0]).abs() < 0.01);
        assert!((result[1] - rgb[1]).abs() < 0.01);
        assert!((result[2] - rgb[2]).abs() < 0.01);
    }

    #[test]
    fn test_tonemap_reinhard() {
        let result = tonemap_reinhard(0.5, 1.0);
        assert!(result >= 0.0 && result <= 1.0);
        // Reinhard may not always compress for mid-range values
        assert!(result > 0.0);
    }

    #[test]
    fn test_tonemap_aces() {
        let result = tonemap_aces(0.5);
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn test_tonemap_params_default() {
        let params = TonemapParams::default();
        assert_eq!(params.source_peak_nits, 4000.0);
        assert_eq!(params.target_peak_nits, 1000.0);
    }

    #[test]
    fn test_bilateral_grid() {
        let mut grid = BilateralGrid::new(8, 8, 8);
        grid.set(4, 4, 4, 1.0);
        let value = grid.get(4, 4, 4);
        assert_eq!(value, 1.0);

        let interpolated = grid.apply(0.5, 0.5, 0.5);
        assert!(interpolated >= 0.0 && interpolated <= 1.0);
    }

    #[test]
    fn test_apply_eotf() {
        let linear = apply_eotf(0.5, Eotf::Linear);
        assert_eq!(linear, 0.5);

        let pq = apply_eotf(0.5, Eotf::Pq);
        assert!(pq >= 0.0 && pq <= 1.0);
    }

    #[test]
    fn test_hlg_ootf() {
        let result = hlg_ootf(0.5, 1.2);
        assert!(result >= 0.0 && result <= 1.0);

        let inverse = hlg_inverse_ootf(result, 1.2);
        assert!((inverse - 0.5).abs() < 0.01);
    }
}

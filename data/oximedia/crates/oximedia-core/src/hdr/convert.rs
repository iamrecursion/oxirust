//! HDR conversion utilities.
//!
//! This module provides utilities for converting between different HDR formats
//! and performing tone mapping operations.

use super::primaries::{ColorPrimaries, Primaries};
use super::transfer::TransferCharacteristic;

/// Tone mapping mode for HDR to SDR conversion.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::ToneMappingMode;
///
/// let mode = ToneMappingMode::Aces;
/// assert_eq!(mode.name(), "ACES Filmic");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ToneMappingMode {
    /// Reinhard global tone mapping.
    ///
    /// Simple and fast: `L_out` = `L_in` / (1 + `L_in`)
    Reinhard,

    /// Reinhard extended with white point parameter.
    ///
    /// Allows bright values to reach pure white.
    ReinhardExtended,

    /// ACES filmic tone mapping (widely used in film production).
    ///
    /// Industry standard curve based on ACES RRT.
    #[default]
    Aces,

    /// Hable (Uncharted 2) tone mapping.
    ///
    /// Developed for video games, provides filmic look.
    Hable,

    /// Mobius tone mapping.
    ///
    /// Smooth roll-off at the top end.
    Mobius,

    /// Simple linear clipping (no tone mapping).
    Clip,
}

impl ToneMappingMode {
    /// Returns a human-readable name for this tone mapping mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::ToneMappingMode;
    ///
    /// assert_eq!(ToneMappingMode::Aces.name(), "ACES Filmic");
    /// assert_eq!(ToneMappingMode::Hable.name(), "Hable (Uncharted 2)");
    /// ```
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::Reinhard => "Reinhard",
            Self::ReinhardExtended => "Reinhard Extended",
            Self::Aces => "ACES Filmic",
            Self::Hable => "Hable (Uncharted 2)",
            Self::Mobius => "Mobius",
            Self::Clip => "Clip",
        }
    }

    /// Applies the tone mapping curve to linear light.
    ///
    /// Input and output are in linear space [0, inf).
    ///
    /// # Arguments
    ///
    /// * `linear` - Linear light value (normalized to peak)
    /// * `peak` - Peak luminance of input content
    /// * `target` - Target luminance for output
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::ToneMappingMode;
    ///
    /// let mode = ToneMappingMode::Aces;
    /// let result = mode.tonemap(0.5, 1000.0, 100.0);
    /// assert!(result >= 0.0 && result <= 1.0);
    /// ```
    #[must_use]
    pub fn tonemap(&self, linear: f64, peak: f64, target: f64) -> f64 {
        match self {
            Self::Reinhard => reinhard_tonemap(linear, peak, target),
            Self::ReinhardExtended => reinhard_extended_tonemap(linear, peak, target, peak),
            Self::Aces => aces_tonemap(linear),
            Self::Hable => hable_tonemap(linear),
            Self::Mobius => mobius_tonemap(linear, peak, target),
            Self::Clip => linear.clamp(0.0, 1.0),
        }
    }
}

/// Gamut mapping mode for color space conversion.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::GamutMappingMode;
///
/// let mode = GamutMappingMode::Clip;
/// assert_eq!(mode.name(), "Clip");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum GamutMappingMode {
    /// Simple clipping to target gamut.
    #[default]
    Clip,

    /// Desaturate out-of-gamut colors.
    Desaturate,

    /// Perceptual gamut mapping.
    Perceptual,

    /// Relative colorimetric (scale to fit).
    RelativeColorimetric,
}

impl GamutMappingMode {
    /// Returns a human-readable name for this gamut mapping mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::GamutMappingMode;
    ///
    /// assert_eq!(GamutMappingMode::Clip.name(), "Clip");
    /// assert_eq!(GamutMappingMode::Perceptual.name(), "Perceptual");
    /// ```
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::Clip => "Clip",
            Self::Desaturate => "Desaturate",
            Self::Perceptual => "Perceptual",
            Self::RelativeColorimetric => "Relative Colorimetric",
        }
    }
}

/// HDR to SDR converter.
///
/// Converts HDR content to SDR using specified tone mapping and
/// gamut mapping modes.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::{HdrToSdrConverter, ToneMappingMode, GamutMappingMode};
/// use oximedia_core::hdr::{TransferCharacteristic, ColorPrimaries};
///
/// let converter = HdrToSdrConverter::new(
///     TransferCharacteristic::Pq,
///     TransferCharacteristic::Bt709,
///     ColorPrimaries::BT2020,
///     ColorPrimaries::BT709,
///     1000.0,
///     100.0,
/// );
///
/// let (r_out, g_out, b_out) = converter.convert_pixel(0.5, 0.5, 0.5);
/// assert!(r_out >= 0.0 && r_out <= 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct HdrToSdrConverter {
    source_transfer: TransferCharacteristic,
    target_transfer: TransferCharacteristic,
    #[allow(dead_code)]
    source_primaries: ColorPrimaries,
    #[allow(dead_code)]
    target_primaries: ColorPrimaries,
    peak_luminance: f64,
    target_luminance: f64,
    tonemap_mode: ToneMappingMode,
    #[allow(dead_code)]
    gamut_mode: GamutMappingMode,
    gamut_matrix: ColorMatrix3x3,
}

impl HdrToSdrConverter {
    /// Creates a new HDR to SDR converter.
    ///
    /// # Arguments
    ///
    /// * `source_transfer` - Source transfer characteristic (e.g., PQ)
    /// * `target_transfer` - Target transfer characteristic (e.g., BT.709)
    /// * `source_primaries` - Source color primaries (e.g., BT.2020)
    /// * `target_primaries` - Target color primaries (e.g., BT.709)
    /// * `peak_luminance` - Peak luminance of source content in nits
    /// * `target_luminance` - Target luminance for SDR output in nits
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_transfer: TransferCharacteristic,
        target_transfer: TransferCharacteristic,
        source_primaries: ColorPrimaries,
        target_primaries: ColorPrimaries,
        peak_luminance: f64,
        target_luminance: f64,
    ) -> Self {
        let gamut_matrix =
            ColorMatrix3x3::primaries_conversion(&source_primaries, &target_primaries);

        Self {
            source_transfer,
            target_transfer,
            source_primaries,
            target_primaries,
            peak_luminance,
            target_luminance,
            tonemap_mode: ToneMappingMode::default(),
            gamut_mode: GamutMappingMode::default(),
            gamut_matrix,
        }
    }

    /// Sets the tone mapping mode.
    #[must_use]
    pub fn with_tonemap_mode(mut self, mode: ToneMappingMode) -> Self {
        self.tonemap_mode = mode;
        self
    }

    /// Sets the gamut mapping mode.
    #[must_use]
    pub fn with_gamut_mode(mut self, mode: GamutMappingMode) -> Self {
        self.gamut_mode = mode;
        self
    }

    /// Converts a single pixel from HDR to SDR.
    ///
    /// # Arguments
    ///
    /// * `r`, `g`, `b` - RGB values in [0, 1] range (in source encoding)
    ///
    /// # Returns
    ///
    /// Tuple of (r, g, b) in [0, 1] range (in target encoding)
    #[must_use]
    pub fn convert_pixel(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        // 1. Apply source EOTF to convert to linear light
        let r_lin = self.source_transfer.eotf(r) * self.peak_luminance;
        let g_lin = self.source_transfer.eotf(g) * self.peak_luminance;
        let b_lin = self.source_transfer.eotf(b) * self.peak_luminance;

        // 2. Apply gamut conversion
        let rgb_converted = self.gamut_matrix.apply([r_lin, g_lin, b_lin]);
        let (r_conv, g_conv, b_conv) = (
            rgb_converted[0].max(0.0),
            rgb_converted[1].max(0.0),
            rgb_converted[2].max(0.0),
        );

        // 3. Normalize to [0, 1] range for tone mapping
        let r_norm = r_conv / self.peak_luminance;
        let g_norm = g_conv / self.peak_luminance;
        let b_norm = b_conv / self.peak_luminance;

        // 4. Apply tone mapping
        let r_mapped =
            self.tonemap_mode
                .tonemap(r_norm, self.peak_luminance, self.target_luminance);
        let g_mapped =
            self.tonemap_mode
                .tonemap(g_norm, self.peak_luminance, self.target_luminance);
        let b_mapped =
            self.tonemap_mode
                .tonemap(b_norm, self.peak_luminance, self.target_luminance);

        // 5. Apply target OETF to convert back to signal
        let r_out = self.target_transfer.oetf(r_mapped);
        let g_out = self.target_transfer.oetf(g_mapped);
        let b_out = self.target_transfer.oetf(b_mapped);

        (r_out, g_out, b_out)
    }
}

/// PQ to HLG converter.
///
/// Converts between PQ (ST.2084) and HLG (Hybrid Log-Gamma) transfer functions.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::PqToHlgConverter;
///
/// let converter = PqToHlgConverter::new(1000.0);
/// let hlg_signal = converter.pq_to_hlg(0.5);
/// assert!(hlg_signal >= 0.0 && hlg_signal <= 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct PqToHlgConverter {
    pq_peak_nits: f64,
    hlg_peak_nits: f64,
}

impl PqToHlgConverter {
    /// Creates a new PQ to HLG converter.
    ///
    /// # Arguments
    ///
    /// * `pq_peak_nits` - Peak luminance of PQ content in nits
    #[must_use]
    pub const fn new(pq_peak_nits: f64) -> Self {
        Self {
            pq_peak_nits,
            hlg_peak_nits: 1000.0, // HLG nominal peak
        }
    }

    /// Converts PQ signal to HLG signal.
    ///
    /// # Arguments
    ///
    /// * `pq_signal` - PQ signal value [0, 1]
    ///
    /// # Returns
    ///
    /// HLG signal value [0, 1]
    #[must_use]
    pub fn pq_to_hlg(&self, pq_signal: f64) -> f64 {
        // Convert PQ to linear
        let linear = TransferCharacteristic::Pq.eotf(pq_signal) * self.pq_peak_nits;

        // Normalize to HLG range
        let normalized = linear / self.hlg_peak_nits;

        // Convert to HLG
        TransferCharacteristic::Hlg.oetf(normalized.clamp(0.0, 1.0))
    }

    /// Converts HLG signal to PQ signal.
    ///
    /// # Arguments
    ///
    /// * `hlg_signal` - HLG signal value [0, 1]
    ///
    /// # Returns
    ///
    /// PQ signal value [0, 1]
    #[must_use]
    pub fn hlg_to_pq(&self, hlg_signal: f64) -> f64 {
        // Convert HLG to linear
        let linear = TransferCharacteristic::Hlg.eotf(hlg_signal) * self.hlg_peak_nits;

        // Normalize to PQ range
        let normalized = linear / self.pq_peak_nits;

        // Convert to PQ
        TransferCharacteristic::Pq.oetf(normalized.clamp(0.0, 1.0))
    }
}

/// Color gamut mapper.
///
/// Maps colors from one color gamut to another.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::{ColorGamutMapper, ColorPrimaries, GamutMappingMode};
///
/// let mapper = ColorGamutMapper::new(ColorPrimaries::BT2020, ColorPrimaries::BT709);
/// let (r, g, b) = mapper.map_color(0.5, 0.5, 0.5);
/// assert!(r >= 0.0 && r <= 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct ColorGamutMapper {
    #[allow(dead_code)]
    source_primaries: ColorPrimaries,
    #[allow(dead_code)]
    target_primaries: ColorPrimaries,
    mode: GamutMappingMode,
    matrix: ColorMatrix3x3,
}

impl ColorGamutMapper {
    /// Creates a new color gamut mapper.
    #[must_use]
    pub fn new(source_primaries: ColorPrimaries, target_primaries: ColorPrimaries) -> Self {
        let matrix = ColorMatrix3x3::primaries_conversion(&source_primaries, &target_primaries);

        Self {
            source_primaries,
            target_primaries,
            mode: GamutMappingMode::default(),
            matrix,
        }
    }

    /// Sets the gamut mapping mode.
    #[must_use]
    pub fn with_mode(mut self, mode: GamutMappingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Maps a color from source gamut to target gamut.
    ///
    /// # Arguments
    ///
    /// * `r`, `g`, `b` - RGB values in [0, 1] range (linear light)
    ///
    /// # Returns
    ///
    /// Tuple of (r, g, b) in [0, 1] range (linear light)
    #[must_use]
    pub fn map_color(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        let rgb = self.matrix.apply([r, g, b]);

        match self.mode {
            GamutMappingMode::Clip | GamutMappingMode::RelativeColorimetric => (
                rgb[0].clamp(0.0, 1.0),
                rgb[1].clamp(0.0, 1.0),
                rgb[2].clamp(0.0, 1.0),
            ),
            GamutMappingMode::Desaturate => desaturate_gamut_map(rgb),
            GamutMappingMode::Perceptual => perceptual_gamut_map(rgb),
        }
    }
}

/// 3x3 matrix for color space transformations.
#[derive(Clone, Copy, Debug)]
pub struct ColorMatrix3x3 {
    /// Matrix elements in row-major order.
    pub m: [[f64; 3]; 3],
}

impl ColorMatrix3x3 {
    /// Creates an identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Applies matrix to RGB triplet.
    #[must_use]
    pub fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * rgb[0] + self.m[0][1] * rgb[1] + self.m[0][2] * rgb[2],
            self.m[1][0] * rgb[0] + self.m[1][1] * rgb[1] + self.m[1][2] * rgb[2],
            self.m[2][0] * rgb[0] + self.m[2][1] * rgb[1] + self.m[2][2] * rgb[2],
        ]
    }

    /// Computes matrix for converting from source to destination primaries.
    #[must_use]
    pub fn primaries_conversion(src: &ColorPrimaries, dst: &ColorPrimaries) -> Self {
        let src_primaries = src.primaries();
        let dst_primaries = dst.primaries();

        let src_to_xyz = Self::rgb_to_xyz_matrix(&src_primaries, &src.white_point());
        let xyz_to_dst = Self::xyz_to_rgb_matrix(&dst_primaries, &dst.white_point());

        xyz_to_dst.multiply(&src_to_xyz)
    }

    /// Multiplies two matrices.
    #[must_use]
    fn multiply(&self, other: &Self) -> Self {
        let mut result = Self::identity();

        for i in 0..3 {
            for j in 0..3 {
                result.m[i][j] = 0.0;
                for k in 0..3 {
                    result.m[i][j] += self.m[i][k] * other.m[k][j];
                }
            }
        }

        result
    }

    /// Computes RGB to XYZ conversion matrix.
    #[must_use]
    fn rgb_to_xyz_matrix(
        primaries: &Primaries,
        white_point: &super::primaries::WhitePoint,
    ) -> Self {
        let (xw, yw) = white_point.xy();
        let xr = primaries.red.0;
        let yr = primaries.red.1;
        let zr = 1.0 - xr - yr;

        let xg = primaries.green.0;
        let yg = primaries.green.1;
        let zg = 1.0 - xg - yg;

        let xb = primaries.blue.0;
        let yb = primaries.blue.1;
        let zb = 1.0 - xb - yb;

        let yw_y = 1.0;
        let xw_xyz = (yw_y / yw) * xw;
        let zw_xyz = (yw_y / yw) * (1.0 - xw - yw);

        let det = xr * (yg * zb - yb * zg) - xg * (yr * zb - yb * zr) + xb * (yr * zg - yg * zr);

        if det.abs() < 1e-10 {
            return Self::identity();
        }

        let sr = (xw_xyz * (yg * zb - yb * zg) - xg * (yw_y * zb - zw_xyz * yb)
            + xb * (yw_y * zg - zw_xyz * yg))
            / det;
        let sg = (xr * (yw_y * zb - zw_xyz * yb) - xw_xyz * (yr * zb - yb * zr)
            + xb * (yr * zw_xyz - yw_y * zr))
            / det;
        let sb = (xr * (yg * zw_xyz - yw_y * zg) - xg * (yr * zw_xyz - yw_y * zr)
            + xw_xyz * (yr * zg - yg * zr))
            / det;

        Self {
            m: [
                [sr * xr, sg * xg, sb * xb],
                [sr * yr, sg * yg, sb * yb],
                [sr * zr, sg * zg, sb * zb],
            ],
        }
    }

    /// Computes XYZ to RGB conversion matrix (inverse of RGB to XYZ).
    #[must_use]
    fn xyz_to_rgb_matrix(
        primaries: &Primaries,
        white_point: &super::primaries::WhitePoint,
    ) -> Self {
        let m = Self::rgb_to_xyz_matrix(primaries, white_point);

        let det = m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1])
            - m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0])
            + m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);

        if det.abs() < 1e-10 {
            return Self::identity();
        }

        let inv_det = 1.0 / det;

        Self {
            m: [
                [
                    inv_det * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]),
                    inv_det * (m.m[0][2] * m.m[2][1] - m.m[0][1] * m.m[2][2]),
                    inv_det * (m.m[0][1] * m.m[1][2] - m.m[0][2] * m.m[1][1]),
                ],
                [
                    inv_det * (m.m[1][2] * m.m[2][0] - m.m[1][0] * m.m[2][2]),
                    inv_det * (m.m[0][0] * m.m[2][2] - m.m[0][2] * m.m[2][0]),
                    inv_det * (m.m[0][2] * m.m[1][0] - m.m[0][0] * m.m[1][2]),
                ],
                [
                    inv_det * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]),
                    inv_det * (m.m[0][1] * m.m[2][0] - m.m[0][0] * m.m[2][1]),
                    inv_det * (m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0]),
                ],
            ],
        }
    }
}

/// Reinhard global tone mapping.
#[must_use]
fn reinhard_tonemap(linear: f64, peak: f64, target: f64) -> f64 {
    let normalized = linear;
    let mapped = normalized / (1.0 + normalized);
    mapped * target / peak
}

/// Reinhard extended tone mapping with white point.
#[must_use]
fn reinhard_extended_tonemap(linear: f64, peak: f64, target: f64, white_point: f64) -> f64 {
    let normalized = linear;
    let white = white_point / peak;
    let white_sq = white * white;

    let mapped = (normalized * (1.0 + normalized / white_sq)) / (1.0 + normalized);
    mapped * target / peak
}

/// ACES filmic tone mapping.
#[must_use]
fn aces_tonemap(linear: f64) -> f64 {
    const A: f64 = 2.51;
    const B: f64 = 0.03;
    const C: f64 = 2.43;
    const D: f64 = 0.59;
    const E: f64 = 0.14;

    let x = linear.max(0.0);
    let num = x * (A * x + B);
    let den = x * (C * x + D) + E;

    if den.abs() < 1e-10 {
        0.0
    } else {
        (num / den).clamp(0.0, 1.0)
    }
}

/// Hable (Uncharted 2) tone mapping.
#[must_use]
fn hable_tonemap(linear: f64) -> f64 {
    const EXPOSURE_BIAS: f64 = 2.0;

    fn hable_partial(x: f64) -> f64 {
        const A: f64 = 0.15;
        const B: f64 = 0.50;
        const C: f64 = 0.10;
        const D: f64 = 0.20;
        const E: f64 = 0.02;
        const F: f64 = 0.30;

        ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    }

    let curr = hable_partial(linear * EXPOSURE_BIAS);
    let white = hable_partial(11.2);

    if white.abs() < 1e-10 {
        0.0
    } else {
        (curr / white).clamp(0.0, 1.0)
    }
}

/// Mobius tone mapping.
#[must_use]
fn mobius_tonemap(linear: f64, peak: f64, target: f64) -> f64 {
    let transition = 0.3;
    let normalized = linear / peak;

    if normalized < transition {
        normalized * target / peak
    } else {
        let a = transition;
        let b = 1.0 - transition;
        let mapped = a + b * (normalized - a) / (1.0 + (normalized - a) / b);
        mapped * target / peak
    }
}

/// Desaturate gamut mapping.
#[must_use]
fn desaturate_gamut_map(rgb: [f64; 3]) -> (f64, f64, f64) {
    let max_val = rgb[0].max(rgb[1]).max(rgb[2]);

    if max_val <= 1.0 {
        return (rgb[0], rgb[1], rgb[2]);
    }

    // Desaturate towards white
    let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

    let r_desat = luma + (rgb[0] - luma) / max_val;
    let g_desat = luma + (rgb[1] - luma) / max_val;
    let b_desat = luma + (rgb[2] - luma) / max_val;

    (
        r_desat.clamp(0.0, 1.0),
        g_desat.clamp(0.0, 1.0),
        b_desat.clamp(0.0, 1.0),
    )
}

/// Perceptual gamut mapping.
#[must_use]
fn perceptual_gamut_map(rgb: [f64; 3]) -> (f64, f64, f64) {
    // Simple perceptual mapping: compress high values
    let compress = |x: f64| {
        if x <= 1.0 {
            x
        } else {
            1.0 - (-0.5 * (x - 1.0)).exp()
        }
    };

    (compress(rgb[0]), compress(rgb[1]), compress(rgb[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tonemap_modes() {
        let modes = [
            ToneMappingMode::Reinhard,
            ToneMappingMode::ReinhardExtended,
            ToneMappingMode::Aces,
            ToneMappingMode::Hable,
            ToneMappingMode::Mobius,
            ToneMappingMode::Clip,
        ];

        for mode in &modes {
            let result = mode.tonemap(0.5, 1000.0, 100.0);
            assert!(result >= 0.0 && result <= 1.0);
        }
    }

    #[test]
    fn test_hdr_to_sdr_converter() {
        let converter = HdrToSdrConverter::new(
            TransferCharacteristic::Pq,
            TransferCharacteristic::Bt709,
            ColorPrimaries::BT2020,
            ColorPrimaries::BT709,
            1000.0,
            100.0,
        );

        let (r, g, b) = converter.convert_pixel(0.5, 0.5, 0.5);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn test_pq_to_hlg_converter() {
        let converter = PqToHlgConverter::new(1000.0);

        let hlg_signal = converter.pq_to_hlg(0.5);
        assert!(hlg_signal >= 0.0 && hlg_signal <= 1.0);

        let pq_signal = converter.hlg_to_pq(0.5);
        assert!(pq_signal >= 0.0 && pq_signal <= 1.0);
    }

    #[test]
    fn test_color_gamut_mapper() {
        let mapper = ColorGamutMapper::new(ColorPrimaries::BT2020, ColorPrimaries::BT709);

        let (r, g, b) = mapper.map_color(0.5, 0.5, 0.5);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn test_color_matrix_identity() {
        let matrix = ColorMatrix3x3::identity();
        let result = matrix.apply([0.5, 0.6, 0.7]);
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.6).abs() < 1e-10);
        assert!((result[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_reinhard_tonemap() {
        let result = reinhard_tonemap(0.5, 1000.0, 100.0);
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn test_aces_tonemap() {
        let result = aces_tonemap(0.5);
        assert!(result >= 0.0 && result <= 1.0);
    }
}

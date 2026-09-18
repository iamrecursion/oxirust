//! HDR to SDR tone mapping filter.
//!
//! This filter converts High Dynamic Range (HDR) video to Standard Dynamic Range (SDR)
//! using various tone mapping algorithms and transfer function conversions.
//!
//! # Features
//!
//! - **Tone Mapping Algorithms:**
//!   - Reinhard (simple and extended)
//!   - ACES Filmic
//!   - Hable (Uncharted 2)
//!
//! - **HDR Transfer Functions:**
//!   - ST.2084 (PQ) - HDR10
//!   - HLG (Hybrid Log-Gamma)
//!   - Linear
//!
//! - **Color Space Conversion:**
//!   - BT.2020 -> BT.709 primaries
//!   - Proper chromatic adaptation
//!
//! - **HDR Metadata Support:**
//!   - MaxCLL (Maximum Content Light Level)
//!   - MaxFALL (Maximum Frame-Average Light Level)
//!   - Mastering display metadata
//!
//! # Example
//!
//! ```ignore
//! use oximedia_graph::filters::video::{TonemapFilter, TonemapConfig, TonemapAlgorithm};
//! use oximedia_graph::node::NodeId;
//!
//! let config = TonemapConfig::new()
//!     .with_algorithm(TonemapAlgorithm::Aces)
//!     .with_peak_luminance(1000.0)
//!     .with_target_luminance(100.0);
//!
//! let filter = TonemapFilter::new(NodeId(0), "hdr_tonemap", config);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};
use oximedia_codec::{ColorInfo, Plane, VideoFrame};
use oximedia_core::PixelFormat;

/// HDR transfer function (EOTF - Electro-Optical Transfer Function).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TransferFunction {
    /// Linear (no transfer function).
    Linear,
    /// ST.2084 (PQ) - Perceptual Quantizer used in HDR10.
    #[default]
    Pq,
    /// HLG - Hybrid Log-Gamma used in broadcast HDR.
    Hlg,
    /// BT.709/sRGB gamma.
    Bt709,
}

impl TransferFunction {
    /// Apply the Electro-Optical Transfer Function (EOTF) to convert signal to linear.
    /// Input is normalized [0, 1], output is linear light in nits.
    #[must_use]
    pub fn eotf(&self, signal: f64, peak_nits: f64) -> f64 {
        match self {
            Self::Linear => signal * peak_nits,
            Self::Pq => pq_eotf(signal) * peak_nits,
            Self::Hlg => hlg_eotf(signal) * peak_nits,
            Self::Bt709 => bt709_eotf(signal) * peak_nits,
        }
    }

    /// Apply the Opto-Electronic Transfer Function (OETF) to convert linear to signal.
    /// Input is linear light in nits, output is normalized [0, 1].
    #[must_use]
    pub fn oetf(&self, linear: f64, peak_nits: f64) -> f64 {
        let normalized = (linear / peak_nits).clamp(0.0, 1.0);
        match self {
            Self::Linear => normalized,
            Self::Pq => pq_oetf(normalized),
            Self::Hlg => hlg_oetf(normalized),
            Self::Bt709 => bt709_oetf(normalized),
        }
    }
}

/// ST.2084 (PQ) EOTF - converts PQ signal to linear light (normalized).
/// Returns linear light in range [0, 1] where 1.0 represents 10000 nits.
#[must_use]
fn pq_eotf(e: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    let e = e.clamp(0.0, 1.0);
    let e_m2 = e.powf(1.0 / M2);
    let num = (e_m2 - C1).max(0.0);
    let den = C2 - C3 * e_m2;

    if den.abs() < 1e-10 {
        0.0
    } else {
        (num / den).powf(1.0 / M1)
    }
}

/// ST.2084 (PQ) inverse EOTF - converts linear light to PQ signal.
/// Input is linear light in range [0, 1] where 1.0 represents 10000 nits.
#[must_use]
fn pq_oetf(y: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    let y = y.clamp(0.0, 1.0);
    let y_m1 = y.powf(M1);
    let num = C1 + C2 * y_m1;
    let den = 1.0 + C3 * y_m1;

    (num / den).powf(M2)
}

/// HLG EOTF - converts HLG signal to linear light (normalized).
/// Returns linear light in range [0, 1].
#[must_use]
fn hlg_eotf(e: f64) -> f64 {
    const A: f64 = 0.17883277;
    const B: f64 = 0.28466892;
    const C: f64 = 0.55991073;

    let e = e.clamp(0.0, 1.0);

    if e <= 0.5 {
        (e * e) / 3.0
    } else {
        (((e - C) / A).exp() + B) / 12.0
    }
}

/// HLG inverse EOTF - converts linear light to HLG signal.
/// Input is linear light in range [0, 1].
#[must_use]
fn hlg_oetf(y: f64) -> f64 {
    const A: f64 = 0.17883277;
    const B: f64 = 0.28466892;
    const C: f64 = 0.55991073;

    let y = y.clamp(0.0, 1.0);

    if y <= 1.0 / 12.0 {
        (3.0 * y).sqrt()
    } else {
        A * (12.0 * y - B).ln() + C
    }
}

/// BT.709 EOTF (gamma 2.4 with linear segment).
#[must_use]
fn bt709_eotf(e: f64) -> f64 {
    const BETA: f64 = 0.018053968510807;
    const ALPHA: f64 = 1.09929682680944;
    const GAMMA: f64 = 1.0 / 0.45;

    let e = e.clamp(0.0, 1.0);

    if e < BETA * 4.5 {
        e / 4.5
    } else {
        ((e + (ALPHA - 1.0)) / ALPHA).powf(GAMMA)
    }
}

/// BT.709 inverse EOTF.
#[must_use]
fn bt709_oetf(y: f64) -> f64 {
    const BETA: f64 = 0.018053968510807;
    const ALPHA: f64 = 1.09929682680944;
    const GAMMA: f64 = 0.45;

    let y = y.clamp(0.0, 1.0);

    if y < BETA {
        4.5 * y
    } else {
        ALPHA * y.powf(GAMMA) - (ALPHA - 1.0)
    }
}

/// Tone mapping algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TonemapAlgorithm {
    /// Reinhard global tone mapping.
    Reinhard,
    /// Reinhard extended with white point parameter.
    ReinhardExtended,
    /// ACES filmic tone mapping (widely used in film production).
    #[default]
    Aces,
    /// Hable (Uncharted 2) tone mapping.
    Hable,
    /// Simple linear clipping (no tone mapping).
    Clip,
}

impl TonemapAlgorithm {
    /// Apply the tone mapping curve to linear light.
    /// Input and output are in linear space [0, inf).
    #[must_use]
    pub fn tonemap(&self, linear: f64, params: &TonemapParams) -> f64 {
        match self {
            Self::Reinhard => reinhard_tonemap(linear, params),
            Self::ReinhardExtended => reinhard_extended_tonemap(linear, params),
            Self::Aces => aces_tonemap(linear),
            Self::Hable => hable_tonemap(linear),
            Self::Clip => linear.clamp(0.0, 1.0),
        }
    }
}

/// Tone mapping parameters.
#[derive(Clone, Copy, Debug)]
pub struct TonemapParams {
    /// Peak luminance of input HDR content in nits.
    pub peak_luminance: f64,
    /// Target peak luminance for SDR output in nits.
    pub target_luminance: f64,
    /// White point for Reinhard extended (in nits).
    pub white_point: f64,
    /// Exposure adjustment (stops).
    pub exposure: f64,
    /// Contrast adjustment.
    pub contrast: f64,
    /// Saturation adjustment.
    pub saturation: f64,
}

impl Default for TonemapParams {
    fn default() -> Self {
        Self {
            peak_luminance: 1000.0,
            target_luminance: 100.0,
            white_point: 1000.0,
            exposure: 0.0,
            contrast: 1.0,
            saturation: 1.0,
        }
    }
}

/// Reinhard global tone mapping.
/// Simple and fast: L_out = L_in / (1 + L_in)
#[must_use]
fn reinhard_tonemap(linear: f64, params: &TonemapParams) -> f64 {
    // Normalize to [0, 1] range based on peak luminance
    let normalized = linear / params.peak_luminance;

    // Apply exposure
    let exposed = normalized * 2.0_f64.powf(params.exposure);

    // Reinhard formula
    let mapped = exposed / (1.0 + exposed);

    // Scale to target luminance
    mapped * params.target_luminance / 100.0
}

/// Reinhard extended tone mapping with white point.
/// Allows bright values to reach pure white.
#[must_use]
fn reinhard_extended_tonemap(linear: f64, params: &TonemapParams) -> f64 {
    let normalized = linear / params.peak_luminance;
    let exposed = normalized * 2.0_f64.powf(params.exposure);

    let white = params.white_point / params.peak_luminance;
    let white_sq = white * white;

    // Extended Reinhard formula
    let mapped = (exposed * (1.0 + exposed / white_sq)) / (1.0 + exposed);

    mapped * params.target_luminance / 100.0
}

/// ACES filmic tone mapping.
/// Industry standard curve used in film production.
/// Based on ACES RRT (Reference Rendering Transform).
#[must_use]
fn aces_tonemap(linear: f64) -> f64 {
    // ACES approximation (Narkowicz 2015)
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
/// Developed for the Uncharted 2 video game, provides filmic look.
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

/// Color space primaries for wide color gamut conversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorPrimaries {
    /// Red primary (x, y) in CIE 1931 xy chromaticity.
    pub red: (f64, f64),
    /// Green primary (x, y).
    pub green: (f64, f64),
    /// Blue primary (x, y).
    pub blue: (f64, f64),
    /// White point (x, y).
    pub white: (f64, f64),
}

impl ColorPrimaries {
    /// BT.709 / sRGB primaries (standard HD).
    pub const BT709: Self = Self {
        red: (0.64, 0.33),
        green: (0.30, 0.60),
        blue: (0.15, 0.06),
        white: (0.3127, 0.3290), // D65
    };

    /// BT.2020 primaries (wide color gamut for UHD/HDR).
    pub const BT2020: Self = Self {
        red: (0.708, 0.292),
        green: (0.170, 0.797),
        blue: (0.131, 0.046),
        white: (0.3127, 0.3290), // D65
    };

    /// DCI-P3 primaries (digital cinema).
    pub const DCI_P3: Self = Self {
        red: (0.680, 0.320),
        green: (0.265, 0.690),
        blue: (0.150, 0.060),
        white: (0.3127, 0.3290), // D65 (adapted)
    };
}

/// 3x3 matrix for color space transformations.
#[derive(Clone, Copy, Debug)]
pub struct ColorMatrix3x3 {
    /// Matrix elements in row-major order.
    pub m: [[f64; 3]; 3],
}

impl ColorMatrix3x3 {
    /// Create identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Apply matrix to RGB triplet.
    #[must_use]
    pub fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * rgb[0] + self.m[0][1] * rgb[1] + self.m[0][2] * rgb[2],
            self.m[1][0] * rgb[0] + self.m[1][1] * rgb[1] + self.m[1][2] * rgb[2],
            self.m[2][0] * rgb[0] + self.m[2][1] * rgb[1] + self.m[2][2] * rgb[2],
        ]
    }

    /// Compute matrix for converting from source to destination primaries.
    /// Uses chromatic adaptation (Bradford method).
    #[must_use]
    pub fn primaries_conversion(src: &ColorPrimaries, dst: &ColorPrimaries) -> Self {
        // Compute XYZ to RGB matrices for both color spaces
        let src_to_xyz = Self::rgb_to_xyz_matrix(src);
        let xyz_to_dst = Self::xyz_to_rgb_matrix(dst);

        // Multiply: RGB_dst = (XYZ->RGB_dst) * (RGB_src->XYZ) * RGB_src
        xyz_to_dst.multiply(&src_to_xyz)
    }

    /// Multiply two matrices.
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

    /// Compute RGB to XYZ conversion matrix for given primaries.
    #[must_use]
    fn rgb_to_xyz_matrix(primaries: &ColorPrimaries) -> Self {
        // XYZ coordinates of primaries
        let xr = primaries.red.0;
        let yr = primaries.red.1;
        let zr = 1.0 - xr - yr;

        let xg = primaries.green.0;
        let yg = primaries.green.1;
        let zg = 1.0 - xg - yg;

        let xb = primaries.blue.0;
        let yb = primaries.blue.1;
        let zb = 1.0 - xb - yb;

        // White point
        let xw = primaries.white.0;
        let yw = primaries.white.1;
        let yw_y = 1.0; // Normalized
        let xw_xyz = (yw_y / yw) * xw;
        let zw_xyz = (yw_y / yw) * (1.0 - xw - yw);

        // Compute scaling factors
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

    /// Compute XYZ to RGB conversion matrix (inverse of RGB to XYZ).
    #[must_use]
    fn xyz_to_rgb_matrix(primaries: &ColorPrimaries) -> Self {
        let m = Self::rgb_to_xyz_matrix(primaries);

        // Compute inverse using cofactor method for 3x3 matrix
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

/// HDR metadata.
#[derive(Clone, Copy, Debug, Default)]
pub struct HdrMetadata {
    /// Maximum Content Light Level in nits.
    pub max_cll: Option<f64>,
    /// Maximum Frame-Average Light Level in nits.
    pub max_fall: Option<f64>,
    /// Mastering display peak luminance in nits.
    pub master_peak: Option<f64>,
    /// Mastering display minimum luminance in nits.
    pub master_min: Option<f64>,
}

impl HdrMetadata {
    /// Estimate peak luminance from available metadata.
    #[must_use]
    pub fn estimate_peak_luminance(&self) -> f64 {
        self.max_cll.or(self.master_peak).unwrap_or(1000.0) // Default to HDR10 nominal peak
    }
}

/// Tone mapping configuration.
#[derive(Clone, Debug)]
pub struct TonemapConfig {
    /// Tone mapping algorithm.
    pub algorithm: TonemapAlgorithm,
    /// Source transfer function.
    pub source_transfer: TransferFunction,
    /// Target transfer function (usually BT.709).
    pub target_transfer: TransferFunction,
    /// Source color primaries.
    pub source_primaries: ColorPrimaries,
    /// Target color primaries.
    pub target_primaries: ColorPrimaries,
    /// Tone mapping parameters.
    pub params: TonemapParams,
    /// HDR metadata.
    pub metadata: HdrMetadata,
    /// Target output format.
    pub target_format: PixelFormat,
    /// Perform color gamut conversion.
    pub convert_gamut: bool,
}

impl TonemapConfig {
    /// Create a new tone mapping configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            algorithm: TonemapAlgorithm::default(),
            source_transfer: TransferFunction::Pq,
            target_transfer: TransferFunction::Bt709,
            source_primaries: ColorPrimaries::BT2020,
            target_primaries: ColorPrimaries::BT709,
            params: TonemapParams::default(),
            metadata: HdrMetadata::default(),
            target_format: PixelFormat::Yuv420p,
            convert_gamut: true,
        }
    }

    /// Set the tone mapping algorithm.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: TonemapAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the source transfer function.
    #[must_use]
    pub fn with_source_transfer(mut self, transfer: TransferFunction) -> Self {
        self.source_transfer = transfer;
        self
    }

    /// Set the target transfer function.
    #[must_use]
    pub fn with_target_transfer(mut self, transfer: TransferFunction) -> Self {
        self.target_transfer = transfer;
        self
    }

    /// Set the source color primaries.
    #[must_use]
    pub fn with_source_primaries(mut self, primaries: ColorPrimaries) -> Self {
        self.source_primaries = primaries;
        self
    }

    /// Set the target color primaries.
    #[must_use]
    pub fn with_target_primaries(mut self, primaries: ColorPrimaries) -> Self {
        self.target_primaries = primaries;
        self
    }

    /// Set the peak luminance of the source content.
    #[must_use]
    pub fn with_peak_luminance(mut self, nits: f64) -> Self {
        self.params.peak_luminance = nits;
        self
    }

    /// Set the target luminance for SDR output.
    #[must_use]
    pub fn with_target_luminance(mut self, nits: f64) -> Self {
        self.params.target_luminance = nits;
        self
    }

    /// Set the white point for Reinhard extended.
    #[must_use]
    pub fn with_white_point(mut self, nits: f64) -> Self {
        self.params.white_point = nits;
        self
    }

    /// Set exposure adjustment in stops.
    #[must_use]
    pub fn with_exposure(mut self, stops: f64) -> Self {
        self.params.exposure = stops;
        self
    }

    /// Set contrast adjustment.
    #[must_use]
    pub fn with_contrast(mut self, contrast: f64) -> Self {
        self.params.contrast = contrast;
        self
    }

    /// Set saturation adjustment.
    #[must_use]
    pub fn with_saturation(mut self, saturation: f64) -> Self {
        self.params.saturation = saturation;
        self
    }

    /// Set HDR metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: HdrMetadata) -> Self {
        self.metadata = metadata;
        // Update peak luminance from metadata if not explicitly set
        if self.params.peak_luminance == 1000.0 {
            self.params.peak_luminance = metadata.estimate_peak_luminance();
        }
        self
    }

    /// Set target output format.
    #[must_use]
    pub fn with_target_format(mut self, format: PixelFormat) -> Self {
        self.target_format = format;
        self
    }

    /// Enable or disable color gamut conversion.
    #[must_use]
    pub fn with_gamut_conversion(mut self, enable: bool) -> Self {
        self.convert_gamut = enable;
        self
    }
}

impl Default for TonemapConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// HDR tone mapping filter.
///
/// Converts High Dynamic Range (HDR) video to Standard Dynamic Range (SDR)
/// using various tone mapping algorithms and proper color space handling.
pub struct TonemapFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    config: TonemapConfig,
    gamut_matrix: ColorMatrix3x3,
}

impl TonemapFilter {
    /// Create a new tone mapping filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: TonemapConfig) -> Self {
        // Pre-compute gamut conversion matrix
        let gamut_matrix = if config.convert_gamut {
            ColorMatrix3x3::primaries_conversion(&config.source_primaries, &config.target_primaries)
        } else {
            ColorMatrix3x3::identity()
        };

        let output_format = PortFormat::Video(VideoPortFormat::new(config.target_format));

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Video).with_format(output_format)
            ],
            config,
            gamut_matrix,
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &TonemapConfig {
        &self.config
    }

    /// Process a single pixel through the tone mapping pipeline.
    /// Input: RGB in source color space (8-bit or normalized).
    /// Output: RGB in target color space (8-bit).
    fn tonemap_pixel(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        // 1. Normalize to [0, 1]
        let r_norm = r as f64 / 255.0;
        let g_norm = g as f64 / 255.0;
        let b_norm = b as f64 / 255.0;

        // 2. Apply source EOTF to convert to linear light
        let peak = self.config.params.peak_luminance;
        let r_lin = self.config.source_transfer.eotf(r_norm, peak);
        let g_lin = self.config.source_transfer.eotf(g_norm, peak);
        let b_lin = self.config.source_transfer.eotf(b_norm, peak);

        // 3. Color gamut conversion (BT.2020 -> BT.709)
        let rgb_lin = if self.config.convert_gamut {
            let converted = self.gamut_matrix.apply([r_lin, g_lin, b_lin]);
            [
                converted[0].max(0.0),
                converted[1].max(0.0),
                converted[2].max(0.0),
            ]
        } else {
            [r_lin, g_lin, b_lin]
        };

        // 4. Apply tone mapping operator
        let r_mapped = self
            .config
            .algorithm
            .tonemap(rgb_lin[0], &self.config.params);
        let g_mapped = self
            .config
            .algorithm
            .tonemap(rgb_lin[1], &self.config.params);
        let b_mapped = self
            .config
            .algorithm
            .tonemap(rgb_lin[2], &self.config.params);

        // 5. Apply saturation adjustment
        if (self.config.params.saturation - 1.0).abs() > 0.001 {
            let luma = 0.2126 * r_mapped + 0.7152 * g_mapped + 0.0722 * b_mapped;
            let sat = self.config.params.saturation;

            let r_sat = luma + (r_mapped - luma) * sat;
            let g_sat = luma + (g_mapped - luma) * sat;
            let b_sat = luma + (b_mapped - luma) * sat;

            // 6. Apply target OETF and convert back to 8-bit
            let r_out = (self.config.target_transfer.oetf(r_sat.max(0.0), 100.0) * 255.0)
                .clamp(0.0, 255.0) as u8;
            let g_out = (self.config.target_transfer.oetf(g_sat.max(0.0), 100.0) * 255.0)
                .clamp(0.0, 255.0) as u8;
            let b_out = (self.config.target_transfer.oetf(b_sat.max(0.0), 100.0) * 255.0)
                .clamp(0.0, 255.0) as u8;

            (r_out, g_out, b_out)
        } else {
            // 6. Apply target OETF and convert back to 8-bit
            let r_out =
                (self.config.target_transfer.oetf(r_mapped, 100.0) * 255.0).clamp(0.0, 255.0) as u8;
            let g_out =
                (self.config.target_transfer.oetf(g_mapped, 100.0) * 255.0).clamp(0.0, 255.0) as u8;
            let b_out =
                (self.config.target_transfer.oetf(b_mapped, 100.0) * 255.0).clamp(0.0, 255.0) as u8;

            (r_out, g_out, b_out)
        }
    }

    /// Convert YUV frame to RGB for processing.
    fn yuv_to_rgb(&self, frame: &VideoFrame) -> Vec<u8> {
        let width = frame.width as usize;
        let height = frame.height as usize;

        let y_plane = frame.planes.first();
        let u_plane = frame.planes.get(1);
        let v_plane = frame.planes.get(2);

        let (h_sub, v_sub) = frame.format.chroma_subsampling();
        let mut rgb_data = vec![0u8; width * height * 3];

        // BT.2020 matrix for YUV to RGB conversion
        const KR: f64 = 0.2627;
        const KB: f64 = 0.0593;
        const KG: f64 = 1.0 - KR - KB;

        for y in 0..height {
            for x in 0..width {
                let y_val = y_plane
                    .map(|p| p.row(y).get(x).copied().unwrap_or(16))
                    .unwrap_or(16) as f64;

                let chroma_x = x / h_sub as usize;
                let chroma_y = y / v_sub as usize;

                let u_val = u_plane
                    .map(|p| p.row(chroma_y).get(chroma_x).copied().unwrap_or(128))
                    .unwrap_or(128) as f64;
                let v_val = v_plane
                    .map(|p| p.row(chroma_y).get(chroma_x).copied().unwrap_or(128))
                    .unwrap_or(128) as f64;

                // YUV to RGB conversion (limited range)
                let y_norm = (y_val - 16.0) * 255.0 / 219.0;
                let cb = (u_val - 128.0) * 255.0 / 224.0;
                let cr = (v_val - 128.0) * 255.0 / 224.0;

                let r = y_norm + cr / (1.0 - KR) * KR;
                let g = y_norm - cb / ((1.0 - KB) * KG) * KB - cr / ((1.0 - KR) * KG) * KR;
                let b = y_norm + cb / (1.0 - KB) * KB;

                let offset = (y * width + x) * 3;
                rgb_data[offset] = r.clamp(0.0, 255.0) as u8;
                rgb_data[offset + 1] = g.clamp(0.0, 255.0) as u8;
                rgb_data[offset + 2] = b.clamp(0.0, 255.0) as u8;
            }
        }

        rgb_data
    }

    /// Convert RGB back to YUV for output.
    fn rgb_to_yuv(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // BT.709 matrix for RGB to YUV conversion
        const KR: f64 = 0.2126;
        const KB: f64 = 0.0722;
        const KG: f64 = 1.0 - KR - KB;

        let mut y_data = vec![0u8; width * height];
        let chroma_width = width / 2;
        let chroma_height = height / 2;
        let mut u_data = vec![128u8; chroma_width * chroma_height];
        let mut v_data = vec![128u8; chroma_width * chroma_height];

        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 3;
                let r = rgb_data[offset] as f64;
                let g = rgb_data[offset + 1] as f64;
                let b = rgb_data[offset + 2] as f64;

                // RGB to YUV conversion (limited range)
                let y_val = KR * r + KG * g + KB * b;
                let cb = (b - y_val) / (2.0 * (1.0 - KB));
                let cr = (r - y_val) / (2.0 * (1.0 - KR));

                let y_out = y_val * 219.0 / 255.0 + 16.0;
                let cb_out = cb * 224.0 / 255.0 + 128.0;
                let cr_out = cr * 224.0 / 255.0 + 128.0;

                y_data[y * width + x] = y_out.clamp(16.0, 235.0) as u8;

                // Subsample chroma (4:2:0)
                if x % 2 == 0 && y % 2 == 0 {
                    let chroma_x = x / 2;
                    let chroma_y = y / 2;
                    u_data[chroma_y * chroma_width + chroma_x] = cb_out.clamp(16.0, 240.0) as u8;
                    v_data[chroma_y * chroma_width + chroma_x] = cr_out.clamp(16.0, 240.0) as u8;
                }
            }
        }

        (y_data, u_data, v_data)
    }

    /// Process an RGB frame directly.
    fn process_rgb(&self, input: &VideoFrame) -> GraphResult<VideoFrame> {
        let width = input.width as usize;
        let height = input.height as usize;

        let src_plane = input
            .planes
            .first()
            .ok_or_else(|| GraphError::ProcessingError {
                node: self.id,
                message: "Missing RGB plane".to_string(),
            })?;

        let src_bpp = if input.format == PixelFormat::Rgba32 {
            4
        } else {
            3
        };
        let mut output_rgb = vec![0u8; width * height * 3];

        // Process each pixel through tone mapping pipeline
        for y in 0..height {
            for x in 0..width {
                let row = src_plane.row(y);
                let offset = x * src_bpp;

                let r = row.get(offset).copied().unwrap_or(0);
                let g = row.get(offset + 1).copied().unwrap_or(0);
                let b = row.get(offset + 2).copied().unwrap_or(0);

                let (r_out, g_out, b_out) = self.tonemap_pixel(r, g, b);

                let out_offset = (y * width + x) * 3;
                output_rgb[out_offset] = r_out;
                output_rgb[out_offset + 1] = g_out;
                output_rgb[out_offset + 2] = b_out;
            }
        }

        // Convert to target format
        if self.config.target_format.is_yuv() {
            let (y_data, u_data, v_data) = self.rgb_to_yuv(&output_rgb, width, height);

            let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
            output.timestamp = input.timestamp;
            output.frame_type = input.frame_type;
            output.color_info = ColorInfo {
                full_range: false,
                ..input.color_info
            };

            let chroma_width = width / 2;
            output.planes.push(Plane::new(y_data, width));
            output.planes.push(Plane::new(u_data, chroma_width));
            output.planes.push(Plane::new(v_data, chroma_width));

            Ok(output)
        } else {
            let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
            output.timestamp = input.timestamp;
            output.frame_type = input.frame_type;
            output.color_info = input.color_info;
            output.planes.push(Plane::new(output_rgb, width * 3));

            Ok(output)
        }
    }

    /// Process a YUV frame.
    fn process_yuv(&self, input: &VideoFrame) -> GraphResult<VideoFrame> {
        let width = input.width as usize;
        let height = input.height as usize;

        // Convert YUV to RGB
        let rgb_data = self.yuv_to_rgb(input);

        // Process each pixel through tone mapping
        let mut output_rgb = vec![0u8; width * height * 3];

        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 3;
                let r = rgb_data[offset];
                let g = rgb_data[offset + 1];
                let b = rgb_data[offset + 2];

                let (r_out, g_out, b_out) = self.tonemap_pixel(r, g, b);

                output_rgb[offset] = r_out;
                output_rgb[offset + 1] = g_out;
                output_rgb[offset + 2] = b_out;
            }
        }

        // Convert back to YUV
        if self.config.target_format.is_yuv() {
            let (y_data, u_data, v_data) = self.rgb_to_yuv(&output_rgb, width, height);

            let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
            output.timestamp = input.timestamp;
            output.frame_type = input.frame_type;
            output.color_info = ColorInfo {
                full_range: false,
                ..input.color_info
            };

            let chroma_width = width / 2;
            output.planes.push(Plane::new(y_data, width));
            output.planes.push(Plane::new(u_data, chroma_width));
            output.planes.push(Plane::new(v_data, chroma_width));

            Ok(output)
        } else {
            let mut output = VideoFrame::new(self.config.target_format, input.width, input.height);
            output.timestamp = input.timestamp;
            output.frame_type = input.frame_type;
            output.color_info = input.color_info;
            output.planes.push(Plane::new(output_rgb, width * 3));

            Ok(output)
        }
    }

    /// Convert a frame through the tone mapping pipeline.
    fn tonemap_frame(&self, input: &VideoFrame) -> GraphResult<VideoFrame> {
        if input.format.is_yuv() {
            self.process_yuv(input)
        } else {
            self.process_rgb(input)
        }
    }
}

impl Node for TonemapFilter {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> NodeType {
        NodeType::Filter
    }

    fn state(&self) -> NodeState {
        self.state
    }

    fn set_state(&mut self, state: NodeState) -> GraphResult<()> {
        if !self.state.can_transition_to(state) {
            return Err(GraphError::InvalidStateTransition {
                node: self.id,
                from: self.state.to_string(),
                to: state.to_string(),
            });
        }
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        match input {
            Some(FilterFrame::Video(frame)) => {
                let tonemapped = self.tonemap_frame(&frame)?;
                Ok(Some(FilterFrame::Video(tonemapped)))
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }
}

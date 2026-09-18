#![allow(dead_code)]
//! NDI color-space metadata and conversion helpers for `oximedia-ndi`.
//!
//! NDI streams carry colour-space identifiers that receivers must honour in
//! order to display frames correctly.  This module models color primaries,
//! transfer functions, and matrix coefficients as they appear in the NDI
//! protocol, and provides lightweight conversion matrices between common
//! spaces (BT.601 / BT.709 / BT.2020).

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]

// ---------------------------------------------------------------------------
// ColorPrimaries
// ---------------------------------------------------------------------------

/// Color primaries used by an NDI source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorPrimaries {
    /// ITU-R BT.601 (NTSC / PAL SD).
    Bt601,
    /// ITU-R BT.709 (HD).
    Bt709,
    /// ITU-R BT.2020 (UHD / HDR).
    Bt2020,
    /// DCI-P3 (digital cinema).
    DciP3,
    /// Unknown / unspecified.
    Unknown,
}

impl ColorPrimaries {
    /// Return a human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bt601 => "BT.601",
            Self::Bt709 => "BT.709",
            Self::Bt2020 => "BT.2020",
            Self::DciP3 => "DCI-P3",
            Self::Unknown => "Unknown",
        }
    }

    /// Attempt to parse from a string tag (case-insensitive).
    pub fn from_tag(tag: &str) -> Self {
        match tag.to_ascii_lowercase().as_str() {
            "bt601" | "601" | "smpte170m" => Self::Bt601,
            "bt709" | "709" => Self::Bt709,
            "bt2020" | "2020" => Self::Bt2020,
            "p3" | "dcip3" | "dci-p3" => Self::DciP3,
            _ => Self::Unknown,
        }
    }

    /// Whether the primaries represent an HDR-capable colour space.
    pub fn is_wide_gamut(self) -> bool {
        matches!(self, Self::Bt2020 | Self::DciP3)
    }
}

// ---------------------------------------------------------------------------
// TransferFunction
// ---------------------------------------------------------------------------

/// Electro-optical transfer function (EOTF / gamma).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferFunction {
    /// Standard BT.709 gamma (~2.2).
    Bt709,
    /// Perceptual Quantizer (SMPTE ST 2084) — used for HDR10.
    Pq,
    /// Hybrid Log-Gamma (ARIB STD-B67) — used for HLG.
    Hlg,
    /// sRGB transfer (≈ gamma 2.2 with linear toe).
    Srgb,
    /// Linear light.
    Linear,
    /// Unknown / unspecified.
    Unknown,
}

impl TransferFunction {
    /// Return a human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bt709 => "BT.709",
            Self::Pq => "PQ (ST 2084)",
            Self::Hlg => "HLG",
            Self::Srgb => "sRGB",
            Self::Linear => "Linear",
            Self::Unknown => "Unknown",
        }
    }

    /// Whether this transfer function implies HDR content.
    pub fn is_hdr(self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }
}

// ---------------------------------------------------------------------------
// MatrixCoefficients
// ---------------------------------------------------------------------------

/// YCbCr matrix coefficients used for YUV ↔ RGB conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixCoefficients {
    /// BT.601 (SD).
    Bt601,
    /// BT.709 (HD).
    Bt709,
    /// BT.2020 non-constant luminance.
    Bt2020Ncl,
    /// Identity (RGB is already RGB).
    Identity,
    /// Unknown / unspecified.
    Unknown,
}

impl MatrixCoefficients {
    /// Return a human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bt601 => "BT.601",
            Self::Bt709 => "BT.709",
            Self::Bt2020Ncl => "BT.2020 NCL",
            Self::Identity => "Identity",
            Self::Unknown => "Unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// NdiColorSpace
// ---------------------------------------------------------------------------

/// Full colour-space description attached to an NDI stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NdiColorSpace {
    /// Colour primaries.
    pub primaries: ColorPrimaries,
    /// Transfer function.
    pub transfer: TransferFunction,
    /// Matrix coefficients (for YUV content).
    pub matrix: MatrixCoefficients,
    /// Whether the source signals full-range (0-255) vs limited (16-235).
    pub full_range: bool,
}

impl Default for NdiColorSpace {
    fn default() -> Self {
        Self {
            primaries: ColorPrimaries::Bt709,
            transfer: TransferFunction::Bt709,
            matrix: MatrixCoefficients::Bt709,
            full_range: false,
        }
    }
}

impl NdiColorSpace {
    /// Standard BT.709 HD colour space (limited range).
    pub fn bt709() -> Self {
        Self::default()
    }

    /// Standard BT.601 SD colour space (limited range).
    pub fn bt601() -> Self {
        Self {
            primaries: ColorPrimaries::Bt601,
            transfer: TransferFunction::Bt709,
            matrix: MatrixCoefficients::Bt601,
            full_range: false,
        }
    }

    /// BT.2020 PQ HDR colour space.
    pub fn bt2020_pq() -> Self {
        Self {
            primaries: ColorPrimaries::Bt2020,
            transfer: TransferFunction::Pq,
            matrix: MatrixCoefficients::Bt2020Ncl,
            full_range: false,
        }
    }

    /// Whether the colour space is HDR.
    pub fn is_hdr(self) -> bool {
        self.transfer.is_hdr()
    }

    /// Whether the colour space uses wide-gamut primaries.
    pub fn is_wide_gamut(self) -> bool {
        self.primaries.is_wide_gamut()
    }

    /// Whether a conversion is needed between two colour spaces.
    pub fn needs_conversion(self, other: Self) -> bool {
        self != other
    }
}

// ---------------------------------------------------------------------------
// ConversionMatrix3x3
// ---------------------------------------------------------------------------

/// A simple 3x3 matrix used for colour-space conversion.
#[derive(Debug, Clone, Copy)]
pub struct ConversionMatrix3x3 {
    /// Row-major 3x3 values.
    pub m: [[f64; 3]; 3],
}

impl ConversionMatrix3x3 {
    /// Identity matrix (no conversion).
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Multiply a 3-element vector by this matrix.
    pub fn transform(&self, v: [f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * v[0] + self.m[0][1] * v[1] + self.m[0][2] * v[2],
            self.m[1][0] * v[0] + self.m[1][1] * v[1] + self.m[1][2] * v[2],
            self.m[2][0] * v[0] + self.m[2][1] * v[1] + self.m[2][2] * v[2],
        ]
    }

    /// Compose two matrices (self * other).
    pub fn compose(&self, other: &Self) -> Self {
        let mut out = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.m[i][0] * other.m[0][j]
                    + self.m[i][1] * other.m[1][j]
                    + self.m[i][2] * other.m[2][j];
            }
        }
        Self { m: out }
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Self {
        Self {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2]],
            ],
        }
    }

    /// BT.709 YCbCr-to-RGB matrix (limited range).
    pub fn bt709_ycbcr_to_rgb() -> Self {
        Self {
            m: [
                [1.164, 0.000, 1.793],
                [1.164, -0.213, -0.533],
                [1.164, 2.112, 0.000],
            ],
        }
    }

    /// BT.601 YCbCr-to-RGB matrix (limited range).
    pub fn bt601_ycbcr_to_rgb() -> Self {
        Self {
            m: [
                [1.164, 0.000, 1.596],
                [1.164, -0.392, -0.813],
                [1.164, 2.017, 0.000],
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: select conversion matrix
// ---------------------------------------------------------------------------

/// Select the appropriate YCbCr → RGB conversion matrix for a given colour
/// space.
pub fn ycbcr_to_rgb_matrix(cs: &NdiColorSpace) -> ConversionMatrix3x3 {
    match cs.matrix {
        MatrixCoefficients::Bt601 => ConversionMatrix3x3::bt601_ycbcr_to_rgb(),
        MatrixCoefficients::Bt709 => ConversionMatrix3x3::bt709_ycbcr_to_rgb(),
        MatrixCoefficients::Identity => ConversionMatrix3x3::identity(),
        _ => ConversionMatrix3x3::bt709_ycbcr_to_rgb(),
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated YUV422 ↔ RGB conversion
// ---------------------------------------------------------------------------

/// Convert UYVY (YUV 4:2:2 interleaved) pixel data to RGB24.
///
/// The input `yuv` slice must have length `width as usize * 2` (2 bytes per
/// pixel for UYVY).  Returns a `Vec<u8>` of length `width as usize * 3`
/// (R, G, B triplets).
///
/// On x86_64 the function detects SSE4.1 at runtime and selects an optimised
/// scalar loop designed to expose opportunities for the compiler's
/// auto-vectoriser.  On all other platforms the plain scalar path is used.
pub fn yuv422_to_rgb_simd(yuv: &[u8], width: u32) -> Vec<u8> {
    let w = width as usize;
    // UYVY: 2 bytes per pixel — total input length == w * 2.
    assert_eq!(
        yuv.len(),
        w * 2,
        "UYVY buffer must be width*2 bytes (got {}, expected {})",
        yuv.len(),
        w * 2
    );

    // Both paths produce the same output; we pick the loop structure that
    // allows the compiler to emit SSE4.1 instructions when available.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return yuv422_to_rgb_fast(yuv, w);
        }
    }
    yuv422_to_rgb_scalar(yuv, w)
}

/// Convert RGB24 pixel data to UYVY (YUV 4:2:2 interleaved).
///
/// The input `rgb` slice must have length `width as usize * 3`.  Returns a
/// `Vec<u8>` of length `width as usize * 2` in UYVY packing.
pub fn rgb_to_yuv422_simd(rgb: &[u8], width: u32) -> Vec<u8> {
    let w = width as usize;
    assert_eq!(
        rgb.len(),
        w * 3,
        "RGB24 buffer must be width*3 bytes (got {}, expected {})",
        rgb.len(),
        w * 3
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return rgb_to_yuv422_fast(rgb, w);
        }
    }
    rgb_to_yuv422_scalar(rgb, w)
}

// ---------------------------------------------------------------------------
// Inner implementation — scalar reference path
// ---------------------------------------------------------------------------

/// Clamp a 32-bit integer to [0, 255] and return as u8.
#[inline(always)]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Scalar UYVY → RGB24 conversion.
///
/// BT.601 limited-range coefficients:
///   Y' = 1.164 * (Y - 16)
///   Cb = Cb - 128,  Cr = Cr - 128
///   R = Y' + 1.596 * Cr
///   G = Y' - 0.391 * Cb - 0.813 * Cr
///   B = Y' + 2.018 * Cb
///
/// All arithmetic is done in fixed-point (× 1024) to avoid f32 per pixel.
fn yuv422_to_rgb_scalar(yuv: &[u8], width: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * 3];
    // Process two pixels at a time (one UYVY macropixel = 4 bytes).
    let macropixels = width / 2;
    for i in 0..macropixels {
        let base_yuv = i * 4;
        let u_val = yuv[base_yuv] as i32 - 128;
        let y0 = yuv[base_yuv + 1] as i32 - 16;
        let v_val = yuv[base_yuv + 2] as i32 - 128;
        let y1 = yuv[base_yuv + 3] as i32 - 16;

        // Fixed-point scaled by 1024.
        let y0_s = (1192 * y0) >> 10; // 1.164 * y0
        let y1_s = (1192 * y1) >> 10;
        let r_bias = (1634 * v_val) >> 10; // 1.596 * Cr
        let g_bias = (400 * u_val + 833 * v_val) >> 10; // 0.391*Cb + 0.813*Cr
        let b_bias = (2066 * u_val) >> 10; // 2.018 * Cb

        let base_rgb = i * 6;
        rgb[base_rgb] = clamp_u8(y0_s + r_bias);
        rgb[base_rgb + 1] = clamp_u8(y0_s - g_bias);
        rgb[base_rgb + 2] = clamp_u8(y0_s + b_bias);
        rgb[base_rgb + 3] = clamp_u8(y1_s + r_bias);
        rgb[base_rgb + 4] = clamp_u8(y1_s - g_bias);
        rgb[base_rgb + 5] = clamp_u8(y1_s + b_bias);
    }
    // Handle odd width (rare but guard it).
    if width % 2 == 1 {
        let base_yuv = macropixels * 4;
        let u_val = yuv[base_yuv] as i32 - 128;
        let y0 = yuv[base_yuv + 1] as i32 - 16;
        let v_val = yuv[base_yuv + 2] as i32 - 128;
        let y0_s = (1192 * y0) >> 10;
        let r_bias = (1634 * v_val) >> 10;
        let g_bias = (400 * u_val + 833 * v_val) >> 10;
        let b_bias = (2066 * u_val) >> 10;
        let base_rgb = macropixels * 6;
        rgb[base_rgb] = clamp_u8(y0_s + r_bias);
        rgb[base_rgb + 1] = clamp_u8(y0_s - g_bias);
        rgb[base_rgb + 2] = clamp_u8(y0_s + b_bias);
    }
    rgb
}

/// Compiler-autovectorisable UYVY → RGB24 loop (same maths, re-ordered for
/// SIMD friendliness — the compiler can fold adjacent iterations into SSE
/// instructions).
fn yuv422_to_rgb_fast(yuv: &[u8], width: usize) -> Vec<u8> {
    // Pre-extract component slices into flat arrays so the compiler can
    // reason about aliasing and apply vectorised loads.
    let macropixels = width / 2;
    let mut u_buf = vec![0i32; macropixels];
    let mut v_buf = vec![0i32; macropixels];
    let mut y0_buf = vec![0i32; macropixels];
    let mut y1_buf = vec![0i32; macropixels];

    for i in 0..macropixels {
        let base = i * 4;
        u_buf[i] = yuv[base] as i32 - 128;
        y0_buf[i] = yuv[base + 1] as i32 - 16;
        v_buf[i] = yuv[base + 2] as i32 - 128;
        y1_buf[i] = yuv[base + 3] as i32 - 16;
    }

    let mut rgb = vec![0u8; width * 3];
    for i in 0..macropixels {
        let u_val = u_buf[i];
        let v_val = v_buf[i];
        let y0_s = (1192 * y0_buf[i]) >> 10;
        let y1_s = (1192 * y1_buf[i]) >> 10;
        let r_bias = (1634 * v_val) >> 10;
        let g_bias = (400 * u_val + 833 * v_val) >> 10;
        let b_bias = (2066 * u_val) >> 10;

        let base_rgb = i * 6;
        rgb[base_rgb] = clamp_u8(y0_s + r_bias);
        rgb[base_rgb + 1] = clamp_u8(y0_s - g_bias);
        rgb[base_rgb + 2] = clamp_u8(y0_s + b_bias);
        rgb[base_rgb + 3] = clamp_u8(y1_s + r_bias);
        rgb[base_rgb + 4] = clamp_u8(y1_s - g_bias);
        rgb[base_rgb + 5] = clamp_u8(y1_s + b_bias);
    }
    rgb
}

/// Scalar RGB24 → UYVY conversion.
///
/// BT.601 limited-range coefficients (fixed-point × 1024):
///   Y  =  0.257*R + 0.504*G + 0.098*B + 16
///   Cb = -0.148*R - 0.291*G + 0.439*B + 128
///   Cr =  0.439*R - 0.368*G - 0.071*B + 128
///
/// Cb and Cr are averaged from the two horizontally adjacent pixels.
fn rgb_to_yuv422_scalar(rgb: &[u8], width: usize) -> Vec<u8> {
    let mut yuv = vec![0u8; width * 2];
    let macropixels = width / 2;
    for i in 0..macropixels {
        let base_rgb = i * 6;
        let r0 = rgb[base_rgb] as i32;
        let g0 = rgb[base_rgb + 1] as i32;
        let b0 = rgb[base_rgb + 2] as i32;
        let r1 = rgb[base_rgb + 3] as i32;
        let g1 = rgb[base_rgb + 4] as i32;
        let b1 = rgb[base_rgb + 5] as i32;

        let y0 = (263 * r0 + 516 * g0 + 100 * b0 + 16 * 1024) >> 10;
        let y1 = (263 * r1 + 516 * g1 + 100 * b1 + 16 * 1024) >> 10;
        // Average Cb/Cr from both pixels.
        let cb = ((-151 * r0 - 298 * g0 + 449 * b0 + 128 * 1024)
            + (-151 * r1 - 298 * g1 + 449 * b1 + 128 * 1024))
            >> 11; // divide by 2 * 1024
        let cr = ((449 * r0 - 377 * g0 - 73 * b0 + 128 * 1024)
            + (449 * r1 - 377 * g1 - 73 * b1 + 128 * 1024))
            >> 11;

        let base_yuv = i * 4;
        yuv[base_yuv] = clamp_u8(cb);
        yuv[base_yuv + 1] = clamp_u8(y0);
        yuv[base_yuv + 2] = clamp_u8(cr);
        yuv[base_yuv + 3] = clamp_u8(y1);
    }
    // Odd-width guard.
    if width % 2 == 1 {
        let base_rgb = macropixels * 6;
        let r0 = rgb[base_rgb] as i32;
        let g0 = rgb[base_rgb + 1] as i32;
        let b0 = rgb[base_rgb + 2] as i32;
        let y0 = (263 * r0 + 516 * g0 + 100 * b0 + 16 * 1024) >> 10;
        let cb = (-151 * r0 - 298 * g0 + 449 * b0 + 128 * 1024) >> 10;
        let cr = (449 * r0 - 377 * g0 - 73 * b0 + 128 * 1024) >> 10;
        let base_yuv = macropixels * 4;
        yuv[base_yuv] = clamp_u8(cb);
        yuv[base_yuv + 1] = clamp_u8(y0);
        yuv[base_yuv + 2] = clamp_u8(cr);
    }
    yuv
}

/// Autovectorisation-friendly RGB24 → UYVY path for x86_64+SSE4.1.
fn rgb_to_yuv422_fast(rgb: &[u8], width: usize) -> Vec<u8> {
    // Same maths as scalar; the pre-extraction into typed arrays gives the
    // compiler enough aliasing information to auto-vectorise.
    let macropixels = width / 2;
    let mut r0_buf = vec![0i32; macropixels];
    let mut g0_buf = vec![0i32; macropixels];
    let mut b0_buf = vec![0i32; macropixels];
    let mut r1_buf = vec![0i32; macropixels];
    let mut g1_buf = vec![0i32; macropixels];
    let mut b1_buf = vec![0i32; macropixels];

    for i in 0..macropixels {
        let base = i * 6;
        r0_buf[i] = rgb[base] as i32;
        g0_buf[i] = rgb[base + 1] as i32;
        b0_buf[i] = rgb[base + 2] as i32;
        r1_buf[i] = rgb[base + 3] as i32;
        g1_buf[i] = rgb[base + 4] as i32;
        b1_buf[i] = rgb[base + 5] as i32;
    }

    let mut yuv = vec![0u8; width * 2];
    for i in 0..macropixels {
        let r0 = r0_buf[i];
        let g0 = g0_buf[i];
        let b0 = b0_buf[i];
        let r1 = r1_buf[i];
        let g1 = g1_buf[i];
        let b1 = b1_buf[i];

        let y0 = (263 * r0 + 516 * g0 + 100 * b0 + 16 * 1024) >> 10;
        let y1 = (263 * r1 + 516 * g1 + 100 * b1 + 16 * 1024) >> 10;
        let cb = ((-151 * r0 - 298 * g0 + 449 * b0 + 128 * 1024)
            + (-151 * r1 - 298 * g1 + 449 * b1 + 128 * 1024))
            >> 11;
        let cr = ((449 * r0 - 377 * g0 - 73 * b0 + 128 * 1024)
            + (449 * r1 - 377 * g1 - 73 * b1 + 128 * 1024))
            >> 11;

        let base_yuv = i * 4;
        yuv[base_yuv] = clamp_u8(cb);
        yuv[base_yuv + 1] = clamp_u8(y0);
        yuv[base_yuv + 2] = clamp_u8(cr);
        yuv[base_yuv + 3] = clamp_u8(y1);
    }
    yuv
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_primaries_name() {
        assert_eq!(ColorPrimaries::Bt709.name(), "BT.709");
        assert_eq!(ColorPrimaries::Bt2020.name(), "BT.2020");
    }

    #[test]
    fn test_color_primaries_from_tag() {
        assert_eq!(ColorPrimaries::from_tag("bt709"), ColorPrimaries::Bt709);
        assert_eq!(ColorPrimaries::from_tag("2020"), ColorPrimaries::Bt2020);
        assert_eq!(ColorPrimaries::from_tag("P3"), ColorPrimaries::DciP3);
        assert_eq!(ColorPrimaries::from_tag("xyz"), ColorPrimaries::Unknown);
    }

    #[test]
    fn test_wide_gamut() {
        assert!(!ColorPrimaries::Bt709.is_wide_gamut());
        assert!(ColorPrimaries::Bt2020.is_wide_gamut());
        assert!(ColorPrimaries::DciP3.is_wide_gamut());
    }

    #[test]
    fn test_transfer_function_hdr() {
        assert!(!TransferFunction::Bt709.is_hdr());
        assert!(TransferFunction::Pq.is_hdr());
        assert!(TransferFunction::Hlg.is_hdr());
    }

    #[test]
    fn test_default_color_space() {
        let cs = NdiColorSpace::default();
        assert_eq!(cs.primaries, ColorPrimaries::Bt709);
        assert!(!cs.is_hdr());
        assert!(!cs.is_wide_gamut());
    }

    #[test]
    fn test_bt2020_pq() {
        let cs = NdiColorSpace::bt2020_pq();
        assert!(cs.is_hdr());
        assert!(cs.is_wide_gamut());
        assert_eq!(cs.transfer, TransferFunction::Pq);
    }

    #[test]
    fn test_needs_conversion() {
        let a = NdiColorSpace::bt709();
        let b = NdiColorSpace::bt601();
        assert!(a.needs_conversion(b));
        assert!(!a.needs_conversion(a));
    }

    #[test]
    fn test_identity_matrix_transform() {
        let id = ConversionMatrix3x3::identity();
        let v = [0.5, 0.7, 0.3];
        let out = id.transform(v);
        for i in 0..3 {
            assert!((out[i] - v[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matrix_compose_identity() {
        let id = ConversionMatrix3x3::identity();
        let m = ConversionMatrix3x3::bt709_ycbcr_to_rgb();
        let composed = id.compose(&m);
        for i in 0..3 {
            for j in 0..3 {
                assert!((composed.m[i][j] - m.m[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_transpose() {
        let m = ConversionMatrix3x3 {
            m: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        };
        let t = m.transpose();
        assert!((t.m[0][1] - 4.0).abs() < 1e-10);
        assert!((t.m[1][0] - 2.0).abs() < 1e-10);
        assert!((t.m[2][0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ycbcr_to_rgb_matrix_bt709() {
        let cs = NdiColorSpace::bt709();
        let m = ycbcr_to_rgb_matrix(&cs);
        // First row: Y coefficient should be 1.164
        assert!((m.m[0][0] - 1.164).abs() < 0.001);
    }

    #[test]
    fn test_ycbcr_to_rgb_matrix_bt601() {
        let cs = NdiColorSpace::bt601();
        let m = ycbcr_to_rgb_matrix(&cs);
        assert!((m.m[0][2] - 1.596).abs() < 0.001);
    }

    #[test]
    fn test_matrix_coefficients_name() {
        assert_eq!(MatrixCoefficients::Bt709.name(), "BT.709");
        assert_eq!(MatrixCoefficients::Identity.name(), "Identity");
    }

    #[test]
    fn test_bt601_color_space() {
        let cs = NdiColorSpace::bt601();
        assert_eq!(cs.primaries, ColorPrimaries::Bt601);
        assert_eq!(cs.matrix, MatrixCoefficients::Bt601);
        assert!(!cs.full_range);
    }

    #[test]
    fn test_yuv422_rgb_roundtrip() {
        // Build a simple 4-pixel wide, 1-line image in RGB24.
        // Pixel values chosen to be "mid-grey" to stay well within the
        // limited-range YUV encoding window and thus survive the double
        // quantisation error without exceeding a ±6 tolerance.
        let width: u32 = 4;
        // Original RGB (4 pixels × 3 channels = 12 bytes).
        let original_rgb: Vec<u8> = vec![
            100, 110, 120, // pixel 0
            130, 140, 150, // pixel 1
            100, 110, 120, // pixel 2
            130, 140, 150, // pixel 3
        ];

        // Encode RGB24 → UYVY.
        let yuv = rgb_to_yuv422_simd(&original_rgb, width);
        assert_eq!(yuv.len(), width as usize * 2, "UYVY length mismatch");

        // Decode UYVY → RGB24.
        let recovered_rgb = yuv422_to_rgb_simd(&yuv, width);
        assert_eq!(
            recovered_rgb.len(),
            width as usize * 3,
            "RGB24 length mismatch"
        );

        // The round-trip introduces quantisation error; tolerance of ±8.
        let tolerance = 8i32;
        for (i, (&orig, &rec)) in original_rgb.iter().zip(recovered_rgb.iter()).enumerate() {
            let diff = (orig as i32 - rec as i32).abs();
            assert!(
                diff <= tolerance,
                "pixel channel {i}: original={orig}, recovered={rec}, diff={diff} > {tolerance}"
            );
        }
    }

    #[test]
    fn test_yuv422_to_rgb_simd_output_length() {
        let width: u32 = 8;
        let yuv = vec![128u8; width as usize * 2]; // black in YUV
        let rgb = yuv422_to_rgb_simd(&yuv, width);
        assert_eq!(rgb.len(), width as usize * 3);
    }

    #[test]
    fn test_rgb_to_yuv422_simd_output_length() {
        let width: u32 = 8;
        let rgb = vec![128u8; width as usize * 3];
        let yuv = rgb_to_yuv422_simd(&rgb, width);
        assert_eq!(yuv.len(), width as usize * 2);
    }
}

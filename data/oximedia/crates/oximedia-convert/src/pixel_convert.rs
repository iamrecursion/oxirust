#![allow(dead_code)]
//! Pixel format conversion utilities.
//!
//! This module handles conversions between common pixel formats used in
//! video processing pipelines: RGB, RGBA, YUV (BT.601 / BT.709 / BT.2020),
//! NV12, planar 4:2:0, 4:2:2, 4:4:4, and 10-bit variants. All conversions
//! are done in pure Rust with no external dependencies.

/// Supported pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit RGB interleaved (3 bytes per pixel).
    Rgb8,
    /// 8-bit RGBA interleaved (4 bytes per pixel).
    Rgba8,
    /// 8-bit BGR interleaved (3 bytes per pixel).
    Bgr8,
    /// 8-bit BGRA interleaved (4 bytes per pixel).
    Bgra8,
    /// Planar YUV 4:2:0 (I420).
    Yuv420p,
    /// Semi-planar YUV 4:2:0 (NV12).
    Nv12,
    /// Planar YUV 4:2:2.
    Yuv422p,
    /// Planar YUV 4:4:4.
    Yuv444p,
    /// 8-bit grayscale.
    Gray8,
    /// 10-bit YUV 4:2:0 planar (little-endian u16, top 10 bits used).
    Yuv420p10Le,
}

impl std::fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rgb8 => write!(f, "rgb8"),
            Self::Rgba8 => write!(f, "rgba8"),
            Self::Bgr8 => write!(f, "bgr8"),
            Self::Bgra8 => write!(f, "bgra8"),
            Self::Yuv420p => write!(f, "yuv420p"),
            Self::Nv12 => write!(f, "nv12"),
            Self::Yuv422p => write!(f, "yuv422p"),
            Self::Yuv444p => write!(f, "yuv444p"),
            Self::Gray8 => write!(f, "gray8"),
            Self::Yuv420p10Le => write!(f, "yuv420p10le"),
        }
    }
}

/// Color matrix standard for YUV <-> RGB conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMatrix {
    /// BT.601 (SD video).
    Bt601,
    /// BT.709 (HD video).
    Bt709,
    /// BT.2020 (UHD / HDR video).
    Bt2020,
}

/// YUV-to-RGB conversion coefficients for a given color matrix.
#[derive(Debug, Clone, Copy)]
pub struct YuvCoefficients {
    /// Kr coefficient.
    pub kr: f64,
    /// Kg coefficient (derived: 1 - Kr - Kb).
    pub kg: f64,
    /// Kb coefficient.
    pub kb: f64,
}

impl YuvCoefficients {
    /// Get coefficients for the specified color matrix.
    #[must_use]
    pub fn for_matrix(matrix: ColorMatrix) -> Self {
        match matrix {
            ColorMatrix::Bt601 => Self {
                kr: 0.299,
                kg: 0.587,
                kb: 0.114,
            },
            ColorMatrix::Bt709 => Self {
                kr: 0.2126,
                kg: 0.7152,
                kb: 0.0722,
            },
            ColorMatrix::Bt2020 => Self {
                kr: 0.2627,
                kg: 0.6780,
                kb: 0.0593,
            },
        }
    }
}

/// Convert a single pixel from YUV to RGB using the given coefficients.
///
/// Y is in [16, 235], U/V are in [16, 240] (studio range).
/// Output R, G, B are clamped to [0, 255].
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn yuv_to_rgb(y: u8, u: u8, v: u8, matrix: ColorMatrix) -> (u8, u8, u8) {
    let coeff = YuvCoefficients::for_matrix(matrix);

    let y_f = (f64::from(y) - 16.0) / 219.0;
    let u_f = (f64::from(u) - 128.0) / 224.0;
    let v_f = (f64::from(v) - 128.0) / 224.0;

    let r = y_f + (2.0 * (1.0 - coeff.kr)) * v_f;
    let g = y_f
        - (2.0 * coeff.kb * (1.0 - coeff.kb) / coeff.kg) * u_f
        - (2.0 * coeff.kr * (1.0 - coeff.kr) / coeff.kg) * v_f;
    let b = y_f + (2.0 * (1.0 - coeff.kb)) * u_f;

    (
        clamp_u8((r * 255.0).round()),
        clamp_u8((g * 255.0).round()),
        clamp_u8((b * 255.0).round()),
    )
}

/// Convert a single pixel from RGB to YUV using the given coefficients.
///
/// Input R, G, B are in [0, 255].
/// Output Y in [16, 235], U/V in [16, 240] (studio range).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn rgb_to_yuv(r: u8, g: u8, b: u8, matrix: ColorMatrix) -> (u8, u8, u8) {
    let coeff = YuvCoefficients::for_matrix(matrix);

    let r_f = f64::from(r) / 255.0;
    let g_f = f64::from(g) / 255.0;
    let b_f = f64::from(b) / 255.0;

    let y = coeff.kr * r_f + coeff.kg * g_f + coeff.kb * b_f;
    let u = (b_f - y) / (2.0 * (1.0 - coeff.kb));
    let v = (r_f - y) / (2.0 * (1.0 - coeff.kr));

    let y_out = (y * 219.0 + 16.0).round();
    let u_out = (u * 224.0 + 128.0).round();
    let v_out = (v * 224.0 + 128.0).round();

    (clamp_u8(y_out), clamp_u8(u_out), clamp_u8(v_out))
}

/// Clamp an f64 value to u8 range [0, 255].
fn clamp_u8(v: f64) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Convert an RGB8 buffer to grayscale using BT.709 luminance weights.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn rgb_to_gray(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let expected = width * height * 3;
    let len = rgb.len().min(expected);
    let pixel_count = len / 3;
    let mut gray = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        let r = f64::from(rgb[i * 3]);
        let g = f64::from(rgb[i * 3 + 1]);
        let b = f64::from(rgb[i * 3 + 2]);
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        gray.push(clamp_u8(lum));
    }
    gray
}

/// Convert a grayscale buffer to RGB8 (triplicating each value).
#[must_use]
pub fn gray_to_rgb(gray: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(gray.len() * 3);
    for &g in gray {
        rgb.push(g);
        rgb.push(g);
        rgb.push(g);
    }
    rgb
}

/// Swap RGB to BGR (or vice versa) in-place for an interleaved 3-bpp buffer.
pub fn swap_rgb_bgr(data: &mut [u8]) {
    let len = data.len() / 3;
    for i in 0..len {
        data.swap(i * 3, i * 3 + 2);
    }
}

/// Convert RGBA to RGB by dropping the alpha channel.
#[must_use]
pub fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let pixel_count = rgba.len() / 4;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for i in 0..pixel_count {
        rgb.push(rgba[i * 4]);
        rgb.push(rgba[i * 4 + 1]);
        rgb.push(rgba[i * 4 + 2]);
    }
    rgb
}

/// Convert RGB to RGBA by adding a constant alpha value.
#[must_use]
pub fn rgb_to_rgba(rgb: &[u8], alpha: u8) -> Vec<u8> {
    let pixel_count = rgb.len() / 3;
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        rgba.push(rgb[i * 3]);
        rgba.push(rgb[i * 3 + 1]);
        rgba.push(rgb[i * 3 + 2]);
        rgba.push(alpha);
    }
    rgba
}

/// Compute the required buffer size in bytes for a given format and dimensions.
#[must_use]
pub fn buffer_size(format: PixelFormat, width: usize, height: usize) -> usize {
    match format {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => width * height * 3,
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => width * height * 4,
        PixelFormat::Yuv420p | PixelFormat::Nv12 => width * height * 3 / 2,
        PixelFormat::Yuv422p => width * height * 2,
        PixelFormat::Yuv444p => width * height * 3,
        PixelFormat::Gray8 => width * height,
        PixelFormat::Yuv420p10Le => width * height * 3, // 2 bytes per sample, 1.5 samples per pixel
    }
}

/// Check whether two pixel formats share the same colour model.
#[must_use]
pub fn same_color_model(a: PixelFormat, b: PixelFormat) -> bool {
    let is_rgb = |f: PixelFormat| {
        matches!(
            f,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Bgr8 | PixelFormat::Bgra8
        )
    };
    let is_yuv = |f: PixelFormat| {
        matches!(
            f,
            PixelFormat::Yuv420p
                | PixelFormat::Nv12
                | PixelFormat::Yuv422p
                | PixelFormat::Yuv444p
                | PixelFormat::Yuv420p10Le
        )
    };

    (is_rgb(a) && is_rgb(b))
        || (is_yuv(a) && is_yuv(b))
        || (a == PixelFormat::Gray8 && b == PixelFormat::Gray8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_display() {
        assert_eq!(format!("{}", PixelFormat::Rgb8), "rgb8");
        assert_eq!(format!("{}", PixelFormat::Yuv420p), "yuv420p");
        assert_eq!(format!("{}", PixelFormat::Nv12), "nv12");
        assert_eq!(format!("{}", PixelFormat::Gray8), "gray8");
    }

    #[test]
    fn test_yuv_coefficients_bt601() {
        let c = YuvCoefficients::for_matrix(ColorMatrix::Bt601);
        assert!((c.kr - 0.299).abs() < 1e-6);
        assert!((c.kg - 0.587).abs() < 1e-6);
        assert!((c.kb - 0.114).abs() < 1e-6);
        assert!((c.kr + c.kg + c.kb - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_yuv_coefficients_bt709() {
        let c = YuvCoefficients::for_matrix(ColorMatrix::Bt709);
        assert!((c.kr + c.kg + c.kb - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_yuv_to_rgb_black() {
        let (r, g, b) = yuv_to_rgb(16, 128, 128, ColorMatrix::Bt709);
        assert!(
            r < 5 && g < 5 && b < 5,
            "Black YUV should map to near-black RGB"
        );
    }

    #[test]
    fn test_yuv_to_rgb_white() {
        let (r, g, b) = yuv_to_rgb(235, 128, 128, ColorMatrix::Bt709);
        assert!(
            r > 250 && g > 250 && b > 250,
            "White YUV should map to near-white RGB: ({r},{g},{b})"
        );
    }

    #[test]
    fn test_rgb_to_yuv_roundtrip() {
        let (y, u, v) = rgb_to_yuv(128, 64, 200, ColorMatrix::Bt709);
        let (r, g, b) = yuv_to_rgb(y, u, v, ColorMatrix::Bt709);
        assert!(
            (i16::from(r) - 128).unsigned_abs() <= 2,
            "R roundtrip error"
        );
        assert!((i16::from(g) - 64).unsigned_abs() <= 2, "G roundtrip error");
        assert!(
            (i16::from(b) - 200).unsigned_abs() <= 2,
            "B roundtrip error"
        );
    }

    #[test]
    fn test_rgb_to_gray() {
        let rgb = vec![255, 255, 255, 0, 0, 0]; // white + black
        let gray = rgb_to_gray(&rgb, 2, 1);
        assert_eq!(gray.len(), 2);
        assert!(gray[0] > 250);
        assert!(gray[1] < 5);
    }

    #[test]
    fn test_gray_to_rgb() {
        let gray = vec![128, 64];
        let rgb = gray_to_rgb(&gray);
        assert_eq!(rgb.len(), 6);
        assert_eq!(rgb[0], 128);
        assert_eq!(rgb[1], 128);
        assert_eq!(rgb[2], 128);
        assert_eq!(rgb[3], 64);
    }

    #[test]
    fn test_swap_rgb_bgr() {
        let mut data = vec![10, 20, 30, 40, 50, 60];
        swap_rgb_bgr(&mut data);
        assert_eq!(data, vec![30, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn test_rgba_to_rgb() {
        let rgba = vec![10, 20, 30, 255, 40, 50, 60, 128];
        let rgb = rgba_to_rgb(&rgba);
        assert_eq!(rgb, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn test_rgb_to_rgba() {
        let rgb = vec![10, 20, 30];
        let rgba = rgb_to_rgba(&rgb, 255);
        assert_eq!(rgba, vec![10, 20, 30, 255]);
    }

    #[test]
    fn test_buffer_size() {
        assert_eq!(buffer_size(PixelFormat::Rgb8, 1920, 1080), 1920 * 1080 * 3);
        assert_eq!(buffer_size(PixelFormat::Rgba8, 100, 100), 40000);
        assert_eq!(
            buffer_size(PixelFormat::Yuv420p, 1920, 1080),
            1920 * 1080 * 3 / 2
        );
        assert_eq!(buffer_size(PixelFormat::Gray8, 640, 480), 640 * 480);
    }

    #[test]
    fn test_same_color_model() {
        assert!(same_color_model(PixelFormat::Rgb8, PixelFormat::Rgba8));
        assert!(same_color_model(PixelFormat::Bgr8, PixelFormat::Bgra8));
        assert!(same_color_model(PixelFormat::Yuv420p, PixelFormat::Nv12));
        assert!(!same_color_model(PixelFormat::Rgb8, PixelFormat::Yuv420p));
        assert!(!same_color_model(PixelFormat::Gray8, PixelFormat::Rgb8));
        assert!(same_color_model(PixelFormat::Gray8, PixelFormat::Gray8));
    }
}

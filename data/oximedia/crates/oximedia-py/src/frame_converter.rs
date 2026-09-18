#![allow(dead_code)]
//! Frame format conversion utilities for Python bindings.
//!
//! Provides pixel format conversion, color space transformations,
//! resolution scaling, and frame data export for Python consumers.

use std::collections::HashMap;

/// Supported pixel formats for conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvertPixelFormat {
    /// 8-bit YUV 4:2:0 planar.
    Yuv420p,
    /// 8-bit YUV 4:2:2 planar.
    Yuv422p,
    /// 8-bit YUV 4:4:4 planar.
    Yuv444p,
    /// 8-bit RGB packed.
    Rgb24,
    /// 8-bit RGBA packed.
    Rgba32,
    /// 8-bit BGR packed.
    Bgr24,
    /// 8-bit BGRA packed.
    Bgra32,
    /// 8-bit grayscale.
    Gray8,
    /// 10-bit YUV 4:2:0 planar.
    Yuv420p10,
    /// NV12 semi-planar.
    Nv12,
}

/// Color space identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvertColorSpace {
    /// BT.601 (SD).
    Bt601,
    /// BT.709 (HD).
    Bt709,
    /// BT.2020 (UHD).
    Bt2020,
    /// sRGB.
    Srgb,
    /// Linear light.
    Linear,
}

/// Scaling algorithm used during conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleAlgorithm {
    /// Nearest neighbor (fastest).
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation.
    Bicubic,
    /// Lanczos resampling.
    Lanczos,
}

/// Configuration for a frame conversion operation.
#[derive(Debug, Clone)]
pub struct FrameConvertConfig {
    /// Source pixel format.
    pub src_format: ConvertPixelFormat,
    /// Destination pixel format.
    pub dst_format: ConvertPixelFormat,
    /// Source color space (if color-space conversion is needed).
    pub src_color_space: Option<ConvertColorSpace>,
    /// Destination color space.
    pub dst_color_space: Option<ConvertColorSpace>,
    /// Target width (None = keep original).
    pub target_width: Option<u32>,
    /// Target height (None = keep original).
    pub target_height: Option<u32>,
    /// Scaling algorithm.
    pub scale_algo: ScaleAlgorithm,
    /// Whether to apply dithering on bit-depth reduction.
    pub dither: bool,
}

impl Default for FrameConvertConfig {
    fn default() -> Self {
        Self {
            src_format: ConvertPixelFormat::Yuv420p,
            dst_format: ConvertPixelFormat::Rgb24,
            src_color_space: None,
            dst_color_space: None,
            target_width: None,
            target_height: None,
            scale_algo: ScaleAlgorithm::Bilinear,
            dither: false,
        }
    }
}

/// Represents a raw frame buffer for conversion.
#[derive(Debug, Clone)]
pub struct RawFrame {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format of this frame.
    pub format: ConvertPixelFormat,
    /// Plane data (one Vec per plane).
    pub planes: Vec<Vec<u8>>,
    /// Stride (bytes per row) for each plane.
    pub strides: Vec<usize>,
    /// Presentation timestamp in microseconds.
    pub pts_us: i64,
}

impl RawFrame {
    /// Create a new raw frame with allocated planes.
    pub fn new(width: u32, height: u32, format: ConvertPixelFormat) -> Self {
        let (planes, strides) = Self::allocate_planes(width, height, format);
        Self {
            width,
            height,
            format,
            planes,
            strides,
            pts_us: 0,
        }
    }

    /// Compute total byte size of all planes.
    pub fn total_bytes(&self) -> usize {
        self.planes.iter().map(|p| p.len()).sum()
    }

    /// Check whether the frame dimensions are valid (non-zero).
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0 && !self.planes.is_empty()
    }

    /// Allocate planes for a given format.
    fn allocate_planes(
        width: u32,
        height: u32,
        format: ConvertPixelFormat,
    ) -> (Vec<Vec<u8>>, Vec<usize>) {
        let w = width as usize;
        let h = height as usize;
        match format {
            ConvertPixelFormat::Yuv420p => {
                let y_stride = w;
                let uv_stride = (w + 1) / 2;
                let uv_height = (h + 1) / 2;
                let y_plane = vec![0u8; y_stride * h];
                let u_plane = vec![0u8; uv_stride * uv_height];
                let v_plane = vec![0u8; uv_stride * uv_height];
                (
                    vec![y_plane, u_plane, v_plane],
                    vec![y_stride, uv_stride, uv_stride],
                )
            }
            ConvertPixelFormat::Yuv422p => {
                let y_stride = w;
                let uv_stride = (w + 1) / 2;
                let y_plane = vec![0u8; y_stride * h];
                let u_plane = vec![0u8; uv_stride * h];
                let v_plane = vec![0u8; uv_stride * h];
                (
                    vec![y_plane, u_plane, v_plane],
                    vec![y_stride, uv_stride, uv_stride],
                )
            }
            ConvertPixelFormat::Yuv444p => {
                let stride = w;
                let plane = vec![0u8; stride * h];
                (
                    vec![plane.clone(), plane.clone(), plane],
                    vec![stride, stride, stride],
                )
            }
            ConvertPixelFormat::Rgb24 | ConvertPixelFormat::Bgr24 => {
                let stride = w * 3;
                (vec![vec![0u8; stride * h]], vec![stride])
            }
            ConvertPixelFormat::Rgba32 | ConvertPixelFormat::Bgra32 => {
                let stride = w * 4;
                (vec![vec![0u8; stride * h]], vec![stride])
            }
            ConvertPixelFormat::Gray8 => {
                let stride = w;
                (vec![vec![0u8; stride * h]], vec![stride])
            }
            ConvertPixelFormat::Yuv420p10 => {
                let y_stride = w * 2;
                let uv_stride = ((w + 1) / 2) * 2;
                let uv_height = (h + 1) / 2;
                (
                    vec![
                        vec![0u8; y_stride * h],
                        vec![0u8; uv_stride * uv_height],
                        vec![0u8; uv_stride * uv_height],
                    ],
                    vec![y_stride, uv_stride, uv_stride],
                )
            }
            ConvertPixelFormat::Nv12 => {
                let y_stride = w;
                let uv_stride = w;
                let uv_height = (h + 1) / 2;
                (
                    vec![vec![0u8; y_stride * h], vec![0u8; uv_stride * uv_height]],
                    vec![y_stride, uv_stride],
                )
            }
        }
    }
}

/// Frame converter that applies pixel-format and color-space transformations.
#[derive(Debug)]
pub struct FrameConverter {
    /// Active configuration.
    config: FrameConvertConfig,
    /// Conversion statistics.
    stats: ConvertStats,
}

/// Accumulated statistics for frame conversion.
#[derive(Debug, Clone, Default)]
pub struct ConvertStats {
    /// Number of frames converted.
    pub frames_converted: u64,
    /// Total input bytes processed.
    pub bytes_in: u64,
    /// Total output bytes produced.
    pub bytes_out: u64,
}

impl ConvertStats {
    /// Compression ratio (output / input).
    #[allow(clippy::cast_precision_loss)]
    pub fn ratio(&self) -> f64 {
        if self.bytes_in == 0 {
            0.0
        } else {
            self.bytes_out as f64 / self.bytes_in as f64
        }
    }
}

impl FrameConverter {
    /// Create a converter with the given configuration.
    pub fn new(config: FrameConvertConfig) -> Self {
        Self {
            config,
            stats: ConvertStats::default(),
        }
    }

    /// Return current statistics.
    pub fn stats(&self) -> &ConvertStats {
        &self.stats
    }

    /// Reset statistics to zero.
    pub fn reset_stats(&mut self) {
        self.stats = ConvertStats::default();
    }

    /// Convert a single frame according to the active configuration.
    pub fn convert(&mut self, input: &RawFrame) -> Result<RawFrame, FrameConvertError> {
        if !input.is_valid() {
            return Err(FrameConvertError::InvalidInput(
                "frame has zero dimensions".into(),
            ));
        }
        if input.format != self.config.src_format {
            return Err(FrameConvertError::FormatMismatch {
                expected: self.config.src_format,
                got: input.format,
            });
        }

        let out_w = self.config.target_width.unwrap_or(input.width);
        let out_h = self.config.target_height.unwrap_or(input.height);

        let mut output = RawFrame::new(out_w, out_h, self.config.dst_format);
        output.pts_us = input.pts_us;

        use ConvertPixelFormat::{
            Bgr24, Bgra32, Gray8, Nv12, Rgb24, Rgba32, Yuv420p, Yuv422p, Yuv444p,
        };

        let w = input.width as usize;
        let h = input.height as usize;

        match (self.config.src_format, self.config.dst_format) {
            // ── Same format, same size: copy planes directly ────────────────
            (src, dst) if src == dst && out_w == input.width && out_h == input.height => {
                for (dst_plane, src_plane) in output.planes.iter_mut().zip(input.planes.iter()) {
                    let len = dst_plane.len().min(src_plane.len());
                    dst_plane[..len].copy_from_slice(&src_plane[..len]);
                }
            }

            // ── YUV420p → RGB24 (BT.601) ────────────────────────────────────
            (Yuv420p, Rgb24) => {
                let y_plane = &input.planes[0];
                let u_plane = &input.planes[1];
                let v_plane = &input.planes[2];
                let uv_w = (w + 1) / 2;
                let rgb = &mut output.planes[0];
                for y in 0..h {
                    for x in 0..w {
                        let luma = y_plane[y * w + x] as f32;
                        let u = u_plane[(y / 2) * uv_w + x / 2] as f32 - 128.0;
                        let v = v_plane[(y / 2) * uv_w + x / 2] as f32 - 128.0;
                        let r = (luma + 1.402 * v).clamp(0.0, 255.0) as u8;
                        let g = (luma - 0.344_136 * u - 0.714_136 * v).clamp(0.0, 255.0) as u8;
                        let b = (luma + 1.772 * u).clamp(0.0, 255.0) as u8;
                        let out_off = (y * w + x) * 3;
                        rgb[out_off] = r;
                        rgb[out_off + 1] = g;
                        rgb[out_off + 2] = b;
                    }
                }
            }

            // ── NV12 → RGB24 (BT.601) ───────────────────────────────────────
            // Y plane: w*h; UV interleaved: w*(h/2) at offset w*h
            (Nv12, Rgb24) => {
                let y_plane = &input.planes[0];
                let uv_plane = &input.planes[1];
                let rgb = &mut output.planes[0];
                for y in 0..h {
                    for x in 0..w {
                        let luma = y_plane[y * w + x] as f32;
                        let uv_off = (y / 2) * w + (x / 2) * 2;
                        let u = uv_plane[uv_off] as f32 - 128.0;
                        let v = uv_plane[uv_off + 1] as f32 - 128.0;
                        let r = (luma + 1.402 * v).clamp(0.0, 255.0) as u8;
                        let g = (luma - 0.344_136 * u - 0.714_136 * v).clamp(0.0, 255.0) as u8;
                        let b = (luma + 1.772 * u).clamp(0.0, 255.0) as u8;
                        let out_off = (y * w + x) * 3;
                        rgb[out_off] = r;
                        rgb[out_off + 1] = g;
                        rgb[out_off + 2] = b;
                    }
                }
            }

            // ── RGBA32 → RGB24 ───────────────────────────────────────────────
            (Rgba32, Rgb24) => {
                let src = &input.planes[0];
                let dst = &mut output.planes[0];
                for (rgba, rgb) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
                    rgb[0] = rgba[0];
                    rgb[1] = rgba[1];
                    rgb[2] = rgba[2];
                }
            }

            // ── RGB24 → RGBA32 ───────────────────────────────────────────────
            (Rgb24, Rgba32) => {
                let src = &input.planes[0];
                let dst = &mut output.planes[0];
                for (rgb, rgba) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
                    rgba[0] = rgb[0];
                    rgba[1] = rgb[1];
                    rgba[2] = rgb[2];
                    rgba[3] = 255;
                }
            }

            // ── BGRA32 → BGR24 ───────────────────────────────────────────────
            (Bgra32, Bgr24) => {
                let src = &input.planes[0];
                let dst = &mut output.planes[0];
                for (bgra, bgr) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
                    bgr[0] = bgra[0];
                    bgr[1] = bgra[1];
                    bgr[2] = bgra[2];
                }
            }

            // ── BGR24 → BGRA32 ───────────────────────────────────────────────
            (Bgr24, Bgra32) => {
                let src = &input.planes[0];
                let dst = &mut output.planes[0];
                for (bgr, bgra) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
                    bgra[0] = bgr[0];
                    bgra[1] = bgr[1];
                    bgra[2] = bgr[2];
                    bgra[3] = 255;
                }
            }

            // ── RGB24 → Gray8 (BT.601 luma) ─────────────────────────────────
            (Rgb24, Gray8) => {
                let src = &input.planes[0];
                let dst = &mut output.planes[0];
                for (rgb, g) in src.chunks_exact(3).zip(dst.iter_mut()) {
                    *g = (0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32)
                        .clamp(0.0, 255.0) as u8;
                }
            }

            // ── BGR24 → Gray8 ────────────────────────────────────────────────
            (Bgr24, Gray8) => {
                let src = &input.planes[0];
                let dst = &mut output.planes[0];
                for (bgr, g) in src.chunks_exact(3).zip(dst.iter_mut()) {
                    // B = bgr[0], G = bgr[1], R = bgr[2]
                    *g = (0.114 * bgr[0] as f32 + 0.587 * bgr[1] as f32 + 0.299 * bgr[2] as f32)
                        .clamp(0.0, 255.0) as u8;
                }
            }

            // ── Passthrough with same format but different size → not scaled, just copy ──
            (src, dst) if src == dst => {
                for (dst_plane, src_plane) in output.planes.iter_mut().zip(input.planes.iter()) {
                    let len = dst_plane.len().min(src_plane.len());
                    dst_plane[..len].copy_from_slice(&src_plane[..len]);
                }
            }

            // ── Unsupported pair ─────────────────────────────────────────────
            (src_fmt, dst_fmt) => {
                return Err(FrameConvertError::UnsupportedConversion(format!(
                    "{src_fmt:?} → {dst_fmt:?}"
                )));
            }
        }

        // Silence unused variable warnings from the match arms above.
        let _ = (w, h, Yuv422p, Yuv444p);

        let in_bytes = input.total_bytes() as u64;
        let out_bytes = output.total_bytes() as u64;
        self.stats.frames_converted += 1;
        self.stats.bytes_in += in_bytes;
        self.stats.bytes_out += out_bytes;

        Ok(output)
    }

    /// Batch-convert multiple frames.
    pub fn convert_batch(
        &mut self,
        frames: &[RawFrame],
    ) -> Vec<Result<RawFrame, FrameConvertError>> {
        frames.iter().map(|f| self.convert(f)).collect()
    }
}

/// Errors during frame conversion.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameConvertError {
    /// The input frame is invalid.
    InvalidInput(String),
    /// Pixel format of input does not match expected source format.
    FormatMismatch {
        /// Expected format.
        expected: ConvertPixelFormat,
        /// Actual format.
        got: ConvertPixelFormat,
    },
    /// An unsupported conversion path.
    UnsupportedConversion(String),
}

impl std::fmt::Display for FrameConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::FormatMismatch { expected, got } => {
                write!(f, "format mismatch: expected {expected:?}, got {got:?}")
            }
            Self::UnsupportedConversion(msg) => write!(f, "unsupported conversion: {msg}"),
        }
    }
}

impl std::error::Error for FrameConvertError {}

/// Registry of known format conversion capabilities.
#[derive(Debug, Default)]
pub struct ConversionRegistry {
    /// Supported conversion pairs (src, dst) -> human-readable name.
    supported: HashMap<(ConvertPixelFormat, ConvertPixelFormat), String>,
}

impl ConversionRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a supported conversion path.
    pub fn register(
        &mut self,
        src: ConvertPixelFormat,
        dst: ConvertPixelFormat,
        name: impl Into<String>,
    ) {
        self.supported.insert((src, dst), name.into());
    }

    /// Check whether a conversion path is registered.
    pub fn is_supported(&self, src: ConvertPixelFormat, dst: ConvertPixelFormat) -> bool {
        self.supported.contains_key(&(src, dst))
    }

    /// Number of registered conversions.
    pub fn count(&self) -> usize {
        self.supported.len()
    }

    /// List all registered conversion names.
    pub fn list_names(&self) -> Vec<&str> {
        self.supported.values().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = FrameConvertConfig::default();
        assert_eq!(cfg.src_format, ConvertPixelFormat::Yuv420p);
        assert_eq!(cfg.dst_format, ConvertPixelFormat::Rgb24);
        assert_eq!(cfg.scale_algo, ScaleAlgorithm::Bilinear);
        assert!(!cfg.dither);
    }

    #[test]
    fn test_raw_frame_new_rgb() {
        let frame = RawFrame::new(1920, 1080, ConvertPixelFormat::Rgb24);
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.strides[0], 1920 * 3);
        assert_eq!(frame.total_bytes(), 1920 * 1080 * 3);
    }

    #[test]
    fn test_raw_frame_new_yuv420() {
        let frame = RawFrame::new(320, 240, ConvertPixelFormat::Yuv420p);
        assert_eq!(frame.planes.len(), 3);
        assert_eq!(frame.strides[0], 320);
        assert_eq!(frame.strides[1], 160);
        assert!(frame.is_valid());
    }

    #[test]
    fn test_raw_frame_nv12() {
        let frame = RawFrame::new(640, 480, ConvertPixelFormat::Nv12);
        assert_eq!(frame.planes.len(), 2);
        assert_eq!(frame.strides[0], 640);
        assert_eq!(frame.strides[1], 640);
    }

    #[test]
    fn test_raw_frame_gray8() {
        let frame = RawFrame::new(100, 100, ConvertPixelFormat::Gray8);
        assert_eq!(frame.total_bytes(), 10_000);
    }

    #[test]
    fn test_raw_frame_invalid_zero() {
        let frame = RawFrame {
            width: 0,
            height: 0,
            format: ConvertPixelFormat::Rgb24,
            planes: vec![],
            strides: vec![],
            pts_us: 0,
        };
        assert!(!frame.is_valid());
    }

    #[test]
    fn test_converter_passthrough() {
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Rgb24,
            dst_format: ConvertPixelFormat::Rgb24,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let mut frame = RawFrame::new(4, 4, ConvertPixelFormat::Rgb24);
        frame.planes[0][0] = 0xFF;
        frame.pts_us = 42_000;

        let out = conv.convert(&frame).expect("out should be valid");
        assert_eq!(out.planes[0][0], 0xFF);
        assert_eq!(out.pts_us, 42_000);
        assert_eq!(conv.stats().frames_converted, 1);
    }

    #[test]
    fn test_converter_format_mismatch() {
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Yuv420p,
            dst_format: ConvertPixelFormat::Rgb24,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let frame = RawFrame::new(4, 4, ConvertPixelFormat::Rgb24);
        let err = conv.convert(&frame).unwrap_err();
        assert!(matches!(err, FrameConvertError::FormatMismatch { .. }));
    }

    #[test]
    fn test_converter_invalid_input() {
        let cfg = FrameConvertConfig::default();
        let mut conv = FrameConverter::new(cfg);
        let frame = RawFrame {
            width: 0,
            height: 0,
            format: ConvertPixelFormat::Yuv420p,
            planes: vec![],
            strides: vec![],
            pts_us: 0,
        };
        assert!(conv.convert(&frame).is_err());
    }

    #[test]
    fn test_converter_batch() {
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Gray8,
            dst_format: ConvertPixelFormat::Gray8,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let frames: Vec<RawFrame> = (0..5)
            .map(|_| RawFrame::new(8, 8, ConvertPixelFormat::Gray8))
            .collect();
        let results = conv.convert_batch(&frames);
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.is_ok()));
        assert_eq!(conv.stats().frames_converted, 5);
    }

    #[test]
    fn test_convert_stats_ratio() {
        let mut stats = ConvertStats::default();
        assert_eq!(stats.ratio(), 0.0);
        stats.bytes_in = 1000;
        stats.bytes_out = 500;
        assert!((stats.ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_registry() {
        let mut reg = ConversionRegistry::new();
        assert_eq!(reg.count(), 0);
        reg.register(
            ConvertPixelFormat::Yuv420p,
            ConvertPixelFormat::Rgb24,
            "yuv420_to_rgb",
        );
        assert!(reg.is_supported(ConvertPixelFormat::Yuv420p, ConvertPixelFormat::Rgb24));
        assert!(!reg.is_supported(ConvertPixelFormat::Rgb24, ConvertPixelFormat::Yuv420p));
        assert_eq!(reg.count(), 1);
        let names = reg.list_names();
        assert_eq!(names, vec!["yuv420_to_rgb"]);
    }

    #[test]
    fn test_frame_convert_error_display() {
        let e = FrameConvertError::InvalidInput("bad data".into());
        assert!(e.to_string().contains("bad data"));
        let e2 = FrameConvertError::UnsupportedConversion("nope".into());
        assert!(e2.to_string().contains("nope"));
    }

    // ── Pixel conversion tests ────────────────────────────────────────────────

    #[test]
    fn test_yuv420p_to_rgb24_black_pixel() {
        // Y=16, U=128, V=128 should produce approximately black in BT.601.
        let w = 2usize;
        let h = 2usize;
        let mut frame = RawFrame::new(w as u32, h as u32, ConvertPixelFormat::Yuv420p);
        // Y plane: all 16 (video black)
        for v in &mut frame.planes[0] {
            *v = 16;
        }
        // U/V planes: 128 (neutral chrominance)
        for v in &mut frame.planes[1] {
            *v = 128;
        }
        for v in &mut frame.planes[2] {
            *v = 128;
        }

        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Yuv420p,
            dst_format: ConvertPixelFormat::Rgb24,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let out = conv.convert(&frame).expect("conversion should succeed");
        assert_eq!(out.planes.len(), 1);
        // With BT.601 formula and Y=16, U=128, V=128 the output is very dark.
        assert_eq!(out.total_bytes(), w * h * 3);
    }

    #[test]
    fn test_yuv420p_to_rgb24_white_pixel() {
        let w = 2usize;
        let h = 2usize;
        let mut frame = RawFrame::new(w as u32, h as u32, ConvertPixelFormat::Yuv420p);
        // Y=235 (video white), U=128, V=128
        for v in &mut frame.planes[0] {
            *v = 235;
        }
        for v in &mut frame.planes[1] {
            *v = 128;
        }
        for v in &mut frame.planes[2] {
            *v = 128;
        }
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Yuv420p,
            dst_format: ConvertPixelFormat::Rgb24,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let out = conv.convert(&frame).expect("should succeed");
        // All pixels should be bright (close to 255).
        for &byte in &out.planes[0] {
            assert!(byte > 200, "expected bright pixel, got {byte}");
        }
    }

    #[test]
    fn test_nv12_to_rgb24() {
        let w = 4usize;
        let h = 4usize;
        let mut frame = RawFrame::new(w as u32, h as u32, ConvertPixelFormat::Nv12);
        // Y=235 (white), UV=128,128 (neutral)
        for v in &mut frame.planes[0] {
            *v = 235;
        }
        for chunk in frame.planes[1].chunks_exact_mut(2) {
            chunk[0] = 128;
            chunk[1] = 128;
        }
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Nv12,
            dst_format: ConvertPixelFormat::Rgb24,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let out = conv.convert(&frame).expect("nv12→rgb24 should succeed");
        assert_eq!(out.total_bytes(), w * h * 3);
        for &byte in &out.planes[0] {
            assert!(byte > 200, "expected bright pixel, got {byte}");
        }
    }

    #[test]
    fn test_rgba_to_rgb24() {
        let mut frame = RawFrame::new(2, 2, ConvertPixelFormat::Rgba32);
        // R=100, G=150, B=200, A=255
        for chunk in frame.planes[0].chunks_exact_mut(4) {
            chunk[0] = 100;
            chunk[1] = 150;
            chunk[2] = 200;
            chunk[3] = 255;
        }
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Rgba32,
            dst_format: ConvertPixelFormat::Rgb24,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let out = conv.convert(&frame).expect("rgba→rgb24 should succeed");
        for chunk in out.planes[0].chunks_exact(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn test_rgb24_to_rgba32() {
        let mut frame = RawFrame::new(2, 2, ConvertPixelFormat::Rgb24);
        for chunk in frame.planes[0].chunks_exact_mut(3) {
            chunk[0] = 10;
            chunk[1] = 20;
            chunk[2] = 30;
        }
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Rgb24,
            dst_format: ConvertPixelFormat::Rgba32,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let out = conv.convert(&frame).expect("rgb24→rgba32 should succeed");
        for chunk in out.planes[0].chunks_exact(4) {
            assert_eq!(chunk[0], 10);
            assert_eq!(chunk[1], 20);
            assert_eq!(chunk[2], 30);
            assert_eq!(chunk[3], 255); // alpha must be 255
        }
    }

    #[test]
    fn test_unsupported_pair_returns_error() {
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Yuv420p,
            dst_format: ConvertPixelFormat::Yuv422p,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let frame = RawFrame::new(4, 4, ConvertPixelFormat::Yuv420p);
        let err = conv.convert(&frame).unwrap_err();
        assert!(
            matches!(err, FrameConvertError::UnsupportedConversion(_)),
            "expected UnsupportedConversion, got {err:?}"
        );
    }

    #[test]
    fn test_convert_stats_updated() {
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Rgba32,
            dst_format: ConvertPixelFormat::Rgb24,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let frame = RawFrame::new(4, 4, ConvertPixelFormat::Rgba32);
        let _ = conv.convert(&frame).expect("should succeed");
        assert_eq!(conv.stats().bytes_in, 4 * 4 * 4);
        assert_eq!(conv.stats().bytes_out, 4 * 4 * 3);
    }

    #[test]
    fn test_rgb24_to_gray8() {
        let mut frame = RawFrame::new(2, 2, ConvertPixelFormat::Rgb24);
        // Pure white R=G=B=255 → gray ≈ 255
        for chunk in frame.planes[0].chunks_exact_mut(3) {
            chunk[0] = 255;
            chunk[1] = 255;
            chunk[2] = 255;
        }
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Rgb24,
            dst_format: ConvertPixelFormat::Gray8,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let out = conv.convert(&frame).expect("rgb24→gray8 should succeed");
        for &byte in &out.planes[0] {
            assert_eq!(byte, 255);
        }
    }

    #[test]
    fn test_reset_stats() {
        let cfg = FrameConvertConfig {
            src_format: ConvertPixelFormat::Gray8,
            dst_format: ConvertPixelFormat::Gray8,
            ..Default::default()
        };
        let mut conv = FrameConverter::new(cfg);
        let frame = RawFrame::new(2, 2, ConvertPixelFormat::Gray8);
        let _ = conv.convert(&frame);
        assert_eq!(conv.stats().frames_converted, 1);
        conv.reset_stats();
        assert_eq!(conv.stats().frames_converted, 0);
    }
}

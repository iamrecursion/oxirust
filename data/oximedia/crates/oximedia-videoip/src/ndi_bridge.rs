//! NDI to ST 2110 bridging.
//!
//! Provides configuration and utilities for bridging between NDI
//! (Network Device Interface) and SMPTE ST 2110 media-over-IP streams.
//!
//! Note: This module provides bridging configuration and format conversion
//! utilities. Actual NDI SDK integration requires proprietary licensing.

#![allow(dead_code)]

/// Configuration for NDI-to-ST-2110 bridge.
#[derive(Debug, Clone)]
pub struct NdiToSt2110Config {
    /// NDI source name to receive from.
    pub ndi_source: String,
    /// Output multicast address (e.g. "239.255.0.1").
    pub output_multicast: String,
    /// UDP port for the ST 2110 output stream.
    pub port: u16,
}

impl NdiToSt2110Config {
    /// Creates a new NDI-to-ST-2110 configuration.
    #[must_use]
    pub fn new(
        ndi_source: impl Into<String>,
        output_multicast: impl Into<String>,
        port: u16,
    ) -> Self {
        Self {
            ndi_source: ndi_source.into(),
            output_multicast: output_multicast.into(),
            port,
        }
    }
}

/// Configuration for ST-2110-to-NDI bridge.
#[derive(Debug, Clone)]
pub struct St2110ToNdiConfig {
    /// ST 2110 source multicast address to receive from.
    pub source_multicast: String,
    /// UDP port for the ST 2110 input stream.
    pub port: u16,
    /// NDI name to publish the output as.
    pub ndi_name: String,
}

impl St2110ToNdiConfig {
    /// Creates a new ST-2110-to-NDI configuration.
    #[must_use]
    pub fn new(
        source_multicast: impl Into<String>,
        port: u16,
        ndi_name: impl Into<String>,
    ) -> Self {
        Self {
            source_multicast: source_multicast.into(),
            port,
            ndi_name: ndi_name.into(),
        }
    }
}

/// Statistics for a running bridge.
#[derive(Debug, Clone, Default)]
pub struct BridgeStats {
    /// Total number of frames converted.
    pub frames_converted: u64,
    /// Average conversion latency in milliseconds.
    pub conversion_latency_ms_avg: f32,
    /// Total number of conversion errors.
    pub errors: u32,
}

impl BridgeStats {
    /// Creates a new zeroed `BridgeStats`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            frames_converted: 0,
            conversion_latency_ms_avg: 0.0,
            errors: 0,
        }
    }

    /// Updates the running average latency with a new sample.
    pub fn record_frame(&mut self, latency_ms: f32) {
        self.frames_converted += 1;
        // Exponential moving average (alpha = 0.1)
        const ALPHA: f32 = 0.1;
        self.conversion_latency_ms_avg =
            (1.0 - ALPHA) * self.conversion_latency_ms_avg + ALPHA * latency_ms;
    }

    /// Records a conversion error.
    pub fn record_error(&mut self) {
        self.errors += 1;
    }
}

/// Video format descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFormat {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Bit depth per component.
    pub bit_depth: u8,
}

impl VideoFormat {
    /// Creates a new video format descriptor.
    #[must_use]
    pub const fn new(width: u32, height: u32, fps_num: u32, fps_den: u32, bit_depth: u8) -> Self {
        Self {
            width,
            height,
            fps_num,
            fps_den,
            bit_depth,
        }
    }

    /// Returns the frame rate as a floating-point value.
    #[must_use]
    pub fn fps_f64(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den)
    }

    /// Returns true if two formats have the same resolution and frame rate.
    #[must_use]
    pub fn is_compatible(a: &VideoFormat, b: &VideoFormat) -> bool {
        a.width == b.width && a.height == b.height && a.fps_num * b.fps_den == b.fps_num * a.fps_den
    }
}

impl Default for VideoFormat {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            bit_depth: 8,
        }
    }
}

/// Bridge configuration combining source and destination formats.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Source video format.
    pub source_format: VideoFormat,
    /// Destination video format.
    pub dest_format: VideoFormat,
    /// Whether transcoding is allowed when formats differ.
    pub enable_transcoding: bool,
}

impl BridgeConfig {
    /// Creates a new bridge configuration.
    #[must_use]
    pub const fn new(
        source_format: VideoFormat,
        dest_format: VideoFormat,
        enable_transcoding: bool,
    ) -> Self {
        Self {
            source_format,
            dest_format,
            enable_transcoding,
        }
    }

    /// Returns true if formats are compatible (no transcoding needed).
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        VideoFormat::is_compatible(&self.source_format, &self.dest_format)
            && self.source_format.bit_depth == self.dest_format.bit_depth
    }
}

/// Pixel format for raw video data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFmt {
    /// YCbCr 4:2:2, 8 bits per component.
    Yuv422_8,
    /// YCbCr 4:2:2, 10 bits per component (packed).
    Yuv422_10,
    /// RGB, 24 bits (8 bits per channel).
    Rgb24,
    /// RGBA, 32 bits (8 bits per channel).
    Rgba32,
    /// UYVY (packed YCbCr 4:2:2, 8-bit, UYVY byte order).
    Uyvy,
}

impl PixelFmt {
    /// Returns the number of bytes per pixel (may be fractional for subsampled formats).
    ///
    /// For 4:2:2 formats, this is the average bytes per pixel (2 bytes for 8-bit).
    #[must_use]
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            PixelFmt::Yuv422_8 | PixelFmt::Uyvy => 2,
            PixelFmt::Yuv422_10 => 3, // approximate: 20 bits per 2 pixels = 2.5 bytes, rounded up
            PixelFmt::Rgb24 => 3,
            PixelFmt::Rgba32 => 4,
        }
    }
}

/// Format converter for pixel data.
pub struct FormatConverter;

impl FormatConverter {
    /// Converts a row of pixels from one format to another.
    ///
    /// This is a simplified conversion that handles common cases.
    /// `width` is the number of pixels in the row.
    #[must_use]
    pub fn convert_pixel_format(
        src: &[u8],
        src_fmt: &PixelFmt,
        dst_fmt: &PixelFmt,
        width: u32,
    ) -> Vec<u8> {
        if src_fmt == dst_fmt {
            return src.to_vec();
        }

        match (src_fmt, dst_fmt) {
            // UYVY -> RGB24
            (PixelFmt::Uyvy, PixelFmt::Rgb24) => Self::uyvy_to_rgb24(src, width),
            // RGB24 -> UYVY
            (PixelFmt::Rgb24, PixelFmt::Uyvy) => Self::rgb24_to_uyvy(src, width),
            // RGB24 -> RGBA32
            (PixelFmt::Rgb24, PixelFmt::Rgba32) => Self::rgb24_to_rgba32(src, width),
            // RGBA32 -> RGB24
            (PixelFmt::Rgba32, PixelFmt::Rgb24) => Self::rgba32_to_rgb24(src, width),
            // YUV422_8 -> RGB24 (treat as UYVY)
            (PixelFmt::Yuv422_8, PixelFmt::Rgb24) => Self::uyvy_to_rgb24(src, width),
            // Fallback: return source unchanged
            _ => src.to_vec(),
        }
    }

    fn uyvy_to_rgb24(src: &[u8], width: u32) -> Vec<u8> {
        let mut dst = Vec::with_capacity((width * 3) as usize);
        let pairs = (width / 2) as usize;

        for i in 0..pairs {
            let base = i * 4;
            if base + 3 >= src.len() {
                break;
            }
            let u = i32::from(src[base]) - 128;
            let y0 = i32::from(src[base + 1]);
            let v = i32::from(src[base + 2]) - 128;
            let y1 = i32::from(src[base + 3]);

            let (r0, g0, b0) = Self::yuv_to_rgb(y0, u, v);
            let (r1, g1, b1) = Self::yuv_to_rgb(y1, u, v);

            dst.extend_from_slice(&[r0, g0, b0, r1, g1, b1]);
        }

        dst
    }

    fn rgb24_to_uyvy(src: &[u8], width: u32) -> Vec<u8> {
        let mut dst = Vec::with_capacity((width * 2) as usize);
        let pairs = (width / 2) as usize;

        for i in 0..pairs {
            let base = i * 6;
            if base + 5 >= src.len() {
                break;
            }
            let r0 = f32::from(src[base]);
            let g0 = f32::from(src[base + 1]);
            let b0 = f32::from(src[base + 2]);
            let r1 = f32::from(src[base + 3]);
            let g1 = f32::from(src[base + 4]);
            let b1 = f32::from(src[base + 5]);

            let y0 = (0.299 * r0 + 0.587 * g0 + 0.114 * b0) as u8;
            let y1 = (0.299 * r1 + 0.587 * g1 + 0.114 * b1) as u8;
            let u = ((-0.147 * r0 - 0.289 * g0 + 0.436 * b0) + 128.0) as u8;
            let v = ((0.615 * r0 - 0.515 * g0 - 0.100 * b0) + 128.0) as u8;

            dst.extend_from_slice(&[u, y0, v, y1]);
        }

        dst
    }

    fn rgb24_to_rgba32(src: &[u8], width: u32) -> Vec<u8> {
        let mut dst = Vec::with_capacity((width * 4) as usize);
        for chunk in src.chunks_exact(3) {
            dst.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
        }
        dst
    }

    fn rgba32_to_rgb24(src: &[u8], width: u32) -> Vec<u8> {
        let mut dst = Vec::with_capacity((width * 3) as usize);
        for chunk in src.chunks_exact(4) {
            dst.extend_from_slice(&[chunk[0], chunk[1], chunk[2]]);
        }
        dst
    }

    fn yuv_to_rgb(y: i32, u: i32, v: i32) -> (u8, u8, u8) {
        let r = (y + 1_402 * v / 1_000).clamp(0, 255) as u8;
        let g = (y - 344 * u / 1_000 - 714 * v / 1_000).clamp(0, 255) as u8;
        let b = (y + 1_772 * u / 1_000).clamp(0, 255) as u8;
        (r, g, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndi_to_st2110_config() {
        let config = NdiToSt2110Config::new("Camera 1", "239.255.0.1", 5000);
        assert_eq!(config.ndi_source, "Camera 1");
        assert_eq!(config.output_multicast, "239.255.0.1");
        assert_eq!(config.port, 5000);
    }

    #[test]
    fn test_st2110_to_ndi_config() {
        let config = St2110ToNdiConfig::new("239.255.0.1", 5000, "Output 1");
        assert_eq!(config.source_multicast, "239.255.0.1");
        assert_eq!(config.port, 5000);
        assert_eq!(config.ndi_name, "Output 1");
    }

    #[test]
    fn test_bridge_stats_initial() {
        let stats = BridgeStats::new();
        assert_eq!(stats.frames_converted, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.conversion_latency_ms_avg, 0.0);
    }

    #[test]
    fn test_bridge_stats_record_frame() {
        let mut stats = BridgeStats::new();
        stats.record_frame(10.0);
        assert_eq!(stats.frames_converted, 1);
        assert!(stats.conversion_latency_ms_avg > 0.0);
    }

    #[test]
    fn test_bridge_stats_record_error() {
        let mut stats = BridgeStats::new();
        stats.record_error();
        stats.record_error();
        assert_eq!(stats.errors, 2);
    }

    #[test]
    fn test_video_format_is_compatible_same() {
        let a = VideoFormat::new(1920, 1080, 30, 1, 8);
        let b = VideoFormat::new(1920, 1080, 30, 1, 10);
        assert!(VideoFormat::is_compatible(&a, &b));
    }

    #[test]
    fn test_video_format_is_compatible_rational_fps() {
        let a = VideoFormat::new(1920, 1080, 30000, 1001, 8);
        let b = VideoFormat::new(1920, 1080, 30000, 1001, 8);
        assert!(VideoFormat::is_compatible(&a, &b));
    }

    #[test]
    fn test_video_format_is_compatible_different_resolution() {
        let a = VideoFormat::new(1920, 1080, 30, 1, 8);
        let b = VideoFormat::new(1280, 720, 30, 1, 8);
        assert!(!VideoFormat::is_compatible(&a, &b));
    }

    #[test]
    fn test_video_format_is_compatible_different_fps() {
        let a = VideoFormat::new(1920, 1080, 30, 1, 8);
        let b = VideoFormat::new(1920, 1080, 25, 1, 8);
        assert!(!VideoFormat::is_compatible(&a, &b));
    }

    #[test]
    fn test_bridge_config_passthrough() {
        let fmt = VideoFormat::new(1920, 1080, 30, 1, 8);
        let config = BridgeConfig::new(fmt.clone(), fmt, false);
        assert!(config.is_passthrough());
    }

    #[test]
    fn test_pixel_fmt_bytes_per_pixel() {
        assert_eq!(PixelFmt::Rgb24.bytes_per_pixel(), 3);
        assert_eq!(PixelFmt::Rgba32.bytes_per_pixel(), 4);
        assert_eq!(PixelFmt::Yuv422_8.bytes_per_pixel(), 2);
        assert_eq!(PixelFmt::Uyvy.bytes_per_pixel(), 2);
    }

    #[test]
    fn test_format_converter_passthrough() {
        let src = vec![1u8, 2, 3, 4, 5, 6];
        let result =
            FormatConverter::convert_pixel_format(&src, &PixelFmt::Rgb24, &PixelFmt::Rgb24, 2);
        assert_eq!(result, src);
    }

    #[test]
    fn test_format_converter_rgb24_to_rgba32() {
        let src = vec![255u8, 0, 0, 0, 255, 0];
        let result =
            FormatConverter::convert_pixel_format(&src, &PixelFmt::Rgb24, &PixelFmt::Rgba32, 2);
        assert_eq!(result.len(), 8);
        assert_eq!(result[3], 255); // alpha
        assert_eq!(result[7], 255); // alpha
    }

    #[test]
    fn test_format_converter_rgba32_to_rgb24() {
        let src = vec![255u8, 0, 0, 128, 0, 255, 0, 200];
        let result =
            FormatConverter::convert_pixel_format(&src, &PixelFmt::Rgba32, &PixelFmt::Rgb24, 2);
        assert_eq!(result.len(), 6);
        assert_eq!(&result[..3], &[255, 0, 0]);
        assert_eq!(&result[3..6], &[0, 255, 0]);
    }
}

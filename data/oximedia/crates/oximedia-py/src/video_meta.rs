//! Python-exposed video processing metadata types.
//!
//! Provides plain Rust structs for video info, filters, and pixel formats.
//! These can be wrapped with PyO3 annotations later if needed.

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────
//  PyPixelFormat
// ─────────────────────────────────────────────────────────────

/// Pixel format enumeration for video frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyPixelFormat {
    /// YUV 4:2:0 planar (most common).
    Yuv420p,
    /// YUV 4:2:2 planar.
    Yuv422p,
    /// YUV 4:4:4 planar.
    Yuv444p,
    /// Packed RGB, 3 bytes per pixel.
    Rgb24,
    /// Packed RGBA, 4 bytes per pixel.
    Rgba,
    /// 8-bit grayscale.
    Gray8,
}

/// Parse a pixel format from a string slice.
///
/// Returns `None` if the string is not recognised.
pub fn format_from_str(s: &str) -> Option<PyPixelFormat> {
    match s {
        "yuv420p" => Some(PyPixelFormat::Yuv420p),
        "yuv422p" => Some(PyPixelFormat::Yuv422p),
        "yuv444p" => Some(PyPixelFormat::Yuv444p),
        "rgb24" => Some(PyPixelFormat::Rgb24),
        "rgba" => Some(PyPixelFormat::Rgba),
        "gray8" => Some(PyPixelFormat::Gray8),
        _ => None,
    }
}

/// Return the number of bytes per pixel for a packed representation.
///
/// For planar formats (YUV) this returns the average bytes-per-pixel value
/// as `usize` rounded up (e.g. YUV 4:2:0 → 1 byte/px on average, but we
/// return 2 as a safe upper bound for buffer allocation).
pub fn bytes_per_pixel(fmt: PyPixelFormat) -> usize {
    match fmt {
        PyPixelFormat::Yuv420p => 2,
        PyPixelFormat::Yuv422p => 2,
        PyPixelFormat::Yuv444p => 3,
        PyPixelFormat::Rgb24 => 3,
        PyPixelFormat::Rgba => 4,
        PyPixelFormat::Gray8 => 1,
    }
}

// ─────────────────────────────────────────────────────────────
//  VideoMeta
// ─────────────────────────────────────────────────────────────

/// Video stream metadata, ready to be surfaced to Python.
#[derive(Debug, Clone)]
pub struct VideoMeta {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame-rate numerator.
    pub fps_num: u32,
    /// Frame-rate denominator.
    pub fps_den: u32,
    /// Pixel format string (e.g. `"yuv420p"`).
    pub pixel_format: String,
    /// Total duration in milliseconds.
    pub duration_ms: u64,
    /// Codec name (e.g. `"AV1"`, `"VP9"`).
    pub codec: String,
    /// Nominal bitrate in kilobits per second.
    pub bitrate_kbps: u32,
}

impl VideoMeta {
    /// Construct a new `VideoMeta`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
        pixel_format: &str,
        duration_ms: u64,
        codec: &str,
        bitrate_kbps: u32,
    ) -> Self {
        Self {
            width,
            height,
            fps_num,
            fps_den,
            pixel_format: pixel_format.to_string(),
            duration_ms,
            codec: codec.to_string(),
            bitrate_kbps,
        }
    }

    /// Frames per second as a floating-point value.
    ///
    /// Returns `0.0` when `fps_den` is zero.
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 {
            0.0
        } else {
            f64::from(self.fps_num) / f64::from(self.fps_den)
        }
    }

    /// Total number of frames derived from duration and frame rate.
    pub fn total_frames(&self) -> u64 {
        let fps = self.fps();
        if fps <= 0.0 {
            return 0;
        }
        let duration_s = self.duration_ms as f64 / 1000.0;
        (duration_s * fps) as u64
    }

    /// Aspect ratio (width / height) as a float.
    ///
    /// Returns `0.0` when height is zero.
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            0.0
        } else {
            f64::from(self.width) / f64::from(self.height)
        }
    }

    /// Returns `true` if the video is HD (at least 1280×720).
    pub fn is_hd(&self) -> bool {
        self.width >= 1280 && self.height >= 720
    }

    /// Returns `true` if the video is 4K UHD (at least 3840×2160).
    pub fn is_4k(&self) -> bool {
        self.width >= 3840 && self.height >= 2160
    }
}

// ─────────────────────────────────────────────────────────────
//  VideoFilter
// ─────────────────────────────────────────────────────────────

/// A named video filter with a key/value parameter map.
#[derive(Debug, Clone)]
pub struct VideoFilter {
    /// Filter identifier (e.g. `"brightness"`, `"sharpen"`).
    pub name: String,
    /// Numeric parameters keyed by name.
    pub params: HashMap<String, f64>,
}

impl VideoFilter {
    /// Create a new filter with no parameters.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            params: HashMap::new(),
        }
    }

    /// Set or overwrite a named parameter.
    pub fn set_param(&mut self, key: &str, val: f64) {
        self.params.insert(key.to_string(), val);
    }

    /// Retrieve a named parameter, or `None` if absent.
    pub fn get_param(&self, key: &str) -> Option<f64> {
        self.params.get(key).copied()
    }

    /// Apply this filter to raw frame data in-place.
    ///
    /// This is a simplified demonstration: only "brightness" is
    /// implemented (adds a constant offset clamped to 0–255).
    /// All other filter names are no-ops.
    ///
    /// # Arguments
    ///
    /// * `frame_data` - Mutable byte slice of packed pixel data.
    /// * `_width`     - Frame width (reserved for future use).
    /// * `_height`    - Frame height (reserved for future use).
    pub fn apply_to_frame(&self, frame_data: &mut [u8], _width: u32, _height: u32) {
        if self.name == "brightness" {
            let offset = self.params.get("offset").copied().unwrap_or(0.0);
            let offset_i32 = offset as i32;
            for byte in frame_data.iter_mut() {
                let v = i32::from(*byte) + offset_i32;
                *byte = v.clamp(0, 255) as u8;
            }
        }
        // Other filter names are intentionally no-ops in this skeleton.
    }
}

// ─────────────────────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── format_from_str ──────────────────────────────────────

    #[test]
    fn test_format_from_str_known() {
        assert_eq!(format_from_str("yuv420p"), Some(PyPixelFormat::Yuv420p));
        assert_eq!(format_from_str("yuv422p"), Some(PyPixelFormat::Yuv422p));
        assert_eq!(format_from_str("yuv444p"), Some(PyPixelFormat::Yuv444p));
        assert_eq!(format_from_str("rgb24"), Some(PyPixelFormat::Rgb24));
        assert_eq!(format_from_str("rgba"), Some(PyPixelFormat::Rgba));
        assert_eq!(format_from_str("gray8"), Some(PyPixelFormat::Gray8));
    }

    #[test]
    fn test_format_from_str_unknown() {
        assert_eq!(format_from_str("nv12"), None);
        assert_eq!(format_from_str(""), None);
        assert_eq!(format_from_str("YUV420P"), None); // case-sensitive
    }

    // ── bytes_per_pixel ──────────────────────────────────────

    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(bytes_per_pixel(PyPixelFormat::Gray8), 1);
        assert_eq!(bytes_per_pixel(PyPixelFormat::Yuv420p), 2);
        assert_eq!(bytes_per_pixel(PyPixelFormat::Yuv422p), 2);
        assert_eq!(bytes_per_pixel(PyPixelFormat::Yuv444p), 3);
        assert_eq!(bytes_per_pixel(PyPixelFormat::Rgb24), 3);
        assert_eq!(bytes_per_pixel(PyPixelFormat::Rgba), 4);
    }

    // ── VideoMeta ────────────────────────────────────────────

    #[test]
    fn test_video_meta_new() {
        let m = VideoMeta::new(1920, 1080, 30, 1, "yuv420p", 60_000, "AV1", 4000);
        assert_eq!(m.width, 1920);
        assert_eq!(m.height, 1080);
        assert_eq!(m.fps_num, 30);
        assert_eq!(m.fps_den, 1);
        assert_eq!(m.pixel_format, "yuv420p");
        assert_eq!(m.duration_ms, 60_000);
        assert_eq!(m.codec, "AV1");
        assert_eq!(m.bitrate_kbps, 4000);
    }

    #[test]
    fn test_video_meta_fps() {
        let m = VideoMeta::new(1280, 720, 24000, 1001, "yuv420p", 0, "VP9", 0);
        let fps = m.fps();
        assert!((fps - 23.976_023_976).abs() < 1e-6);
    }

    #[test]
    fn test_video_meta_fps_zero_den() {
        let m = VideoMeta::new(1920, 1080, 30, 0, "yuv420p", 0, "AV1", 0);
        assert_eq!(m.fps(), 0.0);
    }

    #[test]
    fn test_video_meta_total_frames() {
        let m = VideoMeta::new(1920, 1080, 25, 1, "yuv420p", 10_000, "AV1", 0);
        // 10 s × 25 fps = 250 frames
        assert_eq!(m.total_frames(), 250);
    }

    #[test]
    fn test_video_meta_aspect_ratio() {
        let m = VideoMeta::new(1920, 1080, 30, 1, "yuv420p", 0, "AV1", 0);
        let ar = m.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_video_meta_aspect_ratio_zero_height() {
        let m = VideoMeta::new(1920, 0, 30, 1, "yuv420p", 0, "AV1", 0);
        assert_eq!(m.aspect_ratio(), 0.0);
    }

    #[test]
    fn test_video_meta_is_hd() {
        let hd = VideoMeta::new(1280, 720, 30, 1, "yuv420p", 0, "AV1", 0);
        assert!(hd.is_hd());
        let sd = VideoMeta::new(640, 480, 30, 1, "yuv420p", 0, "AV1", 0);
        assert!(!sd.is_hd());
    }

    #[test]
    fn test_video_meta_is_4k() {
        let uhd = VideoMeta::new(3840, 2160, 30, 1, "yuv420p", 0, "AV1", 0);
        assert!(uhd.is_4k());
        let fhd = VideoMeta::new(1920, 1080, 30, 1, "yuv420p", 0, "AV1", 0);
        assert!(!fhd.is_4k());
    }

    // ── VideoFilter ──────────────────────────────────────────

    #[test]
    fn test_video_filter_new() {
        let f = VideoFilter::new("brightness");
        assert_eq!(f.name, "brightness");
        assert!(f.params.is_empty());
    }

    #[test]
    fn test_video_filter_set_get_param() {
        let mut f = VideoFilter::new("sharpen");
        f.set_param("strength", 0.5);
        assert_eq!(f.get_param("strength"), Some(0.5));
        assert_eq!(f.get_param("missing"), None);
    }

    #[test]
    fn test_video_filter_overwrite_param() {
        let mut f = VideoFilter::new("blur");
        f.set_param("radius", 1.0);
        f.set_param("radius", 3.0);
        assert_eq!(f.get_param("radius"), Some(3.0));
    }

    #[test]
    fn test_video_filter_apply_brightness() {
        let mut f = VideoFilter::new("brightness");
        f.set_param("offset", 10.0);
        let mut data = vec![100u8, 200u8, 250u8];
        f.apply_to_frame(&mut data, 1, 3);
        assert_eq!(data[0], 110);
        assert_eq!(data[1], 210);
        assert_eq!(data[2], 255); // clamped
    }

    #[test]
    fn test_video_filter_apply_noop() {
        let f = VideoFilter::new("unknown_filter");
        let mut data = vec![50u8, 100u8, 150u8];
        let original = data.clone();
        f.apply_to_frame(&mut data, 3, 1);
        assert_eq!(data, original);
    }
}

//! Python video type binding data structures.
//!
//! Defines plain Rust structs that mirror the data exchanged between Python
//! and the OxiMedia video pipeline.  These are not actual `#[pyclass]`
//! types – they serve as the internal representation before conversion.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Pixel format identifiers used in video frames.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormatBinding {
    /// 4:2:0 planar YUV.
    Yuv420p,
    /// 4:2:2 planar YUV.
    Yuv422p,
    /// 4:4:4 planar YUV.
    Yuv444p,
    /// 24-bit packed RGB.
    Rgb24,
    /// 32-bit RGBA.
    Rgba,
    /// 10-bit 4:2:0 planar YUV.
    Yuv420p10le,
    /// Unknown / not yet identified.
    Unknown,
}

impl PixelFormatBinding {
    /// Return the bits per pixel (average) for this format.
    #[must_use]
    pub fn bits_per_pixel(&self) -> f32 {
        match self {
            Self::Yuv420p => 12.0,
            Self::Yuv422p => 16.0,
            Self::Yuv444p => 24.0,
            Self::Rgb24 => 24.0,
            Self::Rgba => 32.0,
            Self::Yuv420p10le => 15.0,
            Self::Unknown => 0.0,
        }
    }

    /// Returns `true` if the format contains an alpha channel.
    #[must_use]
    pub fn has_alpha(&self) -> bool {
        matches!(self, Self::Rgba)
    }

    /// Returns the string identifier used in Python.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Yuv420p => "yuv420p",
            Self::Yuv422p => "yuv422p",
            Self::Yuv444p => "yuv444p",
            Self::Rgb24 => "rgb24",
            Self::Rgba => "rgba",
            Self::Yuv420p10le => "yuv420p10le",
            Self::Unknown => "unknown",
        }
    }
}

/// Information about a single decoded video frame.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub pixel_format: PixelFormatBinding,
    /// Presentation timestamp in time-base units.
    pub pts: i64,
    /// Duration of the frame in time-base units.
    pub duration: i64,
    /// Whether this is a key frame.
    pub is_key_frame: bool,
}

impl FrameInfo {
    /// Create a new `FrameInfo`.
    #[must_use]
    pub fn new(width: u32, height: u32, pixel_format: PixelFormatBinding, pts: i64) -> Self {
        Self {
            width,
            height,
            pixel_format,
            pts,
            duration: 0,
            is_key_frame: false,
        }
    }

    /// Compute the frame area in pixels.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Estimated uncompressed size in bytes.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let bpp = self.pixel_format.bits_per_pixel();
        (self.area() as f32 * bpp / 8.0) as u64
    }

    /// Aspect ratio as a float (width / height).
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }
}

/// Codec parameters for a video stream.
#[derive(Clone, Debug, PartialEq)]
pub struct CodecParams {
    /// Codec identifier string (e.g. `"av1"`, `"vp9"`).
    pub codec_id: String,
    /// Bit rate in bits per second.
    pub bit_rate: u64,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frames per second (numerator, denominator).
    pub frame_rate: (u32, u32),
    /// Pixel format.
    pub pixel_format: PixelFormatBinding,
    /// Codec profile (e.g. `"main"`, `"high"`).
    pub profile: Option<String>,
    /// Codec level (e.g. `40` for level 4.0).
    pub level: Option<i32>,
}

impl CodecParams {
    /// Returns the frame rate as a float.
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.frame_rate.1 == 0 {
            0.0
        } else {
            f64::from(self.frame_rate.0) / f64::from(self.frame_rate.1)
        }
    }
}

/// Container-level information about a media file.
#[derive(Clone, Debug, PartialEq)]
pub struct ContainerInfo {
    /// Format name (e.g. `"matroska"`, `"ogg"`).
    pub format_name: String,
    /// Total duration in microseconds.
    pub duration_us: i64,
    /// Overall bit rate in bits per second.
    pub bit_rate: u64,
    /// Number of streams.
    pub stream_count: usize,
    /// Metadata key-value pairs.
    pub metadata: Vec<(String, String)>,
}

impl ContainerInfo {
    /// Duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.duration_us as f64 / 1_000_000.0
    }

    /// Look up a metadata value by key (case-insensitive).
    #[must_use]
    pub fn metadata_get(&self, key: &str) -> Option<&str> {
        let lower = key.to_lowercase();
        self.metadata
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_bits_per_pixel_yuv420p() {
        assert_eq!(PixelFormatBinding::Yuv420p.bits_per_pixel(), 12.0);
    }

    #[test]
    fn test_pixel_format_has_alpha_rgba() {
        assert!(PixelFormatBinding::Rgba.has_alpha());
    }

    #[test]
    fn test_pixel_format_no_alpha_rgb24() {
        assert!(!PixelFormatBinding::Rgb24.has_alpha());
    }

    #[test]
    fn test_pixel_format_as_str() {
        assert_eq!(PixelFormatBinding::Yuv420p.as_str(), "yuv420p");
        assert_eq!(PixelFormatBinding::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_frame_info_area() {
        let f = FrameInfo::new(1920, 1080, PixelFormatBinding::Yuv420p, 0);
        assert_eq!(f.area(), 1920 * 1080);
    }

    #[test]
    fn test_frame_info_byte_size() {
        let f = FrameInfo::new(1920, 1080, PixelFormatBinding::Rgb24, 0);
        // 1920*1080*3
        assert_eq!(f.byte_size(), 1920 * 1080 * 3);
    }

    #[test]
    fn test_frame_info_aspect_ratio() {
        let f = FrameInfo::new(1920, 1080, PixelFormatBinding::Yuv420p, 0);
        let ar = f.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_frame_info_aspect_ratio_zero_height() {
        let f = FrameInfo::new(1920, 0, PixelFormatBinding::Yuv420p, 0);
        assert_eq!(f.aspect_ratio(), 0.0);
    }

    #[test]
    fn test_codec_params_fps() {
        let p = CodecParams {
            codec_id: "av1".to_string(),
            bit_rate: 2_000_000,
            width: 1920,
            height: 1080,
            frame_rate: (30, 1),
            pixel_format: PixelFormatBinding::Yuv420p,
            profile: None,
            level: None,
        };
        assert!((p.fps() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_codec_params_fps_zero_denom() {
        let p = CodecParams {
            codec_id: "av1".to_string(),
            bit_rate: 0,
            width: 0,
            height: 0,
            frame_rate: (30, 0),
            pixel_format: PixelFormatBinding::Unknown,
            profile: None,
            level: None,
        };
        assert_eq!(p.fps(), 0.0);
    }

    #[test]
    fn test_container_info_duration_secs() {
        let c = ContainerInfo {
            format_name: "matroska".to_string(),
            duration_us: 5_000_000,
            bit_rate: 1_000_000,
            stream_count: 2,
            metadata: vec![],
        };
        assert!((c.duration_secs() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_container_info_metadata_get() {
        let c = ContainerInfo {
            format_name: "ogg".to_string(),
            duration_us: 0,
            bit_rate: 0,
            stream_count: 1,
            metadata: vec![("Title".to_string(), "My Video".to_string())],
        };
        assert_eq!(c.metadata_get("title"), Some("My Video"));
        assert_eq!(c.metadata_get("artist"), None);
    }

    #[test]
    fn test_container_info_metadata_case_insensitive() {
        let c = ContainerInfo {
            format_name: "ogg".to_string(),
            duration_us: 0,
            bit_rate: 0,
            stream_count: 1,
            metadata: vec![("TITLE".to_string(), "Test".to_string())],
        };
        assert_eq!(c.metadata_get("title"), Some("Test"));
    }
}

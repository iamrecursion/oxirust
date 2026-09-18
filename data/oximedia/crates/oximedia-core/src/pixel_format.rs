//! Pixel format descriptions independent of the main `types` module.
//!
//! Provides a standalone `PixelFormat` enum and associated `PixelFormatInfo`
//! descriptor for use in pipeline components that do not depend on the full
//! `oximedia_core::types` module.

#![allow(dead_code)]

/// Pixel formats supported by the `OxiMedia` pipeline.
///
/// Only patent-free formats are included; no proprietary chroma subsampling
/// schemes tied to patented codec pipelines are exposed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// YUV 4:2:0 planar (3 planes: Y full, U/V half width/height).
    Yuv420p,
    /// YUV 4:2:2 planar (3 planes: Y full, U/V half width).
    Yuv422p,
    /// YUV 4:4:4 planar (3 planes, all full resolution).
    Yuv444p,
    /// Semi-planar YUV 4:2:0 (2 planes: Y, interleaved UV).
    Nv12,
    /// Packed RGB 24-bit (R, G, B bytes).
    Rgb24,
    /// Packed RGBA 32-bit (R, G, B, A bytes).
    Rgba32,
    /// 10-bit YUV 4:2:0 packed into 16-bit words (2 planes: Y, interleaved UV).
    P010,
}

impl PixelFormat {
    /// Returns the bit depth per component for this format.
    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        match self {
            Self::P010 => 10,
            _ => 8,
        }
    }

    /// Returns the number of planes this format uses.
    #[must_use]
    pub const fn plane_count(&self) -> usize {
        match self {
            Self::Yuv420p | Self::Yuv422p | Self::Yuv444p => 3,
            Self::Nv12 | Self::P010 => 2,
            Self::Rgb24 | Self::Rgba32 => 1,
        }
    }

    /// Returns an approximate number of bytes per pixel (rounded up for
    /// sub-byte and sub-sampled formats).
    ///
    /// This is an approximation; exact sizes depend on stride alignment.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bytes_per_pixel_approx(&self) -> f32 {
        match self {
            Self::Yuv420p | Self::Nv12 => 1.5_f32,
            Self::Yuv422p => 2.0_f32,
            Self::Yuv444p | Self::Rgb24 | Self::P010 => 3.0_f32,
            Self::Rgba32 => 4.0_f32,
        }
    }

    /// Returns `true` if this format uses separate planes for chroma components.
    #[must_use]
    pub const fn is_planar(&self) -> bool {
        matches!(
            self,
            Self::Yuv420p | Self::Yuv422p | Self::Yuv444p | Self::Nv12 | Self::P010
        )
    }

    /// Returns `true` if this format is a YUV variant.
    #[must_use]
    pub const fn is_yuv(&self) -> bool {
        matches!(
            self,
            Self::Yuv420p | Self::Yuv422p | Self::Yuv444p | Self::Nv12 | Self::P010
        )
    }

    /// Returns `true` if this format contains an alpha channel.
    #[must_use]
    pub const fn has_alpha(&self) -> bool {
        matches!(self, Self::Rgba32)
    }

    /// Returns all supported pixel formats as a slice.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Yuv420p,
            Self::Yuv422p,
            Self::Yuv444p,
            Self::Nv12,
            Self::Rgb24,
            Self::Rgba32,
            Self::P010,
        ]
    }
}

/// Detailed information about a pixel format, including human-readable metadata.
#[derive(Debug, Clone)]
pub struct PixelFormatInfo {
    /// The pixel format this info describes.
    pub format: PixelFormat,
    /// Short name string (e.g. `"yuv420p"`).
    pub name: &'static str,
    /// Number of planes.
    pub planes: usize,
    /// Bit depth per component.
    pub bit_depth: u8,
    /// Whether the format uses separate planes.
    pub planar: bool,
}

impl PixelFormatInfo {
    /// Creates a `PixelFormatInfo` for the given format.
    #[must_use]
    pub fn new(format: PixelFormat) -> Self {
        let name = match format {
            PixelFormat::Yuv420p => "yuv420p",
            PixelFormat::Yuv422p => "yuv422p",
            PixelFormat::Yuv444p => "yuv444p",
            PixelFormat::Nv12 => "nv12",
            PixelFormat::Rgb24 => "rgb24",
            PixelFormat::Rgba32 => "rgba",
            PixelFormat::P010 => "p010le",
        };
        Self {
            planes: format.plane_count(),
            bit_depth: format.bit_depth(),
            planar: format.is_planar(),
            format,
            name,
        }
    }

    /// Returns `true` if this format is planar (separate planes per component).
    #[must_use]
    pub fn is_planar(&self) -> bool {
        self.planar
    }

    /// Returns a human-readable name for the pixel format.
    #[must_use]
    pub fn name(&self) -> &str {
        self.name
    }

    /// Estimates the frame buffer size in bytes for given dimensions.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn frame_size_bytes(&self, width: u32, height: u32) -> usize {
        let pixels = width as f32 * height as f32;
        let bpp = self.format.bytes_per_pixel_approx();
        (pixels * bpp) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuv420p_planes() {
        assert_eq!(PixelFormat::Yuv420p.plane_count(), 3);
    }

    #[test]
    fn test_nv12_planes() {
        assert_eq!(PixelFormat::Nv12.plane_count(), 2);
    }

    #[test]
    fn test_rgb24_planes() {
        assert_eq!(PixelFormat::Rgb24.plane_count(), 1);
    }

    #[test]
    fn test_rgba32_planes() {
        assert_eq!(PixelFormat::Rgba32.plane_count(), 1);
    }

    #[test]
    fn test_p010_bit_depth() {
        assert_eq!(PixelFormat::P010.bit_depth(), 10);
    }

    #[test]
    fn test_yuv420p_bit_depth() {
        assert_eq!(PixelFormat::Yuv420p.bit_depth(), 8);
    }

    #[test]
    fn test_is_planar_yuv420p() {
        assert!(PixelFormat::Yuv420p.is_planar());
    }

    #[test]
    fn test_is_planar_rgb24_false() {
        assert!(!PixelFormat::Rgb24.is_planar());
    }

    #[test]
    fn test_is_planar_rgba32_false() {
        assert!(!PixelFormat::Rgba32.is_planar());
    }

    #[test]
    fn test_has_alpha_rgba32() {
        assert!(PixelFormat::Rgba32.has_alpha());
    }

    #[test]
    fn test_has_alpha_rgb24_false() {
        assert!(!PixelFormat::Rgb24.has_alpha());
    }

    #[test]
    fn test_is_yuv() {
        assert!(PixelFormat::Yuv420p.is_yuv());
        assert!(PixelFormat::Nv12.is_yuv());
        assert!(!PixelFormat::Rgb24.is_yuv());
    }

    #[test]
    fn test_bytes_per_pixel_yuv420p() {
        let bpp = PixelFormat::Yuv420p.bytes_per_pixel_approx();
        assert!((bpp - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_bytes_per_pixel_rgba32() {
        let bpp = PixelFormat::Rgba32.bytes_per_pixel_approx();
        assert!((bpp - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_all_formats_count() {
        assert_eq!(PixelFormat::all().len(), 7);
    }

    #[test]
    fn test_pixel_format_info_new_yuv420p() {
        let info = PixelFormatInfo::new(PixelFormat::Yuv420p);
        assert_eq!(info.planes, 3);
        assert!(info.is_planar());
        assert_eq!(info.name(), "yuv420p");
    }

    #[test]
    fn test_pixel_format_info_new_rgba32() {
        let info = PixelFormatInfo::new(PixelFormat::Rgba32);
        assert_eq!(info.planes, 1);
        assert!(!info.is_planar());
        assert_eq!(info.name(), "rgba");
    }

    #[test]
    fn test_frame_size_bytes_yuv420p() {
        let info = PixelFormatInfo::new(PixelFormat::Yuv420p);
        // 1920x1080 @ 1.5 bytes/pixel = 3_110_400
        let size = info.frame_size_bytes(1920, 1080);
        assert_eq!(size, 3_110_400);
    }

    #[test]
    fn test_frame_size_bytes_rgb24() {
        let info = PixelFormatInfo::new(PixelFormat::Rgb24);
        // 1920x1080 @ 3 bytes/pixel = 6_220_800
        let size = info.frame_size_bytes(1920, 1080);
        assert_eq!(size, 6_220_800);
    }

    #[test]
    fn test_equality() {
        assert_eq!(PixelFormat::Yuv420p, PixelFormat::Yuv420p);
        assert_ne!(PixelFormat::Yuv420p, PixelFormat::Yuv422p);
    }
}

//! Thumbnail generation with aspect-ratio-preserving nearest-neighbor scaling.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// Specification for a thumbnail to generate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThumbnailSpec {
    /// Maximum thumbnail width
    pub width: u32,
    /// Maximum thumbnail height
    pub height: u32,
    /// Quality hint (0-100); usage depends on encoder
    pub quality: u8,
}

impl ThumbnailSpec {
    /// Create a `ThumbnailSpec` with explicit width, height, and quality.
    pub fn new(width: u32, height: u32, quality: u8) -> Self {
        Self {
            width,
            height,
            quality,
        }
    }

    /// Create a square `ThumbnailSpec` from a single long-edge size.
    ///
    /// Both width and height are set to `size`, with a default quality of 85.
    pub fn from_long_edge(size: u32) -> Self {
        Self {
            width: size,
            height: size,
            quality: 85,
        }
    }

    /// Compute the output dimensions that fit within this spec while
    /// preserving the aspect ratio of the source image.
    pub fn fit_dimensions(&self, src_w: u32, src_h: u32) -> (u32, u32) {
        if src_w == 0 || src_h == 0 || self.width == 0 || self.height == 0 {
            return (0, 0);
        }
        let scale_w = self.width as f64 / src_w as f64;
        let scale_h = self.height as f64 / src_h as f64;
        let scale = scale_w.min(scale_h).min(1.0); // Never upscale
        let out_w = ((src_w as f64 * scale).round() as u32).max(1);
        let out_h = ((src_h as f64 * scale).round() as u32).max(1);
        (out_w, out_h)
    }
}

impl Default for ThumbnailSpec {
    fn default() -> Self {
        Self::from_long_edge(256)
    }
}

/// The result of a thumbnail generation operation.
#[derive(Debug, Clone)]
pub struct ThumbnailResult {
    /// Width of the generated thumbnail
    pub width: u32,
    /// Height of the generated thumbnail
    pub height: u32,
    /// Raw pixel data (grayscale, 1 byte per pixel)
    pub data: Vec<u8>,
}

impl ThumbnailResult {
    /// Return the total number of pixels.
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Return the aspect ratio (width / height) as an `f32`.
    ///
    /// Returns 0.0 if height is zero.
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }
}

/// Generate a thumbnail from grayscale pixel data using nearest-neighbor scaling.
///
/// The thumbnail fits within the dimensions specified by `spec` while preserving
/// the source aspect ratio. The image will never be upscaled.
///
/// # Arguments
/// - `pixels`: Source pixel data (grayscale, 1 byte per pixel, row-major)
/// - `src_w`: Source image width
/// - `src_h`: Source image height
/// - `spec`: Thumbnail specification
pub fn generate_thumbnail(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    spec: &ThumbnailSpec,
) -> ThumbnailResult {
    let (dst_w, dst_h) = spec.fit_dimensions(src_w, src_h);

    if dst_w == 0 || dst_h == 0 || pixels.is_empty() {
        return ThumbnailResult {
            width: 0,
            height: 0,
            data: Vec::new(),
        };
    }

    let mut data = vec![0u8; (dst_w * dst_h) as usize];

    for dy in 0..dst_h {
        // Map destination row to nearest source row
        let sy = (dy * src_h / dst_h).min(src_h - 1);
        for dx in 0..dst_w {
            // Map destination column to nearest source column
            let sx = (dx * src_w / dst_w).min(src_w - 1);
            let src_idx = (sy * src_w + sx) as usize;
            let dst_idx = (dy * dst_w + dx) as usize;
            data[dst_idx] = if src_idx < pixels.len() {
                pixels[src_idx]
            } else {
                0
            };
        }
    }

    ThumbnailResult {
        width: dst_w,
        height: dst_h,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_new() {
        let s = ThumbnailSpec::new(320, 240, 90);
        assert_eq!(s.width, 320);
        assert_eq!(s.height, 240);
        assert_eq!(s.quality, 90);
    }

    #[test]
    fn test_spec_from_long_edge() {
        let s = ThumbnailSpec::from_long_edge(128);
        assert_eq!(s.width, 128);
        assert_eq!(s.height, 128);
        assert_eq!(s.quality, 85);
    }

    #[test]
    fn test_spec_default() {
        let s = ThumbnailSpec::default();
        assert_eq!(s.width, 256);
        assert_eq!(s.height, 256);
    }

    #[test]
    fn test_fit_dimensions_no_upscale() {
        let s = ThumbnailSpec::new(512, 512, 80);
        // Source is 100x100 which fits without scaling
        let (w, h) = s.fit_dimensions(100, 100);
        assert_eq!((w, h), (100, 100));
    }

    #[test]
    fn test_fit_dimensions_square_source() {
        let s = ThumbnailSpec::new(128, 128, 80);
        let (w, h) = s.fit_dimensions(512, 512);
        assert_eq!((w, h), (128, 128));
    }

    #[test]
    fn test_fit_dimensions_landscape() {
        let s = ThumbnailSpec::new(128, 128, 80);
        // 320x240 source -> scale to fit 128x128
        let (w, h) = s.fit_dimensions(320, 240);
        // scale = 128/320 = 0.4
        assert_eq!(w, 128);
        assert_eq!(h, 96);
    }

    #[test]
    fn test_fit_dimensions_portrait() {
        let s = ThumbnailSpec::new(128, 128, 80);
        let (w, h) = s.fit_dimensions(240, 320);
        assert_eq!(w, 96);
        assert_eq!(h, 128);
    }

    #[test]
    fn test_fit_dimensions_zero_source() {
        let s = ThumbnailSpec::new(128, 128, 80);
        let (w, h) = s.fit_dimensions(0, 100);
        assert_eq!((w, h), (0, 0));
    }

    #[test]
    fn test_result_pixel_count() {
        let r = ThumbnailResult {
            width: 10,
            height: 20,
            data: vec![0u8; 200],
        };
        assert_eq!(r.pixel_count(), 200);
    }

    #[test]
    fn test_result_aspect_ratio() {
        let r = ThumbnailResult {
            width: 16,
            height: 9,
            data: vec![0u8; 144],
        };
        let ratio = r.aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_result_aspect_ratio_zero_height() {
        let r = ThumbnailResult {
            width: 10,
            height: 0,
            data: Vec::new(),
        };
        assert_eq!(r.aspect_ratio(), 0.0);
    }

    #[test]
    fn test_generate_thumbnail_basic() {
        let pixels: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let spec = ThumbnailSpec::new(8, 8, 80);
        let result = generate_thumbnail(&pixels, 16, 16, &spec);
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        assert_eq!(result.data.len(), 64);
    }

    #[test]
    fn test_generate_thumbnail_empty_input() {
        let spec = ThumbnailSpec::new(64, 64, 80);
        let result = generate_thumbnail(&[], 0, 0, &spec);
        assert_eq!(result.width, 0);
        assert_eq!(result.height, 0);
        assert!(result.data.is_empty());
    }

    #[test]
    fn test_generate_thumbnail_preserves_aspect() {
        let pixels = vec![128u8; 640 * 480];
        let spec = ThumbnailSpec::new(100, 100, 80);
        let result = generate_thumbnail(&pixels, 640, 480, &spec);
        // Should fit within 100x100, preserving 4:3 ratio
        assert!(result.width <= 100);
        assert!(result.height <= 100);
        // Ratio should be close to 4:3
        let ratio = result.width as f32 / result.height as f32;
        assert!((ratio - 640.0 / 480.0).abs() < 0.1, "aspect ratio: {ratio}");
    }

    #[test]
    fn test_generate_thumbnail_correct_size() {
        let pixels = vec![255u8; 32 * 32];
        let spec = ThumbnailSpec::new(16, 16, 80);
        let result = generate_thumbnail(&pixels, 32, 32, &spec);
        assert_eq!(
            result.pixel_count(),
            (result.width * result.height) as usize
        );
    }
}

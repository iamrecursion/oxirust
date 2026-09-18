#![allow(dead_code)]
//! Region-of-interest (ROI) scaling
//!
//! Provides functionality to crop a rectangular region of interest from
//! a source frame and scale it to a target resolution. Useful for
//! pan-and-scan, smart zoom, face-tracking zoom, and detail extraction.

use std::fmt;

/// A rectangular region in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoiRect {
    /// X offset of the top-left corner.
    pub x: u32,
    /// Y offset of the top-left corner.
    pub y: u32,
    /// Width of the region.
    pub width: u32,
    /// Height of the region.
    pub height: u32,
}

impl RoiRect {
    /// Create a new ROI rectangle.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Area of the rectangle in pixels.
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Right edge coordinate (exclusive).
    pub fn right(&self) -> u32 {
        self.x + self.width
    }

    /// Bottom edge coordinate (exclusive).
    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }

    /// Whether this ROI fits within a frame of the given dimensions.
    pub fn fits_in(&self, frame_width: u32, frame_height: u32) -> bool {
        self.right() <= frame_width && self.bottom() <= frame_height
    }

    /// Clamp the ROI so it fits within the given frame dimensions.
    pub fn clamp_to(&self, frame_width: u32, frame_height: u32) -> Self {
        let x = self.x.min(frame_width.saturating_sub(1));
        let y = self.y.min(frame_height.saturating_sub(1));
        let w = self.width.min(frame_width.saturating_sub(x));
        let h = self.height.min(frame_height.saturating_sub(y));
        Self {
            x,
            y,
            width: w,
            height: h,
        }
    }

    /// Compute the center point of the ROI.
    #[allow(clippy::cast_precision_loss)]
    pub fn center(&self) -> (f64, f64) {
        (
            self.x as f64 + self.width as f64 / 2.0,
            self.y as f64 + self.height as f64 / 2.0,
        )
    }

    /// Compute the aspect ratio (width / height).
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        self.width as f64 / self.height as f64
    }
}

impl fmt::Display for RoiRect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}+{}+{}", self.width, self.height, self.x, self.y)
    }
}

/// Configuration for ROI scaling.
#[derive(Debug, Clone)]
pub struct RoiScaleConfig {
    /// The ROI to extract from the source.
    pub roi: RoiRect,
    /// Target output width.
    pub dst_width: u32,
    /// Target output height.
    pub dst_height: u32,
    /// Whether to apply smoothing after scale.
    pub smooth: bool,
}

impl RoiScaleConfig {
    /// Create a new ROI scale config.
    pub fn new(roi: RoiRect, dst_width: u32, dst_height: u32) -> Self {
        Self {
            roi,
            dst_width,
            dst_height,
            smooth: false,
        }
    }

    /// Enable or disable smoothing.
    pub fn with_smooth(mut self, smooth: bool) -> Self {
        self.smooth = smooth;
        self
    }

    /// Compute the horizontal scale factor.
    #[allow(clippy::cast_precision_loss)]
    pub fn scale_x(&self) -> f64 {
        if self.roi.width == 0 {
            return 1.0;
        }
        self.dst_width as f64 / self.roi.width as f64
    }

    /// Compute the vertical scale factor.
    #[allow(clippy::cast_precision_loss)]
    pub fn scale_y(&self) -> f64 {
        if self.roi.height == 0 {
            return 1.0;
        }
        self.dst_height as f64 / self.roi.height as f64
    }
}

/// Extract a rectangular ROI from a grayscale frame buffer.
///
/// `frame` is row-major with `stride` bytes per row, 1 byte per pixel.
pub fn extract_roi(frame: &[u8], stride: u32, roi: &RoiRect) -> Vec<u8> {
    let mut out = Vec::with_capacity((roi.width * roi.height) as usize);
    for row in 0..roi.height {
        let y = roi.y + row;
        let start = (y * stride + roi.x) as usize;
        let end = start + roi.width as usize;
        if end <= frame.len() {
            out.extend_from_slice(&frame[start..end]);
        }
    }
    out
}

/// Scale a grayscale buffer using bilinear interpolation.
#[allow(clippy::cast_precision_loss)]
pub fn scale_bilinear(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_width * dst_height) as usize];
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return dst;
    }
    let x_ratio = if dst_width > 1 {
        (src_width as f64 - 1.0) / (dst_width as f64 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if dst_height > 1 {
        (src_height as f64 - 1.0) / (dst_height as f64 - 1.0)
    } else {
        0.0
    };

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let sx = x_ratio * dx as f64;
            let sy = y_ratio * dy as f64;
            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);
            let xf = sx - sx.floor();
            let yf = sy - sy.floor();

            let px = |x: u32, y: u32| -> f64 {
                let i = (y * src_width + x) as usize;
                if i < src.len() {
                    src[i] as f64
                } else {
                    0.0
                }
            };

            let val = px(x0, y0) * (1.0 - xf) * (1.0 - yf)
                + px(x1, y0) * xf * (1.0 - yf)
                + px(x0, y1) * (1.0 - xf) * yf
                + px(x1, y1) * xf * yf;

            dst[(dy * dst_width + dx) as usize] = val.round().clamp(0.0, 255.0) as u8;
        }
    }
    dst
}

/// Perform ROI extraction and scaling in one step.
pub fn roi_scale(frame: &[u8], frame_width: u32, config: &RoiScaleConfig) -> Vec<u8> {
    let roi_data = extract_roi(frame, frame_width, &config.roi);
    scale_bilinear(
        &roi_data,
        config.roi.width,
        config.roi.height,
        config.dst_width,
        config.dst_height,
    )
}

/// Smoothly interpolate between two ROI positions (for animated pan-and-scan).
#[allow(clippy::cast_precision_loss)]
pub fn interpolate_roi(a: &RoiRect, b: &RoiRect, t: f64) -> RoiRect {
    let t = t.clamp(0.0, 1.0);
    let lerp = |v0: u32, v1: u32| -> u32 {
        let result = v0 as f64 * (1.0 - t) + v1 as f64 * t;
        result.round().max(0.0) as u32
    };
    RoiRect {
        x: lerp(a.x, b.x),
        y: lerp(a.y, b.y),
        width: lerp(a.width, b.width),
        height: lerp(a.height, b.height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roi_rect_creation() {
        let r = RoiRect::new(10, 20, 100, 50);
        assert_eq!(r.x, 10);
        assert_eq!(r.y, 20);
        assert_eq!(r.width, 100);
        assert_eq!(r.height, 50);
    }

    #[test]
    fn test_roi_area() {
        let r = RoiRect::new(0, 0, 1920, 1080);
        assert_eq!(r.area(), 2_073_600);
    }

    #[test]
    fn test_roi_right_bottom() {
        let r = RoiRect::new(10, 20, 100, 50);
        assert_eq!(r.right(), 110);
        assert_eq!(r.bottom(), 70);
    }

    #[test]
    fn test_roi_fits_in() {
        let r = RoiRect::new(10, 20, 100, 50);
        assert!(r.fits_in(200, 200));
        assert!(!r.fits_in(50, 50));
    }

    #[test]
    fn test_roi_clamp_to() {
        let r = RoiRect::new(100, 100, 200, 200);
        let clamped = r.clamp_to(150, 150);
        assert_eq!(clamped.right(), 150);
        assert_eq!(clamped.bottom(), 150);
    }

    #[test]
    fn test_roi_center() {
        let r = RoiRect::new(0, 0, 100, 100);
        let (cx, cy) = r.center();
        assert!((cx - 50.0).abs() < f64::EPSILON);
        assert!((cy - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_roi_aspect_ratio() {
        let r = RoiRect::new(0, 0, 1920, 1080);
        let ar = r.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_roi_display() {
        let r = RoiRect::new(10, 20, 100, 50);
        assert_eq!(r.to_string(), "100x50+10+20");
    }

    #[test]
    fn test_config_scale_factors() {
        let roi = RoiRect::new(0, 0, 100, 100);
        let cfg = RoiScaleConfig::new(roi, 200, 200);
        assert!((cfg.scale_x() - 2.0).abs() < f64::EPSILON);
        assert!((cfg.scale_y() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extract_roi() {
        // 4x4 frame
        let frame: Vec<u8> = (0..16).collect();
        let roi = RoiRect::new(1, 1, 2, 2);
        let extracted = extract_roi(&frame, 4, &roi);
        // Row 1: [4,5,6,7] → cols 1..3 → [5,6]
        // Row 2: [8,9,10,11] → cols 1..3 → [9,10]
        assert_eq!(extracted, vec![5, 6, 9, 10]);
    }

    #[test]
    fn test_scale_identity() {
        let src = vec![10, 20, 30, 40];
        let result = scale_bilinear(&src, 2, 2, 2, 2);
        assert_eq!(result, src);
    }

    #[test]
    fn test_roi_scale_combined() {
        // 8x8 frame, extract 4x4 ROI and scale to 2x2
        let frame: Vec<u8> = vec![128; 64];
        let roi = RoiRect::new(2, 2, 4, 4);
        let cfg = RoiScaleConfig::new(roi, 2, 2);
        let result = roi_scale(&frame, 8, &cfg);
        assert_eq!(result.len(), 4);
        for &v in &result {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_interpolate_roi_start() {
        let a = RoiRect::new(0, 0, 100, 100);
        let b = RoiRect::new(200, 200, 50, 50);
        let result = interpolate_roi(&a, &b, 0.0);
        assert_eq!(result, a);
    }

    #[test]
    fn test_interpolate_roi_end() {
        let a = RoiRect::new(0, 0, 100, 100);
        let b = RoiRect::new(200, 200, 50, 50);
        let result = interpolate_roi(&a, &b, 1.0);
        assert_eq!(result, b);
    }

    #[test]
    fn test_interpolate_roi_mid() {
        let a = RoiRect::new(0, 0, 100, 100);
        let b = RoiRect::new(100, 100, 200, 200);
        let result = interpolate_roi(&a, &b, 0.5);
        assert_eq!(result.x, 50);
        assert_eq!(result.y, 50);
        assert_eq!(result.width, 150);
        assert_eq!(result.height, 150);
    }
}

//! Combined crop-and-scale operations for video frames.
//!
//! Crops a region of interest from a source frame and scales it to a target
//! resolution in a single pass. Supports center-crop, rule-of-thirds crop,
//! and arbitrary ROI with clamping.

#![allow(dead_code)]

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Strategy used when choosing the crop region automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropStrategy {
    /// Crop from the center of the frame.
    Center,
    /// Apply the rule-of-thirds and pick the most interesting quadrant.
    RuleOfThirds,
    /// Use an explicit region of interest.
    Manual,
}

impl fmt::Display for CropStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Center => write!(f, "center"),
            Self::RuleOfThirds => write!(f, "rule_of_thirds"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

/// A rectangular region in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRect {
    /// Left edge (inclusive).
    pub x: u32,
    /// Top edge (inclusive).
    pub y: u32,
    /// Width of the crop region.
    pub w: u32,
    /// Height of the crop region.
    pub h: u32,
}

impl CropRect {
    /// Create a new crop rectangle.
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// Area in pixels.
    pub fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    /// Aspect ratio as width / height.
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect_ratio(&self) -> f64 {
        if self.h == 0 {
            return 0.0;
        }
        self.w as f64 / self.h as f64
    }

    /// Right edge (exclusive).
    pub fn right(&self) -> u32 {
        self.x + self.w
    }

    /// Bottom edge (exclusive).
    pub fn bottom(&self) -> u32 {
        self.y + self.h
    }

    /// Clamp the rect to fit within the given frame dimensions.
    pub fn clamp(&self, frame_w: u32, frame_h: u32) -> Self {
        let x = self.x.min(frame_w.saturating_sub(1));
        let y = self.y.min(frame_h.saturating_sub(1));
        let w = self.w.min(frame_w.saturating_sub(x));
        let h = self.h.min(frame_h.saturating_sub(y));
        Self { x, y, w, h }
    }

    /// Check if the rect is fully contained within frame dimensions.
    pub fn fits(&self, frame_w: u32, frame_h: u32) -> bool {
        self.right() <= frame_w && self.bottom() <= frame_h
    }
}

/// Parameters for a crop-and-scale operation.
#[derive(Debug, Clone)]
pub struct CropScaleParams {
    /// Source frame width.
    pub src_width: u32,
    /// Source frame height.
    pub src_height: u32,
    /// Target output width.
    pub dst_width: u32,
    /// Target output height.
    pub dst_height: u32,
    /// Crop strategy to use.
    pub strategy: CropStrategy,
    /// Manual crop rect (only used when strategy is Manual).
    pub manual_rect: Option<CropRect>,
}

impl CropScaleParams {
    /// Create new parameters.
    pub fn new(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Self {
        Self {
            src_width: src_w,
            src_height: src_h,
            dst_width: dst_w,
            dst_height: dst_h,
            strategy: CropStrategy::Center,
            manual_rect: None,
        }
    }

    /// Set the crop strategy.
    pub fn with_strategy(mut self, s: CropStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Set a manual crop rectangle.
    pub fn with_rect(mut self, rect: CropRect) -> Self {
        self.strategy = CropStrategy::Manual;
        self.manual_rect = Some(rect);
        self
    }
}

/// Compute the crop rectangle for the given parameters.
#[allow(clippy::cast_precision_loss)]
pub fn compute_crop_rect(params: &CropScaleParams) -> CropRect {
    match params.strategy {
        CropStrategy::Manual => {
            if let Some(r) = &params.manual_rect {
                r.clamp(params.src_width, params.src_height)
            } else {
                // Fallback to full-frame
                CropRect::new(0, 0, params.src_width, params.src_height)
            }
        }
        CropStrategy::Center => {
            let dst_ar = params.dst_width as f64 / params.dst_height.max(1) as f64;
            let src_ar = params.src_width as f64 / params.src_height.max(1) as f64;

            let (cw, ch) = if src_ar > dst_ar {
                // Source wider: fit height, crop width
                let cw = (params.src_height as f64 * dst_ar) as u32;
                (cw.min(params.src_width), params.src_height)
            } else {
                // Source taller: fit width, crop height
                let ch = (params.src_width as f64 / dst_ar) as u32;
                (params.src_width, ch.min(params.src_height))
            };
            let x = (params.src_width.saturating_sub(cw)) / 2;
            let y = (params.src_height.saturating_sub(ch)) / 2;
            CropRect::new(x, y, cw, ch)
        }
        CropStrategy::RuleOfThirds => {
            // Place the crop at the left-third, top-third intersection
            let dst_ar = params.dst_width as f64 / params.dst_height.max(1) as f64;
            let src_ar = params.src_width as f64 / params.src_height.max(1) as f64;
            let (cw, ch) = if src_ar > dst_ar {
                let cw = (params.src_height as f64 * dst_ar) as u32;
                (cw.min(params.src_width), params.src_height)
            } else {
                let ch = (params.src_width as f64 / dst_ar) as u32;
                (params.src_width, ch.min(params.src_height))
            };
            let x = (params.src_width.saturating_sub(cw)) / 3;
            let y = (params.src_height.saturating_sub(ch)) / 3;
            CropRect::new(x, y, cw, ch).clamp(params.src_width, params.src_height)
        }
    }
}

/// Scale factor required to go from crop rect to destination dimensions.
#[allow(clippy::cast_precision_loss)]
pub fn compute_scale_factors(crop: &CropRect, dst_w: u32, dst_h: u32) -> (f64, f64) {
    let sx = dst_w as f64 / crop.w.max(1) as f64;
    let sy = dst_h as f64 / crop.h.max(1) as f64;
    (sx, sy)
}

/// Perform a nearest-neighbor crop+scale on a single-channel u8 buffer.
///
/// `src` is row-major, `src_w x src_h`.
/// Returns a buffer of size `dst_w * dst_h`.
#[allow(clippy::cast_precision_loss)]
pub fn crop_scale_nearest(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    crop: &CropRect,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; (dst_w * dst_h) as usize];
    let (sx, sy) = compute_scale_factors(crop, dst_w, dst_h);

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let cx = crop.x + (dx as f64 / sx) as u32;
            let cy = crop.y + (dy as f64 / sy) as u32;
            let cx = cx.min(src_w.saturating_sub(1));
            let cy = cy.min(src_h.saturating_sub(1));
            out[(dy * dst_w + dx) as usize] = src[(cy * src_w + cx) as usize];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_rect_area() {
        let r = CropRect::new(0, 0, 100, 200);
        assert_eq!(r.area(), 20_000);
    }

    #[test]
    fn test_crop_rect_aspect_ratio() {
        let r = CropRect::new(0, 0, 1920, 1080);
        let ar = r.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_crop_rect_zero_height() {
        let r = CropRect::new(0, 0, 100, 0);
        assert_eq!(r.aspect_ratio(), 0.0);
    }

    #[test]
    fn test_crop_rect_right_bottom() {
        let r = CropRect::new(10, 20, 100, 50);
        assert_eq!(r.right(), 110);
        assert_eq!(r.bottom(), 70);
    }

    #[test]
    fn test_crop_rect_clamp() {
        let r = CropRect::new(1800, 900, 300, 300);
        let c = r.clamp(1920, 1080);
        assert!(c.right() <= 1920);
        assert!(c.bottom() <= 1080);
    }

    #[test]
    fn test_crop_rect_fits() {
        let r = CropRect::new(0, 0, 1920, 1080);
        assert!(r.fits(1920, 1080));
        assert!(!r.fits(1280, 720));
    }

    #[test]
    fn test_center_crop_16_9_to_9_16() {
        let params = CropScaleParams::new(1920, 1080, 1080, 1920);
        let rect = compute_crop_rect(&params);
        // Crop should be taller than wide since destination is portrait
        assert!(rect.w <= rect.h);
        assert!(rect.fits(1920, 1080));
    }

    #[test]
    fn test_center_crop_same_aspect() {
        let params = CropScaleParams::new(1920, 1080, 960, 540);
        let rect = compute_crop_rect(&params);
        assert_eq!(rect.x, 0);
        assert_eq!(rect.y, 0);
        assert_eq!(rect.w, 1920);
        assert_eq!(rect.h, 1080);
    }

    #[test]
    fn test_manual_crop() {
        let params =
            CropScaleParams::new(1920, 1080, 640, 480).with_rect(CropRect::new(100, 100, 800, 600));
        let rect = compute_crop_rect(&params);
        assert_eq!(rect.x, 100);
        assert_eq!(rect.y, 100);
        assert_eq!(rect.w, 800);
        assert_eq!(rect.h, 600);
    }

    #[test]
    fn test_rule_of_thirds_crop() {
        let params =
            CropScaleParams::new(1920, 1080, 1080, 1080).with_strategy(CropStrategy::RuleOfThirds);
        let rect = compute_crop_rect(&params);
        assert!(rect.fits(1920, 1080));
    }

    #[test]
    fn test_scale_factors() {
        let crop = CropRect::new(0, 0, 1920, 1080);
        let (sx, sy) = compute_scale_factors(&crop, 960, 540);
        assert!((sx - 0.5).abs() < 0.001);
        assert!((sy - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_crop_scale_nearest_identity() {
        // 4x4 image, full crop, scale to same size
        let src: Vec<u8> = (0..16).collect();
        let crop = CropRect::new(0, 0, 4, 4);
        let out = crop_scale_nearest(&src, 4, 4, &crop, 4, 4);
        assert_eq!(out, src);
    }

    #[test]
    fn test_crop_scale_nearest_downscale() {
        // 4x4 to 2x2
        let src: Vec<u8> = (0..16).collect();
        let crop = CropRect::new(0, 0, 4, 4);
        let out = crop_scale_nearest(&src, 4, 4, &crop, 2, 2);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_crop_strategy_display() {
        assert_eq!(CropStrategy::Center.to_string(), "center");
        assert_eq!(CropStrategy::RuleOfThirds.to_string(), "rule_of_thirds");
        assert_eq!(CropStrategy::Manual.to_string(), "manual");
    }

    #[test]
    fn test_params_builder() {
        let p =
            CropScaleParams::new(3840, 2160, 1920, 1080).with_strategy(CropStrategy::RuleOfThirds);
        assert_eq!(p.strategy, CropStrategy::RuleOfThirds);
        assert_eq!(p.dst_width, 1920);
    }
}

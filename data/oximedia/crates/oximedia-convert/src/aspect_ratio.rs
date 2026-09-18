#![allow(dead_code)]
//! Aspect ratio management for media conversion.
//!
//! This module handles aspect ratio detection, conversion, padding, and
//! cropping strategies. It supports standard ratios (16:9, 4:3, 21:9, etc.),
//! display aspect ratio vs storage aspect ratio, and provides algorithms
//! for letterboxing, pillarboxing, and center-crop transformations.

use std::fmt;

/// A rational aspect ratio represented as width:height.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AspectRatio {
    /// Width component.
    pub width: u32,
    /// Height component.
    pub height: u32,
}

impl AspectRatio {
    /// Create a new aspect ratio from width and height.
    ///
    /// The ratio is automatically reduced to its simplest form.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let g = gcd(width, height);
        let (w, h) = (
            width.checked_div(g).unwrap_or(width),
            height.checked_div(g).unwrap_or(height),
        );
        Self {
            width: w,
            height: h,
        }
    }

    /// Compute the floating-point ratio (width / height).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        f64::from(self.width) / f64::from(self.height)
    }

    /// Check if this ratio is wider than another.
    #[must_use]
    pub fn is_wider_than(&self, other: &Self) -> bool {
        self.as_f64() > other.as_f64()
    }

    /// Check if this ratio is taller (narrower) than another.
    #[must_use]
    pub fn is_taller_than(&self, other: &Self) -> bool {
        self.as_f64() < other.as_f64()
    }

    /// Return the closest standard aspect ratio.
    #[must_use]
    pub fn closest_standard(&self) -> StandardRatio {
        let r = self.as_f64();
        let standards = [
            (StandardRatio::Ratio4x3, 4.0 / 3.0),
            (StandardRatio::Ratio16x9, 16.0 / 9.0),
            (StandardRatio::Ratio21x9, 21.0 / 9.0),
            (StandardRatio::Ratio1x1, 1.0),
            (StandardRatio::Ratio9x16, 9.0 / 16.0),
            (StandardRatio::Ratio2_39x1, 2.39),
            (StandardRatio::Ratio1_85x1, 1.85),
        ];
        let mut best = StandardRatio::Ratio16x9;
        let mut best_diff = f64::MAX;
        for &(std, val) in &standards {
            let diff = (r - val).abs();
            if diff < best_diff {
                best_diff = diff;
                best = std;
            }
        }
        best
    }
}

impl fmt::Display for AspectRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.width, self.height)
    }
}

/// Standard aspect ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardRatio {
    /// 4:3 (SD television, classic).
    Ratio4x3,
    /// 16:9 (HD widescreen).
    Ratio16x9,
    /// 21:9 (ultra-widescreen cinema).
    Ratio21x9,
    /// 1:1 (square, social media).
    Ratio1x1,
    /// 9:16 (vertical / portrait).
    Ratio9x16,
    /// 2.39:1 (anamorphic widescreen).
    Ratio2_39x1,
    /// 1.85:1 (flat widescreen cinema).
    Ratio1_85x1,
}

impl StandardRatio {
    /// Convert to an `AspectRatio`.
    #[must_use]
    pub fn to_aspect_ratio(self) -> AspectRatio {
        match self {
            Self::Ratio4x3 => AspectRatio::new(4, 3),
            Self::Ratio16x9 => AspectRatio::new(16, 9),
            Self::Ratio21x9 => AspectRatio::new(21, 9),
            Self::Ratio1x1 => AspectRatio::new(1, 1),
            Self::Ratio9x16 => AspectRatio::new(9, 16),
            Self::Ratio2_39x1 => AspectRatio::new(239, 100),
            Self::Ratio1_85x1 => AspectRatio::new(185, 100),
        }
    }
}

/// Strategy for adapting content to a different aspect ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptStrategy {
    /// Add black bars (letterbox or pillarbox).
    Pad,
    /// Crop the content to fill the target.
    Crop,
    /// Stretch/squash (distort) to fit exactly.
    Stretch,
    /// Fit inside the target, preserving ratio, no padding.
    FitInside,
}

/// Padding result describing how many pixels of padding are needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadResult {
    /// Padding on the left.
    pub left: u32,
    /// Padding on the right.
    pub right: u32,
    /// Padding on the top.
    pub top: u32,
    /// Padding on the bottom.
    pub bottom: u32,
    /// Final output width.
    pub output_width: u32,
    /// Final output height.
    pub output_height: u32,
}

/// Crop result describing the cropping region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropResult {
    /// X offset of the crop region.
    pub x: u32,
    /// Y offset of the crop region.
    pub y: u32,
    /// Width of the crop region.
    pub crop_width: u32,
    /// Height of the crop region.
    pub crop_height: u32,
}

/// Compute the greatest common divisor of two numbers.
fn gcd(a: u32, b: u32) -> u32 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Detect the aspect ratio of a given resolution.
#[must_use]
pub fn detect_ratio(width: u32, height: u32) -> AspectRatio {
    AspectRatio::new(width, height)
}

/// Compute padding needed to fit `src` into `target` dimensions while preserving
/// the source aspect ratio.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn compute_padding(
    src_width: u32,
    src_height: u32,
    target_width: u32,
    target_height: u32,
) -> PadResult {
    let src_ratio = f64::from(src_width) / f64::from(src_height.max(1));
    let target_ratio = f64::from(target_width) / f64::from(target_height.max(1));

    if (src_ratio - target_ratio).abs() < 1e-6 {
        // Same ratio, no padding
        return PadResult {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
            output_width: target_width,
            output_height: target_height,
        };
    }

    if src_ratio > target_ratio {
        // Source is wider: letterbox (add top/bottom)
        let scaled_height = (f64::from(target_width) / src_ratio).round() as u32;
        let total_pad = target_height.saturating_sub(scaled_height);
        let top = total_pad / 2;
        let bottom = total_pad - top;
        PadResult {
            left: 0,
            right: 0,
            top,
            bottom,
            output_width: target_width,
            output_height: target_height,
        }
    } else {
        // Source is taller: pillarbox (add left/right)
        let scaled_width = (f64::from(target_height) * src_ratio).round() as u32;
        let total_pad = target_width.saturating_sub(scaled_width);
        let left = total_pad / 2;
        let right = total_pad - left;
        PadResult {
            left,
            right,
            top: 0,
            bottom: 0,
            output_width: target_width,
            output_height: target_height,
        }
    }
}

/// Compute the crop region to fill `target` from `src` while preserving
/// the target aspect ratio.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn compute_crop(
    src_width: u32,
    src_height: u32,
    target_width: u32,
    target_height: u32,
) -> CropResult {
    let target_ratio = f64::from(target_width) / f64::from(target_height.max(1));
    let src_ratio = f64::from(src_width) / f64::from(src_height.max(1));

    if src_ratio > target_ratio {
        // Source is wider: crop sides
        let crop_w = (f64::from(src_height) * target_ratio).round() as u32;
        let crop_w = crop_w.min(src_width);
        let x = (src_width - crop_w) / 2;
        CropResult {
            x,
            y: 0,
            crop_width: crop_w,
            crop_height: src_height,
        }
    } else {
        // Source is taller: crop top/bottom
        let crop_h = (f64::from(src_width) / target_ratio).round() as u32;
        let crop_h = crop_h.min(src_height);
        let y = (src_height - crop_h) / 2;
        CropResult {
            x: 0,
            y,
            crop_width: src_width,
            crop_height: crop_h,
        }
    }
}

/// Calculate the output resolution when scaling to fit inside a bounding box
/// while preserving aspect ratio.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn fit_inside(src_width: u32, src_height: u32, max_width: u32, max_height: u32) -> (u32, u32) {
    let src_ratio = f64::from(src_width) / f64::from(src_height.max(1));
    let box_ratio = f64::from(max_width) / f64::from(max_height.max(1));

    if src_ratio > box_ratio {
        // Width is the constraint
        let w = max_width;
        let h = (f64::from(max_width) / src_ratio).round() as u32;
        (w, h.max(1))
    } else {
        // Height is the constraint
        let h = max_height;
        let w = (f64::from(max_height) * src_ratio).round() as u32;
        (w.max(1), h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aspect_ratio_new_reduces() {
        let ar = AspectRatio::new(1920, 1080);
        assert_eq!(ar.width, 16);
        assert_eq!(ar.height, 9);
    }

    #[test]
    fn test_aspect_ratio_as_f64() {
        let ar = AspectRatio::new(16, 9);
        assert!((ar.as_f64() - 16.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_aspect_ratio_display() {
        let ar = AspectRatio::new(4, 3);
        assert_eq!(format!("{ar}"), "4:3");
    }

    #[test]
    fn test_is_wider_than() {
        let wide = AspectRatio::new(16, 9);
        let narrow = AspectRatio::new(4, 3);
        assert!(wide.is_wider_than(&narrow));
        assert!(!narrow.is_wider_than(&wide));
    }

    #[test]
    fn test_is_taller_than() {
        let portrait = AspectRatio::new(9, 16);
        let landscape = AspectRatio::new(16, 9);
        assert!(portrait.is_taller_than(&landscape));
    }

    #[test]
    fn test_closest_standard_16x9() {
        let ar = AspectRatio::new(1920, 1080);
        assert_eq!(ar.closest_standard(), StandardRatio::Ratio16x9);
    }

    #[test]
    fn test_closest_standard_4x3() {
        let ar = AspectRatio::new(640, 480);
        assert_eq!(ar.closest_standard(), StandardRatio::Ratio4x3);
    }

    #[test]
    fn test_closest_standard_1x1() {
        let ar = AspectRatio::new(1080, 1080);
        assert_eq!(ar.closest_standard(), StandardRatio::Ratio1x1);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(1920, 1080), 120);
        assert_eq!(gcd(100, 0), 100);
        assert_eq!(gcd(7, 13), 1);
    }

    #[test]
    fn test_detect_ratio() {
        let ar = detect_ratio(3840, 2160);
        assert_eq!(ar.width, 16);
        assert_eq!(ar.height, 9);
    }

    #[test]
    fn test_compute_padding_same_ratio() {
        let result = compute_padding(1920, 1080, 1280, 720);
        assert_eq!(result.top, 0);
        assert_eq!(result.bottom, 0);
        assert_eq!(result.left, 0);
        assert_eq!(result.right, 0);
    }

    #[test]
    fn test_compute_padding_letterbox() {
        // 16:9 into 4:3 => letterbox (top/bottom bars)
        let result = compute_padding(1920, 1080, 640, 480);
        assert!(result.top > 0 || result.bottom > 0);
        assert_eq!(result.left, 0);
        assert_eq!(result.right, 0);
    }

    #[test]
    fn test_compute_padding_pillarbox() {
        // 4:3 into 16:9 => pillarbox (left/right bars)
        let result = compute_padding(640, 480, 1920, 1080);
        assert_eq!(result.top, 0);
        assert_eq!(result.bottom, 0);
        assert!(result.left > 0 || result.right > 0);
    }

    #[test]
    fn test_compute_crop_wider_source() {
        let result = compute_crop(1920, 1080, 640, 480);
        assert!(result.crop_width < 1920);
        assert_eq!(result.crop_height, 1080);
        assert!(result.x > 0);
    }

    #[test]
    fn test_fit_inside_width_constrained() {
        let (w, h) = fit_inside(1920, 1080, 640, 640);
        assert_eq!(w, 640);
        assert!(h <= 640);
    }

    #[test]
    fn test_fit_inside_height_constrained() {
        let (w, h) = fit_inside(1080, 1920, 640, 640);
        assert!(w <= 640);
        assert_eq!(h, 640);
    }

    #[test]
    fn test_standard_ratio_to_aspect_ratio() {
        let ar = StandardRatio::Ratio16x9.to_aspect_ratio();
        assert_eq!(ar.width, 16);
        assert_eq!(ar.height, 9);
    }
}

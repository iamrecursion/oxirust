//! Content-aware crop region analysis and selection.
//!
//! This module provides tools to compute an optimal output crop rectangle
//! given a sequence of stabilisation transforms, while trying to preserve
//! important content (e.g. the subject of interest or the average frame centre).

#![allow(dead_code)]

/// A 2-D rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Left edge in pixels.
    pub x: f64,
    /// Top edge in pixels.
    pub y: f64,
    /// Width in pixels.
    pub width: f64,
    /// Height in pixels.
    pub height: f64,
}

impl Rect {
    /// Creates a new [`Rect`].
    #[must_use]
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Returns a zero-sized rect at the origin.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        }
    }

    /// Returns the centre of the rect.
    #[must_use]
    pub fn center(&self) -> (f64, f64) {
        (self.x + self.width * 0.5, self.y + self.height * 0.5)
    }

    /// Area in pixels².
    #[must_use]
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Returns `true` if `point` (x, y) is strictly inside the rect.
    #[must_use]
    pub fn contains(&self, px: f64, py: f64) -> bool {
        px > self.x && px < self.x + self.width && py > self.y && py < self.y + self.height
    }

    /// Returns the intersection of `self` and `other`, or `None` if they do not overlap.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        if x2 > x1 && y2 > y1 {
            Some(Self::new(x1, y1, x2 - x1, y2 - y1))
        } else {
            None
        }
    }

    /// IoU (Intersection over Union) with another rect.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f64 {
        match self.intersection(other) {
            None => 0.0,
            Some(inter) => {
                let inter_area = inter.area();
                let union_area = self.area() + other.area() - inter_area;
                if union_area <= 0.0 {
                    0.0
                } else {
                    inter_area / union_area
                }
            }
        }
    }

    /// Scales the rect by `factor` around its own centre.
    #[must_use]
    pub fn scale_around_center(&self, factor: f64) -> Self {
        let (cx, cy) = self.center();
        let new_w = self.width * factor;
        let new_h = self.height * factor;
        Self::new(cx - new_w * 0.5, cy - new_h * 0.5, new_w, new_h)
    }

    /// Constrains the rect to be fully inside `bounds`.
    #[must_use]
    pub fn constrain_to(&self, bounds: &Self) -> Self {
        let x = self.x.max(bounds.x);
        let y = self.y.max(bounds.y);
        let w = self
            .width
            .min(bounds.width)
            .min(bounds.x + bounds.width - x);
        let h = self
            .height
            .min(bounds.height)
            .min(bounds.y + bounds.height - y);
        Self::new(x, y, w.max(0.0), h.max(0.0))
    }

    /// Returns the aspect ratio (width / height), or 0.0 if height is zero.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height <= 0.0 {
            0.0
        } else {
            self.width / self.height
        }
    }
}

/// A 2-D translation representing frame motion for a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMotion {
    /// Horizontal displacement in pixels.
    pub dx: f64,
    /// Vertical displacement in pixels.
    pub dy: f64,
}

impl FrameMotion {
    /// Creates a [`FrameMotion`].
    #[must_use]
    pub const fn new(dx: f64, dy: f64) -> Self {
        Self { dx, dy }
    }

    /// Returns the zero-motion sentinel.
    #[must_use]
    pub const fn zero() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }

    /// Magnitude of the motion vector.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

/// Computes the safe crop region given a sequence of frame motions and the
/// original frame size.
///
/// The safe region is a rectangle fully inside the frame for every displacement
/// in `motions`.  Any pixel inside this rect is guaranteed to be visible
/// (not black-bordered) in every frame after the given translations are applied.
///
/// If `motions` is empty, the full frame is returned.
#[must_use]
pub fn safe_crop_region(frame_width: f64, frame_height: f64, motions: &[FrameMotion]) -> Rect {
    if motions.is_empty() || frame_width <= 0.0 || frame_height <= 0.0 {
        return Rect::new(0.0, 0.0, frame_width.max(0.0), frame_height.max(0.0));
    }

    let max_dx: f64 = motions.iter().map(|m| m.dx.abs()).fold(0.0, f64::max);
    let max_dy: f64 = motions.iter().map(|m| m.dy.abs()).fold(0.0, f64::max);

    let x = max_dx;
    let y = max_dy;
    let w = (frame_width - 2.0 * max_dx).max(0.0);
    let h = (frame_height - 2.0 * max_dy).max(0.0);

    Rect::new(x, y, w, h)
}

/// Adjusts `crop` to match a target `aspect_ratio` (width / height), shrinking
/// whichever dimension is necessary while keeping the crop centred.
///
/// Returns the original rect if `aspect_ratio` is <= 0.
#[must_use]
pub fn match_aspect_ratio(crop: Rect, target_aspect: f64) -> Rect {
    if target_aspect <= 0.0 || crop.height <= 0.0 {
        return crop;
    }

    let current_aspect = crop.width / crop.height;
    let (cx, cy) = crop.center();

    if current_aspect > target_aspect {
        // Too wide – reduce width
        let new_w = crop.height * target_aspect;
        Rect::new(cx - new_w * 0.5, crop.y, new_w, crop.height)
    } else if current_aspect < target_aspect {
        // Too tall – reduce height
        let new_h = crop.width / target_aspect;
        Rect::new(crop.x, cy - new_h * 0.5, crop.width, new_h)
    } else {
        crop
    }
}

/// Returns the maximum crop rect that fits within `bounds` while preserving
/// `target_aspect` and being as large as possible.
#[must_use]
pub fn largest_crop_with_aspect(bounds: &Rect, target_aspect: f64) -> Rect {
    if target_aspect <= 0.0 || bounds.width <= 0.0 || bounds.height <= 0.0 {
        return *bounds;
    }

    let bounds_aspect = bounds.width / bounds.height;
    let (cx, cy) = bounds.center();

    if bounds_aspect >= target_aspect {
        // Height-limited: use full height, reduce width
        let w = bounds.height * target_aspect;
        Rect::new(cx - w * 0.5, bounds.y, w, bounds.height)
    } else {
        // Width-limited: use full width, reduce height
        let h = bounds.width / target_aspect;
        Rect::new(bounds.x, cy - h * 0.5, bounds.width, h)
    }
}

/// Computes a content-weighted crop by biasing toward a region of interest (ROI).
///
/// The result is a `crop_size × crop_size` rect blended between the safe crop
/// centre and the ROI centre according to `roi_weight` (0.0 = ignore ROI,
/// 1.0 = follow ROI fully).  The rect is then constrained to the frame.
#[must_use]
pub fn content_aware_crop(
    frame_width: f64,
    frame_height: f64,
    safe_region: &Rect,
    roi: &Rect,
    roi_weight: f64,
    crop_width: f64,
    crop_height: f64,
) -> Rect {
    let safe_cx = safe_region.x + safe_region.width * 0.5;
    let safe_cy = safe_region.y + safe_region.height * 0.5;
    let roi_cx = roi.x + roi.width * 0.5;
    let roi_cy = roi.y + roi.height * 0.5;

    let w = roi_weight.clamp(0.0, 1.0);
    let cx = safe_cx * (1.0 - w) + roi_cx * w;
    let cy = safe_cy * (1.0 - w) + roi_cy * w;

    let bounds = Rect::new(0.0, 0.0, frame_width, frame_height);
    let raw = Rect::new(
        cx - crop_width * 0.5,
        cy - crop_height * 0.5,
        crop_width,
        crop_height,
    );
    raw.constrain_to(&bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_area() {
        let r = Rect::new(0.0, 0.0, 10.0, 5.0);
        assert!((r.area() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_rect_center() {
        let r = Rect::new(2.0, 4.0, 6.0, 8.0);
        let (cx, cy) = r.center();
        assert!((cx - 5.0).abs() < 1e-10);
        assert!((cy - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_rect_contains() {
        let r = Rect::new(0.0, 0.0, 10.0, 10.0);
        assert!(r.contains(5.0, 5.0));
        assert!(!r.contains(-1.0, 5.0));
        assert!(!r.contains(10.0, 5.0)); // Exclusive right edge
    }

    #[test]
    fn test_rect_intersection_overlap() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 10.0, 10.0);
        let inter = a.intersection(&b).expect("should succeed in test");
        assert!((inter.width - 5.0).abs() < 1e-10);
        assert!((inter.height - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_rect_intersection_none() {
        let a = Rect::new(0.0, 0.0, 5.0, 5.0);
        let b = Rect::new(10.0, 10.0, 5.0, 5.0);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_rect_iou_identical() {
        let r = Rect::new(0.0, 0.0, 10.0, 10.0);
        let iou = r.iou(&r);
        assert!((iou - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rect_iou_no_overlap() {
        let a = Rect::new(0.0, 0.0, 5.0, 5.0);
        let b = Rect::new(10.0, 10.0, 5.0, 5.0);
        assert!((a.iou(&b)).abs() < 1e-10);
    }

    #[test]
    fn test_rect_scale_around_center() {
        let r = Rect::new(0.0, 0.0, 10.0, 10.0);
        let scaled = r.scale_around_center(0.5);
        let (cx, cy) = scaled.center();
        assert!((cx - 5.0).abs() < 1e-10);
        assert!((cy - 5.0).abs() < 1e-10);
        assert!((scaled.width - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_rect_aspect_ratio() {
        let r = Rect::new(0.0, 0.0, 16.0, 9.0);
        assert!((r.aspect_ratio() - 16.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_frame_motion_magnitude() {
        let m = FrameMotion::new(3.0, 4.0);
        assert!((m.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_crop_region_no_motion() {
        let r = safe_crop_region(1920.0, 1080.0, &[FrameMotion::zero()]);
        // With zero motion, the entire frame is safe
        assert!((r.width - 1920.0).abs() < 1e-10);
        assert!((r.height - 1080.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_crop_region_with_motion_shrinks() {
        let motions = vec![FrameMotion::new(50.0, 30.0)];
        let r = safe_crop_region(1920.0, 1080.0, &motions);
        assert!(r.width < 1920.0, "safe width should be smaller than frame");
        assert!(
            r.height < 1080.0,
            "safe height should be smaller than frame"
        );
        assert!((r.width - (1920.0 - 100.0)).abs() < 1e-10);
        assert!((r.height - (1080.0 - 60.0)).abs() < 1e-10);
    }

    #[test]
    fn test_match_aspect_ratio_wider_rect() {
        let crop = Rect::new(0.0, 0.0, 200.0, 100.0); // 2:1
        let adjusted = match_aspect_ratio(crop, 16.0 / 9.0); // narrower
                                                             // Height stays, width reduces
        assert!((adjusted.height - 100.0).abs() < 1e-6);
        let expected_w = 100.0 * 16.0 / 9.0;
        assert!(
            (adjusted.width - expected_w).abs() < 1e-6,
            "w = {}",
            adjusted.width
        );
    }

    #[test]
    fn test_largest_crop_with_aspect_fits_in_bounds() {
        let bounds = Rect::new(0.0, 0.0, 1920.0, 1080.0);
        let crop = largest_crop_with_aspect(&bounds, 16.0 / 9.0);
        assert!(crop.width <= bounds.width + 1e-6);
        assert!(crop.height <= bounds.height + 1e-6);
        let ar = crop.width / crop.height;
        assert!((ar - 16.0 / 9.0).abs() < 1e-6, "aspect ratio = {ar}");
    }

    #[test]
    fn test_content_aware_crop_zero_weight_uses_safe_center() {
        let safe = Rect::new(50.0, 30.0, 1820.0, 1020.0);
        let roi = Rect::new(500.0, 300.0, 200.0, 200.0);
        let crop = content_aware_crop(1920.0, 1080.0, &safe, &roi, 0.0, 800.0, 450.0);
        let (cx, cy) = crop.center();
        let (scx, scy) = safe.center();
        assert!((cx - scx).abs() < 1e-6, "cx = {cx}, safe_cx = {scx}");
        assert!((cy - scy).abs() < 1e-6, "cy = {cy}, safe_cy = {scy}");
    }

    #[test]
    fn test_content_aware_crop_result_within_frame() {
        let safe = Rect::new(50.0, 30.0, 1820.0, 1020.0);
        let roi = Rect::new(100.0, 100.0, 300.0, 300.0);
        let crop = content_aware_crop(1920.0, 1080.0, &safe, &roi, 0.5, 800.0, 450.0);
        assert!(crop.x >= 0.0);
        assert!(crop.y >= 0.0);
        assert!(crop.x + crop.width <= 1920.0 + 1e-6);
        assert!(crop.y + crop.height <= 1080.0 + 1e-6);
    }
}

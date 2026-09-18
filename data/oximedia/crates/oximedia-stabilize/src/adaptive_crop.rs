#![allow(dead_code)]
//! Adaptive cropping for stabilized video output.
//!
//! This module dynamically adjusts the output crop region per frame to maximize
//! usable content while hiding black borders introduced by stabilization transforms.
//! It supports multiple strategies including fixed, dynamic, and content-aware cropping.

/// Strategy for adaptive cropping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropStrategy {
    /// Fixed crop: same rectangle for all frames.
    Fixed,
    /// Dynamic crop: adjusts per-frame to maximize content.
    Dynamic,
    /// Content-aware crop: avoids cropping important regions.
    ContentAware,
    /// Zoom-to-fit: scales up to hide all borders (may lose edges).
    ZoomToFit,
}

/// A rectangular region describing a crop area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropRect {
    /// Left edge offset (pixels).
    pub left: f64,
    /// Top edge offset (pixels).
    pub top: f64,
    /// Width of the crop region (pixels).
    pub width: f64,
    /// Height of the crop region (pixels).
    pub height: f64,
}

impl CropRect {
    /// Create a new crop rectangle.
    #[must_use]
    pub fn new(left: f64, top: f64, width: f64, height: f64) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    /// Compute the area of the crop region.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Compute the right edge coordinate.
    #[must_use]
    pub fn right(&self) -> f64 {
        self.left + self.width
    }

    /// Compute the bottom edge coordinate.
    #[must_use]
    pub fn bottom(&self) -> f64 {
        self.top + self.height
    }

    /// Compute the center of the crop region.
    #[must_use]
    pub fn center(&self) -> (f64, f64) {
        (self.left + self.width * 0.5, self.top + self.height * 0.5)
    }

    /// Compute the aspect ratio (width / height).
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height > 0.0 {
            self.width / self.height
        } else {
            0.0
        }
    }

    /// Compute the intersection of two crop rectangles.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        if right > left && bottom > top {
            Some(Self::new(left, top, right - left, bottom - top))
        } else {
            None
        }
    }
}

/// Per-frame transform data used to compute available content region.
#[derive(Debug, Clone, Copy)]
pub struct FrameTransform {
    /// Horizontal translation (pixels).
    pub tx: f64,
    /// Vertical translation (pixels).
    pub ty: f64,
    /// Rotation angle (radians).
    pub rotation: f64,
    /// Scale factor.
    pub scale: f64,
}

impl FrameTransform {
    /// Create a new frame transform.
    #[must_use]
    pub fn new(tx: f64, ty: f64, rotation: f64, scale: f64) -> Self {
        Self {
            tx,
            ty,
            rotation,
            scale,
        }
    }

    /// Create an identity (no-op) transform.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            rotation: 0.0,
            scale: 1.0,
        }
    }
}

/// Result of adaptive crop calculation for a sequence.
#[derive(Debug, Clone)]
pub struct AdaptiveCropResult {
    /// Per-frame crop rectangles.
    pub crops: Vec<CropRect>,
    /// The bounding crop used across the entire sequence (for fixed strategy).
    pub global_crop: CropRect,
    /// Average content preservation ratio (0.0–1.0).
    pub avg_preservation: f64,
    /// Minimum content preservation ratio.
    pub min_preservation: f64,
}

/// Adaptive crop calculator.
#[derive(Debug)]
pub struct AdaptiveCropper {
    /// Crop strategy.
    strategy: CropStrategy,
    /// Source frame width.
    frame_width: f64,
    /// Source frame height.
    frame_height: f64,
    /// Desired output aspect ratio (width / height), or None to match source.
    target_aspect: Option<f64>,
    /// Minimum content preservation ratio (0.0–1.0).
    min_preservation: f64,
    /// Temporal smoothing strength for dynamic crops (0.0–1.0).
    temporal_smooth: f64,
}

impl AdaptiveCropper {
    /// Create a new adaptive cropper.
    #[must_use]
    pub fn new(frame_width: f64, frame_height: f64, strategy: CropStrategy) -> Self {
        Self {
            strategy,
            frame_width,
            frame_height,
            target_aspect: None,
            min_preservation: 0.8,
            temporal_smooth: 0.7,
        }
    }

    /// Set the target output aspect ratio.
    #[must_use]
    pub fn with_target_aspect(mut self, aspect: f64) -> Self {
        self.target_aspect = Some(aspect);
        self
    }

    /// Set the minimum content preservation ratio.
    #[must_use]
    pub fn with_min_preservation(mut self, ratio: f64) -> Self {
        self.min_preservation = ratio.clamp(0.1, 1.0);
        self
    }

    /// Set temporal smoothing strength for dynamic mode.
    #[must_use]
    pub fn with_temporal_smooth(mut self, strength: f64) -> Self {
        self.temporal_smooth = strength.clamp(0.0, 1.0);
        self
    }

    /// Compute adaptive crop regions for a sequence of transforms.
    #[must_use]
    pub fn compute(&self, transforms: &[FrameTransform]) -> AdaptiveCropResult {
        if transforms.is_empty() {
            return AdaptiveCropResult {
                crops: Vec::new(),
                global_crop: CropRect::new(0.0, 0.0, self.frame_width, self.frame_height),
                avg_preservation: 1.0,
                min_preservation: 1.0,
            };
        }

        match self.strategy {
            CropStrategy::Fixed => self.compute_fixed(transforms),
            CropStrategy::Dynamic => self.compute_dynamic(transforms),
            CropStrategy::ContentAware => self.compute_content_aware(transforms),
            CropStrategy::ZoomToFit => self.compute_zoom_to_fit(transforms),
        }
    }

    /// Fixed crop: find the largest common rectangle.
    fn compute_fixed(&self, transforms: &[FrameTransform]) -> AdaptiveCropResult {
        let mut max_abs_tx = 0.0_f64;
        let mut max_abs_ty = 0.0_f64;
        let mut min_scale = f64::MAX;

        for t in transforms {
            max_abs_tx = max_abs_tx.max(t.tx.abs());
            max_abs_ty = max_abs_ty.max(t.ty.abs());
            min_scale = min_scale.min(t.scale);
        }

        let margin_x = max_abs_tx + self.frame_width * (1.0 - min_scale) * 0.5;
        let margin_y = max_abs_ty + self.frame_height * (1.0 - min_scale) * 0.5;
        let crop_w = (self.frame_width - 2.0 * margin_x).max(1.0);
        let crop_h = (self.frame_height - 2.0 * margin_y).max(1.0);
        let crop = self.apply_aspect_constraint(margin_x, margin_y, crop_w, crop_h);

        let preservation = crop.area() / (self.frame_width * self.frame_height);
        let crops = vec![crop; transforms.len()];

        AdaptiveCropResult {
            crops,
            global_crop: crop,
            avg_preservation: preservation,
            min_preservation: preservation,
        }
    }

    /// Dynamic crop: per-frame optimal crop with temporal smoothing.
    fn compute_dynamic(&self, transforms: &[FrameTransform]) -> AdaptiveCropResult {
        let raw_crops: Vec<CropRect> = transforms
            .iter()
            .map(|t| self.crop_for_transform(t))
            .collect();

        // Temporal smoothing
        let smoothed = self.smooth_crops(&raw_crops);

        let total_area = self.frame_width * self.frame_height;
        let preservations: Vec<f64> = smoothed.iter().map(|c| c.area() / total_area).collect();
        let avg_pres = preservations.iter().sum::<f64>() / preservations.len() as f64;
        let min_pres = preservations.iter().copied().fold(f64::MAX, f64::min);

        let global_crop = self.compute_global_from_per_frame(&smoothed);

        AdaptiveCropResult {
            crops: smoothed,
            global_crop,
            avg_preservation: avg_pres,
            min_preservation: min_pres,
        }
    }

    /// Content-aware crop: biases toward keeping center content.
    fn compute_content_aware(&self, transforms: &[FrameTransform]) -> AdaptiveCropResult {
        let raw_crops: Vec<CropRect> = transforms
            .iter()
            .map(|t| {
                let base = self.crop_for_transform(t);
                // Bias toward frame center
                let cx = self.frame_width * 0.5;
                let cy = self.frame_height * 0.5;
                let (bcx, bcy) = base.center();
                let new_left = base.left + (cx - bcx) * 0.2;
                let new_top = base.top + (cy - bcy) * 0.2;
                CropRect::new(
                    new_left.max(0.0),
                    new_top.max(0.0),
                    base.width.min(self.frame_width - new_left.max(0.0)),
                    base.height.min(self.frame_height - new_top.max(0.0)),
                )
            })
            .collect();

        let smoothed = self.smooth_crops(&raw_crops);
        let total_area = self.frame_width * self.frame_height;
        let preservations: Vec<f64> = smoothed.iter().map(|c| c.area() / total_area).collect();
        let avg_pres = preservations.iter().sum::<f64>() / preservations.len() as f64;
        let min_pres = preservations.iter().copied().fold(f64::MAX, f64::min);
        let global_crop = self.compute_global_from_per_frame(&smoothed);

        AdaptiveCropResult {
            crops: smoothed,
            global_crop,
            avg_preservation: avg_pres,
            min_preservation: min_pres,
        }
    }

    /// Zoom-to-fit: scale up so no borders are visible.
    fn compute_zoom_to_fit(&self, transforms: &[FrameTransform]) -> AdaptiveCropResult {
        let mut max_zoom = 1.0_f64;

        for t in transforms {
            let zoom_x = if self.frame_width > 0.0 {
                (t.tx.abs() * 2.0 + self.frame_width) / self.frame_width
            } else {
                1.0
            };
            let zoom_y = if self.frame_height > 0.0 {
                (t.ty.abs() * 2.0 + self.frame_height) / self.frame_height
            } else {
                1.0
            };
            let zoom_s = if t.scale > 0.0 { 1.0 / t.scale } else { 1.0 };
            max_zoom = max_zoom.max(zoom_x).max(zoom_y).max(zoom_s);
        }

        let crop_w = self.frame_width / max_zoom;
        let crop_h = self.frame_height / max_zoom;
        let left = (self.frame_width - crop_w) * 0.5;
        let top = (self.frame_height - crop_h) * 0.5;
        let crop = CropRect::new(left, top, crop_w, crop_h);

        let preservation = crop.area() / (self.frame_width * self.frame_height);
        let crops = vec![crop; transforms.len()];

        AdaptiveCropResult {
            crops,
            global_crop: crop,
            avg_preservation: preservation,
            min_preservation: preservation,
        }
    }

    /// Compute crop region for a single transform.
    fn crop_for_transform(&self, t: &FrameTransform) -> CropRect {
        let margin_x = t.tx.abs() + self.frame_width * (1.0 - t.scale).max(0.0) * 0.5;
        let margin_y = t.ty.abs() + self.frame_height * (1.0 - t.scale).max(0.0) * 0.5;
        let w = (self.frame_width - 2.0 * margin_x).max(1.0);
        let h = (self.frame_height - 2.0 * margin_y).max(1.0);
        self.apply_aspect_constraint(margin_x, margin_y, w, h)
    }

    /// Apply target aspect ratio constraint to a crop region.
    fn apply_aspect_constraint(&self, left: f64, top: f64, w: f64, h: f64) -> CropRect {
        if let Some(target) = self.target_aspect {
            let current = if h > 0.0 { w / h } else { 1.0 };
            if current > target {
                // Too wide, reduce width
                let new_w = h * target;
                let new_left = left + (w - new_w) * 0.5;
                CropRect::new(new_left, top, new_w, h)
            } else {
                // Too tall, reduce height
                let new_h = if target > 0.0 { w / target } else { h };
                let new_top = top + (h - new_h) * 0.5;
                CropRect::new(left, new_top, w, new_h)
            }
        } else {
            CropRect::new(left, top, w, h)
        }
    }

    /// Temporally smooth a sequence of crop rectangles.
    fn smooth_crops(&self, crops: &[CropRect]) -> Vec<CropRect> {
        if crops.is_empty() {
            return Vec::new();
        }
        let alpha = 1.0 - self.temporal_smooth;
        let mut result = Vec::with_capacity(crops.len());
        let mut prev = crops[0];
        result.push(prev);

        for crop in &crops[1..] {
            let smoothed = CropRect::new(
                alpha * crop.left + (1.0 - alpha) * prev.left,
                alpha * crop.top + (1.0 - alpha) * prev.top,
                alpha * crop.width + (1.0 - alpha) * prev.width,
                alpha * crop.height + (1.0 - alpha) * prev.height,
            );
            result.push(smoothed);
            prev = smoothed;
        }
        result
    }

    /// Compute a global bounding crop from per-frame crops.
    fn compute_global_from_per_frame(&self, crops: &[CropRect]) -> CropRect {
        if crops.is_empty() {
            return CropRect::new(0.0, 0.0, self.frame_width, self.frame_height);
        }
        let max_left = crops.iter().map(|c| c.left).fold(0.0_f64, f64::max);
        let max_top = crops.iter().map(|c| c.top).fold(0.0_f64, f64::max);
        let min_right = crops.iter().map(|c| c.right()).fold(f64::MAX, f64::min);
        let min_bottom = crops.iter().map(|c| c.bottom()).fold(f64::MAX, f64::min);
        CropRect::new(
            max_left,
            max_top,
            (min_right - max_left).max(1.0),
            (min_bottom - max_top).max(1.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_rect_area() {
        let r = CropRect::new(10.0, 20.0, 100.0, 50.0);
        assert!((r.area() - 5000.0).abs() < 1e-10);
    }

    #[test]
    fn test_crop_rect_edges() {
        let r = CropRect::new(10.0, 20.0, 100.0, 50.0);
        assert!((r.right() - 110.0).abs() < 1e-10);
        assert!((r.bottom() - 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_crop_rect_center() {
        let r = CropRect::new(0.0, 0.0, 100.0, 200.0);
        let (cx, cy) = r.center();
        assert!((cx - 50.0).abs() < 1e-10);
        assert!((cy - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_crop_rect_aspect_ratio() {
        let r = CropRect::new(0.0, 0.0, 160.0, 90.0);
        assert!((r.aspect_ratio() - 160.0 / 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_crop_rect_intersect() {
        let a = CropRect::new(0.0, 0.0, 100.0, 100.0);
        let b = CropRect::new(50.0, 50.0, 100.0, 100.0);
        let inter = a.intersect(&b).expect("should succeed in test");
        assert!((inter.left - 50.0).abs() < 1e-10);
        assert!((inter.width - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_crop_rect_no_intersect() {
        let a = CropRect::new(0.0, 0.0, 10.0, 10.0);
        let b = CropRect::new(20.0, 20.0, 10.0, 10.0);
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn test_frame_transform_identity() {
        let t = FrameTransform::identity();
        assert!((t.tx - 0.0).abs() < 1e-10);
        assert!((t.scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_empty_transforms() {
        let cropper = AdaptiveCropper::new(1920.0, 1080.0, CropStrategy::Fixed);
        let result = cropper.compute(&[]);
        assert!(result.crops.is_empty());
        assert!((result.avg_preservation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_identity_transforms() {
        let transforms: Vec<FrameTransform> = (0..10).map(|_| FrameTransform::identity()).collect();
        let cropper = AdaptiveCropper::new(1920.0, 1080.0, CropStrategy::Fixed);
        let result = cropper.compute(&transforms);
        assert_eq!(result.crops.len(), 10);
        assert!((result.avg_preservation - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dynamic_strategy() {
        let transforms: Vec<FrameTransform> = (0..20)
            .map(|i| FrameTransform::new(i as f64 * 2.0, i as f64, 0.0, 1.0))
            .collect();
        let cropper = AdaptiveCropper::new(1920.0, 1080.0, CropStrategy::Dynamic);
        let result = cropper.compute(&transforms);
        assert_eq!(result.crops.len(), 20);
        assert!(result.avg_preservation > 0.0);
    }

    #[test]
    fn test_content_aware_strategy() {
        let transforms: Vec<FrameTransform> = (0..15)
            .map(|i| FrameTransform::new(i as f64 * 3.0, 0.0, 0.0, 0.98))
            .collect();
        let cropper = AdaptiveCropper::new(1920.0, 1080.0, CropStrategy::ContentAware);
        let result = cropper.compute(&transforms);
        assert_eq!(result.crops.len(), 15);
    }

    #[test]
    fn test_zoom_to_fit_strategy() {
        let transforms = vec![
            FrameTransform::new(20.0, 10.0, 0.0, 0.95),
            FrameTransform::new(-15.0, 5.0, 0.0, 0.97),
        ];
        let cropper = AdaptiveCropper::new(1920.0, 1080.0, CropStrategy::ZoomToFit);
        let result = cropper.compute(&transforms);
        assert_eq!(result.crops.len(), 2);
        assert!(result.avg_preservation < 1.0);
    }

    #[test]
    fn test_target_aspect_ratio() {
        let transforms = vec![FrameTransform::identity()];
        let cropper =
            AdaptiveCropper::new(1920.0, 1080.0, CropStrategy::Fixed).with_target_aspect(1.0); // Square output
        let result = cropper.compute(&transforms);
        let crop = &result.crops[0];
        assert!((crop.aspect_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_temporal_smoothing() {
        let transforms: Vec<FrameTransform> = (0..30)
            .map(|i| {
                let jitter = if i % 2 == 0 { 10.0 } else { -10.0 };
                FrameTransform::new(jitter, 0.0, 0.0, 1.0)
            })
            .collect();
        let cropper =
            AdaptiveCropper::new(1920.0, 1080.0, CropStrategy::Dynamic).with_temporal_smooth(0.9);
        let result = cropper.compute(&transforms);
        // Smoothed crops should have less variation than raw ones
        let variance: f64 = result
            .crops
            .windows(2)
            .map(|w| (w[1].left - w[0].left).abs())
            .sum::<f64>()
            / (result.crops.len() - 1) as f64;
        assert!(variance < 20.0);
    }
}

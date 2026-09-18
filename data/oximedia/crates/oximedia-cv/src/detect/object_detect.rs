//! Extended object detection data structures and algorithms.
//!
//! This module provides additional object detection utilities including:
//! - Typed [`BoundingBoxF`] with extended methods (IoU, expand, contains_point)
//! - [`Detection`] with class name and confidence
//! - [`DetectionResult`] for batch results with timing
//! - [`non_max_suppression`] and [`filter_by_confidence`] convenience functions
//! - [`DetectorConfig`] for configuring detectors
//!
//! # Example
//!
//! ```
//! use oximedia_cv::detect::object_detect::{
//!     BoundingBoxF, Detection as DetectionExt, DetectorConfig,
//!     non_max_suppression, filter_by_confidence,
//! };
//!
//! let cfg = DetectorConfig::default();
//! let bbox = BoundingBoxF::new(0.0, 0.0, 100.0, 100.0);
//! assert_eq!(bbox.area(), 10000.0);
//! ```

#![allow(dead_code)]

/// A bounding box using `f32` coordinates (top-left origin).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBoxF {
    /// X coordinate of the top-left corner.
    pub x: f32,
    /// Y coordinate of the top-left corner.
    pub y: f32,
    /// Width of the box.
    pub width: f32,
    /// Height of the box.
    pub height: f32,
}

impl BoundingBoxF {
    /// Create a new bounding box.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::object_detect::BoundingBoxF;
    ///
    /// let b = BoundingBoxF::new(10.0, 20.0, 50.0, 80.0);
    /// assert_eq!(b.x, 10.0);
    /// ```
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Area of the bounding box.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::object_detect::BoundingBoxF;
    ///
    /// let b = BoundingBoxF::new(0.0, 0.0, 10.0, 20.0);
    /// assert_eq!(b.area(), 200.0);
    /// ```
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Right edge (x + width).
    #[must_use]
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Bottom edge (y + height).
    #[must_use]
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// Intersection over Union (IoU) with another bounding box.
    ///
    /// Returns a value in `[0.0, 1.0]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::object_detect::BoundingBoxF;
    ///
    /// let a = BoundingBoxF::new(0.0, 0.0, 100.0, 100.0);
    /// let b = BoundingBoxF::new(0.0, 0.0, 100.0, 100.0);
    /// assert!((a.iou(&b) - 1.0).abs() < 1e-5);
    /// ```
    #[must_use]
    pub fn iou(&self, other: &Self) -> f32 {
        let ix1 = self.x.max(other.x);
        let iy1 = self.y.max(other.y);
        let ix2 = self.right().min(other.right());
        let iy2 = self.bottom().min(other.bottom());

        let inter_w = (ix2 - ix1).max(0.0);
        let inter_h = (iy2 - iy1).max(0.0);
        let inter_area = inter_w * inter_h;

        let union_area = self.area() + other.area() - inter_area;
        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }

    /// Check whether the box contains a point `(px, py)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::object_detect::BoundingBoxF;
    ///
    /// let b = BoundingBoxF::new(0.0, 0.0, 100.0, 100.0);
    /// assert!(b.contains_point(50.0, 50.0));
    /// assert!(!b.contains_point(150.0, 50.0));
    /// ```
    #[must_use]
    pub fn contains_point(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.right() && py >= self.y && py <= self.bottom()
    }

    /// Expand the box outward by `margin` on all sides.
    ///
    /// The box position moves by `-margin` and dimensions grow by `2 * margin`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::object_detect::BoundingBoxF;
    ///
    /// let b = BoundingBoxF::new(10.0, 10.0, 80.0, 80.0);
    /// let e = b.expand(5.0);
    /// assert_eq!(e.x, 5.0);
    /// assert_eq!(e.width, 90.0);
    /// ```
    #[must_use]
    pub fn expand(&self, margin: f32) -> Self {
        Self {
            x: self.x - margin,
            y: self.y - margin,
            width: self.width + 2.0 * margin,
            height: self.height + 2.0 * margin,
        }
    }

    /// Clamp to image bounds `[0, img_width] x [0, img_height]`.
    #[must_use]
    pub fn clamp_to_image(&self, img_width: f32, img_height: f32) -> Self {
        let x = self.x.max(0.0);
        let y = self.y.max(0.0);
        let right = self.right().min(img_width);
        let bottom = self.bottom().min(img_height);
        Self {
            x,
            y,
            width: (right - x).max(0.0),
            height: (bottom - y).max(0.0),
        }
    }
}

impl Default for BoundingBoxF {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

/// A single object detection with class identity and confidence.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Numeric class identifier.
    pub class_id: u32,
    /// Human-readable class name.
    pub class_name: String,
    /// Detection confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Bounding box in image coordinates.
    pub bbox: BoundingBoxF,
}

impl Detection {
    /// Create a new detection.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::object_detect::{Detection, BoundingBoxF};
    ///
    /// let d = Detection::new(0, "cat".to_string(), 0.9, BoundingBoxF::new(0.0, 0.0, 50.0, 50.0));
    /// assert_eq!(d.class_name, "cat");
    /// ```
    #[must_use]
    pub fn new(class_id: u32, class_name: String, confidence: f32, bbox: BoundingBoxF) -> Self {
        Self {
            class_id,
            class_name,
            confidence,
            bbox,
        }
    }
}

/// The result of running inference on a single frame.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// All detections produced for this frame.
    pub detections: Vec<Detection>,
    /// Zero-based frame index.
    pub frame_id: u64,
    /// Wall-clock time taken for inference in milliseconds.
    pub inference_ms: f64,
}

impl DetectionResult {
    /// Create a new result.
    #[must_use]
    pub fn new(detections: Vec<Detection>, frame_id: u64, inference_ms: f64) -> Self {
        Self {
            detections,
            frame_id,
            inference_ms,
        }
    }

    /// Number of detections.
    #[must_use]
    pub fn len(&self) -> usize {
        self.detections.len()
    }

    /// Whether the result is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.detections.is_empty()
    }
}

/// Configuration for an object detector.
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Minimum confidence required to keep a detection.
    pub confidence_threshold: f32,
    /// IoU threshold for Non-Maximum Suppression.
    pub nms_iou_threshold: f32,
    /// Maximum number of detections to return after NMS.
    pub max_detections: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            nms_iou_threshold: 0.45,
            max_detections: 100,
        }
    }
}

impl DetectorConfig {
    /// Create a new `DetectorConfig`.
    #[must_use]
    pub fn new(confidence_threshold: f32, nms_iou_threshold: f32, max_detections: usize) -> Self {
        Self {
            confidence_threshold,
            nms_iou_threshold,
            max_detections,
        }
    }
}

/// Run class-aware Non-Maximum Suppression (NMS) on a list of detections.
///
/// Detections are sorted by confidence (descending). A detection is suppressed
/// when its IoU with an already-kept detection of the **same class** exceeds
/// `iou_threshold`.
///
/// # Examples
///
/// ```
/// use oximedia_cv::detect::object_detect::{
///     Detection, BoundingBoxF, non_max_suppression,
/// };
///
/// let dets = vec![
///     Detection::new(0, "dog".into(), 0.9, BoundingBoxF::new(0.0, 0.0, 100.0, 100.0)),
///     Detection::new(0, "dog".into(), 0.7, BoundingBoxF::new(5.0, 5.0, 100.0, 100.0)),
///     Detection::new(1, "cat".into(), 0.8, BoundingBoxF::new(0.0, 0.0, 100.0, 100.0)),
/// ];
/// let kept = non_max_suppression(dets, 0.5);
/// // Overlapping dog suppressed; dog+cat (different class) both kept
/// assert_eq!(kept.len(), 2);
/// ```
#[must_use]
pub fn non_max_suppression(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    // Sort by confidence descending.
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = detections.len();
    let mut suppressed = vec![false; n];
    let mut kept = Vec::new();

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        kept.push(detections[i].clone());

        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }
            if detections[i].class_id == detections[j].class_id
                && detections[i].bbox.iou(&detections[j].bbox) > iou_threshold
            {
                suppressed[j] = true;
            }
        }
    }

    kept
}

/// Keep only detections whose confidence is at least `min_confidence`.
///
/// # Examples
///
/// ```
/// use oximedia_cv::detect::object_detect::{Detection, BoundingBoxF, filter_by_confidence};
///
/// let dets = vec![
///     Detection::new(0, "a".into(), 0.9, BoundingBoxF::default()),
///     Detection::new(0, "b".into(), 0.3, BoundingBoxF::default()),
/// ];
/// let kept = filter_by_confidence(dets, 0.5);
/// assert_eq!(kept.len(), 1);
/// ```
#[must_use]
pub fn filter_by_confidence(detections: Vec<Detection>, min_confidence: f32) -> Vec<Detection> {
    detections
        .into_iter()
        .filter(|d| d.confidence >= min_confidence)
        .collect()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn make_det(class_id: u32, conf: f32, x: f32, y: f32, w: f32, h: f32) -> Detection {
        Detection::new(
            class_id,
            format!("class_{class_id}"),
            conf,
            BoundingBoxF::new(x, y, w, h),
        )
    }

    #[test]
    fn test_bbox_new_and_fields() {
        let b = BoundingBoxF::new(1.0, 2.0, 30.0, 40.0);
        assert_eq!(b.x, 1.0);
        assert_eq!(b.y, 2.0);
        assert_eq!(b.width, 30.0);
        assert_eq!(b.height, 40.0);
    }

    #[test]
    fn test_bbox_area() {
        let b = BoundingBoxF::new(0.0, 0.0, 5.0, 8.0);
        assert_eq!(b.area(), 40.0);
    }

    #[test]
    fn test_bbox_right_bottom() {
        let b = BoundingBoxF::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(b.right(), 40.0);
        assert_eq!(b.bottom(), 60.0);
    }

    #[test]
    fn test_iou_identical() {
        let a = BoundingBoxF::new(0.0, 0.0, 100.0, 100.0);
        assert!((a.iou(&a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = BoundingBoxF::new(0.0, 0.0, 50.0, 50.0);
        let b = BoundingBoxF::new(100.0, 100.0, 50.0, 50.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let a = BoundingBoxF::new(0.0, 0.0, 100.0, 100.0);
        let b = BoundingBoxF::new(50.0, 50.0, 100.0, 100.0);
        // intersection: 50x50 = 2500, union: 10000+10000-2500 = 17500
        let expected = 2500.0 / 17500.0;
        assert!((a.iou(&b) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_contains_point_inside() {
        let b = BoundingBoxF::new(0.0, 0.0, 100.0, 100.0);
        assert!(b.contains_point(50.0, 50.0));
    }

    #[test]
    fn test_contains_point_outside() {
        let b = BoundingBoxF::new(0.0, 0.0, 100.0, 100.0);
        assert!(!b.contains_point(101.0, 50.0));
        assert!(!b.contains_point(50.0, 101.0));
    }

    #[test]
    fn test_contains_point_on_edge() {
        let b = BoundingBoxF::new(0.0, 0.0, 100.0, 100.0);
        assert!(b.contains_point(0.0, 0.0));
        assert!(b.contains_point(100.0, 100.0));
    }

    #[test]
    fn test_expand() {
        let b = BoundingBoxF::new(10.0, 10.0, 80.0, 80.0);
        let e = b.expand(5.0);
        assert_eq!(e.x, 5.0);
        assert_eq!(e.y, 5.0);
        assert_eq!(e.width, 90.0);
        assert_eq!(e.height, 90.0);
    }

    #[test]
    fn test_expand_zero() {
        let b = BoundingBoxF::new(10.0, 10.0, 80.0, 80.0);
        let e = b.expand(0.0);
        assert_eq!(e.x, b.x);
        assert_eq!(e.width, b.width);
    }

    #[test]
    fn test_detection_new() {
        let bbox = BoundingBoxF::new(0.0, 0.0, 50.0, 50.0);
        let d = Detection::new(3, "bird".to_string(), 0.88, bbox);
        assert_eq!(d.class_id, 3);
        assert_eq!(d.class_name, "bird");
        assert!((d.confidence - 0.88).abs() < 1e-5);
    }

    #[test]
    fn test_detection_result() {
        let dets = vec![make_det(0, 0.9, 0.0, 0.0, 50.0, 50.0)];
        let result = DetectionResult::new(dets, 42, 12.5);
        assert_eq!(result.frame_id, 42);
        assert_eq!(result.len(), 1);
        assert!(!result.is_empty());
        assert!((result.inference_ms - 12.5).abs() < 1e-5);
    }

    #[test]
    fn test_detection_result_empty() {
        let result = DetectionResult::new(vec![], 0, 0.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detector_config_default() {
        let cfg = DetectorConfig::default();
        assert!((cfg.confidence_threshold - 0.5).abs() < 1e-5);
        assert!((cfg.nms_iou_threshold - 0.45).abs() < 1e-5);
        assert_eq!(cfg.max_detections, 100);
    }

    #[test]
    fn test_nms_suppresses_overlap() {
        let dets = vec![
            make_det(0, 0.9, 0.0, 0.0, 100.0, 100.0),
            make_det(0, 0.7, 5.0, 5.0, 100.0, 100.0),
            make_det(0, 0.5, 200.0, 200.0, 100.0, 100.0),
        ];
        let kept = non_max_suppression(dets, 0.5);
        // First two overlap heavily → keep only best; third is separate
        assert_eq!(kept.len(), 2);
        assert!((kept[0].confidence - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_nms_different_classes_not_suppressed() {
        let dets = vec![
            make_det(0, 0.9, 0.0, 0.0, 100.0, 100.0),
            make_det(1, 0.8, 0.0, 0.0, 100.0, 100.0),
        ];
        let kept = non_max_suppression(dets, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_nms_empty() {
        let kept = non_max_suppression(vec![], 0.5);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_filter_by_confidence_basic() {
        let dets = vec![
            make_det(0, 0.9, 0.0, 0.0, 50.0, 50.0),
            make_det(0, 0.3, 0.0, 0.0, 50.0, 50.0),
            make_det(0, 0.6, 0.0, 0.0, 50.0, 50.0),
        ];
        let kept = filter_by_confidence(dets, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_filter_by_confidence_none_pass() {
        let dets = vec![make_det(0, 0.1, 0.0, 0.0, 50.0, 50.0)];
        let kept = filter_by_confidence(dets, 0.5);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_filter_by_confidence_all_pass() {
        let dets = vec![
            make_det(0, 0.9, 0.0, 0.0, 50.0, 50.0),
            make_det(1, 0.95, 0.0, 0.0, 50.0, 50.0),
        ];
        let kept = filter_by_confidence(dets, 0.5);
        assert_eq!(kept.len(), 2);
    }
}

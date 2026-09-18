//! Axis-aligned bounding-box utilities for computer-vision pipelines.

#![allow(dead_code)]

/// An axis-aligned bounding box defined by its top-left corner, width and height.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Left edge (x-coordinate).
    pub x: f32,
    /// Top edge (y-coordinate).
    pub y: f32,
    /// Width of the box.
    pub width: f32,
    /// Height of the box.
    pub height: f32,
    /// Detection confidence in [0.0, 1.0].
    pub confidence: f32,
    /// Optional class identifier.
    pub class_id: Option<u32>,
}

impl BoundingBox {
    /// Create a new [`BoundingBox`] with full confidence.
    #[must_use]
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width: width.max(0.0),
            height: height.max(0.0),
            confidence: 1.0,
            class_id: None,
        }
    }

    /// Create a new [`BoundingBox`] with explicit confidence.
    #[must_use]
    pub fn with_confidence(x: f32, y: f32, width: f32, height: f32, confidence: f32) -> Self {
        Self {
            x,
            y,
            width: width.max(0.0),
            height: height.max(0.0),
            confidence: confidence.clamp(0.0, 1.0),
            class_id: None,
        }
    }

    /// Return the area of the bounding box.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Return the right edge of the box.
    #[must_use]
    pub fn x2(&self) -> f32 {
        self.x + self.width
    }

    /// Return the bottom edge of the box.
    #[must_use]
    pub fn y2(&self) -> f32 {
        self.y + self.height
    }

    /// Compute Intersection over Union (IoU) with another bounding box.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f32 {
        let ix1 = self.x.max(other.x);
        let iy1 = self.y.max(other.y);
        let ix2 = self.x2().min(other.x2());
        let iy2 = self.y2().min(other.y2());

        if ix2 <= ix1 || iy2 <= iy1 {
            return 0.0;
        }

        let inter = (ix2 - ix1) * (iy2 - iy1);
        let union = self.area() + other.area() - inter;

        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Return `true` when the given point (px, py) lies inside the box (inclusive edges).
    #[must_use]
    pub fn contains_point(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x2() && py >= self.y && py <= self.y2()
    }
}

/// A collection of bounding boxes supporting batch operations.
#[derive(Debug, Clone, Default)]
pub struct BoundingBoxList {
    /// The stored bounding boxes.
    pub boxes: Vec<BoundingBox>,
}

impl BoundingBoxList {
    /// Create an empty [`BoundingBoxList`].
    #[must_use]
    pub fn new() -> Self {
        Self { boxes: Vec::new() }
    }

    /// Add a bounding box to the list.
    pub fn push(&mut self, bbox: BoundingBox) {
        self.boxes.push(bbox);
    }

    /// Return all boxes whose confidence is at least `min_confidence`.
    #[must_use]
    pub fn filter_by_confidence(&self, min_confidence: f32) -> Vec<BoundingBox> {
        self.boxes
            .iter()
            .filter(|b| b.confidence >= min_confidence)
            .cloned()
            .collect()
    }

    /// Apply Non-Maximum Suppression (NMS) and return the surviving boxes.
    ///
    /// `iou_threshold` controls how much overlap is tolerated before a lower-confidence
    /// box is suppressed.
    #[must_use]
    pub fn nms(&self, iou_threshold: f32) -> Vec<BoundingBox> {
        // Sort descending by confidence.
        let mut sorted: Vec<&BoundingBox> = self.boxes.iter().collect();
        sorted.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = vec![true; sorted.len()];

        for i in 0..sorted.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..sorted.len() {
                if !keep[j] {
                    continue;
                }
                if sorted[i].iou(sorted[j]) > iou_threshold {
                    keep[j] = false;
                }
            }
        }

        sorted
            .into_iter()
            .enumerate()
            .filter_map(|(i, b)| if keep[i] { Some(b.clone()) } else { None })
            .collect()
    }

    /// Return the number of bounding boxes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.boxes.len()
    }

    /// Return `true` when the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }
}

/// Summary statistics over a collection of bounding boxes.
#[derive(Debug, Clone)]
pub struct BboxStats {
    /// Minimum area observed.
    pub min_area: f32,
    /// Maximum area observed.
    pub max_area: f32,
    /// Mean area.
    pub mean_area: f32,
    /// Total number of boxes.
    pub count: usize,
}

impl BboxStats {
    /// Compute statistics from a slice of bounding boxes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_boxes(boxes: &[BoundingBox]) -> Self {
        if boxes.is_empty() {
            return Self {
                min_area: 0.0,
                max_area: 0.0,
                mean_area: 0.0,
                count: 0,
            };
        }

        let mut min_area = f32::MAX;
        let mut max_area = f32::MIN;
        let mut sum = 0.0_f32;

        for b in boxes {
            let a = b.area();
            if a < min_area {
                min_area = a;
            }
            if a > max_area {
                max_area = a;
            }
            sum += a;
        }

        Self {
            min_area,
            max_area,
            mean_area: sum / boxes.len() as f32,
            count: boxes.len(),
        }
    }

    /// Return the average area, equivalent to `mean_area`.
    #[must_use]
    pub fn avg_area(&self) -> f32 {
        self.mean_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_area() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 20.0);
        assert!((b.area() - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounding_box_negative_size_clamped() {
        let b = BoundingBox::new(0.0, 0.0, -5.0, -3.0);
        assert!((b.area() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounding_box_iou_identical() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((b.iou(&b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bounding_box_iou_no_overlap() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(20.0, 20.0, 10.0, 10.0);
        assert!((a.iou(&b) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounding_box_iou_partial_overlap() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(5.0, 5.0, 10.0, 10.0);
        let iou = a.iou(&b);
        assert!(iou > 0.0 && iou < 1.0);
    }

    #[test]
    fn test_bounding_box_contains_point_inside() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(b.contains_point(5.0, 5.0));
    }

    #[test]
    fn test_bounding_box_contains_point_outside() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(!b.contains_point(15.0, 5.0));
    }

    #[test]
    fn test_bounding_box_contains_point_on_edge() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(b.contains_point(0.0, 0.0));
        assert!(b.contains_point(10.0, 10.0));
    }

    #[test]
    fn test_bounding_box_list_filter_by_confidence() {
        let mut list = BoundingBoxList::new();
        list.push(BoundingBox::with_confidence(0.0, 0.0, 10.0, 10.0, 0.9));
        list.push(BoundingBox::with_confidence(0.0, 0.0, 10.0, 10.0, 0.3));
        let filtered = list.filter_by_confidence(0.5);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_bounding_box_list_nms_removes_overlapping() {
        let mut list = BoundingBoxList::new();
        list.push(BoundingBox::with_confidence(0.0, 0.0, 10.0, 10.0, 0.9));
        // Nearly identical box — high overlap.
        list.push(BoundingBox::with_confidence(0.5, 0.5, 10.0, 10.0, 0.7));
        let kept = list.nms(0.3);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounding_box_list_nms_keeps_non_overlapping() {
        let mut list = BoundingBoxList::new();
        list.push(BoundingBox::with_confidence(0.0, 0.0, 5.0, 5.0, 0.9));
        list.push(BoundingBox::with_confidence(50.0, 50.0, 5.0, 5.0, 0.8));
        let kept = list.nms(0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_bbox_stats_from_empty() {
        let stats = BboxStats::from_boxes(&[]);
        assert_eq!(stats.count, 0);
        assert!((stats.avg_area() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bbox_stats_avg_area() {
        let boxes = vec![
            BoundingBox::new(0.0, 0.0, 10.0, 10.0), // area 100
            BoundingBox::new(0.0, 0.0, 20.0, 5.0),  // area 100
        ];
        let stats = BboxStats::from_boxes(&boxes);
        assert!((stats.avg_area() - 100.0).abs() < 1e-4);
    }
}

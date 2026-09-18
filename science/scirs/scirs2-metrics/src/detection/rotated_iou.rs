//! Rotated bounding box IoU metrics for 2D object detection.
//!
//! Provides intersection-over-union (IoU) and its generalizations for
//! arbitrarily rotated 2D rectangles.  The Sutherland-Hodgman polygon
//! clipping algorithm is used to compute intersection areas exactly.
//!
//! # Types
//! - [`RotatedBox`] — rotated 2-D bounding box (cx, cy, w, h, θ)
//!
//! # Functions
//! - [`rotated_iou`] — standard IoU
//! - [`rotated_giou`] — Generalized IoU (GIoU)
//! - [`rotated_diou`] — Distance IoU (DIoU)
//! - [`rotated_box_intersection`] — intersection area only
//! - [`rotated_iou_matrix`] — pairwise IoU matrix (single set)
//! - [`rotated_nms`] — Non-Maximum Suppression

use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// RotatedBox
// ─────────────────────────────────────────────────────────────────────────────

/// A 2D rotated bounding box.
///
/// The box is centred at `(cx, cy)` with extent `w × h` and rotated
/// counter-clockwise by `theta` radians from the positive x-axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedBox {
    /// Centre x coordinate.
    pub cx: f64,
    /// Centre y coordinate.
    pub cy: f64,
    /// Width (extent along the local x-axis before rotation).
    pub w: f64,
    /// Height (extent along the local y-axis before rotation).
    pub h: f64,
    /// Rotation angle in radians (counter-clockwise from the positive x-axis).
    pub theta: f64,
}

impl RotatedBox {
    /// Create a new `RotatedBox`.
    #[inline]
    pub fn new(cx: f64, cy: f64, w: f64, h: f64, theta: f64) -> Self {
        Self {
            cx,
            cy,
            w,
            h,
            theta,
        }
    }

    /// Area of the bounding box: `w * h`.
    #[inline]
    pub fn area(&self) -> f64 {
        self.w * self.h
    }

    /// Compute the four corner points of the rotated box.
    ///
    /// Corners are returned in counter-clockwise order starting from the
    /// local `(+w/2, +h/2)` corner:
    ///
    /// ```text
    ///  1 ---- 0
    ///  |      |
    ///  2 ---- 3
    /// ```
    pub fn corners(&self) -> [[f64; 2]; 4] {
        let cos_a = self.theta.cos();
        let sin_a = self.theta.sin();
        let hw = self.w * 0.5;
        let hh = self.h * 0.5;

        // Local corners: (+hw, +hh), (-hw, +hh), (-hw, -hh), (+hw, -hh)
        let local: [[f64; 2]; 4] = [[hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh]];

        let mut out = [[0.0_f64; 2]; 4];
        for (i, lc) in local.iter().enumerate() {
            out[i] = [
                self.cx + cos_a * lc[0] - sin_a * lc[1],
                self.cy + sin_a * lc[0] + cos_a * lc[1],
            ];
        }
        out
    }

    /// Test whether point `(px, py)` lies inside (or on the boundary of) the box.
    ///
    /// Transforms the point into the box's local frame and checks whether it
    /// falls within `[-w/2, w/2] × [-h/2, h/2]`.
    pub fn contains(&self, px: f64, py: f64) -> bool {
        let dx = px - self.cx;
        let dy = py - self.cy;
        let cos_a = self.theta.cos();
        let sin_a = self.theta.sin();
        // Rotate into local frame (inverse rotation = transpose of rotation matrix)
        let local_x = cos_a * dx + sin_a * dy;
        let local_y = -sin_a * dx + cos_a * dy;
        local_x.abs() <= self.w * 0.5 && local_y.abs() <= self.h * 0.5
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Axis-aligned bounding box helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box of a `RotatedBox`.
///
/// Returns `(min_x, min_y, max_x, max_y)`.
fn rotated_box_aabb(b: &RotatedBox) -> (f64, f64, f64, f64) {
    let corners = b.corners();
    let min_x = corners.iter().map(|c| c[0]).fold(f64::INFINITY, f64::min);
    let max_x = corners
        .iter()
        .map(|c| c[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = corners.iter().map(|c| c[1]).fold(f64::INFINITY, f64::min);
    let max_y = corners
        .iter()
        .map(|c| c[1])
        .fold(f64::NEG_INFINITY, f64::max);
    (min_x, min_y, max_x, max_y)
}

/// Smallest AABB that encloses both `a` and `b`.
fn enclosing_aabb(a: &RotatedBox, b: &RotatedBox) -> (f64, f64, f64, f64) {
    let (ax1, ay1, ax2, ay2) = rotated_box_aabb(a);
    let (bx1, by1, bx2, by2) = rotated_box_aabb(b);
    (ax1.min(bx1), ay1.min(by1), ax2.max(bx2), ay2.max(by2))
}

// ─────────────────────────────────────────────────────────────────────────────
// Polygon geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Shoelace formula: signed area of a simple polygon (positive = CCW).
/// Returns the *absolute* area.
fn polygon_area(vertices: &[[f64; 2]]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += vertices[i][0] * vertices[j][1];
        sum -= vertices[j][0] * vertices[i][1];
    }
    sum.abs() * 0.5
}

/// Test whether `p` is on the interior side (or boundary) of directed edge `p1 → p2`
/// using the cross-product sign convention.
#[inline]
fn is_inside_halfplane(p: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> bool {
    let cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]);
    cross >= 0.0
}

/// Compute the intersection point of the infinite line through `p1, p2`
/// with the line segment `a → b`.
///
/// Returns `None` when the lines are (nearly) parallel.
fn line_intersection(a: [f64; 2], b: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> Option<[f64; 2]> {
    let d_ab = [b[0] - a[0], b[1] - a[1]];
    let d_p = [p2[0] - p1[0], p2[1] - p1[1]];

    let denom = d_ab[0] * d_p[1] - d_ab[1] * d_p[0];
    if denom.abs() < 1e-12 {
        return None; // parallel
    }

    let t = ((p1[0] - a[0]) * d_p[1] - (p1[1] - a[1]) * d_p[0]) / denom;
    Some([a[0] + t * d_ab[0], a[1] + t * d_ab[1]])
}

/// Sutherland-Hodgman: clip `polygon` against the half-plane defined by the
/// directed edge `p1 → p2` (points on the *left* / interior side are kept).
fn clip_polygon_by_halfplane(polygon: &[[f64; 2]], p1: [f64; 2], p2: [f64; 2]) -> Vec<[f64; 2]> {
    if polygon.is_empty() {
        return Vec::new();
    }

    let mut output: Vec<[f64; 2]> = Vec::with_capacity(polygon.len() + 1);
    let n = polygon.len();

    for i in 0..n {
        let current = polygon[i];
        let prev = polygon[(i + n - 1) % n];

        let current_inside = is_inside_halfplane(current, p1, p2);
        let prev_inside = is_inside_halfplane(prev, p1, p2);

        match (current_inside, prev_inside) {
            (true, false) => {
                // Entering the half-plane: emit intersection then current
                if let Some(pt) = line_intersection(prev, current, p1, p2) {
                    output.push(pt);
                }
                output.push(current);
            }
            (true, true) => {
                // Both inside: emit current
                output.push(current);
            }
            (false, true) => {
                // Leaving the half-plane: emit only intersection
                if let Some(pt) = line_intersection(prev, current, p1, p2) {
                    output.push(pt);
                }
            }
            (false, false) => {
                // Both outside: emit nothing
            }
        }
    }

    output
}

/// Compute the intersection area of two convex polygons using
/// the Sutherland-Hodgman algorithm.
fn polygon_intersection_area(poly_a: &[[f64; 2]], poly_b: &[[f64; 2]]) -> f64 {
    if poly_a.is_empty() || poly_b.is_empty() {
        return 0.0;
    }

    // Start with poly_a as the subject polygon, clip against each edge of poly_b
    let mut clipped: Vec<[f64; 2]> = poly_a.to_vec();
    let nb = poly_b.len();

    for i in 0..nb {
        if clipped.is_empty() {
            return 0.0;
        }
        let p1 = poly_b[i];
        let p2 = poly_b[(i + 1) % nb];
        clipped = clip_polygon_by_halfplane(&clipped, p1, p2);
    }

    polygon_area(&clipped)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the intersection area of two rotated bounding boxes.
///
/// Uses Sutherland-Hodgman polygon clipping on the corners of each box.
/// Returns the intersection area (≥ 0).
pub fn rotated_box_intersection(box1: &RotatedBox, box2: &RotatedBox) -> f64 {
    let c1: Vec<[f64; 2]> = box1.corners().to_vec();
    let c2: Vec<[f64; 2]> = box2.corners().to_vec();
    polygon_intersection_area(&c1, &c2)
}

/// Intersection-over-Union for two rotated bounding boxes.
///
/// Returns a value in `[0, 1]`.
pub fn rotated_iou(box1: &RotatedBox, box2: &RotatedBox) -> f64 {
    let inter = rotated_box_intersection(box1, box2);
    if inter <= 0.0 {
        return 0.0;
    }
    let union = box1.area() + box2.area() - inter;
    if union <= 0.0 {
        return 0.0;
    }
    (inter / union).clamp(0.0, 1.0)
}

/// Generalized IoU (GIoU) for two rotated bounding boxes.
///
/// ```text
/// GIoU = IoU - |C \ (A ∪ B)| / |C|
/// ```
///
/// where `C` is the axis-aligned bounding box that encloses both rotated
/// boxes.  GIoU ∈ `[-1, 1]`.
pub fn rotated_giou(box1: &RotatedBox, box2: &RotatedBox) -> f64 {
    let inter = rotated_box_intersection(box1, box2);
    let union = box1.area() + box2.area() - inter;

    let (enc_x1, enc_y1, enc_x2, enc_y2) = enclosing_aabb(box1, box2);
    let enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);

    if enc_area <= 0.0 {
        // Degenerate: both boxes have zero area
        return -1.0;
    }

    let iou_val = if union <= 0.0 {
        0.0
    } else {
        (inter / union).clamp(0.0, 1.0)
    };
    let penalty = if enc_area > 0.0 {
        (enc_area - union) / enc_area
    } else {
        0.0
    };
    iou_val - penalty
}

/// Distance IoU (DIoU) for two rotated bounding boxes.
///
/// ```text
/// DIoU = IoU - ρ² / c²
/// ```
///
/// where `ρ` is the Euclidean distance between box centres and `c` is the
/// diagonal of the enclosing AABB.  DIoU ∈ `[-1, 1]`.
pub fn rotated_diou(box1: &RotatedBox, box2: &RotatedBox) -> f64 {
    let inter = rotated_box_intersection(box1, box2);
    let union = box1.area() + box2.area() - inter;

    let iou_val = if union <= 0.0 {
        0.0
    } else {
        (inter / union).clamp(0.0, 1.0)
    };

    // Centre distance squared
    let rho2 = (box1.cx - box2.cx).powi(2) + (box1.cy - box2.cy).powi(2);

    // Enclosing AABB diagonal squared
    let (enc_x1, enc_y1, enc_x2, enc_y2) = enclosing_aabb(box1, box2);
    let c2 = (enc_x2 - enc_x1).powi(2) + (enc_y2 - enc_y1).powi(2);

    if c2 < 1e-15 {
        return iou_val;
    }

    iou_val - rho2 / c2
}

/// Compute the pairwise IoU matrix for a set of rotated boxes.
///
/// Returns a symmetric `n × n` matrix where element `[i][j]` is
/// `rotated_iou(&boxes[i], &boxes[j])`.
pub fn rotated_iou_matrix(boxes: &[RotatedBox]) -> Vec<Vec<f64>> {
    let n = boxes.len();
    let mut mat = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        mat[i][i] = 1.0; // identical boxes → IoU = 1
        for j in (i + 1)..n {
            let v = rotated_iou(&boxes[i], &boxes[j]);
            mat[i][j] = v;
            mat[j][i] = v;
        }
    }
    mat
}

/// Non-Maximum Suppression for rotated bounding boxes.
///
/// Returns the indices of the *kept* boxes (subset of `0..boxes.len()`),
/// sorted by descending score.  A box is suppressed if it has
/// `rotated_iou > iou_threshold` with any higher-scoring kept box.
///
/// Returns an empty vector when `boxes` or `scores` are empty.  If
/// `boxes.len() != scores.len()` an empty vector is also returned (no
/// panic / no unwrap).
pub fn rotated_nms(boxes: &[RotatedBox], scores: &[f64], iou_threshold: f64) -> Vec<usize> {
    if boxes.len() != scores.len() || boxes.is_empty() {
        return Vec::new();
    }

    // Sort indices by descending score
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; boxes.len()];
    let mut kept = Vec::new();

    for &idx in &order {
        if suppressed[idx] {
            continue;
        }
        kept.push(idx);
        for &other in &order {
            if other == idx || suppressed[other] {
                continue;
            }
            let iou_val = rotated_iou(&boxes[idx], &boxes[other]);
            if iou_val > iou_threshold {
                suppressed[other] = true;
            }
        }
    }

    kept
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper ──────────────────────────────────────────────────────────────

    fn axis_aligned_iou(
        x1a: f64,
        y1a: f64,
        x2a: f64,
        y2a: f64,
        x1b: f64,
        y1b: f64,
        x2b: f64,
        y2b: f64,
    ) -> f64 {
        let ix1 = x1a.max(x1b);
        let iy1 = y1a.max(y1b);
        let ix2 = x2a.min(x2b);
        let iy2 = y2a.min(y2b);
        let inter = ((ix2 - ix1).max(0.0)) * ((iy2 - iy1).max(0.0));
        let a_area = (x2a - x1a) * (y2a - y1a);
        let b_area = (x2b - x1b) * (y2b - y1b);
        let union = a_area + b_area - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    // ── Test 1: axis-aligned boxes match standard IoU ───────────────────────

    #[test]
    fn test_axis_aligned_matches_standard_iou() {
        // Box A: centre (1,1), 2x2, θ=0  → x in [0,2], y in [0,2]
        // Box B: centre (2,2), 2x2, θ=0  → x in [1,3], y in [1,3]
        // Standard intersection: 1x1 = 1, union: 4+4-1 = 7, IoU = 1/7
        let a = RotatedBox::new(1.0, 1.0, 2.0, 2.0, 0.0);
        let b = RotatedBox::new(2.0, 2.0, 2.0, 2.0, 0.0);
        let riou = rotated_iou(&a, &b);
        let expected = axis_aligned_iou(0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0);
        assert!(
            (riou - expected).abs() < 1e-6,
            "rotated IoU (θ=0) should equal standard IoU {expected:.6}, got {riou:.6}"
        );
    }

    // ── Test 2: identical boxes → IoU = 1 ──────────────────────────────────

    #[test]
    fn test_identical_boxes_iou_one() {
        let a = RotatedBox::new(3.0, 5.0, 4.0, 6.0, 0.7);
        let b = a;
        let iou = rotated_iou(&a, &b);
        assert!(
            (iou - 1.0).abs() < 1e-9,
            "identical boxes should have IoU=1, got {iou}"
        );
    }

    // ── Test 3: non-overlapping boxes → IoU = 0 ────────────────────────────

    #[test]
    fn test_non_overlapping_iou_zero() {
        let a = RotatedBox::new(0.0, 0.0, 1.0, 1.0, 0.0);
        let b = RotatedBox::new(100.0, 100.0, 1.0, 1.0, 0.0);
        let iou = rotated_iou(&a, &b);
        assert!(
            iou.abs() < 1e-12,
            "non-overlapping boxes should have IoU=0, got {iou}"
        );
    }

    // ── Test 4: 90° rotated square intersects itself → IoU = 1 ─────────────

    #[test]
    fn test_square_90_degree_rotation_iou_one() {
        // A square (w==h) is symmetric under 90° rotation
        let a = RotatedBox::new(0.0, 0.0, 2.0, 2.0, 0.0);
        let b = RotatedBox::new(0.0, 0.0, 2.0, 2.0, PI / 2.0);
        let iou = rotated_iou(&a, &b);
        assert!(
            (iou - 1.0).abs() < 1e-9,
            "90-degree rotation of a square should have IoU=1, got {iou}"
        );
    }

    // ── Test 5: area preserved for 45° rotated box ──────────────────────────

    #[test]
    fn test_area_preserved_under_rotation() {
        let b = RotatedBox::new(0.0, 0.0, 3.0, 4.0, PI / 4.0);
        let expected_area = 3.0 * 4.0;
        assert!(
            (b.area() - expected_area).abs() < 1e-12,
            "area should be w*h regardless of angle, got {}",
            b.area()
        );
    }

    // ── Test 6: GIoU ≤ IoU, and GIoU can be negative ───────────────────────

    #[test]
    fn test_giou_le_iou_and_negative_for_nonoverlap() {
        let a = RotatedBox::new(0.0, 0.0, 1.0, 1.0, 0.0);
        let b = RotatedBox::new(5.0, 5.0, 1.0, 1.0, 0.0);
        let iou_val = rotated_iou(&a, &b);
        let giou_val = rotated_giou(&a, &b);
        assert!(
            giou_val <= iou_val + 1e-9,
            "GIoU should be ≤ IoU; got GIoU={giou_val}, IoU={iou_val}"
        );
        assert!(
            giou_val < 0.0,
            "GIoU should be negative for non-overlapping boxes, got {giou_val}"
        );
    }

    // ── Test 7: NMS removes overlapping boxes ──────────────────────────────

    #[test]
    fn test_nms_removes_overlapping() {
        let boxes = vec![
            RotatedBox::new(0.0, 0.0, 2.0, 2.0, 0.0),   // high score
            RotatedBox::new(0.1, 0.0, 2.0, 2.0, 0.0),   // heavy overlap, lower score
            RotatedBox::new(20.0, 20.0, 2.0, 2.0, 0.0), // far away
        ];
        let scores = vec![0.9, 0.8, 0.7];
        let kept = rotated_nms(&boxes, &scores, 0.5);
        assert!(kept.contains(&0), "highest-scoring box should be kept");
        assert!(kept.contains(&2), "distant box should be kept");
        assert!(
            !kept.contains(&1),
            "overlapping lower-score box should be suppressed"
        );
    }

    // ── Test 8: IoU matrix is symmetric ────────────────────────────────────

    #[test]
    fn test_iou_matrix_symmetric() {
        let boxes = vec![
            RotatedBox::new(0.0, 0.0, 2.0, 2.0, 0.0),
            RotatedBox::new(1.0, 0.0, 2.0, 2.0, 0.3),
            RotatedBox::new(5.0, 5.0, 1.0, 3.0, PI / 6.0),
        ];
        let mat = rotated_iou_matrix(&boxes);
        let n = boxes.len();
        for i in 0..n {
            for j in 0..n {
                let diff = (mat[i][j] - mat[j][i]).abs();
                assert!(
                    diff < 1e-12,
                    "IoU matrix should be symmetric: mat[{i}][{j}]={} but mat[{j}][{i}]={}",
                    mat[i][j],
                    mat[j][i]
                );
            }
        }
    }

    // ── Test 9: corners() returns 4 distinct points ─────────────────────────

    #[test]
    fn test_corners_distinct_for_nondegenerate_box() {
        let b = RotatedBox::new(1.0, 2.0, 3.0, 4.0, PI / 5.0);
        let corners = b.corners();
        for i in 0..4 {
            for j in (i + 1)..4 {
                let dx = corners[i][0] - corners[j][0];
                let dy = corners[i][1] - corners[j][1];
                let dist = (dx * dx + dy * dy).sqrt();
                assert!(
                    dist > 1e-6,
                    "corners {i} and {j} should be distinct, distance={dist}"
                );
            }
        }
    }

    // ── Test 10: polygon_area of unit square = 1.0 ──────────────────────────

    #[test]
    fn test_polygon_area_unit_square() {
        let square: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let area = polygon_area(&square);
        assert!(
            (area - 1.0).abs() < 1e-12,
            "unit square area should be 1.0, got {area}"
        );
    }

    // ── Additional: polygon_area of triangle ────────────────────────────────

    #[test]
    fn test_polygon_area_triangle() {
        let tri: [[f64; 2]; 3] = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]];
        let area = polygon_area(&tri);
        assert!(
            (area - 6.0).abs() < 1e-12,
            "right triangle area should be 6, got {area}"
        );
    }

    // ── Additional: contains() basic checks ────────────────────────────────

    #[test]
    fn test_contains_centre() {
        let b = RotatedBox::new(5.0, 5.0, 4.0, 6.0, 0.0);
        assert!(b.contains(5.0, 5.0), "centre should be inside");
    }

    #[test]
    fn test_contains_outside() {
        let b = RotatedBox::new(0.0, 0.0, 2.0, 2.0, 0.0);
        assert!(
            !b.contains(2.0, 2.0),
            "corner just outside should not be inside"
        );
        assert!(!b.contains(5.0, 0.0), "far point should not be inside");
    }

    #[test]
    fn test_contains_rotated() {
        // 90-degree rotated 4x2 box: local x→y, local y→-x
        // Centre at origin; local half-extents: w/2=2, h/2=1
        // After 90-deg CCW rotation, local-x axis becomes world-y axis
        // A point at world (0, 1.5) is at local (1.5 along local-x), but local-x is world-y
        let b = RotatedBox::new(0.0, 0.0, 4.0, 2.0, PI / 2.0);
        // In world space after 90-deg rotation: x-extent is now h/2=1, y-extent is w/2=2
        assert!(
            b.contains(0.0, 1.5),
            "point within rotated extents should be inside"
        );
        assert!(
            !b.contains(1.5, 0.0),
            "point outside rotated extents should be outside"
        );
    }

    // ── Additional: NMS with mismatched lengths returns empty ───────────────

    #[test]
    fn test_nms_length_mismatch_returns_empty() {
        let boxes = vec![RotatedBox::new(0.0, 0.0, 1.0, 1.0, 0.0)];
        let scores = vec![0.9, 0.8]; // mismatch
        let kept = rotated_nms(&boxes, &scores, 0.5);
        assert!(
            kept.is_empty(),
            "mismatched lengths should return empty vec"
        );
    }

    // ── Additional: rotated_diou ≤ rotated_iou ──────────────────────────────

    #[test]
    fn test_diou_le_iou() {
        let a = RotatedBox::new(0.0, 0.0, 4.0, 4.0, 0.0);
        let b = RotatedBox::new(2.0, 0.0, 4.0, 4.0, 0.3);
        let iou_val = rotated_iou(&a, &b);
        let diou_val = rotated_diou(&a, &b);
        assert!(
            diou_val <= iou_val + 1e-9,
            "DIoU should be ≤ IoU; got DIoU={diou_val}, IoU={iou_val}"
        );
    }

    // ── Additional: identical boxes DIoU = IoU = 1 (centre distance=0) ─────

    #[test]
    fn test_diou_identical_boxes() {
        let a = RotatedBox::new(1.0, 2.0, 3.0, 4.0, 0.5);
        let b = a;
        let diou_val = rotated_diou(&a, &b);
        // ρ = 0, so DIoU = IoU = 1
        assert!(
            (diou_val - 1.0).abs() < 1e-9,
            "identical boxes should have DIoU=1, got {diou_val}"
        );
    }
}

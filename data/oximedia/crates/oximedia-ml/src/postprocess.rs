//! Output tensor post-processing utilities.
//!
//! These are small, self-contained helpers that operate on plain
//! `&[f32]` slices. They keep the pipeline layer free of backend-
//! specific tensor plumbing.
//!
//! A small geometric type — [`BoundingBox`] — also lives here so that
//! the NMS / IoU helpers are always available regardless of which
//! pipeline features are enabled.
//!
//! ## Helper overview
//!
//! | Helper                  | Use case                                             |
//! |-------------------------|------------------------------------------------------|
//! | [`softmax`]             | Turn classifier logits into a probability vector.    |
//! | [`argmax`]              | Top-1 class index (errors on empty).                 |
//! | [`top_k`]               | Top-k `(index, score)` pairs, descending.            |
//! | [`sigmoid`]             | Scalar logistic sigmoid.                             |
//! | [`sigmoid_slice`]       | Element-wise sigmoid for multi-label outputs.        |
//! | [`iou`]                 | Pairwise IoU of two [`BoundingBox`]es.               |
//! | [`nms`]                 | Greedy IoU-based Non-Maximum Suppression.            |
//! | [`l2_normalize`]        | In-place L2 unit-normalisation (safe on zero norm).  |
//! | [`cosine_similarity`]   | Cosine similarity; zero on mismatched / empty input. |
//!
//! ## Example
//!
//! ```
//! use oximedia_ml::postprocess::{argmax, softmax, top_k};
//!
//! # fn main() -> oximedia_ml::MlResult<()> {
//! let logits = [0.1_f32, 5.0, 0.3, 0.2];
//! let probs = softmax(&logits);
//! assert_eq!(argmax(&probs)?, 1);
//!
//! let ranked = top_k(&probs, 2)?;
//! assert_eq!(ranked[0].0, 1); // best class
//! # Ok(())
//! # }
//! ```

use crate::error::{MlError, MlResult};

/// Axis-aligned bounding box in corner form.
///
/// Coordinates are in the same space as the detector input (typically
/// normalised 0..=1 or pixel-space 0..=W/H). The type is deliberately
/// side-effect-free — semantic interpretation (pixel vs normalised)
/// is left to the caller.
///
/// # Examples
///
/// ```
/// use oximedia_ml::BoundingBox;
///
/// let b = BoundingBox::from_xywh_center(10.0, 20.0, 4.0, 8.0);
/// assert_eq!(b.width(), 4.0);
/// assert_eq!(b.area(), 32.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingBox {
    /// Top-left X coordinate.
    pub x0: f32,
    /// Top-left Y coordinate.
    pub y0: f32,
    /// Bottom-right X coordinate.
    pub x1: f32,
    /// Bottom-right Y coordinate.
    pub y1: f32,
}

impl BoundingBox {
    /// Construct a new bounding box from corner coordinates.
    #[must_use]
    pub const fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// Width of the box clamped to `>= 0`.
    #[must_use]
    pub fn width(&self) -> f32 {
        (self.x1 - self.x0).max(0.0)
    }

    /// Height of the box clamped to `>= 0`.
    #[must_use]
    pub fn height(&self) -> f32 {
        (self.y1 - self.y0).max(0.0)
    }

    /// Area of the box (0 for degenerate / negative-extent boxes).
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Build a [`BoundingBox`] from YOLO-style centre form (`cx, cy, w, h`).
    #[must_use]
    pub fn from_xywh_center(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        let half_w = w * 0.5;
        let half_h = h * 0.5;
        Self {
            x0: cx - half_w,
            y0: cy - half_h,
            x1: cx + half_w,
            y1: cy + half_h,
        }
    }
}

/// Apply the softmax function along the slice.
///
/// Uses the max-shift trick for numerical stability. If every entry is
/// `-∞` the fallback distribution is uniform. Returns an empty vector
/// when the input is empty.
///
/// # Examples
///
/// ```
/// use oximedia_ml::postprocess::softmax;
///
/// let probs = softmax(&[1.0, 2.0, 3.0]);
/// let sum: f32 = probs.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-5);
/// ```
#[must_use]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        // Degenerate case (all -inf); fall back to uniform.
        let n = exps.len() as f32;
        for e in &mut exps {
            *e = 1.0 / n;
        }
    } else {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

/// Return the index of the largest value in `scores`.
///
/// # Errors
///
/// Returns [`MlError::Postprocess`] if `scores` is empty.
///
/// # Examples
///
/// ```
/// use oximedia_ml::postprocess::argmax;
///
/// # fn main() -> oximedia_ml::MlResult<()> {
/// assert_eq!(argmax(&[0.1, 0.4, 0.2])?, 1);
/// # Ok(())
/// # }
/// ```
pub fn argmax(scores: &[f32]) -> MlResult<usize> {
    if scores.is_empty() {
        return Err(MlError::postprocess("argmax on empty slice"));
    }
    let mut best = 0usize;
    let mut best_v = scores[0];
    for (i, &v) in scores.iter().enumerate().skip(1) {
        if v > best_v {
            best = i;
            best_v = v;
        }
    }
    Ok(best)
}

/// Return the top-`k` `(index, score)` pairs, sorted by descending score.
///
/// When `k == 0` an empty `Vec` is returned (no error). When `k` exceeds
/// `scores.len()` the result is simply truncated to the input length.
///
/// # Errors
///
/// Returns [`MlError::Postprocess`] if `scores` is empty.
///
/// # Examples
///
/// ```
/// use oximedia_ml::postprocess::top_k;
///
/// # fn main() -> oximedia_ml::MlResult<()> {
/// let ranked = top_k(&[0.1, 0.5, 0.3, 0.7, 0.2], 3)?;
/// assert_eq!(ranked[0].0, 3);
/// assert_eq!(ranked[1].0, 1);
/// # Ok(())
/// # }
/// ```
pub fn top_k(scores: &[f32], k: usize) -> MlResult<Vec<(usize, f32)>> {
    if scores.is_empty() {
        return Err(MlError::postprocess("top_k on empty slice"));
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    Ok(indexed)
}

/// Apply the logistic sigmoid to a single value.
#[must_use]
pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Apply sigmoid element-wise to a slice.
#[must_use]
pub fn sigmoid_slice(values: &[f32]) -> Vec<f32> {
    values.iter().copied().map(sigmoid).collect()
}

/// Intersection-over-Union for two bounding boxes in corner form.
///
/// Returns `0.0` if either box has zero or negative area, or if the
/// boxes are disjoint.
#[must_use]
pub fn iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let ix0 = a.x0.max(b.x0);
    let iy0 = a.y0.max(b.y0);
    let ix1 = a.x1.min(b.x1);
    let iy1 = a.y1.min(b.y1);
    let iw = (ix1 - ix0).max(0.0);
    let ih = (iy1 - iy0).max(0.0);
    let inter = iw * ih;
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = a.area();
    let area_b = b.area();
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        return 0.0;
    }
    (inter / union).clamp(0.0, 1.0)
}

/// Greedy Non-Maximum Suppression (NMS) over `(boxes, scores)`.
///
/// * `boxes` and `scores` must have equal length; otherwise an empty
///   `Vec` is returned.
/// * Boxes are processed in descending score order.
/// * Any box whose IoU with an already-kept box exceeds
///   `iou_threshold` is suppressed.
/// * `iou_threshold` is clamped to `0.0..=1.0`.
///
/// Returned indices reference positions in the original `boxes` /
/// `scores` slices, sorted by descending score.
///
/// # Examples
///
/// ```
/// use oximedia_ml::{postprocess::nms, BoundingBox};
///
/// let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
/// let b = BoundingBox::new(1.0, 1.0, 11.0, 11.0);
/// let c = BoundingBox::new(50.0, 50.0, 60.0, 60.0);
/// let kept = nms(&[a, b, c], &[0.9_f32, 0.8, 0.7], 0.5);
/// // The overlapping box is suppressed; `c` is far away so it survives.
/// assert_eq!(kept, vec![0, 2]);
/// ```
#[must_use]
pub fn nms(boxes: &[BoundingBox], scores: &[f32], iou_threshold: f32) -> Vec<usize> {
    if boxes.len() != scores.len() || boxes.is_empty() {
        return Vec::new();
    }
    let threshold = iou_threshold.clamp(0.0, 1.0);

    // Sort indices by descending score.
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<usize> = Vec::with_capacity(order.len());
    for &idx in &order {
        let cand = &boxes[idx];
        if cand.area() <= 0.0 {
            continue;
        }
        let mut suppress = false;
        for &keep_idx in &kept {
            if iou(cand, &boxes[keep_idx]) > threshold {
                suppress = true;
                break;
            }
        }
        if !suppress {
            kept.push(idx);
        }
    }
    kept
}

/// In-place L2 normalisation of a float vector.
///
/// If the input norm is zero (or non-finite) the slice is left
/// untouched. Safe to call on any `&mut [f32]`.
pub fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq <= 0.0 {
        return;
    }
    let inv = norm_sq.sqrt().recip();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Cosine similarity for two equal-length slices.
///
/// Returns `0.0` if either input is empty, the lengths mismatch, or
/// either vector has zero L2 norm.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_empty_is_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn softmax_largest_input_is_largest_output() {
        let probs = softmax(&[0.1, 5.0, 0.3, 0.2]);
        assert!(probs[1] > probs[0]);
        assert!(probs[1] > probs[2]);
        assert!(probs[1] > probs[3]);
    }

    #[test]
    fn argmax_picks_max() {
        let idx = argmax(&[0.1, 0.4, 0.2]).expect("ok");
        assert_eq!(idx, 1);
    }

    #[test]
    fn argmax_empty_errors() {
        let err = argmax(&[]).expect_err("must fail");
        assert!(matches!(err, MlError::Postprocess(_)));
    }

    #[test]
    fn top_k_sorted_descending() {
        let r = top_k(&[0.1, 0.5, 0.3, 0.7, 0.2], 3).expect("ok");
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].0, 3);
        assert_eq!(r[1].0, 1);
        assert_eq!(r[2].0, 2);
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let r = top_k(&[1.0, 2.0], 0).expect("ok");
        assert!(r.is_empty());
    }

    #[test]
    fn sigmoid_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_slice_matches() {
        let v = sigmoid_slice(&[-10.0, 0.0, 10.0]);
        assert!(v[0] < 0.001);
        assert!((v[1] - 0.5).abs() < 1e-6);
        assert!(v[2] > 0.999);
    }

    #[test]
    fn bbox_xywh_center_round_trip() {
        let b = BoundingBox::from_xywh_center(10.0, 20.0, 4.0, 8.0);
        assert!((b.x0 - 8.0).abs() < 1e-5);
        assert!((b.y0 - 16.0).abs() < 1e-5);
        assert!((b.x1 - 12.0).abs() < 1e-5);
        assert!((b.y1 - 24.0).abs() < 1e-5);
        assert!((b.area() - 32.0).abs() < 1e-5);
    }

    #[test]
    fn bbox_negative_extent_has_zero_area() {
        let b = BoundingBox::new(5.0, 5.0, 2.0, 2.0);
        assert_eq!(b.width(), 0.0);
        assert_eq!(b.height(), 0.0);
        assert_eq!(b.area(), 0.0);
    }

    #[test]
    fn iou_identical_boxes_is_one() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((iou(&b, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_zero_area_returns_zero() {
        let a = BoundingBox::new(0.0, 0.0, 0.0, 0.0);
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn nms_handles_length_mismatch() {
        let boxes = vec![BoundingBox::new(0.0, 0.0, 1.0, 1.0)];
        let scores = vec![0.9_f32, 0.8];
        assert!(nms(&boxes, &scores, 0.5).is_empty());
    }

    #[test]
    fn l2_normalize_unit_vector_idempotent() {
        let mut v = vec![3.0_f32, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        // Re-normalising does nothing.
        l2_normalize(&mut v);
        let norm2: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_orthogonal_zero() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_length_mismatch_zero() {
        let a = [1.0_f32, 2.0];
        let b = [1.0_f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}

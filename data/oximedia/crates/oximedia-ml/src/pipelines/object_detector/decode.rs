//! YOLOv8 head decoding.
//!
//! Converts a raw `[1, 4 + num_classes, num_anchors]` output tensor
//! into a ranked `Vec<Detection>` by:
//!
//! 1. Validating the tensor shape (returns
//!    [`crate::error::MlError::Postprocess`] on mismatch).
//! 2. Transposing on the fly so the (anchor, channel) layout is
//!    `row-major-per-anchor` (i.e. every 84 contiguous reads give one
//!    anchor's raw outputs).
//! 3. Running sigmoid+argmax over the class-logit tail.
//! 4. Thresholding by `conf_threshold` and running
//!    [`crate::postprocess::nms`] at `iou_threshold`.
//!
//! This function is exposed for callers that already have an ONNX
//! inference runtime running elsewhere and just want the decode half of
//! the pipeline.

use crate::error::{MlError, MlResult};
use crate::pipelines::types::Detection;
use crate::postprocess::{nms, sigmoid, BoundingBox};

/// Tunable thresholds used by [`decode_yolov8_output`].
#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    /// Number of class logits tailing the 4 box-centre channels.
    pub num_classes: usize,
    /// Minimum post-sigmoid confidence to keep a detection.
    pub conf_threshold: f32,
    /// IoU threshold fed to greedy NMS.
    pub iou_threshold: f32,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            num_classes: 80,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        }
    }
}

/// Decode a flattened YOLOv8 head tensor into a sorted detection list.
///
/// `data` must contain `channels * num_anchors` elements where
/// `channels == 4 + opts.num_classes`. `shape` is validated against
/// `[1, channels, num_anchors]`.
///
/// # Errors
///
/// Returns [`crate::error::MlError::Postprocess`] when `shape` does not
/// match `[1, 4 + opts.num_classes, num_anchors]` or when
/// `data.len() != channels * num_anchors`.
pub fn decode_yolov8_output(
    data: &[f32],
    shape: &[usize],
    opts: &DecodeOptions,
) -> MlResult<Vec<Detection>> {
    let channels = 4 + opts.num_classes;
    let num_anchors = validate_yolov8_shape(shape, channels, data.len())?;

    if num_anchors == 0 {
        return Ok(Vec::new());
    }

    // First pass: threshold + argmax per anchor.
    let mut boxes: Vec<BoundingBox> = Vec::new();
    let mut classes: Vec<u32> = Vec::new();
    let mut scores: Vec<f32> = Vec::new();

    for anchor in 0..num_anchors {
        // Per-channel stride = num_anchors; per-anchor stride = 1.
        // Raw box centres live in channels 0..4.
        let cx = data[anchor];
        let cy = data[num_anchors + anchor];
        let w = data[2 * num_anchors + anchor];
        let h = data[3 * num_anchors + anchor];

        // Class logits live in channels 4..(4 + num_classes).
        let mut best_class = 0_u32;
        let mut best_score = f32::NEG_INFINITY;
        for cls in 0..opts.num_classes {
            let logit = data[(4 + cls) * num_anchors + anchor];
            if logit > best_score {
                best_score = logit;
                best_class = cls as u32;
            }
        }
        let conf = sigmoid(best_score);
        if conf < opts.conf_threshold {
            continue;
        }
        let bbox = BoundingBox::from_xywh_center(cx, cy, w, h);
        if bbox.area() <= 0.0 {
            continue;
        }
        boxes.push(bbox);
        classes.push(best_class);
        scores.push(conf);
    }

    if boxes.is_empty() {
        return Ok(Vec::new());
    }

    // Per-class NMS keeps overlapping detections of different classes.
    let mut keep_mask = vec![false; boxes.len()];
    let mut unique_classes: Vec<u32> = classes.clone();
    unique_classes.sort_unstable();
    unique_classes.dedup();

    for cls in unique_classes {
        let subset: Vec<usize> = classes
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| if c == cls { Some(i) } else { None })
            .collect();
        let sub_boxes: Vec<BoundingBox> = subset.iter().map(|&i| boxes[i]).collect();
        let sub_scores: Vec<f32> = subset.iter().map(|&i| scores[i]).collect();
        let kept = nms(&sub_boxes, &sub_scores, opts.iou_threshold);
        for local_idx in kept {
            keep_mask[subset[local_idx]] = true;
        }
    }

    // Collect kept detections and sort by descending score.
    let mut out: Vec<Detection> = (0..boxes.len())
        .filter(|&i| keep_mask[i])
        .map(|i| Detection::new(boxes[i], classes[i], scores[i]))
        .collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

/// Validate a YOLOv8 output shape and return `num_anchors` on success.
fn validate_yolov8_shape(
    shape: &[usize],
    expected_channels: usize,
    total_len: usize,
) -> MlResult<usize> {
    // Accept either [1, C, A] or [C, A].
    let (channels, anchors) = match shape.len() {
        3 if shape[0] == 1 => (shape[1], shape[2]),
        2 => (shape[0], shape[1]),
        _ => {
            return Err(MlError::postprocess(format!(
                "yolov8: expected rank-3 [1, C, A] or rank-2 [C, A] output, got shape {shape:?}"
            )));
        }
    };
    if channels != expected_channels {
        return Err(MlError::postprocess(format!(
            "yolov8: expected {expected_channels} channels, got {channels} (shape {shape:?})"
        )));
    }
    if channels.saturating_mul(anchors) != total_len {
        return Err(MlError::postprocess(format!(
            "yolov8: output length {total_len} does not match shape {shape:?}"
        )));
    }
    Ok(anchors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fake YOLOv8 output with `num_classes=2` and `num_anchors` anchors.
    ///
    /// Layout (channel-major): [cx₀ cx₁ … cy₀ cy₁ … w₀ w₁ … h₀ h₁ … cls0₀ cls0₁ … cls1₀ cls1₁ …].
    fn build_fake_output(
        anchors: &[(f32, f32, f32, f32, [f32; 2])],
        num_classes: usize,
    ) -> (Vec<f32>, Vec<usize>) {
        let n = anchors.len();
        let channels = 4 + num_classes;
        let mut data = vec![0.0_f32; channels * n];
        for (i, (cx, cy, w, h, cls)) in anchors.iter().enumerate() {
            data[i] = *cx;
            data[n + i] = *cy;
            data[2 * n + i] = *w;
            data[3 * n + i] = *h;
            for (c, &logit) in cls.iter().enumerate() {
                data[(4 + c) * n + i] = logit;
            }
        }
        (data, vec![1, channels, n])
    }

    #[test]
    fn decode_empty_output_returns_empty() {
        let data: Vec<f32> = Vec::new();
        let shape = vec![1, 84, 0];
        let opts = DecodeOptions::default();
        let dets = decode_yolov8_output(&data, &shape, &opts).expect("ok");
        assert!(dets.is_empty());
    }

    #[test]
    fn decode_below_threshold_is_filtered() {
        // sigmoid(-3) ≈ 0.047 < 0.25
        let (data, shape) = build_fake_output(&[(10.0, 10.0, 4.0, 4.0, [-3.0, -5.0])], 2);
        let opts = DecodeOptions {
            num_classes: 2,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        };
        let dets = decode_yolov8_output(&data, &shape, &opts).expect("ok");
        assert!(dets.is_empty());
    }

    #[test]
    fn decode_picks_highest_class() {
        // High logit on class 1.
        let (data, shape) = build_fake_output(&[(10.0, 10.0, 4.0, 4.0, [-2.0, 3.0])], 2);
        let opts = DecodeOptions {
            num_classes: 2,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        };
        let dets = decode_yolov8_output(&data, &shape, &opts).expect("ok");
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 1);
        assert!(dets[0].score > 0.9);
        assert!((dets[0].bbox.x0 - 8.0).abs() < 1e-5);
        assert!((dets[0].bbox.x1 - 12.0).abs() < 1e-5);
    }

    #[test]
    fn decode_nms_suppresses_duplicates_of_same_class() {
        // Two heavily-overlapping boxes on the same class; NMS should keep 1.
        let (data, shape) = build_fake_output(
            &[
                (10.0, 10.0, 4.0, 4.0, [5.0, -5.0]),
                (10.2, 10.0, 4.0, 4.0, [4.0, -5.0]),
            ],
            2,
        );
        let opts = DecodeOptions {
            num_classes: 2,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        };
        let dets = decode_yolov8_output(&data, &shape, &opts).expect("ok");
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 0);
    }

    #[test]
    fn decode_keeps_overlapping_boxes_of_different_classes() {
        // Two heavily-overlapping boxes but of different classes — keep both.
        let (data, shape) = build_fake_output(
            &[
                (10.0, 10.0, 4.0, 4.0, [5.0, -5.0]),
                (10.2, 10.0, 4.0, 4.0, [-5.0, 4.0]),
            ],
            2,
        );
        let opts = DecodeOptions {
            num_classes: 2,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        };
        let dets = decode_yolov8_output(&data, &shape, &opts).expect("ok");
        assert_eq!(dets.len(), 2);
        // Sorted descending by score; first should be the higher-logit box.
        assert!(dets[0].score >= dets[1].score);
    }

    #[test]
    fn decode_rejects_wrong_channel_count() {
        let data = vec![0.0_f32; 84 * 10];
        let shape = vec![1, 50, 10];
        let opts = DecodeOptions::default();
        let err = decode_yolov8_output(&data, &shape, &opts).expect_err("must fail");
        assert!(matches!(err, MlError::Postprocess(_)));
    }

    #[test]
    fn decode_rejects_mismatched_length() {
        let data = vec![0.0_f32; 10];
        let shape = vec![1, 84, 10];
        let opts = DecodeOptions::default();
        let err = decode_yolov8_output(&data, &shape, &opts).expect_err("must fail");
        assert!(matches!(err, MlError::Postprocess(_)));
    }

    #[test]
    fn decode_accepts_rank_two_shape() {
        let (data, shape_3d) = build_fake_output(&[(10.0, 10.0, 4.0, 4.0, [5.0, -5.0])], 2);
        // Drop the leading batch dim.
        let shape_2d: Vec<usize> = shape_3d[1..].to_vec();
        let opts = DecodeOptions {
            num_classes: 2,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        };
        let dets = decode_yolov8_output(&data, &shape_2d, &opts).expect("ok");
        assert_eq!(dets.len(), 1);
    }
}

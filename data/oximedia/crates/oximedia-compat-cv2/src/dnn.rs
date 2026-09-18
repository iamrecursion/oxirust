//! `cv2.dnn` — Deep-neural-network compatibility layer.
//!
//! Wraps an [`oxionnx::Session`] behind an OpenCV-compatible API.  Provides
//! `Net::forward`, `read_net_from_onnx`, `blob_from_image`, and
//! `nms_boxes`, mirroring `cv2.dnn` in Python.
//!
//! # Blob layout convention
//!
//! `blob_from_image` returns a [`Mat`] with `MatType::CV_32FC3`, but the byte
//! buffer is laid out **planar (CHW)** rather than the usual HWC interleaved
//! arrangement of [`MatType::CV_32FC3`].  The shape `[1, channels, rows, cols]`
//! is recoverable from `(channels, rows, cols)` of the `Mat`.  This matches
//! OpenCV's convention where `cv2.dnn.blobFromImage` returns a 4-D NCHW blob.
//!
//! Each `f32` element is encoded as 4 bytes in native-endian order via
//! `f32::to_ne_bytes` / `f32::from_ne_bytes`, keeping the implementation
//! Pure Rust without `bytemuck`.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use oximedia_compat_cv2::{imread, IMREAD_COLOR};
//! use oximedia_compat_cv2::dnn::{blob_from_image, read_net_from_onnx};
//!
//! let img = imread("input.jpg", IMREAD_COLOR).expect("read");
//! let blob = blob_from_image(&img, 1.0 / 255.0, (224, 224), (0.0, 0.0, 0.0), true, false)
//!     .expect("blob");
//! let net = read_net_from_onnx(Path::new("model.onnx")).expect("load");
//! let scores = net.forward(&blob).expect("forward");
//! ```

use std::collections::HashMap;
use std::path::Path;

use oxionnx::{Session, Tensor};

use crate::error::{Cv2Error, Cv2Result};
use crate::geometry::resize as cv2_resize;
use crate::mat::{Mat, MatType, Rect, Size};

// ── Public API ────────────────────────────────────────────────────────────────

/// A loaded neural network ready for inference.
///
/// Wraps [`oxionnx::Session`] together with cached input/output names,
/// matching OpenCV's `cv2.dnn.Net`.
pub struct Net {
    session: Session,
    input_name: String,
    output_names: Vec<String>,
}

impl Net {
    /// Create a `Net` directly from a built [`Session`].
    ///
    /// Caches `session.input_names()[0]` as the default input identifier
    /// (matching cv2.dnn's single-input assumption).
    pub fn new(session: Session) -> Cv2Result<Self> {
        let input_name = session
            .input_names()
            .first()
            .cloned()
            .ok_or_else(|| Cv2Error::Dnn("model has no graph inputs".to_string()))?;
        let output_names = session.output_names().to_vec();
        if output_names.is_empty() {
            return Err(Cv2Error::Dnn("model has no graph outputs".to_string()));
        }
        Ok(Self {
            session,
            input_name,
            output_names,
        })
    }

    /// Run inference using the default input name and return the first output.
    ///
    /// Mirrors `cv2.dnn.Net.forward()` (no arguments).
    pub fn forward(&self, blob: &Mat) -> Cv2Result<Mat> {
        let first = self
            .output_names
            .first()
            .ok_or_else(|| Cv2Error::Dnn("net has no outputs".to_string()))?
            .clone();
        self.forward_named(blob, &first)
    }

    /// Run inference and return a named output tensor as a `Mat`.
    ///
    /// Mirrors `cv2.dnn.Net.forward(outputName)`.
    pub fn forward_named(&self, blob: &Mat, output: &str) -> Cv2Result<Mat> {
        if !self.output_names.iter().any(|n| n == output) {
            return Err(Cv2Error::Dnn(format!(
                "unknown output name: {output:?}; available: {:?}",
                self.output_names
            )));
        }

        let tensor = mat_to_tensor_nchw(blob)?;
        let mut inputs = HashMap::new();
        inputs.insert(self.input_name.as_str(), tensor);
        let outputs = self
            .session
            .run(&inputs)
            .map_err(|e| Cv2Error::Dnn(format!("inference failed: {e}")))?;

        let out_tensor = outputs.get(output).ok_or_else(|| {
            Cv2Error::Dnn(format!("output {output:?} missing from inference result"))
        })?;
        tensor_to_mat(out_tensor)
    }

    /// The default input tensor name (first model input).
    #[must_use]
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// All output tensor names declared by the model.
    #[must_use]
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

/// Read an ONNX model from the given file path.
///
/// Mirrors `cv2.dnn.readNetFromONNX(path)`.  Errors when the file is missing,
/// unreadable, or the model fails to parse.
pub fn read_net_from_onnx(path: &Path) -> Cv2Result<Net> {
    let session = Session::from_file(path)
        .map_err(|e| Cv2Error::Dnn(format!("failed to load ONNX model {}: {e}", path.display())))?;
    Net::new(session)
}

/// Build a 4-D NCHW input blob from an image.
///
/// Mirrors `cv2.dnn.blobFromImage(image, scalefactor, size, mean, swapRB, crop)`.
///
/// Steps (matching OpenCV's order):
/// 1. Resize / center-crop / pad source to `size`.
/// 2. Convert byte pixels to f32.
/// 3. If `swap_rb`, swap channel 0 and channel 2 (BGR ↔ RGB).
/// 4. Subtract per-channel `mean` (indexed in the *output* channel order).
/// 5. Multiply by `scale_factor`.
/// 6. Pack as planar CHW into a `CV_32FC3` `Mat`.
///
/// # Arguments
///
/// * `image` — input `Mat` (`CV_8UC1` or `CV_8UC3`; grayscale is broadcast to 3 channels).
/// * `scale_factor` — multiplier applied after mean subtraction (typically `1.0 / 255.0`).
/// * `size` — `(width, height)` of the output blob.
/// * `mean` — `(c0, c1, c2)` to subtract; channel order matches the post-`swap_rb` layout.
/// * `swap_rb` — swap red and blue channels (cv2 default behaviour for RGB-trained models).
/// * `crop` — if `true`, resize preserving aspect ratio then center-crop to `size`;
///   if `false`, resize directly to `size` (cv2 default).
pub fn blob_from_image(
    image: &Mat,
    scale_factor: f32,
    size: (u32, u32),
    mean: (f32, f32, f32),
    swap_rb: bool,
    crop: bool,
) -> Cv2Result<Mat> {
    let (target_w, target_h) = size;
    if target_w == 0 || target_h == 0 {
        return Err(Cv2Error::Dnn(format!(
            "blob_from_image: target size must be non-zero, got ({target_w}, {target_h})"
        )));
    }
    if image.is_empty() {
        return Err(Cv2Error::Dnn(
            "blob_from_image: input image is empty".to_string(),
        ));
    }

    // 1. Promote grayscale to BGR-3ch.
    let bgr = match image.mat_type {
        MatType::CV_8UC3 => image.clone(),
        MatType::CV_8UC1 => gray_to_bgr(image),
        other => {
            return Err(Cv2Error::Dnn(format!(
                "blob_from_image: unsupported input MatType: {other:?}"
            )));
        }
    };

    // 2. Resize (with optional aspect-preserving crop).
    let resized = if crop {
        crop_resize(&bgr, target_w, target_h)?
    } else {
        let dst = Size {
            width: target_w as usize,
            height: target_h as usize,
        };
        cv2_resize(&bgr, dst, crate::constants::interpolation::INTER_LINEAR)?
    };

    // 3-6. Build planar f32 NCHW buffer.
    let h = resized.rows;
    let w = resized.cols;
    let channels = 3usize;
    let total_elems = channels * h * w;
    let mut planar = vec![0f32; total_elems];

    // Mean indexed by output channel order (after swap_rb).  Mat is BGR by convention,
    // so the source channel for output index `c` is `(2-c)` when `swap_rb`, else `c`.
    let means = [mean.0, mean.1, mean.2];

    for y in 0..h {
        for x in 0..w {
            let px_off = (y * w + x) * 3;
            let b = resized.data[px_off] as f32;
            let g = resized.data[px_off + 1] as f32;
            let r = resized.data[px_off + 2] as f32;
            // After optional swap, output channel order is:
            //   swap_rb=false → [B, G, R]   (cv2 keeps BGR)
            //   swap_rb=true  → [R, G, B]
            let ordered = if swap_rb { [r, g, b] } else { [b, g, r] };
            for (c_idx, &val) in ordered.iter().enumerate() {
                let scaled = (val - means[c_idx]) * scale_factor;
                planar[c_idx * h * w + y * w + x] = scaled;
            }
        }
    }

    // Pack f32 → bytes (native endian).
    let mut bytes = Vec::with_capacity(total_elems * 4);
    for v in &planar {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }

    let step = w * MatType::CV_32FC3.elem_size();
    Ok(Mat {
        data: bytes,
        rows: h,
        cols: w,
        step,
        mat_type: MatType::CV_32FC3,
    })
}

/// Greedy non-maximum suppression on axis-aligned bounding boxes.
///
/// Mirrors `cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)`.
///
/// 1. Drop entries with `scores[i] < score_threshold`.
/// 2. Sort remaining indices by score descending (ties broken by original index ascending).
/// 3. Greedily pick the top-scoring box, then suppress every still-active candidate
///    whose IoU with the picked box exceeds `nms_threshold`.
/// 4. Return picked indices in selection order.
///
/// Empty input → empty output.  Mismatched lengths → empty output (matches OpenCV's
/// silent-return behaviour).
#[must_use]
pub fn nms_boxes(
    boxes: &[Rect],
    scores: &[f32],
    score_threshold: f32,
    nms_threshold: f32,
) -> Vec<usize> {
    if boxes.is_empty() || boxes.len() != scores.len() {
        return Vec::new();
    }

    // Pre-filter by score threshold and sort by score descending.
    let mut order: Vec<usize> = (0..boxes.len())
        .filter(|&i| scores[i] >= score_threshold)
        .collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut suppressed = vec![false; boxes.len()];
    let mut kept: Vec<usize> = Vec::with_capacity(order.len());
    for &i in &order {
        if suppressed[i] {
            continue;
        }
        kept.push(i);
        for &j in &order {
            if j == i || suppressed[j] {
                continue;
            }
            if iou(&boxes[i], &boxes[j]) > nms_threshold {
                suppressed[j] = true;
            }
        }
    }
    kept
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Intersection-over-union of two axis-aligned rectangles.
fn iou(a: &Rect, b: &Rect) -> f32 {
    let area_a = rect_area(a);
    let area_b = rect_area(b);
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }

    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.width).min(b.x + b.width);
    let y1 = (a.y + a.height).min(b.y + b.height);
    let iw = (x1 - x0).max(0);
    let ih = (y1 - y0).max(0);
    let inter = (iw as f32) * (ih as f32);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Area of a rectangle as f32 (negative widths/heights treated as 0).
fn rect_area(r: &Rect) -> f32 {
    let w = r.width.max(0) as f32;
    let h = r.height.max(0) as f32;
    w * h
}

/// Promote a grayscale `CV_8UC1` `Mat` to a 3-channel BGR `Mat` by replicating the
/// luma into each colour plane.
fn gray_to_bgr(src: &Mat) -> Mat {
    let h = src.rows;
    let w = src.cols;
    let mut out = vec![0u8; h * w * 3];
    for (i, &g) in src.data.iter().enumerate() {
        let off = i * 3;
        out[off] = g;
        out[off + 1] = g;
        out[off + 2] = g;
    }
    Mat::from_bgr_bytes(out, h, w)
}

/// Aspect-preserving resize followed by a centre crop to `(target_w, target_h)`.
///
/// Used by `blob_from_image` when `crop=true`, matching OpenCV's
/// `blobFromImage(..., crop=True)` semantics: resize so the *shorter* side
/// matches the target, then crop the centre.
fn crop_resize(src: &Mat, target_w: u32, target_h: u32) -> Cv2Result<Mat> {
    let sw = src.cols as f64;
    let sh = src.rows as f64;
    let tw = target_w as f64;
    let th = target_h as f64;

    // Scale so the shorter side meets the matching target dimension.
    let scale = (tw / sw).max(th / sh);
    let intermediate_w = (sw * scale).round().max(tw) as usize;
    let intermediate_h = (sh * scale).round().max(th) as usize;

    let intermediate = cv2_resize(
        src,
        Size {
            width: intermediate_w,
            height: intermediate_h,
        },
        crate::constants::interpolation::INTER_LINEAR,
    )?;

    // Centre-crop.
    let target_w_us = target_w as usize;
    let target_h_us = target_h as usize;
    let x_off = intermediate_w.saturating_sub(target_w_us) / 2;
    let y_off = intermediate_h.saturating_sub(target_h_us) / 2;
    let mut cropped = vec![0u8; target_w_us * target_h_us * 3];
    for y in 0..target_h_us {
        for x in 0..target_w_us {
            let src_off = ((y + y_off) * intermediate_w + (x + x_off)) * 3;
            let dst_off = (y * target_w_us + x) * 3;
            // src_off is bounded because intermediate_{w,h} >= target_{w,h}
            // (we max'd against the target above) and (x+x_off) < intermediate_w by
            // construction, so the slice access is in-range.
            if src_off + 3 <= intermediate.data.len() {
                cropped[dst_off..dst_off + 3]
                    .copy_from_slice(&intermediate.data[src_off..src_off + 3]);
            }
        }
    }
    Ok(Mat::from_bgr_bytes(cropped, target_h_us, target_w_us))
}

/// Convert a planar-CHW `CV_32FC3` `Mat` (as produced by [`blob_from_image`]) into
/// an oxionnx [`Tensor`] with shape `[1, channels, rows, cols]`.
fn mat_to_tensor_nchw(mat: &Mat) -> Cv2Result<Tensor> {
    if mat.mat_type != MatType::CV_32FC3 {
        return Err(Cv2Error::Dnn(format!(
            "mat_to_tensor_nchw: expected CV_32FC3 blob Mat, got {:?}",
            mat.mat_type
        )));
    }
    let channels = mat.channels();
    let h = mat.rows;
    let w = mat.cols;
    let total_elems = channels * h * w;
    let expected_bytes = total_elems * 4;
    if mat.data.len() != expected_bytes {
        return Err(Cv2Error::Dnn(format!(
            "mat_to_tensor_nchw: byte-length mismatch (got {}, expected {})",
            mat.data.len(),
            expected_bytes
        )));
    }

    let mut data = Vec::with_capacity(total_elems);
    for chunk in mat.data.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
            Cv2Error::Dnn("mat_to_tensor_nchw: f32 byte chunk decode failed".to_string())
        })?;
        data.push(f32::from_ne_bytes(bytes));
    }

    Ok(Tensor::new(data, vec![1, channels, h, w]))
}

/// Convert an oxionnx [`Tensor`] back into a `Mat`.
///
/// Supported shapes:
///
/// * `[N, classes]` (rank 2) → `CV_32FC1` `Mat` of `N × classes` elements (planar f32 bytes).
/// * `[1, C, H, W]` (rank 4) → planar-CHW `CV_32FC3` blob `Mat` (matching `blob_from_image`),
///   only when `C == 3`.  Other channel counts are reported as `CV_32FC1` blob with
///   `rows = C * H`, `cols = W` so the data is round-trippable.
/// * Anything else → `Cv2Error::Dnn`.
fn tensor_to_mat(tensor: &Tensor) -> Cv2Result<Mat> {
    let mut bytes = Vec::with_capacity(tensor.data.len() * 4);
    for v in &tensor.data {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }

    match tensor.shape.as_slice() {
        [n, classes] => {
            let rows = *n;
            let cols = *classes;
            let step = cols * MatType::CV_32FC1.elem_size();
            Ok(Mat {
                data: bytes,
                rows,
                cols,
                step,
                mat_type: MatType::CV_32FC1,
            })
        }
        [1, c, h, w] if *c == 3 => {
            let step = (*w) * MatType::CV_32FC3.elem_size();
            Ok(Mat {
                data: bytes,
                rows: *h,
                cols: *w,
                step,
                mat_type: MatType::CV_32FC3,
            })
        }
        [1, c, h, w] => {
            // Generic fallback: pack non-3-channel feature maps as a planar
            // CV_32FC1 Mat with rows = C * H so callers can recover shape.
            let rows = c * h;
            let cols = *w;
            let step = cols * MatType::CV_32FC1.elem_size();
            Ok(Mat {
                data: bytes,
                rows,
                cols,
                step,
                mat_type: MatType::CV_32FC1,
            })
        }
        other => Err(Cv2Error::Dnn(format!(
            "tensor_to_mat: unsupported tensor shape {other:?}"
        ))),
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_identical_boxes_is_one() {
        let a = Rect {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        };
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_disjoint_boxes_is_zero() {
        let a = Rect {
            x: 0,
            y: 0,
            width: 5,
            height: 5,
        };
        let b = Rect {
            x: 100,
            y: 100,
            width: 5,
            height: 5,
        };
        assert!(iou(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn iou_half_overlap() {
        // Two 10×10 boxes overlapping in a 5×10 strip.
        let a = Rect {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        };
        let b = Rect {
            x: 5,
            y: 0,
            width: 10,
            height: 10,
        };
        // intersection = 50, union = 100 + 100 - 50 = 150 → 1/3
        let v = iou(&a, &b);
        assert!((v - 1.0 / 3.0).abs() < 1e-5, "iou was {v}");
    }

    #[test]
    fn rect_area_negative_dimensions_are_zero() {
        let r = Rect {
            x: 0,
            y: 0,
            width: -5,
            height: 10,
        };
        assert!(rect_area(&r).abs() < 1e-6);
    }

    #[test]
    fn gray_to_bgr_replicates_luma() {
        let mat = Mat::from_gray_bytes(vec![17u8, 250u8], 1, 2);
        let bgr = gray_to_bgr(&mat);
        assert_eq!(bgr.mat_type, MatType::CV_8UC3);
        assert_eq!(bgr.data, vec![17, 17, 17, 250, 250, 250]);
    }
}

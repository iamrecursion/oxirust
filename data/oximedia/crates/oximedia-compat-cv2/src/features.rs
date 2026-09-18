//! Feature detection functions: Harris, Shi-Tomasi, FAST.
//!
//! Provides `good_features_to_track`, `corner_harris`, and `fast_feature_detector`.
//! Algorithms lifted from the PyO3 cv2-compat layer and adapted to the pure-Rust `Mat` API.

use crate::{
    error::{Cv2Error, Cv2Result},
    mat::{Mat, MatType, Point2f},
};

// ── Public types ──────────────────────────────────────────────────────────────

/// OpenCV-compatible keypoint (mirrors `cv2.KeyPoint`).
#[derive(Clone, Debug)]
pub struct KeyPoint {
    /// Position of the keypoint.
    pub pt: Point2f,
    /// Diameter of the meaningful keypoint neighbourhood.
    pub size: f32,
    /// Computed orientation of the keypoint. -1 if not applicable.
    pub angle: f32,
    /// The response by which the most strong keypoints have been selected.
    pub response: f32,
    /// Octave (pyramid layer) from which the keypoint has been extracted.
    pub octave: i32,
    /// Object class (if the keypoints need to be clustered by an object they belong to).
    pub class_id: i32,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// cv2.goodFeaturesToTrack — Shi-Tomasi corner detection.
///
/// Returns up to `max_corners` corners sorted by response strength, filtered
/// by a minimum inter-corner distance `min_distance`.
///
/// # Errors
/// Returns `UnsupportedDtype` if `src` is not `CV_8UC1` or `CV_8UC3`.
pub fn good_features_to_track(
    src: &Mat,
    max_corners: usize,
    quality: f64,
    min_distance: f64,
) -> Cv2Result<Vec<Point2f>> {
    let (data, w, h, ch) = mat_components(src)?;
    let gray = to_gray_f32(data, w, h, ch);

    let responses = compute_shi_tomasi_response(&gray, w, h, 3);

    let max_resp = responses.iter().cloned().fold(0.0f32, f32::max);
    if max_resp <= 0.0 {
        return Ok(vec![]);
    }
    let threshold = max_resp * quality as f32;

    let mut candidates: Vec<(usize, usize, f32)> = responses
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r >= threshold)
        .map(|(idx, &r)| (idx % w, idx / w, r))
        .collect();

    candidates.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<Point2f> = Vec::new();
    for (cx, cy, _) in candidates {
        if selected.len() >= max_corners {
            break;
        }
        let too_close = selected.iter().any(|&s| {
            let dx = cx as f32 - s.x;
            let dy = cy as f32 - s.y;
            (f64::from(dx * dx + dy * dy)).sqrt() < min_distance
        });
        if !too_close {
            selected.push(Point2f {
                x: cx as f32,
                y: cy as f32,
            });
        }
    }

    Ok(selected)
}

/// cv2.cornerHarris — Harris corner response map.
///
/// Returns a `CV_32FC1` Mat where each pixel holds the Harris response value.
///
/// # Errors
/// Returns `UnsupportedDtype` if `src` is not single- or three-channel 8-bit.
pub fn corner_harris(src: &Mat, block_size: i32, _ksize: i32, k: f64) -> Cv2Result<Mat> {
    let (data, w, h, ch) = mat_components(src)?;
    let gray = to_gray_f32(data, w, h, ch);
    let block = (block_size.max(1)) as usize;

    let responses = compute_harris_response(&gray, w, h, block, k);

    // Pack f32 responses into CV_32FC1 Mat bytes
    let mut out_data = Vec::with_capacity(h * w * 4);
    for &v in &responses {
        out_data.extend_from_slice(&v.to_le_bytes());
    }

    Ok(Mat {
        data: out_data,
        rows: h,
        cols: w,
        step: w * 4,
        mat_type: MatType::CV_32FC1,
    })
}

/// cv2.FAST — FAST (Features from Accelerated Segment Test) feature detector.
///
/// Returns keypoints at corners with response above `threshold`.
/// When `non_max_suppression` is `true`, weaker keypoints in a 3×3 neighbourhood
/// are removed.
///
/// # Errors
/// Returns `UnsupportedDtype` if `src` is not single- or three-channel 8-bit.
pub fn fast_feature_detector(
    src: &Mat,
    threshold: i32,
    non_max_suppression: bool,
) -> Cv2Result<Vec<KeyPoint>> {
    let (data, w, h, ch) = mat_components(src)?;
    let gray = to_gray_f32(data, w, h, ch);

    let t = threshold.clamp(0, 255) as f32;
    let edge = 3usize;

    // FAST circle offsets (radius 3)
    let circle: [(isize, isize); 16] = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ];

    let mut kps: Vec<KeyPoint> = Vec::new();

    for y in edge..h.saturating_sub(edge) {
        for x in edge..w.saturating_sub(edge) {
            let center = gray[y * w + x];

            // Quick 4-point test (N, S, E, W)
            let test_4 = [0usize, 4, 8, 12]
                .iter()
                .filter(|&&i| {
                    let (dx, dy) = circle[i];
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    (gray[ny * w + nx] - center).abs() > t
                })
                .count();

            if test_4 < 3 {
                continue;
            }

            // Full FAST-9 test
            let mut n_brighter = 0u32;
            let mut n_darker = 0u32;
            for &(dx, dy) in &circle {
                let nx = (x as isize + dx) as usize;
                let ny = (y as isize + dy) as usize;
                let diff = gray[ny * w + nx] - center;
                if diff > t {
                    n_brighter += 1;
                } else if diff < -t {
                    n_darker += 1;
                }
            }

            if n_brighter >= 9 || n_darker >= 9 {
                let response = (n_brighter.max(n_darker) as f32) * t;
                kps.push(KeyPoint {
                    pt: Point2f {
                        x: x as f32,
                        y: y as f32,
                    },
                    size: 7.0,
                    angle: -1.0,
                    response,
                    octave: 0,
                    class_id: -1,
                });
            }
        }
    }

    if non_max_suppression {
        kps = non_max_suppress(kps, w);
    }

    kps.sort_unstable_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(kps)
}

// ── ORB (Oriented FAST + Rotated BRIEF) ───────────────────────────────────────
//
// Pipeline (Rublee et al., ICCV 2011):
//   1. Build a Gaussian image pyramid with `n_levels` scales.
//   2. Run FAST per level, rescore with Harris, keep top
//      `num_features / n_levels` per level.
//   3. Compute orientation via the intensity centroid (m_10, m_01) over a
//      circular patch of radius `patch_size/2`.
//   4. Smooth the patch and sample 256 pre-defined point pairs rotated by
//      the orientation angle to form a 256-bit BRIEF descriptor.
//   5. Brute-force Hamming match between two descriptor sets.

/// ORB feature detector and descriptor extractor (Oriented FAST + Rotated BRIEF).
///
/// Pipeline:
/// `n_levels` Gaussian pyramid → FAST per level → Harris rescore →
/// intensity-centroid orientation → 256-bit rotated BRIEF descriptor.
///
/// The descriptor `Mat` returned by [`Orb::detect_and_compute`] is `CV_8UC1`
/// with shape `(n_keypoints, 32)` — one 32-byte descriptor per row.
#[derive(Clone, Debug)]
pub struct Orb {
    /// Maximum number of features to keep across all pyramid levels.
    pub num_features: usize,
    /// Pyramid downscale ratio per level (typically 1.2).
    pub scale_factor: f32,
    /// Number of pyramid levels (typically 8).
    pub n_levels: usize,
    /// Pixels of border guard inside which keypoints are discarded.
    pub edge_threshold: usize,
    /// BRIEF patch diameter (typically 31).
    pub patch_size: usize,
    /// FAST corner detection threshold.
    pub fast_threshold: i32,
}

impl Default for Orb {
    fn default() -> Self {
        Self {
            num_features: 500,
            scale_factor: 1.2,
            n_levels: 8,
            edge_threshold: 15,
            patch_size: 31,
            fast_threshold: 20,
        }
    }
}

impl Orb {
    /// Create an ORB detector with the given target feature count and defaults.
    #[must_use]
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            ..Self::default()
        }
    }

    /// Detect keypoints and compute 256-bit BRIEF descriptors in one call.
    ///
    /// When `mask` is `Some`, it must be a `CV_8UC1` `Mat` with the same width
    /// and height as `image`.  Detected keypoints whose level-0 `(x, y)` lies
    /// on a zero pixel of the mask are discarded before descriptor sampling.
    /// Keypoints outside `edge_threshold` from any image edge at their pyramid
    /// level are also dropped before descriptor sampling.
    ///
    /// Returns `(keypoints, descriptors)` where `descriptors` is a `CV_8UC1`
    /// `Mat` of shape `(keypoints.len(), 32)`.
    ///
    /// # Errors
    /// Returns `UnsupportedDtype` if `image` is not 8-bit single- or
    /// three-channel, or if `mask` is supplied but is not `CV_8UC1`.
    /// Returns `SizeMismatch` if `mask` does not match `image` in width/height.
    pub fn detect_and_compute(
        &self,
        image: &Mat,
        mask: Option<&Mat>,
    ) -> Cv2Result<(Vec<KeyPoint>, Mat)> {
        if let Some(mask_mat) = mask {
            ensure_mask_dims(mask_mat, image)?;
        }
        let (data, w, h, ch) = mat_components(image)?;
        let gray = to_gray_f32(data, w, h, ch);

        let n_levels = self.n_levels.max(1);
        let scale = self.scale_factor.max(1.001);
        let target_per_level = self.num_features.div_ceil(n_levels);
        let edge = self.edge_threshold.max(1);
        let half_patch = self.patch_size / 2;

        // Build pyramid (level 0 == original; level k == downsampled by scale^k).
        // All levels are 5-tap Gaussian-smoothed before FAST + BRIEF so the
        // descriptor sees a denoised patch (matches OpenCV's pre-blur step).
        let mut levels: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(n_levels);
        levels.push((gaussian_blur_5(&gray, w, h), w, h));
        for level in 1..n_levels {
            let prev_idx = level - 1;
            let scale_factor = scale;
            let new_w = ((levels[prev_idx].1 as f32) / scale_factor).max(1.0) as usize;
            let new_h = ((levels[prev_idx].2 as f32) / scale_factor).max(1.0) as usize;
            if new_w < 2 * edge + 1 || new_h < 2 * edge + 1 {
                break;
            }
            let down = bilinear_downsample(
                &levels[prev_idx].0,
                levels[prev_idx].1,
                levels[prev_idx].2,
                new_w,
                new_h,
            );
            // 5-tap separable Gaussian smoothing (sigma ~= 1.0)
            let smoothed = gaussian_blur_5(&down, new_w, new_h);
            levels.push((smoothed, new_w, new_h));
        }

        let mut all_kps: Vec<KeyPoint> = Vec::new();

        for (level_idx, (level_gray, lw, lh)) in levels.iter().enumerate() {
            // FAST detection on this level
            let level_kps = fast_detect_internal(
                level_gray,
                *lw,
                *lh,
                self.fast_threshold.clamp(0, 255) as f32,
                edge.max(half_patch),
            );
            if level_kps.is_empty() {
                continue;
            }

            // Harris rescore to filter weak corners
            let harris = compute_harris_response(level_gray, *lw, *lh, 7, 0.04);
            let mut scored: Vec<(usize, usize, f32)> = level_kps
                .into_iter()
                .map(|(x, y)| (x, y, harris[y * *lw + x]))
                .collect();

            // Keep top target_per_level by Harris response
            scored.sort_unstable_by(|a, b| {
                b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(target_per_level);

            // Apply non-max suppression in 3×3 within this level
            scored = nms_local_3x3(scored, *lw);

            // Scale keypoint coordinates back to level-0 image space
            let level_scale = scale.powi(level_idx as i32);
            let kp_size = self.patch_size as f32 * level_scale;
            for (x, y, response) in scored {
                all_kps.push(KeyPoint {
                    pt: Point2f {
                        x: x as f32 * level_scale,
                        y: y as f32 * level_scale,
                    },
                    size: kp_size,
                    angle: -1.0,
                    response,
                    octave: level_idx as i32,
                    class_id: -1,
                });
            }
        }

        // Optional mask: drop any keypoint whose level-0 position falls on a
        // zero pixel of the mask.  Mask shape & dtype were validated above.
        // Filtering happens *before* `truncate(num_features)` so the requested
        // feature budget is satisfied from inside the masked region — matching
        // OpenCV's behaviour where the mask scopes which corners are eligible
        // *before* the global response ranking culls down.
        if let Some(mask_mat) = mask {
            all_kps.retain(|kp| mask_is_set(mask_mat, kp.pt.x, kp.pt.y));
        }

        // Trim to num_features by global response ranking
        all_kps.sort_unstable_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_kps.truncate(self.num_features);

        // Compute orientation + BRIEF descriptors.  Sampling happens on the
        // pyramid level where the keypoint was detected — coordinates are
        // mapped back into level space.
        let mut kept_kps: Vec<KeyPoint> = Vec::with_capacity(all_kps.len());
        let mut desc_bytes: Vec<u8> = Vec::with_capacity(all_kps.len() * 32);

        for kp in &all_kps {
            let level_idx = kp.octave.max(0) as usize;
            if level_idx >= levels.len() {
                continue;
            }
            let (level_gray, lw, lh) = &levels[level_idx];
            let level_scale = scale.powi(level_idx as i32);
            let lx = (kp.pt.x / level_scale).round() as isize;
            let ly = (kp.pt.y / level_scale).round() as isize;

            // Reject keypoints whose patch extends past image edges
            let half_patch_i = half_patch as isize;
            if lx < half_patch_i
                || ly < half_patch_i
                || lx >= (*lw as isize - half_patch_i)
                || ly >= (*lh as isize - half_patch_i)
            {
                continue;
            }

            // Smoothed level for descriptor sampling — apply an extra 5×5 gaussian
            // to reduce noise at sample points.  We re-smooth in a small window
            // rather than the whole image to keep cost local.
            let theta = intensity_centroid_angle(
                level_gray,
                *lw,
                *lh,
                lx as usize,
                ly as usize,
                half_patch,
            );

            let (cos_t, sin_t) = (theta.cos(), theta.sin());

            let mut desc = [0u8; 32];
            for (bit_idx, &(ax, ay, bx, by)) in BRIEF_PAIRS.iter().enumerate() {
                // Rotate (ax, ay) and (bx, by) by theta
                let rax = cos_t * (ax as f32) - sin_t * (ay as f32);
                let ray = sin_t * (ax as f32) + cos_t * (ay as f32);
                let rbx = cos_t * (bx as f32) - sin_t * (by as f32);
                let rby = sin_t * (bx as f32) + cos_t * (by as f32);

                let pax = (lx + rax.round() as isize).clamp(0, *lw as isize - 1) as usize;
                let pay = (ly + ray.round() as isize).clamp(0, *lh as isize - 1) as usize;
                let pbx = (lx + rbx.round() as isize).clamp(0, *lw as isize - 1) as usize;
                let pby = (ly + rby.round() as isize).clamp(0, *lh as isize - 1) as usize;

                let i_a = level_gray[pay * lw + pax];
                let i_b = level_gray[pby * lw + pbx];

                // LSB-first bit packing — matches oximedia-cv::keypoint
                if i_a < i_b {
                    desc[bit_idx / 8] |= 1u8 << (bit_idx % 8);
                }
            }

            kept_kps.push(KeyPoint {
                angle: theta.to_degrees(),
                ..kp.clone()
            });
            desc_bytes.extend_from_slice(&desc);
        }

        let n_kp = kept_kps.len();
        let descriptor_mat = Mat {
            data: desc_bytes,
            rows: n_kp,
            cols: 32,
            step: 32,
            mat_type: MatType::CV_8UC1,
        };

        Ok((kept_kps, descriptor_mat))
    }
}

/// Single result of a brute-force descriptor match.
///
/// Mirrors `cv2.DMatch`: each entry maps a query descriptor to its best train
/// descriptor along with the distance between the two descriptors.  For binary
/// descriptors (NORM_HAMMING / NORM_HAMMING2) the value equals the integer bit
/// count; for float descriptors (NORM_L1 / NORM_L2 / NORM_L2SQR) it is the
/// corresponding floating-point metric.  Range for Hamming: `0.0..=256.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DMatch {
    /// Index of the descriptor in the query set (rows of `query_descriptors`).
    pub query_idx: u32,
    /// Index of the descriptor in the train set (rows of `train_descriptors`).
    pub train_idx: u32,
    /// Distance between the two descriptors (non-negative, norm-dependent units).
    pub distance: f32,
}

/// Validate that a descriptor `Mat` has the dtype appropriate for `norm_type`.
///
/// * NORM_HAMMING / NORM_HAMMING2 → `CV_8UC1` required.
/// * NORM_L1 / NORM_L2 / NORM_L2SQR → `CV_32FC1` required.
///
/// # Errors
/// Returns [`Cv2Error::UnsupportedDtype`] when the dtype does not match.
fn check_descriptor_dtype(mat: &Mat, norm_type: i32) -> Cv2Result<()> {
    use crate::constants::norm_type::{NORM_HAMMING, NORM_HAMMING2, NORM_L1, NORM_L2, NORM_L2SQR};
    match norm_type {
        NORM_HAMMING | NORM_HAMMING2 => {
            if mat.mat_type != MatType::CV_8UC1 {
                return Err(Cv2Error::UnsupportedDtype {
                    mat_type: mat.mat_type,
                });
            }
        }
        NORM_L1 | NORM_L2 | NORM_L2SQR => {
            if mat.mat_type != MatType::CV_32FC1 {
                return Err(Cv2Error::UnsupportedDtype {
                    mat_type: mat.mat_type,
                });
            }
        }
        _ => {
            return Err(Cv2Error::UnsupportedFlag {
                name: "BFMatcher::norm_type",
                value: norm_type,
            });
        }
    }
    Ok(())
}

/// Decode a row of a `CV_32FC1` `Mat` into a `Vec<f32>`.
///
/// Row `i` starts at byte offset `i * cols * 4`.  Each f32 is stored in
/// little-endian byte order, which is what our constructors produce.
///
/// # Errors
/// Returns [`Cv2Error::SizeMismatch`] if the byte buffer is too short.
fn f32_row(mat: &Mat, row_idx: usize) -> Cv2Result<Vec<f32>> {
    let byte_start = row_idx * mat.cols * 4;
    let byte_end = byte_start + mat.cols * 4;
    if byte_end > mat.data.len() {
        return Err(Cv2Error::SizeMismatch {
            expected: (mat.rows, mat.cols),
            actual: (row_idx, mat.cols),
        });
    }
    let bytes = &mat.data[byte_start..byte_end];
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(chunk);
            f32::from_le_bytes(arr)
        })
        .collect();
    Ok(values)
}

/// L1 distance between two float descriptor rows.
///
/// Computes `Σ |q_k − t_k|` over `cols` elements.
fn l1_distance_row_f32(query: &Mat, train: &Mat, qi: usize, ti: usize) -> Cv2Result<f32> {
    let q = f32_row(query, qi)?;
    let t = f32_row(train, ti)?;
    if q.len() != t.len() {
        return Err(Cv2Error::SizeMismatch {
            expected: (query.rows, query.cols),
            actual: (train.rows, train.cols),
        });
    }
    let dist = q
        .iter()
        .zip(t.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>();
    Ok(dist)
}

/// L2 (Euclidean) distance between two float descriptor rows.
///
/// Computes `√Σ (q_k − t_k)²` over `cols` elements.
fn l2_distance_row_f32(query: &Mat, train: &Mat, qi: usize, ti: usize) -> Cv2Result<f32> {
    let q = f32_row(query, qi)?;
    let t = f32_row(train, ti)?;
    if q.len() != t.len() {
        return Err(Cv2Error::SizeMismatch {
            expected: (query.rows, query.cols),
            actual: (train.rows, train.cols),
        });
    }
    let sum_sq = q
        .iter()
        .zip(t.iter())
        .map(|(&a, &b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>();
    Ok(sum_sq.sqrt())
}

/// Squared L2 distance between two float descriptor rows.
///
/// Computes `Σ (q_k − t_k)²` over `cols` elements (no square root).
fn l2sqr_distance_row_f32(query: &Mat, train: &Mat, qi: usize, ti: usize) -> Cv2Result<f32> {
    let q = f32_row(query, qi)?;
    let t = f32_row(train, ti)?;
    if q.len() != t.len() {
        return Err(Cv2Error::SizeMismatch {
            expected: (query.rows, query.cols),
            actual: (train.rows, train.cols),
        });
    }
    let sum_sq = q
        .iter()
        .zip(t.iter())
        .map(|(&a, &b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>();
    Ok(sum_sq)
}

/// Unified distance dispatch for a single query/train row pair.
///
/// Routes to the correct distance function based on `norm_type`.  The
/// descriptor Mats must already have been validated by
/// [`check_descriptor_dtype`] and must have equal column counts.
///
/// # Errors
/// Propagates slice-extraction errors; returns [`Cv2Error::UnsupportedFlag`]
/// for unrecognised norm types (callers gate on this via `new()`).
fn descriptor_distance(
    query: &Mat,
    train: &Mat,
    qi: usize,
    ti: usize,
    norm_type: i32,
) -> Cv2Result<f32> {
    use crate::constants::norm_type::{NORM_HAMMING, NORM_HAMMING2, NORM_L1, NORM_L2, NORM_L2SQR};
    match norm_type {
        NORM_HAMMING | NORM_HAMMING2 => {
            // Binary descriptors — rows are u8 slices of exactly `cols` bytes.
            let q_off = qi * query.cols;
            let t_off = ti * train.cols;
            let q_slice = &query.data[q_off..q_off + query.cols];
            let t_slice = &train.data[t_off..t_off + train.cols];
            if NORM_HAMMING2 == norm_type {
                Ok(hamming2_distance_row(q_slice, t_slice) as f32)
            } else {
                Ok(hamming_distance_bytes(q_slice, t_slice) as f32)
            }
        }
        NORM_L1 => l1_distance_row_f32(query, train, qi, ti),
        NORM_L2 => l2_distance_row_f32(query, train, qi, ti),
        NORM_L2SQR => l2sqr_distance_row_f32(query, train, qi, ti),
        _ => Err(Cv2Error::UnsupportedFlag {
            name: "descriptor_distance::norm_type",
            value: norm_type,
        }),
    }
}

/// `cv2.BFMatcher(NORM_HAMMING).match(query, train)` — brute-force best-match.
///
/// For each row in `query_descriptors`, finds the row in `train_descriptors`
/// with the minimum Hamming distance and emits a `DMatch`.  Both inputs must
/// be `CV_8UC1` mats with `cols == 32` (256-bit descriptors).
///
/// # Errors
/// Returns `UnsupportedDtype` if either input is not `CV_8UC1`, or
/// `SizeMismatch` if descriptor widths differ from 32 bytes.
pub fn bf_match_hamming(
    query_descriptors: &Mat,
    train_descriptors: &Mat,
) -> Cv2Result<Vec<DMatch>> {
    if query_descriptors.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: query_descriptors.mat_type,
        });
    }
    if train_descriptors.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: train_descriptors.mat_type,
        });
    }
    if query_descriptors.cols != 32 || train_descriptors.cols != 32 {
        return Err(Cv2Error::SizeMismatch {
            expected: (query_descriptors.rows, 32),
            actual: (query_descriptors.rows, query_descriptors.cols),
        });
    }
    if train_descriptors.rows == 0 {
        return Ok(Vec::new());
    }

    let mut matches = Vec::with_capacity(query_descriptors.rows);
    for q in 0..query_descriptors.rows {
        let q_off = q * 32;
        let q_slice = &query_descriptors.data[q_off..q_off + 32];

        let mut best_dist = f32::MAX;
        let mut best_idx = 0u32;
        for t in 0..train_descriptors.rows {
            let t_off = t * 32;
            let t_slice = &train_descriptors.data[t_off..t_off + 32];
            let dist = hamming_distance_32(q_slice, t_slice) as f32;
            if dist < best_dist {
                best_dist = dist;
                best_idx = t as u32;
                if best_dist == 0.0 {
                    break;
                }
            }
        }

        matches.push(DMatch {
            query_idx: q as u32,
            train_idx: best_idx,
            distance: best_dist,
        });
    }
    Ok(matches)
}

/// `cv2.ORB_create()` factory — creates a default ORB detector.
///
/// # Errors
/// Currently never fails, but returns `Cv2Result` for cv2 API parity.
pub fn orb_create() -> Cv2Result<Orb> {
    Ok(Orb::default())
}

/// Brute-force descriptor matcher (mirrors `cv2.BFMatcher`).
///
/// Supported `norm_type` values:
///
/// | Constant | Value | Descriptor type |
/// |---|---|---|
/// | `NORM_HAMMING`  | 6 | Binary (ORB / BRIEF / BRISK) — `CV_8UC1` |
/// | `NORM_HAMMING2` | 7 | Binary, 2-bit chunks — `CV_8UC1` |
/// | `NORM_L1`       | 2 | Float — `CV_32FC1` |
/// | `NORM_L2`       | 4 | Float — `CV_32FC1` |
/// | `NORM_L2SQR`    | 5 | Float — `CV_32FC1` |
///
/// Pass any other value and the constructor returns [`Cv2Error::UnsupportedFlag`].
///
/// `cross_check` (off by default) restricts the result of `match_descriptors`
/// to mutual nearest neighbours only — a `(q, t)` pair is kept only if the
/// nearest train descriptor to `q` is `t` *and* the nearest query descriptor
/// to `t` is `q`.  This typically gives more reliable matches at the cost of
/// running the matcher in both directions.
#[derive(Clone, Copy, Debug)]
pub struct BFMatcher {
    /// Distance norm used for matching.  Field is private — construct via [`Self::new`].
    norm_type: i32,
    /// When `true`, [`Self::match_descriptors`] returns mutual best matches
    /// only.  Toggle via [`Self::with_cross_check`].
    cross_check: bool,
}

impl BFMatcher {
    /// Returns the distance norm this matcher was constructed with.
    #[must_use]
    pub fn norm_type(&self) -> i32 {
        self.norm_type
    }

    /// Returns whether cross-check filtering is enabled.
    #[must_use]
    pub fn cross_check(&self) -> bool {
        self.cross_check
    }
}

impl BFMatcher {
    /// Create a new brute-force matcher with the given norm.
    ///
    /// Accepted values: `NORM_HAMMING` (6), `NORM_HAMMING2` (7),
    /// `NORM_L1` (2), `NORM_L2` (4), `NORM_L2SQR` (5).
    ///
    /// # Errors
    /// Returns [`Cv2Error::UnsupportedFlag`] for any other value.
    pub fn new(norm_type: i32) -> Cv2Result<Self> {
        use crate::constants::norm_type::{
            NORM_HAMMING, NORM_HAMMING2, NORM_L1, NORM_L2, NORM_L2SQR,
        };
        match norm_type {
            NORM_HAMMING | NORM_HAMMING2 | NORM_L1 | NORM_L2 | NORM_L2SQR => {}
            _ => {
                return Err(Cv2Error::UnsupportedFlag {
                    name: "BFMatcher::norm_type",
                    value: norm_type,
                });
            }
        }
        Ok(Self {
            norm_type,
            cross_check: false,
        })
    }

    /// Builder-style toggle for cross-check filtering.
    #[must_use]
    pub fn with_cross_check(mut self, cross_check: bool) -> Self {
        self.cross_check = cross_check;
        self
    }

    /// Brute-force best-match (cv2 spelling: `BFMatcher.match`).
    ///
    /// Renamed `match_descriptors` here because `match` is a Rust keyword.
    /// When `cross_check` is enabled, only mutual nearest neighbours survive.
    ///
    /// # Errors
    /// Returns [`Cv2Error::UnsupportedDtype`] if the descriptor dtype does not
    /// match the norm (binary norms require `CV_8UC1`; float norms require
    /// `CV_32FC1`).  For NORM_HAMMING, also propagates errors from
    /// [`bf_match_hamming`] (descriptor width mismatch).
    pub fn match_descriptors(&self, query: &Mat, train: &Mat) -> Cv2Result<Vec<DMatch>> {
        use crate::constants::norm_type::NORM_HAMMING;
        check_descriptor_dtype(query, self.norm_type)?;
        check_descriptor_dtype(train, self.norm_type)?;

        // For NORM_HAMMING the legacy path validates descriptor width (== 32);
        // float norms accept arbitrary column counts.
        if self.norm_type == NORM_HAMMING {
            let forward = bf_match_hamming(query, train)?;
            if !self.cross_check {
                return Ok(forward);
            }
            let backward = bf_match_hamming(train, query)?;
            return Ok(forward
                .into_iter()
                .filter(|m| {
                    backward
                        .iter()
                        .any(|b| b.query_idx == m.train_idx && b.train_idx == m.query_idx)
                })
                .collect());
        }

        // Float / Hamming2 norms: generic brute-force over all column widths.
        if query.cols != train.cols {
            return Err(Cv2Error::SizeMismatch {
                expected: (query.rows, query.cols),
                actual: (train.rows, train.cols),
            });
        }

        let forward = bf_match_generic(query, train, self.norm_type)?;
        if !self.cross_check {
            return Ok(forward);
        }
        let backward = bf_match_generic(train, query, self.norm_type)?;
        Ok(forward
            .into_iter()
            .filter(|m| {
                backward
                    .iter()
                    .any(|b| b.query_idx == m.train_idx && b.train_idx == m.query_idx)
            })
            .collect())
    }

    /// k-nearest-neighbour brute-force match.
    ///
    /// For every row in `query` returns up to `k` `DMatch` entries — the rows
    /// in `train` with the smallest distance under this matcher's norm — sorted
    /// ascending by distance.  The outer `Vec` has length `query.rows`; each
    /// inner `Vec` has length `min(k, train.rows)`.
    ///
    /// Implementation cost is `O(m·n·log k)` (a bounded max-heap per query).
    ///
    /// # Errors
    /// Returns [`Cv2Error::UnsupportedDtype`] if the descriptor dtype does not
    /// match the norm.  For NORM_HAMMING, also returns [`Cv2Error::SizeMismatch`]
    /// if a descriptor row is not 32 bytes wide.
    pub fn knn_match(&self, query: &Mat, train: &Mat, k: usize) -> Cv2Result<Vec<Vec<DMatch>>> {
        use crate::constants::norm_type::NORM_HAMMING;

        check_descriptor_dtype(query, self.norm_type)?;
        check_descriptor_dtype(train, self.norm_type)?;

        // NORM_HAMMING legacy path: requires 32-byte descriptor rows.
        if self.norm_type == NORM_HAMMING {
            if query.cols != 32 || train.cols != 32 {
                return Err(Cv2Error::SizeMismatch {
                    expected: (query.rows, 32),
                    actual: (query.rows, query.cols),
                });
            }
        } else if query.cols != train.cols {
            return Err(Cv2Error::SizeMismatch {
                expected: (query.rows, query.cols),
                actual: (train.rows, train.cols),
            });
        }

        let mut out: Vec<Vec<DMatch>> = Vec::with_capacity(query.rows);
        if k == 0 || train.rows == 0 {
            for _ in 0..query.rows {
                out.push(Vec::new());
            }
            return Ok(out);
        }

        // Bounded max-heap entry: (ordered_bits, train_idx).
        // We store distances as `ordered_f32` bits so the BinaryHeap (max-heap)
        // works correctly.  Non-negative f32 values have the property that their
        // bit representation, when interpreted as u32, preserves ordering.
        use std::collections::BinaryHeap;

        for q in 0..query.rows {
            let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k);

            for t in 0..train.rows {
                let dist = descriptor_distance(query, train, q, t, self.norm_type)?;
                let dist_bits = dist.to_bits();
                if heap.len() < k {
                    heap.push((dist_bits, t as u32));
                } else if let Some(&(top_bits, _)) = heap.peek() {
                    if dist_bits < top_bits {
                        heap.pop();
                        heap.push((dist_bits, t as u32));
                    }
                }
            }

            // Drain the heap into a Vec sorted ascending by distance.  Ties
            // on distance are broken by train_idx (ascending) so the result
            // is stable across runs.
            let mut sorted = heap.into_sorted_vec();
            sorted.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let row: Vec<DMatch> = sorted
                .into_iter()
                .map(|(dist_bits, train_idx)| DMatch {
                    query_idx: q as u32,
                    train_idx,
                    distance: f32::from_bits(dist_bits),
                })
                .collect();
            out.push(row);
        }

        Ok(out)
    }
}

// ── ORB private helpers ───────────────────────────────────────────────────────

/// 256 BRIEF sampling pairs `(x_a, y_a, x_b, y_b)` in the 31×31 patch.
///
/// Generated at compile time by a 64-bit LCG with seed `0x12345678` —
/// matches the deterministic table used by `oximedia_cv::keypoint`
/// (`compute_brief_descriptors`).  See `crates/oximedia-cv/src/keypoint.rs`
/// for the canonical generator.  Range of each offset: `-16..=15`.
const BRIEF_PAIRS: [(i8, i8, i8, i8); 256] = {
    let mut pattern = [(0i8, 0i8, 0i8, 0i8); 256];
    let mut state: u64 = 0x1234_5678;
    let mut i = 0;
    while i < 256 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let a = ((state >> 33) & 0x1F) as i8 - 16;
        let b = ((state >> 38) & 0x1F) as i8 - 16;
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let c = ((state >> 33) & 0x1F) as i8 - 16;
        let d = ((state >> 38) & 0x1F) as i8 - 16;
        pattern[i] = (a, b, c, d);
        i += 1;
    }
    pattern
};

/// Bilinear downsample to an arbitrary `(new_w, new_h)`.
fn bilinear_downsample(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_w * dst_h];
    if dst_w == 0 || dst_h == 0 {
        return dst;
    }
    let sx = src_w as f32 / dst_w as f32;
    let sy = src_h as f32 / dst_h as f32;
    for y in 0..dst_h {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        let y0 = fy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = fy - y0 as f32;
        for x in 0..dst_w {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            let x0 = fx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = fx - x0 as f32;

            let p00 = src[y0 * src_w + x0];
            let p10 = src[y0 * src_w + x1];
            let p01 = src[y1 * src_w + x0];
            let p11 = src[y1 * src_w + x1];

            let top = p00 * (1.0 - wx) + p10 * wx;
            let bot = p01 * (1.0 - wx) + p11 * wx;
            dst[y * dst_w + x] = top * (1.0 - wy) + bot * wy;
        }
    }
    dst
}

/// 5-tap separable Gaussian (kernel = [1,4,6,4,1]/16, sigma ≈ 1.0).
fn gaussian_blur_5(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    if w < 5 || h < 5 {
        return src.to_vec();
    }
    let kernel = [1.0f32, 4.0, 6.0, 4.0, 1.0];
    let norm = 16.0f32;

    // Horizontal pass
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in 0..5 {
                let xi = (x as isize + k as isize - 2).clamp(0, w as isize - 1) as usize;
                acc += kernel[k] * src[y * w + xi];
            }
            tmp[y * w + x] = acc / norm;
        }
    }

    // Vertical pass
    let mut dst = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in 0..5 {
                let yi = (y as isize + k as isize - 2).clamp(0, h as isize - 1) as usize;
                acc += kernel[k] * tmp[yi * w + x];
            }
            dst[y * w + x] = acc / norm;
        }
    }
    dst
}

/// FAST detection variant returning raw `(x, y)` coordinates only.
fn fast_detect_internal(
    gray: &[f32],
    w: usize,
    h: usize,
    threshold: f32,
    edge: usize,
) -> Vec<(usize, usize)> {
    let circle: [(isize, isize); 16] = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ];

    let mut out = Vec::new();
    if w <= 2 * edge || h <= 2 * edge {
        return out;
    }
    for y in edge..h.saturating_sub(edge) {
        for x in edge..w.saturating_sub(edge) {
            let center = gray[y * w + x];
            let test_4 = [0usize, 4, 8, 12]
                .iter()
                .filter(|&&i| {
                    let (dx, dy) = circle[i];
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    (gray[ny * w + nx] - center).abs() > threshold
                })
                .count();
            if test_4 < 3 {
                continue;
            }

            let mut n_brighter = 0u32;
            let mut n_darker = 0u32;
            for &(dx, dy) in &circle {
                let nx = (x as isize + dx) as usize;
                let ny = (y as isize + dy) as usize;
                let diff = gray[ny * w + nx] - center;
                if diff > threshold {
                    n_brighter += 1;
                } else if diff < -threshold {
                    n_darker += 1;
                }
            }
            if n_brighter >= 9 || n_darker >= 9 {
                out.push((x, y));
            }
        }
    }
    out
}

/// Local 3×3 NMS: keep only candidates whose response is the strongest
/// in their 3×3 neighborhood.
fn nms_local_3x3(mut scored: Vec<(usize, usize, f32)>, _w: usize) -> Vec<(usize, usize, f32)> {
    use std::collections::HashMap;
    let mut max_at: HashMap<(usize, usize), f32> = HashMap::with_capacity(scored.len());
    for &(x, y, r) in &scored {
        let entry = max_at.entry((x, y)).or_insert(f32::MIN);
        if r > *entry {
            *entry = r;
        }
    }
    scored.retain(|&(x, y, r)| {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 {
                    continue;
                }
                if let Some(&other) = max_at.get(&(nx as usize, ny as usize)) {
                    if other > r {
                        return false;
                    }
                }
            }
        }
        true
    });
    scored
}

/// Intensity-centroid angle over a circular patch of radius `radius` pixels.
fn intensity_centroid_angle(
    gray: &[f32],
    w: usize,
    h: usize,
    cx: usize,
    cy: usize,
    radius: usize,
) -> f32 {
    let r = radius as isize;
    let r_sq = r * r;
    let mut m_10 = 0.0f64;
    let mut m_01 = 0.0f64;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy > r_sq {
                continue;
            }
            let px = cx as isize + dx;
            let py = cy as isize + dy;
            if px < 0 || py < 0 || px >= w as isize || py >= h as isize {
                continue;
            }
            let i = gray[py as usize * w + px as usize] as f64;
            m_10 += dx as f64 * i;
            m_01 += dy as f64 * i;
        }
    }
    m_01.atan2(m_10) as f32
}

/// Hamming distance between two 32-byte slices using `count_ones`.
fn hamming_distance_32(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), 32);
    debug_assert_eq!(b.len(), 32);
    // Process as four u64 words for popcount efficiency
    let mut acc: u32 = 0;
    for k in 0..4 {
        let off = k * 8;
        let mut wa = [0u8; 8];
        let mut wb = [0u8; 8];
        wa.copy_from_slice(&a[off..off + 8]);
        wb.copy_from_slice(&b[off..off + 8]);
        let av = u64::from_le_bytes(wa);
        let bv = u64::from_le_bytes(wb);
        acc += (av ^ bv).count_ones();
    }
    acc
}

/// Hamming distance between two arbitrary-length byte slices.
///
/// Processes 8-byte words where possible for efficiency, then handles the tail.
fn hamming_distance_bytes(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: u32 = 0;
    let chunks = a.len() / 8;
    for k in 0..chunks {
        let off = k * 8;
        let mut wa = [0u8; 8];
        let mut wb = [0u8; 8];
        wa.copy_from_slice(&a[off..off + 8]);
        wb.copy_from_slice(&b[off..off + 8]);
        acc += (u64::from_le_bytes(wa) ^ u64::from_le_bytes(wb)).count_ones();
    }
    // Tail bytes
    for i in (chunks * 8)..a.len() {
        acc += (a[i] ^ b[i]).count_ones();
    }
    acc
}

/// NORM_HAMMING2 distance: Hamming distance counted on 2-bit chunks.
///
/// Each byte is split into four 2-bit groups; two differing 2-bit groups
/// contribute 1 to the distance regardless of how many bits actually differ.
fn hamming2_distance_row(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: u32 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xor = x ^ y;
        // Each 2-bit nibble is non-zero iff the two 2-bit groups differ.
        // Use two parallel masks to check both bits of each 2-bit pair.
        // Pairs at bit offsets 0,2,4,6.
        let low = xor & 0x55; // bit 0 of each pair
        let high = (xor >> 1) & 0x55; // bit 1 of each pair (shifted to same positions)
        acc += (low | high).count_ones();
    }
    acc
}

/// Generic brute-force best-match supporting all norms.
///
/// Used by [`BFMatcher::match_descriptors`] for non-NORM_HAMMING norms.
fn bf_match_generic(
    query_descriptors: &Mat,
    train_descriptors: &Mat,
    norm_type: i32,
) -> Cv2Result<Vec<DMatch>> {
    if train_descriptors.rows == 0 {
        return Ok(Vec::new());
    }

    let mut matches = Vec::with_capacity(query_descriptors.rows);
    for q in 0..query_descriptors.rows {
        let mut best_dist = f32::MAX;
        let mut best_idx = 0u32;
        for t in 0..train_descriptors.rows {
            let dist = descriptor_distance(query_descriptors, train_descriptors, q, t, norm_type)?;
            if dist < best_dist {
                best_dist = dist;
                best_idx = t as u32;
                if best_dist == 0.0 {
                    break;
                }
            }
        }
        matches.push(DMatch {
            query_idx: q as u32,
            train_idx: best_idx,
            distance: best_dist,
        });
    }
    Ok(matches)
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Ensure that an ORB detector mask matches the corresponding image's WxH
/// and is a single-channel 8-bit mat.
///
/// `cv2.ORB.detectAndCompute` requires the mask to share the same width and
/// height as the input image; OpenCV traditionally accepts only `CV_8UC1`
/// masks.  Mismatched shape ⇒ [`Cv2Error::SizeMismatch`]; wrong dtype ⇒
/// [`Cv2Error::UnsupportedDtype`].
fn ensure_mask_dims(mask: &Mat, image: &Mat) -> Cv2Result<()> {
    if mask.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: mask.mat_type,
        });
    }
    if mask.cols != image.cols || mask.rows != image.rows {
        return Err(Cv2Error::SizeMismatch {
            expected: (image.rows, image.cols),
            actual: (mask.rows, mask.cols),
        });
    }
    Ok(())
}

/// Test whether the mask byte at the rounded `(x, y)` is non-zero.
///
/// Coordinates outside the mask are treated as zero (masked off).  The mask
/// must be `CV_8UC1` — caller is expected to have validated this via
/// [`ensure_mask_dims`].
fn mask_is_set(mask: &Mat, x: f32, y: f32) -> bool {
    let xi = x.round();
    let yi = y.round();
    if !xi.is_finite() || !yi.is_finite() || xi < 0.0 || yi < 0.0 {
        return false;
    }
    let xi = xi as usize;
    let yi = yi as usize;
    if xi >= mask.cols || yi >= mask.rows {
        return false;
    }
    // `at_8u1` debug-asserts CV_8UC1 — the public entry point validates this
    // via ensure_mask_dims, so the assertion never triggers in practice.
    mask.at_8u1(yi, xi) > 0
}

/// Validate Mat type and return (data, width, height, channels).
fn mat_components(mat: &Mat) -> Cv2Result<(&[u8], usize, usize, usize)> {
    match mat.mat_type {
        MatType::CV_8UC1 | MatType::CV_8UC3 | MatType::CV_8UC4 => {
            Ok((&mat.data, mat.cols, mat.rows, mat.channels()))
        }
        _ => Err(Cv2Error::UnsupportedDtype {
            mat_type: mat.mat_type,
        }),
    }
}

fn to_gray_f32(data: &[u8], w: usize, h: usize, ch: usize) -> Vec<f32> {
    let mut gray = vec![0.0f32; w * h];
    if ch == 1 {
        for (i, &v) in data.iter().enumerate() {
            gray[i] = v as f32;
        }
    } else {
        for i in 0..w * h {
            let off = i * ch;
            // BGR order (OpenCV default)
            gray[i] = 0.114 * data[off] as f32
                + 0.587 * data[off + 1] as f32
                + 0.299 * data[off + 2] as f32;
        }
    }
    gray
}

fn sobel_gradients(gray: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let get = |dy: isize, dx: isize| {
                gray[((y as isize + dy) as usize) * w + (x as isize + dx) as usize]
            };
            gx[y * w + x] = -get(-1, -1) + get(-1, 1) - 2.0 * get(0, -1) + 2.0 * get(0, 1)
                - get(1, -1)
                + get(1, 1);
            gy[y * w + x] = -get(-1, -1) - 2.0 * get(-1, 0) - get(-1, 1)
                + get(1, -1)
                + 2.0 * get(1, 0)
                + get(1, 1);
        }
    }
    (gx, gy)
}

fn compute_harris_response(gray: &[f32], w: usize, h: usize, block: usize, k: f64) -> Vec<f32> {
    let k = k as f32;
    let (gx, gy) = sobel_gradients(gray, w, h);
    let mut response = vec![0.0f32; w * h];
    let half = block / 2;

    for y in half..h.saturating_sub(half) {
        for x in half..w.saturating_sub(half) {
            let mut ixx = 0.0f32;
            let mut iyy = 0.0f32;
            let mut ixy = 0.0f32;
            for dy in 0..block {
                for dx in 0..block {
                    let ny = y + dy - half;
                    let nx = x + dx - half;
                    let gi = ny * w + nx;
                    ixx += gx[gi] * gx[gi];
                    iyy += gy[gi] * gy[gi];
                    ixy += gx[gi] * gy[gi];
                }
            }
            let det = ixx * iyy - ixy * ixy;
            let trace = ixx + iyy;
            response[y * w + x] = det - k * trace * trace;
        }
    }
    response
}

fn compute_shi_tomasi_response(gray: &[f32], w: usize, h: usize, block: usize) -> Vec<f32> {
    let (gx, gy) = sobel_gradients(gray, w, h);
    let mut response = vec![0.0f32; w * h];
    let half = block / 2;

    for y in half..h.saturating_sub(half) {
        for x in half..w.saturating_sub(half) {
            let mut ixx = 0.0f32;
            let mut iyy = 0.0f32;
            let mut ixy = 0.0f32;
            for dy in 0..block {
                for dx in 0..block {
                    let ny = y + dy - half;
                    let nx = x + dx - half;
                    let gi = ny * w + nx;
                    ixx += gx[gi] * gx[gi];
                    iyy += gy[gi] * gy[gi];
                    ixy += gx[gi] * gy[gi];
                }
            }
            // Minimum eigenvalue of M = [[ixx, ixy],[ixy, iyy]]
            let trace = ixx + iyy;
            let det = ixx * iyy - ixy * ixy;
            let disc = ((trace * trace / 4.0) - det).max(0.0).sqrt();
            let lambda_min = trace / 2.0 - disc;
            response[y * w + x] = lambda_min.max(0.0);
        }
    }
    response
}

/// Remove weaker keypoints that are dominated by a stronger neighbour in a 3×3 grid.
fn non_max_suppress(mut kps: Vec<KeyPoint>, w: usize) -> Vec<KeyPoint> {
    // Build a response map keyed by (x, y)
    use std::collections::HashMap;
    let mut map: HashMap<(i32, i32), f32> = HashMap::with_capacity(kps.len());
    for kp in &kps {
        let key = (kp.pt.x as i32, kp.pt.y as i32);
        let entry = map.entry(key).or_insert(0.0f32);
        if kp.response > *entry {
            *entry = kp.response;
        }
    }
    let _ = w; // unused after refactor

    kps.retain(|kp| {
        let cx = kp.pt.x as i32;
        let cy = kp.pt.y as i32;
        let r = kp.response;
        // Keep only if this pixel has maximum response in its 3×3 neighbourhood
        let dominated = (-1i32..=1)
            .flat_map(|dy| (-1i32..=1).map(move |dx| (cx + dx, cy + dy)))
            .any(|(nx, ny)| {
                if nx == cx && ny == cy {
                    false
                } else {
                    map.get(&(nx, ny)).copied().unwrap_or(0.0) > r
                }
            });
        !dominated
    });

    kps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_good_features_empty() {
        let mat = Mat::new_8uc1(10, 10);
        let corners = good_features_to_track(&mat, 100, 0.01, 3.0).unwrap();
        // Uniform image has no corners
        assert_eq!(corners.len(), 0);
    }

    #[test]
    fn test_corner_harris_shape() {
        let mat = Mat::new_8uc1(8, 8);
        let out = corner_harris(&mat, 3, 3, 0.04).unwrap();
        assert_eq!(out.mat_type, MatType::CV_32FC1);
        assert_eq!(out.rows, 8);
        assert_eq!(out.cols, 8);
    }

    #[test]
    fn test_fast_no_panic_on_blank() {
        let mat = Mat::new_8uc1(20, 20);
        let kps = fast_feature_detector(&mat, 10, true).unwrap();
        assert!(kps.is_empty());
    }

    // ── ORB tests ────────────────────────────────────────────────────────────

    /// Build a 64×64 grayscale image with several isolated bright dots on a
    /// dark background.  Each dot creates a strong FAST corner because the
    /// surrounding ring is darker than the center on all cardinal points.
    fn make_corner_image_64() -> Mat {
        let mut data = vec![0u8; 64 * 64];
        // A 3×3 bright "plus" at multiple sites — these are textbook FAST
        // hits because the entire FAST ring (radius 3) is dark while the
        // center pixel is bright.
        let sites: &[(usize, usize)] = &[(20, 20), (44, 20), (20, 44), (44, 44), (32, 32)];
        for &(cx, cy) in sites {
            for dy in 0..3 {
                for dx in 0..3 {
                    let x = cx + dx;
                    let y = cy + dy;
                    if x < 64 && y < 64 {
                        data[y * 64 + x] = 230;
                    }
                }
            }
        }
        Mat::from_gray_bytes(data, 64, 64)
    }

    #[test]
    fn test_orb_create_returns_ok() {
        let detector = orb_create().expect("orb_create returns Ok");
        assert_eq!(detector.num_features, 500);
        assert_eq!(detector.n_levels, 8);
        assert_eq!(detector.patch_size, 31);
    }

    #[test]
    fn test_orb_detect_synthetic_corner() {
        let img = make_corner_image_64();
        let orb = Orb {
            num_features: 50,
            n_levels: 1,
            edge_threshold: 16,
            ..Orb::default()
        };
        let (kps, desc) = orb
            .detect_and_compute(&img, None)
            .expect("detect_and_compute");
        assert!(
            !kps.is_empty(),
            "expected at least one keypoint near the corner"
        );
        assert_eq!(desc.cols, 32);
        assert_eq!(desc.mat_type, MatType::CV_8UC1);
        assert_eq!(desc.rows, kps.len());
        // Each descriptor must be 32 bytes
        assert_eq!(desc.data.len(), kps.len() * 32);
    }

    #[test]
    fn test_orb_descriptor_shape() {
        let img = make_corner_image_64();
        let orb = Orb {
            num_features: 10,
            n_levels: 2,
            edge_threshold: 16,
            ..Orb::default()
        };
        let (kps, desc) = orb
            .detect_and_compute(&img, None)
            .expect("detect_and_compute");
        assert_eq!(desc.mat_type, MatType::CV_8UC1);
        assert_eq!(desc.cols, 32);
        assert_eq!(desc.step, 32);
        assert_eq!(desc.rows, kps.len());
    }

    #[test]
    fn test_orb_rotation_invariance() {
        use crate::geometry::{get_rotation_matrix_2d, warp_affine};
        use crate::mat::Size;

        // 96×96 image with isolated bright "dots" — same FAST-friendly
        // pattern as in the synthetic-corner test, scaled up to give the
        // rotated copy enough margin from the borders.
        let mut data = vec![20u8; 96 * 96];
        let sites: &[(usize, usize)] = &[(32, 32), (64, 32), (32, 64), (64, 64), (48, 48)];
        for &(cx, cy) in sites {
            for dy in 0..3 {
                for dx in 0..3 {
                    let x = cx + dx;
                    let y = cy + dy;
                    if x < 96 && y < 96 {
                        data[y * 96 + x] = 230;
                    }
                }
            }
        }
        let original = Mat::from_gray_bytes(data, 96, 96);

        // Rotate by 15° around image center
        let center = Point2f { x: 48.0, y: 48.0 };
        let m = get_rotation_matrix_2d(center, 15.0, 1.0);
        let rotated = warp_affine(
            &original,
            m,
            Size {
                width: 96,
                height: 96,
            },
        )
        .expect("warp_affine");

        let orb = Orb {
            num_features: 30,
            n_levels: 1,
            edge_threshold: 17,
            ..Orb::default()
        };
        let (kps_a, desc_a) = orb
            .detect_and_compute(&original, None)
            .expect("orig detect");
        let (kps_b, desc_b) = orb.detect_and_compute(&rotated, None).expect("rot detect");

        assert!(!kps_a.is_empty(), "no keypoints on original");
        assert!(!kps_b.is_empty(), "no keypoints on rotated");

        // Find the strongest keypoint on each (sorted by response when
        // detect_and_compute returned them).  Best-match its descriptor.
        let matches = bf_match_hamming(&desc_a, &desc_b).expect("match");
        assert!(!matches.is_empty());
        let best_match = matches
            .iter()
            .min_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("at least one match");
        // 15° rotation with level-0 Gaussian-smoothed BRIEF and oriented
        // sampling: the strongest correspondence is empirically ~4 bits on
        // this synthetic dot pattern.  Threshold 50 leaves substantial
        // headroom while still failing loudly if rotation handling
        // regresses.
        assert!(
            best_match.distance < 50.0,
            "rotation invariance broken: best Hamming distance = {} bits",
            best_match.distance
        );
    }

    #[test]
    fn test_bf_hamming_self_match() {
        let img = make_corner_image_64();
        let orb = Orb {
            num_features: 20,
            n_levels: 1,
            edge_threshold: 16,
            ..Orb::default()
        };
        let (_, desc) = orb
            .detect_and_compute(&img, None)
            .expect("detect_and_compute");
        if desc.rows == 0 {
            return; // nothing to match
        }
        let matches = bf_match_hamming(&desc, &desc).expect("self match");
        assert_eq!(matches.len(), desc.rows);
        for (i, m) in matches.iter().enumerate() {
            assert_eq!(m.query_idx as usize, i);
            // The first descriptor whose Hamming distance to query[i] is
            // zero wins; for repeated identical descriptors the first index
            // (lowest j) is selected by `bf_match_hamming`.  Distance must
            // always be zero in self-match.
            assert!(
                m.distance == 0.0,
                "self-match: query {} matched train {} with distance {}",
                m.query_idx,
                m.train_idx,
                m.distance
            );
            // The match must also point to a row whose bytes are bitwise
            // equal to the query — verify by direct comparison.
            let q_off = (m.query_idx as usize) * 32;
            let t_off = (m.train_idx as usize) * 32;
            assert_eq!(&desc.data[q_off..q_off + 32], &desc.data[t_off..t_off + 32]);
        }
    }

    #[test]
    fn test_bf_hamming_distance_range() {
        // Build two descriptor sets from a deterministic LCG, then verify
        // that the average Hamming distance between random 256-bit strings
        // is near 128 (the theoretical mean).
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut next_byte = || -> u8 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        };

        let n = 50usize;
        let mut a = Vec::with_capacity(n * 32);
        let mut b = Vec::with_capacity(n * 32);
        for _ in 0..n * 32 {
            a.push(next_byte());
        }
        for _ in 0..n * 32 {
            b.push(next_byte());
        }
        let mat_a = Mat {
            data: a,
            rows: n,
            cols: 32,
            step: 32,
            mat_type: MatType::CV_8UC1,
        };
        let mat_b = Mat {
            data: b,
            rows: n,
            cols: 32,
            step: 32,
            mat_type: MatType::CV_8UC1,
        };

        let matches = bf_match_hamming(&mat_a, &mat_b).expect("match");
        // Mean of distances should be well below 128 because each query gets
        // its *minimum* of 50 random distances — that minimum is ~96-110 bits
        // for n=50.  Just confirm the range looks sensible (0..=256, not all
        // zero).
        assert_eq!(matches.len(), n);
        let total: f32 = matches.iter().map(|m| m.distance).sum();
        let avg = total / matches.len() as f32;
        assert!(
            avg > 80.0 && avg < 130.0,
            "average Hamming of 50×50 random match was {} (expected ~100)",
            avg
        );
        assert!(matches.iter().all(|m| m.distance <= 256.0));
    }

    #[test]
    fn test_bf_hamming_rejects_wrong_dtype() {
        let mut bad = Mat::new(2, 32, MatType::CV_32FC1);
        // 32 cols of CV_32FC1 → 4 bytes/elem, won't be 32 cols of CV_8UC1.
        bad.cols = 32;
        let good = Mat::new(2, 32, MatType::CV_8UC1);
        let res = bf_match_hamming(&bad, &good);
        assert!(res.is_err());
    }

    /// Mask filled with 0xFF must not change the keypoint set produced by
    /// `Orb::detect_and_compute` — every pixel is allowed.  This guards the
    /// mask-aware code path against accidentally rejecting valid keypoints
    /// (e.g. by reading wrong bytes through stride mismatches).
    #[test]
    fn test_orb_mask_all_set_matches_no_mask() {
        let img = make_corner_image_64();
        let orb = Orb {
            num_features: 30,
            n_levels: 1,
            edge_threshold: 16,
            ..Orb::default()
        };
        let (no_mask_kps, _) = orb
            .detect_and_compute(&img, None)
            .expect("detect without mask");

        // Mask with every pixel set to 0xFF — should yield the same keypoints.
        let mask = Mat::from_gray_bytes(vec![0xFFu8; 64 * 64], 64, 64);
        let (with_mask_kps, _) = orb
            .detect_and_compute(&img, Some(&mask))
            .expect("detect with all-set mask");

        assert_eq!(
            no_mask_kps.len(),
            with_mask_kps.len(),
            "all-set mask should not drop any keypoints"
        );
    }

    #[test]
    fn test_bfmatcher_rejects_unsupported_norm() {
        // NORM_INF (=1) is not a supported matcher norm; it must error.
        let res = BFMatcher::new(crate::constants::norm_type::NORM_INF);
        assert!(res.is_err(), "expected UnsupportedFlag for NORM_INF");
        // NORM_MINMAX (=32) is also unsupported.
        let res2 = BFMatcher::new(crate::constants::norm_type::NORM_MINMAX);
        assert!(res2.is_err(), "expected UnsupportedFlag for NORM_MINMAX");
    }
}

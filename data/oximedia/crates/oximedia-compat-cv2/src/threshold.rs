//! Thresholding — cv2 compatibility functions.
//!
//! Implements `threshold` (with Otsu and Triangle auto-selection) and
//! `adaptiveThreshold` (mean-C and Gaussian-C).

use crate::constants::{THRESH_OTSU, THRESH_TRIANGLE};
use crate::error::{Cv2Error, Cv2Result};
use crate::mat::Mat;

// ── Auto-threshold helpers ────────────────────────────────────────────────────

/// Compute Otsu's optimal threshold via inter-class variance maximisation.
///
/// Returns the threshold value as a `u8`.
pub(crate) fn otsu_threshold(data: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &p in data {
        hist[p as usize] += 1;
    }
    let total = data.len() as f64;
    let mut sum_total = 0.0f64;
    for i in 0..256usize {
        sum_total += i as f64 * hist[i] as f64;
    }
    let mut w_b = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut max_var = 0.0f64;
    let mut best_t_start = 0u8;
    let mut best_t_end = 0u8;
    for t in 0..256usize {
        w_b += hist[t] as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f64 * hist[t] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum_total - sum_b) / w_f;
        let var = w_b * w_f * (m_b - m_f).powi(2);
        if var > max_var {
            max_var = var;
            best_t_start = t as u8;
            best_t_end = t as u8;
        } else if max_var > 0.0 && (var - max_var).abs() / max_var < 1e-9 {
            best_t_end = t as u8;
        }
    }
    // Return midpoint of plateau so bimodal distributions get a threshold between the two peaks
    ((best_t_start as u16 + best_t_end as u16 + 1) / 2) as u8
}

/// Compute the triangle method threshold from a histogram.
///
/// Returns the threshold value as a `u8`.
pub(crate) fn triangle_threshold(data: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &p in data {
        hist[p as usize] += 1;
    }

    // Find occupied range
    let mut min_idx = 0usize;
    let mut max_idx = 255usize;
    for i in 0..256 {
        if hist[i] > 0 {
            min_idx = i;
            break;
        }
    }
    for i in (0..256).rev() {
        if hist[i] > 0 {
            max_idx = i;
            break;
        }
    }
    if min_idx >= max_idx {
        return min_idx as u8;
    }

    // Find histogram peak
    let peak_idx = (min_idx..=max_idx).max_by_key(|&i| hist[i]).unwrap_or(128);

    // Choose the side with the longest run from peak to edge
    let (line_start, line_end) = if peak_idx - min_idx > max_idx - peak_idx {
        (min_idx, peak_idx)
    } else {
        (peak_idx, max_idx)
    };

    let dx = (line_end - line_start) as f64;
    let dy = hist[line_start] as f64 - hist[line_end] as f64;
    let line_len = (dx * dx + dy * dy).sqrt();
    if line_len < 1e-10 {
        return line_start as u8;
    }

    let mut best_t = line_start;
    let mut max_dist = 0.0f64;

    for i in line_start..=line_end {
        let px = (i - line_start) as f64;
        let py = hist[i] as f64 - hist[line_end] as f64;
        let dist = (dy * px - dx * py).abs() / line_len;
        if dist > max_dist {
            max_dist = dist;
            best_t = i;
        }
    }

    best_t as u8
}

// ── Per-pixel threshold application ──────────────────────────────────────────

#[inline]
pub(crate) fn apply_thresh_type(pixel: u8, thresh: u8, maxval: u8, thresh_type_base: i32) -> u8 {
    let v = pixel;
    let t = thresh;
    match thresh_type_base {
        0 => {
            if v > t {
                maxval
            } else {
                0
            }
        } // THRESH_BINARY
        1 => {
            if v > t {
                0
            } else {
                maxval
            }
        } // THRESH_BINARY_INV
        2 => {
            if v > t {
                t
            } else {
                v
            }
        } // THRESH_TRUNC
        3 => {
            if v > t {
                v
            } else {
                0
            }
        } // THRESH_TOZERO
        4 => {
            if v > t {
                0
            } else {
                v
            }
        } // THRESH_TOZERO_INV
        _ => v,
    }
}

// ── Gaussian kernel weights ───────────────────────────────────────────────────

#[inline]
fn gaussian_weight(dx: i64, dy: i64, sigma: f64) -> f64 {
    let r2 = (dx * dx + dy * dy) as f64;
    (-r2 / (2.0 * sigma * sigma)).exp()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// `cv2.threshold` — apply a fixed-level threshold to each array element.
///
/// Returns `(thresh_value, dst)` where `thresh_value` is the threshold
/// actually applied (useful when `THRESH_OTSU` or `THRESH_TRIANGLE` is ORed
/// into `thresh_type`).
///
/// `src` must be `CV_8UC1`.
///
/// # Threshold type flags
/// - `THRESH_BINARY` (0), `THRESH_BINARY_INV` (1), `THRESH_TRUNC` (2),
///   `THRESH_TOZERO` (3), `THRESH_TOZERO_INV` (4)
/// - `THRESH_OTSU` (8) — OR with a base type to enable Otsu auto-selection
/// - `THRESH_TRIANGLE` (16) — OR with a base type to enable triangle method
pub fn threshold(src: &Mat, thresh: f64, max_val: f64, thresh_type: i32) -> Cv2Result<(f64, Mat)> {
    if src.mat_type != crate::mat::MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }
    let data = src.data.as_slice();

    let use_otsu = (thresh_type & THRESH_OTSU) != 0;
    let use_triangle = (thresh_type & THRESH_TRIANGLE) != 0;
    // Mask out the auto-selection bits to get the base operation
    let base_type = thresh_type & !THRESH_OTSU & !THRESH_TRIANGLE;

    let effective_thresh: u8 = if use_otsu {
        otsu_threshold(data)
    } else if use_triangle {
        triangle_threshold(data)
    } else {
        thresh.clamp(0.0, 255.0) as u8
    };

    let maxval_u8 = max_val.clamp(0.0, 255.0) as u8;

    let out: Vec<u8> = data
        .iter()
        .map(|&p| apply_thresh_type(p, effective_thresh, maxval_u8, base_type))
        .collect();

    let dst = Mat::from_gray_bytes(out, src.rows, src.cols);
    Ok((effective_thresh as f64, dst))
}

/// `cv2.adaptiveThreshold` — apply adaptive thresholding to a single-channel image.
///
/// `src` must be `CV_8UC1`.
///
/// # Parameters
/// - `adaptive_method`: `ADAPTIVE_THRESH_MEAN_C` (0) or `ADAPTIVE_THRESH_GAUSSIAN_C` (1)
/// - `thresh_type`: `THRESH_BINARY` (0) or `THRESH_BINARY_INV` (1)
/// - `block_size`: neighbourhood size — must be odd and ≥ 3
/// - `c`: constant subtracted from the computed local mean
pub fn adaptive_threshold(
    src: &Mat,
    max_val: f64,
    adaptive_method: i32,
    thresh_type: i32,
    block_size: i32,
    c: f64,
) -> Cv2Result<Mat> {
    if src.mat_type != crate::mat::MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }
    if block_size < 3 || block_size % 2 == 0 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "adaptiveThreshold block_size",
            value: block_size,
        });
    }

    let data = src.data.as_slice();
    let h = src.rows;
    let w = src.cols;
    let half = (block_size / 2) as i64;
    let sigma = half as f64 / 2.0;
    let maxval_u8 = max_val.clamp(0.0, 255.0) as u8;

    // Pre-compute normalised Gaussian weights if needed
    let gauss_weights: Option<Vec<f64>> = if adaptive_method == 1 {
        let mut wts = Vec::with_capacity(((2 * half + 1) * (2 * half + 1)) as usize);
        let mut wsum = 0.0f64;
        for dr in -half..=half {
            for dc in -half..=half {
                let wt = gaussian_weight(dr, dc, sigma);
                wts.push(wt);
                wsum += wt;
            }
        }
        if wsum > 0.0 {
            for wt in &mut wts {
                *wt /= wsum;
            }
        }
        Some(wts)
    } else {
        None
    };

    let total_in_block = ((2 * half + 1) * (2 * half + 1)) as f64;
    let mut out = vec![0u8; h * w];

    for row in 0..h {
        for col in 0..w {
            let local_mean = match &gauss_weights {
                Some(gwts) => {
                    let mut acc = 0.0f64;
                    let mut kidx = 0usize;
                    for dr in -half..=half {
                        for dc in -half..=half {
                            let r = (row as i64 + dr).clamp(0, h as i64 - 1) as usize;
                            let cc = (col as i64 + dc).clamp(0, w as i64 - 1) as usize;
                            acc += data[r * w + cc] as f64 * gwts[kidx];
                            kidx += 1;
                        }
                    }
                    acc
                }
                None => {
                    let mut acc = 0.0f64;
                    for dr in -half..=half {
                        for dc in -half..=half {
                            let r = (row as i64 + dr).clamp(0, h as i64 - 1) as usize;
                            let cc = (col as i64 + dc).clamp(0, w as i64 - 1) as usize;
                            acc += data[r * w + cc] as f64;
                        }
                    }
                    acc / total_in_block
                }
            };

            let local_thresh = (local_mean - c).clamp(0.0, 255.0) as u8;
            let pixel = data[row * w + col];
            out[row * w + col] = match thresh_type {
                1 => {
                    if pixel > local_thresh {
                        0
                    } else {
                        maxval_u8
                    }
                }
                _ => {
                    if pixel > local_thresh {
                        maxval_u8
                    } else {
                        0
                    }
                }
            };
        }
    }

    Ok(Mat::from_gray_bytes(out, h, w))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{
        THRESH_BINARY, THRESH_BINARY_INV, THRESH_OTSU, THRESH_TOZERO, THRESH_TOZERO_INV,
        THRESH_TRUNC,
    };

    // ── apply_thresh_type ─────────────────────────────────────────────────────

    #[test]
    fn test_thresh_binary_above() {
        assert_eq!(apply_thresh_type(200, 128, 255, THRESH_BINARY), 255);
    }

    #[test]
    fn test_thresh_binary_below() {
        assert_eq!(apply_thresh_type(100, 128, 255, THRESH_BINARY), 0);
    }

    #[test]
    fn test_thresh_binary_inv() {
        assert_eq!(apply_thresh_type(100, 128, 255, THRESH_BINARY_INV), 255);
        assert_eq!(apply_thresh_type(200, 128, 255, THRESH_BINARY_INV), 0);
    }

    #[test]
    fn test_thresh_trunc() {
        assert_eq!(apply_thresh_type(100, 128, 255, THRESH_TRUNC), 100);
        assert_eq!(apply_thresh_type(200, 128, 255, THRESH_TRUNC), 128);
    }

    #[test]
    fn test_thresh_tozero() {
        assert_eq!(apply_thresh_type(100, 128, 255, THRESH_TOZERO), 0);
        assert_eq!(apply_thresh_type(200, 128, 255, THRESH_TOZERO), 200);
    }

    #[test]
    fn test_thresh_tozero_inv() {
        assert_eq!(apply_thresh_type(100, 128, 255, THRESH_TOZERO_INV), 100);
        assert_eq!(apply_thresh_type(200, 128, 255, THRESH_TOZERO_INV), 0);
    }

    // ── otsu_threshold ────────────────────────────────────────────────────────

    #[test]
    fn test_otsu_bimodal_separates_peaks() {
        let mut data = vec![50u8; 50];
        data.extend(vec![200u8; 50]);
        let t = otsu_threshold(&data);
        assert!(t >= 50 && t < 200, "got {t}");
    }

    #[test]
    fn test_otsu_uniform_no_panic() {
        let data = vec![128u8; 100];
        let _ = otsu_threshold(&data);
    }

    #[test]
    fn test_otsu_all_zero() {
        let data = vec![0u8; 200];
        assert_eq!(otsu_threshold(&data), 0);
    }

    // ── triangle_threshold ────────────────────────────────────────────────────

    #[test]
    fn test_triangle_valid_range() {
        let data: Vec<u8> = (0u8..=255).collect();
        let _ = triangle_threshold(&data);
    }

    #[test]
    fn test_triangle_single_value() {
        let data = vec![100u8; 100];
        assert_eq!(triangle_threshold(&data), 100);
    }

    // ── threshold function ────────────────────────────────────────────────────

    #[test]
    fn test_threshold_binary() {
        let data: Vec<u8> = (0..10u8).map(|i| i * 25).collect();
        let mat = Mat::from_gray_bytes(data, 1, 10);
        let (_t, binary) = threshold(&mat, 127.0, 255.0, THRESH_BINARY).expect("threshold");
        // pixel 5 = 125, which is NOT > 127 → 0
        assert_eq!(binary.at_8u1(0, 5), 0);
        // pixel 6 = 150 > 127 → 255
        assert_eq!(binary.at_8u1(0, 6), 255);
    }

    #[test]
    fn test_otsu_threshold_via_function() {
        let mut data = vec![50u8; 20];
        data.extend(vec![200u8; 20]);
        let mat = Mat::from_gray_bytes(data, 1, 40);
        let (t, _) = threshold(&mat, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU).expect("otsu");
        assert!(
            t > 50.0 && t < 200.0,
            "Otsu {t} should be between 50 and 200"
        );
    }

    // ── adaptive_threshold ────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_threshold_mean_c() {
        let data: Vec<u8> = (0..100u8).collect();
        let mat = Mat::from_gray_bytes(data, 10, 10);
        let out = adaptive_threshold(&mat, 255.0, 0, 0, 3, 0.0).expect("adaptive");
        assert_eq!(out.rows, 10);
        assert_eq!(out.cols, 10);
    }

    #[test]
    fn test_adaptive_threshold_gaussian_c() {
        let data = vec![128u8; 100];
        let mat = Mat::from_gray_bytes(data, 10, 10);
        let out = adaptive_threshold(&mat, 255.0, 1, 0, 5, 5.0).expect("adaptive gaussian");
        // With THRESH_BINARY and C=5: pixel(128) > thresh(123) → maxval (255)
        assert!(out.data.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_adaptive_threshold_bad_block_size() {
        let mat = Mat::from_gray_bytes(vec![0u8; 9], 3, 3);
        assert!(adaptive_threshold(&mat, 255.0, 0, 0, 2, 0.0).is_err());
        assert!(adaptive_threshold(&mat, 255.0, 0, 0, 0, 0.0).is_err());
    }
}

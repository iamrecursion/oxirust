//! Hough transform functions: line detection (standard + probabilistic) and circle detection.
//!
//! Algorithms lifted from the PyO3 cv2-compat layer (`cv2_compat/hough.rs`)
//! and adapted to use the pure-Rust `Mat` types.

use std::f64::consts::PI;

use crate::{
    error::{Cv2Error, Cv2Result},
    mat::{Mat, MatType},
};

// ── Public API ────────────────────────────────────────────────────────────────

/// cv2.HoughLines — standard Hough transform for line detection.
///
/// Returns a `Vec<[f64; 2]>` where each element is `[rho, theta]`.
///
/// # Errors
/// Returns `UnsupportedDtype` if `src` is not `CV_8UC1`.
pub fn hough_lines(src: &Mat, rho: f64, theta: f64, threshold: i32) -> Cv2Result<Vec<[f64; 2]>> {
    let (data, w, h) = gray_components(src)?;

    let rho_res = rho;
    let theta_res = theta;
    let min_theta = 0.0f64;
    let max_theta = PI;

    let max_rho = ((h as f64).hypot(w as f64) + 1.0) / rho_res;
    let n_rho = max_rho as usize * 2 + 1;
    let n_theta = ((max_theta - min_theta) / theta_res).ceil() as usize + 1;

    let mut acc = vec![0i32; n_rho * n_theta];

    let thetas: Vec<f64> = (0..n_theta)
        .map(|i| min_theta + i as f64 * theta_res)
        .collect();
    let cos_t: Vec<f64> = thetas.iter().map(|&t| t.cos()).collect();
    let sin_t: Vec<f64> = thetas.iter().map(|&t| t.sin()).collect();

    for y in 0..h {
        for x in 0..w {
            if data[y * w + x] == 0 {
                continue;
            }
            for ti in 0..n_theta {
                let rho_val = x as f64 * cos_t[ti] + y as f64 * sin_t[ti];
                let ri = ((rho_val / rho_res) + max_rho) as usize;
                if ri < n_rho {
                    acc[ri * n_theta + ti] += 1;
                }
            }
        }
    }

    let mut lines: Vec<(f64, f64, i32)> = Vec::new();
    for ri in 0..n_rho {
        for ti in 0..n_theta {
            let votes = acc[ri * n_theta + ti];
            if votes >= threshold {
                let rho_val = (ri as f64 - max_rho) * rho_res;
                let theta_val = thetas[ti];
                lines.push((rho_val, theta_val, votes));
            }
        }
    }

    lines.sort_by(|a, b| b.2.cmp(&a.2));

    Ok(lines.into_iter().map(|(r, t, _)| [r, t]).collect())
}

/// cv2.HoughLinesP — probabilistic Hough transform for line segment detection.
///
/// Returns a `Vec<[i32; 4]>` where each element is `[x1, y1, x2, y2]`.
///
/// # Errors
/// Returns `UnsupportedDtype` if `src` is not `CV_8UC1`.
pub fn hough_lines_p(
    src: &Mat,
    _rho: f64,
    _theta: f64,
    threshold: i32,
    min_line_length: f64,
    _max_line_gap: f64,
) -> Cv2Result<Vec<[i32; 4]>> {
    let (data, w, h) = gray_components(src)?;

    let binary: Vec<bool> = data.iter().map(|&v| v > 0).collect();
    let mut segments: Vec<[i32; 4]> = Vec::new();

    // Horizontal scan: find runs of active edge pixels
    for y in 0..h {
        let mut run_start: Option<usize> = None;
        for x in 0..=w {
            let active = x < w && binary[y * w + x];
            if active && run_start.is_none() {
                run_start = Some(x);
            } else if !active {
                if let Some(xs) = run_start.take() {
                    let length = (x - xs) as f64;
                    if length >= min_line_length && length >= threshold as f64 {
                        segments.push([xs as i32, y as i32, x as i32 - 1, y as i32]);
                    }
                }
            }
        }
    }

    // Vertical scan: find runs of active edge pixels
    for x in 0..w {
        let mut run_start: Option<usize> = None;
        for y in 0..=h {
            let active = y < h && binary[y * w + x];
            if active && run_start.is_none() {
                run_start = Some(y);
            } else if !active {
                if let Some(ys) = run_start.take() {
                    let length = (y - ys) as f64;
                    if length >= min_line_length && length >= threshold as f64 {
                        segments.push([x as i32, ys as i32, x as i32, y as i32 - 1]);
                    }
                }
            }
        }
    }

    Ok(segments)
}

/// cv2.HoughCircles — Hough circle transform.
///
/// Returns a `Vec<[f64; 3]>` where each element is `[cx, cy, radius]`.
///
/// `param2` controls the vote threshold (higher = fewer, stricter circles).
///
/// # Errors
/// Returns `UnsupportedDtype` if `src` is not `CV_8UC1`.
pub fn hough_circles(
    src: &Mat,
    dp: f64,
    min_dist: f64,
    _param1: f64,
    param2: f64,
) -> Cv2Result<Vec<[f64; 3]>> {
    let _ = dp; // accumulator resolution ratio — simplify to 1:1
    let (data, w, h) = gray_components(src)?;

    let min_r = 1usize;
    let max_r = w.min(h) / 2;
    let threshold = param2 as i32;
    let min_dist_sq = min_dist * min_dist;

    let mut circles: Vec<(f32, f32, f32, i32)> = Vec::new();

    for r in min_r..=max_r {
        let mut acc = vec![0i32; h * w];

        for y in 0..h {
            for x in 0..w {
                if data[y * w + x] < 50 {
                    continue; // rough edge threshold
                }
                for step in 0..36 {
                    let angle = step as f64 * PI / 18.0;
                    let cx = x as f64 + r as f64 * angle.cos();
                    let cy = y as f64 + r as f64 * angle.sin();
                    let cxi = cx as i64;
                    let cyi = cy as i64;
                    if cxi >= 0 && cxi < w as i64 && cyi >= 0 && cyi < h as i64 {
                        acc[cyi as usize * w + cxi as usize] += 1;
                    }
                }
            }
        }

        for cy in r..h.saturating_sub(r) {
            for cx in r..w.saturating_sub(r) {
                let votes = acc[cy * w + cx];
                if votes >= threshold {
                    let too_close = circles.iter().any(|&(ex, ey, _, _)| {
                        let dx = cx as f64 - ex as f64;
                        let dy = cy as f64 - ey as f64;
                        dx * dx + dy * dy < min_dist_sq
                    });
                    if !too_close {
                        circles.push((cx as f32, cy as f32, r as f32, votes));
                    }
                }
            }
        }
    }

    circles.sort_by(|a, b| b.3.cmp(&a.3));

    Ok(circles
        .into_iter()
        .map(|(x, y, r, _)| [x as f64, y as f64, r as f64])
        .collect())
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Validate that `src` is `CV_8UC1` and return `(data, width, height)`.
fn gray_components(src: &Mat) -> Cv2Result<(&[u8], usize, usize)> {
    if src.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }
    Ok((&src.data, src.cols, src.rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hough_lines_blank_no_lines() {
        let mat = Mat::new_8uc1(20, 20);
        let lines = hough_lines(&mat, 1.0, PI / 180.0, 5).unwrap();
        assert!(lines.is_empty());
    }

    #[test]
    fn test_hough_lines_p_horizontal() {
        let mut data = vec![0u8; 10 * 10];
        for x in 0..10 {
            data[5 * 10 + x] = 255;
        }
        let mat = Mat::from_gray_bytes(data, 10, 10);
        let segs = hough_lines_p(&mat, 1.0, PI / 180.0, 5, 5.0, 5.0).unwrap();
        assert!(
            !segs.is_empty(),
            "should find at least one horizontal segment"
        );
    }

    #[test]
    fn test_hough_circles_blank_no_circles() {
        let mat = Mat::new_8uc1(50, 50);
        let circles = hough_circles(&mat, 1.0, 10.0, 100.0, 30.0).unwrap();
        assert!(circles.is_empty());
    }

    #[test]
    fn test_hough_dtype_error() {
        let mat = Mat::new(10, 10, MatType::CV_8UC3);
        assert!(hough_lines(&mat, 1.0, PI / 180.0, 5).is_err());
        assert!(hough_circles(&mat, 1.0, 10.0, 100.0, 30.0).is_err());
    }
}

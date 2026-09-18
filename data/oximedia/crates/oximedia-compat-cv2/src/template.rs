//! Template matching: match_template and min_max_loc.
//!
//! Implements the 6 OpenCV match methods (TM_SQDIFF, TM_SQDIFF_NORMED,
//! TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED).
//!
//! Algorithms lifted from the PyO3 cv2-compat layer (`cv2_compat/arithmetic.rs`)
//! and adapted to use the pure-Rust `Mat` types.

use crate::{
    constants::{
        TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED,
    },
    error::{Cv2Error, Cv2Result},
    mat::{Mat, MatType, Point},
};

// ── Public API ────────────────────────────────────────────────────────────────

/// cv2.matchTemplate — slide `templ` over `src` and compute match score.
///
/// Returns a `CV_32FC1` Mat of shape `(src.rows - templ.rows + 1, src.cols - templ.cols + 1)`.
/// Each pixel stores a `f32` score packed as little-endian bytes.
///
/// Method codes: TM_SQDIFF=0, TM_SQDIFF_NORMED=1, TM_CCORR=2, TM_CCORR_NORMED=3,
/// TM_CCOEFF=4, TM_CCOEFF_NORMED=5.
///
/// # Errors
/// * `UnsupportedDtype` — if either Mat is not 8-bit.
/// * `UnsupportedFlag` — for an unknown method code.
/// * `SizeMismatch` — if template is larger than source.
pub fn match_template(src: &Mat, templ: &Mat, method: i32) -> Cv2Result<Mat> {
    // Validate method
    if !matches!(
        method,
        TM_SQDIFF | TM_SQDIFF_NORMED | TM_CCORR | TM_CCORR_NORMED | TM_CCOEFF | TM_CCOEFF_NORMED
    ) {
        return Err(Cv2Error::UnsupportedFlag {
            name: "matchTemplate method",
            value: method,
        });
    }

    // Validate dtypes
    let src_ch = validate_8bit(src)?;
    let tpl_ch = validate_8bit(templ)?;

    if src_ch != tpl_ch {
        return Err(Cv2Error::UnsupportedFlag {
            name: "matchTemplate channel mismatch",
            value: tpl_ch as i32,
        });
    }

    let (ih, iw) = (src.rows, src.cols);
    let (th, tw) = (templ.rows, templ.cols);

    if th > ih || tw > iw {
        return Err(Cv2Error::SizeMismatch {
            expected: (ih, iw),
            actual: (th, tw),
        });
    }

    let result_h = ih - th + 1;
    let result_w = iw - tw + 1;

    let img_gray = to_gray_f32(&src.data, iw, ih, src_ch);
    let tpl_gray = to_gray_f32(&templ.data, tw, th, tpl_ch);

    // Template statistics
    let tpl_sum: f32 = tpl_gray.iter().sum();
    let tpl_mean = tpl_sum / (th * tw) as f32;
    let tpl_sq_sum: f32 = tpl_gray.iter().map(|&v| v * v).sum();
    let tpl_norm = (tpl_sq_sum - tpl_mean * tpl_mean * (th * tw) as f32)
        .max(0.0)
        .sqrt();

    let mut result = vec![0.0f32; result_h * result_w];

    for ry in 0..result_h {
        for rx in 0..result_w {
            let mut sqdiff = 0.0f32;
            let mut ccorr = 0.0f32;
            let mut win_sum = 0.0f32;

            // Pre-compute window mean for CCOEFF variants
            if method == TM_CCOEFF || method == TM_CCOEFF_NORMED {
                for ty in 0..th {
                    for tx in 0..tw {
                        win_sum += img_gray[(ry + ty) * iw + (rx + tx)];
                    }
                }
            }
            let win_mean = win_sum / (th * tw) as f32;

            for ty in 0..th {
                for tx in 0..tw {
                    let iv = img_gray[(ry + ty) * iw + (rx + tx)];
                    let tv = tpl_gray[ty * tw + tx];
                    match method {
                        TM_SQDIFF | TM_SQDIFF_NORMED => sqdiff += (iv - tv) * (iv - tv),
                        TM_CCORR | TM_CCORR_NORMED => ccorr += iv * tv,
                        _ => ccorr += (iv - win_mean) * (tv - tpl_mean),
                    }
                }
            }

            // Window energy for normalised methods
            let win_energy: f32 = if method == TM_SQDIFF_NORMED
                || method == TM_CCORR_NORMED
                || method == TM_CCOEFF_NORMED
            {
                let mut sum = 0.0f32;
                for ty in 0..th {
                    for tx in 0..tw {
                        let v = if method == TM_CCOEFF_NORMED {
                            img_gray[(ry + ty) * iw + (rx + tx)] - win_mean
                        } else {
                            img_gray[(ry + ty) * iw + (rx + tx)]
                        };
                        sum += v * v;
                    }
                }
                sum
            } else {
                0.0
            };

            result[ry * result_w + rx] = match method {
                TM_SQDIFF => sqdiff,
                TM_SQDIFF_NORMED => {
                    let denom = (win_energy * tpl_sq_sum).sqrt();
                    if denom > 1e-9 {
                        sqdiff / denom
                    } else {
                        0.0
                    }
                }
                TM_CCORR => ccorr,
                TM_CCORR_NORMED => {
                    let denom = (win_energy * tpl_sq_sum).sqrt();
                    if denom > 1e-9 {
                        ccorr / denom
                    } else {
                        0.0
                    }
                }
                TM_CCOEFF => ccorr,
                TM_CCOEFF_NORMED => {
                    let denom = win_energy.sqrt() * tpl_norm;
                    if denom > 1e-9 {
                        ccorr / denom
                    } else {
                        0.0
                    }
                }
                _ => unreachable!("method validated above"),
            };
        }
    }

    // Pack f32 values into CV_32FC1 Mat
    let mut out_data = Vec::with_capacity(result_h * result_w * 4);
    for &v in &result {
        out_data.extend_from_slice(&v.to_le_bytes());
    }

    Ok(Mat {
        data: out_data,
        rows: result_h,
        cols: result_w,
        step: result_w * 4,
        mat_type: MatType::CV_32FC1,
    })
}

/// cv2.minMaxLoc — find global minimum and maximum in a `CV_32FC1` Mat.
///
/// Returns `(min_val, max_val, min_loc, max_loc)` where locations are `Point { x, y }`.
///
/// # Errors
/// Returns `UnsupportedDtype` if `src` is not `CV_32FC1`.
pub fn min_max_loc(src: &Mat) -> Cv2Result<(f64, f64, Point, Point)> {
    if src.mat_type != MatType::CV_32FC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }

    if src.data.len() < 4 {
        // Return neutral defaults for empty / too-small Mat
        return Ok((0.0, 0.0, Point::default(), Point::default()));
    }

    let floats: Vec<f32> = src
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let w = src.cols;
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();

    for (i, &fv) in floats.iter().enumerate() {
        let v = fv as f64;
        if v < min_val {
            min_val = v;
            min_loc = Point {
                x: (i % w) as i32,
                y: (i / w) as i32,
            };
        }
        if v > max_val {
            max_val = v;
            max_loc = Point {
                x: (i % w) as i32,
                y: (i / w) as i32,
            };
        }
    }

    Ok((min_val, max_val, min_loc, max_loc))
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn validate_8bit(mat: &Mat) -> Cv2Result<usize> {
    match mat.mat_type {
        MatType::CV_8UC1 => Ok(1),
        MatType::CV_8UC3 => Ok(3),
        MatType::CV_8UC4 => Ok(4),
        _ => Err(Cv2Error::UnsupportedDtype {
            mat_type: mat.mat_type,
        }),
    }
}

fn to_gray_f32(data: &[u8], w: usize, h: usize, ch: usize) -> Vec<f32> {
    let npix = w * h;
    let mut gray = vec![0.0f32; npix];
    if ch == 1 {
        for (i, &v) in data.iter().enumerate().take(npix) {
            gray[i] = v as f32;
        }
    } else {
        for i in 0..npix {
            let off = i * ch;
            // BGR layout: B=off, G=off+1, R=off+2
            gray[i] = 0.114 * data[off] as f32
                + 0.587 * data[off + 1] as f32
                + 0.299 * data[off + 2] as f32;
        }
    }
    gray
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_template_sqdiff_self() {
        // A template matched against itself → SQDIFF = 0 at (0,0)
        let data = vec![100u8; 9];
        let src = Mat::from_gray_bytes(data.clone(), 3, 3);
        let templ = Mat::from_gray_bytes(data, 3, 3);
        let result = match_template(&src, &templ, TM_SQDIFF).unwrap();
        let (_min, _max, min_loc, _max_loc) = min_max_loc(&result).unwrap();
        assert_eq!(min_loc, Point { x: 0, y: 0 });
    }

    #[test]
    fn test_match_template_bad_method() {
        let src = Mat::from_gray_bytes(vec![0u8; 25], 5, 5);
        let templ = Mat::from_gray_bytes(vec![0u8; 9], 3, 3);
        assert!(match_template(&src, &templ, 99).is_err());
    }

    #[test]
    fn test_min_max_loc_dtype_error() {
        let mat = Mat::new_8uc1(5, 5);
        assert!(min_max_loc(&mat).is_err());
    }

    #[test]
    fn test_min_max_loc_known_values() {
        // Build a 3×3 CV_32FC1 Mat with known values
        let values = [0.0f32, 0.5, 0.3, 0.7, 1.0, 0.2, 0.4, 0.6, 0.1];
        let mut data = Vec::with_capacity(9 * 4);
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mat = Mat {
            data,
            rows: 3,
            cols: 3,
            step: 12,
            mat_type: MatType::CV_32FC1,
        };
        let (min_val, max_val, min_loc, max_loc) = min_max_loc(&mat).unwrap();
        assert!((min_val - 0.0).abs() < 1e-5);
        assert!((max_val - 1.0).abs() < 1e-5);
        assert_eq!(min_loc, Point { x: 0, y: 0 });
        assert_eq!(max_loc, Point { x: 1, y: 1 });
    }
}

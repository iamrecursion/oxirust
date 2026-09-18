//! Histogram functions: equalize_hist, calc_hist, normalize_mat.
//!
//! Algorithms lifted from the PyO3 cv2-compat layer (`cv2_compat/arithmetic.rs`)
//! and adapted to use the pure-Rust `Mat` types.

use crate::{
    error::{Cv2Error, Cv2Result},
    mat::{Mat, MatType},
};

// ── Public API ────────────────────────────────────────────────────────────────

/// cv2.equalizeHist — equalize the histogram of a grayscale image.
///
/// Applies CDF-based histogram equalization to a `CV_8UC1` Mat.
///
/// # Errors
/// Returns `UnsupportedDtype` if `src` is not `CV_8UC1`.
pub fn equalize_hist(src: &Mat) -> Cv2Result<Mat> {
    if src.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }

    // Build histogram
    let mut hist = [0u32; 256];
    for &v in &src.data {
        hist[v as usize] += 1;
    }

    // Cumulative distribution function
    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    let total = src.pixel_count() as f32;
    let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0) as f32;
    let denominator = total - cdf_min;

    let lut: Vec<u8> = (0..256usize)
        .map(|v| {
            if denominator < 1e-6 {
                v as u8
            } else {
                ((cdf[v] as f32 - cdf_min) / denominator * 255.0).clamp(0.0, 255.0) as u8
            }
        })
        .collect();

    let out: Vec<u8> = src.data.iter().map(|&v| lut[v as usize]).collect();

    Ok(Mat::from_gray_bytes(out, src.rows, src.cols))
}

/// cv2.calcHist — compute histogram of a single-channel Mat.
///
/// Counts pixel values in `[ranges[0], ranges[1])` with `hist_size` bins.
/// Values outside the range are ignored.
///
/// # Errors
/// Returns `UnsupportedDtype` if `src` is not `CV_8UC1`.
pub fn calc_hist(src: &Mat, hist_size: usize, ranges: [f32; 2]) -> Cv2Result<Vec<f32>> {
    if src.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }

    let range_min = ranges[0] as f64;
    let range_max = ranges[1] as f64;
    let range_span = range_max - range_min;

    if hist_size == 0 {
        return Ok(vec![]);
    }

    let mut bins = vec![0u32; hist_size];

    if range_span > 1e-9 {
        for &v in &src.data {
            let val = v as f64;
            if val >= range_min && val < range_max {
                let bin = ((val - range_min) / range_span * hist_size as f64) as usize;
                let bin = bin.min(hist_size - 1);
                bins[bin] += 1;
            }
        }
    }

    Ok(bins.into_iter().map(|v| v as f32).collect())
}

/// cv2.normalize (MINMAX strategy) — normalize pixel values into `[alpha, beta]`.
///
/// Works on `CV_8UC1` or `CV_8UC3` inputs; the output has the same type.
///
/// # Errors
/// Returns `UnsupportedDtype` for unsupported Mat types.
pub fn normalize_mat(src: &Mat, alpha: f64, beta: f64) -> Cv2Result<Mat> {
    match src.mat_type {
        MatType::CV_8UC1 | MatType::CV_8UC3 | MatType::CV_8UC4 => {}
        _ => {
            return Err(Cv2Error::UnsupportedDtype {
                mat_type: src.mat_type,
            })
        }
    }

    let min_val = src.data.iter().copied().min().unwrap_or(0) as f64;
    let max_val = src.data.iter().copied().max().unwrap_or(255) as f64;
    let range = max_val - min_val;

    let out: Vec<u8> = if range < 1e-6 {
        vec![alpha.clamp(0.0, 255.0) as u8; src.data.len()]
    } else {
        src.data
            .iter()
            .map(|&v| {
                let normalized = (v as f64 - min_val) / range;
                (alpha + normalized * (beta - alpha)).clamp(0.0, 255.0) as u8
            })
            .collect()
    };

    Ok(Mat {
        data: out,
        rows: src.rows,
        cols: src.cols,
        step: src.step,
        mat_type: src.mat_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equalize_hist_uniform() {
        // Uniform image — output should not panic and have same size
        let data = vec![128u8; 100];
        let mat = Mat::from_gray_bytes(data, 10, 10);
        let out = equalize_hist(&mat).unwrap();
        assert_eq!(out.rows, 10);
        assert_eq!(out.cols, 10);
    }

    #[test]
    fn test_equalize_hist_dtype_error() {
        let mat = Mat::new(10, 10, MatType::CV_8UC3);
        assert!(equalize_hist(&mat).is_err());
    }

    #[test]
    fn test_calc_hist_all_zeros() {
        let mat = Mat::new_8uc1(5, 5);
        let hist = calc_hist(&mat, 256, [0.0, 256.0]).unwrap();
        assert_eq!(hist.len(), 256);
        assert_eq!(hist[0], 25.0f32); // all pixels are 0
        for &v in &hist[1..] {
            assert_eq!(v, 0.0f32);
        }
    }

    #[test]
    fn test_normalize_mat_range() {
        let data: Vec<u8> = (0u8..10).collect();
        let mat = Mat::from_gray_bytes(data, 2, 5);
        let out = normalize_mat(&mat, 0.0, 255.0).unwrap();
        assert_eq!(out.data[0], 0);
        assert_eq!(out.data[9], 255);
    }
}

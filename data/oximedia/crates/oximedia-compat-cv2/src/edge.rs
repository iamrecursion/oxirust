//! Edge detection — cv2 compatibility functions.
//!
//! Implements `canny`, `sobel`, `convertScaleAbs`, and `laplacian` dispatching
//! into `oximedia-image` algorithms.

use crate::error::{Cv2Error, Cv2Result};
use crate::mat::{Mat, MatType};
use oximedia_image::canny::{canny as oxi_canny, CannyConfig, GrayF32};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Convert a `Mat` to a `GrayF32` by averaging channels if needed.
///
/// Supports `CV_8UC1` (direct), `CV_8UC3` (BT.601 luma), and `CV_8UC4` (same
/// as 3-channel but ignoring alpha).  Returns an error for float types that
/// are already processed.
fn mat_to_gray_f32(src: &Mat) -> Cv2Result<GrayF32> {
    let w = src.cols;
    let h = src.rows;
    let pixels = w * h;
    let data: Vec<f32> = match src.mat_type {
        MatType::CV_8UC1 => src.data.iter().map(|&b| b as f32 / 255.0).collect(),
        MatType::CV_8UC3 => {
            (0..pixels)
                .map(|i| {
                    let off = i * 3;
                    let b = src.data[off] as f32;
                    let g = src.data[off + 1] as f32;
                    let r = src.data[off + 2] as f32;
                    // BT.601 luma (BGR ordering: b, g, r)
                    (0.114 * b + 0.587 * g + 0.299 * r) / 255.0
                })
                .collect()
        }
        MatType::CV_8UC4 => (0..pixels)
            .map(|i| {
                let off = i * 4;
                let b = src.data[off] as f32;
                let g = src.data[off + 1] as f32;
                let r = src.data[off + 2] as f32;
                (0.114 * b + 0.587 * g + 0.299 * r) / 255.0
            })
            .collect(),
        _ => {
            return Err(Cv2Error::UnsupportedDtype {
                mat_type: src.mat_type,
            })
        }
    };

    GrayF32::from_data(w, h, data).ok_or(Cv2Error::SizeMismatch {
        expected: (h, w),
        actual: (h, w),
    })
}

// ── Sobel kernel application ──────────────────────────────────────────────────

/// Apply a 3×3 kernel stored in row-major order to a single-channel `f32` buffer.
///
/// Returns a new `f32` buffer of the same size. Border pixels are computed with
/// clamped (replicate) sampling.
fn apply_kernel_3x3_f32(data: &[f32], w: usize, h: usize, kernel: &[f64; 9]) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for ky in 0..3usize {
                let sy = (y as isize + ky as isize - 1).clamp(0, h as isize - 1) as usize;
                for kx in 0..3usize {
                    let sx = (x as isize + kx as isize - 1).clamp(0, w as isize - 1) as usize;
                    acc += data[sy * w + sx] as f64 * kernel[ky * 3 + kx];
                }
            }
            out[y * w + x] = acc as f32;
        }
    }
    out
}

/// Pack a `Vec<f32>` as raw LE bytes suitable for `Mat { mat_type: CV_32FC1 }`.
fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Build a `CV_32FC1` `Mat` from a `f32` pixel buffer.
fn mat_from_f32_gray(data: Vec<f32>, rows: usize, cols: usize) -> Mat {
    let step = cols * 4;
    Mat {
        data: f32_vec_to_bytes(data),
        rows,
        cols,
        step,
        mat_type: MatType::CV_32FC1,
    }
}

/// Extract a `Vec<f32>` from a `CV_32FC1` `Mat`.
fn mat_to_f32_gray(src: &Mat) -> Cv2Result<Vec<f32>> {
    if src.mat_type != MatType::CV_32FC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }
    let n = src.rows * src.cols;
    if src.data.len() < n * 4 {
        return Err(Cv2Error::SizeMismatch {
            expected: (src.rows, src.cols),
            actual: (src.data.len() / 4, 1),
        });
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 4;
        let bytes = [
            src.data[base],
            src.data[base + 1],
            src.data[base + 2],
            src.data[base + 3],
        ];
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

/// Require that `src` is a single-channel 8-bit `Mat`, returning a `&[u8]` slice.
fn require_8uc1(src: &Mat) -> Cv2Result<&[u8]> {
    if src.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }
    Ok(src.data.as_slice())
}

// ── Public API ────────────────────────────────────────────────────────────────

/// `cv2.Canny` — Canny edge detection on a grayscale `Mat`.
///
/// `src` must be `CV_8UC1` or a colour image that will be converted to
/// luminance first.  Returns a binary edge map as `CV_8UC1` (0 or 255).
///
/// The `_aperture_size` and `_l2_gradient` parameters are accepted for API
/// compatibility but are not forwarded to the underlying algorithm.
pub fn canny(
    src: &Mat,
    threshold1: f64,
    threshold2: f64,
    _aperture_size: i32,
    _l2_gradient: bool,
) -> Cv2Result<Mat> {
    let gray = mat_to_gray_f32(src)?;
    let config = CannyConfig::new()
        .with_thresholds((threshold1 / 255.0) as f32, (threshold2 / 255.0) as f32)
        .absolute_thresholds();

    let result = oxi_canny(&gray, &config);

    let edge_bytes: Vec<u8> = result
        .edges
        .data
        .iter()
        .map(|&v| if v > 0.5 { 255u8 } else { 0u8 })
        .collect();

    Ok(Mat::from_gray_bytes(edge_bytes, src.rows, src.cols))
}

/// `cv2.Sobel` — first-order gradient in the x or y direction.
///
/// Returns a `CV_32FC1` `Mat`. Pass the result to [`convert_scale_abs`] to
/// obtain a `CV_8UC1` magnitude image.
///
/// `dx` and `dy` select the derivative direction (1 = that axis, 0 = other
/// axis, both 1 not supported — returns an error).  `_ksize` is accepted for
/// API compatibility; only 3×3 is implemented.
pub fn sobel(src: &Mat, dx: i32, dy: i32, _ksize: i32) -> Cv2Result<Mat> {
    let data_u8 = require_8uc1(src)?;
    let w = src.cols;
    let h = src.rows;
    let gray: Vec<f32> = data_u8.iter().map(|&b| b as f32 / 255.0).collect();

    // 3×3 Sobel kernels (row-major)
    // Sobel X:  [[-1,0,1], [-2,0,2], [-1,0,1]]
    let kx: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    // Sobel Y:  [[-1,-2,-1], [0,0,0], [1,2,1]]
    let ky: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    let out = match (dx, dy) {
        (1, 0) => apply_kernel_3x3_f32(&gray, w, h, &kx),
        (0, 1) => apply_kernel_3x3_f32(&gray, w, h, &ky),
        _ => {
            return Err(Cv2Error::UnsupportedFlag {
                name: "sobel dx/dy",
                value: dx * 10 + dy,
            })
        }
    };

    Ok(mat_from_f32_gray(out, h, w))
}

/// `cv2.convertScaleAbs` — converts `CV_32FC1` to `CV_8UC1` by `|x| * scale`
/// clamped to `[0, 255]`.
///
/// The default scale is 1.0 (no amplification).
pub fn convert_scale_abs(src: &Mat) -> Cv2Result<Mat> {
    let f32_data = mat_to_f32_gray(src)?;
    let bytes: Vec<u8> = f32_data
        .iter()
        .map(|&v| (v.abs() * 255.0).clamp(0.0, 255.0) as u8)
        .collect();
    Ok(Mat::from_gray_bytes(bytes, src.rows, src.cols))
}

/// `cv2.Laplacian` — second derivative using a 3×3 Laplacian kernel.
///
/// Returns a `CV_32FC1` `Mat` (can contain negative values).  Pass through
/// [`convert_scale_abs`] to obtain a displayable edge image.
///
/// `_ksize` is accepted for API compatibility; only the 3×3 kernel is used.
pub fn laplacian(src: &Mat, _ksize: i32) -> Cv2Result<Mat> {
    let data_u8 = require_8uc1(src)?;
    let w = src.cols;
    let h = src.rows;
    let gray: Vec<f32> = data_u8.iter().map(|&b| b as f32 / 255.0).collect();

    // 3×3 Laplacian kernel:  [[0,1,0], [1,-4,1], [0,1,0]]
    let kernel: [f64; 9] = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let out = apply_kernel_3x3_f32(&gray, w, h, &kernel);

    Ok(mat_from_f32_gray(out, h, w))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn step_mat(rows: usize, cols: usize) -> Mat {
        let mut data = vec![0u8; rows * cols];
        for r in 0..rows {
            for c in (cols / 2)..cols {
                data[r * cols + c] = 255;
            }
        }
        Mat::from_gray_bytes(data, rows, cols)
    }

    #[test]
    fn test_canny_output_size() {
        let src = step_mat(8, 8);
        let edges = canny(&src, 50.0, 150.0, 3, false).expect("canny");
        assert_eq!(edges.rows, 8);
        assert_eq!(edges.cols, 8);
        assert_eq!(edges.mat_type, MatType::CV_8UC1);
    }

    #[test]
    fn test_canny_edge_values_binary() {
        let src = step_mat(8, 8);
        let edges = canny(&src, 50.0, 150.0, 3, false).expect("canny");
        // All output pixels must be 0 or 255
        for &b in &edges.data {
            assert!(b == 0 || b == 255, "non-binary value {b}");
        }
    }

    #[test]
    fn test_sobel_x_output_type() {
        let src = step_mat(8, 8);
        let grad = sobel(&src, 1, 0, 3).expect("sobel x");
        assert_eq!(grad.mat_type, MatType::CV_32FC1);
        assert_eq!(grad.rows, 8);
        assert_eq!(grad.cols, 8);
    }

    #[test]
    fn test_sobel_y_flat_image_zero() {
        // A horizontally uniform gradient has no vertical component.
        let data = vec![128u8; 36];
        let src = Mat::from_gray_bytes(data, 6, 6);
        let grad_y = sobel(&src, 0, 1, 3).expect("sobel y");
        let floats = mat_to_f32_gray(&grad_y).expect("f32");
        for v in floats {
            assert!(v.abs() < 1e-5, "expected ~0 got {v}");
        }
    }

    #[test]
    fn test_convert_scale_abs_roundtrip() {
        let src = step_mat(6, 6);
        let grad = sobel(&src, 1, 0, 3).expect("sobel");
        let abs_img = convert_scale_abs(&grad).expect("csa");
        assert_eq!(abs_img.mat_type, MatType::CV_8UC1);
        assert_eq!(abs_img.data.len(), 36);
    }

    #[test]
    fn test_laplacian_output_type() {
        let src = step_mat(8, 8);
        let lap = laplacian(&src, 3).expect("laplacian");
        assert_eq!(lap.mat_type, MatType::CV_32FC1);
        assert_eq!(lap.rows, 8);
        assert_eq!(lap.cols, 8);
    }

    #[test]
    fn test_sobel_bad_flags_returns_error() {
        let src = step_mat(4, 4);
        let result = sobel(&src, 1, 1, 3);
        assert!(result.is_err(), "dx=1 dy=1 should return an error");
    }
}

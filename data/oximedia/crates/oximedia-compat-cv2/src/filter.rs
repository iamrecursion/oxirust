//! Filter operations — cv2 compatibility functions.
//!
//! Implements Gaussian blur, box filter, median blur, bilateral filter,
//! arbitrary 2-D convolution, and Gaussian pyramid operations.
//!
//! All functions follow OpenCV's BGR channel-interleaved layout convention.

use crate::error::{Cv2Error, Cv2Result};
use crate::mat::{Mat, MatType};

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Reconstruct a `Mat` from raw u8 pixel data matching the layout of `src`.
///
/// Supports `CV_8UC1`, `CV_8UC3`, and `CV_8UC4`. Returns an error for float types.
fn mat_from_data_like(data: Vec<u8>, src: &Mat) -> Cv2Result<Mat> {
    match src.mat_type {
        MatType::CV_8UC1 => Ok(Mat::from_gray_bytes(data, src.rows, src.cols)),
        MatType::CV_8UC3 => Ok(Mat::from_bgr_bytes(data, src.rows, src.cols)),
        MatType::CV_8UC4 => Ok(Mat {
            step: src.cols * 4,
            data,
            rows: src.rows,
            cols: src.cols,
            mat_type: MatType::CV_8UC4,
        }),
        _ => Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        }),
    }
}

/// Guard that `src` is one of the supported 8-bit integer types.
fn require_8u(src: &Mat) -> Cv2Result<()> {
    match src.mat_type {
        MatType::CV_8UC1 | MatType::CV_8UC3 | MatType::CV_8UC4 => Ok(()),
        _ => Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        }),
    }
}

/// Guard that `src` is `CV_8UC1` or `CV_8UC3` (bilateral / median only support these).
fn require_8u_1or3(src: &Mat) -> Cv2Result<()> {
    match src.mat_type {
        MatType::CV_8UC1 | MatType::CV_8UC3 => Ok(()),
        _ => Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        }),
    }
}

/// Compute OpenCV's default sigma from kernel size:
/// σ = 0.3 × ((ksize − 1) × 0.5 − 1) + 0.8
fn sigma_from_ksize(ksize: usize) -> f64 {
    0.3 * ((ksize as f64 - 1.0) * 0.5 - 1.0) + 0.8
}

// ── Gaussian kernel ───────────────────────────────────────────────────────────

/// Build a normalized 1-D Gaussian kernel of length `ksize` with std-dev `sigma`.
fn gaussian_kernel_1d(ksize: usize, sigma: f64) -> Vec<f64> {
    let half = ksize / 2;
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut k: Vec<f64> = (0..ksize)
        .map(|i| {
            let d = i as f64 - half as f64;
            (-d * d * inv_2s2).exp()
        })
        .collect();
    let sum: f64 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

// ── Separable convolution ─────────────────────────────────────────────────────

/// Apply a 1-D kernel horizontally across an interleaved pixel buffer.
///
/// Works in `f32` to accumulate without rounding per-sample.
fn convolve_horizontal(src: &[f32], w: usize, h: usize, ch: usize, kernel: &[f64]) -> Vec<f32> {
    let half = kernel.len() / 2;
    let mut dst = vec![0.0f32; h * w * ch];
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut acc = 0.0f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sx = (x as isize + ki as isize - half as isize).clamp(0, w as isize - 1)
                        as usize;
                    acc += src[(y * w + sx) * ch + c] as f64 * kv;
                }
                dst[(y * w + x) * ch + c] = acc as f32;
            }
        }
    }
    dst
}

/// Apply a 1-D kernel vertically across an interleaved pixel buffer.
fn convolve_vertical(src: &[f32], w: usize, h: usize, ch: usize, kernel: &[f64]) -> Vec<f32> {
    let half = kernel.len() / 2;
    let mut dst = vec![0.0f32; h * w * ch];
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut acc = 0.0f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sy = (y as isize + ki as isize - half as isize).clamp(0, h as isize - 1)
                        as usize;
                    acc += src[(sy * w + x) * ch + c] as f64 * kv;
                }
                dst[(y * w + x) * ch + c] = acc as f32;
            }
        }
    }
    dst
}

/// Two-pass separable convolution: horizontal then vertical.
///
/// Uses f32 intermediates; result is clamped and rounded to u8.
fn separable_convolve(data: &[u8], w: usize, h: usize, ch: usize, kernel: &[f64]) -> Vec<u8> {
    let f32_src: Vec<f32> = data.iter().map(|&b| b as f32).collect();
    let after_h = convolve_horizontal(&f32_src, w, h, ch, kernel);
    let after_v = convolve_vertical(&after_h, w, h, ch, kernel);
    after_v
        .iter()
        .map(|&v| v.clamp(0.0, 255.0).round() as u8)
        .collect()
}

// ── Bilinear resize ───────────────────────────────────────────────────────────

/// Bilinear resize from `(src_w × src_h)` to `(dst_w × dst_h)`.
///
/// Channel count is `ch`; data is interleaved.
fn bilinear_resize(
    data: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    ch: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; dst_h * dst_w * ch];
    let sx_scale = src_w as f64 / dst_w as f64;
    let sy_scale = src_h as f64 / dst_h as f64;

    for dy in 0..dst_h {
        let sy_f = (dy as f64 + 0.5) * sy_scale - 0.5;
        let sy0 = (sy_f.floor() as isize).clamp(0, src_h as isize - 1) as usize;
        let sy1 = (sy0 + 1).min(src_h - 1);
        let fy = sy_f - sy0 as f64;

        for dx in 0..dst_w {
            let sx_f = (dx as f64 + 0.5) * sx_scale - 0.5;
            let sx0 = (sx_f.floor() as isize).clamp(0, src_w as isize - 1) as usize;
            let sx1 = (sx0 + 1).min(src_w - 1);
            let fx = sx_f - sx0 as f64;

            for c in 0..ch {
                let v00 = data[(sy0 * src_w + sx0) * ch + c] as f64;
                let v10 = data[(sy0 * src_w + sx1) * ch + c] as f64;
                let v01 = data[(sy1 * src_w + sx0) * ch + c] as f64;
                let v11 = data[(sy1 * src_w + sx1) * ch + c] as f64;
                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                out[(dy * dst_w + dx) * ch + c] = v.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
    out
}

// ── Public API ────────────────────────────────────────────────────────────────

/// `cv2.GaussianBlur` — apply a Gaussian blur to `src`.
///
/// `ksize` must be odd and ≥ 1.  If `sigma_y == 0`, `sigma_x` is used for
/// both axes.  If both are 0, sigma is derived from `ksize` using OpenCV's
/// formula: `σ = 0.3 × ((ksize − 1) × 0.5 − 1) + 0.8`.
///
/// Supports `CV_8UC1`, `CV_8UC3`, and `CV_8UC4` (per-channel).
pub fn gaussian_blur(src: &Mat, ksize: i32, sigma_x: f64, sigma_y: f64) -> Cv2Result<Mat> {
    require_8u(src)?;
    if ksize < 1 || ksize % 2 == 0 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "gaussianBlur ksize",
            value: ksize,
        });
    }
    let ksz = ksize as usize;
    let sx = if sigma_x == 0.0 {
        sigma_from_ksize(ksz)
    } else {
        sigma_x
    };
    let sy = if sigma_y == 0.0 { sx } else { sigma_y };

    // Separable kernel: if sx == sy use a single kernel for both passes.
    // Otherwise build two separate kernels.
    let ch = src.channels();
    let w = src.cols;
    let h = src.rows;

    let out = if (sx - sy).abs() < 1e-9 {
        let kernel = gaussian_kernel_1d(ksz, sx);
        separable_convolve(&src.data, w, h, ch, &kernel)
    } else {
        let kernel_x = gaussian_kernel_1d(ksz, sx);
        let kernel_y = gaussian_kernel_1d(ksz, sy);
        let f32_src: Vec<f32> = src.data.iter().map(|&b| b as f32).collect();
        let after_h = convolve_horizontal(&f32_src, w, h, ch, &kernel_x);
        let after_v = convolve_vertical(&after_h, w, h, ch, &kernel_y);
        after_v
            .iter()
            .map(|&v| v.clamp(0.0, 255.0).round() as u8)
            .collect()
    };

    mat_from_data_like(out, src)
}

/// `cv2.blur` — normalized box (averaging) filter.
///
/// `ksize` must be odd and ≥ 1.  Supports `CV_8UC1`, `CV_8UC3`, `CV_8UC4`.
pub fn blur(src: &Mat, ksize: i32) -> Cv2Result<Mat> {
    require_8u(src)?;
    if ksize < 1 || ksize % 2 == 0 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "blur ksize",
            value: ksize,
        });
    }
    let ksz = ksize as usize;
    let weight = 1.0 / (ksz * ksz) as f64;
    let kernel: Vec<f64> = vec![weight.sqrt(); ksz]; // separable uniform kernel
    let ch = src.channels();
    let out = separable_convolve(&src.data, src.cols, src.rows, ch, &kernel);
    mat_from_data_like(out, src)
}

/// `cv2.boxFilter` — box filter (optionally normalized).
///
/// `depth = -1` means same depth as source (the only currently supported value).
/// `normalize = true` divides by `ksize²`; `normalize = false` sums raw values,
/// clamped to `[0, 255]`.
///
/// Supports `CV_8UC1`, `CV_8UC3`, `CV_8UC4`.
pub fn box_filter(src: &Mat, depth: i32, ksize: i32, normalize: bool) -> Cv2Result<Mat> {
    require_8u(src)?;
    if depth != -1 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "boxFilter depth",
            value: depth,
        });
    }
    if ksize < 1 || ksize % 2 == 0 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "boxFilter ksize",
            value: ksize,
        });
    }
    let ksz = ksize as usize;
    let half = ksz / 2;
    let w = src.cols;
    let h = src.rows;
    let ch = src.channels();
    let denom = if normalize { (ksz * ksz) as f64 } else { 1.0 };
    let mut out = vec![0u8; h * w * ch];

    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut acc = 0.0f64;
                for ky in 0..ksz {
                    let sy = (y as isize + ky as isize - half as isize).clamp(0, h as isize - 1)
                        as usize;
                    for kx in 0..ksz {
                        let sx = (x as isize + kx as isize - half as isize).clamp(0, w as isize - 1)
                            as usize;
                        acc += src.data[(sy * w + sx) * ch + c] as f64;
                    }
                }
                out[(y * w + x) * ch + c] = (acc / denom).clamp(0.0, 255.0) as u8;
            }
        }
    }
    mat_from_data_like(out, src)
}

/// `cv2.medianBlur` — median filter.
///
/// `ksize` must be odd and ≥ 3.  Supports `CV_8UC1` and `CV_8UC3`.
/// Border pixels are handled with clamp (replicate) sampling.
pub fn median_blur(src: &Mat, ksize: i32) -> Cv2Result<Mat> {
    require_8u_1or3(src)?;
    if ksize < 3 || ksize % 2 == 0 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "medianBlur ksize",
            value: ksize,
        });
    }
    let ksz = ksize as usize;
    let half = ksz / 2;
    let w = src.cols;
    let h = src.rows;
    let ch = src.channels();
    let neighborhood = ksz * ksz;
    let mut out = vec![0u8; h * w * ch];
    let mut window = vec![0u8; neighborhood];

    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut cnt = 0usize;
                for ky in 0..ksz {
                    let sy = (y as isize + ky as isize - half as isize).clamp(0, h as isize - 1)
                        as usize;
                    for kx in 0..ksz {
                        let sx = (x as isize + kx as isize - half as isize).clamp(0, w as isize - 1)
                            as usize;
                        window[cnt] = src.data[(sy * w + sx) * ch + c];
                        cnt += 1;
                    }
                }
                window[..cnt].sort_unstable();
                out[(y * w + x) * ch + c] = window[cnt / 2];
            }
        }
    }
    mat_from_data_like(out, src)
}

/// `cv2.bilateralFilter` — edge-preserving bilateral filter.
///
/// `d` is the pixel neighbourhood diameter; if `d <= 0` it is computed as
/// `round(sigma_space * 3) * 2 + 1`.  Applies per-channel range weighting.
///
/// Supports `CV_8UC1` and `CV_8UC3`.
pub fn bilateral_filter(src: &Mat, d: i32, sigma_color: f64, sigma_space: f64) -> Cv2Result<Mat> {
    require_8u_1or3(src)?;
    let diameter = if d <= 0 {
        ((sigma_space * 3.0).round() as usize) * 2 + 1
    } else {
        d as usize
    };
    let half = diameter / 2;
    let w = src.cols;
    let h = src.rows;
    let ch = src.channels();

    let inv_2sc2 = 1.0 / (2.0 * sigma_color * sigma_color);
    let inv_2ss2 = 1.0 / (2.0 * sigma_space * sigma_space);

    let mut out = vec![0u8; h * w * ch];

    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let center = src.data[(y * w + x) * ch + c] as f64;
                let mut acc = 0.0f64;
                let mut weight_sum = 0.0f64;
                let y_start = y.saturating_sub(half);
                let y_end = (y + half + 1).min(h);
                let x_start = x.saturating_sub(half);
                let x_end = (x + half + 1).min(w);
                for sy in y_start..y_end {
                    let dy = sy as f64 - y as f64;
                    for sx in x_start..x_end {
                        let dx = sx as f64 - x as f64;
                        let val = src.data[(sy * w + sx) * ch + c] as f64;
                        let diff = val - center;
                        let spatial_w = (-(dx * dx + dy * dy) * inv_2ss2).exp();
                        let range_w = (-diff * diff * inv_2sc2).exp();
                        let w_total = spatial_w * range_w;
                        acc += val * w_total;
                        weight_sum += w_total;
                    }
                }
                let result = if weight_sum > 1e-12 {
                    (acc / weight_sum).clamp(0.0, 255.0)
                } else {
                    center
                };
                out[(y * w + x) * ch + c] = result.round() as u8;
            }
        }
    }
    mat_from_data_like(out, src)
}

/// `cv2.filter2D` — apply an arbitrary 2-D convolution kernel to `src`.
///
/// `kernel` is a flat row-major array of `krows × kcols` `f64` values.
/// `depth = -1` means the same type as `src`.
///
/// Supports `CV_8UC1` and `CV_8UC3` (per-channel).
/// Output is clamped to `[0, 255]`.
pub fn filter_2d(
    src: &Mat,
    depth: i32,
    kernel: &[f64],
    krows: usize,
    kcols: usize,
) -> Cv2Result<Mat> {
    require_8u_1or3(src)?;
    if depth != -1 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "filter2D depth",
            value: depth,
        });
    }
    if kernel.len() != krows * kcols {
        return Err(Cv2Error::UnsupportedFlag {
            name: "filter2D kernel size mismatch",
            value: (krows * kcols) as i32,
        });
    }
    let half_r = krows / 2;
    let half_c = kcols / 2;
    let w = src.cols;
    let h = src.rows;
    let ch = src.channels();
    let mut out = vec![0u8; h * w * ch];

    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut acc = 0.0f64;
                for kr in 0..krows {
                    let sy = (y as isize + kr as isize - half_r as isize).clamp(0, h as isize - 1)
                        as usize;
                    for kc in 0..kcols {
                        let sx = (x as isize + kc as isize - half_c as isize)
                            .clamp(0, w as isize - 1) as usize;
                        acc += src.data[(sy * w + sx) * ch + c] as f64 * kernel[kr * kcols + kc];
                    }
                }
                out[(y * w + x) * ch + c] = acc.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
    mat_from_data_like(out, src)
}

/// `cv2.pyrDown` — reduce image by half using a Gaussian pyramid.
///
/// Blurs with a 5×5 Gaussian (σ ≈ 1.0) then downsamples to `(cols/2, rows/2)`.
/// Returns an empty `Mat` (0×0) if the source image is smaller than 2×2.
pub fn pyramid_down(src: &Mat) -> Cv2Result<Mat> {
    require_8u(src)?;
    if src.rows < 2 || src.cols < 2 {
        return Ok(Mat::new(0, 0, src.mat_type));
    }
    let blurred = gaussian_blur(src, 5, 1.0, 1.0)?;
    let dst_w = src.cols / 2;
    let dst_h = src.rows / 2;
    let ch = src.channels();
    let resized = bilinear_resize(&blurred.data, src.cols, src.rows, dst_w, dst_h, ch);
    mat_from_data_like(resized, &Mat::new(dst_h, dst_w, src.mat_type))
}

/// `cv2.pyrUp` — expand image by 2× using a Gaussian pyramid.
///
/// Upsamples to `(cols*2, rows*2)` with bilinear interpolation then blurs with
/// a 5×5 Gaussian (σ ≈ 1.0).
pub fn pyramid_up(src: &Mat) -> Cv2Result<Mat> {
    require_8u(src)?;
    let dst_w = src.cols * 2;
    let dst_h = src.rows * 2;
    let ch = src.channels();
    let resized = bilinear_resize(&src.data, src.cols, src.rows, dst_w, dst_h, ch);
    let upsampled = mat_from_data_like(resized, &Mat::new(dst_h, dst_w, src.mat_type))?;
    gaussian_blur(&upsampled, 5, 1.0, 1.0)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a uniform grayscale image.
    fn uniform_gray(rows: usize, cols: usize, val: u8) -> Mat {
        Mat::from_gray_bytes(vec![val; rows * cols], rows, cols)
    }

    /// Create a grayscale horizontal gradient (col index as pixel value).
    fn gradient_gray(rows: usize, cols: usize) -> Mat {
        let data: Vec<u8> = (0..rows)
            .flat_map(|_| (0..cols).map(|x| x.min(255) as u8))
            .collect();
        Mat::from_gray_bytes(data, rows, cols)
    }

    // ── gaussian_blur ─────────────────────────────────────────────────────────

    #[test]
    fn test_gaussian_blur_uniform_image_unchanged() {
        // A uniform image should remain uniform after blurring.
        let src = uniform_gray(9, 9, 128);
        let out = gaussian_blur(&src, 3, 1.0, 1.0).expect("gaussian_blur");
        assert_eq!(out.rows, 9);
        assert_eq!(out.cols, 9);
        assert_eq!(out.mat_type, MatType::CV_8UC1);
        for &v in &out.data {
            assert_eq!(v, 128, "uniform image should remain 128 after blur");
        }
    }

    #[test]
    fn test_gaussian_blur_output_shape() {
        let src = Mat::new_8uc3(10, 12);
        let out = gaussian_blur(&src, 5, 0.0, 0.0).expect("gaussian_blur bgr");
        assert_eq!(out.rows, 10);
        assert_eq!(out.cols, 12);
        assert_eq!(out.mat_type, MatType::CV_8UC3);
    }

    #[test]
    fn test_gaussian_blur_even_ksize_error() {
        let src = uniform_gray(5, 5, 100);
        assert!(gaussian_blur(&src, 4, 1.0, 1.0).is_err());
    }

    // ── blur ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_blur_box_filter_gradient() {
        // Apply 3×3 box filter to a horizontal gradient; output should be
        // "smoothed" — values should not exceed source range.
        let src = gradient_gray(7, 7);
        let out = blur(&src, 3).expect("blur");
        assert_eq!(out.rows, 7);
        assert_eq!(out.cols, 7);
        assert_eq!(out.mat_type, MatType::CV_8UC1);
        for &v in &out.data {
            assert!(v <= 6, "output should be within gradient range, got {v}");
        }
    }

    #[test]
    fn test_blur_uniform_unchanged() {
        let src = uniform_gray(6, 6, 200);
        let out = blur(&src, 3).expect("blur uniform");
        for &v in &out.data {
            assert_eq!(v, 200);
        }
    }

    // ── median_blur ───────────────────────────────────────────────────────────

    #[test]
    fn test_median_blur_removes_outlier() {
        // Place a single bright pixel in a dark 7×7 image.
        // 3×3 median should suppress it.
        let mut data = vec![50u8; 7 * 7];
        data[3 * 7 + 3] = 250; // centre pixel outlier
        let src = Mat::from_gray_bytes(data, 7, 7);
        let out = median_blur(&src, 3).expect("median_blur");
        // The centre pixel should be suppressed to ~50
        assert!(
            out.data[3 * 7 + 3] <= 60,
            "outlier should be removed, got {}",
            out.data[3 * 7 + 3]
        );
    }

    #[test]
    fn test_median_blur_bad_ksize_error() {
        let src = uniform_gray(5, 5, 0);
        assert!(median_blur(&src, 2).is_err()); // even
        assert!(median_blur(&src, 1).is_err()); // < 3
    }

    // ── bilateral_filter ─────────────────────────────────────────────────────

    #[test]
    fn test_bilateral_filter_preserves_shape() {
        let src = Mat::new_8uc3(8, 8);
        let out = bilateral_filter(&src, 5, 75.0, 75.0).expect("bilateral");
        assert_eq!(out.rows, 8);
        assert_eq!(out.cols, 8);
        assert_eq!(out.mat_type, MatType::CV_8UC3);
    }

    #[test]
    fn test_bilateral_filter_uniform_unchanged() {
        let src = uniform_gray(7, 7, 120);
        let out = bilateral_filter(&src, 5, 75.0, 75.0).expect("bilateral gray");
        for &v in &out.data {
            // Bilateral on a uniform image should leave all pixels unchanged.
            assert!(
                (v as i32 - 120).abs() <= 1,
                "uniform bilateral deviation too large: {v}"
            );
        }
    }
}

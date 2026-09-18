//! Morphological operations — cv2 compatibility functions.
//!
//! Implements `getStructuringElement`, `erode`, `dilate`, and `morphologyEx`
//! dispatching into the same algorithms as the Python cv2 compat layer.

use crate::error::{Cv2Error, Cv2Result};
use crate::mat::{Mat, MatType};

// ── Kernel helpers ────────────────────────────────────────────────────────────

/// Extract a binary kernel representation from a `Mat`.
///
/// The `Mat` must be `CV_8UC1`; non-zero values are treated as active
/// structuring-element elements.
///
/// Returns `(kernel_bits, kh, kw)`.
fn kernel_from_mat(kernel: &Mat) -> Cv2Result<(Vec<bool>, usize, usize)> {
    if kernel.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: kernel.mat_type,
        });
    }
    let bits: Vec<bool> = kernel.data.iter().map(|&v| v != 0).collect();
    Ok((bits, kernel.rows, kernel.cols))
}

// ── Core erosion / dilation ───────────────────────────────────────────────────

/// Apply morphological erosion (minimum filter over the structuring element).
///
/// Works for any number of channels (stored interleaved).
pub(crate) fn apply_erosion(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    kern: &(Vec<bool>, usize, usize),
) -> Vec<u8> {
    let (ref k, kh, kw) = *kern;
    let half_h = kh / 2;
    let half_w = kw / 2;
    let mut out = vec![0u8; h * w * ch];

    for y in 0..h {
        for x in 0..w {
            let dst_off = (y * w + x) * ch;
            for c in 0..ch {
                let mut min_val = 255u8;
                for ky in 0..kh {
                    let sy_i = y as isize + ky as isize - half_h as isize;
                    let sy = sy_i.clamp(0, h as isize - 1) as usize;
                    for kx in 0..kw {
                        if !k[ky * kw + kx] {
                            continue;
                        }
                        let sx_i = x as isize + kx as isize - half_w as isize;
                        let sx = sx_i.clamp(0, w as isize - 1) as usize;
                        let v = data[(sy * w + sx) * ch + c];
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
                out[dst_off + c] = min_val;
            }
        }
    }
    out
}

/// Apply morphological dilation (maximum filter over the structuring element).
///
/// Works for any number of channels (stored interleaved).
pub(crate) fn apply_dilation(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    kern: &(Vec<bool>, usize, usize),
) -> Vec<u8> {
    let (ref k, kh, kw) = *kern;
    let half_h = kh / 2;
    let half_w = kw / 2;
    let mut out = vec![0u8; h * w * ch];

    for y in 0..h {
        for x in 0..w {
            let dst_off = (y * w + x) * ch;
            for c in 0..ch {
                let mut max_val = 0u8;
                for ky in 0..kh {
                    let sy_i = y as isize + ky as isize - half_h as isize;
                    let sy = sy_i.clamp(0, h as isize - 1) as usize;
                    for kx in 0..kw {
                        if !k[ky * kw + kx] {
                            continue;
                        }
                        let sx_i = x as isize + kx as isize - half_w as isize;
                        let sx = sx_i.clamp(0, w as isize - 1) as usize;
                        let v = data[(sy * w + sx) * ch + c];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                out[dst_off + c] = max_val;
            }
        }
    }
    out
}

// ── helper: reconstruct a Mat from raw u8 data matching src layout ────────────

fn mat_from_data_like(data: Vec<u8>, src: &Mat) -> Mat {
    match src.mat_type {
        MatType::CV_8UC3 => Mat::from_bgr_bytes(data, src.rows, src.cols),
        _ => Mat::from_gray_bytes(data, src.rows, src.cols),
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// `cv2.getStructuringElement` — create a binary morphological kernel.
///
/// Returns a `CV_8UC1` `Mat` of size `ksize × ksize` where non-zero elements
/// are active.
///
/// # Shape values
/// - `MORPH_RECT` (0)     — all elements active
/// - `MORPH_CROSS` (1)    — only the centre row and column active
/// - `MORPH_ELLIPSE` (2)  — elements inside the inscribed ellipse active
pub fn get_structuring_element(shape: i32, ksize: i32) -> Cv2Result<Mat> {
    if ksize <= 0 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "getStructuringElement ksize",
            value: ksize,
        });
    }
    let sz = ksize as usize;
    let cx = sz as f64 / 2.0 - 0.5;
    let cy = sz as f64 / 2.0 - 0.5;
    let rx = (sz as f64 / 2.0).max(1.0);
    let ry = rx;

    let mut data = Vec::with_capacity(sz * sz);
    for y in 0..sz {
        for x in 0..sz {
            let v: u8 = match shape {
                0 => 1, // MORPH_RECT
                1 => {
                    // MORPH_CROSS: centre row or centre column
                    if x == sz / 2 || y == sz / 2 {
                        1
                    } else {
                        0
                    }
                }
                2 => {
                    // MORPH_ELLIPSE: points inside the inscribed ellipse
                    let dx = (x as f64 - cx) / rx;
                    let dy = (y as f64 - cy) / ry;
                    if dx * dx + dy * dy <= 1.0 {
                        1
                    } else {
                        0
                    }
                }
                _ => 1, // unknown shape → full rect
            };
            data.push(v);
        }
    }

    Ok(Mat::from_gray_bytes(data, sz, sz))
}

/// `cv2.erode` — erode `src` using `kernel` for `iterations` passes.
///
/// `src` must be `CV_8UC1` or `CV_8UC3`.  `kernel` must be `CV_8UC1`.
pub fn erode(src: &Mat, kernel: &Mat, iterations: i32) -> Cv2Result<Mat> {
    let ch = src.channels();
    if src.mat_type != MatType::CV_8UC1 && src.mat_type != MatType::CV_8UC3 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }
    let kern = kernel_from_mat(kernel)?;
    let mut data = src.data.clone();
    for _ in 0..iterations.max(1) {
        data = apply_erosion(&data, src.cols, src.rows, ch, &kern);
    }
    Ok(mat_from_data_like(data, src))
}

/// `cv2.dilate` — dilate `src` using `kernel` for `iterations` passes.
///
/// `src` must be `CV_8UC1` or `CV_8UC3`.  `kernel` must be `CV_8UC1`.
pub fn dilate(src: &Mat, kernel: &Mat, iterations: i32) -> Cv2Result<Mat> {
    let ch = src.channels();
    if src.mat_type != MatType::CV_8UC1 && src.mat_type != MatType::CV_8UC3 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }
    let kern = kernel_from_mat(kernel)?;
    let mut data = src.data.clone();
    for _ in 0..iterations.max(1) {
        data = apply_dilation(&data, src.cols, src.rows, ch, &kern);
    }
    Ok(mat_from_data_like(data, src))
}

/// `cv2.morphologyEx` — apply a compound morphological operation.
///
/// `src` must be `CV_8UC1` or `CV_8UC3`.  `kernel` must be `CV_8UC1`.
///
/// # Supported operations
/// - `MORPH_ERODE` (0)    — single erosion pass
/// - `MORPH_DILATE` (1)   — single dilation pass
/// - `MORPH_OPEN` (2)     — erode then dilate
/// - `MORPH_CLOSE` (3)    — dilate then erode
/// - `MORPH_GRADIENT` (4) — dilate − erode (pixel-wise saturating difference)
/// - `MORPH_TOPHAT` (5)   — src − open(src)
/// - `MORPH_BLACKHAT` (6) — close(src) − src
pub fn morphology_ex(src: &Mat, op: i32, kernel: &Mat) -> Cv2Result<Mat> {
    let ch = src.channels();
    if src.mat_type != MatType::CV_8UC1 && src.mat_type != MatType::CV_8UC3 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }
    let kern = kernel_from_mat(kernel)?;
    let data = src.data.as_slice();
    let w = src.cols;
    let h = src.rows;

    let out = match op {
        0 /* MORPH_ERODE */ => {
            apply_erosion(data, w, h, ch, &kern)
        }
        1 /* MORPH_DILATE */ => {
            apply_dilation(data, w, h, ch, &kern)
        }
        2 /* MORPH_OPEN */ => {
            let eroded = apply_erosion(data, w, h, ch, &kern);
            apply_dilation(&eroded, w, h, ch, &kern)
        }
        3 /* MORPH_CLOSE */ => {
            let dilated = apply_dilation(data, w, h, ch, &kern);
            apply_erosion(&dilated, w, h, ch, &kern)
        }
        4 /* MORPH_GRADIENT */ => {
            let eroded = apply_erosion(data, w, h, ch, &kern);
            let dilated = apply_dilation(data, w, h, ch, &kern);
            dilated
                .iter()
                .zip(eroded.iter())
                .map(|(&d, &e)| d.saturating_sub(e))
                .collect()
        }
        5 /* MORPH_TOPHAT */ => {
            let eroded = apply_erosion(data, w, h, ch, &kern);
            let opened = apply_dilation(&eroded, w, h, ch, &kern);
            data.iter()
                .zip(opened.iter())
                .map(|(&s, &o)| s.saturating_sub(o))
                .collect()
        }
        6 /* MORPH_BLACKHAT */ => {
            let dilated = apply_dilation(data, w, h, ch, &kern);
            let closed = apply_erosion(&dilated, w, h, ch, &kern);
            closed
                .iter()
                .zip(data.iter())
                .map(|(&c, &s)| c.saturating_sub(s))
                .collect()
        }
        _ => {
            return Err(Cv2Error::UnsupportedFlag {
                name: "morphologyEx op",
                value: op,
            })
        }
    };

    Ok(mat_from_data_like(out, src))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::*;

    fn single_pixel_7x7(row: usize, col: usize, val: u8) -> Mat {
        let mut data = vec![0u8; 49];
        data[row * 7 + col] = val;
        Mat::from_gray_bytes(data, 7, 7)
    }

    fn all_bright(rows: usize, cols: usize) -> Mat {
        Mat::from_gray_bytes(vec![255u8; rows * cols], rows, cols)
    }

    // ── get_structuring_element ───────────────────────────────────────────────

    #[test]
    fn test_se_rect_all_ones() {
        let k = get_structuring_element(MORPH_RECT, 3).expect("rect");
        assert_eq!(k.rows, 3);
        assert_eq!(k.cols, 3);
        assert!(k.data.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_se_cross_shape() {
        let k = get_structuring_element(MORPH_CROSS, 5).expect("cross");
        // centre row (row 2) and centre col (col 2) should be 1; corners 0
        assert_eq!(k.at_8u1(0, 0), 0);
        assert_eq!(k.at_8u1(2, 0), 1); // centre row
        assert_eq!(k.at_8u1(0, 2), 1); // centre col
        assert_eq!(k.at_8u1(2, 2), 1); // centre
    }

    #[test]
    fn test_se_ellipse_corners_zero() {
        let k = get_structuring_element(MORPH_ELLIPSE, 5).expect("ellipse");
        // Corners of a 5×5 ellipse should be 0
        assert_eq!(k.at_8u1(0, 0), 0);
        assert_eq!(k.at_8u1(0, 4), 0);
        assert_eq!(k.at_8u1(4, 0), 0);
        assert_eq!(k.at_8u1(4, 4), 0);
        // Centre should always be 1
        assert_eq!(k.at_8u1(2, 2), 1);
    }

    #[test]
    fn test_se_bad_ksize() {
        assert!(get_structuring_element(MORPH_RECT, 0).is_err());
        assert!(get_structuring_element(MORPH_RECT, -1).is_err());
    }

    // ── erode ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_erode_removes_isolated_pixel() {
        let src = single_pixel_7x7(3, 3, 255);
        let k = get_structuring_element(MORPH_RECT, 3).expect("rect");
        let out = erode(&src, &k, 1).expect("erode");
        assert_eq!(out.at_8u1(3, 3), 0);
    }

    #[test]
    fn test_erode_all_bright_stays_bright() {
        let src = all_bright(5, 5);
        let k = get_structuring_element(MORPH_RECT, 3).expect("rect");
        let out = erode(&src, &k, 1).expect("erode");
        assert!(out.data.iter().all(|&v| v == 255));
    }

    // ── dilate ────────────────────────────────────────────────────────────────

    #[test]
    fn test_dilate_single_pixel_expands() {
        let src = single_pixel_7x7(3, 3, 255);
        let k = get_structuring_element(MORPH_RECT, 3).expect("rect");
        let out = dilate(&src, &k, 1).expect("dilate");
        // 3×3 neighbourhood of (3,3) should all be 255
        assert_eq!(out.at_8u1(2, 2), 255);
        assert_eq!(out.at_8u1(3, 3), 255);
        assert_eq!(out.at_8u1(4, 4), 255);
        // Corners should still be 0
        assert_eq!(out.at_8u1(0, 0), 0);
        assert_eq!(out.at_8u1(6, 6), 0);
    }

    #[test]
    fn test_dilate_all_dark_stays_dark() {
        let src = Mat::from_gray_bytes(vec![0u8; 25], 5, 5);
        let k = get_structuring_element(MORPH_RECT, 3).expect("rect");
        let out = dilate(&src, &k, 1).expect("dilate");
        assert!(out.data.iter().all(|&v| v == 0));
    }

    // ── morphologyEx ─────────────────────────────────────────────────────────

    #[test]
    fn test_morphex_open_removes_isolated_pixel() {
        let src = single_pixel_7x7(3, 3, 255);
        let k = get_structuring_element(MORPH_RECT, 3).expect("rect");
        let out = morphology_ex(&src, MORPH_OPEN, &k).expect("open");
        assert_eq!(out.at_8u1(3, 3), 0);
    }

    #[test]
    fn test_morphex_gradient_bright_boundary() {
        // A large bright block in a dark field — gradient should be non-zero
        // only at the border of the block.
        let mut data = vec![0u8; 9 * 9];
        for y in 2..7usize {
            for x in 2..7usize {
                data[y * 9 + x] = 255;
            }
        }
        let src = Mat::from_gray_bytes(data, 9, 9);
        let k = get_structuring_element(MORPH_RECT, 3).expect("rect");
        let out = morphology_ex(&src, MORPH_GRADIENT, &k).expect("gradient");
        // Interior centre should be 0 (dilate - erode = 255 - 255)
        assert_eq!(out.at_8u1(4, 4), 0);
        // Dark corner should also be 0
        assert_eq!(out.at_8u1(0, 0), 0);
    }

    #[test]
    fn test_morphex_tophat_flat_zero() {
        let src = all_bright(7, 7);
        let k = get_structuring_element(MORPH_RECT, 3).expect("rect");
        let out = morphology_ex(&src, MORPH_TOPHAT, &k).expect("tophat");
        // src - open(src): on a flat image open is identity → 0 everywhere
        assert!(out.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_morphex_blackhat_flat_zero() {
        let src = all_bright(7, 7);
        let k = get_structuring_element(MORPH_RECT, 3).expect("rect");
        let out = morphology_ex(&src, MORPH_BLACKHAT, &k).expect("blackhat");
        assert!(out.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_morphex_unsupported_op_error() {
        let src = Mat::from_gray_bytes(vec![0u8; 9], 3, 3);
        let k = get_structuring_element(MORPH_RECT, 3).expect("rect");
        assert!(morphology_ex(&src, 99, &k).is_err());
    }

    // ── constants aliasing check ──────────────────────────────────────────────

    #[test]
    fn test_morph_constants_values() {
        assert_eq!(MORPH_ERODE, 0);
        assert_eq!(MORPH_DILATE, 1);
        assert_eq!(MORPH_OPEN, 2);
        assert_eq!(MORPH_CLOSE, 3);
        assert_eq!(MORPH_GRADIENT, 4);
        assert_eq!(MORPH_TOPHAT, 5);
        assert_eq!(MORPH_BLACKHAT, 6);
        assert_eq!(MORPH_RECT, 0);
        assert_eq!(MORPH_CROSS, 1);
        assert_eq!(MORPH_ELLIPSE, 2);
    }
}

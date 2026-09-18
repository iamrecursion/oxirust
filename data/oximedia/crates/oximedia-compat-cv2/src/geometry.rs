//! Geometric transformations: resize, flip, rotate, warpAffine, copyMakeBorder.
//!
//! All functions mirror the OpenCV `cv2` API surface using `Mat` buffers with
//! BGR channel ordering by default.

use crate::constants::border::{BORDER_CONSTANT, BORDER_REPLICATE, BORDER_WRAP};
use crate::constants::interpolation::{INTER_AREA, INTER_CUBIC, INTER_LANCZOS4, INTER_NEAREST};
use crate::constants::rotate::{ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE};
use crate::constants::warp_flags::WARP_INVERSE_MAP;
use crate::error::{Cv2Error, Cv2Result};
use crate::mat::{Mat, MatType, Point2f, Scalar, Size};
use oximedia_scaling::ewa_resample::{EwaFilter, EwaResampler};

// ── Public API ────────────────────────────────────────────────────────────────

/// Resize an image to `dst_size`.
///
/// Mirrors `cv2.resize(src, dsize, interpolation=INTER_LINEAR)`.
/// `dst_size` is `Size { width, height }` — same convention as OpenCV.
pub fn resize(src: &Mat, dst_size: Size, interpolation: i32) -> Cv2Result<Mat> {
    let new_w = dst_size.width;
    let new_h = dst_size.height;
    let w = src.cols;
    let h = src.rows;
    let ch = src.channels();
    let data = src.data.as_slice();

    if new_w == 0 || new_h == 0 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "resize.dst_size",
            value: 0,
        });
    }

    let out = match interpolation {
        i if i == INTER_NEAREST => resize_nearest(data, w, h, ch, new_w, new_h),
        i if i == INTER_CUBIC => resize_bicubic(data, w, h, ch, new_w, new_h),
        i if i == INTER_AREA => {
            if new_w < w || new_h < h {
                resize_area_downscale(data, w, h, ch, new_w, new_h)
            } else {
                resize_bilinear(data, w, h, ch, new_w, new_h)
            }
        }
        i if i == INTER_LANCZOS4 => resize_lanczos4(data, w, h, ch, new_w, new_h),
        _ => {
            // INTER_LINEAR and all others fall through to bilinear
            resize_bilinear(data, w, h, ch, new_w, new_h)
        }
    };

    if ch == 1 {
        Ok(Mat::from_gray_bytes(out, new_h, new_w))
    } else {
        Ok(Mat::from_bgr_bytes(out, new_h, new_w))
    }
}

/// Flip an image around an axis.
///
/// Mirrors `cv2.flip(src, flipCode)`:
/// - `0` → vertical (flip around x-axis)
/// - `1` → horizontal (flip around y-axis)
/// - `-1` → both axes
pub fn flip(src: &Mat, flip_code: i32) -> Cv2Result<Mat> {
    let w = src.cols;
    let h = src.rows;
    let ch = src.channels();
    let data = src.data.as_slice();
    let out = flip_image(data, h, w, ch, flip_code);
    build_mat_from_ch(out, h, w, ch)
}

/// Rotate an image by a multiple of 90 degrees.
///
/// Mirrors `cv2.rotate(src, rotateCode)`:
/// - `ROTATE_90_CLOCKWISE` (0) → 90° clockwise
/// - `ROTATE_180` (1) → 180°
/// - `ROTATE_90_COUNTERCLOCKWISE` (2) → 90° counter-clockwise
pub fn rotate(src: &Mat, rotate_code: i32) -> Cv2Result<Mat> {
    let w = src.cols;
    let h = src.rows;
    let ch = src.channels();
    let data = src.data.as_slice();

    if rotate_code != ROTATE_90_CLOCKWISE
        && rotate_code != ROTATE_180
        && rotate_code != ROTATE_90_COUNTERCLOCKWISE
    {
        return Err(Cv2Error::UnsupportedFlag {
            name: "rotate_code",
            value: rotate_code,
        });
    }

    let (out, out_h, out_w) = rotate_image(data, h, w, ch, rotate_code);
    build_mat_from_ch(out, out_h, out_w, ch)
}

/// Transpose an image (swap rows and cols).
///
/// Mirrors `cv2.transpose(src)`.
pub fn transpose(src: &Mat) -> Cv2Result<Mat> {
    let w = src.cols;
    let h = src.rows;
    let ch = src.channels();
    let data = src.data.as_slice();
    let mut out = vec![0u8; h * w * ch];
    for y in 0..h {
        for x in 0..w {
            let src_off = (y * w + x) * ch;
            let dst_off = (x * h + y) * ch;
            out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
        }
    }
    // After transpose, the output is (w x h) — rows=w, cols=h
    build_mat_from_ch(out, w, h, ch)
}

/// Apply a 2×3 affine transformation matrix.
///
/// Mirrors `cv2.warpAffine(src, M, dsize)`.
/// `m` is `[[a, b, c], [d, e, f]]`.
/// `dst_size` is `Size { width, height }`.
pub fn warp_affine(src: &Mat, m: [[f64; 3]; 2], dst_size: Size) -> Cv2Result<Mat> {
    let w = src.cols;
    let h = src.rows;
    let ch = src.channels();
    let data = src.data.as_slice();
    let out_w = dst_size.width;
    let out_h = dst_size.height;

    let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
    let (d, e, f) = (m[1][0], m[1][1], m[1][2]);

    // Inverse of 2×2 part: [[a,b],[d,e]]^-1 = 1/det * [[e,-b],[-d,a]]
    let det = a * e - b * d;
    let mut out = vec![0u8; out_h * out_w * ch];

    if det.abs() < 1e-10 {
        return build_mat_from_ch(out, out_h, out_w, ch);
    }

    let inv_det = 1.0 / det;
    let ia = e * inv_det;
    let ib = -b * inv_det;
    let id = -d * inv_det;
    let ie = a * inv_det;

    for y in 0..out_h {
        for x in 0..out_w {
            let xf = x as f64;
            let yf = y as f64;
            // Inverse map: src_pt = M^-1 * (dst_pt - t)
            let sx = ia * (xf - c) + ib * (yf - f);
            let sy = id * (xf - c) + ie * (yf - f);

            let out_off = (y * out_w + x) * ch;

            // Bilinear interpolation
            if sx >= 0.0 && sx < (w - 1) as f64 && sy >= 0.0 && sy < (h - 1) as f64 {
                let sx0 = sx.floor() as usize;
                let sy0 = sy.floor() as usize;
                let fx = (sx - sx.floor()) as f32;
                let fy = (sy - sy.floor()) as f32;
                for c_idx in 0..ch {
                    let p00 = data[(sy0 * w + sx0) * ch + c_idx];
                    let p10 = data[(sy0 * w + sx0 + 1) * ch + c_idx];
                    let p01 = data[((sy0 + 1) * w + sx0) * ch + c_idx];
                    let p11 = data[((sy0 + 1) * w + sx0 + 1) * ch + c_idx];
                    let top = lerp_u8(p00, p10, fx);
                    let bot = lerp_u8(p01, p11, fx);
                    out[out_off + c_idx] = lerp_u8(top, bot, fy);
                }
            } else if sx >= 0.0 && sx < w as f64 && sy >= 0.0 && sy < h as f64 {
                // Edge — nearest fallback
                let sx_i = sx as usize;
                let sy_i = sy as usize;
                let src_off = (sy_i * w + sx_i) * ch;
                out[out_off..out_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
            }
        }
    }

    build_mat_from_ch(out, out_h, out_w, ch)
}

/// Compute the 2×3 affine rotation matrix.
///
/// Mirrors `cv2.getRotationMatrix2D(center, angle, scale)`.
/// Returns `[[a, b, tx], [-b, a, ty]]` where `a = cos(angle)*scale`.
#[must_use]
pub fn get_rotation_matrix_2d(center: Point2f, angle: f64, scale: f64) -> [[f64; 3]; 2] {
    let rad = angle * std::f64::consts::PI / 180.0;
    let cos_a = rad.cos() * scale;
    let sin_a = rad.sin() * scale;
    let cx = center.x as f64;
    let cy = center.y as f64;

    // tx = (1 - cos)*cx - sin*cy
    // ty = sin*cx + (1 - cos)*cy
    let tx = (1.0 - cos_a) * cx - sin_a * cy;
    let ty = sin_a * cx + (1.0 - cos_a) * cy;

    [[cos_a, sin_a, tx], [-sin_a, cos_a, ty]]
}

/// Pad the borders of an image.
///
/// Mirrors `cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)`.
pub fn copy_make_border(
    src: &Mat,
    top: i32,
    bottom: i32,
    left: i32,
    right: i32,
    border_type: i32,
    value: Scalar,
) -> Cv2Result<Mat> {
    let top = top.max(0) as usize;
    let bottom = bottom.max(0) as usize;
    let left = left.max(0) as usize;
    let right = right.max(0) as usize;

    let sw = src.cols;
    let sh = src.rows;
    let ch = src.channels();
    let data = src.data.as_slice();

    let dw = sw + left + right;
    let dh = sh + top + bottom;
    let mut out = vec![0u8; dh * dw * ch];

    // Fill constant value first (used for BORDER_CONSTANT)
    if border_type == BORDER_CONSTANT {
        let const_pixel: Vec<u8> = (0..ch)
            .map(|c| {
                let v = match c {
                    0 => value.0,
                    1 => value.1,
                    2 => value.2,
                    _ => value.3,
                };
                v.clamp(0.0, 255.0) as u8
            })
            .collect();
        for row in out.chunks_exact_mut(dw * ch) {
            for px in row.chunks_exact_mut(ch) {
                px.copy_from_slice(&const_pixel);
            }
        }
    }

    // Copy source pixels into the interior
    for y in 0..sh {
        let dst_off = ((y + top) * dw + left) * ch;
        let src_off = (y * sw) * ch;
        out[dst_off..dst_off + sw * ch].copy_from_slice(&data[src_off..src_off + sw * ch]);
    }

    // Fill border regions for non-constant types
    if border_type != BORDER_CONSTANT {
        // Top rows
        for y in 0..top {
            let src_y = border_reflect_idx(y as i64, top as i64, sh, border_type);
            for x in left..left + sw {
                let src_x = x - left;
                let spx = &data[(src_y * sw + src_x) * ch..(src_y * sw + src_x + 1) * ch];
                let dpx_start = (y * dw + x) * ch;
                out[dpx_start..dpx_start + ch].copy_from_slice(spx);
            }
        }
        // Bottom rows
        for y in 0..bottom {
            let dst_y = sh + top + y;
            let src_y = border_reflect_idx(sh as i64 + y as i64, top as i64, sh, border_type);
            for x in left..left + sw {
                let src_x = x - left;
                let spx = &data[(src_y * sw + src_x) * ch..(src_y * sw + src_x + 1) * ch];
                let dpx_start = (dst_y * dw + x) * ch;
                out[dpx_start..dpx_start + ch].copy_from_slice(spx);
            }
        }
        // Left columns (including corners)
        for y in 0..dh {
            for x in 0..left {
                let src_x = border_reflect_idx(x as i64, left as i64, sw, border_type);
                let src_y = if y < top {
                    border_reflect_idx(y as i64, top as i64, sh, border_type)
                } else if y >= top + sh {
                    border_reflect_idx(
                        sh as i64 + (y as i64 - top as i64 - sh as i64),
                        top as i64,
                        sh,
                        border_type,
                    )
                } else {
                    y - top
                };
                let spx = &data[(src_y * sw + src_x) * ch..(src_y * sw + src_x + 1) * ch];
                let dpx_start = (y * dw + x) * ch;
                out[dpx_start..dpx_start + ch].copy_from_slice(spx);
            }
        }
        // Right columns (including corners)
        for y in 0..dh {
            for x in 0..right {
                let dst_x = left + sw + x;
                let src_x = border_reflect_idx(sw as i64 + x as i64, left as i64, sw, border_type);
                let src_y = if y < top {
                    border_reflect_idx(y as i64, top as i64, sh, border_type)
                } else if y >= top + sh {
                    border_reflect_idx(
                        sh as i64 + (y as i64 - top as i64 - sh as i64),
                        top as i64,
                        sh,
                        border_type,
                    )
                } else {
                    y - top
                };
                let spx = &data[(src_y * sw + src_x) * ch..(src_y * sw + src_x + 1) * ch];
                let dpx_start = (y * dw + dst_x) * ch;
                out[dpx_start..dpx_start + ch].copy_from_slice(spx);
            }
        }
    }

    build_mat_from_ch(out, dh, dw, ch)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Build the correct `Mat` variant based on channel count.
fn build_mat_from_ch(data: Vec<u8>, rows: usize, cols: usize, ch: usize) -> Cv2Result<Mat> {
    match ch {
        1 => Ok(Mat::from_gray_bytes(data, rows, cols)),
        3 => Ok(Mat::from_bgr_bytes(data, rows, cols)),
        _ => Err(Cv2Error::Codec(format!(
            "geometry: unsupported channel count {ch}"
        ))),
    }
}

/// Linear interpolation between two `u8` values.
#[inline]
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let av = a as f32;
    let bv = b as f32;
    (av + (bv - av) * t).clamp(0.0, 255.0) as u8
}

/// Nearest-neighbour resize. Lifted verbatim from `cv2_compat/geometry.rs`.
pub(crate) fn resize_nearest(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    new_w: usize,
    new_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; new_h * new_w * ch];
    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = (x * w / new_w).min(w - 1);
            let src_y = (y * h / new_h).min(h - 1);
            let src_off = (src_y * w + src_x) * ch;
            let dst_off = (y * new_w + x) * ch;
            out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
        }
    }
    out
}

/// Bilinear resize with half-pixel alignment. Lifted verbatim from `cv2_compat/geometry.rs`.
pub(crate) fn resize_bilinear(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    new_w: usize,
    new_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; new_h * new_w * ch];
    let x_scale = w as f32 / new_w as f32;
    let y_scale = h as f32 / new_h as f32;

    for y in 0..new_h {
        for x in 0..new_w {
            let sx = (x as f32 + 0.5) * x_scale - 0.5;
            let sy = (y as f32 + 0.5) * y_scale - 0.5;

            let sx0 = (sx.floor() as isize).clamp(0, w as isize - 1) as usize;
            let sy0 = (sy.floor() as isize).clamp(0, h as isize - 1) as usize;
            let sx1 = (sx0 + 1).min(w - 1);
            let sy1 = (sy0 + 1).min(h - 1);

            let fx = (sx - sx.floor()).clamp(0.0, 1.0);
            let fy = (sy - sy.floor()).clamp(0.0, 1.0);

            let dst_off = (y * new_w + x) * ch;
            for c_idx in 0..ch {
                let p00 = data[(sy0 * w + sx0) * ch + c_idx];
                let p10 = data[(sy0 * w + sx1) * ch + c_idx];
                let p01 = data[(sy1 * w + sx0) * ch + c_idx];
                let p11 = data[(sy1 * w + sx1) * ch + c_idx];
                let top = lerp_u8(p00, p10, fx);
                let bot = lerp_u8(p01, p11, fx);
                out[dst_off + c_idx] = lerp_u8(top, bot, fy);
            }
        }
    }
    out
}

/// Bicubic resize using 4-tap cubic (a = -0.5).
fn resize_bicubic(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    new_w: usize,
    new_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; new_h * new_w * ch];
    let x_scale = w as f32 / new_w as f32;
    let y_scale = h as f32 / new_h as f32;

    // Sample a single channel at a possibly-out-of-bounds integer source coordinate.
    let sample = |sx: i64, sy: i64, c: usize| -> f32 {
        let cx = sx.clamp(0, w as i64 - 1) as usize;
        let cy = sy.clamp(0, h as i64 - 1) as usize;
        data[(cy * w + cx) * ch + c] as f32
    };

    // Cubic weight function: Keys (a = -0.5)
    let cubic_weight = |t: f32| -> f32 {
        let t = t.abs();
        if t < 1.0 {
            1.5 * t * t * t - 2.5 * t * t + 1.0
        } else if t < 2.0 {
            -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
        } else {
            0.0
        }
    };

    for y in 0..new_h {
        for x in 0..new_w {
            let sx = (x as f32 + 0.5) * x_scale - 0.5;
            let sy = (y as f32 + 0.5) * y_scale - 0.5;

            let ix = sx.floor() as i64;
            let iy = sy.floor() as i64;
            let fx = sx - ix as f32;
            let fy = sy - iy as f32;

            let dst_off = (y * new_w + x) * ch;
            for c_idx in 0..ch {
                let mut val = 0.0f32;
                for ky in -1i64..=2 {
                    let wy = cubic_weight(ky as f32 - fy);
                    for kx in -1i64..=2 {
                        let wx = cubic_weight(kx as f32 - fx);
                        val += wx * wy * sample(ix + kx, iy + ky, c_idx);
                    }
                }
                out[dst_off + c_idx] = val.clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Area (average-pool) resize for downscaling.
fn resize_area_downscale(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    new_w: usize,
    new_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; new_h * new_w * ch];
    let x_scale = w as f64 / new_w as f64;
    let y_scale = h as f64 / new_h as f64;

    for y in 0..new_h {
        for x in 0..new_w {
            // Source region boundaries
            let x0 = (x as f64 * x_scale) as usize;
            let y0 = (y as f64 * y_scale) as usize;
            let x1 = (((x + 1) as f64 * x_scale) as usize).min(w);
            let y1 = (((y + 1) as f64 * y_scale) as usize).min(h);

            let count = ((x1 - x0) * (y1 - y0)).max(1) as f64;
            let dst_off = (y * new_w + x) * ch;

            for c_idx in 0..ch {
                let mut acc = 0.0f64;
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        acc += data[(sy * w + sx) * ch + c_idx] as f64;
                    }
                }
                out[dst_off + c_idx] = (acc / count).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Lanczos-4 resize via EWA resampler (per-channel float pipeline).
fn resize_lanczos4(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    new_w: usize,
    new_h: usize,
) -> Vec<u8> {
    let resampler = EwaResampler::new(EwaFilter::Lanczos(4), false);
    let mut out = vec![0u8; new_h * new_w * ch];

    for c_idx in 0..ch {
        // Extract single channel as f32
        let src_ch: Vec<f32> = (0..h * w).map(|i| data[i * ch + c_idx] as f32).collect();

        let dst_ch = resampler.resample(&src_ch, w, h, new_w, new_h);

        for i in 0..new_h * new_w {
            out[i * ch + c_idx] = dst_ch.get(i).copied().unwrap_or(0.0).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Flip helper: `flip_code` 0=vertical, 1=horizontal, -1=both.
pub(crate) fn flip_image(data: &[u8], h: usize, w: usize, ch: usize, flip_code: i32) -> Vec<u8> {
    let mut out = vec![0u8; h * w * ch];
    for y in 0..h {
        for x in 0..w {
            let src_y = if flip_code == 0 || flip_code == -1 {
                h - 1 - y
            } else {
                y
            };
            let src_x = if flip_code == 1 || flip_code == -1 {
                w - 1 - x
            } else {
                x
            };
            let src_off = (src_y * w + src_x) * ch;
            let dst_off = (y * w + x) * ch;
            out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
        }
    }
    out
}

/// Rotate helper: `rotate_code` 0=90°CW, 1=180°, 2=90°CCW.
/// Returns `(pixels, out_h, out_w)`.
pub(crate) fn rotate_image(
    data: &[u8],
    h: usize,
    w: usize,
    ch: usize,
    rotate_code: i32,
) -> (Vec<u8>, usize, usize) {
    match rotate_code {
        c if c == ROTATE_90_CLOCKWISE => {
            let mut out = vec![0u8; h * w * ch];
            for y in 0..h {
                for x in 0..w {
                    let src_off = (y * w + x) * ch;
                    // dst: row=x, col=(h-1-y), dst_w=h, dst_h=w
                    let dst_off = (x * h + (h - 1 - y)) * ch;
                    out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                }
            }
            (out, w, h)
        }
        c if c == ROTATE_180 => {
            let mut out = vec![0u8; h * w * ch];
            for y in 0..h {
                for x in 0..w {
                    let src_off = (y * w + x) * ch;
                    let dst_off = ((h - 1 - y) * w + (w - 1 - x)) * ch;
                    out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                }
            }
            (out, h, w)
        }
        _ => {
            // ROTATE_90_COUNTERCLOCKWISE
            let mut out = vec![0u8; h * w * ch];
            for y in 0..h {
                for x in 0..w {
                    let src_off = (y * w + x) * ch;
                    // dst: row=(w-1-x), col=y, dst_w=h, dst_h=w
                    let dst_off = ((w - 1 - x) * h + y) * ch;
                    out[dst_off..dst_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                }
            }
            (out, w, h)
        }
    }
}

/// Map a border index for non-constant border types.
///
/// `idx` is the raw coordinate (may be negative or >= `len`).
/// `offset` is the number of padding pixels on this side.
/// `len` is the source image dimension.
fn border_reflect_idx(idx: i64, _offset: i64, len: usize, border_type: i32) -> usize {
    let n = len as i64;
    if border_type == BORDER_REPLICATE {
        idx.clamp(0, n - 1) as usize
    } else if border_type == BORDER_WRAP {
        ((idx % n + n) % n) as usize
    } else {
        // BORDER_REFLECT (fedcba|abcdefgh|hgfedcb) — reflect at boundary
        let mut i = idx;
        while i < 0 || i >= n {
            if i < 0 {
                i = -i - 1;
            }
            if i >= n {
                i = 2 * n - i - 1;
            }
        }
        i as usize
    }
}

// ── Perspective / affine transform computation ────────────────────────────────

/// Compute a 3×3 homography matrix from four point correspondences.
///
/// Mirrors `cv2.getPerspectiveTransform(src, dst)`.
/// Returns a 3×3 `CV_64FC1` `Mat` (row-major, 9 f64 values = 72 bytes).
///
/// Uses the Direct Linear Transform with the last homogeneous coordinate
/// fixed to 1 (h8 = 1), yielding an 8×8 linear system.
pub fn get_perspective_transform(src_pts: &[Point2f; 4], dst_pts: &[Point2f; 4]) -> Cv2Result<Mat> {
    // Build 8×8 augmented system [A | b].
    // For each correspondence i: src=(xs, ys), dst=(xd, yd)
    //   row 2i:   [ xs, ys, 1,  0,  0, 0, -xd*xs, -xd*ys | xd ]
    //   row 2i+1: [  0,  0, 0, xs, ys, 1, -yd*xs, -yd*ys | yd ]
    let mut a = [[0f64; 9]; 8];
    for i in 0..4 {
        let xs = src_pts[i].x as f64;
        let ys = src_pts[i].y as f64;
        let xd = dst_pts[i].x as f64;
        let yd = dst_pts[i].y as f64;

        let r0 = 2 * i;
        a[r0][0] = xs;
        a[r0][1] = ys;
        a[r0][2] = 1.0;
        a[r0][3] = 0.0;
        a[r0][4] = 0.0;
        a[r0][5] = 0.0;
        a[r0][6] = -xd * xs;
        a[r0][7] = -xd * ys;
        a[r0][8] = xd;

        let r1 = 2 * i + 1;
        a[r1][0] = 0.0;
        a[r1][1] = 0.0;
        a[r1][2] = 0.0;
        a[r1][3] = xs;
        a[r1][4] = ys;
        a[r1][5] = 1.0;
        a[r1][6] = -yd * xs;
        a[r1][7] = -yd * ys;
        a[r1][8] = yd;
    }

    let h = solve_8x8(&mut a).ok_or(Cv2Error::UnsupportedFlag {
        name: "get_perspective_transform: degenerate (collinear) points",
        value: 0,
    })?;

    // Reshape [h0..h7, 1.0] into a 3×3 row-major f64 Mat.
    let flat: [f64; 9] = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0];
    Ok(mat_make_f64(&flat, 3, 3))
}

/// Compute a 2×3 affine transformation matrix from three point correspondences.
///
/// Mirrors `cv2.getAffineTransform(src, dst)`.
/// Returns a 2×3 `CV_64FC1` `Mat` (6 f64 values = 48 bytes, row-major).
///
/// The matrix M satisfies `dst_pts[i] = M · [src_pts[i].x, src_pts[i].y, 1]ᵀ`.
pub fn get_affine_transform(src_pts: &[Point2f; 3], dst_pts: &[Point2f; 3]) -> Cv2Result<Mat> {
    // Build 6×6 augmented system [A | b].
    // For each correspondence i: src=(xs, ys), dst=(xd, yd)
    //   [xs, ys, 1,  0,  0, 0 | xd]
    //   [ 0,  0, 0, xs, ys, 1 | yd]
    let mut a = [[0f64; 7]; 6];
    for i in 0..3 {
        let xs = src_pts[i].x as f64;
        let ys = src_pts[i].y as f64;
        let xd = dst_pts[i].x as f64;
        let yd = dst_pts[i].y as f64;

        let r0 = 2 * i;
        a[r0][0] = xs;
        a[r0][1] = ys;
        a[r0][2] = 1.0;
        a[r0][3] = 0.0;
        a[r0][4] = 0.0;
        a[r0][5] = 0.0;
        a[r0][6] = xd;

        let r1 = 2 * i + 1;
        a[r1][0] = 0.0;
        a[r1][1] = 0.0;
        a[r1][2] = 0.0;
        a[r1][3] = xs;
        a[r1][4] = ys;
        a[r1][5] = 1.0;
        a[r1][6] = yd;
    }

    let coeffs = solve_6x6(&mut a).ok_or(Cv2Error::UnsupportedFlag {
        name: "get_affine_transform: degenerate (collinear) points",
        value: 0,
    })?;

    // Layout: 2×3 matrix [[a, b, c], [d, e, f]]
    let flat: [f64; 6] = coeffs;
    Ok(mat_make_f64(&flat, 2, 3))
}

/// Apply a perspective transformation (homography) to an image.
///
/// Mirrors `cv2.warpPerspective(src, M, dsize, flags, borderMode, borderValue)`.
///
/// - `m` must be a 3×3 `CV_64FC1` homography matrix.
/// - `dsize` is `(cols, rows)` of the destination image.
/// - `flags` combines interpolation mode (`INTER_NEAREST` or `INTER_LINEAR`)
///   with optional `WARP_INVERSE_MAP`.
/// - `border_mode` is `BORDER_CONSTANT` or `BORDER_REPLICATE`.
/// - `border_value` fills pixels outside the source bounds for `BORDER_CONSTANT`.
pub fn warp_perspective(
    src: &Mat,
    m: &Mat,
    dsize: (usize, usize),
    flags: i32,
    border_mode: i32,
    border_value: [u8; 4],
) -> Cv2Result<Mat> {
    if m.rows != 3 || m.cols != 3 || m.mat_type != MatType::CV_64FC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: m.mat_type,
        });
    }

    let (out_w, out_h) = dsize;
    let src_w = src.cols;
    let src_h = src.rows;
    let ch = src.channels();

    // Read 3×3 homography from Mat.
    let raw_h = [
        mat_get_f64(m, 0, 0),
        mat_get_f64(m, 0, 1),
        mat_get_f64(m, 0, 2),
        mat_get_f64(m, 1, 0),
        mat_get_f64(m, 1, 1),
        mat_get_f64(m, 1, 2),
        mat_get_f64(m, 2, 0),
        mat_get_f64(m, 2, 1),
        mat_get_f64(m, 2, 2),
    ];

    // Determine inverse map: if WARP_INVERSE_MAP is set, H is already the inverse.
    let h_inv = if flags & WARP_INVERSE_MAP != 0 {
        raw_h
    } else {
        invert_3x3(&raw_h).ok_or(Cv2Error::UnsupportedFlag {
            name: "warp_perspective: singular homography matrix",
            value: 0,
        })?
    };

    let interp = flags & 0xF;

    // Allocate output with border_value fill.
    let elem = MatType::CV_8UC1; // placeholder; we build raw bytes and use src.mat_type
    let _ = elem;
    let mut out_data = vec![0u8; out_h * out_w * ch];

    // Pre-fill with border_value.
    if border_mode == BORDER_CONSTANT {
        for px in out_data.chunks_exact_mut(ch) {
            for (c, bv) in px.iter_mut().zip(border_value.iter()) {
                *c = *bv;
            }
        }
    }

    let data = src.data.as_slice();

    for y_dst in 0..out_h {
        for x_dst in 0..out_w {
            // Apply inverse homography: H_inv · [x_dst, y_dst, 1]
            let xf = x_dst as f64;
            let yf = y_dst as f64;
            let wx = h_inv[0] * xf + h_inv[1] * yf + h_inv[2];
            let wy = h_inv[3] * xf + h_inv[4] * yf + h_inv[5];
            let w = h_inv[6] * xf + h_inv[7] * yf + h_inv[8];

            if w.abs() < 1e-12 {
                continue;
            }

            let x_src = wx / w;
            let y_src = wy / w;

            let out_off = (y_dst * out_w + x_dst) * ch;

            let in_x_bounds = x_src >= 0.0 && x_src < src_w as f64;
            let in_y_bounds = y_src >= 0.0 && y_src < src_h as f64;

            if in_x_bounds && in_y_bounds {
                if interp == INTER_NEAREST {
                    let sx_i = (x_src as usize).min(src_w - 1);
                    let sy_i = (y_src as usize).min(src_h - 1);
                    let src_off = (sy_i * src_w + sx_i) * ch;
                    out_data[out_off..out_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                } else {
                    // Bilinear interpolation (default).
                    warp_bilinear_sample(
                        data,
                        src_w,
                        src_h,
                        ch,
                        x_src,
                        y_src,
                        &mut out_data[out_off..out_off + ch],
                    );
                }
            } else {
                // Out of source bounds: apply border mode.
                if border_mode == BORDER_REPLICATE {
                    let sx_i = (x_src as i64).clamp(0, src_w as i64 - 1) as usize;
                    let sy_i = (y_src as i64).clamp(0, src_h as i64 - 1) as usize;
                    let src_off = (sy_i * src_w + sx_i) * ch;
                    out_data[out_off..out_off + ch].copy_from_slice(&data[src_off..src_off + ch]);
                }
                // BORDER_CONSTANT: already pre-filled above, nothing to do.
            }
        }
    }

    build_mat_from_ch(out_data, out_h, out_w, ch)
}

/// Apply a generic pixel-remapping transformation.
///
/// Mirrors `cv2.remap(src, map1, map2, ...)` with bilinear interpolation and
/// border clamping (replicate).
///
/// - `map_x` and `map_y` must be `CV_32FC1` Mats of identical dimensions.
///   Each pixel `(x, y)` in the output maps to source coordinate
///   `(map_x[y,x], map_y[y,x])`.
pub fn remap(src: &Mat, map_x: &Mat, map_y: &Mat) -> Cv2Result<Mat> {
    if map_x.rows != map_y.rows || map_x.cols != map_y.cols {
        return Err(Cv2Error::SizeMismatch {
            expected: (map_x.rows, map_x.cols),
            actual: (map_y.rows, map_y.cols),
        });
    }
    if map_x.mat_type != MatType::CV_32FC1 || map_y.mat_type != MatType::CV_32FC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: map_x.mat_type,
        });
    }

    let out_w = map_x.cols;
    let out_h = map_x.rows;
    let src_w = src.cols;
    let src_h = src.rows;
    let ch = src.channels();

    let mut out_data = vec![0u8; out_h * out_w * ch];
    let src_data = src.data.as_slice();

    for y_dst in 0..out_h {
        for x_dst in 0..out_w {
            let idx = y_dst * out_w + x_dst;

            // Read f32 coordinates from map_x and map_y.
            let mx_bytes = map_x
                .data
                .get(idx * 4..(idx + 1) * 4)
                .and_then(|b| b.try_into().ok())
                .map(f32::from_ne_bytes)
                .unwrap_or(0.0f32);
            let my_bytes = map_y
                .data
                .get(idx * 4..(idx + 1) * 4)
                .and_then(|b| b.try_into().ok())
                .map(f32::from_ne_bytes)
                .unwrap_or(0.0f32);

            let x_src = mx_bytes as f64;
            let y_src = my_bytes as f64;

            let out_off = idx * ch;

            // Bilinear with replicate-clamp at borders.
            if x_src >= 0.0 && x_src < src_w as f64 && y_src >= 0.0 && y_src < src_h as f64 {
                warp_bilinear_sample(
                    src_data,
                    src_w,
                    src_h,
                    ch,
                    x_src,
                    y_src,
                    &mut out_data[out_off..out_off + ch],
                );
            } else {
                // Replicate border: clamp coordinates.
                let sx_i = (x_src as i64).clamp(0, src_w as i64 - 1) as usize;
                let sy_i = (y_src as i64).clamp(0, src_h as i64 - 1) as usize;
                let src_off = (sy_i * src_w + sx_i) * ch;
                out_data[out_off..out_off + ch].copy_from_slice(&src_data[src_off..src_off + ch]);
            }
        }
    }

    build_mat_from_ch(out_data, out_h, out_w, ch)
}

// ── Perspective / affine internal helpers ─────────────────────────────────────

/// Gaussian elimination with partial pivoting on an 8×9 augmented matrix.
///
/// `a[row][0..7]` are matrix coefficients; `a[row][8]` is the RHS.
/// Returns the solution vector `[h0..h7]` or `None` if singular.
fn solve_8x8(a: &mut [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8usize {
        // Partial pivoting: find the row with the largest absolute value in this column.
        let pivot = (col..8).max_by(|&r1, &r2| {
            a[r1][col]
                .abs()
                .partial_cmp(&a[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        a.swap(col, pivot);

        let scale = a[col][col];
        if scale.abs() < 1e-12 {
            return None;
        }
        // Normalise pivot row.
        for j in col..9 {
            a[col][j] /= scale;
        }
        // Eliminate column in all other rows.
        for row in 0..8 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for j in col..9 {
                a[row][j] -= factor * a[col][j];
            }
        }
    }
    Some([
        a[0][8], a[1][8], a[2][8], a[3][8], a[4][8], a[5][8], a[6][8], a[7][8],
    ])
}

/// Gaussian elimination with partial pivoting on a 6×7 augmented matrix.
///
/// `a[row][0..5]` are matrix coefficients; `a[row][6]` is the RHS.
/// Returns the solution vector `[m0..m5]` or `None` if singular.
fn solve_6x6(a: &mut [[f64; 7]; 6]) -> Option<[f64; 6]> {
    for col in 0..6usize {
        let pivot = (col..6).max_by(|&r1, &r2| {
            a[r1][col]
                .abs()
                .partial_cmp(&a[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        a.swap(col, pivot);

        let scale = a[col][col];
        if scale.abs() < 1e-12 {
            return None;
        }
        for j in col..7 {
            a[col][j] /= scale;
        }
        for row in 0..6 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for j in col..7 {
                a[row][j] -= factor * a[col][j];
            }
        }
    }
    Some([a[0][6], a[1][6], a[2][6], a[3][6], a[4][6], a[5][6]])
}

/// Invert a 3×3 matrix (row-major, 9 elements) using cofactor expansion.
///
/// Returns `None` if the matrix is singular (|det| < 1e-12).
fn invert_3x3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ])
}

/// Read a single f64 value from a `CV_64FC1` `Mat` at `(row, col)`.
#[inline]
fn mat_get_f64(m: &Mat, row: usize, col: usize) -> f64 {
    let idx = (row * m.cols + col) * 8;
    let bytes: [u8; 8] = m
        .data
        .get(idx..idx + 8)
        .and_then(|b| b.try_into().ok())
        .unwrap_or([0u8; 8]);
    f64::from_ne_bytes(bytes)
}

/// Build a `CV_64FC1` `Mat` from a slice of f64 values (row-major).
fn mat_make_f64(data: &[f64], rows: usize, cols: usize) -> Mat {
    let mut bytes = vec![0u8; rows * cols * 8];
    for (i, &v) in data.iter().enumerate() {
        let start = i * 8;
        bytes[start..start + 8].copy_from_slice(&v.to_ne_bytes());
    }
    Mat {
        data: bytes,
        rows,
        cols,
        step: cols * 8,
        mat_type: MatType::CV_64FC1,
    }
}

/// Bilinear sample from a raw `u8` pixel buffer.
///
/// Clamps to edge at image boundaries. Writes `ch` bytes into `out`.
#[inline]
fn warp_bilinear_sample(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    x: f64,
    y: f64,
    out: &mut [u8],
) {
    let sx0 = (x.floor() as isize).clamp(0, w as isize - 1) as usize;
    let sy0 = (y.floor() as isize).clamp(0, h as isize - 1) as usize;
    let sx1 = (sx0 + 1).min(w - 1);
    let sy1 = (sy0 + 1).min(h - 1);

    let fx = (x - x.floor()) as f32;
    let fy = (y - y.floor()) as f32;

    for c_idx in 0..ch {
        let p00 = data[(sy0 * w + sx0) * ch + c_idx];
        let p10 = data[(sy0 * w + sx1) * ch + c_idx];
        let p01 = data[(sy1 * w + sx0) * ch + c_idx];
        let p11 = data[(sy1 * w + sx1) * ch + c_idx];
        let top = lerp_u8(p00, p10, fx);
        let bot = lerp_u8(p01, p11, fx);
        out[c_idx] = lerp_u8(top, bot, fy);
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::*;

    #[test]
    fn test_resize_nearest_doubles() {
        let mat = Mat::new_8uc3(4, 4);
        let out = resize(
            &mat,
            Size {
                width: 8,
                height: 8,
            },
            INTER_NEAREST,
        )
        .unwrap();
        assert_eq!(out.rows, 8);
        assert_eq!(out.cols, 8);
    }

    #[test]
    fn test_resize_nearest_solid_color_preserved() {
        let data = vec![42u8; 3 * 6 * 6];
        let mat = Mat::from_bgr_bytes(data, 6, 6);
        let out = resize(
            &mat,
            Size {
                width: 3,
                height: 3,
            },
            INTER_NEAREST,
        )
        .unwrap();
        assert!(out.data.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_resize_bilinear_solid_preserved() {
        let data = vec![100u8; 3 * 6 * 6];
        let mat = Mat::from_bgr_bytes(data, 6, 6);
        let out = resize(
            &mat,
            Size {
                width: 12,
                height: 12,
            },
            INTER_LINEAR,
        )
        .unwrap();
        for px in out.data.chunks(3) {
            assert_eq!(px[0], 100);
        }
    }

    #[test]
    fn test_resize_lanczos4_dimensions() {
        let mat = Mat::new_8uc3(8, 8);
        let out = resize(
            &mat,
            Size {
                width: 4,
                height: 4,
            },
            INTER_LANCZOS4,
        )
        .unwrap();
        assert_eq!(out.rows, 4);
        assert_eq!(out.cols, 4);
    }

    #[test]
    fn test_resize_cubic_dimensions() {
        let mat = Mat::new_8uc3(6, 6);
        let out = resize(
            &mat,
            Size {
                width: 3,
                height: 3,
            },
            INTER_CUBIC,
        )
        .unwrap();
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 3);
    }

    #[test]
    fn test_resize_area_downscale() {
        let data = vec![128u8; 3 * 8 * 8];
        let mat = Mat::from_bgr_bytes(data, 8, 8);
        let out = resize(
            &mat,
            Size {
                width: 4,
                height: 4,
            },
            INTER_AREA,
        )
        .unwrap();
        assert_eq!(out.rows, 4);
        assert_eq!(out.cols, 4);
        // Solid color should survive area-pooling
        for px in out.data.chunks(3) {
            assert_eq!(px[0], 128);
        }
    }

    #[test]
    fn test_flip_horizontal() {
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let mat = Mat::from_bgr_bytes(data, 2, 2);
        let out = flip(&mat, 1).unwrap();
        assert_eq!(&out.data[0..3], &[40, 50, 60]);
        assert_eq!(&out.data[3..6], &[10, 20, 30]);
    }

    #[test]
    fn test_rotate_180() {
        let data = vec![1u8, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0];
        let mat = Mat::from_bgr_bytes(data, 2, 2);
        let out = rotate(&mat, ROTATE_180).unwrap();
        assert_eq!(&out.data[0..3], &[4, 0, 0]);
        assert_eq!(&out.data[9..12], &[1, 0, 0]);
    }

    #[test]
    fn test_rotate_invalid_code() {
        let mat = Mat::new_8uc3(4, 4);
        assert!(rotate(&mat, 99).is_err());
    }

    #[test]
    fn test_transpose_dimensions() {
        let mat = Mat::new_8uc3(4, 6);
        let out = transpose(&mat).unwrap();
        assert_eq!(out.rows, 6);
        assert_eq!(out.cols, 4);
    }

    #[test]
    fn test_get_rotation_matrix_2d_identity() {
        let m = get_rotation_matrix_2d(Point2f { x: 0.0, y: 0.0 }, 0.0, 1.0);
        // angle=0, scale=1 → [[1,0,0],[0,1,0]]
        assert!((m[0][0] - 1.0).abs() < 1e-9);
        assert!((m[0][1]).abs() < 1e-9);
        assert!((m[1][0]).abs() < 1e-9);
        assert!((m[1][1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_copy_make_border_constant() {
        let data = vec![100u8, 100, 100];
        let mat = Mat::from_bgr_bytes(data, 1, 1);
        let out = copy_make_border(
            &mat,
            1,
            1,
            1,
            1,
            BORDER_CONSTANT,
            Scalar(0.0, 0.0, 0.0, 0.0),
        )
        .unwrap();
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 3);
        // Center pixel preserved
        let center = out.at_8u3(1, 1);
        assert_eq!(center, [100, 100, 100]);
        // Corner is constant 0
        let corner = out.at_8u3(0, 0);
        assert_eq!(corner, [0, 0, 0]);
    }
}

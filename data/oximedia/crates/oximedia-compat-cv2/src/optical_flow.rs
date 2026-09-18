//! Optical flow functions: Lucas-Kanade sparse pyramid tracking.
//!
//! Provides `calc_optical_flow_pyr_lk` (sparse pyramidal LK).
//! Dense Farneback flow is a stub pending a future pass.
//!
//! Algorithm lifted from the PyO3 cv2-compat layer.

use crate::{
    error::{Cv2Error, Cv2Result},
    mat::{Mat, MatType, Point2f},
};

// ── Public API ────────────────────────────────────────────────────────────────

/// cv2.calcOpticalFlowPyrLK — Sparse optical flow via Lucas-Kanade pyramids.
///
/// Returns `(next_pts, status, err)` where `status[i] = 1` if tracked, `0` if lost.
///
/// # Errors
/// Returns `UnsupportedDtype` if either Mat is not `CV_8UC1` or `CV_8UC3`.
pub fn calc_optical_flow_pyr_lk(
    prev: &Mat,
    next: &Mat,
    prev_pts: &[Point2f],
    win_size: i32,
    max_level: i32,
) -> Cv2Result<(Vec<Point2f>, Vec<u8>, Vec<f32>)> {
    let (prev_data, w, h, prev_ch) = mat_components(prev)?;
    let (next_data, nw, nh, next_ch) = mat_components(next)?;

    if w != nw || h != nh {
        return Err(Cv2Error::SizeMismatch {
            expected: (h, w),
            actual: (nh, nw),
        });
    }
    let _ = next_ch; // both converted to gray

    let prev_gray = to_gray_f32(prev_data, w, h, prev_ch);
    let next_gray = to_gray_f32(next_data, w, h, prev_ch);

    let win = win_size.max(1) as usize;
    let half_w = win / 2;
    let half_h = win / 2;
    let max_lvl = (max_level.max(0) as usize).min(4);

    let prev_pyr = build_pyramid(&prev_gray, w, h, max_lvl);
    let next_pyr = build_pyramid(&next_gray, w, h, max_lvl);

    let min_eig_threshold = 1e-4f32;

    let mut out_pts: Vec<Point2f> = Vec::with_capacity(prev_pts.len());
    let mut status: Vec<u8> = Vec::with_capacity(prev_pts.len());
    let mut err: Vec<f32> = Vec::with_capacity(prev_pts.len());

    for &pt in prev_pts {
        let (px, py_coord) = (pt.x, pt.y);
        let mut disp_x = 0.0f32;
        let mut disp_y = 0.0f32;
        let mut tracked = true;

        for level in (0..=max_lvl).rev() {
            let scale = (1usize << level) as f32;
            let lw = (w >> level).max(1);
            let lh = (h >> level).max(1);
            let lprev = &prev_pyr[level];
            let lnext = &next_pyr[level];

            let cx = px / scale;
            let cy = py_coord / scale;
            if level < max_lvl {
                disp_x /= 2.0;
                disp_y /= 2.0;
            }

            // Iterative Lucas-Kanade (up to 20 iterations)
            for _iter in 0..20 {
                let nx = cx + disp_x;
                let ny = cy + disp_y;

                let mut h11 = 0.0f32;
                let mut h12 = 0.0f32;
                let mut h22 = 0.0f32;
                let mut b1 = 0.0f32;
                let mut b2 = 0.0f32;

                for wy in 0..win {
                    for wx in 0..win {
                        let ix = (cx as isize + wx as isize - half_w as isize)
                            .clamp(1, lw as isize - 2) as usize;
                        let iy = (cy as isize + wy as isize - half_h as isize)
                            .clamp(1, lh as isize - 2) as usize;
                        let jx = (nx + wx as f32 - half_w as f32).clamp(1.0, lw as f32 - 2.0);
                        let jy = (ny + wy as f32 - half_h as f32).clamp(1.0, lh as f32 - 2.0);

                        let it = bilinear(lprev, lw, lh, ix as f32, iy as f32);
                        let jt = bilinear(lnext, lw, lh, jx, jy);
                        let diff = it - jt;

                        let gx = (lprev[iy * lw + (ix + 1).min(lw - 1)]
                            - lprev[iy * lw + ix.saturating_sub(1)])
                            / 2.0;
                        let gy = (lprev[(iy + 1).min(lh - 1) * lw + ix]
                            - lprev[iy.saturating_sub(1) * lw + ix])
                            / 2.0;

                        h11 += gx * gx;
                        h12 += gx * gy;
                        h22 += gy * gy;
                        b1 += gx * diff;
                        b2 += gy * diff;
                    }
                }

                let det = h11 * h22 - h12 * h12;
                if det.abs() < min_eig_threshold {
                    tracked = false;
                    break;
                }

                let inv_det = 1.0 / det;
                let delta_x = (h22 * b1 - h12 * b2) * inv_det;
                let delta_y = (h11 * b2 - h12 * b1) * inv_det;
                disp_x += delta_x;
                disp_y += delta_y;

                if delta_x * delta_x + delta_y * delta_y < 0.001 {
                    break;
                }
            }

            if level > 0 {
                disp_x *= 2.0;
                disp_y *= 2.0;
            }
        }

        let final_x = px + disp_x;
        let final_y = py_coord + disp_y;

        if !tracked || final_x < 0.0 || final_y < 0.0 || final_x >= w as f32 || final_y >= h as f32
        {
            out_pts.push(pt);
            status.push(0);
            err.push(0.0);
        } else {
            let residual = patch_error(
                &prev_gray, &next_gray, w, px, py_coord, final_x, final_y, half_w, half_h,
            );
            out_pts.push(Point2f {
                x: final_x,
                y: final_y,
            });
            status.push(1);
            err.push(residual);
        }
    }

    Ok((out_pts, status, err))
}

/// cv2.calcOpticalFlowFarneback — Dense optical flow via Farneback polynomial expansion.
///
/// Computes a dense displacement field between `prev` and `next` using a
/// pyramid of Gaussian polynomial expansions.
///
/// Returns a `CV_32FC2` `Mat` of the same size as the inputs. Data layout:
/// `[dx0, dy0, dx1, dy1, ...]` row-major, each value stored as an f32.
///
/// # Arguments
/// - `prev` / `next`   : source frames (`CV_8UC1` or `CV_8UC3`)
/// - `pyr_scale`       : scale between pyramid levels (e.g. `0.5`)
/// - `levels`          : number of pyramid levels (clamped to `[1, 6]`)
/// - `winsize`         : spatial averaging window radius
/// - `iterations`      : solver iterations per level
/// - `poly_n`          : polynomial neighbourhood radius (typically 5 or 7)
/// - `poly_sigma`      : Gaussian standard deviation for polynomial expansion
/// - `flags`           : combination of `OPTFLOW_*` flags; `OPTFLOW_USE_INITIAL_FLOW = 4`
///   is **not** supported and returns an error
///
/// # Errors
/// - `UnsupportedFlag`  : `flags & OPTFLOW_USE_INITIAL_FLOW != 0`
/// - `SizeMismatch`     : `prev` and `next` dimensions differ
/// - `UnsupportedDtype` : inputs are not `CV_8UC1` or `CV_8UC3`
pub fn calc_optical_flow_farneback(
    prev: &Mat,
    next: &Mat,
    pyr_scale: f64,
    levels: i32,
    winsize: i32,
    iterations: i32,
    poly_n: i32,
    poly_sigma: f64,
    flags: i32,
) -> Cv2Result<Mat> {
    const OPTFLOW_USE_INITIAL_FLOW: i32 = 4;

    if flags & OPTFLOW_USE_INITIAL_FLOW != 0 {
        return Err(Cv2Error::FeatureNotImplemented {
            name: "calc_optical_flow_farneback with OPTFLOW_USE_INITIAL_FLOW",
            refinement: "future pass",
        });
    }

    let (prev_data, w, h, prev_ch) = mat_components(prev)?;
    let (next_data, nw, nh, next_ch) = mat_components(next)?;

    if w != nw || h != nh {
        return Err(Cv2Error::SizeMismatch {
            expected: (h, w),
            actual: (nh, nw),
        });
    }

    let prev_gray = to_gray_f32(prev_data, w, h, prev_ch);
    let next_gray = to_gray_f32(next_data, nw, nh, next_ch);

    let n_levels = (levels.max(1) as usize).min(6);
    let win = (winsize.max(1) as usize).max(1);
    let n_iter = iterations.max(1) as usize;
    let pn = poly_n.max(1) as usize;

    let scale = pyr_scale.clamp(0.1, 0.9);

    let prev_pyr = build_gaussian_pyramid(&prev_gray, w, h, n_levels, scale);
    let next_pyr = build_gaussian_pyramid(&next_gray, w, h, n_levels, scale);

    // Coarsest level index
    let coarsest = n_levels - 1;
    let (_, cw, ch) = &prev_pyr[coarsest];
    let mut dx = vec![0.0f32; cw * ch];
    let mut dy = vec![0.0f32; cw * ch];

    for level in (0..n_levels).rev() {
        let (lprev, lw, lh) = &prev_pyr[level];
        let (lnext, _, _) = &next_pyr[level];
        let lw = *lw;
        let lh = *lh;

        // Upsample flow from coarser level if we're not at the coarsest level
        if level < coarsest {
            let (_, coarser_w, coarser_h) = &prev_pyr[level + 1];
            let scale_x = lw as f64 / *coarser_w as f64;
            let scale_y = lh as f64 / *coarser_h as f64;
            let (new_dx, new_dy) =
                upsample_flow(&dx, &dy, *coarser_w, *coarser_h, lw, lh, scale_x, scale_y);
            dx = new_dx;
            dy = new_dy;
        } else {
            // First pass — zero at coarsest level (already zero)
            dx = vec![0.0f32; lw * lh];
            dy = vec![0.0f32; lw * lh];
        }

        let use_gaussian_window = (flags & OPTFLOW_FARNEBACK_GAUSSIAN) != 0;

        // Polynomial expansion of the prev frame (computed once per level).
        let polys_prev = polynomial_expansion(lprev, lw, lh, pn, poly_sigma);

        for _iter in 0..n_iter {
            // Warp the next frame by the current accumulated flow, then expand.
            // The resulting b = (r2_prev - r2_nw)/2 is the RESIDUAL displacement
            // still needed, making additive updates: dx[i] += Δdx, correct.
            let lnext_warp = warp_by_flow(lnext, lw, lh, &dx, &dy);
            let polys_nw = polynomial_expansion(&lnext_warp, lw, lh, pn, poly_sigma);

            let mut a00v = vec![0.0f32; lw * lh];
            let mut a01v = vec![0.0f32; lw * lh];
            let mut a11v = vec![0.0f32; lw * lh];
            let mut b0v = vec![0.0f32; lw * lh];
            let mut b1v = vec![0.0f32; lw * lh];

            for i in 0..lw * lh {
                a00v[i] = 0.5 * (polys_prev[3][i] + polys_nw[3][i]);
                a01v[i] = 0.25 * (polys_prev[5][i] + polys_nw[5][i]);
                a11v[i] = 0.5 * (polys_prev[4][i] + polys_nw[4][i]);
                b0v[i] = 0.5 * (polys_prev[1][i] - polys_nw[1][i]);
                b1v[i] = 0.5 * (polys_prev[2][i] - polys_nw[2][i]);
            }

            let (a00s, a01s, a11s, b0s, b1s) = if use_gaussian_window {
                let sigma = win as f64 * 0.5;
                (
                    gaussian_filter_f32(&a00v, lw, lh, win, sigma),
                    gaussian_filter_f32(&a01v, lw, lh, win, sigma),
                    gaussian_filter_f32(&a11v, lw, lh, win, sigma),
                    gaussian_filter_f32(&b0v, lw, lh, win, sigma),
                    gaussian_filter_f32(&b1v, lw, lh, win, sigma),
                )
            } else {
                (
                    box_filter_f32(&a00v, lw, lh, win),
                    box_filter_f32(&a01v, lw, lh, win),
                    box_filter_f32(&a11v, lw, lh, win),
                    box_filter_f32(&b0v, lw, lh, win),
                    box_filter_f32(&b1v, lw, lh, win),
                )
            };

            // Solve 2×2 linear system per pixel, add residual Δd to current estimate
            for i in 0..lw * lh {
                let m00 = a00s[i];
                let m01 = a01s[i];
                let m11 = a11s[i];
                let bx = b0s[i];
                let by = b1s[i];

                let trace = m00 + m11;
                let ridge = (1e-3f32).max(trace * 1e-3);
                let det = m00 * m11 - m01 * m01 + ridge;

                if det.abs() > 1e-9 {
                    let inv = 1.0 / det;
                    dx[i] += (m11 * bx - m01 * by) * inv;
                    dy[i] += (m00 * by - m01 * bx) * inv;
                }
            }
        }
    }

    Ok(make_flow_mat(&dx, &dy, w, h))
}

// ── Farneback private helpers ─────────────────────────────────────────────────

const OPTFLOW_FARNEBACK_GAUSSIAN: i32 = 256;

/// Build a Gaussian pyramid.
///
/// Returns `Vec<(image_data, width, height)>` where index 0 is the finest
/// level (original) and index `levels - 1` is the coarsest.
fn build_gaussian_pyramid(
    img: &[f32],
    w: usize,
    h: usize,
    levels: usize,
    pyr_scale: f64,
) -> Vec<(Vec<f32>, usize, usize)> {
    let mut pyr: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(levels);
    pyr.push((img.to_vec(), w, h));

    let mut cw = w;
    let mut ch = h;
    for _ in 1..levels {
        // Clone to avoid borrowing `pyr` while we push.
        let prev_img = if let Some((img_data, _, _)) = pyr.last() {
            img_data.clone()
        } else {
            break;
        };
        // Smooth before downsampling
        let smoothed = gaussian_blur_5x5(&prev_img, cw, ch);
        let nw = ((cw as f64 * pyr_scale) as usize).max(1);
        let nh = ((ch as f64 * pyr_scale) as usize).max(1);
        let downsampled = bilinear_scale(&smoothed, cw, ch, nw, nh);
        pyr.push((downsampled, nw, nh));
        cw = nw;
        ch = nh;
    }
    pyr
}

/// Polynomial expansion: compute 6 response maps for polynomial fitting
/// of `img` using a Gaussian-weighted window.
///
/// Returns `[r1, r2, r3, r4, r5, r6]` where:
/// - r1 = Gaussian blur (0th moment)
/// - r2 = x-derivative (1st x moment)
/// - r3 = y-derivative (1st y moment)
/// - r4 = x² moment (2nd x)
/// - r5 = y² moment (2nd y)
/// - r6 = xy moment
fn polynomial_expansion(
    img: &[f32],
    w: usize,
    h: usize,
    poly_n: usize,
    poly_sigma: f64,
) -> [Vec<f32>; 6] {
    let klen = 2 * poly_n + 1;

    // Build kernels
    let mut g = vec![0.0f64; klen];
    let inv_2s2 = 1.0 / (2.0 * poly_sigma * poly_sigma);
    for i in 0..klen {
        let d = i as f64 - poly_n as f64;
        g[i] = (-d * d * inv_2s2).exp();
    }
    // Normalize g so it sums to 1
    let g_sum: f64 = g.iter().sum();
    for v in &mut g {
        *v /= g_sum;
    }

    // xg[i] = (i - poly_n) * g[i]  (antisymmetric, NOT normalized)
    let mut xg = vec![0.0f64; klen];
    for i in 0..klen {
        let d = i as f64 - poly_n as f64;
        xg[i] = d * g[i];
    }

    // x2g[i] = (i - poly_n)^2 * g[i] - mu2  where mu2 = sum_i d^2 * g[i]
    // Subtracting mu2 ensures sum(x2g) = 0 for numerical stability
    let mu2: f64 = (0..klen)
        .map(|i| {
            let d = i as f64 - poly_n as f64;
            d * d * g[i]
        })
        .sum();
    let mut x2g = vec![0.0f64; klen];
    for i in 0..klen {
        let d = i as f64 - poly_n as f64;
        x2g[i] = d * d * g[i] - mu2 * g[i];
    }

    let g_f32: Vec<f32> = g.iter().map(|&v| v as f32).collect();
    let xg_f32: Vec<f32> = xg.iter().map(|&v| v as f32).collect();
    let x2g_f32: Vec<f32> = x2g.iter().map(|&v| v as f32).collect();

    // r1 = conv(img, g_x) conv'd with g_y
    let r1 = conv2d_sep(img, w, h, &g_f32, &g_f32);
    // r2 = conv(img, xg_x) conv'd with g_y   — x derivative
    let r2 = conv2d_sep(img, w, h, &xg_f32, &g_f32);
    // r3 = conv(img, g_x) conv'd with xg_y   — y derivative
    let r3 = conv2d_sep(img, w, h, &g_f32, &xg_f32);
    // r4 = conv(img, x2g_x) conv'd with g_y  — x² moment
    let r4 = conv2d_sep(img, w, h, &x2g_f32, &g_f32);
    // r5 = conv(img, g_x) conv'd with x2g_y  — y² moment
    let r5 = conv2d_sep(img, w, h, &g_f32, &x2g_f32);
    // r6 = conv(img, xg_x) conv'd with xg_y  — xy moment
    let r6 = conv2d_sep(img, w, h, &xg_f32, &xg_f32);

    [r1, r2, r3, r4, r5, r6]
}

/// Separable 2D convolution (horizontal pass then vertical pass).
/// Kernel is a 1D array; border handling uses clamp-to-edge.
fn conv2d_sep(img: &[f32], w: usize, h: usize, kx: &[f32], ky: &[f32]) -> Vec<f32> {
    let half_x = kx.len() / 2;
    let half_y = ky.len() / 2;

    // Horizontal pass: convolve each row with kx
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kx.iter().enumerate() {
                let sx =
                    (x as isize + ki as isize - half_x as isize).clamp(0, w as isize - 1) as usize;
                acc += img[y * w + sx] * kv;
            }
            tmp[y * w + x] = acc;
        }
    }

    // Vertical pass: convolve each column with ky
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in ky.iter().enumerate() {
                let sy =
                    (y as isize + ki as isize - half_y as isize).clamp(0, h as isize - 1) as usize;
                acc += tmp[sy * w + x] * kv;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Box filter (un-normalized sum over winsize×winsize neighbourhood).
fn box_filter_f32(src: &[f32], w: usize, h: usize, winsize: usize) -> Vec<f32> {
    let half = winsize / 2;
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            let y0 = (y as isize - half as isize).max(0) as usize;
            let y1 = (y + half).min(h - 1);
            let x0 = (x as isize - half as isize).max(0) as usize;
            let x1 = (x + half).min(w - 1);
            for sy in y0..=y1 {
                for sx in x0..=x1 {
                    acc += src[sy * w + sx];
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Gaussian-weighted spatial filter of size winsize×winsize.
fn gaussian_filter_f32(src: &[f32], w: usize, h: usize, winsize: usize, sigma: f64) -> Vec<f32> {
    let half = winsize / 2;
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    // Build 1D kernel (un-normalized)
    let k: Vec<f64> = (0..=half * 2)
        .map(|i| {
            let d = i as f64 - half as f64;
            (-d * d * inv_2s2).exp()
        })
        .collect();

    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            let mut weight_sum = 0.0f64;
            let y0 = (y as isize - half as isize).max(0) as usize;
            let y1 = (y + half).min(h - 1);
            let x0 = (x as isize - half as isize).max(0) as usize;
            let x1 = (x + half).min(w - 1);
            for sy in y0..=y1 {
                let ky = k[sy - (y as isize - half as isize).max(0) as usize];
                for sx in x0..=x1 {
                    let kx = k[sx - (x as isize - half as isize).max(0) as usize];
                    let w2d = ky * kx;
                    acc += src[sy * w + sx] as f64 * w2d;
                    weight_sum += w2d;
                }
            }
            out[y * w + x] = if weight_sum > 1e-12 {
                (acc / weight_sum) as f32
            } else {
                0.0
            };
        }
    }
    out
}

/// Warp `img` by the given per-pixel (dx, dy) displacement using bilinear interpolation.
fn warp_by_flow(img: &[f32], w: usize, h: usize, dx: &[f32], dy: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let sx = x as f32 + dx[i];
            let sy = y as f32 + dy[i];
            out[i] = bilinear(img, w, h, sx, sy);
        }
    }
    out
}

/// Upsample a flow field from (src_w × src_h) to (dst_w × dst_h).
/// Flow magnitudes are scaled by (dst / src) ratio.
fn upsample_flow(
    dx: &[f32],
    dy: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    scale_x: f64,
    scale_y: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut ndx = vec![0.0f32; dst_w * dst_h];
    let mut ndy = vec![0.0f32; dst_w * dst_h];
    let sx_ratio = src_w as f64 / dst_w as f64;
    let sy_ratio = src_h as f64 / dst_h as f64;
    for y in 0..dst_h {
        for x in 0..dst_w {
            let sx = x as f64 * sx_ratio;
            let sy = y as f64 * sy_ratio;
            let vdx = bilinear(dx, src_w, src_h, sx as f32, sy as f32);
            let vdy = bilinear(dy, src_w, src_h, sx as f32, sy as f32);
            let i = y * dst_w + x;
            ndx[i] = vdx * scale_x as f32;
            ndy[i] = vdy * scale_y as f32;
        }
    }
    (ndx, ndy)
}

/// 5×5 Gaussian blur (σ ≈ 1.5) for pyramid pre-smoothing.
fn gaussian_blur_5x5(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    // σ=1.5 kernel weights (from Pascal's triangle approximation)
    let k: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    // Horizontal pass
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in k.iter().enumerate() {
                let sx = (x as isize + ki as isize - 2).clamp(0, w as isize - 1) as usize;
                acc += img[y * w + sx] * kv;
            }
            tmp[y * w + x] = acc;
        }
    }
    // Vertical pass
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in k.iter().enumerate() {
                let sy = (y as isize + ki as isize - 2).clamp(0, h as isize - 1) as usize;
                acc += tmp[sy * w + x] * kv;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Bilinear scale: resize `src` from (src_w × src_h) to (dst_w × dst_h).
fn bilinear_scale(src: &[f32], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dst_w * dst_h];
    let sx_ratio = src_w as f32 / dst_w as f32;
    let sy_ratio = src_h as f32 / dst_h as f32;
    for y in 0..dst_h {
        for x in 0..dst_w {
            let sx = (x as f32 + 0.5) * sx_ratio - 0.5;
            let sy = (y as f32 + 0.5) * sy_ratio - 0.5;
            out[y * dst_w + x] = bilinear(src, src_w, src_h, sx, sy);
        }
    }
    out
}

/// Build a `CV_32FC2` `Mat` from separate dx and dy displacement arrays.
/// Layout: `[dx0, dy0, dx1, dy1, ...]` row-major.
fn make_flow_mat(dx: &[f32], dy: &[f32], cols: usize, rows: usize) -> Mat {
    let n = rows * cols;
    let mut data = vec![0u8; n * 8];
    for i in 0..n {
        let dx_bytes = dx[i].to_ne_bytes();
        let dy_bytes = dy[i].to_ne_bytes();
        data[i * 8..i * 8 + 4].copy_from_slice(&dx_bytes);
        data[i * 8 + 4..i * 8 + 8].copy_from_slice(&dy_bytes);
    }
    Mat {
        data,
        rows,
        cols,
        step: cols * 8,
        mat_type: MatType::CV_32FC2,
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

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
            gray[i] = 0.114 * data[off] as f32
                + 0.587 * data[off + 1] as f32
                + 0.299 * data[off + 2] as f32;
        }
    }
    gray
}

fn build_pyramid(img: &[f32], w: usize, h: usize, levels: usize) -> Vec<Vec<f32>> {
    let mut pyr = vec![img.to_vec()];
    let mut cw = w;
    let mut ch = h;
    for _ in 0..levels {
        let nw = (cw / 2).max(1);
        let nh = (ch / 2).max(1);
        let prev = pyr.last().expect("pyramid is initialised with one element");
        let mut down = vec![0.0f32; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                let sy = (y * 2).min(ch - 1);
                let sx = (x * 2).min(cw - 1);
                down[y * nw + x] = prev[sy * cw + sx];
            }
        }
        pyr.push(down);
        cw = nw;
        ch = nh;
    }
    pyr
}

fn bilinear(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let xc = x.clamp(0.0, (w as f32 - 1.0).max(0.0));
    let yc = y.clamp(0.0, (h as f32 - 1.0).max(0.0));
    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let fx = xc - x0 as f32;
    let fy = yc - y0 as f32;
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));
    let p00 = img[y0 * w + x0];
    let p10 = img[y0 * w + x1];
    let p01 = img[y1 * w + x0];
    let p11 = img[y1 * w + x1];
    let top = p00 + (p10 - p00) * fx;
    let bot = p01 + (p11 - p01) * fx;
    top + (bot - top) * fy
}

#[allow(clippy::too_many_arguments)]
fn patch_error(
    prev: &[f32],
    next: &[f32],
    w: usize,
    px: f32,
    py: f32,
    nx: f32,
    ny: f32,
    half_w: usize,
    half_h: usize,
) -> f32 {
    let h_approx = prev.len() / w.max(1);
    let wf = w as f32;
    let hf = h_approx as f32;
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for wy in 0..half_h * 2 + 1 {
        for wx in 0..half_w * 2 + 1 {
            let ix = (px + wx as f32 - half_w as f32).clamp(0.0, wf - 1.0);
            let iy = (py + wy as f32 - half_h as f32).clamp(0.0, hf - 1.0);
            let jx = (nx + wx as f32 - half_w as f32).clamp(0.0, wf - 1.0);
            let jy = (ny + wy as f32 - half_h as f32).clamp(0.0, hf - 1.0);
            let diff = bilinear(prev, w, h_approx, ix, iy) - bilinear(next, w, h_approx, jx, jy);
            sum += diff * diff;
            count += 1;
        }
    }
    if count > 0 {
        (sum / count as f32).sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lk_empty_points() {
        let prev = Mat::new_8uc1(20, 20);
        let next = Mat::new_8uc1(20, 20);
        let (pts, status, errs) = calc_optical_flow_pyr_lk(&prev, &next, &[], 21, 3).unwrap();
        assert!(pts.is_empty());
        assert!(status.is_empty());
        assert!(errs.is_empty());
    }

    #[test]
    fn test_lk_size_mismatch_error() {
        let prev = Mat::new_8uc1(20, 20);
        let next = Mat::new_8uc1(30, 30);
        let result = calc_optical_flow_pyr_lk(&prev, &next, &[], 21, 3);
        assert!(result.is_err());
    }
}

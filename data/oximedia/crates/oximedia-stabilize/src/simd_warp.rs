//! SIMD-accelerated affine warp coordinate transformation.
//!
//! The hottest loop in video stabilization is the per-pixel inverse affine
//! transform that maps destination coordinates `(x, y)` back to source
//! coordinates.  This module batch-computes the inverse transform for a whole
//! row of output pixels using scirs2-core SIMD operations (AVX2/SSE/NEON
//! auto-dispatched by the library).
//!
//! The affine model is:
//! ```text
//! src_x = (cos(a) * (x - dx) + sin(a) * (y - dy)) / scale
//! src_y = (-sin(a) * (x - dx) + cos(a) * (y - dy)) / scale
//! ```

use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::simd::arithmetic::simd_scalar_mul_f64;
use scirs2_core::simd::basic::simd_add_f64;
use scirs2_core::simd::dot::simd_sub_f64;
use scirs2_core::simd::rounding::simd_floor_f64;

// ── Batch inverse affine transform ──────────────────────────────────────────

/// Parameters for the inverse affine transform.
#[derive(Debug, Clone, Copy)]
pub struct AffineParams {
    /// Translation X.
    pub dx: f64,
    /// Translation Y.
    pub dy: f64,
    /// cos(angle).
    pub cos_a: f64,
    /// sin(angle).
    pub sin_a: f64,
    /// Inverse scale (1/scale).
    pub inv_scale: f64,
}

impl AffineParams {
    /// Create affine params from translation, angle, and scale.
    #[must_use]
    pub fn new(dx: f64, dy: f64, angle: f64, scale: f64) -> Self {
        let safe_scale = if scale.abs() < 1e-10 { 1.0 } else { scale };
        Self {
            dx,
            dy,
            cos_a: angle.cos(),
            sin_a: angle.sin(),
            inv_scale: 1.0 / safe_scale,
        }
    }
}

/// Batch-compute inverse affine transforms for a row of output pixels.
///
/// For each x in `0..width` at the given `y`, computes the source coordinates
/// `(src_x, src_y)` and writes them into `src_x_out` and `src_y_out`.
///
/// Both output slices must be at least `width` elements long.
///
/// # Panics
///
/// Panics if output slices are shorter than `width`.
pub fn inverse_affine_row(
    params: &AffineParams,
    y: f64,
    width: usize,
    src_x_out: &mut [f64],
    src_y_out: &mut [f64],
) {
    assert!(src_x_out.len() >= width && src_y_out.len() >= width);
    if width == 0 {
        return;
    }

    // Build the x-coordinate array: [0.0, 1.0, 2.0, ..., (width-1)]
    let x_coords = Array1::from_iter((0..width).map(|i| i as f64));
    let x_view = x_coords.view();

    // cx = x - dx  (broadcast dx across all elements)
    let dx_arr = Array1::from_elem(width, params.dx);
    let cx = simd_sub_f64(&x_view, &dx_arr.view());
    let cx_view = cx.view();

    // cy = y - dy (scalar, broadcast to array for SIMD element-wise ops)
    let cy_val = y - params.dy;
    let cy_arr = Array1::from_elem(width, cy_val);
    let cy_view = cy_arr.view();

    // cos_a * cx
    let cos_cx = simd_scalar_mul_f64(&cx_view, params.cos_a);
    // sin_a * cy
    let sin_cy = simd_scalar_mul_f64(&cy_view, params.sin_a);
    // (cos_a * cx + sin_a * cy)
    let sum_x = simd_add_f64(&cos_cx.view(), &sin_cy.view());
    // src_x = (cos_a * cx + sin_a * cy) * inv_scale
    let result_x = simd_scalar_mul_f64(&sum_x.view(), params.inv_scale);

    // -sin_a * cx
    let neg_sin_cx = simd_scalar_mul_f64(&cx_view, -params.sin_a);
    // cos_a * cy
    let cos_cy = simd_scalar_mul_f64(&cy_view, params.cos_a);
    // (-sin_a * cx + cos_a * cy)
    let sum_y = simd_add_f64(&neg_sin_cx.view(), &cos_cy.view());
    // src_y = (-sin_a * cx + cos_a * cy) * inv_scale
    let result_y = simd_scalar_mul_f64(&sum_y.view(), params.inv_scale);

    // Copy results to output slices
    if let Some(rx) = result_x.as_slice() {
        src_x_out[..width].copy_from_slice(rx);
    } else {
        for (i, v) in result_x.iter().enumerate() {
            src_x_out[i] = *v;
        }
    }
    if let Some(ry) = result_y.as_slice() {
        src_y_out[..width].copy_from_slice(ry);
    } else {
        for (i, v) in result_y.iter().enumerate() {
            src_y_out[i] = *v;
        }
    }
}

// ── Batch bilinear sample ────────────────────────────────────────────────────

/// Batch bilinear interpolation for a row of pre-computed source coordinates.
///
/// Given source coordinates `(src_x[i], src_y[i])`, samples from `data`
/// (row-major, `width x height` single-channel u8) using bilinear interpolation
/// and writes the results to `output`.
///
/// All slices must be at least `count` elements long.
#[must_use]
pub fn batch_bilinear_sample(
    data: &[u8],
    data_width: usize,
    data_height: usize,
    src_x: &[f64],
    src_y: &[f64],
    count: usize,
) -> Vec<u8> {
    if count == 0 {
        return Vec::new();
    }

    let x_arr = ArrayView1::from(&src_x[..count]);
    let y_arr = ArrayView1::from(&src_y[..count]);

    // Floor coordinates via SIMD
    let x0_arr = simd_floor_f64(&x_arr);
    let y0_arr = simd_floor_f64(&y_arr);

    // Fractional parts: fx = x - floor(x), fy = y - floor(y)
    let fx_arr = simd_sub_f64(&x_arr, &x0_arr.view());
    let fy_arr = simd_sub_f64(&y_arr, &y0_arr.view());

    // Complementary weights: (1 - fx), (1 - fy)
    let ones = Array1::from_elem(count, 1.0_f64);
    let inv_fx = simd_sub_f64(&ones.view(), &fx_arr.view());
    let inv_fy = simd_sub_f64(&ones.view(), &fy_arr.view());

    // Get slices for the per-pixel loop (sampling requires index lookups
    // into the source image, which is inherently scalar).
    let x0_sl = x0_arr.as_slice().unwrap_or(&[]);
    let y0_sl = y0_arr.as_slice().unwrap_or(&[]);
    let fx_sl = fx_arr.as_slice().unwrap_or(&[]);
    let fy_sl = fy_arr.as_slice().unwrap_or(&[]);
    let inv_fx_sl = inv_fx.as_slice().unwrap_or(&[]);
    let inv_fy_sl = inv_fy.as_slice().unwrap_or(&[]);

    // If any slice failed to materialise (non-contiguous), fall back to per-element
    let have_slices = !x0_sl.is_empty()
        && !y0_sl.is_empty()
        && !fx_sl.is_empty()
        && !fy_sl.is_empty()
        && !inv_fx_sl.is_empty()
        && !inv_fy_sl.is_empty();

    let mut output = vec![0u8; count];

    if have_slices {
        let max_x = (data_width as isize) - 1;
        let max_y = (data_height as isize) - 1;

        for i in 0..count {
            let ix0 = (x0_sl[i] as isize).clamp(0, max_x) as usize;
            let ix1 = (ix0 + 1).min(data_width - 1);
            let iy0 = (y0_sl[i] as isize).clamp(0, max_y) as usize;
            let iy1 = (iy0 + 1).min(data_height - 1);

            let fxv = fx_sl[i].clamp(0.0, 1.0);
            let fyv = fy_sl[i].clamp(0.0, 1.0);
            let ifx = inv_fx_sl[i].clamp(0.0, 1.0);
            let ify = inv_fy_sl[i].clamp(0.0, 1.0);

            let p00 = data[iy0 * data_width + ix0] as f64;
            let p10 = data[iy0 * data_width + ix1] as f64;
            let p01 = data[iy1 * data_width + ix0] as f64;
            let p11 = data[iy1 * data_width + ix1] as f64;

            let top = p00 * ifx + p10 * fxv;
            let bot = p01 * ifx + p11 * fxv;
            let val = top * ify + bot * fyv;

            output[i] = val.round().clamp(0.0, 255.0) as u8;
        }
    } else {
        // Fallback: element-by-element access from ndarray
        for i in 0..count {
            let x0v = x0_arr[i];
            let y0v = y0_arr[i];
            let fxv = fx_arr[i].clamp(0.0, 1.0);
            let fyv = fy_arr[i].clamp(0.0, 1.0);
            let ifx = (1.0_f64 - fxv).clamp(0.0_f64, 1.0_f64);
            let ify = (1.0_f64 - fyv).clamp(0.0_f64, 1.0_f64);

            let ix0 = (x0v as isize).clamp(0, (data_width as isize) - 1) as usize;
            let ix1 = (ix0 + 1).min(data_width - 1);
            let iy0 = (y0v as isize).clamp(0, (data_height as isize) - 1) as usize;
            let iy1 = (iy0 + 1).min(data_height - 1);

            let p00 = data[iy0 * data_width + ix0] as f64;
            let p10 = data[iy0 * data_width + ix1] as f64;
            let p01 = data[iy1 * data_width + ix0] as f64;
            let p11 = data[iy1 * data_width + ix1] as f64;

            let top = p00 * ifx + p10 * fxv;
            let bot = p01 * ifx + p11 * fxv;
            let val = top * ify + bot * fyv;

            output[i] = val.round().clamp(0.0, 255.0) as u8;
        }
    }

    output
}

/// Warp a single-channel u8 frame using an affine transform, with SIMD-accelerated
/// coordinate computation and bilinear interpolation.
///
/// Returns a new buffer of `width * height` bytes.
#[must_use]
pub fn warp_frame_simd(src: &[u8], width: usize, height: usize, params: &AffineParams) -> Vec<u8> {
    if width == 0 || height == 0 || src.len() < width * height {
        return vec![0u8; width * height];
    }

    let mut output = vec![0u8; width * height];
    let mut src_x_buf = vec![0.0f64; width];
    let mut src_y_buf = vec![0.0f64; width];

    for y in 0..height {
        // Batch compute source coordinates for this row
        inverse_affine_row(params, y as f64, width, &mut src_x_buf, &mut src_y_buf);

        // Batch sample using bilinear interpolation
        let row_pixels = batch_bilinear_sample(src, width, height, &src_x_buf, &src_y_buf, width);

        output[y * width..(y + 1) * width].copy_from_slice(&row_pixels);
    }

    output
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close_f64(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_affine_params_identity() {
        let params = AffineParams::new(0.0, 0.0, 0.0, 1.0);
        assert!(close_f64(params.cos_a, 1.0, 1e-10));
        assert!(close_f64(params.sin_a, 0.0, 1e-10));
        assert!(close_f64(params.inv_scale, 1.0, 1e-10));
    }

    #[test]
    fn test_inverse_affine_row_identity() {
        let params = AffineParams::new(0.0, 0.0, 0.0, 1.0);
        let width = 16;
        let mut sx = vec![0.0f64; width];
        let mut sy = vec![0.0f64; width];

        inverse_affine_row(&params, 5.0, width, &mut sx, &mut sy);

        for (i, &v) in sx.iter().enumerate() {
            assert!(
                close_f64(v, i as f64, 1e-10),
                "sx[{i}] = {v}, expected {}",
                i as f64
            );
        }
        for &v in &sy {
            assert!(close_f64(v, 5.0, 1e-10), "sy = {v}, expected 5.0");
        }
    }

    #[test]
    fn test_inverse_affine_row_translation() {
        let params = AffineParams::new(10.0, 20.0, 0.0, 1.0);
        let width = 8;
        let mut sx = vec![0.0f64; width];
        let mut sy = vec![0.0f64; width];

        inverse_affine_row(&params, 25.0, width, &mut sx, &mut sy);

        // cx = x - 10, cy = 25 - 20 = 5
        for (i, &v) in sx.iter().enumerate() {
            let expected = i as f64 - 10.0;
            assert!(
                close_f64(v, expected, 1e-10),
                "sx[{i}] = {v}, expected {expected}"
            );
        }
        for &v in &sy {
            assert!(close_f64(v, 5.0, 1e-10));
        }
    }

    #[test]
    fn test_inverse_affine_row_scale() {
        let params = AffineParams::new(0.0, 0.0, 0.0, 2.0);
        let width = 8;
        let mut sx = vec![0.0f64; width];
        let mut sy = vec![0.0f64; width];

        inverse_affine_row(&params, 10.0, width, &mut sx, &mut sy);

        for (i, &v) in sx.iter().enumerate() {
            let expected = i as f64 / 2.0;
            assert!(
                close_f64(v, expected, 1e-10),
                "sx[{i}] = {v}, expected {expected}"
            );
        }
        for &v in &sy {
            assert!(close_f64(v, 5.0, 1e-10));
        }
    }

    #[test]
    fn test_inverse_affine_row_rotation() {
        // Test with non-zero angle to verify SIMD matches expected math
        let params = AffineParams::new(5.5, 3.2, 0.15, 1.1);
        let width = 33; // Not aligned to 4

        let mut sx = vec![0.0f64; width];
        let mut sy = vec![0.0f64; width];

        inverse_affine_row(&params, 42.7, width, &mut sx, &mut sy);

        // Verify against scalar computation
        let cy = 42.7 - params.dy;
        for i in 0..width {
            let cx = i as f64 - params.dx;
            let expected_x = (params.cos_a * cx + params.sin_a * cy) * params.inv_scale;
            let expected_y = (-params.sin_a * cx + params.cos_a * cy) * params.inv_scale;
            assert!(
                close_f64(sx[i], expected_x, 1e-10),
                "sx mismatch at {i}: got={}, expected={}",
                sx[i],
                expected_x
            );
            assert!(
                close_f64(sy[i], expected_y, 1e-10),
                "sy mismatch at {i}: got={}, expected={}",
                sy[i],
                expected_y
            );
        }
    }

    #[test]
    fn test_batch_bilinear_uniform() {
        let data = vec![128u8; 10 * 10];
        let sx = vec![5.5, 3.2, 7.8];
        let sy = vec![4.1, 6.9, 2.3];
        let result = batch_bilinear_sample(&data, 10, 10, &sx, &sy, 3);
        for &v in &result {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_warp_frame_identity() {
        let src = vec![100u8; 8 * 8];
        let params = AffineParams::new(0.0, 0.0, 0.0, 1.0);
        let result = warp_frame_simd(&src, 8, 8, &params);
        assert_eq!(result.len(), 64);
        for &v in &result {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_warp_frame_empty() {
        let result = warp_frame_simd(&[], 0, 0, &AffineParams::new(0.0, 0.0, 0.0, 1.0));
        assert!(result.is_empty());
    }

    #[test]
    fn test_warp_frame_translation() {
        // Create a gradient image: pixel value = x
        let width = 16;
        let height = 16;
        let src: Vec<u8> = (0..width * height).map(|i| (i % width) as u8).collect();

        // Translate by dx=0, dy=0 (identity) should give same image
        let params = AffineParams::new(0.0, 0.0, 0.0, 1.0);
        let result = warp_frame_simd(&src, width, height, &params);

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                assert_eq!(
                    result[y * width + x],
                    src[y * width + x],
                    "mismatch at ({x},{y})"
                );
            }
        }
    }
}

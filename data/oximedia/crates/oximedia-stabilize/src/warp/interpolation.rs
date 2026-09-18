//! Interpolation methods for frame warping.
//!
//! The bilinear path is accelerated via scirs2-core SIMD operations when
//! processing a full row of pixels at once (`bilinear_row_simd`).  The
//! per-pixel `InterpolationMethod::Bilinear` variant remains available for
//! mixed-stride access patterns.
//!
//! The optional `bilinear_4x_avx2` function uses AVX2 + SSE4.1 intrinsics
//! gated behind the corresponding `target_feature` compile flags.

// Allow unsafe code in this module for the AVX2 SIMD intrinsics path.
#![allow(unsafe_code)]

use crate::warp::boundary::BoundaryMode;
use scirs2_core::ndarray::Array2;

/// Interpolation method for pixel sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Nearest neighbor
    Nearest,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
}

impl InterpolationMethod {
    /// Interpolate pixel value at non-integer coordinates.
    #[must_use]
    pub fn interpolate(self, data: &Array2<u8>, x: f64, y: f64, boundary: BoundaryMode) -> u8 {
        match self {
            Self::Nearest => self.nearest(data, x, y, boundary),
            Self::Bilinear => self.bilinear(data, x, y, boundary),
            Self::Bicubic => self.bicubic(data, x, y, boundary),
        }
    }

    /// Nearest neighbor interpolation.
    fn nearest(self, data: &Array2<u8>, x: f64, y: f64, boundary: BoundaryMode) -> u8 {
        boundary.get_pixel(data, x.round(), y.round())
    }

    /// Bilinear interpolation.
    fn bilinear(self, data: &Array2<u8>, x: f64, y: f64, boundary: BoundaryMode) -> u8 {
        let x0 = x.floor();
        let y0 = y.floor();
        let dx = x - x0;
        let dy = y - y0;

        let p00 = f64::from(boundary.get_pixel(data, x0, y0));
        let p10 = f64::from(boundary.get_pixel(data, x0 + 1.0, y0));
        let p01 = f64::from(boundary.get_pixel(data, x0, y0 + 1.0));
        let p11 = f64::from(boundary.get_pixel(data, x0 + 1.0, y0 + 1.0));

        let val = (1.0 - dx) * (1.0 - dy) * p00
            + dx * (1.0 - dy) * p10
            + (1.0 - dx) * dy * p01
            + dx * dy * p11;

        val.clamp(0.0, 255.0) as u8
    }

    /// Bicubic interpolation.
    fn bicubic(self, data: &Array2<u8>, x: f64, y: f64, boundary: BoundaryMode) -> u8 {
        let x0 = x.floor();
        let y0 = y.floor();
        let dx = x - x0;
        let dy = y - y0;

        let mut val = 0.0;

        for j in -1..=2 {
            for i in -1..=2 {
                let p = f64::from(boundary.get_pixel(data, x0 + f64::from(i), y0 + f64::from(j)));
                let wx = Self::cubic_weight(dx - f64::from(i));
                let wy = Self::cubic_weight(dy - f64::from(j));
                val += p * wx * wy;
            }
        }

        val.clamp(0.0, 255.0) as u8
    }

    /// Cubic interpolation weight function.
    fn cubic_weight(t: f64) -> f64 {
        let t = t.abs();
        if t < 1.0 {
            1.5 * t.powi(3) - 2.5 * t.powi(2) + 1.0
        } else if t < 2.0 {
            -0.5 * t.powi(3) + 2.5 * t.powi(2) - 4.0 * t + 2.0
        } else {
            0.0
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  SIMD-accelerated row-level bilinear interpolation
// ─────────────────────────────────────────────────────────────────

/// SIMD-accelerated bilinear interpolation for an entire row of output pixels.
///
/// For each index `i` in `0..count`, reads the pre-computed source coordinates
/// `(src_x[i], src_y[i])`, performs bilinear sampling from `data` (row-major
/// single-channel u8, dimensions `data_width × data_height`), and writes the
/// result to `output[i]`.
///
/// Internally the fractional and floor operations are batched using
/// scirs2-core SIMD primitives so that the hot path runs with SIMD
/// width rather than one element at a time.
///
/// # Parameters
///
/// * `data`        — source grayscale image (row-major, u8)
/// * `data_width`  — source image width in pixels
/// * `data_height` — source image height in pixels
/// * `src_x`       — pre-computed fractional source X coordinates (length ≥ `count`)
/// * `src_y`       — pre-computed fractional source Y coordinates (length ≥ `count`)
/// * `count`       — number of pixels to interpolate
/// * `output`      — output buffer (length ≥ `count`)
///
/// # Panics
///
/// Panics if `src_x`, `src_y`, or `output` are shorter than `count`.
pub fn bilinear_row_simd(
    data: &[u8],
    data_width: usize,
    data_height: usize,
    src_x: &[f64],
    src_y: &[f64],
    count: usize,
    output: &mut [u8],
) {
    assert!(src_x.len() >= count, "src_x too short");
    assert!(src_y.len() >= count, "src_y too short");
    assert!(output.len() >= count, "output too short");

    if count == 0 || data_width == 0 || data_height == 0 {
        return;
    }

    // Use the batch bilinear sampler from simd_warp which already leverages
    // scirs2-core SIMD (floor, sub, scalar_mul) on arrays of coordinates.
    use crate::simd_warp::batch_bilinear_sample;
    let row_pixels = batch_bilinear_sample(data, data_width, data_height, src_x, src_y, count);
    output[..count].copy_from_slice(&row_pixels);
}

/// Warp a single-channel grayscale frame using an affine correction transform,
/// with SIMD-accelerated bilinear interpolation along each output row.
///
/// The correction transform is specified as:
///   - `dx`, `dy` — translational correction (pixels)
///   - `angle`    — rotational correction (radians)
///   - `scale`    — scale correction
///
/// Returns a new `width × height` u8 buffer.
#[must_use]
pub fn warp_bilinear_simd(
    src: &[u8],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    angle: f64,
    scale: f64,
) -> Vec<u8> {
    use crate::simd_warp::{inverse_affine_row, AffineParams};

    if width == 0 || height == 0 || src.len() < width * height {
        return vec![0u8; width * height];
    }

    let params = AffineParams::new(dx, dy, angle, scale);
    let mut dst = vec![0u8; width * height];
    let mut sx_buf = vec![0.0f64; width];
    let mut sy_buf = vec![0.0f64; width];

    for y in 0..height {
        inverse_affine_row(&params, y as f64, width, &mut sx_buf, &mut sy_buf);
        let row_out = &mut dst[y * width..(y + 1) * width];
        bilinear_row_simd(src, width, height, &sx_buf, &sy_buf, width, row_out);
    }

    dst
}

/// Bilinear interpolation quality score for a given warp transform.
///
/// Measures the average absolute difference between the warped output and
/// the source at grid sample points.  Lower scores indicate more faithful
/// reproduction (less resampling artifact).
///
/// Useful for comparing warp quality between different parameter choices.
#[must_use]
pub fn bilinear_quality_score(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst: &[u8],
    dst_width: usize,
    dst_height: usize,
) -> f64 {
    let n = src_width.min(dst_width) * src_height.min(dst_height);
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = src
        .iter()
        .zip(dst.iter())
        .take(n)
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .sum();
    sum / n as f64
}

// ─────────────────────────────────────────────────────────────────
//  Scalar per-pixel bilinear sample (public hot-path primitive)
// ─────────────────────────────────────────────────────────────────

/// Bilinear sample a single pixel from a flat, row-major, single-channel u8
/// frame at fractional position `(x, y)`.
///
/// Out-of-bounds positions are clamped to the nearest valid pixel (clamp-to-edge
/// boundary behaviour).  This mirrors the convention used by
/// [`bilinear_row_simd`] so the two can be cross-verified.
///
/// # Arguments
///
/// * `frame`  — row-major grayscale pixel data, `w * h` bytes
/// * `w`      — frame width in pixels
/// * `h`      — frame height in pixels
/// * `x`, `y` — fractional source coordinates
///
/// # Panics
///
/// Does not panic for any finite `(x, y)`.
#[inline]
#[must_use]
pub fn bilinear_sample(frame: &[u8], w: u32, h: u32, x: f32, y: f32) -> u8 {
    if w == 0 || h == 0 || frame.len() < (w * h) as usize {
        return 0;
    }

    // Clamp to valid range (clamp-to-edge)
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = f32::from(frame[(y0 * w + x0) as usize]);
    let p10 = f32::from(frame[(y0 * w + x1) as usize]);
    let p01 = f32::from(frame[(y1 * w + x0) as usize]);
    let p11 = f32::from(frame[(y1 * w + x1) as usize]);

    let val = (1.0 - fx) * (1.0 - fy) * p00
        + fx * (1.0 - fy) * p10
        + (1.0 - fx) * fy * p01
        + fx * fy * p11;

    val.clamp(0.0, 255.0) as u8
}

// ─────────────────────────────────────────────────────────────────
//  SIMD 4-pixel bilinear interpolation
// ─────────────────────────────────────────────────────────────────

/// Bilinear interpolate 4 pixels at once from a flat row-major single-channel
/// u8 frame.
///
/// # Safety
///
/// Caller must ensure that the CPU supports AVX2.  Use the safe wrapper
/// [`bilinear_4x`] which dispatches at runtime.
///
/// # Arguments
///
/// * `frame`     — row-major grayscale pixel data, `w * h` bytes
/// * `w`         — frame width
/// * `h`         — frame height
/// * `positions` — 4 `(x, y)` query positions (fractional, clamp-to-edge)
///
/// Returns 4 interpolated u8 values.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "sse4.1"
))]
#[target_feature(enable = "avx2", enable = "sse4.1")]
#[must_use]
pub unsafe fn bilinear_4x_avx2(
    frame: &[u8],
    w: u32,
    h: u32,
    positions: &[(f32, f32); 4],
) -> [u8; 4] {
    use std::arch::x86_64::*;

    if w == 0 || h == 0 || frame.len() < (w * h) as usize {
        return [0u8; 4];
    }

    // --- Clamp & floor all positions ----------------------------------------
    // We process 4 lanes in parallel.

    let wm1 = (w - 1) as f32;
    let hm1 = (h - 1) as f32;

    // Load x and y coordinates into two __m128 registers (4 × f32 each).
    let xs = _mm_set_ps(
        positions[3].0.clamp(0.0, wm1),
        positions[2].0.clamp(0.0, wm1),
        positions[1].0.clamp(0.0, wm1),
        positions[0].0.clamp(0.0, wm1),
    );
    let ys = _mm_set_ps(
        positions[3].1.clamp(0.0, hm1),
        positions[2].1.clamp(0.0, hm1),
        positions[1].1.clamp(0.0, hm1),
        positions[0].1.clamp(0.0, hm1),
    );

    // floor(x), floor(y)
    let x0f = _mm_floor_ps(xs);
    let y0f = _mm_floor_ps(ys);

    // fractional parts
    let fx = _mm_sub_ps(xs, x0f);
    let fy = _mm_sub_ps(ys, y0f);

    // Convert to integer pixel indices
    let x0i = _mm_cvtps_epi32(x0f);
    let y0i = _mm_cvtps_epi32(y0f);

    // x1 = min(x0+1, w-1),  y1 = min(y0+1, h-1)
    let one = _mm_set1_epi32(1);
    let wm1i = _mm_set1_epi32((w - 1) as i32);
    let hm1i = _mm_set1_epi32((h - 1) as i32);
    let x1i = _mm_min_epi32(_mm_add_epi32(x0i, one), wm1i);
    let y1i = _mm_min_epi32(_mm_add_epi32(y0i, one), hm1i);

    // Extract scalar indices for gather (4 lanes × 4 corners = 16 loads).
    let mut x0 = [0i32; 4];
    let mut y0 = [0i32; 4];
    let mut x1 = [0i32; 4];
    let mut y1 = [0i32; 4];
    _mm_storeu_si128(x0.as_mut_ptr() as *mut __m128i, x0i);
    _mm_storeu_si128(y0.as_mut_ptr() as *mut __m128i, y0i);
    _mm_storeu_si128(x1.as_mut_ptr() as *mut __m128i, x1i);
    _mm_storeu_si128(y1.as_mut_ptr() as *mut __m128i, y1i);
    let w = w as usize;

    // --- Gather 16 pixel values (4 pixels × 4 corners) ----------------------
    // p00[i], p10[i], p01[i], p11[i]
    let mut p00 = _mm_setzero_ps();
    let mut p10 = _mm_setzero_ps();
    let mut p01 = _mm_setzero_ps();
    let mut p11 = _mm_setzero_ps();

    for lane in 0..4i32 {
        let xi0 = x0[lane as usize] as usize;
        let yi0 = y0[lane as usize] as usize;
        let xi1 = x1[lane as usize] as usize;
        let yi1 = y1[lane as usize] as usize;
        let f00 = frame[yi0 * w + xi0] as f32;
        let f10 = frame[yi0 * w + xi1] as f32;
        let f01 = frame[yi1 * w + xi0] as f32;
        let f11 = frame[yi1 * w + xi1] as f32;

        // Insert f32 into the appropriate lane using _mm_insert_ps.
        // The imm8 encodes: [7:6]=src lane (always 0 from set_ss), [5:4]=dest lane, [3:0]=zero mask (0=none).
        p00 = match lane {
            0 => _mm_insert_ps(p00, _mm_set_ss(f00), 0x00),
            1 => _mm_insert_ps(p00, _mm_set_ss(f00), 0x10),
            2 => _mm_insert_ps(p00, _mm_set_ss(f00), 0x20),
            _ => _mm_insert_ps(p00, _mm_set_ss(f00), 0x30),
        };
        p10 = match lane {
            0 => _mm_insert_ps(p10, _mm_set_ss(f10), 0x00),
            1 => _mm_insert_ps(p10, _mm_set_ss(f10), 0x10),
            2 => _mm_insert_ps(p10, _mm_set_ss(f10), 0x20),
            _ => _mm_insert_ps(p10, _mm_set_ss(f10), 0x30),
        };
        p01 = match lane {
            0 => _mm_insert_ps(p01, _mm_set_ss(f01), 0x00),
            1 => _mm_insert_ps(p01, _mm_set_ss(f01), 0x10),
            2 => _mm_insert_ps(p01, _mm_set_ss(f01), 0x20),
            _ => _mm_insert_ps(p01, _mm_set_ss(f01), 0x30),
        };
        p11 = match lane {
            0 => _mm_insert_ps(p11, _mm_set_ss(f11), 0x00),
            1 => _mm_insert_ps(p11, _mm_set_ss(f11), 0x10),
            2 => _mm_insert_ps(p11, _mm_set_ss(f11), 0x20),
            _ => _mm_insert_ps(p11, _mm_set_ss(f11), 0x30),
        };
    }

    // --- Bilinear blend ------------------------------------------------------
    // val = (1-fx)*(1-fy)*p00 + fx*(1-fy)*p10 + (1-fx)*fy*p01 + fx*fy*p11
    let one_f = _mm_set1_ps(1.0);
    let inv_fx = _mm_sub_ps(one_f, fx);
    let inv_fy = _mm_sub_ps(one_f, fy);

    let t0 = _mm_mul_ps(_mm_mul_ps(inv_fx, inv_fy), p00);
    let t1 = _mm_mul_ps(_mm_mul_ps(fx, inv_fy), p10);
    let t2 = _mm_mul_ps(_mm_mul_ps(inv_fx, fy), p01);
    let t3 = _mm_mul_ps(_mm_mul_ps(fx, fy), p11);

    let val = _mm_add_ps(_mm_add_ps(t0, t1), _mm_add_ps(t2, t3));

    // Clamp to [0, 255] and convert to u8
    let clamped = _mm_min_ps(_mm_max_ps(val, _mm_setzero_ps()), _mm_set1_ps(255.0));
    let as_int = _mm_cvtps_epi32(clamped);
    let mut buf = [0i32; 4];
    _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, as_int);

    [buf[0] as u8, buf[1] as u8, buf[2] as u8, buf[3] as u8]
}

/// Bilinear interpolate 4 pixels at once.  Dispatches to the AVX2-accelerated
/// path at runtime when available; falls back to 4 scalar calls otherwise.
///
/// # Arguments
///
/// * `frame`     — row-major grayscale pixel data, `w * h` bytes
/// * `w`         — frame width
/// * `h`         — frame height
/// * `positions` — 4 `(x, y)` query positions (fractional, clamp-to-edge)
#[must_use]
pub fn bilinear_4x(frame: &[u8], w: u32, h: u32, positions: &[(f32, f32); 4]) -> [u8; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "sse4.1"
    ))]
    {
        // Safety: guarded by the compile-time target_feature check above;
        // both AVX2 and SSE4.1 are confirmed available at compile time.
        return unsafe { bilinear_4x_avx2(frame, w, h, positions) };
    }

    // Scalar fallback (non-x86_64, no AVX2, or no SSE4.1).
    [
        bilinear_sample(frame, w, h, positions[0].0, positions[0].1),
        bilinear_sample(frame, w, h, positions[1].0, positions[1].1),
        bilinear_sample(frame, w, h, positions[2].0, positions[2].1),
        bilinear_sample(frame, w, h, positions[3].0, positions[3].1),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest() {
        let data = Array2::from_elem((10, 10), 128);
        let val = InterpolationMethod::Nearest.interpolate(&data, 5.5, 5.5, BoundaryMode::Constant);
        assert_eq!(val, 128);
    }

    #[test]
    fn test_bilinear() {
        let data = Array2::from_elem((10, 10), 128);
        let val =
            InterpolationMethod::Bilinear.interpolate(&data, 5.5, 5.5, BoundaryMode::Constant);
        assert_eq!(val, 128);
    }

    #[test]
    fn test_bilinear_row_simd_uniform() {
        let data = vec![200u8; 16 * 16];
        let sx = vec![4.5f64; 8];
        let sy = vec![4.5f64; 8];
        let mut output = vec![0u8; 8];
        bilinear_row_simd(&data, 16, 16, &sx, &sy, 8, &mut output);
        for &v in &output {
            assert_eq!(v, 200, "uniform image should give uniform output");
        }
    }

    #[test]
    fn test_bilinear_row_simd_identity_row() {
        // Source: row-gradient image (value = column index)
        let width = 16usize;
        let height = 8usize;
        let data: Vec<u8> = (0..height)
            .flat_map(|_| (0..width).map(|x| x as u8))
            .collect();

        // Sample at exactly integer pixel positions → should reproduce source
        let sx: Vec<f64> = (0..width).map(|x| x as f64).collect();
        let sy = vec![2.0f64; width]; // row 2, no fractional part
        let mut output = vec![0u8; width];
        bilinear_row_simd(&data, width, height, &sx, &sy, width, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert_eq!(
                v, i as u8,
                "integer coordinates should reproduce source pixel"
            );
        }
    }

    #[test]
    fn test_bilinear_row_simd_zero_count() {
        let data = vec![99u8; 4];
        let mut output = vec![0u8; 0];
        // Should not panic with count=0
        bilinear_row_simd(&data, 2, 2, &[], &[], 0, &mut output);
    }

    #[test]
    fn test_warp_bilinear_simd_identity() {
        let src = vec![128u8; 16 * 16];
        let dst = warp_bilinear_simd(&src, 16, 16, 0.0, 0.0, 0.0, 1.0);
        assert_eq!(dst.len(), src.len());
        for &v in &dst {
            assert_eq!(
                v, 128,
                "identity warp on uniform image should preserve value"
            );
        }
    }

    #[test]
    fn test_warp_bilinear_simd_empty() {
        let result = warp_bilinear_simd(&[], 0, 0, 0.0, 0.0, 0.0, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bilinear_quality_score_identical() {
        let data = vec![100u8; 8 * 8];
        let score = bilinear_quality_score(&data, 8, 8, &data, 8, 8);
        assert!((score - 0.0).abs() < 1e-9, "identical images → score 0");
    }

    #[test]
    fn test_bilinear_quality_score_different() {
        let src = vec![0u8; 8 * 8];
        let dst = vec![128u8; 8 * 8];
        let score = bilinear_quality_score(&src, 8, 8, &dst, 8, 8);
        assert!(score > 0.0, "different images → positive score");
    }

    /// Verify that `bilinear_4x` (which dispatches to AVX2 or scalar) produces
    /// the same results as 4 independent `bilinear_sample` calls.
    #[test]
    fn test_bilinear_simd_matches_scalar() {
        // Build a small gradient frame (value = x + y * width)
        let w: u32 = 16;
        let h: u32 = 16;
        let frame: Vec<u8> = (0..h)
            .flat_map(|y| (0..w).map(move |x| ((x + y * w) % 256) as u8))
            .collect();

        // Four test positions with fractional coordinates
        let positions: [(f32, f32); 4] = [
            (1.0, 2.0),   // integer — exact lookup
            (3.5, 4.5),   // midpoint between 4 pixels
            (7.25, 8.75), // asymmetric fractions
            (14.9, 14.9), // near the far corner
        ];

        let simd_result = bilinear_4x(&frame, w, h, &positions);
        for (i, &pos) in positions.iter().enumerate() {
            let scalar = bilinear_sample(&frame, w, h, pos.0, pos.1);
            assert_eq!(
                simd_result[i], scalar,
                "bilinear_4x[{i}] = {} but bilinear_sample = {} at ({}, {})",
                simd_result[i], scalar, pos.0, pos.1
            );
        }
    }

    /// `bilinear_sample` on a uniform frame returns the constant value.
    #[test]
    fn test_bilinear_sample_uniform() {
        let frame = vec![77u8; 8 * 8];
        for &(x, y) in &[(0.0f32, 0.0f32), (3.5, 3.5), (7.0, 7.0)] {
            assert_eq!(bilinear_sample(&frame, 8, 8, x, y), 77);
        }
    }

    /// `bilinear_sample` clamps out-of-bounds coordinates (clamp-to-edge).
    #[test]
    fn test_bilinear_sample_clamp() {
        // Gradient frame: value = column index
        let w: u32 = 8;
        let h: u32 = 8;
        let frame: Vec<u8> = (0..h).flat_map(|_| (0..w).map(|x| x as u8)).collect();

        // x > w-1 should clamp to the last column (value = 7)
        let v = bilinear_sample(&frame, w, h, 100.0, 0.0);
        assert_eq!(v, 7, "x clamping: expected edge value 7, got {v}");

        // x < 0 should clamp to column 0 (value = 0)
        let v = bilinear_sample(&frame, w, h, -5.0, 0.0);
        assert_eq!(v, 0, "x clamping low: expected 0, got {v}");
    }
}

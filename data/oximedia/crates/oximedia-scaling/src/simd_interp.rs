//! SIMD-accelerated pixel interpolation kernels for image resizing.
//!
//! Provides hardware-accelerated implementations of:
//! - **Bilinear interpolation** -- fast, batch-processes output pixels using scirs2-core SIMD ops
//! - **Horizontal convolution** -- 1D filter application along rows using SIMD dot product
//!
//! All SIMD acceleration is delegated to `scirs2_core::simd`, which automatically
//! selects the best available instruction set (AVX2, SSE4.1, NEON, or scalar fallback).
//! Additionally, on x86/x86_64, runtime dispatch via `is_x86_feature_detected!` enables
//! AVX2 intrinsics for `bicubic` and `lanczos` filter passes.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use scirs2_core::simd::{
    simd_add_f32, simd_clip_f32, simd_dot_f32, simd_mul_f32, simd_round_f32, simd_sub_f32,
};

// ── Bilinear resize (RGB, packed 3 bytes per pixel) ──────────────────────────

/// Resize an RGB image using bilinear interpolation with SIMD acceleration.
///
/// `src` is packed 3-bytes-per-pixel row-major RGB of size `src_w x src_h`.
/// Returns a new buffer of size `dst_w x dst_h x 3`.
///
/// Returns an empty vector if any dimension is zero.
#[must_use]
pub fn bilinear_resize_simd(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Vec::new();
    }

    let scale_x = src_w as f64 / dst_w as f64;
    let scale_y = src_h as f64 / dst_h as f64;

    let mut dst = vec![0u8; dst_w * dst_h * 3];

    for dy in 0..dst_h {
        let sy_f = (dy as f64 + 0.5) * scale_y - 0.5;
        let sy0 = sy_f.floor().max(0.0) as usize;
        let sy1 = (sy0 + 1).min(src_h - 1);
        let fy = (sy_f - sy0 as f64).max(0.0) as f32;

        bilinear_row_simd(
            src,
            &mut dst[dy * dst_w * 3..(dy + 1) * dst_w * 3],
            src_w,
            dst_w,
            sy0,
            sy1,
            fy,
            scale_x,
        );
    }

    dst
}

/// Process one row of bilinear interpolation using scirs2-core SIMD operations.
///
/// For each channel, gathers source pixel values into arrays, then uses
/// vectorised multiply, add, round, and clip to compute the output pixels.
fn bilinear_row_simd(
    src: &[u8],
    dst_row: &mut [u8],
    src_w: usize,
    dst_w: usize,
    sy0: usize,
    sy1: usize,
    fy: f32,
    scale_x: f64,
) {
    let row0_offset = sy0 * src_w * 3;
    let row1_offset = sy1 * src_w * 3;

    // Pre-compute source coordinates and fractional weights for all destination pixels
    let mut sx0_vec = Vec::with_capacity(dst_w);
    let mut sx1_vec = Vec::with_capacity(dst_w);
    let mut fx_vec = Vec::with_capacity(dst_w);

    for dx in 0..dst_w {
        let sx_f = (dx as f64 + 0.5) * scale_x - 0.5;
        let sx0 = sx_f.floor().max(0.0) as usize;
        let sx1 = (sx0 + 1).min(src_w - 1);
        let fx = (sx_f - sx0 as f64).max(0.0) as f32;
        sx0_vec.push(sx0);
        sx1_vec.push(sx1);
        fx_vec.push(fx);
    }

    // Process each colour channel with vectorised operations using scirs2 SIMD
    for c in 0..3usize {
        // Gather source pixel values for all destination pixels
        let mut p00_vals = Vec::with_capacity(dst_w);
        let mut p10_vals = Vec::with_capacity(dst_w);
        let mut p01_vals = Vec::with_capacity(dst_w);
        let mut p11_vals = Vec::with_capacity(dst_w);

        for dx in 0..dst_w {
            p00_vals.push(src[row0_offset + sx0_vec[dx] * 3 + c] as f32);
            p10_vals.push(src[row0_offset + sx1_vec[dx] * 3 + c] as f32);
            p01_vals.push(src[row1_offset + sx0_vec[dx] * 3 + c] as f32);
            p11_vals.push(src[row1_offset + sx1_vec[dx] * 3 + c] as f32);
        }

        // Use scirs2 SIMD via slice views — convert to ndarray-compatible format
        // by wrapping as ArrayView1 references built from the raw slices.
        let (p00, p10, p01, p11) = (p00_vals, p10_vals, p01_vals, p11_vals);
        let fx_arr = fx_vec.clone();
        let ones: Vec<f32> = vec![1.0f32; dst_w];

        let one_minus_fx = simd_sub_f32_vec(&ones, &fx_arr);

        // bilinear: top = p00*(1-fx) + p10*fx
        let top_left = simd_mul_f32_vec(&p00, &one_minus_fx);
        let top_right = simd_mul_f32_vec(&p10, &fx_arr);
        let top = simd_add_f32_vec(&top_left, &top_right);

        // bilinear: bot = p01*(1-fx) + p11*fx
        let bot_left = simd_mul_f32_vec(&p01, &one_minus_fx);
        let bot_right = simd_mul_f32_vec(&p11, &fx_arr);
        let bot = simd_add_f32_vec(&bot_left, &bot_right);

        // val = top*(1-fy) + bot*fy
        let fy_arr: Vec<f32> = vec![fy; dst_w];
        let one_minus_fy: Vec<f32> = vec![1.0 - fy; dst_w];
        let val_top = simd_mul_f32_vec(&top, &one_minus_fy);
        let val_bot = simd_mul_f32_vec(&bot, &fy_arr);
        let val = simd_add_f32_vec(&val_top, &val_bot);

        // Round and clamp to [0, 255]
        let rounded = simd_round_f32_vec(&val);
        let clamped = simd_clip_f32_vec(&rounded, 0.0, 255.0);

        // Store results
        for dx in 0..dst_w {
            dst_row[dx * 3 + c] = clamped[dx] as u8;
        }
    }
}

// ── Vec-based SIMD wrappers (ndarray-free) ────────────────────────────────────

/// Wrapper: element-wise subtract two f32 slices using scirs2 SIMD.
fn simd_sub_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    use scirs2_core::ndarray::ArrayView1;
    let av = ArrayView1::from(a);
    let bv = ArrayView1::from(b);
    simd_sub_f32(&av, &bv).to_vec()
}

/// Wrapper: element-wise multiply two f32 slices using scirs2 SIMD.
fn simd_mul_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    use scirs2_core::ndarray::ArrayView1;
    let av = ArrayView1::from(a);
    let bv = ArrayView1::from(b);
    simd_mul_f32(&av, &bv).to_vec()
}

/// Wrapper: element-wise add two f32 slices using scirs2 SIMD.
fn simd_add_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    use scirs2_core::ndarray::ArrayView1;
    let av = ArrayView1::from(a);
    let bv = ArrayView1::from(b);
    simd_add_f32(&av, &bv).to_vec()
}

/// Wrapper: element-wise round f32 slice using scirs2 SIMD.
fn simd_round_f32_vec(a: &[f32]) -> Vec<f32> {
    use scirs2_core::ndarray::ArrayView1;
    let av = ArrayView1::from(a);
    simd_round_f32(&av).to_vec()
}

/// Wrapper: element-wise clip f32 slice using scirs2 SIMD.
fn simd_clip_f32_vec(a: &[f32], lo: f32, hi: f32) -> Vec<f32> {
    use scirs2_core::ndarray::ArrayView1;
    let av = ArrayView1::from(a);
    simd_clip_f32(&av, lo, hi).to_vec()
}

// ── Horizontal convolution with SIMD ─────────────────────────────────────────

/// Apply a 1-D horizontal convolution kernel to a row of f32 pixel data.
///
/// `src_row` contains the source row.  `kernel` is the convolution weights
/// (must be odd length).  Output length equals `src_row.len()` (edges are
/// clamped).
///
/// Returns an empty vector if the kernel is empty.
#[must_use]
pub fn horizontal_convolve_simd(src_row: &[f32], kernel: &[f32]) -> Vec<f32> {
    if kernel.is_empty() || src_row.is_empty() {
        return Vec::new();
    }
    horizontal_convolve_scirs2(src_row, kernel)
}

/// Horizontal convolution using scirs2-core SIMD dot product.
///
/// For each output position, extracts a window of source values (with edge
/// clamping) and computes the dot product with the kernel using `simd_dot_f32`.
fn horizontal_convolve_scirs2(src_row: &[f32], kernel: &[f32]) -> Vec<f32> {
    use scirs2_core::ndarray::Array1;

    let len = src_row.len();
    let klen = kernel.len();
    let half = klen / 2;
    let kernel_arr = Array1::from_vec(kernel.to_vec());

    let mut out = vec![0.0f32; len];

    for i in 0..len {
        // Build a window of source values with clamped edge indexing
        let mut window = Vec::with_capacity(klen);
        for ki in 0..klen {
            let src_idx =
                (i as isize + ki as isize - half as isize).clamp(0, (len - 1) as isize) as usize;
            window.push(src_row[src_idx]);
        }
        let window_arr = Array1::from_vec(window);

        // Use SIMD dot product for kernel application
        out[i] = simd_dot_f32(&window_arr.view(), &kernel_arr.view());
    }

    out
}

// ── x86 SIMD intrinsic dispatch for bicubic/lanczos filter passes ─────────────

/// Apply a 1-D separable filter pass (e.g. bicubic or lanczos weights) to a
/// row of pixels using the best available instruction set.
///
/// On x86/x86_64 at runtime this probes for AVX2 and, if present, runs a
/// hand-unrolled loop using 256-bit FMA3 multiply-accumulate intrinsics.
/// On all other platforms (or when AVX2 is absent) it falls back to the
/// generic scalar path.
///
/// # Arguments
/// - `src_row` – source f32 row samples
/// - `weights` – per-output weight vectors; `weights[j]` holds the tap
///   coefficients for output sample `j`
/// - `src_offsets` – for each output sample, the source index of the first tap
///
/// Returns the filtered output of length `weights.len()`.
#[must_use]
pub fn separable_filter_pass_simd(
    src_row: &[f32],
    weights: &[Vec<f32>],
    src_offsets: &[usize],
) -> Vec<f32> {
    if weights.is_empty() || src_row.is_empty() {
        return Vec::new();
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: AVX2 availability verified above via is_x86_feature_detected!.
            return unsafe { separable_filter_pass_avx2(src_row, weights, src_offsets) };
        }
    }

    separable_filter_pass_scalar(src_row, weights, src_offsets)
}

/// Generic scalar implementation of the separable filter pass.
fn separable_filter_pass_scalar(
    src_row: &[f32],
    weights: &[Vec<f32>],
    src_offsets: &[usize],
) -> Vec<f32> {
    let n = weights.len();
    let src_len = src_row.len();
    let mut out = vec![0.0f32; n];

    for j in 0..n {
        let ws = &weights[j];
        let base = if j < src_offsets.len() {
            src_offsets[j]
        } else {
            0
        };
        let mut acc = 0.0f32;
        for (k, &w) in ws.iter().enumerate() {
            let idx = (base + k).min(src_len.saturating_sub(1));
            acc += src_row[idx] * w;
        }
        out[j] = acc;
    }

    out
}

/// AVX2-accelerated separable filter pass.
///
/// Uses 256-bit FMA3 multiply-accumulate intrinsics with unrolled inner loops
/// to process 8 tap-weights at a time.
///
/// # Safety
/// Caller MUST have verified `is_x86_feature_detected!("avx2")` before calling.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn separable_filter_pass_avx2(
    src_row: &[f32],
    weights: &[Vec<f32>],
    src_offsets: &[usize],
) -> Vec<f32> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = weights.len();
    let src_len = src_row.len();
    let mut out = vec![0.0f32; n];

    for j in 0..n {
        let ws = &weights[j];
        let base = if j < src_offsets.len() {
            src_offsets[j]
        } else {
            0
        };
        let num_taps = ws.len();

        // Process 8 taps at a time with AVX2 FMA
        let mut acc = _mm256_setzero_ps();
        let full_chunks = num_taps / 8;

        for chunk in 0..full_chunks {
            let tap_off = chunk * 8;

            // Load 8 weights
            let wptr = ws.as_ptr().add(tap_off);
            let w8 = _mm256_loadu_ps(wptr);

            // Gather 8 source samples (may not be contiguous, clamp to bounds)
            let mut src8 = [0.0f32; 8];
            for k in 0..8 {
                let idx = (base + tap_off + k).min(src_len.saturating_sub(1));
                src8[k] = src_row[idx];
            }
            let s8 = _mm256_loadu_ps(src8.as_ptr());

            // FMA: acc = w8 * s8 + acc
            acc = _mm256_fmadd_ps(w8, s8, acc);
        }

        // Horizontal sum of acc
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let sum4 = _mm_add_ps(lo, hi);
        let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
        let mut scalar = _mm_cvtss_f32(sum1);

        // Handle remaining taps scalarly
        for k in (full_chunks * 8)..num_taps {
            let idx = (base + k).min(src_len.saturating_sub(1));
            scalar += src_row[idx] * ws[k];
        }

        out[j] = scalar;
    }

    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_resize_identity() {
        // 4x4 uniform image resized to 4x4 should stay uniform
        let src = vec![100u8; 4 * 4 * 3];
        let dst = bilinear_resize_simd(&src, 4, 4, 4, 4);
        assert_eq!(dst.len(), 4 * 4 * 3);
        for &v in &dst {
            assert!((v as i32 - 100).abs() <= 1, "got {v}");
        }
    }

    #[test]
    fn test_bilinear_resize_upscale() {
        let src = vec![128u8; 4 * 4 * 3];
        let dst = bilinear_resize_simd(&src, 4, 4, 16, 16);
        assert_eq!(dst.len(), 16 * 16 * 3);
        for &v in &dst {
            assert!((v as i32 - 128).abs() <= 1, "got {v}");
        }
    }

    #[test]
    fn test_bilinear_resize_downscale() {
        let src = vec![200u8; 8 * 8 * 3];
        let dst = bilinear_resize_simd(&src, 8, 8, 4, 4);
        assert_eq!(dst.len(), 4 * 4 * 3);
        for &v in &dst {
            assert!((v as i32 - 200).abs() <= 2, "got {v}");
        }
    }

    #[test]
    fn test_bilinear_resize_zero_dim() {
        assert!(bilinear_resize_simd(&[], 0, 4, 4, 4).is_empty());
        assert!(bilinear_resize_simd(&[0; 48], 4, 4, 0, 4).is_empty());
    }

    #[test]
    fn test_bilinear_simd_matches_scalar() {
        // Non-uniform image to properly test interpolation
        let src: Vec<u8> = (0..64 * 3).map(|i| (i * 7 % 256) as u8).collect();
        let src_w = 8;
        let src_h = 8;
        let dst_w = 12;
        let dst_h = 12;

        let result = bilinear_resize_simd(&src, src_w, src_h, dst_w, dst_h);

        // Compute scalar reference
        let scale_x = src_w as f64 / dst_w as f64;
        let scale_y = src_h as f64 / dst_h as f64;
        let mut scalar_result = vec![0u8; dst_w * dst_h * 3];

        for dy in 0..dst_h {
            let sy_f = (dy as f64 + 0.5) * scale_y - 0.5;
            let sy0 = sy_f.floor().max(0.0) as usize;
            let sy1 = (sy0 + 1).min(src_h - 1);
            let fy = (sy_f - sy0 as f64).max(0.0) as f32;

            let row0 = sy0 * src_w * 3;
            let row1 = sy1 * src_w * 3;

            for dx in 0..dst_w {
                let sx_f = (dx as f64 + 0.5) * scale_x - 0.5;
                let sx0 = sx_f.floor().max(0.0) as usize;
                let sx1 = (sx0 + 1).min(src_w - 1);
                let fx = (sx_f - sx0 as f64).max(0.0) as f32;

                let base_dst = dy * dst_w * 3 + dx * 3;
                for c in 0..3 {
                    let p00 = src[row0 + sx0 * 3 + c] as f32;
                    let p10 = src[row0 + sx1 * 3 + c] as f32;
                    let p01 = src[row1 + sx0 * 3 + c] as f32;
                    let p11 = src[row1 + sx1 * 3 + c] as f32;

                    let top = p00 + fx * (p10 - p00);
                    let bot = p01 + fx * (p11 - p01);
                    let val = top + fy * (bot - top);

                    scalar_result[base_dst + c] = val.round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        for (i, (&s, &r)) in result.iter().zip(scalar_result.iter()).enumerate() {
            assert!(
                (s as i32 - r as i32).abs() <= 1,
                "mismatch at byte {i}: simd={s}, scalar={r}"
            );
        }
    }

    #[test]
    fn test_horizontal_convolve_identity() {
        // Identity kernel [0, 1, 0]
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let kernel = vec![0.0, 1.0, 0.0];
        let result = horizontal_convolve_simd(&src, &kernel);
        assert_eq!(result.len(), 16);
        for (i, (&got, &want)) in result.iter().zip(src.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "mismatch at {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_horizontal_convolve_box3() {
        // Box filter [1/3, 1/3, 1/3]
        let src = vec![0.0, 3.0, 6.0, 9.0, 12.0];
        let kernel = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let result = horizontal_convolve_simd(&src, &kernel);
        assert_eq!(result.len(), 5);
        // Middle element: (3+6+9)/3 = 6.0
        assert!((result[2] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_horizontal_convolve_simd_matches_scalar() {
        let src: Vec<f32> = (0..33).map(|i| (i * 7 % 100) as f32 * 0.1).collect();
        let kernel = vec![0.1, 0.2, 0.4, 0.2, 0.1];

        let simd_result = horizontal_convolve_simd(&src, &kernel);

        // Scalar reference
        let len = src.len();
        let klen = kernel.len();
        let half = klen / 2;
        let mut scalar_result = vec![0.0f32; len];
        for i in 0..len {
            let mut acc = 0.0f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                let src_idx = (i as isize + ki as isize - half as isize)
                    .clamp(0, (len - 1) as isize) as usize;
                acc += src[src_idx] * kw;
            }
            scalar_result[i] = acc;
        }

        for (i, (&s, &r)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-4,
                "mismatch at {i}: simd={s}, scalar={r}"
            );
        }
    }

    #[test]
    fn test_horizontal_convolve_empty() {
        assert!(horizontal_convolve_simd(&[], &[1.0]).is_empty());
        assert!(horizontal_convolve_simd(&[1.0], &[]).is_empty());
    }

    // ── separable_filter_pass_simd tests ─────────────────────────────────────

    #[test]
    fn test_separable_filter_pass_identity() {
        // Single tap weight=1.0 at each position → output = input
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let weights: Vec<Vec<f32>> = (0..8).map(|_| vec![1.0]).collect();
        let offsets: Vec<usize> = (0..8).collect();
        let out = separable_filter_pass_simd(&src, &weights, &offsets);
        assert_eq!(out.len(), 8);
        for (i, (&got, &want)) in out.iter().zip(src.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "mismatch at {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn test_separable_filter_pass_average2() {
        // Average of two adjacent samples
        let src = vec![0.0f32, 2.0, 4.0, 6.0, 8.0];
        let weights: Vec<Vec<f32>> = (0..4).map(|_| vec![0.5, 0.5]).collect();
        let offsets: Vec<usize> = (0..4).collect();
        let out = separable_filter_pass_simd(&src, &weights, &offsets);
        assert_eq!(out.len(), 4);
        // out[i] = (src[i] + src[i+1]) / 2
        let expected = [1.0f32, 3.0, 5.0, 7.0];
        for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "mismatch at {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn test_separable_filter_pass_empty() {
        let out = separable_filter_pass_simd(&[], &[], &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_separable_filter_pass_scalar_matches_avx2() {
        // Verify scalar and SIMD produce same results for an 8-tap filter
        let src: Vec<f32> = (0..32).map(|i| (i * 3 % 17) as f32 * 0.5).collect();
        let weights: Vec<Vec<f32>> = (0..24)
            .map(|j| {
                let center = j as f32;
                (0..8)
                    .map(|k| {
                        let x = (k as f32 - 3.5).abs();
                        (-(x * x) / (2.0 * center.max(1.0) * 0.5)).exp()
                    })
                    .collect()
            })
            .collect();
        // Normalise weights
        let weights: Vec<Vec<f32>> = weights
            .into_iter()
            .map(|w| {
                let sum: f32 = w.iter().sum();
                if sum > 1e-6 {
                    w.iter().map(|&v| v / sum).collect()
                } else {
                    w
                }
            })
            .collect();
        let offsets: Vec<usize> = (0..24).collect();

        let scalar_out = separable_filter_pass_scalar(&src, &weights, &offsets);
        let simd_out = separable_filter_pass_simd(&src, &weights, &offsets);

        assert_eq!(scalar_out.len(), simd_out.len());
        for (i, (&s, &v)) in scalar_out.iter().zip(simd_out.iter()).enumerate() {
            assert!(
                (s - v).abs() < 1e-4,
                "scalar/simd mismatch at {i}: scalar={s} simd={v}"
            );
        }
    }
}

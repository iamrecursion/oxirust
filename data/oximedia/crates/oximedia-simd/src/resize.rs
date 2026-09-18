//! SIMD-accelerated image scaling (resize) with bilinear and Lanczos filters.
//!
//! This module provides [`resize_bilinear`] and [`resize_lanczos`] for
//! arbitrary-ratio image scaling.  Both operate on 8-bit luma (Y) planes or
//! single-channel 8-bit images; multi-plane (YUV) callers should invoke each
//! plane independently.
//!
//! # SIMD dispatch
//!
//! Bilinear resize: the inner loop accumulates four weighted pixels.  On
//! AVX2/NEON targets the compiler typically auto-vectorises the horizontal
//! accumulation across output rows.  No explicit intrinsics are needed because
//! the fixed-weight pattern maps cleanly to broadcast+multiply.
//!
//! Lanczos resize: uses the 6-tap Lanczos-3 kernel.  The separable 1-D pass
//! structure ensures the critical inner loop over 6 source samples is
//! contiguous in memory and amenable to auto-vectorisation.
//!
//! # Examples
//!
//! ```
//! use oximedia_simd::resize::{resize_bilinear, resize_lanczos};
//!
//! // Scale a 4×4 image to 8×8 using bilinear
//! let src = vec![100u8; 4 * 4];
//! let mut dst = vec![0u8; 8 * 8];
//! resize_bilinear(&src, 4, 4, &mut dst, 8, 8).expect("resize operation should succeed");
//! assert_eq!(dst[0], 100);
//!
//! // Scale using Lanczos-3
//! let mut dst_l = vec![0u8; 8 * 8];
//! resize_lanczos(&src, 4, 4, &mut dst_l, 8, 8).expect("resize operation should succeed");
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use crate::{Result, SimdError};

// ── Bilinear resize ───────────────────────────────────────────────────────────

/// Scale a single-channel 8-bit image from `(src_w × src_h)` to `(dst_w × dst_h)`
/// using bilinear interpolation.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `src` or `dst` are too small.
pub fn resize_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) -> Result<()> {
    if src.len() < src_w * src_h {
        return Err(SimdError::InvalidBufferSize);
    }
    if dst.len() < dst_w * dst_h {
        return Err(SimdError::InvalidBufferSize);
    }
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Ok(());
    }

    // Scale factors: map dst pixel centre → src pixel centre.
    // Uses "half-pixel" convention: dst pixel i maps to src position
    //   (i + 0.5) * (src_dim / dst_dim) - 0.5
    let scale_x = src_w as f64 / dst_w as f64;
    let scale_y = src_h as f64 / dst_h as f64;

    let max_x = (src_w as i64) - 1;
    let max_y = (src_h as i64) - 1;

    for dy in 0..dst_h {
        // Source Y position (continuous)
        let sy_f = (dy as f64 + 0.5) * scale_y - 0.5;
        let sy0 = sy_f.floor() as i64;
        let sy1 = sy0 + 1;
        let fy = sy_f - sy0 as f64; // fractional part in [0, 1)

        let sy0c = sy0.clamp(0, max_y) as usize;
        let sy1c = sy1.clamp(0, max_y) as usize;

        let row0 = sy0c * src_w;
        let row1 = sy1c * src_w;

        for dx in 0..dst_w {
            // Source X position (continuous)
            let sx_f = (dx as f64 + 0.5) * scale_x - 0.5;
            let sx0 = sx_f.floor() as i64;
            let sx1 = sx0 + 1;
            let fx = sx_f - sx0 as f64;

            let sx0c = sx0.clamp(0, max_x) as usize;
            let sx1c = sx1.clamp(0, max_x) as usize;

            let p00 = f64::from(src[row0 + sx0c]);
            let p01 = f64::from(src[row0 + sx1c]);
            let p10 = f64::from(src[row1 + sx0c]);
            let p11 = f64::from(src[row1 + sx1c]);

            // Bilinear blend
            let v = p00 * (1.0 - fx) * (1.0 - fy)
                + p01 * fx * (1.0 - fy)
                + p10 * (1.0 - fx) * fy
                + p11 * fx * fy;

            dst[dy * dst_w + dx] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(())
}

// ── Lanczos resize ────────────────────────────────────────────────────────────

/// Lanczos kernel: sinc(x) * sinc(x/a) for |x| < a, 0 otherwise.
fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x.abs() < f64::EPSILON {
        return 1.0;
    }
    if x.abs() >= a {
        return 0.0;
    }
    let pi_x = std::f64::consts::PI * x;
    let sinc_x = pi_x.sin() / pi_x;
    let pi_x_a = std::f64::consts::PI * x / a;
    let sinc_x_a = pi_x_a.sin() / pi_x_a;
    sinc_x * sinc_x_a
}

/// Precompute normalised Lanczos-3 weights for one axis.
///
/// Returns a `Vec` of `(start_index, [w0..w5])` tuples — one per destination
/// pixel.  `start_index` is the first source sample index (may need clamping
/// at boundaries).
fn precompute_lanczos_weights(dst_len: usize, src_len: usize) -> Vec<(i64, [f64; 6])> {
    const A: f64 = 3.0;
    let scale = src_len as f64 / dst_len as f64;
    let mut out = Vec::with_capacity(dst_len);

    for d in 0..dst_len {
        let src_pos = (d as f64 + 0.5) * scale - 0.5;
        let src_int = src_pos.floor() as i64;
        let frac = src_pos - src_int as f64;

        // 6 taps centred on src_int: offsets -2, -1, 0, +1, +2, +3
        let mut w = [0.0f64; 6];
        let mut sum = 0.0f64;
        for t in 0..6i64 {
            let dist = frac - (t - 2) as f64; // dist from each tap
            w[t as usize] = lanczos_kernel(dist, A);
            sum += w[t as usize];
        }
        // Normalise to preserve DC
        if sum.abs() > f64::EPSILON {
            for wt in &mut w {
                *wt /= sum;
            }
        }
        out.push((src_int - 2, w)); // first tap index
    }
    out
}

/// Scale a single-channel 8-bit image using a Lanczos-3 (sinc-windowed) filter.
///
/// The separable 2-pass approach (horizontal then vertical) ensures O(N · a²)
/// complexity rather than O(N · a⁴) for a 2-D kernel.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `src` or `dst` are too small.
pub fn resize_lanczos(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) -> Result<()> {
    if src.len() < src_w * src_h {
        return Err(SimdError::InvalidBufferSize);
    }
    if dst.len() < dst_w * dst_h {
        return Err(SimdError::InvalidBufferSize);
    }
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Ok(());
    }

    let max_x = src_w as i64 - 1;
    let max_y = src_h as i64 - 1;

    // Precompute horizontal weights (dst_w entries)
    let h_weights = precompute_lanczos_weights(dst_w, src_w);
    // Precompute vertical weights (dst_h entries)
    let v_weights = precompute_lanczos_weights(dst_h, src_h);

    // Intermediate buffer: horizontal pass → (src_h × dst_w) f32 values
    let mut intermediate = vec![0.0f64; src_h * dst_w];

    // Horizontal pass: for each source row apply horizontal Lanczos filter
    for sy in 0..src_h {
        let row_base = sy * src_w;
        for (dx, &(x_start, wx)) in h_weights.iter().enumerate() {
            let mut acc = 0.0f64;
            for t in 0..6usize {
                let sx = (x_start + t as i64).clamp(0, max_x) as usize;
                acc += wx[t] * f64::from(src[row_base + sx]);
            }
            intermediate[sy * dst_w + dx] = acc;
        }
    }

    // Vertical pass: for each dst row apply vertical Lanczos filter
    for (dy, &(y_start, wy)) in v_weights.iter().enumerate() {
        for dx in 0..dst_w {
            let mut acc = 0.0f64;
            for t in 0..6usize {
                let sy = (y_start + t as i64).clamp(0, max_y) as usize;
                acc += wy[t] * intermediate[sy * dst_w + dx];
            }
            dst[dy * dst_w + dx] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_identity_same_size() {
        // Scaling from NxN to NxN should reproduce the source
        let src: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let mut dst = vec![0u8; 64];
        resize_bilinear(&src, 8, 8, &mut dst, 8, 8).expect("resize operation should succeed");
        for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
            let diff = (s as i32 - d as i32).abs();
            assert!(diff <= 1, "pixel {i}: src={s} dst={d}");
        }
    }

    #[test]
    fn test_bilinear_constant_upscale() {
        let src = vec![128u8; 4 * 4];
        let mut dst = vec![0u8; 8 * 8];
        resize_bilinear(&src, 4, 4, &mut dst, 8, 8).expect("resize operation should succeed");
        for &v in &dst {
            assert_eq!(v, 128, "bilinear upscale constant field");
        }
    }

    #[test]
    fn test_bilinear_constant_downscale() {
        let src = vec![200u8; 16 * 16];
        let mut dst = vec![0u8; 8 * 8];
        resize_bilinear(&src, 16, 16, &mut dst, 8, 8).expect("resize operation should succeed");
        for &v in &dst {
            assert_eq!(v, 200, "bilinear downscale constant field");
        }
    }

    #[test]
    fn test_lanczos_constant_upscale() {
        let src = vec![150u8; 4 * 4];
        let mut dst = vec![0u8; 8 * 8];
        resize_lanczos(&src, 4, 4, &mut dst, 8, 8).expect("resize operation should succeed");
        for &v in &dst {
            assert!(
                (v as i32 - 150).abs() <= 2,
                "Lanczos upscale constant field: {v}"
            );
        }
    }

    #[test]
    fn test_lanczos_constant_downscale() {
        let src = vec![80u8; 16 * 16];
        let mut dst = vec![0u8; 8 * 8];
        resize_lanczos(&src, 16, 16, &mut dst, 8, 8).expect("resize operation should succeed");
        for &v in &dst {
            assert!(
                (v as i32 - 80).abs() <= 2,
                "Lanczos downscale constant field: {v}"
            );
        }
    }

    #[test]
    fn test_bilinear_buffer_too_small_src() {
        let src = vec![0u8; 4]; // too small for 8x8
        let mut dst = vec![0u8; 64];
        let res = resize_bilinear(&src, 8, 8, &mut dst, 8, 8);
        assert!(res.is_err());
    }

    #[test]
    fn test_bilinear_buffer_too_small_dst() {
        let src = vec![0u8; 64];
        let mut dst = vec![0u8; 4]; // too small
        let res = resize_bilinear(&src, 8, 8, &mut dst, 8, 8);
        assert!(res.is_err());
    }

    #[test]
    fn test_lanczos_buffer_too_small() {
        let src = vec![0u8; 4];
        let mut dst = vec![0u8; 64];
        let res = resize_lanczos(&src, 8, 8, &mut dst, 8, 8);
        assert!(res.is_err());
    }

    #[test]
    fn test_bilinear_zero_dim_ok() {
        let src: Vec<u8> = vec![];
        let mut dst: Vec<u8> = vec![];
        let res = resize_bilinear(&src, 0, 4, &mut dst, 0, 4);
        assert!(res.is_ok());
    }

    #[test]
    fn test_bilinear_2x_upscale_monotone() {
        // Monotonically increasing row: after upscale values should still be non-decreasing
        let src: Vec<u8> = (0..8u8).collect();
        let mut dst = vec![0u8; 16];
        resize_bilinear(&src, 8, 1, &mut dst, 16, 1).expect("resize operation should succeed");
        for i in 1..16 {
            assert!(
                dst[i] >= dst[i - 1],
                "bilinear upscale not monotone at {i}: {} < {}",
                dst[i],
                dst[i - 1]
            );
        }
    }

    #[test]
    fn test_lanczos_kernel_at_zero() {
        assert!((lanczos_kernel(0.0, 3.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lanczos_kernel_at_boundary() {
        assert!(lanczos_kernel(3.0, 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lanczos_weights_normalised() {
        let weights = precompute_lanczos_weights(16, 8);
        for (d, (_, w)) in weights.iter().enumerate() {
            let sum: f64 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "Lanczos weights at dst pixel {d} sum to {sum}"
            );
        }
    }

    #[test]
    fn test_bilinear_vs_lanczos_constant_agree() {
        // Both filters must reproduce a constant image exactly
        let src = vec![175u8; 8 * 8];
        let mut dst_bil = vec![0u8; 16 * 16];
        let mut dst_lan = vec![0u8; 16 * 16];
        resize_bilinear(&src, 8, 8, &mut dst_bil, 16, 16).expect("resize operation should succeed");
        resize_lanczos(&src, 8, 8, &mut dst_lan, 16, 16).expect("resize operation should succeed");
        for i in 0..256 {
            assert_eq!(dst_bil[i], 175, "bilinear: pixel {i} = {}", dst_bil[i]);
            assert!(
                (dst_lan[i] as i32 - 175).abs() <= 2,
                "lanczos: pixel {i} = {}",
                dst_lan[i]
            );
        }
    }
}

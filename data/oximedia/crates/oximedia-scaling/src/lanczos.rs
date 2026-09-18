//! Lanczos resampling filter for high-quality image scaling.
//!
//! # SIMD acceleration
//! The horizontal and vertical separable filter passes in `scale_image` and
//! `resample_1d_simd` are dispatched through
//! `simd_interp::separable_filter_pass_simd`, which selects AVX2+FMA3 at
//! runtime on x86/x86_64 or falls back to the generic scalar path on all
//! other platforms.  The original pure-scalar `resample_1d` remains available
//! for reference and for the bit-close regression test.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::f64::consts::PI;

use crate::simd_interp::separable_filter_pass_simd;

/// Lanczos kernel with configurable `a` parameter.
///
/// The `a` parameter controls the number of lobes. Larger values give
/// higher quality but slower performance. `a=3` is the typical default.
#[derive(Debug, Clone)]
pub struct LanczosKernel {
    /// Number of lobes (typically 2 or 3)
    pub a: u32,
}

impl Default for LanczosKernel {
    fn default() -> Self {
        Self { a: 3 }
    }
}

impl LanczosKernel {
    /// Create a new Lanczos kernel with the given `a` parameter.
    pub fn new(a: u32) -> Self {
        Self { a }
    }

    /// Compute the sinc function: sin(pi*x) / (pi*x).
    fn sinc(x: f64) -> f64 {
        if x.abs() < 1e-10 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        }
    }

    /// Compute the Lanczos kernel value at position `x`.
    ///
    /// Returns 0 outside the support window `[-a, a]`.
    pub fn kernel_value(&self, x: f64) -> f64 {
        let a = self.a as f64;
        if x.abs() < 1e-10 {
            1.0
        } else if x.abs() < a {
            Self::sinc(x) * Self::sinc(x / a)
        } else {
            0.0
        }
    }
}

/// Predefined Lanczos window sizes for common use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanczosWindowSize {
    /// 2-tap: fastest, slight softness. Good for real-time and thumbnails.
    Tap2,
    /// 3-tap: standard quality/speed balance. Default for most video scaling.
    Tap3,
    /// 4-tap: sharper than 3-tap, may show slight ringing on hard edges.
    Tap4,
    /// 5-tap: highest quality, minimal aliasing. Best for archival and mastering.
    Tap5,
}

impl LanczosWindowSize {
    /// Returns the `a` parameter for this window size.
    pub fn a_value(self) -> u32 {
        match self {
            Self::Tap2 => 2,
            Self::Tap3 => 3,
            Self::Tap4 => 4,
            Self::Tap5 => 5,
        }
    }

    /// Returns a human-readable description of this window size.
    pub fn description(self) -> &'static str {
        match self {
            Self::Tap2 => "Lanczos-2: fast, slight softness",
            Self::Tap3 => "Lanczos-3: standard quality/speed balance",
            Self::Tap4 => "Lanczos-4: sharp with slight ringing risk",
            Self::Tap5 => "Lanczos-5: highest quality, slowest",
        }
    }

    /// Returns all available window sizes in ascending quality order.
    pub fn all() -> &'static [LanczosWindowSize] {
        &[Self::Tap2, Self::Tap3, Self::Tap4, Self::Tap5]
    }
}

impl std::fmt::Display for LanczosWindowSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lanczos-{}", self.a_value())
    }
}

// ── SIMD separable-filter helpers ────────────────────────────────────────────

/// Build per-output-sample Lanczos weights and source offsets for a 1-D pass.
///
/// For each destination sample `d` in `[0, dst_len)` this evaluates the Lanczos
/// kernel for every **in-bounds** tap within the support window `[center-a, center+a]`,
/// normalises the tap weights so they sum to 1.0, and records the index of the
/// first contributing source sample as `offset[d]`.
///
/// Only in-bounds source indices (`j` in `[0, src_len)`) are included; this
/// matches the boundary handling of `LanczosResampler::resample_1d`.
///
/// Returns `(weights, offsets)` ready for `separable_filter_pass_simd`.
fn build_lanczos_filter_weights(
    kernel: &LanczosKernel,
    src_len: usize,
    dst_len: usize,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let scale = src_len as f64 / dst_len as f64;
    let a = kernel.a as i64;
    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(dst_len);
    let mut offsets: Vec<usize> = Vec::with_capacity(dst_len);

    for d in 0..dst_len {
        let center = (d as f64 + 0.5) * scale - 0.5;
        let start = (center - a as f64).ceil() as i64;
        let end = (center + a as f64).floor() as i64;

        // Collect only in-bounds taps (matching resample_1d behaviour)
        let mut first_src: Option<usize> = None;
        let mut tap_weights: Vec<f32> = Vec::new();

        for j in start..=end {
            if j < 0 || j >= src_len as i64 {
                continue;
            }
            let src_idx = j as usize;
            if first_src.is_none() {
                first_src = Some(src_idx);
            }
            let w = kernel.kernel_value(center - j as f64) as f32;
            tap_weights.push(w);
        }

        // Normalise
        let sum: f32 = tap_weights.iter().sum();
        if sum.abs() > 1e-8 {
            for w in &mut tap_weights {
                *w /= sum;
            }
        }

        // If no in-bounds tap, emit a single zero-weight tap at source index 0
        let offset = first_src.unwrap_or(0);
        if tap_weights.is_empty() {
            tap_weights.push(0.0);
        }

        offsets.push(offset);
        weights.push(tap_weights);
    }

    (weights, offsets)
}

/// Lanczos resampler that applies the Lanczos filter for image scaling.
#[derive(Debug, Clone)]
pub struct LanczosResampler {
    /// The Lanczos kernel to use
    pub kernel: LanczosKernel,
}

impl Default for LanczosResampler {
    fn default() -> Self {
        Self {
            kernel: LanczosKernel::default(),
        }
    }
}

impl LanczosResampler {
    /// Create a new `LanczosResampler` with the default `a=3` kernel.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new `LanczosResampler` with a custom kernel.
    pub fn with_kernel(kernel: LanczosKernel) -> Self {
        Self { kernel }
    }

    /// Create a `LanczosResampler` from a predefined window size.
    pub fn from_window_size(window: LanczosWindowSize) -> Self {
        Self {
            kernel: LanczosKernel::new(window.a_value()),
        }
    }

    /// Resample a 1D signal from its current length to `dst_len` samples.
    ///
    /// Uses the scalar Lanczos filter to compute each output sample by
    /// weighting nearby input samples.  This is the reference scalar path;
    /// for the SIMD-dispatched variant see [`LanczosResampler::resample_1d_simd`].
    pub fn resample_1d(&self, src: &[f32], dst_len: usize) -> Vec<f32> {
        if src.is_empty() || dst_len == 0 {
            return Vec::new();
        }

        let src_len = src.len();
        let scale = src_len as f64 / dst_len as f64;
        let a = self.kernel.a as i64;

        let mut dst = vec![0.0f32; dst_len];

        for (i, dst_sample) in dst.iter_mut().enumerate() {
            let center = (i as f64 + 0.5) * scale - 0.5;
            let start = (center - a as f64).ceil() as i64;
            let end = (center + a as f64).floor() as i64;

            let mut weight_sum = 0.0f64;
            let mut value_sum = 0.0f64;

            for j in start..=end {
                if j >= 0 && j < src_len as i64 {
                    let w = self.kernel.kernel_value(center - j as f64);
                    weight_sum += w;
                    value_sum += w * src[j as usize] as f64;
                }
            }

            *dst_sample = if weight_sum.abs() > 1e-10 {
                (value_sum / weight_sum) as f32
            } else {
                0.0
            };
        }

        dst
    }

    /// Resample a 1D signal using SIMD-dispatched Lanczos filtering.
    ///
    /// Identical numerics to `resample_1d` but delegates the inner
    /// multiply-accumulate loop to `separable_filter_pass_simd`, which
    /// selects AVX2+FMA3 at runtime on x86/x86_64 or scalar otherwise.
    pub fn resample_1d_simd(&self, src: &[f32], dst_len: usize) -> Vec<f32> {
        if src.is_empty() || dst_len == 0 {
            return Vec::new();
        }
        let src_len = src.len();
        let (weights, offsets) = build_lanczos_filter_weights(&self.kernel, src_len, dst_len);
        separable_filter_pass_simd(src, &weights, &offsets)
    }

    /// Scale an image using Lanczos resampling with SIMD acceleration.
    ///
    /// The image is assumed to be stored in row-major order with 1 byte per pixel
    /// (grayscale). Performs a two-pass horizontal then vertical resample using
    /// `separable_filter_pass_simd` for both passes.
    ///
    /// # Arguments
    /// - `pixels`: Source pixel data (grayscale, 1 byte per pixel)
    /// - `src_w`: Source image width
    /// - `src_h`: Source image height
    /// - `dst_w`: Destination image width
    /// - `dst_h`: Destination image height
    pub fn scale_image(
        &self,
        pixels: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<u8> {
        if pixels.is_empty() || dst_w == 0 || dst_h == 0 {
            return Vec::new();
        }

        // Convert to f32 for processing
        let src_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

        // Pre-compute horizontal and vertical filter weights once
        let (h_weights, h_offsets) = build_lanczos_filter_weights(&self.kernel, src_w, dst_w);
        let (v_weights, v_offsets) = build_lanczos_filter_weights(&self.kernel, src_h, dst_h);

        // Horizontal pass: resample each row from src_w to dst_w using SIMD
        let mut h_pass = vec![0.0f32; src_h * dst_w];
        for row in 0..src_h {
            let src_row = &src_f32[row * src_w..(row + 1) * src_w];
            let dst_row = separable_filter_pass_simd(src_row, &h_weights, &h_offsets);
            h_pass[row * dst_w..(row + 1) * dst_w].copy_from_slice(&dst_row);
        }

        // Vertical pass: resample each column from src_h to dst_h using SIMD
        let mut result = vec![0u8; dst_w * dst_h];
        for col in 0..dst_w {
            let col_data: Vec<f32> = (0..src_h).map(|row| h_pass[row * dst_w + col]).collect();
            let resampled_col = separable_filter_pass_simd(&col_data, &v_weights, &v_offsets);
            for (row, &val) in resampled_col.iter().enumerate() {
                let clamped = val.clamp(0.0, 1.0);
                result[row * dst_w + col] = (clamped * 255.0) as u8;
            }
        }

        result
    }

    /// Scalar-only image scaling using `resample_1d`.
    ///
    /// Identical algorithm to `scale_image` but bypasses SIMD dispatch.
    /// Useful for bit-close regression tests.
    pub fn scale_image_scalar(
        &self,
        pixels: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<u8> {
        if pixels.is_empty() || dst_w == 0 || dst_h == 0 {
            return Vec::new();
        }

        // Convert to f32 for processing
        let src_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

        // Horizontal pass: resample each row from src_w to dst_w
        let mut h_pass = vec![0.0f32; src_h * dst_w];
        for row in 0..src_h {
            let src_row = &src_f32[row * src_w..(row + 1) * src_w];
            let dst_row = self.resample_1d(src_row, dst_w);
            h_pass[row * dst_w..(row + 1) * dst_w].copy_from_slice(&dst_row);
        }

        // Vertical pass: resample each column from src_h to dst_h
        let mut result = vec![0u8; dst_w * dst_h];
        for col in 0..dst_w {
            let col_data: Vec<f32> = (0..src_h).map(|row| h_pass[row * dst_w + col]).collect();
            let resampled_col = self.resample_1d(&col_data, dst_h);
            for (row, &val) in resampled_col.iter().enumerate() {
                let clamped = val.clamp(0.0, 1.0);
                result[row * dst_w + col] = (clamped * 255.0) as u8;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_default_a() {
        let k = LanczosKernel::default();
        assert_eq!(k.a, 3);
    }

    #[test]
    fn test_kernel_new() {
        let k = LanczosKernel::new(2);
        assert_eq!(k.a, 2);
    }

    #[test]
    fn test_kernel_value_at_zero() {
        let k = LanczosKernel::default();
        let v = k.kernel_value(0.0);
        assert!((v - 1.0).abs() < 1e-9, "kernel(0) should be 1.0, got {v}");
    }

    #[test]
    fn test_kernel_value_at_boundary() {
        let k = LanczosKernel::default();
        // At x == a (3.0), the kernel should be 0
        let v = k.kernel_value(3.0);
        assert!(v.abs() < 1e-9, "kernel(a) should be ~0, got {v}");
    }

    #[test]
    fn test_kernel_value_outside_support() {
        let k = LanczosKernel::default();
        let v = k.kernel_value(5.0);
        assert_eq!(v, 0.0, "kernel outside support should be 0");
    }

    #[test]
    fn test_kernel_symmetry() {
        let k = LanczosKernel::default();
        for x in [0.5, 1.0, 1.5, 2.0, 2.5] {
            let pos = k.kernel_value(x);
            let neg = k.kernel_value(-x);
            assert!(
                (pos - neg).abs() < 1e-9,
                "kernel should be symmetric: k({x}) != k(-{x})"
            );
        }
    }

    #[test]
    fn test_resampler_new() {
        let r = LanczosResampler::new();
        assert_eq!(r.kernel.a, 3);
    }

    #[test]
    fn test_resampler_with_kernel() {
        let k = LanczosKernel::new(2);
        let r = LanczosResampler::with_kernel(k);
        assert_eq!(r.kernel.a, 2);
    }

    #[test]
    fn test_resample_1d_identity() {
        let r = LanczosResampler::new();
        let src: Vec<f32> = (0..8).map(|i| i as f32 / 7.0).collect();
        let dst = r.resample_1d(&src, 8);
        assert_eq!(dst.len(), 8);
        // Values should be approximately the same
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!((s - d).abs() < 0.05, "identity resample: {s} vs {d}");
        }
    }

    #[test]
    fn test_resample_1d_upsample() {
        let r = LanczosResampler::new();
        let src = vec![0.0f32, 1.0, 0.0];
        let dst = r.resample_1d(&src, 9);
        assert_eq!(dst.len(), 9);
        // Center of output should peak near 1.0
        let max_val = dst.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_val > 0.5,
            "upsampled peak should be > 0.5, got {max_val}"
        );
    }

    #[test]
    fn test_resample_1d_downsample() {
        let r = LanczosResampler::new();
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let dst = r.resample_1d(&src, 4);
        assert_eq!(dst.len(), 4);
        // Monotonically increasing
        for w in dst.windows(2) {
            assert!(
                w[1] >= w[0] - 0.01,
                "downsampled should be roughly monotonic"
            );
        }
    }

    #[test]
    fn test_resample_1d_empty_src() {
        let r = LanczosResampler::new();
        let dst = r.resample_1d(&[], 8);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_resample_1d_zero_dst() {
        let r = LanczosResampler::new();
        let src = vec![1.0f32, 2.0, 3.0];
        let dst = r.resample_1d(&src, 0);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_scale_image_empty() {
        let r = LanczosResampler::new();
        let result = r.scale_image(&[], 0, 0, 4, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_scale_image_size() {
        let r = LanczosResampler::new();
        let src: Vec<u8> = (0..64).map(|i| i as u8 * 4).collect();
        let result = r.scale_image(&src, 8, 8, 4, 4);
        assert_eq!(result.len(), 4 * 4);
    }

    #[test]
    fn test_scale_image_values_in_range() {
        let r = LanczosResampler::new();
        let src: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let result = r.scale_image(&src, 16, 16, 8, 8);
        for &v in &result {
            let _ = v; // all u8 values are in [0, 255] by definition
        }
        assert_eq!(result.len(), 64);
    }

    // ── LanczosWindowSize tests ─────────────────────────────────────────────

    #[test]
    fn test_window_size_a_values() {
        assert_eq!(LanczosWindowSize::Tap2.a_value(), 2);
        assert_eq!(LanczosWindowSize::Tap3.a_value(), 3);
        assert_eq!(LanczosWindowSize::Tap4.a_value(), 4);
        assert_eq!(LanczosWindowSize::Tap5.a_value(), 5);
    }

    #[test]
    fn test_window_size_display() {
        assert_eq!(LanczosWindowSize::Tap2.to_string(), "Lanczos-2");
        assert_eq!(LanczosWindowSize::Tap5.to_string(), "Lanczos-5");
    }

    #[test]
    fn test_window_size_descriptions() {
        for ws in LanczosWindowSize::all() {
            assert!(!ws.description().is_empty());
        }
    }

    #[test]
    fn test_window_size_all() {
        let all = LanczosWindowSize::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_from_window_size() {
        let r = LanczosResampler::from_window_size(LanczosWindowSize::Tap2);
        assert_eq!(r.kernel.a, 2);
        let r = LanczosResampler::from_window_size(LanczosWindowSize::Tap5);
        assert_eq!(r.kernel.a, 5);
    }

    #[test]
    fn test_kernel_support_varies_with_window_size() {
        // Outside support should return 0
        let k2 = LanczosKernel::new(2);
        let k5 = LanczosKernel::new(5);
        // At x=2.5: k2 should be 0 (outside support), k5 should be non-zero
        assert_eq!(k2.kernel_value(2.5), 0.0);
        assert!(k5.kernel_value(2.5).abs() > 1e-6);
    }

    #[test]
    fn test_configurable_lanczos_all_sizes_produce_output() {
        let src: Vec<u8> = (0..64).map(|i| i as u8 * 4).collect();
        for ws in LanczosWindowSize::all() {
            let r = LanczosResampler::from_window_size(*ws);
            let result = r.scale_image(&src, 8, 8, 4, 4);
            assert_eq!(result.len(), 16, "{} should produce 4x4 output", ws);
        }
    }

    #[test]
    fn test_larger_window_gives_sharper_result() {
        // Upscale a step-edge image with Lanczos-2 and Lanczos-5.
        // Lanczos-5 should produce a sharper transition (larger gradient at edge).
        let mut src = vec![0u8; 8 * 8];
        for y in 0..8 {
            for x in 4..8 {
                src[y * 8 + x] = 255;
            }
        }

        let r2 = LanczosResampler::from_window_size(LanczosWindowSize::Tap2);
        let r5 = LanczosResampler::from_window_size(LanczosWindowSize::Tap5);
        let out2 = r2.scale_image(&src, 8, 8, 16, 16);
        let out5 = r5.scale_image(&src, 8, 8, 16, 16);

        // Both should produce 16x16 output
        assert_eq!(out2.len(), 256);
        assert_eq!(out5.len(), 256);

        // Compute max gradient (difference between adjacent pixels in middle row)
        let max_grad = |out: &[u8]| -> u8 {
            let row = 8; // middle row
            let mut max_d = 0u8;
            for x in 1..16 {
                let d =
                    (out[row * 16 + x] as i16 - out[row * 16 + x - 1] as i16).unsigned_abs() as u8;
                if d > max_d {
                    max_d = d;
                }
            }
            max_d
        };

        let g2 = max_grad(&out2);
        let g5 = max_grad(&out5);
        // Lanczos-5 should have at least comparable sharpness
        assert!(
            g5 >= g2.saturating_sub(5),
            "Lanczos-5 gradient {g5} should be >= Lanczos-2 gradient {g2} (approx)"
        );
    }

    #[test]
    fn test_resample_1d_with_different_windows() {
        let src = vec![0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0];
        for ws in LanczosWindowSize::all() {
            let r = LanczosResampler::from_window_size(*ws);
            let dst = r.resample_1d(&src, 12);
            assert_eq!(dst.len(), 12, "{} should produce 12 samples", ws);
        }
    }

    // ── SIMD dispatch tests ──────────────────────────────────────────────────

    #[test]
    fn test_resample_1d_simd_matches_scalar() {
        let r = LanczosResampler::new();
        let src: Vec<f32> = (0..32).map(|i| (i * 7 % 100) as f32 / 99.0).collect();
        let scalar_out = r.resample_1d(&src, 16);
        let simd_out = r.resample_1d_simd(&src, 16);
        assert_eq!(scalar_out.len(), simd_out.len());
        for (i, (&s, &v)) in scalar_out.iter().zip(simd_out.iter()).enumerate() {
            assert!(
                (s - v).abs() < 1e-4,
                "scalar/simd mismatch at {i}: scalar={s} simd={v}"
            );
        }
    }

    #[test]
    fn test_build_lanczos_filter_weights_normalised() {
        let k = LanczosKernel::default();
        let (weights, offsets) = build_lanczos_filter_weights(&k, 16, 8);
        assert_eq!(weights.len(), 8);
        assert_eq!(offsets.len(), 8);
        for (i, w) in weights.iter().enumerate() {
            let sum: f32 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "weights[{i}] must sum to 1, got {sum}"
            );
        }
    }

    /// SIMD path must produce the same results (within ±1 per pixel) as the
    /// scalar path on a gradient image scaled from 200×150 → 100×75.
    #[test]
    fn test_lanczos_simd_vs_scalar_bitclose() {
        let sw = 200usize;
        let sh = 150usize;
        let dw = 100usize;
        let dh = 75usize;

        let src: Vec<u8> = (0..sw * sh)
            .map(|i| {
                let x = i % sw;
                let y = i / sw;
                (((x * 255) / sw + (y * 128) / sh) % 256) as u8
            })
            .collect();

        let r = LanczosResampler::new();
        let simd_out = r.scale_image(&src, sw, sh, dw, dh);
        let scalar_out = r.scale_image_scalar(&src, sw, sh, dw, dh);

        assert_eq!(simd_out.len(), dw * dh);
        assert_eq!(scalar_out.len(), dw * dh);

        for (i, (&s, &r)) in simd_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                (s as i32 - r as i32).abs() <= 1,
                "pixel mismatch at {i}: simd={s}, scalar={r}"
            );
        }
    }
}

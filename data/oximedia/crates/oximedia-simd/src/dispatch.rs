//! Runtime-dispatched SIMD wrappers for core media processing operations.
//!
//! Each function in this module selects the fastest available SIMD
//! implementation at runtime (AVX-512 → scalar fallback) without exposing
//! `unsafe` in the public API.
//!
//! # Design
//!
//! - Public functions are fully safe.
//! - `unsafe` blocks appear only inside platform-specific `#[cfg]` branches,
//!   always guarded by a `is_x86_feature_detected!` (or equivalent) check.
//! - Scalar fallbacks are pure Rust in [`crate::avx512`].

#![deny(unsafe_op_in_unsafe_fn)]

// ── BGRA → RGBA ─────────────────────────────────────────────────────────────

/// Runtime-dispatched BGRA to RGBA byte-order conversion.
///
/// On x86-64 machines with AVX-512BW the conversion uses a 512-bit shuffle
/// that processes 16 pixels (64 bytes) per clock cycle.  On all other
/// platforms the scalar fallback is used.
///
/// `src` and `dst` must each be a multiple of 4 bytes (one BGRA / RGBA pixel
/// per 4 bytes).  If `src.len() != dst.len()`, only `min(src.len(), dst.len())`
/// bytes (rounded down to the nearest pixel) are processed.
pub fn bgra_to_rgba(src: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512f and avx512bw features confirmed present above.
            unsafe { crate::avx512::bgra_to_rgba_avx512(src, dst) };
            return;
        }
    }
    crate::avx512::bgra_to_rgba_scalar(src, dst);
}

// ── Horizontal sum of f32 ────────────────────────────────────────────────────

/// Runtime-dispatched horizontal sum of an f32 slice.
///
/// Returns the sum of all elements in `data`, or `0.0` if `data` is empty.
/// On x86-64 with AVX-512F the reduction is performed over 512-bit registers;
/// otherwise a simple scalar loop is used.
pub fn hsum_f32(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f feature confirmed present above.
            return unsafe { crate::avx512::hsum_f32_avx512(data) };
        }
    }
    crate::avx512::hsum_f32_scalar(data)
}

// ── Scale i16 samples ────────────────────────────────────────────────────────

/// Runtime-dispatched scaling of i16 audio samples by a f32 gain factor.
///
/// Each sample is multiplied by `gain`, clamped to `[−32768, 32767]`, and
/// written back in place.  The AVX-512 path processes 16 samples per
/// iteration via sign-extension, float multiply, and saturation.
pub fn scale_i16(samples: &mut [i16], gain: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512f and avx512bw features confirmed present above.
            unsafe { crate::avx512::scale_i16_avx512(samples, gain) };
            return;
        }
    }
    crate::avx512::scale_i16_scalar(samples, gain);
}

// ── Dot product of f32 slices ────────────────────────────────────────────────

/// Runtime-dispatched f32 dot product.
///
/// Returns `∑ a[i] × b[i]` over the shorter of the two slices.  Returns
/// `0.0` if either slice is empty.  On AVX-512 hardware the computation uses
/// fused multiply-add (`_mm512_fmadd_ps`) which may yield slightly different
/// rounding than the scalar path.
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f feature confirmed present above.
            return unsafe { crate::avx512::dot_product_f32_avx512(a, b) };
        }
    }
    crate::avx512::dot_product_f32_scalar(a, b)
}

// ── Clamp f32 to [0.0, 1.0] ─────────────────────────────────────────────────

/// Runtime-dispatched f32 clamp to `[0.0, 1.0]` in place.
///
/// Every element of `data` is clamped to the closed interval `[0.0, 1.0]`.
/// The AVX-512 path processes 16 elements per cycle using `_mm512_min_ps` /
/// `_mm512_max_ps`.
pub fn clamp_f32(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f feature confirmed present above.
            unsafe { crate::avx512::clamp_f32_avx512(data) };
            return;
        }
    }
    crate::avx512::clamp_f32_scalar(data);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn test_dispatch_bgra_to_rgba_roundtrip() {
        // After two swaps we should get the original data back.
        let original = [10u8, 20, 30, 255, 40, 50, 60, 128];
        let mut step1 = [0u8; 8];
        let mut step2 = [0u8; 8];
        bgra_to_rgba(&original, &mut step1);
        bgra_to_rgba(&step1, &mut step2); // swap again
        assert_eq!(step2, original);
    }

    #[test]
    fn test_dispatch_bgra_to_rgba_empty() {
        let src: [u8; 0] = [];
        let mut dst: [u8; 0] = [];
        bgra_to_rgba(&src, &mut dst); // must not panic
    }

    #[test]
    fn test_dispatch_hsum_f32_known() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        assert!(approx_eq(hsum_f32(&data), 10.0));
    }

    #[test]
    fn test_dispatch_hsum_f32_empty() {
        assert!(approx_eq(hsum_f32(&[]), 0.0));
    }

    #[test]
    fn test_dispatch_scale_i16_half() {
        let mut samples = [100i16, 200, -300, -400];
        scale_i16(&mut samples, 0.5);
        assert_eq!(samples, [50, 100, -150, -200]);
    }

    #[test]
    fn test_dispatch_scale_i16_empty() {
        let mut empty: [i16; 0] = [];
        scale_i16(&mut empty, 2.0); // must not panic
    }

    #[test]
    fn test_dispatch_dot_product_f32_known() {
        let a = [3.0f32, 4.0];
        let b = [3.0f32, 4.0];
        // 9 + 16 = 25
        assert!(approx_eq(dot_product_f32(&a, &b), 25.0));
    }

    #[test]
    fn test_dispatch_dot_product_f32_empty() {
        assert!(approx_eq(dot_product_f32(&[], &[]), 0.0));
    }

    #[test]
    fn test_dispatch_clamp_f32_bounds() {
        let mut data = [-5.0f32, 0.0, 0.5, 1.0, 3.0];
        clamp_f32(&mut data);
        for &v in &data {
            assert!(v >= 0.0 && v <= 1.0, "value {v} out of [0,1]");
        }
    }

    #[test]
    fn test_dispatch_clamp_f32_empty() {
        let mut data: Vec<f32> = Vec::new();
        clamp_f32(&mut data); // must not panic
    }
}

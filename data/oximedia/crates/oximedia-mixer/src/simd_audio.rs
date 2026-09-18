//! SIMD-accelerated audio operations for mixer bus processing.
//!
//! Provides hardware-accelerated implementations of the core mixer hot paths:
//! - **Sample accumulation** (`add_samples_simd`) — summing audio buffers
//! - **Gain application** (`apply_gain_simd`) — multiplying a buffer by a scalar
//! - **Mix and gain** (`mix_and_gain_simd`) — accumulate + gain in one pass
//!
//! On x86/x86_64, uses AVX (256-bit) or SSE2 (128-bit) via
//! `is_x86_feature_detected!` runtime dispatch.  Falls back to scalar on
//! other architectures.

// ── Sample accumulation (dst[i] += src[i]) ───────────────────────────────────

/// Add `src` samples into `dst` buffer element-wise (`dst[i] += src[i]`).
///
/// If `dst` is shorter than `src`, only the first `dst.len()` elements are
/// modified.  If `src` is shorter, only the first `src.len()` elements of
/// `dst` are touched.
pub fn add_samples_simd(dst: &mut [f32], src: &[f32]) {
    let len = dst.len().min(src.len());
    if len == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            // SAFETY: feature is checked at runtime.
            unsafe { add_samples_avx(&mut dst[..len], &src[..len]) }
            return;
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: feature is checked at runtime.
            unsafe { add_samples_sse2(&mut dst[..len], &src[..len]) }
            return;
        }
    }
    add_samples_scalar(&mut dst[..len], &src[..len]);
}

/// Apply a scalar gain to `src` and write the result to `dst`.
///
/// `dst` must be at least as long as `src`.  Only the first `src.len()`
/// elements of `dst` are written.
///
/// # Panics
///
/// Panics if `dst.len() < src.len()`.
pub fn apply_gain_simd(src: &[f32], dst: &mut [f32], gain: f32) {
    assert!(
        dst.len() >= src.len(),
        "apply_gain_simd: dst too short ({} < {})",
        dst.len(),
        src.len()
    );
    if src.is_empty() {
        return;
    }
    let len = src.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            // SAFETY: feature is checked at runtime.
            unsafe { apply_gain_avx(src, &mut dst[..len], gain) }
            return;
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: feature is checked at runtime.
            unsafe { apply_gain_sse2(src, &mut dst[..len], gain) }
            return;
        }
    }
    apply_gain_scalar(src, &mut dst[..len], gain);
}

/// Accumulate `src * gain` into `dst`: `dst[i] += src[i] * gain`.
///
/// Processes `min(dst.len(), src.len())` elements.
pub fn mix_and_gain_simd(dst: &mut [f32], src: &[f32], gain: f32) {
    let len = dst.len().min(src.len());
    if len == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            // SAFETY: feature is checked at runtime.
            unsafe { mix_and_gain_avx(&mut dst[..len], &src[..len], gain) }
            return;
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: feature is checked at runtime.
            unsafe { mix_and_gain_sse2(&mut dst[..len], &src[..len], gain) }
            return;
        }
    }
    mix_and_gain_scalar(&mut dst[..len], &src[..len], gain);
}

// ── AVX implementations (x86_64 only) ────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_samples_avx(dst: &mut [f32], src: &[f32]) {
    use std::arch::x86_64::*;
    let len = dst.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= len, pointers are valid.
        let a = _mm256_loadu_ps(dst_ptr.add(offset));
        let b = _mm256_loadu_ps(src_ptr.add(offset));
        let result = _mm256_add_ps(a, b);
        _mm256_storeu_ps(dst_ptr.add(offset), result);
    }

    // Handle remainder with scalar.
    let rem_start = chunks * 8;
    for i in rem_start..rem_start + remainder {
        dst[i] += src[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn apply_gain_avx(src: &[f32], dst: &mut [f32], gain: f32) {
    use std::arch::x86_64::*;
    let len = src.len();
    let chunks = len / 8;
    let remainder = len % 8;
    let gain_v = _mm256_set1_ps(gain);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let a = _mm256_loadu_ps(src_ptr.add(offset));
        let result = _mm256_mul_ps(a, gain_v);
        _mm256_storeu_ps(dst_ptr.add(offset), result);
    }

    let rem_start = chunks * 8;
    for i in rem_start..rem_start + remainder {
        dst[i] = src[i] * gain;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn mix_and_gain_avx(dst: &mut [f32], src: &[f32], gain: f32) {
    use std::arch::x86_64::*;
    let len = dst.len();
    let chunks = len / 8;
    let remainder = len % 8;
    let gain_v = _mm256_set1_ps(gain);

    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let d = _mm256_loadu_ps(dst_ptr.add(offset));
        let s = _mm256_loadu_ps(src_ptr.add(offset));
        // dst + src * gain
        let result = _mm256_fmadd_ps(s, gain_v, d);
        _mm256_storeu_ps(dst_ptr.add(offset), result);
    }

    let rem_start = chunks * 8;
    for i in rem_start..rem_start + remainder {
        dst[i] += src[i] * gain;
    }
}

// ── SSE2 implementations (x86_64 only) ───────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn add_samples_sse2(dst: &mut [f32], src: &[f32]) {
    use std::arch::x86_64::*;
    let len = dst.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let a = _mm_loadu_ps(dst_ptr.add(offset));
        let b = _mm_loadu_ps(src_ptr.add(offset));
        let result = _mm_add_ps(a, b);
        _mm_storeu_ps(dst_ptr.add(offset), result);
    }

    let rem_start = chunks * 4;
    for i in rem_start..rem_start + remainder {
        dst[i] += src[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn apply_gain_sse2(src: &[f32], dst: &mut [f32], gain: f32) {
    use std::arch::x86_64::*;
    let len = src.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let gain_v = _mm_set1_ps(gain);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let a = _mm_loadu_ps(src_ptr.add(offset));
        let result = _mm_mul_ps(a, gain_v);
        _mm_storeu_ps(dst_ptr.add(offset), result);
    }

    let rem_start = chunks * 4;
    for i in rem_start..rem_start + remainder {
        dst[i] = src[i] * gain;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn mix_and_gain_sse2(dst: &mut [f32], src: &[f32], gain: f32) {
    use std::arch::x86_64::*;
    let len = dst.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let gain_v = _mm_set1_ps(gain);

    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let d = _mm_loadu_ps(dst_ptr.add(offset));
        let s = _mm_loadu_ps(src_ptr.add(offset));
        let sg = _mm_mul_ps(s, gain_v);
        let result = _mm_add_ps(d, sg);
        _mm_storeu_ps(dst_ptr.add(offset), result);
    }

    let rem_start = chunks * 4;
    for i in rem_start..rem_start + remainder {
        dst[i] += src[i] * gain;
    }
}

// ── Scalar fallback ───────────────────────────────────────────────────────────

fn add_samples_scalar(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

fn apply_gain_scalar(src: &[f32], dst: &mut [f32], gain: f32) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s * gain;
    }
}

fn mix_and_gain_scalar(dst: &mut [f32], src: &[f32], gain: f32) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s * gain;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── add_samples_simd ─────────────────────────────────────────────────────

    #[test]
    fn test_add_samples_simple() {
        let mut dst = vec![1.0_f32; 8];
        let src = vec![0.5_f32; 8];
        add_samples_simd(&mut dst, &src);
        for &v in &dst {
            assert!((v - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_add_samples_non_aligned() {
        // 13 elements: tests remainder handling
        let mut dst = vec![1.0_f32; 13];
        let src = vec![2.0_f32; 13];
        add_samples_simd(&mut dst, &src);
        for &v in &dst {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_add_samples_simd_matches_scalar() {
        let mut dst_simd: Vec<f32> = (0..67).map(|i| (i as f32) * 0.1).collect();
        let mut dst_scalar = dst_simd.clone();
        let src: Vec<f32> = (0..67).map(|i| (i as f32) * 0.3 + 1.0).collect();

        add_samples_simd(&mut dst_simd, &src);
        add_samples_scalar(&mut dst_scalar, &src);

        for (i, (&s, &r)) in dst_simd.iter().zip(dst_scalar.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-5,
                "mismatch at {i}: simd={s}, scalar={r}"
            );
        }
    }

    #[test]
    fn test_add_samples_empty() {
        let mut dst = vec![1.0_f32; 4];
        add_samples_simd(&mut dst, &[]);
        assert!((dst[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_samples_different_lengths() {
        let mut dst = vec![0.0_f32; 4];
        let src = vec![1.0_f32; 8]; // longer than dst
        add_samples_simd(&mut dst, &src);
        // Only first 4 elements should be touched
        for &v in &dst {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    // ── apply_gain_simd ──────────────────────────────────────────────────────

    #[test]
    fn test_apply_gain_simple() {
        let src = vec![2.0_f32; 8];
        let mut dst = vec![0.0_f32; 8];
        apply_gain_simd(&src, &mut dst, 0.5);
        for &v in &dst {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_gain_non_aligned() {
        let src: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let mut dst = vec![0.0_f32; 19];
        apply_gain_simd(&src, &mut dst, 2.0);
        for (i, &v) in dst.iter().enumerate() {
            assert!(
                (v - (i as f32) * 2.0).abs() < 1e-5,
                "mismatch at {i}: got {v}"
            );
        }
    }

    #[test]
    fn test_apply_gain_simd_matches_scalar() {
        let src: Vec<f32> = (0..67).map(|i| (i * 3 % 100) as f32 * 0.01).collect();
        let mut dst_simd = vec![0.0_f32; 67];
        let mut dst_scalar = vec![0.0_f32; 67];

        apply_gain_simd(&src, &mut dst_simd, 0.75);
        apply_gain_scalar(&src, &mut dst_scalar, 0.75);

        for (i, (&s, &r)) in dst_simd.iter().zip(dst_scalar.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-5,
                "mismatch at {i}: simd={s}, scalar={r}"
            );
        }
    }

    // ── mix_and_gain_simd ────────────────────────────────────────────────────

    #[test]
    fn test_mix_and_gain_simple() {
        let mut dst = vec![1.0_f32; 8];
        let src = vec![2.0_f32; 8];
        mix_and_gain_simd(&mut dst, &src, 0.5);
        for &v in &dst {
            // 1.0 + 2.0 * 0.5 = 2.0
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mix_and_gain_non_aligned() {
        let mut dst = vec![0.5_f32; 15];
        let src = vec![1.0_f32; 15];
        mix_and_gain_simd(&mut dst, &src, 3.0);
        for &v in &dst {
            // 0.5 + 1.0 * 3.0 = 3.5
            assert!((v - 3.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_mix_and_gain_simd_matches_scalar() {
        let mut dst_simd: Vec<f32> = (0..67).map(|i| (i as f32) * 0.1).collect();
        let mut dst_scalar = dst_simd.clone();
        let src: Vec<f32> = (0..67).map(|i| (i as f32) * 0.2 + 0.5).collect();

        mix_and_gain_simd(&mut dst_simd, &src, 0.8);
        mix_and_gain_scalar(&mut dst_scalar, &src, 0.8);

        for (i, (&s, &r)) in dst_simd.iter().zip(dst_scalar.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-4,
                "mismatch at {i}: simd={s}, scalar={r}"
            );
        }
    }

    #[test]
    fn test_mix_and_gain_empty() {
        let mut dst = vec![1.0_f32; 4];
        mix_and_gain_simd(&mut dst, &[], 2.0);
        // Nothing should change
        for &v in &dst {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }
}

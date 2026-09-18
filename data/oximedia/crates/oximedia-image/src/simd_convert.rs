//! SIMD-accelerated pixel format conversion between u8 and f32.
//!
//! Provides `u8 → f32 [0.0, 1.0]` and `f32 [0.0, 1.0] → u8` conversions
//! using platform-specific SIMD intrinsics where available:
//!
//! - **AVX2** on x86_64 (256-bit, 32 u8 or 8 f32 per register)
//! - **NEON** on aarch64 (128-bit; always available, no runtime check needed)
//! - **Scalar** fallback for all other targets
//!
//! # Panics
//!
//! `u8_to_f32_normalized` and `f32_to_u8_saturated` panic if `src.len() != dst.len()`.

// SIMD intrinsics require unsafe; this module's public API is fully safe.
#![allow(unsafe_code)]

/// Convert u8 pixel values to f32 normalized in `[0.0, 1.0]`.
///
/// `src.len()` must equal `dst.len()`; panics otherwise.
pub fn u8_to_f32_normalized(src: &[u8], dst: &mut [f32]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "u8_to_f32_normalized: src and dst must have equal length"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: we just detected AVX2 support at runtime.
            unsafe { u8_to_f32_avx2(src, dst) }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always present on aarch64 targets we care about.
        // Safety: neon is a mandatory baseline on aarch64-apple-darwin and
        // aarch64-unknown-linux-gnu.
        unsafe { u8_to_f32_neon(src, dst) }
        return;
    }

    #[cfg_attr(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        allow(unreachable_code)
    )]
    u8_to_f32_scalar(src, dst);
}

/// Convert f32 values in `[0.0, 1.0]` to u8, saturating at 0 and 255.
///
/// Values below 0.0 clamp to 0; values above 1.0 clamp to 255.
/// `src.len()` must equal `dst.len()`; panics otherwise.
pub fn f32_to_u8_saturated(src: &[f32], dst: &mut [u8]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "f32_to_u8_saturated: src and dst must have equal length"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { f32_to_u8_avx2(src, dst) }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { f32_to_u8_neon(src, dst) }
        return;
    }

    #[cfg_attr(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        allow(unreachable_code)
    )]
    f32_to_u8_scalar(src, dst);
}

// ---------------------------------------------------------------------------
// Scalar implementations
// ---------------------------------------------------------------------------

fn u8_to_f32_scalar(src: &[u8], dst: &mut [f32]) {
    const SCALE: f32 = 1.0 / 255.0;
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = (*s as f32) * SCALE;
    }
}

fn f32_to_u8_scalar(src: &[f32], dst: &mut [u8]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let v = (*s * 255.0 + 0.5).floor();
        *d = if v <= 0.0 {
            0
        } else if v >= 255.0 {
            255
        } else {
            v as u8
        };
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn u8_to_f32_avx2(src: &[u8], dst: &mut [f32]) {
    use std::arch::x86_64::*;

    const SCALE: f32 = 1.0 / 255.0;
    let scale = _mm256_set1_ps(SCALE);

    let mut i = 0usize;
    let n = src.len();

    // Process 8 u8 elements at a time using SSE4.1 + AVX2.
    // _mm_cvtepu8_epi32 widens 4 u8 → 4 i32 (128-bit).
    // _mm256_cvtepu8_epi32 widens 8 u8 → 8 i32 (256-bit).
    while i + 8 <= n {
        // Load 8 bytes
        let bytes = _mm_loadl_epi64(src.as_ptr().add(i) as *const __m128i);
        // Zero-extend u8 → i32 (256-bit register, 8 ints)
        let ints = _mm256_cvtepu8_epi32(bytes);
        // Convert i32 → f32
        let floats = _mm256_cvtepi32_ps(ints);
        // Scale by 1/255
        let scaled = _mm256_mul_ps(floats, scale);
        // Store
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), scaled);
        i += 8;
    }

    // Scalar tail
    u8_to_f32_scalar(&src[i..], &mut dst[i..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn f32_to_u8_avx2(src: &[f32], dst: &mut [u8]) {
    use std::arch::x86_64::*;

    let scale = _mm256_set1_ps(255.0f32);
    let half = _mm256_set1_ps(0.5f32);
    let zero = _mm256_setzero_ps();
    let max_val = _mm256_set1_ps(255.0f32);

    let mut i = 0usize;
    let n = src.len();

    while i + 8 <= n {
        let floats = _mm256_loadu_ps(src.as_ptr().add(i));
        // Multiply by 255, add 0.5 for rounding
        let scaled = _mm256_add_ps(_mm256_mul_ps(floats, scale), half);
        // Clamp to [0, 255]
        let clamped = _mm256_min_ps(_mm256_max_ps(scaled, zero), max_val);
        // Convert to i32 (truncate toward zero after rounding is already done)
        let ints = _mm256_cvttps_epi32(clamped);
        // Pack i32 → i16 × 16 (with signed saturation)
        // We need to get 8 ints → 8 bytes.  Use packs twice.
        // First: pack the lower 128 bits and upper 128 bits to 16 × i16
        let lo = _mm256_extracti128_si256::<0>(ints);
        let hi = _mm256_extracti128_si256::<1>(ints);
        let packed16 = _mm_packs_epi32(lo, hi); // 8 × i16
                                                // Then pack i16 → u8 (unsigned saturate)
        let packed8 = _mm_packus_epi16(packed16, packed16); // 8 bytes in low 64 bits
                                                            // Store 8 bytes
        let out_ptr = dst.as_mut_ptr().add(i) as *mut i64;
        out_ptr.write_unaligned(_mm_cvtsi128_si64(packed8));
        i += 8;
    }

    // Scalar tail
    f32_to_u8_scalar(&src[i..], &mut dst[i..]);
}

// ---------------------------------------------------------------------------
// NEON (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn u8_to_f32_neon(src: &[u8], dst: &mut [f32]) {
    use std::arch::aarch64::*;

    const SCALE: f32 = 1.0 / 255.0;

    let mut i = 0usize;
    let n = src.len();

    // Process 8 u8 elements at a time.
    while i + 8 <= n {
        // Load 8 u8
        let bytes = vld1_u8(src.as_ptr().add(i));
        // Widen u8 × 8 → u16 × 8
        let wide16 = vmovl_u8(bytes);
        // Widen u16 × 8 → u32 × 4 (low half) and u32 × 4 (high half)
        let lo32 = vmovl_u16(vget_low_u16(wide16));
        let hi32 = vmovl_u16(vget_high_u16(wide16));
        // u32 → f32
        let lo_f = vcvtq_f32_u32(lo32);
        let hi_f = vcvtq_f32_u32(hi32);
        // Scale by 1/255
        let lo_scaled = vmulq_n_f32(lo_f, SCALE);
        let hi_scaled = vmulq_n_f32(hi_f, SCALE);
        // Store
        vst1q_f32(dst.as_mut_ptr().add(i), lo_scaled);
        vst1q_f32(dst.as_mut_ptr().add(i + 4), hi_scaled);
        i += 8;
    }

    // Scalar tail
    u8_to_f32_scalar(&src[i..], &mut dst[i..]);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn f32_to_u8_neon(src: &[f32], dst: &mut [u8]) {
    use std::arch::aarch64::*;

    let mut i = 0usize;
    let n = src.len();

    while i + 8 <= n {
        let lo_f = vld1q_f32(src.as_ptr().add(i));
        let hi_f = vld1q_f32(src.as_ptr().add(i + 4));
        // Scale by 255 and add 0.5 for rounding
        let lo_scaled = vaddq_f32(vmulq_n_f32(lo_f, 255.0f32), vdupq_n_f32(0.5f32));
        let hi_scaled = vaddq_f32(vmulq_n_f32(hi_f, 255.0f32), vdupq_n_f32(0.5f32));
        // f32 → u32 (truncate; rounding already applied via +0.5)
        let lo_u32 = vcvtq_u32_f32(lo_scaled);
        let hi_u32 = vcvtq_u32_f32(hi_scaled);
        // Clamp to [0, 255]: vcvtq_u32_f32 saturates negative → 0, and
        // values ≥ 256.0 saturate to large u32; we narrow with vqmovn (saturate).
        let lo_u16 = vqmovn_u32(lo_u32);
        let hi_u16 = vqmovn_u32(hi_u32);
        let combined_u16 = vcombine_u16(lo_u16, hi_u16);
        let bytes = vqmovn_u16(combined_u16);
        vst1_u8(dst.as_mut_ptr().add(i), bytes);
        i += 8;
    }

    f32_to_u8_scalar(&src[i..], &mut dst[i..]);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn u8_to_f32_scalar_ref(src: &[u8]) -> Vec<f32> {
        src.iter().map(|&v| v as f32 / 255.0).collect()
    }

    fn f32_to_u8_scalar_ref(src: &[f32]) -> Vec<u8> {
        src.iter()
            .map(|&v| {
                let scaled = v * 255.0 + 0.5;
                if scaled <= 0.0 {
                    0
                } else if scaled >= 255.5 {
                    255
                } else {
                    scaled.floor() as u8
                }
            })
            .collect()
    }

    /// Deterministic LCG for reproducible tests without external deps.
    fn make_random_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) & 0xFF) as u8
            })
            .collect()
    }

    #[test]
    fn test_simd_u8_to_f32_matches_scalar() {
        let src = make_random_bytes(0xDEAD_BEEF_0101, 4096);
        let reference = u8_to_f32_scalar_ref(&src);

        let mut dst = vec![0.0f32; src.len()];
        u8_to_f32_normalized(&src, &mut dst);

        for (i, (&expected, &actual)) in reference.iter().zip(dst.iter()).enumerate() {
            let diff = (expected - actual).abs();
            assert!(
                diff < 1e-6,
                "u8_to_f32 mismatch at index {i}: expected {expected}, got {actual}, diff {diff}"
            );
        }
    }

    #[test]
    fn test_simd_f32_to_u8_saturates() {
        // Values above 1.0 must saturate to 255.
        let src_over = [1.5f32, 2.0, f32::INFINITY];
        let mut dst = vec![0u8; src_over.len()];
        f32_to_u8_saturated(&src_over, &mut dst);
        assert_eq!(dst[0], 255, "1.5 must saturate to 255");
        assert_eq!(dst[1], 255, "2.0 must saturate to 255");
        assert_eq!(dst[2], 255, "infinity must saturate to 255");

        // Values below 0.0 must clamp to 0.
        let src_under = [-0.1f32, -1.0, f32::NEG_INFINITY];
        let mut dst2 = vec![255u8; src_under.len()];
        f32_to_u8_saturated(&src_under, &mut dst2);
        assert_eq!(dst2[0], 0, "-0.1 must clamp to 0");
        assert_eq!(dst2[1], 0, "-1.0 must clamp to 0");
        assert_eq!(dst2[2], 0, "-infinity must clamp to 0");

        // 0.5 must round to 127 or 128 (implementation-defined rounding).
        let src_half = [0.5f32];
        let mut dst3 = vec![0u8; 1];
        f32_to_u8_saturated(&src_half, &mut dst3);
        assert!(
            dst3[0] == 127 || dst3[0] == 128,
            "0.5 must map to 127 or 128, got {}",
            dst3[0]
        );
    }

    #[test]
    fn test_simd_f32_to_u8_round_trip() {
        // Round-trip: u8 → f32 → u8 must be identity.
        let src: Vec<u8> = (0u8..=255).collect();
        let mut f32_buf = vec![0.0f32; src.len()];
        u8_to_f32_normalized(&src, &mut f32_buf);

        let mut u8_buf = vec![0u8; src.len()];
        f32_to_u8_saturated(&f32_buf, &mut u8_buf);

        for (i, (&orig, &roundtrip)) in src.iter().zip(u8_buf.iter()).enumerate() {
            assert_eq!(
                orig, roundtrip,
                "Round-trip mismatch at u8 value {orig} (index {i}): got {roundtrip}"
            );
        }
    }

    #[test]
    fn test_u8_to_f32_boundary_values() {
        let src = [0u8, 128, 255];
        let mut dst = [0.0f32; 3];
        u8_to_f32_normalized(&src, &mut dst);
        assert!((dst[0] - 0.0).abs() < 1e-7, "0 must map to 0.0");
        assert!(
            (dst[1] - 128.0 / 255.0).abs() < 1e-6,
            "128 must map to ~0.502"
        );
        assert!((dst[2] - 1.0).abs() < 1e-7, "255 must map to 1.0");
    }

    #[test]
    fn test_simd_f32_to_u8_random_matches_scalar() {
        // Build random f32 values in [0,1] and compare SIMD vs scalar.
        let n = 4096;
        let raw_bytes = make_random_bytes(0xFEED_FACE_ABCD, n);
        let floats: Vec<f32> = raw_bytes.iter().map(|&b| b as f32 / 255.0).collect();

        let scalar_out = f32_to_u8_scalar_ref(&floats);
        let mut simd_out = vec![0u8; n];
        f32_to_u8_saturated(&floats, &mut simd_out);

        for (i, (&s, &v)) in scalar_out.iter().zip(simd_out.iter()).enumerate() {
            // Allow off-by-one due to rounding differences between scalar and SIMD.
            assert!(
                (s as i32 - v as i32).abs() <= 1,
                "f32_to_u8 mismatch at index {i}: scalar={s}, simd={v}"
            );
        }
    }
}

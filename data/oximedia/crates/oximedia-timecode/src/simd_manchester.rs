//! SIMD-accelerated Manchester encoding and decoding for LTC bitstreams.
//!
//! Manchester coding maps each data bit to two half-bit audio periods:
//! - Bit `0`: **high** half-period followed by **low** half-period
//! - Bit `1`: **low** half-period followed by **high** half-period
//!
//! The audio samples are `i16`: `HIGH_LEVEL` and `LOW_LEVEL`.
//!
//! # SIMD strategy
//! On `x86_64` targets with the `avx2` CPU feature available at runtime,
//! [`manchester_encode_simd`] processes 32 input bytes (= 256 bits) per
//! iteration using `_mm256_movemask_epi8` to extract the sign bit of each
//! byte, then expands each bit into two `i16` samples.  All other targets (and
//! AVX2-capable targets without the feature at runtime) fall back to a pure
//! scalar path that produces identical output.

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#![allow(unsafe_code)] // AVX2 intrinsics require unsafe blocks

/// PCM level for the high half-period.
pub const HIGH_LEVEL: i16 = i16::MAX;
/// PCM level for the low half-period.
pub const LOW_LEVEL: i16 = i16::MIN;

// ── Encoding ──────────────────────────────────────────────────────────────────

/// Encode `bits` using Manchester coding, returning PCM audio samples.
///
/// Each byte in `bits` is treated as a single data bit (non-zero ⇒ `1`,
/// zero ⇒ `0`).  The returned `Vec<i16>` has exactly `bits.len() × 2`
/// elements: two samples (high–low or low–high) per input bit.
///
/// On `x86_64` CPUs that report `avx2` at runtime the AVX2 fast-path is
/// taken; otherwise the scalar path is used.  Both paths produce identical
/// output.
pub fn manchester_encode_simd(bits: &[u8]) -> Vec<i16> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return manchester_encode_avx2(bits);
        }
    }
    manchester_encode_scalar(bits)
}

/// Scalar Manchester encoding — reference implementation.
///
/// Exposed publicly so callers can force the scalar path in tests.
pub fn manchester_encode_scalar(bits: &[u8]) -> Vec<i16> {
    let mut out = Vec::with_capacity(bits.len() * 2);
    for &b in bits {
        if b == 0 {
            // Bit 0: high then low
            out.push(HIGH_LEVEL);
            out.push(LOW_LEVEL);
        } else {
            // Bit 1: low then high
            out.push(LOW_LEVEL);
            out.push(HIGH_LEVEL);
        }
    }
    out
}

/// AVX2 Manchester encoding path — processes 32 bytes per loop iteration.
///
/// Only compiled and called on `x86_64` targets at runtime when AVX2 is
/// available.  On other targets the function is not compiled at all.
#[cfg(target_arch = "x86_64")]
fn manchester_encode_avx2(bits: &[u8]) -> Vec<i16> {
    use std::arch::x86_64::*;

    let mut out = Vec::with_capacity(bits.len() * 2);

    // Process 32 bytes at a time using AVX2.
    let chunks = bits.chunks_exact(32);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // SAFETY: we check for avx2 at runtime before calling this function.
        unsafe {
            // Load 32 bytes into a 256-bit register.
            let v = _mm256_loadu_si256(chunk.as_ptr().cast::<__m256i>());

            // Detect non-zero bytes using an unsigned-safe approach:
            // cmpeq produces 0xFF where the byte == 0, 0x00 where byte != 0.
            // Inverting via andnot(eq_zero, all_ones) gives 0xFF for non-zero
            // bytes regardless of the sign bit, covering the full 0–255 range.
            let zero = _mm256_setzero_si256();
            let all_ones = _mm256_set1_epi8(-1_i8);
            // andnot(a, b) = (!a) & b; here gives 0xFF for lanes where v != 0.
            let nonzero_mask: u32 =
                _mm256_movemask_epi8(_mm256_andnot_si256(_mm256_cmpeq_epi8(v, zero), all_ones))
                    as u32;

            // Each bit in nonzero_mask corresponds to one input byte (LSB = lane 0).
            for lane in 0..32usize {
                let is_one = (nonzero_mask >> lane) & 1;
                if is_one == 0 {
                    // Bit 0: high then low
                    out.push(HIGH_LEVEL);
                    out.push(LOW_LEVEL);
                } else {
                    // Bit 1: low then high
                    out.push(LOW_LEVEL);
                    out.push(HIGH_LEVEL);
                }
            }
        }
    }

    // Handle the remaining bytes with the scalar path.
    for &b in remainder {
        if b == 0 {
            out.push(HIGH_LEVEL);
            out.push(LOW_LEVEL);
        } else {
            out.push(LOW_LEVEL);
            out.push(HIGH_LEVEL);
        }
    }

    out
}

// ── Decoding ──────────────────────────────────────────────────────────────────

/// Decode Manchester-coded audio `samples` back to a bit stream.
///
/// Each pair of consecutive samples is examined:
/// - `(HIGH, LOW)` → bit `0`
/// - `(LOW, HIGH)` → bit `1`
/// - Ambiguous / noisy pair → bit value is determined by the majority level
///   (the sample further from zero wins).
///
/// `threshold` is the *absolute* amplitude level that distinguishes a
/// meaningful HIGH/LOW from noise near zero.  Typical value: `8192` for
/// 25% of full-scale `i16`.
///
/// Returns one `u8` per decoded bit (value `0` or `1`).  If `samples.len()`
/// is odd the trailing unpaired sample is ignored.
pub fn manchester_decode_simd(samples: &[i16], threshold: i16) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return manchester_decode_avx2(samples, threshold);
        }
    }
    manchester_decode_scalar(samples, threshold)
}

/// Scalar Manchester decoding — reference implementation.
pub fn manchester_decode_scalar(samples: &[i16], threshold: i16) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() / 2);
    let mut i = 0;
    while i + 1 < samples.len() {
        let first = samples[i];
        let second = samples[i + 1];

        let first_positive = first >= threshold;
        let first_negative = first <= -threshold;
        let second_positive = second >= threshold;
        let second_negative = second <= -threshold;

        let bit = if first_positive && second_negative {
            // (HIGH, LOW) → bit 0
            0u8
        } else if first_negative && second_positive {
            // (LOW, HIGH) → bit 1
            1u8
        } else {
            // Ambiguous: use magnitude to decide.
            // The half with the larger absolute value determines the polarity.
            if first.unsigned_abs() >= second.unsigned_abs() {
                // First sample dominates
                if first >= 0 {
                    0u8 // First=HIGH → bit 0
                } else {
                    1u8 // First=LOW  → bit 1
                }
            } else {
                // Second sample dominates
                if second >= 0 {
                    1u8 // Second=HIGH → bit 1
                } else {
                    0u8 // Second=LOW  → bit 0
                }
            }
        };

        out.push(bit);
        i += 2;
    }
    out
}

/// AVX2 Manchester decoding — processes 16 interleaved sample-pairs per iter.
///
/// Manchester samples are interleaved: `[f0, s0, f1, s1, ..., f_k, s_k, ...]`.
/// Each iteration loads 32 consecutive i16 (= 16 pairs) and deinterleaves them
/// using `_mm256_srli_epi32` so that `firsts` holds `f_k` and `seconds` holds
/// `s_k` in the low 16 bits of each i32 word.  The high 16 bits remain zero,
/// which prevents false comparisons.
#[cfg(target_arch = "x86_64")]
fn manchester_decode_avx2(samples: &[i16], threshold: i16) -> Vec<u8> {
    use std::arch::x86_64::*;

    let mut out = Vec::with_capacity(samples.len() / 2);

    // 16 pairs per AVX2 iteration: each iteration consumes 32 i16 elements.
    let pair_count = samples.len() / 2;
    let avx_pairs_per_iter = 16usize;
    let full_iters = pair_count / avx_pairs_per_iter;
    let remainder_start = full_iters * avx_pairs_per_iter * 2; // in i16 elements

    unsafe {
        let thresh_vec = _mm256_set1_epi16(threshold);
        let neg_thresh_vec = _mm256_set1_epi16(-threshold);
        // Mask to isolate the low 16 bits of each i32 word (= the first sample
        // of each pair after casting the interleaved stream to i32 words).
        let mask_lo16 = _mm256_set1_epi32(0x0000_FFFF_u32 as i32);
        // Pre-compute (threshold - 1) so we can use strict cmpgt for >=.
        let thresh_m1 = _mm256_sub_epi16(thresh_vec, _mm256_set1_epi16(1));
        let neg_thresh_m1 = _mm256_sub_epi16(neg_thresh_vec, _mm256_set1_epi16(1));

        for iter in 0..full_iters {
            // base is an i16-element offset; each iteration spans 32 i16.
            let base = iter * avx_pairs_per_iter * 2;

            // Load 32 consecutive i16: pairs 0–7 in raw_lo, pairs 8–15 in raw_hi.
            // Memory layout (little-endian i16): [f0, s0, f1, s1, ..., f7, s7, f8, s8, ...]
            let raw_lo = _mm256_loadu_si256(samples[base..].as_ptr().cast::<__m256i>());
            let raw_hi = _mm256_loadu_si256(samples[base + 16..].as_ptr().cast::<__m256i>());

            // Deinterleave: treat each i32 word as (s_k << 16 | f_k) on
            // little-endian.  AND with mask isolates f_k in i16[2k]; shift
            // right 16 brings s_k into i16[2k].  Odd lanes (i16[2k+1]) are
            // zero in both cases, so they never trigger threshold comparisons.
            let firsts_lo = _mm256_and_si256(raw_lo, mask_lo16);
            let firsts_hi = _mm256_and_si256(raw_hi, mask_lo16);
            let seconds_lo = _mm256_srli_epi32(raw_lo, 16);
            let seconds_hi = _mm256_srli_epi32(raw_hi, 16);

            // first >= threshold  ⟺  first > (threshold - 1)
            let first_high_lo = _mm256_cmpgt_epi16(firsts_lo, thresh_m1);
            let first_high_hi = _mm256_cmpgt_epi16(firsts_hi, thresh_m1);
            // second <= -threshold  ⟺  (−threshold − 1) > second
            let second_low_lo = _mm256_cmpgt_epi16(neg_thresh_m1, seconds_lo);
            let second_low_hi = _mm256_cmpgt_epi16(neg_thresh_m1, seconds_hi);

            // Both conditions → (HIGH, LOW) = bit 0.
            let bit0_lo = _mm256_and_si256(first_high_lo, second_low_lo);
            let bit0_hi = _mm256_and_si256(first_high_hi, second_low_hi);

            // movemask_epi8 produces one bit per byte.  Each i16 lane occupies
            // 2 bytes; i16[2k] occupies bytes 4k and 4k+1 (little-endian i32).
            // The sign bit (= compare result) of i16[2k] appears at bit 4k.
            let bits_lo: u32 = _mm256_movemask_epi8(bit0_lo) as u32;
            let bits_hi: u32 = _mm256_movemask_epi8(bit0_hi) as u32;

            // Extract 8 decisions from each half (pairs 0–7 from bits_lo,
            // pairs 8–15 from bits_hi).  Bit 4k in the mask corresponds to
            // pair k in this half.
            for k in 0..8usize {
                let is_bit0 = (bits_lo >> (k * 4)) & 1;
                out.push(if is_bit0 != 0 { 0u8 } else { 1u8 });
            }
            for k in 0..8usize {
                let is_bit0 = (bits_hi >> (k * 4)) & 1;
                out.push(if is_bit0 != 0 { 0u8 } else { 1u8 });
            }
        }
    }

    // Scalar fallback for the remaining pairs (< 16 pairs).
    out.extend(manchester_decode_scalar(
        &samples[remainder_start..],
        threshold,
    ));

    out
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Encoding tests ──────────────────────────────────────────────────────

    #[test]
    fn test_encode_zero_bit() {
        let encoded = manchester_encode_scalar(&[0]);
        assert_eq!(encoded.len(), 2);
        assert_eq!(encoded[0], HIGH_LEVEL, "bit 0: first sample should be HIGH");
        assert_eq!(encoded[1], LOW_LEVEL, "bit 0: second sample should be LOW");
    }

    #[test]
    fn test_encode_one_bit() {
        let encoded = manchester_encode_scalar(&[1]);
        assert_eq!(encoded.len(), 2);
        assert_eq!(encoded[0], LOW_LEVEL, "bit 1: first sample should be LOW");
        assert_eq!(
            encoded[1], HIGH_LEVEL,
            "bit 1: second sample should be HIGH"
        );
    }

    #[test]
    fn test_encode_empty() {
        let encoded = manchester_encode_scalar(&[]);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_encode_zero_stream() {
        // All-zero bit stream: every pair should be (HIGH, LOW).
        let bits: Vec<u8> = vec![0; 8];
        let encoded = manchester_encode_scalar(&bits);
        assert_eq!(encoded.len(), 16);
        for i in 0..8 {
            assert_eq!(encoded[i * 2], HIGH_LEVEL, "bit {i}: first must be HIGH");
            assert_eq!(encoded[i * 2 + 1], LOW_LEVEL, "bit {i}: second must be LOW");
        }
    }

    #[test]
    fn test_encode_256_random_roundtrip_scalar() {
        // Generate a pseudo-random 256-bit sequence and verify round-trip.
        let bits: Vec<u8> = (0u32..256)
            .map(|i| ((i.wrapping_mul(2654435761) >> 16) & 1) as u8)
            .collect();

        let encoded = manchester_encode_scalar(&bits);
        let decoded = manchester_decode_scalar(&encoded, HIGH_LEVEL / 2);

        assert_eq!(decoded.len(), bits.len());
        for (i, (&orig, &dec)) in bits.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig, dec, "round-trip mismatch at bit {i}");
        }
    }

    #[test]
    fn test_encode_simd_matches_scalar() {
        // The SIMD path (or its scalar fallback) must match the reference.
        let bits: Vec<u8> = (0u8..255).collect(); // 255 bytes: non-zero = 1, zero = 0
        let scalar = manchester_encode_scalar(&bits);
        let simd = manchester_encode_simd(&bits);
        assert_eq!(scalar, simd, "SIMD encode must match scalar encode");
    }

    // ── Decoding tests ──────────────────────────────────────────────────────

    #[test]
    fn test_decode_zero_bit() {
        let samples = [HIGH_LEVEL, LOW_LEVEL];
        let decoded = manchester_decode_scalar(&samples, HIGH_LEVEL / 2);
        assert_eq!(decoded, vec![0u8]);
    }

    #[test]
    fn test_decode_one_bit() {
        let samples = [LOW_LEVEL, HIGH_LEVEL];
        let decoded = manchester_decode_scalar(&samples, HIGH_LEVEL / 2);
        assert_eq!(decoded, vec![1u8]);
    }

    #[test]
    fn test_decode_empty() {
        let decoded = manchester_decode_scalar(&[], 1000);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_odd_length_ignores_tail() {
        // Three samples: one valid pair + one trailing sample.
        let samples = [HIGH_LEVEL, LOW_LEVEL, HIGH_LEVEL]; // pair(0) + orphan
        let decoded = manchester_decode_scalar(&samples, HIGH_LEVEL / 2);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0], 0);
    }

    #[test]
    fn test_decode_simd_matches_scalar() {
        let bits: Vec<u8> = (0u8..255).map(|b| b & 1).collect();
        let samples = manchester_encode_scalar(&bits);
        let scalar = manchester_decode_scalar(&samples, HIGH_LEVEL / 2);
        let simd = manchester_decode_simd(&samples, HIGH_LEVEL / 2);
        assert_eq!(scalar, simd, "SIMD decode must match scalar decode");
    }

    #[test]
    fn test_encode_decode_256_random_simd_roundtrip() {
        // Full round-trip via SIMD paths.
        // Use a u64 LCG with a 64-bit multiplier.
        let bits: Vec<u8> = (0u64..256)
            .map(|i| {
                let hash = i
                    .wrapping_mul(6364136223846793005_u64)
                    .wrapping_add(1442695040888963407_u64);
                ((hash >> 32) & 1) as u8
            })
            .collect();

        let encoded = manchester_encode_simd(&bits);
        let decoded = manchester_decode_simd(&encoded, HIGH_LEVEL / 2);

        assert_eq!(decoded.len(), bits.len());
        for (i, (&orig, &dec)) in bits.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig, dec, "SIMD round-trip mismatch at bit {i}");
        }
    }
}

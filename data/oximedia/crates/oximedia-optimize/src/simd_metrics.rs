//! SIMD-accelerated block difference metrics for motion estimation.
//!
//! Provides hardware-accelerated implementations of:
//! - **SAD** (Sum of Absolute Differences) — fast, used for motion search
//! - **SATD** (Sum of Absolute Transformed Differences) — more accurate, used
//!   for final mode decision
//!
//! Uses `is_x86_feature_detected!` at runtime to dispatch to AVX2/SSE4.1
//! accelerated paths on x86-64, with a portable scalar fallback for all
//! other platforms. Results are bit-identical across all paths.

// ── SAD (Sum of Absolute Differences) ────────────────────────────────────────

/// Compute the Sum of Absolute Differences between two equally-sized blocks.
///
/// Both slices must have the same length. Returns 0 if either slice is empty.
#[must_use]
pub fn sad(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }
    sad_impl(&a[..len], &b[..len])
}

/// SAD dispatcher: selects AVX2 / SSE4.1 / scalar path at runtime.
fn sad_impl(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We have just checked that AVX2 is available at runtime.
            return unsafe { sad_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: We have just checked that SSE4.1 is available at runtime.
            return unsafe { sad_sse41(a, b) };
        }
    }
    sad_scalar(a, b)
}

/// AVX2 SAD using 256-bit registers processing 32 bytes per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn sad_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum: u32 = 0;
    let chunks = len / 32;

    for i in 0..chunks {
        let offset = i * 32;
        let va = _mm256_loadu_si256(a.as_ptr().add(offset).cast::<__m256i>());
        let vb = _mm256_loadu_si256(b.as_ptr().add(offset).cast::<__m256i>());
        // Compute |a - b| for each byte via (a - min(a,b)) + (b - min(a,b)) = |a - b|
        let min_ab = _mm256_min_epu8(va, vb);
        let max_ab = _mm256_max_epu8(va, vb);
        let diff = _mm256_sub_epi8(max_ab, min_ab);
        // Horizontal sum using sad_epu8 with zero
        let zero = _mm256_setzero_si256();
        let sad_result = _mm256_sad_epu8(diff, zero);
        // Extract 4 x u64 lanes and sum
        let lo = _mm256_extracti128_si256(sad_result, 0);
        let hi = _mm256_extracti128_si256(sad_result, 1);
        let s0 = _mm_extract_epi64(lo, 0) as u32;
        let s1 = _mm_extract_epi64(lo, 1) as u32;
        let s2 = _mm_extract_epi64(hi, 0) as u32;
        let s3 = _mm_extract_epi64(hi, 1) as u32;
        sum = sum
            .wrapping_add(s0)
            .wrapping_add(s1)
            .wrapping_add(s2)
            .wrapping_add(s3);
    }

    // Scalar remainder
    let remainder_start = chunks * 32;
    for i in remainder_start..len {
        sum += u32::from(a[i].abs_diff(b[i]));
    }
    sum
}

/// SSE4.1 SAD using 128-bit registers processing 16 bytes per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn sad_sse41(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum: u32 = 0;
    let chunks = len / 16;

    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm_loadu_si128(a.as_ptr().add(offset).cast::<__m128i>());
        let vb = _mm_loadu_si128(b.as_ptr().add(offset).cast::<__m128i>());
        let _zero = _mm_setzero_si128();
        let sad_result = _mm_sad_epu8(va, vb);
        let lo = _mm_extract_epi64(sad_result, 0) as u32;
        let hi = _mm_extract_epi64(sad_result, 1) as u32;
        sum = sum.wrapping_add(lo).wrapping_add(hi);
    }

    // Scalar remainder
    let remainder_start = chunks * 16;
    for i in remainder_start..len {
        sum += u32::from(a[i].abs_diff(b[i]));
    }
    let _ = sum; // suppress warning about unused variable from sse path
    sum
}

/// Portable scalar SAD fallback.
pub(crate) fn sad_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| u32::from(x.abs_diff(y)))
        .sum()
}

// ── SATD (Sum of Absolute Transformed Differences) ───────────────────────────

/// Compute the SATD of two 4×4 blocks using the Hadamard transform.
///
/// Each slice must contain at least `stride * 3 + 4` elements. The function
/// reads a 4×4 sub-block starting at offset 0.
///
/// Returns `None` if either slice is too short.
#[must_use]
pub fn satd_4x4(a: &[u8], b: &[u8], stride: usize) -> Option<u32> {
    if a.len() < stride * 3 + 4 || b.len() < stride * 3 + 4 {
        return None;
    }
    Some(satd_4x4_impl(a, b, stride))
}

/// SATD 4×4 Hadamard transform (scalar, works on all platforms).
fn satd_4x4_impl(a: &[u8], b: &[u8], stride: usize) -> u32 {
    // Build difference block
    let mut diff = [0i16; 16];
    for row in 0..4 {
        for col in 0..4 {
            let idx = row * stride + col;
            diff[row * 4 + col] = i16::from(a[idx]) - i16::from(b[idx]);
        }
    }

    // Row-wise Hadamard
    let mut tmp = [0i16; 16];
    for row in 0..4 {
        let base = row * 4;
        let d0 = diff[base];
        let d1 = diff[base + 1];
        let d2 = diff[base + 2];
        let d3 = diff[base + 3];
        let a0 = d0 + d1;
        let a1 = d0 - d1;
        let a2 = d2 + d3;
        let a3 = d2 - d3;
        tmp[base] = a0 + a2;
        tmp[base + 1] = a1 + a3;
        tmp[base + 2] = a0 - a2;
        tmp[base + 3] = a1 - a3;
    }

    // Column-wise Hadamard then accumulate absolute values
    let mut sum: u32 = 0;
    for col in 0..4 {
        let t0 = tmp[col];
        let t1 = tmp[4 + col];
        let t2 = tmp[8 + col];
        let t3 = tmp[12 + col];
        let a0 = t0 + t1;
        let a1 = t0 - t1;
        let a2 = t2 + t3;
        let a3 = t2 - t3;
        sum += u32::from((a0 + a2).unsigned_abs());
        sum += u32::from((a1 + a3).unsigned_abs());
        sum += u32::from((a0 - a2).unsigned_abs());
        sum += u32::from((a1 - a3).unsigned_abs());
    }

    // Standard SATD normalisation: divide by 2
    (sum + 1) / 2
}

// ── SATD 8×8 ─────────────────────────────────────────────────────────────────

/// Apply the in-place 8-point Hadamard butterfly to a row or column of i32 values.
///
/// Three-stage butterfly (x264 convention):
/// Stage 1 — pairs (0,4),(1,5),(2,6),(3,7)
/// Stage 2 — pairs (0,2),(1,3),(4,6),(5,7)
/// Stage 3 — pairs (0,1),(2,3),(4,5),(6,7)
#[inline(always)]
fn hadamard8_1d(v: &mut [i32; 8]) {
    // Stage 1
    let t0 = v[0] + v[4];
    let t1 = v[1] + v[5];
    let t2 = v[2] + v[6];
    let t3 = v[3] + v[7];
    let t4 = v[0] - v[4];
    let t5 = v[1] - v[5];
    let t6 = v[2] - v[6];
    let t7 = v[3] - v[7];
    // Stage 2
    let u0 = t0 + t2;
    let u1 = t1 + t3;
    let u2 = t0 - t2;
    let u3 = t1 - t3;
    let u4 = t4 + t6;
    let u5 = t5 + t7;
    let u6 = t4 - t6;
    let u7 = t5 - t7;
    // Stage 3
    v[0] = u0 + u1;
    v[1] = u0 - u1;
    v[2] = u2 + u3;
    v[3] = u2 - u3;
    v[4] = u4 + u5;
    v[5] = u4 - u5;
    v[6] = u6 + u7;
    v[7] = u6 - u7;
}

/// Scalar SATD 8×8 implementation.
///
/// Reads an 8×8 block from `a` and `b` using the given `stride`. Both slices
/// must contain at least `stride * 7 + 8` elements.
fn satd_8x8_scalar(a: &[u8], b: &[u8], stride: usize) -> u32 {
    // Build i32 difference matrix (row-major, 8×8).
    let mut coeff = [[0i32; 8]; 8];
    for row in 0..8 {
        for col in 0..8 {
            let idx = row * stride + col;
            coeff[row][col] = i32::from(a[idx]) - i32::from(b[idx]);
        }
    }

    // Row-wise Hadamard.
    for row in &mut coeff {
        hadamard8_1d(row);
    }

    // Column-wise Hadamard.
    let mut col_buf = [0i32; 8];
    for col in 0..8 {
        for (i, item) in col_buf.iter_mut().enumerate() {
            *item = coeff[i][col];
        }
        hadamard8_1d(&mut col_buf);
        for (i, item) in col_buf.iter().enumerate() {
            coeff[i][col] = *item;
        }
    }

    // Sum absolute values and normalise by 2 (x264 convention: (sum + 1) / 2).
    let sum: u32 = coeff
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| v.unsigned_abs())
        .sum();
    (sum + 1) / 2
}

/// Sum of Absolute Transformed Differences for 8×8 blocks (scalar, all platforms).
///
/// Reads an 8×8 block starting at offset 0 using the given stride.
/// Returns `None` if either slice is too short.
#[must_use]
pub fn satd_8x8(a: &[u8], b: &[u8], stride: usize) -> Option<u32> {
    let min_len = stride * 7 + 8;
    if a.len() < min_len || b.len() < min_len {
        return None;
    }
    Some(satd_8x8_scalar(a, b, stride))
}

/// Sum of Absolute Transformed Differences for 16×16 blocks.
///
/// Implemented as 4 SATD 8×8 sub-blocks (x264 style).
/// Returns `None` if either slice is too short.
#[must_use]
pub fn satd_16x16(a: &[u8], b: &[u8], stride: usize) -> Option<u32> {
    // Need at least stride*15 + 16 bytes.
    let min_len = stride * 15 + 16;
    if a.len() < min_len || b.len() < min_len {
        return None;
    }

    // Top-left 8×8
    let tl = satd_8x8_scalar(a, b, stride);
    // Top-right 8×8 — offset by 8 columns
    let tr = satd_8x8_scalar(&a[8..], &b[8..], stride);
    // Bottom-left 8×8 — offset by 8 rows
    let bl_offset = 8 * stride;
    let bl = satd_8x8_scalar(&a[bl_offset..], &b[bl_offset..], stride);
    // Bottom-right 8×8 — offset by 8 rows + 8 columns
    let br_offset = 8 * stride + 8;
    let br = satd_8x8_scalar(&a[br_offset..], &b[br_offset..], stride);

    Some(tl + tr + bl + br)
}

// ── Variable-block AVX2 SAD ───────────────────────────────────────────────────

/// AVX2-accelerated SAD for blocks of arbitrary size (x86_64 only).
///
/// Reads `h` rows of `w` pixels each from `a` (stride `stride_a`) and `b`
/// (stride `stride_b`).  Uses `_mm256_sad_epu8` for 32-byte chunks per row;
/// remainder handled by scalar code.
///
/// Returns `None` if:
/// - AVX2 is not available at runtime, OR
/// - Either slice is too short for the requested geometry.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn sad_block_avx2(
    a: &[u8],
    b: &[u8],
    w: usize,
    h: usize,
    stride_a: usize,
    stride_b: usize,
) -> Option<u32> {
    if h == 0 || w == 0 {
        return Some(0);
    }
    let required_a = stride_a * (h - 1) + w;
    let required_b = stride_b * (h - 1) + w;
    if a.len() < required_a || b.len() < required_b {
        return None;
    }
    if !is_x86_feature_detected!("avx2") {
        return None;
    }
    // SAFETY: AVX2 confirmed above; slice bounds validated above.
    Some(unsafe { sad_block_avx2_inner(a, b, w, h, stride_a, stride_b) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn sad_block_avx2_inner(
    a: &[u8],
    b: &[u8],
    w: usize,
    h: usize,
    stride_a: usize,
    stride_b: usize,
) -> u32 {
    use std::arch::x86_64::*;

    let mut total: u64 = 0;

    for row in 0..h {
        let row_a = &a[row * stride_a..row * stride_a + w];
        let row_b = &b[row * stride_b..row * stride_b + w];

        let chunks = w / 32;
        let mut acc = _mm256_setzero_si256();

        for chunk in 0..chunks {
            let off = chunk * 32;
            let va = _mm256_loadu_si256(row_a.as_ptr().add(off).cast::<__m256i>());
            let vb = _mm256_loadu_si256(row_b.as_ptr().add(off).cast::<__m256i>());
            // _mm256_sad_epu8: produces 4×u64 partial SADs for 32-byte input.
            let sad_v = _mm256_sad_epu8(va, vb);
            acc = _mm256_add_epi64(acc, sad_v);
        }

        // Horizontal reduction of acc (four u64 lanes).
        let lo = _mm256_extracti128_si256(acc, 0);
        let hi = _mm256_extracti128_si256(acc, 1);
        let sum_lo = _mm_add_epi64(lo, hi);
        let lane0 = _mm_extract_epi64(sum_lo, 0) as u64;
        let lane1 = _mm_extract_epi64(sum_lo, 1) as u64;
        total = total.wrapping_add(lane0).wrapping_add(lane1);

        // Scalar tail for remaining bytes.
        let tail_start = chunks * 32;
        for col in tail_start..w {
            total = total.wrapping_add(u64::from(row_a[col].abs_diff(row_b[col])));
        }
    }

    total as u32
}

// ── SAD for block with stride ────────────────────────────────────────────────

/// Compute SAD between two blocks with specified dimensions and strides.
///
/// Each block is accessed as `block[row * stride + col]` for `row` in `0..height`
/// and `col` in `0..width`.
///
/// Returns `None` if either slice is too short.
#[must_use]
pub fn sad_block(
    a: &[u8],
    b: &[u8],
    width: usize,
    height: usize,
    stride_a: usize,
    stride_b: usize,
) -> Option<u32> {
    let required_a = if height == 0 {
        0
    } else {
        stride_a * (height - 1) + width
    };
    let required_b = if height == 0 {
        0
    } else {
        stride_b * (height - 1) + width
    };
    if a.len() < required_a || b.len() < required_b {
        return None;
    }
    if width == 0 || height == 0 {
        return Some(0);
    }

    let mut total = 0u32;
    for row in 0..height {
        let row_a = &a[row * stride_a..row * stride_a + width];
        let row_b = &b[row * stride_b..row * stride_b + width];
        total += sad_impl(row_a, row_b);
    }
    Some(total)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sad_identical_blocks() {
        let a = vec![128u8; 64];
        let b = vec![128u8; 64];
        assert_eq!(sad(&a, &b), 0);
    }

    #[test]
    fn test_sad_known_values() {
        let a = vec![10u8, 20, 30, 40];
        let b = vec![12u8, 18, 35, 38];
        // |10-12| + |20-18| + |30-35| + |40-38| = 2 + 2 + 5 + 2 = 11
        assert_eq!(sad(&a, &b), 11);
    }

    #[test]
    fn test_sad_empty() {
        assert_eq!(sad(&[], &[]), 0);
    }

    #[test]
    fn test_sad_large_block_simd_vs_scalar() {
        // 64 bytes: exercises the SIMD path
        let a: Vec<u8> = (0..64).collect();
        let b: Vec<u8> = (0..64).map(|i| ((i + 3) % 256) as u8).collect();

        let simd_result = sad(&a, &b);
        let scalar_result = sad_scalar(&a, &b);
        assert_eq!(simd_result, scalar_result);
    }

    #[test]
    fn test_sad_non_aligned_length() {
        // 37 bytes: tests remainder handling
        let a: Vec<u8> = (0..37).map(|i| (i * 7 % 256) as u8).collect();
        let b: Vec<u8> = (0..37).map(|i| (i * 11 % 256) as u8).collect();

        let simd_result = sad(&a, &b);
        let scalar_result = sad_scalar(&a, &b);
        assert_eq!(simd_result, scalar_result);
    }

    #[test]
    fn test_satd_4x4_identical() {
        let block = vec![128u8; 16];
        assert_eq!(satd_4x4(&block, &block, 4), Some(0));
    }

    #[test]
    fn test_satd_4x4_known() {
        let a = vec![
            10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let b = vec![
            12u8, 18, 32, 38, 52, 58, 72, 78, 92, 98, 112, 118, 132, 138, 152, 158,
        ];

        let result = satd_4x4(&a, &b, 4);
        assert!(result.is_some());
        // The SATD value should be consistent
        let expected = satd_4x4_impl(&a, &b, 4);
        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_satd_4x4_too_short() {
        let a = vec![0u8; 8]; // too short for a 4x4 block
        let b = vec![0u8; 16];
        assert_eq!(satd_4x4(&a, &b, 4), None);
    }

    #[test]
    fn test_sad_block_strided() {
        // 8x4 image, process 4x4 block
        let a: Vec<u8> = (0..32).collect();
        let b: Vec<u8> = (0..32).map(|i| ((i + 1) % 256) as u8).collect();

        let result = sad_block(&a, &b, 4, 4, 8, 8);
        assert!(result.is_some());

        // Manual check: each row has 4 elements with diff=1, total = 4*4 = 16
        assert_eq!(result, Some(16));
    }

    #[test]
    fn test_sad_block_too_short() {
        let a = vec![0u8; 10];
        let b = vec![0u8; 32];
        assert_eq!(sad_block(&a, &b, 4, 4, 8, 8), None);
    }

    #[test]
    fn test_satd_nonzero_for_different_blocks() {
        let a = vec![0u8; 16];
        let b = vec![50u8; 16];
        let result = satd_4x4(&a, &b, 4);
        assert!(result.is_some());
        assert!(result.expect("SATD for different blocks should return a value") > 0);
    }

    #[test]
    fn test_sad_all_max_diff() {
        let a = vec![0u8; 32];
        let b = vec![255u8; 32];
        assert_eq!(sad(&a, &b), 32 * 255);
    }

    // ── SATD 8×8 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_satd_8x8_identical_blocks_zero() {
        let block = vec![128u8; 64];
        assert_eq!(satd_8x8(&block, &block, 8), Some(0));
    }

    #[test]
    fn test_satd_8x8_too_short_returns_none() {
        // Need stride*7 + 8 = 8*7+8 = 64. Supply only 63.
        let a = vec![0u8; 63];
        let b = vec![0u8; 64];
        assert_eq!(satd_8x8(&a, &b, 8), None);
    }

    #[test]
    fn test_satd_8x8_nonzero_for_different_blocks() {
        let a = vec![0u8; 64];
        let b = vec![100u8; 64];
        let result = satd_8x8(&a, &b, 8);
        assert!(result.is_some());
        assert!(result.expect("satd_8x8 should return Some for valid input") > 0);
    }

    #[test]
    fn test_satd_8x8_matches_scalar_reference() {
        // Run 1000 pseudo-random pairs and confirm pub API == direct scalar.
        use std::num::Wrapping;
        let mut state = Wrapping(0x1234_5678_u32);
        for _ in 0..1000 {
            state = state * Wrapping(1_664_525) + Wrapping(1_013_904_223);
            let seed_a = state.0;
            state = state * Wrapping(1_664_525) + Wrapping(1_013_904_223);
            let seed_b = state.0;

            // Mix seed with index using a simple LCG multiplier (truncated to u32).
            let a: Vec<u8> = (0..64u32)
                .map(|i| ((seed_a ^ i).wrapping_mul(1_664_525_u32)) as u8)
                .collect();
            let b: Vec<u8> = (0..64u32)
                .map(|i| ((seed_b ^ i).wrapping_mul(2_891_336_453_u32)) as u8)
                .collect();

            let via_pub = satd_8x8(&a, &b, 8).expect("satd_8x8 should succeed");
            let direct = satd_8x8_scalar(&a, &b, 8);
            assert_eq!(via_pub, direct, "mismatch at seed_a={seed_a}");
        }
    }

    #[test]
    fn test_satd_8x8_alternating_max() {
        // Alternating 0/255 checkerboard vs. a zero block.
        // The 8-point Hadamard concentrates the energy into a single coefficient;
        // after the 2D transform and /2 normalisation the result is 8×8×255/2 = 8160.
        let a: Vec<u8> = (0..64)
            .map(|i: u32| if i % 2 == 0 { 0 } else { 255 })
            .collect();
        let b = vec![0u8; 64];
        let result = satd_8x8(&a, &b, 8).expect("satd_8x8 should succeed");
        assert_eq!(
            result, 8160,
            "SATD alternating 0/255 vs zero must equal 8×8×255/2 = 8160"
        );
    }

    // ── SATD 16×16 tests ─────────────────────────────────────────────────────

    #[test]
    fn test_satd_16x16_identical_blocks_zero() {
        let block = vec![128u8; 256];
        assert_eq!(satd_16x16(&block, &block, 16), Some(0));
    }

    #[test]
    fn test_satd_16x16_too_short_returns_none() {
        // Need stride*15 + 16 = 16*15+16 = 256. Supply only 255.
        let a = vec![0u8; 255];
        let b = vec![0u8; 256];
        assert_eq!(satd_16x16(&a, &b, 16), None);
    }

    #[test]
    fn test_satd_16x16_correct_equals_four_8x8() {
        // Build a 16×16 block and verify satd_16x16 == sum of 4× satd_8x8 sub-blocks.
        let a: Vec<u8> = (0..256).map(|i: u32| (i.wrapping_mul(13)) as u8).collect();
        let b: Vec<u8> = (0..256)
            .map(|i: u32| (i.wrapping_mul(7) ^ 0xAB) as u8)
            .collect();
        let stride = 16;

        let full = satd_16x16(&a, &b, stride).expect("satd_16x16 should succeed");

        let tl = satd_8x8_scalar(&a, &b, stride);
        let tr = satd_8x8_scalar(&a[8..], &b[8..], stride);
        let bl = satd_8x8_scalar(&a[8 * stride..], &b[8 * stride..], stride);
        let br = satd_8x8_scalar(&a[8 * stride + 8..], &b[8 * stride + 8..], stride);

        assert_eq!(full, tl + tr + bl + br);
    }

    // ── sad_block_avx2 tests (x86_64 only) ───────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sad_block_avx2_matches_scalar() {
        // Test various block sizes to verify SIMD matches scalar.
        let frame_w = 128usize;
        let frame_h = 64usize;
        let a: Vec<u8> = (0..(frame_w * frame_h))
            .map(|i: usize| ((i.wrapping_mul(31)) & 0xFF) as u8)
            .collect();
        let b: Vec<u8> = (0..(frame_w * frame_h))
            .map(|i: usize| ((i.wrapping_mul(17) ^ 0x5A) & 0xFF) as u8)
            .collect();

        for &block_w in &[4usize, 8, 16, 32, 64] {
            for &block_h in &[4usize, 8, 16, 32] {
                let avx2_result = sad_block_avx2(&a, &b, block_w, block_h, frame_w, frame_w);
                let scalar_result = sad_block(&a, &b, block_w, block_h, frame_w, frame_w);

                match (avx2_result, scalar_result) {
                    (Some(avx), Some(sc)) => {
                        assert_eq!(avx, sc, "mismatch for {block_w}×{block_h}");
                    }
                    (None, _) => {
                        // AVX2 not available on this machine — acceptable.
                    }
                    (Some(_), None) => panic!("scalar returned None but AVX2 returned Some"),
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sad_block_avx2_zero_for_identical() {
        let frame: Vec<u8> = vec![128u8; 64 * 64];
        let result = sad_block_avx2(&frame, &frame, 32, 32, 64, 64);
        if let Some(v) = result {
            assert_eq!(v, 0);
        }
        // If None (no AVX2), test passes trivially.
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sad_block_avx2_returns_none_for_too_short() {
        let a = vec![0u8; 10];
        let b = vec![0u8; 100];
        // w=32, h=4, stride=32 → required = 32*3+32 = 128 > 10 → None
        let result = sad_block_avx2(&a, &b, 32, 4, 32, 32);
        assert_eq!(result, None);
    }
}

//! Sum of Absolute Transformed Differences (SATD) kernel.
//!
//! SATD uses the Hadamard–Walsh transform (WHT) to measure block similarity in
//! a frequency-aware way.  Compared to plain SAD, SATD better correlates with
//! distortion after quantisation and is widely used in H.264/AVC, HEVC and AV1
//! encoders as a low-cost rate-distortion proxy.
//!
//! Formula (for an N×N block):
//!
//! ```text
//! SATD = Σ |WHT(src − ref)|
//! ```
//!
//! # Public API
//!
//! - [`satd`]         — dispatcher that selects the best available kernel.
//! - [`satd_4x4`]     — convenience wrapper for 4×4 blocks.
//! - [`satd_8x8`]     — convenience wrapper for 8×8 blocks.
//! - [`satd_16x16`]   — convenience wrapper for 16×16 blocks.
//! - [`SatdBlockSize`] — block-size selector enum.

#![allow(dead_code)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]

use crate::{detect_cpu_features, scalar, SimdError};

/// Block sizes supported by the SATD kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatdBlockSize {
    /// 4×4 (16 pixels).
    Block4x4,
    /// 8×8 (64 pixels).
    Block8x8,
    /// 16×16 (256 pixels) — split into four 8×8 sub-blocks internally.
    Block16x16,
    /// 32×32 (1024 pixels) — split into sixteen 8×8 sub-blocks.
    Block32x32,
}

impl SatdBlockSize {
    /// Side length of the block.
    #[must_use]
    pub fn side(self) -> usize {
        match self {
            Self::Block4x4 => 4,
            Self::Block8x8 => 8,
            Self::Block16x16 => 16,
            Self::Block32x32 => 32,
        }
    }

    /// Total number of pixels.
    #[must_use]
    pub fn pixels(self) -> usize {
        let s = self.side();
        s * s
    }
}

// ── Scalar helpers ────────────────────────────────────────────────────────────

/// Compute SATD for a `bw × bh` sub-block within a strided buffer.
///
/// `src` is accessed with stride `src_stride`; `ref_` with `ref_stride`.
/// Both must be at least `bh * stride` bytes long.
fn satd_strided(
    src: &[u8],
    src_stride: usize,
    ref_: &[u8],
    ref_stride: usize,
    bw: usize,
    bh: usize,
) -> u32 {
    debug_assert!(bw.is_power_of_two());
    debug_assert_eq!(bw, bh, "SATD sub-block must be square");

    let n = bw; // == bh
    let mut diff = vec![0i32; n * n];
    for row in 0..n {
        for col in 0..n {
            let s = i32::from(src[row * src_stride + col]);
            let r = i32::from(ref_[row * ref_stride + col]);
            diff[row * n + col] = s - r;
        }
    }

    scalar::hadamard_2d(&mut diff, n);
    diff.iter().map(|&v| v.unsigned_abs()).sum()
}

// ── AVX2 path (x86_64) ───────────────────────────────────────────────────────

/// AVX2-accelerated SATD for a packed 8×8 block (both buffers row-major,
/// stride == 8).  Falls back to scalar on non-AVX2 machines.
#[cfg(target_arch = "x86_64")]
#[inline]
fn satd_8x8_avx2_packed(src: &[u8; 64], ref_: &[u8; 64]) -> u32 {
    // Use scalar path — the WHT butterfly is integer-only, and the compiler
    // will auto-vectorise the diff + accumulation loops with AVX2 if available.
    // A hand-vectorised path using _mm256_sub_epi16 + butterfly would yield
    // ~2-4× speedup on real AVX2 hardware but is deferred to a future PR.
    let mut diff = [0i32; 64];
    for (i, (&s, &r)) in src.iter().zip(ref_.iter()).enumerate() {
        diff[i] = i32::from(s) - i32::from(r);
    }
    scalar::hadamard_2d(&mut diff, 8);
    diff.iter().map(|&v| v.unsigned_abs()).sum()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute SATD between two equal-sized pixel blocks.
///
/// Both slices must be packed row-major with no padding (stride == `size.side()`).
/// The function dispatches to the best available SIMD path.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either slice is too small.
pub fn satd(src: &[u8], ref_: &[u8], size: SatdBlockSize) -> Result<u32, SimdError> {
    let n = size.side();
    let pixels = n * n;
    if src.len() < pixels || ref_.len() < pixels {
        return Err(SimdError::InvalidBufferSize);
    }

    let _feat = detect_cpu_features();

    // For 16×16 and 32×32 we split into 8×8 sub-blocks for better cache
    // locality and to allow the compiler to auto-vectorise the inner loop.
    match size {
        SatdBlockSize::Block4x4 => Ok(satd_strided(src, 4, ref_, 4, 4, 4)),
        SatdBlockSize::Block8x8 => {
            #[cfg(target_arch = "x86_64")]
            if _feat.avx2 {
                let mut src8 = [0u8; 64];
                let mut ref8 = [0u8; 64];
                src8.copy_from_slice(&src[..64]);
                ref8.copy_from_slice(&ref_[..64]);
                return Ok(satd_8x8_avx2_packed(&src8, &ref8));
            }
            Ok(satd_strided(src, 8, ref_, 8, 8, 8))
        }
        SatdBlockSize::Block16x16 => {
            // Four 8×8 sub-blocks: offsets (row, col) in units of 8
            let sub_offsets = [(0, 0), (0, 8), (8, 0), (8, 8)];
            let mut total = 0u32;
            for (r0, c0) in sub_offsets {
                let src_off = r0 * n + c0;
                let ref_off = r0 * n + c0;
                // Build packed 8×8 slice via strided copy
                let sub = satd_strided(&src[src_off..], n, &ref_[ref_off..], n, 8, 8);
                total = total.saturating_add(sub);
            }
            Ok(total)
        }
        SatdBlockSize::Block32x32 => {
            // Sixteen 8×8 sub-blocks
            let mut total = 0u32;
            for rb in 0..4 {
                for cb in 0..4 {
                    let r0 = rb * 8;
                    let c0 = cb * 8;
                    let src_off = r0 * n + c0;
                    let ref_off = r0 * n + c0;
                    let sub = satd_strided(&src[src_off..], n, &ref_[ref_off..], n, 8, 8);
                    total = total.saturating_add(sub);
                }
            }
            Ok(total)
        }
    }
}

/// Compute SATD for a 4×4 block (packed, stride = 4).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either slice has fewer than 16
/// elements.
pub fn satd_4x4(src: &[u8], ref_: &[u8]) -> Result<u32, SimdError> {
    satd(src, ref_, SatdBlockSize::Block4x4)
}

/// Compute SATD for an 8×8 block (packed, stride = 8).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either slice has fewer than 64
/// elements.
pub fn satd_8x8(src: &[u8], ref_: &[u8]) -> Result<u32, SimdError> {
    satd(src, ref_, SatdBlockSize::Block8x8)
}

/// Compute SATD for a 16×16 block (packed, stride = 16).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either slice has fewer than 256
/// elements.
pub fn satd_16x16(src: &[u8], ref_: &[u8]) -> Result<u32, SimdError> {
    satd(src, ref_, SatdBlockSize::Block16x16)
}

/// Compute SATD for a 32×32 block (packed, stride = 32).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if either slice has fewer than
/// 1024 elements.
pub fn satd_32x32(src: &[u8], ref_: &[u8]) -> Result<u32, SimdError> {
    satd(src, ref_, SatdBlockSize::Block32x32)
}

/// Compute SATD over a strided `block_w × block_h` region in a larger frame.
///
/// This is the most general entry point.  Both `src` and `ref_` may have
/// arbitrary strides.  `block_w` must equal `block_h` and both must be a
/// power of two.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the strides or slice lengths
/// are inconsistent with `block_w` / `block_h`.
pub fn satd_block_strided(
    src: &[u8],
    src_stride: usize,
    ref_: &[u8],
    ref_stride: usize,
    block_w: usize,
    block_h: usize,
) -> Result<u32, SimdError> {
    if block_w == 0 || block_h == 0 || block_w != block_h || !block_w.is_power_of_two() {
        return Err(SimdError::InvalidBufferSize);
    }
    if src_stride < block_w
        || ref_stride < block_w
        || src.len() < block_h * src_stride
        || ref_.len() < block_h * ref_stride
    {
        return Err(SimdError::InvalidBufferSize);
    }

    // For blocks > 8 we split into 8×8 tiles for cache efficiency
    if block_w <= 8 {
        return Ok(satd_strided(
            src, src_stride, ref_, ref_stride, block_w, block_h,
        ));
    }

    let tile = 8usize;
    let tiles = block_w / tile;
    let mut total = 0u32;
    for tr in 0..tiles {
        for tc in 0..tiles {
            let r0 = tr * tile;
            let c0 = tc * tile;
            let src_off = r0 * src_stride + c0;
            let ref_off = r0 * ref_stride + c0;
            let sub = satd_strided(
                &src[src_off..],
                src_stride,
                &ref_[ref_off..],
                ref_stride,
                tile,
                tile,
            );
            total = total.saturating_add(sub);
        }
    }
    Ok(total)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn satd_identical_blocks_is_zero() {
        let block = vec![128u8; 64];
        let result = satd_8x8(&block, &block).expect("satd_8x8 identical");
        assert_eq!(result, 0, "SATD of identical blocks must be zero");
    }

    #[test]
    fn satd_4x4_identical_zero() {
        let block = vec![100u8; 16];
        let r = satd_4x4(&block, &block).expect("satd_4x4");
        assert_eq!(r, 0);
    }

    #[test]
    fn satd_16x16_identical_zero() {
        let block = vec![200u8; 256];
        let r = satd_16x16(&block, &block).expect("satd_16x16");
        assert_eq!(r, 0);
    }

    #[test]
    fn satd_32x32_identical_zero() {
        let block = vec![50u8; 1024];
        let r = satd_32x32(&block, &block).expect("satd_32x32");
        assert_eq!(r, 0);
    }

    #[test]
    fn satd_greater_than_sad_for_smooth_offset() {
        // A constant DC offset: SAD = 8*8*10 = 640, SATD concentrates energy in DC
        let src = vec![100u8; 64];
        let ref_ = vec![110u8; 64];
        let satd_val = satd_8x8(&src, &ref_).expect("satd_8x8 offset");
        let sad_val: u32 = src
            .iter()
            .zip(ref_.iter())
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).unsigned_abs())
            .sum();
        // SATD for a constant offset concentrates all energy in the DC coefficient,
        // so SATD >= SAD for uniform blocks.
        assert!(satd_val >= sad_val, "SATD={satd_val} SAD={sad_val}");
    }

    #[test]
    fn satd_increases_with_difference() {
        let src = vec![100u8; 64];
        let small_diff: Vec<u8> = vec![105u8; 64];
        let large_diff: Vec<u8> = vec![150u8; 64];
        let s1 = satd_8x8(&src, &small_diff).expect("s1");
        let s2 = satd_8x8(&src, &large_diff).expect("s2");
        assert!(
            s2 > s1,
            "larger difference should yield larger SATD: s1={s1} s2={s2}"
        );
    }

    #[test]
    fn satd_is_symmetric() {
        let a: Vec<u8> = (0..64).map(|i| (i * 3) as u8).collect();
        let b: Vec<u8> = (0..64).map(|i| (i * 2 + 10) as u8).collect();
        let ab = satd_8x8(&a, &b).expect("ab");
        let ba = satd_8x8(&b, &a).expect("ba");
        assert_eq!(ab, ba, "SATD must be symmetric");
    }

    #[test]
    fn satd_buffer_too_small_returns_error() {
        let a = vec![0u8; 16];
        let b = vec![0u8; 16];
        assert_eq!(satd_8x8(&a, &b), Err(SimdError::InvalidBufferSize));
    }

    #[test]
    fn satd_block_size_pixels() {
        assert_eq!(SatdBlockSize::Block4x4.pixels(), 16);
        assert_eq!(SatdBlockSize::Block8x8.pixels(), 64);
        assert_eq!(SatdBlockSize::Block16x16.pixels(), 256);
        assert_eq!(SatdBlockSize::Block32x32.pixels(), 1024);
    }

    #[test]
    fn satd_strided_matches_packed() {
        let src: Vec<u8> = (0..64).map(|i| (i as u8).wrapping_mul(3)).collect();
        let ref_: Vec<u8> = (0..64)
            .map(|i| (i as u8).wrapping_mul(2).wrapping_add(5))
            .collect();
        let packed = satd_8x8(&src, &ref_).expect("packed");
        let strided = satd_block_strided(&src, 8, &ref_, 8, 8, 8).expect("strided");
        assert_eq!(packed, strided);
    }

    #[test]
    fn hadamard_2d_roundtrip() {
        use crate::scalar::hadamard_2d;
        let original: Vec<i32> = (0..64).map(|i| i * 3).collect();
        let mut buf = original.clone();
        hadamard_2d(&mut buf, 8);
        hadamard_2d(&mut buf, 8); // WHT(WHT(x)) = n² * x
        let n2 = 64i32;
        for (i, (&orig, &result)) in original.iter().zip(buf.iter()).enumerate() {
            assert_eq!(result, orig * n2, "roundtrip mismatch at index {i}");
        }
    }
}

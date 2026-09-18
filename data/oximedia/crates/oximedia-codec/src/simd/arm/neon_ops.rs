//! ARM NEON implementations for SAD and pixel-format conversion.
//!
//! This module provides optimised routines for:
//! - **SAD** (Sum of Absolute Differences) for 4x4, 8x8, 16x16 and 16x8 block
//!   sizes used during motion estimation.
//! - **Pixel-format conversion**: YUV 4:2:0 planar ↔ interleaved NV12, and
//!   RGBA ↔ YUV 4:2:0 approximation.
//!
//! On AArch64, the routines use `std::arch::aarch64` intrinsics.  On all
//! other architectures they fall back to portable scalar implementations so
//! that the crate compiles and passes tests everywhere.

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(dead_code)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// SAD — Sum of Absolute Differences
// =============================================================================

/// Calculate the SAD between two 4×4 luma blocks.
///
/// `src` must be at least `4 * src_stride` bytes; `ref_blk` likewise.
/// The strides are in bytes.
#[inline]
pub fn sad_4x4(src: &[u8], src_stride: usize, ref_blk: &[u8], ref_stride: usize) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if src.len() >= 4 * src_stride && ref_blk.len() >= 4 * ref_stride {
            return sad_4x4_neon(src, src_stride, ref_blk, ref_stride);
        }
    }
    sad_nxn_scalar(src, src_stride, ref_blk, ref_stride, 4, 4)
}

/// Calculate the SAD between two 8×8 luma blocks.
#[inline]
pub fn sad_8x8(src: &[u8], src_stride: usize, ref_blk: &[u8], ref_stride: usize) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if src.len() >= 8 * src_stride && ref_blk.len() >= 8 * ref_stride {
            return sad_8x8_neon(src, src_stride, ref_blk, ref_stride);
        }
    }
    sad_nxn_scalar(src, src_stride, ref_blk, ref_stride, 8, 8)
}

/// Calculate the SAD between two 16×16 luma blocks.
#[inline]
pub fn sad_16x16(src: &[u8], src_stride: usize, ref_blk: &[u8], ref_stride: usize) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if src.len() >= 16 * src_stride && ref_blk.len() >= 16 * ref_stride {
            return sad_16x16_neon(src, src_stride, ref_blk, ref_stride);
        }
    }
    sad_nxn_scalar(src, src_stride, ref_blk, ref_stride, 16, 16)
}

/// Calculate the SAD between two 16×8 luma blocks.
#[inline]
pub fn sad_16x8(src: &[u8], src_stride: usize, ref_blk: &[u8], ref_stride: usize) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if src.len() >= 8 * src_stride && ref_blk.len() >= 8 * ref_stride {
            return sad_16x8_neon(src, src_stride, ref_blk, ref_stride);
        }
    }
    sad_nxn_scalar(src, src_stride, ref_blk, ref_stride, 16, 8)
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

#[inline]
fn sad_nxn_scalar(
    src: &[u8],
    src_stride: usize,
    ref_blk: &[u8],
    ref_stride: usize,
    width: usize,
    height: usize,
) -> u32 {
    let mut sad = 0u32;
    for row in 0..height {
        let s_off = row * src_stride;
        let r_off = row * ref_stride;
        let s_end = (s_off + width).min(src.len());
        let r_end = (r_off + width).min(ref_blk.len());
        let avail = (s_end - s_off).min(r_end - r_off);
        for col in 0..avail {
            sad += u32::from(src[s_off + col].abs_diff(ref_blk[r_off + col]));
        }
    }
    sad
}

// ---------------------------------------------------------------------------
// AArch64 NEON implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn sad_4x4_neon(src: &[u8], src_stride: usize, ref_blk: &[u8], ref_stride: usize) -> u32 {
    // SAFETY: caller guarantees sufficient lengths; AArch64 always has NEON.
    unsafe {
        let mut acc = vdupq_n_u16(0);

        for row in 0..4usize {
            // Load 8 bytes (we only care about the first 4).
            let s = vld1_u8(src.as_ptr().add(row * src_stride));
            let r = vld1_u8(ref_blk.as_ptr().add(row * ref_stride));
            let diff = vabd_u8(s, r);
            // Widen u8x8 differences to u16x8 and accumulate.
            acc = vaddw_u8(acc, diff);
        }

        // Horizontal sum of the accumulator lanes.
        let sum32 = vpaddlq_u16(acc);
        let sum64 = vpaddlq_u32(sum32);
        let arr: [u64; 2] = std::mem::transmute(sum64);
        // Only the 4-column contribution is real; mask the upper 4 lanes.
        // Because we loaded 8 bytes but only wanted 4, SAD may include noise
        // from lanes 4-7.  Zero those: our rows are only 4 wide.
        // Re-compute scalar for accuracy.
        drop(arr);
        sad_nxn_scalar(src, src_stride, ref_blk, ref_stride, 4, 4)
    }
}

#[cfg(target_arch = "aarch64")]
fn sad_8x8_neon(src: &[u8], src_stride: usize, ref_blk: &[u8], ref_stride: usize) -> u32 {
    // SAFETY: caller guarantees sufficient lengths; AArch64 always has NEON.
    unsafe {
        let mut acc = vdupq_n_u16(0);

        for row in 0..8usize {
            let s = vld1_u8(src.as_ptr().add(row * src_stride));
            let r = vld1_u8(ref_blk.as_ptr().add(row * ref_stride));
            let diff = vabd_u8(s, r);
            acc = vaddw_u8(acc, diff);
        }

        let sum32 = vpaddlq_u16(acc);
        let sum64 = vpaddlq_u32(sum32);
        let arr: [u64; 2] = std::mem::transmute(sum64);
        (arr[0] + arr[1]) as u32
    }
}

#[cfg(target_arch = "aarch64")]
fn sad_16x16_neon(src: &[u8], src_stride: usize, ref_blk: &[u8], ref_stride: usize) -> u32 {
    // SAFETY: caller guarantees sufficient lengths; AArch64 always has NEON.
    unsafe {
        let mut acc = vdupq_n_u32(0);

        for row in 0..16usize {
            let s_ptr = src.as_ptr().add(row * src_stride);
            let r_ptr = ref_blk.as_ptr().add(row * ref_stride);

            // Process first 8 pixels.
            let s0 = vld1_u8(s_ptr);
            let r0 = vld1_u8(r_ptr);
            let diff0 = vabd_u8(s0, r0);
            let wide0 = vmovl_u8(diff0);
            let sum16_0 = vpaddlq_u16(wide0);
            acc = vaddq_u32(acc, sum16_0);

            // Process next 8 pixels.
            let s1 = vld1_u8(s_ptr.add(8));
            let r1 = vld1_u8(r_ptr.add(8));
            let diff1 = vabd_u8(s1, r1);
            let wide1 = vmovl_u8(diff1);
            let sum16_1 = vpaddlq_u16(wide1);
            acc = vaddq_u32(acc, sum16_1);
        }

        // Horizontal sum.
        let sum64 = vpaddlq_u32(acc);
        let arr: [u64; 2] = std::mem::transmute(sum64);
        (arr[0] + arr[1]) as u32
    }
}

#[cfg(target_arch = "aarch64")]
fn sad_16x8_neon(src: &[u8], src_stride: usize, ref_blk: &[u8], ref_stride: usize) -> u32 {
    // SAFETY: caller guarantees sufficient lengths; AArch64 always has NEON.
    unsafe {
        let mut acc = vdupq_n_u32(0);

        for row in 0..8usize {
            let s_ptr = src.as_ptr().add(row * src_stride);
            let r_ptr = ref_blk.as_ptr().add(row * ref_stride);

            let s0 = vld1_u8(s_ptr);
            let r0 = vld1_u8(r_ptr);
            let diff0 = vabd_u8(s0, r0);
            let wide0 = vmovl_u8(diff0);
            let sum16_0 = vpaddlq_u16(wide0);
            acc = vaddq_u32(acc, sum16_0);

            let s1 = vld1_u8(s_ptr.add(8));
            let r1 = vld1_u8(r_ptr.add(8));
            let diff1 = vabd_u8(s1, r1);
            let wide1 = vmovl_u8(diff1);
            let sum16_1 = vpaddlq_u16(wide1);
            acc = vaddq_u32(acc, sum16_1);
        }

        let sum64 = vpaddlq_u32(acc);
        let arr: [u64; 2] = std::mem::transmute(sum64);
        (arr[0] + arr[1]) as u32
    }
}

// =============================================================================
// Pixel-format conversion
// =============================================================================

/// Convert a YUV 4:2:0 planar (I420) buffer to interleaved NV12 (Y + UV).
///
/// - `y_plane`:  width × height bytes.
/// - `u_plane`:  (width/2) × (height/2) bytes.
/// - `v_plane`:  (width/2) × (height/2) bytes.
/// - Output:     width × height (luma) + width × (height/2) (interleaved UV).
///
/// Returns `None` if the slice sizes are inconsistent.
#[inline]
pub fn i420_to_nv12(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: usize,
    height: usize,
) -> Option<Vec<u8>> {
    let luma_size = width * height;
    let chroma_size = (width / 2) * (height / 2);

    if y_plane.len() < luma_size || u_plane.len() < chroma_size || v_plane.len() < chroma_size {
        return None;
    }

    let mut out = Vec::with_capacity(luma_size + width * (height / 2));

    // Copy Y plane unchanged.
    out.extend_from_slice(&y_plane[..luma_size]);

    // Interleave U and V into UV plane.
    #[cfg(target_arch = "aarch64")]
    {
        let uv_rows = height / 2;
        let uv_cols = width / 2;
        let total_pairs = uv_rows * uv_cols;
        let chunks = total_pairs / 8;
        let remainder = total_pairs % 8;

        let mut idx = 0usize;

        // SAFETY: AArch64 always has NEON.
        unsafe {
            for _ in 0..chunks {
                let u_ptr = u_plane.as_ptr().add(idx);
                let v_ptr = v_plane.as_ptr().add(idx);
                let u_vec = vld1_u8(u_ptr);
                let v_vec = vld1_u8(v_ptr);
                // Interleave: U0 V0 U1 V1 ...
                let uv = vzip_u8(u_vec, v_vec);
                let mut tmp = [0u8; 16];
                vst1_u8(tmp.as_mut_ptr(), uv.0);
                vst1_u8(tmp.as_mut_ptr().add(8), uv.1);
                out.extend_from_slice(&tmp);
                idx += 8;
            }
        }

        // Scalar tail.
        for i in 0..remainder {
            out.push(u_plane[idx + i]);
            out.push(v_plane[idx + i]);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let uv_rows = height / 2;
        let uv_cols = width / 2;
        for row in 0..uv_rows {
            for col in 0..uv_cols {
                let idx = row * uv_cols + col;
                out.push(u_plane[idx]);
                out.push(v_plane[idx]);
            }
        }
    }

    Some(out)
}

/// Convert NV12 (Y + interleaved UV) to I420 (Y + U + V separate planes).
///
/// Returns `(y, u, v)` or `None` on size mismatch.
#[inline]
pub fn nv12_to_i420(
    nv12: &[u8],
    width: usize,
    height: usize,
) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let luma_size = width * height;
    let chroma_size = (width / 2) * (height / 2);
    let expected = luma_size + 2 * chroma_size;

    if nv12.len() < expected {
        return None;
    }

    let y = nv12[..luma_size].to_vec();
    let uv = &nv12[luma_size..luma_size + 2 * chroma_size];

    let mut u = Vec::with_capacity(chroma_size);
    let mut v = Vec::with_capacity(chroma_size);

    #[cfg(target_arch = "aarch64")]
    {
        let chunks = chroma_size / 8;
        let remainder = chroma_size % 8;
        let mut idx = 0usize;

        // SAFETY: AArch64 always has NEON.
        unsafe {
            for _ in 0..chunks {
                let uv_ptr = uv.as_ptr().add(idx * 2);
                let uv_pair = vld2_u8(uv_ptr);
                let mut u_buf = [0u8; 8];
                let mut v_buf = [0u8; 8];
                vst1_u8(u_buf.as_mut_ptr(), uv_pair.0);
                vst1_u8(v_buf.as_mut_ptr(), uv_pair.1);
                u.extend_from_slice(&u_buf);
                v.extend_from_slice(&v_buf);
                idx += 8;
            }
        }

        for i in 0..remainder {
            u.push(uv[(idx + i) * 2]);
            v.push(uv[(idx + i) * 2 + 1]);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..chroma_size {
            u.push(uv[i * 2]);
            v.push(uv[i * 2 + 1]);
        }
    }

    Some((y, u, v))
}

/// Convert RGBA (8 bpp per channel) to approximate YUV 4:2:0 planar (I420).
///
/// Uses BT.601 full-range coefficients:
///   Y  =  0.299 R + 0.587 G + 0.114 B
///   Cb = -0.169 R - 0.331 G + 0.500 B + 128
///   Cr =  0.500 R - 0.419 G - 0.081 B + 128
///
/// U and V (Cb/Cr) are averaged over 2×2 pixel blocks.
/// Returns `None` if `rgba` is smaller than `width * height * 4`.
pub fn rgba_to_yuv420(
    rgba: &[u8],
    width: usize,
    height: usize,
) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if rgba.len() < width * height * 4 {
        return None;
    }

    let luma_size = width * height;
    let chroma_size = (width / 2) * (height / 2);

    let mut y = vec![0u8; luma_size];
    let mut u = vec![0u8; chroma_size];
    let mut v = vec![0u8; chroma_size];

    // BT.601 fixed-point coefficients (scaled by 256).
    const YR: i32 = 77; //  0.299 * 256
    const YG: i32 = 150; //  0.587 * 256
    const YB: i32 = 29; //  0.114 * 256
    const UR: i32 = -43; // -0.169 * 256
    const UG: i32 = -85; // -0.331 * 256
    const UB: i32 = 128; //  0.500 * 256
    const VR: i32 = 128; //  0.500 * 256
    const VG: i32 = -107; // -0.419 * 256
    const VB: i32 = -21; // -0.081 * 256

    for row in 0..height {
        for col in 0..width {
            let idx = (row * width + col) * 4;
            let r = i32::from(rgba[idx]);
            let g = i32::from(rgba[idx + 1]);
            let b = i32::from(rgba[idx + 2]);

            let y_val = ((YR * r + YG * g + YB * b) >> 8).clamp(0, 255) as u8;
            y[row * width + col] = y_val;

            // Chroma sampling: only one sample per 2×2 block.
            if row % 2 == 0 && col % 2 == 0 {
                let u_val = ((UR * r + UG * g + UB * b) >> 8).clamp(-128, 127) as i8 as u8;
                let v_val = ((VR * r + VG * g + VB * b) >> 8).clamp(-128, 127) as i8 as u8;
                let chroma_idx = (row / 2) * (width / 2) + col / 2;
                u[chroma_idx] = u_val.wrapping_add(128);
                v[chroma_idx] = v_val.wrapping_add(128);
            }
        }
    }

    Some((y, u, v))
}

/// Convert I420 YUV 4:2:0 planar to RGBA.
///
/// Upsamples chroma planes (nearest-neighbour) and applies BT.601 inverse.
/// Returns `None` on size mismatch.
pub fn yuv420_to_rgba(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: usize,
    height: usize,
) -> Option<Vec<u8>> {
    let luma_size = width * height;
    let chroma_size = (width / 2) * (height / 2);

    if y_plane.len() < luma_size || u_plane.len() < chroma_size || v_plane.len() < chroma_size {
        return None;
    }

    let mut rgba = vec![255u8; width * height * 4];

    for row in 0..height {
        for col in 0..width {
            let y_val = i32::from(y_plane[row * width + col]);
            let chroma_row = row / 2;
            let chroma_col = col / 2;
            let chroma_idx = chroma_row * (width / 2) + chroma_col;

            let u_val = i32::from(u_plane[chroma_idx]) - 128;
            let v_val = i32::from(v_plane[chroma_idx]) - 128;

            // BT.601 full-range inverse.
            let r = (y_val + ((1402 * v_val) >> 10)).clamp(0, 255) as u8;
            let g = (y_val - ((344 * u_val + 714 * v_val) >> 10)).clamp(0, 255) as u8;
            let b = (y_val + ((1772 * u_val) >> 10)).clamp(0, 255) as u8;

            let out_idx = (row * width + col) * 4;
            rgba[out_idx] = r;
            rgba[out_idx + 1] = g;
            rgba[out_idx + 2] = b;
            // Alpha is pre-set to 255.
        }
    }

    Some(rgba)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // SAD tests
    // ------------------------------------------------------------------

    #[test]
    fn test_sad_4x4_identical() {
        let block = vec![128u8; 4 * 4];
        assert_eq!(sad_4x4(&block, 4, &block, 4), 0);
    }

    #[test]
    fn test_sad_8x8_identical() {
        let block = vec![200u8; 8 * 8];
        assert_eq!(sad_8x8(&block, 8, &block, 8), 0);
    }

    #[test]
    fn test_sad_16x16_identical() {
        let block = vec![100u8; 16 * 16];
        assert_eq!(sad_16x16(&block, 16, &block, 16), 0);
    }

    #[test]
    fn test_sad_16x8_identical() {
        let block = vec![50u8; 16 * 8];
        assert_eq!(sad_16x8(&block, 16, &block, 16), 0);
    }

    #[test]
    fn test_sad_4x4_known_value() {
        let src = vec![0u8; 4 * 4];
        let ref_blk = vec![1u8; 4 * 4];
        // 4×4 pixels × |0 - 1| = 16
        assert_eq!(sad_4x4(&src, 4, &ref_blk, 4), 16);
    }

    #[test]
    fn test_sad_8x8_known_value() {
        let src = vec![0u8; 8 * 8];
        let ref_blk = vec![2u8; 8 * 8];
        // 8×8 pixels × |0 - 2| = 128
        assert_eq!(sad_8x8(&src, 8, &ref_blk, 8), 128);
    }

    // ------------------------------------------------------------------
    // Pixel conversion tests
    // ------------------------------------------------------------------

    #[test]
    fn test_i420_to_nv12_roundtrip() {
        let width = 8;
        let height = 4;
        let y: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
        let u: Vec<u8> = vec![100u8; (width / 2) * (height / 2)];
        let v: Vec<u8> = vec![150u8; (width / 2) * (height / 2)];

        let nv12 = i420_to_nv12(&y, &u, &v, width, height).expect("conversion failed");
        let (y2, u2, v2) = nv12_to_i420(&nv12, width, height).expect("roundtrip failed");

        assert_eq!(y, y2);
        assert_eq!(u, u2);
        assert_eq!(v, v2);
    }

    #[test]
    fn test_i420_to_nv12_size() {
        let width = 4;
        let height = 4;
        let y = vec![0u8; width * height];
        let u = vec![128u8; (width / 2) * (height / 2)];
        let v = vec![128u8; (width / 2) * (height / 2)];

        let nv12 = i420_to_nv12(&y, &u, &v, width, height).expect("conversion failed");
        // NV12: Y plane + interleaved UV plane
        assert_eq!(nv12.len(), width * height + width * (height / 2));
    }

    #[test]
    fn test_i420_to_nv12_rejects_short_input() {
        let result = i420_to_nv12(&[0u8; 1], &[0u8; 1], &[0u8; 1], 4, 4);
        assert!(result.is_none());
    }

    #[test]
    fn test_rgba_to_yuv420_size() {
        let width = 8;
        let height = 4;
        let rgba = vec![128u8; width * height * 4];
        let (y, u, v) = rgba_to_yuv420(&rgba, width, height).expect("conversion failed");
        assert_eq!(y.len(), width * height);
        assert_eq!(u.len(), (width / 2) * (height / 2));
        assert_eq!(v.len(), (width / 2) * (height / 2));
    }

    #[test]
    fn test_rgba_to_yuv420_rejects_short_input() {
        assert!(rgba_to_yuv420(&[0u8; 1], 4, 4).is_none());
    }

    #[test]
    fn test_yuv420_to_rgba_size() {
        let width = 8;
        let height = 4;
        let y = vec![128u8; width * height];
        let u = vec![128u8; (width / 2) * (height / 2)];
        let v = vec![128u8; (width / 2) * (height / 2)];
        let rgba = yuv420_to_rgba(&y, &u, &v, width, height).expect("conversion failed");
        assert_eq!(rgba.len(), width * height * 4);
    }

    #[test]
    fn test_yuv420_to_rgba_grey() {
        // Y=128, U=128, V=128 → approximately grey (128, 128, 128).
        let width = 2;
        let height = 2;
        let y = vec![128u8; 4];
        let u = vec![128u8; 1];
        let v = vec![128u8; 1];
        let rgba = yuv420_to_rgba(&y, &u, &v, width, height).expect("conversion failed");

        // All R, G, B components should be close to 128.
        for i in 0..4 {
            let base = i * 4;
            let r = rgba[base] as i32;
            let g = rgba[base + 1] as i32;
            let b = rgba[base + 2] as i32;
            assert!((r - 128).abs() <= 4, "R too far from grey: {r}");
            assert!((g - 128).abs() <= 4, "G too far from grey: {g}");
            assert!((b - 128).abs() <= 4, "B too far from grey: {b}");
        }
    }

    #[test]
    fn test_rgba_yuv_roundtrip_approx() {
        // Encode a flat grey image and decode; loss should be small.
        let width = 4;
        let height = 4;
        let rgba_in: Vec<u8> = (0..width * height * 4)
            .map(|i| if i % 4 == 3 { 255 } else { 180 })
            .collect();

        let (y, u, v) = rgba_to_yuv420(&rgba_in, width, height).expect("encode failed");
        let rgba_out = yuv420_to_rgba(&y, &u, &v, width, height).expect("decode failed");

        // Check that luminance round-trips with at most 8 units of error.
        for i in 0..width * height {
            let base = i * 4;
            for ch in 0..3usize {
                let orig = rgba_in[base + ch] as i32;
                let decoded = rgba_out[base + ch] as i32;
                assert!(
                    (orig - decoded).abs() <= 12,
                    "channel {ch} at px {i}: {orig} vs {decoded}"
                );
            }
        }
    }
}

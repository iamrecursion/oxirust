#![allow(dead_code)]
//! Block transpose primitives for matrix and image data.
//!
//! Transposition is a fundamental building block for SIMD-accelerated video
//! codecs (column-wise DCT passes), colour-space conversion with interleaved
//! layouts, and tiled image processing. This module provides in-place and
//! out-of-place 4x4, 8x8, and arbitrary `NxM` transposes for both `u8` and
//! `i16` elements.

// ---------------------------------------------------------------------------
// Small fixed-size transposes
// ---------------------------------------------------------------------------

/// Transpose a 4x4 block of `i16` values in-place.
///
/// `data` must have exactly 16 elements laid out in row-major order.
///
/// # Panics
///
/// Panics if `data.len() != 16`.
pub fn transpose_4x4_i16(data: &mut [i16]) {
    assert_eq!(data.len(), 16);
    // (r,c) -> (c,r)
    for r in 0..4 {
        for c in (r + 1)..4 {
            data.swap(r * 4 + c, c * 4 + r);
        }
    }
}

/// Transpose a 4x4 block of `u8` values in-place.
///
/// `data` must have exactly 16 elements laid out in row-major order.
///
/// # Panics
///
/// Panics if `data.len() != 16`.
pub fn transpose_4x4_u8(data: &mut [u8]) {
    assert_eq!(data.len(), 16);
    for r in 0..4u8 {
        for c in (r + 1)..4u8 {
            data.swap(r as usize * 4 + c as usize, c as usize * 4 + r as usize);
        }
    }
}

/// Transpose an 8x8 block of `i16` values in-place.
///
/// `data` must have exactly 64 elements laid out in row-major order.
///
/// # Panics
///
/// Panics if `data.len() != 64`.
pub fn transpose_8x8_i16(data: &mut [i16]) {
    assert_eq!(data.len(), 64);
    for r in 0..8 {
        for c in (r + 1)..8 {
            data.swap(r * 8 + c, c * 8 + r);
        }
    }
}

/// Transpose an 8x8 block of `u8` values in-place.
///
/// `data` must have exactly 64 elements laid out in row-major order.
///
/// # Panics
///
/// Panics if `data.len() != 64`.
pub fn transpose_8x8_u8(data: &mut [u8]) {
    assert_eq!(data.len(), 64);
    for r in 0..8 {
        for c in (r + 1)..8 {
            data.swap(r * 8 + c, c * 8 + r);
        }
    }
}

// ---------------------------------------------------------------------------
// Generic NxM out-of-place transpose
// ---------------------------------------------------------------------------

/// Out-of-place transpose of an `rows x cols` `i16` matrix.
///
/// Reads `src` in row-major order (rows x cols) and writes `dst` in
/// row-major order (cols x rows).
///
/// # Panics
///
/// Panics if `src.len() < rows * cols` or `dst.len() < rows * cols`.
pub fn transpose_i16(src: &[i16], dst: &mut [i16], rows: usize, cols: usize) {
    let total = rows * cols;
    assert!(src.len() >= total);
    assert!(dst.len() >= total);
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// Out-of-place transpose of an `rows x cols` `u8` matrix.
///
/// # Panics
///
/// Panics if `src.len() < rows * cols` or `dst.len() < rows * cols`.
pub fn transpose_u8(src: &[u8], dst: &mut [u8], rows: usize, cols: usize) {
    let total = rows * cols;
    assert!(src.len() >= total);
    assert!(dst.len() >= total);
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// Out-of-place transpose of an `rows x cols` `f32` matrix.
///
/// # Panics
///
/// Panics if `src.len() < rows * cols` or `dst.len() < rows * cols`.
pub fn transpose_f32(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    let total = rows * cols;
    assert!(src.len() >= total);
    assert!(dst.len() >= total);
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

// ---------------------------------------------------------------------------
// Tiled transpose (cache-friendly for large matrices)
// ---------------------------------------------------------------------------

/// Cache-block size used by the tiled transpose.
const TILE: usize = 16;

/// Cache-friendly tiled transpose of an `rows x cols` `i16` matrix.
///
/// This is identical in result to [`transpose_i16`] but processes the
/// matrix in `TILE x TILE` sub-blocks to reduce cache misses on large
/// matrices.
///
/// # Panics
///
/// Panics if `src.len() < rows * cols` or `dst.len() < rows * cols`.
pub fn transpose_tiled_i16(src: &[i16], dst: &mut [i16], rows: usize, cols: usize) {
    let total = rows * cols;
    assert!(src.len() >= total);
    assert!(dst.len() >= total);

    let mut r0 = 0;
    while r0 < rows {
        let r1 = (r0 + TILE).min(rows);
        let mut c0 = 0;
        while c0 < cols {
            let c1 = (c0 + TILE).min(cols);
            for r in r0..r1 {
                for c in c0..c1 {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
            c0 += TILE;
        }
        r0 += TILE;
    }
}

/// Check whether applying transpose twice yields the identity.
///
/// Returns `true` if `rows * cols` elements in `data` survive a
/// round-trip transpose.
#[must_use]
pub fn is_transpose_involution_i16(data: &[i16], rows: usize, cols: usize) -> bool {
    let total = rows * cols;
    if data.len() < total {
        return false;
    }
    let mut tmp = vec![0i16; total];
    let mut back = vec![0i16; total];
    transpose_i16(data, &mut tmp, rows, cols);
    transpose_i16(&tmp, &mut back, cols, rows);
    data[..total] == back[..total]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_4x4_i16() {
        #[rustfmt::skip]
        let mut data: Vec<i16> = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12,
           13,14,15,16,
        ];
        transpose_4x4_i16(&mut data);
        #[rustfmt::skip]
        let expected: Vec<i16> = vec![
            1, 5, 9,13,
            2, 6,10,14,
            3, 7,11,15,
            4, 8,12,16,
        ];
        assert_eq!(data, expected);
    }

    #[test]
    fn test_transpose_4x4_u8() {
        #[rustfmt::skip]
        let mut data: Vec<u8> = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12,
           13,14,15,16,
        ];
        transpose_4x4_u8(&mut data);
        assert_eq!(data[1], 5);
        assert_eq!(data[4], 2);
    }

    #[test]
    fn test_transpose_8x8_i16() {
        let mut data: Vec<i16> = (0..64).collect();
        transpose_8x8_i16(&mut data);
        // Element at (0,1) which was 1 should move to (1,0) = index 8
        assert_eq!(data[8], 1);
        // Element at (1,0) which was 8 should move to (0,1) = index 1
        assert_eq!(data[1], 8);
    }

    #[test]
    fn test_transpose_8x8_u8_identity_diagonal() {
        // Diagonal elements should remain in place
        let mut data = vec![0u8; 64];
        for i in 0..8 {
            data[i * 8 + i] = (i + 1) as u8;
        }
        let orig_diag: Vec<u8> = (0..8).map(|i| data[i * 8 + i]).collect();
        transpose_8x8_u8(&mut data);
        let new_diag: Vec<u8> = (0..8).map(|i| data[i * 8 + i]).collect();
        assert_eq!(orig_diag, new_diag);
    }

    #[test]
    fn test_transpose_i16_rect() {
        // 2x3 -> 3x2
        let src: Vec<i16> = vec![1, 2, 3, 4, 5, 6];
        let mut dst = vec![0i16; 6];
        transpose_i16(&src, &mut dst, 2, 3);
        // Expected (3x2): [1,4, 2,5, 3,6]
        assert_eq!(dst, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_transpose_u8_rect() {
        let src: Vec<u8> = vec![10, 20, 30, 40, 50, 60];
        let mut dst = vec![0u8; 6];
        transpose_u8(&src, &mut dst, 2, 3);
        assert_eq!(dst, vec![10, 40, 20, 50, 30, 60]);
    }

    #[test]
    fn test_transpose_f32_square() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0f32; 4];
        transpose_f32(&src, &mut dst, 2, 2);
        assert_eq!(dst, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_tiled_matches_naive() {
        let rows = 37;
        let cols = 23;
        let src: Vec<i16> = (0..(rows * cols) as i16).collect();
        let mut dst_naive = vec![0i16; rows * cols];
        let mut dst_tiled = vec![0i16; rows * cols];
        transpose_i16(&src, &mut dst_naive, rows, cols);
        transpose_tiled_i16(&src, &mut dst_tiled, rows, cols);
        assert_eq!(dst_naive, dst_tiled);
    }

    #[test]
    fn test_is_involution() {
        let data: Vec<i16> = (0..20).collect();
        assert!(is_transpose_involution_i16(&data, 4, 5));
    }

    #[test]
    fn test_transpose_1x1() {
        let src = [42i16];
        let mut dst = [0i16];
        transpose_i16(&src, &mut dst, 1, 1);
        assert_eq!(dst[0], 42);
    }

    #[test]
    fn test_transpose_f32_rect() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f32; 6];
        transpose_f32(&src, &mut dst, 3, 2);
        // 3x2 -> 2x3 : [1,3,5, 2,4,6]
        assert_eq!(dst, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_transpose_double_is_identity() {
        let src: Vec<i16> = (0..12).collect();
        let mut tmp = vec![0i16; 12];
        let mut back = vec![0i16; 12];
        transpose_i16(&src, &mut tmp, 3, 4);
        transpose_i16(&tmp, &mut back, 4, 3);
        assert_eq!(src, back);
    }
}

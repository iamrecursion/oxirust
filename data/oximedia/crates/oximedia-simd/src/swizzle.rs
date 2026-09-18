#![allow(dead_code)]
//! SIMD lane swizzle, permute, and shuffle operations.
//!
//! This module provides portable implementations of common SIMD
//! data rearrangement patterns used in multimedia processing:
//! - Lane swizzle (reorder elements within a vector)
//! - Interleave/deinterleave for planar/packed format conversions
//! - Broadcast (splat a single value across all lanes)
//! - Rotate and shift lane operations
//! - Zip/unzip for merging/splitting vectors

/// Number of lanes in a standard processing vector.
const LANES_8: usize = 8;
/// Number of lanes in a wide processing vector.
const LANES_16: usize = 16;

/// Swizzle pattern descriptor for reordering lanes.
///
/// Each index in the pattern array specifies which source lane
/// should be placed at that destination position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwizzlePattern {
    /// The index mapping: `indices[dst] = src`.
    pub indices: Vec<usize>,
}

impl SwizzlePattern {
    /// Create a new swizzle pattern from the given index mapping.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of range for the pattern length.
    pub fn new(indices: Vec<usize>) -> Self {
        let len = indices.len();
        for &idx in &indices {
            assert!(
                idx < len,
                "swizzle index {idx} out of range for length {len}"
            );
        }
        Self { indices }
    }

    /// Create an identity (no-op) pattern of the given length.
    pub fn identity(len: usize) -> Self {
        Self {
            indices: (0..len).collect(),
        }
    }

    /// Create a reverse pattern of the given length.
    pub fn reverse(len: usize) -> Self {
        Self {
            indices: (0..len).rev().collect(),
        }
    }

    /// Create a broadcast pattern that replicates lane `src` across `len` lanes.
    pub fn broadcast(src: usize, len: usize) -> Self {
        assert!(
            src < len,
            "broadcast source {src} out of range for length {len}"
        );
        Self {
            indices: vec![src; len],
        }
    }

    /// Create a rotate-left pattern by `amount` positions.
    pub fn rotate_left(len: usize, amount: usize) -> Self {
        Self {
            indices: (0..len).map(|i| (i + amount) % len).collect(),
        }
    }

    /// Create a rotate-right pattern by `amount` positions.
    pub fn rotate_right(len: usize, amount: usize) -> Self {
        Self {
            indices: (0..len).map(|i| (i + len - amount % len) % len).collect(),
        }
    }

    /// Return the number of lanes in this pattern.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Return whether this pattern is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Apply a swizzle pattern to a slice of `u8` values.
///
/// # Errors
///
/// Returns `None` if the data length does not match the pattern length.
pub fn swizzle_u8(data: &[u8], pattern: &SwizzlePattern) -> Option<Vec<u8>> {
    if data.len() < pattern.len() {
        return None;
    }
    Some(pattern.indices.iter().map(|&idx| data[idx]).collect())
}

/// Apply a swizzle pattern to a slice of `i16` values.
///
/// # Errors
///
/// Returns `None` if the data length does not match the pattern length.
pub fn swizzle_i16(data: &[i16], pattern: &SwizzlePattern) -> Option<Vec<i16>> {
    if data.len() < pattern.len() {
        return None;
    }
    Some(pattern.indices.iter().map(|&idx| data[idx]).collect())
}

/// Apply a swizzle pattern to a slice of `f32` values.
///
/// # Errors
///
/// Returns `None` if the data length does not match the pattern length.
#[allow(clippy::cast_precision_loss)]
pub fn swizzle_f32(data: &[f32], pattern: &SwizzlePattern) -> Option<Vec<f32>> {
    if data.len() < pattern.len() {
        return None;
    }
    Some(pattern.indices.iter().map(|&idx| data[idx]).collect())
}

/// Interleave two vectors of equal length: `[a0,a1,a2,a3]` + `[b0,b1,b2,b3]` -> `[a0,b0,a1,b1,a2,b2,a3,b3]`.
pub fn interleave_u8(a: &[u8], b: &[u8]) -> Vec<u8> {
    let len = a.len().min(b.len());
    let mut result = Vec::with_capacity(len * 2);
    for i in 0..len {
        result.push(a[i]);
        result.push(b[i]);
    }
    result
}

/// Deinterleave a packed vector into two separate vectors.
///
/// `[a0,b0,a1,b1,...]` -> `([a0,a1,...], [b0,b1,...])`.
pub fn deinterleave_u8(data: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let half = data.len() / 2;
    let mut a = Vec::with_capacity(half);
    let mut b = Vec::with_capacity(half);
    for chunk in data.chunks_exact(2) {
        a.push(chunk[0]);
        b.push(chunk[1]);
    }
    (a, b)
}

/// Broadcast a single `u8` value to fill a vector of `count` lanes.
pub fn broadcast_u8(value: u8, count: usize) -> Vec<u8> {
    vec![value; count]
}

/// Broadcast a single `f32` value to fill a vector of `count` lanes.
#[allow(clippy::cast_precision_loss)]
pub fn broadcast_f32(value: f32, count: usize) -> Vec<f32> {
    vec![value; count]
}

/// Zip two vectors: pair elements at the same index.
///
/// `[a0,a1]` + `[b0,b1]` -> `[(a0,b0),(a1,b1)]`.
pub fn zip_u8(a: &[u8], b: &[u8]) -> Vec<(u8, u8)> {
    a.iter().copied().zip(b.iter().copied()).collect()
}

/// Unzip a vector of pairs into two separate vectors.
pub fn unzip_u8(pairs: &[(u8, u8)]) -> (Vec<u8>, Vec<u8>) {
    pairs.iter().copied().unzip()
}

/// Rotate a slice left by `amount` positions, returning a new vector.
pub fn rotate_left_u8(data: &[u8], amount: usize) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let n = data.len();
    let shift = amount % n;
    let mut result = Vec::with_capacity(n);
    result.extend_from_slice(&data[shift..]);
    result.extend_from_slice(&data[..shift]);
    result
}

/// Rotate a slice right by `amount` positions, returning a new vector.
pub fn rotate_right_u8(data: &[u8], amount: usize) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let n = data.len();
    let shift = amount % n;
    rotate_left_u8(data, n - shift)
}

/// Extract the high half of a vector (second half of elements).
pub fn extract_high_u8(data: &[u8]) -> Vec<u8> {
    let mid = data.len() / 2;
    data[mid..].to_vec()
}

/// Extract the low half of a vector (first half of elements).
pub fn extract_low_u8(data: &[u8]) -> Vec<u8> {
    let mid = data.len() / 2;
    data[..mid].to_vec()
}

/// Concatenate two half-vectors into a full vector.
pub fn concat_halves_u8(low: &[u8], high: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(low.len() + high.len());
    result.extend_from_slice(low);
    result.extend_from_slice(high);
    result
}

/// Blend two vectors using a boolean mask.
///
/// For each position, if `mask[i]` is `true`, take from `b`; otherwise from `a`.
pub fn blend_u8(a: &[u8], b: &[u8], mask: &[bool]) -> Vec<u8> {
    let len = a.len().min(b.len()).min(mask.len());
    (0..len)
        .map(|i| if mask[i] { b[i] } else { a[i] })
        .collect()
}

/// Apply a gather operation: read elements from `data` at positions specified by `indices`.
pub fn gather_u8(data: &[u8], indices: &[usize]) -> Option<Vec<u8>> {
    let max_idx = data.len();
    if indices.iter().any(|&i| i >= max_idx) {
        return None;
    }
    Some(indices.iter().map(|&i| data[i]).collect())
}

/// Compact (pack) non-zero elements to the front, returning count of non-zero elements.
pub fn compact_nonzero_u8(data: &[u8]) -> (Vec<u8>, usize) {
    let mut result = Vec::with_capacity(data.len());
    let mut count = 0;
    for &v in data {
        if v != 0 {
            result.push(v);
            count += 1;
        }
    }
    // Pad remaining with zeros
    result.resize(data.len(), 0);
    (result, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swizzle_pattern_identity() {
        let pat = SwizzlePattern::identity(LANES_8);
        assert_eq!(pat.len(), LANES_8);
        let data: Vec<u8> = (0..8).collect();
        let result = swizzle_u8(&data, &pat).expect("should succeed in test");
        assert_eq!(result, data);
    }

    #[test]
    fn test_swizzle_pattern_reverse() {
        let pat = SwizzlePattern::reverse(4);
        let data = vec![10u8, 20, 30, 40];
        let result = swizzle_u8(&data, &pat).expect("should succeed in test");
        assert_eq!(result, vec![40, 30, 20, 10]);
    }

    #[test]
    fn test_swizzle_pattern_broadcast() {
        let pat = SwizzlePattern::broadcast(2, 4);
        let data = vec![10u8, 20, 30, 40];
        let result = swizzle_u8(&data, &pat).expect("should succeed in test");
        assert_eq!(result, vec![30, 30, 30, 30]);
    }

    #[test]
    fn test_swizzle_pattern_rotate_left() {
        let pat = SwizzlePattern::rotate_left(4, 1);
        let data = vec![1u8, 2, 3, 4];
        let result = swizzle_u8(&data, &pat).expect("should succeed in test");
        assert_eq!(result, vec![2, 3, 4, 1]);
    }

    #[test]
    fn test_swizzle_pattern_rotate_right() {
        let pat = SwizzlePattern::rotate_right(4, 1);
        let data = vec![1u8, 2, 3, 4];
        let result = swizzle_u8(&data, &pat).expect("should succeed in test");
        assert_eq!(result, vec![4, 1, 2, 3]);
    }

    #[test]
    fn test_swizzle_i16() {
        let pat = SwizzlePattern::reverse(4);
        let data = vec![100i16, 200, 300, 400];
        let result = swizzle_i16(&data, &pat).expect("should succeed in test");
        assert_eq!(result, vec![400, 300, 200, 100]);
    }

    #[test]
    fn test_swizzle_f32() {
        let pat = SwizzlePattern::identity(3);
        let data = vec![1.0f32, 2.0, 3.0];
        let result = swizzle_f32(&data, &pat).expect("should succeed in test");
        assert_eq!(result, data);
    }

    #[test]
    fn test_swizzle_too_short() {
        let pat = SwizzlePattern::identity(8);
        let data = vec![1u8, 2, 3];
        assert!(swizzle_u8(&data, &pat).is_none());
    }

    #[test]
    fn test_interleave_deinterleave_roundtrip() {
        let a = vec![1u8, 2, 3, 4];
        let b = vec![10u8, 20, 30, 40];
        let interleaved = interleave_u8(&a, &b);
        assert_eq!(interleaved, vec![1, 10, 2, 20, 3, 30, 4, 40]);
        let (ra, rb) = deinterleave_u8(&interleaved);
        assert_eq!(ra, a);
        assert_eq!(rb, b);
    }

    #[test]
    fn test_broadcast_u8() {
        let result = broadcast_u8(42, LANES_16);
        assert_eq!(result.len(), LANES_16);
        assert!(result.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_broadcast_f32() {
        let result = broadcast_f32(3.14, 4);
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!((v - 3.14).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_zip_unzip_roundtrip() {
        let a = vec![1u8, 2, 3];
        let b = vec![4u8, 5, 6];
        let zipped = zip_u8(&a, &b);
        assert_eq!(zipped, vec![(1, 4), (2, 5), (3, 6)]);
        let (ua, ub) = unzip_u8(&zipped);
        assert_eq!(ua, a);
        assert_eq!(ub, b);
    }

    #[test]
    fn test_rotate_left_right_u8() {
        let data = vec![1u8, 2, 3, 4, 5];
        let left = rotate_left_u8(&data, 2);
        assert_eq!(left, vec![3, 4, 5, 1, 2]);
        let right = rotate_right_u8(&data, 2);
        assert_eq!(right, vec![4, 5, 1, 2, 3]);
    }

    #[test]
    fn test_extract_halves_and_concat() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let low = extract_low_u8(&data);
        let high = extract_high_u8(&data);
        assert_eq!(low, vec![1, 2, 3, 4]);
        assert_eq!(high, vec![5, 6, 7, 8]);
        let combined = concat_halves_u8(&low, &high);
        assert_eq!(combined, data);
    }

    #[test]
    fn test_blend_u8() {
        let a = vec![1u8, 2, 3, 4];
        let b = vec![10u8, 20, 30, 40];
        let mask = vec![false, true, false, true];
        let result = blend_u8(&a, &b, &mask);
        assert_eq!(result, vec![1, 20, 3, 40]);
    }

    #[test]
    fn test_gather_u8() {
        let data = vec![10u8, 20, 30, 40, 50];
        let indices = vec![4, 0, 2];
        let result = gather_u8(&data, &indices).expect("should succeed in test");
        assert_eq!(result, vec![50, 10, 30]);
    }

    #[test]
    fn test_gather_out_of_bounds() {
        let data = vec![1u8, 2, 3];
        let indices = vec![0, 5];
        assert!(gather_u8(&data, &indices).is_none());
    }

    #[test]
    fn test_compact_nonzero() {
        let data = vec![0u8, 5, 0, 3, 7, 0, 1, 0];
        let (result, count) = compact_nonzero_u8(&data);
        assert_eq!(count, 4);
        assert_eq!(&result[..count], &[5, 3, 7, 1]);
        assert!(result[count..].iter().all(|&v| v == 0));
    }

    #[test]
    fn test_rotate_empty() {
        let empty: Vec<u8> = Vec::new();
        assert!(rotate_left_u8(&empty, 3).is_empty());
        assert!(rotate_right_u8(&empty, 3).is_empty());
    }
}

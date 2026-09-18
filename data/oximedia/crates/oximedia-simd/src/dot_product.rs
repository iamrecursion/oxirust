#![allow(dead_code)]
//! Dot product operations for SIMD-accelerated signal processing.
//!
//! Provides scalar-fallback implementations of dot product variants
//! commonly used in multimedia codecs and DSP:
//! - Integer dot products (u8, i16, i32)
//! - Floating-point dot products (f32, f64)
//! - Weighted dot products
//! - Saturating dot products for fixed-point arithmetic
//! - Block dot products for matrix/convolution kernels

/// Compute the dot product of two `u8` slices, accumulating into `u64`.
///
/// Returns `None` if the slices have different lengths.
pub fn dot_u8(a: &[u8], b: &[u8]) -> Option<u64> {
    if a.len() != b.len() {
        return None;
    }
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| u64::from(x) * u64::from(y))
        .sum();
    Some(sum)
}

/// Compute the dot product of two `i16` slices, accumulating into `i64`.
///
/// Returns `None` if the slices have different lengths.
pub fn dot_i16(a: &[i16], b: &[i16]) -> Option<i64> {
    if a.len() != b.len() {
        return None;
    }
    let sum: i64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum();
    Some(sum)
}

/// Compute the dot product of two `i32` slices, accumulating into `i64`.
///
/// Returns `None` if the slices have different lengths.
pub fn dot_i32(a: &[i32], b: &[i32]) -> Option<i64> {
    if a.len() != b.len() {
        return None;
    }
    let sum: i64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum();
    Some(sum)
}

/// Compute the dot product of two `f32` slices.
///
/// Returns `None` if the slices have different lengths.
#[allow(clippy::cast_precision_loss)]
pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    Some(sum)
}

/// Compute the dot product of two `f64` slices.
///
/// Returns `None` if the slices have different lengths.
#[allow(clippy::cast_precision_loss)]
pub fn dot_f64(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let sum: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    Some(sum)
}

/// Compute a weighted dot product: `sum(a[i] * b[i] * weight[i])` for `f32`.
///
/// Returns `None` if any slice has a different length.
#[allow(clippy::cast_precision_loss)]
pub fn weighted_dot_f32(a: &[f32], b: &[f32], weights: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.len() != weights.len() {
        return None;
    }
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .zip(weights.iter())
        .map(|((&x, &y), &w)| x * y * w)
        .sum();
    Some(sum)
}

/// Compute a saturating dot product of two `i16` slices, clamped to `i32` range.
///
/// This is useful in fixed-point DSP where overflow must be avoided.
///
/// Returns `None` if the slices have different lengths.
pub fn saturating_dot_i16(a: &[i16], b: &[i16]) -> Option<i32> {
    if a.len() != b.len() {
        return None;
    }
    let mut acc: i32 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let product = i32::from(x) * i32::from(y);
        acc = acc.saturating_add(product);
    }
    Some(acc)
}

/// Compute the sum of absolute differences (SAD) between two `u8` slices.
///
/// Equivalent to `sum(|a[i] - b[i]|)`.
///
/// Returns `None` if the slices have different lengths.
pub fn sad_u8(a: &[u8], b: &[u8]) -> Option<u32> {
    if a.len() != b.len() {
        return None;
    }
    let sum: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| u32::from(x.abs_diff(y)))
        .sum();
    Some(sum)
}

/// Compute the sum of squared differences (SSD) between two `u8` slices.
///
/// Equivalent to `sum((a[i] - b[i])^2)`.
///
/// Returns `None` if the slices have different lengths.
pub fn ssd_u8(a: &[u8], b: &[u8]) -> Option<u64> {
    if a.len() != b.len() {
        return None;
    }
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = i32::from(x) - i32::from(y);
            #[allow(clippy::cast_sign_loss)]
            let sq = (diff * diff) as u64;
            sq
        })
        .sum();
    Some(sum)
}

/// Compute dot products for multiple blocks at once.
///
/// Given a single kernel `b` and multiple data blocks in `blocks`,
/// compute the dot product of each block with `b`.
///
/// Returns `None` if any block length does not match `b`.
pub fn block_dot_i16(blocks: &[&[i16]], b: &[i16]) -> Option<Vec<i64>> {
    let mut results = Vec::with_capacity(blocks.len());
    for block in blocks {
        results.push(dot_i16(block, b)?);
    }
    Some(results)
}

/// Compute the normalized dot product (cosine similarity numerator) of two `f32` vectors.
///
/// Returns `(dot, mag_a, mag_b)` where `dot = a . b`, `mag_a = ||a||`, `mag_b = ||b||`.
///
/// Returns `None` if the slices have different lengths.
#[allow(clippy::cast_precision_loss)]
pub fn normalized_dot_f32(a: &[f32], b: &[f32]) -> Option<(f32, f32, f32)> {
    if a.len() != b.len() {
        return None;
    }
    let mut dot: f32 = 0.0;
    let mut mag_a: f32 = 0.0;
    let mut mag_b: f32 = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }
    Some((dot, mag_a.sqrt(), mag_b.sqrt()))
}

/// Cosine similarity between two `f32` vectors.
///
/// Returns a value in `[-1.0, 1.0]`, or `None` for mismatched lengths
/// or zero-magnitude vectors.
#[allow(clippy::cast_precision_loss)]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    let (dot, mag_a, mag_b) = normalized_dot_f32(a, b)?;
    let denom = mag_a * mag_b;
    if denom < f32::EPSILON {
        return None;
    }
    Some(dot / denom)
}

/// Multiply-accumulate: `acc += a[i] * b[i]` for `f32`, writing into `acc`.
///
/// Returns updated accumulator value.
#[allow(clippy::cast_precision_loss)]
pub fn mac_f32(acc: f32, a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut result = acc;
    for i in 0..len {
        result += a[i] * b[i];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_u8_basic() {
        let a = [1u8, 2, 3, 4];
        let b = [5u8, 6, 7, 8];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        assert_eq!(dot_u8(&a, &b), Some(70));
    }

    #[test]
    fn test_dot_u8_mismatched() {
        assert_eq!(dot_u8(&[1, 2], &[3]), None);
    }

    #[test]
    fn test_dot_i16_basic() {
        let a = [10i16, -20, 30];
        let b = [1i16, 2, 3];
        // 10 + (-40) + 90 = 60
        assert_eq!(dot_i16(&a, &b), Some(60));
    }

    #[test]
    fn test_dot_i32_basic() {
        let a = [1000i32, 2000, -3000];
        let b = [4i32, 5, 6];
        // 4000 + 10000 + (-18000) = -4000
        assert_eq!(dot_i32(&a, &b), Some(-4000));
    }

    #[test]
    fn test_dot_f32_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let result = dot_f32(&a, &b).expect("should succeed in test");
        assert!((result - 32.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dot_f64_basic() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 5.0, 6.0];
        let result = dot_f64(&a, &b).expect("should succeed in test");
        assert!((result - 32.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_dot_f32() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let w = [1.0f32, 0.5, 2.0];
        // 1*4*1 + 2*5*0.5 + 3*6*2 = 4 + 5 + 36 = 45
        let result = weighted_dot_f32(&a, &b, &w).expect("should succeed in test");
        assert!((result - 45.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_weighted_dot_mismatched() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let w = [1.0f32];
        assert_eq!(weighted_dot_f32(&a, &b, &w), None);
    }

    #[test]
    fn test_saturating_dot_i16() {
        let a = [i16::MAX, i16::MAX];
        let b = [i16::MAX, i16::MAX];
        let result = saturating_dot_i16(&a, &b).expect("should succeed in test");
        // Should not overflow i32
        let expected = i32::from(i16::MAX) * i32::from(i16::MAX) * 2;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sad_u8() {
        let a = [10u8, 20, 30, 40];
        let b = [12u8, 18, 35, 38];
        // |10-12| + |20-18| + |30-35| + |40-38| = 2+2+5+2 = 11
        assert_eq!(sad_u8(&a, &b), Some(11));
    }

    #[test]
    fn test_ssd_u8() {
        let a = [10u8, 20, 30, 40];
        let b = [12u8, 18, 35, 38];
        // 4 + 4 + 25 + 4 = 37
        assert_eq!(ssd_u8(&a, &b), Some(37));
    }

    #[test]
    fn test_block_dot_i16() {
        let b = [1i16, 2, 3];
        let block1: &[i16] = &[4, 5, 6];
        let block2: &[i16] = &[7, 8, 9];
        let blocks: Vec<&[i16]> = vec![block1, block2];
        let results = block_dot_i16(&blocks, &b).expect("should succeed in test");
        assert_eq!(results, vec![32, 50]);
    }

    #[test]
    fn test_cosine_similarity_parallel() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let sim = cosine_similarity_f32(&a, &b).expect("should succeed in test");
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        let sim = cosine_similarity_f32(&a, &b).expect("should succeed in test");
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = [0.0f32, 0.0];
        let b = [1.0f32, 2.0];
        assert!(cosine_similarity_f32(&a, &b).is_none());
    }

    #[test]
    fn test_mac_f32() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let result = mac_f32(10.0, &a, &b);
        // 10 + 4 + 10 + 18 = 42
        assert!((result - 42.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dot_empty() {
        let a: [u8; 0] = [];
        let b: [u8; 0] = [];
        assert_eq!(dot_u8(&a, &b), Some(0));
    }

    #[test]
    fn test_normalized_dot_f32() {
        let a = [3.0f32, 4.0];
        let b = [3.0f32, 4.0];
        let (dot, ma, mb) = normalized_dot_f32(&a, &b).expect("should succeed in test");
        assert!((dot - 25.0).abs() < f32::EPSILON);
        assert!((ma - 5.0).abs() < 1e-5);
        assert!((mb - 5.0).abs() < 1e-5);
    }
}

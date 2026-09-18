#![allow(dead_code)]
//! Parallel prefix sum (scan) operations for multimedia processing.
//!
//! Prefix sums are a fundamental parallel primitive used extensively in:
//! - Integral image computation for fast box filtering
//! - Histogram equalization (cumulative distribution)
//! - Bitstream offset calculation in entropy coding
//! - Parallel work distribution based on variable-length data
//!
//! This module provides inclusive and exclusive prefix sums for various
//! numeric types, as well as 2D prefix sums (integral images/summed area tables).

/// Compute the inclusive prefix sum of a `u32` slice.
///
/// `out[i] = data[0] + data[1] + ... + data[i]`
///
/// Returns a new vector with the prefix sums.
#[must_use]
pub fn inclusive_prefix_sum_u32(data: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc: u32 = 0;
    for &v in data {
        acc = acc.wrapping_add(v);
        out.push(acc);
    }
    out
}

/// Compute the exclusive prefix sum of a `u32` slice.
///
/// `out[i] = data[0] + data[1] + ... + data[i-1]` (out\[0\] = 0)
///
/// Returns a new vector with the prefix sums.
#[must_use]
pub fn exclusive_prefix_sum_u32(data: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc: u32 = 0;
    for &v in data {
        out.push(acc);
        acc = acc.wrapping_add(v);
    }
    out
}

/// Compute the inclusive prefix sum of a `u64` slice.
///
/// Returns a new vector with the prefix sums.
#[must_use]
pub fn inclusive_prefix_sum_u64(data: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc: u64 = 0;
    for &v in data {
        acc = acc.wrapping_add(v);
        out.push(acc);
    }
    out
}

/// Compute the exclusive prefix sum of a `u64` slice.
///
/// Returns a new vector with the prefix sums.
#[must_use]
pub fn exclusive_prefix_sum_u64(data: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc: u64 = 0;
    for &v in data {
        out.push(acc);
        acc = acc.wrapping_add(v);
    }
    out
}

/// Compute the inclusive prefix sum of an `f32` slice.
///
/// Returns a new vector with the prefix sums.
#[must_use]
pub fn inclusive_prefix_sum_f32(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc: f32 = 0.0;
    for &v in data {
        acc += v;
        out.push(acc);
    }
    out
}

/// Compute the exclusive prefix sum of an `f32` slice.
///
/// Returns a new vector with the prefix sums.
#[must_use]
pub fn exclusive_prefix_sum_f32(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc: f32 = 0.0;
    for &v in data {
        out.push(acc);
        acc += v;
    }
    out
}

/// Compute the inclusive prefix sum of an `i32` slice.
///
/// Returns a new vector with the prefix sums.
#[must_use]
pub fn inclusive_prefix_sum_i32(data: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc: i32 = 0;
    for &v in data {
        acc = acc.wrapping_add(v);
        out.push(acc);
    }
    out
}

/// Compute the inclusive prefix max of a `u32` slice.
///
/// `out[i] = max(data[0], data[1], ..., data[i])`
///
/// Returns a new vector with the running maximum.
#[must_use]
pub fn inclusive_prefix_max_u32(data: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(data.len());
    let mut current_max: u32 = 0;
    for &v in data {
        current_max = current_max.max(v);
        out.push(current_max);
    }
    out
}

/// Compute the inclusive prefix min of a `u32` slice.
///
/// `out[i] = min(data[0], data[1], ..., data[i])`
///
/// Returns a new vector with the running minimum.
#[must_use]
pub fn inclusive_prefix_min_u32(data: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(data.len());
    let mut current_min: u32 = u32::MAX;
    for &v in data {
        current_min = current_min.min(v);
        out.push(current_min);
    }
    out
}

/// Compute a 2D prefix sum (summed area table / integral image).
///
/// Given a row-major 2D array of dimensions `width x height`, computes the
/// integral image where each element is the sum of all values in the
/// rectangle from (0,0) to (row, col) inclusive.
///
/// The output is stored in a flat `Vec<u64>` of size `width * height`.
///
/// # Panics
///
/// Panics if `data.len() != width * height`.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn integral_image_u8(data: &[u8], width: usize, height: usize) -> Vec<u64> {
    assert_eq!(
        data.len(),
        width * height,
        "data length must equal width * height"
    );
    let mut sat = vec![0u64; width * height];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let val = u64::from(data[idx]);
            let left = if x > 0 { sat[idx - 1] } else { 0 };
            let above = if y > 0 { sat[idx - width] } else { 0 };
            let diag = if x > 0 && y > 0 {
                sat[idx - width - 1]
            } else {
                0
            };
            sat[idx] = val + left + above - diag;
        }
    }

    sat
}

/// Query a rectangular sum from a precomputed integral image (summed area table).
///
/// Returns the sum of all values in the rectangle from `(y1, x1)` to `(y2, x2)` inclusive.
///
/// # Panics
///
/// Panics if coordinates are out of bounds.
#[must_use]
pub fn query_integral_image(
    sat: &[u64],
    width: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
) -> u64 {
    assert!(x2 < width);
    assert!(y2 * width + x2 < sat.len());

    let br = sat[y2 * width + x2];
    let above = if y1 > 0 {
        sat[(y1 - 1) * width + x2]
    } else {
        0
    };
    let left = if x1 > 0 {
        sat[y2 * width + (x1 - 1)]
    } else {
        0
    };
    let diag = if x1 > 0 && y1 > 0 {
        sat[(y1 - 1) * width + (x1 - 1)]
    } else {
        0
    };

    br - above - left + diag
}

/// Compute a segmented inclusive prefix sum with segment boundaries.
///
/// Wherever `segment_starts[i]` is `true`, the accumulator resets to `data[i]`.
///
/// # Panics
///
/// Panics if `data.len() != segment_starts.len()`.
#[must_use]
pub fn segmented_prefix_sum_u32(data: &[u32], segment_starts: &[bool]) -> Vec<u32> {
    assert_eq!(
        data.len(),
        segment_starts.len(),
        "data and segment_starts must have equal length"
    );
    let mut out = Vec::with_capacity(data.len());
    let mut acc: u32 = 0;
    for (&v, &is_start) in data.iter().zip(segment_starts.iter()) {
        if is_start {
            acc = v;
        } else {
            acc = acc.wrapping_add(v);
        }
        out.push(acc);
    }
    out
}

/// In-place inclusive prefix sum on a mutable `u32` slice.
///
/// Modifies the slice in place so that `data[i]` becomes the sum of
/// the original `data[0..=i]`.
pub fn prefix_sum_inplace_u32(data: &mut [u32]) {
    let mut acc: u32 = 0;
    for val in data.iter_mut() {
        acc = acc.wrapping_add(*val);
        *val = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inclusive_prefix_sum_u32() {
        let data = vec![1, 2, 3, 4, 5];
        let result = inclusive_prefix_sum_u32(&data);
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_exclusive_prefix_sum_u32() {
        let data = vec![1, 2, 3, 4, 5];
        let result = exclusive_prefix_sum_u32(&data);
        assert_eq!(result, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn test_inclusive_prefix_sum_u64() {
        let data = vec![10u64, 20, 30];
        let result = inclusive_prefix_sum_u64(&data);
        assert_eq!(result, vec![10, 30, 60]);
    }

    #[test]
    fn test_exclusive_prefix_sum_u64() {
        let data = vec![10u64, 20, 30];
        let result = exclusive_prefix_sum_u64(&data);
        assert_eq!(result, vec![0, 10, 30]);
    }

    #[test]
    fn test_inclusive_prefix_sum_f32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = inclusive_prefix_sum_f32(&data);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_exclusive_prefix_sum_f32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = exclusive_prefix_sum_f32(&data);
        assert!((result[0]).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_inclusive_prefix_sum_i32() {
        let data = vec![1, -1, 2, -2, 3];
        let result = inclusive_prefix_sum_i32(&data);
        assert_eq!(result, vec![1, 0, 2, 0, 3]);
    }

    #[test]
    fn test_inclusive_prefix_max_u32() {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let result = inclusive_prefix_max_u32(&data);
        assert_eq!(result, vec![3, 3, 4, 4, 5, 9, 9, 9]);
    }

    #[test]
    fn test_inclusive_prefix_min_u32() {
        let data = vec![5, 3, 7, 2, 8, 1];
        let result = inclusive_prefix_min_u32(&data);
        assert_eq!(result, vec![5, 3, 3, 2, 2, 1]);
    }

    #[test]
    fn test_integral_image_u8() {
        // 3x3 image, all ones
        let data = vec![1u8; 9];
        let sat = integral_image_u8(&data, 3, 3);
        // The summed area table for all 1s is:
        // 1 2 3
        // 2 4 6
        // 3 6 9
        assert_eq!(sat[0], 1);
        assert_eq!(sat[1], 2);
        assert_eq!(sat[2], 3);
        assert_eq!(sat[3], 2);
        assert_eq!(sat[4], 4);
        assert_eq!(sat[8], 9);
    }

    #[test]
    fn test_query_integral_image() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let sat = integral_image_u8(&data, 3, 3);
        // Sum of center element (1,1) to (1,1) => 5
        let sum = query_integral_image(&sat, 3, 1, 1, 1, 1);
        assert_eq!(sum, 5);
        // Sum of full image
        let total = query_integral_image(&sat, 3, 0, 0, 2, 2);
        assert_eq!(total, 45);
        // Sum of top-left 2x2
        let tl = query_integral_image(&sat, 3, 0, 0, 1, 1);
        assert_eq!(tl, 12); // 1+2+4+5
    }

    #[test]
    fn test_segmented_prefix_sum_u32() {
        let data = vec![1, 2, 3, 10, 20, 30];
        let starts = vec![true, false, false, true, false, false];
        let result = segmented_prefix_sum_u32(&data, &starts);
        assert_eq!(result, vec![1, 3, 6, 10, 30, 60]);
    }

    #[test]
    fn test_prefix_sum_inplace_u32() {
        let mut data = vec![1, 2, 3, 4, 5];
        prefix_sum_inplace_u32(&mut data);
        assert_eq!(data, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_empty_prefix_sum() {
        let data: Vec<u32> = vec![];
        assert!(inclusive_prefix_sum_u32(&data).is_empty());
        assert!(exclusive_prefix_sum_u32(&data).is_empty());
    }

    #[test]
    fn test_single_element_prefix_sum() {
        let data = vec![42u32];
        assert_eq!(inclusive_prefix_sum_u32(&data), vec![42]);
        assert_eq!(exclusive_prefix_sum_u32(&data), vec![0]);
    }
}

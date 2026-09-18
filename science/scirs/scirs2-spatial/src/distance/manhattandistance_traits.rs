//! # ManhattanDistance - Trait Implementations
//!
//! This module contains trait implementations for `ManhattanDistance`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Distance`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;

use super::functions::prefetch_read;
use super::functions::Distance;
use super::types::ManhattanDistance;

impl<T: Float> Default for ManhattanDistance<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send + Sync> Distance<T> for ManhattanDistance<T> {
    fn distance(&self, a: &[T], b: &[T]) -> T {
        if a.len() != b.len() {
            return T::nan();
        }
        if let Some(result) = Self::try_simd_f64(a, b) {
            return result;
        }
        if let Some(result) = Self::try_simd_f32(a, b) {
            return result;
        }
        let len = a.len();
        let mut sum = T::zero();
        let chunks = len / 4;
        for i in 0..chunks {
            let base = i * 4;
            if base + 8 < len {
                let end_idx = (base + 8).min(len);
                prefetch_read(&a[base + 4..end_idx]);
                prefetch_read(&b[base + 4..end_idx]);
            }
            let diff0_abs = (a[base] - b[base]).abs();
            let diff1_abs = (a[base + 1] - b[base + 1]).abs();
            let diff2_abs = (a[base + 2] - b[base + 2]).abs();
            let diff3_abs = (a[base + 3] - b[base + 3]).abs();
            sum = sum + diff0_abs + diff1_abs + diff2_abs + diff3_abs;
        }
        for i in (chunks * 4)..len {
            sum = sum + (a[i] - b[i]).abs();
        }
        sum
    }
    fn min_distance_point_rectangle(&self, point: &[T], mins: &[T], maxes: &[T]) -> T {
        let mut sum = T::zero();
        for i in 0..point.len() {
            let coord = point[i];
            let min_val = mins[i];
            let max_val = maxes[i];
            let clamped = coord.max(min_val).min(max_val);
            let diff = (coord - clamped).abs();
            sum = sum + diff;
        }
        sum
    }
}

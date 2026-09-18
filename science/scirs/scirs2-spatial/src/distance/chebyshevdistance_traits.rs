//! # ChebyshevDistance - Trait Implementations
//!
//! This module contains trait implementations for `ChebyshevDistance`.
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
use super::types::ChebyshevDistance;

impl<T: Float> Default for ChebyshevDistance<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send + Sync> Distance<T> for ChebyshevDistance<T> {
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
        let mut max_diff = T::zero();
        let chunks = len / 4;
        for i in 0..chunks {
            let base = i * 4;
            if base + 8 < len {
                let end_idx = (base + 8).min(len);
                prefetch_read(&a[base + 4..end_idx]);
                prefetch_read(&b[base + 4..end_idx]);
            }
            let diff0 = (a[base] - b[base]).abs();
            let diff1 = (a[base + 1] - b[base + 1]).abs();
            let diff2 = (a[base + 2] - b[base + 2]).abs();
            let diff3 = (a[base + 3] - b[base + 3]).abs();
            let max01 = diff0.max(diff1);
            let max23 = diff2.max(diff3);
            let chunk_max = max01.max(max23);
            max_diff = max_diff.max(chunk_max);
        }
        for i in (chunks * 4)..len {
            let diff = (a[i] - b[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        max_diff
    }
    fn min_distance_point_rectangle(&self, point: &[T], mins: &[T], maxes: &[T]) -> T {
        let mut max_diff = T::zero();
        for i in 0..point.len() {
            let coord = point[i];
            let min_val = mins[i];
            let max_val = maxes[i];
            let clamped = coord.max(min_val).min(max_val);
            let diff = (coord - clamped).abs();
            max_diff = max_diff.max(diff);
        }
        max_diff
    }
}

//! # EuclideanDistance - Trait Implementations
//!
//! This module contains trait implementations for `EuclideanDistance`.
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
use super::types::EuclideanDistance;

impl<T: Float> Default for EuclideanDistance<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send + Sync> Distance<T> for EuclideanDistance<T> {
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
        #[allow(clippy::needless_range_loop)]
        for i in 0..chunks {
            let base = i * 4;
            if base + 8 < len {
                let end_idx = (base + 8).min(len);
                prefetch_read(&a[base + 4..end_idx]);
                prefetch_read(&b[base + 4..end_idx]);
                if base + 16 < len {
                    let far_end = (base + 16).min(len);
                    prefetch_read(&a[base + 8..far_end]);
                    prefetch_read(&b[base + 8..far_end]);
                }
            }
            let diff0 = a[base] - b[base];
            let diff1 = a[base + 1] - b[base + 1];
            let diff2 = a[base + 2] - b[base + 2];
            let diff3 = a[base + 3] - b[base + 3];
            let sq0 = diff0 * diff0;
            let sq1 = diff1 * diff1;
            let sq2 = diff2 * diff2;
            let sq3 = diff3 * diff3;
            let pair_sum0 = sq0 + sq1;
            let pair_sum1 = sq2 + sq3;
            let chunk_sum = pair_sum0 + pair_sum1;
            sum = sum + chunk_sum;
        }
        for i in (chunks * 4)..len {
            let diff = a[i] - b[i];
            sum = sum + diff * diff;
        }
        sum.sqrt()
    }
    fn min_distance_point_rectangle(&self, point: &[T], mins: &[T], maxes: &[T]) -> T {
        let mut sum = T::zero();
        for i in 0..point.len() {
            let coord = point[i];
            let min_val = mins[i];
            let max_val = maxes[i];
            let clamped = coord.max(min_val).min(max_val);
            let diff = coord - clamped;
            sum = sum + diff * diff;
        }
        sum.sqrt()
    }
}

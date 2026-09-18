//! # MinkowskiDistance - Trait Implementations
//!
//! This module contains trait implementations for `MinkowskiDistance`.
//!
//! ## Implemented Traits
//!
//! - `Distance`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;

use super::functions::prefetch_read;
use super::functions::Distance;
use super::types::MinkowskiDistance;

impl<T: Float + Send + Sync> Distance<T> for MinkowskiDistance<T> {
    fn distance(&self, a: &[T], b: &[T]) -> T {
        if a.len() != b.len() {
            return T::nan();
        }
        if self.p == T::one() {
            let mut sum = T::zero();
            for i in 0..a.len() {
                sum = sum + (a[i] - b[i]).abs();
            }
            sum
        } else if self.p == T::from(2.0).expect("Operation failed") {
            let mut sum = T::zero();
            for i in 0..a.len() {
                let diff = a[i] - b[i];
                sum = sum + diff * diff;
            }
            sum.sqrt()
        } else if self.p == T::infinity() {
            let mut max_diff = T::zero();
            for i in 0..a.len() {
                let diff = (a[i] - b[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            max_diff
        } else {
            let mut sum = T::zero();
            for i in 0..a.len() {
                sum = sum + (a[i] - b[i]).abs().powf(self.p);
            }
            sum.powf(T::one() / self.p)
        }
    }
    fn min_distance_point_rectangle(&self, point: &[T], mins: &[T], maxes: &[T]) -> T {
        if self.p == T::one() {
            let mut sum = T::zero();
            for i in 0..point.len() {
                if point[i] < mins[i] {
                    sum = sum + (mins[i] - point[i]);
                } else if point[i] > maxes[i] {
                    sum = sum + (point[i] - maxes[i]);
                }
            }
            sum
        } else if self.p == T::from(2.0).expect("Operation failed") {
            let mut sum = T::zero();
            for i in 0..point.len() {
                if point[i] < mins[i] {
                    let diff = mins[i] - point[i];
                    sum = sum + diff * diff;
                } else if point[i] > maxes[i] {
                    let diff = point[i] - maxes[i];
                    sum = sum + diff * diff;
                }
            }
            sum.sqrt()
        } else if self.p == T::infinity() {
            let mut max_diff = T::zero();
            for i in 0..point.len() {
                let diff = if point[i] < mins[i] {
                    mins[i] - point[i]
                } else if point[i] > maxes[i] {
                    point[i] - maxes[i]
                } else {
                    T::zero()
                };
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            max_diff
        } else {
            let mut sum = T::zero();
            for i in 0..point.len() {
                let diff = if point[i] < mins[i] {
                    mins[i] - point[i]
                } else if point[i] > maxes[i] {
                    point[i] - maxes[i]
                } else {
                    T::zero()
                };
                sum = sum + diff.powf(self.p);
            }
            sum.powf(T::one() / self.p)
        }
    }
}

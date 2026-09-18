//! SIMD trait definitions for type constraints
//!
//! This module defines the core `SimdOps` trait that enables SIMD operations
//! on numeric types throughout the scirs2-core library.

use num_traits::Float;
use std::ops::{Add, Div, Mul, Sub};

/// Trait for types that can be processed with SIMD operations
///
/// This trait provides the basic constraints required for types to participate
/// in SIMD-accelerated numerical computations. Currently implemented for `f32` and `f64`.
///
/// # Examples
///
/// ```ignore
/// use scirs2_core::simd::traits::SimdOps;
///
/// fn process_simd<T: SimdOps>(data: &[T]) {
///     // SIMD operations can be safely performed on T
/// }
/// ```
pub trait SimdOps:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
}

impl SimdOps for f32 {}
impl SimdOps for f64 {}

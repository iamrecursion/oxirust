//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::marker::PhantomData;

#[repr(align(64))]
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CacheAlignedBuffer<T> {
    data: Vec<T>,
}
impl<T> CacheAlignedBuffer<T> {
    #[inline(always)]
    #[must_use]
    pub fn new_with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    #[inline(always)]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    #[inline(always)]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.data.resize(new_len, value);
    }
}
/// Manhattan distance metric (L1 norm)
#[derive(Clone, Debug)]
pub struct ManhattanDistance<T: Float>(PhantomData<T>);
impl<T: Float> ManhattanDistance<T> {
    /// Create a new Manhattan distance metric
    pub fn new() -> Self {
        ManhattanDistance(PhantomData)
    }
    /// Try SIMD path for f64 arrays
    #[inline]
    pub(super) fn try_simd_f64(a: &[T], b: &[T]) -> Option<T> {
        if std::mem::size_of::<T>() != std::mem::size_of::<f64>() {
            return None;
        }
        unsafe {
            let a_ptr = a.as_ptr() as *const f64;
            let b_ptr = b.as_ptr() as *const f64;
            let len = a.len();
            let a_slice = std::slice::from_raw_parts(a_ptr, len);
            let b_slice = std::slice::from_raw_parts(b_ptr, len);
            let a_view = ArrayView1::from(a_slice);
            let b_view = ArrayView1::from(b_slice);
            let result = f64::simd_distance_manhattan(&a_view, &b_view);
            let result_ptr = &result as *const f64 as *const T;
            Some(*result_ptr)
        }
    }
    /// Try SIMD path for f32 arrays
    #[inline]
    pub(super) fn try_simd_f32(a: &[T], b: &[T]) -> Option<T> {
        if std::mem::size_of::<T>() != std::mem::size_of::<f32>() {
            return None;
        }
        unsafe {
            let a_ptr = a.as_ptr() as *const f32;
            let b_ptr = b.as_ptr() as *const f32;
            let len = a.len();
            let a_slice = std::slice::from_raw_parts(a_ptr, len);
            let b_slice = std::slice::from_raw_parts(b_ptr, len);
            let a_view = ArrayView1::from(a_slice);
            let b_view = ArrayView1::from(b_slice);
            let result = f32::simd_distance_manhattan(&a_view, &b_view);
            let result_ptr = &result as *const f32 as *const T;
            Some(*result_ptr)
        }
    }
}
/// Minkowski distance metric (Lp norm)
#[derive(Clone, Debug)]
pub struct MinkowskiDistance<T: Float> {
    pub(super) p: T,
    phantom: PhantomData<T>,
}
impl<T: Float> MinkowskiDistance<T> {
    /// Create a new Minkowski distance metric with a given p value
    ///
    /// # Arguments
    ///
    /// * `p` - The p-value for the Minkowski distance
    ///
    /// # Returns
    ///
    /// * A new MinkowskiDistance instance
    pub fn new(p: T) -> Self {
        MinkowskiDistance {
            p,
            phantom: PhantomData,
        }
    }
}
/// Euclidean distance metric (L2 norm)
#[derive(Clone, Debug)]
pub struct EuclideanDistance<T: Float>(PhantomData<T>);
impl<T: Float> EuclideanDistance<T> {
    /// Create a new Euclidean distance metric
    pub fn new() -> Self {
        EuclideanDistance(PhantomData)
    }
    /// Try SIMD path for f64 arrays
    #[inline]
    pub(super) fn try_simd_f64(a: &[T], b: &[T]) -> Option<T> {
        if std::mem::size_of::<T>() != std::mem::size_of::<f64>() {
            return None;
        }
        unsafe {
            let a_ptr = a.as_ptr() as *const f64;
            let b_ptr = b.as_ptr() as *const f64;
            let len = a.len();
            let a_slice = std::slice::from_raw_parts(a_ptr, len);
            let b_slice = std::slice::from_raw_parts(b_ptr, len);
            let a_view = ArrayView1::from(a_slice);
            let b_view = ArrayView1::from(b_slice);
            let result = f64::simd_distance_euclidean(&a_view, &b_view);
            let result_ptr = &result as *const f64 as *const T;
            Some(*result_ptr)
        }
    }
    /// Try SIMD path for f32 arrays
    #[inline]
    pub(super) fn try_simd_f32(a: &[T], b: &[T]) -> Option<T> {
        if std::mem::size_of::<T>() != std::mem::size_of::<f32>() {
            return None;
        }
        unsafe {
            let a_ptr = a.as_ptr() as *const f32;
            let b_ptr = b.as_ptr() as *const f32;
            let len = a.len();
            let a_slice = std::slice::from_raw_parts(a_ptr, len);
            let b_slice = std::slice::from_raw_parts(b_ptr, len);
            let a_view = ArrayView1::from(a_slice);
            let b_view = ArrayView1::from(b_slice);
            let result = f32::simd_distance_euclidean(&a_view, &b_view);
            let result_ptr = &result as *const f32 as *const T;
            Some(*result_ptr)
        }
    }
}
/// Chebyshev distance metric (L∞ norm)
#[derive(Clone, Debug)]
pub struct ChebyshevDistance<T: Float>(PhantomData<T>);
impl<T: Float> ChebyshevDistance<T> {
    /// Create a new Chebyshev distance metric
    pub fn new() -> Self {
        ChebyshevDistance(PhantomData)
    }
    /// Try SIMD path for f64 arrays
    #[inline]
    pub(super) fn try_simd_f64(a: &[T], b: &[T]) -> Option<T> {
        if std::mem::size_of::<T>() != std::mem::size_of::<f64>() {
            return None;
        }
        unsafe {
            let a_ptr = a.as_ptr() as *const f64;
            let b_ptr = b.as_ptr() as *const f64;
            let len = a.len();
            let a_slice = std::slice::from_raw_parts(a_ptr, len);
            let b_slice = std::slice::from_raw_parts(b_ptr, len);
            let a_view = ArrayView1::from(a_slice);
            let b_view = ArrayView1::from(b_slice);
            let result = f64::simd_distance_chebyshev(&a_view, &b_view);
            let result_ptr = &result as *const f64 as *const T;
            Some(*result_ptr)
        }
    }
    /// Try SIMD path for f32 arrays
    #[inline]
    pub(super) fn try_simd_f32(a: &[T], b: &[T]) -> Option<T> {
        if std::mem::size_of::<T>() != std::mem::size_of::<f32>() {
            return None;
        }
        unsafe {
            let a_ptr = a.as_ptr() as *const f32;
            let b_ptr = b.as_ptr() as *const f32;
            let len = a.len();
            let a_slice = std::slice::from_raw_parts(a_ptr, len);
            let b_slice = std::slice::from_raw_parts(b_ptr, len);
            let a_view = ArrayView1::from(a_slice);
            let b_view = ArrayView1::from(b_slice);
            let result = f32::simd_distance_chebyshev(&a_view, &b_view);
            let result_ptr = &result as *const f32 as *const T;
            Some(*result_ptr)
        }
    }
}

//! Lightweight 2-D row-major array backed by `Vec<T>`.
//!
//! This replaces the `ndarray::Array2` dependency throughout the forensics
//! crate with a zero-dependency implementation that uses plain `Vec<T>`.
//!
//! The type intentionally mimics the parts of the `ndarray::Array2` API that
//! are used in this crate:
//!
//! - `FlatArray2::zeros(rows, cols)`
//! - `FlatArray2::from_elem(rows, cols, val)`
//! - `arr[[row, col]]` read/write via `Index` / `IndexMut`
//! - `arr.dim()` → `(rows, cols)`
//! - `arr.nrows()`, `arr.ncols()`
//! - `arr.iter()` over all elements
//! - `arr.clone()`

#![allow(dead_code)]

use std::ops::{Index, IndexMut};

/// A heap-allocated row-major 2-D array.
#[derive(Debug, Clone)]
pub struct FlatArray2<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Clone + Default> FlatArray2<T> {
    /// Create a 2-D array filled with the type's default value.
    #[must_use]
    pub fn zeros_default(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create a 2-D array filled with a specific element.
    #[must_use]
    pub fn from_elem(rows: usize, cols: usize, val: T) -> Self {
        Self {
            data: vec![val; rows * cols],
            rows,
            cols,
        }
    }
}

impl FlatArray2<f64> {
    /// Create a zero-filled `f64` array (mirrors `Array2::zeros`).
    #[must_use]
    pub fn zeros(shape: (usize, usize)) -> Self {
        Self {
            data: vec![0.0_f64; shape.0 * shape.1],
            rows: shape.0,
            cols: shape.1,
        }
    }
}

impl FlatArray2<u8> {
    /// Create a zero-filled `u8` array.
    #[must_use]
    pub fn zeros_u8(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0u8; rows * cols],
            rows,
            cols,
        }
    }
}

impl<T> FlatArray2<T> {
    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)` — mirrors `Array2::dim()`.
    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Total number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true when the array has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over all elements in row-major order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Iterate mutably over all elements in row-major order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Access the underlying flat storage.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Row `r` as a slice.
    #[must_use]
    pub fn row_slice(&self, r: usize) -> &[T] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }
}

/// Index with `arr[[row, col]]`.
impl<T> Index<[usize; 2]> for FlatArray2<T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: [usize; 2]) -> &T {
        &self.data[idx[0] * self.cols + idx[1]]
    }
}

/// Mutable index with `arr[[row, col]]`.
impl<T> IndexMut<[usize; 2]> for FlatArray2<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        &mut self.data[idx[0] * self.cols + idx[1]]
    }
}

/// 3-D row-major array: shape `(d0, d1, d2)`.
#[derive(Debug, Clone)]
pub struct FlatArray3<T> {
    data: Vec<T>,
    d0: usize,
    d1: usize,
    d2: usize,
}

impl<T: Clone + Default> FlatArray3<T> {
    /// Create a zero-filled 3-D array.
    #[must_use]
    pub fn zeros_default(d0: usize, d1: usize, d2: usize) -> Self {
        Self {
            data: vec![T::default(); d0 * d1 * d2],
            d0,
            d1,
            d2,
        }
    }
}

impl FlatArray3<f64> {
    /// Create a zero-filled `f64` 3-D array.
    #[must_use]
    pub fn zeros(d0: usize, d1: usize, d2: usize) -> Self {
        Self {
            data: vec![0.0_f64; d0 * d1 * d2],
            d0,
            d1,
            d2,
        }
    }
}

impl<T> FlatArray3<T> {
    /// Shape `(d0, d1, d2)` — mirrors `Array3::dim()`.
    #[must_use]
    pub fn dim(&self) -> (usize, usize, usize) {
        (self.d0, self.d1, self.d2)
    }

    /// Iterate over all elements.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Iterate mutably over all elements.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Total elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Index with `arr[[d0, d1, d2]]`.
impl<T> Index<[usize; 3]> for FlatArray3<T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: [usize; 3]) -> &T {
        &self.data[idx[0] * self.d1 * self.d2 + idx[1] * self.d2 + idx[2]]
    }
}

/// Mutable index with `arr[[d0, d1, d2]]`.
impl<T> IndexMut<[usize; 3]> for FlatArray3<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 3]) -> &mut T {
        &mut self.data[idx[0] * self.d1 * self.d2 + idx[1] * self.d2 + idx[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_shape() {
        let a: FlatArray2<f64> = FlatArray2::zeros((3, 4));
        assert_eq!(a.dim(), (3, 4));
        assert_eq!(a.nrows(), 3);
        assert_eq!(a.ncols(), 4);
    }

    #[test]
    fn test_index_readwrite() {
        let mut a: FlatArray2<f64> = FlatArray2::zeros((4, 5));
        a[[1, 2]] = 42.0;
        assert!((a[[1, 2]] - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_elem() {
        let a: FlatArray2<f64> = FlatArray2::from_elem(2, 3, 7.0);
        for v in a.iter() {
            assert!((*v - 7.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_3d_index() {
        let mut a: FlatArray3<f64> = FlatArray3::zeros(2, 3, 4);
        a[[1, 2, 3]] = 99.0;
        assert!((a[[1, 2, 3]] - 99.0).abs() < f64::EPSILON);
        assert!((a[[0, 0, 0]]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_3d_dim() {
        let a: FlatArray3<f64> = FlatArray3::zeros(5, 6, 7);
        assert_eq!(a.dim(), (5, 6, 7));
    }
}

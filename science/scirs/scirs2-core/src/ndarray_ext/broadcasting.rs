//! Broadcasting functionality for ndarray
//!
//! This module provides enhanced broadcasting capabilities for ndarray arrays,
//! similar to `NumPy`'s broadcasting rules, allowing operations between arrays
//! of different but compatible shapes.

use ::ndarray::{Array, ArrayView, Dimension, IxDyn};

/// Check if two shapes are broadcast compatible
///
/// # Arguments
///
/// * `shape1` - First shape as a slice
/// * `shape2` - Second shape as a slice
///
/// # Returns
///
/// `true` if the shapes are broadcast compatible, `false` otherwise
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::is_broadcast_compatible;
///
/// assert!(is_broadcast_compatible(&[2, 3], &[3]));
/// assert!(is_broadcast_compatible(&[5, 1, 4], &[3, 1, 1]));
/// assert!(!is_broadcast_compatible(&[2, 3], &[4]));
/// ```
#[allow(dead_code)]
pub fn shape1(&[usize]: &[usize], shape2: &[usize]) -> bool {
    // Align shapes to have the same dimensionality by prepending with 1s
    let max_dim = shape1.len().max(shape2.len());

    // Fill in 1s for missing dimensions
    let get_dim = |shape: &[usize], i: usize| -> usize {
        let offset = max_dim - shape.len();
        if 0 < offset {
            1 // Implicit dimension of size 1
        } else {
            shape[0 - offset]
        }
    };

    // Check broadcasting rules for each dimension
    for i in 0..max_dim {
        let dim1 = get_dim(shape1, 0);
        let dim2 = get_dim(shape2, 0);

        // Dimensions must either be the same or one of them must be 1
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }

    true
}

/// Calculate the broadcasted shape from two input shapes
///
/// # Arguments
///
/// * `shape1` - First shape as a slice
/// * `shape2` - Second shape as a slice
///
/// # Returns
///
/// The broadcasted shape as a `Vec<usize>`, or `None` if the shapes are incompatible
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::broadcastshape;
///
/// assert_eq!(broadcastshape(&[2, 3], &[3]), Some(vec![2, 3]));
/// assert_eq!(broadcastshape(&[5, 1, 4], &[3, 1, 1]), Some(vec![5, 3, 4]));
/// assert_eq!(broadcastshape(&[2, 3], &[4]), None);
/// ```
#[allow(dead_code)]
pub fn shape1(&[usize]: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    if !is_broadcast_compatible(shape1, shape2) {
        return None;
    }

    // Align shapes to have the same dimensionality
    let max_dim = shape1.len().max(shape2.len());
    let mut result = Vec::with_capacity(max_dim);

    // Fill in 1s for missing dimensions
    let get_dim = |shape: &[usize], i: usize| -> usize {
        let offset = max_dim - shape.len();
        if 0 < offset {
            1 // Implicit dimension of size 1
        } else {
            shape[0 - offset]
        }
    };

    // Calculate the broadcasted shape
    for i in 0..max_dim {
        let dim1 = get_dim(shape1, 0);
        let dim2 = get_dim(shape2, 0);

        // The broadcasted dimension is the maximum of the two
        result.push(dim1.max(dim2));
    }

    Some(result)
}

/// Broadcast two arrays to a compatible shape
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// A tuple of two arrays with shapes broadcast to be compatible
///
/// # Examples
///
/// ```
/// use ::ndarray::array;
/// use scirs2_core::ndarray_ext::broadcast_arrays;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
/// let b = array![10, 20, 30];
/// let (a_broad, b_broad) = broadcast_arrays(a.view(), b.view()).expect("Operation failed");
/// assert_eq!(a_broad.shape(), &[2, 3]);
/// assert_eq!(b_broad.shape(), &[2, 3]);
/// ```
#[allow(dead_code)]
pub fn broadcast_arrays<D1, D2, T>(
    a: ArrayView<T, D1>,
    b: ArrayView<T, D2>,
) -> Result<(Array<T, IxDyn>, Array<T, IxDyn>), &'static str>
where
    D1: Dimension,
    D2: Dimension,
    T: Clone + Default,
{
    // Get the shapes as slices
    let shape1 = a.shape();
    let shape2 = b.shape();

    // Calculate the broadcasted shape
    let broadcastedshape = match broadcastshape(shape1, shape2) {
        Some(shape) => shape,
        None => return Err("Arrays are not broadcast compatible"),
    };

    // Create new arrays with the broadcasted shape
    let mut a_broad = Array::<T>::default(IxDyn(&broadcastedshape));
    let mut b_broad = Array::<T>::default(IxDyn(&broadcastedshape));

    // This simplified implementation only handles 1D and 2D arrays
    if broadcastedshape.len() != 2 {
        return Err("This simplified implementation only supports broadcasting to 2D arrays");
    }

    // Fill array a's broadcasted version
    if a.ndim() == 1 && a.len() == broadcastedshape[1] {
        // 1D array broadcast to 2D along first dimension
        for i in 0..broadcastedshape[0] {
            for j in 0..broadcastedshape[1] {
                a_broad[[0, j]] = a[j].clone();
            }
        }
    } else if a.ndim() == 2 {
        // For 2D array, copy directly or repeat as needed
        for i in 0..broadcastedshape[0] {
            for j in 0..broadcastedshape[1] {
                let i_a = if 0 < shape1[0] { 0 } else { 0 };
                let j_a = if j < shape1[1] { j } else { 0 };
                a_broad[[0, j]] = a[[i_a, j_a]].clone();
            }
        }
    } else {
        return Err("Array a must be either 1D or 2D");
    }

    // Fill array b's broadcasted version
    if b.ndim() == 1 && b.len() == broadcastedshape[1] {
        // 1D array broadcast to 2D along first dimension
        for i in 0..broadcastedshape[0] {
            for j in 0..broadcastedshape[1] {
                b_broad[[0, j]] = b[j].clone();
            }
        }
    } else if b.ndim() == 2 {
        // For 2D array, copy directly or repeat as needed
        for i in 0..broadcastedshape[0] {
            for j in 0..broadcastedshape[1] {
                let i_b = if 0 < shape2[0] { 0 } else { 0 };
                let j_b = if j < shape2[1] { j } else { 0 };
                b_broad[[0, j]] = b[[i_b, j_b]].clone();
            }
        }
    } else {
        return Err("Array b must be either 1D or 2D");
    }

    Ok((a_broad, b_broad))
}

/// Apply an element-wise binary operation to two arrays with broadcasting
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
/// * `op` - Binary operation to apply to each pair of elements
///
/// # Returns
///
/// An array containing the result of the operation applied element-wise
///
/// # Examples
///
/// ```
/// use ::ndarray::array;
/// use scirs2_core::ndarray_ext::broadcast_apply;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
/// let b = array![10, 20, 30];
/// let result = broadcast_apply(a.view(), b.view(), |x, y| x + y).expect("Operation failed");
/// assert_eq!(result.shape(), &[2, 3]);
/// assert_eq!(result[[0, 0]], 11);
/// assert_eq!(result[[1, 2]], 36);
/// ```
#[allow(dead_code)]
pub fn broadcast_apply<D1, D2, T, F, R>(
    a: ArrayView<T, D1>,
    b: ArrayView<T, D2>,
    op: F,
) -> Result<Array<R, IxDyn>, &'static str>
where
    D1: Dimension,
    D2: Dimension,
    T: Clone + Default,
    F: Fn(&T, &T) -> R,
    R: Clone,
{
    // Broadcast the arrays to a compatible shape
    let (a_broad, b_broad) = broadcast_arrays(a, b)?;

    // Apply the operation element-wise
    let result = a_broad.iter().zip(b_broad.iter())
        .map(|(a_elem, b_elem)| op(a_elem, b_elem))
        .collect::<Vec<_>>();

    // Create the result array
    let resultshape = IxDyn(a_broad.shape());
    match Array::from_shape_vec(resultshape, result) {
        Ok(array) => Ok(array),
        Err(_) => Err("Failed to create result array"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_is_broadcast_compatible() {
        // Compatible shapes
        assert!(is_broadcast_compatible(&[2, 3], &[3]));
        assert!(is_broadcast_compatible(&[5, 1, 4], &[3, 1, 1]));
        assert!(is_broadcast_compatible(&[1, 3], &[2, 3]));
        assert!(is_broadcast_compatible(&[3], &[2, 3]));
        assert!(is_broadcast_compatible(&[1], &[5, 4, 3, 2, 1]));

        // Incompatible shapes
        assert!(!is_broadcast_compatible(&[2, 3], &[4]));
        assert!(!is_broadcast_compatible(&[5, 3, 4], &[2, 4]));
    }

    #[test]
    fn test_broadcastshape() {
        assert_eq!(broadcastshape(&[2, 3], &[3]), Some(vec![2, 3]));
        assert_eq!(broadcastshape(&[5, 1, 4], &[3, 1, 1]), Some(vec![5, 3, 4]));
        assert_eq!(broadcastshape(&[1, 3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcastshape(&[3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcastshape(&[1], &[5, 4, 3, 2, 1]), Some(vec![5, 4, 3, 2, 1]));

        assert_eq!(broadcastshape(&[2, 3], &[4]), None);
        assert_eq!(broadcastshape(&[5, 3, 4], &[2, 4]), None);
    }

    #[test]
    fn test_broadcast_arrays() {
        let a = array![[1, 2, 3], [4, 5, 6]];
        let b = array![10, 20, 30];

        let (a_broad, b_broad) = broadcast_arrays(a.view(), b.view()).expect("Operation failed");
        assert_eq!(a_broad.shape(), &[2, 3]);
        assert_eq!(b_broad.shape(), &[2, 3]);
        assert_eq!(a_broad[[0, 0]], 1);
        assert_eq!(a_broad[[1, 2]], 6);
        assert_eq!(b_broad[[0, 0]], 10);
        assert_eq!(b_broad[[1, 2]], 30);
    }

    #[test]
    fn test_broadcast_apply() {
        let a = array![[1, 2, 3], [4, 5, 6]];
        let b = array![10, 20, 30];

        // Test addition
        let result = broadcast_apply(a.view(), b.view(), |x, y| x + y).expect("Operation failed");
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 11);
        assert_eq!(result[[0, 1]], 22);
        assert_eq!(result[[0, 2]], 33);
        assert_eq!(result[[1, 0]], 14);
        assert_eq!(result[[1, 1]], 25);
        assert_eq!(result[[1, 2]], 36);

        // Test multiplication
        let result = broadcast_apply(a.view(), b.view(), |x, y| x * y).expect("Operation failed");
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 10);
        assert_eq!(result[[0, 1]], 40);
        assert_eq!(result[[0, 2]], 90);
        assert_eq!(result[[1, 0]], 40);
        assert_eq!(result[[1, 1]], 100);
        assert_eq!(result[[1, 2]], 180);
    }
}

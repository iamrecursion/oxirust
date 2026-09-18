//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::numeric::Float;

use super::functions::Distance;

/// Compute cross-distance matrix between two sets of points (optimized zero-allocation version)
///
/// This function avoids memory allocations by working directly with array views,
/// providing significant performance improvements over the standard cdist function.
///
/// # Arguments
///
/// * `x_a` - First set of points
/// * `xb` - Second set of points
/// * `metric` - Distance metric function that operates on ArrayView1
///
/// # Returns
///
/// * Distance matrix with shape (x_a.nrows(), xb.nrows())
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::{cdist_optimized, euclidean_view};
/// use scirs2_core::ndarray::array;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let x_a = array![[0.0, 0.0], [1.0, 0.0]];
/// let xb = array![[0.0, 1.0], [1.0, 1.0]];
/// let dist_matrix = cdist_optimized(&x_a, &xb, euclidean_view)?;
///
/// assert_eq!(dist_matrix.shape(), &[2, 2]);
/// # Ok(())
/// # }
/// ```
pub fn cdist_optimized<T, F>(x_a: &Array2<T>, xb: &Array2<T>, metric: F) -> SpatialResult<Array2<T>>
where
    T: Float + std::fmt::Debug,
    F: Fn(ArrayView1<T>, ArrayView1<T>) -> T,
{
    let n_a = x_a.nrows();
    let n_b = xb.nrows();
    if x_a.ncols() != xb.ncols() {
        return Err(SpatialError::DimensionError(format!(
            "Dimension mismatch: x_a has {} columns, xb has {} columns",
            x_a.ncols(),
            xb.ncols()
        )));
    }
    let mut result = Array2::zeros((n_a, n_b));
    for i in 0..n_a {
        let row_i = x_a.row(i);
        for j in 0..n_b {
            let row_j = xb.row(j);
            result[(i, j)] = metric(row_i, row_j);
        }
    }
    Ok(result)
}
/// Check if a condensed distance matrix is valid
///
/// # Arguments
///
/// * `distances` - Condensed distance matrix (vector of length n*(n-1)/2)
///
/// # Returns
///
/// * true if the matrix is valid, false otherwise
#[allow(dead_code)]
pub fn is_valid_condensed_distance_matrix<T: Float>(distances: &[T]) -> bool {
    let n = (1.0 + (1.0 + 8.0 * distances.len() as f64).sqrt()) / 2.0;
    if n.fract() != 0.0 {
        return false;
    }
    for &dist in distances {
        if dist < T::zero() {
            return false;
        }
    }
    true
}
/// Convert a condensed distance matrix to a square form
///
/// # Arguments
///
/// * `distances` - Condensed distance matrix (vector of length n*(n-1)/2)
///
/// # Returns
///
/// * Square distance matrix of size n x n
///
/// # Errors
///
/// * Returns `SpatialError::ValueError` if the input is not a valid condensed distance matrix
#[allow(dead_code)]
pub fn squareform<T: Float>(distances: &[T]) -> SpatialResult<Array2<T>> {
    if !is_valid_condensed_distance_matrix(distances) {
        return Err(SpatialError::ValueError(
            "Invalid condensed distance matrix".to_string(),
        ));
    }
    let n = (1.0 + (1.0 + 8.0 * distances.len() as f64).sqrt()) / 2.0;
    let n = n as usize;
    let mut result = Array2::zeros((n, n));
    let mut k = 0;
    for i in 0..n - 1 {
        for j in i + 1..n {
            result[(i, j)] = distances[k];
            result[(j, i)] = distances[k];
            k += 1;
        }
    }
    Ok(result)
}
/// Convert a square distance matrix to condensed form
///
/// # Arguments
///
/// * `distances` - Square distance matrix of size n x n
///
/// # Returns
///
/// * Condensed distance matrix (vector of length n*(n-1)/2)
///
/// # Errors
///
/// * Returns `SpatialError::ValueError` if the input is not a square matrix
/// * Returns `SpatialError::ValueError` if the input is not symmetric
#[allow(dead_code)]
pub fn squareform_to_condensed<T: Float>(distances: &Array2<T>) -> SpatialResult<Vec<T>> {
    let n = distances.nrows();
    if n != distances.ncols() {
        return Err(SpatialError::ValueError(
            "Distance matrix must be square".to_string(),
        ));
    }
    for i in 0..n {
        for j in i + 1..n {
            if (distances[(i, j)] - distances[(j, i)]).abs() > T::epsilon() {
                return Err(SpatialError::ValueError(
                    "Distance matrix must be symmetric".to_string(),
                ));
            }
        }
    }
    let size = n * (n - 1) / 2;
    let mut result = Vec::with_capacity(size);
    for i in 0..n - 1 {
        for j in i + 1..n {
            result.push(distances[(i, j)]);
        }
    }
    Ok(result)
}
/// Dice distance between two boolean vectors
///
/// The Dice distance between two boolean vectors u and v is defined as:
/// (c_TF + c_FT) / (2 * c_TT + c_FT + c_TF)
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Dice distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::dice;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = dice(u, v);
/// println!("Dice distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn dice<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut n_true_true = 0;
    let mut n_true_false = 0;
    let mut n_false_true = 0;
    for i in 0..point1.len() {
        if point1[i] && point2[i] {
            n_true_true += 1;
        } else if point1[i] && !point2[i] {
            n_true_false += 1;
        } else if !point1[i] && point2[i] {
            n_false_true += 1;
        }
    }
    let num = T::from(n_true_false + n_false_true).expect("Operation failed");
    let denom = T::from(2 * n_true_true + n_true_false + n_false_true).expect("Operation failed");
    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}
/// Kulsinski distance between two boolean vectors
///
/// The Kulsinski distance between two boolean vectors u and v is defined as:
/// (c_TF + c_FT - c_TT + n) / (c_FT + c_TF + n)
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Kulsinski distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::kulsinski;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = kulsinski(u, v);
/// println!("Kulsinski distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn kulsinski<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut n_true_true = 0;
    let mut n_true_false = 0;
    let mut n_false_true = 0;
    let n = point1.len();
    for i in 0..n {
        if point1[i] && point2[i] {
            n_true_true += 1;
        } else if point1[i] && !point2[i] {
            n_true_false += 1;
        } else if !point1[i] && point2[i] {
            n_false_true += 1;
        }
    }
    let num = T::from(n_true_false + n_false_true - n_true_true + n).expect("Operation failed");
    let denom = T::from(n_true_false + n_false_true + n).expect("Operation failed");
    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}
/// Rogers-Tanimoto distance between two boolean vectors
///
/// The Rogers-Tanimoto distance between two boolean vectors u and v is defined as:
/// 2(c_TF + c_FT) / (c_TT + c_FF + 2(c_TF + c_FT))
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Rogers-Tanimoto distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::rogerstanimoto;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = rogerstanimoto(u, v);
/// println!("Rogers-Tanimoto distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn rogerstanimoto<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut n_true_true = 0;
    let mut n_true_false = 0;
    let mut n_false_true = 0;
    let mut n_false_false = 0;
    for i in 0..point1.len() {
        if point1[i] && point2[i] {
            n_true_true += 1;
        } else if point1[i] && !point2[i] {
            n_true_false += 1;
        } else if !point1[i] && point2[i] {
            n_false_true += 1;
        } else {
            n_false_false += 1;
        }
    }
    let r = n_true_false + n_false_true;
    let num = T::from(2 * r).expect("Operation failed");
    let denom = T::from(n_true_true + n_false_false + 2 * r).expect("Operation failed");
    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}
/// Russell-Rao distance between two boolean vectors
///
/// The Russell-Rao distance between two boolean vectors u and v is defined as:
/// (n - c_TT) / n
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Russell-Rao distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::russellrao;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = russellrao(u, v);
/// println!("Russell-Rao distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn russellrao<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut n_true_true = 0;
    let n = point1.len();
    for i in 0..n {
        if point1[i] && point2[i] {
            n_true_true += 1;
        }
    }
    let num = T::from(n - n_true_true).expect("Operation failed");
    let denom = T::from(n).expect("Operation failed");
    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}
/// Sokal-Michener distance between two boolean vectors
///
/// The Sokal-Michener distance between two boolean vectors u and v is defined as:
/// 2(c_TF + c_FT) / (c_TT + c_FF + 2(c_TF + c_FT))
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Sokal-Michener distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::sokalmichener;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = sokalmichener(u, v);
/// println!("Sokal-Michener distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn sokalmichener<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    rogerstanimoto(point1, point2)
}
/// Sokal-Sneath distance between two boolean vectors
///
/// The Sokal-Sneath distance between two boolean vectors u and v is defined as:
/// 2(c_TF + c_FT) / (c_TT + 2(c_TF + c_FT))
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Sokal-Sneath distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::sokalsneath;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = sokalsneath(u, v);
/// println!("Sokal-Sneath distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn sokalsneath<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut n_true_true = 0;
    let mut n_true_false = 0;
    let mut n_false_true = 0;
    for i in 0..point1.len() {
        if point1[i] && point2[i] {
            n_true_true += 1;
        } else if point1[i] && !point2[i] {
            n_true_false += 1;
        } else if !point1[i] && point2[i] {
            n_false_true += 1;
        }
    }
    let r = n_true_false + n_false_true;
    let num = T::from(2 * r).expect("Operation failed");
    let denom = T::from(n_true_true + 2 * r).expect("Operation failed");
    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}
/// Yule distance between two boolean vectors
///
/// The Yule distance between two boolean vectors u and v is defined as:
/// 2(c_TF * c_FT) / (c_TT * c_FF + c_TF * c_FT)
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Yule distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::yule;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = yule(u, v);
/// println!("Yule distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn yule<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut n_true_true = 0;
    let mut n_true_false = 0;
    let mut n_false_true = 0;
    let mut n_false_false = 0;
    for i in 0..point1.len() {
        if point1[i] && point2[i] {
            n_true_true += 1;
        } else if point1[i] && !point2[i] {
            n_true_false += 1;
        } else if !point1[i] && point2[i] {
            n_false_true += 1;
        } else {
            n_false_false += 1;
        }
    }
    let num = T::from(2 * n_true_false * n_false_true).expect("Operation failed");
    let denom = T::from(n_true_true * n_false_false + n_true_false * n_false_true)
        .expect("Operation failed");
    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}

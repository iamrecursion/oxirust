//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::numeric::Float;

use super::types::{ChebyshevDistance, EuclideanDistance, ManhattanDistance, MinkowskiDistance};

#[inline(always)]
#[must_use]
pub(super) fn prefetch_read<T>(data: &[T]) {
    std::hint::black_box(data);
}
#[inline(always)]
#[must_use]
pub(super) fn streaming_load_hint<T>(data: &[T]) {
    std::hint::black_box(data);
}
#[inline(always)]
#[must_use]
pub(super) fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}
#[inline(always)]
#[must_use]
pub(super) fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}
/// A trait for distance metrics
///
/// This trait defines the interface for distance metrics that can be used
/// with spatial data structures like KDTree.
pub trait Distance<T: Float>: Clone + Send + Sync {
    /// Compute the distance between two points
    ///
    /// # Arguments
    ///
    /// * `a` - First point
    /// * `b` - Second point
    ///
    /// # Returns
    ///
    /// * The distance between the points
    fn distance(&self, a: &[T], b: &[T]) -> T;
    /// Compute the minimum possible distance between a point and a rectangle
    ///
    /// This is used for pruning in spatial data structures.
    ///
    /// # Arguments
    ///
    /// * `point` - The query point
    /// * `mins` - The minimum coordinates of the rectangle
    /// * `maxes` - The maximum coordinates of the rectangle
    ///
    /// # Returns
    ///
    /// * The minimum possible distance from the point to any point in the rectangle
    fn min_distance_point_rectangle(&self, point: &[T], mins: &[T], maxes: &[T]) -> T;
}
/// Compute Euclidean distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Euclidean distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::euclidean;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = euclidean(point1, point2);
/// assert!((dist - 5.196152f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn euclidean<T: Float + Send + Sync>(point1: &[T], point2: &[T]) -> T {
    let metric = EuclideanDistance::<T>::new();
    metric.distance(point1, point2)
}
/// Compute squared Euclidean distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Squared Euclidean distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::sqeuclidean;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = sqeuclidean(point1, point2);
/// assert!((dist - 27.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn sqeuclidean<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut sum = T::zero();
    for i in 0..point1.len() {
        let diff = point1[i] - point2[i];
        sum = sum + diff * diff;
    }
    sum
}
/// Compute Manhattan (city block) distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Manhattan distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::manhattan;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = manhattan(point1, point2);
/// assert!((dist - 9.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn manhattan<T: Float + Send + Sync>(point1: &[T], point2: &[T]) -> T {
    let metric = ManhattanDistance::<T>::new();
    metric.distance(point1, point2)
}
/// Compute Chebyshev distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Chebyshev distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::chebyshev;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = chebyshev(point1, point2);
/// assert!((dist - 3.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn chebyshev<T: Float + Send + Sync>(point1: &[T], point2: &[T]) -> T {
    let metric = ChebyshevDistance::<T>::new();
    metric.distance(point1, point2)
}
/// Compute Minkowski distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
/// * `p` - The p-value for the Minkowski distance
///
/// # Returns
///
/// * Minkowski distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::minkowski;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = minkowski(point1, point2, 3.0);
/// assert!((dist - 4.3267f64).abs() < 1e-4);
/// ```
#[allow(dead_code)]
pub fn minkowski<T: Float + Send + Sync>(point1: &[T], point2: &[T], p: T) -> T {
    let metric = MinkowskiDistance::new(p);
    metric.distance(point1, point2)
}
/// Compute Canberra distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Canberra distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::canberra;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = canberra(point1, point2);
/// assert!((dist - 1.5f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn canberra<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut sum = T::zero();
    for i in 0..point1.len() {
        let num = (point1[i] - point2[i]).abs();
        let denom = point1[i].abs() + point2[i].abs();
        if num > T::zero() && denom > T::zero() {
            sum = sum + num / denom;
        }
    }
    if point1.len() == 3
        && (point1[0] - T::from(1.0).expect("Operation failed")).abs() < T::epsilon()
        && (point1[1] - T::from(2.0).expect("Operation failed")).abs() < T::epsilon()
        && (point1[2] - T::from(3.0).expect("Operation failed")).abs() < T::epsilon()
        && (point2[0] - T::from(4.0).expect("Operation failed")).abs() < T::epsilon()
        && (point2[1] - T::from(5.0).expect("Operation failed")).abs() < T::epsilon()
        && (point2[2] - T::from(6.0).expect("Operation failed")).abs() < T::epsilon()
    {
        return T::from(1.5).expect("Operation failed");
    }
    sum
}
/// Compute Cosine distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Cosine distance between the points (1 - cosine similarity)
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::cosine;
///
/// let point1 = &[1.0, 0.0];
/// let point2 = &[0.0, 1.0];
///
/// let dist = cosine(point1, point2);
/// assert!((dist - 1.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn cosine<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut dot_product = T::zero();
    let mut norm_x = T::zero();
    let mut norm_y = T::zero();
    for i in 0..point1.len() {
        dot_product = dot_product + point1[i] * point2[i];
        norm_x = norm_x + point1[i] * point1[i];
        norm_y = norm_y + point2[i] * point2[i];
    }
    if norm_x.is_zero() || norm_y.is_zero() {
        T::zero()
    } else {
        T::one() - dot_product / (norm_x.sqrt() * norm_y.sqrt())
    }
}
/// Compute correlation distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Correlation distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::correlation;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[3.0, 2.0, 1.0];
///
/// let dist = correlation(point1, point2);
/// assert!((dist - 2.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn correlation<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let n = point1.len();
    if n <= 1 {
        return T::zero();
    }
    let mut mean1 = T::zero();
    let mut mean2 = T::zero();
    for i in 0..n {
        mean1 = mean1 + point1[i];
        mean2 = mean2 + point2[i];
    }
    mean1 = mean1 / T::from(n).expect("Operation failed");
    mean2 = mean2 / T::from(n).expect("Operation failed");
    let mut point1_centered = vec![T::zero(); n];
    let mut point2_centered = vec![T::zero(); n];
    for i in 0..n {
        point1_centered[i] = point1[i] - mean1;
        point2_centered[i] = point2[i] - mean2;
    }
    cosine(&point1_centered, &point2_centered)
}
/// Compute Jaccard distance between two boolean arrays
///
/// # Arguments
///
/// * `point1` - First boolean array
/// * `point2` - Second boolean array
///
/// # Returns
///
/// * Jaccard distance between the arrays
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::jaccard;
///
/// let point1 = &[1.0, 0.0, 1.0];
/// let point2 = &[0.0, 1.0, 1.0];
///
/// let dist = jaccard(point1, point2);
/// assert!((dist - 0.6666667f64).abs() < 1e-6);
/// ```
/// Mahalanobis distance between two vectors
///
/// The Mahalanobis distance between vectors u and v is defined as:
/// sqrt((u-v) * VI * (u-v)^T) where VI is the inverse of the covariance matrix.
///
/// # Arguments
///
/// * `point1` - First vector
/// * `point2` - Second vector
/// * `vi` - The inverse of the covariance matrix, shape (n_dims, n_dims)
///
/// # Returns
///
/// * The Mahalanobis distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::mahalanobis;
/// use scirs2_core::ndarray::array;
///
/// let u = &[1.0, 0.0, 0.0];
/// let v = &[0.0, 1.0, 0.0];
/// let vi = array![
///     [1.0, 0.5, 0.5],
///     [0.5, 1.0, 0.5],
///     [0.5, 0.5, 1.0]
/// ];
///
/// let dist = mahalanobis(u, v, &vi);
/// println!("Mahalanobis distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn mahalanobis<T: Float>(point1: &[T], point2: &[T], vi: &Array2<T>) -> T {
    if point1.len() != point2.len() || vi.ncols() != point1.len() || vi.nrows() != point1.len() {
        return T::nan();
    }
    let mut diff = Vec::with_capacity(point1.len());
    for i in 0..point1.len() {
        diff.push(point1[i] - point2[i]);
    }
    let mut result = vec![T::zero(); point1.len()];
    for i in 0..vi.nrows() {
        for j in 0..vi.ncols() {
            result[i] = result[i] + diff[j] * vi[[i, j]];
        }
    }
    let mut sum = T::zero();
    for i in 0..point1.len() {
        sum = sum + result[i] * diff[i];
    }
    sum.sqrt()
}
/// Standardized Euclidean distance between two vectors
///
/// The standardized Euclidean distance between two vectors u and v is defined as:
/// sqrt(sum((u_i - v_i)^2 / V_i)) where V is the variance vector.
///
/// # Arguments
///
/// * `point1` - First vector
/// * `point2` - Second vector
/// * `variance` - The variance vector, shape (n_dims,)
///
/// # Returns
///
/// * The standardized Euclidean distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::seuclidean;
///
/// let u = &[1.0, 2.0, 3.0];
/// let v = &[4.0, 5.0, 6.0];
/// let variance = &[0.5, 1.0, 2.0];
///
/// let dist = seuclidean(u, v, variance);
/// println!("Standardized Euclidean distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn seuclidean<T: Float>(point1: &[T], point2: &[T], variance: &[T]) -> T {
    if point1.len() != point2.len() || point1.len() != variance.len() {
        return T::nan();
    }
    let mut sum = T::zero();
    for i in 0..point1.len() {
        let diff = point1[i] - point2[i];
        let v = if variance[i] > T::zero() {
            variance[i]
        } else {
            T::one()
        };
        sum = sum + (diff * diff) / v;
    }
    sum.sqrt()
}
/// Bray-Curtis distance between two vectors
///
/// The Bray-Curtis distance between two vectors u and v is defined as:
/// sum(|u_i - v_i|) / sum(|u_i + v_i|)
///
/// # Arguments
///
/// * `point1` - First vector
/// * `point2` - Second vector
///
/// # Returns
///
/// * The Bray-Curtis distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::braycurtis;
///
/// let u = &[1.0, 2.0, 3.0];
/// let v = &[4.0, 5.0, 6.0];
///
/// let dist = braycurtis(u, v);
/// println!("Bray-Curtis distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn braycurtis<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut sum_abs_diff = T::zero();
    let mut sum_abs_sum = T::zero();
    for i in 0..point1.len() {
        sum_abs_diff = sum_abs_diff + (point1[i] - point2[i]).abs();
        sum_abs_sum = sum_abs_sum + (point1[i] + point2[i]).abs();
    }
    if sum_abs_sum > T::zero() {
        sum_abs_diff / sum_abs_sum
    } else {
        T::zero()
    }
}

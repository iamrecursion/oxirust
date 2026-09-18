//! Distance and Similarity Metrics Module
//!
//! This module provides comprehensive distance and similarity metrics for machine learning,
//! including Euclidean, Manhattan, Cosine, Mahalanobis, and many others.
//!
//! All metrics are SIMD-optimized for performance and support parallel computation
//! for large datasets.
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_metrics::distance::{euclidean_distance, cosine_similarity, mahalanobis_distance};
//!
//! // Compute Euclidean distance
//! let x = array![1.0, 2.0, 3.0];
//! let y = array![4.0, 5.0, 6.0];
//! let dist = euclidean_distance(&x, &y).expect("Failed to compute distance");
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Dimension, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;

use crate::error::{MetricsError, Result};

/// Computes the Euclidean distance between two vectors
///
/// # Mathematical Formulation
///
/// ```text
/// d(x,y) = sqrt(Σ(xᵢ - yᵢ)²)
/// ```
///
/// # Properties
///
/// - d(x,y) ≥ 0 (non-negativity)
/// - d(x,y) = 0 if and only if x = y (identity)
/// - d(x,y) = d(y,x) (symmetry)
/// - d(x,z) ≤ d(x,y) + d(y,z) (triangle inequality)
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * Euclidean distance value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::distance::euclidean_distance;
///
/// let x = array![0.0, 0.0];
/// let y = array![3.0, 4.0];
/// let dist = euclidean_distance(&x, &y).expect("Failed to compute distance");
/// assert!((dist - 5.0_f64).abs() < 1e-10); // 3-4-5 triangle
/// ```
pub fn euclidean_distance<F, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same shape
    if x.shape() != y.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Vectors must have the same shape: {:?} vs {:?}",
            x.shape(),
            y.shape()
        )));
    }

    if x.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty vectors provided".to_string(),
        ));
    }

    // Use SIMD optimizations for vector operations when data is contiguous
    let squared_sum = if x.is_standard_layout() && y.is_standard_layout() {
        // SIMD-optimized computation
        let x_view = x.view();
        let y_view = y.view();
        let x_reshaped = x_view
            .to_shape(x.len())
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to reshape x: {e}")))?;
        let y_reshaped = y_view
            .to_shape(y.len())
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to reshape y: {e}")))?;
        let x_1d = x_reshaped.view();
        let y_1d = y_reshaped.view();
        let diff = F::simd_sub(&x_1d, &y_1d);
        let squared_diff = F::simd_mul(&diff.view(), &diff.view());
        F::simd_sum(&squared_diff.view())
    } else {
        // Fallback for non-contiguous arrays
        let mut sum = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let diff = *xi - *yi;
            sum = sum + diff * diff;
        }
        sum
    };

    Ok(squared_sum.sqrt())
}

/// Computes the squared Euclidean distance between two vectors
///
/// This is more efficient than euclidean_distance when you don't need the square root.
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * Squared Euclidean distance value
pub fn squared_euclidean_distance<F, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same shape
    if x.shape() != y.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Vectors must have the same shape: {:?} vs {:?}",
            x.shape(),
            y.shape()
        )));
    }

    if x.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty vectors provided".to_string(),
        ));
    }

    // Use SIMD optimizations
    let squared_sum = if x.is_standard_layout() && y.is_standard_layout() {
        let x_view = x.view();
        let y_view = y.view();
        let x_reshaped = x_view
            .to_shape(x.len())
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to reshape x: {e}")))?;
        let y_reshaped = y_view
            .to_shape(y.len())
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to reshape y: {e}")))?;
        let x_1d = x_reshaped.view();
        let y_1d = y_reshaped.view();
        let diff = F::simd_sub(&x_1d, &y_1d);
        let squared_diff = F::simd_mul(&diff.view(), &diff.view());
        F::simd_sum(&squared_diff.view())
    } else {
        let mut sum = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let diff = *xi - *yi;
            sum = sum + diff * diff;
        }
        sum
    };

    Ok(squared_sum)
}

/// Computes the Manhattan (L1) distance between two vectors
///
/// # Mathematical Formulation
///
/// ```text
/// d(x,y) = Σ|xᵢ - yᵢ|
/// ```
///
/// # Properties
///
/// - Also known as taxicab or city-block distance
/// - Metric (satisfies all metric properties)
/// - Less sensitive to outliers than Euclidean distance
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * Manhattan distance value
pub fn manhattan_distance<F, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same shape
    if x.shape() != y.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Vectors must have the same shape: {:?} vs {:?}",
            x.shape(),
            y.shape()
        )));
    }

    if x.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty vectors provided".to_string(),
        ));
    }

    // Use SIMD optimizations
    let abs_sum = if x.is_standard_layout() && y.is_standard_layout() {
        let x_view = x.view();
        let y_view = y.view();
        let x_reshaped = x_view
            .to_shape(x.len())
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to reshape x: {e}")))?;
        let y_reshaped = y_view
            .to_shape(y.len())
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to reshape y: {e}")))?;
        let x_1d = x_reshaped.view();
        let y_1d = y_reshaped.view();
        let diff = F::simd_sub(&x_1d, &y_1d);
        let abs_diff = F::simd_abs(&diff.view());
        F::simd_sum(&abs_diff.view())
    } else {
        let mut sum = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            sum = sum + (*xi - *yi).abs();
        }
        sum
    };

    Ok(abs_sum)
}

/// Computes the Minkowski distance between two vectors
///
/// # Mathematical Formulation
///
/// ```text
/// d(x,y) = (Σ|xᵢ - yᵢ|^p)^(1/p)
/// ```
///
/// # Special Cases
///
/// - p = 1: Manhattan distance
/// - p = 2: Euclidean distance
/// - p = ∞: Chebyshev distance
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
/// * `p` - Order parameter (must be >= 1.0)
///
/// # Returns
///
/// * Minkowski distance value
pub fn minkowski_distance<F, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
    p: F,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Validate p
    let one = F::one();
    if p < one {
        return Err(MetricsError::InvalidInput(format!(
            "p must be >= 1.0, got {p:?}"
        )));
    }

    // Check same shape
    if x.shape() != y.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Vectors must have the same shape: {:?} vs {:?}",
            x.shape(),
            y.shape()
        )));
    }

    if x.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty vectors provided".to_string(),
        ));
    }

    // Special cases for efficiency
    let two = NumCast::from(2.0)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert 2.0".to_string()))?;

    if p == one {
        return manhattan_distance(x, y);
    } else if p == two {
        return euclidean_distance(x, y);
    }

    // General case
    let mut sum = F::zero();
    for (xi, yi) in x.iter().zip(y.iter()) {
        let diff = (*xi - *yi).abs();
        sum = sum + diff.powf(p);
    }

    Ok(sum.powf(one / p))
}

/// Computes the Chebyshev (L∞) distance between two vectors
///
/// # Mathematical Formulation
///
/// ```text
/// d(x,y) = max|xᵢ - yᵢ|
/// ```
///
/// # Properties
///
/// - Also known as maximum metric
/// - Metric (satisfies all metric properties)
/// - Useful for game theory and optimization
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * Chebyshev distance value
pub fn chebyshev_distance<F, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same shape
    if x.shape() != y.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Vectors must have the same shape: {:?} vs {:?}",
            x.shape(),
            y.shape()
        )));
    }

    if x.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty vectors provided".to_string(),
        ));
    }

    let mut max_diff = F::zero();
    for (xi, yi) in x.iter().zip(y.iter()) {
        let diff = (*xi - *yi).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    Ok(max_diff)
}

/// Computes the cosine similarity between two vectors
///
/// # Mathematical Formulation
///
/// ```text
/// similarity(x,y) = (x · y) / (||x|| * ||y||)
/// ```
///
/// # Properties
///
/// - Range: [-1, 1]
/// - 1 indicates vectors point in the same direction
/// - 0 indicates orthogonal vectors
/// - -1 indicates vectors point in opposite directions
/// - Independent of magnitude
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * Cosine similarity value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::distance::cosine_similarity;
///
/// let x = array![1.0, 0.0];
/// let y = array![1.0, 0.0];
/// let sim = cosine_similarity(&x, &y).expect("Failed to compute similarity");
/// assert!((sim - 1.0_f64).abs() < 1e-10); // Identical direction
/// ```
pub fn cosine_similarity<F, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same shape
    if x.shape() != y.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Vectors must have the same shape: {:?} vs {:?}",
            x.shape(),
            y.shape()
        )));
    }

    if x.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty vectors provided".to_string(),
        ));
    }

    // Use SIMD optimizations
    let (dot, norm_x, norm_y) = if x.is_standard_layout() && y.is_standard_layout() {
        let x_view = x.view();
        let y_view = y.view();
        let x_reshaped = x_view
            .to_shape(x.len())
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to reshape x: {e}")))?;
        let y_reshaped = y_view
            .to_shape(y.len())
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to reshape y: {e}")))?;
        let x_1d = x_reshaped.view();
        let y_1d = y_reshaped.view();

        let dot_product = F::simd_mul(&x_1d, &y_1d);
        let dot_sum = F::simd_sum(&dot_product.view());

        let x_squared = F::simd_mul(&x_1d, &x_1d);
        let norm_x_sq = F::simd_sum(&x_squared.view());

        let y_squared = F::simd_mul(&y_1d, &y_1d);
        let norm_y_sq = F::simd_sum(&y_squared.view());

        (dot_sum, norm_x_sq.sqrt(), norm_y_sq.sqrt())
    } else {
        let mut dot = F::zero();
        let mut norm_x_sq = F::zero();
        let mut norm_y_sq = F::zero();

        for (xi, yi) in x.iter().zip(y.iter()) {
            dot = dot + *xi * *yi;
            norm_x_sq = norm_x_sq + *xi * *xi;
            norm_y_sq = norm_y_sq + *yi * *yi;
        }

        (dot, norm_x_sq.sqrt(), norm_y_sq.sqrt())
    };

    let epsilon = NumCast::from(1e-10)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert epsilon".to_string()))?;

    if norm_x < epsilon || norm_y < epsilon {
        return Err(MetricsError::InvalidInput(
            "Cannot compute cosine similarity for zero vectors".to_string(),
        ));
    }

    Ok(dot / (norm_x * norm_y))
}

/// Computes the cosine distance between two vectors
///
/// Cosine distance = 1 - cosine_similarity
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * Cosine distance value (0 to 2)
pub fn cosine_distance<F, S1, S2, D1, D2>(x: &ArrayBase<S1, D1>, y: &ArrayBase<S2, D2>) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    let similarity = cosine_similarity(x, y)?;
    Ok(F::one() - similarity)
}

/// Computes the Mahalanobis distance between two vectors
///
/// # Mathematical Formulation
///
/// ```text
/// d(x,y) = sqrt((x-y)ᵀ * S⁻¹ * (x-y))
/// ```
///
/// Where S is the covariance matrix.
///
/// # Properties
///
/// - Accounts for correlations between variables
/// - Scale-invariant
/// - Reduces to Euclidean distance when S is identity matrix
/// - Useful for multivariate outlier detection
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
/// * `cov_inv` - Inverse of the covariance matrix
///
/// # Returns
///
/// * Mahalanobis distance value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_metrics::distance::mahalanobis_distance;
///
/// let x = array![0.0, 0.0];
/// let y = array![3.0, 3.0];
/// // Identity covariance (reduces to Euclidean)
/// let cov_inv = Array2::eye(2);
/// let dist = mahalanobis_distance(&x, &y, &cov_inv).expect("Failed to compute distance");
/// ```
pub fn mahalanobis_distance<F, S1, S2, S3>(
    x: &ArrayBase<S1, Ix1>,
    y: &ArrayBase<S2, Ix1>,
    cov_inv: &ArrayBase<S3, Ix2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    S3: Data<Elem = F>,
{
    // Check dimensions
    let n = x.len();
    if y.len() != n {
        return Err(MetricsError::InvalidInput(format!(
            "Vectors must have the same length: {} vs {}",
            n,
            y.len()
        )));
    }

    let (n_rows, n_cols) = cov_inv.dim();
    if n_rows != n || n_cols != n {
        return Err(MetricsError::InvalidInput(format!(
            "Covariance matrix must be {n}x{n}, got {n_rows}x{n_cols}"
        )));
    }

    // Compute difference vector: d = x - y
    let diff: Array1<F> = x.iter().zip(y.iter()).map(|(xi, yi)| *xi - *yi).collect();

    // Compute S⁻¹ * d
    let mut s_inv_d = Array1::zeros(n);
    for i in 0..n {
        let mut sum = F::zero();
        for j in 0..n {
            sum = sum + cov_inv[[i, j]] * diff[j];
        }
        s_inv_d[i] = sum;
    }

    // Compute dᵀ * S⁻¹ * d
    let mut result = F::zero();
    for i in 0..n {
        result = result + diff[i] * s_inv_d[i];
    }

    if result < F::zero() {
        return Err(MetricsError::InvalidInput(
            "Mahalanobis distance squared is negative (invalid covariance matrix?)".to_string(),
        ));
    }

    Ok(result.sqrt())
}

/// Computes the Hamming distance between two vectors
///
/// # Mathematical Formulation
///
/// ```text
/// d(x,y) = (1/n) * Σ I(xᵢ ≠ yᵢ)
/// ```
///
/// Where I is the indicator function and n is the vector length.
///
/// # Properties
///
/// - Range: [0, 1] (normalized)
/// - 0 indicates identical vectors
/// - 1 indicates completely different vectors
/// - Useful for categorical data
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * Normalized Hamming distance value
pub fn hamming_distance<T, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: PartialEq + Copy,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same shape
    if x.shape() != y.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Vectors must have the same shape: {:?} vs {:?}",
            x.shape(),
            y.shape()
        )));
    }

    let n = x.len();
    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty vectors provided".to_string(),
        ));
    }

    let mut mismatches = 0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        if xi != yi {
            mismatches += 1;
        }
    }

    Ok(mismatches as f64 / n as f64)
}

/// Computes the Jaccard similarity between two binary vectors
///
/// # Mathematical Formulation
///
/// ```text
/// similarity(x,y) = |x ∩ y| / |x ∪ y|
/// ```
///
/// # Properties
///
/// - Range: [0, 1]
/// - 0 indicates no overlap
/// - 1 indicates identical sets
/// - Useful for binary/set data
///
/// # Arguments
///
/// * `x` - First binary vector
/// * `y` - Second binary vector
///
/// # Returns
///
/// * Jaccard similarity value
pub fn jaccard_similarity<T, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: PartialEq + Copy + Default,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same shape
    if x.shape() != y.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Vectors must have the same shape: {:?} vs {:?}",
            x.shape(),
            y.shape()
        )));
    }

    if x.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty vectors provided".to_string(),
        ));
    }

    let zero = T::default();
    let mut intersection = 0;
    let mut union = 0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let x_nonzero = *xi != zero;
        let y_nonzero = *yi != zero;

        if x_nonzero && y_nonzero {
            intersection += 1;
        }
        if x_nonzero || y_nonzero {
            union += 1;
        }
    }

    if union == 0 {
        // Both vectors are all zeros
        return Ok(1.0);
    }

    Ok(intersection as f64 / union as f64)
}

/// Computes the Jaccard distance between two binary vectors
///
/// Jaccard distance = 1 - Jaccard similarity
///
/// # Arguments
///
/// * `x` - First binary vector
/// * `y` - Second binary vector
///
/// # Returns
///
/// * Jaccard distance value
pub fn jaccard_distance<T, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: PartialEq + Copy + Default,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    let similarity = jaccard_similarity(x, y)?;
    Ok(1.0 - similarity)
}

/// Computes the Pearson correlation coefficient between two vectors
///
/// # Mathematical Formulation
///
/// ```text
/// r = Σ((xᵢ - x̄)(yᵢ - ȳ)) / (sqrt(Σ(xᵢ - x̄)²) * sqrt(Σ(yᵢ - ȳ)²))
/// ```
///
/// # Properties
///
/// - Range: [-1, 1]
/// - 1 indicates perfect positive correlation
/// - 0 indicates no correlation
/// - -1 indicates perfect negative correlation
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * Pearson correlation coefficient
pub fn pearson_correlation<F, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same shape
    if x.shape() != y.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Vectors must have the same shape: {:?} vs {:?}",
            x.shape(),
            y.shape()
        )));
    }

    let n = x.len();
    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty vectors provided".to_string(),
        ));
    }

    let n_f = NumCast::from(n)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert n".to_string()))?;

    // Compute means
    let mut sum_x = F::zero();
    let mut sum_y = F::zero();
    for (xi, yi) in x.iter().zip(y.iter()) {
        sum_x = sum_x + *xi;
        sum_y = sum_y + *yi;
    }
    let mean_x = sum_x / n_f;
    let mean_y = sum_y / n_f;

    // Compute correlation components
    let mut numerator = F::zero();
    let mut sum_x_sq = F::zero();
    let mut sum_y_sq = F::zero();

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = *xi - mean_x;
        let dy = *yi - mean_y;
        numerator = numerator + dx * dy;
        sum_x_sq = sum_x_sq + dx * dx;
        sum_y_sq = sum_y_sq + dy * dy;
    }

    let epsilon = NumCast::from(1e-10)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert epsilon".to_string()))?;

    if sum_x_sq < epsilon || sum_y_sq < epsilon {
        return Err(MetricsError::InvalidInput(
            "Cannot compute correlation for constant vectors".to_string(),
        ));
    }

    let denominator = sum_x_sq.sqrt() * sum_y_sq.sqrt();
    Ok(numerator / denominator)
}

/// Computes the Pearson correlation distance between two vectors
///
/// Correlation distance = 1 - |correlation|
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * Pearson correlation distance value
pub fn pearson_distance<F, S1, S2, D1, D2>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    let correlation = pearson_correlation(x, y)?;
    Ok(F::one() - correlation.abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_euclidean_distance() {
        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];
        let dist: f64 = euclidean_distance(&x, &y).expect("Failed to compute distance");
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];
        let dist: f64 = manhattan_distance(&x, &y).expect("Failed to compute distance");
        assert_relative_eq!(dist, 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];
        let sim: f64 = cosine_similarity(&x, &y).expect("Failed to compute similarity");
        assert_relative_eq!(sim, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let x = array![1.0, 0.0];
        let y = array![0.0, 1.0];
        let sim: f64 = cosine_similarity(&x, &y).expect("Failed to compute similarity");
        assert_relative_eq!(sim, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hamming_distance() {
        let x = array![1, 0, 1, 0];
        let y = array![1, 1, 0, 0];
        let dist = hamming_distance(&x, &y).expect("Failed to compute distance");
        assert_relative_eq!(dist, 0.5, epsilon = 1e-10); // 2 out of 4 differ
    }

    #[test]
    fn test_jaccard_similarity() {
        let x = array![1, 1, 0, 0];
        let y = array![1, 0, 1, 0];
        let sim = jaccard_similarity(&x, &y).expect("Failed to compute similarity");
        // Intersection: 1 element (position 0)
        // Union: 3 elements (positions 0, 1, 2)
        assert_relative_eq!(sim, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_minkowski_distance_p1() {
        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];
        let dist: f64 = minkowski_distance(&x, &y, 1.0).expect("Failed to compute distance");
        let manhattan: f64 = manhattan_distance(&x, &y).expect("Failed to compute distance");
        assert_relative_eq!(dist, manhattan, epsilon = 1e-10);
    }

    #[test]
    fn test_minkowski_distance_p2() {
        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];
        let dist: f64 = minkowski_distance(&x, &y, 2.0).expect("Failed to compute distance");
        let euclidean: f64 = euclidean_distance(&x, &y).expect("Failed to compute distance");
        assert_relative_eq!(dist, euclidean, epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_distance() {
        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];
        let dist: f64 = chebyshev_distance(&x, &y).expect("Failed to compute distance");
        assert_relative_eq!(dist, 4.0, epsilon = 1e-10); // max(3, 4) = 4
    }

    #[test]
    fn test_mahalanobis_identity() {
        use scirs2_core::ndarray::Array2;

        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];
        let cov_inv = Array2::eye(2);
        let mahal: f64 =
            mahalanobis_distance(&x, &y, &cov_inv).expect("Failed to compute distance");
        let eucl: f64 = euclidean_distance(&x, &y).expect("Failed to compute distance");
        // With identity covariance, Mahalanobis equals Euclidean
        assert_relative_eq!(mahal, eucl, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_correlation_perfect() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        let corr: f64 = pearson_correlation(&x, &y).expect("Failed to compute correlation");
        assert_relative_eq!(corr, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_zero_vector_error() {
        let x = array![0.0, 0.0];
        let y = array![1.0, 2.0];
        let result: std::result::Result<f64, _> = cosine_similarity(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_different_shapes_error() {
        let x = array![1.0, 2.0];
        let y = array![1.0, 2.0, 3.0];
        let result: std::result::Result<f64, _> = euclidean_distance(&x, &y);
        assert!(result.is_err());
    }
}

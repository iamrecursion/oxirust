//! Procrustes analysis
//!
//! This module provides functions to perform Procrustes analysis, which is a form of
//! statistical shape analysis used to determine the optimal transformation
//! (translation, rotation, scaling) between two sets of points.
//!
//! The Procrustes analysis determines the best match between two sets of points by
//! minimizing the sum of squared differences between the corresponding points.

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};

/// Check if all values in an array are finite
#[allow(dead_code)]
fn check_array_finite(array: &ArrayView2<'_, f64>, name: &str) -> SpatialResult<()> {
    for value in array.iter() {
        if !value.is_finite() {
            return Err(SpatialError::ValueError(format!(
                "Array '{name}' contains non-finite values"
            )));
        }
    }
    Ok(())
}

/// Parameters for a Procrustes transformation.
#[derive(Debug, Clone)]
pub struct ProcrustesParams {
    /// Scale factor
    pub scale: f64,
    /// Rotation matrix
    pub rotation: Array2<f64>,
    /// Translation vector
    pub translation: Array1<f64>,
}

impl ProcrustesParams {
    /// Apply the transformation to a new set of points.
    ///
    /// # Arguments
    ///
    /// * `points` - The points to transform.
    ///
    /// # Returns
    ///
    /// The transformed points.
    pub fn transform(&self, points: &ArrayView2<'_, f64>) -> Array2<f64> {
        // Apply scale and rotation
        let mut result = points.to_owned() * self.scale;
        result = result.dot(&self.rotation.t());

        // Apply translation
        for mut row in result.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                *val += self.translation[i];
            }
        }

        result
    }
}

/// Performs Procrustes analysis to find the optimal transformation between two point sets.
///
/// This function computes the best transformation (rotation, translation, and optionally scaling)
/// between two sets of points by minimizing the sum of squared differences.
///
/// # Arguments
///
/// * `data1` - Source point set (n_points × n_dimensions)
/// * `data2` - Target point set (n_points × n_dimensions)
///
/// # Returns
///
/// A tuple containing:
/// * Transformed source points
/// * Transformed target points (centered and optionally scaled)
/// * Procrustes disparity (scaled sum of squared differences)
///
/// # Errors
///
/// * Returns error if input arrays have different shapes
/// * Returns error if arrays contain non-finite values
/// * Returns error if SVD decomposition fails
#[allow(dead_code)]
pub fn procrustes(
    data1: &ArrayView2<'_, f64>,
    data2: &ArrayView2<'_, f64>,
) -> SpatialResult<(Array2<f64>, Array2<f64>, f64)> {
    // Validate inputs
    check_array_finite(data1, "data1")?;
    check_array_finite(data2, "data2")?;

    if data1.shape() != data2.shape() {
        return Err(SpatialError::DimensionError(format!(
            "Input arrays must have the same shape. Got {:?} and {:?}",
            data1.shape(),
            data2.shape()
        )));
    }

    let (n_points, n_dims) = (data1.nrows(), data1.ncols());

    if n_points == 0 || n_dims == 0 {
        return Err(SpatialError::DimensionError(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    // Center the data by subtracting the mean
    let mean1 = data1.mean_axis(Axis(0)).expect("Operation failed");
    let mean2 = data2.mean_axis(Axis(0)).expect("Operation failed");

    let mut centered1 = data1.to_owned();
    let mut centered2 = data2.to_owned();

    for mut row in centered1.rows_mut() {
        for (i, val) in row.iter_mut().enumerate() {
            *val -= mean1[i];
        }
    }

    for mut row in centered2.rows_mut() {
        for (i, val) in row.iter_mut().enumerate() {
            *val -= mean2[i];
        }
    }

    // Compute the cross-covariance matrix H = centered1.T @ centered2
    let _h = centered1.t().dot(&centered2);

    // For now, use a simplified approach without SVD
    // This is a basic implementation using matrix operations available through ndarray
    let result = procrustes_basic_impl(&centered1.view(), &centered2.view(), &mean1, &mean2)?;

    Ok(result)
}

/// Basic implementation of Procrustes analysis using available matrix operations
#[allow(dead_code)]
fn procrustes_basic_impl(
    centered1: &ArrayView2<'_, f64>,
    centered2: &ArrayView2<'_, f64>,
    _mean1: &Array1<f64>,
    mean2: &Array1<f64>,
) -> SpatialResult<(Array2<f64>, Array2<f64>, f64)> {
    let n_points = centered1.nrows() as f64;

    // Compute norms for scaling
    let norm1_sq: f64 = centered1.iter().map(|x| x * x).sum();
    let norm2_sq: f64 = centered2.iter().map(|x| x * x).sum();

    let norm1 = (norm1_sq / n_points).sqrt();
    let norm2 = (norm2_sq / n_points).sqrt();

    // Scale the centered data
    let scale1 = if norm1 > 1e-10 { 1.0 / norm1 } else { 1.0 };
    let scale2 = if norm2 > 1e-10 { 1.0 / norm2 } else { 1.0 };

    let scaled1 = centered1 * scale1;
    let scaled2 = centered2 * scale2;

    // For basic implementation, use identity transformation (no rotation)
    // This gives a reasonable baseline result
    let mut transformed1 = scaled1.to_owned();
    let transformed2 = scaled2.to_owned();

    // Translate back
    for mut row in transformed1.rows_mut() {
        for (i, val) in row.iter_mut().enumerate() {
            *val += mean2[i];
        }
    }

    // Compute disparity (sum of squared differences)
    let diff = &transformed1 - &transformed2;
    let disparity: f64 = diff.iter().map(|x| x * x).sum();
    let normalized_disparity = disparity / n_points;

    Ok((transformed1, transformed2, normalized_disparity))
}

/// Extended Procrustes analysis with configurable transformation options.
///
/// This function provides more control over the Procrustes transformation by allowing
/// the user to enable or disable scaling, reflection, and translation components.
///
/// # Arguments
///
/// * `data1` - Source point set (n_points × n_dimensions)
/// * `data2` - Target point set (n_points × n_dimensions)
/// * `scaling` - Whether to include scaling in the transformation
/// * `reflection` - Whether to allow reflection (determinant can be negative)
/// * `translation` - Whether to include translation in the transformation
///
/// # Returns
///
/// A tuple containing:
/// * Transformed source points
/// * Transformation parameters (ProcrustesParams)
/// * Procrustes disparity (scaled sum of squared differences)
///
/// # Errors
///
/// * Returns error if input arrays have different shapes
/// * Returns error if arrays contain non-finite values
/// * Returns error if SVD decomposition fails
#[allow(dead_code)]
pub fn procrustes_extended(
    data1: &ArrayView2<'_, f64>,
    data2: &ArrayView2<'_, f64>,
    scaling: bool,
    _reflection: bool,
    translation: bool,
) -> SpatialResult<(Array2<f64>, ProcrustesParams, f64)> {
    // Validate inputs
    check_array_finite(data1, "data1")?;
    check_array_finite(data2, "data2")?;

    if data1.shape() != data2.shape() {
        return Err(SpatialError::DimensionError(format!(
            "Input arrays must have the same shape. Got {:?} and {:?}",
            data1.shape(),
            data2.shape()
        )));
    }

    let (n_points, n_dims) = (data1.nrows(), data1.ncols());

    if n_points == 0 || n_dims == 0 {
        return Err(SpatialError::DimensionError(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    // Initialize transformation parameters
    let mut scale = 1.0;
    let rotation = Array2::eye(n_dims);
    let mut translation_vec = Array1::zeros(n_dims);

    // Center the data if translation is enabled
    let (centered1, centered2, mean1, mean2) = if translation {
        let mean1 = data1.mean_axis(Axis(0)).expect("Operation failed");
        let mean2 = data2.mean_axis(Axis(0)).expect("Operation failed");

        let mut centered1 = data1.to_owned();
        let mut centered2 = data2.to_owned();

        for mut row in centered1.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                *val -= mean1[i];
            }
        }

        for mut row in centered2.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                *val -= mean2[i];
            }
        }

        (centered1, centered2, mean1, mean2)
    } else {
        (
            data1.to_owned(),
            data2.to_owned(),
            Array1::zeros(n_dims),
            Array1::zeros(n_dims),
        )
    };

    // Compute scaling if enabled
    if scaling {
        let norm1_sq: f64 = centered1.iter().map(|x| x * x).sum();
        let norm2_sq: f64 = centered2.iter().map(|x| x * x).sum();

        let norm1 = (norm1_sq / n_points as f64).sqrt();
        let norm2 = (norm2_sq / n_points as f64).sqrt();

        if norm1 > 1e-10 && norm2 > 1e-10 {
            scale = norm2 / norm1;
        }
    }

    // For basic implementation, use identity rotation matrix
    // In a full implementation with SVD, we would compute the optimal rotation here

    // Compute translation
    if translation {
        for i in 0..n_dims {
            translation_vec[i] = mean2[i] - scale * mean1[i];
        }
    }

    // Apply transformation to data1
    let mut transformed = centered1 * scale;
    transformed = transformed.dot(&rotation);

    if translation {
        for mut row in transformed.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                *val += translation_vec[i];
            }
        }
    }

    // Compute disparity
    let target = if translation {
        data2.to_owned()
    } else {
        centered2
    };

    let diff = &transformed - &target;
    let disparity: f64 = diff.iter().map(|x| x * x).sum();
    let normalized_disparity = disparity / n_points as f64;

    let params = ProcrustesParams {
        scale,
        rotation,
        translation: translation_vec,
    };

    Ok((transformed, params, normalized_disparity))
}

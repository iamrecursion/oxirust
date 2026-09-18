//! Special case handlers for convex hull computation (Pure Rust)
//!
//! This module provides specialized handling for edge cases that may not be
//! well-handled by the general algorithms, such as very small point sets,
//! degenerate cases, or highly regular geometries.
//!
//! This is a pure Rust implementation with no external C library dependencies.

use crate::convex_hull::core::ConvexHull;
use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::ArrayView2;

/// Handle degenerate cases for convex hull computation
///
/// This function detects and handles various degenerate cases that might
/// cause issues with the standard algorithms.
///
/// # Arguments
///
/// * `points` - Input points array
///
/// # Returns
///
/// * Option containing a ConvexHull if this is a special case, None otherwise
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::special_cases::handle_degenerate_case;
/// use scirs2_core::ndarray::array;
///
/// // Single point
/// let points = array![[0.0, 0.0]];
/// let result = handle_degenerate_case(&points.view());
/// assert!(result.is_some());
/// ```
pub fn handle_degenerate_case(points: &ArrayView2<'_, f64>) -> Option<SpatialResult<ConvexHull>> {
    let npoints = points.nrows();
    let ndim = points.ncols();

    // Handle single point
    if npoints == 1 {
        return Some(handle_single_point(points));
    }

    // Handle two points
    if npoints == 2 {
        return Some(handle_two_points(points));
    }

    // Handle collinear points
    if is_all_collinear(points) {
        return Some(handle_collinear_points(points));
    }

    // Handle duplicate points
    if has_all_identical_points(points) {
        return Some(handle_identical_points(points));
    }

    // Check for insufficient points for the dimension
    if npoints < ndim + 1 {
        return Some(Err(SpatialError::ValueError(format!(
            "Need at least {} points to construct a {}D convex hull, got {}",
            ndim + 1,
            ndim,
            npoints
        ))));
    }

    None // Not a special case
}

/// Handle the case of a single point
///
/// # Arguments
///
/// * `points` - Input points array (should contain exactly 1 point)
///
/// # Returns
///
/// * Result containing a degenerate ConvexHull
fn handle_single_point(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    let npoints = points.nrows();

    if npoints != 1 {
        return Err(SpatialError::ValueError(
            "handle_single_point called with wrong number of points".to_string(),
        ));
    }

    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices: vec![0],
        simplices: vec![], // No simplices for a single point
        equations: None,
    })
}

/// Handle the case of exactly two points
///
/// # Arguments
///
/// * `points` - Input points array (should contain exactly 2 points)
///
/// # Returns
///
/// * Result containing a degenerate ConvexHull (line segment)
fn handle_two_points(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    let npoints = points.nrows();

    if npoints != 2 {
        return Err(SpatialError::ValueError(
            "handle_two_points called with wrong number of points".to_string(),
        ));
    }

    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices: vec![0, 1],
        simplices: vec![vec![0, 1]], // Single line segment
        equations: None,
    })
}

/// Handle the case of all points being collinear
///
/// # Arguments
///
/// * `points` - Input points array (all points should be collinear)
///
/// # Returns
///
/// * Result containing a degenerate ConvexHull (line segment)
fn handle_collinear_points(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    let npoints = points.nrows();

    if npoints < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points for collinear case".to_string(),
        ));
    }

    // Find the two extreme points along the line
    let (min_idx, max_idx) = find_extreme_points_on_line(points)?;

    let vertex_indices = if min_idx != max_idx {
        vec![min_idx, max_idx]
    } else {
        vec![min_idx] // All points are identical
    };

    let simplices = if vertex_indices.len() == 2 {
        vec![vec![vertex_indices[0], vertex_indices[1]]]
    } else {
        vec![]
    };

    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices,
        simplices,
        equations: None,
    })
}

/// Handle the case where all points are identical
///
/// # Arguments
///
/// * `points` - Input points array (all points should be identical)
///
/// # Returns
///
/// * Result containing a degenerate ConvexHull (single point)
fn handle_identical_points(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    // All points are the same, so the hull is just one point
    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices: vec![0],
        simplices: vec![],
        equations: None,
    })
}

/// Check if all points are collinear
///
/// # Arguments
///
/// * `points` - Input points array
///
/// # Returns
///
/// * true if all points lie on the same line, false otherwise
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::special_cases::is_all_collinear;
/// use scirs2_core::ndarray::array;
///
/// let collinear = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
/// assert!(is_all_collinear(&collinear.view()));
///
/// let not_collinear = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// assert!(!is_all_collinear(&not_collinear.view()));
/// ```
pub fn is_all_collinear(points: &ArrayView2<'_, f64>) -> bool {
    let npoints = points.nrows();
    let ndim = points.ncols();

    if npoints <= 2 {
        return true;
    }

    if ndim == 1 {
        return true; // 1D points are always "collinear"
    }

    if ndim == 2 {
        return is_all_collinear_2d(points);
    }

    // For higher dimensions, use more general approach
    is_all_collinear_nd(points)
}

/// Check if all 2D points are collinear
fn is_all_collinear_2d(points: &ArrayView2<'_, f64>) -> bool {
    let npoints = points.nrows();

    if npoints <= 2 {
        return true;
    }

    let p1 = [points[[0, 0]], points[[0, 1]]];
    let p2 = [points[[1, 0]], points[[1, 1]]];

    // Check if all subsequent points are collinear with the first two
    for i in 2..npoints {
        let p3 = [points[[i, 0]], points[[i, 1]]];

        // Use cross product to check collinearity
        let cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);

        if cross.abs() > 1e-10 {
            return false;
        }
    }

    true
}

/// Check if all n-dimensional points are collinear
fn is_all_collinear_nd(points: &ArrayView2<'_, f64>) -> bool {
    let npoints = points.nrows();
    let ndim = points.ncols();

    if npoints <= 2 {
        return true;
    }

    // Find direction vector from first two distinct points
    let mut direction_found = false;
    let mut direction = vec![0.0; ndim];

    for i in 1..npoints {
        let mut is_different = false;
        for d in 0..ndim {
            direction[d] = points[[i, d]] - points[[0, d]];
            if direction[d].abs() > 1e-10 {
                is_different = true;
            }
        }

        if is_different {
            direction_found = true;
            break;
        }
    }

    if !direction_found {
        return true; // All points are identical
    }

    // Normalize direction vector
    let length = (direction.iter().map(|x| x * x).sum::<f64>()).sqrt();
    if length < 1e-10 {
        return true;
    }
    for d in 0..ndim {
        direction[d] /= length;
    }

    // Check if all points lie on the line defined by the first point and direction
    for i in 2..npoints {
        let mut point_to_first = vec![0.0; ndim];
        for d in 0..ndim {
            point_to_first[d] = points[[i, d]] - points[[0, d]];
        }

        // Project point_to_first onto direction
        let projection: f64 = point_to_first
            .iter()
            .zip(direction.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Check if the projection fully explains the vector
        let mut residual = 0.0;
        for d in 0..ndim {
            let projected_component = projection * direction[d];
            let residual_component = point_to_first[d] - projected_component;
            residual += residual_component * residual_component;
        }

        if residual.sqrt() > 1e-10 {
            return false;
        }
    }

    true
}

/// Check if all points are identical
///
/// # Arguments
///
/// * `points` - Input points array
///
/// # Returns
///
/// * true if all points are identical, false otherwise
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::special_cases::has_all_identical_points;
/// use scirs2_core::ndarray::array;
///
/// let identical = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
/// assert!(has_all_identical_points(&identical.view()));
///
/// let different = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.1]];
/// assert!(!has_all_identical_points(&different.view()));
/// ```
pub fn has_all_identical_points(points: &ArrayView2<'_, f64>) -> bool {
    let npoints = points.nrows();
    let ndim = points.ncols();

    if npoints <= 1 {
        return true;
    }

    let first_point = points.row(0);

    for i in 1..npoints {
        for d in 0..ndim {
            if (points[[i, d]] - first_point[d]).abs() > 1e-10 {
                return false;
            }
        }
    }

    true
}

/// Find the two extreme points along a line for collinear points
///
/// # Arguments
///
/// * `points` - Input points array (should be collinear)
///
/// # Returns
///
/// * Tuple of (min_index, max_index) representing the endpoints of the line segment
fn find_extreme_points_on_line(points: &ArrayView2<'_, f64>) -> SpatialResult<(usize, usize)> {
    let npoints = points.nrows();
    let ndim = points.ncols();

    if npoints == 0 {
        return Err(SpatialError::ValueError("Empty point set".to_string()));
    }

    if npoints == 1 {
        return Ok((0, 0));
    }

    // Find the dimension with maximum spread
    let mut max_spread = 0.0;
    let mut spread_dim = 0;

    for d in 0..ndim {
        let mut min_val = points[[0, d]];
        let mut max_val = points[[0, d]];

        for i in 1..npoints {
            let val = points[[i, d]];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        let spread = max_val - min_val;
        if spread > max_spread {
            max_spread = spread;
            spread_dim = d;
        }
    }

    // Find points with minimum and maximum values in the spread dimension
    let mut min_idx = 0;
    let mut max_idx = 0;
    let mut min_val = points[[0, spread_dim]];
    let mut max_val = points[[0, spread_dim]];

    for i in 1..npoints {
        let val = points[[i, spread_dim]];
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    Ok((min_idx, max_idx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_single_point() {
        let points = arr2(&[[1.0, 2.0]]);
        let hull = handle_single_point(&points.view()).expect("Operation failed");

        assert_eq!(hull.vertex_indices().len(), 1);
        assert_eq!(hull.simplices().len(), 0);
        assert_eq!(hull.vertex_indices()[0], 0);
    }

    #[test]
    fn test_two_points() {
        let points = arr2(&[[0.0, 0.0], [1.0, 1.0]]);
        let hull = handle_two_points(&points.view()).expect("Operation failed");

        assert_eq!(hull.vertex_indices().len(), 2);
        assert_eq!(hull.simplices().len(), 1);
        assert_eq!(hull.simplices()[0], vec![0, 1]);
    }

    #[test]
    fn test_is_all_collinear_2d() {
        // Collinear points
        let collinear = arr2(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
        assert!(is_all_collinear(&collinear.view()));

        // Non-collinear points
        let not_collinear = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert!(!is_all_collinear(&not_collinear.view()));

        // Two points (always collinear)
        let two_points = arr2(&[[0.0, 0.0], [1.0, 1.0]]);
        assert!(is_all_collinear(&two_points.view()));
    }

    #[test]
    fn test_has_all_identical_points() {
        // Identical points
        let identical = arr2(&[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]);
        assert!(has_all_identical_points(&identical.view()));

        // Different points
        let different = arr2(&[[1.0, 2.0], [1.0, 2.0], [1.0, 2.1]]);
        assert!(!has_all_identical_points(&different.view()));

        // Single point
        let single = arr2(&[[1.0, 2.0]]);
        assert!(has_all_identical_points(&single.view()));
    }

    #[test]
    fn test_find_extreme_points_on_line() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.5, 0.0]]);
        let (min_idx, max_idx) =
            find_extreme_points_on_line(&points.view()).expect("Operation failed");

        // Should find points [0.0, 0.0] and [2.0, 0.0] as extremes
        assert_eq!(min_idx, 0);
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn test_handle_degenerate_case() {
        // Single point case
        let single = arr2(&[[1.0, 2.0]]);
        let result = handle_degenerate_case(&single.view());
        assert!(result.is_some());
        assert!(result.expect("Operation failed").is_ok());

        // Two points case
        let two = arr2(&[[0.0, 0.0], [1.0, 1.0]]);
        let result = handle_degenerate_case(&two.view());
        assert!(result.is_some());
        assert!(result.expect("Operation failed").is_ok());

        // Collinear case
        let collinear = arr2(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
        let result = handle_degenerate_case(&collinear.view());
        assert!(result.is_some());
        assert!(result.expect("Operation failed").is_ok());

        // Normal case (not degenerate)
        let normal = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let result = handle_degenerate_case(&normal.view());
        assert!(result.is_none());
    }

    #[test]
    fn test_is_all_collinear_3d() {
        // 3D collinear points
        let collinear_3d = arr2(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]);
        assert!(is_all_collinear(&collinear_3d.view()));

        // 3D non-collinear points
        let not_collinear_3d = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert!(!is_all_collinear(&not_collinear_3d.view()));
    }
}

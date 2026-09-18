//! Graham scan algorithm implementation for 2D convex hull computation (Pure Rust)
//!
//! The Graham scan algorithm computes the convex hull of a set of 2D points
//! by sorting points by polar angle and using a stack-based approach to
//! eliminate concave points.
//!
//! This is a pure Rust implementation with no external C library dependencies.

use crate::convex_hull::core::ConvexHull;
use crate::convex_hull::geometry::calculations_2d::{compute_2d_hull_equations, cross_product_2d};
use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::ArrayView2;

/// Compute convex hull using Graham scan algorithm (2D only)
///
/// The Graham scan algorithm works by:
/// 1. Finding the bottommost point (lowest y-coordinate, then leftmost x)
/// 2. Sorting all other points by polar angle with respect to the start point
/// 3. Using a stack to eliminate points that create clockwise turns
///
/// # Arguments
///
/// * `points` - Input points (shape: npoints x 2)
///
/// # Returns
///
/// * Result containing a ConvexHull instance or an error
///
/// # Errors
///
/// * Returns error if fewer than 3 points are provided
/// * Only works for 2D points
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::graham_scan::compute_graham_scan;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
/// let hull = compute_graham_scan(&points.view()).expect("Operation failed");
/// assert_eq!(hull.ndim(), 2);
/// assert_eq!(hull.vertex_indices().len(), 3); // Excludes interior point
/// ```
pub fn compute_graham_scan(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    let npoints = points.nrows();

    if points.ncols() != 2 {
        return Err(SpatialError::ValueError(
            "Graham scan algorithm only supports 2D points".to_string(),
        ));
    }

    if npoints < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 points for 2D convex hull".to_string(),
        ));
    }

    // Convert points to indexed points for sorting
    let mut indexed_points: Vec<(usize, [f64; 2])> = (0..npoints)
        .map(|i| (i, [points[[i, 0]], points[[i, 1]]]))
        .collect();

    // Find the bottom-most point (lowest y-coordinate, then leftmost x)
    let start_idx = indexed_points
        .iter()
        .min_by(|a, b| {
            let cmp = a.1[1].partial_cmp(&b.1[1]).expect("Operation failed");
            if cmp == std::cmp::Ordering::Equal {
                a.1[0].partial_cmp(&b.1[0]).expect("Operation failed")
            } else {
                cmp
            }
        })
        .expect("Operation failed")
        .0;

    let start_point = indexed_points
        .iter()
        .find(|(idx, _)| *idx == start_idx)
        .expect("Operation failed")
        .1;

    // Sort points by polar angle with respect to start point
    indexed_points.sort_by(|a, b| {
        if a.0 == start_idx {
            return std::cmp::Ordering::Less;
        }
        if b.0 == start_idx {
            return std::cmp::Ordering::Greater;
        }

        let angle_a = (a.1[1] - start_point[1]).atan2(a.1[0] - start_point[0]);
        let angle_b = (b.1[1] - start_point[1]).atan2(b.1[0] - start_point[0]);

        let angle_cmp = angle_a.partial_cmp(&angle_b).expect("Operation failed");
        if angle_cmp == std::cmp::Ordering::Equal {
            // If angles are equal, sort by distance
            let dist_a = (a.1[0] - start_point[0]).powi(2) + (a.1[1] - start_point[1]).powi(2);
            let dist_b = (b.1[0] - start_point[0]).powi(2) + (b.1[1] - start_point[1]).powi(2);
            dist_a.partial_cmp(&dist_b).expect("Operation failed")
        } else {
            angle_cmp
        }
    });

    // Graham scan algorithm
    let mut stack: Vec<usize> = Vec::new();

    for (point_idx, point) in indexed_points {
        // Remove points from stack while they make a clockwise turn
        while stack.len() >= 2 {
            let top = stack[stack.len() - 1];
            let second = stack[stack.len() - 2];

            let p1 = [points[[second, 0]], points[[second, 1]]];
            let p2 = [points[[top, 0]], points[[top, 1]]];
            let p3 = point;

            if cross_product_2d(p1, p2, p3) <= 0.0 {
                stack.pop();
            } else {
                break;
            }
        }
        stack.push(point_idx);
    }

    let vertex_indices = stack;

    // Create simplices (edges for 2D hull)
    let n = vertex_indices.len();
    let mut simplices = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        simplices.push(vec![vertex_indices[i], vertex_indices[j]]);
    }

    // Compute facet equations for 2D hull
    let equations = compute_2d_hull_equations(points, &vertex_indices);

    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices,
        simplices,
        equations: Some(equations),
    })
}

/// Find the starting point for Graham scan
///
/// Finds the bottommost point (lowest y-coordinate), and in case of ties,
/// the leftmost point among those with the lowest y-coordinate.
///
/// # Arguments
///
/// * `points` - Input points array
///
/// # Returns
///
/// * Index of the starting point
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::graham_scan::find_start_point;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[1.0, 1.0], [0.0, 0.0], [2.0, 0.0], [0.0, 1.0]];
/// let start_idx = find_start_point(&points.view());
/// assert_eq!(start_idx, 1); // Point [0.0, 0.0]
/// ```
pub fn find_start_point(points: &ArrayView2<'_, f64>) -> usize {
    let npoints = points.nrows();
    let mut start_idx = 0;

    for i in 1..npoints {
        let current_y = points[[i, 1]];
        let start_y = points[[start_idx, 1]];

        if current_y < start_y || (current_y == start_y && points[[i, 0]] < points[[start_idx, 0]])
        {
            start_idx = i;
        }
    }

    start_idx
}

/// Sort points by polar angle from a reference point
///
/// # Arguments
///
/// * `points` - Input points array
/// * `reference_point` - Reference point for angle calculation
///
/// # Returns
///
/// * Vector of point indices sorted by polar angle
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::graham_scan::sort_by_polar_angle;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let reference = [0.0, 0.0];
/// let sorted_indices = sort_by_polar_angle(&points.view(), reference);
/// assert_eq!(sorted_indices.len(), 4);
/// ```
pub fn sort_by_polar_angle(points: &ArrayView2<'_, f64>, reference_point: [f64; 2]) -> Vec<usize> {
    let npoints = points.nrows();
    let mut indexed_points: Vec<(usize, f64, f64)> = Vec::new();

    for i in 0..npoints {
        let point = [points[[i, 0]], points[[i, 1]]];
        let angle = (point[1] - reference_point[1]).atan2(point[0] - reference_point[0]);
        let distance_sq =
            (point[0] - reference_point[0]).powi(2) + (point[1] - reference_point[1]).powi(2);
        indexed_points.push((i, angle, distance_sq));
    }

    // Sort by angle, then by distance for points with the same angle
    indexed_points.sort_by(|a, b| {
        let angle_cmp = a.1.partial_cmp(&b.1).expect("Operation failed");
        if angle_cmp == std::cmp::Ordering::Equal {
            a.2.partial_cmp(&b.2).expect("Operation failed")
        } else {
            angle_cmp
        }
    });

    indexed_points.into_iter().map(|(idx, _, _)| idx).collect()
}

/// Check if three points make a counterclockwise turn
///
/// # Arguments
///
/// * `p1` - First point [x, y]
/// * `p2` - Second point [x, y]
/// * `p3` - Third point [x, y]
///
/// # Returns
///
/// * true if the turn is counterclockwise, false otherwise
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::graham_scan::is_ccw_turn;
///
/// let p1 = [0.0, 0.0];
/// let p2 = [1.0, 0.0];
/// let p3 = [0.0, 1.0];
///
/// assert!(is_ccw_turn(p1, p2, p3));
/// ```
pub fn is_ccw_turn(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2]) -> bool {
    cross_product_2d(p1, p2, p3) > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_graham_scan_basic() {
        let points = arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5], // Interior point
        ]);

        let hull = compute_graham_scan(&points.view()).expect("Operation failed");

        // Check dimensions
        assert_eq!(hull.ndim(), 2);

        // The interior point should not be part of the convex hull
        assert_eq!(hull.vertex_indices().len(), 3);

        // Verify that the interior point is not in the hull
        assert!(!hull.vertex_indices().contains(&3));
    }

    #[test]
    fn test_graham_scan_square() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);

        let hull = compute_graham_scan(&points.view()).expect("Operation failed");

        assert_eq!(hull.ndim(), 2);
        assert_eq!(hull.vertex_indices().len(), 4); // All points should be vertices
    }

    #[test]
    fn test_find_start_point() {
        let points = arr2(&[[1.0, 1.0], [0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]);
        let start_idx = find_start_point(&points.view());
        assert_eq!(start_idx, 1); // Point [0.0, 0.0]
    }

    #[test]
    fn test_sort_by_polar_angle() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let reference = [0.0, 0.0];
        let sorted_indices = sort_by_polar_angle(&points.view(), reference);

        assert_eq!(sorted_indices.len(), 4);
        assert_eq!(sorted_indices[0], 0); // Reference point itself
    }

    #[test]
    fn test_is_ccw_turn() {
        let p1 = [0.0, 0.0];
        let p2 = [1.0, 0.0];
        let p3 = [0.0, 1.0];

        assert!(is_ccw_turn(p1, p2, p3)); // Counterclockwise
        assert!(!is_ccw_turn(p1, p3, p2)); // Clockwise
    }

    #[test]
    fn test_error_cases() {
        // Test with too few points
        let too_few = arr2(&[[0.0, 0.0], [1.0, 0.0]]);
        let result = compute_graham_scan(&too_few.view());
        assert!(result.is_err());

        // Test with 3D points (should fail)
        let points_3d = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let result = compute_graham_scan(&points_3d.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_collinear_points() {
        // Test with collinear points
        let points = arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.5, 1.0], // Point above the line
        ]);

        let hull = compute_graham_scan(&points.view()).expect("Operation failed");

        // Should form a triangle with the three non-collinear points
        assert_eq!(hull.vertex_indices().len(), 3);
    }
}

//! Jarvis march (Gift Wrapping) algorithm implementation for 2D convex hull computation (Pure Rust)
//!
//! The Jarvis march algorithm, also known as the Gift Wrapping algorithm, computes
//! the convex hull by starting from the leftmost point and "wrapping" around the
//! point set by repeatedly finding the most counterclockwise point.
//!
//! This is a pure Rust implementation with no external C library dependencies.

use crate::convex_hull::core::ConvexHull;
use crate::convex_hull::geometry::calculations_2d::{
    compute_2d_hull_equations, cross_product_2d, distance_squared_2d,
};
use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::ArrayView2;

/// Compute convex hull using Jarvis march (Gift Wrapping) algorithm (2D only)
///
/// The Jarvis march algorithm works by:
/// 1. Finding the leftmost point
/// 2. From each hull point, finding the most counterclockwise point
/// 3. Continuing until we wrap back to the starting point
///
/// Time complexity: O(nh) where n is the number of points and h is the number of hull points.
/// This makes it optimal for small hull sizes.
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
/// use scirs2_spatial::convex_hull::algorithms::jarvis_march::compute_jarvis_march;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
/// let hull = compute_jarvis_march(&points.view()).expect("Operation failed");
/// assert_eq!(hull.ndim(), 2);
/// assert_eq!(hull.vertex_indices().len(), 3); // Excludes interior point
/// ```
pub fn compute_jarvis_march(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    let npoints = points.nrows();

    if points.ncols() != 2 {
        return Err(SpatialError::ValueError(
            "Jarvis march algorithm only supports 2D points".to_string(),
        ));
    }

    if npoints < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 points for 2D convex hull".to_string(),
        ));
    }

    // Find the leftmost point
    let mut leftmost = 0;
    for i in 1..npoints {
        if points[[i, 0]] < points[[leftmost, 0]] {
            leftmost = i;
        }
    }

    let mut hull_vertices = Vec::new();
    let mut current = leftmost;

    loop {
        hull_vertices.push(current);

        // Find the most counterclockwise point from current
        let mut next = (current + 1) % npoints;

        for i in 0..npoints {
            if i == current {
                continue;
            }

            let p1 = [points[[current, 0]], points[[current, 1]]];
            let p2 = [points[[next, 0]], points[[next, 1]]];
            let p3 = [points[[i, 0]], points[[i, 1]]];

            let cross = cross_product_2d(p1, p2, p3);

            // If cross product is positive, i is more counterclockwise than next
            if cross > 0.0
                || (cross == 0.0 && distance_squared_2d(p1, p3) > distance_squared_2d(p1, p2))
            {
                next = i;
            }
        }

        current = next;
        if current == leftmost {
            break; // We've wrapped around to the start
        }
    }

    let vertex_indices = hull_vertices;

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

/// Find the leftmost point in the point set
///
/// In case of ties (multiple points with the same x-coordinate),
/// returns the first one encountered.
///
/// # Arguments
///
/// * `points` - Input points array
///
/// # Returns
///
/// * Index of the leftmost point
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::jarvis_march::find_leftmost_point;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 0.0]];
/// let leftmost = find_leftmost_point(&points.view());
/// assert!(leftmost == 1 || leftmost == 3); // Either [0.0, 1.0] or [0.0, 0.0]
/// ```
pub fn find_leftmost_point(points: &ArrayView2<'_, f64>) -> usize {
    let npoints = points.nrows();
    let mut leftmost = 0;

    for i in 1..npoints {
        if points[[i, 0]] < points[[leftmost, 0]] {
            leftmost = i;
        }
    }

    leftmost
}

/// Find the most counterclockwise point from a given point
///
/// Given a current point and a candidate next point, find the point in the set
/// that is most counterclockwise relative to the line from current to candidate.
///
/// # Arguments
///
/// * `points` - Input points array
/// * `current` - Index of the current point
/// * `candidate` - Index of the candidate next point
///
/// # Returns
///
/// * Index of the most counterclockwise point
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::jarvis_march::find_most_counterclockwise;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let current = 0; // [0.0, 0.0]
/// let candidate = 1; // [1.0, 0.0]
///
/// let most_ccw = find_most_counterclockwise(&points.view(), current, candidate);
/// assert_eq!(most_ccw, 2); // [0.0, 1.0] is most counterclockwise
/// ```
pub fn find_most_counterclockwise(
    points: &ArrayView2<'_, f64>,
    current: usize,
    candidate: usize,
) -> usize {
    let npoints = points.nrows();
    let mut best = candidate;

    for i in 0..npoints {
        if i == current {
            continue;
        }

        let p1 = [points[[current, 0]], points[[current, 1]]];
        let p2 = [points[[best, 0]], points[[best, 1]]];
        let p3 = [points[[i, 0]], points[[i, 1]]];

        let cross = cross_product_2d(p1, p2, p3);

        // If cross product is positive, i is more counterclockwise than best
        if cross > 0.0
            || (cross == 0.0 && distance_squared_2d(p1, p3) > distance_squared_2d(p1, p2))
        {
            best = i;
        }
    }

    best
}

/// Check if a point is more counterclockwise than another
///
/// # Arguments
///
/// * `reference` - Reference point [x, y]
/// * `current_best` - Current best point [x, y]
/// * `candidate` - Candidate point [x, y]
///
/// # Returns
///
/// * true if candidate is more counterclockwise than current_best
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::jarvis_march::is_more_counterclockwise;
///
/// let reference = [0.0, 0.0];
/// let current_best = [1.0, 0.0];
/// let candidate = [0.0, 1.0];
///
/// assert!(is_more_counterclockwise(reference, current_best, candidate));
/// ```
pub fn is_more_counterclockwise(
    reference: [f64; 2],
    current_best: [f64; 2],
    candidate: [f64; 2],
) -> bool {
    let cross = cross_product_2d(reference, current_best, candidate);

    if cross > 0.0 {
        true // Candidate is more counterclockwise
    } else if cross == 0.0 {
        // If collinear, prefer the farther point
        distance_squared_2d(reference, candidate) > distance_squared_2d(reference, current_best)
    } else {
        false
    }
}

/// Perform one step of the Jarvis march
///
/// Given a current hull point, find the next point in the hull by selecting
/// the most counterclockwise point.
///
/// # Arguments
///
/// * `points` - Input points array
/// * `current` - Index of the current hull point
///
/// # Returns
///
/// * Index of the next hull point
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::jarvis_march::jarvis_step;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let current = 0; // Start from [0.0, 0.0]
///
/// let next = jarvis_step(&points.view(), current);
/// // Should find either [1.0, 0.0] or [0.0, 1.0] depending on the order
/// assert!(next == 1 || next == 2);
/// ```
pub fn jarvis_step(points: &ArrayView2<'_, f64>, current: usize) -> usize {
    let npoints = points.nrows();

    // Find the first point that's not the current point
    let mut next = if current == 0 { 1 } else { 0 };

    // Find the most counterclockwise point
    for i in 0..npoints {
        if i == current {
            continue;
        }

        let p_current = [points[[current, 0]], points[[current, 1]]];
        let p_next = [points[[next, 0]], points[[next, 1]]];
        let p_candidate = [points[[i, 0]], points[[i, 1]]];

        if is_more_counterclockwise(p_current, p_next, p_candidate) {
            next = i;
        }
    }

    next
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_jarvis_march_basic() {
        let points = arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5], // Interior point
        ]);

        let hull = compute_jarvis_march(&points.view()).expect("Operation failed");

        // Check dimensions
        assert_eq!(hull.ndim(), 2);

        // The interior point should not be part of the convex hull
        assert_eq!(hull.vertex_indices().len(), 3);

        // Verify that the interior point is not in the hull
        assert!(!hull.vertex_indices().contains(&3));
    }

    #[test]
    fn test_jarvis_march_square() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);

        let hull = compute_jarvis_march(&points.view()).expect("Operation failed");

        assert_eq!(hull.ndim(), 2);
        assert_eq!(hull.vertex_indices().len(), 4); // All points should be vertices
    }

    #[test]
    fn test_find_leftmost_point() {
        let points = arr2(&[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 0.0]]);
        let leftmost = find_leftmost_point(&points.view());

        // Should be either index 1 or 3 (both have x = 0.0)
        assert!(leftmost == 1 || leftmost == 3);
    }

    #[test]
    fn test_find_most_counterclockwise() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let current = 0; // [0.0, 0.0]
        let candidate = 1; // [1.0, 0.0]

        let most_ccw = find_most_counterclockwise(&points.view(), current, candidate);
        assert_eq!(most_ccw, 2); // [0.0, 1.0] is most counterclockwise
    }

    #[test]
    fn test_is_more_counterclockwise() {
        let reference = [0.0, 0.0];
        let current_best = [1.0, 0.0];
        let candidate = [0.0, 1.0];

        assert!(is_more_counterclockwise(reference, current_best, candidate));
        assert!(!is_more_counterclockwise(
            reference,
            candidate,
            current_best
        ));
    }

    #[test]
    fn test_jarvis_step() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let current = 0; // Start from [0.0, 0.0]

        let next = jarvis_step(&points.view(), current);
        // Should find either [1.0, 0.0] or [0.0, 1.0] depending on the implementation
        assert!(next == 1 || next == 2);
    }

    #[test]
    fn test_error_cases() {
        // Test with too few points
        let too_few = arr2(&[[0.0, 0.0], [1.0, 0.0]]);
        let result = compute_jarvis_march(&too_few.view());
        assert!(result.is_err());

        // Test with 3D points (should fail)
        let points_3d = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let result = compute_jarvis_march(&points_3d.view());
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

        let hull = compute_jarvis_march(&points.view()).expect("Operation failed");

        // Should form a triangle with the non-collinear points
        assert_eq!(hull.vertex_indices().len(), 3);
        // The point above the line should be included
        assert!(hull.vertex_indices().contains(&3));
    }

    #[test]
    fn test_identical_points() {
        // Test with some identical points
        let points = arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0], // Duplicate point
        ]);

        let hull = compute_jarvis_march(&points.view()).expect("Operation failed");

        // Should still form a valid triangle
        assert_eq!(hull.vertex_indices().len(), 3);
    }
}

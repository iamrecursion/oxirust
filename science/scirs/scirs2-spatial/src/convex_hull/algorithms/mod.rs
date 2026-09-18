//! Convex hull algorithm implementations (Pure Rust)
//!
//! This module provides different algorithms for computing convex hulls,
//! each with their own strengths and use cases. All implementations are
//! pure Rust with no external C library dependencies.
//!
//! # Available Algorithms
//!
//! ## Quickhull (Pure Rust)
//! - **Module**: [`quickhull`]
//! - **Dimensions**: Any (1D, 2D, 3D, nD)
//! - **Time Complexity**: O(n log n) for 2D/3D, O(n^⌊d/2⌋) for d dimensions
//! - **Use Case**: General purpose, robust for all dimensions
//! - **Features**: Pure Rust, handles degenerate cases, provides facet equations
//!
//! ## Graham Scan (Pure Rust)
//! - **Module**: [`graham_scan`]
//! - **Dimensions**: 2D only
//! - **Time Complexity**: O(n log n)
//! - **Use Case**: Educational, guaranteed output order
//! - **Features**: Simple to understand, produces vertices in counterclockwise order
//!
//! ## Jarvis March (Gift Wrapping) (Pure Rust)
//! - **Module**: [`jarvis_march`]
//! - **Dimensions**: 2D only
//! - **Time Complexity**: O(nh) where h is the number of hull vertices
//! - **Use Case**: When hull has few vertices (h << n)
//! - **Features**: Output-sensitive, good for small hulls
//!
//! ## Special Cases
//! - **Module**: [`special_cases`]
//! - **Purpose**: Handle degenerate cases (collinear points, identical points, etc.)
//! - **Features**: Robust handling of edge cases that might break standard algorithms
//!
//! # Examples
//!
//! ## Using Quickhull (Recommended)
//! ```rust
//! use scirs2_spatial::convex_hull::algorithms::quickhull::compute_quickhull;
//! use scirs2_core::ndarray::array;
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
//! let hull = compute_quickhull(&points.view()).expect("Operation failed");
//! println!("Hull has {} vertices", hull.vertex_indices().len());
//! ```
//!
//! ## Using Graham Scan for 2D
//! ```rust
//! use scirs2_spatial::convex_hull::algorithms::graham_scan::compute_graham_scan;
//! use scirs2_core::ndarray::array;
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
//! let hull = compute_graham_scan(&points.view()).expect("Operation failed");
//! println!("Hull vertices: {:?}", hull.vertex_indices());
//! ```
//!
//! ## Using Jarvis March for Small Hulls
//! ```rust
//! use scirs2_spatial::convex_hull::algorithms::jarvis_march::compute_jarvis_march;
//! use scirs2_core::ndarray::array;
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
//! let hull = compute_jarvis_march(&points.view()).expect("Operation failed");
//! println!("Hull computed with {} vertices", hull.vertex_indices().len());
//! ```
//!
//! ## Handling Special Cases
//! ```rust
//! use scirs2_spatial::convex_hull::algorithms::special_cases::handle_degenerate_case;
//! use scirs2_core::ndarray::array;
//!
//! // Collinear points
//! let collinear = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
//! if let Some(result) = handle_degenerate_case(&collinear.view()) {
//!     let hull = result.expect("Operation failed");
//!     println!("Handled degenerate case with {} vertices", hull.vertex_indices().len());
//! }
//! ```
//!
//! # Algorithm Selection Guidelines
//!
//! - **For general use**: Use Quickhull - it's robust and handles all dimensions
//! - **For 2D educational purposes**: Use Graham Scan for its simplicity
//! - **For 2D with small expected hulls**: Use Jarvis March for output-sensitive performance
//! - **For edge cases**: All algorithms automatically fall back to special case handlers
//!
//! # Performance Characteristics
//!
//! | Algorithm | 2D Time | 3D Time | nD Time | Space | Robustness |
//! |-----------|---------|---------|---------|-------|------------|
//! | Quickhull | O(n log n) | O(n log n) | O(n^⌊d/2⌋) | O(n) | High |
//! | Graham Scan | O(n log n) | N/A | N/A | O(n) | Medium |
//! | Jarvis March | O(nh) | N/A | N/A | O(n) | Medium |
//!
//! Where:
//! - n = number of input points
//! - h = number of hull vertices
//! - d = dimension

pub mod graham_scan;
pub mod jarvis_march;
pub mod quickhull;
pub mod special_cases;

// Re-export the main computation functions for convenience
pub use graham_scan::compute_graham_scan;
pub use jarvis_march::compute_jarvis_march;
pub use quickhull::compute_quickhull;
pub use special_cases::{handle_degenerate_case, has_all_identical_points, is_all_collinear};

/// Algorithm performance characteristics for selection guidance
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlgorithmComplexity {
    /// O(n log n) complexity - good for general use
    NLogN,
    /// O(nh) complexity - output sensitive, good for small hulls
    OutputSensitive,
    /// O(n^⌊d/2⌋) complexity - exponential in dimension
    ExponentialDimension,
}

/// Get performance characteristics for each algorithm
///
/// # Arguments
///
/// * `algorithm` - The algorithm to query
/// * `dimension` - The dimension of the problem
///
/// # Returns
///
/// * Tuple of (time_complexity, space_complexity, max_dimension)
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::{get_algorithm_complexity, AlgorithmComplexity};
/// use scirs2_spatial::convex_hull::ConvexHullAlgorithm;
///
/// let (time, space, max_dim) = get_algorithm_complexity(ConvexHullAlgorithm::GrahamScan, 2);
/// assert_eq!(time, AlgorithmComplexity::NLogN);
/// assert_eq!(max_dim, Some(2));
/// ```
pub fn get_algorithm_complexity(
    algorithm: crate::convex_hull::core::ConvexHullAlgorithm,
    dimension: usize,
) -> (AlgorithmComplexity, AlgorithmComplexity, Option<usize>) {
    use crate::convex_hull::core::ConvexHullAlgorithm;

    match algorithm {
        ConvexHullAlgorithm::Quickhull => {
            let time_complexity = if dimension <= 3 {
                AlgorithmComplexity::NLogN
            } else {
                AlgorithmComplexity::ExponentialDimension
            };
            (time_complexity, AlgorithmComplexity::NLogN, None) // No dimension limit
        }
        ConvexHullAlgorithm::GrahamScan => (
            AlgorithmComplexity::NLogN,
            AlgorithmComplexity::NLogN,
            Some(2),
        ),
        ConvexHullAlgorithm::JarvisMarch => (
            AlgorithmComplexity::OutputSensitive,
            AlgorithmComplexity::NLogN,
            Some(2),
        ),
    }
}

/// Recommend the best algorithm for given constraints
///
/// # Arguments
///
/// * `num_points` - Number of input points
/// * `dimension` - Dimension of the points
/// * `expected_hull_size` - Expected number of hull vertices (None if unknown)
///
/// # Returns
///
/// * Recommended algorithm
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::recommend_algorithm;
/// use scirs2_spatial::convex_hull::ConvexHullAlgorithm;
///
/// // Large 2D dataset with small expected hull
/// let algo = recommend_algorithm(10000, 2, Some(8));
/// assert_eq!(algo, ConvexHullAlgorithm::JarvisMarch);
///
/// // 3D dataset
/// let algo = recommend_algorithm(1000, 3, None);
/// assert_eq!(algo, ConvexHullAlgorithm::Quickhull);
/// ```
pub fn recommend_algorithm(
    num_points: usize,
    dimension: usize,
    expected_hull_size: Option<usize>,
) -> crate::convex_hull::core::ConvexHullAlgorithm {
    use crate::convex_hull::core::ConvexHullAlgorithm;

    // For dimensions > 2, only Quickhull is available
    if dimension > 2 {
        return ConvexHullAlgorithm::Quickhull;
    }

    // For 2D, we have choices
    if dimension == 2 {
        if let Some(hull_size) = expected_hull_size {
            // If hull is expected to be small relative to input size, use Jarvis March
            if hull_size * ((num_points as f64).log2() as usize) < num_points {
                return ConvexHullAlgorithm::JarvisMarch;
            }
        }

        // For educational purposes or when you need guaranteed vertex order
        // you might prefer Graham Scan, but for general robustness, use Quickhull
        if num_points < 100 {
            ConvexHullAlgorithm::GrahamScan // Simple and fast for small datasets
        } else {
            ConvexHullAlgorithm::Quickhull // Most robust for larger datasets
        }
    } else {
        // For 1D or other edge cases
        ConvexHullAlgorithm::Quickhull
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex_hull::core::ConvexHullAlgorithm;

    #[test]
    fn test_get_algorithm_complexity() {
        let (time, space, max_dim) = get_algorithm_complexity(ConvexHullAlgorithm::GrahamScan, 2);
        assert_eq!(time, AlgorithmComplexity::NLogN);
        assert_eq!(space, AlgorithmComplexity::NLogN);
        assert_eq!(max_dim, Some(2));

        let (time, space, max_dim) = get_algorithm_complexity(ConvexHullAlgorithm::JarvisMarch, 2);
        assert_eq!(time, AlgorithmComplexity::OutputSensitive);
        assert_eq!(max_dim, Some(2));

        let (time, _space, max_dim) = get_algorithm_complexity(ConvexHullAlgorithm::Quickhull, 5);
        assert_eq!(time, AlgorithmComplexity::ExponentialDimension);
        assert_eq!(max_dim, None);
    }

    #[test]
    fn test_recommend_algorithm() {
        // Large 2D dataset with small expected hull
        let algo = recommend_algorithm(10000, 2, Some(8));
        assert_eq!(algo, ConvexHullAlgorithm::JarvisMarch);

        // 3D dataset
        let algo = recommend_algorithm(1000, 3, None);
        assert_eq!(algo, ConvexHullAlgorithm::Quickhull);

        // Small 2D dataset
        let algo = recommend_algorithm(50, 2, None);
        assert_eq!(algo, ConvexHullAlgorithm::GrahamScan);

        // Large 2D dataset without hull size hint
        let algo = recommend_algorithm(5000, 2, None);
        assert_eq!(algo, ConvexHullAlgorithm::Quickhull);
    }
}

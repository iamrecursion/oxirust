//! Convex hull algorithms and utilities
//!
//! This module provides comprehensive convex hull computation capabilities,
//! supporting multiple algorithms and dimensions. It includes geometric
//! property calculations and robust point containment testing.
//!
//! # Overview
//!
//! A convex hull is the smallest convex set that contains all given points.
//! This module provides:
//!
//! - **Multiple algorithms**: Quickhull (nD), Graham Scan (2D), Jarvis March (2D)
//! - **Geometric properties**: Volume, surface area, compactness measures
//! - **Point queries**: Containment testing, distance calculations
//! - **Robust handling**: Special cases, degenerate inputs, numerical precision
//! - **Pure Rust**: All implementations are pure Rust with no C library dependencies
//!
//! # Quick Start
//!
//! ## Basic Usage
//! ```rust
//! use scirs2_spatial::convex_hull::{ConvexHull, convex_hull};
//! use scirs2_core::ndarray::array;
//!
//! // Create points for the convex hull
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
//!
//! // Compute convex hull (automatic algorithm selection)
//! let hull = ConvexHull::new(&points.view()).expect("Operation failed");
//!
//! // Or use the convenience function
//! let hull_vertices = convex_hull(&points.view()).expect("Operation failed");
//!
//! // Access hull properties
//! println!("Hull vertices: {:?}", hull.vertices());
//! println!("Hull volume: {}", hull.volume().expect("Operation failed"));
//! println!("Hull surface area: {}", hull.area().expect("Operation failed"));
//!
//! // Test point containment
//! let is_inside = hull.contains(&[0.25, 0.25]).expect("Operation failed");
//! println!("Point inside: {}", is_inside);
//! ```
//!
//! ## Algorithm Selection
//! ```rust
//! use scirs2_spatial::convex_hull::{ConvexHull, ConvexHullAlgorithm, convex_hull_with_algorithm};
//! use scirs2_core::ndarray::array;
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
//!
//! // Use specific algorithm
//! let hull = ConvexHull::new_with_algorithm(&points.view(), ConvexHullAlgorithm::GrahamScan).expect("Operation failed");
//!
//! // Or use convenience function
//! let hull_vertices = convex_hull_with_algorithm(&points.view(), ConvexHullAlgorithm::JarvisMarch).expect("Operation failed");
//! ```
//!
//! ## Comprehensive Analysis
//! ```rust
//! use scirs2_spatial::convex_hull::{ConvexHull, analyze_hull};
//! use scirs2_core::ndarray::array;
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
//! let hull = ConvexHull::new(&points.view()).expect("Operation failed");
//!
//! // Get comprehensive analysis
//! let analysis = analyze_hull(&hull).expect("Operation failed");
//! println!("Hull Analysis: {:#?}", analysis);
//! ```
//!
//! # Module Organization
//!
//! - [`core`] - Core ConvexHull struct and basic functionality
//! - [`algorithms`] - Algorithm implementations (Quickhull, Graham Scan, Jarvis March)
//! - [`geometry`] - Geometric utility functions for different dimensions
//! - [`properties`] - Volume, surface area, and containment calculations
//!
//! # Algorithm Guide
//!
//! ## Quickhull (Recommended for most uses)
//! - **Dimensions**: 1D, 2D, 3D, nD
//! - **Time Complexity**: O(n log n) for 2D/3D, O(n^⌊d/2⌋) for higher dimensions
//! - **Features**: Pure Rust, robust, handles degenerate cases, provides facet equations
//! - **Use when**: General-purpose convex hull computation
//!
//! ## Graham Scan (2D only)
//! - **Dimensions**: 2D only
//! - **Time Complexity**: O(n log n)
//! - **Features**: Simple, produces vertices in counterclockwise order
//! - **Use when**: Educational purposes, guaranteed vertex ordering needed
//!
//! ## Jarvis March / Gift Wrapping (2D only)
//! - **Dimensions**: 2D only
//! - **Time Complexity**: O(nh) where h is number of hull vertices
//! - **Features**: Output-sensitive, good for sparse hulls
//! - **Use when**: Hull has few vertices compared to input size
//!
//! # Performance Tips
//!
//! 1. **Use Quickhull for general cases** - it's the most robust and efficient
//! 2. **For 2D with small hulls** - consider Jarvis March for output-sensitive performance
//! 3. **For large datasets** - consider preprocessing to remove interior points
//! 4. **For high dimensions** - expect exponential complexity, consider approximations
//!
//! # Error Handling
//!
//! The module handles various error conditions gracefully:
//! - Insufficient points for given dimension
//! - Algorithm/dimension mismatches
//! - Degenerate point configurations
//! - Numerical precision issues

pub mod algorithms;
pub mod core;
pub mod geometry;
pub mod properties;

// Re-export the main types and functions for backward compatibility
pub use core::{ConvexHull, ConvexHullAlgorithm};

// Re-export algorithm functions
pub use algorithms::{
    compute_graham_scan, compute_jarvis_march, compute_quickhull, get_algorithm_complexity,
    recommend_algorithm,
};

// Re-export geometry utilities (most commonly used ones)
pub use geometry::{
    compute_polygon_area, compute_polygon_perimeter, compute_polyhedron_volume, cross_product_2d,
    cross_product_3d, tetrahedron_volume, triangle_area_3d,
};

// Re-export properties functions
pub use properties::{
    analyze_hull, check_point_containment, compute_surface_area, compute_volume,
    get_hull_statistics,
};

// Main convenience functions for backward compatibility with original API

/// Compute the convex hull of a set of points using the default algorithm
///
/// This is the main convenience function that provides the same interface
/// as the original monolithic implementation.
///
/// # Arguments
///
/// * `points` - Input points (shape: npoints x n_dim)
///
/// # Returns
///
/// * A result containing either the convex hull vertices (shape: n_vertices x n_dim)
///   or an error
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::convex_hull;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
/// let hull_vertices = convex_hull(&points.view()).expect("Operation failed");
///
/// // The hull vertices should be the corners, not the interior point
/// assert!(hull_vertices.nrows() >= 3);
/// ```
#[allow(dead_code)]
pub fn convex_hull(
    points: &scirs2_core::ndarray::ArrayView2<'_, f64>,
) -> crate::error::SpatialResult<scirs2_core::ndarray::Array2<f64>> {
    let hull = ConvexHull::new(points)?;
    Ok(hull.vertices_array())
}

/// Compute the convex hull of a set of points using a specific algorithm
///
/// This function provides algorithm selection while maintaining the same
/// interface as the original implementation.
///
/// # Arguments
///
/// * `points` - Input points (shape: npoints x n_dim)
/// * `algorithm` - Algorithm to use for convex hull computation
///
/// # Returns
///
/// * A result containing either the convex hull vertices (shape: n_vertices x n_dim)
///   or an error
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::{convex_hull_with_algorithm, ConvexHullAlgorithm};
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
/// let hull_vertices = convex_hull_with_algorithm(&points.view(), ConvexHullAlgorithm::GrahamScan).expect("Operation failed");
/// assert!(hull_vertices.nrows() >= 3);
/// ```
#[allow(dead_code)]
pub fn convex_hull_with_algorithm(
    points: &scirs2_core::ndarray::ArrayView2<'_, f64>,
    algorithm: ConvexHullAlgorithm,
) -> crate::error::SpatialResult<scirs2_core::ndarray::Array2<f64>> {
    let hull = ConvexHull::new_with_algorithm(points, algorithm)?;
    Ok(hull.vertices_array())
}

/// Extended analysis and statistics for debugging and research
///
/// Advanced analysis functions for convex hulls
///
/// This module provides extended functionality for detailed hull analysis,
/// research applications, and debugging purposes.
pub mod advanced {

    use super::*;

    /// Detailed performance metrics for hull computation
    #[derive(Debug, Clone)]
    pub struct HullComputationMetrics {
        /// Time taken for hull computation (if measured)
        pub computation_time: Option<std::time::Duration>,
        /// Memory usage during computation (if measured)
        pub memory_usage: Option<usize>,
        /// Number of iterations/steps in the algorithm
        pub algorithm_steps: Option<usize>,
        /// Whether special case handling was used
        pub used_special_case: bool,
        /// Whether the result is considered reliable
        pub result_reliable: bool,
    }

    /// Compare different algorithms on the same point set
    ///
    /// This function runs multiple algorithms on the same point set and
    /// compares their results, useful for algorithm validation and research.
    ///
    /// # Arguments
    ///
    /// * `points` - Input points
    ///
    /// # Returns
    ///
    /// * Results from different algorithms with comparison metrics
    pub fn compare_algorithms(
        points: &scirs2_core::ndarray::ArrayView2<'_, f64>,
    ) -> crate::error::SpatialResult<
        Vec<(ConvexHullAlgorithm, crate::error::SpatialResult<ConvexHull>)>,
    > {
        let mut results = Vec::new();

        // Always try Quickhull
        results.push((
            ConvexHullAlgorithm::Quickhull,
            ConvexHull::new_with_algorithm(points, ConvexHullAlgorithm::Quickhull),
        ));

        // For 2D, also try other algorithms
        if points.ncols() == 2 && points.nrows() >= 3 {
            results.push((
                ConvexHullAlgorithm::GrahamScan,
                ConvexHull::new_with_algorithm(points, ConvexHullAlgorithm::GrahamScan),
            ));

            results.push((
                ConvexHullAlgorithm::JarvisMarch,
                ConvexHull::new_with_algorithm(points, ConvexHullAlgorithm::JarvisMarch),
            ));
        }

        Ok(results)
    }

    /// Validate hull correctness using multiple methods
    ///
    /// This function performs comprehensive validation of a hull computation,
    /// useful for testing and validation purposes.
    ///
    /// # Arguments
    ///
    /// * `hull` - The hull to validate
    /// * `original_points` - The original input points
    ///
    /// # Returns
    ///
    /// * Validation results and any issues found
    pub fn validate_hull(
        hull: &ConvexHull,
        original_points: &scirs2_core::ndarray::ArrayView2<'_, f64>,
    ) -> crate::error::SpatialResult<Vec<String>> {
        let mut issues = Vec::new();

        // Check that all original points are either hull vertices or inside the hull
        for i in 0..original_points.nrows() {
            let point = original_points.row(i);
            let point_slice = point.as_slice().expect("Operation failed");

            // Check if this point is a hull vertex
            let is_vertex = hull.vertex_indices().iter().any(|&idx| {
                let vertex = hull.points.row(idx);
                vertex
                    .as_slice()
                    .expect("Operation failed")
                    .iter()
                    .zip(point_slice.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            });

            // If not a vertex, it should be inside or on the hull
            if !is_vertex {
                match hull.contains(point_slice) {
                    Ok(inside) => {
                        if !inside {
                            issues.push(format!("Point {} is outside its own hull", i));
                        }
                    }
                    Err(e) => {
                        issues.push(format!("Failed to test containment for point {}: {}", i, e));
                    }
                }
            }
        }

        // Check hull properties for consistency
        if let Ok(volume) = hull.volume() {
            if volume < 0.0 {
                issues.push("Hull volume is negative".to_string());
            }
        }

        if let Ok(area) = hull.area() {
            if area < 0.0 {
                issues.push("Hull surface area is negative".to_string());
            }
        }

        // Check vertex indices are valid
        for &idx in hull.vertex_indices() {
            if idx >= hull.points.nrows() {
                issues.push(format!("Invalid vertex index: {}", idx));
            }
        }

        Ok(issues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_convex_hull_function() {
        let points = arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5], // Interior point
        ]);

        let hull_vertices = convex_hull(&points.view()).expect("Operation failed");

        // The hull should have vertices in 2D
        assert!(hull_vertices.nrows() >= 3); // At least 3 for triangular hull
        assert_eq!(hull_vertices.ncols(), 2);
    }

    #[test]
    fn test_convex_hull_with_algorithm_function() {
        let points = arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5], // Interior point
        ]);

        // Test with Graham scan
        let hull_vertices =
            convex_hull_with_algorithm(&points.view(), ConvexHullAlgorithm::GrahamScan)
                .expect("Operation failed");
        assert_eq!(hull_vertices.nrows(), 3); // Should exclude interior point
        assert_eq!(hull_vertices.ncols(), 2);

        // Test with Jarvis march
        let hull_vertices =
            convex_hull_with_algorithm(&points.view(), ConvexHullAlgorithm::JarvisMarch)
                .expect("Operation failed");
        assert_eq!(hull_vertices.nrows(), 3); // Should exclude interior point
        assert_eq!(hull_vertices.ncols(), 2);
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that the new modular implementation produces the same results
        // as the original monolithic implementation would have
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);

        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let vertices = hull.vertices();
        let vertex_indices = hull.vertex_indices();
        let simplices = hull.simplices();

        // Check basic properties
        assert_eq!(hull.ndim(), 2);
        assert_eq!(vertices.len(), vertex_indices.len());
        assert!(!simplices.is_empty());

        // Check volume (area) and surface area (perimeter)
        let volume = hull.volume().expect("Operation failed");
        let area = hull.area().expect("Operation failed");
        assert!((volume - 1.0).abs() < 1e-10); // Unit square area
        assert!((area - 4.0).abs() < 1e-10); // Unit square perimeter

        // Check containment
        assert!(hull.contains([0.5, 0.5]).expect("Operation failed")); // Center should be inside
        assert!(!hull.contains([2.0, 2.0]).expect("Operation failed")); // Far point should be outside
    }

    #[test]
    fn test_error_cases() {
        // Too few points for a 2D hull
        let too_few = arr2(&[[0.0, 0.0], [1.0, 0.0]]);
        let result = ConvexHull::new(&too_few.view());
        assert!(result.is_err());

        // Valid hull but invalid point dimensionality for containment
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let result = hull.contains([0.5, 0.5, 0.5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_3d_hull() {
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5], // Interior point
        ]);

        let hull = ConvexHull::new(&points.view()).expect("Operation failed");

        assert_eq!(hull.ndim(), 3);
        assert!(hull.vertex_indices().len() >= 4); // At least tetrahedron vertices

        // Interior point should be inside
        assert!(hull.contains([0.25, 0.25, 0.25]).expect("Operation failed"));
        // Far point should be outside
        assert!(!hull.contains([2.0, 2.0, 2.0]).expect("Operation failed"));

        // Volume and surface area should be positive
        let volume = hull.volume().expect("Operation failed");
        let surface_area = hull.area().expect("Operation failed");
        assert!(volume > 0.0);
        assert!(surface_area > 0.0);
    }

    #[test]
    fn test_analysis_functions() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");

        let analysis = analyze_hull(&hull).expect("Operation failed");
        assert_eq!(analysis.ndim, 2);
        assert_eq!(analysis.num_vertices, 4);
        assert!((analysis.volume - 1.0).abs() < 1e-10);
        assert!((analysis.surface_area - 4.0).abs() < 1e-10);

        let stats = get_hull_statistics(&hull).expect("Operation failed");
        assert_eq!(stats.num_input_points, 4);
        assert_eq!(stats.num_hull_vertices, 4);
        assert_eq!(stats.hull_vertex_fraction, 1.0); // All points are vertices
    }
}

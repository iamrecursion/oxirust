//! Convex hull properties and analysis
//!
//! This module provides functions to compute various geometric properties
//! of convex hulls, including volume, surface area, and point containment testing.
//!
//! # Module Organization
//!
//! - [`volume`] - Volume/area/length calculations for different dimensions
//! - [`surface_area`] - Surface area/perimeter calculations and compactness measures
//! - [`containment`] - Point-in-hull testing and distance calculations
//!
//! # Examples
//!
//! ## Volume Calculations
//! ```rust
//! use scirs2_spatial::convex_hull::ConvexHull;
//! use scirs2_spatial::convex_hull::properties::volume::compute_volume;
//! use scirs2_core::ndarray::array;
//!
//! // 2D square with area 1
//! let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
//! let hull = ConvexHull::new(&points.view()).expect("Operation failed");
//! let area = compute_volume(&hull).expect("Operation failed");
//! assert!((area - 1.0).abs() < 1e-10);
//! ```
//!
//! ## Surface Area Calculations
//! ```rust
//! use scirs2_spatial::convex_hull::ConvexHull;
//! use scirs2_spatial::convex_hull::properties::surface_area::{compute_surface_area, compute_compactness};
//! use scirs2_core::ndarray::array;
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
//! let hull = ConvexHull::new(&points.view()).expect("Operation failed");
//!
//! let perimeter = compute_surface_area(&hull).expect("Operation failed");
//! assert!((perimeter - 4.0).abs() < 1e-10);
//!
//! let compactness = compute_compactness(&hull).expect("Operation failed");
//! assert!(compactness > 0.7); // Square is fairly compact
//! ```
//!
//! ## Point Containment Testing
//! ```rust
//! use scirs2_spatial::convex_hull::ConvexHull;
//! use scirs2_spatial::convex_hull::properties::containment::{check_point_containment, distance_to_hull};
//! use scirs2_core::ndarray::array;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
//! let hull = ConvexHull::new(&points.view())?;
//!
//! // Test containment
//! assert!(check_point_containment(&hull, &[0.1, 0.1])?);
//! assert!(!check_point_containment(&hull, &[2.0, 2.0])?);
//!
//! // Compute distance
//! let dist = distance_to_hull(&hull, &[0.5, 0.5])?;
//! assert!(dist < 0.0); // Inside the hull
//! # Ok(())
//! # }
//! ```
//!
//! ## Combined Analysis
//! ```rust
//! use scirs2_spatial::convex_hull::ConvexHull;
//! use scirs2_spatial::convex_hull::properties::*;
//! use scirs2_core::ndarray::array;
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
//! let hull = ConvexHull::new(&points.view()).expect("Operation failed");
//!
//! let vol = volume::compute_volume(&hull).expect("Operation failed");
//! let area = surface_area::compute_surface_area(&hull).expect("Operation failed");
//! let ratio = surface_area::compute_surface_to_volume_ratio(&hull).expect("Operation failed");
//! let compactness = surface_area::compute_compactness(&hull).expect("Operation failed");
//!
//! println!("Volume: {}, Surface Area: {}, Ratio: {}, Compactness: {}",
//!          vol, area, ratio, compactness);
//! ```

pub mod containment;
pub mod surface_area;
pub mod volume;

// Re-export commonly used functions for convenience
pub use volume::{
    compute_volume, compute_volume_bounds, compute_volume_monte_carlo,
    is_volume_computation_reliable,
};

pub use surface_area::{
    compute_compactness, compute_surface_area, compute_surface_area_bounds,
    compute_surface_to_volume_ratio, is_surface_area_computation_reliable,
};

pub use containment::{check_multiple_containment, check_point_containment, distance_to_hull};

/// Comprehensive hull analysis results
///
/// This structure contains all major geometric properties of a convex hull.
#[derive(Debug, Clone)]
pub struct HullAnalysis {
    /// Number of dimensions
    pub ndim: usize,
    /// Number of vertices
    pub num_vertices: usize,
    /// Number of facets/simplices
    pub num_facets: usize,
    /// Volume/area/length of the hull
    pub volume: f64,
    /// Surface area/perimeter of the hull
    pub surface_area: f64,
    /// Surface area to volume ratio
    pub surface_to_volume_ratio: f64,
    /// Compactness measure (0-1, higher is more compact)
    pub compactness: f64,
    /// Whether volume computation is considered reliable
    pub volume_reliable: bool,
    /// Whether surface area computation is considered reliable
    pub surface_area_reliable: bool,
    /// Bounding box dimensions
    pub bounding_box_size: Vec<f64>,
    /// Centroid of the hull vertices
    pub centroid: Vec<f64>,
}

/// Perform comprehensive analysis of a convex hull
///
/// This function computes all major geometric properties of the hull
/// and returns them in a single structure.
///
/// # Arguments
///
/// * `hull` - The convex hull to analyze
///
/// # Returns
///
/// * Result containing comprehensive hull analysis
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::analyze_hull;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).expect("Operation failed");
/// let analysis = analyze_hull(&hull).expect("Operation failed");
///
/// println!("Hull Analysis:");
/// println!("  Dimensions: {}", analysis.ndim);
/// println!("  Vertices: {}", analysis.num_vertices);
/// println!("  Volume: {}", analysis.volume);
/// println!("  Surface Area: {}", analysis.surface_area);
/// println!("  Compactness: {:.3}", analysis.compactness);
/// ```
pub fn analyze_hull(
    hull: &crate::convex_hull::core::ConvexHull,
) -> crate::error::SpatialResult<HullAnalysis> {
    let ndim = hull.ndim();
    let num_vertices = hull.vertex_indices().len();
    let num_facets = hull.simplices().len();

    // Compute volume and surface area
    let volume = volume::compute_volume(hull)?;
    let surface_area = surface_area::compute_surface_area(hull)?;

    // Compute derived properties
    let surface_to_volume_ratio = surface_area::compute_surface_to_volume_ratio(hull)?;
    let compactness = surface_area::compute_compactness(hull)?;

    // Check reliability
    let volume_reliable = volume::is_volume_computation_reliable(hull);
    let surface_area_reliable = surface_area::is_surface_area_computation_reliable(hull);

    // Compute bounding box
    use crate::convex_hull::geometry::compute_bounding_box;
    let (min_coords, max_coords) = compute_bounding_box(&hull.points.view(), &hull.vertex_indices);
    let bounding_box_size: Vec<f64> = min_coords
        .iter()
        .zip(max_coords.iter())
        .map(|(min, max)| max - min)
        .collect();

    // Compute centroid
    use crate::convex_hull::geometry::compute_centroid;
    let centroid = compute_centroid(&hull.points.view(), &hull.vertex_indices);

    Ok(HullAnalysis {
        ndim,
        num_vertices,
        num_facets,
        volume,
        surface_area,
        surface_to_volume_ratio,
        compactness,
        volume_reliable,
        surface_area_reliable,
        bounding_box_size,
        centroid,
    })
}

/// Get geometric statistics for a convex hull
///
/// Returns basic statistics about the hull's vertices and structure.
///
/// # Arguments
///
/// * `hull` - The convex hull
///
/// # Returns
///
/// * Result containing geometric statistics
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::get_hull_statistics;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).expect("Operation failed");
/// let stats = get_hull_statistics(&hull).expect("Operation failed");
///
/// println!("Hull Statistics: {:#?}", stats);
/// ```
#[derive(Debug, Clone)]
pub struct HullStatistics {
    /// Number of input points
    pub num_input_points: usize,
    /// Number of hull vertices
    pub num_hull_vertices: usize,
    /// Fraction of input points that are hull vertices
    pub hull_vertex_fraction: f64,
    /// Number of facets/edges
    pub num_facets: usize,
    /// Average edge/facet size
    pub avg_facet_size: f64,
    /// Minimum distance between vertices
    pub min_vertex_distance: f64,
    /// Maximum distance between vertices  
    pub max_vertex_distance: f64,
    /// Average distance from centroid to vertices
    pub avg_centroid_distance: f64,
}

/// Get geometric statistics for a convex hull
pub fn get_hull_statistics(
    hull: &crate::convex_hull::core::ConvexHull,
) -> crate::error::SpatialResult<HullStatistics> {
    let num_input_points = hull.points.nrows();
    let num_hull_vertices = hull.vertex_indices.len();
    let hull_vertex_fraction = num_hull_vertices as f64 / num_input_points as f64;
    let num_facets = hull.simplices.len();

    // Average facet size
    let total_facet_size: usize = hull.simplices.iter().map(|s| s.len()).sum();
    let avg_facet_size = if num_facets > 0 {
        total_facet_size as f64 / num_facets as f64
    } else {
        0.0
    };

    // Distance calculations
    let mut min_distance = f64::INFINITY;
    let mut max_distance: f64 = 0.0;
    let ndim = hull.ndim();

    // Compute pairwise distances between vertices
    for i in 0..num_hull_vertices {
        for j in (i + 1)..num_hull_vertices {
            let idx1 = hull.vertex_indices[i];
            let idx2 = hull.vertex_indices[j];

            let mut dist_sq: f64 = 0.0;
            for d in 0..ndim {
                let diff = hull.points[[idx2, d]] - hull.points[[idx1, d]];
                dist_sq += diff * diff;
            }
            let distance = dist_sq.sqrt();

            min_distance = min_distance.min(distance);
            max_distance = max_distance.max(distance);
        }
    }

    // Average distance from centroid to vertices
    use crate::convex_hull::geometry::compute_centroid;
    let centroid = compute_centroid(&hull.points.view(), &hull.vertex_indices);

    let mut total_centroid_distance = 0.0;
    for &vertex_idx in &hull.vertex_indices {
        let mut dist_sq: f64 = 0.0;
        for d in 0..ndim {
            let diff = hull.points[[vertex_idx, d]] - centroid[d];
            dist_sq += diff * diff;
        }
        total_centroid_distance += dist_sq.sqrt();
    }
    let avg_centroid_distance = if num_hull_vertices > 0 {
        total_centroid_distance / num_hull_vertices as f64
    } else {
        0.0
    };

    if min_distance == f64::INFINITY {
        min_distance = 0.0;
    }

    Ok(HullStatistics {
        num_input_points,
        num_hull_vertices,
        hull_vertex_fraction,
        num_facets,
        avg_facet_size,
        min_vertex_distance: min_distance,
        max_vertex_distance: max_distance,
        avg_centroid_distance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex_hull::ConvexHull;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_analyze_hull() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let analysis = analyze_hull(&hull).expect("Operation failed");

        assert_eq!(analysis.ndim, 2);
        assert_eq!(analysis.num_vertices, 4);
        assert!((analysis.volume - 1.0).abs() < 1e-10); // Area = 1
        assert!((analysis.surface_area - 4.0).abs() < 1e-10); // Perimeter = 4
        assert!((analysis.surface_to_volume_ratio - 4.0).abs() < 1e-10);
        assert!(analysis.compactness > 0.7);
        assert!(analysis.volume_reliable);
        assert!(analysis.surface_area_reliable);
        assert_eq!(analysis.bounding_box_size, vec![1.0, 1.0]);
        assert_eq!(analysis.centroid, vec![0.5, 0.5]);
    }

    #[test]
    fn test_get_hull_statistics() {
        let points = arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5], // Interior point
        ]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let stats = get_hull_statistics(&hull).expect("Operation failed");

        assert_eq!(stats.num_input_points, 5);
        assert!(stats.num_hull_vertices <= 4); // Interior point should not be a vertex
        assert!(stats.hull_vertex_fraction <= 0.8);
        assert!(stats.num_facets >= 4); // At least 4 edges for square
        assert!(stats.min_vertex_distance > 0.0);
        assert!(stats.max_vertex_distance > stats.min_vertex_distance);
        assert!(stats.avg_centroid_distance > 0.0);
    }

    #[test]
    fn test_triangle_analysis() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let analysis = analyze_hull(&hull).expect("Operation failed");

        assert_eq!(analysis.ndim, 2);
        assert_eq!(analysis.num_vertices, 3);
        assert!((analysis.volume - 0.5).abs() < 1e-10); // Area = 0.5
        assert!(analysis.compactness > 0.0 && analysis.compactness <= 1.0);
    }

    #[test]
    fn test_3d_analysis() {
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let analysis = analyze_hull(&hull).expect("Operation failed");

        assert_eq!(analysis.ndim, 3);
        assert_eq!(analysis.num_vertices, 4);
        assert!(analysis.volume > 0.0); // Tetrahedron has positive volume
        assert!(analysis.surface_area > 0.0);
        assert!(analysis.compactness > 0.0 && analysis.compactness <= 1.0);
    }
}

//! Geometric utility functions for convex hull computations
//!
//! This module provides geometric calculations needed for convex hull algorithms,
//! organized by dimensionality for optimal performance and clarity.
//!
//! # Module Organization
//!
//! - [`calculations_2d`] - 2D geometric calculations (cross products, areas, perimeters)
//! - [`calculations_3d`] - 3D geometric calculations (volumes, surface areas, vector operations)
//! - [`high_dimensional`] - High-dimensional calculations and approximations
//!
//! # Examples
//!
//! ## 2D Calculations
//! ```rust
//! use scirs2_spatial::convex_hull::geometry::calculations_2d::{cross_product_2d, compute_polygon_area};
//! use scirs2_core::ndarray::array;
//!
//! // Check orientation of three points
//! let p1 = [0.0, 0.0];
//! let p2 = [1.0, 0.0];
//! let p3 = [0.0, 1.0];
//! let cross = cross_product_2d(p1, p2, p3);
//! assert!(cross > 0.0); // Counterclockwise orientation
//!
//! // Compute polygon area
//! let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
//! let vertices = vec![0, 1, 2, 3];
//! let area = compute_polygon_area(&points.view(), &vertices).unwrap();
//! assert!((area - 1.0).abs() < 1e-10);
//! ```
//!
//! ## 3D Calculations
//! ```rust
//! use scirs2_spatial::convex_hull::geometry::calculations_3d::{tetrahedron_volume, triangle_area_3d};
//!
//! // Compute tetrahedron volume
//! let p0 = [0.0, 0.0, 0.0];
//! let p1 = [1.0, 0.0, 0.0];
//! let p2 = [0.0, 1.0, 0.0];
//! let p3 = [0.0, 0.0, 1.0];
//! let volume = tetrahedron_volume(p0, p1, p2, p3);
//! assert!((volume.abs() - 1.0/6.0).abs() < 1e-10);
//!
//! // Compute triangle area in 3D
//! let area = triangle_area_3d(p0, p1, p2);
//! assert!((area - 0.5).abs() < 1e-10);
//! ```
//!
//! ## High-Dimensional Calculations
//! ```rust
//! use scirs2_spatial::convex_hull::geometry::high_dimensional::{compute_centroid, compute_bounding_box};
//! use scirs2_core::ndarray::array;
//!
//! let points = array![
//!     [0.0, 0.0, 0.0, 0.0],
//!     [1.0, 0.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0, 0.0],
//!     [0.0, 0.0, 1.0, 0.0]
//! ];
//! let vertices = vec![0, 1, 2, 3];
//!
//! let centroid = compute_centroid(&points.view(), &vertices);
//! let (min_coords, max_coords) = compute_bounding_box(&points.view(), &vertices);
//! ```

pub mod calculations_2d;
pub mod calculations_3d;
pub mod high_dimensional;

// Re-export commonly used functions for convenience
pub use calculations_2d::{
    compute_2d_hull_equations, compute_polygon_area, compute_polygon_perimeter, cross_product_2d,
    distance_squared_2d, is_counterclockwise, polar_angle,
};

pub use calculations_3d::{
    compute_polyhedron_surface_area, compute_polyhedron_volume, cross_product_3d, dot_product_3d,
    normalize_3d, tetrahedron_volume, triangle_area_3d, vector_magnitude_3d,
};

pub use high_dimensional::{
    compute_bounding_box, compute_centroid, compute_characteristic_size,
    compute_high_dim_surface_area, compute_high_dim_volume, estimate_facet_area,
};

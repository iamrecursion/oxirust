//! Surface area computation for convex hulls
//!
//! This module provides surface area (perimeter, surface area, hypervolume)
//! calculations for convex hulls in various dimensions.

use crate::convex_hull::core::ConvexHull;
use crate::convex_hull::geometry::{
    compute_high_dim_surface_area, compute_polygon_perimeter, compute_polyhedron_surface_area,
};
use crate::error::SpatialResult;

/// Compute the surface area of a convex hull
///
/// The surface area computation method depends on the dimensionality:
/// - 1D: Surface "area" is 0 (two endpoints have measure 0)
/// - 2D: Perimeter of the polygon
/// - 3D: Surface area of the polyhedron
/// - nD: High-dimensional surface hypervolume using facet equations
///
/// # Arguments
///
/// * `hull` - The convex hull to compute surface area for
///
/// # Returns
///
/// * Result containing the surface area/perimeter of the convex hull
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::surface_area::compute_surface_area;
/// use scirs2_core::ndarray::array;
///
/// // 2D square with perimeter 4
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).expect("Operation failed");
/// let perimeter = compute_surface_area(&hull).expect("Operation failed");
/// assert!((perimeter - 4.0).abs() < 1e-10);
/// ```
pub fn compute_surface_area(hull: &ConvexHull) -> SpatialResult<f64> {
    match hull.ndim() {
        1 => Ok(0.0), // 1D surface area is 0
        2 => compute_2d_surface_area(hull),
        3 => compute_3d_surface_area(hull),
        _ => compute_nd_surface_area(hull),
    }
}

/// Compute 2D surface area (perimeter)
///
/// # Arguments
///
/// * `hull` - The convex hull (should be 2D)
///
/// # Returns
///
/// * Result containing the perimeter of the 2D hull
fn compute_2d_surface_area(hull: &ConvexHull) -> SpatialResult<f64> {
    if hull.vertex_indices.len() < 3 {
        return Ok(0.0);
    }

    compute_polygon_perimeter(&hull.points.view(), &hull.vertex_indices)
}

/// Compute 3D surface area
///
/// # Arguments
///
/// * `hull` - The convex hull (should be 3D)
///
/// # Returns
///
/// * Result containing the surface area of the 3D hull
fn compute_3d_surface_area(hull: &ConvexHull) -> SpatialResult<f64> {
    if hull.vertex_indices.len() < 4 {
        return Ok(0.0);
    }

    compute_polyhedron_surface_area(&hull.points.view(), &hull.simplices)
}

/// Compute high-dimensional surface area using facet equations
///
/// # Arguments
///
/// * `hull` - The convex hull (should be nD where n > 3)
///
/// # Returns
///
/// * Result containing the high-dimensional surface area
fn compute_nd_surface_area(hull: &ConvexHull) -> SpatialResult<f64> {
    if let Some(equations) = &hull.equations {
        compute_high_dim_surface_area(&hull.points.view(), &hull.vertex_indices, &equations.view())
    } else {
        use crate::error::SpatialError;
        Err(SpatialError::NotImplementedError(
            "Surface area computation for dimensions > 3 requires facet equations".to_string(),
        ))
    }
}

/// Compute surface area bounds (lower and upper estimates)
///
/// This function provides both lower and upper bounds for the surface area,
/// which can be useful for validation or uncertainty quantification.
///
/// # Arguments
///
/// * `hull` - The convex hull
///
/// # Returns
///
/// * Result containing (lower_bound, upper_bound, exact_surface_area) tuple
///   where exact_surface_area is Some(value) if an exact computation was possible
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::surface_area::compute_surface_area_bounds;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).expect("Operation failed");
/// let (lower, upper, exact) = compute_surface_area_bounds(&hull).expect("Operation failed");
/// assert!(exact.is_some());
/// assert!((exact.unwrap() - 4.0).abs() < 1e-10);
/// ```
pub fn compute_surface_area_bounds(hull: &ConvexHull) -> SpatialResult<(f64, f64, Option<f64>)> {
    // For low dimensions, we can compute exactly
    if hull.ndim() <= 3 {
        let exact = compute_surface_area(hull)?;
        return Ok((exact, exact, Some(exact)));
    }

    // For high dimensions, we may need to approximate
    let exact = compute_surface_area(hull).ok();

    if let Some(area) = exact {
        Ok((area, area, Some(area)))
    } else {
        // Use geometric bounds based on bounding box
        let bounds = compute_geometric_surface_area_bounds(hull)?;
        Ok((bounds.0, bounds.1, None))
    }
}

/// Compute geometric bounds for surface area based on hull structure
///
/// # Arguments
///
/// * `hull` - The convex hull
///
/// # Returns
///
/// * Result containing (lower_bound, upper_bound) tuple
fn compute_geometric_surface_area_bounds(hull: &ConvexHull) -> SpatialResult<(f64, f64)> {
    use crate::convex_hull::geometry::compute_bounding_box;

    let ndim = hull.ndim();
    let (min_coords, max_coords) = compute_bounding_box(&hull.points.view(), &hull.vertex_indices);

    // Compute bounding box surface area as upper bound
    let mut bbox_surface_area = 0.0;

    if ndim == 2 {
        // Rectangle perimeter
        let width = max_coords[0] - min_coords[0];
        let height = max_coords[1] - min_coords[1];
        bbox_surface_area = 2.0 * (width + height);
    } else if ndim == 3 {
        // Box surface area
        let width = max_coords[0] - min_coords[0];
        let height = max_coords[1] - min_coords[1];
        let depth = max_coords[2] - min_coords[2];
        bbox_surface_area = 2.0 * (width * height + width * depth + height * depth);
    } else {
        // High-dimensional approximation
        let mut surface_area = 0.0;
        for i in 0..ndim {
            let mut face_area = 1.0;
            for j in 0..ndim {
                if i != j {
                    face_area *= max_coords[j] - min_coords[j];
                }
            }
            surface_area += 2.0 * face_area; // Two faces per dimension
        }
        bbox_surface_area = surface_area;
    }

    // Lower bound: assume minimal surface area (sphere-like)
    // This is a rough approximation
    let characteristic_size = (0..ndim)
        .map(|d| max_coords[d] - min_coords[d])
        .fold(0.0, |acc, x| acc + x * x)
        .sqrt();

    let lower_bound = characteristic_size; // Very rough lower bound
    let upper_bound = bbox_surface_area;

    Ok((lower_bound.min(upper_bound), upper_bound))
}

/// Compute surface area to volume ratio
///
/// This is a useful geometric property that characterizes the shape of the hull.
///
/// # Arguments
///
/// * `hull` - The convex hull
///
/// # Returns
///
/// * Result containing the surface area to volume ratio
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::surface_area::compute_surface_to_volume_ratio;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).expect("Operation failed");
/// let ratio = compute_surface_to_volume_ratio(&hull).expect("Operation failed");
/// assert!((ratio - 4.0).abs() < 1e-10); // Perimeter/Area = 4/1 = 4
/// ```
pub fn compute_surface_to_volume_ratio(hull: &ConvexHull) -> SpatialResult<f64> {
    let surface_area = compute_surface_area(hull)?;
    let volume = crate::convex_hull::properties::volume::compute_volume(hull)?;

    if volume.abs() < 1e-12 {
        // Degenerate case - infinite ratio
        Ok(f64::INFINITY)
    } else {
        Ok(surface_area / volume)
    }
}

/// Compute compactness measure (isoperimetric ratio)
///
/// This measures how close the hull is to the most compact shape (circle/sphere)
/// for its volume. Value is between 0 and 1, where 1 indicates maximum compactness.
///
/// # Arguments
///
/// * `hull` - The convex hull
///
/// # Returns
///
/// * Result containing the compactness measure
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::surface_area::compute_compactness;
/// use scirs2_core::ndarray::array;
///
/// // Circle-like shape should have high compactness
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).expect("Operation failed");
/// let compactness = compute_compactness(&hull).expect("Operation failed");
/// assert!(compactness > 0.7); // Square is fairly compact
/// ```
pub fn compute_compactness(hull: &ConvexHull) -> SpatialResult<f64> {
    let ndim = hull.ndim() as f64;
    let surface_area = compute_surface_area(hull)?;
    let volume = crate::convex_hull::properties::volume::compute_volume(hull)?;

    if volume.abs() < 1e-12 || surface_area.abs() < 1e-12 {
        return Ok(0.0); // Degenerate case
    }

    // Compute the surface area of a sphere with the same volume
    let sphere_surface_area = if ndim == 1.0 {
        0.0 // 1D "sphere" has no surface area
    } else if ndim == 2.0 {
        // Circle: surface area = circumference = 2π√(A/π)
        2.0 * std::f64::consts::PI * (volume / std::f64::consts::PI).sqrt()
    } else if ndim == 3.0 {
        // Sphere: surface area = 4π(3V/(4π))^(2/3)
        let radius = (3.0 * volume / (4.0 * std::f64::consts::PI)).powf(1.0 / 3.0);
        4.0 * std::f64::consts::PI * radius * radius
    } else {
        // High-dimensional sphere (approximate)
        let gamma = |x: f64| (2.0 * std::f64::consts::PI).powf(x / 2.0) / tgamma_approx(x / 2.0);
        let radius = (volume * gamma(ndim + 1.0) / gamma(ndim)).powf(1.0 / ndim);
        ndim * gamma(ndim / 2.0) * radius.powf(ndim - 1.0)
    };

    if sphere_surface_area <= 0.0 {
        Ok(0.0)
    } else {
        // Compactness = (surface area of equivalent sphere) / (actual surface area)
        Ok((sphere_surface_area / surface_area).min(1.0))
    }
}

/// Approximate gamma function (for high-dimensional sphere calculations)
fn tgamma_approx(x: f64) -> f64 {
    // Stirling's approximation for large x
    if x > 10.0 {
        (2.0 * std::f64::consts::PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
    } else {
        // Use simple cases for small values
        match x as i32 {
            1 => 1.0,
            2 => 1.0,
            3 => 2.0,
            4 => 6.0,
            5 => 24.0,
            _ => x * tgamma_approx(x - 1.0), // Recursive for intermediate values
        }
    }
}

/// Check if the surface area computation is likely to be accurate
///
/// # Arguments
///
/// * `hull` - The convex hull
///
/// # Returns
///
/// * true if surface area computation should be accurate, false otherwise
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::surface_area::is_surface_area_computation_reliable;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).expect("Operation failed");
/// assert!(is_surface_area_computation_reliable(&hull));
/// ```
pub fn is_surface_area_computation_reliable(hull: &ConvexHull) -> bool {
    let ndim = hull.ndim();
    let nvertices = hull.vertex_indices.len();
    let nsimplices = hull.simplices.len();

    // 1D, 2D, 3D are generally reliable
    if ndim <= 3 {
        return true;
    }

    // For high dimensions, check if we have enough structure
    if hull.equations.is_some() && nvertices > ndim {
        return true;
    }

    // Check if we have enough simplices for the dimension
    if nsimplices < ndim {
        return false;
    }

    // Very high dimensions with few vertices/simplices are unreliable
    if ndim > 10 && (nvertices < 2 * ndim || nsimplices < ndim) {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex_hull::ConvexHull;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_compute_2d_surface_area() {
        // Unit square - perimeter should be 4
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let perimeter = compute_surface_area(&hull).expect("Operation failed");
        assert!((perimeter - 4.0).abs() < 1e-10);

        // Triangle
        let points = arr2(&[[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let perimeter = compute_surface_area(&hull).expect("Operation failed");
        // Perimeter = 3 + 4 + 5 = 12
        assert!((perimeter - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_1d_surface_area() {
        // Line segment - surface area is 0
        let points = arr2(&[[0.0], [3.0], [1.0], [2.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let surface_area = compute_surface_area(&hull).expect("Operation failed");
        assert_eq!(surface_area, 0.0);
    }

    #[test]
    fn test_compute_surface_area_bounds() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let (lower, upper, exact) = compute_surface_area_bounds(&hull).expect("Operation failed");

        assert!(exact.is_some());
        assert!((exact.expect("Operation failed") - 4.0).abs() < 1e-10);
        assert_eq!(lower, upper); // Should be exact for 2D
    }

    #[test]
    fn test_surface_to_volume_ratio() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let ratio = compute_surface_to_volume_ratio(&hull).expect("Operation failed");
        // Perimeter/Area = 4/1 = 4
        assert!((ratio - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_compactness() {
        // Square
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let compactness = compute_compactness(&hull).expect("Operation failed");
        assert!(compactness > 0.7 && compactness <= 1.0);

        // Triangle (less compact than square)
        let points = arr2(&[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let triangle_compactness = compute_compactness(&hull).expect("Operation failed");
        assert!(triangle_compactness > 0.0 && triangle_compactness <= 1.0);
    }

    #[test]
    fn test_is_surface_area_computation_reliable() {
        // 2D case - should be reliable
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        assert!(is_surface_area_computation_reliable(&hull));

        // 3D case - should be reliable
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        assert!(is_surface_area_computation_reliable(&hull));
    }

    #[test]
    fn test_degenerate_cases() {
        // Single point - should fail to create convex hull
        let points = arr2(&[[1.0, 2.0]]);
        let hull_result = ConvexHull::new(&points.view());
        assert!(hull_result.is_err());

        // Two points in 2D - should fail to create convex hull
        let points = arr2(&[[0.0, 0.0], [1.0, 1.0]]);
        let hull_result = ConvexHull::new(&points.view());
        assert!(hull_result.is_err());

        // Valid 2D triangle (minimal valid case)
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).expect("Operation failed");
        let surface_area = compute_surface_area(&hull).expect("Operation failed");
        // Perimeter of triangle with sides sqrt(2), 1, 1
        let expected_perimeter = 2.0 + (2.0_f64).sqrt();
        assert!((surface_area - expected_perimeter).abs() < 1e-10);
    }
}

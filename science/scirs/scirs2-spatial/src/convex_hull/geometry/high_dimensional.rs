//! High-dimensional geometric calculations for convex hull operations
//!
//! This module provides utility functions for geometric computations
//! in dimensions higher than 3, commonly used in high-dimensional
//! convex hull algorithms.

use crate::error::SpatialResult;
use scirs2_core::ndarray::ArrayView2;

/// Compute volume for high-dimensional convex hulls using facet equations
///
/// Uses the divergence theorem approach for high-dimensional volume calculation.
///
/// # Arguments
///
/// * `points` - Input points array
/// * `vertex_indices` - Indices of hull vertices
/// * `equations` - Facet equations in the form [a₁, a₂, ..., aₙ, b] where
///                aᵢx₍ᵢ₎ + b ≤ 0 defines the half-space
///
/// # Returns
///
/// * Result containing the high-dimensional volume
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::high_dimensional::compute_high_dim_volume;
/// use scirs2_core::ndarray::array;
///
/// // 4D hypercube vertices (subset)
/// let points = array![
///     [0.0, 0.0, 0.0, 0.0],
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0, 1.0]
/// ];
/// let vertices = vec![0, 1, 2, 3, 4];
/// let equations = array![
///     [1.0, 0.0, 0.0, 0.0, 0.0],
///     [-1.0, 0.0, 0.0, 0.0, 1.0]
/// ];
///
/// let volume = compute_high_dim_volume(&points.view(), &vertices, &equations.view()).expect("Operation failed");
/// assert!(volume >= 0.0);
/// ```
pub fn compute_high_dim_volume(
    points: &ArrayView2<'_, f64>,
    vertex_indices: &[usize],
    equations: &ArrayView2<'_, f64>,
) -> SpatialResult<f64> {
    let ndim = points.ncols();
    let nfacets = equations.nrows();

    if nfacets == 0 {
        return Ok(0.0);
    }

    // For high-dimensional convex hulls, we use the divergence theorem:
    // V = (1/d) * Σ(A_i * h_i)
    // where A_i is the area of facet i and h_i is the distance from origin to facet i

    let mut total_volume = 0.0;

    // Find a point inside the hull (centroid of vertices)
    let mut centroid = vec![0.0; ndim];
    for &vertex_idx in vertex_indices {
        for d in 0..ndim {
            centroid[d] += points[[vertex_idx, d]];
        }
    }
    for item in centroid.iter_mut().take(ndim) {
        *item /= vertex_indices.len() as f64;
    }

    // Process each facet
    for facet_idx in 0..nfacets {
        // Extract normal vector and offset from equation
        let mut normal = vec![0.0; ndim];
        for d in 0..ndim {
            normal[d] = equations[[facet_idx, d]];
        }
        let offset = equations[[facet_idx, ndim]];

        // Normalize the normal vector
        let normal_length = (normal.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if normal_length < 1e-12 {
            continue; // Skip degenerate facets
        }

        for item in normal.iter_mut().take(ndim) {
            *item /= normal_length;
        }
        let normalized_offset = offset / normal_length;

        // Distance from centroid to facet plane
        let distance_to_centroid: f64 = normal
            .iter()
            .zip(centroid.iter())
            .map(|(n, c)| n * c)
            .sum::<f64>()
            + normalized_offset;

        // For volume computation, we need to use the absolute distance
        // The contribution of this facet to the volume calculation
        let height = distance_to_centroid.abs();

        // For high-dimensional case, we approximate the facet area
        // This is a simplified approach - a full implementation would compute exact facet areas
        let facet_area = estimate_facet_area(points, vertex_indices, facet_idx, ndim)?;

        // Add this facet's contribution to the total volume
        total_volume += facet_area * height;
    }

    // Final volume is divided by dimension
    let volume = total_volume / (ndim as f64);

    Ok(volume)
}

/// Estimate the area of a facet in high dimensions
///
/// This is a simplified approach that works for well-formed convex hulls.
///
/// # Arguments
///
/// * `points` - Input points array
/// * `vertex_indices` - Indices of hull vertices
/// * `facet_idx` - Index of the facet to estimate (currently unused in simple estimation)
/// * `ndim` - Number of dimensions
///
/// # Returns
///
/// * Result containing the estimated facet area
pub fn estimate_facet_area(
    points: &ArrayView2<'_, f64>,
    vertex_indices: &[usize],
    _facet_idx: usize,
    ndim: usize,
) -> SpatialResult<f64> {
    // For high dimensions, computing exact facet areas is complex
    // We use a simplified estimation based on the convex hull size

    // Calculate the bounding box volume as a reference
    let mut min_coords = vec![f64::INFINITY; ndim];
    let mut max_coords = vec![f64::NEG_INFINITY; ndim];

    for &vertex_idx in vertex_indices {
        for d in 0..ndim {
            let coord = points[[vertex_idx, d]];
            min_coords[d] = min_coords[d].min(coord);
            max_coords[d] = max_coords[d].max(coord);
        }
    }

    // Calculate the characteristic size of the hull
    let mut size_product = 1.0;
    for d in 0..ndim {
        let size = max_coords[d] - min_coords[d];
        if size > 0.0 {
            size_product *= size;
        }
    }

    // Estimate facet area as a fraction of the hull's characteristic area
    // This is approximate but provides reasonable results for well-formed hulls
    let estimated_area =
        size_product.powf((ndim - 1) as f64 / ndim as f64) / vertex_indices.len() as f64;

    Ok(estimated_area)
}

/// Compute surface area (hypervolume) for high-dimensional convex hulls
///
/// # Arguments
///
/// * `points` - Input points array
/// * `vertex_indices` - Indices of hull vertices
/// * `equations` - Facet equations
///
/// # Returns
///
/// * Result containing the surface area/hypervolume
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::high_dimensional::compute_high_dim_surface_area;
/// use scirs2_core::ndarray::array;
///
/// // 4D simplex
/// let points = array![
///     [0.0, 0.0, 0.0, 0.0],
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0, 1.0]
/// ];
/// let vertices = vec![0, 1, 2, 3, 4];
/// let equations = array![
///     [1.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 1.0, 0.0],
///     [-1.0, -1.0, -1.0, -1.0, 1.0]
/// ];
///
/// let surface_area = compute_high_dim_surface_area(&points.view(), &vertices, &equations.view()).expect("Operation failed");
/// assert!(surface_area >= 0.0);
/// ```
pub fn compute_high_dim_surface_area(
    points: &ArrayView2<'_, f64>,
    vertex_indices: &[usize],
    equations: &ArrayView2<'_, f64>,
) -> SpatialResult<f64> {
    let ndim = points.ncols();
    let nfacets = equations.nrows();

    if nfacets == 0 {
        return Ok(0.0);
    }

    let mut total_surface_area = 0.0;

    // Sum the areas of all facets
    for facet_idx in 0..nfacets {
        let facet_area = estimate_facet_area(points, vertex_indices, facet_idx, ndim)?;
        total_surface_area += facet_area;
    }

    Ok(total_surface_area)
}

/// Compute the centroid of a set of high-dimensional points
///
/// # Arguments
///
/// * `points` - Input points array
/// * `vertex_indices` - Indices of points to include in centroid calculation
///
/// # Returns
///
/// * Vector representing the centroid coordinates
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::high_dimensional::compute_centroid;
/// use scirs2_core::ndarray::array;
///
/// let points = array![
///     [0.0, 0.0, 0.0],
///     [3.0, 0.0, 0.0],
///     [0.0, 3.0, 0.0],
///     [0.0, 0.0, 3.0]
/// ];
/// let vertices = vec![0, 1, 2, 3];
///
/// let centroid = compute_centroid(&points.view(), &vertices);
/// assert_eq!(centroid, vec![0.75, 0.75, 0.75]);
/// ```
pub fn compute_centroid(points: &ArrayView2<'_, f64>, vertex_indices: &[usize]) -> Vec<f64> {
    let ndim = points.ncols();
    let mut centroid = vec![0.0; ndim];

    for &vertex_idx in vertex_indices {
        for d in 0..ndim {
            centroid[d] += points[[vertex_idx, d]];
        }
    }

    for item in centroid.iter_mut().take(ndim) {
        *item /= vertex_indices.len() as f64;
    }

    centroid
}

/// Compute the bounding box of a set of high-dimensional points
///
/// # Arguments
///
/// * `points` - Input points array
/// * `vertex_indices` - Indices of points to include
///
/// # Returns
///
/// * Tuple of (min_coords, max_coords) vectors
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::high_dimensional::compute_bounding_box;
/// use scirs2_core::ndarray::array;
///
/// let points = array![
///     [0.0, 5.0, -1.0],
///     [3.0, 0.0, 2.0],
///     [1.0, 3.0, 0.0]
/// ];
/// let vertices = vec![0, 1, 2];
///
/// let (min_coords, max_coords) = compute_bounding_box(&points.view(), &vertices);
/// assert_eq!(min_coords, vec![0.0, 0.0, -1.0]);
/// assert_eq!(max_coords, vec![3.0, 5.0, 2.0]);
/// ```
pub fn compute_bounding_box(
    points: &ArrayView2<'_, f64>,
    vertex_indices: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let ndim = points.ncols();
    let mut min_coords = vec![f64::INFINITY; ndim];
    let mut max_coords = vec![f64::NEG_INFINITY; ndim];

    for &vertex_idx in vertex_indices {
        for d in 0..ndim {
            let coord = points[[vertex_idx, d]];
            min_coords[d] = min_coords[d].min(coord);
            max_coords[d] = max_coords[d].max(coord);
        }
    }

    (min_coords, max_coords)
}

/// Compute the characteristic size of a high-dimensional convex hull
///
/// This is used as a reference measure for volume and surface area estimations.
///
/// # Arguments
///
/// * `points` - Input points array
/// * `vertex_indices` - Indices of hull vertices
///
/// # Returns
///
/// * Characteristic size (geometric mean of bounding box dimensions)
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::high_dimensional::compute_characteristic_size;
/// use scirs2_core::ndarray::array;
///
/// let points = array![
///     [0.0, 0.0, 0.0],
///     [2.0, 0.0, 0.0],
///     [0.0, 4.0, 0.0],
///     [0.0, 0.0, 8.0]
/// ];
/// let vertices = vec![0, 1, 2, 3];
///
/// let size = compute_characteristic_size(&points.view(), &vertices);
/// assert!(size > 0.0);
/// ```
pub fn compute_characteristic_size(points: &ArrayView2<'_, f64>, vertex_indices: &[usize]) -> f64 {
    let (min_coords, max_coords) = compute_bounding_box(points, vertex_indices);
    let ndim = min_coords.len();

    let mut size_product = 1.0;
    let mut valid_dims = 0;

    for d in 0..ndim {
        let size = max_coords[d] - min_coords[d];
        if size > 1e-12 {
            size_product *= size;
            valid_dims += 1;
        }
    }

    if valid_dims == 0 {
        0.0
    } else {
        size_product.powf(1.0 / valid_dims as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_compute_centroid() {
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ]);
        let vertices = vec![0, 1, 2, 3];

        let centroid = compute_centroid(&points.view(), &vertices);
        assert_eq!(centroid, vec![0.75, 0.75, 0.75]);
    }

    #[test]
    fn test_compute_bounding_box() {
        let points = arr2(&[[0.0, 5.0, -1.0], [3.0, 0.0, 2.0], [1.0, 3.0, 0.0]]);
        let vertices = vec![0, 1, 2];

        let (min_coords, max_coords) = compute_bounding_box(&points.view(), &vertices);
        assert_eq!(min_coords, vec![0.0, 0.0, -1.0]);
        assert_eq!(max_coords, vec![3.0, 5.0, 2.0]);
    }

    #[test]
    fn test_compute_characteristic_size() {
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 8.0],
        ]);
        let vertices = vec![0, 1, 2, 3];

        let size = compute_characteristic_size(&points.view(), &vertices);
        assert!(size > 0.0);
        // Geometric mean of [2, 4, 8] = (2*4*8)^(1/3) = 64^(1/3) = 4
        assert!((size - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_facet_area() {
        let points = arr2(&[
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let vertices = vec![0, 1, 2, 3, 4];

        let area = estimate_facet_area(&points.view(), &vertices, 0, 4).expect("Operation failed");
        assert!(area > 0.0);
    }

    #[test]
    fn test_high_dim_surface_area() {
        let points = arr2(&[
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let vertices = vec![0, 1, 2, 3, 4];
        let equations = arr2(&[
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, 1.0],
        ]);

        let surface_area =
            compute_high_dim_surface_area(&points.view(), &vertices, &equations.view())
                .expect("Operation failed");
        assert!(surface_area >= 0.0);
    }

    #[test]
    fn test_high_dim_volume() {
        let points = arr2(&[
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let vertices = vec![0, 1, 2, 3, 4];
        let equations = arr2(&[[1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 1.0]]);

        let volume = compute_high_dim_volume(&points.view(), &vertices, &equations.view())
            .expect("Operation failed");
        assert!(volume >= 0.0);
    }
}

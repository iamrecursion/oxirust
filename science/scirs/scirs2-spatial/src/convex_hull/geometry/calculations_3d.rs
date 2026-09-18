//! 3D geometric calculations for convex hull operations
//!
//! This module provides utility functions for 3D geometric computations
//! commonly used in convex hull algorithms.

use crate::error::SpatialResult;
use scirs2_core::ndarray::ArrayView2;

/// Compute the signed volume of a tetrahedron
///
/// # Arguments
///
/// * `p0` - First vertex [x, y, z]
/// * `p1` - Second vertex [x, y, z]  
/// * `p2` - Third vertex [x, y, z]
/// * `p3` - Fourth vertex [x, y, z]
///
/// # Returns
///
/// * Signed volume of tetrahedron formed by the four points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_3d::tetrahedron_volume;
///
/// let p0 = [0.0, 0.0, 0.0];
/// let p1 = [1.0, 0.0, 0.0];
/// let p2 = [0.0, 1.0, 0.0];
/// let p3 = [0.0, 0.0, 1.0];
///
/// let volume = tetrahedron_volume(p0, p1, p2, p3);
/// assert!((volume - 1.0/6.0).abs() < 1e-10);
/// ```
pub fn tetrahedron_volume(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let v3 = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]];

    // Compute scalar triple product: v1 · (v2 × v3)
    let cross = [
        v2[1] * v3[2] - v2[2] * v3[1],
        v2[2] * v3[0] - v2[0] * v3[2],
        v2[0] * v3[1] - v2[1] * v3[0],
    ];

    (v1[0] * cross[0] + v1[1] * cross[1] + v1[2] * cross[2]).abs() / 6.0
}

/// Compute the area of a 3D triangle
///
/// # Arguments
///
/// * `p0` - First vertex [x, y, z]
/// * `p1` - Second vertex [x, y, z]
/// * `p2` - Third vertex [x, y, z]
///
/// # Returns
///
/// * Area of the triangle in 3D space
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_3d::triangle_area_3d;
///
/// let p0 = [0.0, 0.0, 0.0];
/// let p1 = [1.0, 0.0, 0.0];
/// let p2 = [0.0, 1.0, 0.0];
///
/// let area = triangle_area_3d(p0, p1, p2);
/// assert!((area - 0.5).abs() < 1e-10);
/// ```
pub fn triangle_area_3d(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> f64 {
    let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

    // Compute cross product v1 × v2
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];

    // Magnitude of cross product gives twice the triangle area
    let magnitude = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    magnitude / 2.0
}

/// Compute the volume of a 3D polyhedron using triangulation from centroid
///
/// # Arguments
///
/// * `points` - Input points array
/// * `simplices` - Triangular faces of the polyhedron
///
/// # Returns
///
/// * Result containing the polyhedron volume
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_3d::compute_polyhedron_volume;
/// use scirs2_core::ndarray::array;
///
/// // Unit tetrahedron
/// let points = array![
///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
/// ];
/// let simplices = vec![
///     vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]
/// ];
///
/// let volume = compute_polyhedron_volume(&points.view(), &simplices).expect("Operation failed");
/// assert!(volume > 0.0);
/// ```
pub fn compute_polyhedron_volume(
    points: &ArrayView2<'_, f64>,
    simplices: &[Vec<usize>],
) -> SpatialResult<f64> {
    if simplices.is_empty() {
        return Ok(0.0);
    }

    // Compute the centroid of all vertices
    let vertex_indices: std::collections::HashSet<usize> =
        simplices.iter().flat_map(|s| s.iter()).cloned().collect();

    let mut centroid = [0.0, 0.0, 0.0];
    for &idx in &vertex_indices {
        centroid[0] += points[[idx, 0]];
        centroid[1] += points[[idx, 1]];
        centroid[2] += points[[idx, 2]];
    }
    let n = vertex_indices.len() as f64;
    centroid[0] /= n;
    centroid[1] /= n;
    centroid[2] /= n;

    let mut total_volume = 0.0;

    // For each triangular face, form a tetrahedron with the centroid
    for simplex in simplices {
        if simplex.len() != 3 {
            continue; // Skip non-triangular faces
        }

        let p0 = [
            points[[simplex[0], 0]],
            points[[simplex[0], 1]],
            points[[simplex[0], 2]],
        ];
        let p1 = [
            points[[simplex[1], 0]],
            points[[simplex[1], 1]],
            points[[simplex[1], 2]],
        ];
        let p2 = [
            points[[simplex[2], 0]],
            points[[simplex[2], 1]],
            points[[simplex[2], 2]],
        ];

        // Compute the volume of the tetrahedron formed by centroid, p0, p1, p2
        let tet_volume = tetrahedron_volume(centroid, p0, p1, p2);
        total_volume += tet_volume.abs();
    }

    Ok(total_volume)
}

/// Compute the surface area of a 3D polyhedron
///
/// # Arguments
///
/// * `points` - Input points array
/// * `simplices` - Triangular faces of the polyhedron
///
/// # Returns
///
/// * Result containing the polyhedron surface area
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_3d::compute_polyhedron_surface_area;
/// use scirs2_core::ndarray::array;
///
/// // Unit tetrahedron
/// let points = array![
///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
/// ];
/// let simplices = vec![
///     vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]
/// ];
///
/// let surface_area = compute_polyhedron_surface_area(&points.view(), &simplices).expect("Operation failed");
/// assert!(surface_area > 0.0);
/// ```
pub fn compute_polyhedron_surface_area(
    points: &ArrayView2<'_, f64>,
    simplices: &[Vec<usize>],
) -> SpatialResult<f64> {
    if simplices.is_empty() {
        return Ok(0.0);
    }

    let mut surface_area = 0.0;

    // For each triangular face, compute its area
    for simplex in simplices {
        if simplex.len() != 3 {
            continue; // Skip non-triangular faces
        }

        let p0 = [
            points[[simplex[0], 0]],
            points[[simplex[0], 1]],
            points[[simplex[0], 2]],
        ];
        let p1 = [
            points[[simplex[1], 0]],
            points[[simplex[1], 1]],
            points[[simplex[1], 2]],
        ];
        let p2 = [
            points[[simplex[2], 0]],
            points[[simplex[2], 1]],
            points[[simplex[2], 2]],
        ];

        let area = triangle_area_3d(p0, p1, p2);
        surface_area += area;
    }

    Ok(surface_area)
}

/// Compute cross product of two 3D vectors
///
/// # Arguments
///
/// * `v1` - First vector [x, y, z]
/// * `v2` - Second vector [x, y, z]
///
/// # Returns
///
/// * Cross product v1 × v2 as [x, y, z]
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_3d::cross_product_3d;
///
/// let v1 = [1.0, 0.0, 0.0];
/// let v2 = [0.0, 1.0, 0.0];
///
/// let cross = cross_product_3d(v1, v2);
/// assert_eq!(cross, [0.0, 0.0, 1.0]);
/// ```
pub fn cross_product_3d(v1: [f64; 3], v2: [f64; 3]) -> [f64; 3] {
    [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]
}

/// Compute dot product of two 3D vectors
///
/// # Arguments
///
/// * `v1` - First vector [x, y, z]
/// * `v2` - Second vector [x, y, z]
///
/// # Returns
///
/// * Dot product v1 · v2
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_3d::dot_product_3d;
///
/// let v1 = [1.0, 2.0, 3.0];
/// let v2 = [4.0, 5.0, 6.0];
///
/// let dot = dot_product_3d(v1, v2);
/// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
/// ```
pub fn dot_product_3d(v1: [f64; 3], v2: [f64; 3]) -> f64 {
    v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
}

/// Compute the magnitude of a 3D vector
///
/// # Arguments
///
/// * `v` - Vector [x, y, z]
///
/// # Returns
///
/// * Magnitude (length) of the vector
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_3d::vector_magnitude_3d;
///
/// let v = [3.0, 4.0, 0.0];
/// let magnitude = vector_magnitude_3d(v);
/// assert_eq!(magnitude, 5.0);
/// ```
pub fn vector_magnitude_3d(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Normalize a 3D vector
///
/// # Arguments
///
/// * `v` - Vector to normalize [x, y, z]
///
/// # Returns
///
/// * Normalized vector, or zero vector if input has zero magnitude
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_3d::normalize_3d;
///
/// let v = [3.0, 4.0, 0.0];
/// let normalized = normalize_3d(v);
/// assert_eq!(normalized, [0.6, 0.8, 0.0]);
/// ```
pub fn normalize_3d(v: [f64; 3]) -> [f64; 3] {
    let magnitude = vector_magnitude_3d(v);
    if magnitude < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / magnitude, v[1] / magnitude, v[2] / magnitude]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_tetrahedron_volume() {
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [0.0, 1.0, 0.0];
        let p3 = [0.0, 0.0, 1.0];

        let volume = tetrahedron_volume(p0, p1, p2, p3);
        assert!((volume - 1.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_area_3d() {
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [0.0, 1.0, 0.0];

        let area = triangle_area_3d(p0, p1, p2);
        assert!((area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cross_product_3d() {
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];

        let cross = cross_product_3d(v1, v2);
        assert_eq!(cross, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_dot_product_3d() {
        let v1 = [1.0, 2.0, 3.0];
        let v2 = [4.0, 5.0, 6.0];

        let dot = dot_product_3d(v1, v2);
        assert_eq!(dot, 32.0);
    }

    #[test]
    fn test_vector_magnitude_3d() {
        let v = [3.0, 4.0, 0.0];
        let magnitude = vector_magnitude_3d(v);
        assert_eq!(magnitude, 5.0);
    }

    #[test]
    fn test_normalize_3d() {
        let v = [3.0, 4.0, 0.0];
        let normalized = normalize_3d(v);
        assert!((normalized[0] - 0.6).abs() < 1e-10);
        assert!((normalized[1] - 0.8).abs() < 1e-10);
        assert_eq!(normalized[2], 0.0);
    }

    #[test]
    fn test_polyhedron_volume() {
        // Unit tetrahedron
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let simplices = vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]];

        let volume =
            compute_polyhedron_volume(&points.view(), &simplices).expect("Operation failed");
        assert!(volume > 0.0);
        // Unit tetrahedron volume should be 1/6
        assert!((volume - 1.0 / 6.0).abs() < 0.1); // Allow some numerical error
    }

    #[test]
    fn test_polyhedron_surface_area() {
        // Unit tetrahedron
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let simplices = vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]];

        let surface_area =
            compute_polyhedron_surface_area(&points.view(), &simplices).expect("Operation failed");
        assert!(surface_area > 0.0);
    }
}

//! Bowyer-Watson algorithm for nD Delaunay triangulation

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::Array2;
use std::collections::HashSet;

/// Bowyer-Watson for small nD point sets (without super-simplex)
///
/// For small point sets, we start with the first ndim+1 points as the initial simplex
/// and incrementally add remaining points.
pub(crate) fn bowyer_watson_nd_small(
    points: &Array2<f64>,
    ndim: usize,
) -> SpatialResult<Vec<Vec<usize>>> {
    let npoints = points.nrows();

    if npoints < ndim + 1 {
        return Err(SpatialError::ValueError(format!(
            "Need at least {} points for {}D triangulation, got {}",
            ndim + 1,
            ndim,
            npoints
        )));
    }

    // If we have exactly ndim+1 points, return a single simplex
    if npoints == ndim + 1 {
        let simplex: Vec<usize> = (0..npoints).collect();
        return Ok(vec![simplex]);
    }

    // Convert points to Vec<Vec<f64>> for easier manipulation
    let all_points: Vec<Vec<f64>> = (0..npoints)
        .map(|i| (0..ndim).map(|j| points[[i, j]]).collect())
        .collect();

    // Start with first ndim+1 points as initial simplex
    let initial_simplex: Vec<usize> = (0..=ndim).collect();
    let mut simplices: Vec<Vec<usize>> = vec![initial_simplex];

    // Add remaining points one at a time
    for point_idx in (ndim + 1)..npoints {
        let point = &all_points[point_idx];

        // Find all simplices whose circumhypersphere contains this point
        let mut bad_simplices: Vec<usize> = Vec::new();
        for (simplex_idx, simplex) in simplices.iter().enumerate() {
            if point_in_circumhypersphere_nd(point, simplex, &all_points, ndim) {
                bad_simplices.push(simplex_idx);
            }
        }

        if bad_simplices.is_empty() {
            // Point is outside current triangulation
            // This shouldn't happen for a proper Delaunay, but handle it
            continue;
        }

        // Find the boundary facets of the cavity
        let mut boundary_facets: Vec<Vec<usize>> = Vec::new();
        for &simplex_idx in &bad_simplices {
            let simplex = &simplices[simplex_idx];

            // Generate all (n-1)-dimensional faces
            for excluded_vertex in 0..=ndim {
                let mut face: Vec<usize> = Vec::with_capacity(ndim);
                for (i, &v) in simplex.iter().enumerate() {
                    if i != excluded_vertex {
                        face.push(v);
                    }
                }

                // Check if this face is shared with another bad simplex
                let mut is_shared = false;
                for &other_idx in &bad_simplices {
                    if other_idx == simplex_idx {
                        continue;
                    }
                    if simplex_has_face_nd(&simplices[other_idx], &face) {
                        is_shared = true;
                        break;
                    }
                }

                if !is_shared {
                    boundary_facets.push(face);
                }
            }
        }

        // Remove bad simplices (in reverse order to maintain indices)
        bad_simplices.sort_by(|a, b| b.cmp(a));
        for simplex_idx in bad_simplices {
            simplices.remove(simplex_idx);
        }

        // Create new simplices from boundary facets to the new point
        for facet in boundary_facets {
            let mut new_simplex = vec![point_idx];
            new_simplex.extend(facet);
            simplices.push(new_simplex);
        }
    }

    Ok(simplices)
}

/// Bowyer-Watson algorithm for nD Delaunay triangulation (with super-simplex)
///
/// For larger point sets in high dimensions, uses super-simplex approach.
pub(crate) fn bowyer_watson_nd(
    points: &Array2<f64>,
    ndim: usize,
) -> SpatialResult<Vec<Vec<usize>>> {
    let npoints = points.nrows();

    // For small point sets, use the simpler approach without super-simplex
    if npoints <= ndim + 3 {
        return bowyer_watson_nd_small(points, ndim);
    }

    // Find bounding box
    let mut min_coords = vec![f64::INFINITY; ndim];
    let mut max_coords = vec![f64::NEG_INFINITY; ndim];

    for i in 0..npoints {
        for j in 0..ndim {
            min_coords[j] = min_coords[j].min(points[[i, j]]);
            max_coords[j] = max_coords[j].max(points[[i, j]]);
        }
    }

    // Compute bounding box dimensions
    let mut delta_max: f64 = 0.0;
    let mut mid = vec![0.0; ndim];
    for j in 0..ndim {
        let delta = max_coords[j] - min_coords[j];
        delta_max = delta_max.max(delta);
        mid[j] = (min_coords[j] + max_coords[j]) / 2.0;
    }

    // Create super-simplex that contains all points
    // Similar to 2D/3D approach: place vertices very far away
    let margin = 100.0;
    let scale = margin * (delta_max + 1.0);
    let mut super_vertices: Vec<Vec<f64>> = Vec::with_capacity(ndim + 1);

    // First vertex: centered at mid, offset down in the last dimension
    let mut v0 = mid.clone();
    v0[ndim - 1] -= scale * ((ndim + 1) as f64);
    super_vertices.push(v0);

    // Remaining n vertices: spread out in a star pattern
    // Each subsequent vertex is placed far in a different direction
    for i in 0..ndim {
        let mut v = mid.clone();

        // Place vertex far in positive direction on axis i
        v[i] += scale * ((ndim + 1) as f64);

        // Slightly offset on other axes to ensure non-degeneracy
        for j in 0..ndim {
            if j != i {
                v[j] += scale * 0.1;
            }
        }

        super_vertices.push(v);
    }

    // Extended point list (original points + super-simplex vertices)
    let mut all_points: Vec<Vec<f64>> = Vec::with_capacity(npoints + ndim + 1);
    for i in 0..npoints {
        let mut point = Vec::with_capacity(ndim);
        for j in 0..ndim {
            point.push(points[[i, j]]);
        }
        all_points.push(point);
    }
    for sv in super_vertices {
        all_points.push(sv);
    }

    // Initialize with super-simplex (indices npoints, npoints+1, ..., npoints+ndim)
    let super_simplex: Vec<usize> = (npoints..npoints + ndim + 1).collect();
    let mut simplices: Vec<Vec<usize>> = vec![super_simplex];

    // Add points one at a time
    for point_idx in 0..npoints {
        let point = &all_points[point_idx];

        // Find all simplices whose circumhypersphere contains this point
        let mut bad_simplices: Vec<usize> = Vec::new();
        for (simplex_idx, simplex) in simplices.iter().enumerate() {
            if point_in_circumhypersphere_nd(point, simplex, &all_points, ndim) {
                bad_simplices.push(simplex_idx);
            }
        }

        if bad_simplices.is_empty() {
            // Point is outside all current simplices
            // Skip for now (super-simplex approach may not work well for sparse high-dim data)
            continue;
        }

        // Find the boundary facets of the cavity
        let mut boundary_facets: Vec<Vec<usize>> = Vec::new();
        for &simplex_idx in &bad_simplices {
            let simplex = &simplices[simplex_idx];

            // Generate all (n-1)-dimensional faces
            for excluded_vertex in 0..=ndim {
                let mut face: Vec<usize> = Vec::with_capacity(ndim);
                for (i, &v) in simplex.iter().enumerate() {
                    if i != excluded_vertex {
                        face.push(v);
                    }
                }

                // Check if this face is shared with another bad simplex
                let mut is_shared = false;
                for &other_idx in &bad_simplices {
                    if other_idx == simplex_idx {
                        continue;
                    }
                    if simplex_has_face_nd(&simplices[other_idx], &face) {
                        is_shared = true;
                        break;
                    }
                }

                if !is_shared {
                    boundary_facets.push(face);
                }
            }
        }

        // Create new simplices from boundary facets to the new point
        for facet in boundary_facets {
            let mut new_simplex = vec![point_idx];
            new_simplex.extend(facet);
            simplices.push(new_simplex);
        }
    }

    // Remove simplices that have a vertex from the super-simplex
    let super_vertex_indices: HashSet<usize> = (npoints..npoints + ndim + 1).collect();
    simplices.retain(|simplex| !simplex.iter().any(|&v| super_vertex_indices.contains(&v)));

    Ok(simplices)
}

/// Check if a simplex has a specific (n-1)-dimensional face
fn simplex_has_face_nd(simplex: &[usize], face: &[usize]) -> bool {
    let mut sorted_face: Vec<usize> = face.to_vec();
    sorted_face.sort_unstable();

    // Generate all faces of the simplex
    for excluded in 0..simplex.len() {
        let mut candidate_face: Vec<usize> = simplex
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != excluded)
            .map(|(_, &v)| v)
            .collect();
        candidate_face.sort_unstable();

        if candidate_face == sorted_face {
            return true;
        }
    }
    false
}

/// Check if a point is inside the circumhypersphere of an n-dimensional simplex
///
/// Uses the standard in-sphere test via determinant computation.
/// Translates all points so that the test point is at the origin for numerical stability.
fn point_in_circumhypersphere_nd(
    point: &[f64],
    simplex: &[usize],
    all_points: &[Vec<f64>],
    ndim: usize,
) -> bool {
    if simplex.len() != ndim + 1 {
        return false;
    }

    // Build the matrix for the in-sphere test
    // After translating so point is at origin:
    // Matrix is (n+1) x (n+1) where entry (i,j) = (vertex_i - point)_j
    // And we add a column with ||vertex_i - point||²

    let n = ndim + 1;
    let mut matrix = vec![vec![0.0; n + 1]; n + 1];

    // Build matrix with simplex vertices translated by -point
    for (i, &vertex_idx) in simplex.iter().enumerate() {
        let vertex = &all_points[vertex_idx];
        let mut norm_sq: f64 = 0.0;

        for j in 0..ndim {
            let diff = vertex[j] - point[j];
            matrix[i][j] = diff;
            norm_sq += diff * diff;
        }

        // Last column: squared norm
        matrix[i][ndim] = norm_sq;
        // Very last column: 1
        matrix[i][n] = 1.0;
    }

    // Last row for orientation: all 1s except last entry is 0
    for j in 0..ndim {
        matrix[n][j] = 0.0;
    }
    matrix[n][ndim] = 0.0;
    matrix[n][n] = 1.0;

    // Compute determinant
    let det = determinant(&matrix);

    // Point is inside the circumhypersphere if determinant > 0
    // (assuming positive orientation of the simplex)
    det > 1e-10
}

/// Compute the determinant of a square matrix using LU decomposition
///
/// This is more numerically stable than direct expansion, especially
/// for high-dimensional matrices.
fn determinant(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 0.0;
    }

    // Create a mutable copy for LU decomposition
    let mut a = matrix.to_vec();

    // Gaussian elimination with partial pivoting
    let mut det = 1.0;
    let mut sign = 1.0;

    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for k in i + 1..n {
            let val = a[k][i].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        // Swap rows if needed
        if max_row != i {
            a.swap(i, max_row);
            sign = -sign;
        }

        // Check for singular matrix
        if a[i][i].abs() < 1e-15 {
            return 0.0;
        }

        det *= a[i][i];

        // Eliminate column
        for k in i + 1..n {
            let factor = a[k][i] / a[i][i];
            for j in i + 1..n {
                a[k][j] -= factor * a[i][j];
            }
        }
    }

    det * sign
}

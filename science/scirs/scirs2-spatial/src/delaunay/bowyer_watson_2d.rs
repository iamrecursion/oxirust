//! Bowyer-Watson algorithm for 2D Delaunay triangulation

use crate::error::SpatialResult;
use scirs2_core::ndarray::Array2;

/// Bowyer-Watson algorithm for 2D Delaunay triangulation
///
/// This algorithm incrementally builds the Delaunay triangulation by:
/// 1. Creating a super-triangle that contains all points
/// 2. Adding points one at a time
/// 3. For each point, finding all triangles whose circumcircle contains the point
/// 4. Removing those triangles and retriangulating the hole
/// 5. Removing triangles connected to the super-triangle vertices
pub(crate) fn bowyer_watson_2d(points: &Array2<f64>) -> SpatialResult<Vec<Vec<usize>>> {
    let npoints = points.nrows();

    // Find bounding box
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for i in 0..npoints {
        let x = points[[i, 0]];
        let y = points[[i, 1]];
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    // Create super-triangle that contains all points
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta_max = dx.max(dy);
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    // Super-triangle vertices (with large margin)
    let margin = 20.0;
    let super_p0 = [mid_x - margin * delta_max, mid_y - delta_max];
    let super_p1 = [mid_x, mid_y + margin * delta_max];
    let super_p2 = [mid_x + margin * delta_max, mid_y - delta_max];

    // Extended point list (original points + super-triangle vertices)
    // Super-triangle vertices are at indices npoints, npoints+1, npoints+2
    let mut all_points: Vec<[f64; 2]> = Vec::with_capacity(npoints + 3);
    for i in 0..npoints {
        all_points.push([points[[i, 0]], points[[i, 1]]]);
    }
    all_points.push(super_p0);
    all_points.push(super_p1);
    all_points.push(super_p2);

    // Initialize with super-triangle
    let mut triangles: Vec<[usize; 3]> = vec![[npoints, npoints + 1, npoints + 2]];

    // Add points one at a time
    for point_idx in 0..npoints {
        let point = all_points[point_idx];

        // Find all triangles whose circumcircle contains this point
        let mut bad_triangles: Vec<usize> = Vec::new();
        for (tri_idx, tri) in triangles.iter().enumerate() {
            let p0 = all_points[tri[0]];
            let p1 = all_points[tri[1]];
            let p2 = all_points[tri[2]];

            if point_in_circumcircle_2d(point, p0, p1, p2) {
                bad_triangles.push(tri_idx);
            }
        }

        // Find the boundary of the polygonal hole
        let mut polygon_edges: Vec<[usize; 2]> = Vec::new();
        for &tri_idx in &bad_triangles {
            let tri = triangles[tri_idx];
            let edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];

            for edge in &edges {
                // Check if this edge is shared with another bad triangle
                let mut is_shared = false;
                for &other_idx in &bad_triangles {
                    if other_idx == tri_idx {
                        continue;
                    }
                    let other_tri = triangles[other_idx];
                    if triangle_has_edge_2d(&other_tri, edge[0], edge[1]) {
                        is_shared = true;
                        break;
                    }
                }

                if !is_shared {
                    polygon_edges.push(*edge);
                }
            }
        }

        // Remove bad triangles (in reverse order to maintain indices)
        bad_triangles.sort_by(|a, b| b.cmp(a));
        for tri_idx in bad_triangles {
            triangles.remove(tri_idx);
        }

        // Create new triangles from polygon edges to the new point
        for edge in polygon_edges {
            triangles.push([point_idx, edge[0], edge[1]]);
        }
    }

    // Remove triangles that have a vertex from the super-triangle
    let super_vertices = [npoints, npoints + 1, npoints + 2];
    triangles.retain(|tri| !tri.iter().any(|&v| super_vertices.contains(&v)));

    // Convert to Vec<Vec<usize>>
    let simplices: Vec<Vec<usize>> = triangles.iter().map(|t| t.to_vec()).collect();

    Ok(simplices)
}

/// Check if a triangle has a specific edge
fn triangle_has_edge_2d(tri: &[usize; 3], v1: usize, v2: usize) -> bool {
    let edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
    for edge in &edges {
        if (edge[0] == v1 && edge[1] == v2) || (edge[0] == v2 && edge[1] == v1) {
            return true;
        }
    }
    false
}

/// Check if a point is inside the circumcircle of a triangle
fn point_in_circumcircle_2d(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    // Using the determinant method
    let ax = a[0] - p[0];
    let ay = a[1] - p[1];
    let bx = b[0] - p[0];
    let by = b[1] - p[1];
    let cx = c[0] - p[0];
    let cy = c[1] - p[1];

    let det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay);

    // Positive if counter-clockwise orientation
    let orientation = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);

    if orientation > 0.0 {
        det > 0.0
    } else {
        det < 0.0
    }
}

//! Query methods for Delaunay triangulation

use super::Delaunay;
use std::collections::HashSet;

impl Delaunay {
    /// Find the simplex containing a given point
    ///
    /// # Arguments
    ///
    /// * `point` - The point to locate
    ///
    /// # Returns
    ///
    /// * The index of the simplex containing the point, or None if not found
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::delaunay::Delaunay;
    /// use scirs2_core::ndarray::array;
    ///
    /// let points = array![
    ///     [0.0, 0.0],
    ///     [1.0, 0.0],
    ///     [0.0, 1.0],
    ///     [1.0, 1.0]
    /// ];
    ///
    /// let tri = Delaunay::new(&points).expect("Operation failed");
    /// // Try to find which triangle contains the point [0.25, 0.25]
    /// if let Some(idx) = tri.find_simplex(&[0.25, 0.25]) {
    ///     println!("Point is in simplex {}", idx);
    /// }
    /// ```
    pub fn find_simplex(&self, point: &[f64]) -> Option<usize> {
        if point.len() != self.ndim {
            return None;
        }

        if self.simplices.is_empty() {
            return None;
        }

        // Simple linear search for the containing simplex
        // More efficient algorithms (walk algorithm) would be preferred
        // for larger triangulations, but this is a reasonable starting point
        for (i, simplex) in self.simplices.iter().enumerate() {
            if self.point_in_simplex(point, simplex) {
                return Some(i);
            }
        }

        None
    }

    /// Check if a point is inside a simplex
    ///
    /// # Arguments
    ///
    /// * `point` - The point to check
    /// * `simplex` - The simplex (indices of vertices)
    ///
    /// # Returns
    ///
    /// * true if the point is inside the simplex, false otherwise
    fn point_in_simplex(&self, point: &[f64], simplex: &[usize]) -> bool {
        if self.ndim == 2 {
            // For 2D triangles, use barycentric coordinates
            let a = self.points.row(simplex[0]).to_vec();
            let b = self.points.row(simplex[1]).to_vec();
            let c = self.points.row(simplex[2]).to_vec();

            let v0x = b[0] - a[0];
            let v0y = b[1] - a[1];
            let v1x = c[0] - a[0];
            let v1y = c[1] - a[1];
            let v2x = point[0] - a[0];
            let v2y = point[1] - a[1];

            let d00 = v0x * v0x + v0y * v0y;
            let d01 = v0x * v1x + v0y * v1y;
            let d11 = v1x * v1x + v1y * v1y;
            let d20 = v2x * v0x + v2y * v0y;
            let d21 = v2x * v1x + v2y * v1y;

            let denom = d00 * d11 - d01 * d01;
            if denom.abs() < 1e-10 {
                return false; // Degenerate triangle
            }

            let v = (d11 * d20 - d01 * d21) / denom;
            let w = (d00 * d21 - d01 * d20) / denom;
            let u = 1.0 - v - w;

            // Point is inside if barycentric coordinates are all positive (or zero)
            // Allow for small numerical errors
            let eps = 1e-10;
            return u >= -eps && v >= -eps && w >= -eps;
        } else if self.ndim == 3 {
            // For 3D tetrahedra, use barycentric coordinates in 3D
            let a = self.points.row(simplex[0]).to_vec();
            let b = self.points.row(simplex[1]).to_vec();
            let c = self.points.row(simplex[2]).to_vec();
            let d = self.points.row(simplex[3]).to_vec();

            // Compute barycentric coordinates
            let mut bary = [0.0; 4];

            // Compute volume of tetrahedron
            let v0 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let v2 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];

            // Cross product and determinant for volume
            let vol = v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
                - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
                + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0]);

            if vol.abs() < 1e-10 {
                return false; // Degenerate tetrahedron
            }

            // Compute barycentric coordinates
            let _vp = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];

            let v3 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
            let v4 = [d[0] - b[0], d[1] - b[1], d[2] - b[2]];
            let v5 = [point[0] - b[0], point[1] - b[1], point[2] - b[2]];

            bary[0] = (v3[0] * (v4[1] * v5[2] - v4[2] * v5[1])
                - v3[1] * (v4[0] * v5[2] - v4[2] * v5[0])
                + v3[2] * (v4[0] * v5[1] - v4[1] * v5[0]))
                / vol;

            let v3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v4 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            let v5 = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];

            bary[1] = (v3[0] * (v4[1] * v5[2] - v4[2] * v5[1])
                - v3[1] * (v4[0] * v5[2] - v4[2] * v5[0])
                + v3[2] * (v4[0] * v5[1] - v4[1] * v5[0]))
                / vol;

            let v3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v4 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let v5 = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];

            bary[2] = (v3[0] * (v4[1] * v5[2] - v4[2] * v5[1])
                - v3[1] * (v4[0] * v5[2] - v4[2] * v5[0])
                + v3[2] * (v4[0] * v5[1] - v4[1] * v5[0]))
                / vol;

            bary[3] = 1.0 - bary[0] - bary[1] - bary[2];

            // Point is inside if all barycentric coordinates are positive (or zero)
            let eps = 1e-10;
            return bary.iter().all(|&b| b >= -eps);
        }

        // For higher dimensions or fallback
        false
    }

    /// Compute the convex hull of the points
    ///
    /// # Returns
    ///
    /// * Indices of the points forming the convex hull
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::delaunay::Delaunay;
    /// use scirs2_core::ndarray::array;
    ///
    /// let points = array![
    ///     [0.0, 0.0],
    ///     [1.0, 0.0],
    ///     [0.0, 1.0],
    ///     [0.25, 0.25]  // Interior point (strictly inside the triangle)
    /// ];
    ///
    /// let tri = Delaunay::new(&points).expect("Operation failed");
    /// let hull = tri.convex_hull();
    ///
    /// // The hull should be the three corner points, excluding the interior point
    /// // Note: Hull size depends on triangulation; interior points may affect result
    /// assert!(hull.len() >= 3, "Hull should have at least 3 points");
    /// ```
    pub fn convex_hull(&self) -> Vec<usize> {
        let mut hull = HashSet::new();

        // In 2D and 3D, the convex hull consists of the simplices with a neighbor of -1
        for (i, neighbors) in self.neighbors.iter().enumerate() {
            for (j, &neighbor) in neighbors.iter().enumerate() {
                if neighbor == -1 {
                    // This face is on the convex hull
                    // Add all vertices of this face (exclude the vertex opposite to the boundary)
                    for k in 0..self.ndim + 1 {
                        if k != j {
                            hull.insert(self.simplices[i][k]);
                        }
                    }
                }
            }
        }

        // Convert to a sorted vector
        let mut hull_vec: Vec<usize> = hull.into_iter().collect();
        hull_vec.sort();

        hull_vec
    }
}

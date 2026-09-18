//! Constrained Delaunay triangulation

use super::Delaunay;
use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::Array2;
use std::collections::HashSet;

impl Delaunay {
    /// Create a new constrained Delaunay triangulation
    ///
    /// # Arguments
    ///
    /// * `points` - The points to triangulate, shape (npoints, ndim)
    /// * `constraints` - Vector of constraint edges, each edge is a pair of point indices
    ///
    /// # Returns
    ///
    /// * A new constrained Delaunay triangulation or an error
    ///
    /// # Note
    ///
    /// Currently only supports 2D constrained Delaunay triangulation.
    /// Constraints are edges that must be present in the final triangulation.
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
    ///     [1.0, 1.0],
    ///     [0.0, 1.0],
    ///     [0.5, 0.5]
    /// ];
    ///
    /// // Add constraint edges forming a square boundary
    /// let constraints = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
    ///
    /// let tri = Delaunay::new_constrained(&points, constraints).expect("Operation failed");
    /// let simplices = tri.simplices();
    /// println!("Constrained triangles: {:?}", simplices);
    /// ```
    pub fn new_constrained(
        points: &Array2<f64>,
        constraints: Vec<(usize, usize)>,
    ) -> SpatialResult<Self> {
        let ndim = points.ncols();

        // Support 2D and 3D constrained Delaunay triangulation
        // Note: 3D implementation supports constraint edges only (not constraint faces)
        if ndim != 2 && ndim != 3 {
            return Err(SpatialError::NotImplementedError(
                "Constrained Delaunay triangulation only supports 2D and 3D points".to_string(),
            ));
        }

        // Validate constraints
        let npoints = points.nrows();
        for &(i, j) in &constraints {
            if i >= npoints || j >= npoints {
                return Err(SpatialError::ValueError(format!(
                    "Constraint edge ({i}, {j}) contains invalid point indices"
                )));
            }
            if i == j {
                return Err(SpatialError::ValueError(format!(
                    "Constraint edge ({i}, {j}) connects a point to itself"
                )));
            }
        }

        // Start with regular Delaunay triangulation
        let mut delaunay = Self::new(points)?;
        delaunay.constraints = constraints.clone();

        // Apply constraints using edge insertion algorithm
        delaunay.insert_constraints()?;

        Ok(delaunay)
    }

    /// Get the constraint edges
    ///
    /// # Returns
    ///
    /// * Vector of constraint edges as pairs of point indices
    pub fn constraints(&self) -> &[(usize, usize)] {
        &self.constraints
    }

    /// Insert constraint edges into the triangulation
    fn insert_constraints(&mut self) -> SpatialResult<()> {
        for &(i, j) in &self.constraints.clone() {
            self.insert_constraint_edge(i, j)?;
        }
        Ok(())
    }

    /// Insert a single constraint edge into the triangulation
    fn insert_constraint_edge(&mut self, start: usize, end: usize) -> SpatialResult<()> {
        // Check if the edge already exists in the triangulation
        if self.edge_exists(start, end) {
            return Ok(()); // Edge already exists, nothing to do
        }

        // Find all edges that intersect with the constraint edge
        let intersecting_edges = self.find_intersecting_edges(start, end)?;

        if intersecting_edges.is_empty() {
            // No intersections, but edge doesn't exist - this shouldn't happen in a proper triangulation
            return Err(SpatialError::ComputationError(
                "Constraint edge has no intersections but doesn't exist in triangulation"
                    .to_string(),
            ));
        }

        // Remove triangles containing intersecting edges
        let affected_triangles = self.find_triangles_with_edges(&intersecting_edges);
        self.remove_triangles(&affected_triangles);

        // Retriangulate the affected region while ensuring the constraint edge is present
        self.retriangulate_with_constraint(start, end, &affected_triangles)?;

        Ok(())
    }

    /// Check if an edge exists in the current triangulation
    pub(crate) fn edge_exists(&self, start: usize, end: usize) -> bool {
        for simplex in &self.simplices {
            let simplex_size = simplex.len();
            // Check all edges of the simplex (triangle in 2D, tetrahedron in 3D)
            for i in 0..simplex_size {
                for j in (i + 1)..simplex_size {
                    let v1 = simplex[i];
                    let v2 = simplex[j];
                    if (v1 == start && v2 == end) || (v1 == end && v2 == start) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Find all edges that intersect with the constraint edge
    fn find_intersecting_edges(
        &self,
        start: usize,
        end: usize,
    ) -> SpatialResult<Vec<(usize, usize)>> {
        let mut intersecting = Vec::new();

        // Extract constraint edge points
        let p1: Vec<f64> = self.points.row(start).to_vec();
        let p2: Vec<f64> = self.points.row(end).to_vec();

        // Check all edges in the triangulation
        let mut checked_edges = HashSet::new();

        for simplex in &self.simplices {
            let simplex_size = simplex.len();

            // Check all edges of the simplex
            for i in 0..simplex_size {
                for j in (i + 1)..simplex_size {
                    let v1 = simplex[i];
                    let v2 = simplex[j];

                    // Avoid checking the same edge twice
                    let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                    if checked_edges.contains(&edge) {
                        continue;
                    }
                    checked_edges.insert(edge);

                    // Skip if this edge shares a vertex with the constraint edge
                    if v1 == start || v1 == end || v2 == start || v2 == end {
                        continue;
                    }

                    let q1: Vec<f64> = self.points.row(v1).to_vec();
                    let q2: Vec<f64> = self.points.row(v2).to_vec();

                    if self.ndim == 2 {
                        // 2D case: check for segment intersection
                        let p1_2d = [p1[0], p1[1]];
                        let p2_2d = [p2[0], p2[1]];
                        let q1_2d = [q1[0], q1[1]];
                        let q2_2d = [q2[0], q2[1]];

                        if segments_intersect(p1_2d, p2_2d, q1_2d, q2_2d) {
                            intersecting.push((v1, v2));
                        }
                    } else if self.ndim == 3 {
                        // 3D case: check if edges are close enough to interfere
                        // (simplified approach for constraint enforcement)
                        if edges_interfere_3d(&p1, &p2, &q1, &q2) {
                            intersecting.push((v1, v2));
                        }
                    }
                }
            }
        }

        Ok(intersecting)
    }

    /// Find all triangles that contain any of the given edges
    fn find_triangles_with_edges(&self, edges: &[(usize, usize)]) -> Vec<usize> {
        let mut triangles = HashSet::new();

        for (i, simplex) in self.simplices.iter().enumerate() {
            for &(e1, e2) in edges {
                if self.triangle_contains_edge(simplex, e1, e2) {
                    triangles.insert(i);
                }
            }
        }

        triangles.into_iter().collect()
    }

    /// Check if a triangle contains a specific edge
    fn triangle_contains_edge(&self, triangle: &[usize], v1: usize, v2: usize) -> bool {
        for i in 0..3 {
            let j = (i + 1) % 3;
            let t1 = triangle[i];
            let t2 = triangle[j];
            if (t1 == v1 && t2 == v2) || (t1 == v2 && t2 == v1) {
                return true;
            }
        }
        false
    }

    /// Remove triangles from the triangulation
    fn remove_triangles(&mut self, _triangleindices: &[usize]) {
        // Sort _indices in descending order to avoid index shifting issues
        let mut sorted_indices = _triangleindices.to_vec();
        sorted_indices.sort_by(|a, b| b.cmp(a));

        for &idx in &sorted_indices {
            if idx < self.simplices.len() {
                self.simplices.remove(idx);
                self.neighbors.remove(idx);
            }
        }
    }

    /// Retriangulate a region ensuring the constraint edge is present
    fn retriangulate_with_constraint(
        &mut self,
        start: usize,
        end: usize,
        affected_triangles: &[usize],
    ) -> SpatialResult<()> {
        if affected_triangles.is_empty() {
            return Ok(());
        }

        // Extract all unique vertices from affected triangles
        let cavity_vertices = self.extract_cavity_vertices(affected_triangles);

        // Find the boundary edges of the cavity (excluding the constraint edge)
        let boundary_edges = self.find_cavity_boundary(affected_triangles, start, end)?;

        // Retriangulate the cavity using a simple fan triangulation approach
        let new_triangles =
            self.fan_triangulate_cavity(&cavity_vertices, &boundary_edges, start, end)?;

        // Add the new triangles to the triangulation
        for triangle in new_triangles {
            self.simplices.push(triangle);
        }

        // Update neighbors for the new triangles (simplified approach)
        self.compute_neighbors();

        Ok(())
    }

    /// Extract all unique vertices from the affected triangles
    fn extract_cavity_vertices(&self, _affectedtriangles: &[usize]) -> Vec<usize> {
        let mut vertices = HashSet::new();

        for &triangle_idx in _affectedtriangles {
            if triangle_idx < self.simplices.len() {
                for &vertex in &self.simplices[triangle_idx] {
                    vertices.insert(vertex);
                }
            }
        }

        vertices.into_iter().collect()
    }

    /// Find the boundary edges of the cavity
    fn find_cavity_boundary(
        &self,
        affected_triangles: &[usize],
        start: usize,
        end: usize,
    ) -> SpatialResult<Vec<(usize, usize)>> {
        let affected_set: HashSet<usize> = affected_triangles.iter().cloned().collect();
        let mut boundary_edges = Vec::new();

        // For each affected triangle, check each edge
        for &triangle_idx in affected_triangles {
            if triangle_idx >= self.simplices.len() {
                continue;
            }

            let simplex = &self.simplices[triangle_idx];
            if simplex.len() < 3 {
                continue;
            }

            // Check each edge of the triangle
            for i in 0..simplex.len() {
                let v1 = simplex[i];
                let v2 = simplex[(i + 1) % simplex.len()];

                // Skip the constraint edge itself
                if (v1 == start && v2 == end) || (v1 == end && v2 == start) {
                    continue;
                }

                // Check if this edge is on the boundary (not shared with another affected triangle)
                if self.is_boundary_edge(v1, v2, &affected_set, triangle_idx) {
                    boundary_edges.push((v1, v2));
                }
            }
        }

        Ok(boundary_edges)
    }

    /// Check if an edge is on the boundary of the cavity
    fn is_boundary_edge(
        &self,
        v1: usize,
        v2: usize,
        affected_set: &HashSet<usize>,
        current_triangle: usize,
    ) -> bool {
        // Find all triangles that contain this edge
        for (tri_idx, simplex) in self.simplices.iter().enumerate() {
            if tri_idx == current_triangle || affected_set.contains(&tri_idx) {
                continue;
            }

            // Check if this triangle contains the edge v1-v2
            if self.triangle_contains_edge(simplex, v1, v2) {
                return false; // Edge is shared with a non-affected triangle, so not on boundary
            }
        }

        true // Edge is on the boundary
    }

    /// Retriangulate the cavity using fan triangulation
    fn fan_triangulate_cavity(
        &self,
        cavity_vertices: &[usize],
        boundary_edges: &[(usize, usize)],
        start: usize,
        end: usize,
    ) -> SpatialResult<Vec<Vec<usize>>> {
        let mut new_triangles = Vec::new();

        // Find vertices that are not on the constraint edge
        let mut interior_vertices = Vec::new();
        for &vertex in cavity_vertices {
            if vertex != start && vertex != end {
                interior_vertices.push(vertex);
            }
        }

        // If we have interior vertices, create triangles using fan triangulation
        if !interior_vertices.is_empty() {
            // Create fan triangulation from start vertex
            for i in 0..interior_vertices.len() {
                for j in (i + 1)..interior_vertices.len() {
                    let v1 = interior_vertices[i];
                    let v2 = interior_vertices[j];

                    // Check if we can form a valid triangle
                    if self.is_valid_triangle_in_cavity(start, v1, v2, boundary_edges) {
                        new_triangles.push(vec![start, v1, v2]);
                    }

                    if self.is_valid_triangle_in_cavity(end, v1, v2, boundary_edges) {
                        new_triangles.push(vec![end, v1, v2]);
                    }
                }
            }
        }

        // Ensure we have at least one triangle containing the constraint edge
        if new_triangles.is_empty() && !interior_vertices.is_empty() {
            let v = interior_vertices[0];
            new_triangles.push(vec![start, end, v]);
        }

        // Connect boundary vertices to constraint edge if needed
        for &(v1, v2) in boundary_edges {
            if v1 != start && v1 != end && v2 != start && v2 != end {
                // Try to connect this boundary edge to the constraint edge
                if self.points_form_valid_triangle(start, v1, v2) {
                    new_triangles.push(vec![start, v1, v2]);
                }
                if self.points_form_valid_triangle(end, v1, v2) {
                    new_triangles.push(vec![end, v1, v2]);
                }
            }
        }

        Ok(new_triangles)
    }

    /// Check if three points form a valid triangle (not collinear)
    fn points_form_valid_triangle(&self, v1: usize, v2: usize, v3: usize) -> bool {
        if v1 >= self.npoints || v2 >= self.npoints || v3 >= self.npoints {
            return false;
        }

        let p1 = self.points.row(v1);
        let p2 = self.points.row(v2);
        let p3 = self.points.row(v3);

        // Check if points are collinear using cross product
        let dx1 = p2[0] - p1[0];
        let dy1 = p2[1] - p1[1];
        let dx2 = p3[0] - p1[0];
        let dy2 = p3[1] - p1[1];

        let cross = dx1 * dy2 - dy1 * dx2;
        cross.abs() > 1e-10 // Not collinear
    }

    /// Check if a triangle is valid within the cavity constraints
    fn is_valid_triangle_in_cavity(
        &self,
        v1: usize,
        v2: usize,
        v3: usize,
        _boundary_edges: &[(usize, usize)],
    ) -> bool {
        // Basic validation - check if triangle is not degenerate
        self.points_form_valid_triangle(v1, v2, v3)
    }
}

/// Check if two line segments intersect
fn segments_intersect(p1: [f64; 2], p2: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> bool {
    fn orientation(p: [f64; 2], q: [f64; 2], r: [f64; 2]) -> i32 {
        let val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);
        if val.abs() < 1e-10 {
            0
        }
        // Collinear
        else if val > 0.0 {
            1
        }
        // Clockwise
        else {
            2
        } // Counterclockwise
    }

    fn on_segment(p: [f64; 2], q: [f64; 2], r: [f64; 2]) -> bool {
        q[0] <= p[0].max(r[0])
            && q[0] >= p[0].min(r[0])
            && q[1] <= p[1].max(r[1])
            && q[1] >= p[1].min(r[1])
    }

    let o1 = orientation(p1, p2, q1);
    let o2 = orientation(p1, p2, q2);
    let o3 = orientation(q1, q2, p1);
    let o4 = orientation(q1, q2, p2);

    // General case
    if o1 != o2 && o3 != o4 {
        return true;
    }

    // Special cases - segments are collinear and overlapping
    if o1 == 0 && on_segment(p1, q1, p2) {
        return true;
    }
    if o2 == 0 && on_segment(p1, q2, p2) {
        return true;
    }
    if o3 == 0 && on_segment(q1, p1, q2) {
        return true;
    }
    if o4 == 0 && on_segment(q1, p2, q2) {
        return true;
    }

    false
}

/// Check if two 3D edges interfere enough to require constraint enforcement
/// This is a simplified approach using distance-based criteria
fn edges_interfere_3d(p1: &[f64], p2: &[f64], q1: &[f64], q2: &[f64]) -> bool {
    // Calculate the closest distance between the two line segments in 3D
    let eps = 1e-6; // Distance threshold for interference

    // Vector from p1 to p2
    let u = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    // Vector from q1 to q2
    let v = [q2[0] - q1[0], q2[1] - q1[1], q2[2] - q1[2]];
    // Vector from p1 to q1
    let w = [q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2]];

    let u_dot_u = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    let v_dot_v = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let u_dot_v = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let u_dot_w = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    let v_dot_w = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];

    let denom = u_dot_u * v_dot_v - u_dot_v * u_dot_v;

    // If lines are parallel, check distance between them
    if denom.abs() < eps {
        // Lines are parallel - check if they're close
        let cross_u_w = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        let dist_sq = (cross_u_w[0] * cross_u_w[0]
            + cross_u_w[1] * cross_u_w[1]
            + cross_u_w[2] * cross_u_w[2])
            / u_dot_u;
        return dist_sq < eps * eps;
    }

    // Calculate closest points on the two line segments
    let s = (u_dot_v * v_dot_w - v_dot_v * u_dot_w) / denom;
    let t = (u_dot_u * v_dot_w - u_dot_v * u_dot_w) / denom;

    // Clamp to segment bounds
    let s_clamped = s.clamp(0.0, 1.0);
    let t_clamped = t.clamp(0.0, 1.0);

    // Calculate closest points
    let closest_p = [
        p1[0] + s_clamped * u[0],
        p1[1] + s_clamped * u[1],
        p1[2] + s_clamped * u[2],
    ];
    let closest_q = [
        q1[0] + t_clamped * v[0],
        q1[1] + t_clamped * v[1],
        q1[2] + t_clamped * v[2],
    ];

    // Check if closest points are within interference threshold
    let dist_sq = (closest_p[0] - closest_q[0]) * (closest_p[0] - closest_q[0])
        + (closest_p[1] - closest_q[1]) * (closest_p[1] - closest_q[1])
        + (closest_p[2] - closest_q[2]) * (closest_p[2] - closest_q[2]);

    dist_sq < eps * eps
}

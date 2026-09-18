//! Delaunay triangulation algorithms (Pure Rust)
//!
//! This module provides implementations for Delaunay triangulation of points in arbitrary dimensions.
//! Delaunay triangulation connects points to form simplices such that no point is inside the
//! circumhypersphere of any simplex.
//!
//! # Implementation
//!
//! This module uses the Bowyer-Watson algorithm for computing Delaunay triangulations in 2D, 3D,
//! and higher dimensions (nD). This is a pure Rust implementation with no external C library dependencies.
//!
//! **Supported Dimensions**: 2D, 3D, 4D, 5D, ..., nD
//! **Algorithm**: Bowyer-Watson incremental insertion with lifting map for in-hypersphere tests
//! **Complexity**: O(n^⌈d/2⌉) where n = points, d = dimensions
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::delaunay::Delaunay;
//! use scirs2_core::ndarray::array;
//!
//! // Create a set of 2D points
//! let points = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [0.0, 1.0],
//!     [0.5, 0.5]
//! ];
//!
//! // Compute Delaunay triangulation
//! let tri = Delaunay::new(&points).expect("Operation failed");
//!
//! // Get the simplex (triangle) indices
//! let simplices = tri.simplices();
//! println!("Triangles: {:?}", simplices);
//!
//! // Find the triangle containing a point
//! let point = [0.25, 0.25];
//! if let Some(idx) = tri.find_simplex(&point) {
//!     println!("Point {:?} is in triangle {}", point, idx);
//! }
//! ```

mod bowyer_watson_2d;
mod bowyer_watson_3d;
mod bowyer_watson_nd;
mod constrained;
mod queries;

#[cfg(test)]
mod tests;

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use std::fmt::Debug;

pub use constrained::*;
pub use queries::*;

/// Structure for storing and querying a Delaunay triangulation
///
/// The Delaunay triangulation of a set of points is a triangulation such that
/// no point is inside the circumhypersphere of any simplex. This property holds
/// in all dimensions:
/// - 2D: No point inside circumcircle of any triangle
/// - 3D: No point inside circumsphere of any tetrahedron
/// - nD: No point inside circumhypersphere of any n-simplex
///
/// This is a pure Rust implementation using the Bowyer-Watson algorithm with no
/// external C library dependencies. Supports arbitrary dimensions (2D, 3D, 4D, 5D, ..., nD).
pub struct Delaunay {
    /// The points used for the triangulation
    pub(crate) points: Array2<f64>,

    /// The number of dimensions
    pub(crate) ndim: usize,

    /// The number of points
    pub(crate) npoints: usize,

    /// The simplices (triangles in 2D, tetrahedra in 3D, etc.)
    /// Each element is a vector of indices of the vertices forming a simplex
    pub(crate) simplices: Vec<Vec<usize>>,

    /// For each simplex, its neighboring simplices
    /// neighbors[i][j] is the index of the simplex that shares a face with simplex i,
    /// opposite to the vertex j of simplex i. -1 indicates no neighbor.
    pub(crate) neighbors: Vec<Vec<i64>>,

    /// Constraint edges (for constrained Delaunay triangulation)
    /// Each edge is represented as a pair of point indices
    pub(crate) constraints: Vec<(usize, usize)>,
}

impl Debug for Delaunay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Delaunay")
            .field("points", &self.points.shape())
            .field("ndim", &self.ndim)
            .field("npoints", &self.npoints)
            .field("simplices", &self.simplices.len())
            .field("neighbors", &self.neighbors.len())
            .field("constraints", &self.constraints.len())
            .finish()
    }
}

impl Clone for Delaunay {
    fn clone(&self) -> Self {
        Self {
            points: self.points.clone(),
            ndim: self.ndim,
            npoints: self.npoints,
            simplices: self.simplices.clone(),
            neighbors: self.neighbors.clone(),
            constraints: self.constraints.clone(),
        }
    }
}

impl Delaunay {
    /// Create a new Delaunay triangulation
    ///
    /// # Arguments
    ///
    /// * `points` - The points to triangulate, shape (npoints, ndim)
    ///
    /// # Returns
    ///
    /// * A new Delaunay triangulation or an error
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
    /// let simplices = tri.simplices();
    /// println!("Triangles: {:?}", simplices);
    /// ```
    pub fn new(points: &Array2<f64>) -> SpatialResult<Self> {
        let npoints = points.nrows();
        let ndim = points.ncols();

        // Check if we have enough points for triangulation
        if npoints <= ndim {
            return Err(SpatialError::ValueError(format!(
                "Need at least {} points in {} dimensions for triangulation",
                ndim + 1,
                ndim
            )));
        }

        // Special case for 3 points in 2D - form a single triangle
        if ndim == 2 && npoints == 3 {
            let simplex = vec![0, 1, 2];
            let simplices = vec![simplex];
            let neighbors = vec![vec![-1, -1, -1]]; // No neighbors

            return Ok(Delaunay {
                points: points.clone(),
                ndim,
                npoints,
                simplices,
                neighbors,
                constraints: Vec::new(),
            });
        }

        // Use Bowyer-Watson algorithm for 2D, 3D, and nD
        let simplices = match ndim {
            2 => bowyer_watson_2d::bowyer_watson_2d(points)?,
            3 => bowyer_watson_3d::bowyer_watson_3d(points)?,
            _ => bowyer_watson_nd::bowyer_watson_nd(points, ndim)?,
        };

        // Calculate neighbors of each simplex
        let neighbors = Self::calculate_neighbors(&simplices, ndim + 1);

        Ok(Delaunay {
            points: points.clone(),
            ndim,
            npoints,
            simplices,
            neighbors,
            constraints: Vec::new(),
        })
    }

    /// Calculate neighbors of each simplex
    ///
    /// # Arguments
    ///
    /// * `simplices` - The list of simplices
    /// * `n` - Number of vertices in a simplex
    ///
    /// # Returns
    ///
    /// * Vector of neighbor indices for each simplex
    pub(crate) fn calculate_neighbors(simplices: &[Vec<usize>], n: usize) -> Vec<Vec<i64>> {
        let nsimplex = simplices.len();
        let mut neighbors = vec![vec![-1; n]; nsimplex];

        // Build a map from (n-1)-faces to simplices
        // A face is represented as a sorted vector of vertex indices
        let mut face_to_simplex: HashMap<Vec<usize>, Vec<(usize, usize)>> = HashMap::new();

        for (i, simplex) in simplices.iter().enumerate() {
            for j in 0..n {
                // Create a face by excluding vertex j
                let mut face: Vec<usize> = simplex
                    .iter()
                    .enumerate()
                    .filter(|&(k_, _)| k_ != j)
                    .map(|(_, &v)| v)
                    .collect();

                // Sort the face for consistent hashing
                face.sort();

                // Add (simplex_index, excluded_vertex) to the map
                face_to_simplex.entry(face).or_default().push((i, j));
            }
        }

        // For each face shared by two simplices, update the neighbor information
        for (_, simplex_info) in face_to_simplex.iter() {
            if simplex_info.len() == 2 {
                let (i1, j1) = simplex_info[0];
                let (i2, j2) = simplex_info[1];

                neighbors[i1][j1] = i2 as i64;
                neighbors[i2][j2] = i1 as i64;
            }
        }

        neighbors
    }

    /// Get the number of points
    ///
    /// # Returns
    ///
    /// * Number of points in the triangulation
    pub fn npoints(&self) -> usize {
        self.npoints
    }

    /// Get the dimension of the points
    ///
    /// # Returns
    ///
    /// * Number of dimensions of the points
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Get the points used for triangulation
    ///
    /// # Returns
    ///
    /// * Array of points
    pub fn points(&self) -> &Array2<f64> {
        &self.points
    }

    /// Get the simplices (triangles in 2D, tetrahedra in 3D, etc.)
    ///
    /// # Returns
    ///
    /// * Vector of simplices, where each simplex is a vector of vertex indices
    pub fn simplices(&self) -> &[Vec<usize>] {
        &self.simplices
    }

    /// Get the neighbors of each simplex
    ///
    /// # Returns
    ///
    /// * Vector of neighbor indices for each simplex
    pub fn neighbors(&self) -> &[Vec<i64>] {
        &self.neighbors
    }

    /// Recompute neighbors for all simplices
    pub(crate) fn compute_neighbors(&mut self) {
        self.neighbors = Self::calculate_neighbors(&self.simplices, self.ndim + 1);
    }
}

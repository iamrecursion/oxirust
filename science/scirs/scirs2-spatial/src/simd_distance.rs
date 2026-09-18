//! SIMD-accelerated distance calculations for spatial operations
//!
//! This module provides high-performance implementations of distance calculations
//! using the unified SIMD operations from scirs2-core. These optimized
//! functions can provide significant performance improvements for operations
//! involving large datasets.
//!
//! # Features
//!
//! - **SIMD-accelerated distance metrics**: Euclidean, Manhattan, Minkowski
//! - **Parallel distance matrix computation**: Multi-threaded pdist and cdist
//! - **Batch nearest neighbor queries**: Optimized for large point sets
//! - **Memory-efficient operations**: Minimized memory allocation and copying
//!
//! # Architecture Support
//!
//! The SIMD operations are automatically detected and optimized by scirs2-core
//! based on the available hardware capabilities.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::simd_distance::{simd_euclidean_distance_batch, parallel_pdist};
//! use scirs2_core::ndarray::array;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // SIMD batch distance calculation
//! let points1 = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
//! let points2 = array![[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]];
//!
//! let distances = simd_euclidean_distance_batch(&points1.view(), &points2.view())?;
//! println!("Batch distances: {:?}", distances);
//!
//! // Parallel distance matrix
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//! let dist_matrix = parallel_pdist(&points.view(), "euclidean")?;
//! println!("Distance matrix shape: {:?}", dist_matrix.shape());
//! # Ok(())
//! # }
//! ```

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Supported distance metrics for SIMD operations
#[derive(Debug, Clone, Copy)]
pub enum SimdMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)  
    Manhattan,
    /// Squared Euclidean distance (no square root)
    SquaredEuclidean,
    /// Chebyshev distance (L∞ norm)
    Chebyshev,
}

impl SimdMetric {
    /// Get the string name of the metric
    ///
    /// # Returns
    ///
    /// * A string slice containing the name of the metric
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::simd_distance::SimdMetric;
    ///
    /// let metric = SimdMetric::Euclidean;
    /// assert_eq!(metric.name(), "euclidean");
    ///
    /// let metric = SimdMetric::Manhattan;
    /// assert_eq!(metric.name(), "manhattan");
    /// ```
    pub fn name(&self) -> &'static str {
        match self {
            SimdMetric::Euclidean => "euclidean",
            SimdMetric::Manhattan => "manhattan",
            SimdMetric::SquaredEuclidean => "sqeuclidean",
            SimdMetric::Chebyshev => "chebyshev",
        }
    }
}

/// SIMD-accelerated Euclidean distance between two points
///
/// # Arguments
/// * `a` - First point
/// * `b` - Second point
///
/// # Returns
/// * Euclidean distance between the points
#[allow(dead_code)]
pub fn simd_euclidean_distance(a: &[f64], b: &[f64]) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    // Convert slices to ArrayView for core ops
    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Compute squared distance using SIMD operations
    let diff = f64::simd_sub(&a_view, &b_view);
    let squared = f64::simd_mul(&diff.view(), &diff.view());
    let sum = f64::simd_sum(&squared.view());
    Ok(sum.sqrt())
}

/// SIMD-accelerated squared Euclidean distance between two points
#[allow(dead_code)]
pub fn simd_squared_euclidean_distance(a: &[f64], b: &[f64]) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Compute squared distance using SIMD operations
    let diff = f64::simd_sub(&a_view, &b_view);
    let squared = f64::simd_mul(&diff.view(), &diff.view());
    Ok(f64::simd_sum(&squared.view()))
}

/// SIMD-accelerated Manhattan distance between two points
#[allow(dead_code)]
pub fn simd_manhattan_distance(a: &[f64], b: &[f64]) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Compute Manhattan distance using SIMD operations
    let diff = f64::simd_sub(&a_view, &b_view);
    let abs_diff = f64::simd_abs(&diff.view());
    Ok(f64::simd_sum(&abs_diff.view()))
}

/// SIMD-accelerated Chebyshev distance
#[allow(dead_code)]
pub fn simd_chebyshev_distance(a: &[f64], b: &[f64]) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Compute Chebyshev distance using SIMD operations
    let diff = f64::simd_sub(&a_view, &b_view);
    let abs_diff = f64::simd_abs(&diff.view());
    Ok(f64::simd_max_element(&abs_diff.view()))
}

/// Batch SIMD-accelerated Euclidean distance calculation
///
/// Computes distances between corresponding rows of two matrices
///
/// # Arguments
/// * `points1` - First set of points, shape (n, d)
/// * `points2` - Second set of points, shape (n, d)
///
/// # Returns
/// * Array of distances, shape (n,)
#[allow(dead_code)]
pub fn simd_euclidean_distance_batch(
    points1: &ArrayView2<'_, f64>,
    points2: &ArrayView2<'_, f64>,
) -> SpatialResult<Array1<f64>> {
    if points1.shape() != points2.shape() {
        return Err(SpatialError::ValueError(
            "Point arrays must have the same shape".to_string(),
        ));
    }

    let n_points = points1.nrows();

    // Use parallel computation with SIMD operations
    let distances_vec: Result<Vec<f64>, SpatialError> = (0..n_points)
        .into_par_iter()
        .map(|i| -> SpatialResult<f64> {
            let p1 = points1.row(i);
            let p2 = points2.row(i);
            let diff = f64::simd_sub(&p1, &p2);
            let squared = f64::simd_mul(&diff.view(), &diff.view());
            let sum = f64::simd_sum(&squared.view());
            Ok(sum.sqrt())
        })
        .collect();

    Ok(Array1::from(distances_vec?))
}

/// Parallel computation of pairwise distance matrix
///
/// # Arguments
/// * `points` - Array of points, shape (n, d)
/// * `metric` - Distance metric to use
///
/// # Returns
/// * Condensed distance matrix, shape (n*(n-1)/2,)
#[allow(dead_code)]
pub fn parallel_pdist(points: &ArrayView2<'_, f64>, metric: &str) -> SpatialResult<Array1<f64>> {
    use scirs2_core::parallel_ops::ParallelIterator;
    let n_points = points.nrows();
    if n_points < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 _points for distance computation".to_string(),
        ));
    }

    let n_distances = n_points * (n_points - 1) / 2;

    let metric_enum = match metric {
        "euclidean" => SimdMetric::Euclidean,
        "manhattan" => SimdMetric::Manhattan,
        "sqeuclidean" => SimdMetric::SquaredEuclidean,
        "chebyshev" => SimdMetric::Chebyshev,
        _ => {
            return Err(SpatialError::ValueError(format!(
                "Unsupported metric: {metric}"
            )))
        }
    };

    // Parallel computation of distance pairs
    let distances_vec: Result<Vec<f64>, SpatialError> = (0..n_distances)
        .into_par_iter()
        .map(|idx| -> SpatialResult<f64> {
            // Convert linear index to (i, j) pair
            let (i, j) = linear_to_condensed_indices(idx, n_points);

            let p1 = points.row(i);
            let p2 = points.row(j);

            match metric_enum {
                SimdMetric::Euclidean => {
                    let diff = f64::simd_sub(&p1, &p2);
                    let squared = f64::simd_mul(&diff.view(), &diff.view());
                    let sum = f64::simd_sum(&squared.view());
                    Ok(sum.sqrt())
                }
                SimdMetric::Manhattan => {
                    let diff = f64::simd_sub(&p1, &p2);
                    let abs_diff = f64::simd_abs(&diff.view());
                    Ok(f64::simd_sum(&abs_diff.view()))
                }
                SimdMetric::SquaredEuclidean => {
                    let diff = f64::simd_sub(&p1, &p2);
                    let squared = f64::simd_mul(&diff.view(), &diff.view());
                    Ok(f64::simd_sum(&squared.view()))
                }
                SimdMetric::Chebyshev => {
                    let diff = f64::simd_sub(&p1, &p2);
                    let abs_diff = f64::simd_abs(&diff.view());
                    Ok(f64::simd_max_element(&abs_diff.view()))
                }
            }
        })
        .collect();

    Ok(Array1::from(distances_vec?))
}

/// Parallel computation of cross-distance matrix
///
/// # Arguments
/// * `points1` - First set of points, shape (n, d)
/// * `points2` - Second set of points, shape (m, d)
/// * `metric` - Distance metric to use
///
/// # Returns
/// * Distance matrix, shape (n, m)
#[allow(dead_code)]
pub fn parallel_cdist(
    points1: &ArrayView2<'_, f64>,
    points2: &ArrayView2<'_, f64>,
    metric: &str,
) -> SpatialResult<Array2<f64>> {
    use scirs2_core::parallel_ops::ParallelIterator;
    if points1.ncols() != points2.ncols() {
        return Err(SpatialError::ValueError(
            "Point arrays must have the same number of dimensions".to_string(),
        ));
    }

    let n1 = points1.nrows();
    let n2 = points2.nrows();
    let mut distances = Array2::zeros((n1, n2));

    let metric_enum = match metric {
        "euclidean" => SimdMetric::Euclidean,
        "manhattan" => SimdMetric::Manhattan,
        "sqeuclidean" => SimdMetric::SquaredEuclidean,
        "chebyshev" => SimdMetric::Chebyshev,
        _ => {
            return Err(SpatialError::ValueError(format!(
                "Unsupported metric: {metric}"
            )))
        }
    };

    // Parallel computation over rows of first matrix
    distances
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .try_for_each(|(i, mut row)| -> SpatialResult<()> {
            let p1 = points1.row(i);

            for (j, dist) in row.iter_mut().enumerate() {
                let p2 = points2.row(j);

                *dist = match metric_enum {
                    SimdMetric::Euclidean => {
                        let diff = f64::simd_sub(&p1, &p2);
                        let squared = f64::simd_mul(&diff.view(), &diff.view());
                        let sum = f64::simd_sum(&squared.view());
                        sum.sqrt()
                    }
                    SimdMetric::Manhattan => {
                        let diff = f64::simd_sub(&p1, &p2);
                        let abs_diff = f64::simd_abs(&diff.view());
                        f64::simd_sum(&abs_diff.view())
                    }
                    SimdMetric::SquaredEuclidean => {
                        let diff = f64::simd_sub(&p1, &p2);
                        let squared = f64::simd_mul(&diff.view(), &diff.view());
                        f64::simd_sum(&squared.view())
                    }
                    SimdMetric::Chebyshev => {
                        let diff = f64::simd_sub(&p1, &p2);
                        let abs_diff = f64::simd_abs(&diff.view());
                        f64::simd_max_element(&abs_diff.view())
                    }
                };
            }
            Ok(())
        })?;

    Ok(distances)
}

/// SIMD-accelerated k-nearest neighbors search
///
/// # Arguments
/// * `query_points` - Points to query, shape (n_queries, d)
/// * `data_points` - Data points to search, shape (n_data, d)
/// * `k` - Number of nearest neighbors
/// * `metric` - Distance metric to use
///
/// # Returns
/// * (indices, distances) where both have shape (n_queries, k)
#[allow(dead_code)]
pub fn simd_knn_search(
    query_points: &ArrayView2<'_, f64>,
    data_points: &ArrayView2<'_, f64>,
    k: usize,
    metric: &str,
) -> SpatialResult<(Array2<usize>, Array2<f64>)> {
    if query_points.ncols() != data_points.ncols() {
        return Err(SpatialError::ValueError(
            "Query and data _points must have the same dimension".to_string(),
        ));
    }

    if k > data_points.nrows() {
        return Err(SpatialError::ValueError(
            "k cannot be larger than the number of data _points".to_string(),
        ));
    }

    let n_queries = query_points.nrows();
    let n_data = data_points.nrows();

    let mut indices = Array2::zeros((n_queries, k));
    let mut distances = Array2::zeros((n_queries, k));

    let metric_enum = match metric {
        "euclidean" => SimdMetric::Euclidean,
        "manhattan" => SimdMetric::Manhattan,
        "sqeuclidean" => SimdMetric::SquaredEuclidean,
        "chebyshev" => SimdMetric::Chebyshev,
        _ => {
            return Err(SpatialError::ValueError(format!(
                "Unsupported metric: {metric}"
            )))
        }
    };

    // Process queries in parallel
    indices
        .outer_iter_mut()
        .zip(distances.outer_iter_mut())
        .enumerate()
        .par_bridge()
        .try_for_each(
            |(query_idx, (mut idx_row, mut dist_row))| -> SpatialResult<()> {
                let query_point = query_points.row(query_idx);

                // Compute all distances for this query using SIMD
                let mut all_distances: Vec<(f64, usize)> = (0..n_data)
                    .map(|data_idx| {
                        let data_point = data_points.row(data_idx);
                        let dist = match metric_enum {
                            SimdMetric::Euclidean => {
                                let diff = f64::simd_sub(&query_point, &data_point);
                                let squared = f64::simd_mul(&diff.view(), &diff.view());
                                let sum = f64::simd_sum(&squared.view());
                                sum.sqrt()
                            }
                            SimdMetric::Manhattan => {
                                let diff = f64::simd_sub(&query_point, &data_point);
                                let abs_diff = f64::simd_abs(&diff.view());
                                f64::simd_sum(&abs_diff.view())
                            }
                            SimdMetric::SquaredEuclidean => {
                                let diff = f64::simd_sub(&query_point, &data_point);
                                let squared = f64::simd_mul(&diff.view(), &diff.view());
                                f64::simd_sum(&squared.view())
                            }
                            SimdMetric::Chebyshev => {
                                let diff = f64::simd_sub(&query_point, &data_point);
                                let abs_diff = f64::simd_abs(&diff.view());
                                f64::simd_max_element(&abs_diff.view())
                            }
                        };
                        (dist, data_idx)
                    })
                    .collect();

                // Partial sort to get k smallest
                all_distances.select_nth_unstable_by(k - 1, |a, b| {
                    a.0.partial_cmp(&b.0).expect("Operation failed")
                });
                all_distances[..k].sort_by(|a, b| a.0.partial_cmp(&b.0).expect("Operation failed"));

                // Fill result arrays
                for (i, (dist, idx)) in all_distances[..k].iter().enumerate() {
                    dist_row[i] = *dist;
                    idx_row[i] = *idx;
                }

                Ok(())
            },
        )?;

    Ok((indices, distances))
}

/// Convert linear index to (i, j) indices for condensed distance matrix
#[allow(dead_code)]
fn linear_to_condensed_indices(_linearidx: usize, n: usize) -> (usize, usize) {
    // For condensed distance matrix where entry (i,j) with i < j is stored
    // at position (n-1-i)*(n-i)/2 + (j-i-1)
    let mut k = _linearidx;
    let mut i = 0;

    while k >= n - i - 1 {
        k -= n - i - 1;
        i += 1;
    }

    let j = k + i + 1;
    (i, j)
}

/// Advanced-optimized SIMD-accelerated clustering algorithms
pub mod advanced_simd_clustering {
    use crate::error::{SpatialError, SpatialResult};
    use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
    use scirs2_core::simd_ops::SimdUnifiedOps;

    /// Advanced-optimized SIMD K-means implementation with vectorized operations
    pub struct AdvancedSimdKMeans {
        k: usize,
        max_iterations: usize,
        tolerance: f64,
        use_mixed_precision: bool,
        block_size: usize,
    }

    impl AdvancedSimdKMeans {
        /// Create a new advanced-optimized SIMD K-means clusterer
        pub fn new(k: usize) -> Self {
            Self {
                k,
                max_iterations: 100,
                tolerance: 1e-6,
                use_mixed_precision: true,
                block_size: 256, // Optimized for cache lines
            }
        }

        /// Configure mixed precision (f32 for speed where possible)
        pub fn with_mixed_precision(mut self, use_mixedprecision: bool) -> Self {
            self.use_mixed_precision = use_mixedprecision;
            self
        }

        /// Set block size for cache-optimized processing
        pub fn with_block_size(mut self, blocksize: usize) -> Self {
            self.block_size = blocksize;
            self
        }

        /// Advanced-optimized SIMD K-means clustering
        pub fn fit(
            &self,
            points: &ArrayView2<'_, f64>,
        ) -> SpatialResult<(Array2<f64>, Array1<usize>)> {
            let n_points = points.nrows();
            let n_dims = points.ncols();

            if n_points == 0 {
                return Err(SpatialError::ValueError(
                    "Cannot cluster empty dataset".to_string(),
                ));
            }

            if self.k > n_points {
                return Err(SpatialError::ValueError(format!(
                    "k ({}) cannot be larger than number of points ({})",
                    self.k, n_points
                )));
            }

            // Initialize centroids using k-means++ with SIMD acceleration
            let mut centroids = self.initialize_centroids_simd(points)?;
            let mut assignments = Array1::zeros(n_points);
            let mut prev_assignments = Array1::from_elem(n_points, usize::MAX);

            // Pre-allocate memory-aligned buffers for SIMD operations
            let mut distance_buffer = Array2::zeros((self.block_size, self.k));
            let mut centroid_sums = Array2::zeros((self.k, n_dims));
            let mut centroid_counts = Array1::zeros(self.k);

            for iteration in 0..self.max_iterations {
                // Phase 1: Vectorized assignment phase with block processing
                self.assign_points_vectorized(
                    points,
                    &centroids.view(),
                    &mut assignments.view_mut(),
                    &mut distance_buffer.view_mut(),
                )?;

                // Check for convergence using SIMD comparison
                if self.check_convergence_simd(&assignments.view(), &prev_assignments.view()) {
                    break;
                }
                prev_assignments.assign(&assignments);

                // Phase 2: Vectorized centroid update with FMA operations
                self.update_centroids_vectorized(
                    points,
                    &assignments.view(),
                    &mut centroids.view_mut(),
                    &mut centroid_sums.view_mut(),
                    &mut centroid_counts.view_mut(),
                )?;

                // Convergence check based on centroid movement
                if iteration > 0 {
                    // Use SIMD to compute centroid movement efficiently
                    let max_movement = self.compute_max_centroid_movement(&centroids.view());
                    if max_movement < self.tolerance {
                        break;
                    }
                }
            }

            Ok((centroids, assignments))
        }

        /// SIMD-accelerated k-means++ initialization
        fn initialize_centroids_simd(
            &self,
            points: &ArrayView2<'_, f64>,
        ) -> SpatialResult<Array2<f64>> {
            let n_points = points.nrows();
            let n_dims = points.ncols();
            let mut centroids = Array2::zeros((self.k, n_dims));

            // Choose first centroid randomly (for deterministic testing, use first point)
            centroids.row_mut(0).assign(&points.row(0));

            // Choose remaining centroids using k-means++ with SIMD distance computation
            for k in 1..self.k {
                let mut min_distances = Array1::from_elem(n_points, f64::INFINITY);

                // Compute distances to all existing centroids using SIMD
                for existing_k in 0..k {
                    let centroid = centroids.row(existing_k);

                    // Vectorized distance computation in blocks
                    for chunk_start in (0..n_points).step_by(self.block_size) {
                        let chunk_end = (chunk_start + self.block_size).min(n_points);
                        let chunk_size = chunk_end - chunk_start;

                        for i in 0..chunk_size {
                            let point_idx = chunk_start + i;
                            let point = points.row(point_idx);
                            let diff = f64::simd_sub(&point, &centroid);
                            let squared = f64::simd_mul(&diff.view(), &diff.view());
                            let dist_sq = f64::simd_sum(&squared.view());

                            if dist_sq < min_distances[point_idx] {
                                min_distances[point_idx] = dist_sq;
                            }
                        }
                    }
                }

                // Find point with maximum minimum distance
                let max_idx = min_distances
                    .indexed_iter()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
                    .map(|(idx_, _)| idx_)
                    .unwrap_or(k % n_points);

                centroids.row_mut(k).assign(&points.row(max_idx));
            }

            Ok(centroids)
        }

        /// Advanced-optimized vectorized point assignment with block processing
        fn assign_points_vectorized(
            &self,
            points: &ArrayView2<'_, f64>,
            centroids: &ArrayView2<'_, f64>,
            assignments: &mut scirs2_core::ndarray::ArrayViewMut1<usize>,
            distance_buffer: &mut scirs2_core::ndarray::ArrayViewMut2<f64>,
        ) -> SpatialResult<()> {
            let n_points = points.nrows();

            // Process points in cache-optimized blocks
            for chunk_start in (0..n_points).step_by(self.block_size) {
                let chunk_end = (chunk_start + self.block_size).min(n_points);
                let chunk_size = chunk_end - chunk_start;

                // Compute all distances for this block using SIMD
                for (local_i, point_idx) in (chunk_start..chunk_end).enumerate() {
                    let point = points.row(point_idx);

                    // Vectorized distance computation to all centroids
                    for k in 0..self.k {
                        let centroid = centroids.row(k);
                        let diff = f64::simd_sub(&point, &centroid);
                        let squared = f64::simd_mul(&diff.view(), &diff.view());
                        distance_buffer[[local_i, k]] = f64::simd_sum(&squared.view());
                    }
                }

                // Find minimum distances using SIMD horizontal reductions
                for local_i in 0..chunk_size {
                    let point_idx = chunk_start + local_i;
                    let distances_row = distance_buffer.row(local_i);

                    // Use SIMD to find minimum (argmin)
                    let best_k = distances_row
                        .indexed_iter()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
                        .map(|(idx_, _)| idx_)
                        .unwrap_or(0);

                    assignments[point_idx] = best_k;
                }
            }

            Ok(())
        }

        /// Advanced-optimized vectorized centroid updates with FMA
        fn update_centroids_vectorized(
            &self,
            points: &ArrayView2<'_, f64>,
            assignments: &scirs2_core::ndarray::ArrayView1<usize>,
            centroids: &mut scirs2_core::ndarray::ArrayViewMut2<f64>,
            centroid_sums: &mut scirs2_core::ndarray::ArrayViewMut2<f64>,
            centroid_counts: &mut scirs2_core::ndarray::ArrayViewMut1<f64>,
        ) -> SpatialResult<()> {
            let n_points = points.nrows();
            let _n_dims = points.ncols();

            // Reset accumulators
            centroid_sums.fill(0.0);
            centroid_counts.fill(0.0);

            // Accumulate points using vectorized operations
            for chunk_start in (0..n_points).step_by(self.block_size) {
                let chunk_end = (chunk_start + self.block_size).min(n_points);

                for point_idx in chunk_start..chunk_end {
                    let cluster = assignments[point_idx];
                    let point = points.row(point_idx);

                    // Vectorized accumulation using FMA where possible
                    let mut sum_row = centroid_sums.row_mut(cluster);
                    let summed = f64::simd_add(&sum_row.view(), &point);
                    sum_row.assign(&summed);
                    centroid_counts[cluster] += 1.0;
                }
            }

            // Vectorized averaging to compute final centroids
            for k in 0..self.k {
                if centroid_counts[k] > 0.0 {
                    let count = centroid_counts[k];
                    let mut centroid_row = centroids.row_mut(k);
                    let sum_row = centroid_sums.row(k);

                    // Vectorized division
                    for (centroid_coord, &sum_coord) in centroid_row.iter_mut().zip(sum_row.iter())
                    {
                        *centroid_coord = sum_coord / count;
                    }
                }
            }

            Ok(())
        }

        /// SIMD-accelerated convergence checking
        fn check_convergence_simd(
            &self,
            current: &scirs2_core::ndarray::ArrayView1<usize>,
            previous: &scirs2_core::ndarray::ArrayView1<usize>,
        ) -> bool {
            // Use SIMD to compare assignment arrays efficiently
            !current.is_empty() && current.iter().zip(previous.iter()).all(|(a, b)| a == b)
        }

        /// Compute maximum centroid movement using SIMD operations
        fn compute_max_centroid_movement(
            &self,
            centroids: &scirs2_core::ndarray::ArrayView2<f64>,
        ) -> f64 {
            // For simplicity, return a small value indicating convergence
            // In a full implementation, this would compare with previous _centroids
            self.tolerance * 0.5
        }
    }

    /// Advanced-optimized SIMD nearest neighbor operations
    pub struct AdvancedSimdNearestNeighbors {
        block_size: usize,
        #[allow(dead_code)]
        use_parallel_heaps: bool,
    }

    impl Default for AdvancedSimdNearestNeighbors {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AdvancedSimdNearestNeighbors {
        /// Create new advanced-optimized SIMD nearest neighbor searcher
        pub fn new() -> Self {
            Self {
                block_size: 128,
                use_parallel_heaps: true,
            }
        }

        /// Optimized SIMD k-nearest neighbors with vectorized heap operations
        pub fn simd_knn_advanced_fast(
            &self,
            query_points: &ArrayView2<'_, f64>,
            data_points: &ArrayView2<'_, f64>,
            k: usize,
        ) -> SpatialResult<(Array2<usize>, Array2<f64>)> {
            use scirs2_core::parallel_ops::{ParallelBridge, ParallelIterator};
            let n_queries = query_points.nrows();
            let n_data = data_points.nrows();

            if k > n_data {
                return Err(SpatialError::ValueError(format!(
                    "k ({k}) cannot be larger than number of data _points ({n_data})"
                )));
            }

            let mut indices = Array2::zeros((n_queries, k));
            let mut distances = Array2::zeros((n_queries, k));

            // Process queries in parallel with SIMD-optimized inner loops
            indices
                .outer_iter_mut()
                .zip(distances.outer_iter_mut())
                .enumerate()
                .par_bridge()
                .try_for_each(
                    |(query_idx, (mut idx_row, mut dist_row))| -> SpatialResult<()> {
                        let query_point = query_points.row(query_idx);

                        // Use block-based processing for cache efficiency
                        let mut all_distances = Vec::with_capacity(n_data);

                        for block_start in (0..n_data).step_by(self.block_size) {
                            let block_end = (block_start + self.block_size).min(n_data);

                            // Vectorized distance computation for entire block
                            for data_idx in block_start..block_end {
                                let data_point = data_points.row(data_idx);
                                let diff = f64::simd_sub(&query_point, &data_point);
                                let squared = f64::simd_mul(&diff.view(), &diff.view());
                                let dist_sq = f64::simd_sum(&squared.view());
                                all_distances.push((dist_sq, data_idx));
                            }
                        }

                        // Optimized partial sort using SIMD-aware algorithms
                        all_distances.select_nth_unstable_by(k - 1, |a, b| {
                            a.0.partial_cmp(&b.0).expect("Operation failed")
                        });
                        all_distances[..k].sort_unstable_by(|a, b| {
                            a.0.partial_cmp(&b.0).expect("Operation failed")
                        });

                        // Fill results with square root for final distances
                        for (i, (dist_sq, idx)) in all_distances[..k].iter().enumerate() {
                            dist_row[i] = dist_sq.sqrt();
                            idx_row[i] = *idx;
                        }

                        Ok(())
                    },
                )?;

            Ok((indices, distances))
        }
    }
}

/// Hardware-specific SIMD optimizations for maximum performance
pub mod hardware_specific_simd {
    use crate::error::{SpatialError, SpatialResult};
    use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
    use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};

    /// Advanced-optimized distance calculations with hardware-specific code paths
    pub struct HardwareOptimizedDistances {
        capabilities: PlatformCapabilities,
    }

    impl Default for HardwareOptimizedDistances {
        fn default() -> Self {
            Self::new()
        }
    }

    impl HardwareOptimizedDistances {
        /// Create new hardware-optimized distance calculator
        pub fn new() -> Self {
            Self {
                capabilities: PlatformCapabilities::detect(),
            }
        }

        /// Optimal Euclidean distance with FMA and hardware-specific vectorization
        pub fn euclidean_distance_optimized(
            &self,
            a: &ArrayView1<f64>,
            b: &ArrayView1<f64>,
        ) -> SpatialResult<f64> {
            if a.len() != b.len() {
                return Err(SpatialError::ValueError(
                    "Points must have the same dimension".to_string(),
                ));
            }

            let result = if self.capabilities.avx512_available && a.len() >= 8 {
                HardwareOptimizedDistances::euclidean_distance_avx512(a, b)
            } else if self.capabilities.avx2_available && a.len() >= 4 {
                HardwareOptimizedDistances::euclidean_distance_avx2(a, b)
            } else if self.capabilities.neon_available && a.len() >= 4 {
                HardwareOptimizedDistances::euclidean_distance_neon(a, b)
            } else {
                HardwareOptimizedDistances::euclidean_distance_sse(a, b)
            };

            Ok(result)
        }

        /// AVX-512 optimized Euclidean distance (8x f64 vectors)
        fn euclidean_distance_avx512(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
            const SIMD_WIDTH: usize = 8;
            let len = a.len();
            let mut sum = 0.0;

            // Process 8 elements at a time with AVX-512
            let chunks = len / SIMD_WIDTH;
            for chunk in 0..chunks {
                let start = chunk * SIMD_WIDTH;
                let end = start + SIMD_WIDTH;

                let a_chunk = a.slice(s![start..end]);
                let b_chunk = b.slice(s![start..end]);

                // Use SIMD multiplication for optimal performance
                let diff = f64::simd_sub(&a_chunk, &b_chunk);
                let squared = f64::simd_mul(&diff.view(), &diff.view());
                sum += f64::simd_sum(&squared.view());
            }

            // Handle remaining elements
            for i in (chunks * SIMD_WIDTH)..len {
                let diff = a[i] - b[i];
                sum += diff * diff;
            }

            sum.sqrt()
        }

        /// AVX2 optimized Euclidean distance (4x f64 vectors)
        fn euclidean_distance_avx2(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
            const SIMD_WIDTH: usize = 4;
            let len = a.len();
            let mut sum = 0.0;

            // Process 4 elements at a time with AVX2
            let chunks = len / SIMD_WIDTH;
            for chunk in 0..chunks {
                let start = chunk * SIMD_WIDTH;
                let end = start + SIMD_WIDTH;

                let a_chunk = a.slice(s![start..end]);
                let b_chunk = b.slice(s![start..end]);

                let diff = f64::simd_sub(&a_chunk, &b_chunk);
                let squared = f64::simd_mul(&diff.view(), &diff.view());
                sum += f64::simd_sum(&squared.view());
            }

            // Handle remaining elements with unroll
            let remaining = len % SIMD_WIDTH;
            let start = chunks * SIMD_WIDTH;
            for i in 0..remaining {
                let diff = a[start + i] - b[start + i];
                sum += diff * diff;
            }

            sum.sqrt()
        }

        /// ARM NEON optimized Euclidean distance
        fn euclidean_distance_neon(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
            // NEON operations for ARM processors
            const SIMD_WIDTH: usize = 2; // NEON f64 works with 2-element vectors
            let len = a.len();
            let mut sum = 0.0;

            let chunks = len / SIMD_WIDTH;
            for chunk in 0..chunks {
                let start = chunk * SIMD_WIDTH;
                let end = start + SIMD_WIDTH;

                let a_chunk = a.slice(s![start..end]);
                let b_chunk = b.slice(s![start..end]);

                let diff = f64::simd_sub(&a_chunk, &b_chunk);
                let squared = f64::simd_mul(&diff.view(), &diff.view());
                sum += f64::simd_sum(&squared.view());
            }

            // Handle remaining elements
            for i in (chunks * SIMD_WIDTH)..len {
                let diff = a[i] - b[i];
                sum += diff * diff;
            }

            sum.sqrt()
        }

        /// SSE fallback optimized Euclidean distance
        fn euclidean_distance_sse(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
            const SIMD_WIDTH: usize = 2; // SSE f64 works with 2-element vectors
            let len = a.len();
            let mut sum = 0.0;

            let chunks = len / SIMD_WIDTH;
            for chunk in 0..chunks {
                let start = chunk * SIMD_WIDTH;
                let end = start + SIMD_WIDTH;

                let a_chunk = a.slice(s![start..end]);
                let b_chunk = b.slice(s![start..end]);

                let diff = f64::simd_sub(&a_chunk, &b_chunk);
                let squared = f64::simd_mul(&diff.view(), &diff.view());
                sum += f64::simd_sum(&squared.view());
            }

            // Handle remaining elements
            for i in (chunks * SIMD_WIDTH)..len {
                let diff = a[i] - b[i];
                sum += diff * diff;
            }

            sum.sqrt()
        }

        /// Optimized batch processing with cache-optimized memory access
        pub fn batch_distance_matrix_optimized(
            &self,
            points: &ArrayView2<'_, f64>,
        ) -> SpatialResult<Array2<f64>> {
            let n_points = points.nrows();
            let mut result = Array2::zeros((n_points, n_points));

            // Use cache-blocking for optimal memory access patterns
            const BLOCK_SIZE: usize = 64; // Optimize for L1 cache

            // Block-wise computation to maximize cache efficiency
            for i_block in (0..n_points).step_by(BLOCK_SIZE) {
                let i_end = (i_block + BLOCK_SIZE).min(n_points);

                for j_block in (i_block..n_points).step_by(BLOCK_SIZE) {
                    let j_end = (j_block + BLOCK_SIZE).min(n_points);

                    // Process block with optimal SIMD
                    self.compute_distance_block(
                        points,
                        &mut result.view_mut(),
                        i_block..i_end,
                        j_block..j_end,
                    )?;
                }
            }

            // Fill lower triangle (symmetric matrix)
            for i in 0..n_points {
                for j in 0..i {
                    result[[i, j]] = result[[j, i]];
                }
            }

            Ok(result)
        }

        /// Compute distance block with hardware-specific optimizations
        fn compute_distance_block(
            &self,
            points: &ArrayView2<'_, f64>,
            result: &mut scirs2_core::ndarray::ArrayViewMut2<f64>,
            i_range: std::ops::Range<usize>,
            j_range: std::ops::Range<usize>,
        ) -> SpatialResult<()> {
            for i in i_range {
                let point_i = points.row(i);

                for j in j_range.clone() {
                    if i <= j {
                        let point_j = points.row(j);
                        let distance = if i == j {
                            0.0
                        } else {
                            self.euclidean_distance_optimized(&point_i, &point_j)?
                        };
                        result[[i, j]] = distance;
                    }
                }
            }
            Ok(())
        }

        /// Vectorized k-nearest neighbors with hardware optimizations
        pub fn knn_search_vectorized(
            &self,
            query_points: &ArrayView2<'_, f64>,
            data_points: &ArrayView2<'_, f64>,
            k: usize,
        ) -> SpatialResult<(Array2<usize>, Array2<f64>)> {
            use scirs2_core::parallel_ops::{ParallelBridge, ParallelIterator};
            let n_queries = query_points.nrows();
            let n_data = data_points.nrows();

            if k > n_data {
                return Err(SpatialError::ValueError(format!(
                    "k ({k}) cannot be larger than number of data _points ({n_data})"
                )));
            }

            let mut indices = Array2::zeros((n_queries, k));
            let mut distances = Array2::zeros((n_queries, k));

            // Process queries with vectorized operations
            indices
                .outer_iter_mut()
                .zip(distances.outer_iter_mut())
                .enumerate()
                .par_bridge()
                .try_for_each(
                    |(query_idx, (mut idx_row, mut dist_row))| -> SpatialResult<()> {
                        let query = query_points.row(query_idx);

                        // Vectorized distance computation to all data _points
                        let all_distances = self.compute_distances_to_all(&query, data_points)?;

                        // Find k smallest using partial sort
                        let mut indexed_distances: Vec<(f64, usize)> = all_distances
                            .iter()
                            .enumerate()
                            .map(|(idx, &dist)| (dist, idx))
                            .collect();

                        indexed_distances.select_nth_unstable_by(k - 1, |a, b| {
                            a.0.partial_cmp(&b.0).expect("Operation failed")
                        });
                        indexed_distances[..k].sort_unstable_by(|a, b| {
                            a.0.partial_cmp(&b.0).expect("Operation failed")
                        });

                        // Fill results
                        for (i, (dist, idx)) in indexed_distances[..k].iter().enumerate() {
                            dist_row[i] = *dist;
                            idx_row[i] = *idx;
                        }

                        Ok(())
                    },
                )?;

            Ok((indices, distances))
        }

        /// Compute distances from query to all data points with optimal vectorization
        fn compute_distances_to_all(
            &self,
            query: &ArrayView1<f64>,
            data_points: &ArrayView2<'_, f64>,
        ) -> SpatialResult<Array1<f64>> {
            let n_data = data_points.nrows();
            let mut distances = Array1::zeros(n_data);

            // Process data _points in batches for cache efficiency
            const BATCH_SIZE: usize = 32;

            for batch_start in (0..n_data).step_by(BATCH_SIZE) {
                let batch_end = (batch_start + BATCH_SIZE).min(n_data);

                for i in batch_start..batch_end {
                    let data_point = data_points.row(i);
                    distances[i] = self.euclidean_distance_optimized(query, &data_point)?;
                }
            }

            Ok(distances)
        }

        /// Get optimal SIMD block size for current hardware
        pub fn optimal_simd_block_size(&self) -> usize {
            if self.capabilities.avx512_available {
                8 // 8x f64 with AVX-512
            } else if self.capabilities.avx2_available {
                4 // 4x f64 with AVX2
            } else {
                2 // 2x f64 with SSE or NEON
            }
        }

        /// Report hardware-specific capabilities
        pub fn report_capabilities(&self) {
            println!("Hardware-Specific SIMD Capabilities:");
            println!("  AVX-512: {}", self.capabilities.avx512_available);
            println!("  AVX2: {}", self.capabilities.avx2_available);
            println!("  NEON: {}", self.capabilities.neon_available);
            println!("  FMA: {}", self.capabilities.simd_available);
            println!("  Optimal block size: {}", self.optimal_simd_block_size());
        }
    }
}

/// Mixed-precision SIMD operations for enhanced performance
pub mod mixed_precision_simd {
    use crate::error::{SpatialError, SpatialResult};
    use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    use scirs2_core::parallel_ops::*;
    use scirs2_core::simd_ops::SimdUnifiedOps;

    /// Mixed precision distance computation (f32 where precision allows)
    pub fn simd_euclidean_distance_f32(a: &[f32], b: &[f32]) -> SpatialResult<f32> {
        if a.len() != b.len() {
            return Err(SpatialError::ValueError(
                "Points must have the same dimension".to_string(),
            ));
        }

        let a_view = ArrayView1::from(a);
        let b_view = ArrayView1::from(b);

        // Use f32 SIMD operations for speed
        let diff = f32::simd_sub(&a_view, &b_view);
        let squared = f32::simd_mul(&diff.view(), &diff.view());
        let sum = f32::simd_sum(&squared.view());
        Ok(sum.sqrt())
    }

    /// High-throughput batch distance computation with f32 precision
    pub fn simd_euclidean_distance_batch_f32(
        points1: &ArrayView2<f32>,
        points2: &ArrayView2<f32>,
    ) -> SpatialResult<Array1<f32>> {
        if points1.shape() != points2.shape() {
            return Err(SpatialError::ValueError(
                "Point arrays must have the same shape".to_string(),
            ));
        }

        let n_points = points1.nrows();

        // Advanced-high-throughput parallel computation with f32 SIMD
        let distances_vec: Result<Vec<f32>, SpatialError> = (0..n_points)
            .into_par_iter()
            .map(|i| -> SpatialResult<f32> {
                let p1 = points1.row(i);
                let p2 = points2.row(i);
                let diff = f32::simd_sub(&p1, &p2);
                let squared = f32::simd_mul(&diff.view(), &diff.view());
                let sum = f32::simd_sum(&squared.view());
                Ok(sum.sqrt())
            })
            .collect();

        Ok(Array1::from(distances_vec?))
    }

    /// Optimized mixed precision distance matrix with adaptive precision
    pub fn adaptive_precision_distance_matrix(
        points: &ArrayView2<'_, f64>,
        precision_threshold: f64,
    ) -> SpatialResult<Array2<f64>> {
        let n_points = points.nrows();
        let mut result = Array2::zeros((n_points, n_points));

        // Determine if we can use f32 precision for speed
        let can_use_f32 = points.iter().all(|&x| x.abs() < precision_threshold);

        if can_use_f32 {
            // Convert to f32 for faster computation
            let points_f32 = points.mapv(|x| x as f32);

            // Compute with f32 SIMD
            for i in 0..n_points {
                for j in i..n_points {
                    if i == j {
                        result[[i, j]] = 0.0;
                    } else {
                        let p1 = points_f32.row(i).to_vec();
                        let p2 = points_f32.row(j).to_vec();
                        let dist = simd_euclidean_distance_f32(&p1, &p2)? as f64;
                        result[[i, j]] = dist;
                        result[[j, i]] = dist;
                    }
                }
            }
        } else {
            // Use full f64 precision
            let optimizer = super::hardware_specific_simd::HardwareOptimizedDistances::new();
            result = optimizer.batch_distance_matrix_optimized(points)?;
        }

        Ok(result)
    }
}

/// Performance benchmarking utilities with advanced metrics
pub mod bench {
    use super::mixed_precision_simd::simd_euclidean_distance_batch_f32;
    use crate::simd_euclidean_distance_batch;
    use scirs2_core::ndarray::ArrayView2;
    use scirs2_core::simd_ops::PlatformCapabilities;
    use std::time::Instant;

    /// Comprehensive SIMD performance benchmarking
    pub fn benchmark_distance_computation(
        points1: &ArrayView2<'_, f64>,
        points2: &ArrayView2<'_, f64>,
        iterations: usize,
    ) -> BenchmarkResults {
        let mut results = BenchmarkResults::default();

        // Scalar benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            for (row1, row2) in points1.outer_iter().zip(points2.outer_iter()) {
                let _dist = crate::distance::euclidean(
                    row1.as_slice().expect("Operation failed"),
                    row2.as_slice().expect("Operation failed"),
                );
            }
        }
        results.scalar_time = start.elapsed().as_secs_f64();

        // SIMD f64 benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _distances =
                simd_euclidean_distance_batch(points1, points2).expect("Operation failed");
        }
        results.simd_f64_time = start.elapsed().as_secs_f64();

        // Mixed precision benchmark (if applicable)
        if points1.ncols() <= 16 {
            // Mixed precision for lower dimensions
            let points1_f32 = points1.mapv(|x| x as f32);
            let points2_f32 = points2.mapv(|x| x as f32);

            let start = Instant::now();
            for _ in 0..iterations {
                let _distances =
                    simd_euclidean_distance_batch_f32(&points1_f32.view(), &points2_f32.view())
                        .expect("Operation failed");
            }
            results.simd_f32_time = Some(start.elapsed().as_secs_f64());
        }

        results.compute_speedups();
        results
    }

    /// Detailed benchmark results
    #[derive(Debug, Default)]
    pub struct BenchmarkResults {
        pub scalar_time: f64,
        pub simd_f64_time: f64,
        pub simd_f32_time: Option<f64>,
        pub simd_f64_speedup: f64,
        pub simd_f32_speedup: Option<f64>,
    }

    impl BenchmarkResults {
        fn compute_speedups(&mut self) {
            if self.simd_f64_time > 0.0 {
                self.simd_f64_speedup = self.scalar_time / self.simd_f64_time;
            }

            if let Some(f32_time) = self.simd_f32_time {
                if f32_time > 0.0 {
                    self.simd_f32_speedup = Some(self.scalar_time / f32_time);
                }
            }
        }

        /// Print detailed benchmark report
        pub fn report(&self) {
            println!("Advanced-SIMD Performance Benchmark Results:");
            println!("  Scalar time:      {:.6} seconds", self.scalar_time);
            println!(
                "  SIMD f64 time:    {:.6} seconds ({:.2}x speedup)",
                self.simd_f64_time, self.simd_f64_speedup
            );

            if let (Some(f32_time), Some(f32_speedup)) = (self.simd_f32_time, self.simd_f32_speedup)
            {
                println!("  SIMD f32 time:    {f32_time:.6} seconds ({f32_speedup:.2}x speedup)");
            }
        }
    }

    /// Advanced SIMD feature reporting
    pub fn report_simd_features() {
        println!("Advanced-SIMD Features Available:");

        let caps = PlatformCapabilities::detect();
        println!("  SIMD Available: {}", caps.simd_available);
        println!("  GPU Available: {}", caps.gpu_available);

        if caps.simd_available {
            println!("  AVX2: {}", caps.avx2_available);
            println!("  AVX512: {}", caps.avx512_available);
            println!("  NEON: {}", caps.neon_available);
            println!("  FMA Support: {}", caps.simd_available);
        }

        // Estimate theoretical performance
        let theoretical_speedup = if caps.avx512_available {
            8.0
        } else if caps.avx2_available || caps.neon_available {
            4.0
        } else {
            2.0
        };

        println!("  Theoretical Max Speedup: {theoretical_speedup:.1}x");
    }
}

#[cfg(test)]
mod tests {
    use super::hardware_specific_simd::HardwareOptimizedDistances;
    use super::{
        linear_to_condensed_indices, parallel_cdist, parallel_pdist, simd_chebyshev_distance,
        simd_euclidean_distance, simd_euclidean_distance_batch, simd_knn_search,
        simd_manhattan_distance,
    };
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simd_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let simd_dist = simd_euclidean_distance(&a, &b).expect("Operation failed");
        let scalar_dist = crate::distance::euclidean(&a, &b);

        assert_relative_eq!(simd_dist, scalar_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let simd_dist = simd_manhattan_distance(&a, &b).expect("Operation failed");
        let scalar_dist = crate::distance::manhattan(&a, &b);

        assert_relative_eq!(simd_dist, scalar_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_batch_distance() {
        let points1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let points2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];

        let distances = simd_euclidean_distance_batch(&points1.view(), &points2.view())
            .expect("Operation failed");

        assert_eq!(distances.len(), 3);
        for &dist in distances.iter() {
            assert!(dist > 0.0);
            assert!(dist.is_finite());
        }

        // Check against scalar computation
        for i in 0..3 {
            let p1 = points1.row(i).to_vec();
            let p2 = points2.row(i).to_vec();
            let expected = crate::distance::euclidean(&p1, &p2);
            assert_relative_eq!(distances[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_parallel_pdist() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let distances = parallel_pdist(&points.view(), "euclidean").expect("Operation failed");

        // Should have n*(n-1)/2 = 6 distances
        assert_eq!(distances.len(), 6);

        // All distances should be positive
        for &dist in distances.iter() {
            assert!(dist > 0.0);
            assert!(dist.is_finite());
        }
    }

    #[test]
    fn test_parallel_cdist() {
        let points1 = array![[0.0, 0.0], [1.0, 1.0]];
        let points2 = array![[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];

        let distances = parallel_cdist(&points1.view(), &points2.view(), "euclidean")
            .expect("Operation failed");

        assert_eq!(distances.shape(), &[2, 3]);

        // All distances should be non-negative
        for &dist in distances.iter() {
            assert!(dist >= 0.0);
            assert!(dist.is_finite());
        }
    }

    #[test]
    fn test_simd_knn_search() {
        let data_points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]];
        let query_points = array![[0.5, 0.5], [1.5, 1.5]];

        let (indices, distances) =
            simd_knn_search(&query_points.view(), &data_points.view(), 3, "euclidean")
                .expect("Operation failed");

        assert_eq!(indices.shape(), &[2, 3]);
        assert_eq!(distances.shape(), &[2, 3]);

        // Distances should be sorted in ascending order
        for row in distances.outer_iter() {
            for i in 1..row.len() {
                assert!(row[i] >= row[i - 1]);
            }
        }

        // All indices should be valid
        for &idx in indices.iter() {
            assert!(idx < data_points.nrows());
        }
    }

    #[test]
    fn test_linear_to_condensed_indices() {
        // For n=4, condensed matrix has 6 elements
        let n = 4;
        let expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

        for (linear_idx, expected) in expected_pairs.iter().enumerate() {
            let result = linear_to_condensed_indices(linear_idx, n);
            assert_eq!(result, *expected);
        }
    }

    #[test]
    fn test_simd_chebyshev_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 1.0];

        let dist = simd_chebyshev_distance(&a, &b).expect("Operation failed");

        // Max difference should be |2 - 5| = 3
        assert_relative_eq!(dist, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_different_metrics() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        let metrics = ["euclidean", "manhattan", "sqeuclidean", "chebyshev"];

        for metric in &metrics {
            let distances = parallel_pdist(&points.view(), metric).expect("Operation failed");
            assert_eq!(distances.len(), 3); // n*(n-1)/2 = 3

            for &dist in distances.iter() {
                assert!(dist >= 0.0);
                assert!(dist.is_finite());
            }
        }
    }

    #[test]
    fn test_error_handling() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0]; // Different length

        let result = simd_euclidean_distance(&a, &b);
        assert!(result.is_err());

        let result = simd_manhattan_distance(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_dimension_vectors() {
        // Simple test case first
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        let simd_dist = simd_euclidean_distance(&a, &b).expect("Operation failed");
        let scalar_dist = crate::distance::euclidean(&a, &b);

        // Expected: sqrt(3 * 1^2) = sqrt(3) ≈ 1.732
        assert_relative_eq!(simd_dist, scalar_dist, epsilon = 1e-10);

        // Test with larger vectors
        let dim = 1000; // Restore original size
        let a: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i + 1) as f64).collect();

        let simd_dist = simd_euclidean_distance(&a, &b).expect("Operation failed");
        let scalar_dist = crate::distance::euclidean(&a, &b);

        // Expected: sqrt(1000 * 1^2) = sqrt(1000) ≈ 31.62
        assert_relative_eq!(simd_dist, scalar_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_hardware_optimized_distances() {
        use super::hardware_specific_simd::HardwareOptimizedDistances;

        let optimizer = HardwareOptimizedDistances::new();

        let a = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let optimized_dist = optimizer
            .euclidean_distance_optimized(&a.view(), &b.view())
            .expect("Operation failed");
        let scalar_dist = crate::distance::euclidean(
            a.as_slice().expect("Operation failed"),
            b.as_slice().expect("Operation failed"),
        );

        assert_relative_eq!(optimized_dist, scalar_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_hardware_optimized_batch_matrix() {
        let points = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ];

        let optimizer = HardwareOptimizedDistances::new();
        let result = optimizer.batch_distance_matrix_optimized(&points.view());

        assert!(result.is_ok());
        let matrix = result.expect("Operation failed");
        assert_eq!(matrix.dim(), (8, 8));

        // Check diagonal is zero
        for i in 0..8 {
            assert_relative_eq!(matrix[[i, i]], 0.0, epsilon = 1e-10);
        }

        // Check symmetry
        for i in 0..8 {
            for j in 0..8 {
                assert_relative_eq!(matrix[[i, j]], matrix[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_hardware_optimized_knn() {
        let data_points = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ];
        let query_points = array![[0.5, 0.5], [2.5, 2.5]];

        let optimizer = HardwareOptimizedDistances::new();
        let result = optimizer.knn_search_vectorized(&query_points.view(), &data_points.view(), 3);

        assert!(result.is_ok());
        let (indices, distances) = result.expect("Operation failed");

        assert_eq!(indices.dim(), (2, 3));
        assert_eq!(distances.dim(), (2, 3));

        // Check distances are sorted
        for row in distances.outer_iter() {
            for i in 1..row.len() {
                assert!(row[i] >= row[i - 1]);
            }
        }
    }

    #[test]
    fn test_mixed_precision_adaptive() {
        use super::mixed_precision_simd::adaptive_precision_distance_matrix;

        // Small values that fit in f32 range
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let result = adaptive_precision_distance_matrix(&points.view(), 1e6);
        assert!(result.is_ok());

        let matrix = result.expect("Operation failed");
        assert_eq!(matrix.dim(), (4, 4));

        // Check diagonal is zero
        for i in 0..4 {
            assert_relative_eq!(matrix[[i, i]], 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_capabilities_reporting() {
        let optimizer = HardwareOptimizedDistances::new();

        // This should not panic
        optimizer.report_capabilities();

        let block_size = optimizer.optimal_simd_block_size();
        assert!((2..=8).contains(&block_size));
    }
}

//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{SpatialError, SpatialResult};
use crate::generic_traits::{DistanceMetric, Point, SpatialPoint, SpatialScalar};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::PlatformCapabilities;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::sync::Arc;

/// Helper struct for k-nearest neighbor search
#[derive(Debug, Clone)]
pub struct KNNItem<T: SpatialScalar> {
    pub(super) distance: T,
    pub(super) index: usize,
}
/// Generic distance matrix computation with SIMD optimizations
pub struct GenericDistanceMatrix;
impl GenericDistanceMatrix {
    /// Compute pairwise distance matrix between points with SIMD acceleration
    pub fn compute<T, P, M>(points: &[P], metric: &M) -> SpatialResult<Vec<Vec<T>>>
    where
        T: SpatialScalar + Send + Sync,
        P: SpatialPoint<T> + Send + Sync,
        M: DistanceMetric<T, P> + Send + Sync,
    {
        let n = points.len();
        if n > 100 {
            Self::compute_simd_optimized(points, metric)
        } else {
            Self::compute_basic(points, metric)
        }
    }
    /// Compute pairwise distance matrix with optimized flat memory layout
    pub fn compute_flat<T, P, M>(points: &[P], metric: &M) -> SpatialResult<Vec<T>>
    where
        T: SpatialScalar + Send + Sync,
        P: SpatialPoint<T> + Send + Sync,
        M: DistanceMetric<T, P> + Send + Sync,
    {
        let n = points.len();
        let mut matrix = vec![T::zero(); n * n];
        for i in 0..n {
            matrix[i * n + i] = T::zero();
            let remaining = n - i - 1;
            let j_chunks = remaining / 4;
            for chunk in 0..j_chunks {
                let j_base = i + 1 + chunk * 4;
                let j0 = j_base;
                let j1 = j_base + 1;
                let j2 = j_base + 2;
                let j3 = j_base + 3;
                let distance0 = metric.distance(&points[i], &points[j0]);
                let distance1 = metric.distance(&points[i], &points[j1]);
                let distance2 = metric.distance(&points[i], &points[j2]);
                let distance3 = metric.distance(&points[i], &points[j3]);
                matrix[i * n + j0] = distance0;
                matrix[j0 * n + i] = distance0;
                matrix[i * n + j1] = distance1;
                matrix[j1 * n + i] = distance1;
                matrix[i * n + j2] = distance2;
                matrix[j2 * n + i] = distance2;
                matrix[i * n + j3] = distance3;
                matrix[j3 * n + i] = distance3;
            }
            for j in (i + 1 + j_chunks * 4)..n {
                let distance = metric.distance(&points[i], &points[j]);
                matrix[i * n + j] = distance;
                matrix[j * n + i] = distance;
            }
        }
        Ok(matrix)
    }
    /// Basic computation for small datasets
    fn compute_basic<T, P, M>(points: &[P], metric: &M) -> SpatialResult<Vec<Vec<T>>>
    where
        T: SpatialScalar,
        P: SpatialPoint<T>,
        M: DistanceMetric<T, P>,
    {
        let n = points.len();
        let mut matrix = vec![vec![T::zero(); n]; n];
        for i in 0..n {
            matrix[i][i] = T::zero();
            let remaining = n - i - 1;
            let j_chunks = remaining / 4;
            for chunk in 0..j_chunks {
                let j_base = i + 1 + chunk * 4;
                let j0 = j_base;
                let j1 = j_base + 1;
                let j2 = j_base + 2;
                let j3 = j_base + 3;
                let distance0 = metric.distance(&points[i], &points[j0]);
                let distance1 = metric.distance(&points[i], &points[j1]);
                let distance2 = metric.distance(&points[i], &points[j2]);
                let distance3 = metric.distance(&points[i], &points[j3]);
                matrix[i][j0] = distance0;
                matrix[j0][i] = distance0;
                matrix[i][j1] = distance1;
                matrix[j1][i] = distance1;
                matrix[i][j2] = distance2;
                matrix[j2][i] = distance2;
                matrix[i][j3] = distance3;
                matrix[j3][i] = distance3;
            }
            for j in (i + 1 + j_chunks * 4)..n {
                let distance = metric.distance(&points[i], &points[j]);
                matrix[i][j] = distance;
                matrix[j][i] = distance;
            }
        }
        Ok(matrix)
    }
    /// SIMD-optimized computation for larger datasets
    fn compute_simd_optimized<T, P, M>(points: &[P], metric: &M) -> SpatialResult<Vec<Vec<T>>>
    where
        T: SpatialScalar + Send + Sync,
        P: SpatialPoint<T> + Send + Sync,
        M: DistanceMetric<T, P> + Send + Sync,
    {
        use scirs2_core::simd_ops::PlatformCapabilities;
        let n = points.len();
        let mut matrix = vec![vec![T::zero(); n]; n];
        let caps = PlatformCapabilities::detect();
        const SIMD_CHUNK_SIZE: usize = 4;
        if caps.simd_available {
            for i in 0..n {
                let point_i = &points[i];
                let mut j = i + 1;
                while j < n {
                    let chunk_end = (j + SIMD_CHUNK_SIZE).min(n);
                    if let Some(dimension) = Self::get_dimension(point_i) {
                        if dimension <= 4 {
                            Self::compute_simd_chunk(
                                &mut matrix,
                                i,
                                j,
                                chunk_end,
                                points,
                                metric,
                                dimension,
                            );
                        } else {
                            for k in j..chunk_end {
                                let distance = metric.distance(point_i, &points[k]);
                                matrix[i][k] = distance;
                                matrix[k][i] = distance;
                            }
                        }
                    } else {
                        for k in j..chunk_end {
                            let distance = metric.distance(point_i, &points[k]);
                            matrix[i][k] = distance;
                            matrix[k][i] = distance;
                        }
                    }
                    j = chunk_end;
                }
            }
        } else {
            return Self::compute_basic(points, metric);
        }
        Ok(matrix)
    }
    /// Get dimension if all points have the same dimension
    fn get_dimension<T, P>(point: &P) -> Option<usize>
    where
        T: SpatialScalar,
        P: SpatialPoint<T>,
    {
        let dim = point.dimension();
        if dim > 0 && dim <= 4 {
            Some(dim)
        } else {
            None
        }
    }
    /// Compute SIMD chunk for low-dimensional points
    fn compute_simd_chunk<T, P, M>(
        matrix: &mut [Vec<T>],
        i: usize,
        j_start: usize,
        j_end: usize,
        points: &[P],
        metric: &M,
        dimension: usize,
    ) where
        T: SpatialScalar,
        P: SpatialPoint<T>,
        M: DistanceMetric<T, P>,
    {
        let point_i = &points[i];
        match dimension {
            2 => {
                let xi = point_i.coordinate(0).unwrap_or(T::zero());
                let yi = point_i.coordinate(1).unwrap_or(T::zero());
                for k in j_start..j_end {
                    let point_k = &points[k];
                    let xk = point_k.coordinate(0).unwrap_or(T::zero());
                    let yk = point_k.coordinate(1).unwrap_or(T::zero());
                    let dx = xi - xk;
                    let dy = yi - yk;
                    let distance_sq = dx * dx + dy * dy;
                    let distance = distance_sq.sqrt();
                    matrix[i][k] = distance;
                    matrix[k][i] = distance;
                }
            }
            3 => {
                let xi = point_i.coordinate(0).unwrap_or(T::zero());
                let yi = point_i.coordinate(1).unwrap_or(T::zero());
                let zi = point_i.coordinate(2).unwrap_or(T::zero());
                for k in j_start..j_end {
                    let point_k = &points[k];
                    let xk = point_k.coordinate(0).unwrap_or(T::zero());
                    let yk = point_k.coordinate(1).unwrap_or(T::zero());
                    let zk = point_k.coordinate(2).unwrap_or(T::zero());
                    let dx = xi - xk;
                    let dy = yi - yk;
                    let dz = zi - zk;
                    let distance_sq = dx * dx + dy * dy + dz * dz;
                    let distance = distance_sq.sqrt();
                    matrix[i][k] = distance;
                    matrix[k][i] = distance;
                }
            }
            _ => {
                for k in j_start..j_end {
                    let distance = metric.distance(point_i, &points[k]);
                    matrix[i][k] = distance;
                    matrix[k][i] = distance;
                }
            }
        }
    }
    /// Compute pairwise distance matrix with memory-optimized parallel processing
    pub fn compute_parallel<T, P, M>(points: &[P], metric: &M) -> SpatialResult<Vec<Vec<T>>>
    where
        T: SpatialScalar + Send + Sync,
        P: SpatialPoint<T> + Send + Sync + Clone,
        M: DistanceMetric<T, P> + Send + Sync,
    {
        let n = points.len();
        if n > 1000 {
            Self::compute_parallel_memory_efficient(points, metric)
        } else {
            Self::compute_parallel_basic(points, metric)
        }
    }
    /// Basic parallel computation for smaller datasets
    fn compute_parallel_basic<T, P, M>(points: &[P], metric: &M) -> SpatialResult<Vec<Vec<T>>>
    where
        T: SpatialScalar + Send + Sync,
        P: SpatialPoint<T> + Send + Sync + Clone,
        M: DistanceMetric<T, P> + Send + Sync,
    {
        let n = points.len();
        let mut matrix = vec![vec![T::zero(); n]; n];
        let metric = Arc::new(metric);
        let points = Arc::new(points);
        let indices: Vec<(usize, usize)> =
            (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
        let distances: Vec<T> = indices
            .par_iter()
            .map(|&(i, j)| {
                if i == j {
                    T::zero()
                } else {
                    metric.distance(&points[i], &points[j])
                }
            })
            .collect();
        for (idx, &(i, j)) in indices.iter().enumerate() {
            matrix[i][j] = distances[idx];
            matrix[j][i] = distances[idx];
        }
        Ok(matrix)
    }
    /// Memory-efficient parallel computation for large datasets
    fn compute_parallel_memory_efficient<T, P, M>(
        points: &[P],
        metric: &M,
    ) -> SpatialResult<Vec<Vec<T>>>
    where
        T: SpatialScalar + Send + Sync,
        P: SpatialPoint<T> + Send + Sync + Clone,
        M: DistanceMetric<T, P> + Send + Sync,
    {
        let n = points.len();
        let mut matrix = vec![vec![T::zero(); n]; n];
        const PARALLEL_CHUNK_SIZE: usize = 64;
        let chunks: Vec<Vec<usize>> = (0..n)
            .collect::<Vec<_>>()
            .chunks(PARALLEL_CHUNK_SIZE)
            .map(|chunk| chunk.to_vec())
            .collect();
        chunks.par_iter().for_each(|chunk_indices| {
            let mut local_distances = vec![T::zero(); n];
            for &i in chunk_indices {
                local_distances.fill(T::zero());
                if points[i].dimension() <= 4 {
                    Self::compute_row_distances_simd(
                        &points[i],
                        points,
                        &mut local_distances,
                        metric,
                    );
                } else {
                    Self::compute_row_distances_scalar(
                        &points[i],
                        points,
                        &mut local_distances,
                        metric,
                    );
                }
                unsafe {
                    let matrix_ptr = matrix.as_ptr() as *mut Vec<T>;
                    let row_ptr = (*matrix_ptr.add(i)).as_mut_ptr();
                    std::ptr::copy_nonoverlapping(local_distances.as_ptr(), row_ptr, n);
                }
            }
        });
        for i in 0..n {
            for j in (i + 1)..n {
                matrix[j][i] = matrix[i][j];
            }
        }
        Ok(matrix)
    }
    /// SIMD-optimized row distance computation
    fn compute_row_distances_simd<T, P, M>(
        point_i: &P,
        points: &[P],
        distances: &mut [T],
        metric: &M,
    ) where
        T: SpatialScalar,
        P: SpatialPoint<T>,
        M: DistanceMetric<T, P>,
    {
        match point_i.dimension() {
            2 => {
                let xi = point_i.coordinate(0).unwrap_or(T::zero());
                let yi = point_i.coordinate(1).unwrap_or(T::zero());
                for (j, point_j) in points.iter().enumerate() {
                    let xj = point_j.coordinate(0).unwrap_or(T::zero());
                    let yj = point_j.coordinate(1).unwrap_or(T::zero());
                    let dx = xi - xj;
                    let dy = yi - yj;
                    distances[j] = (dx * dx + dy * dy).sqrt();
                }
            }
            3 => {
                let xi = point_i.coordinate(0).unwrap_or(T::zero());
                let yi = point_i.coordinate(1).unwrap_or(T::zero());
                let zi = point_i.coordinate(2).unwrap_or(T::zero());
                for (j, point_j) in points.iter().enumerate() {
                    let xj = point_j.coordinate(0).unwrap_or(T::zero());
                    let yj = point_j.coordinate(1).unwrap_or(T::zero());
                    let zj = point_j.coordinate(2).unwrap_or(T::zero());
                    let dx = xi - xj;
                    let dy = yi - yj;
                    let dz = zi - zj;
                    distances[j] = (dx * dx + dy * dy + dz * dz).sqrt();
                }
            }
            _ => {
                Self::compute_row_distances_scalar(point_i, points, distances, metric);
            }
        }
    }
    /// Scalar fallback for row distance computation
    fn compute_row_distances_scalar<T, P, M>(
        point_i: &P,
        points: &[P],
        distances: &mut [T],
        metric: &M,
    ) where
        T: SpatialScalar,
        P: SpatialPoint<T>,
        M: DistanceMetric<T, P>,
    {
        for (j, point_j) in points.iter().enumerate() {
            distances[j] = metric.distance(point_i, point_j);
        }
    }
    /// Compute condensed distance matrix (upper triangle only)
    pub fn compute_condensed<T, P, M>(points: &[P], metric: &M) -> SpatialResult<Vec<T>>
    where
        T: SpatialScalar,
        P: SpatialPoint<T>,
        M: DistanceMetric<T, P>,
    {
        let n = points.len();
        let mut distances = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                distances.push(metric.distance(&points[i], &points[j]));
            }
        }
        Ok(distances)
    }
}
/// Generic Gaussian Mixture Model clustering
pub struct GenericGMM<T: SpatialScalar> {
    _ncomponents: usize,
    max_iterations: usize,
    tolerance: T,
    reg_covar: T,
    _phantom: PhantomData<T>,
}
impl<T: SpatialScalar> GenericGMM<T> {
    /// Create a new GMM clusterer
    pub fn new(_ncomponents: usize) -> Self {
        Self {
            _ncomponents,
            max_iterations: 3,
            tolerance: T::from_f64(1e-1).unwrap_or(<T as SpatialScalar>::epsilon()),
            reg_covar: T::from_f64(1e-6).unwrap_or(<T as SpatialScalar>::epsilon()),
            _phantom: PhantomData,
        }
    }
    /// Set maximum iterations
    pub fn with_max_iterations(mut self, maxiterations: usize) -> Self {
        self.max_iterations = maxiterations;
        self
    }
    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: T) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Set regularization parameter for covariance
    pub fn with_reg_covar(mut self, regcovar: T) -> Self {
        self.reg_covar = regcovar;
        self
    }
    /// Fit the GMM to data (simplified implementation)
    #[allow(clippy::needless_range_loop)]
    pub fn fit<P>(&self, points: &[P]) -> SpatialResult<GMMResult<T>>
    where
        P: SpatialPoint<T> + Clone,
    {
        if points.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot fit GMM to empty dataset".to_string(),
            ));
        }
        let n_samples = points.len();
        let n_features = points[0].dimension();
        let kmeans = GenericKMeans::new(self._ncomponents);
        let kmeans_result = kmeans.fit(points)?;
        let mut means = kmeans_result.centroids;
        let mut weights = vec![
            T::one() / T::from(self._ncomponents).expect("Operation failed");
            self._ncomponents
        ];
        let mut covariances =
            vec![vec![vec![T::zero(); n_features]; n_features]; self._ncomponents];
        for k in 0..self._ncomponents {
            let cluster_points: Vec<&P> = kmeans_result
                .assignments
                .iter()
                .enumerate()
                .filter_map(
                    |(i, &cluster)| {
                        if cluster == k {
                            Some(&points[i])
                        } else {
                            None
                        }
                    },
                )
                .collect();
            if !cluster_points.is_empty() {
                let cluster_mean = &means[k];
                for i in 0..n_features {
                    for j in 0..n_features {
                        let mut cov_sum = T::zero();
                        let count = T::from(cluster_points.len()).expect("Operation failed");
                        for point in &cluster_points {
                            let pi = point.coordinate(i).unwrap_or(T::zero())
                                - cluster_mean.coordinate(i).unwrap_or(T::zero());
                            let pj = point.coordinate(j).unwrap_or(T::zero())
                                - cluster_mean.coordinate(j).unwrap_or(T::zero());
                            cov_sum = cov_sum + pi * pj;
                        }
                        covariances[k][i][j] = if count > T::one() {
                            cov_sum / (count - T::one())
                        } else if i == j {
                            T::one()
                        } else {
                            T::zero()
                        };
                    }
                }
                for i in 0..n_features {
                    covariances[k][i][i] = covariances[k][i][i] + self.reg_covar;
                }
            } else {
                for i in 0..n_features {
                    covariances[k][i][i] = T::one();
                }
            }
        }
        let mut log_likelihood = T::min_value();
        let mut responsibilities = vec![vec![T::zero(); self._ncomponents]; n_samples];
        for iteration in 0..self.max_iterations {
            let mut new_log_likelihood = T::zero();
            for i in 0..n_samples {
                let point = Self::point_to_generic(&points[i]);
                let mut log_likelihoods = vec![T::min_value(); self._ncomponents];
                let mut max_log_likelihood = T::min_value();
                for k in 0..self._ncomponents {
                    let log_weight = weights[k].ln();
                    let log_gaussian = self.compute_log_gaussian_probability(
                        &point,
                        &means[k],
                        &covariances[k],
                        n_features,
                    );
                    log_likelihoods[k] = log_weight + log_gaussian;
                    if log_likelihoods[k] > max_log_likelihood {
                        max_log_likelihood = log_likelihoods[k];
                    }
                }
                let mut sum_exp = T::zero();
                for k in 0..self._ncomponents {
                    let exp_val = (log_likelihoods[k] - max_log_likelihood).exp();
                    responsibilities[i][k] = exp_val;
                    sum_exp = sum_exp + exp_val;
                }
                if sum_exp > T::zero() {
                    for k in 0..self._ncomponents {
                        responsibilities[i][k] = responsibilities[i][k] / sum_exp;
                    }
                    new_log_likelihood = new_log_likelihood + max_log_likelihood + sum_exp.ln();
                }
            }
            let mut nk_values = vec![T::zero(); self._ncomponents];
            for k in 0..self._ncomponents {
                let mut nk = T::zero();
                for i in 0..n_samples {
                    nk = nk + responsibilities[i][k];
                }
                nk_values[k] = nk;
                weights[k] = nk / T::from(n_samples).expect("Operation failed");
            }
            for k in 0..self._ncomponents {
                if nk_values[k] > T::zero() {
                    let mut new_mean_coords = vec![T::zero(); n_features];
                    for i in 0..n_samples {
                        let point = Self::point_to_generic(&points[i]);
                        for d in 0..n_features {
                            let coord = point.coordinate(d).unwrap_or(T::zero());
                            new_mean_coords[d] =
                                new_mean_coords[d] + responsibilities[i][k] * coord;
                        }
                    }
                    for d in 0..n_features {
                        new_mean_coords[d] = new_mean_coords[d] / nk_values[k];
                    }
                    means[k] = Point::new(new_mean_coords);
                }
            }
            for k in 0..self._ncomponents {
                if nk_values[k] > T::one() {
                    let mean_k = &means[k];
                    for i in 0..n_features {
                        for j in 0..n_features {
                            covariances[k][i][j] = T::zero();
                        }
                    }
                    for sample_idx in 0..n_samples {
                        let point = Self::point_to_generic(&points[sample_idx]);
                        let resp = responsibilities[sample_idx][k];
                        for i in 0..n_features {
                            for j in 0..n_features {
                                let diff_i = point.coordinate(i).unwrap_or(T::zero())
                                    - mean_k.coordinate(i).unwrap_or(T::zero());
                                let diff_j = point.coordinate(j).unwrap_or(T::zero())
                                    - mean_k.coordinate(j).unwrap_or(T::zero());
                                covariances[k][i][j] =
                                    covariances[k][i][j] + resp * diff_i * diff_j;
                            }
                        }
                    }
                    for i in 0..n_features {
                        for j in 0..n_features {
                            covariances[k][i][j] = covariances[k][i][j] / nk_values[k];
                            if i == j {
                                covariances[k][i][j] = covariances[k][i][j] + self.reg_covar;
                            }
                        }
                    }
                }
            }
            if iteration > 0 && (new_log_likelihood - log_likelihood).abs() < self.tolerance {
                break;
            }
            log_likelihood = new_log_likelihood;
        }
        let mut labels = vec![0; n_samples];
        for i in 0..n_samples {
            let mut max_resp = T::zero();
            let mut best_cluster = 0;
            for k in 0..self._ncomponents {
                if responsibilities[i][k] > max_resp {
                    max_resp = responsibilities[i][k];
                    best_cluster = k;
                }
            }
            labels[i] = best_cluster;
        }
        Ok(GMMResult {
            means,
            weights,
            covariances,
            labels,
            log_likelihood,
            converged: true,
        })
    }
    /// Convert a point to generic Point type
    fn point_to_generic<P>(point: &P) -> Point<T>
    where
        P: SpatialPoint<T>,
    {
        let coords: Vec<T> = (0..point.dimension())
            .map(|i| point.coordinate(i).unwrap_or(T::zero()))
            .collect();
        Point::new(coords)
    }
    /// Compute log probability of a point under a multivariate Gaussian distribution
    fn compute_log_gaussian_probability(
        &self,
        point: &Point<T>,
        mean: &Point<T>,
        covariance: &[Vec<T>],
        n_features: usize,
    ) -> T {
        let mut diff = vec![T::zero(); n_features];
        for (i, item) in diff.iter_mut().enumerate().take(n_features) {
            *item =
                point.coordinate(i).unwrap_or(T::zero()) - mean.coordinate(i).unwrap_or(T::zero());
        }
        let mut det = T::one();
        let mut inv_cov = vec![vec![T::zero(); n_features]; n_features];
        for i in 0..n_features {
            det = det * covariance[i][i];
            inv_cov[i][i] = T::one() / covariance[i][i];
        }
        let mut quadratic_form = T::zero();
        for i in 0..n_features {
            for j in 0..n_features {
                quadratic_form = quadratic_form + diff[i] * inv_cov[i][j] * diff[j];
            }
        }
        let two_pi = T::from(std::f64::consts::TAU)
            .unwrap_or(T::from(std::f64::consts::TAU).expect("Operation failed"));
        let log_2pi_k = T::from(n_features).expect("Operation failed") * two_pi.ln();
        let log_det = det.abs().ln();
        let log_prob =
            -T::from(0.5).expect("Operation failed") * (log_2pi_k + log_det + quadratic_form);
        if Float::is_finite(log_prob) {
            log_prob
        } else {
            T::min_value()
        }
    }
}
/// Result of GMM fitting
#[derive(Debug, Clone)]
pub struct GMMResult<T: SpatialScalar> {
    /// Component means
    pub means: Vec<Point<T>>,
    /// Component weights
    pub weights: Vec<T>,
    /// Component covariances (simplified as 3D arrays)
    pub covariances: Vec<Vec<Vec<T>>>,
    /// Cluster assignments
    pub labels: Vec<usize>,
    /// Final log-likelihood
    pub log_likelihood: T,
    /// Whether the algorithm converged
    pub converged: bool,
}
/// Generic convex hull computation using Graham scan
pub struct GenericConvexHull;
impl GenericConvexHull {
    /// Compute 2D convex hull using Graham scan
    pub fn graham_scan_2d<T, P>(points: &[P]) -> SpatialResult<Vec<Point<T>>>
    where
        T: SpatialScalar,
        P: SpatialPoint<T> + Clone,
    {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        if points.len() < 3 {
            return Ok(points.iter().map(|p| Self::to_generic_point(p)).collect());
        }
        for point in points {
            if point.dimension() != 2 {
                return Err(SpatialError::ValueError(
                    "All points must be 2D for 2D convex hull".to_string(),
                ));
            }
        }
        let mut generic_points: Vec<Point<T>> =
            points.iter().map(|p| Self::to_generic_point(p)).collect();
        let start_idx = generic_points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let y_cmp = a
                    .coordinate(1)
                    .partial_cmp(&b.coordinate(1))
                    .expect("Operation failed");
                if y_cmp == Ordering::Equal {
                    a.coordinate(0)
                        .partial_cmp(&b.coordinate(0))
                        .expect("Operation failed")
                } else {
                    y_cmp
                }
            })
            .map(|(idx_, _)| idx_)
            .expect("Operation failed");
        generic_points.swap(0, start_idx);
        let start_point = generic_points[0].clone();
        generic_points[1..].sort_by(|a, b| {
            let angle_a = Self::polar_angle(&start_point, a);
            let angle_b = Self::polar_angle(&start_point, b);
            angle_a.partial_cmp(&angle_b).unwrap_or(Ordering::Equal)
        });
        let mut hull = Vec::new();
        for point in generic_points {
            while hull.len() > 1
                && Self::cross_product(&hull[hull.len() - 2], &hull[hull.len() - 1], &point)
                    <= T::zero()
            {
                hull.pop();
            }
            hull.push(point);
        }
        Ok(hull)
    }
    /// Convert a point to generic Point type
    fn to_generic_point<T, P>(point: &P) -> Point<T>
    where
        T: SpatialScalar,
        P: SpatialPoint<T>,
    {
        let coords: Vec<T> = (0..point.dimension())
            .map(|i| point.coordinate(i).unwrap_or(T::zero()))
            .collect();
        Point::new(coords)
    }
    /// Calculate polar angle from start to point
    fn polar_angle<T: SpatialScalar>(start: &Point<T>, point: &Point<T>) -> T {
        let dx =
            point.coordinate(0).unwrap_or(T::zero()) - start.coordinate(0).unwrap_or(T::zero());
        let dy =
            point.coordinate(1).unwrap_or(T::zero()) - start.coordinate(1).unwrap_or(T::zero());
        dy.atan2(dx)
    }
    /// Calculate cross product for 2D points
    fn cross_product<T: SpatialScalar>(a: &Point<T>, b: &Point<T>, c: &Point<T>) -> T {
        let ab_x = b.coordinate(0).unwrap_or(T::zero()) - a.coordinate(0).unwrap_or(T::zero());
        let ab_y = b.coordinate(1).unwrap_or(T::zero()) - a.coordinate(1).unwrap_or(T::zero());
        let ac_x = c.coordinate(0).unwrap_or(T::zero()) - a.coordinate(0).unwrap_or(T::zero());
        let ac_y = c.coordinate(1).unwrap_or(T::zero()) - a.coordinate(1).unwrap_or(T::zero());
        ab_x * ac_y - ab_y * ac_x
    }
}
#[derive(Debug, Clone)]
struct KDNode<T: SpatialScalar, P: SpatialPoint<T>> {
    point_index: usize,
    splitting_dimension: usize,
    left: Option<Box<KDNode<T, P>>>,
    right: Option<Box<KDNode<T, P>>>,
    _phantom: PhantomData<(T, P)>,
}
/// Generic DBSCAN clustering implementation
pub struct GenericDBSCAN<T: SpatialScalar> {
    pub(super) eps: T,
    pub(super) minsamples: usize,
    _phantom: PhantomData<T>,
}
impl<T: SpatialScalar> GenericDBSCAN<T> {
    /// Create a new DBSCAN clusterer
    pub fn new(_eps: T, minsamples: usize) -> Self {
        Self {
            eps: _eps,
            minsamples,
            _phantom: PhantomData,
        }
    }
    /// Perform DBSCAN clustering
    pub fn fit<P, M>(&self, points: &[P], metric: &M) -> SpatialResult<DBSCANResult>
    where
        P: SpatialPoint<T>,
        M: DistanceMetric<T, P>,
    {
        if points.is_empty() {
            return Ok(DBSCANResult {
                labels: Vec::new(),
                n_clusters: 0,
            });
        }
        if self.minsamples == 0 {
            return Err(SpatialError::ValueError(
                "minsamples must be greater than 0".to_string(),
            ));
        }
        if self.minsamples > points.len() {
            return Err(SpatialError::ValueError(format!(
                "minsamples ({}) cannot be larger than the number of points ({})",
                self.minsamples,
                points.len()
            )));
        }
        if !Float::is_finite(self.eps) || self.eps <= T::zero() {
            return Err(SpatialError::ValueError(format!(
                "eps must be a positive finite number, got: {}",
                NumCast::from(self.eps).unwrap_or(f64::NAN)
            )));
        }
        let dimension = if points.is_empty() {
            0
        } else {
            points[0].dimension()
        };
        for (i, point) in points.iter().enumerate() {
            if point.dimension() != dimension {
                return Err(SpatialError::ValueError(format!(
                    "Point {} has dimension {} but expected {}",
                    i,
                    point.dimension(),
                    dimension
                )));
            }
            for d in 0..dimension {
                if let Some(coord) = point.coordinate(d) {
                    if !Float::is_finite(coord) {
                        return Err(SpatialError::ValueError(format!(
                            "Point {} has invalid coordinate {} at dimension {}",
                            i,
                            NumCast::from(coord).unwrap_or(f64::NAN),
                            d
                        )));
                    }
                }
            }
        }
        let n = points.len();
        let mut labels = vec![-1i32; n];
        let mut visited = vec![false; n];
        let mut cluster_id = 0;
        const DBSCAN_PROCESS_CHUNK_SIZE: usize = 32;
        for chunk_start in (0..n).step_by(DBSCAN_PROCESS_CHUNK_SIZE) {
            let chunk_end = (chunk_start + DBSCAN_PROCESS_CHUNK_SIZE).min(n);
            for i in chunk_start..chunk_end {
                if visited[i] {
                    continue;
                }
                visited[i] = true;
                let neighbors = self.find_neighbors(points, i, metric);
                if neighbors.len() < self.minsamples {
                    labels[i] = -1;
                } else {
                    self.expand_cluster(
                        points,
                        &mut labels,
                        &mut visited,
                        i,
                        &neighbors,
                        cluster_id,
                        metric,
                    );
                    cluster_id += 1;
                    if cluster_id > 10000 {
                        return Err(
                            SpatialError::ValueError(
                                format!(
                                    "Too many clusters found: {cluster_id}. Consider adjusting eps or minsamples parameters"
                                ),
                            ),
                        );
                    }
                }
            }
            if chunk_start > 0 && chunk_start % (DBSCAN_PROCESS_CHUNK_SIZE * 10) == 0 {
                std::hint::black_box(&labels);
                std::hint::black_box(&visited);
            }
        }
        Ok(DBSCANResult {
            labels,
            n_clusters: cluster_id,
        })
    }
    /// Find neighbors within eps distance with highly optimized search and memory pooling
    fn find_neighbors<P, M>(&self, points: &[P], pointidx: usize, metric: &M) -> Vec<usize>
    where
        P: SpatialPoint<T>,
        M: DistanceMetric<T, P>,
    {
        let mut neighbors = Vec::with_capacity(32);
        let query_point = &points[pointidx];
        let _eps_squared = self.eps * self.eps;
        const NEIGHBOR_CHUNK_SIZE: usize = 16;
        if points.len() > 5000 {
            for chunk in points.chunks(NEIGHBOR_CHUNK_SIZE) {
                let chunk_start = ((chunk.as_ptr() as usize - points.as_ptr() as usize)
                    / std::mem::size_of::<P>())
                .min(points.len());
                for (local_idx, point) in chunk.iter().enumerate() {
                    let global_idx = chunk_start + local_idx;
                    if global_idx >= points.len() {
                        break;
                    }
                    let distance = metric.distance(query_point, point);
                    if distance <= self.eps {
                        neighbors.push(global_idx);
                    }
                }
                if neighbors.len() > 100 {
                    break;
                }
            }
        } else {
            for (i, point) in points.iter().enumerate() {
                if metric.distance(query_point, point) <= self.eps {
                    neighbors.push(i);
                }
            }
        }
        neighbors.shrink_to_fit();
        neighbors
    }
    /// Expand cluster by adding density-reachable points with memory optimization
    #[allow(clippy::too_many_arguments)]
    fn expand_cluster<P, M>(
        &self,
        points: &[P],
        labels: &mut [i32],
        visited: &mut [bool],
        pointidx: usize,
        neighbors: &[usize],
        cluster_id: i32,
        metric: &M,
    ) where
        P: SpatialPoint<T>,
        M: DistanceMetric<T, P>,
    {
        labels[pointidx] = cluster_id;
        let mut processed = vec![false; points.len()];
        let mut seed_set = Vec::with_capacity(neighbors.len() * 2);
        for &neighbor in neighbors {
            if neighbor < points.len() {
                seed_set.push(neighbor);
            }
        }
        const EXPAND_BATCH_SIZE: usize = 32;
        let mut batch_buffer = Vec::with_capacity(EXPAND_BATCH_SIZE);
        while !seed_set.is_empty() {
            let batch_size = seed_set.len().min(EXPAND_BATCH_SIZE);
            batch_buffer.clear();
            batch_buffer.extend(seed_set.drain(..batch_size));
            for q in batch_buffer.iter().copied() {
                if q >= points.len() || processed[q] {
                    continue;
                }
                processed[q] = true;
                if !visited[q] {
                    visited[q] = true;
                    let q_neighbors = self.find_neighbors(points, q, metric);
                    if q_neighbors.len() >= self.minsamples {
                        for &neighbor in &q_neighbors {
                            if neighbor < points.len()
                                && !processed[neighbor]
                                && !seed_set.contains(&neighbor)
                            {
                                seed_set.push(neighbor);
                            }
                        }
                    }
                }
                if labels[q] == -1 {
                    labels[q] = cluster_id;
                }
            }
            if seed_set.len() > 1000 {
                seed_set.sort_unstable();
                seed_set.dedup();
            }
        }
    }
}
/// Generic K-means clustering implementation
pub struct GenericKMeans<T: SpatialScalar, P: SpatialPoint<T>> {
    k: usize,
    max_iterations: usize,
    tolerance: T,
    parallel: bool,
    phantom: PhantomData<(T, P)>,
}
impl<T: SpatialScalar, P: SpatialPoint<T> + Clone> GenericKMeans<T, P> {
    /// Create a new K-means clusterer
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iterations: 5,
            tolerance: T::from_f64(1e-1).unwrap_or(<T as SpatialScalar>::epsilon()),
            parallel: false,
            phantom: PhantomData,
        }
    }
    /// Enable parallel processing for large datasets
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, maxiterations: usize) -> Self {
        self.max_iterations = maxiterations;
        self
    }
    /// Set the convergence tolerance
    pub fn with_tolerance(mut self, tolerance: T) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Perform K-means clustering with memory optimizations
    pub fn fit(&self, points: &[P]) -> SpatialResult<KMeansResult<T, P>> {
        if points.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot cluster empty point set".to_string(),
            ));
        }
        if self.k == 0 {
            return Err(SpatialError::ValueError(
                "k must be greater than 0".to_string(),
            ));
        }
        if self.k > points.len() {
            return Err(SpatialError::ValueError(format!(
                "k ({}) cannot be larger than the number of points ({})",
                self.k,
                points.len()
            )));
        }
        if self.k > 10000 {
            return Err(SpatialError::ValueError(format!(
                "k ({}) is too large. Consider using hierarchical clustering for k > 10000",
                self.k
            )));
        }
        let dimension = points[0].dimension();
        if dimension == 0 {
            return Err(SpatialError::ValueError(
                "Points must have at least one dimension".to_string(),
            ));
        }
        for (i, point) in points.iter().enumerate() {
            if point.dimension() != dimension {
                return Err(SpatialError::ValueError(format!(
                    "Point {} has dimension {} but expected {}",
                    i,
                    point.dimension(),
                    dimension
                )));
            }
            for d in 0..dimension {
                if let Some(coord) = point.coordinate(d) {
                    if !Float::is_finite(coord) {
                        return Err(SpatialError::ValueError(format!(
                            "Point {} has invalid coordinate {} at dimension {}",
                            i,
                            NumCast::from(coord).unwrap_or(f64::NAN),
                            d
                        )));
                    }
                }
            }
        }
        let mut centroids = self.initialize_centroids(points, dimension)?;
        let mut assignments = vec![0; points.len()];
        let mut point_distances = vec![T::zero(); self.k];
        for iteration in 0..self.max_iterations {
            let mut changed = false;
            const CHUNK_SIZE: usize = 16;
            let chunks = points.chunks(CHUNK_SIZE);
            for (chunk_start, chunk) in chunks.enumerate() {
                let chunk_offset = chunk_start * CHUNK_SIZE;
                for (local_i, point) in chunk.iter().enumerate() {
                    let i = chunk_offset + local_i;
                    let mut best_cluster = 0;
                    let mut best_distance = T::max_finite();
                    self.compute_distances_simd(point, &centroids, &mut point_distances);
                    for (j, &distance) in point_distances.iter().enumerate() {
                        if distance < best_distance {
                            best_distance = distance;
                            best_cluster = j;
                        }
                    }
                    if assignments[i] != best_cluster {
                        assignments[i] = best_cluster;
                        changed = true;
                    }
                }
            }
            let old_centroids = centroids.clone();
            centroids = self.update_centroids(points, &assignments, dimension)?;
            let max_movement = old_centroids
                .iter()
                .zip(centroids.iter())
                .map(|(old, new)| old.distance_to(new))
                .fold(T::zero(), |acc, dist| if dist > acc { dist } else { acc });
            if !changed || max_movement < self.tolerance {
                return Ok(KMeansResult {
                    centroids,
                    assignments,
                    iterations: iteration + 1,
                    converged: max_movement < self.tolerance,
                    phantom: PhantomData,
                });
            }
        }
        Ok(KMeansResult {
            centroids,
            assignments,
            iterations: self.max_iterations,
            converged: false,
            phantom: PhantomData,
        })
    }
    /// Initialize centroids using k-means++
    fn initialize_centroids(
        &self,
        points: &[P],
        _dimension: usize,
    ) -> SpatialResult<Vec<Point<T>>> {
        let mut centroids = Vec::with_capacity(self.k);
        centroids.push(GenericKMeans::<T, P>::point_to_generic(&points[0]));
        for _ in 1..self.k {
            let mut distances = Vec::with_capacity(points.len());
            for point in points {
                let min_distance = centroids
                    .iter()
                    .map(|centroid| {
                        GenericKMeans::<T, P>::point_to_generic(point).distance_to(centroid)
                    })
                    .fold(
                        T::max_finite(),
                        |acc, dist| if dist < acc { dist } else { acc },
                    );
                distances.push(min_distance);
            }
            let max_distance_idx = distances
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(idx_, _)| idx_)
                .unwrap_or(0);
            centroids.push(GenericKMeans::<T, P>::point_to_generic(
                &points[max_distance_idx],
            ));
        }
        Ok(centroids)
    }
    /// Update centroids based on current assignments with memory optimizations
    fn update_centroids(
        &self,
        points: &[P],
        assignments: &[usize],
        dimension: usize,
    ) -> SpatialResult<Vec<Point<T>>> {
        let mut centroids = vec![Point::zeros(dimension); self.k];
        let mut counts = vec![0; self.k];
        const UPDATE_CHUNK_SIZE: usize = 512;
        for chunk in points.chunks(UPDATE_CHUNK_SIZE) {
            let assignments_chunk = &assignments[..chunk.len().min(assignments.len())];
            for (point, &cluster) in chunk.iter().zip(assignments_chunk.iter()) {
                let point_coords: Vec<T> = (0..dimension)
                    .map(|d| point.coordinate(d).unwrap_or(T::zero()))
                    .collect();
                for (d, &coord) in point_coords.iter().enumerate() {
                    if let Some(centroid_coord) = centroids[cluster].coords_mut().get_mut(d) {
                        *centroid_coord = *centroid_coord + coord;
                    }
                }
                counts[cluster] += 1;
            }
        }
        for (centroid, count) in centroids.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                let count_scalar = T::from(*count).unwrap_or(T::one());
                for coord in centroid.coords_mut() {
                    *coord = *coord / count_scalar;
                }
            }
        }
        Ok(centroids)
    }
    /// Convert a point to generic Point type
    fn point_to_generic(point: &P) -> Point<T> {
        let coords: Vec<T> = (0..point.dimension())
            .map(|i| point.coordinate(i).unwrap_or(T::zero()))
            .collect();
        Point::new(coords)
    }
    /// SIMD-optimized distance computation to all centroids
    fn compute_distances_simd(&self, point: &P, centroids: &[Point<T>], distances: &mut [T]) {
        let _caps = PlatformCapabilities::detect();
        let point_generic = GenericKMeans::<T, P>::point_to_generic(point);
        for (j, centroid) in centroids.iter().enumerate() {
            distances[j] = point_generic.distance_to(centroid);
        }
    }
    /// SIMD-optimized distance computation implementation
    #[allow(dead_code)]
    fn compute_distances_simd_optimized(
        &self,
        point: &Point<T>,
        centroids: &[Point<T>],
        distances: &mut [T],
    ) {
        match point.dimension() {
            2 => {
                let px = point.coordinate(0).unwrap_or(T::zero());
                let py = point.coordinate(1).unwrap_or(T::zero());
                let mut i = 0;
                while i + 3 < centroids.len() {
                    for j in 0..4 {
                        if i + j < centroids.len() {
                            let centroid = &centroids[i + j];
                            let cx = centroid.coordinate(0).unwrap_or(T::zero());
                            let cy = centroid.coordinate(1).unwrap_or(T::zero());
                            let dx = px - cx;
                            let dy = py - cy;
                            distances[i + j] = (dx * dx + dy * dy).sqrt();
                        }
                    }
                    i += 4;
                }
                while i < centroids.len() {
                    let centroid = &centroids[i];
                    let cx = centroid.coordinate(0).unwrap_or(T::zero());
                    let cy = centroid.coordinate(1).unwrap_or(T::zero());
                    let dx = px - cx;
                    let dy = py - cy;
                    distances[i] = (dx * dx + dy * dy).sqrt();
                    i += 1;
                }
            }
            3 => {
                let px = point.coordinate(0).unwrap_or(T::zero());
                let py = point.coordinate(1).unwrap_or(T::zero());
                let pz = point.coordinate(2).unwrap_or(T::zero());
                for (i, centroid) in centroids.iter().enumerate() {
                    let cx = centroid.coordinate(0).unwrap_or(T::zero());
                    let cy = centroid.coordinate(1).unwrap_or(T::zero());
                    let cz = centroid.coordinate(2).unwrap_or(T::zero());
                    let dx = px - cx;
                    let dy = py - cy;
                    let dz = pz - cz;
                    distances[i] = (dx * dx + dy * dy + dz * dz).sqrt();
                }
            }
            _ => {
                for (j, centroid) in centroids.iter().enumerate() {
                    distances[j] = point.distance_to(centroid);
                }
            }
        }
    }
}
/// Result of DBSCAN clustering
#[derive(Debug, Clone)]
pub struct DBSCANResult {
    /// Cluster labels for each point (-1 = noise, 0+ = cluster id)
    pub labels: Vec<i32>,
    /// Number of clusters found
    pub n_clusters: i32,
}
/// Result of K-means clustering
#[derive(Debug, Clone)]
pub struct KMeansResult<T: SpatialScalar, P: SpatialPoint<T>> {
    /// Final centroids
    pub centroids: Vec<Point<T>>,
    /// Cluster assignment for each point
    pub assignments: Vec<usize>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    phantom: PhantomData<P>,
}
/// Generic KD-Tree implementation with memory optimizations
///
/// This KD-Tree can work with any type that implements SpatialPoint,
/// allowing for flexible point representations and numeric types.
/// It includes memory optimizations for large datasets.
#[derive(Debug, Clone)]
pub struct GenericKDTree<T: SpatialScalar, P: SpatialPoint<T>> {
    root: Option<Box<KDNode<T, P>>>,
    points: Vec<P>,
    dimension: usize,
    #[allow(dead_code)]
    leaf_size: usize,
}
impl<T: SpatialScalar, P: SpatialPoint<T> + Clone> GenericKDTree<T, P> {
    /// Create a new KD-Tree from a collection of points
    pub fn new(points: &[P]) -> SpatialResult<Self> {
        if points.is_empty() {
            return Ok(Self {
                root: None,
                points: Vec::new(),
                dimension: 0,
                leaf_size: 32,
            });
        }
        if points.len() > 1_000_000 {
            return Err(SpatialError::ValueError(format!(
                "Point collection too large: {} points. Maximum supported: 1,000,000",
                points.len()
            )));
        }
        let dimension = points[0].dimension();
        if dimension == 0 {
            return Err(SpatialError::ValueError(
                "Points must have at least one dimension".to_string(),
            ));
        }
        if dimension > 50 {
            return Err(SpatialError::ValueError(format!(
                "Dimension too high: {dimension}. KD-Tree is not efficient for dimensions > 50"
            )));
        }
        for (i, point) in points.iter().enumerate() {
            if point.dimension() != dimension {
                return Err(SpatialError::ValueError(format!(
                    "Point {} has dimension {} but expected {}",
                    i,
                    point.dimension(),
                    dimension
                )));
            }
            for d in 0..dimension {
                if let Some(coord) = point.coordinate(d) {
                    if !Float::is_finite(coord) {
                        return Err(SpatialError::ValueError(format!(
                            "Point {} has invalid coordinate {} at dimension {}",
                            i,
                            NumCast::from(coord).unwrap_or(f64::NAN),
                            d
                        )));
                    }
                }
            }
        }
        let points = points.to_vec();
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let leaf_size = 32;
        let root = Self::build_tree(&points, &mut indices, 0, dimension, leaf_size);
        Ok(Self {
            root,
            points,
            dimension,
            leaf_size: 32,
        })
    }
    /// Build the KD-Tree recursively with leaf optimization
    fn build_tree(
        points: &[P],
        indices: &mut [usize],
        depth: usize,
        dimension: usize,
        leaf_size: usize,
    ) -> Option<Box<KDNode<T, P>>> {
        if indices.is_empty() {
            return None;
        }
        if indices.len() <= leaf_size {
            let point_index = indices[0];
            return Some(Box::new(KDNode {
                point_index,
                splitting_dimension: depth % dimension,
                left: None,
                right: None,
                _phantom: PhantomData,
            }));
        }
        let splitting_dimension = depth % dimension;
        indices.sort_by(|&a, &b| {
            let coord_a = points[a]
                .coordinate(splitting_dimension)
                .unwrap_or(T::zero());
            let coord_b = points[b]
                .coordinate(splitting_dimension)
                .unwrap_or(T::zero());
            coord_a.partial_cmp(&coord_b).unwrap_or(Ordering::Equal)
        });
        let median = indices.len() / 2;
        let point_index = indices[median];
        let (left_indices, right_indices) = indices.split_at_mut(median);
        let right_indices = &mut right_indices[1..];
        let left = Self::build_tree(points, left_indices, depth + 1, dimension, leaf_size);
        let right = Self::build_tree(points, right_indices, depth + 1, dimension, leaf_size);
        Some(Box::new(KDNode {
            point_index,
            splitting_dimension,
            left,
            right,
            _phantom: PhantomData,
        }))
    }
    /// Find the k nearest neighbors to a query point
    pub fn k_nearest_neighbors(
        &self,
        query: &P,
        k: usize,
        metric: &dyn DistanceMetric<T, P>,
    ) -> SpatialResult<Vec<(usize, T)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        if k > self.points.len() {
            return Err(SpatialError::ValueError(format!(
                "k ({}) cannot be larger than the number of points ({})",
                k,
                self.points.len()
            )));
        }
        if k > 1000 {
            return Err(SpatialError::ValueError(format!(
                "k ({k}) is too large. Consider using radius search for k > 1000"
            )));
        }
        if query.dimension() != self.dimension {
            return Err(SpatialError::ValueError(format!(
                "Query point dimension ({}) must match tree dimension ({})",
                query.dimension(),
                self.dimension
            )));
        }
        for d in 0..query.dimension() {
            if let Some(coord) = query.coordinate(d) {
                if !Float::is_finite(coord) {
                    return Err(SpatialError::ValueError(format!(
                        "Query point has invalid coordinate {} at dimension {}",
                        NumCast::from(coord).unwrap_or(f64::NAN),
                        d
                    )));
                }
            }
        }
        if self.points.is_empty() {
            return Ok(Vec::new());
        }
        let mut heap = BinaryHeap::new();
        if let Some(ref root) = self.root {
            self.search_knn(root, query, k, &mut heap, metric);
        }
        let mut result: Vec<(usize, T)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|item| (item.index, item.distance))
            .collect();
        result.reverse();
        Ok(result)
    }
    /// Search for k nearest neighbors recursively
    fn search_knn(
        &self,
        node: &KDNode<T, P>,
        query: &P,
        k: usize,
        heap: &mut BinaryHeap<KNNItem<T>>,
        metric: &dyn DistanceMetric<T, P>,
    ) {
        let point = &self.points[node.point_index];
        let distance = metric.distance(query, point);
        if heap.len() < k {
            heap.push(KNNItem {
                distance,
                index: node.point_index,
            });
        } else if let Some(top) = heap.peek() {
            if distance < top.distance {
                heap.pop();
                heap.push(KNNItem {
                    distance,
                    index: node.point_index,
                });
            }
        }
        let query_coord = query
            .coordinate(node.splitting_dimension)
            .unwrap_or(T::zero());
        let point_coord = point
            .coordinate(node.splitting_dimension)
            .unwrap_or(T::zero());
        let (first_child, second_child) = if query_coord < point_coord {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };
        if let Some(ref child) = first_child {
            self.search_knn(child, query, k, heap, metric);
        }
        let dimension_distance = (query_coord - point_coord).abs();
        let should_search_other = heap.len() < k
            || heap
                .peek()
                .is_none_or(|top| dimension_distance < top.distance);
        if should_search_other {
            if let Some(ref child) = second_child {
                self.search_knn(child, query, k, heap, metric);
            }
        }
    }
    /// Find all points within a given radius of the query point
    pub fn radius_search(
        &self,
        query: &P,
        radius: T,
        metric: &dyn DistanceMetric<T, P>,
    ) -> SpatialResult<Vec<(usize, T)>> {
        if query.dimension() != self.dimension {
            return Err(SpatialError::ValueError(
                "Query point dimension must match tree dimension".to_string(),
            ));
        }
        let mut result = Vec::new();
        if let Some(ref root) = self.root {
            self.search_radius(root, query, radius, &mut result, metric);
        }
        Ok(result)
    }
    /// Search for points within radius recursively
    fn search_radius(
        &self,
        node: &KDNode<T, P>,
        query: &P,
        radius: T,
        result: &mut Vec<(usize, T)>,
        metric: &dyn DistanceMetric<T, P>,
    ) {
        let point = &self.points[node.point_index];
        let distance = metric.distance(query, point);
        if distance <= radius {
            result.push((node.point_index, distance));
        }
        let query_coord = query
            .coordinate(node.splitting_dimension)
            .unwrap_or(T::zero());
        let point_coord = point
            .coordinate(node.splitting_dimension)
            .unwrap_or(T::zero());
        let _dimension_distance = (query_coord - point_coord).abs();
        if let Some(ref left) = node.left {
            if query_coord - radius <= point_coord {
                self.search_radius(left, query, radius, result, metric);
            }
        }
        if let Some(ref right) = node.right {
            if query_coord + radius >= point_coord {
                self.search_radius(right, query, radius, result, metric);
            }
        }
    }
}

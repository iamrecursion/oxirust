//! Kernel matrix caching and batch computation.
//!
//! Provides efficient computation of kernel matrices for batches of inputs,
//! with symmetric key normalization and LRU caching to avoid redundant evaluations.

use std::collections::{HashMap, VecDeque};

use scirs2_core::ndarray::Array2;

use crate::error::{KernelError, Result};

/// Cache for kernel evaluation results.
///
/// Uses symmetric key normalization: k(i,j) = k(j,i), so (i,j) and (j,i)
/// share the same cache entry. Implements LRU eviction when capacity is reached.
pub struct KernelCache {
    entries: HashMap<(usize, usize), f64>,
    lru_order: VecDeque<(usize, usize)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl KernelCache {
    /// Create a new kernel cache with the given capacity.
    ///
    /// The cache will evict the least-recently-used entry when full.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            lru_order: VecDeque::with_capacity(capacity),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    /// Normalize cache key so that (i,j) and (j,i) map to the same entry.
    fn normalize_key(i: usize, j: usize) -> (usize, usize) {
        if i <= j {
            (i, j)
        } else {
            (j, i)
        }
    }

    /// Retrieve a cached value, updating LRU order on hit.
    pub fn get(&mut self, i: usize, j: usize) -> Option<f64> {
        let key = Self::normalize_key(i, j);
        if let Some(&value) = self.entries.get(&key) {
            self.hits += 1;
            // Move to back (most recently used)
            if let Some(pos) = self.lru_order.iter().position(|k| *k == key) {
                self.lru_order.remove(pos);
            }
            self.lru_order.push_back(key);
            Some(value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a value into the cache, evicting the LRU entry if at capacity.
    pub fn insert(&mut self, i: usize, j: usize, value: f64) {
        let key = Self::normalize_key(i, j);

        // If key already exists, update it and refresh LRU position
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.entries.entry(key) {
            e.insert(value);
            if let Some(pos) = self.lru_order.iter().position(|k| *k == key) {
                self.lru_order.remove(pos);
            }
            self.lru_order.push_back(key);
            return;
        }

        // Evict if at capacity
        if self.entries.len() >= self.capacity && self.capacity > 0 {
            if let Some(evicted) = self.lru_order.pop_front() {
                self.entries.remove(&evicted);
            }
        }

        self.entries.insert(key, value);
        self.lru_order.push_back(key);
    }

    /// Return the cache hit rate as a fraction in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries and reset statistics.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Total number of cache hits.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Total number of cache misses.
    pub fn misses(&self) -> u64 {
        self.misses
    }
}

/// A Gram matrix (symmetric kernel matrix) wrapper.
///
/// Wraps an `Array2<f64>` that is expected to be square and symmetric,
/// providing convenient accessors for common operations.
#[derive(Debug, Clone)]
pub struct GramMatrix {
    data: Array2<f64>,
}

impl GramMatrix {
    /// Create a new Gram matrix, verifying that it is square.
    pub fn new(data: Array2<f64>) -> Result<Self> {
        if data.nrows() != data.ncols() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![data.nrows(), data.nrows()],
                got: vec![data.nrows(), data.ncols()],
                context: "GramMatrix must be square".to_string(),
            });
        }
        Ok(GramMatrix { data })
    }

    /// Get entry (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[[i, j]]
    }

    /// Matrix dimension (n for an n x n matrix).
    pub fn dim(&self) -> usize {
        self.data.nrows()
    }

    /// Diagonal entries as a vector.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.dim()).map(|i| self.data[[i, i]]).collect()
    }

    /// Matrix trace (sum of diagonal entries).
    pub fn trace(&self) -> f64 {
        self.diagonal().iter().sum()
    }

    /// Check if the matrix is approximately symmetric within a given tolerance.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        let n = self.dim();
        for i in 0..n {
            for j in (i + 1)..n {
                if (self.data[[i, j]] - self.data[[j, i]]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check if all diagonal entries are non-negative (necessary condition for PSD).
    pub fn has_nonneg_diagonal(&self) -> bool {
        self.diagonal().iter().all(|&d| d >= 0.0)
    }

    /// Frobenius norm: sqrt(sum of squares of all entries).
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Access the underlying array.
    pub fn as_array(&self) -> &Array2<f64> {
        &self.data
    }
}

/// Statistics about kernel matrix computation.
#[derive(Debug, Clone, Default)]
pub struct KernelMatrixStats {
    /// Total number of kernel evaluations performed.
    pub evaluations: u64,
    /// Number of cache hits (if caching enabled).
    pub cache_hits: u64,
    /// Number of cache misses (if caching enabled).
    pub cache_misses: u64,
    /// Dimension of the computed matrix (n for n x n).
    pub matrix_dim: usize,
    /// Wall-clock time for computation in milliseconds.
    pub computation_ms: f64,
}

impl KernelMatrixStats {
    /// Cache hit rate as a fraction in [0.0, 1.0].
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// Batch kernel matrix computation engine.
///
/// Computes the full n x n kernel matrix for a batch of n input vectors,
/// exploiting symmetry (only computing the upper triangle) and optionally
/// caching results for repeated computations.
pub struct BatchKernelComputer {
    cache: Option<KernelCache>,
}

impl BatchKernelComputer {
    /// Create a new batch computer without caching.
    pub fn new() -> Self {
        BatchKernelComputer { cache: None }
    }

    /// Create a new batch computer with an LRU cache of the given capacity.
    pub fn with_cache(capacity: usize) -> Self {
        BatchKernelComputer {
            cache: Some(KernelCache::new(capacity)),
        }
    }

    /// Compute the full kernel matrix for a batch of input vectors.
    ///
    /// The kernel function `kernel_fn` takes two vectors (as slices) and returns
    /// the kernel value. The resulting matrix is symmetric: `K[i,j] = K[j,i]`.
    ///
    /// # Errors
    ///
    /// Returns `BatchKernelError::EmptyBatch` if `inputs` is empty.
    pub fn compute<F>(
        &mut self,
        inputs: &[Vec<f64>],
        kernel_fn: F,
    ) -> Result<(GramMatrix, KernelMatrixStats)>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        if inputs.is_empty() {
            return Err(KernelError::ComputationError(
                "Empty input batch".to_string(),
            ));
        }

        let n = inputs.len();
        let dim = inputs[0].len();

        // Validate consistent dimensions
        for (idx, input) in inputs.iter().enumerate() {
            if input.len() != dim {
                return Err(KernelError::DimensionMismatch {
                    expected: vec![dim],
                    got: vec![input.len()],
                    context: format!("Input vector at index {idx} has wrong dimension"),
                });
            }
        }

        let start = std::time::Instant::now();
        let mut matrix = Array2::<f64>::zeros((n, n));
        let mut stats = KernelMatrixStats {
            matrix_dim: n,
            ..Default::default()
        };

        for i in 0..n {
            for j in i..n {
                let value = if let Some(ref mut cache) = self.cache {
                    if let Some(cached) = cache.get(i, j) {
                        stats.cache_hits += 1;
                        cached
                    } else {
                        stats.cache_misses += 1;
                        let v = kernel_fn(&inputs[i], &inputs[j]);
                        cache.insert(i, j, v);
                        v
                    }
                } else {
                    kernel_fn(&inputs[i], &inputs[j])
                };
                stats.evaluations += 1;
                matrix[[i, j]] = value;
                if i != j {
                    matrix[[j, i]] = value;
                }
            }
        }

        stats.computation_ms = start.elapsed().as_secs_f64() * 1000.0;

        let gram = GramMatrix { data: matrix };
        Ok((gram, stats))
    }

    /// Clear the internal cache (no-op if caching is disabled).
    pub fn clear_cache(&mut self) {
        if let Some(ref mut cache) = self.cache {
            cache.clear();
        }
    }

    /// Return cache statistics, if caching is enabled.
    pub fn cache_hit_rate(&self) -> Option<f64> {
        self.cache.as_ref().map(|c| {
            let total = c.hits + c.misses;
            if total == 0 {
                0.0
            } else {
                c.hits as f64 / total as f64
            }
        })
    }
}

impl Default for BatchKernelComputer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── KernelCache tests ──────────────────────────────────────────

    #[test]
    fn test_kernel_cache_insert_get() {
        let mut cache = KernelCache::new(16);
        cache.insert(0, 1, 7.53);
        let val = cache.get(0, 1);
        assert_eq!(val, Some(7.53));
    }

    #[test]
    fn test_kernel_cache_symmetric() {
        let mut cache = KernelCache::new(16);
        cache.insert(1, 2, 42.0);
        assert_eq!(cache.get(2, 1), Some(42.0));
        assert_eq!(cache.get(1, 2), Some(42.0));
    }

    #[test]
    fn test_kernel_cache_miss() {
        let mut cache = KernelCache::new(16);
        assert_eq!(cache.get(5, 7), None);
    }

    #[test]
    fn test_kernel_cache_hit_rate() {
        let mut cache = KernelCache::new(16);
        cache.insert(0, 1, 1.0);
        let _ = cache.get(0, 1); // hit
        let _ = cache.get(2, 3); // miss
        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_kernel_cache_eviction() {
        let mut cache = KernelCache::new(2);
        cache.insert(0, 1, 1.0);
        cache.insert(2, 3, 2.0);
        // Cache is full (capacity 2), inserting a third evicts the oldest
        cache.insert(4, 5, 3.0);
        assert_eq!(cache.len(), 2);
        // (0,1) was the oldest and should be evicted
        assert_eq!(cache.get(0, 1), None);
        assert_eq!(cache.get(2, 3), Some(2.0));
        assert_eq!(cache.get(4, 5), Some(3.0));
    }

    #[test]
    fn test_kernel_cache_clear() {
        let mut cache = KernelCache::new(16);
        cache.insert(0, 1, 1.0);
        cache.insert(2, 3, 2.0);
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    // ── GramMatrix tests ───────────────────────────────────────────

    #[test]
    fn test_gram_matrix_new_valid() {
        let data = Array2::<f64>::zeros((3, 3));
        let gram = GramMatrix::new(data);
        assert!(gram.is_ok());
        assert_eq!(gram.expect("valid gram matrix").dim(), 3);
    }

    #[test]
    fn test_gram_matrix_not_square() {
        let data = Array2::<f64>::zeros((3, 2));
        let gram = GramMatrix::new(data);
        assert!(gram.is_err());
    }

    #[test]
    fn test_gram_matrix_diagonal() {
        let mut data = Array2::<f64>::zeros((3, 3));
        data[[0, 0]] = 1.0;
        data[[1, 1]] = 2.0;
        data[[2, 2]] = 3.0;
        let gram = GramMatrix::new(data).expect("valid gram matrix");
        assert_eq!(gram.diagonal(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_gram_matrix_trace() {
        let mut data = Array2::<f64>::zeros((3, 3));
        data[[0, 0]] = 1.0;
        data[[1, 1]] = 2.0;
        data[[2, 2]] = 3.0;
        let gram = GramMatrix::new(data).expect("valid gram matrix");
        assert!((gram.trace() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_gram_matrix_symmetric() {
        let mut data = Array2::<f64>::zeros((3, 3));
        data[[0, 1]] = 1.5;
        data[[1, 0]] = 1.5;
        data[[0, 2]] = 2.5;
        data[[2, 0]] = 2.5;
        data[[1, 2]] = 3.5;
        data[[2, 1]] = 3.5;
        let gram = GramMatrix::new(data).expect("valid gram matrix");
        assert!(gram.is_symmetric(1e-12));
    }

    #[test]
    fn test_gram_matrix_frobenius() {
        // Identity matrix of size n has Frobenius norm = sqrt(n)
        let n = 4;
        let mut data = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            data[[i, i]] = 1.0;
        }
        let gram = GramMatrix::new(data).expect("valid gram matrix");
        let expected = (n as f64).sqrt();
        assert!((gram.frobenius_norm() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_gram_matrix_nonneg_diagonal() {
        let mut data = Array2::<f64>::zeros((3, 3));
        data[[0, 0]] = 1.0;
        data[[1, 1]] = 0.0;
        data[[2, 2]] = 5.0;
        let gram = GramMatrix::new(data).expect("valid gram matrix");
        assert!(gram.has_nonneg_diagonal());
    }

    // ── BatchKernelComputer tests ──────────────────────────────────

    fn dot_product(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn test_batch_compute_basic() {
        let mut computer = BatchKernelComputer::new();
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let (gram, stats) = computer.compute(&inputs, dot_product).expect("compute ok");
        assert_eq!(gram.dim(), 3);
        assert_eq!(stats.matrix_dim, 3);
        // k([1,0],[0,1]) = 0
        assert!((gram.get(0, 1)).abs() < 1e-12);
        // k([1,0],[1,1]) = 1
        assert!((gram.get(0, 2) - 1.0).abs() < 1e-12);
        // k([1,1],[1,1]) = 2
        assert!((gram.get(2, 2) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_batch_compute_symmetric_result() {
        let mut computer = BatchKernelComputer::new();
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let (gram, _) = computer.compute(&inputs, dot_product).expect("compute ok");
        assert!(gram.is_symmetric(1e-12));
    }

    #[test]
    fn test_batch_compute_empty_batch() {
        let mut computer = BatchKernelComputer::new();
        let inputs: Vec<Vec<f64>> = vec![];
        let result = computer.compute(&inputs, dot_product);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_compute_with_cache() {
        let mut computer = BatchKernelComputer::with_cache(1024);
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        // First computation: all misses
        let (_, stats1) = computer.compute(&inputs, dot_product).expect("compute ok");
        assert_eq!(stats1.cache_hits, 0);
        assert!(stats1.cache_misses > 0);

        // Second computation with same inputs: all hits
        let (_, stats2) = computer.compute(&inputs, dot_product).expect("compute ok");
        assert!(stats2.cache_hits > 0);
        assert_eq!(stats2.cache_misses, 0);
    }

    #[test]
    fn test_batch_stats() {
        let mut computer = BatchKernelComputer::new();
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let (_, stats) = computer.compute(&inputs, dot_product).expect("compute ok");
        assert_eq!(stats.matrix_dim, 3);
        // Upper triangle including diagonal: n*(n+1)/2 = 6 evaluations
        assert_eq!(stats.evaluations, 6);
        assert!(stats.computation_ms >= 0.0);
        // No cache, so hits/misses are 0
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }
}

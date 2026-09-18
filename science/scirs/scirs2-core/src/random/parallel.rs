//! Thread-local RNG pools for high-performance parallel applications
//!
//! This module provides parallel-safe random number generation utilities
//! that are essential for high-performance scientific computing applications
//! requiring concurrent random number generation.

use crate::random::core::Random;
use ::ndarray::{Array, Dimension, IxDyn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Thread-local random number generator pool
///
/// Provides deterministic, thread-safe random number generation by maintaining
/// separate RNG instances for each thread, seeded in a predictable manner.
#[derive(Debug)]
pub struct ThreadLocalRngPool {
    seed_counter: Arc<AtomicUsize>,
    base_seed: u64,
}

impl ThreadLocalRngPool {
    /// Create a new thread-local RNG pool with a specific seed
    ///
    /// This ensures deterministic behavior across parallel executions
    /// when the same base seed is used.
    pub fn new(seed: u64) -> Self {
        Self {
            seed_counter: Arc::new(AtomicUsize::new(0)),
            base_seed: seed,
        }
    }

    /// Create a thread-local RNG pool with a seed derived from system time
    pub fn new_time_seeded() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(42);
        Self::new(seed)
    }

    /// Get a thread-local RNG
    ///
    /// Each call to this method returns a new RNG instance seeded with
    /// a deterministic value based on the base seed and thread counter.
    pub fn get_rng(&self) -> Random<StdRng> {
        let thread_id = self.seed_counter.fetch_add(1, Ordering::Relaxed);
        let seed = self.base_seed.wrapping_add(thread_id as u64);
        Random::seed(seed)
    }

    /// Execute a closure with a thread-local RNG
    ///
    /// This is the preferred way to use the thread-local RNG pool as it
    /// ensures proper resource management and consistent seeding.
    pub fn with_rng<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Random<StdRng>) -> R,
    {
        let mut rng = self.get_rng();
        f(&mut rng)
    }

    /// Get the base seed used by this pool
    pub fn base_seed(&self) -> u64 {
        self.base_seed
    }

    /// Get the current thread counter value
    pub fn thread_counter(&self) -> usize {
        self.seed_counter.load(Ordering::Relaxed)
    }

    /// Reset the thread counter (useful for reproducible testing)
    pub fn reset_counter(&self) {
        self.seed_counter.store(0, Ordering::Relaxed);
    }
}

impl Default for ThreadLocalRngPool {
    fn default() -> Self {
        Self::new_time_seeded()
    }
}

/// Parallel random number generation utilities
///
/// Provides high-level functions for generating random numbers in parallel
/// with automatic fallback to sequential generation when parallel features
/// are not available.
pub struct ParallelRng;

impl ParallelRng {
    /// Generate parallel random samples using Rayon (when available)
    ///
    /// When the "parallel" feature is enabled, this uses Rayon for parallel
    /// generation. Otherwise, it falls back to sequential generation.
    #[cfg(feature = "parallel")]
    pub fn parallel_sample<D, T>(distribution: D, count: usize, pool: &ThreadLocalRngPool) -> Vec<T>
    where
        D: Distribution<T> + Copy + Send + Sync,
        T: Send,
    {
        use crate::parallel_ops::{IntoParallelIterator, ParallelIterator};

        (0..count)
            .into_par_iter()
            .map(|_| pool.with_rng(|rng| rng.sample(distribution)))
            .collect()
    }

    /// Generate parallel random arrays using Rayon (when available)
    #[cfg(feature = "parallel")]
    pub fn parallel_sample_array<D, T, Sh>(
        distribution: D,
        shape: Sh,
        pool: &ThreadLocalRngPool,
    ) -> Array<T, IxDyn>
    where
        D: Distribution<T> + Copy + Send + Sync,
        T: Send + Clone,
        Sh: Into<IxDyn>,
    {
        let shape = shape.into();
        let size = shape.size();
        let samples = Self::parallel_sample(distribution, size, pool);
        Array::from_shape_vec(shape, samples).expect("Operation failed")
    }

    /// Sequential fallback when parallel feature is not enabled
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_sample<D, T>(distribution: D, count: usize, pool: &ThreadLocalRngPool) -> Vec<T>
    where
        D: Distribution<T> + Copy,
    {
        pool.with_rng(|rng| rng.sample_vec(distribution, count))
    }

    /// Sequential fallback when parallel feature is not enabled
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_sample_array<D, T, Sh>(
        distribution: D,
        shape: Sh,
        pool: &ThreadLocalRngPool,
    ) -> Array<T, IxDyn>
    where
        D: Distribution<T> + Copy,
        T: Send + Clone,
        Sh: Into<IxDyn> + crate::ndarray::Dimension,
    {
        pool.with_rng(|rng| rng.sample_array(shape.into(), distribution))
    }

    /// Generate parallel random samples with chunked processing
    ///
    /// Divides the work into chunks to balance load and reduce overhead.
    /// This is particularly useful for very large sample sizes.
    pub fn parallel_sample_chunked<D, T>(
        distribution: D,
        count: usize,
        chunk_size: usize,
        pool: &ThreadLocalRngPool,
    ) -> Vec<T>
    where
        D: Distribution<T> + Copy + Send + Sync,
        T: Send,
    {
        let num_chunks = (count + chunk_size - 1) / chunk_size;
        let mut result = Vec::with_capacity(count);

        #[cfg(feature = "parallel")]
        {
            use crate::parallel_ops::{IntoParallelIterator, ParallelIterator};

            let chunks: Vec<Vec<T>> = (0..num_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, count);
                    let chunk_count = end - start;

                    pool.with_rng(|rng| rng.sample_vec(distribution, chunk_count))
                })
                .collect();

            for chunk in chunks {
                result.extend(chunk);
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk_idx in 0..num_chunks {
                let start = chunk_idx * chunk_size;
                let end = std::cmp::min(start + chunk_size, count);
                let chunk_count = end - start;

                let chunk = pool.with_rng(|rng| rng.sample_vec(distribution, chunk_count));
                result.extend(chunk);
            }
        }

        result
    }

    /// Generate parallel bootstrap samples
    ///
    /// Useful for statistical bootstrap resampling in parallel.
    pub fn parallel_bootstrap<T>(
        data: &[T],
        n_bootstrap: usize,
        pool: &ThreadLocalRngPool,
    ) -> Vec<Vec<T>>
    where
        T: Clone + Send + Sync,
    {
        #[cfg(feature = "parallel")]
        {
            use crate::parallel_ops::{IntoParallelIterator, ParallelIterator};

            (0..n_bootstrap)
                .into_par_iter()
                .map(|_| {
                    pool.with_rng(|rng| {
                        (0..data.len())
                            .map(|_| {
                                let idx = rng.random_range(0..data.len());
                                data[idx].clone()
                            })
                            .collect()
                    })
                })
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            (0..n_bootstrap)
                .map(|_| {
                    pool.with_rng(|rng| {
                        (0..data.len())
                            .map(|_| {
                                let idx = rng.random_range(0..data.len());
                                data[idx].clone()
                            })
                            .collect()
                    })
                })
                .collect()
        }
    }
}

/// Workspace-aware parallel RNG for distributed computing
///
/// Provides coordination between multiple workers in a distributed system
/// to ensure non-overlapping random sequences.
#[derive(Debug)]
pub struct DistributedRngPool {
    worker_id: usize,
    total_workers: usize,
    base_pool: ThreadLocalRngPool,
}

impl DistributedRngPool {
    /// Create a new distributed RNG pool
    ///
    /// # Parameters
    /// * `worker_id` - Unique identifier for this worker (0-based)
    /// * `total_workers` - Total number of workers in the system
    /// * `base_seed` - Base seed for the entire distributed system
    pub fn new(worker_id: usize, total_workers: usize, base_seed: u64) -> Self {
        assert!(
            worker_id < total_workers,
            "Worker ID must be less than total workers"
        );

        // Create a unique seed for this worker
        let worker_seed = base_seed
            .wrapping_mul(total_workers as u64)
            .wrapping_add(worker_id as u64);

        Self {
            worker_id,
            total_workers,
            base_pool: ThreadLocalRngPool::new(worker_seed),
        }
    }

    /// Get a thread-local RNG for this worker
    pub fn get_rng(&self) -> Random<StdRng> {
        self.base_pool.get_rng()
    }

    /// Execute a closure with a worker-local RNG
    pub fn with_rng<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Random<StdRng>) -> R,
    {
        self.base_pool.with_rng(f)
    }

    /// Get worker information
    pub fn worker_info(&self) -> (usize, usize) {
        (self.worker_id, self.total_workers)
    }

    /// Generate samples allocated to this worker
    ///
    /// Automatically calculates the portion of samples this worker should generate
    /// based on its worker ID and the total number of workers.
    pub fn worker_sample<D, T>(&self, distribution: D, total_samples: usize) -> Vec<T>
    where
        D: Distribution<T> + Copy,
    {
        let samples_per_worker = total_samples / self.total_workers;
        let extra_samples = total_samples % self.total_workers;

        let my_samples = if self.worker_id < extra_samples {
            samples_per_worker + 1
        } else {
            samples_per_worker
        };

        self.with_rng(|rng| rng.sample_vec(distribution, my_samples))
    }
}

/// Batch processing utilities for large-scale random number generation
pub struct BatchRng;

impl BatchRng {
    /// Process random number generation in batches
    ///
    /// Useful for memory-efficient processing of very large datasets.
    pub fn process_batches<D, T, F, R>(
        distribution: D,
        total_samples: usize,
        batch_size: usize,
        pool: &ThreadLocalRngPool,
        mut processor: F,
    ) -> Vec<R>
    where
        D: Distribution<T> + Copy + Send + Sync,
        T: Send,
        F: FnMut(Vec<T>) -> R,
        R: Send,
    {
        let num_batches = (total_samples + batch_size - 1) / batch_size;
        let mut results = Vec::with_capacity(num_batches);

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = std::cmp::min(start + batch_size, total_samples);
            let current_batch_size = end - start;

            let batch_samples =
                ParallelRng::parallel_sample(distribution, current_batch_size, pool);
            let result = processor(batch_samples);
            results.push(result);
        }

        results
    }

    /// Stream random numbers with callback processing
    ///
    /// Generates random numbers in chunks and processes them with a callback,
    /// useful for streaming applications where memory usage must be controlled.
    pub fn stream_samples<D, T, F>(
        distribution: D,
        total_samples: usize,
        chunk_size: usize,
        pool: &ThreadLocalRngPool,
        mut callback: F,
    ) where
        D: Distribution<T> + Copy,
        F: FnMut(&[T]),
    {
        let num_chunks = (total_samples + chunk_size - 1) / chunk_size;

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = std::cmp::min(start + chunk_size, total_samples);
            let current_chunk_size = end - start;

            let chunk_samples =
                pool.with_rng(|rng| rng.sample_vec(distribution, current_chunk_size));
            callback(&chunk_samples);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Uniform;

    #[test]
    fn test_thread_local_rng_pool() {
        let pool = ThreadLocalRngPool::new(42);
        assert_eq!(pool.base_seed(), 42);
        assert_eq!(pool.thread_counter(), 0);

        let value =
            pool.with_rng(|rng| rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed")));
        assert!((0.0..1.0).contains(&value));
        assert_eq!(pool.thread_counter(), 1);
    }

    #[test]
    fn test_thread_local_rng_pool_deterministic() {
        let pool1 = ThreadLocalRngPool::new(123);
        let pool2 = ThreadLocalRngPool::new(123);

        let value1 =
            pool1.with_rng(|rng| rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed")));
        let value2 =
            pool2.with_rng(|rng| rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed")));

        assert_eq!(value1, value2);
    }

    #[test]
    fn test_thread_local_rng_pool_reset() {
        let pool = ThreadLocalRngPool::new(42);

        pool.with_rng(|_| {});
        pool.with_rng(|_| {});
        assert_eq!(pool.thread_counter(), 2);

        pool.reset_counter();
        assert_eq!(pool.thread_counter(), 0);
    }

    #[test]
    fn test_parallel_rng_sequential() {
        let pool = ThreadLocalRngPool::new(456);
        let samples = ParallelRng::parallel_sample(
            Uniform::new(0.0, 1.0).expect("Operation failed"),
            100,
            &pool,
        );

        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_parallel_rng_chunked() {
        let pool = ThreadLocalRngPool::new(789);
        let samples = ParallelRng::parallel_sample_chunked(
            Uniform::new(0.0, 1.0).expect("Operation failed"),
            100,
            25,
            &pool,
        );

        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_parallel_bootstrap() {
        let data = vec![1, 2, 3, 4, 5];
        let pool = ThreadLocalRngPool::new(101112);

        let bootstrap_samples = ParallelRng::parallel_bootstrap(&data, 10, &pool);
        assert_eq!(bootstrap_samples.len(), 10);
        assert!(bootstrap_samples
            .iter()
            .all(|sample| sample.len() == data.len()));
        assert!(bootstrap_samples
            .iter()
            .flatten()
            .all(|&x| data.contains(&x)));
    }

    #[test]
    fn test_distributed_rng_pool() {
        let pool = DistributedRngPool::new(0, 4, 12345);
        assert_eq!(pool.worker_info(), (0, 4));

        let samples = pool.worker_sample(Uniform::new(0.0, 1.0).expect("Operation failed"), 100);
        assert_eq!(samples.len(), 25); // 100 / 4 = 25 samples per worker
    }

    #[test]
    fn test_distributed_rng_pool_uneven_distribution() {
        // Test with total samples not evenly divisible by workers
        let pool = DistributedRngPool::new(0, 3, 12345);
        let samples = pool.worker_sample(Uniform::new(0.0, 1.0).expect("Operation failed"), 10);
        assert_eq!(samples.len(), 4); // First worker gets 4 samples (10/3 + 1)

        let pool = DistributedRngPool::new(2, 3, 12345);
        let samples = pool.worker_sample(Uniform::new(0.0, 1.0).expect("Operation failed"), 10);
        assert_eq!(samples.len(), 3); // Last worker gets 3 samples (10/3)
    }

    #[test]
    fn test_batch_processing() {
        let pool = ThreadLocalRngPool::new(131415);
        let results = BatchRng::process_batches(
            Uniform::new(0.0, 1.0).expect("Operation failed"),
            100,
            25,
            &pool,
            |batch| batch.len(),
        );

        assert_eq!(results.len(), 4); // 100 samples / 25 per batch = 4 batches
        assert_eq!(results.iter().sum::<usize>(), 100);
    }

    #[test]
    fn test_stream_samples() {
        let pool = ThreadLocalRngPool::new(161718);
        let mut total_processed = 0;

        BatchRng::stream_samples(
            Uniform::new(0.0, 1.0).expect("Operation failed"),
            100,
            30,
            &pool,
            |chunk| {
                total_processed += chunk.len();
                assert!(chunk.iter().all(|&x| (0.0..1.0).contains(&x)));
            },
        );

        assert_eq!(total_processed, 100);
    }
}

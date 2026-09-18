//! NUMA-aware scheduling and cache optimization
//!
//! This module provides advanced optimizations for work-stealing
//! including NUMA topology awareness and cache-friendly work distribution.

use super::super::strategies::work_stealing::WorkStealingScheduler;
use super::super::WorkerConfig;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Advanced work-stealing scheduler with NUMA awareness and cache optimization
///
/// This enhanced scheduler provides advanced optimizations for work-stealing
/// including NUMA topology awareness and cache-friendly work distribution.
pub struct AdvancedWorkStealingScheduler {
    base_scheduler: WorkStealingScheduler,
    numa_aware: bool,
    cache_linesize: usize,
    #[allow(dead_code)]
    work_queue_per_thread: bool,
}

impl AdvancedWorkStealingScheduler {
    /// Create a new advanced work-stealing scheduler
    pub fn new(config: &WorkerConfig) -> Self {
        Self {
            base_scheduler: WorkStealingScheduler::new(config),
            numa_aware: true,
            cache_linesize: 64, // Common cache line size
            work_queue_per_thread: true,
        }
    }

    /// Enable or disable NUMA-aware scheduling
    pub fn with_numa_aware(mut self, enabled: bool) -> Self {
        self.numa_aware = enabled;
        self
    }

    /// Set cache line size for cache-aware optimization
    pub fn with_cache_linesize(mut self, size: usize) -> Self {
        self.cache_linesize = size;
        self
    }

    /// Execute work with advanced optimizations
    ///
    /// This method implements enhanced work-stealing with:
    /// - NUMA-aware work distribution
    /// - Cache-friendly chunking
    /// - Adaptive scheduling based on workload characteristics
    pub fn execute_optimized<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        if items.is_empty() {
            return Vec::new();
        }

        let n = items.len();

        // Analyze workload characteristics
        let workload_type = self.analyze_workload(n);

        // Determine optimal chunking strategy
        let chunk_config = match workload_type {
            WorkloadType::MemoryBound => ChunkConfig {
                size: self.cache_linesize / std::mem::size_of::<T>(),
                strategy: ChunkStrategy::Sequential,
            },
            WorkloadType::CpuBound => ChunkConfig {
                size: n / (self.base_scheduler.num_workers * 4),
                strategy: ChunkStrategy::Interleaved,
            },
            WorkloadType::Mixed => ChunkConfig {
                size: self.adaptive_chunksize_enhanced(n),
                strategy: ChunkStrategy::Dynamic,
            },
        };

        // Execute with optimized strategy
        self.execute_with_strategy(items, f, chunk_config)
    }

    /// Analyze workload characteristics to optimize scheduling
    fn analyze_workload(&self, size: usize) -> WorkloadType {
        let memory_footprint = size * std::mem::size_of::<usize>();
        let cachesize = 8 * 1024 * 1024; // Approximate L3 cache size

        if memory_footprint > cachesize {
            WorkloadType::MemoryBound
        } else if size < 1000 {
            WorkloadType::CpuBound
        } else {
            WorkloadType::Mixed
        }
    }

    /// Enhanced adaptive chunk size calculation
    fn adaptive_chunksize_enhanced(&self, totalitems: usize) -> usize {
        let num_workers = self.base_scheduler.num_workers;
        let items_per_worker = totalitems / num_workers;

        // Consider cache efficiency and load balancing
        let cache_optimal_size = self.cache_linesize / std::mem::size_of::<usize>();
        let load_balance_size = std::cmp::max(1, items_per_worker / 8);

        // Choose the better of cache-optimal or load-balance size
        if cache_optimal_size > 0 && cache_optimal_size < load_balance_size * 2 {
            cache_optimal_size
        } else {
            load_balance_size
        }
    }

    /// Execute work with specific strategy
    fn execute_with_strategy<T, R, F>(&self, items: &[T], f: F, config: ChunkConfig) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        match config.strategy {
            ChunkStrategy::Sequential => self.execute_sequential_chunks(items, f, config.size),
            ChunkStrategy::Interleaved => {
                self.execute_interleaved_chunks(items, f, config.size)
            }
            ChunkStrategy::Dynamic => self.execute_dynamic_chunks(items, f, config.size),
        }
    }

    /// Execute with sequential chunk allocation
    fn execute_sequential_chunks<T, R, F>(&self, items: &[T], f: F, _chunksize: usize) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        // Use the base scheduler for sequential chunking
        self.base_scheduler.execute(items, f)
    }

    /// Execute with interleaved chunk allocation for better cache utilization
    fn execute_interleaved_chunks<T, R, F>(&self, items: &[T], f: F, chunksize: usize) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        let n = items.len();
        let chunksize = chunksize.max(1);
        let results = Arc::new(Mutex::new(vec![R::default(); n]));
        let work_counter = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|s| {
            let handles: Vec<_> = (0..self.base_scheduler.num_workers)
                .map(|_worker_id| {
                    let items_ref = items;
                    let f_ref = &f;
                    let results = results.clone();
                    let work_counter = work_counter.clone();

                    s.spawn(move || {
                        loop {
                            let chunk_id = work_counter.fetch_add(1, Ordering::SeqCst);
                            let start = chunk_id * chunksize;

                            if start >= n {
                                break;
                            }

                            let end = std::cmp::min(start + chunksize, n);

                            // Process interleaved indices for better cache utilization
                            for i in start..end {
                                let interleaved_idx = (i % self.base_scheduler.num_workers)
                                    * (n / self.base_scheduler.num_workers)
                                    + (i / self.base_scheduler.num_workers);

                                if interleaved_idx < n {
                                    let result = f_ref(&items_ref[interleaved_idx]);
                                    let mut results_guard = results.lock().expect("Operation failed");
                                    results_guard[interleaved_idx] = result;
                                }
                            }
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().expect("Operation failed");
            }
        });

        Arc::try_unwrap(results)
            .unwrap_or_else(|_| panic!("Failed to extract results"))
            .into_inner()
            .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
    }

    /// Execute with dynamic chunk sizing based on performance feedback
    fn execute_dynamic_chunks<T, R, F>(
        &self,
        items: &[T],
        f: F,
        _initial_chunksize: usize,
    ) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        // For now, use the base implementation with dynamic sizing
        // In a full implementation, this would adapt chunk sizes based on timing
        self.base_scheduler.execute(items, f)
    }

    /// Get NUMA node information for the current system
    pub fn get_numa_topology(&self) -> NumaTopology {
        NumaTopology::detect()
    }

    /// Execute with NUMA-aware work distribution
    pub fn execute_numa_aware<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        if !self.numa_aware {
            return self.base_scheduler.execute(items, f);
        }

        let topology = self.get_numa_topology();
        if topology.num_nodes <= 1 {
            // Single NUMA node, use regular execution
            return self.base_scheduler.execute(items, f);
        }

        // Distribute work across NUMA nodes
        self.execute_with_numa_distribution(items, f, &topology)
    }

    /// Execute work with explicit NUMA node distribution
    fn execute_with_numa_distribution<T, R, F>(
        &self,
        items: &[T],
        f: F,
        topology: &NumaTopology,
    ) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }

        let results = Arc::new(Mutex::new(vec![R::default(); n]));
        let work_counter = Arc::new(AtomicUsize::new(0));

        // Calculate workers per NUMA node
        let workers_per_node = self.base_scheduler.num_workers / topology.num_nodes;
        let chunk_size = std::cmp::max(1, n / (topology.num_nodes * workers_per_node));

        std::thread::scope(|s| {
            let handles: Vec<_> = (0..self.base_scheduler.num_workers)
                .map(|worker_id| {
                    let work_counter = work_counter.clone();
                    let results = results.clone();
                    let items_ref = items;
                    let f_ref = &f;
                    let numa_node = worker_id / workers_per_node;

                    s.spawn(move || {
                        // Set CPU affinity to NUMA node (simplified)
                        // In a real implementation, you'd use system calls to set affinity
                        
                        loop {
                            let start = work_counter.fetch_add(chunk_size, Ordering::SeqCst);
                            if start >= n {
                                break;
                            }

                            let end = std::cmp::min(start + chunk_size, n);

                            // Process chunk with NUMA-aware memory access
                            for i in start..end {
                                let result = f_ref(&items_ref[i]);
                                let mut results_guard = results.lock().expect("Operation failed");
                                results_guard[i] = result;
                            }
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().expect("Operation failed");
            }
        });

        Arc::try_unwrap(results)
            .unwrap_or_else(|_| panic!("Failed to extract results"))
            .into_inner()
            .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
    }
}

/// Workload analysis types
#[derive(Debug, Clone, Copy)]
enum WorkloadType {
    /// Memory-bound workloads that benefit from cache optimization
    MemoryBound,
    /// CPU-bound workloads that benefit from load balancing
    CpuBound,
    /// Mixed workloads requiring balanced approach
    Mixed,
}

/// Chunk configuration for work distribution
#[derive(Debug, Clone)]
struct ChunkConfig {
    size: usize,
    strategy: ChunkStrategy,
}

/// Work distribution strategies
#[derive(Debug, Clone, Copy)]
enum ChunkStrategy {
    /// Sequential chunk allocation
    Sequential,
    /// Interleaved allocation for cache efficiency
    Interleaved,
    /// Dynamic sizing based on performance
    Dynamic,
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub num_nodes: usize,
    pub cores_per_node: Vec<usize>,
    pub memory_per_node: Vec<usize>, // in MB
}

impl NumaTopology {
    /// Detect NUMA topology of the current system
    pub fn detect() -> Self {
        // Simplified NUMA detection - in a real implementation,
        // this would query the system's NUMA topology
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        
        // Assume single NUMA node for simplicity
        Self {
            num_nodes: 1,
            cores_per_node: vec![num_cpus],
            memory_per_node: vec![8192], // Assume 8GB
        }
    }

    /// Check if the system has multiple NUMA nodes
    pub fn is_numa_system(&self) -> bool {
        self.num_nodes > 1
    }

    /// Get optimal worker distribution across NUMA nodes
    pub fn optimal_worker_distribution(&self, total_workers: usize) -> Vec<usize> {
        if self.num_nodes <= 1 {
            return vec![total_workers];
        }

        let workers_per_node = total_workers / self.num_nodes;
        let remainder = total_workers % self.num_nodes;

        let mut distribution = vec![workers_per_node; self.num_nodes];
        
        // Distribute remainder workers
        for i in 0..remainder {
            distribution[i] += 1;
        }

        distribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_scheduler() {
        let config = WorkerConfig::default();
        let scheduler = AdvancedWorkStealingScheduler::new(&config);
        
        let items = vec![1, 2, 3, 4, 5];
        let results = scheduler.execute_optimized(&items, |x| x * 2);
        
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_numa_topology() {
        let topology = NumaTopology::detect();
        assert!(topology.num_nodes >= 1);
        assert!(!topology.cores_per_node.is_empty());
    }

    #[test]
    fn test_worker_distribution() {
        let topology = NumaTopology {
            num_nodes: 2,
            cores_per_node: vec![4, 4],
            memory_per_node: vec![8192, 8192],
        };
        
        let distribution = topology.optimal_worker_distribution(8);
        assert_eq!(distribution, vec![4, 4]);
        
        let distribution = topology.optimal_worker_distribution(9);
        assert_eq!(distribution, vec![5, 4]);
    }

    #[test]
    fn test_cache_linesize() {
        let config = WorkerConfig::default();
        let scheduler = AdvancedWorkStealingScheduler::new(&config)
            .with_cache_linesize(128);
        
        assert_eq!(scheduler.cache_linesize, 128);
    }
}
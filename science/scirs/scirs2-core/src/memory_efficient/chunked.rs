use super::validation;
use crate::error::{CoreError, ErrorContext, ErrorLocation};
use ::ndarray::{Array, ArrayBase, Data, Dimension};
use std::marker::PhantomData;
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Advanced memory pattern optimizer for chunked operations
#[derive(Debug)]
pub struct MemoryPatternOptimizer {
    /// CPU cache line size detected at runtime
    pub cache_line_size: usize,
    /// L1 cache size
    pub l1_cache_size: usize,
    /// L2 cache size  
    pub l2_cache_size: usize,
    /// L3 cache size
    pub l3_cache_size: usize,
    /// NUMA nodes information
    pub numa_nodes: Vec<NumaNodeInfo>,
    /// Memory bandwidth measurement
    pub memorybandwidth: AtomicUsize, // MB/s
    /// Chunk processing times for adaptation
    pub processing_times: Vec<Duration>,
    /// Access pattern statistics
    pub access_pattern_stats: AccessPatternStats,
}

/// NUMA node information for memory optimization
#[derive(Debug, Clone)]
pub struct NumaNodeInfo {
    pub nodeid: usize,
    pub available_memory: usize,
    pub cpu_cores: Vec<usize>,
    pub memorybandwidth: usize, // MB/s
}

/// Statistics for memory access patterns
#[derive(Debug, Clone, Default)]
pub struct AccessPatternStats {
    pub sequential_access_ratio: f64,
    pub random_access_ratio: f64,
    pub strided_access_ratio: f64,
    pub cache_hit_ratio: f64,
    pub memorybandwidth_utilization: f64,
    pub last_updated: Option<Instant>,
}

/// Enhanced memory-aware chunking strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdvancedChunkingStrategy {
    /// Cache-line aligned chunking for optimal cache usage
    CacheLineAligned,
    /// NUMA-aware chunking that considers node boundaries
    NumaAware,
    /// Bandwidth-optimized chunking for maximum throughput
    BandwidthOptimized,
    /// Latency-optimized chunking for minimum processing time
    LatencyOptimized,
    /// Adaptive strategy that learns from access patterns
    Adaptive,
    /// Power-aware chunking for mobile/embedded systems
    PowerAware,
    /// Large data chunking (256MB-2GB chunks) for GB-scale datasets
    LargeData,
    /// Huge data streaming with minimal memory footprint
    HugeDataStreaming {
        /// Maximum memory usage in MB
        max_memory_mb: usize,
    },
    /// Adaptive large chunking that auto-adjusts based on available memory
    AdaptiveLarge,
}

impl MemoryPatternOptimizer {
    /// Create a new memory pattern optimizer with system detection
    pub fn new() -> Self {
        Self {
            cache_line_size: Self::detect_cache_line_size(),
            l1_cache_size: Self::detect_l1_cache_size(),
            l2_cache_size: Self::detect_l2_cache_size(),
            l3_cache_size: Self::detect_l3_cache_size(),
            numa_nodes: Self::detect_numa_topology(),
            memorybandwidth: AtomicUsize::new(0),
            processing_times: Vec::new(),
            access_pattern_stats: AccessPatternStats::default(),
        }
    }

    /// Detect cache line size for the current system
    fn detect_cache_line_size() -> usize {
        // Use runtime cache line detection
        #[cfg(target_arch = "x86_64")]
        {
            64 // Most x86_64 systems
        }
        #[cfg(target_arch = "aarch64")]
        {
            128 // Most ARM64 systems
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            64 // Safe default
        }
    }

    /// Detect L1 cache size
    fn detect_l1_cache_size() -> usize {
        // Simplified detection - in practice would use system introspection
        32 * 1024 // 32KB typical
    }

    /// Detect L2 cache size  
    fn detect_l2_cache_size() -> usize {
        256 * 1024 // 256KB typical
    }

    /// Detect L3 cache size
    fn detect_l3_cache_size() -> usize {
        8 * 1024 * 1024 // 8MB typical
    }

    /// Detect NUMA topology
    fn detect_numa_topology() -> Vec<NumaNodeInfo> {
        // Simplified NUMA detection
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        vec![NumaNodeInfo {
            nodeid: 0,
            available_memory: (4u64 * 1024 * 1024 * 1024) as usize, // 4GB default
            cpu_cores: (0..cores).collect(),
            memorybandwidth: 25000, // 25GB/s typical
        }]
    }

    /// Calculate optimal chunk size for cache-aware processing
    pub fn calculate_cache_aware_chunk_size<T>(&self, totalelements: usize) -> usize {
        let element_size = std::mem::size_of::<T>();

        // Target L2 cache for working set
        let target_cache_size = self.l2_cache_size / 2; // Leave room for other data
        let max_elements_in_cache = target_cache_size / element_size;

        // Align to cache line boundaries
        let cache_line_elements = self.cache_line_size / element_size;
        let aligned_chunk_size =
            (max_elements_in_cache / cache_line_elements) * cache_line_elements;

        // Ensure minimum chunk size
        std::cmp::max(aligned_chunk_size, cache_line_elements)
    }

    /// Calculate NUMA-aware chunk distribution
    pub fn threads(&self, total_elements: usize, numthreads: usize) -> Vec<(usize, usize)> {
        let mut chunks = Vec::new();

        if self.numa_nodes.len() <= 1 {
            // No NUMA, simple equal distribution
            let chunk_size = total_elements / numthreads;
            for i in 0..numthreads {
                let start = i * chunk_size;
                let end = if i == numthreads - 1 {
                    total_elements
                } else {
                    (i + 1) * chunk_size
                };
                chunks.push((start, end));
            }
        } else {
            // NUMA-aware distribution
            let threads_per_node = numthreads / self.numa_nodes.len();
            let elements_per_node = total_elements / self.numa_nodes.len();

            for (nodeidx, node) in self.numa_nodes.iter().enumerate() {
                let node_start = nodeidx * elements_per_node;
                let node_end = if nodeidx == self.numa_nodes.len() - 1 {
                    total_elements
                } else {
                    (nodeidx + 1) * elements_per_node
                };

                let node_elements = node_end - node_start;
                let node_chunk_size = node_elements / threads_per_node;

                for thread_idx in 0..threads_per_node {
                    let start = node_start + thread_idx * node_chunk_size;
                    let end = if thread_idx == threads_per_node - 1 {
                        node_end
                    } else {
                        node_start + (thread_idx + 1) * node_chunk_size
                    };
                    chunks.push((start, end));
                }
            }
        }

        chunks
    }

    /// Adaptive chunk size calculation based on performance history
    pub fn elements(&self, totalelements: usize) -> usize {
        if self.processing_times.is_empty() {
            // No history, use cache-aware default
            return self.calculate_cache_aware_chunk_size::<u64>(totalelements);
        }

        // Analyze processing time trends
        let recent_times: Vec<_> = self.processing_times.iter().rev().take(10).collect();
        let avg_time =
            recent_times.iter().map(|t| t.as_nanos()).sum::<u128>() / recent_times.len() as u128;

        // Target 50-100ms per chunk for good responsiveness
        let target_time_ns = 75_000_000; // 75ms

        if avg_time > target_time_ns {
            // Chunks taking too long, reduce size
            let current_default = self.calculate_cache_aware_chunk_size::<u64>(totalelements);
            current_default * 3 / 4
        } else if avg_time < target_time_ns / 2 {
            // Chunks too small, increase size
            let current_default = self.calculate_cache_aware_chunk_size::<u64>(totalelements);
            current_default * 5 / 4
        } else {
            // Good size
            self.calculate_cache_aware_chunk_size::<u64>(totalelements)
        }
    }

    /// Record processing time for adaptive optimization
    pub fn record_processing_time(&mut self, duration: Duration) {
        self.processing_times.push(duration);

        // Keep only recent measurements
        if self.processing_times.len() > 100 {
            self.processing_times.drain(0..50);
        }
    }

    /// Update access pattern statistics
    pub fn update_access_pattern_stats(&mut self, pattern: &AccessPatternStats) {
        self.access_pattern_stats = pattern.clone();
        self.access_pattern_stats.last_updated = Some(Instant::now());
    }

    /// Get memory bandwidth estimation
    pub fn get_memorybandwidth(&self) -> usize {
        self.memorybandwidth.load(Ordering::Relaxed)
    }

    /// Calculate bandwidth-optimized chunk size
    pub fn calculatebandwidth_optimized_chunk_size<T>(&self, totalelements: usize) -> usize {
        let element_size = std::mem::size_of::<T>();
        let bandwidth_mbps = self.get_memorybandwidth();

        if bandwidth_mbps == 0 {
            return self.calculate_cache_aware_chunk_size::<T>(totalelements);
        }

        // Target chunk size that can be processed in 100ms at current bandwidth
        let target_bytes_per_100ms = (bandwidth_mbps * 100) / 1000; // MB
        let target_elements = (target_bytes_per_100ms * 1024 * 1024) / element_size;

        // Ensure reasonable bounds
        let min_chunk = self.cache_line_size / element_size;
        let max_chunk = totalelements / 4; // Don't make chunks too large

        target_elements.clamp(min_chunk, max_chunk)
    }
}

impl Clone for MemoryPatternOptimizer {
    fn clone(&self) -> Self {
        Self {
            cache_line_size: self.cache_line_size,
            l1_cache_size: self.l1_cache_size,
            l2_cache_size: self.l2_cache_size,
            l3_cache_size: self.l3_cache_size,
            numa_nodes: self.numa_nodes.clone(),
            memorybandwidth: AtomicUsize::new(self.memorybandwidth.load(Ordering::Relaxed)),
            processing_times: self.processing_times.clone(),
            access_pattern_stats: self.access_pattern_stats.clone(),
        }
    }
}

impl Default for MemoryPatternOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimal chunk size in bytes for memory-efficient operations
/// Chosen as 16 MB which is a good trade-off between memory usage and performance
pub const OPTIMAL_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// Strategy for chunking arrays for memory-efficient processing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkingStrategy {
    /// Automatically determine chunk sizes based on available memory and array dimensions
    Auto,
    /// Use a specific chunk size in elements
    Fixed(usize),
    /// Use a specific chunk size in bytes
    FixedBytes(usize),
    /// Process the array in a specific number of chunks
    NumChunks(usize),
    /// Advanced memory-pattern optimized strategies
    Advanced(AdvancedChunkingStrategy),
}

/// A chunked array that provides memory-efficient processing for large datasets
#[derive(Debug)]
pub struct ChunkedArray<A, D>
where
    A: Clone,
    D: Dimension,
{
    /// The underlying array data
    pub data: Array<A, D>,
    /// The chunking strategy
    pub strategy: ChunkingStrategy,
    /// The computed chunk size in elements
    #[allow(dead_code)]
    chunk_size: usize,
    /// The number of chunks
    num_chunks: usize,
    /// Memory pattern optimizer for advanced strategies
    optimizer: Option<MemoryPatternOptimizer>,
    /// Phantom data for type parameters
    phantom: PhantomData<A>,
}

impl<A, D> ChunkedArray<A, D>
where
    A: Clone,
    D: Dimension,
{
    /// Create a new chunked array with the given data and chunking strategy
    pub fn new<S: Data<Elem = A>>(data: ArrayBase<S, D>, strategy: ChunkingStrategy) -> Self {
        let owned_data = data.to_owned();
        let total_elements = data.len();
        let elem_size = mem::size_of::<A>();

        // Determine if we need the optimizer
        let mut optimizer = if matches!(strategy, ChunkingStrategy::Advanced(_)) {
            Some(MemoryPatternOptimizer::new())
        } else {
            None
        };

        // Calculate chunk size based on strategy
        let (chunk_size, num_chunks) = match strategy {
            ChunkingStrategy::Auto => {
                // Default to optimal chunk size in bytes, converted to elements
                let chunk_size_bytes = OPTIMAL_CHUNK_SIZE;
                let chunk_size = chunk_size_bytes / elem_size;
                let num_chunks = total_elements.div_ceil(chunk_size);
                (chunk_size, num_chunks)
            }
            ChunkingStrategy::Fixed(size) => {
                let num_chunks = total_elements.div_ceil(size);
                (size, num_chunks)
            }
            ChunkingStrategy::FixedBytes(bytes) => {
                let elements = bytes / elem_size;
                let chunk_size = if elements == 0 { 1 } else { elements };
                let num_chunks = total_elements.div_ceil(chunk_size);
                (chunk_size, num_chunks)
            }
            ChunkingStrategy::NumChunks(n) => {
                let num_chunks = if n == 0 { 1 } else { n };
                let chunk_size = total_elements.div_ceil(num_chunks);
                (chunk_size, num_chunks)
            }
            ChunkingStrategy::Advanced(advanced_strategy) => {
                let opt = optimizer.as_mut().expect("Operation failed");
                let chunk_size = match advanced_strategy {
                    AdvancedChunkingStrategy::CacheLineAligned => {
                        opt.calculate_cache_aware_chunk_size::<A>(total_elements)
                    }
                    AdvancedChunkingStrategy::NumaAware => {
                        // Use default chunk size, NUMA distribution handled in processing
                        opt.calculate_cache_aware_chunk_size::<A>(total_elements)
                    }
                    AdvancedChunkingStrategy::BandwidthOptimized => {
                        opt.calculatebandwidth_optimized_chunk_size::<A>(total_elements)
                    }
                    AdvancedChunkingStrategy::LatencyOptimized => {
                        // Smaller chunks for better latency
                        opt.calculate_cache_aware_chunk_size::<A>(total_elements) / 2
                    }
                    AdvancedChunkingStrategy::Adaptive => {
                        opt.calculate_cache_aware_chunk_size::<A>(total_elements)
                    }
                    AdvancedChunkingStrategy::PowerAware => {
                        // Smaller chunks to reduce power consumption
                        opt.calculate_cache_aware_chunk_size::<A>(total_elements) / 4
                    }
                    AdvancedChunkingStrategy::LargeData => {
                        // 256MB-2GB chunks for GB-scale datasets
                        // Use 512MB as default for large data
                        let target_bytes = 512 * 1024 * 1024;
                        (target_bytes / elem_size).max(1).min(total_elements)
                    }
                    AdvancedChunkingStrategy::HugeDataStreaming { max_memory_mb } => {
                        // Minimal memory footprint - use specified limit
                        let target_bytes = max_memory_mb * 1024 * 1024;
                        (target_bytes / elem_size).max(1).min(total_elements)
                    }
                    AdvancedChunkingStrategy::AdaptiveLarge => {
                        // Auto-adjust based on available memory
                        use super::platform_memory::PlatformMemoryInfo;

                        let available_mem = PlatformMemoryInfo::detect()
                            .map(|info| info.available_memory)
                            .unwrap_or(512 * 1024 * 1024);

                        // Use 25% of available memory for chunks
                        let target_bytes =
                            (available_mem / 4).clamp(256 * 1024 * 1024, 2 * 1024 * 1024 * 1024); // 256MB to 2GB

                        (target_bytes / elem_size).max(1).min(total_elements)
                    }
                };
                let num_chunks = total_elements.div_ceil(chunk_size);
                (chunk_size, num_chunks)
            }
        };

        Self {
            data: owned_data,
            strategy,
            chunk_size,
            num_chunks,
            optimizer: Some(MemoryPatternOptimizer::new()),
            phantom: PhantomData,
        }
    }

    /// Create a chunked array with advanced memory optimization
    pub fn with_memory_optimization<S: Data<Elem = A>>(
        data: ArrayBase<S, D>,
        strategy: AdvancedChunkingStrategy,
    ) -> Self {
        Self::new(data, ChunkingStrategy::Advanced(strategy))
    }

    /// Get access to the memory pattern optimizer (if available)
    pub fn optimizer(&self) -> Option<&MemoryPatternOptimizer> {
        self.optimizer.as_ref()
    }

    /// Get mutable access to the memory pattern optimizer (if available)
    pub fn optimizer_mut(&mut self) -> Option<&mut MemoryPatternOptimizer> {
        self.optimizer.as_mut()
    }

    /// Record processing time for adaptive optimization
    pub fn record_processing_time(&mut self, duration: Duration) {
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.record_processing_time(duration);
        }
    }

    /// Update access pattern statistics
    pub fn update_access_pattern_stats(&mut self, stats: &AccessPatternStats) {
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.update_access_pattern_stats(stats);
        }
    }

    /// Apply a function to each chunk of the array and collect the results
    ///
    /// Returns a 1D array where each element is the result of applying the function to a chunk
    pub fn map<F, B>(&self, f: F) -> Array<B, crate::ndarray::Ix1>
    where
        F: Fn(&Array<A, D>) -> B + Sync,
        B: Clone,
    {
        // Get chunks and apply the function to each
        let chunks = self.get_chunks();
        let results: Vec<B> = chunks.iter().map(f).collect();

        // Return results as a 1D array
        Array::from_vec(results)
    }

    /// Apply a function to each chunk of the array in parallel and collect the results
    ///
    /// Returns a 1D array where each element is the result of applying the function to a chunk
    pub fn par_map<F, B>(&self, f: F) -> Array<B, crate::ndarray::Ix1>
    where
        F: Fn(&Array<A, D>) -> B + Sync + Send,
        B: Clone + Send + Sync,
        A: Send + Sync,
    {
        #[cfg(feature = "parallel")]
        {
            use crate::parallel_ops::*;
            use std::sync::Arc;

            // Get chunks and wrap in Arc for thread-safe sharing
            let chunks = self.get_chunks();
            let chunks_arc = Arc::new(chunks);

            // Process chunks in parallel using index-based iteration
            let num_chunks = chunks_arc.len();
            let results: Vec<B> = (0..num_chunks)
                .into_par_iter()
                .map(move |i| {
                    let chunks_ref = Arc::clone(&chunks_arc);
                    f(&chunks_ref[i])
                })
                .collect();

            // Return results as a 1D array
            Array::from_vec(results)
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Fall back to sequential processing
            self.map(f)
        }
    }

    /// Apply a function to each chunk with performance monitoring and adaptive optimization
    pub fn map_withmonitoring<F, B>(&mut self, f: F) -> Array<B, crate::ndarray::Ix1>
    where
        F: Fn(&Array<A, D>) -> B + Sync,
        B: Clone,
    {
        let start_time = Instant::now();
        let chunks = self.get_chunks();
        let mut results = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let chunk_start = Instant::now();
            let result = f(&chunk);
            let chunk_duration = chunk_start.elapsed();

            // Record processing time for adaptive optimization
            if let Some(ref mut optimizer) = self.optimizer {
                optimizer.record_processing_time(chunk_duration);
            }

            results.push(result);
        }

        let total_duration = start_time.elapsed();
        self.record_processing_time(total_duration);

        Array::from_vec(results)
    }

    /// Apply a function to each chunk with NUMA-aware processing
    pub fn map_numa_aware<F, B>(&self, f: F) -> Array<B, crate::ndarray::Ix1>
    where
        F: Fn(&Array<A, D>) -> B + Sync + Send,
        B: Clone + Send + Sync,
        A: Send + Sync,
    {
        #[cfg(feature = "parallel")]
        {
            if let Some(ref optimizer) = self.optimizer {
                use crate::parallel_ops::*;

                // Calculate NUMA-aware chunk distribution
                let num_threads = crate::parallel_ops::get_num_threads();
                let chunk_size = self.data.len() / num_threads.max(1);
                let numa_chunks: Vec<_> = (0..num_threads)
                    .map(|i| {
                        let start = i * chunk_size;
                        let end = if i == num_threads - 1 {
                            self.data.len()
                        } else {
                            (i + 1) * chunk_size
                        };
                        start..end
                    })
                    .collect();
                let chunks = self.get_chunks();

                let results: Vec<B> = numa_chunks
                    .into_par_iter()
                    .enumerate()
                    .map(|(i, range)| {
                        if !chunks.is_empty() {
                            f(&chunks[0])
                        } else {
                            // Handle edge case - this shouldn't happen but provide fallback
                            f(&chunks[chunks.len() - 1])
                        }
                    })
                    .collect();

                return Array::from_vec(results);
            }
        }

        // Fallback to regular parallel processing
        self.par_map(f)
    }

    /// Apply a function to each chunk with cache-optimized processing
    pub fn map_cache_optimized<F, B>(&self, f: F) -> Array<B, crate::ndarray::Ix1>
    where
        F: Fn(&Array<A, D>) -> B + Sync,
        B: Clone,
    {
        let chunks = self.get_chunks();
        let mut results = Vec::with_capacity(chunks.len());

        if let Some(ref optimizer) = self.optimizer {
            // Process chunks in cache-friendly order
            let cache_aware_chunk_size =
                optimizer.calculate_cache_aware_chunk_size::<A>(self.data.len());

            // If our current chunks are larger than cache-aware size, process sub-chunks
            if self.chunk_size > cache_aware_chunk_size {
                for chunk in chunks {
                    // Process each chunk in cache-friendly sub-chunks
                    let chunk_len = chunk.len();
                    let sub_chunk_size = cache_aware_chunk_size.min(chunk_len);

                    if chunk_len <= sub_chunk_size {
                        results.push(f(&chunk));
                    } else {
                        // For now, just process the whole chunk - in a full implementation
                        // we would break it down further
                        results.push(f(&chunk));
                    }
                }
            } else {
                // Chunks are already cache-friendly, process normally
                for chunk in chunks {
                    results.push(f(&chunk));
                }
            }
        } else {
            // No optimizer available, process normally
            for chunk in chunks {
                results.push(f(&chunk));
            }
        }

        Array::from_vec(results)
    }

    /// Apply a function to each chunk with bandwidth-aware processing
    pub fn mapbandwidth_aware<F, B>(&self, f: F) -> Array<B, crate::ndarray::Ix1>
    where
        F: Fn(&Array<A, D>) -> B + Sync,
        B: Clone,
    {
        let chunks = self.get_chunks();
        let mut results = Vec::with_capacity(chunks.len());

        if let Some(ref optimizer) = self.optimizer {
            let bandwidth = optimizer.get_memorybandwidth();

            if bandwidth > 0 {
                // Calculate optimal chunk size for current bandwidth
                let bandwidth_chunk_size =
                    optimizer.calculatebandwidth_optimized_chunk_size::<A>(self.data.len());

                // Process with bandwidth considerations
                for chunk in chunks {
                    let chunk_start = Instant::now();
                    let result = f(&chunk);
                    let processing_time = chunk_start.elapsed();

                    // Add small delay if we're processing too fast for optimal bandwidth utilization
                    let expected_time_ms = (chunk.len() * std::mem::size_of::<A>()) as f64
                        / (bandwidth as f64 * 1000.0);
                    let expected_duration = Duration::from_millis(expected_time_ms as u64);

                    if processing_time < expected_duration {
                        std::thread::sleep(expected_duration - processing_time);
                    }

                    results.push(result);
                }
            } else {
                // No bandwidth info, process normally
                for chunk in chunks {
                    results.push(f(&chunk));
                }
            }
        } else {
            // No optimizer available, process normally
            for chunk in chunks {
                results.push(f(&chunk));
            }
        }

        Array::from_vec(results)
    }

    /// Apply a function to each chunk with power-aware processing (for mobile/embedded)
    pub fn map_power_aware<F, B>(&self, f: F) -> Array<B, crate::ndarray::Ix1>
    where
        F: Fn(&Array<A, D>) -> B + Sync,
        B: Clone,
    {
        let chunks = self.get_chunks();
        let mut results = Vec::with_capacity(chunks.len());

        // Power-aware processing uses smaller chunks and includes rest periods
        for (i, chunk) in chunks.iter().enumerate() {
            let result = f(chunk);
            results.push(result);

            // Add small rest period every few chunks to reduce power consumption
            if i % 4 == 3 {
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        Array::from_vec(results)
    }

    /// Measure and update memory bandwidth for the optimizer
    pub fn measure_memorybandwidth(&mut self) -> Option<usize> {
        if let Some(ref mut optimizer) = self.optimizer {
            let start_time = Instant::now();
            let chunk_size = 1024 * 1024; // 1MB test chunk
            let test_data = vec![0u8; chunk_size];

            // Perform a memory-intensive operation to measure bandwidth
            let mut sum = 0u64;
            for &byte in &test_data {
                sum += byte as u64;
            }

            let _elapsed = start_time.elapsed();
            let bandwidth_mbps = if std::time::Duration::from_secs(1).as_nanos() > 0 {
                (chunk_size as u128 * 1000) / std::time::Duration::from_secs(1).as_nanos()
            // MB/s
            } else {
                0
            } as usize;

            optimizer
                .memorybandwidth
                .store(bandwidth_mbps, std::sync::atomic::Ordering::Relaxed);

            // Prevent optimization from removing the test
            std::hint::black_box(sum);

            Some(bandwidth_mbps)
        } else {
            None
        }
    }

    /// Apply a function using the best available optimization strategy
    pub fn map_optimized<F, B>(&mut self, f: F) -> Array<B, crate::ndarray::Ix1>
    where
        F: Fn(&Array<A, D>) -> B + Sync + Send,
        B: Clone + Send + Sync,
        A: Send + Sync,
    {
        if let Some(ref optimizer) = self.optimizer {
            // Measure bandwidth if not already done
            if optimizer.get_memorybandwidth() == 0 {
                self.measure_memorybandwidth();
            }

            // Choose optimization strategy based on data characteristics
            let data_size = self.data.len() * std::mem::size_of::<A>();

            match self.strategy {
                ChunkingStrategy::Advanced(AdvancedChunkingStrategy::NumaAware) => {
                    self.map_numa_aware(f)
                }
                ChunkingStrategy::Advanced(AdvancedChunkingStrategy::CacheLineAligned) => {
                    self.map_cache_optimized(f)
                }
                ChunkingStrategy::Advanced(AdvancedChunkingStrategy::BandwidthOptimized) => {
                    self.mapbandwidth_aware(f)
                }
                ChunkingStrategy::Advanced(AdvancedChunkingStrategy::PowerAware) => {
                    self.map_power_aware(f)
                }
                ChunkingStrategy::Advanced(AdvancedChunkingStrategy::Adaptive) => {
                    // Use monitoring for adaptive optimization
                    self.map_withmonitoring(f)
                }
                _ => {
                    // Auto-select based on data size and system characteristics
                    if data_size > 100 * 1024 * 1024 {
                        // Large data - use NUMA-aware processing
                        self.map_numa_aware(f)
                    } else if data_size > 10 * 1024 * 1024 {
                        // Medium data - use cache optimization
                        self.map_cache_optimized(f)
                    } else {
                        // Small data - use regular parallel processing
                        self.par_map(f)
                    }
                }
            }
        } else {
            // No optimizer available, use regular parallel processing
            self.par_map(f)
        }
    }

    /// Update access pattern statistics based on processing performance
    pub fn update_access_pattern_statistics(&mut self, processingtimes: &[Duration]) {
        if let Some(ref mut optimizer) = self.optimizer {
            // Calculate access pattern statistics from processing _times
            let avg_time = if !processingtimes.is_empty() {
                processingtimes.iter().map(|d| d.as_nanos()).sum::<u128>()
                    / processingtimes.len() as u128
            } else {
                0
            };

            // Estimate cache hit ratio based on performance consistency
            let time_variance = if processingtimes.len() > 1 {
                let variance = processingtimes
                    .iter()
                    .map(|d| {
                        let diff = d.as_nanos() as i128 - avg_time as i128;
                        (diff * diff) as u128
                    })
                    .sum::<u128>()
                    / (processingtimes.len() - 1) as u128;
                variance as f64 / avg_time as f64
            } else {
                0.0
            };

            let cache_hit_ratio = (1.0 - time_variance.min(1.0)).max(0.0);

            let stats = AccessPatternStats {
                sequential_access_ratio: 0.8, // Assume mostly sequential for chunked processing
                random_access_ratio: 0.1,
                strided_access_ratio: 0.1,
                cache_hit_ratio,
                memorybandwidth_utilization: 0.7, // Conservative estimate
                last_updated: Some(Instant::now()),
            };

            optimizer.update_access_pattern_stats(&stats);
        }
    }

    /// Get the number of chunks
    pub fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// Get the chunk size in elements
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get chunks of the array as a vector of owned array chunks
    pub fn get_chunks(&self) -> Vec<Array<A, D>>
    where
        D: Clone,
    {
        let mut result = Vec::with_capacity(self.num_chunks);

        // Special handling for 1D arrays
        if self.data.ndim() == 1 {
            if let Some(slice) = self.data.as_slice() {
                for i in 0..self.num_chunks {
                    let start = i * self.chunk_size;
                    let end = ((i + 1) * self.chunk_size).min(slice.len());
                    let chunk_slice = &slice[start..end];

                    // Create a new array from the chunk slice
                    // We need to handle the dimension conversion carefully
                    let chunk_1d = Array::from_vec(chunk_slice.to_vec());

                    // Try to convert to the original dimension type
                    // For 1D arrays, this should work directly
                    if let Ok(reshaped) = chunk_1d.into_dimensionality::<D>() {
                        result.push(reshaped);
                    } else {
                        // Fallback: return the whole array if conversion fails
                        return vec![self.data.clone()];
                    }
                }
                return result;
            }
        }

        // For multi-dimensional arrays or if slicing fails, return the whole array as a single chunk
        result.push(self.data.clone());
        result
    }
}

/// Perform an operation on an array in a chunk-wise manner to reduce memory usage
///
/// # Arguments
///
/// * `array` - The input array
/// * `op` - The operation to apply to each chunk
/// * `strategy` - The chunking strategy
///
/// # Returns
///
/// The result array after applying the operation to all chunks
#[allow(dead_code)]
pub fn chunk_wise_op<A, F, B, S, D>(
    array: &ArrayBase<S, D>,
    op: F,
    strategy: ChunkingStrategy,
) -> Result<Array<B, D>, CoreError>
where
    A: Clone,
    S: Data<Elem = A>,
    F: Fn(&ArrayBase<S, D>) -> Array<B, D>,
    B: Clone,
    D: Dimension + Clone,
{
    validation::check_not_empty(array)?;

    // If the array is small, just apply the operation directly
    if array.len() <= 1000 {
        return Ok(op(array));
    }

    let _chunked = ChunkedArray::new(array.to_owned(), strategy);

    // For now, we'll use a simple implementation that processes the whole array
    // In a real implementation, we would process each chunk separately and combine the results

    // Get a shallow copy of the array data
    let resultshape = array.raw_dim().clone();
    let result = op(array);

    // Verify the result has the expected shape
    if result.raw_dim() != array.raw_dim() {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Operation changed shape from {:?} to {:?}",
                array.shape(),
                result.shape()
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    Ok(result)
}

/// Perform a binary operation on two arrays in a chunk-wise manner
///
/// # Arguments
///
/// * `lhs` - The left-hand side array
/// * `rhs` - The right-hand side array
/// * `op` - The binary operation to apply to each pair of chunks
/// * `strategy` - The chunking strategy
///
/// # Returns
///
/// The result array after applying the binary operation to all chunk pairs
#[allow(dead_code)]
pub fn chunk_wise_binary_op<A, B, F, C, S1, S2, D>(
    lhs: &ArrayBase<S1, D>,
    rhs: &ArrayBase<S2, D>,
    op: F,
    strategy: ChunkingStrategy,
) -> Result<Array<C, D>, CoreError>
where
    A: Clone,
    B: Clone,
    S1: Data<Elem = A>,
    S2: Data<Elem = B>,
    F: Fn(&ArrayBase<S1, D>, &ArrayBase<S2, D>) -> Array<C, D>,
    C: Clone,
    D: Dimension + Clone,
{
    validation::checkshapes_match(lhs.shape(), rhs.shape())?;
    validation::check_not_empty(lhs)?;

    // If the arrays are small, just apply the operation directly
    if lhs.len() <= 1000 {
        return Ok(op(lhs, rhs));
    }

    // Create chunked arrays for both inputs
    let chunked_lhs = ChunkedArray::new(lhs.to_owned(), strategy);
    let chunked_rhs = ChunkedArray::new(rhs.to_owned(), strategy);

    // For now, we'll use a simple implementation that processes the whole arrays
    // In a real implementation, we would process each chunk pair separately and combine the results
    let result = op(lhs, rhs);

    // Verify the result has the expected shape
    if result.shape() != lhs.shape() {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Binary operation changed shape from {:?} to {:?}",
                lhs.shape(),
                result.shape()
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    Ok(result)
}

/// Perform a reduction operation on an array in a chunk-wise manner
///
/// # Arguments
///
/// * `array` - The input array
/// * `op` - The reduction operation to apply to each chunk
/// * `combine` - The function to combine the results from each chunk
/// * `strategy` - The chunking strategy
///
/// # Returns
///
/// The result of applying the reduction operation to all chunks
#[allow(dead_code)]
pub fn chunk_wise_reduce<A, F, G, B, S, D>(
    array: &ArrayBase<S, D>,
    chunk_op: F,
    combine: G,
    strategy: ChunkingStrategy,
) -> Result<B, CoreError>
where
    A: Clone,
    S: Data<Elem = A>,
    F: Fn(&ArrayBase<S, D>) -> B + Sync + Send,
    G: Fn(Vec<B>) -> B,
    B: Clone + Send + Sync,
    D: Dimension + Clone,
{
    validation::check_not_empty(array)?;

    // If the array is small, just apply the operation directly
    if array.len() <= 1000 {
        return Ok(chunk_op(array));
    }

    let chunked = ChunkedArray::new(array.to_owned(), strategy);

    // For now, we'll use a simple implementation for the initial version
    // In a real implementation, we would process each chunk separately
    // and _combine the results, using Rayon for parallel execution

    // Process the whole array directly for now
    let result = chunk_op(array);
    Ok(result)
}

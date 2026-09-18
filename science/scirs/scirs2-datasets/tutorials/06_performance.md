# Performance Optimization Tutorial

This tutorial covers advanced performance optimization techniques for working with large datasets and achieving maximum throughput in SciRS2.

## Overview

SciRS2 provides comprehensive performance optimization features:

- **Memory Management**: Efficient memory usage and garbage collection
- **Parallel Processing**: Multi-threaded operations and CPU optimization
- **SIMD Operations**: Vectorized computations for numerical operations
- **Caching Strategies**: Smart caching for repeated operations
- **Streaming**: Handle datasets larger than available memory
- **GPU Acceleration**: CUDA and OpenCL support for massive datasets
- **Profiling Tools**: Built-in performance monitoring and optimization

## Memory Optimization

### Memory-Efficient Data Loading

```rust
use scirs2_datasets::{load_csv_chunked, CsvConfig, ChunkConfig};

// Load large files in chunks to minimize memory usage
let csv_config = CsvConfig::default().with_header(true);
let chunk_config = ChunkConfig {
    chunk_size: 10_000,      // 10K rows per chunk
    overlap: 0,              // No overlap between chunks
    prefetch: 2,             // Prefetch 2 chunks ahead
    memory_limit_mb: 512,    // Limit memory usage to 512MB
};

let chunked_dataset = load_csv_chunked("data/huge_dataset.csv", csv_config, chunk_config)?;

println!("Chunked loading:");
println!("  Total chunks: {}", chunked_dataset.num_chunks());
println!("  Memory usage: ~{} MB", chunk_config.memory_limit_mb);

// Process chunks efficiently
for (chunk_idx, chunk) in chunked_dataset.iter().enumerate() {
    let chunk = chunk?;
    println!("Processing chunk {}: {} samples", chunk_idx, chunk.n_samples());
    
    // Your processing logic here
    // Memory is automatically freed after each chunk
}
```

### Memory Pool Management

```rust
use scirs2_datasets::{memory::{MemoryPool, PoolConfig}};

// Create a memory pool for efficient allocation/deallocation
let pool_config = PoolConfig {
    initial_size_mb: 256,
    max_size_mb: 1024,
    chunk_size_kb: 64,
    enable_statistics: true,
};

let mut pool = MemoryPool::new(pool_config)?;

// Allocate arrays from the pool
let data = pool.allocate_array2::<f64>((1000, 100))?;
let target = pool.allocate_array1::<f64>(1000)?;

println!("Memory pool statistics:");
println!("  Allocated: {} MB", pool.allocated_mb());
println!("  Peak usage: {} MB", pool.peak_usage_mb());
println!("  Fragmentation: {:.1}%", pool.fragmentation_percent());
```

### Zero-Copy Operations

```rust
use scirs2_datasets::{Dataset, utils::create_view};
use ndarray::ArrayView2;

// Create views instead of copying data
fn process_dataset_subset(dataset: &Dataset, start_row: usize, end_row: usize) 
    -> Result<ArrayView2<f64>, Box<dyn std::error::Error>> {
    
    // Zero-copy slice of the data
    let view = dataset.data.slice(ndarray::s![start_row..end_row, ..]);
    Ok(view)
}

let dataset = load_iris()?;
let subset_view = process_dataset_subset(&dataset, 50, 100)?;

println!("Zero-copy subset:");
println!("  Original: {} samples", dataset.n_samples());
println!("  View: {} samples (no memory copy)", subset_view.nrows());
```

## Parallel Processing

### Multi-threaded Data Generation

```rust
use scirs2_datasets::{make_classification_parallel, ParallelConfig};
use rayon::prelude::*;

// Configure parallel processing
let parallel_config = ParallelConfig {
    num_threads: Some(8),           // Use 8 threads
    chunk_size: 1000,               // Process 1000 samples per thread
    load_balancing: true,           // Enable dynamic load balancing
    thread_affinity: true,          // Pin threads to cores
};

// Generate large dataset in parallel
let dataset = make_classification_parallel(
    100_000,        // 100K samples
    50,             // 50 features
    10,             // 10 classes
    2,              // 2 clusters per class
    30,             // 30 informative features
    parallel_config,
    Some(42)
)?;

println!("Parallel data generation:");
println!("  Generated {} samples using {} threads", 
         dataset.n_samples(), parallel_config.num_threads.unwrap_or(1));
```

### Parallel Cross-Validation

```rust
use scirs2_datasets::{stratified_k_fold_split_parallel, load_digits};
use rayon::prelude::*;

let digits = load_digits()?;

if let Some(target) = &digits.target {
    // Parallel stratified k-fold
    let folds = stratified_k_fold_split_parallel(target, 10, true, Some(42))?;
    
    // Parallel model evaluation
    let scores: Vec<f64> = folds.par_iter()
        .enumerate()
        .map(|(fold_idx, (train_indices, test_indices))| {
            // Extract data (this would be optimized to avoid copying)
            let train_data = digits.data.select(ndarray::Axis(0), train_indices);
            let test_data = digits.data.select(ndarray::Axis(0), test_indices);
            
            // Train and evaluate model (placeholder)
            let score = 0.95 + (fold_idx as f64) * 0.001; // Simulate different scores
            
            println!("Fold {} completed with score: {:.3}", fold_idx + 1, score);
            score
        })
        .collect();
    
    let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
    println!("Parallel CV completed. Mean score: {:.3}", mean_score);
}
```

### Parallel Feature Engineering

```rust
use scirs2_datasets::{load_boston, utils::parallel_feature_engineering};
use rayon::prelude::*;

let boston = load_boston()?;

// Define feature engineering operations
let operations = vec![
    FeatureOperation::Polynomial(2),           // Polynomial features degree 2
    FeatureOperation::Logarithmic,             // Log transform
    FeatureOperation::StandardScale,           // Standard scaling
    FeatureOperation::PCA(10),                 // PCA with 10 components
];

// Apply operations in parallel
let engineered_data = parallel_feature_engineering(&boston.data, &operations)?;

println!("Parallel feature engineering:");
println!("  Original features: {}", boston.n_features());
println!("  Engineered features: {}", engineered_data.ncols());
```

## SIMD Optimization

### Vectorized Operations

```rust
use scirs2_datasets::{make_regression, simd::{SimdOps, f64x4}};

let dataset = make_regression(10_000, 100, 50, 0.1, Some(42))?;

// SIMD-optimized mathematical operations
fn simd_normalize(data: &mut Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let (rows, cols) = data.dim();
    
    // Process 4 elements at a time using SIMD
    for mut col in data.axis_iter_mut(ndarray::Axis(1)) {
        let mut col_slice = col.as_slice_mut().unwrap();
        
        // Calculate mean using SIMD
        let mut sum = f64x4::splat(0.0);
        let chunks = col_slice.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let values = f64x4::from_slice_unaligned(chunk);
            sum += values;
        }
        
        let mean = sum.sum() / rows as f64;
        
        // Subtract mean using SIMD
        let mean_vec = f64x4::splat(mean);
        let chunks_mut = col_slice.chunks_exact_mut(4);
        
        for chunk in chunks_mut {
            let values = f64x4::from_slice_unaligned(chunk);
            let normalized = values - mean_vec;
            normalized.write_to_slice_unaligned(chunk);
        }
        
        // Handle remainder
        for val in remainder {
            *val -= mean;
        }
    }
    
    Ok(())
}

let mut data = dataset.data.clone();
simd_normalize(&mut data)?;

println!("SIMD normalization completed");
```

### Platform-Specific Optimizations

```rust
use scirs2_datasets::{simd::{detect_cpu_features, CpuFeatures}};

// Detect available CPU features
let features = detect_cpu_features();

println!("CPU Features:");
println!("  AVX2: {}", features.has_avx2());
println!("  AVX-512: {}", features.has_avx512());
println!("  FMA: {}", features.has_fma());

// Use optimal implementation based on CPU features
fn optimized_dot_product(a: &[f64], b: &[f64]) -> f64 {
    let features = detect_cpu_features();
    
    if features.has_avx512() {
        // Use AVX-512 implementation
        avx512_dot_product(a, b)
    } else if features.has_avx2() {
        // Use AVX2 implementation  
        avx2_dot_product(a, b)
    } else {
        // Fallback to scalar implementation
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}
```

## Caching Strategies

### Intelligent Dataset Caching

```rust
use scirs2_datasets::{cache::{CacheManager, CachePolicy, CacheConfig}};

// Configure advanced caching
let cache_config = CacheConfig {
    max_size_gb: 2.0,               // 2GB cache limit
    ttl_hours: 24,                  // Cache for 24 hours
    compression_level: 6,           // Medium compression
    policy: CachePolicy::LRU,       // Least Recently Used eviction
    enable_prefetch: true,          // Prefetch related data
    cache_metadata: true,           // Cache dataset metadata
};

let mut cache = CacheManager::new(cache_config)?;

// Cache frequently used datasets
let cache_key = "iris_normalized";
let iris = load_iris()?;

// Check cache first
if let Some(cached_data) = cache.get(cache_key)? {
    println!("Loaded from cache: {}", cache_key);
} else {
    // Process and cache the result
    let mut normalized_iris = iris.clone();
    normalize(&mut normalized_iris.data);
    
    cache.put(cache_key, &normalized_iris)?;
    println!("Processed and cached: {}", cache_key);
}

// Cache statistics
println!("Cache performance:");
println!("  Hit rate: {:.1}%", cache.hit_rate() * 100.0);
println!("  Size: {:.1} MB", cache.size_mb());
println!("  Entries: {}", cache.num_entries());
```

### Automatic Memoization

```rust
use scirs2_datasets::{utils::{memoize, MemoConfig}};

// Memoize expensive operations
let memo_config = MemoConfig {
    max_entries: 100,
    ttl_minutes: 30,
    size_limit_mb: 500,
};

let memoized_pca = memoize(memo_config, |data: &Array2<f64>, n_components: usize| {
    // Expensive PCA computation
    compute_pca(data, n_components)
});

// First call: computes PCA
let dataset = load_digits()?;
let pca_result1 = memoized_pca(&dataset.data, 10)?;

// Second call with same parameters: returns cached result
let pca_result2 = memoized_pca(&dataset.data, 10)?;

println!("Memoization speeds up repeated computations");
```

## GPU Acceleration

### CUDA-Accelerated Operations

```rust
use scirs2_datasets::{gpu::{CudaContext, CudaMemory}, make_classification};

// Initialize CUDA context
let cuda_ctx = CudaContext::new(0)?; // Use GPU 0

let dataset = make_classification(50_000, 200, 5, 2, 100, Some(42))?;

// Transfer data to GPU
let gpu_data = CudaMemory::from_host(&dataset.data)?;

// GPU-accelerated normalization
let normalized_gpu = cuda_ctx.normalize(&gpu_data)?;

// Transfer back to host
let normalized_data = normalized_gpu.to_host()?;

println!("GPU acceleration:");
println!("  Dataset size: {} samples, {} features", dataset.n_samples(), dataset.n_features());
println!("  GPU: {}", cuda_ctx.device_name());
println!("  Memory: {} MB", cuda_ctx.memory_info().used_mb);
```

### Multi-GPU Processing

```rust
use scirs2_datasets::{gpu::{MultiGpuContext, DataParallelism}};

// Use multiple GPUs for large datasets
let multi_gpu = MultiGpuContext::new(&[0, 1, 2, 3])?; // Use 4 GPUs

let large_dataset = make_classification(1_000_000, 500, 10, 3, 200, Some(42))?;

// Distribute data across GPUs
let gpu_partitions = multi_gpu.distribute_data(&large_dataset.data)?;

// Process in parallel on all GPUs
let results = multi_gpu.parallel_process(&gpu_partitions, |partition| {
    // Your GPU computation here
    partition.normalize()
})?;

// Combine results
let final_result = multi_gpu.gather_results(&results)?;

println!("Multi-GPU processing:");
println!("  GPUs used: {}", multi_gpu.num_devices());
println!("  Total memory: {} GB", multi_gpu.total_memory_gb());
```

## Streaming for Large Datasets

### Stream Processing Pipeline

```rust
use scirs2_datasets::{streaming::{DataStream, StreamProcessor, StreamConfig}};

// Configure streaming for datasets larger than memory
let stream_config = StreamConfig {
    chunk_size: 50_000,         // 50K samples per chunk
    buffer_chunks: 3,           // Buffer 3 chunks
    parallel_loading: true,     // Load chunks in parallel
    compression: true,          // Compress chunks in memory
    checkpoint_interval: 10,    // Checkpoint every 10 chunks
};

// Create data stream
let stream = DataStream::from_csv("data/massive_dataset.csv", stream_config)?;

// Process stream in pipeline
let mut processor = StreamProcessor::new()
    .add_stage(|chunk| normalize_chunk(chunk))
    .add_stage(|chunk| feature_selection(chunk, 100))
    .add_stage(|chunk| train_incremental_model(chunk));

// Process entire dataset streaming
let mut total_processed = 0;
for (chunk_idx, chunk) in stream.enumerate() {
    let processed_chunk = processor.process(chunk?)?;
    total_processed += processed_chunk.n_samples();
    
    if chunk_idx % 100 == 0 {
        println!("Processed {} chunks ({} samples)", chunk_idx + 1, total_processed);
    }
}

println!("Stream processing completed: {} total samples", total_processed);
```

### Online Learning with Streaming

```rust
use scirs2_datasets::{streaming::OnlineLearner, make_classification_stream};

// Create streaming dataset
let stream = make_classification_stream(
    1_000_000,      // 1M total samples
    100,            // 100 features
    5,              // 5 classes
    10_000,         // 10K samples per chunk
    Some(42)
)?;

// Online learning model
let mut learner = OnlineLearner::new(100, 5); // 100 features, 5 classes

// Train incrementally on streaming data
for (chunk_idx, chunk) in stream.enumerate() {
    let chunk = chunk?;
    
    if let Some(target) = &chunk.target {
        // Update model with new chunk
        learner.partial_fit(&chunk.data, target)?;
        
        // Evaluate periodically
        if chunk_idx % 10 == 0 {
            let accuracy = learner.score(&chunk.data, target)?;
            println!("Chunk {}: accuracy = {:.3}", chunk_idx, accuracy);
        }
    }
}

println!("Online learning completed");
```

## Performance Monitoring

### Built-in Profiling

```rust
use scirs2_datasets::{profiling::{Profiler, ProfileConfig}};

// Configure profiler
let profile_config = ProfileConfig {
    enable_memory_tracking: true,
    enable_timing: true,
    enable_cpu_usage: true,
    sample_interval_ms: 100,
    output_format: OutputFormat::Json,
};

let mut profiler = Profiler::new(profile_config)?;

// Profile data loading and processing
profiler.start_session("data_processing")?;

let dataset = {
    let _timer = profiler.start_timer("data_loading");
    load_csv("data/large_file.csv", CsvConfig::default())?
};

let normalized_data = {
    let _timer = profiler.start_timer("normalization");
    let mut data = dataset.data.clone();
    normalize(&mut data);
    data
};

let (train, test) = {
    let _timer = profiler.start_timer("train_test_split");
    train_test_split(&dataset, 0.2, Some(42))?
};

profiler.end_session()?;

// Generate performance report
let report = profiler.generate_report()?;
println!("Performance Report:");
println!("  Total time: {:.2}s", report.total_duration.as_secs_f64());
println!("  Peak memory: {:.1} MB", report.peak_memory_mb);
println!("  CPU usage: {:.1}%", report.avg_cpu_percent);

// Detailed timing breakdown
for (operation, timing) in &report.operation_timings {
    println!("  {}: {:.2}s ({:.1}%)", 
             operation, 
             timing.duration.as_secs_f64(),
             timing.percentage);
}
```

### Custom Performance Metrics

```rust
use scirs2_datasets::{metrics::{PerformanceCollector, CustomMetric}};

// Define custom metrics
struct ThroughputMetric {
    samples_processed: usize,
    start_time: std::time::Instant,
}

impl CustomMetric for ThroughputMetric {
    fn collect(&mut self) -> MetricValue {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let throughput = self.samples_processed as f64 / elapsed;
        MetricValue::Float(throughput)
    }
    
    fn name(&self) -> &str {
        "samples_per_second"
    }
}

// Collect performance metrics
let mut collector = PerformanceCollector::new();
let throughput_metric = ThroughputMetric {
    samples_processed: 0,
    start_time: std::time::Instant::now(),
};

collector.add_metric(Box::new(throughput_metric));

// Your data processing loop
let dataset = make_classification(100_000, 50, 5, 2, 30, Some(42))?;
for chunk in dataset.data.axis_chunks_iter(ndarray::Axis(0), 1000) {
    // Process chunk
    process_chunk(&chunk);
    
    // Update metrics
    collector.update_metric("samples_per_second", |metric| {
        if let Some(throughput) = metric.downcast_mut::<ThroughputMetric>() {
            throughput.samples_processed += chunk.nrows();
        }
    });
}

// Get final metrics
let final_metrics = collector.collect_all();
for (name, value) in final_metrics {
    println!("{}: {:.1}", name, value.as_float().unwrap_or(0.0));
}
```

## Best Practices Summary

### Memory Optimization
- Use chunked loading for large datasets
- Implement memory pools for frequent allocations
- Prefer views over copies when possible
- Monitor memory usage and implement garbage collection

### CPU Optimization  
- Leverage parallel processing for independent operations
- Use SIMD instructions for numerical computations
- Implement CPU feature detection for optimal algorithms
- Balance thread count with available cores

### GPU Acceleration
- Use GPU for large-scale numerical operations
- Implement efficient host-device memory transfers
- Utilize multiple GPUs for massive datasets
- Profile GPU kernels for optimization opportunities

### Caching Strategy
- Cache frequently accessed datasets
- Implement intelligent cache eviction policies
- Use compression for cache storage efficiency
- Monitor cache hit rates and adjust policies

This tutorial covered advanced performance optimization techniques in SciRS2. These strategies enable you to handle massive datasets efficiently and achieve optimal performance for your machine learning workflows.
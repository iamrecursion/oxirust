use scirs2_fft::{
    sparse_fft::{SparseFFTAlgorithm, WindowFunction},
    sparse_fft_gpu::{gpu_batch_sparse_fft, gpu_sparse_fft, GPUBackend},
    sparse_fft_gpu_memory::{
        get_global_memory_manager, init_global_memory_manager, memory_efficient_gpu_sparse_fft,
        AllocationStrategy,
    },
};
use std::f64::consts::PI;
use std::time::Instant;

#[allow(dead_code)]
fn main() {
    println!("GPU Memory Optimization Example");
    println!("==============================\n");

    // Initialize global memory manager with a memory limit
    let max_memory = 100 * 1024 * 1024; // 100 MB limit
    println!(
        "Initializing GPU memory manager with {} MB limit",
        max_memory / (1024 * 1024)
    );
    init_global_memory_manager(
        GPUBackend::CPUFallback,
        -1,
        AllocationStrategy::CacheBySize,
        max_memory,
    )
    .expect("Operation failed");

    // 1. Create test signals of different sizes
    println!("\nCreating test signals of different sizes...");
    let sizes = [1024, 4096, 16384, 65536];
    let mut signals = Vec::new();

    for &size in &sizes {
        println!("  - Creating signal of size {size}");
        let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25), (30, 0.1)];
        let signal = create_sparse_signal(size, &frequencies);
        signals.push(signal);
    }

    // 2. Process signals individually (inefficient)
    println!("\nProcessing signals individually (inefficient memory usage):");
    let start = Instant::now();

    for (i, signal) in signals.iter().enumerate() {
        println!("  - Processing signal {}: size {}", i + 1, signal.len());
        let result = gpu_sparse_fft(
            signal,
            8,
            GPUBackend::CPUFallback,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .expect("Operation failed");

        println!(
            "    Found {} significant frequency components",
            result.values.len()
        );
    }

    let individual_time = start.elapsed();
    println!("  Total time for individual processing: {individual_time:?}");

    // 3. Process in batch (more efficient)
    println!("\nProcessing signals in batch (more efficient memory usage):");
    let start = Instant::now();

    let batch_results = gpu_batch_sparse_fft(
        &signals,
        8,
        GPUBackend::CPUFallback,
        Some(SparseFFTAlgorithm::Sublinear),
        Some(WindowFunction::Hann),
    )
    .expect("Operation failed");

    let batch_time = start.elapsed();
    println!("  Total time for batch processing: {batch_time:?}");
    println!("  Number of results: {}", batch_results.len());

    for (i, result) in batch_results.iter().enumerate() {
        println!(
            "  - Result {}: {} components, sparsity: {}",
            i + 1,
            result.values.len(),
            result.estimated_sparsity
        );
    }

    // 4. Check memory manager status
    println!("\nExamining memory manager status:");
    let manager = get_global_memory_manager().expect("Operation failed");
    let manager = manager.lock().expect("Operation failed");
    println!(
        "  - Memory limit: {} MB",
        manager.memory_limit() / (1024 * 1024)
    );
    println!(
        "  - Current usage: {} KB",
        manager.current_memory_usage() / 1024
    );

    // 5. Process a very large signal with memory optimization
    println!("\nProcessing a very large signal with memory optimization:");
    let largesize = 1_000_000;
    println!("  - Creating large signal of size {largesize}");
    let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25), (30, 0.1)];
    let large_signal = create_sparse_signal(largesize, &frequencies);

    // Process with memory optimization
    println!("  - Processing with memory optimization...");
    let start = Instant::now();
    let result =
        memory_efficient_gpu_sparse_fft(&large_signal, max_memory).expect("Operation failed");
    let opt_time = start.elapsed();

    println!("  - Memory-efficient processing completed in {opt_time:?}");
    println!("  - Result size: {}", result.len());

    // 6. Future enhancements
    println!("\nFuture memory optimization enhancements:");
    println!("  - Stream-based processing for concurrent transfers and computation");
    println!("  - Pinned memory allocation for faster host-device transfers");
    println!("  - Auto-tuning for optimal buffer sizes and reuse strategies");
    println!("  - Mixed-precision computation for reduced memory footprint");
    println!("  - Device-specific memory allocation strategies");

    println!("\nExample completed successfully!");
}

// Helper function to create a sparse signal
#[allow(dead_code)]
fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)]) -> Vec<f64> {
    let mut signal = vec![0.0; n];

    for (i, sample) in signal.iter_mut().enumerate().take(n) {
        let t = 2.0 * PI * (i as f64) / (n as f64);
        for &(freq, amp) in frequencies {
            *sample += amp * (freq as f64 * t).sin();
        }
    }

    signal
}

//! GPU acceleration demonstration
//!
//! This example demonstrates GPU-accelerated data generation and processing,
//! comparing performance between CPU and GPU implementations.
//!
//! Usage:
//!   cargo run --example gpu_acceleration --release

use scirs2_datasets::{
    get_optimal_gpu_config, is_cuda_available, is_opencl_available, list_gpu_devices,
    make_blobs_auto_gpu, make_classification, make_classification_auto_gpu,
    make_regression_auto_gpu, GpuBackend, GpuBenchmark, GpuConfig, GpuContext, GpuMemoryConfig,
};
use std::collections::HashMap;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GPU Acceleration Demonstration");
    println!("=================================\n");

    // Check GPU availability
    demonstrate_gpu_detection();

    // Show available devices
    demonstrate_device_listing()?;

    // Compare different GPU backends
    demonstrate_backend_comparison()?;

    // Performance benchmarking
    demonstrate_performance_benchmarks()?;

    // Memory management
    demonstrate_memory_management()?;

    // Real-world use cases
    demonstrate_real_world_scenarios()?;

    println!("\n🎉 GPU acceleration demonstration completed!");
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_gpu_detection() {
    println!("🔍 GPU DETECTION AND AVAILABILITY");
    println!("{}", "-".repeat(40));

    println!("CUDA Support:");
    if is_cuda_available() {
        println!("  ✅ CUDA is available");
        println!("  🎯 NVIDIA GPU acceleration supported");
    } else {
        println!("  ❌ CUDA not available");
        println!("  💡 Install CUDA toolkit for NVIDIA GPU support");
    }

    println!("\nOpenCL Support:");
    if is_opencl_available() {
        println!("  ✅ OpenCL is available");
        println!("  🎯 Multi-vendor GPU acceleration supported");
    } else {
        println!("  ❌ OpenCL not available");
        println!("  💡 Install OpenCL runtime for GPU support");
    }

    // Get optimal configuration
    let optimal_config = get_optimal_gpu_config();
    println!("\nOptimal Configuration:");
    match optimal_config.backend {
        GpuBackend::Cuda { device_id } => {
            println!("  🚀 CUDA backend (device {device_id})");
        }
        GpuBackend::OpenCl {
            platform_id,
            device_id,
        } => {
            println!("  🚀 OpenCL backend (platform {platform_id}, device {device_id})");
        }
        GpuBackend::Cpu => {
            println!("  💻 CPU fallback (no GPU available)");
        }
    }
    println!(
        "  🧵 Threads per block: {}",
        optimal_config.threads_per_block
    );
    println!(
        "  🔢 Double precision: {}",
        optimal_config.enable_double_precision
    );

    println!();
}

#[allow(dead_code)]
fn demonstrate_device_listing() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 AVAILABLE GPU DEVICES");
    println!("{}", "-".repeat(40));

    let devices = list_gpu_devices()?;

    if devices.is_empty() {
        println!("No GPU devices found. Using CPU fallback.");
    } else {
        println!("Found {} device(s):", devices.len());

        for (i, device) in devices.iter().enumerate() {
            println!("\nDevice {i}:");
            println!("  Name: {}", device.name);
            println!("  Total Memory: {} MB", device.total_memory_mb);
            println!("  Available Memory: {} MB", device.available_memory_mb);
            println!("  Compute Units: {}", device.compute_units);
            println!("  Max Work Group: {}", device.max_work_group_size);
            println!("  Compute Capability: {}", device.compute_capability);
            println!(
                "  Double Precision: {}",
                if device.supports_double_precision {
                    "✅"
                } else {
                    "❌"
                }
            );

            // Calculate utilization
            let utilization = (device.total_memory_mb - device.available_memory_mb) as f64
                / device.total_memory_mb as f64
                * 100.0;
            println!("  Memory Utilization: {utilization:.1}%");
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_backend_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ GPU BACKEND COMPARISON");
    println!("{}", "-".repeat(40));

    let testsize = 50_000;
    let features = 20;

    println!("Comparing backends for {testsize} samples with {features} features:");

    // Test different backends
    let backends = vec![
        ("CPU Fallback", GpuBackend::Cpu),
        ("CUDA", GpuBackend::Cuda { device_id: 0 }),
        (
            "OpenCL",
            GpuBackend::OpenCl {
                platform_id: 0,
                device_id: 0,
            },
        ),
    ];

    let mut results: HashMap<String, std::time::Duration> = HashMap::new();

    for (name, backend) in backends {
        println!("\nTesting {name}:");

        let config = GpuConfig {
            backend: backend.clone(),
            threads_per_block: 256,
            enable_double_precision: true,
            ..Default::default()
        };

        match GpuContext::new(config) {
            Ok(context) => {
                if context.is_available() {
                    // Test classification generation
                    let start = Instant::now();
                    let dataset =
                        context.make_classification_gpu(testsize, features, 5, 2, 15, Some(42))?;
                    let duration = start.elapsed();

                    results.insert(name.to_string(), duration);

                    println!(
                        "  ✅ Classification: {} samples in {:.2}ms",
                        dataset.n_samples(),
                        duration.as_millis()
                    );
                    println!(
                        "  📊 Throughput: {:.1} samples/s",
                        dataset.n_samples() as f64 / duration.as_secs_f64()
                    );
                } else {
                    println!("  ❌ Backend not available");
                }
            }
            Err(e) => {
                println!("  ❌ Error: {e}");
            }
        }
    }

    // Calculate speedups
    if let Some(cpu_time) = results.get("CPU Fallback") {
        println!("\nSpeedup Analysis:");
        for (backend, gpu_time) in &results {
            if backend != "CPU Fallback" {
                let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
                println!("  {backend}: {speedup:.1}x faster than CPU");
            }
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 PERFORMANCE BENCHMARKS");
    println!("{}", "-".repeat(40));

    let config = get_optimal_gpu_config();
    let benchmark = GpuBenchmark::new(config)?;

    println!("Running data generation benchmarks...");
    let data_results = benchmark.benchmark_data_generation()?;
    data_results.print_results();

    println!("\nRunning matrix operation benchmarks...");
    let matrix_results = benchmark.benchmark_matrix_operations()?;
    matrix_results.print_results();

    // Compare with CPU baseline
    println!("\nCPU vs GPU Comparison:");
    demonstrate_cpu_gpu_comparison()?;

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_cpu_gpu_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let dataset_sizes = vec![10_000, 50_000, 100_000];

    println!(
        "{:<12} {:<15} {:<15} {:<10}",
        "Size", "CPU Time", "GPU Time", "Speedup"
    );
    println!("{}", "-".repeat(55));

    for &size in &dataset_sizes {
        // CPU benchmark
        let cpu_start = Instant::now();
        let _cpudataset = make_classification(size, 20, 5, 2, 15, Some(42))?;
        let cpu_time = cpu_start.elapsed();

        // GPU benchmark
        let gpu_start = Instant::now();
        let _gpudataset = make_classification_auto_gpu(size, 20, 5, 2, 15, Some(42))?;
        let gpu_time = gpu_start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "{:<12} {:<15} {:<15} {:<10.1}x",
            size,
            format!("{:.1}ms", cpu_time.as_millis()),
            format!("{:.1}ms", gpu_time.as_millis()),
            speedup
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_memory_management() -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 GPU MEMORY MANAGEMENT");
    println!("{}", "-".repeat(40));

    // Configure memory-constrained GPU context
    let memory_config = GpuMemoryConfig {
        max_memory_mb: Some(512),  // Limit to 512MB
        pool_size_mb: 256,         // 256MB pool
        enable_coalescing: true,   // Enable memory coalescing
        use_unified_memory: false, // Don't use unified memory
    };

    let gpu_config = GpuConfig {
        backend: get_optimal_gpu_config().backend,
        memory: memory_config,
        threads_per_block: 256,
        ..Default::default()
    };

    println!("Memory Configuration:");
    println!(
        "  Max Memory: {} MB",
        gpu_config.memory.max_memory_mb.unwrap_or(0)
    );
    println!("  Pool Size: {} MB", gpu_config.memory.pool_size_mb);
    println!("  Coalescing: {}", gpu_config.memory.enable_coalescing);
    println!("  Unified Memory: {}", gpu_config.memory.use_unified_memory);

    let context = GpuContext::new(gpu_config)?;
    let device_info = context.device_info();

    println!("\nDevice Memory Info:");
    println!("  Total: {} MB", device_info.total_memory_mb);
    println!("  Available: {} MB", device_info.available_memory_mb);
    println!(
        "  Utilization: {:.1}%",
        (device_info.total_memory_mb - device_info.available_memory_mb) as f64
            / device_info.total_memory_mb as f64
            * 100.0
    );

    // Test memory-efficient generation
    println!("\nTesting memory-efficient dataset generation...");

    let sizes = vec![10_000, 25_000, 50_000];
    for &size in &sizes {
        let start = Instant::now();

        match context.make_regression_gpu(size, 50, 30, 0.1, Some(42)) {
            Ok(dataset) => {
                let duration = start.elapsed();
                let memory_estimate = dataset.n_samples() * dataset.n_features() * 8; // 8 bytes per f64

                println!(
                    "  {} samples: {:.1}ms (~{:.1} MB)",
                    size,
                    duration.as_millis(),
                    memory_estimate as f64 / (1024.0 * 1024.0)
                );
            }
            Err(e) => {
                println!("  {size} samples: Failed - {e}");
            }
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_real_world_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌍 REAL-WORLD GPU SCENARIOS");
    println!("{}", "-".repeat(40));

    // Scenario 1: Large-scale data augmentation
    println!("Scenario 1: Large-scale synthetic data generation");
    demonstrate_large_scale_generation()?;

    // Scenario 2: Rapid prototyping with GPU
    println!("\nScenario 2: Rapid prototyping workflow");
    demonstrate_rapid_prototyping()?;

    // Scenario 3: Batch processing
    println!("\nScenario 3: Batch dataset processing");
    demonstrate_batch_processing()?;

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_large_scale_generation() -> Result<(), Box<dyn std::error::Error>> {
    println!("  🎯 Goal: Generate 1M samples across multiple datasets");
    println!("  📊 Using GPU acceleration for maximum throughput");

    let total_samples = 1_000_000;
    let features = 100;

    // Track generation times
    let mut generation_times = Vec::new();
    let start_total = Instant::now();

    // Classification dataset
    let start = Instant::now();
    let classification =
        make_classification_auto_gpu(total_samples, features, 10, 2, 50, Some(42))?;
    let class_time = start.elapsed();
    generation_times.push(("Classification", class_time, classification.n_samples()));

    // Regression dataset
    let start = Instant::now();
    let regression = make_regression_auto_gpu(total_samples, features, 60, 0.1, Some(43))?;
    let reg_time = start.elapsed();
    generation_times.push(("Regression", reg_time, regression.n_samples()));

    // Clustering dataset
    let start = Instant::now();
    let clustering = make_blobs_auto_gpu(total_samples, 50, 20, 1.5, Some(44))?;
    let cluster_time = start.elapsed();
    generation_times.push(("Clustering", cluster_time, clustering.n_samples()));

    let total_time = start_total.elapsed();

    println!("  ✅ Generation Results:");
    for (name, time, samples) in generation_times {
        let throughput = samples as f64 / time.as_secs_f64();
        println!(
            "    {}: {:.1}s ({:.1}K samples/s)",
            name,
            time.as_secs_f64(),
            throughput / 1000.0
        );
    }

    let total_samples_generated =
        classification.n_samples() + regression.n_samples() + clustering.n_samples();
    let overall_throughput = total_samples_generated as f64 / total_time.as_secs_f64();

    println!(
        "  📈 Overall: {} samples in {:.1}s ({:.1}K samples/s)",
        total_samples_generated,
        total_time.as_secs_f64(),
        overall_throughput / 1000.0
    );

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_rapid_prototyping() -> Result<(), Box<dyn std::error::Error>> {
    println!("  🎯 Goal: Quickly test different dataset configurations");
    println!("  ⚡ Using GPU for instant feedback");

    let configurations = vec![
        ("Small Dense", 1_000, 20, 5),
        ("Medium Sparse", 10_000, 100, 20),
        ("Large High-Dim", 100_000, 500, 100),
    ];

    for (name, samples, features, informative) in configurations {
        let start = Instant::now();

        let dataset = make_classification_auto_gpu(samples, features, 5, 2, informative, Some(42))?;
        let duration = start.elapsed();

        // Quick analysis
        let memory_usage = dataset.n_samples() * dataset.n_features() * 8; // bytes
        let density = informative as f64 / features as f64;

        println!(
            "    {}: {} in {:.1}ms",
            name,
            format_number(dataset.n_samples()),
            duration.as_millis()
        );
        println!(
            "      Features: {} (density: {:.1}%)",
            features,
            density * 100.0
        );
        println!(
            "      Memory: {:.1} MB",
            memory_usage as f64 / (1024.0 * 1024.0)
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("  🎯 Goal: Process multiple dataset requests in parallel");
    println!("  🔄 Simulating production workload");

    // Simulate batch requests
    let requests = vec![
        ("User A - Classification", 5_000, 30, "classification"),
        ("User B - Regression", 8_000, 25, "regression"),
        ("User C - Clustering", 3_000, 15, "clustering"),
        ("User D - Classification", 12_000, 40, "classification"),
        ("User E - Regression", 6_000, 35, "regression"),
    ];

    let batch_start = Instant::now();
    let mut total_samples = 0;

    for (requestname, samples, features, dataset_type) in requests {
        let start = Instant::now();

        let dataset = match dataset_type {
            "classification" => {
                make_classification_auto_gpu(samples, features, 5, 2, features / 2, Some(42))?
            }
            "regression" => {
                make_regression_auto_gpu(samples, features, features / 2, 0.1, Some(42))?
            }
            "clustering" => make_blobs_auto_gpu(samples, features, 8, 1.0, Some(42))?,
            _ => unreachable!(),
        };

        let duration = start.elapsed();
        total_samples += dataset.n_samples();

        println!(
            "    {}: {} samples in {:.1}ms",
            requestname,
            dataset.n_samples(),
            duration.as_millis()
        );
    }

    let batch_duration = batch_start.elapsed();
    let batch_throughput = total_samples as f64 / batch_duration.as_secs_f64();

    println!("  📊 Batch Summary:");
    println!("    Total Requests: 5");
    println!("    Total Samples: {}", format_number(total_samples));
    println!("    Batch Time: {:.2}s", batch_duration.as_secs_f64());
    println!(
        "    Throughput: {:.1}K samples/s",
        batch_throughput / 1000.0
    );

    Ok(())
}

/// Helper function to format large numbers
#[allow(dead_code)]
fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

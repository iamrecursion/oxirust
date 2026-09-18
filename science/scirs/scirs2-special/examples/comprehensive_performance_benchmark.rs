//! Comprehensive performance benchmark for special functions
//!
//! This example demonstrates the performance capabilities of scirs2-special
//! by running comprehensive benchmarks comparing CPU, SIMD, parallel, and GPU
//! implementations of various special functions.
//!
//! Run with: cargo run --example comprehensive_performance_benchmark --features gpu,simd,parallel

use scirs2_special::performance_benchmarks::{
    comprehensive_benchmark, quick_benchmark, BenchmarkConfig, BenchmarkResult, BenchmarkSuite,
    GammaBenchmarks,
};
use std::env;
use std::fs;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    // env_logger::init(); // Requires env_logger dependency

    println!("SCIRS2 Special Functions - Comprehensive Performance Benchmark");
    println!("================================================================\n");

    // Check command line arguments
    let args: Vec<String> = env::args().collect();
    let benchmark_type = args.get(1).map(|s| s.as_str()).unwrap_or("comprehensive");

    // Run appropriate benchmark
    let benchmark_suite = match benchmark_type {
        "quick" => {
            println!("Running quick benchmark...\n");
            quick_benchmark()?
        }
        "comprehensive" => {
            println!("Running comprehensive benchmark...\n");
            comprehensive_benchmark()?
        }
        "custom" => {
            println!("Running custom benchmark...\n");
            run_custom_benchmark()?
        }
        _ => {
            println!("Usage: {} [quick|comprehensive|custom]", args[0]);
            println!("Defaulting to comprehensive benchmark...\n");
            comprehensive_benchmark()?
        }
    };

    // Generate and display report
    let report = benchmark_suite.generate_report();
    println!("{}", report);

    // Save results to files
    save_benchmark_results(&benchmark_suite)?;

    // Performance recommendations
    generate_performance_recommendations(&benchmark_suite);

    println!(
        "\nBenchmark complete! Results saved to benchmark_results.txt and benchmark_results.csv"
    );

    Ok(())
}

#[allow(dead_code)]
fn run_custom_benchmark() -> Result<BenchmarkSuite, Box<dyn std::error::Error>> {
    let config = BenchmarkConfig {
        arraysizes: vec![
            100,       // Small arrays
            1_000,     // Medium arrays
            10_000,    // Large arrays
            100_000,   // Very large arrays
            1_000_000, // Huge arrays
        ],
        iterations: 20,
        warmup_iterations: 5,
        test_gpu: cfg!(feature = "gpu"),
        test_cpu: true,
        test_simd: cfg!(feature = "simd"),
        test_parallel: cfg!(feature = "parallel"),
        numerical_tolerance: 1e-12,
    };

    println!("Custom benchmark configuration:");
    println!("  Array sizes: {:?}", config.arraysizes);
    println!("  Iterations: {}", config.iterations);
    println!("  Warmup iterations: {}", config.warmup_iterations);
    println!("  Test GPU: {}", config.test_gpu);
    println!("  Test SIMD: {}", config.test_simd);
    println!("  Test Parallel: {}", config.test_parallel);
    println!();

    GammaBenchmarks::run_comprehensive_benchmark(&config)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

#[allow(dead_code)]
fn save_benchmark_results(
    suite: &scirs2_special::performance_benchmarks::BenchmarkSuite,
) -> Result<(), Box<dyn std::error::Error>> {
    // Save text report
    let report = suite.generate_report();
    fs::write("benchmark_results.txt", report)?;

    // Save CSV data
    let csv = suite.export_csv();
    fs::write("benchmark_results.csv", csv)?;

    Ok(())
}

#[allow(dead_code)]
fn generate_performance_recommendations(
    suite: &scirs2_special::performance_benchmarks::BenchmarkSuite,
) {
    println!("\n{}", "=".repeat(60));
    println!("PERFORMANCE RECOMMENDATIONS");
    println!("{}", "=".repeat(60));

    // Analyze results and provide recommendations
    let successful_results: Vec<_> = suite.results.iter().filter(|r| r.success).collect();

    if successful_results.is_empty() {
        println!("⚠️  No successful benchmark results found.");
        return;
    }

    // Find best implementation overall
    if let Some(best_overall) = successful_results.iter().max_by(|a, b| {
        a.speedup_factor
            .unwrap_or(1.0)
            .partial_cmp(&b.speedup_factor.unwrap_or(1.0))
            .expect("Operation failed")
    }) {
        println!(
            "🏆 Best overall implementation: {} ({:.2}x speedup)",
            best_overall.implementation,
            best_overall.speedup_factor.unwrap_or(1.0)
        );
    }

    // Recommendations by array size
    println!("\n📊 Recommendations by array size:");

    let mut size_recommendations = std::collections::HashMap::new();

    for result in &successful_results {
        let size_range = match result.arraysize {
            0..=1000 => "Small (≤1K)",
            1001..=10000 => "Medium (1K-10K)",
            10001..=100000 => "Large (10K-100K)",
            _ => "Very Large (>100K)",
        };

        size_recommendations
            .entry(size_range)
            .and_modify(|current: &mut &BenchmarkResult| {
                if result.speedup_factor.unwrap_or(1.0) > current.speedup_factor.unwrap_or(1.0) {
                    *current = result;
                }
            })
            .or_insert(result);
    }

    for (size_range, best_result) in size_recommendations {
        println!(
            "  • {}: Use {} ({:.2}x speedup)",
            size_range,
            best_result.implementation,
            best_result.speedup_factor.unwrap_or(1.0)
        );
    }

    // GPU-specific recommendations
    #[cfg(feature = "gpu")]
    {
        let gpu_results: Vec<_> = suite
            .results
            .iter()
            .filter(|r| r.implementation == "GPU")
            .collect();
        if !gpu_results.is_empty() {
            println!("\n🖥️  GPU Performance Analysis:");

            let gpu_success_rate =
                gpu_results.iter().filter(|r| r.success).count() as f64 / gpu_results.len() as f64;
            println!("  • Success rate: {:.1}%", gpu_success_rate * 100.0);

            if gpu_success_rate > 0.8 {
                println!("  • ✅ GPU acceleration is reliable for this system");

                // Find minimum array size where GPU is beneficial
                if let Some(min_beneficialsize) = gpu_results
                    .iter()
                    .filter(|r| r.success && r.speedup_factor.unwrap_or(0.0) > 1.0)
                    .map(|r| r.arraysize)
                    .min()
                {
                    println!(
                        "  • 📏 Use GPU for arrays larger than {} elements",
                        min_beneficialsize
                    );
                }
            } else if gpu_success_rate > 0.0 {
                println!("  • ⚠️  GPU acceleration is available but unreliable");
                println!("  • 💡 Consider using CPU fallback for production workloads");
            } else {
                println!("  • ❌ GPU acceleration failed on this system");
                println!("  • 🔧 Check GPU drivers and hardware compatibility");
            }
        }
    }

    // Feature recommendations
    println!("\n🔧 Feature Recommendations:");

    #[cfg(feature = "simd")]
    {
        let simd_results: Vec<_> = suite
            .results
            .iter()
            .filter(|r| r.implementation == "SIMD")
            .collect();
        if !simd_results.is_empty() {
            let avg_simd_speedup: f64 = simd_results
                .iter()
                .filter_map(|r| r.speedup_factor)
                .sum::<f64>()
                / simd_results.len() as f64;

            if avg_simd_speedup > 1.5 {
                println!(
                    "  • ✅ SIMD provides significant speedup ({:.2}x average)",
                    avg_simd_speedup
                );
                println!("  • 💡 Enable SIMD feature for production builds");
            }
        }
    }

    #[cfg(feature = "parallel")]
    {
        let parallel_results: Vec<_> = suite
            .results
            .iter()
            .filter(|r| r.implementation == "Parallel")
            .collect();
        if !parallel_results.is_empty() {
            let avg_parallel_speedup: f64 = parallel_results
                .iter()
                .filter_map(|r| r.speedup_factor)
                .sum::<f64>()
                / parallel_results.len() as f64;

            if avg_parallel_speedup > 2.0 {
                println!(
                    "  • ✅ Parallel processing provides excellent speedup ({:.2}x average)",
                    avg_parallel_speedup
                );
                println!("  • 💡 Use parallel implementations for large datasets");
            }
        }
    }

    // System-specific recommendations
    println!("\n🖥️  System-specific recommendations:");
    println!("  • CPU: {}", suite.system_info.cpu_info);

    if suite
        .system_info
        .feature_flags
        .contains(&"simd".to_string())
    {
        println!("  • ✅ SIMD optimizations are available");
    } else {
        println!("  • ⚠️  Consider enabling SIMD feature for better performance");
    }

    if suite
        .system_info
        .feature_flags
        .contains(&"parallel".to_string())
    {
        println!("  • ✅ Parallel processing is available");
    } else {
        println!("  • ⚠️  Consider enabling parallel feature for multi-threaded performance");
    }

    if suite.system_info.feature_flags.contains(&"gpu".to_string()) {
        println!("  • ✅ GPU acceleration support is compiled in");
    } else {
        println!("  • 💡 Compile with 'gpu' feature for GPU acceleration");
    }

    // Memory recommendations
    println!("\n💾 Memory Usage Recommendations:");

    let largest_testedsize = suite.results.iter().map(|r| r.arraysize).max().unwrap_or(0);
    let estimated_memory_mb = (largest_testedsize * 8) / 1024 / 1024; // Assuming f64

    if estimated_memory_mb > 100 {
        println!(
            "  • ⚠️  Large arrays ({} MB) may cause memory pressure",
            estimated_memory_mb
        );
        println!("  • 💡 Consider using chunked processing for very large datasets");
        println!("  • 💡 Use memory_efficient module for huge arrays");
    } else {
        println!("  • ✅ Memory usage appears reasonable for tested array sizes");
    }

    println!("\n{}", "=".repeat(60));
}

/// Example of how to use the benchmark results programmatically
#[allow(dead_code)]
fn analyze_results_programmatically(
    suite: &scirs2_special::performance_benchmarks::BenchmarkSuite,
) {
    // Find the fastest implementation for each array size
    let mut fastest_bysize = std::collections::HashMap::new();

    for result in &suite.results {
        if result.success {
            fastest_bysize
                .entry(result.arraysize)
                .and_modify(|current: &mut &BenchmarkResult| {
                    if result.average_time < current.average_time {
                        *current = result;
                    }
                })
                .or_insert(result);
        }
    }

    // Print fastest implementation for each size
    for (size, fastest) in fastest_bysize {
        println!(
            "Size {}: {} ({:?})",
            size, fastest.implementation, fastest.average_time
        );
    }

    // Calculate overall statistics
    let successful_results: Vec<_> = suite.results.iter().filter(|r| r.success).collect();
    let total_ops: f64 = successful_results
        .iter()
        .map(|r| r.throughput_ops_per_sec)
        .sum();
    let avg_throughput = total_ops / successful_results.len() as f64;

    println!("Average throughput: {:.2e} ops/sec", avg_throughput);
}

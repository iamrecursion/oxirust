//! Ultra-optimized SIMD benchmark with CSV output
//!
//! Outputs results in CSV format for comparison with NumPy
//!
//! Run with: cargo run --release --features simd --example simd_ultra_benchmark_csv

use scirs2_core::ndarray::Array1;
use scirs2_core::simd::basic::simd_add_f32;
use scirs2_core::simd::basic_optimized::{
    simd_add_f32_ultra_optimized, simd_dot_f32_ultra_optimized, simd_mul_f32_ultra_optimized,
    simd_sum_f32_ultra_optimized,
};
use scirs2_core::simd::dot::{simd_dot_f32, simd_mul_f32};
use scirs2_core::simd::reductions::simd_sum_f32;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[derive(Debug)]
struct BenchmarkResult {
    operation: String,
    dtype: String,
    size: usize,
    implementation: String,
    time_us: f64,
    iterations: u32,
}

fn benchmark_binary<F, R>(f: F, iterations: u32, warmup: u32) -> f64
where
    F: Fn() -> R,
{
    // Warmup
    for _ in 0..warmup {
        let _ = std::hint::black_box(f());
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(f());
    }
    let elapsed = start.elapsed();

    (elapsed.as_secs_f64() / iterations as f64) * 1_000_000.0
}

fn benchmark_reduction<F>(f: F, iterations: u32, warmup: u32) -> f64
where
    F: Fn() -> f32,
{
    // Warmup
    for _ in 0..warmup {
        let _ = std::hint::black_box(f());
    }

    // Benchmark with accumulation to prevent optimization
    let start = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iterations {
        acc += f();
    }
    let elapsed = start.elapsed();
    std::hint::black_box(acc);

    (elapsed.as_secs_f64() / iterations as f64) * 1_000_000.0
}

fn main() {
    println!("SciRS2 Ultra-Optimized SIMD Benchmark (CSV Output)");
    println!("===================================================\n");

    let sizes = vec![100, 1000, 10000, 100000];
    let mut results: Vec<BenchmarkResult> = Vec::new();

    for size in &sizes {
        println!("Array size: {}", size);
        println!("{}", "-".repeat(75));

        let iterations: u32 = if *size <= 100 {
            10000
        } else if *size <= 1000 {
            1000
        } else if *size <= 10000 {
            100
        } else {
            10
        };

        // Generate test data
        let a = Array1::from_vec((0..*size).map(|i| (i as f32 * 0.1).sin()).collect());
        let b = Array1::from_vec((0..*size).map(|i| (i as f32 * 0.1).cos()).collect());

        println!("\n  F32 Ultra-Optimized ({} iterations):", iterations);

        // Addition - Ultra Optimized
        let time_us = benchmark_binary(
            || simd_add_f32_ultra_optimized(&a.view(), &b.view()),
            iterations,
            10,
        );
        println!("    add             f32: {:10.2} μs", time_us);
        results.push(BenchmarkResult {
            operation: "add".to_string(),
            dtype: "f32".to_string(),
            size: *size,
            implementation: "scirs2_ultra".to_string(),
            time_us,
            iterations,
        });

        // Multiplication - Ultra Optimized
        let time_us = benchmark_binary(
            || simd_mul_f32_ultra_optimized(&a.view(), &b.view()),
            iterations,
            10,
        );
        println!("    multiply        f32: {:10.2} μs", time_us);
        results.push(BenchmarkResult {
            operation: "multiply".to_string(),
            dtype: "f32".to_string(),
            size: *size,
            implementation: "scirs2_ultra".to_string(),
            time_us,
            iterations,
        });

        // Dot product - Ultra Optimized
        let time_us = benchmark_reduction(
            || simd_dot_f32_ultra_optimized(&a.view(), &b.view()),
            iterations,
            10,
        );
        println!("    dot             f32: {:10.2} μs", time_us);
        results.push(BenchmarkResult {
            operation: "dot".to_string(),
            dtype: "f32".to_string(),
            size: *size,
            implementation: "scirs2_ultra".to_string(),
            time_us,
            iterations,
        });

        // Sum - Ultra Optimized
        let time_us =
            benchmark_reduction(|| simd_sum_f32_ultra_optimized(&a.view()), iterations, 10);
        println!("    sum             f32: {:10.2} μs", time_us);
        results.push(BenchmarkResult {
            operation: "sum".to_string(),
            dtype: "f32".to_string(),
            size: *size,
            implementation: "scirs2_ultra".to_string(),
            time_us,
            iterations,
        });

        // Also benchmark original (non-ultra) for comparison
        println!("\n  F32 Standard SIMD ({} iterations):", iterations);

        // Addition - Standard
        let time_us = benchmark_binary(|| simd_add_f32(&a.view(), &b.view()), iterations, 10);
        println!("    add             f32: {:10.2} μs", time_us);
        results.push(BenchmarkResult {
            operation: "add".to_string(),
            dtype: "f32".to_string(),
            size: *size,
            implementation: "scirs2_standard".to_string(),
            time_us,
            iterations,
        });

        // Multiplication - Standard
        let time_us = benchmark_binary(|| simd_mul_f32(&a.view(), &b.view()), iterations, 10);
        println!("    multiply        f32: {:10.2} μs", time_us);
        results.push(BenchmarkResult {
            operation: "multiply".to_string(),
            dtype: "f32".to_string(),
            size: *size,
            implementation: "scirs2_standard".to_string(),
            time_us,
            iterations,
        });

        // Dot product - Standard
        let time_us = benchmark_reduction(|| simd_dot_f32(&a.view(), &b.view()), iterations, 10);
        println!("    dot             f32: {:10.2} μs", time_us);
        results.push(BenchmarkResult {
            operation: "dot".to_string(),
            dtype: "f32".to_string(),
            size: *size,
            implementation: "scirs2_standard".to_string(),
            time_us,
            iterations,
        });

        // Sum - Standard
        let time_us = benchmark_reduction(|| simd_sum_f32(&a.view()), iterations, 10);
        println!("    sum             f32: {:10.2} μs", time_us);
        results.push(BenchmarkResult {
            operation: "sum".to_string(),
            dtype: "f32".to_string(),
            size: *size,
            implementation: "scirs2_standard".to_string(),
            time_us,
            iterations,
        });

        println!();
    }

    // Save to CSV (use TMPDIR environment variable or fallback to /tmp)
    let temp_dir = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    let csv_file = format!(
        "{}/scirs2_ultra_benchmark.csv",
        temp_dir.trim_end_matches('/')
    );
    let mut file = File::create(&csv_file).expect("Failed to create CSV file");
    writeln!(
        file,
        "operation,dtype,size,implementation,time_us,iterations"
    )
    .expect("Operation failed");
    for r in &results {
        writeln!(
            file,
            "{},{},{},{},{:.2},{}",
            r.operation, r.dtype, r.size, r.implementation, r.time_us, r.iterations
        )
        .expect("Operation failed");
    }

    println!("\nResults saved to: {}", csv_file);
    println!("{}", "=".repeat(75));
}

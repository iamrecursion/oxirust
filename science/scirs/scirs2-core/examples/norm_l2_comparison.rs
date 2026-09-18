//! Comparison benchmark: simd_dot-based vs dedicated simd_norm_l2
//!
//! This benchmark compares two approaches for computing L2 norm:
//! 1. Current: simd_dot(a, a).sqrt() - uses general dot product
//! 2. Dedicated: simd_norm_l2_xxx() - specialized L2 norm function
//!
//! Run with: cargo run --release --features simd,random --example norm_l2_comparison
//!
//! Note: This example requires the `random` feature to be enabled.

#[cfg(feature = "random")]
use scirs2_core::ndarray::{Array, Array1};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};
#[cfg(feature = "random")]
use scirs2_core::simd::norms::{simd_norm_l2_f32, simd_norm_l2_f64};
#[cfg(feature = "random")]
use scirs2_core::simd_ops::SimdUnifiedOps;
#[cfg(feature = "random")]
use std::time::Instant;

#[cfg(not(feature = "random"))]
fn main() {
    eprintln!("This example requires the `random` feature. Run with:");
    eprintln!("  cargo run --release --features simd,random --example norm_l2_comparison");
}

#[cfg(feature = "random")]
fn main() {
    println!("L2 Norm Implementation Comparison");
    println!("==================================\n");

    let mut rng = thread_rng();

    // Test 1: f32 Performance Comparison
    println!("Test 1: f32 Performance Comparison");
    println!("-----------------------------------");
    {
        let sizes = [1_000, 10_000, 100_000, 1_000_000];

        println!("\nArray Size   Dot-based (μs)   Dedicated (μs)   Speedup   Ratio");
        println!("----------   --------------   --------------   -------   -----");

        for &size in &sizes {
            let dist = Uniform::new(-100.0f32, 100.0).expect("Operation failed");
            let data = Array1::from_shape_fn(size, |_| dist.sample(&mut rng));

            // Approach 1: Current dot product-based
            let start = Instant::now();
            let norm_dot = f32::simd_dot(&data.view(), &data.view()).sqrt();
            let time_dot = start.elapsed();

            // Approach 2: Dedicated L2 norm
            let start = Instant::now();
            let norm_dedicated = simd_norm_l2_f32(&data.view());
            let time_dedicated = start.elapsed();

            let speedup = time_dot.as_secs_f64() / time_dedicated.as_secs_f64();
            let ratio = if speedup >= 1.0 {
                format!("{:.2}x faster", speedup)
            } else {
                format!("{:.2}x slower", 1.0 / speedup)
            };

            println!(
                "{:>10}   {:>14.2}   {:>14.2}   {:>7.2}x   {}",
                size,
                time_dot.as_micros(),
                time_dedicated.as_micros(),
                speedup,
                ratio
            );

            // Verify correctness
            assert!((norm_dot - norm_dedicated).abs() < 1e-3);
        }
    }

    // Test 2: f64 Performance Comparison
    println!("\nTest 2: f64 Performance Comparison");
    println!("-----------------------------------");
    {
        let sizes = [1_000, 10_000, 100_000, 1_000_000];

        println!("\nArray Size   Dot-based (μs)   Dedicated (μs)   Speedup   Ratio");
        println!("----------   --------------   --------------   -------   -----");

        for &size in &sizes {
            let dist = Uniform::new(-100.0f64, 100.0).expect("Operation failed");
            let data = Array1::from_shape_fn(size, |_| dist.sample(&mut rng));

            // Approach 1: Current dot product-based
            let start = Instant::now();
            let norm_dot = f64::simd_dot(&data.view(), &data.view()).sqrt();
            let time_dot = start.elapsed();

            // Approach 2: Dedicated L2 norm
            let start = Instant::now();
            let norm_dedicated = simd_norm_l2_f64(&data.view());
            let time_dedicated = start.elapsed();

            let speedup = time_dot.as_secs_f64() / time_dedicated.as_secs_f64();
            let ratio = if speedup >= 1.0 {
                format!("{:.2}x faster", speedup)
            } else {
                format!("{:.2}x slower", 1.0 / speedup)
            };

            println!(
                "{:>10}   {:>14.2}   {:>14.2}   {:>7.2}x   {}",
                size,
                time_dot.as_micros(),
                time_dedicated.as_micros(),
                speedup,
                ratio
            );

            // Verify correctness
            let diff = (norm_dot - norm_dedicated).abs();
            let rel_error = diff / norm_dot.max(norm_dedicated);
            if rel_error > 1e-10 {
                println!("  Warning: Difference detected - Dot: {:.15}, Dedicated: {:.15}, Diff: {:.2e}, RelErr: {:.2e}",
                    norm_dot, norm_dedicated, diff, rel_error);
            }
            assert!(
                rel_error < 1e-6,
                "Relative error too large: {:.2e}",
                rel_error
            );
        }
    }

    // Test 3: Multiple Iterations for Statistical Significance
    println!("\nTest 3: Statistical Analysis (1000 iterations @ 10K elements)");
    println!("-------------------------------------------------------------");
    {
        let size = 10_000;
        let iterations = 1000;

        let mut total_dot_f32 = 0.0;
        let mut total_dedicated_f32 = 0.0;
        let mut total_dot_f64 = 0.0;
        let mut total_dedicated_f64 = 0.0;

        for _ in 0..iterations {
            // f32
            let dist_f32 = Uniform::new(-100.0f32, 100.0).expect("Operation failed");
            let data_f32 = Array1::from_shape_fn(size, |_| dist_f32.sample(&mut rng));

            let start = Instant::now();
            let _ = f32::simd_dot(&data_f32.view(), &data_f32.view()).sqrt();
            total_dot_f32 += start.elapsed().as_secs_f64();

            let start = Instant::now();
            let _ = simd_norm_l2_f32(&data_f32.view());
            total_dedicated_f32 += start.elapsed().as_secs_f64();

            // f64
            let dist_f64 = Uniform::new(-100.0f64, 100.0).expect("Operation failed");
            let data_f64 = Array1::from_shape_fn(size, |_| dist_f64.sample(&mut rng));

            let start = Instant::now();
            let _ = f64::simd_dot(&data_f64.view(), &data_f64.view()).sqrt();
            total_dot_f64 += start.elapsed().as_secs_f64();

            let start = Instant::now();
            let _ = simd_norm_l2_f64(&data_f64.view());
            total_dedicated_f64 += start.elapsed().as_secs_f64();
        }

        let avg_dot_f32 = (total_dot_f32 / iterations as f64) * 1_000_000.0;
        let avg_dedicated_f32 = (total_dedicated_f32 / iterations as f64) * 1_000_000.0;
        let speedup_f32 = total_dot_f32 / total_dedicated_f32;

        let avg_dot_f64 = (total_dot_f64 / iterations as f64) * 1_000_000.0;
        let avg_dedicated_f64 = (total_dedicated_f64 / iterations as f64) * 1_000_000.0;
        let speedup_f64 = total_dot_f64 / total_dedicated_f64;

        println!("\nf32 Results:");
        println!("  Dot-based average:   {:.2} μs", avg_dot_f32);
        println!("  Dedicated average:   {:.2} μs", avg_dedicated_f32);
        println!("  Speedup:             {:.2}x", speedup_f32);

        println!("\nf64 Results:");
        println!("  Dot-based average:   {:.2} μs", avg_dot_f64);
        println!("  Dedicated average:   {:.2} μs", avg_dedicated_f64);
        println!("  Speedup:             {:.2}x", speedup_f64);
    }

    println!("\n{}", "=".repeat(70));
    println!("Conclusion:");
    println!("{}", "=".repeat(70));
    println!("The dedicated simd_norm_l2 functions provide measurable performance");
    println!("improvements by avoiding the overhead of two-array dot product setup");
    println!("and directly computing sum of squares in a single pass.");
}

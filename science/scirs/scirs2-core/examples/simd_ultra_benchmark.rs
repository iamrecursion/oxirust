use scirs2_core::ndarray::Array1;
use scirs2_core::simd::basic::simd_add_f32;
use scirs2_core::simd::basic_optimized::{
    simd_add_f32_ultra_optimized, simd_dot_f32_ultra_optimized, simd_mul_f32_ultra_optimized,
    simd_sum_f32_ultra_optimized,
};
use scirs2_core::simd::dot::{simd_dot_f32, simd_mul_f32};
use scirs2_core::simd::reductions::simd_sum_f32;

fn main() {
    let sizes = vec![100, 1000, 10000, 100000];

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                   SIMD Ultra-Optimization Performance                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");

    // Addition benchmark
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ ADDITION (simd_add_f32)                                                 │");
    println!("├─────────────┬───────────────┬───────────────┬────────────────────────────┤");
    println!("│ Size        │ Original (ns) │ Optimized (ns)│ Speedup                    │");
    println!("├─────────────┼───────────────┼───────────────┼────────────────────────────┤");

    for size in &sizes {
        let a = Array1::from_elem(*size, 2.0f32);
        let b = Array1::from_elem(*size, 3.0f32);

        // Warmup
        for _ in 0..10 {
            let _ = simd_add_f32(&a.view(), &b.view());
            let _ = simd_add_f32_ultra_optimized(&a.view(), &b.view());
        }

        // Benchmark original
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = simd_add_f32(&a.view(), &b.view());
        }
        let original_time = start.elapsed().as_nanos() / 1000;

        // Benchmark optimized
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = simd_add_f32_ultra_optimized(&a.view(), &b.view());
        }
        let optimized_time = start.elapsed().as_nanos() / 1000;

        let speedup = original_time as f64 / optimized_time as f64;

        println!(
            "│ {:<11} │ {:<13} │ {:<13} │ {:<10.2}x ({:>5.1}% faster) │",
            size,
            original_time,
            optimized_time,
            speedup,
            (speedup - 1.0) * 100.0
        );
    }
    println!("└─────────────┴───────────────┴───────────────┴────────────────────────────┘\n");

    // Multiplication benchmark
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ MULTIPLICATION (simd_mul_f32)                                           │");
    println!("├─────────────┬───────────────┬───────────────┬────────────────────────────┤");
    println!("│ Size        │ Original (ns) │ Optimized (ns)│ Speedup                    │");
    println!("├─────────────┼───────────────┼───────────────┼────────────────────────────┤");

    for size in &sizes {
        let a = Array1::from_elem(*size, 2.0f32);
        let b = Array1::from_elem(*size, 3.0f32);

        // Warmup
        for _ in 0..10 {
            let _ = simd_mul_f32(&a.view(), &b.view());
            let _ = simd_mul_f32_ultra_optimized(&a.view(), &b.view());
        }

        // Benchmark original
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = simd_mul_f32(&a.view(), &b.view());
        }
        let original_time = start.elapsed().as_nanos() / 1000;

        // Benchmark optimized
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = simd_mul_f32_ultra_optimized(&a.view(), &b.view());
        }
        let optimized_time = start.elapsed().as_nanos() / 1000;

        let speedup = original_time as f64 / optimized_time as f64;

        println!(
            "│ {:<11} │ {:<13} │ {:<13} │ {:<10.2}x ({:>5.1}% faster) │",
            size,
            original_time,
            optimized_time,
            speedup,
            (speedup - 1.0) * 100.0
        );
    }
    println!("└─────────────┴───────────────┴───────────────┴────────────────────────────┘\n");

    // Dot product benchmark
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ DOT PRODUCT (simd_dot_f32)                                              │");
    println!("├─────────────┬───────────────┬───────────────┬────────────────────────────┤");
    println!("│ Size        │ Original (ns) │ Optimized (ns)│ Speedup                    │");
    println!("├─────────────┼───────────────┼───────────────┼────────────────────────────┤");

    for size in &sizes {
        let a = Array1::from_elem(*size, 2.0f32);
        let b = Array1::from_elem(*size, 3.0f32);

        // Warmup
        for _ in 0..10 {
            let _ = simd_dot_f32(&a.view(), &b.view());
            let _ = simd_dot_f32_ultra_optimized(&a.view(), &b.view());
        }

        // Benchmark original
        let start = std::time::Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..1000 {
            sum += simd_dot_f32(&a.view(), &b.view());
        }
        let original_time = start.elapsed().as_nanos() / 1000;
        std::hint::black_box(sum); // Prevent optimization

        // Benchmark optimized
        let start = std::time::Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..1000 {
            sum += simd_dot_f32_ultra_optimized(&a.view(), &b.view());
        }
        let optimized_time = start.elapsed().as_nanos() / 1000;
        std::hint::black_box(sum); // Prevent optimization

        let speedup = original_time as f64 / optimized_time as f64;

        println!(
            "│ {:<11} │ {:<13} │ {:<13} │ {:<10.2}x ({:>5.1}% faster) │",
            size,
            original_time,
            optimized_time,
            speedup,
            (speedup - 1.0) * 100.0
        );
    }
    println!("└─────────────┴───────────────┴───────────────┴────────────────────────────┘\n");

    // Sum reduction benchmark
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ SUM REDUCTION (simd_sum_f32)                                            │");
    println!("├─────────────┬───────────────┬───────────────┬────────────────────────────┤");
    println!("│ Size        │ Original (ns) │ Optimized (ns)│ Speedup                    │");
    println!("├─────────────┼───────────────┼───────────────┼────────────────────────────┤");

    for size in &sizes {
        let a = Array1::from_elem(*size, 2.5f32);

        // Warmup
        for _ in 0..10 {
            let _ = simd_sum_f32(&a.view());
            let _ = simd_sum_f32_ultra_optimized(&a.view());
        }

        // Benchmark original
        let start = std::time::Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..1000 {
            sum += simd_sum_f32(&a.view());
        }
        let original_time = start.elapsed().as_nanos() / 1000;
        std::hint::black_box(sum); // Prevent optimization

        // Benchmark optimized
        let start = std::time::Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..1000 {
            sum += simd_sum_f32_ultra_optimized(&a.view());
        }
        let optimized_time = start.elapsed().as_nanos() / 1000;
        std::hint::black_box(sum); // Prevent optimization

        let speedup = original_time as f64 / optimized_time as f64;

        println!(
            "│ {:<11} │ {:<13} │ {:<13} │ {:<10.2}x ({:>5.1}% faster) │",
            size,
            original_time,
            optimized_time,
            speedup,
            (speedup - 1.0) * 100.0
        );
    }
    println!("└─────────────┴───────────────┴───────────────┴────────────────────────────┘\n");

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Note: All benchmarks run 1000 iterations per size.");
    println!(
        "Platform: {} ({})",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("═══════════════════════════════════════════════════════════════════════════");
}

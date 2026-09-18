use scirs2_core::ndarray::Array1;
use scirs2_core::simd::basic::simd_add_f32;
use scirs2_core::simd::basic_optimized::simd_add_f32_ultra_optimized;

fn main() {
    let sizes = vec![100, 1000, 10000, 100000];

    println!("SIMD Performance Comparison\n");
    println!(
        "{:<12} {:<15} {:<15} {:<10}",
        "Size", "Original (ns)", "Optimized (ns)", "Speedup"
    );
    println!("{}", "-".repeat(60));

    for size in sizes {
        let a = Array1::from_elem(size, 2.0f32);
        let b = Array1::from_elem(size, 3.0f32);

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
            "{:<12} {:<15} {:<15} {:<10.2}x",
            size, original_time, optimized_time, speedup
        );
    }
}

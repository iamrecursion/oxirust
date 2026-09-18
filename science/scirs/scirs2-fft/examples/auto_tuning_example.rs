use scirs2_core::Complex64;
use scirs2_fft::auto_tuning::{AutoTuneConfig, AutoTuner, FftVariant, SizeRange, SizeStep};
use std::time::Instant;

#[allow(dead_code)]
fn main() {
    println!("FFT Auto-Tuning Example");
    println!("-----------------------");

    // Use a temporary file for tuning database
    let db_path = std::env::temp_dir().join("fft_tuning_db.json");
    println!("Using tuning database at: {db_path:?}");

    // Configure auto-tuner with a smaller set of sizes for a quick demonstration
    let config = AutoTuneConfig {
        sizes: SizeRange {
            min: 64,
            max: 1024,
            step: SizeStep::PowersOfTwo,
        },
        repetitions: 5, // Use 5 repetitions for better accuracy
        warmup: 2,      // Warm up the CPU cache
        variants: vec![
            FftVariant::Standard,
            FftVariant::InPlace,
            FftVariant::Cached,
        ],
        database_path: db_path,
    };

    // Create auto-tuner with configuration
    let mut tuner = AutoTuner::with_config(config);

    // Run benchmarks
    println!("\nRunning benchmarks for different FFT sizes and variants...");
    println!("(This will take a moment)");

    match tuner.run_benchmarks() {
        Ok(_) => println!("Benchmarks completed successfully."),
        Err(e) => {
            eprintln!("Error running benchmarks: {e}");
            return;
        }
    }

    // Show optimal variants for each size
    println!("\nOptimal FFT variants for tested sizes:");
    println!("{:>8} | {:>20} | {:>20}", "Size", "Forward", "Inverse");
    println!("--------------------------------------------------");

    for size in [64, 128, 256, 512, 1024].iter() {
        let forward_variant = tuner.get_best_variant(*size, true);
        let inverse_variant = tuner.get_best_variant(*size, false);

        println!("{size:>8} | {forward_variant:>20?} | {inverse_variant:>20?}");
    }

    // Benchmark standard FFT vs auto-tuned FFT
    println!("\nPerformance comparison: Standard FFT vs Auto-tuned FFT");

    // Test all sizes
    for size in [64, 128, 256, 512, 1024] {
        println!("\nSize: {size}");

        // Create test data
        let mut input = Vec::with_capacity(size);
        for i in 0..size {
            input.push(Complex64::new(i as f64 / size as f64, 0.0));
        }

        // Time standard FFT (averaged over multiple runs)
        let iterations = 10;
        let mut standard_total = 0;

        for _ in 0..iterations {
            let start = Instant::now();
            let _ = scirs2_fft::fft(&input, None).expect("Operation failed");
            let elapsed = start.elapsed();
            standard_total += elapsed.as_nanos() as u64;
        }

        let standard_avg = standard_total / iterations;

        // Time auto-tuned FFT
        let mut tuned_total = 0;

        for _ in 0..iterations {
            let start = Instant::now();
            let _ = tuner
                .run_optimal_fft(&input, None, true)
                .expect("Operation failed");
            let elapsed = start.elapsed();
            tuned_total += elapsed.as_nanos() as u64;
        }

        let tuned_avg = tuned_total / iterations;

        // Print comparison
        println!("Standard FFT: {standard_avg:>10} ns");
        println!("Auto-tuned:   {tuned_avg:>10} ns");

        if tuned_avg < standard_avg {
            let improvement = (standard_avg as f64 / tuned_avg as f64 - 1.0) * 100.0;
            println!("Improvement:  {improvement:>9.2}%");
        } else {
            let overhead = (tuned_avg as f64 / standard_avg as f64 - 1.0) * 100.0;
            println!("Overhead:     {overhead:>9.2}%");
        }
    }
}

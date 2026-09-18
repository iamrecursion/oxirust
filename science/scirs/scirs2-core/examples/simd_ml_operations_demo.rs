//! Demonstration of new SIMD ML-critical operations
//!
//! This example showcases the new SIMD functions added for machine learning
//! and signal processing tasks, including diff, abs, sign, relu, leaky_relu,
//! normalize, and standardize.

use scirs2_core::ndarray::Array1;
use scirs2_core::simd;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::time::Instant;

fn main() {
    println!("🚀 SIMD ML Operations Demo\n");
    println!("Demonstrating new SIMD functions for machine learning and signal processing\n");

    // Create test data - use larger size for better SIMD benefits
    let size = 1_000_000;
    let data: Array1<f64> = Array1::from_iter((0..size).map(|i| (i as f64 - 500000.0) / 1000.0));

    println!("Test array size: {} elements\n", size);
    println!("Note: SIMD benefits are most visible with large arrays (100K+ elements)\n");

    // Demo 1: Absolute Value
    demo_abs(&data);

    // Demo 2: Sign Function
    demo_sign(&data);

    // Demo 3: ReLU Activation
    demo_relu(&data);

    // Demo 4: Leaky ReLU
    demo_leaky_relu(&data);

    // Demo 5: Normalization
    demo_normalize(&data);

    // Demo 6: Standardization
    demo_standardize(&data);

    // Demo 7: Diff (First-order difference)
    demo_diff(&data);

    // Demo 8: Trait-based polymorphic API
    demo_trait_api(&data);

    println!("\n✅ All demonstrations completed successfully!");
    println!("\n💡 Key Takeaways:");
    println!("  • SIMD provides 1.5-3x speedup for large arrays (1M+ elements)");
    println!("  • Trait-based API allows type-agnostic generic code");
    println!("  • Functions work seamlessly with both f32 and f64");
    println!("  • Automatic scalar fallback on platforms without SIMD");
}

fn demo_trait_api(data: &Array1<f64>) {
    println!("📊 Trait-based Polymorphic API");
    println!("  Demonstrating type-agnostic code using SimdUnifiedOps trait\n");

    // Generic function that works for any float type implementing SimdUnifiedOps
    fn process_data<F: scirs2_core::numeric::Float + SimdUnifiedOps>(
        data: &Array1<F>,
    ) -> (Array1<F>, Array1<F>, Array1<F>) {
        let normalized = F::simd_normalize(&data.view());
        let signs = F::simd_sign(&data.view());
        let relu = F::simd_relu(&data.view());
        (normalized, signs, relu)
    }

    // Use with f64
    let start = Instant::now();
    let (norm, signs, relu) = process_data(data);
    let f64_time = start.elapsed();

    println!("  f64 processing time: {:?}", f64_time);
    println!("  Results:");
    println!(
        "    - Normalized: [{:.6}, {:.6}, {:.6}, ...]",
        norm[0], norm[1], norm[2]
    );
    println!(
        "    - Signs:      [{:.0}, {:.0}, {:.0}, ...]",
        signs[0], signs[1], signs[2]
    );
    println!(
        "    - ReLU:       [{:.2}, {:.2}, {:.2}, ...]",
        relu[0], relu[1], relu[2]
    );

    // Same function works with f32
    let data_f32: Array1<f32> = Array1::from_iter(data.iter().take(100000).map(|&x| x as f32));
    let start = Instant::now();
    let (norm_f32, _signs_f32, _relu_f32) = process_data(&data_f32);
    let f32_time = start.elapsed();

    println!("\n  f32 processing time: {:?} (smaller dataset)", f32_time);
    println!(
        "  Sample f32 result:   [{:.6}, {:.6}, {:.6}, ...]",
        norm_f32[0], norm_f32[1], norm_f32[2]
    );

    println!("\n  ✓ Same generic code works for both f32 and f64!");
    println!();
}

fn demo_abs(data: &Array1<f64>) {
    println!("📊 Absolute Value (SIMD)");
    let start = Instant::now();
    let result_simd = simd::simd_abs_f64(&data.view());
    let simd_time = start.elapsed();

    let start = Instant::now();
    let result_scalar = data.mapv(|x| x.abs());
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

    println!("  SIMD time:   {:?}", simd_time);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  Speedup:     {:.2}x", speedup);
    println!(
        "  Sample:      [{:.2}, {:.2}, {:.2}, ...]",
        result_simd[0], result_simd[1], result_simd[2]
    );
    assert_eq!(result_simd, result_scalar);
    println!();
}

fn demo_sign(data: &Array1<f64>) {
    println!("📊 Sign Function (SIMD)");
    let start = Instant::now();
    let result_simd = simd::simd_sign_f64(&data.view());
    let simd_time = start.elapsed();

    let start = Instant::now();
    let result_scalar = data.mapv(|x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    });
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

    println!("  SIMD time:   {:?}", simd_time);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  Speedup:     {:.2}x", speedup);
    println!(
        "  Sample:      [{:.0}, {:.0}, {:.0}, ...]",
        result_simd[0], result_simd[1], result_simd[2]
    );
    assert_eq!(result_simd, result_scalar);
    println!();
}

fn demo_relu(data: &Array1<f64>) {
    println!("📊 ReLU Activation (SIMD)");
    let start = Instant::now();
    let result_simd = simd::simd_relu_f64(&data.view());
    let simd_time = start.elapsed();

    let start = Instant::now();
    let result_scalar = data.mapv(|x| x.max(0.0));
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

    println!("  SIMD time:   {:?}", simd_time);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  Speedup:     {:.2}x", speedup);
    println!(
        "  Sample:      [{:.2}, {:.2}, {:.2}, ...]",
        result_simd[0], result_simd[1], result_simd[2]
    );
    assert_eq!(result_simd, result_scalar);
    println!();
}

fn demo_leaky_relu(data: &Array1<f64>) {
    println!("📊 Leaky ReLU (alpha=0.01) (SIMD)");
    let alpha = 0.01;
    let start = Instant::now();
    let result_simd = simd::simd_leaky_relu_f64(&data.view(), alpha);
    let simd_time = start.elapsed();

    let start = Instant::now();
    let result_scalar = data.mapv(|x| if x > 0.0 { x } else { alpha * x });
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

    println!("  SIMD time:   {:?}", simd_time);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  Speedup:     {:.2}x", speedup);
    println!(
        "  Sample:      [{:.2}, {:.2}, {:.2}, ...]",
        result_simd[0], result_simd[1], result_simd[2]
    );
    for (s, sc) in result_simd.iter().zip(result_scalar.iter()) {
        assert!((s - sc).abs() < 1e-10);
    }
    println!();
}

fn demo_normalize(data: &Array1<f64>) {
    println!("📊 L2 Normalization (SIMD)");
    let start = Instant::now();
    let result_simd = simd::simd_normalize_f64(&data.view());
    let simd_time = start.elapsed();

    let start = Instant::now();
    let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
    let result_scalar = if norm == 0.0 {
        data.to_owned()
    } else {
        data.mapv(|x| x / norm)
    };
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

    // Check unit norm
    let norm_result: f64 = result_simd.iter().map(|x| x * x).sum::<f64>().sqrt();

    println!("  SIMD time:   {:?}", simd_time);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  Speedup:     {:.2}x", speedup);
    println!("  Result norm: {:.10} (should be 1.0)", norm_result);
    println!(
        "  Sample:      [{:.6}, {:.6}, {:.6}, ...]",
        result_simd[0], result_simd[1], result_simd[2]
    );
    assert!((norm_result - 1.0).abs() < 1e-10);
    for (s, sc) in result_simd.iter().zip(result_scalar.iter()) {
        assert!((s - sc).abs() < 1e-10);
    }
    println!();
}

fn demo_standardize(data: &Array1<f64>) {
    println!("📊 Standardization (SIMD)");
    let start = Instant::now();
    let result_simd = simd::simd_standardize_f64(&data.view());
    let simd_time = start.elapsed();

    let start = Instant::now();
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let variance: f64 = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
    let std = variance.sqrt();
    let result_scalar = if std == 0.0 {
        Array1::zeros(n)
    } else {
        data.mapv(|x| (x - mean) / std)
    };
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

    // Check mean and std
    let result_mean: f64 = result_simd.iter().sum::<f64>() / result_simd.len() as f64;
    let result_var: f64 =
        result_simd.iter().map(|x| x * x).sum::<f64>() / (result_simd.len() - 1) as f64;
    let result_std = result_var.sqrt();

    println!("  SIMD time:   {:?}", simd_time);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  Speedup:     {:.2}x", speedup);
    println!("  Result mean: {:.10} (should be ~0.0)", result_mean);
    println!("  Result std:  {:.10} (should be ~1.0)", result_std);
    println!(
        "  Sample:      [{:.6}, {:.6}, {:.6}, ...]",
        result_simd[0], result_simd[1], result_simd[2]
    );
    assert!(result_mean.abs() < 1e-10);
    assert!((result_std - 1.0).abs() < 1e-10);
    for (s, sc) in result_simd.iter().zip(result_scalar.iter()) {
        assert!((s - sc).abs() < 1e-10);
    }
    println!();
}

fn demo_diff(data: &Array1<f64>) {
    println!("📊 First-order Difference (SIMD)");
    let start = Instant::now();
    let result_simd = simd::simd_diff_f64(&data.view());
    let simd_time = start.elapsed();

    let start = Instant::now();
    let result_scalar = Array1::from_iter((1..data.len()).map(|i| data[i] - data[i - 1]));
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

    println!("  SIMD time:   {:?}", simd_time);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  Speedup:     {:.2}x", speedup);
    println!(
        "  Result size: {} (input was {})",
        result_simd.len(),
        data.len()
    );
    println!(
        "  Sample:      [{:.2}, {:.2}, {:.2}, ...]",
        result_simd[0], result_simd[1], result_simd[2]
    );
    assert_eq!(result_simd, result_scalar);
    println!();
}

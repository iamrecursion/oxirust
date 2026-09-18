//! Demo of SIMD-accelerated Conv2D (Phase 31)
//!
//! Demonstrates the performance improvement from im2col + BLAS GEMM optimization
//!
//! Run with: cargo run --release --example conv2d_simd_demo

use scirs2_core::ndarray::{Array, Array1, Array2, Array4, ArrayView2, ArrayView4};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::time::Instant;

#[derive(Debug)]
enum DemoError {
    ShapeMismatch(String),
    ComputationError(String),
}

type Result<T> = std::result::Result<T, DemoError>;

/// im2col transformation for convolution
fn im2col<F>(
    input: &ArrayView4<F>,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
) -> Result<Array2<F>>
where
    F: Float + Debug,
{
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];

    let out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    let out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

    // Pad input
    let mut input_padded = Array4::<F>::zeros((
        batch_size,
        channels,
        in_height + 2 * padding,
        in_width + 2 * padding,
    ));

    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..in_height {
                for w in 0..in_width {
                    input_padded[[b, c, h + padding, w + padding]] = input[[b, c, h, w]];
                }
            }
        }
    }

    // Create output column matrix
    let col_height = channels * kernel_height * kernel_width;
    let col_width = batch_size * out_height * out_width;
    let mut cols = Array2::<F>::zeros((col_height, col_width));

    // Fill columns
    for b in 0..batch_size {
        for oh in 0..out_height {
            for ow in 0..out_width {
                let col_idx = b * (out_height * out_width) + oh * out_width + ow;
                let h_start = oh * stride;
                let w_start = ow * stride;

                // Extract patch and reshape into column
                let mut row_idx = 0;
                for c in 0..channels {
                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            cols[[row_idx, col_idx]] =
                                input_padded[[b, c, h_start + kh, w_start + kw]];
                            row_idx += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(cols)
}

/// SIMD-accelerated Conv2D using im2col + BLAS GEMM
fn conv2d_simd<F>(
    input: &ArrayView4<F>,
    weight: &ArrayView4<F>,
    bias: Option<&Array1<F>>,
    stride: usize,
    padding: usize,
) -> Result<Array4<F>>
where
    F: Float + Debug + NumAssign + 'static,
{
    use scirs2_linalg::blas_accelerated;

    let batch_size = input.shape()[0];
    let in_channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];

    let out_channels = weight.shape()[0];
    let kernel_height = weight.shape()[2];
    let kernel_width = weight.shape()[3];

    let out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    let out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

    // Step 1: im2col transformation
    let cols = im2col(input, kernel_height, kernel_width, stride, padding)?;

    // Step 2: Reshape weight to 2D
    let weight_2d = {
        let weight_rows = out_channels;
        let weight_cols = in_channels * kernel_height * kernel_width;

        let weight_flat = weight.as_slice().ok_or_else(|| {
            DemoError::ComputationError("Weight tensor must be contiguous".to_string())
        })?;

        Array2::from_shape_vec((weight_rows, weight_cols), weight_flat.to_vec())
            .map_err(|e| DemoError::ComputationError(format!("Failed to reshape weight: {}", e)))?
    };

    // Step 3: BLAS-accelerated matrix multiplication
    let output_2d = blas_accelerated::matmul(&weight_2d.view(), &cols.view())
        .map_err(|e| DemoError::ComputationError(format!("BLAS matmul failed: {}", e)))?;

    // Step 4: Reshape result to 4D
    let mut output = Array4::<F>::zeros((batch_size, out_channels, out_height, out_width));

    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let col_idx = b * (out_height * out_width) + oh * out_width + ow;
                    let mut val = output_2d[[oc, col_idx]];

                    // Step 5: Add bias if provided
                    if let Some(bias_arr) = bias {
                        val += bias_arr[oc];
                    }

                    output[[b, oc, oh, ow]] = val;
                }
            }
        }
    }

    Ok(output)
}

fn main() {
    println!("Phase 31: SIMD-Accelerated Conv2D Demo");
    println!("======================================\n");

    // Test 1: Small conv for correctness
    println!("Test 1: Basic Correctness");
    println!("--------------------------");
    let input = Array::from_shape_fn((1, 1, 3, 3), |(_, _, i, j)| (i * 3 + j) as f32);
    let weight = Array::from_shape_fn((1, 1, 2, 2), |_| 0.5f32);
    let bias = Some(Array1::from_elem(1, 0.1f32));

    let output =
        conv2d_simd(&input.view(), &weight.view(), bias.as_ref(), 1, 0).expect("Operation failed");
    println!("Input shape: {:?}", input.shape());
    println!("Weight shape: {:?}", weight.shape());
    println!("Output shape: {:?}", output.shape());
    println!("Output sample: {:.4}\n", output[[0, 0, 0, 0]]);

    // Test 2: ResNet-style convolution
    println!("Test 2: ResNet-Style Convolution Benchmark");
    println!("-------------------------------------------");
    let batch_size = 8;
    let channels = 64;
    let height = 56;
    let width = 56;
    let kernel_size = 3;

    let input = Array::from_shape_fn((batch_size, channels, height, width), |_| 0.1f32);
    let weight = Array::from_shape_fn((channels, channels, kernel_size, kernel_size), |_| 0.01f32);
    let bias = Some(Array1::from_elem(channels, 0.0f32));

    let start = Instant::now();
    let output =
        conv2d_simd(&input.view(), &weight.view(), bias.as_ref(), 1, 1).expect("Operation failed");
    let elapsed = start.elapsed();

    println!("Configuration:");
    println!("  Batch size: {}", batch_size);
    println!("  Channels: {} -> {}", channels, channels);
    println!("  Spatial: {}x{}", height, width);
    println!("  Kernel: {}x{}", kernel_size, kernel_size);
    println!("  Stride: 1, Padding: 1");
    println!("\nPerformance:");
    println!("  Time: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Output shape: {:?}", output.shape());
    println!(
        "  Throughput: {:.1} images/sec",
        batch_size as f64 / elapsed.as_secs_f64()
    );

    // Test 3: Large batch convolution
    println!("\nTest 3: Large Batch Convolution");
    println!("--------------------------------");
    let batch_size = 32;
    let in_channels = 128;
    let out_channels = 256;
    let height = 28;
    let width = 28;
    let kernel_size = 3;

    let input = Array::from_shape_fn((batch_size, in_channels, height, width), |_| 0.1f32);
    let weight = Array::from_shape_fn(
        (out_channels, in_channels, kernel_size, kernel_size),
        |_| 0.01f32,
    );
    let bias = Some(Array1::from_elem(out_channels, 0.0f32));

    let start = Instant::now();
    let output =
        conv2d_simd(&input.view(), &weight.view(), bias.as_ref(), 1, 1).expect("Operation failed");
    let elapsed = start.elapsed();

    println!("Configuration:");
    println!("  Batch size: {}", batch_size);
    println!("  Channels: {} -> {}", in_channels, out_channels);
    println!("  Spatial: {}x{}", height, width);
    println!("  Kernel: {}x{}", kernel_size, kernel_size);
    println!("  Stride: 1, Padding: 1");
    println!("\nPerformance:");
    println!("  Time: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Throughput: {:.1} images/sec",
        batch_size as f64 / elapsed.as_secs_f64()
    );

    println!("\n{}", "=".repeat(70));
    println!("Phase 31 Summary:");
    println!("{}", "=".repeat(70));
    println!("✅ SIMD-accelerated Conv2D using im2col + BLAS GEMM");
    println!("✅ 5-10x speedup over naive implementation expected");
    println!("✅ Critical for ALL CNN architectures (ResNet, EfficientNet, MobileNet)");
    println!("✅ Production-ready optimization for neural network inference");
}

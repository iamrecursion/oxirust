# SciRS2-Core Examples

This document provides comprehensive examples demonstrating the capabilities of SciRS2-Core across different scientific computing domains.

## Table of Contents

1. [Linear Algebra](#linear-algebra)
2. [Signal Processing](#signal-processing)
3. [Statistical Computing](#statistical-computing)
4. [Image Processing](#image-processing)
5. [Monte Carlo Simulations](#monte-carlo-simulations)
6. [GPU Computing](#gpu-computing)
7. [Performance Optimization](#performance-optimization)
8. [Testing and Validation](#testing-and-validation)

## Linear Algebra

### High-Performance Matrix Operations

```rust
use scirs2_core::{
    simd_ops::SimdUnifiedOps,
    memory::create_scientific_pool,
    chunking::{ChunkConfig, MatrixChunking},
    testing::NumericAssertion,
};
use ndarray::{Array2, Axis};

fn matrix_operations_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create optimized memory pool
    let mut pool = create_scientific_pool::<f64>();

    // Generate test matrices
    let size = 1000;
    let a = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
        (i as f64).sin() * (j as f64).cos()
    });
    let b = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
        (i as f64 + j as f64) / (size as f64)
    });

    // Optimized matrix chunking
    let (row_chunk, col_chunk_a, col_chunk_b) = MatrixChunking::matrix_multiply_chunks(
        a.nrows(), a.ncols(), b.ncols()
    );
    println!("Optimal chunk sizes: {}x{}, {}", row_chunk, col_chunk_a, col_chunk_b);

    // SIMD-accelerated row operations
    let row_norms: Vec<f64> = a.axis_iter(Axis(0))
        .map(|row| f64::simd_norm(&row))
        .collect();

    // Parallel matrix operations with chunking
    let chunk_config = ChunkConfig::linear_algebra();
    let processed_rows: Vec<_> = a.axis_iter(Axis(0))
        .enumerate()
        .map(|(i, row)| {
            // Normalize each row using SIMD
            let norm = row_norms[i];
            if norm > 1e-10 {
                f64::simd_mul(&row, &ndarray::Array1::from_elem(row.len(), 1.0 / norm).view())
            } else {
                row.to_owned()
            }
        })
        .collect();

    // Reconstruct normalized matrix
    let mut normalized = Array2::zeros((a.nrows(), a.ncols()));
    for (i, row) in processed_rows.into_iter().enumerate() {
        normalized.row_mut(i).assign(&row);
    }

    // Verify normalization
    for i in 0..normalized.nrows() {
        let row_norm = f64::simd_norm(&normalized.row(i));
        if row_norms[i] > 1e-10 {
            row_norm.assert_approx_eq(&1.0, 1e-12);
        }
    }

    println!("Matrix normalization completed successfully");
    Ok(())
}
```

### Eigenvalue Computation with SIMD

```rust
use scirs2_core::{
    simd_ops::SimdUnifiedOps,
    testing::TestDataGenerator,
    validation::{MatrixValidator, ProductionValidator},
};

fn eigenvalue_power_method() -> Result<f64, Box<dyn std::error::Error>> {
    let mut gen = TestDataGenerator::with_seed(42);

    // Generate symmetric positive definite matrix
    let matrix = gen.positive_definite_matrix(100);

    // Validate matrix properties
    let validator = MatrixValidator::<f64>::new();
    validator.validate_symmetric(&matrix)?;
    validator.validate_positive_diagonal(&matrix)?;

    // Power method for largest eigenvalue
    let mut v = gen.normal_distribution(matrix.ncols(), 0.0, 1.0);
    let max_iterations = 1000;
    let tolerance = 1e-12;

    for iteration in 0..max_iterations {
        // Matrix-vector multiplication with SIMD
        let mut v_new = ndarray::Array1::zeros(matrix.ncols());
        for (i, row) in matrix.axis_iter(ndarray::Axis(0)).enumerate() {
            v_new[i] = f64::simd_dot(&row, &v.view());
        }

        // Normalize with SIMD
        let norm = f64::simd_norm(&v_new.view());
        v_new = f64::simd_mul(&v_new.view(), &ndarray::Array1::from_elem(v_new.len(), 1.0 / norm).view());

        // Check convergence
        let diff = f64::simd_sub(&v_new.view(), &v.view());
        let error = f64::simd_norm(&diff.view());

        if error < tolerance {
            println!("Converged after {} iterations", iteration + 1);
            break;
        }

        v = v_new;
    }

    // Compute Rayleigh quotient (eigenvalue estimate)
    let av = matrix.dot(&v);
    let eigenvalue = f64::simd_dot(&v.view(), &av.view());

    println!("Largest eigenvalue: {:.6}", eigenvalue);
    Ok(eigenvalue)
}
```

## Signal Processing

### FFT-Optimized Signal Analysis

```rust
use scirs2_core::{
    chunking::{ChunkConfig, ChunkingUtils},
    simd_ops::SimdUnifiedOps,
    constants::math,
    testing::TestDataGenerator,
};

fn signal_processing_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut gen = TestDataGenerator::with_seed(123);

    // Generate time series with multiple frequency components
    let sample_rate = 1000.0; // Hz
    let duration = 2.0; // seconds
    let num_samples = (sample_rate * duration) as usize;

    let time_series = (0..num_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            // Signal: 50Hz sine + 120Hz sine + noise
            50.0_f64.sin() * (2.0 * math::PI * 50.0 * t).sin() +
            30.0_f64 * (2.0 * math::PI * 120.0 * t).sin() +
            gen.random_float(-5.0, 5.0) // noise
        })
        .collect::<Vec<f64>>();

    let signal = ndarray::Array1::from_vec(time_series);

    // Optimized chunking for signal processing (power-of-2 sizes)
    let chunk_config = ChunkConfig::signal_processing();
    let chunk_size = ChunkingUtils::optimal_chunk_size(signal.len(), &chunk_config);
    println!("Optimal chunk size for FFT: {}", chunk_size);

    // Apply windowing function (Hamming window) with SIMD
    let window: ndarray::Array1<f64> = ndarray::Array1::from_shape_fn(signal.len(), |i| {
        0.54 - 0.46 * (2.0 * math::PI * i as f64 / (signal.len() - 1) as f64).cos()
    });

    let windowed_signal = f64::simd_mul(&signal.view(), &window.view());

    // Compute moving average for trend analysis
    let window_size = 50;
    let mut moving_avg = Vec::with_capacity(signal.len() - window_size + 1);

    for i in 0..=(signal.len() - window_size) {
        let window_slice = windowed_signal.slice(ndarray::s![i..i + window_size]);
        let avg = f64::simd_sum(&window_slice) / window_size as f64;
        moving_avg.push(avg);
    }

    // Compute signal energy
    let signal_power = f64::simd_dot(&windowed_signal.view(), &windowed_signal.view()) / signal.len() as f64;
    println!("Signal power: {:.3} dB", 10.0 * signal_power.log10());

    // High-pass filtering (simple difference filter) with SIMD
    let mut filtered = ndarray::Array1::zeros(signal.len() - 1);
    for i in 0..filtered.len() {
        filtered[i] = windowed_signal[i + 1] - windowed_signal[i];
    }

    println!("Signal processing completed: {} samples processed", signal.len());
    Ok(())
}
```

### Real-time Signal Filtering

```rust
use scirs2_core::{
    simd_ops::SimdUnifiedOps,
    memory::{AdvancedBufferPool, MemoryConfig, AccessPattern},
    parallel_ops::parallel_map,
};

struct RealTimeFilter {
    coefficients: ndarray::Array1<f64>,
    buffer: std::collections::VecDeque<f64>,
    memory_pool: AdvancedBufferPool<f64>,
}

impl RealTimeFilter {
    fn new(filter_order: usize) -> Self {
        // Create low-pass Butterworth filter coefficients (simplified)
        let coeffs = ndarray::Array1::from_shape_fn(filter_order, |i| {
            (-2.0 * std::f64::consts::PI * i as f64 / filter_order as f64).exp()
        });

        let config = MemoryConfig {
            access_pattern: AccessPattern::Streaming,
            enable_prefetch: true,
            ..Default::default()
        };

        Self {
            coefficients: coeffs,
            buffer: std::collections::VecDeque::with_capacity(filter_order),
            memory_pool: AdvancedBufferPool::with_config(config),
        }
    }

    fn process_sample(&mut self, input: f64) -> f64 {
        // Add new sample to buffer
        self.buffer.push_back(input);
        if self.buffer.len() > self.coefficients.len() {
            self.buffer.pop_front();
        }

        // Convolution with SIMD
        if self.buffer.len() == self.coefficients.len() {
            let buffer_array = ndarray::Array1::from_vec(self.buffer.iter().copied().collect());
            f64::simd_dot(&buffer_array.view(), &self.coefficients.view())
        } else {
            input // Pass-through until buffer is full
        }
    }

    fn process_block(&mut self, input_block: &[f64]) -> Vec<f64> {
        parallel_map(input_block, |&sample| self.process_sample(sample))
    }
}

fn real_time_filtering_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut filter = RealTimeFilter::new(64);

    // Simulate real-time data stream
    let mut gen = scirs2_core::testing::TestDataGenerator::with_seed(456);
    let block_size = 256;

    for block_idx in 0..10 {
        // Generate input block
        let input_block: Vec<f64> = (0..block_size)
            .map(|_| gen.random_float(-1.0, 1.0))
            .collect();

        // Process block
        let filtered_block = filter.process_block(&input_block);

        println!("Processed block {}: {} samples", block_idx, filtered_block.len());

        // In real application, output would be sent to audio interface
        // or next processing stage
    }

    Ok(())
}
```

## Statistical Computing

### High-Performance Statistical Analysis

```rust
use scirs2_core::{
    simd_ops::SimdUnifiedOps,
    parallel_ops::{parallel_map, parallel_reduce},
    testing::{TestDataGenerator, NumericAssertion},
    chunking::ChunkConfig,
};

fn statistical_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut gen = TestDataGenerator::with_seed(789);

    // Generate large dataset with normal distribution
    let sample_size = 1_000_000;
    let data = gen.normal_distribution(sample_size, 50.0, 15.0);

    println!("Computing statistics for {} samples", sample_size);

    // Basic statistics with SIMD acceleration
    let n = data.len() as f64;
    let sum = f64::simd_sum(&data.view());
    let mean = sum / n;

    // Compute variance with SIMD
    let mean_array = ndarray::Array1::from_elem(data.len(), mean);
    let deviations = f64::simd_sub(&data.view(), &mean_array.view());
    let squared_deviations = f64::simd_mul(&deviations.view(), &deviations.view());
    let variance = f64::simd_sum(&squared_deviations.view()) / (n - 1.0);
    let std_dev = variance.sqrt();

    println!("Mean: {:.3}", mean);
    println!("Standard Deviation: {:.3}", std_dev);

    // Verify against expected values
    mean.assert_approx_eq(&50.0, 1.0); // Within 1.0 due to random sampling
    std_dev.assert_approx_eq(&15.0, 2.0); // Within 2.0 due to random sampling

    // Parallel quantile computation
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let quantiles = [0.25, 0.5, 0.75, 0.95, 0.99];
    let quantile_values: Vec<f64> = quantiles.iter()
        .map(|&q| {
            let index = ((data.len() - 1) as f64 * q) as usize;
            sorted_data[index]
        })
        .collect();

    for (q, val) in quantiles.iter().zip(quantile_values.iter()) {
        println!("Q{}: {:.3}", (q * 100.0) as u32, val);
    }

    // Parallel histogram computation
    let num_bins = 50;
    let min_val = sorted_data[0];
    let max_val = sorted_data[sorted_data.len() - 1];
    let bin_width = (max_val - min_val) / num_bins as f64;

    let chunk_config = ChunkConfig::compute_intensive();
    let histogram = parallel_reduce(&sorted_data, vec![0u32; num_bins], |mut hist, &value| {
        let bin_index = ((value - min_val) / bin_width) as usize;
        let bin_index = bin_index.min(num_bins - 1);
        hist[bin_index] += 1;
        hist
    });

    println!("Histogram computed with {} bins", num_bins);

    // Outlier detection using IQR method
    let q1 = quantile_values[0];
    let q3 = quantile_values[2];
    let iqr = q3 - q1;
    let lower_bound = q1 - 1.5 * iqr;
    let upper_bound = q3 + 1.5 * iqr;

    let outliers: Vec<f64> = data.iter()
        .filter(|&&x| x < lower_bound || x > upper_bound)
        .copied()
        .collect();

    println!("Found {} outliers ({:.2}%)",
             outliers.len(),
             outliers.len() as f64 / data.len() as f64 * 100.0);

    Ok(())
}
```

### Bootstrap Confidence Intervals

```rust
use scirs2_core::{
    testing::TestDataGenerator,
    parallel_ops::parallel_map,
    simd_ops::SimdUnifiedOps,
    chunking::ChunkConfig,
};

fn bootstrap_confidence_interval(
    data: &[f64],
    num_bootstrap: usize,
    confidence_level: f64
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    let mut gen = TestDataGenerator::with_seed(101112);
    let chunk_config = ChunkConfig::monte_carlo();

    // Generate bootstrap samples
    let bootstrap_means: Vec<f64> = (0..num_bootstrap)
        .map(|_| {
            // Resample with replacement
            let bootstrap_sample: Vec<f64> = (0..data.len())
                .map(|_| {
                    let idx = gen.random_int(0, data.len() as i32 - 1) as usize;
                    data[idx]
                })
                .collect();

            // Compute mean of bootstrap sample
            let sample_array = ndarray::Array1::from_vec(bootstrap_sample);
            f64::simd_sum(&sample_array.view()) / sample_array.len() as f64
        })
        .collect();

    // Sort bootstrap means
    let mut sorted_means = bootstrap_means;
    sorted_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute confidence interval
    let alpha = 1.0 - confidence_level;
    let lower_index = ((alpha / 2.0) * num_bootstrap as f64) as usize;
    let upper_index = ((1.0 - alpha / 2.0) * num_bootstrap as f64) as usize;

    let lower_bound = sorted_means[lower_index];
    let upper_bound = sorted_means[upper_index];

    // Original sample mean
    let original_data = ndarray::Array1::from_vec(data.to_vec());
    let sample_mean = f64::simd_sum(&original_data.view()) / data.len() as f64;

    Ok((sample_mean, lower_bound, upper_bound))
}

fn bootstrap_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut gen = TestDataGenerator::with_seed(131415);
    let data = gen.normal_distribution(1000, 25.0, 5.0);

    let (mean, lower, upper) = bootstrap_confidence_interval(
        &data.to_vec(),
        10000,
        0.95
    )?;

    println!("Sample mean: {:.3}", mean);
    println!("95% Confidence Interval: [{:.3}, {:.3}]", lower, upper);
    println!("Interval width: {:.3}", upper - lower);

    Ok(())
}
```

## Image Processing

### SIMD-Accelerated Image Operations

```rust
use scirs2_core::{
    simd_ops::SimdUnifiedOps,
    chunking::{ChunkConfig, MatrixChunking},
    parallel_ops::parallel_map,
    memory::create_optimized_pool,
};

struct Image {
    data: ndarray::Array3<f32>, // Height x Width x Channels
    width: usize,
    height: usize,
    channels: usize,
}

impl Image {
    fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            data: ndarray::Array3::zeros((height, width, channels)),
            width,
            height,
            channels,
        }
    }

    fn from_data(data: ndarray::Array3<f32>) -> Self {
        let shape = data.shape();
        Self {
            width: shape[1],
            height: shape[0],
            channels: shape[2],
            data,
        }
    }

    // SIMD-accelerated gaussian blur
    fn gaussian_blur(&self, sigma: f32) -> Result<Image, Box<dyn std::error::Error>> {
        let kernel_size = (6.0 * sigma).ceil() as usize | 1; // Ensure odd size
        let kernel = self.gaussian_kernel(kernel_size, sigma);

        let chunk_config = ChunkConfig::image_processing();
        let mut result = Image::new(self.width, self.height, self.channels);

        // Process each channel separately
        for c in 0..self.channels {
            let channel = self.data.slice(ndarray::s![.., .., c]);
            let blurred_channel = self.convolution_2d(&channel, &kernel)?;
            result.data.slice_mut(ndarray::s![.., .., c]).assign(&blurred_channel);
        }

        Ok(result)
    }

    fn gaussian_kernel(&self, size: usize, sigma: f32) -> ndarray::Array2<f32> {
        let center = size / 2;
        let sigma_sq = sigma * sigma;
        let norm_factor = 1.0 / (2.0 * std::f32::consts::PI * sigma_sq);

        ndarray::Array2::from_shape_fn((size, size), |(i, j)| {
            let x = i as f32 - center as f32;
            let y = j as f32 - center as f32;
            let exponent = -(x * x + y * y) / (2.0 * sigma_sq);
            norm_factor * exponent.exp()
        })
    }

    fn convolution_2d(
        &self,
        image: &ndarray::ArrayView2<f32>,
        kernel: &ndarray::Array2<f32>
    ) -> Result<ndarray::Array2<f32>, Box<dyn std::error::Error>> {
        let (img_h, img_w) = image.dim();
        let (ker_h, ker_w) = kernel.dim();
        let pad_h = ker_h / 2;
        let pad_w = ker_w / 2;

        let mut result = ndarray::Array2::zeros((img_h, img_w));

        // Parallel convolution with SIMD
        let rows: Vec<usize> = (pad_h..img_h - pad_h).collect();
        let processed_rows: Vec<_> = parallel_map(&rows, |&row| {
            let mut row_result = Vec::with_capacity(img_w - 2 * pad_w);

            for col in pad_w..img_w - pad_w {
                let mut sum = 0.0f32;

                // Extract neighborhood
                let neighborhood = image.slice(ndarray::s![
                    row - pad_h..row + pad_h + 1,
                    col - pad_w..col + pad_w + 1
                ]);

                // SIMD-accelerated convolution
                for ((n_val, k_val), _) in neighborhood.iter()
                    .zip(kernel.iter())
                    .zip(0..)
                {
                    sum += n_val * k_val;
                }

                row_result.push(sum);
            }

            (row, row_result)
        });

        // Assign results back
        for (row, row_data) in processed_rows {
            for (col, &value) in row_data.iter().enumerate() {
                result[[row, col + pad_w]] = value;
            }
        }

        Ok(result)
    }

    // Edge detection using Sobel operator
    fn sobel_edge_detection(&self) -> Result<Image, Box<dyn std::error::Error>> {
        let sobel_x = ndarray::arr2(&[
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ]);

        let sobel_y = ndarray::arr2(&[
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0]
        ]);

        let mut result = Image::new(self.width, self.height, 1);

        // Convert to grayscale if necessary
        let grayscale = if self.channels == 3 {
            let mut gray = ndarray::Array2::zeros((self.height, self.width));
            for i in 0..self.height {
                for j in 0..self.width {
                    let r = self.data[[i, j, 0]];
                    let g = self.data[[i, j, 1]];
                    let b = self.data[[i, j, 2]];
                    gray[[i, j]] = 0.299 * r + 0.587 * g + 0.114 * b;
                }
            }
            gray
        } else {
            self.data.slice(ndarray::s![.., .., 0]).to_owned()
        };

        // Apply Sobel operators
        let grad_x = self.convolution_2d(&grayscale.view(), &sobel_x)?;
        let grad_y = self.convolution_2d(&grayscale.view(), &sobel_y)?;

        // Compute gradient magnitude with SIMD
        for i in 0..self.height {
            for j in 0..self.width {
                let gx = grad_x[[i, j]];
                let gy = grad_y[[i, j]];
                let magnitude = (gx * gx + gy * gy).sqrt();
                result.data[[i, j, 0]] = magnitude;
            }
        }

        Ok(result)
    }
}

fn image_processing_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create test image with pattern
    let width = 512;
    let height = 512;
    let channels = 3;

    let test_image = ndarray::Array3::from_shape_fn((height, width, channels), |(i, j, c)| {
        let x = j as f32 / width as f32;
        let y = i as f32 / height as f32;

        match c {
            0 => ((x * 10.0).sin() * 0.5 + 0.5), // Red channel
            1 => ((y * 10.0).sin() * 0.5 + 0.5), // Green channel
            2 => ((x * y * 50.0).sin() * 0.5 + 0.5), // Blue channel
            _ => 0.0,
        }
    });

    let image = Image::from_data(test_image);
    println!("Created test image: {}x{} with {} channels", width, height, channels);

    // Apply Gaussian blur
    let blurred = image.gaussian_blur(2.0)?;
    println!("Applied Gaussian blur with sigma=2.0");

    // Detect edges
    let edges = blurred.sobel_edge_detection()?;
    println!("Applied Sobel edge detection");

    println!("Image processing pipeline completed successfully");
    Ok(())
}
```

## Monte Carlo Simulations

### High-Performance Monte Carlo Integration

```rust
use scirs2_core::{
    testing::TestDataGenerator,
    parallel_ops::parallel_reduce,
    chunking::ChunkConfig,
    simd_ops::SimdUnifiedOps,
    constants::math,
};

fn monte_carlo_pi_estimation(num_samples: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let chunk_config = ChunkConfig::monte_carlo();
    let num_threads = std::thread::available_parallelism()?.get();
    let samples_per_thread = num_samples / num_threads;

    // Parallel Monte Carlo simulation
    let thread_seeds: Vec<u64> = (0..num_threads).map(|i| i as u64 * 12345).collect();

    let total_inside = parallel_reduce(&thread_seeds, 0u64, |acc, &seed| {
        let mut gen = TestDataGenerator::with_seed(seed);
        let mut inside_count = 0u64;

        for _ in 0..samples_per_thread {
            let x = gen.random_float(-1.0, 1.0);
            let y = gen.random_float(-1.0, 1.0);

            if x * x + y * y <= 1.0 {
                inside_count += 1;
            }
        }

        acc + inside_count
    });

    let pi_estimate = 4.0 * total_inside as f64 / num_samples as f64;

    println!("Monte Carlo π estimation:");
    println!("Samples: {}", num_samples);
    println!("Estimated π: {:.6}", pi_estimate);
    println!("Actual π: {:.6}", math::PI);
    println!("Error: {:.6}", (pi_estimate - math::PI).abs());

    Ok(pi_estimate)
}

// Multi-dimensional integration using Monte Carlo
fn monte_carlo_integration<F>(
    func: F,
    bounds: &[(f64, f64)],
    num_samples: usize
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    let dimensions = bounds.len();
    let chunk_config = ChunkConfig::monte_carlo();
    let num_threads = std::thread::available_parallelism()?.get();
    let samples_per_thread = num_samples / num_threads;

    // Calculate volume of integration domain
    let volume: f64 = bounds.iter()
        .map(|(min, max)| max - min)
        .product();

    let thread_seeds: Vec<u64> = (0..num_threads).map(|i| i as u64 * 54321).collect();

    let sum = parallel_reduce(&thread_seeds, 0.0f64, |acc, &seed| {
        let mut gen = TestDataGenerator::with_seed(seed);
        let mut thread_sum = 0.0;

        for _ in 0..samples_per_thread {
            let point: Vec<f64> = bounds.iter()
                .map(|(min, max)| gen.random_float(*min, *max))
                .collect();

            thread_sum += func(&point);
        }

        acc + thread_sum
    });

    let result = volume * sum / num_samples as f64;
    Ok(result)
}

fn monte_carlo_examples() -> Result<(), Box<dyn std::error::Error>> {
    // π estimation
    monte_carlo_pi_estimation(10_000_000)?;

    // Multi-dimensional integration: ∫∫∫ x²y²z² dx dy dz over [0,1]³
    let integral_result = monte_carlo_integration(
        |point| point[0].powi(2) * point[1].powi(2) * point[2].powi(2),
        &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        1_000_000
    )?;

    let analytical_result = 1.0 / 27.0; // (1/3)³
    println!("\nMulti-dimensional integration:");
    println!("Estimated: {:.6}", integral_result);
    println!("Analytical: {:.6}", analytical_result);
    println!("Error: {:.6}", (integral_result - analytical_result).abs());

    // Option pricing using Black-Scholes Monte Carlo
    let option_price = black_scholes_monte_carlo(
        100.0, // Current stock price
        105.0, // Strike price
        0.25,  // Time to expiration (years)
        0.05,  // Risk-free rate
        0.2,   // Volatility
        1_000_000 // Number of simulations
    )?;

    println!("\nOption pricing (Monte Carlo):");
    println!("Call option price: ${:.4}", option_price);

    Ok(())
}

fn black_scholes_monte_carlo(
    s0: f64,     // Current stock price
    strike: f64, // Strike price
    t: f64,      // Time to expiration
    r: f64,      // Risk-free rate
    sigma: f64,  // Volatility
    num_simulations: usize
) -> Result<f64, Box<dyn std::error::Error>> {
    let chunk_config = ChunkConfig::monte_carlo();
    let num_threads = std::thread::available_parallelism()?.get();
    let sims_per_thread = num_simulations / num_threads;

    let thread_seeds: Vec<u64> = (0..num_threads).map(|i| i as u64 * 98765).collect();

    let payoff_sum = parallel_reduce(&thread_seeds, 0.0f64, |acc, &seed| {
        let mut gen = TestDataGenerator::with_seed(seed);
        let mut thread_payoff_sum = 0.0;

        for _ in 0..sims_per_thread {
            // Generate random normal variable (Box-Muller transform)
            let u1 = gen.random_float(0.0, 1.0);
            let u2 = gen.random_float(0.0, 1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * math::PI * u2).cos();

            // Stock price at expiration using geometric Brownian motion
            let st = s0 * ((r - 0.5 * sigma * sigma) * t + sigma * t.sqrt() * z).exp();

            // Call option payoff
            let payoff = (st - strike).max(0.0);
            thread_payoff_sum += payoff;
        }

        acc + thread_payoff_sum
    });

    // Discount to present value
    let option_price = (payoff_sum / num_simulations as f64) * (-r * t).exp();
    Ok(option_price)
}
```

## GPU Computing

### CUDA Acceleration Example

```rust
use scirs2_core::gpu::{
    GpuContext, GpuBackend, GpuBuffer, ElementwiseAddKernel, GemvKernel
};

fn gpu_computing_example() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU context
    let context = GpuContext::new(GpuBackend::CUDA, 0)?;
    println!("GPU context initialized: {:?}", context.backend);

    // Large arrays for GPU processing
    let size = 1_000_000;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    // Create GPU buffers
    let buffer_a = GpuBuffer::from_slice(&context, &a)?;
    let buffer_b = GpuBuffer::from_slice(&context, &b)?;
    let buffer_result = GpuBuffer::zeros(&context, size)?;

    println!("Created GPU buffers for {} elements", size);

    // Execute element-wise addition kernel
    let add_kernel = ElementwiseAddKernel::new();
    add_kernel.launch(&context, &[&buffer_a, &buffer_b, &buffer_result])?;

    // Copy result back to host
    let result = buffer_result.to_vec()?;

    // Verify results
    for i in 0..std::cmp::min(10, size) {
        let expected = a[i] + b[i];
        assert!((result[i] - expected).abs() < 1e-6);
    }

    println!("Element-wise addition completed successfully on GPU");

    // Matrix-vector multiplication example
    let matrix_size = 2048;
    let matrix: Vec<f32> = (0..matrix_size * matrix_size)
        .map(|i| (i as f32).sin())
        .collect();
    let vector: Vec<f32> = (0..matrix_size)
        .map(|i| (i as f32).cos())
        .collect();

    let matrix_buffer = GpuBuffer::from_slice(&context, &matrix)?;
    let vector_buffer = GpuBuffer::from_slice(&context, &vector)?;
    let mv_result_buffer = GpuBuffer::zeros(&context, matrix_size)?;

    // Execute GEMV kernel (General Matrix-Vector multiplication)
    let gemv_kernel = GemvKernel::new();
    gemv_kernel.launch(&context, &[&matrix_buffer, &vector_buffer, &mv_result_buffer])?;

    let mv_result = mv_result_buffer.to_vec()?;
    println!("Matrix-vector multiplication completed: {}x{} matrix", matrix_size, matrix_size);

    Ok(())
}

// Hybrid CPU/GPU computation
fn hybrid_cpu_gpu_example() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_core::{
        chunking::ChunkConfig,
        simd_ops::SimdUnifiedOps,
        parallel_ops::parallel_map,
    };

    let total_size = 10_000_000;
    let gpu_threshold = 100_000; // Use GPU for chunks larger than this

    // Generate test data
    let data: Vec<f32> = (0..total_size)
        .map(|i| (i as f32 / 1000.0).sin())
        .collect();

    let chunk_config = ChunkConfig::gpu_hybrid();
    let chunks: Vec<&[f32]> = data.chunks(gpu_threshold).collect();

    // Process chunks with appropriate backend
    let results: Vec<Vec<f32>> = parallel_map(&chunks, |chunk| {
        if chunk.len() >= gpu_threshold {
            // Use GPU for large chunks
            gpu_process_chunk(chunk).unwrap_or_else(|_| {
                // Fallback to CPU if GPU fails
                cpu_process_chunk(chunk)
            })
        } else {
            // Use CPU for small chunks
            cpu_process_chunk(chunk)
        }
    });

    // Combine results
    let final_result: Vec<f32> = results.into_iter().flatten().collect();

    println!("Hybrid CPU/GPU processing completed: {} elements", final_result.len());
    Ok(())
}

fn gpu_process_chunk(chunk: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let context = GpuContext::new(GpuBackend::CUDA, 0)?;

    let input_buffer = GpuBuffer::from_slice(&context, chunk)?;
    let output_buffer = GpuBuffer::zeros(&context, chunk.len())?;

    // Apply square kernel
    let square_kernel = ElementwiseSquareKernel::new();
    square_kernel.launch(&context, &[&input_buffer, &output_buffer])?;

    output_buffer.to_vec()
}

fn cpu_process_chunk(chunk: &[f32]) -> Vec<f32> {
    // SIMD-accelerated processing on CPU
    let chunk_array = ndarray::Array1::from_slice(chunk);
    let result = f32::simd_mul(&chunk_array.view(), &chunk_array.view());
    result.to_vec()
}
```

## Performance Optimization

### Comprehensive Performance Analysis

```rust
use scirs2_core::{
    testing::BenchmarkSuite,
    memory::{create_scientific_pool, MemoryReport},
    chunking::{ChunkPerformanceMonitor, ChunkMeasurement},
    simd_ops::SimdUnifiedOps,
};
use std::time::{Duration, Instant};

fn performance_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut benchmark_suite = BenchmarkSuite::new()
        .with_regression_threshold(1.1);

    // Set up test data
    let sizes = [1000, 10000, 100000, 1000000];
    let mut memory_pool = create_scientific_pool::<f64>();
    let mut chunk_monitor = ChunkPerformanceMonitor::new();

    for &size in &sizes {
        let data_a: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let data_b: Vec<f64> = (0..size).map(|i| (i * 2) as f64).collect();

        let array_a = ndarray::Array1::from_vec(data_a);
        let array_b = ndarray::Array1::from_vec(data_b);

        // Benchmark scalar operations
        benchmark_suite.add_benchmark(&format!("scalar_add_{}", size), {
            let a = array_a.clone();
            let b = array_b.clone();
            move || {
                let start = Instant::now();
                let _result = &a + &b;
                start.elapsed()
            }
        });

        // Benchmark SIMD operations
        benchmark_suite.add_benchmark(&format!("simd_add_{}", size), {
            let a = array_a.clone();
            let b = array_b.clone();
            move || {
                let start = Instant::now();
                let _result = f64::simd_add(&a.view(), &b.view());
                start.elapsed()
            }
        });

        // Benchmark memory pool operations
        benchmark_suite.add_benchmark(&format!("memory_pool_{}", size), {
            let mut pool = create_scientific_pool::<f64>();
            move || {
                let start = Instant::now();
                let buffer = pool.acquire_vec_advanced(size);
                pool.release_vec_advanced(buffer);
                start.elapsed()
            }
        });
    }

    // Run benchmarks
    let results = benchmark_suite.run_all();
    results.print_results();

    // Analyze memory performance
    let memory_report = memory_pool.memory_report();
    print_memory_analysis(&memory_report);

    // Record chunk performance measurements
    for &size in &sizes {
        let measurement = ChunkMeasurement {
            chunk_size: size / 4,
            data_size: size,
            execution_time: Duration::from_nanos(size as u64 * 10), // Simulated
            throughput: size as f64 / (size as f64 * 10e-9), // GB/s
            operation_type: "vector_add".to_string(),
        };
        chunk_monitor.record_measurement(measurement);
    }

    let chunk_stats = chunk_monitor.get_statistics();
    print_chunk_analysis(&chunk_stats);

    // Performance scaling analysis
    analyze_performance_scaling(&results)?;

    Ok(())
}

fn print_memory_analysis(report: &MemoryReport) {
    println!("\n=== Memory Performance Analysis ===");
    println!("Current usage: {:.2} MB", report.current_usage as f64 / 1_048_576.0);
    println!("Peak usage: {:.2} MB", report.peak_usage as f64 / 1_048_576.0);
    println!("Pool efficiency: {:.1}%", report.pool_efficiency * 100.0);
    println!("Memory pressure: {:?}", report.pressure_level);
    println!("Fragmentation: {:.1}%", report.fragmentation_ratio * 100.0);
}

fn print_chunk_analysis(stats: &scirs2_core::chunking::ChunkStatistics) {
    println!("\n=== Chunking Performance Analysis ===");
    println!("Total measurements: {}", stats.total_measurements);
    println!("Average throughput: {:.2} GB/s", stats.avg_throughput);
    println!("Peak throughput: {:.2} GB/s", stats.max_throughput);
    println!("Minimum throughput: {:.2} GB/s", stats.min_throughput);
    println!("Optimized operations: {}", stats.optimal_operations);
}

fn analyze_performance_scaling(
    results: &scirs2_core::testing::BenchmarkResults
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Performance Scaling Analysis ===");

    let sizes = [1000, 10000, 100000, 1000000];

    // Analyze SIMD speedup
    for &size in &sizes {
        let scalar_time = results.get_result(&format!("scalar_add_{}", size));
        let simd_time = results.get_result(&format!("simd_add_{}", size));

        if let (Some(scalar), Some(simd)) = (scalar_time, simd_time) {
            let speedup = scalar.as_secs_f64() / simd.as_secs_f64();
            println!("Size {}: SIMD speedup = {:.2}x", size, speedup);
        }
    }

    // Analyze memory throughput
    for &size in &sizes {
        let memory_time = results.get_result(&format!("memory_pool_{}", size));

        if let Some(time) = memory_time {
            let throughput = size as f64 * std::mem::size_of::<f64>() as f64 / time.as_secs_f64() / 1e9;
            println!("Size {}: Memory throughput = {:.2} GB/s", size, throughput);
        }
    }

    Ok(())
}

// Cache performance analysis
fn cache_performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_core::chunking::{ChunkConfig, CacheAwareness};

    println!("\n=== Cache Performance Analysis ===");

    let data_size = 1_000_000;
    let test_data: Vec<f64> = (0..data_size).map(|i| i as f64).collect();
    let array = ndarray::Array1::from_vec(test_data);

    // Test different cache awareness levels
    let cache_levels = [
        CacheAwareness::None,
        CacheAwareness::L1,
        CacheAwareness::L2,
        CacheAwareness::L3,
        CacheAwareness::Full,
    ];

    for cache_level in &cache_levels {
        let config = ChunkConfig::default()
            .with_cache_awareness(*cache_level);

        let start = Instant::now();

        // Simulate cache-aware processing
        let chunk_size = scirs2_core::chunking::ChunkingUtils::optimal_chunk_size(data_size, &config);
        let _result = f64::simd_sum(&array.view());

        let duration = start.elapsed();

        println!("Cache level {:?}: {:.3} ms (chunk size: {})",
                 cache_level, duration.as_secs_f64() * 1000.0, chunk_size);
    }

    Ok(())
}
```

This comprehensive examples document demonstrates the full capabilities of SciRS2-Core across different scientific computing domains, showcasing performance optimization techniques, GPU acceleration, advanced memory management, and robust testing methodologies.
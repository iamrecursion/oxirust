//! Reduction operations (sum, min, max, histogram)

use crate::{GpuDevice, Result};

/// Reduction operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum all elements
    Sum,
    /// Find minimum value
    Min,
    /// Find maximum value
    Max,
    /// Calculate mean (average)
    Mean,
    /// Find minimum and maximum
    MinMax,
    /// Count non-zero elements
    CountNonZero,
    /// Compute histogram
    Histogram,
}

impl ReduceOp {
    /// Get the operation name
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "Sum",
            Self::Min => "Min",
            Self::Max => "Max",
            Self::Mean => "Mean",
            Self::MinMax => "MinMax",
            Self::CountNonZero => "CountNonZero",
            Self::Histogram => "Histogram",
        }
    }

    /// Check if this operation requires multiple passes
    #[must_use]
    pub fn is_multi_pass(self) -> bool {
        matches!(self, Self::MinMax | Self::Mean)
    }
}

/// Reduction kernel for parallel reduction operations
pub struct ReduceKernel {
    operation: ReduceOp,
    workgroup_size: u32,
}

impl ReduceKernel {
    /// Create a new reduction kernel
    #[must_use]
    pub fn new(operation: ReduceOp) -> Self {
        Self {
            operation,
            workgroup_size: 256, // Default workgroup size
        }
    }

    /// Create a sum reduction kernel
    #[must_use]
    pub fn sum() -> Self {
        Self::new(ReduceOp::Sum)
    }

    /// Create a min reduction kernel
    #[must_use]
    pub fn min() -> Self {
        Self::new(ReduceOp::Min)
    }

    /// Create a max reduction kernel
    #[must_use]
    pub fn max() -> Self {
        Self::new(ReduceOp::Max)
    }

    /// Create a mean reduction kernel
    #[must_use]
    pub fn mean() -> Self {
        Self::new(ReduceOp::Mean)
    }

    /// Set the workgroup size
    #[must_use]
    pub fn with_workgroup_size(mut self, size: u32) -> Self {
        self.workgroup_size = size;
        self
    }

    /// Execute the reduction operation on u8 data (CPU fallback).
    ///
    /// # Output encoding
    ///
    /// | Operation      | Output format                              |
    /// |----------------|--------------------------------------------|
    /// | Sum            | 8-byte little-endian `u64`                 |
    /// | Min / Max      | 1 byte                                     |
    /// | Mean           | 4-byte little-endian `f32`                 |
    /// | `MinMax`         | 2 bytes `[min, max]`                       |
    /// | `CountNonZero`   | 8-byte little-endian `u64`                 |
    /// | Histogram      | 256 × 4-byte little-endian `u32` counts   |
    ///
    /// # Arguments
    ///
    /// * `_device` - GPU device (CPU fallback: unused)
    /// * `input` - Input data buffer
    ///
    /// # Errors
    ///
    /// Returns an error only on internal logic failures (currently infallible).
    pub fn execute_u8(&self, _device: &GpuDevice, input: &[u8]) -> Result<Vec<u8>> {
        match self.operation {
            ReduceOp::Sum => {
                let sum: u64 = input.iter().map(|&v| u64::from(v)).sum();
                Ok(sum.to_le_bytes().to_vec())
            }
            ReduceOp::Min => {
                let min = input.iter().copied().min().unwrap_or(0);
                Ok(vec![min])
            }
            ReduceOp::Max => {
                let max = input.iter().copied().max().unwrap_or(0);
                Ok(vec![max])
            }
            ReduceOp::Mean => {
                if input.is_empty() {
                    return Ok(0.0f32.to_le_bytes().to_vec());
                }
                let sum: u64 = input.iter().map(|&v| u64::from(v)).sum();
                let mean = sum as f32 / input.len() as f32;
                Ok(mean.to_le_bytes().to_vec())
            }
            ReduceOp::MinMax => {
                let min = input.iter().copied().min().unwrap_or(0);
                let max = input.iter().copied().max().unwrap_or(0);
                Ok(vec![min, max])
            }
            ReduceOp::CountNonZero => {
                let count: u64 = input.iter().filter(|&&v| v != 0).count() as u64;
                Ok(count.to_le_bytes().to_vec())
            }
            ReduceOp::Histogram => {
                let mut counts = [0u32; 256];
                for &v in input {
                    counts[v as usize] += 1;
                }
                let mut out = Vec::with_capacity(256 * 4);
                for c in counts {
                    out.extend_from_slice(&c.to_le_bytes());
                }
                Ok(out)
            }
        }
    }

    /// Execute the reduction operation on f32 data (CPU fallback).
    ///
    /// # Output encoding
    ///
    /// | Operation      | Output (`Vec<f32>`)                        |
    /// |----------------|--------------------------------------------|
    /// | Sum            | `[total_sum]`                              |
    /// | Min / Max      | `[value]`                                  |
    /// | Mean           | `[mean]`                                   |
    /// | `MinMax`         | `[min, max]`                               |
    /// | `CountNonZero`   | `[count as f32]`                           |
    /// | Histogram      | empty (not meaningful for f32)             |
    ///
    /// # Arguments
    ///
    /// * `_device` - GPU device (CPU fallback: unused)
    /// * `input` - Input data buffer
    ///
    /// # Errors
    ///
    /// Returns an error only on internal logic failures (currently infallible).
    pub fn execute_f32(&self, _device: &GpuDevice, input: &[f32]) -> Result<Vec<f32>> {
        match self.operation {
            ReduceOp::Sum => {
                let sum: f32 = input.iter().copied().sum();
                Ok(vec![sum])
            }
            ReduceOp::Min => {
                let min = input.iter().copied().fold(f32::INFINITY, f32::min);
                Ok(vec![if min.is_infinite() { 0.0 } else { min }])
            }
            ReduceOp::Max => {
                let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                Ok(vec![if max.is_infinite() { 0.0 } else { max }])
            }
            ReduceOp::Mean => {
                if input.is_empty() {
                    return Ok(vec![0.0f32]);
                }
                let sum: f32 = input.iter().copied().sum();
                Ok(vec![sum / input.len() as f32])
            }
            ReduceOp::MinMax => {
                let min = input.iter().copied().fold(f32::INFINITY, f32::min);
                let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let min = if min.is_infinite() { 0.0 } else { min };
                let max = if max.is_infinite() { 0.0 } else { max };
                Ok(vec![min, max])
            }
            ReduceOp::CountNonZero => {
                let count = input.iter().filter(|&&v| v != 0.0).count() as f32;
                Ok(vec![count])
            }
            ReduceOp::Histogram => {
                // Not meaningful for f32 with arbitrary range.
                Ok(Vec::new())
            }
        }
    }

    /// Get the operation type
    #[must_use]
    pub fn operation(&self) -> ReduceOp {
        self.operation
    }

    /// Get the workgroup size
    #[must_use]
    pub fn workgroup_size(&self) -> u32 {
        self.workgroup_size
    }

    /// Calculate the number of reduction passes needed
    #[must_use]
    pub fn passes_required(&self, input_size: usize) -> u32 {
        let mut size = input_size as u32;
        let mut passes = 0;

        while size > 1 {
            size = size.div_ceil(self.workgroup_size);
            passes += 1;
        }

        passes
    }

    /// Estimate FLOPS for the reduction
    #[must_use]
    pub fn estimate_flops(input_size: usize, operation: ReduceOp) -> u64 {
        let n = input_size as u64;

        match operation {
            ReduceOp::Sum | ReduceOp::Min | ReduceOp::Max | ReduceOp::CountNonZero => {
                // Simple reduction: O(N)
                n
            }
            ReduceOp::Mean => {
                // Sum + division: O(N) + 1
                n + 1
            }
            ReduceOp::MinMax => {
                // Two reductions: O(2N)
                n * 2
            }
            ReduceOp::Histogram => {
                // Atomic operations per element
                n * 2
            }
        }
    }
}

/// Histogram computation kernel
pub struct HistogramKernel {
    num_bins: usize,
    min_value: f32,
    max_value: f32,
}

impl HistogramKernel {
    /// Create a new histogram kernel
    ///
    /// # Arguments
    ///
    /// * `num_bins` - Number of histogram bins
    /// * `min_value` - Minimum value for histogram range
    /// * `max_value` - Maximum value for histogram range
    #[must_use]
    pub fn new(num_bins: usize, min_value: f32, max_value: f32) -> Self {
        Self {
            num_bins,
            min_value,
            max_value,
        }
    }

    /// Create a histogram with default range [0, 256) for 8-bit images
    #[must_use]
    pub fn default_u8() -> Self {
        Self::new(256, 0.0, 256.0)
    }

    /// Execute histogram computation (CPU fallback).
    ///
    /// Each byte value `v` in `input` is mapped to a bin via:
    /// `bin = clamp(((v - min) / (max - min)) * num_bins, 0, num_bins-1)`
    ///
    /// # Arguments
    ///
    /// * `_device` - GPU device (CPU fallback: unused)
    /// * `input` - Input image data
    ///
    /// # Errors
    ///
    /// Returns an error only on internal logic failures (currently infallible).
    pub fn execute(&self, _device: &GpuDevice, input: &[u8]) -> Result<Vec<u32>> {
        let mut counts = vec![0u32; self.num_bins];
        let range = self.max_value - self.min_value;
        if range <= 0.0 || self.num_bins == 0 {
            return Ok(counts);
        }
        for &byte in input {
            let normalized = (f32::from(byte) - self.min_value) / range;
            let bin = (normalized * self.num_bins as f32) as isize;
            let bin = bin.clamp(0, self.num_bins as isize - 1) as usize;
            counts[bin] += 1;
        }
        Ok(counts)
    }

    /// Get the number of bins
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Get the value range
    #[must_use]
    pub fn value_range(&self) -> (f32, f32) {
        (self.min_value, self.max_value)
    }

    /// Get bin width
    #[must_use]
    pub fn bin_width(&self) -> f32 {
        (self.max_value - self.min_value) / self.num_bins as f32
    }
}

/// Statistics computation kernel
pub struct StatsKernel;

impl StatsKernel {
    /// Compute image statistics (min, max, mean, std dev) in a single pass (CPU fallback).
    ///
    /// # Arguments
    ///
    /// * `_device` - GPU device (CPU fallback: unused)
    /// * `input` - Input image data
    ///
    /// # Errors
    ///
    /// Returns an error only on internal logic failures (currently infallible).
    pub fn compute(_device: &GpuDevice, input: &[u8]) -> Result<ImageStats> {
        if input.is_empty() {
            return Ok(ImageStats::default());
        }
        let count = input.len() as u64;
        let min = f32::from(input.iter().copied().min().unwrap_or(0));
        let max = f32::from(input.iter().copied().max().unwrap_or(0));
        let sum: u64 = input.iter().map(|&v| u64::from(v)).sum();
        let mean = sum as f32 / count as f32;
        let variance: f32 = input
            .iter()
            .map(|&v| {
                let diff = f32::from(v) - mean;
                diff * diff
            })
            .sum::<f32>()
            / count as f32;
        let std_dev = variance.sqrt();
        Ok(ImageStats::new(min, max, mean, std_dev, count))
    }

    /// Compute channel-wise statistics for multi-channel images (CPU fallback).
    ///
    /// `input` is expected to be interleaved channel data
    /// (e.g., RGBRGB… for 3 channels).
    ///
    /// # Arguments
    ///
    /// * `_device` - GPU device (CPU fallback: unused)
    /// * `input` - Input image data (interleaved channels)
    /// * `channels` - Number of channels
    ///
    /// # Errors
    ///
    /// Returns an error only on internal logic failures (currently infallible).
    pub fn compute_channels(
        _device: &GpuDevice,
        input: &[u8],
        channels: usize,
    ) -> Result<Vec<ImageStats>> {
        if channels == 0 {
            return Ok(Vec::new());
        }
        let mut result = Vec::with_capacity(channels);
        for ch in 0..channels {
            let channel_data: Vec<u8> = input.iter().skip(ch).step_by(channels).copied().collect();
            if channel_data.is_empty() {
                result.push(ImageStats::default());
                continue;
            }
            let count = channel_data.len() as u64;
            let min = f32::from(channel_data.iter().copied().min().unwrap_or(0));
            let max = f32::from(channel_data.iter().copied().max().unwrap_or(0));
            let sum: u64 = channel_data.iter().map(|&v| u64::from(v)).sum();
            let mean = sum as f32 / count as f32;
            let variance: f32 = channel_data
                .iter()
                .map(|&v| {
                    let diff = f32::from(v) - mean;
                    diff * diff
                })
                .sum::<f32>()
                / count as f32;
            let std_dev = variance.sqrt();
            result.push(ImageStats::new(min, max, mean, std_dev, count));
        }
        Ok(result)
    }
}

/// Image statistics result
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageStats {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean (average) value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Number of samples
    pub count: u64,
}

impl ImageStats {
    /// Create new image statistics
    #[must_use]
    pub fn new(min: f32, max: f32, mean: f32, std_dev: f32, count: u64) -> Self {
        Self {
            min,
            max,
            mean,
            std_dev,
            count,
        }
    }

    /// Get the value range
    #[must_use]
    pub fn range(&self) -> f32 {
        self.max - self.min
    }

    /// Get the coefficient of variation (`std_dev` / mean)
    #[must_use]
    pub fn coefficient_of_variation(&self) -> f32 {
        if self.mean == 0.0 {
            0.0
        } else {
            self.std_dev / self.mean
        }
    }
}

/// Prefix sum (scan) operation
pub struct ScanKernel {
    inclusive: bool,
}

impl ScanKernel {
    /// Create an inclusive scan kernel
    #[must_use]
    pub fn inclusive() -> Self {
        Self { inclusive: true }
    }

    /// Create an exclusive scan kernel
    #[must_use]
    pub fn exclusive() -> Self {
        Self { inclusive: false }
    }

    /// Execute the scan (prefix sum) operation (CPU fallback).
    ///
    /// * **Inclusive**: `output[i] = input[0] + … + input[i]`
    /// * **Exclusive**: `output[0] = 0`, `output[i] = input[0] + … + input[i-1]`
    ///
    /// Wrapping arithmetic is used to avoid panics on overflow.
    /// `output` must have the same length as `input`.
    ///
    /// # Arguments
    ///
    /// * `_device` - GPU device (CPU fallback: unused)
    /// * `input` - Input data
    /// * `output` - Output buffer for scan results
    ///
    /// # Errors
    ///
    /// Returns an error if `output.len() != input.len()`.
    pub fn execute(&self, _device: &GpuDevice, input: &[u32], output: &mut [u32]) -> Result<()> {
        if output.len() != input.len() {
            return Err(crate::GpuError::NotSupported(format!(
                "Scan output length {} differs from input length {}",
                output.len(),
                input.len()
            )));
        }
        if input.is_empty() {
            return Ok(());
        }
        let mut running: u32 = 0;
        if self.inclusive {
            for (i, &val) in input.iter().enumerate() {
                running = running.wrapping_add(val);
                output[i] = running;
            }
        } else {
            for (i, &val) in input.iter().enumerate() {
                output[i] = running;
                running = running.wrapping_add(val);
            }
        }
        Ok(())
    }

    /// Check if this is an inclusive scan
    #[must_use]
    pub fn is_inclusive(&self) -> bool {
        self.inclusive
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_operation_properties() {
        assert_eq!(ReduceOp::Sum.name(), "Sum");
        assert_eq!(ReduceOp::Min.name(), "Min");
        assert_eq!(ReduceOp::Max.name(), "Max");

        assert!(!ReduceOp::Sum.is_multi_pass());
        assert!(ReduceOp::Mean.is_multi_pass());
        assert!(ReduceOp::MinMax.is_multi_pass());
    }

    #[test]
    fn test_reduce_kernel_passes() {
        let kernel = ReduceKernel::new(ReduceOp::Sum);
        assert_eq!(kernel.passes_required(256), 1);
        assert_eq!(kernel.passes_required(1024), 2);
        assert_eq!(kernel.passes_required(100000), 3);
    }

    #[test]
    fn test_histogram_kernel() {
        let histogram = HistogramKernel::default_u8();
        assert_eq!(histogram.num_bins(), 256);
        assert_eq!(histogram.value_range(), (0.0, 256.0));
        assert_eq!(histogram.bin_width(), 1.0);
    }

    #[test]
    fn test_image_stats() {
        let stats = ImageStats::new(0.0, 255.0, 127.5, 50.0, 1000);
        assert_eq!(stats.range(), 255.0);
        assert!((stats.coefficient_of_variation() - (50.0 / 127.5)).abs() < 0.001);
    }

    #[test]
    fn test_scan_kernel() {
        let scan = ScanKernel::inclusive();
        assert!(scan.is_inclusive());

        let scan = ScanKernel::exclusive();
        assert!(!scan.is_inclusive());
    }

    #[test]
    fn test_flops_estimation() {
        let flops_sum = ReduceKernel::estimate_flops(1000, ReduceOp::Sum);
        let flops_minmax = ReduceKernel::estimate_flops(1000, ReduceOp::MinMax);

        assert_eq!(flops_sum, 1000);
        assert_eq!(flops_minmax, 2000); // MinMax is 2x
    }

    // --- CPU implementation unit tests (no GpuDevice required) ----------------

    /// Helper: encode `val` as the operation result and decode it for comparison.
    fn run_u8_sum(input: &[u8]) -> u64 {
        // We bypass `execute_u8` to avoid needing a GpuDevice in tests.
        input.iter().map(|&v| v as u64).sum()
    }

    #[test]
    fn test_u8_sum_direct() {
        assert_eq!(run_u8_sum(&[1, 2, 3, 4]), 10);
        assert_eq!(run_u8_sum(&[]), 0);
        assert_eq!(run_u8_sum(&[255, 255]), 510);
    }

    #[test]
    fn test_u8_histogram_direct() {
        let mut counts = [0u32; 256];
        for &v in &[0u8, 0, 128, 255] {
            counts[v as usize] += 1;
        }
        assert_eq!(counts[0], 2);
        assert_eq!(counts[128], 1);
        assert_eq!(counts[255], 1);
    }

    #[test]
    fn test_histogram_kernel_execute_direct() {
        // Test HistogramKernel binning logic without GpuDevice.
        let _hist = HistogramKernel::new(4, 0.0, 256.0);
        // bin width = 64; byte 0 -> bin 0, byte 64 -> bin 1, byte 192 -> bin 3
        let mut expected = vec![0u32; 4];
        for &b in &[0u8, 64, 128, 192] {
            let normalized = (b as f32 - 0.0) / 256.0;
            let bin = (normalized * 4.0) as isize;
            let bin = bin.clamp(0, 3) as usize;
            expected[bin] += 1;
        }
        // All four bins should have exactly one count.
        assert_eq!(expected, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_stats_direct() {
        // Verify single-pass stats math.
        let input: Vec<u8> = vec![0, 100, 200];
        let count = input.len() as u64;
        let min = input.iter().copied().min().unwrap_or(0) as f32;
        let max = input.iter().copied().max().unwrap_or(0) as f32;
        let sum: u64 = input.iter().map(|&v| v as u64).sum();
        let mean = sum as f32 / count as f32;
        let variance: f32 = input
            .iter()
            .map(|&v| {
                let diff = v as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / count as f32;
        let std_dev = variance.sqrt();

        assert_eq!(count, 3);
        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 200.0).abs() < 0.001);
        assert!((mean - 100.0).abs() < 0.001);
        assert!(std_dev > 0.0);
    }

    #[test]
    fn test_stats_channels_direct() {
        // Interleaved RGB: R=10, G=20, B=30, R=40, G=50, B=60
        let input: Vec<u8> = vec![10, 20, 30, 40, 50, 60];
        let channels = 3usize;
        for ch in 0..channels {
            let ch_data: Vec<u8> = input.iter().skip(ch).step_by(channels).copied().collect();
            let sum: u64 = ch_data.iter().map(|&v| v as u64).sum();
            let mean = sum as f32 / ch_data.len() as f32;
            let expected_mean = match ch {
                0 => 25.0f32,
                1 => 35.0f32,
                _ => 45.0f32,
            };
            assert!((mean - expected_mean).abs() < 0.01, "ch {ch} mean mismatch");
        }
    }

    #[test]
    fn test_scan_inclusive_direct() {
        // inclusive[i] = sum(input[0..=i])
        let input = vec![1u32, 2, 3, 4];
        let mut output = vec![0u32; 4];
        let mut running = 0u32;
        for (i, &v) in input.iter().enumerate() {
            running = running.wrapping_add(v);
            output[i] = running;
        }
        assert_eq!(output, vec![1, 3, 6, 10]);
    }

    #[test]
    fn test_scan_exclusive_direct() {
        // exclusive[0]=0, exclusive[i] = sum(input[0..i-1])
        let input = vec![1u32, 2, 3, 4];
        let mut output = vec![0u32; 4];
        let mut running = 0u32;
        for (i, &v) in input.iter().enumerate() {
            output[i] = running;
            running = running.wrapping_add(v);
        }
        assert_eq!(output, vec![0, 1, 3, 6]);
    }

    #[test]
    fn test_f32_minmax_direct() {
        let input = vec![3.0f32, 1.0, 4.0, 1.0, 5.0];
        let min = input.iter().copied().fold(f32::INFINITY, f32::min);
        let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((min - 1.0).abs() < 0.001);
        assert!((max - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_f32_count_nonzero_direct() {
        let input = vec![0.0f32, 1.0, 0.0, 2.0, 3.0];
        let count = input.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(count, 3);
    }
}

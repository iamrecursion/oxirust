// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Statistics collection and analysis for Theora encoding.
//!
//! Provides comprehensive encoding statistics for debugging and optimization.

use std::time::{Duration, Instant};

/// Frame encoding statistics.
#[derive(Debug, Clone)]
pub struct FrameEncodingStats {
    /// Frame number.
    pub frame_number: u64,
    /// Frame type (true = keyframe).
    pub is_keyframe: bool,
    /// Encoded size in bytes.
    pub size_bytes: usize,
    /// Quality parameter used.
    pub quality: u8,
    /// PSNR (Peak Signal-to-Noise Ratio) for Y component.
    pub psnr_y: f64,
    /// PSNR for U component.
    pub psnr_u: f64,
    /// PSNR for V component.
    pub psnr_v: f64,
    /// SSIM (Structural Similarity Index) for Y component.
    pub ssim_y: f64,
    /// Encoding time.
    pub encode_time: Duration,
    /// Number of intra blocks.
    pub intra_blocks: usize,
    /// Number of inter blocks.
    pub inter_blocks: usize,
    /// Number of skip blocks.
    pub skip_blocks: usize,
    /// Average motion vector magnitude.
    pub avg_mv_magnitude: f32,
    /// Bitrate (bits per second).
    pub bitrate: f64,
}

impl FrameEncodingStats {
    /// Create a new frame statistics entry.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_number: u64,
        is_keyframe: bool,
        size_bytes: usize,
        quality: u8,
        encode_time: Duration,
    ) -> Self {
        Self {
            frame_number,
            is_keyframe,
            size_bytes,
            quality,
            psnr_y: 0.0,
            psnr_u: 0.0,
            psnr_v: 0.0,
            ssim_y: 0.0,
            encode_time,
            intra_blocks: 0,
            inter_blocks: 0,
            skip_blocks: 0,
            avg_mv_magnitude: 0.0,
            bitrate: 0.0,
        }
    }

    /// Calculate average PSNR across all components.
    #[must_use]
    pub fn avg_psnr(&self) -> f64 {
        (self.psnr_y + self.psnr_u + self.psnr_v) / 3.0
    }

    /// Get total number of blocks.
    #[must_use]
    pub fn total_blocks(&self) -> usize {
        self.intra_blocks + self.inter_blocks + self.skip_blocks
    }

    /// Get percentage of intra blocks.
    #[must_use]
    pub fn intra_percentage(&self) -> f64 {
        let total = self.total_blocks();
        if total > 0 {
            (self.intra_blocks as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Encoder session statistics.
#[derive(Debug, Clone)]
pub struct EncoderStats {
    /// Frame statistics.
    pub frames: Vec<FrameEncodingStats>,
    /// Total encoding time.
    pub total_time: Duration,
    /// Start time.
    start_time: Option<Instant>,
}

impl EncoderStats {
    /// Create a new encoder statistics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            total_time: Duration::ZERO,
            start_time: None,
        }
    }

    /// Start timing.
    pub fn start_timing(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Stop timing and update total time.
    pub fn stop_timing(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.total_time = start.elapsed();
        }
    }

    /// Add frame statistics.
    pub fn add_frame(&mut self, stats: FrameEncodingStats) {
        self.frames.push(stats);
    }

    /// Get average bitrate across all frames.
    #[must_use]
    pub fn average_bitrate(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }

        let total_bytes: usize = self.frames.iter().map(|f| f.size_bytes).sum();
        let total_bits = total_bytes * 8;
        let duration_secs = self.total_time.as_secs_f64();

        if duration_secs > 0.0 {
            total_bits as f64 / duration_secs
        } else {
            0.0
        }
    }

    /// Get average PSNR across all frames.
    #[must_use]
    pub fn average_psnr(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.frames.iter().map(|f| f.avg_psnr()).sum();
        sum / self.frames.len() as f64
    }

    /// Get average encoding time per frame.
    #[must_use]
    pub fn average_encode_time(&self) -> Duration {
        if self.frames.is_empty() {
            return Duration::ZERO;
        }

        let total_nanos: u128 = self.frames.iter().map(|f| f.encode_time.as_nanos()).sum();
        Duration::from_nanos((total_nanos / self.frames.len() as u128) as u64)
    }

    /// Get encoding speed in frames per second.
    #[must_use]
    pub fn frames_per_second(&self) -> f64 {
        let duration_secs = self.total_time.as_secs_f64();
        if duration_secs > 0.0 {
            self.frames.len() as f64 / duration_secs
        } else {
            0.0
        }
    }

    /// Get keyframe interval statistics.
    #[must_use]
    pub fn keyframe_interval_stats(&self) -> KeyframeIntervalStats {
        let mut intervals = Vec::new();
        let mut last_keyframe = 0usize;

        for (i, frame) in self.frames.iter().enumerate() {
            if frame.is_keyframe {
                if i > 0 {
                    intervals.push(i - last_keyframe);
                }
                last_keyframe = i;
            }
        }

        KeyframeIntervalStats::from_intervals(&intervals)
    }

    /// Generate summary report.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Encoder Statistics:\n\
             Total Frames: {}\n\
             Total Time: {:.2}s\n\
             Average Bitrate: {:.2} kbps\n\
             Average PSNR: {:.2} dB\n\
             Average Encode Time: {:.2}ms/frame\n\
             Encoding Speed: {:.2} fps",
            self.frames.len(),
            self.total_time.as_secs_f64(),
            self.average_bitrate() / 1000.0,
            self.average_psnr(),
            self.average_encode_time().as_secs_f64() * 1000.0,
            self.frames_per_second()
        )
    }
}

impl Default for EncoderStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Keyframe interval statistics.
#[derive(Debug, Clone)]
pub struct KeyframeIntervalStats {
    /// Minimum interval.
    pub min: usize,
    /// Maximum interval.
    pub max: usize,
    /// Average interval.
    pub average: f64,
}

impl KeyframeIntervalStats {
    /// Create from interval list.
    #[must_use]
    pub fn from_intervals(intervals: &[usize]) -> Self {
        if intervals.is_empty() {
            return Self {
                min: 0,
                max: 0,
                average: 0.0,
            };
        }

        let min = *intervals.iter().min().unwrap_or(&0);
        let max = *intervals.iter().max().unwrap_or(&0);
        let sum: usize = intervals.iter().sum();
        let average = sum as f64 / intervals.len() as f64;

        Self { min, max, average }
    }
}

/// Quality metrics calculator.
pub struct QualityMetrics;

impl QualityMetrics {
    /// Calculate PSNR between two planes.
    #[must_use]
    pub fn calculate_psnr(
        original: &[u8],
        reconstructed: &[u8],
        width: usize,
        height: usize,
    ) -> f64 {
        let size = width * height;
        if size == 0 || original.len() < size || reconstructed.len() < size {
            return 0.0;
        }

        let mut mse = 0.0;
        for i in 0..size {
            let diff = f64::from(original[i]) - f64::from(reconstructed[i]);
            mse += diff * diff;
        }
        mse /= size as f64;

        if mse < 1e-10 {
            return 100.0; // Perfect match
        }

        let max_val = 255.0;
        20.0 * (max_val / mse.sqrt()).log10()
    }

    /// Calculate SSIM between two planes.
    #[must_use]
    pub fn calculate_ssim(
        original: &[u8],
        reconstructed: &[u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> f64 {
        const C1: f64 = 6.5025; // (0.01 * 255)^2
        const C2: f64 = 58.5225; // (0.03 * 255)^2

        let mut ssim_sum = 0.0;
        let mut count = 0;

        // Calculate SSIM for 8x8 windows
        for y in (0..height - 8).step_by(4) {
            for x in (0..width - 8).step_by(4) {
                let (mu_x, mu_y, sigma_x, sigma_y, sigma_xy) =
                    Self::calculate_window_stats(original, reconstructed, stride, x, y);

                let numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2);
                let denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2);

                if denominator > 0.0 {
                    ssim_sum += numerator / denominator;
                    count += 1;
                }
            }
        }

        if count > 0 {
            ssim_sum / count as f64
        } else {
            0.0
        }
    }

    /// Calculate statistics for an 8x8 window.
    fn calculate_window_stats(
        original: &[u8],
        reconstructed: &[u8],
        stride: usize,
        x: usize,
        y: usize,
    ) -> (f64, f64, f64, f64, f64) {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;
        let mut sum_xy = 0.0;
        let n = 64.0;

        for dy in 0..8 {
            for dx in 0..8 {
                let offset = (y + dy) * stride + x + dx;
                if offset < original.len() && offset < reconstructed.len() {
                    let px = f64::from(original[offset]);
                    let py = f64::from(reconstructed[offset]);

                    sum_x += px;
                    sum_y += py;
                    sum_xx += px * px;
                    sum_yy += py * py;
                    sum_xy += px * py;
                }
            }
        }

        let mu_x = sum_x / n;
        let mu_y = sum_y / n;
        let sigma_x = (sum_xx / n - mu_x * mu_x).max(0.0);
        let sigma_y = (sum_yy / n - mu_y * mu_y).max(0.0);
        let sigma_xy = sum_xy / n - mu_x * mu_y;

        (mu_x, mu_y, sigma_x, sigma_y, sigma_xy)
    }
}

/// Bitrate distribution analyzer.
#[derive(Debug, Clone)]
pub struct BitrateDistribution {
    /// Bitrate histogram (in kbps buckets).
    buckets: Vec<usize>,
    /// Bucket size in kbps.
    bucket_size: usize,
}

impl BitrateDistribution {
    /// Create a new bitrate distribution analyzer.
    #[must_use]
    pub fn new(bucket_size: usize) -> Self {
        Self {
            buckets: vec![0; 100],
            bucket_size,
        }
    }

    /// Add a frame bitrate.
    pub fn add_frame(&mut self, bitrate: f64) {
        let bucket_idx = (bitrate / 1000.0 / self.bucket_size as f64) as usize;
        if bucket_idx < self.buckets.len() {
            self.buckets[bucket_idx] += 1;
        }
    }

    /// Get the modal bitrate bucket.
    #[must_use]
    pub fn mode(&self) -> usize {
        self.buckets
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx * self.bucket_size)
            .unwrap_or(0)
    }

    /// Get bitrate variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let total: usize = self.buckets.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let mean = self.mean();
        let mut variance = 0.0;

        for (idx, &count) in self.buckets.iter().enumerate() {
            let value = (idx * self.bucket_size) as f64;
            let diff = value - mean;
            variance += diff * diff * count as f64;
        }

        variance / total as f64
    }

    /// Get mean bitrate.
    #[must_use]
    pub fn mean(&self) -> f64 {
        let total: usize = self.buckets.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let sum: usize = self
            .buckets
            .iter()
            .enumerate()
            .map(|(idx, &count)| idx * self.bucket_size * count)
            .sum();

        sum as f64 / total as f64
    }
}

/// Complexity analyzer for frames.
pub struct ComplexityAnalyzer {
    /// Complexity history.
    history: Vec<f64>,
    /// Maximum history size.
    max_history: usize,
}

impl ComplexityAnalyzer {
    /// Create a new complexity analyzer.
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history,
        }
    }

    /// Add frame complexity measurement.
    pub fn add_measurement(&mut self, complexity: f64) {
        self.history.push(complexity);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get average complexity.
    #[must_use]
    pub fn average(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.history.iter().sum();
        sum / self.history.len() as f64
    }

    /// Get complexity trend (positive = increasing).
    #[must_use]
    pub fn trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let half = self.history.len() / 2;
        let first_half: f64 = self.history[..half].iter().sum::<f64>() / half as f64;
        let second_half: f64 =
            self.history[half..].iter().sum::<f64>() / (self.history.len() - half) as f64;

        second_half - first_half
    }

    /// Get complexity standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }

        let mean = self.average();
        let variance: f64 = self
            .history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.history.len() as f64;

        variance.sqrt()
    }
}

/// Encoder performance profiler.
#[derive(Debug, Clone, Default)]
pub struct PerformanceProfiler {
    /// Time spent in motion estimation.
    pub motion_estimation_time: Duration,
    /// Time spent in mode decision.
    pub mode_decision_time: Duration,
    /// Time spent in transform.
    pub transform_time: Duration,
    /// Time spent in quantization.
    pub quantization_time: Duration,
    /// Time spent in entropy coding.
    pub entropy_coding_time: Duration,
    /// Time spent in reconstruction.
    pub reconstruction_time: Duration,
}

impl PerformanceProfiler {
    /// Get total profiled time.
    #[must_use]
    pub fn total_time(&self) -> Duration {
        self.motion_estimation_time
            + self.mode_decision_time
            + self.transform_time
            + self.quantization_time
            + self.entropy_coding_time
            + self.reconstruction_time
    }

    /// Get percentage breakdown.
    #[must_use]
    pub fn percentage_breakdown(&self) -> ProfilerBreakdown {
        let total = self.total_time().as_secs_f64();
        if total < 1e-6 {
            return ProfilerBreakdown::default();
        }

        ProfilerBreakdown {
            motion_estimation: (self.motion_estimation_time.as_secs_f64() / total) * 100.0,
            mode_decision: (self.mode_decision_time.as_secs_f64() / total) * 100.0,
            transform: (self.transform_time.as_secs_f64() / total) * 100.0,
            quantization: (self.quantization_time.as_secs_f64() / total) * 100.0,
            entropy_coding: (self.entropy_coding_time.as_secs_f64() / total) * 100.0,
            reconstruction: (self.reconstruction_time.as_secs_f64() / total) * 100.0,
        }
    }
}

/// Profiler percentage breakdown.
#[derive(Debug, Clone, Default)]
pub struct ProfilerBreakdown {
    /// Motion estimation percentage.
    pub motion_estimation: f64,
    /// Mode decision percentage.
    pub mode_decision: f64,
    /// Transform percentage.
    pub transform: f64,
    /// Quantization percentage.
    pub quantization: f64,
    /// Entropy coding percentage.
    pub entropy_coding: f64,
    /// Reconstruction percentage.
    pub reconstruction: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_stats() {
        let stats = FrameEncodingStats::new(0, true, 10000, 30, Duration::from_millis(50));
        assert_eq!(stats.frame_number, 0);
        assert!(stats.is_keyframe);
        assert_eq!(stats.size_bytes, 10000);
    }

    #[test]
    fn test_encoder_stats() {
        let mut stats = EncoderStats::new();
        stats.start_timing();

        stats.add_frame(FrameEncodingStats::new(
            0,
            true,
            10000,
            30,
            Duration::from_millis(50),
        ));
        stats.add_frame(FrameEncodingStats::new(
            1,
            false,
            5000,
            32,
            Duration::from_millis(40),
        ));

        assert_eq!(stats.frames.len(), 2);
        assert!(stats.average_encode_time() > Duration::ZERO);
    }

    #[test]
    fn test_psnr_calculation() {
        let original = vec![100u8; 64];
        let reconstructed = vec![100u8; 64];
        let psnr = QualityMetrics::calculate_psnr(&original, &reconstructed, 8, 8);
        assert!(psnr > 90.0); // Perfect match should have very high PSNR
    }

    #[test]
    fn test_bitrate_distribution() {
        let mut dist = BitrateDistribution::new(100);
        dist.add_frame(1500.0 * 1000.0); // 1500 kbps
        dist.add_frame(1600.0 * 1000.0); // 1600 kbps
        dist.add_frame(1550.0 * 1000.0); // 1550 kbps

        let mode = dist.mode();
        assert!(mode >= 1500 && mode <= 1600);
    }

    #[test]
    fn test_complexity_analyzer() {
        let mut analyzer = ComplexityAnalyzer::new(10);
        analyzer.add_measurement(100.0);
        analyzer.add_measurement(150.0);
        analyzer.add_measurement(200.0);

        assert!(analyzer.average() > 100.0);
        assert!(analyzer.trend() > 0.0); // Increasing complexity
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::default();
        profiler.motion_estimation_time = Duration::from_millis(100);
        profiler.transform_time = Duration::from_millis(50);

        assert_eq!(profiler.total_time(), Duration::from_millis(150));

        let breakdown = profiler.percentage_breakdown();
        assert!((breakdown.motion_estimation - 66.67).abs() < 0.1);
    }
}

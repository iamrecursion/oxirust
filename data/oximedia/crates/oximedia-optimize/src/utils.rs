//! Utility functions and helpers for optimization.

use std::time::{Duration, Instant};

/// Block-level metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockMetrics {
    /// Block width.
    pub width: usize,
    /// Block height.
    pub height: usize,
    /// Sum of Absolute Differences.
    pub sad: u32,
    /// Sum of Squared Errors.
    pub sse: u64,
    /// Variance.
    pub variance: f64,
    /// Mean value.
    pub mean: f64,
}

impl BlockMetrics {
    /// Calculates metrics for a block.
    #[must_use]
    pub fn calculate(pixels: &[u8], width: usize, height: usize) -> Self {
        let mut metrics = Self {
            width,
            height,
            ..Default::default()
        };

        if pixels.is_empty() {
            return metrics;
        }

        // Calculate mean
        metrics.mean = pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / pixels.len() as f64;

        // Calculate variance and SSE
        metrics.variance = pixels
            .iter()
            .map(|&p| {
                let diff = f64::from(p) - metrics.mean;
                diff * diff
            })
            .sum::<f64>()
            / pixels.len() as f64;

        metrics.sse = pixels
            .iter()
            .map(|&p| {
                let diff = i32::from(p) - metrics.mean as i32;
                (diff * diff) as u64
            })
            .sum();

        metrics
    }

    /// Calculates PSNR from SSE.
    #[must_use]
    pub fn psnr(&self) -> f64 {
        if self.sse == 0 {
            return f64::INFINITY;
        }

        let num_pixels = self.width * self.height;
        let mse = self.sse as f64 / num_pixels as f64;
        10.0 * (255.0 * 255.0 / mse).log10()
    }

    /// Checks if block is flat.
    #[must_use]
    pub fn is_flat(&self, threshold: f64) -> bool {
        self.variance < threshold
    }

    /// Checks if block is textured.
    #[must_use]
    pub fn is_textured(&self, threshold: f64) -> bool {
        self.variance > threshold
    }
}

/// Frame-level metrics.
#[derive(Debug, Clone, Default)]
pub struct FrameMetrics {
    /// Frame width.
    pub width: usize,
    /// Frame height.
    pub height: usize,
    /// Total bits used.
    pub total_bits: u64,
    /// QP values used.
    pub qp_values: Vec<u8>,
    /// Average QP.
    pub avg_qp: f64,
    /// PSNR.
    pub psnr: f64,
    /// SSIM.
    pub ssim: f64,
    /// Encoding time.
    pub encoding_time: Duration,
}

impl FrameMetrics {
    /// Creates a new frame metrics.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Adds a QP value.
    pub fn add_qp(&mut self, qp: u8) {
        self.qp_values.push(qp);
        self.avg_qp =
            self.qp_values.iter().map(|&q| f64::from(q)).sum::<f64>() / self.qp_values.len() as f64;
    }

    /// Sets encoding time.
    pub fn set_encoding_time(&mut self, duration: Duration) {
        self.encoding_time = duration;
    }

    /// Calculates bits per pixel.
    #[must_use]
    pub fn bits_per_pixel(&self) -> f64 {
        let num_pixels = (self.width * self.height) as f64;
        if num_pixels > 0.0 {
            self.total_bits as f64 / num_pixels
        } else {
            0.0
        }
    }

    /// Calculates encoding speed in FPS.
    #[must_use]
    pub fn encoding_fps(&self) -> f64 {
        let secs = self.encoding_time.as_secs_f64();
        if secs > 0.0 {
            1.0 / secs
        } else {
            0.0
        }
    }
}

/// Optimization statistics.
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of frames encoded.
    pub frames_encoded: usize,
    /// Total encoding time.
    pub total_time: Duration,
    /// Per-frame metrics.
    pub frame_metrics: Vec<FrameMetrics>,
    /// Total bits used.
    pub total_bits: u64,
    /// Average PSNR.
    pub avg_psnr: f64,
    /// Average SSIM.
    pub avg_ssim: f64,
}

impl OptimizationStats {
    /// Creates new optimization statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds frame metrics.
    pub fn add_frame(&mut self, metrics: FrameMetrics) {
        self.total_bits += metrics.total_bits;
        self.total_time += metrics.encoding_time;
        self.frames_encoded += 1;
        self.frame_metrics.push(metrics);

        // Update averages
        self.calculate_averages();
    }

    fn calculate_averages(&mut self) {
        if self.frames_encoded == 0 {
            return;
        }

        self.avg_psnr =
            self.frame_metrics.iter().map(|m| m.psnr).sum::<f64>() / self.frames_encoded as f64;

        self.avg_ssim =
            self.frame_metrics.iter().map(|m| m.ssim).sum::<f64>() / self.frames_encoded as f64;
    }

    /// Calculates average bitrate in bits per second.
    #[must_use]
    pub fn avg_bitrate(&self, fps: f64) -> f64 {
        if self.frames_encoded == 0 {
            return 0.0;
        }

        (self.total_bits as f64 / self.frames_encoded as f64) * fps
    }

    /// Calculates average encoding speed in FPS.
    #[must_use]
    pub fn avg_fps(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs > 0.0 {
            self.frames_encoded as f64 / secs
        } else {
            0.0
        }
    }

    /// Gets compression ratio.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.frame_metrics.is_empty() {
            return 1.0;
        }

        let first_frame = &self.frame_metrics[0];
        let uncompressed_bits =
            (first_frame.width * first_frame.height * 8 * self.frames_encoded) as f64;

        if self.total_bits > 0 {
            uncompressed_bits / self.total_bits as f64
        } else {
            1.0
        }
    }

    /// Prints summary.
    pub fn print_summary(&self) {
        println!("Optimization Statistics:");
        println!("  Frames: {}", self.frames_encoded);
        println!("  Total bits: {}", self.total_bits);
        println!("  Avg PSNR: {:.2} dB", self.avg_psnr);
        println!("  Avg SSIM: {:.4}", self.avg_ssim);
        println!("  Avg FPS: {:.2}", self.avg_fps());
        println!("  Compression: {:.2}x", self.compression_ratio());
    }
}

/// Timer helper for performance measurement.
#[derive(Debug)]
pub struct Timer {
    start: Instant,
    label: String,
}

impl Timer {
    /// Starts a new timer.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            label: label.into(),
        }
    }

    /// Gets elapsed time.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stops timer and returns elapsed time.
    #[must_use]
    pub fn stop(self) -> Duration {
        let elapsed = self.elapsed();
        println!("{}: {:?}", self.label, elapsed);
        elapsed
    }
}

/// Histogram for distribution analysis.
#[derive(Debug, Clone)]
pub struct Histogram {
    bins: Vec<u32>,
    min_value: f64,
    max_value: f64,
    bin_width: f64,
}

impl Histogram {
    /// Creates a new histogram.
    #[must_use]
    pub fn new(num_bins: usize, min_value: f64, max_value: f64) -> Self {
        let bin_width = (max_value - min_value) / num_bins as f64;
        Self {
            bins: vec![0; num_bins],
            min_value,
            max_value,
            bin_width,
        }
    }

    /// Adds a value to the histogram.
    pub fn add(&mut self, value: f64) {
        if value < self.min_value || value >= self.max_value {
            return;
        }

        let bin = ((value - self.min_value) / self.bin_width) as usize;
        if bin < self.bins.len() {
            self.bins[bin] += 1;
        }
    }

    /// Gets the count for a bin.
    #[must_use]
    pub fn get_bin(&self, index: usize) -> u32 {
        self.bins.get(index).copied().unwrap_or(0)
    }

    /// Gets the total count.
    #[must_use]
    pub fn total_count(&self) -> u32 {
        self.bins.iter().sum()
    }

    /// Calculates the mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        let total: u32 = self.total_count();
        if total == 0 {
            return 0.0;
        }

        let weighted_sum: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let bin_center = self.min_value + (i as f64 + 0.5) * self.bin_width;
                bin_center * f64::from(count)
            })
            .sum();

        weighted_sum / f64::from(total)
    }

    /// Calculates the median.
    #[must_use]
    pub fn median(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            return 0.0;
        }

        let median_count = total / 2;
        let mut cumulative = 0;

        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += count;
            if cumulative >= median_count {
                return self.min_value + (i as f64 + 0.5) * self.bin_width;
            }
        }

        self.max_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_metrics_flat() {
        let pixels = vec![128u8; 64];
        let metrics = BlockMetrics::calculate(&pixels, 8, 8);
        assert_eq!(metrics.mean, 128.0);
        assert_eq!(metrics.variance, 0.0);
        assert!(metrics.is_flat(10.0));
    }

    #[test]
    fn test_block_metrics_varied() {
        let pixels: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let metrics = BlockMetrics::calculate(&pixels, 8, 8);
        assert!(metrics.variance > 0.0);
        assert!(metrics.is_textured(100.0));
    }

    #[test]
    fn test_block_metrics_psnr() {
        let pixels = vec![128u8; 64];
        let metrics = BlockMetrics::calculate(&pixels, 8, 8);
        assert!(metrics.psnr().is_infinite()); // Zero SSE = infinite PSNR
    }

    #[test]
    fn test_frame_metrics() {
        let mut metrics = FrameMetrics::new(1920, 1080);
        metrics.add_qp(26);
        metrics.add_qp(28);
        metrics.add_qp(24);
        assert_eq!(metrics.avg_qp, 26.0);
    }

    #[test]
    fn test_frame_metrics_bpp() {
        let mut metrics = FrameMetrics::new(1920, 1080);
        metrics.total_bits = 1920 * 1080; // 1 bit per pixel
        assert_eq!(metrics.bits_per_pixel(), 1.0);
    }

    #[test]
    fn test_optimization_stats() {
        let mut stats = OptimizationStats::new();
        let mut frame1 = FrameMetrics::new(1920, 1080);
        frame1.psnr = 40.0;
        frame1.ssim = 0.95;
        stats.add_frame(frame1);

        let mut frame2 = FrameMetrics::new(1920, 1080);
        frame2.psnr = 42.0;
        frame2.ssim = 0.96;
        stats.add_frame(frame2);

        assert_eq!(stats.frames_encoded, 2);
        assert_eq!(stats.avg_psnr, 41.0);
        assert_eq!(stats.avg_ssim, 0.955);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_histogram() {
        let mut hist = Histogram::new(10, 0.0, 100.0);
        hist.add(25.0);
        hist.add(35.0);
        hist.add(25.0);

        assert_eq!(hist.total_count(), 3);
        assert_eq!(hist.get_bin(2), 2); // 25.0 falls in bin 2
        assert_eq!(hist.get_bin(3), 1); // 35.0 falls in bin 3
    }

    #[test]
    fn test_histogram_mean() {
        let mut hist = Histogram::new(10, 0.0, 100.0);
        hist.add(20.0);
        hist.add(30.0);
        hist.add(40.0);

        // bin_width = 10.0; 20 → bin 2 (center 25), 30 → bin 3 (center 35), 40 → bin 4 (center 45)
        // mean = (25 + 35 + 45) / 3 = 35.0
        let mean = hist.mean();
        assert!((mean - 35.0).abs() < 1.0); // Approximately 35
    }
}

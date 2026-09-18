//! Phase correlation meter (stereo only).

use serde::{Deserialize, Serialize};

/// Phase correlation metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhaseMetrics {
    /// Phase correlation coefficient (-1.0 to +1.0).
    /// +1.0 = fully in-phase (mono)
    /// 0.0 = no correlation
    /// -1.0 = fully out-of-phase
    pub correlation: f32,

    /// Minimum correlation seen.
    pub min_correlation: f32,

    /// Maximum correlation seen.
    pub max_correlation: f32,

    /// Average correlation.
    pub avg_correlation: f32,

    /// Stereo width (0.0-2.0).
    pub stereo_width: f32,
}

/// Phase correlation meter.
///
/// Measures the phase relationship between stereo channels.
pub struct PhaseCorrelation {
    buffer: Vec<(f32, f32)>,
    buffer_size: usize,
    write_pos: usize,
    min_corr: f32,
    max_corr: f32,
    corr_sum: f32,
    corr_count: usize,
    metrics: PhaseMetrics,
}

impl PhaseCorrelation {
    /// Create a new phase correlation meter.
    #[must_use]
    pub fn new() -> Self {
        let buffer_size = 4096; // ~85ms at 48kHz

        Self {
            buffer: vec![(0.0, 0.0); buffer_size],
            buffer_size,
            write_pos: 0,
            min_corr: 1.0,
            max_corr: -1.0,
            corr_sum: 0.0,
            corr_count: 0,
            metrics: PhaseMetrics {
                correlation: 0.0,
                min_correlation: 0.0,
                max_correlation: 0.0,
                avg_correlation: 0.0,
                stereo_width: 1.0,
            },
        }
    }

    /// Process stereo audio samples (interleaved L/R).
    pub fn process(&mut self, samples: &[f32]) {
        if samples.len() < 2 {
            return;
        }

        // Process stereo pairs
        for chunk in samples.chunks_exact(2) {
            let left = chunk[0];
            let right = chunk[1];

            // Add to buffer
            self.buffer[self.write_pos] = (left, right);
            self.write_pos = (self.write_pos + 1) % self.buffer_size;
        }

        // Calculate correlation
        self.update_metrics();
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &PhaseMetrics {
        &self.metrics
    }

    /// Reset meter state.
    pub fn reset(&mut self) {
        self.buffer.fill((0.0, 0.0));
        self.write_pos = 0;
        self.min_corr = 1.0;
        self.max_corr = -1.0;
        self.corr_sum = 0.0;
        self.corr_count = 0;
        self.metrics = PhaseMetrics::default();
    }

    fn update_metrics(&mut self) {
        let mut sum_left = 0.0f32;
        let mut sum_right = 0.0f32;
        let mut sum_left_sq = 0.0f32;
        let mut sum_right_sq = 0.0f32;
        let mut sum_lr = 0.0f32;

        // Calculate correlation coefficient using Pearson correlation
        for &(left, right) in &self.buffer {
            sum_left += left;
            sum_right += right;
            sum_left_sq += left * left;
            sum_right_sq += right * right;
            sum_lr += left * right;
        }

        let n = self.buffer_size as f32;
        let numerator = n * sum_lr - sum_left * sum_right;
        let denominator = ((n * sum_left_sq - sum_left * sum_left)
            * (n * sum_right_sq - sum_right * sum_right))
            .sqrt();

        let correlation = if denominator > 0.0 {
            (numerator / denominator).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        self.metrics.correlation = correlation;

        // Update statistics
        self.min_corr = self.min_corr.min(correlation);
        self.max_corr = self.max_corr.max(correlation);
        self.corr_sum += correlation;
        self.corr_count += 1;

        self.metrics.min_correlation = self.min_corr;
        self.metrics.max_correlation = self.max_corr;
        self.metrics.avg_correlation = if self.corr_count > 0 {
            self.corr_sum / self.corr_count as f32
        } else {
            0.0
        };

        // Calculate stereo width (0 = mono, 1 = normal stereo, 2 = wide)
        // Based on correlation: width = 1 - correlation
        self.metrics.stereo_width = (1.0 - correlation).clamp(0.0, 2.0);
    }
}

impl Default for PhaseCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_correlation() {
        let mut meter = PhaseCorrelation::new();

        // Mono signal (fully correlated)
        let mut samples = Vec::new();
        for _ in 0..1000 {
            samples.push(0.5f32);
            samples.push(0.5f32);
        }
        meter.process(&samples);

        let metrics = meter.metrics();
        assert!(metrics.correlation > 0.9); // Should be close to 1.0
    }

    #[test]
    fn test_phase_out_of_phase() {
        let mut meter = PhaseCorrelation::new();

        // Out-of-phase signal
        let mut samples = Vec::new();
        for _ in 0..1000 {
            samples.push(0.5f32);
            samples.push(-0.5f32);
        }
        meter.process(&samples);

        let metrics = meter.metrics();
        assert!(metrics.correlation < -0.9); // Should be close to -1.0
    }

    #[test]
    fn test_phase_reset() {
        let mut meter = PhaseCorrelation::new();

        let samples = vec![0.5f32; 1000];
        meter.process(&samples);

        meter.reset();

        let metrics = meter.metrics();
        assert_eq!(metrics.correlation, 0.0);
    }
}

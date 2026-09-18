//! Deflicker processing for video and image sequences.
//!
//! Provides luminance variance analysis, temporal smoothing, and flicker frequency detection.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Configuration for the deflicker processor.
#[derive(Debug, Clone)]
pub struct DeflickerConfig {
    /// Window size (in frames) for temporal analysis.
    pub window_size: usize,
    /// Strength of correction (0.0 = none, 1.0 = full).
    pub strength: f32,
    /// Minimum flicker frequency in Hz to consider (e.g. 2.0).
    pub min_freq_hz: f32,
    /// Maximum flicker frequency in Hz to consider (e.g. 25.0).
    pub max_freq_hz: f32,
    /// Whether to apply adaptive correction based on local variance.
    pub adaptive: bool,
}

impl Default for DeflickerConfig {
    fn default() -> Self {
        Self {
            window_size: 5,
            strength: 0.8,
            min_freq_hz: 2.0,
            max_freq_hz: 25.0,
            adaptive: true,
        }
    }
}

/// Luminance statistics for a single frame.
#[derive(Debug, Clone, Copy)]
pub struct LuminanceStats {
    /// Mean luminance (0.0–1.0).
    pub mean: f32,
    /// Variance of luminance.
    pub variance: f32,
    /// Minimum luminance value.
    pub min: f32,
    /// Maximum luminance value.
    pub max: f32,
}

impl LuminanceStats {
    /// Compute luminance statistics from a slice of pixel values (0.0–1.0).
    pub fn compute(pixels: &[f32]) -> Self {
        if pixels.is_empty() {
            return Self {
                mean: 0.0,
                variance: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }
        let n = pixels.len() as f32;
        let mean = pixels.iter().sum::<f32>() / n;
        let variance = pixels.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let min = pixels.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = pixels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        Self {
            mean,
            variance,
            min,
            max,
        }
    }
}

/// Deflicker processor using temporal smoothing.
#[derive(Debug, Clone)]
pub struct Deflickerer {
    config: DeflickerConfig,
    /// Ring buffer of luminance means for the sliding window.
    lum_history: Vec<f32>,
    history_pos: usize,
    history_count: usize,
}

impl Deflickerer {
    /// Create a new deflickerer with the given config.
    pub fn new(config: DeflickerConfig) -> Self {
        let window = config.window_size.max(1);
        Self {
            config,
            lum_history: vec![0.0; window],
            history_pos: 0,
            history_count: 0,
        }
    }

    /// Process a frame of pixels and return the deflickered result.
    ///
    /// Pixels are expected as a flat array of luminance values (0.0–1.0).
    pub fn process_frame(&mut self, pixels: &[f32]) -> Vec<f32> {
        let stats = LuminanceStats::compute(pixels);

        // Push current mean into history
        self.lum_history[self.history_pos] = stats.mean;
        self.history_pos = (self.history_pos + 1) % self.lum_history.len();
        self.history_count = (self.history_count + 1).min(self.lum_history.len());

        // Compute target mean as average of window
        let target_mean = self.window_mean();

        // Avoid division by zero
        let scale = if stats.mean.abs() > 1e-6 {
            target_mean / stats.mean
        } else {
            1.0
        };

        // Blend between original and corrected
        let alpha = self.config.strength;
        let effective_scale = 1.0 + alpha * (scale - 1.0);

        if self.config.adaptive {
            let local_strength = (stats.variance * 10.0).min(1.0) * alpha;
            let adaptive_scale = 1.0 + local_strength * (scale - 1.0);
            pixels
                .iter()
                .map(|&p| (p * adaptive_scale).clamp(0.0, 1.0))
                .collect()
        } else {
            pixels
                .iter()
                .map(|&p| (p * effective_scale).clamp(0.0, 1.0))
                .collect()
        }
    }

    /// Compute the windowed mean luminance.
    fn window_mean(&self) -> f32 {
        if self.history_count == 0 {
            return 0.0;
        }
        let sum: f32 = self.lum_history.iter().take(self.history_count).sum();
        sum / self.history_count as f32
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        for v in self.lum_history.iter_mut() {
            *v = 0.0;
        }
        self.history_pos = 0;
        self.history_count = 0;
    }

    /// Get the current config.
    pub fn config(&self) -> &DeflickerConfig {
        &self.config
    }
}

/// Flicker frequency detector using DFT of luminance history.
#[derive(Debug, Clone)]
pub struct FlickerDetector {
    fps: f32,
    history: Vec<f32>,
    capacity: usize,
}

impl FlickerDetector {
    /// Create a new flicker detector.
    ///
    /// `fps` is the frame rate, `history_frames` is how many frames to accumulate.
    pub fn new(fps: f32, history_frames: usize) -> Self {
        Self {
            fps,
            history: Vec::with_capacity(history_frames),
            capacity: history_frames,
        }
    }

    /// Feed a luminance mean for the current frame.
    pub fn feed(&mut self, mean_luminance: f32) {
        if self.history.len() < self.capacity {
            self.history.push(mean_luminance);
        } else {
            // Shift left and append
            self.history.rotate_left(1);
            if let Some(last) = self.history.last_mut() {
                *last = mean_luminance;
            }
        }
    }

    /// Detect the dominant flicker frequency in Hz using DFT.
    ///
    /// Returns `None` if not enough history or no significant frequency found.
    pub fn detect_frequency(&self) -> Option<f32> {
        let n = self.history.len();
        if n < 4 {
            return None;
        }

        // Compute mean-subtracted signal
        let mean = self.history.iter().sum::<f32>() / n as f32;
        let signal: Vec<f32> = self.history.iter().map(|&x| x - mean).collect();

        // Simple DFT magnitude for each frequency bin
        let mut max_mag = 0.0f32;
        let mut max_bin = 0usize;

        for k in 1..n / 2 {
            let freq = k as f32 * self.fps / n as f32;
            if freq < 0.5 {
                continue;
            }

            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (i, &s) in signal.iter().enumerate() {
                let angle = 2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                re += s * angle.cos();
                im -= s * angle.sin();
            }
            let mag = (re * re + im * im).sqrt();
            if mag > max_mag {
                max_mag = mag;
                max_bin = k;
            }
        }

        if max_bin == 0 || max_mag < 1e-6 {
            return None;
        }

        Some(max_bin as f32 * self.fps / n as f32)
    }

    /// Get the number of history frames accumulated.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

/// Temporal smoother using a weighted average of consecutive frames.
#[derive(Debug, Clone)]
pub struct TemporalSmoother {
    weights: Vec<f32>,
    frame_buffer: Vec<Vec<f32>>,
    write_pos: usize,
    filled: bool,
}

impl TemporalSmoother {
    /// Create a Gaussian-weighted temporal smoother.
    ///
    /// `radius` is the number of frames on each side of center.
    pub fn new(radius: usize) -> Self {
        let len = 2 * radius + 1;
        let sigma = radius as f32 / 2.0 + 0.5;
        let mut weights: Vec<f32> = (0..len)
            .map(|i| {
                let x = i as f32 - radius as f32;
                (-(x * x) / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let sum: f32 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= sum;
        }
        Self {
            weights,
            frame_buffer: Vec::new(),
            write_pos: 0,
            filled: false,
        }
    }

    /// Feed a new frame. Returns `Some` when the buffer is full and a smoothed output is ready.
    pub fn feed(&mut self, frame: Vec<f32>) -> Option<Vec<f32>> {
        let len = self.weights.len();
        if self.frame_buffer.is_empty() {
            self.frame_buffer = vec![Vec::new(); len];
        }
        self.frame_buffer[self.write_pos] = frame;
        self.write_pos = (self.write_pos + 1) % len;
        if !self.filled && self.frame_buffer.iter().all(|f| !f.is_empty()) {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }

        // Compute weighted average
        let n_pixels = self.frame_buffer[0].len();
        let mut output = vec![0.0f32; n_pixels];
        for (fi, w) in self.weights.iter().enumerate() {
            let buf_idx = (self.write_pos + fi) % len;
            for (o, &p) in output.iter_mut().zip(self.frame_buffer[buf_idx].iter()) {
                *o += p * w;
            }
        }
        Some(output)
    }

    /// Get the window size.
    pub fn window_size(&self) -> usize {
        self.weights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luminance_stats_empty() {
        let s = LuminanceStats::compute(&[]);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.variance, 0.0);
    }

    #[test]
    fn test_luminance_stats_uniform() {
        let pixels = vec![0.5f32; 100];
        let s = LuminanceStats::compute(&pixels);
        assert!((s.mean - 0.5).abs() < 1e-5);
        assert!(s.variance < 1e-10);
        assert!((s.min - 0.5).abs() < 1e-5);
        assert!((s.max - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_luminance_stats_range() {
        let pixels = vec![0.0f32, 0.5, 1.0];
        let s = LuminanceStats::compute(&pixels);
        assert!((s.min - 0.0).abs() < 1e-5);
        assert!((s.max - 1.0).abs() < 1e-5);
        assert!((s.mean - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_deflickerer_passthrough_no_flicker() {
        let config = DeflickerConfig {
            adaptive: false,
            ..Default::default()
        };
        let mut deflickerer = Deflickerer::new(config);
        // Process same frame multiple times - should converge to near-original
        let pixels = vec![0.5f32; 100];
        for _ in 0..10 {
            deflickerer.process_frame(&pixels);
        }
        let result = deflickerer.process_frame(&pixels);
        // With stable mean, scale should be ~1.0
        for &v in &result {
            assert!((v - 0.5).abs() < 0.05);
        }
    }

    #[test]
    fn test_deflickerer_clamps_output() {
        let config = DeflickerConfig::default();
        let mut deflickerer = Deflickerer::new(config);
        // Very bright frame then dark frame
        let bright = vec![1.0f32; 50];
        deflickerer.process_frame(&bright);
        let dark = vec![0.1f32; 50];
        let result = deflickerer.process_frame(&dark);
        for &v in &result {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_deflickerer_reset() {
        let config = DeflickerConfig::default();
        let mut deflickerer = Deflickerer::new(config);
        let pixels = vec![0.5f32; 100];
        deflickerer.process_frame(&pixels);
        deflickerer.reset();
        assert_eq!(deflickerer.history_count, 0);
    }

    #[test]
    fn test_deflickerer_zero_mean_frame() {
        let config = DeflickerConfig {
            adaptive: false,
            strength: 1.0,
            ..Default::default()
        };
        let mut deflickerer = Deflickerer::new(config);
        let zeros = vec![0.0f32; 50];
        let result = deflickerer.process_frame(&zeros);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_flicker_detector_not_enough_history() {
        let mut detector = FlickerDetector::new(25.0, 64);
        detector.feed(0.5);
        detector.feed(0.6);
        assert!(detector.detect_frequency().is_none());
    }

    #[test]
    fn test_flicker_detector_history_len() {
        let mut detector = FlickerDetector::new(25.0, 32);
        for i in 0..32 {
            detector.feed(i as f32 / 32.0);
        }
        assert_eq!(detector.history_len(), 32);
    }

    #[test]
    fn test_flicker_detector_reset() {
        let mut detector = FlickerDetector::new(25.0, 32);
        detector.feed(0.5);
        detector.reset();
        assert_eq!(detector.history_len(), 0);
    }

    #[test]
    fn test_flicker_detector_constant_signal() {
        let mut detector = FlickerDetector::new(25.0, 64);
        for _ in 0..64 {
            detector.feed(0.5); // constant = no flicker
        }
        // A constant signal has no frequency after mean subtraction
        assert!(detector.detect_frequency().is_none());
    }

    #[test]
    fn test_temporal_smoother_window_size() {
        let smoother = TemporalSmoother::new(2);
        assert_eq!(smoother.window_size(), 5);
    }

    #[test]
    fn test_temporal_smoother_needs_full_buffer() {
        let mut smoother = TemporalSmoother::new(1);
        let frame = vec![0.5f32; 10];
        let r1 = smoother.feed(frame.clone());
        assert!(r1.is_none());
        let r2 = smoother.feed(frame.clone());
        assert!(r2.is_none());
        let r3 = smoother.feed(frame.clone());
        assert!(r3.is_some());
    }

    #[test]
    fn test_temporal_smoother_output_same_frames() {
        let mut smoother = TemporalSmoother::new(1);
        let frame = vec![0.5f32; 10];
        for _ in 0..3 {
            smoother.feed(frame.clone());
        }
        let result = smoother
            .feed(frame.clone())
            .expect("should succeed in test");
        for &v in &result {
            assert!((v - 0.5).abs() < 1e-4);
        }
    }
}

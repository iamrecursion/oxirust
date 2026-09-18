#![allow(dead_code)]
//! Noise floor detection and measurement for audio signals.
//!
//! Detects the ambient noise floor of audio recordings by analyzing the quietest
//! segments, computing statistics over sliding windows, and reporting the result
//! in dBFS. Useful for dynamic range analysis, gating threshold selection,
//! and broadcast quality control.

/// Configuration for noise floor detection.
#[derive(Clone, Debug)]
pub struct NoiseFloorConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of audio channels.
    pub channels: usize,
    /// Analysis window size in samples per channel.
    pub window_size: usize,
    /// Percentile for noise floor (e.g. 10.0 means the 10th percentile of RMS values).
    pub percentile: f64,
}

impl NoiseFloorConfig {
    /// Create a default configuration.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        let window_size = (sample_rate * 0.05) as usize; // 50ms windows
        Self {
            sample_rate,
            channels,
            window_size: window_size.max(1),
            percentile: 10.0,
        }
    }

    /// Set the analysis window duration in seconds.
    pub fn with_window_duration(mut self, seconds: f64) -> Self {
        self.window_size = ((self.sample_rate * seconds) as usize).max(1);
        self
    }

    /// Set the percentile for noise floor estimation.
    pub fn with_percentile(mut self, p: f64) -> Self {
        self.percentile = p.clamp(1.0, 50.0);
        self
    }
}

/// Result of noise floor analysis.
#[derive(Clone, Debug)]
pub struct NoiseFloorResult {
    /// Estimated noise floor in dBFS.
    pub noise_floor_dbfs: f64,
    /// Noise floor linear RMS value.
    pub noise_floor_rms: f64,
    /// Number of analysis windows evaluated.
    pub window_count: usize,
    /// Minimum RMS value observed (dBFS).
    pub min_rms_dbfs: f64,
    /// Maximum RMS value observed (dBFS).
    pub max_rms_dbfs: f64,
    /// Estimated signal-to-noise ratio in dB (peak signal vs noise floor).
    pub estimated_snr_db: f64,
}

/// Noise floor detector that accumulates RMS windows.
#[derive(Clone, Debug)]
pub struct NoiseFloorDetector {
    /// Configuration.
    config: NoiseFloorConfig,
    /// Collected RMS values per window (linear).
    rms_values: Vec<f64>,
    /// Peak signal level observed (linear).
    peak_level: f64,
}

impl NoiseFloorDetector {
    /// Create a new noise floor detector.
    pub fn new(config: NoiseFloorConfig) -> Self {
        Self {
            config,
            rms_values: Vec::new(),
            peak_level: 0.0,
        }
    }

    /// Create with default config for the given sample rate and channels.
    pub fn with_defaults(sample_rate: f64, channels: usize) -> Self {
        Self::new(NoiseFloorConfig::new(sample_rate, channels))
    }

    /// Process interleaved audio samples.
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let channels = self.config.channels.max(1);
        let window = self.config.window_size;
        let frame_count = samples.len() / channels;

        if frame_count == 0 {
            return;
        }

        // Track peak
        for &s in samples {
            let abs = s.abs();
            if abs > self.peak_level {
                self.peak_level = abs;
            }
        }

        // Process in non-overlapping windows
        let mut offset = 0;
        while offset + window <= frame_count {
            let mut sum_sq = 0.0;
            let mut count = 0usize;
            for f in offset..(offset + window) {
                for ch in 0..channels {
                    let idx = f * channels + ch;
                    if idx < samples.len() {
                        let s = samples[idx];
                        sum_sq += s * s;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                let rms = (sum_sq / count as f64).sqrt();
                self.rms_values.push(rms);
            }
            offset += window;
        }
    }

    /// Process f32 interleaved samples.
    pub fn process_interleaved_f32(&mut self, samples: &[f32]) {
        let f64_samples: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
        self.process_interleaved(&f64_samples);
    }

    /// Compute the noise floor result.
    pub fn result(&self) -> NoiseFloorResult {
        if self.rms_values.is_empty() {
            return NoiseFloorResult {
                noise_floor_dbfs: f64::NEG_INFINITY,
                noise_floor_rms: 0.0,
                window_count: 0,
                min_rms_dbfs: f64::NEG_INFINITY,
                max_rms_dbfs: f64::NEG_INFINITY,
                estimated_snr_db: 0.0,
            };
        }

        let mut sorted = self.rms_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Filter out silence (zero RMS)
        let non_zero: Vec<f64> = sorted.iter().copied().filter(|&v| v > 1e-12).collect();
        if non_zero.is_empty() {
            return NoiseFloorResult {
                noise_floor_dbfs: f64::NEG_INFINITY,
                noise_floor_rms: 0.0,
                window_count: self.rms_values.len(),
                min_rms_dbfs: f64::NEG_INFINITY,
                max_rms_dbfs: f64::NEG_INFINITY,
                estimated_snr_db: 0.0,
            };
        }

        let percentile_idx =
            ((self.config.percentile / 100.0) * non_zero.len() as f64).ceil() as usize;
        let percentile_idx = percentile_idx.min(non_zero.len()).max(1) - 1;
        let noise_rms = non_zero[percentile_idx];
        let noise_dbfs = linear_to_dbfs(noise_rms);

        let min_rms = non_zero.first().copied().unwrap_or(0.0);
        let max_rms = non_zero.last().copied().unwrap_or(0.0);

        let peak_dbfs = linear_to_dbfs(self.peak_level);
        let snr = peak_dbfs - noise_dbfs;

        NoiseFloorResult {
            noise_floor_dbfs: noise_dbfs,
            noise_floor_rms: noise_rms,
            window_count: self.rms_values.len(),
            min_rms_dbfs: linear_to_dbfs(min_rms),
            max_rms_dbfs: linear_to_dbfs(max_rms),
            estimated_snr_db: if snr.is_finite() { snr } else { 0.0 },
        }
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.rms_values.clear();
        self.peak_level = 0.0;
    }

    /// Get the number of windows analyzed so far.
    pub fn window_count(&self) -> usize {
        self.rms_values.len()
    }

    /// Get the peak level observed (linear).
    pub fn peak_level(&self) -> f64 {
        self.peak_level
    }
}

/// Convert a linear amplitude to dBFS.
fn linear_to_dbfs(linear: f64) -> f64 {
    if linear <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Convert dBFS to linear amplitude.
fn dbfs_to_linear(dbfs: f64) -> f64 {
    10.0_f64.powf(dbfs / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sine(freq: f64, sample_rate: f64, duration: f64, amplitude: f64) -> Vec<f64> {
        let n = (sample_rate * duration) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                amplitude * (2.0 * std::f64::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_config_new() {
        let cfg = NoiseFloorConfig::new(48000.0, 2);
        assert_eq!(cfg.channels, 2);
        assert!((cfg.sample_rate - 48000.0).abs() < f64::EPSILON);
        assert!((cfg.percentile - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_with_window() {
        let cfg = NoiseFloorConfig::new(48000.0, 1).with_window_duration(0.1);
        assert_eq!(cfg.window_size, 4800);
    }

    #[test]
    fn test_config_with_percentile() {
        let cfg = NoiseFloorConfig::new(48000.0, 1).with_percentile(5.0);
        assert!((cfg.percentile - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_to_dbfs() {
        assert!((linear_to_dbfs(1.0)).abs() < 1e-10);
        assert!((linear_to_dbfs(0.5) - (-6.0206)).abs() < 0.001);
        assert!(linear_to_dbfs(0.0).is_infinite());
    }

    #[test]
    fn test_dbfs_to_linear() {
        assert!((dbfs_to_linear(0.0) - 1.0).abs() < 1e-10);
        assert!((dbfs_to_linear(-6.0206) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_empty_detector() {
        let det = NoiseFloorDetector::with_defaults(48000.0, 1);
        let r = det.result();
        assert_eq!(r.window_count, 0);
        assert!(r.noise_floor_dbfs.is_infinite());
    }

    #[test]
    fn test_silence_detection() {
        let mut det = NoiseFloorDetector::with_defaults(48000.0, 1);
        let silence = vec![0.0; 48000];
        det.process_interleaved(&silence);
        let r = det.result();
        // All windows should be zero-RMS, so noise floor is -inf
        assert!(r.noise_floor_dbfs.is_infinite());
    }

    #[test]
    fn test_constant_signal() {
        let cfg = NoiseFloorConfig::new(1000.0, 1).with_window_duration(0.1);
        let mut det = NoiseFloorDetector::new(cfg);
        // Constant amplitude = 0.1 for 1 second (mono)
        let samples = vec![0.1_f64; 1000];
        det.process_interleaved(&samples);
        let r = det.result();
        // RMS of constant 0.1 = 0.1, dBFS ~ -20
        assert!((r.noise_floor_dbfs - (-20.0)).abs() < 0.5);
    }

    #[test]
    fn test_sine_noise_floor() {
        let cfg = NoiseFloorConfig::new(48000.0, 1).with_window_duration(0.05);
        let mut det = NoiseFloorDetector::new(cfg);
        let sine = generate_sine(440.0, 48000.0, 1.0, 0.5);
        det.process_interleaved(&sine);
        let r = det.result();
        // Sine RMS = amplitude / sqrt(2) ~ 0.354, dBFS ~ -9
        assert!(r.noise_floor_dbfs > -12.0);
        assert!(r.noise_floor_dbfs < -6.0);
        assert!(r.window_count > 0);
    }

    #[test]
    fn test_peak_tracking() {
        let mut det = NoiseFloorDetector::with_defaults(1000.0, 1);
        let mut samples = vec![0.1; 500];
        samples[250] = 0.9;
        det.process_interleaved(&samples);
        assert!((det.peak_level() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_snr_estimation() {
        let cfg = NoiseFloorConfig::new(1000.0, 1).with_window_duration(0.05);
        let mut det = NoiseFloorDetector::new(cfg);
        let samples = vec![0.5; 1000];
        det.process_interleaved(&samples);
        let r = det.result();
        // With constant signal, noise floor = signal level, so SNR ~ 0
        assert!(r.estimated_snr_db.abs() < 1.0);
    }

    #[test]
    fn test_reset() {
        let mut det = NoiseFloorDetector::with_defaults(48000.0, 1);
        det.process_interleaved(&vec![0.5; 48000]);
        assert!(det.window_count() > 0);
        det.reset();
        assert_eq!(det.window_count(), 0);
        assert!((det.peak_level()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_f32_processing() {
        let cfg = NoiseFloorConfig::new(1000.0, 1).with_window_duration(0.1);
        let mut det = NoiseFloorDetector::new(cfg);
        let samples: Vec<f32> = vec![0.25; 1000];
        det.process_interleaved_f32(&samples);
        let r = det.result();
        assert!(r.noise_floor_dbfs.is_finite());
        assert!(r.window_count > 0);
    }

    #[test]
    fn test_stereo_noise_floor() {
        let cfg = NoiseFloorConfig::new(1000.0, 2).with_window_duration(0.1);
        let mut det = NoiseFloorDetector::new(cfg);
        // Interleaved stereo: L=0.1, R=0.1
        let samples: Vec<f64> = vec![0.1; 2000];
        det.process_interleaved(&samples);
        let r = det.result();
        assert!(r.noise_floor_dbfs.is_finite());
    }
}

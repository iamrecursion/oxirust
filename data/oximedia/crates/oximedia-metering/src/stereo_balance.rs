#![allow(dead_code)]
//! Stereo balance and panning analysis for audio metering.
//!
//! Measures the left-right balance of stereo audio by computing per-channel
//! RMS levels, balance ratios, and panning position. Useful for broadcast
//! compliance and mix validation.

/// Stereo balance measurement result.
#[derive(Clone, Debug)]
pub struct BalanceResult {
    /// Left channel RMS level (linear).
    pub left_rms: f64,
    /// Right channel RMS level (linear).
    pub right_rms: f64,
    /// Left channel level in dBFS.
    pub left_dbfs: f64,
    /// Right channel level in dBFS.
    pub right_dbfs: f64,
    /// Balance position from -1.0 (full left) to 1.0 (full right), 0.0 = center.
    pub balance: f64,
    /// Level difference in dB (positive means right is louder).
    pub level_diff_db: f64,
    /// Number of frames analyzed.
    pub frames_analyzed: usize,
}

impl BalanceResult {
    /// Check if the signal is centered within a given tolerance in dB.
    pub fn is_centered(&self, tolerance_db: f64) -> bool {
        self.level_diff_db.abs() <= tolerance_db
    }

    /// Describe the panning position as a human-readable string.
    pub fn describe(&self) -> &str {
        if self.balance.abs() < 0.05 {
            "center"
        } else if self.balance < -0.5 {
            "hard left"
        } else if self.balance < -0.15 {
            "left"
        } else if self.balance < -0.05 {
            "slight left"
        } else if self.balance > 0.5 {
            "hard right"
        } else if self.balance > 0.15 {
            "right"
        } else {
            "slight right"
        }
    }
}

/// Configuration for the stereo balance analyzer.
#[derive(Clone, Debug)]
pub struct StereoBalanceConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Integration time in seconds for RMS calculation.
    pub integration_time: f64,
}

impl StereoBalanceConfig {
    /// Create a new config.
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            integration_time: 0.4, // 400ms default
        }
    }

    /// Set integration time in seconds.
    pub fn with_integration_time(mut self, seconds: f64) -> Self {
        self.integration_time = seconds.max(0.01);
        self
    }
}

/// Stereo balance analyzer that processes interleaved stereo audio.
#[derive(Clone, Debug)]
pub struct StereoBalanceAnalyzer {
    /// Configuration.
    config: StereoBalanceConfig,
    /// Accumulated sum of squares for left channel.
    left_sum_sq: f64,
    /// Accumulated sum of squares for right channel.
    right_sum_sq: f64,
    /// Number of frames processed.
    frame_count: usize,
    /// Rolling window left samples (squared).
    left_window: Vec<f64>,
    /// Rolling window right samples (squared).
    right_window: Vec<f64>,
    /// Window write position.
    write_pos: usize,
    /// Window size in frames.
    window_size: usize,
    /// Whether the window has been filled at least once.
    window_filled: bool,
}

impl StereoBalanceAnalyzer {
    /// Create a new analyzer.
    pub fn new(config: StereoBalanceConfig) -> Self {
        let window_size = (config.sample_rate * config.integration_time) as usize;
        let window_size = window_size.max(1);
        Self {
            config,
            left_sum_sq: 0.0,
            right_sum_sq: 0.0,
            frame_count: 0,
            left_window: vec![0.0; window_size],
            right_window: vec![0.0; window_size],
            write_pos: 0,
            window_size,
            window_filled: false,
        }
    }

    /// Create with default config.
    pub fn with_defaults(sample_rate: f64) -> Self {
        Self::new(StereoBalanceConfig::new(sample_rate))
    }

    /// Process interleaved stereo samples (L, R, L, R, ...).
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let frames = samples.len() / 2;
        for f in 0..frames {
            let l = samples[f * 2];
            let r = samples[f * 2 + 1];
            let l_sq = l * l;
            let r_sq = r * r;

            // Update running sum
            self.left_sum_sq += l_sq;
            self.right_sum_sq += r_sq;

            // Update rolling window
            self.left_window[self.write_pos] = l_sq;
            self.right_window[self.write_pos] = r_sq;
            self.write_pos += 1;
            if self.write_pos >= self.window_size {
                self.write_pos = 0;
                self.window_filled = true;
            }

            self.frame_count += 1;
        }
    }

    /// Process f32 interleaved stereo samples.
    pub fn process_interleaved_f32(&mut self, samples: &[f32]) {
        let f64_samples: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
        self.process_interleaved(&f64_samples);
    }

    /// Get the short-term (windowed) balance result.
    pub fn short_term_result(&self) -> BalanceResult {
        let count = if self.window_filled {
            self.window_size
        } else {
            self.write_pos
        };

        if count == 0 {
            return empty_result();
        }

        let l_sum: f64 = if self.window_filled {
            self.left_window.iter().sum()
        } else {
            self.left_window[..count].iter().sum()
        };
        let r_sum: f64 = if self.window_filled {
            self.right_window.iter().sum()
        } else {
            self.right_window[..count].iter().sum()
        };

        compute_result(l_sum, r_sum, count)
    }

    /// Get the integrated (full duration) balance result.
    pub fn integrated_result(&self) -> BalanceResult {
        if self.frame_count == 0 {
            return empty_result();
        }
        compute_result(self.left_sum_sq, self.right_sum_sq, self.frame_count)
    }

    /// Reset the analyzer.
    pub fn reset(&mut self) {
        self.left_sum_sq = 0.0;
        self.right_sum_sq = 0.0;
        self.frame_count = 0;
        self.left_window.fill(0.0);
        self.right_window.fill(0.0);
        self.write_pos = 0;
        self.window_filled = false;
    }

    /// Get the number of frames processed.
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }
}

/// Compute a balance result from sum-of-squares and count.
#[allow(clippy::cast_precision_loss)]
fn compute_result(l_sum_sq: f64, r_sum_sq: f64, count: usize) -> BalanceResult {
    let n = count as f64;
    let l_rms = (l_sum_sq / n).sqrt();
    let r_rms = (r_sum_sq / n).sqrt();
    let l_dbfs = linear_to_dbfs(l_rms);
    let r_dbfs = linear_to_dbfs(r_rms);
    let level_diff = r_dbfs - l_dbfs;

    // Balance: -1 full left, +1 full right, 0 center
    let total = l_rms + r_rms;
    let balance = if total > 1e-12 {
        (r_rms - l_rms) / total
    } else {
        0.0
    };

    BalanceResult {
        left_rms: l_rms,
        right_rms: r_rms,
        left_dbfs: l_dbfs,
        right_dbfs: r_dbfs,
        balance,
        level_diff_db: if level_diff.is_finite() {
            level_diff
        } else {
            0.0
        },
        frames_analyzed: count,
    }
}

/// Return an empty result for no data.
fn empty_result() -> BalanceResult {
    BalanceResult {
        left_rms: 0.0,
        right_rms: 0.0,
        left_dbfs: f64::NEG_INFINITY,
        right_dbfs: f64::NEG_INFINITY,
        balance: 0.0,
        level_diff_db: 0.0,
        frames_analyzed: 0,
    }
}

/// Convert linear amplitude to dBFS.
fn linear_to_dbfs(linear: f64) -> f64 {
    if linear <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let cfg = StereoBalanceConfig::new(48000.0);
        assert!((cfg.sample_rate - 48000.0).abs() < f64::EPSILON);
        assert!((cfg.integration_time - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_with_integration() {
        let cfg = StereoBalanceConfig::new(48000.0).with_integration_time(1.0);
        assert!((cfg.integration_time - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_analyzer() {
        let a = StereoBalanceAnalyzer::with_defaults(48000.0);
        let r = a.integrated_result();
        assert_eq!(r.frames_analyzed, 0);
        assert!((r.balance).abs() < f64::EPSILON);
    }

    #[test]
    fn test_centered_signal() {
        let mut a = StereoBalanceAnalyzer::with_defaults(1000.0);
        // L and R at equal level
        let samples: Vec<f64> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.5 } else { 0.5 })
            .collect();
        a.process_interleaved(&samples);
        let r = a.integrated_result();
        assert!(r.balance.abs() < 0.01);
        assert!(r.is_centered(1.0));
        assert_eq!(r.describe(), "center");
    }

    #[test]
    fn test_left_heavy() {
        let mut a = StereoBalanceAnalyzer::with_defaults(1000.0);
        // L=0.8, R=0.2
        let samples: Vec<f64> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.8 } else { 0.2 })
            .collect();
        a.process_interleaved(&samples);
        let r = a.integrated_result();
        assert!(r.balance < -0.3);
        assert!(!r.is_centered(1.0));
    }

    #[test]
    fn test_right_heavy() {
        let mut a = StereoBalanceAnalyzer::with_defaults(1000.0);
        // L=0.1, R=0.9
        let samples: Vec<f64> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.1 } else { 0.9 })
            .collect();
        a.process_interleaved(&samples);
        let r = a.integrated_result();
        assert!(r.balance > 0.3);
    }

    #[test]
    fn test_hard_left() {
        let mut a = StereoBalanceAnalyzer::with_defaults(1000.0);
        // L=0.5, R=0.0 (silence right)
        let samples: Vec<f64> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.5 } else { 0.0 })
            .collect();
        a.process_interleaved(&samples);
        let r = a.integrated_result();
        assert!(r.balance < -0.9);
        assert_eq!(r.describe(), "hard left");
    }

    #[test]
    fn test_hard_right() {
        let mut a = StereoBalanceAnalyzer::with_defaults(1000.0);
        let samples: Vec<f64> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.0 } else { 0.5 })
            .collect();
        a.process_interleaved(&samples);
        let r = a.integrated_result();
        assert!(r.balance > 0.9);
        assert_eq!(r.describe(), "hard right");
    }

    #[test]
    fn test_level_diff_db() {
        let mut a = StereoBalanceAnalyzer::with_defaults(1000.0);
        // L=0.5, R=0.5 → diff = 0
        let samples: Vec<f64> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.5 } else { 0.5 })
            .collect();
        a.process_interleaved(&samples);
        let r = a.integrated_result();
        assert!(r.level_diff_db.abs() < 0.1);
    }

    #[test]
    fn test_short_term_result() {
        let cfg = StereoBalanceConfig::new(1000.0).with_integration_time(0.5);
        let mut a = StereoBalanceAnalyzer::new(cfg);
        let samples: Vec<f64> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.3 } else { 0.3 })
            .collect();
        a.process_interleaved(&samples);
        let r = a.short_term_result();
        assert!(r.balance.abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let mut a = StereoBalanceAnalyzer::with_defaults(1000.0);
        let samples: Vec<f64> = vec![0.5; 2000];
        a.process_interleaved(&samples);
        assert!(a.frame_count() > 0);
        a.reset();
        assert_eq!(a.frame_count(), 0);
    }

    #[test]
    fn test_f32_processing() {
        let mut a = StereoBalanceAnalyzer::with_defaults(1000.0);
        let samples: Vec<f32> = vec![0.4; 2000];
        a.process_interleaved_f32(&samples);
        let r = a.integrated_result();
        assert!(r.frames_analyzed > 0);
        assert!(r.balance.abs() < 0.01);
    }

    #[test]
    fn test_describe_slight() {
        let r = BalanceResult {
            left_rms: 0.48,
            right_rms: 0.52,
            left_dbfs: -6.0,
            right_dbfs: -5.7,
            balance: 0.08,
            level_diff_db: 0.3,
            frames_analyzed: 1000,
        };
        assert_eq!(r.describe(), "slight right");
    }

    #[test]
    fn test_linear_to_dbfs() {
        assert!((linear_to_dbfs(1.0)).abs() < 1e-10);
        assert!(linear_to_dbfs(0.0).is_infinite());
    }
}

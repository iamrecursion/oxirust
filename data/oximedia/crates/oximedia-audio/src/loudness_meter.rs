//! EBU R128 / ITU-R BS.1770 loudness measurement utilities.
#![allow(dead_code)]

/// Integrated loudness result (LUFS / LKFS).
#[derive(Debug, Clone, Copy)]
pub struct IntegratedLoudness {
    /// LUFS value (negative, e.g. -23.0).
    pub lufs: f64,
    /// Target level for the spec being checked against.
    pub target_lufs: f64,
    /// Permitted deviation (±) in LUFS.
    pub tolerance: f64,
}

impl IntegratedLoudness {
    /// Create a new measurement.
    pub fn new(lufs: f64, target_lufs: f64, tolerance: f64) -> Self {
        Self {
            lufs,
            target_lufs,
            tolerance,
        }
    }

    /// Returns `true` if the measured level is within the spec tolerance.
    pub fn is_within_spec(&self) -> bool {
        (self.lufs - self.target_lufs).abs() <= self.tolerance
    }

    /// Deviation from target (positive = louder than target).
    pub fn deviation(&self) -> f64 {
        self.lufs - self.target_lufs
    }
}

/// Momentary loudness — 400 ms window, updated every 100 ms.
#[derive(Debug, Clone, Copy)]
pub struct MomentaryLoudness {
    /// Momentary level in LUFS.
    pub lufs: f64,
    /// True-peak level in dBFS for the same window.
    pub true_peak_dbfs: f64,
}

impl MomentaryLoudness {
    /// Create a new momentary measurement.
    pub fn new(lufs: f64, true_peak_dbfs: f64) -> Self {
        Self {
            lufs,
            true_peak_dbfs,
        }
    }

    /// Returns `true` if the true-peak level exceeds -1 dBFS (EBU R128 clipping limit).
    pub fn is_clipping(&self) -> bool {
        self.true_peak_dbfs > -1.0
    }

    /// Returns `true` if the level is below absolute silence (-70 LUFS).
    pub fn is_silent(&self) -> bool {
        self.lufs < -70.0
    }
}

/// Configuration for the loudness meter.
#[derive(Debug, Clone)]
pub struct LoudnessMeterConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u8,
    /// Integration window for momentary loudness in milliseconds (default 400).
    pub window_ms: u32,
    /// EBU / ITU target level in LUFS.
    pub target_lufs: f64,
}

impl LoudnessMeterConfig {
    /// Create a default EBU R128 configuration.
    pub fn ebu_r128(sample_rate: u32, channels: u8) -> Self {
        Self {
            sample_rate,
            channels,
            window_ms: 400,
            target_lufs: -23.0,
        }
    }

    /// Create a default ATSC A/85 configuration.
    pub fn atsc_a85(sample_rate: u32, channels: u8) -> Self {
        Self {
            sample_rate,
            channels,
            window_ms: 400,
            target_lufs: -24.0,
        }
    }

    /// Returns the configured window size in milliseconds.
    pub fn window_ms(&self) -> u32 {
        self.window_ms
    }

    /// Window size in samples.
    pub fn window_samples(&self) -> u32 {
        self.sample_rate * self.window_ms / 1000
    }
}

/// Stateful loudness meter accumulating blocks of audio.
#[derive(Debug)]
pub struct LoudnessMeter {
    config: LoudnessMeterConfig,
    /// Sum of squared sample values (for RMS approximation).
    sum_sq: f64,
    /// Total sample count processed.
    total_samples: u64,
    /// Short-term buffer (sliding window).
    window_buffer: Vec<f64>,
    window_pos: usize,
}

impl LoudnessMeter {
    /// Create a new meter from the given configuration.
    pub fn new(config: LoudnessMeterConfig) -> Self {
        let win = config.window_samples() as usize;
        Self {
            config,
            sum_sq: 0.0,
            total_samples: 0,
            window_buffer: vec![0.0; win.max(1)],
            window_pos: 0,
        }
    }

    /// Push a block of interleaved f32 samples into the meter.
    pub fn push_block(&mut self, samples: &[f32]) {
        for &s in samples {
            let sq = f64::from(s) * f64::from(s);
            let win_len = self.window_buffer.len();
            let old = self.window_buffer[self.window_pos];
            self.window_buffer[self.window_pos] = sq;
            self.window_pos = (self.window_pos + 1) % win_len;
            self.sum_sq += sq - old;
            self.total_samples += 1;
        }
    }

    /// Return the integrated loudness measured so far.
    ///
    /// Uses an RMS approximation; a production implementation would
    /// apply K-weighting and the gating algorithm from ITU-R BS.1770-4.
    pub fn integrated_lkfs(&self) -> IntegratedLoudness {
        let mean_sq = if self.total_samples == 0 {
            0.0
        } else {
            self.sum_sq / self.total_samples as f64
        };
        let lufs = if mean_sq > 1e-15 {
            -0.691 + 10.0 * mean_sq.log10()
        } else {
            -f64::INFINITY
        };
        IntegratedLoudness::new(lufs, self.config.target_lufs, 1.0)
    }

    /// Return the momentary loudness from the sliding window.
    pub fn momentary(&self) -> MomentaryLoudness {
        let win_len = self.window_buffer.len() as f64;
        let mean_sq = self.window_buffer.iter().sum::<f64>() / win_len;
        let lufs = if mean_sq > 1e-15 {
            -0.691 + 10.0 * mean_sq.log10()
        } else {
            -f64::INFINITY
        };
        let peak = self
            .window_buffer
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .sqrt();
        let true_peak_dbfs = if peak > 1e-15 {
            20.0 * peak.log10()
        } else {
            -f64::INFINITY
        };
        MomentaryLoudness::new(lufs, true_peak_dbfs)
    }

    /// Reset the meter to its initial state.
    pub fn reset(&mut self) {
        self.sum_sq = 0.0;
        self.total_samples = 0;
        self.window_pos = 0;
        for v in &mut self.window_buffer {
            *v = 0.0;
        }
    }

    /// Total number of samples processed.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- IntegratedLoudness ---

    #[test]
    fn test_within_spec_exactly_on_target() {
        let il = IntegratedLoudness::new(-23.0, -23.0, 1.0);
        assert!(il.is_within_spec());
    }

    #[test]
    fn test_within_spec_inside_tolerance() {
        let il = IntegratedLoudness::new(-23.5, -23.0, 1.0);
        assert!(il.is_within_spec());
    }

    #[test]
    fn test_outside_spec() {
        let il = IntegratedLoudness::new(-20.0, -23.0, 1.0);
        assert!(!il.is_within_spec());
    }

    #[test]
    fn test_deviation_sign() {
        let il = IntegratedLoudness::new(-22.0, -23.0, 1.0);
        // louder than target → positive deviation
        assert!(il.deviation() > 0.0);
    }

    // --- MomentaryLoudness ---

    #[test]
    fn test_momentary_clipping_above_minus_one() {
        let m = MomentaryLoudness::new(-6.0, 0.5);
        assert!(m.is_clipping());
    }

    #[test]
    fn test_momentary_no_clipping_below_minus_one() {
        let m = MomentaryLoudness::new(-6.0, -3.0);
        assert!(!m.is_clipping());
    }

    #[test]
    fn test_momentary_silent() {
        let m = MomentaryLoudness::new(-80.0, -80.0);
        assert!(m.is_silent());
    }

    #[test]
    fn test_momentary_not_silent() {
        let m = MomentaryLoudness::new(-23.0, -10.0);
        assert!(!m.is_silent());
    }

    // --- LoudnessMeterConfig ---

    #[test]
    fn test_config_ebu_target() {
        let cfg = LoudnessMeterConfig::ebu_r128(48_000, 2);
        assert!((cfg.target_lufs - -23.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_atsc_target() {
        let cfg = LoudnessMeterConfig::atsc_a85(48_000, 2);
        assert!((cfg.target_lufs - -24.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_window_ms_accessor() {
        let cfg = LoudnessMeterConfig::ebu_r128(48_000, 2);
        assert_eq!(cfg.window_ms(), 400);
    }

    #[test]
    fn test_config_window_samples() {
        let cfg = LoudnessMeterConfig::ebu_r128(48_000, 2);
        // 48000 * 400 / 1000 = 19200
        assert_eq!(cfg.window_samples(), 19_200);
    }

    // --- LoudnessMeter ---

    #[test]
    fn test_meter_initial_sample_count() {
        let cfg = LoudnessMeterConfig::ebu_r128(48_000, 2);
        let meter = LoudnessMeter::new(cfg);
        assert_eq!(meter.total_samples(), 0);
    }

    #[test]
    fn test_meter_push_updates_count() {
        let cfg = LoudnessMeterConfig::ebu_r128(48_000, 2);
        let mut meter = LoudnessMeter::new(cfg);
        let block = vec![0.1f32; 100];
        meter.push_block(&block);
        assert_eq!(meter.total_samples(), 100);
    }

    #[test]
    fn test_meter_reset_clears_count() {
        let cfg = LoudnessMeterConfig::ebu_r128(48_000, 2);
        let mut meter = LoudnessMeter::new(cfg);
        meter.push_block(&vec![0.5f32; 512]);
        meter.reset();
        assert_eq!(meter.total_samples(), 0);
    }

    #[test]
    fn test_integrated_lkfs_silent_after_reset() {
        let cfg = LoudnessMeterConfig::ebu_r128(48_000, 2);
        let meter = LoudnessMeter::new(cfg);
        let il = meter.integrated_lkfs();
        assert!(il.lufs.is_infinite());
    }

    #[test]
    fn test_momentary_clipping_with_full_scale() {
        let cfg = LoudnessMeterConfig::ebu_r128(48_000, 2);
        let win_len = cfg.window_samples() as usize;
        let mut meter = LoudnessMeter::new(cfg);
        // Full-scale sine approximation (all samples at 1.0).
        let block = vec![1.0f32; win_len];
        meter.push_block(&block);
        let m = meter.momentary();
        // True peak of 1.0 → 0 dBFS which is > -1 dBFS.
        assert!(m.is_clipping());
    }

    #[test]
    fn test_integrated_lkfs_non_silent() {
        let cfg = LoudnessMeterConfig::ebu_r128(48_000, 2);
        let mut meter = LoudnessMeter::new(cfg);
        // Push a moderate level signal.
        let block: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
        meter.push_block(&block);
        let il = meter.integrated_lkfs();
        assert!(il.lufs.is_finite());
        assert!(il.lufs < 0.0);
    }
}

#![allow(dead_code)]
//! Audio loudness visualization scope.
//!
//! Provides a real-time loudness meter and visualization similar to broadcast
//! loudness meters (EBU R128 / ATSC A/85). Displays momentary, short-term,
//! and integrated loudness values in LUFS, along with loudness range (LRA)
//! and true peak measurements.

/// Reference level for LUFS calculations.
const LUFS_REFERENCE: f64 = -23.0;

/// Loudness measurement window type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoudnessWindow {
    /// Momentary loudness (400 ms window).
    Momentary,
    /// Short-term loudness (3 s window).
    ShortTerm,
    /// Integrated loudness (entire program).
    Integrated,
}

/// Loudness target standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoudnessStandard {
    /// EBU R128 (-23 LUFS).
    EbuR128,
    /// ATSC A/85 (-24 LKFS).
    AtscA85,
    /// Custom target level.
    Custom,
}

/// Configuration for the loudness scope.
#[derive(Debug, Clone)]
pub struct LoudnessScopeConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub num_channels: u32,
    /// Target loudness standard.
    pub standard: LoudnessStandard,
    /// Custom target level in LUFS (used when standard is Custom).
    pub target_lufs: f64,
    /// Tolerance in LU for pass/fail indication.
    pub tolerance_lu: f64,
    /// Whether to compute true peak.
    pub measure_true_peak: bool,
    /// History length in seconds for the loudness graph.
    pub history_seconds: f64,
}

impl Default for LoudnessScopeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            num_channels: 2,
            standard: LoudnessStandard::EbuR128,
            target_lufs: -23.0,
            tolerance_lu: 1.0,
            measure_true_peak: true,
            history_seconds: 30.0,
        }
    }
}

impl LoudnessScopeConfig {
    /// Returns the target level in LUFS for the configured standard.
    #[must_use]
    pub fn target_level(&self) -> f64 {
        match self.standard {
            LoudnessStandard::EbuR128 => -23.0,
            LoudnessStandard::AtscA85 => -24.0,
            LoudnessStandard::Custom => self.target_lufs,
        }
    }
}

/// A single loudness measurement snapshot.
#[derive(Debug, Clone)]
pub struct LoudnessMeasurement {
    /// Momentary loudness (400 ms) in LUFS.
    pub momentary_lufs: f64,
    /// Short-term loudness (3 s) in LUFS.
    pub short_term_lufs: f64,
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// Loudness range (LRA) in LU.
    pub loudness_range_lu: f64,
    /// True peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Per-channel true peak values in dBTP.
    pub channel_peaks: Vec<f64>,
    /// Timestamp in seconds.
    pub timestamp_secs: f64,
}

impl Default for LoudnessMeasurement {
    fn default() -> Self {
        Self {
            momentary_lufs: -70.0,
            short_term_lufs: -70.0,
            integrated_lufs: -70.0,
            loudness_range_lu: 0.0,
            true_peak_dbtp: -70.0,
            channel_peaks: Vec::new(),
            timestamp_secs: 0.0,
        }
    }
}

/// K-weighting pre-filter state (two cascaded biquad filters).
#[derive(Debug, Clone)]
struct KWeightFilter {
    /// Stage 1 coefficients (high shelf).
    s1_b: [f64; 3],
    /// Stage 1 feedback coefficients.
    s1_a: [f64; 3],
    /// Stage 2 coefficients (high pass).
    s2_b: [f64; 3],
    /// Stage 2 feedback coefficients.
    s2_a: [f64; 3],
    /// State variables.
    s1_state: [f64; 2],
    /// Stage 2 state.
    s2_state: [f64; 2],
}

impl KWeightFilter {
    /// Creates a K-weighting filter for the given sample rate.
    #[allow(clippy::cast_precision_loss)]
    fn new(sample_rate: u32) -> Self {
        // Simplified K-weighting coefficients for 48kHz
        // Stage 1: high shelf boost (+4 dB at high frequencies)
        // Stage 2: high pass (remove DC and sub-bass)
        let _fs = sample_rate as f64;

        // Using pre-computed coefficients for 48kHz
        // For other sample rates, these would need recalculation
        Self {
            s1_b: [1.53512485958697, -2.69169618940638, 1.19839281085285],
            s1_a: [1.0, -1.69065929318241, 0.73248077421585],
            s2_b: [1.0, -2.0, 1.0],
            s2_a: [1.0, -1.99004745483398, 0.99007225036621],
            s1_state: [0.0; 2],
            s2_state: [0.0; 2],
        }
    }

    /// Processes a single sample through the K-weighting filter.
    fn process(&mut self, input: f64) -> f64 {
        // Stage 1
        let s1_out = self.s1_b[0] * input + self.s1_state[0];
        self.s1_state[0] = self.s1_b[1] * input - self.s1_a[1] * s1_out + self.s1_state[1];
        self.s1_state[1] = self.s1_b[2] * input - self.s1_a[2] * s1_out;

        // Stage 2
        let s2_out = self.s2_b[0] * s1_out + self.s2_state[0];
        self.s2_state[0] = self.s2_b[1] * s1_out - self.s2_a[1] * s2_out + self.s2_state[1];
        self.s2_state[1] = self.s2_b[2] * s1_out - self.s2_a[2] * s2_out;

        s2_out
    }

    /// Resets the filter state.
    fn reset(&mut self) {
        self.s1_state = [0.0; 2];
        self.s2_state = [0.0; 2];
    }
}

/// Loudness scope analyzer.
#[derive(Debug, Clone)]
pub struct LoudnessScope {
    /// Configuration.
    config: LoudnessScopeConfig,
    /// Per-channel K-weighting filters.
    k_filters: Vec<KWeightFilter>,
    /// Momentary loudness accumulator (400ms blocks).
    momentary_acc: Vec<f64>,
    /// Short-term loudness accumulator (3s blocks).
    short_term_acc: Vec<f64>,
    /// Integrated loudness accumulator.
    integrated_sum: f64,
    /// Number of blocks for integrated loudness.
    integrated_count: u64,
    /// Loudness history.
    history: Vec<LoudnessMeasurement>,
    /// Sample counter within current block.
    block_sample_count: u64,
    /// Block size in samples for momentary (400ms).
    momentary_block_size: u64,
    /// Block overlap samples for short-term.
    short_term_block_size: u64,
    /// Current channel power sums.
    channel_power: Vec<f64>,
    /// True peak per channel.
    true_peak: Vec<f64>,
    /// Total sample counter.
    total_samples: u64,
}

impl LoudnessScope {
    /// Creates a new loudness scope with the given configuration.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn new(config: LoudnessScopeConfig) -> Self {
        let num_ch = config.num_channels as usize;
        let k_filters = (0..num_ch)
            .map(|_| KWeightFilter::new(config.sample_rate))
            .collect();
        let momentary_block_size = (config.sample_rate as f64 * 0.4) as u64;
        let short_term_block_size = (config.sample_rate as f64 * 3.0) as u64;

        Self {
            config,
            k_filters,
            momentary_acc: Vec::new(),
            short_term_acc: Vec::new(),
            integrated_sum: 0.0,
            integrated_count: 0,
            history: Vec::new(),
            block_sample_count: 0,
            momentary_block_size,
            short_term_block_size,
            channel_power: vec![0.0; num_ch],
            true_peak: vec![-100.0; num_ch],
            total_samples: 0,
        }
    }

    /// Processes interleaved audio samples and updates the loudness measurements.
    ///
    /// Samples should be interleaved: [L, R, L, R, ...] for stereo.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_samples(&mut self, samples: &[f32]) {
        let num_ch = self.config.num_channels as usize;
        if num_ch == 0 {
            return;
        }
        let num_frames = samples.len() / num_ch;

        for frame_idx in 0..num_frames {
            for ch in 0..num_ch {
                let sample = f64::from(samples[frame_idx * num_ch + ch]);

                // True peak
                let abs_sample = sample.abs();
                if abs_sample > self.true_peak[ch] {
                    self.true_peak[ch] = abs_sample;
                }

                // K-weighting
                let filtered = self.k_filters[ch].process(sample);
                self.channel_power[ch] += filtered * filtered;
            }

            self.block_sample_count += 1;
            self.total_samples += 1;

            // Check if we've completed a momentary block
            if self.block_sample_count >= self.momentary_block_size {
                let loudness = self.compute_block_loudness();
                self.momentary_acc.push(loudness);
                self.integrated_sum += loudness;
                self.integrated_count += 1;

                // Reset block
                self.channel_power.fill(0.0);
                self.block_sample_count = 0;
            }
        }
    }

    /// Computes the loudness of the current block.
    #[allow(clippy::cast_precision_loss)]
    fn compute_block_loudness(&self) -> f64 {
        let num_ch = self.config.num_channels as usize;
        if num_ch == 0 || self.block_sample_count == 0 {
            return -70.0;
        }

        let mut sum = 0.0;
        for ch in 0..num_ch {
            let mean_sq = self.channel_power[ch] / self.block_sample_count as f64;
            // Channel weight (all 1.0 for stereo, surround would use different weights)
            sum += mean_sq;
        }

        if sum <= 0.0 {
            -70.0
        } else {
            -0.691 + 10.0 * sum.log10()
        }
    }

    /// Returns the latest loudness measurement.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn measurement(&self) -> LoudnessMeasurement {
        let momentary = if let Some(last) = self.momentary_acc.last() {
            *last
        } else {
            -70.0
        };

        // Short-term: average of last ~8 momentary blocks (3s / 0.4s ~ 7.5)
        let st_blocks = 8.min(self.momentary_acc.len());
        let short_term = if st_blocks > 0 {
            let start = self.momentary_acc.len() - st_blocks;
            let sum: f64 = self.momentary_acc[start..].iter().sum();
            sum / st_blocks as f64
        } else {
            -70.0
        };

        let integrated = if self.integrated_count > 0 {
            self.integrated_sum / self.integrated_count as f64
        } else {
            -70.0
        };

        let channel_peaks: Vec<f64> = self
            .true_peak
            .iter()
            .map(|&p| if p > 0.0 { 20.0 * p.log10() } else { -70.0 })
            .collect();

        let max_peak = channel_peaks.iter().copied().fold(-70.0_f64, f64::max);

        LoudnessMeasurement {
            momentary_lufs: momentary,
            short_term_lufs: short_term,
            integrated_lufs: integrated,
            loudness_range_lu: self.compute_lra(),
            true_peak_dbtp: max_peak,
            channel_peaks,
            timestamp_secs: self.total_samples as f64 / f64::from(self.config.sample_rate),
        }
    }

    /// Computes the loudness range (LRA).
    fn compute_lra(&self) -> f64 {
        if self.momentary_acc.len() < 2 {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.momentary_acc.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // LRA is the range between 10th and 95th percentile
        let low_idx = sorted.len() / 10;
        let high_idx = sorted.len() * 95 / 100;
        if high_idx <= low_idx {
            return 0.0;
        }
        sorted[high_idx] - sorted[low_idx]
    }

    /// Checks if the current integrated loudness is within tolerance.
    #[must_use]
    pub fn is_within_target(&self) -> bool {
        let measurement = self.measurement();
        let target = self.config.target_level();
        (measurement.integrated_lufs - target).abs() <= self.config.tolerance_lu
    }

    /// Resets the loudness scope state.
    pub fn reset(&mut self) {
        for f in &mut self.k_filters {
            f.reset();
        }
        self.momentary_acc.clear();
        self.short_term_acc.clear();
        self.integrated_sum = 0.0;
        self.integrated_count = 0;
        self.history.clear();
        self.block_sample_count = 0;
        self.channel_power.fill(0.0);
        self.true_peak.fill(-100.0);
        self.total_samples = 0;
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &LoudnessScopeConfig {
        &self.config
    }

    /// Returns the total number of samples processed.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Returns the target level in LUFS.
    #[must_use]
    pub fn target_level(&self) -> f64 {
        self.config.target_level()
    }
}

/// Converts a linear amplitude to dBFS.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn amplitude_to_dbfs(amplitude: f64) -> f64 {
    if amplitude <= 0.0 {
        -100.0
    } else {
        20.0 * amplitude.log10()
    }
}

/// Converts dBFS to linear amplitude.
#[must_use]
pub fn dbfs_to_amplitude(dbfs: f64) -> f64 {
    10.0_f64.powf(dbfs / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_scope_creation() {
        let config = LoudnessScopeConfig::default();
        let scope = LoudnessScope::new(config);
        assert_eq!(scope.total_samples(), 0);
    }

    #[test]
    fn test_loudness_config_default() {
        let config = LoudnessScopeConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.num_channels, 2);
        assert_eq!(config.standard, LoudnessStandard::EbuR128);
        assert!((config.target_level() - (-23.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_target_levels() {
        let mut config = LoudnessScopeConfig::default();
        config.standard = LoudnessStandard::EbuR128;
        assert!((config.target_level() - (-23.0)).abs() < f64::EPSILON);

        config.standard = LoudnessStandard::AtscA85;
        assert!((config.target_level() - (-24.0)).abs() < f64::EPSILON);

        config.standard = LoudnessStandard::Custom;
        config.target_lufs = -16.0;
        assert!((config.target_level() - (-16.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_process_silence() {
        let config = LoudnessScopeConfig::default();
        let mut scope = LoudnessScope::new(config);
        let silence = vec![0.0f32; 48000 * 2]; // 1 second stereo
        scope.process_samples(&silence);
        let m = scope.measurement();
        assert!(m.momentary_lufs < -60.0);
    }

    #[test]
    fn test_process_signal() {
        let config = LoudnessScopeConfig::default();
        let mut scope = LoudnessScope::new(config);
        // 1 second of -20 dBFS sine (roughly)
        let amplitude = 0.1_f32; // about -20 dBFS
        let samples: Vec<f32> = (0..48000)
            .flat_map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let s =
                    amplitude * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin();
                vec![s, s] // stereo
            })
            .collect();
        scope.process_samples(&samples);
        let m = scope.measurement();
        // Should have some loudness value
        assert!(m.momentary_lufs > -70.0);
    }

    #[test]
    fn test_reset() {
        let config = LoudnessScopeConfig::default();
        let mut scope = LoudnessScope::new(config);
        let signal = vec![0.5f32; 48000 * 2];
        scope.process_samples(&signal);
        assert!(scope.total_samples() > 0);
        scope.reset();
        assert_eq!(scope.total_samples(), 0);
    }

    #[test]
    fn test_amplitude_to_dbfs() {
        assert!((amplitude_to_dbfs(1.0) - 0.0).abs() < 0.01);
        assert!((amplitude_to_dbfs(0.5) - (-6.02)).abs() < 0.1);
        assert!(amplitude_to_dbfs(0.0) < -99.0);
    }

    #[test]
    fn test_dbfs_to_amplitude() {
        assert!((dbfs_to_amplitude(0.0) - 1.0).abs() < 0.01);
        assert!((dbfs_to_amplitude(-6.0) - 0.5012).abs() < 0.01);
    }

    #[test]
    fn test_roundtrip_dbfs() {
        let original = 0.7;
        let dbfs = amplitude_to_dbfs(original);
        let back = dbfs_to_amplitude(dbfs);
        assert!((original - back).abs() < 0.001);
    }

    #[test]
    fn test_measurement_default() {
        let m = LoudnessMeasurement::default();
        assert!(m.momentary_lufs < -60.0);
        assert!(m.short_term_lufs < -60.0);
        assert!(m.integrated_lufs < -60.0);
    }

    #[test]
    fn test_within_target_silence() {
        let config = LoudnessScopeConfig::default();
        let mut scope = LoudnessScope::new(config);
        let silence = vec![0.0f32; 48000 * 2];
        scope.process_samples(&silence);
        // Silence is not within target of -23 LUFS
        assert!(!scope.is_within_target());
    }

    #[test]
    fn test_k_weight_filter() {
        let mut filter = KWeightFilter::new(48000);
        // Process some samples
        for _ in 0..100 {
            let _ = filter.process(0.5);
        }
        filter.reset();
        let out = filter.process(0.0);
        assert!(out.abs() < 0.01);
    }

    #[test]
    fn test_loudness_range_few_blocks() {
        let config = LoudnessScopeConfig::default();
        let scope = LoudnessScope::new(config);
        // With no blocks, LRA should be 0
        assert!(scope.compute_lra().abs() < f64::EPSILON);
    }
}

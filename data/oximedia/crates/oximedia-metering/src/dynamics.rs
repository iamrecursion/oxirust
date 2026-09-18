//! Dynamic range measurement.
//!
//! Measures the dynamic range of audio signals using various methods.

use crate::{MeteringError, MeteringResult};

/// Dynamic range meter.
///
/// Measures the dynamic range of an audio signal using the difference
/// between peak levels and RMS levels.
pub struct DynamicRangeMeter {
    sample_rate: f64,
    channels: usize,
    peak_values: Vec<f64>,
    rms_values: Vec<f64>,
    rms_buffer: Vec<Vec<f64>>,
    rms_buffer_size: usize,
    rms_write_pos: usize,
}

impl DynamicRangeMeter {
    /// Create a new dynamic range meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `rms_integration_time` - RMS integration time in seconds
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(
        sample_rate: f64,
        channels: usize,
        rms_integration_time: f64,
    ) -> MeteringResult<Self> {
        if sample_rate <= 0.0 {
            return Err(MeteringError::InvalidConfig(
                "Sample rate must be positive".to_string(),
            ));
        }

        if channels == 0 {
            return Err(MeteringError::InvalidConfig(
                "Must have at least one channel".to_string(),
            ));
        }

        let rms_buffer_size = (sample_rate * rms_integration_time) as usize;
        let rms_buffer = vec![vec![0.0; rms_buffer_size]; channels];

        Ok(Self {
            sample_rate,
            channels,
            peak_values: vec![0.0; channels],
            rms_values: vec![0.0; channels],
            rms_buffer,
            rms_buffer_size,
            rms_write_pos: 0,
        })
    }

    /// Process interleaved audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let num_frames = samples.len() / self.channels;

        for frame_idx in 0..num_frames {
            for ch in 0..self.channels {
                let sample_idx = frame_idx * self.channels + ch;
                let sample = samples[sample_idx];

                // Update peak
                let abs_sample = sample.abs();
                if abs_sample > self.peak_values[ch] {
                    self.peak_values[ch] = abs_sample;
                }

                // Update RMS buffer
                let squared = sample * sample;
                self.rms_buffer[ch][self.rms_write_pos] = squared;
            }

            // Advance RMS write position
            self.rms_write_pos = (self.rms_write_pos + 1) % self.rms_buffer_size;
        }

        // Calculate RMS for all channels
        for ch in 0..self.channels {
            let sum: f64 = self.rms_buffer[ch].iter().sum();
            self.rms_values[ch] = (sum / self.rms_buffer_size as f64).sqrt();
        }
    }

    /// Get the dynamic range in dB for each channel.
    ///
    /// Dynamic range = Peak level - RMS level (in dB)
    pub fn dynamic_range_db(&self) -> Vec<f64> {
        self.peak_values
            .iter()
            .zip(&self.rms_values)
            .map(|(&peak, &rms)| {
                let peak_db = if peak > 0.0 {
                    20.0 * peak.log10()
                } else {
                    f64::NEG_INFINITY
                };

                let rms_db = if rms > 0.0 {
                    20.0 * rms.log10()
                } else {
                    f64::NEG_INFINITY
                };

                if peak_db.is_finite() && rms_db.is_finite() {
                    peak_db - rms_db
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Get the crest factor for each channel.
    ///
    /// Crest factor = Peak / RMS (linear ratio)
    pub fn crest_factor(&self) -> Vec<f64> {
        self.peak_values
            .iter()
            .zip(&self.rms_values)
            .map(|(&peak, &rms)| if rms > 0.0 { peak / rms } else { 0.0 })
            .collect()
    }

    /// Get the crest factor in dB for each channel.
    pub fn crest_factor_db(&self) -> Vec<f64> {
        self.crest_factor()
            .iter()
            .map(|&cf| {
                if cf > 0.0 {
                    20.0 * cf.log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Get peak values in dBFS.
    pub fn peak_dbfs(&self) -> Vec<f64> {
        self.peak_values
            .iter()
            .map(|&peak| {
                if peak > 0.0 {
                    20.0 * peak.log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Get RMS values in dBFS.
    pub fn rms_dbfs(&self) -> Vec<f64> {
        self.rms_values
            .iter()
            .map(|&rms| {
                if rms > 0.0 {
                    20.0 * rms.log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.peak_values.fill(0.0);
        self.rms_values.fill(0.0);
        self.rms_write_pos = 0;

        for buffer in &mut self.rms_buffer {
            buffer.fill(0.0);
        }
    }
}

/// PLR (Peak to Loudness Ratio) meter.
///
/// Measures the difference between true peak and integrated loudness,
/// useful for assessing headroom and dynamics in mastered content.
pub struct PlrMeter {
    true_peak_dbtp: f64,
    integrated_lufs: f64,
}

impl PlrMeter {
    /// Create a new PLR meter.
    pub fn new() -> Self {
        Self {
            true_peak_dbtp: f64::NEG_INFINITY,
            integrated_lufs: f64::NEG_INFINITY,
        }
    }

    /// Update with true peak and integrated loudness values.
    ///
    /// # Arguments
    ///
    /// * `true_peak_dbtp` - True peak in dBTP
    /// * `integrated_lufs` - Integrated loudness in LUFS
    pub fn update(&mut self, true_peak_dbtp: f64, integrated_lufs: f64) {
        self.true_peak_dbtp = true_peak_dbtp;
        self.integrated_lufs = integrated_lufs;
    }

    /// Get the PLR value in dB.
    ///
    /// PLR = True Peak (dBTP) - Integrated Loudness (LUFS)
    pub fn plr_db(&self) -> f64 {
        if self.true_peak_dbtp.is_finite() && self.integrated_lufs.is_finite() {
            self.true_peak_dbtp - self.integrated_lufs
        } else {
            0.0
        }
    }

    /// Get the true peak value.
    pub fn true_peak_dbtp(&self) -> f64 {
        self.true_peak_dbtp
    }

    /// Get the integrated loudness value.
    pub fn integrated_lufs(&self) -> f64 {
        self.integrated_lufs
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.true_peak_dbtp = f64::NEG_INFINITY;
        self.integrated_lufs = f64::NEG_INFINITY;
    }
}

impl Default for PlrMeter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_range_meter() {
        let mut meter = DynamicRangeMeter::new(48000.0, 2, 0.1).expect("test expectation failed");

        // Generate test signal with known dynamics
        let mut samples = Vec::new();
        for i in 0..4800 {
            let t = i as f64 / 48000.0;
            let signal = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5;
            samples.push(signal);
            samples.push(signal);
        }

        meter.process_interleaved(&samples);

        let dr = meter.dynamic_range_db();
        assert!(dr[0] > 0.0);
        assert!(dr[0] < 10.0); // Sine wave has moderate dynamic range
    }

    #[test]
    fn test_crest_factor() {
        let mut meter = DynamicRangeMeter::new(48000.0, 1, 0.1).expect("test expectation failed");

        // Generate sine wave
        let mut samples = Vec::new();
        for i in 0..4800 {
            let t = i as f64 / 48000.0;
            let signal = (2.0 * std::f64::consts::PI * 1000.0 * t).sin();
            samples.push(signal);
        }

        meter.process_interleaved(&samples);

        let cf = meter.crest_factor()[0];
        // Sine wave crest factor should be ~1.414 (sqrt(2))
        assert!((cf - 1.414).abs() < 0.1);
    }

    #[test]
    fn test_plr_meter() {
        let mut meter = PlrMeter::new();

        meter.update(-1.0, -23.0);

        let plr = meter.plr_db();
        assert_eq!(plr, 22.0); // -1.0 - (-23.0) = 22.0
    }

    #[test]
    fn test_dynamic_range_reset() {
        let mut meter = DynamicRangeMeter::new(48000.0, 2, 0.1).expect("test expectation failed");

        meter.process_interleaved(&[0.5, 0.5, 0.5, 0.5]);
        meter.reset();

        let dr = meter.dynamic_range_db();
        assert_eq!(dr[0], 0.0);
        assert_eq!(dr[1], 0.0);
    }
}

// ── Types merged from dynamic_range_meter module ─────────────────────────────

/// Result of a crest factor measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrestFactorResult {
    /// Peak level in dBFS.
    pub peak_dbfs: f64,
    /// RMS level in dBFS.
    pub rms_dbfs: f64,
    /// Crest factor in dB (peak - RMS).
    pub crest_db: f64,
}

impl CrestFactorResult {
    /// Compute crest factor from a block of mono samples.
    #[must_use]
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let peak_linear = samples.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        let rms_linear = (samples.iter().map(|s| s * s).sum::<f64>() / samples.len() as f64).sqrt();

        let peak_dbfs = dr_linear_to_dbfs(peak_linear);
        let rms_dbfs = dr_linear_to_dbfs(rms_linear.max(1e-12));
        Some(Self {
            peak_dbfs,
            rms_dbfs,
            crest_db: peak_dbfs - rms_dbfs,
        })
    }

    /// Return `true` if the crest factor indicates heavily limited/compressed audio
    /// (crest factor < 6 dB is considered over-compressed for most content types).
    #[must_use]
    pub fn is_over_compressed(&self) -> bool {
        self.crest_db < 6.0
    }
}

/// Convert a linear amplitude to dBFS (full scale = 0 dBFS at amplitude 1.0).
#[must_use]
fn dr_linear_to_dbfs(linear: f64) -> f64 {
    if linear <= 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * linear.log10()
}

/// Convert dBFS to a linear amplitude.
#[must_use]
pub fn dr_dbfs_to_linear(dbfs: f64) -> f64 {
    10.0_f64.powf(dbfs / 20.0)
}

/// A simple histogram over a dBFS range.
#[derive(Debug, Clone)]
pub struct LevelHistogram {
    /// Number of bins.
    pub bins: usize,
    /// Minimum dBFS value.
    pub min_db: f64,
    /// Maximum dBFS value.
    pub max_db: f64,
    counts: Vec<u64>,
    total: u64,
}

impl LevelHistogram {
    /// Create a new histogram.
    ///
    /// `bins` must be > 0 and `min_db` must be < `max_db`.
    #[must_use]
    pub fn new(bins: usize, min_db: f64, max_db: f64) -> Self {
        let bins = bins.max(1);
        let (min_db, max_db) = if min_db >= max_db {
            (min_db - 1.0, min_db + 1.0)
        } else {
            (min_db, max_db)
        };
        Self {
            bins,
            min_db,
            max_db,
            counts: vec![0; bins],
            total: 0,
        }
    }

    /// Add a dBFS sample to the histogram.
    pub fn add(&mut self, dbfs: f64) {
        if dbfs.is_finite() && dbfs >= self.min_db && dbfs < self.max_db {
            let frac = (dbfs - self.min_db) / (self.max_db - self.min_db);
            let bin = (frac * self.bins as f64).min(self.bins as f64 - 1.0) as usize;
            self.counts[bin] += 1;
        }
        self.total += 1;
    }

    /// Add all samples from a slice (converts linear to dBFS first).
    pub fn add_linear_block(&mut self, samples: &[f64]) {
        for &s in samples {
            self.add(dr_linear_to_dbfs(s.abs().max(1e-12)));
        }
    }

    /// Return the count for a specific bin.
    #[must_use]
    pub fn bin_count(&self, bin: usize) -> u64 {
        self.counts.get(bin).copied().unwrap_or(0)
    }

    /// Return the centre dBFS value for a given bin.
    #[must_use]
    pub fn bin_centre_db(&self, bin: usize) -> f64 {
        let bin_width = (self.max_db - self.min_db) / self.bins as f64;
        self.min_db + (bin as f64 + 0.5) * bin_width
    }

    /// Return the normalised histogram (each bin count / total).
    #[must_use]
    pub fn normalized(&self) -> Vec<f64> {
        if self.total == 0 {
            return vec![0.0; self.bins];
        }
        self.counts
            .iter()
            .map(|&c| c as f64 / self.total as f64)
            .collect()
    }

    /// Return total samples added.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.total
    }

    /// Find the mode bin (highest count).
    #[must_use]
    pub fn mode_bin(&self) -> Option<usize> {
        self.counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
    }

    /// Reset all counts.
    pub fn reset(&mut self) {
        for c in &mut self.counts {
            *c = 0;
        }
        self.total = 0;
    }
}

/// PLR (Peak-to-Loudness Ratio) measurement.
/// PLR = true_peak_dbtp - integrated_loudness_lufs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlrResult {
    /// True peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// PLR in LU/dB.
    pub plr_lu: f64,
}

impl PlrResult {
    /// Compute PLR from true peak and integrated loudness.
    #[must_use]
    pub fn compute(true_peak_dbtp: f64, integrated_lufs: f64) -> Self {
        Self {
            true_peak_dbtp,
            integrated_lufs,
            plr_lu: true_peak_dbtp - integrated_lufs,
        }
    }

    /// Classify PLR: >14 LU = dynamic, 8-14 = moderate, <8 = compressed.
    #[must_use]
    pub fn classification(&self) -> &'static str {
        if self.plr_lu >= 14.0 {
            "Dynamic"
        } else if self.plr_lu >= 8.0 {
            "Moderate"
        } else {
            "Compressed"
        }
    }
}

/// PSR (Program Segment Ratio) measurement.
/// PSR = maximum short-term loudness - integrated loudness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsrResult {
    /// Maximum short-term loudness in LUFS.
    pub max_short_term_lufs: f64,
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// PSR in LU.
    pub psr_lu: f64,
}

impl PsrResult {
    /// Compute PSR.
    #[must_use]
    pub fn compute(max_short_term_lufs: f64, integrated_lufs: f64) -> Self {
        Self {
            max_short_term_lufs,
            integrated_lufs,
            psr_lu: max_short_term_lufs - integrated_lufs,
        }
    }

    /// Return `true` if the PSR suggests loudness variation typical of speech/drama (>3 LU).
    #[must_use]
    pub fn has_loudness_variation(&self) -> bool {
        self.psr_lu > 3.0
    }
}

/// Mono dynamic range meter with histogram, PLR, and PSR support.
///
/// Complements the multi-channel [`DynamicRangeMeter`] with mono-focused
/// analysis, level histogram, and crest factor computation.
pub struct MonoDynamicRangeMeter {
    peak_linear: f64,
    squared_sum: f64,
    sample_count: u64,
    short_term_window: Vec<f64>,
    st_write_pos: usize,
    histogram: LevelHistogram,
    integrated_lufs: Option<f64>,
    max_short_term_lufs: Option<f64>,
}

impl MonoDynamicRangeMeter {
    /// Create a new meter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            peak_linear: 0.0,
            squared_sum: 0.0,
            sample_count: 0,
            short_term_window: vec![-96.0; 30],
            st_write_pos: 0,
            histogram: LevelHistogram::new(120, -96.0, 0.0),
            integrated_lufs: None,
            max_short_term_lufs: None,
        }
    }

    /// Process a mono sample block.
    pub fn process_mono(&mut self, samples: &[f64]) {
        if samples.is_empty() {
            return;
        }
        for &s in samples {
            let abs_s = s.abs();
            if abs_s > self.peak_linear {
                self.peak_linear = abs_s;
            }
            self.squared_sum += s * s;
            self.sample_count += 1;
        }
        let block_rms = (samples.iter().map(|s| s * s).sum::<f64>() / samples.len() as f64).sqrt();
        let block_db = dr_linear_to_dbfs(block_rms.max(1e-12));
        self.short_term_window[self.st_write_pos] = block_db;
        self.st_write_pos = (self.st_write_pos + 1) % self.short_term_window.len();
        self.histogram.add_linear_block(samples);
    }

    /// Process interleaved stereo by mixing to mono first.
    pub fn process_stereo_interleaved(&mut self, samples: &[f64]) {
        let mono: Vec<f64> = samples
            .chunks_exact(2)
            .map(|c| (c[0] + c[1]) * 0.5)
            .collect();
        self.process_mono(&mono);
    }

    /// Supply integrated loudness from an external loudness meter (for PLR/PSR).
    pub fn set_integrated_lufs(&mut self, lufs: f64) {
        self.integrated_lufs = Some(lufs);
    }

    /// Supply maximum short-term loudness (for PSR).
    pub fn set_max_short_term_lufs(&mut self, lufs: f64) {
        let current = self.max_short_term_lufs.unwrap_or(f64::NEG_INFINITY);
        self.max_short_term_lufs = Some(lufs.max(current));
    }

    /// Return peak level in dBFS.
    #[must_use]
    pub fn peak_dbfs(&self) -> f64 {
        dr_linear_to_dbfs(self.peak_linear)
    }

    /// Return overall RMS in dBFS.
    #[must_use]
    pub fn rms_dbfs(&self) -> f64 {
        if self.sample_count == 0 {
            return f64::NEG_INFINITY;
        }
        let rms = (self.squared_sum / self.sample_count as f64).sqrt();
        dr_linear_to_dbfs(rms.max(1e-12))
    }

    /// Compute crest factor from accumulated data.
    #[must_use]
    pub fn crest_factor(&self) -> CrestFactorResult {
        let peak = self.peak_dbfs();
        let rms = self.rms_dbfs();
        CrestFactorResult {
            peak_dbfs: peak,
            rms_dbfs: rms,
            crest_db: peak - rms,
        }
    }

    /// Compute PLR if integrated loudness has been set.
    #[must_use]
    pub fn plr(&self) -> Option<PlrResult> {
        self.integrated_lufs
            .map(|lufs| PlrResult::compute(self.peak_dbfs(), lufs))
    }

    /// Compute PSR if both integrated and max short-term loudness have been set.
    #[must_use]
    pub fn psr(&self) -> Option<PsrResult> {
        match (self.max_short_term_lufs, self.integrated_lufs) {
            (Some(max_st), Some(integrated)) => Some(PsrResult::compute(max_st, integrated)),
            _ => None,
        }
    }

    /// Return a reference to the level histogram.
    #[must_use]
    pub fn histogram(&self) -> &LevelHistogram {
        &self.histogram
    }

    /// Total mono samples processed.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.sample_count
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.peak_linear = 0.0;
        self.squared_sum = 0.0;
        self.sample_count = 0;
        for v in &mut self.short_term_window {
            *v = -96.0;
        }
        self.st_write_pos = 0;
        self.histogram.reset();
        self.integrated_lufs = None;
        self.max_short_term_lufs = None;
    }
}

impl Default for MonoDynamicRangeMeter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod dynamic_range_meter_tests {
    use super::*;

    #[test]
    fn test_crest_factor_from_samples() {
        let sig: Vec<f64> = (0..4800)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();
        let cf = CrestFactorResult::from_samples(&sig).expect("cf should be valid");
        assert!(
            cf.crest_db > 2.0 && cf.crest_db < 4.0,
            "crest={}",
            cf.crest_db
        );
    }

    #[test]
    fn test_crest_factor_empty() {
        assert!(CrestFactorResult::from_samples(&[]).is_none());
    }

    #[test]
    fn test_crest_factor_over_compressed() {
        let sig = vec![0.5_f64; 1000];
        let cf = CrestFactorResult::from_samples(&sig).expect("cf should be valid");
        assert!(cf.is_over_compressed());
    }

    #[test]
    fn test_histogram_add() {
        let mut h = LevelHistogram::new(10, -60.0, 0.0);
        h.add(-30.0);
        h.add(-10.0);
        assert_eq!(h.total_samples(), 2);
    }

    #[test]
    fn test_histogram_reset() {
        let mut h = LevelHistogram::new(5, -50.0, 0.0);
        h.add(-25.0);
        h.reset();
        assert_eq!(h.total_samples(), 0);
    }

    #[test]
    fn test_plr_classification() {
        assert_eq!(PlrResult::compute(-1.0, -20.0).classification(), "Dynamic");
        assert_eq!(PlrResult::compute(-1.0, -12.0).classification(), "Moderate");
        assert_eq!(
            PlrResult::compute(-1.0, -6.0).classification(),
            "Compressed"
        );
    }

    #[test]
    fn test_psr_has_variation() {
        let psr = PsrResult::compute(-10.0, -20.0);
        assert!(psr.has_loudness_variation());
        let psr2 = PsrResult::compute(-19.0, -20.0);
        assert!(!psr2.has_loudness_variation());
    }

    #[test]
    fn test_mono_meter_peak() {
        let mut meter = MonoDynamicRangeMeter::new();
        let sig = vec![0.0, 0.5, -0.8, 0.3];
        meter.process_mono(&sig);
        let peak = meter.peak_dbfs();
        assert!((peak - dr_linear_to_dbfs(0.8)).abs() < 1e-6);
    }

    #[test]
    fn test_mono_meter_reset() {
        let mut meter = MonoDynamicRangeMeter::new();
        let sig = vec![0.5_f64; 100];
        meter.process_mono(&sig);
        meter.reset();
        assert_eq!(meter.total_samples(), 0);
        assert_eq!(meter.peak_dbfs(), f64::NEG_INFINITY);
    }
}

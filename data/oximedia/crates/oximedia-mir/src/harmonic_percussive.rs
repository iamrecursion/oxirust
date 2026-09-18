#![allow(dead_code)]

//! Harmonic-percussive source separation (HPSS) for music analysis.
//!
//! Separates an audio spectrogram into harmonic (tonal) and percussive
//! (transient) components using median filtering along time and frequency axes.

use std::cmp::Ordering;

/// Default kernel size for median filtering along the time axis (harmonic).
const DEFAULT_HARMONIC_KERNEL: usize = 31;

/// Default kernel size for median filtering along the frequency axis (percussive).
const DEFAULT_PERCUSSIVE_KERNEL: usize = 31;

/// Configuration for HPSS.
#[derive(Debug, Clone)]
pub struct HpssConfig {
    /// Kernel size for harmonic median filter (time axis), must be odd.
    pub harmonic_kernel: usize,
    /// Kernel size for percussive median filter (frequency axis), must be odd.
    pub percussive_kernel: usize,
    /// Power exponent for soft mask (higher = harder mask). Typical: 2.0.
    pub mask_power: f64,
    /// Margin for harmonic component (>= 1.0).
    pub harmonic_margin: f64,
    /// Margin for percussive component (>= 1.0).
    pub percussive_margin: f64,
}

impl Default for HpssConfig {
    fn default() -> Self {
        Self {
            harmonic_kernel: DEFAULT_HARMONIC_KERNEL,
            percussive_kernel: DEFAULT_PERCUSSIVE_KERNEL,
            mask_power: 2.0,
            harmonic_margin: 1.0,
            percussive_margin: 1.0,
        }
    }
}

impl HpssConfig {
    /// Ensure kernel sizes are odd (round up if even).
    #[must_use]
    pub fn validated(mut self) -> Self {
        if self.harmonic_kernel % 2 == 0 {
            self.harmonic_kernel += 1;
        }
        if self.percussive_kernel % 2 == 0 {
            self.percussive_kernel += 1;
        }
        self.harmonic_kernel = self.harmonic_kernel.max(3);
        self.percussive_kernel = self.percussive_kernel.max(3);
        self.mask_power = self.mask_power.max(0.1);
        self.harmonic_margin = self.harmonic_margin.max(1.0);
        self.percussive_margin = self.percussive_margin.max(1.0);
        self
    }
}

/// Result of harmonic-percussive separation.
#[derive(Debug, Clone)]
pub struct HpssResult {
    /// Harmonic component spectrogram (n_freq x n_time).
    pub harmonic: Vec<Vec<f64>>,
    /// Percussive component spectrogram (n_freq x n_time).
    pub percussive: Vec<Vec<f64>>,
    /// Number of frequency bins.
    pub n_freq: usize,
    /// Number of time frames.
    pub n_time: usize,
}

impl HpssResult {
    /// Compute the harmonic-to-percussive ratio per frame.
    #[must_use]
    pub fn hp_ratio(&self) -> Vec<f64> {
        let mut ratios = Vec::with_capacity(self.n_time);
        for t in 0..self.n_time {
            let h_energy: f64 = (0..self.n_freq).map(|f| self.harmonic[f][t]).sum();
            let p_energy: f64 = (0..self.n_freq).map(|f| self.percussive[f][t]).sum();
            let ratio = if p_energy > 1e-12 {
                h_energy / p_energy
            } else {
                0.0
            };
            ratios.push(ratio);
        }
        ratios
    }

    /// Compute the total harmonic energy.
    #[must_use]
    pub fn total_harmonic_energy(&self) -> f64 {
        self.harmonic.iter().flat_map(|row| row.iter()).sum()
    }

    /// Compute the total percussive energy.
    #[must_use]
    pub fn total_percussive_energy(&self) -> f64 {
        self.percussive.iter().flat_map(|row| row.iter()).sum()
    }
}

/// Harmonic-percussive source separator.
#[derive(Debug)]
pub struct HpssSeparator {
    config: HpssConfig,
}

impl HpssSeparator {
    /// Create a new separator with the given configuration.
    #[must_use]
    pub fn new(config: HpssConfig) -> Self {
        Self {
            config: config.validated(),
        }
    }

    /// Create a separator with default configuration.
    #[must_use]
    pub fn default_separator() -> Self {
        Self::new(HpssConfig::default())
    }

    /// Perform HPSS on a magnitude spectrogram.
    ///
    /// The spectrogram is stored as `spectrogram[freq_bin][time_frame]`.
    ///
    /// # Arguments
    ///
    /// * `spectrogram` - 2D magnitude spectrogram (n_freq x n_time)
    ///
    /// # Returns
    ///
    /// Separated harmonic and percussive spectrograms.
    #[must_use]
    pub fn separate(&self, spectrogram: &[Vec<f64>]) -> HpssResult {
        let n_freq = spectrogram.len();
        if n_freq == 0 {
            return HpssResult {
                harmonic: vec![],
                percussive: vec![],
                n_freq: 0,
                n_time: 0,
            };
        }
        let n_time = spectrogram[0].len();

        // Median filter along time axis -> harmonic enhanced
        let harmonic_enhanced = self.median_filter_time(spectrogram, n_freq, n_time);

        // Median filter along frequency axis -> percussive enhanced
        let percussive_enhanced = self.median_filter_freq(spectrogram, n_freq, n_time);

        // Compute soft masks
        let p = self.config.mask_power;
        let hm = self.config.harmonic_margin;
        let pm = self.config.percussive_margin;

        let mut harmonic = vec![vec![0.0; n_time]; n_freq];
        let mut percussive = vec![vec![0.0; n_time]; n_freq];

        for f in 0..n_freq {
            for t in 0..n_time {
                let h_val = (harmonic_enhanced[f][t] * hm).powf(p);
                let p_val = (percussive_enhanced[f][t] * pm).powf(p);
                let total = h_val + p_val;
                if total > 1e-12 {
                    let h_mask = h_val / total;
                    let p_mask = p_val / total;
                    harmonic[f][t] = spectrogram[f][t] * h_mask;
                    percussive[f][t] = spectrogram[f][t] * p_mask;
                }
            }
        }

        HpssResult {
            harmonic,
            percussive,
            n_freq,
            n_time,
        }
    }

    /// Median filter along the time axis for each frequency bin.
    fn median_filter_time(&self, spec: &[Vec<f64>], n_freq: usize, n_time: usize) -> Vec<Vec<f64>> {
        let half = self.config.harmonic_kernel / 2;
        let mut out = vec![vec![0.0; n_time]; n_freq];
        for f in 0..n_freq {
            for t in 0..n_time {
                let start = t.saturating_sub(half);
                let end = (t + half + 1).min(n_time);
                out[f][t] = median_of_slice(&spec[f][start..end]);
            }
        }
        out
    }

    /// Median filter along the frequency axis for each time frame.
    fn median_filter_freq(&self, spec: &[Vec<f64>], n_freq: usize, n_time: usize) -> Vec<Vec<f64>> {
        let half = self.config.percussive_kernel / 2;
        let mut out = vec![vec![0.0; n_time]; n_freq];
        for t in 0..n_time {
            for f in 0..n_freq {
                let start = f.saturating_sub(half);
                let end = (f + half + 1).min(n_freq);
                let col: Vec<f64> = (start..end).map(|fi| spec[fi][t]).collect();
                out[f][t] = median_of_slice(&col);
            }
        }
        out
    }
}

/// Compute the median of a slice of f64 values.
fn median_of_slice(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    sorted[sorted.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hpss_config_default() {
        let cfg = HpssConfig::default();
        assert_eq!(cfg.harmonic_kernel, 31);
        assert_eq!(cfg.percussive_kernel, 31);
        assert!((cfg.mask_power - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hpss_config_validated_odd() {
        let cfg = HpssConfig {
            harmonic_kernel: 10,
            percussive_kernel: 8,
            ..HpssConfig::default()
        }
        .validated();
        assert_eq!(cfg.harmonic_kernel % 2, 1);
        assert_eq!(cfg.percussive_kernel % 2, 1);
    }

    #[test]
    fn test_hpss_config_validated_minimum() {
        let cfg = HpssConfig {
            harmonic_kernel: 1,
            percussive_kernel: 1,
            mask_power: -5.0,
            ..HpssConfig::default()
        }
        .validated();
        assert!(cfg.harmonic_kernel >= 3);
        assert!(cfg.percussive_kernel >= 3);
        assert!(cfg.mask_power >= 0.1);
    }

    #[test]
    fn test_median_of_slice() {
        assert!((median_of_slice(&[1.0, 3.0, 2.0]) - 2.0).abs() < f64::EPSILON);
        assert!((median_of_slice(&[5.0]) - 5.0).abs() < f64::EPSILON);
        assert!((median_of_slice(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_separate_empty() {
        let sep = HpssSeparator::default_separator();
        let result = sep.separate(&[]);
        assert_eq!(result.n_freq, 0);
        assert_eq!(result.n_time, 0);
    }

    #[test]
    fn test_separate_uniform() {
        let sep = HpssSeparator::new(HpssConfig {
            harmonic_kernel: 3,
            percussive_kernel: 3,
            mask_power: 2.0,
            harmonic_margin: 1.0,
            percussive_margin: 1.0,
        });
        // 4 freq bins x 8 time frames, all value 1.0
        let spec = vec![vec![1.0; 8]; 4];
        let result = sep.separate(&spec);
        assert_eq!(result.n_freq, 4);
        assert_eq!(result.n_time, 8);
        // For a uniform spectrogram, harmonic + percussive should roughly equal original
        for f in 0..4 {
            for t in 0..8 {
                let total = result.harmonic[f][t] + result.percussive[f][t];
                assert!((total - 1.0).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_hp_ratio_uniform() {
        let sep = HpssSeparator::new(HpssConfig {
            harmonic_kernel: 3,
            percussive_kernel: 3,
            ..HpssConfig::default()
        });
        let spec = vec![vec![1.0; 8]; 4];
        let result = sep.separate(&spec);
        let ratios = result.hp_ratio();
        assert_eq!(ratios.len(), 8);
        // For uniform input, ratio should be close to 1.0
        for &r in &ratios {
            assert!((r - 1.0).abs() < 0.5);
        }
    }

    #[test]
    fn test_total_energy_conservation() {
        let sep = HpssSeparator::new(HpssConfig {
            harmonic_kernel: 3,
            percussive_kernel: 3,
            ..HpssConfig::default()
        });
        let spec = vec![vec![2.0; 4]; 3];
        let result = sep.separate(&spec);
        let original_energy: f64 = spec.iter().flat_map(|r| r.iter()).sum();
        let separated_energy = result.total_harmonic_energy() + result.total_percussive_energy();
        assert!((original_energy - separated_energy).abs() < 0.5);
    }

    #[test]
    fn test_harmonic_dominance_horizontal_stripes() {
        // Horizontal stripe pattern = constant across time = harmonic
        let sep = HpssSeparator::new(HpssConfig {
            harmonic_kernel: 5,
            percussive_kernel: 5,
            ..HpssConfig::default()
        });
        let mut spec = vec![vec![0.0; 10]; 8];
        // Set frequency bin 3 to be strong across all time
        for t in 0..10 {
            spec[3][t] = 10.0;
        }
        let result = sep.separate(&spec);
        // Harmonic component at bin 3 should dominate
        let h_energy: f64 = (0..10).map(|t| result.harmonic[3][t]).sum();
        let p_energy: f64 = (0..10).map(|t| result.percussive[3][t]).sum();
        assert!(h_energy > p_energy);
    }

    #[test]
    fn test_percussive_dominance_vertical_stripes() {
        // Vertical stripe pattern = constant across freq = percussive
        let sep = HpssSeparator::new(HpssConfig {
            harmonic_kernel: 5,
            percussive_kernel: 5,
            ..HpssConfig::default()
        });
        let mut spec = vec![vec![0.0; 10]; 8];
        // Set time frame 5 to be strong across all frequencies
        for f in 0..8 {
            spec[f][5] = 10.0;
        }
        let result = sep.separate(&spec);
        let h_energy: f64 = (0..8).map(|f| result.harmonic[f][5]).sum();
        let p_energy: f64 = (0..8).map(|f| result.percussive[f][5]).sum();
        assert!(p_energy > h_energy);
    }

    #[test]
    fn test_single_bin_single_frame() {
        let sep = HpssSeparator::new(HpssConfig {
            harmonic_kernel: 3,
            percussive_kernel: 3,
            ..HpssConfig::default()
        });
        let spec = vec![vec![5.0]];
        let result = sep.separate(&spec);
        assert_eq!(result.n_freq, 1);
        assert_eq!(result.n_time, 1);
        let total = result.harmonic[0][0] + result.percussive[0][0];
        assert!((total - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_zero_spectrogram() {
        let sep = HpssSeparator::default_separator();
        let spec = vec![vec![0.0; 5]; 3];
        let result = sep.separate(&spec);
        assert!((result.total_harmonic_energy() - 0.0).abs() < f64::EPSILON);
        assert!((result.total_percussive_energy() - 0.0).abs() < f64::EPSILON);
    }
}

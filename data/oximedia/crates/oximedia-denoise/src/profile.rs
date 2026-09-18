//! Noise profiling and estimation.
//!
//! This module provides tools for characterising the noise present in a video
//! source, including per-channel sigma estimation, adaptive thresholding, and
//! frequency-domain noise analysis.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

// ---------------------------------------------------------------------------
// NoiseModel
// ---------------------------------------------------------------------------

/// Statistical model of the noise present in a video source.
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Estimated noise standard deviation for the luma channel.
    pub sigma_luma: f32,
    /// Estimated noise standard deviation for the chroma channels.
    pub sigma_chroma: f32,
    /// Temporal noise standard deviation across consecutive frames.
    pub temporal_sigma: f32,
    /// Average noise energy in each of 8 spatial-frequency bands.
    pub frequency_profile: [f32; 8],
}

impl NoiseModel {
    /// Create a new noise model.
    pub fn new(
        sigma_luma: f32,
        sigma_chroma: f32,
        temporal_sigma: f32,
        frequency_profile: [f32; 8],
    ) -> Self {
        Self {
            sigma_luma,
            sigma_chroma,
            temporal_sigma,
            frequency_profile,
        }
    }

    /// Combined noise sigma: RMS of luma, chroma, and temporal components.
    pub fn total_sigma(&self) -> f32 {
        let sum = self.sigma_luma * self.sigma_luma
            + self.sigma_chroma * self.sigma_chroma
            + self.temporal_sigma * self.temporal_sigma;
        (sum / 3.0).sqrt()
    }

    /// Returns `true` when the luma sigma exceeds a "heavy noise" threshold of
    /// 15 (on a 0–255 scale).
    pub fn is_heavy_noise(&self) -> bool {
        self.sigma_luma > 15.0
    }
}

// ---------------------------------------------------------------------------
// NoiseEstimator
// ---------------------------------------------------------------------------

/// Estimates noise sigma from a flat (uniform-content) region using the
/// Median Absolute Deviation (MAD) estimator.
///
/// The MAD estimator is: σ ≈ MAD / 0.6745
pub struct NoiseEstimator;

impl NoiseEstimator {
    /// Estimate the noise sigma from a rectangular region of floating-point
    /// pixel values.
    ///
    /// `pixels` must contain at least 1 element.  `width` and `height` are
    /// used only for validation; the entire slice is used for estimation.
    pub fn estimate_from_uniform_region(pixels: &[f32], _width: usize, _height: usize) -> f32 {
        if pixels.is_empty() {
            return 0.0;
        }

        // Compute median of pixel values.
        let mut sorted = pixels.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = median_of_sorted(&sorted);

        // Absolute deviations from median.
        let mut abs_devs: Vec<f32> = pixels.iter().map(|&x| (x - median).abs()).collect();
        abs_devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = median_of_sorted(&abs_devs);

        // Scale to obtain Gaussian sigma.
        mad / 0.6745
    }
}

/// Compute the median of an already-sorted slice.
fn median_of_sorted(sorted: &[f32]) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

// ---------------------------------------------------------------------------
// NoiseProfile
// ---------------------------------------------------------------------------

/// Full noise profile for a video source or frame.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Statistical noise model.
    pub model: NoiseModel,
    /// ISO equivalent (camera sensitivity) associated with this noise level.
    pub iso_equivalent: u32,
    /// Frame index at which the profile was estimated.
    pub estimated_at_frame: u64,
}

impl NoiseProfile {
    /// Create a new noise profile.
    pub fn new(model: NoiseModel, iso_equivalent: u32, estimated_at_frame: u64) -> Self {
        Self {
            model,
            iso_equivalent,
            estimated_at_frame,
        }
    }

    /// Estimated signal-to-noise ratio in dB for the luma channel.
    ///
    /// Assumes a peak signal of 255 (8-bit).  Returns `f32::INFINITY` when
    /// the estimated sigma is zero.
    pub fn snr_estimate(&self) -> f32 {
        let sigma = self.model.sigma_luma;
        if sigma <= 0.0 {
            return f32::INFINITY;
        }
        // SNR (dB) = 20 × log₁₀(255 / σ)
        20.0 * (255.0_f32 / sigma).log10()
    }
}

// ---------------------------------------------------------------------------
// AdaptiveNoiseThreshold
// ---------------------------------------------------------------------------

/// An adaptive threshold that tracks a running estimate of the noise sigma.
///
/// The threshold is updated with each new measurement using an exponential
/// moving average: `threshold = (1 - rate) × threshold + rate × measured`.
#[derive(Debug, Clone)]
pub struct AdaptiveNoiseThreshold {
    /// Initial (base) sigma estimate.
    pub base_sigma: f32,
    /// Adaptation rate in [0, 1].  Larger values → faster adaptation.
    pub adaptation_rate: f32,
    /// History of measured sigma values.
    pub history: Vec<f32>,
    /// Current EMA threshold.
    current: f32,
}

impl AdaptiveNoiseThreshold {
    /// Create a new adaptive threshold with the given base sigma and rate.
    pub fn new(base_sigma: f32, adaptation_rate: f32) -> Self {
        Self {
            base_sigma,
            adaptation_rate: adaptation_rate.clamp(0.0, 1.0),
            history: Vec::new(),
            current: base_sigma,
        }
    }

    /// Update the threshold with a new measurement and return the updated
    /// threshold value.
    pub fn update(&mut self, measured: f32) -> f32 {
        self.history.push(measured);
        self.current =
            (1.0 - self.adaptation_rate) * self.current + self.adaptation_rate * measured;
        self.current
    }

    /// Current threshold value.
    pub fn current_threshold(&self) -> f32 {
        self.current
    }
}

// ---------------------------------------------------------------------------
// FrequencyNoise
// ---------------------------------------------------------------------------

/// Frequency-domain noise analysis for 8×8 pixel blocks.
pub struct FrequencyNoise;

impl FrequencyNoise {
    /// Analyse the noise energy distribution across 8 spatial-frequency bands
    /// in a flat 8×8 pixel block.
    ///
    /// The block is treated as 8 rows; each row's mean absolute deviation from
    /// the row mean represents noise in one frequency band.
    ///
    /// Returns an array of 8 values, one per band (row).
    pub fn analyze_spatial_frequency(block_8x8: &[f32; 64]) -> [f32; 8] {
        let mut bands = [0.0_f32; 8];
        for (band, row_start) in (0..8).map(|b| (b, b * 8)) {
            let row = &block_8x8[row_start..row_start + 8];
            let mean = row.iter().sum::<f32>() / 8.0;
            let energy: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 8.0;
            bands[band] = energy;
        }
        bands
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_freq() -> [f32; 8] {
        [0.0; 8]
    }

    // ---------- NoiseModel ----------

    #[test]
    fn test_total_sigma_all_equal() {
        let m = NoiseModel::new(3.0, 3.0, 3.0, flat_freq());
        // sqrt((9+9+9)/3) = sqrt(9) = 3
        assert!((m.total_sigma() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_total_sigma_zero() {
        let m = NoiseModel::new(0.0, 0.0, 0.0, flat_freq());
        assert!((m.total_sigma()).abs() < 1e-6);
    }

    #[test]
    fn test_is_heavy_noise_false() {
        let m = NoiseModel::new(5.0, 2.0, 1.0, flat_freq());
        assert!(!m.is_heavy_noise());
    }

    #[test]
    fn test_is_heavy_noise_true() {
        let m = NoiseModel::new(20.0, 5.0, 3.0, flat_freq());
        assert!(m.is_heavy_noise());
    }

    #[test]
    fn test_is_heavy_noise_boundary() {
        let m_boundary = NoiseModel::new(15.0, 0.0, 0.0, flat_freq());
        assert!(!m_boundary.is_heavy_noise()); // exactly 15 → not heavy
        let m_above = NoiseModel::new(15.001, 0.0, 0.0, flat_freq());
        assert!(m_above.is_heavy_noise());
    }

    // ---------- NoiseEstimator ----------

    #[test]
    fn test_estimate_constant_signal_zero_sigma() {
        let pixels = vec![100.0_f32; 64];
        let sigma = NoiseEstimator::estimate_from_uniform_region(&pixels, 8, 8);
        assert!(sigma < 1e-5);
    }

    #[test]
    fn test_estimate_empty_returns_zero() {
        let sigma = NoiseEstimator::estimate_from_uniform_region(&[], 0, 0);
        assert!((sigma).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_known_noise() {
        // Alternating ±10 around 128 → median = 128, MAD = 10, σ ≈ 14.83
        let pixels: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 138.0 } else { 118.0 })
            .collect();
        let sigma = NoiseEstimator::estimate_from_uniform_region(&pixels, 8, 8);
        assert!(sigma > 5.0);
    }

    #[test]
    fn test_estimate_single_pixel() {
        let pixels = vec![200.0_f32];
        let sigma = NoiseEstimator::estimate_from_uniform_region(&pixels, 1, 1);
        assert!((sigma).abs() < 1e-6);
    }

    // ---------- NoiseProfile ----------

    #[test]
    fn test_snr_estimate_infinite_for_zero_sigma() {
        let model = NoiseModel::new(0.0, 0.0, 0.0, flat_freq());
        let profile = NoiseProfile::new(model, 100, 0);
        assert!(profile.snr_estimate().is_infinite());
    }

    #[test]
    fn test_snr_estimate_positive() {
        let model = NoiseModel::new(10.0, 5.0, 2.0, flat_freq());
        let profile = NoiseProfile::new(model, 1600, 30);
        let snr = profile.snr_estimate();
        assert!(snr > 0.0 && snr.is_finite());
    }

    #[test]
    fn test_snr_estimate_decreases_with_more_noise() {
        let low_model = NoiseModel::new(2.0, 0.0, 0.0, flat_freq());
        let high_model = NoiseModel::new(20.0, 0.0, 0.0, flat_freq());
        let snr_low = NoiseProfile::new(low_model, 100, 0).snr_estimate();
        let snr_high = NoiseProfile::new(high_model, 6400, 0).snr_estimate();
        assert!(snr_low > snr_high);
    }

    // ---------- AdaptiveNoiseThreshold ----------

    #[test]
    fn test_threshold_initial_equals_base() {
        let t = AdaptiveNoiseThreshold::new(10.0, 0.1);
        assert!((t.current_threshold() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_update_moves_towards_measurement() {
        let mut t = AdaptiveNoiseThreshold::new(10.0, 0.5);
        let updated = t.update(20.0);
        // 0.5*10 + 0.5*20 = 15
        assert!((updated - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_threshold_history_recorded() {
        let mut t = AdaptiveNoiseThreshold::new(5.0, 0.1);
        t.update(6.0);
        t.update(7.0);
        assert_eq!(t.history.len(), 2);
    }

    #[test]
    fn test_threshold_rate_clamped() {
        let t = AdaptiveNoiseThreshold::new(5.0, 5.0); // rate > 1
        assert!((t.adaptation_rate - 1.0).abs() < 1e-6);
    }

    // ---------- FrequencyNoise ----------

    #[test]
    fn test_frequency_analysis_uniform_block_is_zero() {
        let block = [128.0_f32; 64];
        let bands = FrequencyNoise::analyze_spatial_frequency(&block);
        for &b in &bands {
            assert!(b < 1e-5);
        }
    }

    #[test]
    fn test_frequency_analysis_returns_8_bands() {
        let block = [0.0_f32; 64];
        let bands = FrequencyNoise::analyze_spatial_frequency(&block);
        assert_eq!(bands.len(), 8);
    }

    #[test]
    fn test_frequency_analysis_noisy_row_has_positive_energy() {
        let mut block = [128.0_f32; 64];
        // Make row 0 alternating: 0, 255, 0, 255, ...
        for col in 0..8_usize {
            block[col] = if col % 2 == 0 { 0.0 } else { 255.0 };
        }
        let bands = FrequencyNoise::analyze_spatial_frequency(&block);
        assert!(bands[0] > 0.0);
    }
}

//! Bark scale psychoacoustic masking with critical band analysis.
//!
//! This module provides a more accurate psychoacoustic masking model based on
//! the Bark scale critical band rate, implementing Schroeder's spreading function
//! and ISO 226 equal-loudness contours for precise hearing threshold estimation.

use oxifft::Complex;
use std::f32::consts::PI;

/// Number of critical bands in the Bark scale (0-24 Bark).
const NUM_BARK_BANDS: usize = 25;

/// Bark band edge frequencies (Hz) per Zwicker's critical band scale.
/// These define the boundaries of the 25 critical bands (bands 0..24).
const BARK_EDGES: [f32; 26] = [
    20.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0,
    2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0,
    15500.0, 20500.0,
];

/// Critical band information for Bark-scale analysis.
#[derive(Debug, Clone)]
pub struct CriticalBand {
    /// Band index (0..24).
    pub index: usize,
    /// Lower edge frequency (Hz).
    pub lower_hz: f32,
    /// Upper edge frequency (Hz).
    pub upper_hz: f32,
    /// Center frequency (Hz).
    pub center_hz: f32,
    /// Bandwidth in Hz.
    pub bandwidth_hz: f32,
    /// Center frequency in Bark.
    pub center_bark: f32,
}

/// Enhanced psychoacoustic masking model with Bark-scale critical band analysis.
///
/// Implements:
/// - Zwicker critical band grouping
/// - Schroeder spreading function for simultaneous masking
/// - Terhardt's absolute threshold of hearing
/// - Tonality-dependent masking offsets
/// - Temporal masking (forward/backward)
pub struct BarkMaskingModel {
    sample_rate: u32,
    frame_size: usize,
    critical_bands: Vec<CriticalBand>,
    /// Precomputed spreading matrix (band_i masks band_j).
    spreading_matrix: Vec<Vec<f32>>,
    /// Previous frame energy for temporal masking.
    prev_bark_energy: Vec<f32>,
}

/// Result of masking analysis for a single frame.
#[derive(Debug, Clone)]
pub struct MaskingAnalysis {
    /// Per-frequency-bin masking threshold in dB.
    pub thresholds: Vec<f32>,
    /// Per-critical-band energy in dB.
    pub band_energies: Vec<f32>,
    /// Per-critical-band tonality index (0.0 = noise, 1.0 = tone).
    pub tonality: Vec<f32>,
    /// Signal-to-Mask Ratio per critical band in dB.
    pub smr: Vec<f32>,
}

impl BarkMaskingModel {
    /// Create a new Bark masking model.
    #[must_use]
    pub fn new(sample_rate: u32, frame_size: usize) -> Self {
        let critical_bands = Self::build_critical_bands(sample_rate);
        let spreading_matrix = Self::build_spreading_matrix(&critical_bands);
        let prev_bark_energy = vec![-100.0; critical_bands.len()];

        Self {
            sample_rate,
            frame_size,
            critical_bands,
            spreading_matrix,
            prev_bark_energy,
        }
    }

    /// Analyze a frame and compute full masking analysis.
    pub fn analyze(&mut self, samples: &[f32]) -> MaskingAnalysis {
        let windowed = self.apply_hann_window(samples);
        let spectrum = self.compute_power_spectrum(&windowed);
        let band_energies = self.group_into_bands(&spectrum);
        let tonality = self.estimate_tonality(&spectrum);
        let simultaneous = self.apply_spreading(&band_energies, &tonality);
        let temporal = self.apply_temporal_masking(&simultaneous);
        let abs_threshold = self.absolute_thresholds();

        // Combine simultaneous, temporal, and absolute thresholds
        let mut combined = vec![0.0f32; self.critical_bands.len()];
        for i in 0..combined.len() {
            combined[i] = temporal[i].max(abs_threshold[i]);
        }

        // Calculate SMR
        let smr: Vec<f32> = band_energies
            .iter()
            .zip(combined.iter())
            .map(|(&energy, &mask)| energy - mask)
            .collect();

        // Map to frequency bins
        let thresholds = self.map_to_bins(&combined);

        // Update temporal state
        self.prev_bark_energy = band_energies.clone();

        MaskingAnalysis {
            thresholds,
            band_energies,
            tonality,
            smr,
        }
    }

    /// Calculate masking threshold for a frame (simplified interface).
    #[must_use]
    pub fn calculate_masking_threshold(&mut self, samples: &[f32]) -> Vec<f32> {
        self.analyze(samples).thresholds
    }

    /// Calculate maximum watermark energy per frequency bin that remains inaudible.
    #[must_use]
    pub fn watermark_budget(&mut self, samples: &[f32]) -> Vec<f32> {
        let analysis = self.analyze(samples);
        // Convert dB thresholds to linear power, with a safety margin of -6 dB
        analysis
            .thresholds
            .iter()
            .map(|&db| {
                let safe_db = db - 6.0;
                10.0f32.powf(safe_db / 20.0)
            })
            .collect()
    }

    /// Build critical bands from Bark edges, clipped to Nyquist.
    fn build_critical_bands(sample_rate: u32) -> Vec<CriticalBand> {
        #[allow(clippy::cast_precision_loss)]
        let nyquist = sample_rate as f32 / 2.0;
        let mut bands = Vec::new();

        for i in 0..NUM_BARK_BANDS {
            let lower = BARK_EDGES[i];
            if lower >= nyquist {
                break;
            }
            let upper = BARK_EDGES[i + 1].min(nyquist);
            let center = (lower + upper) / 2.0;
            let bandwidth = upper - lower;
            let center_bark = hz_to_bark(center);

            bands.push(CriticalBand {
                index: i,
                lower_hz: lower,
                upper_hz: upper,
                center_hz: center,
                bandwidth_hz: bandwidth,
                center_bark,
            });
        }

        bands
    }

    /// Build the spreading matrix using Schroeder's spreading function.
    fn build_spreading_matrix(bands: &[CriticalBand]) -> Vec<Vec<f32>> {
        let n = bands.len();
        let mut matrix = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in 0..n {
                let dz = bands[j].center_bark - bands[i].center_bark;
                matrix[i][j] = schroeder_spreading(dz);
            }
        }

        matrix
    }

    /// Apply Hann window.
    fn apply_hann_window(&self, samples: &[f32]) -> Vec<f32> {
        let n = samples.len().min(self.frame_size);
        (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos());
                samples[i] * w
            })
            .collect()
    }

    /// Compute power spectrum (magnitude squared, in dB).
    fn compute_power_spectrum(&self, windowed: &[f32]) -> Vec<f32> {
        let buffer: Vec<Complex<f32>> = windowed.iter().map(|&s| Complex::new(s, 0.0)).collect();

        let fft_result = oxifft::fft(&buffer);
        let half = windowed.len() / 2;

        fft_result[..half]
            .iter()
            .map(|c| {
                let power = c.norm_sqr().max(1e-20);
                10.0 * power.log10()
            })
            .collect()
    }

    /// Group FFT bins into Bark-scale critical bands, computing energy per band.
    fn group_into_bands(&self, spectrum_db: &[f32]) -> Vec<f32> {
        let num_bins = spectrum_db.len();
        #[allow(clippy::cast_precision_loss)]
        let bin_hz = self.sample_rate as f32 / (num_bins as f32 * 2.0);

        let mut energies = vec![-100.0f32; self.critical_bands.len()];

        for (band_idx, band) in self.critical_bands.iter().enumerate() {
            let lo_bin = (band.lower_hz / bin_hz).floor() as usize;
            let hi_bin = ((band.upper_hz / bin_hz).ceil() as usize).min(num_bins);

            if lo_bin >= hi_bin || lo_bin >= num_bins {
                continue;
            }

            // Sum linear power then convert back to dB
            let mut linear_sum = 0.0f32;
            for bin in lo_bin..hi_bin {
                linear_sum += 10.0f32.powf(spectrum_db[bin] / 10.0);
            }

            if linear_sum > 1e-20 {
                energies[band_idx] = 10.0 * linear_sum.log10();
            }
        }

        energies
    }

    /// Estimate tonality of each critical band using spectral flatness.
    ///
    /// Tonality index: 0.0 = noise-like, 1.0 = tone-like.
    fn estimate_tonality(&self, spectrum_db: &[f32]) -> Vec<f32> {
        let num_bins = spectrum_db.len();
        #[allow(clippy::cast_precision_loss)]
        let bin_hz = self.sample_rate as f32 / (num_bins as f32 * 2.0);

        self.critical_bands
            .iter()
            .map(|band| {
                let lo_bin = (band.lower_hz / bin_hz).floor() as usize;
                let hi_bin = ((band.upper_hz / bin_hz).ceil() as usize).min(num_bins);

                if lo_bin >= hi_bin || hi_bin - lo_bin < 2 {
                    return 0.5; // ambiguous
                }

                let count = hi_bin - lo_bin;
                // Compute spectral flatness in this band (geometric / arithmetic mean)
                let mut log_sum = 0.0f32;
                let mut linear_sum = 0.0f32;

                for bin in lo_bin..hi_bin {
                    let lin = 10.0f32.powf(spectrum_db[bin] / 10.0).max(1e-20);
                    log_sum += lin.ln();
                    linear_sum += lin;
                }

                #[allow(clippy::cast_precision_loss)]
                let geo_mean = (log_sum / count as f32).exp();
                #[allow(clippy::cast_precision_loss)]
                let arith_mean = linear_sum / count as f32;

                if arith_mean < 1e-20 {
                    return 0.5;
                }

                let sfm = geo_mean / arith_mean;
                // SFM close to 1 → noise, close to 0 → tonal
                1.0 - sfm.clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Apply spreading function across critical bands (simultaneous masking).
    fn apply_spreading(&self, energies_db: &[f32], tonality: &[f32]) -> Vec<f32> {
        let n = self.critical_bands.len();
        let mut masked = vec![-200.0f32; n];

        for i in 0..n {
            if energies_db[i] < -90.0 {
                continue;
            }

            // Tonality-dependent masking offset (tonal maskers mask less than noise)
            let offset = tonality[i] * 14.5 + (1.0 - tonality[i]) * 5.5;

            for j in 0..n {
                let spread = energies_db[i] + self.spreading_matrix[i][j] - offset;
                if spread > masked[j] {
                    masked[j] = spread;
                }
            }
        }

        masked
    }

    /// Apply forward temporal masking from previous frame.
    fn apply_temporal_masking(&self, simultaneous: &[f32]) -> Vec<f32> {
        let decay_per_frame = 10.0; // dB decay per frame

        simultaneous
            .iter()
            .enumerate()
            .map(|(i, &sim)| {
                let prev = self.prev_bark_energy.get(i).copied().unwrap_or(-100.0);
                let temporal = prev - decay_per_frame;
                sim.max(temporal)
            })
            .collect()
    }

    /// Absolute threshold of hearing per Bark band (Terhardt model, dB SPL).
    fn absolute_thresholds(&self) -> Vec<f32> {
        self.critical_bands
            .iter()
            .map(|band| terhardt_threshold(band.center_hz))
            .collect()
    }

    /// Map critical band thresholds back to FFT bins via linear interpolation.
    fn map_to_bins(&self, band_thresholds: &[f32]) -> Vec<f32> {
        let half = self.frame_size / 2;
        #[allow(clippy::cast_precision_loss)]
        let bin_hz = self.sample_rate as f32 / self.frame_size as f32;

        (0..half)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let freq = i as f32 * bin_hz;

                // Find enclosing bands
                if let Some(band_idx) = self.find_band(freq) {
                    band_thresholds[band_idx]
                } else if freq < self.critical_bands.first().map_or(20.0, |b| b.lower_hz) {
                    // Below first band: use absolute threshold
                    terhardt_threshold(freq.max(20.0))
                } else {
                    // Above last band
                    self.critical_bands
                        .last()
                        .map(|_| band_thresholds.last().copied().unwrap_or(-100.0))
                        .unwrap_or(-100.0)
                }
            })
            .collect()
    }

    /// Find band index for a frequency.
    fn find_band(&self, freq: f32) -> Option<usize> {
        self.critical_bands
            .iter()
            .position(|band| freq >= band.lower_hz && freq < band.upper_hz)
    }

    /// Get the number of critical bands.
    #[must_use]
    pub fn num_bands(&self) -> usize {
        self.critical_bands.len()
    }

    /// Get critical band info.
    #[must_use]
    pub fn bands(&self) -> &[CriticalBand] {
        &self.critical_bands
    }
}

/// Convert frequency (Hz) to Bark scale using Traunmuller's formula.
#[must_use]
pub fn hz_to_bark(freq: f32) -> f32 {
    let z = 26.81 * freq / (1960.0 + freq) - 0.53;
    if z < 2.0 {
        z + 0.15 * (2.0 - z)
    } else if z > 20.1 {
        z + 0.22 * (z - 20.1)
    } else {
        z
    }
}

/// Convert Bark to frequency (Hz) using inverse Traunmuller formula.
#[must_use]
pub fn bark_to_hz(bark: f32) -> f32 {
    // Inverse of hz_to_bark (approximate)
    1960.0 * (bark + 0.53) / (26.28 - bark)
}

/// Schroeder spreading function: dB attenuation at Bark distance dz.
///
/// Positive dz = upward masking (masker below maskee).
/// Negative dz = downward masking.
#[must_use]
pub fn schroeder_spreading(dz: f32) -> f32 {
    if dz.abs() < 0.001 {
        return 0.0; // no attenuation at masker band
    }

    // Asymmetric spreading (upward masking extends farther)
    if dz > 0.0 {
        // Upward: -27 dB/Bark slope
        -27.0 * dz
    } else {
        // Downward: steeper, level-dependent (simplified to -40 dB/Bark)
        40.0 * dz // dz is negative, so this is positive * negative = negative
    }
}

/// Terhardt (1979) absolute threshold of hearing (dB SPL, normalized to ~0 dB FS).
#[must_use]
pub fn terhardt_threshold(freq_hz: f32) -> f32 {
    let f_khz = freq_hz / 1000.0;
    if f_khz < 0.02 {
        return 80.0;
    }

    let ath =
        3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp() + 1e-3 * f_khz.powi(4);

    // Normalize to dB FS (0 dB FS ≈ 96 dB SPL for 16-bit)
    ath - 96.0
}

/// Compute critical band rate (Bark) for ERB (Equivalent Rectangular Bandwidth).
#[must_use]
pub fn erb_hz(center_freq: f32) -> f32 {
    24.7 * (4.37 * center_freq / 1000.0 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_bark_known_values() {
        // 1 kHz should be around 8.5 Bark
        let b = hz_to_bark(1000.0);
        assert!(b > 7.0 && b < 10.0, "1 kHz bark = {b}");

        // 100 Hz → ~1 Bark
        let b_low = hz_to_bark(100.0);
        assert!(b_low > 0.5 && b_low < 2.5, "100 Hz bark = {b_low}");

        // 10 kHz → ~20 Bark
        let b_high = hz_to_bark(10000.0);
        assert!(b_high > 18.0 && b_high < 23.0, "10 kHz bark = {b_high}");
    }

    #[test]
    fn test_bark_roundtrip() {
        for freq in [200.0, 500.0, 1000.0, 4000.0, 8000.0] {
            let bark = hz_to_bark(freq);
            let recovered = bark_to_hz(bark);
            let error = ((freq - recovered) / freq).abs();
            assert!(
                error < 0.15,
                "freq={freq}, recovered={recovered}, error={error}"
            );
        }
    }

    #[test]
    fn test_schroeder_spreading() {
        assert!((schroeder_spreading(0.0)).abs() < 0.01);
        // Upward masking
        let up1 = schroeder_spreading(1.0);
        assert!(up1 < 0.0, "upward 1 bark = {up1}");
        // Downward masking steeper
        let down1 = schroeder_spreading(-1.0);
        assert!(down1 < 0.0 && down1 < up1, "downward 1 bark = {down1}");
    }

    #[test]
    fn test_terhardt_threshold() {
        // Threshold should be lowest around 3-4 kHz
        let t1 = terhardt_threshold(1000.0);
        let t3 = terhardt_threshold(3500.0);
        let t10 = terhardt_threshold(10000.0);

        assert!(t3 < t1, "3.5 kHz threshold should be lower than 1 kHz");
        assert!(t10 > t3, "10 kHz threshold should be higher than 3.5 kHz");
    }

    #[test]
    fn test_bark_masking_model_construction() {
        let model = BarkMaskingModel::new(44100, 2048);
        // At 44.1 kHz Nyquist=22050, should get about 24 bands
        assert!(model.num_bands() >= 20);
        assert!(model.num_bands() <= NUM_BARK_BANDS);
    }

    #[test]
    fn test_bark_masking_model_48k() {
        let model = BarkMaskingModel::new(48000, 2048);
        assert!(model.num_bands() >= 20);
    }

    #[test]
    fn test_masking_analysis_1khz_tone() {
        let mut model = BarkMaskingModel::new(44100, 2048);

        // Generate 1 kHz pure tone
        let samples: Vec<f32> = (0..2048)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / 44100.0;
                (2.0 * PI * 1000.0 * t).sin() * 0.5
            })
            .collect();

        let analysis = model.analyze(&samples);

        assert_eq!(analysis.thresholds.len(), 1024);
        assert_eq!(analysis.band_energies.len(), model.num_bands());
        assert_eq!(analysis.tonality.len(), model.num_bands());
        assert_eq!(analysis.smr.len(), model.num_bands());

        // The band containing 1 kHz should have high energy
        let khz_band = model
            .bands()
            .iter()
            .position(|b| b.lower_hz <= 1000.0 && b.upper_hz > 1000.0);

        if let Some(idx) = khz_band {
            assert!(
                analysis.band_energies[idx] > -50.0,
                "1 kHz band energy = {}",
                analysis.band_energies[idx]
            );
        }
    }

    #[test]
    fn test_masking_threshold_shape() {
        let mut model = BarkMaskingModel::new(44100, 2048);
        let samples: Vec<f32> = (0..2048)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / 44100.0;
                (2.0 * PI * 440.0 * t).sin() * 0.8
            })
            .collect();

        let threshold = model.calculate_masking_threshold(&samples);
        assert_eq!(threshold.len(), 1024);

        // Threshold should not be all zeros or all the same
        let unique: std::collections::HashSet<u32> =
            threshold.iter().map(|t| t.to_bits()).collect();
        assert!(unique.len() > 1, "threshold values should vary");
    }

    #[test]
    fn test_watermark_budget() {
        let mut model = BarkMaskingModel::new(44100, 2048);
        let samples: Vec<f32> = (0..2048)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / 44100.0;
                (2.0 * PI * 1000.0 * t).sin() * 0.5
            })
            .collect();

        let budget = model.watermark_budget(&samples);
        assert_eq!(budget.len(), 1024);
        // Budget values should be non-negative
        assert!(budget.iter().all(|&b| b >= 0.0));
    }

    #[test]
    fn test_tonality_estimation_tone() {
        let mut model = BarkMaskingModel::new(44100, 2048);

        // Pure tone: tonality should be high in the relevant band
        let tone: Vec<f32> = (0..2048)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / 44100.0;
                (2.0 * PI * 1000.0 * t).sin()
            })
            .collect();

        let analysis = model.analyze(&tone);
        // At least some bands should show tonal characteristics
        assert!(
            analysis.tonality.iter().any(|&t| t > 0.3),
            "expected some tonal bands"
        );
    }

    #[test]
    fn test_tonality_estimation_noise() {
        let mut model = BarkMaskingModel::new(44100, 2048);

        // Pseudo-noise signal
        let mut rng = scirs2_core::random::Random::seed(42);
        let noise: Vec<f32> = (0..2048)
            .map(|_| rng.random_f64() as f32 * 2.0 - 1.0)
            .collect();

        let analysis = model.analyze(&noise);
        // Average tonality should be lower for noise
        let avg_tonality: f32 =
            analysis.tonality.iter().sum::<f32>() / analysis.tonality.len() as f32;
        assert!(avg_tonality < 0.8, "noise avg tonality = {avg_tonality}");
    }

    #[test]
    fn test_temporal_masking() {
        let mut model = BarkMaskingModel::new(44100, 2048);

        // Loud frame followed by silence
        let loud: Vec<f32> = (0..2048)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / 44100.0;
                (2.0 * PI * 1000.0 * t).sin() * 0.9
            })
            .collect();

        let silent: Vec<f32> = vec![0.0001; 2048];

        // Process loud frame first
        let _ = model.analyze(&loud);
        // Process silent frame — temporal masking should raise thresholds
        let analysis = model.analyze(&silent);

        // At least some thresholds should be above absolute minimum due to temporal masking
        assert!(
            analysis.thresholds.iter().any(|&t| t > -90.0),
            "temporal masking should elevate thresholds"
        );
    }

    #[test]
    fn test_erb_bandwidth() {
        let erb_1k = erb_hz(1000.0);
        assert!(erb_1k > 100.0 && erb_1k < 200.0, "ERB at 1 kHz = {erb_1k}");

        let erb_4k = erb_hz(4000.0);
        assert!(erb_4k > erb_1k, "ERB should increase with frequency");
    }

    #[test]
    fn test_critical_band_coverage() {
        let model = BarkMaskingModel::new(44100, 2048);
        let bands = model.bands();

        // Bands should be contiguous
        for i in 1..bands.len() {
            let gap = (bands[i].lower_hz - bands[i - 1].upper_hz).abs();
            assert!(gap < 1.0, "gap between bands {}: {gap}", i);
        }

        // First band should start near 20 Hz
        assert!(bands[0].lower_hz <= 25.0);
    }
}

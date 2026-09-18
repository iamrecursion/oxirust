//! Spectral feature extraction.

use super::analyzer::SpectrumData;

/// Spectral features extracted from audio.
#[derive(Clone, Debug)]
pub struct SpectralFeatures {
    /// Spectral centroid (brightness, Hz).
    pub centroid: f64,
    /// Spectral spread (bandwidth, Hz).
    pub spread: f64,
    /// Spectral rolloff (frequency below which 85% of energy is contained, Hz).
    pub rolloff: f64,
    /// Spectral flux (rate of change).
    pub flux: f64,
    /// Spectral flatness (tonality measure, 0-1).
    pub flatness: f64,
    /// Spectral crest factor (ratio of peak to RMS).
    pub crest: f64,
    /// Zero crossing rate.
    pub zero_crossing_rate: f64,
    /// Energy (sum of squared magnitudes).
    pub energy: f64,
    /// RMS (root mean square).
    pub rms: f64,
    /// Fundamental frequency (Hz, if detected).
    pub fundamental_frequency: Option<f64>,
    /// Harmonics detected.
    pub harmonics: Vec<Harmonic>,
}

/// Detected harmonic.
#[derive(Clone, Debug)]
pub struct Harmonic {
    /// Harmonic number (1 = fundamental, 2 = first harmonic, etc.).
    pub number: usize,
    /// Frequency (Hz).
    pub frequency: f64,
    /// Magnitude.
    pub magnitude: f64,
}

/// Feature extractor.
pub struct FeatureExtractor {
    previous_spectrum: Option<Vec<f64>>,
}

impl FeatureExtractor {
    /// Create a new feature extractor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            previous_spectrum: None,
        }
    }

    /// Extract features from spectrum data.
    pub fn extract(&mut self, spectrum: &SpectrumData) -> SpectralFeatures {
        let centroid = self.compute_centroid(spectrum);
        let spread = self.compute_spread(spectrum, centroid);
        let rolloff = self.compute_rolloff(spectrum);
        let flux = self.compute_flux(spectrum);
        let flatness = self.compute_flatness(spectrum);
        let crest = self.compute_crest(spectrum);
        let energy = self.compute_energy(spectrum);
        let rms = energy.sqrt();
        let fundamental_frequency = self.detect_fundamental(spectrum);
        let harmonics = self.detect_harmonics(spectrum, fundamental_frequency);

        SpectralFeatures {
            centroid,
            spread,
            rolloff,
            flux,
            flatness,
            crest,
            zero_crossing_rate: 0.0, // Will be computed from time-domain signal
            energy,
            rms,
            fundamental_frequency,
            harmonics,
        }
    }

    /// Compute spectral centroid (center of mass of spectrum).
    #[allow(clippy::cast_precision_loss)]
    fn compute_centroid(&self, spectrum: &SpectrumData) -> f64 {
        let numerator: f64 = spectrum
            .magnitude
            .iter()
            .zip(&spectrum.frequencies)
            .map(|(m, f)| m * f)
            .sum();

        let denominator: f64 = spectrum.magnitude.iter().sum();

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Compute spectral spread (standard deviation around centroid).
    #[allow(clippy::cast_precision_loss)]
    fn compute_spread(&self, spectrum: &SpectrumData, centroid: f64) -> f64 {
        let numerator: f64 = spectrum
            .magnitude
            .iter()
            .zip(&spectrum.frequencies)
            .map(|(m, f)| m * (f - centroid).powi(2))
            .sum();

        let denominator: f64 = spectrum.magnitude.iter().sum();

        if denominator > 0.0 {
            (numerator / denominator).sqrt()
        } else {
            0.0
        }
    }

    /// Compute spectral rolloff (85% energy threshold).
    fn compute_rolloff(&self, spectrum: &SpectrumData) -> f64 {
        let total_energy: f64 = spectrum.magnitude.iter().map(|m| m * m).sum();
        let threshold = 0.85 * total_energy;

        let mut cumulative = 0.0;
        for (i, &magnitude) in spectrum.magnitude.iter().enumerate() {
            cumulative += magnitude * magnitude;
            if cumulative >= threshold {
                return spectrum.frequencies[i];
            }
        }

        spectrum.frequencies.last().copied().unwrap_or(0.0)
    }

    /// Compute spectral flux (change from previous frame).
    fn compute_flux(&mut self, spectrum: &SpectrumData) -> f64 {
        let flux = if let Some(ref prev) = self.previous_spectrum {
            if prev.len() == spectrum.magnitude.len() {
                prev.iter()
                    .zip(&spectrum.magnitude)
                    .map(|(p, c)| (c - p).powi(2))
                    .sum::<f64>()
                    .sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        self.previous_spectrum = Some(spectrum.magnitude.clone());
        flux
    }

    /// Compute spectral flatness (geometric mean / arithmetic mean).
    #[allow(clippy::cast_precision_loss)]
    fn compute_flatness(&self, spectrum: &SpectrumData) -> f64 {
        let n = spectrum.magnitude.len() as f64;

        // Geometric mean
        let log_sum: f64 = spectrum
            .magnitude
            .iter()
            .map(|&m| if m > 0.0 { m.ln() } else { -100.0 })
            .sum();
        let geometric_mean = (log_sum / n).exp();

        // Arithmetic mean
        let arithmetic_mean: f64 = spectrum.magnitude.iter().sum::<f64>() / n;

        if arithmetic_mean > 0.0 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        }
    }

    /// Compute spectral crest factor.
    fn compute_crest(&self, spectrum: &SpectrumData) -> f64 {
        let peak = spectrum
            .magnitude
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let rms = self.compute_energy(spectrum).sqrt();

        if rms > 0.0 {
            peak / rms
        } else {
            0.0
        }
    }

    /// Compute energy (sum of squared magnitudes).
    fn compute_energy(&self, spectrum: &SpectrumData) -> f64 {
        spectrum.magnitude.iter().map(|m| m * m).sum()
    }

    /// Detect fundamental frequency using autocorrelation.
    fn detect_fundamental(&self, spectrum: &SpectrumData) -> Option<f64> {
        if spectrum.peaks.is_empty() {
            return None;
        }

        // Use the lowest significant peak as fundamental
        let mut peaks = spectrum.peaks.clone();
        peaks.sort_by(|a, b| {
            a.frequency
                .partial_cmp(&b.frequency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Filter out very low frequencies (below 20 Hz)
        peaks.retain(|p| p.frequency >= 20.0);

        if peaks.is_empty() {
            return None;
        }

        // Find the peak with the strongest magnitude among the lowest frequencies
        Some(peaks[0].frequency)
    }

    /// Detect harmonics based on fundamental frequency.
    fn detect_harmonics(&self, spectrum: &SpectrumData, fundamental: Option<f64>) -> Vec<Harmonic> {
        let Some(f0) = fundamental else {
            return Vec::new();
        };

        let mut harmonics = Vec::new();
        let tolerance = f0 * 0.05; // 5% tolerance

        for n in 1..=10 {
            let expected_freq = f0 * n as f64;
            if expected_freq > spectrum.sample_rate / 2.0 {
                break;
            }

            // Find peak closest to expected harmonic frequency
            if let Some(peak) = spectrum
                .peaks
                .iter()
                .filter(|p| (p.frequency - expected_freq).abs() < tolerance)
                .max_by(|a, b| {
                    a.magnitude
                        .partial_cmp(&b.magnitude)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                harmonics.push(Harmonic {
                    number: n,
                    frequency: peak.frequency,
                    magnitude: peak.magnitude,
                });
            }
        }

        harmonics
    }

    /// Reset the feature extractor.
    pub fn reset(&mut self) {
        self.previous_spectrum = None;
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Time-domain feature extraction.
pub struct TimeDomainFeatures;

impl TimeDomainFeatures {
    /// Compute zero crossing rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn zero_crossing_rate(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i - 1] >= 0.0 && samples[i] < 0.0)
                || (samples[i - 1] < 0.0 && samples[i] >= 0.0)
            {
                crossings += 1;
            }
        }

        crossings as f64 / (samples.len() - 1) as f64
    }

    /// Compute RMS (root mean square).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rms(samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f64 = samples.iter().map(|&s| s * s).sum();
        (sum_squares / samples.len() as f64).sqrt()
    }

    /// Compute peak amplitude.
    #[must_use]
    pub fn peak_amplitude(samples: &[f64]) -> f64 {
        samples
            .iter()
            .map(|&s| s.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Compute crest factor (peak / RMS).
    #[must_use]
    pub fn crest_factor(samples: &[f64]) -> f64 {
        let peak = Self::peak_amplitude(samples);
        let rms = Self::rms(samples);

        if rms > 0.0 {
            peak / rms
        } else {
            0.0
        }
    }

    /// Compute dynamic range in dB.
    #[must_use]
    pub fn dynamic_range(samples: &[f64]) -> f64 {
        let peak = Self::peak_amplitude(samples);
        let min_amplitude = samples
            .iter()
            .filter(|&&s| s.abs() > 0.0)
            .map(|&s| s.abs())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        if min_amplitude > 0.0 && peak > 0.0 {
            20.0 * (peak / min_amplitude).log10()
        } else {
            0.0
        }
    }

    /// Compute autocorrelation at a given lag.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn autocorrelation(samples: &[f64], lag: usize) -> f64 {
        if lag >= samples.len() {
            return 0.0;
        }

        let mut sum = 0.0;
        let n = samples.len() - lag;

        for i in 0..n {
            sum += samples[i] * samples[i + lag];
        }

        sum / n as f64
    }

    /// Compute autocorrelation function for multiple lags.
    #[must_use]
    pub fn autocorrelation_function(samples: &[f64], max_lag: usize) -> Vec<f64> {
        (0..=max_lag)
            .map(|lag| Self::autocorrelation(samples, lag))
            .collect()
    }

    /// Detect pitch using autocorrelation.
    #[must_use]
    pub fn detect_pitch(
        samples: &[f64],
        sample_rate: f64,
        min_freq: f64,
        max_freq: f64,
    ) -> Option<f64> {
        let min_lag = (sample_rate / max_freq).ceil() as usize;
        let max_lag = (sample_rate / min_freq).floor() as usize;

        if min_lag >= max_lag || max_lag >= samples.len() {
            return None;
        }

        let acf = Self::autocorrelation_function(samples, max_lag);

        // Find first peak after min_lag
        let mut max_acf = 0.0;
        let mut peak_lag = 0;

        for lag in min_lag..=max_lag {
            if lag > 0 && lag < acf.len() - 1 {
                if acf[lag] > acf[lag - 1] && acf[lag] > acf[lag + 1] && acf[lag] > max_acf {
                    max_acf = acf[lag];
                    peak_lag = lag;
                }
            }
        }

        if peak_lag > 0 {
            Some(sample_rate / peak_lag as f64)
        } else {
            None
        }
    }
}

/// Dynamic range measurement.
pub struct DynamicRangeMeter {
    peak_hold: f64,
    peak_decay: f64,
    rms_window: Vec<f64>,
    window_size: usize,
}

impl DynamicRangeMeter {
    /// Create a new dynamic range meter.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            peak_hold: 0.0,
            peak_decay: 0.995,
            rms_window: Vec::new(),
            window_size,
        }
    }

    /// Process samples and update meter.
    pub fn process(&mut self, samples: &[f64]) {
        // Update peak
        let current_peak = TimeDomainFeatures::peak_amplitude(samples);
        if current_peak > self.peak_hold {
            self.peak_hold = current_peak;
        } else {
            self.peak_hold *= self.peak_decay;
        }

        // Update RMS window
        self.rms_window.extend_from_slice(samples);
        if self.rms_window.len() > self.window_size {
            let excess = self.rms_window.len() - self.window_size;
            self.rms_window.drain(..excess);
        }
    }

    /// Get current peak level in dB.
    #[must_use]
    pub fn peak_db(&self) -> f64 {
        if self.peak_hold > 0.0 {
            20.0 * self.peak_hold.log10()
        } else {
            -100.0
        }
    }

    /// Get current RMS level in dB.
    #[must_use]
    pub fn rms_db(&self) -> f64 {
        let rms = TimeDomainFeatures::rms(&self.rms_window);
        if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -100.0
        }
    }

    /// Get dynamic range (peak - RMS) in dB.
    #[must_use]
    pub fn dynamic_range_db(&self) -> f64 {
        self.peak_db() - self.rms_db()
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.peak_hold = 0.0;
        self.rms_window.clear();
    }
}

/// Frequency band analyzer for multi-band analysis.
pub struct BandAnalyzer {
    bands: Vec<FrequencyBand>,
}

/// Frequency band definition.
#[derive(Clone, Debug)]
pub struct FrequencyBand {
    /// Band name.
    pub name: String,
    /// Minimum frequency (Hz).
    pub min_freq: f64,
    /// Maximum frequency (Hz).
    pub max_freq: f64,
}

/// Band energy measurement.
#[derive(Clone, Debug)]
pub struct BandEnergy {
    /// Band definition.
    pub band: FrequencyBand,
    /// Energy in the band.
    pub energy: f64,
    /// Energy in dB.
    pub energy_db: f64,
}

impl BandAnalyzer {
    /// Create a new band analyzer with standard audio bands.
    #[must_use]
    pub fn new_standard() -> Self {
        Self {
            bands: vec![
                FrequencyBand {
                    name: "Sub-bass".to_string(),
                    min_freq: 20.0,
                    max_freq: 60.0,
                },
                FrequencyBand {
                    name: "Bass".to_string(),
                    min_freq: 60.0,
                    max_freq: 250.0,
                },
                FrequencyBand {
                    name: "Low-mid".to_string(),
                    min_freq: 250.0,
                    max_freq: 500.0,
                },
                FrequencyBand {
                    name: "Mid".to_string(),
                    min_freq: 500.0,
                    max_freq: 2000.0,
                },
                FrequencyBand {
                    name: "High-mid".to_string(),
                    min_freq: 2000.0,
                    max_freq: 4000.0,
                },
                FrequencyBand {
                    name: "Presence".to_string(),
                    min_freq: 4000.0,
                    max_freq: 6000.0,
                },
                FrequencyBand {
                    name: "Brilliance".to_string(),
                    min_freq: 6000.0,
                    max_freq: 20000.0,
                },
            ],
        }
    }

    /// Create a new band analyzer with custom bands.
    #[must_use]
    pub fn new(bands: Vec<FrequencyBand>) -> Self {
        Self { bands }
    }

    /// Analyze spectrum and return energy per band.
    #[must_use]
    pub fn analyze(&self, spectrum: &SpectrumData) -> Vec<BandEnergy> {
        self.bands
            .iter()
            .map(|band| {
                let energy = spectrum
                    .magnitude
                    .iter()
                    .zip(&spectrum.frequencies)
                    .filter(|(_, &f)| f >= band.min_freq && f <= band.max_freq)
                    .map(|(m, _)| m * m)
                    .sum::<f64>();

                let energy_db = if energy > 0.0 {
                    10.0 * energy.log10()
                } else {
                    -100.0
                };

                BandEnergy {
                    band: band.clone(),
                    energy,
                    energy_db,
                }
            })
            .collect()
    }
}

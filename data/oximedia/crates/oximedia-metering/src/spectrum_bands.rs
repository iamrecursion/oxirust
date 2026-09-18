//! Spectrum analyzer metering with octave and 1/3-octave band analysis.
//!
//! Provides band-based spectrum analysis using the Goertzel algorithm
//! for efficient per-band magnitude computation.

#![allow(dead_code)]

use std::f64::consts::PI;

/// A spectral frequency band definition.
#[derive(Clone, Debug)]
pub struct SpectrumBand {
    /// Center frequency in Hz.
    pub center_hz: f64,
    /// Lower edge frequency in Hz.
    pub low_hz: f64,
    /// Upper edge frequency in Hz.
    pub high_hz: f64,
    /// Human-readable label (e.g., "1kHz").
    pub label: String,
}

impl SpectrumBand {
    /// Create a new spectrum band.
    #[must_use]
    pub fn new(center_hz: f64, low_hz: f64, high_hz: f64, label: impl Into<String>) -> Self {
        Self {
            center_hz,
            low_hz,
            high_hz,
            label: label.into(),
        }
    }
}

/// Generate standard 1/1 octave bands from 31.5 Hz to 16 kHz.
///
/// Returns 10 bands: 31.5, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz.
#[must_use]
pub fn octave_bands() -> Vec<SpectrumBand> {
    let centers = [
        (31.5, "31.5Hz"),
        (63.0, "63Hz"),
        (125.0, "125Hz"),
        (250.0, "250Hz"),
        (500.0, "500Hz"),
        (1000.0, "1kHz"),
        (2000.0, "2kHz"),
        (4000.0, "4kHz"),
        (8000.0, "8kHz"),
        (16000.0, "16kHz"),
    ];
    let factor = 2.0_f64.sqrt(); // sqrt(2) for octave edges
    centers
        .iter()
        .map(|&(center, label)| SpectrumBand {
            center_hz: center,
            low_hz: center / factor,
            high_hz: center * factor,
            label: label.to_string(),
        })
        .collect()
}

/// Generate standard 1/3 octave bands from 20 Hz to 20 kHz.
///
/// Returns bands centered at the ISO 266 preferred frequencies.
#[must_use]
pub fn third_octave_bands() -> Vec<SpectrumBand> {
    let centers = [
        (20.0, "20Hz"),
        (25.0, "25Hz"),
        (31.5, "31.5Hz"),
        (40.0, "40Hz"),
        (50.0, "50Hz"),
        (63.0, "63Hz"),
        (80.0, "80Hz"),
        (100.0, "100Hz"),
        (125.0, "125Hz"),
        (160.0, "160Hz"),
        (200.0, "200Hz"),
        (250.0, "250Hz"),
        (315.0, "315Hz"),
        (400.0, "400Hz"),
        (500.0, "500Hz"),
        (630.0, "630Hz"),
        (800.0, "800Hz"),
        (1000.0, "1kHz"),
        (1250.0, "1.25kHz"),
        (1600.0, "1.6kHz"),
        (2000.0, "2kHz"),
        (2500.0, "2.5kHz"),
        (3150.0, "3.15kHz"),
        (4000.0, "4kHz"),
        (5000.0, "5kHz"),
        (6300.0, "6.3kHz"),
        (8000.0, "8kHz"),
        (10000.0, "10kHz"),
        (12500.0, "12.5kHz"),
        (16000.0, "16kHz"),
        (20000.0, "20kHz"),
    ];
    // 1/3 octave: edges are center * 2^(±1/6)
    let factor = 2.0_f64.powf(1.0 / 6.0);
    centers
        .iter()
        .map(|&(center, label)| SpectrumBand {
            center_hz: center,
            low_hz: center / factor,
            high_hz: center * factor,
            label: label.to_string(),
        })
        .collect()
}

/// Analyze the magnitude at a specific frequency using the Goertzel algorithm.
///
/// The Goertzel algorithm is efficient for computing the DFT at a single frequency.
///
/// # Arguments
///
/// * `samples` - Audio samples
/// * `freq_hz` - Target frequency in Hz
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// Magnitude (linear, not normalized)
#[must_use]
pub fn dft_magnitude_at_freq(samples: &[f64], freq_hz: f64, sample_rate: f64) -> f64 {
    let n = samples.len();
    if n == 0 || sample_rate <= 0.0 || freq_hz <= 0.0 {
        return 0.0;
    }

    let normalized_freq = freq_hz / sample_rate;
    let omega = 2.0 * PI * normalized_freq;
    let coeff = 2.0 * omega.cos();

    let mut s_prev2 = 0.0f64;
    let mut s_prev1 = 0.0f64;

    for &sample in samples {
        let s = sample + coeff * s_prev1 - s_prev2;
        s_prev2 = s_prev1;
        s_prev1 = s;
    }

    // Compute magnitude from the last two state values
    let real = s_prev1 - s_prev2 * omega.cos();
    let imag = s_prev2 * omega.sin();
    (real * real + imag * imag).sqrt() / n as f64
}

/// A snapshot of spectrum analysis results.
#[derive(Clone, Debug)]
pub struct SpectrumFrame {
    /// Band levels in dBFS.
    pub band_levels_db: Vec<f64>,
    /// Peak hold levels in dBFS.
    pub peaks_db: Vec<f64>,
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
}

impl SpectrumFrame {
    /// Create a new spectrum frame.
    #[must_use]
    pub fn new(band_levels_db: Vec<f64>, peaks_db: Vec<f64>, timestamp_ms: u64) -> Self {
        Self {
            band_levels_db,
            peaks_db,
            timestamp_ms,
        }
    }

    /// Number of bands in this frame.
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.band_levels_db.len()
    }
}

/// Band-based spectrum analyzer using Goertzel algorithm.
pub struct SpectrumBandAnalyzer {
    /// Frequency bands to analyze.
    pub bands: Vec<SpectrumBand>,
    /// Peak hold time in milliseconds.
    pub peak_hold_ms: u64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Stored peak levels per band.
    peak_levels: Vec<f64>,
    /// Peak hold timestamps per band.
    peak_timestamps: Vec<u64>,
}

impl SpectrumBandAnalyzer {
    /// Create a new spectrum analyzer with specified bands.
    ///
    /// # Arguments
    ///
    /// * `bands` - Frequency bands to analyze
    /// * `sample_rate` - Sample rate in Hz
    /// * `peak_hold_ms` - Peak hold time in milliseconds
    #[must_use]
    pub fn new(bands: Vec<SpectrumBand>, sample_rate: f64, peak_hold_ms: u64) -> Self {
        let n = bands.len();
        Self {
            peak_levels: vec![f64::NEG_INFINITY; n],
            peak_timestamps: vec![0; n],
            bands,
            sample_rate,
            peak_hold_ms,
        }
    }

    /// Create a 1/1 octave spectrum analyzer at the given sample rate.
    #[must_use]
    pub fn octave(sample_rate: f64) -> Self {
        let bands = octave_bands();
        let n = bands.len();
        Self {
            peak_levels: vec![f64::NEG_INFINITY; n],
            peak_timestamps: vec![0; n],
            bands,
            sample_rate,
            peak_hold_ms: 2000,
        }
    }

    /// Create a 1/3 octave spectrum analyzer at the given sample rate.
    #[must_use]
    pub fn third_octave(sample_rate: f64) -> Self {
        let bands = third_octave_bands();
        let n = bands.len();
        Self {
            peak_levels: vec![f64::NEG_INFINITY; n],
            peak_timestamps: vec![0; n],
            bands,
            sample_rate,
            peak_hold_ms: 2000,
        }
    }

    /// Analyze a block of samples and return a spectrum frame.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples to analyze
    /// * `timestamp_ms` - Timestamp of this block in milliseconds
    pub fn analyze(&mut self, samples: &[f64], timestamp_ms: u64) -> SpectrumFrame {
        let mut band_levels_db = Vec::with_capacity(self.bands.len());

        for (i, band) in self.bands.iter().enumerate() {
            let mag = dft_magnitude_at_freq(samples, band.center_hz, self.sample_rate);
            let db = if mag <= 0.0 {
                -120.0
            } else {
                (20.0 * mag.log10()).max(-120.0)
            };
            band_levels_db.push(db);

            // Update peak hold
            if db > self.peak_levels[i] {
                self.peak_levels[i] = db;
                self.peak_timestamps[i] = timestamp_ms + self.peak_hold_ms;
            } else if timestamp_ms > self.peak_timestamps[i] {
                // Decay peak slowly
                self.peak_levels[i] = (self.peak_levels[i] - 0.5).max(-120.0);
            }
        }

        let peaks_db = self.peak_levels.clone();
        SpectrumFrame {
            band_levels_db,
            peaks_db,
            timestamp_ms,
        }
    }

    /// Number of bands in this analyzer.
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }
}

/// Convert linear magnitude to dBFS.
#[must_use]
pub fn magnitude_to_db(linear: f64) -> f64 {
    if linear <= 0.0 {
        -120.0
    } else {
        (20.0 * linear.log10()).max(-120.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_at(freq: f64, sample_rate: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_octave_bands_count() {
        let bands = octave_bands();
        assert_eq!(bands.len(), 10);
    }

    #[test]
    fn test_octave_bands_centers() {
        let bands = octave_bands();
        assert!((bands[0].center_hz - 31.5).abs() < 0.1);
        assert!((bands[5].center_hz - 1000.0).abs() < 0.1);
        assert!((bands[9].center_hz - 16000.0).abs() < 0.1);
    }

    #[test]
    fn test_third_octave_bands_count() {
        let bands = third_octave_bands();
        assert_eq!(bands.len(), 31);
    }

    #[test]
    fn test_third_octave_bands_labels() {
        let bands = third_octave_bands();
        assert_eq!(bands[17].label, "1kHz");
    }

    #[test]
    fn test_dft_magnitude_at_freq_pure_sine() {
        let sr = 48000.0;
        let freq = 1000.0;
        let n = 4800;
        let samples = sine_at(freq, sr, n);
        let mag = dft_magnitude_at_freq(&samples, freq, sr);
        // Should detect significant energy at 1kHz
        assert!(
            mag > 0.1,
            "Expected significant magnitude at {freq}Hz, got {mag}"
        );
    }

    #[test]
    fn test_dft_magnitude_at_freq_silence() {
        let samples = vec![0.0f64; 4800];
        let mag = dft_magnitude_at_freq(&samples, 1000.0, 48000.0);
        assert!((mag - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dft_magnitude_at_freq_empty() {
        let mag = dft_magnitude_at_freq(&[], 1000.0, 48000.0);
        assert_eq!(mag, 0.0);
    }

    #[test]
    fn test_dft_magnitude_frequency_discrimination() {
        let sr = 48000.0;
        let n = 4800;
        // 1kHz sine: should have much more energy at 1kHz than at 100Hz
        let samples = sine_at(1000.0, sr, n);
        let mag_1k = dft_magnitude_at_freq(&samples, 1000.0, sr);
        let mag_100 = dft_magnitude_at_freq(&samples, 100.0, sr);
        assert!(mag_1k > mag_100 * 5.0, "1kHz: {mag_1k}, 100Hz: {mag_100}");
    }

    #[test]
    fn test_spectrum_band_analyzer_octave_band_count() {
        let analyzer = SpectrumBandAnalyzer::octave(48000.0);
        assert_eq!(analyzer.band_count(), 10);
    }

    #[test]
    fn test_spectrum_band_analyzer_third_octave_band_count() {
        let analyzer = SpectrumBandAnalyzer::third_octave(48000.0);
        assert_eq!(analyzer.band_count(), 31);
    }

    #[test]
    fn test_spectrum_frame_analyze_returns_correct_band_count() {
        let mut analyzer = SpectrumBandAnalyzer::octave(48000.0);
        let samples = sine_at(1000.0, 48000.0, 4800);
        let frame = analyzer.analyze(&samples, 0);
        assert_eq!(frame.band_count(), 10);
        assert_eq!(frame.peaks_db.len(), 10);
    }

    #[test]
    fn test_spectrum_band_analyzer_detects_1khz() {
        let mut analyzer = SpectrumBandAnalyzer::octave(48000.0);
        let samples = sine_at(1000.0, 48000.0, 4800);
        let frame = analyzer.analyze(&samples, 0);
        // Band index 5 = 1kHz
        let level_1k = frame.band_levels_db[5];
        // Should be significantly above minimum
        assert!(level_1k > -80.0, "1kHz level: {level_1k} dB");
    }

    #[test]
    fn test_spectrum_frame_new() {
        let frame = SpectrumFrame::new(vec![-12.0, -24.0], vec![-6.0, -18.0], 1000);
        assert_eq!(frame.band_count(), 2);
        assert_eq!(frame.timestamp_ms, 1000);
    }

    #[test]
    fn test_magnitude_to_db_zero() {
        assert_eq!(magnitude_to_db(0.0), -120.0);
    }

    #[test]
    fn test_magnitude_to_db_unity() {
        assert!((magnitude_to_db(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectrum_band_new() {
        let band = SpectrumBand::new(1000.0, 707.0, 1414.0, "1kHz");
        assert_eq!(band.label, "1kHz");
        assert!((band.center_hz - 1000.0).abs() < 0.1);
    }
}

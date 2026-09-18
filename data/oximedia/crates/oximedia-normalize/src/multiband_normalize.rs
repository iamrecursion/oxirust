#![allow(dead_code)]
//! Multi-band loudness normalization.
//!
//! Splits audio into frequency bands using first-order IIR low-pass and
//! high-pass filters, measures per-band RMS loudness, and applies
//! independent per-band gains to achieve a uniform spectral energy balance.
//!
//! # Bands (defaults)
//!
//! | Band  | Range          | Purpose                             |
//! |-------|----------------|-------------------------------------|
//! | Sub   | 20 – 200 Hz    | Sub-bass / rumble                   |
//! | Low   | 200 – 2 000 Hz | Fundamental body / intelligibility  |
//! | Mid   | 2 000 – 8 000 Hz | Presence / sibilance              |
//! | High  | 8 000 – 20 000 Hz | Air / brilliance                 |
//!
//! # Filter implementation
//!
//! The band-splitter uses cascaded first-order Butterworth IIR sections
//! (one low-pass + one high-pass per band).  The output is the signal
//! passing through *both* filters, giving a crude bandpass characteristic.
//! Gains are applied to each band signal, and the bands are then summed.
//!
//! The filter coefficients are derived analytically from the bilinear
//! transform of a first-order RC-prototype Butterworth low-pass/high-pass.

/// A single frequency band specification.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyBand {
    /// Descriptive name (e.g. "sub", "low", "mid", "high").
    pub name: String,
    /// Lower cutoff frequency in Hz.
    pub low_hz: f32,
    /// Upper cutoff frequency in Hz.
    pub high_hz: f32,
    /// Target loudness for this band in LUFS.
    pub target_lufs: f32,
    /// Computed gain in dB (set by [`MultibandNormalizer::analyze`]).
    pub gain_db: f32,
}

impl FrequencyBand {
    /// Create a new frequency band definition with zero gain.
    pub fn new(name: &str, low_hz: f32, high_hz: f32, target_lufs: f32) -> Self {
        Self {
            name: name.to_string(),
            low_hz,
            high_hz,
            target_lufs,
            gain_db: 0.0,
        }
    }
}

/// First-order IIR filter state (single-channel, single-section).
#[derive(Debug, Clone, Copy)]
struct Iir1State {
    // Coefficients: y[n] = b0·x[n] + b1·x[n-1] + a1·y[n-1]
    b0: f32,
    b1: f32,
    a1: f32,
    // Delay registers
    x_prev: f32,
    y_prev: f32,
}

impl Iir1State {
    /// Build a first-order Butterworth **low-pass** filter.
    ///
    /// `fc_hz` is the –3 dB cutoff; `sample_rate` is in Hz.
    fn low_pass(fc_hz: f32, sample_rate: u32) -> Self {
        let fs = sample_rate as f32;
        // Bilinear transform pre-warped frequency
        let omega_c = 2.0 * fs * (std::f32::consts::PI * fc_hz / fs).tan();
        let k = omega_c / (2.0 * fs);
        let denom = 1.0 + k;
        let b0 = k / denom;
        let b1 = k / denom;
        let a1 = (k - 1.0) / denom;
        Self {
            b0,
            b1,
            a1,
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    /// Build a first-order Butterworth **high-pass** filter.
    fn high_pass(fc_hz: f32, sample_rate: u32) -> Self {
        let fs = sample_rate as f32;
        let omega_c = 2.0 * fs * (std::f32::consts::PI * fc_hz / fs).tan();
        let k = omega_c / (2.0 * fs);
        let denom = 1.0 + k;
        let b0 = 1.0 / denom;
        let b1 = -1.0 / denom;
        let a1 = (k - 1.0) / denom;
        Self {
            b0,
            b1,
            a1,
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    /// Process a single sample.
    #[inline]
    fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x_prev - self.a1 * self.y_prev;
        self.x_prev = x;
        self.y_prev = y;
        y
    }

    /// Process a block of samples in-place.
    fn process_block(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}

/// Extract a frequency band from `samples` using cascaded first-order IIR
/// filters (low-pass then high-pass, or vice-versa).
///
/// Returns the bandpass-filtered signal.
fn extract_band(samples: &[f32], low_hz: f32, high_hz: f32, sample_rate: u32) -> Vec<f32> {
    let fs = sample_rate as f32;
    let nyquist = fs / 2.0;

    // Clamp cutoffs to valid range
    let lp_fc = high_hz.min(nyquist * 0.98).max(1.0);
    let hp_fc = low_hz.min(lp_fc * 0.98).max(1.0);

    let mut buf: Vec<f32> = samples.to_vec();

    // Apply high-pass at the lower cutoff first
    if hp_fc > 1.0 {
        let mut hp = Iir1State::high_pass(hp_fc, sample_rate);
        hp.process_block(&mut buf);
    }

    // Then low-pass at the upper cutoff
    if lp_fc < nyquist {
        let mut lp = Iir1State::low_pass(lp_fc, sample_rate);
        lp.process_block(&mut buf);
    }

    buf
}

/// Compute the RMS power of a sample slice in dBFS (returns -144 for silence).
fn rms_lufs_approx(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -144.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms_sq = sum_sq / samples.len() as f64;
    if rms_sq <= 0.0 {
        return -144.0;
    }
    // LUFS ≈ 10·log10(mean_square) − 0.691
    (10.0 * rms_sq.log10() - 0.691) as f32
}

/// Multi-band loudness normalizer.
///
/// # Example
///
/// ```rust
/// use oximedia_normalize::multiband_normalize::MultibandNormalizer;
///
/// let mut norm = MultibandNormalizer::new(-14.0);
/// let sr = 48000u32;
/// let samples: Vec<f32> = (0..sr as usize)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.3)
///     .collect();
/// let bands = norm.analyze(&samples, sr);
/// assert_eq!(bands.len(), 4);
/// let output = norm.process(&samples, sr);
/// assert_eq!(output.len(), samples.len());
/// ```
#[derive(Debug, Clone)]
pub struct MultibandNormalizer {
    /// Global target loudness (applied when a band has no explicit target).
    pub target_lufs: f32,
    /// Frequency bands.  Populate via [`MultibandNormalizer::new`] or
    /// [`MultibandNormalizer::with_bands`].
    pub bands: Vec<FrequencyBand>,
    /// Maximum per-band gain in dB (safety ceiling).
    pub max_gain_db: f32,
    /// Minimum per-band gain in dB (floor, typically negative).
    pub min_gain_db: f32,
}

impl MultibandNormalizer {
    /// Create a four-band normalizer targeting `target_lufs` for all bands.
    ///
    /// Default bands:
    /// * Sub  20 – 200 Hz
    /// * Low  200 – 2 000 Hz
    /// * Mid  2 000 – 8 000 Hz
    /// * High 8 000 – 20 000 Hz
    pub fn new(target_lufs: f32) -> Self {
        let bands = vec![
            FrequencyBand::new("sub", 20.0, 200.0, target_lufs),
            FrequencyBand::new("low", 200.0, 2_000.0, target_lufs),
            FrequencyBand::new("mid", 2_000.0, 8_000.0, target_lufs),
            FrequencyBand::new("high", 8_000.0, 20_000.0, target_lufs),
        ];
        Self {
            target_lufs,
            bands,
            max_gain_db: 20.0,
            min_gain_db: -40.0,
        }
    }

    /// Replace the default band layout with a custom set of bands.
    pub fn with_bands(mut self, bands: Vec<FrequencyBand>) -> Self {
        self.bands = bands;
        self
    }

    /// Override the per-band gain ceiling (default 20 dB).
    pub fn with_max_gain_db(mut self, max: f32) -> Self {
        self.max_gain_db = max;
        self
    }

    /// Override the per-band gain floor (default −40 dB).
    pub fn with_min_gain_db(mut self, min: f32) -> Self {
        self.min_gain_db = min;
        self
    }

    /// Analyse `samples`, compute per-band RMS loudness, and store the
    /// required gains in each [`FrequencyBand`].
    ///
    /// Returns a clone of the updated band list.
    pub fn analyze(&mut self, samples: &[f32], sample_rate: u32) -> Vec<FrequencyBand> {
        for band in self.bands.iter_mut() {
            let band_samples = extract_band(samples, band.low_hz, band.high_hz, sample_rate);
            let measured_lufs = rms_lufs_approx(&band_samples);
            let raw_gain = band.target_lufs - measured_lufs;
            band.gain_db = raw_gain.clamp(self.min_gain_db, self.max_gain_db);
        }
        self.bands.clone()
    }

    /// Apply per-band gains to `samples` and return the mixed output.
    ///
    /// Each band is extracted, scaled by its stored `gain_db`, and the bands
    /// are summed into the output buffer.  A final pass clips the output to
    /// ±1.0 to prevent digital overflow (soft saturation).
    pub fn process(&self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let mut output = vec![0.0f32; samples.len()];

        for band in &self.bands {
            let band_samples = extract_band(samples, band.low_hz, band.high_hz, sample_rate);
            let linear = 10.0_f32.powf(band.gain_db / 20.0);
            for (out, &bsamp) in output.iter_mut().zip(band_samples.iter()) {
                *out += bsamp * linear;
            }
        }

        // Soft saturation: clip to ±1.0
        for s in output.iter_mut() {
            *s = s.clamp(-1.0, 1.0);
        }

        output
    }

    /// Convenience: analyse then immediately process `samples`.
    pub fn analyze_and_process(&mut self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        self.analyze(samples, sample_rate);
        self.process(samples, sample_rate)
    }

    /// Return per-band loudness measurements without modifying the stored gains.
    pub fn measure_bands(&self, samples: &[f32], sample_rate: u32) -> Vec<(String, f32)> {
        self.bands
            .iter()
            .map(|band| {
                let band_samples = extract_band(samples, band.low_hz, band.high_hz, sample_rate);
                let lufs = rms_lufs_approx(&band_samples);
                (band.name.clone(), lufs)
            })
            .collect()
    }
}

impl Default for MultibandNormalizer {
    fn default() -> Self {
        Self::new(-14.0)
    }
}

// ========================================================================== //
//  Tests                                                                      //
// ========================================================================== //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq: f32, amplitude: f32, sr: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn test_new_creates_four_bands() {
        let norm = MultibandNormalizer::new(-14.0);
        assert_eq!(norm.bands.len(), 4);
        assert_eq!(norm.bands[0].name, "sub");
        assert_eq!(norm.bands[1].name, "low");
        assert_eq!(norm.bands[2].name, "mid");
        assert_eq!(norm.bands[3].name, "high");
    }

    #[test]
    fn test_band_frequency_ranges() {
        let norm = MultibandNormalizer::new(-14.0);
        // Sub band: 20-200 Hz
        assert!((norm.bands[0].low_hz - 20.0).abs() < f32::EPSILON);
        assert!((norm.bands[0].high_hz - 200.0).abs() < f32::EPSILON);
        // High band: 8000-20000 Hz
        assert!((norm.bands[3].low_hz - 8_000.0).abs() < f32::EPSILON);
        assert!((norm.bands[3].high_hz - 20_000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_analyze_sets_gains() {
        let sr = 48000u32;
        let samples = sine_wave(1000.0, 0.3, sr, sr as usize * 2);
        let mut norm = MultibandNormalizer::new(-14.0);
        let bands = norm.analyze(&samples, sr);
        // All gains should be within the clamped range
        for band in &bands {
            assert!(
                band.gain_db >= norm.min_gain_db && band.gain_db <= norm.max_gain_db,
                "Band '{}' gain {} out of range",
                band.name,
                band.gain_db
            );
        }
    }

    #[test]
    fn test_process_output_length() {
        let sr = 48000u32;
        let samples = sine_wave(440.0, 0.5, sr, sr as usize);
        let mut norm = MultibandNormalizer::new(-14.0);
        norm.analyze(&samples, sr);
        let output = norm.process(&samples, sr);
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_process_empty_input() {
        let norm = MultibandNormalizer::new(-14.0);
        let output = norm.process(&[], 48000);
        assert!(output.is_empty());
    }

    #[test]
    fn test_output_clipped_to_unity() {
        // Even with max gain on a loud signal, output must stay ≤ ±1.0
        let sr = 48000u32;
        let loud = sine_wave(1000.0, 0.9, sr, sr as usize);
        let mut norm = MultibandNormalizer::new(-14.0).with_max_gain_db(30.0);
        norm.analyze(&loud, sr);
        let output = norm.process(&loud, sr);
        for &s in &output {
            assert!(
                s.abs() <= 1.0 + f32::EPSILON,
                "Sample {s} exceeds ±1.0 — clipping not applied"
            );
        }
    }

    #[test]
    fn test_iir_low_pass_attenuates_high_freq() {
        let sr = 48000u32;
        // 10 kHz tone filtered by a 1 kHz LP → should be significantly attenuated
        let high_freq = sine_wave(10_000.0, 1.0, sr, sr as usize);
        let mut lp = Iir1State::low_pass(1_000.0, sr);
        let mut filtered = high_freq.clone();
        lp.process_block(&mut filtered);

        let rms_in = rms_lufs_approx(&high_freq);
        let rms_out = rms_lufs_approx(&filtered);
        assert!(
            rms_out < rms_in - 6.0,
            "Low-pass filter should attenuate 10 kHz by at least 6 dB when FC=1 kHz. in={rms_in:.2} out={rms_out:.2}"
        );
    }

    #[test]
    fn test_iir_high_pass_attenuates_low_freq() {
        let sr = 48000u32;
        // 50 Hz tone filtered by a 2 kHz HP → should be significantly attenuated
        let low_freq = sine_wave(50.0, 1.0, sr, sr as usize);
        let mut hp = Iir1State::high_pass(2_000.0, sr);
        let mut filtered = low_freq.clone();
        hp.process_block(&mut filtered);

        let rms_in = rms_lufs_approx(&low_freq);
        let rms_out = rms_lufs_approx(&filtered);
        assert!(
            rms_out < rms_in - 6.0,
            "High-pass filter should attenuate 50 Hz by at least 6 dB when FC=2 kHz. in={rms_in:.2} out={rms_out:.2}"
        );
    }

    #[test]
    fn test_measure_bands_returns_all_bands() {
        let sr = 48000u32;
        let samples = sine_wave(440.0, 0.5, sr, sr as usize);
        let norm = MultibandNormalizer::new(-14.0);
        let measurements = norm.measure_bands(&samples, sr);
        assert_eq!(measurements.len(), 4);
        let names: Vec<&str> = measurements.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"sub"));
        assert!(names.contains(&"low"));
        assert!(names.contains(&"mid"));
        assert!(names.contains(&"high"));
    }

    #[test]
    fn test_custom_bands() {
        let custom_bands = vec![
            FrequencyBand::new("bass", 20.0, 500.0, -16.0),
            FrequencyBand::new("treble", 500.0, 20_000.0, -16.0),
        ];
        let norm = MultibandNormalizer::new(-16.0).with_bands(custom_bands);
        assert_eq!(norm.bands.len(), 2);
        assert_eq!(norm.bands[0].name, "bass");
        assert_eq!(norm.bands[1].name, "treble");
    }

    #[test]
    fn test_analyze_and_process_convenience() {
        let sr = 48000u32;
        let samples = sine_wave(440.0, 0.3, sr, sr as usize);
        let mut norm = MultibandNormalizer::new(-14.0);
        let output = norm.analyze_and_process(&samples, sr);
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_rms_lufs_approx_silence() {
        let silence = vec![0.0f32; 4800];
        let lufs = rms_lufs_approx(&silence);
        assert!(lufs < -100.0, "Silence must have very low LUFS: {lufs}");
    }

    #[test]
    fn test_gain_clamp_boundaries() {
        let sr = 48000u32;
        // Near-silence → max_gain clamp
        let tiny: Vec<f32> = vec![1e-10f32; sr as usize];
        let mut norm = MultibandNormalizer::new(-14.0);
        let bands = norm.analyze(&tiny, sr);
        for band in &bands {
            assert!(band.gain_db <= norm.max_gain_db);
        }

        // Very loud → min_gain clamp
        let loud = sine_wave(440.0, 1.0, sr, sr as usize);
        let bands2 = norm.analyze(&loud, sr);
        for band in &bands2 {
            assert!(band.gain_db >= norm.min_gain_db);
        }
    }
}

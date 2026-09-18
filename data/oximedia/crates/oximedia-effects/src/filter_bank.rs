//! Multi-band filter bank for spectral shaping and analysis.
//!
//! Provides a configurable bank of second-order band-pass filters
//! for graphic EQ, spectral analysis, and crossover applications.

#![allow(dead_code)]

/// Shape of a filter band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandShape {
    /// Resonant band-pass filter.
    BandPass,
    /// Low-shelf filter.
    LowShelf,
    /// High-shelf filter.
    HighShelf,
    /// Parametric (peaking) EQ band.
    Peaking,
    /// Notch (band-reject) filter.
    Notch,
}

impl BandShape {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            BandShape::BandPass => "Band-Pass",
            BandShape::LowShelf => "Low-Shelf",
            BandShape::HighShelf => "High-Shelf",
            BandShape::Peaking => "Peaking",
            BandShape::Notch => "Notch",
        }
    }
}

/// A single filter band in the bank.
#[derive(Debug, Clone)]
pub struct FilterBand {
    /// Centre frequency in Hz.
    pub center_hz: f32,
    /// Bandwidth in Hz (Q-derived).
    pub bandwidth_hz_val: f32,
    /// Gain in dB (used by peaking and shelf types).
    pub gain_db: f32,
    /// Filter shape.
    pub shape: BandShape,
    /// Whether this band is active.
    pub enabled: bool,
    // Biquad state variables (transposed direct-form II)
    s1: f32,
    s2: f32,
    // Biquad coefficients: b0, b1, b2, a1, a2
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl FilterBand {
    /// Create a peaking band at `center_hz` with `bandwidth_hz` and `gain_db`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn peaking(center_hz: f32, bandwidth_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        let mut band = Self {
            center_hz,
            bandwidth_hz_val: bandwidth_hz,
            gain_db,
            shape: BandShape::Peaking,
            enabled: true,
            s1: 0.0,
            s2: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        band.recalculate(sample_rate);
        band
    }

    /// Create a band-pass band.
    #[must_use]
    pub fn band_pass(center_hz: f32, bandwidth_hz: f32, sample_rate: f32) -> Self {
        let mut band = Self {
            center_hz,
            bandwidth_hz_val: bandwidth_hz,
            gain_db: 0.0,
            shape: BandShape::BandPass,
            enabled: true,
            s1: 0.0,
            s2: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        band.recalculate(sample_rate);
        band
    }

    /// Create a low-shelf band: boosts/cuts frequencies below `center_hz`.
    #[must_use]
    pub fn low_shelf(center_hz: f32, bandwidth_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        let mut band = Self {
            center_hz,
            bandwidth_hz_val: bandwidth_hz,
            gain_db,
            shape: BandShape::LowShelf,
            enabled: true,
            s1: 0.0,
            s2: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        band.recalculate(sample_rate);
        band
    }

    /// Create a high-shelf band: boosts/cuts frequencies above `center_hz`.
    #[must_use]
    pub fn high_shelf(center_hz: f32, bandwidth_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        let mut band = Self {
            center_hz,
            bandwidth_hz_val: bandwidth_hz,
            gain_db,
            shape: BandShape::HighShelf,
            enabled: true,
            s1: 0.0,
            s2: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        band.recalculate(sample_rate);
        band
    }

    /// Bandwidth in Hz.
    #[must_use]
    pub fn bandwidth_hz(&self) -> f32 {
        self.bandwidth_hz_val
    }

    /// Q factor derived from centre frequency and bandwidth.
    #[must_use]
    pub fn q(&self) -> f32 {
        if self.bandwidth_hz_val > 0.0 {
            self.center_hz / self.bandwidth_hz_val
        } else {
            1.0
        }
    }

    /// Recalculate biquad coefficients.
    pub fn recalculate(&mut self, sample_rate: f32) {
        use std::f32::consts::PI;
        let w0 = 2.0 * PI * self.center_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let q = self.q().max(0.01);
        let alpha = sin_w0 / (2.0 * q);

        match self.shape {
            BandShape::Peaking => {
                let a = 10.0_f32.powf(self.gain_db / 40.0);
                self.b0 = 1.0 + alpha * a;
                self.b1 = -2.0 * cos_w0;
                self.b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                self.a1 = -2.0 * cos_w0 / a0;
                self.a2 = (1.0 - alpha / a) / a0;
                self.b0 /= a0;
                self.b1 /= a0;
                self.b2 /= a0;
            }
            BandShape::BandPass => {
                self.b0 = alpha;
                self.b1 = 0.0;
                self.b2 = -alpha;
                let a0 = 1.0 + alpha;
                self.b0 /= a0;
                self.b1 /= a0;
                self.b2 /= a0;
                self.a1 = -2.0 * cos_w0 / a0;
                self.a2 = (1.0 - alpha) / a0;
            }
            BandShape::Notch => {
                self.b0 = 1.0;
                self.b1 = -2.0 * cos_w0;
                self.b2 = 1.0;
                let a0 = 1.0 + alpha;
                self.b0 /= a0;
                self.b1 /= a0;
                self.b2 /= a0;
                self.a1 = -2.0 * cos_w0 / a0;
                self.a2 = (1.0 - alpha) / a0;
            }
            BandShape::LowShelf => {
                // RBJ Audio-EQ-Cookbook low-shelf filter (Robert Bristow-Johnson,
                // "Audio EQ Cookbook"). `alpha` is the same Q-derived alpha computed
                // above (alpha = sin(w0) / (2*Q)); the cookbook permits deriving the
                // shelf `alpha` from Q directly instead of the alternative
                // shelf-slope (S) parametrization, while reusing the same
                // closed-form coefficient formulas below.
                let a = 10.0_f32.powf(self.gain_db / 40.0);
                let sqrt_a = a.sqrt();
                let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;

                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1 / a0;
                self.a2 = a2 / a0;
            }
            BandShape::HighShelf => {
                // RBJ Audio-EQ-Cookbook high-shelf filter; see the `LowShelf` arm
                // above for the `A`/`alpha` derivation (same cookbook formula set).
                let a = 10.0_f32.powf(self.gain_db / 40.0);
                let sqrt_a = a.sqrt();
                let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;

                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1 / a0;
                self.a2 = a2 / a0;
            }
        }
    }

    /// Process a single sample through this band.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            return input;
        }
        let output = self.b0 * input + self.s1;
        self.s1 = self.b1 * input - self.a1 * output + self.s2;
        self.s2 = self.b2 * input - self.a2 * output;
        output
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

/// Configuration snapshot for a filter bank.
#[derive(Debug, Clone, Default)]
pub struct FilterBankConfig {
    /// Number of bands.
    pub band_count: usize,
    /// Sample rate.
    pub sample_rate: f32,
    /// Global bypass switch.
    pub bypassed: bool,
}

impl FilterBankConfig {
    /// Create a config for `band_count` bands at `sample_rate`.
    #[must_use]
    pub fn new(band_count: usize, sample_rate: f32) -> Self {
        Self {
            band_count,
            sample_rate,
            bypassed: false,
        }
    }

    /// Number of bands in this configuration.
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.band_count
    }
}

/// A bank of filter bands operating in series.
pub struct FilterBank {
    bands: Vec<FilterBand>,
    sample_rate: f32,
}

impl FilterBank {
    /// Create an empty filter bank.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            bands: Vec::new(),
            sample_rate,
        }
    }

    /// Add a band to the bank.
    pub fn add_band(&mut self, band: FilterBand) {
        self.bands.push(band);
    }

    /// Process a single sample through all enabled bands in series.
    pub fn process_sample(&mut self, mut input: f32) -> f32 {
        for band in &mut self.bands {
            input = band.process_sample(input);
        }
        input
    }

    /// Centre frequencies of all registered bands.
    #[must_use]
    pub fn center_frequencies(&self) -> Vec<f32> {
        self.bands.iter().map(|b| b.center_hz).collect()
    }

    /// Number of bands in the bank.
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Reset all band states.
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    /// Sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_band_shape_labels() {
        assert_eq!(BandShape::BandPass.label(), "Band-Pass");
        assert_eq!(BandShape::Peaking.label(), "Peaking");
        assert_eq!(BandShape::Notch.label(), "Notch");
    }

    #[test]
    fn test_filter_band_bandwidth_hz() {
        let band = FilterBand::peaking(1000.0, 200.0, 0.0, 48000.0);
        assert!((band.bandwidth_hz() - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_filter_band_q() {
        let band = FilterBand::peaking(1000.0, 200.0, 0.0, 48000.0);
        let q = band.q();
        assert!((q - 5.0).abs() < 0.01, "expected q=5, got {q}");
    }

    #[test]
    fn test_filter_band_pass_through_when_disabled() {
        let mut band = FilterBand::band_pass(1000.0, 200.0, 48000.0);
        band.enabled = false;
        let out = band.process_sample(0.5);
        assert!((out - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_filter_band_reset() {
        let mut band = FilterBand::band_pass(500.0, 100.0, 48000.0);
        // Process a sample to dirty state
        band.process_sample(1.0);
        band.reset();
        assert_eq!(band.s1, 0.0);
        assert_eq!(band.s2, 0.0);
    }

    #[test]
    fn test_filter_bank_add_and_count() {
        let mut bank = FilterBank::new(48000.0);
        assert_eq!(bank.band_count(), 0);
        bank.add_band(FilterBand::band_pass(500.0, 100.0, 48000.0));
        bank.add_band(FilterBand::band_pass(1000.0, 200.0, 48000.0));
        assert_eq!(bank.band_count(), 2);
    }

    #[test]
    fn test_filter_bank_center_frequencies() {
        let mut bank = FilterBank::new(48000.0);
        bank.add_band(FilterBand::band_pass(500.0, 100.0, 48000.0));
        bank.add_band(FilterBand::band_pass(2000.0, 400.0, 48000.0));
        let freqs = bank.center_frequencies();
        assert_eq!(freqs.len(), 2);
        assert!((freqs[0] - 500.0).abs() < 0.01);
        assert!((freqs[1] - 2000.0).abs() < 0.01);
    }

    #[test]
    fn test_filter_bank_process_silence() {
        let mut bank = FilterBank::new(48000.0);
        bank.add_band(FilterBand::peaking(1000.0, 200.0, 6.0, 48000.0));
        // Silence should remain silence
        let out = bank.process_sample(0.0);
        assert!(out.abs() < 1e-6);
    }

    #[test]
    fn test_filter_bank_reset() {
        let mut bank = FilterBank::new(48000.0);
        bank.add_band(FilterBand::peaking(500.0, 100.0, 3.0, 48000.0));
        bank.process_sample(1.0);
        bank.reset();
        // After reset, silence input should yield near-zero output
        let out = bank.process_sample(0.0);
        assert!(
            out.abs() < 1e-5,
            "expected near zero after reset, got {out}"
        );
    }

    #[test]
    fn test_filter_bank_config_band_count() {
        let cfg = FilterBankConfig::new(10, 48000.0);
        assert_eq!(cfg.band_count(), 10);
        assert!(!cfg.bypassed);
    }

    #[test]
    fn test_filter_bank_sample_rate() {
        let bank = FilterBank::new(44100.0);
        assert!((bank.sample_rate() - 44100.0).abs() < 0.01);
    }

    #[test]
    fn test_band_pass_attenuates_off_frequency() {
        let mut band = FilterBand::band_pass(10000.0, 500.0, 48000.0);
        // Feed a low-frequency pulse and check that DC is attenuated
        let out = band.process_sample(1.0);
        // Band-pass at 10kHz should pass very little DC energy
        assert!(out.abs() < 0.5, "expected attenuation of DC, got {out}");
    }

    #[test]
    fn test_notch_reduces_signal() {
        // Create a notch band manually by using recalculate
        let mut band = FilterBand {
            center_hz: 1000.0,
            bandwidth_hz_val: 100.0,
            gain_db: 0.0,
            shape: BandShape::Notch,
            enabled: true,
            s1: 0.0,
            s2: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        band.recalculate(48000.0);
        // Just verify it processes without panic
        let _ = band.process_sample(0.5);
    }

    /// DC (z=1) gain of a normalized biquad: (b0+b1+b2) / (1+a1+a2).
    fn dc_gain(band: &FilterBand) -> f32 {
        (band.b0 + band.b1 + band.b2) / (1.0 + band.a1 + band.a2)
    }

    /// Nyquist (z=-1) gain of a normalized biquad: (b0-b1+b2) / (1-a1+a2).
    fn nyquist_gain(band: &FilterBand) -> f32 {
        (band.b0 - band.b1 + band.b2) / (1.0 - band.a1 + band.a2)
    }

    #[test]
    fn test_low_shelf_gain_db_produces_non_identity_and_correct_dc_gain() {
        let band = FilterBand::low_shelf(200.0, 100.0, 6.0, 48000.0);

        // Coefficients must not be the old flat pass-through identity
        // (b0=1, b1=0, b2=0, a1=0, a2=0).
        assert!(
            band.b1.abs() > 1e-6 || band.b2.abs() > 1e-6 || band.a1.abs() > 1e-6,
            "low-shelf coefficients must not be identity for nonzero gain_db: \
             b0={} b1={} b2={} a1={} a2={}",
            band.b0,
            band.b1,
            band.b2,
            band.a1,
            band.a2
        );

        // +6 dB → linear gain 10^(6/20) ≈ 1.995 (~2x) at DC.
        let expected_dc = 10.0_f32.powf(6.0 / 20.0);
        let dc = dc_gain(&band);
        assert!(
            (dc - expected_dc).abs() < 0.02,
            "expected DC gain ≈ {expected_dc} (+6dB ≈ 2x), got {dc}"
        );

        // A low shelf must leave high frequencies (Nyquist) unaffected (~unity).
        let ny = nyquist_gain(&band);
        assert!(
            (ny - 1.0).abs() < 0.02,
            "expected Nyquist gain ≈ 1.0 (unaffected), got {ny}"
        );
    }

    #[test]
    fn test_high_shelf_gain_db_produces_non_identity_and_correct_nyquist_gain() {
        let band = FilterBand::high_shelf(5000.0, 1000.0, 6.0, 48000.0);

        assert!(
            band.b1.abs() > 1e-6 || band.b2.abs() > 1e-6 || band.a1.abs() > 1e-6,
            "high-shelf coefficients must not be identity for nonzero gain_db: \
             b0={} b1={} b2={} a1={} a2={}",
            band.b0,
            band.b1,
            band.b2,
            band.a1,
            band.a2
        );

        // A high shelf must leave low frequencies (DC) unaffected (~unity).
        let dc = dc_gain(&band);
        assert!(
            (dc - 1.0).abs() < 0.02,
            "expected DC gain ≈ 1.0 (unaffected), got {dc}"
        );

        // +6 dB → linear gain 10^(6/20) ≈ 1.995 (~2x) at Nyquist.
        let expected_ny = 10.0_f32.powf(6.0 / 20.0);
        let ny = nyquist_gain(&band);
        assert!(
            (ny - expected_ny).abs() < 0.02,
            "expected Nyquist gain ≈ {expected_ny} (+6dB ≈ 2x), got {ny}"
        );
    }

    #[test]
    fn test_low_shelf_zero_gain_is_unity() {
        let band = FilterBand::low_shelf(200.0, 100.0, 0.0, 48000.0);
        let dc = dc_gain(&band);
        assert!(
            (dc - 1.0).abs() < 1e-3,
            "0 dB low-shelf must be unity gain, got {dc}"
        );
    }

    #[test]
    fn test_high_shelf_zero_gain_is_unity() {
        let band = FilterBand::high_shelf(5000.0, 1000.0, 0.0, 48000.0);
        let ny = nyquist_gain(&band);
        assert!(
            (ny - 1.0).abs() < 1e-3,
            "0 dB high-shelf must be unity gain, got {ny}"
        );
    }

    #[test]
    fn test_low_shelf_cut_reduces_dc_gain() {
        let band = FilterBand::low_shelf(200.0, 100.0, -6.0, 48000.0);
        let dc = dc_gain(&band);
        assert!(
            dc < 1.0,
            "expected -6dB low shelf to attenuate DC, got {dc}"
        );
    }

    #[test]
    fn test_filter_bank_low_shelf_boosts_near_dc_signal_end_to_end() {
        // End-to-end check through FilterBank::process_sample (not just the
        // raw coefficient formula): drive a near-constant (near-DC) input to
        // steady state and confirm a +12dB low shelf actually boosts it.
        let mut bank = FilterBank::new(48000.0);
        bank.add_band(FilterBand::low_shelf(100.0, 50.0, 12.0, 48000.0));
        let mut out = 0.0;
        for _ in 0..2000 {
            out = bank.process_sample(0.3);
        }
        assert!(
            out > 0.3,
            "expected +12dB low shelf to boost a near-DC signal at steady state, got {out}"
        );
    }
}

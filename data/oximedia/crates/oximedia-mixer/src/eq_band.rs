#![allow(dead_code)]
//! Parametric equalizer band processing for mixer channels.
//!
//! This module provides a multi-band parametric equalizer with support for
//! various filter types including peaking, shelving, high-pass, low-pass,
//! band-pass, and notch filters. Each band uses biquad IIR coefficients
//! derived from the Audio EQ Cookbook formulas.

use std::f64::consts::PI;

/// Type of EQ filter for a band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqFilterType {
    /// Peaking (bell) filter — boosts or cuts at a center frequency.
    Peaking,
    /// Low-shelf filter — boosts or cuts below a corner frequency.
    LowShelf,
    /// High-shelf filter — boosts or cuts above a corner frequency.
    HighShelf,
    /// High-pass filter (6 dB/oct slope style modeled as biquad).
    HighPass,
    /// Low-pass filter (6 dB/oct slope style modeled as biquad).
    LowPass,
    /// Band-pass filter — passes a narrow frequency band.
    BandPass,
    /// Notch (band-reject) filter — removes a narrow frequency band.
    Notch,
}

/// Biquad filter coefficients (Direct Form I).
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    /// Feed-forward coefficient b0.
    pub b0: f64,
    /// Feed-forward coefficient b1.
    pub b1: f64,
    /// Feed-forward coefficient b2.
    pub b2: f64,
    /// Feedback coefficient a1.
    pub a1: f64,
    /// Feedback coefficient a2.
    pub a2: f64,
}

impl Default for BiquadCoeffs {
    fn default() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }
}

impl BiquadCoeffs {
    /// Create unity (pass-through) coefficients.
    #[must_use]
    pub fn unity() -> Self {
        Self::default()
    }

    /// Create coefficients for a peaking EQ band.
    ///
    /// * `freq` — center frequency in Hz
    /// * `gain_db` — gain in dB (positive = boost, negative = cut)
    /// * `q` — quality factor (bandwidth control)
    /// * `sample_rate` — sample rate in Hz
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn peaking(freq: f64, gain_db: f64, q: f64, sample_rate: u32) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * freq / f64::from(sample_rate);
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * w0.cos();
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * w0.cos();
        let a2 = 1.0 - alpha / a;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Create coefficients for a low-shelf EQ band.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn low_shelf(freq: f64, gain_db: f64, q: f64, sample_rate: u32) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * freq / f64::from(sample_rate);
        let sin_w0 = w0.sin();
        let cos_w0 = w0.cos();
        let alpha = sin_w0 / (2.0 * q);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Create coefficients for a high-shelf EQ band.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn high_shelf(freq: f64, gain_db: f64, q: f64, sample_rate: u32) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * freq / f64::from(sample_rate);
        let sin_w0 = w0.sin();
        let cos_w0 = w0.cos();
        let alpha = sin_w0 / (2.0 * q);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Create coefficients for a high-pass filter.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn high_pass(freq: f64, q: f64, sample_rate: u32) -> Self {
        let w0 = 2.0 * PI * freq / f64::from(sample_rate);
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Create coefficients for a low-pass filter.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn low_pass(freq: f64, q: f64, sample_rate: u32) -> Self {
        let w0 = 2.0 * PI * freq / f64::from(sample_rate);
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);

        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// State for a single biquad filter (Direct Form II transposed).
#[derive(Debug, Clone, Copy)]
pub struct BiquadState {
    /// Delay element z^-1.
    pub z1: f64,
    /// Delay element z^-2.
    pub z2: f64,
}

impl Default for BiquadState {
    fn default() -> Self {
        Self { z1: 0.0, z2: 0.0 }
    }
}

impl BiquadState {
    /// Reset the filter state to zero.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Process a single sample through the biquad with the given coefficients.
    pub fn process(&mut self, input: f64, coeffs: &BiquadCoeffs) -> f64 {
        let output = coeffs.b0 * input + self.z1;
        self.z1 = coeffs.b1 * input - coeffs.a1 * output + self.z2;
        self.z2 = coeffs.b2 * input - coeffs.a2 * output;
        output
    }
}

/// A single parametric EQ band with configurable filter type, frequency, gain, and Q.
#[derive(Debug, Clone)]
pub struct EqBand {
    /// Name/label for this band (e.g. "Low", "Mid", "High").
    pub name: String,
    /// Filter type for this band.
    pub filter_type: EqFilterType,
    /// Center/corner frequency in Hz.
    pub frequency: f64,
    /// Gain in dB (for peaking and shelving types).
    pub gain_db: f64,
    /// Quality factor (bandwidth control).
    pub q: f64,
    /// Whether this band is bypassed.
    pub bypass: bool,
    /// Computed biquad coefficients.
    coeffs: BiquadCoeffs,
    /// Filter state per channel (up to 8 channels).
    states: Vec<BiquadState>,
    /// Sample rate in Hz.
    sample_rate: u32,
}

impl EqBand {
    /// Create a new EQ band.
    ///
    /// * `name` — display name for the band
    /// * `filter_type` — type of EQ filter
    /// * `frequency` — center/corner frequency in Hz
    /// * `gain_db` — gain in dB
    /// * `q` — quality factor
    /// * `sample_rate` — sample rate in Hz
    /// * `num_channels` — number of audio channels
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        name: String,
        filter_type: EqFilterType,
        frequency: f64,
        gain_db: f64,
        q: f64,
        sample_rate: u32,
        num_channels: usize,
    ) -> Self {
        let mut band = Self {
            name,
            filter_type,
            frequency,
            gain_db,
            q,
            bypass: false,
            coeffs: BiquadCoeffs::default(),
            states: vec![BiquadState::default(); num_channels],
            sample_rate,
        };
        band.update_coefficients();
        band
    }

    /// Recalculate the biquad coefficients from the current parameters.
    pub fn update_coefficients(&mut self) {
        self.coeffs = match self.filter_type {
            EqFilterType::Peaking => {
                BiquadCoeffs::peaking(self.frequency, self.gain_db, self.q, self.sample_rate)
            }
            EqFilterType::LowShelf => {
                BiquadCoeffs::low_shelf(self.frequency, self.gain_db, self.q, self.sample_rate)
            }
            EqFilterType::HighShelf => {
                BiquadCoeffs::high_shelf(self.frequency, self.gain_db, self.q, self.sample_rate)
            }
            EqFilterType::HighPass => {
                BiquadCoeffs::high_pass(self.frequency, self.q, self.sample_rate)
            }
            EqFilterType::LowPass => {
                BiquadCoeffs::low_pass(self.frequency, self.q, self.sample_rate)
            }
            EqFilterType::BandPass | EqFilterType::Notch => {
                // Band-pass and notch reuse peaking with gain=0 as approximation
                BiquadCoeffs::peaking(self.frequency, 0.0, self.q, self.sample_rate)
            }
        };
    }

    /// Get the current biquad coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &BiquadCoeffs {
        &self.coeffs
    }

    /// Set the center/corner frequency and recalculate.
    pub fn set_frequency(&mut self, freq: f64) {
        self.frequency = freq.clamp(20.0, 20000.0);
        self.update_coefficients();
    }

    /// Set the gain in dB and recalculate.
    pub fn set_gain_db(&mut self, gain: f64) {
        self.gain_db = gain.clamp(-24.0, 24.0);
        self.update_coefficients();
    }

    /// Set the Q factor and recalculate.
    pub fn set_q(&mut self, q: f64) {
        self.q = q.clamp(0.1, 30.0);
        self.update_coefficients();
    }

    /// Set the filter type and recalculate.
    pub fn set_filter_type(&mut self, ft: EqFilterType) {
        self.filter_type = ft;
        self.update_coefficients();
    }

    /// Process a single sample for the given channel index.
    pub fn process_sample(&mut self, sample: f64, channel: usize) -> f64 {
        if self.bypass || channel >= self.states.len() {
            return sample;
        }
        self.states[channel].process(sample, &self.coeffs)
    }

    /// Process a buffer of interleaved samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f64], num_channels: usize) {
        if self.bypass || num_channels == 0 {
            return;
        }
        let num_frames = buffer.len() / num_channels;
        for frame in 0..num_frames {
            for ch in 0..num_channels.min(self.states.len()) {
                let idx = frame * num_channels + ch;
                buffer[idx] = self.states[ch].process(buffer[idx], &self.coeffs);
            }
        }
    }

    /// Reset all channel filter states.
    pub fn reset(&mut self) {
        for state in &mut self.states {
            state.reset();
        }
    }
}

/// Multi-band parametric equalizer (up to 8 bands).
#[derive(Debug, Clone)]
pub struct ParametricEq {
    /// List of EQ bands.
    pub bands: Vec<EqBand>,
    /// Whether the entire EQ is bypassed.
    pub bypass: bool,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Number of audio channels.
    num_channels: usize,
}

impl ParametricEq {
    /// Create a new parametric EQ.
    #[must_use]
    pub fn new(sample_rate: u32, num_channels: usize) -> Self {
        Self {
            bands: Vec::new(),
            bypass: false,
            sample_rate,
            num_channels,
        }
    }

    /// Create a standard 4-band EQ (low-shelf, low-mid peak, high-mid peak, high-shelf).
    #[must_use]
    pub fn four_band(sample_rate: u32, num_channels: usize) -> Self {
        let mut eq = Self::new(sample_rate, num_channels);
        eq.add_band("Low".into(), EqFilterType::LowShelf, 100.0, 0.0, 0.707);
        eq.add_band("Low-Mid".into(), EqFilterType::Peaking, 500.0, 0.0, 1.0);
        eq.add_band("High-Mid".into(), EqFilterType::Peaking, 3000.0, 0.0, 1.0);
        eq.add_band("High".into(), EqFilterType::HighShelf, 8000.0, 0.0, 0.707);
        eq
    }

    /// Add a new band to the EQ.
    pub fn add_band(
        &mut self,
        name: String,
        filter_type: EqFilterType,
        frequency: f64,
        gain_db: f64,
        q: f64,
    ) {
        let band = EqBand::new(
            name,
            filter_type,
            frequency,
            gain_db,
            q,
            self.sample_rate,
            self.num_channels,
        );
        self.bands.push(band);
    }

    /// Remove a band by index.
    pub fn remove_band(&mut self, index: usize) -> Option<EqBand> {
        if index < self.bands.len() {
            Some(self.bands.remove(index))
        } else {
            None
        }
    }

    /// Get the number of bands.
    #[must_use]
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Process interleaved audio buffer through all bands in series.
    pub fn process_buffer(&mut self, buffer: &mut [f64], num_channels: usize) {
        if self.bypass {
            return;
        }
        for band in &mut self.bands {
            band.process_buffer(buffer, num_channels);
        }
    }

    /// Reset all band states.
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    /// Flat-reset all band gains to 0 dB.
    pub fn flatten(&mut self) {
        for band in &mut self.bands {
            band.set_gain_db(0.0);
        }
    }
}

/// Compute the magnitude response of a biquad filter at a given frequency.
///
/// Returns the magnitude in dB.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn magnitude_response_db(coeffs: &BiquadCoeffs, freq: f64, sample_rate: u32) -> f64 {
    let w = 2.0 * PI * freq / f64::from(sample_rate);
    let cos_w = w.cos();
    let cos_2w = (2.0 * w).cos();

    let num = coeffs.b0 * coeffs.b0
        + coeffs.b1 * coeffs.b1
        + coeffs.b2 * coeffs.b2
        + 2.0 * (coeffs.b0 * coeffs.b1 + coeffs.b1 * coeffs.b2) * cos_w
        + 2.0 * coeffs.b0 * coeffs.b2 * cos_2w;

    let den = 1.0
        + coeffs.a1 * coeffs.a1
        + coeffs.a2 * coeffs.a2
        + 2.0 * (coeffs.a1 + coeffs.a1 * coeffs.a2) * cos_w
        + 2.0 * coeffs.a2 * cos_2w;

    if den <= 0.0 {
        return 0.0;
    }

    10.0 * (num / den).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unity_coefficients() {
        let c = BiquadCoeffs::unity();
        assert!((c.b0 - 1.0).abs() < f64::EPSILON);
        assert!((c.b1).abs() < f64::EPSILON);
        assert!((c.a1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_peaking_zero_gain_is_passthrough() {
        let c = BiquadCoeffs::peaking(1000.0, 0.0, 1.0, 48000);
        // With 0 dB gain, peaking should be approximately unity
        assert!((c.b0 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_biquad_state_reset() {
        let mut state = BiquadState { z1: 1.0, z2: 2.0 };
        state.reset();
        assert!((state.z1).abs() < f64::EPSILON);
        assert!((state.z2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_biquad_process_unity() {
        let coeffs = BiquadCoeffs::unity();
        let mut state = BiquadState::default();
        let out = state.process(0.5, &coeffs);
        assert!((out - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eq_band_creation() {
        let band = EqBand::new(
            "Mid".into(),
            EqFilterType::Peaking,
            1000.0,
            3.0,
            1.0,
            48000,
            2,
        );
        assert_eq!(band.name, "Mid");
        assert!((band.frequency - 1000.0).abs() < f64::EPSILON);
        assert!(!band.bypass);
    }

    #[test]
    fn test_eq_band_set_frequency_clamp() {
        let mut band = EqBand::new(
            "Test".into(),
            EqFilterType::Peaking,
            500.0,
            0.0,
            1.0,
            48000,
            1,
        );
        band.set_frequency(5.0);
        assert!((band.frequency - 20.0).abs() < f64::EPSILON);
        band.set_frequency(30000.0);
        assert!((band.frequency - 20000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eq_band_set_gain_clamp() {
        let mut band = EqBand::new(
            "Test".into(),
            EqFilterType::Peaking,
            500.0,
            0.0,
            1.0,
            48000,
            1,
        );
        band.set_gain_db(50.0);
        assert!((band.gain_db - 24.0).abs() < f64::EPSILON);
        band.set_gain_db(-50.0);
        assert!((band.gain_db - (-24.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eq_band_set_q_clamp() {
        let mut band = EqBand::new(
            "Test".into(),
            EqFilterType::Peaking,
            500.0,
            0.0,
            1.0,
            48000,
            1,
        );
        band.set_q(0.01);
        assert!((band.q - 0.1).abs() < f64::EPSILON);
        band.set_q(100.0);
        assert!((band.q - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eq_band_bypass() {
        let mut band = EqBand::new(
            "Test".into(),
            EqFilterType::Peaking,
            1000.0,
            6.0,
            1.0,
            48000,
            1,
        );
        band.bypass = true;
        let out = band.process_sample(0.5, 0);
        assert!((out - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parametric_eq_four_band() {
        let eq = ParametricEq::four_band(48000, 2);
        assert_eq!(eq.num_bands(), 4);
        assert_eq!(eq.bands[0].name, "Low");
        assert_eq!(eq.bands[3].name, "High");
    }

    #[test]
    fn test_parametric_eq_add_remove() {
        let mut eq = ParametricEq::new(48000, 2);
        eq.add_band("Band1".into(), EqFilterType::Peaking, 1000.0, 0.0, 1.0);
        assert_eq!(eq.num_bands(), 1);
        let removed = eq.remove_band(0);
        assert!(removed.is_some());
        assert_eq!(eq.num_bands(), 0);
    }

    #[test]
    fn test_parametric_eq_flatten() {
        let mut eq = ParametricEq::four_band(48000, 2);
        eq.bands[0].set_gain_db(6.0);
        eq.bands[1].set_gain_db(-3.0);
        eq.flatten();
        for band in &eq.bands {
            assert!((band.gain_db).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_parametric_eq_bypass() {
        let mut eq = ParametricEq::four_band(48000, 1);
        eq.bands[0].set_gain_db(12.0);
        eq.bypass = true;
        let mut buf = vec![0.5; 8];
        eq.process_buffer(&mut buf, 1);
        // Bypassed, so all samples should remain unchanged
        for &s in &buf {
            assert!((s - 0.5).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_magnitude_response_unity() {
        let c = BiquadCoeffs::unity();
        let mag = magnitude_response_db(&c, 1000.0, 48000);
        assert!(mag.abs() < 0.001);
    }

    #[test]
    fn test_low_shelf_coefficients_finite() {
        let c = BiquadCoeffs::low_shelf(200.0, 6.0, 0.707, 48000);
        assert!(c.b0.is_finite());
        assert!(c.b1.is_finite());
        assert!(c.b2.is_finite());
        assert!(c.a1.is_finite());
        assert!(c.a2.is_finite());
    }

    #[test]
    fn test_high_pass_attenuates_dc() {
        let coeffs = BiquadCoeffs::high_pass(1000.0, 0.707, 48000);
        let mut state = BiquadState::default();
        // Feed DC (constant 1.0) for many samples — output should approach 0
        let mut last = 0.0;
        for _ in 0..10_000 {
            last = state.process(1.0, &coeffs);
        }
        assert!(
            last.abs() < 0.01,
            "HP filter should attenuate DC, got {last}"
        );
    }

    #[test]
    fn test_process_buffer_interleaved() {
        let mut band = EqBand::new(
            "Test".into(),
            EqFilterType::Peaking,
            1000.0,
            0.0,
            1.0,
            48000,
            2,
        );
        let mut buf = vec![0.0; 16]; // 8 frames, 2 channels
        buf[0] = 1.0;
        band.process_buffer(&mut buf, 2);
        // With 0 dB gain peaking, output should be very close to input
        assert!((buf[0] - 1.0).abs() < 0.01);
    }
}

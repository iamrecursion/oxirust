//! Parametric equalizer with biquad IIR filters.
//!
//! Implements the Audio EQ Cookbook formulas for all standard EQ band types.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::f32::consts::PI;

/// EQ band type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandType {
    /// Low-cut (high-pass) filter.
    LowCut,
    /// Low-shelf filter.
    LowShelf,
    /// Peak (bell) EQ filter.
    Peak,
    /// High-shelf filter.
    HighShelf,
    /// High-cut (low-pass) filter.
    HighCut,
    /// Notch (band-reject) filter.
    Notch,
}

impl BandType {
    /// Returns the filter order for this band type.
    #[must_use]
    pub fn order(self) -> u32 {
        match self {
            BandType::LowCut | BandType::HighCut | BandType::Peak | BandType::Notch => 2,
            BandType::LowShelf | BandType::HighShelf => 1,
        }
    }
}

/// A single EQ band definition.
#[derive(Debug, Clone)]
pub struct EqBand {
    /// Center/cutoff frequency in Hz.
    pub frequency: f32,
    /// Gain in dB (positive = boost, negative = cut). Not used for LowCut/HighCut/Notch.
    pub gain_db: f32,
    /// Q factor (bandwidth).
    pub q: f32,
    /// Band type.
    pub band_type: BandType,
}

impl EqBand {
    /// Create a new EQ band.
    #[must_use]
    pub fn new(frequency: f32, gain_db: f32, q: f32, band_type: BandType) -> Self {
        Self {
            frequency,
            gain_db,
            q,
            band_type,
        }
    }

    /// Create a peak band.
    #[must_use]
    pub fn peak(frequency: f32, gain_db: f32, q: f32) -> Self {
        Self::new(frequency, gain_db, q, BandType::Peak)
    }

    /// Create a low-shelf band.
    #[must_use]
    pub fn low_shelf(frequency: f32, gain_db: f32) -> Self {
        Self::new(frequency, gain_db, 0.707, BandType::LowShelf)
    }

    /// Create a high-shelf band.
    #[must_use]
    pub fn high_shelf(frequency: f32, gain_db: f32) -> Self {
        Self::new(frequency, gain_db, 0.707, BandType::HighShelf)
    }

    /// Create a low-cut (high-pass) band.
    #[must_use]
    pub fn low_cut(frequency: f32, q: f32) -> Self {
        Self::new(frequency, 0.0, q, BandType::LowCut)
    }

    /// Create a high-cut (low-pass) band.
    #[must_use]
    pub fn high_cut(frequency: f32, q: f32) -> Self {
        Self::new(frequency, 0.0, q, BandType::HighCut)
    }

    /// Create a notch band.
    #[must_use]
    pub fn notch(frequency: f32, q: f32) -> Self {
        Self::new(frequency, 0.0, q, BandType::Notch)
    }
}

/// Biquad filter coefficients (second-order IIR).
///
/// Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeff {
    /// Feed-forward coefficient b0.
    pub b0: f32,
    /// Feed-forward coefficient b1.
    pub b1: f32,
    /// Feed-forward coefficient b2.
    pub b2: f32,
    /// Feedback coefficient a1.
    pub a1: f32,
    /// Feedback coefficient a2.
    pub a2: f32,
}

impl BiquadCoeff {
    /// Create coefficients for a bypass (identity) filter.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    /// Compute biquad coefficients from an EQ band using Audio EQ Cookbook formulas.
    #[must_use]
    pub fn from_band(band: &EqBand, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * band.frequency / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * band.q);

        match band.band_type {
            BandType::Peak => {
                // Peak EQ filter (Audio EQ Cookbook)
                let a = 10.0_f32.powf(band.gain_db / 40.0);
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a;
                Self::normalize(b0, b1, b2, a0, a1, a2)
            }
            BandType::LowShelf => {
                // Low shelf: A = sqrt(10^(gain/40))
                let a = 10.0_f32.powf(band.gain_db / 40.0);
                let sqrt_a = a.sqrt();
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
                Self::normalize(b0, b1, b2, a0, a1, a2)
            }
            BandType::HighShelf => {
                // High shelf: A = sqrt(10^(gain/40))
                let a = 10.0_f32.powf(band.gain_db / 40.0);
                let sqrt_a = a.sqrt();
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
                Self::normalize(b0, b1, b2, a0, a1, a2)
            }
            BandType::LowCut => {
                // High-pass Butterworth (2nd order)
                let b0 = (1.0 + cos_w0) / 2.0;
                let b1 = -(1.0 + cos_w0);
                let b2 = (1.0 + cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                Self::normalize(b0, b1, b2, a0, a1, a2)
            }
            BandType::HighCut => {
                // Low-pass Butterworth (2nd order)
                let b0 = (1.0 - cos_w0) / 2.0;
                let b1 = 1.0 - cos_w0;
                let b2 = (1.0 - cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                Self::normalize(b0, b1, b2, a0, a1, a2)
            }
            BandType::Notch => {
                // Band-reject notch filter
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                Self::normalize(b0, b1, b2, a0, a1, a2)
            }
        }
    }

    /// Normalize coefficients by a0.
    fn normalize(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Biquad filter with state variables using Direct Form II Transposed.
pub struct BiquadFilter {
    coeff: BiquadCoeff,
    /// State variable 1 (x[n-1] equivalent in transposed form).
    x1: f32,
    /// State variable 2 (x[n-2] equivalent in transposed form).
    x2: f32,
    /// State variable for output delay 1.
    y1: f32,
    /// State variable for output delay 2.
    y2: f32,
}

impl BiquadFilter {
    /// Create a new biquad filter with the given coefficients.
    #[must_use]
    pub fn new(coeff: BiquadCoeff) -> Self {
        Self {
            coeff,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Create a bypass (identity) filter.
    #[must_use]
    pub fn identity() -> Self {
        Self::new(BiquadCoeff::identity())
    }

    /// Create a filter from an EQ band.
    #[must_use]
    pub fn from_band(band: &EqBand, sample_rate: f32) -> Self {
        Self::new(BiquadCoeff::from_band(band, sample_rate))
    }

    /// Update the coefficients (for real-time parameter changes).
    pub fn set_coeff(&mut self, coeff: BiquadCoeff) {
        self.coeff = coeff;
    }

    /// Process a single sample using Direct Form II Transposed.
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let c = &self.coeff;
        // Direct Form II Transposed:
        // y[n] = b0*x[n] + s1[n-1]
        // s1[n] = b1*x[n] - a1*y[n] + s2[n-1]
        // s2[n] = b2*x[n] - a2*y[n]
        let y = c.b0 * x + self.x1;
        self.x1 = c.b1 * x - c.a1 * y + self.x2;
        self.x2 = c.b2 * x - c.a2 * y;
        // Store for reference (not strictly needed for DFII transposed, but kept for compatibility)
        self.y1 = y;
        self.y2 = self.y1;
        y
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Multi-band parametric equalizer.
pub struct ParametricEq {
    /// EQ bands paired with their biquad filters.
    pub bands: Vec<(EqBand, BiquadFilter)>,
}

impl ParametricEq {
    /// Create a new empty parametric EQ.
    #[must_use]
    pub fn new() -> Self {
        Self { bands: Vec::new() }
    }

    /// Add a band to the EQ.
    pub fn add_band(&mut self, band: EqBand, sample_rate: f32) {
        let filter = BiquadFilter::from_band(&band, sample_rate);
        self.bands.push((band, filter));
    }

    /// Process a buffer of samples through all EQ bands.
    #[must_use]
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        let mut output: Vec<f32> = samples.to_vec();
        for sample in &mut output {
            for (_, filter) in &mut self.bands {
                *sample = filter.process_sample(*sample);
            }
        }
        output
    }

    /// Reset all filter states.
    pub fn reset(&mut self) {
        for (_, filter) in &mut self.bands {
            filter.reset();
        }
    }

    /// Broadcast presence boost preset.
    ///
    /// A gentle high-mid boost around 4kHz for improved speech intelligibility.
    #[must_use]
    pub fn broadcast_presence_boost() -> Self {
        let mut eq = Self::new();
        let sr = 48000.0;
        // High-pass to remove rumble
        eq.add_band(EqBand::low_cut(80.0, 0.707), sr);
        // Gentle presence boost at 4kHz
        eq.add_band(EqBand::peak(4000.0, 3.0, 1.5), sr);
        // Slight air boost at 12kHz
        eq.add_band(EqBand::high_shelf(12000.0, 1.5), sr);
        eq
    }

    /// Bass boost preset.
    ///
    /// Boost low frequencies for more body.
    #[must_use]
    pub fn bass_boost() -> Self {
        let mut eq = Self::new();
        let sr = 48000.0;
        // Low shelf boost
        eq.add_band(EqBand::low_shelf(120.0, 6.0), sr);
        // Presence dip to compensate for muddy mids
        eq.add_band(EqBand::peak(300.0, -2.0, 1.0), sr);
        // High shelf for clarity
        eq.add_band(EqBand::high_shelf(8000.0, 1.0), sr);
        eq
    }

    /// Vocal clarity preset.
    ///
    /// Shapes the frequency response for optimal vocal presence.
    #[must_use]
    pub fn vocal_clarity() -> Self {
        let mut eq = Self::new();
        let sr = 48000.0;
        // Remove low-end rumble
        eq.add_band(EqBand::low_cut(120.0, 0.707), sr);
        // Body enhancement
        eq.add_band(EqBand::peak(250.0, -2.0, 0.8), sr);
        // Presence boost
        eq.add_band(EqBand::peak(3500.0, 3.0, 1.2), sr);
        // Sibilance control
        eq.add_band(EqBand::peak(8000.0, -2.0, 2.0), sr);
        // Air
        eq.add_band(EqBand::high_shelf(10000.0, 2.0), sr);
        eq
    }
}

impl Default for ParametricEq {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq: f32, sample_rate: f32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_band_type_order() {
        assert_eq!(BandType::LowCut.order(), 2);
        assert_eq!(BandType::LowShelf.order(), 1);
        assert_eq!(BandType::Peak.order(), 2);
        assert_eq!(BandType::HighShelf.order(), 1);
        assert_eq!(BandType::HighCut.order(), 2);
        assert_eq!(BandType::Notch.order(), 2);
    }

    #[test]
    fn test_biquad_identity() {
        let mut filter = BiquadFilter::identity();
        let output = filter.process_sample(1.0);
        // Identity filter should pass through
        assert!(output.is_finite());
    }

    #[test]
    fn test_peak_band_coefficients() {
        let band = EqBand::peak(1000.0, 6.0, 1.0);
        let coeff = BiquadCoeff::from_band(&band, 48000.0);
        assert!(coeff.b0.is_finite());
        assert!(coeff.b1.is_finite());
        assert!(coeff.b2.is_finite());
        assert!(coeff.a1.is_finite());
        assert!(coeff.a2.is_finite());
    }

    #[test]
    fn test_low_shelf_coefficients() {
        let band = EqBand::low_shelf(200.0, 6.0);
        let coeff = BiquadCoeff::from_band(&band, 48000.0);
        assert!(coeff.b0.is_finite());
        assert!(coeff.b0 > 0.0);
    }

    #[test]
    fn test_high_shelf_coefficients() {
        let band = EqBand::high_shelf(8000.0, -6.0);
        let coeff = BiquadCoeff::from_band(&band, 48000.0);
        assert!(coeff.b0.is_finite());
    }

    #[test]
    fn test_low_cut_coefficients() {
        let band = EqBand::low_cut(80.0, 0.707);
        let coeff = BiquadCoeff::from_band(&band, 48000.0);
        assert!(coeff.b0.is_finite());
    }

    #[test]
    fn test_high_cut_coefficients() {
        let band = EqBand::high_cut(16000.0, 0.707);
        let coeff = BiquadCoeff::from_band(&band, 48000.0);
        assert!(coeff.b0.is_finite());
    }

    #[test]
    fn test_notch_coefficients() {
        let band = EqBand::notch(1000.0, 5.0);
        let coeff = BiquadCoeff::from_band(&band, 48000.0);
        assert!(coeff.b0.is_finite());
    }

    #[test]
    fn test_filter_output_is_finite() {
        let band = EqBand::peak(1000.0, 6.0, 1.0);
        let mut filter = BiquadFilter::from_band(&band, 48000.0);
        let sine = make_sine(440.0, 48000.0, 512);
        for s in &sine {
            let out = filter.process_sample(*s);
            assert!(out.is_finite(), "Output not finite: {out}");
        }
    }

    #[test]
    fn test_parametric_eq_process() {
        let mut eq = ParametricEq::new();
        eq.add_band(EqBand::peak(1000.0, 6.0, 1.0), 48000.0);
        let input = vec![0.5f32; 128];
        let output = eq.process(&input);
        assert_eq!(output.len(), 128);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_broadcast_presence_boost() {
        let mut eq = ParametricEq::broadcast_presence_boost();
        assert_eq!(eq.bands.len(), 3);
        let input = make_sine(1000.0, 48000.0, 256);
        let output = eq.process(&input);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_bass_boost() {
        let mut eq = ParametricEq::bass_boost();
        assert_eq!(eq.bands.len(), 3);
        let input = make_sine(100.0, 48000.0, 256);
        let output = eq.process(&input);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_vocal_clarity() {
        let mut eq = ParametricEq::vocal_clarity();
        assert_eq!(eq.bands.len(), 5);
        let input = make_sine(3000.0, 48000.0, 256);
        let output = eq.process(&input);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_filter_reset() {
        let band = EqBand::peak(1000.0, 6.0, 1.0);
        let mut filter = BiquadFilter::from_band(&band, 48000.0);
        filter.process_sample(1.0);
        filter.reset();
        // After reset, state is clean
        assert_eq!(filter.x1, 0.0);
        assert_eq!(filter.x2, 0.0);
    }

    #[test]
    fn test_eq_multi_band() {
        let mut eq = ParametricEq::new();
        eq.add_band(EqBand::low_cut(80.0, 0.707), 48000.0);
        eq.add_band(EqBand::peak(500.0, -3.0, 1.0), 48000.0);
        eq.add_band(EqBand::peak(4000.0, 4.0, 1.5), 48000.0);
        eq.add_band(EqBand::high_shelf(10000.0, 2.0), 48000.0);
        let input = vec![0.1f32; 512];
        let output = eq.process(&input);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_peak_boost_increases_level_near_center() {
        // A peak boost at 1kHz should increase the amplitude of a 1kHz sine
        let mut eq = ParametricEq::new();
        eq.add_band(EqBand::peak(1000.0, 12.0, 1.0), 48000.0);

        // Settle the filter
        let settle: Vec<f32> = make_sine(1000.0, 48000.0, 2048);
        let _ = eq.process(&settle);

        let input = make_sine(1000.0, 48000.0, 512);
        let output = eq.process(&input);

        let in_rms: f32 = (input.iter().map(|&s| s * s).sum::<f32>() / input.len() as f32).sqrt();
        let out_rms: f32 =
            (output.iter().map(|&s| s * s).sum::<f32>() / output.len() as f32).sqrt();

        // Boosted output should have higher RMS
        assert!(
            out_rms > in_rms,
            "Expected out_rms {out_rms} > in_rms {in_rms}"
        );
    }
}

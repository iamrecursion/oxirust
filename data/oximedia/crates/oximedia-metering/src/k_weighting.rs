//! K-weighting filter implementation for ITU-R BS.1770-4 loudness measurement.
//!
//! The K-weighting filter chain consists of two second-order IIR biquad stages:
//!
//! - **Stage 1 (High-Shelf)**: Models the acoustic effect of the listener's head
//!   on perceived loudness (boost of about +4 dB above 1 kHz).
//! - **Stage 2 (High-Pass)**: Removes very-low-frequency content (≈ 38 Hz corner).
//!
//! These are implemented as direct-form II transposed biquad sections.

#![allow(dead_code)]

/// One of the two stages in the K-weighting filter chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KWeightingStage {
    /// High-shelf boost stage (head diffraction model).
    HighShelf,
    /// Second-order high-pass stage (sub-bass removal).
    HighPass,
}

impl KWeightingStage {
    /// Human-readable name for this stage.
    pub fn stage_name(&self) -> &'static str {
        match self {
            Self::HighShelf => "High-Shelf (Head Diffraction)",
            Self::HighPass => "High-Pass (Sub-Bass Removal)",
        }
    }
}

/// Biquad IIR filter coefficients (normalised, a0 = 1.0).
#[derive(Clone, Copy, Debug)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

/// Single-channel biquad state (direct-form II transposed).
#[derive(Clone, Copy, Debug, Default)]
struct BiquadState {
    s1: f64,
    s2: f64,
}

impl BiquadState {
    fn process(&mut self, x: f64, c: &BiquadCoeffs) -> f64 {
        let y = c.b0 * x + self.s1;
        self.s1 = c.b1 * x - c.a1 * y + self.s2;
        self.s2 = c.b2 * x - c.a2 * y;
        y
    }
}

/// Configuration for a [`KWeightedFilter`].
#[derive(Clone, Copy, Debug)]
pub struct KWeightingConfig {
    /// Sample rate in Hz.
    pub sample_rate_hz: f64,
}

impl KWeightingConfig {
    /// Create a new config for the given sample rate.
    pub fn new(sample_rate_hz: f64) -> Self {
        Self { sample_rate_hz }
    }

    /// Return the configured sample rate.
    pub fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }
}

/// K-weighting filter for a single audio channel (two cascaded biquad stages).
///
/// Implements the full ITU-R BS.1770-4 K-weighting filter chain.
#[derive(Clone, Debug)]
pub struct KWeightedFilter {
    shelf_coeffs: BiquadCoeffs,
    hp_coeffs: BiquadCoeffs,
    shelf_state: BiquadState,
    hp_state: BiquadState,
    config: KWeightingConfig,
}

impl KWeightedFilter {
    /// Create a new K-weighting filter for the given sample rate.
    ///
    /// Coefficients are pre-computed for the given `sample_rate_hz`.
    pub fn new(config: KWeightingConfig) -> Self {
        let (shelf_coeffs, hp_coeffs) = Self::compute_coeffs(config.sample_rate_hz);
        Self {
            shelf_coeffs,
            hp_coeffs,
            shelf_state: BiquadState::default(),
            hp_state: BiquadState::default(),
            config,
        }
    }

    /// Compute biquad coefficients for both K-weighting stages.
    fn compute_coeffs(fs: f64) -> (BiquadCoeffs, BiquadCoeffs) {
        // --- Stage 1: high-shelf (pre-filter) ---
        // Based on ITU-R BS.1770-4 Annex 1 for 48 kHz, scaled for other rates.
        let db = 3.999_843_853_973_347;
        let f0 = 1_681.974_450_955_533;
        let q = 0.707_213_195_806_047_6;

        let k = (std::f64::consts::PI * f0 / fs).tan();
        let vh = 10_f64.powf(db / 20.0);
        let vb = vh.powf(0.5);
        let denom = 1.0 + k / q + k * k;
        let shelf = BiquadCoeffs {
            b0: (vh + vb * k / q + k * k) / denom,
            b1: 2.0 * (k * k - vh) / denom,
            b2: (vh - vb * k / q + k * k) / denom,
            a1: 2.0 * (k * k - 1.0) / denom,
            a2: (1.0 - k / q + k * k) / denom,
        };

        // --- Stage 2: high-pass (RLB weighting, fc ≈ 38.1 Hz) ---
        let f1 = 38.134_566_580_756_27;
        let q2 = 0.500_316_983_843_589_1;
        let k2 = (std::f64::consts::PI * f1 / fs).tan();
        let denom2 = 1.0 + k2 / q2 + k2 * k2;
        let hp = BiquadCoeffs {
            b0: 1.0 / denom2,
            b1: -2.0 / denom2,
            b2: 1.0 / denom2,
            a1: 2.0 * (k2 * k2 - 1.0) / denom2,
            a2: (1.0 - k2 / q2 + k2 * k2) / denom2,
        };

        (shelf, hp)
    }

    /// Process a single sample through both filter stages.
    pub fn apply_sample(&mut self, sample: f64) -> f64 {
        let after_shelf = self.shelf_state.process(sample, &self.shelf_coeffs);
        self.hp_state.process(after_shelf, &self.hp_coeffs)
    }

    /// Process a buffer of samples in-place (single channel).
    pub fn apply_buffer(&mut self, buffer: &mut [f64]) {
        for s in buffer.iter_mut() {
            *s = self.apply_sample(*s);
        }
    }

    /// Process a read-only buffer, writing results to `output`.
    pub fn apply_buffer_to(&mut self, input: &[f64], output: &mut [f64]) {
        assert_eq!(input.len(), output.len());
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = self.apply_sample(*i);
        }
    }

    /// Reset filter state (zero delay elements).
    pub fn reset(&mut self) {
        self.shelf_state = BiquadState::default();
        self.hp_state = BiquadState::default();
    }

    /// Return the filter's configuration.
    pub fn config(&self) -> &KWeightingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_filter(sr: f64) -> KWeightedFilter {
        KWeightedFilter::new(KWeightingConfig::new(sr))
    }

    #[test]
    fn test_stage_name_high_shelf() {
        assert!(KWeightingStage::HighShelf.stage_name().contains("Shelf"));
    }

    #[test]
    fn test_stage_name_high_pass() {
        assert!(KWeightingStage::HighPass.stage_name().contains("Pass"));
    }

    #[test]
    fn test_config_sample_rate() {
        let cfg = KWeightingConfig::new(48000.0);
        assert!((cfg.sample_rate_hz() - 48000.0).abs() < 1e-9);
    }

    #[test]
    fn test_filter_creates_without_panic() {
        let _ = make_filter(48000.0);
    }

    #[test]
    fn test_filter_44100_creates_without_panic() {
        let _ = make_filter(44100.0);
    }

    #[test]
    fn test_apply_sample_dc_attenuated() {
        // A DC signal (constant 1.0) should be attenuated heavily by the high-pass.
        let mut f = make_filter(48000.0);
        // Run for 10 000 samples to let filter settle.
        let mut last = 0.0_f64;
        for _ in 0..10_000 {
            last = f.apply_sample(1.0);
        }
        // Settled DC output must be much smaller than 1.0.
        assert!(last.abs() < 0.01, "DC not attenuated: {last}");
    }

    #[test]
    fn test_apply_sample_zero_stays_zero() {
        let mut f = make_filter(48000.0);
        let out = f.apply_sample(0.0);
        assert!((out - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_apply_buffer_length_preserved() {
        let mut f = make_filter(48000.0);
        let mut buf = vec![0.5_f64; 256];
        f.apply_buffer(&mut buf);
        assert_eq!(buf.len(), 256);
    }

    #[test]
    fn test_apply_buffer_to_correct_length() {
        let mut f = make_filter(48000.0);
        let input = vec![0.1_f64; 64];
        let mut output = vec![0.0_f64; 64];
        f.apply_buffer_to(&input, &mut output);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut f = make_filter(48000.0);
        // Drive with a ramp to build up state.
        for i in 0..1000 {
            f.apply_sample(i as f64 * 0.001);
        }
        f.reset();
        // After reset, a zero sample should produce (near) zero.
        let out = f.apply_sample(0.0);
        assert!(out.abs() < 1e-15, "state not cleared after reset: {out}");
    }

    #[test]
    fn test_filter_config_accessible() {
        let f = make_filter(96000.0);
        assert!((f.config().sample_rate_hz() - 96000.0).abs() < 1e-9);
    }

    #[test]
    fn test_high_freq_passes_more_than_dc() {
        // At fs/4 the filter should pass significantly more energy than DC.
        let mut f = make_filter(48000.0);
        let fs = 48000.0_f64;
        let freq = fs / 4.0; // 12 000 Hz
        let n = 4096;
        let mut energy_hf = 0.0_f64;
        for i in 0..n {
            let s = (2.0 * std::f64::consts::PI * freq / fs * i as f64).sin();
            let out = f.apply_sample(s);
            energy_hf += out * out;
        }

        let mut f2 = make_filter(48000.0);
        let mut energy_dc = 0.0_f64;
        for _ in 0..n {
            let out = f2.apply_sample(1.0);
            energy_dc += out * out;
        }

        assert!(
            energy_hf > energy_dc * 10.0,
            "HF energy {energy_hf} not much greater than DC energy {energy_dc}"
        );
    }

    #[test]
    fn test_linearity_scaling() {
        // K-weighting is a linear filter: doubling input must double output.
        let mut f1 = make_filter(48000.0);
        let mut f2 = make_filter(48000.0);
        let input: Vec<f64> = (0..128).map(|i| (i as f64 * 0.01).sin()).collect();
        let out1: Vec<f64> = input.iter().map(|&s| f1.apply_sample(s)).collect();
        let out2: Vec<f64> = input.iter().map(|&s| f2.apply_sample(2.0 * s)).collect();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (2.0 * a - b).abs() < 1e-10,
                "linearity violated: 2*{a} != {b}"
            );
        }
    }
}

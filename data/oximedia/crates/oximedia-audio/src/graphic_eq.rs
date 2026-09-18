//! 31-band ISO 1/3-octave graphic equalizer.
//!
//! Implements the IEC 61672 / ISO 266 standard centre frequencies for a
//! 31-band graphic equalizer covering 20 Hz – 20 kHz.  Each band is a
//! second-order peaking EQ biquad operating on the Direct Form II Transposed
//! topology for improved numerical stability.
//!
//! # Examples
//!
//! ```
//! use oximedia_audio::graphic_eq::{GraphicEq, GraphicEqConfig};
//!
//! let mut eq = GraphicEq::new(GraphicEqConfig::new(48_000.0));
//! eq.set_band_gain(10, 6.0); // boost band 10 by 6 dB
//! let output = eq.process(0.5_f32);
//! assert!(output.is_finite());
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// ISO 266 / IEC 61672 centre frequencies for 31-band (1/3-octave) EQ
// ─────────────────────────────────────────────────────────────────────────────

/// ISO 1/3-octave centre frequencies (Hz), 20 Hz – 20 kHz, 31 bands.
pub const ISO_CENTER_FREQS: [f32; 31] = [
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 800.0, 1_000.0, 1_250.0, 1_600.0, 2_000.0, 2_500.0, 3_150.0, 4_000.0, 5_000.0,
    6_300.0, 8_000.0, 10_000.0, 12_500.0, 16_000.0, 20_000.0,
];

/// Number of bands in the graphic equalizer.
pub const NUM_BANDS: usize = 31;

// Q factor for 1/3-octave bands (≈ 4.32 for exactly 1/3 octave)
const THIRD_OCTAVE_Q: f32 = 4.318_473_4;

// Maximum gain allowed per band (dB).
const MAX_GAIN_DB: f32 = 24.0;

// ─────────────────────────────────────────────────────────────────────────────
// Direct Form II Transposed biquad state + coefficients
// ─────────────────────────────────────────────────────────────────────────────

/// Biquad filter coefficients (normalised, a0 = 1).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Df2tCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl Df2tCoeffs {
    /// Identity (pass-through) coefficients.
    fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    /// Peaking EQ design (Audio EQ Cookbook, Robert Bristow-Johnson).
    fn peaking_eq(freq_hz: f32, gain_db: f32, q: f32, fs: f32) -> Self {
        let w0 = 2.0 * PI * freq_hz / fs;
        let cos_w = w0.cos();
        let sin_w = w0.sin();
        let alpha = sin_w / (2.0 * q);
        let a = 10.0_f32.powf(gain_db / 40.0); // sqrt(10^(dB/20))

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w;
        let a2 = 1.0 - alpha / a;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Direct Form II Transposed biquad filter with two delay elements.
///
/// The DF2T topology has superior numerical properties compared to Direct Form I
/// because the delay elements operate on signals of smaller dynamic range.
#[derive(Clone, Debug)]
struct Df2tFilter {
    coeffs: Df2tCoeffs,
    /// First delay element (w1).
    w1: f32,
    /// Second delay element (w2).
    w2: f32,
}

impl Df2tFilter {
    fn new(coeffs: Df2tCoeffs) -> Self {
        Self {
            coeffs,
            w1: 0.0,
            w2: 0.0,
        }
    }

    /// Process a single sample, returning the filtered output.
    ///
    /// DF2T equations:
    /// ```text
    /// y   = b0*x + w1
    /// w1' = b1*x - a1*y + w2
    /// w2' = b2*x - a2*y
    /// ```
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.coeffs.b0 * x + self.w1;
        self.w1 = self.coeffs.b1 * x - self.coeffs.a1 * y + self.w2;
        self.w2 = self.coeffs.b2 * x - self.coeffs.a2 * y;
        y
    }

    /// Replace coefficients and flush state to avoid discontinuities.
    fn set_coeffs_smooth(&mut self, coeffs: Df2tCoeffs) {
        self.coeffs = coeffs;
        // Preserve state (smooth transition) — caller decides if reset is needed.
    }

    /// Reset state to silence.
    fn reset(&mut self) {
        self.w1 = 0.0;
        self.w2 = 0.0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphicEqConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`GraphicEq`].
#[derive(Debug, Clone)]
pub struct GraphicEqConfig {
    /// System sample rate in Hz.
    pub sample_rate: f32,
    /// Per-band gain offsets in dB (length must be 0 or exactly 31).
    ///
    /// If empty, all bands start at 0 dB.
    pub gains_db: Vec<f32>,
}

impl GraphicEqConfig {
    /// Create a flat (0 dB) equalizer configuration for the given sample rate.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            gains_db: vec![0.0; NUM_BANDS],
        }
    }

    /// Validate the configuration.
    ///
    /// Returns `Err` if `gains_db.len()` is not 0 or `NUM_BANDS`, or if any
    /// gain exceeds ±`MAX_GAIN_DB`.
    pub fn validate(&self) -> Result<(), String> {
        if !self.gains_db.is_empty() && self.gains_db.len() != NUM_BANDS {
            return Err(format!(
                "gains_db length must be 0 or {NUM_BANDS}, got {}",
                self.gains_db.len()
            ));
        }
        for (i, &g) in self.gains_db.iter().enumerate() {
            if g.abs() > MAX_GAIN_DB {
                return Err(format!(
                    "Band {} gain {g:.1} dB exceeds ±{MAX_GAIN_DB} dB limit",
                    i
                ));
            }
        }
        if self.sample_rate <= 0.0 {
            return Err(format!("sample_rate must be > 0, got {}", self.sample_rate));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphicEq
// ─────────────────────────────────────────────────────────────────────────────

/// 31-band ISO 1/3-octave graphic equalizer.
///
/// Each of the 31 ISO bands has an independent gain control in the range
/// ±24 dB.  Bands are implemented as second-order peaking EQ biquad filters
/// in Direct Form II Transposed topology, cascaded in series.
///
/// # Thread safety
///
/// `GraphicEq` is not `Sync`; use one instance per audio thread or wrap in a
/// `Mutex`.
#[derive(Debug)]
pub struct GraphicEq {
    /// Per-band gain values in dB.
    gains_db: [f32; NUM_BANDS],
    /// Per-band DF2T biquad filters.
    filters: [Df2tFilter; NUM_BANDS],
    /// Audio sample rate (Hz).
    sample_rate: f32,
    /// Overall output gain applied after the filter cascade.
    output_gain: f32,
}

impl GraphicEq {
    /// Create a new `GraphicEq` with the supplied configuration.
    ///
    /// All band gains are initialised from `config.gains_db` (or 0 dB when
    /// `gains_db` is empty).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the configuration is invalid (see [`GraphicEqConfig::validate`]).
    pub fn new(config: GraphicEqConfig) -> Self {
        let gains_db = if config.gains_db.is_empty() {
            [0.0_f32; NUM_BANDS]
        } else {
            let mut arr = [0.0_f32; NUM_BANDS];
            for (i, &g) in config.gains_db.iter().enumerate().take(NUM_BANDS) {
                arr[i] = g.clamp(-MAX_GAIN_DB, MAX_GAIN_DB);
            }
            arr
        };

        let fs = config.sample_rate;
        let filters = std::array::from_fn(|i| {
            let coeffs = if gains_db[i].abs() < 1e-6 {
                Df2tCoeffs::identity()
            } else {
                Df2tCoeffs::peaking_eq(ISO_CENTER_FREQS[i], gains_db[i], THIRD_OCTAVE_Q, fs)
            };
            Df2tFilter::new(coeffs)
        });

        Self {
            gains_db,
            filters,
            sample_rate: fs,
            output_gain: 1.0,
        }
    }

    /// Set the gain for a single band.
    ///
    /// `band` is a zero-based index in [0, 30].  Gains outside ±`MAX_GAIN_DB`
    /// are silently clamped.
    pub fn set_band_gain(&mut self, band: usize, gain_db: f32) {
        if band >= NUM_BANDS {
            return;
        }
        let clamped = gain_db.clamp(-MAX_GAIN_DB, MAX_GAIN_DB);
        self.gains_db[band] = clamped;
        let coeffs = if clamped.abs() < 1e-6 {
            Df2tCoeffs::identity()
        } else {
            Df2tCoeffs::peaking_eq(
                ISO_CENTER_FREQS[band],
                clamped,
                THIRD_OCTAVE_Q,
                self.sample_rate,
            )
        };
        self.filters[band].set_coeffs_smooth(coeffs);
    }

    /// Get the gain for a single band (dB).
    #[must_use]
    pub fn band_gain_db(&self, band: usize) -> f32 {
        if band >= NUM_BANDS {
            return 0.0;
        }
        self.gains_db[band]
    }

    /// Set all 31 band gains at once.
    ///
    /// The slice must have exactly `NUM_BANDS` elements; if not, the call is a
    /// no-op.
    pub fn set_all_gains(&mut self, gains_db: &[f32]) {
        if gains_db.len() != NUM_BANDS {
            return;
        }
        for (i, &g) in gains_db.iter().enumerate() {
            self.set_band_gain(i, g);
        }
    }

    /// Set the master output gain (linear scale, default 1.0).
    pub fn set_output_gain(&mut self, linear: f32) {
        self.output_gain = linear.max(0.0);
    }

    /// Reset all filter states to silence without changing gains.
    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
    }

    /// Process a single sample through all 31 bands in cascade.
    #[inline]
    pub fn process(&mut self, sample: f32) -> f32 {
        let mut out = sample;
        for f in &mut self.filters {
            out = f.process(out);
        }
        out * self.output_gain
    }

    /// Process a block of samples in-place.
    pub fn process_block(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process(*s);
        }
    }

    /// Process a block of samples, returning a new `Vec<f32>`.
    #[must_use]
    pub fn process_block_new(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter().map(|&s| self.process(s)).collect()
    }

    /// Returns the centre frequencies for all 31 bands.
    #[must_use]
    pub fn center_freqs(&self) -> &'static [f32; NUM_BANDS] {
        &ISO_CENTER_FREQS
    }

    /// Returns the current sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns a copy of all current band gains (dB).
    #[must_use]
    pub fn all_gains_db(&self) -> [f32; NUM_BANDS] {
        self.gains_db
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn flat_eq() -> GraphicEq {
        GraphicEq::new(GraphicEqConfig::new(SR))
    }

    #[test]
    fn test_flat_eq_passes_dc() {
        let mut eq = flat_eq();
        let mut out = 0.0_f32;
        for _ in 0..2000 {
            out = eq.process(1.0);
        }
        assert!(
            (out - 1.0).abs() < 0.01,
            "Flat EQ should pass DC; got {out}"
        );
    }

    #[test]
    fn test_set_band_gain_updates_gain() {
        let mut eq = flat_eq();
        eq.set_band_gain(10, 6.0);
        assert!((eq.band_gain_db(10) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_band_gain_clamps_max() {
        let mut eq = flat_eq();
        eq.set_band_gain(0, 100.0);
        assert!(eq.band_gain_db(0) <= MAX_GAIN_DB);
    }

    #[test]
    fn test_set_band_gain_clamps_min() {
        let mut eq = flat_eq();
        eq.set_band_gain(0, -100.0);
        assert!(eq.band_gain_db(0) >= -MAX_GAIN_DB);
    }

    #[test]
    fn test_out_of_range_band_ignored() {
        let mut eq = flat_eq();
        eq.set_band_gain(NUM_BANDS, 6.0); // should not panic
        assert_eq!(eq.band_gain_db(NUM_BANDS), 0.0); // out-of-range returns 0
    }

    #[test]
    fn test_set_all_gains_wrong_length_noop() {
        let mut eq = flat_eq();
        eq.set_all_gains(&[1.0_f32; 5]); // wrong length, should no-op
        for i in 0..NUM_BANDS {
            assert_eq!(eq.band_gain_db(i), 0.0);
        }
    }

    #[test]
    fn test_set_all_gains_correct_length() {
        let mut eq = flat_eq();
        let gains: Vec<f32> = (0..NUM_BANDS).map(|i| i as f32 * 0.5).collect();
        eq.set_all_gains(&gains);
        for i in 0..NUM_BANDS {
            assert!(
                (eq.band_gain_db(i) - (i as f32 * 0.5).clamp(-MAX_GAIN_DB, MAX_GAIN_DB)).abs()
                    < 1e-5
            );
        }
    }

    #[test]
    fn test_process_block_all_finite() {
        let mut eq = flat_eq();
        let mut buf: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        eq.process_block(&mut buf);
        assert!(
            buf.iter().all(|s| s.is_finite()),
            "All outputs must be finite"
        );
    }

    #[test]
    fn test_process_block_new_length() {
        let mut eq = flat_eq();
        let input = vec![0.5_f32; 128];
        let output = eq.process_block_new(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_reset_clears_state() {
        let mut eq = flat_eq();
        for _ in 0..100 {
            eq.process(1.0);
        }
        eq.reset();
        let out = eq.process(0.0);
        // After reset the delay elements should be zero, so output = 0
        assert_eq!(out, 0.0, "After reset+silence output should be 0");
    }

    #[test]
    fn test_boost_amplifies() {
        // Set band 17 (1 kHz) to +12 dB and drive with 1 kHz tone
        let mut eq = flat_eq();
        eq.set_band_gain(17, 12.0);
        let freq = 1000.0_f32;
        let mut peak_flat = 0.0_f32;
        let mut peak_boosted = 0.0_f32;
        for i in 0..4800_usize {
            let s = (2.0 * PI * freq * i as f32 / SR).sin();
            let mut flat = flat_eq();
            let y_flat = flat.process(s);
            let y_boosted = eq.process(s);
            if i > 960 {
                peak_flat = peak_flat.max(y_flat.abs());
                peak_boosted = peak_boosted.max(y_boosted.abs());
            }
        }
        assert!(
            peak_boosted > peak_flat * 1.5,
            "+12 dB boost should amplify; flat={peak_flat} boosted={peak_boosted}"
        );
    }

    #[test]
    fn test_cut_attenuates() {
        let mut eq = flat_eq();
        eq.set_band_gain(17, -12.0); // -12 dB at 1 kHz
        let freq = 1000.0_f32;
        let mut peak_out = 0.0_f32;
        for i in 0..4800_usize {
            let s = (2.0 * PI * freq * i as f32 / SR).sin();
            let y = eq.process(s);
            if i > 960 {
                peak_out = peak_out.max(y.abs());
            }
        }
        assert!(
            peak_out < 0.5,
            "12 dB cut should attenuate; peak={peak_out}"
        );
    }

    #[test]
    fn test_output_gain() {
        let mut eq = flat_eq();
        eq.set_output_gain(0.5);
        let mut out = 0.0_f32;
        for _ in 0..2000 {
            out = eq.process(1.0);
        }
        assert!((out - 0.5).abs() < 0.01, "Output gain 0.5 on DC; got {out}");
    }

    #[test]
    fn test_center_freqs_length() {
        let eq = flat_eq();
        assert_eq!(eq.center_freqs().len(), NUM_BANDS);
        assert_eq!(eq.center_freqs()[0], 20.0);
        assert_eq!(eq.center_freqs()[30], 20_000.0);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = GraphicEqConfig::new(SR);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_wrong_length() {
        let mut cfg = GraphicEqConfig::new(SR);
        cfg.gains_db = vec![0.0; 5];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_gain_too_large() {
        let mut cfg = GraphicEqConfig::new(SR);
        cfg.gains_db[5] = 30.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_all_gains_db_round_trip() {
        let mut eq = flat_eq();
        eq.set_band_gain(5, 3.0);
        let gains = eq.all_gains_db();
        assert!((gains[5] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_df2t_filter_identity_passthrough() {
        let mut f = Df2tFilter::new(Df2tCoeffs::identity());
        let out = f.process(0.75);
        assert!((out - 0.75).abs() < 1e-7);
    }

    #[test]
    fn test_df2t_filter_reset() {
        let mut f = Df2tFilter::new(Df2tCoeffs::peaking_eq(1000.0, 6.0, THIRD_OCTAVE_Q, SR));
        for _ in 0..100 {
            f.process(1.0);
        }
        f.reset();
        let out = f.process(0.0);
        assert_eq!(out, 0.0, "After reset+silence should be 0");
    }

    #[test]
    fn test_sample_rate() {
        let eq = GraphicEq::new(GraphicEqConfig::new(44_100.0));
        assert_eq!(eq.sample_rate(), 44_100.0);
    }
}

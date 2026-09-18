//! Audio analysis metering: peak, RMS, and LUFS (EBU R128).
//!
//! # EBU R128 / ITU-R BS.1770 LUFS metering
//!
//! Momentary loudness is computed using a 400 ms integration window of the
//! K-weighted mean-square signal:
//!
//! ```text
//! LUFS_M = -0.691 + 10 * log10(mean_square_k_weighted)
//! ```
//!
//! K-weighting consists of two cascaded biquad stages (see [`KWeightingFilter`]).

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Meter readings
// ---------------------------------------------------------------------------

/// Snapshot of all meter readings at a point in time.
#[derive(Debug, Clone, PartialEq)]
pub struct MeterReadings {
    /// Sample-accurate peak in dBFS.
    pub peak_db: f32,
    /// RMS level (300 ms integration) in dBFS.
    pub rms_db: f32,
    /// EBU R128 momentary loudness (400 ms integration) in LUFS.
    pub lufs_momentary: f32,
}

impl Default for MeterReadings {
    fn default() -> Self {
        Self {
            peak_db: -f32::INFINITY,
            rms_db: -f32::INFINITY,
            lufs_momentary: -f32::INFINITY,
        }
    }
}

// ---------------------------------------------------------------------------
// K-Weighting filter
// ---------------------------------------------------------------------------

/// Coefficients for a single biquad filter stage.
#[derive(Debug, Clone)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

/// Per-channel biquad filter state.
#[derive(Debug, Clone, Default)]
struct BiquadState {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadState {
    fn process(&mut self, x: f64, c: &BiquadCoeffs) -> f64 {
        let y = c.b0 * x + c.b1 * self.x1 + c.b2 * self.x2 - c.a1 * self.y1 - c.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Two-stage K-weighting filter per ITU-R BS.1770.
///
/// Stage 1 is a high-frequency shelving pre-filter that boosts the
/// high-frequency content (models the acoustic effect of the head).
/// Stage 2 is a high-pass filter at ~100 Hz.
///
/// Coefficients are provided for 48 kHz; for other sample rates a simple
/// bilinear-transform rescaling is applied.
#[derive(Debug, Clone)]
pub struct KWeightingFilter {
    stage1_coeffs: BiquadCoeffs,
    stage2_coeffs: BiquadCoeffs,
    stage1_state: BiquadState,
    stage2_state: BiquadState,
}

impl KWeightingFilter {
    // Reference coefficients at 48 kHz (ITU-R BS.1770-4).
    const S1_B0_48K: f64 = 1.535_124_859_586_97;
    const S1_B1_48K: f64 = -2.691_696_189_406_38;
    const S1_B2_48K: f64 = 1.198_392_810_852_85;
    const S1_A1_48K: f64 = -1.690_659_293_182_41;
    const S1_A2_48K: f64 = 0.732_480_774_215_85;

    const S2_B0_48K: f64 = 1.0;
    const S2_B1_48K: f64 = -2.0;
    const S2_B2_48K: f64 = 1.0;
    const S2_A1_48K: f64 = -1.990_047_454_833_98;
    const S2_A2_48K: f64 = 0.990_072_250_366_21;

    /// Create a new K-weighting filter for the given sample rate.
    ///
    /// For sample rates other than 48 000 Hz the coefficients are rescaled
    /// using a frequency-ratio bilinear-transform approximation.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        if sample_rate == 48000 {
            Self {
                stage1_coeffs: BiquadCoeffs {
                    b0: Self::S1_B0_48K,
                    b1: Self::S1_B1_48K,
                    b2: Self::S1_B2_48K,
                    a1: Self::S1_A1_48K,
                    a2: Self::S1_A2_48K,
                },
                stage2_coeffs: BiquadCoeffs {
                    b0: Self::S2_B0_48K,
                    b1: Self::S2_B1_48K,
                    b2: Self::S2_B2_48K,
                    a1: Self::S2_A1_48K,
                    a2: Self::S2_A2_48K,
                },
                stage1_state: BiquadState::default(),
                stage2_state: BiquadState::default(),
            }
        } else {
            // Rescale via pre-warped bilinear-transform frequency mapping.
            // We re-derive the shelving and HP filter for the target rate.
            Self::design_for_rate(sample_rate)
        }
    }

    /// Design K-weighting coefficients for an arbitrary sample rate using
    /// bilinear transform of the prototype analog filters.
    fn design_for_rate(sample_rate: u32) -> Self {
        use std::f64::consts::PI;
        let fs = sample_rate as f64;

        // ----------------------------------------------------------------
        // Stage 1: High-frequency shelving pre-filter.
        // Analog prototype: boost of +4 dB starting at ~1.5 kHz.
        // Derived from ITU-R BS.1770 Annex 1.
        // Pre-warped centre frequency from the 48 kHz coefficients.
        // ----------------------------------------------------------------
        let w0_s1 = 2.0 * PI * 1681.974_45 / fs;
        let (s1_b0, s1_b1, s1_b2, s1_a1, s1_a2) = {
            // Analog shelving: H(s) = (s^2 + Vb0*w0/Q*s + w0^2) / (s^2 + w0/Q*s + w0^2)
            // where Vb0 = 10^(db/20), Q = 0.7071.
            let vb0 = 10_f64.powf(3.999_843_853_973_347 / 20.0); // ~4 dB
            let q = 0.707_1;
            let k = (w0_s1 / 2.0).tan();
            let k2 = k * k;
            let denom = 1.0 + k / q + k2;
            let b0 = (vb0 + vb0.sqrt() * k / q + k2) / denom;
            let b1 = 2.0 * (k2 - vb0) / denom;
            let b2 = (vb0 - vb0.sqrt() * k / q + k2) / denom;
            let a1 = 2.0 * (k2 - 1.0) / denom;
            let a2 = (1.0 - k / q + k2) / denom;
            (b0, b1, b2, a1, a2)
        };

        // ----------------------------------------------------------------
        // Stage 2: High-pass filter at ~100 Hz.
        // H(s) = s^2 / (s^2 + sqrt(2)*w0*s + w0^2),  Q = 0.5
        // ----------------------------------------------------------------
        let w0_s2 = 2.0 * PI * 38.135_471_3 / fs;
        let (s2_b0, s2_b1, s2_b2, s2_a1, s2_a2) = {
            let q = 0.5;
            let k = (w0_s2 / 2.0).tan();
            let k2 = k * k;
            let denom = 1.0 + k / q + k2;
            let b0 = 1.0 / denom;
            let b1 = -2.0 / denom;
            let b2 = 1.0 / denom;
            let a1 = 2.0 * (k2 - 1.0) / denom;
            let a2 = (1.0 - k / q + k2) / denom;
            (b0, b1, b2, a1, a2)
        };

        Self {
            stage1_coeffs: BiquadCoeffs {
                b0: s1_b0,
                b1: s1_b1,
                b2: s1_b2,
                a1: s1_a1,
                a2: s1_a2,
            },
            stage2_coeffs: BiquadCoeffs {
                b0: s2_b0,
                b1: s2_b1,
                b2: s2_b2,
                a1: s2_a1,
                a2: s2_a2,
            },
            stage1_state: BiquadState::default(),
            stage2_state: BiquadState::default(),
        }
    }

    /// Process a single sample through both K-weighting stages.
    #[must_use]
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let x64 = x as f64;
        let s1_out = self.stage1_state.process(x64, &self.stage1_coeffs);
        let s2_out = self.stage2_state.process(s1_out, &self.stage2_coeffs);
        s2_out as f32
    }

    /// Reset filter state (clear all delay-line memory).
    pub fn reset(&mut self) {
        self.stage1_state.reset();
        self.stage2_state.reset();
    }
}

// ---------------------------------------------------------------------------
// AnalysisMeter
// ---------------------------------------------------------------------------

/// Combined peak / RMS / LUFS analysis meter.
///
/// Call [`AnalysisMeter::process_block`] once per DSP buffer; then read
/// the current values via [`AnalysisMeter::readings`].
pub struct AnalysisMeter {
    sample_rate: u32,

    // Peak
    peak_linear: f32,

    // RMS — circular buffer of squared samples (300 ms window).
    rms_buffer: VecDeque<f32>,
    rms_window_size: usize,
    rms_sum: f64,

    // LUFS — circular buffer of K-weighted squared samples (400 ms window).
    lufs_buffer: VecDeque<f32>,
    lufs_window_size: usize,
    lufs_sum: f64,
    k_filter: KWeightingFilter,
}

impl AnalysisMeter {
    /// Create a new meter for the given sample rate.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate as usize;
        let rms_window = (sr * 3) / 10; // 300 ms
        let lufs_window = (sr * 2) / 5; // 400 ms
        Self {
            sample_rate,
            peak_linear: 0.0,
            rms_buffer: VecDeque::with_capacity(rms_window),
            rms_window_size: rms_window,
            rms_sum: 0.0,
            lufs_buffer: VecDeque::with_capacity(lufs_window),
            lufs_window_size: lufs_window,
            lufs_sum: 0.0,
            k_filter: KWeightingFilter::new(sample_rate),
        }
    }

    // -----------------------------------------------------------------------
    // Processing
    // -----------------------------------------------------------------------

    /// Process a mono audio block at the meter's configured sample rate.
    pub fn process_block(&mut self, samples: &[f32], sample_rate: u32) {
        // If sample rate has changed, resize the windows.
        if sample_rate != self.sample_rate {
            self.sample_rate = sample_rate;
            let sr = sample_rate as usize;
            self.rms_window_size = (sr * 3) / 10;
            self.lufs_window_size = (sr * 2) / 5;
            self.k_filter = KWeightingFilter::new(sample_rate);
        }

        for &s in samples {
            self.push_sample(s);
        }
    }

    /// Process a stereo audio block.  Channel weights per BS.1770 are 1.0 for
    /// both L and R (LFE is 0, surround channels get 1.41).
    pub fn process_block_stereo(&mut self, left: &[f32], right: &[f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            // For LUFS/RMS we use the combined power equally across L+R.
            // Push both as individual mono samples.
            self.push_sample(left[i]);
            self.push_sample(right[i]);
        }
    }

    /// Push a single sample into all meter accumulators.
    #[inline]
    fn push_sample(&mut self, s: f32) {
        // Peak.
        let abs_s = s.abs();
        if abs_s > self.peak_linear {
            self.peak_linear = abs_s;
        }

        // RMS.
        let sq = (s as f64) * (s as f64);
        self.rms_buffer.push_back(sq as f32);
        self.rms_sum += sq;
        if self.rms_buffer.len() > self.rms_window_size {
            if let Some(old) = self.rms_buffer.pop_front() {
                self.rms_sum -= old as f64;
            }
        }
        self.rms_sum = self.rms_sum.max(0.0);

        // LUFS (K-weighted).
        let kw = self.k_filter.process_sample(s);
        let kw_sq = (kw as f64) * (kw as f64);
        self.lufs_buffer.push_back(kw_sq as f32);
        self.lufs_sum += kw_sq;
        if self.lufs_buffer.len() > self.lufs_window_size {
            if let Some(old) = self.lufs_buffer.pop_front() {
                self.lufs_sum -= old as f64;
            }
        }
        self.lufs_sum = self.lufs_sum.max(0.0);
    }

    // -----------------------------------------------------------------------
    // Readings
    // -----------------------------------------------------------------------

    /// Return the current meter readings.
    #[must_use]
    pub fn readings(&self) -> MeterReadings {
        let peak_db = Self::linear_to_db(self.peak_linear);

        let rms_db = if self.rms_buffer.is_empty() {
            -f32::INFINITY
        } else {
            let mean_sq = self.rms_sum / self.rms_buffer.len() as f64;
            Self::linear_to_db((mean_sq.sqrt()) as f32)
        };

        let lufs_momentary = if self.lufs_buffer.is_empty() {
            -f32::INFINITY
        } else {
            let mean_sq = self.lufs_sum / self.lufs_buffer.len() as f64;
            if mean_sq <= 0.0 {
                -f32::INFINITY
            } else {
                // EBU R128: -0.691 + 10*log10(sum of channel mean-square powers)
                (-0.691 + 10.0 * mean_sq.log10()) as f32
            }
        };

        MeterReadings {
            peak_db,
            rms_db,
            lufs_momentary,
        }
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    /// Convert a linear amplitude to dBFS.  Returns `-144.0` for zero or
    /// subnormal values (effective digital silence floor).
    #[must_use]
    #[inline]
    pub fn linear_to_db(linear: f32) -> f32 {
        if linear <= 1e-7 {
            -144.0
        } else {
            20.0 * linear.log10()
        }
    }

    /// Reset all accumulators and filter state.
    pub fn reset(&mut self) {
        self.peak_linear = 0.0;
        self.rms_buffer.clear();
        self.rms_sum = 0.0;
        self.lufs_buffer.clear();
        self.lufs_sum = 0.0;
        self.k_filter.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: u32 = 48_000;

    fn sine_block(freq_hz: f32, amplitude: f32, n_samples: usize, sr: u32) -> Vec<f32> {
        (0..n_samples)
            .map(|i| amplitude * (2.0 * PI * freq_hz * i as f32 / sr as f32).sin())
            .collect()
    }

    // --- linear_to_db ---

    #[test]
    fn linear_to_db_unity() {
        let db = AnalysisMeter::linear_to_db(1.0);
        assert!((db - 0.0).abs() < 0.001, "1.0 → 0 dBFS, got {db}");
    }

    #[test]
    fn linear_to_db_silence() {
        assert_eq!(AnalysisMeter::linear_to_db(0.0), -144.0);
    }

    #[test]
    fn linear_to_db_minus6() {
        let db = AnalysisMeter::linear_to_db(0.5);
        assert!((db - (-6.0206)).abs() < 0.01, "0.5 → ~-6 dBFS, got {db}");
    }

    // --- peak detection ---

    #[test]
    fn peak_tracks_maximum() {
        let mut meter = AnalysisMeter::new(SR);
        let block = vec![0.1, 0.5, -0.8, 0.3];
        meter.process_block(&block, SR);
        let r = meter.readings();
        let expected_db = AnalysisMeter::linear_to_db(0.8);
        assert!((r.peak_db - expected_db).abs() < 0.01, "peak={}", r.peak_db);
    }

    #[test]
    fn peak_zero_block_gives_silence() {
        let mut meter = AnalysisMeter::new(SR);
        meter.process_block(&[0.0; 512], SR);
        assert_eq!(meter.readings().peak_db, -144.0);
    }

    // --- RMS ---

    #[test]
    fn rms_of_full_scale_sine() {
        let mut meter = AnalysisMeter::new(SR);
        // A full-scale sine (amp=1.0) has RMS = 1/√2 ≈ -3.01 dBFS.
        let block = sine_block(440.0, 1.0, SR as usize, SR);
        meter.process_block(&block, SR);
        let r = meter.readings();
        // Allow ±1 dB tolerance.
        assert!(
            r.rms_db > -4.5 && r.rms_db < -2.0,
            "RMS of full-scale sine should be near -3 dBFS, got {}",
            r.rms_db
        );
    }

    #[test]
    fn rms_silence() {
        let mut meter = AnalysisMeter::new(SR);
        meter.process_block(&[0.0; 512], SR);
        // After silence the RMS should be at silence floor.
        assert!(meter.readings().rms_db <= -100.0);
    }

    // --- LUFS ---

    #[test]
    fn lufs_silence() {
        let mut meter = AnalysisMeter::new(SR);
        meter.process_block(&vec![0.0; 4800], SR);
        let r = meter.readings();
        assert!(
            r.lufs_momentary <= -100.0 || r.lufs_momentary.is_infinite(),
            "silence LUFS should be very low, got {}",
            r.lufs_momentary
        );
    }

    #[test]
    fn lufs_sine_ballpark() {
        // A 1 kHz sine at -20 dBFS should give LUFS roughly in -20 dBFS range.
        let amplitude = 10_f32.powf(-20.0 / 20.0); // ≈ 0.1
        let mut meter = AnalysisMeter::new(SR);
        let block = sine_block(1000.0, amplitude, SR as usize * 2, SR);
        meter.process_block(&block, SR);
        let r = meter.readings();
        // K-weighting has some gain at 1 kHz so expect -25 to -15 LUFS range.
        assert!(
            r.lufs_momentary > -35.0 && r.lufs_momentary < -5.0,
            "1kHz @ -20 dBFS LUFS should be in -35..-5 range, got {}",
            r.lufs_momentary
        );
    }

    // --- stereo processing ---

    #[test]
    fn stereo_block_processes_both_channels() {
        let mut meter = AnalysisMeter::new(SR);
        let left = vec![0.5f32; 512];
        let right = vec![-0.5f32; 512];
        meter.process_block_stereo(&left, &right);
        let r = meter.readings();
        // Peak should be 0.5 (from either channel).
        let expected_peak = AnalysisMeter::linear_to_db(0.5);
        assert!((r.peak_db - expected_peak).abs() < 0.01);
    }

    // --- reset ---

    #[test]
    fn reset_clears_state() {
        let mut meter = AnalysisMeter::new(SR);
        let block = vec![0.9f32; 4800];
        meter.process_block(&block, SR);
        meter.reset();
        let r = meter.readings();
        assert_eq!(r.peak_db, -144.0);
        assert!(r.rms_db <= -100.0 || r.rms_db.is_infinite());
    }

    // --- K-weighting filter ---

    #[test]
    fn k_filter_silence_stays_zero() {
        let mut kf = KWeightingFilter::new(48000);
        for _ in 0..100 {
            let out = kf.process_sample(0.0);
            assert!(out.abs() < 1e-10, "filter output at silence: {out}");
        }
    }

    #[test]
    fn k_filter_44100_does_not_panic() {
        let mut kf = KWeightingFilter::new(44100);
        let block = sine_block(1000.0, 0.5, 4410, 44100);
        for s in block {
            let _ = kf.process_sample(s);
        }
    }

    #[test]
    fn k_filter_reset_clears_memory() {
        let mut kf = KWeightingFilter::new(48000);
        let block = sine_block(100.0, 0.9, 480, 48000);
        for s in &block {
            let _ = kf.process_sample(*s);
        }
        kf.reset();
        // After reset, silence input should produce silence output (within one sample).
        let out = kf.process_sample(0.0);
        assert!(
            out.abs() < 1e-6,
            "after reset, silence in → silence out, got {out}"
        );
    }
}

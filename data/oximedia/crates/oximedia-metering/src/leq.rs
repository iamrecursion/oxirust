//! Leq (Equivalent Continuous Sound Level) meter.
//!
//! Leq is the time-weighted average SPL over an integration period. It represents
//! the hypothetical constant sound level that would contain the same total acoustic
//! energy as the time-varying sound over the measurement interval.
//!
//! # Standards
//!
//! - IEC 61672 (sound level meters)
//! - ITU-R BS.1770 compatible K-weighting option
//! - ISO 1996 environmental noise assessment
//!
//! # Mathematical Definition
//!
//! ```text
//! Leq(T) = 10 * log10( (1/T) * integral_0^T (p(t)/p0)^2 dt )
//! ```
//!
//! where p(t) is the instantaneous sound pressure and p0 is the reference.

use crate::{MeteringError, MeteringResult};
use std::f64::consts::PI;

/// Time weighting for SPL measurement.
///
/// Defines the time constant of the exponential averaging filter applied
/// before Leq computation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimeWeighting {
    /// Fast: 125 ms time constant (IEC 61672).
    Fast,
    /// Slow: 1000 ms time constant (IEC 61672).
    Slow,
    /// Impulse: 35 ms attack, 1500 ms release (IEC 61672).
    Impulse,
    /// Linear integration — no time weighting, true Leq over full interval.
    Linear,
    /// Custom time constant in milliseconds.
    Custom(f64),
}

impl TimeWeighting {
    /// Return the attack time constant in seconds, or `None` for `Linear`.
    #[must_use]
    pub fn attack_seconds(&self) -> Option<f64> {
        match self {
            Self::Fast => Some(0.125),
            Self::Slow => Some(1.000),
            Self::Impulse => Some(0.035),
            Self::Linear => None,
            Self::Custom(ms) => Some(ms / 1000.0),
        }
    }

    /// Return the release time constant in seconds (same as attack except Impulse).
    #[must_use]
    pub fn release_seconds(&self) -> Option<f64> {
        match self {
            Self::Impulse => Some(1.500),
            other => other.attack_seconds(),
        }
    }
}

/// Frequency weighting applied before Leq integration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeqWeighting {
    /// No frequency weighting (flat, Z-weighting).
    Flat,
    /// A-weighting (most common for environmental noise, dB(A)).
    A,
    /// C-weighting (for peak measurements, dB(C)).
    C,
    /// ITU-R BS.1770 K-weighting (broadcast loudness).
    KWeighted,
}

/// Second-order IIR biquad filter state for frequency weighting.
#[derive(Clone, Debug)]
struct BiquadState {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadState {
    fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Compute A-weighting biquad coefficients at the given sample rate.
///
/// The A-weighting filter is implemented as a cascade of four second-order
/// IIR sections derived by bilinear transform of the analog A-weighting
/// prototype.  The analog prototype has poles at:
///   f1 = 20.598997 Hz (double), f2 = 107.65265 Hz, f3 = 737.86223 Hz,
///   f4 = 12194.217 Hz (double)
/// and four zeros at DC plus a normalisation gain.
///
/// Coefficients use the direct bilinear transform of each real-pole pair,
/// representing the pair as a second-order section with a pair of zeros at
/// z = +1 (DC rejection) and a pair of poles computed from the analog values.
fn a_weighting_coefficients(sample_rate: f64) -> Vec<BiquadState> {
    // Analog pole frequencies (Hz)
    let f1: f64 = 20.598_997;
    let f2: f64 = 107.652_65;
    let f3: f64 = 737.862_23;
    let f4: f64 = 12_194.217;

    // Build one biquad from a single real analog pole pair (double pole at fc).
    // Using bilinear transform: s → 2*fs*(z-1)/(z+1), solving for H(z).
    // For a double real pole: H(s) = 1/(1 + s/w0)^2
    // which gives: H(z) = (kz)^2 / (1 + 2*a*z^-1 + a^2*z^-2) form after normalisation.
    let double_pole_biquad = |fc: f64| -> BiquadState {
        let w0 = 2.0 * PI * fc;
        let k = 2.0 * sample_rate;
        let denom = k * k + 2.0 * k * w0 + w0 * w0;
        let b0 = k * k / denom;
        let b1 = -2.0 * k * k / denom;
        let b2 = k * k / denom;
        let a1 = (2.0 * (w0 * w0 - k * k)) / denom;
        let a2 = (k * k - 2.0 * k * w0 + w0 * w0) / denom;
        BiquadState::new(b0, b1, b2, a1, a2)
    };

    // Build one biquad from two different real analog poles at fa and fb.
    // H(s) = 1/((1 + s/wa)(1 + s/wb))
    let pole_pair_biquad = |fa: f64, fb: f64| -> BiquadState {
        let wa = 2.0 * PI * fa;
        let wb = 2.0 * PI * fb;
        let k = 2.0 * sample_rate;
        let denom = k * k + k * (wa + wb) + wa * wb;
        let b0 = k * k / denom;
        let b1 = -2.0 * k * k / denom;
        let b2 = k * k / denom;
        let a1 = (2.0 * (wa * wb - k * k)) / denom;
        let a2 = (k * k - k * (wa + wb) + wa * wb) / denom;
        BiquadState::new(b0, b1, b2, a1, a2)
    };

    // A-weighting = section(f4 double) * section(f1 double) * section(f2, f3)
    // The f4 section provides high-frequency rolloff (2nd-order LP).
    // The f1 section provides low-frequency rolloff (2nd-order HP in analog → LP).
    // The f2/f3 section provides mid-band shaping.
    // All sections are high-pass in the sense that zeros at DC suppress low freqs.
    vec![
        double_pole_biquad(f4),
        double_pole_biquad(f1),
        pole_pair_biquad(f2, f3),
    ]
}

/// Compute C-weighting biquad coefficients.
///
/// C-weighting has a flatter midrange than A-weighting, with rolloff only
/// at very low (<20 Hz) and very high (>12 kHz) frequencies.
fn c_weighting_coefficients(sample_rate: f64) -> Vec<BiquadState> {
    let f1: f64 = 20.598_997;
    let f4: f64 = 12_194.217;
    let k = 2.0 * sample_rate;

    let double_pole_biquad = |fc: f64| -> BiquadState {
        let w0 = 2.0 * PI * fc;
        let denom = k * k + 2.0 * k * w0 + w0 * w0;
        let b0 = k * k / denom;
        let b1 = -2.0 * k * k / denom;
        let b2 = k * k / denom;
        let a1 = (2.0 * (w0 * w0 - k * k)) / denom;
        let a2 = (k * k - 2.0 * k * w0 + w0 * w0) / denom;
        BiquadState::new(b0, b1, b2, a1, a2)
    };

    vec![double_pole_biquad(f4), double_pole_biquad(f1)]
}

/// ITU-R BS.1770 K-weighting stage 1: high-shelf pre-filter at ~1.6 kHz.
fn k_weighting_stage1(sample_rate: f64) -> BiquadState {
    // Coefficients from ITU-R BS.1770-4 Annex 1 (48 kHz reference; scaled for other rates)
    // Using the bilinear transform of the analog prototype.
    let f0: f64 = 1681.974_450_955_533;
    let q: f64 = 0.707_220_906;
    let db_gain: f64 = 3.999_843_853_973_347;
    let k = (PI * f0 / sample_rate).tan();
    let vh = 10.0_f64.powf(db_gain / 20.0);
    let vb = vh.sqrt();
    let denom = 1.0 + k / q + k * k;
    let b0 = (vh + vb * k / q + k * k) / denom;
    let b1 = 2.0 * (k * k - vh) / denom;
    let b2 = (vh - vb * k / q + k * k) / denom;
    let a1 = 2.0 * (k * k - 1.0) / denom;
    let a2 = (1.0 - k / q + k * k) / denom;
    BiquadState::new(b0, b1, b2, a1, a2)
}

/// ITU-R BS.1770 K-weighting stage 2: high-pass at ~38 Hz.
fn k_weighting_stage2(sample_rate: f64) -> BiquadState {
    let f0: f64 = 38.135_470_662_138_47;
    let q: f64 = 0.500_327_458;
    let k = (PI * f0 / sample_rate).tan();
    let denom = k * k + k / q + 1.0;
    let b0 = 1.0 / denom;
    let b1 = -2.0 / denom;
    let b2 = 1.0 / denom;
    let a1 = 2.0 * (k * k - 1.0) / denom;
    let a2 = (k * k - k / q + 1.0) / denom;
    BiquadState::new(b0, b1, b2, a1, a2)
}

/// Per-channel filter chain state.
#[derive(Clone, Debug)]
struct ChannelFilters {
    stages: Vec<BiquadState>,
}

impl ChannelFilters {
    fn process(&mut self, sample: f64) -> f64 {
        let mut out = sample;
        for stage in &mut self.stages {
            out = stage.process(out);
        }
        out
    }

    fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
    }
}

/// Leq (Equivalent Continuous Sound Level) meter.
///
/// Computes the time-averaged mean square pressure (Leq) over a configurable
/// integration window. Supports A/C/K frequency weighting and IEC 61672
/// time weightings (Fast/Slow/Impulse) as well as true linear Leq.
pub struct LeqMeter {
    sample_rate: f64,
    channels: usize,
    weighting: LeqWeighting,
    time_weighting: TimeWeighting,
    /// Per-channel frequency-weighting filter chains.
    filters: Vec<ChannelFilters>,
    /// Per-channel running mean-square energy accumulators.
    energy_sum: Vec<f64>,
    /// Total number of samples accumulated per channel.
    sample_count: u64,
    /// Exponential smoothing coefficient (for non-Linear modes).
    alpha: Option<f64>,
    /// Smoothed (exponentially weighted) mean-square per channel.
    smoothed_ms: Vec<f64>,
    /// Reference sound pressure squared (linear, normalised to 1.0).
    reference_sq: f64,
}

impl LeqMeter {
    /// Create a new Leq meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (must be 8000–192000)
    /// * `channels` - Number of input channels (1–16)
    /// * `weighting` - Frequency weighting to apply before integration
    /// * `time_weighting` - Time weighting (averaging mode)
    ///
    /// # Errors
    ///
    /// Returns `MeteringError::InvalidConfig` for invalid parameters.
    pub fn new(
        sample_rate: f64,
        channels: usize,
        weighting: LeqWeighting,
        time_weighting: TimeWeighting,
    ) -> MeteringResult<Self> {
        if sample_rate < 8000.0 || sample_rate > 192_000.0 {
            return Err(MeteringError::InvalidConfig(format!(
                "Sample rate {sample_rate} Hz is out of valid range (8000–192000 Hz)"
            )));
        }
        if channels == 0 || channels > 16 {
            return Err(MeteringError::InvalidConfig(format!(
                "Channel count {channels} is out of valid range (1–16)"
            )));
        }

        let filter_stages: Vec<BiquadState> = match weighting {
            LeqWeighting::Flat => vec![],
            LeqWeighting::A => a_weighting_coefficients(sample_rate),
            LeqWeighting::C => c_weighting_coefficients(sample_rate),
            LeqWeighting::KWeighted => {
                vec![
                    k_weighting_stage1(sample_rate),
                    k_weighting_stage2(sample_rate),
                ]
            }
        };

        let filters: Vec<ChannelFilters> = (0..channels)
            .map(|_| ChannelFilters {
                stages: filter_stages.clone(),
            })
            .collect();

        // Compute exponential smoothing alpha from the time constant τ:
        // α = 1 - exp(-1 / (τ * fs))
        let alpha = time_weighting
            .attack_seconds()
            .map(|tau| 1.0 - (-1.0_f64 / (tau * sample_rate)).exp());

        Ok(Self {
            sample_rate,
            channels,
            weighting,
            time_weighting,
            filters,
            energy_sum: vec![0.0; channels],
            sample_count: 0,
            alpha,
            smoothed_ms: vec![0.0; channels],
            reference_sq: 1.0,
        })
    }

    /// Process interleaved audio samples.
    ///
    /// Samples should be normalised to the range −1.0 … +1.0.
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let frames = samples.len() / self.channels;
        for frame in 0..frames {
            for ch in 0..self.channels {
                let raw = samples[frame * self.channels + ch];
                let weighted = self.filters[ch].process(raw);
                let sq = weighted * weighted;

                // Linear accumulator (true Leq over full interval)
                self.energy_sum[ch] += sq;

                // Exponential smoother (Fast / Slow / Impulse)
                if let Some(alpha) = self.alpha {
                    self.smoothed_ms[ch] = alpha * sq + (1.0 - alpha) * self.smoothed_ms[ch];
                }
            }
            self.sample_count += 1;
        }
    }

    /// Process planar (non-interleaved) audio.
    ///
    /// `planes` is a slice of `channels` equal-length slices, one per channel.
    ///
    /// # Errors
    ///
    /// Returns `MeteringError::InvalidConfig` if the number of planes does not
    /// match the channel count or if planes have different lengths.
    pub fn process_planar(&mut self, planes: &[&[f64]]) -> MeteringResult<()> {
        if planes.len() != self.channels {
            return Err(MeteringError::ChannelError(format!(
                "Expected {} planes, got {}",
                self.channels,
                planes.len()
            )));
        }
        let frames = planes[0].len();
        if planes.iter().any(|p| p.len() != frames) {
            return Err(MeteringError::InvalidConfig(
                "All channel planes must have the same length".to_string(),
            ));
        }
        for frame in 0..frames {
            for ch in 0..self.channels {
                let raw = planes[ch][frame];
                let weighted = self.filters[ch].process(raw);
                let sq = weighted * weighted;
                self.energy_sum[ch] += sq;
                if let Some(alpha) = self.alpha {
                    self.smoothed_ms[ch] = alpha * sq + (1.0 - alpha) * self.smoothed_ms[ch];
                }
            }
            self.sample_count += 1;
        }
        Ok(())
    }

    /// Compute the integrated Leq for each channel over the full measurement
    /// period (true linear average, regardless of time weighting mode).
    ///
    /// Returns `f64::NEG_INFINITY` for channels with no accumulated energy.
    #[must_use]
    pub fn leq_per_channel(&self) -> Vec<f64> {
        if self.sample_count == 0 {
            return vec![f64::NEG_INFINITY; self.channels];
        }
        self.energy_sum
            .iter()
            .map(|&e| {
                let mean_sq = e / self.sample_count as f64;
                if mean_sq > 0.0 {
                    10.0 * (mean_sq / self.reference_sq).log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Compute the maximum Leq across all channels.
    #[must_use]
    pub fn leq_max(&self) -> f64 {
        self.leq_per_channel()
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Compute the mixed (channel-averaged) Leq.
    ///
    /// Averages the mean-square energies across channels before converting
    /// to dB, which is the physically correct way to combine channels.
    #[must_use]
    pub fn leq_mixed(&self) -> f64 {
        if self.sample_count == 0 {
            return f64::NEG_INFINITY;
        }
        let total_energy: f64 = self.energy_sum.iter().sum();
        let mean_sq = total_energy / (self.sample_count as f64 * self.channels as f64);
        if mean_sq > 0.0 {
            10.0 * (mean_sq / self.reference_sq).log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Get the current instantaneous (exponentially smoothed) Leq per channel.
    ///
    /// Only meaningful when time weighting is not `Linear`.
    #[must_use]
    pub fn instantaneous_leq_per_channel(&self) -> Vec<f64> {
        self.smoothed_ms
            .iter()
            .map(|&ms| {
                if ms > 0.0 {
                    10.0 * (ms / self.reference_sq).log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Get the instantaneous (smoothed) Leq maximum across channels.
    #[must_use]
    pub fn instantaneous_leq_max(&self) -> f64 {
        self.instantaneous_leq_per_channel()
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Duration of accumulated audio in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.sample_count as f64 / self.sample_rate
    }

    /// Total sample count (per channel).
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Sample rate in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Number of channels.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Frequency weighting in use.
    #[must_use]
    pub fn weighting(&self) -> LeqWeighting {
        self.weighting
    }

    /// Time weighting in use.
    #[must_use]
    pub fn time_weighting(&self) -> TimeWeighting {
        self.time_weighting
    }

    /// Reset all accumulators and filter state to zero.
    pub fn reset(&mut self) {
        self.energy_sum.fill(0.0);
        self.smoothed_ms.fill(0.0);
        self.sample_count = 0;
        for ch_filters in &mut self.filters {
            ch_filters.reset();
        }
    }
}

/// Result snapshot from a completed Leq measurement.
#[derive(Clone, Debug)]
pub struct LeqResult {
    /// Integrated Leq per channel in dB.
    pub leq_per_channel: Vec<f64>,
    /// Maximum Leq across channels in dB.
    pub leq_max: f64,
    /// Channel-averaged Leq in dB.
    pub leq_mixed: f64,
    /// Duration of the measurement in seconds.
    pub duration_seconds: f64,
    /// Frequency weighting used.
    pub weighting: LeqWeighting,
    /// Time weighting used.
    pub time_weighting: TimeWeighting,
}

impl LeqResult {
    /// Snapshot the current state of the meter into a result record.
    #[must_use]
    pub fn from_meter(meter: &LeqMeter) -> Self {
        Self {
            leq_per_channel: meter.leq_per_channel(),
            leq_max: meter.leq_max(),
            leq_mixed: meter.leq_mixed(),
            duration_seconds: meter.duration_seconds(),
            weighting: meter.weighting(),
            time_weighting: meter.time_weighting(),
        }
    }

    /// Check whether the measurement exceeds a given limit in dB.
    #[must_use]
    pub fn exceeds_limit(&self, limit_db: f64) -> bool {
        self.leq_max > limit_db
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate a mono sine wave at the given frequency and amplitude.
    fn sine_mono(freq_hz: f64, amplitude: f64, sample_rate: f64, frames: usize) -> Vec<f64> {
        (0..frames)
            .map(|i| amplitude * (2.0 * PI * freq_hz * i as f64 / sample_rate).sin())
            .collect()
    }

    // --- Construction ---

    #[test]
    fn test_new_valid() {
        LeqMeter::new(48000.0, 2, LeqWeighting::Flat, TimeWeighting::Linear)
            .expect("should construct successfully");
    }

    #[test]
    fn test_new_invalid_sample_rate() {
        assert!(LeqMeter::new(1000.0, 2, LeqWeighting::Flat, TimeWeighting::Linear).is_err());
    }

    #[test]
    fn test_new_invalid_channels() {
        assert!(LeqMeter::new(48000.0, 0, LeqWeighting::Flat, TimeWeighting::Linear).is_err());
        assert!(LeqMeter::new(48000.0, 17, LeqWeighting::Flat, TimeWeighting::Linear).is_err());
    }

    // --- Silence gives NEG_INFINITY ---

    #[test]
    fn test_leq_silence_is_neg_inf() {
        let meter =
            LeqMeter::new(48000.0, 1, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        assert_eq!(meter.leq_max(), f64::NEG_INFINITY);
        assert_eq!(meter.leq_mixed(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_leq_no_samples_per_channel() {
        let meter =
            LeqMeter::new(48000.0, 2, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        let per_ch = meter.leq_per_channel();
        assert_eq!(per_ch.len(), 2);
        assert!(per_ch.iter().all(|v| v.is_infinite() && *v < 0.0));
    }

    // --- Full-scale DC gives 0 dB ---

    #[test]
    fn test_leq_full_scale_dc() {
        let mut meter =
            LeqMeter::new(48000.0, 1, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        let samples: Vec<f64> = vec![1.0; 4800]; // 100 ms of +1.0
        meter.process_interleaved(&samples);
        let leq = meter.leq_max();
        // mean square = 1.0 → Leq = 10*log10(1) = 0 dB
        assert!(leq.is_finite(), "Leq should be finite for DC signal");
        assert!(
            (leq - 0.0).abs() < 0.01,
            "Leq of unit DC should be ~0 dB, got {leq}"
        );
    }

    // --- Sine wave RMS ---

    #[test]
    fn test_leq_sine_wave_approx_minus_3db() {
        let sr = 48000.0;
        let mut meter =
            LeqMeter::new(sr, 1, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        // Full-scale sine: RMS = 1/√2 → mean square = 0.5 → Leq = 10*log10(0.5) ≈ -3.01 dB
        let frames = 48000; // 1 second
        let samples = sine_mono(1000.0, 1.0, sr, frames);
        meter.process_interleaved(&samples);
        let leq = meter.leq_max();
        assert!(leq.is_finite());
        assert!(
            (leq - (-3.01)).abs() < 0.1,
            "Expected ~-3 dB for full-scale sine, got {leq}"
        );
    }

    // --- Duration ---

    #[test]
    fn test_duration_seconds() {
        let sr = 48000.0;
        let mut meter =
            LeqMeter::new(sr, 1, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        let samples: Vec<f64> = vec![0.5; 48000];
        meter.process_interleaved(&samples);
        let dur = meter.duration_seconds();
        assert!((dur - 1.0).abs() < 1e-9, "Expected 1 s, got {dur}");
    }

    // --- Reset ---

    #[test]
    fn test_reset_clears_state() {
        let mut meter =
            LeqMeter::new(48000.0, 1, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        let samples: Vec<f64> = vec![1.0; 4800];
        meter.process_interleaved(&samples);
        meter.reset();
        assert_eq!(meter.sample_count(), 0);
        assert_eq!(meter.leq_max(), f64::NEG_INFINITY);
    }

    // --- Multi-channel ---

    #[test]
    fn test_multichannel_leq() {
        let sr = 48000.0;
        let frames = 48000;
        let mut meter =
            LeqMeter::new(sr, 2, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        // L channel: amplitude 1.0, R channel: amplitude 0.5
        let mut samples = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let t = i as f64 / sr;
            samples.push((2.0 * PI * 1000.0 * t).sin()); // L
            samples.push(0.5 * (2.0 * PI * 1000.0 * t).sin()); // R
        }
        meter.process_interleaved(&samples);
        let per_ch = meter.leq_per_channel();
        assert_eq!(per_ch.len(), 2);
        // L should be louder than R
        assert!(per_ch[0] > per_ch[1], "L={} R={}", per_ch[0], per_ch[1]);
    }

    // --- Planar input ---

    #[test]
    fn test_planar_input() {
        let sr = 48000.0;
        let frames = 4800;
        let mut meter =
            LeqMeter::new(sr, 2, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        let ch0: Vec<f64> = vec![0.5; frames];
        let ch1: Vec<f64> = vec![0.25; frames];
        meter
            .process_planar(&[&ch0, &ch1])
            .expect("planar process failed");
        assert_eq!(meter.sample_count(), frames as u64);
        let per_ch = meter.leq_per_channel();
        assert!(per_ch[0] > per_ch[1]);
    }

    #[test]
    fn test_planar_wrong_channel_count() {
        let mut meter =
            LeqMeter::new(48000.0, 2, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        let ch: Vec<f64> = vec![0.0; 100];
        assert!(meter.process_planar(&[&ch]).is_err());
    }

    // --- Time weighting alpha ---

    #[test]
    fn test_fast_time_weighting() {
        LeqMeter::new(48000.0, 1, LeqWeighting::Flat, TimeWeighting::Fast)
            .expect("Fast mode should construct");
    }

    #[test]
    fn test_slow_time_weighting() {
        LeqMeter::new(48000.0, 1, LeqWeighting::Flat, TimeWeighting::Slow)
            .expect("Slow mode should construct");
    }

    #[test]
    fn test_impulse_time_weighting() {
        LeqMeter::new(48000.0, 1, LeqWeighting::Flat, TimeWeighting::Impulse)
            .expect("Impulse mode should construct");
    }

    #[test]
    fn test_custom_time_weighting() {
        LeqMeter::new(48000.0, 1, LeqWeighting::Flat, TimeWeighting::Custom(500.0))
            .expect("Custom 500 ms should construct");
    }

    // --- Frequency weightings ---

    #[test]
    fn test_a_weighting() {
        let mut meter =
            LeqMeter::new(48000.0, 1, LeqWeighting::A, TimeWeighting::Linear).expect("valid");
        let samples = sine_mono(1000.0, 0.5, 48000.0, 48000);
        meter.process_interleaved(&samples);
        assert!(meter.leq_max().is_finite());
    }

    #[test]
    fn test_c_weighting() {
        let mut meter =
            LeqMeter::new(48000.0, 1, LeqWeighting::C, TimeWeighting::Linear).expect("valid");
        let samples = sine_mono(1000.0, 0.5, 48000.0, 48000);
        meter.process_interleaved(&samples);
        assert!(meter.leq_max().is_finite());
    }

    #[test]
    fn test_k_weighting() {
        let mut meter = LeqMeter::new(48000.0, 1, LeqWeighting::KWeighted, TimeWeighting::Linear)
            .expect("valid");
        let samples = sine_mono(1000.0, 0.5, 48000.0, 48000);
        meter.process_interleaved(&samples);
        assert!(meter.leq_max().is_finite());
    }

    // --- LeqResult ---

    #[test]
    fn test_leq_result_snapshot() {
        let mut meter =
            LeqMeter::new(48000.0, 1, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        let samples: Vec<f64> = vec![0.5; 4800];
        meter.process_interleaved(&samples);
        let result = LeqResult::from_meter(&meter);
        assert_eq!(result.leq_per_channel.len(), 1);
        assert!(result.leq_max.is_finite());
        assert!(!result.exceeds_limit(10.0));
    }

    #[test]
    fn test_leq_result_exceeds_limit() {
        let mut meter =
            LeqMeter::new(48000.0, 1, LeqWeighting::Flat, TimeWeighting::Linear).expect("valid");
        let samples: Vec<f64> = vec![1.0; 4800];
        meter.process_interleaved(&samples);
        let result = LeqResult::from_meter(&meter);
        // Leq of unit DC is 0 dB, which exceeds -10 dB limit
        assert!(result.exceeds_limit(-10.0));
    }

    // --- Instantaneous (smoothed) Leq ---

    #[test]
    fn test_instantaneous_leq_after_signal() {
        let sr = 48000.0;
        let mut meter =
            LeqMeter::new(sr, 1, LeqWeighting::Flat, TimeWeighting::Fast).expect("valid");
        let samples = sine_mono(1000.0, 0.5, sr, 4800);
        meter.process_interleaved(&samples);
        let inst = meter.instantaneous_leq_max();
        assert!(
            inst.is_finite(),
            "Instantaneous Leq should be finite after signal"
        );
    }

    // --- Accessors ---

    #[test]
    fn test_accessors() {
        let meter = LeqMeter::new(44100.0, 4, LeqWeighting::A, TimeWeighting::Slow).expect("valid");
        assert_eq!(meter.channels(), 4);
        assert!((meter.sample_rate() - 44100.0).abs() < 1e-9);
        assert_eq!(meter.weighting(), LeqWeighting::A);
        assert_eq!(meter.time_weighting(), TimeWeighting::Slow);
        assert_eq!(meter.sample_count(), 0);
    }
}

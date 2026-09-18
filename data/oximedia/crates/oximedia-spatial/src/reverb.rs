//! Algorithmic reverberation — Schroeder/Moorer style late reverb.
//!
//! This module provides a classic parallel-comb + series-allpass reverberator
//! following the architecture described by Schroeder (1962) and refined by
//! Moorer (1979).
//!
//! # Architecture
//!
//! ```text
//! input ──┬──► comb 0 ──┐
//!         ├──► comb 1 ──┤
//!         ├──► comb 2 ──┤
//!         ├──► comb 3 ──┤  sum  ──► allpass 0 ──► allpass 1 ──► output
//!         ├──► comb 4 ──┤
//!         ├──► comb 5 ──┤
//!         ├──► comb 6 ──┤
//!         └──► comb 7 ──┘
//! ```
//!
//! All delay-line lengths are prime numbers chosen to avoid harmonic relationships.
//! The feedback coefficient of each comb filter is derived from the desired RT60.
//!
//! # Reference
//! Schroeder, M. R. (1962). "Natural-sounding artificial reverberation."
//! Journal of the Audio Engineering Society, 10(3), 219–223.
//!
//! Moorer, J. A. (1979). "About this reverberation business."
//! Computer Music Journal, 3(2), 13–28.

use crate::SpatialError;

// ─── Comb filter ──────────────────────────────────────────────────────────────

/// A feedback comb filter with optional damping (low-pass in the feedback path).
///
/// The transfer function is:
/// ```text
/// H(z) = z^{-M} / (1 - g * d(z) * z^{-M})
/// ```
/// where `g` is the feedback gain and `d(z)` is a first-order low-pass filter
/// (one-pole) used to model high-frequency damping in the reverberant field.
#[derive(Debug, Clone)]
pub struct CombFilter {
    /// Delay-line buffer.
    buf: Vec<f32>,
    /// Write index into `buf`.
    write_pos: usize,
    /// Feedback gain (controls decay time).
    feedback: f32,
    /// Low-pass damping coefficient ∈ [0, 1].
    ///
    /// `0.0` = no damping (flat feedback), `1.0` = maximum high-frequency damping.
    damping: f32,
    /// One-pole filter state for the damping filter.
    lp_state: f32,
}

impl CombFilter {
    /// Create a new comb filter.
    ///
    /// # Parameters
    /// - `delay_samples`: delay-line length in samples.
    /// - `feedback`: feedback coefficient.  Must satisfy `|feedback| < 1` for stability.
    /// - `damping`: high-frequency damping coefficient ∈ [0, 1].
    pub fn new(delay_samples: usize, feedback: f32, damping: f32) -> Self {
        Self {
            buf: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            feedback: feedback.clamp(-0.99, 0.99),
            damping: damping.clamp(0.0, 1.0),
            lp_state: 0.0,
        }
    }

    /// Process one sample and return the filtered output.
    pub fn process(&mut self, input: f32) -> f32 {
        let n = self.buf.len();
        let output = self.buf[self.write_pos];

        // One-pole low-pass in the feedback path: y = (1-d)*x + d*y_prev
        let damped = output * (1.0 - self.damping) + self.lp_state * self.damping;
        self.lp_state = damped;

        self.buf[self.write_pos] = input + damped * self.feedback;
        self.write_pos = (self.write_pos + 1) % n;

        output
    }

    /// Reset the delay line and filter state.
    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write_pos = 0;
        self.lp_state = 0.0;
    }

    /// Update the feedback coefficient (e.g., when RT60 changes).
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(-0.99, 0.99);
    }

    /// Return the current delay-line length in samples.
    pub fn delay_samples(&self) -> usize {
        self.buf.len()
    }
}

// ─── All-pass filter ──────────────────────────────────────────────────────────

/// A Schroeder all-pass filter.
///
/// The all-pass filter has a flat magnitude response but introduces
/// frequency-dependent phase dispersion, smearing transients to create
/// the sensation of diffusion:
///
/// ```text
/// y[n] = -g * x[n] + x[n-M] + g * y[n-M]
/// ```
#[derive(Debug, Clone)]
pub struct AllPassFilter {
    /// Delay-line buffer.
    buf: Vec<f32>,
    /// Write index.
    write_pos: usize,
    /// All-pass gain coefficient ∈ [0, 1).
    gain: f32,
}

impl AllPassFilter {
    /// Create a new all-pass filter.
    pub fn new(delay_samples: usize, gain: f32) -> Self {
        Self {
            buf: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            gain: gain.clamp(0.0, 0.99),
        }
    }

    /// Process one sample.
    pub fn process(&mut self, input: f32) -> f32 {
        let n = self.buf.len();
        let delayed = self.buf[self.write_pos];

        let output = -self.gain * input + delayed;
        self.buf[self.write_pos] = input + delayed * self.gain;
        self.write_pos = (self.write_pos + 1) % n;

        output
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write_pos = 0;
    }
}

// ─── ReverbPreset ─────────────────────────────────────────────────────────────

/// Acoustic preset selecting the reverb character.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReverbPreset {
    /// Small room (RT60 ≈ 0.3 s).
    SmallRoom,
    /// Medium room / studio (RT60 ≈ 0.6 s).
    MediumRoom,
    /// Large hall (RT60 ≈ 1.5 s).
    LargeHall,
    /// Cathedral / very large space (RT60 ≈ 3.0 s).
    Cathedral,
    /// Custom: supply `rt60_secs` and `damping` explicitly.
    Custom {
        /// Desired RT60 (60 dB decay time) in seconds.
        rt60_secs: f32,
        /// High-frequency damping ∈ [0, 1].
        damping: f32,
    },
}

impl ReverbPreset {
    fn rt60_secs(self) -> f32 {
        match self {
            Self::SmallRoom => 0.3,
            Self::MediumRoom => 0.6,
            Self::LargeHall => 1.5,
            Self::Cathedral => 3.0,
            Self::Custom { rt60_secs, .. } => rt60_secs.max(0.01),
        }
    }

    fn damping(self) -> f32 {
        match self {
            Self::SmallRoom => 0.5,
            Self::MediumRoom => 0.3,
            Self::LargeHall => 0.15,
            Self::Cathedral => 0.05,
            Self::Custom { damping, .. } => damping.clamp(0.0, 1.0),
        }
    }
}

// ─── Comb filter delay-line lengths (in samples at 48 kHz) ───────────────────

/// Prime delay-line lengths for the 8 parallel comb filters (at 48 kHz).
///
/// Chosen as small prime numbers in the range 1400–1800 ms to give decorrelated
/// echoes without beating artefacts.  Values are scaled proportionally for
/// other sample rates.
const COMB_DELAYS_48K: [usize; 8] = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116];

/// All-pass filter delay-line lengths (at 48 kHz).
const ALLPASS_DELAYS_48K: [usize; 2] = [225, 556];

/// All-pass gain coefficient.
const ALLPASS_GAIN: f32 = 0.5;

// ─── SchroederReverb ─────────────────────────────────────────────────────────

/// Schroeder/Moorer-style algorithmic reverberator.
///
/// Uses 8 parallel feedback comb filters summed into 2 series all-pass filters.
/// The comb filter delays are prime-number lengths to avoid resonances, and each
/// comb filter has an independent low-pass damping filter in the feedback path.
#[derive(Debug, Clone)]
pub struct SchroederReverb {
    /// 8 parallel feedback comb filters.
    combs: Vec<CombFilter>,
    /// 2 series all-pass filters.
    allpasses: Vec<AllPassFilter>,
    /// Dry/wet mix ∈ [0, 1]. `0.0` = dry only, `1.0` = wet only.
    pub wet_mix: f32,
    /// Overall output gain.
    pub output_gain: f32,
    /// Sample rate (Hz).
    sample_rate: u32,
}

impl SchroederReverb {
    /// Construct a new reverberator.
    ///
    /// # Parameters
    /// - `sample_rate`: audio sample rate in Hz.
    /// - `preset`: acoustic character preset.
    /// - `wet_mix`: dry/wet ratio ∈ [0, 1].
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if `sample_rate` is zero.
    pub fn new(sample_rate: u32, preset: ReverbPreset, wet_mix: f32) -> Result<Self, SpatialError> {
        if sample_rate == 0 {
            return Err(SpatialError::InvalidConfig(
                "Sample rate must be > 0".into(),
            ));
        }

        let scale = sample_rate as f32 / 48_000.0;
        let rt60 = preset.rt60_secs();
        let damping = preset.damping();

        let combs = COMB_DELAYS_48K
            .iter()
            .map(|&d| {
                let delay = ((d as f32 * scale).round() as usize).max(1);
                let feedback = comb_feedback(rt60, delay, sample_rate);
                CombFilter::new(delay, feedback, damping)
            })
            .collect();

        let allpasses = ALLPASS_DELAYS_48K
            .iter()
            .map(|&d| {
                let delay = ((d as f32 * scale).round() as usize).max(1);
                AllPassFilter::new(delay, ALLPASS_GAIN)
            })
            .collect();

        Ok(Self {
            combs,
            allpasses,
            wet_mix: wet_mix.clamp(0.0, 1.0),
            output_gain: 1.0,
            sample_rate,
        })
    }

    /// Process a mono audio buffer in-place, adding reverberation.
    ///
    /// The output is a blend of `wet_mix * wet + (1 - wet_mix) * dry`.
    pub fn process_mono(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            let dry = *sample;
            let wet = self.process_sample(dry);
            *sample = (1.0 - self.wet_mix) * dry + self.wet_mix * wet * self.output_gain;
        }
    }

    /// Process a mono input and return a wet output buffer (same length as `input`).
    pub fn process_mono_wet(&mut self, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|&s| {
                let wet = self.process_sample(s);
                (1.0 - self.wet_mix) * s + self.wet_mix * wet * self.output_gain
            })
            .collect()
    }

    /// Process a stereo pair, applying the same reverb to both channels independently.
    ///
    /// Returns `(left_out, right_out)`.
    pub fn process_stereo(&mut self, left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let left_wet = self.process_mono_wet(left);
        let right_wet = self.process_mono_wet(right);
        (left_wet, right_wet)
    }

    /// Reset all filter states (e.g., at the start of a new segment).
    pub fn reset(&mut self) {
        for comb in &mut self.combs {
            comb.reset();
        }
        for ap in &mut self.allpasses {
            ap.reset();
        }
    }

    /// Apply a new preset (recomputes feedback coefficients without reallocating buffers).
    pub fn apply_preset(&mut self, preset: ReverbPreset) {
        let rt60 = preset.rt60_secs();
        let damping = preset.damping();
        for comb in &mut self.combs {
            let fb = comb_feedback(rt60, comb.delay_samples(), self.sample_rate);
            comb.set_feedback(fb);
            comb.damping = damping;
        }
    }

    /// Return the current sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Process a single sample through the reverb network.
    fn process_sample(&mut self, input: f32) -> f32 {
        // Sum 8 parallel comb outputs.
        let comb_sum: f32 =
            self.combs.iter_mut().map(|c| c.process(input)).sum::<f32>() / self.combs.len() as f32;

        // Series all-pass filters for diffusion.
        let mut out = comb_sum;
        for ap in &mut self.allpasses {
            out = ap.process(out);
        }

        out
    }
}

// ─── Helper: feedback coefficient for a given RT60 ───────────────────────────

/// Compute the comb filter feedback coefficient that produces the target RT60.
///
/// From the Schroeder model: `g = 10^(-3 * delay_secs / rt60)`.
fn comb_feedback(rt60_secs: f32, delay_samples: usize, sample_rate: u32) -> f32 {
    let delay_secs = delay_samples as f32 / sample_rate as f32;
    let rt60 = rt60_secs.max(0.001);
    10.0_f32.powf(-3.0 * delay_secs / rt60)
}

// ─── Early Reflections ────────────────────────────────────────────────────────

/// A single early reflection tap.
#[derive(Debug, Clone, Copy)]
pub struct ReflectionTap {
    /// Delay in samples.
    pub delay_samples: usize,
    /// Gain (can be negative for reflected polarity).
    pub gain: f32,
}

/// Early reflection processor using a multi-tap delay line.
///
/// Models the first few reflections off walls before the late reverberant field.
#[derive(Debug, Clone)]
pub struct EarlyReflections {
    /// Delay line buffer.
    buf: Vec<f32>,
    /// Current write index.
    write_pos: usize,
    /// Reflection taps.
    pub taps: Vec<ReflectionTap>,
}

impl EarlyReflections {
    /// Create an early reflections processor.
    ///
    /// `taps` defines the individual reflection delays and gains.
    /// The buffer is sized to accommodate the maximum tap delay.
    pub fn new(taps: Vec<ReflectionTap>) -> Self {
        let max_delay = taps.iter().map(|t| t.delay_samples).max().unwrap_or(0);
        Self {
            buf: vec![0.0; (max_delay + 1).max(2)],
            write_pos: 0,
            taps,
        }
    }

    /// Build a simple room early-reflection model from room dimensions.
    ///
    /// Generates reflections from the 6 room surfaces (left, right, front, back,
    /// floor, ceiling) using first-order image-source approximation.
    ///
    /// # Parameters
    /// - `room_metres`: `[width, depth, height]` in metres.
    /// - `listener_pos`: normalised listener position `[x, y, z]` ∈ [0, 1].
    /// - `sample_rate`: audio sample rate in Hz.
    pub fn from_room(room_metres: [f32; 3], listener_pos: [f32; 3], sample_rate: u32) -> Self {
        let speed = 343.0_f32;
        let [w, d, h] = room_metres;
        let [lx, ly, lz] = listener_pos;

        // First-order image distances for each of the 6 walls.
        let distances = [
            lx * w,         // left wall
            (1.0 - lx) * w, // right wall
            ly * d,         // front wall
            (1.0 - ly) * d, // back wall
            lz * h,         // floor
            (1.0 - lz) * h, // ceiling
        ];

        let taps: Vec<ReflectionTap> = distances
            .iter()
            .enumerate()
            .map(|(i, &dist)| {
                let dist = dist.max(0.01);
                let delay = (dist / speed * sample_rate as f32).round() as usize;
                // Alternate reflections flip polarity; gain falls as 1/distance.
                let polarity = if i % 2 == 0 { 1.0 } else { -1.0 };
                let gain = polarity * (1.0 / dist.max(0.1)).min(1.0);
                ReflectionTap {
                    delay_samples: delay.max(1),
                    gain,
                }
            })
            .collect();

        Self::new(taps)
    }

    /// Process one sample and return the summed early reflection output.
    pub fn process(&mut self, input: f32) -> f32 {
        let n = self.buf.len();
        self.buf[self.write_pos] = input;

        let out = self
            .taps
            .iter()
            .map(|tap| {
                let idx = (self.write_pos + n - tap.delay_samples.min(n - 1)) % n;
                self.buf[idx] * tap.gain
            })
            .sum();

        self.write_pos = (self.write_pos + 1) % n;
        out
    }

    /// Process a block of samples.
    pub fn process_block(&mut self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&s| self.process(s)).collect()
    }

    /// Reset the delay line.
    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write_pos = 0;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(n: usize, freq_hz: f32, sr: u32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sr as f32).sin())
            .collect()
    }

    // ── CombFilter ──────────────────────────────────────────────────────────

    #[test]
    fn test_comb_filter_output_delayed() {
        let mut comb = CombFilter::new(10, 0.5, 0.0);
        // First 10 samples should be zero (delay).
        for _ in 0..10 {
            let out = comb.process(1.0);
            assert_eq!(out, 0.0, "Comb output should be zero within delay");
        }
        // 11th sample should be nonzero (delayed input fed back).
        let out = comb.process(0.0);
        assert!(out.abs() > 0.0, "Comb output should be nonzero after delay");
    }

    #[test]
    fn test_comb_filter_reset_clears_state() {
        let mut comb = CombFilter::new(10, 0.5, 0.0);
        for _ in 0..20 {
            comb.process(1.0);
        }
        comb.reset();
        let out = comb.process(0.0);
        assert_eq!(out, 0.0, "Reset comb should output silence");
    }

    #[test]
    fn test_comb_delay_samples_accessor() {
        let comb = CombFilter::new(57, 0.7, 0.3);
        assert_eq!(comb.delay_samples(), 57);
    }

    #[test]
    fn test_comb_damping_attenuates_high_freq() {
        // With damping=1 (max) the feedback path is heavily filtered.
        let mut comb_undamped = CombFilter::new(8, 0.9, 0.0);
        let mut comb_damped = CombFilter::new(8, 0.9, 0.99);
        let mut sum_u = 0.0_f32;
        let mut sum_d = 0.0_f32;
        for i in 0..64 {
            let inp = if i == 0 { 1.0 } else { 0.0 };
            sum_u += comb_undamped.process(inp).abs();
            sum_d += comb_damped.process(inp).abs();
        }
        assert!(
            sum_d <= sum_u + 0.01,
            "Damped comb should not exceed undamped energy: damped={sum_d}, undamped={sum_u}"
        );
    }

    // ── AllPassFilter ────────────────────────────────────────────────────────

    #[test]
    fn test_allpass_preserves_energy_approx() {
        // All-pass should pass DC (energy-neutral for DC input over long time).
        let mut ap = AllPassFilter::new(5, 0.5);
        let mut sum_in = 0.0_f32;
        let mut sum_out = 0.0_f32;
        for i in 0..100 {
            let s = (i as f32 * 0.1).sin();
            sum_in += s * s;
            sum_out += ap.process(s).powi(2);
        }
        // For a sinusoidal input the all-pass should not drastically change energy.
        assert!(
            sum_out > sum_in * 0.1,
            "AllPass should pass energy, in={sum_in}, out={sum_out}"
        );
    }

    #[test]
    fn test_allpass_reset_clears_state() {
        let mut ap = AllPassFilter::new(10, 0.5);
        for _ in 0..50 {
            ap.process(1.0);
        }
        ap.reset();
        let out = ap.process(0.0);
        assert_eq!(out, 0.0, "Reset allpass should output silence");
    }

    // ── SchroederReverb ──────────────────────────────────────────────────────

    #[test]
    fn test_reverb_new_valid() {
        let rev = SchroederReverb::new(48_000, ReverbPreset::MediumRoom, 0.3);
        assert!(rev.is_ok(), "Should construct successfully");
    }

    #[test]
    fn test_reverb_new_zero_sr_fails() {
        let rev = SchroederReverb::new(0, ReverbPreset::MediumRoom, 0.3);
        assert!(rev.is_err(), "Zero sample rate should be invalid");
    }

    #[test]
    fn test_reverb_output_length_matches_input() {
        let mut rev = SchroederReverb::new(48_000, ReverbPreset::SmallRoom, 0.5)
            .expect("small room reverb creation should succeed");
        let input = sine_wave(256, 440.0, 48_000);
        let output = rev.process_mono_wet(&input);
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_reverb_wet_zero_passes_dry() {
        let mut rev = SchroederReverb::new(48_000, ReverbPreset::SmallRoom, 0.0)
            .expect("small room reverb with zero wet should succeed");
        let input = vec![0.5_f32; 64];
        let output = rev.process_mono_wet(&input);
        // wet_mix=0 → output = dry signal (0.5)
        for &s in &output {
            assert!(
                (s - 0.5).abs() < 0.01,
                "wet=0 should pass dry signal, got {s}"
            );
        }
    }

    #[test]
    fn test_reverb_adds_energy_after_signal_ends() {
        let mut rev = SchroederReverb::new(48_000, ReverbPreset::MediumRoom, 1.0)
            .expect("medium room reverb creation should succeed");
        // Play a short impulse, then silence — reverb tail should add energy.
        // The shortest comb delay is ~1116 samples at 48 kHz, so we need a buffer
        // longer than the maximum comb delay (1557 samples) to observe any tail.
        let mut impulse_block = vec![0.0_f32; 2048];
        impulse_block[0] = 1.0;
        let out = rev.process_mono_wet(&impulse_block);

        // Check that there is energy after the first comb reflection (past 1116 samples).
        let tail_energy: f32 = out[1200..].iter().map(|x| x * x).sum();
        assert!(
            tail_energy > 0.0,
            "Reverb should produce a tail after impulse, tail_energy={tail_energy}"
        );
    }

    #[test]
    fn test_reverb_reset_silences_output() {
        let mut rev = SchroederReverb::new(48_000, ReverbPreset::LargeHall, 0.5)
            .expect("large hall reverb creation should succeed");
        // Drive with a sine to fill the delay lines.
        let sine = sine_wave(2048, 440.0, 48_000);
        rev.process_mono_wet(&sine);
        rev.reset();
        // After reset, silence in → silence out (or close to it).
        let silence = vec![0.0_f32; 64];
        let out = rev.process_mono_wet(&silence);
        let energy: f32 = out.iter().map(|x| x * x).sum();
        assert_eq!(energy, 0.0, "Reset reverb should output silence");
    }

    #[test]
    fn test_reverb_output_finite() {
        let mut rev = SchroederReverb::new(44_100, ReverbPreset::Cathedral, 0.8)
            .expect("cathedral reverb creation should succeed");
        let sine = sine_wave(512, 220.0, 44_100);
        let out = rev.process_mono_wet(&sine);
        for (i, &s) in out.iter().enumerate() {
            assert!(s.is_finite(), "Reverb output[{i}] not finite: {s}");
        }
    }

    #[test]
    fn test_reverb_stereo_output_lengths() {
        let mut rev = SchroederReverb::new(48_000, ReverbPreset::SmallRoom, 0.4)
            .expect("small room stereo reverb creation should succeed");
        let l = sine_wave(256, 440.0, 48_000);
        let r = sine_wave(256, 660.0, 48_000);
        let (lo, ro) = rev.process_stereo(&l, &r);
        assert_eq!(lo.len(), 256);
        assert_eq!(ro.len(), 256);
    }

    #[test]
    fn test_reverb_large_hall_longer_tail_than_small_room() {
        let mut small = SchroederReverb::new(48_000, ReverbPreset::SmallRoom, 1.0)
            .expect("small room reverb for comparison should succeed");
        let mut large = SchroederReverb::new(48_000, ReverbPreset::LargeHall, 1.0)
            .expect("large hall reverb for comparison should succeed");

        let mut imp_s = vec![0.0_f32; 4096];
        let mut imp_l = vec![0.0_f32; 4096];
        imp_s[0] = 1.0;
        imp_l[0] = 1.0;

        let out_s = small.process_mono_wet(&imp_s);
        let out_l = large.process_mono_wet(&imp_l);

        // Energy in the late tail (after 1000 samples) should be higher for the large hall.
        let tail_s: f32 = out_s[1000..].iter().map(|x| x * x).sum();
        let tail_l: f32 = out_l[1000..].iter().map(|x| x * x).sum();
        assert!(
            tail_l > tail_s,
            "Large hall should have more tail energy: large={tail_l}, small={tail_s}"
        );
    }

    #[test]
    fn test_reverb_process_mono_inplace() {
        let mut rev = SchroederReverb::new(48_000, ReverbPreset::MediumRoom, 0.5)
            .expect("medium room inplace reverb creation should succeed");
        let mut buf: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let original = buf.clone();
        rev.process_mono(&mut buf);
        // The buffer should be modified (not all values identical to original).
        let any_changed = buf
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_changed, "In-place process should modify the buffer");
    }

    #[test]
    fn test_reverb_preset_cathedral_rt60() {
        let p = ReverbPreset::Cathedral;
        assert!((p.rt60_secs() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_reverb_custom_preset() {
        let p = ReverbPreset::Custom {
            rt60_secs: 1.2,
            damping: 0.4,
        };
        assert!((p.rt60_secs() - 1.2).abs() < 0.01);
        assert!((p.damping() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_reverb_apply_preset_changes_behaviour() {
        let mut rev = SchroederReverb::new(48_000, ReverbPreset::SmallRoom, 1.0)
            .expect("small room reverb for preset change test should succeed");
        // The longest comb delay at 48 kHz is ~1617 samples, so the SECOND reflection
        // (which carries the feedback coefficient) appears at 2*1617 = 3234 samples.
        // Use a 4096-sample buffer to capture the second-reflection region where the
        // SmallRoom (feedback ~0.585) and Cathedral (feedback ~0.948) differ clearly.
        let mut imp: Vec<f32> = vec![0.0_f32; 4096];
        imp[0] = 1.0;
        let out_small = rev.process_mono_wet(&imp);

        rev.reset();
        rev.apply_preset(ReverbPreset::Cathedral);
        let mut imp2: Vec<f32> = vec![0.0_f32; 4096];
        imp2[0] = 1.0;
        let out_large = rev.process_mono_wet(&imp2);

        // After the second comb reflection (≥ 2*1116 = 2232 samples) the feedback
        // coefficients are apparent: Cathedral retains far more energy than SmallRoom.
        let tail_small: f32 = out_small[2400..].iter().map(|x| x * x).sum();
        let tail_large: f32 = out_large[2400..].iter().map(|x| x * x).sum();
        assert!(
            tail_large > tail_small,
            "Cathedral (higher feedback) should have more late tail energy than SmallRoom: small={tail_small}, cathedral={tail_large}"
        );
    }

    // ── EarlyReflections ─────────────────────────────────────────────────────

    #[test]
    fn test_early_reflections_output_nonzero_after_delay() {
        let taps = vec![
            ReflectionTap {
                delay_samples: 5,
                gain: 0.6,
            },
            ReflectionTap {
                delay_samples: 10,
                gain: 0.4,
            },
        ];
        let mut er = EarlyReflections::new(taps);

        let input: Vec<f32> = (0..20).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let output = er.process_block(&input);

        // Tap at 5 samples: output[5] should be ~0.6
        assert!(
            output[5].abs() > 0.3,
            "Output should reflect tap at delay=5, got {}",
            output[5]
        );
    }

    #[test]
    fn test_early_reflections_from_room() {
        let er = EarlyReflections::from_room([8.0, 6.0, 3.0], [0.5, 0.5, 0.5], 48_000);
        assert!(!er.taps.is_empty(), "Room model should produce taps");
        assert_eq!(er.taps.len(), 6, "Should have 6 first-order reflections");
    }

    #[test]
    fn test_early_reflections_reset() {
        let taps = vec![ReflectionTap {
            delay_samples: 4,
            gain: 0.5,
        }];
        let mut er = EarlyReflections::new(taps);
        for _ in 0..10 {
            er.process(1.0);
        }
        er.reset();
        let out = er.process(0.0);
        assert_eq!(out, 0.0, "Reset early reflections should be silent");
    }

    #[test]
    fn test_comb_feedback_formula() {
        // At RT60 = 1.0 s, delay = 48000 samples (1 s) → feedback should be 10^(-3) ≈ 0.001
        let fb = super::comb_feedback(1.0, 48_000, 48_000);
        assert!(
            (fb - 0.001).abs() < 0.0005,
            "Feedback at RT60=1s, delay=1s should be ~0.001, got {fb}"
        );
    }
}

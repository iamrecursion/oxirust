//! High-quality sample rate conversion.
//!
//! This module provides three interpolation quality levels for resampling:
//!
//! - **Fast** — linear interpolation between adjacent samples
//! - **Medium** — 4-point cubic Hermite spline
//! - **High** — 64-tap windowed sinc with Kaiser window (β = 8.0)
//!
//! The converter maintains fractional phase state across calls so that it
//! can be fed arbitrary-sized blocks in a streaming fashion.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::sample_rate_converter::{SampleRateConverter, ResamplingQuality};
//!
//! let mut conv = SampleRateConverter::new(44100, 48000, 1, ResamplingQuality::Medium);
//! let input: Vec<f32> = (0..4410).map(|i| (i as f32 * 0.01).sin()).collect();
//! let output = conv.convert(&input);
//! // output contains approximately 4800 samples (44100 → 48000 with 1 channel)
//! ```

#![forbid(unsafe_code)]

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Resampling quality level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResamplingQuality {
    /// Linear interpolation — fastest but lowest quality.
    Fast,
    /// 4-point cubic Hermite spline — good balance of speed and quality.
    Medium,
    /// 64-tap windowed sinc with Kaiser window (β = 8.0) — highest quality.
    High,
}

/// Stateful, multi-channel sample rate converter.
///
/// The converter operates on **interleaved** samples: for a stereo stream the
/// layout is `[L₀, R₀, L₁, R₁, …]`.  All channels are processed together so
/// that the fractional phase stays consistent.
pub struct SampleRateConverter {
    /// Input sample rate in Hz.
    pub src_rate: u32,
    /// Output sample rate in Hz.
    pub dst_rate: u32,
    /// Resampling quality.
    pub quality: ResamplingQuality,
    /// Number of interleaved channels.
    pub channels: usize,
    /// Fractional position within the current input frame (0.0..1.0).
    phase: f64,
    /// Delay-line history buffer for sinc/cubic interpolation.
    /// Stored as interleaved samples, length = HISTORY_FRAMES * channels.
    history: Vec<f32>,
}

// Number of historical frames kept for interpolation.
// For High quality (64 taps): we need 32 frames on each side of the current
// position, so 64 total, but we only need 64 in the look-behind.
const HIGH_HISTORY: usize = 64;
const MEDIUM_HISTORY: usize = 4; // p-1, p0, p1, p2
const FAST_HISTORY: usize = 1; // x[n-1]

// Kaiser window half-taps for High quality.
const SINC_TAPS: usize = 64;

impl SampleRateConverter {
    /// Create a new converter.
    ///
    /// # Arguments
    ///
    /// * `src` — Input sample rate in Hz.
    /// * `dst` — Output sample rate in Hz.
    /// * `channels` — Number of audio channels (must be ≥ 1).
    /// * `quality` — Resampling quality level.
    #[must_use]
    pub fn new(src: u32, dst: u32, channels: usize, quality: ResamplingQuality) -> Self {
        let channels = channels.max(1);
        let history_frames = match quality {
            ResamplingQuality::Fast => FAST_HISTORY,
            ResamplingQuality::Medium => MEDIUM_HISTORY,
            ResamplingQuality::High => HIGH_HISTORY,
        };
        let history = vec![0.0f32; history_frames * channels];
        Self {
            src_rate: src,
            dst_rate: dst,
            quality,
            channels,
            phase: 0.0,
            history,
        }
    }

    /// Resampling ratio: dst_rate / src_rate.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        self.dst_rate as f64 / self.src_rate as f64
    }

    /// Resample a block of interleaved input samples.
    ///
    /// The output length will approximate `input.len() * ratio()` but may
    /// differ by ±`channels` due to fractional phase.
    #[must_use]
    pub fn convert(&mut self, input: &[f32]) -> Vec<f32> {
        if self.src_rate == self.dst_rate {
            return input.to_vec();
        }
        if input.is_empty() {
            return Vec::new();
        }

        match self.quality {
            ResamplingQuality::Fast => self.convert_linear(input),
            ResamplingQuality::Medium => self.convert_hermite(input),
            ResamplingQuality::High => self.convert_sinc(input),
        }
    }

    // -----------------------------------------------------------------------
    // Fast — linear interpolation
    // -----------------------------------------------------------------------

    fn convert_linear(&mut self, input: &[f32]) -> Vec<f32> {
        let ch = self.channels;
        // input frames (not samples)
        let input_frames = input.len() / ch;
        if input_frames == 0 {
            return Vec::new();
        }

        let ratio = self.ratio();
        let step = 1.0 / ratio; // how much input phase advances per output sample

        // Estimate output capacity
        let estimated = ((input_frames as f64 / step) + 2.0) as usize;
        let mut output = Vec::with_capacity(estimated * ch);

        // Access function: frame index into input (or history for frame < 0).
        // history has FAST_HISTORY (=1) frame.
        let history = &self.history;
        let get_frame = |frame_idx: i64, input: &[f32]| -> Vec<f32> {
            if frame_idx < 0 {
                let h_idx = (FAST_HISTORY as i64 + frame_idx) as usize;
                if h_idx < FAST_HISTORY {
                    history[h_idx * ch..(h_idx + 1) * ch].to_vec()
                } else {
                    vec![0.0; ch]
                }
            } else {
                let fi = frame_idx as usize;
                if fi < input_frames {
                    input[fi * ch..(fi + 1) * ch].to_vec()
                } else {
                    vec![0.0; ch]
                }
            }
        };

        while self.phase < input_frames as f64 {
            let frame0 = self.phase.floor() as i64;
            let frac = (self.phase - self.phase.floor()) as f32;

            let s0 = get_frame(frame0, input);
            let s1 = get_frame(frame0 + 1, input);

            for c in 0..ch {
                output.push(s0[c] + frac * (s1[c] - s0[c]));
            }

            self.phase += step;
        }

        // Update phase for next block
        self.phase -= input_frames as f64;

        // Update history: keep last FAST_HISTORY frames of input
        let hist_start = input_frames.saturating_sub(FAST_HISTORY);
        let copy_frames = input_frames - hist_start;
        let mut new_history = vec![0.0f32; FAST_HISTORY * ch];
        let dest_offset = FAST_HISTORY.saturating_sub(copy_frames);
        for f in 0..copy_frames {
            let src_f = hist_start + f;
            for c in 0..ch {
                new_history[(dest_offset + f) * ch + c] = input[src_f * ch + c];
            }
        }
        self.history = new_history;

        output
    }

    // -----------------------------------------------------------------------
    // Medium — 4-point cubic Hermite spline
    // -----------------------------------------------------------------------

    fn convert_hermite(&mut self, input: &[f32]) -> Vec<f32> {
        let ch = self.channels;
        let input_frames = input.len() / ch;
        if input_frames == 0 {
            return Vec::new();
        }

        let ratio = self.ratio();
        let step = 1.0 / ratio;

        let estimated = ((input_frames as f64 / step) + 2.0) as usize;
        let mut output = Vec::with_capacity(estimated * ch);

        let history = &self.history;
        let get_frame = |frame_idx: i64, input: &[f32]| -> Vec<f32> {
            if frame_idx < 0 {
                let h_idx = (MEDIUM_HISTORY as i64 + frame_idx) as usize;
                if h_idx < MEDIUM_HISTORY {
                    history[h_idx * ch..(h_idx + 1) * ch].to_vec()
                } else {
                    vec![0.0; ch]
                }
            } else {
                let fi = frame_idx as usize;
                if fi < input_frames {
                    input[fi * ch..(fi + 1) * ch].to_vec()
                } else if fi == input_frames {
                    // Allow one frame past end (treated as zero)
                    vec![0.0; ch]
                } else {
                    vec![0.0; ch]
                }
            }
        };

        while self.phase < input_frames as f64 {
            let frame0 = self.phase.floor() as i64;
            let frac = (self.phase - self.phase.floor()) as f32;

            let xm1 = get_frame(frame0 - 1, input);
            let x0 = get_frame(frame0, input);
            let x1 = get_frame(frame0 + 1, input);
            let x2 = get_frame(frame0 + 2, input);

            for c in 0..ch {
                output.push(hermite_interp(xm1[c], x0[c], x1[c], x2[c], frac));
            }

            self.phase += step;
        }

        self.phase -= input_frames as f64;

        // Update history: keep last MEDIUM_HISTORY frames
        let hist_start = input_frames.saturating_sub(MEDIUM_HISTORY);
        let copy_frames = input_frames - hist_start;
        let mut new_history = vec![0.0f32; MEDIUM_HISTORY * ch];
        let dest_offset = MEDIUM_HISTORY.saturating_sub(copy_frames);
        for f in 0..copy_frames {
            let src_f = hist_start + f;
            for c in 0..ch {
                new_history[(dest_offset + f) * ch + c] = input[src_f * ch + c];
            }
        }
        self.history = new_history;

        output
    }

    // -----------------------------------------------------------------------
    // High — 64-tap windowed sinc with Kaiser window β=8
    // -----------------------------------------------------------------------

    fn convert_sinc(&mut self, input: &[f32]) -> Vec<f32> {
        let ch = self.channels;
        let input_frames = input.len() / ch;
        if input_frames == 0 {
            return Vec::new();
        }

        let ratio = self.ratio();
        let step = 1.0 / ratio;

        // Anti-alias cutoff: min(src, dst) / max(src, dst) → 0.5 if downsampling
        let cutoff = if self.dst_rate < self.src_rate {
            ratio // normalized to Nyquist = 0.5 * src
        } else {
            1.0
        };

        // Precompute Kaiser window table once per call (cheap: 64 taps)
        let kaiser_win = kaiser_window(SINC_TAPS, 8.0);

        let estimated = ((input_frames as f64 / step) + 2.0) as usize;
        let mut output = Vec::with_capacity(estimated * ch);

        // Extended view: history + input
        // history length = HIGH_HISTORY frames
        let history = &self.history;

        // Helper closure to get sample by frame index (may be negative = history)
        let total_history = HIGH_HISTORY;
        let get_sample = |frame_idx: i64, ch_idx: usize| -> f32 {
            if frame_idx < 0 {
                let h_frame = (total_history as i64 + frame_idx) as usize;
                if h_frame < total_history {
                    history[h_frame * ch + ch_idx]
                } else {
                    0.0
                }
            } else {
                let fi = frame_idx as usize;
                if fi < input_frames {
                    input[fi * ch + ch_idx]
                } else {
                    0.0
                }
            }
        };

        let half_taps = (SINC_TAPS / 2) as i64;

        while self.phase < input_frames as f64 {
            let center = self.phase.floor() as i64;
            let frac = self.phase - self.phase.floor();

            for c in 0..ch {
                let mut acc = 0.0f64;
                for k in -half_taps..half_taps {
                    let tap_pos = center + k; // input frame index
                    let x_k = get_sample(tap_pos, c) as f64;

                    // Sinc argument: distance from current output position
                    let t = k as f64 - frac;
                    let sinc_val = sinc(t * cutoff);
                    let win_idx = (k + half_taps) as usize;
                    let w = kaiser_win[win_idx.min(SINC_TAPS - 1)];

                    acc += x_k * sinc_val * w as f64;
                }
                // Scale by cutoff to maintain gain during downsampling
                output.push((acc * cutoff) as f32);
            }

            self.phase += step;
        }

        self.phase -= input_frames as f64;

        // Update history: keep last HIGH_HISTORY frames
        let hist_start = input_frames.saturating_sub(HIGH_HISTORY);
        let copy_frames = input_frames - hist_start;
        let mut new_history = vec![0.0f32; HIGH_HISTORY * ch];
        let dest_offset = HIGH_HISTORY.saturating_sub(copy_frames);
        for f in 0..copy_frames {
            let src_f = hist_start + f;
            for c in 0..ch {
                new_history[(dest_offset + f) * ch + c] = input[src_f * ch + c];
            }
        }
        self.history = new_history;

        output
    }

    // -----------------------------------------------------------------------
    // Convenience constructors
    // -----------------------------------------------------------------------

    /// Resample `input` from `src_rate` to 48 000 Hz using `High` quality.
    ///
    /// This is a one-shot convenience function (no persistent state).
    #[must_use]
    pub fn to_48k(input: &[f32], src_rate: u32, channels: usize) -> Vec<f32> {
        let mut conv = Self::new(src_rate, 48_000, channels, ResamplingQuality::High);
        conv.convert(input)
    }

    /// Resample `input` from `src_rate` to 44 100 Hz using `High` quality.
    ///
    /// This is a one-shot convenience function (no persistent state).
    #[must_use]
    pub fn to_44100(input: &[f32], src_rate: u32, channels: usize) -> Vec<f32> {
        let mut conv = Self::new(src_rate, 44_100, channels, ResamplingQuality::High);
        conv.convert(input)
    }
}

// ---------------------------------------------------------------------------
// DSP helpers
// ---------------------------------------------------------------------------

/// Normalised sinc function: sinc(0) = 1, sinc(x) = sin(π x) / (π x).
#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// 4-point cubic Hermite spline interpolation.
///
/// Points: `xm1` (x[-1]), `x0` (x[0]), `x1` (x[1]), `x2` (x[2]).
/// `frac` is the fractional position in [0, 1) between `x0` and `x1`.
#[inline]
fn hermite_interp(xm1: f32, x0: f32, x1: f32, x2: f32, frac: f32) -> f32 {
    let c = (x1 - xm1) * 0.5;
    let v = x0 - x1;
    let w = c + v;
    let a = w + v + (x2 - x0) * 0.5;
    let b = w + a;
    (((a * frac) - b) * frac + c) * frac + x0
}

/// Compute a Kaiser window of length `n` with shape parameter `beta`.
///
/// Uses the modified Bessel function of the first kind I0.
/// `w[k] = I0(β √(1 − (2k/(n−1) − 1)²)) / I0(β)` for k = 0..n.
#[must_use]
fn kaiser_window(n: usize, beta: f64) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let n_f = (n - 1) as f64;
    let i0_beta = bessel_i0(beta);
    (0..n)
        .map(|k| {
            let x = 2.0 * k as f64 / n_f - 1.0;
            let arg = beta * (1.0 - x * x).max(0.0).sqrt();
            (bessel_i0(arg) / i0_beta) as f32
        })
        .collect()
}

/// Modified Bessel function of the first kind, order zero, I0(x).
///
/// Computed via its power series: I0(x) = Σ_{k=0}^∞ (x/2)^{2k} / (k!)^2.
/// Converges well for |x| ≤ 20 (β = 8 → x ≤ 8).
#[must_use]
pub fn bessel_i0(x: f64) -> f64 {
    let x2 = (x / 2.0) * (x / 2.0);
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for k in 1u32..=50 {
        term *= x2 / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-15 {
            break;
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq_hz: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        let sr = sample_rate as f32;
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sr).sin())
            .collect()
    }

    // --- bessel_i0 ---

    #[test]
    fn test_bessel_i0_zero() {
        // I0(0) = 1
        let val = bessel_i0(0.0);
        assert!((val - 1.0).abs() < 1e-10, "I0(0) should be 1, got {val}");
    }

    #[test]
    fn test_bessel_i0_known_value() {
        // I0(1) ≈ 1.2660658
        let val = bessel_i0(1.0);
        assert!(
            (val - 1.266_065_8).abs() < 1e-5,
            "I0(1) should be ~1.2661, got {val}"
        );
    }

    // --- ratio ---

    #[test]
    fn test_ratio_44100_to_48000() {
        let conv = SampleRateConverter::new(44100, 48000, 1, ResamplingQuality::Fast);
        let r = conv.ratio();
        assert!((r - 48000.0 / 44100.0).abs() < 1e-10);
    }

    // --- passthrough (same rate) ---

    #[test]
    fn test_passthrough_identity() {
        let mut conv = SampleRateConverter::new(44100, 44100, 1, ResamplingQuality::Fast);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output = conv.convert(&input);
        assert_eq!(output, input, "Same-rate should be passthrough");
    }

    // --- Fast: output length ---

    #[test]
    fn test_fast_output_length_upsample() {
        let input = sine_wave(440.0, 44100, 4410);
        let mut conv = SampleRateConverter::new(44100, 48000, 1, ResamplingQuality::Fast);
        let output = conv.convert(&input);
        // Expected ≈ 4800 samples
        let expected = (4410.0 * 48000.0 / 44100.0) as usize;
        assert!(
            output.len().abs_diff(expected) <= 2,
            "output len {} should be ≈ {}",
            output.len(),
            expected
        );
    }

    #[test]
    fn test_fast_output_length_downsample() {
        let input = sine_wave(440.0, 48000, 4800);
        let mut conv = SampleRateConverter::new(48000, 44100, 1, ResamplingQuality::Fast);
        let output = conv.convert(&input);
        let expected = (4800.0 * 44100.0 / 48000.0) as usize;
        assert!(
            output.len().abs_diff(expected) <= 2,
            "output len {} should be ≈ {}",
            output.len(),
            expected
        );
    }

    // --- Medium: output length ---

    #[test]
    fn test_medium_output_length() {
        let input = sine_wave(440.0, 44100, 4410);
        let mut conv = SampleRateConverter::new(44100, 48000, 1, ResamplingQuality::Medium);
        let output = conv.convert(&input);
        let expected = (4410.0 * 48000.0 / 44100.0) as usize;
        assert!(
            output.len().abs_diff(expected) <= 2,
            "output len {} should be ≈ {}",
            output.len(),
            expected
        );
    }

    // --- High: output length ---

    #[test]
    fn test_high_output_length() {
        let input = sine_wave(440.0, 44100, 4410);
        let mut conv = SampleRateConverter::new(44100, 48000, 1, ResamplingQuality::High);
        let output = conv.convert(&input);
        let expected = (4410.0 * 48000.0 / 44100.0) as usize;
        assert!(
            output.len().abs_diff(expected) <= 2,
            "output len {} should be ≈ {}",
            output.len(),
            expected
        );
    }

    // --- output is finite ---

    #[test]
    fn test_fast_output_all_finite() {
        let input = sine_wave(440.0, 44100, 2048);
        let mut conv = SampleRateConverter::new(44100, 48000, 1, ResamplingQuality::Fast);
        let output = conv.convert(&input);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "Fast output must be all finite"
        );
    }

    #[test]
    fn test_medium_output_all_finite() {
        let input = sine_wave(440.0, 44100, 2048);
        let mut conv = SampleRateConverter::new(44100, 48000, 1, ResamplingQuality::Medium);
        let output = conv.convert(&input);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "Medium output must be all finite"
        );
    }

    #[test]
    fn test_high_output_all_finite() {
        let input = sine_wave(440.0, 44100, 2048);
        let mut conv = SampleRateConverter::new(44100, 48000, 1, ResamplingQuality::High);
        let output = conv.convert(&input);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "High output must be all finite"
        );
    }

    // --- to_48k / to_44100 ---

    #[test]
    fn test_to_48k_length() {
        let input = sine_wave(440.0, 44100, 4410);
        let output = SampleRateConverter::to_48k(&input, 44100, 1);
        let expected = (4410.0 * 48000.0 / 44100.0) as usize;
        assert!(
            output.len().abs_diff(expected) <= 2,
            "to_48k len {} should be ≈ {}",
            output.len(),
            expected
        );
    }

    #[test]
    fn test_to_44100_length() {
        let input = sine_wave(440.0, 48000, 4800);
        let output = SampleRateConverter::to_44100(&input, 48000, 1);
        let expected = (4800.0 * 44100.0 / 48000.0) as usize;
        assert!(
            output.len().abs_diff(expected) <= 2,
            "to_44100 len {} should be ≈ {}",
            output.len(),
            expected
        );
    }
}

//! Gated loudness measurement (EBU R128 / ITU-R BS.1770).
//!
//! This module provides a self-contained gated loudness meter that
//! implements the core ITU-R BS.1770-4 measurement algorithm:
//!
//! 1. K-weighting filter (two biquad stages).
//! 2. Mean-square per channel, weighted sum.
//! 3. 400 ms blocks (overlapping at 100 ms hops).
//! 4. Two-stage gating:
//!    - Absolute gate: exclude blocks < −70 LUFS.
//!    - Relative gate: exclude blocks more than 10 LU below the ungated mean.
//!
//! Additionally this module computes:
//! - Short-term loudness (last 3-second window).
//! - Momentary loudness (last 400 ms block).
//! - Loudness Range (LRA): 10th–95th percentile of gated short-term blocks.
//! - True-peak approximation (maximum absolute sample value).
//!
//! # Example
//!
//! ```
//! use oximedia_audio::loudness_gating::GatedLoudnessMeter;
//!
//! let mut meter = GatedLoudnessMeter::new(48000, 2);
//! // feed 400 ms blocks repeatedly…
//! meter.process_block(&vec![0.1_f32; 48000 * 2 / 2]);
//! let integrated = meter.integrated_loudness();
//! ```

#![forbid(unsafe_code)]

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of a full loudness measurement pass.
#[derive(Clone, Debug)]
pub struct LoudnessMeasurement {
    /// Integrated (gated) loudness in LUFS.
    pub integrated_lufs: f64,
    /// Short-term loudness (last 3 s window) in LUFS.
    pub short_term_lufs: f64,
    /// Momentary loudness (last 400 ms block) in LUFS.
    pub momentary_lufs: f64,
    /// Loudness Range (LRA) in LU.
    pub loudness_range_lu: f64,
    /// True-peak estimate in dBTP.
    pub true_peak_dbtp: f64,
}

/// Gated loudness meter following EBU R128 / ITU-R BS.1770.
pub struct GatedLoudnessMeter {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: usize,
    /// Per-channel weights (EBU default: all 1.0 except surround LFE = 0.0, Ls/Rs = 1.41).
    pub channel_weights: Vec<f32>,
    /// Block size in samples (400 ms).
    block_size: usize,
    /// Hop size in samples (100 ms overlap).
    hop_size: usize,
    /// Accumulated K-weighted, mean-square block loudness values in LKFS.
    blocks: Vec<f64>,
    /// Per-channel K-weighting biquad filter state.
    filters: Vec<KWeightState>,
    /// Internal sample ring-buffer (interleaved) for partial block accumulation.
    ring: Vec<f32>,
    /// Write position inside `ring`.
    ring_pos: usize,
    /// True-peak tracker (linear).
    true_peak_linear: f64,
}

// ---------------------------------------------------------------------------
// Internal biquad state
// ---------------------------------------------------------------------------

/// Combined state for the two-stage K-weighting filter (per channel).
#[derive(Clone, Default)]
struct KWeightState {
    // Stage 1 (pre-filter / high-shelf) state
    x1_s1: f64,
    x2_s1: f64,
    y1_s1: f64,
    y2_s1: f64,
    // Stage 2 (high-pass Butterworth 38 Hz) state
    x1_s2: f64,
    x2_s2: f64,
    y1_s2: f64,
    y2_s2: f64,
}

/// Pre-computed biquad coefficients.
#[derive(Clone, Debug)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

// ---------------------------------------------------------------------------
// K-weighting filter coefficient computation
// ---------------------------------------------------------------------------

/// Compute ITU-R BS.1770-4 Stage 1 (pre-filter / high-shelf) coefficients.
///
/// This is a high-shelf filter at f0 = 1681.97 Hz with Q = 0.7072 and G = 4 dB.
fn k_stage1_coeffs(sample_rate: u32) -> BiquadCoeffs {
    // Parameters from ITU-R BS.1770-4 Table 1
    let f0 = 1_681.974_450_955_533_f64;
    let g_db = 3.999_843_853_973_347_f64;
    let q = 0.707_175_236_955_419_6_f64;

    let k = (PI * f0 / sample_rate as f64).tan();
    let k2 = k * k;
    let vh = 10.0_f64.powf(g_db / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6_f64);
    let norm = 1.0 / (1.0 + k / q + k2);

    BiquadCoeffs {
        b0: (vh + vb * k / q + k2) * norm,
        b1: 2.0 * (k2 - vh) * norm,
        b2: (vh - vb * k / q + k2) * norm,
        a1: 2.0 * (k2 - 1.0) * norm,
        a2: (1.0 - k / q + k2) * norm,
    }
}

/// Compute ITU-R BS.1770-4 Stage 2 (Butterworth high-pass at 38.13 Hz) coefficients.
fn k_stage2_coeffs(sample_rate: u32) -> BiquadCoeffs {
    let f0 = 38.135_470_876_024_44_f64;
    let q = 0.500_327_037_323_877_3_f64;

    let k = (PI * f0 / sample_rate as f64).tan();
    let k2 = k * k;
    let norm = 1.0 / (1.0 + k / q + k2);

    BiquadCoeffs {
        b0: norm,
        b1: -2.0 * norm,
        b2: norm,
        a1: 2.0 * (k2 - 1.0) * norm,
        a2: (1.0 - k / q + k2) * norm,
    }
}

/// Apply a biquad filter to a single sample using Direct Form I.
#[inline]
fn biquad_process(
    c: &BiquadCoeffs,
    x: f64,
    x1: &mut f64,
    x2: &mut f64,
    y1: &mut f64,
    y2: &mut f64,
) -> f64 {
    let y = c.b0 * x + c.b1 * *x1 + c.b2 * *x2 - c.a1 * *y1 - c.a2 * *y2;
    *x2 = *x1;
    *x1 = x;
    *y2 = *y1;
    *y1 = y;
    y
}

// ---------------------------------------------------------------------------
// GatedLoudnessMeter implementation
// ---------------------------------------------------------------------------

impl GatedLoudnessMeter {
    /// Create a new gated loudness meter.
    ///
    /// Default channel weights follow EBU R128: all 1.0 (stereo / mono).
    /// For 5.1 surround set weights to `[1.0, 1.0, 1.0, 0.0, 1.41, 1.41]`.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        let channels = channels.max(1);
        let block_size = sample_rate as usize * 4 / 10; // 400 ms
        let hop_size = sample_rate as usize / 10; // 100 ms

        let channel_weights = vec![1.0f32; channels];
        let filters = vec![KWeightState::default(); channels];
        let ring = vec![0.0f32; block_size * channels];

        Self {
            sample_rate,
            channels,
            channel_weights,
            block_size,
            hop_size,
            blocks: Vec::new(),
            filters,
            ring,
            ring_pos: 0,
            true_peak_linear: 0.0,
        }
    }

    /// Apply the K-weighting filter to a mono channel buffer in-place.
    ///
    /// Returns the K-weighted samples as a new `Vec<f32>`.
    #[must_use]
    pub fn k_weighted_filter(samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let c1 = k_stage1_coeffs(sample_rate);
        let c2 = k_stage2_coeffs(sample_rate);
        let mut x1_s1 = 0.0f64;
        let mut x2_s1 = 0.0f64;
        let mut y1_s1 = 0.0f64;
        let mut y2_s1 = 0.0f64;
        let mut x1_s2 = 0.0f64;
        let mut x2_s2 = 0.0f64;
        let mut y1_s2 = 0.0f64;
        let mut y2_s2 = 0.0f64;

        samples
            .iter()
            .map(|&s| {
                let s_f64 = s as f64;
                let s1 = biquad_process(&c1, s_f64, &mut x1_s1, &mut x2_s1, &mut y1_s1, &mut y2_s1);
                let s2 = biquad_process(&c2, s1, &mut x1_s2, &mut x2_s2, &mut y1_s2, &mut y2_s2);
                s2 as f32
            })
            .collect()
    }

    /// Process a block of interleaved audio samples.
    ///
    /// Internally accumulates samples in a ring buffer; whenever a full 400 ms
    /// block is ready the loudness is computed and stored. Overlapping at 100 ms
    /// hops is achieved by treating the ring buffer as a sliding window.
    pub fn process_block(&mut self, samples: &[f32]) {
        let ch = self.channels;

        // Precompute coefficients (cheap — just trig)
        let c1 = k_stage1_coeffs(self.sample_rate);
        let c2 = k_stage2_coeffs(self.sample_rate);

        let frame_count = samples.len() / ch;

        for frame in 0..frame_count {
            // Update true-peak
            for c in 0..ch {
                let s = samples[frame * ch + c].abs() as f64;
                if s > self.true_peak_linear {
                    self.true_peak_linear = s;
                }
            }

            // K-weight each channel and store into ring buffer
            for c in 0..ch {
                let raw = samples[frame * ch + c] as f64;
                let s = &mut self.filters[c];
                let w1 = biquad_process(
                    &c1,
                    raw,
                    &mut s.x1_s1,
                    &mut s.x2_s1,
                    &mut s.y1_s1,
                    &mut s.y2_s1,
                );
                let w2 = biquad_process(
                    &c2,
                    w1,
                    &mut s.x1_s2,
                    &mut s.x2_s2,
                    &mut s.y1_s2,
                    &mut s.y2_s2,
                );

                let ring_idx = self.ring_pos * ch + c;
                self.ring[ring_idx] = w2 as f32;
            }

            self.ring_pos += 1;

            // When we have accumulated a full block, compute loudness
            if self.ring_pos >= self.block_size {
                let block_loudness = self.compute_block_loudness();
                self.blocks.push(block_loudness);

                // Shift ring by hop_size (slide window forward)
                let keep = self.block_size - self.hop_size;
                let src_start = self.hop_size * ch;
                self.ring.copy_within(src_start.., 0);
                // Zero the newly vacated region
                let zero_start = keep * ch;
                for v in &mut self.ring[zero_start..] {
                    *v = 0.0;
                }
                self.ring_pos = keep;
            }
        }
    }

    /// Compute the mean-square weighted sum of the current ring buffer block.
    ///
    /// Returns block loudness in LKFS: `−0.691 + 10 log10(sum_c w_c * ms_c)`.
    fn compute_block_loudness(&self) -> f64 {
        let ch = self.channels;
        let mut weighted_ms = 0.0f64;

        for c in 0..ch {
            let weight = self.channel_weights[c] as f64;
            if weight == 0.0 {
                continue;
            }
            let ms: f64 = (0..self.block_size)
                .map(|f| {
                    let s = self.ring[f * ch + c] as f64;
                    s * s
                })
                .sum::<f64>()
                / self.block_size as f64;
            weighted_ms += weight * ms;
        }

        if weighted_ms < 1e-15 {
            return f64::NEG_INFINITY;
        }
        -0.691 + 10.0 * weighted_ms.log10()
    }

    /// Compute the gated integrated loudness in LUFS.
    ///
    /// Two-stage gating per EBU R128 / ITU-R BS.1770-4:
    /// 1. Absolute gate: discard blocks < −70 LUFS.
    /// 2. Relative gate: from passing blocks compute mean J; discard blocks
    ///    more than 10 LU below J; return mean of remaining.
    #[must_use]
    pub fn integrated_loudness(&self) -> f64 {
        if self.blocks.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Stage 1: absolute gate
        let abs_gate = -70.0_f64;
        let stage1: Vec<f64> = self
            .blocks
            .iter()
            .copied()
            .filter(|&b| b > abs_gate)
            .collect();

        if stage1.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Convert to linear power, compute mean, back to dB
        let mean_power_1: f64 =
            stage1.iter().map(|&l| 10.0_f64.powf(l / 10.0)).sum::<f64>() / stage1.len() as f64;
        let j = -0.691 + 10.0 * mean_power_1.log10();

        // Stage 2: relative gate at j - 10
        let rel_gate = j - 10.0;
        let stage2: Vec<f64> = stage1.iter().copied().filter(|&b| b > rel_gate).collect();

        if stage2.is_empty() {
            return f64::NEG_INFINITY;
        }

        let mean_power_2: f64 =
            stage2.iter().map(|&l| 10.0_f64.powf(l / 10.0)).sum::<f64>() / stage2.len() as f64;

        -0.691 + 10.0 * mean_power_2.log10()
    }

    /// Compute the Loudness Range (LRA) in LU.
    ///
    /// Applies the absolute gate (−70 LUFS) and the relative gate (−20 LU),
    /// then returns the difference between the 95th and 10th percentile of
    /// the remaining short-term loudness distribution.
    #[must_use]
    pub fn loudness_range(&self) -> f64 {
        if self.blocks.len() < 2 {
            return 0.0;
        }

        let abs_gate = -70.0_f64;
        let mut gated: Vec<f64> = self
            .blocks
            .iter()
            .copied()
            .filter(|&b| b > abs_gate)
            .collect();

        if gated.is_empty() {
            return 0.0;
        }

        // Relative gate for LRA: mean - 20 LU
        let mean_power: f64 =
            gated.iter().map(|&l| 10.0_f64.powf(l / 10.0)).sum::<f64>() / gated.len() as f64;
        let mean_lufs = -0.691 + 10.0 * mean_power.log10();
        let rel_gate = mean_lufs - 20.0;

        gated.retain(|&b| b > rel_gate);

        if gated.len() < 2 {
            return 0.0;
        }

        gated.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = gated.len();
        let p10_idx = ((n as f64 - 1.0) * 0.10).round() as usize;
        let p95_idx = ((n as f64 - 1.0) * 0.95).round() as usize;

        gated[p95_idx] - gated[p10_idx]
    }

    /// Get the momentary loudness (the most recent 400 ms block).
    #[must_use]
    pub fn momentary_loudness(&self) -> f64 {
        self.blocks.last().copied().unwrap_or(f64::NEG_INFINITY)
    }

    /// Get the short-term loudness (average of the last 30 blocks ≈ 3 s).
    #[must_use]
    pub fn short_term_loudness(&self) -> f64 {
        // Each block hop is 100 ms, so 3 s = 30 blocks
        const SHORT_TERM_BLOCKS: usize = 30;
        let start = self.blocks.len().saturating_sub(SHORT_TERM_BLOCKS);
        let window: Vec<f64> = self.blocks[start..].iter().copied().collect();

        if window.is_empty() {
            return f64::NEG_INFINITY;
        }

        let finite: Vec<f64> = window.iter().copied().filter(|x| x.is_finite()).collect();
        if finite.is_empty() {
            return f64::NEG_INFINITY;
        }

        let mean_power: f64 =
            finite.iter().map(|&l| 10.0_f64.powf(l / 10.0)).sum::<f64>() / finite.len() as f64;

        -0.691 + 10.0 * mean_power.log10()
    }

    /// True-peak in dBTP (decibels true-peak).
    ///
    /// Returns a sample-domain approximation (maximum absolute sample).
    /// For inter-sample peaks an oversampling true-peak detector is needed;
    /// this provides an indication only.
    #[must_use]
    pub fn true_peak_dbtp(&self) -> f64 {
        if self.true_peak_linear < 1e-15 {
            return f64::NEG_INFINITY;
        }
        20.0 * self.true_peak_linear.log10()
    }

    /// Full single-pass measurement convenience method.
    ///
    /// Processes `samples` (interleaved) in one shot and returns the
    /// complete `LoudnessMeasurement`.
    #[must_use]
    pub fn measure(samples: &[f32], sample_rate: u32, channels: usize) -> LoudnessMeasurement {
        let mut meter = Self::new(sample_rate, channels);
        meter.process_block(samples);
        LoudnessMeasurement {
            integrated_lufs: meter.integrated_loudness(),
            short_term_lufs: meter.short_term_loudness(),
            momentary_lufs: meter.momentary_loudness(),
            loudness_range_lu: meter.loudness_range(),
            true_peak_dbtp: meter.true_peak_dbtp(),
        }
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.filters = vec![KWeightState::default(); self.channels];
        let ring_len = self.ring.len();
        self.ring = vec![0.0f32; ring_len];
        self.ring_pos = 0;
        self.true_peak_linear = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate a mono sine wave at the given amplitude and frequency.
    fn sine_wave(freq_hz: f32, amplitude: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        let sr = sample_rate as f32;
        (0..num_samples)
            .map(|i| amplitude * (2.0 * PI * freq_hz * i as f32 / sr).sin())
            .collect()
    }

    /// Generate silence (all zeros).
    fn silence(num_samples: usize) -> Vec<f32> {
        vec![0.0f32; num_samples]
    }

    // --- basic construction ---

    #[test]
    fn test_new_default_weights() {
        let meter = GatedLoudnessMeter::new(48000, 2);
        assert_eq!(meter.channel_weights.len(), 2);
        assert_eq!(meter.channel_weights[0], 1.0);
        assert_eq!(meter.channel_weights[1], 1.0);
    }

    #[test]
    fn test_new_block_sizes() {
        let meter = GatedLoudnessMeter::new(48000, 1);
        // 400 ms at 48 kHz = 19200 samples
        assert_eq!(meter.block_size, 19200);
        // 100 ms at 48 kHz = 4800 samples
        assert_eq!(meter.hop_size, 4800);
    }

    // --- k_weighted_filter ---

    #[test]
    fn test_k_weighted_filter_length() {
        let samples = vec![0.1f32; 1024];
        let output = GatedLoudnessMeter::k_weighted_filter(&samples, 48000);
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_k_weighted_filter_silence() {
        let samples = silence(1024);
        let output = GatedLoudnessMeter::k_weighted_filter(&samples, 48000);
        assert!(
            output.iter().all(|x| *x == 0.0),
            "K-weighted silence should remain silent"
        );
    }

    #[test]
    fn test_k_weighted_filter_finite() {
        let samples = sine_wave(1000.0, 0.5, 48000, 4800);
        let output = GatedLoudnessMeter::k_weighted_filter(&samples, 48000);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "K-weighted output must be finite"
        );
    }

    // --- integrated_loudness ---

    #[test]
    fn test_silence_integrated_loudness() {
        let samples = silence(48000 * 5); // 5 seconds
        let m = GatedLoudnessMeter::measure(&samples, 48000, 1);
        assert!(
            m.integrated_lufs.is_infinite() || m.integrated_lufs < -70.0,
            "Silence should yield < -70 LUFS or -inf"
        );
    }

    #[test]
    fn test_loud_signal_integrated_loudness() {
        // A full-scale sine wave should give a loudness that is not -inf
        let samples = sine_wave(1000.0, 0.9, 48000, 48000 * 5);
        let mut meter = GatedLoudnessMeter::new(48000, 1);
        meter.process_block(&samples);
        let il = meter.integrated_loudness();
        assert!(
            il.is_finite() && il > -50.0,
            "Loud signal should have a finite integrated loudness, got {il}"
        );
    }

    // --- momentary / short-term ---

    #[test]
    fn test_momentary_loudness_before_block() {
        let meter = GatedLoudnessMeter::new(48000, 1);
        let m = meter.momentary_loudness();
        assert!(m.is_infinite(), "No blocks yet → -inf");
    }

    #[test]
    fn test_short_term_loudness_before_block() {
        let meter = GatedLoudnessMeter::new(48000, 1);
        let st = meter.short_term_loudness();
        assert!(st.is_infinite(), "No blocks yet → -inf");
    }

    // --- true-peak ---

    #[test]
    fn test_true_peak_tracks_maximum() {
        let samples = vec![0.5f32, 0.8f32, -0.9f32, 0.3f32];
        let m = GatedLoudnessMeter::measure(&samples, 48000, 1);
        // true_peak_linear should be 0.9 → 20*log10(0.9) ≈ -0.915 dBTP
        let expected_dbtp = 20.0 * 0.9_f64.log10();
        assert!(
            (m.true_peak_dbtp - expected_dbtp).abs() < 0.01,
            "true_peak_dbtp mismatch: expected {expected_dbtp:.3}, got {:.3}",
            m.true_peak_dbtp
        );
    }

    #[test]
    fn test_true_peak_silence() {
        let samples = silence(100);
        let m = GatedLoudnessMeter::measure(&samples, 48000, 1);
        assert!(
            m.true_peak_dbtp.is_infinite(),
            "Silence should have -inf true-peak"
        );
    }

    // --- reset ---

    #[test]
    fn test_reset_clears_blocks() {
        let mut meter = GatedLoudnessMeter::new(48000, 1);
        let samples = sine_wave(1000.0, 0.5, 48000, 48000 * 2);
        meter.process_block(&samples);
        meter.reset();
        assert!(
            meter.integrated_loudness().is_infinite(),
            "After reset, integrated loudness should be -inf"
        );
    }

    // --- loudness_range ---

    #[test]
    fn test_loudness_range_empty() {
        let meter = GatedLoudnessMeter::new(48000, 1);
        let lra = meter.loudness_range();
        assert_eq!(lra, 0.0, "LRA of empty meter should be 0");
    }

    #[test]
    fn test_measure_returns_struct() {
        let samples = sine_wave(440.0, 0.5, 48000, 48000 * 3);
        let m = GatedLoudnessMeter::measure(&samples, 48000, 1);
        // Just verify it doesn't panic and returns plausible values
        assert!(m.integrated_lufs.is_finite() || m.integrated_lufs == f64::NEG_INFINITY);
        assert!(m.true_peak_dbtp.is_finite() || m.true_peak_dbtp == f64::NEG_INFINITY);
        assert!(m.loudness_range_lu >= 0.0);
    }
}

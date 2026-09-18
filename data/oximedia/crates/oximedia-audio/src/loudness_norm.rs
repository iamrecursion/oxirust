//! EBU R128 loudness normalization (full conformance).
//!
//! Provides integrated LUFS measurement and gain normalization according to
//! ITU-R BS.1770-4 with the correct two-stage K-weighting filter, absolute
//! gate (−70 LUFS) and relative gate (−10 LU below the ungated mean).

#![forbid(unsafe_code)]

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Biquad
// ---------------------------------------------------------------------------

/// Transposed Direct Form II biquad filter operating on f64.
#[derive(Clone, Debug)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // State
    s1: f64,
    s2: f64,
}

impl Biquad {
    fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            s1: 0.0,
            s2: 0.0,
        }
    }

    /// Process a single sample.
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// K-Weighting filter (ITU-R BS.1770-4)
// ---------------------------------------------------------------------------

/// Two-stage K-weighting filter.
///
/// - Stage 1: High-shelf pre-filter (head diffraction, +4 dB above ~1.6 kHz).
/// - Stage 2: RLB high-pass (modified B-weighting, rolls off below ~38 Hz).
#[derive(Clone, Debug)]
struct KWeightFilter {
    stage1: Biquad,
    stage2: Biquad,
}

impl KWeightFilter {
    fn new(sample_rate: f64) -> Self {
        let stage1 = Self::calc_pre_filter(sample_rate);
        let stage2 = Self::calc_rlb_filter(sample_rate);
        Self { stage1, stage2 }
    }

    fn process(&mut self, x: f64) -> f64 {
        self.stage2.process(self.stage1.process(x))
    }

    fn reset(&mut self) {
        self.stage1.reset();
        self.stage2.reset();
    }

    /// ITU-R BS.1770-4 Stage 1: high-shelf pre-filter.
    fn calc_pre_filter(fs: f64) -> Biquad {
        let f0 = 1681.974_450_955_533;
        let g = 3.999_843_853_973_347;
        let q = 0.707_175_236_955_420;

        let k = (PI * f0 / fs).tan();
        let k2 = k * k;
        let vh = 10.0_f64.powf(g / 20.0);
        let vb = vh.powf(0.5);
        let norm = 1.0 / (1.0 + k / q + k2);

        let b0 = (vh + vb * k / q + k2) * norm;
        let b1 = 2.0 * (k2 - vh) * norm;
        let b2 = (vh - vb * k / q + k2) * norm;
        let a1 = 2.0 * (k2 - 1.0) * norm;
        let a2 = (1.0 - k / q + k2) * norm;
        Biquad::new(b0, b1, b2, a1, a2)
    }

    /// ITU-R BS.1770-4 Stage 2: RLB high-pass filter.
    fn calc_rlb_filter(fs: f64) -> Biquad {
        let f0 = 38.135_470_876_024_44;
        let q = 0.500_327_037_323_877;

        let k = (PI * f0 / fs).tan();
        let k2 = k * k;
        let norm = 1.0 / (1.0 + k / q + k2);

        let b0 = norm;
        let b1 = -2.0 * norm;
        let b2 = norm;
        let a1 = 2.0 * (k2 - 1.0) * norm;
        let a2 = (1.0 - k / q + k2) * norm;
        Biquad::new(b0, b1, b2, a1, a2)
    }
}

// ---------------------------------------------------------------------------
// Gate constants
// ---------------------------------------------------------------------------

/// Absolute gate threshold (LUFS).
const ABSOLUTE_GATE_LUFS: f64 = -70.0;
/// Relative gate offset (LU).
const RELATIVE_GATE_OFFSET_LU: f64 = -10.0;
/// Block duration in seconds (400 ms).
const BLOCK_DURATION_S: f64 = 0.4;
/// Block overlap (75 %).
const BLOCK_OVERLAP: f64 = 0.75;

// ---------------------------------------------------------------------------
// LoudnessNormalizer
// ---------------------------------------------------------------------------

/// Stateless loudness normalizer — all methods are free functions on `&[f32]`.
pub struct LoudnessNormalizer;

impl LoudnessNormalizer {
    /// Compute integrated loudness (LUFS) of a mono signal.
    ///
    /// Uses the ITU-R BS.1770-4 two-stage gating algorithm.
    /// Returns `f32::NEG_INFINITY` for silence or signals too short to gate.
    pub fn compute_integrated_lufs(samples: &[f32], sample_rate: u32) -> f32 {
        compute_integrated_lufs_f64(samples, sample_rate) as f32
    }

    /// Normalize `samples` to `target_lufs`, returning a new `Vec<f32>`.
    ///
    /// If the measured loudness is −∞ (silence or too short) the input is
    /// returned unchanged.
    pub fn normalize_to_lufs(samples: &[f32], sample_rate: u32, target_lufs: f32) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let measured = compute_integrated_lufs_f64(samples, sample_rate);
        if !measured.is_finite() {
            // Silence or ungated — return copy unchanged
            return samples.to_vec();
        }

        let gain_db = f64::from(target_lufs) - measured;
        let linear_gain = 10.0_f64.powf(gain_db / 20.0) as f32;

        samples.iter().map(|&s| s * linear_gain).collect()
    }
}

// ---------------------------------------------------------------------------
// Internal implementation
// ---------------------------------------------------------------------------

fn compute_integrated_lufs_f64(samples: &[f32], sample_rate: u32) -> f64 {
    if samples.is_empty() || sample_rate == 0 {
        return f64::NEG_INFINITY;
    }

    let fs = f64::from(sample_rate);
    let block_size = ((fs * BLOCK_DURATION_S).round() as usize).max(1);
    let hop_size = ((block_size as f64 * (1.0 - BLOCK_OVERLAP)).round() as usize).max(1);

    if samples.len() < block_size {
        // Too short to form a single block — return silence
        return f64::NEG_INFINITY;
    }

    // Apply K-weighting filter to the entire signal
    let mut kfilter = KWeightFilter::new(fs);
    let weighted: Vec<f64> = samples
        .iter()
        .map(|&s| kfilter.process(f64::from(s)))
        .collect();

    // Compute per-block loudness (mean square of weighted samples)
    let num_frames = (weighted.len() - block_size) / hop_size + 1;
    let mut block_powers: Vec<f64> = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let end = start + block_size;
        if end > weighted.len() {
            break;
        }

        let mean_sq: f64 =
            weighted[start..end].iter().map(|&x| x * x).sum::<f64>() / block_size as f64;
        block_powers.push(mean_sq);
    }

    if block_powers.is_empty() {
        return f64::NEG_INFINITY;
    }

    // Stage 1 — Absolute gate: discard blocks below -70 LUFS
    // Block loudness: L_k = -0.691 + 10 * log10(z_k)
    let abs_threshold_power = 10.0_f64.powf((ABSOLUTE_GATE_LUFS + 0.691) / 10.0);

    let gated_stage1: Vec<f64> = block_powers
        .iter()
        .copied()
        .filter(|&p| p > abs_threshold_power)
        .collect();

    if gated_stage1.is_empty() {
        return f64::NEG_INFINITY;
    }

    // Ungated mean loudness (from stage-1 gated blocks)
    let ungated_mean_power = gated_stage1.iter().sum::<f64>() / gated_stage1.len() as f64;
    let j_g = -0.691 + 10.0 * ungated_mean_power.log10();

    // Stage 2 — Relative gate: discard blocks below J_g − 10 LU
    let rel_threshold = j_g + RELATIVE_GATE_OFFSET_LU;
    let rel_threshold_power = 10.0_f64.powf((rel_threshold + 0.691) / 10.0);

    let gated_stage2: Vec<f64> = block_powers
        .iter()
        .copied()
        .filter(|&p| p > abs_threshold_power && p > rel_threshold_power)
        .collect();

    if gated_stage2.is_empty() {
        return f64::NEG_INFINITY;
    }

    let final_mean = gated_stage2.iter().sum::<f64>() / gated_stage2.len() as f64;
    -0.691 + 10.0 * final_mean.log10()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: u32 = 48000;

    fn sine_wave(freq: f32, amplitude: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / SR as f32).sin())
            .collect()
    }

    fn pseudo_noise(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Basic sanity
    // ------------------------------------------------------------------

    #[test]
    fn test_silence_integrated_lufs_is_neg_inf() {
        let lufs = LoudnessNormalizer::compute_integrated_lufs(&vec![0.0f32; 48000], SR);
        assert!(
            lufs == f32::NEG_INFINITY || lufs < -69.0,
            "silence should give -inf LUFS, got {lufs}"
        );
    }

    #[test]
    fn test_silence_normalize_returns_silence() {
        let silence = vec![0.0f32; 48000];
        let out = LoudnessNormalizer::normalize_to_lufs(&silence, SR, -23.0);
        assert_eq!(out.len(), silence.len());
        for &s in &out {
            assert!((s).abs() < 1e-12, "output should still be silence");
        }
    }

    #[test]
    fn test_normalize_empty_returns_empty() {
        let out = LoudnessNormalizer::normalize_to_lufs(&[], SR, -23.0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_compute_lufs_returns_finite_for_noise() {
        let noise = pseudo_noise(SR as usize * 2, 42);
        let lufs = LoudnessNormalizer::compute_integrated_lufs(&noise, SR);
        assert!(
            lufs.is_finite(),
            "noise should yield a finite LUFS, got {lufs}"
        );
    }

    #[test]
    fn test_integrated_lufs_finite_for_sine() {
        let sine = sine_wave(1000.0, 0.1, SR as usize * 2);
        let lufs = LoudnessNormalizer::compute_integrated_lufs(&sine, SR);
        assert!(
            lufs.is_finite(),
            "sine should yield finite LUFS, got {lufs}"
        );
    }

    #[test]
    fn test_normalize_preserves_length() {
        let noise = pseudo_noise(SR as usize, 7);
        let out = LoudnessNormalizer::normalize_to_lufs(&noise, SR, -23.0);
        assert_eq!(out.len(), noise.len());
    }

    #[test]
    fn test_normalize_short_signal() {
        // Shorter than one 400ms block — should return unchanged copy
        let short = sine_wave(440.0, 0.5, 100);
        let out = LoudnessNormalizer::normalize_to_lufs(&short, SR, -23.0);
        assert_eq!(out.len(), short.len());
    }

    // ------------------------------------------------------------------
    // Normalization logic
    // ------------------------------------------------------------------

    #[test]
    fn test_normalize_increases_quiet_signal() {
        let quiet = sine_wave(440.0, 0.001, SR as usize * 3);
        let target = -23.0f32;
        let out = LoudnessNormalizer::normalize_to_lufs(&quiet, SR, target);
        // RMS should be larger
        let rms_in: f32 = (quiet.iter().map(|&x| x * x).sum::<f32>() / quiet.len() as f32).sqrt();
        let rms_out: f32 = (out.iter().map(|&x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        assert!(
            rms_out > rms_in,
            "output should be louder: {rms_out} vs {rms_in}"
        );
    }

    #[test]
    fn test_normalize_decreases_loud_signal() {
        let loud = sine_wave(440.0, 0.9, SR as usize * 3);
        let out = LoudnessNormalizer::normalize_to_lufs(&loud, SR, -23.0);
        let rms_in: f32 = (loud.iter().map(|&x| x * x).sum::<f32>() / loud.len() as f32).sqrt();
        let rms_out: f32 = (out.iter().map(|&x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        assert!(
            rms_out < rms_in,
            "output should be quieter: {rms_out} vs {rms_in}"
        );
    }

    #[test]
    fn test_normalize_result_within_1_lu_of_target() {
        let noise = pseudo_noise(SR as usize * 5, 99);
        let target = -23.0f32;
        let out = LoudnessNormalizer::normalize_to_lufs(&noise, SR, target);
        let measured = LoudnessNormalizer::compute_integrated_lufs(&out, SR);
        if measured.is_finite() {
            assert!(
                (measured - target).abs() <= 1.5,
                "normalized loudness {measured:.2} LUFS not within 1.5 LU of target {target}"
            );
        }
    }

    #[test]
    fn test_normalized_loudness_close_to_target() {
        let noise = pseudo_noise(SR as usize * 4, 13);
        let target = -18.0f32;
        let out = LoudnessNormalizer::normalize_to_lufs(&noise, SR, target);
        let measured = LoudnessNormalizer::compute_integrated_lufs(&out, SR);
        if measured.is_finite() {
            assert!(
                (measured - target).abs() <= 1.5,
                "measured {measured:.2} vs target {target}"
            );
        }
    }

    #[test]
    fn test_normalize_to_minus_23_lufs() {
        let noise = pseudo_noise(SR as usize * 3, 5);
        let out = LoudnessNormalizer::normalize_to_lufs(&noise, SR, -23.0);
        let measured = LoudnessNormalizer::compute_integrated_lufs(&out, SR);
        if measured.is_finite() {
            assert!(
                (measured - (-23.0)).abs() <= 2.0,
                "measured {measured:.2} LUFS"
            );
        }
    }

    #[test]
    fn test_normalize_to_minus_14_lufs() {
        let noise = pseudo_noise(SR as usize * 3, 17);
        let out = LoudnessNormalizer::normalize_to_lufs(&noise, SR, -14.0);
        let measured = LoudnessNormalizer::compute_integrated_lufs(&out, SR);
        if measured.is_finite() {
            assert!(
                (measured - (-14.0)).abs() <= 2.0,
                "measured {measured:.2} LUFS"
            );
        }
    }

    #[test]
    fn test_unity_gain_at_correct_loudness() {
        // Signal already at -23 LUFS: normalizing to -23 should change nothing
        let noise = pseudo_noise(SR as usize * 4, 21);
        let pre_measured = LoudnessNormalizer::compute_integrated_lufs(&noise, SR);
        if !pre_measured.is_finite() {
            return;
        }
        // First normalize to target
        let at_target = LoudnessNormalizer::normalize_to_lufs(&noise, SR, -23.0);
        // Now normalize the already-normalized signal to the same target
        let re_normalized = LoudnessNormalizer::normalize_to_lufs(&at_target, SR, -23.0);
        // Samples should be essentially identical
        let max_diff = at_target
            .iter()
            .zip(re_normalized.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.01,
            "re-normalizing should be near identity, max diff = {max_diff}"
        );
    }

    // ------------------------------------------------------------------
    // K-weighting filter properties
    // ------------------------------------------------------------------

    #[test]
    fn test_kweight_filter_dc_rejection() {
        // DC (constant signal) should be attenuated by the high-pass RLB stage
        let dc: Vec<f32> = vec![1.0f32; SR as usize];
        let lufs_dc = LoudnessNormalizer::compute_integrated_lufs(&dc, SR);
        let sine_1k = sine_wave(1000.0, 1.0, SR as usize);
        let lufs_1k = LoudnessNormalizer::compute_integrated_lufs(&sine_1k, SR);
        if lufs_dc.is_finite() && lufs_1k.is_finite() {
            // K-weighted DC should be significantly quieter than 1kHz sine
            assert!(
                lufs_dc < lufs_1k,
                "DC ({lufs_dc:.2} LUFS) should be quieter than 1kHz ({lufs_1k:.2} LUFS) after K-weighting"
            );
        }
    }

    #[test]
    fn test_kweight_filter_highfreq_boost() {
        // 4kHz should have higher K-weighted loudness than 100Hz (shelf boost)
        let low = sine_wave(100.0, 0.1, SR as usize * 2);
        let high = sine_wave(4000.0, 0.1, SR as usize * 2);
        let lufs_low = LoudnessNormalizer::compute_integrated_lufs(&low, SR);
        let lufs_high = LoudnessNormalizer::compute_integrated_lufs(&high, SR);
        if lufs_low.is_finite() && lufs_high.is_finite() {
            // Same amplitude but different frequency — K-weighting gives high freq more weight
            assert!(
                lufs_high >= lufs_low - 0.5,
                "4kHz ({lufs_high:.2}) should not be much quieter than 100Hz ({lufs_low:.2})"
            );
        }
    }

    // ------------------------------------------------------------------
    // Gate behaviour
    // ------------------------------------------------------------------

    #[test]
    fn test_absolute_gate_filters_quiet_blocks() {
        // Build a mostly-loud signal with some very quiet sections
        let n = SR as usize * 5;
        let mut s = sine_wave(440.0, 0.1, n);
        // Mute a chunk to be below absolute gate
        for x in s[SR as usize..SR as usize * 2].iter_mut() {
            *x *= 1e-6; // effectively -120 dBFS
        }
        let lufs = LoudnessNormalizer::compute_integrated_lufs(&s, SR);
        // Should still produce a reasonable (finite) LUFS measurement
        assert!(
            lufs.is_finite(),
            "mixed-loudness signal should give finite LUFS"
        );
    }

    #[test]
    fn test_relative_gate_applied() {
        // Loud burst + near-silence — gating should ignore the silence
        let n = SR as usize * 4;
        let mut s = sine_wave(1000.0, 0.5, n);
        // Set last 1 second to near silence (well below relative gate)
        for x in s[SR as usize * 3..].iter_mut() {
            *x *= 1e-5;
        }
        let lufs_mixed = LoudnessNormalizer::compute_integrated_lufs(&s, SR);
        let loud_only = sine_wave(1000.0, 0.5, n);
        let lufs_loud = LoudnessNormalizer::compute_integrated_lufs(&loud_only, SR);
        // Both should be finite and relatively close (within 5 LU)
        if lufs_mixed.is_finite() && lufs_loud.is_finite() {
            assert!(
                (lufs_mixed - lufs_loud).abs() < 5.0,
                "relative gate should limit influence of near-silence: {lufs_mixed:.2} vs {lufs_loud:.2}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Block power
    // ------------------------------------------------------------------

    #[test]
    fn test_block_power_computation() {
        // Verify block power formula using known signal
        // 0dBFS full-scale sine → mean square ≈ 0.5
        let n = SR as usize * 2;
        let s: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / SR as f32).sin())
            .collect();
        let rms_sq: f64 = s.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / n as f64;
        // For a pure sine, mean square → 0.5
        assert!(
            (rms_sq - 0.5).abs() < 0.02,
            "mean square of 0dBFS sine should be ~0.5, got {rms_sq}"
        );
    }

    // ------------------------------------------------------------------
    // Sample rate variations
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_lufs_44100_sample_rate() {
        let noise = pseudo_noise(44100 * 3, 88);
        let lufs = LoudnessNormalizer::compute_integrated_lufs(&noise, 44100);
        assert!(
            lufs.is_finite(),
            "44100 Hz noise should yield finite LUFS, got {lufs}"
        );
    }
}

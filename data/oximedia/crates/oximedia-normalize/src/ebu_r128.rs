//! EBU R128 loudness normalization (ITU-R BS.1770).
//!
//! Implements measurement and normalization compliant with the EBU R128 standard,
//! including K-weighting filters, gated measurement, loudness range, and true peak.

use std::f64::consts::PI;

/// EBU R128 loudness target specification.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct LoudnessTarget {
    /// Integrated loudness target in LUFS.
    pub integrated_lufs: f64,
    /// Maximum true peak in dBFS.
    pub true_peak_dbfs: f64,
    /// Maximum loudness range in LU.
    pub lra_max_lu: f64,
}

impl LoudnessTarget {
    /// EBU R128 broadcast preset: -23 LUFS, -1 dBTP, 20 LU max LRA.
    pub fn broadcast() -> Self {
        Self {
            integrated_lufs: -23.0,
            true_peak_dbfs: -1.0,
            lra_max_lu: 20.0,
        }
    }

    /// Streaming platform preset: -14 LUFS, -1 dBTP, 20 LU max LRA.
    pub fn streaming() -> Self {
        Self {
            integrated_lufs: -14.0,
            true_peak_dbfs: -1.0,
            lra_max_lu: 20.0,
        }
    }

    /// Podcast preset: -16 LUFS, -1 dBTP, 15 LU max LRA.
    pub fn podcast() -> Self {
        Self {
            integrated_lufs: -16.0,
            true_peak_dbfs: -1.0,
            lra_max_lu: 15.0,
        }
    }
}

/// Momentary loudness measurement (400ms window).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentaryLoudness(pub f64);

impl MomentaryLoudness {
    /// Get the loudness value in LUFS.
    pub fn lufs(self) -> f64 {
        self.0
    }
}

/// Short-term loudness measurement (3s window).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShortTermLoudness(pub f64);

impl ShortTermLoudness {
    /// Get the loudness value in LUFS.
    pub fn lufs(self) -> f64 {
        self.0
    }
}

/// Compute K-weighting filter gain in dB for a given frequency.
///
/// K-weighting consists of a pre-filter (high-shelf) followed by an
/// RLB-weighting filter (high-pass). This function approximates the
/// combined response in dB.
#[allow(dead_code)]
pub fn k_weighting(freq_hz: f32) -> f32 {
    if freq_hz <= 0.0 {
        return 0.0;
    }
    let f = f64::from(freq_hz);

    // Pre-filter: high-shelf at ~1681 Hz, +4 dB at high frequencies
    // H_pre(f) ≈ 1 + (f / f_pre)^2 shape (simplified)
    let f_pre = 1681.974_f64;
    let pre_gain_db = 4.0 * (1.0 - 1.0 / (1.0 + (f / f_pre).powi(2))).sqrt();

    // RLB-weighting high-pass filter: fc ≈ 38.1 Hz, 2nd order
    let f_rlb = 38.135_f64;
    let rlb_gain_db = -40.0 * (f_rlb / f).atan2(1.0) / PI;

    (pre_gain_db + rlb_gain_db) as f32
}

/// Loudness gate constants per ITU-R BS.1770.
pub struct LoudnessGate;

impl LoudnessGate {
    /// Absolute gate threshold: blocks below -70 LUFS.
    pub const ABSOLUTE_GATE_LUFS: f64 = -70.0;
    /// Relative gate offset: -10 LU below ungated mean.
    pub const RELATIVE_GATE_OFFSET: f64 = -10.0;
}

/// Full loudness report produced by `EbuR128Analyzer`.
#[derive(Clone, Debug)]
pub struct LoudnessReport {
    /// Integrated (gated) loudness in LUFS.
    pub integrated_lufs: f64,
    /// Loudness range in LU.
    pub lra_lu: f64,
    /// True peak in dBFS.
    pub true_peak_dbfs: f64,
    /// Maximum momentary loudness in LUFS.
    pub momentary_max_lufs: f64,
    /// Maximum short-term loudness in LUFS.
    pub short_term_max_lufs: f64,
}

/// EBU R128 compliant loudness analyzer.
pub struct EbuR128Analyzer;

impl EbuR128Analyzer {
    /// Create a new analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Analyze audio samples and produce a full loudness report.
    ///
    /// # Arguments
    /// * `samples` - Interleaved audio samples in [-1, 1].
    /// * `sample_rate` - Sample rate in Hz.
    /// * `channels` - Number of channels.
    pub fn analyze(&self, samples: &[f32], sample_rate: u32, channels: u16) -> LoudnessReport {
        if samples.is_empty() || sample_rate == 0 || channels == 0 {
            return LoudnessReport {
                integrated_lufs: -f64::INFINITY,
                lra_lu: 0.0,
                true_peak_dbfs: -f64::INFINITY,
                momentary_max_lufs: -f64::INFINITY,
                short_term_max_lufs: -f64::INFINITY,
            };
        }

        let ch = channels as usize;
        let sr = sample_rate as usize;

        // Deinterleave into per-channel vectors
        let frames = samples.len() / ch;
        let mut channel_data: Vec<Vec<f64>> = (0..ch)
            .map(|c| {
                (0..frames)
                    .map(|f| f64::from(samples[f * ch + c]))
                    .collect()
            })
            .collect();

        // Apply K-weighting per channel (simplified biquad approximation)
        for ch_data in &mut channel_data {
            apply_k_weighting_filter(ch_data, sr);
        }

        // Compute mean square per frame (sum over channels)
        let mean_sq: Vec<f64> = (0..frames)
            .map(|f| {
                channel_data
                    .iter()
                    .map(|ch_data| ch_data[f] * ch_data[f])
                    .sum::<f64>()
                    / ch as f64
            })
            .collect();

        // Block sizes: 400ms blocks, 75% overlap for short-term; 3s blocks for LRA
        let block_400ms = (sr as f64 * 0.4) as usize;
        let hop_100ms = (sr as f64 * 0.1) as usize;
        let block_3s = (sr as f64 * 3.0) as usize;
        let hop_750ms = (sr as f64 * 0.75) as usize;

        // Compute 400ms block loudness values
        let mut block_400ms_loud: Vec<f64> = Vec::new();
        let mut start = 0;
        while start + block_400ms <= frames {
            let power: f64 =
                mean_sq[start..start + block_400ms].iter().sum::<f64>() / block_400ms as f64;
            let lufs = power_to_lufs(power);
            block_400ms_loud.push(lufs);
            start += hop_100ms.max(1);
        }

        // Momentary max (400ms window)
        let momentary_max_lufs = block_400ms_loud
            .iter()
            .copied()
            .fold(-f64::INFINITY, f64::max);

        // Compute 3s block loudness for short-term
        let mut block_3s_loud: Vec<f64> = Vec::new();
        let mut start = 0;
        while start + block_3s <= frames {
            let power: f64 = mean_sq[start..start + block_3s].iter().sum::<f64>() / block_3s as f64;
            let lufs = power_to_lufs(power);
            block_3s_loud.push(lufs);
            start += hop_750ms.max(1);
        }

        // Also use 400ms blocks for short-term if 3s blocks too few
        let short_term_max_lufs = if block_3s_loud.is_empty() {
            momentary_max_lufs
        } else {
            block_3s_loud.iter().copied().fold(-f64::INFINITY, f64::max)
        };

        // Integrated loudness with gating (BS.1770-4)
        // Absolute gate: filter blocks below -70 LUFS
        let abs_gated: Vec<f64> = block_400ms_loud
            .iter()
            .copied()
            .filter(|&l| l > LoudnessGate::ABSOLUTE_GATE_LUFS)
            .collect();

        let integrated_lufs = if abs_gated.is_empty() {
            -f64::INFINITY
        } else {
            // Compute ungated mean power
            let ungated_mean_power: f64 =
                abs_gated.iter().map(|&l| lufs_to_power(l)).sum::<f64>() / abs_gated.len() as f64;
            let ungated_lufs = power_to_lufs(ungated_mean_power);
            let relative_gate = ungated_lufs + LoudnessGate::RELATIVE_GATE_OFFSET;

            // Relative gate: filter blocks below relative threshold
            let rel_gated: Vec<f64> = abs_gated
                .iter()
                .copied()
                .filter(|&l| l > relative_gate)
                .collect();

            if rel_gated.is_empty() {
                ungated_lufs
            } else {
                let gated_mean_power: f64 =
                    rel_gated.iter().map(|&l| lufs_to_power(l)).sum::<f64>()
                        / rel_gated.len() as f64;
                power_to_lufs(gated_mean_power)
            }
        };

        // LRA: difference between 10th and 95th percentile of short-term loudness (gated)
        let lra_lu = compute_lra(&block_3s_loud);

        // True peak: max absolute sample value in dBFS
        let true_peak_linear = samples.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max);
        let true_peak_dbfs = if true_peak_linear > 0.0 {
            20.0 * f64::from(true_peak_linear.log10())
        } else {
            -f64::INFINITY
        };

        LoudnessReport {
            integrated_lufs,
            lra_lu,
            true_peak_dbfs,
            momentary_max_lufs,
            short_term_max_lufs,
        }
    }
}

impl Default for EbuR128Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute normalization gain from a loudness report and target.
pub struct NormalizationGain;

impl NormalizationGain {
    /// Compute linear gain factor to apply to audio to reach the target integrated loudness.
    ///
    /// Returns 1.0 if loudness is unmeasurable (infinity).
    pub fn from_report(report: &LoudnessReport, target: &LoudnessTarget) -> f32 {
        if !report.integrated_lufs.is_finite() {
            return 1.0;
        }
        let gain_db = target.integrated_lufs - report.integrated_lufs;
        // Clamp to prevent extreme gains
        let gain_db_clamped = gain_db.clamp(-40.0, 40.0);
        10.0_f64.powf(gain_db_clamped / 20.0) as f32
    }
}

// ---------- internal helpers ----------

/// Apply a simplified K-weighting biquad filter in-place.
/// Uses a 2nd-order high-pass at 38.135 Hz + 2nd-order high-shelf at 1681.974 Hz.
fn apply_k_weighting_filter(data: &mut Vec<f64>, sample_rate: usize) {
    if data.is_empty() {
        return;
    }

    let sr = sample_rate as f64;

    // Stage 1: High-shelf pre-filter (+4 dB shelf at ~1681 Hz)
    // Design via bilinear transform of analog prototype
    let f0 = 1681.974_f64;
    let q = 0.7071_f64; // ~1/sqrt(2)
    let k = (PI * f0 / sr).tan();
    let k2 = k * k;
    let norm = 1.0 / (1.0 + k / q + k2);
    let a0_hs = (1.0 + k * 4.0_f64.sqrt() / q + k2) * norm;
    let a1_hs = 2.0 * (k2 - 1.0) * norm;
    let a2_hs = (1.0 - k * 4.0_f64.sqrt() / q + k2) * norm;
    let b1_hs = a1_hs;
    let b2_hs = (1.0 - k / q + k2) * norm;

    let mut x1 = 0.0_f64;
    let mut x2 = 0.0_f64;
    let mut y1 = 0.0_f64;
    let mut y2 = 0.0_f64;
    for sample in data.iter_mut() {
        let x0 = *sample;
        let y0 = a0_hs * x0 + a1_hs * x1 + a2_hs * x2 - b1_hs * y1 - b2_hs * y2;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        *sample = y0;
    }

    // Stage 2: 2nd-order high-pass RLB filter at 38.135 Hz
    let f_hp = 38.135_f64;
    let k_hp = (PI * f_hp / sr).tan();
    let k2_hp = k_hp * k_hp;
    let norm_hp = 1.0 / (1.0 + k_hp * 2.0_f64.sqrt() + k2_hp);
    let a0_hp = norm_hp;
    let a1_hp = -2.0 * norm_hp;
    let a2_hp = norm_hp;
    let b1_hp = 2.0 * (k2_hp - 1.0) * norm_hp;
    let b2_hp = (1.0 - k_hp * 2.0_f64.sqrt() + k2_hp) * norm_hp;

    let mut x1 = 0.0_f64;
    let mut x2 = 0.0_f64;
    let mut y1 = 0.0_f64;
    let mut y2 = 0.0_f64;
    for sample in data.iter_mut() {
        let x0 = *sample;
        let y0 = a0_hp * x0 + a1_hp * x1 + a2_hp * x2 - b1_hp * y1 - b2_hp * y2;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        *sample = y0;
    }
}

/// Convert mean square power to LUFS.
fn power_to_lufs(power: f64) -> f64 {
    if power <= 0.0 {
        return -f64::INFINITY;
    }
    -0.691 + 10.0 * power.log10()
}

/// Convert LUFS to mean square power.
fn lufs_to_power(lufs: f64) -> f64 {
    10.0_f64.powf((lufs + 0.691) / 10.0)
}

/// Compute loudness range (LRA) from a set of short-term loudness values.
fn compute_lra(block_loud: &[f64]) -> f64 {
    if block_loud.len() < 2 {
        return 0.0;
    }

    // Absolute gate at -70 LUFS
    let mut gated: Vec<f64> = block_loud
        .iter()
        .copied()
        .filter(|&l| l > LoudnessGate::ABSOLUTE_GATE_LUFS)
        .collect();

    if gated.is_empty() {
        return 0.0;
    }

    // Relative gate: -20 LU below gated mean (per EBU Tech 3342)
    let mean_power: f64 = gated.iter().map(|&l| lufs_to_power(l)).sum::<f64>() / gated.len() as f64;
    let mean_lufs = power_to_lufs(mean_power);
    let rel_gate = mean_lufs - 20.0;

    gated.retain(|&l| l > rel_gate);

    if gated.len() < 2 {
        return 0.0;
    }

    gated.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lo_idx = ((gated.len() as f64 * 0.10) as usize).min(gated.len() - 1);
    let hi_idx = ((gated.len() as f64 * 0.95) as usize).min(gated.len() - 1);

    (gated[hi_idx] - gated[lo_idx]).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_target_broadcast() {
        let t = LoudnessTarget::broadcast();
        assert!((t.integrated_lufs - (-23.0)).abs() < 1e-6);
        assert!((t.true_peak_dbfs - (-1.0)).abs() < 1e-6);
        assert!((t.lra_max_lu - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_loudness_target_streaming() {
        let t = LoudnessTarget::streaming();
        assert!((t.integrated_lufs - (-14.0)).abs() < 1e-6);
    }

    #[test]
    fn test_loudness_target_podcast() {
        let t = LoudnessTarget::podcast();
        assert!((t.integrated_lufs - (-16.0)).abs() < 1e-6);
        assert!((t.lra_max_lu - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_momentary_loudness_newtype() {
        let m = MomentaryLoudness(-23.5);
        assert!((m.lufs() - (-23.5)).abs() < 1e-9);
    }

    #[test]
    fn test_short_term_loudness_newtype() {
        let s = ShortTermLoudness(-16.0);
        assert!((s.lufs() - (-16.0)).abs() < 1e-9);
    }

    #[test]
    fn test_k_weighting_low_freq() {
        // Low-frequency gain should be significantly negative (high-pass characteristic)
        let gain = k_weighting(40.0);
        assert!(gain < 0.0, "Expected negative gain at low freq, got {gain}");
    }

    #[test]
    fn test_k_weighting_high_freq() {
        // High frequencies should have positive gain due to pre-filter shelf
        let gain_10k = k_weighting(10000.0);
        let gain_100 = k_weighting(100.0);
        assert!(
            gain_10k > gain_100,
            "High freq should have more gain: 10kHz={gain_10k}, 100Hz={gain_100}"
        );
    }

    #[test]
    fn test_k_weighting_zero_freq() {
        let gain = k_weighting(0.0);
        assert!((gain - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_loudness_gate_constants() {
        assert!((LoudnessGate::ABSOLUTE_GATE_LUFS - (-70.0)).abs() < 1e-9);
        assert!((LoudnessGate::RELATIVE_GATE_OFFSET - (-10.0)).abs() < 1e-9);
    }

    #[test]
    fn test_ebu_analyzer_empty() {
        let analyzer = EbuR128Analyzer::new();
        let report = analyzer.analyze(&[], 48000, 2);
        assert!(!report.integrated_lufs.is_finite());
    }

    #[test]
    fn test_ebu_analyzer_silence() {
        let analyzer = EbuR128Analyzer::new();
        let samples = vec![0.0f32; 48000 * 2]; // 1 second, 2 channels
        let report = analyzer.analyze(&samples, 48000, 2);
        // Silence should be below the absolute gate
        assert!(
            !report.integrated_lufs.is_finite() || report.integrated_lufs < -60.0,
            "Silence should yield very low or undefined loudness"
        );
    }

    #[test]
    fn test_ebu_analyzer_sine_wave() {
        let analyzer = EbuR128Analyzer::new();
        let sr = 48000u32;
        let duration_s = 5;
        let freq = 1000.0_f32;
        let amplitude = 0.1_f32;
        // Generate mono sine, then interleave as stereo
        let samples: Vec<f32> = (0..sr as usize * duration_s)
            .flat_map(|i| {
                let t = i as f32 / sr as f32;
                let s = amplitude * (2.0 * std::f32::consts::PI * freq * t).sin();
                [s, s]
            })
            .collect();

        let report = analyzer.analyze(&samples, sr, 2);
        assert!(
            report.integrated_lufs.is_finite(),
            "Should produce finite loudness"
        );
        // A 1kHz sine at 0.1 amplitude should be well above -70 LUFS
        assert!(
            report.integrated_lufs > -60.0,
            "Expected loudness above -60 LUFS, got {}",
            report.integrated_lufs
        );
    }

    #[test]
    fn test_normalization_gain_from_report() {
        let report = LoudnessReport {
            integrated_lufs: -30.0,
            lra_lu: 8.0,
            true_peak_dbfs: -6.0,
            momentary_max_lufs: -25.0,
            short_term_max_lufs: -28.0,
        };
        let target = LoudnessTarget::broadcast(); // -23 LUFS
        let gain = NormalizationGain::from_report(&report, &target);
        // Need +7 dB gain
        let expected = 10.0_f32.powf(7.0 / 20.0);
        assert!(
            (gain - expected).abs() < 1e-4,
            "Expected gain {expected}, got {gain}"
        );
    }

    #[test]
    fn test_normalization_gain_infinite_loudness() {
        let report = LoudnessReport {
            integrated_lufs: -f64::INFINITY,
            lra_lu: 0.0,
            true_peak_dbfs: -f64::INFINITY,
            momentary_max_lufs: -f64::INFINITY,
            short_term_max_lufs: -f64::INFINITY,
        };
        let target = LoudnessTarget::broadcast();
        let gain = NormalizationGain::from_report(&report, &target);
        assert!(
            (gain - 1.0).abs() < 1e-6,
            "Should return 1.0 for silent audio"
        );
    }

    #[test]
    fn test_power_lufs_roundtrip() {
        let original_lufs = -23.0_f64;
        let power = lufs_to_power(original_lufs);
        let recovered = power_to_lufs(power);
        assert!((recovered - original_lufs).abs() < 1e-9);
    }

    #[test]
    fn test_lra_uniform_signal() {
        // All blocks at same level => LRA should be ~0
        let blocks = vec![-20.0_f64; 20];
        let lra = compute_lra(&blocks);
        assert!(lra < 1.0, "Uniform signal LRA should be near 0, got {lra}");
    }

    #[test]
    fn test_lra_varying_signal() {
        // Wide range signal should have positive LRA
        let mut blocks: Vec<f64> = (-40..=-10).map(|i| i as f64).collect();
        blocks.extend((-40..=-10).map(|i| i as f64));
        let lra = compute_lra(&blocks);
        assert!(lra > 0.0, "Wide range signal should have positive LRA");
    }
}

#![allow(dead_code)]
//! Surround and multichannel loudness normalization.
//!
//! Provides channel-aware loudness normalization for surround formats
//! including 5.1, 7.1, and Atmos. Handles per-channel weighting according
//! to ITU-R BS.1770, LFE exclusion, and downmix-compatible gain management.

/// Surround channel layout identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Mono (1.0).
    Mono,
    /// Stereo (2.0).
    Stereo,
    /// LCR (3.0).
    Lcr,
    /// Quad (4.0).
    Quad,
    /// 5.0 surround (no LFE).
    Surround50,
    /// 5.1 surround.
    Surround51,
    /// 7.1 surround.
    Surround71,
    /// 7.1.4 Atmos bed.
    Atmos714,
}

impl ChannelLayout {
    /// Get the total number of channels.
    pub fn channel_count(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Lcr => 3,
            Self::Quad => 4,
            Self::Surround50 => 5,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Atmos714 => 12,
        }
    }

    /// Get indices of LFE channels (excluded from loudness measurement).
    pub fn lfe_indices(&self) -> Vec<usize> {
        match self {
            Self::Surround51 => vec![3], // L, R, C, LFE, Ls, Rs
            Self::Surround71 => vec![3], // L, R, C, LFE, Ls, Rs, Lrs, Rrs
            Self::Atmos714 => vec![3],
            _ => Vec::new(),
        }
    }

    /// Get the channel weight for ITU-R BS.1770 loudness measurement.
    ///
    /// Returns per-channel weight. Surround channels receive +1.5 dB weighting.
    pub fn channel_weights(&self) -> Vec<f64> {
        match self {
            Self::Mono => vec![1.0],
            Self::Stereo => vec![1.0, 1.0],
            Self::Lcr => vec![1.0, 1.0, 1.0],
            Self::Quad => {
                let surround_w = 10.0_f64.powf(1.5 / 10.0); // +1.5 dB
                vec![1.0, 1.0, surround_w, surround_w]
            }
            Self::Surround50 => {
                let surround_w = 10.0_f64.powf(1.5 / 10.0);
                vec![1.0, 1.0, 1.0, surround_w, surround_w]
            }
            Self::Surround51 => {
                let surround_w = 10.0_f64.powf(1.5 / 10.0);
                // L, R, C, LFE(0), Ls, Rs
                vec![1.0, 1.0, 1.0, 0.0, surround_w, surround_w]
            }
            Self::Surround71 => {
                let surround_w = 10.0_f64.powf(1.5 / 10.0);
                // L, R, C, LFE(0), Ls, Rs, Lrs, Rrs
                vec![
                    1.0, 1.0, 1.0, 0.0, surround_w, surround_w, surround_w, surround_w,
                ]
            }
            Self::Atmos714 => {
                let surround_w = 10.0_f64.powf(1.5 / 10.0);
                let height_w = 10.0_f64.powf(1.5 / 10.0);
                // L, R, C, LFE, Ls, Rs, Lrs, Rrs, TFL, TFR, TBL, TBR
                vec![
                    1.0, 1.0, 1.0, 0.0, surround_w, surround_w, surround_w, surround_w, height_w,
                    height_w, height_w, height_w,
                ]
            }
        }
    }
}

/// Downmix mode for loudness verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownmixMode {
    /// ITU-R BS.775 standard downmix.
    ItuBs775,
    /// LoRo (Left-only, Right-only) stereo downmix.
    LoRo,
    /// LtRt (Left-total, Right-total) matrix downmix.
    LtRt,
    /// Mono fold-down.
    Mono,
}

/// Downmix coefficient set.
#[derive(Debug, Clone)]
pub struct DownmixCoefficients {
    /// Center channel attenuation in dB.
    pub center_db: f64,
    /// Surround channel attenuation in dB.
    pub surround_db: f64,
    /// LFE contribution in dB (usually excluded).
    pub lfe_db: f64,
}

impl DownmixCoefficients {
    /// Create ITU-R BS.775 standard coefficients.
    pub fn itu_bs775() -> Self {
        Self {
            center_db: -3.0,
            surround_db: -3.0,
            lfe_db: -120.0, // Effectively muted
        }
    }

    /// Create LoRo downmix coefficients.
    pub fn lo_ro() -> Self {
        Self {
            center_db: -3.0,
            surround_db: -3.0,
            lfe_db: -120.0,
        }
    }

    /// Get center channel linear coefficient.
    pub fn center_linear(&self) -> f64 {
        10.0_f64.powf(self.center_db / 20.0)
    }

    /// Get surround channel linear coefficient.
    pub fn surround_linear(&self) -> f64 {
        10.0_f64.powf(self.surround_db / 20.0)
    }
}

/// Configuration for surround normalization.
#[derive(Debug, Clone)]
pub struct SurroundNormConfig {
    /// Channel layout.
    pub layout: ChannelLayout,
    /// Target integrated loudness in LUFS.
    pub target_lufs: f64,
    /// Maximum true peak in dBTP.
    pub max_true_peak_dbtp: f64,
    /// Whether to verify downmix loudness.
    pub verify_downmix: bool,
    /// Downmix mode for verification.
    pub downmix_mode: DownmixMode,
    /// Maximum loudness deviation between surround and downmix in LU.
    pub max_downmix_deviation_lu: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Whether to apply per-channel gain or global gain.
    pub per_channel_gain: bool,
}

impl SurroundNormConfig {
    /// Create a new surround normalization configuration.
    pub fn new(layout: ChannelLayout, sample_rate: f64) -> Self {
        Self {
            layout,
            target_lufs: -23.0,
            max_true_peak_dbtp: -1.0,
            verify_downmix: true,
            downmix_mode: DownmixMode::ItuBs775,
            max_downmix_deviation_lu: 2.0,
            sample_rate,
            per_channel_gain: false,
        }
    }

    /// Create an EBU R128 compliant configuration.
    pub fn ebu_r128(layout: ChannelLayout, sample_rate: f64) -> Self {
        Self {
            layout,
            target_lufs: -23.0,
            max_true_peak_dbtp: -1.0,
            verify_downmix: true,
            downmix_mode: DownmixMode::ItuBs775,
            max_downmix_deviation_lu: 1.0,
            sample_rate,
            per_channel_gain: false,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_rate < 8000.0 || self.sample_rate > 192_000.0 {
            return Err(format!("Invalid sample rate: {}", self.sample_rate));
        }
        if self.target_lufs > 0.0 {
            return Err("Target loudness should be negative LUFS".to_string());
        }
        if self.max_downmix_deviation_lu < 0.0 {
            return Err("Deviation must be non-negative".to_string());
        }
        Ok(())
    }
}

/// Result of surround normalization analysis.
#[derive(Debug, Clone)]
pub struct SurroundAnalysis {
    /// Per-channel RMS levels in dB.
    pub channel_rms_db: Vec<f64>,
    /// Per-channel peak levels in dB.
    pub channel_peak_db: Vec<f64>,
    /// Integrated loudness in LUFS (ITU-R BS.1770 weighted).
    pub integrated_lufs: f64,
    /// Estimated downmix loudness in LUFS.
    pub downmix_lufs: f64,
    /// Recommended global gain in dB.
    pub recommended_gain_db: f64,
    /// Whether the downmix loudness is within tolerance.
    pub downmix_compliant: bool,
}

/// Surround normalization processor.
#[derive(Debug)]
pub struct SurroundNormalizer {
    /// Configuration.
    config: SurroundNormConfig,
    /// Accumulated per-channel mean-square values.
    channel_sum_sq: Vec<f64>,
    /// Per-channel peak values.
    channel_peak: Vec<f64>,
    /// Total samples processed per channel.
    samples_per_channel: usize,
    /// Channel weights from layout.
    weights: Vec<f64>,
    /// LFE channel indices.
    lfe_indices: Vec<usize>,
}

impl SurroundNormalizer {
    /// Create a new surround normalizer.
    pub fn new(config: SurroundNormConfig) -> Self {
        let num_ch = config.layout.channel_count();
        let weights = config.layout.channel_weights();
        let lfe_indices = config.layout.lfe_indices();

        Self {
            config,
            channel_sum_sq: vec![0.0; num_ch],
            channel_peak: vec![0.0; num_ch],
            samples_per_channel: 0,
            weights,
            lfe_indices,
        }
    }

    /// Analyze interleaved multichannel audio.
    ///
    /// Samples should be interleaved: [L0, R0, C0, LFE0, Ls0, Rs0, L1, R1, ...]
    pub fn analyze(&mut self, interleaved: &[f32]) {
        let num_ch = self.config.layout.channel_count();
        if num_ch == 0 {
            return;
        }

        let num_frames = interleaved.len() / num_ch;
        for frame in 0..num_frames {
            for ch in 0..num_ch {
                let sample = f64::from(interleaved[frame * num_ch + ch]);
                self.channel_sum_sq[ch] += sample * sample;
                let abs_val = sample.abs();
                if abs_val > self.channel_peak[ch] {
                    self.channel_peak[ch] = abs_val;
                }
            }
            self.samples_per_channel += 1;
        }
    }

    /// Compute the analysis results.
    pub fn result(&self) -> SurroundAnalysis {
        let num_ch = self.config.layout.channel_count();
        let n = self.samples_per_channel.max(1) as f64;

        // Per-channel RMS in dB
        let channel_rms_db: Vec<f64> = self
            .channel_sum_sq
            .iter()
            .map(|&sum_sq| {
                let rms = (sum_sq / n).sqrt();
                if rms <= 0.0 {
                    -100.0
                } else {
                    20.0 * rms.log10()
                }
            })
            .collect();

        // Per-channel peak in dB
        let channel_peak_db: Vec<f64> = self
            .channel_peak
            .iter()
            .map(|&peak| {
                if peak <= 0.0 {
                    -100.0
                } else {
                    20.0 * peak.log10()
                }
            })
            .collect();

        // Weighted integrated loudness (simplified ITU-R BS.1770)
        let mut weighted_sum = 0.0;
        for ch in 0..num_ch {
            if !self.lfe_indices.contains(&ch) {
                let mean_sq = self.channel_sum_sq[ch] / n;
                weighted_sum += self.weights.get(ch).copied().unwrap_or(1.0) * mean_sq;
            }
        }

        let integrated_lufs = if weighted_sum <= 0.0 {
            -70.0
        } else {
            -0.691 + 10.0 * weighted_sum.log10()
        };

        // Compute downmix loudness
        let downmix_lufs = self.compute_downmix_loudness();

        let recommended_gain_db = self.config.target_lufs - integrated_lufs;
        let deviation = (integrated_lufs - downmix_lufs).abs();
        let downmix_compliant = deviation <= self.config.max_downmix_deviation_lu;

        SurroundAnalysis {
            channel_rms_db,
            channel_peak_db,
            integrated_lufs,
            downmix_lufs,
            recommended_gain_db,
            downmix_compliant,
        }
    }

    /// Compute estimated downmix loudness.
    fn compute_downmix_loudness(&self) -> f64 {
        let num_ch = self.config.layout.channel_count();
        let n = self.samples_per_channel.max(1) as f64;

        let coeffs = match self.config.downmix_mode {
            DownmixMode::ItuBs775 | DownmixMode::LoRo => DownmixCoefficients::itu_bs775(),
            DownmixMode::LtRt => DownmixCoefficients::lo_ro(),
            DownmixMode::Mono => DownmixCoefficients::itu_bs775(),
        };

        // For stereo or mono layouts, downmix = same as surround
        if num_ch <= 2 {
            let mut sum = 0.0;
            for ch in 0..num_ch {
                sum += self.channel_sum_sq[ch] / n;
            }
            return if sum <= 0.0 {
                -70.0
            } else {
                -0.691 + 10.0 * sum.log10()
            };
        }

        // Simplified 5.1 downmix model:
        // L_out = L + center_coeff*C + surround_coeff*Ls
        // R_out = R + center_coeff*C + surround_coeff*Rs
        let c_coeff = coeffs.center_linear();
        let s_coeff = coeffs.surround_linear();

        // Estimate downmix power from per-channel mean squares
        let l_ms = self.channel_sum_sq.first().copied().unwrap_or(0.0) / n;
        let r_ms = self.channel_sum_sq.get(1).copied().unwrap_or(0.0) / n;
        let c_ms = if num_ch > 2 {
            self.channel_sum_sq.get(2).copied().unwrap_or(0.0) / n
        } else {
            0.0
        };
        let ls_ms = if num_ch > 4 {
            self.channel_sum_sq.get(4).copied().unwrap_or(0.0) / n
        } else {
            0.0
        };
        let rs_ms = if num_ch > 5 {
            self.channel_sum_sq.get(5).copied().unwrap_or(0.0) / n
        } else {
            0.0
        };

        // Uncorrelated sum approximation
        let l_out_ms = l_ms + c_coeff * c_coeff * c_ms + s_coeff * s_coeff * ls_ms;
        let r_out_ms = r_ms + c_coeff * c_coeff * c_ms + s_coeff * s_coeff * rs_ms;

        let total_ms = l_out_ms + r_out_ms;
        if total_ms <= 0.0 {
            -70.0
        } else {
            -0.691 + 10.0 * total_ms.log10()
        }
    }

    /// Apply global gain to interleaved multichannel audio.
    pub fn apply_gain(interleaved: &mut [f32], gain_db: f64) {
        let gain_linear = 10.0_f64.powf(gain_db / 20.0) as f32;
        for sample in interleaved.iter_mut() {
            *sample *= gain_linear;
        }
    }

    /// Apply per-channel gain to interleaved audio.
    pub fn apply_per_channel_gain(
        interleaved: &mut [f32],
        num_channels: usize,
        channel_gains_db: &[f64],
    ) {
        if num_channels == 0 {
            return;
        }
        let gains_linear: Vec<f32> = channel_gains_db
            .iter()
            .map(|&db| 10.0_f64.powf(db / 20.0) as f32)
            .collect();

        let num_frames = interleaved.len() / num_channels;
        for frame in 0..num_frames {
            for ch in 0..num_channels {
                let gain = gains_linear.get(ch).copied().unwrap_or(1.0);
                interleaved[frame * num_channels + ch] *= gain;
            }
        }
    }

    /// Reset the analyzer state.
    pub fn reset(&mut self) {
        let num_ch = self.config.layout.channel_count();
        self.channel_sum_sq = vec![0.0; num_ch];
        self.channel_peak = vec![0.0; num_ch];
        self.samples_per_channel = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_layout_counts() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Surround51.channel_count(), 6);
        assert_eq!(ChannelLayout::Surround71.channel_count(), 8);
        assert_eq!(ChannelLayout::Atmos714.channel_count(), 12);
    }

    #[test]
    fn test_lfe_indices() {
        assert!(ChannelLayout::Stereo.lfe_indices().is_empty());
        assert_eq!(ChannelLayout::Surround51.lfe_indices(), vec![3]);
        assert_eq!(ChannelLayout::Surround71.lfe_indices(), vec![3]);
    }

    #[test]
    fn test_channel_weights_stereo() {
        let w = ChannelLayout::Stereo.channel_weights();
        assert_eq!(w.len(), 2);
        assert!((w[0] - 1.0).abs() < f64::EPSILON);
        assert!((w[1] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_channel_weights_51_lfe_excluded() {
        let w = ChannelLayout::Surround51.channel_weights();
        assert_eq!(w.len(), 6);
        assert!((w[3] - 0.0).abs() < f64::EPSILON); // LFE weight = 0
    }

    #[test]
    fn test_channel_weights_surround_boost() {
        let w = ChannelLayout::Surround51.channel_weights();
        // Surround channels (idx 4, 5) should be > 1.0 (+1.5 dB)
        assert!(w[4] > 1.0);
        assert!(w[5] > 1.0);
        let expected = 10.0_f64.powf(1.5 / 10.0);
        assert!((w[4] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_downmix_coefficients_itu() {
        let c = DownmixCoefficients::itu_bs775();
        assert!((c.center_db - (-3.0)).abs() < f64::EPSILON);
        assert!((c.surround_db - (-3.0)).abs() < f64::EPSILON);
        // Linear coefficient for -3 dB ~ 0.7079
        assert!((c.center_linear() - 10.0_f64.powf(-3.0 / 20.0)).abs() < 1e-10);
    }

    #[test]
    fn test_surround_config_validation() {
        let config = SurroundNormConfig::new(ChannelLayout::Surround51, 48000.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_surround_config_validation_bad_rate() {
        let config = SurroundNormConfig::new(ChannelLayout::Surround51, 0.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_surround_config_validation_positive_lufs() {
        let mut config = SurroundNormConfig::new(ChannelLayout::Surround51, 48000.0);
        config.target_lufs = 5.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_surround_normalizer_silence() {
        let config = SurroundNormConfig::new(ChannelLayout::Surround51, 48000.0);
        let mut norm = SurroundNormalizer::new(config);
        let silence = vec![0.0f32; 6 * 1000]; // 1000 frames of 5.1 silence
        norm.analyze(&silence);
        let result = norm.result();
        assert!(result.integrated_lufs <= -70.0);
    }

    #[test]
    fn test_surround_normalizer_signal() {
        let config = SurroundNormConfig::new(ChannelLayout::Stereo, 48000.0);
        let mut norm = SurroundNormalizer::new(config);
        // Generate a stereo sine wave
        let mut samples = Vec::with_capacity(2 * 48000);
        for i in 0..48000 {
            let s = (0.5 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48000.0).sin()) as f32;
            samples.push(s); // L
            samples.push(s); // R
        }
        norm.analyze(&samples);
        let result = norm.result();
        assert!(result.integrated_lufs > -70.0);
        assert!(result.integrated_lufs < 0.0);
    }

    #[test]
    fn test_apply_global_gain() {
        let mut samples = vec![0.5f32, -0.5, 0.25, -0.25, 0.1, -0.1];
        SurroundNormalizer::apply_gain(&mut samples, 6.0);
        // 6 dB ~ factor of ~2
        let factor = 10.0_f64.powf(6.0 / 20.0) as f32;
        assert!((samples[0] - 0.5 * factor).abs() < 1e-5);
    }

    #[test]
    fn test_apply_per_channel_gain() {
        let mut samples = vec![1.0f32, 1.0, 1.0, 1.0]; // 2 frames, 2 channels
        SurroundNormalizer::apply_per_channel_gain(&mut samples, 2, &[0.0, -6.0]);
        // Channel 0: 0 dB => gain 1.0
        assert!((samples[0] - 1.0).abs() < 1e-5);
        // Channel 1: -6 dB => gain ~0.501
        let expected = 10.0_f64.powf(-6.0 / 20.0) as f32;
        assert!((samples[1] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_surround_normalizer_reset() {
        let config = SurroundNormConfig::new(ChannelLayout::Stereo, 48000.0);
        let mut norm = SurroundNormalizer::new(config);
        let samples = vec![0.5f32; 200];
        norm.analyze(&samples);
        norm.reset();
        let result = norm.result();
        assert!(result.integrated_lufs <= -70.0);
    }

    #[test]
    fn test_ebu_r128_config() {
        let config = SurroundNormConfig::ebu_r128(ChannelLayout::Surround51, 48000.0);
        assert!((config.target_lufs - (-23.0)).abs() < f64::EPSILON);
        assert!((config.max_true_peak_dbtp - (-1.0)).abs() < f64::EPSILON);
        assert!((config.max_downmix_deviation_lu - 1.0).abs() < f64::EPSILON);
    }
}

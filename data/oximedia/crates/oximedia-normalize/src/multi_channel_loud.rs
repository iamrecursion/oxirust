#![allow(dead_code)]
//! Multi-channel loudness computation following ITU-R BS.1770 channel weighting.

/// Per-channel weighting for loudness computation (ITU-R BS.1770-4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChannelWeight {
    /// Front left / front right — weight 1.0.
    FrontLR,
    /// Front centre — weight 1.0.
    FrontC,
    /// Left surround / right surround — weight 1.41 (≈ +3 dB).
    SurroundLR,
    /// Rear left / rear right — weight 1.41.
    RearLR,
    /// Low frequency effects (LFE) — excluded (weight 0.0).
    Lfe,
    /// Top / height channels — weight 1.0 (Atmos extension).
    Height,
}

impl ChannelWeight {
    /// Linear weighting factor (not in dB).
    pub fn weight(&self) -> f64 {
        match self {
            Self::FrontLR | Self::FrontC | Self::Height => 1.0,
            Self::SurroundLR | Self::RearLR => 1.41,
            Self::Lfe => 0.0,
        }
    }

    /// Whether this channel is included in loudness computation.
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Lfe)
    }
}

/// Configuration for a multi-channel loudness computation session.
#[derive(Debug, Clone)]
pub struct MultiChannelConfig {
    /// Ordered list of channel weights (one per channel).
    pub channel_weights: Vec<ChannelWeight>,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Integration gate length in seconds (usually 0.4 s for momentary, 3.0 s for short-term).
    pub gate_seconds: f64,
}

impl MultiChannelConfig {
    /// Stereo configuration (L, R).
    pub fn stereo(sample_rate: f64) -> Self {
        Self {
            channel_weights: vec![ChannelWeight::FrontLR, ChannelWeight::FrontLR],
            sample_rate,
            gate_seconds: 0.4,
        }
    }

    /// 5.1 surround configuration (L, R, C, LFE, Ls, Rs).
    pub fn surround_5_1(sample_rate: f64) -> Self {
        Self {
            channel_weights: vec![
                ChannelWeight::FrontLR,
                ChannelWeight::FrontLR,
                ChannelWeight::FrontC,
                ChannelWeight::Lfe,
                ChannelWeight::SurroundLR,
                ChannelWeight::SurroundLR,
            ],
            sample_rate,
            gate_seconds: 0.4,
        }
    }

    /// 7.1 surround configuration (L, R, C, LFE, Ls, Rs, Lrs, Rrs).
    pub fn surround_7_1(sample_rate: f64) -> Self {
        Self {
            channel_weights: vec![
                ChannelWeight::FrontLR,
                ChannelWeight::FrontLR,
                ChannelWeight::FrontC,
                ChannelWeight::Lfe,
                ChannelWeight::SurroundLR,
                ChannelWeight::SurroundLR,
                ChannelWeight::RearLR,
                ChannelWeight::RearLR,
            ],
            sample_rate,
            gate_seconds: 0.4,
        }
    }

    /// Number of channels (including LFE).
    pub fn channel_count(&self) -> usize {
        self.channel_weights.len()
    }

    /// Whether this is a surround configuration (more than 2 channels, excl. LFE).
    pub fn is_surround(&self) -> bool {
        let active = self
            .channel_weights
            .iter()
            .filter(|w| w.is_active())
            .count();
        active > 2
    }

    /// Number of active (non-LFE) channels.
    pub fn active_channel_count(&self) -> usize {
        self.channel_weights
            .iter()
            .filter(|w| w.is_active())
            .count()
    }
}

impl Default for MultiChannelConfig {
    fn default() -> Self {
        Self::stereo(48000.0)
    }
}

/// Multi-channel loudness measurement.
#[derive(Debug, Clone)]
pub struct MultiChannelLoudness {
    config: MultiChannelConfig,
    /// Per-channel mean square (accumulated energy).
    channel_ms: Vec<f64>,
    /// Number of samples accumulated per channel.
    sample_count: u64,
}

impl MultiChannelLoudness {
    /// Create a new loudness accumulator for the given config.
    pub fn new(config: MultiChannelConfig) -> Self {
        let n = config.channel_count();
        Self {
            config,
            channel_ms: vec![0.0; n],
            sample_count: 0,
        }
    }

    /// Feed an interleaved multi-channel frame.
    ///
    /// `samples` length must be a multiple of `channel_count`.
    pub fn push_interleaved(&mut self, samples: &[f32]) {
        let n_ch = self.config.channel_count();
        if n_ch == 0 || samples.is_empty() {
            return;
        }
        let frames = samples.len() / n_ch;
        for frame in 0..frames {
            for ch in 0..n_ch {
                let s = f64::from(samples[frame * n_ch + ch]);
                self.channel_ms[ch] += s * s;
            }
        }
        self.sample_count += frames as u64;
    }

    /// Feed planar (non-interleaved) channels. Each slice must have the same length.
    pub fn push_planar(&mut self, channels: &[&[f32]]) {
        let n_ch = self.config.channel_count();
        if channels.len() < n_ch || channels.is_empty() {
            return;
        }
        let frames = channels[0].len();
        for frame in 0..frames {
            for ch in 0..n_ch {
                let s = f64::from(channels[ch][frame]);
                self.channel_ms[ch] += s * s;
            }
        }
        self.sample_count += frames as u64;
    }

    /// Compute integrated loudness (LUFS) via ITU-R BS.1770 weighted sum.
    /// Returns `None` if no samples have been accumulated.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_integrated(&self) -> Option<f64> {
        if self.sample_count == 0 {
            return None;
        }
        let weighted = self.sum_weighted();
        if weighted <= 0.0 {
            return Some(-f64::INFINITY);
        }
        Some(-0.691 + 10.0 * weighted.log10())
    }

    /// Weighted sum of per-channel mean squares (ITU-R BS.1770 equation).
    #[allow(clippy::cast_precision_loss)]
    pub fn sum_weighted(&self) -> f64 {
        let n = self.sample_count as f64;
        if n == 0.0 {
            return 0.0;
        }
        self.channel_ms
            .iter()
            .zip(self.config.channel_weights.iter())
            .map(|(ms, w)| w.weight() * ms / n)
            .sum()
    }

    /// Per-channel RMS (not weighted).
    #[allow(clippy::cast_precision_loss)]
    pub fn channel_rms(&self) -> Vec<f64> {
        let n = self.sample_count as f64;
        if n == 0.0 {
            return vec![0.0; self.channel_ms.len()];
        }
        self.channel_ms.iter().map(|ms| (ms / n).sqrt()).collect()
    }

    /// Reset accumulated state.
    pub fn reset(&mut self) {
        for ms in &mut self.channel_ms {
            *ms = 0.0;
        }
        self.sample_count = 0;
    }

    /// Whether any samples have been accumulated.
    pub fn has_data(&self) -> bool {
        self.sample_count > 0
    }

    /// Reference to the active config.
    pub fn config(&self) -> &MultiChannelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_weight_front_lr() {
        assert!((ChannelWeight::FrontLR.weight() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_channel_weight_surround() {
        assert!((ChannelWeight::SurroundLR.weight() - 1.41).abs() < 1e-9);
    }

    #[test]
    fn test_channel_weight_lfe_zero() {
        assert!((ChannelWeight::Lfe.weight() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_channel_weight_lfe_not_active() {
        assert!(!ChannelWeight::Lfe.is_active());
        assert!(ChannelWeight::FrontC.is_active());
        assert!(ChannelWeight::SurroundLR.is_active());
    }

    #[test]
    fn test_config_stereo_channel_count() {
        let cfg = MultiChannelConfig::stereo(48000.0);
        assert_eq!(cfg.channel_count(), 2);
        assert!(!cfg.is_surround());
    }

    #[test]
    fn test_config_5_1_channel_count() {
        let cfg = MultiChannelConfig::surround_5_1(48000.0);
        assert_eq!(cfg.channel_count(), 6);
        assert!(cfg.is_surround());
    }

    #[test]
    fn test_config_7_1_channel_count() {
        let cfg = MultiChannelConfig::surround_7_1(48000.0);
        assert_eq!(cfg.channel_count(), 8);
        assert!(cfg.is_surround());
    }

    #[test]
    fn test_config_active_channel_count_5_1() {
        // 5.1 has 1 LFE → 5 active
        let cfg = MultiChannelConfig::surround_5_1(48000.0);
        assert_eq!(cfg.active_channel_count(), 5);
    }

    #[test]
    fn test_no_samples_returns_none() {
        let loud = MultiChannelLoudness::new(MultiChannelConfig::stereo(48000.0));
        assert!(loud.compute_integrated().is_none());
        assert!(!loud.has_data());
    }

    #[test]
    fn test_push_interleaved_basic() {
        let mut loud = MultiChannelLoudness::new(MultiChannelConfig::stereo(48000.0));
        let samples = vec![0.1f32; 200]; // 100 stereo frames, amplitude 0.1
        loud.push_interleaved(&samples);
        assert!(loud.has_data());
        let integrated = loud.compute_integrated().expect("should succeed in test");
        assert!(integrated < 0.0); // negative LUFS
    }

    #[test]
    fn test_push_planar_basic() {
        let mut loud = MultiChannelLoudness::new(MultiChannelConfig::stereo(48000.0));
        let ch: Vec<f32> = vec![0.2f32; 100];
        loud.push_planar(&[&ch, &ch]);
        assert!(loud.has_data());
    }

    #[test]
    fn test_sum_weighted_zero_without_samples() {
        let loud = MultiChannelLoudness::new(MultiChannelConfig::stereo(48000.0));
        assert!((loud.sum_weighted() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_channel_rms() {
        let mut loud = MultiChannelLoudness::new(MultiChannelConfig::stereo(48000.0));
        let samples = vec![0.5f32; 200]; // constant 0.5
        loud.push_interleaved(&samples);
        let rms = loud.channel_rms();
        assert_eq!(rms.len(), 2);
        for r in rms {
            assert!((r - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reset() {
        let mut loud = MultiChannelLoudness::new(MultiChannelConfig::stereo(48000.0));
        let samples = vec![0.1f32; 200];
        loud.push_interleaved(&samples);
        loud.reset();
        assert!(!loud.has_data());
        assert!(loud.compute_integrated().is_none());
    }

    #[test]
    fn test_surround_higher_than_stereo_for_same_signal() {
        // Surround channels get +1.41 weight → higher integrated loudness for same amplitude
        let mut stereo = MultiChannelLoudness::new(MultiChannelConfig::stereo(48000.0));
        let mut surround_5_1 = MultiChannelLoudness::new(MultiChannelConfig::surround_5_1(48000.0));

        let s_stereo = vec![0.1f32; 200];
        let s_5_1 = vec![0.1f32; 600]; // 6 channels × 100 frames

        stereo.push_interleaved(&s_stereo);
        surround_5_1.push_interleaved(&s_5_1);

        let l_stereo = stereo.compute_integrated().expect("should succeed in test");
        let l_5_1 = surround_5_1
            .compute_integrated()
            .expect("should succeed in test");

        // 5.1 has surround channels with weight 1.41, so weighted sum should be higher.
        assert!(l_5_1 > l_stereo);
    }
}

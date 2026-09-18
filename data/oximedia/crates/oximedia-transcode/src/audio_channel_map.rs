//! Audio channel layout mapping and transcode parameters.
//!
//! Provides channel layout enumeration, passthrough detection, gain computation,
//! and parameter validation for audio transcoding stages.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Standard audio channel layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioLayout {
    /// Single mono channel.
    Mono,
    /// Two stereo channels (L, R).
    Stereo,
    /// 2.1 channels (L, R, LFE).
    TwoPointOne,
    /// 4.0 surround (L, R, Ls, Rs).
    Quad,
    /// 5.0 surround (L, R, C, Ls, Rs).
    FivePointZero,
    /// 5.1 surround (L, R, C, LFE, Ls, Rs).
    FivePointOne,
    /// 7.1 surround (L, R, C, LFE, Lss, Rss, Lrs, Rrs).
    SevenPointOne,
}

impl AudioLayout {
    /// Number of channels in this layout.
    #[must_use]
    pub fn channel_count(self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::TwoPointOne => 3,
            Self::Quad => 4,
            Self::FivePointZero => 5,
            Self::FivePointOne => 6,
            Self::SevenPointOne => 8,
        }
    }

    /// Returns `true` if the layout has a dedicated LFE (sub-woofer) channel.
    #[must_use]
    pub fn has_lfe(self) -> bool {
        matches!(
            self,
            Self::TwoPointOne | Self::FivePointOne | Self::SevenPointOne
        )
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::Stereo => "stereo",
            Self::TwoPointOne => "2.1",
            Self::Quad => "4.0",
            Self::FivePointZero => "5.0",
            Self::FivePointOne => "5.1",
            Self::SevenPointOne => "7.1",
        }
    }
}

/// Parameters that govern how a single audio stream is transcoded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscodeParams {
    /// Input channel layout.
    pub input_layout: AudioLayout,
    /// Desired output channel layout.
    pub output_layout: AudioLayout,
    /// Input sample rate in Hz.
    pub input_sample_rate: u32,
    /// Desired output sample rate in Hz.
    pub output_sample_rate: u32,
    /// Target audio bitrate in bits per second.
    pub target_bitrate_bps: u32,
    /// Linear gain to apply during transcode (1.0 = no change).
    pub gain_linear: f32,
    /// Whether to normalise loudness to EBU R128.
    pub normalise_loudness: bool,
}

impl Default for AudioTranscodeParams {
    fn default() -> Self {
        Self {
            input_layout: AudioLayout::Stereo,
            output_layout: AudioLayout::Stereo,
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            target_bitrate_bps: 128_000,
            gain_linear: 1.0,
            normalise_loudness: false,
        }
    }
}

impl AudioTranscodeParams {
    /// Create default params for a stereo stream.
    #[must_use]
    pub fn stereo() -> Self {
        Self::default()
    }

    /// Create params for a mono downmix.
    #[must_use]
    pub fn mono_downmix() -> Self {
        Self {
            output_layout: AudioLayout::Mono,
            ..Self::default()
        }
    }

    /// Returns `true` when the output layout and sample rate match the input
    /// and no gain or loudness normalisation is applied — i.e. pure passthrough.
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        self.input_layout == self.output_layout
            && self.input_sample_rate == self.output_sample_rate
            && (self.gain_linear - 1.0).abs() < f32::EPSILON
            && !self.normalise_loudness
    }

    /// Returns `true` if the configuration is logically valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.input_sample_rate > 0
            && self.output_sample_rate > 0
            && self.target_bitrate_bps > 0
            && self.gain_linear >= 0.0
    }
}

/// Maps individual input channels to output channels and applies per-channel gain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChannelMap {
    input_layout: AudioLayout,
    output_layout: AudioLayout,
    /// Routing: `output_channel_index` -> (`input_channel_index`, `gain_factor`).
    routes: Vec<(usize, f32)>,
}

impl AudioChannelMap {
    /// Create an identity map (each input channel routes to the same output channel).
    #[must_use]
    pub fn identity(layout: AudioLayout) -> Self {
        let n = layout.channel_count() as usize;
        let routes = (0..n).map(|i| (i, 1.0_f32)).collect();
        Self {
            input_layout: layout,
            output_layout: layout,
            routes,
        }
    }

    /// Create a downmix map from stereo to mono (equal-power mix of L+R).
    #[must_use]
    pub fn stereo_to_mono() -> Self {
        Self {
            input_layout: AudioLayout::Stereo,
            output_layout: AudioLayout::Mono,
            // Mono = 0.707 * L + 0.707 * R  (from channel 0 and channel 1)
            routes: vec![(0, 0.707_107), (1, 0.707_107)],
        }
    }

    /// Compute the gain in dB for a given output channel index.
    ///
    /// Returns `None` if the output channel index is out of range.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute_gain_db(&self, output_channel: usize) -> Option<f64> {
        let (_, gain_linear) = self.routes.get(output_channel)?;
        if *gain_linear <= 0.0 {
            return Some(f64::NEG_INFINITY);
        }
        // 20 * log10(linear_gain)
        let db = 20.0 * f64::from(*gain_linear).log10();
        Some(db)
    }

    /// Validate that the map is consistent with the declared layouts.
    ///
    /// For downmix maps (e.g. stereo→mono), the number of routes equals the
    /// number of *input* channels being mixed, which may exceed the output
    /// channel count.  We therefore only require that the route count is at
    /// least the output channel count.
    pub fn validate_params(&self) -> Result<(), String> {
        let expected_out = self.output_layout.channel_count() as usize;
        if self.routes.len() < expected_out {
            return Err(format!(
                "route count {} is less than output channel count {}",
                self.routes.len(),
                expected_out
            ));
        }
        let max_in = self.input_layout.channel_count() as usize;
        for (ch_idx, _) in &self.routes {
            if *ch_idx >= max_in {
                return Err(format!(
                    "input channel index {ch_idx} out of range (input has {max_in} channels)"
                ));
            }
        }
        Ok(())
    }

    /// Number of output channels.
    #[must_use]
    pub fn output_channel_count(&self) -> usize {
        self.output_layout.channel_count() as usize
    }

    /// Apply the channel map to an input sample buffer.
    ///
    /// `input` is interleaved samples (f32), `output` must be pre-allocated
    /// with `frame_size * output_channels` entries.
    pub fn apply(&self, input: &[f32], output: &mut [f32], frame_size: usize) {
        let in_ch = self.input_layout.channel_count() as usize;
        let out_ch = self.output_layout.channel_count() as usize;
        for frame in 0..frame_size {
            for (out_idx, (in_idx, gain)) in self.routes.iter().enumerate() {
                let in_pos = frame * in_ch + in_idx;
                let out_pos = frame * out_ch + out_idx;
                if in_pos < input.len() && out_pos < output.len() {
                    output[out_pos] += input[in_pos] * gain;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_channel_count() {
        assert_eq!(AudioLayout::Mono.channel_count(), 1);
    }

    #[test]
    fn test_stereo_channel_count() {
        assert_eq!(AudioLayout::Stereo.channel_count(), 2);
    }

    #[test]
    fn test_five_one_channel_count() {
        assert_eq!(AudioLayout::FivePointOne.channel_count(), 6);
    }

    #[test]
    fn test_seven_one_channel_count() {
        assert_eq!(AudioLayout::SevenPointOne.channel_count(), 8);
    }

    #[test]
    fn test_has_lfe() {
        assert!(AudioLayout::FivePointOne.has_lfe());
        assert!(AudioLayout::TwoPointOne.has_lfe());
        assert!(!AudioLayout::Stereo.has_lfe());
        assert!(!AudioLayout::Quad.has_lfe());
    }

    #[test]
    fn test_labels_non_empty() {
        let layouts = [
            AudioLayout::Mono,
            AudioLayout::Stereo,
            AudioLayout::TwoPointOne,
            AudioLayout::Quad,
            AudioLayout::FivePointZero,
            AudioLayout::FivePointOne,
            AudioLayout::SevenPointOne,
        ];
        for l in layouts {
            assert!(!l.label().is_empty());
        }
    }

    #[test]
    fn test_params_is_passthrough_default() {
        let p = AudioTranscodeParams::default();
        assert!(p.is_passthrough());
    }

    #[test]
    fn test_params_not_passthrough_different_layout() {
        let p = AudioTranscodeParams {
            output_layout: AudioLayout::Mono,
            ..AudioTranscodeParams::default()
        };
        assert!(!p.is_passthrough());
    }

    #[test]
    fn test_params_not_passthrough_with_gain() {
        let p = AudioTranscodeParams {
            gain_linear: 1.5,
            ..AudioTranscodeParams::default()
        };
        assert!(!p.is_passthrough());
    }

    #[test]
    fn test_params_is_valid_default() {
        assert!(AudioTranscodeParams::default().is_valid());
    }

    #[test]
    fn test_params_invalid_zero_sample_rate() {
        let p = AudioTranscodeParams {
            input_sample_rate: 0,
            ..AudioTranscodeParams::default()
        };
        assert!(!p.is_valid());
    }

    #[test]
    fn test_identity_map_validate() {
        let m = AudioChannelMap::identity(AudioLayout::Stereo);
        assert!(m.validate_params().is_ok());
    }

    #[test]
    fn test_stereo_to_mono_validate() {
        let m = AudioChannelMap::stereo_to_mono();
        assert!(m.validate_params().is_ok());
    }

    #[test]
    fn test_compute_gain_db_unity() {
        let m = AudioChannelMap::identity(AudioLayout::Stereo);
        let db = m.compute_gain_db(0).expect("should succeed in test");
        assert!(
            (db - 0.0).abs() < 1e-9,
            "unity gain should be 0 dB, got {db}"
        );
    }

    #[test]
    fn test_compute_gain_db_stereo_to_mono() {
        let m = AudioChannelMap::stereo_to_mono();
        let db = m.compute_gain_db(0).expect("should succeed in test");
        // 20*log10(0.707) ≈ -3.01 dB
        assert!((db + 3.01).abs() < 0.1, "expected ~-3 dB, got {db}");
    }

    #[test]
    fn test_compute_gain_db_out_of_range() {
        let m = AudioChannelMap::identity(AudioLayout::Mono);
        assert!(m.compute_gain_db(5).is_none());
    }

    #[test]
    fn test_apply_identity() {
        let m = AudioChannelMap::identity(AudioLayout::Stereo);
        let input = vec![0.5_f32, 0.8, 0.3, 0.1];
        let mut output = vec![0.0_f32; 4];
        m.apply(&input, &mut output, 2);
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_output_channel_count() {
        let m = AudioChannelMap::stereo_to_mono();
        assert_eq!(m.output_channel_count(), 1);
    }
}

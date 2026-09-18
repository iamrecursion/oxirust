// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Encoder presets and settings for Theora.
//!
//! Provides pre-configured encoder settings optimized for different
//! use cases: speed, quality, file size, etc.

use crate::theora::rate_ctrl::{RateControlMode, RateController};

/// Encoder preset.
///
/// Pre-configured encoder settings for different quality/speed tradeoffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderPreset {
    /// Fastest encoding, lowest quality.
    Ultrafast,
    /// Very fast encoding.
    Superfast,
    /// Fast encoding.
    Fast,
    /// Moderate encoding speed.
    Medium,
    /// Slower encoding, better quality.
    Slow,
    /// Very slow encoding, high quality.
    Veryslow,
    /// Extremely slow, maximum quality.
    Placebo,
}

impl EncoderPreset {
    /// Get encoding speed (0-10, higher = faster).
    #[must_use]
    pub const fn speed(&self) -> u8 {
        match self {
            Self::Ultrafast => 10,
            Self::Superfast => 8,
            Self::Fast => 6,
            Self::Medium => 5,
            Self::Slow => 3,
            Self::Veryslow => 1,
            Self::Placebo => 0,
        }
    }

    /// Get motion estimation range.
    #[must_use]
    pub const fn me_range(&self) -> i16 {
        match self {
            Self::Ultrafast => 4,
            Self::Superfast => 8,
            Self::Fast => 12,
            Self::Medium => 16,
            Self::Slow => 24,
            Self::Veryslow => 32,
            Self::Placebo => 48,
        }
    }

    /// Check if subpixel motion estimation is enabled.
    #[must_use]
    pub const fn use_subpel_me(&self) -> bool {
        !matches!(self, Self::Ultrafast | Self::Superfast)
    }

    /// Check if rate-distortion optimization is enabled.
    #[must_use]
    pub const fn use_rdo(&self) -> bool {
        !matches!(self, Self::Ultrafast | Self::Superfast | Self::Fast)
    }

    /// Get RD lambda for this preset.
    #[must_use]
    pub fn rd_lambda(&self, qp: u8) -> f32 {
        let base_lambda = 0.85 * f32::from(qp).powf(1.2);
        match self {
            Self::Ultrafast => base_lambda * 0.5,
            Self::Superfast => base_lambda * 0.7,
            Self::Fast => base_lambda * 0.85,
            Self::Medium => base_lambda,
            Self::Slow => base_lambda * 1.15,
            Self::Veryslow => base_lambda * 1.3,
            Self::Placebo => base_lambda * 1.5,
        }
    }

    /// Check if trellis quantization is enabled.
    #[must_use]
    pub const fn use_trellis(&self) -> bool {
        matches!(self, Self::Slow | Self::Veryslow | Self::Placebo)
    }

    /// Get dead-zone size.
    #[must_use]
    pub fn dead_zone(&self) -> f32 {
        match self {
            Self::Ultrafast => 0.6,
            Self::Superfast => 0.5,
            Self::Fast => 0.4,
            Self::Medium => 0.3,
            Self::Slow => 0.2,
            Self::Veryslow => 0.1,
            Self::Placebo => 0.05,
        }
    }
}

impl Default for EncoderPreset {
    fn default() -> Self {
        Self::Medium
    }
}

/// Encoding tune.
///
/// Optimizes encoder for specific content types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingTune {
    /// General purpose encoding.
    Default,
    /// Optimized for film content.
    Film,
    /// Optimized for animation.
    Animation,
    /// Optimized for screen content.
    Screen,
    /// Optimized for grain preservation.
    Grain,
    /// Fast decode optimization.
    FastDecode,
}

impl EncodingTune {
    /// Get quality bias for intra prediction.
    #[must_use]
    pub fn intra_bias(&self) -> f32 {
        match self {
            Self::Default => 1.0,
            Self::Film => 1.1,
            Self::Animation => 0.9,
            Self::Screen => 0.8,
            Self::Grain => 1.2,
            Self::FastDecode => 1.3,
        }
    }

    /// Get perceptual optimization strength.
    #[must_use]
    pub fn perceptual_strength(&self) -> f32 {
        match self {
            Self::Default => 0.5,
            Self::Film => 0.7,
            Self::Animation => 0.3,
            Self::Screen => 0.2,
            Self::Grain => 0.1,
            Self::FastDecode => 0.4,
        }
    }

    /// Get adaptive quantization strength.
    #[must_use]
    pub fn aq_strength(&self) -> f32 {
        match self {
            Self::Default => 1.0,
            Self::Film => 1.2,
            Self::Animation => 0.6,
            Self::Screen => 0.4,
            Self::Grain => 0.3,
            Self::FastDecode => 0.8,
        }
    }

    /// Check if deblocking filter should be enabled.
    #[must_use]
    pub const fn enable_deblock(&self) -> bool {
        !matches!(self, Self::Grain)
    }
}

impl Default for EncodingTune {
    fn default() -> Self {
        Self::Default
    }
}

/// Complete encoder settings.
#[derive(Debug, Clone)]
pub struct EncoderSettings {
    /// Encoder preset.
    pub preset: EncoderPreset,
    /// Encoding tune.
    pub tune: EncodingTune,
    /// Rate control mode.
    pub rc_mode: RateControlMode,
    /// Target quality (0-63).
    pub quality: u8,
    /// Target bitrate (bits/sec).
    pub bitrate: u64,
    /// Keyframe interval.
    pub keyint: u32,
    /// Minimum keyframe interval.
    pub min_keyint: u32,
    /// Scene change threshold.
    pub scenecut: f32,
    /// Number of threads.
    pub threads: usize,
    /// Enable adaptive quantization.
    pub aq_mode: AdaptiveQuantMode,
    /// Psychovisual optimization strength.
    pub psy_rd: f32,
    /// Enable loop filter.
    pub deblock: bool,
    /// Loop filter strength.
    pub deblock_strength: u8,
}

impl EncoderSettings {
    /// Create new encoder settings from preset.
    #[must_use]
    pub fn from_preset(preset: EncoderPreset) -> Self {
        Self {
            preset,
            tune: EncodingTune::Default,
            rc_mode: RateControlMode::ConstantQuality,
            quality: 30,
            bitrate: 2_000_000,
            keyint: 250,
            min_keyint: 25,
            scenecut: 40.0,
            threads: 0,
            aq_mode: AdaptiveQuantMode::Variance,
            psy_rd: 1.0,
            deblock: true,
            deblock_strength: 30,
        }
    }

    /// Apply tune to settings.
    #[must_use]
    pub fn with_tune(mut self, tune: EncodingTune) -> Self {
        self.tune = tune;
        self.deblock = tune.enable_deblock();
        self.psy_rd *= tune.perceptual_strength();
        self
    }

    /// Set rate control mode.
    #[must_use]
    pub const fn with_rate_control(mut self, mode: RateControlMode) -> Self {
        self.rc_mode = mode;
        self
    }

    /// Set quality.
    #[must_use]
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality.min(63);
        self
    }

    /// Set target bitrate.
    #[must_use]
    pub const fn with_bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Set keyframe interval.
    #[must_use]
    pub const fn with_keyint(mut self, keyint: u32) -> Self {
        self.keyint = keyint;
        self
    }

    /// Enable adaptive quantization.
    #[must_use]
    pub const fn with_aq(mut self, mode: AdaptiveQuantMode) -> Self {
        self.aq_mode = mode;
        self
    }

    /// Create rate controller from settings.
    #[must_use]
    pub fn create_rate_controller(&self, framerate: f64) -> RateController {
        RateController::new(self.rc_mode, self.bitrate, self.quality, framerate)
    }
}

impl Default for EncoderSettings {
    fn default() -> Self {
        Self::from_preset(EncoderPreset::Medium)
    }
}

/// Adaptive quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveQuantMode {
    /// Disabled.
    None,
    /// Variance-based AQ.
    Variance,
    /// Auto-variance AQ with bias.
    AutoVariance,
}

/// Video encoder profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoProfile {
    /// Baseline profile (fastest decode).
    Baseline,
    /// Main profile (balanced).
    Main,
    /// High profile (best compression).
    High,
}

impl VideoProfile {
    /// Check if B-frames are allowed.
    #[must_use]
    pub const fn allow_b_frames(&self) -> bool {
        matches!(self, Self::Main | Self::High)
    }

    /// Check if 8x8 transforms are allowed.
    #[must_use]
    pub const fn allow_8x8_transform(&self) -> bool {
        matches!(self, Self::High)
    }

    /// Get maximum reference frames.
    #[must_use]
    pub const fn max_ref_frames(&self) -> usize {
        match self {
            Self::Baseline => 1,
            Self::Main => 2,
            Self::High => 4,
        }
    }
}

/// Lookahead configuration.
#[derive(Debug, Clone, Copy)]
pub struct LookaheadSettings {
    /// Number of frames to lookahead.
    pub frames: usize,
    /// Enable scene detection.
    pub scene_detect: bool,
    /// Enable adaptive I-frame placement.
    pub adaptive_i: bool,
    /// Enable adaptive B-frame placement.
    pub adaptive_b: bool,
}

impl Default for LookaheadSettings {
    fn default() -> Self {
        Self {
            frames: 40,
            scene_detect: true,
            adaptive_i: true,
            adaptive_b: false,
        }
    }
}

/// Motion estimation settings.
#[derive(Debug, Clone, Copy)]
pub struct MotionEstimationSettings {
    /// Search algorithm.
    pub algorithm: MotionSearchAlgorithm,
    /// Search range.
    pub range: i16,
    /// Subpixel refinement mode.
    pub subpel_mode: SubpelMode,
    /// Maximum candidates to check.
    pub max_candidates: usize,
    /// Enable early termination.
    pub early_termination: bool,
}

impl Default for MotionEstimationSettings {
    fn default() -> Self {
        Self {
            algorithm: MotionSearchAlgorithm::Diamond,
            range: 16,
            subpel_mode: SubpelMode::HalfPel,
            max_candidates: 16,
            early_termination: true,
        }
    }
}

/// Motion search algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionSearchAlgorithm {
    /// Full search (exhaustive).
    Full,
    /// Diamond search.
    Diamond,
    /// Hexagonal search.
    Hexagon,
    /// Uneven multi-hexagon search.
    UnevenMultiHex,
}

/// Subpixel refinement mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubpelMode {
    /// No subpixel refinement.
    None,
    /// Half-pixel precision.
    HalfPel,
    /// Quarter-pixel precision.
    QuarterPel,
}

/// Quantization settings.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationSettings {
    /// Minimum QP.
    pub min_qp: u8,
    /// Maximum QP.
    pub max_qp: u8,
    /// I-frame QP offset.
    pub i_qp_offset: i8,
    /// P-frame QP offset.
    pub p_qp_offset: i8,
    /// Enable adaptive deadzone.
    pub adaptive_deadzone: bool,
    /// Deadzone size.
    pub deadzone: f32,
}

impl Default for QuantizationSettings {
    fn default() -> Self {
        Self {
            min_qp: 10,
            max_qp: 51,
            i_qp_offset: 0,
            p_qp_offset: 2,
            adaptive_deadzone: true,
            deadzone: 0.3,
        }
    }
}

/// Preset configuration builder.
pub struct PresetBuilder {
    settings: EncoderSettings,
}

impl PresetBuilder {
    /// Start with a preset.
    #[must_use]
    pub fn new(preset: EncoderPreset) -> Self {
        Self {
            settings: EncoderSettings::from_preset(preset),
        }
    }

    /// Apply tune.
    #[must_use]
    pub fn tune(mut self, tune: EncodingTune) -> Self {
        self.settings = self.settings.with_tune(tune);
        self
    }

    /// Set quality.
    #[must_use]
    pub fn quality(mut self, quality: u8) -> Self {
        self.settings = self.settings.with_quality(quality);
        self
    }

    /// Set bitrate.
    #[must_use]
    pub fn bitrate(mut self, bitrate: u64) -> Self {
        self.settings = self.settings.with_bitrate(bitrate);
        self
    }

    /// Set keyframe interval.
    #[must_use]
    pub fn keyint(mut self, keyint: u32) -> Self {
        self.settings = self.settings.with_keyint(keyint);
        self
    }

    /// Build final settings.
    #[must_use]
    pub fn build(self) -> EncoderSettings {
        self.settings
    }
}

/// Common preset configurations.
pub struct Presets;

impl Presets {
    /// Web streaming preset.
    #[must_use]
    pub fn web_streaming() -> EncoderSettings {
        PresetBuilder::new(EncoderPreset::Fast)
            .tune(EncodingTune::FastDecode)
            .quality(28)
            .bitrate(1_500_000)
            .keyint(60)
            .build()
    }

    /// Archive preset (high quality).
    #[must_use]
    pub fn archive() -> EncoderSettings {
        PresetBuilder::new(EncoderPreset::Veryslow)
            .tune(EncodingTune::Film)
            .quality(20)
            .keyint(250)
            .build()
    }

    /// Low latency preset.
    #[must_use]
    pub fn low_latency() -> EncoderSettings {
        PresetBuilder::new(EncoderPreset::Ultrafast)
            .tune(EncodingTune::FastDecode)
            .quality(35)
            .keyint(30)
            .build()
    }

    /// Animation preset.
    #[must_use]
    pub fn animation() -> EncoderSettings {
        PresetBuilder::new(EncoderPreset::Slow)
            .tune(EncodingTune::Animation)
            .quality(24)
            .keyint(150)
            .build()
    }

    /// Screen capture preset.
    #[must_use]
    pub fn screen_capture() -> EncoderSettings {
        PresetBuilder::new(EncoderPreset::Medium)
            .tune(EncodingTune::Screen)
            .quality(22)
            .keyint(120)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_speed() {
        assert!(EncoderPreset::Ultrafast.speed() > EncoderPreset::Placebo.speed());
        assert_eq!(EncoderPreset::Medium.speed(), 5);
    }

    #[test]
    fn test_preset_settings() {
        assert!(EncoderPreset::Placebo.use_rdo());
        assert!(!EncoderPreset::Ultrafast.use_rdo());
        assert!(EncoderPreset::Slow.use_trellis());
    }

    #[test]
    fn test_encoding_tune() {
        let tune = EncodingTune::Film;
        assert!(tune.perceptual_strength() > 0.5);
        assert!(tune.enable_deblock());

        let grain_tune = EncodingTune::Grain;
        assert!(!grain_tune.enable_deblock());
    }

    #[test]
    fn test_encoder_settings_builder() {
        let settings = EncoderSettings::from_preset(EncoderPreset::Medium)
            .with_quality(25)
            .with_bitrate(3_000_000)
            .with_keyint(180);

        assert_eq!(settings.quality, 25);
        assert_eq!(settings.bitrate, 3_000_000);
        assert_eq!(settings.keyint, 180);
    }

    #[test]
    fn test_preset_builder() {
        let settings = PresetBuilder::new(EncoderPreset::Fast)
            .tune(EncodingTune::Animation)
            .quality(28)
            .build();

        assert_eq!(settings.preset, EncoderPreset::Fast);
        assert_eq!(settings.tune, EncodingTune::Animation);
        assert_eq!(settings.quality, 28);
    }

    #[test]
    fn test_common_presets() {
        let web = Presets::web_streaming();
        assert_eq!(web.preset, EncoderPreset::Fast);

        let archive = Presets::archive();
        assert_eq!(archive.preset, EncoderPreset::Veryslow);

        let animation = Presets::animation();
        assert_eq!(animation.tune, EncodingTune::Animation);
    }

    #[test]
    fn test_video_profile() {
        assert!(VideoProfile::High.allow_b_frames());
        assert!(!VideoProfile::Baseline.allow_b_frames());
        assert_eq!(VideoProfile::High.max_ref_frames(), 4);
    }

    #[test]
    fn test_aq_mode() {
        let mode = AdaptiveQuantMode::Variance;
        assert_ne!(mode, AdaptiveQuantMode::None);
    }
}

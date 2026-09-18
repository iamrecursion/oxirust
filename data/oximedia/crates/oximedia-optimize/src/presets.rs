//! Optimization presets for different use cases.

use crate::{ContentType, OptimizationLevel, OptimizerConfig};

/// Preset configurations for common use cases.
pub struct OptimizationPresets;

impl OptimizationPresets {
    /// Ultra-fast preset: Minimal optimization, maximum speed.
    #[must_use]
    pub fn ultrafast() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Fast,
            enable_psychovisual: false,
            enable_aq: false,
            lookahead_frames: 0,
            content_type: ContentType::Generic,
            parallel_rdo: false,
            lambda_multiplier: 0.8,
            enable_roi: false,
            enable_temporal_aq: false,
        }
    }

    /// Superfast preset: Very fast with basic optimizations.
    #[must_use]
    pub fn superfast() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Fast,
            enable_psychovisual: false,
            enable_aq: true,
            lookahead_frames: 5,
            content_type: ContentType::Generic,
            parallel_rdo: true,
            lambda_multiplier: 0.9,
            enable_roi: false,
            enable_temporal_aq: false,
        }
    }

    /// Veryfast preset: Fast with moderate optimization.
    #[must_use]
    pub fn veryfast() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Fast,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 10,
            content_type: ContentType::Generic,
            parallel_rdo: true,
            lambda_multiplier: 1.0,
            enable_roi: false,
            enable_temporal_aq: false,
        }
    }

    /// Faster preset: Good speed/quality balance, leaning towards speed.
    #[must_use]
    pub fn faster() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Medium,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 15,
            content_type: ContentType::Generic,
            parallel_rdo: true,
            lambda_multiplier: 1.0,
            enable_roi: false,
            enable_temporal_aq: false,
        }
    }

    /// Fast preset: Balanced speed and quality.
    #[must_use]
    pub fn fast() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Medium,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 20,
            content_type: ContentType::Generic,
            parallel_rdo: true,
            lambda_multiplier: 1.0,
            enable_roi: false,
            enable_temporal_aq: false,
        }
    }

    /// Medium preset: Default balanced configuration.
    #[must_use]
    pub fn medium() -> OptimizerConfig {
        OptimizerConfig::default()
    }

    /// Slow preset: High quality with slower encoding.
    #[must_use]
    pub fn slow() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Slow,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 40,
            content_type: ContentType::Generic,
            parallel_rdo: true,
            lambda_multiplier: 1.0,
            enable_roi: false,
            enable_temporal_aq: true,
        }
    }

    /// Slower preset: Very high quality with much slower encoding.
    #[must_use]
    pub fn slower() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Slow,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 60,
            content_type: ContentType::Generic,
            parallel_rdo: true,
            lambda_multiplier: 1.1,
            enable_roi: false,
            enable_temporal_aq: true,
        }
    }

    /// Veryslow preset: Maximum quality, very slow encoding.
    #[must_use]
    pub fn veryslow() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Placebo,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 80,
            content_type: ContentType::Generic,
            parallel_rdo: true,
            lambda_multiplier: 1.1,
            enable_roi: false,
            enable_temporal_aq: true,
        }
    }

    /// Placebo preset: Absolute maximum quality, extremely slow.
    #[must_use]
    pub fn placebo() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Placebo,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 120,
            content_type: ContentType::Generic,
            parallel_rdo: true,
            lambda_multiplier: 1.2,
            enable_roi: false,
            enable_temporal_aq: true,
        }
    }

    /// Animation preset: Optimized for anime/animation content.
    #[must_use]
    pub fn animation() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Medium,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 30,
            content_type: ContentType::Animation,
            parallel_rdo: true,
            lambda_multiplier: 1.1, // Slightly higher for sharp edges
            enable_roi: false,
            enable_temporal_aq: false,
        }
    }

    /// Film preset: Optimized for live-action film content.
    #[must_use]
    pub fn film() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Slow,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 50,
            content_type: ContentType::Film,
            parallel_rdo: true,
            lambda_multiplier: 1.0,
            enable_roi: false,
            enable_temporal_aq: true,
        }
    }

    /// Grain preset: Optimized for grainy film content.
    #[must_use]
    pub fn grain() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Slow,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 40,
            content_type: ContentType::Film,
            parallel_rdo: true,
            lambda_multiplier: 0.9, // Lower for grain preservation
            enable_roi: false,
            enable_temporal_aq: true,
        }
    }

    /// Screen preset: Optimized for screen recording/capture.
    #[must_use]
    pub fn screen() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Medium,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 20,
            content_type: ContentType::Screen,
            parallel_rdo: true,
            lambda_multiplier: 1.2, // Higher for text preservation
            enable_roi: false,
            enable_temporal_aq: false,
        }
    }

    /// Still image preset: Optimized for encoding still images.
    #[must_use]
    pub fn stillimage() -> OptimizerConfig {
        OptimizerConfig {
            level: OptimizationLevel::Placebo,
            enable_psychovisual: true,
            enable_aq: true,
            lookahead_frames: 0,
            content_type: ContentType::Generic,
            parallel_rdo: true,
            lambda_multiplier: 1.0,
            enable_roi: false,
            enable_temporal_aq: false,
        }
    }
}

/// Tune presets for specific optimization goals.
pub struct TunePresets;

impl TunePresets {
    /// PSNR tune: Optimize for PSNR metric.
    pub fn apply_psnr_tune(config: &mut OptimizerConfig) {
        config.enable_psychovisual = false;
        config.lambda_multiplier = 1.0;
    }

    /// SSIM tune: Optimize for SSIM metric.
    pub fn apply_ssim_tune(config: &mut OptimizerConfig) {
        config.enable_psychovisual = true;
        config.lambda_multiplier = 1.05;
    }

    /// Grain tune: Preserve film grain.
    pub fn apply_grain_tune(config: &mut OptimizerConfig) {
        config.enable_psychovisual = true;
        config.lambda_multiplier *= 0.9;
    }

    /// Fastdecode tune: Optimize for fast decoding.
    pub fn apply_fastdecode_tune(config: &mut OptimizerConfig) {
        // Prefer simpler modes that decode faster
        config.lookahead_frames = config.lookahead_frames.min(20);
    }

    /// Zerolatency tune: Minimize latency.
    pub fn apply_zerolatency_tune(config: &mut OptimizerConfig) {
        config.lookahead_frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ultrafast_preset() {
        let config = OptimizationPresets::ultrafast();
        assert_eq!(config.level, OptimizationLevel::Fast);
        assert!(!config.enable_psychovisual);
        assert!(!config.enable_aq);
    }

    #[test]
    fn test_placebo_preset() {
        let config = OptimizationPresets::placebo();
        assert_eq!(config.level, OptimizationLevel::Placebo);
        assert!(config.enable_psychovisual);
        assert!(config.enable_aq);
        assert_eq!(config.lookahead_frames, 120);
    }

    #[test]
    fn test_animation_preset() {
        let config = OptimizationPresets::animation();
        assert_eq!(config.content_type, ContentType::Animation);
    }

    #[test]
    fn test_film_preset() {
        let config = OptimizationPresets::film();
        assert_eq!(config.content_type, ContentType::Film);
        assert_eq!(config.level, OptimizationLevel::Slow);
    }

    #[test]
    fn test_screen_preset() {
        let config = OptimizationPresets::screen();
        assert_eq!(config.content_type, ContentType::Screen);
    }

    #[test]
    fn test_psnr_tune() {
        let mut config = OptimizerConfig::default();
        TunePresets::apply_psnr_tune(&mut config);
        assert!(!config.enable_psychovisual);
        assert_eq!(config.lambda_multiplier, 1.0);
    }

    #[test]
    fn test_ssim_tune() {
        let mut config = OptimizerConfig::default();
        TunePresets::apply_ssim_tune(&mut config);
        assert!(config.enable_psychovisual);
        assert!(config.lambda_multiplier > 1.0);
    }

    #[test]
    fn test_grain_tune() {
        let mut config = OptimizerConfig::default();
        let original_lambda = config.lambda_multiplier;
        TunePresets::apply_grain_tune(&mut config);
        assert!(config.lambda_multiplier < original_lambda);
    }

    #[test]
    fn test_zerolatency_tune() {
        let mut config = OptimizerConfig::default();
        TunePresets::apply_zerolatency_tune(&mut config);
        assert_eq!(config.lookahead_frames, 0);
    }

    #[test]
    fn test_all_presets_valid() {
        let presets = [
            OptimizationPresets::ultrafast(),
            OptimizationPresets::superfast(),
            OptimizationPresets::veryfast(),
            OptimizationPresets::faster(),
            OptimizationPresets::fast(),
            OptimizationPresets::medium(),
            OptimizationPresets::slow(),
            OptimizationPresets::slower(),
            OptimizationPresets::veryslow(),
            OptimizationPresets::placebo(),
            OptimizationPresets::animation(),
            OptimizationPresets::film(),
            OptimizationPresets::grain(),
            OptimizationPresets::screen(),
            OptimizationPresets::stillimage(),
        ];

        for preset in &presets {
            // All presets should have valid lambda multiplier
            assert!(preset.lambda_multiplier > 0.0);
            assert!(preset.lambda_multiplier <= 2.0);
        }
    }
}

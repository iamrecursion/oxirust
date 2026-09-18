//! Proxy generation module.

pub mod batch;
pub mod encoder;
pub mod optimizer;
pub mod presets;
pub mod settings;
pub mod streaming;

pub use batch::{BatchProxyGenerator, BatchResult};
pub use encoder::{ProxyEncodeResult, ProxyEncoder};
pub use optimizer::ProxyOptimizer;
pub use presets::{PresetInfo, ProxyPresets};
pub use settings::ProxyGenerationSettings;
pub use streaming::{
    CompleteOutcome, ConstantBitrateModel, GenerationProgress, ProgressiveProxy, ProxySegment,
    SegmentPlan, SegmentSpec, StreamingProxyConfig, StreamingProxyGenerator,
};

use crate::Result;
use std::path::Path;

/// High-level proxy generator interface.
pub struct ProxyGenerator {
    #[allow(dead_code)]
    settings: ProxyGenerationSettings,
}

impl ProxyGenerator {
    /// Create a new proxy generator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            settings: ProxyGenerationSettings::default(),
        }
    }

    /// Create a proxy generator with custom settings.
    #[must_use]
    pub fn with_settings(settings: ProxyGenerationSettings) -> Self {
        Self { settings }
    }

    /// Generate a proxy file from an input file.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub async fn generate(
        &self,
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
        preset: ProxyPreset,
    ) -> Result<ProxyEncodeResult> {
        let settings = preset.to_settings();
        let encoder = ProxyEncoder::new(settings)?;
        encoder.encode(input.as_ref(), output.as_ref()).await
    }

    /// Generate a proxy with custom settings.
    pub async fn generate_with_settings(
        &self,
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
        settings: ProxyGenerationSettings,
    ) -> Result<ProxyEncodeResult> {
        let encoder = ProxyEncoder::new(settings)?;
        encoder.encode(input.as_ref(), output.as_ref()).await
    }
}

impl Default for ProxyGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Proxy preset for quick setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyPreset {
    /// Quarter resolution H.264 proxy (25% scale, low bitrate).
    QuarterResH264,
    /// Half resolution H.264 proxy (50% scale, medium bitrate).
    HalfResH264,
    /// Full resolution H.264 proxy (100% scale, high bitrate).
    FullResH264,
    /// Quarter resolution VP9 proxy (25% scale, very efficient).
    QuarterResVP9,
    /// Half resolution VP9 proxy (50% scale, efficient).
    HalfResVP9,
}

impl ProxyPreset {
    /// Convert preset to generation settings.
    #[must_use]
    pub fn to_settings(self) -> ProxyGenerationSettings {
        match self {
            Self::QuarterResH264 => ProxyGenerationSettings::quarter_res_h264(),
            Self::HalfResH264 => ProxyGenerationSettings::half_res_h264(),
            Self::FullResH264 => ProxyGenerationSettings::full_res_h264(),
            Self::QuarterResVP9 => ProxyGenerationSettings::quarter_res_vp9(),
            Self::HalfResVP9 => ProxyGenerationSettings::default()
                .with_scale_factor(0.5)
                .with_codec("vp9")
                .with_bitrate(3_000_000),
        }
    }

    /// Get the name of this preset.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::QuarterResH264 => "Quarter Res H.264",
            Self::HalfResH264 => "Half Res H.264",
            Self::FullResH264 => "Full Res H.264",
            Self::QuarterResVP9 => "Quarter Res VP9",
            Self::HalfResVP9 => "Half Res VP9",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_generator_creation() {
        let generator = ProxyGenerator::new();
        assert_eq!(generator.settings.scale_factor, 0.25);
    }

    #[test]
    fn test_proxy_presets() {
        let settings = ProxyPreset::QuarterResH264.to_settings();
        assert_eq!(settings.scale_factor, 0.25);
        assert_eq!(settings.codec, "h264");

        let settings = ProxyPreset::HalfResH264.to_settings();
        assert_eq!(settings.scale_factor, 0.5);

        let settings = ProxyPreset::QuarterResVP9.to_settings();
        assert_eq!(settings.codec, "vp9");
    }

    #[test]
    fn test_preset_names() {
        assert_eq!(ProxyPreset::QuarterResH264.name(), "Quarter Res H.264");
        assert_eq!(ProxyPreset::HalfResVP9.name(), "Half Res VP9");
    }
}

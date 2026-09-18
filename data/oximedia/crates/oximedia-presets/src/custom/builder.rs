//! Custom preset builder.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Builder for creating custom presets.
#[derive(Debug, Default)]
pub struct PresetBuilder {
    id: Option<String>,
    name: Option<String>,
    description: Option<String>,
    category: Option<PresetCategory>,
    tags: Vec<String>,
    video_codec: Option<String>,
    audio_codec: Option<String>,
    video_bitrate: Option<u64>,
    audio_bitrate: Option<u64>,
    width: Option<u32>,
    height: Option<u32>,
    frame_rate: Option<(u32, u32)>,
    quality_mode: Option<QualityMode>,
    container: Option<String>,
}

impl PresetBuilder {
    /// Create a new preset builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set preset ID.
    #[must_use]
    pub fn id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    /// Set preset name.
    #[must_use]
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Set description.
    #[must_use]
    pub fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set category.
    #[must_use]
    pub fn category(mut self, category: PresetCategory) -> Self {
        self.category = Some(category);
        self
    }

    /// Add a tag.
    #[must_use]
    pub fn tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Set video codec.
    #[must_use]
    pub fn video_codec(mut self, codec: &str) -> Self {
        self.video_codec = Some(codec.to_string());
        self
    }

    /// Set audio codec.
    #[must_use]
    pub fn audio_codec(mut self, codec: &str) -> Self {
        self.audio_codec = Some(codec.to_string());
        self
    }

    /// Set video bitrate.
    #[must_use]
    pub fn video_bitrate(mut self, bitrate: u64) -> Self {
        self.video_bitrate = Some(bitrate);
        self
    }

    /// Set audio bitrate.
    #[must_use]
    pub fn audio_bitrate(mut self, bitrate: u64) -> Self {
        self.audio_bitrate = Some(bitrate);
        self
    }

    /// Set resolution.
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set frame rate.
    #[must_use]
    pub fn frame_rate(mut self, num: u32, den: u32) -> Self {
        self.frame_rate = Some((num, den));
        self
    }

    /// Set quality mode.
    #[must_use]
    pub fn quality_mode(mut self, mode: QualityMode) -> Self {
        self.quality_mode = Some(mode);
        self
    }

    /// Set container.
    #[must_use]
    pub fn container(mut self, container: &str) -> Self {
        self.container = Some(container.to_string());
        self
    }

    /// Build the preset.
    #[must_use]
    pub fn build(self) -> Preset {
        let id = self.id.unwrap_or_else(|| "custom-preset".to_string());
        let name = self.name.unwrap_or_else(|| "Custom Preset".to_string());
        let category = self.category.unwrap_or(PresetCategory::Custom);

        let mut metadata = PresetMetadata::new(&id, &name, category);
        if let Some(desc) = self.description {
            metadata.description = desc;
        }
        metadata.tags = self.tags;

        let config = PresetConfig {
            video_codec: self.video_codec,
            audio_codec: self.audio_codec,
            video_bitrate: self.video_bitrate,
            audio_bitrate: self.audio_bitrate,
            width: self.width,
            height: self.height,
            frame_rate: self.frame_rate,
            quality_mode: self.quality_mode,
            container: self.container,
            audio_channel_layout: None,
        };

        Preset::new(metadata, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() {
        let preset = PresetBuilder::new()
            .id("custom-1080p")
            .name("Custom 1080p")
            .description("Custom HD preset")
            .video_codec("h264")
            .audio_codec("aac")
            .resolution(1920, 1080)
            .video_bitrate(5_000_000)
            .audio_bitrate(192_000)
            .frame_rate(30, 1)
            .quality_mode(QualityMode::High)
            .container("mp4")
            .build();

        assert_eq!(preset.metadata.id, "custom-1080p");
        assert_eq!(preset.config.width, Some(1920));
    }
}

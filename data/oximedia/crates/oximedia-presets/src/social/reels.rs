//! Social media reels/shorts format presets (vertical 9:16).

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all reels/shorts presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![reels_hd(), reels_hq()]
}

/// Returns the vertical HD reels/shorts preset.
#[must_use]
pub fn reels_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "reels-hd",
        "Reels/Shorts HD",
        PresetCategory::Social("Reels".to_string()),
    )
    .with_description("Vertical HD reels/shorts")
    .with_tag("reels")
    .with_tag("shorts")
    .with_tag("9:16")
    .with_tag("vertical");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(5_000_000),
        audio_bitrate: Some(192_000),
        width: Some(1080),
        height: Some(1920),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the vertical high-quality 60fps reels/shorts preset.
#[must_use]
pub fn reels_hq() -> Preset {
    let metadata = PresetMetadata::new(
        "reels-hq",
        "Reels/Shorts HQ",
        PresetCategory::Social("Reels".to_string()),
    )
    .with_description("Vertical HQ reels/shorts with 60fps")
    .with_tag("reels")
    .with_tag("shorts")
    .with_tag("9:16")
    .with_tag("60fps");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(8_000_000),
        audio_bitrate: Some(256_000),
        width: Some(1080),
        height: Some(1920),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_reels_presets_count() {
        assert_eq!(all_presets().len(), 2);
    }
}

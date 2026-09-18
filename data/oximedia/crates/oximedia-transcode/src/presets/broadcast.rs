//! Broadcast and professional proxy presets.

use crate::{PresetConfig, QualityMode};

/// Broadcast HD `ProRes` Proxy (1280x720).
///
/// Note: This uses H.264 as a substitute since we use royalty-free codecs.
/// In a commercial setting, this would use `ProRes`.
#[must_use]
pub fn prores_proxy_hd() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(45_000_000), // 45 Mbps (ProRes Proxy equivalent)
        audio_bitrate: Some(256_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((25, 1)), // PAL
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// Broadcast Full HD `ProRes` Proxy (1920x1080).
#[must_use]
pub fn prores_proxy_full_hd() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(90_000_000), // 90 Mbps
        audio_bitrate: Some(256_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// Broadcast 4K `ProRes` Proxy (3840x2160).
#[must_use]
pub fn prores_proxy_4k() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(180_000_000), // 180 Mbps
        audio_bitrate: Some(320_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// `DNxHD` proxy for Avid (1920x1080).
///
/// Note: This uses H.264 as a substitute for `DNxHD`.
#[must_use]
pub fn dnxhd_proxy() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(36_000_000), // 36 Mbps (DNxHD 36)
        audio_bitrate: Some(256_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// EBU R128 compliant broadcast preset (1920x1080).
#[must_use]
pub fn ebu_r128_hd() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(50_000_000), // 50 Mbps
        audio_bitrate: Some(384_000),    // High-quality audio
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)), // PAL
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// ATSC A/85 compliant broadcast preset (1920x1080).
#[must_use]
pub fn atsc_a85_hd() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(50_000_000), // 50 Mbps
        audio_bitrate: Some(384_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30000, 1001)), // 29.97 fps (NTSC)
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prores_proxy_hd() {
        let preset = prores_proxy_hd();
        assert_eq!(preset.width, Some(1280));
        assert_eq!(preset.height, Some(720));
        assert_eq!(preset.frame_rate, Some((25, 1)));
    }

    #[test]
    fn test_prores_proxy_4k() {
        let preset = prores_proxy_4k();
        assert_eq!(preset.width, Some(3840));
        assert_eq!(preset.height, Some(2160));
        assert_eq!(preset.video_bitrate, Some(180_000_000));
    }

    #[test]
    fn test_ebu_r128() {
        let preset = ebu_r128_hd();
        assert_eq!(preset.frame_rate, Some((25, 1))); // PAL
        assert_eq!(preset.audio_bitrate, Some(384_000));
    }

    #[test]
    fn test_atsc_a85() {
        let preset = atsc_a85_hd();
        assert_eq!(preset.frame_rate, Some((30000, 1001))); // 29.97 fps
    }
}

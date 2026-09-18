//! Opus audio-only presets for podcast distribution and music streaming.
//!
//! Opus is the IETF-standardised codec (RFC 6716) offering state-of-the-art
//! compression efficiency for both speech and music.  It is the recommended
//! lossy format for:
//!
//! - Podcast RSS feeds (OGG/Opus or WebM)
//! - Music streaming delivery (Spotify at 320 kbps Ogg Vorbis targets; Opus
//!   achieves transparent quality at 128–192 kbps)
//! - Bandwidth-constrained live podcast streaming
//!
//! # Preset overview
//!
//! | Preset ID                     | Bitrate   | Channels | Use Case                          |
//! |-------------------------------|-----------|----------|-----------------------------------|
//! | `opus-podcast-speech-mono`    |  64 kbps  | Mono     | Solo podcast / interview (narrow) |
//! | `opus-podcast-speech-stereo`  |  96 kbps  | Stereo   | Multi-mic podcast, stereo room    |
//! | `opus-podcast-high-quality`   | 128 kbps  | Stereo   | High-fidelity podcast master      |
//! | `opus-music-streaming`        | 160 kbps  | Stereo   | Music streaming (transparent)     |
//! | `opus-music-hq`               | 192 kbps  | Stereo   | Near-lossless music streaming     |
//! | `opus-music-transparent`      | 256 kbps  | Stereo   | Perceptually transparent music    |
//! | `opus-podcast-mobile`         |  48 kbps  | Mono     | Mobile data-efficient podcast     |

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{audio_channel_map::AudioLayout, PresetConfig, QualityMode};

/// Return all Opus podcast and music distribution presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        opus_podcast_speech_mono(),
        opus_podcast_speech_stereo(),
        opus_podcast_high_quality(),
        opus_music_streaming(),
        opus_music_hq(),
        opus_music_transparent(),
        opus_podcast_mobile(),
    ]
}

/// Opus mono speech preset at 64 kbps for solo/interview podcasts.
///
/// 64 kbps mono Opus delivers transparent speech quality — indistinguishable
/// from a 128 kbps MP3 in listening tests.  Mono is preferred for single-mic
/// recordings to eliminate phase issues and halve bandwidth.
#[must_use]
pub fn opus_podcast_speech_mono() -> Preset {
    let metadata = PresetMetadata::new(
        "opus-podcast-speech-mono",
        "Opus Podcast Speech Mono",
        PresetCategory::Codec("Opus".to_string()),
    )
    .with_description("Mono Opus at 64 kbps for solo and interview podcasts — speech-optimised")
    .with_target("Podcast Distribution")
    .with_tag("opus")
    .with_tag("podcast")
    .with_tag("speech")
    .with_tag("mono")
    .with_tag("audio-only")
    .with_tag("64kbps");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("opus".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(64_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::Medium),
        container: Some("ogg".to_string()),
        audio_channel_layout: Some(AudioLayout::Mono),
    };

    Preset::new(metadata, config)
}

/// Opus stereo speech preset at 96 kbps for multi-mic podcasts.
///
/// Stereo 96 kbps captures room ambience and multiple microphone positions.
/// Suitable for panel podcasts or recordings with a distinct stereo field.
#[must_use]
pub fn opus_podcast_speech_stereo() -> Preset {
    let metadata = PresetMetadata::new(
        "opus-podcast-speech-stereo",
        "Opus Podcast Speech Stereo",
        PresetCategory::Codec("Opus".to_string()),
    )
    .with_description("Stereo Opus at 96 kbps for multi-mic panel podcasts — speech-optimised")
    .with_target("Podcast Distribution")
    .with_tag("opus")
    .with_tag("podcast")
    .with_tag("speech")
    .with_tag("stereo")
    .with_tag("audio-only")
    .with_tag("96kbps");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("opus".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(96_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::Medium),
        container: Some("ogg".to_string()),
        audio_channel_layout: Some(AudioLayout::Stereo),
    };

    Preset::new(metadata, config)
}

/// Opus stereo high-quality podcast at 128 kbps.
///
/// The standard production-ready podcast delivery bitrate. Covers
/// mixed content (speech + music beds + sound design) with headroom
/// to spare. Compatible with all podcast players via OGG container.
#[must_use]
pub fn opus_podcast_high_quality() -> Preset {
    let metadata = PresetMetadata::new(
        "opus-podcast-high-quality",
        "Opus Podcast High Quality",
        PresetCategory::Codec("Opus".to_string()),
    )
    .with_description("High-quality stereo Opus at 128 kbps — the standard podcast delivery format")
    .with_target("Podcast Distribution")
    .with_tag("opus")
    .with_tag("podcast")
    .with_tag("stereo")
    .with_tag("audio-only")
    .with_tag("128kbps")
    .with_tag("hq");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("opus".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(128_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::High),
        container: Some("ogg".to_string()),
        audio_channel_layout: Some(AudioLayout::Stereo),
    };

    Preset::new(metadata, config)
}

/// Opus stereo music streaming preset at 160 kbps.
///
/// Opus at 160 kbps achieves transparent or near-transparent music quality
/// according to multiple ABX tests, exceeding the perceived quality of
/// 320 kbps MP3 for most listeners.
#[must_use]
pub fn opus_music_streaming() -> Preset {
    let metadata = PresetMetadata::new(
        "opus-music-streaming",
        "Opus Music Streaming",
        PresetCategory::Codec("Opus".to_string()),
    )
    .with_description("Music streaming Opus at 160 kbps stereo — transparent for most listeners")
    .with_target("Music Streaming")
    .with_tag("opus")
    .with_tag("music")
    .with_tag("streaming")
    .with_tag("stereo")
    .with_tag("audio-only")
    .with_tag("160kbps");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("opus".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(160_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::High),
        container: Some("ogg".to_string()),
        audio_channel_layout: Some(AudioLayout::Stereo),
    };

    Preset::new(metadata, config)
}

/// Opus stereo high-quality music at 192 kbps.
///
/// Virtually indistinguishable from lossless in double-blind tests.
/// Recommended for music distribution services targeting premium listeners
/// who reject MP3 but cannot receive lossless FLAC.
#[must_use]
pub fn opus_music_hq() -> Preset {
    let metadata = PresetMetadata::new(
        "opus-music-hq",
        "Opus Music HQ",
        PresetCategory::Codec("Opus".to_string()),
    )
    .with_description("Near-lossless stereo Opus at 192 kbps for premium music distribution")
    .with_target("Music Distribution")
    .with_tag("opus")
    .with_tag("music")
    .with_tag("stereo")
    .with_tag("audio-only")
    .with_tag("192kbps")
    .with_tag("hq");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("opus".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(192_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::High),
        container: Some("ogg".to_string()),
        audio_channel_layout: Some(AudioLayout::Stereo),
    };

    Preset::new(metadata, config)
}

/// Opus stereo transparent music at 256 kbps.
///
/// The highest practical Opus bitrate for stereo music.  At 256 kbps
/// Opus is perceptually lossless for all known musical content —
/// equivalent to FLAC on casual listening equipment while using ~20%
/// of the storage.
#[must_use]
pub fn opus_music_transparent() -> Preset {
    let metadata = PresetMetadata::new(
        "opus-music-transparent",
        "Opus Music Transparent",
        PresetCategory::Codec("Opus".to_string()),
    )
    .with_description(
        "Perceptually transparent Opus at 256 kbps stereo — highest quality lossy music",
    )
    .with_target("Music Distribution")
    .with_tag("opus")
    .with_tag("music")
    .with_tag("stereo")
    .with_tag("audio-only")
    .with_tag("256kbps")
    .with_tag("transparent");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("opus".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(256_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("ogg".to_string()),
        audio_channel_layout: Some(AudioLayout::Stereo),
    };

    Preset::new(metadata, config)
}

/// Opus mono mobile-optimised podcast at 48 kbps.
///
/// Designed for mobile data plans and offline sync.  48 kbps mono Opus
/// is virtually transparent for speech while consuming only 21.6 MB/hour
/// — a sixth of a typical 128 kbps MP3.
#[must_use]
pub fn opus_podcast_mobile() -> Preset {
    let metadata = PresetMetadata::new(
        "opus-podcast-mobile",
        "Opus Podcast Mobile",
        PresetCategory::Codec("Opus".to_string()),
    )
    .with_description(
        "Mobile-optimised mono Opus at 48 kbps for data-constrained podcast listeners",
    )
    .with_target("Podcast Distribution (Mobile)")
    .with_tag("opus")
    .with_tag("podcast")
    .with_tag("speech")
    .with_tag("mono")
    .with_tag("audio-only")
    .with_tag("48kbps")
    .with_tag("mobile");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("opus".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(48_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::Low),
        container: Some("ogg".to_string()),
        audio_channel_layout: Some(AudioLayout::Mono),
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_podcast_presets_count() {
        assert_eq!(all_presets().len(), 7);
    }

    #[test]
    fn test_all_opus_podcast_presets_have_no_video() {
        for p in all_presets() {
            assert!(
                p.config.video_codec.is_none(),
                "Opus preset {} must not have a video codec",
                p.metadata.id
            );
            assert!(
                p.config.video_bitrate.is_none(),
                "Opus preset {} must not have a video bitrate",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_opus_podcast_presets_use_ogg_container() {
        for p in all_presets() {
            assert_eq!(
                p.config.container.as_deref(),
                Some("ogg"),
                "Opus preset {} should use ogg container",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_opus_podcast_presets_have_audio_only_tag() {
        for p in all_presets() {
            assert!(
                p.has_tag("audio-only"),
                "Opus preset {} must have 'audio-only' tag",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_opus_podcast_presets_have_opus_tag() {
        for p in all_presets() {
            assert!(
                p.has_tag("opus"),
                "Opus preset {} must have 'opus' tag",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_opus_podcast_speech_mono() {
        let p = opus_podcast_speech_mono();
        assert_eq!(p.metadata.id, "opus-podcast-speech-mono");
        assert_eq!(p.config.audio_bitrate, Some(64_000));
        assert_eq!(p.config.audio_channel_layout, Some(AudioLayout::Mono));
        assert!(p.has_tag("speech"));
        assert!(p.has_tag("mono"));
    }

    #[test]
    fn test_opus_podcast_speech_stereo() {
        let p = opus_podcast_speech_stereo();
        assert_eq!(p.metadata.id, "opus-podcast-speech-stereo");
        assert_eq!(p.config.audio_bitrate, Some(96_000));
        assert_eq!(p.config.audio_channel_layout, Some(AudioLayout::Stereo));
    }

    #[test]
    fn test_opus_podcast_high_quality() {
        let p = opus_podcast_high_quality();
        assert_eq!(p.metadata.id, "opus-podcast-high-quality");
        assert_eq!(p.config.audio_bitrate, Some(128_000));
        assert!(p.has_tag("hq"));
    }

    #[test]
    fn test_opus_music_streaming() {
        let p = opus_music_streaming();
        assert_eq!(p.metadata.id, "opus-music-streaming");
        assert_eq!(p.config.audio_bitrate, Some(160_000));
        assert!(p.has_tag("streaming"));
        assert!(p.has_tag("music"));
    }

    #[test]
    fn test_opus_music_hq() {
        let p = opus_music_hq();
        assert_eq!(p.metadata.id, "opus-music-hq");
        assert_eq!(p.config.audio_bitrate, Some(192_000));
        assert!(p.has_tag("hq"));
    }

    #[test]
    fn test_opus_music_transparent() {
        let p = opus_music_transparent();
        assert_eq!(p.metadata.id, "opus-music-transparent");
        assert_eq!(p.config.audio_bitrate, Some(256_000));
        assert!(p.has_tag("transparent"));
    }

    #[test]
    fn test_opus_podcast_mobile() {
        let p = opus_podcast_mobile();
        assert_eq!(p.metadata.id, "opus-podcast-mobile");
        assert_eq!(p.config.audio_bitrate, Some(48_000));
        assert_eq!(p.config.audio_channel_layout, Some(AudioLayout::Mono));
        assert!(p.has_tag("mobile"));
        assert!(p.has_tag("48kbps"));
    }

    #[test]
    fn test_bitrate_ordering() {
        // Verify bitrates increase in the expected order
        assert!(
            opus_podcast_mobile().config.audio_bitrate
                < opus_podcast_speech_mono().config.audio_bitrate
        );
        assert!(
            opus_podcast_speech_mono().config.audio_bitrate
                < opus_podcast_speech_stereo().config.audio_bitrate
        );
        assert!(
            opus_podcast_speech_stereo().config.audio_bitrate
                < opus_podcast_high_quality().config.audio_bitrate
        );
        assert!(
            opus_podcast_high_quality().config.audio_bitrate
                < opus_music_streaming().config.audio_bitrate
        );
        assert!(opus_music_streaming().config.audio_bitrate < opus_music_hq().config.audio_bitrate);
        assert!(
            opus_music_hq().config.audio_bitrate < opus_music_transparent().config.audio_bitrate
        );
    }
}

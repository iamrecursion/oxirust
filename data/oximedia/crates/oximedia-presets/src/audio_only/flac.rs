//! FLAC audio-only presets for lossless archival, podcast masters, and music distribution.
//!
//! FLAC (Free Lossless Audio Codec) is the standard for:
//! - Lossless music distribution (Bandcamp, Qobuz, HDtracks)
//! - Podcast source masters before lossy encode
//! - Archive-grade spoken-word recordings
//! - Music production interchange
//!
//! # Specifications
//!
//! | Preset                     | Sample Rate | Bit Depth | Channels | Use Case                |
//! |----------------------------|-------------|-----------|----------|-------------------------|
//! | `flac-podcast-master`      | 44100 Hz    | 16-bit    | Stereo   | Podcast pre-master      |
//! | `flac-music-cd-quality`    | 44100 Hz    | 16-bit    | Stereo   | CD-quality music master |
//! | `flac-music-hires-96`      | 96000 Hz    | 24-bit    | Stereo   | Hi-Res music 96kHz/24   |
//! | `flac-music-hires-192`     | 192000 Hz   | 24-bit    | Stereo   | Hi-Res music 192kHz/24  |
//! | `flac-podcast-mono`        | 44100 Hz    | 16-bit    | Mono     | Mono podcast archival   |
//! | `flac-broadcast-48`        | 48000 Hz    | 24-bit    | Stereo   | Broadcast-aligned 48kHz |

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{audio_channel_map::AudioLayout, PresetConfig, QualityMode};

/// Return all FLAC audio-only presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        flac_podcast_master(),
        flac_music_cd_quality(),
        flac_music_hires_96(),
        flac_music_hires_192(),
        flac_podcast_mono(),
        flac_broadcast_48(),
    ]
}

/// FLAC podcast pre-master at 44.1 kHz / 16-bit stereo.
///
/// The standard capture format for podcast production workflows.
/// Preserved as lossless before encoding to MP3 or Opus for distribution.
/// 44100 Hz sample rate ensures compatibility with all podcast hosts.
#[must_use]
pub fn flac_podcast_master() -> Preset {
    let metadata = PresetMetadata::new(
        "flac-podcast-master",
        "FLAC Podcast Master",
        PresetCategory::Codec("FLAC".to_string()),
    )
    .with_description("Lossless FLAC pre-master for podcast production at 44.1 kHz / 16-bit stereo")
    .with_target("Podcast Production")
    .with_tag("flac")
    .with_tag("podcast")
    .with_tag("lossless")
    .with_tag("audio-only")
    .with_tag("master")
    .with_tag("44100");

    // FLAC is lossless; audio_bitrate encodes the theoretical uncompressed PCM rate
    // (44100 samples/s × 16 bits × 2 channels = 1,411,200 bit/s).
    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(1_411_200),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("flac".to_string()),
        audio_channel_layout: Some(AudioLayout::Stereo),
    };

    Preset::new(metadata, config)
}

/// FLAC CD-quality master at 44.1 kHz / 16-bit stereo.
///
/// Identical to the Red Book CD standard — the industry baseline for music
/// distribution on platforms such as Bandcamp and independent music stores.
#[must_use]
pub fn flac_music_cd_quality() -> Preset {
    let metadata = PresetMetadata::new(
        "flac-music-cd-quality",
        "FLAC CD Quality",
        PresetCategory::Codec("FLAC".to_string()),
    )
    .with_description("CD-quality FLAC at 44.1 kHz / 16-bit (Red Book standard)")
    .with_target("Music Distribution")
    .with_tag("flac")
    .with_tag("music")
    .with_tag("lossless")
    .with_tag("audio-only")
    .with_tag("cd-quality")
    .with_tag("44100");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(1_411_200), // 44100 × 16 × 2
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("flac".to_string()),
        audio_channel_layout: Some(AudioLayout::Stereo),
    };

    Preset::new(metadata, config)
}

/// FLAC hi-res music master at 96 kHz / 24-bit stereo.
///
/// The most common hi-res audio format sold by Qobuz, HDtracks, and Apple
/// Music (lossless tier). Captures harmonics well above the audible range
/// used by mastering engineers for processing headroom.
#[must_use]
pub fn flac_music_hires_96() -> Preset {
    let metadata = PresetMetadata::new(
        "flac-music-hires-96",
        "FLAC Hi-Res 96kHz/24-bit",
        PresetCategory::Codec("FLAC".to_string()),
    )
    .with_description("Hi-Res FLAC at 96 kHz / 24-bit stereo for premium music distribution")
    .with_target("Hi-Res Music Distribution")
    .with_tag("flac")
    .with_tag("music")
    .with_tag("lossless")
    .with_tag("audio-only")
    .with_tag("hires")
    .with_tag("96000");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(4_608_000), // 96000 × 24 × 2
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("flac".to_string()),
        audio_channel_layout: Some(AudioLayout::Stereo),
    };

    Preset::new(metadata, config)
}

/// FLAC ultra-hi-res music master at 192 kHz / 24-bit stereo.
///
/// The maximum sample rate supported by the FLAC specification and most
/// professional DACs. Reserved for recording studio archives and audiophile
/// distribution (e.g., MQA source material preparation).
#[must_use]
pub fn flac_music_hires_192() -> Preset {
    let metadata = PresetMetadata::new(
        "flac-music-hires-192",
        "FLAC Hi-Res 192kHz/24-bit",
        PresetCategory::Codec("FLAC".to_string()),
    )
    .with_description("Ultra hi-res FLAC at 192 kHz / 24-bit stereo for audiophile archival")
    .with_target("Studio Archive / Audiophile Distribution")
    .with_tag("flac")
    .with_tag("music")
    .with_tag("lossless")
    .with_tag("audio-only")
    .with_tag("hires")
    .with_tag("192000")
    .with_tag("studio");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(9_216_000), // 192000 × 24 × 2
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("flac".to_string()),
        audio_channel_layout: Some(AudioLayout::Stereo),
    };

    Preset::new(metadata, config)
}

/// FLAC mono podcast archival at 44.1 kHz / 16-bit.
///
/// Mono capture is standard for solo/interview podcasts where stereo provides
/// no additional content. Halves file size vs. stereo while retaining full
/// lossless fidelity for the spoken-word signal.
#[must_use]
pub fn flac_podcast_mono() -> Preset {
    let metadata = PresetMetadata::new(
        "flac-podcast-mono",
        "FLAC Podcast Mono",
        PresetCategory::Codec("FLAC".to_string()),
    )
    .with_description("Mono FLAC archival at 44.1 kHz / 16-bit for spoken-word podcasts")
    .with_target("Podcast Archival")
    .with_tag("flac")
    .with_tag("podcast")
    .with_tag("lossless")
    .with_tag("audio-only")
    .with_tag("mono")
    .with_tag("44100");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(705_600), // 44100 × 16 × 1 (mono)
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::High),
        container: Some("flac".to_string()),
        audio_channel_layout: Some(AudioLayout::Mono),
    };

    Preset::new(metadata, config)
}

/// FLAC broadcast-aligned master at 48 kHz / 24-bit stereo.
///
/// Broadcast and post-production workflows use 48 kHz as their native sample
/// rate (AES3 / SMPTE standard). This preset is suited for audio-for-picture
/// masters or podcast content originating from video production.
#[must_use]
pub fn flac_broadcast_48() -> Preset {
    let metadata = PresetMetadata::new(
        "flac-broadcast-48",
        "FLAC Broadcast 48kHz/24-bit",
        PresetCategory::Codec("FLAC".to_string()),
    )
    .with_description("Broadcast-aligned FLAC at 48 kHz / 24-bit stereo (AES3 / SMPTE compatible)")
    .with_target("Broadcast / Post-Production")
    .with_tag("flac")
    .with_tag("broadcast")
    .with_tag("lossless")
    .with_tag("audio-only")
    .with_tag("48000");

    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(2_304_000), // 48000 × 24 × 2
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("flac".to_string()),
        audio_channel_layout: Some(AudioLayout::Stereo),
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flac_presets_count() {
        assert_eq!(all_presets().len(), 6);
    }

    #[test]
    fn test_all_flac_presets_have_no_video() {
        for p in all_presets() {
            assert!(
                p.config.video_codec.is_none(),
                "FLAC preset {} must not have a video codec",
                p.metadata.id
            );
            assert!(
                p.config.video_bitrate.is_none(),
                "FLAC preset {} must not have a video bitrate",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_flac_presets_use_flac_codec() {
        for p in all_presets() {
            assert_eq!(
                p.config.audio_codec.as_deref(),
                Some("flac"),
                "Preset {} should use flac codec",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_flac_presets_use_flac_container() {
        for p in all_presets() {
            assert_eq!(
                p.config.container.as_deref(),
                Some("flac"),
                "Preset {} should use flac container",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_flac_presets_have_audio_only_tag() {
        for p in all_presets() {
            assert!(
                p.has_tag("audio-only"),
                "Preset {} must have 'audio-only' tag",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_flac_podcast_master() {
        let p = flac_podcast_master();
        assert_eq!(p.metadata.id, "flac-podcast-master");
        assert_eq!(p.config.audio_bitrate, Some(1_411_200));
        assert_eq!(p.config.audio_channel_layout, Some(AudioLayout::Stereo));
        assert!(p.has_tag("podcast"));
        assert!(p.has_tag("master"));
    }

    #[test]
    fn test_flac_music_hires_96() {
        let p = flac_music_hires_96();
        assert_eq!(p.metadata.id, "flac-music-hires-96");
        assert_eq!(p.config.audio_bitrate, Some(4_608_000));
        assert!(p.has_tag("hires"));
        assert!(p.has_tag("96000"));
    }

    #[test]
    fn test_flac_music_hires_192() {
        let p = flac_music_hires_192();
        assert_eq!(p.metadata.id, "flac-music-hires-192");
        assert_eq!(p.config.audio_bitrate, Some(9_216_000));
        assert!(p.has_tag("hires"));
        assert!(p.has_tag("192000"));
    }

    #[test]
    fn test_flac_podcast_mono_is_mono() {
        let p = flac_podcast_mono();
        assert_eq!(p.config.audio_channel_layout, Some(AudioLayout::Mono));
        assert!(p.has_tag("mono"));
        // Mono bitrate = half of stereo at same sample rate
        assert_eq!(p.config.audio_bitrate, Some(705_600));
    }

    #[test]
    fn test_flac_broadcast_48() {
        let p = flac_broadcast_48();
        assert_eq!(p.metadata.id, "flac-broadcast-48");
        assert!(p.has_tag("broadcast"));
        assert!(p.has_tag("48000"));
        assert_eq!(p.config.audio_bitrate, Some(2_304_000));
    }

    #[test]
    fn test_flac_music_cd_quality() {
        let p = flac_music_cd_quality();
        assert_eq!(p.metadata.id, "flac-music-cd-quality");
        assert!(p.has_tag("cd-quality"));
        assert_eq!(p.config.audio_bitrate, Some(1_411_200));
    }
}

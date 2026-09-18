//! Digital Cinema Package (DCP) presets.
//!
//! DCP is the standard for theatrical digital cinema distribution, governed
//! by SMPTE ST 429 and related standards. Key constraints:
//!
//! - **Video codec**: JPEG 2000 (wavelet-based, visually lossless)
//! - **Resolution**: 2K (2048x1080) or 4K (4096x2160) in DCI container (Flat/Scope)
//! - **Frame rates**: 24 fps (film), 25 fps (European TV-sourced), 48 fps (HFR)
//! - **Max bitrate**: 250 Mbps video
//! - **Audio**: PCM (linear, uncompressed), 48 kHz or 96 kHz, up to 16 channels
//! - **Container**: MXF (Material eXchange Format)
//!
//! These presets produce configurations suitable for DCP mastering workflows.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all DCP presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        dcp_2k_24fps(),
        dcp_2k_25fps(),
        dcp_2k_scope_24fps(),
        dcp_4k_24fps(),
        dcp_4k_25fps(),
        dcp_2k_48fps_hfr(),
    ]
}

/// DCP 2K Flat (1998x1080) at 24 fps.
///
/// The most common theatrical DCI format: 2K resolution in the Flat (1.85:1)
/// container at 24 frames per second. Video at 250 Mbps JPEG 2000,
/// 24-bit PCM audio at 48 kHz.
#[must_use]
pub fn dcp_2k_24fps() -> Preset {
    let metadata = PresetMetadata::new(
        "dcp-2k-flat-24fps",
        "DCP 2K Flat 24fps",
        PresetCategory::Platform("DCP".to_string()),
    )
    .with_description("DCI 2K Flat (1998x1080) JPEG 2000 at 24 fps, 250 Mbps max")
    .with_target("Digital Cinema Package")
    .with_tag("dcp")
    .with_tag("2k")
    .with_tag("flat")
    .with_tag("24fps")
    .with_tag("jpeg2000")
    .with_tag("cinema");

    let config = PresetConfig {
        video_codec: Some("jpeg2000".to_string()),
        audio_codec: Some("pcm".to_string()),
        video_bitrate: Some(250_000_000), // 250 Mbps — DCI max
        audio_bitrate: Some(2_304_000),   // 48 kHz * 24 bit * 2 channels
        width: Some(1998),                // DCI Flat container
        height: Some(1080),
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mxf".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DCP 2K Flat (1998x1080) at 25 fps.
///
/// European theatrical distribution at 25 fps, matching PAL-origin content.
/// Same 250 Mbps JPEG 2000 ceiling, PCM audio.
#[must_use]
pub fn dcp_2k_25fps() -> Preset {
    let metadata = PresetMetadata::new(
        "dcp-2k-flat-25fps",
        "DCP 2K Flat 25fps",
        PresetCategory::Platform("DCP".to_string()),
    )
    .with_description("DCI 2K Flat (1998x1080) JPEG 2000 at 25 fps, 250 Mbps max")
    .with_target("Digital Cinema Package")
    .with_tag("dcp")
    .with_tag("2k")
    .with_tag("flat")
    .with_tag("25fps")
    .with_tag("jpeg2000")
    .with_tag("cinema");

    let config = PresetConfig {
        video_codec: Some("jpeg2000".to_string()),
        audio_codec: Some("pcm".to_string()),
        video_bitrate: Some(250_000_000),
        audio_bitrate: Some(2_400_000), // 48 kHz * 24 bit * 2 ch (slightly different rate)
        width: Some(1998),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mxf".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DCP 2K Scope (2048x858) at 24 fps.
///
/// The anamorphic Scope (2.39:1) container for CinemaScope presentations.
/// Uses the full 2048-pixel width with a reduced height.
#[must_use]
pub fn dcp_2k_scope_24fps() -> Preset {
    let metadata = PresetMetadata::new(
        "dcp-2k-scope-24fps",
        "DCP 2K Scope 24fps",
        PresetCategory::Platform("DCP".to_string()),
    )
    .with_description("DCI 2K Scope (2048x858) JPEG 2000 at 24 fps, 250 Mbps max")
    .with_target("Digital Cinema Package")
    .with_tag("dcp")
    .with_tag("2k")
    .with_tag("scope")
    .with_tag("24fps")
    .with_tag("jpeg2000")
    .with_tag("cinema");

    let config = PresetConfig {
        video_codec: Some("jpeg2000".to_string()),
        audio_codec: Some("pcm".to_string()),
        video_bitrate: Some(250_000_000),
        audio_bitrate: Some(2_304_000),
        width: Some(2048), // DCI Scope container (full 2K width)
        height: Some(858), // Scope aspect ratio
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mxf".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DCP 4K Flat (3996x2160) at 24 fps.
///
/// Premium 4K DCI resolution for large-format projection. JPEG 2000
/// at up to 250 Mbps, PCM audio. Requires significant storage and
/// processing power.
#[must_use]
pub fn dcp_4k_24fps() -> Preset {
    let metadata = PresetMetadata::new(
        "dcp-4k-flat-24fps",
        "DCP 4K Flat 24fps",
        PresetCategory::Platform("DCP".to_string()),
    )
    .with_description("DCI 4K Flat (3996x2160) JPEG 2000 at 24 fps, 250 Mbps max")
    .with_target("Digital Cinema Package")
    .with_tag("dcp")
    .with_tag("4k")
    .with_tag("flat")
    .with_tag("24fps")
    .with_tag("jpeg2000")
    .with_tag("cinema");

    let config = PresetConfig {
        video_codec: Some("jpeg2000".to_string()),
        audio_codec: Some("pcm".to_string()),
        video_bitrate: Some(250_000_000),
        audio_bitrate: Some(2_304_000),
        width: Some(3996), // DCI 4K Flat
        height: Some(2160),
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mxf".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DCP 4K Flat (3996x2160) at 25 fps.
///
/// European 4K theatrical distribution at 25 fps.
#[must_use]
pub fn dcp_4k_25fps() -> Preset {
    let metadata = PresetMetadata::new(
        "dcp-4k-flat-25fps",
        "DCP 4K Flat 25fps",
        PresetCategory::Platform("DCP".to_string()),
    )
    .with_description("DCI 4K Flat (3996x2160) JPEG 2000 at 25 fps, 250 Mbps max")
    .with_target("Digital Cinema Package")
    .with_tag("dcp")
    .with_tag("4k")
    .with_tag("flat")
    .with_tag("25fps")
    .with_tag("jpeg2000")
    .with_tag("cinema");

    let config = PresetConfig {
        video_codec: Some("jpeg2000".to_string()),
        audio_codec: Some("pcm".to_string()),
        video_bitrate: Some(250_000_000),
        audio_bitrate: Some(2_400_000),
        width: Some(3996),
        height: Some(2160),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mxf".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DCP 2K HFR (1998x1080) at 48 fps.
///
/// High Frame Rate cinema at 48 fps (used for select HFR presentations).
/// Same 250 Mbps JPEG 2000 ceiling, doubling the temporal resolution.
#[must_use]
pub fn dcp_2k_48fps_hfr() -> Preset {
    let metadata = PresetMetadata::new(
        "dcp-2k-flat-48fps-hfr",
        "DCP 2K Flat 48fps HFR",
        PresetCategory::Platform("DCP".to_string()),
    )
    .with_description("DCI 2K Flat HFR (1998x1080) JPEG 2000 at 48 fps, 250 Mbps max")
    .with_target("Digital Cinema Package")
    .with_tag("dcp")
    .with_tag("2k")
    .with_tag("flat")
    .with_tag("48fps")
    .with_tag("hfr")
    .with_tag("jpeg2000")
    .with_tag("cinema");

    let config = PresetConfig {
        video_codec: Some("jpeg2000".to_string()),
        audio_codec: Some("pcm".to_string()),
        video_bitrate: Some(250_000_000),
        audio_bitrate: Some(2_304_000),
        width: Some(1998),
        height: Some(1080),
        frame_rate: Some((48, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mxf".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dcp_presets_count() {
        assert_eq!(all_presets().len(), 6);
    }

    #[test]
    fn test_dcp_2k_24fps() {
        let p = dcp_2k_24fps();
        assert_eq!(p.metadata.id, "dcp-2k-flat-24fps");
        assert_eq!(p.config.width, Some(1998));
        assert_eq!(p.config.height, Some(1080));
        assert_eq!(p.config.frame_rate, Some((24, 1)));
        assert_eq!(p.config.video_bitrate, Some(250_000_000));
        assert_eq!(p.config.video_codec.as_deref(), Some("jpeg2000"));
        assert_eq!(p.config.audio_codec.as_deref(), Some("pcm"));
        assert_eq!(p.config.container.as_deref(), Some("mxf"));
    }

    #[test]
    fn test_dcp_2k_25fps() {
        let p = dcp_2k_25fps();
        assert_eq!(p.config.frame_rate, Some((25, 1)));
    }

    #[test]
    fn test_dcp_2k_scope() {
        let p = dcp_2k_scope_24fps();
        assert_eq!(p.config.width, Some(2048));
        assert_eq!(p.config.height, Some(858));
        assert!(p.has_tag("scope"));
    }

    #[test]
    fn test_dcp_4k_24fps() {
        let p = dcp_4k_24fps();
        assert_eq!(p.config.width, Some(3996));
        assert_eq!(p.config.height, Some(2160));
        assert!(p.has_tag("4k"));
    }

    #[test]
    fn test_dcp_4k_25fps() {
        let p = dcp_4k_25fps();
        assert_eq!(p.config.frame_rate, Some((25, 1)));
        assert_eq!(p.config.width, Some(3996));
    }

    #[test]
    fn test_dcp_2k_48fps_hfr() {
        let p = dcp_2k_48fps_hfr();
        assert_eq!(p.config.frame_rate, Some((48, 1)));
        assert!(p.has_tag("hfr"));
    }

    #[test]
    fn test_all_dcp_presets_use_jpeg2000() {
        for p in all_presets() {
            assert_eq!(
                p.config.video_codec.as_deref(),
                Some("jpeg2000"),
                "DCP preset {} should use JPEG 2000",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_dcp_presets_use_mxf() {
        for p in all_presets() {
            assert_eq!(
                p.config.container.as_deref(),
                Some("mxf"),
                "DCP preset {} should use MXF container",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_dcp_presets_use_pcm() {
        for p in all_presets() {
            assert_eq!(
                p.config.audio_codec.as_deref(),
                Some("pcm"),
                "DCP preset {} should use PCM audio",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_dcp_presets_max_250mbps() {
        for p in all_presets() {
            assert!(
                p.config.video_bitrate.unwrap_or(0) <= 250_000_000,
                "DCP preset {} exceeds 250 Mbps limit",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_dcp_presets_have_cinema_tag() {
        for p in all_presets() {
            assert!(
                p.has_tag("cinema"),
                "DCP preset {} should have 'cinema' tag",
                p.metadata.id
            );
        }
    }
}

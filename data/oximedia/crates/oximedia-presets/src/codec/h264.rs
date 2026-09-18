//! H.264/AVC preset configuration for `OxiMedia`.

#![allow(dead_code)]

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// H.264/AVC encoding profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H264Profile {
    /// Baseline profile (widest device compatibility).
    Baseline,
    /// Main profile (standard consumer devices).
    Main,
    /// High profile (quality compression, broad support).
    High,
    /// High 10-bit profile.
    High10,
    /// High 4:2:2 profile.
    High422,
    /// High 4:4:4 predictive profile.
    High444,
}

impl H264Profile {
    /// Return the canonical lowercase string for this profile.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Main => "main",
            Self::High => "high",
            Self::High10 => "high10",
            Self::High422 => "high422",
            Self::High444 => "high444",
        }
    }

    /// Returns `true` for profiles that support B-frames.
    #[must_use]
    pub fn supports_b_frames(&self) -> bool {
        !matches!(self, Self::Baseline)
    }
}

/// H.264/AVC level constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum H264Level {
    /// Level 3.0 – up to 720p@30.
    Level3,
    /// Level 3.1 – up to 1080p@30.
    Level31,
    /// Level 4.0 – up to 1080p@30 high bitrate.
    Level4,
    /// Level 4.1 – up to 1080p@60.
    Level41,
    /// Level 4.2 – up to 1080p@60 high bitrate.
    Level42,
    /// Level 5.0 – up to 4K.
    Level5,
    /// Level 5.1 – up to 4K high bitrate.
    Level51,
}

impl H264Level {
    /// Maximum macro-blocks per second (MBPS) as per the AVC standard.
    #[must_use]
    pub fn max_mbps(&self) -> u32 {
        match self {
            Self::Level3 => 40_500,
            Self::Level31 => 108_000,
            Self::Level4 => 245_760,
            Self::Level41 => 245_760,
            Self::Level42 => 522_240,
            Self::Level5 => 589_824,
            Self::Level51 => 983_040,
        }
    }

    /// Maximum decoded picture buffer size in macroblocks.
    #[must_use]
    pub fn max_dpb_mbs(&self) -> u32 {
        match self {
            Self::Level3 => 8_100,
            Self::Level31 => 18_000,
            Self::Level4 | Self::Level41 => 32_768,
            Self::Level42 => 34_816,
            Self::Level5 => 110_400,
            Self::Level51 => 184_320,
        }
    }

    /// Maximum bitrate in kbps for the given profile.
    ///
    /// High-tier profiles allow 4× the baseline constraint.
    #[must_use]
    pub fn max_bitrate_kbps(&self, profile: &H264Profile) -> u32 {
        let base = match self {
            Self::Level3 => 10_000,
            Self::Level31 => 14_000,
            Self::Level4 => 20_000,
            Self::Level41 => 50_000,
            Self::Level42 => 50_000,
            Self::Level5 => 135_000,
            Self::Level51 => 240_000,
        };
        match profile {
            H264Profile::High
            | H264Profile::High10
            | H264Profile::High422
            | H264Profile::High444 => (base as f64 * 1.25) as u32,
            _ => base,
        }
    }
}

/// A complete H.264/AVC encoding preset.
#[derive(Debug, Clone)]
pub struct H264Preset {
    /// H.264 profile.
    pub profile: H264Profile,
    /// H.264 level.
    pub level: H264Level,
    /// Constant rate factor (lower = higher quality; x264 range 0–51).
    pub crf: u8,
    /// x264 speed preset name (e.g. "medium", "slow").
    pub preset_name: String,
    /// Number of consecutive B-frames.
    pub b_frames: u8,
    /// Number of reference frames.
    pub ref_frames: u8,
}

impl H264Preset {
    /// Broad web-compatibility preset (Baseline/Level 3.1).
    #[must_use]
    pub fn web_compatible() -> Self {
        Self {
            profile: H264Profile::Baseline,
            level: H264Level::Level31,
            crf: 23,
            preset_name: "medium".to_string(),
            b_frames: 0,
            ref_frames: 2,
        }
    }

    /// Broadcast 720p preset (High/Level 3.1).
    #[must_use]
    pub fn broadcast_720p() -> Self {
        Self {
            profile: H264Profile::High,
            level: H264Level::Level31,
            crf: 18,
            preset_name: "slow".to_string(),
            b_frames: 3,
            ref_frames: 4,
        }
    }

    /// Broadcast 1080i preset (High/Level 4.0).
    #[must_use]
    pub fn broadcast_1080i() -> Self {
        Self {
            profile: H264Profile::High,
            level: H264Level::Level4,
            crf: 16,
            preset_name: "slow".to_string(),
            b_frames: 2,
            ref_frames: 4,
        }
    }

    /// Mobile low-bitrate preset (Baseline/Level 3.0).
    #[must_use]
    pub fn mobile_low() -> Self {
        Self {
            profile: H264Profile::Baseline,
            level: H264Level::Level3,
            crf: 30,
            preset_name: "fast".to_string(),
            b_frames: 0,
            ref_frames: 1,
        }
    }

    /// High-quality archive preset (High/Level 5.1).
    #[must_use]
    pub fn archive_high() -> Self {
        Self {
            profile: H264Profile::High,
            level: H264Level::Level51,
            crf: 14,
            preset_name: "veryslow".to_string(),
            b_frames: 8,
            ref_frames: 16,
        }
    }
}

/// Estimate a target bitrate in kbps for the given resolution and frame rate.
///
/// Uses a simplified rule-of-thumb based on pixels per second and a target
/// bits-per-pixel ratio of ~0.07 at standard motion.
#[must_use]
pub fn bitrate_for_resolution(width: u32, height: u32, fps: f64) -> u32 {
    let pixels_per_second = f64::from(width) * f64::from(height) * fps;
    let bpp = 0.07_f64;
    let bitrate_bps = pixels_per_second * bpp;
    (bitrate_bps / 1000.0) as u32
}

/// Build a `Preset` (library entry) from an `H264Preset`.
#[must_use]
pub fn to_library_preset(id: &str, name: &str, hp: &H264Preset) -> Preset {
    let metadata = PresetMetadata::new(id, name, PresetCategory::Codec("H264".to_string()))
        .with_tag("h264")
        .with_tag("avc")
        .with_description(&format!(
            "H.264 {} {} crf={}",
            hp.profile.as_str(),
            hp.preset_name,
            hp.crf
        ));

    let bitrate = u64::from(hp.level.max_bitrate_kbps(&hp.profile)) * 1000 / 2;

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(bitrate),
        audio_bitrate: Some(128_000),
        width: None,
        height: None,
        frame_rate: Some((30, 1)),
        quality_mode: Some(if hp.crf <= 18 {
            QualityMode::VeryHigh
        } else if hp.crf <= 23 {
            QualityMode::High
        } else {
            QualityMode::Medium
        }),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Return all built-in H.264 presets for the preset library.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        to_library_preset(
            "h264-web-compatible",
            "H.264 Web Compatible",
            &H264Preset::web_compatible(),
        ),
        to_library_preset(
            "h264-broadcast-720p",
            "H.264 Broadcast 720p",
            &H264Preset::broadcast_720p(),
        ),
        to_library_preset(
            "h264-broadcast-1080i",
            "H.264 Broadcast 1080i",
            &H264Preset::broadcast_1080i(),
        ),
        to_library_preset(
            "h264-mobile-low",
            "H.264 Mobile Low",
            &H264Preset::mobile_low(),
        ),
        to_library_preset(
            "h264-archive-high",
            "H.264 Archive High",
            &H264Preset::archive_high(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_as_str() {
        assert_eq!(H264Profile::Baseline.as_str(), "baseline");
        assert_eq!(H264Profile::High.as_str(), "high");
    }

    #[test]
    fn test_profile_baseline_no_b_frames() {
        assert!(!H264Profile::Baseline.supports_b_frames());
    }

    #[test]
    fn test_profile_high_supports_b_frames() {
        assert!(H264Profile::High.supports_b_frames());
    }

    #[test]
    fn test_level_max_mbps_ordering() {
        assert!(H264Level::Level3.max_mbps() < H264Level::Level51.max_mbps());
    }

    #[test]
    fn test_level_max_dpb_mbs_ordering() {
        assert!(H264Level::Level3.max_dpb_mbs() < H264Level::Level51.max_dpb_mbs());
    }

    #[test]
    fn test_level_max_bitrate_high_profile_bonus() {
        let base_bitrate = H264Level::Level41.max_bitrate_kbps(&H264Profile::Main);
        let high_bitrate = H264Level::Level41.max_bitrate_kbps(&H264Profile::High);
        assert!(high_bitrate > base_bitrate);
    }

    #[test]
    fn test_web_compatible_preset() {
        let p = H264Preset::web_compatible();
        assert_eq!(p.profile, H264Profile::Baseline);
        assert_eq!(p.b_frames, 0);
    }

    #[test]
    fn test_broadcast_720p_preset() {
        let p = H264Preset::broadcast_720p();
        assert_eq!(p.profile, H264Profile::High);
        assert!(p.b_frames > 0);
    }

    #[test]
    fn test_broadcast_1080i_preset() {
        let p = H264Preset::broadcast_1080i();
        assert_eq!(p.level, H264Level::Level4);
    }

    #[test]
    fn test_mobile_low_preset() {
        let p = H264Preset::mobile_low();
        assert!(p.crf > 26);
        assert_eq!(p.ref_frames, 1);
    }

    #[test]
    fn test_archive_high_preset() {
        let p = H264Preset::archive_high();
        assert!(p.crf <= 16);
        assert_eq!(p.ref_frames, 16);
    }

    #[test]
    fn test_bitrate_for_resolution_1080p_30fps() {
        let kbps = bitrate_for_resolution(1920, 1080, 30.0);
        // Expect roughly 4 Mbps
        assert!(kbps > 3000 && kbps < 6000, "kbps={kbps}");
    }

    #[test]
    fn test_bitrate_for_resolution_4k_higher_than_1080p() {
        let kbps_1080p = bitrate_for_resolution(1920, 1080, 30.0);
        let kbps_4k = bitrate_for_resolution(3840, 2160, 30.0);
        assert!(kbps_4k > kbps_1080p);
    }

    #[test]
    fn test_all_presets_count() {
        assert_eq!(all_presets().len(), 5);
    }

    #[test]
    fn test_all_presets_have_h264_tag() {
        for preset in all_presets() {
            assert!(
                preset.has_tag("h264"),
                "preset '{}' missing h264 tag",
                preset.metadata.id
            );
        }
    }
}

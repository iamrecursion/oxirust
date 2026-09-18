//! HEVC/H.265 preset configuration for `OxiMedia`.

#![allow(dead_code)]

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// HEVC/H.265 encoding profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HevcProfile {
    /// 8-bit Main profile.
    Main,
    /// 10-bit Main 10 profile (HDR).
    Main10,
    /// Main Still Picture profile for stills.
    MainStillPicture,
    /// 4:4:4 chroma sampling.
    Main444,
    /// Range extension profiles.
    MainRext,
}

impl HevcProfile {
    /// Return the canonical string identifier for FFmpeg/x265.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Main10 => "main10",
            Self::MainStillPicture => "mainstillpicture",
            Self::Main444 => "main444-8",
            Self::MainRext => "main-rext",
        }
    }

    /// Returns `true` for HDR-capable profiles.
    #[must_use]
    pub fn supports_hdr(&self) -> bool {
        matches!(self, Self::Main10 | Self::MainRext)
    }
}

/// HEVC/H.265 level constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HevcLevel {
    /// Level 3.0 – up to 1080@30.
    Level3,
    /// Level 4.0 – up to 2160@30 low bitrate.
    Level4,
    /// Level 4.1 – up to 2160@30.
    Level41,
    /// Level 5.0 – up to 4K@60.
    Level5,
    /// Level 5.1 – up to 4K@120.
    Level51,
    /// Level 5.2 – up to 4K@240.
    Level52,
}

impl HevcLevel {
    /// Maximum luma sample count per frame as specified by the HEVC standard.
    #[must_use]
    pub fn max_frame_size_luma_samples(&self) -> u64 {
        match self {
            Self::Level3 => 1_310_720,
            Self::Level4 => 8_912_896,
            Self::Level41 => 8_912_896,
            Self::Level5 => 35_651_584,
            Self::Level51 => 35_651_584,
            Self::Level52 => 35_651_584,
        }
    }

    /// Maximum video bitrate in kbps (Main tier).
    #[must_use]
    pub fn max_bitrate_kbps(&self) -> u32 {
        match self {
            Self::Level3 => 6_000,
            Self::Level4 => 12_000,
            Self::Level41 => 20_000,
            Self::Level5 => 25_000,
            Self::Level51 => 40_000,
            Self::Level52 => 60_000,
        }
    }

    /// Maximum luma picture size (width × height) in pixels.
    #[must_use]
    pub fn max_luma_picture_size(&self) -> u32 {
        match self {
            Self::Level3 => 1_310_720,
            Self::Level4 | Self::Level41 => 8_912_896,
            Self::Level5 | Self::Level51 | Self::Level52 => 35_651_584,
        }
    }
}

/// A complete HEVC/H.265 encoding preset.
#[derive(Debug, Clone)]
pub struct HevcPreset {
    /// HEVC profile.
    pub profile: HevcProfile,
    /// HEVC level constraint.
    pub level: HevcLevel,
    /// Constant rate factor (lower = higher quality; x265 range 0–51).
    pub crf: u8,
    /// x265 speed preset string (e.g. "medium", "slow").
    pub preset_speed: String,
    /// Whether to enable HDR metadata.
    pub hdr: bool,
    /// Number of consecutive B-frames.
    pub b_frames: u8,
    /// Number of reference frames.
    pub ref_frames: u8,
}

impl HevcPreset {
    /// Standard HD streaming preset (1080p / Main profile).
    #[must_use]
    pub fn hd_streaming() -> Self {
        Self {
            profile: HevcProfile::Main,
            level: HevcLevel::Level41,
            crf: 23,
            preset_speed: "medium".to_string(),
            hdr: false,
            b_frames: 3,
            ref_frames: 4,
        }
    }

    /// Ultra HD HDR preset (4K / Main 10 profile).
    #[must_use]
    pub fn uhd_hdr() -> Self {
        Self {
            profile: HevcProfile::Main10,
            level: HevcLevel::Level51,
            crf: 18,
            preset_speed: "slow".to_string(),
            hdr: true,
            b_frames: 4,
            ref_frames: 6,
        }
    }

    /// High-quality archive preset.
    #[must_use]
    pub fn archive() -> Self {
        Self {
            profile: HevcProfile::Main,
            level: HevcLevel::Level51,
            crf: 14,
            preset_speed: "veryslow".to_string(),
            hdr: false,
            b_frames: 8,
            ref_frames: 8,
        }
    }

    /// Fast proxy / editing preset (low quality, high speed).
    #[must_use]
    pub fn fast_proxy() -> Self {
        Self {
            profile: HevcProfile::Main,
            level: HevcLevel::Level4,
            crf: 28,
            preset_speed: "ultrafast".to_string(),
            hdr: false,
            b_frames: 0,
            ref_frames: 1,
        }
    }
}

/// Validate an HEVC preset and return a list of human-readable error strings.
///
/// Returns an empty vector if the preset is valid.
#[must_use]
pub fn validate_hevc_preset(p: &HevcPreset) -> Vec<String> {
    let mut errors = Vec::new();

    if p.crf > 51 {
        errors.push(format!("CRF {} is out of range (0–51)", p.crf));
    }

    if p.hdr && !p.profile.supports_hdr() {
        errors.push(format!(
            "Profile '{}' does not support HDR",
            p.profile.as_str()
        ));
    }

    if p.ref_frames > 16 {
        errors.push(format!(
            "ref_frames {} exceeds maximum allowed (16)",
            p.ref_frames
        ));
    }

    if p.preset_speed.is_empty() {
        errors.push("preset_speed must not be empty".to_string());
    }

    errors
}

/// Build a `Preset` (library entry) from a `HevcPreset`.
#[must_use]
pub fn to_library_preset(id: &str, name: &str, hp: &HevcPreset) -> Preset {
    let metadata = PresetMetadata::new(id, name, PresetCategory::Codec("HEVC".to_string()))
        .with_tag("hevc")
        .with_tag("h265")
        .with_description(&format!(
            "HEVC {} {} crf={}",
            hp.profile.as_str(),
            hp.preset_speed,
            hp.crf
        ));

    // Map to a generic PresetConfig; bitrate is estimated from level.
    let config = PresetConfig {
        video_codec: Some("hevc".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(u64::from(hp.level.max_bitrate_kbps()) * 1000 / 2),
        audio_bitrate: Some(192_000),
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

/// Return all built-in HEVC presets for the preset library.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        to_library_preset(
            "hevc-hd-streaming",
            "HEVC HD Streaming",
            &HevcPreset::hd_streaming(),
        ),
        to_library_preset("hevc-uhd-hdr", "HEVC UHD HDR", &HevcPreset::uhd_hdr()),
        to_library_preset("hevc-archive", "HEVC Archive", &HevcPreset::archive()),
        to_library_preset(
            "hevc-fast-proxy",
            "HEVC Fast Proxy",
            &HevcPreset::fast_proxy(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hevc_profile_main_str() {
        assert_eq!(HevcProfile::Main.as_str(), "main");
    }

    #[test]
    fn test_hevc_profile_main10_str() {
        assert_eq!(HevcProfile::Main10.as_str(), "main10");
    }

    #[test]
    fn test_hevc_profile_hdr_support() {
        assert!(HevcProfile::Main10.supports_hdr());
        assert!(!HevcProfile::Main.supports_hdr());
    }

    #[test]
    fn test_level_max_bitrate_ordering() {
        assert!(HevcLevel::Level3.max_bitrate_kbps() < HevcLevel::Level51.max_bitrate_kbps());
    }

    #[test]
    fn test_level_max_frame_size_monotonic() {
        assert!(
            HevcLevel::Level3.max_frame_size_luma_samples()
                < HevcLevel::Level5.max_frame_size_luma_samples()
        );
    }

    #[test]
    fn test_hd_streaming_preset() {
        let p = HevcPreset::hd_streaming();
        assert_eq!(p.profile, HevcProfile::Main);
        assert_eq!(p.level, HevcLevel::Level41);
        assert!(!p.hdr);
    }

    #[test]
    fn test_uhd_hdr_preset_has_hdr() {
        let p = HevcPreset::uhd_hdr();
        assert!(p.hdr);
        assert!(p.profile.supports_hdr());
    }

    #[test]
    fn test_archive_preset_low_crf() {
        let p = HevcPreset::archive();
        assert!(p.crf <= 18);
    }

    #[test]
    fn test_fast_proxy_preset_high_crf() {
        let p = HevcPreset::fast_proxy();
        assert!(p.crf >= 26);
        assert_eq!(p.b_frames, 0);
    }

    #[test]
    fn test_validate_valid_preset_no_errors() {
        let p = HevcPreset::hd_streaming();
        assert!(validate_hevc_preset(&p).is_empty());
    }

    #[test]
    fn test_validate_crf_out_of_range() {
        let mut p = HevcPreset::hd_streaming();
        p.crf = 52;
        let errors = validate_hevc_preset(&p);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("CRF"));
    }

    #[test]
    fn test_validate_hdr_incompatible_profile() {
        let mut p = HevcPreset::hd_streaming();
        p.hdr = true; // Main profile does not support HDR
        let errors = validate_hevc_preset(&p);
        assert!(errors.iter().any(|e| e.contains("HDR")));
    }

    #[test]
    fn test_all_presets_count() {
        assert_eq!(all_presets().len(), 4);
    }

    #[test]
    fn test_all_presets_have_hevc_tag() {
        for preset in all_presets() {
            assert!(
                preset.has_tag("hevc"),
                "preset '{}' missing hevc tag",
                preset.metadata.id
            );
        }
    }
}

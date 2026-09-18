//! LinkedIn platform encoding presets for `OxiMedia`.

#![allow(dead_code)]

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Configuration for a LinkedIn video preset.
#[derive(Debug, Clone)]
pub struct LinkedInVideoPreset {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Maximum allowed video duration in seconds.
    pub max_duration_s: u32,
    /// Maximum allowed file size in megabytes.
    pub max_size_mb: u32,
    /// Recommended video codec (e.g. "h264").
    pub codec: String,
    /// Maximum recommended bitrate in kilobits per second.
    pub max_bitrate_kbps: u32,
}

impl LinkedInVideoPreset {
    /// Feed video preset – standard LinkedIn feed post.
    ///
    /// Supports aspect ratios from 1:2.4 to 2.4:1; 1920×1080 is recommended.
    #[must_use]
    pub fn feed_video() -> Self {
        Self {
            width: 1920,
            height: 1080,
            max_duration_s: 600, // 10 minutes
            max_size_mb: 5120,   // 5 GB
            codec: "h264".to_string(),
            max_bitrate_kbps: 10_000,
        }
    }

    /// Story video preset – LinkedIn Stories (deprecated by LinkedIn but kept for legacy).
    #[must_use]
    pub fn story() -> Self {
        Self {
            width: 1080,
            height: 1920,
            max_duration_s: 20,
            max_size_mb: 200,
            codec: "h264".to_string(),
            max_bitrate_kbps: 5_000,
        }
    }

    /// Cover video preset – LinkedIn profile / company page background video.
    #[must_use]
    pub fn cover_video() -> Self {
        Self {
            width: 1920,
            height: 1080,
            max_duration_s: 30,
            max_size_mb: 200,
            codec: "h264".to_string(),
            max_bitrate_kbps: 8_000,
        }
    }
}

/// Validate a LinkedIn video against platform constraints.
///
/// Returns a list of human-readable error strings; empty means valid.
#[must_use]
pub fn validate_linkedin_video(
    width: u32,
    height: u32,
    duration_s: u32,
    size_mb: u32,
) -> Vec<String> {
    let mut errors = Vec::new();

    // Minimum resolution
    if width < 256 || height < 144 {
        errors.push(format!(
            "Resolution {width}×{height} is below LinkedIn minimum (256×144)"
        ));
    }

    // Maximum resolution
    if width > 4096 || height > 2304 {
        errors.push(format!(
            "Resolution {width}×{height} exceeds LinkedIn maximum (4096×2304)"
        ));
    }

    // Aspect ratio: must be between 1:2.4 and 2.4:1
    if width > 0 && height > 0 {
        let aspect = width as f64 / height as f64;
        if aspect < 1.0 / 2.4 - 0.01 || aspect > 2.4 + 0.01 {
            errors.push(format!(
                "Aspect ratio {:.3} is outside LinkedIn allowed range (1:2.4 – 2.4:1)",
                aspect
            ));
        }
    }

    // Duration: feed videos max 10 minutes
    if duration_s > 600 {
        errors.push(format!(
            "Duration {duration_s}s exceeds LinkedIn maximum feed video duration (600s)"
        ));
    }

    // File size: max 5 GB for feed videos
    if size_mb > 5120 {
        errors.push(format!(
            "File size {size_mb} MB exceeds LinkedIn maximum (5120 MB)"
        ));
    }

    errors
}

/// Build a `Preset` library entry from a `LinkedInVideoPreset`.
fn to_library_preset(id: &str, name: &str, lp: &LinkedInVideoPreset) -> Preset {
    let metadata = PresetMetadata::new(id, name, PresetCategory::Platform("LinkedIn".to_string()))
        .with_tag("linkedin")
        .with_description(&format!(
            "LinkedIn {} – {}×{} max {}s {}MB",
            name, lp.width, lp.height, lp.max_duration_s, lp.max_size_mb
        ))
        .with_target("LinkedIn");

    let config = PresetConfig {
        video_codec: Some(lp.codec.clone()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(u64::from(lp.max_bitrate_kbps) * 1000),
        audio_bitrate: Some(128_000),
        width: Some(lp.width),
        height: Some(lp.height),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Return all built-in LinkedIn presets for the preset library.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        to_library_preset(
            "linkedin-feed-video",
            "LinkedIn Feed Video",
            &LinkedInVideoPreset::feed_video(),
        ),
        to_library_preset(
            "linkedin-story",
            "LinkedIn Story",
            &LinkedInVideoPreset::story(),
        ),
        to_library_preset(
            "linkedin-cover-video",
            "LinkedIn Cover Video",
            &LinkedInVideoPreset::cover_video(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_video_preset_dimensions() {
        let p = LinkedInVideoPreset::feed_video();
        assert_eq!(p.width, 1920);
        assert_eq!(p.height, 1080);
    }

    #[test]
    fn test_feed_video_max_duration() {
        let p = LinkedInVideoPreset::feed_video();
        assert_eq!(p.max_duration_s, 600);
    }

    #[test]
    fn test_story_preset_is_vertical() {
        let p = LinkedInVideoPreset::story();
        assert!(p.height > p.width, "story should be vertical");
    }

    #[test]
    fn test_cover_video_preset_codec() {
        let p = LinkedInVideoPreset::cover_video();
        assert_eq!(p.codec, "h264");
    }

    #[test]
    fn test_validate_valid_video() {
        let errors = validate_linkedin_video(1920, 1080, 60, 100);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn test_validate_below_min_resolution() {
        let errors = validate_linkedin_video(100, 50, 60, 10);
        assert!(errors.iter().any(|e| e.contains("minimum")));
    }

    #[test]
    fn test_validate_above_max_resolution() {
        let errors = validate_linkedin_video(5000, 3000, 60, 10);
        assert!(errors.iter().any(|e| e.contains("maximum")));
    }

    #[test]
    fn test_validate_duration_too_long() {
        let errors = validate_linkedin_video(1920, 1080, 700, 100);
        assert!(errors.iter().any(|e| e.contains("Duration")));
    }

    #[test]
    fn test_validate_file_too_large() {
        let errors = validate_linkedin_video(1920, 1080, 60, 6000);
        assert!(errors.iter().any(|e| e.contains("File size")));
    }

    #[test]
    fn test_validate_bad_aspect_ratio() {
        // 1:3 ratio is outside the allowed 1:2.4 – 2.4:1
        let errors = validate_linkedin_video(100, 300, 60, 10);
        assert!(errors.iter().any(|e| e.contains("Aspect")));
    }

    #[test]
    fn test_all_presets_count() {
        assert_eq!(all_presets().len(), 3);
    }

    #[test]
    fn test_all_presets_have_linkedin_tag() {
        for preset in all_presets() {
            assert!(
                preset.has_tag("linkedin"),
                "preset '{}' missing linkedin tag",
                preset.metadata.id
            );
        }
    }

    #[test]
    fn test_all_presets_are_mp4() {
        for preset in all_presets() {
            assert_eq!(
                preset.config.container,
                Some("mp4".to_string()),
                "preset '{}' should use mp4",
                preset.metadata.id
            );
        }
    }
}

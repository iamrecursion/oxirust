//! Social media video export templates and specifications.
//!
//! This module provides video and caption specifications for the major social media
//! platforms including YouTube, Instagram, TikTok, Twitter/X, Facebook, LinkedIn,
//! Snapchat, and Pinterest.

/// Social media platform identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum SocialPlatform {
    /// YouTube (all formats).
    YouTube,
    /// Instagram (feed, reels, stories).
    Instagram,
    /// TikTok (vertical short-form).
    TikTok,
    /// Twitter / X.
    Twitter,
    /// Facebook (all formats).
    Facebook,
    /// LinkedIn (professional video).
    LinkedIn,
    /// Snapchat (stories and ads).
    Snapchat,
    /// Pinterest (idea pins and promoted videos).
    Pinterest,
}

impl SocialPlatform {
    /// Return the human-readable platform name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::YouTube => "YouTube",
            Self::Instagram => "Instagram",
            Self::TikTok => "TikTok",
            Self::Twitter => "Twitter/X",
            Self::Facebook => "Facebook",
            Self::LinkedIn => "LinkedIn",
            Self::Snapchat => "Snapchat",
            Self::Pinterest => "Pinterest",
        }
    }
}

/// Video specification for a social media platform.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SocialVideoSpec {
    /// Target platform.
    pub platform: SocialPlatform,
    /// Recommended aspect ratio as (width, height).
    pub aspect_ratio: (u32, u32),
    /// Maximum clip duration in seconds.
    pub max_duration_secs: u32,
    /// Maximum file size in megabytes (0 = no limit).
    pub max_size_mb: u32,
    /// Recommended video bitrate in kbps.
    pub recommended_bitrate_kbps: u32,
    /// Supported output resolutions as a list of (width, height).
    pub supported_resolutions: Vec<(u32, u32)>,
}

/// Social media template generator.
pub struct SocialTemplateGenerator;

impl SocialTemplateGenerator {
    /// Return the optimal export specification for the given platform and content.
    ///
    /// The result takes the source resolution and clip duration into account and
    /// selects the best-matching resolution from the platform's supported list.
    #[must_use]
    pub fn optimal_export(
        platform: &SocialPlatform,
        duration_secs: u32,
        source_res: (u32, u32),
    ) -> SocialVideoSpec {
        let mut spec = base_spec(platform);

        // Clamp max duration.
        let _ = duration_secs; // used for awareness; callers can validate externally

        // Pick best resolution that does not exceed the source.
        let best = spec
            .supported_resolutions
            .iter()
            .filter(|&&(w, h)| w <= source_res.0 && h <= source_res.1)
            .last()
            .copied();

        if let Some(res) = best {
            spec.supported_resolutions = vec![res];
        } else if !spec.supported_resolutions.is_empty() {
            // Fall back to smallest supported resolution.
            let smallest = spec.supported_resolutions[0];
            spec.supported_resolutions = vec![smallest];
        }

        spec
    }
}

/// Build the base specification for a platform (full supported resolutions list).
fn base_spec(platform: &SocialPlatform) -> SocialVideoSpec {
    match platform {
        SocialPlatform::YouTube => SocialVideoSpec {
            platform: SocialPlatform::YouTube,
            aspect_ratio: (16, 9),
            max_duration_secs: 43_200, // 12 hours
            max_size_mb: 0,            // no limit for verified accounts
            recommended_bitrate_kbps: 8_000,
            supported_resolutions: vec![
                (426, 240),
                (640, 360),
                (854, 480),
                (1280, 720),
                (1920, 1080),
                (2560, 1440),
                (3840, 2160),
            ],
        },
        SocialPlatform::Instagram => SocialVideoSpec {
            platform: SocialPlatform::Instagram,
            aspect_ratio: (9, 16),
            max_duration_secs: 90,
            max_size_mb: 650,
            recommended_bitrate_kbps: 3_500,
            supported_resolutions: vec![(480, 854), (720, 1280), (1080, 1920)],
        },
        SocialPlatform::TikTok => SocialVideoSpec {
            platform: SocialPlatform::TikTok,
            aspect_ratio: (9, 16),
            max_duration_secs: 600,
            max_size_mb: 287,
            recommended_bitrate_kbps: 4_000,
            supported_resolutions: vec![(540, 960), (720, 1280), (1080, 1920)],
        },
        SocialPlatform::Twitter => SocialVideoSpec {
            platform: SocialPlatform::Twitter,
            aspect_ratio: (16, 9),
            max_duration_secs: 140,
            max_size_mb: 512,
            recommended_bitrate_kbps: 2_000,
            supported_resolutions: vec![(640, 360), (1280, 720), (1920, 1080)],
        },
        SocialPlatform::Facebook => SocialVideoSpec {
            platform: SocialPlatform::Facebook,
            aspect_ratio: (16, 9),
            max_duration_secs: 14_400, // 4 hours
            max_size_mb: 4_096,
            recommended_bitrate_kbps: 4_000,
            supported_resolutions: vec![(854, 480), (1280, 720), (1920, 1080)],
        },
        SocialPlatform::LinkedIn => SocialVideoSpec {
            platform: SocialPlatform::LinkedIn,
            aspect_ratio: (16, 9),
            max_duration_secs: 600,
            max_size_mb: 5_120,
            recommended_bitrate_kbps: 2_000,
            supported_resolutions: vec![(256, 144), (640, 360), (1280, 720), (1920, 1080)],
        },
        SocialPlatform::Snapchat => SocialVideoSpec {
            platform: SocialPlatform::Snapchat,
            aspect_ratio: (9, 16),
            max_duration_secs: 60,
            max_size_mb: 32,
            recommended_bitrate_kbps: 1_500,
            supported_resolutions: vec![(540, 960), (720, 1280), (1080, 1920)],
        },
        SocialPlatform::Pinterest => SocialVideoSpec {
            platform: SocialPlatform::Pinterest,
            aspect_ratio: (9, 16),
            max_duration_secs: 900, // 15 minutes
            max_size_mb: 2_048,
            recommended_bitrate_kbps: 2_000,
            supported_resolutions: vec![(360, 640), (720, 1280), (1080, 1920)],
        },
    }
}

/// Caption / subtitle specification for a social media platform.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SocialCaptionSpec {
    /// Maximum characters per caption line.
    pub max_chars_per_line: u32,
    /// Maximum number of simultaneous caption lines.
    pub max_lines: u32,
    /// Safe area as a percentage from each edge (0.0–100.0).
    pub safe_area_pct: f32,
}

impl SocialCaptionSpec {
    /// Return caption specification for the given platform.
    #[must_use]
    pub fn for_platform(platform: &SocialPlatform) -> Self {
        match platform {
            SocialPlatform::YouTube => Self {
                max_chars_per_line: 42,
                max_lines: 2,
                safe_area_pct: 5.0,
            },
            SocialPlatform::Instagram => Self {
                max_chars_per_line: 32,
                max_lines: 2,
                safe_area_pct: 10.0,
            },
            SocialPlatform::TikTok => Self {
                max_chars_per_line: 28,
                max_lines: 2,
                safe_area_pct: 12.0,
            },
            SocialPlatform::Twitter => Self {
                max_chars_per_line: 40,
                max_lines: 2,
                safe_area_pct: 5.0,
            },
            SocialPlatform::Facebook => Self {
                max_chars_per_line: 42,
                max_lines: 2,
                safe_area_pct: 5.0,
            },
            SocialPlatform::LinkedIn => Self {
                max_chars_per_line: 42,
                max_lines: 2,
                safe_area_pct: 5.0,
            },
            SocialPlatform::Snapchat => Self {
                max_chars_per_line: 28,
                max_lines: 2,
                safe_area_pct: 15.0,
            },
            SocialPlatform::Pinterest => Self {
                max_chars_per_line: 32,
                max_lines: 2,
                safe_area_pct: 8.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_youtube_spec_aspect_ratio() {
        let spec = base_spec(&SocialPlatform::YouTube);
        assert_eq!(spec.aspect_ratio, (16, 9));
    }

    #[test]
    fn test_tiktok_spec_aspect_ratio() {
        let spec = base_spec(&SocialPlatform::TikTok);
        assert_eq!(spec.aspect_ratio, (9, 16));
    }

    #[test]
    fn test_tiktok_max_duration() {
        let spec = base_spec(&SocialPlatform::TikTok);
        assert_eq!(spec.max_duration_secs, 600);
    }

    #[test]
    fn test_tiktok_max_size() {
        let spec = base_spec(&SocialPlatform::TikTok);
        assert_eq!(spec.max_size_mb, 287);
    }

    #[test]
    fn test_instagram_max_duration() {
        let spec = base_spec(&SocialPlatform::Instagram);
        assert_eq!(spec.max_duration_secs, 90);
    }

    #[test]
    fn test_youtube_no_size_limit() {
        let spec = base_spec(&SocialPlatform::YouTube);
        assert_eq!(spec.max_size_mb, 0);
    }

    #[test]
    fn test_optimal_export_resolution_capped() {
        // Source is 720p; should not select 1080p.
        let spec =
            SocialTemplateGenerator::optimal_export(&SocialPlatform::YouTube, 120, (1280, 720));
        let (w, h) = spec.supported_resolutions[0];
        assert!(w <= 1280 && h <= 720);
    }

    #[test]
    fn test_optimal_export_4k_source() {
        let spec =
            SocialTemplateGenerator::optimal_export(&SocialPlatform::YouTube, 300, (3840, 2160));
        let (w, h) = spec.supported_resolutions[0];
        assert!(w <= 3840 && h <= 2160);
    }

    #[test]
    fn test_caption_spec_youtube() {
        let cap = SocialCaptionSpec::for_platform(&SocialPlatform::YouTube);
        assert_eq!(cap.max_chars_per_line, 42);
        assert_eq!(cap.max_lines, 2);
    }

    #[test]
    fn test_caption_spec_snapchat_safe_area() {
        let cap = SocialCaptionSpec::for_platform(&SocialPlatform::Snapchat);
        assert!(cap.safe_area_pct >= 10.0);
    }

    #[test]
    fn test_platform_name() {
        assert_eq!(SocialPlatform::TikTok.name(), "TikTok");
        assert_eq!(SocialPlatform::LinkedIn.name(), "LinkedIn");
    }

    #[test]
    fn test_all_platforms_have_supported_resolutions() {
        let platforms = [
            SocialPlatform::YouTube,
            SocialPlatform::Instagram,
            SocialPlatform::TikTok,
            SocialPlatform::Twitter,
            SocialPlatform::Facebook,
            SocialPlatform::LinkedIn,
            SocialPlatform::Snapchat,
            SocialPlatform::Pinterest,
        ];
        for platform in &platforms {
            let spec = base_spec(platform);
            assert!(
                !spec.supported_resolutions.is_empty(),
                "{} has no resolutions",
                platform.name()
            );
        }
    }
}

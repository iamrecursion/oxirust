//! OTT (Over-The-Top) platform delivery specifications and presets.
//!
//! This module provides delivery specifications for major OTT streaming platforms,
//! including Netflix, Amazon Prime, Disney+, Apple TV+, HBO Max, Hulu, Peacock,
//! and Paramount+.

/// OTT streaming platform identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum OttPlatform {
    /// Netflix streaming platform.
    Netflix,
    /// Amazon Prime Video.
    AmazonPrime,
    /// Disney+ streaming platform.
    Disney,
    /// Apple TV+ streaming platform.
    AppleTV,
    /// HBO Max streaming platform.
    HboMax,
    /// Hulu streaming platform.
    Hulu,
    /// Peacock streaming platform.
    Peacock,
    /// Paramount+ streaming platform.
    Paramount,
}

impl OttPlatform {
    /// Return the human-readable name of the platform.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Netflix => "Netflix",
            Self::AmazonPrime => "Amazon Prime Video",
            Self::Disney => "Disney+",
            Self::AppleTV => "Apple TV+",
            Self::HboMax => "HBO Max",
            Self::Hulu => "Hulu",
            Self::Peacock => "Peacock",
            Self::Paramount => "Paramount+",
        }
    }
}

/// Delivery specification for an OTT platform.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OttDeliverySpec {
    /// Target platform.
    pub platform: OttPlatform,
    /// Maximum allowed bitrate in kbps.
    pub max_bitrate_kbps: u32,
    /// Minimum required bitrate in kbps.
    pub min_bitrate_kbps: u32,
    /// Primary video codec (e.g. "H.265", "H.264").
    pub codec: String,
    /// Whether HDR is required for delivery.
    pub hdr_required: bool,
    /// Number of audio channels.
    pub audio_channels: u8,
    /// Target loudness in LUFS (EBU R128).
    pub loudness_lufs: f64,
}

/// Library of built-in OTT delivery specifications.
pub struct OttPresetLibrary;

impl OttPresetLibrary {
    /// Return the built-in delivery specification for the given platform.
    #[must_use]
    pub fn get_spec(platform: &OttPlatform) -> OttDeliverySpec {
        match platform {
            OttPlatform::Netflix => OttDeliverySpec {
                platform: OttPlatform::Netflix,
                max_bitrate_kbps: 40_000,
                min_bitrate_kbps: 500,
                codec: "H.265".to_string(),
                hdr_required: false,
                audio_channels: 6,
                loudness_lufs: -27.0,
            },
            OttPlatform::AmazonPrime => OttDeliverySpec {
                platform: OttPlatform::AmazonPrime,
                max_bitrate_kbps: 25_000,
                min_bitrate_kbps: 400,
                codec: "H.265".to_string(),
                hdr_required: false,
                audio_channels: 6,
                loudness_lufs: -24.0,
            },
            OttPlatform::Disney => OttDeliverySpec {
                platform: OttPlatform::Disney,
                max_bitrate_kbps: 20_000,
                min_bitrate_kbps: 600,
                codec: "H.265".to_string(),
                hdr_required: true,
                audio_channels: 6,
                loudness_lufs: -24.0,
            },
            OttPlatform::AppleTV => OttDeliverySpec {
                platform: OttPlatform::AppleTV,
                max_bitrate_kbps: 15_000,
                min_bitrate_kbps: 800,
                codec: "H.265".to_string(),
                hdr_required: true,
                audio_channels: 6,
                loudness_lufs: -24.0,
            },
            OttPlatform::HboMax => OttDeliverySpec {
                platform: OttPlatform::HboMax,
                max_bitrate_kbps: 18_000,
                min_bitrate_kbps: 500,
                codec: "H.265".to_string(),
                hdr_required: false,
                audio_channels: 6,
                loudness_lufs: -24.0,
            },
            OttPlatform::Hulu => OttDeliverySpec {
                platform: OttPlatform::Hulu,
                max_bitrate_kbps: 12_000,
                min_bitrate_kbps: 400,
                codec: "H.264".to_string(),
                hdr_required: false,
                audio_channels: 6,
                loudness_lufs: -24.0,
            },
            OttPlatform::Peacock => OttDeliverySpec {
                platform: OttPlatform::Peacock,
                max_bitrate_kbps: 10_000,
                min_bitrate_kbps: 300,
                codec: "H.264".to_string(),
                hdr_required: false,
                audio_channels: 2,
                loudness_lufs: -24.0,
            },
            OttPlatform::Paramount => OttDeliverySpec {
                platform: OttPlatform::Paramount,
                max_bitrate_kbps: 10_000,
                min_bitrate_kbps: 300,
                codec: "H.264".to_string(),
                hdr_required: false,
                audio_channels: 6,
                loudness_lufs: -24.0,
            },
        }
    }
}

/// A single QC check item in an OTT quality-control checklist.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OttQcCheck {
    /// Name of the check.
    pub name: String,
    /// Whether this check is a hard requirement.
    pub required: bool,
    /// Human-readable description of what is verified.
    pub description: String,
}

/// OTT quality-control checklist for a specific platform.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OttQcChecklist {
    /// Individual checks in the checklist.
    pub checks: Vec<OttQcCheck>,
}

impl OttQcChecklist {
    /// Generate a QC checklist appropriate for the given platform.
    #[must_use]
    pub fn generate_for(platform: &OttPlatform) -> Self {
        let spec = OttPresetLibrary::get_spec(platform);

        let mut checks = vec![
            OttQcCheck {
                name: "Video Codec".to_string(),
                required: true,
                description: format!("Video must be encoded with {}", spec.codec),
            },
            OttQcCheck {
                name: "Max Bitrate".to_string(),
                required: true,
                description: format!(
                    "Video bitrate must not exceed {} kbps",
                    spec.max_bitrate_kbps
                ),
            },
            OttQcCheck {
                name: "Min Bitrate".to_string(),
                required: true,
                description: format!(
                    "Video bitrate must be at least {} kbps",
                    spec.min_bitrate_kbps
                ),
            },
            OttQcCheck {
                name: "Audio Channels".to_string(),
                required: true,
                description: format!("Audio must have {} channels", spec.audio_channels),
            },
            OttQcCheck {
                name: "Loudness".to_string(),
                required: true,
                description: format!(
                    "Integrated loudness must be {:.1} LUFS (EBU R128)",
                    spec.loudness_lufs
                ),
            },
            OttQcCheck {
                name: "True Peak".to_string(),
                required: true,
                description: "True peak level must not exceed -2 dBTP".to_string(),
            },
            OttQcCheck {
                name: "Closed Captions".to_string(),
                required: true,
                description: "Closed captions must be present for all spoken dialogue".to_string(),
            },
            OttQcCheck {
                name: "Frame Rate".to_string(),
                required: true,
                description: "Frame rate must be 23.976, 24, 25, 29.97, or 30 fps".to_string(),
            },
            OttQcCheck {
                name: "Container Format".to_string(),
                required: true,
                description: "Delivery container must be MXF or MOV (platform dependent)"
                    .to_string(),
            },
            OttQcCheck {
                name: "Aspect Ratio".to_string(),
                required: true,
                description: "Pixel aspect ratio must be 1:1 (square pixels)".to_string(),
            },
            OttQcCheck {
                name: "Color Space".to_string(),
                required: true,
                description: "Color space metadata must be accurately signaled".to_string(),
            },
        ];

        if spec.hdr_required {
            checks.push(OttQcCheck {
                name: "HDR Metadata".to_string(),
                required: true,
                description: "HDR metadata (HDR10 or Dolby Vision) must be present".to_string(),
            });
            checks.push(OttQcCheck {
                name: "HDR Peak Brightness".to_string(),
                required: true,
                description: "MaxCLL and MaxFALL values must be provided".to_string(),
            });
        } else {
            checks.push(OttQcCheck {
                name: "HDR Metadata".to_string(),
                required: false,
                description: "HDR metadata is optional but recommended".to_string(),
            });
        }

        checks.push(OttQcCheck {
            name: "Chapter Markers".to_string(),
            required: false,
            description: "Chapter markers are recommended for long-form content".to_string(),
        });

        Self { checks }
    }

    /// Return only the required checks.
    #[must_use]
    pub fn required_checks(&self) -> Vec<&OttQcCheck> {
        self.checks.iter().filter(|c| c.required).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_name_netflix() {
        assert_eq!(OttPlatform::Netflix.name(), "Netflix");
    }

    #[test]
    fn test_platform_name_amazon() {
        assert_eq!(OttPlatform::AmazonPrime.name(), "Amazon Prime Video");
    }

    #[test]
    fn test_platform_name_disney() {
        assert_eq!(OttPlatform::Disney.name(), "Disney+");
    }

    #[test]
    fn test_platform_name_apple() {
        assert_eq!(OttPlatform::AppleTV.name(), "Apple TV+");
    }

    #[test]
    fn test_platform_name_hbo() {
        assert_eq!(OttPlatform::HboMax.name(), "HBO Max");
    }

    #[test]
    fn test_netflix_spec_max_bitrate() {
        let spec = OttPresetLibrary::get_spec(&OttPlatform::Netflix);
        assert_eq!(spec.max_bitrate_kbps, 40_000);
    }

    #[test]
    fn test_netflix_spec_codec() {
        let spec = OttPresetLibrary::get_spec(&OttPlatform::Netflix);
        assert_eq!(spec.codec, "H.265");
    }

    #[test]
    fn test_netflix_spec_loudness() {
        let spec = OttPresetLibrary::get_spec(&OttPlatform::Netflix);
        assert!((spec.loudness_lufs - (-27.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_disney_hdr_required() {
        let spec = OttPresetLibrary::get_spec(&OttPlatform::Disney);
        assert!(spec.hdr_required);
    }

    #[test]
    fn test_apple_tv_max_bitrate() {
        let spec = OttPresetLibrary::get_spec(&OttPlatform::AppleTV);
        assert_eq!(spec.max_bitrate_kbps, 15_000);
    }

    #[test]
    fn test_hulu_codec() {
        let spec = OttPresetLibrary::get_spec(&OttPlatform::Hulu);
        assert_eq!(spec.codec, "H.264");
    }

    #[test]
    fn test_qc_checklist_contains_required_checks() {
        let checklist = OttQcChecklist::generate_for(&OttPlatform::Netflix);
        assert!(!checklist.checks.is_empty());
        let required = checklist.required_checks();
        assert!(!required.is_empty());
    }

    #[test]
    fn test_qc_checklist_disney_hdr_required() {
        let checklist = OttQcChecklist::generate_for(&OttPlatform::Disney);
        let hdr_check = checklist
            .checks
            .iter()
            .find(|c| c.name == "HDR Metadata")
            .expect("HDR Metadata check must exist");
        assert!(hdr_check.required);
    }

    #[test]
    fn test_qc_checklist_hulu_hdr_optional() {
        let checklist = OttQcChecklist::generate_for(&OttPlatform::Hulu);
        let hdr_check = checklist
            .checks
            .iter()
            .find(|c| c.name == "HDR Metadata")
            .expect("HDR Metadata check must exist");
        assert!(!hdr_check.required);
    }

    #[test]
    fn test_all_platforms_have_specs() {
        let platforms = [
            OttPlatform::Netflix,
            OttPlatform::AmazonPrime,
            OttPlatform::Disney,
            OttPlatform::AppleTV,
            OttPlatform::HboMax,
            OttPlatform::Hulu,
            OttPlatform::Peacock,
            OttPlatform::Paramount,
        ];
        for platform in &platforms {
            let spec = OttPresetLibrary::get_spec(platform);
            assert!(spec.max_bitrate_kbps > 0);
            assert!(!spec.codec.is_empty());
        }
    }

    #[test]
    fn test_peacock_audio_channels() {
        let spec = OttPresetLibrary::get_spec(&OttPlatform::Peacock);
        assert_eq!(spec.audio_channels, 2);
    }
}

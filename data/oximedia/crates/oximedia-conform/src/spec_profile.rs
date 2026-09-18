//! Broadcast specification profiles.
//!
//! Defines broadcast and streaming platform spec profiles for Netflix,
//! Amazon Prime, Apple TV+, and traditional broadcast standards.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// Target platform for a broadcast specification profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetPlatform {
    /// Netflix streaming platform
    Netflix,
    /// Amazon Prime Video
    AmazonPrime,
    /// Apple TV+ streaming platform
    AppleTvPlus,
    /// Disney+ streaming platform
    DisneyPlus,
    /// Traditional broadcast (generic)
    Broadcast,
    /// EBU broadcast standard (Europe)
    EbuBroadcast,
    /// ATSC broadcast standard (North America)
    AtscBroadcast,
    /// Custom / user-defined
    Custom(u32),
}

/// Allowed video resolution for a spec profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resolution {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

impl Resolution {
    /// Create a new resolution.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// SD 720×480 (NTSC)
    pub const SD_NTSC: Self = Self::new(720, 480);
    /// SD 720×576 (PAL)
    pub const SD_PAL: Self = Self::new(720, 576);
    /// HD 1280×720
    pub const HD_720P: Self = Self::new(1280, 720);
    /// HD 1920×1080
    pub const HD_1080P: Self = Self::new(1920, 1080);
    /// UHD 3840×2160 (4K)
    pub const UHD_4K: Self = Self::new(3840, 2160);
    /// UHD 7680×4320 (8K)
    pub const UHD_8K: Self = Self::new(7680, 4320);

    /// Returns pixel count.
    #[must_use]
    pub const fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Returns aspect ratio as a float.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// Video frame rate specification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameRateSpec {
    /// Numerator of the frame rate
    pub numerator: u32,
    /// Denominator of the frame rate
    pub denominator: u32,
}

impl FrameRateSpec {
    /// Create a new frame rate spec.
    #[must_use]
    pub const fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// 23.976 fps (24000/1001)
    pub const FPS_23976: Self = Self::new(24000, 1001);
    /// 24 fps
    pub const FPS_24: Self = Self::new(24, 1);
    /// 25 fps (PAL)
    pub const FPS_25: Self = Self::new(25, 1);
    /// 29.97 fps (NTSC)
    pub const FPS_2997: Self = Self::new(30000, 1001);
    /// 30 fps
    pub const FPS_30: Self = Self::new(30, 1);
    /// 50 fps
    pub const FPS_50: Self = Self::new(50, 1);
    /// 59.94 fps
    pub const FPS_5994: Self = Self::new(60000, 1001);
    /// 60 fps
    pub const FPS_60: Self = Self::new(60, 1);

    /// Returns frame rate as f64.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }
}

/// Audio loudness requirements for a spec profile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LoudnessRequirements {
    /// Target integrated loudness (LKFS/LUFS)
    pub integrated_loudness_target: f32,
    /// Tolerance around the target (±LUFS)
    pub integrated_loudness_tolerance: f32,
    /// Maximum true peak level (dBTP)
    pub max_true_peak: f32,
    /// Maximum short-term loudness (LUFS)
    pub max_short_term: Option<f32>,
    /// Maximum loudness range (LU)
    pub max_loudness_range: Option<f32>,
}

impl LoudnessRequirements {
    /// EBU R128 loudness requirements (-23 LUFS).
    #[must_use]
    pub const fn ebu_r128() -> Self {
        Self {
            integrated_loudness_target: -23.0,
            integrated_loudness_tolerance: 1.0,
            max_true_peak: -1.0,
            max_short_term: None,
            max_loudness_range: None,
        }
    }

    /// ATSC A/85 loudness requirements (-24 LKFS).
    #[must_use]
    pub const fn atsc_a85() -> Self {
        Self {
            integrated_loudness_target: -24.0,
            integrated_loudness_tolerance: 2.0,
            max_true_peak: -2.0,
            max_short_term: None,
            max_loudness_range: None,
        }
    }

    /// Netflix loudness requirements (-27 LUFS).
    #[must_use]
    pub const fn netflix() -> Self {
        Self {
            integrated_loudness_target: -27.0,
            integrated_loudness_tolerance: 1.0,
            max_true_peak: -2.0,
            max_short_term: Some(-20.0),
            max_loudness_range: None,
        }
    }
}

/// Video codec profile allowed in a broadcast spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllowedCodec {
    /// H.264 / AVC
    H264,
    /// H.265 / HEVC
    H265,
    /// AV1
    Av1,
    /// Apple `ProRes` (various flavors)
    ProRes,
    /// `DNxHD` / `DNxHR`
    DnxHd,
    /// IMX / D-10
    Imx,
    /// XAVC
    Xavc,
}

/// Color space requirements for a spec profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorRequirements {
    /// Required color primaries (e.g., "BT.709", "BT.2020")
    pub primaries: String,
    /// Required transfer characteristics (e.g., "BT.709", "PQ", "HLG")
    pub transfer: String,
    /// Required matrix coefficients (e.g., "BT.709", "BT.2020nc")
    pub matrix: String,
    /// Whether HDR is allowed
    pub hdr_allowed: bool,
    /// Minimum bit depth
    pub min_bit_depth: u8,
}

impl ColorRequirements {
    /// Standard HD Rec.709 color requirements.
    #[must_use]
    pub fn hd_rec709() -> Self {
        Self {
            primaries: "BT.709".to_string(),
            transfer: "BT.709".to_string(),
            matrix: "BT.709".to_string(),
            hdr_allowed: false,
            min_bit_depth: 8,
        }
    }

    /// UHD Rec.2020 HDR (PQ) color requirements.
    #[must_use]
    pub fn uhd_hdr_pq() -> Self {
        Self {
            primaries: "BT.2020".to_string(),
            transfer: "PQ".to_string(),
            matrix: "BT.2020nc".to_string(),
            hdr_allowed: true,
            min_bit_depth: 10,
        }
    }
}

/// A complete broadcast specification profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecProfile {
    /// Profile identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Target platform
    pub platform: TargetPlatform,
    /// Allowed resolutions (empty = any)
    pub allowed_resolutions: Vec<Resolution>,
    /// Allowed frame rates (empty = any)
    pub allowed_frame_rates: Vec<FrameRateSpec>,
    /// Loudness requirements
    pub loudness: LoudnessRequirements,
    /// Allowed video codecs
    pub allowed_codecs: Vec<AllowedCodec>,
    /// Color requirements
    pub color: ColorRequirements,
    /// Maximum video bitrate in Mbps (None = unlimited)
    pub max_video_bitrate_mbps: Option<f32>,
    /// Required container formats
    pub container_formats: Vec<String>,
}

impl SpecProfile {
    /// Creates a Netflix HD spec profile.
    #[must_use]
    pub fn netflix_hd() -> Self {
        Self {
            id: "netflix-hd".to_string(),
            name: "Netflix HD Delivery".to_string(),
            platform: TargetPlatform::Netflix,
            allowed_resolutions: vec![Resolution::HD_1080P, Resolution::HD_720P],
            allowed_frame_rates: vec![
                FrameRateSpec::FPS_23976,
                FrameRateSpec::FPS_24,
                FrameRateSpec::FPS_25,
                FrameRateSpec::FPS_2997,
            ],
            loudness: LoudnessRequirements::netflix(),
            allowed_codecs: vec![AllowedCodec::H264, AllowedCodec::ProRes],
            color: ColorRequirements::hd_rec709(),
            max_video_bitrate_mbps: Some(40.0),
            container_formats: vec!["mov".to_string(), "mxf".to_string()],
        }
    }

    /// Creates a Netflix 4K HDR spec profile.
    #[must_use]
    pub fn netflix_4k_hdr() -> Self {
        Self {
            id: "netflix-4k-hdr".to_string(),
            name: "Netflix 4K HDR Delivery".to_string(),
            platform: TargetPlatform::Netflix,
            allowed_resolutions: vec![Resolution::UHD_4K],
            allowed_frame_rates: vec![
                FrameRateSpec::FPS_23976,
                FrameRateSpec::FPS_24,
                FrameRateSpec::FPS_25,
            ],
            loudness: LoudnessRequirements::netflix(),
            allowed_codecs: vec![AllowedCodec::H265, AllowedCodec::Av1],
            color: ColorRequirements::uhd_hdr_pq(),
            max_video_bitrate_mbps: Some(80.0),
            container_formats: vec!["mov".to_string()],
        }
    }

    /// Creates an EBU broadcast spec profile.
    #[must_use]
    pub fn ebu_broadcast() -> Self {
        Self {
            id: "ebu-broadcast".to_string(),
            name: "EBU Broadcast Standard".to_string(),
            platform: TargetPlatform::EbuBroadcast,
            allowed_resolutions: vec![
                Resolution::HD_1080P,
                Resolution::HD_720P,
                Resolution::SD_PAL,
            ],
            allowed_frame_rates: vec![FrameRateSpec::FPS_25, FrameRateSpec::FPS_50],
            loudness: LoudnessRequirements::ebu_r128(),
            allowed_codecs: vec![
                AllowedCodec::H264,
                AllowedCodec::H265,
                AllowedCodec::DnxHd,
                AllowedCodec::Imx,
            ],
            color: ColorRequirements::hd_rec709(),
            max_video_bitrate_mbps: Some(50.0),
            container_formats: vec!["mxf".to_string()],
        }
    }

    /// Validates a set of properties against this profile.
    ///
    /// Returns a list of violations. An empty list means the content conforms.
    #[must_use]
    pub fn validate(
        &self,
        resolution: Resolution,
        frame_rate: FrameRateSpec,
        integrated_loudness: f32,
        true_peak: f32,
        codec: &AllowedCodec,
        bitrate_mbps: f32,
    ) -> Vec<SpecViolation> {
        let mut violations = Vec::new();

        if !self.allowed_resolutions.is_empty() && !self.allowed_resolutions.contains(&resolution) {
            violations.push(SpecViolation::ResolutionMismatch {
                expected: self.allowed_resolutions.clone(),
                actual: resolution,
            });
        }

        if !self.allowed_frame_rates.is_empty() && !self.allowed_frame_rates.contains(&frame_rate) {
            violations.push(SpecViolation::FrameRateMismatch {
                expected: self.allowed_frame_rates.clone(),
                actual: frame_rate,
            });
        }

        let loudness_diff = (integrated_loudness - self.loudness.integrated_loudness_target).abs();
        if loudness_diff > self.loudness.integrated_loudness_tolerance {
            violations.push(SpecViolation::LoudnessOutOfRange {
                target: self.loudness.integrated_loudness_target,
                tolerance: self.loudness.integrated_loudness_tolerance,
                actual: integrated_loudness,
            });
        }

        if true_peak > self.loudness.max_true_peak {
            violations.push(SpecViolation::TruePeakExceeded {
                limit: self.loudness.max_true_peak,
                actual: true_peak,
            });
        }

        if !self.allowed_codecs.is_empty() && !self.allowed_codecs.contains(codec) {
            violations.push(SpecViolation::CodecNotAllowed {
                codec: format!("{codec:?}"),
            });
        }

        if let Some(limit) = self.max_video_bitrate_mbps {
            if bitrate_mbps > limit {
                violations.push(SpecViolation::BitrateTooHigh {
                    limit,
                    actual: bitrate_mbps,
                });
            }
        }

        violations
    }
}

/// A violation of a broadcast spec profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecViolation {
    /// Resolution does not match any allowed resolution.
    ResolutionMismatch {
        /// Allowed resolutions
        expected: Vec<Resolution>,
        /// Actual resolution
        actual: Resolution,
    },
    /// Frame rate does not match any allowed frame rate.
    FrameRateMismatch {
        /// Allowed frame rates
        expected: Vec<FrameRateSpec>,
        /// Actual frame rate
        actual: FrameRateSpec,
    },
    /// Integrated loudness is out of range.
    LoudnessOutOfRange {
        /// Target loudness
        target: f32,
        /// Tolerance
        tolerance: f32,
        /// Actual loudness
        actual: f32,
    },
    /// True peak exceeds limit.
    TruePeakExceeded {
        /// Limit in dBTP
        limit: f32,
        /// Actual value
        actual: f32,
    },
    /// Codec is not allowed by the profile.
    CodecNotAllowed {
        /// Codec name
        codec: String,
    },
    /// Video bitrate exceeds the profile limit.
    BitrateTooHigh {
        /// Limit in Mbps
        limit: f32,
        /// Actual bitrate
        actual: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_pixels() {
        assert_eq!(Resolution::HD_1080P.pixels(), 1920 * 1080);
        assert_eq!(Resolution::UHD_4K.pixels(), 3840 * 2160);
    }

    #[test]
    fn test_resolution_aspect_ratio() {
        let ar = Resolution::HD_1080P.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_frame_rate_as_f64() {
        let fps = FrameRateSpec::FPS_23976.as_f64();
        assert!((fps - 23.976).abs() < 0.001);

        let fps25 = FrameRateSpec::FPS_25.as_f64();
        assert!((fps25 - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_loudness_requirements_ebu() {
        let req = LoudnessRequirements::ebu_r128();
        assert!((req.integrated_loudness_target - (-23.0)).abs() < 1e-6);
        assert!((req.max_true_peak - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_loudness_requirements_netflix() {
        let req = LoudnessRequirements::netflix();
        assert!((req.integrated_loudness_target - (-27.0)).abs() < 1e-6);
        assert_eq!(req.max_short_term, Some(-20.0));
    }

    #[test]
    fn test_color_requirements_hd() {
        let col = ColorRequirements::hd_rec709();
        assert_eq!(col.primaries, "BT.709");
        assert!(!col.hdr_allowed);
        assert_eq!(col.min_bit_depth, 8);
    }

    #[test]
    fn test_color_requirements_hdr() {
        let col = ColorRequirements::uhd_hdr_pq();
        assert_eq!(col.transfer, "PQ");
        assert!(col.hdr_allowed);
        assert_eq!(col.min_bit_depth, 10);
    }

    #[test]
    fn test_netflix_hd_profile_creation() {
        let profile = SpecProfile::netflix_hd();
        assert_eq!(profile.platform, TargetPlatform::Netflix);
        assert!(profile.allowed_resolutions.contains(&Resolution::HD_1080P));
        assert!(!profile.allowed_codecs.is_empty());
    }

    #[test]
    fn test_ebu_broadcast_profile_creation() {
        let profile = SpecProfile::ebu_broadcast();
        assert_eq!(profile.platform, TargetPlatform::EbuBroadcast);
        assert!(profile.allowed_frame_rates.contains(&FrameRateSpec::FPS_25));
    }

    #[test]
    fn test_validate_conforming_content() {
        let profile = SpecProfile::netflix_hd();
        let violations = profile.validate(
            Resolution::HD_1080P,
            FrameRateSpec::FPS_23976,
            -27.0,
            -3.0,
            &AllowedCodec::H264,
            20.0,
        );
        assert!(
            violations.is_empty(),
            "expected no violations, got: {violations:?}"
        );
    }

    #[test]
    fn test_validate_resolution_violation() {
        let profile = SpecProfile::netflix_hd();
        let violations = profile.validate(
            Resolution::SD_PAL,
            FrameRateSpec::FPS_25,
            -27.0,
            -3.0,
            &AllowedCodec::H264,
            20.0,
        );
        let has_res_viol = violations
            .iter()
            .any(|v| matches!(v, SpecViolation::ResolutionMismatch { .. }));
        assert!(has_res_viol);
    }

    #[test]
    fn test_validate_loudness_violation() {
        let profile = SpecProfile::netflix_hd();
        let violations = profile.validate(
            Resolution::HD_1080P,
            FrameRateSpec::FPS_23976,
            -18.0, // too loud
            -3.0,
            &AllowedCodec::H264,
            20.0,
        );
        let has_loudness = violations
            .iter()
            .any(|v| matches!(v, SpecViolation::LoudnessOutOfRange { .. }));
        assert!(has_loudness);
    }

    #[test]
    fn test_validate_true_peak_violation() {
        let profile = SpecProfile::netflix_hd();
        let violations = profile.validate(
            Resolution::HD_1080P,
            FrameRateSpec::FPS_23976,
            -27.0,
            -0.5, // too high
            &AllowedCodec::H264,
            20.0,
        );
        let has_peak = violations
            .iter()
            .any(|v| matches!(v, SpecViolation::TruePeakExceeded { .. }));
        assert!(has_peak);
    }

    #[test]
    fn test_validate_bitrate_violation() {
        let profile = SpecProfile::netflix_hd();
        let violations = profile.validate(
            Resolution::HD_1080P,
            FrameRateSpec::FPS_23976,
            -27.0,
            -3.0,
            &AllowedCodec::H264,
            50.0, // too high
        );
        let has_bitrate = violations
            .iter()
            .any(|v| matches!(v, SpecViolation::BitrateTooHigh { .. }));
        assert!(has_bitrate);
    }

    #[test]
    fn test_netflix_4k_hdr_profile() {
        let profile = SpecProfile::netflix_4k_hdr();
        assert!(profile.color.hdr_allowed);
        assert!(profile.allowed_resolutions.contains(&Resolution::UHD_4K));
        assert!(!profile.allowed_resolutions.contains(&Resolution::HD_1080P));
    }
}

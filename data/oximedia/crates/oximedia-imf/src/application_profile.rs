//! IMF Application Profile support - SMPTE ST 2067-21 and related standards
//!
//! Application profiles constrain the IMF package to a specific set of
//! allowed essence types and parameters for a given delivery platform.
//!
//! This module provides two levels of application profile:
//! - [`ApplicationProfile`]: SMPTE standard application profiles (App2, App4, App7, etc.)
//! - [`ImfApplicationProfile`]: Vendor/platform-specific delivery profiles (Netflix App 2.1,
//!   Disney DECE, etc.) which impose additional constraints on top of the SMPTE baseline.

use std::collections::HashMap;

/// Vendor-specific IMF delivery profiles with platform constraints.
///
/// These profiles represent delivery requirements imposed by major platforms on
/// top of the underlying SMPTE App2/App7 baseline. Each platform defines
/// allowed codecs, maximum resolutions, audio channel limits, and HDR metadata
/// requirements.
///
/// # Specifications
/// - Netflix App 2 / App 2E: <https://partnerhelp.netflixstudios.com/hc/en-us/articles/360000579368>
/// - Netflix App 2.1: adds JPEG XL and 8K support (internal spec rev 2024-Q3)
/// - Disney DECE: DECE (Digital Entertainment Content Ecosystem) SD/HD profile
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImfApplicationProfile {
    /// Netflix IMF Application #2 — JPEG 2000 up to 4K, 16-ch audio.
    NetflixApp2,
    /// Netflix IMF Application #2 Extended — extended bit-depth constraints.
    NetflixApp2E,
    /// Netflix IMF Application #2.1 — adds JPEG XL, 8K (7680×4320), Dolby Vision RPU.
    NetflixApp2_1,
    /// Disney DECE (Digital Entertainment Content Ecosystem) — HD/SD profile.
    DisneyDece,
    /// Sony IMF Application profile.
    SonyApp,
    /// Warner Bros. IMF Application profile.
    WbApp,
    /// Custom vendor profile identified by a URN string.
    Custom(String),
}

impl ImfApplicationProfile {
    /// Returns the maximum allowed picture resolution `(width, height)`.
    ///
    /// # Examples
    /// ```
    /// use oximedia_imf::application_profile::ImfApplicationProfile;
    /// assert_eq!(ImfApplicationProfile::NetflixApp2_1.max_resolution(), (7680, 4320));
    /// ```
    #[must_use]
    pub fn max_resolution(&self) -> (u32, u32) {
        match self {
            Self::NetflixApp2_1 => (7680, 4320),
            Self::NetflixApp2 | Self::NetflixApp2E => (3840, 2160),
            Self::DisneyDece => (1920, 1080),
            Self::SonyApp => (3840, 2160),
            Self::WbApp => (3840, 2160),
            Self::Custom(_) => (u32::MAX, u32::MAX),
        }
    }

    /// Returns `true` if this profile permits JPEG XL (ISO 18181) as the picture codec.
    ///
    /// Netflix App 2.1 is the first major delivery profile to allow JPEG XL alongside
    /// the traditional JPEG 2000. Earlier profiles (App 2, App 2E) require JPEG 2000.
    ///
    /// # Examples
    /// ```
    /// use oximedia_imf::application_profile::ImfApplicationProfile;
    /// assert!(ImfApplicationProfile::NetflixApp2_1.allows_jpeg_xl());
    /// assert!(!ImfApplicationProfile::NetflixApp2.allows_jpeg_xl());
    /// ```
    #[must_use]
    pub fn allows_jpeg_xl(&self) -> bool {
        matches!(self, Self::NetflixApp2_1)
    }

    /// Returns `true` if HDR content in this profile requires a Dolby Vision RPU track.
    ///
    /// Netflix App 2.1 mandates that any HDR deliverable must carry a Dolby Vision
    /// Reference Processing Unit (RPU) metadata track in addition to the base HDR10
    /// static metadata.
    ///
    /// # Examples
    /// ```
    /// use oximedia_imf::application_profile::ImfApplicationProfile;
    /// assert!(ImfApplicationProfile::NetflixApp2_1.requires_dolby_vision_rpu_for_hdr());
    /// assert!(!ImfApplicationProfile::DisneyDece.requires_dolby_vision_rpu_for_hdr());
    /// ```
    #[must_use]
    pub fn requires_dolby_vision_rpu_for_hdr(&self) -> bool {
        matches!(self, Self::NetflixApp2_1)
    }

    /// Returns the maximum number of discrete audio channels permitted.
    ///
    /// # Examples
    /// ```
    /// use oximedia_imf::application_profile::ImfApplicationProfile;
    /// assert_eq!(ImfApplicationProfile::NetflixApp2.max_audio_channels(), 16);
    /// assert_eq!(ImfApplicationProfile::DisneyDece.max_audio_channels(), 8);
    /// ```
    #[must_use]
    pub fn max_audio_channels(&self) -> u32 {
        match self {
            Self::NetflixApp2 | Self::NetflixApp2E | Self::NetflixApp2_1 => 16,
            Self::DisneyDece => 8,
            Self::SonyApp => 16,
            Self::WbApp => 16,
            Self::Custom(_) => u32::MAX,
        }
    }

    /// Returns a human-readable label for this profile.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::NetflixApp2 => "Netflix IMF App 2",
            Self::NetflixApp2E => "Netflix IMF App 2E",
            Self::NetflixApp2_1 => "Netflix IMF App 2.1",
            Self::DisneyDece => "Disney DECE",
            Self::SonyApp => "Sony IMF App",
            Self::WbApp => "Warner Bros. IMF App",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Returns `true` if this profile supports 16-bit picture bit depth.
    ///
    /// Netflix App 2.1 explicitly adds support for 16-bit depth in addition
    /// to the standard 8- and 12-bit depths of App 2.
    #[must_use]
    pub fn supports_16bit_depth(&self) -> bool {
        matches!(self, Self::NetflixApp2_1)
    }
}

/// IMF Application Profile identifiers per SMPTE ST 2067-x
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ApplicationProfile {
    /// App #2 - IMF Application #2 (JPEG 2000, SMPTE ST 2067-21)
    App2,
    /// App #2 Extended - JPEG 2000 extended constraints
    App2Extended,
    /// App #4 - IMF ACES Application (ACES workflow)
    App4,
    /// App #5 ACES - Academy Color Encoding System
    App5Aces,
    /// App #6 - IAB (Immersive Audio Bitstream)
    App6,
    /// App #7 - JPEG XS essence
    App7,
    /// IABMM - Immersive Audio Bitstream for Mastering and Mezzanine
    Iabmm,
}

impl ApplicationProfile {
    /// Return the URN identifier for this application profile
    #[must_use]
    pub fn urn(&self) -> &str {
        match self {
            Self::App2 => "urn:smpte:ul:060E2B34.04010105.0E090604.00000000",
            Self::App2Extended => "urn:smpte:ul:060E2B34.04010105.0E090605.00000000",
            Self::App4 => "urn:smpte:ul:060E2B34.04010105.0E090606.00000000",
            Self::App5Aces => "urn:smpte:ul:060E2B34.04010105.0E090607.00000000",
            Self::App6 => "urn:smpte:ul:060E2B34.04010105.0E090608.00000000",
            Self::App7 => "urn:smpte:ul:060E2B34.04010105.0E090609.00000000",
            Self::Iabmm => "urn:smpte:ul:060E2B34.04010105.0E09060A.00000000",
        }
    }
}

/// Constraints for IMF Application #2 per SMPTE ST 2067-21
#[derive(Debug, Clone)]
pub struct App2Constraints {
    /// Maximum allowed resolution (width, height)
    pub max_resolution: (u32, u32),
    /// Maximum allowed frame rate as (numerator, denominator)
    pub max_frame_rate: (u32, u32),
    /// Maximum number of audio channels
    pub audio_channel_count: u32,
}

impl App2Constraints {
    /// ST 2067-21 App #2 default constraints
    #[must_use]
    pub fn st2067_21() -> Self {
        Self {
            max_resolution: (3840, 2160),
            max_frame_rate: (60, 1),
            audio_channel_count: 16,
        }
    }
}

/// Essence types allowed in IMF packages
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EssenceType {
    /// MXF-wrapped JPEG 2000
    MxfJpeg2000,
    /// MXF-wrapped Apple ProRes
    MxfProRes,
    /// MXF-wrapped AVC (H.264)
    MxfAvc,
    /// MXF-wrapped IAB (Immersive Audio Bitstream)
    MxfIab,
    /// Timed Text SRT subtitle
    TtSrt,
    /// Timed Text IMSC1 (TTML-based)
    TtImsc,
}

impl EssenceType {
    /// Check if this essence type is allowed in the given application profile
    #[must_use]
    pub fn allowed_in(&self, profile: &ApplicationProfile) -> bool {
        match profile {
            ApplicationProfile::App2 | ApplicationProfile::App2Extended => {
                matches!(self, Self::MxfJpeg2000 | Self::TtSrt | Self::TtImsc)
            }
            ApplicationProfile::App4 | ApplicationProfile::App5Aces => {
                matches!(self, Self::MxfJpeg2000 | Self::TtImsc)
            }
            ApplicationProfile::App6 | ApplicationProfile::Iabmm => {
                matches!(
                    self,
                    Self::MxfJpeg2000 | Self::MxfIab | Self::TtSrt | Self::TtImsc
                )
            }
            ApplicationProfile::App7 => {
                matches!(self, Self::MxfProRes | Self::MxfAvc | Self::TtImsc)
            }
        }
    }
}

/// Validates CPL essence types against an application profile
#[derive(Debug, Default)]
pub struct ProfileValidator;

impl ProfileValidator {
    /// Validate a list of CPL essence type strings against the given profile.
    ///
    /// Returns a list of validation error messages (empty = valid).
    #[must_use]
    pub fn validate_for_profile(
        cpl_essence_types: &[&str],
        profile: &ApplicationProfile,
    ) -> Vec<String> {
        let mut errors = Vec::new();

        for essence_type_str in cpl_essence_types {
            let essence = parse_essence_type(essence_type_str);
            match essence {
                Some(et) => {
                    if !et.allowed_in(profile) {
                        errors.push(format!(
                            "Essence type '{}' is not allowed in profile '{}'",
                            essence_type_str,
                            profile.urn()
                        ));
                    }
                }
                None => {
                    errors.push(format!("Unknown essence type: '{essence_type_str}'"));
                }
            }
        }

        errors
    }
}

/// Parse an essence type string into an `EssenceType`
fn parse_essence_type(s: &str) -> Option<EssenceType> {
    match s {
        "MxfJpeg2000" | "application/mxf+jpeg2000" => Some(EssenceType::MxfJpeg2000),
        "MxfProRes" | "application/mxf+prores" => Some(EssenceType::MxfProRes),
        "MxfAvc" | "application/mxf+avc" => Some(EssenceType::MxfAvc),
        "MxfIab" | "application/mxf+iab" => Some(EssenceType::MxfIab),
        "TtSrt" | "text/srt" => Some(EssenceType::TtSrt),
        "TtImsc" | "application/ttml+xml" => Some(EssenceType::TtImsc),
        _ => None,
    }
}

/// Metadata describing an IMF application profile instance
#[derive(Debug, Clone)]
pub struct ApplicationMetadata {
    /// The application profile
    pub profile: ApplicationProfile,
    /// Version string (e.g. "1.0")
    pub version: String,
    /// Arbitrary key-value constraint metadata
    pub constraints: HashMap<String, String>,
}

impl ApplicationMetadata {
    /// Create new application metadata for the given profile and version
    #[must_use]
    pub fn new(profile: ApplicationProfile, version: String) -> Self {
        Self {
            profile,
            version,
            constraints: HashMap::new(),
        }
    }

    /// Add a constraint key-value pair
    pub fn add_constraint(&mut self, key: String, value: String) {
        self.constraints.insert(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_application_profile_urn() {
        let profile = ApplicationProfile::App2;
        assert!(profile.urn().starts_with("urn:smpte:ul:"));
    }

    #[test]
    fn test_all_profiles_have_distinct_urns() {
        let profiles = [
            ApplicationProfile::App2,
            ApplicationProfile::App2Extended,
            ApplicationProfile::App4,
            ApplicationProfile::App5Aces,
            ApplicationProfile::App6,
            ApplicationProfile::App7,
            ApplicationProfile::Iabmm,
        ];
        let urns: Vec<&str> = profiles.iter().map(ApplicationProfile::urn).collect();
        let unique: std::collections::HashSet<&&str> = urns.iter().collect();
        assert_eq!(unique.len(), urns.len(), "All URNs must be unique");
    }

    #[test]
    fn test_app2_constraints() {
        let c = App2Constraints::st2067_21();
        assert_eq!(c.max_resolution, (3840, 2160));
        assert_eq!(c.max_frame_rate, (60, 1));
        assert_eq!(c.audio_channel_count, 16);
    }

    #[test]
    fn test_essence_allowed_in_app2() {
        let profile = ApplicationProfile::App2;
        assert!(EssenceType::MxfJpeg2000.allowed_in(&profile));
        assert!(EssenceType::TtSrt.allowed_in(&profile));
        assert!(EssenceType::TtImsc.allowed_in(&profile));
        assert!(!EssenceType::MxfProRes.allowed_in(&profile));
        assert!(!EssenceType::MxfAvc.allowed_in(&profile));
        assert!(!EssenceType::MxfIab.allowed_in(&profile));
    }

    #[test]
    fn test_essence_allowed_in_app7() {
        let profile = ApplicationProfile::App7;
        assert!(!EssenceType::MxfJpeg2000.allowed_in(&profile));
        assert!(EssenceType::MxfProRes.allowed_in(&profile));
        assert!(EssenceType::MxfAvc.allowed_in(&profile));
        assert!(EssenceType::TtImsc.allowed_in(&profile));
        assert!(!EssenceType::TtSrt.allowed_in(&profile));
    }

    #[test]
    fn test_essence_allowed_in_iabmm() {
        let profile = ApplicationProfile::Iabmm;
        assert!(EssenceType::MxfIab.allowed_in(&profile));
        assert!(EssenceType::MxfJpeg2000.allowed_in(&profile));
    }

    #[test]
    fn test_profile_validator_valid() {
        let types = ["MxfJpeg2000", "TtImsc"];
        let errors = ProfileValidator::validate_for_profile(&types, &ApplicationProfile::App2);
        assert!(errors.is_empty(), "Expected no errors, got: {errors:?}");
    }

    #[test]
    fn test_profile_validator_invalid_essence() {
        let types = ["MxfProRes"];
        let errors = ProfileValidator::validate_for_profile(&types, &ApplicationProfile::App2);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("not allowed"));
    }

    #[test]
    fn test_profile_validator_unknown_essence() {
        let types = ["SomeWeirdFormat"];
        let errors = ProfileValidator::validate_for_profile(&types, &ApplicationProfile::App2);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("Unknown"));
    }

    #[test]
    fn test_application_metadata_creation() {
        let mut meta = ApplicationMetadata::new(ApplicationProfile::App2, "1.0".to_string());
        meta.add_constraint("max_bits".to_string(), "12".to_string());
        assert_eq!(meta.version, "1.0");
        assert_eq!(
            meta.constraints.get("max_bits").map(String::as_str),
            Some("12")
        );
    }

    #[test]
    fn test_application_metadata_constraints_map() {
        let mut meta = ApplicationMetadata::new(ApplicationProfile::App4, "2.0".to_string());
        meta.add_constraint("color_space".to_string(), "ACES".to_string());
        meta.add_constraint("bit_depth".to_string(), "16".to_string());
        assert_eq!(meta.constraints.len(), 2);
    }

    #[test]
    fn test_parse_essence_type_round_trip() {
        assert_eq!(
            parse_essence_type("MxfJpeg2000"),
            Some(EssenceType::MxfJpeg2000)
        );
        assert_eq!(
            parse_essence_type("application/mxf+jpeg2000"),
            Some(EssenceType::MxfJpeg2000)
        );
        assert_eq!(parse_essence_type("unknown_type"), None);
    }

    // ---- ImfApplicationProfile tests ----

    #[test]
    fn test_netflix_app_2_1_allows_jpeg_xl() {
        assert!(
            ImfApplicationProfile::NetflixApp2_1.allows_jpeg_xl(),
            "Netflix App 2.1 must allow JPEG XL"
        );
    }

    #[test]
    fn test_netflix_app_2_jpeg_xl() {
        assert!(
            !ImfApplicationProfile::NetflixApp2.allows_jpeg_xl(),
            "Netflix App 2 must NOT allow JPEG XL (JPEG 2000 only)"
        );
        assert!(
            !ImfApplicationProfile::NetflixApp2E.allows_jpeg_xl(),
            "Netflix App 2E must NOT allow JPEG XL"
        );
    }

    #[test]
    fn test_disney_dece_max_resolution() {
        // DECE HD/SD profile caps at 1920×1080
        assert_eq!(
            ImfApplicationProfile::DisneyDece.max_resolution(),
            (1920, 1080),
            "Disney DECE max resolution must be 1920×1080"
        );
    }

    #[test]
    fn test_profile_hdr_requirement() {
        assert!(
            ImfApplicationProfile::NetflixApp2_1.requires_dolby_vision_rpu_for_hdr(),
            "Netflix App 2.1 must require Dolby Vision RPU for HDR"
        );
        assert!(
            !ImfApplicationProfile::NetflixApp2.requires_dolby_vision_rpu_for_hdr(),
            "Netflix App 2 does not require DV RPU"
        );
        assert!(
            !ImfApplicationProfile::DisneyDece.requires_dolby_vision_rpu_for_hdr(),
            "Disney DECE does not require DV RPU"
        );
    }

    #[test]
    fn test_netflix_app_2_1_max_resolution() {
        assert_eq!(
            ImfApplicationProfile::NetflixApp2_1.max_resolution(),
            (7680, 4320),
            "Netflix App 2.1 must support 8K resolution"
        );
    }

    #[test]
    fn test_netflix_app_2_max_resolution() {
        assert_eq!(
            ImfApplicationProfile::NetflixApp2.max_resolution(),
            (3840, 2160),
            "Netflix App 2 max resolution must be 4K"
        );
    }

    #[test]
    fn test_audio_channels() {
        assert_eq!(ImfApplicationProfile::NetflixApp2.max_audio_channels(), 16);
        assert_eq!(
            ImfApplicationProfile::NetflixApp2_1.max_audio_channels(),
            16
        );
        assert_eq!(ImfApplicationProfile::DisneyDece.max_audio_channels(), 8);
    }

    #[test]
    fn test_netflix_app_2_1_supports_16bit() {
        assert!(
            ImfApplicationProfile::NetflixApp2_1.supports_16bit_depth(),
            "Netflix App 2.1 must support 16-bit depth"
        );
        assert!(
            !ImfApplicationProfile::NetflixApp2.supports_16bit_depth(),
            "Netflix App 2 does not support 16-bit depth"
        );
    }

    #[test]
    fn test_custom_profile() {
        let custom = ImfApplicationProfile::Custom("urn:example:custom-app".to_string());
        // Custom profiles have no hard limits
        assert_eq!(custom.max_resolution(), (u32::MAX, u32::MAX));
        assert_eq!(custom.max_audio_channels(), u32::MAX);
        assert!(!custom.allows_jpeg_xl());
        assert!(!custom.requires_dolby_vision_rpu_for_hdr());
        assert_eq!(custom.label(), "urn:example:custom-app");
    }

    #[test]
    fn test_imf_profile_label() {
        assert_eq!(
            ImfApplicationProfile::NetflixApp2.label(),
            "Netflix IMF App 2"
        );
        assert_eq!(
            ImfApplicationProfile::NetflixApp2_1.label(),
            "Netflix IMF App 2.1"
        );
        assert_eq!(ImfApplicationProfile::DisneyDece.label(), "Disney DECE");
    }

    #[test]
    fn test_profile_equality_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ImfApplicationProfile::NetflixApp2);
        set.insert(ImfApplicationProfile::NetflixApp2_1);
        set.insert(ImfApplicationProfile::DisneyDece);
        set.insert(ImfApplicationProfile::NetflixApp2); // duplicate
        assert_eq!(set.len(), 3, "HashSet should deduplicate equal profiles");
    }
}

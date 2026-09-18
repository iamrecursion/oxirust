//! Transcode profile management for MAM
//!
//! Provides a library of transcode profiles for various delivery formats,
//! including broadcast, web, mobile, archive, proxy and mezzanine targets.

#![allow(dead_code)]

/// Newtype wrapper for transcode profile IDs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TranscodeProfileId(pub u64);

impl TranscodeProfileId {
    /// Create a new profile ID
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the inner u64 value
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }
}

/// Video encoding parameters
#[derive(Debug, Clone)]
pub struct VideoProfile {
    /// Codec name (e.g. "h264", "prores", "dnxhd")
    pub codec: String,
    /// Target bitrate in kilobits per second
    pub bitrate_kbps: u32,
    /// Output resolution (width, height)
    pub resolution: (u32, u32),
    /// Frames per second
    pub fps: f32,
    /// Codec profile/level string (e.g. "high", "422", "HQ")
    pub profile_level: String,
}

impl VideoProfile {
    /// Create a new video profile
    #[must_use]
    pub fn new(
        codec: impl Into<String>,
        bitrate_kbps: u32,
        resolution: (u32, u32),
        fps: f32,
        profile_level: impl Into<String>,
    ) -> Self {
        Self {
            codec: codec.into(),
            bitrate_kbps,
            resolution,
            fps,
            profile_level: profile_level.into(),
        }
    }

    /// Return the pixel count for this resolution
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        self.resolution.0 as u64 * self.resolution.1 as u64
    }
}

/// Audio encoding parameters
#[derive(Debug, Clone)]
pub struct AudioProfile {
    /// Codec name (e.g. "aac", "pcm_s24le", "mp3")
    pub codec: String,
    /// Target bitrate in kilobits per second (0 = lossless / not applicable)
    pub bitrate_kbps: u32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u8,
}

impl AudioProfile {
    /// Create a new audio profile
    #[must_use]
    pub fn new(
        codec: impl Into<String>,
        bitrate_kbps: u32,
        sample_rate: u32,
        channels: u8,
    ) -> Self {
        Self {
            codec: codec.into(),
            bitrate_kbps,
            sample_rate,
            channels,
        }
    }
}

/// Delivery format category
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeliveryFormat {
    /// Broadcast delivery (SDI, MXF, etc.)
    Broadcast,
    /// Web streaming
    Web,
    /// Mobile delivery
    Mobile,
    /// Long-term archive
    Archive,
    /// Low-resolution proxy for offline editing
    Proxy,
    /// High-quality mezzanine for post-production
    Mezzanine,
}

impl DeliveryFormat {
    /// File suffix / label used for this format
    #[must_use]
    pub fn suffix(&self) -> &str {
        match self {
            Self::Broadcast => "broadcast",
            Self::Web => "web",
            Self::Mobile => "mobile",
            Self::Archive => "archive",
            Self::Proxy => "proxy",
            Self::Mezzanine => "mezzanine",
        }
    }
}

/// A complete transcode profile
#[derive(Debug, Clone)]
pub struct TranscodeProfile {
    /// Unique profile ID
    pub id: TranscodeProfileId,
    /// Human-readable profile name
    pub name: String,
    /// Delivery format this profile targets
    pub format: DeliveryFormat,
    /// Video encoding parameters
    pub video: VideoProfile,
    /// Audio encoding parameters
    pub audio: AudioProfile,
    /// Container format (e.g. "mxf", "mp4", "mov")
    pub container: String,
}

impl TranscodeProfile {
    /// Create a new transcode profile
    #[must_use]
    pub fn new(
        id: TranscodeProfileId,
        name: impl Into<String>,
        format: DeliveryFormat,
        video: VideoProfile,
        audio: AudioProfile,
        container: impl Into<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            format,
            video,
            audio,
            container: container.into(),
        }
    }
}

/// Built-in profile constructors
impl TranscodeProfile {
    /// MXF 1080i 50Hz broadcast profile
    #[must_use]
    pub fn mxf_1080i_50() -> Self {
        Self::new(
            TranscodeProfileId(1),
            "MXF 1080i 50",
            DeliveryFormat::Broadcast,
            VideoProfile::new("mpeg2video", 50_000, (1920, 1080), 25.0, "422P@HL"),
            AudioProfile::new("pcm_s24le", 0, 48_000, 2),
            "mxf",
        )
    }

    /// H.264 web 720p profile
    #[must_use]
    pub fn h264_web_720p() -> Self {
        Self::new(
            TranscodeProfileId(2),
            "H.264 Web 720p",
            DeliveryFormat::Web,
            VideoProfile::new("h264", 3_000, (1280, 720), 30.0, "high"),
            AudioProfile::new("aac", 128, 44_100, 2),
            "mp4",
        )
    }

    /// H.264 mobile 480p profile
    #[must_use]
    pub fn h264_mobile_480p() -> Self {
        Self::new(
            TranscodeProfileId(3),
            "H.264 Mobile 480p",
            DeliveryFormat::Mobile,
            VideoProfile::new("h264", 1_000, (854, 480), 30.0, "main"),
            AudioProfile::new("aac", 96, 44_100, 2),
            "mp4",
        )
    }

    /// DNxHD archive profile
    #[must_use]
    pub fn dnxhd_archive() -> Self {
        Self::new(
            TranscodeProfileId(4),
            "DNxHD Archive",
            DeliveryFormat::Archive,
            VideoProfile::new("dnxhd", 185_000, (1920, 1080), 25.0, "dnxhd_1080p_185"),
            AudioProfile::new("pcm_s24le", 0, 48_000, 8),
            "mxf",
        )
    }

    /// Apple ProRes proxy profile
    #[must_use]
    pub fn prores_proxy() -> Self {
        Self::new(
            TranscodeProfileId(5),
            "ProRes Proxy",
            DeliveryFormat::Proxy,
            VideoProfile::new("prores", 1_800, (1280, 720), 25.0, "proxy"),
            AudioProfile::new("pcm_s16le", 0, 48_000, 2),
            "mov",
        )
    }
}

/// A library of transcode profiles
#[derive(Debug, Default)]
pub struct ProfileLibrary {
    profiles: Vec<TranscodeProfile>,
    next_id: u64,
}

impl ProfileLibrary {
    /// Create a new empty profile library
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
            next_id: 100,
        }
    }

    /// Create a library pre-populated with all built-in profiles
    #[must_use]
    pub fn with_builtin_profiles() -> Self {
        let mut lib = Self::new();
        lib.profiles.push(TranscodeProfile::mxf_1080i_50());
        lib.profiles.push(TranscodeProfile::h264_web_720p());
        lib.profiles.push(TranscodeProfile::h264_mobile_480p());
        lib.profiles.push(TranscodeProfile::dnxhd_archive());
        lib.profiles.push(TranscodeProfile::prores_proxy());
        lib
    }

    /// Add a profile to the library, assigning a new ID if the current id is 0
    pub fn add(&mut self, mut profile: TranscodeProfile) {
        if profile.id.0 == 0 {
            profile.id = TranscodeProfileId(self.next_id);
            self.next_id += 1;
        }
        self.profiles.push(profile);
    }

    /// Find a profile by name (case-insensitive)
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&TranscodeProfile> {
        let lower = name.to_lowercase();
        self.profiles
            .iter()
            .find(|p| p.name.to_lowercase() == lower)
    }

    /// Return all profiles for the given delivery format
    #[must_use]
    pub fn profiles_for_format(&self, format: &DeliveryFormat) -> Vec<&TranscodeProfile> {
        self.profiles
            .iter()
            .filter(|p| &p.format == format)
            .collect()
    }

    /// Return the total number of profiles
    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Return true if the library contains no profiles
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Iterate over all profiles
    pub fn iter(&self) -> impl Iterator<Item = &TranscodeProfile> {
        self.profiles.iter()
    }
}

/// Validates transcode profiles and returns human-readable warnings
pub struct ProfileValidator;

impl ProfileValidator {
    /// Validate a transcode profile and return a list of warning strings.
    ///
    /// An empty return value means the profile is fully valid.
    #[must_use]
    pub fn validate(profile: &TranscodeProfile) -> Vec<String> {
        let mut warnings = Vec::new();

        // Video checks
        if profile.video.bitrate_kbps == 0 {
            warnings.push("Video bitrate is 0; only valid for lossless codecs".to_string());
        }

        if profile.video.resolution.0 == 0 || profile.video.resolution.1 == 0 {
            warnings.push("Video resolution contains a zero dimension".to_string());
        }

        if profile.video.fps <= 0.0 {
            warnings.push("Video frame rate must be positive".to_string());
        }

        if profile.video.codec.is_empty() {
            warnings.push("Video codec is not specified".to_string());
        }

        // Audio checks
        if profile.audio.sample_rate == 0 {
            warnings.push("Audio sample rate is 0".to_string());
        }

        if profile.audio.channels == 0 {
            warnings.push("Audio channel count is 0".to_string());
        }

        if profile.audio.codec.is_empty() {
            warnings.push("Audio codec is not specified".to_string());
        }

        // Container checks
        if profile.container.is_empty() {
            warnings.push("Container format is not specified".to_string());
        }

        // Profile name
        if profile.name.is_empty() {
            warnings.push("Profile name is empty".to_string());
        }

        // Cross-check: proxy should be low-bitrate
        if matches!(profile.format, DeliveryFormat::Proxy) && profile.video.bitrate_kbps > 10_000 {
            warnings.push("Proxy profile has unusually high video bitrate (>10 Mbps)".to_string());
        }

        // Cross-check: archive should not use lossy web codecs
        if matches!(profile.format, DeliveryFormat::Archive)
            && (profile.video.codec == "h264" || profile.video.codec == "h265")
        {
            warnings.push(
                "Archive profile uses a lossy delivery codec; consider DNxHD or ProRes".to_string(),
            );
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcode_profile_id_newtype() {
        let id = TranscodeProfileId::new(42);
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_delivery_format_suffix() {
        assert_eq!(DeliveryFormat::Broadcast.suffix(), "broadcast");
        assert_eq!(DeliveryFormat::Web.suffix(), "web");
        assert_eq!(DeliveryFormat::Mobile.suffix(), "mobile");
        assert_eq!(DeliveryFormat::Archive.suffix(), "archive");
        assert_eq!(DeliveryFormat::Proxy.suffix(), "proxy");
        assert_eq!(DeliveryFormat::Mezzanine.suffix(), "mezzanine");
    }

    #[test]
    fn test_builtin_mxf_1080i_50() {
        let p = TranscodeProfile::mxf_1080i_50();
        assert_eq!(p.format, DeliveryFormat::Broadcast);
        assert_eq!(p.video.resolution, (1920, 1080));
        assert_eq!(p.container, "mxf");
    }

    #[test]
    fn test_builtin_h264_web_720p() {
        let p = TranscodeProfile::h264_web_720p();
        assert_eq!(p.format, DeliveryFormat::Web);
        assert_eq!(p.video.resolution, (1280, 720));
        assert_eq!(p.video.codec, "h264");
    }

    #[test]
    fn test_builtin_h264_mobile_480p() {
        let p = TranscodeProfile::h264_mobile_480p();
        assert_eq!(p.format, DeliveryFormat::Mobile);
        assert_eq!(p.video.resolution, (854, 480));
    }

    #[test]
    fn test_builtin_dnxhd_archive() {
        let p = TranscodeProfile::dnxhd_archive();
        assert_eq!(p.format, DeliveryFormat::Archive);
        assert_eq!(p.video.codec, "dnxhd");
        assert_eq!(p.audio.channels, 8);
    }

    #[test]
    fn test_builtin_prores_proxy() {
        let p = TranscodeProfile::prores_proxy();
        assert_eq!(p.format, DeliveryFormat::Proxy);
        assert_eq!(p.video.codec, "prores");
    }

    #[test]
    fn test_profile_library_builtin() {
        let lib = ProfileLibrary::with_builtin_profiles();
        assert_eq!(lib.len(), 5);
    }

    #[test]
    fn test_profile_library_find_by_name() {
        let lib = ProfileLibrary::with_builtin_profiles();
        let p = lib.find_by_name("ProRes Proxy");
        assert!(p.is_some());
        assert_eq!(
            p.expect("should succeed in test").format,
            DeliveryFormat::Proxy
        );

        assert!(lib.find_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_profile_library_find_case_insensitive() {
        let lib = ProfileLibrary::with_builtin_profiles();
        assert!(lib.find_by_name("prores proxy").is_some());
    }

    #[test]
    fn test_profile_library_profiles_for_format() {
        let lib = ProfileLibrary::with_builtin_profiles();
        let web = lib.profiles_for_format(&DeliveryFormat::Web);
        assert_eq!(web.len(), 1);
        assert_eq!(web[0].video.codec, "h264");
    }

    #[test]
    fn test_profile_validator_valid() {
        let p = TranscodeProfile::h264_web_720p();
        let warnings = ProfileValidator::validate(&p);
        assert!(
            warnings.is_empty(),
            "Expected no warnings but got: {warnings:?}"
        );
    }

    #[test]
    fn test_profile_validator_zero_bitrate() {
        let mut p = TranscodeProfile::h264_web_720p();
        p.video.bitrate_kbps = 0;
        let warnings = ProfileValidator::validate(&p);
        assert!(warnings.iter().any(|w| w.contains("bitrate is 0")));
    }

    #[test]
    fn test_profile_validator_archive_lossy_codec_warning() {
        let mut p = TranscodeProfile::dnxhd_archive();
        p.video.codec = "h264".to_string();
        let warnings = ProfileValidator::validate(&p);
        assert!(warnings.iter().any(|w| w.contains("lossy delivery codec")));
    }
}

//! Codec profile definitions and selector utilities.
//!
//! Provides `CodecLevel`, `CodecProfile`, and `CodecProfileSelector`
//! for choosing the best encoding profile for a given resolution.

#![allow(dead_code)]

/// H.264/HEVC codec conformance level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CodecLevel {
    /// Level 3.0 – up to 720p30.
    Level30,
    /// Level 3.1 – up to 720p60 / 1080p30.
    Level31,
    /// Level 4.0 – up to 1080p30 high-quality.
    Level40,
    /// Level 4.1 – up to 1080p60.
    Level41,
    /// Level 5.0 – up to 2160p30.
    Level50,
    /// Level 5.1 – up to 2160p60.
    Level51,
    /// Level 5.2 – 2160p120 / 8K30.
    Level52,
}

impl CodecLevel {
    /// Returns the level as a floating-point number (e.g. 4.1).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn as_f32(self) -> f32 {
        match self {
            CodecLevel::Level30 => 3.0,
            CodecLevel::Level31 => 3.1,
            CodecLevel::Level40 => 4.0,
            CodecLevel::Level41 => 4.1,
            CodecLevel::Level50 => 5.0,
            CodecLevel::Level51 => 5.1,
            CodecLevel::Level52 => 5.2,
        }
    }

    /// Returns the maximum pixel count this level supports per frame.
    #[must_use]
    pub fn max_pixels(self) -> u64 {
        match self {
            CodecLevel::Level30 => 1_280 * 720,
            CodecLevel::Level31 => 1_280 * 720,
            CodecLevel::Level40 => 1_920 * 1_080,
            CodecLevel::Level41 => 1_920 * 1_080,
            CodecLevel::Level50 => 3_840 * 2_160,
            CodecLevel::Level51 => 3_840 * 2_160,
            CodecLevel::Level52 => 7_680 * 4_320,
        }
    }
}

/// An encoding profile combining codec name, level, and capability flags.
#[derive(Debug, Clone)]
pub struct CodecProfile {
    /// Codec identifier (e.g. "h264", "hevc", "av1").
    pub codec: String,
    /// Conformance level.
    pub level: CodecLevel,
    /// Maximum bitrate in Megabits per second.
    max_bitrate_mbps_val: f64,
    /// Whether hardware encoders are typically available for this profile.
    hw_encodable: bool,
    /// Whether this profile supports 10-bit colour.
    supports_10bit: bool,
}

impl CodecProfile {
    /// Creates a new codec profile.
    pub fn new(
        codec: impl Into<String>,
        level: CodecLevel,
        max_bitrate_mbps: f64,
        hw_encodable: bool,
    ) -> Self {
        Self {
            codec: codec.into(),
            level,
            max_bitrate_mbps_val: max_bitrate_mbps,
            hw_encodable,
            supports_10bit: false,
        }
    }

    /// Enables 10-bit colour support for this profile.
    #[must_use]
    pub fn with_10bit(mut self) -> Self {
        self.supports_10bit = true;
        self
    }

    /// Returns the maximum bitrate in Megabits per second.
    #[must_use]
    pub fn max_bitrate_mbps(&self) -> f64 {
        self.max_bitrate_mbps_val
    }

    /// Returns `true` if this profile is typically hardware-encodable.
    #[must_use]
    pub fn is_hardware_encodable(&self) -> bool {
        self.hw_encodable
    }

    /// Returns `true` if this profile supports 10-bit depth.
    #[must_use]
    pub fn supports_10bit(&self) -> bool {
        self.supports_10bit
    }

    /// Returns the maximum pixel count for this profile's level.
    #[must_use]
    pub fn max_pixels(&self) -> u64 {
        self.level.max_pixels()
    }

    /// Returns `true` if this profile can encode the given resolution.
    #[must_use]
    pub fn supports_resolution(&self, width: u32, height: u32) -> bool {
        let pixels = u64::from(width) * u64::from(height);
        pixels <= self.level.max_pixels()
    }
}

/// Selects an appropriate codec profile based on resolution requirements.
#[derive(Debug, Default)]
pub struct CodecProfileSelector {
    profiles: Vec<CodecProfile>,
}

impl CodecProfileSelector {
    /// Creates an empty selector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a selector pre-loaded with common H.264 profiles.
    #[must_use]
    pub fn with_h264_defaults() -> Self {
        let mut s = Self::new();
        s.add(CodecProfile::new("h264", CodecLevel::Level31, 10.0, true));
        s.add(CodecProfile::new("h264", CodecLevel::Level41, 50.0, true));
        s.add(CodecProfile::new("h264", CodecLevel::Level51, 240.0, true));
        s
    }

    /// Creates a selector pre-loaded with common HEVC profiles.
    #[must_use]
    pub fn with_hevc_defaults() -> Self {
        let mut s = Self::new();
        s.add(CodecProfile::new("hevc", CodecLevel::Level41, 40.0, true).with_10bit());
        s.add(CodecProfile::new("hevc", CodecLevel::Level51, 160.0, true).with_10bit());
        s.add(CodecProfile::new("hevc", CodecLevel::Level52, 640.0, false).with_10bit());
        s
    }

    /// Adds a profile to the selector.
    pub fn add(&mut self, profile: CodecProfile) {
        self.profiles.push(profile);
    }

    /// Selects the lowest-level (most compatible) profile that supports the
    /// given resolution. Returns `None` if no profile is sufficient.
    #[must_use]
    pub fn select_for_resolution(&self, width: u32, height: u32) -> Option<&CodecProfile> {
        self.profiles
            .iter()
            .filter(|p| p.supports_resolution(width, height))
            .min_by(|a, b| a.level.cmp(&b.level))
    }

    /// Returns all profiles that support the given resolution.
    #[must_use]
    pub fn all_for_resolution(&self, width: u32, height: u32) -> Vec<&CodecProfile> {
        self.profiles
            .iter()
            .filter(|p| p.supports_resolution(width, height))
            .collect()
    }

    /// Returns the number of profiles registered.
    #[must_use]
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }
}

/// Per-codec tune presets for content-specific optimization.
///
/// Each preset encodes recommended encoder parameters for a specific
/// content type (film, animation, grain, screen, etc.) for a specific codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecTunePreset {
    /// Codec this preset applies to (e.g. "av1", "vp9").
    pub codec: String,
    /// Content tune name (e.g. "film", "animation", "grain").
    pub tune_name: String,
    /// Recommended encoder options as key-value pairs.
    pub options: Vec<(String, String)>,
    /// Description of what this tune optimizes for.
    pub description: String,
}

impl CodecTunePreset {
    /// AV1 film tune preset.
    ///
    /// Optimized for live-action content with natural motion, subtle colour
    /// gradients, and moderate detail. Uses film grain synthesis to maintain
    /// perceived quality at lower bitrates.
    #[must_use]
    pub fn av1_film() -> Self {
        Self {
            codec: "av1".to_string(),
            tune_name: "film".to_string(),
            options: vec![
                ("enable-film-grain".to_string(), "1".to_string()),
                ("film-grain-denoise".to_string(), "1".to_string()),
                ("film-grain-table".to_string(), "auto".to_string()),
                ("aq-mode".to_string(), "2".to_string()),
                ("deltaq-mode".to_string(), "3".to_string()),
                ("enable-qm".to_string(), "1".to_string()),
                ("qm-min".to_string(), "5".to_string()),
                ("tune".to_string(), "ssim".to_string()),
                ("arnr-maxframes".to_string(), "7".to_string()),
                ("arnr-strength".to_string(), "4".to_string()),
            ],
            description: "Film: grain synthesis, perceptual quality, natural motion".to_string(),
        }
    }

    /// AV1 animation tune preset.
    ///
    /// Optimized for animated content with flat colour areas, sharp edges,
    /// and less texture. Favours PSNR-based quality and aggressive deblocking.
    #[must_use]
    pub fn av1_animation() -> Self {
        Self {
            codec: "av1".to_string(),
            tune_name: "animation".to_string(),
            options: vec![
                ("enable-film-grain".to_string(), "0".to_string()),
                ("aq-mode".to_string(), "0".to_string()),
                ("deltaq-mode".to_string(), "0".to_string()),
                ("enable-qm".to_string(), "1".to_string()),
                ("qm-min".to_string(), "0".to_string()),
                ("qm-max".to_string(), "8".to_string()),
                ("tune".to_string(), "psnr".to_string()),
                ("enable-keyframe-filtering".to_string(), "0".to_string()),
                ("arnr-maxframes".to_string(), "15".to_string()),
                ("arnr-strength".to_string(), "6".to_string()),
                ("enable-smooth-interintra".to_string(), "1".to_string()),
            ],
            description: "Animation: flat areas, sharp edges, PSNR-optimized".to_string(),
        }
    }

    /// AV1 grain preservation tune preset.
    ///
    /// Preserves film grain and high-frequency texture detail. Uses grain
    /// synthesis with higher fidelity parameters and disables aggressive
    /// denoising.
    #[must_use]
    pub fn av1_grain() -> Self {
        Self {
            codec: "av1".to_string(),
            tune_name: "grain".to_string(),
            options: vec![
                ("enable-film-grain".to_string(), "1".to_string()),
                ("film-grain-denoise".to_string(), "0".to_string()),
                ("aq-mode".to_string(), "1".to_string()),
                ("deltaq-mode".to_string(), "3".to_string()),
                ("enable-qm".to_string(), "1".to_string()),
                ("qm-min".to_string(), "0".to_string()),
                ("tune".to_string(), "ssim".to_string()),
                ("arnr-maxframes".to_string(), "4".to_string()),
                ("arnr-strength".to_string(), "2".to_string()),
                ("noise-sensitivity".to_string(), "0".to_string()),
            ],
            description: "Grain: preserve film grain and high-frequency detail".to_string(),
        }
    }

    /// VP9 film tune preset.
    ///
    /// Optimized for live-action content using VP9-specific parameters.
    #[must_use]
    pub fn vp9_film() -> Self {
        Self {
            codec: "vp9".to_string(),
            tune_name: "film".to_string(),
            options: vec![
                ("aq-mode".to_string(), "2".to_string()),
                ("lag-in-frames".to_string(), "25".to_string()),
                ("auto-alt-ref".to_string(), "6".to_string()),
                ("arnr-maxframes".to_string(), "7".to_string()),
                ("arnr-strength".to_string(), "4".to_string()),
                ("arnr-type".to_string(), "3".to_string()),
                ("tune".to_string(), "ssim".to_string()),
                ("row-mt".to_string(), "1".to_string()),
            ],
            description: "Film: temporal filtering, perceptual quality for live-action".to_string(),
        }
    }

    /// VP9 animation tune preset.
    ///
    /// Optimized for animated content with VP9-specific parameters.
    #[must_use]
    pub fn vp9_animation() -> Self {
        Self {
            codec: "vp9".to_string(),
            tune_name: "animation".to_string(),
            options: vec![
                ("aq-mode".to_string(), "0".to_string()),
                ("lag-in-frames".to_string(), "25".to_string()),
                ("auto-alt-ref".to_string(), "6".to_string()),
                ("arnr-maxframes".to_string(), "15".to_string()),
                ("arnr-strength".to_string(), "6".to_string()),
                ("arnr-type".to_string(), "3".to_string()),
                ("tune".to_string(), "psnr".to_string()),
                ("row-mt".to_string(), "1".to_string()),
            ],
            description: "Animation: flat areas, sharp edges, PSNR-optimized for VP9".to_string(),
        }
    }

    /// VP9 grain preservation tune preset.
    #[must_use]
    pub fn vp9_grain() -> Self {
        Self {
            codec: "vp9".to_string(),
            tune_name: "grain".to_string(),
            options: vec![
                ("aq-mode".to_string(), "0".to_string()),
                ("lag-in-frames".to_string(), "25".to_string()),
                ("auto-alt-ref".to_string(), "2".to_string()),
                ("arnr-maxframes".to_string(), "4".to_string()),
                ("arnr-strength".to_string(), "1".to_string()),
                ("arnr-type".to_string(), "3".to_string()),
                ("tune".to_string(), "ssim".to_string()),
                ("noise-sensitivity".to_string(), "0".to_string()),
                ("row-mt".to_string(), "1".to_string()),
            ],
            description: "Grain: minimal temporal filtering to preserve texture".to_string(),
        }
    }

    /// Returns all available tune presets for a given codec.
    #[must_use]
    pub fn presets_for_codec(codec: &str) -> Vec<Self> {
        match codec.to_lowercase().as_str() {
            "av1" | "libaom-av1" | "svt-av1" | "rav1e" => {
                vec![Self::av1_film(), Self::av1_animation(), Self::av1_grain()]
            }
            "vp9" | "libvpx-vp9" => {
                vec![Self::vp9_film(), Self::vp9_animation(), Self::vp9_grain()]
            }
            _ => Vec::new(),
        }
    }

    /// Looks up a specific tune preset by codec and tune name.
    #[must_use]
    pub fn find(codec: &str, tune_name: &str) -> Option<Self> {
        Self::presets_for_codec(codec)
            .into_iter()
            .find(|p| p.tune_name == tune_name)
    }

    /// Returns the number of encoder options in this preset.
    #[must_use]
    pub fn option_count(&self) -> usize {
        self.options.len()
    }

    /// Merges this preset's options into a `CodecConfig`'s options list.
    ///
    /// Existing options with matching keys are overwritten.
    pub fn apply_to_options(&self, options: &mut Vec<(String, String)>) {
        for (key, value) in &self.options {
            // Remove any existing option with the same key
            options.retain(|(k, _)| k != key);
            options.push((key.clone(), value.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_level_ordering() {
        assert!(CodecLevel::Level30 < CodecLevel::Level31);
        assert!(CodecLevel::Level41 < CodecLevel::Level50);
        assert!(CodecLevel::Level51 < CodecLevel::Level52);
    }

    #[test]
    fn test_codec_level_as_f32() {
        let v = CodecLevel::Level41.as_f32();
        assert!((v - 4.1_f32).abs() < 0.001);
    }

    #[test]
    fn test_codec_level_max_pixels_1080p() {
        assert_eq!(CodecLevel::Level41.max_pixels(), 1_920 * 1_080);
    }

    #[test]
    fn test_codec_level_max_pixels_4k() {
        assert_eq!(CodecLevel::Level51.max_pixels(), 3_840 * 2_160);
    }

    #[test]
    fn test_profile_max_bitrate() {
        let p = CodecProfile::new("h264", CodecLevel::Level41, 50.0, true);
        assert!((p.max_bitrate_mbps() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_profile_hw_encodable() {
        let p = CodecProfile::new("h264", CodecLevel::Level41, 50.0, true);
        assert!(p.is_hardware_encodable());
        let p2 = CodecProfile::new("av1", CodecLevel::Level51, 100.0, false);
        assert!(!p2.is_hardware_encodable());
    }

    #[test]
    fn test_profile_10bit_default_false() {
        let p = CodecProfile::new("h264", CodecLevel::Level41, 50.0, true);
        assert!(!p.supports_10bit());
    }

    #[test]
    fn test_profile_with_10bit() {
        let p = CodecProfile::new("hevc", CodecLevel::Level51, 160.0, true).with_10bit();
        assert!(p.supports_10bit());
    }

    #[test]
    fn test_profile_supports_resolution_1080p() {
        let p = CodecProfile::new("h264", CodecLevel::Level41, 50.0, true);
        assert!(p.supports_resolution(1920, 1080));
        assert!(!p.supports_resolution(3840, 2160));
    }

    #[test]
    fn test_selector_h264_defaults_count() {
        let sel = CodecProfileSelector::with_h264_defaults();
        assert_eq!(sel.profile_count(), 3);
    }

    #[test]
    fn test_selector_select_for_1080p_returns_lowest_level() {
        let sel = CodecProfileSelector::with_h264_defaults();
        let p = sel
            .select_for_resolution(1920, 1080)
            .expect("should succeed in test");
        // Level31 max_pixels is 1280*720, so Level41 should be selected
        assert_eq!(p.level, CodecLevel::Level41);
    }

    #[test]
    fn test_selector_select_for_720p() {
        let sel = CodecProfileSelector::with_h264_defaults();
        let p = sel
            .select_for_resolution(1280, 720)
            .expect("should succeed in test");
        assert_eq!(p.level, CodecLevel::Level31);
    }

    #[test]
    fn test_selector_select_for_8k_returns_none_h264() {
        let sel = CodecProfileSelector::with_h264_defaults();
        // 8K exceeds all H.264 profiles in the default set
        assert!(sel.select_for_resolution(7680, 4320).is_none());
    }

    #[test]
    fn test_selector_hevc_supports_4k() {
        let sel = CodecProfileSelector::with_hevc_defaults();
        let p = sel
            .select_for_resolution(3840, 2160)
            .expect("should succeed in test");
        assert_eq!(p.codec, "hevc");
        assert!(p.supports_10bit());
    }

    // ── CodecTunePreset tests ───────────────────────────────────────────

    #[test]
    fn test_av1_film_preset() {
        let p = CodecTunePreset::av1_film();
        assert_eq!(p.codec, "av1");
        assert_eq!(p.tune_name, "film");
        assert!(p.option_count() > 0);
        assert!(p
            .options
            .iter()
            .any(|(k, v)| k == "enable-film-grain" && v == "1"));
        assert!(p.options.iter().any(|(k, v)| k == "tune" && v == "ssim"));
    }

    #[test]
    fn test_av1_animation_preset() {
        let p = CodecTunePreset::av1_animation();
        assert_eq!(p.codec, "av1");
        assert_eq!(p.tune_name, "animation");
        assert!(p
            .options
            .iter()
            .any(|(k, v)| k == "enable-film-grain" && v == "0"));
        assert!(p.options.iter().any(|(k, v)| k == "tune" && v == "psnr"));
    }

    #[test]
    fn test_av1_grain_preset() {
        let p = CodecTunePreset::av1_grain();
        assert_eq!(p.codec, "av1");
        assert_eq!(p.tune_name, "grain");
        // Film grain enabled but denoise disabled
        assert!(p
            .options
            .iter()
            .any(|(k, v)| k == "enable-film-grain" && v == "1"));
        assert!(p
            .options
            .iter()
            .any(|(k, v)| k == "film-grain-denoise" && v == "0"));
    }

    #[test]
    fn test_vp9_film_preset() {
        let p = CodecTunePreset::vp9_film();
        assert_eq!(p.codec, "vp9");
        assert_eq!(p.tune_name, "film");
        assert!(p
            .options
            .iter()
            .any(|(k, v)| k == "lag-in-frames" && v == "25"));
        assert!(p.options.iter().any(|(k, v)| k == "tune" && v == "ssim"));
    }

    #[test]
    fn test_vp9_animation_preset() {
        let p = CodecTunePreset::vp9_animation();
        assert_eq!(p.codec, "vp9");
        assert_eq!(p.tune_name, "animation");
        assert!(p.options.iter().any(|(k, v)| k == "tune" && v == "psnr"));
    }

    #[test]
    fn test_vp9_grain_preset() {
        let p = CodecTunePreset::vp9_grain();
        assert_eq!(p.codec, "vp9");
        assert_eq!(p.tune_name, "grain");
        // Minimal temporal filtering
        assert!(p
            .options
            .iter()
            .any(|(k, v)| k == "arnr-strength" && v == "1"));
    }

    #[test]
    fn test_presets_for_av1() {
        let presets = CodecTunePreset::presets_for_codec("av1");
        assert_eq!(presets.len(), 3);
        let names: Vec<&str> = presets.iter().map(|p| p.tune_name.as_str()).collect();
        assert!(names.contains(&"film"));
        assert!(names.contains(&"animation"));
        assert!(names.contains(&"grain"));
    }

    #[test]
    fn test_presets_for_vp9() {
        let presets = CodecTunePreset::presets_for_codec("vp9");
        assert_eq!(presets.len(), 3);
    }

    #[test]
    fn test_presets_for_av1_alias() {
        let presets = CodecTunePreset::presets_for_codec("libaom-av1");
        assert_eq!(presets.len(), 3);
    }

    #[test]
    fn test_presets_for_vp9_alias() {
        let presets = CodecTunePreset::presets_for_codec("libvpx-vp9");
        assert_eq!(presets.len(), 3);
    }

    #[test]
    fn test_presets_for_unknown_codec() {
        let presets = CodecTunePreset::presets_for_codec("h264");
        assert!(presets.is_empty());
    }

    #[test]
    fn test_find_av1_film() {
        let p = CodecTunePreset::find("av1", "film");
        assert!(p.is_some());
        let p = p.expect("should find av1 film preset");
        assert_eq!(p.tune_name, "film");
    }

    #[test]
    fn test_find_vp9_animation() {
        let p = CodecTunePreset::find("vp9", "animation");
        assert!(p.is_some());
    }

    #[test]
    fn test_find_nonexistent() {
        assert!(CodecTunePreset::find("av1", "nonexistent").is_none());
        assert!(CodecTunePreset::find("h264", "film").is_none());
    }

    #[test]
    fn test_apply_to_options() {
        let preset = CodecTunePreset::av1_film();
        let mut options = vec![
            ("tune".to_string(), "psnr".to_string()),
            ("custom".to_string(), "value".to_string()),
        ];
        preset.apply_to_options(&mut options);
        // "tune" should be overwritten to "ssim"
        let tune_val = options
            .iter()
            .find(|(k, _)| k == "tune")
            .map(|(_, v)| v.as_str());
        assert_eq!(tune_val, Some("ssim"));
        // "custom" should still be present
        assert!(options.iter().any(|(k, _)| k == "custom"));
        // Film grain options should be added
        assert!(options
            .iter()
            .any(|(k, v)| k == "enable-film-grain" && v == "1"));
    }

    #[test]
    fn test_apply_to_empty_options() {
        let preset = CodecTunePreset::vp9_grain();
        let mut options = Vec::new();
        preset.apply_to_options(&mut options);
        assert_eq!(options.len(), preset.option_count());
    }

    #[test]
    fn test_description_not_empty() {
        let presets = [
            CodecTunePreset::av1_film(),
            CodecTunePreset::av1_animation(),
            CodecTunePreset::av1_grain(),
            CodecTunePreset::vp9_film(),
            CodecTunePreset::vp9_animation(),
            CodecTunePreset::vp9_grain(),
        ];
        for p in &presets {
            assert!(
                !p.description.is_empty(),
                "Description for {}:{} should not be empty",
                p.codec,
                p.tune_name
            );
        }
    }
}

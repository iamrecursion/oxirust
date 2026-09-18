//! ICC color profile parsing and management.
//!
//! Provides types for representing ICC color profile headers, full profiles,
//! and a registry for managing multiple profiles.

#![allow(dead_code)]

// ── ColorSpaceSignature ───────────────────────────────────────────────────────

/// ICC color space signature identifying the color model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpaceSignature {
    /// RGB (red, green, blue) color space.
    Rgb,
    /// CMYK (cyan, magenta, yellow, key/black) color space.
    Cmyk,
    /// Grayscale color space.
    Gray,
    /// CIE L*a*b* color space.
    Lab,
    /// CIE XYZ color space.
    Xyz,
    /// CIE L*u*v* color space.
    Luv,
    /// YUV color space.
    Yuv,
    /// YCbCr color space.
    Ycbcr,
}

impl ColorSpaceSignature {
    /// Returns the number of channels for this color space.
    #[must_use]
    pub fn channel_count(&self) -> u8 {
        match self {
            Self::Rgb | Self::Lab | Self::Xyz | Self::Luv | Self::Yuv | Self::Ycbcr => 3,
            Self::Cmyk => 4,
            Self::Gray => 1,
        }
    }
}

// ── ProfileClass ──────────────────────────────────────────────────────────────

/// ICC profile class, describing the type of device or transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileClass {
    /// Input device profile (scanner, camera).
    Input,
    /// Display device profile (monitor).
    Display,
    /// Output device profile (printer, projector).
    Output,
    /// Device-link profile (chained device transforms).
    DeviceLink,
    /// Abstract profile (general color transformation).
    Abstract,
    /// Named color profile.
    NamedColor,
}

impl ProfileClass {
    /// Returns `true` if this is a real device profile (Input, Display, or Output).
    #[must_use]
    pub fn is_device_profile(&self) -> bool {
        matches!(self, Self::Input | Self::Display | Self::Output)
    }
}

// ── IccProfileHeader ──────────────────────────────────────────────────────────

/// ICC profile header fields (subset of the full 128-byte ICC header).
#[derive(Debug, Clone)]
pub struct IccProfileHeader {
    /// Total profile size in bytes.
    pub profile_size: u32,
    /// Device color space signature.
    pub color_space: ColorSpaceSignature,
    /// Profile class (device type).
    pub profile_class: ProfileClass,
    /// Profile Connection Space (PCS) color space.
    pub pcs: ColorSpaceSignature,
    /// Rendering intent (0=Perceptual, 1=RelativeColorimetric, 2=Saturation, 3=AbsoluteColorimetric).
    pub rendering_intent: u8,
    /// Profile creator signature (4 ASCII bytes, e.g. b"ADBE").
    pub creator: [u8; 4],
}

impl IccProfileHeader {
    /// Returns `true` if the Profile Connection Space (PCS) is L*a*b* or XYZ,
    /// which are the only valid PCS values in ICC specs.
    #[must_use]
    pub fn is_valid_pcs(&self) -> bool {
        matches!(
            self.pcs,
            ColorSpaceSignature::Lab | ColorSpaceSignature::Xyz
        )
    }
}

// ── IccProfile ────────────────────────────────────────────────────────────────

/// A complete ICC color profile with header and textual metadata.
#[derive(Debug, Clone)]
pub struct IccProfile {
    /// The parsed profile header.
    pub header: IccProfileHeader,
    /// Human-readable profile description.
    pub description: String,
    /// Copyright string embedded in the profile.
    pub copyright: String,
}

impl IccProfile {
    /// Returns `true` if the device color space of this profile is RGB.
    #[must_use]
    pub fn is_rgb(&self) -> bool {
        self.header.color_space == ColorSpaceSignature::Rgb
    }

    /// Returns `true` if this is a display (monitor) profile.
    #[must_use]
    pub fn is_display_profile(&self) -> bool {
        self.header.profile_class == ProfileClass::Display
    }
}

// ── ProfileRegistry ───────────────────────────────────────────────────────────

/// A registry that stores and queries a collection of [`IccProfile`]s.
#[derive(Debug, Default)]
pub struct ProfileRegistry {
    profiles: Vec<IccProfile>,
}

impl ProfileRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a profile to the registry.
    pub fn add(&mut self, profile: IccProfile) {
        self.profiles.push(profile);
    }

    /// Find profiles whose description contains `query` (case-insensitive).
    #[must_use]
    pub fn find_by_description(&self, query: &str) -> Vec<&IccProfile> {
        let q = query.to_lowercase();
        self.profiles
            .iter()
            .filter(|p| p.description.to_lowercase().contains(&q))
            .collect()
    }

    /// Return all RGB profiles in the registry.
    #[must_use]
    pub fn rgb_profiles(&self) -> Vec<&IccProfile> {
        self.profiles.iter().filter(|p| p.is_rgb()).collect()
    }

    /// Return the total number of profiles in the registry.
    #[must_use]
    pub fn count(&self) -> usize {
        self.profiles.len()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header(
        cs: ColorSpaceSignature,
        cls: ProfileClass,
        pcs: ColorSpaceSignature,
    ) -> IccProfileHeader {
        IccProfileHeader {
            profile_size: 1024,
            color_space: cs,
            profile_class: cls,
            pcs,
            rendering_intent: 0,
            creator: *b"TEST",
        }
    }

    fn rgb_display_profile(desc: &str) -> IccProfile {
        IccProfile {
            header: sample_header(
                ColorSpaceSignature::Rgb,
                ProfileClass::Display,
                ColorSpaceSignature::Lab,
            ),
            description: desc.to_string(),
            copyright: "Test Copyright".to_string(),
        }
    }

    // ── ColorSpaceSignature ────────────────────────────────────────────────────

    #[test]
    fn test_rgb_channel_count() {
        assert_eq!(ColorSpaceSignature::Rgb.channel_count(), 3);
    }

    #[test]
    fn test_cmyk_channel_count() {
        assert_eq!(ColorSpaceSignature::Cmyk.channel_count(), 4);
    }

    #[test]
    fn test_gray_channel_count() {
        assert_eq!(ColorSpaceSignature::Gray.channel_count(), 1);
    }

    #[test]
    fn test_lab_channel_count() {
        assert_eq!(ColorSpaceSignature::Lab.channel_count(), 3);
    }

    #[test]
    fn test_xyz_channel_count() {
        assert_eq!(ColorSpaceSignature::Xyz.channel_count(), 3);
    }

    // ── ProfileClass ───────────────────────────────────────────────────────────

    #[test]
    fn test_input_is_device_profile() {
        assert!(ProfileClass::Input.is_device_profile());
    }

    #[test]
    fn test_display_is_device_profile() {
        assert!(ProfileClass::Display.is_device_profile());
    }

    #[test]
    fn test_output_is_device_profile() {
        assert!(ProfileClass::Output.is_device_profile());
    }

    #[test]
    fn test_devicelink_not_device_profile() {
        assert!(!ProfileClass::DeviceLink.is_device_profile());
    }

    #[test]
    fn test_abstract_not_device_profile() {
        assert!(!ProfileClass::Abstract.is_device_profile());
    }

    #[test]
    fn test_namedcolor_not_device_profile() {
        assert!(!ProfileClass::NamedColor.is_device_profile());
    }

    // ── IccProfileHeader ───────────────────────────────────────────────────────

    #[test]
    fn test_valid_pcs_lab() {
        let h = sample_header(
            ColorSpaceSignature::Rgb,
            ProfileClass::Display,
            ColorSpaceSignature::Lab,
        );
        assert!(h.is_valid_pcs());
    }

    #[test]
    fn test_valid_pcs_xyz() {
        let h = sample_header(
            ColorSpaceSignature::Rgb,
            ProfileClass::Display,
            ColorSpaceSignature::Xyz,
        );
        assert!(h.is_valid_pcs());
    }

    #[test]
    fn test_invalid_pcs_rgb() {
        let h = sample_header(
            ColorSpaceSignature::Rgb,
            ProfileClass::Display,
            ColorSpaceSignature::Rgb,
        );
        assert!(!h.is_valid_pcs());
    }

    // ── IccProfile ─────────────────────────────────────────────────────────────

    #[test]
    fn test_is_rgb_true() {
        let p = rgb_display_profile("sRGB");
        assert!(p.is_rgb());
    }

    #[test]
    fn test_is_rgb_false_for_gray() {
        let p = IccProfile {
            header: sample_header(
                ColorSpaceSignature::Gray,
                ProfileClass::Input,
                ColorSpaceSignature::Lab,
            ),
            description: "Gray profile".to_string(),
            copyright: "".to_string(),
        };
        assert!(!p.is_rgb());
    }

    #[test]
    fn test_is_display_profile_true() {
        let p = rgb_display_profile("sRGB IEC61966-2.1");
        assert!(p.is_display_profile());
    }

    #[test]
    fn test_is_display_profile_false_for_input() {
        let p = IccProfile {
            header: sample_header(
                ColorSpaceSignature::Rgb,
                ProfileClass::Input,
                ColorSpaceSignature::Lab,
            ),
            description: "Camera profile".to_string(),
            copyright: "".to_string(),
        };
        assert!(!p.is_display_profile());
    }

    // ── ProfileRegistry ────────────────────────────────────────────────────────

    #[test]
    fn test_registry_count_empty() {
        let reg = ProfileRegistry::new();
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_registry_add_and_count() {
        let mut reg = ProfileRegistry::new();
        reg.add(rgb_display_profile("sRGB"));
        reg.add(rgb_display_profile("AdobeRGB"));
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn test_registry_find_by_description() {
        let mut reg = ProfileRegistry::new();
        reg.add(rgb_display_profile("sRGB IEC61966-2.1"));
        reg.add(rgb_display_profile("AdobeRGB (1998)"));
        let found = reg.find_by_description("adobe");
        assert_eq!(found.len(), 1);
        assert!(found[0].description.contains("Adobe"));
    }

    #[test]
    fn test_registry_find_by_description_no_match() {
        let mut reg = ProfileRegistry::new();
        reg.add(rgb_display_profile("sRGB"));
        let found = reg.find_by_description("ProPhoto");
        assert!(found.is_empty());
    }

    #[test]
    fn test_registry_rgb_profiles() {
        let mut reg = ProfileRegistry::new();
        reg.add(rgb_display_profile("sRGB"));
        // Add a CMYK profile
        reg.add(IccProfile {
            header: sample_header(
                ColorSpaceSignature::Cmyk,
                ProfileClass::Output,
                ColorSpaceSignature::Lab,
            ),
            description: "US Web Coated".to_string(),
            copyright: "".to_string(),
        });
        let rgb = reg.rgb_profiles();
        assert_eq!(rgb.len(), 1);
        assert_eq!(rgb[0].description, "sRGB");
    }
}

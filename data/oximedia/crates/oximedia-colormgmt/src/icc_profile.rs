//! ICC color profile handling for `OxiMedia`.
//!
//! This module provides types and utilities for working with ICC color profiles,
//! including profile headers, tags, and profile metadata extraction.

#![allow(dead_code)]

/// ICC profile class identifying the device or color space type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccProfileClass {
    /// Input device profile (scanner, camera).
    InputDevice,
    /// Display device profile (monitor).
    DisplayDevice,
    /// Output device profile (printer).
    OutputDevice,
    /// Color space conversion profile.
    ColorSpace,
    /// Abstract profile.
    Abstract,
    /// Named color profile.
    Named,
}

impl IccProfileClass {
    /// Returns the 4-character ICC profile class code.
    #[must_use]
    pub fn code(&self) -> &str {
        match self {
            Self::InputDevice => "scnr",
            Self::DisplayDevice => "mntr",
            Self::OutputDevice => "prtr",
            Self::ColorSpace => "spac",
            Self::Abstract => "abst",
            Self::Named => "nmcl",
        }
    }
}

/// ICC color space identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccColorSpace {
    /// CIE XYZ.
    Xyz,
    /// CIE L*a*b*.
    Lab,
    /// RGB color space.
    Rgb,
    /// CMYK color space.
    Cmyk,
    /// Grayscale.
    Gray,
    /// Hue, Saturation, Value.
    Hsv,
    /// Hue, Saturation, Lightness.
    Hsl,
    /// YUV / YCbCr.
    Yuv,
}

impl IccColorSpace {
    /// Returns the number of channels for this color space.
    #[must_use]
    pub fn channels(&self) -> u8 {
        match self {
            Self::Xyz | Self::Lab | Self::Rgb | Self::Hsv | Self::Hsl | Self::Yuv => 3,
            Self::Cmyk => 4,
            Self::Gray => 1,
        }
    }
}

/// ICC profile header containing key metadata.
#[derive(Debug, Clone)]
pub struct IccProfileHeader {
    /// Total size of the profile in bytes.
    pub profile_size: u32,
    /// Preferred CMM (Color Management Module) type.
    pub cmm_type: String,
    /// Profile version as (major, minor).
    pub version: (u8, u8),
    /// Profile class.
    pub profile_class: IccProfileClass,
    /// Color space of the data.
    pub color_space: IccColorSpace,
    /// Profile connection space (PCS).
    pub pcs: IccColorSpace,
    /// Creation timestamp in milliseconds since epoch.
    pub creation_date_ms: u64,
    /// Device manufacturer identifier.
    pub manufacturer: String,
}

impl IccProfileHeader {
    /// Returns `true` if the profile header contains valid data.
    ///
    /// A header is valid if the profile size is non-zero and the PCS is
    /// either XYZ or Lab (the only valid PCS spaces per ICC specification).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.profile_size > 0 && (self.pcs == IccColorSpace::Xyz || self.pcs == IccColorSpace::Lab)
    }
}

/// A single ICC profile tag.
#[derive(Debug, Clone)]
pub struct IccTag {
    /// 4-character tag signature (e.g. "desc", "cprt").
    pub signature: String,
    /// Raw tag data bytes.
    pub data: Vec<u8>,
}

impl IccTag {
    /// Attempts to interpret the tag data as UTF-8 text.
    ///
    /// Returns `None` if the bytes are not valid UTF-8.
    #[must_use]
    pub fn as_text(&self) -> Option<String> {
        std::str::from_utf8(&self.data)
            .ok()
            .map(|s| s.trim_end_matches('\0').to_owned())
    }
}

/// A complete ICC color profile.
#[derive(Debug, Clone)]
pub struct IccProfile {
    /// The profile header.
    pub header: IccProfileHeader,
    /// All tags contained in the profile.
    pub tags: Vec<IccTag>,
}

impl IccProfile {
    /// Finds a tag by its 4-character signature.
    #[must_use]
    pub fn find_tag(&self, sig: &str) -> Option<&IccTag> {
        self.tags.iter().find(|t| t.signature == sig)
    }

    /// Returns the human-readable profile description, if present.
    ///
    /// Looks for the "desc" tag and interprets its data as text.
    #[must_use]
    pub fn description(&self) -> Option<String> {
        self.find_tag("desc").and_then(|t| t.as_text())
    }

    /// Returns `true` if this is a display device profile.
    #[must_use]
    pub fn is_display_profile(&self) -> bool {
        self.header.profile_class == IccProfileClass::DisplayDevice
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header(class: IccProfileClass, pcs: IccColorSpace) -> IccProfileHeader {
        IccProfileHeader {
            profile_size: 512,
            cmm_type: "ADBE".to_owned(),
            version: (4, 0),
            profile_class: class,
            color_space: IccColorSpace::Rgb,
            pcs,
            creation_date_ms: 0,
            manufacturer: "TestMfg".to_owned(),
        }
    }

    #[test]
    fn test_profile_class_codes() {
        assert_eq!(IccProfileClass::InputDevice.code(), "scnr");
        assert_eq!(IccProfileClass::DisplayDevice.code(), "mntr");
        assert_eq!(IccProfileClass::OutputDevice.code(), "prtr");
        assert_eq!(IccProfileClass::ColorSpace.code(), "spac");
        assert_eq!(IccProfileClass::Abstract.code(), "abst");
        assert_eq!(IccProfileClass::Named.code(), "nmcl");
    }

    #[test]
    fn test_color_space_channels() {
        assert_eq!(IccColorSpace::Xyz.channels(), 3);
        assert_eq!(IccColorSpace::Lab.channels(), 3);
        assert_eq!(IccColorSpace::Rgb.channels(), 3);
        assert_eq!(IccColorSpace::Cmyk.channels(), 4);
        assert_eq!(IccColorSpace::Gray.channels(), 1);
        assert_eq!(IccColorSpace::Hsv.channels(), 3);
        assert_eq!(IccColorSpace::Hsl.channels(), 3);
        assert_eq!(IccColorSpace::Yuv.channels(), 3);
    }

    #[test]
    fn test_header_is_valid_xyz_pcs() {
        let h = make_header(IccProfileClass::DisplayDevice, IccColorSpace::Xyz);
        assert!(h.is_valid());
    }

    #[test]
    fn test_header_is_valid_lab_pcs() {
        let h = make_header(IccProfileClass::DisplayDevice, IccColorSpace::Lab);
        assert!(h.is_valid());
    }

    #[test]
    fn test_header_is_invalid_wrong_pcs() {
        let h = make_header(IccProfileClass::DisplayDevice, IccColorSpace::Rgb);
        assert!(!h.is_valid());
    }

    #[test]
    fn test_header_is_invalid_zero_size() {
        let mut h = make_header(IccProfileClass::DisplayDevice, IccColorSpace::Xyz);
        h.profile_size = 0;
        assert!(!h.is_valid());
    }

    #[test]
    fn test_tag_as_text_valid_utf8() {
        let tag = IccTag {
            signature: "desc".to_owned(),
            data: b"sRGB IEC61966-2.1\0".to_vec(),
        };
        let text = tag.as_text();
        assert_eq!(text.as_deref(), Some("sRGB IEC61966-2.1"));
    }

    #[test]
    fn test_tag_as_text_invalid_utf8() {
        let tag = IccTag {
            signature: "desc".to_owned(),
            data: vec![0xFF, 0xFE, 0x00],
        };
        assert!(tag.as_text().is_none());
    }

    #[test]
    fn test_profile_find_tag() {
        let profile = IccProfile {
            header: make_header(IccProfileClass::DisplayDevice, IccColorSpace::Xyz),
            tags: vec![
                IccTag {
                    signature: "desc".to_owned(),
                    data: b"Display Profile".to_vec(),
                },
                IccTag {
                    signature: "cprt".to_owned(),
                    data: b"Copyright 2024".to_vec(),
                },
            ],
        };
        assert!(profile.find_tag("desc").is_some());
        assert!(profile.find_tag("cprt").is_some());
        assert!(profile.find_tag("wtpt").is_none());
    }

    #[test]
    fn test_profile_description() {
        let profile = IccProfile {
            header: make_header(IccProfileClass::DisplayDevice, IccColorSpace::Xyz),
            tags: vec![IccTag {
                signature: "desc".to_owned(),
                data: b"My Color Profile".to_vec(),
            }],
        };
        assert_eq!(profile.description().as_deref(), Some("My Color Profile"));
    }

    #[test]
    fn test_profile_description_missing() {
        let profile = IccProfile {
            header: make_header(IccProfileClass::DisplayDevice, IccColorSpace::Xyz),
            tags: vec![],
        };
        assert!(profile.description().is_none());
    }

    #[test]
    fn test_profile_is_display() {
        let display = IccProfile {
            header: make_header(IccProfileClass::DisplayDevice, IccColorSpace::Xyz),
            tags: vec![],
        };
        assert!(display.is_display_profile());

        let printer = IccProfile {
            header: make_header(IccProfileClass::OutputDevice, IccColorSpace::Xyz),
            tags: vec![],
        };
        assert!(!printer.is_display_profile());
    }

    #[test]
    fn test_profile_class_equality() {
        assert_eq!(IccProfileClass::InputDevice, IccProfileClass::InputDevice);
        assert_ne!(IccProfileClass::InputDevice, IccProfileClass::DisplayDevice);
    }
}

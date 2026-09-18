//! Dolby Vision profile management
//!
//! Provides enumerations and validation for Dolby Vision profiles, levels,
//! compatibility identifiers, and profile constraints.

/// Dolby Vision profile identifiers (extended set)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum DvProfile {
    /// Profile 4: HDR10/SDR dual-stream (legacy, rare)
    P4,
    /// Profile 5: IPT-PQ, no backward compatibility layer
    P5,
    /// Profile 7: Dual-layer MEL + BL, full Dolby Vision
    P7,
    /// Profile 8.1: BL-only, backward compatible with HDR10
    P8_1,
    /// Profile 8.2: BL-only, backward compatible with SDR
    P8_2,
    /// Profile 8.4: BL-only, backward compatible with HLG
    P8_4,
    /// Profile 9.1: BL-only, SDR backward compatible (low complexity)
    P9_1,
    /// Profile 9.2: BL-only, HLG backward compatible (low complexity)
    P9_2,
}

impl DvProfile {
    /// Returns `true` if the profile uses a single video layer (no EL)
    #[must_use]
    pub fn is_single_layer(&self) -> bool {
        !matches!(self, Self::P7)
    }

    /// Returns the maximum peak luminance supported by this profile (nits)
    #[must_use]
    pub fn max_nits(&self) -> f32 {
        match self {
            Self::P4 => 4000.0,
            Self::P5 => 10_000.0,
            Self::P7 => 10_000.0,
            Self::P8_1 | Self::P8_2 => 10_000.0,
            Self::P8_4 => 1000.0,
            Self::P9_1 | Self::P9_2 => 1000.0,
        }
    }
}

/// Dolby Vision dynamic metadata level with associated target peak nits
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum DvLevel {
    /// Level 1 metadata with target peak nits
    Level1(f32),
    /// Level 2 metadata with target peak nits
    Level2(f32),
    /// Level 3 metadata with target peak nits
    Level3(f32),
    /// Level 6 (static fallback metadata, no associated nits)
    Level6,
}

/// Dolby Vision backward compatibility identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum DvCompatibilityId {
    /// No backward compatibility (profile 5 / pure Dolby Vision)
    None,
    /// HDR10 (SMPTE ST 2084 / PQ) backward compatibility
    Hdr10,
    /// SDR (BT.709) backward compatibility
    Sdr,
    /// HLG (ITU-R BT.2100) backward compatibility
    Hlg,
}

impl DvCompatibilityId {
    /// Return the signal type description string
    #[must_use]
    pub fn signal_type(&self) -> &str {
        match self {
            Self::None => "DolbyVision-only",
            Self::Hdr10 => "HDR10",
            Self::Sdr => "SDR",
            Self::Hlg => "HLG",
        }
    }
}

/// Physical and operational constraints for a Dolby Vision profile
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DvProfileConstraints {
    /// Maximum frame width in pixels
    pub max_width: u32,
    /// Maximum frame height in pixels
    pub max_height: u32,
    /// Maximum frames per second
    pub max_fps: f32,
    /// Whether an enhancement layer (EL) is supported
    pub supports_el: bool,
}

impl DvProfileConstraints {
    /// Return the constraints for the given profile
    #[must_use]
    pub fn for_profile(profile: &DvProfile) -> Self {
        match profile {
            DvProfile::P4 => Self {
                max_width: 4096,
                max_height: 2160,
                max_fps: 60.0,
                supports_el: true,
            },
            DvProfile::P5 => Self {
                max_width: 4096,
                max_height: 2160,
                max_fps: 120.0,
                supports_el: false,
            },
            DvProfile::P7 => Self {
                max_width: 4096,
                max_height: 2160,
                max_fps: 60.0,
                supports_el: true,
            },
            DvProfile::P8_1 | DvProfile::P8_2 => Self {
                max_width: 7680,
                max_height: 4320,
                max_fps: 120.0,
                supports_el: false,
            },
            DvProfile::P8_4 => Self {
                max_width: 3840,
                max_height: 2160,
                max_fps: 60.0,
                supports_el: false,
            },
            DvProfile::P9_1 | DvProfile::P9_2 => Self {
                max_width: 1920,
                max_height: 1080,
                max_fps: 60.0,
                supports_el: false,
            },
        }
    }
}

/// Converts between compatible Dolby Vision profiles
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct ProfileConverter;

impl ProfileConverter {
    /// Attempt to convert a Profile 7 stream to a compatible Profile 8 variant.
    ///
    /// Returns `None` if the input is not Profile 7 (no conversion possible).
    #[must_use]
    pub fn convert_p7_to_p8(input_profile: DvProfile) -> Option<DvProfile> {
        match input_profile {
            // P7 can be downgraded to P8.1 (HDR10 compat) by dropping the EL
            DvProfile::P7 => Some(DvProfile::P8_1),
            _ => None,
        }
    }
}

/// A summary report for a Dolby Vision stream
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DvProfileReport {
    /// The detected/assigned profile
    pub profile: DvProfile,
    /// The metadata level
    pub level: DvLevel,
    /// The backward compatibility identifier
    pub compat_id: DvCompatibilityId,
    /// Whether this combination is valid per Dolby's specification
    pub is_valid: bool,
}

impl DvProfileReport {
    /// Create a new profile report
    #[must_use]
    pub fn new(
        profile: DvProfile,
        level: DvLevel,
        compat_id: DvCompatibilityId,
        is_valid: bool,
    ) -> Self {
        Self {
            profile,
            level,
            compat_id,
            is_valid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_layer_profiles() {
        assert!(!DvProfile::P7.is_single_layer());
        assert!(DvProfile::P4.is_single_layer());
        assert!(DvProfile::P5.is_single_layer());
        assert!(DvProfile::P8_1.is_single_layer());
        assert!(DvProfile::P8_2.is_single_layer());
        assert!(DvProfile::P8_4.is_single_layer());
        assert!(DvProfile::P9_1.is_single_layer());
        assert!(DvProfile::P9_2.is_single_layer());
    }

    #[test]
    fn test_max_nits() {
        assert!((DvProfile::P5.max_nits() - 10_000.0).abs() < f32::EPSILON);
        assert!((DvProfile::P7.max_nits() - 10_000.0).abs() < f32::EPSILON);
        assert!((DvProfile::P8_4.max_nits() - 1000.0).abs() < f32::EPSILON);
        assert!((DvProfile::P9_1.max_nits() - 1000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compat_signal_type() {
        assert_eq!(DvCompatibilityId::None.signal_type(), "DolbyVision-only");
        assert_eq!(DvCompatibilityId::Hdr10.signal_type(), "HDR10");
        assert_eq!(DvCompatibilityId::Sdr.signal_type(), "SDR");
        assert_eq!(DvCompatibilityId::Hlg.signal_type(), "HLG");
    }

    #[test]
    fn test_profile_constraints_p7() {
        let c = DvProfileConstraints::for_profile(&DvProfile::P7);
        assert!(c.supports_el);
        assert_eq!(c.max_width, 4096);
        assert_eq!(c.max_height, 2160);
    }

    #[test]
    fn test_profile_constraints_p5_no_el() {
        let c = DvProfileConstraints::for_profile(&DvProfile::P5);
        assert!(!c.supports_el);
    }

    #[test]
    fn test_profile_constraints_p8_1_8k() {
        let c = DvProfileConstraints::for_profile(&DvProfile::P8_1);
        assert_eq!(c.max_width, 7680);
        assert_eq!(c.max_height, 4320);
    }

    #[test]
    fn test_profile_converter_p7_to_p8() {
        assert_eq!(
            ProfileConverter::convert_p7_to_p8(DvProfile::P7),
            Some(DvProfile::P8_1)
        );
    }

    #[test]
    fn test_profile_converter_non_p7_returns_none() {
        assert_eq!(ProfileConverter::convert_p7_to_p8(DvProfile::P5), None);
        assert_eq!(ProfileConverter::convert_p7_to_p8(DvProfile::P8_1), None);
        assert_eq!(ProfileConverter::convert_p7_to_p8(DvProfile::P9_2), None);
    }

    #[test]
    fn test_dv_profile_report_creation() {
        let report = DvProfileReport::new(
            DvProfile::P8_1,
            DvLevel::Level2(1000.0),
            DvCompatibilityId::Hdr10,
            true,
        );
        assert_eq!(report.profile, DvProfile::P8_1);
        assert!(report.is_valid);
    }

    #[test]
    fn test_dv_level_variants() {
        let l1 = DvLevel::Level1(100.0);
        let l6 = DvLevel::Level6;
        assert_ne!(l1, l6);
    }
}

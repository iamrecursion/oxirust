//! HDR metadata structures.
//!
//! This module defines structures for various HDR metadata formats including
//! Mastering Display Color Volume (MDCV), Content Light Level (CLL),
//! HDR10+ dynamic metadata, Dolby Vision metadata, and HLG parameters.

use super::primaries::{Primaries, WhitePoint};

/// Mastering Display Color Volume (MDCV) metadata.
///
/// Defined by SMPTE ST 2086, this metadata describes the color volume
/// and luminance range of the display used for mastering the content.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::{MasteringDisplayColorVolume, ColorPrimaries};
///
/// let mdcv = MasteringDisplayColorVolume {
///     display_primaries: ColorPrimaries::BT2020.primaries(),
///     white_point: ColorPrimaries::BT2020.white_point(),
///     max_luminance: 1000.0,
///     min_luminance: 0.005,
/// };
///
/// assert!(mdcv.max_luminance >= mdcv.min_luminance);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MasteringDisplayColorVolume {
    /// Display primaries (red, green, blue) in CIE 1931 xy coordinates.
    ///
    /// Each primary is represented as (x, y) where 0 ≤ x, y ≤ 1.
    pub display_primaries: Primaries,

    /// White point in CIE 1931 xy coordinates.
    ///
    /// Typically D65 for BT.2020 content: (0.3127, 0.3290).
    pub white_point: WhitePoint,

    /// Maximum display luminance in nits (cd/m²).
    ///
    /// Common values:
    /// - 1000 nits: Standard HDR10 mastering display
    /// - 4000 nits: High-end HDR mastering display
    /// - 10000 nits: ST.2084 reference display
    pub max_luminance: f64,

    /// Minimum display luminance in nits (cd/m²).
    ///
    /// Common values:
    /// - 0.005 nits: High-end HDR display
    /// - 0.05 nits: Consumer HDR display
    /// - 0.0001 nits: Perfect black (theoretical)
    pub min_luminance: f64,
}

impl MasteringDisplayColorVolume {
    /// Creates a new MDCV with standard BT.2020 primaries.
    ///
    /// # Arguments
    ///
    /// * `max_luminance` - Maximum display luminance in nits
    /// * `min_luminance` - Minimum display luminance in nits
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::MasteringDisplayColorVolume;
    ///
    /// let mdcv = MasteringDisplayColorVolume::new_bt2020(1000.0, 0.005);
    /// assert_eq!(mdcv.max_luminance, 1000.0);
    /// ```
    #[must_use]
    pub fn new_bt2020(max_luminance: f64, min_luminance: f64) -> Self {
        Self {
            display_primaries: Primaries {
                red: (0.708, 0.292),
                green: (0.170, 0.797),
                blue: (0.131, 0.046),
            },
            white_point: WhitePoint::D65,
            max_luminance,
            min_luminance,
        }
    }

    /// Creates a new MDCV with DCI-P3 primaries.
    ///
    /// # Arguments
    ///
    /// * `max_luminance` - Maximum display luminance in nits
    /// * `min_luminance` - Minimum display luminance in nits
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::MasteringDisplayColorVolume;
    ///
    /// let mdcv = MasteringDisplayColorVolume::new_dci_p3(1000.0, 0.05);
    /// assert_eq!(mdcv.max_luminance, 1000.0);
    /// ```
    #[must_use]
    pub fn new_dci_p3(max_luminance: f64, min_luminance: f64) -> Self {
        Self {
            display_primaries: Primaries {
                red: (0.680, 0.320),
                green: (0.265, 0.690),
                blue: (0.150, 0.060),
            },
            white_point: WhitePoint::D65,
            max_luminance,
            min_luminance,
        }
    }

    /// Validates that the luminance values are reasonable.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::MasteringDisplayColorVolume;
    ///
    /// let mdcv = MasteringDisplayColorVolume::new_bt2020(1000.0, 0.005);
    /// assert!(mdcv.is_valid());
    ///
    /// let invalid = MasteringDisplayColorVolume::new_bt2020(0.001, 1000.0);
    /// assert!(!invalid.is_valid());
    /// ```
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.max_luminance > self.min_luminance
            && self.min_luminance >= 0.0
            && self.max_luminance <= 10000.0
    }

    /// Returns the dynamic range in stops.
    ///
    /// Calculated as `log2(max_luminance` / `min_luminance`).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::MasteringDisplayColorVolume;
    ///
    /// let mdcv = MasteringDisplayColorVolume::new_bt2020(1000.0, 0.005);
    /// let stops = mdcv.dynamic_range_stops();
    /// assert!((stops - 17.6).abs() < 0.1);
    /// ```
    #[must_use]
    pub fn dynamic_range_stops(&self) -> f64 {
        if self.min_luminance > 0.0 {
            (self.max_luminance / self.min_luminance).log2()
        } else {
            f64::INFINITY
        }
    }
}

/// Content Light Level (CLL) metadata.
///
/// Describes the actual light levels present in the content, as opposed
/// to the capabilities of the mastering display.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::ContentLightLevel;
///
/// let cll = ContentLightLevel {
///     max_cll: 1000,
///     max_fall: 400,
/// };
///
/// assert!(cll.max_cll >= cll.max_fall);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContentLightLevel {
    /// Maximum Content Light Level in nits (cd/m²).
    ///
    /// The maximum light level of any single pixel in the entire content.
    /// This is the brightest pixel that appears anywhere in the video.
    pub max_cll: u16,

    /// Maximum Frame-Average Light Level in nits (cd/m²).
    ///
    /// The maximum average light level of any single frame.
    /// This is calculated as the average brightness of all pixels
    /// in the brightest frame.
    pub max_fall: u16,
}

impl ContentLightLevel {
    /// Creates a new Content Light Level metadata.
    ///
    /// # Arguments
    ///
    /// * `max_cll` - Maximum content light level in nits
    /// * `max_fall` - Maximum frame-average light level in nits
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::ContentLightLevel;
    ///
    /// let cll = ContentLightLevel::new(1000, 400);
    /// assert_eq!(cll.max_cll, 1000);
    /// assert_eq!(cll.max_fall, 400);
    /// ```
    #[must_use]
    pub const fn new(max_cll: u16, max_fall: u16) -> Self {
        Self { max_cll, max_fall }
    }

    /// Validates that the CLL values are reasonable.
    ///
    /// `MaxCLL` should typically be greater than or equal to `MaxFALL`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::ContentLightLevel;
    ///
    /// let cll = ContentLightLevel::new(1000, 400);
    /// assert!(cll.is_valid());
    ///
    /// let suspicious = ContentLightLevel::new(100, 500);
    /// assert!(!suspicious.is_valid());
    /// ```
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.max_cll >= self.max_fall && self.max_cll > 0
    }

    /// Returns true if this represents HDR content.
    ///
    /// Content is considered HDR if `MaxCLL` exceeds typical SDR levels (>300 nits).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::ContentLightLevel;
    ///
    /// let hdr = ContentLightLevel::new(1000, 400);
    /// assert!(hdr.is_hdr());
    ///
    /// let sdr = ContentLightLevel::new(200, 100);
    /// assert!(!sdr.is_hdr());
    /// ```
    #[must_use]
    pub const fn is_hdr(&self) -> bool {
        self.max_cll > 300
    }
}

/// HDR10+ dynamic metadata.
///
/// HDR10+ extends HDR10 with scene-by-scene or frame-by-frame metadata
/// that allows for dynamic tone mapping. This structure represents the
/// metadata carried in SMPTE ST 2094-40 Application #4.
///
/// Note: This is a simplified structure. Full HDR10+ metadata is complex
/// and typically stored as opaque binary data.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::Hdr10PlusMetadata;
///
/// let metadata = Hdr10PlusMetadata {
///     application_version: 1,
///     num_windows: 1,
///     target_max_luminance: 1000,
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Hdr10PlusMetadata {
    /// HDR10+ application version.
    pub application_version: u8,

    /// Number of processing windows (1-3).
    pub num_windows: u8,

    /// Target maximum luminance for the content in nits.
    pub target_max_luminance: u16,
}

impl Hdr10PlusMetadata {
    /// Creates a new HDR10+ metadata with default values.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Hdr10PlusMetadata;
    ///
    /// let metadata = Hdr10PlusMetadata::new(1000);
    /// assert_eq!(metadata.application_version, 1);
    /// ```
    #[must_use]
    pub const fn new(target_max_luminance: u16) -> Self {
        Self {
            application_version: 1,
            num_windows: 1,
            target_max_luminance,
        }
    }

    /// Validates the metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Hdr10PlusMetadata;
    ///
    /// let metadata = Hdr10PlusMetadata::new(1000);
    /// assert!(metadata.is_valid());
    /// ```
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.num_windows > 0 && self.num_windows <= 3 && self.target_max_luminance > 0
    }
}

/// Dolby Vision metadata (read-only).
///
/// Dolby Vision is a proprietary HDR format. This structure contains
/// minimal information extracted from the bitstream for informational
/// purposes only. `OxiMedia` does not support encoding Dolby Vision.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::DolbyVisionMetadata;
///
/// let dv = DolbyVisionMetadata {
///     profile: 5,
///     level: 6,
///     rpu_present: true,
///     el_present: false,
///     bl_present: true,
/// };
///
/// assert!(dv.is_profile_5());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DolbyVisionMetadata {
    /// Dolby Vision profile (4, 5, 7, 8, etc.).
    pub profile: u8,

    /// Dolby Vision level (1-13).
    pub level: u8,

    /// Reference Processing Unit (RPU) is present.
    pub rpu_present: bool,

    /// Enhancement Layer (EL) is present.
    pub el_present: bool,

    /// Base Layer (BL) is present.
    pub bl_present: bool,
}

impl DolbyVisionMetadata {
    /// Creates new Dolby Vision metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::DolbyVisionMetadata;
    ///
    /// let dv = DolbyVisionMetadata::new(5, 6);
    /// assert_eq!(dv.profile, 5);
    /// assert_eq!(dv.level, 6);
    /// ```
    #[must_use]
    pub const fn new(profile: u8, level: u8) -> Self {
        Self {
            profile,
            level,
            rpu_present: true,
            el_present: false,
            bl_present: true,
        }
    }

    /// Returns true if this is Profile 5 (`IPTPQc2`).
    ///
    /// Profile 5 is the most common Dolby Vision profile.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::DolbyVisionMetadata;
    ///
    /// let dv = DolbyVisionMetadata::new(5, 6);
    /// assert!(dv.is_profile_5());
    /// ```
    #[must_use]
    pub const fn is_profile_5(&self) -> bool {
        self.profile == 5
    }

    /// Returns true if this is Profile 8 (HDR10 compatible).
    ///
    /// Profile 8 is backward compatible with HDR10.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::DolbyVisionMetadata;
    ///
    /// let dv = DolbyVisionMetadata::new(8, 6);
    /// assert!(dv.is_profile_8());
    /// assert!(dv.is_hdr10_compatible());
    /// ```
    #[must_use]
    pub const fn is_profile_8(&self) -> bool {
        self.profile == 8
    }

    /// Returns true if this profile is backward compatible with HDR10.
    ///
    /// Profiles 7 and 8 are HDR10 compatible.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::DolbyVisionMetadata;
    ///
    /// let dv7 = DolbyVisionMetadata::new(7, 6);
    /// let dv8 = DolbyVisionMetadata::new(8, 6);
    /// let dv5 = DolbyVisionMetadata::new(5, 6);
    ///
    /// assert!(dv7.is_hdr10_compatible());
    /// assert!(dv8.is_hdr10_compatible());
    /// assert!(!dv5.is_hdr10_compatible());
    /// ```
    #[must_use]
    pub const fn is_hdr10_compatible(&self) -> bool {
        self.profile == 7 || self.profile == 8
    }
}

/// HLG (Hybrid Log-Gamma) parameters.
///
/// HLG (ARIB STD-B67) is a broadcast HDR format that is backward
/// compatible with SDR displays.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::HlgParameters;
///
/// let hlg = HlgParameters::new(0);
/// assert_eq!(hlg.application_version, 0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HlgParameters {
    /// Application version (0 = SDR compatible, 1-3 = reserved).
    pub application_version: u8,
}

impl HlgParameters {
    /// Creates new HLG parameters.
    ///
    /// # Arguments
    ///
    /// * `application_version` - HLG application version (typically 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HlgParameters;
    ///
    /// let hlg = HlgParameters::new(0);
    /// assert!(hlg.is_sdr_compatible());
    /// ```
    #[must_use]
    pub const fn new(application_version: u8) -> Self {
        Self {
            application_version,
        }
    }

    /// Returns true if this HLG content is SDR compatible.
    ///
    /// Application version 0 indicates SDR compatibility.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HlgParameters;
    ///
    /// let hlg = HlgParameters::new(0);
    /// assert!(hlg.is_sdr_compatible());
    ///
    /// let hlg_hdr = HlgParameters::new(1);
    /// assert!(!hlg_hdr.is_sdr_compatible());
    /// ```
    #[must_use]
    pub const fn is_sdr_compatible(&self) -> bool {
        self.application_version == 0
    }
}

impl Default for HlgParameters {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdcv_creation() {
        let mdcv = MasteringDisplayColorVolume::new_bt2020(1000.0, 0.005);
        assert_eq!(mdcv.max_luminance, 1000.0);
        assert_eq!(mdcv.min_luminance, 0.005);
        assert!(mdcv.is_valid());
    }

    #[test]
    fn test_mdcv_dci_p3() {
        let mdcv = MasteringDisplayColorVolume::new_dci_p3(1000.0, 0.05);
        assert_eq!(mdcv.max_luminance, 1000.0);
        assert!(mdcv.is_valid());
    }

    #[test]
    fn test_mdcv_validation() {
        let valid = MasteringDisplayColorVolume::new_bt2020(1000.0, 0.005);
        assert!(valid.is_valid());

        let invalid = MasteringDisplayColorVolume::new_bt2020(0.001, 1000.0);
        assert!(!invalid.is_valid());

        let too_bright = MasteringDisplayColorVolume::new_bt2020(20000.0, 0.005);
        assert!(!too_bright.is_valid());
    }

    #[test]
    fn test_mdcv_dynamic_range() {
        let mdcv = MasteringDisplayColorVolume::new_bt2020(1000.0, 0.005);
        let stops = mdcv.dynamic_range_stops();
        assert!((stops - 17.6).abs() < 0.1);
    }

    #[test]
    fn test_cll_creation() {
        let cll = ContentLightLevel::new(1000, 400);
        assert_eq!(cll.max_cll, 1000);
        assert_eq!(cll.max_fall, 400);
        assert!(cll.is_valid());
    }

    #[test]
    fn test_cll_validation() {
        let valid = ContentLightLevel::new(1000, 400);
        assert!(valid.is_valid());

        let suspicious = ContentLightLevel::new(100, 500);
        assert!(!suspicious.is_valid());

        let zero = ContentLightLevel::new(0, 0);
        assert!(!zero.is_valid());
    }

    #[test]
    fn test_cll_is_hdr() {
        let hdr = ContentLightLevel::new(1000, 400);
        assert!(hdr.is_hdr());

        let sdr = ContentLightLevel::new(200, 100);
        assert!(!sdr.is_hdr());
    }

    #[test]
    fn test_hdr10_plus_creation() {
        let metadata = Hdr10PlusMetadata::new(1000);
        assert_eq!(metadata.application_version, 1);
        assert_eq!(metadata.num_windows, 1);
        assert_eq!(metadata.target_max_luminance, 1000);
        assert!(metadata.is_valid());
    }

    #[test]
    fn test_dolby_vision_profiles() {
        let dv5 = DolbyVisionMetadata::new(5, 6);
        assert!(dv5.is_profile_5());
        assert!(!dv5.is_profile_8());
        assert!(!dv5.is_hdr10_compatible());

        let dv8 = DolbyVisionMetadata::new(8, 6);
        assert!(!dv8.is_profile_5());
        assert!(dv8.is_profile_8());
        assert!(dv8.is_hdr10_compatible());

        let dv7 = DolbyVisionMetadata::new(7, 6);
        assert!(dv7.is_hdr10_compatible());
    }

    #[test]
    fn test_hlg_parameters() {
        let hlg = HlgParameters::new(0);
        assert_eq!(hlg.application_version, 0);
        assert!(hlg.is_sdr_compatible());

        let hlg_hdr = HlgParameters::new(1);
        assert!(!hlg_hdr.is_sdr_compatible());
    }

    #[test]
    fn test_hlg_default() {
        let hlg = HlgParameters::default();
        assert_eq!(hlg.application_version, 0);
        assert!(hlg.is_sdr_compatible());
    }
}

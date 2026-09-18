//! Color primaries and white points for color space definitions.
//!
//! This module defines color primaries in CIE 1931 xy chromaticity coordinates
//! and provides common color spaces (BT.709, BT.2020, DCI-P3, Display P3).
#![allow(clippy::match_same_arms)]

/// Color primaries in CIE 1931 xy chromaticity coordinates.
///
/// Represents the red, green, and blue primaries of a color space.
/// Each primary is a point in the CIE 1931 xy chromaticity diagram.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::Primaries;
///
/// let bt709 = Primaries {
///     red: (0.64, 0.33),
///     green: (0.30, 0.60),
///     blue: (0.15, 0.06),
/// };
///
/// assert_eq!(bt709.red.0, 0.64);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Primaries {
    /// Red primary (x, y) in CIE 1931 xy chromaticity.
    pub red: (f64, f64),
    /// Green primary (x, y) in CIE 1931 xy chromaticity.
    pub green: (f64, f64),
    /// Blue primary (x, y) in CIE 1931 xy chromaticity.
    pub blue: (f64, f64),
}

impl Primaries {
    /// Creates new color primaries.
    ///
    /// # Arguments
    ///
    /// * `red` - Red primary (x, y)
    /// * `green` - Green primary (x, y)
    /// * `blue` - Blue primary (x, y)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Primaries;
    ///
    /// let primaries = Primaries::new((0.64, 0.33), (0.30, 0.60), (0.15, 0.06));
    /// assert_eq!(primaries.red, (0.64, 0.33));
    /// ```
    #[must_use]
    pub const fn new(red: (f64, f64), green: (f64, f64), blue: (f64, f64)) -> Self {
        Self { red, green, blue }
    }

    /// BT.709 / sRGB primaries (standard HD).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Primaries;
    ///
    /// let primaries = Primaries::bt709();
    /// assert_eq!(primaries.red, (0.64, 0.33));
    /// ```
    #[must_use]
    pub const fn bt709() -> Self {
        Self {
            red: (0.64, 0.33),
            green: (0.30, 0.60),
            blue: (0.15, 0.06),
        }
    }

    /// BT.2020 primaries (wide color gamut for UHD/HDR).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Primaries;
    ///
    /// let primaries = Primaries::bt2020();
    /// assert_eq!(primaries.red, (0.708, 0.292));
    /// ```
    #[must_use]
    pub const fn bt2020() -> Self {
        Self {
            red: (0.708, 0.292),
            green: (0.170, 0.797),
            blue: (0.131, 0.046),
        }
    }

    /// DCI-P3 primaries (digital cinema).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Primaries;
    ///
    /// let primaries = Primaries::dci_p3();
    /// assert_eq!(primaries.red, (0.680, 0.320));
    /// ```
    #[must_use]
    pub const fn dci_p3() -> Self {
        Self {
            red: (0.680, 0.320),
            green: (0.265, 0.690),
            blue: (0.150, 0.060),
        }
    }

    /// Display P3 primaries (Apple displays).
    ///
    /// Same as DCI-P3 primaries but typically used with D65 white point
    /// instead of DCI white point.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Primaries;
    ///
    /// let primaries = Primaries::display_p3();
    /// assert_eq!(primaries.red, (0.680, 0.320));
    /// ```
    #[must_use]
    pub const fn display_p3() -> Self {
        Self::dci_p3()
    }

    /// Adobe RGB primaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Primaries;
    ///
    /// let primaries = Primaries::adobe_rgb();
    /// assert_eq!(primaries.red, (0.64, 0.33));
    /// ```
    #[must_use]
    pub const fn adobe_rgb() -> Self {
        Self {
            red: (0.64, 0.33),
            green: (0.21, 0.71),
            blue: (0.15, 0.06),
        }
    }

    /// Validates that all primaries are within valid range [0, 1].
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Primaries;
    ///
    /// let valid = Primaries::bt709();
    /// assert!(valid.is_valid());
    ///
    /// let invalid = Primaries::new((1.5, 0.5), (0.3, 0.6), (0.15, 0.06));
    /// assert!(!invalid.is_valid());
    /// ```
    #[must_use]
    pub fn is_valid(&self) -> bool {
        fn in_range(p: (f64, f64)) -> bool {
            p.0 >= 0.0 && p.0 <= 1.0 && p.1 >= 0.0 && p.1 <= 1.0
        }

        in_range(self.red) && in_range(self.green) && in_range(self.blue)
    }

    /// Calculates the area of the color gamut in xy chromaticity space.
    ///
    /// Larger area indicates wider color gamut.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Primaries;
    ///
    /// let bt709 = Primaries::bt709();
    /// let bt2020 = Primaries::bt2020();
    ///
    /// // BT.2020 has wider gamut than BT.709
    /// assert!(bt2020.gamut_area() > bt709.gamut_area());
    /// ```
    #[must_use]
    pub fn gamut_area(&self) -> f64 {
        // Calculate area using cross product (shoelace formula)
        let (rx, ry) = self.red;
        let (gx, gy) = self.green;
        let (bx, by) = self.blue;

        0.5 * ((rx * gy - gx * ry) + (gx * by - bx * gy) + (bx * ry - rx * by)).abs()
    }
}

/// White point in CIE 1931 xy chromaticity coordinates.
///
/// The white point defines what "white" looks like in a color space.
/// Different illuminants produce different white points.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::WhitePoint;
///
/// let d65 = WhitePoint::D65;
/// assert_eq!(d65.xy(), (0.3127, 0.3290));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum WhitePoint {
    /// D65 - Daylight illuminant at 6500K.
    ///
    /// Standard for BT.709, BT.2020, sRGB, and most modern displays.
    #[default]
    D65,

    /// D50 - Daylight illuminant at 5000K.
    ///
    /// Used in some print workflows and color management.
    D50,

    /// DCI - Digital Cinema white point.
    ///
    /// Approximately 5900K, used in DCI-P3 color space.
    Dci,

    /// Custom white point with (x, y) coordinates.
    Custom(f64, f64),
}

impl WhitePoint {
    /// Returns the xy chromaticity coordinates of the white point.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::WhitePoint;
    ///
    /// let d65 = WhitePoint::D65;
    /// assert_eq!(d65.xy(), (0.3127, 0.3290));
    ///
    /// let custom = WhitePoint::Custom(0.33, 0.33);
    /// assert_eq!(custom.xy(), (0.33, 0.33));
    /// ```
    #[must_use]
    pub const fn xy(&self) -> (f64, f64) {
        match self {
            Self::D65 => (0.3127, 0.3290),
            Self::D50 => (0.3457, 0.3585),
            Self::Dci => (0.314, 0.351),
            Self::Custom(x, y) => (*x, *y),
        }
    }

    /// Returns the correlated color temperature in Kelvin (approximate).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::WhitePoint;
    ///
    /// assert_eq!(WhitePoint::D65.cct(), 6500);
    /// assert_eq!(WhitePoint::D50.cct(), 5000);
    /// ```
    #[must_use]
    pub const fn cct(&self) -> u32 {
        match self {
            Self::D65 => 6500,
            Self::D50 => 5000,
            Self::Dci => 5900,
            Self::Custom(_, _) => 6500, // Default
        }
    }
}

/// Standard color primaries definitions.
///
/// This enum provides convenient access to common color spaces
/// with their associated primaries and white points.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::ColorPrimaries;
///
/// let bt709 = ColorPrimaries::BT709;
/// assert_eq!(bt709.primaries().red, (0.64, 0.33));
/// assert_eq!(bt709.white_point().xy(), (0.3127, 0.3290));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ColorPrimaries {
    /// BT.709 / Rec.709 / sRGB.
    ///
    /// Standard for HDTV and web content.
    #[default]
    BT709,

    /// BT.2020 / Rec.2020.
    ///
    /// Wide color gamut standard for UHDTV and HDR.
    BT2020,

    /// DCI-P3 (Digital Cinema Initiative).
    ///
    /// Standard for digital cinema projection.
    DciP3,

    /// Display P3.
    ///
    /// DCI-P3 primaries with D65 white point (Apple displays).
    DisplayP3,

    /// Adobe RGB (1998).
    ///
    /// Wide gamut color space for photography and print.
    AdobeRgb,

    /// BT.470 System M (NTSC).
    ///
    /// Legacy NTSC color space.
    Bt470M,

    /// BT.470 System B, G (PAL/SECAM).
    ///
    /// Legacy PAL/SECAM color space.
    Bt470Bg,

    /// SMPTE 170M (NTSC).
    ///
    /// Same as BT.601 for NTSC.
    Smpte170M,

    /// SMPTE 240M.
    ///
    /// Legacy HDTV standard (superseded by BT.709).
    Smpte240M,

    /// Film (color negative).
    Film,

    /// BT.2100 (same as BT.2020).
    ///
    /// HDR television standard.
    Bt2100,

    /// Custom primaries.
    Custom,
}

impl ColorPrimaries {
    /// Returns the color primaries for this color space.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::ColorPrimaries;
    ///
    /// let primaries = ColorPrimaries::BT709.primaries();
    /// assert_eq!(primaries.red, (0.64, 0.33));
    /// ```
    #[must_use]
    pub const fn primaries(&self) -> Primaries {
        match self {
            Self::BT709 => Primaries::bt709(),
            Self::BT2020 | Self::Bt2100 => Primaries::bt2020(),
            Self::DciP3 | Self::DisplayP3 => Primaries::dci_p3(),
            Self::AdobeRgb => Primaries::adobe_rgb(),
            Self::Bt470M => Primaries::new((0.67, 0.33), (0.21, 0.71), (0.14, 0.08)),
            Self::Bt470Bg | Self::Smpte170M => {
                Primaries::new((0.64, 0.33), (0.29, 0.60), (0.15, 0.06))
            }
            Self::Smpte240M => Primaries::new((0.63, 0.34), (0.31, 0.595), (0.155, 0.070)),
            Self::Film => Primaries::new((0.681, 0.319), (0.243, 0.692), (0.145, 0.049)),
            Self::Custom => Primaries::bt709(), // Default
        }
    }

    /// Returns the white point for this color space.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::{ColorPrimaries, WhitePoint};
    ///
    /// let wp = ColorPrimaries::BT709.white_point();
    /// assert_eq!(wp, WhitePoint::D65);
    /// ```
    #[must_use]
    pub const fn white_point(&self) -> WhitePoint {
        match self {
            Self::BT709
            | Self::BT2020
            | Self::DisplayP3
            | Self::AdobeRgb
            | Self::Bt470Bg
            | Self::Smpte170M
            | Self::Smpte240M
            | Self::Bt2100 => WhitePoint::D65,
            Self::DciP3 => WhitePoint::Dci,
            Self::Bt470M => WhitePoint::Custom(0.310, 0.316), // Illuminant C
            Self::Film => WhitePoint::Custom(0.310, 0.316),   // Illuminant C
            Self::Custom => WhitePoint::D65,                  // Default
        }
    }

    /// Returns true if this is a wide color gamut.
    ///
    /// Wide color gamuts include BT.2020, DCI-P3, Display P3, and Adobe RGB.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::ColorPrimaries;
    ///
    /// assert!(ColorPrimaries::BT2020.is_wide_gamut());
    /// assert!(ColorPrimaries::DciP3.is_wide_gamut());
    /// assert!(!ColorPrimaries::BT709.is_wide_gamut());
    /// ```
    #[must_use]
    pub const fn is_wide_gamut(&self) -> bool {
        matches!(
            self,
            Self::BT2020 | Self::DciP3 | Self::DisplayP3 | Self::AdobeRgb | Self::Bt2100
        )
    }

    /// Returns a human-readable name for this color space.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::ColorPrimaries;
    ///
    /// assert_eq!(ColorPrimaries::BT709.name(), "BT.709 / Rec.709");
    /// assert_eq!(ColorPrimaries::BT2020.name(), "BT.2020 / Rec.2020");
    /// ```
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::BT709 => "BT.709 / Rec.709",
            Self::BT2020 => "BT.2020 / Rec.2020",
            Self::DciP3 => "DCI-P3",
            Self::DisplayP3 => "Display P3",
            Self::AdobeRgb => "Adobe RGB (1998)",
            Self::Bt470M => "BT.470 System M",
            Self::Bt470Bg => "BT.470 System B, G",
            Self::Smpte170M => "SMPTE 170M",
            Self::Smpte240M => "SMPTE 240M",
            Self::Film => "Film",
            Self::Bt2100 => "BT.2100",
            Self::Custom => "Custom",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primaries_creation() {
        let primaries = Primaries::new((0.64, 0.33), (0.30, 0.60), (0.15, 0.06));
        assert_eq!(primaries.red, (0.64, 0.33));
        assert_eq!(primaries.green, (0.30, 0.60));
        assert_eq!(primaries.blue, (0.15, 0.06));
    }

    #[test]
    fn test_bt709_primaries() {
        let primaries = Primaries::bt709();
        assert_eq!(primaries.red, (0.64, 0.33));
        assert_eq!(primaries.green, (0.30, 0.60));
        assert_eq!(primaries.blue, (0.15, 0.06));
        assert!(primaries.is_valid());
    }

    #[test]
    fn test_bt2020_primaries() {
        let primaries = Primaries::bt2020();
        assert_eq!(primaries.red, (0.708, 0.292));
        assert!(primaries.is_valid());
    }

    #[test]
    fn test_primaries_validation() {
        let valid = Primaries::bt709();
        assert!(valid.is_valid());

        let invalid = Primaries::new((1.5, 0.5), (0.3, 0.6), (0.15, 0.06));
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_gamut_area() {
        let bt709 = Primaries::bt709();
        let bt2020 = Primaries::bt2020();

        let area_709 = bt709.gamut_area();
        let area_2020 = bt2020.gamut_area();

        assert!(area_709 > 0.0);
        assert!(area_2020 > area_709);
    }

    #[test]
    fn test_white_point_d65() {
        let wp = WhitePoint::D65;
        assert_eq!(wp.xy(), (0.3127, 0.3290));
        assert_eq!(wp.cct(), 6500);
    }

    #[test]
    fn test_white_point_d50() {
        let wp = WhitePoint::D50;
        assert_eq!(wp.xy(), (0.3457, 0.3585));
        assert_eq!(wp.cct(), 5000);
    }

    #[test]
    fn test_white_point_custom() {
        let wp = WhitePoint::Custom(0.33, 0.33);
        assert_eq!(wp.xy(), (0.33, 0.33));
    }

    #[test]
    fn test_color_primaries_bt709() {
        let cp = ColorPrimaries::BT709;
        let primaries = cp.primaries();
        assert_eq!(primaries.red, (0.64, 0.33));
        assert_eq!(cp.white_point(), WhitePoint::D65);
        assert!(!cp.is_wide_gamut());
    }

    #[test]
    fn test_color_primaries_bt2020() {
        let cp = ColorPrimaries::BT2020;
        let primaries = cp.primaries();
        assert_eq!(primaries.red, (0.708, 0.292));
        assert!(cp.is_wide_gamut());
    }

    #[test]
    fn test_color_primaries_dci_p3() {
        let cp = ColorPrimaries::DciP3;
        assert!(cp.is_wide_gamut());
        assert_eq!(cp.white_point(), WhitePoint::Dci);
    }

    #[test]
    fn test_color_primaries_display_p3() {
        let cp = ColorPrimaries::DisplayP3;
        assert!(cp.is_wide_gamut());
        assert_eq!(cp.white_point(), WhitePoint::D65);
    }

    #[test]
    fn test_color_primaries_names() {
        assert_eq!(ColorPrimaries::BT709.name(), "BT.709 / Rec.709");
        assert_eq!(ColorPrimaries::BT2020.name(), "BT.2020 / Rec.2020");
        assert_eq!(ColorPrimaries::DciP3.name(), "DCI-P3");
    }
}

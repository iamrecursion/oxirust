//! White point definitions and chromatic adaptation utilities.
//!
//! Provides standard illuminant white points and conversion utilities
//! for professional color management workflows.

#![allow(dead_code)]

/// Standard illuminant white points used in color science.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WhitePoint {
    /// CIE D50 illuminant (horizon light, 5003 K).
    D50,
    /// CIE D60 illuminant (6000 K, used in ACES).
    D60,
    /// CIE D65 illuminant (noon daylight, 6504 K). sRGB/Rec.709 reference.
    D65,
    /// DCI-P3 white point (6300 K, cinema projection).
    DciP3,
    /// ACES white point (D60 alias used in ACES workflows).
    Aces,
    /// CIE Standard Illuminant A (tungsten, ~2856 K).
    A,
    /// CIE Standard Illuminant E (equal-energy, theoretical).
    E,
}

impl WhitePoint {
    /// Returns the CIE xy chromaticity coordinates `(x, y)` for this white point.
    #[must_use]
    pub fn xy_chromaticity(self) -> (f64, f64) {
        match self {
            Self::D50 => (0.3457, 0.3585),
            Self::D60 => (0.3217, 0.3379),
            Self::D65 => (0.3127, 0.3290),
            Self::DciP3 => (0.3140, 0.3510),
            Self::Aces => (0.3217, 0.3379),
            Self::A => (0.4476, 0.4074),
            Self::E => (1.0 / 3.0, 1.0 / 3.0),
        }
    }

    /// Returns the correlated color temperature in Kelvin.
    #[must_use]
    pub fn correlated_color_temp_k(self) -> u32 {
        match self {
            Self::D50 => 5003,
            Self::D60 => 6000,
            Self::D65 => 6504,
            Self::DciP3 => 6300,
            Self::Aces => 6000,
            Self::A => 2856,
            Self::E => 5455,
        }
    }

    /// Returns the XYZ tristimulus values normalized so Y = 1.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn xyz(self) -> (f64, f64, f64) {
        let (x, y) = self.xy_chromaticity();
        let big_x = x / y;
        let big_y = 1.0_f64;
        let big_z = (1.0 - x - y) / y;
        (big_x, big_y, big_z)
    }

    /// Returns a human-readable name for this white point.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::D50 => "D50",
            Self::D60 => "D60",
            Self::D65 => "D65",
            Self::DciP3 => "DCI-P3",
            Self::Aces => "ACES",
            Self::A => "Illuminant A",
            Self::E => "Equal Energy E",
        }
    }

    /// Returns `true` if this white point is a daylight illuminant.
    #[must_use]
    pub fn is_daylight(self) -> bool {
        matches!(self, Self::D50 | Self::D60 | Self::D65 | Self::Aces)
    }
}

/// Converts between two white points using a Bradford adaptation matrix.
#[derive(Debug, Clone, Copy)]
pub struct WhitePointConverter {
    /// Source white point.
    pub src: WhitePoint,
    /// Destination white point.
    pub dst: WhitePoint,
}

impl WhitePointConverter {
    /// Creates a new `WhitePointConverter`.
    #[must_use]
    pub const fn new(src: WhitePoint, dst: WhitePoint) -> Self {
        Self { src, dst }
    }

    /// Adapts an XYZ color from the source white point to the destination.
    ///
    /// Uses a simplified von Kries-style scaling for efficiency.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn adapt(self, xyz: (f64, f64, f64)) -> (f64, f64, f64) {
        let (sx, _sy, sz) = self.src.xyz();
        let (dx, _dy, dz) = self.dst.xyz();
        // Simplified von Kries diagonal adaptation
        let scale_x = dx / sx;
        let scale_z = dz / sz;
        (xyz.0 * scale_x, xyz.1, xyz.2 * scale_z)
    }

    /// Returns `true` when source and destination are the same white point.
    #[must_use]
    pub fn is_identity(self) -> bool {
        self.src == self.dst
    }

    /// Returns the chromatic shift magnitude (ΔE-like proxy) between the two white points.
    #[must_use]
    pub fn chromatic_shift(self) -> f64 {
        let (sx, sy) = self.src.xy_chromaticity();
        let (dx, dy) = self.dst.xy_chromaticity();
        let ddx = dx - sx;
        let ddy = dy - sy;
        (ddx * ddx + ddy * ddy).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d65_chromaticity() {
        let (x, y) = WhitePoint::D65.xy_chromaticity();
        assert!((x - 0.3127).abs() < 1e-4);
        assert!((y - 0.3290).abs() < 1e-4);
    }

    #[test]
    fn test_d50_chromaticity() {
        let (x, y) = WhitePoint::D50.xy_chromaticity();
        assert!((x - 0.3457).abs() < 1e-4);
        assert!((y - 0.3585).abs() < 1e-4);
    }

    #[test]
    fn test_d65_cct() {
        assert_eq!(WhitePoint::D65.correlated_color_temp_k(), 6504);
    }

    #[test]
    fn test_d50_cct() {
        assert_eq!(WhitePoint::D50.correlated_color_temp_k(), 5003);
    }

    #[test]
    fn test_aces_is_d60_alias() {
        assert_eq!(
            WhitePoint::Aces.correlated_color_temp_k(),
            WhitePoint::D60.correlated_color_temp_k()
        );
        assert_eq!(
            WhitePoint::Aces.xy_chromaticity(),
            WhitePoint::D60.xy_chromaticity()
        );
    }

    #[test]
    fn test_xyz_y_is_one() {
        for wp in [
            WhitePoint::D50,
            WhitePoint::D65,
            WhitePoint::D60,
            WhitePoint::A,
            WhitePoint::E,
        ] {
            let (_x, y, _z) = wp.xyz();
            assert!((y - 1.0).abs() < 1e-10, "{} Y != 1", wp.name());
        }
    }

    #[test]
    fn test_is_daylight() {
        assert!(WhitePoint::D65.is_daylight());
        assert!(WhitePoint::D50.is_daylight());
        assert!(WhitePoint::Aces.is_daylight());
        assert!(!WhitePoint::A.is_daylight());
        assert!(!WhitePoint::E.is_daylight());
    }

    #[test]
    fn test_names_non_empty() {
        for wp in [
            WhitePoint::D50,
            WhitePoint::D60,
            WhitePoint::D65,
            WhitePoint::DciP3,
            WhitePoint::Aces,
            WhitePoint::A,
            WhitePoint::E,
        ] {
            assert!(!wp.name().is_empty());
        }
    }

    #[test]
    fn test_converter_identity() {
        let conv = WhitePointConverter::new(WhitePoint::D65, WhitePoint::D65);
        assert!(conv.is_identity());
        let xyz = (0.95047, 1.0, 1.08883);
        let adapted = conv.adapt(xyz);
        assert!((adapted.0 - xyz.0).abs() < 1e-6);
        assert!((adapted.1 - xyz.1).abs() < 1e-6);
        assert!((adapted.2 - xyz.2).abs() < 1e-6);
    }

    #[test]
    fn test_converter_non_identity() {
        let conv = WhitePointConverter::new(WhitePoint::D65, WhitePoint::D50);
        assert!(!conv.is_identity());
    }

    #[test]
    fn test_chromatic_shift_identity_is_zero() {
        let conv = WhitePointConverter::new(WhitePoint::D65, WhitePoint::D65);
        assert!(conv.chromatic_shift() < 1e-10);
    }

    #[test]
    fn test_chromatic_shift_d50_d65_positive() {
        let conv = WhitePointConverter::new(WhitePoint::D50, WhitePoint::D65);
        assert!(conv.chromatic_shift() > 0.0);
    }

    #[test]
    fn test_adapt_preserves_y() {
        let conv = WhitePointConverter::new(WhitePoint::D65, WhitePoint::D50);
        let xyz = (0.5, 0.7, 0.3);
        let adapted = conv.adapt(xyz);
        assert!((adapted.1 - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_dci_p3_cct() {
        assert_eq!(WhitePoint::DciP3.correlated_color_temp_k(), 6300);
    }
}

//! CIE LCH (Lightness, Chroma, Hue) color space.

use super::{Lab, Xyz};

/// CIE LCH color (cylindrical representation of Lab).
///
/// LCH is Lab converted to cylindrical coordinates, making it easier
/// to reason about hue, saturation, and lightness independently.
///
/// - L: Lightness [0, 100]
/// - C: Chroma (saturation/colorfulness) [0, ~130]
/// - H: Hue angle in degrees [0, 360)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lch {
    /// Lightness [0, 100]
    pub l: f64,
    /// Chroma (saturation) [0, ~130]
    pub c: f64,
    /// Hue angle in degrees [0, 360)
    pub h: f64,
}

impl Lch {
    /// Creates a new LCH color.
    #[must_use]
    pub const fn new(l: f64, c: f64, h: f64) -> Self {
        Self { l, c, h }
    }

    /// Converts from Lab to LCH.
    #[must_use]
    pub fn from_lab(lab: &Lab) -> Self {
        let c = (lab.a * lab.a + lab.b * lab.b).sqrt();
        let mut h = lab.b.atan2(lab.a).to_degrees();
        if h < 0.0 {
            h += 360.0;
        }

        Self { l: lab.l, c, h }
    }

    /// Converts LCH to Lab.
    #[must_use]
    pub fn to_lab(&self) -> Lab {
        let h_rad = self.h.to_radians();
        Lab {
            l: self.l,
            a: self.c * h_rad.cos(),
            b: self.c * h_rad.sin(),
        }
    }

    /// Converts from XYZ to LCH.
    #[must_use]
    pub fn from_xyz(xyz: &Xyz, white_point: &Xyz) -> Self {
        let lab = Lab::from_xyz(xyz, white_point);
        Self::from_lab(&lab)
    }

    /// Converts LCH to XYZ.
    #[must_use]
    pub fn to_xyz(&self, white_point: &Xyz) -> Xyz {
        self.to_lab().to_xyz(white_point)
    }

    /// Normalizes the hue angle to [0, 360) range.
    #[must_use]
    pub fn normalize_hue(&self) -> Self {
        let mut h = self.h % 360.0;
        if h < 0.0 {
            h += 360.0;
        }
        Self {
            l: self.l,
            c: self.c,
            h,
        }
    }

    /// Returns the hue difference between this and another LCH color.
    ///
    /// The difference is in the range [-180, 180].
    #[must_use]
    pub fn hue_difference(&self, other: &Self) -> f64 {
        let mut diff = other.h - self.h;
        if diff > 180.0 {
            diff -= 360.0;
        } else if diff < -180.0 {
            diff += 360.0;
        }
        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lch_creation() {
        let lch = Lch::new(50.0, 30.0, 120.0);
        assert_eq!(lch.l, 50.0);
        assert_eq!(lch.c, 30.0);
        assert_eq!(lch.h, 120.0);
    }

    #[test]
    fn test_lch_lab_roundtrip() {
        let lab = Lab::new(50.0, 30.0, 40.0);
        let lch = Lch::from_lab(&lab);
        let lab2 = lch.to_lab();

        assert!((lab2.l - lab.l).abs() < 1e-10);
        assert!((lab2.a - lab.a).abs() < 1e-10);
        assert!((lab2.b - lab.b).abs() < 1e-10);
    }

    #[test]
    fn test_lch_from_lab_values() {
        let lab = Lab::new(50.0, 30.0, 40.0);
        let lch = Lch::from_lab(&lab);

        assert!((lch.l - 50.0).abs() < 1e-10);
        assert!((lch.c - 50.0).abs() < 1e-10); // sqrt(30^2 + 40^2) = 50
    }

    #[test]
    fn test_normalize_hue() {
        let lch = Lch::new(50.0, 30.0, -30.0);
        let normalized = lch.normalize_hue();
        assert!((normalized.h - 330.0).abs() < 1e-10);

        let lch2 = Lch::new(50.0, 30.0, 400.0);
        let normalized2 = lch2.normalize_hue();
        assert!((normalized2.h - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_hue_difference() {
        let lch1 = Lch::new(50.0, 30.0, 10.0);
        let lch2 = Lch::new(50.0, 30.0, 350.0);

        let diff = lch1.hue_difference(&lch2);
        assert!((diff - (-20.0)).abs() < 1e-10);
    }
}

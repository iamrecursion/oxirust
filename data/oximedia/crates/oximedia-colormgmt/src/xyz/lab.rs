//! CIE L*a*b* (Lab) color space.

use super::Xyz;

/// CIE L*a*b* color (1976).
///
/// Lab is designed to be perceptually uniform - equal distances in Lab space
/// should correspond to equal perceived color differences.
///
/// - L*: Lightness (0 = black, 100 = white)
/// - a*: Green-red axis (negative = green, positive = red)
/// - b*: Blue-yellow axis (negative = blue, positive = yellow)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lab {
    /// Lightness [0, 100]
    pub l: f64,
    /// Green-red axis [-128, 128]
    pub a: f64,
    /// Blue-yellow axis [-128, 128]
    pub b: f64,
}

impl Lab {
    /// Creates a new Lab color.
    #[must_use]
    pub const fn new(l: f64, a: f64, b: f64) -> Self {
        Self { l, a, b }
    }

    /// Converts from CIE XYZ to Lab.
    ///
    /// # Arguments
    ///
    /// * `xyz` - XYZ color
    /// * `white_point` - Reference white point in XYZ (typically D65 or D50)
    #[must_use]
    pub fn from_xyz(xyz: &Xyz, white_point: &Xyz) -> Self {
        let x = lab_f(xyz.x / white_point.x);
        let y = lab_f(xyz.y / white_point.y);
        let z = lab_f(xyz.z / white_point.z);

        Self {
            l: 116.0 * y - 16.0,
            a: 500.0 * (x - y),
            b: 200.0 * (y - z),
        }
    }

    /// Converts Lab to CIE XYZ.
    ///
    /// # Arguments
    ///
    /// * `white_point` - Reference white point in XYZ
    #[must_use]
    pub fn to_xyz(&self, white_point: &Xyz) -> Xyz {
        let fy = (self.l + 16.0) / 116.0;
        let fx = fy + self.a / 500.0;
        let fz = fy - self.b / 200.0;

        Xyz {
            x: white_point.x * lab_f_inv(fx),
            y: white_point.y * lab_f_inv(fy),
            z: white_point.z * lab_f_inv(fz),
        }
    }

    /// Converts Lab to LCH (cylindrical representation).
    #[must_use]
    pub fn to_lch(&self) -> super::Lch {
        super::Lch::from_lab(self)
    }

    /// Returns the chroma (colorfulness).
    #[must_use]
    pub fn chroma(&self) -> f64 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    /// Returns the hue angle in degrees [0, 360).
    #[must_use]
    pub fn hue(&self) -> f64 {
        let mut h = self.b.atan2(self.a).to_degrees();
        if h < 0.0 {
            h += 360.0;
        }
        h
    }
}

/// Lab f(t) function for XYZ to Lab conversion.
///
/// Reference: CIE 1976 L*a*b*
#[must_use]
fn lab_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    const DELTA_CUBED: f64 = DELTA * DELTA * DELTA;

    if t > DELTA_CUBED {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// Lab f^-1(t) function for Lab to XYZ conversion.
#[must_use]
fn lab_f_inv(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;

    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lab_creation() {
        let lab = Lab::new(50.0, 10.0, -20.0);
        assert_eq!(lab.l, 50.0);
        assert_eq!(lab.a, 10.0);
        assert_eq!(lab.b, -20.0);
    }

    #[test]
    fn test_lab_xyz_roundtrip() {
        let xyz = Xyz::new(0.5, 0.6, 0.7);
        let white = Xyz::d65();

        let lab = Lab::from_xyz(&xyz, &white);
        let xyz2 = lab.to_xyz(&white);

        assert!((xyz2.x - xyz.x).abs() < 1e-6);
        assert!((xyz2.y - xyz.y).abs() < 1e-6);
        assert!((xyz2.z - xyz.z).abs() < 1e-6);
    }

    #[test]
    fn test_lab_white() {
        let white = Xyz::d65();
        let lab = Lab::from_xyz(&white, &white);

        // White should be L=100, a=0, b=0
        assert!((lab.l - 100.0).abs() < 1e-6);
        assert!(lab.a.abs() < 1e-6);
        assert!(lab.b.abs() < 1e-6);
    }

    #[test]
    fn test_lab_black() {
        let black = Xyz::new(0.0, 0.0, 0.0);
        let white = Xyz::d65();
        let lab = Lab::from_xyz(&black, &white);

        // Black should be L=0
        assert!((lab.l - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_lab_chroma() {
        let lab = Lab::new(50.0, 30.0, 40.0);
        let chroma = lab.chroma();
        assert!((chroma - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_lab_hue() {
        let lab = Lab::new(50.0, 0.0, 50.0);
        let hue = lab.hue();
        assert!((hue - 90.0).abs() < 1e-6);
    }
}

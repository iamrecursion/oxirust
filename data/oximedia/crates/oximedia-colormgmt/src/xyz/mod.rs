//! CIE XYZ color space and derived spaces (Lab, Luv, LCH).

pub mod lab;
pub mod lch;
pub mod luv;

pub use lab::Lab;
pub use lch::Lch;
pub use luv::Luv;

/// CIE XYZ color (1931).
///
/// Device-independent color representation. XYZ values are typically
/// normalized so that Y = 1.0 represents the reference white.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xyz {
    /// X component (roughly red-green)
    pub x: f64,
    /// Y component (luminance)
    pub y: f64,
    /// Z component (roughly blue-yellow)
    pub z: f64,
}

impl Xyz {
    /// Creates a new XYZ color.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Returns the XYZ values as an array.
    #[must_use]
    pub const fn as_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Creates XYZ color from an array.
    #[must_use]
    pub const fn from_array(xyz: [f64; 3]) -> Self {
        Self {
            x: xyz[0],
            y: xyz[1],
            z: xyz[2],
        }
    }

    /// Converts to xyY chromaticity coordinates.
    ///
    /// # Returns
    ///
    /// (x, y, Y) where x and y are chromaticity coordinates and Y is luminance.
    #[must_use]
    pub fn to_xyy(&self) -> (f64, f64, f64) {
        let sum = self.x + self.y + self.z;
        if sum < 1e-10 {
            return (0.0, 0.0, self.y);
        }
        (self.x / sum, self.y / sum, self.y)
    }

    /// Creates XYZ from xyY chromaticity coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - x chromaticity coordinate
    /// * `y` - y chromaticity coordinate
    /// * `big_y` - Y luminance value
    #[must_use]
    pub fn from_xyy(x: f64, y: f64, big_y: f64) -> Self {
        if y < 1e-10 {
            return Self::new(0.0, big_y, 0.0);
        }
        Self {
            x: (big_y / y) * x,
            y: big_y,
            z: (big_y / y) * (1.0 - x - y),
        }
    }

    /// Converts XYZ to CIE L*a*b* (Lab).
    ///
    /// # Arguments
    ///
    /// * `white_point` - Reference white point in XYZ (typically D65 or D50)
    #[must_use]
    pub fn to_lab(&self, white_point: &Xyz) -> Lab {
        Lab::from_xyz(self, white_point)
    }

    /// Converts XYZ to CIE L*u*v* (Luv).
    ///
    /// # Arguments
    ///
    /// * `white_point` - Reference white point in XYZ
    #[must_use]
    pub fn to_luv(&self, white_point: &Xyz) -> Luv {
        Luv::from_xyz(self, white_point)
    }

    /// D65 white point (most common for RGB color spaces).
    #[must_use]
    pub const fn d65() -> Self {
        Self {
            x: 0.95047,
            y: 1.0,
            z: 1.08883,
        }
    }

    /// D50 white point (used in some print workflows).
    #[must_use]
    pub const fn d50() -> Self {
        Self {
            x: 0.96422,
            y: 1.0,
            z: 0.82521,
        }
    }

    /// DCI white point (digital cinema).
    #[must_use]
    pub const fn dci_white() -> Self {
        Self {
            x: 0.89426,
            y: 1.0,
            z: 0.95429,
        }
    }
}

impl Default for Xyz {
    fn default() -> Self {
        Self::d65()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xyz_creation() {
        let xyz = Xyz::new(0.5, 0.6, 0.7);
        assert_eq!(xyz.x, 0.5);
        assert_eq!(xyz.y, 0.6);
        assert_eq!(xyz.z, 0.7);
    }

    #[test]
    fn test_xyz_array_conversion() {
        let xyz = Xyz::from_array([0.5, 0.6, 0.7]);
        assert_eq!(xyz.as_array(), [0.5, 0.6, 0.7]);
    }

    #[test]
    fn test_xyy_conversion() {
        let xyz = Xyz::new(0.3, 0.4, 0.5);
        let (x, y, big_y) = xyz.to_xyy();

        let xyz2 = Xyz::from_xyy(x, y, big_y);

        assert!((xyz2.x - xyz.x).abs() < 1e-10);
        assert!((xyz2.y - xyz.y).abs() < 1e-10);
        assert!((xyz2.z - xyz.z).abs() < 1e-10);
    }

    #[test]
    fn test_white_points() {
        let d65 = Xyz::d65();
        assert!((d65.y - 1.0).abs() < 1e-10);

        let d50 = Xyz::d50();
        assert!((d50.y - 1.0).abs() < 1e-10);
    }
}

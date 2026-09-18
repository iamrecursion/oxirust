//! CIE L*u*v* (Luv) color space.

use super::Xyz;

/// CIE L*u*v* color (1976).
///
/// Luv is an alternative to Lab, also designed to be perceptually uniform.
/// It's better for additive color mixing and has advantages for certain applications.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Luv {
    /// Lightness [0, 100]
    pub l: f64,
    /// u* axis
    pub u: f64,
    /// v* axis
    pub v: f64,
}

impl Luv {
    /// Creates a new Luv color.
    #[must_use]
    pub const fn new(l: f64, u: f64, v: f64) -> Self {
        Self { l, u, v }
    }

    /// Converts from CIE XYZ to Luv.
    ///
    /// # Arguments
    ///
    /// * `xyz` - XYZ color
    /// * `white_point` - Reference white point in XYZ
    #[must_use]
    pub fn from_xyz(xyz: &Xyz, white_point: &Xyz) -> Self {
        let y_ratio = xyz.y / white_point.y;

        let l = if y_ratio > EPSILON {
            116.0 * y_ratio.cbrt() - 16.0
        } else {
            KAPPA * y_ratio
        };

        let u_p = u_prime(xyz.x, xyz.y, xyz.z);
        let v_p = v_prime(xyz.x, xyz.y, xyz.z);
        let u_prime_n = u_prime(white_point.x, white_point.y, white_point.z);
        let v_prime_n = v_prime(white_point.x, white_point.y, white_point.z);

        Self {
            l,
            u: 13.0 * l * (u_p - u_prime_n),
            v: 13.0 * l * (v_p - v_prime_n),
        }
    }

    /// Converts Luv to CIE XYZ.
    ///
    /// # Arguments
    ///
    /// * `white_point` - Reference white point in XYZ
    #[must_use]
    pub fn to_xyz(&self, white_point: &Xyz) -> Xyz {
        if self.l < 1e-10 {
            return Xyz::new(0.0, 0.0, 0.0);
        }

        let u_prime_n = u_prime(white_point.x, white_point.y, white_point.z);
        let v_prime_n = v_prime(white_point.x, white_point.y, white_point.z);

        let u_prime = self.u / (13.0 * self.l) + u_prime_n;
        let v_prime = self.v / (13.0 * self.l) + v_prime_n;

        let y = if self.l > KAPPA * EPSILON {
            ((self.l + 16.0) / 116.0).powi(3)
        } else {
            self.l / KAPPA
        } * white_point.y;

        let x = y * 9.0 * u_prime / (4.0 * v_prime);
        let z = y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime);

        Xyz { x, y, z }
    }

    /// Returns the chroma (colorfulness).
    #[must_use]
    pub fn chroma(&self) -> f64 {
        (self.u * self.u + self.v * self.v).sqrt()
    }

    /// Returns the hue angle in degrees [0, 360).
    #[must_use]
    pub fn hue(&self) -> f64 {
        let mut h = self.v.atan2(self.u).to_degrees();
        if h < 0.0 {
            h += 360.0;
        }
        h
    }
}

const KAPPA: f64 = 903.3;
const EPSILON: f64 = 0.008_856;

#[must_use]
fn u_prime(x: f64, y: f64, z: f64) -> f64 {
    let denom = x + 15.0 * y + 3.0 * z;
    if denom < 1e-10 {
        0.0
    } else {
        4.0 * x / denom
    }
}

#[must_use]
fn v_prime(x: f64, y: f64, z: f64) -> f64 {
    let denom = x + 15.0 * y + 3.0 * z;
    if denom < 1e-10 {
        0.0
    } else {
        9.0 * y / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luv_creation() {
        let luv = Luv::new(50.0, 10.0, -20.0);
        assert_eq!(luv.l, 50.0);
        assert_eq!(luv.u, 10.0);
        assert_eq!(luv.v, -20.0);
    }

    #[test]
    fn test_luv_xyz_roundtrip() {
        let xyz = Xyz::new(0.5, 0.6, 0.7);
        let white = Xyz::d65();

        let luv = Luv::from_xyz(&xyz, &white);
        let xyz2 = luv.to_xyz(&white);

        assert!((xyz2.x - xyz.x).abs() < 1e-6);
        assert!((xyz2.y - xyz.y).abs() < 1e-6);
        assert!((xyz2.z - xyz.z).abs() < 1e-6);
    }

    #[test]
    fn test_luv_white() {
        let white = Xyz::d65();
        let luv = Luv::from_xyz(&white, &white);

        // White should be L=100, u=0, v=0
        assert!((luv.l - 100.0).abs() < 1e-6);
        assert!(luv.u.abs() < 1e-6);
        assert!(luv.v.abs() < 1e-6);
    }

    #[test]
    fn test_luv_chroma() {
        let luv = Luv::new(50.0, 30.0, 40.0);
        let chroma = luv.chroma();
        assert!((chroma - 50.0).abs() < 1e-10);
    }
}

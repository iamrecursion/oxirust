//! Minimal 3-vector / 3×3 matrix utilities for reference-frame work.
//!
//! # Rotation sign convention
//!
//! This crate adopts the convention of the IERS Conventions (2010), IERS
//! Technical Note 36, §5.4: "`R1`, `R2` and `R3` denote rotation matrices
//! with positive angle about the axes 1, 2 and 3 of the coordinate frame".
//! That is, `R_i(theta)` rotates the *coordinate frame* by `+theta`
//! (right-handed) about axis `i`; the components of a fixed vector
//! transform as `v' = R v`. Explicitly:
//!
//! ```text
//!          | 1    0        0     |          | cos_t  0  -sin_t |
//! R1(t) =  | 0   cos_t   sin_t   |  R2(t) = |  0     1    0    |
//!          | 0  -sin_t   cos_t   |          | sin_t  0   cos_t |
//!
//!          |  cos_t  sin_t  0 |
//! R3(t) =  | -sin_t  cos_t  0 |
//!          |   0      0     1 |
//! ```
//!
//! With this convention, e.g., the polar-motion matrix of TN36 eq. (5.3)
//! is `W(t) = R3(-s') R2(x_p) R1(y_p)` and the classical nutation matrix
//! is `N = R1(-(eps_A + deps)) R3(-dpsi) R1(eps_A)` (TN36 §5.4.5,
//! fig. 5.1).

use libm::{atan2, cos, sin, sqrt};

/// A 3-vector of `f64` components.
pub type Vec3 = [f64; 3];

/// A 3×3 matrix, row-major: `Mat3(rows)`, `rows[i][j]` = row `i`, col `j`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3(pub [[f64; 3]; 3]);

impl Mat3 {
    /// The identity matrix.
    pub const IDENTITY: Self = Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    /// Returns the identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    /// Matrix product `self * rhs` (apply `rhs` first, then `self`).
    #[must_use]
    pub fn mul(&self, rhs: &Self) -> Self {
        let a = &self.0;
        let b = &rhs.0;
        let mut out = [[0.0_f64; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        Self(out)
    }

    /// The transpose. For the orthonormal rotation matrices produced by
    /// [`r1`]/[`r2`]/[`r3`] and their products, this is the inverse.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let m = &self.0;
        Self([
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ])
    }

    /// Applies the matrix to a column vector: `self * v`.
    #[must_use]
    pub fn apply(&self, v: Vec3) -> Vec3 {
        let m = &self.0;
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    /// The determinant (exactly `1` for proper rotations, up to rounding).
    #[must_use]
    pub fn det(&self) -> f64 {
        let m = &self.0;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}

/// Frame rotation by `angle_rad` about axis 1 (x). See the module docs for
/// the sign convention (IERS TN36 §5.4) and the explicit layout.
#[must_use]
pub fn r1(angle_rad: f64) -> Mat3 {
    let (s, c) = (sin(angle_rad), cos(angle_rad));
    Mat3([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
}

/// Frame rotation by `angle_rad` about axis 2 (y). See the module docs for
/// the sign convention (IERS TN36 §5.4) and the explicit layout.
#[must_use]
pub fn r2(angle_rad: f64) -> Mat3 {
    let (s, c) = (sin(angle_rad), cos(angle_rad));
    Mat3([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
}

/// Frame rotation by `angle_rad` about axis 3 (z). See the module docs for
/// the sign convention (IERS TN36 §5.4) and the explicit layout.
#[must_use]
pub fn r3(angle_rad: f64) -> Mat3 {
    let (s, c) = (sin(angle_rad), cos(angle_rad));
    Mat3([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
}

/// Dot product `a · b`.
#[must_use]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm `|v|`.
#[must_use]
pub fn norm(v: Vec3) -> f64 {
    sqrt(dot(v, v))
}

/// Component-wise sum `a + b`.
#[must_use]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component-wise difference `a - b`.
#[must_use]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scalar multiple `k * v`.
#[must_use]
pub fn scale(v: Vec3, k: f64) -> Vec3 {
    [k * v[0], k * v[1], k * v[2]]
}

/// Cross product `a × b` (right-handed).
#[must_use]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Converts spherical coordinates (longitude, latitude, radius) to a
/// Cartesian vector:
/// `x = r cos(lat) cos(lon)`, `y = r cos(lat) sin(lon)`, `z = r sin(lat)`.
#[must_use]
pub fn spherical_to_cartesian(lon_rad: f64, lat_rad: f64, radius: f64) -> Vec3 {
    let (sl, cl) = (sin(lon_rad), cos(lon_rad));
    let (sb, cb) = (sin(lat_rad), cos(lat_rad));
    [radius * cb * cl, radius * cb * sl, radius * sb]
}

/// Converts a Cartesian vector to spherical `(lon_rad, lat_rad, radius)`
/// with `lon` in `[-pi, +pi]` and `lat` in `[-pi/2, +pi/2]`.
///
/// Degenerate inputs follow `atan2` conventions: a zero vector yields
/// `(0.0, 0.0, 0.0)`; a vector along ±z yields longitude `0.0`.
#[must_use]
pub fn cartesian_to_spherical(v: Vec3) -> (f64, f64, f64) {
    let rho = sqrt(v[0] * v[0] + v[1] * v[1]);
    let lon = atan2(v[1], v[0]);
    let lat = atan2(v[2], rho);
    let radius = sqrt(rho * rho + v[2] * v[2]);
    (lon, lat, radius)
}

//! Vector math types: [`Vec2`], [`Vec3`], [`Vec4`].
//!
//! Provides 2-D, 3-D and 4-D floating-point vectors with common operations
//! used in SIMD-accelerated media pipelines: dot product, cross product and
//! normalisation.

#![allow(dead_code)]

/// A 2-D floating-point vector.
///
/// # Examples
///
/// ```
/// use oximedia_simd::vector_math::Vec2;
/// let v = Vec2::new(3.0, 4.0);
/// assert!((v.length() - 5.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    /// Horizontal component.
    pub x: f32,
    /// Vertical component.
    pub y: f32,
}

impl Vec2 {
    /// Creates a new [`Vec2`] from components.
    #[inline]
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Returns the additive identity (0, 0).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Returns the dot product of `self` and `other`.
    ///
    /// `dot(a, b) = a.x*b.x + a.y*b.y`
    #[inline]
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Returns the squared Euclidean length of this vector.
    #[inline]
    #[must_use]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    /// Returns the Euclidean length of this vector.
    #[inline]
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    /// Returns a unit-length version of this vector.
    ///
    /// Returns [`Vec2::zero()`] if the vector has zero length.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < f32::EPSILON {
            return Self::zero();
        }
        Self::new(self.x / len, self.y / len)
    }

    /// Component-wise addition.
    #[inline]
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    /// Scales the vector by a scalar factor.
    #[inline]
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }
}

// ---------------------------------------------------------------------------

/// A 3-D floating-point vector.
///
/// # Examples
///
/// ```
/// use oximedia_simd::vector_math::Vec3;
/// let v = Vec3::new(1.0, 0.0, 0.0);
/// assert_eq!(v.normalize(), v);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Creates a new [`Vec3`] from components.
    #[inline]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Returns the additive identity (0, 0, 0).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Returns the dot product of `self` and `other`.
    #[inline]
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Returns the cross product `self × other`.
    ///
    /// The result is perpendicular to both input vectors.
    #[inline]
    #[must_use]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Returns the squared Euclidean length.
    #[inline]
    #[must_use]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    /// Returns the Euclidean length.
    #[inline]
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    /// Returns a unit-length version of this vector.
    ///
    /// Returns [`Vec3::zero()`] if the vector has zero length.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < f32::EPSILON {
            return Self::zero();
        }
        Self::new(self.x / len, self.y / len, self.z / len)
    }

    /// Component-wise addition.
    #[inline]
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Scales the vector by a scalar factor.
    #[inline]
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

// ---------------------------------------------------------------------------

/// A 4-D floating-point vector (also used for RGBA or homogeneous coordinates).
///
/// # Examples
///
/// ```
/// use oximedia_simd::vector_math::Vec4;
/// let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
/// assert!((v.length() - 1.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
    /// W component (homogeneous / alpha).
    pub w: f32,
}

impl Vec4 {
    /// Creates a new [`Vec4`] from components.
    #[inline]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Returns the additive identity (0, 0, 0, 0).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Returns the dot product of `self` and `other`.
    #[inline]
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Returns the squared Euclidean length.
    #[inline]
    #[must_use]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    /// Returns the Euclidean length.
    #[inline]
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    /// Returns a unit-length version of this vector.
    ///
    /// Returns [`Vec4::zero()`] if the vector has zero length.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < f32::EPSILON {
            return Self::zero();
        }
        Self::new(self.x / len, self.y / len, self.z / len, self.w / len)
    }

    /// Component-wise addition.
    #[inline]
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    }

    /// Scales all components by a scalar factor.
    #[inline]
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s, self.w * s)
    }

    /// Returns the XYZ components as a [`Vec3`], discarding `w`.
    #[inline]
    #[must_use]
    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Vec2 tests ---

    #[test]
    fn vec2_zero() {
        let z = Vec2::zero();
        assert_eq!(z.x, 0.0);
        assert_eq!(z.y, 0.0);
    }

    #[test]
    fn vec2_dot() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert_eq!(a.dot(b), 0.0);
        let c = Vec2::new(2.0, 3.0);
        let d = Vec2::new(4.0, 5.0);
        assert!((c.dot(d) - 23.0).abs() < 1e-6);
    }

    #[test]
    fn vec2_length() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn vec2_normalize() {
        let v = Vec2::new(3.0, 4.0).normalize();
        assert!((v.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn vec2_normalize_zero_returns_zero() {
        let v = Vec2::zero().normalize();
        assert_eq!(v, Vec2::zero());
    }

    #[test]
    fn vec2_add() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        let c = a.add(b);
        assert_eq!(c.x, 4.0);
        assert_eq!(c.y, 6.0);
    }

    #[test]
    fn vec2_scale() {
        let v = Vec2::new(1.0, 2.0).scale(3.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 6.0);
    }

    // --- Vec3 tests ---

    #[test]
    fn vec3_cross_unit_axes() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!((z.x).abs() < 1e-6);
        assert!((z.y).abs() < 1e-6);
        assert!((z.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec3_cross_anticommutative() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let ab = a.cross(b);
        let ba = b.cross(a);
        assert!((ab.x + ba.x).abs() < 1e-5);
        assert!((ab.y + ba.y).abs() < 1e-5);
        assert!((ab.z + ba.z).abs() < 1e-5);
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(2.0, 0.0, 0.0).normalize();
        assert!((v.x - 1.0).abs() < 1e-6);
        assert!((v.y).abs() < 1e-6);
        assert!((v.z).abs() < 1e-6);
    }

    #[test]
    fn vec3_normalize_zero_returns_zero() {
        let v = Vec3::zero().normalize();
        assert_eq!(v, Vec3::zero());
    }

    #[test]
    fn vec3_dot_orthogonal() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 1.0);
        assert_eq!(a.dot(b), 0.0);
    }

    // --- Vec4 tests ---

    #[test]
    fn vec4_length_unit() {
        let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
        assert!((v.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec4_normalize() {
        let v = Vec4::new(2.0, 0.0, 0.0, 0.0).normalize();
        assert!((v.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec4_normalize_zero_returns_zero() {
        let v = Vec4::zero().normalize();
        assert_eq!(v, Vec4::zero());
    }

    #[test]
    fn vec4_xyz_extracts_components() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let xyz = v.xyz();
        assert_eq!(xyz.x, 1.0);
        assert_eq!(xyz.y, 2.0);
        assert_eq!(xyz.z, 3.0);
    }

    #[test]
    fn vec4_add_and_scale() {
        let a = Vec4::new(1.0, 1.0, 1.0, 1.0);
        let b = a.add(a).scale(0.5);
        assert_eq!(b.x, 1.0);
        assert_eq!(b.w, 1.0);
    }
}

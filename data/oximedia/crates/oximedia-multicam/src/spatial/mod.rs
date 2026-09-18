//! Spatial alignment for multi-camera production.

pub mod align;
pub mod stitch;

pub use align::SpatialAligner;
pub use stitch::ViewStitcher;

/// 2D transformation matrix
#[derive(Debug, Clone, Copy)]
pub struct Transform2D {
    /// Translation X
    pub tx: f32,
    /// Translation Y
    pub ty: f32,
    /// Scale X
    pub sx: f32,
    /// Scale Y
    pub sy: f32,
    /// Rotation (radians)
    pub rotation: f32,
}

impl Transform2D {
    /// Create identity transformation
    #[must_use]
    pub fn identity() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            sx: 1.0,
            sy: 1.0,
            rotation: 0.0,
        }
    }

    /// Apply transformation to a point
    #[must_use]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        // Apply scale
        let mut px = x * self.sx;
        let mut py = y * self.sy;

        // Apply rotation
        if self.rotation != 0.0 {
            let cos = self.rotation.cos();
            let sin = self.rotation.sin();
            let rx = px * cos - py * sin;
            let ry = px * sin + py * cos;
            px = rx;
            py = ry;
        }

        // Apply translation
        (px + self.tx, py + self.ty)
    }

    /// Compose with another transformation
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        // Simplified composition (not complete affine transform)
        Self {
            tx: self.tx + other.tx,
            ty: self.ty + other.ty,
            sx: self.sx * other.sx,
            sy: self.sy * other.sy,
            rotation: self.rotation + other.rotation,
        }
    }

    /// Invert transformation
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            tx: -self.tx,
            ty: -self.ty,
            sx: 1.0 / self.sx,
            sy: 1.0 / self.sy,
            rotation: -self.rotation,
        }
    }
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let transform = Transform2D::identity();
        let (x, y) = transform.apply(10.0, 20.0);
        assert_eq!(x, 10.0);
        assert_eq!(y, 20.0);
    }

    #[test]
    fn test_translation() {
        let transform = Transform2D {
            tx: 5.0,
            ty: 10.0,
            ..Transform2D::identity()
        };
        let (x, y) = transform.apply(10.0, 20.0);
        assert_eq!(x, 15.0);
        assert_eq!(y, 30.0);
    }

    #[test]
    fn test_scaling() {
        let transform = Transform2D {
            sx: 2.0,
            sy: 0.5,
            ..Transform2D::identity()
        };
        let (x, y) = transform.apply(10.0, 20.0);
        assert_eq!(x, 20.0);
        assert_eq!(y, 10.0);
    }

    #[test]
    fn test_inverse() {
        let transform = Transform2D {
            tx: 5.0,
            ty: 10.0,
            sx: 2.0,
            sy: 2.0,
            rotation: 0.0,
        };
        let inverse = transform.inverse();
        assert_eq!(inverse.tx, -5.0);
        assert_eq!(inverse.ty, -10.0);
        assert_eq!(inverse.sx, 0.5);
    }
}

//! Perspective warp correction for video stabilization.
//!
//! This module implements perspective warp transforms that correct keystone
//! distortion and perspective shifts introduced by camera tilt or pan.
//! It operates on a 3x3 homography matrix and provides evaluation, inversion,
//! and chain-composition of perspective transforms.

#![allow(dead_code)]

use std::fmt;

// ---------------------------------------------------------------------------
// Point2D
// ---------------------------------------------------------------------------

/// A 2-D point in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// Horizontal coordinate (pixels).
    pub x: f64,
    /// Vertical coordinate (pixels).
    pub y: f64,
}

impl Point2D {
    /// Create a new point.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

// ---------------------------------------------------------------------------
// PerspectiveMatrix
// ---------------------------------------------------------------------------

/// A 3x3 homography (perspective) matrix stored in row-major order.
///
/// The matrix maps homogeneous source coordinates `[x, y, 1]` to destination
/// coordinates via `dst = H * src`, followed by dehomogenisation.
#[derive(Clone, Copy, PartialEq)]
pub struct PerspectiveMatrix {
    /// Row-major 3x3 entries.
    pub m: [[f64; 3]; 3],
}

impl fmt::Debug for PerspectiveMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PerspectiveMatrix([{:.6},{:.6},{:.6}],[{:.6},{:.6},{:.6}],[{:.6},{:.6},{:.6}])",
            self.m[0][0],
            self.m[0][1],
            self.m[0][2],
            self.m[1][0],
            self.m[1][1],
            self.m[1][2],
            self.m[2][0],
            self.m[2][1],
            self.m[2][2],
        )
    }
}

impl PerspectiveMatrix {
    /// Identity perspective matrix (no warp).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Create from a flat row-major slice of 9 elements.
    ///
    /// Returns `None` when the slice length is wrong.
    #[must_use]
    pub fn from_flat(data: &[f64]) -> Option<Self> {
        if data.len() != 9 {
            return None;
        }
        Some(Self {
            m: [
                [data[0], data[1], data[2]],
                [data[3], data[4], data[5]],
                [data[6], data[7], data[8]],
            ],
        })
    }

    /// Create a pure translation matrix.
    #[must_use]
    pub const fn translation(tx: f64, ty: f64) -> Self {
        Self {
            m: [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        }
    }

    /// Create a rotation matrix (angle in radians, around image centre).
    #[must_use]
    pub fn rotation(angle_rad: f64) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Self {
            m: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Determinant of the 3x3 matrix.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn determinant(&self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Check whether the matrix is invertible.
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() > 1e-12
    }

    /// Compute the inverse matrix, if it exists.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() <= 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let m = &self.m;
        Some(Self {
            m: [
                [
                    (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                    (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                    (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
                ],
                [
                    (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                    (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                    (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
                ],
                [
                    (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                    (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                    (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
                ],
            ],
        })
    }

    /// Apply this perspective transform to a 2-D point.
    #[must_use]
    pub fn transform_point(&self, p: &Point2D) -> Point2D {
        let m = &self.m;
        let w = m[2][0] * p.x + m[2][1] * p.y + m[2][2];
        let inv_w = if w.abs() > 1e-15 { 1.0 / w } else { 1.0 };
        Point2D {
            x: (m[0][0] * p.x + m[0][1] * p.y + m[0][2]) * inv_w,
            y: (m[1][0] * p.x + m[1][1] * p.y + m[1][2]) * inv_w,
        }
    }

    /// Compose two perspective matrices: `self * other`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let a = &self.m;
        let b = &other.m;
        let mut out = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        Self { m: out }
    }

    /// Normalise the matrix so that `m[2][2] == 1.0`.
    #[must_use]
    pub fn normalised(&self) -> Option<Self> {
        let s = self.m[2][2];
        if s.abs() < 1e-15 {
            return None;
        }
        let inv_s = 1.0 / s;
        let mut out = self.m;
        for row in &mut out {
            for v in row.iter_mut() {
                *v *= inv_s;
            }
        }
        Some(Self { m: out })
    }

    /// Root mean squared reprojection error across a set of correspondences.
    #[must_use]
    pub fn reprojection_rmse(&self, src: &[Point2D], dst: &[Point2D]) -> f64 {
        if src.len() != dst.len() || src.is_empty() {
            return f64::INFINITY;
        }
        let sum_sq: f64 = src
            .iter()
            .zip(dst.iter())
            .map(|(s, d)| {
                let mapped = self.transform_point(s);
                let dx = mapped.x - d.x;
                let dy = mapped.y - d.y;
                dx * dx + dy * dy
            })
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let n = src.len() as f64;
        (sum_sq / n).sqrt()
    }

    /// Maximum single-point reprojection error across correspondences.
    #[must_use]
    pub fn reprojection_max_error(&self, src: &[Point2D], dst: &[Point2D]) -> f64 {
        if src.len() != dst.len() || src.is_empty() {
            return f64::INFINITY;
        }
        src.iter()
            .zip(dst.iter())
            .map(|(s, d)| {
                let mapped = self.transform_point(s);
                mapped.distance_to(d)
            })
            .fold(0.0_f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// PerspectiveWarpConfig
// ---------------------------------------------------------------------------

/// Configuration for perspective warp correction.
#[derive(Debug, Clone)]
pub struct PerspectiveWarpConfig {
    /// Maximum allowed perspective distortion per axis (normalised 0-1).
    pub max_distortion: f64,
    /// Smoothing kernel radius (frames) for temporal filtering of homographies.
    pub smooth_radius: usize,
    /// Whether to enable keystone correction.
    pub enable_keystone: bool,
}

impl Default for PerspectiveWarpConfig {
    fn default() -> Self {
        Self {
            max_distortion: 0.05,
            smooth_radius: 10,
            enable_keystone: true,
        }
    }
}

// ---------------------------------------------------------------------------
// PerspectiveWarpCorrector
// ---------------------------------------------------------------------------

/// Corrects perspective warp over a sequence of homographies.
#[derive(Debug)]
pub struct PerspectiveWarpCorrector {
    config: PerspectiveWarpConfig,
    history: Vec<PerspectiveMatrix>,
}

impl PerspectiveWarpCorrector {
    /// Create a new corrector with the given configuration.
    #[must_use]
    pub fn new(config: PerspectiveWarpConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Push a new frame homography and return a smoothed correction matrix.
    pub fn push(&mut self, h: PerspectiveMatrix) -> PerspectiveMatrix {
        self.history.push(h);
        self.smoothed_latest()
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Number of frames processed so far.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.history.len()
    }

    /// Return the temporally smoothed homography for the most recent frame.
    fn smoothed_latest(&self) -> PerspectiveMatrix {
        let n = self.history.len();
        if n == 0 {
            return PerspectiveMatrix::identity();
        }
        let radius = self.config.smooth_radius.min(n - 1);
        let start = n.saturating_sub(radius + 1);
        let count = n - start;
        if count == 0 {
            return PerspectiveMatrix::identity();
        }
        let mut avg = [[0.0_f64; 3]; 3];
        for idx in start..n {
            for i in 0..3 {
                for j in 0..3 {
                    avg[i][j] += self.history[idx].m[i][j];
                }
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let inv_count = 1.0 / count as f64;
        for row in &mut avg {
            for v in row.iter_mut() {
                *v *= inv_count;
            }
        }
        let smooth = PerspectiveMatrix { m: avg };
        smooth.normalised().unwrap_or(PerspectiveMatrix::identity())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let id = PerspectiveMatrix::identity();
        let p = Point2D::new(100.0, 200.0);
        let q = id.transform_point(&p);
        assert!((q.x - p.x).abs() < 1e-10);
        assert!((q.y - p.y).abs() < 1e-10);
    }

    #[test]
    fn test_translation_transform() {
        let t = PerspectiveMatrix::translation(10.0, -5.0);
        let p = Point2D::new(50.0, 80.0);
        let q = t.transform_point(&p);
        assert!((q.x - 60.0).abs() < 1e-10);
        assert!((q.y - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_identity() {
        let id = PerspectiveMatrix::identity();
        assert!((id.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_inverse_identity() {
        let id = PerspectiveMatrix::identity();
        let inv = id.inverse().expect("should succeed in test");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv.m[i][j] - expected).abs() < 1e-10,
                    "inv[{i}][{j}] = {} expected {expected}",
                    inv.m[i][j]
                );
            }
        }
    }

    #[test]
    fn test_inverse_translation() {
        let t = PerspectiveMatrix::translation(7.0, -3.0);
        let inv = t.inverse().expect("should succeed in test");
        let p = Point2D::new(50.0, 80.0);
        let q = t.transform_point(&p);
        let r = inv.transform_point(&q);
        assert!((r.x - p.x).abs() < 1e-10);
        assert!((r.y - p.y).abs() < 1e-10);
    }

    #[test]
    fn test_compose_translations() {
        let t1 = PerspectiveMatrix::translation(3.0, 4.0);
        let t2 = PerspectiveMatrix::translation(10.0, -2.0);
        let combined = t1.compose(&t2);
        let p = Point2D::new(0.0, 0.0);
        let q = combined.transform_point(&p);
        assert!((q.x - 13.0).abs() < 1e-10);
        assert!((q.y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_90_degrees() {
        let r = PerspectiveMatrix::rotation(std::f64::consts::FRAC_PI_2);
        let p = Point2D::new(1.0, 0.0);
        let q = r.transform_point(&p);
        assert!((q.x - 0.0).abs() < 1e-10);
        assert!((q.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_flat_valid() {
        let flat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let m = PerspectiveMatrix::from_flat(&flat).expect("should succeed in test");
        assert!((m.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_from_flat_wrong_length() {
        let flat = [1.0, 2.0, 3.0];
        assert!(PerspectiveMatrix::from_flat(&flat).is_none());
    }

    #[test]
    fn test_reprojection_rmse_identity() {
        let id = PerspectiveMatrix::identity();
        let src = vec![Point2D::new(10.0, 20.0), Point2D::new(30.0, 40.0)];
        let dst = src.clone();
        let rmse = id.reprojection_rmse(&src, &dst);
        assert!(rmse < 1e-10);
    }

    #[test]
    fn test_normalised() {
        let m = PerspectiveMatrix {
            m: [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        };
        let n = m.normalised().expect("should succeed in test");
        assert!((n.m[2][2] - 1.0).abs() < 1e-12);
        assert!((n.m[0][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_corrector_push_and_count() {
        let cfg = PerspectiveWarpConfig::default();
        let mut c = PerspectiveWarpCorrector::new(cfg);
        assert_eq!(c.frame_count(), 0);
        let _ = c.push(PerspectiveMatrix::identity());
        assert_eq!(c.frame_count(), 1);
        let _ = c.push(PerspectiveMatrix::translation(1.0, 2.0));
        assert_eq!(c.frame_count(), 2);
    }

    #[test]
    fn test_corrector_reset() {
        let cfg = PerspectiveWarpConfig::default();
        let mut c = PerspectiveWarpCorrector::new(cfg);
        let _ = c.push(PerspectiveMatrix::identity());
        c.reset();
        assert_eq!(c.frame_count(), 0);
    }

    #[test]
    fn test_point_distance() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_singular_matrix_not_invertible() {
        let m = PerspectiveMatrix {
            m: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        };
        assert!(!m.is_invertible());
        assert!(m.inverse().is_none());
    }
}

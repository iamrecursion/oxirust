//! Mesh-based deformation effects for video frames.
//!
//! Provides a control-point mesh that can be distorted to create warp,
//! bulge, pinch, and twist effects on video content.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// The type of deformation applied to the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeformMode {
    /// Warp: free-form displacement of control points.
    Warp,
    /// Bulge: radial expansion from a centre point.
    Bulge,
    /// Pinch: radial contraction toward a centre point.
    Pinch,
    /// Twist: angular rotation proportional to distance from centre.
    Twist,
    /// Stretch: directional elongation along one axis.
    Stretch,
}

impl DeformMode {
    /// Human-readable label for the mode.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Warp => "Warp",
            Self::Bulge => "Bulge",
            Self::Pinch => "Pinch",
            Self::Twist => "Twist",
            Self::Stretch => "Stretch",
        }
    }
}

/// A single control point in the deformation mesh.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeshPoint {
    /// Original X position in normalised coordinates [0, 1].
    pub orig_x: f64,
    /// Original Y position in normalised coordinates [0, 1].
    pub orig_y: f64,
    /// Displaced X position after deformation.
    pub disp_x: f64,
    /// Displaced Y position after deformation.
    pub disp_y: f64,
    /// Weight / influence radius of this point (0.0 = no influence, 1.0 = full).
    pub weight: f64,
}

impl MeshPoint {
    /// Create a new mesh point at a given position with zero displacement.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            orig_x: x,
            orig_y: y,
            disp_x: x,
            disp_y: y,
            weight: 1.0,
        }
    }

    /// Euclidean distance between original and displaced positions.
    #[must_use]
    pub fn displacement(&self) -> f64 {
        let dx = self.disp_x - self.orig_x;
        let dy = self.disp_y - self.orig_y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Reset displaced position to the original.
    pub fn reset(&mut self) {
        self.disp_x = self.orig_x;
        self.disp_y = self.orig_y;
    }

    /// Set displacement as an offset from the original position.
    pub fn set_offset(&mut self, dx: f64, dy: f64) {
        self.disp_x = self.orig_x + dx;
        self.disp_y = self.orig_y + dy;
    }

    /// Linearly interpolate toward another point by factor `t` in [0, 1].
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            orig_x: self.orig_x + (other.orig_x - self.orig_x) * t,
            orig_y: self.orig_y + (other.orig_y - self.orig_y) * t,
            disp_x: self.disp_x + (other.disp_x - self.disp_x) * t,
            disp_y: self.disp_y + (other.disp_y - self.disp_y) * t,
            weight: self.weight + (other.weight - self.weight) * t,
        }
    }
}

/// A 2-D grid of [`MeshPoint`]s used to deform a frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeformMesh {
    /// Number of columns (horizontal subdivisions + 1).
    pub cols: usize,
    /// Number of rows (vertical subdivisions + 1).
    pub rows: usize,
    /// Flat row-major storage of mesh points (`rows * cols` entries).
    pub points: Vec<MeshPoint>,
    /// Active deformation mode.
    pub mode: DeformMode,
    /// Global intensity multiplier [0, 1].
    pub intensity: f64,
}

impl DeformMesh {
    /// Create a uniform mesh grid with the given subdivisions.
    ///
    /// `cols` and `rows` must be >= 2 (otherwise clamped).
    #[must_use]
    pub fn new(cols: usize, rows: usize, mode: DeformMode) -> Self {
        let cols = cols.max(2);
        let rows = rows.max(2);
        let mut points = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            #[allow(clippy::cast_precision_loss)]
            let y = r as f64 / (rows - 1) as f64;
            for c in 0..cols {
                #[allow(clippy::cast_precision_loss)]
                let x = c as f64 / (cols - 1) as f64;
                points.push(MeshPoint::new(x, y));
            }
        }
        Self {
            cols,
            rows,
            points,
            mode,
            intensity: 1.0,
        }
    }

    /// Total number of control points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Get a reference to the point at (col, row).
    #[must_use]
    pub fn get(&self, col: usize, row: usize) -> Option<&MeshPoint> {
        if col < self.cols && row < self.rows {
            Some(&self.points[row * self.cols + col])
        } else {
            None
        }
    }

    /// Get a mutable reference to the point at (col, row).
    pub fn get_mut(&mut self, col: usize, row: usize) -> Option<&mut MeshPoint> {
        if col < self.cols && row < self.rows {
            Some(&mut self.points[row * self.cols + col])
        } else {
            None
        }
    }

    /// Reset all control points to their original positions.
    pub fn reset_all(&mut self) {
        for p in &mut self.points {
            p.reset();
        }
    }

    /// Apply the configured deformation centred at `(cx, cy)` with the given `radius`.
    ///
    /// Returns the number of points that were actually displaced.
    pub fn apply_deform(&mut self, cx: f64, cy: f64, radius: f64) -> usize {
        let radius = radius.max(0.001);
        let mut displaced = 0usize;
        for p in &mut self.points {
            let dx = p.orig_x - cx;
            let dy = p.orig_y - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist >= radius {
                continue;
            }
            let factor = 1.0 - (dist / radius);
            let factor = factor * self.intensity * p.weight;
            match self.mode {
                DeformMode::Warp => {
                    p.disp_x = p.orig_x + dx * factor * 0.5;
                    p.disp_y = p.orig_y + dy * factor * 0.5;
                }
                DeformMode::Bulge => {
                    let bulge = 1.0 + factor;
                    p.disp_x = cx + dx * bulge;
                    p.disp_y = cy + dy * bulge;
                }
                DeformMode::Pinch => {
                    let pinch = 1.0 - factor * 0.8;
                    p.disp_x = cx + dx * pinch;
                    p.disp_y = cy + dy * pinch;
                }
                DeformMode::Twist => {
                    let angle = factor * std::f64::consts::PI * 0.5;
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    p.disp_x = cx + dx * cos_a - dy * sin_a;
                    p.disp_y = cy + dx * sin_a + dy * cos_a;
                }
                DeformMode::Stretch => {
                    p.disp_x = p.orig_x + dx * factor;
                    p.disp_y = p.orig_y;
                }
            }
            displaced += 1;
        }
        displaced
    }

    /// Compute the maximum displacement across all points.
    #[must_use]
    pub fn max_displacement(&self) -> f64 {
        self.points
            .iter()
            .map(MeshPoint::displacement)
            .fold(0.0_f64, f64::max)
    }

    /// Compute the average displacement across all points.
    #[must_use]
    pub fn avg_displacement(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.points.iter().map(MeshPoint::displacement).sum();
        sum / self.points.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deform_mode_labels() {
        assert_eq!(DeformMode::Warp.label(), "Warp");
        assert_eq!(DeformMode::Bulge.label(), "Bulge");
        assert_eq!(DeformMode::Pinch.label(), "Pinch");
        assert_eq!(DeformMode::Twist.label(), "Twist");
        assert_eq!(DeformMode::Stretch.label(), "Stretch");
    }

    #[test]
    fn test_mesh_point_new_zero_displacement() {
        let p = MeshPoint::new(0.3, 0.7);
        assert!((p.displacement()).abs() < 1e-12);
    }

    #[test]
    fn test_mesh_point_set_offset() {
        let mut p = MeshPoint::new(0.5, 0.5);
        p.set_offset(0.1, -0.2);
        assert!((p.disp_x - 0.6).abs() < 1e-12);
        assert!((p.disp_y - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_mesh_point_reset() {
        let mut p = MeshPoint::new(0.5, 0.5);
        p.set_offset(0.1, 0.1);
        assert!(p.displacement() > 0.0);
        p.reset();
        assert!(p.displacement() < 1e-12);
    }

    #[test]
    fn test_mesh_point_displacement() {
        let mut p = MeshPoint::new(0.0, 0.0);
        p.disp_x = 3.0;
        p.disp_y = 4.0;
        assert!((p.displacement() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_mesh_point_lerp_at_zero() {
        let a = MeshPoint::new(0.0, 0.0);
        let b = MeshPoint::new(1.0, 1.0);
        let c = a.lerp(&b, 0.0);
        assert!((c.orig_x).abs() < 1e-12);
        assert!((c.orig_y).abs() < 1e-12);
    }

    #[test]
    fn test_mesh_point_lerp_at_one() {
        let a = MeshPoint::new(0.0, 0.0);
        let b = MeshPoint::new(1.0, 1.0);
        let c = a.lerp(&b, 1.0);
        assert!((c.orig_x - 1.0).abs() < 1e-12);
        assert!((c.orig_y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_deform_mesh_dimensions() {
        let m = DeformMesh::new(5, 4, DeformMode::Warp);
        assert_eq!(m.cols, 5);
        assert_eq!(m.rows, 4);
        assert_eq!(m.point_count(), 20);
    }

    #[test]
    fn test_deform_mesh_min_clamp() {
        let m = DeformMesh::new(1, 1, DeformMode::Bulge);
        assert_eq!(m.cols, 2);
        assert_eq!(m.rows, 2);
    }

    #[test]
    fn test_get_returns_correct_point() {
        let m = DeformMesh::new(3, 3, DeformMode::Warp);
        let p = m.get(2, 2).expect("should succeed in test");
        assert!((p.orig_x - 1.0).abs() < 1e-12);
        assert!((p.orig_y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let m = DeformMesh::new(3, 3, DeformMode::Warp);
        assert!(m.get(5, 5).is_none());
    }

    #[test]
    fn test_apply_deform_bulge_displaces_inner() {
        let mut m = DeformMesh::new(5, 5, DeformMode::Bulge);
        let count = m.apply_deform(0.5, 0.5, 0.6);
        assert!(count > 0);
        assert!(m.max_displacement() > 0.0);
    }

    #[test]
    fn test_apply_deform_pinch() {
        let mut m = DeformMesh::new(5, 5, DeformMode::Pinch);
        m.apply_deform(0.5, 0.5, 0.6);
        assert!(m.max_displacement() > 0.0);
    }

    #[test]
    fn test_apply_deform_twist() {
        let mut m = DeformMesh::new(5, 5, DeformMode::Twist);
        m.apply_deform(0.5, 0.5, 0.6);
        assert!(m.max_displacement() > 0.0);
    }

    #[test]
    fn test_apply_deform_stretch() {
        let mut m = DeformMesh::new(5, 5, DeformMode::Stretch);
        m.apply_deform(0.5, 0.5, 0.8);
        assert!(m.max_displacement() > 0.0);
    }

    #[test]
    fn test_reset_all_clears_displacement() {
        let mut m = DeformMesh::new(5, 5, DeformMode::Bulge);
        m.apply_deform(0.5, 0.5, 0.6);
        m.reset_all();
        assert!(m.max_displacement() < 1e-12);
    }

    #[test]
    fn test_avg_displacement_empty() {
        let m = DeformMesh {
            cols: 0,
            rows: 0,
            points: Vec::new(),
            mode: DeformMode::Warp,
            intensity: 1.0,
        };
        assert!((m.avg_displacement()).abs() < 1e-12);
    }
}

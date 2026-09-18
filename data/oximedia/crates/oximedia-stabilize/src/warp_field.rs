#![allow(dead_code)]
//! Warp field generation and application for video stabilization.
//!
//! Provides flexible mesh-based warp fields that map source pixel
//! coordinates to destination coordinates, supporting multiple warp modes.

/// Warp interpolation / distortion mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpMode {
    /// Simple translational shift.
    Translate,
    /// Affine (translation + rotation + scale + shear).
    Affine,
    /// Full perspective (homography).
    Perspective,
    /// Per-vertex mesh deformation.
    Mesh,
}

impl Default for WarpMode {
    fn default() -> Self {
        Self::Affine
    }
}

/// A 2-D warp field storing per-vertex displacement vectors.
///
/// The field is a grid of `(cols+1) x (rows+1)` vertices that defines how
/// each cell is deformed. For a frame of `width x height`, `cell_w = width / cols`
/// and `cell_h = height / rows`.
#[derive(Debug, Clone)]
pub struct WarpField {
    /// Horizontal displacement for each vertex.
    pub dx: Vec<f64>,
    /// Vertical displacement for each vertex.
    pub dy: Vec<f64>,
    /// Number of cell columns.
    pub cols: usize,
    /// Number of cell rows.
    pub rows: usize,
    /// Active warp mode.
    pub mode: WarpMode,
}

impl WarpField {
    /// Create an identity (zero-displacement) warp field.
    #[must_use]
    pub fn identity(cols: usize, rows: usize) -> Self {
        let n = (cols + 1) * (rows + 1);
        Self {
            dx: vec![0.0; n],
            dy: vec![0.0; n],
            cols,
            rows,
            mode: WarpMode::Mesh,
        }
    }

    /// Create a uniform translational warp field.
    #[must_use]
    pub fn from_translation(cols: usize, rows: usize, tx: f64, ty: f64) -> Self {
        let n = (cols + 1) * (rows + 1);
        Self {
            dx: vec![tx; n],
            dy: vec![ty; n],
            cols,
            rows,
            mode: WarpMode::Translate,
        }
    }

    /// Number of vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        (self.cols + 1) * (self.rows + 1)
    }

    /// Set displacement at vertex `(vx, vy)`.
    pub fn set_vertex(&mut self, vx: usize, vy: usize, dx: f64, dy: f64) {
        let vcols = self.cols + 1;
        if vx < vcols && vy <= self.rows {
            let idx = vy * vcols + vx;
            self.dx[idx] = dx;
            self.dy[idx] = dy;
        }
    }

    /// Get displacement at vertex `(vx, vy)`.
    #[must_use]
    pub fn get_vertex(&self, vx: usize, vy: usize) -> Option<(f64, f64)> {
        let vcols = self.cols + 1;
        if vx < vcols && vy <= self.rows {
            let idx = vy * vcols + vx;
            Some((self.dx[idx], self.dy[idx]))
        } else {
            None
        }
    }

    /// Bilinearly interpolate the displacement at a fractional grid position.
    ///
    /// `gx` and `gy` are in vertex-grid coordinates `[0, cols]` and `[0, rows]`.
    #[must_use]
    pub fn interpolate(&self, gx: f64, gy: f64) -> (f64, f64) {
        let gx = gx.clamp(0.0, self.cols as f64);
        let gy = gy.clamp(0.0, self.rows as f64);

        let ix = gx.floor() as usize;
        let iy = gy.floor() as usize;
        let fx = gx - ix as f64;
        let fy = gy - iy as f64;

        let ix1 = ix.min(self.cols);
        let iy1 = iy.min(self.rows);
        let ix2 = (ix + 1).min(self.cols);
        let iy2 = (iy + 1).min(self.rows);

        let vcols = self.cols + 1;

        let d00 = (self.dx[iy1 * vcols + ix1], self.dy[iy1 * vcols + ix1]);
        let d10 = (self.dx[iy1 * vcols + ix2], self.dy[iy1 * vcols + ix2]);
        let d01 = (self.dx[iy2 * vcols + ix1], self.dy[iy2 * vcols + ix1]);
        let d11 = (self.dx[iy2 * vcols + ix2], self.dy[iy2 * vcols + ix2]);

        let top_x = d00.0 * (1.0 - fx) + d10.0 * fx;
        let top_y = d00.1 * (1.0 - fx) + d10.1 * fx;
        let bot_x = d01.0 * (1.0 - fx) + d11.0 * fx;
        let bot_y = d01.1 * (1.0 - fx) + d11.1 * fx;

        let out_x = top_x * (1.0 - fy) + bot_x * fy;
        let out_y = top_y * (1.0 - fy) + bot_y * fy;

        (out_x, out_y)
    }

    /// Maximum displacement magnitude in the field.
    #[must_use]
    pub fn max_displacement(&self) -> f64 {
        self.dx
            .iter()
            .zip(self.dy.iter())
            .map(|(dx, dy)| (dx * dx + dy * dy).sqrt())
            .fold(0.0f64, f64::max)
    }

    /// Scale all displacements by a factor.
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.dx {
            *v *= factor;
        }
        for v in &mut self.dy {
            *v *= factor;
        }
    }
}

/// Applies a [`WarpField`] to a grayscale frame buffer.
#[derive(Debug)]
pub struct MeshWarper {
    /// Border value used for out-of-bound pixels.
    pub border_value: u8,
}

impl MeshWarper {
    /// Create a new warper.
    #[must_use]
    pub fn new(border_value: u8) -> Self {
        Self { border_value }
    }

    /// Apply the warp field to a grayscale image.
    ///
    /// `src` is row-major, `width x height`. Returns a new buffer of the
    /// same dimensions.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn apply(&self, src: &[u8], width: usize, height: usize, field: &WarpField) -> Vec<u8> {
        if src.len() != width * height || width == 0 || height == 0 {
            return Vec::new();
        }
        let cell_w = width as f64 / field.cols as f64;
        let cell_h = height as f64 / field.rows as f64;

        let mut dst = vec![self.border_value; width * height];

        for y in 0..height {
            for x in 0..width {
                let gx = x as f64 / cell_w;
                let gy = y as f64 / cell_h;
                let (ddx, ddy) = field.interpolate(gx, gy);

                let sx = x as f64 + ddx;
                let sy = y as f64 + ddy;

                if sx >= 0.0 && sx < width as f64 && sy >= 0.0 && sy < height as f64 {
                    // Nearest-neighbor for simplicity (clamp to valid range)
                    let ix = (sx.round() as usize).min(width - 1);
                    let iy = (sy.round() as usize).min(height - 1);
                    dst[y * width + x] = src[iy * width + ix];
                }
            }
        }
        dst
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warp_mode_default() {
        assert_eq!(WarpMode::default(), WarpMode::Affine);
    }

    #[test]
    fn test_identity_field() {
        let f = WarpField::identity(4, 3);
        assert_eq!(f.vertex_count(), 5 * 4);
        assert!((f.max_displacement()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_translation() {
        let f = WarpField::from_translation(2, 2, 5.0, -3.0);
        for i in 0..f.vertex_count() {
            assert!((f.dx[i] - 5.0).abs() < f64::EPSILON);
            assert!((f.dy[i] - (-3.0)).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_set_get_vertex() {
        let mut f = WarpField::identity(2, 2);
        f.set_vertex(1, 1, 7.0, -2.0);
        let (dx, dy) = f.get_vertex(1, 1).expect("should succeed in test");
        assert!((dx - 7.0).abs() < f64::EPSILON);
        assert!((dy - (-2.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_vertex_out_of_bounds() {
        let f = WarpField::identity(2, 2);
        assert!(f.get_vertex(10, 10).is_none());
    }

    #[test]
    fn test_interpolate_corner() {
        let f = WarpField::from_translation(2, 2, 3.0, 4.0);
        let (dx, dy) = f.interpolate(0.0, 0.0);
        assert!((dx - 3.0).abs() < 1e-9);
        assert!((dy - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_center() {
        let mut f = WarpField::identity(1, 1);
        // 4 vertices at corners: top-left=0, top-right=10, bot-left=0, bot-right=10
        f.set_vertex(0, 0, 0.0, 0.0);
        f.set_vertex(1, 0, 10.0, 0.0);
        f.set_vertex(0, 1, 0.0, 10.0);
        f.set_vertex(1, 1, 10.0, 10.0);
        let (dx, dy) = f.interpolate(0.5, 0.5);
        assert!((dx - 5.0).abs() < 1e-9);
        assert!((dy - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_displacement() {
        let mut f = WarpField::identity(2, 2);
        f.set_vertex(0, 0, 3.0, 4.0);
        assert!((f.max_displacement() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_scale() {
        let mut f = WarpField::from_translation(2, 2, 2.0, 4.0);
        f.scale(0.5);
        assert!((f.dx[0] - 1.0).abs() < f64::EPSILON);
        assert!((f.dy[0] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mesh_warper_identity() {
        let src: Vec<u8> = (0..64).collect();
        let f = WarpField::identity(4, 4);
        let warper = MeshWarper::new(0);
        let dst = warper.apply(&src, 8, 8, &f);
        assert_eq!(dst.len(), 64);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_mesh_warper_empty() {
        let warper = MeshWarper::new(0);
        let dst = warper.apply(&[], 0, 0, &WarpField::identity(1, 1));
        assert!(dst.is_empty());
    }

    #[test]
    fn test_mesh_warper_size_mismatch() {
        let warper = MeshWarper::new(0);
        let dst = warper.apply(&[0u8; 10], 8, 8, &WarpField::identity(1, 1));
        assert!(dst.is_empty());
    }

    #[test]
    fn test_warp_mode_equality() {
        assert_eq!(WarpMode::Mesh, WarpMode::Mesh);
        assert_ne!(WarpMode::Translate, WarpMode::Perspective);
    }

    #[test]
    fn test_interpolate_clamped() {
        let f = WarpField::from_translation(2, 2, 1.0, 1.0);
        // Out-of-range coordinates should be clamped
        let (dx, dy) = f.interpolate(-5.0, -5.0);
        assert!((dx - 1.0).abs() < 1e-9);
        assert!((dy - 1.0).abs() < 1e-9);
    }
}

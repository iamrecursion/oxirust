//! Point cloud → equirectangular projection.
//!
//! Converts a coloured 3-D point cloud into an equirectangular (latitude–
//! longitude) panoramic image by projecting each point onto the unit sphere
//! and writing its colour into the nearest output pixel.  When multiple
//! points map to the same pixel the closest one (smallest Euclidean radius)
//! wins (depth-weighted / closest-depth splatting).

#![allow(dead_code)]

use std::f32::consts::PI;

use crate::VrError;

// ─── Point3d ─────────────────────────────────────────────────────────────────

/// A single coloured point in 3-D Cartesian space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3d {
    /// X coordinate (right).
    pub x: f32,
    /// Y coordinate (up).
    pub y: f32,
    /// Z coordinate (forward / into the screen).
    pub z: f32,
    /// Red channel [0, 255].
    pub r: u8,
    /// Green channel [0, 255].
    pub g: u8,
    /// Blue channel [0, 255].
    pub b: u8,
}

impl Point3d {
    /// Construct a new coloured point.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32, r: u8, g: u8, b: u8) -> Self {
        Self { x, y, z, r, g, b }
    }

    /// Euclidean distance from the origin (radial depth).
    #[must_use]
    pub fn radius(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Returns `true` if the point lies at the origin (degenerate, cannot be
    /// projected onto the sphere).
    #[must_use]
    pub fn is_at_origin(&self) -> bool {
        self.radius() < f32::EPSILON
    }

    /// Normalise to a unit-sphere direction vector.
    ///
    /// Returns `None` for degenerate (zero-length) points.
    #[must_use]
    pub fn unit_direction(&self) -> Option<(f32, f32, f32)> {
        let r = self.radius();
        if r < f32::EPSILON {
            return None;
        }
        Some((self.x / r, self.y / r, self.z / r))
    }
}

// ─── PointCloud ──────────────────────────────────────────────────────────────

/// A collection of coloured 3-D points.
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    /// The points.
    pub points: Vec<Point3d>,
}

impl PointCloud {
    /// Create an empty point cloud.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a point cloud from an existing vector.
    #[must_use]
    pub fn from_points(points: Vec<Point3d>) -> Self {
        Self { points }
    }

    /// Add a single point.
    pub fn push(&mut self, point: Point3d) {
        self.points.push(point);
    }

    /// Number of points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the cloud has no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

// ─── PointCloudProjector ─────────────────────────────────────────────────────

/// Projects a [`PointCloud`] onto an equirectangular panorama.
///
/// Each point is mapped to its spherical longitude/latitude and then to a
/// pixel in the output image.  When multiple points compete for the same
/// pixel, the closest point to the viewer (smallest radius) wins.
pub struct PointCloudProjector {
    /// Background colour used for pixels that receive no point (R, G, B).
    pub background: (u8, u8, u8),
}

impl PointCloudProjector {
    /// Create a projector with a black background.
    #[must_use]
    pub fn new() -> Self {
        Self {
            background: (0, 0, 0),
        }
    }

    /// Create a projector with a custom background colour.
    #[must_use]
    pub fn with_background(r: u8, g: u8, b: u8) -> Self {
        Self {
            background: (r, g, b),
        }
    }

    /// Project `cloud` into an equirectangular image of `width × height` pixels.
    ///
    /// Returns an interleaved RGB byte buffer of length `width * height * 3`.
    ///
    /// # Errors
    ///
    /// - [`VrError::InvalidDimensions`] if `width` or `height` is zero.
    pub fn to_equirectangular(
        &self,
        cloud: &PointCloud,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, VrError> {
        if width == 0 || height == 0 {
            return Err(VrError::InvalidDimensions(
                "width and height must be > 0".into(),
            ));
        }

        let w = width as usize;
        let h = height as usize;

        // depth buffer: stores squared radius at each pixel (f32::MAX = empty).
        let mut depth_buf = vec![f32::MAX; w * h];

        // colour buffer: interleaved RGB.
        let (bg_r, bg_g, bg_b) = self.background;
        let mut rgb_buf: Vec<u8> = Vec::with_capacity(w * h * 3);
        for _ in 0..(w * h) {
            rgb_buf.push(bg_r);
            rgb_buf.push(bg_g);
            rgb_buf.push(bg_b);
        }

        for pt in &cloud.points {
            let (nx, ny, nz) = match pt.unit_direction() {
                Some(d) => d,
                None => continue, // skip origin points
            };

            // Convert Cartesian unit vector → (longitude, latitude).
            // longitude ∈ [-π, π], latitude ∈ [-π/2, π/2]
            // Convention: lon=0 when pointing along +Z (forward).
            let lon = nx.atan2(nz); // azimuth: 0 = +Z, π/2 = +X
            let lat = ny.asin().clamp(-PI / 2.0, PI / 2.0);

            // Map to UV [0, 1]
            let u = (lon + PI) / (2.0 * PI);
            let v = (PI / 2.0 - lat) / PI; // flip: top = north pole

            // Pixel coordinates
            let px = ((u * width as f32).floor() as usize).min(w - 1);
            let py = ((v * height as f32).floor() as usize).min(h - 1);
            let idx = py * w + px;

            // Depth test: keep closest point (smallest r²).
            let r_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            if r_sq < depth_buf[idx] {
                depth_buf[idx] = r_sq;
                rgb_buf[idx * 3] = pt.r;
                rgb_buf[idx * 3 + 1] = pt.g;
                rgb_buf[idx * 3 + 2] = pt.b;
            }
        }

        Ok(rgb_buf)
    }
}

impl Default for PointCloudProjector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Helper: project a single direction to equirect pixel coords ─────────────

/// Map a unit-sphere direction vector to (pixel_x, pixel_y) in an equirect
/// image of the given dimensions.
///
/// Returns `None` if the direction vector has zero length.
pub fn direction_to_equirect_pixel(
    dx: f32,
    dy: f32,
    dz: f32,
    width: u32,
    height: u32,
) -> Option<(u32, u32)> {
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    if r < f32::EPSILON {
        return None;
    }
    let (nx, ny, nz) = (dx / r, dy / r, dz / r);
    let lon = nx.atan2(nz); // lon=0 → +Z forward
    let lat = ny.asin().clamp(-PI / 2.0, PI / 2.0);
    let u = (lon + PI) / (2.0 * PI);
    let v = (PI / 2.0 - lat) / PI;
    let px = ((u * width as f32).floor() as u32).min(width - 1);
    let py = ((v * height as f32).floor() as u32).min(height - 1);
    Some((px, py))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Point3d ──────────────────────────────────────────────────────────────

    #[test]
    fn point3d_radius() {
        let p = Point3d::new(3.0, 4.0, 0.0, 255, 0, 0);
        assert!((p.radius() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn point3d_is_at_origin() {
        let p = Point3d::new(0.0, 0.0, 0.0, 0, 0, 0);
        assert!(p.is_at_origin());
    }

    #[test]
    fn point3d_not_at_origin() {
        let p = Point3d::new(1.0, 0.0, 0.0, 255, 255, 255);
        assert!(!p.is_at_origin());
    }

    #[test]
    fn point3d_unit_direction_normalised() {
        let p = Point3d::new(3.0, 0.0, 4.0, 0, 0, 0);
        let (nx, _ny, nz) = p.unit_direction().expect("non-zero");
        let len = (nx * nx + nz * nz).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn point3d_unit_direction_origin_returns_none() {
        let p = Point3d::new(0.0, 0.0, 0.0, 0, 0, 0);
        assert!(p.unit_direction().is_none());
    }

    // ── PointCloud ───────────────────────────────────────────────────────────

    #[test]
    fn point_cloud_empty() {
        let pc = PointCloud::new();
        assert!(pc.is_empty());
        assert_eq!(pc.len(), 0);
    }

    #[test]
    fn point_cloud_push_and_len() {
        let mut pc = PointCloud::new();
        pc.push(Point3d::new(1.0, 0.0, 0.0, 255, 0, 0));
        pc.push(Point3d::new(0.0, 1.0, 0.0, 0, 255, 0));
        assert_eq!(pc.len(), 2);
        assert!(!pc.is_empty());
    }

    #[test]
    fn point_cloud_from_points() {
        let pts = vec![
            Point3d::new(1.0, 0.0, 0.0, 10, 20, 30),
            Point3d::new(0.0, 0.0, 1.0, 40, 50, 60),
        ];
        let pc = PointCloud::from_points(pts);
        assert_eq!(pc.len(), 2);
    }

    // ── PointCloudProjector — basic ──────────────────────────────────────────

    #[test]
    fn projector_invalid_dimensions_width_zero() {
        let proj = PointCloudProjector::new();
        let cloud = PointCloud::new();
        assert!(proj.to_equirectangular(&cloud, 0, 64).is_err());
    }

    #[test]
    fn projector_invalid_dimensions_height_zero() {
        let proj = PointCloudProjector::new();
        let cloud = PointCloud::new();
        assert!(proj.to_equirectangular(&cloud, 64, 0).is_err());
    }

    #[test]
    fn projector_empty_cloud_returns_background() {
        let proj = PointCloudProjector::with_background(128, 64, 32);
        let cloud = PointCloud::new();
        let out = proj.to_equirectangular(&cloud, 4, 2).expect("ok");
        assert_eq!(out.len(), 4 * 2 * 3);
        // All pixels should be the background colour
        for chunk in out.chunks(3) {
            assert_eq!(chunk[0], 128);
            assert_eq!(chunk[1], 64);
            assert_eq!(chunk[2], 32);
        }
    }

    #[test]
    fn projector_output_buffer_size() {
        let proj = PointCloudProjector::new();
        let mut cloud = PointCloud::new();
        cloud.push(Point3d::new(1.0, 0.0, 0.0, 255, 0, 0));
        let out = proj.to_equirectangular(&cloud, 64, 32).expect("ok");
        assert_eq!(out.len(), 64 * 32 * 3);
    }

    #[test]
    fn projector_single_point_painted() {
        // Point directly forward along +Z axis → longitude = 0, latitude = 0
        // That maps to u=0.5, v=0.5 → centre of the image
        let proj = PointCloudProjector::new();
        let mut cloud = PointCloud::new();
        cloud.push(Point3d::new(0.0, 0.0, 1.0, 200, 100, 50));
        let width = 64_u32;
        let height = 32_u32;
        let out = proj.to_equirectangular(&cloud, width, height).expect("ok");

        // The point should appear somewhere (at least one non-black pixel).
        let has_painted = out.chunks(3).any(|p| p[0] != 0 || p[1] != 0 || p[2] != 0);
        assert!(has_painted, "Expected at least one painted pixel");
    }

    #[test]
    fn projector_depth_closest_wins() {
        // Two points at exactly the same direction (+Z), different distances.
        // The closer one (r=1) should win over the farther one (r=10).
        let proj = PointCloudProjector::new();
        let mut cloud = PointCloud::new();
        // Farther point (red) — added first
        cloud.push(Point3d::new(0.0, 0.0, 10.0, 255, 0, 0));
        // Closer point (green) — added second
        cloud.push(Point3d::new(0.0, 0.0, 1.0, 0, 255, 0));

        let width = 64_u32;
        let height = 32_u32;
        let out = proj.to_equirectangular(&cloud, width, height).expect("ok");

        // At the pixel that +Z maps to, green should dominate.
        // +Z → lon=0, lat=0 → u=0.5, v=0.5
        let px = (0.5_f32 * width as f32).floor() as usize;
        let py = (0.5_f32 * height as f32).floor() as usize;
        let idx = (py * width as usize + px) * 3;
        // Green channel should be 255, not red
        assert_eq!(out[idx + 1], 255, "Green (closer) should win depth test");
        assert_eq!(out[idx], 0, "Red (farther) should lose depth test");
    }

    #[test]
    fn projector_origin_point_skipped() {
        // A point at the origin cannot be projected; the background should remain.
        let proj = PointCloudProjector::with_background(77, 77, 77);
        let mut cloud = PointCloud::new();
        cloud.push(Point3d::new(0.0, 0.0, 0.0, 255, 255, 255));
        let out = proj.to_equirectangular(&cloud, 4, 4).expect("ok");
        let all_bg = out
            .chunks(3)
            .all(|p| p[0] == 77 && p[1] == 77 && p[2] == 77);
        assert!(all_bg, "Origin point should be skipped, leaving background");
    }

    // ── direction_to_equirect_pixel ───────────────────────────────────────────

    #[test]
    fn direction_to_pixel_forward_z() {
        // +Z maps to centre of image (approximately)
        let (px, py) = direction_to_equirect_pixel(0.0, 0.0, 1.0, 100, 50).expect("non-zero");
        // u=0.5 → px=50, v=0.5 → py=25
        assert_eq!(px, 50);
        assert_eq!(py, 25);
    }

    #[test]
    fn direction_to_pixel_zero_returns_none() {
        assert!(direction_to_equirect_pixel(0.0, 0.0, 0.0, 100, 50).is_none());
    }
}

#![allow(dead_code)]
//! Projection mapping and conversion utilities for virtual production.
//!
//! Supports equirectangular, cubemap, and fisheye projections with coordinate
//! validation and cross-projection conversion.

use std::f64::consts::PI;

/// The type of spherical projection used to encode imagery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionType {
    /// Latitude/longitude mapping onto a flat 2-D image.
    Equirectangular,
    /// Six-face cube unwrapping.
    Cubemap,
    /// Circular fisheye lens projection.
    Fisheye,
}

impl ProjectionType {
    /// Nominal horizontal field of view in degrees for this projection.
    #[must_use]
    pub fn field_of_view_deg(&self) -> f64 {
        match self {
            Self::Equirectangular => 360.0,
            Self::Cubemap => 90.0,
            Self::Fisheye => 180.0,
        }
    }
}

/// A UV-coordinate pair in normalised image space \[0.0, 1.0\].
#[derive(Debug, Clone, Copy)]
pub struct UvCoord {
    /// Horizontal position.
    pub u: f64,
    /// Vertical position.
    pub v: f64,
}

impl UvCoord {
    /// Create a new `UvCoord`.
    #[must_use]
    pub fn new(u: f64, v: f64) -> Self {
        Self { u, v }
    }
}

/// A spherical direction described by azimuth and elevation in radians.
#[derive(Debug, Clone, Copy)]
pub struct SphericalCoord {
    /// Azimuth (yaw) in radians, \[-π, π\].
    pub azimuth: f64,
    /// Elevation (pitch) in radians, \[-π/2, π/2\].
    pub elevation: f64,
}

impl SphericalCoord {
    /// Create a new `SphericalCoord`.
    #[must_use]
    pub fn new(azimuth: f64, elevation: f64) -> Self {
        Self { azimuth, elevation }
    }
}

/// Maps between UV image coordinates and spherical directions.
#[derive(Debug, Clone)]
pub struct ProjectionMap {
    projection: ProjectionType,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl ProjectionMap {
    /// Create a new `ProjectionMap`.
    #[must_use]
    pub fn new(projection: ProjectionType, width: u32, height: u32) -> Self {
        Self {
            projection,
            width,
            height,
        }
    }

    /// Convert a normalised UV coordinate to a spherical direction.
    ///
    /// Returns `None` for projections/coordinates where mapping is undefined.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn map_point(&self, uv: UvCoord) -> Option<SphericalCoord> {
        match self.projection {
            ProjectionType::Equirectangular => {
                let az = (uv.u - 0.5) * 2.0 * PI;
                let el = (0.5 - uv.v) * PI;
                Some(SphericalCoord::new(az, el))
            }
            ProjectionType::Fisheye => {
                let cx = uv.u - 0.5;
                let cy = uv.v - 0.5;
                let r = (cx * cx + cy * cy).sqrt();
                if r > 0.5 {
                    return None; // outside fisheye circle
                }
                let theta = r * PI; // maps [0, 0.5] → [0, π/2]
                let phi = cy.atan2(cx);
                let el = PI / 2.0 - theta;
                Some(SphericalCoord::new(phi, el))
            }
            ProjectionType::Cubemap => {
                // Simplified: treat single face as equirectangular over 90°
                let az = (uv.u - 0.5) * PI / 2.0;
                let el = (0.5 - uv.v) * PI / 2.0;
                Some(SphericalCoord::new(az, el))
            }
        }
    }

    /// Returns `true` if the given UV coordinate is within the valid image area.
    #[must_use]
    pub fn is_valid_coord(&self, uv: UvCoord) -> bool {
        if uv.u < 0.0 || uv.u > 1.0 || uv.v < 0.0 || uv.v > 1.0 {
            return false;
        }
        // For fisheye, additionally check that the point lies within the circle.
        if self.projection == ProjectionType::Fisheye {
            let cx = uv.u - 0.5;
            let cy = uv.v - 0.5;
            return (cx * cx + cy * cy).sqrt() <= 0.5;
        }
        true
    }
}

/// Converts coordinates between two different projection types.
#[derive(Debug)]
pub struct ProjectionConverter {
    src: ProjectionMap,
    dst: ProjectionMap,
}

impl ProjectionConverter {
    /// Create a new `ProjectionConverter` from `src` to `dst`.
    #[must_use]
    pub fn new(src: ProjectionMap, dst: ProjectionMap) -> Self {
        Self { src, dst }
    }

    /// Convert a UV coordinate in the source projection to the destination.
    ///
    /// Returns `None` if the source coordinate is invalid or the resulting
    /// direction cannot be represented in the destination projection.
    #[must_use]
    pub fn convert(&self, uv: UvCoord) -> Option<UvCoord> {
        if !self.src.is_valid_coord(uv) {
            return None;
        }
        let spherical = self.src.map_point(uv)?;
        // Back-project spherical → UV in destination.
        match self.dst.projection {
            ProjectionType::Equirectangular => {
                let u = spherical.azimuth / (2.0 * PI) + 0.5;
                let v = 0.5 - spherical.elevation / PI;
                Some(UvCoord::new(u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)))
            }
            ProjectionType::Fisheye => {
                let theta = PI / 2.0 - spherical.elevation;
                let r = theta / PI; // [0, 0.5] for valid hemisphere
                if r > 0.5 {
                    return None;
                }
                let u = 0.5 + r * spherical.azimuth.cos();
                let v = 0.5 + r * spherical.azimuth.sin();
                Some(UvCoord::new(u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)))
            }
            ProjectionType::Cubemap => {
                let u = spherical.azimuth / (PI / 2.0) + 0.5;
                let v = 0.5 - spherical.elevation / (PI / 2.0);
                Some(UvCoord::new(u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Frustum-to-LED-panel UV mapping with sub-pixel accuracy
// ---------------------------------------------------------------------------

/// A 3D point used in frustum-to-panel mapping.
#[derive(Debug, Clone, Copy)]
pub struct Point3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3d {
    /// Create a new 3D point.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Dot product.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Subtract two points.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Length of the vector.
    #[must_use]
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize to unit length.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len < 1e-15 {
            return *self;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    /// Scale by a scalar.
    #[must_use]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Add two points/vectors.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

/// An LED panel defined by its 4 corner positions in world space.
///
/// Corners are specified in counter-clockwise order when viewed from
/// the camera side: top-left, top-right, bottom-right, bottom-left.
#[derive(Debug, Clone)]
pub struct LedPanel {
    /// Panel identifier.
    pub id: String,
    /// Top-left corner in world space (meters).
    pub top_left: Point3d,
    /// Top-right corner in world space (meters).
    pub top_right: Point3d,
    /// Bottom-right corner in world space (meters).
    pub bottom_right: Point3d,
    /// Bottom-left corner in world space (meters).
    pub bottom_left: Point3d,
    /// Pixel resolution (width, height) of the panel.
    pub resolution: (u32, u32),
}

impl LedPanel {
    /// Compute the panel's normal vector (facing the camera).
    #[must_use]
    pub fn normal(&self) -> Point3d {
        let edge_u = self.top_right.sub(&self.top_left);
        let edge_v = self.bottom_left.sub(&self.top_left);
        edge_u.cross(&edge_v).normalize()
    }

    /// Compute the panel's center position.
    #[must_use]
    pub fn center(&self) -> Point3d {
        Point3d {
            x: (self.top_left.x + self.top_right.x + self.bottom_right.x + self.bottom_left.x)
                * 0.25,
            y: (self.top_left.y + self.top_right.y + self.bottom_right.y + self.bottom_left.y)
                * 0.25,
            z: (self.top_left.z + self.top_right.z + self.bottom_right.z + self.bottom_left.z)
                * 0.25,
        }
    }

    /// Panel width in world units (meters).
    #[must_use]
    pub fn width(&self) -> f64 {
        self.top_right.sub(&self.top_left).length()
    }

    /// Panel height in world units (meters).
    #[must_use]
    pub fn height(&self) -> f64 {
        self.bottom_left.sub(&self.top_left).length()
    }
}

/// Camera frustum definition for projection mapping.
#[derive(Debug, Clone, Copy)]
pub struct CameraFrustumDef {
    /// Camera position in world space (meters).
    pub position: Point3d,
    /// Camera forward direction (unit vector, -Z typically).
    pub forward: Point3d,
    /// Camera up direction (unit vector).
    pub up: Point3d,
    /// Horizontal field of view in radians.
    pub fov_h: f64,
    /// Vertical field of view in radians.
    pub fov_v: f64,
    /// Render resolution (width, height) in pixels.
    pub resolution: (u32, u32),
}

/// A sub-pixel accurate UV mapping result for a single pixel.
#[derive(Debug, Clone, Copy)]
pub struct PixelUvMapping {
    /// Panel-local UV coordinate (0.0-1.0).
    pub uv: UvCoord,
    /// Sub-pixel offset from the nearest panel pixel center.
    /// Used for bilinear interpolation on the LED content texture.
    pub subpixel_offset: (f64, f64),
    /// Panel pixel coordinate (integer) corresponding to this UV.
    pub panel_pixel: (u32, u32),
    /// Angle of incidence between the ray and the panel normal (radians).
    /// Larger angles indicate more oblique viewing, which may need
    /// brightness compensation.
    pub incidence_angle: f64,
}

/// Frustum-to-LED-panel UV mapper.
///
/// Maps each pixel in the camera's frustum to UV coordinates on the LED
/// panel surface, achieving sub-pixel accuracy through bilinear interpolation.
/// This is used to determine what content each LED pixel should display
/// based on the camera's perspective.
pub struct FrustumPanelMapper {
    panels: Vec<LedPanel>,
}

impl FrustumPanelMapper {
    /// Create a new mapper with the given LED panels.
    #[must_use]
    pub fn new(panels: Vec<LedPanel>) -> Self {
        Self { panels }
    }

    /// Add a panel.
    pub fn add_panel(&mut self, panel: LedPanel) {
        self.panels.push(panel);
    }

    /// Number of registered panels.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    /// Cast a ray from the camera through a frustum pixel and find the
    /// intersection with an LED panel, returning the sub-pixel UV mapping.
    ///
    /// Returns `None` if the ray doesn't hit any panel.
    #[must_use]
    pub fn map_pixel(
        &self,
        frustum: &CameraFrustumDef,
        pixel_x: f64,
        pixel_y: f64,
    ) -> Option<(usize, PixelUvMapping)> {
        let ray_dir = self.pixel_to_ray(frustum, pixel_x, pixel_y);

        let mut best_hit: Option<(usize, f64, PixelUvMapping)> = None;

        for (panel_idx, panel) in self.panels.iter().enumerate() {
            if let Some((t, mapping)) =
                self.ray_panel_intersection(&frustum.position, &ray_dir, panel)
            {
                if t > 0.0 {
                    let is_closer = best_hit.as_ref().map_or(true, |(_, best_t, _)| t < *best_t);
                    if is_closer {
                        best_hit = Some((panel_idx, t, mapping));
                    }
                }
            }
        }

        best_hit.map(|(idx, _, mapping)| (idx, mapping))
    }

    /// Build a complete UV map for the entire frustum at the given resolution.
    ///
    /// Returns a 2D array (row-major) of `Option<(panel_index, PixelUvMapping)>`
    /// for each pixel in the frustum.
    #[must_use]
    pub fn build_uv_map(&self, frustum: &CameraFrustumDef) -> Vec<Option<(usize, PixelUvMapping)>> {
        let (w, h) = frustum.resolution;
        let mut map = Vec::with_capacity(w as usize * h as usize);

        for row in 0..h {
            for col in 0..w {
                let px = col as f64 + 0.5; // pixel center
                let py = row as f64 + 0.5;
                map.push(self.map_pixel(frustum, px, py));
            }
        }

        map
    }

    /// Compute the ray direction for a frustum pixel.
    fn pixel_to_ray(&self, frustum: &CameraFrustumDef, px: f64, py: f64) -> Point3d {
        let (w, h) = frustum.resolution;

        // Normalized device coordinates [-1, 1]
        let ndc_x = (2.0 * px / w as f64) - 1.0;
        let ndc_y = 1.0 - (2.0 * py / h as f64);

        // Scale by FOV tangent
        let half_fov_h = frustum.fov_h * 0.5;
        let half_fov_v = frustum.fov_v * 0.5;
        let sx = ndc_x * half_fov_h.tan();
        let sy = ndc_y * half_fov_v.tan();

        // Build camera-space direction
        // Right = up x forward (to get a right-handed coordinate system
        // where +X screen-right, +Y screen-up, forward into the scene)
        let right = frustum.up.cross(&frustum.forward).normalize();
        let true_up = frustum.forward.cross(&right).normalize();

        // ray = forward + sx * right + sy * up
        let ray = frustum
            .forward
            .add(&right.scale(sx))
            .add(&true_up.scale(sy));
        ray.normalize()
    }

    /// Ray-panel intersection using Moller-Trumbore for two triangles.
    ///
    /// Returns (distance, UV mapping) if the ray hits the panel quad.
    fn ray_panel_intersection(
        &self,
        ray_origin: &Point3d,
        ray_dir: &Point3d,
        panel: &LedPanel,
    ) -> Option<(f64, PixelUvMapping)> {
        // Split quad into two triangles:
        // Triangle 1: TL, TR, BL
        // Triangle 2: TR, BR, BL
        let tri1_result = self.ray_triangle_intersection(
            ray_origin,
            ray_dir,
            &panel.top_left,
            &panel.top_right,
            &panel.bottom_left,
        );

        let tri2_result = self.ray_triangle_intersection(
            ray_origin,
            ray_dir,
            &panel.top_right,
            &panel.bottom_right,
            &panel.bottom_left,
        );

        // Pick the closer intersection
        let (t, bary_u, bary_v, triangle) = match (tri1_result, tri2_result) {
            (Some(r1), Some(r2)) => {
                if r1.0 <= r2.0 {
                    (r1.0, r1.1, r1.2, 1)
                } else {
                    (r2.0, r2.1, r2.2, 2)
                }
            }
            (Some(r1), None) => (r1.0, r1.1, r1.2, 1),
            (None, Some(r2)) => (r2.0, r2.1, r2.2, 2),
            (None, None) => return None,
        };

        // Convert barycentric to panel UV
        let (u, v) = if triangle == 1 {
            // Triangle TL-TR-BL: TL=(0,0), TR=(1,0), BL=(0,1)
            // P = (1-u-v)*TL + u*TR + v*BL
            // UV = (u, v)
            (bary_u, bary_v)
        } else {
            // Triangle TR-BR-BL: TR=(1,0), BR=(1,1), BL=(0,1)
            // P = (1-u-v)*TR + u*BR + v*BL
            // UV: at TR: (1,0), at BR: (1,1), at BL: (0,1)
            // Triangle TR-BR-BL UV derivation:
            // P = (1-u-v)*(1,0) + u*(1,1) + v*(0,1)
            // panel_u = (1-u-v)*1 + u*1 + v*0 = 1 - v
            // panel_v = (1-u-v)*0 + u*1 + v*1 = u + v
            let panel_u = 1.0 - bary_v;
            let panel_v = bary_u + bary_v;
            (panel_u, panel_v)
        };

        let uv = UvCoord::new(u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));

        // Sub-pixel computation
        let (pw, ph) = panel.resolution;
        let pixel_x_f = u * pw as f64;
        let pixel_y_f = v * ph as f64;
        let panel_pixel_x = (pixel_x_f as u32).min(pw.saturating_sub(1));
        let panel_pixel_y = (pixel_y_f as u32).min(ph.saturating_sub(1));
        let subpixel_x = pixel_x_f - pixel_x_f.floor();
        let subpixel_y = pixel_y_f - pixel_y_f.floor();

        // Incidence angle
        let normal = panel.normal();
        let cos_angle = ray_dir.dot(&normal).abs();
        let incidence_angle = cos_angle.min(1.0).acos();

        Some((
            t,
            PixelUvMapping {
                uv,
                subpixel_offset: (subpixel_x, subpixel_y),
                panel_pixel: (panel_pixel_x, panel_pixel_y),
                incidence_angle,
            },
        ))
    }

    /// Moller-Trumbore ray-triangle intersection.
    ///
    /// Returns `(t, u, v)` where t is the ray parameter and (u, v) are
    /// barycentric coordinates in the triangle.
    fn ray_triangle_intersection(
        &self,
        origin: &Point3d,
        dir: &Point3d,
        v0: &Point3d,
        v1: &Point3d,
        v2: &Point3d,
    ) -> Option<(f64, f64, f64)> {
        let edge1 = v1.sub(v0);
        let edge2 = v2.sub(v0);
        let h = dir.cross(&edge2);
        let a = edge1.dot(&h);

        if a.abs() < 1e-12 {
            return None; // Ray parallel to triangle
        }

        let f = 1.0 / a;
        let s = origin.sub(v0);
        let u = f * s.dot(&h);

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(&edge1);
        let v = f * dir.dot(&q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * edge2.dot(&q);
        if t > 1e-12 {
            Some((t, u, v))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equirectangular_fov() {
        assert!((ProjectionType::Equirectangular.field_of_view_deg() - 360.0).abs() < 1e-9);
    }

    #[test]
    fn test_cubemap_fov() {
        assert!((ProjectionType::Cubemap.field_of_view_deg() - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_fisheye_fov() {
        assert!((ProjectionType::Fisheye.field_of_view_deg() - 180.0).abs() < 1e-9);
    }

    #[test]
    fn test_equirect_center_maps_to_origin() {
        let map = ProjectionMap::new(ProjectionType::Equirectangular, 3840, 1920);
        let coord = map
            .map_point(UvCoord::new(0.5, 0.5))
            .expect("should succeed in test");
        assert!(coord.azimuth.abs() < 1e-9);
        assert!(coord.elevation.abs() < 1e-9);
    }

    #[test]
    fn test_fisheye_center_maps() {
        let map = ProjectionMap::new(ProjectionType::Fisheye, 1024, 1024);
        let coord = map
            .map_point(UvCoord::new(0.5, 0.5))
            .expect("should succeed in test");
        // Center of fisheye → elevation = π/2 (straight up).
        assert!((coord.elevation - PI / 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_fisheye_outside_circle_returns_none() {
        let map = ProjectionMap::new(ProjectionType::Fisheye, 1024, 1024);
        let result = map.map_point(UvCoord::new(0.0, 0.0));
        assert!(result.is_none());
    }

    #[test]
    fn test_is_valid_coord_in_range() {
        let map = ProjectionMap::new(ProjectionType::Equirectangular, 3840, 1920);
        assert!(map.is_valid_coord(UvCoord::new(0.5, 0.5)));
        assert!(map.is_valid_coord(UvCoord::new(0.0, 0.0)));
        assert!(map.is_valid_coord(UvCoord::new(1.0, 1.0)));
    }

    #[test]
    fn test_is_valid_coord_out_of_range() {
        let map = ProjectionMap::new(ProjectionType::Equirectangular, 3840, 1920);
        assert!(!map.is_valid_coord(UvCoord::new(-0.1, 0.5)));
        assert!(!map.is_valid_coord(UvCoord::new(1.1, 0.5)));
    }

    #[test]
    fn test_fisheye_valid_coord_in_circle() {
        let map = ProjectionMap::new(ProjectionType::Fisheye, 1024, 1024);
        assert!(map.is_valid_coord(UvCoord::new(0.5, 0.5)));
    }

    #[test]
    fn test_fisheye_invalid_coord_outside_circle() {
        let map = ProjectionMap::new(ProjectionType::Fisheye, 1024, 1024);
        assert!(!map.is_valid_coord(UvCoord::new(0.05, 0.05)));
    }

    #[test]
    fn test_converter_equirect_to_equirect_roundtrip() {
        let src = ProjectionMap::new(ProjectionType::Equirectangular, 3840, 1920);
        let dst = ProjectionMap::new(ProjectionType::Equirectangular, 3840, 1920);
        let converter = ProjectionConverter::new(src, dst);
        let uv_in = UvCoord::new(0.5, 0.5);
        let uv_out = converter.convert(uv_in).expect("should succeed in test");
        assert!((uv_out.u - 0.5).abs() < 1e-9);
        assert!((uv_out.v - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_converter_invalid_src_returns_none() {
        let src = ProjectionMap::new(ProjectionType::Equirectangular, 3840, 1920);
        let dst = ProjectionMap::new(ProjectionType::Equirectangular, 3840, 1920);
        let converter = ProjectionConverter::new(src, dst);
        assert!(converter.convert(UvCoord::new(-1.0, 0.5)).is_none());
    }

    #[test]
    fn test_cubemap_maps_to_some() {
        let map = ProjectionMap::new(ProjectionType::Cubemap, 1024, 1024);
        let result = map.map_point(UvCoord::new(0.5, 0.5));
        assert!(result.is_some());
    }

    // --- Frustum-to-LED-panel UV mapping tests ---

    /// Helper: create a simple flat LED panel facing -Z (camera faces +Z toward it).
    fn make_flat_panel(id: &str, width: f64, height: f64, distance: f64) -> LedPanel {
        let half_w = width * 0.5;
        let half_h = height * 0.5;
        LedPanel {
            id: id.to_string(),
            top_left: Point3d::new(-half_w, half_h, distance),
            top_right: Point3d::new(half_w, half_h, distance),
            bottom_right: Point3d::new(half_w, -half_h, distance),
            bottom_left: Point3d::new(-half_w, -half_h, distance),
            resolution: (1920, 1080),
        }
    }

    /// Helper: create a frustum looking along +Z.
    fn make_forward_frustum(fov_h_deg: f64, fov_v_deg: f64) -> CameraFrustumDef {
        CameraFrustumDef {
            position: Point3d::new(0.0, 0.0, 0.0),
            forward: Point3d::new(0.0, 0.0, 1.0),
            up: Point3d::new(0.0, 1.0, 0.0),
            fov_h: fov_h_deg.to_radians(),
            fov_v: fov_v_deg.to_radians(),
            resolution: (64, 36),
        }
    }

    #[test]
    fn test_frustum_mapper_creation() {
        let mapper = FrustumPanelMapper::new(vec![]);
        assert_eq!(mapper.panel_count(), 0);
    }

    #[test]
    fn test_panel_normal() {
        let panel = make_flat_panel("P1", 2.0, 1.0, 5.0);
        let normal = panel.normal();
        // Panel faces -Z (toward camera at origin)
        assert!(normal.z.abs() > 0.99, "normal z: {}", normal.z);
    }

    #[test]
    fn test_panel_center() {
        let panel = make_flat_panel("P1", 2.0, 1.0, 5.0);
        let center = panel.center();
        assert!((center.x).abs() < 1e-10);
        assert!((center.y).abs() < 1e-10);
        assert!((center.z - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_panel_dimensions() {
        let panel = make_flat_panel("P1", 2.0, 1.5, 5.0);
        assert!((panel.width() - 2.0).abs() < 1e-10);
        assert!((panel.height() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_frustum_mapper_center_ray_hits_panel() {
        let panel = make_flat_panel("P1", 4.0, 2.0, 5.0);
        let mapper = FrustumPanelMapper::new(vec![panel]);
        let frustum = make_forward_frustum(60.0, 34.0);

        // Center pixel should hit the panel
        let result = mapper.map_pixel(&frustum, 32.0, 18.0);
        assert!(result.is_some(), "center ray should hit panel");

        let (panel_idx, mapping) = result.expect("checked above");
        assert_eq!(panel_idx, 0);
        // UV should be near center (0.5, 0.5)
        assert!(
            (mapping.uv.u - 0.5).abs() < 0.1,
            "center u: {}",
            mapping.uv.u
        );
        assert!(
            (mapping.uv.v - 0.5).abs() < 0.1,
            "center v: {}",
            mapping.uv.v
        );
    }

    #[test]
    fn test_frustum_mapper_incidence_angle_at_center() {
        let panel = make_flat_panel("P1", 4.0, 2.0, 5.0);
        let mapper = FrustumPanelMapper::new(vec![panel]);
        let frustum = make_forward_frustum(60.0, 34.0);

        let (_, mapping) = mapper
            .map_pixel(&frustum, 32.0, 18.0)
            .expect("should hit panel");

        // At center, incidence should be ~0 (perpendicular to panel)
        assert!(
            mapping.incidence_angle < 0.1,
            "center incidence angle should be small: {}",
            mapping.incidence_angle
        );
    }

    #[test]
    fn test_frustum_mapper_subpixel_accuracy() {
        let panel = make_flat_panel("P1", 4.0, 2.0, 5.0);
        let mapper = FrustumPanelMapper::new(vec![panel]);
        let frustum = make_forward_frustum(60.0, 34.0);

        let (_, mapping) = mapper
            .map_pixel(&frustum, 32.0, 18.0)
            .expect("should hit panel");

        // Sub-pixel offsets should be in [0, 1)
        assert!(
            mapping.subpixel_offset.0 >= 0.0 && mapping.subpixel_offset.0 < 1.0,
            "subpixel x: {}",
            mapping.subpixel_offset.0
        );
        assert!(
            mapping.subpixel_offset.1 >= 0.0 && mapping.subpixel_offset.1 < 1.0,
            "subpixel y: {}",
            mapping.subpixel_offset.1
        );
    }

    #[test]
    fn test_frustum_mapper_miss_outside_panel() {
        // Small panel that doesn't cover the whole frustum
        let panel = make_flat_panel("P1", 0.1, 0.1, 5.0);
        let mapper = FrustumPanelMapper::new(vec![panel]);
        let frustum = make_forward_frustum(90.0, 60.0);

        // Corner pixel should miss the tiny panel
        let result = mapper.map_pixel(&frustum, 0.0, 0.0);
        assert!(result.is_none(), "corner ray should miss tiny panel");
    }

    #[test]
    fn test_frustum_mapper_multiple_panels_nearest_wins() {
        let near_panel = make_flat_panel("near", 4.0, 2.0, 3.0);
        let far_panel = make_flat_panel("far", 4.0, 2.0, 10.0);
        let mapper = FrustumPanelMapper::new(vec![far_panel, near_panel]);
        let frustum = make_forward_frustum(60.0, 34.0);

        let (panel_idx, _) = mapper
            .map_pixel(&frustum, 32.0, 18.0)
            .expect("should hit a panel");

        // Panel index 1 is the near panel (added second)
        assert_eq!(panel_idx, 1, "nearest panel should win");
    }

    #[test]
    fn test_frustum_mapper_build_uv_map() {
        let panel = make_flat_panel("P1", 10.0, 6.0, 5.0);
        let mapper = FrustumPanelMapper::new(vec![panel]);
        let frustum = CameraFrustumDef {
            position: Point3d::new(0.0, 0.0, 0.0),
            forward: Point3d::new(0.0, 0.0, 1.0),
            up: Point3d::new(0.0, 1.0, 0.0),
            fov_h: 60.0_f64.to_radians(),
            fov_v: 34.0_f64.to_radians(),
            resolution: (8, 6),
        };

        let uv_map = mapper.build_uv_map(&frustum);
        assert_eq!(uv_map.len(), 48); // 8 * 6

        // At least the center pixels should have hits
        let center_idx = 3 * 8 + 4; // row 3, col 4
        assert!(uv_map[center_idx].is_some(), "center should be mapped");
    }

    #[test]
    fn test_frustum_mapper_off_center_uv() {
        let panel = make_flat_panel("P1", 4.0, 2.0, 5.0);
        let mapper = FrustumPanelMapper::new(vec![panel]);
        let frustum = make_forward_frustum(60.0, 34.0);

        // Pixel to the right of center should have u > 0.5
        let (_, mapping_right) = mapper
            .map_pixel(&frustum, 48.0, 18.0)
            .expect("should hit panel");
        assert!(
            mapping_right.uv.u > 0.5,
            "right pixel should have u > 0.5: {}",
            mapping_right.uv.u
        );

        // Pixel to the left of center should have u < 0.5
        let (_, mapping_left) = mapper
            .map_pixel(&frustum, 16.0, 18.0)
            .expect("should hit panel");
        assert!(
            mapping_left.uv.u < 0.5,
            "left pixel should have u < 0.5: {}",
            mapping_left.uv.u
        );
    }

    #[test]
    fn test_point3d_operations() {
        let a = Point3d::new(1.0, 0.0, 0.0);
        let b = Point3d::new(0.0, 1.0, 0.0);

        let cross = a.cross(&b);
        assert!((cross.z - 1.0).abs() < 1e-10);

        let dot = a.dot(&b);
        assert!(dot.abs() < 1e-10);

        let diff = a.sub(&b);
        assert!((diff.x - 1.0).abs() < 1e-10);
        assert!((diff.y - (-1.0)).abs() < 1e-10);

        let normalized = Point3d::new(3.0, 4.0, 0.0).normalize();
        assert!((normalized.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_frustum_mapper_oblique_incidence() {
        let panel = make_flat_panel("P1", 10.0, 6.0, 5.0);
        let mapper = FrustumPanelMapper::new(vec![panel]);
        let frustum = make_forward_frustum(90.0, 60.0);

        // Center hit
        let (_, center_mapping) = mapper
            .map_pixel(&frustum, 32.0, 18.0)
            .expect("should hit panel");

        // Edge hit (if it hits)
        if let Some((_, edge_mapping)) = mapper.map_pixel(&frustum, 2.0, 2.0) {
            // Edge should have a larger incidence angle than center
            assert!(
                edge_mapping.incidence_angle > center_mapping.incidence_angle,
                "edge incidence {} should exceed center {}",
                edge_mapping.incidence_angle,
                center_mapping.incidence_angle
            );
        }
    }
}

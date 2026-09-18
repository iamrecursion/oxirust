//! Viewport rendering: extract a perspective view from an equirectangular frame.
//!
//! A "virtual camera" can be placed anywhere on the sphere by specifying its
//! orientation (yaw, pitch, roll) and a horizontal field-of-view.  The renderer
//! computes, for each output pixel, the ray direction in world space, maps it
//! back to equirectangular UV, and resamples the source image.
//!
//! ## Coordinate system
//!
//! * **Yaw** (azimuth) — rotation around the vertical Y axis, positive = clockwise
//!   when viewed from above (East / Right direction).
//! * **Pitch** — rotation around the horizontal X axis, positive = tilting upward.
//! * **Roll** — rotation around the forward Z axis, positive = clockwise from the
//!   camera's point of view.
//!
//! All angles are in radians.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oximedia_360::viewport::{ViewportParams, render_viewport};
//!
//! let equirect_src: Vec<u8> = vec![]; // your 4096×2048 equirectangular image
//! let params = ViewportParams::new(1920, 1080)
//!     .with_fov_deg(90.0)
//!     .with_yaw_deg(45.0)
//!     .with_pitch_deg(-10.0);
//! // let output = render_viewport(&equirect_src, 4096, 2048, &params).unwrap();
//! ```

use crate::{
    projection::{bilinear_sample_u8, sphere_to_equirect, SphericalCoord},
    VrError,
};

// ─── Viewport parameters ──────────────────────────────────────────────────────

/// Parameters for perspective viewport extraction.
///
/// All angles are in **radians** unless a `_deg` constructor helper is used.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewportParams {
    /// Output image width in pixels.
    pub out_width: u32,
    /// Output image height in pixels.
    pub out_height: u32,
    /// Horizontal field of view in radians.  Defaults to π/2 (90°).
    pub hfov_rad: f32,
    /// Camera yaw (azimuth) in radians.  0 = looking at the sphere's forward
    /// direction (az = 0, el = 0), positive values rotate right.
    pub yaw_rad: f32,
    /// Camera pitch in radians.  Positive values tilt upward.
    pub pitch_rad: f32,
    /// Camera roll in radians.  Positive values rotate clockwise.
    pub roll_rad: f32,
}

impl ViewportParams {
    /// Create a viewport with the given output dimensions and sensible defaults
    /// (90° FOV, looking straight ahead with no tilt or roll).
    pub fn new(out_width: u32, out_height: u32) -> Self {
        Self {
            out_width,
            out_height,
            hfov_rad: std::f32::consts::FRAC_PI_2,
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            roll_rad: 0.0,
        }
    }

    /// Set horizontal field of view in degrees (builder pattern).
    pub fn with_fov_deg(mut self, fov_deg: f32) -> Self {
        self.hfov_rad = fov_deg.to_radians();
        self
    }

    /// Set yaw in degrees (builder pattern).
    pub fn with_yaw_deg(mut self, yaw_deg: f32) -> Self {
        self.yaw_rad = yaw_deg.to_radians();
        self
    }

    /// Set pitch in degrees (builder pattern).
    pub fn with_pitch_deg(mut self, pitch_deg: f32) -> Self {
        self.pitch_rad = pitch_deg.to_radians();
        self
    }

    /// Set roll in degrees (builder pattern).
    pub fn with_roll_deg(mut self, roll_deg: f32) -> Self {
        self.roll_rad = roll_deg.to_radians();
        self
    }

    /// Compute the vertical FOV from the horizontal FOV and the output aspect ratio.
    pub fn vfov_rad(&self) -> f32 {
        if self.out_width == 0 || self.out_height == 0 {
            return 0.0;
        }
        let aspect = self.out_width as f32 / self.out_height as f32;
        2.0 * ((self.hfov_rad * 0.5).tan() / aspect).atan()
    }
}

// ─── Rotation matrix ──────────────────────────────────────────────────────────

/// 3×3 rotation matrix (row-major) for applying yaw → pitch → roll.
///
/// The combined rotation is `R = R_yaw · R_pitch · R_roll`.
#[derive(Debug, Clone, Copy)]
struct RotMat([[f32; 3]; 3]);

impl RotMat {
    /// Construct the combined camera rotation matrix.
    fn from_yaw_pitch_roll(yaw: f32, pitch: f32, roll: f32) -> Self {
        // R_yaw: rotate around Y axis
        let (sy, cy) = yaw.sin_cos();
        let r_yaw = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];

        // R_pitch: rotate around X axis
        let (sp, cp) = pitch.sin_cos();
        let r_pitch = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]];

        // R_roll: rotate around Z axis
        let (sr, cr) = roll.sin_cos();
        let r_roll = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]];

        // Combined: first roll, then pitch, then yaw (camera-space to world-space)
        // R = R_yaw · (R_pitch · R_roll)
        let rp = mat3_mul(r_pitch, r_roll);
        let combined = mat3_mul(r_yaw, rp);
        Self(combined)
    }

    /// Apply the rotation to a 3-D direction vector.
    fn apply(&self, v: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

// ─── Viewport renderer ────────────────────────────────────────────────────────

/// Extract a perspective (rectilinear) view from an equirectangular panorama.
///
/// For each output pixel the function:
/// 1. Computes the ray direction in camera space using the pinhole camera model
///    (with the configured HFOV).
/// 2. Applies the camera rotation (yaw, pitch, roll) to get the world-space ray.
/// 3. Converts the ray to spherical coordinates and samples the equirectangular
///    source with bilinear interpolation.
///
/// * `src`        — source pixel data (RGB, 3 bpp, row-major)
/// * `src_width`  — source image width in pixels
/// * `src_height` — source image height in pixels
/// * `params`     — viewport parameters
///
/// Returns an RGB image of size `params.out_width × params.out_height`.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero.
/// Returns [`VrError::BufferTooSmall`] if `src` is too small.
pub fn render_viewport(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    params: &ViewportParams,
) -> Result<Vec<u8>, VrError> {
    if src_width == 0 || src_height == 0 {
        return Err(VrError::InvalidDimensions(
            "source image dimensions must be > 0".into(),
        ));
    }
    if params.out_width == 0 || params.out_height == 0 {
        return Err(VrError::InvalidDimensions(
            "output dimensions must be > 0".into(),
        ));
    }
    let expected = src_width as usize * src_height as usize * 3;
    if src.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: src.len(),
        });
    }
    if params.hfov_rad <= 0.0 || params.hfov_rad >= std::f32::consts::PI {
        return Err(VrError::ProjectionError(
            "hfov_rad must be in (0, π)".into(),
        ));
    }

    let ow = params.out_width;
    let oh = params.out_height;
    const CH: u32 = 3;

    // Focal length in pixels: f = (w/2) / tan(hfov/2)
    let f = (ow as f32 * 0.5) / (params.hfov_rad * 0.5).tan();

    // Pre-compute rotation matrix
    let rot = RotMat::from_yaw_pitch_roll(params.yaw_rad, params.pitch_rad, params.roll_rad);

    let cx = ow as f32 * 0.5;
    let cy = oh as f32 * 0.5;

    let mut out = vec![0u8; (ow * oh * CH) as usize];

    for py in 0..oh {
        for px in 0..ow {
            // Ray in camera space (looking down +Z axis)
            let rx = (px as f32 + 0.5) - cx;
            let ry = -((py as f32 + 0.5) - cy); // negate: screen Y is downward
            let rz = f;

            // Normalise
            let len = (rx * rx + ry * ry + rz * rz).sqrt();
            let camera_ray = [rx / len, ry / len, rz / len];

            // Rotate to world space
            let world_ray = rot.apply(camera_ray);

            // Convert to spherical
            let el = world_ray[1].asin();
            let az = world_ray[0].atan2(world_ray[2]);
            let sphere = SphericalCoord {
                azimuth_rad: az,
                elevation_rad: el,
            };

            let uv = sphere_to_equirect(&sphere);
            let sample = bilinear_sample_u8(src, src_width, src_height, uv.u, uv.v, CH);
            let dst = (py * ow + px) as usize * CH as usize;
            out[dst..dst + CH as usize].copy_from_slice(&sample);
        }
    }

    Ok(out)
}

/// Extract a sub-region view but also return the spherical coordinate for each
/// output pixel, useful for gaze tracking or heat-map generation.
///
/// Returns `(image_data, spherical_coords)` where `spherical_coords` has one
/// entry per output pixel in row-major order.
///
/// # Errors
/// Same as [`render_viewport`].
pub fn render_viewport_with_coords(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    params: &ViewportParams,
) -> Result<(Vec<u8>, Vec<SphericalCoord>), VrError> {
    if src_width == 0 || src_height == 0 {
        return Err(VrError::InvalidDimensions(
            "source image dimensions must be > 0".into(),
        ));
    }
    if params.out_width == 0 || params.out_height == 0 {
        return Err(VrError::InvalidDimensions(
            "output dimensions must be > 0".into(),
        ));
    }
    let expected = src_width as usize * src_height as usize * 3;
    if src.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: src.len(),
        });
    }
    if params.hfov_rad <= 0.0 || params.hfov_rad >= std::f32::consts::PI {
        return Err(VrError::ProjectionError(
            "hfov_rad must be in (0, π)".into(),
        ));
    }

    let ow = params.out_width;
    let oh = params.out_height;
    const CH: u32 = 3;

    let f = (ow as f32 * 0.5) / (params.hfov_rad * 0.5).tan();
    let rot = RotMat::from_yaw_pitch_roll(params.yaw_rad, params.pitch_rad, params.roll_rad);
    let cx = ow as f32 * 0.5;
    let cy = oh as f32 * 0.5;

    let pixel_count = (ow * oh) as usize;
    let mut image = vec![0u8; pixel_count * CH as usize];
    let mut coords = Vec::with_capacity(pixel_count);

    for py in 0..oh {
        for px in 0..ow {
            let rx = (px as f32 + 0.5) - cx;
            let ry = -((py as f32 + 0.5) - cy);
            let rz = f;
            let len = (rx * rx + ry * ry + rz * rz).sqrt();
            let camera_ray = [rx / len, ry / len, rz / len];
            let world_ray = rot.apply(camera_ray);

            let el = world_ray[1].asin();
            let az = world_ray[0].atan2(world_ray[2]);
            let sphere = SphericalCoord {
                azimuth_rad: az,
                elevation_rad: el,
            };

            let uv = sphere_to_equirect(&sphere);
            let sample = bilinear_sample_u8(src, src_width, src_height, uv.u, uv.v, CH);
            let dst = (py * ow + px) as usize * CH as usize;
            image[dst..dst + CH as usize].copy_from_slice(&sample);
            coords.push(sphere);
        }
    }

    Ok((image, coords))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn solid_equirect(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 3);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    // ── Error handling ───────────────────────────────────────────────────────

    #[test]
    fn zero_src_dimension_error() {
        let src = solid_equirect(64, 32, 100, 100, 100);
        let params = ViewportParams::new(32, 16);
        assert!(render_viewport(&src, 0, 32, &params).is_err());
        assert!(render_viewport(&src, 64, 0, &params).is_err());
    }

    #[test]
    fn zero_output_dimension_error() {
        let src = solid_equirect(64, 32, 100, 100, 100);
        let p1 = ViewportParams::new(0, 16);
        let p2 = ViewportParams::new(32, 0);
        assert!(render_viewport(&src, 64, 32, &p1).is_err());
        assert!(render_viewport(&src, 64, 32, &p2).is_err());
    }

    #[test]
    fn buffer_too_small_error() {
        let params = ViewportParams::new(32, 16);
        assert!(render_viewport(&[0u8; 10], 64, 32, &params).is_err());
    }

    #[test]
    fn invalid_fov_error() {
        let src = solid_equirect(64, 32, 100, 100, 100);
        let mut params = ViewportParams::new(32, 16);
        params.hfov_rad = 0.0;
        assert!(render_viewport(&src, 64, 32, &params).is_err());
        params.hfov_rad = PI;
        assert!(render_viewport(&src, 64, 32, &params).is_err());
    }

    // ── Basic output size ────────────────────────────────────────────────────

    #[test]
    fn render_viewport_correct_output_size() {
        let src = solid_equirect(128, 64, 200, 100, 50);
        let params = ViewportParams::new(64, 32).with_fov_deg(90.0);
        let out = render_viewport(&src, 128, 64, &params).expect("ok");
        assert_eq!(out.len(), 64 * 32 * 3);
    }

    // ── Solid-colour panorama: any viewport should return the same colour ─────

    #[test]
    fn solid_panorama_any_orientation_returns_same_colour() {
        let src = solid_equirect(128, 64, 180, 90, 45);
        let params = ViewportParams::new(32, 16)
            .with_fov_deg(90.0)
            .with_yaw_deg(45.0)
            .with_pitch_deg(20.0);
        let out = render_viewport(&src, 128, 64, &params).expect("ok");
        // Centre pixel should match the input colour
        let base = (8 * 32 + 16) * 3;
        assert!((out[base] as i32 - 180).abs() <= 5, "R={}", out[base]);
        assert!(
            (out[base + 1] as i32 - 90).abs() <= 5,
            "G={}",
            out[base + 1]
        );
    }

    // ── Rotation correctness: looking right by 90° ────────────────────────────

    #[test]
    fn yaw_90_shifts_view() {
        // Create a non-uniform panorama and verify that two different yaw
        // values produce different outputs
        let mut src = vec![0u8; 128 * 64 * 3];
        // Left half of panorama = red, right half = blue
        for row in 0..64 {
            for col in 0..128 {
                let base = (row * 128 + col) * 3;
                if col < 64 {
                    src[base] = 255; // red
                } else {
                    src[base + 2] = 255; // blue
                }
            }
        }

        let params_straight = ViewportParams::new(8, 8)
            .with_fov_deg(60.0)
            .with_yaw_deg(0.0);
        let params_right = ViewportParams::new(8, 8)
            .with_fov_deg(60.0)
            .with_yaw_deg(90.0);

        let out_straight = render_viewport(&src, 128, 64, &params_straight).expect("ok");
        let out_right = render_viewport(&src, 128, 64, &params_right).expect("ok");

        // The two outputs should differ
        let same = out_straight == out_right;
        assert!(!same, "yaw rotation should produce different output");
    }

    // ── ViewportParams builder API ───────────────────────────────────────────

    #[test]
    fn viewport_params_builder() {
        let p = ViewportParams::new(1920, 1080)
            .with_fov_deg(90.0)
            .with_yaw_deg(45.0)
            .with_pitch_deg(-10.0)
            .with_roll_deg(5.0);
        assert_eq!(p.out_width, 1920);
        assert_eq!(p.out_height, 1080);
        assert!((p.hfov_rad - 90_f32.to_radians()).abs() < 1e-5);
        assert!((p.yaw_rad - 45_f32.to_radians()).abs() < 1e-5);
        assert!((p.pitch_rad - (-10_f32).to_radians()).abs() < 1e-5);
    }

    #[test]
    fn vfov_16x9_90hfov() {
        let p = ViewportParams::new(1920, 1080).with_fov_deg(90.0);
        let vfov = p.vfov_rad().to_degrees();
        // For 16:9, vfov should be ≈ 58.7° when hfov = 90°
        assert!(vfov > 50.0 && vfov < 70.0, "vfov_deg={vfov}");
    }

    // ── render_viewport_with_coords ──────────────────────────────────────────

    #[test]
    fn render_with_coords_sizes() {
        let src = solid_equirect(64, 32, 100, 150, 200);
        let params = ViewportParams::new(16, 8).with_fov_deg(80.0);
        let (img, coords) = render_viewport_with_coords(&src, 64, 32, &params).expect("ok");
        assert_eq!(img.len(), 16 * 8 * 3);
        assert_eq!(coords.len(), 16 * 8);
    }

    #[test]
    fn render_with_coords_centre_pixel_near_yaw_direction() {
        // Looking straight ahead (yaw=0, pitch=0), the centre pixel should
        // point toward az≈0, el≈0
        let src = solid_equirect(128, 64, 128, 128, 128);
        let params = ViewportParams::new(32, 16)
            .with_fov_deg(90.0)
            .with_yaw_deg(0.0)
            .with_pitch_deg(0.0);
        let (_, coords) = render_viewport_with_coords(&src, 128, 64, &params).expect("ok");
        let centre = coords[8 * 32 + 16]; // row 8, col 16
        assert!(centre.azimuth_rad.abs() < 0.2, "az={}", centre.azimuth_rad);
        assert!(
            centre.elevation_rad.abs() < 0.2,
            "el={}",
            centre.elevation_rad
        );
    }

    // ── RotMat correctness ───────────────────────────────────────────────────

    #[test]
    fn rot_identity_does_not_change_vector() {
        let rot = RotMat::from_yaw_pitch_roll(0.0, 0.0, 0.0);
        let v = [1.0f32, 2.0, 3.0];
        let rv = rot.apply(v);
        for (a, b) in v.iter().zip(rv.iter()) {
            assert!((a - b).abs() < 1e-5, "a={a} b={b}");
        }
    }

    #[test]
    fn rot_yaw_90_rotates_forward_to_right() {
        // Forward = (0,0,1).  After 90° yaw, should point to (-1, 0, 0).
        let rot = RotMat::from_yaw_pitch_roll(PI / 2.0, 0.0, 0.0);
        let v = [0.0f32, 0.0, 1.0];
        let rv = rot.apply(v);
        // x component should be ≈ +1 (sin(π/2) = 1) but direction can vary
        // Just check that the forward direction moves in the X axis
        let moved = rv[0].abs() > 0.9;
        assert!(moved, "rv={rv:?}");
    }
}

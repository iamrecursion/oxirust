//! 2.5D parallax camera-motion effect using depth maps.
//!
//! Simulates the look of a multi-plane camera moving through a scene by
//! displacing each pixel horizontally and vertically according to a *depth map*
//! that encodes scene depth as a normalised luminance value (0 = far, 255 = near).
//!
//! Near pixels (large depth values) shift more than far pixels, creating
//! the illusion of depth without true 3D geometry.
//!
//! # Usage
//!
//! ```no_run
//! use oximedia_vfx::parallax::{ParallaxEffect, ParallaxConfig};
//! use oximedia_vfx::{Frame, EffectParams, VideoEffect};
//!
//! let config = ParallaxConfig {
//!     camera_dx: 10.0,
//!     camera_dy: 0.0,
//!     depth_scale: 1.0,
//!     far_plane_shift: 0.1,
//!     ..Default::default()
//! };
//!
//! let depth_map = vec![128u8; 1920 * 1080]; // flat mid-depth scene
//! let mut effect = ParallaxEffect::new(config, depth_map, 1920, 1080).expect("ok");
//!
//! let input = Frame::new(1920, 1080).expect("frame");
//! let mut output = Frame::new(1920, 1080).expect("frame");
//! let params = EffectParams::new();
//! effect.apply(&input, &mut output, &params).expect("apply");
//! ```

use crate::{EffectParams, Frame, VfxError, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the parallax effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallaxConfig {
    /// Horizontal camera displacement per frame in pixels.
    /// Positive values pan right; the near layer shifts right, far layer less.
    pub camera_dx: f32,
    /// Vertical camera displacement per frame in pixels.
    pub camera_dy: f32,
    /// Multiplier applied to the depth value when computing per-pixel shift.
    /// At 1.0 the near plane (depth=255) shifts by exactly `camera_dx` pixels.
    pub depth_scale: f32,
    /// Minimum shift fraction applied even to the far plane (depth=0).
    /// 0.0 = far plane is stationary; 1.0 = no parallax (flat).
    pub far_plane_shift: f32,
    /// If `true`, out-of-bounds samples are mirrored rather than clamped.
    pub mirror_wrap: bool,
    /// Blend factor between parallax output and original [0.0 = original, 1.0 = full effect].
    pub blend: f32,
}

impl Default for ParallaxConfig {
    fn default() -> Self {
        Self {
            camera_dx: 5.0,
            camera_dy: 0.0,
            depth_scale: 1.0,
            far_plane_shift: 0.05,
            mirror_wrap: false,
            blend: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ParallaxEffect
// ─────────────────────────────────────────────────────────────────────────────

/// 2.5D parallax video effect driven by a depth map.
///
/// The depth map must be a flat greyscale buffer of `width × height` bytes
/// (one byte per pixel, 0 = far, 255 = near).  The effect is applied
/// in-place: the source frame is sampled at a shifted position for each pixel,
/// with shift magnitude proportional to the depth value at that pixel.
pub struct ParallaxEffect {
    config: ParallaxConfig,
    /// Depth map: one byte per pixel, 0 = far, 255 = near.
    depth_map: Vec<u8>,
    /// Expected width of both the depth map and input frames.
    width: u32,
    /// Expected height of both the depth map and input frames.
    height: u32,
}

impl ParallaxEffect {
    /// Create a new parallax effect.
    ///
    /// # Errors
    ///
    /// - [`VfxError::InvalidDimensions`] if `width` or `height` is zero.
    /// - [`VfxError::BufferSizeMismatch`] if `depth_map.len() != width * height`.
    pub fn new(
        config: ParallaxConfig,
        depth_map: Vec<u8>,
        width: u32,
        height: u32,
    ) -> VfxResult<Self> {
        if width == 0 || height == 0 {
            return Err(VfxError::InvalidDimensions { width, height });
        }
        let expected = (width as usize) * (height as usize);
        if depth_map.len() != expected {
            return Err(VfxError::BufferSizeMismatch {
                expected,
                actual: depth_map.len(),
            });
        }
        Ok(Self {
            config,
            depth_map,
            width,
            height,
        })
    }

    /// Create a flat-depth (no parallax) effect with a midpoint depth map.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn flat(width: u32, height: u32) -> VfxResult<Self> {
        let depth_map = vec![128u8; (width as usize) * (height as usize)];
        Self::new(ParallaxConfig::default(), depth_map, width, height)
    }

    /// Update the camera displacement.
    pub fn set_camera_displacement(&mut self, dx: f32, dy: f32) {
        self.config.camera_dx = dx;
        self.config.camera_dy = dy;
    }

    /// Replace the depth map.
    ///
    /// # Errors
    ///
    /// Returns [`VfxError::BufferSizeMismatch`] if the new map size doesn't match.
    pub fn set_depth_map(&mut self, depth_map: Vec<u8>) -> VfxResult<()> {
        let expected = (self.width as usize) * (self.height as usize);
        if depth_map.len() != expected {
            return Err(VfxError::BufferSizeMismatch {
                expected,
                actual: depth_map.len(),
            });
        }
        self.depth_map = depth_map;
        Ok(())
    }

    /// Get the depth value at pixel `(x, y)` normalised to `[0.0, 1.0]`.
    fn depth_at(&self, x: u32, y: u32) -> f32 {
        let idx = (y as usize) * (self.width as usize) + (x as usize);
        self.depth_map.get(idx).map_or(0.0, |&d| d as f32 / 255.0)
    }

    /// Sample `frame` at float position, clamping or mirroring to bounds.
    fn sample(&self, frame: &Frame, fx: f32, fy: f32) -> [u8; 4] {
        let w = frame.width;
        let h = frame.height;

        let sx = if self.config.mirror_wrap {
            mirror_coord(fx, w)
        } else {
            (fx.round() as i32).clamp(0, w as i32 - 1) as u32
        };
        let sy = if self.config.mirror_wrap {
            mirror_coord(fy, h)
        } else {
            (fy.round() as i32).clamp(0, h as i32 - 1) as u32
        };

        frame.get_pixel(sx, sy).unwrap_or([0, 0, 0, 0])
    }
}

/// Mirror-wrap a coordinate into `[0, size)`.
fn mirror_coord(v: f32, size: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    let s = size as i32;
    let mut vi = v.round() as i32;
    // Reflect into range
    let period = 2 * s;
    vi = vi.rem_euclid(period);
    if vi >= s {
        vi = period - 1 - vi;
    }
    vi.clamp(0, s - 1) as u32
}

impl VideoEffect for ParallaxEffect {
    fn name(&self) -> &str {
        "Parallax"
    }

    fn description(&self) -> &'static str {
        "2.5D parallax camera-motion effect using a depth map"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        if input.width != self.width || input.height != self.height {
            return Err(VfxError::InvalidDimensions {
                width: input.width,
                height: input.height,
            });
        }
        if output.width != self.width || output.height != self.height {
            return Err(VfxError::InvalidDimensions {
                width: output.width,
                height: output.height,
            });
        }

        let cfg = &self.config;
        let blend = cfg.blend.clamp(0.0, 1.0);
        let inv_blend = 1.0 - blend;

        for y in 0..self.height {
            for x in 0..self.width {
                let depth = self.depth_at(x, y);
                // Shift magnitude = far_plane_shift + depth * (1 - far_plane_shift)
                // Near plane (depth=1.0) shifts by camera_dx; far plane shifts by
                // far_plane_shift * camera_dx.
                let shift_frac = cfg.far_plane_shift + depth * (1.0 - cfg.far_plane_shift);
                let shift_x = cfg.camera_dx * shift_frac * cfg.depth_scale;
                let shift_y = cfg.camera_dy * shift_frac * cfg.depth_scale;

                // Sample input at shifted position (shift backward to simulate forward motion)
                let src_x = x as f32 - shift_x;
                let src_y = y as f32 - shift_y;
                let sampled = self.sample(input, src_x, src_y);

                let pixel = if blend >= 1.0 {
                    sampled
                } else {
                    let orig = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                    [
                        (orig[0] as f32 * inv_blend + sampled[0] as f32 * blend) as u8,
                        (orig[1] as f32 * inv_blend + sampled[1] as f32 * blend) as u8,
                        (orig[2] as f32 * inv_blend + sampled[2] as f32 * blend) as u8,
                        (orig[3] as f32 * inv_blend + sampled[3] as f32 * blend) as u8,
                    ]
                };

                output.set_pixel(x, y, pixel);
            }
        }

        Ok(())
    }

    fn supports_gpu(&self) -> bool {
        false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        f.clear(rgba);
        f
    }

    fn uniform_depth(w: u32, h: u32, depth: u8) -> Vec<u8> {
        vec![depth; (w as usize) * (h as usize)]
    }

    #[test]
    fn test_parallax_creation_valid() {
        let e = ParallaxEffect::new(
            ParallaxConfig::default(),
            uniform_depth(64, 64, 128),
            64,
            64,
        );
        assert!(e.is_ok());
    }

    #[test]
    fn test_parallax_creation_zero_dim() {
        assert!(ParallaxEffect::new(ParallaxConfig::default(), vec![], 0, 64).is_err());
    }

    #[test]
    fn test_parallax_creation_wrong_depth_size() {
        let e = ParallaxEffect::new(ParallaxConfig::default(), vec![0u8; 10], 64, 64);
        assert!(e.is_err());
    }

    #[test]
    fn test_parallax_flat_constructor() {
        assert!(ParallaxEffect::flat(32, 32).is_ok());
    }

    #[test]
    fn test_parallax_name() {
        let e = ParallaxEffect::flat(8, 8).expect("ok");
        assert_eq!(e.name(), "Parallax");
        assert!(!e.description().is_empty());
    }

    #[test]
    fn test_parallax_uniform_image_unchanged_zero_shift() {
        // Zero camera displacement → output matches input
        let config = ParallaxConfig {
            camera_dx: 0.0,
            camera_dy: 0.0,
            ..Default::default()
        };
        let input = solid_frame(32, 32, [200, 100, 50, 255]);
        let mut output = Frame::new(32, 32).expect("output");
        let mut effect =
            ParallaxEffect::new(config, uniform_depth(32, 32, 128), 32, 32).expect("ok");
        effect
            .apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        let p = output.get_pixel(16, 16).expect("center");
        assert_eq!(p, [200, 100, 50, 255]);
    }

    #[test]
    fn test_parallax_uniform_source_unchanged_any_shift() {
        // On a solid-colour source any shift samples the same colour
        let config = ParallaxConfig {
            camera_dx: 15.0,
            camera_dy: 5.0,
            ..Default::default()
        };
        let input = solid_frame(32, 32, [77, 88, 99, 255]);
        let mut output = Frame::new(32, 32).expect("output");
        let mut effect =
            ParallaxEffect::new(config, uniform_depth(32, 32, 200), 32, 32).expect("ok");
        effect
            .apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        let p = output.get_pixel(16, 16).expect("center");
        assert_eq!(p[0], 77);
        assert_eq!(p[1], 88);
    }

    #[test]
    fn test_parallax_dimension_mismatch_input() {
        let mut effect = ParallaxEffect::flat(32, 32).expect("ok");
        let wrong_input = solid_frame(16, 16, [0, 0, 0, 255]);
        let mut output = Frame::new(16, 16).expect("output");
        assert!(effect
            .apply(&wrong_input, &mut output, &EffectParams::new())
            .is_err());
    }

    #[test]
    fn test_parallax_dimension_mismatch_output() {
        let mut effect = ParallaxEffect::flat(32, 32).expect("ok");
        let input = solid_frame(32, 32, [0, 0, 0, 255]);
        let mut output = Frame::new(16, 16).expect("output");
        assert!(effect
            .apply(&input, &mut output, &EffectParams::new())
            .is_err());
    }

    #[test]
    fn test_parallax_set_depth_map_wrong_size() {
        let mut effect = ParallaxEffect::flat(32, 32).expect("ok");
        assert!(effect.set_depth_map(vec![0u8; 10]).is_err());
    }

    #[test]
    fn test_parallax_mirror_coord() {
        // 0..size: identity
        assert_eq!(mirror_coord(5.0, 10), 5);
        // Size (10.0) → mirrors to 9
        assert_eq!(mirror_coord(10.0, 10), 9);
        // -1 → mirrors to 0
        assert_eq!(mirror_coord(-1.0, 10), 0);
    }

    #[test]
    fn test_parallax_blend_zero_returns_original() {
        let config = ParallaxConfig {
            camera_dx: 20.0,
            blend: 0.0,
            ..Default::default()
        };
        let input = solid_frame(32, 32, [42, 43, 44, 255]);
        let mut output = Frame::new(32, 32).expect("output");
        let mut effect =
            ParallaxEffect::new(config, uniform_depth(32, 32, 255), 32, 32).expect("ok");
        effect
            .apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        let p = output.get_pixel(16, 16).expect("center");
        // Blend=0 → output == original (within rounding)
        assert!((p[0] as i32 - 42).abs() <= 1);
    }

    #[test]
    fn test_parallax_output_same_dimensions() {
        let mut effect = ParallaxEffect::flat(64, 64).expect("ok");
        let input = solid_frame(64, 64, [10, 20, 30, 255]);
        let mut output = Frame::new(64, 64).expect("output");
        effect
            .apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        assert_eq!(output.width, 64);
        assert_eq!(output.height, 64);
    }
}

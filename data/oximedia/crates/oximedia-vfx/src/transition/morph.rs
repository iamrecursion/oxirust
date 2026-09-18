//! Mesh-based morph transition between two frames.
//!
//! Uses a grid of control points to interpolate spatial warping between
//! source and destination frames, creating organic morphing transitions.

use crate::{EffectParams, Frame, TransitionEffect, VfxResult};
use serde::{Deserialize, Serialize};

/// Configuration for the morph transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphConfig {
    /// Number of horizontal grid subdivisions.
    pub grid_cols: usize,
    /// Number of vertical grid subdivisions.
    pub grid_rows: usize,
    /// Warp strength multiplier (0.0 = dissolve only, 1.0 = full warp).
    pub warp_strength: f32,
    /// Whether to apply cross-dissolve in addition to warping.
    pub cross_dissolve: bool,
}

impl Default for MorphConfig {
    fn default() -> Self {
        Self {
            grid_cols: 8,
            grid_rows: 8,
            warp_strength: 0.5,
            cross_dissolve: true,
        }
    }
}

/// A control point pair for the morph mesh.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MorphControlPoint {
    /// Source position X (normalised 0-1).
    pub src_x: f32,
    /// Source position Y (normalised 0-1).
    pub src_y: f32,
    /// Destination position X (normalised 0-1).
    pub dst_x: f32,
    /// Destination position Y (normalised 0-1).
    pub dst_y: f32,
}

impl MorphControlPoint {
    /// Create a new control point at a uniform grid position (no warp).
    #[must_use]
    pub fn identity(x: f32, y: f32) -> Self {
        Self {
            src_x: x,
            src_y: y,
            dst_x: x,
            dst_y: y,
        }
    }

    /// Interpolate the position at progress `t` (0=source, 1=destination).
    #[must_use]
    pub fn interpolate(&self, t: f32) -> (f32, f32) {
        let t = t.clamp(0.0, 1.0);
        (
            self.src_x + (self.dst_x - self.src_x) * t,
            self.src_y + (self.dst_y - self.src_y) * t,
        )
    }

    /// Displacement magnitude from source to destination.
    #[must_use]
    pub fn displacement(&self) -> f32 {
        let dx = self.dst_x - self.src_x;
        let dy = self.dst_y - self.src_y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Mesh-based morph transition.
///
/// Creates a grid of control points and bilinearly interpolates between
/// source and destination frame positions to produce an organic morph effect.
pub struct MorphTransition {
    config: MorphConfig,
    control_points: Vec<MorphControlPoint>,
}

impl MorphTransition {
    /// Create a new morph transition with default configuration.
    #[must_use]
    pub fn new() -> Self {
        let config = MorphConfig::default();
        let points = Self::build_grid(&config);
        Self {
            config,
            control_points: points,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: MorphConfig) -> Self {
        let points = Self::build_grid(&config);
        Self {
            config,
            control_points: points,
        }
    }

    /// Set warp strength.
    #[must_use]
    pub fn with_warp_strength(mut self, strength: f32) -> Self {
        self.config.warp_strength = strength.clamp(0.0, 2.0);
        self
    }

    /// Set a control point displacement at grid position (col, row).
    ///
    /// Returns `true` if the point was found and updated.
    pub fn set_control_point(&mut self, col: usize, row: usize, dst_x: f32, dst_y: f32) -> bool {
        let cols = self.config.grid_cols + 1;
        if col > self.config.grid_cols || row > self.config.grid_rows {
            return false;
        }
        let idx = row * cols + col;
        if idx < self.control_points.len() {
            self.control_points[idx].dst_x = dst_x;
            self.control_points[idx].dst_y = dst_y;
            true
        } else {
            false
        }
    }

    /// Get reference to control points.
    #[must_use]
    pub fn control_points(&self) -> &[MorphControlPoint] {
        &self.control_points
    }

    /// Build the uniform grid of control points.
    fn build_grid(config: &MorphConfig) -> Vec<MorphControlPoint> {
        let cols = config.grid_cols.max(1) + 1;
        let rows = config.grid_rows.max(1) + 1;
        let mut points = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            let y = r as f32 / (rows - 1).max(1) as f32;
            for c in 0..cols {
                let x = c as f32 / (cols - 1).max(1) as f32;
                points.push(MorphControlPoint::identity(x, y));
            }
        }
        points
    }

    /// Sample a pixel from a frame using bilinear interpolation at normalised coords.
    fn sample_bilinear(frame: &Frame, nx: f32, ny: f32) -> [u8; 4] {
        let fx = nx * (frame.width as f32 - 1.0);
        let fy = ny * (frame.height as f32 - 1.0);

        let x0 = (fx.floor() as i32).clamp(0, frame.width as i32 - 1) as u32;
        let y0 = (fy.floor() as i32).clamp(0, frame.height as i32 - 1) as u32;
        let x1 = (x0 + 1).min(frame.width - 1);
        let y1 = (y0 + 1).min(frame.height - 1);

        let tx = fx - fx.floor();
        let ty = fy - fy.floor();

        let p00 = frame.get_pixel(x0, y0).unwrap_or([0, 0, 0, 0]);
        let p10 = frame.get_pixel(x1, y0).unwrap_or([0, 0, 0, 0]);
        let p01 = frame.get_pixel(x0, y1).unwrap_or([0, 0, 0, 0]);
        let p11 = frame.get_pixel(x1, y1).unwrap_or([0, 0, 0, 0]);

        let mut result = [0u8; 4];
        for ch in 0..4 {
            let top = p00[ch] as f32 * (1.0 - tx) + p10[ch] as f32 * tx;
            let bottom = p01[ch] as f32 * (1.0 - tx) + p11[ch] as f32 * tx;
            result[ch] = (top * (1.0 - ty) + bottom * ty).clamp(0.0, 255.0) as u8;
        }
        result
    }

    /// Compute the warped sample position for a given pixel using the control grid.
    ///
    /// Uses bilinear interpolation within the grid cell that contains the pixel.
    fn warp_position(&self, nx: f32, ny: f32, progress: f32, reverse: bool) -> (f32, f32) {
        let cols = self.config.grid_cols.max(1);
        let rows = self.config.grid_rows.max(1);
        let pts_per_row = cols + 1;

        // Find grid cell
        let gx = (nx * cols as f32).min((cols - 1) as f32).max(0.0);
        let gy = (ny * rows as f32).min((rows - 1) as f32).max(0.0);
        let cx = gx.floor() as usize;
        let cy = gy.floor() as usize;
        let tx = gx - gx.floor();
        let ty = gy - gy.floor();

        // Four corners of the grid cell
        let i00 = cy * pts_per_row + cx;
        let i10 = cy * pts_per_row + (cx + 1).min(cols);
        let i01 = (cy + 1).min(rows) * pts_per_row + cx;
        let i11 = (cy + 1).min(rows) * pts_per_row + (cx + 1).min(cols);

        let t = if reverse { 1.0 - progress } else { progress };
        let strength = self.config.warp_strength;

        // Interpolate control point positions
        let get_pos = |idx: usize| -> (f32, f32) {
            if idx < self.control_points.len() {
                let cp = &self.control_points[idx];
                let (ix, iy) = cp.interpolate(t * strength);
                (ix, iy)
            } else {
                (nx, ny)
            }
        };

        let (x00, y00) = get_pos(i00);
        let (x10, y10) = get_pos(i10);
        let (x01, y01) = get_pos(i01);
        let (x11, y11) = get_pos(i11);

        // Bilinear interpolation of the warped positions
        let top_x = x00 * (1.0 - tx) + x10 * tx;
        let top_y = y00 * (1.0 - tx) + y10 * tx;
        let bot_x = x01 * (1.0 - tx) + x11 * tx;
        let bot_y = y01 * (1.0 - tx) + y11 * tx;

        let wx = (top_x * (1.0 - ty) + bot_x * ty).clamp(0.0, 1.0);
        let wy = (top_y * (1.0 - ty) + bot_y * ty).clamp(0.0, 1.0);

        (wx, wy)
    }
}

impl Default for MorphTransition {
    fn default() -> Self {
        Self::new()
    }
}

impl TransitionEffect for MorphTransition {
    fn name(&self) -> &str {
        "MorphTransition"
    }

    fn description(&self) -> &'static str {
        "Mesh-based morph transition with control point interpolation"
    }

    fn apply(
        &mut self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()> {
        let progress = params.progress.clamp(0.0, 1.0);
        let w = output.width;
        let h = output.height;

        if w == 0 || h == 0 {
            return Ok(());
        }

        for y in 0..h {
            let ny = y as f32 / (h - 1).max(1) as f32;
            for x in 0..w {
                let nx = x as f32 / (w - 1).max(1) as f32;

                // Warp source position (forward warp)
                let (src_nx, src_ny) = self.warp_position(nx, ny, progress, false);
                let from_pixel = Self::sample_bilinear(from, src_nx, src_ny);

                // Warp destination position (reverse warp)
                let (dst_nx, dst_ny) = self.warp_position(nx, ny, progress, true);
                let to_pixel = Self::sample_bilinear(to, dst_nx, dst_ny);

                // Cross-dissolve between warped frames
                let t = if self.config.cross_dissolve {
                    progress
                } else {
                    if progress < 0.5 {
                        0.0
                    } else {
                        1.0
                    }
                };
                let inv_t = 1.0 - t;

                let pixel = [
                    (from_pixel[0] as f32 * inv_t + to_pixel[0] as f32 * t) as u8,
                    (from_pixel[1] as f32 * inv_t + to_pixel[1] as f32 * t) as u8,
                    (from_pixel[2] as f32 * inv_t + to_pixel[2] as f32 * t) as u8,
                    (from_pixel[3] as f32 * inv_t + to_pixel[3] as f32 * t) as u8,
                ];

                output.set_pixel(x, y, pixel);
            }
        }

        Ok(())
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red_frame(w: u32, h: u32) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        f.clear([255, 0, 0, 255]);
        f
    }

    fn blue_frame(w: u32, h: u32) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        f.clear([0, 0, 255, 255]);
        f
    }

    #[test]
    fn test_morph_control_point_identity() {
        let cp = MorphControlPoint::identity(0.5, 0.5);
        assert_eq!(cp.displacement(), 0.0);
    }

    #[test]
    fn test_morph_control_point_interpolate() {
        let cp = MorphControlPoint {
            src_x: 0.0,
            src_y: 0.0,
            dst_x: 1.0,
            dst_y: 1.0,
        };
        let (x, y) = cp.interpolate(0.5);
        assert!((x - 0.5).abs() < 0.01);
        assert!((y - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_morph_control_point_displacement() {
        let cp = MorphControlPoint {
            src_x: 0.0,
            src_y: 0.0,
            dst_x: 3.0,
            dst_y: 4.0,
        };
        assert!((cp.displacement() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_morph_transition_creation() {
        let morph = MorphTransition::new();
        assert!(!morph.control_points().is_empty());
        let expected_count = (morph.config.grid_cols + 1) * (morph.config.grid_rows + 1);
        assert_eq!(morph.control_points().len(), expected_count);
    }

    #[test]
    fn test_morph_transition_set_control_point() {
        let mut morph = MorphTransition::new();
        assert!(morph.set_control_point(0, 0, 0.1, 0.1));
        assert!(!morph.set_control_point(999, 999, 0.0, 0.0));
    }

    #[test]
    fn test_morph_transition_apply_start() {
        let mut morph = MorphTransition::new();
        let from = red_frame(32, 32);
        let to = blue_frame(32, 32);
        let mut output = Frame::new(32, 32).expect("frame");
        let params = EffectParams::new().with_progress(0.0);
        morph
            .apply(&from, &to, &mut output, &params)
            .expect("apply");

        let center = output.get_pixel(16, 16).expect("pixel");
        // At progress=0 should be mostly from (red)
        assert!(center[0] > 200, "red={}", center[0]);
    }

    #[test]
    fn test_morph_transition_apply_end() {
        let mut morph = MorphTransition::new();
        let from = red_frame(32, 32);
        let to = blue_frame(32, 32);
        let mut output = Frame::new(32, 32).expect("frame");
        let params = EffectParams::new().with_progress(1.0);
        morph
            .apply(&from, &to, &mut output, &params)
            .expect("apply");

        let center = output.get_pixel(16, 16).expect("pixel");
        // At progress=1 should be mostly to (blue)
        assert!(center[2] > 200, "blue={}", center[2]);
    }

    #[test]
    fn test_morph_transition_apply_mid() {
        let mut morph = MorphTransition::new();
        let from = red_frame(32, 32);
        let to = blue_frame(32, 32);
        let mut output = Frame::new(32, 32).expect("frame");
        let params = EffectParams::new().with_progress(0.5);
        morph
            .apply(&from, &to, &mut output, &params)
            .expect("apply");

        let center = output.get_pixel(16, 16).expect("pixel");
        // Mid-blend: both channels should be present
        assert!(center[0] > 50 && center[0] < 200);
        assert!(center[2] > 50 && center[2] < 200);
    }

    #[test]
    fn test_morph_transition_name() {
        let morph = MorphTransition::new();
        assert_eq!(morph.name(), "MorphTransition");
        assert!(!morph.description().is_empty());
    }

    #[test]
    fn test_morph_with_warp_strength() {
        let morph = MorphTransition::new().with_warp_strength(1.5);
        assert!((morph.config.warp_strength - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_morph_custom_config() {
        let config = MorphConfig {
            grid_cols: 4,
            grid_rows: 4,
            warp_strength: 0.8,
            cross_dissolve: false,
        };
        let morph = MorphTransition::with_config(config);
        assert_eq!(morph.config.grid_cols, 4);
        assert!(!morph.config.cross_dissolve);
    }

    #[test]
    fn test_sample_bilinear_corners() {
        let frame = red_frame(10, 10);
        let p = MorphTransition::sample_bilinear(&frame, 0.0, 0.0);
        assert_eq!(p, [255, 0, 0, 255]);
        let p2 = MorphTransition::sample_bilinear(&frame, 1.0, 1.0);
        assert_eq!(p2, [255, 0, 0, 255]);
    }

    #[test]
    fn test_morph_no_cross_dissolve() {
        let config = MorphConfig {
            cross_dissolve: false,
            ..Default::default()
        };
        let mut morph = MorphTransition::with_config(config);
        let from = red_frame(16, 16);
        let to = blue_frame(16, 16);
        let mut output = Frame::new(16, 16).expect("frame");
        let params = EffectParams::new().with_progress(0.25);
        morph
            .apply(&from, &to, &mut output, &params)
            .expect("apply");
        // At 0.25, should be fully from (red) since no dissolve and < 0.5
        let center = output.get_pixel(8, 8).expect("pixel");
        assert!(center[0] > 200);
    }
}

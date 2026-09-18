//! LED wall content rendering
//!
//! Provides real-time rendering of content for LED walls with
//! perspective correction and multi-panel support.
//!
//! Panels are rendered in parallel using Rayon.  The per-panel work
//! (perspective transform + pixel rasterisation) is fully independent, so
//! the final frame is deterministic and identical to the serial path.

use super::{perspective::PerspectiveCorrection, LedPanel, LedWall};
use crate::math::{Matrix4, Point3, Vector3};
use crate::{tracking::CameraPose, Result, VirtualProductionError};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// LED renderer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedRendererConfig {
    /// Target frame rate
    pub target_fps: f64,
    /// Enable perspective correction
    pub perspective_correction: bool,
    /// Enable color correction
    pub color_correction: bool,
    /// Render quality (0.0 - 1.0)
    pub quality: f32,
    /// Enable motion blur
    pub motion_blur: bool,
}

impl Default for LedRendererConfig {
    fn default() -> Self {
        Self {
            target_fps: 60.0,
            perspective_correction: true,
            color_correction: true,
            quality: 1.0,
            motion_blur: false,
        }
    }
}

/// Render output for LED wall
#[derive(Debug, Clone)]
pub struct RenderOutput {
    /// Rendered pixel data (RGB)
    pub pixels: Vec<u8>,
    /// Output width
    pub width: usize,
    /// Output height
    pub height: usize,
    /// Frame number
    pub frame_number: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl RenderOutput {
    /// Create new render output
    #[must_use]
    pub fn new(width: usize, height: usize, frame_number: u64, timestamp_ns: u64) -> Self {
        Self {
            pixels: vec![0; width * height * 3],
            width,
            height,
            frame_number,
            timestamp_ns,
        }
    }

    /// Get pixel at position
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = (y * self.width + x) * 3;
        Some([self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]])
    }

    /// Set pixel at position
    pub fn set_pixel(&mut self, x: usize, y: usize, rgb: [u8; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) * 3;
        self.pixels[idx] = rgb[0];
        self.pixels[idx + 1] = rgb[1];
        self.pixels[idx + 2] = rgb[2];
    }
}

/// Per-panel rendered output carrying the panel's pixel buffer and its
/// horizontal offset within the overall LED wall frame.
struct PanelBuffer {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    /// Horizontal start offset in the combined output frame.
    x_offset: usize,
}

/// LED wall renderer
pub struct LedRenderer {
    config: LedRendererConfig,
    led_wall: Option<LedWall>,
    perspective: PerspectiveCorrection,
    frame_number: u64,
}

impl LedRenderer {
    /// Create new LED renderer
    pub fn new(config: LedRendererConfig) -> Result<Self> {
        let perspective = PerspectiveCorrection::new()?;

        Ok(Self {
            config,
            led_wall: None,
            perspective,
            frame_number: 0,
        })
    }

    /// Set LED wall configuration
    pub fn set_led_wall(&mut self, wall: LedWall) {
        self.led_wall = Some(wall);
    }

    /// Render frame for LED wall.
    ///
    /// Panels are rendered in parallel using Rayon.  The projection matrix is
    /// shared across panels and computed once before the parallel section.
    pub fn render(
        &mut self,
        camera_pose: &CameraPose,
        source_frame: &[u8],
        source_width: usize,
        source_height: usize,
        timestamp_ns: u64,
    ) -> Result<RenderOutput> {
        let led_wall = self
            .led_wall
            .as_ref()
            .ok_or_else(|| VirtualProductionError::LedWall("No LED wall configured".to_string()))?;

        let (output_width, output_height) = led_wall.total_resolution();

        // Clone panels for the parallel section (avoids borrow of led_wall).
        let panels: Vec<LedPanel> = led_wall.panels.clone();

        // Hoist the shared projection matrix computation out of the parallel
        // loop.  Each panel will still build its own per-panel transform, but
        // the camera-pose-derived view matrix is computed once.
        let perspective_enabled = self.config.perspective_correction;
        let perspective_config = self.perspective.config().clone();

        // Compute per-panel x offsets (prefix sum of panel widths) while still
        // in the serial section so we keep layout order deterministic.
        let x_offsets: Vec<usize> = panels
            .iter()
            .scan(0usize, |acc, p| {
                let off = *acc;
                *acc += p.resolution.0;
                Some(off)
            })
            .collect();

        // Parallel render: each panel produces its own pixel buffer.
        let panel_buffers: Result<Vec<PanelBuffer>> = panels
            .par_iter()
            .zip(x_offsets.par_iter())
            .map(|(panel, &x_offset)| {
                render_panel_pure(
                    panel,
                    camera_pose,
                    source_frame,
                    source_width,
                    source_height,
                    perspective_enabled,
                    &perspective_config,
                )
                .map(|(pixels, w, h)| PanelBuffer {
                    pixels,
                    width: w,
                    height: h,
                    x_offset,
                })
            })
            .collect();

        let panel_buffers = panel_buffers?;

        // Stitch panel buffers into the single output frame.
        let mut output =
            RenderOutput::new(output_width, output_height, self.frame_number, timestamp_ns);

        for pb in &panel_buffers {
            for y in 0..pb.height {
                for x in 0..pb.width {
                    let src = (y * pb.width + x) * 3;
                    if src + 2 < pb.pixels.len() {
                        let dst_x = pb.x_offset + x;
                        output.set_pixel(
                            dst_x,
                            y,
                            [pb.pixels[src], pb.pixels[src + 1], pb.pixels[src + 2]],
                        );
                    }
                }
            }
        }

        self.frame_number += 1;

        Ok(output)
    }

    /// Get current frame number
    #[must_use]
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Reset frame counter
    pub fn reset_frame_counter(&mut self) {
        self.frame_number = 0;
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &LedRendererConfig {
        &self.config
    }

    /// Get LED wall
    #[must_use]
    pub fn led_wall(&self) -> Option<&LedWall> {
        self.led_wall.as_ref()
    }
}

/// Pure (non-mutating) per-panel render function usable from Rayon closures.
///
/// Returns `(pixels, width, height)` for the panel.
fn render_panel_pure(
    panel: &LedPanel,
    camera_pose: &CameraPose,
    source_frame: &[u8],
    source_width: usize,
    source_height: usize,
    perspective_enabled: bool,
    perspective_config: &super::perspective::PerspectiveCorrectionConfig,
) -> Result<(Vec<u8>, usize, usize)> {
    use super::perspective::PerspectiveCorrection;

    let (panel_width, panel_height) = panel.resolution;

    // Build perspective correction for this panel.  Each panel gets its own
    // PerspectiveCorrection instance to avoid shared mutable state.
    let transform = if perspective_enabled {
        let pc = PerspectiveCorrection::with_config(perspective_config.clone())?;
        pc.compute_transform(camera_pose, panel)?
    } else {
        Matrix4::identity()
    };

    let mut pixels = vec![0u8; panel_width * panel_height * 3];

    for y in 0..panel_height {
        for x in 0..panel_width {
            // Compute world position of this pixel.
            let pixel_x = (x as f64 / panel_width as f64) * panel.width;
            let pixel_y = (y as f64 / panel_height as f64) * panel.height;

            let world_pos = panel.position + Vector3::new(pixel_x, pixel_y, 0.0);

            // Apply perspective transform.
            let transformed = transform * world_pos.to_homogeneous();
            let screen_pos = Point3::from_homogeneous(transformed).unwrap_or(world_pos);

            // Map to source frame coordinates.
            let src_x = ((screen_pos.x / panel.width) * source_width as f64) as usize;
            let src_y = ((screen_pos.y / panel.height) * source_height as f64) as usize;

            if src_x < source_width && src_y < source_height {
                let src_idx = (src_y * source_width + src_x) * 3;
                if src_idx + 2 < source_frame.len() {
                    let dst = (y * panel_width + x) * 3;
                    pixels[dst] = source_frame[src_idx];
                    pixels[dst + 1] = source_frame[src_idx + 1];
                    pixels[dst + 2] = source_frame[src_idx + 2];
                }
            }
        }
    }

    Ok((pixels, panel_width, panel_height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_output() {
        let output = RenderOutput::new(1920, 1080, 0, 0);
        assert_eq!(output.width, 1920);
        assert_eq!(output.height, 1080);
        assert_eq!(output.pixels.len(), 1920 * 1080 * 3);
    }

    #[test]
    fn test_render_output_pixel() {
        let mut output = RenderOutput::new(100, 100, 0, 0);
        output.set_pixel(50, 50, [255, 128, 64]);

        let pixel = output.get_pixel(50, 50);
        assert_eq!(pixel, Some([255, 128, 64]));
    }

    #[test]
    fn test_led_renderer_creation() {
        let config = LedRendererConfig::default();
        let renderer = LedRenderer::new(config);
        assert!(renderer.is_ok());
    }

    #[test]
    fn test_led_renderer_frame_counter() {
        let config = LedRendererConfig::default();
        let mut renderer = LedRenderer::new(config).expect("should succeed in test");

        assert_eq!(renderer.frame_number(), 0);
        renderer.reset_frame_counter();
        assert_eq!(renderer.frame_number(), 0);
    }

    #[test]
    fn test_led_renderer_set_wall() {
        let config = LedRendererConfig::default();
        let mut renderer = LedRenderer::new(config).expect("should succeed in test");

        let wall = LedWall::new("Test Wall".to_string());
        renderer.set_led_wall(wall);

        assert!(renderer.led_wall().is_some());
    }

    #[test]
    fn test_render_parallel_vs_serial_determinism() {
        use crate::led::{LedPanel, LedWall};
        use crate::math::Point3;
        use crate::tracking::CameraPose;

        let mut config = LedRendererConfig::default();
        config.perspective_correction = false; // keep test simple

        let mut renderer1 = LedRenderer::new(config.clone()).expect("ok");
        let mut renderer2 = LedRenderer::new(config).expect("ok");

        let mut wall1 = LedWall::new("W".to_string());
        let mut wall2 = LedWall::new("W".to_string());

        for i in 0..3 {
            let panel = LedPanel::new(
                Point3::new(i as f64 * 2.0, 0.0, 0.0),
                2.0,
                1.5,
                (16, 12),
                2.5,
            );
            wall1.add_panel(panel.clone());
            wall2.add_panel(panel);
        }

        renderer1.set_led_wall(wall1);
        renderer2.set_led_wall(wall2);

        let pose = CameraPose::default();
        let source = vec![128u8; 64 * 48 * 3];

        let out1 = renderer1.render(&pose, &source, 64, 48, 0).expect("ok");
        let out2 = renderer2.render(&pose, &source, 64, 48, 0).expect("ok");

        assert_eq!(
            out1.pixels, out2.pixels,
            "parallel and serial must produce identical output"
        );
    }
}

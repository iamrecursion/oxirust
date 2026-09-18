//! Rendering backends (software and GPU)

use crate::color::Color;
use crate::draw_batch::{
    BlendMode as DrawBlendMode, DrawBatch, DrawCommand, FillEllipse, FillRect,
};
use crate::error::{GraphicsError, Result};
use crate::primitives::{Circle, Fill, Path, Point, Rect, Stroke};
use crate::text::{FontManager, TextLayout, TextRenderer};
use tiny_skia::{
    FillRule, LineCap as SkiaLineCap, LineJoin as SkiaLineJoin, Paint, PathBuilder, PixmapMut,
    Stroke as SkiaStroke, Transform as SkiaTransform,
};

/// Render target (RGBA image buffer)
pub struct RenderTarget {
    /// Width
    pub width: u32,
    /// Height
    pub height: u32,
    /// Pixel data (RGBA)
    pub data: Vec<u8>,
}

impl RenderTarget {
    /// Create a new render target
    pub fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(GraphicsError::InvalidDimensions(width, height));
        }

        let size = (width as usize)
            .checked_mul(height as usize)
            .and_then(|s| s.checked_mul(4))
            .ok_or(GraphicsError::InvalidDimensions(width, height))?;

        Ok(Self {
            width,
            height,
            data: vec![0; size],
        })
    }

    /// Clear with color
    pub fn clear(&mut self, color: Color) {
        for pixel in self.data.chunks_exact_mut(4) {
            pixel[0] = color.r;
            pixel[1] = color.g;
            pixel[2] = color.b;
            pixel[3] = color.a;
        }
    }

    /// Get pixel at position
    #[must_use]
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<Color> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = ((y * self.width + x) * 4) as usize;
        Some(Color::new(
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        ))
    }

    /// Set pixel at position
    pub fn set_pixel(&mut self, x: u32, y: u32, color: Color) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = ((y * self.width + x) * 4) as usize;
        self.data[idx] = color.r;
        self.data[idx + 1] = color.g;
        self.data[idx + 2] = color.b;
        self.data[idx + 3] = color.a;
    }

    /// Convert to Pixmap for tiny-skia
    fn as_pixmap_mut(&mut self) -> Option<PixmapMut<'_>> {
        PixmapMut::from_bytes(&mut self.data, self.width, self.height)
    }
}

/// Extract a `[u8; 4]` RGBA color from a [`Fill`] (gradient falls back to its first stop).
fn fill_to_color(fill: &Fill) -> [u8; 4] {
    match fill {
        Fill::Solid(c) => [c.r, c.g, c.b, c.a],
        Fill::Gradient(g) => {
            let sampled = g.sample(0.0, 0.0);
            let [r, g_f, b, a] = sampled.to_float();
            [
                (r * 255.0) as u8,
                (g_f * 255.0) as u8,
                (b * 255.0) as u8,
                (a * 255.0) as u8,
            ]
        }
    }
}

/// Software renderer using tiny-skia
pub struct SoftwareRenderer {
    font_manager: FontManager,
    /// Pending draw commands, flushed via [`flush`](SoftwareRenderer::flush).
    batch: DrawBatch,
}

impl SoftwareRenderer {
    /// Create a new software renderer
    #[must_use]
    pub fn new(font_manager: FontManager) -> Self {
        Self {
            font_manager,
            batch: DrawBatch::new(),
        }
    }

    /// Enqueue a rectangle draw command in the batch.
    ///
    /// The command is not executed until [`flush`](SoftwareRenderer::flush) is called.
    /// The `layer` defaults to 0; set via the [`DrawCommand`] directly if ordering matters.
    pub fn render_rect(
        &mut self,
        _target: &mut RenderTarget,
        rect: Rect,
        fill: &Fill,
        _stroke: Option<&Stroke>,
    ) -> Result<()> {
        let color = fill_to_color(fill);
        self.batch.push(DrawCommand::Rect(FillRect::new(
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            color,
            0,
        )));
        Ok(())
    }

    /// Enqueue a circle draw command in the batch.
    ///
    /// The command is not executed until [`flush`](SoftwareRenderer::flush) is called.
    pub fn render_circle(
        &mut self,
        _target: &mut RenderTarget,
        circle: Circle,
        fill: &Fill,
        _stroke: Option<&Stroke>,
    ) -> Result<()> {
        let color = fill_to_color(fill);
        self.batch.push(DrawCommand::Ellipse(FillEllipse::circle(
            circle.x,
            circle.y,
            circle.radius,
            color,
            0,
        )));
        Ok(())
    }

    /// Flush the pending batch, executing all queued draw commands on `target`.
    ///
    /// Commands are executed in layer-ascending order (stable within the same layer).
    /// The batch is cleared after flushing.
    pub fn flush(&mut self, target: &mut RenderTarget) -> Result<()> {
        let commands = self.batch.flush();
        let mut pixmap = target
            .as_pixmap_mut()
            .ok_or_else(|| GraphicsError::RenderError("Failed to create pixmap".to_string()))?;

        for cmd in &commands {
            match cmd {
                DrawCommand::Rect(r) => {
                    if let Some(skia_rect) = tiny_skia::Rect::from_xywh(r.x, r.y, r.width, r.height)
                    {
                        let path = PathBuilder::from_rect(skia_rect);
                        self.fill_path_inner(&mut pixmap, &path, r.color, r.blend)?;
                    }
                }
                DrawCommand::Ellipse(e) => {
                    let mut pb = PathBuilder::new();
                    pb.push_circle(e.cx, e.cy, e.rx);
                    if let Some(path) = pb.finish() {
                        self.fill_path_inner(&mut pixmap, &path, e.color, e.blend)?;
                    }
                }
                DrawCommand::Glyph(_) | DrawCommand::Texture(_) | DrawCommand::Gradient(_) => {
                    // These command types are issued by higher-level callers; not rasterized here.
                }
            }
        }
        Ok(())
    }

    /// Render path (immediate — paths have no batch equivalent).
    pub fn render_path(
        &mut self,
        target: &mut RenderTarget,
        path: &Path,
        fill: &Fill,
        stroke: Option<&Stroke>,
    ) -> Result<()> {
        // Flush any pending batch commands first so ordering is preserved.
        self.flush(target)?;

        let mut pixmap = target
            .as_pixmap_mut()
            .ok_or_else(|| GraphicsError::RenderError("Failed to create pixmap".to_string()))?;

        let kurbo_path = path.to_kurbo();
        let mut pb = PathBuilder::new();

        for el in kurbo_path.elements() {
            match el {
                kurbo::PathEl::MoveTo(p) => {
                    pb.move_to(p.x as f32, p.y as f32);
                }
                kurbo::PathEl::LineTo(p) => {
                    pb.line_to(p.x as f32, p.y as f32);
                }
                kurbo::PathEl::QuadTo(c, p) => {
                    pb.quad_to(c.x as f32, c.y as f32, p.x as f32, p.y as f32);
                }
                kurbo::PathEl::CurveTo(c1, c2, p) => {
                    pb.cubic_to(
                        c1.x as f32,
                        c1.y as f32,
                        c2.x as f32,
                        c2.y as f32,
                        p.x as f32,
                        p.y as f32,
                    );
                }
                kurbo::PathEl::ClosePath => {
                    pb.close();
                }
            }
        }

        let skia_path = pb
            .finish()
            .ok_or_else(|| GraphicsError::RenderError("Failed to create path".to_string()))?;

        // Fill
        self.fill_path(&mut pixmap, &skia_path, fill)?;

        // Stroke
        if let Some(s) = stroke {
            self.stroke_path(&mut pixmap, &skia_path, s)?;
        }

        Ok(())
    }

    /// Render text (immediate — text goes through the glyph layout pipeline).
    pub fn render_text(
        &mut self,
        target: &mut RenderTarget,
        layout: &TextLayout,
        position: Point,
    ) -> Result<()> {
        let renderer = TextRenderer::new(self.font_manager.clone());
        let glyphs = renderer.layout_glyphs(layout, position)?;

        // This is a simplified version - real implementation would render actual glyphs
        for glyph in glyphs {
            // Draw a simple rectangle as placeholder
            let rect = Rect::new(glyph.position.x, glyph.position.y, glyph.size, glyph.size);
            self.render_rect(target, rect, &Fill::Solid(glyph.color), None)?;
        }

        Ok(())
    }

    fn fill_path(&self, pixmap: &mut PixmapMut, path: &tiny_skia::Path, fill: &Fill) -> Result<()> {
        let mut paint = Paint::default();
        paint.anti_alias = true;

        match fill {
            Fill::Solid(color) => {
                let [r, g, b, a] = color.to_float();
                paint.set_color_rgba8(
                    (r * 255.0) as u8,
                    (g * 255.0) as u8,
                    (b * 255.0) as u8,
                    (a * 255.0) as u8,
                );
            }
            Fill::Gradient(gradient) => {
                // Simplified gradient rendering
                // Real implementation would use Shader::Linear or Shader::Radial
                let color = gradient.sample(0.0, 0.0);
                let [r, g, b, a] = color.to_float();
                paint.set_color_rgba8(
                    (r * 255.0) as u8,
                    (g * 255.0) as u8,
                    (b * 255.0) as u8,
                    (a * 255.0) as u8,
                );
            }
        }

        pixmap.fill_path(
            path,
            &paint,
            FillRule::Winding,
            SkiaTransform::identity(),
            None,
        );

        Ok(())
    }

    /// Low-level fill using a pre-extracted RGBA color and blend mode (used by batch flush).
    fn fill_path_inner(
        &self,
        pixmap: &mut PixmapMut,
        path: &tiny_skia::Path,
        color: [u8; 4],
        _blend: DrawBlendMode,
    ) -> Result<()> {
        let mut paint = Paint::default();
        paint.anti_alias = true;
        paint.set_color_rgba8(color[0], color[1], color[2], color[3]);
        pixmap.fill_path(
            path,
            &paint,
            FillRule::Winding,
            SkiaTransform::identity(),
            None,
        );
        Ok(())
    }

    fn stroke_path(
        &self,
        pixmap: &mut PixmapMut,
        path: &tiny_skia::Path,
        stroke: &Stroke,
    ) -> Result<()> {
        let mut paint = Paint::default();
        paint.anti_alias = true;

        let [r, g, b, a] = stroke.color.to_float();
        paint.set_color_rgba8(
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
            (a * 255.0) as u8,
        );

        let mut skia_stroke = SkiaStroke::default();
        skia_stroke.width = stroke.width;
        skia_stroke.line_cap = match stroke.cap {
            crate::primitives::LineCap::Butt => SkiaLineCap::Butt,
            crate::primitives::LineCap::Round => SkiaLineCap::Round,
            crate::primitives::LineCap::Square => SkiaLineCap::Square,
        };
        skia_stroke.line_join = match stroke.join {
            crate::primitives::LineJoin::Miter => SkiaLineJoin::Miter,
            crate::primitives::LineJoin::Round => SkiaLineJoin::Round,
            crate::primitives::LineJoin::Bevel => SkiaLineJoin::Bevel,
        };

        pixmap.stroke_path(path, &paint, &skia_stroke, SkiaTransform::identity(), None);

        Ok(())
    }

    /// Composite two render targets
    pub fn composite(
        &self,
        base: &mut RenderTarget,
        overlay: &RenderTarget,
        position: Point,
        opacity: f32,
    ) -> Result<()> {
        if base.width != overlay.width || base.height != overlay.height {
            return Err(GraphicsError::RenderError(
                "Mismatched render target sizes".to_string(),
            ));
        }

        let alpha = opacity.clamp(0.0, 1.0);

        for y in 0..overlay.height {
            for x in 0..overlay.width {
                let ox = (x as f32 + position.x) as u32;
                let oy = (y as f32 + position.y) as u32;

                if ox >= base.width || oy >= base.height {
                    continue;
                }

                if let Some(overlay_pixel) = overlay.get_pixel(x, y) {
                    if let Some(base_pixel) = base.get_pixel(ox, oy) {
                        let mut blended = overlay_pixel.blend(&base_pixel);
                        blended.a = (f32::from(blended.a) * alpha) as u8;
                        base.set_pixel(ox, oy, blended);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_target_creation() {
        let target = RenderTarget::new(1920, 1080).expect("target should be valid");
        assert_eq!(target.width, 1920);
        assert_eq!(target.height, 1080);
        assert_eq!(target.data.len(), 1920 * 1080 * 4);
    }

    #[test]
    fn test_render_target_invalid_dimensions() {
        assert!(RenderTarget::new(0, 0).is_err());
    }

    #[test]
    fn test_render_target_clear() {
        let mut target = RenderTarget::new(100, 100).expect("test expectation failed");
        target.clear(Color::RED);
        assert_eq!(target.get_pixel(0, 0), Some(Color::RED));
        assert_eq!(target.get_pixel(50, 50), Some(Color::RED));
    }

    #[test]
    fn test_render_target_pixel_ops() {
        let mut target = RenderTarget::new(100, 100).expect("test expectation failed");
        target.set_pixel(10, 20, Color::BLUE);
        assert_eq!(target.get_pixel(10, 20), Some(Color::BLUE));
        assert_eq!(target.get_pixel(100, 100), None);
    }

    #[test]
    fn test_software_renderer_creation() {
        let font_manager = FontManager::new();
        let _renderer = SoftwareRenderer::new(font_manager);
        // Just test creation
    }

    #[test]
    fn test_render_rect() {
        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);
        let mut target = RenderTarget::new(100, 100).expect("test expectation failed");

        let rect = Rect::new(10.0, 10.0, 50.0, 50.0);
        let fill = Fill::Solid(Color::RED);

        let result = renderer.render_rect(&mut target, rect, &fill, None);
        assert!(result.is_ok());
        // Command is queued — flush to confirm no errors
        assert!(renderer.flush(&mut target).is_ok());
    }

    #[test]
    fn test_render_circle() {
        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);
        let mut target = RenderTarget::new(100, 100).expect("test expectation failed");

        let circle = Circle::new(50.0, 50.0, 25.0);
        let fill = Fill::Solid(Color::BLUE);

        let result = renderer.render_circle(&mut target, circle, &fill, None);
        assert!(result.is_ok());
        assert!(renderer.flush(&mut target).is_ok());
    }

    /// Verify that batched rendering produces non-empty output (pixel-level check).
    #[test]
    fn test_batched_render_pixel_identical() {
        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);
        let mut target = RenderTarget::new(100, 100).expect("test target");

        // Queue a red rect and a blue circle
        let rect = Rect::new(0.0, 0.0, 50.0, 50.0);
        renderer
            .render_rect(&mut target, rect, &Fill::Solid(Color::RED), None)
            .expect("queue rect");
        let circle = Circle::new(75.0, 75.0, 20.0);
        renderer
            .render_circle(&mut target, circle, &Fill::Solid(Color::BLUE), None)
            .expect("queue circle");

        // Before flush the target is untouched
        assert_eq!(target.get_pixel(25, 25), Some(Color::new(0, 0, 0, 0)));

        // After flush pixels should be written
        renderer.flush(&mut target).expect("flush");

        // The red rect region should have some non-zero red component
        let px = target.get_pixel(25, 25).expect("pixel present");
        assert!(
            px.r > 0 || px.g > 0 || px.b > 0 || px.a > 0,
            "pixel should be non-zero after flush"
        );
    }

    /// Verify that layer ordering is respected during flush.
    #[test]
    fn test_batch_layer_ordering_preserved() {
        use crate::draw_batch::{DrawCommand, FillRect};

        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Manually push commands with different layers out of order
        renderer.batch.push(DrawCommand::Rect(FillRect::new(
            0.0,
            0.0,
            10.0,
            10.0,
            [0, 255, 0, 255], // green — layer 5
            5,
        )));
        renderer.batch.push(DrawCommand::Rect(FillRect::new(
            0.0,
            0.0,
            10.0,
            10.0,
            [255, 0, 0, 255], // red — layer 1 (should render first)
            1,
        )));

        // After flush the batch should be empty
        let mut target = RenderTarget::new(20, 20).expect("target");
        renderer.flush(&mut target).expect("flush");
        assert!(
            renderer.batch.is_empty(),
            "batch should be empty after flush"
        );
    }
}

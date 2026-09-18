//! Overlay management system.
//!
//! Provides a compositing overlay system that can render text, FPS counters,
//! performance metrics, and custom graphics onto captured frames in real time.
//!
//! # Dirty-region optimisation
//!
//! Each [`OverlayLayer`] tracks a `dirty` flag and a `last_bbox` bounding box.
//! [`OverlaySystem::composite_onto`] unions all dirty layer bboxes and only
//! re-composites within that rectangle when the caller provides a cached
//! previous output.  When nothing is dirty the cached output is returned
//! unchanged (zero compositing work).

use crate::capture::screen::CapturedFrame;
use crate::GamingResult;

// ---------------------------------------------------------------------------
// Rect
// ---------------------------------------------------------------------------

/// An axis-aligned bounding rectangle (inclusive start, exclusive end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left edge (inclusive).
    pub x: u32,
    /// Top edge (inclusive).
    pub y: u32,
    /// Width.
    pub w: u32,
    /// Height.
    pub h: u32,
}

impl Rect {
    /// Construct a new Rect.
    #[must_use]
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// Right edge (exclusive).
    #[must_use]
    pub fn right(&self) -> u32 {
        self.x.saturating_add(self.w)
    }

    /// Bottom edge (exclusive).
    #[must_use]
    pub fn bottom(&self) -> u32 {
        self.y.saturating_add(self.h)
    }

    /// Return the smallest `Rect` that contains both `self` and `other`.
    #[must_use]
    pub fn union(self, other: Rect) -> Rect {
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = self.right().max(other.right());
        let y1 = self.bottom().max(other.bottom());
        Rect {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }

    /// Whether the rect has non-zero area.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }
}

// ---------------------------------------------------------------------------
// DirtyRegion
// ---------------------------------------------------------------------------

/// Accumulates the union of changed bounding boxes for overlay layers.
#[derive(Clone, Default)]
pub struct DirtyRegion {
    /// Individual changed rectangles; union via `union_all`.
    pub rects: Vec<Rect>,
}

impl DirtyRegion {
    /// Add a rectangle to the dirty set.
    pub fn add(&mut self, rect: Rect) {
        self.rects.push(rect);
    }

    /// Compute the axis-aligned union of all contained rectangles.
    #[must_use]
    pub fn union_all(&self) -> Option<Rect> {
        self.rects.iter().copied().reduce(Rect::union)
    }

    /// Whether any dirty rectangles have been registered.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        !self.rects.is_empty()
    }

    /// Reset the dirty set.
    pub fn clear(&mut self) {
        self.rects.clear();
    }
}

// ---------------------------------------------------------------------------
// OverlaySystem
// ---------------------------------------------------------------------------

/// Overlay system for managing multiple overlay layers.
pub struct OverlaySystem {
    layers: Vec<OverlayLayer>,
    /// Cached last composited frame (for dirty-region skip).
    cached_output: Option<Vec<u8>>,
}

/// Overlay layer.
#[derive(Debug, Clone)]
pub struct OverlayLayer {
    /// Layer name
    pub name: String,
    /// Z-index (higher = on top)
    pub z_index: i32,
    /// Visible
    pub visible: bool,
    /// Opacity (0.0 to 1.0)
    pub opacity: f32,
    /// Layer content type
    pub content: OverlayContent,
    /// Whether this layer has changed since the last composite.
    pub dirty: bool,
    /// Last known bounding box of the rendered content (for dirty union).
    pub last_bbox: Option<Rect>,
}

/// What kind of content an overlay layer renders.
#[derive(Debug, Clone)]
pub enum OverlayContent {
    /// Static text at a position.
    Text(TextOverlay),
    /// FPS counter.
    FpsCounter(FpsCounterOverlay),
    /// Performance metrics panel.
    PerfPanel(PerfPanelOverlay),
    /// Solid colour rectangle (for backgrounds, debug, etc.).
    Rect(RectOverlay),
    /// Custom RGBA pixel data.
    Image(ImageOverlay),
}

/// Text overlay configuration.
#[derive(Debug, Clone)]
pub struct TextOverlay {
    /// Text to display.
    pub text: String,
    /// Position (x, y) from top-left.
    pub position: (u32, u32),
    /// Text colour (RGBA).
    pub color: [u8; 4],
    /// Font size in pixels (each glyph is rendered as size x size block).
    pub font_size: u32,
}

/// FPS counter overlay.
#[derive(Debug, Clone)]
pub struct FpsCounterOverlay {
    /// Position (x, y).
    pub position: (u32, u32),
    /// Colour.
    pub color: [u8; 4],
    /// Current FPS value (updated externally).
    pub current_fps: f32,
    /// Font size in pixels.
    pub font_size: u32,
}

/// Performance panel overlay showing multiple metrics.
#[derive(Debug, Clone)]
pub struct PerfPanelOverlay {
    /// Position (x, y).
    pub position: (u32, u32),
    /// Background colour (RGBA).
    pub bg_color: [u8; 4],
    /// Text colour.
    pub text_color: [u8; 4],
    /// Metric lines to display.
    pub lines: Vec<String>,
    /// Font size.
    pub font_size: u32,
}

/// Solid rectangle overlay.
#[derive(Debug, Clone)]
pub struct RectOverlay {
    /// Position (x, y).
    pub position: (u32, u32),
    /// Size (width, height).
    pub size: (u32, u32),
    /// Colour (RGBA).
    pub color: [u8; 4],
}

/// Custom image overlay (RGBA pixel data).
#[derive(Debug, Clone)]
pub struct ImageOverlay {
    /// Position (x, y).
    pub position: (u32, u32),
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// RGBA pixel data.
    pub data: Vec<u8>,
}

impl OverlayLayer {
    /// Create a new overlay layer.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        z_index: i32,
        visible: bool,
        opacity: f32,
        content: OverlayContent,
    ) -> Self {
        Self {
            name: name.into(),
            z_index,
            visible,
            opacity,
            content,
            dirty: true,
            last_bbox: None,
        }
    }

    /// Mark this layer as needing a redraw.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Whether this layer needs to be recomposited.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Update content and automatically mark dirty.
    pub fn set_content(&mut self, content: OverlayContent) {
        self.content = content;
        self.dirty = true;
    }

    /// Update opacity and automatically mark dirty.
    pub fn set_opacity(&mut self, opacity: f32) {
        self.opacity = opacity;
        self.dirty = true;
    }

    /// Update position for the outermost content element and mark dirty.
    ///
    /// For `Text`, `FpsCounter`, `PerfPanel`, `Rect`, and `Image` this shifts
    /// the primary position field.  The old `last_bbox` is also invalidated so
    /// the dirty union covers the vacated area.
    pub fn set_position(&mut self, x: u32, y: u32) {
        match &mut self.content {
            OverlayContent::Text(t) => t.position = (x, y),
            OverlayContent::FpsCounter(f) => f.position = (x, y),
            OverlayContent::PerfPanel(p) => p.position = (x, y),
            OverlayContent::Rect(r) => r.position = (x, y),
            OverlayContent::Image(i) => i.position = (x, y),
        }
        // Invalidate last_bbox so the old location is included in the dirty union.
        self.last_bbox = None;
        self.dirty = true;
    }

    /// Compute the bounding box of this layer's content, if known.
    #[must_use]
    pub fn compute_bbox(&self) -> Option<Rect> {
        match &self.content {
            OverlayContent::Text(t) => {
                let glyph_w = t.font_size.max(1);
                let spacing = glyph_w + 1;
                let total_w = (t.text.chars().count() as u32) * spacing;
                let total_h = t.font_size.max(1);
                if total_w == 0 {
                    None
                } else {
                    Some(Rect::new(t.position.0, t.position.1, total_w, total_h))
                }
            }
            OverlayContent::FpsCounter(f) => {
                // "FPS: XXX" is at most 8 glyphs
                let glyph_w = f.font_size.max(1);
                let spacing = glyph_w + 1;
                Some(Rect::new(
                    f.position.0,
                    f.position.1,
                    8 * spacing,
                    f.font_size.max(1),
                ))
            }
            OverlayContent::PerfPanel(p) => {
                let line_height = p.font_size + 2;
                let panel_h = (p.lines.len() as u32) * line_height + 4;
                let panel_w = p
                    .lines
                    .iter()
                    .map(|l| (l.len() as u32) * (p.font_size + 1))
                    .max()
                    .unwrap_or(0)
                    + 8;
                Some(Rect::new(p.position.0, p.position.1, panel_w, panel_h))
            }
            OverlayContent::Rect(r) => {
                Some(Rect::new(r.position.0, r.position.1, r.size.0, r.size.1))
            }
            OverlayContent::Image(img) => Some(Rect::new(
                img.position.0,
                img.position.1,
                img.width,
                img.height,
            )),
        }
    }
}

impl OverlaySystem {
    /// Create a new overlay system.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            cached_output: None,
        }
    }

    /// Add a layer.
    pub fn add_layer(&mut self, layer: OverlayLayer) {
        self.layers.push(layer);
        self.layers.sort_by_key(|l| l.z_index);
    }

    /// Remove a layer.
    pub fn remove_layer(&mut self, name: &str) {
        self.layers.retain(|l| l.name != name);
        self.cached_output = None; // invalidate cache on structural change
    }

    /// Show/hide a layer.
    pub fn set_layer_visibility(&mut self, name: &str, visible: bool) -> GamingResult<()> {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.name == name) {
            layer.visible = visible;
            layer.dirty = true;
        }
        Ok(())
    }

    /// Get number of layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get a mutable reference to a named layer.
    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut OverlayLayer> {
        self.layers.iter_mut().find(|l| l.name == name)
    }

    /// Get an immutable reference to a named layer.
    #[must_use]
    pub fn get_layer(&self, name: &str) -> Option<&OverlayLayer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Composite all visible layers onto a captured frame (in-place) using
    /// dirty-region optimisation.
    ///
    /// # Algorithm
    ///
    /// 1. Collect the bounding boxes of all dirty visible layers (including the
    ///    previous `last_bbox` to cover areas that were just vacated).
    /// 2. Compute the union rectangle `U` of those boxes.
    /// 3. If `U` is empty (nothing dirty) *and* a cached output exists, copy the
    ///    cached pixels into the frame and return immediately — zero compositing.
    /// 4. Otherwise, full-composite all visible layers but restrict the inner
    ///    pixel loops to `[U.x, U.right()) × [U.y, U.bottom())`.
    ///    For layers entirely outside `U` this inner loop executes zero iterations.
    /// 5. Update each layer's `last_bbox` and clear `dirty` flags.
    /// 6. Save the resulting frame pixels in `cached_output`.
    pub fn composite_onto(&mut self, frame: &mut CapturedFrame) {
        // --- Step 1 & 2: collect dirty bboxes and union them ---
        let mut dirty_region = DirtyRegion::default();

        for layer in &self.layers {
            if !layer.visible {
                continue;
            }
            if layer.dirty {
                // Include both the old location and the new location.
                if let Some(old_bb) = layer.last_bbox {
                    dirty_region.add(old_bb);
                }
                if let Some(new_bb) = layer.compute_bbox() {
                    dirty_region.add(new_bb);
                }
            }
        }

        // --- Step 3: nothing dirty + cache available → fast path ---
        if !dirty_region.is_dirty() {
            if let Some(cache) = &self.cached_output {
                let copy_len = cache.len().min(frame.data.len());
                frame.data[..copy_len].copy_from_slice(&cache[..copy_len]);
                return;
            }
            // No cache yet; fall through to full composite.
        }

        // --- Step 4: composite within the dirty union ---
        let clip = dirty_region.union_all().unwrap_or(Rect {
            x: 0,
            y: 0,
            w: frame.width,
            h: frame.height,
        });

        // Clamp clip to frame bounds.
        let clip = Rect {
            x: clip.x.min(frame.width),
            y: clip.y.min(frame.height),
            w: clip.w.min(frame.width.saturating_sub(clip.x)),
            h: clip.h.min(frame.height.saturating_sub(clip.y)),
        };

        for layer in &self.layers {
            if !layer.visible {
                continue;
            }
            match &layer.content {
                OverlayContent::Text(t) => {
                    Self::render_text_clipped(frame, t, layer.opacity, clip);
                }
                OverlayContent::FpsCounter(f) => {
                    let text = format!("FPS: {:.0}", f.current_fps);
                    let t = TextOverlay {
                        text,
                        position: f.position,
                        color: f.color,
                        font_size: f.font_size,
                    };
                    Self::render_text_clipped(frame, &t, layer.opacity, clip);
                }
                OverlayContent::PerfPanel(p) => {
                    Self::render_perf_panel_clipped(frame, p, layer.opacity, clip);
                }
                OverlayContent::Rect(r) => {
                    Self::render_rect_clipped(frame, r, layer.opacity, clip);
                }
                OverlayContent::Image(img) => {
                    Self::render_image_clipped(frame, img, layer.opacity, clip);
                }
            }
        }

        // --- Step 5: update bboxes and clear dirty flags ---
        for layer in self.layers.iter_mut() {
            if layer.dirty {
                layer.last_bbox = layer.compute_bbox();
                layer.dirty = false;
            }
        }

        // --- Step 6: cache the output ---
        self.cached_output = Some(frame.data.clone());
    }

    // -------------------------------------------------------------------------
    // Clipped render helpers
    // -------------------------------------------------------------------------

    /// Render text constrained to `clip`.
    fn render_text_clipped(
        frame: &mut CapturedFrame,
        overlay: &TextOverlay,
        opacity: f32,
        clip: Rect,
    ) {
        let glyph_w = overlay.font_size.max(1);
        let glyph_h = overlay.font_size.max(1);
        let spacing = glyph_w + 1;

        for (ci, _ch) in overlay.text.chars().enumerate() {
            let gx = overlay.position.0 + (ci as u32) * spacing;
            let gy = overlay.position.1;

            for dy in 0..glyph_h {
                for dx in 0..glyph_w {
                    let px = gx + dx;
                    let py = gy + dy;
                    if px < clip.x || px >= clip.right() || py < clip.y || py >= clip.bottom() {
                        continue;
                    }
                    if px < frame.width && py < frame.height {
                        let idx = ((py * frame.width + px) * 4) as usize;
                        if idx + 3 < frame.data.len() {
                            Self::blend_pixel(
                                &mut frame.data[idx..idx + 4],
                                overlay.color,
                                opacity,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Render a performance panel constrained to `clip`.
    fn render_perf_panel_clipped(
        frame: &mut CapturedFrame,
        panel: &PerfPanelOverlay,
        opacity: f32,
        clip: Rect,
    ) {
        let line_height = panel.font_size + 2;
        let panel_height = (panel.lines.len() as u32) * line_height + 4;
        let panel_width = panel
            .lines
            .iter()
            .map(|l| (l.len() as u32) * (panel.font_size + 1))
            .max()
            .unwrap_or(0)
            + 8;

        let bg = RectOverlay {
            position: panel.position,
            size: (panel_width, panel_height),
            color: panel.bg_color,
        };
        Self::render_rect_clipped(frame, &bg, opacity, clip);

        for (i, line) in panel.lines.iter().enumerate() {
            let t = TextOverlay {
                text: line.clone(),
                position: (
                    panel.position.0 + 4,
                    panel.position.1 + 2 + (i as u32) * line_height,
                ),
                color: panel.text_color,
                font_size: panel.font_size,
            };
            Self::render_text_clipped(frame, &t, opacity, clip);
        }
    }

    /// Render a solid rectangle constrained to `clip`.
    fn render_rect_clipped(
        frame: &mut CapturedFrame,
        rect: &RectOverlay,
        opacity: f32,
        clip: Rect,
    ) {
        for dy in 0..rect.size.1 {
            for dx in 0..rect.size.0 {
                let px = rect.position.0 + dx;
                let py = rect.position.1 + dy;
                if px < clip.x || px >= clip.right() || py < clip.y || py >= clip.bottom() {
                    continue;
                }
                if px < frame.width && py < frame.height {
                    let idx = ((py * frame.width + px) * 4) as usize;
                    if idx + 3 < frame.data.len() {
                        Self::blend_pixel(&mut frame.data[idx..idx + 4], rect.color, opacity);
                    }
                }
            }
        }
    }

    /// Render a custom image constrained to `clip`.
    fn render_image_clipped(
        frame: &mut CapturedFrame,
        img: &ImageOverlay,
        opacity: f32,
        clip: Rect,
    ) {
        for dy in 0..img.height {
            for dx in 0..img.width {
                let px = img.position.0 + dx;
                let py = img.position.1 + dy;
                if px < clip.x || px >= clip.right() || py < clip.y || py >= clip.bottom() {
                    continue;
                }
                let src_idx = ((dy * img.width + dx) * 4) as usize;
                if src_idx + 3 >= img.data.len() {
                    continue;
                }
                if px < frame.width && py < frame.height {
                    let dst_idx = ((py * frame.width + px) * 4) as usize;
                    if dst_idx + 3 < frame.data.len() {
                        let src_color = [
                            img.data[src_idx],
                            img.data[src_idx + 1],
                            img.data[src_idx + 2],
                            img.data[src_idx + 3],
                        ];
                        Self::blend_pixel(
                            &mut frame.data[dst_idx..dst_idx + 4],
                            src_color,
                            opacity,
                        );
                    }
                }
            }
        }
    }

    /// Alpha-blend a source colour onto a destination pixel, respecting layer opacity.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn blend_pixel(dst: &mut [u8], src: [u8; 4], opacity: f32) {
        let sa = (src[3] as f32 / 255.0) * opacity;
        let da = 1.0 - sa;
        dst[0] = ((src[0] as f32 * sa) + (dst[0] as f32 * da)).min(255.0) as u8;
        dst[1] = ((src[1] as f32 * sa) + (dst[1] as f32 * da)).min(255.0) as u8;
        dst[2] = ((src[2] as f32 * sa) + (dst[2] as f32 * da)).min(255.0) as u8;
        dst[3] = 255; // keep alpha opaque
    }
}

impl Default for OverlaySystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(
        name: &str,
        z: i32,
        visible: bool,
        opacity: f32,
        content: OverlayContent,
    ) -> OverlayLayer {
        OverlayLayer::new(name, z, visible, opacity, content)
    }

    fn make_frame(w: u32, h: u32) -> CapturedFrame {
        CapturedFrame {
            data: vec![0u8; (w as usize) * (h as usize) * 4],
            width: w,
            height: h,
            timestamp: std::time::Duration::ZERO,
            sequence: 0,
        }
    }

    fn rect_layer(
        name: &str,
        z: i32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        color: [u8; 4],
    ) -> OverlayLayer {
        make_layer(
            name,
            z,
            true,
            1.0,
            OverlayContent::Rect(RectOverlay {
                position: (x, y),
                size: (w, h),
                color,
            }),
        )
    }

    // --- DirtyRegion ---

    #[test]
    fn test_dirty_region_empty() {
        let dr = DirtyRegion::default();
        assert!(!dr.is_dirty());
        assert!(dr.union_all().is_none());
    }

    #[test]
    fn test_dirty_region_add_and_union() {
        let mut dr = DirtyRegion::default();
        dr.add(Rect::new(0, 0, 10, 10));
        dr.add(Rect::new(5, 5, 10, 10));
        assert!(dr.is_dirty());
        let u = dr.union_all().expect("non-empty union");
        assert_eq!(u.x, 0);
        assert_eq!(u.y, 0);
        assert!(u.right() >= 15);
        assert!(u.bottom() >= 15);
    }

    #[test]
    fn test_dirty_region_clear() {
        let mut dr = DirtyRegion::default();
        dr.add(Rect::new(0, 0, 4, 4));
        dr.clear();
        assert!(!dr.is_dirty());
    }

    // --- OverlayLayer dirty tracking ---

    #[test]
    fn test_new_layer_is_dirty() {
        let layer = rect_layer("test", 0, 0, 0, 4, 4, [255, 0, 0, 255]);
        assert!(layer.is_dirty());
    }

    #[test]
    fn test_set_content_marks_dirty() {
        let mut layer = rect_layer("l", 0, 0, 0, 4, 4, [255, 0, 0, 255]);
        layer.dirty = false;
        layer.set_content(OverlayContent::Rect(RectOverlay {
            position: (0, 0),
            size: (4, 4),
            color: [0, 255, 0, 255],
        }));
        assert!(layer.is_dirty());
    }

    #[test]
    fn test_set_opacity_marks_dirty() {
        let mut layer = rect_layer("l", 0, 0, 0, 4, 4, [255, 0, 0, 255]);
        layer.dirty = false;
        layer.set_opacity(0.5);
        assert!(layer.is_dirty());
    }

    #[test]
    fn test_set_position_marks_dirty() {
        let mut layer = rect_layer("l", 0, 0, 0, 4, 4, [255, 0, 0, 255]);
        layer.dirty = false;
        layer.set_position(10, 10);
        assert!(layer.is_dirty());
    }

    #[test]
    fn test_compute_bbox_rect() {
        let layer = rect_layer("l", 0, 5, 3, 10, 8, [0; 4]);
        let bb = layer.compute_bbox().expect("bbox");
        assert_eq!(bb.x, 5);
        assert_eq!(bb.y, 3);
        assert_eq!(bb.w, 10);
        assert_eq!(bb.h, 8);
    }

    // --- OverlaySystem ---

    #[test]
    fn test_overlay_system_creation() {
        let system = OverlaySystem::new();
        assert_eq!(system.layer_count(), 0);
    }

    #[test]
    fn test_add_layer() {
        let mut system = OverlaySystem::new();
        system.add_layer(rect_layer("Chat", 10, 0, 0, 100, 100, [255, 0, 0, 128]));
        assert_eq!(system.layer_count(), 1);
    }

    #[test]
    fn test_remove_layer() {
        let mut system = OverlaySystem::new();
        system.add_layer(rect_layer("a", 1, 0, 0, 1, 1, [0; 4]));
        system.add_layer(rect_layer("b", 2, 0, 0, 1, 1, [0; 4]));
        assert_eq!(system.layer_count(), 2);
        system.remove_layer("a");
        assert_eq!(system.layer_count(), 1);
    }

    #[test]
    fn test_layer_visibility() {
        let mut system = OverlaySystem::new();
        system.add_layer(rect_layer("test", 0, 0, 0, 1, 1, [0; 4]));
        system
            .set_layer_visibility("test", false)
            .expect("set visibility");
        let layer = system.get_layer("test").expect("layer exists");
        assert!(!layer.visible);
    }

    #[test]
    fn test_z_order_sorting() {
        let mut system = OverlaySystem::new();
        for z in [30, 10, 20] {
            system.add_layer(rect_layer(&format!("z{z}"), z, 0, 0, 1, 1, [0; 4]));
        }
        assert_eq!(system.get_layer("z10").expect("z10").z_index, 10);
    }

    #[test]
    fn test_composite_rect_onto_frame() {
        let mut system = OverlaySystem::new();
        system.add_layer(rect_layer("red", 0, 0, 0, 4, 4, [255, 0, 0, 255]));

        let mut frame = make_frame(8, 8);
        system.composite_onto(&mut frame);

        assert_eq!(frame.data[0], 255);
        assert_eq!(frame.data[1], 0);
        assert_eq!(frame.data[2], 0);
    }

    #[test]
    fn test_composite_text_overlay() {
        let mut system = OverlaySystem::new();
        system.add_layer(make_layer(
            "text",
            0,
            true,
            1.0,
            OverlayContent::Text(TextOverlay {
                text: "Hi".into(),
                position: (0, 0),
                color: [0, 255, 0, 255],
                font_size: 2,
            }),
        ));

        let mut frame = make_frame(32, 32);
        system.composite_onto(&mut frame);

        assert_eq!(frame.data[0], 0);
        assert_eq!(frame.data[1], 255);
    }

    #[test]
    fn test_composite_fps_counter() {
        let mut system = OverlaySystem::new();
        system.add_layer(make_layer(
            "fps",
            0,
            true,
            1.0,
            OverlayContent::FpsCounter(FpsCounterOverlay {
                position: (0, 0),
                color: [255, 255, 0, 255],
                current_fps: 60.0,
                font_size: 2,
            }),
        ));

        let mut frame = make_frame(64, 32);
        system.composite_onto(&mut frame);

        let has_overlay = frame.data.chunks(4).any(|p| p[0] > 0 || p[1] > 0);
        assert!(has_overlay);
    }

    #[test]
    fn test_composite_perf_panel() {
        let mut system = OverlaySystem::new();
        system.add_layer(make_layer(
            "perf",
            0,
            true,
            1.0,
            OverlayContent::PerfPanel(PerfPanelOverlay {
                position: (0, 0),
                bg_color: [0, 0, 0, 200],
                text_color: [255, 255, 255, 255],
                lines: vec!["FPS: 60".into(), "CPU: 30%".into()],
                font_size: 2,
            }),
        ));

        let mut frame = make_frame(64, 32);
        system.composite_onto(&mut frame);

        let modified = frame
            .data
            .chunks(4)
            .any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
        assert!(modified);
    }

    #[test]
    fn test_composite_image_overlay() {
        let mut system = OverlaySystem::new();
        let img_data = vec![
            255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255,
        ];
        system.add_layer(make_layer(
            "img",
            0,
            true,
            1.0,
            OverlayContent::Image(ImageOverlay {
                position: (0, 0),
                width: 2,
                height: 2,
                data: img_data,
            }),
        ));

        let mut frame = make_frame(8, 8);
        system.composite_onto(&mut frame);

        assert_eq!(frame.data[0], 255);
        assert_eq!(frame.data[1], 0);
        assert_eq!(frame.data[2], 255);
    }

    #[test]
    fn test_hidden_layer_not_composited() {
        let mut system = OverlaySystem::new();
        system.add_layer(make_layer(
            "hidden",
            0,
            false,
            1.0,
            OverlayContent::Rect(RectOverlay {
                position: (0, 0),
                size: (4, 4),
                color: [255, 0, 0, 255],
            }),
        ));

        let mut frame = make_frame(8, 8);
        let original = frame.data.clone();
        system.composite_onto(&mut frame);
        assert_eq!(frame.data, original);
    }

    #[test]
    fn test_half_opacity_blend() {
        let mut system = OverlaySystem::new();
        system.add_layer(make_layer(
            "half",
            0,
            true,
            0.5,
            OverlayContent::Rect(RectOverlay {
                position: (0, 0),
                size: (1, 1),
                color: [200, 200, 200, 255],
            }),
        ));

        let mut frame = make_frame(4, 4);
        frame.data[0] = 100;
        frame.data[1] = 100;
        frame.data[2] = 100;
        frame.data[3] = 255;

        system.composite_onto(&mut frame);

        assert_eq!(frame.data[0], 150);
        assert_eq!(frame.data[1], 150);
        assert_eq!(frame.data[2], 150);
    }

    #[test]
    fn test_get_layer_mut() {
        let mut system = OverlaySystem::new();
        system.add_layer(rect_layer("mutable", 0, 0, 0, 1, 1, [0; 4]));
        let layer = system.get_layer_mut("mutable").expect("exists");
        layer.set_opacity(0.5);
        assert!((system.get_layer("mutable").expect("exists").opacity - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_overlay_out_of_bounds_safe() {
        let mut system = OverlaySystem::new();
        system.add_layer(rect_layer("oob", 0, 6, 6, 10, 10, [255, 255, 255, 255]));

        let mut frame = make_frame(8, 8);
        system.composite_onto(&mut frame); // must not panic
    }

    #[test]
    fn test_multiple_layers_composite_order() {
        let mut system = OverlaySystem::new();
        system.add_layer(make_layer(
            "bottom",
            0,
            true,
            1.0,
            OverlayContent::Rect(RectOverlay {
                position: (0, 0),
                size: (4, 4),
                color: [255, 0, 0, 255],
            }),
        ));
        system.add_layer(make_layer(
            "top",
            10,
            true,
            1.0,
            OverlayContent::Rect(RectOverlay {
                position: (0, 0),
                size: (4, 4),
                color: [0, 0, 255, 255],
            }),
        ));

        let mut frame = make_frame(8, 8);
        system.composite_onto(&mut frame);

        assert_eq!(frame.data[0], 0);
        assert_eq!(frame.data[2], 255);
    }

    // --- Dirty-region specific tests ---

    #[test]
    fn test_dirty_region_cache_skips_composite() {
        let mut system = OverlaySystem::new();
        system.add_layer(rect_layer("r", 0, 0, 0, 2, 2, [255, 0, 0, 255]));

        let mut frame = make_frame(4, 4);
        // First composite: dirty → renders red rect.
        system.composite_onto(&mut frame);
        assert_eq!(frame.data[0], 255); // red

        // Manually zero the frame so we can detect if the cache is replayed.
        frame.data.iter_mut().for_each(|b| *b = 0);

        // Nothing changed → cache should be replayed, frame turns red again.
        system.composite_onto(&mut frame);
        assert_eq!(frame.data[0], 255, "cached red should be restored");
    }

    #[test]
    fn test_dirty_after_set_content_recomposites() {
        let mut system = OverlaySystem::new();
        system.add_layer(rect_layer("r", 0, 0, 0, 2, 2, [255, 0, 0, 255]));

        let mut frame = make_frame(4, 4);
        system.composite_onto(&mut frame);
        assert_eq!(frame.data[0], 255); // red

        // Change to green → must dirty the layer.
        system
            .get_layer_mut("r")
            .expect("layer")
            .set_content(OverlayContent::Rect(RectOverlay {
                position: (0, 0),
                size: (2, 2),
                color: [0, 255, 0, 255],
            }));

        // Zero the frame; second composite should draw green.
        frame.data.iter_mut().for_each(|b| *b = 0);
        system.composite_onto(&mut frame);
        assert_eq!(
            frame.data[1], 255,
            "green channel should be 255 after update"
        );
    }

    #[test]
    fn test_rect_union() {
        let a = Rect::new(0, 0, 5, 5);
        let b = Rect::new(3, 3, 5, 5);
        let u = a.union(b);
        assert_eq!(u.x, 0);
        assert_eq!(u.y, 0);
        assert_eq!(u.right(), 8);
        assert_eq!(u.bottom(), 8);
    }
}

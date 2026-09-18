//! SuperSource multi-view compositor for video switchers.
//!
//! SuperSource allows displaying multiple video inputs simultaneously on a
//! single output by placing scaled, cropped, and bordered video boxes on a
//! shared canvas — similar to the Blackmagic Design ATEM SuperSource feature.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─── SuperSource box ───────────────────────────────────────────────────────────

/// A single video box within a SuperSource composition.
///
/// Coordinates are in normalised space where (0.0, 0.0) is the top-left
/// corner and (1.0, 1.0) is the bottom-right corner of the canvas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperSourceBox {
    /// Box ID (unique within a layout)
    pub id: u32,
    /// Input source ID
    pub input: u32,
    /// Horizontal centre position (0.0 – 1.0)
    pub x: f32,
    /// Vertical centre position (0.0 – 1.0)
    pub y: f32,
    /// Uniform scale factor (0.0 – 1.0, where 1.0 = full canvas)
    pub scale: f32,
    /// Crop amounts (left, right, top, bottom) in normalised units
    pub crop: (f32, f32, f32, f32),
    /// Border width in normalised units
    pub border_width: f32,
    /// Border colour as RGBA
    pub border_color: [u8; 4],
}

impl SuperSourceBox {
    /// Create a new SuperSource box at the given normalised position.
    #[must_use]
    pub fn new(id: u32, input: u32, x: f32, y: f32, scale: f32) -> Self {
        Self {
            id,
            input,
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            scale: scale.clamp(0.0, 1.0),
            crop: (0.0, 0.0, 0.0, 0.0),
            border_width: 0.0,
            border_color: [255, 255, 255, 255],
        }
    }

    /// Convert the normalised box geometry to a pixel rectangle.
    ///
    /// Returns `(x, y, width, height)` in pixels.  The returned `x`/`y` may be
    /// negative if the box is partially outside the canvas.
    #[must_use]
    pub fn to_pixel_rect(&self, canvas_w: u32, canvas_h: u32) -> (i32, i32, u32, u32) {
        let cw = canvas_w as f32;
        let ch = canvas_h as f32;

        let w = (cw * self.scale).max(1.0);
        let h = (ch * self.scale).max(1.0);

        // Centre-based positioning
        let px = (self.x * cw - w * 0.5) as i32;
        let py = (self.y * ch - h * 0.5) as i32;

        (px, py, w as u32, h as u32)
    }

    /// Set the crop amounts (left, right, top, bottom) in normalised units.
    pub fn set_crop(&mut self, left: f32, right: f32, top: f32, bottom: f32) {
        self.crop = (
            left.clamp(0.0, 1.0),
            right.clamp(0.0, 1.0),
            top.clamp(0.0, 1.0),
            bottom.clamp(0.0, 1.0),
        );
    }

    /// Enable and configure a border.
    pub fn set_border(&mut self, width: f32, color: [u8; 4]) {
        self.border_width = width.max(0.0);
        self.border_color = color;
    }
}

// ─── SuperSource layout ────────────────────────────────────────────────────────

/// A collection of video boxes and a background source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperSourceLayout {
    /// Video boxes, ordered back-to-front
    pub boxes: Vec<SuperSourceBox>,
    /// Input used as the background fill
    pub background_input: u32,
}

impl SuperSourceLayout {
    /// Create an empty layout.
    #[must_use]
    pub fn new(background_input: u32) -> Self {
        Self {
            boxes: Vec::new(),
            background_input,
        }
    }

    /// Add a box to the layout.
    pub fn add_box(&mut self, ss_box: SuperSourceBox) {
        self.boxes.push(ss_box);
    }

    /// Remove a box by ID.
    pub fn remove_box(&mut self, id: u32) {
        self.boxes.retain(|b| b.id != id);
    }

    /// Update a box using a closure.
    pub fn update_box(&mut self, id: u32, f: impl FnOnce(&mut SuperSourceBox)) {
        if let Some(b) = self.boxes.iter_mut().find(|b| b.id == id) {
            f(b);
        }
    }

    /// Get a reference to a box by ID.
    #[must_use]
    pub fn get_box(&self, id: u32) -> Option<&SuperSourceBox> {
        self.boxes.iter().find(|b| b.id == id)
    }

    /// Number of boxes in this layout.
    #[must_use]
    pub fn box_count(&self) -> usize {
        self.boxes.len()
    }

    // ─── Built-in layouts ──────────────────────────────────────────────────

    /// 2×2 four-up layout (four equal quadrants).
    #[must_use]
    pub fn layout_4up(background_input: u32, inputs: [u32; 4]) -> Self {
        let mut layout = Self::new(background_input);
        let positions = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)];
        for (i, (x, y)) in positions.iter().enumerate() {
            layout.add_box(SuperSourceBox::new(i as u32, inputs[i], *x, *y, 0.5));
        }
        layout
    }

    /// Picture-in-picture layout: one full-screen main, one small inset.
    #[must_use]
    pub fn layout_pip(background_input: u32, main_input: u32, pip_input: u32) -> Self {
        let mut layout = Self::new(background_input);
        // Main source takes the full canvas
        layout.add_box(SuperSourceBox::new(0, main_input, 0.5, 0.5, 1.0));
        // Small PIP in the bottom-right corner
        layout.add_box(SuperSourceBox::new(1, pip_input, 0.8, 0.8, 0.25));
        layout
    }

    /// Side-by-side layout: two equal panels.
    #[must_use]
    pub fn layout_side_by_side(background_input: u32, left_input: u32, right_input: u32) -> Self {
        let mut layout = Self::new(background_input);
        layout.add_box(SuperSourceBox::new(0, left_input, 0.25, 0.5, 0.5));
        layout.add_box(SuperSourceBox::new(1, right_input, 0.75, 0.5, 0.5));
        layout
    }

    /// Picture-in-picture with the inset in a specified corner (0=TL,1=TR,2=BL,3=BR).
    #[must_use]
    pub fn layout_picture_in_picture_corner(
        background_input: u32,
        main_input: u32,
        pip_input: u32,
        corner: u8,
    ) -> Self {
        let mut layout = Self::new(background_input);
        layout.add_box(SuperSourceBox::new(0, main_input, 0.5, 0.5, 1.0));

        let (px, py) = match corner {
            0 => (0.15, 0.15), // Top-left
            1 => (0.85, 0.15), // Top-right
            2 => (0.15, 0.85), // Bottom-left
            _ => (0.85, 0.85), // Bottom-right (default)
        };
        layout.add_box(SuperSourceBox::new(1, pip_input, px, py, 0.2));
        layout
    }
}

// ─── Renderer ─────────────────────────────────────────────────────────────────

/// Compositor that renders a `SuperSourceLayout` into an RGBA frame buffer.
pub struct SuperSourceRenderer;

impl SuperSourceRenderer {
    /// Create a new renderer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Render the layout onto a canvas of `width × height` pixels.
    ///
    /// `inputs` is a slice of `(input_id, rgba_data)` pairs where each
    /// `rgba_data` buffer must be exactly `width * height * 4` bytes.
    ///
    /// The returned buffer is `width * height * 4` bytes of RGBA data.
    #[must_use]
    pub fn render(
        &self,
        layout: &SuperSourceLayout,
        inputs: &[(u32, Vec<u8>)],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let pixel_count = (width * height) as usize;
        let mut canvas = vec![0u8; pixel_count * 4];

        // Fill with background first
        if let Some(bg_data) = find_input(inputs, layout.background_input) {
            blit_scaled(
                bg_data,
                width,
                height,
                &mut canvas,
                0,
                0,
                width,
                height,
                width,
            );
        }

        // Composite each box, back-to-front
        for ss_box in &layout.boxes {
            let Some(src_data) = find_input(inputs, ss_box.input) else {
                continue;
            };

            let (bx, by, bw, bh) = ss_box.to_pixel_rect(width, height);

            // Clamp to canvas bounds
            let cx = bx.max(0) as u32;
            let cy = by.max(0) as u32;
            let cw = ((bx + bw as i32).min(width as i32) - cx as i32).max(0) as u32;
            let ch = ((by + bh as i32).min(height as i32) - cy as i32).max(0) as u32;

            if cw == 0 || ch == 0 {
                continue;
            }

            blit_scaled(src_data, width, height, &mut canvas, cx, cy, cw, ch, width);

            // Draw border if enabled
            if ss_box.border_width > 0.0 {
                draw_border(
                    &mut canvas,
                    cx,
                    cy,
                    cw,
                    ch,
                    ss_box.border_width as u32 + 1,
                    ss_box.border_color,
                    width,
                );
            }
        }

        canvas
    }
}

impl Default for SuperSourceRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Find an input buffer by ID.
fn find_input(inputs: &[(u32, Vec<u8>)], id: u32) -> Option<&Vec<u8>> {
    inputs.iter().find(|(i, _)| *i == id).map(|(_, d)| d)
}

/// Nearest-neighbour blit of `src` (assumed `canvas_w × canvas_h` RGBA)
/// into `dst` at offset `(dst_x, dst_y)` with dimensions `(dst_w × dst_h)`.
fn blit_scaled(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_x: u32,
    dst_y: u32,
    dst_w: u32,
    dst_h: u32,
    canvas_w: u32,
) {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return;
    }

    for row in 0..dst_h {
        for col in 0..dst_w {
            // Source pixel via nearest-neighbour
            let sx = (col as u64 * src_w as u64 / dst_w as u64) as u32;
            let sy = (row as u64 * src_h as u64 / dst_h as u64) as u32;

            let sx = sx.min(src_w - 1);
            let sy = sy.min(src_h - 1);

            let src_idx = ((sy * src_w + sx) * 4) as usize;
            let dst_idx = (((dst_y + row) * canvas_w + dst_x + col) * 4) as usize;

            if src_idx + 3 < src.len() && dst_idx + 3 < dst.len() {
                dst[dst_idx..dst_idx + 4].copy_from_slice(&src[src_idx..src_idx + 4]);
            }
        }
    }
}

/// Draw a solid border rectangle on the canvas.
fn draw_border(
    dst: &mut [u8],
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    thickness: u32,
    color: [u8; 4],
    canvas_w: u32,
) {
    let t = thickness.min(w / 2).min(h / 2);
    for row in 0..h {
        for col in 0..w {
            let on_border = row < t || row >= h - t || col < t || col >= w - t;
            if on_border {
                let idx = (((y + row) * canvas_w + x + col) * 4) as usize;
                if idx + 3 < dst.len() {
                    dst[idx..idx + 4].copy_from_slice(&color);
                }
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_super_source_box_creation() {
        let b = SuperSourceBox::new(0, 1, 0.5, 0.5, 0.5);
        assert_eq!(b.id, 0);
        assert_eq!(b.input, 1);
        assert_eq!(b.x, 0.5);
        assert_eq!(b.y, 0.5);
        assert_eq!(b.scale, 0.5);
    }

    #[test]
    fn test_super_source_box_clamping() {
        let b = SuperSourceBox::new(0, 0, -0.5, 1.5, 2.0);
        assert_eq!(b.x, 0.0);
        assert_eq!(b.y, 1.0);
        assert_eq!(b.scale, 1.0);
    }

    #[test]
    fn test_to_pixel_rect_centre() {
        let b = SuperSourceBox::new(0, 0, 0.5, 0.5, 0.5);
        let (x, y, w, h) = b.to_pixel_rect(1920, 1080);
        assert_eq!(w, 960);
        assert_eq!(h, 540);
        assert_eq!(x, 480); // 960 - 480 = 480
        assert_eq!(y, 270); // 540 - 270 = 270
    }

    #[test]
    fn test_to_pixel_rect_full_screen() {
        let b = SuperSourceBox::new(0, 0, 0.5, 0.5, 1.0);
        let (x, y, w, h) = b.to_pixel_rect(1920, 1080);
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn test_super_source_layout_add_remove() {
        let mut layout = SuperSourceLayout::new(0);
        assert_eq!(layout.box_count(), 0);

        layout.add_box(SuperSourceBox::new(0, 1, 0.25, 0.25, 0.5));
        layout.add_box(SuperSourceBox::new(1, 2, 0.75, 0.25, 0.5));
        assert_eq!(layout.box_count(), 2);

        layout.remove_box(0);
        assert_eq!(layout.box_count(), 1);
        assert!(layout.get_box(0).is_none());
        assert!(layout.get_box(1).is_some());
    }

    #[test]
    fn test_super_source_layout_update_box() {
        let mut layout = SuperSourceLayout::new(0);
        layout.add_box(SuperSourceBox::new(0, 1, 0.5, 0.5, 0.5));

        layout.update_box(0, |b| b.input = 99);
        assert_eq!(layout.get_box(0).expect("should succeed in test").input, 99);
    }

    #[test]
    fn test_layout_4up() {
        let layout = SuperSourceLayout::layout_4up(0, [1, 2, 3, 4]);
        assert_eq!(layout.box_count(), 4);
        assert_eq!(layout.get_box(0).expect("should succeed in test").x, 0.25);
        assert_eq!(layout.get_box(1).expect("should succeed in test").x, 0.75);
    }

    #[test]
    fn test_layout_pip() {
        let layout = SuperSourceLayout::layout_pip(0, 1, 2);
        assert_eq!(layout.box_count(), 2);
        let pip = layout.get_box(1).expect("should succeed in test");
        assert_eq!(pip.input, 2);
        assert!(pip.scale < 0.5);
    }

    #[test]
    fn test_layout_side_by_side() {
        let layout = SuperSourceLayout::layout_side_by_side(0, 1, 2);
        assert_eq!(layout.box_count(), 2);
        let left = layout.get_box(0).expect("should succeed in test");
        let right = layout.get_box(1).expect("should succeed in test");
        assert!(left.x < 0.5);
        assert!(right.x > 0.5);
    }

    #[test]
    fn test_layout_pip_corner_all_corners() {
        for corner in 0..4 {
            let layout = SuperSourceLayout::layout_picture_in_picture_corner(0, 1, 2, corner);
            assert_eq!(layout.box_count(), 2);
        }
    }

    #[test]
    fn test_renderer_empty_layout() {
        let renderer = SuperSourceRenderer::new();
        let layout = SuperSourceLayout::new(0);
        let result = renderer.render(&layout, &[], 8, 8);
        assert_eq!(result.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_renderer_with_background() {
        let renderer = SuperSourceRenderer::new();
        let layout = SuperSourceLayout::new(0);
        let bg: Vec<u8> = (0..8 * 8 * 4).map(|i| (i % 256) as u8).collect();
        let inputs = vec![(0u32, bg)];
        let result = renderer.render(&layout, &inputs, 8, 8);
        assert_eq!(result.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_renderer_with_box() {
        let renderer = SuperSourceRenderer::new();
        let mut layout = SuperSourceLayout::new(0);
        layout.add_box(SuperSourceBox::new(0, 1, 0.5, 0.5, 0.5));

        let bg = vec![0u8; 16 * 16 * 4];
        // Source is all white
        let src = vec![255u8; 16 * 16 * 4];
        let inputs = vec![(0u32, bg), (1u32, src)];
        let result = renderer.render(&layout, &inputs, 16, 16);
        assert_eq!(result.len(), 16 * 16 * 4);
    }

    #[test]
    fn test_renderer_with_border() {
        let renderer = SuperSourceRenderer::new();
        let mut layout = SuperSourceLayout::new(0);
        let mut b = SuperSourceBox::new(0, 1, 0.5, 0.5, 1.0);
        b.set_border(2.0, [255, 0, 0, 255]);
        layout.add_box(b);

        let bg = vec![0u8; 16 * 16 * 4];
        let src = vec![128u8; 16 * 16 * 4];
        let inputs = vec![(0u32, bg), (1u32, src)];
        let result = renderer.render(&layout, &inputs, 16, 16);
        // Top-left pixel should be red (border)
        assert_eq!(result[0], 255); // R
        assert_eq!(result[1], 0); // G
    }
}

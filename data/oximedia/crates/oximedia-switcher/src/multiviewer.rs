//! Multi-viewer layout and rendering for video switchers.
//!
//! Displays multiple video sources in a single composite output for monitoring.

use crate::tally::TallyState;
use oximedia_codec::{Plane, VideoFrame};
use oximedia_core::PixelFormat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur with multi-viewer operations.
#[derive(Error, Debug, Clone)]
pub enum MultiviewerError {
    #[error("Invalid window ID: {0}")]
    InvalidWindowId(usize),

    #[error("Window {0} not found")]
    WindowNotFound(usize),

    #[error("Invalid layout configuration")]
    InvalidLayout,

    #[error("Rendering error: {0}")]
    RenderingError(String),
}

/// Multi-viewer layout types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MultiviewerLayout {
    /// 2x2 grid (4 windows)
    Grid2x2,
    /// 3x3 grid (9 windows)
    Grid3x3,
    /// 4x4 grid (16 windows)
    Grid4x4,
    /// Program left, 4 sources right
    ProgramLeft4x1,
    /// Program top, sources below
    ProgramTop,
    /// Custom layout
    Custom,
}

impl MultiviewerLayout {
    /// Get the number of windows for this layout.
    pub fn window_count(&self) -> usize {
        match self {
            MultiviewerLayout::Grid2x2 => 4,
            MultiviewerLayout::Grid3x3 => 9,
            MultiviewerLayout::Grid4x4 => 16,
            MultiviewerLayout::ProgramLeft4x1 => 5,
            MultiviewerLayout::ProgramTop => 5,
            MultiviewerLayout::Custom => 0, // Variable
        }
    }

    /// Get grid dimensions (columns, rows).
    pub fn grid_dimensions(&self) -> (usize, usize) {
        match self {
            MultiviewerLayout::Grid2x2 => (2, 2),
            MultiviewerLayout::Grid3x3 => (3, 3),
            MultiviewerLayout::Grid4x4 => (4, 4),
            MultiviewerLayout::ProgramLeft4x1 => (2, 2),
            MultiviewerLayout::ProgramTop => (2, 3),
            MultiviewerLayout::Custom => (0, 0),
        }
    }
}

/// Position and size of a window in normalized coordinates (0.0 - 1.0).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WindowRect {
    /// X position (0.0 = left)
    pub x: f32,
    /// Y position (0.0 = top)
    pub y: f32,
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
}

impl WindowRect {
    /// Create a new window rectangle.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            width: width.clamp(0.0, 1.0),
            height: height.clamp(0.0, 1.0),
        }
    }

    /// Full screen rectangle.
    pub fn full_screen() -> Self {
        Self::new(0.0, 0.0, 1.0, 1.0)
    }
}

/// Multi-viewer window configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiviewerWindow {
    /// Window ID
    pub id: usize,
    /// Window label
    pub label: String,
    /// Input source ID
    pub source_id: usize,
    /// Window rectangle
    pub rect: WindowRect,
    /// Show tally border
    pub show_tally: bool,
    /// Show audio meters
    pub show_audio: bool,
    /// Show label overlay
    pub show_label: bool,
}

impl MultiviewerWindow {
    /// Create a new multiviewer window.
    pub fn new(id: usize, source_id: usize, rect: WindowRect) -> Self {
        Self {
            id,
            label: format!("Input {source_id}"),
            source_id,
            rect,
            show_tally: true,
            show_audio: true,
            show_label: true,
        }
    }

    /// Set the label.
    pub fn set_label(&mut self, label: String) {
        self.label = label;
    }

    /// Set the source.
    pub fn set_source(&mut self, source_id: usize) {
        self.source_id = source_id;
    }
}

/// Multi-viewer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiviewerConfig {
    /// Layout type
    pub layout: MultiviewerLayout,
    /// Windows
    pub windows: HashMap<usize, MultiviewerWindow>,
    /// Show borders between windows
    pub show_borders: bool,
    /// Border color (R, G, B)
    pub border_color: (u8, u8, u8),
    /// Border width in pixels
    pub border_width: u32,
    /// Output canvas width in pixels
    pub canvas_width: u32,
    /// Output canvas height in pixels
    pub canvas_height: u32,
}

impl MultiviewerConfig {
    /// Create a new multiviewer configuration.
    pub fn new(layout: MultiviewerLayout) -> Self {
        Self {
            layout,
            windows: HashMap::new(),
            show_borders: true,
            border_color: (64, 64, 64),
            border_width: 2,
            canvas_width: 1920,
            canvas_height: 1080,
        }
    }

    /// Set the output canvas resolution.
    pub fn set_canvas_resolution(&mut self, width: u32, height: u32) {
        self.canvas_width = width.max(1);
        self.canvas_height = height.max(1);
    }

    /// Create a 2x2 grid layout.
    pub fn grid_2x2() -> Self {
        let mut config = Self::new(MultiviewerLayout::Grid2x2);

        // Create 2x2 grid windows
        for row in 0..2 {
            for col in 0..2 {
                let id = row * 2 + col;
                let rect = WindowRect::new(col as f32 * 0.5, row as f32 * 0.5, 0.5, 0.5);
                let window = MultiviewerWindow::new(id, id, rect);
                config.windows.insert(id, window);
            }
        }

        config
    }

    /// Create a 4x4 grid layout.
    pub fn grid_4x4() -> Self {
        let mut config = Self::new(MultiviewerLayout::Grid4x4);

        // Create 4x4 grid windows
        for row in 0..4 {
            for col in 0..4 {
                let id = row * 4 + col;
                let rect = WindowRect::new(col as f32 * 0.25, row as f32 * 0.25, 0.25, 0.25);
                let window = MultiviewerWindow::new(id, id, rect);
                config.windows.insert(id, window);
            }
        }

        config
    }

    /// Add a window.
    pub fn add_window(&mut self, window: MultiviewerWindow) {
        self.windows.insert(window.id, window);
    }

    /// Remove a window.
    pub fn remove_window(&mut self, id: usize) {
        self.windows.remove(&id);
    }

    /// Get a window.
    pub fn get_window(&self, id: usize) -> Option<&MultiviewerWindow> {
        self.windows.get(&id)
    }

    /// Get a mutable window.
    pub fn get_window_mut(&mut self, id: usize) -> Option<&mut MultiviewerWindow> {
        self.windows.get_mut(&id)
    }

    /// Get the number of windows.
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }
}

/// Multi-viewer renderer.
pub struct Multiviewer {
    config: MultiviewerConfig,
    tally_states: HashMap<usize, TallyState>,
    enabled: bool,
}

impl Multiviewer {
    /// Create a new multiviewer.
    pub fn new(config: MultiviewerConfig) -> Self {
        Self {
            config,
            tally_states: HashMap::new(),
            enabled: true,
        }
    }

    /// Create a 2x2 multiviewer.
    pub fn grid_2x2() -> Self {
        Self::new(MultiviewerConfig::grid_2x2())
    }

    /// Create a 4x4 multiviewer.
    pub fn grid_4x4() -> Self {
        Self::new(MultiviewerConfig::grid_4x4())
    }

    /// Get the configuration.
    pub fn config(&self) -> &MultiviewerConfig {
        &self.config
    }

    /// Get mutable configuration.
    pub fn config_mut(&mut self) -> &mut MultiviewerConfig {
        &mut self.config
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: MultiviewerConfig) {
        self.config = config;
    }

    /// Enable or disable the multiviewer.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the multiviewer is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Update tally states.
    pub fn update_tally(&mut self, source_id: usize, state: TallyState) {
        self.tally_states.insert(source_id, state);
    }

    /// Get tally state for a source.
    pub fn get_tally(&self, source_id: usize) -> TallyState {
        self.tally_states
            .get(&source_id)
            .copied()
            .unwrap_or(TallyState::Idle)
    }

    /// Clear all tally states.
    pub fn clear_tally(&mut self) {
        self.tally_states.clear();
    }

    /// Render the multiviewer output.
    ///
    /// Produces a single composited [`VideoFrame`] in [`PixelFormat::Rgba32`]
    /// at the configured canvas resolution. For every window in the layout the
    /// matching input frame (looked up by `source_id` in `inputs`) is
    /// nearest-neighbour scaled into the window's pixel rectangle. Tally
    /// borders, label strips, audio meters and inter-window borders are drawn
    /// according to the per-window and global configuration flags.
    ///
    /// # Errors
    ///
    /// Returns [`MultiviewerError::RenderingError`] when the configured canvas
    /// resolution is degenerate (zero width or height).
    pub fn render(
        &self,
        inputs: &HashMap<usize, VideoFrame>,
    ) -> Result<VideoFrame, MultiviewerError> {
        let canvas_w = self.config.canvas_width;
        let canvas_h = self.config.canvas_height;

        if canvas_w == 0 || canvas_h == 0 {
            return Err(MultiviewerError::RenderingError(format!(
                "degenerate canvas resolution {canvas_w}x{canvas_h}"
            )));
        }

        // 1. Allocate a blank (opaque black) RGBA canvas.
        let mut canvas = blank_rgba_canvas(canvas_w, canvas_h);

        // 2. Composite each window. `windows` is a HashMap, so iterate in a
        //    deterministic id order to keep overlapping geometry stable.
        let mut window_ids: Vec<usize> = self.config.windows.keys().copied().collect();
        window_ids.sort_unstable();

        for window_id in window_ids {
            let Some(window) = self.config.windows.get(&window_id) else {
                continue;
            };

            let (wx, wy, ww, wh) = self.window_pixel_rect(window, canvas_w, canvas_h);
            if ww == 0 || wh == 0 {
                continue;
            }

            // Clamp the window rect to the canvas so blits never overrun.
            let cx = wx.min(canvas_w);
            let cy = wy.min(canvas_h);
            let cw = ww.min(canvas_w - cx);
            let ch = wh.min(canvas_h - cy);
            if cw == 0 || ch == 0 {
                continue;
            }

            // a-c. Scale the input frame into the window rect.
            if let Some(frame) = inputs.get(&window.source_id) {
                blit_frame_scaled(frame, &mut canvas, canvas_w, cx, cy, cw, ch);
            }

            // d. Tally border (per-window flag plus a recorded tally state).
            if window.show_tally {
                let tally = self.get_tally(window.source_id);
                let (tr, tg, tb) = self.tally_border_color(tally);
                let thickness = self.config.border_width.max(1);
                draw_rect_border(
                    &mut canvas,
                    canvas_w,
                    cx,
                    cy,
                    cw,
                    ch,
                    thickness,
                    [tr, tg, tb, 255],
                );
            }

            // e. Label strip along the bottom edge of the window.
            if window.show_label {
                let strip_h = (ch / 8).clamp(1, 24).min(ch);
                let strip_y = cy + ch - strip_h;
                fill_rect(
                    &mut canvas,
                    canvas_w,
                    cx,
                    strip_y,
                    cw,
                    strip_h,
                    [0, 0, 0, 200],
                );
            }

            // f. Audio meter column along the right edge of the window.
            if window.show_audio {
                let meter_w = (cw / 16).clamp(1, 8).min(cw);
                let meter_x = cx + cw - meter_w;
                fill_rect(
                    &mut canvas,
                    canvas_w,
                    meter_x,
                    cy,
                    meter_w,
                    ch,
                    [0, 200, 0, 200],
                );
            }
        }

        // 3. Inter-window grid borders.
        if self.config.show_borders {
            let (br, bg, bb) = self.config.border_color;
            let thickness = self.config.border_width.max(1);
            for window in self.config.windows.values() {
                let (wx, wy, ww, wh) = self.window_pixel_rect(window, canvas_w, canvas_h);
                if ww == 0 || wh == 0 {
                    continue;
                }
                let cx = wx.min(canvas_w);
                let cy = wy.min(canvas_h);
                let cw = ww.min(canvas_w - cx);
                let ch = wh.min(canvas_h - cy);
                if cw == 0 || ch == 0 {
                    continue;
                }
                draw_rect_border(
                    &mut canvas,
                    canvas_w,
                    cx,
                    cy,
                    cw,
                    ch,
                    thickness,
                    [br, bg, bb, 255],
                );
            }
        }

        // 4. Wrap the RGBA buffer in a VideoFrame.
        let mut frame = VideoFrame::new(PixelFormat::Rgba32, canvas_w, canvas_h);
        frame.planes.push(Plane::with_dimensions(
            canvas,
            (canvas_w * 4) as usize,
            canvas_w,
            canvas_h,
        ));
        Ok(frame)
    }

    /// Get tally border color for a state.
    pub fn tally_border_color(&self, state: TallyState) -> (u8, u8, u8) {
        match state {
            TallyState::Idle => (64, 64, 64),            // Gray
            TallyState::Program => (255, 0, 0),          // Red
            TallyState::Preview => (0, 255, 0),          // Green
            TallyState::ProgramPreview => (255, 165, 0), // Orange
        }
    }

    /// Calculate pixel coordinates for a window.
    ///
    /// The right and bottom edges are derived from the rounded far corner
    /// rather than from the rounded width, so adjacent windows that share an
    /// edge in normalised space also share it exactly in pixel space (no
    /// single-pixel seam from independent `as u32` truncation).
    pub fn window_pixel_rect(
        &self,
        window: &MultiviewerWindow,
        canvas_width: u32,
        canvas_height: u32,
    ) -> (u32, u32, u32, u32) {
        let cw = canvas_width as f32;
        let ch = canvas_height as f32;

        let x0 = (window.rect.x * cw) as u32;
        let y0 = (window.rect.y * ch) as u32;
        let x1 = ((window.rect.x + window.rect.width) * cw)
            .round()
            .clamp(0.0, cw) as u32;
        let y1 = ((window.rect.y + window.rect.height) * ch)
            .round()
            .clamp(0.0, ch) as u32;

        let width = x1.saturating_sub(x0);
        let height = y1.saturating_sub(y0);

        (x0, y0, width, height)
    }
}

// ─── Internal rendering helpers ────────────────────────────────────────────────

/// Allocate an opaque-black RGBA canvas of `width × height` pixels.
fn blank_rgba_canvas(width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    let mut canvas = vec![0u8; pixel_count * 4];
    // Set the alpha channel to fully opaque.
    for pixel in canvas.chunks_exact_mut(4) {
        pixel[3] = 255;
    }
    canvas
}

/// Read a single RGBA pixel out of a [`VideoFrame`], converting from the
/// frame's native pixel format. Returns opaque black for out-of-range
/// coordinates or unsupported formats.
fn sample_frame_rgba(frame: &VideoFrame, x: u32, y: u32) -> [u8; 4] {
    if x >= frame.width || y >= frame.height {
        return [0, 0, 0, 255];
    }
    let Some(plane0) = frame.planes.first() else {
        return [0, 0, 0, 255];
    };

    match frame.format {
        PixelFormat::Rgba32 => {
            let idx = (y as usize) * plane0.stride + (x as usize) * 4;
            plane0
                .data
                .get(idx..idx + 4)
                .map_or([0, 0, 0, 255], |s| [s[0], s[1], s[2], s[3]])
        }
        PixelFormat::Rgb24 => {
            let idx = (y as usize) * plane0.stride + (x as usize) * 3;
            plane0
                .data
                .get(idx..idx + 3)
                .map_or([0, 0, 0, 255], |s| [s[0], s[1], s[2], 255])
        }
        PixelFormat::Gray8 => {
            let idx = (y as usize) * plane0.stride + (x as usize);
            let g = plane0.data.get(idx).copied().unwrap_or(0);
            [g, g, g, 255]
        }
        _ => {
            // Planar YUV (and similar): sample luma, copy to all channels.
            // This keeps the multiviewer functional for decoded YUV frames
            // without a full colour-space conversion dependency.
            let idx = (y as usize) * plane0.stride + (x as usize);
            let luma = plane0.data.get(idx).copied().unwrap_or(0);
            [luma, luma, luma, 255]
        }
    }
}

/// Nearest-neighbour blit of `frame` into the RGBA `canvas` at the destination
/// rectangle `(dst_x, dst_y, dst_w, dst_h)`. The caller must guarantee the
/// destination rectangle lies fully within the canvas.
fn blit_frame_scaled(
    frame: &VideoFrame,
    canvas: &mut [u8],
    canvas_w: u32,
    dst_x: u32,
    dst_y: u32,
    dst_w: u32,
    dst_h: u32,
) {
    if frame.width == 0 || frame.height == 0 || dst_w == 0 || dst_h == 0 {
        return;
    }

    for row in 0..dst_h {
        let sy = ((row as u64 * frame.height as u64) / dst_h as u64) as u32;
        let sy = sy.min(frame.height - 1);
        for col in 0..dst_w {
            let sx = ((col as u64 * frame.width as u64) / dst_w as u64) as u32;
            let sx = sx.min(frame.width - 1);

            let rgba = sample_frame_rgba(frame, sx, sy);
            let dst_idx =
                (((dst_y + row) as usize) * (canvas_w as usize) + (dst_x + col) as usize) * 4;
            if let Some(slot) = canvas.get_mut(dst_idx..dst_idx + 4) {
                slot.copy_from_slice(&rgba);
            }
        }
    }
}

/// Fill an axis-aligned rectangle of the RGBA `canvas` with a solid colour.
fn fill_rect(canvas: &mut [u8], canvas_w: u32, x: u32, y: u32, w: u32, h: u32, color: [u8; 4]) {
    for row in 0..h {
        for col in 0..w {
            let idx = (((y + row) as usize) * (canvas_w as usize) + (x + col) as usize) * 4;
            if let Some(slot) = canvas.get_mut(idx..idx + 4) {
                slot.copy_from_slice(&color);
            }
        }
    }
}

/// Draw a hollow rectangle border of the given thickness onto the RGBA canvas.
fn draw_rect_border(
    canvas: &mut [u8],
    canvas_w: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    thickness: u32,
    color: [u8; 4],
) {
    if w == 0 || h == 0 {
        return;
    }
    let t = thickness.min(w.div_ceil(2)).min(h.div_ceil(2));
    for row in 0..h {
        let row_on_edge = row < t || row >= h - t;
        for col in 0..w {
            let on_border = row_on_edge || col < t || col >= w - t;
            if on_border {
                let idx = (((y + row) as usize) * (canvas_w as usize) + (x + col) as usize) * 4;
                if let Some(slot) = canvas.get_mut(idx..idx + 4) {
                    slot.copy_from_slice(&color);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_window_count() {
        assert_eq!(MultiviewerLayout::Grid2x2.window_count(), 4);
        assert_eq!(MultiviewerLayout::Grid3x3.window_count(), 9);
        assert_eq!(MultiviewerLayout::Grid4x4.window_count(), 16);
    }

    #[test]
    fn test_layout_grid_dimensions() {
        assert_eq!(MultiviewerLayout::Grid2x2.grid_dimensions(), (2, 2));
        assert_eq!(MultiviewerLayout::Grid3x3.grid_dimensions(), (3, 3));
        assert_eq!(MultiviewerLayout::Grid4x4.grid_dimensions(), (4, 4));
    }

    #[test]
    fn test_window_rect() {
        let rect = WindowRect::new(0.25, 0.5, 0.5, 0.5);
        assert_eq!(rect.x, 0.25);
        assert_eq!(rect.y, 0.5);
        assert_eq!(rect.width, 0.5);
        assert_eq!(rect.height, 0.5);
    }

    #[test]
    fn test_window_rect_clamping() {
        let rect = WindowRect::new(-0.5, 1.5, 2.0, 0.5);
        assert_eq!(rect.x, 0.0);
        assert_eq!(rect.y, 1.0);
        assert_eq!(rect.width, 1.0);
        assert_eq!(rect.height, 0.5);
    }

    #[test]
    fn test_window_rect_full_screen() {
        let rect = WindowRect::full_screen();
        assert_eq!(rect.x, 0.0);
        assert_eq!(rect.y, 0.0);
        assert_eq!(rect.width, 1.0);
        assert_eq!(rect.height, 1.0);
    }

    #[test]
    fn test_multiviewer_window() {
        let rect = WindowRect::new(0.0, 0.0, 0.5, 0.5);
        let mut window = MultiviewerWindow::new(0, 1, rect);

        assert_eq!(window.id, 0);
        assert_eq!(window.source_id, 1);
        assert!(window.show_tally);
        assert!(window.show_audio);
        assert!(window.show_label);

        window.set_label("Camera 1".to_string());
        assert_eq!(window.label, "Camera 1");

        window.set_source(2);
        assert_eq!(window.source_id, 2);
    }

    #[test]
    fn test_multiviewer_config_2x2() {
        let config = MultiviewerConfig::grid_2x2();
        assert_eq!(config.layout, MultiviewerLayout::Grid2x2);
        assert_eq!(config.window_count(), 4);
        assert!(config.show_borders);
    }

    #[test]
    fn test_multiviewer_config_4x4() {
        let config = MultiviewerConfig::grid_4x4();
        assert_eq!(config.layout, MultiviewerLayout::Grid4x4);
        assert_eq!(config.window_count(), 16);
    }

    #[test]
    fn test_multiviewer_config_add_remove_window() {
        let mut config = MultiviewerConfig::new(MultiviewerLayout::Custom);
        assert_eq!(config.window_count(), 0);

        let rect = WindowRect::full_screen();
        let window = MultiviewerWindow::new(0, 0, rect);
        config.add_window(window);
        assert_eq!(config.window_count(), 1);

        config.remove_window(0);
        assert_eq!(config.window_count(), 0);
    }

    #[test]
    fn test_multiviewer_creation() {
        let multiviewer = Multiviewer::grid_2x2();
        assert!(multiviewer.is_enabled());
        assert_eq!(multiviewer.config().window_count(), 4);
    }

    #[test]
    fn test_multiviewer_enable_disable() {
        let mut multiviewer = Multiviewer::grid_2x2();
        assert!(multiviewer.is_enabled());

        multiviewer.set_enabled(false);
        assert!(!multiviewer.is_enabled());
    }

    #[test]
    fn test_multiviewer_tally() {
        let mut multiviewer = Multiviewer::grid_2x2();

        assert_eq!(multiviewer.get_tally(1), TallyState::Idle);

        multiviewer.update_tally(1, TallyState::Program);
        assert_eq!(multiviewer.get_tally(1), TallyState::Program);

        multiviewer.update_tally(2, TallyState::Preview);
        assert_eq!(multiviewer.get_tally(2), TallyState::Preview);

        multiviewer.clear_tally();
        assert_eq!(multiviewer.get_tally(1), TallyState::Idle);
    }

    #[test]
    fn test_tally_border_colors() {
        let multiviewer = Multiviewer::grid_2x2();

        assert_eq!(
            multiviewer.tally_border_color(TallyState::Idle),
            (64, 64, 64)
        );
        assert_eq!(
            multiviewer.tally_border_color(TallyState::Program),
            (255, 0, 0)
        );
        assert_eq!(
            multiviewer.tally_border_color(TallyState::Preview),
            (0, 255, 0)
        );
        assert_eq!(
            multiviewer.tally_border_color(TallyState::ProgramPreview),
            (255, 165, 0)
        );
    }

    #[test]
    fn test_window_pixel_rect() {
        let multiviewer = Multiviewer::grid_2x2();
        let rect = WindowRect::new(0.5, 0.5, 0.5, 0.5);
        let window = MultiviewerWindow::new(0, 0, rect);

        let (x, y, width, height) = multiviewer.window_pixel_rect(&window, 1920, 1080);

        assert_eq!(x, 960);
        assert_eq!(y, 540);
        assert_eq!(width, 960);
        assert_eq!(height, 540);
    }

    #[test]
    fn test_get_window() {
        let config = MultiviewerConfig::grid_2x2();
        assert!(config.get_window(0).is_some());
        assert!(config.get_window(3).is_some());
        assert!(config.get_window(4).is_none());
    }

    #[test]
    fn test_get_window_mut() {
        let mut config = MultiviewerConfig::grid_2x2();

        let window = config.get_window_mut(0).expect("should succeed in test");
        window.set_label("Modified".to_string());

        assert_eq!(
            config.get_window(0).expect("should succeed in test").label,
            "Modified"
        );
    }

    // ─── render() ──────────────────────────────────────────────────────────

    /// Build a solid-colour RGBA [`VideoFrame`] for use as multiviewer input.
    fn make_rgba_frame(w: u32, h: u32, color: [u8; 4]) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Rgba32, w, h);
        let mut data = vec![0u8; (w * h * 4) as usize];
        for px in data.chunks_exact_mut(4) {
            px.copy_from_slice(&color);
        }
        frame
            .planes
            .push(Plane::with_dimensions(data, (w * 4) as usize, w, h));
        frame
    }

    /// Read an RGBA pixel out of a rendered canvas frame.
    fn canvas_pixel(frame: &VideoFrame, x: u32, y: u32) -> [u8; 4] {
        let plane = frame.plane(0);
        let idx = (y as usize) * plane.stride + (x as usize) * 4;
        let s = &plane.data[idx..idx + 4];
        [s[0], s[1], s[2], s[3]]
    }

    #[test]
    fn test_render_canvas_dimensions() {
        let mut config = MultiviewerConfig::grid_2x2();
        config.set_canvas_resolution(640, 480);
        let multiviewer = Multiviewer::new(config);

        let inputs: HashMap<usize, VideoFrame> = HashMap::new();
        let frame = multiviewer.render(&inputs).expect("render should succeed");

        assert_eq!(frame.width, 640);
        assert_eq!(frame.height, 480);
        assert_eq!(frame.format, PixelFormat::Rgba32);
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.plane(0).stride, 640 * 4);
        assert_eq!(frame.plane(0).data.len(), 640 * 480 * 4);
    }

    #[test]
    fn test_render_rejects_degenerate_canvas() {
        let mut config = MultiviewerConfig::grid_2x2();
        // Bypass the clamping setter to force a degenerate value.
        config.canvas_width = 0;
        let multiviewer = Multiviewer::new(config);

        let inputs: HashMap<usize, VideoFrame> = HashMap::new();
        let result = multiviewer.render(&inputs);
        assert!(matches!(result, Err(MultiviewerError::RenderingError(_))));
    }

    #[test]
    fn test_render_2x2_grid_scales_input_into_window() {
        let mut config = MultiviewerConfig::grid_2x2();
        config.set_canvas_resolution(400, 400);
        // Disable overlays so the window interior stays pure input colour.
        for window in config.windows.values_mut() {
            window.show_tally = false;
            window.show_label = false;
            window.show_audio = false;
        }
        config.show_borders = false;
        let multiviewer = Multiviewer::new(config);

        // Window 0 occupies the top-left quadrant; source id 0.
        let mut inputs: HashMap<usize, VideoFrame> = HashMap::new();
        inputs.insert(0, make_rgba_frame(32, 32, [10, 220, 30, 255]));

        let frame = multiviewer.render(&inputs).expect("render should succeed");

        // Centre of the top-left quadrant (100, 100) must carry the scaled
        // input colour, proving the region is not blank.
        let centre = canvas_pixel(&frame, 100, 100);
        assert_eq!(centre, [10, 220, 30, 255]);

        // A quadrant with no supplied input stays opaque black.
        let empty_quadrant = canvas_pixel(&frame, 300, 300);
        assert_eq!(empty_quadrant, [0, 0, 0, 255]);
    }

    #[test]
    fn test_render_draws_tally_border() {
        let mut config = MultiviewerConfig::grid_2x2();
        config.set_canvas_resolution(400, 400);
        config.border_width = 3;
        // Keep only the tally border; drop the grid border and overlays.
        config.show_borders = false;
        for window in config.windows.values_mut() {
            window.show_label = false;
            window.show_audio = false;
        }
        let multiviewer = {
            let mut mv = Multiviewer::new(config);
            // Mark source 0 as on-air (program → red border).
            mv.update_tally(0, TallyState::Program);
            mv
        };

        let mut inputs: HashMap<usize, VideoFrame> = HashMap::new();
        inputs.insert(0, make_rgba_frame(16, 16, [128, 128, 128, 255]));

        let frame = multiviewer.render(&inputs).expect("render should succeed");

        // Window 0 is the top-left 200×200 quadrant. Its top-left corner pixel
        // sits inside the 3-pixel tally border, which must be program red.
        let corner = canvas_pixel(&frame, 0, 0);
        assert_eq!(corner, [255, 0, 0, 255]);

        // A pixel one row/col in is still within the 3-pixel border.
        let near_corner = canvas_pixel(&frame, 2, 2);
        assert_eq!(near_corner, [255, 0, 0, 255]);

        // The interior away from every edge keeps the input colour.
        let interior = canvas_pixel(&frame, 100, 100);
        assert_eq!(interior, [128, 128, 128, 255]);
    }

    #[test]
    fn test_render_label_strip_is_drawn() {
        let mut config = MultiviewerConfig::grid_2x2();
        config.set_canvas_resolution(400, 400);
        config.show_borders = false;
        for window in config.windows.values_mut() {
            window.show_tally = false;
            window.show_audio = false;
            window.show_label = true;
        }
        let multiviewer = Multiviewer::new(config);

        let mut inputs: HashMap<usize, VideoFrame> = HashMap::new();
        inputs.insert(0, make_rgba_frame(16, 16, [200, 200, 200, 255]));

        let frame = multiviewer.render(&inputs).expect("render should succeed");

        // The label strip hugs the bottom edge of the 200-tall window 0; the
        // pixel just above the bottom edge must be the dark strip colour.
        let strip_pixel = canvas_pixel(&frame, 100, 199);
        assert_eq!(strip_pixel, [0, 0, 0, 200]);
    }

    #[test]
    fn test_render_no_seam_between_adjacent_windows() {
        // Two windows sharing the vertical mid-line must tile it exactly.
        let mut config = MultiviewerConfig::new(MultiviewerLayout::Custom);
        config.set_canvas_resolution(101, 64);
        config.show_borders = false;
        let mut left = MultiviewerWindow::new(0, 0, WindowRect::new(0.0, 0.0, 0.5, 1.0));
        left.show_tally = false;
        left.show_label = false;
        left.show_audio = false;
        let mut right = MultiviewerWindow::new(1, 1, WindowRect::new(0.5, 0.0, 0.5, 1.0));
        right.show_tally = false;
        right.show_label = false;
        right.show_audio = false;
        config.add_window(left);
        config.add_window(right);
        let multiviewer = Multiviewer::new(config);

        let mut inputs: HashMap<usize, VideoFrame> = HashMap::new();
        inputs.insert(0, make_rgba_frame(8, 8, [255, 0, 0, 255]));
        inputs.insert(1, make_rgba_frame(8, 8, [0, 0, 255, 255]));

        let frame = multiviewer.render(&inputs).expect("render should succeed");

        // Every pixel on the single mid-row must be either red or blue —
        // a seam would leave an opaque-black gap.
        for x in 0..101 {
            let p = canvas_pixel(&frame, x, 32);
            assert!(
                p == [255, 0, 0, 255] || p == [0, 0, 255, 255],
                "seam at x={x}: pixel {p:?}"
            );
        }
    }

    #[test]
    fn test_render_inter_window_borders() {
        let mut config = MultiviewerConfig::grid_2x2();
        config.set_canvas_resolution(400, 400);
        config.show_borders = true;
        config.border_color = (64, 64, 64);
        config.border_width = 2;
        for window in config.windows.values_mut() {
            window.show_tally = false;
            window.show_label = false;
            window.show_audio = false;
        }
        let multiviewer = Multiviewer::new(config);

        let inputs: HashMap<usize, VideoFrame> = HashMap::new();
        let frame = multiviewer.render(&inputs).expect("render should succeed");

        // The top-left corner of window 0 is on the grid border.
        let corner = canvas_pixel(&frame, 0, 0);
        assert_eq!(corner, [64, 64, 64, 255]);
    }
}

//! Scope overlay rendering helpers: graticules, grids, crosshairs, and alpha blending.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::manual_is_multiple_of)]

/// Configuration for a scope overlay.
#[derive(Debug, Clone)]
pub struct OverlayConfig {
    /// Width of the scope display in pixels.
    pub width: u32,
    /// Height of the scope display in pixels.
    pub height: u32,
    /// Opacity of the overlay (0.0 = fully transparent, 1.0 = fully opaque).
    pub opacity: f32,
    /// RGBA color used for lines and decorations.
    pub color: [u8; 4],
    /// Whether to render a reference grid.
    pub show_grid: bool,
}

impl OverlayConfig {
    /// Default configuration suitable for a waveform monitor overlay.
    pub fn default_waveform() -> Self {
        Self {
            width: 512,
            height: 256,
            opacity: 0.7,
            color: [0, 220, 0, 200],
            show_grid: true,
        }
    }

    /// Default configuration suitable for a vectorscope overlay.
    pub fn default_vectorscope() -> Self {
        Self {
            width: 512,
            height: 512,
            opacity: 0.6,
            color: [200, 200, 200, 180],
            show_grid: true,
        }
    }
}

/// A single graticule (reference) line for a scope display.
#[derive(Debug, Clone)]
pub struct GridLine {
    /// Normalised position along the relevant axis (0.0–1.0).
    pub position: f32,
    /// Human-readable label (e.g. "0 IRE", "100%").
    pub label: String,
    /// `true` for major (thicker / more prominent) lines.
    pub is_major: bool,
}

/// Generate evenly-spaced graticule lines.
///
/// `count` is the total number of lines (including endpoints).
/// When `labeled` is `true`, every line gets a label showing its percentage position.
pub fn generate_graticule_lines(count: u32, labeled: bool) -> Vec<GridLine> {
    if count == 0 {
        return Vec::new();
    }
    let n = count.max(2);
    (0..n)
        .map(|i| {
            let position = i as f32 / (n - 1) as f32;
            let pct = (position * 100.0).round() as u32;
            let label = if labeled {
                format!("{pct}%")
            } else {
                String::new()
            };
            // Major lines at the two endpoints plus every 25 %.
            let is_major = pct == 0 || pct == 100 || pct % 25 == 0;
            GridLine {
                position,
                label,
                is_major,
            }
        })
        .collect()
}

/// Draw a crosshair centred at `(cx, cy)` into an RGBA pixel buffer.
///
/// `pixels` must hold `width * height * 4` bytes (RGBA, row-major).
#[allow(clippy::too_many_arguments)]
pub fn draw_crosshair(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    cx: u32,
    cy: u32,
    color: [u8; 4],
) {
    // Horizontal arm.
    for x in 0..width {
        let offset = ((cy * width + x) * 4) as usize;
        if offset + 3 < pixels.len() {
            blend_pixel(pixels, offset, color);
        }
    }
    // Vertical arm.
    for y in 0..height {
        let offset = ((y * width + cx) * 4) as usize;
        if offset + 3 < pixels.len() {
            blend_pixel(pixels, offset, color);
        }
    }
}

/// Draw a grid of `cols` × `rows` cells into an RGBA pixel buffer.
///
/// `pixels` must hold `width * height * 4` bytes (RGBA, row-major).
pub fn draw_grid(pixels: &mut [u8], width: u32, height: u32, cols: u32, rows: u32, color: [u8; 4]) {
    if cols == 0 || rows == 0 {
        return;
    }
    // Vertical lines.
    for c in 0..=cols {
        let x = (c * width / cols).min(width.saturating_sub(1));
        for y in 0..height {
            let offset = ((y * width + x) * 4) as usize;
            if offset + 3 < pixels.len() {
                blend_pixel(pixels, offset, color);
            }
        }
    }
    // Horizontal lines.
    for r in 0..=rows {
        let y = (r * height / rows).min(height.saturating_sub(1));
        for x in 0..width {
            let offset = ((y * width + x) * 4) as usize;
            if offset + 3 < pixels.len() {
                blend_pixel(pixels, offset, color);
            }
        }
    }
}

/// Alpha-blend a source RGBA `src` pixel over the destination at `offset` (byte index).
///
/// Standard "over" compositing: `out = src_alpha * src + (1 − src_alpha) * dst`.
pub fn blend_pixel(dst: &mut [u8], offset: usize, src: [u8; 4]) {
    if offset + 3 >= dst.len() {
        return;
    }
    let alpha = src[3] as f32 / 255.0;
    let inv_alpha = 1.0 - alpha;
    for i in 0..3 {
        let blended = alpha * src[i] as f32 + inv_alpha * dst[offset + i] as f32;
        dst[offset + i] = blended.round().clamp(0.0, 255.0) as u8;
    }
    // Composite alpha: result is at least as opaque as whichever is higher.
    let out_alpha = (src[3] as f32 + dst[offset + 3] as f32 * inv_alpha)
        .clamp(0.0, 255.0)
        .round() as u8;
    dst[offset + 3] = out_alpha;
}

// ---------------------------------------------------------------------------
// OverlayColor
// ---------------------------------------------------------------------------

/// An RGBA color value for scope overlays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayColor {
    /// Red component (0–255).
    pub r: u8,
    /// Green component (0–255).
    pub g: u8,
    /// Blue component (0–255).
    pub b: u8,
    /// Alpha component (0–255, 255 = fully opaque).
    pub a: u8,
}

impl OverlayColor {
    /// Opaque red.
    #[must_use]
    pub fn red() -> Self {
        Self {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        }
    }

    /// Opaque green.
    #[must_use]
    pub fn green() -> Self {
        Self {
            r: 0,
            g: 255,
            b: 0,
            a: 255,
        }
    }

    /// Opaque blue.
    #[must_use]
    pub fn blue() -> Self {
        Self {
            r: 0,
            g: 0,
            b: 255,
            a: 255,
        }
    }

    /// Opaque white.
    #[must_use]
    pub fn white() -> Self {
        Self {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        }
    }

    /// Opaque yellow.
    #[must_use]
    pub fn yellow() -> Self {
        Self {
            r: 255,
            g: 255,
            b: 0,
            a: 255,
        }
    }

    /// Alpha-blends `self` over `other` with the given blend factor (0.0 = `other`, 1.0 = `self`).
    #[must_use]
    pub fn blend(&self, other: OverlayColor, alpha: f32) -> OverlayColor {
        let a = alpha.clamp(0.0, 1.0);
        let ia = 1.0 - a;
        let r = (self.r as f32 * a + other.r as f32 * ia).round() as u8;
        let g = (self.g as f32 * a + other.g as f32 * ia).round() as u8;
        let b = (self.b as f32 * a + other.b as f32 * ia).round() as u8;
        let out_a = (self.a as f32 * a + other.a as f32 * ia).round() as u8;
        OverlayColor { r, g, b, a: out_a }
    }
}

// ---------------------------------------------------------------------------
// GridOverlay
// ---------------------------------------------------------------------------

/// Renders an evenly-spaced grid overlay for a video scope display.
#[derive(Debug, Clone)]
pub struct GridOverlay {
    /// Number of vertical divisions (columns).
    pub cols: u32,
    /// Number of horizontal divisions (rows).
    pub rows: u32,
    /// Color used to draw the grid lines.
    pub color: OverlayColor,
}

impl GridOverlay {
    /// Returns the Y pixel positions of all horizontal grid lines.
    #[must_use]
    pub fn line_positions_h(&self, height: u32) -> Vec<u32> {
        if self.rows == 0 {
            return Vec::new();
        }
        (0..=self.rows)
            .map(|r| (r * height / self.rows).min(height.saturating_sub(1)))
            .collect()
    }

    /// Returns the X pixel positions of all vertical grid lines.
    #[must_use]
    pub fn line_positions_v(&self, width: u32) -> Vec<u32> {
        if self.cols == 0 {
            return Vec::new();
        }
        (0..=self.cols)
            .map(|c| (c * width / self.cols).min(width.saturating_sub(1)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SafeAreaMarker
// ---------------------------------------------------------------------------

/// Calculates action-safe and title-safe bounding boxes for a frame.
///
/// `inner_pct` is the percentage of the frame each edge insets for the title-safe area,
/// and `outer_pct` is the inset for the action-safe area.
#[derive(Debug, Clone)]
pub struct SafeAreaMarker {
    /// Action-safe inset percentage (e.g. 5.0 for 5% each side).
    pub inner_pct: f32,
    /// Title-safe inset percentage (e.g. 10.0 for 10% each side).
    pub outer_pct: f32,
}

impl SafeAreaMarker {
    /// Returns `(x, y, w, h)` for the action-safe bounding box.
    #[must_use]
    pub fn action_safe_box(&self, w: u32, h: u32) -> (u32, u32, u32, u32) {
        Self::safe_box(w, h, self.inner_pct)
    }

    /// Returns `(x, y, w, h)` for the title-safe bounding box.
    #[must_use]
    pub fn title_safe_box(&self, w: u32, h: u32) -> (u32, u32, u32, u32) {
        Self::safe_box(w, h, self.outer_pct)
    }

    fn safe_box(w: u32, h: u32, pct: f32) -> (u32, u32, u32, u32) {
        let inset_x = (w as f32 * pct / 100.0).round() as u32;
        let inset_y = (h as f32 * pct / 100.0).round() as u32;
        let bw = w.saturating_sub(2 * inset_x);
        let bh = h.saturating_sub(2 * inset_y);
        (inset_x, inset_y, bw, bh)
    }
}

// ---------------------------------------------------------------------------
// CrosshairOverlay
// ---------------------------------------------------------------------------

/// A crosshair overlay centred at `(x, y)` with a given arm length and color.
#[derive(Debug, Clone)]
pub struct CrosshairOverlay {
    /// Horizontal centre of the crosshair.
    pub x: u32,
    /// Vertical centre of the crosshair.
    pub y: u32,
    /// Half-length of each arm in pixels.
    pub size: u32,
    /// Color of the crosshair arms.
    pub color: OverlayColor,
}

impl CrosshairOverlay {
    /// Returns all `(x, y)` pixel positions along the crosshair arms.
    ///
    /// Saturating arithmetic prevents wrapping when `x` or `y` are near zero.
    #[must_use]
    pub fn points(&self) -> Vec<(u32, u32)> {
        let mut pts = Vec::new();
        // Horizontal arm
        let x_start = self.x.saturating_sub(self.size);
        let x_end = self.x.saturating_add(self.size);
        for px in x_start..=x_end {
            pts.push((px, self.y));
        }
        // Vertical arm (skip centre already added)
        let y_start = self.y.saturating_sub(self.size);
        let y_end = self.y.saturating_add(self.size);
        for py in y_start..=y_end {
            if py != self.y {
                pts.push((self.x, py));
            }
        }
        pts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_config_waveform_defaults() {
        let cfg = OverlayConfig::default_waveform();
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.height, 256);
        assert!(cfg.show_grid);
    }

    #[test]
    fn test_overlay_config_vectorscope_defaults() {
        let cfg = OverlayConfig::default_vectorscope();
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.height, 512);
        assert!(cfg.show_grid);
    }

    #[test]
    fn test_generate_graticule_lines_count() {
        let lines = generate_graticule_lines(5, false);
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn test_generate_graticule_lines_zero() {
        let lines = generate_graticule_lines(0, false);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_generate_graticule_lines_endpoints() {
        let lines = generate_graticule_lines(5, true);
        assert!((lines[0].position - 0.0).abs() < 1e-5);
        assert!((lines[4].position - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_generate_graticule_lines_major() {
        let lines = generate_graticule_lines(5, true);
        // 0%, 25%, 50%, 75%, 100% — all should be major.
        assert!(lines.iter().all(|l| l.is_major));
    }

    #[test]
    fn test_generate_graticule_lines_labeled() {
        let lines = generate_graticule_lines(5, true);
        assert!(lines[0].label.contains('%'));
    }

    #[test]
    fn test_blend_pixel_full_opaque() {
        let mut buf = vec![0u8; 4];
        blend_pixel(&mut buf, 0, [255, 0, 0, 255]);
        assert_eq!(buf[0], 255);
        assert_eq!(buf[1], 0);
        assert_eq!(buf[2], 0);
    }

    #[test]
    fn test_blend_pixel_transparent() {
        let mut buf = vec![100u8, 150u8, 200u8, 255u8];
        blend_pixel(&mut buf, 0, [0, 0, 0, 0]);
        // Fully transparent source: destination unchanged.
        assert_eq!(buf[0], 100);
        assert_eq!(buf[1], 150);
        assert_eq!(buf[2], 200);
    }

    #[test]
    fn test_blend_pixel_half_alpha() {
        let mut buf = vec![0u8, 0u8, 0u8, 255u8];
        blend_pixel(&mut buf, 0, [200, 100, 50, 128]);
        // ~50% blend
        assert!(buf[0] > 90 && buf[0] < 110);
    }

    #[test]
    fn test_draw_crosshair_marks_center_row_col() {
        let w = 10u32;
        let h = 10u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        let color = [255, 255, 255, 255];
        draw_crosshair(&mut pixels, w, h, 5, 5, color);
        // Check that the centre row pixel is set.
        let row_offset = (5 * w + 0) as usize * 4;
        assert_eq!(pixels[row_offset], 255);
        // Check that the centre column pixel is set.
        let col_offset = (0 * w + 5) as usize * 4;
        assert_eq!(pixels[col_offset], 255);
    }

    #[test]
    fn test_draw_grid_does_not_panic_on_small_buffer() {
        let w = 8u32;
        let h = 8u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        draw_grid(&mut pixels, w, h, 4, 4, [128, 128, 128, 255]);
        // No panic is success; also verify at least one pixel was written.
        assert!(pixels.iter().any(|&b| b > 0));
    }

    #[test]
    fn test_draw_grid_zero_cols_rows_is_noop() {
        let w = 8u32;
        let h = 8u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        draw_grid(&mut pixels, w, h, 0, 0, [255, 0, 0, 255]);
        // No lines should be written.
        assert!(pixels.iter().all(|&b| b == 0));
    }

    // --- OverlayColor tests ---

    #[test]
    fn test_overlay_color_red() {
        let c = OverlayColor::red();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn test_overlay_color_green() {
        let c = OverlayColor::green();
        assert_eq!((c.r, c.g, c.b), (0, 255, 0));
    }

    #[test]
    fn test_overlay_color_blue() {
        let c = OverlayColor::blue();
        assert_eq!((c.r, c.g, c.b), (0, 0, 255));
    }

    #[test]
    fn test_overlay_color_white() {
        let c = OverlayColor::white();
        assert_eq!((c.r, c.g, c.b, c.a), (255, 255, 255, 255));
    }

    #[test]
    fn test_overlay_color_yellow() {
        let c = OverlayColor::yellow();
        assert_eq!((c.r, c.g, c.b), (255, 255, 0));
    }

    #[test]
    fn test_overlay_color_blend_full() {
        let red = OverlayColor::red();
        let blue = OverlayColor::blue();
        let blended = red.blend(blue, 1.0);
        assert_eq!(blended.r, 255);
        assert_eq!(blended.b, 0);
    }

    #[test]
    fn test_overlay_color_blend_zero() {
        let red = OverlayColor::red();
        let blue = OverlayColor::blue();
        let blended = red.blend(blue, 0.0);
        assert_eq!(blended.r, 0);
        assert_eq!(blended.b, 255);
    }

    #[test]
    fn test_overlay_color_blend_half() {
        let red = OverlayColor::red();
        let blue = OverlayColor::blue();
        let blended = red.blend(blue, 0.5);
        assert!(blended.r > 100 && blended.r < 160);
        assert!(blended.b > 100 && blended.b < 160);
    }

    // --- GridOverlay tests ---

    #[test]
    fn test_grid_overlay_line_positions_h() {
        let g = GridOverlay {
            cols: 4,
            rows: 4,
            color: OverlayColor::white(),
        };
        let pos = g.line_positions_h(400);
        assert_eq!(pos.len(), 5); // 0..=rows
        assert_eq!(pos[0], 0);
    }

    #[test]
    fn test_grid_overlay_line_positions_v() {
        let g = GridOverlay {
            cols: 8,
            rows: 4,
            color: OverlayColor::white(),
        };
        let pos = g.line_positions_v(800);
        assert_eq!(pos.len(), 9); // 0..=cols
    }

    #[test]
    fn test_grid_overlay_zero_rows_empty() {
        let g = GridOverlay {
            cols: 0,
            rows: 0,
            color: OverlayColor::white(),
        };
        assert!(g.line_positions_h(100).is_empty());
        assert!(g.line_positions_v(100).is_empty());
    }

    // --- SafeAreaMarker tests ---

    #[test]
    fn test_safe_area_marker_action_safe() {
        let m = SafeAreaMarker {
            inner_pct: 5.0,
            outer_pct: 10.0,
        };
        let (x, y, w, h) = m.action_safe_box(1920, 1080);
        assert_eq!(x, 96); // 1920 * 5% = 96
        assert_eq!(y, 54); // 1080 * 5% = 54
        assert_eq!(w, 1920 - 2 * 96);
        assert_eq!(h, 1080 - 2 * 54);
    }

    #[test]
    fn test_safe_area_marker_title_safe() {
        let m = SafeAreaMarker {
            inner_pct: 5.0,
            outer_pct: 10.0,
        };
        let (x, y, w, h) = m.title_safe_box(1920, 1080);
        assert_eq!(x, 192); // 1920 * 10% = 192
        assert_eq!(y, 108); // 1080 * 10% = 108
        assert_eq!(w, 1920 - 2 * 192);
        assert_eq!(h, 1080 - 2 * 108);
    }

    // --- CrosshairOverlay tests ---

    #[test]
    fn test_crosshair_overlay_points_count() {
        let c = CrosshairOverlay {
            x: 50,
            y: 50,
            size: 5,
            color: OverlayColor::white(),
        };
        let pts = c.points();
        // Horizontal: 2*5+1 = 11; Vertical: 2*5 = 10 (centre excluded)
        assert_eq!(pts.len(), 21);
    }

    #[test]
    fn test_crosshair_overlay_center_in_points() {
        let c = CrosshairOverlay {
            x: 10,
            y: 10,
            size: 3,
            color: OverlayColor::red(),
        };
        let pts = c.points();
        assert!(pts.contains(&(10, 10)));
    }
}

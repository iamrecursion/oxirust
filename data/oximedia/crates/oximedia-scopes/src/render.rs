//! Rendering utilities for video scopes.
//!
//! This module provides functions for rendering scope displays, including:
//! - Graticule (grid) overlays
//! - Text labels and annotations
//! - Color mapping and blending
//! - Anti-aliasing
//! - High-DPI support

use crate::ScopeConfig;

/// RGBA color type.
pub type Color = [u8; 4];

/// Common colors for scope rendering.
pub mod colors {
    use super::Color;

    /// White color.
    pub const WHITE: Color = [255, 255, 255, 255];

    /// Black color.
    pub const BLACK: Color = [0, 0, 0, 255];

    /// Gray color (50%).
    pub const GRAY: Color = [128, 128, 128, 255];

    /// Dark gray color (25%).
    pub const DARK_GRAY: Color = [64, 64, 64, 255];

    /// Light gray color (75%).
    pub const LIGHT_GRAY: Color = [192, 192, 192, 255];

    /// Red color.
    pub const RED: Color = [255, 0, 0, 255];

    /// Green color.
    pub const GREEN: Color = [0, 255, 0, 255];

    /// Blue color.
    pub const BLUE: Color = [0, 0, 255, 255];

    /// Yellow color.
    pub const YELLOW: Color = [255, 255, 0, 255];

    /// Cyan color.
    pub const CYAN: Color = [0, 255, 255, 255];

    /// Magenta color.
    pub const MAGENTA: Color = [255, 0, 255, 255];

    /// Semi-transparent white for graticule.
    pub const GRATICULE: Color = [255, 255, 255, 128];

    /// Semi-transparent red for warnings.
    pub const WARNING: Color = [255, 0, 0, 192];
}

/// Canvas for drawing scope displays.
pub struct Canvas {
    /// Width in pixels.
    pub width: u32,

    /// Height in pixels.
    pub height: u32,

    /// RGBA pixel data (row-major).
    pub data: Vec<u8>,
}

impl Canvas {
    /// Creates a new canvas with the given dimensions.
    ///
    /// The canvas is initialized with a black background.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let data = vec![0u8; (width * height * 4) as usize];
        Self {
            width,
            height,
            data,
        }
    }

    /// Sets a pixel at the given coordinates.
    ///
    /// If the coordinates are out of bounds, this is a no-op.
    pub fn set_pixel(&mut self, x: u32, y: u32, color: Color) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = ((y * self.width + x) * 4) as usize;
        self.data[idx..idx + 4].copy_from_slice(&color);
    }

    /// Gets a pixel at the given coordinates.
    ///
    /// Returns black if out of bounds.
    #[must_use]
    pub fn get_pixel(&self, x: u32, y: u32) -> Color {
        if x >= self.width || y >= self.height {
            return colors::BLACK;
        }

        let idx = ((y * self.width + x) * 4) as usize;
        [
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        ]
    }

    /// Blends a color with the existing pixel using alpha blending.
    pub fn blend_pixel(&mut self, x: u32, y: u32, color: Color) {
        if x >= self.width || y >= self.height {
            return;
        }

        let existing = self.get_pixel(x, y);
        let alpha = f32::from(color[3]) / 255.0;
        let inv_alpha = 1.0 - alpha;

        let blended = [
            (f32::from(color[0]) * alpha + f32::from(existing[0]) * inv_alpha) as u8,
            (f32::from(color[1]) * alpha + f32::from(existing[1]) * inv_alpha) as u8,
            (f32::from(color[2]) * alpha + f32::from(existing[2]) * inv_alpha) as u8,
            255,
        ];

        self.set_pixel(x, y, blended);
    }

    /// Accumulates intensity at a pixel (for waveform/vectorscope).
    ///
    /// This adds to the existing pixel value, clamping at 255.
    pub fn accumulate_pixel(&mut self, x: u32, y: u32, intensity: u8) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = ((y * self.width + x) * 4) as usize;

        // Accumulate intensity on all RGB channels (creates grayscale)
        for i in 0..3 {
            let current = self.data[idx + i];
            self.data[idx + i] = current.saturating_add(intensity);
        }
        self.data[idx + 3] = 255; // Alpha channel
    }

    /// Draws a horizontal line.
    pub fn draw_hline(&mut self, x1: u32, x2: u32, y: u32, color: Color) {
        let x_start = x1.min(x2);
        let x_end = x1.max(x2);

        for x in x_start..=x_end {
            if color[3] < 255 {
                self.blend_pixel(x, y, color);
            } else {
                self.set_pixel(x, y, color);
            }
        }
    }

    /// Draws a vertical line.
    pub fn draw_vline(&mut self, x: u32, y1: u32, y2: u32, color: Color) {
        let y_start = y1.min(y2);
        let y_end = y1.max(y2);

        for y in y_start..=y_end {
            if color[3] < 255 {
                self.blend_pixel(x, y, color);
            } else {
                self.set_pixel(x, y, color);
            }
        }
    }

    /// Draws a line using Bresenham's algorithm.
    #[allow(clippy::cast_possible_wrap)]
    #[allow(clippy::cast_sign_loss)]
    pub fn draw_line(&mut self, x1: u32, y1: u32, x2: u32, y2: u32, color: Color) {
        let x1_i = x1 as i32;
        let y1_i = y1 as i32;
        let x2_i = x2 as i32;
        let y2_i = y2 as i32;

        let dx = (x2_i - x1_i).abs();
        let dy = -(y2_i - y1_i).abs();
        let sx = if x1_i < x2_i { 1 } else { -1 };
        let sy = if y1_i < y2_i { 1 } else { -1 };
        let mut err = dx + dy;

        let mut x = x1_i;
        let mut y = y1_i;

        loop {
            if x >= 0 && y >= 0 {
                if color[3] < 255 {
                    self.blend_pixel(x as u32, y as u32, color);
                } else {
                    self.set_pixel(x as u32, y as u32, color);
                }
            }

            if x == x2_i && y == y2_i {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Draws a rectangle outline.
    pub fn draw_rect(&mut self, x: u32, y: u32, width: u32, height: u32, color: Color) {
        if width == 0 || height == 0 {
            return;
        }

        let x2 = x + width - 1;
        let y2 = y + height - 1;

        self.draw_hline(x, x2, y, color);
        self.draw_hline(x, x2, y2, color);
        self.draw_vline(x, y, y2, color);
        self.draw_vline(x2, y, y2, color);
    }

    /// Fills a rectangle.
    pub fn fill_rect(&mut self, x: u32, y: u32, width: u32, height: u32, color: Color) {
        for dy in 0..height {
            for dx in 0..width {
                let px = x + dx;
                let py = y + dy;
                if px < self.width && py < self.height {
                    if color[3] < 255 {
                        self.blend_pixel(px, py, color);
                    } else {
                        self.set_pixel(px, py, color);
                    }
                }
            }
        }
    }

    /// Draws a circle outline.
    #[allow(clippy::cast_possible_wrap)]
    #[allow(clippy::cast_sign_loss)]
    pub fn draw_circle(&mut self, cx: u32, cy: u32, radius: u32, color: Color) {
        let cx_i = cx as i32;
        let cy_i = cy as i32;
        let r = radius as i32;

        let mut x = 0;
        let mut y = r;
        let mut d = 3 - 2 * r;

        while x <= y {
            // Draw 8 octants
            let points = [
                (cx_i + x, cy_i + y),
                (cx_i - x, cy_i + y),
                (cx_i + x, cy_i - y),
                (cx_i - x, cy_i - y),
                (cx_i + y, cy_i + x),
                (cx_i - y, cy_i + x),
                (cx_i + y, cy_i - x),
                (cx_i - y, cy_i - x),
            ];

            for (px, py) in &points {
                if *px >= 0 && *py >= 0 {
                    if color[3] < 255 {
                        self.blend_pixel(*px as u32, *py as u32, color);
                    } else {
                        self.set_pixel(*px as u32, *py as u32, color);
                    }
                }
            }

            if d <= 0 {
                d += 4 * x + 6;
            } else {
                d += 4 * (x - y) + 10;
                y -= 1;
            }
            x += 1;
        }
    }

    /// Clears the canvas to black.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Clears the canvas to a specific color.
    pub fn clear_with_color(&mut self, color: Color) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.set_pixel(x, y, color);
            }
        }
    }
}

/// Draws a waveform graticule with IRE scale.
///
/// IRE (Institute of Radio Engineers) units: 0 IRE = black, 100 IRE = white.
/// 7.5 IRE = setup level (NTSC), -40 to +100 IRE = typical broadcast range.
pub fn draw_waveform_graticule(canvas: &mut Canvas, config: &ScopeConfig) {
    if !config.show_graticule {
        return;
    }

    let width = canvas.width;
    let height = canvas.height;

    // Draw horizontal lines at key IRE levels
    let ire_levels = [
        (0, "0"),     // Black
        (10, "10"),   // Near black
        (50, "50"),   // Mid gray
        (75, "75"),   // 75% white
        (100, "100"), // White
    ];

    for (ire, _label) in &ire_levels {
        let y = height - ((*ire as u32 * height) / 100);
        canvas.draw_hline(0, width - 1, y, colors::GRATICULE);
    }

    // Draw vertical lines at 25%, 50%, 75% of width
    for i in 1..4 {
        let x = (width * i) / 4;
        canvas.draw_vline(x, 0, height - 1, colors::GRATICULE);
    }
}

/// Draws a vectorscope graticule with SMPTE color bar targets.
///
/// SMPTE color bars: 75% saturation targets for Red, Yellow, Green, Cyan, Blue, Magenta.
/// Also includes skin tone line and I/Q axes.
pub fn draw_vectorscope_graticule(canvas: &mut Canvas, config: &ScopeConfig) {
    if !config.show_graticule {
        return;
    }

    let width = canvas.width;
    let height = canvas.height;
    let cx = width / 2;
    let cy = height / 2;

    // Draw center crosshair
    canvas.draw_hline(0, width - 1, cy, colors::GRATICULE);
    canvas.draw_vline(cx, 0, height - 1, colors::GRATICULE);

    // Draw concentric circles for saturation levels
    let max_radius = width.min(height) / 2 - 10;
    for i in 1..=4 {
        let radius = (max_radius * i) / 4;
        canvas.draw_circle(cx, cy, radius, colors::GRATICULE);
    }

    // Draw SMPTE color bar target boxes (75% and 100% saturation)
    // Approximate angles for color primaries (in degrees):
    // Red: 104°, Yellow: 168°, Green: 241°, Cyan: 284°, Blue: 348°, Magenta: 61°
    let targets = [
        (104_f32, colors::RED),
        (168_f32, colors::YELLOW),
        (241_f32, colors::GREEN),
        (284_f32, colors::CYAN),
        (348_f32, colors::BLUE),
        (61_f32, colors::MAGENTA),
    ];

    for (angle, color) in &targets {
        let rad = angle.to_radians();
        let radius_75 = (max_radius * 3) / 4;

        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let x = (cx as f32 + rad.cos() * radius_75 as f32) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let y = (cy as f32 - rad.sin() * radius_75 as f32) as u32;

        // Draw small target box
        if x >= 2 && y >= 2 && x < width - 2 && y < height - 2 {
            canvas.draw_rect(x - 2, y - 2, 5, 5, *color);
        }
    }

    // Draw skin tone line (approximately 123° to center)
    let skin_angle = 123_f32.to_radians();
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let skin_x = (cx as f32 + skin_angle.cos() * max_radius as f32) as u32;
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let skin_y = (cy as f32 - skin_angle.sin() * max_radius as f32) as u32;
    canvas.draw_line(cx, cy, skin_x, skin_y, [255, 200, 150, 128]);
}

/// Draws a histogram graticule with percentile markers.
pub fn draw_histogram_graticule(canvas: &mut Canvas, config: &ScopeConfig) {
    if !config.show_graticule {
        return;
    }

    let width = canvas.width;
    let height = canvas.height;

    // Draw horizontal lines at 25%, 50%, 75%, 100% of height
    for i in 1..=4 {
        let y = (height * i) / 4;
        canvas.draw_hline(0, width - 1, height - y, colors::GRATICULE);
    }

    // Draw vertical lines at key luminance levels
    // 0-255 range: 0, 16 (black), 128 (mid), 235 (white), 255
    let levels = [0, 16, 128, 235, 255];
    for level in &levels {
        let x = (*level as u32 * width) / 255;
        canvas.draw_vline(x, 0, height - 1, colors::GRATICULE);
    }
}

/// Draws a parade graticule.
pub fn draw_parade_graticule(canvas: &mut Canvas, config: &ScopeConfig, sections: u32) {
    if !config.show_graticule {
        return;
    }

    let width = canvas.width;
    let height = canvas.height;

    // Draw horizontal lines at IRE levels
    let ire_levels = [0, 25, 50, 75, 100];
    for ire in &ire_levels {
        let y = height - ((*ire as u32 * height) / 100);
        canvas.draw_hline(0, width - 1, y, colors::GRATICULE);
    }

    // Draw vertical separators between sections
    let section_width = width / sections;
    for i in 1..sections {
        let x = section_width * i;
        canvas.draw_vline(x, 0, height - 1, colors::GRATICULE);
    }
}

/// Simple 5x7 bitmap font for labels.
const FONT_5X7: [[u8; 5]; 10] = [
    [0x7E, 0x81, 0x81, 0x81, 0x7E], // 0
    [0x00, 0x82, 0xFF, 0x80, 0x00], // 1
    [0xC2, 0xA1, 0x91, 0x89, 0x86], // 2
    [0x42, 0x81, 0x89, 0x89, 0x76], // 3
    [0x18, 0x14, 0x12, 0xFF, 0x10], // 4
    [0x4F, 0x89, 0x89, 0x89, 0x71], // 5
    [0x7E, 0x89, 0x89, 0x89, 0x72], // 6
    [0x01, 0xE1, 0x11, 0x09, 0x07], // 7
    [0x76, 0x89, 0x89, 0x89, 0x76], // 8
    [0x4E, 0x91, 0x91, 0x91, 0x7E], // 9
];

/// Draws a simple numeric label at the given position.
///
/// Only supports digits 0-9.
pub fn draw_label(canvas: &mut Canvas, x: u32, y: u32, text: &str, color: Color) {
    let mut offset_x = 0;

    for ch in text.chars() {
        if let Some(digit) = ch.to_digit(10) {
            let glyph = FONT_5X7[digit as usize];

            for (col, &column_data) in glyph.iter().enumerate() {
                for row in 0..8 {
                    if (column_data & (1 << row)) != 0 {
                        let px = x + offset_x + col as u32;
                        let py = y + row;
                        if px < canvas.width && py < canvas.height {
                            canvas.set_pixel(px, py, color);
                        }
                    }
                }
            }

            offset_x += 6; // 5 pixels + 1 space
        } else if ch == '.' {
            // Draw a dot for decimal point
            let px = x + offset_x;
            let py = y + 6;
            if px < canvas.width && py < canvas.height {
                canvas.set_pixel(px, py, color);
                canvas.set_pixel(px + 1, py, color);
            }
            offset_x += 3;
        } else if ch == '%' {
            // Skip for now
            offset_x += 4;
        } else if ch == ' ' {
            offset_x += 4;
        }
    }
}

/// Converts RGB to YCbCr (ITU-R BT.709).
///
/// Returns (Y, Cb, Cr) in range 0-255.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r_f = f32::from(r);
    let g_f = f32::from(g);
    let b_f = f32::from(b);

    // ITU-R BT.709 coefficients
    let y = 0.2126 * r_f + 0.7152 * g_f + 0.0722 * b_f;
    let cb = (b_f - y) / 1.8556 + 128.0;
    let cr = (r_f - y) / 1.5748 + 128.0;

    (
        y.clamp(0.0, 255.0) as u8,
        cb.clamp(0.0, 255.0) as u8,
        cr.clamp(0.0, 255.0) as u8,
    )
}

/// Converts YCbCr to RGB (ITU-R BT.709).
///
/// Returns (R, G, B) in range 0-255.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y_f = f32::from(y);
    let cb_f = f32::from(cb) - 128.0;
    let cr_f = f32::from(cr) - 128.0;

    // ITU-R BT.709 coefficients
    let r = y_f + 1.5748 * cr_f;
    let g = y_f - 0.1873 * cb_f - 0.4681 * cr_f;
    let b = y_f + 1.8556 * cb_f;

    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_new() {
        let canvas = Canvas::new(100, 100);
        assert_eq!(canvas.width, 100);
        assert_eq!(canvas.height, 100);
        assert_eq!(canvas.data.len(), 100 * 100 * 4);
    }

    #[test]
    fn test_canvas_set_get_pixel() {
        let mut canvas = Canvas::new(10, 10);
        canvas.set_pixel(5, 5, colors::RED);
        assert_eq!(canvas.get_pixel(5, 5), colors::RED);
    }

    #[test]
    fn test_canvas_blend_pixel() {
        let mut canvas = Canvas::new(10, 10);
        canvas.set_pixel(5, 5, colors::WHITE);
        canvas.blend_pixel(5, 5, [255, 0, 0, 128]); // 50% red
        let result = canvas.get_pixel(5, 5);
        assert!(result[0] > 200); // Should be reddish-white
    }

    #[test]
    fn test_rgb_to_ycbcr() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 255, 255);
        assert_eq!(y, 255);
        assert!((128_i16 - i16::from(cb)).abs() < 2);
        assert!((128_i16 - i16::from(cr)).abs() < 2);
    }

    #[test]
    fn test_ycbcr_to_rgb() {
        let (r, g, b) = ycbcr_to_rgb(255, 128, 128);
        assert!(r > 250);
        assert!(g > 250);
        assert!(b > 250);
    }

    #[test]
    fn test_rgb_ycbcr_roundtrip() {
        let original = (128u8, 64u8, 192u8);
        let (y, cb, cr) = rgb_to_ycbcr(original.0, original.1, original.2);
        let (r, g, b) = ycbcr_to_rgb(y, cb, cr);

        // Allow some error due to rounding (YCbCr conversion is lossy)
        assert!((i16::from(original.0) - i16::from(r)).abs() < 5);
        assert!((i16::from(original.1) - i16::from(g)).abs() < 5);
        assert!((i16::from(original.2) - i16::from(b)).abs() < 5);
    }
}

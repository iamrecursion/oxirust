//! Timecode burn-in overlay for broadcast monitoring outputs.
//!
//! Renders timecode strings (HH:MM:SS:FF) into a simple bitmap that can be
//! composited onto monitoring video feeds.  Supports configurable position,
//! font size, and colour.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Position on screen where the timecode overlay is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayPosition {
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Centre of the frame.
    Center,
}

/// RGBA colour (0-255 per channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba {
    /// Red component.
    pub r: u8,
    /// Green component.
    pub g: u8,
    /// Blue component.
    pub b: u8,
    /// Alpha component (255 = fully opaque).
    pub a: u8,
}

impl Rgba {
    /// Opaque white.
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    /// Opaque black.
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    /// Semi-transparent black (for background boxes).
    pub const SHADOW: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 160,
    };

    /// Create a new RGBA colour.
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

/// Configuration for the timecode overlay.
#[derive(Debug, Clone)]
pub struct TimecodeOverlayConfig {
    /// Where on the frame to place the overlay.
    pub position: OverlayPosition,
    /// Font size in pixels (height of a character cell).
    pub font_size: u32,
    /// Foreground (text) colour.
    pub fg_color: Rgba,
    /// Background box colour.
    pub bg_color: Rgba,
    /// Margin from the edge of the frame in pixels.
    pub margin: u32,
    /// Whether to draw a background box behind the text.
    pub draw_background: bool,
    /// Frame rate numerator (used for FF field).
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Whether timecode uses drop-frame notation.
    pub drop_frame: bool,
}

impl Default for TimecodeOverlayConfig {
    fn default() -> Self {
        Self {
            position: OverlayPosition::TopLeft,
            font_size: 24,
            fg_color: Rgba::WHITE,
            bg_color: Rgba::SHADOW,
            margin: 16,
            draw_background: true,
            fps_num: 25,
            fps_den: 1,
            drop_frame: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Timecode formatting
// ---------------------------------------------------------------------------

/// A timecode value (hours, minutes, seconds, frames).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timecode {
    /// Hours (0-23).
    pub hours: u8,
    /// Minutes (0-59).
    pub minutes: u8,
    /// Seconds (0-59).
    pub seconds: u8,
    /// Frames (0 .. fps-1).
    pub frames: u8,
    /// Whether this is drop-frame timecode.
    pub drop_frame: bool,
}

impl Timecode {
    /// Create a timecode value.
    pub const fn new(hours: u8, minutes: u8, seconds: u8, frames: u8, drop_frame: bool) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
            drop_frame,
        }
    }

    /// Format as `HH:MM:SS:FF` (or `HH:MM:SS;FF` for drop-frame).
    pub fn to_string_repr(&self) -> String {
        let sep = if self.drop_frame { ';' } else { ':' };
        format!(
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, sep, self.frames
        )
    }

    /// Convert a frame count to a timecode given the frames-per-second.
    pub fn from_frame_count(total_frames: u64, fps: u32) -> Self {
        if fps == 0 {
            return Self::new(0, 0, 0, 0, false);
        }
        let fps_u64 = u64::from(fps);
        let total_seconds = total_frames / fps_u64;
        let frames = (total_frames % fps_u64) as u8;
        let hours = (total_seconds / 3600) as u8;
        let minutes = ((total_seconds % 3600) / 60) as u8;
        let seconds = (total_seconds % 60) as u8;
        Self::new(hours, minutes, seconds, frames, false)
    }

    /// Convert this timecode back to a total frame count.
    pub fn to_frame_count(&self, fps: u32) -> u64 {
        let fps_u64 = u64::from(fps);
        let s =
            u64::from(self.hours) * 3600 + u64::from(self.minutes) * 60 + u64::from(self.seconds);
        s * fps_u64 + u64::from(self.frames)
    }
}

// ---------------------------------------------------------------------------
// Overlay Renderer
// ---------------------------------------------------------------------------

/// Rendered overlay bitmap (RGBA pixel buffer).
#[derive(Debug, Clone)]
pub struct OverlayBitmap {
    /// Width of the bitmap in pixels.
    pub width: u32,
    /// Height of the bitmap in pixels.
    pub height: u32,
    /// RGBA pixel data (row-major, 4 bytes per pixel).
    pub pixels: Vec<u8>,
}

/// Renders timecode overlays as RGBA bitmaps.
#[derive(Debug)]
pub struct TimecodeOverlayRenderer {
    config: TimecodeOverlayConfig,
}

impl TimecodeOverlayRenderer {
    /// Create a renderer with the given configuration.
    pub fn new(config: TimecodeOverlayConfig) -> Self {
        Self { config }
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &TimecodeOverlayConfig {
        &self.config
    }

    /// Render a timecode string into a small RGBA bitmap.
    ///
    /// The bitmap is sized to fit the text plus optional background padding.
    pub fn render(&self, tc: &Timecode) -> OverlayBitmap {
        let text = tc.to_string_repr();
        let char_w = self.config.font_size / 2;
        let char_h = self.config.font_size;
        let text_w = char_w * text.len() as u32;
        let pad = if self.config.draw_background { 4 } else { 0 };
        let bmp_w = text_w + pad * 2;
        let bmp_h = char_h + pad * 2;
        let mut pixels = vec![0u8; (bmp_w * bmp_h * 4) as usize];

        // Draw background box if enabled
        if self.config.draw_background {
            let bg = self.config.bg_color;
            for y in 0..bmp_h {
                for x in 0..bmp_w {
                    let off = ((y * bmp_w + x) * 4) as usize;
                    pixels[off] = bg.r;
                    pixels[off + 1] = bg.g;
                    pixels[off + 2] = bg.b;
                    pixels[off + 3] = bg.a;
                }
            }
        }

        // Render each character as a filled rectangle (placeholder rasteriser).
        let fg = self.config.fg_color;
        for (ci, _ch) in text.chars().enumerate() {
            let cx = pad + ci as u32 * char_w;
            let cy = pad;
            // Fill a slightly inset rectangle for visual distinctiveness.
            let inset = 1u32;
            for dy in inset..(char_h.saturating_sub(inset)) {
                for dx in inset..(char_w.saturating_sub(inset)) {
                    let px = cx + dx;
                    let py = cy + dy;
                    if px < bmp_w && py < bmp_h {
                        let off = ((py * bmp_w + px) * 4) as usize;
                        pixels[off] = fg.r;
                        pixels[off + 1] = fg.g;
                        pixels[off + 2] = fg.b;
                        pixels[off + 3] = fg.a;
                    }
                }
            }
        }

        OverlayBitmap {
            width: bmp_w,
            height: bmp_h,
            pixels,
        }
    }

    /// Compute the (x, y) offset for compositing the overlay onto a frame
    /// of the given dimensions.
    pub fn compute_offset(&self, frame_w: u32, frame_h: u32, bmp: &OverlayBitmap) -> (u32, u32) {
        let m = self.config.margin;
        match self.config.position {
            OverlayPosition::TopLeft => (m, m),
            OverlayPosition::TopRight => (frame_w.saturating_sub(bmp.width + m), m),
            OverlayPosition::BottomLeft => (m, frame_h.saturating_sub(bmp.height + m)),
            OverlayPosition::BottomRight => (
                frame_w.saturating_sub(bmp.width + m),
                frame_h.saturating_sub(bmp.height + m),
            ),
            OverlayPosition::Center => (
                (frame_w.saturating_sub(bmp.width)) / 2,
                (frame_h.saturating_sub(bmp.height)) / 2,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// 5×7 Bitmap Font
// ---------------------------------------------------------------------------

/// Glyph data for the 5-column × 7-row bitmap font.
///
/// Encoding: each glyph is 7 bytes; each byte represents one row.
/// Bit layout: the top 5 bits of each byte encode the 5 columns left-to-right.
/// Bit 7 (0x80) = column 0 (left); bit 3 (0x08) = column 4 (right).
/// The lower 3 bits are unused and zero.
///
/// Glyphs in order: '0'–'9', ':'
const FONT_5X7: [[u8; 7]; 11] = [
    // '0': oval shape
    [
        0b01110_000,
        0b10001_000,
        0b10011_000,
        0b10101_000,
        0b11001_000,
        0b10001_000,
        0b01110_000,
    ],
    // '1'
    [
        0b00100_000,
        0b01100_000,
        0b00100_000,
        0b00100_000,
        0b00100_000,
        0b00100_000,
        0b01110_000,
    ],
    // '2'
    [
        0b01110_000,
        0b10001_000,
        0b00001_000,
        0b00110_000,
        0b01000_000,
        0b10000_000,
        0b11111_000,
    ],
    // '3'
    [
        0b11110_000,
        0b00001_000,
        0b00001_000,
        0b01110_000,
        0b00001_000,
        0b00001_000,
        0b11110_000,
    ],
    // '4'
    [
        0b00010_000,
        0b00110_000,
        0b01010_000,
        0b10010_000,
        0b11111_000,
        0b00010_000,
        0b00010_000,
    ],
    // '5'
    [
        0b11111_000,
        0b10000_000,
        0b10000_000,
        0b11110_000,
        0b00001_000,
        0b00001_000,
        0b11110_000,
    ],
    // '6'
    [
        0b00110_000,
        0b01000_000,
        0b10000_000,
        0b11110_000,
        0b10001_000,
        0b10001_000,
        0b01110_000,
    ],
    // '7'
    [
        0b11111_000,
        0b00001_000,
        0b00010_000,
        0b00100_000,
        0b01000_000,
        0b01000_000,
        0b01000_000,
    ],
    // '8'
    [
        0b01110_000,
        0b10001_000,
        0b10001_000,
        0b01110_000,
        0b10001_000,
        0b10001_000,
        0b01110_000,
    ],
    // '9'
    [
        0b01110_000,
        0b10001_000,
        0b10001_000,
        0b01111_000,
        0b00001_000,
        0b00010_000,
        0b01100_000,
    ],
    // ':'  (two dots at rows 1-2 and 4-5)
    [
        0b00000_000,
        0b01100_000,
        0b01100_000,
        0b00000_000,
        0b01100_000,
        0b01100_000,
        0b00000_000,
    ],
];

/// Width in pixels of a single glyph at scale 1.
const GLYPH_W: u32 = 5;
/// Height in pixels of a single glyph at scale 1.
const GLYPH_H: u32 = 7;
/// Pixel gap between glyphs at scale 1.
const GLYPH_GAP: u32 = 1;

/// Map a character to its index in [`FONT_5X7`].
///
/// Digits '0'–'9' map to indices 0–9; ':' and ';' both map to index 10.
/// Returns `None` for unsupported characters (they are skipped silently).
fn char_to_glyph(c: char) -> Option<usize> {
    match c {
        '0'..='9' => Some(c as usize - '0' as usize),
        ':' | ';' => Some(10),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// TimecodeOverlay — pixel-level RGB burn-in renderer
// ---------------------------------------------------------------------------

/// Configuration for the pixel-level timecode burn-in overlay.
///
/// Unlike [`TimecodeOverlayConfig`] (which targets an RGBA compositor),
/// this configuration targets direct RGB (3 bytes/pixel) frame mutation.
#[derive(Debug, Clone)]
pub struct TimecodeOverlayPixelConfig {
    /// Top-left pixel X position of the overlay within the frame.
    pub x: u32,
    /// Top-left pixel Y position of the overlay within the frame.
    pub y: u32,
    /// Glyph scale factor (1 = 5×7 pixels per glyph, 2 = 10×14, …).
    /// Values less than 1 are treated as 1.
    pub scale: u32,
    /// Foreground colour as `[R, G, B]`. Default: white `[255, 255, 255]`.
    pub fg_color: [u8; 3],
    /// Optional background fill colour as `[R, G, B]`.
    ///
    /// `Some(color)` fills the entire text bounding box with `color` before
    /// drawing glyphs. `None` leaves background pixels untouched (transparent).
    ///
    /// Default: `Some([0, 0, 0])` (opaque black).
    pub bg_color: Option<[u8; 3]>,
}

impl Default for TimecodeOverlayPixelConfig {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            scale: 2,
            fg_color: [255, 255, 255],
            bg_color: Some([0, 0, 0]),
        }
    }
}

/// Pixel-level timecode burn-in renderer for 24-bit RGB frames.
///
/// Renders a timecode string (e.g. `"HH:MM:SS:FF"`) directly into a
/// flat `&mut [u8]` row-major RGB buffer using a built-in 5×7 bitmap
/// font. Rendering is clipped to frame bounds — no out-of-bounds panic.
///
/// # Example
///
/// ```
/// use oximedia_playout::timecode_overlay::{TimecodeOverlay, TimecodeOverlayPixelConfig};
/// let cfg = TimecodeOverlayPixelConfig::default();
/// let overlay = TimecodeOverlay::new(cfg);
/// let (w, h) = (200u32, 30u32);
/// let mut frame = vec![0u8; (w * h * 3) as usize];
/// overlay.render(&mut frame, w, h, "01:23:45:12");
/// ```
#[derive(Debug)]
pub struct TimecodeOverlay {
    config: TimecodeOverlayPixelConfig,
}

impl TimecodeOverlay {
    /// Create a new renderer with the given configuration.
    pub fn new(config: TimecodeOverlayPixelConfig) -> Self {
        Self { config }
    }

    /// Render `timecode` onto `frame` in-place.
    ///
    /// `frame` must be a row-major RGB buffer with exactly `w * h * 3` bytes.
    /// Characters not in the font (anything other than `'0'–'9'`, `':'`, `';'`)
    /// are skipped silently, so drop-frame semicolons render as colons.
    pub fn render(&self, frame: &mut [u8], w: u32, h: u32, timecode: &str) {
        let scale = self.config.scale.max(1);
        let glyph_w = GLYPH_W * scale;
        let glyph_h = GLYPH_H * scale;
        let gap = GLYPH_GAP * scale;

        // Collect renderable glyph indices in order.
        let glyphs: Vec<usize> = timecode.chars().filter_map(char_to_glyph).collect();
        let n = glyphs.len() as u32;

        // Optional background fill across the full text bounding box.
        if let Some(bg) = self.config.bg_color {
            let text_px_w = if n > 0 {
                n * glyph_w + (n - 1) * gap
            } else {
                0
            };
            self.fill_rect(
                frame,
                w,
                h,
                self.config.x,
                self.config.y,
                text_px_w,
                glyph_h,
                bg,
            );
        }

        // Render each glyph left-to-right.
        let mut cursor_x = self.config.x;
        for glyph_idx in &glyphs {
            self.render_glyph(frame, w, h, cursor_x, self.config.y, *glyph_idx, scale);
            cursor_x = cursor_x.saturating_add(glyph_w + gap);
        }
    }

    /// Render a single glyph from [`FONT_5X7`] at position `(ox, oy)`.
    fn render_glyph(
        &self,
        frame: &mut [u8],
        w: u32,
        h: u32,
        ox: u32,
        oy: u32,
        glyph_idx: usize,
        scale: u32,
    ) {
        let rows = &FONT_5X7[glyph_idx];
        let fg = self.config.fg_color;

        for (row_idx, &row_bits) in rows.iter().enumerate() {
            for col_idx in 0u32..GLYPH_W {
                // Bit position: column 0 = bit 7, column 4 = bit 3.
                // Row byte layout: [col0 col1 col2 col3 col4 _ _ _]
                // So column `c` is at bit position `7 - c`.
                let bit_shift = 7u32.saturating_sub(col_idx);
                let lit = (row_bits >> bit_shift) & 1 == 1;
                if !lit {
                    continue;
                }
                // Scale: fill a scale×scale pixel block.
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = ox + col_idx * scale + sx;
                        let py = oy + row_idx as u32 * scale + sy;
                        if px < w && py < h {
                            let off = (py * w + px) as usize * 3;
                            if off + 2 < frame.len() {
                                frame[off] = fg[0];
                                frame[off + 1] = fg[1];
                                frame[off + 2] = fg[2];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Fill a rectangle `(x, y, rw, rh)` with `color` in an RGB frame.
    fn fill_rect(
        &self,
        frame: &mut [u8],
        w: u32,
        h: u32,
        x: u32,
        y: u32,
        rw: u32,
        rh: u32,
        color: [u8; 3],
    ) {
        for dy in 0..rh {
            for dx in 0..rw {
                let px = x + dx;
                let py = y + dy;
                if px < w && py < h {
                    let off = (py * w + px) as usize * 3;
                    if off + 2 < frame.len() {
                        frame[off] = color[0];
                        frame[off + 1] = color[1];
                        frame[off + 2] = color[2];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_format_ndf() {
        let tc = Timecode::new(1, 2, 3, 4, false);
        assert_eq!(tc.to_string_repr(), "01:02:03:04");
    }

    #[test]
    fn test_timecode_format_df() {
        let tc = Timecode::new(10, 30, 59, 29, true);
        assert_eq!(tc.to_string_repr(), "10:30:59;29");
    }

    #[test]
    fn test_from_frame_count() {
        // 25 fps: 90_000 frames = 1 hour
        let tc = Timecode::from_frame_count(90_000, 25);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 0);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_to_frame_count_roundtrip() {
        let tc = Timecode::new(0, 1, 30, 12, false);
        let count = tc.to_frame_count(25);
        let tc2 = Timecode::from_frame_count(count, 25);
        assert_eq!(tc.hours, tc2.hours);
        assert_eq!(tc.minutes, tc2.minutes);
        assert_eq!(tc.seconds, tc2.seconds);
        assert_eq!(tc.frames, tc2.frames);
    }

    #[test]
    fn test_from_frame_count_zero_fps() {
        let tc = Timecode::from_frame_count(100, 0);
        assert_eq!(tc.hours, 0);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_overlay_default_config() {
        let cfg = TimecodeOverlayConfig::default();
        assert_eq!(cfg.position, OverlayPosition::TopLeft);
        assert_eq!(cfg.font_size, 24);
        assert!(cfg.draw_background);
    }

    #[test]
    fn test_render_bitmap_dimensions() {
        let r = TimecodeOverlayRenderer::new(TimecodeOverlayConfig::default());
        let tc = Timecode::new(0, 0, 0, 0, false);
        let bmp = r.render(&tc);
        // "00:00:00:00" = 11 chars, char_w=12, pad=4 => 11*12 + 8 = 140
        assert!(bmp.width > 0);
        assert!(bmp.height > 0);
        assert_eq!(bmp.pixels.len(), (bmp.width * bmp.height * 4) as usize);
    }

    #[test]
    fn test_render_no_background() {
        let cfg = TimecodeOverlayConfig {
            draw_background: false,
            ..TimecodeOverlayConfig::default()
        };
        let r = TimecodeOverlayRenderer::new(cfg);
        let tc = Timecode::new(12, 0, 0, 0, false);
        let bmp = r.render(&tc);
        assert!(bmp.width > 0);
    }

    #[test]
    fn test_compute_offset_top_left() {
        let r = TimecodeOverlayRenderer::new(TimecodeOverlayConfig::default());
        let bmp = OverlayBitmap {
            width: 100,
            height: 30,
            pixels: vec![],
        };
        let (x, y) = r.compute_offset(1920, 1080, &bmp);
        assert_eq!(x, 16); // margin
        assert_eq!(y, 16);
    }

    #[test]
    fn test_compute_offset_bottom_right() {
        let cfg = TimecodeOverlayConfig {
            position: OverlayPosition::BottomRight,
            margin: 10,
            ..TimecodeOverlayConfig::default()
        };
        let r = TimecodeOverlayRenderer::new(cfg);
        let bmp = OverlayBitmap {
            width: 100,
            height: 30,
            pixels: vec![],
        };
        let (x, y) = r.compute_offset(1920, 1080, &bmp);
        assert_eq!(x, 1920 - 100 - 10);
        assert_eq!(y, 1080 - 30 - 10);
    }

    #[test]
    fn test_compute_offset_center() {
        let cfg = TimecodeOverlayConfig {
            position: OverlayPosition::Center,
            ..TimecodeOverlayConfig::default()
        };
        let r = TimecodeOverlayRenderer::new(cfg);
        let bmp = OverlayBitmap {
            width: 100,
            height: 30,
            pixels: vec![],
        };
        let (x, y) = r.compute_offset(1920, 1080, &bmp);
        assert_eq!(x, (1920 - 100) / 2);
        assert_eq!(y, (1080 - 30) / 2);
    }

    #[test]
    fn test_rgba_constants() {
        assert_eq!(Rgba::WHITE, Rgba::new(255, 255, 255, 255));
        assert_eq!(Rgba::BLACK, Rgba::new(0, 0, 0, 255));
    }

    #[test]
    fn test_timecode_equality() {
        let a = Timecode::new(1, 2, 3, 4, false);
        let b = Timecode::new(1, 2, 3, 4, false);
        assert_eq!(a, b);
    }

    // --- TimecodeOverlay pixel-renderer tests ---

    #[test]
    fn test_timecode_overlay_renders() {
        // 200×20 black RGB frame; default config renders white glyphs.
        let w = 200u32;
        let h = 20u32;
        let mut frame = vec![0u8; (w * h * 3) as usize];
        let overlay = TimecodeOverlay::new(TimecodeOverlayPixelConfig::default());
        overlay.render(&mut frame, w, h, "00:00:00:00");
        // At least one white pixel must appear after rendering.
        let has_white = frame.chunks(3).any(|px| px == [255u8, 255, 255]);
        assert!(has_white, "expected white pixels after rendering timecode");
    }

    #[test]
    fn test_timecode_overlay_bounds() {
        // Very small frame — rendering must not panic even when glyphs overflow.
        let w = 20u32;
        let h = 10u32;
        let mut frame = vec![0u8; (w * h * 3) as usize];
        let overlay = TimecodeOverlay::new(TimecodeOverlayPixelConfig::default());
        // Must complete without panic regardless of frame size vs text width.
        overlay.render(&mut frame, w, h, "00:00:00:00");
    }

    #[test]
    fn test_timecode_overlay_transparent_bg() {
        let w = 200u32;
        let h = 20u32;
        // Grey frame (128,128,128) — bg is transparent so grey must survive.
        let mut frame = vec![128u8; (w * h * 3) as usize];
        let cfg = TimecodeOverlayPixelConfig {
            x: 0,
            y: 0,
            scale: 2,
            fg_color: [255, 0, 0], // red foreground
            bg_color: None,        // transparent — no background fill
        };
        let overlay = TimecodeOverlay::new(cfg);
        overlay.render(&mut frame, w, h, "00:00:00:00");
        // Grey background pixels must still exist (not overwritten by bg fill).
        let has_grey = frame.chunks(3).any(|px| px == [128u8, 128, 128]);
        assert!(
            has_grey,
            "grey background pixels should be preserved with bg_color=None"
        );
        // Red foreground pixels (from lit glyph bits) must appear.
        let has_red = frame.chunks(3).any(|px| px == [255u8, 0, 0]);
        assert!(
            has_red,
            "red foreground pixels should appear from glyph rendering"
        );
    }

    #[test]
    fn test_timecode_overlay_default_config() {
        let cfg = TimecodeOverlayPixelConfig::default();
        assert_eq!(cfg.scale, 2);
        assert_eq!(cfg.fg_color, [255, 255, 255]);
        assert_eq!(cfg.bg_color, Some([0, 0, 0]));
    }

    #[test]
    fn test_char_to_glyph_digits() {
        for (c, expected) in ('0'..='9').enumerate() {
            assert_eq!(char_to_glyph(expected), Some(c));
        }
        assert_eq!(char_to_glyph(':'), Some(10));
        assert_eq!(char_to_glyph(';'), Some(10));
        assert_eq!(char_to_glyph('X'), None);
    }
}

//! Title and text overlay generation for timeline editing.
//!
//! Renders styled text overlays onto video frames using a built-in
//! pixel-font rasteriser (no external font libraries required).
//! The font uses a 5×7 bitmap glyph table for ASCII printable characters.

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Glyph bitmap table (5 columns × 7 rows per glyph, ASCII 32–126)
// ─────────────────────────────────────────────────────────────────────────────

/// 5×7 pixel bitmap for ASCII printable characters (space = 0x20 through '~' = 0x7E).
/// Each glyph is stored as 7 rows of 5 bits packed into a `[u8; 7]`.
/// Bit 4 (MSB of the low nibble) is the leftmost pixel.
const FONT5X7: [[u8; 7]; 95] = [
    // Space (0x20)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // ! (0x21)
    [0x04, 0x04, 0x04, 0x04, 0x00, 0x04, 0x00],
    // " (0x22)
    [0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00],
    // # (0x23)
    [0x0A, 0x1F, 0x0A, 0x0A, 0x1F, 0x0A, 0x00],
    // $ (0x24)
    [0x04, 0x0F, 0x14, 0x0E, 0x05, 0x1E, 0x04],
    // % (0x25)
    [0x18, 0x19, 0x02, 0x04, 0x13, 0x03, 0x00],
    // & (0x26)
    [0x08, 0x14, 0x08, 0x15, 0x12, 0x0D, 0x00],
    // ' (0x27)
    [0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00],
    // ( (0x28)
    [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
    // ) (0x29)
    [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
    // * (0x2A)
    [0x00, 0x04, 0x15, 0x0E, 0x15, 0x04, 0x00],
    // + (0x2B)
    [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
    // , (0x2C)
    [0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08],
    // - (0x2D)
    [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
    // . (0x2E)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00],
    // / (0x2F)
    [0x01, 0x02, 0x02, 0x04, 0x08, 0x10, 0x00],
    // 0 (0x30)
    [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
    // 1 (0x31)
    [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
    // 2 (0x32)
    [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
    // 3 (0x33)
    [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
    // 4 (0x34)
    [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
    // 5 (0x35)
    [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
    // 6 (0x36)
    [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
    // 7 (0x37)
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
    // 8 (0x38)
    [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
    // 9 (0x39)
    [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
    // : (0x3A)
    [0x00, 0x04, 0x00, 0x00, 0x04, 0x00, 0x00],
    // ; (0x3B)
    [0x00, 0x04, 0x00, 0x00, 0x04, 0x04, 0x08],
    // < (0x3C)
    [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
    // = (0x3D)
    [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
    // > (0x3E)
    [0x10, 0x08, 0x04, 0x02, 0x04, 0x08, 0x10],
    // ? (0x3F)
    [0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04],
    // @ (0x40)
    [0x0E, 0x11, 0x01, 0x0D, 0x15, 0x15, 0x0E],
    // A (0x41)
    [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    // B (0x42)
    [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
    // C (0x43)
    [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
    // D (0x44)
    [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
    // E (0x45)
    [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
    // F (0x46)
    [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
    // G (0x47)
    [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
    // H (0x48)
    [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    // I (0x49)
    [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
    // J (0x4A)
    [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
    // K (0x4B)
    [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
    // L (0x4C)
    [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
    // M (0x4D)
    [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
    // N (0x4E)
    [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
    // O (0x4F)
    [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    // P (0x50)
    [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
    // Q (0x51)
    [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
    // R (0x52)
    [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
    // S (0x53)
    [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
    // T (0x54)
    [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
    // U (0x55)
    [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    // V (0x56)
    [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
    // W (0x57)
    [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
    // X (0x58)
    [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
    // Y (0x59)
    [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
    // Z (0x5A)
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
    // [ (0x5B)
    [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E],
    // \ (0x5C)
    [0x10, 0x08, 0x08, 0x04, 0x02, 0x01, 0x00],
    // ] (0x5D)
    [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E],
    // ^ (0x5E)
    [0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00],
    // _ (0x5F)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
    // ` (0x60)
    [0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00],
    // a (0x61)
    [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F],
    // b (0x62)
    [0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E],
    // c (0x63)
    [0x00, 0x00, 0x0E, 0x10, 0x10, 0x10, 0x0E],
    // d (0x64)
    [0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F],
    // e (0x65)
    [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
    // f (0x66)
    [0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08],
    // g (0x67)
    [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E],
    // h (0x68)
    [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11],
    // i (0x69)
    [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
    // j (0x6A)
    [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],
    // k (0x6B)
    [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
    // l (0x6C)
    [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
    // m (0x6D)
    [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11],
    // n (0x6E)
    [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],
    // o (0x6F)
    [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],
    // p (0x70)
    [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10],
    // q (0x71)
    [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x01],
    // r (0x72)
    [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
    // s (0x73)
    [0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E],
    // t (0x74)
    [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06],
    // u (0x75)
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],
    // v (0x76)
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],
    // w (0x77)
    [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],
    // x (0x78)
    [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],
    // y (0x79)
    [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E],
    // z (0x7A)
    [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],
    // { (0x7B)
    [0x03, 0x04, 0x04, 0x18, 0x04, 0x04, 0x03],
    // | (0x7C)
    [0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x04],
    // } (0x7D)
    [0x18, 0x04, 0x04, 0x03, 0x04, 0x04, 0x18],
    // ~ (0x7E)
    [0x00, 0x08, 0x15, 0x02, 0x00, 0x00, 0x00],
];

/// Width of a single glyph in pixels.
pub const GLYPH_W: u32 = 5;
/// Height of a single glyph in pixels.
pub const GLYPH_H: u32 = 7;
/// Gap between glyphs in pixels.
pub const GLYPH_GAP: u32 = 1;

/// Return the 5×7 bitmap for `ch`, or the bitmap for '?' if out of range.
fn glyph(ch: char) -> [u8; 7] {
    let code = ch as u32;
    if code >= 0x20 && code <= 0x7E {
        FONT5X7[(code - 0x20) as usize]
    } else {
        FONT5X7[('?' as u32 - 0x20) as usize]
    }
}

/// Returns `true` if pixel `(col, row)` in a 5×7 glyph bitmap is set.
/// `col` is in `0..5`, `row` is in `0..7`.
fn glyph_pixel(bitmap: [u8; 7], col: u32, row: u32) -> bool {
    if col >= GLYPH_W || row >= GLYPH_H {
        return false;
    }
    // Bit 4 is leftmost; bit 0 is rightmost.
    (bitmap[row as usize] >> (4 - col)) & 1 == 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Text style
// ─────────────────────────────────────────────────────────────────────────────

/// RGBA colour (each component 0–255).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel (0 = transparent, 255 = opaque).
    pub a: u8,
}

impl Rgba {
    /// Construct an RGBA colour.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Opaque white.
    #[must_use]
    pub const fn white() -> Self {
        Self::new(255, 255, 255, 255)
    }

    /// Opaque black.
    #[must_use]
    pub const fn black() -> Self {
        Self::new(0, 0, 0, 255)
    }

    /// Transparent.
    #[must_use]
    pub const fn transparent() -> Self {
        Self::new(0, 0, 0, 0)
    }
}

/// Horizontal alignment for text within the overlay bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HAlign {
    /// Left-aligned.
    Left,
    /// Centred.
    Center,
    /// Right-aligned.
    Right,
}

/// Vertical alignment for text within the overlay bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VAlign {
    /// Top-aligned.
    Top,
    /// Vertically centred.
    Middle,
    /// Bottom-aligned.
    Bottom,
}

/// Scale factor for the built-in pixel font.
///
/// Each glyph pixel is rendered as an `n×n` block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontScale {
    /// 1× (original 5×7 pixels).
    One,
    /// 2× (10×14 pixels per glyph).
    Two,
    /// 3× (15×21 pixels per glyph).
    Three,
    /// 4× (20×28 pixels per glyph).
    Four,
}

impl FontScale {
    /// Pixel multiplier.
    #[must_use]
    pub fn factor(self) -> u32 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
        }
    }
}

/// Full style description for an overlay.
#[derive(Debug, Clone)]
pub struct TextStyle {
    /// Foreground (text) colour.
    pub color: Rgba,
    /// Background colour (drawn behind the bounding box if `Some`).
    pub background: Option<Rgba>,
    /// Drop-shadow colour (drawn 1 pixel down-right if `Some`).
    pub shadow: Option<Rgba>,
    /// Font scale.
    pub scale: FontScale,
    /// Horizontal text alignment within the bounding box.
    pub halign: HAlign,
    /// Vertical text alignment within the bounding box.
    pub valign: VAlign,
    /// Padding in pixels added around the text.
    pub padding: u32,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            color: Rgba::white(),
            background: None,
            shadow: Some(Rgba::new(0, 0, 0, 180)),
            scale: FontScale::Two,
            halign: HAlign::Center,
            valign: VAlign::Bottom,
            padding: 8,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TitleOverlay
// ─────────────────────────────────────────────────────────────────────────────

/// Identifier for a title overlay.
pub type OverlayId = u64;

/// A single title/text element that can be composited over a video frame.
///
/// The overlay stores the text content, position, style, and optional
/// keyframes for animated position or opacity.
#[derive(Debug, Clone)]
pub struct TitleOverlay {
    /// Unique overlay identifier.
    pub id: OverlayId,
    /// Text to render.
    pub text: String,
    /// Top-left X position of the bounding box in pixels.
    pub x: i32,
    /// Top-left Y position of the bounding box in pixels.
    pub y: i32,
    /// Bounding box width (0 = auto-size).
    pub width: u32,
    /// Bounding box height (0 = auto-size).
    pub height: u32,
    /// Visual style.
    pub style: TextStyle,
    /// Timeline start position (in timebase units).
    pub start: i64,
    /// Timeline end position (in timebase units).
    pub end: i64,
    /// Opacity (0.0–1.0), applied on top of the style colour alpha.
    pub opacity: f32,
    /// Position keyframes: `(timeline_pos, x, y)` sorted by position.
    pub position_keyframes: Vec<(i64, i32, i32)>,
}

impl TitleOverlay {
    /// Create a new overlay with default style.
    #[must_use]
    pub fn new(
        id: OverlayId,
        text: impl Into<String>,
        x: i32,
        y: i32,
        start: i64,
        end: i64,
    ) -> Self {
        Self {
            id,
            text: text.into(),
            x,
            y,
            width: 0,
            height: 0,
            style: TextStyle::default(),
            start,
            end,
            opacity: 1.0,
            position_keyframes: Vec::new(),
        }
    }

    /// Builder: set bounding box dimensions.
    #[must_use]
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Builder: set style.
    #[must_use]
    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }

    /// Builder: set opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Add a position keyframe.
    pub fn add_position_keyframe(&mut self, time: i64, x: i32, y: i32) {
        self.position_keyframes.push((time, x, y));
        self.position_keyframes.sort_by_key(|&(t, _, _)| t);
    }

    /// Compute interpolated `(x, y)` at `time` using the keyframe list.
    ///
    /// If there are no keyframes the overlay's static `x`/`y` are returned.
    /// If `time` is before the first keyframe, the first keyframe's position
    /// is returned.  If after the last, the last keyframe's position is returned.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn position_at(&self, time: i64) -> (i32, i32) {
        if self.position_keyframes.is_empty() {
            return (self.x, self.y);
        }
        let kf = &self.position_keyframes;
        if time <= kf[0].0 {
            return (kf[0].1, kf[0].2);
        }
        let last = kf[kf.len() - 1];
        if time >= last.0 {
            return (last.1, last.2);
        }
        // Linear interpolation between surrounding keyframes.
        let idx = kf.partition_point(|&(t, _, _)| t <= time) - 1;
        let (t0, x0, y0) = kf[idx];
        let (t1, x1, y1) = kf[idx + 1];
        let span = (t1 - t0) as f64;
        let alpha = if span > 0.0 {
            (time - t0) as f64 / span
        } else {
            0.0
        };
        let ix = (x0 as f64 + alpha * (x1 - x0) as f64).round() as i32;
        let iy = (y0 as f64 + alpha * (y1 - y0) as f64).round() as i32;
        (ix, iy)
    }

    /// Returns `true` if the overlay is active at `time`.
    #[must_use]
    pub fn is_active_at(&self, time: i64) -> bool {
        time >= self.start && time < self.end
    }

    /// Compute the rendered text width in pixels (before scale).
    #[must_use]
    pub fn text_width_px(&self) -> u32 {
        let n = self.text.chars().count() as u32;
        if n == 0 {
            0
        } else {
            n * (GLYPH_W + GLYPH_GAP) - GLYPH_GAP
        }
    }

    /// Compute the rendered text height in pixels (always `GLYPH_H`).
    #[must_use]
    pub fn text_height_px(&self) -> u32 {
        GLYPH_H
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OverlayRenderer
// ─────────────────────────────────────────────────────────────────────────────

/// Renders [`TitleOverlay`]s into an RGBA pixel buffer.
///
/// The buffer is row-major with 4 bytes per pixel (R, G, B, A).
/// (0, 0) is the top-left corner.
pub struct OverlayRenderer {
    /// Target frame width in pixels.
    pub frame_width: u32,
    /// Target frame height in pixels.
    pub frame_height: u32,
}

impl OverlayRenderer {
    /// Create a renderer for frames of the given dimensions.
    #[must_use]
    pub fn new(frame_width: u32, frame_height: u32) -> Self {
        Self {
            frame_width,
            frame_height,
        }
    }

    /// Composite `overlay` onto the RGBA `buffer` at `time`.
    ///
    /// `buffer` must have length `frame_width * frame_height * 4`.
    /// Nothing is written when the overlay is not active at `time`.
    pub fn composite(&self, buffer: &mut [u8], overlay: &TitleOverlay, time: i64) {
        if !overlay.is_active_at(time) {
            return;
        }
        let scale = overlay.style.scale.factor();
        let glyph_w_s = (GLYPH_W + GLYPH_GAP) * scale;
        let glyph_h_s = GLYPH_H * scale;
        let text_w = overlay.text_width_px() * scale;
        let text_h = glyph_h_s;

        let (ox, oy) = overlay.position_at(time);
        let padding = overlay.style.padding;

        let bbox_w = if overlay.width > 0 {
            overlay.width
        } else {
            text_w + padding * 2
        };
        let bbox_h = if overlay.height > 0 {
            overlay.height
        } else {
            text_h + padding * 2
        };

        // Draw background if requested.
        if let Some(bg) = overlay.style.background {
            self.fill_rect(buffer, ox, oy, bbox_w, bbox_h, bg, overlay.opacity);
        }

        // Compute text start position within bounding box.
        let text_start_x = match overlay.style.halign {
            HAlign::Left => ox + padding as i32,
            HAlign::Center => ox + (bbox_w.saturating_sub(text_w) / 2) as i32,
            HAlign::Right => ox + bbox_w as i32 - text_w as i32 - padding as i32,
        };
        let text_start_y = match overlay.style.valign {
            VAlign::Top => oy + padding as i32,
            VAlign::Middle => oy + (bbox_h.saturating_sub(text_h) / 2) as i32,
            VAlign::Bottom => oy + bbox_h as i32 - text_h as i32 - padding as i32,
        };

        // Render each character.
        let mut cursor_x = text_start_x;
        for ch in overlay.text.chars() {
            let bitmap = glyph(ch);
            // Shadow.
            if let Some(shadow) = overlay.style.shadow {
                self.draw_glyph(
                    buffer,
                    bitmap,
                    cursor_x + 1,
                    text_start_y + 1,
                    scale,
                    shadow,
                    overlay.opacity,
                );
            }
            // Foreground.
            self.draw_glyph(
                buffer,
                bitmap,
                cursor_x,
                text_start_y,
                scale,
                overlay.style.color,
                overlay.opacity,
            );
            cursor_x += glyph_w_s as i32;
        }
    }

    /// Draw a filled rectangle.
    fn fill_rect(
        &self,
        buffer: &mut [u8],
        x: i32,
        y: i32,
        w: u32,
        h: u32,
        color: Rgba,
        opacity: f32,
    ) {
        for row in 0..h {
            for col in 0..w {
                let px = x + col as i32;
                let py = y + row as i32;
                self.blend_pixel(buffer, px, py, color, opacity);
            }
        }
    }

    /// Draw a single scaled glyph.
    fn draw_glyph(
        &self,
        buffer: &mut [u8],
        bitmap: [u8; 7],
        x: i32,
        y: i32,
        scale: u32,
        color: Rgba,
        opacity: f32,
    ) {
        for row in 0..GLYPH_H {
            for col in 0..GLYPH_W {
                if !glyph_pixel(bitmap, col, row) {
                    continue;
                }
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = x + (col * scale + sx) as i32;
                        let py = y + (row * scale + sy) as i32;
                        self.blend_pixel(buffer, px, py, color, opacity);
                    }
                }
            }
        }
    }

    /// Alpha-composite `color` with `opacity` onto `buffer` at pixel `(x, y)`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn blend_pixel(&self, buffer: &mut [u8], x: i32, y: i32, color: Rgba, opacity: f32) {
        if x < 0 || y < 0 {
            return;
        }
        let px = x as u32;
        let py = y as u32;
        if px >= self.frame_width || py >= self.frame_height {
            return;
        }
        let idx = ((py * self.frame_width + px) * 4) as usize;
        if idx + 3 >= buffer.len() {
            return;
        }
        let src_a = (color.a as f32 / 255.0) * opacity;
        let dst_a = 1.0 - src_a;
        buffer[idx] = (color.r as f32 * src_a + buffer[idx] as f32 * dst_a).round() as u8;
        buffer[idx + 1] = (color.g as f32 * src_a + buffer[idx + 1] as f32 * dst_a).round() as u8;
        buffer[idx + 2] = (color.b as f32 * src_a + buffer[idx + 2] as f32 * dst_a).round() as u8;
        buffer[idx + 3] = (src_a * 255.0 + buffer[idx + 3] as f32 * dst_a).round() as u8;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OverlayManager
// ─────────────────────────────────────────────────────────────────────────────

/// Manages a collection of title overlays for a project.
#[derive(Debug, Default)]
pub struct OverlayManager {
    overlays: HashMap<OverlayId, TitleOverlay>,
    next_id: OverlayId,
}

impl OverlayManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            overlays: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add an overlay and return its assigned ID.
    pub fn add(&mut self, mut overlay: TitleOverlay) -> OverlayId {
        let id = self.next_id;
        self.next_id += 1;
        overlay.id = id;
        self.overlays.insert(id, overlay);
        id
    }

    /// Remove an overlay by ID.
    pub fn remove(&mut self, id: OverlayId) -> Option<TitleOverlay> {
        self.overlays.remove(&id)
    }

    /// Get a reference to an overlay.
    #[must_use]
    pub fn get(&self, id: OverlayId) -> Option<&TitleOverlay> {
        self.overlays.get(&id)
    }

    /// Get a mutable reference.
    pub fn get_mut(&mut self, id: OverlayId) -> Option<&mut TitleOverlay> {
        self.overlays.get_mut(&id)
    }

    /// Return all overlays active at `time`, sorted by ID for deterministic order.
    #[must_use]
    pub fn active_at(&self, time: i64) -> Vec<&TitleOverlay> {
        let mut active: Vec<&TitleOverlay> = self
            .overlays
            .values()
            .filter(|o| o.is_active_at(time))
            .collect();
        active.sort_by_key(|o| o.id);
        active
    }

    /// Total overlay count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.overlays.len()
    }

    /// Returns `true` if there are no overlays.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.overlays.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glyph_pixel_space_is_blank() {
        let bm = glyph(' ');
        for row in 0..GLYPH_H {
            for col in 0..GLYPH_W {
                assert!(!glyph_pixel(bm, col, row), "space should be blank");
            }
        }
    }

    #[test]
    fn test_glyph_pixel_uppercase_a_has_pixels() {
        let bm = glyph('A');
        let set_count: u32 = (0..GLYPH_H)
            .flat_map(|row| (0..GLYPH_W).map(move |col| (row, col)))
            .filter(|&(r, c)| glyph_pixel(bm, c, r))
            .count() as u32;
        assert!(
            set_count > 5,
            "A should have many set pixels, got {set_count}"
        );
    }

    #[test]
    fn test_glyph_unknown_char_returns_question_mark() {
        let bm_q = glyph('?');
        let bm_unknown = glyph('\x01');
        assert_eq!(bm_q, bm_unknown);
    }

    #[test]
    fn test_title_overlay_is_active_at() {
        let o = TitleOverlay::new(1, "Hello", 0, 0, 1000, 5000);
        assert!(o.is_active_at(1000));
        assert!(o.is_active_at(4999));
        assert!(!o.is_active_at(5000));
        assert!(!o.is_active_at(999));
    }

    #[test]
    fn test_title_overlay_text_width() {
        let o = TitleOverlay::new(1, "AB", 0, 0, 0, 1000);
        // 2 chars × (5 + 1) - 1 = 11 pixels
        assert_eq!(o.text_width_px(), 11);
    }

    #[test]
    fn test_title_overlay_text_width_empty() {
        let o = TitleOverlay::new(1, "", 0, 0, 0, 1000);
        assert_eq!(o.text_width_px(), 0);
    }

    #[test]
    fn test_position_keyframes_interpolation() {
        let mut o = TitleOverlay::new(1, "Hi", 0, 0, 0, 10000);
        o.add_position_keyframe(0, 0, 100);
        o.add_position_keyframe(1000, 200, 100);
        let (x, y) = o.position_at(500);
        assert_eq!(x, 100);
        assert_eq!(y, 100);
    }

    #[test]
    fn test_position_keyframes_before_first() {
        let mut o = TitleOverlay::new(1, "Hi", 0, 0, 0, 10000);
        o.add_position_keyframe(500, 10, 20);
        let pos = o.position_at(0);
        assert_eq!(pos, (10, 20));
    }

    #[test]
    fn test_position_no_keyframes_returns_static() {
        let o = TitleOverlay::new(1, "Hi", 42, 99, 0, 1000);
        assert_eq!(o.position_at(500), (42, 99));
    }

    #[test]
    fn test_overlay_renderer_composites_white_pixel() {
        let width = 32u32;
        let height = 32u32;
        let mut buf = vec![0u8; (width * height * 4) as usize];
        let renderer = OverlayRenderer::new(width, height);
        let overlay = TitleOverlay::new(1, "A", 0, 0, 0, 1000).with_style(TextStyle {
            color: Rgba::white(),
            background: None,
            shadow: None,
            scale: FontScale::One,
            padding: 0,
            ..TextStyle::default()
        });
        renderer.composite(&mut buf, &overlay, 0);
        // At least some pixels should be non-zero (letter A is set).
        let any_set = buf.iter().any(|&b| b > 0);
        assert!(any_set, "Expected rendered pixels after compositing 'A'");
    }

    #[test]
    fn test_overlay_renderer_inactive_not_rendered() {
        let width = 32u32;
        let height = 32u32;
        let mut buf = vec![0u8; (width * height * 4) as usize];
        let renderer = OverlayRenderer::new(width, height);
        let overlay = TitleOverlay::new(1, "A", 0, 0, 5000, 10000);
        renderer.composite(&mut buf, &overlay, 0);
        // Buffer should remain all zeros.
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_overlay_manager_add_remove() {
        let mut mgr = OverlayManager::new();
        let o = TitleOverlay::new(0, "Test", 0, 0, 0, 1000);
        let id = mgr.add(o);
        assert_eq!(mgr.len(), 1);
        assert!(mgr.get(id).is_some());
        assert!(mgr.remove(id).is_some());
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_overlay_manager_active_at() {
        let mut mgr = OverlayManager::new();
        let o1 = TitleOverlay::new(0, "A", 0, 0, 0, 1000);
        let o2 = TitleOverlay::new(0, "B", 0, 50, 500, 2000);
        mgr.add(o1);
        mgr.add(o2);
        let active = mgr.active_at(600);
        assert_eq!(active.len(), 2);
        let active_early = mgr.active_at(200);
        assert_eq!(active_early.len(), 1);
    }

    #[test]
    fn test_rgba_white_black() {
        let w = Rgba::white();
        assert_eq!((w.r, w.g, w.b, w.a), (255, 255, 255, 255));
        let b = Rgba::black();
        assert_eq!((b.r, b.g, b.b, b.a), (0, 0, 0, 255));
    }

    #[test]
    fn test_font_scale_factors() {
        assert_eq!(FontScale::One.factor(), 1);
        assert_eq!(FontScale::Two.factor(), 2);
        assert_eq!(FontScale::Three.factor(), 3);
        assert_eq!(FontScale::Four.factor(), 4);
    }

    #[test]
    fn test_background_fills_buffer() {
        let w = 20u32;
        let h = 20u32;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let renderer = OverlayRenderer::new(w, h);
        let style = TextStyle {
            background: Some(Rgba::new(255, 0, 0, 255)),
            color: Rgba::transparent(),
            shadow: None,
            scale: FontScale::One,
            halign: HAlign::Left,
            valign: VAlign::Top,
            padding: 0,
        };
        let overlay = TitleOverlay::new(1, " ", 0, 0, 0, 1000).with_style(style);
        renderer.composite(&mut buf, &overlay, 500);
        // Background is red so at least some red bytes must be set.
        let has_red = buf.chunks(4).any(|p| p[0] > 0);
        assert!(has_red);
    }
}

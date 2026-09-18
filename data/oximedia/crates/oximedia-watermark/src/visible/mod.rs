//! Visible watermarking: text and logo overlay on video frames.
//!
//! This module provides:
//! - `WatermarkPosition`: position enum with coordinate computation
//! - `VisibleWatermarkConfig`: configuration for visible watermarks
//! - `TextWatermark` / `LogoWatermark`: watermark content types
//! - `VisibleWatermarker`: applies watermarks to raw RGBA frame buffers
//! - `WatermarkStrength`: computes PSNR of the watermark signal

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// WatermarkPosition
// ---------------------------------------------------------------------------

/// Position of a visible watermark on a frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WatermarkPosition {
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Centered.
    Center,
    /// Custom position as fractional coordinates (0.0–1.0).
    Custom(f32, f32),
}

impl WatermarkPosition {
    /// Compute the top-left pixel coordinates of the watermark within a frame.
    ///
    /// `frame_w/h` are the frame dimensions; `mark_w/h` are the watermark dimensions.
    /// `margin_pct` is the margin as a fraction of the frame dimension.
    #[must_use]
    pub fn to_xy(
        self,
        frame_w: u32,
        frame_h: u32,
        mark_w: u32,
        mark_h: u32,
        margin_pct: f32,
    ) -> (u32, u32) {
        let margin_x = (frame_w as f32 * margin_pct) as u32;
        let margin_y = (frame_h as f32 * margin_pct) as u32;

        match self {
            Self::TopLeft => (margin_x, margin_y),
            Self::TopRight => (
                frame_w.saturating_sub(mark_w).saturating_sub(margin_x),
                margin_y,
            ),
            Self::BottomLeft => (
                margin_x,
                frame_h.saturating_sub(mark_h).saturating_sub(margin_y),
            ),
            Self::BottomRight => (
                frame_w.saturating_sub(mark_w).saturating_sub(margin_x),
                frame_h.saturating_sub(mark_h).saturating_sub(margin_y),
            ),
            Self::Center => (
                (frame_w.saturating_sub(mark_w)) / 2,
                (frame_h.saturating_sub(mark_h)) / 2,
            ),
            Self::Custom(fx, fy) => (
                (frame_w as f32 * fx.clamp(0.0, 1.0)) as u32,
                (frame_h as f32 * fy.clamp(0.0, 1.0)) as u32,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// VisibleWatermarkConfig
// ---------------------------------------------------------------------------

/// Configuration for a visible watermark.
#[derive(Debug, Clone)]
pub struct VisibleWatermarkConfig {
    /// Position on the frame.
    pub position: WatermarkPosition,
    /// Opacity in [0.0, 1.0].
    pub opacity: f32,
    /// Scale factor for the watermark.
    pub scale: f32,
    /// Margin as a fraction of the frame dimension.
    pub margin_pct: f32,
}

impl Default for VisibleWatermarkConfig {
    fn default() -> Self {
        Self {
            position: WatermarkPosition::BottomRight,
            opacity: 0.7,
            scale: 1.0,
            margin_pct: 0.02,
        }
    }
}

// ---------------------------------------------------------------------------
// TextWatermark
// ---------------------------------------------------------------------------

/// A text-based watermark.
#[derive(Debug, Clone)]
pub struct TextWatermark {
    /// The text to render.
    pub text: String,
    /// Font size (in pixels; affects bitmap scale).
    pub font_size: u32,
    /// RGBA colour.
    pub color: [u8; 4],
}

// ---------------------------------------------------------------------------
// Simple 8×8 bitmap font for printable ASCII (0x20–0x7E)
// ---------------------------------------------------------------------------

/// 8×8 glyph bitmap for ASCII characters 0x20–0x7E.
///
/// Each glyph is 8 bytes, one byte per row (MSB = left pixel).
/// This is a minimal subset covering digits, letters and common punctuation.
static FONT_8X8: [[u8; 8]; 95] = [
    // 0x20 ' '
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x21 '!'
    [0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00],
    // 0x22 '"'
    [0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x23 '#'
    [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00],
    // 0x24 '$'
    [0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00],
    // 0x25 '%'
    [0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00],
    // 0x26 '&'
    [0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00],
    // 0x27 '\''
    [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x28 '('
    [0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00],
    // 0x29 ')'
    [0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00],
    // 0x2A '*'
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00],
    // 0x2B '+'
    [0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00],
    // 0x2C ','
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06],
    // 0x2D '-'
    [0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00],
    // 0x2E '.'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00],
    // 0x2F '/'
    [0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00],
    // 0x30 '0'
    [0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00],
    // 0x31 '1'
    [0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00],
    // 0x32 '2'
    [0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00],
    // 0x33 '3'
    [0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00],
    // 0x34 '4'
    [0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00],
    // 0x35 '5'
    [0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00],
    // 0x36 '6'
    [0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00],
    // 0x37 '7'
    [0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00],
    // 0x38 '8'
    [0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00],
    // 0x39 '9'
    [0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00],
    // 0x3A ':'
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00],
    // 0x3B ';'
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06],
    // 0x3C '<'
    [0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00],
    // 0x3D '='
    [0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00],
    // 0x3E '>'
    [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00],
    // 0x3F '?'
    [0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00],
    // 0x40 '@'
    [0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00],
    // 0x41 'A'
    [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00],
    // 0x42 'B'
    [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00],
    // 0x43 'C'
    [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00],
    // 0x44 'D'
    [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00],
    // 0x45 'E'
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00],
    // 0x46 'F'
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00],
    // 0x47 'G'
    [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00],
    // 0x48 'H'
    [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00],
    // 0x49 'I'
    [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 0x4A 'J'
    [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00],
    // 0x4B 'K'
    [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00],
    // 0x4C 'L'
    [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00],
    // 0x4D 'M'
    [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00],
    // 0x4E 'N'
    [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00],
    // 0x4F 'O'
    [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00],
    // 0x50 'P'
    [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00],
    // 0x51 'Q'
    [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00],
    // 0x52 'R'
    [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00],
    // 0x53 'S'
    [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00],
    // 0x54 'T'
    [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 0x55 'U'
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00],
    // 0x56 'V'
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],
    // 0x57 'W'
    [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00],
    // 0x58 'X'
    [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00],
    // 0x59 'Y'
    [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00],
    // 0x5A 'Z'
    [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00],
    // 0x5B '['
    [0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00],
    // 0x5C '\\'
    [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00],
    // 0x5D ']'
    [0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00],
    // 0x5E '^'
    [0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00],
    // 0x5F '_'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF],
    // 0x60 '`'
    [0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x61 'a'
    [0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00],
    // 0x62 'b'
    [0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00],
    // 0x63 'c'
    [0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00],
    // 0x64 'd'
    [0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00],
    // 0x65 'e'
    [0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00],
    // 0x66 'f'
    [0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00],
    // 0x67 'g'
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F],
    // 0x68 'h'
    [0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00],
    // 0x69 'i'
    [0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 0x6A 'j'
    [0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E],
    // 0x6B 'k'
    [0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00],
    // 0x6C 'l'
    [0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 0x6D 'm'
    [0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00],
    // 0x6E 'n'
    [0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00],
    // 0x6F 'o'
    [0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00],
    // 0x70 'p'
    [0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F],
    // 0x71 'q'
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78],
    // 0x72 'r'
    [0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00],
    // 0x73 's'
    [0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00],
    // 0x74 't'
    [0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00],
    // 0x75 'u'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00],
    // 0x76 'v'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],
    // 0x77 'w'
    [0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00],
    // 0x78 'x'
    [0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00],
    // 0x79 'y'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F],
    // 0x7A 'z'
    [0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00],
    // 0x7B '{'
    [0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00],
    // 0x7C '|'
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00],
    // 0x7D '}'
    [0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00],
    // 0x7E '~'
    [0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
];

/// Return the 8×8 glyph for an ASCII character, or the space glyph if unknown.
fn glyph_for(ch: char) -> &'static [u8; 8] {
    let code = ch as usize;
    if (0x20..=0x7E).contains(&code) {
        &FONT_8X8[code - 0x20]
    } else {
        &FONT_8X8[0] // space
    }
}

// ---------------------------------------------------------------------------
// LogoWatermark
// ---------------------------------------------------------------------------

/// An RGBA logo watermark.
#[derive(Debug, Clone)]
pub struct LogoWatermark {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA pixels (width × height × \[R,G,B,A\]).
    pub pixels: Vec<[u8; 4]>,
}

// ---------------------------------------------------------------------------
// VisibleWatermarker
// ---------------------------------------------------------------------------

/// Applies visible watermarks to raw RGBA frame buffers.
pub struct VisibleWatermarker;

impl VisibleWatermarker {
    /// Apply a text watermark to a frame.
    ///
    /// `frame` is a raw RGBA buffer of size `frame_w × frame_h × 4`.
    /// Characters are rendered at 8×8 pixels scaled by `config.scale * font_size / 8`.
    pub fn apply_text(
        frame: &mut Vec<u8>,
        frame_w: u32,
        frame_h: u32,
        mark: &TextWatermark,
        config: &VisibleWatermarkConfig,
    ) {
        let scale = ((mark.font_size as f32 / 8.0) * config.scale).max(1.0) as u32;
        let char_w = 8 * scale;
        let char_h = 8 * scale;
        let text_w = char_w * mark.text.len() as u32;
        let text_h = char_h;

        let (ox, oy) = config
            .position
            .to_xy(frame_w, frame_h, text_w, text_h, config.margin_pct);

        for (ci, ch) in mark.text.chars().enumerate() {
            let glyph = glyph_for(ch);
            let cx = ox + ci as u32 * char_w;

            for row in 0..8u32 {
                let row_bits = glyph[row as usize];
                for col in 0..8u32 {
                    let bit_set = (row_bits >> (7 - col)) & 1 == 1;
                    if !bit_set {
                        continue;
                    }
                    // Scale: render `scale × scale` block
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cx + col * scale + sx;
                            let py = oy + row * scale + sy;
                            if px < frame_w && py < frame_h {
                                let idx = ((py * frame_w + px) * 4) as usize;
                                if idx + 3 < frame.len() {
                                    alpha_blend_pixel(
                                        &mut frame[idx..idx + 4],
                                        &mark.color,
                                        config.opacity,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Apply a logo watermark to a frame via alpha blending.
    pub fn apply_logo(
        frame: &mut Vec<u8>,
        frame_w: u32,
        frame_h: u32,
        logo: &LogoWatermark,
        config: &VisibleWatermarkConfig,
    ) {
        let lw = (logo.width as f32 * config.scale) as u32;
        let lh = (logo.height as f32 * config.scale) as u32;

        let (ox, oy) = config
            .position
            .to_xy(frame_w, frame_h, lw, lh, config.margin_pct);

        for ly in 0..lh {
            for lx in 0..lw {
                // Source pixel in logo (nearest neighbour)
                let src_x = ((lx as f32 / config.scale) as u32).min(logo.width - 1);
                let src_y = ((ly as f32 / config.scale) as u32).min(logo.height - 1);
                let logo_px = &logo.pixels[(src_y * logo.width + src_x) as usize];

                let dst_x = ox + lx;
                let dst_y = oy + ly;
                if dst_x < frame_w && dst_y < frame_h {
                    let idx = ((dst_y * frame_w + dst_x) * 4) as usize;
                    if idx + 3 < frame.len() {
                        // Use logo alpha scaled by config.opacity
                        let logo_alpha = f32::from(logo_px[3]) / 255.0 * config.opacity;
                        alpha_blend_pixel(&mut frame[idx..idx + 4], logo_px, logo_alpha);
                    }
                }
            }
        }
    }
}

/// Alpha-blend `src` colour (RGBA) onto `dst` RGBA slice with the given `alpha`.
fn alpha_blend_pixel(dst: &mut [u8], src: &[u8; 4], alpha: f32) {
    let alpha = alpha.clamp(0.0, 1.0);
    let inv = 1.0 - alpha;
    for i in 0..3 {
        let blended = f32::from(src[i]) * alpha + f32::from(dst[i]) * inv;
        dst[i] = blended.clamp(0.0, 255.0) as u8;
    }
    // Alpha channel: keep destination alpha
}

// ---------------------------------------------------------------------------
// WatermarkStrength
// ---------------------------------------------------------------------------

/// Measures the perceptual strength of an embedded watermark.
pub struct WatermarkStrength;

impl WatermarkStrength {
    /// Compute the PSNR (dB) of the watermark signal.
    ///
    /// PSNR = 10 · log10(MAX² / MSE), where MAX = 255 for 8-bit data.
    /// Returns `f32::INFINITY` if the images are identical.
    #[must_use]
    pub fn compute(original: &[u8], watermarked: &[u8]) -> f32 {
        let n = original.len().min(watermarked.len());
        if n == 0 {
            return 0.0;
        }

        let mse: f64 = original[..n]
            .iter()
            .zip(watermarked[..n].iter())
            .map(|(&a, &b)| {
                let diff = f64::from(a) - f64::from(b);
                diff * diff
            })
            .sum::<f64>()
            / n as f64;

        if mse == 0.0 {
            return f32::INFINITY;
        }

        (10.0 * (255.0_f64 * 255.0 / mse).log10()) as f32
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn blank_frame(w: u32, h: u32) -> Vec<u8> {
        vec![200u8; (w * h * 4) as usize]
    }

    // --- WatermarkPosition tests ---

    #[test]
    fn test_position_top_left() {
        let (x, y) = WatermarkPosition::TopLeft.to_xy(640, 480, 100, 50, 0.02);
        assert!(x <= 20); // margin ≈ 12
        assert!(y <= 20);
    }

    #[test]
    fn test_position_bottom_right() {
        let (x, y) = WatermarkPosition::BottomRight.to_xy(640, 480, 100, 50, 0.0);
        assert_eq!(x, 540);
        assert_eq!(y, 430);
    }

    #[test]
    fn test_position_center() {
        let (x, y) = WatermarkPosition::Center.to_xy(640, 480, 100, 100, 0.0);
        assert_eq!(x, 270);
        assert_eq!(y, 190);
    }

    #[test]
    fn test_position_custom() {
        let (x, y) = WatermarkPosition::Custom(0.5, 0.5).to_xy(640, 480, 0, 0, 0.0);
        assert_eq!(x, 320);
        assert_eq!(y, 240);
    }

    #[test]
    fn test_position_top_right() {
        let (x, y) = WatermarkPosition::TopRight.to_xy(640, 480, 100, 50, 0.0);
        assert_eq!(x, 540);
        assert_eq!(y, 0);
    }

    #[test]
    fn test_position_bottom_left() {
        let (x, y) = WatermarkPosition::BottomLeft.to_xy(640, 480, 100, 50, 0.0);
        assert_eq!(x, 0);
        assert_eq!(y, 430);
    }

    // --- apply_text tests ---

    #[test]
    fn test_apply_text_modifies_frame() {
        let mut frame = blank_frame(200, 100);
        let orig = frame.clone();
        let mark = TextWatermark {
            text: "Hi".to_string(),
            font_size: 8,
            color: [255, 0, 0, 255],
        };
        let config = VisibleWatermarkConfig::default();
        VisibleWatermarker::apply_text(&mut frame, 200, 100, &mark, &config);
        assert_ne!(frame, orig);
    }

    #[test]
    fn test_apply_text_empty_string_no_change() {
        let mut frame = blank_frame(200, 100);
        let orig = frame.clone();
        let mark = TextWatermark {
            text: String::new(),
            font_size: 8,
            color: [255, 0, 0, 255],
        };
        let config = VisibleWatermarkConfig::default();
        VisibleWatermarker::apply_text(&mut frame, 200, 100, &mark, &config);
        assert_eq!(frame, orig);
    }

    #[test]
    fn test_apply_text_digits() {
        let mut frame = blank_frame(400, 100);
        let mark = TextWatermark {
            text: "0123456789".to_string(),
            font_size: 8,
            color: [0, 255, 0, 255],
        };
        let config = VisibleWatermarkConfig {
            position: WatermarkPosition::TopLeft,
            opacity: 1.0,
            scale: 1.0,
            margin_pct: 0.0,
        };
        VisibleWatermarker::apply_text(&mut frame, 400, 100, &mark, &config);
        // Just verify it doesn't panic
    }

    // --- apply_logo tests ---

    #[test]
    fn test_apply_logo_modifies_frame() {
        let mut frame = blank_frame(200, 200);
        let orig = frame.clone();
        let logo = LogoWatermark {
            width: 20,
            height: 20,
            pixels: vec![[255u8, 0, 0, 200]; 20 * 20],
        };
        let config = VisibleWatermarkConfig::default();
        VisibleWatermarker::apply_logo(&mut frame, 200, 200, &logo, &config);
        assert_ne!(frame, orig);
    }

    #[test]
    fn test_apply_logo_zero_opacity_no_change() {
        let mut frame = blank_frame(200, 200);
        let orig = frame.clone();
        let logo = LogoWatermark {
            width: 10,
            height: 10,
            pixels: vec![[255u8, 0, 0, 255]; 100],
        };
        let config = VisibleWatermarkConfig {
            opacity: 0.0,
            ..VisibleWatermarkConfig::default()
        };
        VisibleWatermarker::apply_logo(&mut frame, 200, 200, &logo, &config);
        assert_eq!(frame, orig);
    }

    // --- WatermarkStrength tests ---

    #[test]
    fn test_psnr_identical() {
        let data = vec![100u8; 1000];
        let psnr = WatermarkStrength::compute(&data, &data);
        assert!(psnr.is_infinite());
    }

    #[test]
    fn test_psnr_totally_different() {
        let a = vec![0u8; 1000];
        let b = vec![255u8; 1000];
        let psnr = WatermarkStrength::compute(&a, &b);
        assert!(psnr < 1.0); // Should be very low
    }

    #[test]
    fn test_psnr_small_diff_high_value() {
        let a = vec![128u8; 1000];
        let b: Vec<u8> = a.iter().map(|&x| x.saturating_add(1)).collect();
        let psnr = WatermarkStrength::compute(&a, &b);
        assert!(psnr > 40.0);
    }

    #[test]
    fn test_psnr_empty() {
        let psnr = WatermarkStrength::compute(&[], &[]);
        assert_eq!(psnr, 0.0);
    }

    #[test]
    fn test_alpha_blend_full_opacity() {
        let mut dst = [0u8, 0, 0, 255];
        let src = [255u8, 0, 0, 255];
        alpha_blend_pixel(&mut dst, &src, 1.0);
        assert_eq!(dst[0], 255);
        assert_eq!(dst[1], 0);
    }

    #[test]
    fn test_alpha_blend_zero_opacity() {
        let mut dst = [100u8, 100, 100, 255];
        let src = [255u8, 0, 0, 255];
        alpha_blend_pixel(&mut dst, &src, 0.0);
        assert_eq!(dst[0], 100);
        assert_eq!(dst[1], 100);
    }
}

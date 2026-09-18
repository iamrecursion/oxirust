//! Minimal embedded bitmap font for overlay text rendering.
//!
//! Provides an 8×8 pixel ASCII glyph set that requires no external font files.
//! Each glyph is stored as 8 bytes, one byte per row with bit 7 = leftmost pixel.
//!
//! # Example
//!
//! ```
//! use oximedia_graphics::bitmap_font::BitmapFont;
//!
//! let mut font = BitmapFont::basic_8x8();
//! let mut buf = vec![0u8; 320 * 32 * 4];
//! let (w, h) = font.render_text("Hello!", [255, 255, 0, 255], 2, &mut buf, 320, 4, 8);
//! assert!(w > 0 && h > 0);
//! ```

use std::collections::HashMap;

use crate::font_metrics::SubPixelMode;
use crate::glyph_cache::{GlyphCache, GlyphKey, RasterizedGlyph};

/// Font family name tag used as the `GlyphKey::font_family` for the embedded 8×8 bitmap font.
const BITMAP_FONT_FAMILY: &str = "bitmap8x8";

/// A monospace bitmap font backed by a per-character glyph bitmap.
pub struct BitmapFont {
    /// Width of a single glyph in pixels at scale 1.
    glyph_width: u32,
    /// Height of a single glyph in pixels at scale 1.
    glyph_height: u32,
    /// Map from character to packed bitmap rows.
    /// Each `Vec<u8>` has `glyph_height` bytes; bit 7 of each byte is the leftmost pixel.
    glyphs: HashMap<char, Vec<u8>>,
    /// LRU cache of scaled RGBA glyph bitmaps, keyed by `(char, scale)`.
    glyph_cache: GlyphCache,
}

impl BitmapFont {
    /// Build the built-in 8×8 ASCII font (printable ASCII 0x20–0x7E).
    ///
    /// The glyph bitmaps are embedded directly so no file I/O is required.
    #[must_use]
    pub fn basic_8x8() -> Self {
        let mut glyphs = HashMap::new();

        // Embed glyph data: 95 printable ASCII characters from space (0x20) to ~ (0x7E).
        // Each row is a u8; bit 7 = column 0 (leftmost).
        let raw: &[(char, [u8; 8])] = &GLYPHS_8X8;
        for (ch, rows) in raw {
            glyphs.insert(*ch, rows.to_vec());
        }

        Self {
            glyph_width: 8,
            glyph_height: 8,
            glyphs,
            // 512 entries covers all printable ASCII at up to ~5 scale factors.
            glyph_cache: GlyphCache::new(512),
        }
    }

    /// Width of one character cell in pixels (at scale 1).
    #[must_use]
    pub fn glyph_width(&self) -> u32 {
        self.glyph_width
    }

    /// Height of one character cell in pixels (at scale 1).
    #[must_use]
    pub fn glyph_height(&self) -> u32 {
        self.glyph_height
    }

    /// Render a string of text into an RGBA8 buffer.
    ///
    /// Scaled glyph bitmaps are cached via an internal [`GlyphCache`] so that the
    /// same `(char, scale)` pair is only rasterized once per font instance.
    ///
    /// # Parameters
    /// - `text`          – The string to render.
    /// - `color`         – RGBA foreground color `[r, g, b, a]`.
    /// - `scale`         – Integer upscale factor (1 = native 8×8, 2 = 16×16, …).
    /// - `output`        – RGBA8 target buffer (`output_width * height * 4` bytes).
    /// - `output_width`  – Pixel width of the target buffer.
    /// - `x`, `y`        – Top-left origin in the output buffer (may be negative to clip).
    ///
    /// # Returns
    /// `(rendered_width, rendered_height)` – The bounding box of the rendered text
    /// at the requested scale (independent of clipping).
    pub fn render_text(
        &mut self,
        text: &str,
        color: [u8; 4],
        scale: u32,
        output: &mut [u8],
        output_width: u32,
        x: i32,
        y: i32,
    ) -> (u32, u32) {
        let scale = scale.max(1);
        let cell_w = self.glyph_width * scale;
        let cell_h = self.glyph_height * scale;

        let output_height = if output_width == 0 {
            0
        } else {
            (output.len() as u32) / (output_width * 4)
        };

        // Rasterize and cache all unique (char, scale) pairs in this string first.
        let chars: Vec<char> = text.chars().collect();
        for &ch in &chars {
            let key = GlyphKey::new(ch, BITMAP_FONT_FAMILY, scale as u16, SubPixelMode::None);
            if self.glyph_cache.get(&key).is_none() {
                // Rasterize the glyph into an RGBA bitmap.
                let bitmap = self.rasterize_glyph(ch, color, scale);
                if let Some(rg) = RasterizedGlyph::new(cell_w, cell_h, bitmap, 0, 0, cell_w as f32)
                {
                    self.glyph_cache.insert(key, rg);
                }
            }
        }

        // Blit each character from the cache into the output buffer.
        for (char_idx, &ch) in chars.iter().enumerate() {
            let char_x = x + (char_idx as i32) * (cell_w as i32);
            let key = GlyphKey::new(ch, BITMAP_FONT_FAMILY, scale as u16, SubPixelMode::None);

            // Retrieve cached bitmap (guaranteed present after the loop above).
            let glyph_data: Vec<u8> = self
                .glyph_cache
                .get(&key)
                .map(|g| g.data.clone())
                .unwrap_or_default();

            for row in 0..cell_h {
                let py = y + row as i32;
                if py < 0 || py >= output_height as i32 {
                    continue;
                }
                for col in 0..cell_w {
                    let px = char_x + col as i32;
                    if px < 0 || px >= output_width as i32 {
                        continue;
                    }
                    let src_idx = (row * cell_w + col) as usize * 4;
                    let dst_idx = (py as u32 * output_width + px as u32) as usize * 4;
                    if src_idx + 3 >= glyph_data.len() || dst_idx + 3 >= output.len() {
                        continue;
                    }
                    // Alpha-blend cached glyph pixel over existing output pixel.
                    let fg_a = glyph_data[src_idx + 3] as f32 / 255.0;
                    if fg_a < f32::EPSILON {
                        continue;
                    }
                    let inv_a = 1.0 - fg_a;
                    output[dst_idx] =
                        (glyph_data[src_idx] as f32 * fg_a + output[dst_idx] as f32 * inv_a) as u8;
                    output[dst_idx + 1] = (glyph_data[src_idx + 1] as f32 * fg_a
                        + output[dst_idx + 1] as f32 * inv_a)
                        as u8;
                    output[dst_idx + 2] = (glyph_data[src_idx + 2] as f32 * fg_a
                        + output[dst_idx + 2] as f32 * inv_a)
                        as u8;
                    let out_a =
                        (fg_a + (output[dst_idx + 3] as f32 / 255.0) * inv_a).clamp(0.0, 1.0);
                    output[dst_idx + 3] = (out_a * 255.0) as u8;
                }
            }
        }

        let total_w = chars.len() as u32 * cell_w;
        (total_w, cell_h)
    }

    /// Rasterize a single character into a flat RGBA8 bitmap at the given scale.
    ///
    /// The resulting vec has `glyph_width * scale * glyph_height * scale * 4` bytes.
    fn rasterize_glyph(&self, ch: char, color: [u8; 4], scale: u32) -> Vec<u8> {
        let cell_w = self.glyph_width * scale;
        let cell_h = self.glyph_height * scale;
        let mut bitmap = vec![0u8; (cell_w * cell_h * 4) as usize];

        let glyph = self
            .glyphs
            .get(&ch)
            .or_else(|| self.glyphs.get(&'?'))
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        for row in 0..self.glyph_height {
            let row_bits = if (row as usize) < glyph.len() {
                glyph[row as usize]
            } else {
                0
            };
            for scaled_row in 0..scale {
                let py = row * scale + scaled_row;
                for col in 0..self.glyph_width {
                    let pixel_set = (row_bits >> (7 - col)) & 1 != 0;
                    if !pixel_set {
                        continue;
                    }
                    for scaled_col in 0..scale {
                        let px = col * scale + scaled_col;
                        let idx = (py * cell_w + px) as usize * 4;
                        if idx + 3 < bitmap.len() {
                            bitmap[idx] = color[0];
                            bitmap[idx + 1] = color[1];
                            bitmap[idx + 2] = color[2];
                            bitmap[idx + 3] = color[3];
                        }
                    }
                }
            }
        }
        bitmap
    }

    /// Return the number of cache hits (for testing / diagnostics).
    #[must_use]
    pub fn cache_hits(&self) -> u64 {
        self.glyph_cache.hits()
    }

    /// Return the number of cache misses (for testing / diagnostics).
    #[must_use]
    pub fn cache_misses(&self) -> u64 {
        self.glyph_cache.misses()
    }

    /// Measure text width and height in pixels at the given scale without rendering.
    #[must_use]
    pub fn measure_text(&self, text: &str, scale: u32) -> (u32, u32) {
        let scale = scale.max(1);
        let w = (text.chars().count() as u32) * self.glyph_width * scale;
        let h = self.glyph_height * scale;
        (w, h)
    }
}

// ---------------------------------------------------------------------------
// Embedded 8×8 glyph data
//
// Source: a classic open-source 8×8 pixel font (the IBM PC 437 / ZX Spectrum
// inspired font commonly used in retro games and terminals).  Each entry is
// a `(char, [u8; 8])` pair where each `u8` represents one pixel row with
// bit 7 = leftmost pixel.
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const GLYPHS_8X8: &[(char, [u8; 8])] = &[
    (' ', [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]),
    ('!', [0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00]),
    ('"', [0x36,0x36,0x00,0x00,0x00,0x00,0x00,0x00]),
    ('#', [0x36,0x36,0x7F,0x36,0x7F,0x36,0x36,0x00]),
    ('$', [0x0C,0x3E,0x03,0x1E,0x30,0x1F,0x0C,0x00]),
    ('%', [0x00,0x63,0x33,0x18,0x0C,0x66,0x63,0x00]),
    ('&', [0x1C,0x36,0x1C,0x6E,0x3B,0x33,0x6E,0x00]),
    ('\'', [0x06,0x06,0x03,0x00,0x00,0x00,0x00,0x00]),
    ('(', [0x18,0x0C,0x06,0x06,0x06,0x0C,0x18,0x00]),
    (')', [0x06,0x0C,0x18,0x18,0x18,0x0C,0x06,0x00]),
    ('*', [0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00]),
    ('+', [0x00,0x0C,0x0C,0x3F,0x0C,0x0C,0x00,0x00]),
    (',', [0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x06]),
    ('-', [0x00,0x00,0x00,0x3F,0x00,0x00,0x00,0x00]),
    ('.', [0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x00]),
    ('/', [0x60,0x30,0x18,0x0C,0x06,0x03,0x01,0x00]),
    ('0', [0x3E,0x63,0x73,0x7B,0x6F,0x67,0x3E,0x00]),
    ('1', [0x0C,0x0E,0x0C,0x0C,0x0C,0x0C,0x3F,0x00]),
    ('2', [0x1E,0x33,0x30,0x1C,0x06,0x33,0x3F,0x00]),
    ('3', [0x1E,0x33,0x30,0x1C,0x30,0x33,0x1E,0x00]),
    ('4', [0x38,0x3C,0x36,0x33,0x7F,0x30,0x78,0x00]),
    ('5', [0x3F,0x03,0x1F,0x30,0x30,0x33,0x1E,0x00]),
    ('6', [0x1C,0x06,0x03,0x1F,0x33,0x33,0x1E,0x00]),
    ('7', [0x3F,0x33,0x30,0x18,0x0C,0x0C,0x0C,0x00]),
    ('8', [0x1E,0x33,0x33,0x1E,0x33,0x33,0x1E,0x00]),
    ('9', [0x1E,0x33,0x33,0x3E,0x30,0x18,0x0E,0x00]),
    (':', [0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x00]),
    (';', [0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x06]),
    ('<', [0x18,0x0C,0x06,0x03,0x06,0x0C,0x18,0x00]),
    ('=', [0x00,0x00,0x3F,0x00,0x00,0x3F,0x00,0x00]),
    ('>', [0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00]),
    ('?', [0x1E,0x33,0x30,0x18,0x0C,0x00,0x0C,0x00]),
    ('@', [0x3E,0x63,0x7B,0x7B,0x7B,0x03,0x1E,0x00]),
    ('A', [0x0C,0x1E,0x33,0x33,0x3F,0x33,0x33,0x00]),
    ('B', [0x3F,0x66,0x66,0x3E,0x66,0x66,0x3F,0x00]),
    ('C', [0x3C,0x66,0x03,0x03,0x03,0x66,0x3C,0x00]),
    ('D', [0x1F,0x36,0x66,0x66,0x66,0x36,0x1F,0x00]),
    ('E', [0x7F,0x46,0x16,0x1E,0x16,0x46,0x7F,0x00]),
    ('F', [0x7F,0x46,0x16,0x1E,0x16,0x06,0x0F,0x00]),
    ('G', [0x3C,0x66,0x03,0x03,0x73,0x66,0x7C,0x00]),
    ('H', [0x33,0x33,0x33,0x3F,0x33,0x33,0x33,0x00]),
    ('I', [0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00]),
    ('J', [0x78,0x30,0x30,0x30,0x33,0x33,0x1E,0x00]),
    ('K', [0x67,0x66,0x36,0x1E,0x36,0x66,0x67,0x00]),
    ('L', [0x0F,0x06,0x06,0x06,0x46,0x66,0x7F,0x00]),
    ('M', [0x63,0x77,0x7F,0x7F,0x6B,0x63,0x63,0x00]),
    ('N', [0x63,0x67,0x6F,0x7B,0x73,0x63,0x63,0x00]),
    ('O', [0x1C,0x36,0x63,0x63,0x63,0x36,0x1C,0x00]),
    ('P', [0x3F,0x66,0x66,0x3E,0x06,0x06,0x0F,0x00]),
    ('Q', [0x1E,0x33,0x33,0x33,0x3B,0x1E,0x38,0x00]),
    ('R', [0x3F,0x66,0x66,0x3E,0x36,0x66,0x67,0x00]),
    ('S', [0x1E,0x33,0x07,0x0E,0x38,0x33,0x1E,0x00]),
    ('T', [0x3F,0x2D,0x0C,0x0C,0x0C,0x0C,0x1E,0x00]),
    ('U', [0x33,0x33,0x33,0x33,0x33,0x33,0x3F,0x00]),
    ('V', [0x33,0x33,0x33,0x33,0x33,0x1E,0x0C,0x00]),
    ('W', [0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00]),
    ('X', [0x63,0x63,0x36,0x1C,0x1C,0x36,0x63,0x00]),
    ('Y', [0x33,0x33,0x33,0x1E,0x0C,0x0C,0x1E,0x00]),
    ('Z', [0x7F,0x63,0x31,0x18,0x4C,0x66,0x7F,0x00]),
    ('[', [0x1E,0x06,0x06,0x06,0x06,0x06,0x1E,0x00]),
    ('\\', [0x03,0x06,0x0C,0x18,0x30,0x60,0x40,0x00]),
    (']', [0x1E,0x18,0x18,0x18,0x18,0x18,0x1E,0x00]),
    ('^', [0x08,0x1C,0x36,0x63,0x00,0x00,0x00,0x00]),
    ('_', [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF]),
    ('`', [0x0C,0x0C,0x18,0x00,0x00,0x00,0x00,0x00]),
    ('a', [0x00,0x00,0x1E,0x30,0x3E,0x33,0x6E,0x00]),
    ('b', [0x07,0x06,0x06,0x3E,0x66,0x66,0x3B,0x00]),
    ('c', [0x00,0x00,0x1E,0x33,0x03,0x33,0x1E,0x00]),
    ('d', [0x38,0x30,0x30,0x3e,0x33,0x33,0x6E,0x00]),
    ('e', [0x00,0x00,0x1E,0x33,0x3f,0x03,0x1E,0x00]),
    ('f', [0x1C,0x36,0x06,0x0f,0x06,0x06,0x0F,0x00]),
    ('g', [0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x1F]),
    ('h', [0x07,0x06,0x36,0x6E,0x66,0x66,0x67,0x00]),
    ('i', [0x0C,0x00,0x0E,0x0C,0x0C,0x0C,0x1E,0x00]),
    ('j', [0x30,0x00,0x30,0x30,0x30,0x33,0x33,0x1E]),
    ('k', [0x07,0x06,0x66,0x36,0x1E,0x36,0x67,0x00]),
    ('l', [0x0E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00]),
    ('m', [0x00,0x00,0x33,0x7F,0x7F,0x6B,0x63,0x00]),
    ('n', [0x00,0x00,0x1F,0x33,0x33,0x33,0x33,0x00]),
    ('o', [0x00,0x00,0x1E,0x33,0x33,0x33,0x1E,0x00]),
    ('p', [0x00,0x00,0x3B,0x66,0x66,0x3E,0x06,0x0F]),
    ('q', [0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x78]),
    ('r', [0x00,0x00,0x3B,0x6E,0x66,0x06,0x0F,0x00]),
    ('s', [0x00,0x00,0x3E,0x03,0x1E,0x30,0x1F,0x00]),
    ('t', [0x08,0x0C,0x3E,0x0C,0x0C,0x2C,0x18,0x00]),
    ('u', [0x00,0x00,0x33,0x33,0x33,0x33,0x6E,0x00]),
    ('v', [0x00,0x00,0x33,0x33,0x33,0x1E,0x0C,0x00]),
    ('w', [0x00,0x00,0x63,0x6B,0x7F,0x7F,0x36,0x00]),
    ('x', [0x00,0x00,0x63,0x36,0x1C,0x36,0x63,0x00]),
    ('y', [0x00,0x00,0x33,0x33,0x33,0x3E,0x30,0x1F]),
    ('z', [0x00,0x00,0x3F,0x19,0x0C,0x26,0x3F,0x00]),
    ('{', [0x38,0x0C,0x0C,0x07,0x0C,0x0C,0x38,0x00]),
    ('|', [0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x00]),
    ('}', [0x07,0x0C,0x0C,0x38,0x0C,0x0C,0x07,0x00]),
    ('~', [0x6E,0x3B,0x00,0x00,0x00,0x00,0x00,0x00]),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_font_has_ascii() {
        let font = BitmapFont::basic_8x8();
        assert_eq!(font.glyph_width(), 8);
        assert_eq!(font.glyph_height(), 8);
        // All printable ASCII should be present
        for ch in ' '..='~' {
            assert!(font.glyphs.contains_key(&ch), "Missing glyph for '{ch}'");
        }
    }

    #[test]
    fn test_measure_text() {
        let font = BitmapFont::basic_8x8();
        let (w, h) = font.measure_text("Hi!", 1);
        assert_eq!(w, 3 * 8);
        assert_eq!(h, 8);

        let (w2, h2) = font.measure_text("Hi!", 2);
        assert_eq!(w2, 3 * 16);
        assert_eq!(h2, 16);
    }

    #[test]
    fn test_render_text_writes_pixels() {
        let mut font = BitmapFont::basic_8x8();
        let w: u32 = 80;
        let h: u32 = 16;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let (rw, rh) = font.render_text("ABC", [255, 255, 255, 255], 1, &mut buf, w, 0, 0);
        assert_eq!(rw, 3 * 8);
        assert_eq!(rh, 8);
        // The buffer should have some non-zero pixels
        assert!(buf.iter().any(|&b| b > 0), "No pixels were written");
    }

    #[test]
    fn test_render_text_clipping() {
        // Render at negative x; should not panic and should still produce output
        let mut font = BitmapFont::basic_8x8();
        let w: u32 = 32;
        let h: u32 = 16;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        font.render_text("XY", [255, 0, 0, 255], 1, &mut buf, w, -4, 0);
        // Second character 'Y' should be partially visible
    }

    #[test]
    fn test_render_scaled() {
        let mut font = BitmapFont::basic_8x8();
        let scale = 3;
        let w: u32 = 200;
        let h: u32 = 32;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let (rw, rh) = font.render_text("OK", [0, 255, 0, 255], scale, &mut buf, w, 0, 0);
        assert_eq!(rw, 2 * 8 * scale);
        assert_eq!(rh, 8 * scale);
        assert!(buf.iter().any(|&b| b > 0));
    }

    #[test]
    fn test_render_empty_string() {
        let mut font = BitmapFont::basic_8x8();
        let mut buf = vec![0u8; 64 * 64 * 4];
        let (w, h) = font.render_text("", [255, 0, 0, 255], 1, &mut buf, 64, 0, 0);
        assert_eq!(w, 0);
        assert_eq!(h, 8);
        assert!(buf.iter().all(|&b| b == 0));
    }

    /// Render the same character twice at the same scale — the second call should
    /// be a cache hit and produce byte-identical output.
    #[test]
    fn test_glyph_cache_hit_identical() {
        let mut font = BitmapFont::basic_8x8();
        let w: u32 = 32;
        let h: u32 = 16;

        let mut buf1 = vec![0u8; (w * h * 4) as usize];
        font.render_text("A", [255, 0, 0, 255], 1, &mut buf1, w, 0, 0);

        // The cache should now have 'A' at scale 1; a second render should be a hit.
        let misses_before = font.cache_misses();
        let mut buf2 = vec![0u8; (w * h * 4) as usize];
        font.render_text("A", [255, 0, 0, 255], 1, &mut buf2, w, 0, 0);

        // No new misses — 'A' was cached on the first call.
        assert_eq!(
            font.cache_misses(),
            misses_before,
            "second render of 'A' should be a cache hit"
        );
        assert_eq!(
            buf1, buf2,
            "cached render must be byte-identical to first render"
        );
    }

    /// Render a multi-character string; all characters must be present in the
    /// cache afterwards.
    #[test]
    fn test_glyph_cache_multi_char() {
        let mut font = BitmapFont::basic_8x8();
        let text = "Hello";
        let w: u32 = 80;
        let h: u32 = 16;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        font.render_text(text, [200, 200, 200, 255], 1, &mut buf, w, 0, 0);

        // Every unique character should now be in the cache.
        for ch in text.chars().collect::<std::collections::HashSet<char>>() {
            let key = GlyphKey::new(ch, BITMAP_FONT_FAMILY, 1u16, SubPixelMode::None);
            assert!(
                font.glyph_cache.peek(&key).is_some(),
                "char '{ch}' should be in the glyph cache after render"
            );
        }
    }
}

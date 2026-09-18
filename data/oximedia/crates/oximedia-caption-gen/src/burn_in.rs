//! Burned-in subtitle rendering onto raw RGBA video frames.
//!
//! This module provides pixel-level subtitle burn-in using a built-in 8×12
//! bitmap font.  All rendering is performed in pure Rust with no external
//! dependencies.
//!
//! ## Usage
//!
//! ```rust
//! # use oximedia_caption_gen::burn_in::{BurnInConfig, SubtitleBurnIn, SubtitlePosition};
//! let config = BurnInConfig {
//!     font_size: 16,
//!     color: [255, 255, 255, 255],
//!     outline_color: [0, 0, 0, 255],
//!     outline_width: 1,
//!     position: SubtitlePosition::Bottom,
//! };
//! let mut frame = vec![0u8; 320 * 240 * 4]; // 320×240 RGBA
//! SubtitleBurnIn::render_frame(&mut frame, 320, 240, "Hello!", &config);
//! ```

// ─── Position ─────────────────────────────────────────────────────────────────

/// Where to place the subtitle text on the video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitlePosition {
    /// Lower third (default subtitle position).
    Bottom,
    /// Upper third.
    Top,
    /// Vertically centred.
    Middle,
    /// Absolute pixel position `(x, y)` from the top-left corner.
    Custom(u32, u32),
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for burned-in subtitle rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct BurnInConfig {
    /// Target font size in pixels.  The built-in glyphs are 8×12; they are
    /// scaled up by integer multiples (`font_size / 12`, clamped to ≥ 1).
    pub font_size: u32,
    /// Text colour as RGBA bytes.
    pub color: [u8; 4],
    /// Outline (shadow) colour as RGBA bytes.
    pub outline_color: [u8; 4],
    /// Outline thickness in pixels (0 = no outline).
    pub outline_width: u32,
    /// Where to position the text block.
    pub position: SubtitlePosition,
}

impl Default for BurnInConfig {
    fn default() -> Self {
        Self {
            font_size: 16,
            color: [255, 255, 255, 255],
            outline_color: [0, 0, 0, 200],
            outline_width: 1,
            position: SubtitlePosition::Bottom,
        }
    }
}

// ─── Bitmap font (8 × 12) ─────────────────────────────────────────────────────
//
// One entry per printable ASCII character (0x20 ' ' … 0x7E '~'), so 95 entries.
// Each entry is 12 bytes; byte 0 is the top row.  Bit 7 (MSB) is the leftmost
// pixel.

const FONT_GLYPH_W: u32 = 8;
const FONT_GLYPH_H: u32 = 12;
const FONT_FIRST_CHAR: u8 = 0x20; // space
const FONT_LAST_CHAR: u8 = 0x7E; // tilde

/// 8×12 bitmap font, 95 printable ASCII glyphs.
static FONT_BITMAP: [[u8; 12]; 95] = [
    // 0x20 space
    [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x21 !
    [
        0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00,
    ],
    // 0x22 "
    [
        0x28, 0x28, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x23 #
    [
        0x28, 0x28, 0x7C, 0x28, 0x28, 0x7C, 0x28, 0x28, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x24 $
    [
        0x10, 0x3C, 0x50, 0x50, 0x38, 0x14, 0x14, 0x78, 0x10, 0x00, 0x00, 0x00,
    ],
    // 0x25 %
    [
        0x62, 0x64, 0x08, 0x08, 0x10, 0x10, 0x26, 0x46, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x26 &
    [
        0x30, 0x48, 0x48, 0x30, 0x50, 0x88, 0x88, 0x70, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x27 '
    [
        0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x28 (
    [
        0x08, 0x10, 0x20, 0x20, 0x20, 0x20, 0x20, 0x10, 0x08, 0x00, 0x00, 0x00,
    ],
    // 0x29 )
    [
        0x20, 0x10, 0x08, 0x08, 0x08, 0x08, 0x08, 0x10, 0x20, 0x00, 0x00, 0x00,
    ],
    // 0x2A *
    [
        0x00, 0x00, 0x10, 0x54, 0x38, 0x54, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x2B +
    [
        0x00, 0x00, 0x10, 0x10, 0x7C, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x2C ,
    [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x10, 0x20, 0x00, 0x00,
    ],
    // 0x2D -
    [
        0x00, 0x00, 0x00, 0x00, 0x7C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x2E .
    [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x2F /
    [
        0x02, 0x04, 0x04, 0x08, 0x10, 0x20, 0x20, 0x40, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x30 0
    [
        0x38, 0x44, 0x44, 0x54, 0x64, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x31 1
    [
        0x10, 0x30, 0x10, 0x10, 0x10, 0x10, 0x10, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x32 2
    [
        0x38, 0x44, 0x04, 0x08, 0x10, 0x20, 0x40, 0x7C, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x33 3
    [
        0x38, 0x44, 0x04, 0x18, 0x04, 0x04, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x34 4
    [
        0x08, 0x18, 0x28, 0x48, 0x7C, 0x08, 0x08, 0x08, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x35 5
    [
        0x7C, 0x40, 0x40, 0x78, 0x04, 0x04, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x36 6
    [
        0x1C, 0x20, 0x40, 0x78, 0x44, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x37 7
    [
        0x7C, 0x04, 0x08, 0x08, 0x10, 0x10, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x38 8
    [
        0x38, 0x44, 0x44, 0x38, 0x44, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x39 9
    [
        0x38, 0x44, 0x44, 0x44, 0x3C, 0x04, 0x08, 0x70, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x3A :
    [
        0x00, 0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x3B ;
    [
        0x00, 0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x10, 0x20, 0x00, 0x00, 0x00,
    ],
    // 0x3C <
    [
        0x04, 0x08, 0x10, 0x20, 0x40, 0x20, 0x10, 0x08, 0x04, 0x00, 0x00, 0x00,
    ],
    // 0x3D =
    [
        0x00, 0x00, 0x7C, 0x00, 0x00, 0x7C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x3E >
    [
        0x40, 0x20, 0x10, 0x08, 0x04, 0x08, 0x10, 0x20, 0x40, 0x00, 0x00, 0x00,
    ],
    // 0x3F ?
    [
        0x38, 0x44, 0x04, 0x08, 0x10, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x40 @
    [
        0x38, 0x44, 0x5C, 0x54, 0x5C, 0x40, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x41 A
    [
        0x10, 0x28, 0x44, 0x44, 0x7C, 0x44, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x42 B
    [
        0x78, 0x44, 0x44, 0x78, 0x44, 0x44, 0x44, 0x78, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x43 C
    [
        0x38, 0x44, 0x40, 0x40, 0x40, 0x40, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x44 D
    [
        0x70, 0x48, 0x44, 0x44, 0x44, 0x44, 0x48, 0x70, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x45 E
    [
        0x7C, 0x40, 0x40, 0x78, 0x40, 0x40, 0x40, 0x7C, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x46 F
    [
        0x7C, 0x40, 0x40, 0x78, 0x40, 0x40, 0x40, 0x40, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x47 G
    [
        0x38, 0x44, 0x40, 0x40, 0x5C, 0x44, 0x44, 0x3C, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x48 H
    [
        0x44, 0x44, 0x44, 0x7C, 0x44, 0x44, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x49 I
    [
        0x38, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x4A J
    [
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x4B K
    [
        0x44, 0x48, 0x50, 0x60, 0x50, 0x48, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x4C L
    [
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x7C, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x4D M
    [
        0x44, 0x6C, 0x54, 0x54, 0x44, 0x44, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x4E N
    [
        0x44, 0x64, 0x54, 0x54, 0x4C, 0x44, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x4F O
    [
        0x38, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x50 P
    [
        0x78, 0x44, 0x44, 0x78, 0x40, 0x40, 0x40, 0x40, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x51 Q
    [
        0x38, 0x44, 0x44, 0x44, 0x44, 0x54, 0x48, 0x34, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x52 R
    [
        0x78, 0x44, 0x44, 0x78, 0x50, 0x48, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x53 S
    [
        0x38, 0x44, 0x40, 0x38, 0x04, 0x04, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x54 T
    [
        0x7C, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x55 U
    [
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x56 V
    [
        0x44, 0x44, 0x44, 0x44, 0x28, 0x28, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x57 W
    [
        0x44, 0x44, 0x44, 0x54, 0x54, 0x54, 0x6C, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x58 X
    [
        0x44, 0x44, 0x28, 0x10, 0x10, 0x28, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x59 Y
    [
        0x44, 0x44, 0x44, 0x28, 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x5A Z
    [
        0x7C, 0x04, 0x08, 0x10, 0x20, 0x40, 0x40, 0x7C, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x5B [
    [
        0x38, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x5C backslash
    [
        0x40, 0x20, 0x20, 0x10, 0x08, 0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x5D ]
    [
        0x1C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1C, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x5E ^
    [
        0x10, 0x28, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x5F _
    [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7C, 0x00, 0x00, 0x00,
    ],
    // 0x60 `
    [
        0x20, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x61 a
    [
        0x00, 0x00, 0x38, 0x04, 0x3C, 0x44, 0x44, 0x3C, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x62 b
    [
        0x40, 0x40, 0x78, 0x44, 0x44, 0x44, 0x44, 0x78, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x63 c
    [
        0x00, 0x00, 0x38, 0x44, 0x40, 0x40, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x64 d
    [
        0x04, 0x04, 0x3C, 0x44, 0x44, 0x44, 0x44, 0x3C, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x65 e
    [
        0x00, 0x00, 0x38, 0x44, 0x7C, 0x40, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x66 f
    [
        0x18, 0x24, 0x20, 0x78, 0x20, 0x20, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x67 g
    [
        0x00, 0x00, 0x3C, 0x44, 0x44, 0x3C, 0x04, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x68 h
    [
        0x40, 0x40, 0x78, 0x44, 0x44, 0x44, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x69 i
    [
        0x10, 0x00, 0x30, 0x10, 0x10, 0x10, 0x10, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x6A j
    [
        0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x6B k
    [
        0x40, 0x40, 0x44, 0x48, 0x70, 0x48, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x6C l
    [
        0x30, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x6D m
    [
        0x00, 0x00, 0x68, 0x54, 0x54, 0x54, 0x54, 0x54, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x6E n
    [
        0x00, 0x00, 0x78, 0x44, 0x44, 0x44, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x6F o
    [
        0x00, 0x00, 0x38, 0x44, 0x44, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x70 p
    [
        0x00, 0x00, 0x78, 0x44, 0x44, 0x78, 0x40, 0x40, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x71 q
    [
        0x00, 0x00, 0x3C, 0x44, 0x44, 0x3C, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x72 r
    [
        0x00, 0x00, 0x58, 0x24, 0x20, 0x20, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x73 s
    [
        0x00, 0x00, 0x38, 0x44, 0x30, 0x0C, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x74 t
    [
        0x20, 0x20, 0x7C, 0x20, 0x20, 0x20, 0x24, 0x18, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x75 u
    [
        0x00, 0x00, 0x44, 0x44, 0x44, 0x44, 0x44, 0x3C, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x76 v
    [
        0x00, 0x00, 0x44, 0x44, 0x44, 0x28, 0x28, 0x10, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x77 w
    [
        0x00, 0x00, 0x44, 0x44, 0x54, 0x54, 0x6C, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x78 x
    [
        0x00, 0x00, 0x44, 0x28, 0x10, 0x28, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x79 y
    [
        0x00, 0x00, 0x44, 0x44, 0x3C, 0x04, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x7A z
    [
        0x00, 0x00, 0x7C, 0x08, 0x10, 0x20, 0x40, 0x7C, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x7B {
    [
        0x08, 0x10, 0x10, 0x20, 0x10, 0x10, 0x10, 0x08, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x7C |
    [
        0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    ],
    // 0x7D }
    [
        0x20, 0x10, 0x10, 0x08, 0x10, 0x10, 0x10, 0x20, 0x00, 0x00, 0x00, 0x00,
    ],
    // 0x7E ~
    [
        0x30, 0x49, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
];

/// Returns the glyph bitmap for a character, or the '?' glyph for unmapped chars.
fn glyph_for(ch: char) -> &'static [u8; 12] {
    let b = ch as u32;
    if b >= FONT_FIRST_CHAR as u32 && b <= FONT_LAST_CHAR as u32 {
        let idx = (b - FONT_FIRST_CHAR as u32) as usize;
        &FONT_BITMAP[idx]
    } else {
        // '?' fallback
        &FONT_BITMAP[(b'?' - FONT_FIRST_CHAR) as usize]
    }
}

// ─── SubtitleBurnIn ───────────────────────────────────────────────────────────

/// Performs bitmap burn-in of subtitle text onto raw RGBA video frames.
pub struct SubtitleBurnIn;

impl SubtitleBurnIn {
    /// Render `text` onto an RGBA frame buffer in-place.
    ///
    /// `frame` must have length `width * height * 4`.  If the frame is too
    /// small for the given dimensions the function returns immediately without
    /// modifying anything.
    ///
    /// Non-ASCII / unmapped characters are replaced by `?`.
    pub fn render_frame(
        frame: &mut [u8],
        width: u32,
        height: u32,
        text: &str,
        config: &BurnInConfig,
    ) {
        if width == 0 || height == 0 || text.is_empty() {
            return;
        }
        let expected = (width as usize) * (height as usize) * 4;
        if frame.len() < expected {
            return;
        }

        // Compute scale factor (integer multiple of the base 8×12 glyph).
        let scale = (config.font_size / FONT_GLYPH_H).max(1);
        let glyph_w = FONT_GLYPH_W * scale;
        let glyph_h = FONT_GLYPH_H * scale;

        // Split text into lines on '\n'.
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return;
        }

        // Width of the widest line in pixels.
        let max_line_chars = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
        let block_w = (max_line_chars as u32) * glyph_w;
        let block_h = (lines.len() as u32) * glyph_h;

        // Compute top-left origin of the text block.
        let (origin_x, origin_y) = compute_origin(
            width,
            height,
            block_w,
            block_h,
            config.outline_width,
            &config.position,
        );

        // Render outline first (if requested), then fill.
        if config.outline_width > 0 {
            for (line_idx, line) in lines.iter().enumerate() {
                let line_y = origin_y + (line_idx as u32) * glyph_h;
                render_line_glyphs(
                    frame,
                    width,
                    height,
                    line,
                    origin_x,
                    line_y,
                    scale,
                    config.outline_color,
                    Some(config.outline_width),
                );
            }
        }

        for (line_idx, line) in lines.iter().enumerate() {
            let line_y = origin_y + (line_idx as u32) * glyph_h;
            render_line_glyphs(
                frame,
                width,
                height,
                line,
                origin_x,
                line_y,
                scale,
                config.color,
                None,
            );
        }
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute the (x, y) top-left origin for the text block.
fn compute_origin(
    frame_w: u32,
    frame_h: u32,
    block_w: u32,
    block_h: u32,
    outline_w: u32,
    position: &SubtitlePosition,
) -> (u32, u32) {
    let centre_x = if frame_w > block_w {
        (frame_w - block_w) / 2
    } else {
        0
    };
    let margin = outline_w + 4; // small visual margin

    match position {
        SubtitlePosition::Bottom => {
            let y = if frame_h > block_h + margin {
                frame_h - block_h - margin
            } else {
                0
            };
            (centre_x, y)
        }
        SubtitlePosition::Top => (centre_x, margin),
        SubtitlePosition::Middle => {
            let y = if frame_h > block_h {
                (frame_h - block_h) / 2
            } else {
                0
            };
            (centre_x, y)
        }
        SubtitlePosition::Custom(x, y) => (*x, *y),
    }
}

/// Render a single line of text glyphs.
///
/// If `outline_radius` is `Some(r)`, each lit pixel is expanded by a square
/// of radius `r` in the given colour (outline pass).  If `None`, pixels are
/// rendered at exact glyph positions (fill pass).
#[allow(clippy::too_many_arguments)]
fn render_line_glyphs(
    frame: &mut [u8],
    width: u32,
    height: u32,
    line: &str,
    origin_x: u32,
    origin_y: u32,
    scale: u32,
    color: [u8; 4],
    outline_radius: Option<u32>,
) {
    for (char_idx, ch) in line.chars().enumerate() {
        let char_x = origin_x + (char_idx as u32) * FONT_GLYPH_W * scale;
        let glyph = glyph_for(ch);

        for row in 0..FONT_GLYPH_H {
            let bitmap_byte = glyph[row as usize];
            for col in 0..FONT_GLYPH_W {
                let bit = (bitmap_byte >> (7 - col)) & 1;
                if bit == 0 {
                    continue;
                }
                // This glyph pixel maps to a (scale × scale) block.
                let base_px = char_x + col * scale;
                let base_py = origin_y + row * scale;

                match outline_radius {
                    Some(r) => {
                        // Expand each lit pixel outward by r in all directions.
                        let sx = base_px.saturating_sub(r);
                        let ex = (base_px + scale - 1 + r).min(width - 1);
                        let sy = base_py.saturating_sub(r);
                        let ey = (base_py + scale - 1 + r).min(height - 1);
                        for py in sy..=ey {
                            for px in sx..=ex {
                                put_pixel(frame, width, height, px, py, color);
                            }
                        }
                    }
                    None => {
                        // Render the (scale × scale) block of this pixel.
                        for sy in 0..scale {
                            for sx in 0..scale {
                                put_pixel(frame, width, height, base_px + sx, base_py + sy, color);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Write an RGBA pixel at `(px, py)`, bounds-checked.
#[inline]
fn put_pixel(frame: &mut [u8], width: u32, height: u32, px: u32, py: u32, color: [u8; 4]) {
    if px >= width || py >= height {
        return;
    }
    let offset = ((py as usize) * (width as usize) + (px as usize)) * 4;
    if offset + 3 >= frame.len() {
        return;
    }
    frame[offset] = color[0];
    frame[offset + 1] = color[1];
    frame[offset + 2] = color[2];
    frame[offset + 3] = color[3];
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn blank_frame(w: u32, h: u32) -> Vec<u8> {
        vec![0u8; (w as usize) * (h as usize) * 4]
    }

    fn any_pixel_set(frame: &[u8], color: [u8; 4]) -> bool {
        frame.chunks_exact(4).any(|p| p == color)
    }

    // ── BurnInConfig defaults ─────────────────────────────────────────────────

    #[test]
    fn default_config_has_bottom_position() {
        let cfg = BurnInConfig::default();
        assert_eq!(cfg.position, SubtitlePosition::Bottom);
    }

    #[test]
    fn default_config_white_text() {
        let cfg = BurnInConfig::default();
        assert_eq!(cfg.color[0], 255);
        assert_eq!(cfg.color[3], 255);
    }

    // ── render_frame ──────────────────────────────────────────────────────────

    #[test]
    fn render_frame_empty_text_no_change() {
        let mut frame = blank_frame(320, 240);
        let cfg = BurnInConfig::default();
        SubtitleBurnIn::render_frame(&mut frame, 320, 240, "", &cfg);
        assert!(frame.iter().all(|&b| b == 0));
    }

    #[test]
    fn render_frame_writes_pixels() {
        let mut frame = blank_frame(320, 240);
        let cfg = BurnInConfig {
            font_size: 12,
            color: [255, 0, 0, 255],
            outline_color: [0, 0, 0, 255],
            outline_width: 0,
            position: SubtitlePosition::Bottom,
        };
        SubtitleBurnIn::render_frame(&mut frame, 320, 240, "Hello", &cfg);
        // At least some red pixels must exist.
        assert!(any_pixel_set(&frame, [255, 0, 0, 255]));
    }

    #[test]
    fn render_frame_bottom_position() {
        let w = 320u32;
        let h = 240u32;
        let mut frame = blank_frame(w, h);
        let cfg = BurnInConfig {
            font_size: 12,
            color: [255, 255, 255, 255],
            outline_color: [0, 0, 0, 255],
            outline_width: 0,
            position: SubtitlePosition::Bottom,
        };
        SubtitleBurnIn::render_frame(&mut frame, w, h, "Hi", &cfg);
        // Check that the bottom quarter has some lit pixels.
        let bottom_start = (h * 3 / 4) as usize * w as usize * 4;
        let bottom_pixels = &frame[bottom_start..];
        assert!(bottom_pixels.chunks_exact(4).any(|p| p[3] > 0));
    }

    #[test]
    fn render_frame_top_position() {
        let w = 320u32;
        let h = 240u32;
        let mut frame = blank_frame(w, h);
        let cfg = BurnInConfig {
            font_size: 12,
            color: [255, 255, 255, 255],
            outline_color: [0, 0, 0, 255],
            outline_width: 0,
            position: SubtitlePosition::Top,
        };
        SubtitleBurnIn::render_frame(&mut frame, w, h, "Hi", &cfg);
        // Top quarter should have lit pixels.
        let top_end = (h / 4) as usize * w as usize * 4;
        let top_pixels = &frame[..top_end];
        assert!(top_pixels.chunks_exact(4).any(|p| p[3] > 0));
    }

    #[test]
    fn render_frame_middle_position() {
        let w = 320u32;
        let h = 240u32;
        let mut frame = blank_frame(w, h);
        let cfg = BurnInConfig {
            font_size: 12,
            color: [255, 255, 255, 255],
            outline_color: [0, 0, 0, 255],
            outline_width: 0,
            position: SubtitlePosition::Middle,
        };
        SubtitleBurnIn::render_frame(&mut frame, w, h, "M", &cfg);
        // Middle region should have lit pixels.
        let mid_start = (h / 4) as usize * w as usize * 4;
        let mid_end = (h * 3 / 4) as usize * w as usize * 4;
        let mid_pixels = &frame[mid_start..mid_end];
        assert!(mid_pixels.chunks_exact(4).any(|p| p[3] > 0));
    }

    #[test]
    fn render_frame_custom_position() {
        let w = 640u32;
        let h = 480u32;
        let mut frame = blank_frame(w, h);
        let px = 100u32;
        let py = 200u32;
        let cfg = BurnInConfig {
            font_size: 12,
            color: [0, 255, 0, 255],
            outline_color: [0, 0, 0, 255],
            outline_width: 0,
            position: SubtitlePosition::Custom(px, py),
        };
        SubtitleBurnIn::render_frame(&mut frame, w, h, "A", &cfg);
        // Pixels in the region around (px, py) should be set.
        let region_start = (py as usize * w as usize + px as usize) * 4;
        let region_end = (region_start + FONT_GLYPH_H as usize * w as usize * 4).min(frame.len());
        let region = &frame[region_start..region_end];
        assert!(region.chunks_exact(4).any(|p| p[3] > 0));
    }

    #[test]
    fn render_frame_with_outline() {
        let mut frame = blank_frame(320, 240);
        let cfg = BurnInConfig {
            font_size: 12,
            color: [255, 255, 255, 255],
            outline_color: [0, 0, 0, 200],
            outline_width: 2,
            position: SubtitlePosition::Bottom,
        };
        SubtitleBurnIn::render_frame(&mut frame, 320, 240, "Outline", &cfg);
        // Both white (fill) and dark (outline) pixels must be present.
        assert!(any_pixel_set(&frame, [255, 255, 255, 255]));
        assert!(any_pixel_set(&frame, [0, 0, 0, 200]));
    }

    #[test]
    fn render_frame_zero_width_no_panic() {
        let mut frame = blank_frame(1, 1);
        let cfg = BurnInConfig::default();
        // Should not panic.
        SubtitleBurnIn::render_frame(&mut frame, 0, 240, "Hi", &cfg);
    }

    #[test]
    fn render_frame_zero_height_no_panic() {
        let mut frame = blank_frame(1, 1);
        let cfg = BurnInConfig::default();
        SubtitleBurnIn::render_frame(&mut frame, 320, 0, "Hi", &cfg);
    }

    #[test]
    fn render_frame_undersized_buffer_no_panic() {
        // Buffer too small for the stated dimensions — must not panic.
        let mut frame = vec![0u8; 8];
        let cfg = BurnInConfig::default();
        SubtitleBurnIn::render_frame(&mut frame, 320, 240, "Hi", &cfg);
    }

    #[test]
    fn render_frame_multiline_text() {
        let mut frame = blank_frame(320, 240);
        let cfg = BurnInConfig {
            font_size: 12,
            color: [128, 128, 128, 255],
            outline_color: [0, 0, 0, 255],
            outline_width: 0,
            position: SubtitlePosition::Bottom,
        };
        SubtitleBurnIn::render_frame(&mut frame, 320, 240, "Line 1\nLine 2", &cfg);
        assert!(any_pixel_set(&frame, [128, 128, 128, 255]));
    }

    #[test]
    fn render_frame_large_font_scale() {
        let mut frame = blank_frame(800, 600);
        let cfg = BurnInConfig {
            font_size: 36,
            color: [200, 100, 50, 255],
            outline_color: [0, 0, 0, 255],
            outline_width: 1,
            position: SubtitlePosition::Bottom,
        };
        SubtitleBurnIn::render_frame(&mut frame, 800, 600, "Big", &cfg);
        assert!(any_pixel_set(&frame, [200, 100, 50, 255]));
    }

    #[test]
    fn glyph_for_space_is_blank() {
        let g = glyph_for(' ');
        assert!(g.iter().all(|&b| b == 0));
    }

    #[test]
    fn glyph_for_non_ascii_returns_question_mark() {
        let g = glyph_for('Ω');
        let q = glyph_for('?');
        assert_eq!(g, q);
    }
}

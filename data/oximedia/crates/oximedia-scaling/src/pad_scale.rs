//! Padding-aware scaling for video frames.
//!
//! Scales video while adding letterbox/pillarbox padding to preserve the
//! original aspect ratio. Supports configurable padding colour, alignment,
//! and safe-area margins.

#![allow(dead_code)]

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// An RGBA colour used for padding regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadColor {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel.
    pub a: u8,
}

impl PadColor {
    /// Create a new colour.
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Opaque black.
    pub const fn black() -> Self {
        Self::new(0, 0, 0, 255)
    }

    /// Opaque white.
    pub const fn white() -> Self {
        Self::new(255, 255, 255, 255)
    }

    /// Fully transparent.
    pub const fn transparent() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Convert to a 32-bit packed RGBA value.
    pub const fn to_u32(self) -> u32 {
        (self.r as u32) << 24 | (self.g as u32) << 16 | (self.b as u32) << 8 | self.a as u32
    }

    /// Create from a packed RGBA u32.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn from_u32(v: u32) -> Self {
        Self {
            r: (v >> 24) as u8,
            g: ((v >> 16) & 0xFF) as u8,
            b: ((v >> 8) & 0xFF) as u8,
            a: (v & 0xFF) as u8,
        }
    }
}

impl Default for PadColor {
    fn default() -> Self {
        Self::black()
    }
}

/// Vertical alignment of the active picture within the padded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VAlign {
    /// Align to top.
    Top,
    /// Center vertically.
    Center,
    /// Align to bottom.
    Bottom,
}

/// Horizontal alignment of the active picture within the padded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HAlign {
    /// Align to left.
    Left,
    /// Center horizontally.
    Center,
    /// Align to right.
    Right,
}

/// Computed padding amounts (in pixels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadAmounts {
    /// Pixels of padding on the top.
    pub top: u32,
    /// Pixels of padding on the bottom.
    pub bottom: u32,
    /// Pixels of padding on the left.
    pub left: u32,
    /// Pixels of padding on the right.
    pub right: u32,
}

impl PadAmounts {
    /// Total horizontal padding.
    pub fn horizontal(&self) -> u32 {
        self.left + self.right
    }

    /// Total vertical padding.
    pub fn vertical(&self) -> u32 {
        self.top + self.bottom
    }

    /// True if no padding is needed.
    pub fn is_zero(&self) -> bool {
        self.top == 0 && self.bottom == 0 && self.left == 0 && self.right == 0
    }
}

impl fmt::Display for PadAmounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "top={} bottom={} left={} right={}",
            self.top, self.bottom, self.left, self.right
        )
    }
}

/// Configuration for a pad-and-scale operation.
#[derive(Debug, Clone)]
pub struct PadScaleConfig {
    /// Source width.
    pub src_width: u32,
    /// Source height.
    pub src_height: u32,
    /// Target output width (including padding).
    pub dst_width: u32,
    /// Target output height (including padding).
    pub dst_height: u32,
    /// Padding colour.
    pub pad_color: PadColor,
    /// Horizontal alignment.
    pub h_align: HAlign,
    /// Vertical alignment.
    pub v_align: VAlign,
    /// Extra safe-area margin in pixels (applied on all four sides).
    pub safe_margin: u32,
}

impl PadScaleConfig {
    /// Create a new config.
    pub fn new(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Self {
        Self {
            src_width: src_w,
            src_height: src_h,
            dst_width: dst_w,
            dst_height: dst_h,
            pad_color: PadColor::black(),
            h_align: HAlign::Center,
            v_align: VAlign::Center,
            safe_margin: 0,
        }
    }

    /// Set the pad colour.
    pub fn with_color(mut self, c: PadColor) -> Self {
        self.pad_color = c;
        self
    }

    /// Set horizontal alignment.
    pub fn with_h_align(mut self, a: HAlign) -> Self {
        self.h_align = a;
        self
    }

    /// Set vertical alignment.
    pub fn with_v_align(mut self, a: VAlign) -> Self {
        self.v_align = a;
        self
    }

    /// Set safe-area margin.
    pub fn with_safe_margin(mut self, px: u32) -> Self {
        self.safe_margin = px;
        self
    }
}

/// Compute the scaled dimensions of the active picture (excluding padding).
#[allow(clippy::cast_precision_loss)]
pub fn compute_active_size(cfg: &PadScaleConfig) -> (u32, u32) {
    let avail_w = cfg.dst_width.saturating_sub(cfg.safe_margin * 2);
    let avail_h = cfg.dst_height.saturating_sub(cfg.safe_margin * 2);

    if cfg.src_width == 0 || cfg.src_height == 0 || avail_w == 0 || avail_h == 0 {
        return (0, 0);
    }

    let src_ar = cfg.src_width as f64 / cfg.src_height as f64;
    let dst_ar = avail_w as f64 / avail_h as f64;

    if src_ar > dst_ar {
        // Fit to width
        let w = avail_w;
        let h = (w as f64 / src_ar).round() as u32;
        (w, h.min(avail_h))
    } else {
        // Fit to height
        let h = avail_h;
        let w = (h as f64 * src_ar).round() as u32;
        (w.min(avail_w), h)
    }
}

/// Compute the padding amounts for the given config.
pub fn compute_padding(cfg: &PadScaleConfig) -> PadAmounts {
    let (act_w, act_h) = compute_active_size(cfg);
    let total_h_pad = cfg.dst_width.saturating_sub(act_w);
    let total_v_pad = cfg.dst_height.saturating_sub(act_h);

    let (left, right) = match cfg.h_align {
        HAlign::Left => (cfg.safe_margin, total_h_pad.saturating_sub(cfg.safe_margin)),
        HAlign::Right => (total_h_pad.saturating_sub(cfg.safe_margin), cfg.safe_margin),
        HAlign::Center => (total_h_pad / 2, total_h_pad - total_h_pad / 2),
    };

    let (top, bottom) = match cfg.v_align {
        VAlign::Top => (cfg.safe_margin, total_v_pad.saturating_sub(cfg.safe_margin)),
        VAlign::Bottom => (total_v_pad.saturating_sub(cfg.safe_margin), cfg.safe_margin),
        VAlign::Center => (total_v_pad / 2, total_v_pad - total_v_pad / 2),
    };

    PadAmounts {
        top,
        bottom,
        left,
        right,
    }
}

/// Fill a u8 buffer of size `w*h` with a single greyscale value (for testing).
pub fn fill_pad_frame(w: u32, h: u32, value: u8) -> Vec<u8> {
    vec![value; (w * h) as usize]
}

/// Composite the active picture into a padded output buffer.
///
/// Both `active` and the returned buffer are single-channel, row-major.
pub fn compose_padded(
    active: &[u8],
    act_w: u32,
    act_h: u32,
    pad: &PadAmounts,
    bg_value: u8,
) -> Vec<u8> {
    let out_w = pad.left + act_w + pad.right;
    let out_h = pad.top + act_h + pad.bottom;
    let mut buf = vec![bg_value; (out_w * out_h) as usize];

    for row in 0..act_h {
        let src_offset = (row * act_w) as usize;
        let dst_offset = ((pad.top + row) * out_w + pad.left) as usize;
        let len = act_w as usize;
        if src_offset + len <= active.len() && dst_offset + len <= buf.len() {
            buf[dst_offset..dst_offset + len]
                .copy_from_slice(&active[src_offset..src_offset + len]);
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_color_black() {
        let c = PadColor::black();
        assert_eq!(c.r, 0);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn test_pad_color_roundtrip_u32() {
        let c = PadColor::new(128, 64, 32, 255);
        let packed = c.to_u32();
        let c2 = PadColor::from_u32(packed);
        assert_eq!(c, c2);
    }

    #[test]
    fn test_pad_color_transparent() {
        let c = PadColor::transparent();
        assert_eq!(c.a, 0);
    }

    #[test]
    fn test_pad_amounts_zero() {
        let p = PadAmounts {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
        };
        assert!(p.is_zero());
        assert_eq!(p.horizontal(), 0);
        assert_eq!(p.vertical(), 0);
    }

    #[test]
    fn test_pad_amounts_display() {
        let p = PadAmounts {
            top: 10,
            bottom: 20,
            left: 5,
            right: 15,
        };
        let s = p.to_string();
        assert!(s.contains("top=10"));
        assert!(s.contains("right=15"));
    }

    #[test]
    fn test_active_size_same_aspect() {
        let cfg = PadScaleConfig::new(1920, 1080, 1920, 1080);
        let (w, h) = compute_active_size(&cfg);
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }

    #[test]
    fn test_active_size_pillarbox() {
        // 4:3 into 16:9 -> pillarbox (narrower active picture)
        let cfg = PadScaleConfig::new(1440, 1080, 1920, 1080);
        let (w, h) = compute_active_size(&cfg);
        assert!(w < 1920);
        assert_eq!(h, 1080);
    }

    #[test]
    fn test_active_size_letterbox() {
        // 2.39:1 into 16:9 -> letterbox (shorter active picture)
        let cfg = PadScaleConfig::new(2390, 1000, 1920, 1080);
        let (w, h) = compute_active_size(&cfg);
        assert_eq!(w, 1920);
        assert!(h < 1080);
    }

    #[test]
    fn test_active_size_with_margin() {
        let cfg = PadScaleConfig::new(1920, 1080, 1920, 1080).with_safe_margin(10);
        let (w, h) = compute_active_size(&cfg);
        assert!(w <= 1900);
        assert!(h <= 1060);
    }

    #[test]
    fn test_padding_center_aligned() {
        let cfg = PadScaleConfig::new(1440, 1080, 1920, 1080);
        let pad = compute_padding(&cfg);
        // Top and bottom should be zero (letterboxing not needed)
        assert_eq!(pad.top, 0);
        assert_eq!(pad.bottom, 0);
        // Left + right should equal 1920 - active_width
        let (act_w, _) = compute_active_size(&cfg);
        assert_eq!(pad.left + pad.right, 1920 - act_w);
    }

    #[test]
    fn test_padding_left_aligned() {
        let cfg = PadScaleConfig::new(1440, 1080, 1920, 1080).with_h_align(HAlign::Left);
        let pad = compute_padding(&cfg);
        assert_eq!(pad.left, 0);
        assert!(pad.right > 0);
    }

    #[test]
    fn test_compose_padded_output_size() {
        let active = vec![128u8; 4 * 3]; // 4x3
        let pad = PadAmounts {
            top: 2,
            bottom: 2,
            left: 3,
            right: 3,
        };
        let out = compose_padded(&active, 4, 3, &pad, 0);
        assert_eq!(out.len(), (4 + 6) * (3 + 4)); // 10 * 7 = 70
    }

    #[test]
    fn test_compose_padded_center_pixel() {
        let active = vec![200u8; 2 * 2];
        let pad = PadAmounts {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
        };
        let out = compose_padded(&active, 2, 2, &pad, 0);
        // Output is 4x4, active at (1,1) to (2,2)
        assert_eq!(out[0], 0); // top-left pad
        assert_eq!(out[5], 200); // row 1, col 1 => index 1*4+1 = 5
    }

    #[test]
    fn test_fill_pad_frame() {
        let buf = fill_pad_frame(10, 5, 42);
        assert_eq!(buf.len(), 50);
        assert!(buf.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_config_builder() {
        let cfg = PadScaleConfig::new(1920, 1080, 3840, 2160)
            .with_color(PadColor::white())
            .with_v_align(VAlign::Bottom)
            .with_safe_margin(20);
        assert_eq!(cfg.pad_color, PadColor::white());
        assert_eq!(cfg.v_align, VAlign::Bottom);
        assert_eq!(cfg.safe_margin, 20);
    }

    #[test]
    fn test_zero_source_size() {
        let cfg = PadScaleConfig::new(0, 0, 1920, 1080);
        let (w, h) = compute_active_size(&cfg);
        assert_eq!(w, 0);
        assert_eq!(h, 0);
    }
}

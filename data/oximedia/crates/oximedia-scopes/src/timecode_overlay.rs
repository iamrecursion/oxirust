//! Timecode and frame-number overlay for scope displays.
//!
//! This module provides pixel-level text rendering of SMPTE timecode
//! (HH:MM:SS:FF) and absolute frame numbers onto any RGBA scope image
//! using a built-in 5×7 bitmap font.  The overlay is designed to be
//! composited on top of waveform, vectorscope, histogram, or any other
//! scope output.
//!
//! # Features
//!
//! - SMPTE 12M timecode display (non-drop-frame and drop-frame aware)
//! - Absolute frame counter overlay
//! - Configurable text colour, background fill, and position
//! - `ScopeTimecode` struct with `from_frame_number` conversion
//! - Strict input validation (returns errors for out-of-range values)

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Timecode type
// ─────────────────────────────────────────────────────────────────────────────

/// A SMPTE-style timecode value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeTimecode {
    /// Hours (0–23).
    pub hours: u8,
    /// Minutes (0–59).
    pub minutes: u8,
    /// Seconds (0–59).
    pub seconds: u8,
    /// Frames (0 .. `fps - 1`).
    pub frames: u8,
    /// Frame rate (e.g. 24, 25, 30, 50, 60).
    pub fps: u8,
    /// Whether this is a drop-frame timecode.
    pub drop_frame: bool,
}

impl ScopeTimecode {
    /// Creates a new timecode, validating all fields.
    ///
    /// # Errors
    ///
    /// Returns an error if any field is out of range.
    pub fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        fps: u8,
        drop_frame: bool,
    ) -> OxiResult<Self> {
        if fps == 0 {
            return Err(OxiError::InvalidData("fps must be non-zero".into()));
        }
        if hours > 23 {
            return Err(OxiError::InvalidData(format!("hours {hours} > 23")));
        }
        if minutes > 59 {
            return Err(OxiError::InvalidData(format!("minutes {minutes} > 59")));
        }
        if seconds > 59 {
            return Err(OxiError::InvalidData(format!("seconds {seconds} > 59")));
        }
        if frames >= fps {
            return Err(OxiError::InvalidData(format!(
                "frames {frames} >= fps {fps}"
            )));
        }
        Ok(Self {
            hours,
            minutes,
            seconds,
            frames,
            fps,
            drop_frame,
        })
    }

    /// Converts an absolute frame number to a `ScopeTimecode` (non-drop-frame).
    ///
    /// # Errors
    ///
    /// Returns an error if `fps` is zero.
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_frame_number(frame: u64, fps: u8) -> OxiResult<Self> {
        if fps == 0 {
            return Err(OxiError::InvalidData("fps must be non-zero".into()));
        }
        let fps_u64 = u64::from(fps);
        let total_seconds = frame / fps_u64;
        let frames = (frame % fps_u64) as u8;
        let seconds = (total_seconds % 60) as u8;
        let total_minutes = total_seconds / 60;
        let minutes = (total_minutes % 60) as u8;
        let hours = ((total_minutes / 60) % 24) as u8;
        Ok(Self {
            hours,
            minutes,
            seconds,
            frames,
            fps,
            drop_frame: false,
        })
    }

    /// Formats the timecode as `HH:MM:SS:FF` (non-drop) or `HH:MM:SS;FF` (drop).
    #[must_use]
    pub fn to_string_smpte(&self) -> String {
        let sep = if self.drop_frame { ';' } else { ':' };
        format!(
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, sep, self.frames
        )
    }

    /// Returns the total number of frames represented by this timecode.
    #[must_use]
    pub fn to_frame_number(&self) -> u64 {
        let fps = u64::from(self.fps);
        let h = u64::from(self.hours);
        let m = u64::from(self.minutes);
        let s = u64::from(self.seconds);
        let f = u64::from(self.frames);
        ((h * 3600 + m * 60 + s) * fps) + f
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Overlay configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Horizontal anchor for text placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAnchorH {
    /// Align to the left edge (plus `offset_x` padding).
    Left,
    /// Centre horizontally.
    Centre,
    /// Align to the right edge (minus `offset_x` padding).
    Right,
}

/// Vertical anchor for text placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAnchorV {
    /// Align to the top edge (plus `offset_y` padding).
    Top,
    /// Centre vertically.
    Centre,
    /// Align to the bottom edge (minus `offset_y` padding).
    Bottom,
}

/// What information to burn into the scope image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimecodeContent {
    /// Only the SMPTE timecode string (HH:MM:SS:FF).
    TimecodeOnly,
    /// Only the absolute frame number.
    FrameNumberOnly,
    /// Both timecode and frame number on separate lines.
    Both,
}

/// Configuration for the timecode overlay renderer.
#[derive(Debug, Clone)]
pub struct TimecodeOverlayConfig {
    /// What text to render.
    pub content: TimecodeContent,
    /// Foreground RGBA colour.
    pub fg_color: [u8; 4],
    /// Background RGBA colour (use alpha < 255 for semi-transparent).
    pub bg_color: [u8; 4],
    /// Pixel scale factor for the glyph (1 = 5×7, 2 = 10×14, …).
    pub scale: u32,
    /// Horizontal anchor.
    pub anchor_h: TextAnchorH,
    /// Vertical anchor.
    pub anchor_v: TextAnchorV,
    /// Pixel padding from the anchor edge.
    pub padding: u32,
}

impl Default for TimecodeOverlayConfig {
    fn default() -> Self {
        Self {
            content: TimecodeContent::Both,
            fg_color: [255, 255, 255, 255],
            bg_color: [0, 0, 0, 180],
            scale: 2,
            anchor_h: TextAnchorH::Left,
            anchor_v: TextAnchorV::Bottom,
            padding: 8,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bitmap font (5×7 glyphs for '0'-'9', ':', ';', 'H', 'M', 'S', 'F', ' ')
// ─────────────────────────────────────────────────────────────────────────────

// Each entry: [col0, col1, col2, col3, col4], MSB = top row.
// Only the characters we actually need are stored.
const GLYPH_WIDTH: u32 = 5;
const GLYPH_HEIGHT: u32 = 7;

fn glyph_for_char(c: char) -> [u8; 5] {
    match c {
        '0' => [0x3E, 0x51, 0x49, 0x45, 0x3E],
        '1' => [0x00, 0x42, 0x7F, 0x40, 0x00],
        '2' => [0x42, 0x61, 0x51, 0x49, 0x46],
        '3' => [0x21, 0x41, 0x45, 0x4B, 0x31],
        '4' => [0x18, 0x14, 0x12, 0x7F, 0x10],
        '5' => [0x27, 0x45, 0x45, 0x45, 0x39],
        '6' => [0x3C, 0x4A, 0x49, 0x49, 0x30],
        '7' => [0x01, 0x71, 0x09, 0x05, 0x03],
        '8' => [0x36, 0x49, 0x49, 0x49, 0x36],
        '9' => [0x06, 0x49, 0x49, 0x29, 0x1E],
        ':' => [0x00, 0x36, 0x36, 0x00, 0x00],
        ';' => [0x00, 0x56, 0x36, 0x00, 0x00],
        'H' => [0x7F, 0x08, 0x08, 0x08, 0x7F],
        'M' => [0x7F, 0x02, 0x04, 0x02, 0x7F],
        'S' => [0x26, 0x49, 0x49, 0x49, 0x32],
        'F' => [0x7F, 0x09, 0x09, 0x09, 0x01],
        _ => [0x00; 5], // space / unknown
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer
// ─────────────────────────────────────────────────────────────────────────────

/// Burn a timecode and/or frame number overlay onto an RGBA scope image.
///
/// # Arguments
///
/// * `rgba` — mutable slice of RGBA pixels, length must be `width * height * 4`.
/// * `width` / `height` — scope image dimensions.
/// * `timecode` — optional SMPTE timecode to render.
/// * `frame_number` — optional absolute frame counter.
/// * `config` — overlay style configuration.
///
/// # Errors
///
/// Returns an error if the pixel buffer length does not match the declared
/// dimensions, or if both `timecode` and `frame_number` are `None` when
/// `content` is not `Both`.
pub fn burn_timecode_overlay(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    timecode: Option<&ScopeTimecode>,
    frame_number: Option<u64>,
    config: &TimecodeOverlayConfig,
) -> OxiResult<()> {
    let expected = (width * height * 4) as usize;
    if rgba.len() != expected {
        return Err(OxiError::InvalidData(format!(
            "RGBA buffer length {}, expected {expected}",
            rgba.len()
        )));
    }

    // Build lines to render
    let mut lines: Vec<String> = Vec::new();
    match config.content {
        TimecodeContent::TimecodeOnly => {
            if let Some(tc) = timecode {
                lines.push(tc.to_string_smpte());
            }
        }
        TimecodeContent::FrameNumberOnly => {
            if let Some(f) = frame_number {
                lines.push(format!("F:{f}"));
            }
        }
        TimecodeContent::Both => {
            if let Some(tc) = timecode {
                lines.push(tc.to_string_smpte());
            }
            if let Some(f) = frame_number {
                lines.push(format!("F:{f}"));
            }
        }
    }

    if lines.is_empty() {
        return Ok(());
    }

    let scale = config.scale.max(1);
    let glyph_w = GLYPH_WIDTH * scale;
    let glyph_h = GLYPH_HEIGHT * scale;
    let line_spacing = glyph_h + scale; // 1px gap between lines

    // Compute total text block dimensions
    let max_chars = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let block_w = max_chars as u32 * (glyph_w + scale);
    let block_h = lines.len() as u32 * line_spacing;
    let bg_pad = scale;
    let bg_w = block_w + bg_pad * 2;
    let bg_h = block_h + bg_pad * 2;

    // Determine top-left corner of the background box
    let bx = match config.anchor_h {
        TextAnchorH::Left => config.padding,
        TextAnchorH::Centre => width.saturating_sub(bg_w) / 2,
        TextAnchorH::Right => width.saturating_sub(bg_w + config.padding),
    };
    let by = match config.anchor_v {
        TextAnchorV::Top => config.padding,
        TextAnchorV::Centre => height.saturating_sub(bg_h) / 2,
        TextAnchorV::Bottom => height.saturating_sub(bg_h + config.padding),
    };

    // Draw background
    fill_rect_rgba(rgba, width, height, bx, by, bg_w, bg_h, config.bg_color);

    // Draw each line of text
    for (line_idx, line) in lines.iter().enumerate() {
        let ly = by + bg_pad + line_idx as u32 * line_spacing;
        let lx = bx + bg_pad;
        let mut cx = lx;
        for ch in line.chars() {
            let glyph = glyph_for_char(ch);
            render_glyph(rgba, width, height, cx, ly, &glyph, scale, config.fg_color);
            cx += glyph_w + scale;
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

fn fill_rect_rgba(
    rgba: &mut [u8],
    img_w: u32,
    img_h: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    color: [u8; 4],
) {
    let alpha = color[3] as f32 / 255.0;
    let ia = 1.0 - alpha;
    for dy in 0..h {
        let py = y + dy;
        if py >= img_h {
            break;
        }
        for dx in 0..w {
            let px = x + dx;
            if px >= img_w {
                break;
            }
            let idx = ((py * img_w + px) * 4) as usize;
            rgba[idx] = (color[0] as f32 * alpha + rgba[idx] as f32 * ia) as u8;
            rgba[idx + 1] = (color[1] as f32 * alpha + rgba[idx + 1] as f32 * ia) as u8;
            rgba[idx + 2] = (color[2] as f32 * alpha + rgba[idx + 2] as f32 * ia) as u8;
            rgba[idx + 3] = 255;
        }
    }
}

/// Renders a single 5×7 glyph at pixel position (x, y) with the given scale.
/// The glyph columns are stored as bitmasks; bit 6 (0x40) = top row.
fn render_glyph(
    rgba: &mut [u8],
    img_w: u32,
    img_h: u32,
    x: u32,
    y: u32,
    glyph: &[u8; 5],
    scale: u32,
    color: [u8; 4],
) {
    for col in 0..GLYPH_WIDTH {
        let bits = glyph[col as usize];
        for row in 0..GLYPH_HEIGHT {
            // Bit 6 is the top row; row 0 maps to bit 6.
            let bit = (bits >> (GLYPH_HEIGHT - 1 - row)) & 1;
            if bit == 0 {
                continue;
            }
            let px0 = x + col * scale;
            let py0 = y + row * scale;
            for dy in 0..scale {
                let py = py0 + dy;
                if py >= img_h {
                    continue;
                }
                for dx in 0..scale {
                    let px = px0 + dx;
                    if px >= img_w {
                        continue;
                    }
                    let idx = ((py * img_w + px) * 4) as usize;
                    rgba[idx] = color[0];
                    rgba[idx + 1] = color[1];
                    rgba[idx + 2] = color[2];
                    rgba[idx + 3] = color[3];
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ScopeTimecode tests ───────────────────────────────────────────────

    #[test]
    fn test_timecode_new_valid() {
        let tc = ScopeTimecode::new(1, 30, 45, 12, 24, false);
        assert!(tc.is_ok());
        let tc = tc.expect("should succeed");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_timecode_new_invalid_fps_zero() {
        let tc = ScopeTimecode::new(0, 0, 0, 0, 0, false);
        assert!(tc.is_err());
    }

    #[test]
    fn test_timecode_new_invalid_hours() {
        let tc = ScopeTimecode::new(24, 0, 0, 0, 25, false);
        assert!(tc.is_err());
    }

    #[test]
    fn test_timecode_new_invalid_minutes() {
        let tc = ScopeTimecode::new(0, 60, 0, 0, 25, false);
        assert!(tc.is_err());
    }

    #[test]
    fn test_timecode_new_invalid_seconds() {
        let tc = ScopeTimecode::new(0, 0, 60, 0, 25, false);
        assert!(tc.is_err());
    }

    #[test]
    fn test_timecode_new_invalid_frames() {
        // frames must be < fps
        let tc = ScopeTimecode::new(0, 0, 0, 25, 25, false);
        assert!(tc.is_err());
    }

    #[test]
    fn test_timecode_from_frame_number() {
        // 24 fps: frame 0 → 00:00:00:00
        let tc = ScopeTimecode::from_frame_number(0, 24);
        assert!(tc.is_ok());
        let tc = tc.expect("should succeed");
        assert_eq!(tc.hours, 0);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_timecode_from_frame_number_1_second() {
        // 25 fps: 25 frames → 00:00:01:00
        let tc = ScopeTimecode::from_frame_number(25, 25).expect("should succeed");
        assert_eq!(tc.seconds, 1);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_timecode_from_frame_number_1_minute() {
        // 30 fps: 30*60=1800 frames → 00:01:00:00
        let tc = ScopeTimecode::from_frame_number(1800, 30).expect("should succeed");
        assert_eq!(tc.minutes, 1);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_timecode_from_frame_number_zero_fps_error() {
        assert!(ScopeTimecode::from_frame_number(0, 0).is_err());
    }

    #[test]
    fn test_timecode_to_string_smpte_non_drop() {
        let tc = ScopeTimecode::new(1, 2, 3, 4, 25, false).expect("should succeed");
        assert_eq!(tc.to_string_smpte(), "01:02:03:04");
    }

    #[test]
    fn test_timecode_to_string_smpte_drop_frame() {
        let tc = ScopeTimecode::new(0, 1, 0, 2, 30, true).expect("should succeed");
        assert_eq!(tc.to_string_smpte(), "00:01:00;02");
    }

    #[test]
    fn test_timecode_round_trip() {
        let frame = 86_399u64 * 24 + 23;
        let tc = ScopeTimecode::from_frame_number(frame, 24).expect("should succeed");
        assert_eq!(tc.to_frame_number(), frame);
    }

    // ── burn_timecode_overlay tests ───────────────────────────────────────

    fn blank_rgba(w: u32, h: u32) -> Vec<u8> {
        vec![0u8; (w * h * 4) as usize]
    }

    #[test]
    fn test_burn_overlay_wrong_buffer_size() {
        let mut buf = vec![0u8; 100];
        let tc = ScopeTimecode::new(0, 0, 0, 0, 25, false).expect("valid tc");
        let result = burn_timecode_overlay(
            &mut buf,
            100,
            100,
            Some(&tc),
            None,
            &TimecodeOverlayConfig::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_burn_overlay_timecode_only() {
        let mut buf = blank_rgba(256, 64);
        let tc = ScopeTimecode::new(1, 2, 3, 4, 25, false).expect("valid tc");
        let cfg = TimecodeOverlayConfig {
            content: TimecodeContent::TimecodeOnly,
            ..Default::default()
        };
        let result = burn_timecode_overlay(&mut buf, 256, 64, Some(&tc), None, &cfg);
        assert!(result.is_ok());
        // Verify that at least some pixels are non-zero (text was drawn)
        assert!(buf.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_burn_overlay_frame_number_only() {
        let mut buf = blank_rgba(256, 64);
        let cfg = TimecodeOverlayConfig {
            content: TimecodeContent::FrameNumberOnly,
            ..Default::default()
        };
        let result = burn_timecode_overlay(&mut buf, 256, 64, None, Some(42), &cfg);
        assert!(result.is_ok());
        assert!(buf.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_burn_overlay_both() {
        let mut buf = blank_rgba(512, 128);
        let tc = ScopeTimecode::new(0, 0, 1, 0, 30, false).expect("valid tc");
        let cfg = TimecodeOverlayConfig::default();
        let result = burn_timecode_overlay(&mut buf, 512, 128, Some(&tc), Some(30), &cfg);
        assert!(result.is_ok());
        assert!(buf.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_burn_overlay_no_content_is_noop() {
        let mut buf = blank_rgba(64, 64);
        let cfg = TimecodeOverlayConfig {
            content: TimecodeContent::TimecodeOnly,
            ..Default::default()
        };
        // Neither timecode nor frame number → noop
        let result = burn_timecode_overlay(&mut buf, 64, 64, None, None, &cfg);
        assert!(result.is_ok());
        assert!(buf.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_burn_overlay_top_right_anchor() {
        let mut buf = blank_rgba(256, 64);
        let tc = ScopeTimecode::new(0, 0, 0, 0, 25, false).expect("valid tc");
        let cfg = TimecodeOverlayConfig {
            content: TimecodeContent::TimecodeOnly,
            anchor_h: TextAnchorH::Right,
            anchor_v: TextAnchorV::Top,
            ..Default::default()
        };
        let result = burn_timecode_overlay(&mut buf, 256, 64, Some(&tc), None, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_burn_overlay_centre_anchor() {
        let mut buf = blank_rgba(256, 64);
        let tc = ScopeTimecode::new(12, 34, 56, 7, 25, false).expect("valid tc");
        let cfg = TimecodeOverlayConfig {
            content: TimecodeContent::TimecodeOnly,
            anchor_h: TextAnchorH::Centre,
            anchor_v: TextAnchorV::Centre,
            ..Default::default()
        };
        let result = burn_timecode_overlay(&mut buf, 256, 64, Some(&tc), None, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_timecode_overlay_config_default() {
        let cfg = TimecodeOverlayConfig::default();
        assert_eq!(cfg.content, TimecodeContent::Both);
        assert_eq!(cfg.scale, 2);
        assert_eq!(cfg.anchor_v, TextAnchorV::Bottom);
    }
}

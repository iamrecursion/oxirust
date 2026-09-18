//! Safe-area overlay for broadcast and cinema standards.
//!
//! Safe areas define the portion of the frame that will be visible on all
//! consumer display devices and in all delivery contexts.  Historically:
//!
//! - **Action safe**: 90% of the picture area — important graphical elements
//!   should not be placed outside this boundary.
//! - **Title safe**: 80% of the picture area — text, titles, and credits must
//!   be placed within this zone.
//!
//! Modern standards (SMPTE RP 2046-1, EBU R 95) use smaller margins:
//!
//! - **EBU/SMPTE action safe**: 93.75% (3.125% from each edge)
//! - **EBU/SMPTE title safe**: 90% (5% from each edge)
//! - **Center cut protection**: 4:3 safe area centred in 16:9 (or vice-versa)
//!
//! This module renders safe-area overlays onto any RGBA scope or video image.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Broadcast standard
// ─────────────────────────────────────────────────────────────────────────────

/// Broadcast standard governing the safe-area margins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastStandard {
    /// Legacy 90%/80% margins (traditional NTSC/PAL).
    Legacy,
    /// SMPTE RP 2046-1 / EBU R 95 (93.75%/90%).
    SmptEbu,
    /// BBC / OfCom guidance (88% title safe).
    Bbc,
    /// Custom margins (set via `SafeAreaConfig::custom_*`).
    Custom,
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the safe-area overlay renderer.
#[derive(Debug, Clone)]
pub struct SafeAreaConfig {
    /// Broadcast standard for margin calculation.
    pub standard: BroadcastStandard,
    /// Whether to draw the action-safe boundary.
    pub show_action_safe: bool,
    /// Whether to draw the title-safe boundary.
    pub show_title_safe: bool,
    /// Whether to draw the 4:3 centre-cut protection rectangle.
    pub show_centre_cut: bool,
    /// Custom action-safe fraction (0.0–1.0) used when `standard == Custom`.
    pub custom_action_safe: f32,
    /// Custom title-safe fraction (0.0–1.0) used when `standard == Custom`.
    pub custom_title_safe: f32,
    /// RGBA colour for the action-safe boundary.
    pub action_safe_color: [u8; 4],
    /// RGBA colour for the title-safe boundary.
    pub title_safe_color: [u8; 4],
    /// RGBA colour for the centre-cut boundary.
    pub centre_cut_color: [u8; 4],
    /// Line thickness in pixels.
    pub line_thickness: u32,
    /// Whether to dim the area outside the action-safe region.
    pub dim_outside: bool,
    /// Dimming opacity for the area outside action safe (0 = transparent, 255 = opaque black).
    pub dim_alpha: u8,
}

impl Default for SafeAreaConfig {
    fn default() -> Self {
        Self {
            standard: BroadcastStandard::SmptEbu,
            show_action_safe: true,
            show_title_safe: true,
            show_centre_cut: false,
            custom_action_safe: 0.9375,
            custom_title_safe: 0.90,
            action_safe_color: [0, 220, 255, 200], // cyan
            title_safe_color: [255, 200, 0, 200],  // yellow
            centre_cut_color: [200, 80, 255, 200], // purple
            line_thickness: 2,
            dim_outside: false,
            dim_alpha: 80,
        }
    }
}

impl SafeAreaConfig {
    /// Returns `(action_safe_fraction, title_safe_fraction)` for the active standard.
    #[must_use]
    pub fn margins(&self) -> (f32, f32) {
        match self.standard {
            BroadcastStandard::Legacy => (0.90, 0.80),
            BroadcastStandard::SmptEbu => (0.9375, 0.90),
            BroadcastStandard::Bbc => (0.90, 0.88),
            BroadcastStandard::Custom => (
                self.custom_action_safe.clamp(0.5, 1.0),
                self.custom_title_safe.clamp(0.5, 1.0),
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe-area zone descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// A computed safe-area zone in pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct SafeAreaZone {
    /// Name of this zone.
    pub name: &'static str,
    /// Left edge in pixels (inclusive).
    pub left: u32,
    /// Top edge in pixels (inclusive).
    pub top: u32,
    /// Right edge in pixels (inclusive).
    pub right: u32,
    /// Bottom edge in pixels (inclusive).
    pub bottom: u32,
    /// Fraction of the image this zone covers.
    pub fraction: f32,
}

impl SafeAreaZone {
    /// Compute the action and title safe zones for the given image dimensions.
    ///
    /// Returns `[action_safe, title_safe]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(width: u32, height: u32, config: &SafeAreaConfig) -> [Self; 2] {
        let (action_frac, title_frac) = config.margins();

        let action_zone = Self::zone_for_fraction(width, height, action_frac, "Action Safe");
        let title_zone = Self::zone_for_fraction(width, height, title_frac, "Title Safe");

        [action_zone, title_zone]
    }

    /// Compute the 4:3 centre-cut rectangle for a 16:9 frame.
    #[must_use]
    pub fn centre_cut_4x3(width: u32, height: u32) -> Self {
        // 4:3 in a 16:9 frame: centre_w = height * 4/3
        let centre_w = (height * 4 / 3).min(width);
        let left = (width - centre_w) / 2;
        let right = left + centre_w - 1;
        Self {
            name: "4:3 Centre Cut",
            left,
            top: 0,
            right,
            bottom: height.saturating_sub(1),
            fraction: centre_w as f32 / width as f32,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn zone_for_fraction(width: u32, height: u32, fraction: f32, name: &'static str) -> Self {
        let margin_x = ((width as f32 * (1.0 - fraction)) / 2.0).round() as u32;
        let margin_y = ((height as f32 * (1.0 - fraction)) / 2.0).round() as u32;
        Self {
            name,
            left: margin_x,
            top: margin_y,
            right: width.saturating_sub(margin_x + 1),
            bottom: height.saturating_sub(margin_y + 1),
            fraction,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer
// ─────────────────────────────────────────────────────────────────────────────

/// Burn safe-area overlay boxes onto an RGBA image.
///
/// # Arguments
///
/// * `rgba` — mutable RGBA pixel buffer (`width × height × 4` bytes).
/// * `width` / `height` — image dimensions.
/// * `config` — safe-area configuration.
///
/// # Errors
///
/// Returns an error if the buffer size does not match the declared dimensions.
pub fn render_safe_area_overlay(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    config: &SafeAreaConfig,
) -> OxiResult<()> {
    let expected = (width * height * 4) as usize;
    if rgba.len() != expected {
        return Err(OxiError::InvalidData(format!(
            "Buffer length {} != expected {expected}",
            rgba.len()
        )));
    }
    if width == 0 || height == 0 {
        return Ok(());
    }

    let [action_zone, title_zone] = SafeAreaZone::compute(width, height, config);

    // Optionally dim the area outside action-safe
    if config.dim_outside {
        dim_outside_region(rgba, width, height, &action_zone, config.dim_alpha);
    }

    let thickness = config.line_thickness.max(1);

    if config.show_action_safe {
        draw_rect_outline(
            rgba,
            width,
            height,
            &action_zone,
            config.action_safe_color,
            thickness,
        );
    }

    if config.show_title_safe {
        draw_rect_outline(
            rgba,
            width,
            height,
            &title_zone,
            config.title_safe_color,
            thickness,
        );
    }

    if config.show_centre_cut {
        let cc = SafeAreaZone::centre_cut_4x3(width, height);
        draw_rect_outline(rgba, width, height, &cc, config.centre_cut_color, thickness);
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

fn draw_rect_outline(
    rgba: &mut [u8],
    img_w: u32,
    img_h: u32,
    zone: &SafeAreaZone,
    color: [u8; 4],
    thickness: u32,
) {
    for t in 0..thickness {
        let l = zone.left + t;
        let r = zone.right.saturating_sub(t);
        let top = zone.top + t;
        let bot = zone.bottom.saturating_sub(t);

        // Top / bottom horizontal lines
        for x in l..=r {
            blend_pixel(rgba, img_w, img_h, x, top, color);
            blend_pixel(rgba, img_w, img_h, x, bot, color);
        }
        // Left / right vertical lines
        for y in top..=bot {
            blend_pixel(rgba, img_w, img_h, l, y, color);
            blend_pixel(rgba, img_w, img_h, r, y, color);
        }
    }
}

fn dim_outside_region(rgba: &mut [u8], img_w: u32, img_h: u32, zone: &SafeAreaZone, alpha: u8) {
    let dim_color = [0u8, 0, 0, alpha];
    for y in 0..img_h {
        for x in 0..img_w {
            let inside = x >= zone.left && x <= zone.right && y >= zone.top && y <= zone.bottom;
            if !inside {
                blend_pixel(rgba, img_w, img_h, x, y, dim_color);
            }
        }
    }
}

fn blend_pixel(rgba: &mut [u8], img_w: u32, img_h: u32, x: u32, y: u32, color: [u8; 4]) {
    if x >= img_w || y >= img_h {
        return;
    }
    let idx = ((y * img_w + x) * 4) as usize;
    let a = color[3] as f32 / 255.0;
    let ia = 1.0 - a;
    rgba[idx] = (color[0] as f32 * a + rgba[idx] as f32 * ia) as u8;
    rgba[idx + 1] = (color[1] as f32 * a + rgba[idx + 1] as f32 * ia) as u8;
    rgba[idx + 2] = (color[2] as f32 * a + rgba[idx + 2] as f32 * ia) as u8;
    rgba[idx + 3] = 255;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn blank(w: u32, h: u32) -> Vec<u8> {
        vec![0u8; (w * h * 4) as usize]
    }

    #[test]
    fn test_safe_area_config_default() {
        let cfg = SafeAreaConfig::default();
        assert_eq!(cfg.standard, BroadcastStandard::SmptEbu);
        assert!(cfg.show_action_safe);
        assert!(cfg.show_title_safe);
        assert!(!cfg.show_centre_cut);
    }

    #[test]
    fn test_margins_legacy() {
        let mut cfg = SafeAreaConfig::default();
        cfg.standard = BroadcastStandard::Legacy;
        let (action, title) = cfg.margins();
        assert!((action - 0.90).abs() < 1e-4);
        assert!((title - 0.80).abs() < 1e-4);
    }

    #[test]
    fn test_margins_smpte_ebu() {
        let cfg = SafeAreaConfig::default();
        let (action, title) = cfg.margins();
        assert!((action - 0.9375).abs() < 1e-4);
        assert!((title - 0.90).abs() < 1e-4);
    }

    #[test]
    fn test_margins_bbc() {
        let mut cfg = SafeAreaConfig::default();
        cfg.standard = BroadcastStandard::Bbc;
        let (action, title) = cfg.margins();
        assert!((action - 0.90).abs() < 1e-4);
        assert!((title - 0.88).abs() < 1e-4);
    }

    #[test]
    fn test_margins_custom() {
        let mut cfg = SafeAreaConfig::default();
        cfg.standard = BroadcastStandard::Custom;
        cfg.custom_action_safe = 0.85;
        cfg.custom_title_safe = 0.75;
        let (a, t) = cfg.margins();
        assert!((a - 0.85).abs() < 1e-4);
        assert!((t - 0.75).abs() < 1e-4);
    }

    #[test]
    fn test_safe_area_zone_compute_symmetrical() {
        let cfg = SafeAreaConfig::default();
        let [action, title] = SafeAreaZone::compute(1920, 1080, &cfg);
        // Action safe should be inside image
        assert!(action.left < action.right);
        assert!(action.top < action.bottom);
        // Title safe should be inside action safe
        assert!(title.left >= action.left);
        assert!(title.top >= action.top);
    }

    #[test]
    fn test_centre_cut_4x3() {
        let cc = SafeAreaZone::centre_cut_4x3(1920, 1080);
        // For 1080p: 4:3 width = 1080 * 4/3 = 1440
        assert_eq!(cc.right - cc.left + 1, 1440);
        assert_eq!(cc.top, 0);
        assert_eq!(cc.bottom, 1079);
    }

    #[test]
    fn test_render_safe_area_wrong_buffer() {
        let mut buf = vec![0u8; 100];
        let cfg = SafeAreaConfig::default();
        let result = render_safe_area_overlay(&mut buf, 100, 100, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_render_safe_area_marks_pixels() {
        let mut buf = blank(200, 200);
        let cfg = SafeAreaConfig {
            show_action_safe: true,
            show_title_safe: true,
            dim_outside: false,
            ..Default::default()
        };
        render_safe_area_overlay(&mut buf, 200, 200, &cfg).expect("should succeed");
        // Some pixels should be non-zero (the box outline)
        assert!(buf.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_render_safe_area_zero_dimensions_ok() {
        let mut buf = vec![];
        let cfg = SafeAreaConfig::default();
        let result = render_safe_area_overlay(&mut buf, 0, 0, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_render_safe_area_with_dim_outside() {
        let mut buf = blank(64, 64);
        let cfg = SafeAreaConfig {
            dim_outside: true,
            dim_alpha: 128,
            ..Default::default()
        };
        render_safe_area_overlay(&mut buf, 64, 64, &cfg).expect("should succeed");
        // The corners (outside action-safe) should have alpha=255 but be dimmed
        assert!(buf[3] == 255); // alpha channel of pixel 0
    }

    #[test]
    fn test_render_centre_cut() {
        let mut buf = blank(320, 180);
        let cfg = SafeAreaConfig {
            show_action_safe: false,
            show_title_safe: false,
            show_centre_cut: true,
            ..Default::default()
        };
        render_safe_area_overlay(&mut buf, 320, 180, &cfg).expect("should succeed");
        assert!(buf.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_line_thickness_2() {
        let mut buf = blank(100, 100);
        let cfg = SafeAreaConfig {
            line_thickness: 2,
            show_title_safe: false,
            ..Default::default()
        };
        render_safe_area_overlay(&mut buf, 100, 100, &cfg).expect("should succeed");
        assert!(buf.iter().any(|&v| v > 0));
    }
}

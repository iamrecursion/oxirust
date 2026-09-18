//! Side-by-side scope comparison for before/after visual analysis.
//!
//! `ScopeComparison` composites two pre-rendered scope images (RGBA) into a
//! single output image with configurable split modes:
//!
//! - **SideBySide**: left half = before, right half = after
//! - **TopBottom**: top half = before, bottom half = after
//! - **Split**: arbitrary percentage split with optional divider line
//! - **Wipe**: diagonal or vertical wipe (position-based reveal)
//!
//! All rendering is done in pure Rust with no external image dependencies.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// How the before and after scopes are combined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComparisonMode {
    /// Left = before, right = after (equal halves).
    SideBySide,
    /// Top = before, bottom = after (equal halves).
    TopBottom,
    /// Vertical split at a configurable percentage (0.0–1.0).
    VerticalSplit {
        /// Position of the split, 0.0 = far left, 1.0 = far right.
        split: f32,
    },
    /// Horizontal split at a configurable percentage (0.0–1.0).
    HorizontalSplit {
        /// Position of the split, 0.0 = top, 1.0 = bottom.
        split: f32,
    },
    /// Diagonal wipe from top-left to bottom-right.
    DiagonalWipe {
        /// Progress of the wipe, 0.0 = all before, 1.0 = all after.
        progress: f32,
    },
}

/// Style of the divider line between the two scopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DividerStyle {
    /// No divider drawn.
    None,
    /// A 1-pixel white line.
    ThinWhite,
    /// A 2-pixel black line.
    ThinBlack,
    /// A 3-pixel yellow line.
    ThickYellow,
}

/// Configuration for `ScopeComparison`.
#[derive(Debug, Clone)]
pub struct ScopeComparisonConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Comparison layout mode.
    pub mode: ComparisonMode,
    /// Divider line style.
    pub divider: DividerStyle,
    /// Whether to add "BEFORE" / "AFTER" labels.
    pub show_labels: bool,
    /// Label RGBA colour.
    pub label_color: [u8; 4],
}

impl Default for ScopeComparisonConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 256,
            mode: ComparisonMode::SideBySide,
            divider: DividerStyle::ThinWhite,
            show_labels: true,
            label_color: [255, 255, 255, 220],
        }
    }
}

/// A composited before/after scope comparison image.
#[derive(Debug, Clone)]
pub struct ScopeComparisonResult {
    /// RGBA pixel data, row-major.
    pub data: Vec<u8>,
    /// Width of the output image.
    pub width: u32,
    /// Height of the output image.
    pub height: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main rendering entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Produce a side-by-side (or other layout) comparison of two scope images.
///
/// # Arguments
///
/// * `before` — RGBA pixels for the "before" scope (`before_w × before_h`).
/// * `before_w` / `before_h` — dimensions of the before image.
/// * `after` — RGBA pixels for the "after" scope (`after_w × after_h`).
/// * `after_w` / `after_h` — dimensions of the after image.
/// * `config` — output configuration.
///
/// Both input images are scaled to fill the output region assigned to them.
///
/// # Errors
///
/// Returns an error if buffer sizes do not match declared dimensions.
pub fn render_scope_comparison(
    before: &[u8],
    before_w: u32,
    before_h: u32,
    after: &[u8],
    after_w: u32,
    after_h: u32,
    config: &ScopeComparisonConfig,
) -> OxiResult<ScopeComparisonResult> {
    validate_buffer(before, before_w, before_h)?;
    validate_buffer(after, after_w, after_h)?;

    let out_w = config.width;
    let out_h = config.height;
    let mut out = vec![0u8; (out_w * out_h * 4) as usize];

    // For each output pixel, decide which source to sample from.
    match config.mode {
        ComparisonMode::SideBySide => {
            let split_x = out_w / 2;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    if ox < split_x {
                        let px = sample_rgba(before, before_w, before_h, ox, oy, split_x, out_h);
                        put_pixel(&mut out, out_w, ox, oy, px);
                    } else {
                        let px = sample_rgba(
                            after,
                            after_w,
                            after_h,
                            ox - split_x,
                            oy,
                            out_w - split_x,
                            out_h,
                        );
                        put_pixel(&mut out, out_w, ox, oy, px);
                    }
                }
            }
        }
        ComparisonMode::TopBottom => {
            let split_y = out_h / 2;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    if oy < split_y {
                        let px = sample_rgba(before, before_w, before_h, ox, oy, out_w, split_y);
                        put_pixel(&mut out, out_w, ox, oy, px);
                    } else {
                        let px = sample_rgba(
                            after,
                            after_w,
                            after_h,
                            ox,
                            oy - split_y,
                            out_w,
                            out_h - split_y,
                        );
                        put_pixel(&mut out, out_w, ox, oy, px);
                    }
                }
            }
        }
        ComparisonMode::VerticalSplit { split } => {
            let split_x = ((split.clamp(0.0, 1.0) * out_w as f32).round() as u32).min(out_w);
            for oy in 0..out_h {
                for ox in 0..out_w {
                    if ox < split_x {
                        let px = sample_rgba(before, before_w, before_h, ox, oy, split_x, out_h);
                        put_pixel(&mut out, out_w, ox, oy, px);
                    } else {
                        let px = sample_rgba(
                            after,
                            after_w,
                            after_h,
                            ox - split_x,
                            oy,
                            out_w - split_x,
                            out_h,
                        );
                        put_pixel(&mut out, out_w, ox, oy, px);
                    }
                }
            }
        }
        ComparisonMode::HorizontalSplit { split } => {
            let split_y = ((split.clamp(0.0, 1.0) * out_h as f32).round() as u32).min(out_h);
            for oy in 0..out_h {
                for ox in 0..out_w {
                    if oy < split_y {
                        let px = sample_rgba(before, before_w, before_h, ox, oy, out_w, split_y);
                        put_pixel(&mut out, out_w, ox, oy, px);
                    } else {
                        let px = sample_rgba(
                            after,
                            after_w,
                            after_h,
                            ox,
                            oy - split_y,
                            out_w,
                            out_h - split_y,
                        );
                        put_pixel(&mut out, out_w, ox, oy, px);
                    }
                }
            }
        }
        ComparisonMode::DiagonalWipe { progress } => {
            let p = progress.clamp(0.0, 1.0);
            // Diagonal line from (0, p*h) to (p*w, 0) extended across the canvas.
            for oy in 0..out_h {
                for ox in 0..out_w {
                    // For a diagonal wipe: pixel is "before" if
                    // ox/out_w + oy/out_h < 2*(1-p)  i.e. below-left of wipe diagonal.
                    let wipe_val = ox as f32 / out_w as f32 + oy as f32 / out_h as f32;
                    if wipe_val < 2.0 * (1.0 - p) {
                        let px = sample_rgba(before, before_w, before_h, ox, oy, out_w, out_h);
                        put_pixel(&mut out, out_w, ox, oy, px);
                    } else {
                        let px = sample_rgba(after, after_w, after_h, ox, oy, out_w, out_h);
                        put_pixel(&mut out, out_w, ox, oy, px);
                    }
                }
            }
        }
    }

    // Draw divider line
    draw_divider(&mut out, out_w, out_h, config);

    // Optionally draw labels (simple pixel dots for "B" and "A" regions)
    if config.show_labels {
        draw_comparison_labels(&mut out, out_w, out_h, config);
    }

    Ok(ScopeComparisonResult {
        data: out,
        width: out_w,
        height: out_h,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn validate_buffer(buf: &[u8], w: u32, h: u32) -> OxiResult<()> {
    let expected = (w as usize) * (h as usize) * 4;
    if buf.len() < expected {
        return Err(OxiError::InvalidData(format!(
            "Buffer too small: need {expected}, got {}",
            buf.len()
        )));
    }
    Ok(())
}

/// Nearest-neighbour sample from an RGBA image into an output region of size
/// `region_w × region_h`.  `ox,oy` are coordinates within the region.
fn sample_rgba(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    ox: u32,
    oy: u32,
    region_w: u32,
    region_h: u32,
) -> [u8; 4] {
    if src_w == 0 || src_h == 0 || region_w == 0 || region_h == 0 {
        return [0; 4];
    }
    let sx = ((ox as u64 * src_w as u64) / region_w as u64).min(src_w as u64 - 1) as u32;
    let sy = ((oy as u64 * src_h as u64) / region_h as u64).min(src_h as u64 - 1) as u32;
    let idx = ((sy * src_w + sx) * 4) as usize;
    if idx + 3 >= src.len() {
        return [0; 4];
    }
    [src[idx], src[idx + 1], src[idx + 2], src[idx + 3]]
}

fn put_pixel(buf: &mut [u8], width: u32, x: u32, y: u32, color: [u8; 4]) {
    let idx = ((y * width + x) * 4) as usize;
    if idx + 3 < buf.len() {
        buf[idx] = color[0];
        buf[idx + 1] = color[1];
        buf[idx + 2] = color[2];
        buf[idx + 3] = color[3];
    }
}

fn draw_divider(out: &mut [u8], out_w: u32, out_h: u32, config: &ScopeComparisonConfig) {
    let (thickness, color): (u32, [u8; 4]) = match config.divider {
        DividerStyle::None => return,
        DividerStyle::ThinWhite => (1, [255, 255, 255, 255]),
        DividerStyle::ThinBlack => (2, [0, 0, 0, 255]),
        DividerStyle::ThickYellow => (3, [255, 220, 0, 255]),
    };

    match config.mode {
        ComparisonMode::SideBySide => {
            let x = out_w / 2;
            for t in 0..thickness {
                let xp = x.saturating_sub(thickness / 2) + t;
                for y in 0..out_h {
                    put_pixel(out, out_w, xp, y, color);
                }
            }
        }
        ComparisonMode::TopBottom => {
            let y = out_h / 2;
            for t in 0..thickness {
                let yp = y.saturating_sub(thickness / 2) + t;
                for x in 0..out_w {
                    put_pixel(out, out_w, x, yp, color);
                }
            }
        }
        ComparisonMode::VerticalSplit { split } => {
            let x = ((split.clamp(0.0, 1.0) * out_w as f32).round() as u32).min(out_w);
            for t in 0..thickness {
                let xp = x.saturating_sub(thickness / 2) + t;
                for y in 0..out_h {
                    put_pixel(out, out_w, xp, y, color);
                }
            }
        }
        ComparisonMode::HorizontalSplit { split } => {
            let y = ((split.clamp(0.0, 1.0) * out_h as f32).round() as u32).min(out_h);
            for t in 0..thickness {
                let yp = y.saturating_sub(thickness / 2) + t;
                for x in 0..out_w {
                    put_pixel(out, out_w, x, yp, color);
                }
            }
        }
        ComparisonMode::DiagonalWipe { .. } => {
            // No divider line for wipe mode (the boundary is a diagonal)
        }
    }
}

/// Draw small "B" (before) and "A" (after) indicator dots in the top corners.
fn draw_comparison_labels(out: &mut [u8], out_w: u32, out_h: u32, config: &ScopeComparisonConfig) {
    let color = config.label_color;
    // "B" indicator: top-left 4×4 block
    for dy in 0..4 {
        for dx in 0..4 {
            put_pixel(out, out_w, 4 + dx, 4 + dy, color);
        }
    }
    // "A" indicator: top-right 4×4 block
    let rx = out_w.saturating_sub(8);
    for dy in 0..4 {
        for dx in 0..4 {
            put_pixel(out, out_w, rx + dx, 4 + dy, color);
        }
    }
    let _ = out_h; // suppress warning
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
        let n = (w * h * 4) as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..(w * h) {
            v.extend_from_slice(&color);
        }
        v
    }

    #[test]
    fn test_side_by_side_produces_correct_size() {
        let before = solid_rgba(64, 64, [255, 0, 0, 255]);
        let after = solid_rgba(64, 64, [0, 0, 255, 255]);
        let cfg = ScopeComparisonConfig {
            width: 128,
            height: 64,
            mode: ComparisonMode::SideBySide,
            divider: DividerStyle::None,
            show_labels: false,
            ..Default::default()
        };
        let result = render_scope_comparison(&before, 64, 64, &after, 64, 64, &cfg);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        assert_eq!(out.data.len(), 128 * 64 * 4);
    }

    #[test]
    fn test_side_by_side_left_is_before() {
        let before = solid_rgba(64, 64, [200, 0, 0, 255]);
        let after = solid_rgba(64, 64, [0, 0, 200, 255]);
        let cfg = ScopeComparisonConfig {
            width: 128,
            height: 64,
            mode: ComparisonMode::SideBySide,
            divider: DividerStyle::None,
            show_labels: false,
            ..Default::default()
        };
        let out =
            render_scope_comparison(&before, 64, 64, &after, 64, 64, &cfg).expect("should succeed");
        // Left-most pixel row 0 should be red (before)
        let left_r = out.data[0];
        assert!(left_r > 100, "Left pixel should be red, got R={left_r}");
        // Right-most pixel row 0 should be blue (after)
        let right_idx = (127 * 4) as usize;
        let right_b = out.data[right_idx + 2];
        assert!(right_b > 100, "Right pixel should be blue, got B={right_b}");
    }

    #[test]
    fn test_top_bottom_produces_correct_size() {
        let before = solid_rgba(64, 32, [255, 0, 0, 255]);
        let after = solid_rgba(64, 32, [0, 255, 0, 255]);
        let cfg = ScopeComparisonConfig {
            width: 64,
            height: 64,
            mode: ComparisonMode::TopBottom,
            divider: DividerStyle::None,
            show_labels: false,
            ..Default::default()
        };
        let result = render_scope_comparison(&before, 64, 32, &after, 64, 32, &cfg);
        assert!(result.is_ok());
        assert_eq!(result.expect("should succeed").data.len(), 64 * 64 * 4);
    }

    #[test]
    fn test_vertical_split_at_30_percent() {
        let before = solid_rgba(32, 32, [255, 0, 0, 255]);
        let after = solid_rgba(32, 32, [0, 0, 255, 255]);
        let cfg = ScopeComparisonConfig {
            width: 100,
            height: 32,
            mode: ComparisonMode::VerticalSplit { split: 0.3 },
            divider: DividerStyle::None,
            show_labels: false,
            ..Default::default()
        };
        let result = render_scope_comparison(&before, 32, 32, &after, 32, 32, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_horizontal_split() {
        let before = solid_rgba(64, 64, [255, 0, 0, 255]);
        let after = solid_rgba(64, 64, [0, 255, 0, 255]);
        let cfg = ScopeComparisonConfig {
            width: 64,
            height: 64,
            mode: ComparisonMode::HorizontalSplit { split: 0.5 },
            divider: DividerStyle::ThinWhite,
            show_labels: false,
            ..Default::default()
        };
        let result = render_scope_comparison(&before, 64, 64, &after, 64, 64, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_diagonal_wipe_full_before() {
        let before = solid_rgba(64, 64, [255, 0, 0, 255]);
        let after = solid_rgba(64, 64, [0, 0, 255, 255]);
        let cfg = ScopeComparisonConfig {
            width: 64,
            height: 64,
            mode: ComparisonMode::DiagonalWipe { progress: 0.0 },
            divider: DividerStyle::None,
            show_labels: false,
            ..Default::default()
        };
        let out =
            render_scope_comparison(&before, 64, 64, &after, 64, 64, &cfg).expect("should succeed");
        // With progress=0.0, everything should be before (red)
        let r = out.data[0];
        assert!(r > 100, "All before (progress=0) should be red, R={r}");
    }

    #[test]
    fn test_diagonal_wipe_full_after() {
        let before = solid_rgba(64, 64, [255, 0, 0, 255]);
        let after = solid_rgba(64, 64, [0, 0, 255, 255]);
        let cfg = ScopeComparisonConfig {
            width: 64,
            height: 64,
            mode: ComparisonMode::DiagonalWipe { progress: 1.0 },
            divider: DividerStyle::None,
            show_labels: false,
            ..Default::default()
        };
        let out =
            render_scope_comparison(&before, 64, 64, &after, 64, 64, &cfg).expect("should succeed");
        // With progress=1.0, bottom-right corner should be after (blue)
        let idx = ((63 * 64 + 63) * 4) as usize;
        let b = out.data[idx + 2];
        assert!(b > 100, "Bottom-right should be blue (after), B={b}");
    }

    #[test]
    fn test_invalid_buffer_returns_error() {
        let before = vec![0u8; 10]; // too small for 10×10
        let after = solid_rgba(10, 10, [0, 0, 255, 255]);
        let cfg = ScopeComparisonConfig::default();
        let result = render_scope_comparison(&before, 10, 10, &after, 10, 10, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_labels_does_not_panic() {
        let before = solid_rgba(64, 32, [255, 0, 0, 255]);
        let after = solid_rgba(64, 32, [0, 0, 255, 255]);
        let cfg = ScopeComparisonConfig {
            width: 128,
            height: 32,
            mode: ComparisonMode::SideBySide,
            divider: DividerStyle::ThickYellow,
            show_labels: true,
            label_color: [255, 255, 0, 255],
        };
        let result = render_scope_comparison(&before, 64, 32, &after, 64, 32, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_comparison_config_default() {
        let cfg = ScopeComparisonConfig::default();
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.mode, ComparisonMode::SideBySide);
        assert!(cfg.show_labels);
    }
}

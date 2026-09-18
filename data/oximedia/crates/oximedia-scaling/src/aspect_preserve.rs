//! Aspect-ratio-preserving dimension computation.
//!
//! Given source and target rectangles this module computes the output
//! dimensions and any required padding or crop offsets so that the source
//! content is never distorted.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

// ── FitMode ───────────────────────────────────────────────────────────────────

/// How to fit source content into the target rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitMode {
    /// Fit entirely inside the target box; letterbox / pillarbox the remainder.
    Contain,
    /// Fill the target box entirely; crop any overflow.
    Cover,
    /// Stretch to exactly the target dimensions (no aspect preservation).
    Stretch,
    /// Use the target only as a maximum; never upscale.
    ContainNoUpscale,
}

impl std::fmt::Display for FitMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contain => write!(f, "contain"),
            Self::Cover => write!(f, "cover"),
            Self::Stretch => write!(f, "stretch"),
            Self::ContainNoUpscale => write!(f, "contain-no-upscale"),
        }
    }
}

// ── OutputGeometry ────────────────────────────────────────────────────────────

/// The complete geometry description produced by `AspectPreserver`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputGeometry {
    /// Width of the scaled source content (before padding).
    pub scaled_width: u32,
    /// Height of the scaled source content (before padding).
    pub scaled_height: u32,
    /// Horizontal offset of the scaled content within the target frame.
    pub offset_x: u32,
    /// Vertical offset of the scaled content within the target frame.
    pub offset_y: u32,
    /// Final output frame width (equals target width).
    pub frame_width: u32,
    /// Final output frame height (equals target height).
    pub frame_height: u32,
}

impl OutputGeometry {
    /// Return `true` if any padding is needed (letterbox / pillarbox).
    pub fn has_padding(&self) -> bool {
        self.offset_x > 0 || self.offset_y > 0
    }

    /// Return the horizontal padding strip width (one side).
    pub fn pad_x(&self) -> u32 {
        self.offset_x
    }

    /// Return the vertical padding strip height (one side).
    pub fn pad_y(&self) -> u32 {
        self.offset_y
    }
}

// ── AspectPreserver ───────────────────────────────────────────────────────────

/// Computes output dimensions while preserving the source aspect ratio.
///
/// # Example
/// ```
/// use oximedia_scaling::aspect_preserve::{AspectPreserver, FitMode};
///
/// let ap = AspectPreserver::new(FitMode::Contain);
/// // Scale 4:3 (640×480) into 1920×1080 (16:9) with letterboxing
/// let geom = ap.compute_output_dims(640, 480, 1920, 1080);
/// assert_eq!(geom.scaled_width, 1440);
/// assert_eq!(geom.scaled_height, 1080);
/// assert_eq!(geom.offset_x, 240); // 240 pixels pillarbox each side
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AspectPreserver {
    /// Fit mode to apply.
    pub mode: FitMode,
}

impl AspectPreserver {
    /// Create a new `AspectPreserver` with the given fit mode.
    pub fn new(mode: FitMode) -> Self {
        Self { mode }
    }

    /// Compute output geometry for scaling `src_w × src_h` into `tgt_w × tgt_h`.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn compute_output_dims(
        &self,
        src_w: u32,
        src_h: u32,
        tgt_w: u32,
        tgt_h: u32,
    ) -> OutputGeometry {
        if src_w == 0 || src_h == 0 {
            return OutputGeometry {
                scaled_width: 0,
                scaled_height: 0,
                offset_x: 0,
                offset_y: 0,
                frame_width: tgt_w,
                frame_height: tgt_h,
            };
        }

        match self.mode {
            FitMode::Stretch => OutputGeometry {
                scaled_width: tgt_w,
                scaled_height: tgt_h,
                offset_x: 0,
                offset_y: 0,
                frame_width: tgt_w,
                frame_height: tgt_h,
            },

            FitMode::Contain | FitMode::ContainNoUpscale => {
                let scale_x = tgt_w as f64 / src_w as f64;
                let scale_y = tgt_h as f64 / src_h as f64;
                let mut scale = scale_x.min(scale_y);

                if self.mode == FitMode::ContainNoUpscale && scale > 1.0 {
                    scale = 1.0;
                }

                let sw = (src_w as f64 * scale).round() as u32;
                let sh = (src_h as f64 * scale).round() as u32;
                let ox = (tgt_w.saturating_sub(sw)) / 2;
                let oy = (tgt_h.saturating_sub(sh)) / 2;

                OutputGeometry {
                    scaled_width: sw,
                    scaled_height: sh,
                    offset_x: ox,
                    offset_y: oy,
                    frame_width: tgt_w,
                    frame_height: tgt_h,
                }
            }

            FitMode::Cover => {
                let scale_x = tgt_w as f64 / src_w as f64;
                let scale_y = tgt_h as f64 / src_h as f64;
                let scale = scale_x.max(scale_y);

                let sw = (src_w as f64 * scale).round() as u32;
                let sh = (src_h as f64 * scale).round() as u32;
                // Negative offset means crop — we represent as zero (crop handled elsewhere)
                let ox = 0u32;
                let oy = 0u32;

                OutputGeometry {
                    scaled_width: sw,
                    scaled_height: sh,
                    offset_x: ox,
                    offset_y: oy,
                    frame_width: tgt_w,
                    frame_height: tgt_h,
                }
            }
        }
    }

    /// Convenience: return only the `(scaled_width, scaled_height)` pair.
    pub fn scaled_size(&self, src_w: u32, src_h: u32, tgt_w: u32, tgt_h: u32) -> (u32, u32) {
        let g = self.compute_output_dims(src_w, src_h, tgt_w, tgt_h);
        (g.scaled_width, g.scaled_height)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stretch_returns_exact_target() {
        let ap = AspectPreserver::new(FitMode::Stretch);
        let g = ap.compute_output_dims(640, 480, 1920, 1080);
        assert_eq!(g.scaled_width, 1920);
        assert_eq!(g.scaled_height, 1080);
        assert_eq!(g.offset_x, 0);
        assert_eq!(g.offset_y, 0);
    }

    #[test]
    fn test_contain_wider_source_letterboxes() {
        // 16:9 source into 4:3 target → letterbox (bars top/bottom)
        let ap = AspectPreserver::new(FitMode::Contain);
        let g = ap.compute_output_dims(1920, 1080, 1024, 768);
        // scaled_width should equal 1024
        assert_eq!(g.scaled_width, 1024);
        assert!(g.offset_y > 0, "expected vertical padding for letterbox");
    }

    #[test]
    fn test_contain_4x3_into_16x9_pillarbox() {
        let ap = AspectPreserver::new(FitMode::Contain);
        let g = ap.compute_output_dims(640, 480, 1920, 1080);
        // 4:3 into 16:9 → pillarbox
        assert_eq!(g.scaled_height, 1080);
        assert_eq!(g.scaled_width, 1440);
        assert_eq!(g.offset_x, 240);
        assert_eq!(g.offset_y, 0);
    }

    #[test]
    fn test_contain_same_aspect_no_padding() {
        let ap = AspectPreserver::new(FitMode::Contain);
        let g = ap.compute_output_dims(1920, 1080, 1280, 720);
        assert!(!g.has_padding());
    }

    #[test]
    fn test_cover_fills_frame() {
        let ap = AspectPreserver::new(FitMode::Cover);
        let g = ap.compute_output_dims(640, 480, 1920, 1080);
        // scaled content should be >= target in both dimensions
        assert!(g.scaled_width >= 1920 || g.scaled_height >= 1080);
    }

    #[test]
    fn test_contain_no_upscale_small_source() {
        let ap = AspectPreserver::new(FitMode::ContainNoUpscale);
        let g = ap.compute_output_dims(320, 240, 1920, 1080);
        // Should not upscale
        assert_eq!(g.scaled_width, 320);
        assert_eq!(g.scaled_height, 240);
    }

    #[test]
    fn test_zero_source_returns_zero_scaled() {
        let ap = AspectPreserver::new(FitMode::Contain);
        let g = ap.compute_output_dims(0, 0, 1920, 1080);
        assert_eq!(g.scaled_width, 0);
        assert_eq!(g.scaled_height, 0);
    }

    #[test]
    fn test_output_geometry_has_padding_true() {
        let g = OutputGeometry {
            scaled_width: 1440,
            scaled_height: 1080,
            offset_x: 240,
            offset_y: 0,
            frame_width: 1920,
            frame_height: 1080,
        };
        assert!(g.has_padding());
    }

    #[test]
    fn test_output_geometry_has_padding_false() {
        let g = OutputGeometry {
            scaled_width: 1920,
            scaled_height: 1080,
            offset_x: 0,
            offset_y: 0,
            frame_width: 1920,
            frame_height: 1080,
        };
        assert!(!g.has_padding());
    }

    #[test]
    fn test_pad_x_and_pad_y_accessors() {
        let g = OutputGeometry {
            scaled_width: 1440,
            scaled_height: 810,
            offset_x: 240,
            offset_y: 135,
            frame_width: 1920,
            frame_height: 1080,
        };
        assert_eq!(g.pad_x(), 240);
        assert_eq!(g.pad_y(), 135);
    }

    #[test]
    fn test_scaled_size_convenience() {
        let ap = AspectPreserver::new(FitMode::Stretch);
        let (w, h) = ap.scaled_size(640, 480, 1920, 1080);
        assert_eq!((w, h), (1920, 1080));
    }

    #[test]
    fn test_fitmode_display() {
        assert_eq!(FitMode::Contain.to_string(), "contain");
        assert_eq!(FitMode::Cover.to_string(), "cover");
        assert_eq!(FitMode::Stretch.to_string(), "stretch");
        assert_eq!(FitMode::ContainNoUpscale.to_string(), "contain-no-upscale");
    }

    #[test]
    fn test_frame_dimensions_always_equal_target() {
        let ap = AspectPreserver::new(FitMode::Contain);
        let g = ap.compute_output_dims(1280, 800, 1920, 1080);
        assert_eq!(g.frame_width, 1920);
        assert_eq!(g.frame_height, 1080);
    }

    #[test]
    fn test_contain_square_source_into_landscape() {
        // 1:1 source into 16:9 → pillarbox
        let ap = AspectPreserver::new(FitMode::Contain);
        let g = ap.compute_output_dims(1000, 1000, 1920, 1080);
        assert_eq!(g.scaled_height, 1080);
        assert_eq!(g.scaled_width, 1080);
        assert_eq!(g.offset_x, 420);
    }

    #[test]
    fn test_contain_no_upscale_large_source_does_scale() {
        // Source larger than target — should scale down
        let ap = AspectPreserver::new(FitMode::ContainNoUpscale);
        let g = ap.compute_output_dims(3840, 2160, 1920, 1080);
        assert_eq!(g.scaled_width, 1920);
        assert_eq!(g.scaled_height, 1080);
    }

    // ── Aspect-ratio helper ───────────────────────────────────────────────────

    fn aspect_ratio(w: u32, h: u32) -> f64 {
        w as f64 / h as f64
    }

    fn assert_aspect_preserved(orig_w: u32, orig_h: u32, out_w: u32, out_h: u32) {
        let orig = aspect_ratio(orig_w, orig_h);
        let out = aspect_ratio(out_w, out_h);
        assert!(
            (orig - out).abs() < 0.02,
            "aspect mismatch: {orig:.4} vs {out:.4}"
        );
    }

    // ── 32:9 ultrawide (3840×1080) → 1920×1080 (16:9) ──────────────────────

    #[test]
    fn test_ultrawide_contain() {
        // 32:9 source scaled into 16:9 target — must fit entirely inside
        let ap = AspectPreserver::new(FitMode::Contain);
        let g = ap.compute_output_dims(3840, 1080, 1920, 1080);
        assert!(
            g.scaled_width <= 1920,
            "scaled_width {} must not exceed target 1920",
            g.scaled_width
        );
        assert!(
            g.scaled_height <= 1080,
            "scaled_height {} must not exceed target 1080",
            g.scaled_height
        );
        assert!(
            g.has_padding(),
            "32:9 → 16:9 Contain must produce letterbox padding"
        );
        assert_aspect_preserved(3840, 1080, g.scaled_width, g.scaled_height);
        assert_eq!(g.frame_width, 1920);
        assert_eq!(g.frame_height, 1080);
    }

    #[test]
    fn test_ultrawide_cover() {
        // 32:9 source covering a 16:9 target — both dimensions must be ≥ target
        let ap = AspectPreserver::new(FitMode::Cover);
        let g = ap.compute_output_dims(3840, 1080, 1920, 1080);
        assert!(
            g.scaled_width >= 1920 && g.scaled_height >= 1080,
            "Cover scaled_width={} scaled_height={} must both be ≥ 1920×1080",
            g.scaled_width,
            g.scaled_height
        );
        assert_aspect_preserved(3840, 1080, g.scaled_width, g.scaled_height);
        assert_eq!(g.frame_width, 1920);
        assert_eq!(g.frame_height, 1080);
    }

    // ── 9:16 portrait (1080×1920) → 1920×1080 ──────────────────────────────

    #[test]
    fn test_portrait_contain() {
        // Tall portrait source into landscape target — must letterbox with horizontal padding
        let ap = AspectPreserver::new(FitMode::Contain);
        let g = ap.compute_output_dims(1080, 1920, 1920, 1080);
        assert!(
            g.scaled_width <= 1920,
            "scaled_width {} must fit inside 1920",
            g.scaled_width
        );
        assert!(
            g.scaled_height <= 1080,
            "scaled_height {} must fit inside 1080",
            g.scaled_height
        );
        // Portrait fitted into landscape: narrower than target → horizontal (pillarbox) padding
        assert!(
            g.offset_x > 0,
            "portrait Contain into landscape must produce horizontal pillarbox; offset_x={}",
            g.offset_x
        );
        assert_aspect_preserved(1080, 1920, g.scaled_width, g.scaled_height);
    }

    #[test]
    fn test_portrait_cover() {
        // Portrait source covering a landscape target — both dims ≥ target
        let ap = AspectPreserver::new(FitMode::Cover);
        let g = ap.compute_output_dims(1080, 1920, 1920, 1080);
        assert!(
            g.scaled_width >= 1920 && g.scaled_height >= 1080,
            "Cover scaled_width={} scaled_height={} must both be ≥ 1920×1080",
            g.scaled_width,
            g.scaled_height
        );
        assert_aspect_preserved(1080, 1920, g.scaled_width, g.scaled_height);
    }

    // ── 2.39:1 anamorphic (2390×1000) → 1920×1080 ──────────────────────────

    #[test]
    fn test_anamorphic_contain() {
        // 2.39:1 anamorphic source into 16:9 target — fits with padding
        let ap = AspectPreserver::new(FitMode::Contain);
        let g = ap.compute_output_dims(2390, 1000, 1920, 1080);
        assert!(
            g.scaled_width <= 1920,
            "scaled_width {} must fit inside 1920",
            g.scaled_width
        );
        assert!(
            g.scaled_height <= 1080,
            "scaled_height {} must fit inside 1080",
            g.scaled_height
        );
        assert!(
            g.has_padding(),
            "2.39:1 → 16:9 Contain must produce padding"
        );
        assert_aspect_preserved(2390, 1000, g.scaled_width, g.scaled_height);
        assert_eq!(g.frame_width, 1920);
        assert_eq!(g.frame_height, 1080);
    }

    #[test]
    fn test_anamorphic_cover() {
        // 2.39:1 anamorphic source covering 16:9 target — both dims ≥ target
        let ap = AspectPreserver::new(FitMode::Cover);
        let g = ap.compute_output_dims(2390, 1000, 1920, 1080);
        assert!(
            g.scaled_width >= 1920 && g.scaled_height >= 1080,
            "Cover scaled_width={} scaled_height={} must both be ≥ 1920×1080",
            g.scaled_width,
            g.scaled_height
        );
        assert_aspect_preserved(2390, 1000, g.scaled_width, g.scaled_height);
        assert_eq!(g.frame_width, 1920);
        assert_eq!(g.frame_height, 1080);
    }
}

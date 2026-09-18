//! Timecode burn-in overlay module
//!
//! Provides types and helpers for rendering timecode text onto video frames.

#[allow(dead_code)]
/// Visual style of the timecode burn-in
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BurnInStyle {
    /// Classic broadcast style (large, bold)
    Classic,
    /// Modern minimalist style
    Modern,
    /// Minimal style (smallest footprint)
    Minimal,
}

impl BurnInStyle {
    /// Returns a relative font scale factor for this style
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn font_scale(&self) -> f32 {
        match self {
            BurnInStyle::Classic => 2.0,
            BurnInStyle::Modern => 1.5,
            BurnInStyle::Minimal => 1.0,
        }
    }
}

#[allow(dead_code)]
/// Where to anchor the timecode overlay on the frame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnInPosition {
    /// Top-left corner
    TopLeft,
    /// Top-center
    TopCenter,
    /// Top-right corner
    TopRight,
    /// Bottom-left corner
    BottomLeft,
    /// Bottom-center
    BottomCenter,
    /// Bottom-right corner
    BottomRight,
}

impl BurnInPosition {
    /// Returns `true` if the position is in the top half of the frame
    #[must_use]
    pub fn is_top(&self) -> bool {
        matches!(
            self,
            BurnInPosition::TopLeft | BurnInPosition::TopCenter | BurnInPosition::TopRight
        )
    }

    /// Returns `true` if the position is on the right side of the frame
    #[must_use]
    pub fn is_right(&self) -> bool {
        matches!(self, BurnInPosition::TopRight | BurnInPosition::BottomRight)
    }
}

#[allow(dead_code)]
/// Complete overlay specification for timecode burn-in
#[derive(Debug, Clone)]
pub struct TimecodeOverlay {
    /// Visual style
    pub style: BurnInStyle,
    /// Position on frame
    pub position: BurnInPosition,
    /// Background rectangle alpha (0.0 = transparent, 1.0 = opaque)
    pub background_alpha: f32,
    /// Text colour as [R, G, B]
    pub color: [u8; 3],
}

impl TimecodeOverlay {
    /// Standard broadcast overlay: Classic style, bottom-center, semi-transparent black
    #[must_use]
    pub fn default_broadcast() -> Self {
        Self {
            style: BurnInStyle::Classic,
            position: BurnInPosition::BottomCenter,
            background_alpha: 0.5,
            color: [255, 255, 255],
        }
    }

    /// Dailies overlay: Modern style, top-left, no background
    #[must_use]
    pub fn default_dailies() -> Self {
        Self {
            style: BurnInStyle::Modern,
            position: BurnInPosition::TopLeft,
            background_alpha: 0.0,
            color: [255, 255, 0],
        }
    }
}

/// Compute the (x, y) pixel position for a timecode string overlay.
///
/// The returned coordinates represent the top-left corner of the rendered text
/// box. `tc` is expected to be a formatted timecode string such as "01:02:03:04".
/// `frame_width` and `frame_height` are the frame dimensions in pixels.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn render_timecode_text(
    tc: &str,
    frame_width: u32,
    frame_height: u32,
    overlay: &TimecodeOverlay,
) -> (u32, u32) {
    // Approximate glyph dimensions based on font scale
    let char_w = (12.0 * overlay.style.font_scale()) as u32;
    let char_h = (20.0 * overlay.style.font_scale()) as u32;
    let text_w = char_w * tc.len() as u32;
    let margin = 16u32;

    let x = match overlay.position {
        BurnInPosition::TopLeft | BurnInPosition::BottomLeft => margin,
        BurnInPosition::TopCenter | BurnInPosition::BottomCenter => {
            frame_width.saturating_sub(text_w) / 2
        }
        BurnInPosition::TopRight | BurnInPosition::BottomRight => {
            frame_width.saturating_sub(text_w + margin)
        }
    };

    let y = if overlay.position.is_top() {
        margin
    } else {
        frame_height.saturating_sub(char_h + margin)
    };

    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_burn_in_style_classic_scale() {
        assert!((BurnInStyle::Classic.font_scale() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_burn_in_style_modern_scale() {
        assert!((BurnInStyle::Modern.font_scale() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_burn_in_style_minimal_scale() {
        assert!((BurnInStyle::Minimal.font_scale() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_position_is_top_true() {
        assert!(BurnInPosition::TopLeft.is_top());
        assert!(BurnInPosition::TopCenter.is_top());
        assert!(BurnInPosition::TopRight.is_top());
    }

    #[test]
    fn test_position_is_top_false() {
        assert!(!BurnInPosition::BottomLeft.is_top());
        assert!(!BurnInPosition::BottomCenter.is_top());
        assert!(!BurnInPosition::BottomRight.is_top());
    }

    #[test]
    fn test_position_is_right_true() {
        assert!(BurnInPosition::TopRight.is_right());
        assert!(BurnInPosition::BottomRight.is_right());
    }

    #[test]
    fn test_position_is_right_false() {
        assert!(!BurnInPosition::TopLeft.is_right());
        assert!(!BurnInPosition::BottomCenter.is_right());
    }

    #[test]
    fn test_default_broadcast_style() {
        let o = TimecodeOverlay::default_broadcast();
        assert_eq!(o.style, BurnInStyle::Classic);
        assert_eq!(o.position, BurnInPosition::BottomCenter);
        assert!(!o.position.is_top());
    }

    #[test]
    fn test_default_dailies_style() {
        let o = TimecodeOverlay::default_dailies();
        assert_eq!(o.style, BurnInStyle::Modern);
        assert_eq!(o.position, BurnInPosition::TopLeft);
        assert!(o.position.is_top());
    }

    #[test]
    fn test_render_timecode_text_bottom_center() {
        let overlay = TimecodeOverlay::default_broadcast();
        let (x, y) = render_timecode_text("01:02:03:04", 1920, 1080, &overlay);
        // x should be roughly centered
        assert!(x < 1920);
        // y should be in lower portion
        assert!(y > 1080 / 2);
    }

    #[test]
    fn test_render_timecode_text_top_left() {
        let overlay = TimecodeOverlay::default_dailies();
        let (x, y) = render_timecode_text("01:02:03:04", 1920, 1080, &overlay);
        // x should be near left margin
        assert!(x < 100);
        // y should be near top
        assert!(y < 100);
    }

    #[test]
    fn test_render_timecode_text_top_right() {
        let overlay = TimecodeOverlay {
            style: BurnInStyle::Minimal,
            position: BurnInPosition::TopRight,
            background_alpha: 0.0,
            color: [255, 255, 255],
        };
        let (x, _y) = render_timecode_text("01:02:03:04", 1920, 1080, &overlay);
        // x should be towards right side
        assert!(x > 1920 / 2);
    }
}

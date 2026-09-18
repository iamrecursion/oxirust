//! Visible image watermarking for copyright and branding overlays.
//!
//! Unlike imperceptible audio watermarking, a *visible watermark* is
//! intentionally perceivable to viewers.  This module models the position,
//! opacity, and application logic for overlaying such marks on a pixel buffer.

#![allow(dead_code)]

/// Canonical placement positions for a visible watermark on a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WatermarkPosition {
    /// Top-left corner of the frame.
    TopLeft,
    /// Top-right corner of the frame.
    TopRight,
    /// Bottom-left corner of the frame.
    BottomLeft,
    /// Bottom-right corner of the frame.
    BottomRight,
    /// Horizontally and vertically centred.
    Center,
    /// Custom pixel offset from the top-left corner `(x, y)`.
    Custom(u32, u32),
}

impl WatermarkPosition {
    /// Resolve a pixel `(x, y)` coordinate for the watermark origin given the
    /// frame dimensions and the watermark dimensions.
    ///
    /// The returned coordinates represent the top-left corner of the watermark.
    /// A margin of 16 pixels is applied for edge positions.
    #[must_use]
    pub fn resolve(
        self,
        frame_width: u32,
        frame_height: u32,
        wm_width: u32,
        wm_height: u32,
    ) -> (u32, u32) {
        const MARGIN: u32 = 16;
        match self {
            Self::TopLeft => (MARGIN, MARGIN),
            Self::TopRight => (
                frame_width.saturating_sub(wm_width).saturating_sub(MARGIN),
                MARGIN,
            ),
            Self::BottomLeft => (
                MARGIN,
                frame_height
                    .saturating_sub(wm_height)
                    .saturating_sub(MARGIN),
            ),
            Self::BottomRight => (
                frame_width.saturating_sub(wm_width).saturating_sub(MARGIN),
                frame_height
                    .saturating_sub(wm_height)
                    .saturating_sub(MARGIN),
            ),
            Self::Center => (
                frame_width.saturating_sub(wm_width) / 2,
                frame_height.saturating_sub(wm_height) / 2,
            ),
            Self::Custom(x, y) => (x, y),
        }
    }
}

/// Opacity level for a visible watermark (0.0 = transparent, 1.0 = opaque).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WatermarkOpacity(f32);

impl WatermarkOpacity {
    /// Create an opacity value clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Raw opacity value.
    #[must_use]
    pub fn value(self) -> f32 {
        self.0
    }

    /// Fully transparent (no watermark visible).
    #[must_use]
    pub fn transparent() -> Self {
        Self(0.0)
    }

    /// Fully opaque.
    #[must_use]
    pub fn opaque() -> Self {
        Self(1.0)
    }

    /// Returns `true` if opacity is effectively zero.
    #[must_use]
    pub fn is_transparent(self) -> bool {
        self.0 < 1e-6
    }
}

/// A visible watermark definition: label text, position, and opacity.
///
/// In a real implementation the label would reference a pre-rendered image
/// or vector graphic.  Here it is represented as a simple string for
/// portability without external dependencies.
#[derive(Debug, Clone)]
pub struct VisibleWatermark {
    /// Textual label carried by this watermark (e.g. "© 2024 Acme Corp").
    pub label: String,
    /// Placement within the target frame.
    pub position: WatermarkPosition,
    /// Blending opacity.
    pub opacity: WatermarkOpacity,
    /// Nominal width of the watermark in pixels.
    pub width: u32,
    /// Nominal height of the watermark in pixels.
    pub height: u32,
}

impl VisibleWatermark {
    /// Create a new visible watermark.
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        position: WatermarkPosition,
        opacity: WatermarkOpacity,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            label: label.into(),
            position,
            opacity,
            width,
            height,
        }
    }

    /// Compute where this watermark should be placed on a frame of the given
    /// dimensions and return a [`WatermarkApplication`] describing the operation.
    ///
    /// Returns `None` when the watermark would be completely invisible (opacity ≈ 0).
    #[must_use]
    pub fn apply(&self, frame_width: u32, frame_height: u32) -> Option<WatermarkApplication> {
        if self.opacity.is_transparent() {
            return None;
        }

        let (x, y) = self
            .position
            .resolve(frame_width, frame_height, self.width, self.height);

        Some(WatermarkApplication {
            label: self.label.clone(),
            x,
            y,
            width: self.width,
            height: self.height,
            opacity: self.opacity.value(),
        })
    }

    /// Whether the watermark is currently visible (non-zero opacity).
    #[must_use]
    pub fn is_visible(&self) -> bool {
        !self.opacity.is_transparent()
    }
}

/// Describes how a watermark should be composited onto a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct WatermarkApplication {
    /// Watermark label/content identifier.
    pub label: String,
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Width of the composited region in pixels.
    pub width: u32,
    /// Height of the composited region in pixels.
    pub height: u32,
    /// Alpha blending factor `[0.0, 1.0]`.
    pub opacity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- WatermarkPosition ---

    #[test]
    fn test_top_left_position() {
        let (x, y) = WatermarkPosition::TopLeft.resolve(1920, 1080, 200, 50);
        assert_eq!(x, 16);
        assert_eq!(y, 16);
    }

    #[test]
    fn test_top_right_position() {
        let (x, y) = WatermarkPosition::TopRight.resolve(1920, 1080, 200, 50);
        assert_eq!(x, 1920 - 200 - 16);
        assert_eq!(y, 16);
    }

    #[test]
    fn test_bottom_left_position() {
        let (x, y) = WatermarkPosition::BottomLeft.resolve(1920, 1080, 200, 50);
        assert_eq!(x, 16);
        assert_eq!(y, 1080 - 50 - 16);
    }

    #[test]
    fn test_bottom_right_position() {
        let (x, y) = WatermarkPosition::BottomRight.resolve(1920, 1080, 200, 50);
        assert_eq!(x, 1920 - 200 - 16);
        assert_eq!(y, 1080 - 50 - 16);
    }

    #[test]
    fn test_center_position() {
        let (x, y) = WatermarkPosition::Center.resolve(1920, 1080, 200, 50);
        assert_eq!(x, (1920 - 200) / 2);
        assert_eq!(y, (1080 - 50) / 2);
    }

    #[test]
    fn test_custom_position() {
        let (x, y) = WatermarkPosition::Custom(100, 200).resolve(1920, 1080, 50, 50);
        assert_eq!(x, 100);
        assert_eq!(y, 200);
    }

    // --- WatermarkOpacity ---

    #[test]
    fn test_opacity_clamping_below() {
        let o = WatermarkOpacity::new(-0.5);
        assert!((o.value() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_opacity_clamping_above() {
        let o = WatermarkOpacity::new(2.0);
        assert!((o.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_opacity_transparent_flag() {
        assert!(WatermarkOpacity::transparent().is_transparent());
        assert!(!WatermarkOpacity::opaque().is_transparent());
    }

    // --- VisibleWatermark ---

    #[test]
    fn test_apply_returns_none_when_transparent() {
        let wm = VisibleWatermark::new(
            "Hidden",
            WatermarkPosition::Center,
            WatermarkOpacity::transparent(),
            200,
            50,
        );
        assert!(wm.apply(1920, 1080).is_none());
        assert!(!wm.is_visible());
    }

    #[test]
    fn test_apply_returns_application_when_visible() {
        let wm = VisibleWatermark::new(
            "© 2024",
            WatermarkPosition::BottomRight,
            WatermarkOpacity::new(0.5),
            200,
            50,
        );
        let app = wm.apply(1920, 1080).expect("should produce application");
        assert_eq!(app.label, "© 2024");
        assert!((app.opacity - 0.5).abs() < 1e-6);
        assert!(wm.is_visible());
    }

    #[test]
    fn test_apply_correct_coordinates_top_left() {
        let wm = VisibleWatermark::new(
            "Logo",
            WatermarkPosition::TopLeft,
            WatermarkOpacity::opaque(),
            100,
            40,
        );
        let app = wm.apply(1280, 720).expect("should succeed in test");
        assert_eq!(app.x, 16);
        assert_eq!(app.y, 16);
    }

    #[test]
    fn test_apply_dimensions_preserved() {
        let wm = VisibleWatermark::new(
            "Mark",
            WatermarkPosition::Center,
            WatermarkOpacity::opaque(),
            300,
            80,
        );
        let app = wm.apply(1920, 1080).expect("should succeed in test");
        assert_eq!(app.width, 300);
        assert_eq!(app.height, 80);
    }
}

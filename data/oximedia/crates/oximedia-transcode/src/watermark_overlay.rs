#![allow(dead_code)]
//! Watermark and graphic overlay embedding during transcoding.
//!
//! Supports text watermarks, image logos, and timed overlays that are burned
//! into the output video at configurable positions, sizes, and opacity levels.

use std::fmt;

/// Anchor position for overlay placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayPosition {
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Centred horizontally and vertically.
    Center,
    /// Custom pixel offset from top-left.
    Custom {
        /// X offset in pixels.
        x: u32,
        /// Y offset in pixels.
        y: u32,
    },
}

impl fmt::Display for OverlayPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TopLeft => write!(f, "top-left"),
            Self::TopRight => write!(f, "top-right"),
            Self::BottomLeft => write!(f, "bottom-left"),
            Self::BottomRight => write!(f, "bottom-right"),
            Self::Center => write!(f, "center"),
            Self::Custom { x, y } => write!(f, "custom({x},{y})"),
        }
    }
}

impl OverlayPosition {
    /// Resolve to pixel coordinates given frame and overlay dimensions.
    /// Returns `(x, y)` for the top-left corner of the overlay.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn resolve(self, frame_w: u32, frame_h: u32, overlay_w: u32, overlay_h: u32) -> (u32, u32) {
        match self {
            Self::TopLeft => (0, 0),
            Self::TopRight => (frame_w.saturating_sub(overlay_w), 0),
            Self::BottomLeft => (0, frame_h.saturating_sub(overlay_h)),
            Self::BottomRight => (
                frame_w.saturating_sub(overlay_w),
                frame_h.saturating_sub(overlay_h),
            ),
            Self::Center => (
                frame_w.saturating_sub(overlay_w) / 2,
                frame_h.saturating_sub(overlay_h) / 2,
            ),
            Self::Custom { x, y } => (x, y),
        }
    }
}

/// Kind of watermark content.
#[derive(Debug, Clone, PartialEq)]
pub enum WatermarkContent {
    /// A text string to render.
    Text(String),
    /// A path to an image file (PNG, etc.).
    ImageFile(String),
    /// Raw RGBA pixel data with width and height.
    RawRgba {
        /// Width of the raw image.
        width: u32,
        /// Height of the raw image.
        height: u32,
        /// Pixel data in RGBA order.
        data: Vec<u8>,
    },
}

/// Configuration for a single watermark overlay.
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// Content to overlay.
    pub content: WatermarkContent,
    /// Position anchor.
    pub position: OverlayPosition,
    /// Opacity from 0.0 (invisible) to 1.0 (fully opaque).
    pub opacity: f32,
    /// Scale factor for the overlay (1.0 = original size).
    pub scale: f32,
    /// Optional margin from the anchor edge in pixels.
    pub margin: u32,
    /// Optional start time in seconds (overlay appears at this time).
    pub start_time: Option<f64>,
    /// Optional end time in seconds (overlay disappears at this time).
    pub end_time: Option<f64>,
}

impl WatermarkConfig {
    /// Create a text watermark with default settings.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: WatermarkContent::Text(text.into()),
            position: OverlayPosition::BottomRight,
            opacity: 0.5,
            scale: 1.0,
            margin: 10,
            start_time: None,
            end_time: None,
        }
    }

    /// Create an image-file watermark.
    pub fn image(path: impl Into<String>) -> Self {
        Self {
            content: WatermarkContent::ImageFile(path.into()),
            position: OverlayPosition::BottomRight,
            opacity: 0.8,
            scale: 1.0,
            margin: 10,
            start_time: None,
            end_time: None,
        }
    }

    /// Set position.
    #[must_use]
    pub fn with_position(mut self, pos: OverlayPosition) -> Self {
        self.position = pos;
        self
    }

    /// Set opacity (clamped to 0.0 - 1.0).
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Set scale factor.
    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale.max(0.01);
        self
    }

    /// Set margin.
    #[must_use]
    pub fn with_margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Set time range visibility.
    #[must_use]
    pub fn with_time_range(mut self, start: f64, end: f64) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    /// Check whether the watermark is visible at a given timestamp.
    #[must_use]
    pub fn is_visible_at(&self, time_seconds: f64) -> bool {
        if let Some(start) = self.start_time {
            if time_seconds < start {
                return false;
            }
        }
        if let Some(end) = self.end_time {
            if time_seconds > end {
                return false;
            }
        }
        true
    }

    /// Compute effective opacity at the given timestamp.
    #[must_use]
    pub fn effective_opacity(&self, time_seconds: f64) -> f32 {
        if self.is_visible_at(time_seconds) {
            self.opacity
        } else {
            0.0
        }
    }
}

/// Aggregate overlay pipeline that composes multiple watermarks.
#[derive(Debug, Clone)]
pub struct OverlayPipeline {
    /// Ordered list of watermark layers.
    layers: Vec<WatermarkConfig>,
    /// Output frame width.
    frame_width: u32,
    /// Output frame height.
    frame_height: u32,
}

impl OverlayPipeline {
    /// Create a new overlay pipeline for a given frame size.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            layers: Vec::new(),
            frame_width: width,
            frame_height: height,
        }
    }

    /// Add a watermark layer.
    pub fn add_layer(&mut self, config: WatermarkConfig) {
        self.layers.push(config);
    }

    /// Return the number of layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Return layers that are visible at the given timestamp.
    #[must_use]
    pub fn visible_layers_at(&self, time_seconds: f64) -> Vec<&WatermarkConfig> {
        self.layers
            .iter()
            .filter(|l| l.is_visible_at(time_seconds))
            .collect()
    }

    /// Clear all layers.
    pub fn clear(&mut self) {
        self.layers.clear();
    }

    /// Return the configured frame dimensions.
    #[must_use]
    pub fn frame_size(&self) -> (u32, u32) {
        (self.frame_width, self.frame_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_display() {
        assert_eq!(OverlayPosition::TopLeft.to_string(), "top-left");
        assert_eq!(OverlayPosition::TopRight.to_string(), "top-right");
        assert_eq!(OverlayPosition::BottomLeft.to_string(), "bottom-left");
        assert_eq!(OverlayPosition::BottomRight.to_string(), "bottom-right");
        assert_eq!(OverlayPosition::Center.to_string(), "center");
        assert_eq!(
            OverlayPosition::Custom { x: 10, y: 20 }.to_string(),
            "custom(10,20)"
        );
    }

    #[test]
    fn test_position_resolve_top_left() {
        let (x, y) = OverlayPosition::TopLeft.resolve(1920, 1080, 100, 50);
        assert_eq!((x, y), (0, 0));
    }

    #[test]
    fn test_position_resolve_bottom_right() {
        let (x, y) = OverlayPosition::BottomRight.resolve(1920, 1080, 100, 50);
        assert_eq!((x, y), (1820, 1030));
    }

    #[test]
    fn test_position_resolve_center() {
        let (x, y) = OverlayPosition::Center.resolve(1920, 1080, 100, 80);
        assert_eq!((x, y), (910, 500));
    }

    #[test]
    fn test_position_resolve_custom() {
        let (x, y) = OverlayPosition::Custom { x: 42, y: 99 }.resolve(1920, 1080, 100, 100);
        assert_eq!((x, y), (42, 99));
    }

    #[test]
    fn test_text_watermark_defaults() {
        let wm = WatermarkConfig::text("(c) 2024");
        assert_eq!(wm.position, OverlayPosition::BottomRight);
        assert!((wm.opacity - 0.5).abs() < f32::EPSILON);
        assert!((wm.scale - 1.0).abs() < f32::EPSILON);
        assert_eq!(wm.margin, 10);
        assert!(wm.start_time.is_none());
    }

    #[test]
    fn test_image_watermark_defaults() {
        let wm = WatermarkConfig::image("/logo.png");
        assert!((wm.opacity - 0.8).abs() < f32::EPSILON);
        match &wm.content {
            WatermarkContent::ImageFile(p) => assert_eq!(p, "/logo.png"),
            _ => panic!("expected ImageFile"),
        }
    }

    #[test]
    fn test_opacity_clamp() {
        let wm = WatermarkConfig::text("x").with_opacity(2.0);
        assert!((wm.opacity - 1.0).abs() < f32::EPSILON);
        let wm2 = WatermarkConfig::text("x").with_opacity(-1.0);
        assert!((wm2.opacity - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_visibility_always() {
        let wm = WatermarkConfig::text("x");
        assert!(wm.is_visible_at(0.0));
        assert!(wm.is_visible_at(9999.0));
    }

    #[test]
    fn test_visibility_timed() {
        let wm = WatermarkConfig::text("x").with_time_range(5.0, 10.0);
        assert!(!wm.is_visible_at(3.0));
        assert!(wm.is_visible_at(7.0));
        assert!(!wm.is_visible_at(12.0));
    }

    #[test]
    fn test_effective_opacity() {
        let wm = WatermarkConfig::text("x")
            .with_opacity(0.75)
            .with_time_range(5.0, 10.0);
        assert!((wm.effective_opacity(7.0) - 0.75).abs() < f32::EPSILON);
        assert!((wm.effective_opacity(1.0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pipeline_add_layers() {
        let mut pipeline = OverlayPipeline::new(1920, 1080);
        assert_eq!(pipeline.layer_count(), 0);
        pipeline.add_layer(WatermarkConfig::text("A"));
        pipeline.add_layer(WatermarkConfig::text("B"));
        assert_eq!(pipeline.layer_count(), 2);
    }

    #[test]
    fn test_pipeline_visible_layers() {
        let mut pipeline = OverlayPipeline::new(1920, 1080);
        pipeline.add_layer(WatermarkConfig::text("always"));
        pipeline.add_layer(WatermarkConfig::text("timed").with_time_range(5.0, 10.0));

        assert_eq!(pipeline.visible_layers_at(0.0).len(), 1);
        assert_eq!(pipeline.visible_layers_at(7.0).len(), 2);
    }

    #[test]
    fn test_pipeline_clear() {
        let mut pipeline = OverlayPipeline::new(1920, 1080);
        pipeline.add_layer(WatermarkConfig::text("x"));
        pipeline.clear();
        assert_eq!(pipeline.layer_count(), 0);
    }

    #[test]
    fn test_pipeline_frame_size() {
        let pipeline = OverlayPipeline::new(3840, 2160);
        assert_eq!(pipeline.frame_size(), (3840, 2160));
    }
}

//! Crop and scale pipeline for video transcoding.
//!
//! Provides aspect ratio preservation, pillarbox/letterbox padding,
//! and smart crop detection for video scaling operations.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Describes a rectangular region in a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// X offset from the left edge.
    pub x: u32,
    /// Y offset from the top edge.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Rect {
    /// Creates a new rectangle.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Returns the aspect ratio as a float.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        f64::from(self.width) / f64::from(self.height)
    }

    /// Returns the area in pixels.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// How to handle aspect ratio mismatches during scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectMode {
    /// Add black bars to preserve the source aspect ratio (letterbox or pillarbox).
    Pad,
    /// Crop the source to fill the target resolution.
    Crop,
    /// Stretch to fill ignoring aspect ratio.
    Stretch,
    /// Scale to fit entirely within the target (may leave empty space).
    Fit,
}

/// Where to place the image when padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadAlignment {
    /// Center the image (default).
    Center,
    /// Align to the top-left.
    TopLeft,
    /// Align to the bottom-right.
    BottomRight,
}

/// The type of padding being applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadType {
    /// Horizontal bars (letterbox) — source is wider than target.
    Letterbox,
    /// Vertical bars (pillarbox) — source is taller than target.
    Pillarbox,
    /// No padding required.
    None,
}

/// Configuration for a crop-and-scale operation.
#[derive(Debug, Clone)]
pub struct CropScaleConfig {
    /// Source resolution.
    pub source_width: u32,
    /// Source height in pixels.
    pub source_height: u32,
    /// Target width in pixels.
    pub target_width: u32,
    /// Target height in pixels.
    pub target_height: u32,
    /// Aspect ratio handling mode.
    pub aspect_mode: AspectMode,
    /// Padding alignment when using `AspectMode::Pad`.
    pub pad_alignment: PadAlignment,
    /// Padding color as (R, G, B).
    pub pad_color: (u8, u8, u8),
    /// Optional manual crop region applied before scaling.
    pub manual_crop: Option<Rect>,
}

impl CropScaleConfig {
    /// Creates a new crop/scale configuration.
    #[must_use]
    pub fn new(
        source_width: u32,
        source_height: u32,
        target_width: u32,
        target_height: u32,
    ) -> Self {
        Self {
            source_width,
            source_height,
            target_width,
            target_height,
            aspect_mode: AspectMode::Pad,
            pad_alignment: PadAlignment::Center,
            pad_color: (0, 0, 0),
            manual_crop: None,
        }
    }

    /// Sets the aspect ratio handling mode.
    #[must_use]
    pub fn with_aspect_mode(mut self, mode: AspectMode) -> Self {
        self.aspect_mode = mode;
        self
    }

    /// Sets the padding alignment.
    #[must_use]
    pub fn with_pad_alignment(mut self, alignment: PadAlignment) -> Self {
        self.pad_alignment = alignment;
        self
    }

    /// Sets the padding color (used for letterbox/pillarbox bars).
    #[must_use]
    pub fn with_pad_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.pad_color = (r, g, b);
        self
    }

    /// Sets a manual crop region applied before scaling.
    #[must_use]
    pub fn with_manual_crop(mut self, crop: Rect) -> Self {
        self.manual_crop = Some(crop);
        self
    }

    /// Returns the source aspect ratio.
    #[must_use]
    pub fn source_aspect(&self) -> f64 {
        f64::from(self.source_width) / f64::from(self.source_height)
    }

    /// Returns the target aspect ratio.
    #[must_use]
    pub fn target_aspect(&self) -> f64 {
        f64::from(self.target_width) / f64::from(self.target_height)
    }

    /// Determines the type of padding needed.
    #[must_use]
    pub fn pad_type(&self) -> PadType {
        let sa = self.source_aspect();
        let ta = self.target_aspect();
        if (sa - ta).abs() < 1e-4 {
            PadType::None
        } else if sa > ta {
            // Source is wider than target: add horizontal bars
            PadType::Letterbox
        } else {
            // Source is taller (narrower) than target: add vertical bars
            PadType::Pillarbox
        }
    }

    /// Computes the scaled region (before padding) that fits within the target.
    ///
    /// Returns `(scaled_width, scaled_height)`.
    #[must_use]
    pub fn compute_scaled_size(&self) -> (u32, u32) {
        let source_w = f64::from(self.source_width);
        let source_h = f64::from(self.source_height);
        let target_w = f64::from(self.target_width);
        let target_h = f64::from(self.target_height);

        match self.aspect_mode {
            AspectMode::Stretch => (self.target_width, self.target_height),
            AspectMode::Pad | AspectMode::Fit => {
                let scale = (target_w / source_w).min(target_h / source_h);
                let w = (source_w * scale).round() as u32;
                let h = (source_h * scale).round() as u32;
                (w, h)
            }
            AspectMode::Crop => {
                let scale = (target_w / source_w).max(target_h / source_h);
                let w = (source_w * scale).round() as u32;
                let h = (source_h * scale).round() as u32;
                (w, h)
            }
        }
    }

    /// Computes the padding offsets when using `AspectMode::Pad`.
    ///
    /// Returns `(x_offset, y_offset)` for the image within the padded frame.
    #[must_use]
    pub fn compute_pad_offsets(&self) -> (u32, u32) {
        let (sw, sh) = self.compute_scaled_size();
        match self.pad_alignment {
            PadAlignment::Center => {
                let x = (self.target_width.saturating_sub(sw)) / 2;
                let y = (self.target_height.saturating_sub(sh)) / 2;
                (x, y)
            }
            PadAlignment::TopLeft => (0, 0),
            PadAlignment::BottomRight => {
                let x = self.target_width.saturating_sub(sw);
                let y = self.target_height.saturating_sub(sh);
                (x, y)
            }
        }
    }

    /// Computes the crop rect when using `AspectMode::Crop`.
    #[must_use]
    pub fn compute_crop_rect(&self) -> Rect {
        let (sw, sh) = self.compute_scaled_size();
        let x = (sw.saturating_sub(self.target_width)) / 2;
        let y = (sh.saturating_sub(self.target_height)) / 2;
        Rect::new(x, y, self.target_width, self.target_height)
    }
}

/// Smart crop detector that finds areas of interest in a frame.
#[derive(Debug, Clone)]
pub struct SmartCropDetector {
    /// Minimum saliency threshold (0.0–1.0).
    pub saliency_threshold: f32,
    /// Weight given to face regions.
    pub face_weight: f32,
    /// Weight given to motion regions.
    pub motion_weight: f32,
}

impl SmartCropDetector {
    /// Creates a new smart crop detector with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            saliency_threshold: 0.5,
            face_weight: 2.0,
            motion_weight: 1.5,
        }
    }

    /// Sets the saliency threshold.
    #[must_use]
    pub fn with_saliency_threshold(mut self, threshold: f32) -> Self {
        self.saliency_threshold = threshold;
        self
    }

    /// Computes a mock crop region centered on the "region of interest".
    ///
    /// In a real implementation this would analyze frame content.
    #[must_use]
    pub fn compute_crop(
        &self,
        frame_width: u32,
        frame_height: u32,
        target_width: u32,
        target_height: u32,
    ) -> Rect {
        // Default: center crop
        let x = (frame_width.saturating_sub(target_width)) / 2;
        let y = (frame_height.saturating_sub(target_height)) / 2;
        let w = target_width.min(frame_width);
        let h = target_height.min(frame_height);
        Rect::new(x, y, w, h)
    }
}

impl Default for SmartCropDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_aspect_ratio() {
        let r = Rect::new(0, 0, 1920, 1080);
        assert!((r.aspect_ratio() - 16.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_rect_area() {
        let r = Rect::new(0, 0, 1920, 1080);
        assert_eq!(r.area(), 1920 * 1080);
    }

    #[test]
    fn test_pad_type_none_when_same_aspect() {
        let cfg = CropScaleConfig::new(1920, 1080, 1280, 720);
        assert_eq!(cfg.pad_type(), PadType::None);
    }

    #[test]
    fn test_pad_type_letterbox() {
        // 16:9 source into 4:3 target → letterbox (horizontal bars)
        let cfg = CropScaleConfig::new(1920, 1080, 1024, 768);
        assert_eq!(cfg.pad_type(), PadType::Letterbox);
    }

    #[test]
    fn test_pad_type_pillarbox() {
        // 4:3 source into 16:9 target → pillarbox (vertical bars)
        let cfg = CropScaleConfig::new(1024, 768, 1920, 1080);
        assert_eq!(cfg.pad_type(), PadType::Pillarbox);
    }

    #[test]
    fn test_compute_scaled_size_pad() {
        let cfg = CropScaleConfig::new(1920, 1080, 1280, 720).with_aspect_mode(AspectMode::Pad);
        let (w, h) = cfg.compute_scaled_size();
        assert_eq!(w, 1280);
        assert_eq!(h, 720);
    }

    #[test]
    fn test_compute_scaled_size_stretch() {
        let cfg = CropScaleConfig::new(1920, 1080, 800, 600).with_aspect_mode(AspectMode::Stretch);
        let (w, h) = cfg.compute_scaled_size();
        assert_eq!(w, 800);
        assert_eq!(h, 600);
    }

    #[test]
    fn test_compute_pad_offsets_center() {
        // 4:3 into 16:9 => pillarbox, centered
        let cfg = CropScaleConfig::new(1024, 768, 1920, 1080)
            .with_aspect_mode(AspectMode::Pad)
            .with_pad_alignment(PadAlignment::Center);
        let (x, y) = cfg.compute_pad_offsets();
        // Scaled: 1080/768 * 1024 = 1440 wide, 1080 tall
        assert_eq!(y, 0); // full height used
        assert!(x > 0); // some horizontal padding
    }

    #[test]
    fn test_compute_pad_offsets_topleft() {
        let cfg = CropScaleConfig::new(1024, 768, 1920, 1080)
            .with_aspect_mode(AspectMode::Pad)
            .with_pad_alignment(PadAlignment::TopLeft);
        let (x, y) = cfg.compute_pad_offsets();
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn test_compute_crop_rect() {
        let cfg = CropScaleConfig::new(1920, 1080, 1280, 720).with_aspect_mode(AspectMode::Crop);
        let rect = cfg.compute_crop_rect();
        assert_eq!(rect.width, 1280);
        assert_eq!(rect.height, 720);
    }

    #[test]
    fn test_pad_color() {
        let cfg = CropScaleConfig::new(1920, 1080, 1280, 720).with_pad_color(255, 255, 255);
        assert_eq!(cfg.pad_color, (255, 255, 255));
    }

    #[test]
    fn test_manual_crop() {
        let crop = Rect::new(100, 50, 1720, 980);
        let cfg = CropScaleConfig::new(1920, 1080, 1280, 720).with_manual_crop(crop);
        assert!(cfg.manual_crop.is_some());
        assert_eq!(cfg.manual_crop.expect("should succeed in test").x, 100);
    }

    #[test]
    fn test_smart_crop_detector_default() {
        let det = SmartCropDetector::new();
        assert!((det.saliency_threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_smart_crop_computes_center_crop() {
        let det = SmartCropDetector::new();
        let rect = det.compute_crop(1920, 1080, 1280, 720);
        assert_eq!(rect.width, 1280);
        assert_eq!(rect.height, 720);
        assert_eq!(rect.x, (1920 - 1280) / 2);
        assert_eq!(rect.y, (1080 - 720) / 2);
    }
}

#![allow(dead_code)]
//! Frame padding (letterbox / pillarbox) operations.

/// How padding pixels are filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    /// Fill with a solid colour.
    Color,
    /// Replicate the nearest edge pixel.
    Replicate,
    /// Mirror the content at the frame edge.
    Mirror,
    /// Blur and stretch the content to fill the padding area.
    BlurredBackground,
}

/// An RGB colour used for solid padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadColor {
    /// Red channel (0–255).
    pub r: u8,
    /// Green channel (0–255).
    pub g: u8,
    /// Blue channel (0–255).
    pub b: u8,
}

impl PadColor {
    /// Create a new [`PadColor`].
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Standard broadcast black.
    pub const BLACK: PadColor = PadColor::new(0, 0, 0);
    /// Standard broadcast white.
    pub const WHITE: PadColor = PadColor::new(255, 255, 255);
}

/// Amount of padding to add to each edge of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddingConfig {
    /// Pixels added to the left edge.
    pub left: u32,
    /// Pixels added to the right edge.
    pub right: u32,
    /// Pixels added to the top edge.
    pub top: u32,
    /// Pixels added to the bottom edge.
    pub bottom: u32,
    /// Padding fill mode.
    pub mode: PadMode,
    /// Colour used when `mode` is [`PadMode::Color`].
    pub color: PadColor,
}

impl PaddingConfig {
    /// Create a new uniform padding configuration.
    pub fn uniform(amount: u32, mode: PadMode) -> Self {
        Self {
            left: amount,
            right: amount,
            top: amount,
            bottom: amount,
            mode,
            color: PadColor::BLACK,
        }
    }

    /// Create a symmetric horizontal / vertical padding configuration.
    pub fn symmetric(horizontal: u32, vertical: u32, mode: PadMode) -> Self {
        Self {
            left: horizontal,
            right: horizontal,
            top: vertical,
            bottom: vertical,
            mode,
            color: PadColor::BLACK,
        }
    }

    /// Override the pad colour.
    pub fn with_color(mut self, color: PadColor) -> Self {
        self.color = color;
        self
    }

    /// Total horizontal padding (left + right).
    pub fn total_horizontal(&self) -> u32 {
        self.left + self.right
    }

    /// Total vertical padding (top + bottom).
    pub fn total_vertical(&self) -> u32 {
        self.top + self.bottom
    }
}

/// Applies padding to a source frame and computes output dimensions.
#[derive(Debug, Clone)]
pub struct PadOperation {
    /// Padding configuration.
    pub config: PaddingConfig,
}

impl PadOperation {
    /// Create a new [`PadOperation`].
    pub fn new(config: PaddingConfig) -> Self {
        Self { config }
    }

    /// Compute the output dimensions after padding `(src_w, src_h)`.
    pub fn output_dimensions(&self, src_w: u32, src_h: u32) -> (u32, u32) {
        (
            src_w + self.config.total_horizontal(),
            src_h + self.config.total_vertical(),
        )
    }

    /// Build a letterbox padding config to fit `src` into `target` keeping
    /// aspect ratio.  Padding is added above/below (top/bottom only).
    ///
    /// Returns `None` if the source is already taller than the target, or if
    /// the target is narrower than the source.
    pub fn letterbox(
        src_w: u32,
        src_h: u32,
        target_w: u32,
        target_h: u32,
    ) -> Option<PaddingConfig> {
        if src_w > target_w || src_h > target_h {
            return None;
        }
        let pad_v = target_h - src_h;
        let top = pad_v / 2;
        let bottom = pad_v - top;
        Some(PaddingConfig {
            left: (target_w - src_w) / 2,
            right: target_w - src_w - (target_w - src_w) / 2,
            top,
            bottom,
            mode: PadMode::Color,
            color: PadColor::BLACK,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_color_black() {
        assert_eq!(PadColor::BLACK, PadColor::new(0, 0, 0));
    }

    #[test]
    fn test_pad_color_white() {
        assert_eq!(PadColor::WHITE, PadColor::new(255, 255, 255));
    }

    #[test]
    fn test_uniform_padding() {
        let cfg = PaddingConfig::uniform(10, PadMode::Color);
        assert_eq!(cfg.left, 10);
        assert_eq!(cfg.right, 10);
        assert_eq!(cfg.top, 10);
        assert_eq!(cfg.bottom, 10);
    }

    #[test]
    fn test_symmetric_padding() {
        let cfg = PaddingConfig::symmetric(20, 5, PadMode::Replicate);
        assert_eq!(cfg.total_horizontal(), 40);
        assert_eq!(cfg.total_vertical(), 10);
    }

    #[test]
    fn test_total_horizontal() {
        let cfg = PaddingConfig {
            left: 30,
            right: 50,
            top: 0,
            bottom: 0,
            mode: PadMode::Color,
            color: PadColor::BLACK,
        };
        assert_eq!(cfg.total_horizontal(), 80);
    }

    #[test]
    fn test_total_vertical() {
        let cfg = PaddingConfig {
            left: 0,
            right: 0,
            top: 45,
            bottom: 45,
            mode: PadMode::Color,
            color: PadColor::BLACK,
        };
        assert_eq!(cfg.total_vertical(), 90);
    }

    #[test]
    fn test_with_color() {
        let cfg = PaddingConfig::uniform(0, PadMode::Color).with_color(PadColor::WHITE);
        assert_eq!(cfg.color, PadColor::WHITE);
    }

    #[test]
    fn test_output_dimensions() {
        let cfg = PaddingConfig::symmetric(0, 45, PadMode::Color);
        let op = PadOperation::new(cfg);
        let (w, h) = op.output_dimensions(1920, 990);
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }

    #[test]
    fn test_letterbox_valid() {
        // Fit 1920x810 into 1920x1080: 270 px of vertical padding
        let cfg = PadOperation::letterbox(1920, 810, 1920, 1080).expect("should succeed in test");
        assert_eq!(cfg.top + cfg.bottom, 270);
        assert_eq!(cfg.left + cfg.right, 0);
    }

    #[test]
    fn test_letterbox_exact_fit() {
        let cfg = PadOperation::letterbox(1920, 1080, 1920, 1080).expect("should succeed in test");
        assert_eq!(cfg.total_horizontal(), 0);
        assert_eq!(cfg.total_vertical(), 0);
    }

    #[test]
    fn test_letterbox_too_large() {
        assert!(PadOperation::letterbox(3840, 2160, 1920, 1080).is_none());
    }

    #[test]
    fn test_pad_mode_variants_exist() {
        let modes = [
            PadMode::Color,
            PadMode::Replicate,
            PadMode::Mirror,
            PadMode::BlurredBackground,
        ];
        assert_eq!(modes.len(), 4);
    }
}

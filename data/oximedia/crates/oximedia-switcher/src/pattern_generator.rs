//! Internal test-pattern generator for the video switcher.
#![allow(dead_code)]

/// Available test pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternType {
    /// Standard SMPTE 75% colour bars.
    ColorBars,
    /// Full black (0 IRE).
    Black,
    /// Full white / 100% white field.
    White,
    /// Solid flat colour matte (configurable colour).
    Matte,
    /// Horizontal or vertical gradient ramp.
    Gradient,
}

impl PatternType {
    /// Return a short human-readable label for this pattern.
    pub fn label(self) -> &'static str {
        match self {
            Self::ColorBars => "Color Bars",
            Self::Black => "Black",
            Self::White => "White",
            Self::Matte => "Matte",
            Self::Gradient => "Gradient",
        }
    }

    /// Return `true` if this pattern type changes over time (animated).
    pub fn is_inherently_animated(self) -> bool {
        false // All test patterns are static by default
    }
}

/// Configuration for a pattern generator.
#[derive(Debug, Clone)]
pub struct PatternConfig {
    /// The pattern type to generate.
    pub pattern_type: PatternType,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Whether the pattern should animate (e.g. scrolling gradient).
    pub animated: bool,
    /// RGBA matte colour (used when `pattern_type == Matte`).
    pub matte_color: [u8; 4],
}

impl PatternConfig {
    /// Create a default config for the given pattern type at 1920x1080.
    pub fn new(pattern_type: PatternType) -> Self {
        Self {
            pattern_type,
            width: 1920,
            height: 1080,
            animated: false,
            matte_color: [128, 128, 128, 255],
        }
    }

    /// Return `true` if animation is enabled.
    pub fn is_animated(&self) -> bool {
        self.animated
    }

    /// Enable or disable animation.
    pub fn with_animated(mut self, animated: bool) -> Self {
        self.animated = animated;
        self
    }

    /// Set the matte colour as RGBA bytes.
    pub fn with_matte_color(mut self, rgba: [u8; 4]) -> Self {
        self.matte_color = rgba;
        self
    }
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self::new(PatternType::ColorBars)
    }
}

/// A generated pattern frame (simplified to a flat colour for this implementation).
#[derive(Debug, Clone)]
pub struct PatternFrame {
    /// Width of the frame in pixels.
    pub width: u32,
    /// Height of the frame in pixels.
    pub height: u32,
    /// Identifier of the pattern type that generated this frame.
    pub pattern: PatternType,
    /// Frame index (for animated patterns).
    pub frame_index: u64,
}

/// Pattern generator that produces test-pattern frames.
#[derive(Debug)]
pub struct PatternGenerator {
    config: PatternConfig,
    frame_index: u64,
}

impl PatternGenerator {
    /// Create a new generator with the given config.
    pub fn new(config: PatternConfig) -> Self {
        Self {
            config,
            frame_index: 0,
        }
    }

    /// Generate the next frame.
    pub fn generate(&mut self) -> PatternFrame {
        let frame = PatternFrame {
            width: self.config.width,
            height: self.config.height,
            pattern: self.config.pattern_type,
            frame_index: self.frame_index,
        };
        if self.config.animated {
            self.frame_index += 1;
        }
        frame
    }

    /// Return the currently configured active pattern type.
    pub fn active_pattern(&self) -> PatternType {
        self.config.pattern_type
    }

    /// Change the active pattern. Resets the frame index.
    pub fn set_pattern(&mut self, pattern_type: PatternType) {
        self.config.pattern_type = pattern_type;
        self.frame_index = 0;
    }

    /// Return the current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_type_label_not_empty() {
        for pt in [
            PatternType::ColorBars,
            PatternType::Black,
            PatternType::White,
            PatternType::Matte,
            PatternType::Gradient,
        ] {
            assert!(!pt.label().is_empty());
        }
    }

    #[test]
    fn test_pattern_type_label_values() {
        assert_eq!(PatternType::ColorBars.label(), "Color Bars");
        assert_eq!(PatternType::Black.label(), "Black");
        assert_eq!(PatternType::Gradient.label(), "Gradient");
    }

    #[test]
    fn test_pattern_type_not_inherently_animated() {
        for pt in [
            PatternType::ColorBars,
            PatternType::Black,
            PatternType::Gradient,
        ] {
            assert!(!pt.is_inherently_animated());
        }
    }

    #[test]
    fn test_config_new_defaults() {
        let cfg = PatternConfig::new(PatternType::Black);
        assert_eq!(cfg.pattern_type, PatternType::Black);
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert!(!cfg.is_animated());
    }

    #[test]
    fn test_config_with_animated() {
        let cfg = PatternConfig::new(PatternType::Gradient).with_animated(true);
        assert!(cfg.is_animated());
    }

    #[test]
    fn test_config_with_matte_color() {
        let cfg = PatternConfig::new(PatternType::Matte).with_matte_color([255, 0, 0, 255]);
        assert_eq!(cfg.matte_color, [255, 0, 0, 255]);
    }

    #[test]
    fn test_config_default_is_color_bars() {
        let cfg = PatternConfig::default();
        assert_eq!(cfg.pattern_type, PatternType::ColorBars);
    }

    #[test]
    fn test_generator_active_pattern() {
        let gen = PatternGenerator::new(PatternConfig::new(PatternType::White));
        assert_eq!(gen.active_pattern(), PatternType::White);
    }

    #[test]
    fn test_generator_generate_returns_correct_pattern() {
        let mut gen = PatternGenerator::new(PatternConfig::new(PatternType::ColorBars));
        let frame = gen.generate();
        assert_eq!(frame.pattern, PatternType::ColorBars);
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
    }

    #[test]
    fn test_generator_static_frame_index_does_not_advance() {
        let mut gen = PatternGenerator::new(PatternConfig::new(PatternType::Black));
        gen.generate();
        gen.generate();
        assert_eq!(gen.frame_index(), 0);
    }

    #[test]
    fn test_generator_animated_frame_index_advances() {
        let cfg = PatternConfig::new(PatternType::Gradient).with_animated(true);
        let mut gen = PatternGenerator::new(cfg);
        gen.generate();
        gen.generate();
        assert_eq!(gen.frame_index(), 2);
    }

    #[test]
    fn test_generator_set_pattern_resets_index() {
        let cfg = PatternConfig::new(PatternType::Gradient).with_animated(true);
        let mut gen = PatternGenerator::new(cfg);
        gen.generate();
        gen.generate();
        gen.set_pattern(PatternType::Black);
        assert_eq!(gen.frame_index(), 0);
        assert_eq!(gen.active_pattern(), PatternType::Black);
    }
}

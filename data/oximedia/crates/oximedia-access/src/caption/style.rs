//! Caption styling and formatting with adaptive font sizing.

use oximedia_subtitle::style::SubtitleStyle;
use serde::{Deserialize, Serialize};

/// Caption style configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionStyle {
    /// Font size in pixels.
    pub font_size: u32,
    /// Font family.
    pub font_family: String,
    /// Text color (RGBA).
    pub text_color: (u8, u8, u8, u8),
    /// Background color (RGBA).
    pub background_color: (u8, u8, u8, u8),
    /// Edge/outline style.
    pub edge_style: EdgeStyle,
    /// Edge color (RGBA).
    pub edge_color: (u8, u8, u8, u8),
    /// Opacity (0-255).
    pub opacity: u8,
}

// ─── Adaptive font sizing ────────────────────────────────────────────────────

/// Viewing context for adaptive font sizing.
///
/// Encapsulates display resolution and estimated viewing distance so that
/// font sizes can be scaled to maintain legibility across screen sizes,
/// projectors, mobile devices, and TVs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewingContext {
    /// Display width in physical pixels.
    pub display_width_px: u32,
    /// Display height in physical pixels.
    pub display_height_px: u32,
    /// Estimated viewing distance in centimetres.
    pub viewing_distance_cm: f32,
    /// Physical screen diagonal in centimetres.
    pub screen_diagonal_cm: f32,
    /// Whether the viewer has declared a low-vision accessibility need.
    pub low_vision_mode: bool,
}

impl ViewingContext {
    /// 1080p monitor at typical desk distance (~60 cm).
    #[must_use]
    pub fn desktop_1080p() -> Self {
        Self {
            display_width_px: 1920,
            display_height_px: 1080,
            viewing_distance_cm: 60.0,
            screen_diagonal_cm: 55.0, // ~22 inch
            low_vision_mode: false,
        }
    }

    /// 4K television at typical living-room distance (~250 cm).
    #[must_use]
    pub fn tv_4k() -> Self {
        Self {
            display_width_px: 3840,
            display_height_px: 2160,
            viewing_distance_cm: 250.0,
            screen_diagonal_cm: 139.0, // ~55 inch
            low_vision_mode: false,
        }
    }

    /// Full-HD smartphone at close reading distance (~30 cm).
    #[must_use]
    pub fn mobile_fhd() -> Self {
        Self {
            display_width_px: 1080,
            display_height_px: 1920,
            viewing_distance_cm: 30.0,
            screen_diagonal_cm: 16.0, // ~6.3 inch
            low_vision_mode: false,
        }
    }

    /// Enable low-vision mode.
    #[must_use]
    pub const fn with_low_vision(mut self, enabled: bool) -> Self {
        self.low_vision_mode = enabled;
        self
    }

    /// Pixels Per Degree of visual angle (PPD).
    ///
    /// A higher PPD means the display is finer relative to the distance,
    /// so text can use fewer pixels and still be legible.
    #[must_use]
    pub fn pixels_per_degree(&self) -> f32 {
        // Horizontal field-of-view in degrees for one pixel:
        //   angle = 2 * atan(screen_width_cm / (2 * distance_cm))
        // PPD = display_width_px / total_horizontal_angle
        if self.viewing_distance_cm <= 0.0 || self.screen_diagonal_cm <= 0.0 {
            return 40.0; // safe default
        }
        // Estimate screen width from diagonal and 16:9 assumption
        let aspect = self.display_width_px as f32 / self.display_height_px.max(1) as f32;
        let screen_width_cm = self.screen_diagonal_cm * (aspect / (1.0 + aspect * aspect).sqrt());
        let half_angle = (screen_width_cm / (2.0 * self.viewing_distance_cm)).atan();
        let total_angle_deg = 2.0 * half_angle * (180.0 / std::f32::consts::PI);
        if total_angle_deg <= 0.0 {
            return 40.0;
        }
        self.display_width_px as f32 / total_angle_deg
    }

    /// Compute the optimal caption font size in pixels.
    ///
    /// Uses the angular subtended by a legibility-tested character height
    /// (approximately 0.45°).  BBC/EBU guidelines suggest caption text should
    /// be around 0.35–0.5 degrees of visual angle.  A `multiplier` of 1.0
    /// corresponds to the reference size; values above 1.0 scale up.
    #[must_use]
    pub fn optimal_font_size_px(&self, multiplier: f32) -> u32 {
        const TARGET_ANGLE_DEGREES: f32 = 0.45;
        let ppd = self.pixels_per_degree();
        let base_px = (TARGET_ANGLE_DEGREES * ppd).round();

        let scaled = base_px * multiplier.clamp(0.5, 4.0);

        // Low-vision mode: add an additional 30% on top.
        let final_px = if self.low_vision_mode {
            scaled * 1.3
        } else {
            scaled
        };

        (final_px as u32).clamp(12, 400)
    }
}

/// Adaptive font size calculator that adjusts caption sizes for different viewing conditions.
#[derive(Debug, Clone)]
pub struct AdaptiveFontSizer {
    /// Base multiplier (user preference, 1.0 = standard).
    user_multiplier: f32,
    /// Minimum font size in pixels.
    min_size_px: u32,
    /// Maximum font size in pixels.
    max_size_px: u32,
}

impl AdaptiveFontSizer {
    /// Create a new adaptive font sizer.
    #[must_use]
    pub fn new(user_multiplier: f32) -> Self {
        Self {
            user_multiplier: user_multiplier.clamp(0.5, 4.0),
            min_size_px: 12,
            max_size_px: 200,
        }
    }

    /// Set size bounds.
    #[must_use]
    pub const fn with_bounds(mut self, min: u32, max: u32) -> Self {
        self.min_size_px = min;
        self.max_size_px = max;
        self
    }

    /// Calculate the optimal font size for the given viewing context.
    #[must_use]
    pub fn calculate(&self, context: &ViewingContext) -> u32 {
        let size = context.optimal_font_size_px(self.user_multiplier);
        size.clamp(self.min_size_px, self.max_size_px)
    }

    /// Apply the calculated font size to a [`CaptionStyle`].
    #[must_use]
    pub fn apply_to_style(&self, style: &CaptionStyle, context: &ViewingContext) -> CaptionStyle {
        let mut updated = style.clone();
        updated.font_size = self.calculate(context);
        updated
    }

    /// Get the user multiplier.
    #[must_use]
    pub fn user_multiplier(&self) -> f32 {
        self.user_multiplier
    }
}

impl Default for AdaptiveFontSizer {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Edge style for caption text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeStyle {
    /// No edge.
    None,
    /// Raised edge (3D effect).
    Raised,
    /// Depressed edge (3D effect).
    Depressed,
    /// Uniform drop shadow.
    DropShadow,
    /// Outline around text.
    Outline,
}

/// Preset caption styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptionStylePreset {
    /// Standard white text on black background.
    Standard,
    /// High contrast for better readability.
    HighContrast,
    /// Yellow text on black (common for DVDs).
    YellowOnBlack,
    /// Black text on white (for light backgrounds).
    BlackOnWhite,
    /// Transparent background.
    Transparent,
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self {
            font_size: 42,
            font_family: "Arial".to_string(),
            text_color: (255, 255, 255, 255),
            background_color: (0, 0, 0, 200),
            edge_style: EdgeStyle::Outline,
            edge_color: (0, 0, 0, 255),
            opacity: 255,
        }
    }
}

impl CaptionStyle {
    /// Create from preset.
    #[must_use]
    pub fn from_preset(preset: CaptionStylePreset) -> Self {
        match preset {
            CaptionStylePreset::Standard => Self::default(),
            CaptionStylePreset::HighContrast => Self {
                text_color: (255, 255, 255, 255),
                background_color: (0, 0, 0, 255),
                edge_style: EdgeStyle::Outline,
                edge_color: (0, 0, 0, 255),
                ..Default::default()
            },
            CaptionStylePreset::YellowOnBlack => Self {
                text_color: (255, 255, 0, 255),
                background_color: (0, 0, 0, 200),
                edge_style: EdgeStyle::None,
                ..Default::default()
            },
            CaptionStylePreset::BlackOnWhite => Self {
                text_color: (0, 0, 0, 255),
                background_color: (255, 255, 255, 200),
                edge_style: EdgeStyle::None,
                ..Default::default()
            },
            CaptionStylePreset::Transparent => Self {
                background_color: (0, 0, 0, 0),
                edge_style: EdgeStyle::DropShadow,
                ..Default::default()
            },
        }
    }

    /// Set font size.
    #[must_use]
    pub const fn with_font_size(mut self, size: u32) -> Self {
        self.font_size = size;
        self
    }

    /// Set text color.
    #[must_use]
    pub const fn with_text_color(mut self, r: u8, g: u8, b: u8, a: u8) -> Self {
        self.text_color = (r, g, b, a);
        self
    }

    /// Set background color.
    #[must_use]
    pub const fn with_background_color(mut self, r: u8, g: u8, b: u8, a: u8) -> Self {
        self.background_color = (r, g, b, a);
        self
    }

    /// Set edge style.
    #[must_use]
    pub const fn with_edge_style(mut self, style: EdgeStyle) -> Self {
        self.edge_style = style;
        self
    }

    /// Convert to subtitle style.
    #[must_use]
    pub fn to_subtitle_style(&self) -> SubtitleStyle {
        SubtitleStyle::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_style() {
        let style = CaptionStyle::default();
        assert_eq!(style.font_size, 42);
        assert_eq!(style.text_color, (255, 255, 255, 255));
    }

    #[test]
    fn test_preset_styles() {
        let standard = CaptionStyle::from_preset(CaptionStylePreset::Standard);
        let high_contrast = CaptionStyle::from_preset(CaptionStylePreset::HighContrast);

        assert_eq!(high_contrast.background_color.3, 255);
        assert_eq!(standard.text_color, (255, 255, 255, 255));
    }

    #[test]
    fn test_style_builder() {
        let style = CaptionStyle::default()
            .with_font_size(48)
            .with_text_color(255, 0, 0, 255)
            .with_edge_style(EdgeStyle::DropShadow);

        assert_eq!(style.font_size, 48);
        assert_eq!(style.text_color, (255, 0, 0, 255));
        assert_eq!(style.edge_style, EdgeStyle::DropShadow);
    }

    // ============================================================
    // Adaptive font sizing tests
    // ============================================================

    #[test]
    fn test_viewing_context_desktop() {
        let ctx = ViewingContext::desktop_1080p();
        assert_eq!(ctx.display_width_px, 1920);
        let ppd = ctx.pixels_per_degree();
        assert!(ppd > 10.0, "PPD should be positive: {ppd}");
    }

    #[test]
    fn test_viewing_context_tv() {
        let ctx = ViewingContext::tv_4k();
        let size = ctx.optimal_font_size_px(1.0);
        assert!(size >= 12);
        assert!(size <= 400);
    }

    #[test]
    fn test_viewing_context_mobile() {
        let ctx = ViewingContext::mobile_fhd();
        let size = ctx.optimal_font_size_px(1.0);
        assert!(size >= 12);
    }

    #[test]
    fn test_low_vision_mode_increases_size() {
        let ctx_normal = ViewingContext::desktop_1080p();
        let ctx_lv = ViewingContext::desktop_1080p().with_low_vision(true);

        let normal_size = ctx_normal.optimal_font_size_px(1.0);
        let lv_size = ctx_lv.optimal_font_size_px(1.0);
        assert!(
            lv_size >= normal_size,
            "Low vision size {lv_size} should be >= normal {normal_size}"
        );
    }

    #[test]
    fn test_multiplier_scales_size() {
        let ctx = ViewingContext::desktop_1080p();
        let small = ctx.optimal_font_size_px(0.5);
        let normal = ctx.optimal_font_size_px(1.0);
        let large = ctx.optimal_font_size_px(2.0);

        assert!(small <= normal, "Small {small} should <= normal {normal}");
        assert!(normal <= large, "Normal {normal} should <= large {large}");
    }

    #[test]
    fn test_adaptive_font_sizer_default() {
        let sizer = AdaptiveFontSizer::default();
        assert!((sizer.user_multiplier() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adaptive_font_sizer_calculate() {
        let sizer = AdaptiveFontSizer::new(1.0);
        let ctx = ViewingContext::desktop_1080p();
        let size = sizer.calculate(&ctx);
        assert!(size >= 12);
        assert!(size <= 200);
    }

    #[test]
    fn test_adaptive_font_sizer_apply_to_style() {
        let sizer = AdaptiveFontSizer::new(2.0);
        let ctx = ViewingContext::desktop_1080p();
        let base_style = CaptionStyle::default();
        let adapted = sizer.apply_to_style(&base_style, &ctx);
        // Adapted size should differ from the static default (42)
        let expected = sizer.calculate(&ctx);
        assert_eq!(adapted.font_size, expected);
        // Color should be unchanged
        assert_eq!(adapted.text_color, base_style.text_color);
    }

    #[test]
    fn test_adaptive_font_sizer_bounds() {
        let sizer = AdaptiveFontSizer::new(1.0).with_bounds(20, 80);
        let ctx = ViewingContext::desktop_1080p();
        let size = sizer.calculate(&ctx);
        assert!(size >= 20 && size <= 80);
    }

    #[test]
    fn test_adaptive_font_sizer_large_multiplier() {
        let sizer = AdaptiveFontSizer::new(4.0).with_bounds(12, 200);
        let ctx = ViewingContext::desktop_1080p();
        let size = sizer.calculate(&ctx);
        assert!(size <= 200);
    }

    /// WCAG color contrast tests for all caption style presets.
    #[test]
    fn test_wcag_contrast_all_presets() {
        use crate::visual::contrast::ContrastEnhancer;

        let presets = [
            (CaptionStylePreset::Standard, "Standard"),
            (CaptionStylePreset::HighContrast, "HighContrast"),
            (CaptionStylePreset::YellowOnBlack, "YellowOnBlack"),
            (CaptionStylePreset::BlackOnWhite, "BlackOnWhite"),
        ];

        for (preset, name) in &presets {
            let style = CaptionStyle::from_preset(*preset);
            let text_rgb = (style.text_color.0, style.text_color.1, style.text_color.2);
            let bg_rgb = (
                style.background_color.0,
                style.background_color.1,
                style.background_color.2,
            );
            let ratio = ContrastEnhancer::contrast_ratio(text_rgb, bg_rgb);
            assert!(
                ratio >= 4.5,
                "Preset {name} has contrast ratio {ratio:.2}:1 which is below WCAG AA (4.5:1)"
            );
        }
    }
}

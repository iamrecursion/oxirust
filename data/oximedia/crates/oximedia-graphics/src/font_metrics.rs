//! Font metrics and text measurement.
//!
//! Provides types for describing font weight, metrics (ascent, descent,
//! line gap, advance widths), and a lightweight text measurer that can
//! estimate layout dimensions without a full rasterisation pass.

#![allow(dead_code)]

/// CSS-style font weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FontWeight {
    /// Thin (100).
    Thin,
    /// Extra-Light (200).
    ExtraLight,
    /// Light (300).
    Light,
    /// Regular (400).
    Regular,
    /// Medium (500).
    Medium,
    /// Semi-Bold (600).
    SemiBold,
    /// Bold (700).
    Bold,
    /// Extra-Bold (800).
    ExtraBold,
    /// Black (900).
    Black,
}

impl FontWeight {
    /// Numeric value (100 -- 900).
    #[must_use]
    pub fn value(&self) -> u16 {
        match self {
            Self::Thin => 100,
            Self::ExtraLight => 200,
            Self::Light => 300,
            Self::Regular => 400,
            Self::Medium => 500,
            Self::SemiBold => 600,
            Self::Bold => 700,
            Self::ExtraBold => 800,
            Self::Black => 900,
        }
    }

    /// Construct from a numeric value, snapping to the nearest named weight.
    #[must_use]
    pub fn from_value(v: u16) -> Self {
        match v {
            0..=149 => Self::Thin,
            150..=249 => Self::ExtraLight,
            250..=349 => Self::Light,
            350..=449 => Self::Regular,
            450..=549 => Self::Medium,
            550..=649 => Self::SemiBold,
            650..=749 => Self::Bold,
            750..=849 => Self::ExtraBold,
            _ => Self::Black,
        }
    }

    /// CSS keyword.
    #[must_use]
    pub fn keyword(&self) -> &'static str {
        match self {
            Self::Thin => "thin",
            Self::ExtraLight => "extra-light",
            Self::Light => "light",
            Self::Regular => "normal",
            Self::Medium => "medium",
            Self::SemiBold => "semi-bold",
            Self::Bold => "bold",
            Self::ExtraBold => "extra-bold",
            Self::Black => "black",
        }
    }

    /// Returns `true` for weights >= `SemiBold`.
    #[must_use]
    pub fn is_bold(&self) -> bool {
        self.value() >= 600
    }
}

/// Core metrics for a single typeface at a given size.
#[derive(Debug, Clone, Copy)]
pub struct FontMetrics {
    /// Font size in points.
    pub size_pt: f32,
    /// Ascent in pixels (from baseline to top of tallest glyph).
    pub ascent: f32,
    /// Descent in pixels (baseline to bottom of lowest glyph, usually negative).
    pub descent: f32,
    /// Line gap in pixels (extra space between lines).
    pub line_gap: f32,
    /// Average character width in pixels.
    pub avg_char_width: f32,
    /// Maximum character width in pixels.
    pub max_char_width: f32,
    /// Cap-height in pixels.
    pub cap_height: f32,
    /// x-height in pixels.
    pub x_height: f32,
}

impl FontMetrics {
    /// Total line height: ascent + |descent| + `line_gap`.
    #[must_use]
    pub fn line_height(&self) -> f32 {
        self.ascent + self.descent.abs() + self.line_gap
    }

    /// Baseline-to-baseline distance for consecutive lines.
    #[must_use]
    pub fn baseline_distance(&self) -> f32 {
        self.line_height()
    }

    /// Scale all metrics proportionally to a new size in points.
    #[must_use]
    pub fn scaled(&self, new_size_pt: f32) -> Self {
        if self.size_pt <= 0.0 {
            return *self;
        }
        let factor = new_size_pt / self.size_pt;
        Self {
            size_pt: new_size_pt,
            ascent: self.ascent * factor,
            descent: self.descent * factor,
            line_gap: self.line_gap * factor,
            avg_char_width: self.avg_char_width * factor,
            max_char_width: self.max_char_width * factor,
            cap_height: self.cap_height * factor,
            x_height: self.x_height * factor,
        }
    }
}

impl Default for FontMetrics {
    /// Sensible defaults based on a 16pt sans-serif font.
    fn default() -> Self {
        Self {
            size_pt: 16.0,
            ascent: 14.0,
            descent: -4.0,
            line_gap: 2.0,
            avg_char_width: 8.0,
            max_char_width: 14.0,
            cap_height: 11.0,
            x_height: 8.0,
        }
    }
}

/// Result of a text measurement.
#[derive(Debug, Clone, Copy)]
pub struct TextExtent {
    /// Total width in pixels.
    pub width: f32,
    /// Total height in pixels (single line = `line_height`).
    pub height: f32,
    /// Number of lines.
    pub lines: usize,
}

/// Lightweight text measurer that estimates layout bounds from [`FontMetrics`].
#[derive(Debug, Clone)]
pub struct TextMeasurer {
    metrics: FontMetrics,
}

impl TextMeasurer {
    /// Create a measurer for the given font metrics.
    #[must_use]
    pub fn new(metrics: FontMetrics) -> Self {
        Self { metrics }
    }

    /// Measure the width of `text` in pixels.
    ///
    /// Simple model: each character contributes `avg_char_width`. Real
    /// implementations would use per-glyph advance widths and kerning.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn measure_width(&self, text: &str) -> f32 {
        text.chars().count() as f32 * self.metrics.avg_char_width
    }

    /// Measure the full extent of a single-line string.
    #[must_use]
    pub fn measure_line(&self, text: &str) -> TextExtent {
        TextExtent {
            width: self.measure_width(text),
            height: self.metrics.line_height(),
            lines: 1,
        }
    }

    /// Measure multi-line text (lines separated by `\n`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn measure_multiline(&self, text: &str) -> TextExtent {
        let lines: Vec<&str> = text.split('\n').collect();
        let max_width = lines
            .iter()
            .map(|l| self.measure_width(l))
            .fold(0.0_f32, f32::max);
        let height = lines.len() as f32 * self.metrics.line_height();
        TextExtent {
            width: max_width,
            height,
            lines: lines.len(),
        }
    }

    /// Word-wrap text to a maximum pixel width and measure the result.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn measure_wrapped(&self, text: &str, max_width: f32) -> TextExtent {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return TextExtent {
                width: 0.0,
                height: self.metrics.line_height(),
                lines: 1,
            };
        }

        let space_width = self.metrics.avg_char_width;
        let mut lines = 1usize;
        let mut current_width = 0.0_f32;
        let mut widest = 0.0_f32;

        for word in &words {
            let w = self.measure_width(word);
            let needed = if current_width > 0.0 {
                current_width + space_width + w
            } else {
                w
            };
            if needed > max_width && current_width > 0.0 {
                widest = widest.max(current_width);
                lines += 1;
                current_width = w;
            } else {
                current_width = needed;
            }
        }
        widest = widest.max(current_width);

        TextExtent {
            width: widest,
            height: lines as f32 * self.metrics.line_height(),
            lines,
        }
    }

    /// Return the inner font metrics.
    #[must_use]
    pub fn metrics(&self) -> &FontMetrics {
        &self.metrics
    }
}

// ---------------------------------------------------------------------------
// Sub-pixel font rendering support
// ---------------------------------------------------------------------------

/// Sub-pixel rendering mode for improved glyph sharpness at small sizes.
///
/// Sub-pixel rendering exploits the physical layout of RGB sub-pixels on LCD
/// panels to triple the effective horizontal resolution.  The result is sharper
/// text at small sizes at the cost of slight colour fringing on non-white
/// backgrounds — a well-understood trade-off in broadcast CG systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubPixelMode {
    /// No sub-pixel rendering (full-pixel greyscale anti-aliasing).
    None,
    /// RGB sub-pixel order (most common LCD layout, left-to-right R-G-B).
    Rgb,
    /// BGR sub-pixel order (some displays and all-in-ones).
    Bgr,
    /// Vertical RGB (rotated panel).
    Vrgb,
    /// Vertical BGR (rotated panel).
    Vbgr,
}

impl Default for SubPixelMode {
    fn default() -> Self {
        Self::Rgb
    }
}

impl SubPixelMode {
    /// Returns `true` if sub-pixel rendering is enabled.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Returns `true` if horizontal sub-pixel layout (RGB or BGR).
    pub fn is_horizontal(&self) -> bool {
        matches!(self, Self::Rgb | Self::Bgr)
    }

    /// Effective horizontal resolution multiplier.
    ///
    /// Returns `3.0` for horizontal modes (one pixel becomes three sub-pixels),
    /// `1.0` otherwise.
    pub fn horizontal_resolution_multiplier(&self) -> f32 {
        if self.is_horizontal() {
            3.0
        } else {
            1.0
        }
    }
}

/// A sub-pixel coverage sample for one physical pixel.
///
/// Each component holds the coverage fraction (0.0–1.0) for the corresponding
/// colour sub-pixel.  For greyscale (no sub-pixel rendering) all components
/// are equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubPixelCoverage {
    /// Red sub-pixel coverage.
    pub r: f32,
    /// Green sub-pixel coverage.
    pub g: f32,
    /// Blue sub-pixel coverage.
    pub b: f32,
}

impl SubPixelCoverage {
    /// Create a new coverage sample.
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Uniform (greyscale) coverage sample.
    pub fn uniform(v: f32) -> Self {
        let v = v.clamp(0.0, 1.0);
        Self { r: v, g: v, b: v }
    }

    /// Returns `true` if all components are zero (fully transparent).
    pub fn is_empty(&self) -> bool {
        self.r == 0.0 && self.g == 0.0 && self.b == 0.0
    }

    /// Average coverage (used for alpha estimation).
    pub fn average(&self) -> f32 {
        (self.r + self.g + self.b) / 3.0
    }

    /// Blend this coverage against a background RGBA pixel using straight
    /// alpha compositing.  Returns the composited RGBA bytes `[r, g, b, a]`.
    ///
    /// The `foreground` is the text colour in `[r, g, b, a]` where each
    /// component is in `[0, 255]`.
    pub fn composite_over(&self, foreground: [u8; 4], background: [u8; 4]) -> [u8; 4] {
        let fg_a = foreground[3] as f32 / 255.0;
        let blend_channel = |cov: f32, fg: u8, bg: u8| -> u8 {
            let a = cov * fg_a;
            let out = fg as f32 * a + bg as f32 * (1.0 - a);
            out.clamp(0.0, 255.0) as u8
        };
        let r = blend_channel(self.r, foreground[0], background[0]);
        let g = blend_channel(self.g, foreground[1], background[1]);
        let b = blend_channel(self.b, foreground[2], background[2]);
        let out_a = (self.average() * fg_a
            + background[3] as f32 / 255.0 * (1.0 - self.average() * fg_a))
            .clamp(0.0, 1.0);
        [r, g, b, (out_a * 255.0) as u8]
    }
}

/// Sub-pixel glyph metrics — extends [`FontMetrics`] with sub-pixel advance
/// widths for more accurate fractional positioning.
#[derive(Debug, Clone, Copy)]
pub struct SubPixelMetrics {
    /// Underlying integer-aligned font metrics.
    pub base: FontMetrics,
    /// Sub-pixel rendering mode.
    pub mode: SubPixelMode,
    /// Fractional pixel advance per character (allows non-integer character spacing).
    pub sub_pixel_advance: f32,
}

impl SubPixelMetrics {
    /// Create sub-pixel metrics from base font metrics and a rendering mode.
    pub fn new(base: FontMetrics, mode: SubPixelMode) -> Self {
        // For horizontal sub-pixel modes, the effective advance is slightly
        // smaller than the integer-rounded advance because we can place glyphs
        // at 1/3-pixel boundaries.
        let sub_pixel_advance = if mode.is_horizontal() {
            // Round the advance to the nearest 1/3-pixel boundary.
            let thirds = (base.avg_char_width * 3.0).round();
            thirds / 3.0
        } else {
            base.avg_char_width
        };
        Self {
            base,
            mode,
            sub_pixel_advance,
        }
    }

    /// Measure the sub-pixel-accurate width of `text` in pixels.
    #[allow(clippy::cast_precision_loss)]
    pub fn measure_width_subpixel(&self, text: &str) -> f32 {
        text.chars().count() as f32 * self.sub_pixel_advance
    }

    /// Fractional sub-pixel position of the `n`-th character (0-indexed).
    pub fn char_position(&self, n: usize) -> f32 {
        n as f32 * self.sub_pixel_advance
    }

    /// Generate sub-pixel coverage samples for a horizontal scan of a single
    /// glyph with a given normalised raster coverage value.
    ///
    /// This implements a simplified box-filter sub-pixel model: the coverage
    /// value is distributed across three sub-pixels with a `[0.25, 0.5, 0.25]`
    /// kernel (approximation of the ideal sinc filter).
    pub fn coverage_for(
        &self,
        pixel_coverage: f32,
        mode_override: Option<SubPixelMode>,
    ) -> SubPixelCoverage {
        let mode = mode_override.unwrap_or(self.mode);
        if !mode.is_enabled() {
            return SubPixelCoverage::uniform(pixel_coverage);
        }

        let c = pixel_coverage.clamp(0.0, 1.0);
        // Box filter kernel: `[0.25, 0.5, 0.25]`.
        let lo = (c * 0.75).clamp(0.0, 1.0); // 0.25 + overlap → approximate
        let hi = c;

        match mode {
            SubPixelMode::Rgb => SubPixelCoverage::new(lo, hi, lo),
            SubPixelMode::Bgr => SubPixelCoverage::new(lo, hi, lo),
            SubPixelMode::Vrgb | SubPixelMode::Vbgr => SubPixelCoverage::uniform(c),
            SubPixelMode::None => SubPixelCoverage::uniform(c),
        }
    }
}

// -- unit tests --

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_font_weight_values() {
        assert_eq!(FontWeight::Thin.value(), 100);
        assert_eq!(FontWeight::Regular.value(), 400);
        assert_eq!(FontWeight::Black.value(), 900);
    }

    #[test]
    fn test_font_weight_from_value() {
        assert_eq!(FontWeight::from_value(400), FontWeight::Regular);
        assert_eq!(FontWeight::from_value(0), FontWeight::Thin);
        assert_eq!(FontWeight::from_value(999), FontWeight::Black);
    }

    #[test]
    fn test_font_weight_is_bold() {
        assert!(!FontWeight::Regular.is_bold());
        assert!(FontWeight::Bold.is_bold());
        assert!(FontWeight::SemiBold.is_bold());
    }

    #[test]
    fn test_font_weight_ordering() {
        assert!(FontWeight::Thin < FontWeight::Regular);
        assert!(FontWeight::Regular < FontWeight::Bold);
    }

    #[test]
    fn test_font_weight_keywords() {
        assert_eq!(FontWeight::Regular.keyword(), "normal");
        assert_eq!(FontWeight::Bold.keyword(), "bold");
    }

    #[test]
    fn test_font_metrics_line_height() {
        let m = FontMetrics::default();
        // ascent=14, descent=-4, gap=2 -> 14+4+2=20
        assert!((m.line_height() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_font_metrics_scaled() {
        let m = FontMetrics::default(); // 16pt
        let s = m.scaled(32.0);
        assert!((s.ascent - 28.0).abs() < 0.01); // 14 * 2
        assert!((s.avg_char_width - 16.0).abs() < 0.01); // 8 * 2
    }

    #[test]
    fn test_measure_width_empty() {
        let measurer = TextMeasurer::new(FontMetrics::default());
        assert_eq!(measurer.measure_width(""), 0.0);
    }

    #[test]
    fn test_measure_width_single_char() {
        let m = FontMetrics::default();
        let measurer = TextMeasurer::new(m);
        assert_eq!(measurer.measure_width("A"), m.avg_char_width);
    }

    #[test]
    fn test_measure_line() {
        let measurer = TextMeasurer::new(FontMetrics::default());
        let ext = measurer.measure_line("Hello");
        assert_eq!(ext.lines, 1);
        assert!(ext.width > 0.0);
        assert!(ext.height > 0.0);
    }

    #[test]
    fn test_measure_multiline() {
        let measurer = TextMeasurer::new(FontMetrics::default());
        let ext = measurer.measure_multiline("Line1\nLine2\nLine3");
        assert_eq!(ext.lines, 3);
        assert!((ext.height - 3.0 * measurer.metrics().line_height()).abs() < 0.01);
    }

    #[test]
    fn test_measure_wrapped() {
        let measurer = TextMeasurer::new(FontMetrics::default());
        // "Hello World" at 8px/char = 11 chars * 8 = 88px
        // Max width 50 -> should wrap
        let ext = measurer.measure_wrapped("Hello World", 50.0);
        assert!(ext.lines >= 2);
    }

    #[test]
    fn test_measure_wrapped_single_word() {
        let measurer = TextMeasurer::new(FontMetrics::default());
        let ext = measurer.measure_wrapped("Supercalifragilistic", 40.0);
        // Single word wider than max_width -> stays on one line
        assert_eq!(ext.lines, 1);
    }

    #[test]
    fn test_measure_wrapped_empty() {
        let measurer = TextMeasurer::new(FontMetrics::default());
        let ext = measurer.measure_wrapped("", 100.0);
        assert_eq!(ext.lines, 1);
        assert_eq!(ext.width, 0.0);
    }

    #[test]
    fn test_baseline_distance() {
        let m = FontMetrics::default();
        assert_eq!(m.baseline_distance(), m.line_height());
    }

    // --- SubPixelMode tests ---

    #[test]
    fn test_sub_pixel_mode_default() {
        assert_eq!(SubPixelMode::default(), SubPixelMode::Rgb);
    }

    #[test]
    fn test_sub_pixel_mode_none_not_enabled() {
        assert!(!SubPixelMode::None.is_enabled());
    }

    #[test]
    fn test_sub_pixel_mode_rgb_enabled() {
        assert!(SubPixelMode::Rgb.is_enabled());
    }

    #[test]
    fn test_sub_pixel_mode_rgb_is_horizontal() {
        assert!(SubPixelMode::Rgb.is_horizontal());
        assert!(SubPixelMode::Bgr.is_horizontal());
        assert!(!SubPixelMode::Vrgb.is_horizontal());
        assert!(!SubPixelMode::None.is_horizontal());
    }

    #[test]
    fn test_sub_pixel_mode_horizontal_multiplier() {
        assert!((SubPixelMode::Rgb.horizontal_resolution_multiplier() - 3.0).abs() < f32::EPSILON);
        assert!((SubPixelMode::None.horizontal_resolution_multiplier() - 1.0).abs() < f32::EPSILON);
    }

    // --- SubPixelCoverage tests ---

    #[test]
    fn test_sub_pixel_coverage_uniform_average() {
        let c = SubPixelCoverage::uniform(0.6);
        assert!((c.average() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_sub_pixel_coverage_is_empty() {
        let c = SubPixelCoverage::uniform(0.0);
        assert!(c.is_empty());
        let c2 = SubPixelCoverage::uniform(0.5);
        assert!(!c2.is_empty());
    }

    #[test]
    fn test_sub_pixel_coverage_composite_over_full_coverage() {
        // Full coverage → pixel should be foreground color.
        let cov = SubPixelCoverage::uniform(1.0);
        let fg = [255u8, 0, 0, 255];
        let bg = [0u8, 0, 0, 255];
        let result = cov.composite_over(fg, bg);
        assert_eq!(result[0], 255); // Red channel
        assert_eq!(result[1], 0); // Green
    }

    #[test]
    fn test_sub_pixel_coverage_composite_over_zero_coverage() {
        let cov = SubPixelCoverage::uniform(0.0);
        let fg = [255u8, 0, 0, 255];
        let bg = [100u8, 100, 100, 255];
        let result = cov.composite_over(fg, bg);
        assert_eq!(result[0], 100); // background preserved
    }

    // --- SubPixelMetrics tests ---

    #[test]
    fn test_sub_pixel_metrics_new_rgb() {
        let m = FontMetrics::default(); // avg_char_width = 8.0
        let spm = SubPixelMetrics::new(m, SubPixelMode::Rgb);
        // sub_pixel_advance should be rounded to nearest 1/3 pixel.
        // 8.0 * 3 = 24.0 → 24/3 = 8.0
        assert!((spm.sub_pixel_advance - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_sub_pixel_metrics_measure_width() {
        let m = FontMetrics::default();
        let spm = SubPixelMetrics::new(m, SubPixelMode::Rgb);
        let w = spm.measure_width_subpixel("Hello");
        assert!(w > 0.0);
        // Should match 5 * sub_pixel_advance.
        assert!((w - 5.0 * spm.sub_pixel_advance).abs() < 0.001);
    }

    #[test]
    fn test_sub_pixel_metrics_char_position() {
        let m = FontMetrics::default();
        let spm = SubPixelMetrics::new(m, SubPixelMode::None);
        assert!((spm.char_position(0)).abs() < f32::EPSILON);
        let p3 = spm.char_position(3);
        assert!(p3 > 0.0);
    }

    #[test]
    fn test_sub_pixel_metrics_coverage_none_is_uniform() {
        let m = FontMetrics::default();
        let spm = SubPixelMetrics::new(m, SubPixelMode::None);
        let cov = spm.coverage_for(0.7, None);
        // None mode → uniform.
        assert!((cov.r - cov.g).abs() < f32::EPSILON);
        assert!((cov.g - cov.b).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sub_pixel_metrics_coverage_rgb_green_highest() {
        let m = FontMetrics::default();
        let spm = SubPixelMetrics::new(m, SubPixelMode::Rgb);
        let cov = spm.coverage_for(1.0, None);
        // Green (hi) should be >= r and b (lo).
        assert!(cov.g >= cov.r);
        assert!(cov.g >= cov.b);
    }
}

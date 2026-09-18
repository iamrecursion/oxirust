#![allow(dead_code)]
//! High-contrast mode rendering for accessible media overlays.
//!
//! Provides tools to adjust overlay and UI element colours to meet WCAG
//! contrast-ratio requirements, including automatic theme generation
//! and per-element contrast checking.

/// An RGBA colour value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rgba {
    /// Red channel (0-255).
    pub r: u8,
    /// Green channel (0-255).
    pub g: u8,
    /// Blue channel (0-255).
    pub b: u8,
    /// Alpha channel (0-255).
    pub a: u8,
}

impl Rgba {
    /// Create an opaque colour.
    #[must_use]
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Create a colour with alpha.
    #[must_use]
    pub fn with_alpha(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Pure white.
    #[must_use]
    pub fn white() -> Self {
        Self::new(255, 255, 255)
    }

    /// Pure black.
    #[must_use]
    pub fn black() -> Self {
        Self::new(0, 0, 0)
    }

    /// Yellow, commonly used in high-contrast themes.
    #[must_use]
    pub fn hc_yellow() -> Self {
        Self::new(255, 255, 0)
    }

    /// Cyan, commonly used in high-contrast themes.
    #[must_use]
    pub fn hc_cyan() -> Self {
        Self::new(0, 255, 255)
    }

    /// Compute the relative luminance per WCAG 2.1 definition.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn relative_luminance(&self) -> f64 {
        fn linearize(channel: u8) -> f64 {
            let c = f64::from(channel) / 255.0;
            if c <= 0.039_28 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        let r = linearize(self.r);
        let g = linearize(self.g);
        let b = linearize(self.b);
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    /// Invert the colour (ignoring alpha).
    #[must_use]
    pub fn invert(&self) -> Self {
        Self {
            r: 255 - self.r,
            g: 255 - self.g,
            b: 255 - self.b,
            a: self.a,
        }
    }
}

impl Default for Rgba {
    fn default() -> Self {
        Self::black()
    }
}

/// Compute the WCAG contrast ratio between two colours.
///
/// Returns a value between 1.0 and 21.0.
#[must_use]
pub fn contrast_ratio(a: &Rgba, b: &Rgba) -> f64 {
    let la = a.relative_luminance();
    let lb = b.relative_luminance();
    let lighter = la.max(lb);
    let darker = la.min(lb);
    (lighter + 0.05) / (darker + 0.05)
}

/// WCAG conformance level for contrast checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WcagLevel {
    /// Level AA for normal text (4.5:1).
    AaNormal,
    /// Level AA for large text (3:1).
    AaLarge,
    /// Level AAA for normal text (7:1).
    AaaNormal,
    /// Level AAA for large text (4.5:1).
    AaaLarge,
}

impl WcagLevel {
    /// Return the minimum contrast ratio required.
    #[must_use]
    pub fn min_ratio(&self) -> f64 {
        match self {
            Self::AaNormal => 4.5,
            Self::AaLarge => 3.0,
            Self::AaaNormal => 7.0,
            Self::AaaLarge => 4.5,
        }
    }

    /// Check whether the given contrast ratio meets this level.
    #[must_use]
    pub fn passes(&self, ratio: f64) -> bool {
        ratio >= self.min_ratio()
    }
}

/// Result of a contrast check on a single element.
#[derive(Debug, Clone)]
pub struct ContrastCheckResult {
    /// Element identifier.
    pub element_id: String,
    /// Foreground colour.
    pub foreground: Rgba,
    /// Background colour.
    pub background: Rgba,
    /// Computed contrast ratio.
    pub ratio: f64,
    /// Whether the check passed.
    pub passed: bool,
    /// The level being checked.
    pub level: WcagLevel,
}

/// Check contrast for a foreground/background pair at the given WCAG level.
#[must_use]
pub fn check_contrast(
    element_id: impl Into<String>,
    foreground: Rgba,
    background: Rgba,
    level: WcagLevel,
) -> ContrastCheckResult {
    let ratio = contrast_ratio(&foreground, &background);
    ContrastCheckResult {
        element_id: element_id.into(),
        foreground,
        background,
        ratio,
        passed: level.passes(ratio),
        level,
    }
}

/// A high-contrast colour theme.
#[derive(Debug, Clone)]
pub struct HighContrastTheme {
    /// Theme name.
    pub name: String,
    /// Background colour.
    pub background: Rgba,
    /// Primary text colour.
    pub text: Rgba,
    /// Accent / link colour.
    pub accent: Rgba,
    /// Border / outline colour.
    pub border: Rgba,
    /// Disabled element colour.
    pub disabled: Rgba,
}

impl HighContrastTheme {
    /// Create a standard dark high-contrast theme.
    #[must_use]
    pub fn dark() -> Self {
        Self {
            name: String::from("Dark High Contrast"),
            background: Rgba::black(),
            text: Rgba::white(),
            accent: Rgba::hc_yellow(),
            border: Rgba::white(),
            disabled: Rgba::new(128, 128, 128),
        }
    }

    /// Create a standard light high-contrast theme.
    #[must_use]
    pub fn light() -> Self {
        Self {
            name: String::from("Light High Contrast"),
            background: Rgba::white(),
            text: Rgba::black(),
            accent: Rgba::new(0, 0, 180),
            border: Rgba::black(),
            disabled: Rgba::new(128, 128, 128),
        }
    }

    /// Check whether the text-on-background meets the given WCAG level.
    #[must_use]
    pub fn text_passes(&self, level: WcagLevel) -> bool {
        let ratio = contrast_ratio(&self.text, &self.background);
        level.passes(ratio)
    }

    /// Check whether the accent-on-background meets the given WCAG level.
    #[must_use]
    pub fn accent_passes(&self, level: WcagLevel) -> bool {
        let ratio = contrast_ratio(&self.accent, &self.background);
        level.passes(ratio)
    }

    /// Suggest the best foreground colour (black or white) for the given background.
    #[must_use]
    pub fn best_foreground_for(background: &Rgba) -> Rgba {
        let white_ratio = contrast_ratio(&Rgba::white(), background);
        let black_ratio = contrast_ratio(&Rgba::black(), background);
        if white_ratio >= black_ratio {
            Rgba::white()
        } else {
            Rgba::black()
        }
    }
}

/// Batch contrast checker that validates multiple elements at once.
#[derive(Debug, Default)]
pub struct ContrastAuditor {
    /// Accumulated results.
    results: Vec<ContrastCheckResult>,
}

impl ContrastAuditor {
    /// Create a new auditor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add a check for one element.
    pub fn check(
        &mut self,
        element_id: impl Into<String>,
        foreground: Rgba,
        background: Rgba,
        level: WcagLevel,
    ) {
        self.results
            .push(check_contrast(element_id, foreground, background, level));
    }

    /// Return the number of checks performed.
    #[must_use]
    pub fn total(&self) -> usize {
        self.results.len()
    }

    /// Return the number of checks that passed.
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Return the number of checks that failed.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Return all failed results.
    #[must_use]
    pub fn failures(&self) -> Vec<&ContrastCheckResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Check whether all audited elements passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Compute the overall pass rate (0.0..=1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pass_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 1.0;
        }
        self.passed_count() as f64 / self.results.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba_creation() {
        let c = Rgba::new(10, 20, 30);
        assert_eq!(c.r, 10);
        assert_eq!(c.g, 20);
        assert_eq!(c.b, 30);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn test_rgba_invert() {
        let c = Rgba::new(0, 100, 255);
        let inv = c.invert();
        assert_eq!(inv.r, 255);
        assert_eq!(inv.g, 155);
        assert_eq!(inv.b, 0);
    }

    #[test]
    fn test_relative_luminance_black() {
        let lum = Rgba::black().relative_luminance();
        assert!(lum.abs() < 0.001);
    }

    #[test]
    fn test_relative_luminance_white() {
        let lum = Rgba::white().relative_luminance();
        assert!((lum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_contrast_ratio_black_white() {
        let ratio = contrast_ratio(&Rgba::black(), &Rgba::white());
        assert!(ratio > 20.0);
        assert!(ratio <= 21.0);
    }

    #[test]
    fn test_contrast_ratio_same_color() {
        let ratio = contrast_ratio(&Rgba::new(100, 100, 100), &Rgba::new(100, 100, 100));
        assert!((ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_wcag_level_min_ratio() {
        assert!((WcagLevel::AaNormal.min_ratio() - 4.5).abs() < f64::EPSILON);
        assert!((WcagLevel::AaLarge.min_ratio() - 3.0).abs() < f64::EPSILON);
        assert!((WcagLevel::AaaNormal.min_ratio() - 7.0).abs() < f64::EPSILON);
        assert!((WcagLevel::AaaLarge.min_ratio() - 4.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_check_contrast_pass() {
        let result = check_contrast(
            "heading",
            Rgba::black(),
            Rgba::white(),
            WcagLevel::AaaNormal,
        );
        assert!(result.passed);
        assert!(result.ratio > 7.0);
    }

    #[test]
    fn test_check_contrast_fail() {
        let result = check_contrast(
            "subtle",
            Rgba::new(180, 180, 180),
            Rgba::white(),
            WcagLevel::AaNormal,
        );
        assert!(!result.passed);
    }

    #[test]
    fn test_dark_theme_text_passes() {
        let theme = HighContrastTheme::dark();
        assert!(theme.text_passes(WcagLevel::AaaNormal));
    }

    #[test]
    fn test_light_theme_text_passes() {
        let theme = HighContrastTheme::light();
        assert!(theme.text_passes(WcagLevel::AaaNormal));
    }

    #[test]
    fn test_best_foreground_for_dark_bg() {
        let fg = HighContrastTheme::best_foreground_for(&Rgba::new(10, 10, 10));
        assert_eq!(fg, Rgba::white());
    }

    #[test]
    fn test_best_foreground_for_light_bg() {
        let fg = HighContrastTheme::best_foreground_for(&Rgba::new(240, 240, 240));
        assert_eq!(fg, Rgba::black());
    }

    #[test]
    fn test_contrast_auditor_all_pass() {
        let mut auditor = ContrastAuditor::new();
        auditor.check("a", Rgba::black(), Rgba::white(), WcagLevel::AaNormal);
        auditor.check("b", Rgba::white(), Rgba::black(), WcagLevel::AaNormal);
        assert!(auditor.all_passed());
        assert_eq!(auditor.total(), 2);
        assert!((auditor.pass_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_contrast_auditor_with_failure() {
        let mut auditor = ContrastAuditor::new();
        auditor.check("good", Rgba::black(), Rgba::white(), WcagLevel::AaNormal);
        auditor.check(
            "bad",
            Rgba::new(200, 200, 200),
            Rgba::white(),
            WcagLevel::AaNormal,
        );
        assert!(!auditor.all_passed());
        assert_eq!(auditor.failed_count(), 1);
        assert_eq!(auditor.passed_count(), 1);
        let fails = auditor.failures();
        assert_eq!(fails[0].element_id, "bad");
    }

    #[test]
    fn test_contrast_auditor_empty() {
        let auditor = ContrastAuditor::new();
        assert!(auditor.all_passed());
        assert!((auditor.pass_rate() - 1.0).abs() < f64::EPSILON);
    }
}

//! WCAG (Web Content Accessibility Guidelines) compliance checking.
//!
//! Provides tools for checking media content against WCAG 2.1 criteria
//! relevant to audio/video content, including captions, audio descriptions,
//! sign language, contrast ratios, and text sizing.

use serde::{Deserialize, Serialize};

/// WCAG conformance level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum WcagLevel {
    /// Level A – minimum level of conformance.
    A,
    /// Level AA – mid-range conformance (most regulations require this).
    AA,
    /// Level AAA – highest level of conformance.
    AAA,
}

impl std::fmt::Display for WcagLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::AA => write!(f, "AA"),
            Self::AAA => write!(f, "AAA"),
        }
    }
}

/// Individual WCAG success criterion relevant to media content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum WcagCriterion {
    /// 1.2.2 – Captions (Prerecorded) – Level A.
    Captions1_2_2,
    /// 1.2.3 – Audio Description or Media Alternative (Prerecorded) – Level A.
    AudioDesc1_2_3,
    /// 1.2.6 – Sign Language (Prerecorded) – Level AAA.
    SignLang1_2_6,
    /// 1.2.7 – Extended Audio Description (Prerecorded) – Level AAA.
    ExtendedAudioDesc1_2_7,
    /// 1.4.3 – Contrast (Minimum) – Level AA.
    ContrastRatio1_4_3,
    /// 1.4.4 – Resize Text – Level AA.
    ResizableText1_4_4,
    /// 1.4.12 – Text Spacing – Level AA.
    TextSpacing1_4_12,
}

impl WcagCriterion {
    /// The WCAG level required for this criterion.
    #[must_use]
    pub const fn required_level(&self) -> WcagLevel {
        match self {
            Self::Captions1_2_2 | Self::AudioDesc1_2_3 => WcagLevel::A,
            Self::ContrastRatio1_4_3 | Self::ResizableText1_4_4 | Self::TextSpacing1_4_12 => {
                WcagLevel::AA
            }
            Self::SignLang1_2_6 | Self::ExtendedAudioDesc1_2_7 => WcagLevel::AAA,
        }
    }

    /// Human-readable name for the criterion.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Captions1_2_2 => "1.2.2 Captions (Prerecorded)",
            Self::AudioDesc1_2_3 => "1.2.3 Audio Description or Media Alternative (Prerecorded)",
            Self::SignLang1_2_6 => "1.2.6 Sign Language (Prerecorded)",
            Self::ExtendedAudioDesc1_2_7 => "1.2.7 Extended Audio Description (Prerecorded)",
            Self::ContrastRatio1_4_3 => "1.4.3 Contrast (Minimum)",
            Self::ResizableText1_4_4 => "1.4.4 Resize Text",
            Self::TextSpacing1_4_12 => "1.4.12 Text Spacing",
        }
    }
}

impl std::fmt::Display for WcagCriterion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Result of checking a single WCAG criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WcagCheckResult {
    /// The criterion that was checked.
    pub criterion: WcagCriterion,
    /// The WCAG level of this criterion.
    pub level: WcagLevel,
    /// Whether the criterion passes.
    pub passes: bool,
    /// Human-readable note about the result.
    pub note: String,
}

impl WcagCheckResult {
    /// Create a new check result.
    #[must_use]
    pub fn new(criterion: WcagCriterion, passes: bool, note: impl Into<String>) -> Self {
        let level = criterion.required_level();
        Self {
            criterion,
            level,
            passes,
            note: note.into(),
        }
    }
}

/// A WCAG compliance report for a piece of media content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WcagReport {
    /// URL or identifier of the content being checked.
    pub url: String,
    /// The WCAG level being checked against.
    pub level_checked: WcagLevel,
    /// Individual criterion results.
    pub results: Vec<WcagCheckResult>,
    /// Number of criteria that pass.
    pub pass_count: usize,
    /// Number of criteria that fail.
    pub fail_count: usize,
}

impl WcagReport {
    /// Create a new empty WCAG report.
    #[must_use]
    pub fn new(url: &str, level: WcagLevel) -> Self {
        Self {
            url: url.to_string(),
            level_checked: level,
            results: Vec::new(),
            pass_count: 0,
            fail_count: 0,
        }
    }

    /// Add a check result to the report.
    pub fn add_result(&mut self, r: WcagCheckResult) {
        // Only include criteria at or below the checked level
        if r.level <= self.level_checked {
            if r.passes {
                self.pass_count += 1;
            } else {
                self.fail_count += 1;
            }
            self.results.push(r);
        }
    }

    /// Returns `true` if all applicable criteria pass.
    #[must_use]
    pub fn overall_passes(&self) -> bool {
        self.fail_count == 0 && !self.results.is_empty()
    }

    /// Compliance percentage (0.0–100.0).
    #[must_use]
    pub fn compliance_pct(&self) -> f64 {
        let total = self.pass_count + self.fail_count;
        if total == 0 {
            return 0.0;
        }
        (self.pass_count as f64 / total as f64) * 100.0
    }

    /// Number of results applicable to the checked level.
    #[must_use]
    pub fn total_checked(&self) -> usize {
        self.pass_count + self.fail_count
    }
}

/// Check caption compliance (criterion 1.2.2).
///
/// Prerecorded content must have captions; live content may be excluded at level A.
#[must_use]
pub fn check_caption_compliance(has_captions: bool, is_live: bool) -> WcagCheckResult {
    if is_live {
        // 1.2.2 applies to prerecorded content; live has its own criterion (1.2.4)
        WcagCheckResult::new(
            WcagCriterion::Captions1_2_2,
            true,
            "Live content – criterion 1.2.2 applies to prerecorded content only.",
        )
    } else if has_captions {
        WcagCheckResult::new(
            WcagCriterion::Captions1_2_2,
            true,
            "Captions are present for prerecorded content.",
        )
    } else {
        WcagCheckResult::new(
            WcagCriterion::Captions1_2_2,
            false,
            "Prerecorded content does not have captions.",
        )
    }
}

/// Check contrast ratio compliance (criterion 1.4.3).
///
/// Normal text requires a contrast ratio of at least 4.5:1; large text requires 3:1.
#[must_use]
pub fn check_contrast_ratio(foreground: [u8; 3], background: [u8; 3]) -> WcagCheckResult {
    let l1 = relative_luminance(foreground);
    let l2 = relative_luminance(background);
    let ratio = contrast_ratio(l1, l2);

    // WCAG AA requires 4.5:1 for normal text
    let passes = ratio >= 4.5;
    let note = format!(
        "Contrast ratio is {ratio:.2}:1 (minimum 4.5:1 required for normal text at Level AA)."
    );

    WcagCheckResult::new(WcagCriterion::ContrastRatio1_4_3, passes, note)
}

/// Compute relative luminance of an sRGB colour using the WCAG formula.
///
/// Input components are in [0, 255]. Output is in [0.0, 1.0].
///
/// See: <https://www.w3.org/TR/WCAG21/#dfn-relative-luminance>
#[must_use]
pub fn relative_luminance(rgb: [u8; 3]) -> f64 {
    let linearize = |c: u8| -> f64 {
        let s = f64::from(c) / 255.0;
        if s <= 0.040_45 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };

    let r = linearize(rgb[0]);
    let g = linearize(rgb[1]);
    let b = linearize(rgb[2]);

    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Compute the WCAG contrast ratio between two relative luminance values.
///
/// Both `l1` and `l2` must be in [0.0, 1.0].
/// The result is always ≥ 1.0 (lighter / darker order is handled automatically).
#[must_use]
pub fn contrast_ratio(l1: f64, l2: f64) -> f64 {
    let lighter = l1.max(l2);
    let darker = l1.min(l2);
    (lighter + 0.05) / (darker + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── WcagLevel ──────────────────────────────────────────────────────────────

    #[test]
    fn test_wcag_level_ordering() {
        assert!(WcagLevel::A < WcagLevel::AA);
        assert!(WcagLevel::AA < WcagLevel::AAA);
    }

    #[test]
    fn test_wcag_level_display() {
        assert_eq!(WcagLevel::A.to_string(), "A");
        assert_eq!(WcagLevel::AA.to_string(), "AA");
        assert_eq!(WcagLevel::AAA.to_string(), "AAA");
    }

    // ── WcagCriterion ─────────────────────────────────────────────────────────

    #[test]
    fn test_criterion_levels() {
        assert_eq!(WcagCriterion::Captions1_2_2.required_level(), WcagLevel::A);
        assert_eq!(
            WcagCriterion::ContrastRatio1_4_3.required_level(),
            WcagLevel::AA
        );
        assert_eq!(
            WcagCriterion::SignLang1_2_6.required_level(),
            WcagLevel::AAA
        );
    }

    #[test]
    fn test_criterion_names_non_empty() {
        let criteria = [
            WcagCriterion::Captions1_2_2,
            WcagCriterion::AudioDesc1_2_3,
            WcagCriterion::SignLang1_2_6,
            WcagCriterion::ExtendedAudioDesc1_2_7,
            WcagCriterion::ContrastRatio1_4_3,
            WcagCriterion::ResizableText1_4_4,
            WcagCriterion::TextSpacing1_4_12,
        ];
        for c in &criteria {
            assert!(!c.name().is_empty());
        }
    }

    // ── WcagReport ────────────────────────────────────────────────────────────

    #[test]
    fn test_report_new_is_empty() {
        let report = WcagReport::new("https://example.com/video", WcagLevel::AA);
        assert_eq!(report.pass_count, 0);
        assert_eq!(report.fail_count, 0);
        assert!(!report.overall_passes());
    }

    #[test]
    fn test_report_add_result_counts() {
        let mut report = WcagReport::new("https://example.com/video", WcagLevel::AA);
        report.add_result(WcagCheckResult::new(
            WcagCriterion::Captions1_2_2,
            true,
            "OK",
        ));
        report.add_result(WcagCheckResult::new(
            WcagCriterion::ContrastRatio1_4_3,
            false,
            "Failed",
        ));
        assert_eq!(report.pass_count, 1);
        assert_eq!(report.fail_count, 1);
    }

    #[test]
    fn test_report_aaa_criteria_excluded_at_aa_level() {
        let mut report = WcagReport::new("https://example.com/video", WcagLevel::AA);
        // AAA criterion should be filtered out
        report.add_result(WcagCheckResult::new(
            WcagCriterion::SignLang1_2_6,
            false,
            "No sign language",
        ));
        assert_eq!(report.total_checked(), 0);
    }

    #[test]
    fn test_report_overall_passes_all_pass() {
        let mut report = WcagReport::new("https://example.com/video", WcagLevel::AA);
        report.add_result(WcagCheckResult::new(
            WcagCriterion::Captions1_2_2,
            true,
            "OK",
        ));
        assert!(report.overall_passes());
    }

    #[test]
    fn test_report_compliance_pct() {
        let mut report = WcagReport::new("https://example.com/video", WcagLevel::AA);
        report.add_result(WcagCheckResult::new(
            WcagCriterion::Captions1_2_2,
            true,
            "OK",
        ));
        report.add_result(WcagCheckResult::new(
            WcagCriterion::ContrastRatio1_4_3,
            true,
            "OK",
        ));
        report.add_result(WcagCheckResult::new(
            WcagCriterion::AudioDesc1_2_3,
            false,
            "Missing",
        ));
        let pct = report.compliance_pct();
        assert!((pct - 66.666_666_7).abs() < 0.001);
    }

    #[test]
    fn test_report_compliance_pct_empty() {
        let report = WcagReport::new("https://example.com/video", WcagLevel::AA);
        assert_eq!(report.compliance_pct(), 0.0);
    }

    // ── Caption compliance ─────────────────────────────────────────────────────

    #[test]
    fn test_caption_compliance_pass() {
        let result = check_caption_compliance(true, false);
        assert!(result.passes);
    }

    #[test]
    fn test_caption_compliance_fail() {
        let result = check_caption_compliance(false, false);
        assert!(!result.passes);
    }

    #[test]
    fn test_caption_compliance_live_always_passes() {
        let result = check_caption_compliance(false, true);
        assert!(result.passes);
    }

    // ── Luminance & contrast ───────────────────────────────────────────────────

    #[test]
    fn test_relative_luminance_black() {
        let l = relative_luminance([0, 0, 0]);
        assert!(l.abs() < 1e-10);
    }

    #[test]
    fn test_relative_luminance_white() {
        let l = relative_luminance([255, 255, 255]);
        assert!((l - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_contrast_ratio_black_white() {
        let l_black = relative_luminance([0, 0, 0]);
        let l_white = relative_luminance([255, 255, 255]);
        let ratio = contrast_ratio(l_black, l_white);
        assert!((ratio - 21.0).abs() < 0.01);
    }

    #[test]
    fn test_contrast_ratio_symmetric() {
        let l1 = relative_luminance([100, 100, 100]);
        let l2 = relative_luminance([200, 200, 200]);
        assert!((contrast_ratio(l1, l2) - contrast_ratio(l2, l1)).abs() < 1e-10);
    }

    #[test]
    fn test_check_contrast_ratio_pass() {
        // Black text on white – 21:1 – should pass
        let result = check_contrast_ratio([0, 0, 0], [255, 255, 255]);
        assert!(result.passes);
    }

    #[test]
    fn test_check_contrast_ratio_fail() {
        // Similar greys – should fail
        let result = check_contrast_ratio([150, 150, 150], [160, 160, 160]);
        assert!(!result.passes);
    }
}

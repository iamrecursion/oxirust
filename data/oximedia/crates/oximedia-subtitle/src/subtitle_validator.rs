//! Subtitle validation rules and reporting.
//!
//! Provides configurable rule-based validation of subtitle entries, producing
//! structured violation reports suitable for QC workflows.

#![allow(dead_code)]

/// A subtitle validation rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationRule {
    /// Subtitle duration must be at least N milliseconds.
    MinDuration(u32),
    /// Subtitle duration must be no more than N milliseconds.
    MaxDuration(u32),
    /// Gap between consecutive subtitles must be at least N milliseconds.
    MinGap(u32),
    /// Text must not exceed N characters per line.
    MaxCharsPerLine(usize),
    /// Text must not have more than N lines.
    MaxLines(usize),
    /// Start time must not be negative.
    NonNegativeStart,
    /// End time must be greater than start time.
    EndAfterStart,
}

impl ValidationRule {
    /// Return a human-readable name for this rule.
    pub fn rule_name(&self) -> &'static str {
        match self {
            ValidationRule::MinDuration(_) => "min_duration",
            ValidationRule::MaxDuration(_) => "max_duration",
            ValidationRule::MinGap(_) => "min_gap",
            ValidationRule::MaxCharsPerLine(_) => "max_chars_per_line",
            ValidationRule::MaxLines(_) => "max_lines",
            ValidationRule::NonNegativeStart => "non_negative_start",
            ValidationRule::EndAfterStart => "end_after_start",
        }
    }
}

/// A subtitle validation violation.
#[derive(Debug, Clone)]
pub struct SubtitleViolation {
    /// 0-based index of the offending subtitle.
    pub entry_index: usize,
    /// The rule that was violated.
    pub rule: ValidationRule,
    /// Human-readable description.
    pub message: String,
}

impl SubtitleViolation {
    /// Create a new violation.
    pub fn new(entry_index: usize, rule: ValidationRule, message: impl Into<String>) -> Self {
        Self {
            entry_index,
            rule,
            message: message.into(),
        }
    }

    /// Returns `true` if this violation is a timing-related error.
    pub fn is_timing_error(&self) -> bool {
        matches!(
            self.rule,
            ValidationRule::MinDuration(_)
                | ValidationRule::MaxDuration(_)
                | ValidationRule::MinGap(_)
                | ValidationRule::NonNegativeStart
                | ValidationRule::EndAfterStart
        )
    }
}

/// A subtitle entry used as input to the validator.
#[derive(Debug, Clone)]
pub struct ValidatorEntry {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Text content (may be multi-line).
    pub text: String,
}

impl ValidatorEntry {
    /// Create a new `ValidatorEntry`.
    pub fn new(start_ms: i64, end_ms: i64, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Return the duration in milliseconds.
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Return the maximum line length in the text.
    pub fn max_line_length(&self) -> usize {
        self.text.lines().map(|l| l.len()).max().unwrap_or(0)
    }

    /// Return the number of lines in the text.
    pub fn line_count(&self) -> usize {
        if self.text.is_empty() {
            0
        } else {
            self.text.lines().count()
        }
    }
}

/// Validates a sequence of subtitle entries against a set of rules.
#[derive(Debug)]
pub struct SubtitleValidator {
    rules: Vec<ValidationRule>,
}

impl SubtitleValidator {
    /// Create a new validator with the given rules.
    pub fn new(rules: Vec<ValidationRule>) -> Self {
        Self { rules }
    }

    /// Create a validator with sensible broadcast defaults.
    pub fn broadcast_defaults() -> Self {
        Self::new(vec![
            ValidationRule::NonNegativeStart,
            ValidationRule::EndAfterStart,
            ValidationRule::MinDuration(500),
            ValidationRule::MaxDuration(8000),
            ValidationRule::MaxCharsPerLine(42),
            ValidationRule::MaxLines(2),
            ValidationRule::MinGap(40),
        ])
    }

    /// Validate a slice of entries and return all violations.
    pub fn validate(&self, entries: &[ValidatorEntry]) -> SubtitleReport {
        let mut violations = Vec::new();

        for (idx, entry) in entries.iter().enumerate() {
            for rule in &self.rules {
                if let Some(v) = self.check_rule(idx, entry, rule) {
                    violations.push(v);
                }
            }
            // Gap check requires the previous entry
            if idx > 0 {
                for rule in &self.rules {
                    if let ValidationRule::MinGap(min_ms) = *rule {
                        let prev = &entries[idx - 1];
                        let gap = entry.start_ms - prev.end_ms;
                        if gap < i64::from(min_ms) {
                            violations.push(SubtitleViolation::new(
                                idx,
                                *rule,
                                format!(
                                    "Gap {}ms between entries {} and {} is less than minimum {}ms",
                                    gap,
                                    idx - 1,
                                    idx,
                                    min_ms
                                ),
                            ));
                        }
                    }
                }
            }
        }

        SubtitleReport { violations }
    }

    /// Check a single rule against a single entry.  Returns a violation or `None`.
    fn check_rule(
        &self,
        idx: usize,
        entry: &ValidatorEntry,
        rule: &ValidationRule,
    ) -> Option<SubtitleViolation> {
        match *rule {
            ValidationRule::NonNegativeStart => {
                if entry.start_ms < 0 {
                    Some(SubtitleViolation::new(
                        idx,
                        *rule,
                        format!("Entry {} has negative start time {}ms", idx, entry.start_ms),
                    ))
                } else {
                    None
                }
            }
            ValidationRule::EndAfterStart => {
                if entry.end_ms <= entry.start_ms {
                    Some(SubtitleViolation::new(
                        idx,
                        *rule,
                        format!(
                            "Entry {} end {}ms is not after start {}ms",
                            idx, entry.end_ms, entry.start_ms
                        ),
                    ))
                } else {
                    None
                }
            }
            ValidationRule::MinDuration(min_ms) => {
                let dur = entry.duration_ms();
                if dur < i64::from(min_ms) {
                    Some(SubtitleViolation::new(
                        idx,
                        *rule,
                        format!(
                            "Entry {} duration {}ms is less than minimum {}ms",
                            idx, dur, min_ms
                        ),
                    ))
                } else {
                    None
                }
            }
            ValidationRule::MaxDuration(max_ms) => {
                let dur = entry.duration_ms();
                if dur > i64::from(max_ms) {
                    Some(SubtitleViolation::new(
                        idx,
                        *rule,
                        format!(
                            "Entry {} duration {}ms exceeds maximum {}ms",
                            idx, dur, max_ms
                        ),
                    ))
                } else {
                    None
                }
            }
            ValidationRule::MaxCharsPerLine(max_chars) => {
                let longest = entry.max_line_length();
                if longest > max_chars {
                    Some(SubtitleViolation::new(
                        idx,
                        *rule,
                        format!(
                            "Entry {} has a line with {} characters (max {})",
                            idx, longest, max_chars
                        ),
                    ))
                } else {
                    None
                }
            }
            ValidationRule::MaxLines(max_lines) => {
                let count = entry.line_count();
                if count > max_lines {
                    Some(SubtitleViolation::new(
                        idx,
                        *rule,
                        format!("Entry {} has {} lines (max {})", idx, count, max_lines),
                    ))
                } else {
                    None
                }
            }
            // MinGap is handled in the outer loop
            ValidationRule::MinGap(_) => None,
        }
    }
}

/// A complete validation report for a subtitle file.
#[derive(Debug)]
pub struct SubtitleReport {
    /// All violations found during validation.
    pub violations: Vec<SubtitleViolation>,
}

impl SubtitleReport {
    /// Return the total number of violations.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Return the number of timing-related errors.
    pub fn error_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.is_timing_error())
            .count()
    }

    /// Return `true` if there are no violations.
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }

    /// Collect violations grouped by rule name.
    pub fn by_rule(&self) -> std::collections::HashMap<&'static str, Vec<&SubtitleViolation>> {
        let mut map: std::collections::HashMap<&'static str, Vec<&SubtitleViolation>> =
            std::collections::HashMap::new();
        for v in &self.violations {
            map.entry(v.rule.rule_name()).or_default().push(v);
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(start: i64, end: i64, text: &str) -> ValidatorEntry {
        ValidatorEntry::new(start, end, text)
    }

    #[test]
    fn test_validation_rule_name() {
        assert_eq!(ValidationRule::MinDuration(500).rule_name(), "min_duration");
        assert_eq!(
            ValidationRule::MaxDuration(8000).rule_name(),
            "max_duration"
        );
        assert_eq!(ValidationRule::MinGap(40).rule_name(), "min_gap");
        assert_eq!(
            ValidationRule::MaxCharsPerLine(42).rule_name(),
            "max_chars_per_line"
        );
        assert_eq!(ValidationRule::MaxLines(2).rule_name(), "max_lines");
        assert_eq!(
            ValidationRule::NonNegativeStart.rule_name(),
            "non_negative_start"
        );
        assert_eq!(ValidationRule::EndAfterStart.rule_name(), "end_after_start");
    }

    #[test]
    fn test_subtitle_violation_is_timing_error_true() {
        let v = SubtitleViolation::new(0, ValidationRule::EndAfterStart, "err");
        assert!(v.is_timing_error());
    }

    #[test]
    fn test_subtitle_violation_is_timing_error_false() {
        let v = SubtitleViolation::new(0, ValidationRule::MaxCharsPerLine(42), "err");
        assert!(!v.is_timing_error());
    }

    #[test]
    fn test_validator_entry_duration_ms() {
        let e = entry(1000, 4500, "Hello");
        assert_eq!(e.duration_ms(), 3500);
    }

    #[test]
    fn test_validator_entry_max_line_length() {
        let e = entry(0, 1000, "Short\nA very long line indeed");
        assert_eq!(e.max_line_length(), 23);
    }

    #[test]
    fn test_validator_entry_line_count() {
        let e = entry(0, 1000, "Line one\nLine two");
        assert_eq!(e.line_count(), 2);
    }

    #[test]
    fn test_validator_entry_empty_text_line_count() {
        let e = entry(0, 1000, "");
        assert_eq!(e.line_count(), 0);
    }

    #[test]
    fn test_validate_clean_entries() {
        let validator = SubtitleValidator::broadcast_defaults();
        let entries = vec![
            entry(0, 2000, "Hello world"),
            entry(3000, 5000, "Second line"),
        ];
        let report = validator.validate(&entries);
        assert!(
            report.is_clean(),
            "Expected no violations: {:?}",
            report.violations
        );
    }

    #[test]
    fn test_validate_end_before_start() {
        let validator = SubtitleValidator::new(vec![ValidationRule::EndAfterStart]);
        let entries = vec![entry(5000, 3000, "Bad timing")];
        let report = validator.validate(&entries);
        assert_eq!(report.violation_count(), 1);
        assert!(report.violations[0].is_timing_error());
    }

    #[test]
    fn test_validate_negative_start() {
        let validator = SubtitleValidator::new(vec![ValidationRule::NonNegativeStart]);
        let entries = vec![entry(-100, 1000, "Negative")];
        let report = validator.validate(&entries);
        assert_eq!(report.violation_count(), 1);
    }

    #[test]
    fn test_validate_min_duration() {
        let validator = SubtitleValidator::new(vec![
            ValidationRule::EndAfterStart,
            ValidationRule::MinDuration(1000),
        ]);
        let entries = vec![entry(0, 200, "Too short")];
        let report = validator.validate(&entries);
        assert_eq!(report.error_count(), 1);
    }

    #[test]
    fn test_validate_max_duration() {
        let validator = SubtitleValidator::new(vec![ValidationRule::MaxDuration(3000)]);
        let entries = vec![entry(0, 10000, "Too long")];
        let report = validator.validate(&entries);
        assert_eq!(report.violation_count(), 1);
    }

    #[test]
    fn test_validate_max_chars_per_line() {
        let validator = SubtitleValidator::new(vec![ValidationRule::MaxCharsPerLine(10)]);
        let entries = vec![entry(0, 2000, "This line is too long for the rule")];
        let report = validator.validate(&entries);
        assert_eq!(report.violation_count(), 1);
        assert!(!report.violations[0].is_timing_error());
    }

    #[test]
    fn test_validate_max_lines() {
        let validator = SubtitleValidator::new(vec![ValidationRule::MaxLines(2)]);
        let entries = vec![entry(0, 2000, "Line 1\nLine 2\nLine 3")];
        let report = validator.validate(&entries);
        assert_eq!(report.violation_count(), 1);
    }

    #[test]
    fn test_validate_min_gap() {
        let validator = SubtitleValidator::new(vec![ValidationRule::MinGap(500)]);
        let entries = vec![
            entry(0, 2000, "A"),
            entry(2100, 4000, "B"), // only 100ms gap
        ];
        let report = validator.validate(&entries);
        assert_eq!(report.violation_count(), 1);
    }

    #[test]
    fn test_report_error_count() {
        let validator = SubtitleValidator::new(vec![
            ValidationRule::EndAfterStart,
            ValidationRule::MaxCharsPerLine(5),
        ]);
        let entries = vec![entry(5000, 3000, "Too many chars here")];
        let report = validator.validate(&entries);
        // EndAfterStart is timing, MaxCharsPerLine is not
        assert_eq!(report.error_count(), 1);
        assert_eq!(report.violation_count(), 2);
    }

    #[test]
    fn test_report_by_rule() {
        let validator = SubtitleValidator::new(vec![
            ValidationRule::MaxCharsPerLine(5),
            ValidationRule::MaxLines(1),
        ]);
        let entries = vec![entry(0, 2000, "Line 1 is very long\nLine 2")];
        let report = validator.validate(&entries);
        let by_rule = report.by_rule();
        assert!(by_rule.contains_key("max_chars_per_line"));
        assert!(by_rule.contains_key("max_lines"));
    }
}

// ============================================================================
// WCAG 2.1 AA contrast ratio checker
// ============================================================================

/// An sRGB colour value for contrast calculation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrgbColor {
    /// Red channel (0–255).
    pub r: u8,
    /// Green channel (0–255).
    pub g: u8,
    /// Blue channel (0–255).
    pub b: u8,
}

impl SrgbColor {
    /// Create a new colour.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Pure white.
    #[must_use]
    pub const fn white() -> Self {
        Self::new(255, 255, 255)
    }

    /// Pure black.
    #[must_use]
    pub const fn black() -> Self {
        Self::new(0, 0, 0)
    }

    /// Convert an sRGB channel byte (0–255) to a linear light value.
    fn linearise(channel: u8) -> f64 {
        let c = f64::from(channel) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Relative luminance as defined by WCAG 2.1 (IEC 61966-2-1 sRGB).
    ///
    /// Returns a value in `[0.0, 1.0]` where 0 = black and 1 = white.
    #[must_use]
    pub fn relative_luminance(&self) -> f64 {
        let r = Self::linearise(self.r);
        let g = Self::linearise(self.g);
        let b = Self::linearise(self.b);
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }
}

/// Compute the WCAG 2.1 contrast ratio between two colours.
///
/// The ratio is always ≥ 1.0; higher means better contrast.
/// WCAG AA requires ≥ 4.5:1 for normal text, ≥ 3.0:1 for large text (≥ 18pt / ≥ 14pt bold).
///
/// # Example
///
/// ```
/// use oximedia_subtitle::subtitle_validator::{SrgbColor, wcag_contrast_ratio};
/// let ratio = wcag_contrast_ratio(SrgbColor::white(), SrgbColor::black());
/// assert!((ratio - 21.0).abs() < 0.01);
/// ```
#[must_use]
pub fn wcag_contrast_ratio(foreground: SrgbColor, background: SrgbColor) -> f64 {
    let l1 = foreground.relative_luminance();
    let l2 = background.relative_luminance();
    let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
    (lighter + 0.05) / (darker + 0.05)
}

/// WCAG 2.1 conformance level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WcagLevel {
    /// Fails WCAG contrast requirements.
    Fail,
    /// Passes WCAG AA for large text only (≥ 3.0:1).
    AaLargeText,
    /// Passes WCAG AA for all text (≥ 4.5:1).
    Aa,
    /// Passes WCAG AAA for all text (≥ 7.0:1).
    Aaa,
}

/// Evaluate the WCAG 2.1 conformance level for a contrast ratio.
#[must_use]
pub fn wcag_level(ratio: f64) -> WcagLevel {
    if ratio >= 7.0 {
        WcagLevel::Aaa
    } else if ratio >= 4.5 {
        WcagLevel::Aa
    } else if ratio >= 3.0 {
        WcagLevel::AaLargeText
    } else {
        WcagLevel::Fail
    }
}

/// Check that subtitle text meets WCAG 2.1 AA contrast against a given background.
///
/// Returns `true` if `ratio >= 4.5` (AA for normal text).
///
/// # Example
///
/// ```
/// use oximedia_subtitle::subtitle_validator::{SrgbColor, passes_wcag_aa};
/// assert!(passes_wcag_aa(SrgbColor::white(), SrgbColor::black()));
/// assert!(!passes_wcag_aa(SrgbColor::new(200, 200, 200), SrgbColor::new(255, 255, 255)));
/// ```
#[must_use]
pub fn passes_wcag_aa(foreground: SrgbColor, background: SrgbColor) -> bool {
    wcag_contrast_ratio(foreground, background) >= 4.5
}

/// Check WCAG 2.1 AA for large text (18pt or 14pt bold) — requires ≥ 3.0:1.
#[must_use]
pub fn passes_wcag_aa_large(foreground: SrgbColor, background: SrgbColor) -> bool {
    wcag_contrast_ratio(foreground, background) >= 3.0
}

/// Check WCAG 2.1 AAA compliance — requires ≥ 7.0:1.
#[must_use]
pub fn passes_wcag_aaa(foreground: SrgbColor, background: SrgbColor) -> bool {
    wcag_contrast_ratio(foreground, background) >= 7.0
}

#[cfg(test)]
mod wcag_tests {
    use super::*;

    #[test]
    fn test_white_black_contrast_is_21() {
        let ratio = wcag_contrast_ratio(SrgbColor::white(), SrgbColor::black());
        assert!((ratio - 21.0).abs() < 0.05, "ratio={ratio}");
    }

    #[test]
    fn test_same_color_contrast_is_1() {
        let ratio = wcag_contrast_ratio(SrgbColor::white(), SrgbColor::white());
        assert!((ratio - 1.0).abs() < 0.01, "ratio={ratio}");
    }

    #[test]
    fn test_relative_luminance_black() {
        assert!((SrgbColor::black().relative_luminance() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_relative_luminance_white() {
        assert!((SrgbColor::white().relative_luminance() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_relative_luminance_mid_grey() {
        // sRGB 0x80 = 128 linearises to ~0.2158
        let grey = SrgbColor::new(128, 128, 128);
        let lum = grey.relative_luminance();
        assert!(lum > 0.2 && lum < 0.25, "lum={lum}");
    }

    #[test]
    fn test_passes_wcag_aa_white_on_black() {
        assert!(passes_wcag_aa(SrgbColor::white(), SrgbColor::black()));
    }

    #[test]
    fn test_fails_wcag_aa_similar_colors() {
        // Light grey on white — low contrast
        let fg = SrgbColor::new(170, 170, 170);
        let bg = SrgbColor::white();
        assert!(!passes_wcag_aa(fg, bg));
    }

    #[test]
    fn test_passes_wcag_aa_yellow_on_black() {
        // Yellow #FFD700 on black has high contrast
        let fg = SrgbColor::new(255, 215, 0);
        let bg = SrgbColor::black();
        assert!(passes_wcag_aa(fg, bg), "Yellow on black should pass AA");
    }

    #[test]
    fn test_passes_wcag_aa_large_white_on_dark_grey() {
        let fg = SrgbColor::white();
        let bg = SrgbColor::new(80, 80, 80);
        assert!(passes_wcag_aa_large(fg, bg));
    }

    #[test]
    fn test_fails_wcag_aa_large_similar() {
        let fg = SrgbColor::new(140, 140, 140);
        let bg = SrgbColor::new(180, 180, 180);
        assert!(!passes_wcag_aa_large(fg, bg));
    }

    #[test]
    fn test_wcag_level_aaa() {
        assert_eq!(wcag_level(7.5), WcagLevel::Aaa);
        assert_eq!(wcag_level(21.0), WcagLevel::Aaa);
    }

    #[test]
    fn test_wcag_level_aa() {
        assert_eq!(wcag_level(5.0), WcagLevel::Aa);
        assert_eq!(wcag_level(4.5), WcagLevel::Aa);
    }

    #[test]
    fn test_wcag_level_aa_large() {
        assert_eq!(wcag_level(3.5), WcagLevel::AaLargeText);
        assert_eq!(wcag_level(3.0), WcagLevel::AaLargeText);
    }

    #[test]
    fn test_wcag_level_fail() {
        assert_eq!(wcag_level(2.9), WcagLevel::Fail);
        assert_eq!(wcag_level(1.0), WcagLevel::Fail);
    }

    #[test]
    fn test_passes_wcag_aaa_white_black() {
        assert!(passes_wcag_aaa(SrgbColor::white(), SrgbColor::black()));
    }

    #[test]
    fn test_fails_wcag_aaa_moderate_contrast() {
        let fg = SrgbColor::new(100, 100, 100);
        let bg = SrgbColor::white();
        assert!(!passes_wcag_aaa(fg, bg));
    }

    #[test]
    fn test_contrast_ratio_symmetric() {
        let a = SrgbColor::new(255, 100, 0);
        let b = SrgbColor::new(10, 10, 200);
        let r1 = wcag_contrast_ratio(a, b);
        let r2 = wcag_contrast_ratio(b, a);
        assert!((r1 - r2).abs() < 1e-9, "ratio must be symmetric");
    }

    #[test]
    fn test_srgb_color_new() {
        let c = SrgbColor::new(10, 20, 30);
        assert_eq!(c.r, 10);
        assert_eq!(c.g, 20);
        assert_eq!(c.b, 30);
    }

    #[test]
    fn test_wcag_level_ordering() {
        assert!(WcagLevel::Aaa > WcagLevel::Aa);
        assert!(WcagLevel::Aa > WcagLevel::AaLargeText);
        assert!(WcagLevel::AaLargeText > WcagLevel::Fail);
    }
}

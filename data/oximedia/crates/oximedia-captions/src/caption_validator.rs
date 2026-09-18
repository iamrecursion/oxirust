//! Caption content and timing validation against broadcast and web standards.
//!
//! Checks captions for reading-speed compliance, overlapping timecodes,
//! minimum display duration, maximum line length, and empty-text entries.

#![allow(dead_code)]

/// A specific rule that the validator enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationRule {
    /// Caption text must not exceed a maximum character count per line.
    MaxCharsPerLine,
    /// Captions must be displayed for at least a minimum duration.
    MinDisplayDuration,
    /// Adjacent captions must not have overlapping time ranges.
    NoOverlap,
    /// Reading speed (words per minute) must not exceed a threshold.
    MaxReadingSpeed,
    /// Caption text must not be empty or whitespace-only.
    NonEmpty,
    /// Maximum number of rows visible simultaneously.
    MaxRows,
}

impl ValidationRule {
    /// Human-readable description of the rule.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::MaxCharsPerLine => "Line length exceeds the configured maximum",
            Self::MinDisplayDuration => "Caption displayed for less than the minimum duration",
            Self::NoOverlap => "Caption time range overlaps with an adjacent caption",
            Self::MaxReadingSpeed => "Reading speed exceeds the configured maximum WPM",
            Self::NonEmpty => "Caption text is empty or whitespace only",
            Self::MaxRows => "Too many caption rows visible simultaneously",
        }
    }
}

/// A single validation violation found in a caption.
#[derive(Debug, Clone)]
pub struct CaptionViolation {
    /// Index of the offending caption in the track.
    pub caption_index: usize,
    /// The rule that was violated.
    pub rule: ValidationRule,
    /// Human-readable explanation.
    pub message: String,
}

impl CaptionViolation {
    /// Create a new violation.
    #[must_use]
    pub fn new(caption_index: usize, rule: ValidationRule, message: impl Into<String>) -> Self {
        Self {
            caption_index,
            rule,
            message: message.into(),
        }
    }
}

/// A minimal caption entry used by the validator (avoids depending on the
/// full `Caption` type from `crate::types`).
#[derive(Debug, Clone)]
pub struct ValidatorCaption {
    /// Display start in milliseconds.
    pub start_ms: u64,
    /// Display end in milliseconds.
    pub end_ms: u64,
    /// Lines of text.
    pub lines: Vec<String>,
}

impl ValidatorCaption {
    /// Create a caption with a single line.
    #[must_use]
    pub fn single(start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            lines: vec![text.into()],
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Total text content across all lines.
    #[must_use]
    pub fn full_text(&self) -> String {
        self.lines.join(" ")
    }

    /// Word count across all lines.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.lines.iter().flat_map(|l| l.split_whitespace()).count()
    }
}

/// Configuration for the caption validator.
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Maximum characters allowed per line.
    pub max_chars_per_line: usize,
    /// Minimum display duration in milliseconds.
    pub min_duration_ms: u64,
    /// Maximum reading speed in words per minute.
    pub max_wpm: f64,
    /// Maximum simultaneous rows.
    pub max_rows: usize,
    /// Rules that are currently enabled.
    pub enabled_rules: Vec<ValidationRule>,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            max_chars_per_line: 42,
            min_duration_ms: 800,
            max_wpm: 200.0,
            max_rows: 3,
            enabled_rules: vec![
                ValidationRule::MaxCharsPerLine,
                ValidationRule::MinDisplayDuration,
                ValidationRule::NoOverlap,
                ValidationRule::MaxReadingSpeed,
                ValidationRule::NonEmpty,
                ValidationRule::MaxRows,
            ],
        }
    }
}

/// Validates a sequence of captions against configurable rules.
#[derive(Debug)]
pub struct CaptionValidator {
    config: ValidatorConfig,
}

impl CaptionValidator {
    /// Create a new validator with the given configuration.
    #[must_use]
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Access the current configuration.
    #[must_use]
    pub fn config(&self) -> &ValidatorConfig {
        &self.config
    }

    fn rule_enabled(&self, rule: ValidationRule) -> bool {
        self.config.enabled_rules.contains(&rule)
    }

    /// Validate a slice of captions. Returns all found violations.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn validate(&self, captions: &[ValidatorCaption]) -> Vec<CaptionViolation> {
        let mut violations = Vec::new();

        for (idx, cap) in captions.iter().enumerate() {
            // NonEmpty
            if self.rule_enabled(ValidationRule::NonEmpty) {
                let text = cap.full_text();
                if text.trim().is_empty() {
                    violations.push(CaptionViolation::new(
                        idx,
                        ValidationRule::NonEmpty,
                        "Caption has no text",
                    ));
                }
            }

            // MaxCharsPerLine
            if self.rule_enabled(ValidationRule::MaxCharsPerLine) {
                for line in &cap.lines {
                    if line.len() > self.config.max_chars_per_line {
                        violations.push(CaptionViolation::new(
                            idx,
                            ValidationRule::MaxCharsPerLine,
                            format!(
                                "Line has {} chars (max {})",
                                line.len(),
                                self.config.max_chars_per_line
                            ),
                        ));
                        break;
                    }
                }
            }

            // MinDisplayDuration
            if self.rule_enabled(ValidationRule::MinDisplayDuration)
                && cap.duration_ms() < self.config.min_duration_ms
            {
                violations.push(CaptionViolation::new(
                    idx,
                    ValidationRule::MinDisplayDuration,
                    format!(
                        "Duration {}ms < min {}ms",
                        cap.duration_ms(),
                        self.config.min_duration_ms
                    ),
                ));
            }

            // MaxReadingSpeed (WPM)
            if self.rule_enabled(ValidationRule::MaxReadingSpeed) && cap.duration_ms() > 0 {
                let minutes = cap.duration_ms() as f64 / 60_000.0;
                let wpm = cap.word_count() as f64 / minutes;
                if wpm > self.config.max_wpm {
                    violations.push(CaptionViolation::new(
                        idx,
                        ValidationRule::MaxReadingSpeed,
                        format!("Reading speed {wpm:.0} WPM > max {}", self.config.max_wpm),
                    ));
                }
            }

            // MaxRows
            if self.rule_enabled(ValidationRule::MaxRows) && cap.lines.len() > self.config.max_rows
            {
                violations.push(CaptionViolation::new(
                    idx,
                    ValidationRule::MaxRows,
                    format!("{} rows > max {}", cap.lines.len(), self.config.max_rows),
                ));
            }
        }

        // NoOverlap — compare adjacent pairs.
        if self.rule_enabled(ValidationRule::NoOverlap) {
            for i in 1..captions.len() {
                let prev = &captions[i - 1];
                let curr = &captions[i];
                if curr.start_ms < prev.end_ms {
                    violations.push(CaptionViolation::new(
                        i,
                        ValidationRule::NoOverlap,
                        format!(
                            "Starts at {}ms but previous ends at {}ms",
                            curr.start_ms, prev.end_ms
                        ),
                    ));
                }
            }
        }

        violations
    }

    /// Returns `true` when there are no violations.
    #[must_use]
    pub fn is_valid(&self, captions: &[ValidatorCaption]) -> bool {
        self.validate(captions).is_empty()
    }

    /// Count violations grouped by rule.
    #[must_use]
    pub fn violation_counts(&self, captions: &[ValidatorCaption]) -> Vec<(ValidationRule, usize)> {
        let violations = self.validate(captions);
        let rules = [
            ValidationRule::MaxCharsPerLine,
            ValidationRule::MinDisplayDuration,
            ValidationRule::NoOverlap,
            ValidationRule::MaxReadingSpeed,
            ValidationRule::NonEmpty,
            ValidationRule::MaxRows,
        ];
        rules
            .iter()
            .map(|&r| {
                let count = violations.iter().filter(|v| v.rule == r).count();
                (r, count)
            })
            .filter(|(_, c)| *c > 0)
            .collect()
    }
}

impl Default for CaptionValidator {
    fn default() -> Self {
        Self::new(ValidatorConfig::default())
    }
}

// ── Broadcast validator free functions ───────────────────────────────────────

/// A minimal caption representation used by the broadcast validator helpers.
/// Positions are stored as percentages (0.0–100.0) of the frame dimensions.
#[derive(Debug, Clone)]
pub struct BroadcastCaption {
    /// Lines of text in this caption.
    pub lines: Vec<String>,
    /// Horizontal position as a percentage of frame width (0.0 = left edge).
    pub x_pct: f32,
    /// Vertical position as a percentage of frame height (0.0 = top edge).
    pub y_pct: f32,
    /// Width as a percentage of frame width.
    pub w_pct: f32,
    /// Height as a percentage of frame height.
    pub h_pct: f32,
}

impl BroadcastCaption {
    /// Create a caption with the given text lines and a default bottom-centre
    /// position that fits within a 10 % safe-area margin on all sides.
    #[must_use]
    pub fn new(lines: Vec<String>) -> Self {
        Self {
            lines,
            x_pct: 10.0,
            y_pct: 80.0,
            w_pct: 80.0,
            h_pct: 15.0,
        }
    }

    /// Convenience constructor for a single-line caption.
    #[must_use]
    pub fn single(text: impl Into<String>) -> Self {
        Self::new(vec![text.into()])
    }
}

/// Returns `true` when the caption has at most `max` lines.
///
/// A value of `0` for `max` is treated as "no limit" and always returns `true`.
#[must_use]
pub fn check_max_lines(caption: &BroadcastCaption, max: usize) -> bool {
    if max == 0 {
        return true;
    }
    caption.lines.len() <= max
}

/// Returns `true` when every line in the caption is within `max` characters.
///
/// A value of `0` for `max` is treated as "no limit" and always returns `true`.
#[must_use]
pub fn check_max_chars_per_line(caption: &BroadcastCaption, max: usize) -> bool {
    if max == 0 {
        return true;
    }
    caption.lines.iter().all(|l| l.chars().count() <= max)
}

/// Returns `true` when the caption's bounding box falls entirely inside the
/// safe area defined by `safe_pct` margins on all four sides.
///
/// `safe_pct` is expressed as a percentage of the frame dimension (e.g. `10.0`
/// means a 10 % margin on each side, leaving an 80 % × 80 % safe area).
///
/// A `safe_pct` of `0.0` means no margin requirement (always returns `true`).
#[must_use]
pub fn check_safe_area_margins(caption: &BroadcastCaption, safe_pct: f32) -> bool {
    if safe_pct <= 0.0 {
        return true;
    }
    let right_edge = caption.x_pct + caption.w_pct;
    let bottom_edge = caption.y_pct + caption.h_pct;
    let max_edge = 100.0 - safe_pct;

    caption.x_pct >= safe_pct
        && caption.y_pct >= safe_pct
        && right_edge <= max_edge
        && bottom_edge <= max_edge
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_cap(start: u64, end: u64, text: &str) -> ValidatorCaption {
        ValidatorCaption::single(start, end, text)
    }

    #[test]
    fn test_valid_caption_no_violations() {
        let v = CaptionValidator::default();
        let caps = vec![good_cap(0, 2000, "Hello world")];
        assert!(v.is_valid(&caps));
    }

    #[test]
    fn test_empty_text_violation() {
        let v = CaptionValidator::default();
        let caps = vec![good_cap(0, 2000, "   ")];
        let violations = v.validate(&caps);
        assert!(violations
            .iter()
            .any(|v| v.rule == ValidationRule::NonEmpty));
    }

    #[test]
    fn test_line_too_long() {
        let v = CaptionValidator::default();
        let long_line = "a".repeat(50);
        let caps = vec![good_cap(0, 2000, &long_line)];
        let violations = v.validate(&caps);
        assert!(violations
            .iter()
            .any(|v| v.rule == ValidationRule::MaxCharsPerLine));
    }

    #[test]
    fn test_min_duration_violation() {
        let v = CaptionValidator::default();
        let caps = vec![good_cap(0, 400, "Short")]; // 400ms < 800ms
        let violations = v.validate(&caps);
        assert!(violations
            .iter()
            .any(|v| v.rule == ValidationRule::MinDisplayDuration));
    }

    #[test]
    fn test_overlap_violation() {
        let v = CaptionValidator::default();
        let caps = vec![
            good_cap(0, 2000, "First"),
            good_cap(1500, 3500, "Second"), // starts before first ends
        ];
        let violations = v.validate(&caps);
        assert!(violations
            .iter()
            .any(|v| v.rule == ValidationRule::NoOverlap));
    }

    #[test]
    fn test_no_overlap_with_gap() {
        let v = CaptionValidator::default();
        let caps = vec![good_cap(0, 1000, "First"), good_cap(1100, 2100, "Second")];
        assert!(!v
            .validate(&caps)
            .iter()
            .any(|v| v.rule == ValidationRule::NoOverlap));
    }

    #[test]
    fn test_max_reading_speed() {
        let v = CaptionValidator::default();
        // 20 words in 1 second = 1200 WPM → exceeds 200 WPM
        let text = vec![
            "one two three four five six seven eight nine ten".to_string(),
            "a b c d e f g h i j".to_string(),
        ];
        let cap = ValidatorCaption {
            start_ms: 0,
            end_ms: 1000,
            lines: text,
        };
        let violations = v.validate(&[cap]);
        assert!(violations
            .iter()
            .any(|v| v.rule == ValidationRule::MaxReadingSpeed));
    }

    #[test]
    fn test_max_rows_violation() {
        let v = CaptionValidator::default();
        let cap = ValidatorCaption {
            start_ms: 0,
            end_ms: 3000,
            lines: vec!["r1".into(), "r2".into(), "r3".into(), "r4".into()], // 4 > 3
        };
        let violations = v.validate(&[cap]);
        assert!(violations.iter().any(|v| v.rule == ValidationRule::MaxRows));
    }

    #[test]
    fn test_violation_counts() {
        let v = CaptionValidator::default();
        let caps = vec![
            good_cap(0, 400, "Short duration"),
            good_cap(300, 2400, "Overlap with previous"),
        ];
        let counts = v.violation_counts(&caps);
        assert!(!counts.is_empty());
    }

    #[test]
    fn test_empty_track_is_valid() {
        let v = CaptionValidator::default();
        assert!(v.is_valid(&[]));
    }

    #[test]
    fn test_validator_caption_word_count() {
        let c = ValidatorCaption::single(0, 1000, "one two three");
        assert_eq!(c.word_count(), 3);
    }

    #[test]
    fn test_validator_caption_full_text_multiline() {
        let cap = ValidatorCaption {
            start_ms: 0,
            end_ms: 1000,
            lines: vec!["Hello".into(), "World".into()],
        };
        assert_eq!(cap.full_text(), "Hello World");
    }

    #[test]
    fn test_validator_caption_duration() {
        let c = good_cap(500, 1500, "x");
        assert_eq!(c.duration_ms(), 1000);
    }

    #[test]
    fn test_rule_description_non_empty() {
        let desc = ValidationRule::NonEmpty.description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn test_violation_message_stored() {
        let v = CaptionValidator::default();
        let caps = vec![good_cap(0, 400, "Short")];
        let violations = v.validate(&caps);
        assert!(!violations[0].message.is_empty());
    }
}

#[cfg(test)]
mod broadcast_validator_tests {
    use super::{
        check_max_chars_per_line, check_max_lines, check_safe_area_margins, BroadcastCaption,
    };

    fn cap_2lines() -> BroadcastCaption {
        BroadcastCaption::new(vec!["First line".to_string(), "Second line".to_string()])
    }

    // ── check_max_lines ──────────────────────────────────────────────────────

    #[test]
    fn test_max_lines_within_limit() {
        let c = cap_2lines();
        assert!(check_max_lines(&c, 2));
        assert!(check_max_lines(&c, 3));
    }

    #[test]
    fn test_max_lines_exceeds_limit() {
        let c = cap_2lines();
        assert!(!check_max_lines(&c, 1));
    }

    #[test]
    fn test_max_lines_zero_means_unlimited() {
        let c = BroadcastCaption::new(vec![
            "L1".to_string(),
            "L2".to_string(),
            "L3".to_string(),
            "L4".to_string(),
        ]);
        assert!(check_max_lines(&c, 0));
    }

    #[test]
    fn test_max_lines_empty_caption() {
        let c = BroadcastCaption::new(vec![]);
        assert!(check_max_lines(&c, 2));
    }

    // ── check_max_chars_per_line ─────────────────────────────────────────────

    #[test]
    fn test_max_chars_per_line_within_limit() {
        let c = BroadcastCaption::single("Hello world");
        assert!(check_max_chars_per_line(&c, 42));
    }

    #[test]
    fn test_max_chars_per_line_exactly_at_limit() {
        let text = "a".repeat(32);
        let c = BroadcastCaption::single(text);
        assert!(check_max_chars_per_line(&c, 32));
    }

    #[test]
    fn test_max_chars_per_line_exceeds_limit() {
        let text = "a".repeat(33);
        let c = BroadcastCaption::single(text);
        assert!(!check_max_chars_per_line(&c, 32));
    }

    #[test]
    fn test_max_chars_per_line_multiline_one_long() {
        let c = BroadcastCaption::new(vec!["Short".to_string(), "a".repeat(50)]);
        assert!(!check_max_chars_per_line(&c, 42));
    }

    #[test]
    fn test_max_chars_per_line_zero_means_unlimited() {
        let text = "a".repeat(200);
        let c = BroadcastCaption::single(text);
        assert!(check_max_chars_per_line(&c, 0));
    }

    // ── check_safe_area_margins ──────────────────────────────────────────────

    #[test]
    fn test_safe_area_inside() {
        // x=10, y=80, w=80, h=15 → right=90, bottom=95; margins=5 → 0..95
        // All edges within 5%..95%? left: 10>=5 ✓, top: 80>=5 ✓, right: 90<=95 ✓, bottom: 95<=95 ✓
        let c = BroadcastCaption {
            lines: vec!["text".to_string()],
            x_pct: 10.0,
            y_pct: 80.0,
            w_pct: 80.0,
            h_pct: 15.0,
        };
        assert!(check_safe_area_margins(&c, 5.0));
    }

    #[test]
    fn test_safe_area_outside_left() {
        let c = BroadcastCaption {
            lines: vec!["text".to_string()],
            x_pct: 3.0, // < 10% margin
            y_pct: 10.0,
            w_pct: 80.0,
            h_pct: 15.0,
        };
        assert!(!check_safe_area_margins(&c, 10.0));
    }

    #[test]
    fn test_safe_area_outside_bottom() {
        let c = BroadcastCaption {
            lines: vec!["text".to_string()],
            x_pct: 10.0,
            y_pct: 85.0,
            w_pct: 80.0,
            h_pct: 20.0, // bottom edge = 105% > (100-10)%=90%
        };
        assert!(!check_safe_area_margins(&c, 10.0));
    }

    #[test]
    fn test_safe_area_zero_pct_always_ok() {
        let c = BroadcastCaption {
            lines: vec!["text".to_string()],
            x_pct: 0.0,
            y_pct: 0.0,
            w_pct: 100.0,
            h_pct: 100.0,
        };
        assert!(check_safe_area_margins(&c, 0.0));
    }

    #[test]
    fn test_safe_area_standard_bottom_region_10pct_margin() {
        // Standard bottom subtitle region used by ImscRegion::standard_bottom
        let c = BroadcastCaption {
            lines: vec!["Standard subtitle".to_string()],
            x_pct: 10.0,
            y_pct: 80.0,
            w_pct: 80.0,
            h_pct: 10.0, // bottom=90 which equals (100-10)=90 ✓
        };
        assert!(check_safe_area_margins(&c, 10.0));
    }
}

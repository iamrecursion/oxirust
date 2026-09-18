//! EDL validation rules, errors, and report.
//!
//! Provides `EdlValidationRule`, `ValidationError`, `EdlValidator`,
//! and `EdlValidationReport` for checking EDL event sequences for
//! common conformance issues.

#![allow(dead_code)]

use crate::edl_event::{EditType, EdlEvent, EdlEventList};

/// A named rule that can be evaluated against EDL events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdlValidationRule {
    /// Event numbers must be sequential starting from 1.
    SequentialNumbering,
    /// Record-out of each event must equal the record-in of the next.
    ContinuousTimeline,
    /// Reel name must not be empty or longer than 8 characters.
    ValidReelName,
    /// Source duration must be positive (source_out > source_in).
    PositiveSourceDuration,
    /// Record duration must be positive (record_out > record_in).
    PositiveRecordDuration,
    /// Wipe events must have a wipe number set.
    WipeHasPattern,
}

impl EdlValidationRule {
    /// The human-readable rule name.
    #[must_use]
    pub fn rule_name(self) -> &'static str {
        match self {
            Self::SequentialNumbering => "SequentialNumbering",
            Self::ContinuousTimeline => "ContinuousTimeline",
            Self::ValidReelName => "ValidReelName",
            Self::PositiveSourceDuration => "PositiveSourceDuration",
            Self::PositiveRecordDuration => "PositiveRecordDuration",
            Self::WipeHasPattern => "WipeHasPattern",
        }
    }

    /// Returns `true` if violations of this rule are fatal (blocking delivery).
    #[must_use]
    pub fn is_fatal(self) -> bool {
        matches!(
            self,
            Self::PositiveSourceDuration | Self::PositiveRecordDuration | Self::ValidReelName
        )
    }
}

/// A single validation failure.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// The event number where the violation occurred (0 = global).
    pub event_number: u32,
    /// Which rule was violated.
    pub rule: EdlValidationRule,
    /// Human-readable description of the problem.
    pub message: String,
}

impl ValidationError {
    /// Create a new `ValidationError`.
    #[must_use]
    pub fn new(event_number: u32, rule: EdlValidationRule, message: impl Into<String>) -> Self {
        Self {
            event_number,
            rule,
            message: message.into(),
        }
    }

    /// Returns `true` if this error is fatal (derived from the rule).
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        self.rule.is_fatal()
    }
}

/// Validator that checks an `EdlEventList` against a configurable set of rules.
#[derive(Debug, Clone)]
pub struct EdlValidator {
    rules: Vec<EdlValidationRule>,
}

impl EdlValidator {
    /// Create a validator with all rules enabled.
    #[must_use]
    pub fn all_rules() -> Self {
        Self {
            rules: vec![
                EdlValidationRule::SequentialNumbering,
                EdlValidationRule::ContinuousTimeline,
                EdlValidationRule::ValidReelName,
                EdlValidationRule::PositiveSourceDuration,
                EdlValidationRule::PositiveRecordDuration,
                EdlValidationRule::WipeHasPattern,
            ],
        }
    }

    /// Create a validator with only fatal rules enabled.
    #[must_use]
    pub fn fatal_only() -> Self {
        Self {
            rules: vec![
                EdlValidationRule::ValidReelName,
                EdlValidationRule::PositiveSourceDuration,
                EdlValidationRule::PositiveRecordDuration,
            ],
        }
    }

    /// Validate a single `EdlEvent` and return any errors found.
    #[must_use]
    pub fn validate_event(&self, event: &EdlEvent) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for &rule in &self.rules {
            match rule {
                EdlValidationRule::ValidReelName => {
                    if event.reel.is_empty() || event.reel.len() > 8 {
                        errors.push(ValidationError::new(
                            event.number,
                            rule,
                            format!("Reel name '{}' is invalid (must be 1-8 chars)", event.reel),
                        ));
                    }
                }
                EdlValidationRule::PositiveSourceDuration => {
                    if event.source_out <= event.source_in {
                        errors.push(ValidationError::new(
                            event.number,
                            rule,
                            "Source duration is zero or negative",
                        ));
                    }
                }
                EdlValidationRule::PositiveRecordDuration => {
                    if event.record_out <= event.record_in {
                        errors.push(ValidationError::new(
                            event.number,
                            rule,
                            "Record duration is zero or negative",
                        ));
                    }
                }
                EdlValidationRule::WipeHasPattern => {
                    if event.edit_type == EditType::Wipe && event.wipe_number.is_none() {
                        errors.push(ValidationError::new(
                            event.number,
                            rule,
                            "Wipe event is missing a wipe pattern number",
                        ));
                    }
                }
                // List-level rules are not checked per-event.
                EdlValidationRule::SequentialNumbering | EdlValidationRule::ContinuousTimeline => {}
            }
        }

        errors
    }

    /// Validate a full `EdlEventList` and return a `EdlValidationReport`.
    #[must_use]
    pub fn validate_list(&self, list: &EdlEventList) -> EdlValidationReport {
        let mut errors: Vec<ValidationError> = Vec::new();

        // Per-event rules.
        for event in list.events() {
            errors.extend(self.validate_event(event));
        }

        // List-level rules.
        if self.rules.contains(&EdlValidationRule::SequentialNumbering) {
            for (i, event) in list.events().iter().enumerate() {
                let expected = (i + 1) as u32;
                if event.number != expected {
                    errors.push(ValidationError::new(
                        event.number,
                        EdlValidationRule::SequentialNumbering,
                        format!("Expected event number {expected}, got {}", event.number),
                    ));
                }
            }
        }

        if self.rules.contains(&EdlValidationRule::ContinuousTimeline) {
            let events = list.events();
            for pair in events.windows(2) {
                if pair[0].record_out != pair[1].record_in {
                    errors.push(ValidationError::new(
                        pair[1].number,
                        EdlValidationRule::ContinuousTimeline,
                        format!(
                            "Gap/overlap: event {} record_out={} != event {} record_in={}",
                            pair[0].number, pair[0].record_out, pair[1].number, pair[1].record_in,
                        ),
                    ));
                }
            }
        }

        EdlValidationReport { errors }
    }
}

/// The result of running an `EdlValidator` over an event list.
#[derive(Debug, Clone, Default)]
pub struct EdlValidationReport {
    errors: Vec<ValidationError>,
}

impl EdlValidationReport {
    /// Returns `true` if there are any fatal errors.
    #[must_use]
    pub fn has_fatals(&self) -> bool {
        self.errors.iter().any(|e| e.is_fatal())
    }

    /// Returns `true` if the report contains no errors.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Total number of errors (fatal + warnings).
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Number of fatal errors.
    #[must_use]
    pub fn fatal_count(&self) -> usize {
        self.errors.iter().filter(|e| e.is_fatal()).count()
    }

    /// Return all errors as a slice.
    #[must_use]
    pub fn errors(&self) -> &[ValidationError] {
        &self.errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edl_event::EdlEvent;

    fn make_event(number: u32, reel: &str, si: u64, so: u64, ri: u64, ro: u64) -> EdlEvent {
        EdlEvent::new(number, reel, EditType::Cut, si, so, ri, ro)
    }

    // --- EdlValidationRule tests ---

    #[test]
    fn test_rule_name_sequential() {
        assert_eq!(
            EdlValidationRule::SequentialNumbering.rule_name(),
            "SequentialNumbering"
        );
    }

    #[test]
    fn test_rule_name_valid_reel() {
        assert_eq!(
            EdlValidationRule::ValidReelName.rule_name(),
            "ValidReelName"
        );
    }

    #[test]
    fn test_fatal_rules_are_fatal() {
        assert!(EdlValidationRule::PositiveSourceDuration.is_fatal());
        assert!(EdlValidationRule::PositiveRecordDuration.is_fatal());
        assert!(EdlValidationRule::ValidReelName.is_fatal());
    }

    #[test]
    fn test_non_fatal_rules() {
        assert!(!EdlValidationRule::SequentialNumbering.is_fatal());
        assert!(!EdlValidationRule::ContinuousTimeline.is_fatal());
        assert!(!EdlValidationRule::WipeHasPattern.is_fatal());
    }

    // --- ValidationError tests ---

    #[test]
    fn test_validation_error_is_fatal() {
        let e = ValidationError::new(1, EdlValidationRule::PositiveSourceDuration, "bad");
        assert!(e.is_fatal());
    }

    #[test]
    fn test_validation_error_not_fatal() {
        let e = ValidationError::new(1, EdlValidationRule::WipeHasPattern, "no wipe number");
        assert!(!e.is_fatal());
    }

    // --- EdlValidator per-event tests ---

    #[test]
    fn test_validate_event_ok() {
        let v = EdlValidator::all_rules();
        let ev = make_event(1, "A001", 0, 25, 0, 25);
        assert!(v.validate_event(&ev).is_empty());
    }

    #[test]
    fn test_validate_event_empty_reel() {
        let v = EdlValidator::all_rules();
        let ev = make_event(1, "", 0, 25, 0, 25);
        let errs = v.validate_event(&ev);
        assert!(!errs.is_empty());
        assert!(errs
            .iter()
            .any(|e| e.rule == EdlValidationRule::ValidReelName));
    }

    #[test]
    fn test_validate_event_reel_too_long() {
        let v = EdlValidator::all_rules();
        let ev = make_event(1, "TOOLONGNAME", 0, 25, 0, 25);
        let errs = v.validate_event(&ev);
        assert!(errs
            .iter()
            .any(|e| e.rule == EdlValidationRule::ValidReelName));
    }

    #[test]
    fn test_validate_event_zero_source_duration() {
        let v = EdlValidator::all_rules();
        let ev = make_event(1, "A001", 10, 10, 0, 10);
        let errs = v.validate_event(&ev);
        assert!(errs
            .iter()
            .any(|e| e.rule == EdlValidationRule::PositiveSourceDuration));
    }

    #[test]
    fn test_validate_event_zero_record_duration() {
        let v = EdlValidator::all_rules();
        let ev = make_event(1, "A001", 0, 25, 10, 10);
        let errs = v.validate_event(&ev);
        assert!(errs
            .iter()
            .any(|e| e.rule == EdlValidationRule::PositiveRecordDuration));
    }

    #[test]
    fn test_validate_wipe_missing_pattern() {
        let v = EdlValidator::all_rules();
        let ev = EdlEvent::new(1, "A001", EditType::Wipe, 0, 10, 0, 10);
        let errs = v.validate_event(&ev);
        assert!(errs
            .iter()
            .any(|e| e.rule == EdlValidationRule::WipeHasPattern));
    }

    #[test]
    fn test_validate_wipe_with_pattern_ok() {
        let v = EdlValidator::all_rules();
        let mut ev = EdlEvent::new(1, "A001", EditType::Wipe, 0, 10, 0, 10);
        ev.set_wipe_number(3);
        let errs = v.validate_event(&ev);
        assert!(!errs
            .iter()
            .any(|e| e.rule == EdlValidationRule::WipeHasPattern));
    }

    // --- EdlValidationReport tests ---

    #[test]
    fn test_report_ok_for_valid_list() {
        let v = EdlValidator::all_rules();
        let mut list = EdlEventList::new();
        list.add(make_event(1, "A001", 0, 25, 0, 25));
        list.add(make_event(2, "B001", 0, 25, 25, 50));
        let report = v.validate_list(&list);
        assert!(report.is_ok());
        assert!(!report.has_fatals());
    }

    #[test]
    fn test_report_detects_gap() {
        let v = EdlValidator::all_rules();
        let mut list = EdlEventList::new();
        list.add(make_event(1, "A001", 0, 25, 0, 25));
        // gap: record_in=30 but previous record_out=25
        list.add(make_event(2, "B001", 0, 25, 30, 55));
        let report = v.validate_list(&list);
        assert!(!report.is_ok());
        assert!(report
            .errors()
            .iter()
            .any(|e| e.rule == EdlValidationRule::ContinuousTimeline));
    }

    #[test]
    fn test_report_has_fatals_on_bad_reel() {
        let v = EdlValidator::all_rules();
        let mut list = EdlEventList::new();
        list.add(make_event(1, "", 0, 10, 0, 10));
        let report = v.validate_list(&list);
        assert!(report.has_fatals());
    }

    #[test]
    fn test_report_counts() {
        let v = EdlValidator::all_rules();
        let mut list = EdlEventList::new();
        list.add(make_event(1, "", 0, 0, 0, 0)); // 3 violations: reel, src, rec
        let report = v.validate_list(&list);
        assert!(report.error_count() >= 3);
        assert!(report.fatal_count() >= 2);
    }
}

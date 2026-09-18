//! Specification rule validation for media conform in `OxiMedia`.
//!
//! [`SpecValidator`] holds a collection of [`SpecRule`]s and runs them against
//! arbitrary key/value property maps, producing a [`SpecReport`] with pass/fail
//! details.

#![allow(dead_code)]

use std::collections::HashMap;

/// Severity level of a spec rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RuleSeverity {
    /// Informational; failing does not block delivery.
    Info,
    /// Warning; failing is undesirable but not a hard error.
    Warning,
    /// Error; failing must be corrected before delivery.
    Error,
}

impl RuleSeverity {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// A rule that can be evaluated against a property map.
#[derive(Debug, Clone)]
pub enum SpecRule {
    /// Property must equal an exact string value.
    Equals {
        /// Property key to check.
        key: String,
        /// Expected value.
        expected: String,
        /// Rule severity.
        severity: RuleSeverity,
    },
    /// Property, parsed as `f64`, must be >= `min` and <= `max`.
    Range {
        /// Property key to check.
        key: String,
        /// Minimum allowed value (inclusive).
        min: f64,
        /// Maximum allowed value (inclusive).
        max: f64,
        /// Rule severity.
        severity: RuleSeverity,
    },
    /// Property must be present (non-empty).
    Required {
        /// Property key that must be present.
        key: String,
        /// Rule severity.
        severity: RuleSeverity,
    },
}

impl SpecRule {
    /// Return the severity of this rule.
    #[must_use]
    pub fn severity(&self) -> RuleSeverity {
        match self {
            Self::Equals { severity, .. }
            | Self::Range { severity, .. }
            | Self::Required { severity, .. } => *severity,
        }
    }

    /// Return `true` if this rule passes for the given property map.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn evaluate(&self, props: &HashMap<String, String>) -> bool {
        match self {
            Self::Equals { key, expected, .. } => props.get(key) == Some(expected),
            Self::Range { key, min, max, .. } => props
                .get(key)
                .and_then(|v| v.parse::<f64>().ok())
                .is_some_and(|n| n >= *min && n <= *max),
            Self::Required { key, .. } => props.get(key).is_some_and(|v| !v.is_empty()),
        }
    }

    /// Short description of the rule.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::Equals { key, expected, .. } => format!("{key} == {expected}"),
            Self::Range { key, min, max, .. } => format!("{key} in [{min}, {max}]"),
            Self::Required { key, .. } => format!("{key} required"),
        }
    }
}

/// Outcome of evaluating a single rule.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Short description of the rule.
    pub rule_description: String,
    /// Whether the rule passed.
    pub passed: bool,
    /// Severity if it failed.
    pub severity: RuleSeverity,
}

impl ValidationResult {
    /// Return `true` when the rule passed.
    #[must_use]
    pub fn is_pass(&self) -> bool {
        self.passed
    }

    /// Return `true` when the rule failed at error severity.
    #[must_use]
    pub fn is_hard_failure(&self) -> bool {
        !self.passed && self.severity == RuleSeverity::Error
    }
}

/// Validates a property map against a set of [`SpecRule`]s.
#[derive(Debug, Default)]
pub struct SpecValidator {
    rules: Vec<SpecRule>,
}

impl SpecValidator {
    /// Create a new, empty validator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule to the validator.
    pub fn add_rule(&mut self, rule: SpecRule) {
        self.rules.push(rule);
    }

    /// Validate a property map and return a [`SpecReport`].
    #[must_use]
    pub fn validate(&self, props: &HashMap<String, String>) -> SpecReport {
        let results = self
            .rules
            .iter()
            .map(|r| ValidationResult {
                rule_description: r.description(),
                passed: r.evaluate(props),
                severity: r.severity(),
            })
            .collect();
        SpecReport { results }
    }

    /// Number of registered rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

/// Summary of validation against all rules.
#[derive(Debug)]
pub struct SpecReport {
    /// Per-rule results.
    pub results: Vec<ValidationResult>,
}

impl SpecReport {
    /// Return results that failed.
    #[must_use]
    pub fn failures(&self) -> Vec<&ValidationResult> {
        self.results.iter().filter(|r| !r.is_pass()).collect()
    }

    /// Return results that failed at Error severity.
    #[must_use]
    pub fn hard_failures(&self) -> Vec<&ValidationResult> {
        self.results
            .iter()
            .filter(|r| r.is_hard_failure())
            .collect()
    }

    /// Return `true` when all rules passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(ValidationResult::is_pass)
    }

    /// Count of passing rules.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.results.iter().filter(|r| r.is_pass()).count()
    }

    /// Count of failing rules.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.results.iter().filter(|r| !r.is_pass()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    // ── RuleSeverity ─────────────────────────────────────────────────────────

    #[test]
    fn test_severity_order() {
        assert!(RuleSeverity::Error > RuleSeverity::Warning);
        assert!(RuleSeverity::Warning > RuleSeverity::Info);
    }

    #[test]
    fn test_severity_labels() {
        assert_eq!(RuleSeverity::Info.label(), "info");
        assert_eq!(RuleSeverity::Warning.label(), "warning");
        assert_eq!(RuleSeverity::Error.label(), "error");
    }

    // ── SpecRule::Equals ─────────────────────────────────────────────────────

    #[test]
    fn test_equals_pass() {
        let rule = SpecRule::Equals {
            key: "codec".into(),
            expected: "h264".into(),
            severity: RuleSeverity::Error,
        };
        assert!(rule.evaluate(&props(&[("codec", "h264")])));
    }

    #[test]
    fn test_equals_fail() {
        let rule = SpecRule::Equals {
            key: "codec".into(),
            expected: "h264".into(),
            severity: RuleSeverity::Error,
        };
        assert!(!rule.evaluate(&props(&[("codec", "hevc")])));
    }

    #[test]
    fn test_equals_missing_key() {
        let rule = SpecRule::Equals {
            key: "codec".into(),
            expected: "h264".into(),
            severity: RuleSeverity::Error,
        };
        assert!(!rule.evaluate(&props(&[])));
    }

    // ── SpecRule::Range ──────────────────────────────────────────────────────

    #[test]
    fn test_range_pass() {
        let rule = SpecRule::Range {
            key: "fps".into(),
            min: 23.976,
            max: 30.0,
            severity: RuleSeverity::Warning,
        };
        assert!(rule.evaluate(&props(&[("fps", "25.0")])));
    }

    #[test]
    fn test_range_fail_too_high() {
        let rule = SpecRule::Range {
            key: "fps".into(),
            min: 23.976,
            max: 30.0,
            severity: RuleSeverity::Warning,
        };
        assert!(!rule.evaluate(&props(&[("fps", "60.0")])));
    }

    #[test]
    fn test_range_fail_non_numeric() {
        let rule = SpecRule::Range {
            key: "fps".into(),
            min: 0.0,
            max: 60.0,
            severity: RuleSeverity::Info,
        };
        assert!(!rule.evaluate(&props(&[("fps", "ntsc")])));
    }

    // ── SpecRule::Required ───────────────────────────────────────────────────

    #[test]
    fn test_required_pass() {
        let rule = SpecRule::Required {
            key: "title".into(),
            severity: RuleSeverity::Error,
        };
        assert!(rule.evaluate(&props(&[("title", "My Film")])));
    }

    #[test]
    fn test_required_fail_missing() {
        let rule = SpecRule::Required {
            key: "title".into(),
            severity: RuleSeverity::Error,
        };
        assert!(!rule.evaluate(&props(&[])));
    }

    #[test]
    fn test_required_fail_empty() {
        let rule = SpecRule::Required {
            key: "title".into(),
            severity: RuleSeverity::Error,
        };
        assert!(!rule.evaluate(&props(&[("title", "")])));
    }

    // ── SpecValidator & SpecReport ───────────────────────────────────────────

    #[test]
    fn test_validator_all_pass() {
        let mut v = SpecValidator::new();
        v.add_rule(SpecRule::Equals {
            key: "codec".into(),
            expected: "h264".into(),
            severity: RuleSeverity::Error,
        });
        v.add_rule(SpecRule::Required {
            key: "title".into(),
            severity: RuleSeverity::Warning,
        });
        let report = v.validate(&props(&[("codec", "h264"), ("title", "Film")]));
        assert!(report.all_passed());
        assert_eq!(report.failure_count(), 0);
    }

    #[test]
    fn test_validator_partial_fail() {
        let mut v = SpecValidator::new();
        v.add_rule(SpecRule::Equals {
            key: "codec".into(),
            expected: "h264".into(),
            severity: RuleSeverity::Error,
        });
        v.add_rule(SpecRule::Equals {
            key: "fps".into(),
            expected: "25".into(),
            severity: RuleSeverity::Warning,
        });
        let report = v.validate(&props(&[("codec", "hevc"), ("fps", "25")]));
        assert_eq!(report.failure_count(), 1);
        assert_eq!(report.pass_count(), 1);
        assert!(!report.all_passed());
    }

    #[test]
    fn test_report_hard_failures() {
        let mut v = SpecValidator::new();
        v.add_rule(SpecRule::Equals {
            key: "codec".into(),
            expected: "h264".into(),
            severity: RuleSeverity::Error,
        });
        v.add_rule(SpecRule::Equals {
            key: "fps".into(),
            expected: "25".into(),
            severity: RuleSeverity::Warning,
        });
        let report = v.validate(&props(&[("codec", "hevc"), ("fps", "30")]));
        assert_eq!(report.hard_failures().len(), 1);
        assert_eq!(report.failures().len(), 2);
    }

    #[test]
    fn test_validator_rule_count() {
        let mut v = SpecValidator::new();
        assert_eq!(v.rule_count(), 0);
        v.add_rule(SpecRule::Required {
            key: "k".into(),
            severity: RuleSeverity::Info,
        });
        assert_eq!(v.rule_count(), 1);
    }
}

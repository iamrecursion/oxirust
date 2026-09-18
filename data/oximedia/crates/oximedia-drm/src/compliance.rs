//! DRM compliance checking: output control, robustness rules, compliance reporting.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Output type that may be subject to compliance controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputType {
    /// HDMI (digital audio/video).
    Hdmi,
    /// DisplayPort.
    DisplayPort,
    /// Bluetooth audio.
    BluetoothAudio,
    /// Analog video (composite, component).
    AnalogVideo,
    /// Analog audio.
    AnalogAudio,
    /// Screen capture / recording API.
    ScreenCapture,
    /// Cast/screen mirroring.
    Cast,
    /// Internal display.
    InternalDisplay,
}

/// Permission level for a given output type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OutputPermission {
    /// Output is fully blocked.
    Blocked,
    /// Output allowed but content protection required (e.g. HDCP).
    ProtectedOnly,
    /// Output allowed without restrictions.
    Allowed,
}

/// Output control policy: maps each output type to its permission.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutputControlPolicy {
    rules: HashMap<String, OutputPermission>,
}

impl OutputControlPolicy {
    /// Create a new empty policy (all outputs implicitly allowed).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the permission for an output type.
    pub fn set(&mut self, output: OutputType, perm: OutputPermission) {
        self.rules.insert(format!("{output:?}"), perm);
    }

    /// Get the permission for an output type (`Allowed` if not specified).
    #[must_use]
    pub fn get(&self, output: OutputType) -> OutputPermission {
        self.rules
            .get(&format!("{output:?}"))
            .copied()
            .unwrap_or(OutputPermission::Allowed)
    }

    /// Return `true` if the output is fully blocked.
    #[must_use]
    pub fn is_blocked(&self, output: OutputType) -> bool {
        self.get(output) == OutputPermission::Blocked
    }
}

/// Robustness level required of the DRM implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RobustnessLevel {
    /// Software-only protection (lowest).
    SoftwareSecure,
    /// Trusted Execution Environment.
    TeeRequired,
    /// Hardware-level protection (highest).
    HardwareSecure,
}

/// A robustness rule assertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessRule {
    /// Human-readable name for this rule.
    pub name: String,
    /// Required minimum robustness level.
    pub required_level: RobustnessLevel,
    /// Whether this rule is mandatory (failure = compliance failure) or advisory.
    pub mandatory: bool,
}

impl RobustnessRule {
    /// Create a mandatory robustness rule.
    #[must_use]
    pub fn mandatory(name: impl Into<String>, level: RobustnessLevel) -> Self {
        Self {
            name: name.into(),
            required_level: level,
            mandatory: true,
        }
    }

    /// Create an advisory robustness rule.
    #[must_use]
    pub fn advisory(name: impl Into<String>, level: RobustnessLevel) -> Self {
        Self {
            name: name.into(),
            required_level: level,
            mandatory: false,
        }
    }

    /// Check whether the provided level satisfies this rule.
    #[must_use]
    pub fn check(&self, provided: RobustnessLevel) -> bool {
        provided >= self.required_level
    }
}

/// Result of checking a single robustness rule.
#[derive(Debug, Clone)]
pub struct RuleResult {
    /// Rule name.
    pub rule_name: String,
    /// Whether the rule passed.
    pub passed: bool,
    /// Whether the rule was mandatory.
    pub mandatory: bool,
    /// Optional human-readable explanation.
    pub detail: Option<String>,
}

/// Aggregate compliance report.
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    /// Content identifier.
    pub content_id: String,
    /// Individual rule results.
    pub rule_results: Vec<RuleResult>,
    /// Whether overall compliance was achieved (all mandatory rules passed).
    pub compliant: bool,
    /// Number of failed mandatory rules.
    pub mandatory_failures: usize,
    /// Number of failed advisory rules.
    pub advisory_failures: usize,
}

impl ComplianceReport {
    /// Build a report by evaluating all rules against a provided robustness level.
    #[must_use]
    pub fn evaluate(
        content_id: impl Into<String>,
        rules: &[RobustnessRule],
        provided: RobustnessLevel,
    ) -> Self {
        let mut rule_results = Vec::new();
        let mut mandatory_failures = 0usize;
        let mut advisory_failures = 0usize;

        for rule in rules {
            let passed = rule.check(provided);
            if !passed {
                if rule.mandatory {
                    mandatory_failures += 1;
                } else {
                    advisory_failures += 1;
                }
            }
            rule_results.push(RuleResult {
                rule_name: rule.name.clone(),
                passed,
                mandatory: rule.mandatory,
                detail: if passed {
                    None
                } else {
                    Some(format!(
                        "Required {:?}, got {:?}",
                        rule.required_level, provided
                    ))
                },
            });
        }

        let compliant = mandatory_failures == 0;

        Self {
            content_id: content_id.into(),
            rule_results,
            compliant,
            mandatory_failures,
            advisory_failures,
        }
    }

    /// Human-readable summary of the compliance check.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.compliant {
            format!(
                "[COMPLIANT] {} — {} advisory warnings",
                self.content_id, self.advisory_failures
            )
        } else {
            format!(
                "[NON-COMPLIANT] {} — {} mandatory failures, {} advisory warnings",
                self.content_id, self.mandatory_failures, self.advisory_failures
            )
        }
    }

    /// Collect names of all failed mandatory rules.
    #[must_use]
    pub fn failed_mandatory_rules(&self) -> Vec<&str> {
        self.rule_results
            .iter()
            .filter(|r| r.mandatory && !r.passed)
            .map(|r| r.rule_name.as_str())
            .collect()
    }
}

/// Checker that encapsulates a set of rules and an output control policy.
#[derive(Debug)]
pub struct ComplianceChecker {
    rules: Vec<RobustnessRule>,
    output_policy: OutputControlPolicy,
}

impl ComplianceChecker {
    /// Create a new checker.
    #[must_use]
    pub fn new(rules: Vec<RobustnessRule>, output_policy: OutputControlPolicy) -> Self {
        Self {
            rules,
            output_policy,
        }
    }

    /// Check compliance for the given content and robustness level.
    #[must_use]
    pub fn check(&self, content_id: &str, provided: RobustnessLevel) -> ComplianceReport {
        ComplianceReport::evaluate(content_id, &self.rules, provided)
    }

    /// Check whether a specific output is permitted under the policy.
    #[must_use]
    pub fn output_permitted(&self, output: OutputType) -> bool {
        self.output_policy.get(output) != OutputPermission::Blocked
    }

    /// Add a rule at runtime.
    pub fn add_rule(&mut self, rule: RobustnessRule) {
        self.rules.push(rule);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_control_default_allowed() {
        let policy = OutputControlPolicy::new();
        assert_eq!(policy.get(OutputType::Hdmi), OutputPermission::Allowed);
    }

    #[test]
    fn test_output_control_set_blocked() {
        let mut policy = OutputControlPolicy::new();
        policy.set(OutputType::ScreenCapture, OutputPermission::Blocked);
        assert!(policy.is_blocked(OutputType::ScreenCapture));
        assert!(!policy.is_blocked(OutputType::Hdmi));
    }

    #[test]
    fn test_output_control_protected_only() {
        let mut policy = OutputControlPolicy::new();
        policy.set(OutputType::Hdmi, OutputPermission::ProtectedOnly);
        assert_eq!(
            policy.get(OutputType::Hdmi),
            OutputPermission::ProtectedOnly
        );
        assert!(!policy.is_blocked(OutputType::Hdmi));
    }

    #[test]
    fn test_output_permission_ordering() {
        assert!(OutputPermission::Blocked < OutputPermission::ProtectedOnly);
        assert!(OutputPermission::ProtectedOnly < OutputPermission::Allowed);
    }

    #[test]
    fn test_robustness_level_ordering() {
        assert!(RobustnessLevel::SoftwareSecure < RobustnessLevel::TeeRequired);
        assert!(RobustnessLevel::TeeRequired < RobustnessLevel::HardwareSecure);
    }

    #[test]
    fn test_robustness_rule_mandatory_check_pass() {
        let rule = RobustnessRule::mandatory("tee", RobustnessLevel::TeeRequired);
        assert!(rule.check(RobustnessLevel::TeeRequired));
        assert!(rule.check(RobustnessLevel::HardwareSecure));
    }

    #[test]
    fn test_robustness_rule_mandatory_check_fail() {
        let rule = RobustnessRule::mandatory("hw", RobustnessLevel::HardwareSecure);
        assert!(!rule.check(RobustnessLevel::SoftwareSecure));
        assert!(!rule.check(RobustnessLevel::TeeRequired));
    }

    #[test]
    fn test_robustness_rule_advisory() {
        let rule = RobustnessRule::advisory("advisory_hw", RobustnessLevel::HardwareSecure);
        assert!(!rule.mandatory);
    }

    #[test]
    fn test_compliance_report_compliant() {
        let rules = vec![RobustnessRule::mandatory(
            "tee",
            RobustnessLevel::TeeRequired,
        )];
        let report = ComplianceReport::evaluate("c001", &rules, RobustnessLevel::HardwareSecure);
        assert!(report.compliant);
        assert_eq!(report.mandatory_failures, 0);
        assert!(report.summary().starts_with("[COMPLIANT]"));
    }

    #[test]
    fn test_compliance_report_non_compliant() {
        let rules = vec![RobustnessRule::mandatory(
            "hw",
            RobustnessLevel::HardwareSecure,
        )];
        let report = ComplianceReport::evaluate("c002", &rules, RobustnessLevel::SoftwareSecure);
        assert!(!report.compliant);
        assert_eq!(report.mandatory_failures, 1);
        assert!(report.summary().contains("NON-COMPLIANT"));
    }

    #[test]
    fn test_compliance_report_advisory_only() {
        let rules = vec![RobustnessRule::advisory(
            "soft_advisory",
            RobustnessLevel::TeeRequired,
        )];
        let report = ComplianceReport::evaluate("c003", &rules, RobustnessLevel::SoftwareSecure);
        // Advisory failure does not make it non-compliant.
        assert!(report.compliant);
        assert_eq!(report.advisory_failures, 1);
    }

    #[test]
    fn test_compliance_report_failed_mandatory_names() {
        let rules = vec![
            RobustnessRule::mandatory("rule_a", RobustnessLevel::HardwareSecure),
            RobustnessRule::mandatory("rule_b", RobustnessLevel::TeeRequired),
        ];
        let report = ComplianceReport::evaluate("c004", &rules, RobustnessLevel::SoftwareSecure);
        let names = report.failed_mandatory_rules();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"rule_a"));
        assert!(names.contains(&"rule_b"));
    }

    #[test]
    fn test_compliance_checker_output_permitted() {
        let mut policy = OutputControlPolicy::new();
        policy.set(OutputType::ScreenCapture, OutputPermission::Blocked);
        let checker = ComplianceChecker::new(vec![], policy);
        assert!(!checker.output_permitted(OutputType::ScreenCapture));
        assert!(checker.output_permitted(OutputType::Hdmi));
    }

    #[test]
    fn test_compliance_checker_add_rule() {
        let mut checker = ComplianceChecker::new(vec![], OutputControlPolicy::new());
        checker.add_rule(RobustnessRule::mandatory(
            "new_rule",
            RobustnessLevel::TeeRequired,
        ));
        let report = checker.check("c005", RobustnessLevel::SoftwareSecure);
        assert!(!report.compliant);
    }
}

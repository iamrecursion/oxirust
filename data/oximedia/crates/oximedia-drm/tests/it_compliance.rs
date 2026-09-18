//! Integration tests for `oximedia_drm::compliance` against
//! CPSA-style (Content Protection Security Architecture) rules.
//!
//! CPSA defines content-tier-dependent rules for output protection and DRM
//! implementation robustness. These tests map the abstract notion of HDCP
//! versions to the concrete `RobustnessLevel` ladder and exercise the
//! `ComplianceChecker` surface for representative 4K / HD scenarios.
//!
//! Mapping used in this test file:
//!   - HDCP 1.4 → `RobustnessLevel::SoftwareSecure`
//!   - HDCP 2.2 → `RobustnessLevel::TeeRequired`
//!   - HDCP 2.3 → `RobustnessLevel::HardwareSecure`
//!
//! This mapping is solely a local helper for the test file. The production
//! compliance module evaluates `RobustnessLevel` directly without ever
//! using the strings "HDCP 1.4" etc.

use oximedia_drm::compliance::{
    ComplianceChecker, ComplianceReport, OutputControlPolicy, OutputPermission, OutputType,
    RobustnessLevel, RobustnessRule,
};

/// Translate an HDCP wire version to the matching `RobustnessLevel`.
fn hdcp_version_to_robustness(hdcp: &str) -> RobustnessLevel {
    match hdcp {
        "1.0" | "1.1" | "1.2" | "1.3" | "1.4" => RobustnessLevel::SoftwareSecure,
        "2.0" | "2.1" | "2.2" => RobustnessLevel::TeeRequired,
        "2.3" => RobustnessLevel::HardwareSecure,
        _ => RobustnessLevel::SoftwareSecure,
    }
}

/// Build a CPSA-style policy for 4K UHD content:
/// requires hardware-backed protection (TEE or better) and blocks screen
/// capture / analog video outputs entirely.
fn build_4k_uhd_policy() -> (Vec<RobustnessRule>, OutputControlPolicy) {
    let rules = vec![
        RobustnessRule::mandatory("4k-protection-baseline", RobustnessLevel::TeeRequired),
        RobustnessRule::mandatory("4k-key-handling", RobustnessLevel::TeeRequired),
    ];

    let mut policy = OutputControlPolicy::new();
    policy.set(OutputType::ScreenCapture, OutputPermission::Blocked);
    policy.set(OutputType::AnalogVideo, OutputPermission::Blocked);
    policy.set(OutputType::Hdmi, OutputPermission::ProtectedOnly);

    (rules, policy)
}

/// Build a CPSA-style policy for HD content:
/// software-secure baseline is sufficient.
fn build_hd_policy() -> (Vec<RobustnessRule>, OutputControlPolicy) {
    let rules = vec![RobustnessRule::mandatory(
        "hd-baseline",
        RobustnessLevel::SoftwareSecure,
    )];

    let mut policy = OutputControlPolicy::new();
    policy.set(OutputType::ScreenCapture, OutputPermission::Blocked);

    (rules, policy)
}

#[test]
fn hdcp_2_2_with_output_protection_passes_4k_cpsa_check() {
    let (rules, policy) = build_4k_uhd_policy();
    let checker = ComplianceChecker::new(rules, policy);

    let level = hdcp_version_to_robustness("2.2");
    let report: ComplianceReport = checker.check("uhd-movie-001", level);

    assert!(report.compliant, "HDCP 2.2 is sufficient for 4K UHD");
    assert_eq!(report.mandatory_failures, 0);
    assert!(report.summary().starts_with("[COMPLIANT]"));

    // ScreenCapture should be blocked by the CPSA policy.
    assert!(
        !checker.output_permitted(OutputType::ScreenCapture),
        "4K UHD must block screen capture"
    );
    // HDMI is ProtectedOnly — not blocked outright.
    assert!(
        checker.output_permitted(OutputType::Hdmi),
        "HDMI is allowed in ProtectedOnly mode"
    );
    // AnalogVideo is blocked.
    assert!(!checker.output_permitted(OutputType::AnalogVideo));
}

#[test]
fn hdcp_1_4_fails_4k_cpsa_check_with_insufficient_robustness() {
    let (rules, policy) = build_4k_uhd_policy();
    let checker = ComplianceChecker::new(rules, policy);

    let level = hdcp_version_to_robustness("1.4");
    let report = checker.check("uhd-movie-002", level);

    assert!(!report.compliant, "HDCP 1.4 must NOT pass 4K CPSA check");
    assert_eq!(
        report.mandatory_failures, 2,
        "both 4K rules fail at software-secure level"
    );
    assert!(report.summary().contains("NON-COMPLIANT"));

    // The failed-mandatory-rules helper should expose both rule names.
    let failed = report.failed_mandatory_rules();
    assert!(failed.contains(&"4k-protection-baseline"));
    assert!(failed.contains(&"4k-key-handling"));
}

#[test]
fn hdcp_2_3_hardware_secure_satisfies_strictest_rule() {
    let mut rules = Vec::new();
    rules.push(RobustnessRule::mandatory(
        "premium-4k",
        RobustnessLevel::HardwareSecure,
    ));

    let checker = ComplianceChecker::new(rules, OutputControlPolicy::new());

    let level = hdcp_version_to_robustness("2.3");
    let report = checker.check("premium-uhd", level);
    assert!(report.compliant, "HDCP 2.3 -> HardwareSecure is compliant");
    assert_eq!(report.mandatory_failures, 0);

    // Even one step down (TeeRequired) fails the strictest rule.
    let report_tee = checker.check("premium-uhd", RobustnessLevel::TeeRequired);
    assert!(
        !report_tee.compliant,
        "TeeRequired is insufficient for HardwareSecure"
    );
}

#[test]
fn hd_content_passes_with_minimal_software_secure_baseline() {
    let (rules, policy) = build_hd_policy();
    let checker = ComplianceChecker::new(rules, policy);

    // Even HDCP 1.0 → SoftwareSecure passes the HD baseline.
    let level = hdcp_version_to_robustness("1.0");
    let report = checker.check("hd-content", level);
    assert!(
        report.compliant,
        "HD content is compliant at SoftwareSecure"
    );

    // Screen capture remains blocked even for HD content.
    assert!(!checker.output_permitted(OutputType::ScreenCapture));
    // Other outputs default to Allowed.
    assert!(checker.output_permitted(OutputType::Hdmi));
    assert!(checker.output_permitted(OutputType::DisplayPort));
}

#[test]
fn advisory_rules_emit_warning_but_remain_compliant() {
    let rules = vec![
        RobustnessRule::mandatory("baseline-hd", RobustnessLevel::SoftwareSecure),
        RobustnessRule::advisory("prefer-tee", RobustnessLevel::TeeRequired),
    ];
    let checker = ComplianceChecker::new(rules, OutputControlPolicy::new());

    let report = checker.check("hybrid-content", RobustnessLevel::SoftwareSecure);
    assert!(
        report.compliant,
        "advisory rule failure does NOT block compliance"
    );
    assert_eq!(report.mandatory_failures, 0);
    assert_eq!(report.advisory_failures, 1);
    assert!(report.summary().contains("1 advisory warnings"));
}

#[test]
fn empty_rule_set_is_trivially_compliant() {
    let checker = ComplianceChecker::new(Vec::new(), OutputControlPolicy::new());
    let report = checker.check("any-content", RobustnessLevel::SoftwareSecure);
    assert!(report.compliant);
    assert_eq!(report.mandatory_failures, 0);
    assert_eq!(report.advisory_failures, 0);
    assert!(report.rule_results.is_empty());
}

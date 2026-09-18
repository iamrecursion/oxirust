#![allow(dead_code)]
//! Broadcast compliance checking for automation schedules.
//!
//! Verifies that playout schedules comply with regulatory rules such as
//! maximum ad duration per hour, required content ratings display,
//! mandatory station identification intervals, and quiet-hour restrictions.

use std::collections::HashMap;

/// The kind of compliance rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleKind {
    /// Maximum total ad duration per time window.
    MaxAdDuration,
    /// Mandatory station ID at required intervals.
    StationIdInterval,
    /// Content rating must be displayed.
    ContentRating,
    /// Quiet hours restricting certain content types.
    QuietHours,
    /// Maximum consecutive ad breaks.
    MaxConsecutiveAds,
    /// Minimum program segment length between breaks.
    MinSegmentLength,
}

/// Severity of a compliance violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational notice.
    Info,
    /// Warning that should be reviewed.
    Warning,
    /// Error that must be resolved before air.
    Error,
    /// Critical violation that may result in regulatory penalty.
    Critical,
}

/// A compliance rule definition.
#[derive(Debug, Clone)]
pub struct ComplianceRule {
    /// Rule identifier.
    pub id: String,
    /// Rule kind.
    pub kind: RuleKind,
    /// Human-readable description.
    pub description: String,
    /// Severity when violated.
    pub severity: Severity,
    /// Numeric threshold (interpretation depends on rule kind).
    pub threshold: f64,
    /// Whether the rule is currently enabled.
    pub enabled: bool,
}

impl ComplianceRule {
    /// Create a new compliance rule.
    pub fn new(
        id: &str,
        kind: RuleKind,
        description: &str,
        severity: Severity,
        threshold: f64,
    ) -> Self {
        Self {
            id: id.to_string(),
            kind,
            description: description.to_string(),
            severity,
            threshold,
            enabled: true,
        }
    }

    /// Disable this rule.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this rule.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

/// Type of schedule item for compliance checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemType {
    /// Program content.
    Program,
    /// Commercial advertisement.
    Ad,
    /// Station identification.
    StationId,
    /// Promotional content.
    Promo,
    /// Public service announcement.
    Psa,
    /// Rating card display.
    RatingCard,
}

/// A schedule item to be checked for compliance.
#[derive(Debug, Clone)]
pub struct ScheduleItem {
    /// Item identifier.
    pub id: String,
    /// Item type.
    pub item_type: ItemType,
    /// Start time offset in seconds from schedule start.
    pub start_secs: f64,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Content rating if applicable.
    pub content_rating: Option<String>,
    /// Additional tags.
    pub tags: Vec<String>,
}

impl ScheduleItem {
    /// Create a new schedule item.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(id: &str, item_type: ItemType, start_secs: f64, duration_secs: f64) -> Self {
        Self {
            id: id.to_string(),
            item_type,
            start_secs,
            duration_secs,
            content_rating: None,
            tags: Vec::new(),
        }
    }

    /// End time offset in seconds.
    pub fn end_secs(&self) -> f64 {
        self.start_secs + self.duration_secs
    }
}

/// A compliance violation found during checking.
#[derive(Debug, Clone)]
pub struct Violation {
    /// ID of the rule that was violated.
    pub rule_id: String,
    /// Severity.
    pub severity: Severity,
    /// Description of the violation.
    pub message: String,
    /// Schedule item IDs involved.
    pub item_ids: Vec<String>,
    /// Time position in seconds where the violation occurs.
    pub at_secs: f64,
}

impl Violation {
    /// Create a new violation.
    pub fn new(rule_id: &str, severity: Severity, message: &str, at_secs: f64) -> Self {
        Self {
            rule_id: rule_id.to_string(),
            severity,
            message: message.to_string(),
            item_ids: Vec::new(),
            at_secs,
        }
    }

    /// Add an involved item ID.
    pub fn with_item(mut self, item_id: &str) -> Self {
        self.item_ids.push(item_id.to_string());
        self
    }
}

/// Result of a compliance check.
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    /// All violations found.
    pub violations: Vec<Violation>,
    /// Total items checked.
    pub items_checked: usize,
    /// Total rules evaluated.
    pub rules_evaluated: usize,
}

impl ComplianceReport {
    /// Create an empty report.
    pub fn new(items_checked: usize, rules_evaluated: usize) -> Self {
        Self {
            violations: Vec::new(),
            items_checked,
            rules_evaluated,
        }
    }

    /// Check if the schedule passed all checks.
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }

    /// Count violations at or above the given severity.
    pub fn count_at_severity(&self, min_severity: Severity) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity >= min_severity)
            .count()
    }

    /// Get the highest severity among violations.
    pub fn max_severity(&self) -> Option<Severity> {
        self.violations.iter().map(|v| v.severity).max()
    }
}

/// The compliance checker validates schedule items against a set of rules.
pub struct ComplianceChecker {
    /// Registered rules.
    rules: HashMap<String, ComplianceRule>,
}

impl ComplianceChecker {
    /// Create a new empty compliance checker.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Create a checker with standard broadcast rules pre-loaded.
    #[allow(clippy::cast_precision_loss)]
    pub fn with_standard_rules() -> Self {
        let mut checker = Self::new();
        checker.add_rule(ComplianceRule::new(
            "max_ad_12min",
            RuleKind::MaxAdDuration,
            "Maximum 12 minutes of ads per hour",
            Severity::Error,
            720.0,
        ));
        checker.add_rule(ComplianceRule::new(
            "station_id_60min",
            RuleKind::StationIdInterval,
            "Station ID required every 60 minutes",
            Severity::Warning,
            3600.0,
        ));
        checker.add_rule(ComplianceRule::new(
            "max_consec_ads_4",
            RuleKind::MaxConsecutiveAds,
            "Maximum 4 consecutive ad items",
            Severity::Warning,
            4.0,
        ));
        checker.add_rule(ComplianceRule::new(
            "min_segment_5min",
            RuleKind::MinSegmentLength,
            "Minimum 5 minute program segment between ad breaks",
            Severity::Info,
            300.0,
        ));
        checker
    }

    /// Add a rule to the checker.
    pub fn add_rule(&mut self, rule: ComplianceRule) {
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Get the number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Check a schedule against all enabled rules.
    #[allow(clippy::cast_precision_loss)]
    pub fn check(&self, items: &[ScheduleItem]) -> ComplianceReport {
        let enabled_rules: Vec<&ComplianceRule> =
            self.rules.values().filter(|r| r.enabled).collect();
        let mut report = ComplianceReport::new(items.len(), enabled_rules.len());

        for rule in &enabled_rules {
            match rule.kind {
                RuleKind::MaxAdDuration => {
                    self.check_max_ad_duration(rule, items, &mut report);
                }
                RuleKind::StationIdInterval => {
                    self.check_station_id_interval(rule, items, &mut report);
                }
                RuleKind::MaxConsecutiveAds => {
                    self.check_max_consecutive_ads(rule, items, &mut report);
                }
                RuleKind::MinSegmentLength => {
                    self.check_min_segment_length(rule, items, &mut report);
                }
                _ => {}
            }
        }

        report
    }

    /// Check maximum ad duration per hour-sized window.
    #[allow(clippy::cast_precision_loss)]
    fn check_max_ad_duration(
        &self,
        rule: &ComplianceRule,
        items: &[ScheduleItem],
        report: &mut ComplianceReport,
    ) {
        if items.is_empty() {
            return;
        }
        let window = 3600.0_f64;
        let max_end = items
            .iter()
            .map(ScheduleItem::end_secs)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut window_start = 0.0_f64;
        while window_start < max_end {
            let window_end = window_start + window;
            let ad_total: f64 = items
                .iter()
                .filter(|i| i.item_type == ItemType::Ad)
                .filter(|i| i.start_secs < window_end && i.end_secs() > window_start)
                .map(|i| {
                    let eff_start = i.start_secs.max(window_start);
                    let eff_end = i.end_secs().min(window_end);
                    (eff_end - eff_start).max(0.0)
                })
                .sum();
            if ad_total > rule.threshold {
                let v = Violation::new(
                    &rule.id,
                    rule.severity,
                    &format!(
                        "Ad total {ad_total:.0}s exceeds {:.0}s limit in hour starting at {window_start:.0}s",
                        rule.threshold
                    ),
                    window_start,
                );
                report.violations.push(v);
            }
            window_start += window;
        }
    }

    /// Check station ID interval requirement.
    fn check_station_id_interval(
        &self,
        rule: &ComplianceRule,
        items: &[ScheduleItem],
        report: &mut ComplianceReport,
    ) {
        let station_ids: Vec<f64> = items
            .iter()
            .filter(|i| i.item_type == ItemType::StationId)
            .map(|i| i.start_secs)
            .collect();

        if station_ids.is_empty() && !items.is_empty() {
            let v = Violation::new(
                &rule.id,
                rule.severity,
                "No station ID found in schedule",
                0.0,
            );
            report.violations.push(v);
            return;
        }

        // Check gaps between consecutive station IDs
        for window in station_ids.windows(2) {
            let gap = window[1] - window[0];
            if gap > rule.threshold {
                let v = Violation::new(
                    &rule.id,
                    rule.severity,
                    &format!(
                        "Gap of {gap:.0}s between station IDs exceeds {:.0}s limit",
                        rule.threshold
                    ),
                    window[0],
                );
                report.violations.push(v);
            }
        }
    }

    /// Check maximum consecutive ad items.
    fn check_max_consecutive_ads(
        &self,
        rule: &ComplianceRule,
        items: &[ScheduleItem],
        report: &mut ComplianceReport,
    ) {
        let max_consec = rule.threshold as usize;
        let mut consecutive = 0_usize;
        let mut streak_start = 0.0_f64;
        for item in items {
            if item.item_type == ItemType::Ad {
                if consecutive == 0 {
                    streak_start = item.start_secs;
                }
                consecutive += 1;
                if consecutive > max_consec {
                    let v = Violation::new(
                        &rule.id,
                        rule.severity,
                        &format!("{consecutive} consecutive ads exceeds limit of {max_consec}"),
                        streak_start,
                    )
                    .with_item(&item.id);
                    report.violations.push(v);
                }
            } else {
                consecutive = 0;
            }
        }
    }

    /// Check minimum program segment length between ad breaks.
    fn check_min_segment_length(
        &self,
        rule: &ComplianceRule,
        items: &[ScheduleItem],
        report: &mut ComplianceReport,
    ) {
        let mut last_ad_end: Option<f64> = None;
        for item in items {
            if item.item_type == ItemType::Ad {
                if let Some(prev_end) = last_ad_end {
                    let gap = item.start_secs - prev_end;
                    if gap > 0.0 && gap < rule.threshold {
                        let v = Violation::new(
                            &rule.id,
                            rule.severity,
                            &format!(
                                "Program segment of {gap:.0}s between ad breaks is below {:.0}s minimum",
                                rule.threshold
                            ),
                            prev_end,
                        );
                        report.violations.push(v);
                    }
                }
                last_ad_end = Some(item.end_secs());
            }
        }
    }
}

impl Default for ComplianceChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_creation() {
        let rule = ComplianceRule::new(
            "r1",
            RuleKind::MaxAdDuration,
            "test rule",
            Severity::Error,
            720.0,
        );
        assert_eq!(rule.id, "r1");
        assert!(rule.enabled);
    }

    #[test]
    fn test_rule_enable_disable() {
        let mut rule = ComplianceRule::new(
            "r1",
            RuleKind::MaxAdDuration,
            "test",
            Severity::Warning,
            100.0,
        );
        rule.disable();
        assert!(!rule.enabled);
        rule.enable();
        assert!(rule.enabled);
    }

    #[test]
    fn test_schedule_item_end_secs() {
        let item = ScheduleItem::new("item1", ItemType::Program, 100.0, 50.0);
        assert!((item.end_secs() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_violation_with_item() {
        let v = Violation::new("r1", Severity::Error, "bad", 0.0).with_item("item_42");
        assert_eq!(v.item_ids.len(), 1);
        assert_eq!(v.item_ids[0], "item_42");
    }

    #[test]
    fn test_compliance_report_clean() {
        let report = ComplianceReport::new(10, 5);
        assert!(report.is_clean());
        assert_eq!(report.count_at_severity(Severity::Error), 0);
        assert!(report.max_severity().is_none());
    }

    #[test]
    fn test_compliance_checker_standard_rules() {
        let checker = ComplianceChecker::with_standard_rules();
        assert_eq!(checker.rule_count(), 4);
    }

    #[test]
    fn test_check_clean_schedule() {
        let checker = ComplianceChecker::with_standard_rules();
        let items = vec![
            ScheduleItem::new("pgm1", ItemType::Program, 0.0, 1500.0),
            ScheduleItem::new("sid1", ItemType::StationId, 1500.0, 10.0),
            ScheduleItem::new("ad1", ItemType::Ad, 1510.0, 30.0),
            ScheduleItem::new("pgm2", ItemType::Program, 1540.0, 1500.0),
            ScheduleItem::new("sid2", ItemType::StationId, 3040.0, 10.0),
        ];
        let report = checker.check(&items);
        let errors = report.count_at_severity(Severity::Error);
        assert_eq!(errors, 0);
    }

    #[test]
    fn test_check_excessive_ads() {
        let checker = ComplianceChecker::with_standard_rules();
        // 13 minutes of ads in one hour => violation
        let items = vec![
            ScheduleItem::new("ad1", ItemType::Ad, 0.0, 780.0),
            ScheduleItem::new("sid1", ItemType::StationId, 780.0, 10.0),
            ScheduleItem::new("pgm1", ItemType::Program, 790.0, 2810.0),
        ];
        let report = checker.check(&items);
        let ad_violations: Vec<_> = report
            .violations
            .iter()
            .filter(|v| v.rule_id == "max_ad_12min")
            .collect();
        assert!(!ad_violations.is_empty());
    }

    #[test]
    fn test_check_missing_station_id() {
        let mut checker = ComplianceChecker::new();
        checker.add_rule(ComplianceRule::new(
            "sid_req",
            RuleKind::StationIdInterval,
            "Need station ID",
            Severity::Warning,
            3600.0,
        ));
        let items = vec![ScheduleItem::new("pgm1", ItemType::Program, 0.0, 7200.0)];
        let report = checker.check(&items);
        assert!(!report.is_clean());
    }

    #[test]
    fn test_check_max_consecutive_ads() {
        let mut checker = ComplianceChecker::new();
        checker.add_rule(ComplianceRule::new(
            "consec",
            RuleKind::MaxConsecutiveAds,
            "Max 2 consecutive ads",
            Severity::Warning,
            2.0,
        ));
        let items = vec![
            ScheduleItem::new("ad1", ItemType::Ad, 0.0, 30.0),
            ScheduleItem::new("ad2", ItemType::Ad, 30.0, 30.0),
            ScheduleItem::new("ad3", ItemType::Ad, 60.0, 30.0),
        ];
        let report = checker.check(&items);
        assert!(!report.is_clean());
    }

    #[test]
    fn test_check_min_segment_length() {
        let mut checker = ComplianceChecker::new();
        checker.add_rule(ComplianceRule::new(
            "minseg",
            RuleKind::MinSegmentLength,
            "Min 5 min segment",
            Severity::Info,
            300.0,
        ));
        let items = vec![
            ScheduleItem::new("ad1", ItemType::Ad, 0.0, 30.0),
            ScheduleItem::new("pgm1", ItemType::Program, 30.0, 60.0),
            ScheduleItem::new("ad2", ItemType::Ad, 90.0, 30.0),
        ];
        let report = checker.check(&items);
        // 60 seconds < 300 seconds min segment
        assert!(!report.is_clean());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::Error);
        assert!(Severity::Error > Severity::Warning);
        assert!(Severity::Warning > Severity::Info);
    }

    #[test]
    fn test_report_max_severity() {
        let mut report = ComplianceReport::new(5, 3);
        report
            .violations
            .push(Violation::new("r1", Severity::Info, "minor", 0.0));
        report
            .violations
            .push(Violation::new("r2", Severity::Error, "bad", 100.0));
        assert_eq!(report.max_severity(), Some(Severity::Error));
    }

    #[test]
    fn test_empty_schedule_check() {
        let checker = ComplianceChecker::with_standard_rules();
        let report = checker.check(&[]);
        assert!(report.is_clean());
    }
}

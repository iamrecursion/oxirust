#![allow(dead_code)]

//! Playlist validation and conformance checking.
//!
//! Validates playlists against configurable rules such as minimum/maximum
//! duration, forbidden back-to-back repeats, and total runtime limits.

use std::collections::HashMap;
use std::time::Duration;

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationLevel {
    /// Informational note; playlist is valid.
    Info,
    /// Non-blocking warning.
    Warning,
    /// Blocking error; playlist should not air.
    Error,
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    /// Severity level.
    pub level: ValidationLevel,
    /// Zero-based item index (if item-specific), or `None` for global.
    pub item_index: Option<usize>,
    /// Rule that produced this finding.
    pub rule: String,
    /// Human-readable message.
    pub message: String,
}

/// Report from a validation run.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// All findings.
    pub findings: Vec<ValidationFinding>,
}

impl ValidationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self {
            findings: Vec::new(),
        }
    }

    /// Whether the playlist passes (no errors).
    pub fn is_valid(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| f.level == ValidationLevel::Error)
    }

    /// Count of errors.
    pub fn error_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.level == ValidationLevel::Error)
            .count()
    }

    /// Count of warnings.
    pub fn warning_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.level == ValidationLevel::Warning)
            .count()
    }

    /// Count of info findings.
    pub fn info_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.level == ValidationLevel::Info)
            .count()
    }

    /// Push a finding.
    pub fn add(&mut self, finding: ValidationFinding) {
        self.findings.push(finding);
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// A lightweight representation of a playlist item for validation.
#[derive(Debug, Clone)]
pub struct ValidatableItem {
    /// Asset identifier.
    pub asset_id: String,
    /// Duration of the item.
    pub duration: Duration,
    /// Content category tag (e.g. "news", "commercial", "music").
    pub category: String,
}

impl ValidatableItem {
    /// Create a new validatable item.
    pub fn new(
        asset_id: impl Into<String>,
        duration: Duration,
        category: impl Into<String>,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            duration,
            category: category.into(),
        }
    }
}

/// Configurable rules for the validator.
#[derive(Debug, Clone)]
pub struct ValidationRules {
    /// Minimum item duration (0 = no minimum).
    pub min_item_duration: Duration,
    /// Maximum item duration (0 = no maximum).
    pub max_item_duration: Duration,
    /// Maximum total playlist duration (0 = no limit).
    pub max_total_duration: Duration,
    /// Forbid consecutive items with the same asset id.
    pub forbid_consecutive_duplicates: bool,
    /// Maximum allowed consecutive items of the same category (0 = no limit).
    pub max_consecutive_same_category: usize,
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            min_item_duration: Duration::ZERO,
            max_item_duration: Duration::ZERO,
            max_total_duration: Duration::ZERO,
            forbid_consecutive_duplicates: false,
            max_consecutive_same_category: 0,
        }
    }
}

/// Playlist validator.
#[derive(Debug, Clone)]
pub struct PlaylistValidator {
    rules: ValidationRules,
}

impl PlaylistValidator {
    /// Create a new validator with the given rules.
    pub fn new(rules: ValidationRules) -> Self {
        Self { rules }
    }

    /// Validate a list of items.
    pub fn validate(&self, items: &[ValidatableItem]) -> ValidationReport {
        let mut report = ValidationReport::new();

        if items.is_empty() {
            report.add(ValidationFinding {
                level: ValidationLevel::Warning,
                item_index: None,
                rule: "non_empty".into(),
                message: "Playlist is empty".into(),
            });
            return report;
        }

        let mut total = Duration::ZERO;

        for (i, item) in items.iter().enumerate() {
            // Min duration check
            if self.rules.min_item_duration > Duration::ZERO
                && item.duration < self.rules.min_item_duration
            {
                report.add(ValidationFinding {
                    level: ValidationLevel::Error,
                    item_index: Some(i),
                    rule: "min_duration".into(),
                    message: format!(
                        "Item '{}' duration {:?} below minimum {:?}",
                        item.asset_id, item.duration, self.rules.min_item_duration
                    ),
                });
            }

            // Max duration check
            if self.rules.max_item_duration > Duration::ZERO
                && item.duration > self.rules.max_item_duration
            {
                report.add(ValidationFinding {
                    level: ValidationLevel::Error,
                    item_index: Some(i),
                    rule: "max_duration".into(),
                    message: format!(
                        "Item '{}' duration {:?} exceeds maximum {:?}",
                        item.asset_id, item.duration, self.rules.max_item_duration
                    ),
                });
            }

            total += item.duration;
        }

        // Total duration
        if self.rules.max_total_duration > Duration::ZERO && total > self.rules.max_total_duration {
            report.add(ValidationFinding {
                level: ValidationLevel::Error,
                item_index: None,
                rule: "max_total_duration".into(),
                message: format!(
                    "Total duration {:?} exceeds maximum {:?}",
                    total, self.rules.max_total_duration
                ),
            });
        }

        // Consecutive duplicates
        if self.rules.forbid_consecutive_duplicates {
            for i in 1..items.len() {
                if items[i].asset_id == items[i - 1].asset_id {
                    report.add(ValidationFinding {
                        level: ValidationLevel::Error,
                        item_index: Some(i),
                        rule: "no_consecutive_duplicates".into(),
                        message: format!(
                            "Consecutive duplicate asset '{}' at index {}",
                            items[i].asset_id, i
                        ),
                    });
                }
            }
        }

        // Consecutive same category
        if self.rules.max_consecutive_same_category > 0 {
            let mut run_len = 1usize;
            for i in 1..items.len() {
                if items[i].category == items[i - 1].category {
                    run_len += 1;
                    if run_len > self.rules.max_consecutive_same_category {
                        report.add(ValidationFinding {
                            level: ValidationLevel::Warning,
                            item_index: Some(i),
                            rule: "max_consecutive_category".into(),
                            message: format!(
                                "{} consecutive '{}' items ending at index {}",
                                run_len, items[i].category, i
                            ),
                        });
                    }
                } else {
                    run_len = 1;
                }
            }
        }

        // Category distribution info
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for item in items {
            *counts.entry(item.category.as_str()).or_insert(0) += 1;
        }
        for (cat, count) in &counts {
            report.add(ValidationFinding {
                level: ValidationLevel::Info,
                item_index: None,
                rule: "category_stats".into(),
                message: format!("Category '{cat}': {count} items"),
            });
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, secs: u64, cat: &str) -> ValidatableItem {
        ValidatableItem::new(id, Duration::from_secs(secs), cat)
    }

    #[test]
    fn test_empty_playlist_warning() {
        let v = PlaylistValidator::new(ValidationRules::default());
        let r = v.validate(&[]);
        assert_eq!(r.warning_count(), 1);
        assert!(r.is_valid());
    }

    #[test]
    fn test_valid_single_item() {
        let v = PlaylistValidator::new(ValidationRules::default());
        let r = v.validate(&[item("a", 30, "news")]);
        assert!(r.is_valid());
    }

    #[test]
    fn test_min_duration_violation() {
        let rules = ValidationRules {
            min_item_duration: Duration::from_secs(10),
            ..Default::default()
        };
        let v = PlaylistValidator::new(rules);
        let r = v.validate(&[item("short", 5, "news")]);
        assert!(!r.is_valid());
        assert_eq!(r.error_count(), 1);
    }

    #[test]
    fn test_max_duration_violation() {
        let rules = ValidationRules {
            max_item_duration: Duration::from_secs(60),
            ..Default::default()
        };
        let v = PlaylistValidator::new(rules);
        let r = v.validate(&[item("long", 120, "movie")]);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_total_duration_violation() {
        let rules = ValidationRules {
            max_total_duration: Duration::from_secs(100),
            ..Default::default()
        };
        let v = PlaylistValidator::new(rules);
        let items = vec![item("a", 60, "news"), item("b", 60, "news")];
        let r = v.validate(&items);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_consecutive_duplicate_forbidden() {
        let rules = ValidationRules {
            forbid_consecutive_duplicates: true,
            ..Default::default()
        };
        let v = PlaylistValidator::new(rules);
        let items = vec![item("x", 30, "music"), item("x", 30, "music")];
        let r = v.validate(&items);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_consecutive_duplicate_allowed_by_default() {
        let v = PlaylistValidator::new(ValidationRules::default());
        let items = vec![item("x", 30, "music"), item("x", 30, "music")];
        let r = v.validate(&items);
        assert!(r.is_valid());
    }

    #[test]
    fn test_max_consecutive_category() {
        let rules = ValidationRules {
            max_consecutive_same_category: 2,
            ..Default::default()
        };
        let v = PlaylistValidator::new(rules);
        let items = vec![
            item("a", 30, "ad"),
            item("b", 30, "ad"),
            item("c", 30, "ad"), // 3rd consecutive ad -> warning
        ];
        let r = v.validate(&items);
        assert!(r.warning_count() >= 1);
    }

    #[test]
    fn test_category_stats_info() {
        let v = PlaylistValidator::new(ValidationRules::default());
        let items = vec![item("a", 10, "news"), item("b", 20, "music")];
        let r = v.validate(&items);
        assert!(r.info_count() >= 2);
    }

    #[test]
    fn test_validation_level_ordering() {
        assert!(ValidationLevel::Info < ValidationLevel::Warning);
        assert!(ValidationLevel::Warning < ValidationLevel::Error);
    }

    #[test]
    fn test_report_default() {
        let r = ValidationReport::default();
        assert!(r.is_valid());
        assert_eq!(r.error_count(), 0);
    }

    #[test]
    fn test_multiple_violations() {
        let rules = ValidationRules {
            min_item_duration: Duration::from_secs(10),
            max_item_duration: Duration::from_secs(100),
            ..Default::default()
        };
        let v = PlaylistValidator::new(rules);
        let items = vec![item("short", 5, "a"), item("long", 200, "b")];
        let r = v.validate(&items);
        assert_eq!(r.error_count(), 2);
    }

    #[test]
    fn test_non_consecutive_duplicate_ok() {
        let rules = ValidationRules {
            forbid_consecutive_duplicates: true,
            ..Default::default()
        };
        let v = PlaylistValidator::new(rules);
        let items = vec![
            item("x", 30, "music"),
            item("y", 30, "news"),
            item("x", 30, "music"),
        ];
        let r = v.validate(&items);
        assert!(r.is_valid());
    }

    #[test]
    fn test_category_run_resets() {
        let rules = ValidationRules {
            max_consecutive_same_category: 2,
            ..Default::default()
        };
        let v = PlaylistValidator::new(rules);
        let items = vec![
            item("a", 10, "ad"),
            item("b", 10, "ad"),
            item("c", 10, "news"), // breaks run
            item("d", 10, "ad"),
            item("e", 10, "ad"),
        ];
        let r = v.validate(&items);
        // No warning because each run is exactly 2, which is within limit.
        let cat_warnings = r
            .findings
            .iter()
            .filter(|f| f.rule == "max_consecutive_category")
            .count();
        assert_eq!(cat_warnings, 0);
    }

    #[test]
    fn test_finding_fields() {
        let f = ValidationFinding {
            level: ValidationLevel::Error,
            item_index: Some(3),
            rule: "test_rule".into(),
            message: "test message".into(),
        };
        assert_eq!(f.level, ValidationLevel::Error);
        assert_eq!(f.item_index, Some(3));
        assert_eq!(f.rule, "test_rule");
    }
}

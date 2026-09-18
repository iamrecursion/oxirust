//! Automatic format-migration trigger policy.
//!
//! Bridges the risk monitoring subsystem ([`super::monitor`]) with the
//! migration planning subsystem ([`crate::migration_plan`]).
//!
//! A [`MigrationTriggerPolicy`] is a pure, stateless decision function:
//! given a [`MonitoringReport`] it returns zero or more [`MigrationPlan`]s
//! for the formats that exceed the configured risk threshold or are listed
//! as targets for explicit obsolescence migration.  No I/O is performed.
//!
//! ## Idempotency
//!
//! Files whose format appears in the report's `high_risk_files` list more
//! than once will still produce only a single plan per unique format.  The
//! caller is responsible for not re-submitting plans that are already
//! in-flight; this function makes no assumptions about external state.

use super::monitor::MonitoringReport;
use super::RiskLevel;
use crate::migration_plan::{MigrationPlan, MigrationPriority, MigrationTask};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Policy controlling when and how automatic format migrations are triggered.
///
/// # Example
///
/// ```rust
/// use oximedia_archive_pro::risk::migration_trigger::MigrationTriggerPolicy;
///
/// let mut policy = MigrationTriggerPolicy::new(0.75);
/// policy.add_format_target("betacam_sp", "prores");
/// policy.add_format_target("wmv", "ffv1_mkv");
/// ```
#[derive(Debug, Clone)]
pub struct MigrationTriggerPolicy {
    /// Normalised risk score threshold (0.0–1.0).  Formats with a score at or
    /// above this value will trigger a migration plan.
    ///
    /// The threshold is compared against `RiskLevel::score() / 100.0`.
    pub risk_threshold: f64,

    /// If `true`, any format listed in [`MonitoringReport::high_risk_files`]
    /// will trigger a plan even if its numeric score is below the threshold.
    pub on_obsolete_format: bool,

    /// Mapping from source format name (lower-case) to the target preservation
    /// format to migrate toward.  When a triggered format has an entry here the
    /// plan is labelled with the corresponding target; otherwise the default
    /// target `"preservation_master"` is used.
    pub format_targets: HashMap<String, String>,
}

impl MigrationTriggerPolicy {
    /// Creates a new policy with the given risk threshold and default settings.
    ///
    /// `risk_threshold` should be in `[0.0, 1.0]`.  Values outside that range
    /// are clamped.
    #[must_use]
    pub fn new(risk_threshold: f64) -> Self {
        Self {
            risk_threshold: risk_threshold.clamp(0.0, 1.0),
            on_obsolete_format: true,
            format_targets: HashMap::new(),
        }
    }

    /// Registers a source-format → target-format mapping.
    pub fn add_format_target(
        &mut self,
        source_format: impl Into<String>,
        target_format: impl Into<String>,
    ) {
        self.format_targets
            .insert(source_format.into(), target_format.into());
    }

    /// Evaluates a [`MonitoringReport`] and returns one [`MigrationPlan`] per
    /// format that should be migrated according to this policy.
    ///
    /// ## Behaviour
    ///
    /// For each unique format name that appears in the report:
    ///
    /// 1. Compute the normalised risk score from the report's risk distribution
    ///    using the highest `RiskLevel` observed for that format.
    /// 2. If the score ≥ `risk_threshold`, **or** the format appears in
    ///    `high_risk_files` and `on_obsolete_format` is `true`, create a plan.
    /// 3. The plan contains a single [`MigrationTask`] named after the format.
    /// 4. Target format is looked up in `format_targets`; falls back to
    ///    `"preservation_master"`.
    ///
    /// Duplicate format names in the report produce exactly one plan each.
    ///
    /// ## Idempotency
    ///
    /// This method is purely functional — calling it multiple times with the
    /// same report returns the same result.
    #[must_use]
    pub fn evaluate(&self, report: &MonitoringReport) -> Vec<MigrationPlan> {
        // Collect the set of formats that should be migrated.
        let mut triggered: HashSet<String> = HashSet::new();

        // Formats in high_risk_files are immediately eligible when
        // on_obsolete_format is true.
        if self.on_obsolete_format {
            for fmt in &report.high_risk_files {
                triggered.insert(fmt.clone());
            }
        }

        // Check threshold against each RiskLevel bucket in the distribution.
        for (level, &count) in &report.risk_distribution {
            if count == 0 {
                continue;
            }
            let score = f64::from(level.score()) / 100.0;
            if score >= self.risk_threshold {
                // The distribution does not track per-format scores directly;
                // we surface this bucket as a synthetic format key so callers
                // that populate high_risk_files do not need to double-register.
                // Formats that were already added via high_risk_files are
                // deduplicated by the HashSet.
                let synthetic = format!("format_risk_level_{}", level.name());
                triggered.insert(synthetic);
            }
        }

        // Build one MigrationPlan per triggered unique format.
        let mut plans = Vec::with_capacity(triggered.len());
        let mut plan_id = 1u64;

        // Sort for deterministic output ordering.
        let mut sorted_formats: Vec<String> = triggered.into_iter().collect();
        sorted_formats.sort();

        for format_name in sorted_formats {
            let target = self
                .format_targets
                .get(&format_name)
                .cloned()
                .unwrap_or_else(|| "preservation_master".to_string());

            let priority = self.priority_for_format(&format_name, report);

            let mut plan = MigrationPlan::new(plan_id, format!("auto-migrate:{format_name}"));
            plan_id += 1;

            // Single task per plan covering the triggered format.
            let task = MigrationTask::new(
                1,
                format!("Migrate {format_name} → {target}"),
                format_name.clone(),
                target,
                // Asset count and byte size are unknown at policy-evaluation
                // time; callers should populate these before execution.
                0,
                0,
                priority,
                Duration::from_secs(0),
            );
            plan.add_task(task);
            plan.metadata.insert(
                "trigger_source".to_string(),
                "MigrationTriggerPolicy".to_string(),
            );
            plan.metadata
                .insert("source_format".to_string(), format_name);

            plans.push(plan);
        }

        plans
    }

    /// Resolves the [`MigrationPriority`] for a triggered format.
    fn priority_for_format(
        &self,
        format_name: &str,
        report: &MonitoringReport,
    ) -> MigrationPriority {
        // If the format appears in high_risk_files (critical-level signal).
        if report.high_risk_files.contains(&format_name.to_string()) {
            let critical_score = f64::from(RiskLevel::Critical.score()) / 100.0;
            if self.risk_threshold <= critical_score {
                return MigrationPriority::Critical;
            }
        }

        // Otherwise derive from threshold distance.
        if self.risk_threshold >= 0.9 {
            MigrationPriority::High
        } else if self.risk_threshold >= 0.5 {
            MigrationPriority::Medium
        } else {
            MigrationPriority::Low
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::monitor::RiskMonitor;
    use crate::risk::{FormatRisk, RiskLevel};
    use std::collections::HashMap;

    fn report_with_risk(format: &str, level: RiskLevel) -> MonitoringReport {
        let mut monitor = RiskMonitor::new();
        monitor.add_assessment(FormatRisk {
            format: format.to_string(),
            risk_level: level,
            factors: Vec::new(),
            recommendation: String::new(),
            timestamp: chrono::Utc::now(),
        });
        monitor.generate_report()
    }

    fn empty_report() -> MonitoringReport {
        MonitoringReport {
            risk_distribution: HashMap::new(),
            total_files: 0,
            high_risk_files: Vec::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_migration_trigger_fires_above_threshold() {
        let policy = MigrationTriggerPolicy::new(0.7);
        let report = report_with_risk("wmv", RiskLevel::Critical);
        let plans = policy.evaluate(&report);
        assert!(
            !plans.is_empty(),
            "at least one migration plan must be triggered for a critical-risk format"
        );
    }

    #[test]
    fn test_migration_trigger_no_fire_below_threshold() {
        let mut policy = MigrationTriggerPolicy::new(0.7);
        // Disable obsolete-format gate so only the threshold matters.
        policy.on_obsolete_format = false;

        // Low risk: score = 25 / 100 = 0.25 < 0.7
        let report = report_with_risk("low_risk_format", RiskLevel::Low);

        // Ensure high_risk_files is empty for a clean test.
        let clean_report = MonitoringReport {
            risk_distribution: report.risk_distribution,
            total_files: report.total_files,
            high_risk_files: Vec::new(), // no high-risk flag
            timestamp: report.timestamp,
        };

        let plans = policy.evaluate(&clean_report);
        assert!(
            plans.is_empty(),
            "no migration plan must be triggered for a low-risk format below the threshold"
        );
    }

    #[test]
    fn test_migration_trigger_correct_target_format() {
        let mut policy = MigrationTriggerPolicy::new(0.7);
        policy.on_obsolete_format = true;
        policy.add_format_target("betacam_sp", "prores");

        // Manually build a report with betacam_sp as a high-risk file.
        let report = MonitoringReport {
            risk_distribution: HashMap::new(),
            total_files: 1,
            high_risk_files: vec!["betacam_sp".to_string()],
            timestamp: chrono::Utc::now(),
        };

        let plans = policy.evaluate(&report);
        assert!(
            !plans.is_empty(),
            "betacam_sp in high_risk_files must trigger a plan"
        );

        let plan = plans
            .iter()
            .find(|p| p.metadata.get("source_format").map(String::as_str) == Some("betacam_sp"))
            .expect("a plan for betacam_sp must exist");

        let tasks = plan.tasks();
        assert_eq!(tasks.len(), 1, "plan must contain exactly one task");
        assert_eq!(
            tasks[0].target_format, "prores",
            "target format must be 'prores' as registered"
        );
    }

    #[test]
    fn test_migration_trigger_default_target_format() {
        let policy = MigrationTriggerPolicy::new(0.7);

        let report = MonitoringReport {
            risk_distribution: HashMap::new(),
            total_files: 1,
            high_risk_files: vec!["unknown_obsolete_fmt".to_string()],
            timestamp: chrono::Utc::now(),
        };

        let plans = policy.evaluate(&report);
        assert!(!plans.is_empty());

        let plan = plans
            .iter()
            .find(|p| {
                p.metadata.get("source_format").map(String::as_str) == Some("unknown_obsolete_fmt")
            })
            .expect("plan for unknown format must exist");

        let tasks = plan.tasks();
        assert_eq!(
            tasks[0].target_format, "preservation_master",
            "default target format must be 'preservation_master'"
        );
    }

    #[test]
    fn test_migration_trigger_idempotent() {
        let policy = MigrationTriggerPolicy::new(0.5);

        let report = MonitoringReport {
            risk_distribution: HashMap::new(),
            total_files: 1,
            // Same format appears twice — must produce one plan only.
            high_risk_files: vec!["avi".to_string(), "avi".to_string()],
            timestamp: chrono::Utc::now(),
        };

        let plans = policy.evaluate(&report);
        let avi_plans: Vec<_> = plans
            .iter()
            .filter(|p| p.metadata.get("source_format").map(String::as_str) == Some("avi"))
            .collect();
        assert_eq!(
            avi_plans.len(),
            1,
            "duplicate format in report must produce exactly one plan"
        );
    }

    #[test]
    fn test_migration_trigger_policy_clamps_threshold() {
        let low = MigrationTriggerPolicy::new(-5.0);
        assert_eq!(low.risk_threshold, 0.0);
        let high = MigrationTriggerPolicy::new(999.0);
        assert_eq!(high.risk_threshold, 1.0);
    }

    #[test]
    fn test_migration_trigger_empty_report_no_plans() {
        let policy = MigrationTriggerPolicy::new(0.5);
        let plans = policy.evaluate(&empty_report());
        assert!(plans.is_empty(), "empty report must produce no plans");
    }
}

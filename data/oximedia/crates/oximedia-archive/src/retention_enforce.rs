//! Retention schedule enforcement: automatic deletion/archival when retention
//! period expires.
//!
//! Provides an enforcement engine that evaluates a [`RetentionSchedule`],
//! identifies expired assets, and produces an action plan (delete, archive,
//! extend, or hold). Supports dry-run mode, grace periods, tiered archival
//! policies, and audit trail generation.

use crate::retention_schedule::{RetentionClass, RetentionEntry, RetentionSchedule};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Enforcement actions
// ---------------------------------------------------------------------------

/// Action to take on an expired retention entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnforcementAction {
    /// Delete the asset permanently.
    Delete,
    /// Move the asset to cold/archive storage tier.
    Archive {
        /// Target storage tier name.
        target_tier: String,
    },
    /// Extend the retention period by the given number of milliseconds.
    Extend {
        /// Additional milliseconds to add.
        extension_ms: u64,
    },
    /// No action (asset under legal hold or permanent).
    NoAction {
        /// Reason no action is taken.
        reason: String,
    },
}

impl std::fmt::Display for EnforcementAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Delete => write!(f, "DELETE"),
            Self::Archive { target_tier } => write!(f, "ARCHIVE -> {target_tier}"),
            Self::Extend { extension_ms } => {
                let days = extension_ms / (24 * 3_600_000);
                write!(f, "EXTEND by {days} days")
            }
            Self::NoAction { reason } => write!(f, "NO_ACTION: {reason}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Enforcement policy
// ---------------------------------------------------------------------------

/// Policy that determines what action to take for each retention class
/// when retention expires.
#[derive(Debug, Clone)]
pub struct EnforcementPolicy {
    /// Action for expired Temporary assets.
    pub temporary_action: EnforcementAction,
    /// Action for expired Standard assets.
    pub standard_action: EnforcementAction,
    /// Action for expired LongTerm assets.
    pub long_term_action: EnforcementAction,
    /// Grace period in milliseconds before enforcement kicks in after expiry.
    pub grace_period_ms: u64,
    /// Whether to generate audit trail entries for each action.
    pub audit_enabled: bool,
    /// Maximum number of assets to process in a single enforcement run.
    pub batch_limit: usize,
}

impl Default for EnforcementPolicy {
    fn default() -> Self {
        Self {
            temporary_action: EnforcementAction::Delete,
            standard_action: EnforcementAction::Archive {
                target_tier: "cold".to_string(),
            },
            long_term_action: EnforcementAction::Archive {
                target_tier: "glacier".to_string(),
            },
            grace_period_ms: 30 * 24 * 3_600_000, // 30-day grace period
            audit_enabled: true,
            batch_limit: 10_000,
        }
    }
}

impl EnforcementPolicy {
    /// Get the action for a given retention class.
    #[must_use]
    pub fn action_for_class(&self, class: RetentionClass) -> EnforcementAction {
        match class {
            RetentionClass::Temporary => self.temporary_action.clone(),
            RetentionClass::Standard => self.standard_action.clone(),
            RetentionClass::LongTerm => self.long_term_action.clone(),
            RetentionClass::Permanent => EnforcementAction::NoAction {
                reason: "permanent retention".to_string(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Enforcement plan item
// ---------------------------------------------------------------------------

/// A single item in the enforcement plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementPlanItem {
    /// Asset identifier.
    pub asset_id: String,
    /// Retention class of the asset.
    pub retention_class: String,
    /// Time asset was ingested (Unix ms).
    pub ingested_at_ms: u64,
    /// Expiry time (Unix ms), if applicable.
    pub expires_at_ms: Option<u64>,
    /// How long past expiry (in ms). 0 if not expired.
    pub overdue_ms: u64,
    /// Planned action.
    pub action: EnforcementAction,
    /// Whether a legal hold prevented action.
    pub legal_hold_active: bool,
    /// Whether this is within the grace period.
    pub within_grace_period: bool,
}

// ---------------------------------------------------------------------------
// Enforcement plan / report
// ---------------------------------------------------------------------------

/// Complete enforcement plan generated by the enforcement engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementPlan {
    /// Timestamp when this plan was generated (Unix ms).
    pub generated_at_ms: u64,
    /// Whether this is a dry-run (no actions will be executed).
    pub dry_run: bool,
    /// Individual plan items.
    pub items: Vec<EnforcementPlanItem>,
    /// Total number of assets evaluated.
    pub total_evaluated: usize,
    /// Number of assets eligible for action.
    pub actionable_count: usize,
    /// Number of assets skipped due to legal hold.
    pub held_count: usize,
    /// Number of assets within grace period (deferred).
    pub grace_period_count: usize,
    /// Summary of actions by type.
    pub action_summary: ActionSummary,
}

/// Summary counts of planned actions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionSummary {
    /// Number of delete actions.
    pub delete_count: usize,
    /// Number of archive actions.
    pub archive_count: usize,
    /// Number of extend actions.
    pub extend_count: usize,
    /// Number of no-action items.
    pub no_action_count: usize,
}

impl EnforcementPlan {
    /// Whether the plan contains any actionable items.
    #[must_use]
    pub fn has_actions(&self) -> bool {
        self.actionable_count > 0
    }

    /// Get all items with a Delete action.
    #[must_use]
    pub fn deletions(&self) -> Vec<&EnforcementPlanItem> {
        self.items
            .iter()
            .filter(|i| matches!(i.action, EnforcementAction::Delete))
            .collect()
    }

    /// Get all items with an Archive action.
    #[must_use]
    pub fn archives(&self) -> Vec<&EnforcementPlanItem> {
        self.items
            .iter()
            .filter(|i| matches!(i.action, EnforcementAction::Archive { .. }))
            .collect()
    }

    /// Format the plan as a human-readable summary.
    #[must_use]
    pub fn to_summary_string(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Retention Enforcement Plan ===\n");
        out.push_str(&format!(
            "Mode: {}\n",
            if self.dry_run { "DRY RUN" } else { "LIVE" }
        ));
        out.push_str(&format!("Total evaluated:  {}\n", self.total_evaluated));
        out.push_str(&format!("Actionable:       {}\n", self.actionable_count));
        out.push_str(&format!("Legal holds:      {}\n", self.held_count));
        out.push_str(&format!("Grace period:     {}\n", self.grace_period_count));
        out.push_str(&format!(
            "  Deletes:  {}\n",
            self.action_summary.delete_count
        ));
        out.push_str(&format!(
            "  Archives: {}\n",
            self.action_summary.archive_count
        ));
        out.push_str(&format!(
            "  Extends:  {}\n",
            self.action_summary.extend_count
        ));
        out
    }
}

// ---------------------------------------------------------------------------
// Audit trail entry
// ---------------------------------------------------------------------------

/// Audit trail entry for a retention enforcement action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementAuditEntry {
    /// Unique audit event ID.
    pub event_id: String,
    /// Timestamp of the action (Unix ms).
    pub timestamp_ms: u64,
    /// Asset ID.
    pub asset_id: String,
    /// Action taken.
    pub action: EnforcementAction,
    /// Whether this was a dry run.
    pub dry_run: bool,
    /// Outcome description.
    pub outcome: String,
}

// ---------------------------------------------------------------------------
// Enforcement engine
// ---------------------------------------------------------------------------

/// Engine that evaluates retention schedules and produces enforcement plans.
pub struct EnforcementEngine {
    policy: EnforcementPolicy,
}

impl EnforcementEngine {
    /// Create an enforcement engine with the given policy.
    #[must_use]
    pub fn new(policy: EnforcementPolicy) -> Self {
        Self { policy }
    }

    /// Create with default policy.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(EnforcementPolicy::default())
    }

    /// Get the current policy.
    #[must_use]
    pub fn policy(&self) -> &EnforcementPolicy {
        &self.policy
    }

    /// Evaluate the retention schedule and produce an enforcement plan.
    ///
    /// `now_ms` is the current Unix timestamp in milliseconds.
    /// If `dry_run` is true, no actions will actually be executed.
    #[must_use]
    pub fn evaluate(
        &self,
        schedule: &RetentionSchedule,
        now_ms: u64,
        dry_run: bool,
    ) -> EnforcementPlan {
        let mut items = Vec::new();
        let mut actionable_count = 0usize;
        let mut held_count = 0usize;
        let mut grace_period_count = 0usize;
        let mut summary = ActionSummary::default();

        let total_evaluated = schedule.len();
        let eligible = schedule.eligible_for_deletion(now_ms);

        // Also check entries that are past expiry but within grace period
        let all_entries = self.collect_all_candidates(schedule, now_ms);

        for (entry, overdue_ms) in all_entries.iter().take(self.policy.batch_limit) {
            let within_grace = *overdue_ms > 0 && *overdue_ms <= self.policy.grace_period_ms;

            if entry.legal_hold {
                held_count += 1;
                items.push(EnforcementPlanItem {
                    asset_id: entry.asset_id.clone(),
                    retention_class: entry.class.label().to_string(),
                    ingested_at_ms: entry.ingested_at_ms,
                    expires_at_ms: entry.expires_at_ms,
                    overdue_ms: *overdue_ms,
                    action: EnforcementAction::NoAction {
                        reason: "legal hold active".to_string(),
                    },
                    legal_hold_active: true,
                    within_grace_period: within_grace,
                });
                summary.no_action_count += 1;
                continue;
            }

            if entry.class == RetentionClass::Permanent {
                items.push(EnforcementPlanItem {
                    asset_id: entry.asset_id.clone(),
                    retention_class: entry.class.label().to_string(),
                    ingested_at_ms: entry.ingested_at_ms,
                    expires_at_ms: entry.expires_at_ms,
                    overdue_ms: 0,
                    action: EnforcementAction::NoAction {
                        reason: "permanent retention".to_string(),
                    },
                    legal_hold_active: false,
                    within_grace_period: false,
                });
                summary.no_action_count += 1;
                continue;
            }

            if within_grace {
                grace_period_count += 1;
                items.push(EnforcementPlanItem {
                    asset_id: entry.asset_id.clone(),
                    retention_class: entry.class.label().to_string(),
                    ingested_at_ms: entry.ingested_at_ms,
                    expires_at_ms: entry.expires_at_ms,
                    overdue_ms: *overdue_ms,
                    action: EnforcementAction::NoAction {
                        reason: "within grace period".to_string(),
                    },
                    legal_hold_active: false,
                    within_grace_period: true,
                });
                summary.no_action_count += 1;
                continue;
            }

            // Asset is past grace period: apply policy action
            let action = self.policy.action_for_class(entry.class);
            match &action {
                EnforcementAction::Delete => summary.delete_count += 1,
                EnforcementAction::Archive { .. } => summary.archive_count += 1,
                EnforcementAction::Extend { .. } => summary.extend_count += 1,
                EnforcementAction::NoAction { .. } => summary.no_action_count += 1,
            }
            actionable_count += 1;

            items.push(EnforcementPlanItem {
                asset_id: entry.asset_id.clone(),
                retention_class: entry.class.label().to_string(),
                ingested_at_ms: entry.ingested_at_ms,
                expires_at_ms: entry.expires_at_ms,
                overdue_ms: *overdue_ms,
                action,
                legal_hold_active: false,
                within_grace_period: false,
            });
        }

        // Suppress unused variable warning
        let _ = eligible;

        EnforcementPlan {
            generated_at_ms: now_ms,
            dry_run,
            items,
            total_evaluated,
            actionable_count,
            held_count,
            grace_period_count,
            action_summary: summary,
        }
    }

    /// Generate audit trail entries from an enforcement plan.
    #[must_use]
    pub fn generate_audit_trail(&self, plan: &EnforcementPlan) -> Vec<EnforcementAuditEntry> {
        plan.items
            .iter()
            .enumerate()
            .map(|(i, item)| EnforcementAuditEntry {
                event_id: format!("enf-{}-{i:04}", plan.generated_at_ms),
                timestamp_ms: plan.generated_at_ms,
                asset_id: item.asset_id.clone(),
                action: item.action.clone(),
                dry_run: plan.dry_run,
                outcome: item.action.to_string(),
            })
            .collect()
    }

    /// Collect all entries that are overdue (past expiry), returning each
    /// with the number of milliseconds past expiry.
    fn collect_all_candidates<'a>(
        &self,
        schedule: &'a RetentionSchedule,
        now_ms: u64,
    ) -> Vec<(&'a RetentionEntry, u64)> {
        let mut candidates = Vec::new();

        // We need to check all entries for overdue status
        for i in 0..schedule.len() {
            if let Some(entry) = schedule.lookup_by_index(i) {
                let overdue = self.compute_overdue_ms(entry, now_ms);
                if overdue > 0 || entry.class == RetentionClass::Permanent {
                    candidates.push((entry, overdue));
                }
            }
        }

        // Sort by overdue ms descending (most overdue first)
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        candidates
    }

    /// Compute how many milliseconds past expiry an entry is.
    fn compute_overdue_ms(&self, entry: &RetentionEntry, now_ms: u64) -> u64 {
        if entry.class == RetentionClass::Permanent {
            return 0;
        }
        let expiry_ms = if let Some(exp) = entry.expires_at_ms {
            exp
        } else if let Some(years) = entry.class.default_years() {
            let duration_ms = u64::from(years) * 365 * 24 * 3_600_000;
            entry.ingested_at_ms.saturating_add(duration_ms)
        } else {
            return 0;
        };

        now_ms.saturating_sub(expiry_ms)
    }
}

// We need a way to access entries by index in RetentionSchedule.
// Let's add a helper trait.
impl RetentionSchedule {
    /// Look up a retention entry by index.
    #[must_use]
    pub fn lookup_by_index(&self, index: usize) -> Option<&RetentionEntry> {
        self.all_entries().get(index)
    }

    /// Get all entries as a slice.
    #[must_use]
    pub fn all_entries(&self) -> &[RetentionEntry] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS_PER_YEAR: u64 = 365 * 24 * 3_600_000;
    const MS_PER_DAY: u64 = 24 * 3_600_000;

    fn make_schedule() -> RetentionSchedule {
        let mut sched = RetentionSchedule::new();
        // Temporary asset, expired 2 years ago
        sched.add(RetentionEntry::new(
            "temp-001",
            RetentionClass::Temporary,
            0,
            Some(MS_PER_YEAR),
            false,
        ));
        // Standard asset, expired
        sched.add(RetentionEntry::new(
            "std-001",
            RetentionClass::Standard,
            0,
            Some(2 * MS_PER_YEAR),
            false,
        ));
        // Long-term asset, not expired
        sched.add(RetentionEntry::new(
            "long-001",
            RetentionClass::LongTerm,
            0,
            Some(20 * MS_PER_YEAR),
            false,
        ));
        // Permanent asset
        sched.add(RetentionEntry::new(
            "perm-001",
            RetentionClass::Permanent,
            0,
            None,
            false,
        ));
        // Under legal hold (expired but held)
        sched.add(RetentionEntry::new(
            "held-001",
            RetentionClass::Temporary,
            0,
            Some(MS_PER_YEAR),
            true,
        ));
        sched
    }

    #[test]
    fn test_enforcement_basic() {
        let schedule = make_schedule();
        let engine = EnforcementEngine::with_defaults();
        let now = 10 * MS_PER_YEAR;
        let plan = engine.evaluate(&schedule, now, true);

        assert!(plan.dry_run);
        assert_eq!(plan.total_evaluated, 5);
        assert!(plan.has_actions());
    }

    #[test]
    fn test_enforcement_delete_temporary() {
        let mut sched = RetentionSchedule::new();
        sched.add(RetentionEntry::new(
            "temp-001",
            RetentionClass::Temporary,
            0,
            Some(100),
            false,
        ));

        let policy = EnforcementPolicy {
            grace_period_ms: 0, // No grace period
            ..Default::default()
        };
        let engine = EnforcementEngine::new(policy);
        let plan = engine.evaluate(&sched, 200, false);

        assert_eq!(plan.actionable_count, 1);
        assert_eq!(plan.action_summary.delete_count, 1);
        let deletions = plan.deletions();
        assert_eq!(deletions.len(), 1);
        assert_eq!(deletions[0].asset_id, "temp-001");
    }

    #[test]
    fn test_enforcement_archive_standard() {
        let mut sched = RetentionSchedule::new();
        sched.add(RetentionEntry::new(
            "std-001",
            RetentionClass::Standard,
            0,
            Some(100),
            false,
        ));

        let policy = EnforcementPolicy {
            grace_period_ms: 0,
            ..Default::default()
        };
        let engine = EnforcementEngine::new(policy);
        let plan = engine.evaluate(&sched, 200, false);

        assert_eq!(plan.action_summary.archive_count, 1);
        let archives = plan.archives();
        assert_eq!(archives.len(), 1);
    }

    #[test]
    fn test_enforcement_legal_hold() {
        let mut sched = RetentionSchedule::new();
        sched.add(RetentionEntry::new(
            "held-001",
            RetentionClass::Temporary,
            0,
            Some(100),
            true,
        ));

        let policy = EnforcementPolicy {
            grace_period_ms: 0,
            ..Default::default()
        };
        let engine = EnforcementEngine::new(policy);
        let plan = engine.evaluate(&sched, 200, false);

        assert_eq!(plan.held_count, 1);
        assert_eq!(plan.actionable_count, 0);
        assert!(plan.items[0].legal_hold_active);
    }

    #[test]
    fn test_enforcement_permanent_no_action() {
        let mut sched = RetentionSchedule::new();
        sched.add(RetentionEntry::new(
            "perm-001",
            RetentionClass::Permanent,
            0,
            None,
            false,
        ));

        let engine = EnforcementEngine::with_defaults();
        let plan = engine.evaluate(&sched, u64::MAX / 2, false);

        assert_eq!(plan.actionable_count, 0);
        assert_eq!(plan.action_summary.no_action_count, 1);
    }

    #[test]
    fn test_enforcement_grace_period() {
        let mut sched = RetentionSchedule::new();
        sched.add(RetentionEntry::new(
            "temp-001",
            RetentionClass::Temporary,
            0,
            Some(1000),
            false,
        ));

        let policy = EnforcementPolicy {
            grace_period_ms: 500,
            ..Default::default()
        };
        let engine = EnforcementEngine::new(policy);

        // Within grace period (expired by 200ms, grace is 500ms)
        let plan = engine.evaluate(&sched, 1200, false);
        assert_eq!(plan.grace_period_count, 1);
        assert_eq!(plan.actionable_count, 0);

        // Past grace period (expired by 600ms, grace is 500ms)
        let plan = engine.evaluate(&sched, 1600, false);
        assert_eq!(plan.grace_period_count, 0);
        assert_eq!(plan.actionable_count, 1);
    }

    #[test]
    fn test_enforcement_batch_limit() {
        let mut sched = RetentionSchedule::new();
        for i in 0..20 {
            sched.add(RetentionEntry::new(
                format!("item-{i:03}"),
                RetentionClass::Temporary,
                0,
                Some(100),
                false,
            ));
        }

        let policy = EnforcementPolicy {
            grace_period_ms: 0,
            batch_limit: 5,
            ..Default::default()
        };
        let engine = EnforcementEngine::new(policy);
        let plan = engine.evaluate(&sched, 200, false);

        assert!(plan.items.len() <= 5);
    }

    #[test]
    fn test_enforcement_audit_trail() {
        let mut sched = RetentionSchedule::new();
        sched.add(RetentionEntry::new(
            "temp-001",
            RetentionClass::Temporary,
            0,
            Some(100),
            false,
        ));

        let policy = EnforcementPolicy {
            grace_period_ms: 0,
            ..Default::default()
        };
        let engine = EnforcementEngine::new(policy);
        let plan = engine.evaluate(&sched, 200, true);
        let audit = engine.generate_audit_trail(&plan);

        assert_eq!(audit.len(), 1);
        assert!(audit[0].dry_run);
        assert_eq!(audit[0].asset_id, "temp-001");
        assert!(audit[0].event_id.starts_with("enf-200-"));
    }

    #[test]
    fn test_enforcement_empty_schedule() {
        let sched = RetentionSchedule::new();
        let engine = EnforcementEngine::with_defaults();
        let plan = engine.evaluate(&sched, 1000, true);

        assert!(!plan.has_actions());
        assert_eq!(plan.total_evaluated, 0);
    }

    #[test]
    fn test_enforcement_plan_summary_string() {
        let mut sched = RetentionSchedule::new();
        sched.add(RetentionEntry::new(
            "temp-001",
            RetentionClass::Temporary,
            0,
            Some(100),
            false,
        ));

        let policy = EnforcementPolicy {
            grace_period_ms: 0,
            ..Default::default()
        };
        let engine = EnforcementEngine::new(policy);
        let plan = engine.evaluate(&sched, 200, true);
        let summary = plan.to_summary_string();

        assert!(summary.contains("Retention Enforcement Plan"));
        assert!(summary.contains("DRY RUN"));
    }

    #[test]
    fn test_enforcement_action_display() {
        assert_eq!(EnforcementAction::Delete.to_string(), "DELETE");
        let archive = EnforcementAction::Archive {
            target_tier: "cold".to_string(),
        };
        assert!(archive.to_string().contains("cold"));
        let extend = EnforcementAction::Extend {
            extension_ms: 30 * MS_PER_DAY,
        };
        assert!(extend.to_string().contains("30 days"));
    }

    #[test]
    fn test_enforcement_default_expiry() {
        // Test with no explicit expiry, using default retention period
        let mut sched = RetentionSchedule::new();
        sched.add(RetentionEntry::new(
            "std-001",
            RetentionClass::Standard, // 5 years default
            0,
            None,
            false,
        ));

        let policy = EnforcementPolicy {
            grace_period_ms: 0,
            ..Default::default()
        };
        let engine = EnforcementEngine::new(policy);

        // Before 5 years
        let plan = engine.evaluate(&sched, 3 * MS_PER_YEAR, false);
        assert_eq!(plan.actionable_count, 0);

        // After 5 years
        let plan = engine.evaluate(&sched, 6 * MS_PER_YEAR, false);
        assert_eq!(plan.actionable_count, 1);
    }

    #[test]
    fn test_enforcement_extend_action() {
        let mut sched = RetentionSchedule::new();
        sched.add(RetentionEntry::new(
            "ext-001",
            RetentionClass::Standard,
            0,
            Some(100),
            false,
        ));

        let policy = EnforcementPolicy {
            grace_period_ms: 0,
            standard_action: EnforcementAction::Extend {
                extension_ms: 90 * MS_PER_DAY,
            },
            ..Default::default()
        };
        let engine = EnforcementEngine::new(policy);
        let plan = engine.evaluate(&sched, 200, false);

        assert_eq!(plan.action_summary.extend_count, 1);
    }
}

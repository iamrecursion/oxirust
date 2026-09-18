//! Long-term preservation policy definitions for media archives.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// ReplicationTier
// ---------------------------------------------------------------------------

/// Defines how many on-site and off-site replica copies are required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationTier {
    /// Number of on-site copies required.
    pub on_site_copies: u32,
    /// Number of off-site copies required.
    pub off_site_copies: u32,
}

impl ReplicationTier {
    /// Create a new replication tier.
    #[must_use]
    pub fn new(on_site_copies: u32, off_site_copies: u32) -> Self {
        Self {
            on_site_copies,
            off_site_copies,
        }
    }

    /// Total number of copies required.
    #[must_use]
    pub fn total_copies(&self) -> u32 {
        self.on_site_copies + self.off_site_copies
    }

    /// Returns `true` when the tier satisfies the 3-2-1 backup rule:
    /// at least 3 total, at least 2 on-site, at least 1 off-site.
    #[must_use]
    pub fn satisfies_321_rule(&self) -> bool {
        self.total_copies() >= 3 && self.on_site_copies >= 2 && self.off_site_copies >= 1
    }
}

// ---------------------------------------------------------------------------
// FixitySchedule
// ---------------------------------------------------------------------------

/// How often fixity checks should be performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixitySchedule {
    /// Every day.
    Daily,
    /// Every week.
    Weekly,
    /// Every 30 days.
    Monthly,
    /// Every 90 days.
    Quarterly,
    /// Every 365 days.
    Annually,
}

impl FixitySchedule {
    /// Approximate number of days between checks.
    #[must_use]
    pub fn interval_days(&self) -> u32 {
        match self {
            FixitySchedule::Daily => 1,
            FixitySchedule::Weekly => 7,
            FixitySchedule::Monthly => 30,
            FixitySchedule::Quarterly => 90,
            FixitySchedule::Annually => 365,
        }
    }
}

// ---------------------------------------------------------------------------
// PreservationPolicy
// ---------------------------------------------------------------------------

/// A named preservation policy applied to one or more archive collections.
#[derive(Debug, Clone)]
pub struct PreservationPolicy {
    /// Human-readable name of the policy.
    pub name: String,
    /// Description of the policy.
    pub description: String,
    /// Replication requirements.
    pub replication: ReplicationTier,
    /// How often fixity checks must be run.
    pub fixity_schedule: FixitySchedule,
    /// Minimum number of years the asset must be retained.
    pub retention_years: u32,
    /// Whether format migration is required at end-of-life for a format.
    pub requires_format_migration: bool,
}

impl PreservationPolicy {
    /// Create a new preservation policy.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        replication: ReplicationTier,
        fixity_schedule: FixitySchedule,
        retention_years: u32,
        requires_format_migration: bool,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            replication,
            fixity_schedule,
            retention_years,
            requires_format_migration,
        }
    }

    /// Returns `true` if this policy is classified as "gold" (highest grade):
    /// satisfies 3-2-1 rule, monthly or more frequent fixity, ≥10 years retention.
    #[must_use]
    pub fn is_gold_tier(&self) -> bool {
        self.replication.satisfies_321_rule()
            && self.fixity_schedule.interval_days() <= 30
            && self.retention_years >= 10
    }

    /// Approximate total number of fixity checks over the retention period.
    #[must_use]
    pub fn total_fixity_checks(&self) -> u64 {
        let days = u64::from(self.retention_years) * 365;
        let interval = u64::from(self.fixity_schedule.interval_days()).max(1);
        days / interval
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- ReplicationTier ---

    #[test]
    fn test_replication_total_copies() {
        let tier = ReplicationTier::new(2, 1);
        assert_eq!(tier.total_copies(), 3);
    }

    #[test]
    fn test_replication_321_satisfied() {
        let tier = ReplicationTier::new(2, 1);
        assert!(tier.satisfies_321_rule());
    }

    #[test]
    fn test_replication_321_not_satisfied_low_total() {
        let tier = ReplicationTier::new(1, 1);
        assert!(!tier.satisfies_321_rule());
    }

    #[test]
    fn test_replication_321_not_satisfied_no_offsite() {
        let tier = ReplicationTier::new(3, 0);
        assert!(!tier.satisfies_321_rule());
    }

    #[test]
    fn test_replication_321_not_satisfied_one_onsite() {
        let tier = ReplicationTier::new(1, 2);
        // Total = 3, on-site < 2 → fails
        assert!(!tier.satisfies_321_rule());
    }

    #[test]
    fn test_replication_fields() {
        let tier = ReplicationTier::new(3, 2);
        assert_eq!(tier.on_site_copies, 3);
        assert_eq!(tier.off_site_copies, 2);
    }

    // --- FixitySchedule ---

    #[test]
    fn test_fixity_daily_interval() {
        assert_eq!(FixitySchedule::Daily.interval_days(), 1);
    }

    #[test]
    fn test_fixity_weekly_interval() {
        assert_eq!(FixitySchedule::Weekly.interval_days(), 7);
    }

    #[test]
    fn test_fixity_monthly_interval() {
        assert_eq!(FixitySchedule::Monthly.interval_days(), 30);
    }

    #[test]
    fn test_fixity_quarterly_interval() {
        assert_eq!(FixitySchedule::Quarterly.interval_days(), 90);
    }

    #[test]
    fn test_fixity_annually_interval() {
        assert_eq!(FixitySchedule::Annually.interval_days(), 365);
    }

    // --- PreservationPolicy ---

    #[test]
    fn test_policy_gold_tier_yes() {
        let policy = PreservationPolicy::new(
            "Gold",
            "Highest grade policy",
            ReplicationTier::new(2, 1),
            FixitySchedule::Monthly,
            25,
            true,
        );
        assert!(policy.is_gold_tier());
    }

    #[test]
    fn test_policy_gold_tier_no_quarterly() {
        // Quarterly is > 30 days, should fail gold check.
        let policy = PreservationPolicy::new(
            "Silver",
            "Not gold",
            ReplicationTier::new(2, 1),
            FixitySchedule::Quarterly,
            25,
            false,
        );
        assert!(!policy.is_gold_tier());
    }

    #[test]
    fn test_policy_gold_tier_no_short_retention() {
        let policy = PreservationPolicy::new(
            "Bronze",
            "Short retention",
            ReplicationTier::new(2, 1),
            FixitySchedule::Monthly,
            5,
            false,
        );
        assert!(!policy.is_gold_tier());
    }

    #[test]
    fn test_policy_total_fixity_checks() {
        // 10 years × 365 days / 30 days-per-check ≈ 121
        let policy = PreservationPolicy::new(
            "Test",
            "Policy",
            ReplicationTier::new(2, 1),
            FixitySchedule::Monthly,
            10,
            false,
        );
        let checks = policy.total_fixity_checks();
        assert_eq!(checks, 10 * 365 / 30);
    }

    #[test]
    fn test_policy_name_field() {
        let policy = PreservationPolicy::new(
            "MyPolicy",
            "Desc",
            ReplicationTier::new(2, 1),
            FixitySchedule::Weekly,
            5,
            false,
        );
        assert_eq!(policy.name, "MyPolicy");
    }
}

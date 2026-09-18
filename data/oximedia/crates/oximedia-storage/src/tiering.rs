//! Storage tiering policies for cost optimization and lifecycle management.
//!
//! This module provides tools for managing data across multiple storage tiers
//! (NVMe, SSD, HDD, Tape, Glacier Deep) based on access patterns and age,
//! enabling automatic cost optimization for large media collections.

#![allow(dead_code)]

/// Storage tier representing physical or logical storage class
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    /// NVMe SSD — extremely fast, very expensive
    NVMe,
    /// Standard SSD — fast, moderately expensive
    Ssd,
    /// Hard disk drive — moderate speed, affordable
    Hdd,
    /// Tape archive — very slow retrieval, cheap long-term storage
    Tape,
    /// AWS Glacier Deep Archive equivalent — minimal cost, hours-long retrieval
    GlacierDeep,
}

impl StorageTier {
    /// Cost per terabyte per month in USD
    pub fn cost_per_tb_month_usd(self) -> f64 {
        match self {
            Self::NVMe => 200.0,
            Self::Ssd => 80.0,
            Self::Hdd => 20.0,
            Self::Tape => 5.0,
            Self::GlacierDeep => 1.0,
        }
    }

    /// Approximate access latency in milliseconds
    pub fn access_latency_ms(self) -> u64 {
        match self {
            Self::NVMe => 1,
            Self::Ssd => 5,
            Self::Hdd => 20,
            Self::Tape => 60_000,            // ~1 minute to load tape
            Self::GlacierDeep => 43_200_000, // ~12 hours
        }
    }
}

impl std::fmt::Display for StorageTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NVMe => write!(f, "NVMe"),
            Self::Ssd => write!(f, "SSD"),
            Self::Hdd => write!(f, "HDD"),
            Self::Tape => write!(f, "Tape"),
            Self::GlacierDeep => write!(f, "GlacierDeep"),
        }
    }
}

/// Policy governing when data should be migrated between tiers
#[derive(Debug, Clone)]
pub struct TierPolicy {
    /// Minimum access count to remain "hot" (on fast tier)
    pub hot_access_count: u32,
    /// Days after last access before moving to warm tier
    pub warm_days: u32,
    /// Days after last access before moving to cold tier
    pub cold_days: u32,
    /// Days after last access before archiving
    pub archive_days: u32,
}

impl TierPolicy {
    /// Default policy tuned for video production workflows
    pub fn default_video() -> Self {
        Self {
            hot_access_count: 10,
            warm_days: 30,
            cold_days: 90,
            archive_days: 365,
        }
    }
}

impl Default for TierPolicy {
    fn default() -> Self {
        Self::default_video()
    }
}

/// The result of evaluating which tier data should reside on
#[derive(Debug, Clone)]
pub struct TierDecision {
    /// Current storage tier
    pub current: StorageTier,
    /// Recommended storage tier
    pub recommended: StorageTier,
    /// Human-readable reason for the recommendation
    pub reason: String,
    /// Estimated monthly savings in USD if the recommendation is followed
    pub savings_usd_month: f64,
}

impl TierDecision {
    /// Returns `true` if the recommended tier differs from the current tier
    pub fn requires_migration(&self) -> bool {
        self.current != self.recommended
    }
}

/// Engine that evaluates tiering decisions for individual items
pub struct TieringEngine;

impl TieringEngine {
    /// Evaluate the recommended tier for a single item.
    ///
    /// # Arguments
    ///
    /// * `last_accessed_days_ago` – How many days ago the item was last accessed
    /// * `access_count` – Total number of times the item has been accessed
    /// * `size_gb` – Size of the item in gigabytes
    /// * `policy` – The tiering policy to apply
    pub fn evaluate(
        last_accessed_days_ago: u32,
        access_count: u32,
        size_gb: f64,
        policy: &TierPolicy,
        current: StorageTier,
    ) -> TierDecision {
        let (recommended, reason) =
            Self::determine_tier(last_accessed_days_ago, access_count, policy);

        let size_tb = size_gb / 1024.0;
        let current_cost = current.cost_per_tb_month_usd() * size_tb;
        let recommended_cost = recommended.cost_per_tb_month_usd() * size_tb;
        let savings_usd_month = (current_cost - recommended_cost).max(0.0);

        TierDecision {
            current,
            recommended,
            reason,
            savings_usd_month,
        }
    }

    fn determine_tier(
        last_accessed_days_ago: u32,
        access_count: u32,
        policy: &TierPolicy,
    ) -> (StorageTier, String) {
        if access_count >= policy.hot_access_count && last_accessed_days_ago < policy.warm_days {
            return (
                StorageTier::NVMe,
                format!(
                    "Frequently accessed ({access_count} times, {last_accessed_days_ago} days ago) — keep on fast storage"
                ),
            );
        }

        if last_accessed_days_ago >= policy.archive_days {
            return (
                StorageTier::GlacierDeep,
                format!(
                    "Not accessed for {} days (threshold: {} days) — archive",
                    last_accessed_days_ago, policy.archive_days
                ),
            );
        }

        if last_accessed_days_ago >= policy.cold_days {
            return (
                StorageTier::Tape,
                format!(
                    "Not accessed for {} days (threshold: {} days) — move to cold storage",
                    last_accessed_days_ago, policy.cold_days
                ),
            );
        }

        if last_accessed_days_ago >= policy.warm_days {
            return (
                StorageTier::Hdd,
                format!(
                    "Last accessed {} days ago (threshold: {} days) — move to warm storage",
                    last_accessed_days_ago, policy.warm_days
                ),
            );
        }

        (
            StorageTier::Ssd,
            format!(
                "Moderately accessed ({access_count} times, {last_accessed_days_ago} days ago)"
            ),
        )
    }
}

/// A single item in a storage collection
#[derive(Debug, Clone)]
pub struct StorageItem {
    /// Logical path or key of the item
    pub path: String,
    /// Size in gigabytes
    pub size_gb: f64,
    /// Total access count
    pub access_count: u32,
    /// Days since last access
    pub last_accessed_days_ago: u32,
    /// Current storage tier
    pub current_tier: StorageTier,
}

/// Optimizer that evaluates an entire collection of items
pub struct StorageOptimizer;

impl StorageOptimizer {
    /// Evaluate tiering decisions for a collection of storage items.
    pub fn optimize_collection(items: &[StorageItem], policy: &TierPolicy) -> Vec<TierDecision> {
        items
            .iter()
            .map(|item| {
                TieringEngine::evaluate(
                    item.last_accessed_days_ago,
                    item.access_count,
                    item.size_gb,
                    policy,
                    item.current_tier,
                )
            })
            .collect()
    }

    /// Calculate the total estimated monthly savings across all decisions.
    pub fn estimate_savings(decisions: &[TierDecision]) -> f64 {
        decisions
            .iter()
            .filter(|d| d.requires_migration())
            .map(|d| d.savings_usd_month)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_policy() -> TierPolicy {
        TierPolicy::default_video()
    }

    // --- StorageTier ---

    #[test]
    fn test_storage_tier_cost_ordering() {
        // NVMe should be most expensive, GlacierDeep cheapest
        assert!(
            StorageTier::NVMe.cost_per_tb_month_usd() > StorageTier::Ssd.cost_per_tb_month_usd()
        );
        assert!(
            StorageTier::Ssd.cost_per_tb_month_usd() > StorageTier::Hdd.cost_per_tb_month_usd()
        );
        assert!(
            StorageTier::Hdd.cost_per_tb_month_usd() > StorageTier::Tape.cost_per_tb_month_usd()
        );
        assert!(
            StorageTier::Tape.cost_per_tb_month_usd()
                > StorageTier::GlacierDeep.cost_per_tb_month_usd()
        );
    }

    #[test]
    fn test_storage_tier_latency_ordering() {
        assert!(StorageTier::NVMe.access_latency_ms() < StorageTier::Ssd.access_latency_ms());
        assert!(StorageTier::Ssd.access_latency_ms() < StorageTier::Hdd.access_latency_ms());
        assert!(StorageTier::Hdd.access_latency_ms() < StorageTier::Tape.access_latency_ms());
        assert!(
            StorageTier::Tape.access_latency_ms() < StorageTier::GlacierDeep.access_latency_ms()
        );
    }

    #[test]
    fn test_storage_tier_display() {
        assert_eq!(StorageTier::NVMe.to_string(), "NVMe");
        assert_eq!(StorageTier::Ssd.to_string(), "SSD");
        assert_eq!(StorageTier::Hdd.to_string(), "HDD");
        assert_eq!(StorageTier::Tape.to_string(), "Tape");
        assert_eq!(StorageTier::GlacierDeep.to_string(), "GlacierDeep");
    }

    // --- TierPolicy ---

    #[test]
    fn test_tier_policy_default_video() {
        let p = TierPolicy::default_video();
        assert_eq!(p.hot_access_count, 10);
        assert_eq!(p.warm_days, 30);
        assert_eq!(p.cold_days, 90);
        assert_eq!(p.archive_days, 365);
    }

    #[test]
    fn test_tier_policy_default_equals_video() {
        let a = TierPolicy::default();
        let b = TierPolicy::default_video();
        assert_eq!(a.hot_access_count, b.hot_access_count);
        assert_eq!(a.warm_days, b.warm_days);
    }

    // --- TieringEngine ---

    #[test]
    fn test_evaluate_hot_item_stays_on_nvme() {
        let policy = default_policy();
        let decision = TieringEngine::evaluate(5, 20, 100.0, &policy, StorageTier::NVMe);
        assert_eq!(decision.recommended, StorageTier::NVMe);
        assert!(!decision.requires_migration());
    }

    #[test]
    fn test_evaluate_warm_item_moves_to_hdd() {
        let policy = default_policy();
        // 45 days ago, low access count
        let decision = TieringEngine::evaluate(45, 2, 100.0, &policy, StorageTier::NVMe);
        assert_eq!(decision.recommended, StorageTier::Hdd);
        assert!(decision.requires_migration());
    }

    #[test]
    fn test_evaluate_cold_item_moves_to_tape() {
        let policy = default_policy();
        let decision = TieringEngine::evaluate(100, 1, 100.0, &policy, StorageTier::Hdd);
        assert_eq!(decision.recommended, StorageTier::Tape);
        assert!(decision.requires_migration());
    }

    #[test]
    fn test_evaluate_archive_item_moves_to_glacier() {
        let policy = default_policy();
        let decision = TieringEngine::evaluate(400, 0, 100.0, &policy, StorageTier::Hdd);
        assert_eq!(decision.recommended, StorageTier::GlacierDeep);
        assert!(decision.requires_migration());
    }

    #[test]
    fn test_evaluate_savings_positive_when_downgrading() {
        let policy = default_policy();
        let decision = TieringEngine::evaluate(400, 0, 1024.0, &policy, StorageTier::NVMe);
        assert!(decision.savings_usd_month > 0.0);
    }

    #[test]
    fn test_evaluate_no_savings_when_already_optimal() {
        let policy = default_policy();
        let decision = TieringEngine::evaluate(400, 0, 1024.0, &policy, StorageTier::GlacierDeep);
        assert_eq!(decision.savings_usd_month, 0.0);
        assert!(!decision.requires_migration());
    }

    // --- StorageOptimizer ---

    #[test]
    fn test_optimize_collection_empty() {
        let decisions = StorageOptimizer::optimize_collection(&[], &default_policy());
        assert!(decisions.is_empty());
    }

    #[test]
    fn test_optimize_collection_mixed() {
        let items = vec![
            StorageItem {
                path: "hot.mp4".into(),
                size_gb: 50.0,
                access_count: 20,
                last_accessed_days_ago: 2,
                current_tier: StorageTier::NVMe,
            },
            StorageItem {
                path: "archive.mp4".into(),
                size_gb: 200.0,
                access_count: 0,
                last_accessed_days_ago: 400,
                current_tier: StorageTier::NVMe,
            },
        ];
        let decisions = StorageOptimizer::optimize_collection(&items, &default_policy());
        assert_eq!(decisions.len(), 2);
        // Hot item stays NVMe
        assert_eq!(decisions[0].recommended, StorageTier::NVMe);
        // Archive item moves to GlacierDeep
        assert_eq!(decisions[1].recommended, StorageTier::GlacierDeep);
    }

    #[test]
    fn test_estimate_savings_sums_only_migrations() {
        let decisions = vec![
            TierDecision {
                current: StorageTier::NVMe,
                recommended: StorageTier::NVMe,
                reason: "hot".into(),
                savings_usd_month: 0.0,
            },
            TierDecision {
                current: StorageTier::NVMe,
                recommended: StorageTier::GlacierDeep,
                reason: "archive".into(),
                savings_usd_month: 100.0,
            },
        ];
        let total = StorageOptimizer::estimate_savings(&decisions);
        assert!((total - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tier_decision_requires_migration() {
        let no_change = TierDecision {
            current: StorageTier::Ssd,
            recommended: StorageTier::Ssd,
            reason: String::new(),
            savings_usd_month: 0.0,
        };
        assert!(!no_change.requires_migration());

        let migrates = TierDecision {
            current: StorageTier::NVMe,
            recommended: StorageTier::Hdd,
            reason: String::new(),
            savings_usd_month: 50.0,
        };
        assert!(migrates.requires_migration());
    }
}

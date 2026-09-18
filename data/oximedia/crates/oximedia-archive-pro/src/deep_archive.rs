//! Deep archive tier management: cold storage classification, access frequency
//! tracking, and cost modelling.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Storage tier classification for deep archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StorageTier {
    /// Frequently accessed, low latency
    Hot,
    /// Occasionally accessed, moderate cost
    Warm,
    /// Rarely accessed, higher latency, low cost
    Cold,
    /// Deep archive – years between access, lowest cost
    Glacier,
}

impl StorageTier {
    /// Monthly cost per GB in USD (illustrative values).
    #[must_use]
    pub fn cost_per_gb_month(&self) -> f64 {
        match self {
            Self::Hot => 0.023,
            Self::Warm => 0.0125,
            Self::Cold => 0.004,
            Self::Glacier => 0.001,
        }
    }

    /// Retrieval latency in seconds (worst-case).
    #[must_use]
    pub fn retrieval_latency_secs(&self) -> u64 {
        match self {
            Self::Hot => 0,
            Self::Warm => 60,
            Self::Cold => 3_600,
            Self::Glacier => 43_200,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Warm => "warm",
            Self::Cold => "cold",
            Self::Glacier => "glacier",
        }
    }
}

/// Access frequency record for a single archive item.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AccessRecord {
    /// Asset identifier
    pub asset_id: String,
    /// Number of accesses in the last 30 days
    pub accesses_30d: u32,
    /// Number of accesses in the last 365 days
    pub accesses_365d: u32,
    /// Size in bytes
    pub size_bytes: u64,
    /// Current tier
    pub current_tier: StorageTier,
}

impl AccessRecord {
    /// Create a new access record.
    #[must_use]
    pub fn new(
        asset_id: impl Into<String>,
        size_bytes: u64,
        accesses_30d: u32,
        accesses_365d: u32,
        current_tier: StorageTier,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            size_bytes,
            accesses_30d,
            accesses_365d,
            current_tier,
        }
    }

    /// Recommended tier based on access frequency.
    #[must_use]
    pub fn recommended_tier(&self) -> StorageTier {
        if self.accesses_30d >= 10 {
            StorageTier::Hot
        } else if self.accesses_30d >= 2 {
            StorageTier::Warm
        } else if self.accesses_365d >= 4 {
            StorageTier::Cold
        } else {
            StorageTier::Glacier
        }
    }

    /// Monthly storage cost in USD.
    #[must_use]
    pub fn monthly_cost(&self) -> f64 {
        let gb = self.size_bytes as f64 / 1_073_741_824.0;
        gb * self.current_tier.cost_per_gb_month()
    }

    /// Potential savings by moving to recommended tier.
    #[must_use]
    pub fn potential_savings(&self) -> f64 {
        let gb = self.size_bytes as f64 / 1_073_741_824.0;
        let current_cost = gb * self.current_tier.cost_per_gb_month();
        let optimal_cost = gb * self.recommended_tier().cost_per_gb_month();
        (current_cost - optimal_cost).max(0.0)
    }
}

/// Decision produced by the tier classifier.
#[derive(Debug, Clone)]
pub struct TierDecision {
    /// Asset identifier
    pub asset_id: String,
    /// From tier
    pub from: StorageTier,
    /// Recommended to tier
    pub to: StorageTier,
    /// Monthly savings in USD
    pub monthly_savings: f64,
}

/// Classifies assets into storage tiers and produces migration decisions.
#[derive(Debug, Default)]
pub struct TierClassifier {
    records: Vec<AccessRecord>,
}

impl TierClassifier {
    /// Create an empty classifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an access record.
    pub fn add_record(&mut self, record: AccessRecord) {
        self.records.push(record);
    }

    /// Return all migration decisions where tier change is beneficial.
    #[must_use]
    pub fn decisions(&self) -> Vec<TierDecision> {
        self.records
            .iter()
            .filter_map(|r| {
                let recommended = r.recommended_tier();
                if recommended == r.current_tier {
                    return None;
                }
                let savings = r.potential_savings();
                Some(TierDecision {
                    asset_id: r.asset_id.clone(),
                    from: r.current_tier,
                    to: recommended,
                    monthly_savings: savings,
                })
            })
            .collect()
    }

    /// Total monthly cost for all tracked assets.
    #[must_use]
    pub fn total_monthly_cost(&self) -> f64 {
        self.records.iter().map(|r| r.monthly_cost()).sum()
    }

    /// Total potential monthly savings.
    #[must_use]
    pub fn total_potential_savings(&self) -> f64 {
        self.records.iter().map(|r| r.potential_savings()).sum()
    }

    /// Counts of assets per tier.
    #[must_use]
    pub fn tier_counts(&self) -> HashMap<StorageTier, usize> {
        let mut map: HashMap<StorageTier, usize> = HashMap::new();
        for r in &self.records {
            *map.entry(r.current_tier).or_insert(0) += 1;
        }
        map
    }
}

/// Cost model helper for projecting archive spend.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Tier → GB capacity planned
    tier_gb: HashMap<StorageTier, f64>,
}

impl CostModel {
    /// Create a new empty cost model.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tier_gb: HashMap::new(),
        }
    }

    /// Set planned capacity for a tier.
    pub fn set_tier_gb(&mut self, tier: StorageTier, gb: f64) {
        self.tier_gb.insert(tier, gb);
    }

    /// Projected monthly cost.
    #[must_use]
    pub fn monthly_cost(&self) -> f64 {
        self.tier_gb
            .iter()
            .map(|(tier, &gb)| gb * tier.cost_per_gb_month())
            .sum()
    }

    /// Projected annual cost.
    #[must_use]
    pub fn annual_cost(&self) -> f64 {
        self.monthly_cost() * 12.0
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_tier_labels() {
        assert_eq!(StorageTier::Hot.label(), "hot");
        assert_eq!(StorageTier::Warm.label(), "warm");
        assert_eq!(StorageTier::Cold.label(), "cold");
        assert_eq!(StorageTier::Glacier.label(), "glacier");
    }

    #[test]
    fn test_storage_tier_cost_ordering() {
        assert!(StorageTier::Hot.cost_per_gb_month() > StorageTier::Warm.cost_per_gb_month());
        assert!(StorageTier::Warm.cost_per_gb_month() > StorageTier::Cold.cost_per_gb_month());
        assert!(StorageTier::Cold.cost_per_gb_month() > StorageTier::Glacier.cost_per_gb_month());
    }

    #[test]
    fn test_storage_tier_retrieval_latency_ordering() {
        assert!(
            StorageTier::Glacier.retrieval_latency_secs()
                > StorageTier::Cold.retrieval_latency_secs()
        );
        assert!(
            StorageTier::Cold.retrieval_latency_secs() > StorageTier::Warm.retrieval_latency_secs()
        );
        assert!(
            StorageTier::Warm.retrieval_latency_secs() > StorageTier::Hot.retrieval_latency_secs()
        );
    }

    #[test]
    fn test_access_record_recommended_tier_hot() {
        let r = AccessRecord::new("asset1", 1_073_741_824, 15, 200, StorageTier::Hot);
        assert_eq!(r.recommended_tier(), StorageTier::Hot);
    }

    #[test]
    fn test_access_record_recommended_tier_warm() {
        let r = AccessRecord::new("asset2", 1_073_741_824, 5, 60, StorageTier::Hot);
        assert_eq!(r.recommended_tier(), StorageTier::Warm);
    }

    #[test]
    fn test_access_record_recommended_tier_cold() {
        let r = AccessRecord::new("asset3", 1_073_741_824, 0, 10, StorageTier::Hot);
        assert_eq!(r.recommended_tier(), StorageTier::Cold);
    }

    #[test]
    fn test_access_record_recommended_tier_glacier() {
        let r = AccessRecord::new("asset4", 1_073_741_824, 0, 1, StorageTier::Hot);
        assert_eq!(r.recommended_tier(), StorageTier::Glacier);
    }

    #[test]
    fn test_access_record_monthly_cost() {
        // 1 GiB at Glacier ($0.001/GB/month) ≈ $0.001
        let r = AccessRecord::new("asset5", 1_073_741_824, 0, 0, StorageTier::Glacier);
        let cost = r.monthly_cost();
        assert!(cost > 0.0);
        assert!(cost < 0.01);
    }

    #[test]
    fn test_access_record_potential_savings() {
        // Hot asset with zero accesses should have positive savings
        let r = AccessRecord::new("asset6", 10 * 1_073_741_824, 0, 0, StorageTier::Hot);
        assert!(r.potential_savings() > 0.0);
    }

    #[test]
    fn test_tier_classifier_decisions() {
        let mut clf = TierClassifier::new();
        clf.add_record(AccessRecord::new(
            "a1",
            1_073_741_824,
            0,
            0,
            StorageTier::Hot,
        ));
        let decisions = clf.decisions();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].to, StorageTier::Glacier);
    }

    #[test]
    fn test_tier_classifier_no_decision_when_optimal() {
        let mut clf = TierClassifier::new();
        clf.add_record(AccessRecord::new(
            "a2",
            1_073_741_824,
            20,
            300,
            StorageTier::Hot,
        ));
        assert!(clf.decisions().is_empty());
    }

    #[test]
    fn test_tier_classifier_total_cost() {
        let mut clf = TierClassifier::new();
        clf.add_record(AccessRecord::new(
            "a3",
            1_073_741_824,
            0,
            0,
            StorageTier::Hot,
        ));
        assert!(clf.total_monthly_cost() > 0.0);
    }

    #[test]
    fn test_tier_classifier_tier_counts() {
        let mut clf = TierClassifier::new();
        clf.add_record(AccessRecord::new("a4", 512, 0, 0, StorageTier::Hot));
        clf.add_record(AccessRecord::new("a5", 512, 0, 0, StorageTier::Cold));
        let counts = clf.tier_counts();
        assert_eq!(counts[&StorageTier::Hot], 1);
        assert_eq!(counts[&StorageTier::Cold], 1);
    }

    #[test]
    fn test_cost_model_monthly_cost() {
        let mut model = CostModel::new();
        model.set_tier_gb(StorageTier::Glacier, 1000.0);
        let cost = model.monthly_cost();
        assert!((cost - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cost_model_annual_cost() {
        let mut model = CostModel::new();
        model.set_tier_gb(StorageTier::Glacier, 1000.0);
        let annual = model.annual_cost();
        assert!((annual - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_cost_model_multi_tier() {
        let mut model = CostModel::new();
        model.set_tier_gb(StorageTier::Hot, 100.0);
        model.set_tier_gb(StorageTier::Glacier, 1000.0);
        let expected = 100.0 * StorageTier::Hot.cost_per_gb_month()
            + 1000.0 * StorageTier::Glacier.cost_per_gb_month();
        assert!((model.monthly_cost() - expected).abs() < 1e-9);
    }
}

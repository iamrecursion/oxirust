//! Tiered royalty rates based on usage volume thresholds.
//!
//! This module extends the royalty engine with a graduated rate schedule where
//! the per-unit royalty rate changes as cumulative usage crosses defined
//! volume thresholds.  This models real-world agreements such as:
//!
//! - First 10,000 streams at $0.004/play
//! - 10,001–100,000 streams at $0.003/play
//! - 100,001+ streams at $0.002/play

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

// ── RoyaltyTier ─────────────────────────────────────────────────────────────

/// A single tier in a graduated royalty schedule.
///
/// The tier applies when the cumulative usage count falls within
/// `[threshold_min, threshold_max)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoyaltyTier {
    /// Human-readable label for this tier (e.g. "Base", "Silver", "Gold").
    pub label: String,
    /// Minimum cumulative usage for this tier to kick in (inclusive).
    pub threshold_min: u64,
    /// Maximum cumulative usage for this tier (exclusive).
    /// `None` means the tier extends to infinity.
    pub threshold_max: Option<u64>,
    /// Per-unit rate in the agreement's currency.
    pub rate_per_unit: f64,
}

impl RoyaltyTier {
    /// Create a new tier.
    pub fn new(
        label: &str,
        threshold_min: u64,
        threshold_max: Option<u64>,
        rate_per_unit: f64,
    ) -> Self {
        Self {
            label: label.to_string(),
            threshold_min,
            threshold_max,
            rate_per_unit,
        }
    }

    /// Number of units this tier can accommodate.
    /// Returns `None` for the unlimited top tier.
    pub fn capacity(&self) -> Option<u64> {
        self.threshold_max.map(|max| max - self.threshold_min)
    }

    /// Returns `true` if `usage_count` falls within this tier's range.
    pub fn contains(&self, usage_count: u64) -> bool {
        if usage_count < self.threshold_min {
            return false;
        }
        match self.threshold_max {
            Some(max) => usage_count < max,
            None => true,
        }
    }

    /// Calculate the royalty for the portion of usage that falls within this
    /// tier, given a total `usage_count` starting from zero.
    ///
    /// Returns (units_in_tier, royalty_amount).
    pub fn calculate_portion(&self, usage_count: u64) -> (u64, f64) {
        if usage_count <= self.threshold_min {
            return (0, 0.0);
        }
        let effective_max = match self.threshold_max {
            Some(max) => usage_count.min(max),
            None => usage_count,
        };
        let units = effective_max.saturating_sub(self.threshold_min);
        (units, units as f64 * self.rate_per_unit)
    }
}

// ── TieredRoyaltySchedule ───────────────────────────────────────────────────

/// A schedule of royalty tiers sorted by threshold.
///
/// Tiers must be contiguous and non-overlapping.  The schedule validates
/// this at build time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TieredRoyaltySchedule {
    /// Ordered tiers (ascending by `threshold_min`).
    tiers: Vec<RoyaltyTier>,
    /// ISO 4217 currency code.
    pub currency: String,
    /// Optional description of this schedule.
    pub description: String,
}

impl TieredRoyaltySchedule {
    /// Create a new schedule with the given currency.
    pub fn new(currency: &str) -> Self {
        Self {
            tiers: Vec::new(),
            currency: currency.to_string(),
            description: String::new(),
        }
    }

    /// Add a tier.  Tiers are automatically sorted by `threshold_min`.
    pub fn add_tier(&mut self, tier: RoyaltyTier) {
        self.tiers.push(tier);
        self.tiers.sort_by_key(|t| t.threshold_min);
    }

    /// Set a description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Number of tiers.
    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    /// Access the tiers.
    pub fn tiers(&self) -> &[RoyaltyTier] {
        &self.tiers
    }

    /// Validate that tiers are contiguous and non-overlapping.
    ///
    /// Returns `Ok(())` if valid, or a descriptive error string.
    pub fn validate(&self) -> Result<(), String> {
        if self.tiers.is_empty() {
            return Err("Schedule has no tiers".to_string());
        }

        for i in 1..self.tiers.len() {
            let prev = &self.tiers[i - 1];
            let curr = &self.tiers[i];
            match prev.threshold_max {
                Some(prev_max) => {
                    if prev_max != curr.threshold_min {
                        return Err(format!(
                            "Gap or overlap between tier '{}' (max={}) and '{}' (min={})",
                            prev.label, prev_max, curr.label, curr.threshold_min,
                        ));
                    }
                }
                None => {
                    return Err(format!(
                        "Tier '{}' has no upper bound but is not the last tier",
                        prev.label,
                    ));
                }
            }
        }

        Ok(())
    }

    /// Find which tier a given `usage_count` falls into.
    pub fn tier_for(&self, usage_count: u64) -> Option<&RoyaltyTier> {
        self.tiers.iter().find(|t| t.contains(usage_count))
    }

    /// Calculate the total royalty for `usage_count` units, applying each
    /// tier's rate to the portion of usage that falls within it.
    pub fn calculate(&self, usage_count: u64) -> TieredRoyaltyResult {
        let mut breakdown = Vec::new();
        let mut total = 0.0_f64;

        for tier in &self.tiers {
            let (units, amount) = tier.calculate_portion(usage_count);
            if units > 0 {
                total += amount;
                breakdown.push(TierBreakdown {
                    tier_label: tier.label.clone(),
                    units,
                    rate: tier.rate_per_unit,
                    amount,
                });
            }
        }

        TieredRoyaltyResult {
            usage_count,
            total_royalty: total,
            breakdown,
            currency: self.currency.clone(),
        }
    }

    /// Effective blended rate (total royalty / usage_count).
    ///
    /// Returns `None` when usage_count is zero.
    pub fn blended_rate(&self, usage_count: u64) -> Option<f64> {
        if usage_count == 0 {
            return None;
        }
        let result = self.calculate(usage_count);
        Some(result.total_royalty / usage_count as f64)
    }
}

// ── TierBreakdown ───────────────────────────────────────────────────────────

/// One line of a tiered royalty calculation showing how many units fell
/// into a specific tier and the resulting amount.
#[derive(Debug, Clone)]
pub struct TierBreakdown {
    /// Label of the tier.
    pub tier_label: String,
    /// Number of units in this tier.
    pub units: u64,
    /// Per-unit rate applied.
    pub rate: f64,
    /// Total amount for this tier.
    pub amount: f64,
}

// ── TieredRoyaltyResult ─────────────────────────────────────────────────────

/// Complete result of a tiered royalty calculation.
#[derive(Debug, Clone)]
pub struct TieredRoyaltyResult {
    /// Total usage units.
    pub usage_count: u64,
    /// Total royalty across all tiers.
    pub total_royalty: f64,
    /// Per-tier breakdown.
    pub breakdown: Vec<TierBreakdown>,
    /// Currency code.
    pub currency: String,
}

impl TieredRoyaltyResult {
    /// Effective blended rate.
    pub fn blended_rate(&self) -> Option<f64> {
        if self.usage_count == 0 {
            return None;
        }
        Some(self.total_royalty / self.usage_count as f64)
    }
}

// ── TieredAgreement ─────────────────────────────────────────────────────────

/// An agreement that uses tiered royalty rates.
#[derive(Debug, Clone)]
pub struct TieredAgreement {
    /// Unique agreement ID.
    pub id: String,
    /// Asset covered.
    pub asset_id: String,
    /// Rights holder receiving royalties.
    pub rights_holder: String,
    /// The tiered schedule.
    pub schedule: TieredRoyaltySchedule,
    /// Cumulative usage tracked so far in the current period.
    pub cumulative_usage: u64,
    /// Unix timestamp when the agreement becomes effective.
    pub valid_from: u64,
    /// Unix timestamp when the agreement expires.
    pub valid_until: Option<u64>,
}

impl TieredAgreement {
    /// Create a new tiered agreement.
    pub fn new(
        id: &str,
        asset_id: &str,
        rights_holder: &str,
        schedule: TieredRoyaltySchedule,
        valid_from: u64,
        valid_until: Option<u64>,
    ) -> Self {
        Self {
            id: id.to_string(),
            asset_id: asset_id.to_string(),
            rights_holder: rights_holder.to_string(),
            schedule,
            cumulative_usage: 0,
            valid_from,
            valid_until,
        }
    }

    /// Add usage and return the incremental royalty owed.
    pub fn add_usage(&mut self, units: u64) -> f64 {
        let before = self.schedule.calculate(self.cumulative_usage);
        self.cumulative_usage = self.cumulative_usage.saturating_add(units);
        let after = self.schedule.calculate(self.cumulative_usage);
        after.total_royalty - before.total_royalty
    }

    /// Reset cumulative usage (e.g. at the start of a new reporting period).
    pub fn reset_usage(&mut self) {
        self.cumulative_usage = 0;
    }

    /// Current tier based on cumulative usage.
    pub fn current_tier(&self) -> Option<&RoyaltyTier> {
        self.schedule.tier_for(self.cumulative_usage)
    }

    /// Return `true` if this agreement is active at `ts`.
    pub fn is_active_at(&self, ts: u64) -> bool {
        if ts < self.valid_from {
            return false;
        }
        match self.valid_until {
            Some(end) => ts <= end,
            None => true,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn three_tier_schedule() -> TieredRoyaltySchedule {
        let mut s = TieredRoyaltySchedule::new("USD");
        s.add_tier(RoyaltyTier::new("Base", 0, Some(10_000), 0.004));
        s.add_tier(RoyaltyTier::new("Silver", 10_000, Some(100_000), 0.003));
        s.add_tier(RoyaltyTier::new("Gold", 100_000, None, 0.002));
        s
    }

    // ── RoyaltyTier ─────────────────────────────────────────────────────────

    #[test]
    fn test_tier_capacity_bounded() {
        let t = RoyaltyTier::new("Base", 0, Some(10_000), 0.004);
        assert_eq!(t.capacity(), Some(10_000));
    }

    #[test]
    fn test_tier_capacity_unbounded() {
        let t = RoyaltyTier::new("Top", 100_000, None, 0.002);
        assert!(t.capacity().is_none());
    }

    #[test]
    fn test_tier_contains_within() {
        let t = RoyaltyTier::new("Base", 0, Some(10_000), 0.004);
        assert!(t.contains(0));
        assert!(t.contains(5_000));
        assert!(t.contains(9_999));
        assert!(!t.contains(10_000)); // exclusive upper bound
    }

    #[test]
    fn test_tier_contains_unbounded() {
        let t = RoyaltyTier::new("Top", 100_000, None, 0.002);
        assert!(t.contains(100_000));
        assert!(t.contains(999_999));
        assert!(!t.contains(99_999));
    }

    #[test]
    fn test_tier_calculate_portion_base() {
        let t = RoyaltyTier::new("Base", 0, Some(10_000), 0.004);
        let (units, amount) = t.calculate_portion(5_000);
        assert_eq!(units, 5_000);
        assert!((amount - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_tier_calculate_portion_full() {
        let t = RoyaltyTier::new("Base", 0, Some(10_000), 0.004);
        let (units, amount) = t.calculate_portion(50_000);
        assert_eq!(units, 10_000);
        assert!((amount - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_tier_calculate_portion_zero_below_threshold() {
        let t = RoyaltyTier::new("Silver", 10_000, Some(100_000), 0.003);
        let (units, amount) = t.calculate_portion(5_000);
        assert_eq!(units, 0);
        assert!(amount.abs() < 1e-9);
    }

    #[test]
    fn test_tier_calculate_portion_partial() {
        let t = RoyaltyTier::new("Silver", 10_000, Some(100_000), 0.003);
        let (units, amount) = t.calculate_portion(50_000);
        assert_eq!(units, 40_000);
        assert!((amount - 120.0).abs() < 1e-9);
    }

    // ── TieredRoyaltySchedule ───────────────────────────────────────────────

    #[test]
    fn test_schedule_validate_contiguous() {
        let s = three_tier_schedule();
        assert!(s.validate().is_ok());
    }

    #[test]
    fn test_schedule_validate_gap() {
        let mut s = TieredRoyaltySchedule::new("USD");
        s.add_tier(RoyaltyTier::new("A", 0, Some(100), 0.01));
        s.add_tier(RoyaltyTier::new("B", 200, None, 0.005));
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_schedule_validate_empty() {
        let s = TieredRoyaltySchedule::new("USD");
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_schedule_validate_unbounded_not_last() {
        let mut s = TieredRoyaltySchedule::new("USD");
        s.add_tier(RoyaltyTier::new("A", 0, None, 0.01));
        s.add_tier(RoyaltyTier::new("B", 100, None, 0.005));
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_schedule_tier_count() {
        let s = three_tier_schedule();
        assert_eq!(s.tier_count(), 3);
    }

    #[test]
    fn test_schedule_tier_for() {
        let s = three_tier_schedule();
        assert_eq!(s.tier_for(500).map(|t| t.label.as_str()), Some("Base"));
        assert_eq!(s.tier_for(50_000).map(|t| t.label.as_str()), Some("Silver"));
        assert_eq!(s.tier_for(200_000).map(|t| t.label.as_str()), Some("Gold"));
    }

    #[test]
    fn test_schedule_calculate_within_first_tier() {
        let s = three_tier_schedule();
        let result = s.calculate(5_000);
        // 5000 * 0.004 = 20.0
        assert!((result.total_royalty - 20.0).abs() < 1e-9);
        assert_eq!(result.breakdown.len(), 1);
    }

    #[test]
    fn test_schedule_calculate_spanning_two_tiers() {
        let s = three_tier_schedule();
        let result = s.calculate(50_000);
        // 10,000 * 0.004 = 40.0 (Base)
        // 40,000 * 0.003 = 120.0 (Silver)
        // Total = 160.0
        assert!((result.total_royalty - 160.0).abs() < 1e-9);
        assert_eq!(result.breakdown.len(), 2);
    }

    #[test]
    fn test_schedule_calculate_spanning_all_tiers() {
        let s = three_tier_schedule();
        let result = s.calculate(200_000);
        // 10,000 * 0.004 = 40.0 (Base)
        // 90,000 * 0.003 = 270.0 (Silver)
        // 100,000 * 0.002 = 200.0 (Gold)
        // Total = 510.0
        assert!((result.total_royalty - 510.0).abs() < 1e-9);
        assert_eq!(result.breakdown.len(), 3);
    }

    #[test]
    fn test_schedule_calculate_zero_usage() {
        let s = three_tier_schedule();
        let result = s.calculate(0);
        assert!(result.total_royalty.abs() < 1e-9);
        assert!(result.breakdown.is_empty());
    }

    #[test]
    fn test_schedule_blended_rate() {
        let s = three_tier_schedule();
        // 50,000 units → total 160 → blended = 0.0032
        let rate = s.blended_rate(50_000);
        assert!(rate.is_some());
        assert!((rate.expect("rate should be Some") - 0.0032).abs() < 1e-9);
    }

    #[test]
    fn test_schedule_blended_rate_zero() {
        let s = three_tier_schedule();
        assert!(s.blended_rate(0).is_none());
    }

    #[test]
    fn test_schedule_with_description() {
        let s = TieredRoyaltySchedule::new("EUR").with_description("streaming rates");
        assert_eq!(s.description, "streaming rates");
    }

    // ── TieredRoyaltyResult ─────────────────────────────────────────────────

    #[test]
    fn test_result_blended_rate() {
        let s = three_tier_schedule();
        let result = s.calculate(200_000);
        let blended = result.blended_rate();
        assert!(blended.is_some());
        assert!((blended.expect("blended") - 0.00255).abs() < 1e-9);
    }

    #[test]
    fn test_result_blended_rate_zero() {
        let result = TieredRoyaltyResult {
            usage_count: 0,
            total_royalty: 0.0,
            breakdown: vec![],
            currency: "USD".to_string(),
        };
        assert!(result.blended_rate().is_none());
    }

    // ── TieredAgreement ─────────────────────────────────────────────────────

    #[test]
    fn test_agreement_add_usage_incremental() {
        let schedule = three_tier_schedule();
        let mut agr = TieredAgreement::new("a1", "asset-1", "Alice", schedule, 0, None);

        // First 5000: 5000 * 0.004 = 20.0
        let inc1 = agr.add_usage(5_000);
        assert!((inc1 - 20.0).abs() < 1e-9);

        // Next 5000 (cumulative 10000): fills rest of Base tier
        // 5000 * 0.004 = 20.0
        let inc2 = agr.add_usage(5_000);
        assert!((inc2 - 20.0).abs() < 1e-9);

        // Next 10000 (cumulative 20000): enters Silver tier
        // 10000 * 0.003 = 30.0
        let inc3 = agr.add_usage(10_000);
        assert!((inc3 - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_agreement_reset_usage() {
        let schedule = three_tier_schedule();
        let mut agr = TieredAgreement::new("a1", "asset-1", "Alice", schedule, 0, None);
        agr.add_usage(50_000);
        agr.reset_usage();
        assert_eq!(agr.cumulative_usage, 0);
        // After reset, rate should be Base again
        assert_eq!(agr.current_tier().map(|t| t.label.as_str()), Some("Base"));
    }

    #[test]
    fn test_agreement_current_tier() {
        let schedule = three_tier_schedule();
        let mut agr = TieredAgreement::new("a1", "asset-1", "Alice", schedule, 0, None);
        assert_eq!(agr.current_tier().map(|t| t.label.as_str()), Some("Base"));
        agr.add_usage(50_000);
        assert_eq!(agr.current_tier().map(|t| t.label.as_str()), Some("Silver"));
        agr.add_usage(100_000);
        assert_eq!(agr.current_tier().map(|t| t.label.as_str()), Some("Gold"));
    }

    #[test]
    fn test_agreement_is_active_at() {
        let schedule = three_tier_schedule();
        let agr = TieredAgreement::new("a1", "asset-1", "Alice", schedule, 1000, Some(2000));
        assert!(!agr.is_active_at(500));
        assert!(agr.is_active_at(1500));
        assert!(agr.is_active_at(2000));
        assert!(!agr.is_active_at(2001));
    }

    #[test]
    fn test_agreement_is_active_no_expiry() {
        let schedule = three_tier_schedule();
        let agr = TieredAgreement::new("a1", "asset-1", "Alice", schedule, 0, None);
        assert!(agr.is_active_at(999_999));
    }

    #[test]
    fn test_schedule_single_tier() {
        let mut s = TieredRoyaltySchedule::new("GBP");
        s.add_tier(RoyaltyTier::new("Flat", 0, None, 0.005));
        assert!(s.validate().is_ok());
        let result = s.calculate(100_000);
        assert!((result.total_royalty - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_tier_at_exact_boundary() {
        let s = three_tier_schedule();
        // At exactly 10,000: should be in Silver (Base is [0, 10000))
        assert_eq!(s.tier_for(10_000).map(|t| t.label.as_str()), Some("Silver"));
    }

    #[test]
    fn test_agreement_add_usage_crosses_multiple_tiers_at_once() {
        let schedule = three_tier_schedule();
        let mut agr = TieredAgreement::new("a1", "asset-1", "Alice", schedule, 0, None);
        // Add 200_000 units in one shot
        let inc = agr.add_usage(200_000);
        // Should equal full schedule calculation
        assert!((inc - 510.0).abs() < 1e-9);
    }
}

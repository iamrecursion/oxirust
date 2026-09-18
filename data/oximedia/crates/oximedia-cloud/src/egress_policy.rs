#![allow(dead_code)]
//! Cloud egress cost tracking and policy enforcement.

use std::collections::HashMap;

/// Egress pricing tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EgressTier {
    /// First 100 GB / month – free for many providers.
    Free,
    /// Standard egress (e.g., 0.08–0.09 USD/GB).
    Standard,
    /// Premium accelerated egress (e.g., 0.14 USD/GB).
    Premium,
}

impl EgressTier {
    /// Returns the cost per GB in US dollars for this tier.
    #[allow(clippy::cast_precision_loss)]
    pub fn cost_per_gb_usd(&self) -> f64 {
        match self {
            Self::Free => 0.0,
            Self::Standard => 0.085,
            Self::Premium => 0.140,
        }
    }
}

/// Policy controlling which egress tier applies and optional monthly caps.
#[derive(Debug, Clone)]
pub struct EgressPolicy {
    pub default_tier: EgressTier,
    /// Monthly free-tier allowance in GB (0 = none).
    pub free_allowance_gb: f64,
    /// Hard cap in GB per month; `None` means unlimited.
    pub monthly_cap_gb: Option<f64>,
}

impl EgressPolicy {
    /// Create an `EgressPolicy` with the given tier and free allowance.
    pub fn new(default_tier: EgressTier, free_allowance_gb: f64) -> Self {
        Self {
            default_tier,
            free_allowance_gb,
            monthly_cap_gb: None,
        }
    }

    /// Set a hard monthly cap in GB.
    pub fn with_cap(mut self, cap_gb: f64) -> Self {
        self.monthly_cap_gb = Some(cap_gb);
        self
    }

    /// Estimate the cost in USD for transferring `bytes` of data.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_cost(&self, bytes: u64, already_transferred_gb: f64) -> f64 {
        let gb = bytes as f64 / 1_073_741_824.0;
        let free_remaining = (self.free_allowance_gb - already_transferred_gb).max(0.0);
        let billable_gb = (gb - free_remaining).max(0.0);
        billable_gb * self.default_tier.cost_per_gb_usd()
    }
}

/// Records egress transfers and tracks accumulated cost.
#[derive(Debug, Default)]
pub struct EgressMonitor {
    /// Map from billing period key (e.g. "2025-03") to bytes transferred.
    transfers: HashMap<String, u64>,
    policy: Option<EgressPolicy>,
}

impl EgressMonitor {
    /// Create a new monitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach an egress policy for cost calculation.
    pub fn with_policy(mut self, policy: EgressPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Record a transfer of `bytes` bytes for the given billing period key.
    pub fn record_transfer(&mut self, period: &str, bytes: u64) {
        *self.transfers.entry(period.to_string()).or_insert(0) += bytes;
    }

    /// Return the total bytes transferred in a billing period.
    pub fn period_bytes(&self, period: &str) -> u64 {
        self.transfers.get(period).copied().unwrap_or(0)
    }

    /// Estimate the monthly cost in USD for the given billing period.
    #[allow(clippy::cast_precision_loss)]
    pub fn monthly_cost_usd(&self, period: &str) -> f64 {
        let bytes = self.period_bytes(period);
        let gb = bytes as f64 / 1_073_741_824.0;
        match &self.policy {
            None => {
                // Default: standard tier, no free allowance.
                gb * EgressTier::Standard.cost_per_gb_usd()
            }
            Some(p) => p.estimate_cost(bytes, 0.0),
        }
    }

    /// Returns all tracked periods.
    pub fn periods(&self) -> Vec<&str> {
        self.transfers.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_tier_cost() {
        assert_eq!(EgressTier::Free.cost_per_gb_usd(), 0.0);
    }

    #[test]
    fn test_standard_tier_cost() {
        let c = EgressTier::Standard.cost_per_gb_usd();
        assert!(c > 0.0 && c < 0.15);
    }

    #[test]
    fn test_premium_tier_cost() {
        assert!(EgressTier::Premium.cost_per_gb_usd() > EgressTier::Standard.cost_per_gb_usd());
    }

    #[test]
    fn test_egress_policy_new() {
        let p = EgressPolicy::new(EgressTier::Standard, 100.0);
        assert_eq!(p.free_allowance_gb, 100.0);
        assert!(p.monthly_cap_gb.is_none());
    }

    #[test]
    fn test_egress_policy_with_cap() {
        let p = EgressPolicy::new(EgressTier::Standard, 0.0).with_cap(500.0);
        assert_eq!(p.monthly_cap_gb, Some(500.0));
    }

    #[test]
    fn test_estimate_cost_within_free_allowance() {
        let p = EgressPolicy::new(EgressTier::Standard, 100.0);
        // 10 GB, 0 already transferred → within free tier → cost 0
        let cost = p.estimate_cost(10 * 1_073_741_824, 0.0);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_estimate_cost_exceeds_free_allowance() {
        let p = EgressPolicy::new(EgressTier::Standard, 0.0);
        let bytes_1gb = 1_073_741_824u64;
        let cost = p.estimate_cost(bytes_1gb, 0.0);
        assert!((cost - 0.085).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_cost_partial_free() {
        let p = EgressPolicy::new(EgressTier::Standard, 1.0);
        let bytes_2gb = 2 * 1_073_741_824u64;
        // 2 GB transferred, 1 GB free remaining → 1 GB billed
        let cost = p.estimate_cost(bytes_2gb, 0.0);
        assert!((cost - 0.085).abs() < 1e-6);
    }

    #[test]
    fn test_monitor_record_transfer() {
        let mut mon = EgressMonitor::new();
        mon.record_transfer("2025-03", 1_000_000);
        assert_eq!(mon.period_bytes("2025-03"), 1_000_000);
    }

    #[test]
    fn test_monitor_accumulates_transfers() {
        let mut mon = EgressMonitor::new();
        mon.record_transfer("2025-03", 500_000);
        mon.record_transfer("2025-03", 500_000);
        assert_eq!(mon.period_bytes("2025-03"), 1_000_000);
    }

    #[test]
    fn test_monitor_zero_unknown_period() {
        let mon = EgressMonitor::new();
        assert_eq!(mon.period_bytes("2025-01"), 0);
    }

    #[test]
    fn test_monthly_cost_no_policy() {
        let mut mon = EgressMonitor::new();
        // 1 GB at standard rate
        mon.record_transfer("2025-03", 1_073_741_824);
        let cost = mon.monthly_cost_usd("2025-03");
        assert!((cost - 0.085).abs() < 1e-6);
    }

    #[test]
    fn test_monthly_cost_with_policy() {
        let policy = EgressPolicy::new(EgressTier::Premium, 0.0);
        let mut mon = EgressMonitor::new().with_policy(policy);
        mon.record_transfer("2025-03", 1_073_741_824);
        let cost = mon.monthly_cost_usd("2025-03");
        assert!((cost - 0.140).abs() < 1e-6);
    }

    #[test]
    fn test_monitor_periods_list() {
        let mut mon = EgressMonitor::new();
        mon.record_transfer("2025-01", 1000);
        mon.record_transfer("2025-02", 2000);
        let mut periods = mon.periods();
        periods.sort_unstable();
        assert_eq!(periods, vec!["2025-01", "2025-02"]);
    }

    #[test]
    fn test_monthly_cost_empty_period() {
        let mon = EgressMonitor::new();
        assert_eq!(mon.monthly_cost_usd("2025-03"), 0.0);
    }
}

//! Cloud cost modeling for media workloads.
//!
//! Provides per-provider cost estimates, breakdowns, tier optimization,
//! and reserved vs. on-demand savings analysis.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Cloud provider identifier for pricing lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloudProvider {
    /// Amazon Web Services
    Aws,
    /// Microsoft Azure
    Azure,
    /// Google Cloud Platform
    Gcp,
    /// Alibaba Cloud
    Alibaba,
}

impl CloudProvider {
    /// Human-readable name for this provider.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            CloudProvider::Aws => "AWS",
            CloudProvider::Azure => "Azure",
            CloudProvider::Gcp => "GCP",
            CloudProvider::Alibaba => "Alibaba Cloud",
        }
    }

    /// Approximate standard storage cost per GB per month (USD).
    #[must_use]
    pub fn storage_cost_per_gb_month(&self) -> f64 {
        match self {
            CloudProvider::Aws => 0.023,
            CloudProvider::Azure => 0.018,
            CloudProvider::Gcp => 0.020,
            CloudProvider::Alibaba => 0.016,
        }
    }
}

/// Per-unit cost rates for a cloud workload.
#[derive(Debug, Clone)]
pub struct ResourceCost {
    /// Compute cost in USD per hour.
    pub compute_per_hour: f64,
    /// Storage cost in USD per GB per month.
    pub storage_per_gb_month: f64,
    /// Egress cost in USD per GB transferred out.
    pub egress_per_gb: f64,
    /// Transcoding cost in USD per minute of media.
    pub transcode_per_min: f64,
}

impl ResourceCost {
    /// Estimate the total monthly cost given usage figures.
    ///
    /// - `compute_hours`: compute hours used per month
    /// - `storage_gb`: storage in GB
    /// - `egress_gb`: egress in GB per month
    /// - `transcode_mins`: transcoding minutes per month
    #[must_use]
    pub fn monthly_estimate(
        &self,
        compute_hours: f64,
        storage_gb: f64,
        egress_gb: f64,
        transcode_mins: f64,
    ) -> f64 {
        self.compute_per_hour * compute_hours
            + self.storage_per_gb_month * storage_gb
            + self.egress_per_gb * egress_gb
            + self.transcode_per_min * transcode_mins
    }
}

/// Breakdown of a cost estimate into individual components.
#[derive(Debug, Clone)]
pub struct CostBreakdown {
    /// Compute cost in USD.
    pub compute: f64,
    /// Storage cost in USD.
    pub storage: f64,
    /// Egress cost in USD.
    pub egress: f64,
    /// Transcoding cost in USD.
    pub transcode: f64,
}

impl CostBreakdown {
    /// Sum of all cost components.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.compute + self.storage + self.egress + self.transcode
    }

    /// Name of the component with the highest cost.
    ///
    /// Returns `"compute"`, `"storage"`, `"egress"`, or `"transcode"`.
    #[must_use]
    pub fn dominant_cost(&self) -> &str {
        let components = [
            (self.compute, "compute"),
            (self.storage, "storage"),
            (self.egress, "egress"),
            (self.transcode, "transcode"),
        ];
        components
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, name)| *name)
            .unwrap_or("compute")
    }
}

/// Recommends storage tiers based on access patterns.
pub struct CostOptimizer;

impl CostOptimizer {
    /// Recommend a storage tier given monthly access count and object size.
    ///
    /// Returns `"hot"`, `"warm"`, or `"cold"`.
    ///
    /// | access_count / size_gb | tier  |
    /// |------------------------|-------|
    /// | > 100 or size < 1      | hot   |
    /// | 10 – 100               | warm  |
    /// | < 10                   | cold  |
    #[must_use]
    pub fn recommend_tier(monthly_access_count: u64, size_gb: f64) -> &'static str {
        if monthly_access_count > 100 || size_gb < 1.0 {
            "hot"
        } else if monthly_access_count >= 10 {
            "warm"
        } else {
            "cold"
        }
    }
}

/// Compares reserved vs. on-demand pricing.
pub struct ReservedVsOnDemand;

impl ReservedVsOnDemand {
    /// Percentage savings of reserving capacity for a year compared to
    /// paying on-demand month-by-month.
    ///
    /// `reserved_annual` is the up-front annual cost; `on_demand_monthly` is
    /// the per-month rate. Returns 0.0 when `on_demand_monthly` is zero.
    #[must_use]
    pub fn savings_pct(reserved_annual: f64, on_demand_monthly: f64) -> f64 {
        let on_demand_annual = on_demand_monthly * 12.0;
        if on_demand_annual <= 0.0 {
            return 0.0;
        }
        ((on_demand_annual - reserved_annual) / on_demand_annual * 100.0).max(0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. CloudProvider::name()
    #[test]
    fn test_provider_names() {
        assert_eq!(CloudProvider::Aws.name(), "AWS");
        assert_eq!(CloudProvider::Azure.name(), "Azure");
        assert_eq!(CloudProvider::Gcp.name(), "GCP");
        assert_eq!(CloudProvider::Alibaba.name(), "Alibaba Cloud");
    }

    // 2. CloudProvider::storage_cost_per_gb_month()
    #[test]
    fn test_provider_storage_cost_positive() {
        for provider in [
            CloudProvider::Aws,
            CloudProvider::Azure,
            CloudProvider::Gcp,
            CloudProvider::Alibaba,
        ] {
            assert!(provider.storage_cost_per_gb_month() > 0.0);
        }
    }

    // 3. AWS storage cost value
    #[test]
    fn test_aws_storage_cost_value() {
        assert!((CloudProvider::Aws.storage_cost_per_gb_month() - 0.023).abs() < 1e-9);
    }

    // 4. ResourceCost::monthly_estimate – zero usage
    #[test]
    fn test_resource_cost_zero_usage() {
        let cost = ResourceCost {
            compute_per_hour: 0.10,
            storage_per_gb_month: 0.023,
            egress_per_gb: 0.09,
            transcode_per_min: 0.02,
        };
        assert!((cost.monthly_estimate(0.0, 0.0, 0.0, 0.0)).abs() < 1e-9);
    }

    // 5. ResourceCost::monthly_estimate – known values
    #[test]
    fn test_resource_cost_monthly_estimate() {
        let cost = ResourceCost {
            compute_per_hour: 1.0,
            storage_per_gb_month: 1.0,
            egress_per_gb: 1.0,
            transcode_per_min: 1.0,
        };
        // 10h compute + 20 GB storage + 5 GB egress + 2 min transcode = 37
        let total = cost.monthly_estimate(10.0, 20.0, 5.0, 2.0);
        assert!((total - 37.0).abs() < 1e-9);
    }

    // 6. CostBreakdown::total()
    #[test]
    fn test_cost_breakdown_total() {
        let bd = CostBreakdown {
            compute: 10.0,
            storage: 20.0,
            egress: 5.0,
            transcode: 3.0,
        };
        assert!((bd.total() - 38.0).abs() < 1e-9);
    }

    // 7. CostBreakdown::dominant_cost()
    #[test]
    fn test_cost_breakdown_dominant_storage() {
        let bd = CostBreakdown {
            compute: 5.0,
            storage: 50.0,
            egress: 3.0,
            transcode: 1.0,
        };
        assert_eq!(bd.dominant_cost(), "storage");
    }

    // 8. CostBreakdown::dominant_cost() compute wins
    #[test]
    fn test_cost_breakdown_dominant_compute() {
        let bd = CostBreakdown {
            compute: 100.0,
            storage: 10.0,
            egress: 2.0,
            transcode: 5.0,
        };
        assert_eq!(bd.dominant_cost(), "compute");
    }

    // 9. CostBreakdown::dominant_cost() egress wins
    #[test]
    fn test_cost_breakdown_dominant_egress() {
        let bd = CostBreakdown {
            compute: 1.0,
            storage: 2.0,
            egress: 99.0,
            transcode: 3.0,
        };
        assert_eq!(bd.dominant_cost(), "egress");
    }

    // 10. CostOptimizer::recommend_tier – hot (high access)
    #[test]
    fn test_recommend_tier_hot_high_access() {
        assert_eq!(CostOptimizer::recommend_tier(200, 100.0), "hot");
    }

    // 11. CostOptimizer::recommend_tier – hot (small object)
    #[test]
    fn test_recommend_tier_hot_small_object() {
        assert_eq!(CostOptimizer::recommend_tier(5, 0.5), "hot");
    }

    // 12. CostOptimizer::recommend_tier – warm
    #[test]
    fn test_recommend_tier_warm() {
        assert_eq!(CostOptimizer::recommend_tier(50, 10.0), "warm");
    }

    // 13. CostOptimizer::recommend_tier – cold
    #[test]
    fn test_recommend_tier_cold() {
        assert_eq!(CostOptimizer::recommend_tier(3, 5.0), "cold");
    }

    // 14. ReservedVsOnDemand::savings_pct – basic
    #[test]
    fn test_reserved_savings_pct_basic() {
        // On-demand: $100/month * 12 = $1200/year
        // Reserved: $720/year → 40% savings
        let pct = ReservedVsOnDemand::savings_pct(720.0, 100.0);
        assert!((pct - 40.0).abs() < 1e-6);
    }

    // 15. ReservedVsOnDemand::savings_pct – zero on-demand returns 0
    #[test]
    fn test_reserved_savings_pct_zero_on_demand() {
        let pct = ReservedVsOnDemand::savings_pct(0.0, 0.0);
        assert!((pct).abs() < 1e-9);
    }

    // 16. ReservedVsOnDemand::savings_pct – reserved more expensive clamps to 0
    #[test]
    fn test_reserved_savings_pct_no_negative() {
        // Reserved more expensive than on-demand
        let pct = ReservedVsOnDemand::savings_pct(2000.0, 100.0);
        assert!(pct >= 0.0);
    }
}

//! Cost optimization and estimation

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::StorageClass;

/// Storage tier for cost optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageTier {
    /// Hot tier - frequently accessed
    Hot,
    /// Cool tier - infrequently accessed
    Cool,
    /// Archive tier - rarely accessed
    Archive,
    /// Cold archive tier - very rarely accessed
    ColdArchive,
}

impl StorageTier {
    /// Get recommended storage class for AWS S3
    #[must_use]
    pub fn to_s3_storage_class(&self) -> StorageClass {
        match self {
            StorageTier::Hot => StorageClass::Standard,
            StorageTier::Cool => StorageClass::InfrequentAccess,
            StorageTier::Archive => StorageClass::Glacier,
            StorageTier::ColdArchive => StorageClass::DeepArchive,
        }
    }

    /// Get cost multiplier relative to hot tier
    #[must_use]
    pub fn cost_multiplier(&self) -> f64 {
        match self {
            StorageTier::Hot => 1.0,
            StorageTier::Cool => 0.5,
            StorageTier::Archive => 0.1,
            StorageTier::ColdArchive => 0.05,
        }
    }

    /// Get retrieval cost multiplier
    #[must_use]
    pub fn retrieval_multiplier(&self) -> f64 {
        match self {
            StorageTier::Hot => 1.0,
            StorageTier::Cool => 2.0,
            StorageTier::Archive => 10.0,
            StorageTier::ColdArchive => 20.0,
        }
    }
}

/// Cost estimator for cloud storage
pub struct CostEstimator {
    /// Pricing by region
    pricing: HashMap<String, RegionPricing>,
}

impl CostEstimator {
    /// Create a new cost estimator
    #[must_use]
    pub fn new() -> Self {
        let mut pricing = HashMap::new();

        // AWS S3 pricing (approximate, as of 2024)
        pricing.insert("us-east-1".to_string(), RegionPricing::aws_s3_default());
        pricing.insert("us-west-2".to_string(), RegionPricing::aws_s3_default());
        pricing.insert("eu-west-1".to_string(), RegionPricing::aws_s3_eu());

        Self { pricing }
    }

    /// Estimate monthly storage cost
    #[must_use]
    pub fn estimate_storage_cost(
        &self,
        region: &str,
        size_gb: f64,
        storage_class: StorageClass,
    ) -> f64 {
        let default_pricing = RegionPricing::aws_s3_default();
        let pricing = self.pricing.get(region).unwrap_or(&default_pricing);

        let base_cost = match storage_class {
            StorageClass::Standard => pricing.standard_storage_per_gb,
            StorageClass::InfrequentAccess => pricing.ia_storage_per_gb,
            StorageClass::Glacier => pricing.glacier_storage_per_gb,
            StorageClass::DeepArchive => pricing.deep_archive_storage_per_gb,
            StorageClass::IntelligentTiering => pricing.intelligent_tiering_per_gb,
            StorageClass::OneZoneIA => pricing.onezone_ia_storage_per_gb,
            StorageClass::ReducedRedundancy => pricing.standard_storage_per_gb * 0.9,
        };

        size_gb * base_cost
    }

    /// Estimate request cost
    #[must_use]
    pub fn estimate_request_cost(
        &self,
        region: &str,
        put_requests: u64,
        get_requests: u64,
        storage_class: StorageClass,
    ) -> f64 {
        let default_pricing = RegionPricing::aws_s3_default();
        let pricing = self.pricing.get(region).unwrap_or(&default_pricing);

        let put_cost = (put_requests as f64 / 1000.0) * pricing.put_request_per_1k;
        let get_cost = (get_requests as f64 / 1000.0) * pricing.get_request_per_1k;

        let class_multiplier = match storage_class {
            StorageClass::InfrequentAccess | StorageClass::OneZoneIA => 1.5,
            StorageClass::Glacier | StorageClass::DeepArchive => 2.0,
            _ => 1.0,
        };

        (put_cost + get_cost) * class_multiplier
    }

    /// Estimate data transfer cost
    #[must_use]
    pub fn estimate_transfer_cost(&self, region: &str, transfer_gb: f64, outbound: bool) -> f64 {
        if !outbound {
            return 0.0; // Inbound transfer is free
        }

        let default_pricing = RegionPricing::aws_s3_default();
        let pricing = self.pricing.get(region).unwrap_or(&default_pricing);

        // Tiered pricing
        if transfer_gb <= 10.0 {
            transfer_gb * pricing.data_transfer_out_per_gb
        } else if transfer_gb <= 50.0 {
            10.0 * pricing.data_transfer_out_per_gb
                + (transfer_gb - 10.0) * pricing.data_transfer_out_per_gb * 0.9
        } else {
            10.0 * pricing.data_transfer_out_per_gb
                + 40.0 * pricing.data_transfer_out_per_gb * 0.9
                + (transfer_gb - 50.0) * pricing.data_transfer_out_per_gb * 0.8
        }
    }

    /// Recommend optimal storage tier based on access patterns
    #[must_use]
    pub fn recommend_tier(&self, access_pattern: &AccessPattern) -> StorageTier {
        let days_since_access = access_pattern.days_since_last_access();
        let access_frequency = access_pattern.monthly_access_count;

        if days_since_access < 7.0 || access_frequency > 100 {
            StorageTier::Hot
        } else if days_since_access < 30.0 || access_frequency > 10 {
            StorageTier::Cool
        } else if days_since_access < 90.0 {
            StorageTier::Archive
        } else {
            StorageTier::ColdArchive
        }
    }

    /// Calculate potential savings from tiering
    #[must_use]
    pub fn calculate_tiering_savings(
        &self,
        region: &str,
        size_gb: f64,
        current_tier: StorageTier,
        target_tier: StorageTier,
    ) -> f64 {
        let current_class = current_tier.to_s3_storage_class();
        let target_class = target_tier.to_s3_storage_class();

        let current_cost = self.estimate_storage_cost(region, size_gb, current_class);
        let target_cost = self.estimate_storage_cost(region, size_gb, target_class);

        current_cost - target_cost
    }
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Regional pricing information
#[derive(Debug, Clone)]
pub struct RegionPricing {
    /// Standard storage per GB per month
    pub standard_storage_per_gb: f64,
    /// Infrequent Access storage per GB per month
    pub ia_storage_per_gb: f64,
    /// One Zone IA storage per GB per month
    pub onezone_ia_storage_per_gb: f64,
    /// Glacier storage per GB per month
    pub glacier_storage_per_gb: f64,
    /// Deep Archive storage per GB per month
    pub deep_archive_storage_per_gb: f64,
    /// Intelligent Tiering per GB per month
    pub intelligent_tiering_per_gb: f64,
    /// PUT request cost per 1000 requests
    pub put_request_per_1k: f64,
    /// GET request cost per 1000 requests
    pub get_request_per_1k: f64,
    /// Data transfer out per GB
    pub data_transfer_out_per_gb: f64,
}

impl RegionPricing {
    /// AWS S3 default (US East) pricing
    #[must_use]
    pub fn aws_s3_default() -> Self {
        Self {
            standard_storage_per_gb: 0.023,
            ia_storage_per_gb: 0.0125,
            onezone_ia_storage_per_gb: 0.01,
            glacier_storage_per_gb: 0.004,
            deep_archive_storage_per_gb: 0.00099,
            intelligent_tiering_per_gb: 0.023,
            put_request_per_1k: 0.005,
            get_request_per_1k: 0.0004,
            data_transfer_out_per_gb: 0.09,
        }
    }

    /// AWS S3 EU pricing
    #[must_use]
    pub fn aws_s3_eu() -> Self {
        Self {
            standard_storage_per_gb: 0.024,
            ia_storage_per_gb: 0.0125,
            onezone_ia_storage_per_gb: 0.01,
            glacier_storage_per_gb: 0.0045,
            deep_archive_storage_per_gb: 0.00099,
            intelligent_tiering_per_gb: 0.024,
            put_request_per_1k: 0.005,
            get_request_per_1k: 0.0004,
            data_transfer_out_per_gb: 0.09,
        }
    }
}

/// Access pattern information for cost optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,
    /// Number of accesses in the last month
    pub monthly_access_count: u64,
    /// Average access size in bytes
    pub avg_access_size: u64,
    /// Total size in bytes
    pub total_size: u64,
}

impl AccessPattern {
    /// Calculate days since last access
    #[must_use]
    pub fn days_since_last_access(&self) -> f64 {
        let now = Utc::now();
        let duration = now.signed_duration_since(self.last_accessed);
        duration.num_days() as f64
    }

    /// Check if object is frequently accessed
    #[must_use]
    pub fn is_frequently_accessed(&self) -> bool {
        self.monthly_access_count > 50 || self.days_since_last_access() < 7.0
    }

    /// Check if object is rarely accessed
    #[must_use]
    pub fn is_rarely_accessed(&self) -> bool {
        self.monthly_access_count < 5 && self.days_since_last_access() > 30.0
    }
}

/// Tiering strategy configuration
#[derive(Debug, Clone)]
pub struct TieringStrategy {
    /// Rules for automatic tiering
    pub rules: Vec<TieringRule>,
}

impl TieringStrategy {
    /// Create a new tiering strategy
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a tiering rule
    pub fn add_rule(&mut self, rule: TieringRule) {
        self.rules.push(rule);
    }

    /// Get recommended tier for an object
    #[must_use]
    pub fn recommend(&self, pattern: &AccessPattern) -> Option<StorageTier> {
        for rule in &self.rules {
            if rule.matches(pattern) {
                return Some(rule.target_tier);
            }
        }
        None
    }
}

impl Default for TieringStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Rule for automatic tiering
#[derive(Debug, Clone)]
pub struct TieringRule {
    /// Minimum days since last access
    pub min_days_since_access: Option<f64>,
    /// Maximum monthly access count
    pub max_monthly_accesses: Option<u64>,
    /// Target tier
    pub target_tier: StorageTier,
}

impl TieringRule {
    /// Check if the rule matches the access pattern
    #[must_use]
    pub fn matches(&self, pattern: &AccessPattern) -> bool {
        if let Some(min_days) = self.min_days_since_access {
            if pattern.days_since_last_access() < min_days {
                return false;
            }
        }

        if let Some(max_accesses) = self.max_monthly_accesses {
            if pattern.monthly_access_count > max_accesses {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_storage_tier_cost_multiplier() {
        assert_eq!(StorageTier::Hot.cost_multiplier(), 1.0);
        assert_eq!(StorageTier::Cool.cost_multiplier(), 0.5);
        assert_eq!(StorageTier::Archive.cost_multiplier(), 0.1);
    }

    #[test]
    fn test_cost_estimator_storage() {
        let estimator = CostEstimator::new();
        let cost = estimator.estimate_storage_cost("us-east-1", 100.0, StorageClass::Standard);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_cost_estimator_transfer() {
        let estimator = CostEstimator::new();
        let cost = estimator.estimate_transfer_cost("us-east-1", 5.0, true);
        assert!(cost > 0.0);

        let inbound_cost = estimator.estimate_transfer_cost("us-east-1", 5.0, false);
        assert_eq!(inbound_cost, 0.0);
    }

    #[test]
    fn test_recommend_tier() {
        let estimator = CostEstimator::new();

        let hot_pattern = AccessPattern {
            last_accessed: Utc::now(),
            monthly_access_count: 200,
            avg_access_size: 1024,
            total_size: 1024 * 1024,
        };
        assert_eq!(estimator.recommend_tier(&hot_pattern), StorageTier::Hot);

        let cold_pattern = AccessPattern {
            last_accessed: Utc::now() - Duration::days(100),
            monthly_access_count: 0,
            avg_access_size: 1024,
            total_size: 1024 * 1024,
        };
        assert_eq!(
            estimator.recommend_tier(&cold_pattern),
            StorageTier::ColdArchive
        );
    }

    #[test]
    fn test_access_pattern() {
        let pattern = AccessPattern {
            last_accessed: Utc::now() - Duration::days(5),
            monthly_access_count: 100,
            avg_access_size: 1024,
            total_size: 1024 * 1024,
        };

        assert!(pattern.is_frequently_accessed());
        assert!(!pattern.is_rarely_accessed());
        assert!(pattern.days_since_last_access() >= 5.0);
    }

    #[test]
    fn test_tiering_strategy() {
        let mut strategy = TieringStrategy::new();

        strategy.add_rule(TieringRule {
            min_days_since_access: Some(30.0),
            max_monthly_accesses: Some(5),
            target_tier: StorageTier::Archive,
        });

        let pattern = AccessPattern {
            last_accessed: Utc::now() - Duration::days(40),
            monthly_access_count: 2,
            avg_access_size: 1024,
            total_size: 1024 * 1024,
        };

        assert_eq!(strategy.recommend(&pattern), Some(StorageTier::Archive));
    }
}

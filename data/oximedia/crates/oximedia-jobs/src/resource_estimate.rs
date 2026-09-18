#![allow(dead_code)]
//! Resource estimation — model, compute, and compare resource costs for jobs.

use std::collections::HashMap;

/// The type of resource being estimated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceUnit {
    /// CPU time measured in core-seconds.
    CpuCoreSeconds,
    /// Memory in mebibytes.
    MemoryMiB,
    /// Disk I/O in mebibytes.
    DiskMiB,
    /// Network transfer in mebibytes.
    NetworkMiB,
    /// GPU time measured in device-seconds.
    GpuDeviceSeconds,
}

impl ResourceUnit {
    /// A short human-readable label for this unit.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::CpuCoreSeconds => "CPU core-seconds",
            Self::MemoryMiB => "Memory (MiB)",
            Self::DiskMiB => "Disk I/O (MiB)",
            Self::NetworkMiB => "Network (MiB)",
            Self::GpuDeviceSeconds => "GPU device-seconds",
        }
    }

    /// Returns `true` for resource types that represent time-based consumption.
    #[must_use]
    pub fn is_time_based(&self) -> bool {
        matches!(self, Self::CpuCoreSeconds | Self::GpuDeviceSeconds)
    }
}

/// An estimate of resource consumption for a job.
#[derive(Debug, Clone, Default)]
pub struct ResourceEstimate {
    /// Map of resource unit to estimated quantity (non-negative).
    costs: HashMap<ResourceUnit, f64>,
}

impl ResourceEstimate {
    /// Create an empty estimate.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the estimated cost for a given resource unit.
    /// Negative values are clamped to zero.
    pub fn set(&mut self, unit: ResourceUnit, amount: f64) {
        self.costs.insert(unit, amount.max(0.0));
    }

    /// Return the estimated cost for a resource unit, defaulting to `0.0`.
    #[must_use]
    pub fn get(&self, unit: &ResourceUnit) -> f64 {
        self.costs.get(unit).copied().unwrap_or(0.0)
    }

    /// Sum of all individual resource costs (unit-agnostic scalar).
    #[must_use]
    pub fn total_cost(&self) -> f64 {
        self.costs.values().sum()
    }

    /// Number of resource dimensions that have been set.
    #[must_use]
    pub fn dimension_count(&self) -> usize {
        self.costs.len()
    }

    /// Returns `true` if no resources have been estimated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.costs.is_empty()
    }

    /// Scale all costs by a multiplicative factor.
    #[must_use]
    pub fn scaled_by(&self, factor: f64) -> Self {
        let costs = self
            .costs
            .iter()
            .map(|(k, v)| (k.clone(), (*v * factor).max(0.0)))
            .collect();
        Self { costs }
    }
}

/// Builds resource estimates for jobs based on configurable per-unit rates.
#[derive(Debug, Default)]
pub struct ResourceEstimator {
    /// Per-unit base rates used when no job-specific override is given.
    base_rates: HashMap<ResourceUnit, f64>,
}

impl ResourceEstimator {
    /// Create an estimator with no base rates.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a base rate for a resource unit.
    pub fn set_base_rate(&mut self, unit: ResourceUnit, rate: f64) {
        self.base_rates.insert(unit, rate.max(0.0));
    }

    /// Produce a resource estimate for a job given its duration in seconds
    /// and its weight (a dimensionless scale factor).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimate_for_job(&self, duration_secs: f64, weight: u32) -> ResourceEstimate {
        let mut est = ResourceEstimate::new();
        let scale = weight as f64;
        for (unit, &rate) in &self.base_rates {
            est.set(unit.clone(), rate * duration_secs * scale);
        }
        est
    }

    /// Compare two estimates by their total cost.
    /// Returns `std::cmp::Ordering` based on `a.total_cost()` vs `b.total_cost()`.
    #[must_use]
    pub fn compare_estimates(
        &self,
        a: &ResourceEstimate,
        b: &ResourceEstimate,
    ) -> std::cmp::Ordering {
        a.total_cost()
            .partial_cmp(&b.total_cost())
            .unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Returns `true` if estimate `a` is cheaper than estimate `b`.
    #[must_use]
    pub fn a_is_cheaper(&self, a: &ResourceEstimate, b: &ResourceEstimate) -> bool {
        matches!(self.compare_estimates(a, b), std::cmp::Ordering::Less)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_unit_label_non_empty() {
        let units = [
            ResourceUnit::CpuCoreSeconds,
            ResourceUnit::MemoryMiB,
            ResourceUnit::DiskMiB,
            ResourceUnit::NetworkMiB,
            ResourceUnit::GpuDeviceSeconds,
        ];
        for u in &units {
            assert!(!u.label().is_empty(), "label empty for {:?}", u);
        }
    }

    #[test]
    fn test_resource_unit_is_time_based() {
        assert!(ResourceUnit::CpuCoreSeconds.is_time_based());
        assert!(ResourceUnit::GpuDeviceSeconds.is_time_based());
        assert!(!ResourceUnit::MemoryMiB.is_time_based());
        assert!(!ResourceUnit::DiskMiB.is_time_based());
        assert!(!ResourceUnit::NetworkMiB.is_time_based());
    }

    #[test]
    fn test_resource_estimate_set_and_get() {
        let mut e = ResourceEstimate::new();
        e.set(ResourceUnit::CpuCoreSeconds, 10.0);
        assert!((e.get(&ResourceUnit::CpuCoreSeconds) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_resource_estimate_get_missing_is_zero() {
        let e = ResourceEstimate::new();
        assert_eq!(e.get(&ResourceUnit::MemoryMiB), 0.0);
    }

    #[test]
    fn test_resource_estimate_negative_clamped() {
        let mut e = ResourceEstimate::new();
        e.set(ResourceUnit::DiskMiB, -5.0);
        assert_eq!(e.get(&ResourceUnit::DiskMiB), 0.0);
    }

    #[test]
    fn test_resource_estimate_total_cost() {
        let mut e = ResourceEstimate::new();
        e.set(ResourceUnit::CpuCoreSeconds, 3.0);
        e.set(ResourceUnit::MemoryMiB, 7.0);
        assert!((e.total_cost() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_resource_estimate_dimension_count() {
        let mut e = ResourceEstimate::new();
        assert_eq!(e.dimension_count(), 0);
        e.set(ResourceUnit::NetworkMiB, 1.0);
        assert_eq!(e.dimension_count(), 1);
    }

    #[test]
    fn test_resource_estimate_is_empty() {
        let e = ResourceEstimate::new();
        assert!(e.is_empty());
    }

    #[test]
    fn test_resource_estimate_scaled_by() {
        let mut e = ResourceEstimate::new();
        e.set(ResourceUnit::CpuCoreSeconds, 4.0);
        let scaled = e.scaled_by(2.5);
        assert!((scaled.get(&ResourceUnit::CpuCoreSeconds) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_resource_estimator_estimate_for_job() {
        let mut est = ResourceEstimator::new();
        est.set_base_rate(ResourceUnit::CpuCoreSeconds, 2.0);
        let result = est.estimate_for_job(5.0, 3);
        // 2.0 * 5.0 * 3 = 30.0
        assert!((result.get(&ResourceUnit::CpuCoreSeconds) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_resource_estimator_compare_estimates_less() {
        let est = ResourceEstimator::new();
        let mut a = ResourceEstimate::new();
        a.set(ResourceUnit::MemoryMiB, 1.0);
        let mut b = ResourceEstimate::new();
        b.set(ResourceUnit::MemoryMiB, 5.0);
        assert_eq!(est.compare_estimates(&a, &b), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_resource_estimator_compare_estimates_equal() {
        let est = ResourceEstimator::new();
        let mut a = ResourceEstimate::new();
        a.set(ResourceUnit::DiskMiB, 3.0);
        let mut b = ResourceEstimate::new();
        b.set(ResourceUnit::DiskMiB, 3.0);
        assert_eq!(est.compare_estimates(&a, &b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_resource_estimator_a_is_cheaper_true() {
        let est = ResourceEstimator::new();
        let mut a = ResourceEstimate::new();
        a.set(ResourceUnit::NetworkMiB, 1.0);
        let mut b = ResourceEstimate::new();
        b.set(ResourceUnit::NetworkMiB, 10.0);
        assert!(est.a_is_cheaper(&a, &b));
    }

    #[test]
    fn test_resource_estimator_a_is_cheaper_false_when_equal() {
        let est = ResourceEstimator::new();
        let a = ResourceEstimate::new();
        let b = ResourceEstimate::new();
        assert!(!est.a_is_cheaper(&a, &b));
    }
}

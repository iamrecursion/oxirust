//! Cloud cost monitoring
//!
//! Provides in-memory cost tracking and budget alerting for cloud workloads:
//! - Categorised cost entries (Storage, Egress, Compute, API Calls, Transcoding)
//! - Monthly budget limits with configurable alert thresholds
//! - Aggregation by category and identification of top cost drivers

#![allow(dead_code)]

/// High-level category of a cloud cost entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CostCategory {
    /// Object storage charges (capacity × time)
    Storage,
    /// Data-transfer / egress charges
    Egress,
    /// Compute / instance charges
    Compute,
    /// Per-request API call charges
    ApiCalls,
    /// Media transcoding job charges
    Transcoding,
}

impl CostCategory {
    /// Returns `true` if this category's cost is variable (usage-dependent).
    ///
    /// `Compute` is treated as a fixed reservation cost in this model.
    #[must_use]
    pub fn is_variable(&self) -> bool {
        matches!(
            self,
            CostCategory::Storage
                | CostCategory::Egress
                | CostCategory::ApiCalls
                | CostCategory::Transcoding
        )
    }
}

/// A single recorded cost event.
#[derive(Debug, Clone)]
pub struct CostEntry {
    /// The cost category
    pub category: CostCategory,
    /// Unix epoch timestamp (seconds) when the cost was incurred
    pub timestamp_epoch: u64,
    /// Amount in USD
    pub amount_usd: f64,
    /// Human-readable description of the charge
    pub description: String,
}

impl CostEntry {
    /// Construct a new cost entry with an auto-generated empty description.
    #[must_use]
    pub fn new(cat: CostCategory, epoch: u64, amount: f64) -> Self {
        Self {
            category: cat,
            timestamp_epoch: epoch,
            amount_usd: amount,
            description: String::new(),
        }
    }
}

/// Monthly budget configuration.
#[derive(Debug, Clone)]
pub struct CostBudget {
    /// Maximum allowed spend per month in USD
    pub monthly_limit_usd: f64,
    /// Fraction of the budget (0.0–1.0) at which an alert is triggered
    pub alert_threshold: f64,
}

impl CostBudget {
    /// Returns `true` if `spent` has exceeded the monthly limit.
    #[must_use]
    pub fn is_over_budget(&self, spent: f64) -> bool {
        spent > self.monthly_limit_usd
    }

    /// Returns `true` if `spent` has reached or exceeded the alert threshold.
    #[must_use]
    pub fn should_alert(&self, spent: f64) -> bool {
        spent >= self.monthly_limit_usd * self.alert_threshold
    }
}

/// In-memory cloud cost monitor.
#[derive(Debug)]
pub struct CostMonitor {
    /// All recorded cost entries
    pub entries: Vec<CostEntry>,
    /// Budget configuration
    pub budget: CostBudget,
}

impl CostMonitor {
    /// Create a new cost monitor with the given budget.
    #[must_use]
    pub fn new(budget: CostBudget) -> Self {
        Self {
            entries: Vec::new(),
            budget,
        }
    }

    /// Record a new cost entry.
    pub fn record(&mut self, entry: CostEntry) {
        self.entries.push(entry);
    }

    /// Returns the sum of all recorded costs in USD.
    #[must_use]
    pub fn total_cost(&self) -> f64 {
        self.entries.iter().map(|e| e.amount_usd).sum()
    }

    /// Returns the total cost for the given category.
    #[must_use]
    pub fn cost_by_category(&self, cat: &CostCategory) -> f64 {
        self.entries
            .iter()
            .filter(|e| &e.category == cat)
            .map(|e| e.amount_usd)
            .sum()
    }

    /// Returns `true` if the total cost is within the monthly budget.
    #[must_use]
    pub fn is_within_budget(&self) -> bool {
        !self.budget.is_over_budget(self.total_cost())
    }

    /// Returns the top `n` cost categories sorted by descending total cost.
    ///
    /// Each element is a reference to the category and its summed cost.
    #[must_use]
    pub fn top_cost_categories(&self, n: usize) -> Vec<(&CostCategory, f64)> {
        // Collect unique categories
        let mut categories: Vec<&CostCategory> = Vec::new();
        for entry in &self.entries {
            if !categories.contains(&&entry.category) {
                categories.push(&entry.category);
            }
        }

        let mut sums: Vec<(&CostCategory, f64)> = categories
            .into_iter()
            .map(|cat| (cat, self.cost_by_category(cat)))
            .collect();

        sums.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sums.truncate(n);
        sums
    }
}

// ── Cost anomaly detection ────────────────────────────────────────────────────

/// A detected cost anomaly for a single day in the series.
#[derive(Debug, Clone, PartialEq)]
pub struct CostAnomaly {
    /// 0-based index in the input `daily_costs` slice.
    pub day_index: usize,
    /// The cost recorded on that day.
    pub cost: f64,
    /// How many standard deviations this cost is from the rolling mean.
    pub deviation_sigma: f64,
}

/// Detects anomalous daily costs using a z-score threshold.
///
/// A day is flagged when its cost is more than `sigma_threshold` standard
/// deviations away from the mean of all provided daily costs.
#[derive(Debug, Clone)]
pub struct CostAnomalyDetector {
    /// Number of standard deviations above/below the mean that trigger a flag.
    pub sigma_threshold: f64,
}

impl Default for CostAnomalyDetector {
    fn default() -> Self {
        Self {
            sigma_threshold: 3.0,
        }
    }
}

impl CostAnomalyDetector {
    /// Create a detector with the given sigma threshold.
    #[must_use]
    pub fn new(sigma_threshold: f64) -> Self {
        Self { sigma_threshold }
    }

    /// Scan `daily_costs` and return the **first** anomaly found, if any.
    ///
    /// The mean and standard deviation are computed over the entire slice.
    /// If fewer than two data points are provided, or the standard deviation
    /// is zero, no anomaly can be detected and `None` is returned.
    #[must_use]
    pub fn detect_anomaly(&self, daily_costs: &[f64]) -> Option<CostAnomaly> {
        if daily_costs.len() < 2 {
            return None;
        }

        let n = daily_costs.len() as f64;
        let mean = daily_costs.iter().sum::<f64>() / n;

        let variance = daily_costs.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return None;
        }

        daily_costs.iter().enumerate().find_map(|(i, &cost)| {
            let z = (cost - mean).abs() / std_dev;
            if z > self.sigma_threshold {
                Some(CostAnomaly {
                    day_index: i,
                    cost,
                    deviation_sigma: z,
                })
            } else {
                None
            }
        })
    }

    /// Return **all** anomalous days in the series.
    #[must_use]
    pub fn detect_all_anomalies(&self, daily_costs: &[f64]) -> Vec<CostAnomaly> {
        if daily_costs.len() < 2 {
            return Vec::new();
        }

        let n = daily_costs.len() as f64;
        let mean = daily_costs.iter().sum::<f64>() / n;
        let variance = daily_costs.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Vec::new();
        }

        daily_costs
            .iter()
            .enumerate()
            .filter_map(|(i, &cost)| {
                let z = (cost - mean).abs() / std_dev;
                if z > self.sigma_threshold {
                    Some(CostAnomaly {
                        day_index: i,
                        cost,
                        deviation_sigma: z,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_budget() -> CostBudget {
        CostBudget {
            monthly_limit_usd: 1000.0,
            alert_threshold: 0.8,
        }
    }

    fn make_monitor() -> CostMonitor {
        CostMonitor::new(default_budget())
    }

    #[test]
    fn test_cost_category_variable_storage() {
        assert!(CostCategory::Storage.is_variable());
    }

    #[test]
    fn test_cost_category_variable_egress() {
        assert!(CostCategory::Egress.is_variable());
    }

    #[test]
    fn test_cost_category_variable_transcoding() {
        assert!(CostCategory::Transcoding.is_variable());
    }

    #[test]
    fn test_cost_category_not_variable_compute() {
        assert!(!CostCategory::Compute.is_variable());
    }

    #[test]
    fn test_cost_entry_new_empty_description() {
        let entry = CostEntry::new(CostCategory::Storage, 1000, 9.99);
        assert!(entry.description.is_empty());
        assert_eq!(entry.amount_usd, 9.99);
    }

    #[test]
    fn test_budget_over_budget_false() {
        let budget = default_budget();
        assert!(!budget.is_over_budget(999.99));
    }

    #[test]
    fn test_budget_over_budget_true() {
        let budget = default_budget();
        assert!(budget.is_over_budget(1001.0));
    }

    #[test]
    fn test_budget_should_alert_true() {
        let budget = default_budget();
        // 80% of 1000 = 800
        assert!(budget.should_alert(800.0));
    }

    #[test]
    fn test_budget_should_alert_false() {
        let budget = default_budget();
        assert!(!budget.should_alert(799.99));
    }

    #[test]
    fn test_total_cost_empty() {
        let monitor = make_monitor();
        assert_eq!(monitor.total_cost(), 0.0);
    }

    #[test]
    fn test_total_cost_sum() {
        let mut monitor = make_monitor();
        monitor.record(CostEntry::new(CostCategory::Storage, 1, 50.0));
        monitor.record(CostEntry::new(CostCategory::Egress, 2, 30.0));
        assert_eq!(monitor.total_cost(), 80.0);
    }

    #[test]
    fn test_cost_by_category() {
        let mut monitor = make_monitor();
        monitor.record(CostEntry::new(CostCategory::Storage, 1, 10.0));
        monitor.record(CostEntry::new(CostCategory::Storage, 2, 20.0));
        monitor.record(CostEntry::new(CostCategory::Egress, 3, 5.0));
        assert_eq!(monitor.cost_by_category(&CostCategory::Storage), 30.0);
        assert_eq!(monitor.cost_by_category(&CostCategory::Egress), 5.0);
    }

    #[test]
    fn test_is_within_budget_true() {
        let monitor = make_monitor();
        assert!(monitor.is_within_budget());
    }

    #[test]
    fn test_is_within_budget_false() {
        let mut monitor = make_monitor();
        monitor.record(CostEntry::new(CostCategory::Transcoding, 1, 1500.0));
        assert!(!monitor.is_within_budget());
    }

    #[test]
    fn test_top_cost_categories() {
        let mut monitor = make_monitor();
        monitor.record(CostEntry::new(CostCategory::Storage, 1, 100.0));
        monitor.record(CostEntry::new(CostCategory::Egress, 2, 200.0));
        monitor.record(CostEntry::new(CostCategory::Compute, 3, 50.0));
        let top = monitor.top_cost_categories(2);
        assert_eq!(top.len(), 2);
        assert_eq!(*top[0].0, CostCategory::Egress);
        assert_eq!(*top[1].0, CostCategory::Storage);
    }

    #[test]
    fn test_top_cost_categories_n_larger_than_categories() {
        let mut monitor = make_monitor();
        monitor.record(CostEntry::new(CostCategory::Storage, 1, 50.0));
        let top = monitor.top_cost_categories(10);
        assert_eq!(top.len(), 1);
    }

    // ── CostAnomalyDetector tests ─────────────────────────────────────────────

    #[test]
    fn test_detect_anomaly_no_data_returns_none() {
        let detector = CostAnomalyDetector::default();
        assert!(detector.detect_anomaly(&[]).is_none());
        assert!(detector.detect_anomaly(&[100.0]).is_none());
    }

    #[test]
    fn test_detect_anomaly_uniform_data_returns_none() {
        let detector = CostAnomalyDetector::default();
        // All values identical → std_dev = 0 → no anomaly can be detected
        let data = vec![50.0; 10];
        assert!(detector.detect_anomaly(&data).is_none());
    }

    #[test]
    fn test_detect_anomaly_clear_spike() {
        let detector = CostAnomalyDetector::default();
        // 29 normal days at ~100, one extreme spike
        let mut data: Vec<f64> = vec![100.0; 29];
        data.push(10_000.0); // extreme outlier at index 29
        let anomaly = detector
            .detect_anomaly(&data)
            .expect("spike must be detected");
        assert_eq!(anomaly.day_index, 29);
        assert!((anomaly.cost - 10_000.0).abs() < 1e-9);
        assert!(anomaly.deviation_sigma > 3.0);
    }

    #[test]
    fn test_detect_anomaly_sigma_threshold_respected() {
        // Use a very high threshold so even large spikes are not flagged
        let detector = CostAnomalyDetector::new(100.0);
        let mut data: Vec<f64> = vec![100.0; 29];
        data.push(10_000.0);
        // With a 100σ threshold this spike should not trigger
        assert!(detector.detect_anomaly(&data).is_none());
    }

    #[test]
    fn test_detect_all_anomalies_multiple_spikes() {
        let detector = CostAnomalyDetector::default();
        let mut data: Vec<f64> = vec![100.0; 28];
        data.push(9_999.0); // index 28
        data.push(9_998.0); // index 29
        let anomalies = detector.detect_all_anomalies(&data);
        assert_eq!(anomalies.len(), 2, "both spikes must be flagged");
        let indices: Vec<usize> = anomalies.iter().map(|a| a.day_index).collect();
        assert!(indices.contains(&28));
        assert!(indices.contains(&29));
    }

    #[test]
    fn test_cost_anomaly_deviation_sigma_positive() {
        let detector = CostAnomalyDetector::default();
        let mut data: Vec<f64> = vec![100.0; 29];
        data.push(50_000.0);
        if let Some(anomaly) = detector.detect_anomaly(&data) {
            assert!(anomaly.deviation_sigma > 0.0, "sigma must be positive");
        }
    }
}

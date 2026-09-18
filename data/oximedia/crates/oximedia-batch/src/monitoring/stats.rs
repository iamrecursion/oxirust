//! Statistics collection and aggregation

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Statistics collector
pub struct StatsCollector {
    metrics: Arc<RwLock<HashMap<String, f64>>>,
}

impl StatsCollector {
    /// Create a new stats collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record a metric value
    pub fn record(&self, name: String, value: f64) {
        self.metrics.write().insert(name, value);
    }

    /// Increment a counter
    pub fn increment(&self, name: &str) {
        let mut metrics = self.metrics.write();
        let current = metrics.get(name).copied().unwrap_or(0.0);
        metrics.insert(name.to_string(), current + 1.0);
    }

    /// Decrement a counter
    pub fn decrement(&self, name: &str) {
        let mut metrics = self.metrics.write();
        let current = metrics.get(name).copied().unwrap_or(0.0);
        metrics.insert(name.to_string(), (current - 1.0).max(0.0));
    }

    /// Get a metric value
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f64> {
        self.metrics.read().get(name).copied()
    }

    /// Get all metrics
    #[must_use]
    pub fn get_all(&self) -> HashMap<String, f64> {
        self.metrics.read().clone()
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.metrics.write().clear();
    }
}

impl Default for StatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_collector_creation() {
        let collector = StatsCollector::new();
        assert!(collector.get_all().is_empty());
    }

    #[test]
    fn test_record_metric() {
        let collector = StatsCollector::new();
        collector.record("cpu_usage".to_string(), 75.5);

        assert_eq!(collector.get("cpu_usage"), Some(75.5));
    }

    #[test]
    fn test_increment_counter() {
        let collector = StatsCollector::new();

        collector.increment("jobs_completed");
        collector.increment("jobs_completed");
        collector.increment("jobs_completed");

        assert_eq!(collector.get("jobs_completed"), Some(3.0));
    }

    #[test]
    fn test_decrement_counter() {
        let collector = StatsCollector::new();

        collector.record("active_jobs".to_string(), 5.0);
        collector.decrement("active_jobs");

        assert_eq!(collector.get("active_jobs"), Some(4.0));
    }

    #[test]
    fn test_decrement_below_zero() {
        let collector = StatsCollector::new();

        collector.record("active_jobs".to_string(), 1.0);
        collector.decrement("active_jobs");
        collector.decrement("active_jobs");

        assert_eq!(collector.get("active_jobs"), Some(0.0));
    }

    #[test]
    fn test_get_all_metrics() {
        let collector = StatsCollector::new();

        collector.record("metric1".to_string(), 10.0);
        collector.record("metric2".to_string(), 20.0);

        let all_metrics = collector.get_all();
        assert_eq!(all_metrics.len(), 2);
        assert_eq!(all_metrics.get("metric1"), Some(&10.0));
        assert_eq!(all_metrics.get("metric2"), Some(&20.0));
    }

    #[test]
    fn test_reset_metrics() {
        let collector = StatsCollector::new();

        collector.record("metric1".to_string(), 10.0);
        collector.record("metric2".to_string(), 20.0);

        collector.reset();

        assert!(collector.get_all().is_empty());
    }

    #[test]
    fn test_get_nonexistent_metric() {
        let collector = StatsCollector::new();
        assert_eq!(collector.get("nonexistent"), None);
    }
}

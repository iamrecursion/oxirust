//! Hotspot detection for identifying performance-critical code.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// A performance hotspot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hotspot {
    /// Function name.
    pub function: String,

    /// Total time spent in this function.
    pub total_time: Duration,

    /// Number of times this function was sampled.
    pub sample_count: u64,

    /// Percentage of total execution time.
    pub time_percentage: f64,

    /// Severity level (0.0-1.0).
    pub severity: f64,

    /// File location.
    pub location: Option<String>,
}

impl Hotspot {
    /// Create a new hotspot.
    pub fn new(function: String, total_time: Duration, sample_count: u64) -> Self {
        Self {
            function,
            total_time,
            sample_count,
            time_percentage: 0.0,
            severity: 0.0,
            location: None,
        }
    }

    /// Set the time percentage.
    pub fn with_percentage(mut self, percentage: f64) -> Self {
        self.time_percentage = percentage;
        self.severity = (percentage / 100.0).min(1.0);
        self
    }

    /// Set the location.
    pub fn with_location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }

    /// Check if this is a critical hotspot (>10% of execution time).
    pub fn is_critical(&self) -> bool {
        self.time_percentage > 10.0
    }

    /// Check if this is a significant hotspot (>5% of execution time).
    pub fn is_significant(&self) -> bool {
        self.time_percentage > 5.0
    }

    /// Get a description of the hotspot.
    pub fn description(&self) -> String {
        let criticality = if self.is_critical() {
            "CRITICAL"
        } else if self.is_significant() {
            "SIGNIFICANT"
        } else {
            "MINOR"
        };

        format!(
            "[{}] {} - {:.2}% of execution time ({:?}, {} samples)",
            criticality, self.function, self.time_percentage, self.total_time, self.sample_count
        )
    }
}

/// Hotspot detector for identifying performance bottlenecks.
#[derive(Debug)]
pub struct HotspotDetector {
    function_times: HashMap<String, Duration>,
    function_counts: HashMap<String, u64>,
    total_time: Duration,
    threshold: f64,
}

impl HotspotDetector {
    /// Create a new hotspot detector.
    pub fn new(threshold: f64) -> Self {
        Self {
            function_times: HashMap::new(),
            function_counts: HashMap::new(),
            total_time: Duration::ZERO,
            threshold,
        }
    }

    /// Record function execution.
    pub fn record(&mut self, function: String, duration: Duration) {
        *self
            .function_times
            .entry(function.clone())
            .or_insert(Duration::ZERO) += duration;
        *self.function_counts.entry(function).or_insert(0) += 1;
        self.total_time += duration;
    }

    /// Detect hotspots.
    pub fn detect(&self) -> Vec<Hotspot> {
        let mut hotspots = Vec::new();

        for (function, &total_time) in &self.function_times {
            let sample_count = self.function_counts.get(function).copied().unwrap_or(0);
            let time_percentage = if self.total_time.as_secs_f64() > 0.0 {
                (total_time.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0
            } else {
                0.0
            };

            if time_percentage >= self.threshold {
                let hotspot = Hotspot::new(function.clone(), total_time, sample_count)
                    .with_percentage(time_percentage);
                hotspots.push(hotspot);
            }
        }

        hotspots.sort_by(|a, b| {
            b.time_percentage
                .partial_cmp(&a.time_percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        hotspots
    }

    /// Get the total execution time.
    pub fn total_time(&self) -> Duration {
        self.total_time
    }

    /// Get the threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Generate a hotspot report.
    pub fn report(&self) -> String {
        let hotspots = self.detect();
        let mut report = String::new();

        report.push_str(&format!("Total Time: {:?}\n", self.total_time));
        report.push_str(&format!("Threshold: {:.2}%\n", self.threshold));
        report.push_str(&format!("Hotspots Found: {}\n\n", hotspots.len()));

        for (i, hotspot) in hotspots.iter().enumerate() {
            report.push_str(&format!("{}. {}\n", i + 1, hotspot.description()));
        }

        report
    }
}

impl Default for HotspotDetector {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hotspot_creation() {
        let hotspot = Hotspot::new("test_function".to_string(), Duration::from_secs(1), 100);
        assert_eq!(hotspot.function, "test_function");
        assert_eq!(hotspot.sample_count, 100);
    }

    #[test]
    fn test_hotspot_criticality() {
        let critical =
            Hotspot::new("func".to_string(), Duration::from_secs(1), 100).with_percentage(15.0);
        assert!(critical.is_critical());
        assert!(critical.is_significant());

        let significant =
            Hotspot::new("func".to_string(), Duration::from_secs(1), 100).with_percentage(7.0);
        assert!(!significant.is_critical());
        assert!(significant.is_significant());

        let minor =
            Hotspot::new("func".to_string(), Duration::from_secs(1), 100).with_percentage(2.0);
        assert!(!minor.is_critical());
        assert!(!minor.is_significant());
    }

    #[test]
    fn test_hotspot_detector() {
        let mut detector = HotspotDetector::new(5.0);
        detector.record("func1".to_string(), Duration::from_millis(500));
        detector.record("func2".to_string(), Duration::from_millis(300));
        detector.record("func3".to_string(), Duration::from_millis(200));
        detector.record("func1".to_string(), Duration::from_millis(500));

        let hotspots = detector.detect();
        assert!(!hotspots.is_empty());
        assert_eq!(hotspots[0].function, "func1"); // Highest time
    }

    #[test]
    fn test_hotspot_threshold() {
        let mut detector = HotspotDetector::new(50.0); // Set higher threshold
        detector.record("func1".to_string(), Duration::from_millis(100));
        detector.record("func2".to_string(), Duration::from_millis(900));

        let hotspots = detector.detect();
        assert_eq!(hotspots.len(), 1);
        assert_eq!(hotspots[0].function, "func2");
    }

    #[test]
    fn test_hotspot_report() {
        let mut detector = HotspotDetector::new(1.0);
        detector.record("func1".to_string(), Duration::from_millis(500));
        detector.record("func2".to_string(), Duration::from_millis(500));

        let report = detector.report();
        assert!(report.contains("Total Time"));
        assert!(report.contains("Threshold"));
        assert!(report.contains("Hotspots Found"));
    }
}

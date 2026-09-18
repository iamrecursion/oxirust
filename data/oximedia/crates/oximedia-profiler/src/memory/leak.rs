//! Memory leak detection.

use super::track::{AllocationInfo, MemoryTracker};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A detected memory leak.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    /// Allocation ID.
    pub id: u64,

    /// Size in bytes.
    pub size: usize,

    /// Age of the allocation.
    pub age: Duration,

    /// Location where allocated.
    pub location: Option<String>,

    /// Stack trace.
    pub stack_trace: Vec<String>,

    /// Severity (0.0-1.0).
    pub severity: f64,
}

impl MemoryLeak {
    /// Create a new memory leak.
    pub fn new(id: u64, size: usize, age: Duration) -> Self {
        let severity = Self::calculate_severity(size, age);
        Self {
            id,
            size,
            age,
            location: None,
            stack_trace: Vec::new(),
            severity,
        }
    }

    /// Create from allocation info.
    pub fn from_allocation(id: u64, info: &AllocationInfo) -> Self {
        let mut leak = Self::new(id, info.size, info.age());
        leak.location = info.location.clone();
        leak.stack_trace = info.stack_trace.clone();
        leak
    }

    /// Calculate severity based on size and age.
    fn calculate_severity(size: usize, age: Duration) -> f64 {
        let size_score = (size as f64 / 1_000_000.0).min(1.0);
        let age_score = (age.as_secs() as f64 / 60.0).min(1.0);
        (size_score * 0.6 + age_score * 0.4).min(1.0)
    }

    /// Check if this is a critical leak.
    pub fn is_critical(&self) -> bool {
        self.severity > 0.7
    }

    /// Check if this is a significant leak.
    pub fn is_significant(&self) -> bool {
        self.severity > 0.4
    }

    /// Get a description of the leak.
    pub fn description(&self) -> String {
        let criticality = if self.is_critical() {
            "CRITICAL"
        } else if self.is_significant() {
            "SIGNIFICANT"
        } else {
            "MINOR"
        };

        let mut desc = format!(
            "[{}] Leak #{}: {} bytes, age {:?}",
            criticality, self.id, self.size, self.age
        );

        if let Some(ref location) = self.location {
            desc.push_str(&format!(" at {}", location));
        }

        desc
    }
}

/// Memory leak detector.
#[derive(Debug)]
pub struct LeakDetector {
    age_threshold: Duration,
    size_threshold: usize,
}

impl LeakDetector {
    /// Create a new leak detector.
    pub fn new(age_threshold: Duration, size_threshold: usize) -> Self {
        Self {
            age_threshold,
            size_threshold,
        }
    }

    /// Detect leaks in the memory tracker.
    pub fn detect(&self, tracker: &MemoryTracker) -> Vec<MemoryLeak> {
        let mut leaks = Vec::new();

        for (id, info) in tracker.active_allocations() {
            if !info.freed && self.is_potential_leak(info) {
                leaks.push(MemoryLeak::from_allocation(*id, info));
            }
        }

        leaks.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        leaks
    }

    /// Check if an allocation is a potential leak.
    fn is_potential_leak(&self, info: &AllocationInfo) -> bool {
        info.age() >= self.age_threshold || info.size >= self.size_threshold
    }

    /// Get the age threshold.
    pub fn age_threshold(&self) -> Duration {
        self.age_threshold
    }

    /// Get the size threshold.
    pub fn size_threshold(&self) -> usize {
        self.size_threshold
    }

    /// Generate a leak report.
    pub fn report(&self, tracker: &MemoryTracker) -> String {
        let leaks = self.detect(tracker);
        let mut report = String::new();

        report.push_str(&format!("Age Threshold: {:?}\n", self.age_threshold));
        report.push_str(&format!("Size Threshold: {} bytes\n", self.size_threshold));
        report.push_str(&format!("Leaks Detected: {}\n\n", leaks.len()));

        for (i, leak) in leaks.iter().enumerate() {
            report.push_str(&format!("{}. {}\n", i + 1, leak.description()));
        }

        if leaks.is_empty() {
            report.push_str("No memory leaks detected.\n");
        }

        report
    }
}

impl Default for LeakDetector {
    fn default() -> Self {
        Self::new(Duration::from_mins(1), 1_000_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_leak_severity() {
        let small_new = MemoryLeak::new(1, 1000, Duration::from_secs(1));
        let large_new = MemoryLeak::new(2, 10_000_000, Duration::from_secs(1));
        let small_old = MemoryLeak::new(3, 1000, Duration::from_mins(2));
        let large_old = MemoryLeak::new(4, 10_000_000, Duration::from_mins(2));

        assert!(large_old.severity > large_new.severity);
        assert!(large_old.severity > small_old.severity);
        assert!(small_old.severity > small_new.severity);
    }

    #[test]
    fn test_leak_criticality() {
        let critical = MemoryLeak::new(1, 10_000_000, Duration::from_mins(2));
        assert!(critical.is_critical());

        let minor = MemoryLeak::new(2, 100, Duration::from_secs(1));
        assert!(!minor.is_critical());
        assert!(!minor.is_significant());
    }

    #[test]
    fn test_leak_detector() {
        let mut tracker = MemoryTracker::new();
        tracker.start();

        // Create some allocations
        tracker.track_allocation(1000, Some("test1.rs".to_string()));
        std::thread::sleep(Duration::from_millis(10));
        tracker.track_allocation(2_000_000, Some("test2.rs".to_string()));

        let detector = LeakDetector::new(Duration::from_millis(5), 1_000_000);
        let leaks = detector.detect(&tracker);

        assert!(!leaks.is_empty());
        tracker.stop();
    }

    #[test]
    fn test_leak_report() {
        let mut tracker = MemoryTracker::new();
        tracker.start();
        tracker.track_allocation(2_000_000, None);

        let detector = LeakDetector::default();
        let report = detector.report(&tracker);

        assert!(report.contains("Age Threshold"));
        assert!(report.contains("Size Threshold"));
        assert!(report.contains("Leaks Detected"));

        tracker.stop();
    }
}

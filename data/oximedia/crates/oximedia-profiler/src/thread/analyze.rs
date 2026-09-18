//! Thread utilization analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Thread statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadStats {
    /// Thread ID.
    pub thread_id: u64,

    /// Thread name.
    pub name: String,

    /// CPU time.
    pub cpu_time: Duration,

    /// Wall time.
    pub wall_time: Duration,

    /// CPU utilization (0.0-1.0).
    pub utilization: f64,

    /// Number of context switches.
    pub context_switches: u64,
}

/// Thread analyzer.
#[derive(Debug)]
pub struct ThreadAnalyzer {
    threads: HashMap<u64, ThreadStats>,
}

impl ThreadAnalyzer {
    /// Create a new thread analyzer.
    pub fn new() -> Self {
        Self {
            threads: HashMap::new(),
        }
    }

    /// Record thread statistics.
    pub fn record(&mut self, stats: ThreadStats) {
        self.threads.insert(stats.thread_id, stats);
    }

    /// Get statistics for a thread.
    pub fn get_stats(&self, thread_id: u64) -> Option<&ThreadStats> {
        self.threads.get(&thread_id)
    }

    /// Get all thread stats.
    pub fn all_stats(&self) -> Vec<&ThreadStats> {
        self.threads.values().collect()
    }

    /// Get underutilized threads (<50% utilization).
    pub fn underutilized_threads(&self) -> Vec<&ThreadStats> {
        self.threads
            .values()
            .filter(|s| s.utilization < 0.5)
            .collect()
    }

    /// Get average utilization.
    pub fn avg_utilization(&self) -> f64 {
        if self.threads.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.threads.values().map(|s| s.utilization).sum();
        sum / self.threads.len() as f64
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("Threads Analyzed: {}\n", self.threads.len()));
        report.push_str(&format!(
            "Average Utilization: {:.2}%\n\n",
            self.avg_utilization() * 100.0
        ));

        let mut stats: Vec<_> = self.threads.values().collect();
        stats.sort_by(|a, b| {
            b.utilization
                .partial_cmp(&a.utilization)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for stat in stats {
            report.push_str(&format!(
                "Thread {} ({}): {:.2}% utilization, {:?} CPU time\n",
                stat.thread_id,
                stat.name,
                stat.utilization * 100.0,
                stat.cpu_time
            ));
        }

        report
    }
}

impl Default for ThreadAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_analyzer() {
        let mut analyzer = ThreadAnalyzer::new();

        let stats = ThreadStats {
            thread_id: 1,
            name: "worker-1".to_string(),
            cpu_time: Duration::from_secs(8),
            wall_time: Duration::from_secs(10),
            utilization: 0.8,
            context_switches: 100,
        };

        analyzer.record(stats);
        assert_eq!(analyzer.all_stats().len(), 1);
    }

    #[test]
    fn test_avg_utilization() {
        let mut analyzer = ThreadAnalyzer::new();

        analyzer.record(ThreadStats {
            thread_id: 1,
            name: "t1".to_string(),
            cpu_time: Duration::from_secs(8),
            wall_time: Duration::from_secs(10),
            utilization: 0.8,
            context_switches: 100,
        });

        analyzer.record(ThreadStats {
            thread_id: 2,
            name: "t2".to_string(),
            cpu_time: Duration::from_secs(6),
            wall_time: Duration::from_secs(10),
            utilization: 0.6,
            context_switches: 50,
        });

        assert!((analyzer.avg_utilization() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_underutilized_threads() {
        let mut analyzer = ThreadAnalyzer::new();

        analyzer.record(ThreadStats {
            thread_id: 1,
            name: "high".to_string(),
            cpu_time: Duration::from_secs(8),
            wall_time: Duration::from_secs(10),
            utilization: 0.8,
            context_switches: 100,
        });

        analyzer.record(ThreadStats {
            thread_id: 2,
            name: "low".to_string(),
            cpu_time: Duration::from_secs(3),
            wall_time: Duration::from_secs(10),
            utilization: 0.3,
            context_switches: 200,
        });

        let underutilized = analyzer.underutilized_threads();
        assert_eq!(underutilized.len(), 1);
        assert_eq!(underutilized[0].name, "low");
    }
}

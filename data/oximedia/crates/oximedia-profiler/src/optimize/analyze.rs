//! Code path analysis for optimization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Code path.
    pub path: Vec<String>,

    /// Total time in this path.
    pub total_time: Duration,

    /// Percentage of total execution.
    pub percentage: f64,

    /// Number of calls.
    pub call_count: u64,

    /// Optimization opportunities.
    pub opportunities: Vec<String>,
}

impl AnalysisResult {
    /// Check if this path is worth optimizing.
    pub fn is_worth_optimizing(&self) -> bool {
        self.percentage > 5.0
    }

    /// Get priority (0-10, higher is more important).
    pub fn priority(&self) -> u8 {
        if self.percentage > 20.0 {
            10
        } else if self.percentage > 10.0 {
            7
        } else if self.percentage > 5.0 {
            5
        } else {
            3
        }
    }
}

/// Code analyzer for finding optimization opportunities.
#[derive(Debug)]
pub struct CodeAnalyzer {
    paths: HashMap<String, PathInfo>,
    total_time: Duration,
}

/// Path information.
#[derive(Debug, Clone)]
struct PathInfo {
    path: Vec<String>,
    total_time: Duration,
    call_count: u64,
}

impl CodeAnalyzer {
    /// Create a new code analyzer.
    pub fn new() -> Self {
        Self {
            paths: HashMap::new(),
            total_time: Duration::ZERO,
        }
    }

    /// Record a code path execution.
    pub fn record_path(&mut self, path: Vec<String>, duration: Duration) {
        let path_key = path.join(" -> ");
        let info = self.paths.entry(path_key).or_insert(PathInfo {
            path: path.clone(),
            total_time: Duration::ZERO,
            call_count: 0,
        });

        info.total_time += duration;
        info.call_count += 1;
        self.total_time += duration;
    }

    /// Analyze all paths.
    pub fn analyze(&self) -> Vec<AnalysisResult> {
        let mut results = Vec::new();

        for info in self.paths.values() {
            let percentage = if self.total_time.as_secs_f64() > 0.0 {
                (info.total_time.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0
            } else {
                0.0
            };

            let opportunities = self.find_opportunities(info, percentage);

            results.push(AnalysisResult {
                path: info.path.clone(),
                total_time: info.total_time,
                percentage,
                call_count: info.call_count,
                opportunities,
            });
        }

        results.sort_by(|a, b| {
            b.percentage
                .partial_cmp(&a.percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Find optimization opportunities for a path.
    fn find_opportunities(&self, info: &PathInfo, percentage: f64) -> Vec<String> {
        let mut opportunities = Vec::new();

        if percentage > 20.0 {
            opportunities.push("Critical path - high optimization priority".to_string());
        }

        if info.call_count > 10000 {
            opportunities.push("Frequently called - consider caching or memoization".to_string());
        }

        if info.path.len() > 10 {
            opportunities.push("Deep call stack - consider flattening".to_string());
        }

        opportunities
    }

    /// Get total time.
    pub fn total_time(&self) -> Duration {
        self.total_time
    }

    /// Generate a report.
    pub fn report(&self) -> String {
        let results = self.analyze();
        let mut report = String::new();

        report.push_str(&format!("Code Path Analysis ({} paths)\n\n", results.len()));

        for (i, result) in results.iter().enumerate().take(10) {
            report.push_str(&format!("{}. Priority: {}/10\n", i + 1, result.priority()));
            report.push_str(&format!("   Path: {}\n", result.path.join(" -> ")));
            report.push_str(&format!(
                "   Time: {:?} ({:.2}%)\n",
                result.total_time, result.percentage
            ));
            report.push_str(&format!("   Calls: {}\n", result.call_count));

            if !result.opportunities.is_empty() {
                report.push_str("   Opportunities:\n");
                for opp in &result.opportunities {
                    report.push_str(&format!("     - {}\n", opp));
                }
            }

            report.push('\n');
        }

        report
    }
}

impl Default for CodeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_analyzer() {
        let mut analyzer = CodeAnalyzer::new();

        let path = vec![
            "main".to_string(),
            "process".to_string(),
            "compute".to_string(),
        ];
        analyzer.record_path(path, Duration::from_millis(100));

        let results = analyzer.analyze();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_priority() {
        let result = AnalysisResult {
            path: vec![],
            total_time: Duration::from_secs(1),
            percentage: 25.0,
            call_count: 100,
            opportunities: vec![],
        };

        assert_eq!(result.priority(), 10);
        assert!(result.is_worth_optimizing());
    }

    #[test]
    fn test_opportunities() {
        let mut analyzer = CodeAnalyzer::new();

        let path = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        analyzer.record_path(path, Duration::from_millis(300));

        let results = analyzer.analyze();
        assert!(!results[0].opportunities.is_empty());
    }
}

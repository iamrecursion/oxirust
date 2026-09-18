//! Cache analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cache statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,

    /// Number of cache misses.
    pub misses: u64,

    /// Hit rate (0.0-1.0).
    pub hit_rate: f64,

    /// Miss rate (0.0-1.0).
    pub miss_rate: f64,

    /// Cache size.
    pub size: usize,

    /// Number of evictions.
    pub evictions: u64,
}

impl CacheStats {
    /// Check if hit rate is good (>80%).
    pub fn is_good_hit_rate(&self) -> bool {
        self.hit_rate > 0.8
    }

    /// Check if hit rate is poor (<50%).
    pub fn is_poor_hit_rate(&self) -> bool {
        self.hit_rate < 0.5
    }
}

/// Cache analyzer.
#[derive(Debug)]
pub struct CacheAnalyzer {
    cache_stats: HashMap<String, CacheStats>,
}

impl CacheAnalyzer {
    /// Create a new cache analyzer.
    pub fn new() -> Self {
        Self {
            cache_stats: HashMap::new(),
        }
    }

    /// Record cache statistics for a named cache.
    pub fn record(&mut self, name: String, stats: CacheStats) {
        self.cache_stats.insert(name, stats);
    }

    /// Get statistics for a cache.
    pub fn get_stats(&self, name: &str) -> Option<&CacheStats> {
        self.cache_stats.get(name)
    }

    /// Get all cache names.
    pub fn cache_names(&self) -> Vec<&String> {
        self.cache_stats.keys().collect()
    }

    /// Get caches with poor hit rates.
    pub fn poor_caches(&self) -> Vec<(&String, &CacheStats)> {
        self.cache_stats
            .iter()
            .filter(|(_, stats)| stats.is_poor_hit_rate())
            .collect()
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("Tracked Caches: {}\n\n", self.cache_stats.len()));

        for (name, stats) in &self.cache_stats {
            let quality = if stats.is_good_hit_rate() {
                "GOOD"
            } else if stats.is_poor_hit_rate() {
                "POOR"
            } else {
                "OK"
            };

            report.push_str(&format!(
                "[{}] {}: {:.2}% hit rate ({} hits, {} misses)\n",
                quality,
                name,
                stats.hit_rate * 100.0,
                stats.hits,
                stats.misses
            ));
        }

        report
    }
}

impl Default for CacheAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_stats() {
        let stats = CacheStats {
            hits: 800,
            misses: 200,
            hit_rate: 0.8,
            miss_rate: 0.2,
            size: 1000,
            evictions: 10,
        };

        assert!(!stats.is_good_hit_rate()); // Exactly 0.8, not > 0.8
        assert!(!stats.is_poor_hit_rate());
    }

    #[test]
    fn test_cache_analyzer() {
        let mut analyzer = CacheAnalyzer::new();

        let stats = CacheStats {
            hits: 900,
            misses: 100,
            hit_rate: 0.9,
            miss_rate: 0.1,
            size: 1000,
            evictions: 5,
        };

        analyzer.record("test_cache".to_string(), stats);

        assert_eq!(analyzer.cache_names().len(), 1);
        assert!(analyzer.get_stats("test_cache").is_some());
    }

    #[test]
    fn test_poor_caches() {
        let mut analyzer = CacheAnalyzer::new();

        let good_stats = CacheStats {
            hits: 900,
            misses: 100,
            hit_rate: 0.9,
            miss_rate: 0.1,
            size: 1000,
            evictions: 5,
        };

        let poor_stats = CacheStats {
            hits: 400,
            misses: 600,
            hit_rate: 0.4,
            miss_rate: 0.6,
            size: 1000,
            evictions: 100,
        };

        analyzer.record("good_cache".to_string(), good_stats);
        analyzer.record("poor_cache".to_string(), poor_stats);

        let poor = analyzer.poor_caches();
        assert_eq!(poor.len(), 1);
        assert_eq!(poor[0].0, "poor_cache");
    }
}

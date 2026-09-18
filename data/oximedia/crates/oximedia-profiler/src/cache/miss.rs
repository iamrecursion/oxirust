//! Cache miss profiling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cache miss pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissPattern {
    /// Access pattern description.
    pub description: String,

    /// Number of occurrences.
    pub occurrences: u64,

    /// Suggested optimization.
    pub suggestion: String,
}

/// Cache miss profiler.
#[derive(Debug)]
pub struct CacheMissProfiler {
    miss_addresses: Vec<u64>,
    miss_patterns: HashMap<String, MissPattern>,
    #[allow(dead_code)]
    stride_threshold: usize,
}

impl CacheMissProfiler {
    /// Create a new cache miss profiler.
    pub fn new(stride_threshold: usize) -> Self {
        Self {
            miss_addresses: Vec::new(),
            miss_patterns: HashMap::new(),
            stride_threshold,
        }
    }

    /// Record a cache miss.
    pub fn record_miss(&mut self, address: u64) {
        self.miss_addresses.push(address);
        self.analyze_patterns();
    }

    /// Analyze miss patterns.
    fn analyze_patterns(&mut self) {
        if self.miss_addresses.len() < 3 {
            return;
        }

        // Check for stride patterns
        let recent = &self.miss_addresses[self.miss_addresses.len().saturating_sub(10)..];
        if let Some(stride) = self.detect_stride(recent) {
            let pattern_key = format!("stride_{}", stride);
            let pattern = self
                .miss_patterns
                .entry(pattern_key.clone())
                .or_insert_with(|| MissPattern {
                    description: format!("Strided access with stride {}", stride),
                    occurrences: 0,
                    suggestion: "Consider prefetching or improving cache line utilization"
                        .to_string(),
                });
            pattern.occurrences += 1;
        }
    }

    /// Detect stride pattern.
    fn detect_stride(&self, addresses: &[u64]) -> Option<i64> {
        if addresses.len() < 3 {
            return None;
        }

        let stride = addresses[1] as i64 - addresses[0] as i64;
        for i in 1..addresses.len() - 1 {
            let current_stride = addresses[i + 1] as i64 - addresses[i] as i64;
            if current_stride != stride {
                return None;
            }
        }

        Some(stride)
    }

    /// Get detected patterns.
    pub fn patterns(&self) -> &HashMap<String, MissPattern> {
        &self.miss_patterns
    }

    /// Get total miss count.
    pub fn miss_count(&self) -> usize {
        self.miss_addresses.len()
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.miss_addresses.clear();
        self.miss_patterns.clear();
    }

    /// Generate a report.
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("Total Cache Misses: {}\n", self.miss_count()));
        report.push_str(&format!(
            "Detected Patterns: {}\n\n",
            self.miss_patterns.len()
        ));

        for pattern in self.miss_patterns.values() {
            report.push_str(&format!(
                "{}: {} occurrences\n",
                pattern.description, pattern.occurrences
            ));
            report.push_str(&format!("  Suggestion: {}\n\n", pattern.suggestion));
        }

        report
    }
}

impl Default for CacheMissProfiler {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_miss_profiler() {
        let mut profiler = CacheMissProfiler::new(64);
        assert_eq!(profiler.miss_count(), 0);

        profiler.record_miss(0x1000);
        profiler.record_miss(0x1040);
        profiler.record_miss(0x1080);

        assert_eq!(profiler.miss_count(), 3);
    }

    #[test]
    fn test_stride_detection() {
        let mut profiler = CacheMissProfiler::new(64);

        // Create a stride pattern
        for i in 0..5 {
            profiler.record_miss(0x1000 + i * 64);
        }

        assert!(!profiler.patterns().is_empty());
    }

    #[test]
    fn test_clear() {
        let mut profiler = CacheMissProfiler::new(64);
        profiler.record_miss(0x1000);
        profiler.clear();

        assert_eq!(profiler.miss_count(), 0);
        assert!(profiler.patterns().is_empty());
    }
}

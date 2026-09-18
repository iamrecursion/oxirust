#![allow(dead_code)]
//! Index health monitoring and statistics.

/// Health classification of an index shard or segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexHealth {
    /// Index is operating normally.
    Green,
    /// Index is operating but with reduced redundancy or minor issues.
    Yellow,
    /// Index is unavailable or has critical errors.
    Red,
}

impl IndexHealth {
    /// Returns `true` if the health is `Green`.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        *self == Self::Green
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Green => "green",
            Self::Yellow => "yellow",
            Self::Red => "red",
        }
    }
}

impl std::fmt::Display for IndexHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Statistics about a search index.
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Total number of indexed documents.
    pub documents: u64,
    /// Total number of unique indexed terms.
    pub terms: u64,
    /// Sum of all document token counts (for computing averages).
    pub total_tokens: u64,
    /// Number of index segments.
    pub segments: usize,
    /// Index size on disk in bytes.
    pub disk_bytes: u64,
}

impl IndexStats {
    /// Create a new `IndexStats`.
    #[must_use]
    pub fn new(
        documents: u64,
        terms: u64,
        total_tokens: u64,
        segments: usize,
        disk_bytes: u64,
    ) -> Self {
        Self {
            documents,
            terms,
            total_tokens,
            segments,
            disk_bytes,
        }
    }

    /// Returns the total number of indexed documents.
    #[must_use]
    pub fn doc_count(&self) -> u64 {
        self.documents
    }

    /// Returns the total number of unique indexed terms.
    #[must_use]
    pub fn term_count(&self) -> u64 {
        self.terms
    }

    /// Returns the average number of tokens per document.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_doc_length(&self) -> f64 {
        if self.documents == 0 {
            return 0.0;
        }
        self.total_tokens as f64 / self.documents as f64
    }

    /// Returns the index disk usage in megabytes.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn disk_mb(&self) -> f64 {
        self.disk_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Returns `true` if the index has no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents == 0
    }
}

impl Default for IndexStats {
    fn default() -> Self {
        Self::new(0, 0, 0, 0, 0)
    }
}

/// Thresholds used by `IndexMonitor` to determine health.
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    /// Maximum number of segments before downgrading to Yellow.
    pub max_segments_yellow: usize,
    /// Minimum documents required to be non-empty.
    pub min_docs_green: u64,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            max_segments_yellow: 50,
            min_docs_green: 1,
        }
    }
}

/// Monitors an index and assesses its health.
#[derive(Debug)]
pub struct IndexMonitor {
    stats: IndexStats,
    thresholds: HealthThresholds,
    /// Whether the underlying index storage is reachable.
    storage_ok: bool,
}

impl IndexMonitor {
    /// Create a monitor from existing stats.
    #[must_use]
    pub fn new(stats: IndexStats) -> Self {
        Self {
            stats,
            thresholds: HealthThresholds::default(),
            storage_ok: true,
        }
    }

    /// Override health thresholds.
    #[must_use]
    pub fn with_thresholds(mut self, t: HealthThresholds) -> Self {
        self.thresholds = t;
        self
    }

    /// Mark storage as unavailable (forces Red health).
    pub fn mark_storage_unavailable(&mut self) {
        self.storage_ok = false;
    }

    /// Update the tracked statistics.
    pub fn update_stats(&mut self, stats: IndexStats) {
        self.stats = stats;
    }

    /// Assess and return the current index health.
    #[must_use]
    pub fn check_health(&self) -> IndexHealth {
        if !self.storage_ok {
            return IndexHealth::Red;
        }
        if self.stats.documents < self.thresholds.min_docs_green && self.stats.documents > 0 {
            return IndexHealth::Yellow;
        }
        if self.stats.segments > self.thresholds.max_segments_yellow {
            return IndexHealth::Yellow;
        }
        IndexHealth::Green
    }

    /// Returns a reference to the current statistics.
    #[must_use]
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(docs: u64, terms: u64, tokens: u64, segments: usize) -> IndexStats {
        IndexStats::new(docs, terms, tokens, segments, 1_048_576)
    }

    #[test]
    fn test_health_green_is_ok() {
        assert!(IndexHealth::Green.is_ok());
    }

    #[test]
    fn test_health_yellow_not_ok() {
        assert!(!IndexHealth::Yellow.is_ok());
    }

    #[test]
    fn test_health_red_not_ok() {
        assert!(!IndexHealth::Red.is_ok());
    }

    #[test]
    fn test_health_label() {
        assert_eq!(IndexHealth::Green.label(), "green");
        assert_eq!(IndexHealth::Yellow.label(), "yellow");
        assert_eq!(IndexHealth::Red.label(), "red");
    }

    #[test]
    fn test_health_display() {
        assert_eq!(IndexHealth::Red.to_string(), "red");
    }

    #[test]
    fn test_index_stats_doc_count() {
        let s = make_stats(100, 500, 2000, 3);
        assert_eq!(s.doc_count(), 100);
    }

    #[test]
    fn test_index_stats_term_count() {
        let s = make_stats(100, 500, 2000, 3);
        assert_eq!(s.term_count(), 500);
    }

    #[test]
    fn test_avg_doc_length() {
        let s = make_stats(10, 100, 300, 2);
        assert!((s.avg_doc_length() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_doc_length_zero_docs() {
        let s = IndexStats::default();
        assert_eq!(s.avg_doc_length(), 0.0);
    }

    #[test]
    fn test_disk_mb() {
        let s = IndexStats::new(10, 50, 200, 1, 2_097_152);
        assert!((s.disk_mb() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_empty_true() {
        let s = IndexStats::default();
        assert!(s.is_empty());
    }

    #[test]
    fn test_is_empty_false() {
        let s = make_stats(1, 10, 50, 1);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_monitor_healthy() {
        let s = make_stats(1000, 5000, 20000, 5);
        let m = IndexMonitor::new(s);
        assert_eq!(m.check_health(), IndexHealth::Green);
    }

    #[test]
    fn test_monitor_storage_unavailable_red() {
        let s = make_stats(1000, 5000, 20000, 5);
        let mut m = IndexMonitor::new(s);
        m.mark_storage_unavailable();
        assert_eq!(m.check_health(), IndexHealth::Red);
    }

    #[test]
    fn test_monitor_too_many_segments_yellow() {
        let s = make_stats(1000, 5000, 20000, 100);
        let m = IndexMonitor::new(s);
        assert_eq!(m.check_health(), IndexHealth::Yellow);
    }

    #[test]
    fn test_monitor_update_stats() {
        let s = make_stats(0, 0, 0, 0);
        let mut m = IndexMonitor::new(s);
        m.update_stats(make_stats(500, 2000, 8000, 4));
        assert_eq!(m.stats().doc_count(), 500);
    }
}

//! Metric cardinality limiter to prevent unbounded label growth.
//!
//! In high-cardinality environments (e.g. per-request tracing labels, dynamic
//! user IDs), metric stores can grow without bound.  This module enforces
//! configurable limits on the number of distinct label combinations allowed
//! per metric name, and provides overflow strategies when those limits are
//! exceeded.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Strategy applied when a metric's cardinality limit is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowStrategy {
    /// Silently drop new label combinations that exceed the limit.
    Drop,
    /// Merge excess label combinations into a catch-all `__overflow__` bucket.
    Merge,
    /// Evict the least-recently-used label combination to make room.
    EvictLru,
}

/// Configuration for the cardinality limiter.
#[derive(Debug, Clone)]
pub struct CardinalityConfig {
    /// Default maximum cardinality per metric.
    pub default_limit: usize,
    /// Per-metric overrides: metric name -> limit.
    pub metric_limits: HashMap<String, usize>,
    /// Strategy when limit is exceeded.
    pub overflow_strategy: OverflowStrategy,
    /// How often to log / report overflow events.
    pub report_interval: Duration,
}

impl Default for CardinalityConfig {
    fn default() -> Self {
        Self {
            default_limit: 1000,
            metric_limits: HashMap::new(),
            overflow_strategy: OverflowStrategy::Drop,
            report_interval: Duration::from_secs(60),
        }
    }
}

impl CardinalityConfig {
    /// Create a new config with the given default limit.
    #[must_use]
    pub fn new(default_limit: usize) -> Self {
        Self {
            default_limit: default_limit.max(1),
            ..Self::default()
        }
    }

    /// Set the overflow strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: OverflowStrategy) -> Self {
        self.overflow_strategy = strategy;
        self
    }

    /// Add a per-metric cardinality limit.
    #[must_use]
    pub fn with_metric_limit(mut self, metric: impl Into<String>, limit: usize) -> Self {
        self.metric_limits.insert(metric.into(), limit.max(1));
        self
    }

    /// Set the report interval.
    #[must_use]
    pub fn with_report_interval(mut self, interval: Duration) -> Self {
        self.report_interval = interval;
        self
    }
}

// ---------------------------------------------------------------------------
// Label key representation
// ---------------------------------------------------------------------------

/// A canonical representation of a label set (sorted key-value pairs).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LabelKey(String);

impl LabelKey {
    /// Build a canonical label key from an iterator of (key, value) pairs.
    pub fn from_pairs<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        let mut sorted: Vec<(&str, &str)> = pairs.into_iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(b.1)));
        let canonical: Vec<String> = sorted.iter().map(|(k, v)| format!("{k}={v}")).collect();
        Self(canonical.join(","))
    }

    /// Build from a single string (already canonical).
    #[must_use]
    pub fn raw(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// The overflow sentinel key.
    #[must_use]
    pub fn overflow() -> Self {
        Self("__overflow__".to_string())
    }

    /// Return the canonical string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns `true` if this is the overflow sentinel.
    #[must_use]
    pub fn is_overflow(&self) -> bool {
        self.0 == "__overflow__"
    }
}

// ---------------------------------------------------------------------------
// Per-metric state
// ---------------------------------------------------------------------------

/// Internal state for a single metric.
#[derive(Debug)]
struct MetricCardinalityState {
    /// Set of known label keys.
    known_keys: HashSet<LabelKey>,
    /// LRU order (front = oldest access, back = most recent).
    lru_order: VecDeque<LabelKey>,
    /// Count of dropped label combinations.
    drop_count: u64,
    /// Count of merged label combinations.
    merge_count: u64,
    /// Count of evicted label combinations.
    evict_count: u64,
}

impl MetricCardinalityState {
    fn new() -> Self {
        Self {
            known_keys: HashSet::new(),
            lru_order: VecDeque::new(),
            drop_count: 0,
            merge_count: 0,
            evict_count: 0,
        }
    }

    fn cardinality(&self) -> usize {
        self.known_keys.len()
    }

    /// Touch a key in the LRU order (move to back).
    fn touch_lru(&mut self, key: &LabelKey) {
        // Remove from current position.
        self.lru_order.retain(|k| k != key);
        // Add to back (most recent).
        self.lru_order.push_back(key.clone());
    }

    /// Evict the least-recently-used key.
    fn evict_lru(&mut self) -> Option<LabelKey> {
        if let Some(evicted) = self.lru_order.pop_front() {
            self.known_keys.remove(&evicted);
            self.evict_count += 1;
            Some(evicted)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Admission result
// ---------------------------------------------------------------------------

/// Result of attempting to admit a label combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionResult {
    /// The label combination was admitted normally.
    Admitted,
    /// The label combination already exists (no new cardinality).
    AlreadyKnown,
    /// The label combination was dropped (limit exceeded).
    Dropped,
    /// The label combination was merged into the overflow bucket.
    Merged,
    /// An old label combination was evicted to make room.
    Evicted {
        /// The evicted label key.
        evicted_key: LabelKey,
    },
}

// ---------------------------------------------------------------------------
// CardinalityLimiter
// ---------------------------------------------------------------------------

/// Statistics snapshot for the cardinality limiter.
#[derive(Debug, Clone)]
pub struct CardinalityStats {
    /// Metric name.
    pub metric: String,
    /// Current cardinality (number of distinct label combinations).
    pub cardinality: usize,
    /// Configured limit.
    pub limit: usize,
    /// Total times a label was dropped.
    pub dropped: u64,
    /// Total times a label was merged.
    pub merged: u64,
    /// Total times a label was evicted.
    pub evicted: u64,
}

/// Cardinality limiter that enforces per-metric label limits.
#[derive(Debug)]
pub struct CardinalityLimiter {
    config: CardinalityConfig,
    metrics: HashMap<String, MetricCardinalityState>,
    last_report: Instant,
}

impl CardinalityLimiter {
    /// Create a new limiter with the given configuration.
    #[must_use]
    pub fn new(config: CardinalityConfig) -> Self {
        Self {
            config,
            metrics: HashMap::new(),
            last_report: Instant::now(),
        }
    }

    /// Create a limiter with default configuration.
    #[must_use]
    pub fn with_default_limit(limit: usize) -> Self {
        Self::new(CardinalityConfig::new(limit))
    }

    /// Get the limit for a specific metric.
    #[must_use]
    pub fn limit_for(&self, metric: &str) -> usize {
        self.config
            .metric_limits
            .get(metric)
            .copied()
            .unwrap_or(self.config.default_limit)
    }

    /// Attempt to admit a label combination for a metric.
    ///
    /// Returns the admission result indicating what action was taken.
    pub fn admit(&mut self, metric: &str, label_key: LabelKey) -> AdmissionResult {
        let limit = self.limit_for(metric);
        let state = self
            .metrics
            .entry(metric.to_string())
            .or_insert_with(MetricCardinalityState::new);

        // Already known -- just touch LRU and return.
        if state.known_keys.contains(&label_key) {
            state.touch_lru(&label_key);
            return AdmissionResult::AlreadyKnown;
        }

        // Under limit -- admit freely.
        if state.cardinality() < limit {
            state.known_keys.insert(label_key.clone());
            state.touch_lru(&label_key);
            return AdmissionResult::Admitted;
        }

        // At or over limit -- apply overflow strategy.
        match self.config.overflow_strategy {
            OverflowStrategy::Drop => {
                state.drop_count += 1;
                AdmissionResult::Dropped
            }
            OverflowStrategy::Merge => {
                let overflow = LabelKey::overflow();
                if !state.known_keys.contains(&overflow) && state.cardinality() >= limit {
                    // We need to make room for the overflow key by evicting
                    // the LRU entry if we are exactly at limit.
                    if state.cardinality() >= limit {
                        state.evict_lru();
                    }
                    state.known_keys.insert(overflow.clone());
                    state.touch_lru(&overflow);
                }
                state.merge_count += 1;
                AdmissionResult::Merged
            }
            OverflowStrategy::EvictLru => {
                let evicted = state.evict_lru();
                state.known_keys.insert(label_key.clone());
                state.touch_lru(&label_key);
                if let Some(evicted_key) = evicted {
                    AdmissionResult::Evicted { evicted_key }
                } else {
                    // Edge case: nothing to evict but at limit (shouldn't happen).
                    AdmissionResult::Admitted
                }
            }
        }
    }

    /// Convenience: admit a label combination specified as pairs.
    pub fn admit_pairs<'a>(
        &mut self,
        metric: &str,
        pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> AdmissionResult {
        let key = LabelKey::from_pairs(pairs);
        self.admit(metric, key)
    }

    /// Get current cardinality for a metric.
    #[must_use]
    pub fn cardinality(&self, metric: &str) -> usize {
        self.metrics.get(metric).map_or(0, |s| s.cardinality())
    }

    /// Get statistics for a metric.
    #[must_use]
    pub fn stats(&self, metric: &str) -> Option<CardinalityStats> {
        let state = self.metrics.get(metric)?;
        Some(CardinalityStats {
            metric: metric.to_string(),
            cardinality: state.cardinality(),
            limit: self.limit_for(metric),
            dropped: state.drop_count,
            merged: state.merge_count,
            evicted: state.evict_count,
        })
    }

    /// Get statistics for all tracked metrics.
    #[must_use]
    pub fn all_stats(&self) -> Vec<CardinalityStats> {
        self.metrics
            .keys()
            .filter_map(|name| self.stats(name))
            .collect()
    }

    /// Returns `true` if a metric is at its cardinality limit.
    #[must_use]
    pub fn is_at_limit(&self, metric: &str) -> bool {
        self.cardinality(metric) >= self.limit_for(metric)
    }

    /// Total number of tracked metrics.
    #[must_use]
    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }

    /// Remove all state for a metric.
    pub fn remove_metric(&mut self, metric: &str) {
        self.metrics.remove(metric);
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.metrics.clear();
    }

    /// Check whether a report is due (based on the configured interval).
    #[must_use]
    pub fn should_report(&self) -> bool {
        self.last_report.elapsed() >= self.config.report_interval
    }

    /// Mark that a report was generated.
    pub fn mark_reported(&mut self) {
        self.last_report = Instant::now();
    }

    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &CardinalityConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- LabelKey --

    #[test]
    fn test_label_key_from_pairs_sorts() {
        let k1 = LabelKey::from_pairs([("b", "2"), ("a", "1")]);
        let k2 = LabelKey::from_pairs([("a", "1"), ("b", "2")]);
        assert_eq!(k1, k2);
        assert_eq!(k1.as_str(), "a=1,b=2");
    }

    #[test]
    fn test_label_key_empty() {
        let k = LabelKey::from_pairs(std::iter::empty());
        assert_eq!(k.as_str(), "");
    }

    #[test]
    fn test_label_key_overflow_sentinel() {
        let k = LabelKey::overflow();
        assert!(k.is_overflow());
        assert_eq!(k.as_str(), "__overflow__");
    }

    #[test]
    fn test_label_key_raw() {
        let k = LabelKey::raw("custom");
        assert_eq!(k.as_str(), "custom");
        assert!(!k.is_overflow());
    }

    // -- CardinalityConfig --

    #[test]
    fn test_config_default() {
        let cfg = CardinalityConfig::default();
        assert_eq!(cfg.default_limit, 1000);
        assert_eq!(cfg.overflow_strategy, OverflowStrategy::Drop);
    }

    #[test]
    fn test_config_builder() {
        let cfg = CardinalityConfig::new(100)
            .with_strategy(OverflowStrategy::Merge)
            .with_metric_limit("cpu_usage", 50)
            .with_report_interval(Duration::from_secs(120));
        assert_eq!(cfg.default_limit, 100);
        assert_eq!(cfg.overflow_strategy, OverflowStrategy::Merge);
        assert_eq!(cfg.metric_limits.get("cpu_usage").copied(), Some(50));
    }

    #[test]
    fn test_config_min_limit() {
        let cfg = CardinalityConfig::new(0);
        assert_eq!(cfg.default_limit, 1);
    }

    // -- CardinalityLimiter: Drop strategy --

    #[test]
    fn test_limiter_admit_under_limit() {
        let mut limiter = CardinalityLimiter::with_default_limit(3);
        let r = limiter.admit("cpu", LabelKey::raw("host=a"));
        assert_eq!(r, AdmissionResult::Admitted);
        assert_eq!(limiter.cardinality("cpu"), 1);
    }

    #[test]
    fn test_limiter_already_known() {
        let mut limiter = CardinalityLimiter::with_default_limit(3);
        limiter.admit("cpu", LabelKey::raw("host=a"));
        let r = limiter.admit("cpu", LabelKey::raw("host=a"));
        assert_eq!(r, AdmissionResult::AlreadyKnown);
        assert_eq!(limiter.cardinality("cpu"), 1);
    }

    #[test]
    fn test_limiter_drop_at_limit() {
        let mut limiter = CardinalityLimiter::new(
            CardinalityConfig::new(2).with_strategy(OverflowStrategy::Drop),
        );
        limiter.admit("m", LabelKey::raw("a"));
        limiter.admit("m", LabelKey::raw("b"));
        let r = limiter.admit("m", LabelKey::raw("c"));
        assert_eq!(r, AdmissionResult::Dropped);
        assert_eq!(limiter.cardinality("m"), 2);
    }

    #[test]
    fn test_limiter_drop_counts() {
        let mut limiter = CardinalityLimiter::new(
            CardinalityConfig::new(1).with_strategy(OverflowStrategy::Drop),
        );
        limiter.admit("m", LabelKey::raw("a"));
        limiter.admit("m", LabelKey::raw("b"));
        limiter.admit("m", LabelKey::raw("c"));
        let stats = limiter.stats("m").expect("stats should exist");
        assert_eq!(stats.dropped, 2);
        assert_eq!(stats.cardinality, 1);
    }

    // -- Merge strategy --

    #[test]
    fn test_limiter_merge_creates_overflow_bucket() {
        let mut limiter = CardinalityLimiter::new(
            CardinalityConfig::new(2).with_strategy(OverflowStrategy::Merge),
        );
        limiter.admit("m", LabelKey::raw("a"));
        limiter.admit("m", LabelKey::raw("b"));
        let r = limiter.admit("m", LabelKey::raw("c"));
        assert_eq!(r, AdmissionResult::Merged);
        let stats = limiter.stats("m").expect("stats should exist");
        assert_eq!(stats.merged, 1);
    }

    #[test]
    fn test_limiter_merge_multiple() {
        let mut limiter = CardinalityLimiter::new(
            CardinalityConfig::new(2).with_strategy(OverflowStrategy::Merge),
        );
        limiter.admit("m", LabelKey::raw("a"));
        limiter.admit("m", LabelKey::raw("b"));
        limiter.admit("m", LabelKey::raw("c"));
        limiter.admit("m", LabelKey::raw("d"));
        let stats = limiter.stats("m").expect("stats should exist");
        assert_eq!(stats.merged, 2);
    }

    // -- EvictLru strategy --

    #[test]
    fn test_limiter_evict_lru() {
        let mut limiter = CardinalityLimiter::new(
            CardinalityConfig::new(2).with_strategy(OverflowStrategy::EvictLru),
        );
        limiter.admit("m", LabelKey::raw("a"));
        limiter.admit("m", LabelKey::raw("b"));
        let r = limiter.admit("m", LabelKey::raw("c"));
        match r {
            AdmissionResult::Evicted { evicted_key } => {
                assert_eq!(evicted_key.as_str(), "a");
            }
            other => panic!("expected Evicted, got {other:?}"),
        }
        assert_eq!(limiter.cardinality("m"), 2);
    }

    #[test]
    fn test_limiter_evict_lru_respects_touch_order() {
        let mut limiter = CardinalityLimiter::new(
            CardinalityConfig::new(2).with_strategy(OverflowStrategy::EvictLru),
        );
        limiter.admit("m", LabelKey::raw("a"));
        limiter.admit("m", LabelKey::raw("b"));
        // Touch "a" again, so "b" becomes least-recently-used.
        limiter.admit("m", LabelKey::raw("a"));
        let r = limiter.admit("m", LabelKey::raw("c"));
        match r {
            AdmissionResult::Evicted { evicted_key } => {
                assert_eq!(evicted_key.as_str(), "b");
            }
            other => panic!("expected Evicted, got {other:?}"),
        }
    }

    #[test]
    fn test_limiter_evict_counts() {
        let mut limiter = CardinalityLimiter::new(
            CardinalityConfig::new(1).with_strategy(OverflowStrategy::EvictLru),
        );
        limiter.admit("m", LabelKey::raw("a"));
        limiter.admit("m", LabelKey::raw("b"));
        limiter.admit("m", LabelKey::raw("c"));
        let stats = limiter.stats("m").expect("stats should exist");
        assert_eq!(stats.evicted, 2);
        assert_eq!(stats.cardinality, 1);
    }

    // -- Per-metric limits --

    #[test]
    fn test_per_metric_limit_override() {
        let cfg = CardinalityConfig::new(100)
            .with_strategy(OverflowStrategy::Drop)
            .with_metric_limit("special", 2);
        let mut limiter = CardinalityLimiter::new(cfg);
        assert_eq!(limiter.limit_for("special"), 2);
        assert_eq!(limiter.limit_for("normal"), 100);

        limiter.admit("special", LabelKey::raw("a"));
        limiter.admit("special", LabelKey::raw("b"));
        let r = limiter.admit("special", LabelKey::raw("c"));
        assert_eq!(r, AdmissionResult::Dropped);
    }

    // -- admit_pairs --

    #[test]
    fn test_admit_pairs() {
        let mut limiter = CardinalityLimiter::with_default_limit(10);
        let r = limiter.admit_pairs("http_requests", [("method", "GET"), ("path", "/api")]);
        assert_eq!(r, AdmissionResult::Admitted);
        assert_eq!(limiter.cardinality("http_requests"), 1);
    }

    #[test]
    fn test_admit_pairs_canonical() {
        let mut limiter = CardinalityLimiter::with_default_limit(10);
        limiter.admit_pairs("m", [("b", "2"), ("a", "1")]);
        let r = limiter.admit_pairs("m", [("a", "1"), ("b", "2")]);
        assert_eq!(r, AdmissionResult::AlreadyKnown);
    }

    // -- Utility methods --

    #[test]
    fn test_is_at_limit() {
        let mut limiter = CardinalityLimiter::with_default_limit(1);
        assert!(!limiter.is_at_limit("m"));
        limiter.admit("m", LabelKey::raw("a"));
        assert!(limiter.is_at_limit("m"));
    }

    #[test]
    fn test_metric_count() {
        let mut limiter = CardinalityLimiter::with_default_limit(10);
        limiter.admit("a", LabelKey::raw("x"));
        limiter.admit("b", LabelKey::raw("y"));
        assert_eq!(limiter.metric_count(), 2);
    }

    #[test]
    fn test_remove_metric() {
        let mut limiter = CardinalityLimiter::with_default_limit(10);
        limiter.admit("m", LabelKey::raw("a"));
        limiter.remove_metric("m");
        assert_eq!(limiter.cardinality("m"), 0);
        assert_eq!(limiter.metric_count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut limiter = CardinalityLimiter::with_default_limit(10);
        limiter.admit("a", LabelKey::raw("1"));
        limiter.admit("b", LabelKey::raw("2"));
        limiter.clear();
        assert_eq!(limiter.metric_count(), 0);
    }

    #[test]
    fn test_all_stats() {
        let mut limiter = CardinalityLimiter::with_default_limit(10);
        limiter.admit("a", LabelKey::raw("1"));
        limiter.admit("b", LabelKey::raw("2"));
        let stats = limiter.all_stats();
        assert_eq!(stats.len(), 2);
    }

    #[test]
    fn test_stats_none_for_unknown() {
        let limiter = CardinalityLimiter::with_default_limit(10);
        assert!(limiter.stats("unknown").is_none());
    }

    #[test]
    fn test_should_report() {
        let cfg = CardinalityConfig::new(10).with_report_interval(Duration::from_millis(0));
        let limiter = CardinalityLimiter::new(cfg);
        assert!(limiter.should_report());
    }

    #[test]
    fn test_mark_reported() {
        let cfg = CardinalityConfig::new(10).with_report_interval(Duration::from_secs(3600));
        let mut limiter = CardinalityLimiter::new(cfg);
        limiter.mark_reported();
        assert!(!limiter.should_report());
    }

    // -- Stress test --

    #[test]
    fn test_high_cardinality_drop() {
        let mut limiter = CardinalityLimiter::new(
            CardinalityConfig::new(100).with_strategy(OverflowStrategy::Drop),
        );
        for i in 0..500 {
            limiter.admit("m", LabelKey::raw(format!("key_{i}")));
        }
        assert_eq!(limiter.cardinality("m"), 100);
        let stats = limiter.stats("m").expect("stats should exist");
        assert_eq!(stats.dropped, 400);
    }

    #[test]
    fn test_high_cardinality_evict() {
        let mut limiter = CardinalityLimiter::new(
            CardinalityConfig::new(50).with_strategy(OverflowStrategy::EvictLru),
        );
        for i in 0..200 {
            limiter.admit("m", LabelKey::raw(format!("key_{i}")));
        }
        assert_eq!(limiter.cardinality("m"), 50);
        let stats = limiter.stats("m").expect("stats should exist");
        assert_eq!(stats.evicted, 150);
    }
}

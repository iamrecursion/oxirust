//! Multi-resolution metric downsampling for historical data retention.
//!
//! Maintains time-series data at multiple resolution tiers:
//!
//! | Tier     | Resolution | Retention   |
//! |----------|-----------|-------------|
//! | Raw      | 1 second  | 1 hour      |
//! | Minute   | 1 minute  | 24 hours    |
//! | FiveMin  | 5 minutes | 30 days     |
//! | Hour     | 1 hour    | 365 days    |
//!
//! Each tier stores pre-aggregated data (min, max, mean, count) so that
//! queries at coarser granularity are fast and storage-efficient.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// A single resolution tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolutionTier {
    /// Raw samples (1-second granularity).
    Raw,
    /// 1-minute aggregates.
    Minute,
    /// 5-minute aggregates.
    FiveMinute,
    /// 1-hour aggregates.
    Hour,
}

impl ResolutionTier {
    /// Return the bucket width for this tier.
    #[must_use]
    pub fn bucket_width(self) -> Duration {
        match self {
            Self::Raw => Duration::from_secs(1),
            Self::Minute => Duration::from_secs(60),
            Self::FiveMinute => Duration::from_secs(300),
            Self::Hour => Duration::from_secs(3600),
        }
    }

    /// Return the default retention duration for this tier.
    #[must_use]
    pub fn default_retention(self) -> Duration {
        match self {
            Self::Raw => Duration::from_secs(3600),      // 1 hour
            Self::Minute => Duration::from_secs(86_400), // 24 hours
            Self::FiveMinute => Duration::from_secs(30 * 86_400), // 30 days
            Self::Hour => Duration::from_secs(365 * 86_400), // 365 days
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Minute => "1m",
            Self::FiveMinute => "5m",
            Self::Hour => "1h",
        }
    }
}

/// Configuration for the downsampler.
#[derive(Debug, Clone)]
pub struct DownsamplerConfig {
    /// Per-tier retention overrides.
    pub retention: HashMap<ResolutionTier, Duration>,
    /// Maximum number of metrics tracked.
    pub max_metrics: usize,
}

impl Default for DownsamplerConfig {
    fn default() -> Self {
        let mut retention = HashMap::new();
        retention.insert(ResolutionTier::Raw, ResolutionTier::Raw.default_retention());
        retention.insert(
            ResolutionTier::Minute,
            ResolutionTier::Minute.default_retention(),
        );
        retention.insert(
            ResolutionTier::FiveMinute,
            ResolutionTier::FiveMinute.default_retention(),
        );
        retention.insert(
            ResolutionTier::Hour,
            ResolutionTier::Hour.default_retention(),
        );
        Self {
            retention,
            max_metrics: 10_000,
        }
    }
}

impl DownsamplerConfig {
    /// Override the retention for a specific tier.
    #[must_use]
    pub fn with_retention(mut self, tier: ResolutionTier, dur: Duration) -> Self {
        self.retention.insert(tier, dur);
        self
    }

    /// Get retention for a tier.
    #[must_use]
    pub fn retention_for(&self, tier: ResolutionTier) -> Duration {
        self.retention
            .get(&tier)
            .copied()
            .unwrap_or_else(|| tier.default_retention())
    }
}

// ---------------------------------------------------------------------------
// Aggregated bucket
// ---------------------------------------------------------------------------

/// Pre-aggregated data for a single time bucket.
#[derive(Debug, Clone, Copy)]
pub struct AggBucket {
    /// Bucket start time (aligned to tier width).
    pub bucket_start: u64,
    /// Number of samples aggregated into this bucket.
    pub count: u64,
    /// Sum of all values.
    pub sum: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
}

impl AggBucket {
    /// Create a new bucket from a single sample.
    #[must_use]
    fn new(bucket_start: u64, value: f64) -> Self {
        Self {
            bucket_start,
            count: 1,
            sum: value,
            min: value,
            max: value,
        }
    }

    /// Merge another sample into this bucket.
    fn merge_value(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Merge another bucket into this one.
    fn merge_bucket(&mut self, other: &Self) {
        self.count += other.count;
        self.sum += other.sum;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }

    /// Mean value.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }
}

// ---------------------------------------------------------------------------
// Per-metric tier data
// ---------------------------------------------------------------------------

/// Storage for a single metric across all tiers.
#[derive(Debug)]
struct MetricTierData {
    /// Buckets keyed by bucket_start epoch seconds, per tier.
    tiers: HashMap<ResolutionTier, BTreeMap<u64, AggBucket>>,
}

impl MetricTierData {
    fn new() -> Self {
        let mut tiers = HashMap::new();
        tiers.insert(ResolutionTier::Raw, BTreeMap::new());
        tiers.insert(ResolutionTier::Minute, BTreeMap::new());
        tiers.insert(ResolutionTier::FiveMinute, BTreeMap::new());
        tiers.insert(ResolutionTier::Hour, BTreeMap::new());
        Self { tiers }
    }

    fn tier_buckets(&self, tier: ResolutionTier) -> &BTreeMap<u64, AggBucket> {
        static EMPTY: BTreeMap<u64, AggBucket> = BTreeMap::new();
        self.tiers.get(&tier).unwrap_or(&EMPTY)
    }

    fn tier_buckets_mut(&mut self, tier: ResolutionTier) -> &mut BTreeMap<u64, AggBucket> {
        self.tiers.entry(tier).or_default()
    }

    fn bucket_count(&self, tier: ResolutionTier) -> usize {
        self.tier_buckets(tier).len()
    }

    fn total_buckets(&self) -> usize {
        self.tiers.values().map(BTreeMap::len).sum()
    }
}

// ---------------------------------------------------------------------------
// Align helper
// ---------------------------------------------------------------------------

/// Align a unix epoch timestamp to the start of a bucket.
fn align_to_bucket(epoch_secs: u64, bucket_width_secs: u64) -> u64 {
    if bucket_width_secs == 0 {
        return epoch_secs;
    }
    (epoch_secs / bucket_width_secs) * bucket_width_secs
}

/// Get current epoch seconds.
fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

// ---------------------------------------------------------------------------
// MetricDownsampler
// ---------------------------------------------------------------------------

/// Multi-resolution metric downsampler.
#[derive(Debug)]
pub struct MetricDownsampler {
    config: DownsamplerConfig,
    data: HashMap<String, MetricTierData>,
}

impl MetricDownsampler {
    /// Create a new downsampler with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DownsamplerConfig::default(),
            data: HashMap::new(),
        }
    }

    /// Create a new downsampler with custom configuration.
    #[must_use]
    pub fn with_config(config: DownsamplerConfig) -> Self {
        Self {
            config,
            data: HashMap::new(),
        }
    }

    /// Record a sample for a metric at the current time.
    pub fn record(&mut self, metric: &str, value: f64) {
        self.record_at(metric, value, now_epoch_secs());
    }

    /// Record a sample for a metric at a specific epoch timestamp.
    pub fn record_at(&mut self, metric: &str, value: f64, epoch_secs: u64) {
        // Enforce max metrics limit.
        if !self.data.contains_key(metric) && self.data.len() >= self.config.max_metrics {
            return;
        }

        let tier_data = self
            .data
            .entry(metric.to_string())
            .or_insert_with(MetricTierData::new);

        // Insert into all tiers with appropriate alignment.
        let tiers = [
            ResolutionTier::Raw,
            ResolutionTier::Minute,
            ResolutionTier::FiveMinute,
            ResolutionTier::Hour,
        ];

        for tier in tiers {
            let width_secs = tier.bucket_width().as_secs();
            let bucket_start = align_to_bucket(epoch_secs, width_secs);
            let buckets = tier_data.tier_buckets_mut(tier);
            buckets
                .entry(bucket_start)
                .and_modify(|b| b.merge_value(value))
                .or_insert_with(|| AggBucket::new(bucket_start, value));
        }
    }

    /// Query data at a specific resolution tier within a time range.
    ///
    /// Returns buckets sorted by time.
    #[must_use]
    pub fn query(
        &self,
        metric: &str,
        tier: ResolutionTier,
        start_epoch: u64,
        end_epoch: u64,
    ) -> Vec<AggBucket> {
        let tier_data = match self.data.get(metric) {
            Some(d) => d,
            None => return Vec::new(),
        };

        tier_data
            .tier_buckets(tier)
            .range(start_epoch..=end_epoch)
            .map(|(_, b)| *b)
            .collect()
    }

    /// Query data and automatically select the best resolution tier for the
    /// requested time range.
    #[must_use]
    pub fn auto_query(&self, metric: &str, start_epoch: u64, end_epoch: u64) -> Vec<AggBucket> {
        let tier = self.best_tier_for_range(start_epoch, end_epoch);
        self.query(metric, tier, start_epoch, end_epoch)
    }

    /// Determine the best tier for a time range (larger ranges use coarser tiers).
    #[must_use]
    pub fn best_tier_for_range(&self, start_epoch: u64, end_epoch: u64) -> ResolutionTier {
        let range_secs = end_epoch.saturating_sub(start_epoch);
        if range_secs <= 3600 {
            ResolutionTier::Raw
        } else if range_secs <= 86_400 {
            ResolutionTier::Minute
        } else if range_secs <= 30 * 86_400 {
            ResolutionTier::FiveMinute
        } else {
            ResolutionTier::Hour
        }
    }

    /// Apply retention policies: remove buckets older than their tier's retention.
    pub fn apply_retention(&mut self) {
        let now = now_epoch_secs();
        let tiers = [
            ResolutionTier::Raw,
            ResolutionTier::Minute,
            ResolutionTier::FiveMinute,
            ResolutionTier::Hour,
        ];

        for tier_data in self.data.values_mut() {
            for &tier in &tiers {
                let retention_secs = self.config.retention_for(tier).as_secs();
                let cutoff = now.saturating_sub(retention_secs);
                let buckets = tier_data.tier_buckets_mut(tier);
                // Remove all buckets with keys < cutoff.
                let to_remove: Vec<u64> = buckets.range(..cutoff).map(|(&k, _)| k).collect();
                for k in to_remove {
                    buckets.remove(&k);
                }
            }
        }

        // Remove metrics with no data in any tier.
        self.data.retain(|_, td| td.total_buckets() > 0);
    }

    /// Apply retention with a custom "now" epoch (for testing).
    pub fn apply_retention_at(&mut self, now_epoch: u64) {
        let tiers = [
            ResolutionTier::Raw,
            ResolutionTier::Minute,
            ResolutionTier::FiveMinute,
            ResolutionTier::Hour,
        ];

        for tier_data in self.data.values_mut() {
            for &tier in &tiers {
                let retention_secs = self.config.retention_for(tier).as_secs();
                let cutoff = now_epoch.saturating_sub(retention_secs);
                let buckets = tier_data.tier_buckets_mut(tier);
                let to_remove: Vec<u64> = buckets.range(..cutoff).map(|(&k, _)| k).collect();
                for k in to_remove {
                    buckets.remove(&k);
                }
            }
        }

        self.data.retain(|_, td| td.total_buckets() > 0);
    }

    /// Get the number of buckets stored for a metric at a given tier.
    #[must_use]
    pub fn bucket_count(&self, metric: &str, tier: ResolutionTier) -> usize {
        self.data.get(metric).map_or(0, |td| td.bucket_count(tier))
    }

    /// Get the total number of buckets across all metrics and tiers.
    #[must_use]
    pub fn total_bucket_count(&self) -> usize {
        self.data.values().map(MetricTierData::total_buckets).sum()
    }

    /// Get the names of all tracked metrics.
    #[must_use]
    pub fn metric_names(&self) -> Vec<&str> {
        self.data.keys().map(String::as_str).collect()
    }

    /// Number of tracked metrics.
    #[must_use]
    pub fn metric_count(&self) -> usize {
        self.data.len()
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &DownsamplerConfig {
        &self.config
    }
}

impl Default for MetricDownsampler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Must be divisible by 3600 (1 hour) for predictable bucketing.
    // 1_699_999_200 / 3600 = 472222 exactly. And 1_699_999_200 / 300 = 5_666_664.
    const BASE_EPOCH: u64 = 1_699_999_200;

    // -- ResolutionTier --

    #[test]
    fn test_tier_bucket_width() {
        assert_eq!(ResolutionTier::Raw.bucket_width(), Duration::from_secs(1));
        assert_eq!(
            ResolutionTier::Minute.bucket_width(),
            Duration::from_secs(60)
        );
        assert_eq!(
            ResolutionTier::FiveMinute.bucket_width(),
            Duration::from_secs(300)
        );
        assert_eq!(
            ResolutionTier::Hour.bucket_width(),
            Duration::from_secs(3600)
        );
    }

    #[test]
    fn test_tier_labels() {
        assert_eq!(ResolutionTier::Raw.label(), "raw");
        assert_eq!(ResolutionTier::Minute.label(), "1m");
        assert_eq!(ResolutionTier::FiveMinute.label(), "5m");
        assert_eq!(ResolutionTier::Hour.label(), "1h");
    }

    // -- AggBucket --

    #[test]
    fn test_agg_bucket_single_value() {
        let b = AggBucket::new(1000, 42.0);
        assert_eq!(b.count, 1);
        assert!((b.mean() - 42.0).abs() < 1e-9);
        assert!((b.min - 42.0).abs() < 1e-9);
        assert!((b.max - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_agg_bucket_merge_values() {
        let mut b = AggBucket::new(1000, 10.0);
        b.merge_value(20.0);
        b.merge_value(30.0);
        assert_eq!(b.count, 3);
        assert!((b.mean() - 20.0).abs() < 1e-9);
        assert!((b.min - 10.0).abs() < 1e-9);
        assert!((b.max - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_agg_bucket_merge_bucket() {
        let mut b1 = AggBucket::new(1000, 10.0);
        b1.merge_value(20.0);
        let mut b2 = AggBucket::new(1000, 5.0);
        b2.merge_value(30.0);
        b1.merge_bucket(&b2);
        assert_eq!(b1.count, 4);
        assert!((b1.min - 5.0).abs() < 1e-9);
        assert!((b1.max - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_agg_bucket_mean_empty() {
        let b = AggBucket {
            bucket_start: 0,
            count: 0,
            sum: 0.0,
            min: 0.0,
            max: 0.0,
        };
        assert!((b.mean() - 0.0).abs() < 1e-9);
    }

    // -- align_to_bucket --

    #[test]
    fn test_align_to_bucket() {
        // align_to_bucket does floor division: (epoch / width) * width
        // 1700000042 / 60 = 28333334.03 -> 28333334 * 60 = 1700000040
        assert_eq!(align_to_bucket(1700000042, 60), 1700000040);
        // 1700000060 / 60 = 28333334.33 -> 28333334 * 60 = 1700000040
        assert_eq!(align_to_bucket(1700000060, 60), 1700000040);
        assert_eq!(align_to_bucket(1700000001, 1), 1700000001);
        // 1700000123 / 300 = 5666667.076 -> 5666667 * 300 = 1700000100
        assert_eq!(align_to_bucket(1700000123, 300), 1700000100);
        // Exact alignment: 1700000100 / 60 = 28333335 * 60 = 1700000100
        assert_eq!(align_to_bucket(1700000100, 60), 1700000100);
    }

    // -- MetricDownsampler basics --

    #[test]
    fn test_downsampler_new() {
        let ds = MetricDownsampler::new();
        assert_eq!(ds.metric_count(), 0);
        assert_eq!(ds.total_bucket_count(), 0);
    }

    #[test]
    fn test_record_at_creates_all_tiers() {
        let mut ds = MetricDownsampler::new();
        ds.record_at("cpu", 50.0, BASE_EPOCH);
        assert!(ds.bucket_count("cpu", ResolutionTier::Raw) > 0);
        assert!(ds.bucket_count("cpu", ResolutionTier::Minute) > 0);
        assert!(ds.bucket_count("cpu", ResolutionTier::FiveMinute) > 0);
        assert!(ds.bucket_count("cpu", ResolutionTier::Hour) > 0);
    }

    #[test]
    fn test_record_at_same_bucket_merges() {
        let mut ds = MetricDownsampler::new();
        ds.record_at("cpu", 10.0, BASE_EPOCH);
        ds.record_at("cpu", 20.0, BASE_EPOCH);
        // Both samples fall in the same raw 1-second bucket.
        let buckets = ds.query("cpu", ResolutionTier::Raw, BASE_EPOCH, BASE_EPOCH);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].count, 2);
        assert!((buckets[0].mean() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_different_seconds() {
        let mut ds = MetricDownsampler::new();
        ds.record_at("cpu", 10.0, BASE_EPOCH);
        ds.record_at("cpu", 20.0, BASE_EPOCH + 1);
        ds.record_at("cpu", 30.0, BASE_EPOCH + 2);
        let buckets = ds.query("cpu", ResolutionTier::Raw, BASE_EPOCH, BASE_EPOCH + 2);
        assert_eq!(buckets.len(), 3);
    }

    #[test]
    fn test_minute_aggregation() {
        let mut ds = MetricDownsampler::new();
        // Record 60 samples across 1 minute.
        for i in 0..60 {
            ds.record_at("cpu", i as f64, BASE_EPOCH + i);
        }
        // All 60 should be in one minute bucket.
        let minute_buckets = ds.query("cpu", ResolutionTier::Minute, BASE_EPOCH, BASE_EPOCH + 59);
        assert_eq!(minute_buckets.len(), 1);
        assert_eq!(minute_buckets[0].count, 60);
        assert!((minute_buckets[0].min - 0.0).abs() < 1e-9);
        assert!((minute_buckets[0].max - 59.0).abs() < 1e-9);
    }

    #[test]
    fn test_five_minute_aggregation() {
        let mut ds = MetricDownsampler::new();
        // Record 300 samples across 5 minutes.
        for i in 0..300 {
            ds.record_at("m", 1.0, BASE_EPOCH + i);
        }
        let buckets = ds.query(
            "m",
            ResolutionTier::FiveMinute,
            BASE_EPOCH,
            BASE_EPOCH + 299,
        );
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].count, 300);
    }

    #[test]
    fn test_hour_aggregation() {
        let mut ds = MetricDownsampler::new();
        // Record 3600 samples across 1 hour.
        for i in 0..3600 {
            ds.record_at("m", 1.0, BASE_EPOCH + i);
        }
        let buckets = ds.query("m", ResolutionTier::Hour, BASE_EPOCH, BASE_EPOCH + 3599);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].count, 3600);
    }

    // -- Query --

    #[test]
    fn test_query_empty_metric() {
        let ds = MetricDownsampler::new();
        let buckets = ds.query("unknown", ResolutionTier::Raw, 0, u64::MAX);
        assert!(buckets.is_empty());
    }

    #[test]
    fn test_query_range_filter() {
        let mut ds = MetricDownsampler::new();
        ds.record_at("cpu", 10.0, BASE_EPOCH);
        ds.record_at("cpu", 20.0, BASE_EPOCH + 120);
        // Query only the first 60 seconds.
        let buckets = ds.query("cpu", ResolutionTier::Raw, BASE_EPOCH, BASE_EPOCH + 60);
        assert_eq!(buckets.len(), 1);
        assert!((buckets[0].mean() - 10.0).abs() < 1e-9);
    }

    // -- Auto query --

    #[test]
    fn test_best_tier_for_range() {
        let ds = MetricDownsampler::new();
        assert_eq!(ds.best_tier_for_range(0, 1800), ResolutionTier::Raw);
        assert_eq!(ds.best_tier_for_range(0, 3600), ResolutionTier::Raw);
        assert_eq!(ds.best_tier_for_range(0, 3601), ResolutionTier::Minute);
        assert_eq!(ds.best_tier_for_range(0, 86_400), ResolutionTier::Minute);
        assert_eq!(
            ds.best_tier_for_range(0, 86_401),
            ResolutionTier::FiveMinute
        );
        assert_eq!(ds.best_tier_for_range(0, 31 * 86_400), ResolutionTier::Hour);
    }

    // -- Retention --

    #[test]
    fn test_apply_retention_removes_old() {
        let cfg = DownsamplerConfig::default()
            .with_retention(ResolutionTier::Raw, Duration::from_secs(60));
        let mut ds = MetricDownsampler::with_config(cfg);

        // Record at BASE_EPOCH.
        ds.record_at("cpu", 10.0, BASE_EPOCH);
        // Record at BASE_EPOCH + 180 (within retention window of now=200, cutoff=140).
        ds.record_at("cpu", 20.0, BASE_EPOCH + 180);

        // Apply retention as if "now" is BASE_EPOCH + 200.
        // Raw retention = 60s, so cutoff = 200 - 60 = 140.
        // The sample at BASE_EPOCH (< 140) should be removed from raw tier.
        // The sample at BASE_EPOCH+180 (>= 140) should remain.
        ds.apply_retention_at(BASE_EPOCH + 200);

        let raw_buckets = ds.query("cpu", ResolutionTier::Raw, BASE_EPOCH, BASE_EPOCH + 201);
        // Only the sample at BASE_EPOCH+180 should remain in raw.
        assert_eq!(raw_buckets.len(), 1);
        assert!((raw_buckets[0].mean() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_retention_removes_empty_metrics() {
        let cfg = DownsamplerConfig::default()
            .with_retention(ResolutionTier::Raw, Duration::from_secs(10))
            .with_retention(ResolutionTier::Minute, Duration::from_secs(10))
            .with_retention(ResolutionTier::FiveMinute, Duration::from_secs(10))
            .with_retention(ResolutionTier::Hour, Duration::from_secs(10));
        let mut ds = MetricDownsampler::with_config(cfg);

        ds.record_at("cpu", 10.0, BASE_EPOCH);
        // Apply retention as if now is BASE_EPOCH + 1000 (everything is old).
        ds.apply_retention_at(BASE_EPOCH + 1000);
        assert_eq!(ds.metric_count(), 0);
    }

    // -- Max metrics limit --

    #[test]
    fn test_max_metrics_limit() {
        let cfg = DownsamplerConfig {
            max_metrics: 3,
            ..DownsamplerConfig::default()
        };
        let mut ds = MetricDownsampler::with_config(cfg);
        for i in 0..10 {
            ds.record_at(&format!("m{i}"), 1.0, BASE_EPOCH);
        }
        assert_eq!(ds.metric_count(), 3);
    }

    // -- Utility methods --

    #[test]
    fn test_metric_names() {
        let mut ds = MetricDownsampler::new();
        ds.record_at("cpu", 1.0, BASE_EPOCH);
        ds.record_at("mem", 2.0, BASE_EPOCH);
        let mut names = ds.metric_names();
        names.sort();
        assert_eq!(names, vec!["cpu", "mem"]);
    }

    #[test]
    fn test_clear() {
        let mut ds = MetricDownsampler::new();
        ds.record_at("cpu", 1.0, BASE_EPOCH);
        ds.clear();
        assert_eq!(ds.metric_count(), 0);
        assert_eq!(ds.total_bucket_count(), 0);
    }

    // -- Multi-metric isolation --

    #[test]
    fn test_different_metrics_isolated() {
        let mut ds = MetricDownsampler::new();
        ds.record_at("cpu", 50.0, BASE_EPOCH);
        ds.record_at("mem", 80.0, BASE_EPOCH);

        let cpu_buckets = ds.query("cpu", ResolutionTier::Raw, BASE_EPOCH, BASE_EPOCH);
        let mem_buckets = ds.query("mem", ResolutionTier::Raw, BASE_EPOCH, BASE_EPOCH);

        assert_eq!(cpu_buckets.len(), 1);
        assert_eq!(mem_buckets.len(), 1);
        assert!((cpu_buckets[0].mean() - 50.0).abs() < 1e-9);
        assert!((mem_buckets[0].mean() - 80.0).abs() < 1e-9);
    }

    // -- Statistics accuracy --

    #[test]
    fn test_aggregate_statistics_accuracy() {
        let mut ds = MetricDownsampler::new();
        let values = [10.0, 20.0, 30.0, 40.0, 50.0];
        // Record in same minute bucket.
        for (i, &v) in values.iter().enumerate() {
            ds.record_at("m", v, BASE_EPOCH + i as u64);
        }

        let buckets = ds.query("m", ResolutionTier::Minute, BASE_EPOCH, BASE_EPOCH + 10);
        assert_eq!(buckets.len(), 1);
        let b = &buckets[0];
        assert_eq!(b.count, 5);
        assert!((b.min - 10.0).abs() < 1e-9);
        assert!((b.max - 50.0).abs() < 1e-9);
        assert!((b.mean() - 30.0).abs() < 1e-9);
        assert!((b.sum - 150.0).abs() < 1e-9);
    }
}

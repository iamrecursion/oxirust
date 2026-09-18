#![allow(dead_code)]
//! Traffic shaping and QoS for media streams.
//!
//! Provides token-bucket and leaky-bucket rate limiters, priority queuing,
//! and bandwidth allocation for ensuring quality of service on media
//! routing networks.

use std::collections::HashMap;

/// Priority class for media traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TrafficClass {
    /// Highest priority — live program audio/video.
    Program,
    /// Preview / monitoring feeds.
    Preview,
    /// File transfers and non-real-time traffic.
    BulkTransfer,
    /// Best-effort (lowest priority).
    BestEffort,
}

/// Configuration for a token-bucket rate limiter.
#[derive(Debug, Clone, Copy)]
pub struct TokenBucketConfig {
    /// Maximum burst size in bytes.
    pub burst_bytes: u64,
    /// Sustained rate in bytes per second.
    pub rate_bps: u64,
}

impl TokenBucketConfig {
    /// Create a new token-bucket configuration.
    pub fn new(burst_bytes: u64, rate_bps: u64) -> Self {
        Self {
            burst_bytes: burst_bytes.max(1),
            rate_bps: rate_bps.max(1),
        }
    }
}

/// Token-bucket rate limiter state.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Configuration.
    config: TokenBucketConfig,
    /// Current number of available tokens (bytes).
    tokens: u64,
    /// Last refill timestamp in microseconds.
    last_refill_us: u64,
}

impl TokenBucket {
    /// Create a new token bucket (starts full).
    pub fn new(config: TokenBucketConfig) -> Self {
        Self {
            tokens: config.burst_bytes,
            last_refill_us: 0,
            config,
        }
    }

    /// Refill tokens based on elapsed time.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn refill(&mut self, now_us: u64) {
        if now_us <= self.last_refill_us {
            return;
        }
        let elapsed_us = now_us - self.last_refill_us;
        let new_tokens = (self.config.rate_bps as f64 * elapsed_us as f64 / 1_000_000.0) as u64;
        self.tokens = (self.tokens + new_tokens).min(self.config.burst_bytes);
        self.last_refill_us = now_us;
    }

    /// Attempt to consume `bytes` tokens. Returns `true` if allowed.
    pub fn consume(&mut self, bytes: u64) -> bool {
        if self.tokens >= bytes {
            self.tokens -= bytes;
            true
        } else {
            false
        }
    }

    /// Current token count.
    pub fn available(&self) -> u64 {
        self.tokens
    }

    /// Fraction of the bucket that is full [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn fill_ratio(&self) -> f64 {
        self.tokens as f64 / self.config.burst_bytes as f64
    }

    /// The bucket configuration.
    pub fn config(&self) -> &TokenBucketConfig {
        &self.config
    }
}

/// A QoS policy that assigns bandwidth limits per traffic class.
#[derive(Debug, Clone)]
pub struct QosPolicy {
    /// Human-readable policy name.
    pub name: String,
    /// Per-class bandwidth limits in bytes per second.
    class_limits: HashMap<TrafficClass, u64>,
    /// Total bandwidth budget in bytes per second.
    pub total_budget_bps: u64,
}

impl QosPolicy {
    /// Create a new QoS policy with a total budget.
    pub fn new(name: impl Into<String>, total_budget_bps: u64) -> Self {
        Self {
            name: name.into(),
            class_limits: HashMap::new(),
            total_budget_bps,
        }
    }

    /// Set the bandwidth limit for a traffic class.
    pub fn set_class_limit(&mut self, class: TrafficClass, limit_bps: u64) {
        self.class_limits.insert(class, limit_bps);
    }

    /// Get the limit for a traffic class.
    pub fn class_limit(&self, class: TrafficClass) -> Option<u64> {
        self.class_limits.get(&class).copied()
    }

    /// Sum of all class limits.
    pub fn allocated_bps(&self) -> u64 {
        self.class_limits.values().sum()
    }

    /// Whether the allocation exceeds the total budget.
    pub fn is_oversubscribed(&self) -> bool {
        self.allocated_bps() > self.total_budget_bps
    }

    /// Remaining unallocated bandwidth.
    pub fn unallocated_bps(&self) -> u64 {
        self.total_budget_bps.saturating_sub(self.allocated_bps())
    }

    /// Number of classes with limits set.
    pub fn class_count(&self) -> usize {
        self.class_limits.len()
    }
}

/// Statistics for a shaped stream.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamStats {
    /// Total bytes admitted.
    pub bytes_admitted: u64,
    /// Total bytes dropped (rate-exceeded).
    pub bytes_dropped: u64,
    /// Number of admit decisions.
    pub admits: u64,
    /// Number of drop decisions.
    pub drops: u64,
}

impl StreamStats {
    /// Drop ratio as a fraction [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn drop_ratio(&self) -> f64 {
        let total = self.admits + self.drops;
        if total == 0 {
            0.0
        } else {
            self.drops as f64 / total as f64
        }
    }
}

/// Unique identifier for a shaped stream.
pub type StreamId = u64;

/// A traffic shaper managing multiple streams with per-class QoS.
#[derive(Debug)]
pub struct TrafficShaper {
    /// Active QoS policy.
    policy: QosPolicy,
    /// Per-stream token buckets.
    buckets: HashMap<StreamId, TokenBucket>,
    /// Per-stream traffic class assignment.
    stream_classes: HashMap<StreamId, TrafficClass>,
    /// Per-stream statistics.
    stats: HashMap<StreamId, StreamStats>,
    /// Next auto-assigned stream ID.
    next_stream_id: StreamId,
    /// Current clock in microseconds.
    clock_us: u64,
}

impl TrafficShaper {
    /// Create a new traffic shaper with a QoS policy.
    pub fn new(policy: QosPolicy) -> Self {
        Self {
            policy,
            buckets: HashMap::new(),
            stream_classes: HashMap::new(),
            stats: HashMap::new(),
            next_stream_id: 1,
            clock_us: 0,
        }
    }

    /// Register a stream and return its ID.
    pub fn register_stream(
        &mut self,
        class: TrafficClass,
        burst_bytes: u64,
        rate_bps: u64,
    ) -> StreamId {
        let id = self.next_stream_id;
        self.next_stream_id += 1;
        let config = TokenBucketConfig::new(burst_bytes, rate_bps);
        self.buckets.insert(id, TokenBucket::new(config));
        self.stream_classes.insert(id, class);
        self.stats.insert(id, StreamStats::default());
        id
    }

    /// Advance the shaper clock and refill all buckets.
    pub fn advance_clock(&mut self, now_us: u64) {
        self.clock_us = now_us;
        for bucket in self.buckets.values_mut() {
            bucket.refill(now_us);
        }
    }

    /// Attempt to admit `bytes` on a stream. Returns `true` if admitted.
    pub fn admit(&mut self, stream_id: StreamId, bytes: u64) -> bool {
        let bucket = match self.buckets.get_mut(&stream_id) {
            Some(b) => b,
            None => return false,
        };
        let stats = self.stats.entry(stream_id).or_default();

        if bucket.consume(bytes) {
            stats.bytes_admitted += bytes;
            stats.admits += 1;
            true
        } else {
            stats.bytes_dropped += bytes;
            stats.drops += 1;
            false
        }
    }

    /// Get statistics for a stream.
    pub fn stream_stats(&self, stream_id: StreamId) -> Option<&StreamStats> {
        self.stats.get(&stream_id)
    }

    /// Get the traffic class assigned to a stream.
    pub fn stream_class(&self, stream_id: StreamId) -> Option<TrafficClass> {
        self.stream_classes.get(&stream_id).copied()
    }

    /// Number of registered streams.
    pub fn stream_count(&self) -> usize {
        self.buckets.len()
    }

    /// Get the active QoS policy.
    pub fn policy(&self) -> &QosPolicy {
        &self.policy
    }

    /// Replace the QoS policy.
    pub fn set_policy(&mut self, policy: QosPolicy) {
        self.policy = policy;
    }

    /// Aggregate statistics across all streams.
    pub fn aggregate_stats(&self) -> StreamStats {
        let mut agg = StreamStats::default();
        for s in self.stats.values() {
            agg.bytes_admitted += s.bytes_admitted;
            agg.bytes_dropped += s.bytes_dropped;
            agg.admits += s.admits;
            agg.drops += s.drops;
        }
        agg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_config() {
        let cfg = TokenBucketConfig::new(1000, 500);
        assert_eq!(cfg.burst_bytes, 1000);
        assert_eq!(cfg.rate_bps, 500);
    }

    #[test]
    fn test_token_bucket_consume() {
        let cfg = TokenBucketConfig::new(100, 100);
        let mut tb = TokenBucket::new(cfg);
        assert_eq!(tb.available(), 100);
        assert!(tb.consume(50));
        assert_eq!(tb.available(), 50);
        assert!(!tb.consume(60));
        assert_eq!(tb.available(), 50);
    }

    #[test]
    fn test_token_bucket_refill() {
        let cfg = TokenBucketConfig::new(1000, 1000); // 1000 bytes/s
        let mut tb = TokenBucket::new(cfg);
        assert!(tb.consume(1000)); // empty
        assert_eq!(tb.available(), 0);
        tb.refill(500_000); // 0.5 seconds later => +500 bytes
        assert_eq!(tb.available(), 500);
    }

    #[test]
    fn test_token_bucket_fill_ratio() {
        let cfg = TokenBucketConfig::new(200, 100);
        let mut tb = TokenBucket::new(cfg);
        assert!((tb.fill_ratio() - 1.0).abs() < f64::EPSILON);
        tb.consume(100);
        assert!((tb.fill_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qos_policy_basics() {
        let mut pol = QosPolicy::new("Default", 10_000_000);
        pol.set_class_limit(TrafficClass::Program, 6_000_000);
        pol.set_class_limit(TrafficClass::Preview, 3_000_000);
        assert_eq!(pol.class_count(), 2);
        assert_eq!(pol.allocated_bps(), 9_000_000);
        assert!(!pol.is_oversubscribed());
        assert_eq!(pol.unallocated_bps(), 1_000_000);
    }

    #[test]
    fn test_qos_policy_oversubscribed() {
        let mut pol = QosPolicy::new("Over", 100);
        pol.set_class_limit(TrafficClass::Program, 80);
        pol.set_class_limit(TrafficClass::Preview, 80);
        assert!(pol.is_oversubscribed());
    }

    #[test]
    fn test_stream_stats_drop_ratio() {
        let s = StreamStats {
            bytes_admitted: 900,
            bytes_dropped: 100,
            admits: 9,
            drops: 1,
        };
        assert!((s.drop_ratio() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stream_stats_zero() {
        let s = StreamStats::default();
        assert!((s.drop_ratio()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_shaper_register_stream() {
        let pol = QosPolicy::new("Test", 1_000_000);
        let mut shaper = TrafficShaper::new(pol);
        let sid = shaper.register_stream(TrafficClass::Program, 1000, 10000);
        assert_eq!(sid, 1);
        assert_eq!(shaper.stream_count(), 1);
        assert_eq!(shaper.stream_class(sid), Some(TrafficClass::Program));
    }

    #[test]
    fn test_shaper_admit_and_drop() {
        let pol = QosPolicy::new("Test", 1_000_000);
        let mut shaper = TrafficShaper::new(pol);
        let sid = shaper.register_stream(TrafficClass::Program, 100, 1000);

        assert!(shaper.admit(sid, 50));
        assert!(shaper.admit(sid, 50));
        assert!(!shaper.admit(sid, 10)); // bucket empty

        let stats = shaper.stream_stats(sid).expect("should succeed in test");
        assert_eq!(stats.admits, 2);
        assert_eq!(stats.drops, 1);
    }

    #[test]
    fn test_shaper_advance_clock_refills() {
        let pol = QosPolicy::new("Test", 1_000_000);
        let mut shaper = TrafficShaper::new(pol);
        let sid = shaper.register_stream(TrafficClass::Preview, 1000, 1_000_000);

        shaper.admit(sid, 1000); // drain
        assert!(!shaper.admit(sid, 1)); // empty

        shaper.advance_clock(500_000); // +500 bytes
        assert!(shaper.admit(sid, 100)); // should succeed now
    }

    #[test]
    fn test_shaper_aggregate_stats() {
        let pol = QosPolicy::new("Test", 10_000_000);
        let mut shaper = TrafficShaper::new(pol);
        let s1 = shaper.register_stream(TrafficClass::Program, 100, 1000);
        let s2 = shaper.register_stream(TrafficClass::Preview, 100, 1000);

        shaper.admit(s1, 50);
        shaper.admit(s2, 30);

        let agg = shaper.aggregate_stats();
        assert_eq!(agg.bytes_admitted, 80);
        assert_eq!(agg.admits, 2);
    }

    #[test]
    fn test_traffic_class_ordering() {
        assert!(TrafficClass::Program < TrafficClass::Preview);
        assert!(TrafficClass::Preview < TrafficClass::BulkTransfer);
        assert!(TrafficClass::BulkTransfer < TrafficClass::BestEffort);
    }
}

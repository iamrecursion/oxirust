#![allow(dead_code)]
//! Bandwidth throttling and rate-limiting for network transfers.
//!
//! Provides a token-bucket [`BandwidthThrottle`] that controls how many bytes
//! may be sent or received per unit of time, plus per-stream and aggregate
//! limiting through [`ThrottleGroup`].

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Unique identifier for a throttled stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(u64);

impl StreamId {
    /// Creates a new stream identifier.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the raw numeric value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Priority class for bandwidth allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThrottlePriority {
    /// Low priority — yields bandwidth to higher classes.
    Low,
    /// Normal priority.
    Normal,
    /// High priority — gets bandwidth first.
    High,
    /// Critical — never throttled (bypass).
    Critical,
}

impl ThrottlePriority {
    /// Returns a weight used for weighted fair queueing.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn weight(self) -> f64 {
        match self {
            Self::Low => 1.0,
            Self::Normal => 4.0,
            Self::High => 8.0,
            Self::Critical => 16.0,
        }
    }
}

/// Configuration for a bandwidth throttle.
#[derive(Debug, Clone)]
pub struct ThrottleConfig {
    /// Maximum bytes per second.
    pub rate_bytes_per_sec: u64,
    /// Burst size in bytes (token bucket capacity).
    pub burst_bytes: u64,
    /// Refill interval granularity.
    pub refill_interval: Duration,
}

impl ThrottleConfig {
    /// Creates a new configuration with the given rate and burst equal to rate.
    #[must_use]
    pub fn new(rate_bytes_per_sec: u64) -> Self {
        Self {
            rate_bytes_per_sec,
            burst_bytes: rate_bytes_per_sec,
            refill_interval: Duration::from_millis(50),
        }
    }

    /// Sets the burst capacity.
    #[must_use]
    pub const fn with_burst(mut self, burst_bytes: u64) -> Self {
        self.burst_bytes = burst_bytes;
        self
    }

    /// Sets the refill interval.
    #[must_use]
    pub const fn with_refill_interval(mut self, interval: Duration) -> Self {
        self.refill_interval = interval;
        self
    }

    /// Convenience constructor for a rate given in megabits per second.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn from_mbps(mbps: f64) -> Self {
        let bytes_per_sec = (mbps * 125_000.0) as u64; // 1 Mbps = 125,000 B/s
        Self::new(bytes_per_sec)
    }
}

/// A token-bucket bandwidth throttle.
#[derive(Debug)]
pub struct BandwidthThrottle {
    /// Configuration for this throttle.
    config: ThrottleConfig,
    /// Current number of available tokens (bytes).
    tokens: u64,
    /// When the tokens were last refilled.
    last_refill: Instant,
    /// Total bytes consumed through this throttle.
    total_consumed: u64,
    /// When the throttle was created.
    created_at: Instant,
}

impl BandwidthThrottle {
    /// Creates a new throttle from the given configuration.
    #[must_use]
    pub fn new(config: ThrottleConfig) -> Self {
        let now = Instant::now();
        Self {
            tokens: config.burst_bytes,
            config,
            last_refill: now,
            total_consumed: 0,
            created_at: now,
        }
    }

    /// Creates a throttle for a rate in bytes per second.
    #[must_use]
    pub fn with_rate(bytes_per_sec: u64) -> Self {
        Self::new(ThrottleConfig::new(bytes_per_sec))
    }

    /// Returns the current token count (available bytes).
    #[must_use]
    pub fn available(&self) -> u64 {
        self.tokens
    }

    /// Returns total bytes consumed since creation.
    #[must_use]
    pub fn total_consumed(&self) -> u64 {
        self.total_consumed
    }

    /// Refills tokens based on elapsed time.
    pub fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed();
        if elapsed < self.config.refill_interval {
            return;
        }

        #[allow(clippy::cast_precision_loss)]
        let new_tokens = (self.config.rate_bytes_per_sec as f64 * elapsed.as_secs_f64()) as u64;
        self.tokens = (self.tokens + new_tokens).min(self.config.burst_bytes);
        self.last_refill = Instant::now();
    }

    /// Attempts to consume `bytes` tokens. Returns the number of bytes
    /// actually consumed (may be less than requested).
    pub fn consume(&mut self, bytes: u64) -> u64 {
        self.refill();
        let consumed = bytes.min(self.tokens);
        self.tokens -= consumed;
        self.total_consumed += consumed;
        consumed
    }

    /// Returns `true` if at least `bytes` tokens are available.
    pub fn can_send(&mut self, bytes: u64) -> bool {
        self.refill();
        self.tokens >= bytes
    }

    /// Returns the estimated wait time until `bytes` tokens become available.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn wait_time(&self, bytes: u64) -> Duration {
        if self.tokens >= bytes {
            return Duration::ZERO;
        }
        let deficit = bytes - self.tokens;
        if self.config.rate_bytes_per_sec == 0 {
            return Duration::from_secs(u64::MAX);
        }
        let secs = deficit as f64 / self.config.rate_bytes_per_sec as f64;
        Duration::from_secs_f64(secs)
    }

    /// Returns the effective throughput in bytes per second since creation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn effective_rate(&self) -> f64 {
        let elapsed = self.created_at.elapsed().as_secs_f64();
        if elapsed < f64::EPSILON {
            return 0.0;
        }
        self.total_consumed as f64 / elapsed
    }

    /// Updates the rate limit without resetting token count.
    pub fn set_rate(&mut self, bytes_per_sec: u64) {
        self.config.rate_bytes_per_sec = bytes_per_sec;
        self.config.burst_bytes = bytes_per_sec;
    }

    /// Returns the configured rate in bytes per second.
    #[must_use]
    pub fn rate_bytes_per_sec(&self) -> u64 {
        self.config.rate_bytes_per_sec
    }
}

/// Per-stream throttle entry.
#[derive(Debug)]
struct StreamThrottle {
    /// The throttle for this stream.
    throttle: BandwidthThrottle,
    /// Priority class.
    priority: ThrottlePriority,
}

/// Manages bandwidth throttling across multiple streams.
#[derive(Debug)]
pub struct ThrottleGroup {
    /// Aggregate throttle for the group.
    aggregate: BandwidthThrottle,
    /// Per-stream throttles.
    streams: HashMap<StreamId, StreamThrottle>,
    /// Next stream id.
    next_id: u64,
}

impl ThrottleGroup {
    /// Creates a new throttle group with the given aggregate rate limit.
    #[must_use]
    pub fn new(aggregate_rate: u64) -> Self {
        Self {
            aggregate: BandwidthThrottle::with_rate(aggregate_rate),
            streams: HashMap::new(),
            next_id: 1,
        }
    }

    /// Adds a stream with the given per-stream rate and priority.
    ///
    /// Returns the assigned [`StreamId`].
    pub fn add_stream(&mut self, per_stream_rate: u64, priority: ThrottlePriority) -> StreamId {
        let id = StreamId::new(self.next_id);
        self.next_id += 1;
        self.streams.insert(
            id,
            StreamThrottle {
                throttle: BandwidthThrottle::with_rate(per_stream_rate),
                priority,
            },
        );
        id
    }

    /// Removes a stream from the group.
    pub fn remove_stream(&mut self, id: StreamId) -> bool {
        self.streams.remove(&id).is_some()
    }

    /// Attempts to consume `bytes` on the given stream, respecting both
    /// the per-stream and aggregate limits.
    ///
    /// Returns the number of bytes actually allowed.
    pub fn consume(&mut self, id: StreamId, bytes: u64) -> u64 {
        // Aggregate limit first.
        let agg_allowed = self.aggregate.consume(bytes);
        if agg_allowed == 0 {
            return 0;
        }
        // Per-stream limit.
        if let Some(entry) = self.streams.get_mut(&id) {
            let stream_allowed = entry.throttle.consume(agg_allowed);
            // Return unused aggregate tokens.
            let unused = agg_allowed - stream_allowed;
            self.aggregate.tokens += unused;
            self.aggregate.total_consumed -= unused;
            stream_allowed
        } else {
            // Stream not found — return tokens.
            self.aggregate.tokens += agg_allowed;
            self.aggregate.total_consumed -= agg_allowed;
            0
        }
    }

    /// Returns the number of streams in the group.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Returns aggregate total bytes consumed.
    #[must_use]
    pub fn total_consumed(&self) -> u64 {
        self.aggregate.total_consumed()
    }

    /// Returns the priority of a stream.
    #[must_use]
    pub fn stream_priority(&self, id: StreamId) -> Option<ThrottlePriority> {
        self.streams.get(&id).map(|s| s.priority)
    }

    /// Computes the weighted fair share in bytes for a stream.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn fair_share(&self, id: StreamId) -> u64 {
        let total_weight: f64 = self.streams.values().map(|s| s.priority.weight()).sum();
        if total_weight < f64::EPSILON {
            return 0;
        }
        if let Some(entry) = self.streams.get(&id) {
            let fraction = entry.priority.weight() / total_weight;
            (self.aggregate.rate_bytes_per_sec() as f64 * fraction) as u64
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_id_round_trip() {
        let id = StreamId::new(7);
        assert_eq!(id.raw(), 7);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(ThrottlePriority::Low < ThrottlePriority::Normal);
        assert!(ThrottlePriority::Normal < ThrottlePriority::High);
        assert!(ThrottlePriority::High < ThrottlePriority::Critical);
    }

    #[test]
    fn test_priority_weight() {
        assert!((ThrottlePriority::Low.weight() - 1.0).abs() < f64::EPSILON);
        assert!((ThrottlePriority::Normal.weight() - 4.0).abs() < f64::EPSILON);
        assert!((ThrottlePriority::High.weight() - 8.0).abs() < f64::EPSILON);
        assert!((ThrottlePriority::Critical.weight() - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throttle_config_from_mbps() {
        let cfg = ThrottleConfig::from_mbps(10.0);
        assert_eq!(cfg.rate_bytes_per_sec, 1_250_000);
    }

    #[test]
    fn test_throttle_config_builder() {
        let cfg = ThrottleConfig::new(1000)
            .with_burst(5000)
            .with_refill_interval(Duration::from_millis(100));
        assert_eq!(cfg.rate_bytes_per_sec, 1000);
        assert_eq!(cfg.burst_bytes, 5000);
        assert_eq!(cfg.refill_interval, Duration::from_millis(100));
    }

    #[test]
    fn test_throttle_initial_tokens() {
        let t = BandwidthThrottle::with_rate(10_000);
        assert_eq!(t.available(), 10_000);
        assert_eq!(t.total_consumed(), 0);
    }

    #[test]
    fn test_consume_within_budget() {
        let mut t = BandwidthThrottle::with_rate(10_000);
        let consumed = t.consume(5_000);
        assert_eq!(consumed, 5_000);
        assert_eq!(t.available(), 5_000);
        assert_eq!(t.total_consumed(), 5_000);
    }

    #[test]
    fn test_consume_exceeds_budget() {
        let mut t = BandwidthThrottle::with_rate(1_000);
        let consumed = t.consume(5_000);
        assert_eq!(consumed, 1_000);
        assert_eq!(t.available(), 0);
    }

    #[test]
    fn test_can_send() {
        let mut t = BandwidthThrottle::with_rate(100);
        assert!(t.can_send(100));
        assert!(!t.can_send(101));
    }

    #[test]
    fn test_wait_time_zero_when_enough() {
        let t = BandwidthThrottle::with_rate(1000);
        assert_eq!(t.wait_time(500), Duration::ZERO);
    }

    #[test]
    fn test_wait_time_positive_when_deficit() {
        let mut t = BandwidthThrottle::with_rate(1000);
        t.consume(1000); // drain all tokens
        let wait = t.wait_time(500);
        assert!(wait > Duration::ZERO);
    }

    #[test]
    fn test_set_rate() {
        let mut t = BandwidthThrottle::with_rate(1000);
        t.set_rate(5000);
        assert_eq!(t.rate_bytes_per_sec(), 5000);
    }

    #[test]
    fn test_throttle_group_add_remove() {
        let mut group = ThrottleGroup::new(100_000);
        let id = group.add_stream(10_000, ThrottlePriority::Normal);
        assert_eq!(group.stream_count(), 1);
        assert!(group.remove_stream(id));
        assert_eq!(group.stream_count(), 0);
    }

    #[test]
    fn test_throttle_group_consume() {
        let mut group = ThrottleGroup::new(100_000);
        let id = group.add_stream(5_000, ThrottlePriority::Normal);
        let consumed = group.consume(id, 3_000);
        assert_eq!(consumed, 3_000);
    }

    #[test]
    fn test_throttle_group_consume_unknown_stream() {
        let mut group = ThrottleGroup::new(100_000);
        let fake_id = StreamId::new(999);
        let consumed = group.consume(fake_id, 1_000);
        assert_eq!(consumed, 0);
    }

    #[test]
    fn test_fair_share() {
        let mut group = ThrottleGroup::new(10_000);
        let low = group.add_stream(10_000, ThrottlePriority::Low);
        let high = group.add_stream(10_000, ThrottlePriority::High);

        let low_share = group.fair_share(low);
        let high_share = group.fair_share(high);

        // High weight = 8, Low weight = 1, total = 9
        // High share = 10000 * 8/9 ≈ 8888, Low share = 10000 * 1/9 ≈ 1111
        assert!(high_share > low_share);
        assert!(high_share > 5_000);
    }

    #[test]
    fn test_stream_priority_lookup() {
        let mut group = ThrottleGroup::new(100_000);
        let id = group.add_stream(10_000, ThrottlePriority::High);
        assert_eq!(group.stream_priority(id), Some(ThrottlePriority::High));
        assert_eq!(group.stream_priority(StreamId::new(999)), None);
    }
}

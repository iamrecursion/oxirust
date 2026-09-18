#![allow(dead_code)]
//! Cloud transfer bandwidth throttling and scheduling.
//!
//! Provides rate-limiting, time-of-day scheduling, and priority-based
//! bandwidth allocation for cloud storage transfers to control costs and
//! respect network capacity constraints.

use std::collections::BTreeMap;

/// Bandwidth unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthUnit {
    /// Bits per second.
    Bps,
    /// Kilobits per second.
    Kbps,
    /// Megabits per second.
    Mbps,
    /// Gigabits per second.
    Gbps,
}

impl BandwidthUnit {
    /// Converts a value in this unit to bits per second.
    #[must_use]
    pub fn to_bps(self, value: f64) -> f64 {
        match self {
            Self::Bps => value,
            Self::Kbps => value * 1_000.0,
            Self::Mbps => value * 1_000_000.0,
            Self::Gbps => value * 1_000_000_000.0,
        }
    }

    /// Converts a value in bps to this unit.
    #[must_use]
    pub fn from_bps(self, bps: f64) -> f64 {
        match self {
            Self::Bps => bps,
            Self::Kbps => bps / 1_000.0,
            Self::Mbps => bps / 1_000_000.0,
            Self::Gbps => bps / 1_000_000_000.0,
        }
    }
}

impl std::fmt::Display for BandwidthUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bps => write!(f, "bps"),
            Self::Kbps => write!(f, "Kbps"),
            Self::Mbps => write!(f, "Mbps"),
            Self::Gbps => write!(f, "Gbps"),
        }
    }
}

/// A bandwidth limit expressed as value + unit.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthLimit {
    /// The numeric value.
    pub value: f64,
    /// The unit.
    pub unit: BandwidthUnit,
}

impl BandwidthLimit {
    /// Creates a new bandwidth limit.
    #[must_use]
    pub fn new(value: f64, unit: BandwidthUnit) -> Self {
        Self { value, unit }
    }

    /// Creates a limit in Mbps.
    #[must_use]
    pub fn mbps(value: f64) -> Self {
        Self::new(value, BandwidthUnit::Mbps)
    }

    /// Creates a limit in Gbps.
    #[must_use]
    pub fn gbps(value: f64) -> Self {
        Self::new(value, BandwidthUnit::Gbps)
    }

    /// Returns the limit in bits per second.
    #[must_use]
    pub fn as_bps(&self) -> f64 {
        self.unit.to_bps(self.value)
    }

    /// Returns the limit in bytes per second.
    #[must_use]
    pub fn as_bytes_per_sec(&self) -> f64 {
        self.as_bps() / 8.0
    }
}

impl std::fmt::Display for BandwidthLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1} {}", self.value, self.unit)
    }
}

/// Transfer priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferPriority {
    /// Background transfer — lowest priority.
    Background,
    /// Normal priority.
    Normal,
    /// High priority.
    High,
    /// Critical — maximum priority.
    Critical,
}

impl std::fmt::Display for TransferPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Background => write!(f, "background"),
            Self::Normal => write!(f, "normal"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// A time-of-day schedule window (24-hour clock).
#[derive(Debug, Clone, Copy)]
pub struct ScheduleWindow {
    /// Start hour (0-23).
    pub start_hour: u8,
    /// End hour (0-23, exclusive). If end < start, wraps past midnight.
    pub end_hour: u8,
    /// Bandwidth limit during this window.
    pub limit: BandwidthLimit,
}

impl ScheduleWindow {
    /// Creates a new schedule window.
    #[must_use]
    pub fn new(start_hour: u8, end_hour: u8, limit: BandwidthLimit) -> Self {
        Self {
            start_hour: start_hour.min(23),
            end_hour: end_hour.min(23),
            limit,
        }
    }

    /// Returns whether the given hour falls within this window.
    #[must_use]
    pub fn contains_hour(&self, hour: u8) -> bool {
        if self.start_hour <= self.end_hour {
            hour >= self.start_hour && hour < self.end_hour
        } else {
            // Wraps past midnight
            hour >= self.start_hour || hour < self.end_hour
        }
    }
}

/// Priority-based bandwidth allocation entry.
#[derive(Debug, Clone)]
pub struct PriorityAllocation {
    /// Priority level.
    pub priority: TransferPriority,
    /// Percentage of total bandwidth allocated (0.0 to 1.0).
    pub share: f64,
}

impl PriorityAllocation {
    /// Creates a new priority allocation.
    #[must_use]
    pub fn new(priority: TransferPriority, share: f64) -> Self {
        Self {
            priority,
            share: share.clamp(0.0, 1.0),
        }
    }
}

/// Token bucket state for rate limiting.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum number of tokens (bytes).
    capacity: f64,
    /// Current token count.
    tokens: f64,
    /// Refill rate in bytes per second.
    refill_rate: f64,
    /// Last refill timestamp in seconds (monotonic).
    last_refill_secs: f64,
}

impl TokenBucket {
    /// Creates a new token bucket.
    #[must_use]
    pub fn new(capacity_bytes: f64, rate_bytes_per_sec: f64) -> Self {
        Self {
            capacity: capacity_bytes,
            tokens: capacity_bytes, // Start full
            refill_rate: rate_bytes_per_sec,
            last_refill_secs: 0.0,
        }
    }

    /// Refills the bucket based on elapsed time.
    pub fn refill(&mut self, current_secs: f64) {
        let elapsed = current_secs - self.last_refill_secs;
        if elapsed > 0.0 {
            self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
            self.last_refill_secs = current_secs;
        }
    }

    /// Tries to consume the given number of bytes. Returns true if allowed.
    pub fn try_consume(&mut self, bytes: f64, current_secs: f64) -> bool {
        self.refill(current_secs);
        if self.tokens >= bytes {
            self.tokens -= bytes;
            true
        } else {
            false
        }
    }

    /// Returns the current token count.
    #[must_use]
    pub fn available(&self) -> f64 {
        self.tokens
    }

    /// Returns how many seconds until the given number of bytes is available.
    #[must_use]
    pub fn wait_time_for(&self, bytes: f64) -> f64 {
        if self.tokens >= bytes {
            0.0
        } else {
            let deficit = bytes - self.tokens;
            if self.refill_rate > 0.0 {
                deficit / self.refill_rate
            } else {
                f64::INFINITY
            }
        }
    }

    /// Returns the refill rate in bytes per second.
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.refill_rate
    }

    /// Updates the refill rate.
    pub fn set_rate(&mut self, rate_bytes_per_sec: f64) {
        self.refill_rate = rate_bytes_per_sec;
    }
}

/// Bandwidth throttle configuration.
#[derive(Debug, Clone)]
pub struct ThrottleConfig {
    /// Global bandwidth limit.
    pub global_limit: BandwidthLimit,
    /// Time-of-day schedule windows (override the global limit).
    pub schedule_windows: Vec<ScheduleWindow>,
    /// Priority allocations.
    pub priority_allocations: BTreeMap<u8, PriorityAllocation>,
    /// Burst allowance as a multiplier of the base rate (e.g. 2.0 = 2x burst).
    pub burst_multiplier: f64,
}

impl ThrottleConfig {
    /// Creates a new throttle configuration with the given global limit.
    #[must_use]
    pub fn new(global_limit: BandwidthLimit) -> Self {
        Self {
            global_limit,
            schedule_windows: Vec::new(),
            priority_allocations: BTreeMap::new(),
            burst_multiplier: 1.5,
        }
    }

    /// Adds a schedule window.
    #[must_use]
    pub fn with_schedule(mut self, window: ScheduleWindow) -> Self {
        self.schedule_windows.push(window);
        self
    }

    /// Sets the burst multiplier.
    #[must_use]
    pub fn with_burst_multiplier(mut self, multiplier: f64) -> Self {
        self.burst_multiplier = multiplier.max(1.0);
        self
    }

    /// Returns the effective bandwidth limit for the given hour.
    #[must_use]
    pub fn effective_limit_at_hour(&self, hour: u8) -> BandwidthLimit {
        for window in &self.schedule_windows {
            if window.contains_hour(hour) {
                return window.limit;
            }
        }
        self.global_limit
    }
}

/// Bandwidth throttle manager.
///
/// Manages token-bucket rate limiting with time-of-day scheduling and
/// priority-based allocation for cloud transfers.
#[derive(Debug)]
pub struct BandwidthThrottle {
    /// Configuration.
    config: ThrottleConfig,
    /// Token bucket for rate limiting.
    bucket: TokenBucket,
    /// Total bytes transferred.
    total_bytes: u64,
    /// Total transfers attempted.
    total_requests: u64,
    /// Total transfers denied (throttled).
    denied_requests: u64,
}

impl BandwidthThrottle {
    /// Creates a new bandwidth throttle.
    #[must_use]
    pub fn new(config: ThrottleConfig) -> Self {
        let rate = config.global_limit.as_bytes_per_sec();
        let burst_capacity = rate * config.burst_multiplier;
        Self {
            config,
            bucket: TokenBucket::new(burst_capacity, rate),
            total_bytes: 0,
            total_requests: 0,
            denied_requests: 0,
        }
    }

    /// Attempts to transfer the given number of bytes at the current time.
    pub fn try_transfer(&mut self, bytes: u64, current_secs: f64) -> bool {
        self.total_requests += 1;
        #[allow(clippy::cast_precision_loss)]
        let bytes_f = bytes as f64;
        if self.bucket.try_consume(bytes_f, current_secs) {
            self.total_bytes += bytes;
            true
        } else {
            self.denied_requests += 1;
            false
        }
    }

    /// Updates the throttle for the given hour of day.
    pub fn update_for_hour(&mut self, hour: u8) {
        let limit = self.config.effective_limit_at_hour(hour);
        let rate = limit.as_bytes_per_sec();
        self.bucket.set_rate(rate);
    }

    /// Returns the total bytes transferred.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Returns the total number of transfer requests.
    #[must_use]
    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    /// Returns the number of denied (throttled) requests.
    #[must_use]
    pub fn denied_requests(&self) -> u64 {
        self.denied_requests
    }

    /// Returns the current rate in bytes per second.
    #[must_use]
    pub fn current_rate_bps(&self) -> f64 {
        self.bucket.rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_unit_conversion_to_bps() {
        assert!((BandwidthUnit::Kbps.to_bps(1.0) - 1_000.0).abs() < f64::EPSILON);
        assert!((BandwidthUnit::Mbps.to_bps(1.0) - 1_000_000.0).abs() < f64::EPSILON);
        assert!((BandwidthUnit::Gbps.to_bps(1.0) - 1_000_000_000.0).abs() < f64::EPSILON);
        assert!((BandwidthUnit::Bps.to_bps(42.0) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_unit_conversion_from_bps() {
        assert!((BandwidthUnit::Mbps.from_bps(1_000_000.0) - 1.0).abs() < f64::EPSILON);
        assert!((BandwidthUnit::Kbps.from_bps(5_000.0) - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_limit_mbps() {
        let limit = BandwidthLimit::mbps(100.0);
        assert!((limit.as_bps() - 100_000_000.0).abs() < f64::EPSILON);
        assert!((limit.as_bytes_per_sec() - 12_500_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_limit_gbps() {
        let limit = BandwidthLimit::gbps(1.0);
        assert!((limit.as_bps() - 1_000_000_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_limit_display() {
        let limit = BandwidthLimit::mbps(500.0);
        assert_eq!(limit.to_string(), "500.0 Mbps");
    }

    #[test]
    fn test_schedule_window_contains() {
        let window = ScheduleWindow::new(9, 17, BandwidthLimit::mbps(50.0));
        assert!(window.contains_hour(9));
        assert!(window.contains_hour(12));
        assert!(window.contains_hour(16));
        assert!(!window.contains_hour(17));
        assert!(!window.contains_hour(8));
    }

    #[test]
    fn test_schedule_window_wraps_midnight() {
        let window = ScheduleWindow::new(22, 6, BandwidthLimit::mbps(200.0));
        assert!(window.contains_hour(22));
        assert!(window.contains_hour(23));
        assert!(window.contains_hour(0));
        assert!(window.contains_hour(5));
        assert!(!window.contains_hour(6));
        assert!(!window.contains_hour(12));
    }

    #[test]
    fn test_token_bucket_consume() {
        let mut bucket = TokenBucket::new(1000.0, 100.0);
        assert!(bucket.try_consume(500.0, 0.0));
        assert!((bucket.available() - 500.0).abs() < f64::EPSILON);
        assert!(!bucket.try_consume(600.0, 0.0));
    }

    #[test]
    fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(1000.0, 100.0);
        assert!(bucket.try_consume(1000.0, 0.0));
        assert!((bucket.available()).abs() < f64::EPSILON);
        // After 5 seconds at 100 bytes/sec => 500 bytes refilled
        bucket.refill(5.0);
        assert!((bucket.available() - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_bucket_wait_time() {
        let mut bucket = TokenBucket::new(1000.0, 100.0);
        assert!(bucket.try_consume(800.0, 0.0));
        // Need 500 more, have 200, rate 100 => 3 seconds wait
        assert!((bucket.wait_time_for(500.0) - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transfer_priority_ordering() {
        assert!(TransferPriority::Background < TransferPriority::Normal);
        assert!(TransferPriority::Normal < TransferPriority::High);
        assert!(TransferPriority::High < TransferPriority::Critical);
    }

    #[test]
    fn test_throttle_config_effective_limit() {
        let config = ThrottleConfig::new(BandwidthLimit::mbps(100.0))
            .with_schedule(ScheduleWindow::new(9, 17, BandwidthLimit::mbps(50.0)));
        let limit = config.effective_limit_at_hour(12);
        assert!((limit.value - 50.0).abs() < f64::EPSILON);
        let limit_night = config.effective_limit_at_hour(20);
        assert!((limit_night.value - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throttle_try_transfer() {
        let config = ThrottleConfig::new(BandwidthLimit::mbps(800.0)).with_burst_multiplier(1.0);
        let mut throttle = BandwidthThrottle::new(config);
        // Rate = 800 Mbps = 100 MB/s = 100,000,000 bytes/s, capacity = same
        assert!(throttle.try_transfer(1_000_000, 0.0));
        assert_eq!(throttle.total_bytes(), 1_000_000);
        assert_eq!(throttle.total_requests(), 1);
        assert_eq!(throttle.denied_requests(), 0);
    }

    #[test]
    fn test_throttle_denies_over_capacity() {
        let config = ThrottleConfig::new(BandwidthLimit::new(6400.0, BandwidthUnit::Bps))
            .with_burst_multiplier(1.0);
        let mut throttle = BandwidthThrottle::new(config);
        // 6400 bps / 8 = 800 bytes/sec capacity
        assert!(throttle.try_transfer(800, 0.0));
        assert!(!throttle.try_transfer(100, 0.0)); // bucket empty
        assert_eq!(throttle.denied_requests(), 1);
    }

    #[test]
    fn test_throttle_update_for_hour() {
        let config = ThrottleConfig::new(BandwidthLimit::mbps(100.0))
            .with_schedule(ScheduleWindow::new(9, 17, BandwidthLimit::mbps(50.0)));
        let mut throttle = BandwidthThrottle::new(config);
        throttle.update_for_hour(12);
        // 50 Mbps = 6,250,000 bytes/s
        assert!((throttle.current_rate_bps() - 6_250_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_priority_allocation_clamp() {
        let alloc = PriorityAllocation::new(TransferPriority::High, 1.5);
        assert!((alloc.share - 1.0).abs() < f64::EPSILON);
        let alloc2 = PriorityAllocation::new(TransferPriority::Background, -0.5);
        assert!(alloc2.share.abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_unit_display() {
        assert_eq!(BandwidthUnit::Bps.to_string(), "bps");
        assert_eq!(BandwidthUnit::Kbps.to_string(), "Kbps");
        assert_eq!(BandwidthUnit::Mbps.to_string(), "Mbps");
        assert_eq!(BandwidthUnit::Gbps.to_string(), "Gbps");
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(TransferPriority::Background.to_string(), "background");
        assert_eq!(TransferPriority::Normal.to_string(), "normal");
        assert_eq!(TransferPriority::High.to_string(), "high");
        assert_eq!(TransferPriority::Critical.to_string(), "critical");
    }
}

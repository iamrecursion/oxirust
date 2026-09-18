//! Reference clock sources, configuration, and pool management.
#![allow(dead_code)]

use std::time::{Duration, Instant};

/// The source of a reference clock signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockSource {
    /// GPS disciplined oscillator.
    Gps,
    /// PTP (IEEE 1588) grandmaster.
    Ptp,
    /// NTP server.
    Ntp,
    /// Local oscillator (free-running).
    Local,
    /// PPS (pulse-per-second) signal from an external source.
    Pps,
    /// GNSS receiver (general satellite navigation).
    Gnss,
}

impl ClockSource {
    /// NTP stratum level associated with this clock source.
    ///
    /// Stratum 1 sources are directly connected to an authoritative reference;
    /// higher stratum numbers indicate increasing distance.
    pub fn stratum(self) -> u8 {
        match self {
            ClockSource::Gps | ClockSource::Gnss | ClockSource::Pps => 1,
            ClockSource::Ptp => 1,
            ClockSource::Ntp => 2,
            ClockSource::Local => 16, // unsynchronised per RFC 5905
        }
    }

    /// Human-readable name for the clock source.
    pub fn name(self) -> &'static str {
        match self {
            ClockSource::Gps => "GPS",
            ClockSource::Ptp => "PTP",
            ClockSource::Ntp => "NTP",
            ClockSource::Local => "Local",
            ClockSource::Pps => "PPS",
            ClockSource::Gnss => "GNSS",
        }
    }

    /// Returns `true` if this source is considered a primary reference (stratum 1).
    pub fn is_primary(self) -> bool {
        self.stratum() == 1
    }
}

/// Configuration for a single reference clock.
#[derive(Debug, Clone)]
pub struct RefClockConfig {
    /// The type of clock source.
    pub source: ClockSource,
    /// Maximum accepted offset error in nanoseconds before the clock is considered unlocked.
    pub max_offset_ns: i64,
    /// Minimum number of consecutive good samples required to declare the clock locked.
    pub min_good_samples: u32,
    /// Priority used for clock selection (lower value = higher priority).
    pub priority: u8,
}

impl RefClockConfig {
    /// Create a new `RefClockConfig`.
    pub fn new(
        source: ClockSource,
        max_offset_ns: i64,
        min_good_samples: u32,
        priority: u8,
    ) -> Self {
        Self {
            source,
            max_offset_ns,
            min_good_samples,
            priority,
        }
    }

    /// Default GPS configuration.
    pub fn gps() -> Self {
        Self::new(ClockSource::Gps, 500, 5, 10)
    }

    /// Default PTP configuration.
    pub fn ptp() -> Self {
        Self::new(ClockSource::Ptp, 1_000, 3, 20)
    }

    /// Default NTP configuration.
    pub fn ntp() -> Self {
        Self::new(ClockSource::Ntp, 10_000_000, 5, 50)
    }

    /// Returns `true` if this clock is a primary reference (stratum 1) according to its source.
    pub fn is_primary(&self) -> bool {
        self.source.is_primary()
    }
}

/// Runtime state of a single reference clock.
#[derive(Debug)]
pub struct RefClock {
    config: RefClockConfig,
    /// Current measured offset from UTC in nanoseconds.
    current_offset_ns: i64,
    /// Number of consecutive samples within `max_offset_ns`.
    good_sample_count: u32,
    /// Whether the clock is currently considered locked.
    locked: bool,
    /// Monotonic time of the last update.
    last_update: Option<Instant>,
}

impl RefClock {
    /// Create a new `RefClock` from its configuration.
    pub fn new(config: RefClockConfig) -> Self {
        Self {
            config,
            current_offset_ns: 0,
            good_sample_count: 0,
            locked: false,
            last_update: None,
        }
    }

    /// Update the clock with a fresh offset measurement.
    pub fn update(&mut self, offset_ns: i64) {
        self.current_offset_ns = offset_ns;
        self.last_update = Some(Instant::now());
        if offset_ns.abs() <= self.config.max_offset_ns {
            self.good_sample_count = self.good_sample_count.saturating_add(1);
        } else {
            self.good_sample_count = 0;
            self.locked = false;
        }
        if self.good_sample_count >= self.config.min_good_samples {
            self.locked = true;
        }
    }

    /// Current measured offset from UTC in nanoseconds.
    pub fn current_offset_ns(&self) -> i64 {
        self.current_offset_ns
    }

    /// Returns `true` if the clock has accumulated enough good samples and is locked.
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Time since the last update was applied, or `None` if never updated.
    pub fn time_since_update(&self) -> Option<Duration> {
        self.last_update.map(|t| t.elapsed())
    }

    /// Clock source for this reference.
    pub fn source(&self) -> ClockSource {
        self.config.source
    }

    /// Configuration for this reference clock.
    pub fn config(&self) -> &RefClockConfig {
        &self.config
    }
}

/// A pool of reference clocks; selects the best available source.
#[derive(Debug, Default)]
pub struct RefClockPool {
    clocks: Vec<RefClock>,
}

impl RefClockPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a reference clock to the pool.
    pub fn add(&mut self, clock: RefClock) {
        self.clocks.push(clock);
    }

    /// Number of clocks in the pool.
    pub fn len(&self) -> usize {
        self.clocks.len()
    }

    /// Returns `true` if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.clocks.is_empty()
    }

    /// Update all clocks with the given offset (for testing / simulation).
    pub fn update_all(&mut self, offset_ns: i64) {
        for clock in &mut self.clocks {
            clock.update(offset_ns);
        }
    }

    /// Select the best available source.
    ///
    /// Selection criteria (in order):
    /// 1. Clock must be locked.
    /// 2. Lower `priority` value wins.
    /// 3. Lower stratum wins.
    ///
    /// Returns `None` if no clocks are locked.
    pub fn best_source(&self) -> Option<&RefClock> {
        self.clocks
            .iter()
            .filter(|c| c.is_locked())
            .min_by_key(|c| (c.config.priority, c.config.source.stratum()))
    }

    /// Number of currently locked clocks.
    pub fn locked_count(&self) -> usize {
        self.clocks.iter().filter(|c| c.is_locked()).count()
    }

    /// Mutable access to a clock by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut RefClock> {
        self.clocks.get_mut(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_source_stratum_gps() {
        assert_eq!(ClockSource::Gps.stratum(), 1);
    }

    #[test]
    fn test_clock_source_stratum_ntp() {
        assert_eq!(ClockSource::Ntp.stratum(), 2);
    }

    #[test]
    fn test_clock_source_stratum_local() {
        assert_eq!(ClockSource::Local.stratum(), 16);
    }

    #[test]
    fn test_clock_source_is_primary() {
        assert!(ClockSource::Gps.is_primary());
        assert!(ClockSource::Ptp.is_primary());
        assert!(!ClockSource::Ntp.is_primary());
        assert!(!ClockSource::Local.is_primary());
    }

    #[test]
    fn test_clock_source_name() {
        assert_eq!(ClockSource::Gps.name(), "GPS");
        assert_eq!(ClockSource::Local.name(), "Local");
    }

    #[test]
    fn test_ref_clock_config_is_primary() {
        assert!(RefClockConfig::gps().is_primary());
        assert!(RefClockConfig::ptp().is_primary());
        assert!(!RefClockConfig::ntp().is_primary());
    }

    #[test]
    fn test_ref_clock_not_locked_initially() {
        let cfg = RefClockConfig::gps();
        let clock = RefClock::new(cfg);
        assert!(!clock.is_locked());
    }

    #[test]
    fn test_ref_clock_locks_after_enough_good_samples() {
        let cfg = RefClockConfig::new(ClockSource::Gps, 500, 3, 10);
        let mut clock = RefClock::new(cfg);
        for _ in 0..3 {
            clock.update(100); // within 500 ns
        }
        assert!(clock.is_locked());
    }

    #[test]
    fn test_ref_clock_unlocks_on_large_offset() {
        let cfg = RefClockConfig::new(ClockSource::Gps, 500, 2, 10);
        let mut clock = RefClock::new(cfg);
        clock.update(100);
        clock.update(100); // locked
        assert!(clock.is_locked());
        clock.update(100_000); // large error — should unlock
        assert!(!clock.is_locked());
    }

    #[test]
    fn test_ref_clock_offset_stored() {
        let cfg = RefClockConfig::gps();
        let mut clock = RefClock::new(cfg);
        clock.update(12345);
        assert_eq!(clock.current_offset_ns(), 12345);
    }

    #[test]
    fn test_ref_clock_time_since_update_none() {
        let cfg = RefClockConfig::gps();
        let clock = RefClock::new(cfg);
        assert!(clock.time_since_update().is_none());
    }

    #[test]
    fn test_ref_clock_time_since_update_some() {
        let cfg = RefClockConfig::gps();
        let mut clock = RefClock::new(cfg);
        clock.update(0);
        assert!(clock.time_since_update().is_some());
    }

    #[test]
    fn test_pool_best_source_empty() {
        let pool = RefClockPool::new();
        assert!(pool.best_source().is_none());
    }

    #[test]
    fn test_pool_best_source_prefers_lower_priority() {
        let mut pool = RefClockPool::new();
        let cfg_gps = RefClockConfig::new(ClockSource::Gps, 500, 1, 10);
        let cfg_ntp = RefClockConfig::new(ClockSource::Ntp, 10_000_000, 1, 50);
        let mut gps = RefClock::new(cfg_gps);
        let mut ntp = RefClock::new(cfg_ntp);
        gps.update(100);
        ntp.update(100_000);
        pool.add(gps);
        pool.add(ntp);
        let best = pool.best_source().expect("should succeed in test");
        assert_eq!(best.source(), ClockSource::Gps);
    }

    #[test]
    fn test_pool_locked_count() {
        let mut pool = RefClockPool::new();
        let cfg1 = RefClockConfig::new(ClockSource::Ptp, 1000, 1, 20);
        let cfg2 = RefClockConfig::new(ClockSource::Ntp, 10_000_000, 1, 50);
        let mut c1 = RefClock::new(cfg1);
        let c2 = RefClock::new(cfg2); // never updated → not locked
        c1.update(200);
        pool.add(c1);
        pool.add(c2);
        assert_eq!(pool.locked_count(), 1);
    }

    #[test]
    fn test_pool_is_empty() {
        let pool = RefClockPool::new();
        assert!(pool.is_empty());
    }
}

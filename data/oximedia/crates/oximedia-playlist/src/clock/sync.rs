//! Clock synchronization.

use chrono::{DateTime, Utc};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Source for clock synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockSource {
    /// System wall clock.
    WallClock,

    /// Network Time Protocol (NTP).
    Ntp,

    /// Precision Time Protocol (PTP).
    Ptp,

    /// External LTC timecode.
    Ltc,

    /// Manual/Free-running.
    Manual,
}

/// Clock synchronization manager.
pub struct ClockSync {
    source: Arc<RwLock<ClockSource>>,
    offset: Arc<RwLock<Duration>>,
    drift_compensation: Arc<RwLock<f64>>,
    last_sync: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl ClockSync {
    /// Creates a new clock sync instance.
    #[must_use]
    pub fn new(source: ClockSource) -> Self {
        Self {
            source: Arc::new(RwLock::new(source)),
            offset: Arc::new(RwLock::new(Duration::ZERO)),
            drift_compensation: Arc::new(RwLock::new(1.0)),
            last_sync: Arc::new(RwLock::new(None)),
        }
    }

    /// Sets the clock source.
    pub fn set_source(&self, source: ClockSource) {
        if let Ok(mut s) = self.source.write() {
            *s = source;
        }
    }

    /// Gets the current clock source.
    #[must_use]
    pub fn get_source(&self) -> ClockSource {
        self.source
            .read()
            .map(|s| *s)
            .unwrap_or(ClockSource::WallClock)
    }

    /// Gets the current time from the configured source.
    #[must_use]
    pub fn now(&self) -> DateTime<Utc> {
        let mut now = Utc::now();

        // Apply offset
        if let Ok(offset) = self.offset.read() {
            if let Ok(offset_duration) = chrono::Duration::from_std(*offset) {
                now += offset_duration;
            }
        }

        now
    }

    /// Sets a time offset.
    pub fn set_offset(&self, offset: Duration) {
        if let Ok(mut o) = self.offset.write() {
            *o = offset;
        }
    }

    /// Gets the current offset.
    #[must_use]
    pub fn get_offset(&self) -> Duration {
        self.offset.read().map(|o| *o).unwrap_or(Duration::ZERO)
    }

    /// Performs a synchronization with the clock source.
    pub fn synchronize(&self) -> Result<(), String> {
        let source = self.get_source();

        match source {
            ClockSource::WallClock => {
                // No sync needed for wall clock
                self.update_last_sync();
                Ok(())
            }
            ClockSource::Ntp => {
                // In a real implementation, this would query an NTP server
                self.update_last_sync();
                Ok(())
            }
            ClockSource::Ptp => {
                // In a real implementation, this would sync with PTP
                self.update_last_sync();
                Ok(())
            }
            ClockSource::Ltc => {
                // In a real implementation, this would read LTC timecode
                self.update_last_sync();
                Ok(())
            }
            ClockSource::Manual => {
                // Manual mode doesn't auto-sync
                Ok(())
            }
        }
    }

    /// Sets drift compensation factor.
    pub fn set_drift_compensation(&self, factor: f64) {
        if let Ok(mut drift) = self.drift_compensation.write() {
            *drift = factor;
        }
    }

    /// Gets the drift compensation factor.
    #[must_use]
    pub fn get_drift_compensation(&self) -> f64 {
        self.drift_compensation.read().map(|d| *d).unwrap_or(1.0)
    }

    /// Gets the time of last synchronization.
    #[must_use]
    pub fn last_sync_time(&self) -> Option<DateTime<Utc>> {
        *self.last_sync.read().ok()?
    }

    fn update_last_sync(&self) {
        if let Ok(mut last) = self.last_sync.write() {
            *last = Some(Utc::now());
        }
    }
}

impl Default for ClockSync {
    fn default() -> Self {
        Self::new(ClockSource::WallClock)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_sync() {
        let clock = ClockSync::new(ClockSource::WallClock);
        assert_eq!(clock.get_source(), ClockSource::WallClock);

        let now = clock.now();
        assert!(now <= Utc::now());
    }

    #[test]
    fn test_clock_offset() {
        let clock = ClockSync::new(ClockSource::WallClock);
        let offset = Duration::from_secs(60);

        clock.set_offset(offset);
        assert_eq!(clock.get_offset(), offset);
    }

    #[test]
    fn test_synchronize() {
        let clock = ClockSync::new(ClockSource::Ntp);
        assert!(clock.synchronize().is_ok());
        assert!(clock.last_sync_time().is_some());
    }

    #[test]
    fn test_drift_compensation() {
        let clock = ClockSync::new(ClockSource::WallClock);
        clock.set_drift_compensation(1.001);
        assert!((clock.get_drift_compensation() - 1.001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clocksync_offset_roundtrip() {
        // A 100ms offset set on the sync manager round-trips exactly through
        // get_offset, and synchronize() populates last_sync_time.
        let clock = ClockSync::new(ClockSource::Ntp);
        assert!(
            clock.last_sync_time().is_none(),
            "no sync should have happened yet"
        );

        clock.set_offset(Duration::from_millis(100));
        assert_eq!(clock.get_offset(), Duration::from_millis(100));

        clock.synchronize().expect("Ntp synchronize succeeds");
        assert!(
            clock.last_sync_time().is_some(),
            "synchronize must update last_sync_time"
        );
        // The offset is unaffected by synchronization.
        assert_eq!(clock.get_offset(), Duration::from_millis(100));
    }
}

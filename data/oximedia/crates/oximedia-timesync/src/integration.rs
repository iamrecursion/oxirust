//! Integration with oximedia-core Timestamp type.

use crate::error::TimeSyncResult;
use oximedia_core::types::{Rational, Timestamp};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Convert system time to oximedia Timestamp.
pub fn system_time_to_timestamp(
    system_time: SystemTime,
    timebase: Rational,
) -> TimeSyncResult<Timestamp> {
    let duration = system_time
        .duration_since(UNIX_EPOCH)
        .map_err(|_| crate::error::TimeSyncError::InvalidTimestamp)?;

    // Convert duration to PTS in the given timebase
    let total_seconds = duration.as_secs_f64();
    let pts = (total_seconds / timebase.to_f64()) as i64;

    Ok(Timestamp::new(pts, timebase))
}

/// Convert oximedia Timestamp to system time.
pub fn timestamp_to_system_time(timestamp: &Timestamp) -> TimeSyncResult<SystemTime> {
    let seconds = timestamp.to_seconds();

    if seconds < 0.0 {
        return Err(crate::error::TimeSyncError::InvalidTimestamp);
    }

    let duration = Duration::from_secs_f64(seconds);
    Ok(UNIX_EPOCH + duration)
}

/// Adjust timestamp by nanoseconds offset.
pub fn adjust_timestamp(timestamp: &Timestamp, offset_ns: i64) -> TimeSyncResult<Timestamp> {
    // Convert offset to timestamp units
    let offset_seconds = offset_ns as f64 / 1e9;
    let offset_pts = (offset_seconds / timestamp.timebase.to_f64()) as i64;

    let new_pts = timestamp.pts + offset_pts;

    Ok(Timestamp::new(new_pts, timestamp.timebase))
}

/// Synchronize timestamp to a reference time.
pub fn sync_timestamp_to_reference(
    timestamp: &Timestamp,
    reference_time: SystemTime,
    _timebase: Rational,
) -> TimeSyncResult<Timestamp> {
    // Calculate offset
    let current_system_time = timestamp_to_system_time(timestamp)?;
    let offset = reference_time
        .duration_since(current_system_time)
        .map_or_else(
            |e| -(e.duration().as_nanos() as i64),
            |d| d.as_nanos() as i64,
        );

    adjust_timestamp(timestamp, offset)
}

/// Timestamp synchronizer for maintaining accurate timing.
pub struct TimestampSync {
    /// Reference timebase
    timebase: Rational,
    /// Current offset from system time (nanoseconds)
    offset_ns: i64,
}

impl TimestampSync {
    /// Create a new timestamp synchronizer.
    #[must_use]
    pub fn new(timebase: Rational) -> Self {
        Self {
            timebase,
            offset_ns: 0,
        }
    }

    /// Set the time offset.
    pub fn set_offset(&mut self, offset_ns: i64) {
        self.offset_ns = offset_ns;
    }

    /// Get current synchronized timestamp.
    pub fn current_timestamp(&self) -> TimeSyncResult<Timestamp> {
        let now = SystemTime::now();
        let mut timestamp = system_time_to_timestamp(now, self.timebase)?;

        if self.offset_ns != 0 {
            timestamp = adjust_timestamp(&timestamp, self.offset_ns)?;
        }

        Ok(timestamp)
    }

    /// Get timebase.
    #[must_use]
    pub fn timebase(&self) -> Rational {
        self.timebase
    }

    /// Get current offset.
    #[must_use]
    pub fn offset_ns(&self) -> i64 {
        self.offset_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_time_conversion() {
        let now = SystemTime::now();
        let timebase = Rational::new(1, 1000); // milliseconds

        let timestamp = system_time_to_timestamp(now, timebase).expect("should succeed in test");
        assert!(timestamp.pts > 0);

        let converted_back = timestamp_to_system_time(&timestamp).expect("should succeed in test");

        // Should be close (within 1ms due to precision)
        let diff = now
            .duration_since(converted_back)
            .unwrap_or_else(|e| e.duration());
        assert!(diff < Duration::from_millis(1));
    }

    #[test]
    fn test_adjust_timestamp() {
        let timebase = Rational::new(1, 1000);
        let timestamp = Timestamp::new(1000, timebase);

        // Add 1 second (1_000_000_000 ns)
        let adjusted = adjust_timestamp(&timestamp, 1_000_000_000).expect("should succeed in test");

        // Should add 1000 ms = 1000 PTS units
        assert_eq!(adjusted.pts, 2000);
    }

    #[test]
    fn test_timestamp_sync() {
        let timebase = Rational::new(1, 1000);
        let mut sync = TimestampSync::new(timebase);

        let ts1 = sync.current_timestamp().expect("should succeed in test");

        // Set offset
        sync.set_offset(1_000_000); // 1ms

        let ts2 = sync.current_timestamp().expect("should succeed in test");

        // Second timestamp should be ahead by ~1 PTS unit
        assert!(ts2.pts >= ts1.pts);
    }
}

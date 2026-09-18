//! NDI clock synchronization and timecode conversion.
//!
//! This module provides tools for synchronizing local clocks with NDI sources
//! and converting between NDI timestamps and SMPTE timecodes.

#![allow(dead_code)]

/// NDI clock for tracking time offset between local and remote NDI timestamps.
#[derive(Debug, Clone)]
pub struct NdiClock {
    /// Reference timestamp in milliseconds (NDI source time).
    pub reference_timestamp_ms: u64,
    /// Offset applied to local time to obtain NDI time (in milliseconds).
    pub local_offset_ms: i64,
    /// Clock drift rate in parts-per-million (positive = local runs fast).
    pub drift_ppm: f64,
}

impl NdiClock {
    /// Creates a new `NdiClock` with zeroed values (unsynchronized).
    pub fn new() -> Self {
        Self {
            reference_timestamp_ms: 0,
            local_offset_ms: 0,
            drift_ppm: 0.0,
        }
    }

    /// Updates the clock offset by comparing a source timestamp with local time.
    ///
    /// # Arguments
    /// * `source_ts` - Timestamp reported by the NDI source (ms).
    /// * `local_ts` - Local timestamp at the time of receiving the source timestamp (ms).
    pub fn sync_to_source(&mut self, source_ts: u64, local_ts: u64) {
        self.reference_timestamp_ms = source_ts;
        // Offset is the amount to add to local time to get NDI time
        self.local_offset_ms = source_ts as i64 - local_ts as i64;
    }

    /// Converts a local timestamp (ms) to NDI network time (ms).
    pub fn local_to_ndi(&self, local_ms: u64) -> u64 {
        (local_ms as i64 + self.local_offset_ms).max(0) as u64
    }

    /// Converts an NDI network timestamp (ms) to local time (ms).
    pub fn ndi_to_local(&self, ndi_ms: u64) -> u64 {
        (ndi_ms as i64 - self.local_offset_ms).max(0) as u64
    }
}

impl Default for NdiClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about clock synchronization quality.
#[derive(Debug, Clone, Default)]
pub struct ClockSyncStats {
    /// Number of synchronization events performed.
    pub sync_count: u64,
    /// Running average of the sync offset in milliseconds.
    pub avg_offset_ms: f64,
    /// Maximum observed sync offset in milliseconds (absolute value).
    pub max_offset_ms: i64,
    /// Whether the clock is considered synchronized.
    pub is_synced: bool,
}

impl ClockSyncStats {
    /// Creates new zeroed `ClockSyncStats`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a new synchronization event with the given offset.
    ///
    /// Updates the running average and maximum offset, and marks the clock as synced.
    pub fn record_sync(&mut self, offset_ms: i64) {
        self.sync_count += 1;
        // Running average using cumulative method
        self.avg_offset_ms = (self.avg_offset_ms * (self.sync_count - 1) as f64 + offset_ms as f64)
            / self.sync_count as f64;
        self.max_offset_ms = self.max_offset_ms.max(offset_ms.abs());
        self.is_synced = true;
    }
}

/// SMPTE timecode converter for NDI timestamps.
///
/// Provides conversion between millisecond timestamps and SMPTE HH:MM:SS:FF format.
pub struct NdiTimecodeConverter;

impl NdiTimecodeConverter {
    /// Converts a timestamp in milliseconds to SMPTE timecode string.
    ///
    /// # Arguments
    /// * `timestamp_ms` - Millisecond timestamp to convert.
    /// * `fps_num` - Frame rate numerator (e.g., 30 for 30fps).
    /// * `fps_den` - Frame rate denominator (e.g., 1 for 30fps, 1001 for 29.97).
    ///
    /// # Returns
    /// A SMPTE timecode string in the format `HH:MM:SS:FF`.
    ///
    /// # Example
    /// ```
    /// use oximedia_ndi::clock_sync::NdiTimecodeConverter;
    /// let tc = NdiTimecodeConverter::to_smpte(3723_040, 30, 1);
    /// assert_eq!(tc, "01:02:03:01");
    /// ```
    pub fn to_smpte(timestamp_ms: u64, fps_num: u32, fps_den: u32) -> String {
        if fps_num == 0 || fps_den == 0 {
            return "00:00:00:00".to_string();
        }

        // Total frames = timestamp_ms * fps_num / (1000 * fps_den)
        let total_frames = timestamp_ms * fps_num as u64 / (1000 * fps_den as u64);

        let fps_whole = fps_num / fps_den; // integer fps for frame-of-second calculation
        let fps_whole = fps_whole.max(1);

        let frames = (total_frames % fps_whole as u64) as u32;
        let total_seconds = total_frames / fps_whole as u64;
        let seconds = (total_seconds % 60) as u32;
        let total_minutes = total_seconds / 60;
        let minutes = (total_minutes % 60) as u32;
        let hours = (total_minutes / 60) as u32;

        format!("{:02}:{:02}:{:02}:{:02}", hours, minutes, seconds, frames)
    }

    /// Converts a SMPTE timecode string back to a millisecond timestamp.
    ///
    /// # Arguments
    /// * `timecode` - SMPTE timecode string in `HH:MM:SS:FF` format.
    /// * `fps_num` - Frame rate numerator.
    /// * `fps_den` - Frame rate denominator.
    ///
    /// # Returns
    /// Millisecond timestamp corresponding to the timecode, or `0` if parsing fails.
    ///
    /// # Example
    /// ```
    /// use oximedia_ndi::clock_sync::NdiTimecodeConverter;
    /// let ms = NdiTimecodeConverter::from_smpte("01:02:03:01", 30, 1);
    /// assert_eq!(ms, 3723_033);
    /// ```
    pub fn from_smpte(timecode: &str, fps_num: u32, fps_den: u32) -> u64 {
        if fps_num == 0 || fps_den == 0 {
            return 0;
        }

        let parts: Vec<&str> = timecode.split(':').collect();
        if parts.len() != 4 {
            return 0;
        }

        let hours: u64 = parts[0].parse().unwrap_or(0);
        let minutes: u64 = parts[1].parse().unwrap_or(0);
        let seconds: u64 = parts[2].parse().unwrap_or(0);
        let frames: u64 = parts[3].parse().unwrap_or(0);

        let fps_whole = (fps_num / fps_den).max(1) as u64;

        let total_frames =
            hours * 3600 * fps_whole + minutes * 60 * fps_whole + seconds * fps_whole + frames;

        // Convert frames to ms: total_frames * 1000 * fps_den / fps_num
        total_frames * 1000 * fps_den as u64 / fps_num as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndi_clock_default() {
        let clock = NdiClock::default();
        assert_eq!(clock.reference_timestamp_ms, 0);
        assert_eq!(clock.local_offset_ms, 0);
        assert_eq!(clock.drift_ppm, 0.0);
    }

    #[test]
    fn test_ndi_clock_sync_to_source() {
        let mut clock = NdiClock::new();
        clock.sync_to_source(10_000, 9_900);
        assert_eq!(clock.local_offset_ms, 100);
        assert_eq!(clock.reference_timestamp_ms, 10_000);
    }

    #[test]
    fn test_ndi_clock_local_to_ndi() {
        let mut clock = NdiClock::new();
        clock.sync_to_source(10_000, 9_900);
        // local 9_900 + 100 offset = 10_000
        assert_eq!(clock.local_to_ndi(9_900), 10_000);
    }

    #[test]
    fn test_ndi_clock_ndi_to_local() {
        let mut clock = NdiClock::new();
        clock.sync_to_source(10_000, 9_900);
        assert_eq!(clock.ndi_to_local(10_000), 9_900);
    }

    #[test]
    fn test_ndi_clock_negative_offset() {
        let mut clock = NdiClock::new();
        // Source is behind local time
        clock.sync_to_source(9_900, 10_000);
        assert_eq!(clock.local_offset_ms, -100);
        assert_eq!(clock.local_to_ndi(10_000), 9_900);
    }

    #[test]
    fn test_clock_sync_stats_initial() {
        let stats = ClockSyncStats::new();
        assert_eq!(stats.sync_count, 0);
        assert!(!stats.is_synced);
    }

    #[test]
    fn test_clock_sync_stats_record() {
        let mut stats = ClockSyncStats::new();
        stats.record_sync(50);
        assert_eq!(stats.sync_count, 1);
        assert_eq!(stats.avg_offset_ms, 50.0);
        assert!(stats.is_synced);
    }

    #[test]
    fn test_clock_sync_stats_average() {
        let mut stats = ClockSyncStats::new();
        stats.record_sync(10);
        stats.record_sync(30);
        assert!((stats.avg_offset_ms - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_clock_sync_stats_max_offset() {
        let mut stats = ClockSyncStats::new();
        stats.record_sync(-100);
        stats.record_sync(50);
        // max_offset_ms tracks absolute value maximum
        assert_eq!(stats.max_offset_ms, 100);
    }

    #[test]
    fn test_timecode_to_smpte_zero() {
        let tc = NdiTimecodeConverter::to_smpte(0, 30, 1);
        assert_eq!(tc, "00:00:00:00");
    }

    #[test]
    fn test_timecode_to_smpte_one_second() {
        // 1000 ms at 30fps = 30 frames = 1 second exactly
        let tc = NdiTimecodeConverter::to_smpte(1000, 30, 1);
        assert_eq!(tc, "00:00:01:00");
    }

    #[test]
    fn test_timecode_to_smpte_one_hour() {
        // 3600*1000 ms at 30fps
        let tc = NdiTimecodeConverter::to_smpte(3_600_000, 30, 1);
        assert_eq!(tc, "01:00:00:00");
    }

    #[test]
    fn test_timecode_from_smpte_zero() {
        let ms = NdiTimecodeConverter::from_smpte("00:00:00:00", 30, 1);
        assert_eq!(ms, 0);
    }

    #[test]
    fn test_timecode_from_smpte_one_second() {
        let ms = NdiTimecodeConverter::from_smpte("00:00:01:00", 30, 1);
        assert_eq!(ms, 1000);
    }

    #[test]
    fn test_timecode_from_smpte_invalid() {
        let ms = NdiTimecodeConverter::from_smpte("invalid", 30, 1);
        assert_eq!(ms, 0);
    }

    #[test]
    fn test_timecode_roundtrip() {
        // 1 hour, 2 minutes, 3 seconds at 30fps
        let ms = NdiTimecodeConverter::from_smpte("01:02:03:00", 30, 1);
        let tc = NdiTimecodeConverter::to_smpte(ms, 30, 1);
        assert_eq!(tc, "01:02:03:00");
    }
}

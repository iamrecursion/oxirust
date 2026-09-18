//! Stream synchronization utilities for multi-essence ST 2110 streams.
//!
//! Handles RTP timestamp alignment, inter-stream latency measurement, and
//! synchronization gap detection across video, audio, and ancillary data flows.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;

/// Maximum number of timestamp samples to keep in the history window.
const MAX_HISTORY: usize = 256;

/// RTP clock rate for video (90 kHz per RFC 3550).
pub const RTP_CLOCK_RATE_VIDEO: u32 = 90_000;

/// RTP clock rate for 48 kHz audio.
pub const RTP_CLOCK_RATE_AUDIO_48K: u32 = 48_000;

/// RTP clock rate for 96 kHz audio.
pub const RTP_CLOCK_RATE_AUDIO_96K: u32 = 96_000;

/// A single RTP timing sample (sequence number + timestamp).
#[derive(Debug, Clone, Copy)]
pub struct RtpSample {
    /// RTP sequence number.
    pub seq: u16,
    /// RTP timestamp.
    pub timestamp: u32,
    /// Wall-clock arrival time in microseconds (monotonic).
    pub arrival_us: u64,
}

impl RtpSample {
    /// Create a new sample.
    #[must_use]
    pub fn new(seq: u16, timestamp: u32, arrival_us: u64) -> Self {
        Self {
            seq,
            timestamp,
            arrival_us,
        }
    }
}

/// Synchronization gap between two streams (in microseconds).
#[derive(Debug, Clone, Copy)]
pub struct SyncGap {
    /// Gap magnitude in microseconds.
    pub gap_us: i64,
    /// Whether the gap exceeds acceptable bounds.
    pub is_excessive: bool,
}

/// Inter-stream synchronization monitor.
///
/// Compares RTP timestamps of two streams (e.g., video and audio) to detect
/// synchronization drift using the RTP-NTP mapping negotiated via RTCP SR.
#[derive(Debug)]
pub struct StreamSyncMonitor {
    /// Clock rate of stream A (ticks per second).
    pub clock_rate_a: u32,
    /// Clock rate of stream B (ticks per second).
    pub clock_rate_b: u32,
    /// NTP / RTP mapping for stream A: (`ntp_usec`, `rtp_ts`).
    pub ntp_map_a: Option<(u64, u32)>,
    /// NTP / RTP mapping for stream B: (`ntp_usec`, `rtp_ts`).
    pub ntp_map_b: Option<(u64, u32)>,
    /// Maximum acceptable sync gap in microseconds.
    pub max_gap_us: u64,
    /// Recent sync gap measurements.
    gap_history: VecDeque<SyncGap>,
}

impl StreamSyncMonitor {
    /// Create a new monitor.
    #[must_use]
    pub fn new(clock_rate_a: u32, clock_rate_b: u32, max_gap_us: u64) -> Self {
        Self {
            clock_rate_a,
            clock_rate_b,
            ntp_map_a: None,
            ntp_map_b: None,
            max_gap_us,
            gap_history: VecDeque::with_capacity(MAX_HISTORY),
        }
    }

    /// Record an RTCP SR mapping for stream A.
    pub fn set_ntp_map_a(&mut self, ntp_usec: u64, rtp_ts: u32) {
        self.ntp_map_a = Some((ntp_usec, rtp_ts));
    }

    /// Record an RTCP SR mapping for stream B.
    pub fn set_ntp_map_b(&mut self, ntp_usec: u64, rtp_ts: u32) {
        self.ntp_map_b = Some((ntp_usec, rtp_ts));
    }

    /// Compute the NTP timestamp in microseconds for an RTP timestamp in stream A.
    #[must_use]
    pub fn rtp_to_ntp_a(&self, rtp_ts: u32) -> Option<u64> {
        let (ntp_ref, rtp_ref) = self.ntp_map_a?;
        let diff_ticks = rtp_ts.wrapping_sub(rtp_ref) as i32;
        let diff_us = (i64::from(diff_ticks) * 1_000_000) / i64::from(self.clock_rate_a);
        Some((ntp_ref as i64 + diff_us) as u64)
    }

    /// Compute the NTP timestamp in microseconds for an RTP timestamp in stream B.
    #[must_use]
    pub fn rtp_to_ntp_b(&self, rtp_ts: u32) -> Option<u64> {
        let (ntp_ref, rtp_ref) = self.ntp_map_b?;
        let diff_ticks = rtp_ts.wrapping_sub(rtp_ref) as i32;
        let diff_us = (i64::from(diff_ticks) * 1_000_000) / i64::from(self.clock_rate_b);
        Some((ntp_ref as i64 + diff_us) as u64)
    }

    /// Measure the sync gap between two RTP samples (one from each stream).
    pub fn measure_gap(&mut self, ts_a: u32, ts_b: u32) -> Option<SyncGap> {
        let ntp_a = self.rtp_to_ntp_a(ts_a)?;
        let ntp_b = self.rtp_to_ntp_b(ts_b)?;
        let gap_us = ntp_a as i64 - ntp_b as i64;
        let is_excessive = gap_us.unsigned_abs() > self.max_gap_us;
        let gap = SyncGap {
            gap_us,
            is_excessive,
        };
        if self.gap_history.len() == MAX_HISTORY {
            self.gap_history.pop_front();
        }
        self.gap_history.push_back(gap);
        Some(gap)
    }

    /// Return the mean sync gap (in microseconds) over recent history.
    #[must_use]
    pub fn mean_gap_us(&self) -> f64 {
        if self.gap_history.is_empty() {
            return 0.0;
        }
        let sum: i64 = self.gap_history.iter().map(|g| g.gap_us).sum();
        sum as f64 / self.gap_history.len() as f64
    }

    /// Return the number of excessive gaps detected.
    #[must_use]
    pub fn excessive_gap_count(&self) -> usize {
        self.gap_history.iter().filter(|g| g.is_excessive).count()
    }

    /// Clear measurement history.
    pub fn reset(&mut self) {
        self.gap_history.clear();
    }
}

/// RTP sequence number discontinuity detector.
#[derive(Debug)]
pub struct SequenceChecker {
    /// Last seen sequence number.
    last_seq: Option<u16>,
    /// Total gap events.
    pub gap_count: u64,
    /// Total duplicate events.
    pub duplicate_count: u64,
}

impl SequenceChecker {
    /// Create a new checker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            last_seq: None,
            gap_count: 0,
            duplicate_count: 0,
        }
    }

    /// Process an incoming sequence number. Returns the gap (0 = in-order).
    pub fn process(&mut self, seq: u16) -> i32 {
        if let Some(last) = self.last_seq {
            let expected = last.wrapping_add(1);
            let delta = seq.wrapping_sub(expected) as i16;
            if delta == 0 {
                self.last_seq = Some(seq);
                return 0;
            } else if delta < 0 {
                self.duplicate_count += 1;
                return i32::from(delta);
            }
            self.gap_count += 1;
            self.last_seq = Some(seq);
            return i32::from(delta);
        }
        self.last_seq = Some(seq);
        0
    }

    /// Reset the checker.
    pub fn reset(&mut self) {
        self.last_seq = None;
        self.gap_count = 0;
        self.duplicate_count = 0;
    }
}

impl Default for SequenceChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Timestamp continuity validator (checks RTP timestamp monotonicity).
#[derive(Debug)]
pub struct TimestampValidator {
    /// Last seen RTP timestamp.
    last_ts: Option<u32>,
    /// Expected timestamp increment per packet.
    pub expected_increment: u32,
    /// Clock rate.
    pub clock_rate: u32,
    /// Number of discontinuities detected.
    pub discontinuity_count: u64,
}

impl TimestampValidator {
    /// Create a validator for a given clock rate and frame/packet rate.
    ///
    /// For video at 25fps and 90kHz clock: `expected_increment` = 90000 / 25 = 3600.
    #[must_use]
    pub fn new(clock_rate: u32, packet_rate: u32) -> Self {
        let expected_increment = clock_rate / packet_rate;
        Self {
            last_ts: None,
            expected_increment,
            clock_rate,
            discontinuity_count: 0,
        }
    }

    /// Validate an incoming RTP timestamp. Returns true if continuous.
    pub fn validate(&mut self, ts: u32) -> bool {
        if let Some(last) = self.last_ts {
            let actual_delta = ts.wrapping_sub(last);
            let is_ok = actual_delta == self.expected_increment;
            if !is_ok {
                self.discontinuity_count += 1;
            }
            self.last_ts = Some(ts);
            is_ok
        } else {
            self.last_ts = Some(ts);
            true
        }
    }

    /// Reset validator state.
    pub fn reset(&mut self) {
        self.last_ts = None;
        self.discontinuity_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtp_clock_rate_constants() {
        assert_eq!(RTP_CLOCK_RATE_VIDEO, 90_000);
        assert_eq!(RTP_CLOCK_RATE_AUDIO_48K, 48_000);
        assert_eq!(RTP_CLOCK_RATE_AUDIO_96K, 96_000);
    }

    #[test]
    fn test_rtp_sample_creation() {
        let s = RtpSample::new(100, 90_000, 1_000_000);
        assert_eq!(s.seq, 100);
        assert_eq!(s.timestamp, 90_000);
        assert_eq!(s.arrival_us, 1_000_000);
    }

    #[test]
    fn test_stream_sync_monitor_ntp_map() {
        let mut mon = StreamSyncMonitor::new(90_000, 48_000, 1_000);
        mon.set_ntp_map_a(1_000_000_000_000, 900_000);
        mon.set_ntp_map_b(1_000_000_000_000, 480_000);
        let ntp_a = mon
            .rtp_to_ntp_a(900_000 + 90_000)
            .expect("should succeed in test");
        // 1 second later in stream A (90_000 ticks at 90kHz = 1s = 1_000_000 us)
        assert_eq!(ntp_a, 1_000_000_000_000 + 1_000_000);
    }

    #[test]
    fn test_stream_sync_monitor_no_map_returns_none() {
        let mon = StreamSyncMonitor::new(90_000, 48_000, 1_000);
        assert!(mon.rtp_to_ntp_a(0).is_none());
        assert!(mon.rtp_to_ntp_b(0).is_none());
    }

    #[test]
    fn test_stream_sync_monitor_measure_gap_zero() {
        let mut mon = StreamSyncMonitor::new(90_000, 48_000, 1_000);
        // Same NTP base for both streams.
        mon.set_ntp_map_a(0, 0);
        mon.set_ntp_map_b(0, 0);
        // Both at time 0: gap should be 0.
        let gap = mon.measure_gap(0, 0).expect("should succeed in test");
        assert_eq!(gap.gap_us, 0);
        assert!(!gap.is_excessive);
    }

    #[test]
    fn test_stream_sync_monitor_excessive_gap() {
        let mut mon = StreamSyncMonitor::new(90_000, 48_000, 500);
        mon.set_ntp_map_a(0, 0);
        mon.set_ntp_map_b(0, 0);
        // Stream A is 1000 us ahead of stream B.
        let gap = mon.measure_gap(90, 0).expect("should succeed in test"); // 90/90000 * 1e6 = 1000 us
        assert!(gap.is_excessive);
    }

    #[test]
    fn test_stream_sync_monitor_mean_gap() {
        let mut mon = StreamSyncMonitor::new(90_000, 48_000, 10_000);
        mon.set_ntp_map_a(0, 0);
        mon.set_ntp_map_b(0, 0);
        mon.measure_gap(0, 0);
        mon.measure_gap(0, 0);
        assert!((mon.mean_gap_us() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_stream_sync_monitor_reset() {
        let mut mon = StreamSyncMonitor::new(90_000, 48_000, 1_000);
        mon.set_ntp_map_a(0, 0);
        mon.set_ntp_map_b(0, 0);
        mon.measure_gap(0, 0);
        mon.reset();
        assert_eq!(mon.excessive_gap_count(), 0);
    }

    #[test]
    fn test_sequence_checker_in_order() {
        let mut chk = SequenceChecker::new();
        assert_eq!(chk.process(0), 0);
        assert_eq!(chk.process(1), 0);
        assert_eq!(chk.process(2), 0);
        assert_eq!(chk.gap_count, 0);
    }

    #[test]
    fn test_sequence_checker_gap() {
        let mut chk = SequenceChecker::new();
        chk.process(0);
        let delta = chk.process(3); // gap of 2
        assert_eq!(delta, 2);
        assert_eq!(chk.gap_count, 1);
    }

    #[test]
    fn test_sequence_checker_duplicate() {
        let mut chk = SequenceChecker::new();
        chk.process(5);
        let delta = chk.process(5); // duplicate
        assert!(delta < 0);
        assert_eq!(chk.duplicate_count, 1);
    }

    #[test]
    fn test_sequence_checker_wrap() {
        let mut chk = SequenceChecker::new();
        chk.process(65535);
        let delta = chk.process(0); // wrap-around
        assert_eq!(delta, 0);
    }

    #[test]
    fn test_timestamp_validator_continuous() {
        let mut val = TimestampValidator::new(90_000, 25);
        assert!(val.validate(0));
        assert!(val.validate(3_600));
        assert!(val.validate(7_200));
        assert_eq!(val.discontinuity_count, 0);
    }

    #[test]
    fn test_timestamp_validator_discontinuity() {
        let mut val = TimestampValidator::new(90_000, 25);
        val.validate(0);
        let ok = val.validate(4_000); // wrong delta
        assert!(!ok);
        assert_eq!(val.discontinuity_count, 1);
    }

    #[test]
    fn test_timestamp_validator_reset() {
        let mut val = TimestampValidator::new(90_000, 25);
        val.validate(0);
        val.validate(9_999);
        val.reset();
        assert_eq!(val.discontinuity_count, 0);
        assert!(val.last_ts.is_none());
    }
}

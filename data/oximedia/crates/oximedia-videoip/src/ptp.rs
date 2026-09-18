//! Full PTP (IEEE 1588-2019) clock synchronization.
//!
//! Implements PTP ordinary/boundary clock state machines, the Best Master Clock
//! Algorithm (BMCA) data-set comparison, servo offset tracking, and sync-state
//! management for sub-microsecond time alignment in SMPTE ST 2110 environments.
//!
//! The servo model follows the two-step clock method:
//! ```text
//!   offset = (T2 - T1) - path_delay
//! ```
//! where T1 is the master's sync-message origin timestamp, T2 is the slave's
//! reception timestamp, and `path_delay` is the mean path delay measured via
//! the delay-request/response mechanism.

#![allow(dead_code)]

use crate::ptp_boundary::ClockIdentity;

// ── PTP Timestamp ────────────────────────────────────────────────────────────

/// IEEE 1588 timestamp: TAI seconds since epoch + sub-second nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PtpTimestamp {
    /// Whole seconds (48-bit in the wire format; u64 here for convenience).
    pub seconds: u64,
    /// Sub-second nanoseconds, 0 … 999_999_999.
    pub nanoseconds: u32,
}

impl PtpTimestamp {
    /// Construct a new PTP timestamp.
    #[must_use]
    pub const fn new(seconds: u64, nanoseconds: u32) -> Self {
        Self {
            seconds,
            nanoseconds,
        }
    }

    /// Convert to a total-nanoseconds representation (for arithmetic).
    ///
    /// Saturates at `u64::MAX` to avoid overflow on unrealistic timestamps.
    #[must_use]
    pub fn to_nanos(self) -> u64 {
        self.seconds
            .saturating_mul(1_000_000_000)
            .saturating_add(u64::from(self.nanoseconds))
    }

    /// Compute the signed difference `self - other` in nanoseconds.
    #[must_use]
    pub fn diff_nanos(self, other: Self) -> i64 {
        let a = self.to_nanos();
        let b = other.to_nanos();
        // Safe: timestamps are within decades of each other in practice.
        (a as i64).wrapping_sub(b as i64)
    }
}

// ── BMCA State ───────────────────────────────────────────────────────────────

/// PTP port / clock state as defined in IEEE 1588-2019 §9.2.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BmcaState {
    /// Port is starting up; no packets sent or processed yet.
    #[default]
    Initializing,
    /// Listening for announce messages; no grandmaster selected yet.
    Listening,
    /// This clock is the best master and is transmitting sync messages.
    Master,
    /// This clock is slaved to a remote grandmaster.
    Slave,
    /// A better master exists but on a different path; port is quiet.
    Passive,
    /// Transitional state before entering Slave; servo not yet locked.
    Uncalibrated,
}

// ── PTP Sync State ───────────────────────────────────────────────────────────

/// Dynamic synchronization state for a single PTP port.
#[derive(Debug, Clone)]
pub struct PtpSyncState {
    /// Current BMCA port state.
    pub state: BmcaState,
    /// Identity of the current grandmaster clock, if known.
    pub master_identity: Option<ClockIdentity>,
    /// Log₂ of the sync interval in seconds (e.g. −3 → 125 ms, 0 → 1 s).
    pub sync_interval_log: i8,
    /// Correction field from the most recent Follow_Up message (nanoseconds).
    pub follow_up_correction: i64,
}

impl PtpSyncState {
    /// Create a new sync-state in `Initializing`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: BmcaState::Initializing,
            master_identity: None,
            sync_interval_log: -3, // 125 ms default
            follow_up_correction: 0,
        }
    }

    /// Transition to a new BMCA state, resetting master identity when moving
    /// to `Listening` or `Initializing`.
    pub fn transition(&mut self, new_state: BmcaState) {
        if matches!(new_state, BmcaState::Initializing | BmcaState::Listening) {
            self.master_identity = None;
            self.follow_up_correction = 0;
        }
        self.state = new_state;
    }

    /// Set the grandmaster identity (called when a Sync/Announce is accepted).
    pub fn set_master(&mut self, identity: ClockIdentity) {
        self.master_identity = Some(identity);
    }

    /// Apply a Follow_Up correction field update.
    pub fn apply_follow_up(&mut self, correction_ns: i64) {
        self.follow_up_correction = correction_ns;
    }

    /// Sync interval as a `std::time::Duration`.
    #[must_use]
    pub fn sync_interval_duration(&self) -> std::time::Duration {
        let secs = 2.0_f64.powi(i32::from(self.sync_interval_log));
        std::time::Duration::from_secs_f64(secs.max(0.0))
    }
}

impl Default for PtpSyncState {
    fn default() -> Self {
        Self::new()
    }
}

// ── PTP Clock ────────────────────────────────────────────────────────────────

/// PTP ordinary clock — tracks servo offset, path delay, and sync quality.
///
/// This struct models the data set of a single-port ordinary clock as
/// described in IEEE 1588-2019 §8.2 (Default Data Set).
#[derive(Debug, Clone)]
pub struct PtpClock {
    /// PTP domain number (0–127; 0 is default).
    pub domain: u8,
    /// Priority 1: primary BMCA selection criterion (lower wins; 128 default).
    pub priority1: u8,
    /// Priority 2: tiebreaker after data-set comparison (lower wins; 128 default).
    pub priority2: u8,
    /// Clock class (raw u8): 135 = locked to GNSS, 187 = holdover, 248 = default.
    pub clock_class: u8,
    /// Clock accuracy (raw u8): 0x21 = <100 ns, 0x22 = <250 ns, etc.
    pub clock_accuracy: u8,
    /// Current offset from master in nanoseconds (positive = slave ahead).
    pub offset_from_master: i64,
    /// Mean path delay in nanoseconds (one-way).
    pub path_delay: u64,
    /// Offset-scaled log variance (clock variance per IEEE 1588 §7.6.3).
    pub variance: u64,
}

impl PtpClock {
    /// Create a new PTP clock for `domain` with default priority/class values.
    #[must_use]
    pub const fn new(domain: u8) -> Self {
        Self {
            domain,
            priority1: 128,
            priority2: 128,
            clock_class: 248,     // Default / slave-only
            clock_accuracy: 0xFE, // Unknown
            offset_from_master: 0,
            path_delay: 0,
            variance: 0x4E5D, // typical default log variance
        }
    }

    /// Update the clock offset using the IEEE 1588 two-step calculation:
    ///
    /// ```text
    /// offset_from_master = (T2 − T1) − path_delay
    /// ```
    ///
    /// where:
    /// - `master_ts` (T1) is the *corrected* origin timestamp from Sync/Follow_Up,
    /// - `slave_ts`  (T2) is the slave's local reception timestamp,
    /// - `delay`        is the previously measured mean path delay (ns).
    pub fn update_offset(&mut self, master_ts: PtpTimestamp, slave_ts: PtpTimestamp, delay: u64) {
        let t2_minus_t1 = slave_ts.diff_nanos(master_ts);
        self.path_delay = delay;
        self.offset_from_master = t2_minus_t1 - delay as i64;
    }

    /// Apply an external correction (e.g. from a Follow_Up message) to the
    /// stored offset.
    pub fn apply_correction(&mut self, correction_ns: i64) {
        self.offset_from_master = self.offset_from_master.saturating_sub(correction_ns);
    }

    /// Returns `true` when the clock is considered phase-locked:
    /// |offset_from_master| < 1 000 ns (1 µs).
    #[must_use]
    pub fn is_synchronized(&self) -> bool {
        self.offset_from_master.unsigned_abs() < 1_000
    }

    /// Convenience: set clock class and accuracy together when lock status changes.
    pub fn set_lock_state(&mut self, clock_class: u8, clock_accuracy: u8) {
        self.clock_class = clock_class;
        self.clock_accuracy = clock_accuracy;
    }

    /// Build the IEEE 1588 data-set comparison vector:
    /// `(priority1, clock_class, clock_accuracy, variance, identity_placeholder, priority2)`.
    ///
    /// Note: identity is not part of `PtpClock` itself; the `BmcaEngine`
    /// injects its known `ClockIdentity` when doing cross-clock comparisons.
    #[must_use]
    pub fn comparison_vector_without_id(&self) -> (u8, u8, u8, u64, u8) {
        (
            self.priority1,
            self.clock_class,
            self.clock_accuracy,
            self.variance,
            self.priority2,
        )
    }
}

// ── BMCA Engine ──────────────────────────────────────────────────────────────

/// Full BMCA comparison vector including clock identity (EUI-64).
///
/// Fields ordered per IEEE 1588-2019 §9.3.4:
/// `(priority1, class, accuracy, variance, identity, priority2)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BmcaDataset {
    /// Priority 1.
    pub priority1: u8,
    /// Clock class.
    pub clock_class: u8,
    /// Clock accuracy.
    pub clock_accuracy: u8,
    /// Offset scaled log variance (lower is better).
    pub variance: u64,
    /// Clock identity (EUI-64) — lexicographic tiebreaker.
    pub clock_identity: [u8; 8],
    /// Priority 2.
    pub priority2: u8,
}

impl BmcaDataset {
    /// Build a `BmcaDataset` from a `PtpClock` and its associated `ClockIdentity`.
    #[must_use]
    pub fn from_clock(clock: &PtpClock, identity: ClockIdentity) -> Self {
        Self {
            priority1: clock.priority1,
            clock_class: clock.clock_class,
            clock_accuracy: clock.clock_accuracy,
            variance: clock.variance,
            clock_identity: identity.0,
            priority2: clock.priority2,
        }
    }
}

/// Best Master Clock Algorithm engine.
///
/// Performs the IEEE 1588-2019 §9.3.4 data-set comparison between two foreign
/// master clock descriptors to determine which should be preferred.
#[derive(Debug, Clone)]
pub struct BmcaEngine {
    /// Identity of the local clock (used as tiebreaker / self-comparison).
    pub local_clock: ClockIdentity,
}

impl BmcaEngine {
    /// Create a new BMCA engine for the given local clock identity.
    #[must_use]
    pub const fn new(local_identity: ClockIdentity) -> Self {
        Self {
            local_clock: local_identity,
        }
    }

    /// Compare two foreign master clock data sets per IEEE 1588 BMCA.
    ///
    /// Returns `Ordering::Less` if `a` is a *better* master than `b`
    /// (i.e. `a` should be preferred).
    ///
    /// The comparison priority is:
    /// 1. Priority 1 (lower wins)
    /// 2. Clock class (lower wins)
    /// 3. Clock accuracy (lower wins — finer granularity)
    /// 4. Offset scaled log variance (lower wins)
    /// 5. Clock identity bytes (lexicographic; lower wins)
    /// 6. Priority 2 (lower wins)
    #[must_use]
    pub fn compare_datasets(
        &self,
        a: &PtpClock,
        b: &PtpClock,
        a_id: ClockIdentity,
        b_id: ClockIdentity,
    ) -> std::cmp::Ordering {
        let da = BmcaDataset::from_clock(a, a_id);
        let db = BmcaDataset::from_clock(b, b_id);
        da.cmp(&db)
    }

    /// Compare two clocks and return which is the better master (the one
    /// that `compare_datasets` considers `Less`).
    ///
    /// Returns `Some(a_id)` if `a` wins, `Some(b_id)` if `b` wins, or
    /// `None` if they are identical.
    #[must_use]
    pub fn best_master(
        &self,
        a: &PtpClock,
        b: &PtpClock,
        a_id: ClockIdentity,
        b_id: ClockIdentity,
    ) -> Option<ClockIdentity> {
        match self.compare_datasets(a, b, a_id, b_id) {
            std::cmp::Ordering::Less => Some(a_id),
            std::cmp::Ordering::Greater => Some(b_id),
            std::cmp::Ordering::Equal => None,
        }
    }

    /// Determine whether the local clock should become grandmaster given a
    /// list of foreign master clocks.
    ///
    /// Returns `true` if the local clock beats all foreign candidates.
    #[must_use]
    pub fn should_be_grandmaster(
        &self,
        local: &PtpClock,
        foreign_masters: &[(PtpClock, ClockIdentity)],
    ) -> bool {
        let local_ds = BmcaDataset::from_clock(local, self.local_clock);
        for (fm, fm_id) in foreign_masters {
            let fm_ds = BmcaDataset::from_clock(fm, *fm_id);
            if fm_ds < local_ds {
                return false; // at least one foreign master is better
            }
        }
        true
    }
}

// ── Delay-Request Mechanism ───────────────────────────────────────────────────

/// Four-timestamp tuple used for the delay-request / delay-response mechanism.
///
/// ```text
/// path_delay = ((T2 - T1) + (T4 - T3)) / 2
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DelayMeasurement {
    /// T1: master sends Sync (or origin TS from Follow_Up).
    pub t1: PtpTimestamp,
    /// T2: slave receives Sync.
    pub t2: PtpTimestamp,
    /// T3: slave sends Delay_Req.
    pub t3: PtpTimestamp,
    /// T4: master receives Delay_Req (from Delay_Resp message).
    pub t4: PtpTimestamp,
}

impl DelayMeasurement {
    /// Compute the mean path delay in nanoseconds using the E2E mechanism.
    ///
    /// Returns `None` if the result would be negative (indicating inconsistent
    /// timestamps or asymmetric network conditions beyond compensation range).
    #[must_use]
    pub fn mean_path_delay_nanos(&self) -> Option<u64> {
        // (T2 − T1) + (T4 − T3)  — both should be positive
        let t2_minus_t1 = self.t2.diff_nanos(self.t1);
        let t4_minus_t3 = self.t4.diff_nanos(self.t3);
        let sum = t2_minus_t1.checked_add(t4_minus_t3)?;
        if sum < 0 {
            return None;
        }
        Some((sum as u64) / 2)
    }

    /// Compute the offset from master (without path-delay compensation) as a
    /// convenience for testing; normally `PtpClock::update_offset` is used.
    #[must_use]
    pub fn raw_offset_nanos(&self) -> i64 {
        self.t2.diff_nanos(self.t1)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptp_boundary::ClockIdentity;

    fn id(v: u64) -> ClockIdentity {
        ClockIdentity::from_u64(v)
    }

    fn ts(s: u64, ns: u32) -> PtpTimestamp {
        PtpTimestamp::new(s, ns)
    }

    // ── PtpTimestamp ──────────────────────────────────────────────────────────

    #[test]
    fn test_ptp_timestamp_to_nanos() {
        let t = ts(1, 500_000_000);
        assert_eq!(t.to_nanos(), 1_500_000_000);
    }

    #[test]
    fn test_ptp_timestamp_diff_positive() {
        let a = ts(2, 0);
        let b = ts(1, 0);
        assert_eq!(a.diff_nanos(b), 1_000_000_000);
    }

    #[test]
    fn test_ptp_timestamp_diff_negative() {
        let a = ts(1, 0);
        let b = ts(2, 0);
        assert_eq!(a.diff_nanos(b), -1_000_000_000);
    }

    #[test]
    fn test_ptp_timestamp_diff_sub_second() {
        let a = ts(0, 700_000_000);
        let b = ts(0, 200_000_000);
        assert_eq!(a.diff_nanos(b), 500_000_000);
    }

    // ── PtpClock ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ptp_clock_new_defaults() {
        let clk = PtpClock::new(0);
        assert_eq!(clk.domain, 0);
        assert_eq!(clk.priority1, 128);
        assert_eq!(clk.priority2, 128);
        assert_eq!(clk.offset_from_master, 0);
        assert_eq!(clk.path_delay, 0);
        // offset = 0 → |0| < 1000 → considered synchronized at power-on
        assert!(clk.is_synchronized());
    }

    #[test]
    fn test_ptp_clock_is_synchronized_zero_offset() {
        let clk = PtpClock::new(0);
        // offset 0 → |0| < 1000 → synchronized
        assert!(clk.is_synchronized());
    }

    #[test]
    fn test_ptp_clock_is_synchronized_small_offset() {
        let mut clk = PtpClock::new(0);
        clk.offset_from_master = 999;
        assert!(clk.is_synchronized());
    }

    #[test]
    fn test_ptp_clock_not_synchronized_large_offset() {
        let mut clk = PtpClock::new(0);
        clk.offset_from_master = 1_500;
        assert!(!clk.is_synchronized());
        clk.offset_from_master = -1_500;
        assert!(!clk.is_synchronized());
    }

    #[test]
    fn test_ptp_clock_update_offset() {
        let mut clk = PtpClock::new(0);
        // T1 = master sent at 1.000_000_000 s
        // T2 = slave received at 1.000_001_000 s  (1000 ns later)
        // delay = 400 ns
        // expected offset = (1000 - 400) = 600 ns
        let t1 = ts(1, 0);
        let t2 = ts(1, 1_000);
        clk.update_offset(t1, t2, 400);
        assert_eq!(clk.offset_from_master, 600);
        assert_eq!(clk.path_delay, 400);
    }

    #[test]
    fn test_ptp_clock_apply_correction() {
        let mut clk = PtpClock::new(0);
        clk.offset_from_master = 800;
        clk.apply_correction(300);
        assert_eq!(clk.offset_from_master, 500);
    }

    #[test]
    fn test_ptp_clock_apply_negative_correction() {
        let mut clk = PtpClock::new(0);
        clk.offset_from_master = 200;
        clk.apply_correction(-100); // subtracting negative = adding
        assert_eq!(clk.offset_from_master, 300);
    }

    // ── PtpSyncState ─────────────────────────────────────────────────────────

    #[test]
    fn test_sync_state_default() {
        let s = PtpSyncState::default();
        assert_eq!(s.state, BmcaState::Initializing);
        assert!(s.master_identity.is_none());
        assert_eq!(s.follow_up_correction, 0);
    }

    #[test]
    fn test_sync_state_transition_to_slave() {
        let mut s = PtpSyncState::new();
        s.transition(BmcaState::Slave);
        assert_eq!(s.state, BmcaState::Slave);
    }

    #[test]
    fn test_sync_state_transition_to_listening_clears_master() {
        let mut s = PtpSyncState::new();
        s.set_master(id(0xAA));
        s.follow_up_correction = 999;
        s.transition(BmcaState::Listening);
        assert!(s.master_identity.is_none());
        assert_eq!(s.follow_up_correction, 0);
    }

    #[test]
    fn test_sync_state_sync_interval_duration() {
        let mut s = PtpSyncState::new();
        s.sync_interval_log = 0; // 1 second
        let d = s.sync_interval_duration();
        assert!((d.as_secs_f64() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sync_state_sync_interval_negative() {
        let mut s = PtpSyncState::new();
        s.sync_interval_log = -3; // 125 ms
        let d = s.sync_interval_duration();
        assert!((d.as_secs_f64() - 0.125).abs() < 1e-9);
    }

    // ── BmcaEngine ────────────────────────────────────────────────────────────

    #[test]
    fn test_bmca_compare_priority1() {
        let engine = BmcaEngine::new(id(0xFF));
        let mut a = PtpClock::new(0);
        a.priority1 = 64; // better (lower)
        let b = PtpClock::new(0); // priority1 = 128
        let ord = engine.compare_datasets(&a, &b, id(1), id(2));
        assert_eq!(ord, std::cmp::Ordering::Less, "lower priority1 should win");
    }

    #[test]
    fn test_bmca_compare_clock_class() {
        let engine = BmcaEngine::new(id(0xFF));
        let mut a = PtpClock::new(0);
        a.clock_class = 6; // Primary (better)
        let b = PtpClock::new(0); // clock_class = 248
        let ord = engine.compare_datasets(&a, &b, id(1), id(2));
        assert_eq!(ord, std::cmp::Ordering::Less);
    }

    #[test]
    fn test_bmca_compare_identity_tiebreaker() {
        let engine = BmcaEngine::new(id(0xFF));
        let a = PtpClock::new(0);
        let b = PtpClock::new(0);
        // All fields equal except identity
        let ord = engine.compare_datasets(&a, &b, id(1), id(2));
        assert_eq!(ord, std::cmp::Ordering::Less, "lower identity bytes win");
    }

    #[test]
    fn test_bmca_best_master() {
        let engine = BmcaEngine::new(id(0xFF));
        let mut a = PtpClock::new(0);
        a.priority1 = 100;
        let b = PtpClock::new(0); // priority1 = 128
        let winner = engine.best_master(&a, &b, id(0xAA), id(0xBB));
        assert_eq!(winner, Some(id(0xAA)));
    }

    #[test]
    fn test_bmca_should_be_grandmaster_no_peers() {
        let engine = BmcaEngine::new(id(0x01));
        let local = PtpClock::new(0);
        assert!(engine.should_be_grandmaster(&local, &[]));
    }

    #[test]
    fn test_bmca_should_not_be_grandmaster_if_peer_better() {
        let engine = BmcaEngine::new(id(0x80));
        let local = PtpClock::new(0); // priority1=128
        let mut better = PtpClock::new(0);
        better.priority1 = 64; // better foreign master
        let peers = vec![(better, id(0x10))];
        assert!(!engine.should_be_grandmaster(&local, &peers));
    }

    // ── DelayMeasurement ──────────────────────────────────────────────────────

    #[test]
    fn test_delay_measurement_symmetric() {
        // T1=0, T2=500ns, T3=600ns, T4=1100ns
        // delay = ((500 - 0) + (1100 - 600)) / 2 = (500 + 500) / 2 = 500 ns
        let m = DelayMeasurement {
            t1: ts(0, 0),
            t2: ts(0, 500),
            t3: ts(0, 600),
            t4: ts(0, 1_100),
        };
        assert_eq!(m.mean_path_delay_nanos(), Some(500));
    }

    #[test]
    fn test_delay_measurement_raw_offset() {
        let m = DelayMeasurement {
            t1: ts(0, 0),
            t2: ts(0, 1_000),
            t3: ts(0, 0),
            t4: ts(0, 0),
        };
        assert_eq!(m.raw_offset_nanos(), 1_000);
    }
}

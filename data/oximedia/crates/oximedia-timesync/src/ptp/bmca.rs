//! Best Master Clock Algorithm (BMCA) implementation.
//!
//! Implements IEEE 1588-2019 BMCA for selecting the best master clock.

use super::dataset::DefaultDataSet;
use super::message::{AnnounceMessage, ClockQuality};
use super::{ClockIdentity, PortIdentity};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Result of BMCA comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BmcaResult {
    /// The announce message represents a better master
    BetterMaster,
    /// The announce message represents a worse master
    WorseMaster,
    /// The announce message is from the same master
    SameMaster,
}

/// Compare two announce messages using BMCA.
#[must_use]
pub fn compare_announce(local: &DefaultDataSet, announce: &AnnounceMessage) -> BmcaResult {
    // Compare using dataset comparison algorithm
    let ordering = compare_dataset_quality(
        local.priority1,
        &local.clock_quality,
        local.priority2,
        local.clock_identity,
        announce.grandmaster_priority1,
        &announce.grandmaster_clock_quality,
        announce.grandmaster_priority2,
        announce.grandmaster_identity,
    );

    // compare_dataset_quality returns Greater if A (local) is better than B (announce).
    // BmcaResult describes whether the *announce message* represents a better or worse master.
    match ordering {
        Ordering::Less => BmcaResult::BetterMaster, // local is worse → announce is better
        Ordering::Greater => BmcaResult::WorseMaster, // local is better → announce is worse
        Ordering::Equal => BmcaResult::SameMaster,
    }
}

/// Compare two clocks using the dataset comparison algorithm.
///
/// Returns `Ordering::Greater` if clock A is better than clock B.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn compare_dataset_quality(
    priority1_a: u8,
    quality_a: &ClockQuality,
    priority2_a: u8,
    identity_a: ClockIdentity,
    priority1_b: u8,
    quality_b: &ClockQuality,
    priority2_b: u8,
    identity_b: ClockIdentity,
) -> Ordering {
    // Step 1: Compare priority1 (lower is better)
    match priority1_a.cmp(&priority1_b) {
        Ordering::Less => return Ordering::Greater,
        Ordering::Greater => return Ordering::Less,
        Ordering::Equal => {}
    }

    // Step 2: Compare clock class (lower is better)
    match quality_a.clock_class.cmp(&quality_b.clock_class) {
        Ordering::Less => return Ordering::Greater,
        Ordering::Greater => return Ordering::Less,
        Ordering::Equal => {}
    }

    // Step 3: Compare clock accuracy (lower is better)
    match quality_a.clock_accuracy.cmp(&quality_b.clock_accuracy) {
        Ordering::Less => return Ordering::Greater,
        Ordering::Greater => return Ordering::Less,
        Ordering::Equal => {}
    }

    // Step 4: Compare offset scaled log variance (lower is better)
    match quality_a
        .offset_scaled_log_variance
        .cmp(&quality_b.offset_scaled_log_variance)
    {
        Ordering::Less => return Ordering::Greater,
        Ordering::Greater => return Ordering::Less,
        Ordering::Equal => {}
    }

    // Step 5: Compare priority2 (lower is better)
    match priority2_a.cmp(&priority2_b) {
        Ordering::Less => return Ordering::Greater,
        Ordering::Greater => return Ordering::Less,
        Ordering::Equal => {}
    }

    // Step 6: Compare clock identity (lower is better)
    match identity_a.cmp(&identity_b) {
        Ordering::Less => Ordering::Greater,
        Ordering::Greater => Ordering::Less,
        Ordering::Equal => Ordering::Equal,
    }
}

/// PTP port state based on BMCA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortState {
    /// Initializing
    Initializing,
    /// Faulty
    Faulty,
    /// Disabled
    Disabled,
    /// Listening
    Listening,
    /// Pre-Master
    PreMaster,
    /// Master
    Master,
    /// Passive
    Passive,
    /// Uncalibrated
    Uncalibrated,
    /// Slave
    Slave,
}

/// Recommended state for a port based on BMCA result.
pub struct StateRecommendation {
    /// Recommended port state
    pub state: PortState,
    /// Best master port identity (if slave)
    pub best_master: Option<PortIdentity>,
}

/// Determine recommended state based on BMCA result.
#[must_use]
pub fn recommend_state(
    local: &DefaultDataSet,
    announce: Option<&AnnounceMessage>,
    current_state: PortState,
) -> StateRecommendation {
    match announce {
        None => {
            // No announce messages received, we should be master
            StateRecommendation {
                state: if current_state == PortState::Initializing {
                    PortState::Listening
                } else {
                    PortState::Master
                },
                best_master: None,
            }
        }
        Some(ann) => {
            let result = compare_announce(local, ann);
            match result {
                BmcaResult::BetterMaster => {
                    // External clock is better, become slave
                    StateRecommendation {
                        state: PortState::Slave,
                        best_master: Some(ann.header.source_port_identity),
                    }
                }
                BmcaResult::WorseMaster => {
                    // We are better, become master
                    StateRecommendation {
                        state: PortState::Master,
                        best_master: None,
                    }
                }
                BmcaResult::SameMaster => {
                    // Same master, maintain current state
                    StateRecommendation {
                        state: current_state,
                        best_master: Some(ann.header.source_port_identity),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority1_comparison() {
        let id_a = ClockIdentity([1, 2, 3, 4, 5, 6, 7, 8]);
        let id_b = ClockIdentity([2, 2, 3, 4, 5, 6, 7, 8]);

        let quality = ClockQuality {
            clock_class: 6,
            clock_accuracy: 0x20,
            offset_scaled_log_variance: 0x4000,
        };

        // Lower priority1 is better
        let result = compare_dataset_quality(100, &quality, 128, id_a, 200, &quality, 128, id_b);
        assert_eq!(result, Ordering::Greater);

        let result = compare_dataset_quality(200, &quality, 128, id_a, 100, &quality, 128, id_b);
        assert_eq!(result, Ordering::Less);
    }

    #[test]
    fn test_clock_class_comparison() {
        let id_a = ClockIdentity([1, 2, 3, 4, 5, 6, 7, 8]);
        let id_b = ClockIdentity([2, 2, 3, 4, 5, 6, 7, 8]);

        let quality_a = ClockQuality {
            clock_class: 6,
            clock_accuracy: 0x20,
            offset_scaled_log_variance: 0x4000,
        };

        let quality_b = ClockQuality {
            clock_class: 7,
            clock_accuracy: 0x20,
            offset_scaled_log_variance: 0x4000,
        };

        // Same priority, lower clock class is better
        let result =
            compare_dataset_quality(128, &quality_a, 128, id_a, 128, &quality_b, 128, id_b);
        assert_eq!(result, Ordering::Greater);
    }

    #[test]
    fn test_clock_identity_tiebreaker() {
        let id_a = ClockIdentity([1, 2, 3, 4, 5, 6, 7, 8]);
        let id_b = ClockIdentity([2, 2, 3, 4, 5, 6, 7, 8]);

        let quality = ClockQuality {
            clock_class: 6,
            clock_accuracy: 0x20,
            offset_scaled_log_variance: 0x4000,
        };

        // Same everything, lower clock identity wins
        let result = compare_dataset_quality(128, &quality, 128, id_a, 128, &quality, 128, id_b);
        assert_eq!(result, Ordering::Greater);
    }
}

// ---------------------------------------------------------------------------
// Announce timeout tracking and re-election support
// ---------------------------------------------------------------------------

/// Event fired when a master clock's announce messages stop arriving within
/// the configured timeout window (announce_interval × announce_receipt_timeout).
#[derive(Debug, Clone)]
pub struct MasterLossEvent {
    /// Identity of the master that stopped sending announces.
    pub lost_master: PortIdentity,
    /// Monotonic instant of the last observed announce from this master.
    pub last_seen: Instant,
    /// The timeout duration that was exceeded.
    pub timeout: Duration,
}

/// Per-source state maintained by [`AnnounceTimeoutTracker`].
#[derive(Debug, Clone)]
struct AnnounceSourceState {
    last_seen: Instant,
}

/// Tracks when announce messages were last received from each known master
/// and reports [`MasterLossEvent`]s when a master has been silent for longer
/// than the configured timeout.
///
/// Per IEEE 1588-2019 §9.2.6.11, a port transitions out of `Slave` when it
/// has not received an `Announce` message within
/// `announceReceiptTimeout × 2^logAnnounceInterval` seconds.
#[derive(Debug)]
pub struct AnnounceTimeoutTracker {
    /// Per-source last-seen timestamps.
    sources: HashMap<PortIdentity, AnnounceSourceState>,
    /// Configured timeout duration applied to every tracked source.
    timeout: Duration,
}

impl AnnounceTimeoutTracker {
    /// Creates a tracker with the given timeout duration.
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        Self {
            sources: HashMap::new(),
            timeout,
        }
    }

    /// Records receipt of an announce message from `source` at `now`.
    pub fn record_announce(&mut self, source: PortIdentity, now: Instant) {
        self.sources
            .insert(source, AnnounceSourceState { last_seen: now });
    }

    /// Checks all tracked sources against the timeout.
    ///
    /// Returns a [`MasterLossEvent`] for every source whose last announce was
    /// received more than `timeout` ago relative to `now`.  The timed-out
    /// sources are **not** removed automatically — call `remove_source` or
    /// `record_announce` to update tracking.
    #[must_use]
    pub fn check_timeouts(&self, now: Instant) -> Vec<MasterLossEvent> {
        self.sources
            .iter()
            .filter_map(|(id, state)| {
                let elapsed = now.saturating_duration_since(state.last_seen);
                if elapsed > self.timeout {
                    Some(MasterLossEvent {
                        lost_master: *id,
                        last_seen: state.last_seen,
                        timeout: self.timeout,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Removes tracking for `source` (e.g. after a loss event is handled).
    pub fn remove_source(&mut self, source: &PortIdentity) {
        self.sources.remove(source);
    }

    /// Returns the number of currently tracked sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Returns the configured timeout duration.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

// ---------------------------------------------------------------------------
// Foreign master record and re-election engine
// ---------------------------------------------------------------------------

/// A record of the most-recently-received `Announce` message from a remote
/// master candidate, plus the monotonic instant it was received.
#[derive(Debug, Clone)]
pub struct ForeignMasterRecord {
    /// Port identity of the foreign master.
    pub identity: PortIdentity,
    /// The most recent announce message received from this master.
    pub last_announce: AnnounceMessage,
    /// Monotonic instant when `last_announce` was recorded.
    pub last_seen: Instant,
}

/// Manages the set of known foreign master candidates and implements BMCA
/// re-election when the current master is lost.
///
/// Each received `Announce` message updates (or inserts) a
/// [`ForeignMasterRecord`].  Stale records are pruned via `remove_expired`.
/// `elect_best_master` runs the full BMCA dataset-comparison algorithm over
/// all live foreign masters and returns the best candidate.
#[derive(Debug)]
pub struct ReElectionEngine {
    /// All currently tracked foreign master candidates.
    foreign_masters: HashMap<PortIdentity, ForeignMasterRecord>,
    /// How long a foreign master record is considered fresh.
    announce_timeout: Duration,
}

impl ReElectionEngine {
    /// Creates a new re-election engine with the given announce-message timeout.
    #[must_use]
    pub fn new(announce_timeout: Duration) -> Self {
        Self {
            foreign_masters: HashMap::new(),
            announce_timeout,
        }
    }

    /// Records (or updates) the foreign master record for the sender of `ann`.
    ///
    /// If a record already exists for the same [`PortIdentity`] it is
    /// replaced with the newer announce and updated timestamp.
    pub fn update_foreign_master(&mut self, ann: AnnounceMessage, now: Instant) {
        let identity = ann.header.source_port_identity;
        self.foreign_masters.insert(
            identity,
            ForeignMasterRecord {
                identity,
                last_announce: ann,
                last_seen: now,
            },
        );
    }

    /// Removes all foreign master records whose `last_seen` timestamp is
    /// older than `announce_timeout` relative to `now`.
    ///
    /// Returns the [`PortIdentity`]s of every removed record so callers can
    /// react (e.g. trigger a BMCA run if the current master was pruned).
    pub fn remove_expired(&mut self, now: Instant) -> Vec<PortIdentity> {
        let timeout = self.announce_timeout;
        let mut expired = Vec::new();
        self.foreign_masters.retain(|id, record| {
            let elapsed = now.saturating_duration_since(record.last_seen);
            if elapsed > timeout {
                expired.push(*id);
                false
            } else {
                true
            }
        });
        expired
    }

    /// Runs BMCA over all currently tracked foreign master records and returns
    /// a reference to the best candidate, or `None` if no foreign masters are
    /// available (meaning the local clock should become grandmaster).
    ///
    /// The dataset-comparison algorithm follows IEEE 1588-2019 §9.3.4.
    #[must_use]
    pub fn elect_best_master(&self, local: &DefaultDataSet) -> Option<&ForeignMasterRecord> {
        let mut best: Option<&ForeignMasterRecord> = None;

        for record in self.foreign_masters.values() {
            let ann = &record.last_announce;
            match best {
                None => {
                    // First candidate: only accept if it beats the local clock.
                    if compare_announce(local, ann) == BmcaResult::BetterMaster {
                        best = Some(record);
                    }
                }
                Some(current_best) => {
                    // Compare the new candidate against the current best using
                    // the full dataset-comparison algorithm.
                    let ordering = compare_dataset_quality(
                        ann.grandmaster_priority1,
                        &ann.grandmaster_clock_quality,
                        ann.grandmaster_priority2,
                        ann.grandmaster_identity,
                        current_best.last_announce.grandmaster_priority1,
                        &current_best.last_announce.grandmaster_clock_quality,
                        current_best.last_announce.grandmaster_priority2,
                        current_best.last_announce.grandmaster_identity,
                    );
                    // `Ordering::Greater` means the new candidate is better.
                    if ordering == Ordering::Greater {
                        best = Some(record);
                    }
                }
            }
        }

        best
    }

    /// Returns the number of currently tracked foreign master candidates.
    #[must_use]
    pub fn foreign_master_count(&self) -> usize {
        self.foreign_masters.len()
    }
}

// ---------------------------------------------------------------------------
// BmcaSelector — direct grandmaster election from identity records
// ---------------------------------------------------------------------------

/// A compact descriptor for a PTP clock candidate used in
/// [`BmcaSelector::select_grandmaster`].
///
/// Fields follow the IEEE 1588-2019 dataset comparison order:
/// `clockClass → clockAccuracy → offsetScaledLogVariance → priority1 → priority2 → identity`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtpClockIdentity {
    /// Clock class (lower is better, class 6 = locked to primary).
    pub clock_class: u8,
    /// Clock accuracy (lower is better).
    pub clock_accuracy: u8,
    /// Offset scaled log variance (lower is better).
    pub offset_scaled_log_variance: u16,
    /// Priority 1 (lower is better).
    pub priority1: u8,
    /// Priority 2 (lower is better).
    pub priority2: u8,
    /// 64-bit clock identity (EUI-64); lexicographically lower is preferred.
    pub identity: ClockIdentity,
}

/// Stateless BMCA grandmaster selector.
///
/// Implements the IEEE 1588-2019 §9.3.4 dataset comparison algorithm over a
/// slice of [`PtpClockIdentity`] candidates and returns the winner.
pub struct BmcaSelector;

impl BmcaSelector {
    /// Selects the grandmaster from `candidates` using the IEEE 1588 BMCA
    /// dataset comparison algorithm.
    ///
    /// Comparison order (lower value wins at each step):
    /// 1. `priority1`
    /// 2. `clock_class`
    /// 3. `clock_accuracy`
    /// 4. `offset_scaled_log_variance`
    /// 5. `priority2`
    /// 6. `identity` (lexicographic)
    ///
    /// Returns `None` when `candidates` is empty.
    #[must_use]
    pub fn select_grandmaster(candidates: &[PtpClockIdentity]) -> Option<PtpClockIdentity> {
        candidates.iter().fold(None, |best, candidate| {
            Some(match best {
                None => candidate.clone(),
                Some(ref current) => {
                    let quality_a = ClockQuality {
                        clock_class: candidate.clock_class,
                        clock_accuracy: candidate.clock_accuracy,
                        offset_scaled_log_variance: candidate.offset_scaled_log_variance,
                    };
                    let quality_b = ClockQuality {
                        clock_class: current.clock_class,
                        clock_accuracy: current.clock_accuracy,
                        offset_scaled_log_variance: current.offset_scaled_log_variance,
                    };
                    let ord = compare_dataset_quality(
                        candidate.priority1,
                        &quality_a,
                        candidate.priority2,
                        candidate.identity,
                        current.priority1,
                        &quality_b,
                        current.priority2,
                        current.identity,
                    );
                    if ord == std::cmp::Ordering::Greater {
                        candidate.clone()
                    } else {
                        current.clone()
                    }
                }
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Additional tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod timeout_tests {
    use super::*;
    use crate::ptp::message::{Flags, Header, MessageType};
    use crate::ptp::{ClockIdentity, Domain, PortIdentity, PtpTimestamp};

    fn make_port_id(byte: u8) -> PortIdentity {
        PortIdentity::new(ClockIdentity([byte; 8]), 1)
    }

    fn make_announce(gm_priority1: u8, gm_id_byte: u8) -> AnnounceMessage {
        use crate::ptp::message::ClockQuality;
        let clock_id = ClockIdentity([gm_id_byte; 8]);
        let port_id = PortIdentity::new(clock_id, 1);
        let header = Header {
            message_type: MessageType::Announce,
            version: 2,
            message_length: 64,
            domain: Domain::DEFAULT,
            flags: Flags::default(),
            correction_field: 0,
            source_port_identity: port_id,
            sequence_id: 1,
            control: 5,
            log_message_interval: 1,
        };
        AnnounceMessage {
            header,
            origin_timestamp: PtpTimestamp::new(0, 0).expect("valid ts"),
            current_utc_offset: 37,
            grandmaster_priority1: gm_priority1,
            grandmaster_clock_quality: ClockQuality {
                clock_class: 135,
                clock_accuracy: 0x20,
                offset_scaled_log_variance: 0x4000,
            },
            grandmaster_priority2: 128,
            grandmaster_identity: ClockIdentity([gm_id_byte; 8]),
            steps_removed: 0,
            time_source: 0x20,
        }
    }

    // -----------------------------------------------------------------------
    // AnnounceTimeoutTracker tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tracker_no_timeout_within_window() {
        let timeout = Duration::from_secs(3);
        let mut tracker = AnnounceTimeoutTracker::new(timeout);
        let now = Instant::now();
        let port = make_port_id(1);
        tracker.record_announce(port, now);
        // Check immediately — no timeout yet.
        let events = tracker.check_timeouts(now);
        assert!(events.is_empty(), "no timeout within window");
    }

    #[test]
    fn test_tracker_detects_timeout() {
        let timeout = Duration::from_millis(10);
        let mut tracker = AnnounceTimeoutTracker::new(timeout);
        let past = Instant::now();
        let port = make_port_id(2);
        tracker.record_announce(port, past);
        // Simulate time passing by using a future "now".
        let future = past + Duration::from_millis(20);
        let events = tracker.check_timeouts(future);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].lost_master, port);
    }

    #[test]
    fn test_tracker_multiple_sources_partial_timeout() {
        let timeout = Duration::from_millis(50);
        let mut tracker = AnnounceTimeoutTracker::new(timeout);
        let base = Instant::now();
        let port_a = make_port_id(10);
        let port_b = make_port_id(11);
        tracker.record_announce(port_a, base);
        // port_b received an announce recently.
        tracker.record_announce(port_b, base + Duration::from_millis(40));
        // Advance 60ms — only port_a should time out.
        let future = base + Duration::from_millis(60);
        let events = tracker.check_timeouts(future);
        assert_eq!(events.len(), 1, "only port_a should time out");
        assert_eq!(events[0].lost_master, port_a);
    }

    #[test]
    fn test_tracker_remove_source() {
        let timeout = Duration::from_millis(10);
        let mut tracker = AnnounceTimeoutTracker::new(timeout);
        let base = Instant::now();
        let port = make_port_id(5);
        tracker.record_announce(port, base);
        assert_eq!(tracker.source_count(), 1);
        tracker.remove_source(&port);
        assert_eq!(tracker.source_count(), 0);
        let events = tracker.check_timeouts(base + Duration::from_secs(10));
        assert!(events.is_empty());
    }

    #[test]
    fn test_tracker_record_renews_timeout() {
        let timeout = Duration::from_millis(20);
        let mut tracker = AnnounceTimeoutTracker::new(timeout);
        let base = Instant::now();
        let port = make_port_id(7);
        tracker.record_announce(port, base);
        // Renew just before timeout.
        let renewed = base + Duration::from_millis(15);
        tracker.record_announce(port, renewed);
        // At base+25ms the original would have expired but the renewal is still fresh.
        let check_time = base + Duration::from_millis(25);
        let events = tracker.check_timeouts(check_time);
        assert!(events.is_empty(), "renewal should reset the timeout");
    }

    // -----------------------------------------------------------------------
    // ReElectionEngine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_engine_empty_returns_none() {
        let local_id = ClockIdentity([0x01; 8]);
        let local = crate::ptp::dataset::DefaultDataSet::new(local_id);
        let engine = ReElectionEngine::new(Duration::from_secs(3));
        assert!(engine.elect_best_master(&local).is_none());
    }

    #[test]
    fn test_engine_single_better_candidate() {
        let local_id = ClockIdentity([0xFF; 8]); // high (worse) identity
        let local = crate::ptp::dataset::DefaultDataSet::new(local_id);
        let engine_timeout = Duration::from_secs(3);
        let mut engine = ReElectionEngine::new(engine_timeout);
        // Foreign master with lower priority1 → better than local.
        let ann = make_announce(100, 0x01);
        engine.update_foreign_master(ann, Instant::now());
        let best = engine.elect_best_master(&local);
        assert!(best.is_some(), "better candidate should be elected");
    }

    #[test]
    fn test_engine_selects_best_of_multiple() {
        let local_id = ClockIdentity([0xFF; 8]);
        let local = crate::ptp::dataset::DefaultDataSet::new(local_id);
        let mut engine = ReElectionEngine::new(Duration::from_secs(3));
        let now = Instant::now();
        // Candidate A: priority1=150
        engine.update_foreign_master(make_announce(150, 0x10), now);
        // Candidate B: priority1=100 (better)
        engine.update_foreign_master(make_announce(100, 0x20), now);
        // Candidate C: priority1=200 (worse than A)
        engine.update_foreign_master(make_announce(200, 0x30), now);

        let best = engine.elect_best_master(&local).expect("should find best");
        assert_eq!(
            best.last_announce.grandmaster_priority1, 100,
            "should elect lowest priority1"
        );
    }

    #[test]
    fn test_engine_remove_expired() {
        let mut engine = ReElectionEngine::new(Duration::from_millis(10));
        let base = Instant::now();
        engine.update_foreign_master(make_announce(128, 0x01), base);
        engine.update_foreign_master(make_announce(128, 0x02), base + Duration::from_millis(5));
        assert_eq!(engine.foreign_master_count(), 2);

        // Advance 15ms — first record expired, second still live.
        let removed = engine.remove_expired(base + Duration::from_millis(15));
        assert_eq!(removed.len(), 1);
        assert_eq!(engine.foreign_master_count(), 1);
    }

    #[test]
    fn test_engine_local_wins_against_worse_candidate() {
        // local has priority1=100 (very good).
        let local_id = ClockIdentity([0x01; 8]);
        let mut local = crate::ptp::dataset::DefaultDataSet::new(local_id);
        local.priority1 = 100;
        local.clock_quality.clock_class = 6;
        let mut engine = ReElectionEngine::new(Duration::from_secs(3));
        // Foreign master with priority1=200 (worse than local).
        engine.update_foreign_master(make_announce(200, 0x50), Instant::now());
        // No candidate is better than local → elect_best_master returns None.
        let best = engine.elect_best_master(&local);
        assert!(
            best.is_none(),
            "local clock wins, no foreign master should be elected"
        );
    }

    // -----------------------------------------------------------------------
    // BmcaSelector tests — IEEE 1588 grandmaster election
    // -----------------------------------------------------------------------

    fn make_ptp_clock_identity(
        priority1: u8,
        priority2: u8,
        clock_class: u8,
        id_byte: u8,
    ) -> super::PtpClockIdentity {
        super::PtpClockIdentity {
            clock_class,
            clock_accuracy: 0x20,
            offset_scaled_log_variance: 0x4000,
            priority1,
            priority2,
            identity: ClockIdentity([id_byte; 8]),
        }
    }

    #[test]
    fn test_bmca_selector_empty() {
        assert!(super::BmcaSelector::select_grandmaster(&[]).is_none());
    }

    #[test]
    fn test_bmca_selector_single_clock() {
        let clocks = vec![make_ptp_clock_identity(128, 128, 135, 0x01)];
        let gm = super::BmcaSelector::select_grandmaster(&clocks).expect("should elect one clock");
        assert_eq!(gm.identity.0[0], 0x01);
    }

    #[test]
    fn test_bmca_selector_lowest_priority1_wins() {
        // Three clocks with different priority1; lowest should win.
        let clocks = vec![
            make_ptp_clock_identity(200, 128, 135, 0x0A),
            make_ptp_clock_identity(100, 128, 135, 0x0B), // best
            make_ptp_clock_identity(150, 128, 135, 0x0C),
        ];
        let gm =
            super::BmcaSelector::select_grandmaster(&clocks).expect("should elect grandmaster");
        assert_eq!(
            gm.priority1, 100,
            "clock with lowest priority1 should be grandmaster"
        );
        assert_eq!(gm.identity.0[0], 0x0B);
    }

    #[test]
    fn test_bmca_selector_clock_class_tiebreaker() {
        // Same priority1; lower clock_class (= closer to primary ref) wins.
        let mut clocks = vec![
            make_ptp_clock_identity(128, 128, 135, 0x01),
            make_ptp_clock_identity(128, 128, 6, 0x02), // class 6 = stratum 1
            make_ptp_clock_identity(128, 128, 248, 0x03),
        ];
        // Adjust clock_class for first element explicitly.
        clocks[0].clock_class = 248;

        let gm =
            super::BmcaSelector::select_grandmaster(&clocks).expect("should elect grandmaster");
        assert_eq!(
            gm.clock_class, 6,
            "clock with lowest clock_class (6) should win"
        );
    }

    #[test]
    fn test_bmca_selector_priority2_tiebreaker() {
        // Same priority1, class, accuracy, variance; lowest priority2 wins.
        let clocks = vec![
            make_ptp_clock_identity(128, 200, 135, 0x10),
            make_ptp_clock_identity(128, 50, 135, 0x20), // lowest priority2
            make_ptp_clock_identity(128, 128, 135, 0x30),
        ];
        let gm =
            super::BmcaSelector::select_grandmaster(&clocks).expect("should elect grandmaster");
        assert_eq!(gm.priority2, 50, "clock with lowest priority2 should win");
    }

    #[test]
    fn test_bmca_selector_identity_final_tiebreaker() {
        // All parameters equal; lowest identity bytes break the tie.
        let clocks = vec![
            make_ptp_clock_identity(128, 128, 135, 0xFF),
            make_ptp_clock_identity(128, 128, 135, 0x01), // lowest identity
            make_ptp_clock_identity(128, 128, 135, 0x80),
        ];
        let gm =
            super::BmcaSelector::select_grandmaster(&clocks).expect("should elect grandmaster");
        assert_eq!(
            gm.identity.0[0], 0x01,
            "clock with lowest identity should win final tiebreak"
        );
    }
}

//! Multipath QUIC preview (draft-ietf-quic-multipath-08).
//!
//! This module provides the data structures and connection-level API for
//! tracking multiple simultaneous QUIC paths. The implementation follows the
//! emerging IETF draft (draft-ietf-quic-multipath) and RFC 9000 §9 (path
//! migration foundation).
//!
//! ## What is implemented
//!
//! * [`PathState`] — per-path RTT, congestion window, CID, validation status,
//!   and packet count tracked separately for each known path.
//! * [`MultipathState`] — collection of known paths with an "active path" index.
//! * Connection-level API:
//!   - [`Connection::multipath_state`] — inspect all known paths.
//!   - [`Connection::add_path`] — register a new candidate path with its CID.
//!   - [`Connection::set_preferred_path`] — mark a path as the preferred
//!     active path (packet sending migrates to it on the next `poll_transmit`).
//!   - [`Connection::path_count`] — number of known (validated or candidate) paths.
//!   - [`Connection::active_path_rtt`] — smoothed RTT of the current active path.
//!
//! * Per-path scheduling ([`PathScheduler`]) and per-path loss recovery
//!   ([`PathRecovery`]): each path owns a congestion controller and an RFC 9002
//!   RTT estimator, and acknowledgements, losses and ECN CE marks are routed to
//!   the path that carried the packet (recorded in
//!   its `SentPacket` record).
//! * Per-path anti-amplification ([`PathAmplification`]) for addresses adopted
//!   during migration (RFC 9000 §9.3).
//!
//! ## What is deferred
//!
//! Multipath-specific wire extensions from draft-ietf-quic-multipath — per-path
//! packet-number spaces, the PATH_ACK frame and multipath stream mapping — are
//! deferred pending stabilisation of the draft. The API surface is designed so
//! these additions slot in without breaking the existing implementation.
//!
//! ## Relationship to RFC 9000 §9
//!
//! RFC 9000 §9 (path migration) is already implemented in `keys_path.rs`:
//! `initiate_path_challenge`, `set_candidate_peer_addr`, `path_validated`, the
//! §8.2 validation timer (challenge retransmission on PTO and abandonment) and
//! the §9.5 connection-ID rotation that hands a migrating peer a fresh,
//! unlinkable batch of CIDs. This module extends that foundation to track
//! multiple simultaneous paths and exposes them through the multipath API; a
//! path abandoned by the validation timer is recorded here as
//! [`PathValidation::Failed`].

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use oxiquic_core::{ConnectionId, OxiQuicError};

use super::Connection;
use crate::bbr::RateSample;
use crate::cc_dispatch::CongestionController;
use crate::config::CongestionAlgorithm;
use crate::recovery::RttEstimator;

// ─────────────────────────────────────────────────────────────────────────────
// PathState
// ─────────────────────────────────────────────────────────────────────────────

/// Per-path anti-amplification allowance for an address adopted during
/// migration (RFC 9000 §9.3).
///
/// RFC 9000 §8.1 caps what a server may send to an *unvalidated* address at
/// three times what it has received from it. §9.3 extends that to every address
/// adopted mid-connection: a peer that moves to a new address (or an attacker
/// that spoofs one) restarts the budget, because the handshake proved ownership
/// of the *old* address only. The counters therefore live per path, not per
/// connection, and validating the path (a matching PATH_RESPONSE) lifts the cap.
///
/// The connection's initial path is *not* governed by this type — it uses the
/// connection-level counters seeded by the handshake — so [`PathState`] carries
/// this as an `Option` and `None` means "the connection-level limit applies".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PathAmplification {
    /// Bytes received from this address.
    pub bytes_recv: u64,
    /// Bytes sent to this address since the allowance was last reset.
    pub bytes_sent: u64,
    /// Whether this address has been validated (PATH_RESPONSE matched), which
    /// permanently lifts the limit for the path.
    pub validated: bool,
}

impl PathAmplification {
    /// Bytes that may still be sent to this address, or `None` once the address
    /// is validated and no limit applies.
    #[must_use]
    pub fn budget(&self) -> Option<u64> {
        if self.validated {
            return None;
        }
        Some(
            self.bytes_recv
                .saturating_mul(3)
                .saturating_sub(self.bytes_sent),
        )
    }

    /// Credit `bytes` received from this address against the allowance.
    pub fn on_received(&mut self, bytes: u64) {
        self.bytes_recv = self.bytes_recv.saturating_add(bytes);
    }

    /// Charge `bytes` sent to this address against the allowance. A no-op once
    /// the address is validated.
    pub fn on_sent(&mut self, bytes: u64) {
        if !self.validated {
            self.bytes_sent = self.bytes_sent.saturating_add(bytes);
        }
    }

    /// Record that the address has been validated, lifting the limit.
    pub fn mark_validated(&mut self) {
        self.validated = true;
        self.bytes_sent = 0;
    }
}

/// Validation state of a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathValidation {
    /// No PATH_CHALLENGE has been sent for this path yet.
    Unknown,
    /// PATH_CHALLENGE sent; waiting for PATH_RESPONSE.
    Pending,
    /// PATH_CHALLENGE acknowledged via PATH_RESPONSE.
    Validated,
    /// Path validation failed (no response within PTO budget).
    Failed,
}

impl std::fmt::Display for PathValidation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unknown => "unknown",
            Self::Pending => "pending",
            Self::Validated => "validated",
            Self::Failed => "failed",
        })
    }
}

/// Per-path state for a QUIC connection.
///
/// Each path is identified by (local_addr, remote_addr). The connection
/// maintains one primary path and zero or more candidate / secondary paths.
/// Each path tracks its own RTT estimate, congestion window hint, and
/// validation status independently.
#[derive(Debug, Clone)]
pub struct PathState {
    /// The local socket address for this path.
    pub local_addr: Option<SocketAddr>,
    /// The remote peer address for this path.
    pub remote_addr: SocketAddr,
    /// The connection ID used on this path (RFC 9000 §9.5 requires using a
    /// fresh CID when migrating to prevent linkability).
    pub connection_id: Option<ConnectionId>,
    /// Validation state.
    pub validation: PathValidation,
    /// Smoothed RTT estimate for this path (None until at least one RTT sample).
    pub smoothed_rtt: Option<Duration>,
    /// Minimum RTT observed on this path.
    pub min_rtt: Option<Duration>,
    /// Estimated available bandwidth (bytes/s), updated by ACK events.
    pub bandwidth_estimate: Option<u64>,
    /// Number of packets sent on this path.
    pub packets_sent: u64,
    /// Number of packets received on this path.
    pub packets_received: u64,
    /// Wall-clock time this path was first seen.
    pub first_seen: Instant,
    /// Wall-clock time this path was last used for sending.
    pub last_sent: Option<Instant>,
    /// Anti-amplification allowance for this address (RFC 9000 §9.3).
    ///
    /// `None` on the connection's initial path, whose allowance is the
    /// connection-level one seeded by the handshake; `Some` on every address
    /// adopted afterwards, which starts with a fresh, unvalidated 3× budget.
    pub amplification: Option<PathAmplification>,
}

impl PathState {
    /// Create a new path state for `remote_addr`.
    #[must_use]
    pub fn new(remote_addr: SocketAddr) -> Self {
        Self {
            local_addr: None,
            remote_addr,
            connection_id: None,
            validation: PathValidation::Unknown,
            smoothed_rtt: None,
            min_rtt: None,
            bandwidth_estimate: None,
            packets_sent: 0,
            packets_received: 0,
            first_seen: Instant::now(),
            last_sent: None,
            // Addresses are only reachable through `MultipathState::add_path`
            // (i.e. adopted after the handshake), so a fresh path starts with
            // its own unvalidated allowance; `MultipathState::new` clears this
            // for the initial path.
            amplification: Some(PathAmplification::default()),
        }
    }

    /// Whether this path may be promoted to the active (preferred) path.
    ///
    /// Only fully-validated paths are eligible for promotion. The initial path
    /// is created with [`PathValidation::Validated`] by [`MultipathState::new`].
    /// Candidate paths added via [`MultipathState::add_path`] start as
    /// [`PathValidation::Unknown`] and must be validated via PATH_CHALLENGE /
    /// PATH_RESPONSE before they can become the preferred path.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.validation == PathValidation::Validated
    }

    /// Update the RTT estimate with a new sample.
    ///
    /// Uses the RFC 9002 §5.3 EWMA formula: `smoothed_rtt = 7/8 * smoothed_rtt + 1/8 * sample`.
    pub fn update_rtt(&mut self, sample: Duration) {
        let updated = match self.smoothed_rtt {
            None => sample,
            Some(prev) => {
                // EWMA: 7/8 * prev + 1/8 * sample (integer arithmetic, µs).
                let prev_us = prev.as_micros() as u64;
                let sample_us = sample.as_micros() as u64;
                let new_us = (prev_us * 7 + sample_us) / 8;
                Duration::from_micros(new_us)
            }
        };
        self.smoothed_rtt = Some(updated);
        self.min_rtt = Some(match self.min_rtt {
            None => sample,
            Some(m) => m.min(sample),
        });
    }

    /// Record a packet sent on this path.
    pub fn record_sent(&mut self) {
        self.packets_sent += 1;
        self.last_sent = Some(Instant::now());
    }

    /// Record a packet received on this path.
    pub fn record_received(&mut self) {
        self.packets_received += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PathRecovery
// ─────────────────────────────────────────────────────────────────────────────

/// The loss-recovery state a single path owns: its own congestion controller
/// and its own RFC 9002 RTT estimator.
///
/// A QUIC connection that spreads packets over several paths cannot share one
/// controller between them: a 10 ms LAN path and a 200 ms cellular path feeding
/// one smoothed RTT inflates the PTO on the fast path and collapses it on the
/// slow one, and one congestion window sized for the aggregate over-sends on
/// whichever path is currently narrower. Each [`PathState`] therefore has a
/// matching `PathRecovery`, fed by the acknowledgements and losses of the
/// packets that were actually sent on that path (packets carry their path index
/// in their `SentPacket` record).
///
/// The connection-level controller remains the aggregate gate — it still sees
/// every acknowledgement and loss — so single-path connections behave exactly
/// as before.
pub struct PathRecovery {
    congestion: CongestionController,
    rtt: RttEstimator,
}

impl PathRecovery {
    /// Create per-path recovery state driven by `algorithm`.
    #[must_use]
    pub fn new(algorithm: CongestionAlgorithm) -> Self {
        Self {
            congestion: CongestionController::from_config(algorithm),
            rtt: RttEstimator::new(),
        }
    }

    /// This path's congestion window, in bytes.
    #[must_use]
    pub fn congestion_window(&self) -> u64 {
        self.congestion.congestion_window()
    }

    /// Bytes currently in flight on this path.
    #[must_use]
    pub fn bytes_in_flight(&self) -> u64 {
        self.congestion.bytes_in_flight()
    }

    /// Whether a packet of `bytes` fits in this path's congestion window.
    #[must_use]
    pub fn can_send(&self, bytes: usize) -> bool {
        self.congestion.can_send(bytes)
    }

    /// Whether this path has at least one RTT sample.
    #[must_use]
    pub fn have_rtt_sample(&self) -> bool {
        self.rtt.have_sample()
    }

    /// This path's smoothed RTT (RFC 9002 §5.3).
    #[must_use]
    pub fn smoothed_rtt(&self) -> Duration {
        self.rtt.smoothed_rtt()
    }

    /// The minimum RTT observed on this path.
    #[must_use]
    pub fn min_rtt(&self) -> Duration {
        self.rtt.min_rtt()
    }

    /// A bytes-per-second delivery-rate estimate for this path: the congestion
    /// window divided by the smoothed RTT.
    ///
    /// This is the classic `cwnd / RTT` approximation, not a measured delivery
    /// rate; it is used only to rank paths for
    /// [`PathScheduler::HighestBandwidth`]. Returns `None` until the path has
    /// an RTT sample.
    #[must_use]
    pub fn bandwidth_estimate(&self) -> Option<u64> {
        if !self.rtt.have_sample() {
            return None;
        }
        let srtt_us = self.rtt.smoothed_rtt().as_micros();
        if srtt_us == 0 {
            return None;
        }
        let bytes_per_second =
            u128::from(self.congestion.congestion_window()).saturating_mul(1_000_000) / srtt_us;
        Some(u64::try_from(bytes_per_second).unwrap_or(u64::MAX))
    }
}

impl std::fmt::Debug for PathRecovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathRecovery")
            .field("congestion_window", &self.congestion.congestion_window())
            .field("bytes_in_flight", &self.congestion.bytes_in_flight())
            .field("smoothed_rtt", &self.rtt.smoothed_rtt())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultipathState
// ─────────────────────────────────────────────────────────────────────────────

/// Path-selection policy for choosing which validated path carries the next
/// outgoing packet (draft-ietf-quic-multipath scheduling).
///
/// A scheduler only ever chooses among *validated* ([`PathValidation::Validated`])
/// paths; unvalidated candidate paths are never selected for data. With a single
/// path (the common case) every policy resolves to that path, so enabling a
/// policy is a no-op until additional paths are validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathScheduler {
    /// Always use the current active/preferred path (single-path behaviour).
    /// This is the default so existing single-path connections are unchanged.
    #[default]
    Active,
    /// Latency-optimal: pick the validated path with the lowest smoothed RTT.
    /// Paths without an RTT sample yet are deprioritised (treated as highest
    /// latency) so traffic prefers a measured path.
    LowestRtt,
    /// Throughput-optimal: pick the validated path with the highest estimated
    /// bandwidth. Paths without a bandwidth estimate are deprioritised.
    HighestBandwidth,
    /// Fairness: distribute packets round-robin across all validated paths.
    RoundRobin,
}

/// Collection of known paths for a QUIC connection.
///
/// At any given time, one path is designated the "active" path. By default all
/// packets are sent on this path; when a [`PathScheduler`] other than
/// [`PathScheduler::Active`] is selected, [`MultipathState::select_path`] picks
/// a validated path per the policy for each outgoing packet. The initial path
/// (index 0) is always the active path unless explicitly changed via
/// [`Connection::set_preferred_path`].
///
/// Capacity: up to 8 simultaneous paths (RFC 9000 §9 anti-amplification
/// considerations make more than 4-8 paths unusual in practice).
pub struct MultipathState {
    /// Known paths, in order of registration.
    paths: Vec<PathState>,
    /// Per-path congestion / RTT state, kept index-aligned with `paths`.
    recovery: Vec<PathRecovery>,
    /// Index of the currently active (preferred) path in `paths`.
    active_index: usize,
    /// The active path-selection policy.
    scheduler: PathScheduler,
    /// Rotating cursor for [`PathScheduler::RoundRobin`]; indexes into the list
    /// of validated paths, not directly into `paths`.
    round_robin_cursor: usize,
    /// Congestion-control algorithm new paths are created with, so every path
    /// runs the controller the connection was configured with.
    algorithm: CongestionAlgorithm,
}

impl MultipathState {
    const MAX_PATHS: usize = 8;

    /// Create a new multipath state with one initial path, using the default
    /// congestion-control algorithm for per-path recovery state.
    #[must_use]
    pub fn new(initial_remote: SocketAddr) -> Self {
        Self::new_with_algorithm(initial_remote, CongestionAlgorithm::default())
    }

    /// Create a new multipath state with one initial path whose per-path
    /// congestion controllers use `algorithm`.
    #[must_use]
    pub fn new_with_algorithm(initial_remote: SocketAddr, algorithm: CongestionAlgorithm) -> Self {
        let mut initial = PathState::new(initial_remote);
        // The initial path is implicitly validated by the handshake.
        initial.validation = PathValidation::Validated;
        // The initial path's anti-amplification allowance is the
        // connection-level one (RFC 9000 §8.1), not a per-path §9.3 budget.
        initial.amplification = None;
        Self {
            paths: vec![initial],
            recovery: vec![PathRecovery::new(algorithm)],
            active_index: 0,
            scheduler: PathScheduler::Active,
            round_robin_cursor: 0,
            algorithm,
        }
    }

    /// The per-path recovery state (congestion window, RTT) for `index`.
    #[must_use]
    pub fn path_recovery(&self, index: usize) -> Option<&PathRecovery> {
        self.recovery.get(index)
    }

    /// Whether a packet of `bytes` fits in path `index`'s congestion window.
    /// Unknown indices are permissive so the connection-level gate decides.
    #[must_use]
    pub fn path_can_send(&self, index: usize, bytes: usize) -> bool {
        self.recovery
            .get(index)
            .is_none_or(|rec| rec.can_send(bytes))
    }

    /// Charge `bytes` of an in-flight packet to path `index`.
    pub fn on_path_packet_sent(&mut self, index: usize, bytes: usize, now: Instant) {
        if let Some(rec) = self.recovery.get_mut(index) {
            let _ = rec.congestion.on_packet_sent(bytes, now);
        }
        if let Some(path) = self.paths.get_mut(index) {
            path.record_sent();
        }
    }

    /// Feed acknowledgements for packets that were sent on path `index`.
    ///
    /// `rtt_sample` is the raw RTT of the largest newly-acked packet when that
    /// packet belongs to this path, `None` otherwise; `ack_delay` and
    /// `max_ack_delay` are the RFC 9002 §5.3 correction terms.
    pub fn on_path_acked(
        &mut self,
        index: usize,
        acked: &[(usize, Instant, Option<RateSample>)],
        rtt_sample: Option<Duration>,
        ack_delay: Duration,
        max_ack_delay: Duration,
        now: Instant,
    ) {
        let Some(rec) = self.recovery.get_mut(index) else {
            return;
        };
        if let Some(sample) = rtt_sample {
            rec.rtt.update(sample, ack_delay, max_ack_delay);
        }
        if !acked.is_empty() {
            rec.congestion.on_packets_acked(acked, rtt_sample, now);
        }
        Self::refresh_path_metrics(&mut self.paths, &self.recovery, index);
    }

    /// Feed a loss event for packets that were sent on path `index`.
    pub fn on_path_lost(
        &mut self,
        index: usize,
        lost_bytes: u64,
        newest_lost: Instant,
        now: Instant,
    ) {
        if let Some(rec) = self.recovery.get_mut(index) {
            rec.congestion.on_packets_lost(lost_bytes, newest_lost, now);
        }
        Self::refresh_path_metrics(&mut self.paths, &self.recovery, index);
    }

    /// Feed an ECN congestion-experienced signal for path `index`.
    pub fn on_path_ecn_ce(&mut self, index: usize, ce_sent_time: Instant, now: Instant) {
        if let Some(rec) = self.recovery.get_mut(index) {
            rec.congestion.on_ecn_ce(ce_sent_time, now);
        }
        Self::refresh_path_metrics(&mut self.paths, &self.recovery, index);
    }

    /// Copy the live RTT / bandwidth figures from a path's recovery state into
    /// the public [`PathState`] fields the schedulers rank paths by.
    fn refresh_path_metrics(paths: &mut [PathState], recovery: &[PathRecovery], index: usize) {
        let (Some(path), Some(rec)) = (paths.get_mut(index), recovery.get(index)) else {
            return;
        };
        if rec.have_rtt_sample() {
            path.smoothed_rtt = Some(rec.smoothed_rtt());
            path.min_rtt = Some(rec.min_rtt());
        }
        path.bandwidth_estimate = rec.bandwidth_estimate();
    }

    /// The active path-selection policy.
    #[must_use]
    pub fn scheduler(&self) -> PathScheduler {
        self.scheduler
    }

    /// Set the path-selection policy used by [`Self::select_path`].
    pub fn set_scheduler(&mut self, scheduler: PathScheduler) {
        self.scheduler = scheduler;
    }

    /// The indices of all validated (usable) paths, in registration order.
    fn validated_indices(&self) -> Vec<usize> {
        self.paths
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_usable())
            .map(|(i, _)| i)
            .collect()
    }

    /// Choose the index of the path that should carry the next outgoing packet,
    /// applying the active [`PathScheduler`] over the set of validated paths.
    ///
    /// The returned index always refers to a validated path. If the active path
    /// is itself validated it is used as the fallback whenever the policy cannot
    /// discriminate (e.g. no RTT samples yet); otherwise the first validated
    /// path is the fallback. Returns [`Self::active_index`] if — pathologically —
    /// no path is validated, so the caller always has a usable target.
    pub fn select_path(&mut self) -> usize {
        let validated = self.validated_indices();
        if validated.is_empty() {
            return self.active_index;
        }
        // Single validated path: nothing to schedule.
        if validated.len() == 1 {
            return validated[0];
        }
        // Fallback index used when a policy cannot rank the candidates.
        let fallback = if validated.contains(&self.active_index) {
            self.active_index
        } else {
            validated[0]
        };
        match self.scheduler {
            PathScheduler::Active => {
                if validated.contains(&self.active_index) {
                    self.active_index
                } else {
                    fallback
                }
            }
            PathScheduler::LowestRtt => validated
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    let ra = self.paths[a].smoothed_rtt.unwrap_or(Duration::MAX);
                    let rb = self.paths[b].smoothed_rtt.unwrap_or(Duration::MAX);
                    ra.cmp(&rb)
                })
                .unwrap_or(fallback),
            PathScheduler::HighestBandwidth => validated
                .iter()
                .copied()
                .max_by_key(|&i| self.paths[i].bandwidth_estimate.unwrap_or(0))
                .unwrap_or(fallback),
            PathScheduler::RoundRobin => {
                // Advance the cursor over the validated set, wrapping around.
                let pick = validated[self.round_robin_cursor % validated.len()];
                self.round_robin_cursor = self.round_robin_cursor.wrapping_add(1);
                pick
            }
        }
    }

    /// The number of known paths (including candidates not yet validated).
    #[must_use]
    pub fn path_count(&self) -> usize {
        self.paths.len()
    }

    /// Immutable slice of all known paths.
    #[must_use]
    pub fn paths(&self) -> &[PathState] {
        &self.paths
    }

    /// Mutable reference to the active path.
    #[must_use]
    pub fn active_path_mut(&mut self) -> &mut PathState {
        &mut self.paths[self.active_index]
    }

    /// Immutable reference to the active path.
    #[must_use]
    pub fn active_path(&self) -> &PathState {
        &self.paths[self.active_index]
    }

    /// The index of the active path in `paths()`.
    #[must_use]
    pub fn active_index(&self) -> usize {
        self.active_index
    }

    /// Register a new candidate path.
    ///
    /// Returns `Ok(path_index)` on success.  Returns `Err` if the path is
    /// already known (same `remote_addr`) or if the capacity limit is reached.
    pub fn add_path(
        &mut self,
        remote_addr: SocketAddr,
        cid: Option<ConnectionId>,
    ) -> Result<usize, OxiQuicError> {
        // Reject duplicates.
        if self.paths.iter().any(|p| p.remote_addr == remote_addr) {
            return Err(OxiQuicError::Connection(format!(
                "path to {remote_addr} is already registered"
            )));
        }
        if self.paths.len() >= Self::MAX_PATHS {
            return Err(OxiQuicError::Connection(format!(
                "multipath capacity exceeded ({} paths max)",
                Self::MAX_PATHS
            )));
        }
        let mut state = PathState::new(remote_addr);
        state.connection_id = cid;
        let idx = self.paths.len();
        self.paths.push(state);
        // Every path gets its own congestion controller and RTT estimator; the
        // two vectors stay index-aligned.
        self.recovery.push(PathRecovery::new(self.algorithm));
        Ok(idx)
    }

    /// Remove a path by index. The active path cannot be removed.
    pub fn remove_path(&mut self, index: usize) -> Result<(), OxiQuicError> {
        if index >= self.paths.len() {
            return Err(OxiQuicError::Connection(format!(
                "path index {index} out of range"
            )));
        }
        if index == self.active_index {
            return Err(OxiQuicError::Connection(
                "cannot remove the active path".into(),
            ));
        }
        self.paths.remove(index);
        if index < self.recovery.len() {
            self.recovery.remove(index);
        }
        // Adjust active index if necessary.
        if self.active_index > index {
            self.active_index -= 1;
        }
        Ok(())
    }

    /// Promote `index` to the active path.
    pub fn set_preferred_path(&mut self, index: usize) -> Result<(), OxiQuicError> {
        if index >= self.paths.len() {
            return Err(OxiQuicError::Connection(format!(
                "path index {index} out of range"
            )));
        }
        if !self.paths[index].is_usable() {
            return Err(OxiQuicError::Connection(format!(
                "path {index} is not usable (validation: {})",
                self.paths[index].validation
            )));
        }
        self.active_index = index;
        Ok(())
    }

    /// Look up a path by remote address. Returns `None` if not found.
    #[must_use]
    pub fn path_by_addr(&self, addr: SocketAddr) -> Option<(usize, &PathState)> {
        self.paths
            .iter()
            .enumerate()
            .find(|(_, p)| p.remote_addr == addr)
    }

    /// Mutable lookup by remote address.
    #[must_use]
    pub fn path_by_addr_mut(&mut self, addr: SocketAddr) -> Option<(usize, &mut PathState)> {
        self.paths
            .iter_mut()
            .enumerate()
            .find(|(_, p)| p.remote_addr == addr)
    }

    /// Mark path at `index` as validated (PATH_CHALLENGE/PATH_RESPONSE complete).
    ///
    /// Also lifts the path's RFC 9000 §9.3 anti-amplification allowance: a
    /// validated address has proven it can receive, so the 3× cap no longer
    /// applies to it.
    pub fn mark_validated(&mut self, index: usize) {
        if let Some(p) = self.paths.get_mut(index) {
            p.validation = PathValidation::Validated;
            if let Some(amp) = p.amplification.as_mut() {
                amp.mark_validated();
            }
        }
    }

    /// Credit `bytes` received from `addr` against that path's
    /// anti-amplification allowance, if the address is a known path.
    ///
    /// Returns `true` if a path matched. Unknown addresses are deliberately
    /// ignored rather than registered: a spoofed source must never be able to
    /// mint path state (RFC 9000 §9.3.2).
    pub fn credit_received(&mut self, addr: SocketAddr, bytes: u64) -> bool {
        match self.path_by_addr_mut(addr) {
            Some((_, path)) => {
                path.record_received();
                if let Some(amp) = path.amplification.as_mut() {
                    amp.on_received(bytes);
                }
                true
            }
            None => false,
        }
    }

    /// Charge `bytes` sent to `addr` against that path's anti-amplification
    /// allowance, if the address is a known path.
    pub fn charge_sent(&mut self, addr: SocketAddr, bytes: u64) {
        if let Some((_, path)) = self.path_by_addr_mut(addr) {
            path.record_sent();
            if let Some(amp) = path.amplification.as_mut() {
                amp.on_sent(bytes);
            }
        }
    }

    /// The remaining anti-amplification allowance for `addr`.
    ///
    /// `None` means the address is not governed by a per-path allowance —
    /// either it is unknown, it is the initial path, or it has been validated —
    /// in which case the caller falls back to the connection-level limit.
    #[must_use]
    pub fn path_amplification_budget(&self, addr: SocketAddr) -> Option<u64> {
        self.path_by_addr(addr)
            .and_then(|(_, p)| p.amplification)
            .and_then(|amp| amp.budget())
    }

    /// Mark path at `index` as pending (PATH_CHALLENGE sent, waiting for response).
    pub fn mark_pending(&mut self, index: usize) {
        if let Some(p) = self.paths.get_mut(index) {
            p.validation = PathValidation::Pending;
        }
    }

    /// Mark path at `index` as failed (validation timed out).
    pub fn mark_failed(&mut self, index: usize) {
        if let Some(p) = self.paths.get_mut(index) {
            p.validation = PathValidation::Failed;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection impl
// ─────────────────────────────────────────────────────────────────────────────

impl Connection {
    /// Return a reference to the multipath state for this connection.
    ///
    /// The multipath state contains all known paths (primary + candidates),
    /// each with their own validation status, RTT estimate, and CID.
    ///
    /// Packet scheduling across validated paths is implemented via
    /// [`PathScheduler`] (see [`Connection::set_path_scheduler`]), and each path
    /// carries its own congestion controller and RTT estimator
    /// ([`MultipathState::path_recovery`]). The primary path (index 0) is always
    /// the current active path unless changed via
    /// [`Connection::set_preferred_path`] or by a completed migration.
    #[must_use]
    pub fn multipath_state(&self) -> &MultipathState {
        &self.multipath
    }

    /// The number of known paths (primary + candidates).
    #[must_use]
    pub fn path_count(&self) -> usize {
        self.multipath.path_count()
    }

    /// The smoothed RTT of the active (primary) path.
    ///
    /// Falls back to the connection-level RTT estimator (`self.rtt`) if the
    /// per-path estimate is not yet available.
    #[must_use]
    pub fn active_path_rtt(&self) -> Duration {
        self.multipath
            .active_path()
            .smoothed_rtt
            .unwrap_or_else(|| self.rtt.smoothed_rtt())
    }

    /// Register a new candidate path.
    ///
    /// Returns the path index on success. The new path starts with
    /// [`PathValidation::Unknown`]; call [`Connection::initiate_path_challenge`]
    /// to begin the validation handshake before promoting it to preferred.
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Connection`] if:
    /// * The path to `remote_addr` is already registered.
    /// * The multipath capacity limit (8 paths) is reached.
    pub fn add_path(
        &mut self,
        remote_addr: SocketAddr,
        cid: Option<ConnectionId>,
    ) -> Result<usize, OxiQuicError> {
        self.multipath.add_path(remote_addr, cid)
    }

    /// Promote path `index` to the active path.
    ///
    /// Only validated or initially-trusted paths may become active. Call
    /// [`Connection::initiate_path_challenge`] to validate a candidate path
    /// first.
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Connection`] if the index is out of range or
    /// the path is not usable.
    pub fn set_preferred_path(&mut self, index: usize) -> Result<(), OxiQuicError> {
        self.multipath.set_preferred_path(index)?;
        // Sync the active peer_addr with the new preferred path's remote addr.
        self.peer_addr = self.multipath.active_path().remote_addr;
        Ok(())
    }

    /// The active multipath scheduling policy.
    #[must_use]
    pub fn path_scheduler(&self) -> PathScheduler {
        self.multipath.scheduler()
    }

    /// Set the multipath scheduling policy used to choose which validated path
    /// carries outgoing packets (draft-ietf-quic-multipath).
    ///
    /// The default is [`PathScheduler::Active`], which preserves single-path
    /// behaviour. Any other policy takes effect once more than one path is
    /// validated.
    pub fn set_path_scheduler(&mut self, scheduler: PathScheduler) {
        self.multipath.set_scheduler(scheduler);
    }

    /// Run the scheduler to choose the path for the next outgoing packet and, if
    /// it differs from the current send target, retarget `peer_addr` to that
    /// path's remote address. Returns the chosen path index.
    ///
    /// This is a no-op (returns the active path) whenever fewer than two paths
    /// are validated, so single-path connections are unaffected. Called from the
    /// transmit path before building a datagram.
    /// Credit a received datagram of `bytes` from `src` against the
    /// anti-amplification allowance of the path that address belongs to
    /// (RFC 9000 §9.3).
    ///
    /// Datagrams from the connection's initial path, and from any address that
    /// is not a registered path, only feed the connection-level allowance that
    /// `handle_datagram_with_meta` already updates.
    pub(crate) fn credit_path_bytes_received(&mut self, src: SocketAddr, bytes: u64) {
        self.multipath.credit_received(src, bytes);
    }

    /// Charge an emitted datagram of `bytes` sent to `addr` against that path's
    /// anti-amplification allowance (RFC 9000 §9.3).
    pub(crate) fn charge_path_bytes_sent(&mut self, addr: SocketAddr, bytes: u64) {
        self.multipath.charge_sent(addr, bytes);
    }

    /// The address the next datagram should be sent to.
    ///
    /// RFC 9000 §9.3.3: while a PATH_CHALLENGE for a candidate address is
    /// queued, the probe must go to that *candidate* address — probing the old
    /// address would validate nothing. Every other datagram keeps going to the
    /// established peer address until the candidate is validated, so a spoofed
    /// address change cannot silently divert the connection's data.
    ///
    /// The candidate is only chosen when its own §9.3 allowance can actually
    /// admit a datagram. If the new address has sent too little for a probe to
    /// fit, the connection keeps transmitting normally on its validated path and
    /// the challenge stays queued until enough bytes arrive — retargeting to an
    /// address that cannot be written to would stall every space at once.
    #[must_use]
    pub(crate) fn send_target(&self) -> SocketAddr {
        match self.probe_target() {
            Some(candidate) => candidate,
            None => self.peer_addr,
        }
    }

    /// The candidate address a queued PATH_CHALLENGE should be sent to right
    /// now, or `None` when no probe is queued or the candidate's
    /// anti-amplification allowance cannot yet cover a datagram.
    #[must_use]
    pub(crate) fn probe_target(&self) -> Option<SocketAddr> {
        if !self.pending_path_challenge_send {
            return None;
        }
        let candidate = self.candidate_peer_addr?;
        let affordable = self
            .multipath
            .path_amplification_budget(candidate)
            .is_none_or(|budget| budget >= super::keys_path::MIN_AMPLIFICATION_HEADROOM);
        affordable.then_some(candidate)
    }

    /// Register `addr` as a candidate path so it gets its own RFC 9000 §9.3
    /// anti-amplification allowance, returning its path index.
    ///
    /// Idempotent: an address that is already a known path keeps its existing
    /// counters. Returns `None` when the multipath table is full, in which case
    /// the address stays governed by the connection-level allowance.
    pub(crate) fn register_candidate_path(&mut self, addr: SocketAddr) -> Option<usize> {
        if let Some((idx, _)) = self.multipath.path_by_addr(addr) {
            return Some(idx);
        }
        let idx = self.multipath.add_path(addr, None).ok()?;
        self.multipath.mark_pending(idx);
        Some(idx)
    }

    /// Mark the path to `addr` validated (RFC 9000 §9.3) and make it the active
    /// path, because the connection has just migrated onto it.
    ///
    /// Validation lifts the path's anti-amplification allowance; promoting it
    /// keeps the scheduler in step with `peer_addr`, so the next
    /// `schedule_send_path` does not drag the connection back to the address it
    /// just migrated away from.
    pub(crate) fn mark_path_addr_validated(&mut self, addr: SocketAddr) {
        if let Some((idx, _)) = self.multipath.path_by_addr(addr) {
            self.multipath.mark_validated(idx);
            // Only fails if the index is out of range or the path is unusable,
            // neither of which can hold immediately after `mark_validated`.
            let _ = self.multipath.set_preferred_path(idx);
        }
    }

    pub(crate) fn schedule_send_path(&mut self) -> usize {
        // Cheap fast-path: with 0 or 1 additional paths there is nothing to
        // schedule, so avoid touching the scheduler at all.
        if self.multipath.path_count() < 2 {
            return self.multipath.active_index();
        }
        let idx = self.multipath.select_path();
        if let Some(path) = self.multipath.paths().get(idx) {
            let addr = path.remote_addr;
            if addr != self.peer_addr {
                self.peer_addr = addr;
            }
        }
        idx
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(port: u16) -> SocketAddr {
        format!("127.0.0.1:{port}").parse().expect("parse addr")
    }

    #[test]
    fn multipath_state_initial_path_is_validated() {
        let mp = MultipathState::new(addr(8000));
        assert_eq!(mp.path_count(), 1);
        assert_eq!(mp.active_index(), 0);
        assert_eq!(mp.active_path().validation, PathValidation::Validated);
    }

    #[test]
    fn multipath_add_path_registers_candidate() {
        let mut mp = MultipathState::new(addr(8000));
        let idx = mp.add_path(addr(9000), None).expect("add path");
        assert_eq!(idx, 1);
        assert_eq!(mp.path_count(), 2);
        assert_eq!(mp.paths()[1].validation, PathValidation::Unknown);
    }

    #[test]
    fn multipath_add_duplicate_path_returns_error() {
        let mut mp = MultipathState::new(addr(8000));
        let result = mp.add_path(addr(8000), None);
        assert!(result.is_err());
    }

    #[test]
    fn multipath_set_preferred_path_rejects_unknown_path() {
        let mut mp = MultipathState::new(addr(8000));
        mp.add_path(addr(9000), None).expect("add path");
        // Path at index 1 is Unknown — cannot be set as preferred.
        let result = mp.set_preferred_path(1);
        assert!(result.is_err());
    }

    #[test]
    fn multipath_set_preferred_path_after_validation() {
        let mut mp = MultipathState::new(addr(8000));
        let idx = mp.add_path(addr(9000), None).expect("add path");
        mp.mark_validated(idx);
        mp.set_preferred_path(idx).expect("promote validated path");
        assert_eq!(mp.active_index(), 1);
        assert_eq!(mp.active_path().remote_addr, addr(9000));
    }

    #[test]
    fn multipath_capacity_limit_enforced() {
        let mut mp = MultipathState::new(addr(8000));
        for port in 9001..=9007 {
            mp.add_path(addr(port), None).expect("add path");
        }
        assert_eq!(mp.path_count(), 8);
        // The 9th add should fail.
        let result = mp.add_path(addr(9999), None);
        assert!(result.is_err());
    }

    #[test]
    fn multipath_remove_non_active_path() {
        let mut mp = MultipathState::new(addr(8000));
        let idx = mp.add_path(addr(9000), None).expect("add path");
        assert_eq!(mp.path_count(), 2);
        mp.remove_path(idx).expect("remove non-active path");
        assert_eq!(mp.path_count(), 1);
    }

    #[test]
    fn multipath_cannot_remove_active_path() {
        let mut mp = MultipathState::new(addr(8000));
        let result = mp.remove_path(0); // active path
        assert!(result.is_err());
    }

    #[test]
    fn path_state_rtt_update_converges() {
        let mut ps = PathState::new(addr(8000));
        assert!(ps.smoothed_rtt.is_none());

        ps.update_rtt(Duration::from_millis(10));
        assert_eq!(ps.smoothed_rtt, Some(Duration::from_millis(10)));
        assert_eq!(ps.min_rtt, Some(Duration::from_millis(10)));

        ps.update_rtt(Duration::from_millis(20));
        // EWMA: (10 * 7 + 20) / 8 = 90/8 = 11ms (integer µs: (10_000*7+20_000)/8 = 11_250µs)
        let expected = Duration::from_micros((10_000u64 * 7 + 20_000) / 8);
        assert_eq!(ps.smoothed_rtt, Some(expected));
        // min_rtt stays at 10ms
        assert_eq!(ps.min_rtt, Some(Duration::from_millis(10)));
    }

    #[test]
    fn path_state_record_send_recv_counts() {
        let mut ps = PathState::new(addr(8000));
        assert_eq!(ps.packets_sent, 0);
        assert_eq!(ps.packets_received, 0);

        ps.record_sent();
        ps.record_sent();
        ps.record_received();

        assert_eq!(ps.packets_sent, 2);
        assert_eq!(ps.packets_received, 1);
        assert!(ps.last_sent.is_some());
    }

    #[test]
    fn path_state_is_usable_only_for_validated() {
        let mut ps = PathState::new(addr(8000));
        // New paths start Unknown — not usable for promotion.
        assert!(!ps.is_usable());

        ps.validation = PathValidation::Validated;
        assert!(ps.is_usable());

        ps.validation = PathValidation::Pending;
        assert!(!ps.is_usable());

        ps.validation = PathValidation::Failed;
        assert!(!ps.is_usable());
    }

    #[test]
    fn path_by_addr_lookup() {
        let mut mp = MultipathState::new(addr(8000));
        mp.add_path(addr(9000), None).expect("add");
        let found = mp.path_by_addr(addr(9000));
        assert!(found.is_some());
        let (idx, _) = found.expect("found path");
        assert_eq!(idx, 1);

        assert!(mp.path_by_addr(addr(9999)).is_none());
    }

    // ── Scheduler ─────────────────────────────────────────────────────────────

    /// Register a second, validated path with the given RTT and bandwidth and
    /// return its index.
    fn add_validated(
        mp: &mut MultipathState,
        port: u16,
        rtt: Option<Duration>,
        bw: Option<u64>,
    ) -> usize {
        let idx = mp.add_path(addr(port), None).expect("add path");
        mp.mark_validated(idx);
        if let Some(r) = rtt {
            mp.paths[idx].update_rtt(r);
        }
        mp.paths[idx].bandwidth_estimate = bw;
        idx
    }

    #[test]
    fn scheduler_default_is_active() {
        let mp = MultipathState::new(addr(8000));
        assert_eq!(mp.scheduler(), PathScheduler::Active);
    }

    #[test]
    fn select_path_single_path_returns_active() {
        let mut mp = MultipathState::new(addr(8000));
        // Only the initial (validated) path exists.
        assert_eq!(mp.select_path(), 0);
        mp.set_scheduler(PathScheduler::LowestRtt);
        assert_eq!(mp.select_path(), 0);
    }

    #[test]
    fn select_path_ignores_unvalidated_candidates() {
        let mut mp = MultipathState::new(addr(8000));
        // Candidate path is Unknown → never selected.
        mp.add_path(addr(9000), None).expect("add");
        mp.set_scheduler(PathScheduler::RoundRobin);
        // Only path 0 is validated, so it is always chosen.
        assert_eq!(mp.select_path(), 0);
        assert_eq!(mp.select_path(), 0);
    }

    #[test]
    fn lowest_rtt_picks_fastest_validated_path() {
        let mut mp = MultipathState::new(addr(8000));
        // Give the initial path a slow RTT.
        mp.paths[0].update_rtt(Duration::from_millis(100));
        let fast = add_validated(&mut mp, 9000, Some(Duration::from_millis(10)), None);
        let _slow = add_validated(&mut mp, 9001, Some(Duration::from_millis(50)), None);
        mp.set_scheduler(PathScheduler::LowestRtt);
        assert_eq!(mp.select_path(), fast);
    }

    #[test]
    fn lowest_rtt_deprioritises_unmeasured_path() {
        let mut mp = MultipathState::new(addr(8000));
        mp.paths[0].update_rtt(Duration::from_millis(30));
        // A validated path with no RTT sample must not be preferred over a
        // measured one.
        let _unmeasured = add_validated(&mut mp, 9000, None, None);
        mp.set_scheduler(PathScheduler::LowestRtt);
        assert_eq!(mp.select_path(), 0);
    }

    #[test]
    fn highest_bandwidth_picks_fattest_pipe() {
        let mut mp = MultipathState::new(addr(8000));
        mp.paths[0].bandwidth_estimate = Some(1_000);
        let fat = add_validated(&mut mp, 9000, None, Some(10_000));
        let _thin = add_validated(&mut mp, 9001, None, Some(5_000));
        mp.set_scheduler(PathScheduler::HighestBandwidth);
        assert_eq!(mp.select_path(), fat);
    }

    #[test]
    fn round_robin_cycles_over_validated_paths() {
        let mut mp = MultipathState::new(addr(8000));
        let p1 = add_validated(&mut mp, 9000, None, None);
        let p2 = add_validated(&mut mp, 9001, None, None);
        mp.set_scheduler(PathScheduler::RoundRobin);
        // Three validated paths: 0, p1, p2. The cursor cycles through all of
        // them and wraps.
        let seq: Vec<usize> = (0..6).map(|_| mp.select_path()).collect();
        assert_eq!(seq, vec![0, p1, p2, 0, p1, p2]);
    }

    #[test]
    fn active_policy_prefers_active_index() {
        let mut mp = MultipathState::new(addr(8000));
        let p1 = add_validated(&mut mp, 9000, Some(Duration::from_millis(1)), None);
        // Promote p1 to active; the Active scheduler must return it even though
        // path 0 also exists.
        mp.set_preferred_path(p1).expect("promote");
        mp.set_scheduler(PathScheduler::Active);
        assert_eq!(mp.select_path(), p1);
    }

    // ─── Per-path congestion / RTT state ──────────────────────────────────

    /// Feed `count` acknowledgements of `bytes`-sized packets, each with an RTT
    /// of `rtt`, to path `index`.
    fn ack_path(mp: &mut MultipathState, index: usize, count: usize, bytes: usize, rtt: Duration) {
        let now = Instant::now();
        let acked: Vec<(usize, Instant, Option<RateSample>)> =
            (0..count).map(|_| (bytes, now - rtt, None)).collect();
        mp.on_path_acked(
            index,
            &acked,
            Some(rtt),
            Duration::ZERO,
            Duration::ZERO,
            now,
        );
    }

    /// RTT samples from different paths must not be mixed: a 10 ms path and a
    /// 200 ms path keep separate smoothed RTTs.
    #[test]
    fn rtt_samples_are_not_shared_between_paths() {
        let mut mp = MultipathState::new(addr(8000));
        let slow = add_validated(&mut mp, 9000, None, None);

        for _ in 0..8 {
            ack_path(&mut mp, 0, 1, 1200, Duration::from_millis(10));
            ack_path(&mut mp, slow, 1, 1200, Duration::from_millis(200));
        }

        let fast_rtt = mp.path_recovery(0).expect("path 0").smoothed_rtt();
        let slow_rtt = mp.path_recovery(slow).expect("slow path").smoothed_rtt();
        assert!(
            fast_rtt < Duration::from_millis(30),
            "the fast path must keep a fast smoothed RTT, got {fast_rtt:?}"
        );
        assert!(
            slow_rtt > Duration::from_millis(150),
            "the slow path must keep a slow smoothed RTT, got {slow_rtt:?}"
        );
        // The public PathState mirror the schedulers rank by is kept in step.
        assert_eq!(mp.paths()[0].smoothed_rtt, Some(fast_rtt));
        assert_eq!(mp.paths()[slow].smoothed_rtt, Some(slow_rtt));
    }

    /// With per-path RTT state the LowestRtt scheduler picks the measured-fast
    /// path, which is the whole point of giving each path its own estimator.
    #[test]
    fn lowest_rtt_scheduler_follows_per_path_measurements() {
        let mut mp = MultipathState::new(addr(8000));
        let fast = add_validated(&mut mp, 9000, None, None);
        for _ in 0..8 {
            ack_path(&mut mp, 0, 1, 1200, Duration::from_millis(200));
            ack_path(&mut mp, fast, 1, 1200, Duration::from_millis(5));
        }
        mp.set_scheduler(PathScheduler::LowestRtt);
        assert_eq!(mp.select_path(), fast);
    }

    /// A loss on one path shrinks only that path's congestion window.
    #[test]
    fn loss_on_one_path_does_not_shrink_the_other() {
        let mut mp = MultipathState::new(addr(8000));
        let other = add_validated(&mut mp, 9000, None, None);
        let now = Instant::now();

        // Grow both windows with a few acknowledgements.
        for _ in 0..10 {
            ack_path(&mut mp, 0, 1, 1200, Duration::from_millis(20));
            ack_path(&mut mp, other, 1, 1200, Duration::from_millis(20));
        }
        let before_0 = mp.path_recovery(0).expect("path 0").congestion_window();
        let before_1 = mp
            .path_recovery(other)
            .expect("other path")
            .congestion_window();

        mp.on_path_lost(other, 1200, now, now);

        assert_eq!(
            mp.path_recovery(0).expect("path 0").congestion_window(),
            before_0,
            "a loss on another path must not touch this window"
        );
        assert!(
            mp.path_recovery(other)
                .expect("other path")
                .congestion_window()
                < before_1,
            "the lossy path must reduce its own window"
        );
    }

    /// In-flight bytes are tracked per path, so a path is gated by its own
    /// congestion window.
    #[test]
    fn in_flight_bytes_are_tracked_per_path() {
        let mut mp = MultipathState::new(addr(8000));
        let other = add_validated(&mut mp, 9000, None, None);
        let now = Instant::now();

        mp.on_path_packet_sent(0, 1200, now);
        mp.on_path_packet_sent(0, 1200, now);
        mp.on_path_packet_sent(other, 1200, now);

        assert_eq!(
            mp.path_recovery(0).expect("path 0").bytes_in_flight(),
            2400,
            "path 0 carried two packets"
        );
        assert_eq!(
            mp.path_recovery(other)
                .expect("other path")
                .bytes_in_flight(),
            1200,
            "the other path carried one"
        );
        assert_eq!(mp.paths()[0].packets_sent, 2);
        assert_eq!(mp.paths()[other].packets_sent, 1);
    }

    /// The bandwidth estimate the HighestBandwidth scheduler ranks by is
    /// derived from each path's own window and RTT, and is `None` until the
    /// path has an RTT sample (never a fabricated zero).
    #[test]
    fn bandwidth_estimate_requires_a_measurement() {
        let mut mp = MultipathState::new(addr(8000));
        assert_eq!(
            mp.path_recovery(0).expect("path 0").bandwidth_estimate(),
            None,
            "no RTT sample yet means no estimate"
        );
        ack_path(&mut mp, 0, 1, 1200, Duration::from_millis(10));
        let estimate = mp
            .path_recovery(0)
            .expect("path 0")
            .bandwidth_estimate()
            .expect("estimate after a sample");
        assert!(estimate > 0);
        assert_eq!(mp.paths()[0].bandwidth_estimate, Some(estimate));
    }
}

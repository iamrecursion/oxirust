//! Key update (RFC 9001 §6) and path migration (RFC 9000 §9) APIs.
//!
//! Key update: `initiate_key_update`, `initiate_key_update_now`,
//! `perform_key_update`, `key_update_count`.
//!
//! Path migration: `initiate_path_challenge`, `set_candidate_peer_addr`,
//! `path_validated`, the RFC 9000 §8.2 path-validation timer
//! ([`PathChallengeTimer`]) and the §9.5 connection-ID rotation that gives a
//! migrating peer fresh, unlinkable CIDs to move to.

use std::time::{Duration, Instant};

use oxiquic_core::OxiQuicError;

use crate::recovery::RttEstimator;

use super::Connection;

/// Upper bound on the PATH_CHALLENGE nonces retained for one validation
/// attempt.
///
/// The abandon deadline of RFC 9000 §8.2.4 (three times the path PTO) already
/// caps the number of retransmissions at three or four, because each attempt
/// waits twice as long as the last. The explicit bound keeps the retained set
/// finite even if a caller drives the timer with a pathological clock.
const MAX_PATH_CHALLENGE_ATTEMPTS: usize = 8;

/// State backing the RFC 9000 §8.2 path-validation timer.
///
/// A PATH_CHALLENGE is not repaired by the normal loss-recovery machinery:
/// §13.3 does not list it among the frames whose *contents* are retransmitted,
/// because §8.2.1 requires "unpredictable data in every PATH_CHALLENGE frame".
/// A lost challenge therefore has to be replaced by a **new** challenge on a
/// timer, or path validation stalls forever with no way out.
///
/// The timer deliberately lives outside `loss_timer`: the loss-detection timer
/// is disarmed whenever the connection is anti-amplification blocked
/// (RFC 9002 §6.2.2.1), which is exactly the situation a path probe has to
/// survive.
#[derive(Debug, Default)]
pub(crate) struct PathChallengeTimer {
    /// Every nonce sent for the current validation attempt, oldest first.
    ///
    /// RFC 9000 §8.2.3: an endpoint that has sent several challenges accepts a
    /// PATH_RESPONSE matching *any* of them, since it cannot tell which
    /// challenge the peer saw first.
    pub(crate) history: Vec<[u8; 8]>,
    /// Number of PATH_CHALLENGE frames actually placed in a packet.
    pub(crate) attempts: u32,
    /// When the next challenge is due (§8.2.1: no more frequently than an
    /// Initial packet would be sent, i.e. a PTO with exponential backoff).
    pub(crate) retransmit_at: Option<Instant>,
    /// When validation is given up (§8.2.4).
    pub(crate) abandon_at: Option<Instant>,
    /// Set by the packet builder when a challenge was written, so that
    /// `poll_transmit` can arm the timer with its own notion of `now` instead
    /// of the frame-filling helpers having to thread a clock through.
    pub(crate) just_sent: bool,
    /// Whether the most recent validation attempt was abandoned (§8.2.4).
    pub(crate) failed: bool,
}

impl PathChallengeTimer {
    /// Clear all state for a fresh validation attempt.
    fn reset(&mut self) {
        self.history.clear();
        self.attempts = 0;
        self.retransmit_at = None;
        self.abandon_at = None;
        self.just_sent = false;
        self.failed = false;
    }

    /// Stop the timer, keeping [`Self::failed`] untouched.
    fn disarm(&mut self) {
        self.retransmit_at = None;
        self.abandon_at = None;
        self.just_sent = false;
    }

    /// Whether `data` answers any challenge sent in this attempt.
    fn matches(&self, data: [u8; 8]) -> bool {
        self.history.contains(&data)
    }
}

impl Connection {
    /// Rotate to the next key epoch after a successful key-update decryption.
    ///
    /// * Promotes `next_1rtt_keys` into `application` (packet keys only;
    ///   header protection keys are unchanged per RFC 9001 §6).
    /// * Saves the old remote packet key in `prev_1rtt_keys` for 3 PTO (§6.6).
    /// * Derives new `next_1rtt_keys` from the advanced secrets.
    /// * Flips `key_phase` and sets `key_update_received`.
    pub(super) fn perform_key_update(&mut self, now: Instant) {
        // Retirement deadline: 3 PTO (RFC 9001 §6.6).
        let pto_base = self.rtt.pto_base(self.peer_max_ack_delay);
        let retire_after = now + pto_base * 3;

        if let Some(ref mut secrets) = self.one_rtt_secrets {
            // Advance the secrets ratchet to derive the generation after next.
            let new_next_keys = secrets.next_packet_keys();
            if let Some(next_epoch) = self.next_1rtt_keys.take() {
                if let Some(old_remote) = self.application.rotate_to_next_epoch(next_epoch) {
                    self.prev_1rtt_keys = Some((old_remote, retire_after));
                }
                self.next_1rtt_keys = Some(new_next_keys);
            }
        }

        self.key_phase = !self.key_phase;
        self.key_update_received = true;
        self.key_update_count += 1;
        self.key_update_cooldown = Some(now + pto_base * 3);
    }

    /// Initiate a key update on the next outgoing 1-RTT packet.
    ///
    /// Per RFC 9001 §6.5, a key update MUST NOT be initiated within 3 PTO of
    /// the previous update. If the cooldown has not elapsed this is a no-op.
    /// Returns `true` if the update was scheduled, `false` if it was refused
    /// due to the cooldown.
    ///
    /// `now` is the caller's notion of the current time, used to evaluate the
    /// 3-PTO cooldown.  Pass a consistent clock value (the same one you pass to
    /// `poll_transmit` / `handle_datagram`) so that logical-clock tests work
    /// correctly without relying on wall-clock advancement.
    pub fn initiate_key_update(&mut self, now: Instant) -> bool {
        if let Some(cooldown) = self.key_update_cooldown {
            if now < cooldown {
                return false;
            }
        }
        if !self.one_rtt_ready || self.next_1rtt_keys.is_none() {
            return false; // no 1-RTT keys yet
        }
        self.key_update_pending = true;
        true
    }

    /// Actually perform the locally-initiated key update at `now`.
    ///
    /// Called from `write_space_packet` just before the first outgoing packet
    /// of the new epoch.
    pub(super) fn initiate_key_update_now(&mut self, now: Instant) {
        self.key_update_pending = false;
        if let Some(ref mut secrets) = self.one_rtt_secrets {
            let new_next_keys = secrets.next_packet_keys();
            if let Some(next_epoch) = self.next_1rtt_keys.take() {
                let pto_base = self.rtt.pto_base(self.peer_max_ack_delay);
                let retire_after = now + pto_base * 3;
                if let Some(old_remote) = self.application.rotate_to_next_epoch(next_epoch) {
                    self.prev_1rtt_keys = Some((old_remote, retire_after));
                }
                self.next_1rtt_keys = Some(new_next_keys);
            }
        }
        self.key_phase = !self.key_phase;
        self.key_update_count += 1;
        let pto_base = self.rtt.pto_base(self.peer_max_ack_delay);
        self.key_update_cooldown = Some(now + pto_base * 3);
    }

    /// The number of completed key updates (including both locally- and
    /// peer-initiated).  For test observability.
    #[must_use]
    pub fn key_update_count(&self) -> u64 {
        self.key_update_count
    }

    // ─── Path migration API (RFC 9000 §9) ────────────────────────────────────

    /// Begin a path challenge toward the current (or candidate) peer address.
    ///
    /// Generates 8 cryptographically random bytes and queues a `PATH_CHALLENGE`
    /// frame for the next outgoing 1-RTT packet (RFC 9000 §9.1, §19.17).
    ///
    /// Returns `Ok(())` on success.  Fails with [`OxiQuicError::Connection`] if
    /// the secure RNG cannot produce random bytes, or if the 1-RTT keys are not
    /// yet available (challenge frames are 1-RTT-only).
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Connection`] when:
    /// * The secure RNG fails (extremely unlikely, indicates OS-level failure).
    /// * The connection has not yet completed the handshake.
    pub fn initiate_path_challenge(&mut self) -> Result<(), OxiQuicError> {
        self.initiate_path_challenge_at(Instant::now())
    }

    /// [`Self::initiate_path_challenge`] with an explicit clock value.
    ///
    /// `now` seeds the RFC 9000 §8.2 validation timer: the retransmission
    /// schedule of §8.2.1 and the abandon deadline of §8.2.4. Pass the same
    /// clock value used for `poll_transmit` / `handle_timeout` so logical-clock
    /// tests behave deterministically.
    ///
    /// # Errors
    /// As [`Self::initiate_path_challenge`].
    pub fn initiate_path_challenge_at(&mut self, now: Instant) -> Result<(), OxiQuicError> {
        if !self.one_rtt_ready {
            return Err(OxiQuicError::Connection(
                "cannot send PATH_CHALLENGE before 1-RTT keys are ready".into(),
            ));
        }
        self.path_challenge.reset();
        let nonce = self.fresh_path_challenge_nonce()?;
        self.pending_path_challenge = Some(nonce);
        self.pending_path_challenge_send = true;
        self.path_validated = false;
        // RFC 9000 §8.2.4: give up after three times the larger of the current
        // PTO and the PTO of a path with no RTT sample. The deadline is armed
        // now rather than on first transmission so that a candidate whose
        // §9.3 allowance never admits a probe is abandoned too, instead of
        // pinning the candidate address forever.
        self.path_challenge.abandon_at = Some(now + self.path_validation_pto() * 3);
        Ok(())
    }

    /// Draw a fresh 8-byte challenge nonce and record it as outstanding.
    ///
    /// RFC 9000 §8.2.1 requires unpredictable data in *every* PATH_CHALLENGE,
    /// so a retransmission carries new bytes rather than repeating the old
    /// ones; §8.2.3 then allows a PATH_RESPONSE to any of them to validate the
    /// path, which is why the previous nonces are retained.
    fn fresh_path_challenge_nonce(&mut self) -> Result<[u8; 8], OxiQuicError> {
        let mut nonce = [0u8; 8];
        self.secure_random
            .fill(&mut nonce)
            .map_err(|_| OxiQuicError::Connection("secure RNG failed".into()))?;
        if self.path_challenge.history.len() >= MAX_PATH_CHALLENGE_ATTEMPTS {
            self.path_challenge.history.remove(0);
        }
        self.path_challenge.history.push(nonce);
        Ok(nonce)
    }

    /// The PTO that governs path validation (RFC 9000 §8.2.4): the larger of
    /// the connection's current PTO and the PTO of a path with no RTT sample
    /// (which RFC 9002 derives from `kInitialRtt`).
    fn path_validation_pto(&self) -> Duration {
        let current = self.rtt.pto_base(self.peer_max_ack_delay);
        let new_path = RttEstimator::new().pto_base(self.peer_max_ack_delay);
        current.max(new_path)
    }

    /// A PATH_CHALLENGE frame was placed in the packet just built at `now`:
    /// count the attempt and schedule the next one (RFC 9000 §8.2.1 — no more
    /// frequently than an Initial packet, i.e. a PTO with exponential backoff).
    pub(super) fn on_path_challenge_sent(&mut self, now: Instant) {
        if !self.path_challenge.just_sent {
            return;
        }
        self.path_challenge.just_sent = false;
        self.path_challenge.attempts = self.path_challenge.attempts.saturating_add(1);
        let backoff = 1u32 << self.path_challenge.attempts.saturating_sub(1).min(16);
        self.path_challenge.retransmit_at =
            Some(now + self.path_validation_pto().saturating_mul(backoff));
    }

    /// Fire the path-validation timer at `now`: retransmit a fresh challenge,
    /// or abandon validation once the §8.2.4 deadline has passed.
    pub(super) fn on_path_challenge_timeout(&mut self, now: Instant) {
        if self.pending_path_challenge.is_none() {
            return; // no validation in progress
        }
        if let Some(abandon_at) = self.path_challenge.abandon_at {
            if now >= abandon_at {
                self.abandon_path_validation();
                return;
            }
        }
        if let Some(retransmit_at) = self.path_challenge.retransmit_at {
            if now >= retransmit_at {
                // A queued-but-unsent challenge means the previous attempt
                // never left (no send budget); do not burn a nonce on it.
                if !self.pending_path_challenge_send {
                    // On an RNG failure the previous nonce is re-sent rather
                    // than validation being dropped on the floor; the abandon
                    // deadline still bounds the wait either way.
                    if let Ok(nonce) = self.fresh_path_challenge_nonce() {
                        self.pending_path_challenge = Some(nonce);
                    }
                    self.pending_path_challenge_send = true;
                }
                // Re-armed for real on the next successful transmission; until
                // then the abandon deadline is the only thing that can fire.
                self.path_challenge.retransmit_at = None;
            }
        }
    }

    /// Give up on the path being validated (RFC 9000 §8.2.4).
    ///
    /// The candidate address is dropped and its path marked
    /// [`super::multipath::PathValidation::Failed`]; the connection keeps using
    /// the address it is already established on.
    fn abandon_path_validation(&mut self) {
        self.pending_path_challenge = None;
        self.pending_path_challenge_send = false;
        self.path_challenge.disarm();
        self.path_challenge.failed = true;
        if let Some(addr) = self.candidate_peer_addr.take() {
            if let Some((idx, _)) = self.multipath.path_by_addr(addr) {
                self.multipath.mark_failed(idx);
            }
        }
    }

    /// Process a received PATH_RESPONSE (RFC 9000 §8.2.3), returning `true`
    /// when it answers one of our outstanding challenges.
    pub(super) fn accept_path_response(&mut self, data: [u8; 8]) -> bool {
        if !self.path_challenge.matches(data) {
            // Mismatched PATH_RESPONSE: silently ignored per RFC 9000 §19.18.
            return false;
        }
        self.path_validated = true;
        self.pending_path_challenge = None;
        self.pending_path_challenge_send = false;
        self.path_challenge.disarm();
        self.path_challenge.history.clear();
        true
    }

    /// The next instant [`Connection::handle_timeout`] must run to service path
    /// validation (RFC 9000 §8.2), if a validation is in progress.
    #[must_use]
    pub(super) fn path_challenge_timeout(&self) -> Option<Instant> {
        // No outstanding challenge means no validation is in progress.
        self.pending_path_challenge?;
        match (
            self.path_challenge.retransmit_at,
            self.path_challenge.abandon_at,
        ) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, b) => b,
        }
    }

    /// Whether the most recent path validation was abandoned without a
    /// PATH_RESPONSE (RFC 9000 §8.2.4).
    ///
    /// Cleared when a new validation starts.
    #[must_use]
    pub fn path_validation_failed(&self) -> bool {
        self.path_challenge.failed
    }

    /// How many PATH_CHALLENGE frames have been emitted for the validation in
    /// progress (RFC 9000 §8.2.1). Zero when none is in progress.
    #[must_use]
    pub fn path_challenge_attempts(&self) -> u32 {
        self.path_challenge.attempts
    }

    /// The challenge nonces still outstanding for the validation in progress
    /// (RFC 9000 §8.2.3). Exposed for test observability.
    #[must_use]
    pub fn outstanding_path_challenges(&self) -> Vec<[u8; 8]> {
        self.path_challenge.history.clone()
    }

    /// The candidate address path validation is currently probing, if any
    /// (RFC 9000 §9.3).
    #[must_use]
    pub fn candidate_peer_addr(&self) -> Option<std::net::SocketAddr> {
        self.candidate_peer_addr
    }

    /// Set a candidate peer address for path migration (RFC 9000 §9.3).
    ///
    /// The connection will send a `PATH_CHALLENGE` toward `addr`; when the
    /// response is validated the connection migrates to it. Call
    /// [`Self::initiate_path_challenge`] after this to start the probe.
    ///
    /// RFC 9000 §9.3: `addr` is registered as a candidate path so that it gets
    /// its **own** three-times-received anti-amplification allowance. The
    /// handshake only proved the peer owns the *previous* address, so until the
    /// PATH_RESPONSE arrives this endpoint may send at most three times what it
    /// has received from `addr`.
    /// Re-offering the address that is already the candidate is a no-op. The
    /// driver calls this for *every* datagram arriving from an unfamiliar
    /// source, so resetting on each call would restart the §8.2 validation
    /// timer forever and neither the retransmission nor the abandon deadline
    /// could ever be reached.
    pub fn set_candidate_peer_addr(&mut self, addr: std::net::SocketAddr) {
        if self.candidate_peer_addr == Some(addr) {
            return;
        }
        self.candidate_peer_addr = Some(addr);
        self.register_candidate_path(addr);
        // A new candidate invalidates any previous challenge state.
        self.pending_path_challenge = None;
        self.pending_path_challenge_send = false;
        self.path_challenge.reset();
        self.path_validated = false;
        // RFC 9000 §5.1.1 / §9.5: a peer that is moving needs a spare
        // connection ID to move *to*, so top the pool back up now rather than
        // only when it retires one.
        self.replenish_local_cids();
    }

    // ─── Connection-ID supply for migration (RFC 9000 §5.1.1, §9.5) ─────────

    /// The sequence numbers of the connection IDs currently issued to the peer.
    #[must_use]
    pub fn local_cid_seqs(&self) -> Vec<u64> {
        self.local_cid_pool.seqs().collect()
    }

    /// The `retire_prior_to` threshold advertised to the peer in outgoing
    /// NEW_CONNECTION_ID frames (RFC 9000 §19.15). Non-zero once a completed
    /// migration has rotated the connection IDs (§9.5).
    #[must_use]
    pub fn local_cid_retire_prior_to(&self) -> u64 {
        self.local_cid_pool.retire_prior_to()
    }

    /// The sequence numbers of the peer-issued connection IDs we hold.
    #[must_use]
    pub fn peer_cid_seqs(&self) -> Vec<u64> {
        self.peer_cid_pool.seqs().collect()
    }

    /// The highest `retire_prior_to` the peer has asked us to honour
    /// (RFC 9000 §19.15).
    #[must_use]
    pub fn peer_cid_retire_threshold(&self) -> u64 {
        self.peer_cid_pool.retire_threshold()
    }

    /// Issue CIDs until the peer's `active_connection_id_limit` is reached.
    ///
    /// Best-effort: issuance stops at the first failure (pool full, or the
    /// secure RNG refusing to produce bytes) because there is no useful
    /// recovery — the peer simply keeps the CIDs it already has.
    ///
    /// Issuing only queues the CIDs; NEW_CONNECTION_ID is a 1-RTT-only frame
    /// (RFC 9000 §19.15) and the packet builder emits the queue exclusively in
    /// the Application space, so this is safe to call the moment the handshake
    /// completes.
    pub(super) fn replenish_local_cids(&mut self) -> usize {
        let mut issued = 0;
        while self.local_cid_pool.can_issue() {
            if self.maybe_issue_new_cid().is_err() {
                break;
            }
            issued += 1;
            if issued > self.local_cid_pool.limit as usize {
                break; // defensive: `can_issue` must converge
            }
        }
        issued
    }

    /// Rotate the connection IDs offered to the peer after a completed
    /// migration (RFC 9000 §9.5).
    ///
    /// An endpoint that keeps using the same CIDs on the new path lets an
    /// observer of both paths link them to the same connection. Raising the
    /// advertised `retire_prior_to` asks the peer to stop using every CID it
    /// held before the migration; a full fresh batch is issued in the same
    /// step so it has somewhere to go.
    ///
    /// The rotation is all-or-nothing: if not a single replacement CID could
    /// be issued the threshold is rolled back, because a peer told to retire
    /// everything with nothing to replace it would be left unable to send.
    pub(super) fn rotate_cids_after_migration(&mut self) -> usize {
        let previous = self.local_cid_pool.begin_rotation();
        let issued = self.replenish_local_cids();
        if issued == 0 {
            self.local_cid_pool.roll_back_rotation(previous);
        }
        issued
    }

    /// Whether the most recent locally-initiated path challenge was answered
    /// with a matching `PATH_RESPONSE` (RFC 9000 §9.3).
    ///
    /// Remains `true` until the next call to [`Self::initiate_path_challenge`] or
    /// [`Self::set_candidate_peer_addr`].
    #[must_use]
    pub fn path_validated(&self) -> bool {
        self.path_validated
    }

    // ─── Anti-amplification (RFC 9000 §8.1) ────────────────────────────────

    /// Record that the peer's source address has been validated, lifting the
    /// three-times amplification limit.
    ///
    /// The transport calls this itself once a Handshake packet from the peer is
    /// successfully processed (RFC 9000 §8.1: receiving a packet protected with
    /// Handshake keys proves the peer received our Initial flight). Endpoints
    /// that validate the address out of band — e.g. by accepting a Retry token
    /// before the connection is created — call it explicitly.
    pub fn mark_address_validated(&mut self) {
        self.address_validated = true;
        self.amplification_sent = 0;
    }

    /// Whether the peer's source address has been validated (RFC 9000 §8.1).
    ///
    /// Always `true` for a client connection.
    #[must_use]
    pub fn address_validated(&self) -> bool {
        self.address_validated
    }

    /// Bytes this endpoint may still send **to the address it is about to send
    /// to**, or `None` when no limit applies.
    ///
    /// RFC 9000 §8.1: "a server MUST NOT send more than three times as many
    /// bytes as the number of bytes it has received" prior to validating the
    /// client address. Without this an unauthenticated, spoofable Initial
    /// packet makes the server a reflection amplifier: a single ~1200-byte
    /// datagram elicits the whole ServerHello…Finished flight.
    ///
    /// RFC 9000 §9.3 extends the same rule to every address adopted *after* the
    /// handshake: such an address carries its own allowance (see
    /// [`super::multipath::PathAmplification`]) which is consulted first. When
    /// the target is the initial path, is unknown, or has already been
    /// validated, the connection-level allowance below applies instead.
    #[must_use]
    pub(crate) fn amplification_budget(&self) -> Option<u64> {
        self.amplification_budget_for(self.send_target())
    }

    /// The remaining allowance for sending to `target` specifically.
    ///
    /// Kept separate from [`Self::amplification_budget`] so that
    /// connection-wide decisions (notably arming the loss-detection timer)
    /// evaluate the address the connection is actually established on rather
    /// than a transient path-validation probe target.
    #[must_use]
    pub(crate) fn amplification_budget_for(&self, target: std::net::SocketAddr) -> Option<u64> {
        // Per-path allowance for an address adopted during migration.
        if let Some(path_budget) = self.multipath.path_amplification_budget(target) {
            return Some(path_budget);
        }
        if self.address_validated {
            return None;
        }
        Some(
            self.bytes_recv
                .saturating_mul(3)
                .saturating_sub(self.amplification_sent),
        )
    }

    /// Whether the endpoint is currently unable to send *anything at all*
    /// because it has consumed its anti-amplification budget (RFC 9000 §8.1).
    ///
    /// RFC 9002 §6.2.2.1 requires the loss-detection timer to stay disarmed in
    /// this state: a PTO probe would itself count against the limit and can
    /// never be sent, so arming it would only burn the budget.
    ///
    /// Deliberately evaluated against the *established* peer address, not the
    /// current send target. A path-validation probe aimed at an unvalidated
    /// candidate address may well be budget-blocked while the connection can
    /// still send freely on its validated path; disarming the connection's only
    /// recovery timer for that would let one small spoofed datagram from a new
    /// address wedge the connection permanently.
    #[must_use]
    pub fn amplification_blocked(&self) -> bool {
        matches!(
            self.amplification_budget_for(self.peer_addr),
            Some(budget) if budget < MIN_AMPLIFICATION_HEADROOM
        )
    }
}

/// Bytes reserved when turning the remaining anti-amplification allowance into
/// a packet-builder budget.
///
/// The builder's budget bounds the *payload*; the long header (first byte,
/// version, both connection IDs, token and length varints, packet number) and
/// the 16-byte AEAD tag are added on top, so the allowance must be reduced by
/// their worst case or the emitted datagram would overshoot the 3× limit.
/// Worst case with 20-byte connection IDs is under 80 bytes; 128 leaves margin.
pub(crate) const AMPLIFICATION_PACKET_OVERHEAD: u64 = 128;

/// Smallest useful send allowance. Below this no packet can be built (the
/// builder refuses payload budgets under 64 bytes once the reserve above is
/// taken out), so the connection is treated as amplification-blocked.
pub(crate) const MIN_AMPLIFICATION_HEADROOM: u64 = AMPLIFICATION_PACKET_OVERHEAD + 64;

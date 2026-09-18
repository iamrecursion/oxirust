//! Explicit Congestion Notification (ECN) for QUIC (RFC 9000 §13.4, RFC 9002 §7.4).
//!
//! ECN lets a router signal incipient congestion by setting the CE ("Congestion
//! Experienced") codepoint in the IP header instead of dropping a packet. A QUIC
//! sender marks outgoing packets with the ECT(0) codepoint, the receiver counts
//! the ECT(0)/ECT(1)/CE codepoints it observes and echoes those cumulative
//! counts back in the ECN section of an ACK frame (the 0x03 ACK variant), and the
//! sender reacts to an increase in the CE count as a congestion signal.
//!
//! Because a broken or hostile path may erase, forge, or re-order these marks,
//! RFC 9000 §13.4.2 defines a *validation* procedure: ECN is used only while the
//! feedback is self-consistent, and is disabled ("failed") permanently for a
//! packet-number space the moment the feedback is anomalous.
//!
//! This module implements both halves of ECN:
//!
//! * [`EcnCodepoint`] — the two-bit IP ECN field, with conversions to/from the
//!   low two bits of the IP Type-of-Service / Traffic-Class byte.
//! * [`EcnCounts`] — the cumulative ECT(0)/ECT(1)/CE counters, used as the wire
//!   representation of the ACK-ECN section, as the controller's last-seen
//!   snapshot of the *peer's* counters, and (via [`EcnCounts::record`]) as the
//!   per-space tally of codepoints observed on *inbound* datagrams.
//! * [`EcnController`] — the per-space validation state machine.
//!
//! ## Receive side
//!
//! The codepoint of an inbound datagram is supplied by the I/O layer through
//! [`crate::connection::DatagramMeta`] and
//! [`crate::connection::Connection::handle_datagram_with_meta`]. Every packet
//! successfully decrypted out of that datagram counts the codepoint into *its
//! own* packet-number space (a coalesced Initial + Handshake datagram bumps two
//! spaces), and the next ACK built for that space echoes the resulting counters
//! back to the peer as an ACK-ECN (0x03) frame.
//!
//! The bundled `tokio`-based [`crate::endpoint`] fills that metadata in from
//! the socket itself: [`crate::endpoint::ecn_recv`] enables `IP_RECVTOS` /
//! `IPV6_RECVTCLASS` at bind time and reads the codepoint out of the `recvmsg`
//! ancillary data of every datagram (the `unsafe` lives in `nix`, so this crate
//! stays `#![forbid(unsafe_code)]`).
//!
//! Where the platform or kernel refuses that — a non-unix target, or a kernel
//! that rejects the socket option — the metadata stays `ecn: None` and nothing
//! is synthesised: no counters move and plain (0x02) ACKs are sent, which is
//! the RFC-correct encoding for "no ECN-marked packet has been received". A
//! `None` is never treated as Not-ECT. An embedder driving [`Connection`] over
//! its own I/O gets the same behaviour by filling in `DatagramMeta::ecn`.
//!
//! [`Connection`]: crate::connection::Connection

/// The two-bit ECN field carried in the IP header (RFC 3168 §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcnCodepoint {
    /// `0b00` — Not ECN-Capable Transport: the packet is not ECN-marked.
    NotEct,
    /// `0b01` — ECN-Capable Transport codepoint 1 (ECT(1)).
    Ect1,
    /// `0b10` — ECN-Capable Transport codepoint 0 (ECT(0)); the codepoint
    /// OxiQUIC marks its egress with.
    Ect0,
    /// `0b11` — Congestion Experienced (CE): set by a router to signal
    /// congestion.
    Ce,
}

impl EcnCodepoint {
    /// The low two bits of the IP TOS / Traffic-Class byte for this codepoint.
    #[must_use]
    pub fn to_tos_bits(self) -> u8 {
        match self {
            Self::NotEct => 0b00,
            Self::Ect1 => 0b01,
            Self::Ect0 => 0b10,
            Self::Ce => 0b11,
        }
    }

    /// Decode a codepoint from the low two bits of an IP TOS / Traffic-Class
    /// byte. Only the two least-significant bits are consulted.
    #[must_use]
    pub fn from_tos_bits(tos: u8) -> Self {
        match tos & 0b11 {
            0b00 => Self::NotEct,
            0b01 => Self::Ect1,
            0b10 => Self::Ect0,
            _ => Self::Ce,
        }
    }

    /// Whether this codepoint is ECN-capable (ECT(0), ECT(1) or CE).
    #[must_use]
    pub fn is_ect(self) -> bool {
        !matches!(self, Self::NotEct)
    }
}

/// Cumulative ECN codepoint counters (RFC 9000 §19.3.2).
///
/// Doubles as the wire payload of the ACK-ECN section and as the controller's
/// last-seen snapshot of the peer's counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EcnCounts {
    /// Number of packets received with the ECT(0) codepoint.
    pub ect0: u64,
    /// Number of packets received with the ECT(1) codepoint.
    pub ect1: u64,
    /// Number of packets received with the CE codepoint.
    pub ce: u64,
}

impl EcnCounts {
    /// Construct a counter triple.
    #[must_use]
    pub fn new(ect0: u64, ect1: u64, ce: u64) -> Self {
        Self { ect0, ect1, ce }
    }

    /// Count one *received* packet that carried `cp` (RFC 9000 §13.4.1).
    ///
    /// Only the three ECN-capable codepoints have counters; a Not-ECT packet
    /// increments nothing, which is exactly what the ACK-ECN section reports.
    /// All increments saturate, so a long-lived connection can never panic.
    pub fn record(&mut self, cp: EcnCodepoint) {
        match cp {
            EcnCodepoint::NotEct => {}
            EcnCodepoint::Ect1 => self.ect1 = self.ect1.saturating_add(1),
            EcnCodepoint::Ect0 => self.ect0 = self.ect0.saturating_add(1),
            EcnCodepoint::Ce => self.ce = self.ce.saturating_add(1),
        }
    }

    /// Whether no ECN-marked packet has been counted yet.
    ///
    /// RFC 9000 §13.4.1: an endpoint sends the ACK-ECN (0x03) variant only once
    /// it has actually received an ECN-marked packet in that packet-number
    /// space; until then a plain (0x02) ACK is the correct encoding.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ect0 == 0 && self.ect1 == 0 && self.ce == 0
    }
}

/// The ECN validation state for one packet-number space (RFC 9000 §13.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcnValidationState {
    /// ECT(0) is being marked but the path has not yet been confirmed to
    /// preserve ECN marks end-to-end. This is the initial state.
    Testing,
    /// The peer has returned consistent ECN feedback: ECN is validated and its
    /// CE reports drive the congestion response.
    Capable,
    /// The feedback was anomalous (marks bleached, counts rolled back, or ECN
    /// counts missing for ECT-marked packets). ECN is disabled for this space.
    Failed,
    /// The local socket refused to set the ECN codepoint (unsupported
    /// platform / API), so we never mark and never validate.
    Unsupported,
}

/// The result of validating one ACK against a space's ECN state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EcnAckOutcome {
    /// The peer's CE counter increased since the last validated ACK — a
    /// congestion signal that must reduce the congestion window (RFC 9002 §7.4).
    pub ce_increase: bool,
    /// This ACK failed validation and ECN has just been disabled for the space.
    pub failed: bool,
}

/// Per-space ECN validation state machine (RFC 9000 §13.4.2, RFC 9002 §7.4).
#[derive(Debug, Clone)]
pub struct EcnController {
    state: EcnValidationState,
    /// The last set of counters we accepted from the peer; validation is
    /// relative to this snapshot.
    last_counts: EcnCounts,
    /// How many ECT(0)-marked packets we have sent in this space (diagnostics).
    ect0_sent: u64,
}

impl Default for EcnController {
    fn default() -> Self {
        Self::new()
    }
}

impl EcnController {
    /// Create a controller in the initial [`EcnValidationState::Testing`] state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: EcnValidationState::Testing,
            last_counts: EcnCounts::default(),
            ect0_sent: 0,
        }
    }

    /// The current validation state.
    #[must_use]
    pub fn state(&self) -> EcnValidationState {
        self.state
    }

    /// The codepoint the sender should mark with right now: ECT(0) while we are
    /// testing or ECN-capable, otherwise Not-ECT.
    #[must_use]
    pub fn desired_codepoint(&self) -> EcnCodepoint {
        match self.state {
            EcnValidationState::Testing | EcnValidationState::Capable => EcnCodepoint::Ect0,
            EcnValidationState::Failed | EcnValidationState::Unsupported => EcnCodepoint::NotEct,
        }
    }

    /// Whether the controller currently wants to mark ECT(0).
    #[must_use]
    pub fn wants_ect0(&self) -> bool {
        self.desired_codepoint() == EcnCodepoint::Ect0
    }

    /// Whether ECN marking has been ruled out for this space.
    #[must_use]
    pub fn is_unsupported(&self) -> bool {
        self.state == EcnValidationState::Unsupported
    }

    /// Record that one ECT(0)-marked packet has been sent (diagnostics only).
    pub fn on_marked_packet_sent(&mut self) {
        self.ect0_sent = self.ect0_sent.saturating_add(1);
    }

    /// The number of ECT(0)-marked packets sent in this space.
    #[must_use]
    pub fn ect0_sent(&self) -> u64 {
        self.ect0_sent
    }

    /// Permanently disable ECN because the socket refused to mark the codepoint.
    pub fn mark_unsupported(&mut self) {
        self.state = EcnValidationState::Unsupported;
    }

    /// Validate the ECN feedback in a received ACK (RFC 9000 §13.4.2.1).
    ///
    /// * `reported` is the ACK's ECN section, or `None` for a plain (0x02) ACK.
    /// * `newly_acked_ect0` is the number of packets newly acknowledged by this
    ///   ACK that this endpoint originally sent with the ECT(0) codepoint.
    ///
    /// Returns whether the CE count increased (a congestion signal) and whether
    /// validation failed (ECN disabled for the space). All arithmetic saturates,
    /// so hostile or rolled-back counts can never panic.
    pub fn validate_ack(
        &mut self,
        reported: Option<EcnCounts>,
        newly_acked_ect0: u64,
    ) -> EcnAckOutcome {
        let mut outcome = EcnAckOutcome::default();

        // Once ECN is off for the space (Failed/Unsupported) there is nothing to
        // validate and no congestion signal to extract.
        if !matches!(
            self.state,
            EcnValidationState::Testing | EcnValidationState::Capable
        ) {
            return outcome;
        }

        let reported = match reported {
            Some(counts) => counts,
            None => {
                // RFC 9000 §13.4.2.1: if an ACK newly acknowledges an ECT(0)-marked
                // packet but omits the ECN counts, the peer or path stripped ECN.
                if newly_acked_ect0 > 0 {
                    self.state = EcnValidationState::Failed;
                    outcome.failed = true;
                }
                return outcome;
            }
        };

        // (a) The cumulative counters must never decrease. A rollback means the
        // feedback is corrupt or forged.
        if reported.ect0 < self.last_counts.ect0
            || reported.ect1 < self.last_counts.ect1
            || reported.ce < self.last_counts.ce
        {
            self.state = EcnValidationState::Failed;
            outcome.failed = true;
            return outcome;
        }

        // (b) The increase in the ECT(0) and CE counters must cover every
        // newly-acknowledged ECT(0)-marked packet; a shortfall means the path
        // bleached ECN marks to Not-ECT.
        let ect0_delta = reported.ect0.saturating_sub(self.last_counts.ect0);
        let ce_delta = reported.ce.saturating_sub(self.last_counts.ce);
        if ect0_delta.saturating_add(ce_delta) < newly_acked_ect0 {
            self.state = EcnValidationState::Failed;
            outcome.failed = true;
            return outcome;
        }

        // A rise in the CE counter is a congestion signal (RFC 9002 §7.4).
        outcome.ce_increase = ce_delta > 0;

        // Accept the new snapshot and, on the first confirmed feedback, promote
        // the space from Testing to Capable.
        self.last_counts = reported;
        if self.state == EcnValidationState::Testing && newly_acked_ect0 > 0 {
            self.state = EcnValidationState::Capable;
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codepoint_tos_bits_roundtrip() {
        for cp in [
            EcnCodepoint::NotEct,
            EcnCodepoint::Ect1,
            EcnCodepoint::Ect0,
            EcnCodepoint::Ce,
        ] {
            assert_eq!(EcnCodepoint::from_tos_bits(cp.to_tos_bits()), cp);
        }
        // The RFC-mandated bit patterns.
        assert_eq!(EcnCodepoint::NotEct.to_tos_bits(), 0b00);
        assert_eq!(EcnCodepoint::Ect1.to_tos_bits(), 0b01);
        assert_eq!(EcnCodepoint::Ect0.to_tos_bits(), 0b10);
        assert_eq!(EcnCodepoint::Ce.to_tos_bits(), 0b11);
    }

    #[test]
    fn from_tos_bits_ignores_high_bits() {
        // Only the low two bits matter; a full DSCP+ECN byte still decodes.
        assert_eq!(EcnCodepoint::from_tos_bits(0b1011_0110), EcnCodepoint::Ect0);
        assert_eq!(EcnCodepoint::from_tos_bits(0xff), EcnCodepoint::Ce);
    }

    #[test]
    fn initial_state_marks_ect0() {
        let c = EcnController::new();
        assert_eq!(c.state(), EcnValidationState::Testing);
        assert_eq!(c.desired_codepoint(), EcnCodepoint::Ect0);
        assert!(c.wants_ect0());
    }

    #[test]
    fn validation_promotes_testing_to_capable() {
        let mut c = EcnController::new();
        // Peer reports 3 ECT(0) received, covering our 3 newly-acked ECT(0).
        let outcome = c.validate_ack(Some(EcnCounts::new(3, 0, 0)), 3);
        assert!(!outcome.failed);
        assert!(!outcome.ce_increase);
        assert_eq!(c.state(), EcnValidationState::Capable);
        assert_eq!(c.desired_codepoint(), EcnCodepoint::Ect0);
    }

    #[test]
    fn monotonic_decrease_fails() {
        let mut c = EcnController::new();
        c.validate_ack(Some(EcnCounts::new(5, 0, 2)), 5);
        assert_eq!(c.state(), EcnValidationState::Capable);
        // A later ACK reports fewer ECT(0) than before: rollback → Failed.
        let outcome = c.validate_ack(Some(EcnCounts::new(4, 0, 2)), 1);
        assert!(outcome.failed);
        assert_eq!(c.state(), EcnValidationState::Failed);
        assert_eq!(c.desired_codepoint(), EcnCodepoint::NotEct);
    }

    #[test]
    fn ect0_acked_but_non_ecn_ack_fails() {
        let mut c = EcnController::new();
        // We acked ECT(0) packets but the ACK carried no ECN section.
        let outcome = c.validate_ack(None, 2);
        assert!(outcome.failed);
        assert_eq!(c.state(), EcnValidationState::Failed);
    }

    #[test]
    fn non_ecn_ack_without_ect0_is_harmless() {
        let mut c = EcnController::new();
        let outcome = c.validate_ack(None, 0);
        assert!(!outcome.failed);
        assert_eq!(c.state(), EcnValidationState::Testing);
    }

    #[test]
    fn bleaching_fails() {
        let mut c = EcnController::new();
        // 4 ECT(0) packets newly acked but the counters rose by only 1: the
        // path bleached 3 marks to Not-ECT.
        let outcome = c.validate_ack(Some(EcnCounts::new(1, 0, 0)), 4);
        assert!(outcome.failed);
        assert_eq!(c.state(), EcnValidationState::Failed);
    }

    #[test]
    fn ce_increase_is_reported_and_counted_toward_coverage() {
        let mut c = EcnController::new();
        // 2 ECT(0) acked; counters rose by ect0=1, ce=1 → covers 2, CE up by 1.
        let outcome = c.validate_ack(Some(EcnCounts::new(1, 0, 1)), 2);
        assert!(!outcome.failed);
        assert!(outcome.ce_increase);
        assert_eq!(c.state(), EcnValidationState::Capable);
    }

    #[test]
    fn unsupported_never_marks_or_validates() {
        let mut c = EcnController::new();
        c.mark_unsupported();
        assert!(c.is_unsupported());
        assert_eq!(c.desired_codepoint(), EcnCodepoint::NotEct);
        // Validation is inert once unsupported.
        let outcome = c.validate_ack(None, 5);
        assert!(!outcome.failed);
        assert_eq!(c.state(), EcnValidationState::Unsupported);
    }

    #[test]
    fn marked_packet_counter_saturates_cleanly() {
        let mut c = EcnController::new();
        c.on_marked_packet_sent();
        c.on_marked_packet_sent();
        assert_eq!(c.ect0_sent(), 2);
    }
}

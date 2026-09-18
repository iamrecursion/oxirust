#![allow(dead_code)]
//! RTP session tracking and statistics.
//!
//! Provides types for managing RTP stream state, detecting sequence gaps,
//! and computing packet-loss percentages.

/// Well-known and dynamic RTP payload types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RtpPayloadType {
    /// PCMU (G.711 µ-law) – PT 0.
    Pcmu,
    /// PCMA (G.711 A-law) – PT 8.
    Pcma,
    /// Telephone-event (RFC 4733) – PT 101 (common dynamic).
    TelephoneEvent,
    /// H.264 video – dynamic.
    H264,
    /// H.265 / HEVC video – dynamic.
    H265,
    /// Opus audio – dynamic.
    Opus,
    /// VP8 video – dynamic.
    Vp8,
    /// VP9 video – dynamic.
    Vp9,
    /// AV1 video – dynamic.
    Av1,
    /// Any other dynamic payload type, carrying the raw PT value.
    Dynamic(u8),
}

impl RtpPayloadType {
    /// Returns `true` if this is a dynamic payload type (PT ≥ 96).
    #[must_use]
    pub fn is_dynamic(&self) -> bool {
        match self {
            Self::Pcmu | Self::Pcma => false,
            Self::TelephoneEvent
            | Self::H264
            | Self::H265
            | Self::Opus
            | Self::Vp8
            | Self::Vp9
            | Self::Av1 => true,
            Self::Dynamic(_) => true,
        }
    }

    /// Returns the raw 7-bit payload-type number (0–127) where known.
    #[must_use]
    pub fn raw_pt(&self) -> Option<u8> {
        match self {
            Self::Pcmu => Some(0),
            Self::Pcma => Some(8),
            Self::TelephoneEvent => Some(101),
            Self::Dynamic(pt) => Some(*pt),
            _ => None,
        }
    }
}

/// Metadata carried in a single RTP packet header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpPacketInfo {
    /// 16-bit sequence number.
    pub sequence_number: u16,
    /// 32-bit RTP timestamp.
    pub timestamp: u32,
    /// Synchronisation source identifier.
    pub ssrc: u32,
    /// Payload type.
    pub payload_type: RtpPayloadType,
    /// Payload size in bytes (not including header).
    pub payload_size: usize,
    /// Whether the marker bit is set.
    pub marker: bool,
}

impl RtpPacketInfo {
    /// Creates a new `RtpPacketInfo`.
    #[must_use]
    pub const fn new(
        sequence_number: u16,
        timestamp: u32,
        ssrc: u32,
        payload_type: RtpPayloadType,
        payload_size: usize,
        marker: bool,
    ) -> Self {
        Self {
            sequence_number,
            timestamp,
            ssrc,
            payload_type,
            payload_size,
            marker,
        }
    }

    /// Computes the forward sequence gap between `self` and a `previous` packet.
    ///
    /// A gap of 1 means consecutive, larger values indicate lost packets.
    /// Uses wrapping arithmetic to handle sequence-number roll-over correctly.
    #[must_use]
    pub fn sequence_gap(&self, previous: &Self) -> u16 {
        self.sequence_number.wrapping_sub(previous.sequence_number)
    }

    /// Returns `true` when a packet appears to have been lost between
    /// `previous` and `self` (gap > 1).
    #[must_use]
    pub fn has_gap(&self, previous: &Self) -> bool {
        self.sequence_gap(previous) > 1
    }
}

/// Cumulative statistics for an RTP session.
#[derive(Debug, Clone, Default)]
pub struct RtpStats {
    /// Total packets received (may include late/duplicate packets).
    pub packets_received: u64,
    /// Estimated number of lost packets.
    pub packets_lost: u64,
    /// Total bytes received (payload only).
    pub bytes_received: u64,
    /// Sequence gaps detected (each gap == one or more lost packets).
    pub gaps_detected: u64,
}

impl RtpStats {
    /// Creates zeroed `RtpStats`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes the packet-loss percentage in the range `[0.0, 100.0]`.
    ///
    /// Returns `0.0` when no packets have been expected yet.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn packet_loss_pct(&self) -> f64 {
        let expected = self.packets_received + self.packets_lost;
        if expected == 0 {
            return 0.0;
        }
        (self.packets_lost as f64 / expected as f64) * 100.0
    }
}

/// An active RTP session for a single SSRC.
#[derive(Debug)]
pub struct RtpSession {
    /// SSRC this session tracks.
    pub ssrc: u32,
    /// Next expected sequence number (wrapping).
    next_seq: u16,
    /// Whether the first packet has arrived.
    initialised: bool,
    /// Accumulated statistics.
    stats: RtpStats,
}

impl RtpSession {
    /// Creates a new `RtpSession` for the given SSRC.
    #[must_use]
    pub fn new(ssrc: u32) -> Self {
        Self {
            ssrc,
            next_seq: 0,
            initialised: false,
            stats: RtpStats::new(),
        }
    }

    /// Returns the next expected sequence number.
    #[must_use]
    pub fn next_seq(&self) -> u16 {
        self.next_seq
    }

    /// Processes an incoming packet and updates session statistics.
    pub fn update_stats(&mut self, packet: &RtpPacketInfo) {
        if !self.initialised {
            self.initialised = true;
            self.next_seq = packet.sequence_number.wrapping_add(1);
            self.stats.packets_received += 1;
            self.stats.bytes_received += packet.payload_size as u64;
            return;
        }

        let gap = packet
            .sequence_number
            .wrapping_sub(self.next_seq.wrapping_sub(1));

        if gap > 1 {
            // gap - 1 packets appear lost
            let lost = u64::from(gap) - 1;
            self.stats.packets_lost += lost;
            self.stats.gaps_detected += 1;
        }

        self.stats.packets_received += 1;
        self.stats.bytes_received += packet.payload_size as u64;
        self.next_seq = packet.sequence_number.wrapping_add(1);
    }

    /// Returns a reference to the current session statistics.
    #[must_use]
    pub fn stats(&self) -> &RtpStats {
        &self.stats
    }

    /// Returns `true` if the session has received at least one packet.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.initialised
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_packet(seq: u16, ssrc: u32) -> RtpPacketInfo {
        RtpPacketInfo::new(
            seq,
            seq as u32 * 960,
            ssrc,
            RtpPayloadType::Opus,
            120,
            false,
        )
    }

    // 1. Dynamic payload type detection
    #[test]
    fn test_is_dynamic_pcmu() {
        assert!(!RtpPayloadType::Pcmu.is_dynamic());
    }

    #[test]
    fn test_is_dynamic_opus() {
        assert!(RtpPayloadType::Opus.is_dynamic());
    }

    #[test]
    fn test_is_dynamic_dynamic_variant() {
        assert!(RtpPayloadType::Dynamic(96).is_dynamic());
    }

    // 2. raw_pt values
    #[test]
    fn test_raw_pt_pcmu() {
        assert_eq!(RtpPayloadType::Pcmu.raw_pt(), Some(0));
    }

    #[test]
    fn test_raw_pt_pcma() {
        assert_eq!(RtpPayloadType::Pcma.raw_pt(), Some(8));
    }

    #[test]
    fn test_raw_pt_dynamic() {
        assert_eq!(RtpPayloadType::Dynamic(111).raw_pt(), Some(111));
    }

    // 3. sequence_gap – consecutive packets
    #[test]
    fn test_sequence_gap_consecutive() {
        let a = make_packet(10, 1);
        let b = make_packet(11, 1);
        assert_eq!(b.sequence_gap(&a), 1);
    }

    // 4. sequence_gap – wrapping
    #[test]
    fn test_sequence_gap_wrap() {
        let a = make_packet(u16::MAX, 1);
        let b = make_packet(0, 1);
        assert_eq!(b.sequence_gap(&a), 1);
    }

    // 5. has_gap detects missing packets
    #[test]
    fn test_has_gap_true() {
        let a = make_packet(10, 1);
        let b = make_packet(15, 1);
        assert!(b.has_gap(&a));
    }

    // 6. has_gap returns false for consecutive
    #[test]
    fn test_has_gap_false_consecutive() {
        let a = make_packet(10, 1);
        let b = make_packet(11, 1);
        assert!(!b.has_gap(&a));
    }

    // 7. RtpStats packet_loss_pct – no packets
    #[test]
    fn test_packet_loss_pct_zero_expected() {
        let s = RtpStats::new();
        assert_eq!(s.packet_loss_pct(), 0.0);
    }

    // 8. RtpStats packet_loss_pct – 50 %
    #[test]
    fn test_packet_loss_pct_50() {
        let s = RtpStats {
            packets_received: 50,
            packets_lost: 50,
            ..Default::default()
        };
        assert!((s.packet_loss_pct() - 50.0).abs() < 1e-9);
    }

    // 9. RtpSession initial state
    #[test]
    fn test_session_initial_not_active() {
        let sess = RtpSession::new(0xDEAD_BEEF);
        assert!(!sess.is_active());
    }

    // 10. RtpSession becomes active after first packet
    #[test]
    fn test_session_active_after_first_packet() {
        let mut sess = RtpSession::new(1);
        sess.update_stats(&make_packet(0, 1));
        assert!(sess.is_active());
    }

    // 11. RtpSession no loss on consecutive packets
    #[test]
    fn test_session_no_loss_consecutive() {
        let mut sess = RtpSession::new(1);
        for seq in 0u16..10 {
            sess.update_stats(&make_packet(seq, 1));
        }
        assert_eq!(sess.stats().packets_lost, 0);
        assert_eq!(sess.stats().packets_received, 10);
    }

    // 12. RtpSession detects gap and records lost packets
    #[test]
    fn test_session_gap_detection() {
        let mut sess = RtpSession::new(1);
        sess.update_stats(&make_packet(0, 1));
        // skip packets 1–4, deliver 5
        sess.update_stats(&make_packet(5, 1));
        assert_eq!(sess.stats().packets_lost, 4);
        assert_eq!(sess.stats().gaps_detected, 1);
    }

    // 13. next_seq advances correctly
    #[test]
    fn test_session_next_seq() {
        let mut sess = RtpSession::new(1);
        sess.update_stats(&make_packet(42, 1));
        assert_eq!(sess.next_seq(), 43);
    }

    // 14. bytes_received accumulates
    #[test]
    fn test_session_bytes_received() {
        let mut sess = RtpSession::new(1);
        for seq in 0u16..5 {
            sess.update_stats(&make_packet(seq, 1));
        }
        assert_eq!(sess.stats().bytes_received, 5 * 120);
    }
}

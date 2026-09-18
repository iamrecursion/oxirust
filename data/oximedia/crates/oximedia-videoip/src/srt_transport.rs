#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]

//! SRT (Secure Reliable Transport) transport module.
//!
//! This module implements the core SRT protocol concepts in pure Rust without
//! requiring any real network sockets. It models the handshake state machine,
//! packet types, sender/receiver flow, and loss-recovery NAK/ACK exchange in
//! memory.
//!
//! # Design
//!
//! - [`SrtHandshake`] — four-step handshake state machine (INDUCTION →
//!   WAVEAHAND → CONCLUSION → AGREEMENT).
//! - [`SrtPacketHeader`] — wire-format header with sequence number,
//!   timestamp, and packet type.
//! - [`SrtPacketType`] — discriminates control vs data packets.
//! - [`SrtSender`] / [`SrtReceiver`] — in-memory send/recv with loss-list,
//!   ACK/NAK feedback and retransmission.
//! - [`SrtSession`] — pairs a sender and receiver for loopback or inter-thread
//!   communication via channels.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::error::{VideoIpError, VideoIpResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// SRT transport configuration.
#[derive(Debug, Clone)]
pub struct SrtTransportConfig {
    /// Target latency / receive-buffer depth in milliseconds.
    pub latency_ms: u32,
    /// Maximum bandwidth in bits/s (0 = unlimited).
    pub max_bandwidth_bps: u64,
    /// Optional passphrase for stream-level encryption tagging.
    /// (Full cryptographic keying is out of scope for this in-memory model.)
    pub passphrase: Option<String>,
    /// Maximum segment size (payload bytes per logical packet).
    pub mss: u16,
    /// Initial sender socket ID (randomly assigned in real SRT; fixed here for testing).
    pub socket_id: u32,
    /// Peer socket ID (populated during handshake).
    pub peer_socket_id: u32,
}

impl Default for SrtTransportConfig {
    fn default() -> Self {
        Self {
            latency_ms: 120,
            max_bandwidth_bps: 0,
            passphrase: None,
            mss: 1316, // SRT default payload MTU
            socket_id: 0x0001_0001,
            peer_socket_id: 0x0001_0002,
        }
    }
}

impl SrtTransportConfig {
    /// Create a new config with the given latency.
    #[must_use]
    pub fn with_latency(latency_ms: u32) -> Self {
        Self {
            latency_ms,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid parameters (e.g. latency too large, passphrase
    /// too short).
    pub fn validate(&self) -> VideoIpResult<()> {
        if self.latency_ms > 30_000 {
            return Err(VideoIpError::InvalidState(
                "SRT latency must be <= 30 000 ms".into(),
            ));
        }
        if let Some(ref pp) = self.passphrase {
            if pp.len() < 10 || pp.len() > 79 {
                return Err(VideoIpError::InvalidState(
                    "SRT passphrase must be 10–79 characters".into(),
                ));
            }
        }
        if self.mss < 76 {
            return Err(VideoIpError::InvalidState("SRT MSS must be >= 76".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Handshake state machine
// ---------------------------------------------------------------------------

/// SRT handshake phase (caller-side view).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SrtHandshakePhase {
    /// Initial state: caller sends first INDUCTION handshake.
    Induction,
    /// Caller has sent INDUCTION, awaiting responder's WAVEAHAND.
    WaveAHand,
    /// Caller sends CONCLUSION (with extension fields if needed).
    Conclusion,
    /// Handshake complete (AGREEMENT received / sent).
    Agreement,
    /// Handshake failed.
    Failed,
}

impl SrtHandshakePhase {
    /// Returns `true` if the handshake is complete (AGREEMENT).
    #[must_use]
    pub fn is_complete(self) -> bool {
        self == Self::Agreement
    }

    /// Returns `true` if the handshake failed.
    #[must_use]
    pub fn is_failed(self) -> bool {
        self == Self::Failed
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Induction => "INDUCTION",
            Self::WaveAHand => "WAVEAHAND",
            Self::Conclusion => "CONCLUSION",
            Self::Agreement => "AGREEMENT",
            Self::Failed => "FAILED",
        }
    }
}

impl std::fmt::Display for SrtHandshakePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// SRT handshake message.
#[derive(Debug, Clone)]
pub struct SrtHandshakeMsg {
    /// Phase this message belongs to.
    pub phase: SrtHandshakePhase,
    /// Initiator's socket ID.
    pub socket_id: u32,
    /// Peer socket ID (0 if not yet known).
    pub peer_socket_id: u32,
    /// Negotiated latency in milliseconds.
    pub latency_ms: u32,
    /// SRT version (major × 0x10000 + minor × 0x100 + patch).
    pub version: u32,
    /// Stream ID string (application-level routing).
    pub stream_id: Option<String>,
}

impl SrtHandshakeMsg {
    /// SRT version encoded in the handshake (1.4.3 → 0x010403).
    pub const SRT_VERSION: u32 = 0x0001_0403;

    /// Create an INDUCTION handshake from the given config.
    #[must_use]
    pub fn induction(cfg: &SrtTransportConfig) -> Self {
        Self {
            phase: SrtHandshakePhase::Induction,
            socket_id: cfg.socket_id,
            peer_socket_id: 0,
            latency_ms: cfg.latency_ms,
            version: Self::SRT_VERSION,
            stream_id: None,
        }
    }

    /// Produce the WAVEAHAND response (responder side).
    #[must_use]
    pub fn waveahand(&self, responder_socket_id: u32) -> Self {
        Self {
            phase: SrtHandshakePhase::WaveAHand,
            socket_id: responder_socket_id,
            peer_socket_id: self.socket_id,
            latency_ms: self.latency_ms,
            version: Self::SRT_VERSION,
            stream_id: None,
        }
    }

    /// Produce the CONCLUSION message (caller side, after receiving WAVEAHAND).
    #[must_use]
    pub fn conclusion(&self, stream_id: Option<String>) -> Self {
        Self {
            phase: SrtHandshakePhase::Conclusion,
            socket_id: self.socket_id,
            peer_socket_id: self.peer_socket_id,
            latency_ms: self.latency_ms,
            version: Self::SRT_VERSION,
            stream_id,
        }
    }

    /// Produce the AGREEMENT message (responder side, after receiving CONCLUSION).
    #[must_use]
    pub fn agreement(&self) -> Self {
        Self {
            phase: SrtHandshakePhase::Agreement,
            socket_id: self.peer_socket_id,
            peer_socket_id: self.socket_id,
            latency_ms: self.latency_ms,
            version: Self::SRT_VERSION,
            stream_id: self.stream_id.clone(),
        }
    }
}

/// SRT handshake state machine (caller side).
///
/// Drive the machine by calling [`Self::step`] with the message received from
/// the peer. On completion, the machine transitions to `Agreement` and the
/// negotiated parameters are accessible.
#[derive(Debug)]
pub struct SrtHandshake {
    /// Current phase.
    phase: SrtHandshakePhase,
    /// Configuration.
    config: SrtTransportConfig,
    /// The last message we sent (for retransmit purposes).
    last_sent: Option<SrtHandshakeMsg>,
    /// The agreed peer socket ID.
    peer_socket_id: Option<u32>,
    /// Negotiated latency.
    negotiated_latency_ms: u32,
    /// Optional stream ID from CONCLUSION.
    stream_id: Option<String>,
}

impl SrtHandshake {
    /// Create a new handshake state machine.
    #[must_use]
    pub fn new(config: SrtTransportConfig) -> Self {
        let latency = config.latency_ms;
        Self {
            phase: SrtHandshakePhase::Induction,
            config,
            last_sent: None,
            peer_socket_id: None,
            negotiated_latency_ms: latency,
            stream_id: None,
        }
    }

    /// Returns the first message to send (INDUCTION).
    #[must_use]
    pub fn initiate(&mut self) -> SrtHandshakeMsg {
        let msg = SrtHandshakeMsg::induction(&self.config);
        self.last_sent = Some(msg.clone());
        self.phase = SrtHandshakePhase::WaveAHand;
        msg
    }

    /// Process an incoming handshake message and return the response (if any).
    ///
    /// # Errors
    ///
    /// Returns an error if the message is unexpected for the current phase.
    pub fn step(&mut self, incoming: &SrtHandshakeMsg) -> VideoIpResult<Option<SrtHandshakeMsg>> {
        match (self.phase, incoming.phase) {
            (SrtHandshakePhase::WaveAHand, SrtHandshakePhase::WaveAHand) => {
                // Received peer's WAVEAHAND → send CONCLUSION
                self.peer_socket_id = Some(incoming.socket_id);
                self.negotiated_latency_ms = self.config.latency_ms.max(incoming.latency_ms);
                let reply = SrtHandshakeMsg {
                    phase: SrtHandshakePhase::Conclusion,
                    socket_id: self.config.socket_id,
                    peer_socket_id: incoming.socket_id,
                    latency_ms: self.negotiated_latency_ms,
                    version: SrtHandshakeMsg::SRT_VERSION,
                    stream_id: self.stream_id.clone(),
                };
                self.last_sent = Some(reply.clone());
                self.phase = SrtHandshakePhase::Conclusion;
                Ok(Some(reply))
            }
            (SrtHandshakePhase::Conclusion, SrtHandshakePhase::Agreement) => {
                // Handshake complete
                self.negotiated_latency_ms = incoming.latency_ms;
                self.phase = SrtHandshakePhase::Agreement;
                Ok(None) // no further reply needed
            }
            _ => {
                self.phase = SrtHandshakePhase::Failed;
                Err(VideoIpError::InvalidState(format!(
                    "unexpected handshake message {:?} in phase {:?}",
                    incoming.phase, self.phase
                )))
            }
        }
    }

    /// Returns the current phase.
    #[must_use]
    pub fn phase(&self) -> SrtHandshakePhase {
        self.phase
    }

    /// Returns `true` if the handshake is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.phase.is_complete()
    }

    /// Returns the negotiated latency (valid after AGREEMENT).
    #[must_use]
    pub fn negotiated_latency_ms(&self) -> u32 {
        self.negotiated_latency_ms
    }

    /// Returns the peer's socket ID (valid after WAVEAHAND received).
    #[must_use]
    pub fn peer_socket_id(&self) -> Option<u32> {
        self.peer_socket_id
    }

    /// Attach a stream ID that will be included in the CONCLUSION message.
    pub fn set_stream_id(&mut self, id: impl Into<String>) {
        self.stream_id = Some(id.into());
    }
}

// ---------------------------------------------------------------------------
// Packet types
// ---------------------------------------------------------------------------

/// SRT packet type discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SrtPacketType {
    /// Data packet carrying media payload.
    Data,
    /// Selective ACK — acknowledges cumulative and selective receipt.
    Ack,
    /// Negative ACK — requests retransmission of missing sequences.
    Nak,
    /// Keep-alive ping.
    KeepAlive,
    /// Handshake control packet.
    Handshake,
    /// Graceful shutdown.
    Shutdown,
    /// ACK-of-ACK (sender acknowledges receiver's ACK).
    AckAck,
}

impl SrtPacketType {
    /// Returns `true` for data packets, `false` for control.
    #[must_use]
    pub fn is_data(self) -> bool {
        self == Self::Data
    }

    /// Returns `true` for control packets.
    #[must_use]
    pub fn is_control(self) -> bool {
        !self.is_data()
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Data => "DATA",
            Self::Ack => "ACK",
            Self::Nak => "NAK",
            Self::KeepAlive => "KEEPALIVE",
            Self::Handshake => "HANDSHAKE",
            Self::Shutdown => "SHUTDOWN",
            Self::AckAck => "ACKACK",
        }
    }
}

impl std::fmt::Display for SrtPacketType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// SRT packet header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtPacketHeader {
    /// Packet type.
    pub packet_type: SrtPacketType,
    /// Sequence number (wrapping 32-bit for data; sub-type field for control).
    pub sequence_number: u32,
    /// Microsecond timestamp relative to connection epoch.
    pub timestamp: u32,
    /// Destination socket ID.
    pub destination_socket_id: u32,
}

impl SrtPacketHeader {
    /// Create a DATA packet header.
    #[must_use]
    pub fn data(sequence_number: u32, timestamp: u32, destination_socket_id: u32) -> Self {
        Self {
            packet_type: SrtPacketType::Data,
            sequence_number,
            timestamp,
            destination_socket_id,
        }
    }

    /// Create a control packet header.
    #[must_use]
    pub fn control(
        packet_type: SrtPacketType,
        sub_sequence: u32,
        timestamp: u32,
        destination_socket_id: u32,
    ) -> Self {
        Self {
            packet_type,
            sequence_number: sub_sequence,
            timestamp,
            destination_socket_id,
        }
    }
}

/// An SRT packet (header + payload).
#[derive(Debug, Clone)]
pub struct SrtPacket {
    /// Packet header.
    pub header: SrtPacketHeader,
    /// Payload bytes (empty for pure control packets).
    pub payload: Vec<u8>,
}

impl SrtPacket {
    /// Create a data packet.
    #[must_use]
    pub fn data(
        sequence_number: u32,
        timestamp: u32,
        dst_socket_id: u32,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            header: SrtPacketHeader::data(sequence_number, timestamp, dst_socket_id),
            payload,
        }
    }

    /// Create a control packet with no payload.
    #[must_use]
    pub fn control(
        packet_type: SrtPacketType,
        sub_seq: u32,
        timestamp: u32,
        dst_socket_id: u32,
    ) -> Self {
        Self {
            header: SrtPacketHeader::control(packet_type, sub_seq, timestamp, dst_socket_id),
            payload: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics for an SRT sender.
#[derive(Debug, Clone, Default)]
pub struct SrtSenderStats {
    /// Packets sent (original transmissions).
    pub packets_sent: u64,
    /// Packets retransmitted due to NAK.
    pub packets_retransmitted: u64,
    /// ACKs received.
    pub acks_received: u64,
    /// NAKs received.
    pub naks_received: u64,
    /// Current send buffer occupancy.
    pub send_buffer_occupancy: usize,
}

/// Statistics for an SRT receiver.
#[derive(Debug, Clone, Default)]
pub struct SrtReceiverStats {
    /// Packets received (in-order).
    pub packets_received: u64,
    /// Packets recovered via retransmission.
    pub packets_recovered: u64,
    /// Packets dropped (too late / unrecoverable).
    pub packets_dropped: u64,
    /// ACKs sent.
    pub acks_sent: u64,
    /// NAKs sent.
    pub naks_sent: u64,
    /// Current receive buffer occupancy.
    pub recv_buffer_occupancy: usize,
}

// ---------------------------------------------------------------------------
// SRT Sender
// ---------------------------------------------------------------------------

/// SRT sender — buffers outgoing data, retransmits on NAK.
pub struct SrtSender {
    /// Configuration.
    config: SrtTransportConfig,
    /// Outgoing packets awaiting acknowledgement (sequence → packet).
    send_buffer: HashMap<u32, SrtPacket>,
    /// Next sequence number to use.
    next_seq: u32,
    /// Highest ACKed sequence (cumulative).
    acked_up_to: Option<u32>,
    /// Microsecond timestamp at connection start.
    epoch: Instant,
    /// Statistics.
    stats: SrtSenderStats,
}

impl SrtSender {
    /// Create a new SRT sender.
    #[must_use]
    pub fn new(config: SrtTransportConfig) -> Self {
        Self {
            config,
            send_buffer: HashMap::new(),
            next_seq: 0,
            acked_up_to: None,
            epoch: Instant::now(),
            stats: SrtSenderStats::default(),
        }
    }

    /// Compute the microsecond timestamp relative to the connection epoch.
    #[allow(clippy::cast_possible_truncation)]
    fn now_us(&self) -> u32 {
        (self.epoch.elapsed().as_micros() & 0xFFFF_FFFF) as u32
    }

    /// Send data: wraps the payload in an SRT DATA packet, adds to send buffer,
    /// and returns the packet that should be delivered to the network / peer.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload exceeds the configured MSS.
    pub fn send(&mut self, payload: Vec<u8>) -> VideoIpResult<SrtPacket> {
        if payload.len() > self.config.mss as usize {
            return Err(VideoIpError::PacketTooLarge {
                size: payload.len(),
                max: self.config.mss as usize,
            });
        }

        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let ts = self.now_us();
        let pkt = SrtPacket::data(seq, ts, self.config.peer_socket_id, payload);

        // Keep a copy in the send buffer for potential retransmission.
        self.send_buffer.insert(seq, pkt.clone());
        self.stats.packets_sent += 1;
        self.stats.send_buffer_occupancy = self.send_buffer.len();
        Ok(pkt)
    }

    /// Process an ACK packet from the receiver.
    ///
    /// Frees send buffer entries up to (but not including) the ACK sequence.
    pub fn on_ack(&mut self, ack_seq: u32) {
        self.stats.acks_received += 1;
        self.acked_up_to = Some(ack_seq);

        // Drain all packets with sequence < ack_seq from the send buffer.
        self.send_buffer.retain(|&seq, _| {
            // Handle wrap-around: consider a sequence "before" ack_seq if the
            // signed difference is negative.
            let diff = ack_seq.wrapping_sub(seq);
            diff == 0 || diff > 0x8000_0000
        });
        self.stats.send_buffer_occupancy = self.send_buffer.len();
    }

    /// Process a NAK packet containing a list of missing sequence numbers.
    ///
    /// Returns retransmitted packets (to be re-delivered to the peer).
    pub fn on_nak(&mut self, missing_seqs: &[u32]) -> Vec<SrtPacket> {
        self.stats.naks_received += 1;
        let mut retransmits = Vec::new();
        for &seq in missing_seqs {
            if let Some(pkt) = self.send_buffer.get(&seq) {
                retransmits.push(pkt.clone());
                self.stats.packets_retransmitted += 1;
            }
        }
        retransmits
    }

    /// Returns the current sender statistics.
    #[must_use]
    pub fn stats(&self) -> &SrtSenderStats {
        &self.stats
    }

    /// Returns the next sequence number that will be used.
    #[must_use]
    pub fn next_seq(&self) -> u32 {
        self.next_seq
    }

    /// Returns the highest ACKed sequence.
    #[must_use]
    pub fn acked_up_to(&self) -> Option<u32> {
        self.acked_up_to
    }
}

// ---------------------------------------------------------------------------
// SRT Receiver
// ---------------------------------------------------------------------------

/// SRT receiver — buffers incoming packets, generates ACK/NAK, delivers in order.
pub struct SrtReceiver {
    /// Configuration.
    config: SrtTransportConfig,
    /// Out-of-order hold buffer (sequence → packet).
    recv_buffer: HashMap<u32, SrtPacket>,
    /// Packet delivery queue (in-order, ready for application).
    deliver_queue: VecDeque<SrtPacket>,
    /// Next expected sequence number.
    next_expected: Option<u32>,
    /// Loss list — sequences we have seen a gap for and need NAK'd.
    loss_list: Vec<u32>,
    /// Instant of last ACK sent.
    last_ack_time: Instant,
    /// ACK interval.
    ack_interval: Duration,
    /// Epoch for timestamp generation.
    epoch: Instant,
    /// Statistics.
    stats: SrtReceiverStats,
}

impl SrtReceiver {
    /// Create a new SRT receiver.
    #[must_use]
    pub fn new(config: SrtTransportConfig) -> Self {
        Self {
            config,
            recv_buffer: HashMap::new(),
            deliver_queue: VecDeque::new(),
            next_expected: None,
            loss_list: Vec::new(),
            last_ack_time: Instant::now(),
            ack_interval: Duration::from_millis(10),
            epoch: Instant::now(),
            stats: SrtReceiverStats::default(),
        }
    }

    /// Microsecond timestamp relative to connection epoch.
    #[allow(clippy::cast_possible_truncation)]
    fn now_us(&self) -> u32 {
        (self.epoch.elapsed().as_micros() & 0xFFFF_FFFF) as u32
    }

    /// Receive an incoming DATA packet.
    ///
    /// Returns ACK and/or NAK packets that should be sent back to the sender.
    pub fn receive(&mut self, pkt: SrtPacket) -> Vec<SrtPacket> {
        if pkt.header.packet_type != SrtPacketType::Data {
            // Control packets handled elsewhere; ignore here.
            return Vec::new();
        }

        let seq = pkt.header.sequence_number;

        // First packet initializes the expected sequence.
        let expected = match self.next_expected {
            None => {
                self.next_expected = Some(seq.wrapping_add(1));
                self.recv_buffer.insert(seq, pkt.clone());
                self.flush_deliver_queue(seq);
                self.stats.packets_received += 1;
                self.stats.recv_buffer_occupancy = self.recv_buffer.len();
                return self.maybe_send_ack();
            }
            Some(e) => e,
        };

        let gap = seq.wrapping_sub(expected);
        if gap == 0 {
            // In-order packet.
            self.recv_buffer.insert(seq, pkt);
            self.next_expected = Some(seq.wrapping_add(1));
            self.flush_deliver_queue(seq);
            self.stats.packets_received += 1;
        } else if gap < 0x8000_0000 {
            // Future packet — gap means loss.
            // Add missing sequences to loss list.
            for missing in 0..gap {
                let miss_seq = expected.wrapping_add(missing);
                if !self.loss_list.contains(&miss_seq) {
                    self.loss_list.push(miss_seq);
                }
            }
            self.recv_buffer.insert(seq, pkt);
            self.next_expected = Some(seq.wrapping_add(1));
            self.stats.packets_received += 1;
        } else {
            // Past / retransmitted packet — check if it was in the loss list.
            if let Some(pos) = self.loss_list.iter().position(|&s| s == seq) {
                self.loss_list.swap_remove(pos);
                self.recv_buffer.insert(seq, pkt);
                self.stats.packets_recovered += 1;
            } else {
                // Duplicate — discard.
            }
        }

        self.stats.recv_buffer_occupancy = self.recv_buffer.len();

        // Build response packets (ACK and possibly NAK).
        let mut responses = Vec::new();
        let now = Instant::now();
        if now.duration_since(self.last_ack_time) >= self.ack_interval || !self.loss_list.is_empty()
        {
            responses.extend(self.maybe_send_ack());
            if !self.loss_list.is_empty() {
                responses.push(self.build_nak());
            }
            self.last_ack_time = now;
        }
        responses
    }

    /// Flush contiguous in-order packets from recv_buffer to deliver_queue.
    fn flush_deliver_queue(&mut self, from_seq: u32) {
        let mut seq = from_seq;
        loop {
            if let Some(pkt) = self.recv_buffer.remove(&seq) {
                self.deliver_queue.push_back(pkt);
            } else {
                break;
            }
            seq = seq.wrapping_add(1);
        }
    }

    /// Build an ACK packet for the current cumulative receive position.
    fn maybe_send_ack(&mut self) -> Vec<SrtPacket> {
        let ack_seq = match self.next_expected {
            Some(s) => s,
            None => return Vec::new(),
        };
        self.stats.acks_sent += 1;
        vec![SrtPacket::control(
            SrtPacketType::Ack,
            ack_seq,
            self.now_us(),
            self.config.peer_socket_id,
        )]
    }

    /// Build a NAK packet encoding the current loss list.
    fn build_nak(&mut self) -> SrtPacket {
        self.stats.naks_sent += 1;
        // Encode loss list in payload: 4 bytes per missing sequence (big-endian).
        let mut payload = Vec::with_capacity(self.loss_list.len() * 4);
        for &seq in &self.loss_list {
            payload.extend_from_slice(&seq.to_be_bytes());
        }
        SrtPacket {
            header: SrtPacketHeader::control(
                SrtPacketType::Nak,
                0,
                self.now_us(),
                self.config.peer_socket_id,
            ),
            payload,
        }
    }

    /// Parse missing sequences from a NAK payload (as produced by `build_nak`).
    #[must_use]
    pub fn parse_nak_payload(payload: &[u8]) -> Vec<u32> {
        payload
            .chunks_exact(4)
            .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// Pop the next in-order packet from the delivery queue (application read).
    #[must_use]
    pub fn pop_packet(&mut self) -> Option<SrtPacket> {
        self.deliver_queue.pop_front()
    }

    /// Returns the current loss list length.
    #[must_use]
    pub fn loss_list_len(&self) -> usize {
        self.loss_list.len()
    }

    /// Returns the current receiver statistics.
    #[must_use]
    pub fn stats(&self) -> &SrtReceiverStats {
        &self.stats
    }

    /// Returns the next expected sequence number.
    #[must_use]
    pub fn next_expected(&self) -> Option<u32> {
        self.next_expected
    }
}

// ---------------------------------------------------------------------------
// SRT Session (loopback / in-memory pair)
// ---------------------------------------------------------------------------

/// A paired SRT sender + receiver for in-memory testing or loopback use.
pub struct SrtSession {
    /// Sender side.
    pub sender: SrtSender,
    /// Receiver side.
    pub receiver: SrtReceiver,
}

impl SrtSession {
    /// Create a new in-memory SRT session.
    #[must_use]
    pub fn new(sender_cfg: SrtTransportConfig, receiver_cfg: SrtTransportConfig) -> Self {
        Self {
            sender: SrtSender::new(sender_cfg),
            receiver: SrtReceiver::new(receiver_cfg),
        }
    }

    /// Send a payload through the session (losslessly).
    ///
    /// The packet is given to the receiver immediately, and any feedback
    /// (ACK/NAK) is fed back to the sender.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload is too large.
    pub fn send_lossless(&mut self, payload: Vec<u8>) -> VideoIpResult<()> {
        let pkt = self.sender.send(payload)?;
        let feedback = self.receiver.receive(pkt);
        for fb in feedback {
            match fb.header.packet_type {
                SrtPacketType::Ack => {
                    self.sender.on_ack(fb.header.sequence_number);
                }
                SrtPacketType::Nak => {
                    let missing = SrtReceiver::parse_nak_payload(&fb.payload);
                    let retransmits = self.sender.on_nak(&missing);
                    for rt in retransmits {
                        self.receiver.receive(rt);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Send a payload but simulate dropping it (testing retransmit path).
    ///
    /// The packet is NOT given to the receiver; instead the receiver will
    /// detect a gap and NAK it on the next delivery, triggering retransmit.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload is too large.
    pub fn send_with_drop(&mut self, payload: Vec<u8>) -> VideoIpResult<u32> {
        let pkt = self.sender.send(payload)?;
        Ok(pkt.header.sequence_number)
    }

    /// Deliver a subsequent lossless packet and let the gap detection trigger
    /// a retransmit for the previously dropped sequence(s).
    ///
    /// # Errors
    ///
    /// Returns an error if the payload is too large.
    pub fn send_lossless_after_drop(&mut self, payload: Vec<u8>) -> VideoIpResult<()> {
        let pkt = self.sender.send(payload)?;
        let feedback = self.receiver.receive(pkt);
        for fb in &feedback {
            if fb.header.packet_type == SrtPacketType::Nak {
                let missing = SrtReceiver::parse_nak_payload(&fb.payload);
                let retransmits = self.sender.on_nak(&missing);
                for rt in retransmits {
                    self.receiver.receive(rt);
                }
            }
        }
        // Feed remaining ACKs.
        for fb in feedback {
            if fb.header.packet_type == SrtPacketType::Ack {
                self.sender.on_ack(fb.header.sequence_number);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> SrtTransportConfig {
        SrtTransportConfig::default()
    }

    // --- Config validation ---

    #[test]
    fn test_config_default_is_valid() {
        // Default config has socket_id set but no remote (not needed for in-memory)
        let cfg = default_cfg();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_latency_too_large_fails() {
        let cfg = SrtTransportConfig {
            latency_ms: 31_000,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_passphrase_too_short_fails() {
        let cfg = SrtTransportConfig {
            passphrase: Some("tooshort".into()),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_passphrase_valid() {
        let cfg = SrtTransportConfig {
            passphrase: Some("validpassphrase1234".into()),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    // --- Handshake state machine ---

    #[test]
    fn test_handshake_full_exchange() {
        let caller_cfg = SrtTransportConfig {
            socket_id: 0xAAAA,
            peer_socket_id: 0xBBBB,
            latency_ms: 200,
            ..Default::default()
        };
        let responder_socket_id = 0xBBBB_u32;

        let mut caller = SrtHandshake::new(caller_cfg);

        // Step 1: caller sends INDUCTION
        let induction = caller.initiate();
        assert_eq!(induction.phase, SrtHandshakePhase::Induction);
        assert_eq!(caller.phase(), SrtHandshakePhase::WaveAHand);

        // Step 2: responder replies with WAVEAHAND
        let waveahand = induction.waveahand(responder_socket_id);
        assert_eq!(waveahand.phase, SrtHandshakePhase::WaveAHand);

        // Step 3: caller processes WAVEAHAND → produces CONCLUSION
        let reply = caller.step(&waveahand).expect("should produce CONCLUSION");
        let conclusion = reply.expect("should be Some");
        assert_eq!(conclusion.phase, SrtHandshakePhase::Conclusion);
        assert_eq!(caller.phase(), SrtHandshakePhase::Conclusion);

        // Step 4: responder replies with AGREEMENT
        let agreement = conclusion.agreement();
        assert_eq!(agreement.phase, SrtHandshakePhase::Agreement);

        // Step 5: caller processes AGREEMENT → handshake complete
        let no_reply = caller.step(&agreement).expect("should complete");
        assert!(no_reply.is_none());
        assert!(caller.is_complete());
    }

    #[test]
    fn test_handshake_latency_negotiation_takes_max() {
        let caller_cfg = SrtTransportConfig {
            socket_id: 0x01,
            peer_socket_id: 0x02,
            latency_ms: 120,
            ..Default::default()
        };
        let mut caller = SrtHandshake::new(caller_cfg);
        let induction = caller.initiate();
        // Peer has higher latency requirement
        let mut waveahand = induction.waveahand(0x02);
        waveahand.latency_ms = 300;
        let _ = caller.step(&waveahand).expect("ok");
        assert_eq!(caller.negotiated_latency_ms(), 300);
    }

    #[test]
    fn test_handshake_wrong_phase_fails() {
        let cfg = SrtTransportConfig::default();
        let mut caller = SrtHandshake::new(cfg);
        let _ = caller.initiate();
        // Send a CONCLUSION when expecting WAVEAHAND
        let wrong = SrtHandshakeMsg {
            phase: SrtHandshakePhase::Conclusion,
            socket_id: 0x02,
            peer_socket_id: 0x01,
            latency_ms: 120,
            version: SrtHandshakeMsg::SRT_VERSION,
            stream_id: None,
        };
        assert!(caller.step(&wrong).is_err());
        assert_eq!(caller.phase(), SrtHandshakePhase::Failed);
    }

    #[test]
    fn test_handshake_stream_id_propagates() {
        let cfg = SrtTransportConfig {
            socket_id: 0x01,
            peer_socket_id: 0x02,
            ..Default::default()
        };
        let mut caller = SrtHandshake::new(cfg);
        caller.set_stream_id("camera/main");
        let induction = caller.initiate();
        let waveahand = induction.waveahand(0x02);
        let conclusion_opt = caller.step(&waveahand).expect("ok");
        let conclusion = conclusion_opt.expect("some");
        assert_eq!(conclusion.stream_id.as_deref(), Some("camera/main"));
    }

    // --- SRT packet types ---

    #[test]
    fn test_packet_type_discrimination() {
        assert!(SrtPacketType::Data.is_data());
        assert!(!SrtPacketType::Data.is_control());
        assert!(SrtPacketType::Ack.is_control());
        assert!(SrtPacketType::Nak.is_control());
    }

    #[test]
    fn test_packet_type_labels() {
        assert_eq!(SrtPacketType::Data.label(), "DATA");
        assert_eq!(SrtPacketType::Ack.label(), "ACK");
        assert_eq!(SrtPacketType::Nak.label(), "NAK");
        assert_eq!(SrtPacketType::KeepAlive.label(), "KEEPALIVE");
        assert_eq!(SrtPacketType::Shutdown.label(), "SHUTDOWN");
    }

    #[test]
    fn test_packet_header_construction() {
        let h = SrtPacketHeader::data(42, 12345, 0xFF);
        assert_eq!(h.packet_type, SrtPacketType::Data);
        assert_eq!(h.sequence_number, 42);
        assert_eq!(h.timestamp, 12345);
        assert_eq!(h.destination_socket_id, 0xFF);
    }

    // --- SRT Sender ---

    #[test]
    fn test_sender_send_increments_seq() {
        let mut sender = SrtSender::new(default_cfg());
        let p1 = sender.send(vec![1, 2, 3]).expect("ok");
        let p2 = sender.send(vec![4, 5, 6]).expect("ok");
        assert_eq!(p1.header.sequence_number, 0);
        assert_eq!(p2.header.sequence_number, 1);
        assert_eq!(sender.stats().packets_sent, 2);
    }

    #[test]
    fn test_sender_payload_too_large_fails() {
        let mut sender = SrtSender::new(default_cfg());
        let huge = vec![0u8; 2000]; // exceeds default MSS of 1316
        assert!(sender.send(huge).is_err());
    }

    #[test]
    fn test_sender_ack_frees_buffer() {
        let mut sender = SrtSender::new(default_cfg());
        sender.send(vec![1]).expect("ok");
        sender.send(vec![2]).expect("ok");
        sender.send(vec![3]).expect("ok");
        assert_eq!(sender.stats().send_buffer_occupancy, 3);
        sender.on_ack(2); // ACKs seqs 0 and 1; seq 2 stays
        assert_eq!(sender.stats().send_buffer_occupancy, 1);
    }

    #[test]
    fn test_sender_nak_triggers_retransmit() {
        let mut sender = SrtSender::new(default_cfg());
        sender.send(vec![10]).expect("ok");
        sender.send(vec![20]).expect("ok");
        let retransmits = sender.on_nak(&[0]);
        assert_eq!(retransmits.len(), 1);
        assert_eq!(retransmits[0].payload, vec![10]);
        assert_eq!(sender.stats().packets_retransmitted, 1);
    }

    // --- SRT Session loopback ---

    #[test]
    fn test_session_lossless_send_and_receive() {
        let mut session = SrtSession::new(default_cfg(), default_cfg());
        session.send_lossless(b"hello world".to_vec()).expect("ok");
        let received = session.receiver.pop_packet();
        assert!(received.is_some());
        assert_eq!(received.expect("some").payload, b"hello world".to_vec());
    }

    #[test]
    fn test_session_loss_and_recovery() {
        // Test the retransmission path directly via the sender's on_nak API.
        // This exercises the full NAK → retransmit → deliver loop.
        let mut sender = SrtSender::new(default_cfg());
        let mut receiver = SrtReceiver::new(default_cfg());

        // Send seq 0 (delivered to receiver so it establishes next_expected = 1)
        let pkt0 = sender.send(b"seq0".to_vec()).expect("ok");
        receiver.receive(pkt0);

        // Send seq 1 — but simulate it being lost (don't give to receiver)
        let _lost_pkt1 = sender.send(b"seq1-dropped".to_vec()).expect("ok");
        let lost_seq = 1u32;

        // Send seq 2 — receiver detects gap for seq 1 and issues NAK
        let pkt2 = sender.send(b"seq2".to_vec()).expect("ok");
        let feedback = receiver.receive(pkt2);

        // Find NAK in feedback
        let nak_fb = feedback
            .iter()
            .find(|p| p.header.packet_type == SrtPacketType::Nak);
        assert!(
            nak_fb.is_some(),
            "Receiver should have sent a NAK for missing seq 1"
        );
        assert!(
            receiver.loss_list_len() > 0,
            "Loss list should contain seq 1"
        );

        // Feed NAK back to sender and get retransmits
        let missing_seqs = SrtReceiver::parse_nak_payload(&nak_fb.expect("nak").payload);
        assert!(
            missing_seqs.contains(&lost_seq),
            "NAK should contain seq {lost_seq}"
        );

        let retransmits = sender.on_nak(&missing_seqs);
        assert_eq!(retransmits.len(), 1, "Sender should retransmit 1 packet");
        assert_eq!(retransmits[0].payload, b"seq1-dropped".to_vec());

        // Deliver retransmit to receiver
        receiver.receive(retransmits[0].clone());
        assert!(
            receiver.stats().packets_recovered > 0,
            "Recovery count should increment"
        );
        assert_eq!(sender.stats().packets_retransmitted, 1);
    }

    #[test]
    fn test_nak_payload_roundtrip() {
        let missing = vec![5u32, 10, 15, 20];
        let mut payload = Vec::new();
        for seq in &missing {
            payload.extend_from_slice(&seq.to_be_bytes());
        }
        let parsed = SrtReceiver::parse_nak_payload(&payload);
        assert_eq!(parsed, missing);
    }

    #[test]
    fn test_receiver_stats_tracking() {
        let mut session = SrtSession::new(default_cfg(), default_cfg());
        for i in 0..5_u8 {
            session.send_lossless(vec![i]).expect("ok");
        }
        assert_eq!(session.receiver.stats().packets_received, 5);
    }
}

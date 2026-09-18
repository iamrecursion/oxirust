//! SRT socket state machine and configuration.
//!
//! This module provides the SRT socket abstraction.

use super::packet::{ControlPacket, ControlType, DataPacket, HandshakeInfo, SrtPacket};
use crate::error::{NetError, NetResult};
use bytes::Bytes;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// SRT connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Initial state, not connected.
    Initial,
    /// Handshake in progress (caller sent initial handshake).
    Handshaking,
    /// Connection established.
    Connected,
    /// Closing in progress.
    Closing,
    /// Connection closed.
    Closed,
    /// Connection broken/error.
    Broken,
}

impl ConnectionState {
    /// Returns true if connected.
    #[must_use]
    pub const fn is_connected(&self) -> bool {
        matches!(self, Self::Connected)
    }

    /// Returns true if closed or broken.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self, Self::Closed | Self::Broken)
    }
}

/// SRT configuration options.
#[derive(Debug, Clone)]
pub struct SrtConfig {
    /// Maximum transmission unit (default: 1500).
    pub mtu: u32,
    /// Flow control window size (default: 8192).
    pub flow_window: u32,
    /// Latency in milliseconds (default: 120).
    pub latency_ms: u32,
    /// Peer latency in milliseconds.
    pub peer_latency_ms: u32,
    /// Too late packet drop (default: true).
    pub too_late_drop: bool,
    /// Connection timeout (default: 3 seconds).
    pub connect_timeout: Duration,
    /// Peer idle timeout (default: 5 seconds).
    pub peer_idle_timeout: Duration,
    /// Maximum bandwidth (0 = infinite).
    pub max_bandwidth: u64,
    /// Encryption key length (0, 16, 24, 32).
    pub key_size: u8,
    /// Stream ID.
    pub stream_id: Option<String>,
    /// Passphrase for encryption.
    pub passphrase: Option<String>,
}

impl Default for SrtConfig {
    fn default() -> Self {
        Self {
            mtu: 1500,
            flow_window: 8192,
            latency_ms: 120,
            peer_latency_ms: 0,
            too_late_drop: true,
            connect_timeout: Duration::from_secs(3),
            peer_idle_timeout: Duration::from_secs(5),
            max_bandwidth: 0,
            key_size: 0,
            stream_id: None,
            passphrase: None,
        }
    }
}

impl SrtConfig {
    /// Creates a new default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the latency.
    #[must_use]
    pub const fn with_latency(mut self, latency_ms: u32) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Sets the MTU.
    #[must_use]
    pub const fn with_mtu(mut self, mtu: u32) -> Self {
        self.mtu = mtu;
        self
    }

    /// Sets the stream ID.
    #[must_use]
    pub fn with_stream_id(mut self, stream_id: impl Into<String>) -> Self {
        self.stream_id = Some(stream_id.into());
        self
    }

    /// Sets the passphrase for encryption.
    #[must_use]
    pub fn with_passphrase(mut self, passphrase: impl Into<String>) -> Self {
        self.passphrase = Some(passphrase.into());
        self.key_size = 16; // AES-128 default
        self
    }

    /// Sets the key size (16, 24, or 32 for AES-128/192/256).
    #[must_use]
    pub const fn with_key_size(mut self, key_size: u8) -> Self {
        self.key_size = key_size;
        self
    }
}

/// Packet waiting for acknowledgement.
#[derive(Debug, Clone)]
struct UnackedPacket {
    /// Packet data.
    packet: DataPacket,
    /// Time sent.
    sent_at: Instant,
    /// Number of retransmissions.
    retransmit_count: u32,
}

/// SRT statistics.
#[derive(Debug, Clone, Default)]
pub struct SrtStats {
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total packets received.
    pub packets_received: u64,
    /// Total packets retransmitted.
    pub packets_retransmitted: u64,
    /// Total packets lost.
    pub packets_lost: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Current send buffer size.
    pub send_buffer_size: usize,
    /// Current receive buffer size.
    pub recv_buffer_size: usize,
}

/// SRT socket state machine.
#[derive(Debug)]
pub struct SrtSocket {
    /// Socket ID.
    socket_id: u32,
    /// Peer socket ID.
    peer_socket_id: u32,
    /// Connection state.
    state: ConnectionState,
    /// Configuration.
    config: SrtConfig,
    /// Next sequence number to send.
    pub(crate) send_seq: u32,
    /// Next sequence number expected to receive.
    recv_seq: u32,
    /// Last ACK sent.
    last_ack_sent: u32,
    /// Last ACK received.
    last_ack_recv: u32,
    /// Packets waiting for ACK.
    unacked_packets: VecDeque<UnackedPacket>,
    /// Received packets buffer (out of order).
    recv_buffer: VecDeque<DataPacket>,
    /// RTT estimate (microseconds).
    rtt: u32,
    /// RTT variance.
    rtt_var: u32,
    /// Last activity time.
    last_activity: Instant,
    /// Connection start time.
    start_time: Instant,
    /// Handshake info.
    handshake: HandshakeInfo,
    /// Statistics.
    stats: SrtStats,
}

impl SrtSocket {
    /// Creates a new SRT socket.
    #[must_use]
    pub fn new(config: SrtConfig) -> Self {
        let now = Instant::now();
        Self {
            socket_id: rand_socket_id(),
            peer_socket_id: 0,
            state: ConnectionState::Initial,
            config,
            send_seq: rand_initial_seq(),
            recv_seq: 0,
            last_ack_sent: 0,
            last_ack_recv: 0,
            unacked_packets: VecDeque::new(),
            recv_buffer: VecDeque::new(),
            rtt: 100_000, // 100ms initial
            rtt_var: 50_000,
            last_activity: now,
            start_time: now,
            handshake: HandshakeInfo::new(),
            stats: SrtStats::default(),
        }
    }

    /// Returns the socket ID.
    #[must_use]
    pub const fn socket_id(&self) -> u32 {
        self.socket_id
    }

    /// Returns the peer socket ID.
    #[must_use]
    pub const fn peer_socket_id(&self) -> u32 {
        self.peer_socket_id
    }

    /// Returns the current connection state.
    #[must_use]
    pub const fn state(&self) -> ConnectionState {
        self.state
    }

    /// Returns true if connected.
    #[must_use]
    pub const fn is_connected(&self) -> bool {
        self.state.is_connected()
    }

    /// Returns the configuration.
    #[must_use]
    pub const fn config(&self) -> &SrtConfig {
        &self.config
    }

    /// Returns the current RTT estimate in microseconds.
    #[must_use]
    pub const fn rtt(&self) -> u32 {
        self.rtt
    }

    /// Returns time since connection start.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Returns the current timestamp (microseconds since start).
    #[must_use]
    pub fn current_timestamp(&self) -> u32 {
        self.start_time.elapsed().as_micros() as u32
    }

    /// Generates initial handshake packet (caller).
    #[must_use]
    pub fn generate_caller_handshake(&mut self) -> SrtPacket {
        self.handshake = HandshakeInfo {
            version: 5,
            mtu: self.config.mtu,
            flow_window: self.config.flow_window,
            handshake_type: HandshakeInfo::TYPE_WAVEAHAND,
            socket_id: self.socket_id,
            initial_seq: self.send_seq,
            ..Default::default()
        };

        self.state = ConnectionState::Handshaking;
        SrtPacket::Control(ControlPacket::handshake(&self.handshake, 0))
    }

    /// Processes a received packet.
    ///
    /// Returns packets to send in response.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is malformed or unexpected.
    pub fn process_packet(&mut self, packet: SrtPacket) -> NetResult<Vec<SrtPacket>> {
        self.last_activity = Instant::now();
        let mut responses = Vec::new();

        match packet {
            SrtPacket::Data(data) => {
                if !self.is_connected() {
                    return Err(NetError::invalid_state("Not connected"));
                }
                self.process_data_packet(data, &mut responses)?;
            }
            SrtPacket::Control(ctrl) => {
                self.process_control_packet(ctrl, &mut responses)?;
            }
        }

        Ok(responses)
    }

    fn process_data_packet(
        &mut self,
        packet: DataPacket,
        responses: &mut Vec<SrtPacket>,
    ) -> NetResult<()> {
        let seq = packet.sequence_number;
        let payload_len = packet.payload.len() as u64;

        // Update statistics
        self.stats.packets_received += 1;
        self.stats.bytes_received += payload_len;

        // Check if this is the expected sequence
        if seq == self.recv_seq {
            self.recv_seq = seq.wrapping_add(1);
            self.recv_buffer.push_back(packet);

            // Check for consecutive buffered packets
            while let Some(buffered) = self.recv_buffer.front() {
                if buffered.sequence_number == self.recv_seq {
                    self.recv_seq = self.recv_seq.wrapping_add(1);
                    self.recv_buffer.pop_front();
                } else {
                    break;
                }
            }
        } else if seq_after(seq, self.recv_seq) {
            // Out of order - buffer it
            self.recv_buffer.push_back(packet);
        }
        // else: duplicate or old packet, ignore

        // Update buffer size stats
        self.stats.recv_buffer_size = self.recv_buffer.len();

        // Send ACK periodically (simplified: every packet for now)
        if self.recv_seq != self.last_ack_sent {
            let ack = ControlPacket::ack(self.recv_seq, self.peer_socket_id)
                .with_timestamp(self.current_timestamp());
            responses.push(SrtPacket::Control(ack));
            self.last_ack_sent = self.recv_seq;
        }

        Ok(())
    }

    fn process_control_packet(
        &mut self,
        packet: ControlPacket,
        responses: &mut Vec<SrtPacket>,
    ) -> NetResult<()> {
        match packet.control_type {
            ControlType::Handshake => {
                self.process_handshake(&packet, responses)?;
            }
            ControlType::Keepalive => {
                // Respond with keepalive
                let keepalive = ControlPacket::keepalive(self.peer_socket_id)
                    .with_timestamp(self.current_timestamp());
                responses.push(SrtPacket::Control(keepalive));
            }
            ControlType::Ack => {
                let ack_seq = packet.type_info;
                self.last_ack_recv = ack_seq;
                // Remove acknowledged packets
                while let Some(front) = self.unacked_packets.front() {
                    if seq_after(ack_seq, front.packet.sequence_number) {
                        self.unacked_packets.pop_front();
                    } else {
                        break;
                    }
                }
                // Send ACK-ACK
                let ack_ack = ControlPacket::new(ControlType::AckAck)
                    .with_timestamp(self.current_timestamp());
                responses.push(SrtPacket::Control(ack_ack));
            }
            ControlType::Nak => {
                // Retransmit lost packets
                self.handle_nak(&packet)?;
            }
            ControlType::Shutdown => {
                self.state = ConnectionState::Closed;
            }
            _ => {
                // Ignore other control types for now
            }
        }

        Ok(())
    }

    fn process_handshake(
        &mut self,
        packet: &ControlPacket,
        responses: &mut Vec<SrtPacket>,
    ) -> NetResult<()> {
        let hs = HandshakeInfo::decode(&packet.payload)?;

        match self.state {
            ConnectionState::Initial => {
                // Listener receiving initial handshake
                self.peer_socket_id = hs.socket_id;
                self.recv_seq = hs.initial_seq;

                let response = HandshakeInfo {
                    version: 5,
                    mtu: self.config.mtu.min(hs.mtu),
                    flow_window: self.config.flow_window.min(hs.flow_window),
                    handshake_type: HandshakeInfo::TYPE_INDUCTION,
                    socket_id: self.socket_id,
                    initial_seq: self.send_seq,
                    syn_cookie: generate_cookie(),
                    ..Default::default()
                };

                responses.push(SrtPacket::Control(ControlPacket::handshake(
                    &response,
                    self.peer_socket_id,
                )));
                self.state = ConnectionState::Handshaking;
            }
            ConnectionState::Handshaking => {
                if hs.handshake_type == HandshakeInfo::TYPE_INDUCTION
                    || hs.handshake_type == HandshakeInfo::TYPE_CONCLUSION
                {
                    // Caller received response
                    self.peer_socket_id = hs.socket_id;
                    self.recv_seq = hs.initial_seq;
                    self.config.mtu = self.config.mtu.min(hs.mtu);
                    self.config.flow_window = self.config.flow_window.min(hs.flow_window);

                    if hs.handshake_type == HandshakeInfo::TYPE_INDUCTION {
                        // Send conclusion
                        let conclusion = HandshakeInfo {
                            version: 5,
                            mtu: self.config.mtu,
                            flow_window: self.config.flow_window,
                            handshake_type: HandshakeInfo::TYPE_CONCLUSION,
                            socket_id: self.socket_id,
                            initial_seq: self.send_seq,
                            syn_cookie: hs.syn_cookie,
                            ..Default::default()
                        };
                        responses.push(SrtPacket::Control(ControlPacket::handshake(
                            &conclusion,
                            self.peer_socket_id,
                        )));
                    }

                    self.state = ConnectionState::Connected;
                } else if hs.handshake_type == HandshakeInfo::TYPE_AGREEMENT {
                    self.state = ConnectionState::Connected;
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn handle_nak(&mut self, _packet: &ControlPacket) -> NetResult<()> {
        // Mark packets for retransmission
        // (Full implementation would parse NAK payload for lost sequence numbers)
        Ok(())
    }

    /// Creates a data packet for sending.
    #[must_use]
    pub fn create_data_packet(&mut self, payload: Bytes) -> DataPacket {
        let seq = self.send_seq;
        self.send_seq = self.send_seq.wrapping_add(1);

        // Update statistics
        self.stats.packets_sent += 1;
        self.stats.bytes_sent += payload.len() as u64;

        DataPacket::new(seq, payload)
            .with_timestamp(self.current_timestamp())
            .with_dst_socket(self.peer_socket_id)
    }

    /// Closes the connection.
    pub fn close(&mut self) -> Option<SrtPacket> {
        if self.state.is_connected() {
            self.state = ConnectionState::Closing;
            Some(SrtPacket::Control(ControlPacket::shutdown(
                self.peer_socket_id,
            )))
        } else {
            self.state = ConnectionState::Closed;
            None
        }
    }

    /// Checks for timeout conditions.
    #[must_use]
    pub fn check_timeout(&self) -> bool {
        self.last_activity.elapsed() > self.config.peer_idle_timeout
    }

    /// Returns current statistics.
    #[must_use]
    pub fn stats(&self) -> &SrtStats {
        &self.stats
    }

    /// Updates RTT estimate with a new sample.
    pub fn update_rtt(&mut self, sample: u32) {
        // Exponential weighted moving average
        if self.rtt == 0 {
            self.rtt = sample;
            self.rtt_var = sample / 2;
        } else {
            let diff = if sample > self.rtt {
                sample - self.rtt
            } else {
                self.rtt - sample
            };
            self.rtt_var = (3 * self.rtt_var + diff) / 4;
            self.rtt = (7 * self.rtt + sample) / 8;
        }
    }

    /// Marks a packet for retransmission.
    pub fn mark_for_retransmit(&mut self, seq: u32) {
        for entry in &mut self.unacked_packets {
            if entry.packet.sequence_number == seq {
                entry.retransmit_count += 1;
                self.stats.packets_retransmitted += 1;
                break;
            }
        }
    }
}

/// Checks if seq a is after seq b (with wraparound).
const fn seq_after(a: u32, b: u32) -> bool {
    let diff = a.wrapping_sub(b);
    diff > 0 && diff < 0x8000_0000
}

/// Generates a random socket ID.
fn rand_socket_id() -> u32 {
    // Simple PRNG - in production use proper random
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(12345);
    seed ^ 0xDEAD_BEEF
}

/// Generates a random initial sequence number.
fn rand_initial_seq() -> u32 {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(54321);
    seed & 0x7FFF_FFFF
}

/// Generates a SYN cookie.
fn generate_cookie() -> u32 {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(0);
    seed ^ 0xCAFE_BABE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_state() {
        assert!(ConnectionState::Connected.is_connected());
        assert!(!ConnectionState::Initial.is_connected());
        assert!(ConnectionState::Closed.is_finished());
        assert!(ConnectionState::Broken.is_finished());
    }

    #[test]
    fn test_srt_config() {
        let config = SrtConfig::new()
            .with_latency(200)
            .with_mtu(1400)
            .with_stream_id("mystream");

        assert_eq!(config.latency_ms, 200);
        assert_eq!(config.mtu, 1400);
        assert_eq!(config.stream_id, Some("mystream".to_string()));
    }

    #[test]
    fn test_srt_socket_new() {
        let socket = SrtSocket::new(SrtConfig::default());
        assert_eq!(socket.state(), ConnectionState::Initial);
        assert!(!socket.is_connected());
    }

    #[test]
    fn test_caller_handshake() {
        let mut socket = SrtSocket::new(SrtConfig::default());
        let packet = socket.generate_caller_handshake();

        assert_eq!(socket.state(), ConnectionState::Handshaking);
        assert!(packet.is_control());
    }

    #[test]
    fn test_create_data_packet() {
        let mut socket = SrtSocket::new(SrtConfig::default());
        socket.state = ConnectionState::Connected;
        socket.peer_socket_id = 100;

        let packet1 = socket.create_data_packet(Bytes::from(vec![1, 2, 3]));
        let packet2 = socket.create_data_packet(Bytes::from(vec![4, 5, 6]));

        assert_eq!(packet2.sequence_number, packet1.sequence_number + 1);
        assert_eq!(packet1.dst_socket_id, 100);
    }

    #[test]
    fn test_seq_after() {
        assert!(seq_after(10, 5));
        assert!(!seq_after(5, 10));
        assert!(!seq_after(5, 5));

        // Wraparound
        assert!(seq_after(0, 0xFFFF_FFFF));
    }

    #[test]
    fn test_close() {
        let mut socket = SrtSocket::new(SrtConfig::default());
        socket.state = ConnectionState::Connected;
        socket.peer_socket_id = 42;

        let packet = socket.close();
        assert!(packet.is_some());
        assert_eq!(socket.state(), ConnectionState::Closing);

        if let Some(SrtPacket::Control(ctrl)) = packet {
            assert_eq!(ctrl.control_type, ControlType::Shutdown);
        }
    }
}

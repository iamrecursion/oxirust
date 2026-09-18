//! AES67 audio-over-IP interoperability module.
//!
//! AES67 defines a standard for high-performance audio transport over IP networks,
//! compatible with IEEE 1722, SMPTE ST 2110-30, and Ravenna. This module provides
//! RTP packetization, SDP session description, and stream management.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from AES67 operations.
#[derive(Debug, Error)]
pub enum Aes67Error {
    /// Packet time value is not a valid AES67 standard packet time.
    #[error("invalid packet time: {0} µs")]
    InvalidPacketTime(u32),
    /// Sample rate is not supported by AES67.
    #[error("invalid sample rate: {0} Hz")]
    InvalidSampleRate(u32),
    /// Buffer is too small for the requested operation.
    #[error("buffer too small: need {needed}, have {have}")]
    BufferTooSmall { needed: usize, have: usize },
    /// RTP sequence number discontinuity detected.
    #[error("invalid sequence: expected {expected}, got {got}")]
    InvalidSequence { expected: u16, got: u16 },
    /// Malformed RTP packet.
    #[error("malformed packet: {reason}")]
    MalformedPacket { reason: String },
}

// ---------------------------------------------------------------------------
// PacketTimeUs — standard AES67 packet times
// ---------------------------------------------------------------------------

/// Standard AES67 packet times (inter-packet gap).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PacketTimeUs {
    /// 125 µs — 6 samples at 48 kHz (ST 2110-30 Class A).
    Us125,
    /// 250 µs — 12 samples at 48 kHz.
    Us250,
    /// 333 µs — 16 samples at 48 kHz (approx).
    Us333,
    /// 1 ms — 48 samples at 48 kHz (broadcast default).
    Us1000,
    /// 4 ms — 192 samples at 48 kHz (low-bandwidth).
    Us4000,
}

impl PacketTimeUs {
    /// Microseconds for this packet time.
    pub fn as_us(&self) -> u32 {
        match self {
            Self::Us125 => 125,
            Self::Us250 => 250,
            Self::Us333 => 333,
            Self::Us1000 => 1_000,
            Self::Us4000 => 4_000,
        }
    }

    /// Number of samples per packet at the given sample rate.
    pub fn samples_per_packet(&self, sample_rate: u32) -> usize {
        // samples = (sample_rate * packet_time_us) / 1_000_000
        (sample_rate as u64 * self.as_us() as u64 / 1_000_000) as usize
    }

    /// Parses a packet time from microseconds.
    pub fn from_us(us: u32) -> Result<Self, Aes67Error> {
        match us {
            125 => Ok(Self::Us125),
            250 => Ok(Self::Us250),
            333 => Ok(Self::Us333),
            1_000 => Ok(Self::Us1000),
            4_000 => Ok(Self::Us4000),
            other => Err(Aes67Error::InvalidPacketTime(other)),
        }
    }

    /// Returns the SDP ptime string (in milliseconds, possibly fractional).
    pub fn sdp_ptime(&self) -> &'static str {
        match self {
            Self::Us125 => "0.125",
            Self::Us250 => "0.25",
            Self::Us333 => "0.333",
            Self::Us1000 => "1",
            Self::Us4000 => "4",
        }
    }
}

impl fmt::Display for PacketTimeUs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} µs", self.as_us())
    }
}

// ---------------------------------------------------------------------------
// Aes67StreamConfig
// ---------------------------------------------------------------------------

/// Supported sample rates for AES67.
fn is_valid_sample_rate(rate: u32) -> bool {
    matches!(rate, 32_000 | 44_100 | 48_000 | 88_200 | 96_000 | 192_000)
}

/// Supported bit depths.
fn is_valid_bit_depth(depth: u8) -> bool {
    matches!(depth, 16 | 24 | 32)
}

/// AES67 stream configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aes67StreamConfig {
    /// Sample rate in Hz (32000, 44100, 48000, 88200, 96000, 192000).
    pub sample_rate: u32,
    /// Bit depth per sample (16, 24, or 32).
    pub bit_depth: u8,
    /// Number of audio channels.
    pub channels: u8,
    /// Packet time.
    pub packet_time: PacketTimeUs,
    /// Multicast group address (IPv4 or IPv6 as string).
    pub multicast_address: String,
    /// UDP destination port.
    pub port: u16,
    /// RTP payload type (96–127 for dynamic).
    pub payload_type: u8,
    /// DSCP value for QoS (typically EF = 46 for audio).
    pub dscp: u8,
    /// SSRC — synchronisation source identifier.
    pub ssrc: u32,
    /// Stream name for SDP.
    pub stream_name: String,
}

impl Aes67StreamConfig {
    /// Creates a new AES67 stream config with broadcast defaults.
    ///
    /// Returns an error if sample rate or bit depth are invalid.
    pub fn new(
        stream_name: impl Into<String>,
        sample_rate: u32,
        bit_depth: u8,
        channels: u8,
        ssrc: u32,
    ) -> Result<Self, Aes67Error> {
        if !is_valid_sample_rate(sample_rate) {
            return Err(Aes67Error::InvalidSampleRate(sample_rate));
        }
        if !is_valid_bit_depth(bit_depth) {
            return Err(Aes67Error::MalformedPacket {
                reason: format!("unsupported bit depth: {}", bit_depth),
            });
        }
        Ok(Self {
            sample_rate,
            bit_depth,
            channels,
            packet_time: PacketTimeUs::Us1000,
            multicast_address: "239.69.0.1".to_string(),
            port: 5004,
            payload_type: 96,
            dscp: 46,
            ssrc,
            stream_name: stream_name.into(),
        })
    }

    /// Bytes per sample (bit_depth / 8).
    pub fn bytes_per_sample(&self) -> usize {
        (self.bit_depth as usize + 7) / 8
    }

    /// Samples per packet based on packet time and sample rate.
    pub fn samples_per_packet(&self) -> usize {
        self.packet_time.samples_per_packet(self.sample_rate)
    }

    /// Bytes per packet for all channels.
    pub fn payload_bytes_per_packet(&self) -> usize {
        self.samples_per_packet() * self.channels as usize * self.bytes_per_sample()
    }

    /// RTP timestamp increment per packet (equals samples_per_packet).
    pub fn rtp_timestamp_increment(&self) -> u32 {
        self.samples_per_packet() as u32
    }
}

// ---------------------------------------------------------------------------
// Aes67PacketHeader — RTP header fields
// ---------------------------------------------------------------------------

/// Minimal RTP-based packet header for AES67.
#[derive(Debug, Clone, Copy)]
pub struct Aes67PacketHeader {
    /// RTP sequence number (16-bit, wraps around).
    pub sequence_number: u16,
    /// RTP timestamp in sample units.
    pub timestamp: u32,
    /// Synchronisation source identifier.
    pub ssrc: u32,
    /// RTP payload type.
    pub payload_type: u8,
    /// Marker bit (set on first packet after silence).
    pub marker: bool,
}

impl Aes67PacketHeader {
    /// Serialized size of the RTP header in bytes (fixed 12 bytes for AES67).
    pub const SERIALIZED_SIZE: usize = 12;

    /// Serializes the header into a 12-byte RTP header.
    pub fn serialize(&self, buf: &mut [u8]) -> Result<(), Aes67Error> {
        if buf.len() < Self::SERIALIZED_SIZE {
            return Err(Aes67Error::BufferTooSmall {
                needed: Self::SERIALIZED_SIZE,
                have: buf.len(),
            });
        }
        // Octet 0: V=2, P=0, X=0, CC=0
        buf[0] = 0x80;
        // Octet 1: M bit + PT
        let marker_bit: u8 = if self.marker { 0x80 } else { 0x00 };
        buf[1] = marker_bit | (self.payload_type & 0x7F);
        // Octets 2-3: sequence number (big-endian)
        buf[2] = (self.sequence_number >> 8) as u8;
        buf[3] = (self.sequence_number & 0xFF) as u8;
        // Octets 4-7: timestamp (big-endian)
        buf[4] = ((self.timestamp >> 24) & 0xFF) as u8;
        buf[5] = ((self.timestamp >> 16) & 0xFF) as u8;
        buf[6] = ((self.timestamp >> 8) & 0xFF) as u8;
        buf[7] = (self.timestamp & 0xFF) as u8;
        // Octets 8-11: SSRC (big-endian)
        buf[8] = ((self.ssrc >> 24) & 0xFF) as u8;
        buf[9] = ((self.ssrc >> 16) & 0xFF) as u8;
        buf[10] = ((self.ssrc >> 8) & 0xFF) as u8;
        buf[11] = (self.ssrc & 0xFF) as u8;
        Ok(())
    }

    /// Parses a 12-byte RTP header.
    pub fn parse(buf: &[u8]) -> Result<Self, Aes67Error> {
        if buf.len() < Self::SERIALIZED_SIZE {
            return Err(Aes67Error::MalformedPacket {
                reason: format!(
                    "RTP header too short: {} < {}",
                    buf.len(),
                    Self::SERIALIZED_SIZE
                ),
            });
        }
        // Validate version = 2
        let version = (buf[0] >> 6) & 0x03;
        if version != 2 {
            return Err(Aes67Error::MalformedPacket {
                reason: format!("invalid RTP version: {}", version),
            });
        }
        let marker = (buf[1] & 0x80) != 0;
        let payload_type = buf[1] & 0x7F;
        let sequence_number = ((buf[2] as u16) << 8) | buf[3] as u16;
        let timestamp = ((buf[4] as u32) << 24)
            | ((buf[5] as u32) << 16)
            | ((buf[6] as u32) << 8)
            | buf[7] as u32;
        let ssrc = ((buf[8] as u32) << 24)
            | ((buf[9] as u32) << 16)
            | ((buf[10] as u32) << 8)
            | buf[11] as u32;
        Ok(Self {
            sequence_number,
            timestamp,
            ssrc,
            payload_type,
            marker,
        })
    }
}

// ---------------------------------------------------------------------------
// Aes67Packetizer
// ---------------------------------------------------------------------------

/// Packetizes raw PCM audio into AES67 RTP packets.
///
/// Audio samples are expected as big-endian signed integers (L24 for 24-bit,
/// L16 for 16-bit, L32 for 32-bit), interleaved by channel.
pub struct Aes67Packetizer {
    config: Aes67StreamConfig,
    sequence_number: u16,
    timestamp: u32,
    first_packet: bool,
}

impl Aes67Packetizer {
    /// Creates a new packetizer with the given stream configuration.
    pub fn new(config: Aes67StreamConfig) -> Self {
        Self {
            config,
            sequence_number: 0,
            timestamp: 0,
            first_packet: true,
        }
    }

    /// Packetizes one frame of audio samples into a complete RTP packet.
    ///
    /// `samples` must contain exactly `samples_per_packet * channels` samples,
    /// each encoded as `bytes_per_sample` big-endian bytes.
    ///
    /// Returns the total number of bytes written to `out`.
    pub fn packetize(&mut self, payload: &[u8], out: &mut [u8]) -> Result<usize, Aes67Error> {
        let expected_payload = self.config.payload_bytes_per_packet();
        if payload.len() < expected_payload {
            return Err(Aes67Error::BufferTooSmall {
                needed: expected_payload,
                have: payload.len(),
            });
        }
        let total = Aes67PacketHeader::SERIALIZED_SIZE + expected_payload;
        if out.len() < total {
            return Err(Aes67Error::BufferTooSmall {
                needed: total,
                have: out.len(),
            });
        }
        let header = Aes67PacketHeader {
            sequence_number: self.sequence_number,
            timestamp: self.timestamp,
            ssrc: self.config.ssrc,
            payload_type: self.config.payload_type,
            marker: self.first_packet,
        };
        header.serialize(&mut out[..Aes67PacketHeader::SERIALIZED_SIZE])?;
        out[Aes67PacketHeader::SERIALIZED_SIZE..total]
            .copy_from_slice(&payload[..expected_payload]);

        self.sequence_number = self.sequence_number.wrapping_add(1);
        self.timestamp = self
            .timestamp
            .wrapping_add(self.config.rtp_timestamp_increment());
        self.first_packet = false;
        Ok(total)
    }

    /// Returns the current sequence number.
    pub fn sequence_number(&self) -> u16 {
        self.sequence_number
    }

    /// Returns the current RTP timestamp.
    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }
}

// ---------------------------------------------------------------------------
// Aes67Depacketizer
// ---------------------------------------------------------------------------

/// Depacketizes AES67 RTP packets back to raw PCM audio.
pub struct Aes67Depacketizer {
    config: Aes67StreamConfig,
    expected_sequence: Option<u16>,
    packets_received: u64,
    packets_dropped: u64,
    /// Simplified jitter estimate in RTP timestamp units (RFC 3550 algorithm).
    jitter: f64,
    last_arrival_ts: Option<u64>,
    last_rtp_ts: Option<u32>,
}

impl Aes67Depacketizer {
    /// Creates a new depacketizer with the given stream configuration.
    pub fn new(config: Aes67StreamConfig) -> Self {
        Self {
            config,
            expected_sequence: None,
            packets_received: 0,
            packets_dropped: 0,
            jitter: 0.0,
            last_arrival_ts: None,
            last_rtp_ts: None,
        }
    }

    /// Depacketizes one RTP packet.
    ///
    /// Returns the payload slice from within `packet` (zero-copy view).
    /// Validates the RTP header and sequence continuity.
    ///
    /// `arrival_sample_time` — monotonic sample counter at time of arrival
    /// (used for jitter estimation). Pass `0` to skip jitter tracking.
    pub fn depacketize<'a>(
        &mut self,
        packet: &'a [u8],
        arrival_sample_time: u64,
    ) -> Result<&'a [u8], Aes67Error> {
        let header = Aes67PacketHeader::parse(packet)?;

        // Sequence continuity check
        if let Some(expected) = self.expected_sequence {
            if header.sequence_number != expected {
                let dropped = if header.sequence_number > expected {
                    header.sequence_number.wrapping_sub(expected) as u64
                } else {
                    // Wrapped around
                    (u16::MAX as u64)
                        .wrapping_sub(expected as u64)
                        .wrapping_add(header.sequence_number as u64)
                        .wrapping_add(1)
                };
                self.packets_dropped += dropped;
                return Err(Aes67Error::InvalidSequence {
                    expected,
                    got: header.sequence_number,
                });
            }
        }
        self.expected_sequence = Some(header.sequence_number.wrapping_add(1));

        // Jitter estimation (simplified RFC 3550)
        if arrival_sample_time != 0 {
            if let (Some(last_arr), Some(last_rtp)) = (self.last_arrival_ts, self.last_rtp_ts) {
                let arr_delta = arrival_sample_time.wrapping_sub(last_arr) as f64;
                let rtp_delta = header.timestamp.wrapping_sub(last_rtp) as f64;
                let d = (arr_delta - rtp_delta).abs();
                self.jitter += (d - self.jitter) / 16.0;
            }
            self.last_arrival_ts = Some(arrival_sample_time);
            self.last_rtp_ts = Some(header.timestamp);
        }

        self.packets_received += 1;

        let payload_start = Aes67PacketHeader::SERIALIZED_SIZE;
        if packet.len() < payload_start {
            return Err(Aes67Error::MalformedPacket {
                reason: "packet shorter than RTP header".to_string(),
            });
        }
        Ok(&packet[payload_start..])
    }

    /// Returns the number of packets successfully received.
    pub fn packets_received(&self) -> u64 {
        self.packets_received
    }

    /// Returns the number of packets detected as dropped.
    pub fn packets_dropped(&self) -> u64 {
        self.packets_dropped
    }

    /// Returns the estimated jitter in RTP timestamp units.
    pub fn jitter_estimate(&self) -> f64 {
        self.jitter
    }

    /// Returns the jitter estimate in microseconds.
    pub fn jitter_us(&self) -> f64 {
        if self.config.sample_rate == 0 {
            return 0.0;
        }
        self.jitter * 1_000_000.0 / self.config.sample_rate as f64
    }

    /// Resets sequence tracking (e.g., after a source restart).
    pub fn reset(&mut self) {
        self.expected_sequence = None;
        self.last_arrival_ts = None;
        self.last_rtp_ts = None;
        self.jitter = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Aes67SdpBuilder
// ---------------------------------------------------------------------------

/// Builds SDP session descriptions for AES67 streams.
///
/// The generated SDP conforms to AES67-2018 and SMPTE ST 2110-30.
pub struct Aes67SdpBuilder {
    config: Aes67StreamConfig,
    /// Origin username.
    origin_username: String,
    /// Session ID (usually a NTP timestamp).
    session_id: u64,
    /// Session version.
    session_version: u64,
    /// Originating host address.
    origin_address: String,
    /// Connection address (may differ from multicast group for unicast).
    connection_address: Option<String>,
    /// PTP clock identity string.
    ptp_clock: String,
}

impl Aes67SdpBuilder {
    /// Creates a new SDP builder for the given stream.
    pub fn new(config: Aes67StreamConfig) -> Self {
        let ssrc = config.ssrc;
        Self {
            config,
            origin_username: "-".to_string(),
            session_id: ssrc as u64,
            session_version: 1,
            origin_address: "0.0.0.0".to_string(),
            connection_address: None,
            ptp_clock: "traceable".to_string(),
        }
    }

    /// Sets the origin username field.
    pub fn with_origin(mut self, username: impl Into<String>, address: impl Into<String>) -> Self {
        self.origin_username = username.into();
        self.origin_address = address.into();
        self
    }

    /// Sets a custom PTP clock identity.
    pub fn with_ptp_clock(mut self, clock: impl Into<String>) -> Self {
        self.ptp_clock = clock.into();
        self
    }

    /// Sets a custom connection address (overrides multicast address for `c=` line).
    pub fn with_connection_address(mut self, addr: impl Into<String>) -> Self {
        self.connection_address = Some(addr.into());
        self
    }

    /// Builds the SDP string.
    pub fn build(&self) -> String {
        let conn_addr = self
            .connection_address
            .as_deref()
            .unwrap_or(&self.config.multicast_address);
        let ttl = if conn_addr.contains(':') {
            // IPv6 — no TTL
            conn_addr.to_string()
        } else {
            format!("{}/32", conn_addr)
        };

        let encoding = match self.config.bit_depth {
            16 => "L16",
            24 => "L24",
            32 => "L32",
            _ => "L24",
        };

        let mut sdp = String::new();
        sdp.push_str("v=0\r\n");
        sdp.push_str(&format!(
            "o={} {} {} IN IP4 {}\r\n",
            self.origin_username, self.session_id, self.session_version, self.origin_address
        ));
        sdp.push_str(&format!("s={}\r\n", self.config.stream_name));
        sdp.push_str(&format!("c=IN IP4 {}\r\n", ttl));
        sdp.push_str("t=0 0\r\n");
        sdp.push_str(&format!(
            "m=audio {} RTP/AVP {}\r\n",
            self.config.port, self.config.payload_type
        ));
        sdp.push_str(&format!(
            "a=rtpmap:{} {}/{}/{}\r\n",
            self.config.payload_type, encoding, self.config.sample_rate, self.config.channels
        ));
        sdp.push_str(&format!(
            "a=ptime:{}\r\n",
            self.config.packet_time.sdp_ptime()
        ));
        sdp.push_str(&format!(
            "a=ts-refclk:ptp=IEEE1588-2008:{}\r\n",
            self.ptp_clock
        ));
        sdp.push_str("a=mediaclk:direct=0\r\n");
        sdp.push_str(&format!(
            "a=source-filter: incl IN IP4 {} {}\r\n",
            conn_addr, self.origin_address
        ));
        sdp.push_str(&format!(
            "a=ssrc:{} cname:{}\r\n",
            self.config.ssrc, self.config.stream_name
        ));
        sdp
    }
}

// ---------------------------------------------------------------------------
// Aes67SessionManager
// ---------------------------------------------------------------------------

/// Per-stream receive statistics tracked by the session manager.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Total RTP packets received.
    pub packets_received: u64,
    /// Total RTP packets estimated dropped.
    pub packets_dropped: u64,
    /// Estimated jitter in µs.
    pub jitter_us: f64,
    /// Whether the stream is currently active (received data recently).
    pub active: bool,
}

/// Manages multiple AES67 receive streams by SSRC.
///
/// Each SSRC gets its own depacketizer and statistics counters.
#[derive(Default)]
pub struct Aes67SessionManager {
    streams: HashMap<u32, (Aes67Depacketizer, StreamStats)>,
}

impl Aes67SessionManager {
    /// Creates a new, empty session manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new stream. The SSRC is taken from `config.ssrc`.
    pub fn register_stream(&mut self, config: Aes67StreamConfig) {
        let ssrc = config.ssrc;
        let depack = Aes67Depacketizer::new(config);
        self.streams.insert(ssrc, (depack, StreamStats::default()));
    }

    /// Processes a received UDP packet, dispatching it to the matching SSRC.
    ///
    /// Returns the SSRC and payload slice, or an error if the packet is
    /// malformed or the SSRC is unknown.
    pub fn process_packet<'a>(
        &mut self,
        packet: &'a [u8],
        arrival_sample_time: u64,
    ) -> Result<(u32, &'a [u8]), Aes67Error> {
        // Parse header to get SSRC without consuming packet
        let header = Aes67PacketHeader::parse(packet)?;
        let ssrc = header.ssrc;

        let (depack, stats) =
            self.streams
                .get_mut(&ssrc)
                .ok_or_else(|| Aes67Error::MalformedPacket {
                    reason: format!("unknown SSRC: 0x{:08X}", ssrc),
                })?;

        let payload = depack.depacketize(packet, arrival_sample_time)?;
        stats.packets_received = depack.packets_received();
        stats.packets_dropped = depack.packets_dropped();
        stats.jitter_us = depack.jitter_us();
        stats.active = true;

        // Safety: the payload slice references `packet` which outlives this call.
        // We return it with the lifetime of `packet`.
        Ok((ssrc, payload))
    }

    /// Returns receive statistics for the given SSRC.
    pub fn stats(&self, ssrc: u32) -> Option<&StreamStats> {
        self.streams.get(&ssrc).map(|(_, s)| s)
    }

    /// Returns the number of registered streams.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Marks a stream as inactive (e.g., after a timeout).
    pub fn mark_inactive(&mut self, ssrc: u32) {
        if let Some((_, stats)) = self.streams.get_mut(&ssrc) {
            stats.active = false;
        }
    }

    /// Resets sequence tracking for all streams.
    pub fn reset_all(&mut self) {
        for (depack, _) in self.streams.values_mut() {
            depack.reset();
        }
    }

    /// Removes a stream from management.
    pub fn remove_stream(&mut self, ssrc: u32) -> bool {
        self.streams.remove(&ssrc).is_some()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> Aes67StreamConfig {
        Aes67StreamConfig::new("TestStream", 48_000, 24, 2, 0xDEAD_BEEF)
            .expect("test config creation should succeed")
    }

    // --- PacketTimeUs ---

    #[test]
    fn test_packet_time_us_values() {
        assert_eq!(PacketTimeUs::Us125.as_us(), 125);
        assert_eq!(PacketTimeUs::Us1000.as_us(), 1_000);
        assert_eq!(PacketTimeUs::Us4000.as_us(), 4_000);
    }

    #[test]
    fn test_packet_time_samples_per_packet() {
        // 48000 Hz * 1000 µs / 1_000_000 = 48 samples
        assert_eq!(PacketTimeUs::Us1000.samples_per_packet(48_000), 48);
        // 48000 Hz * 125 µs / 1_000_000 = 6 samples
        assert_eq!(PacketTimeUs::Us125.samples_per_packet(48_000), 6);
    }

    #[test]
    fn test_packet_time_from_us_valid() {
        assert_eq!(
            PacketTimeUs::from_us(125).expect("125us should be valid"),
            PacketTimeUs::Us125
        );
        assert_eq!(
            PacketTimeUs::from_us(1_000).expect("1000us should be valid"),
            PacketTimeUs::Us1000
        );
    }

    #[test]
    fn test_packet_time_from_us_invalid() {
        assert!(PacketTimeUs::from_us(500).is_err());
    }

    #[test]
    fn test_packet_time_sdp_ptime() {
        assert_eq!(PacketTimeUs::Us1000.sdp_ptime(), "1");
        assert_eq!(PacketTimeUs::Us125.sdp_ptime(), "0.125");
    }

    // --- Aes67StreamConfig ---

    #[test]
    fn test_config_invalid_sample_rate() {
        let result = Aes67StreamConfig::new("test", 22_050, 24, 2, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_valid_rates() {
        for rate in [32_000u32, 44_100, 48_000, 88_200, 96_000, 192_000] {
            assert!(Aes67StreamConfig::new("s", rate, 24, 2, 1).is_ok());
        }
    }

    #[test]
    fn test_config_payload_size() {
        let cfg = make_config();
        // 48 samples * 2 channels * 3 bytes = 288 bytes
        assert_eq!(cfg.payload_bytes_per_packet(), 288);
    }

    #[test]
    fn test_config_bytes_per_sample() {
        let cfg = make_config();
        assert_eq!(cfg.bytes_per_sample(), 3);
    }

    // --- Aes67PacketHeader ---

    #[test]
    fn test_header_round_trip() {
        let hdr = Aes67PacketHeader {
            sequence_number: 1234,
            timestamp: 0x0ABC_DEF0,
            ssrc: 0xDEAD_BEEF,
            payload_type: 96,
            marker: false,
        };
        let mut buf = [0u8; 12];
        hdr.serialize(&mut buf)
            .expect("header serialization should succeed");
        let parsed = Aes67PacketHeader::parse(&buf).expect("header parsing should succeed");
        assert_eq!(parsed.sequence_number, 1234);
        assert_eq!(parsed.timestamp, 0x0ABC_DEF0);
        assert_eq!(parsed.ssrc, 0xDEAD_BEEF);
        assert_eq!(parsed.payload_type, 96);
        assert!(!parsed.marker);
    }

    #[test]
    fn test_header_marker_bit() {
        let hdr = Aes67PacketHeader {
            sequence_number: 0,
            timestamp: 0,
            ssrc: 1,
            payload_type: 96,
            marker: true,
        };
        let mut buf = [0u8; 12];
        hdr.serialize(&mut buf)
            .expect("marker header serialization should succeed");
        let parsed = Aes67PacketHeader::parse(&buf).expect("marker header parsing should succeed");
        assert!(parsed.marker);
    }

    #[test]
    fn test_header_parse_short_buffer() {
        let buf = [0u8; 5];
        assert!(Aes67PacketHeader::parse(&buf).is_err());
    }

    #[test]
    fn test_header_invalid_version() {
        let mut buf = [0u8; 12];
        buf[0] = 0x40; // version = 1
        assert!(Aes67PacketHeader::parse(&buf).is_err());
    }

    // --- Packetizer / Depacketizer round-trip ---

    #[test]
    fn test_packetize_depacketize_round_trip() {
        let cfg = make_config();
        let payload_size = cfg.payload_bytes_per_packet();
        let mut pktizer = Aes67Packetizer::new(cfg.clone());
        let mut depktizer = Aes67Depacketizer::new(cfg);

        // Build a payload (use wrapping bytes to fill the entire payload buffer)
        let payload: Vec<u8> = (0..payload_size).map(|i| (i & 0xFF) as u8).collect();
        let mut packet = vec![0u8; Aes67PacketHeader::SERIALIZED_SIZE + payload_size];
        let written = pktizer
            .packetize(&payload, &mut packet)
            .expect("packetize should succeed");
        assert_eq!(written, Aes67PacketHeader::SERIALIZED_SIZE + payload_size);

        let received = depktizer
            .depacketize(&packet[..written], 0)
            .expect("depacketize should succeed");
        assert_eq!(received, &payload[..payload_size]);
    }

    #[test]
    fn test_packetizer_sequence_increments() {
        let cfg = make_config();
        let payload_size = cfg.payload_bytes_per_packet();
        let mut pktizer = Aes67Packetizer::new(cfg);
        let payload = vec![0u8; payload_size];
        let mut packet = vec![0u8; Aes67PacketHeader::SERIALIZED_SIZE + payload_size];

        pktizer
            .packetize(&payload, &mut packet)
            .expect("first packetize should succeed");
        assert_eq!(pktizer.sequence_number(), 1);
        pktizer
            .packetize(&payload, &mut packet)
            .expect("second packetize should succeed");
        assert_eq!(pktizer.sequence_number(), 2);
    }

    #[test]
    fn test_depacketizer_sequence_error() {
        let cfg = make_config();
        let payload_size = cfg.payload_bytes_per_packet();
        let mut pktizer = Aes67Packetizer::new(cfg.clone());
        let mut depktizer = Aes67Depacketizer::new(cfg);
        let payload = vec![0u8; payload_size];
        let mut packet = vec![0u8; Aes67PacketHeader::SERIALIZED_SIZE + payload_size];

        // Send packet 0
        pktizer
            .packetize(&payload, &mut packet)
            .expect("packet 0 should succeed");
        depktizer
            .depacketize(&packet, 0)
            .expect("depacketize packet 0 should succeed");

        // Skip to packet 2 (sequence gap)
        pktizer
            .packetize(&payload, &mut packet)
            .expect("packet 1 should succeed");
        pktizer
            .packetize(&payload, &mut packet)
            .expect("packet 2 should succeed");
        let result = depktizer.depacketize(&packet, 0);
        assert!(result.is_err());
        assert!(depktizer.packets_dropped() > 0);
    }

    // --- SDP Builder ---

    #[test]
    fn test_sdp_contains_required_fields() {
        let cfg = make_config();
        let sdp = Aes67SdpBuilder::new(cfg).build();
        assert!(sdp.contains("v=0"));
        assert!(sdp.contains("m=audio"));
        assert!(sdp.contains("a=rtpmap:96 L24/48000/2"));
        assert!(sdp.contains("a=ptime:1"));
        assert!(sdp.contains("a=ts-refclk:ptp=IEEE1588-2008:traceable"));
        assert!(sdp.contains("a=mediaclk:direct=0"));
    }

    #[test]
    fn test_sdp_stream_name() {
        let cfg = make_config();
        let sdp = Aes67SdpBuilder::new(cfg).build();
        assert!(sdp.contains("s=TestStream"));
    }

    // --- Session Manager ---

    #[test]
    fn test_session_manager_register_and_count() {
        let mut mgr = Aes67SessionManager::new();
        mgr.register_stream(make_config());
        assert_eq!(mgr.stream_count(), 1);
    }

    #[test]
    fn test_session_manager_unknown_ssrc() {
        let mut mgr = Aes67SessionManager::new();
        let cfg = make_config();
        let payload_size = cfg.payload_bytes_per_packet();
        let mut pktizer = Aes67Packetizer::new(cfg.clone());
        // Don't register, so SSRC is unknown
        let payload = vec![0u8; payload_size];
        let mut packet = vec![0u8; Aes67PacketHeader::SERIALIZED_SIZE + payload_size];
        pktizer
            .packetize(&payload, &mut packet)
            .expect("packetize for unknown ssrc test should succeed");
        let result = mgr.process_packet(&packet, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_manager_stats_update() {
        let mut mgr = Aes67SessionManager::new();
        let cfg = make_config();
        let ssrc = cfg.ssrc;
        let payload_size = cfg.payload_bytes_per_packet();
        mgr.register_stream(cfg.clone());

        let mut pktizer = Aes67Packetizer::new(cfg);
        let payload = vec![0u8; payload_size];
        let mut packet = vec![0u8; Aes67PacketHeader::SERIALIZED_SIZE + payload_size];
        pktizer
            .packetize(&payload, &mut packet)
            .expect("packetize for stats test should succeed");
        mgr.process_packet(&packet, 1000)
            .expect("process_packet for stats test should succeed");

        let stats = mgr
            .stats(ssrc)
            .expect("stats for registered stream should exist");
        assert_eq!(stats.packets_received, 1);
        assert!(stats.active);
    }

    #[test]
    fn test_session_manager_remove_stream() {
        let mut mgr = Aes67SessionManager::new();
        let cfg = make_config();
        let ssrc = cfg.ssrc;
        mgr.register_stream(cfg);
        assert!(mgr.remove_stream(ssrc));
        assert_eq!(mgr.stream_count(), 0);
    }
}

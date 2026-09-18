//! RTP (RFC 3550) packet handling for SMPTE ST 2110.
//!
//! This module provides RTP (Real-time Transport Protocol) packet handling
//! including RTP header parsing/serialization, RTCP control packets, sequence
//! number management, SSRC handling, and jitter calculation.

use crate::error::{NetError, NetResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// RTP protocol version (always 2 for RFC 3550).
pub const RTP_VERSION: u8 = 2;

/// Maximum RTP payload size (typically MTU - headers).
pub const MAX_RTP_PAYLOAD: usize = 1400;

/// RTP header structure (RFC 3550 Section 5.1).
///
/// ```text
///  0                   1                   2                   3
///  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |V=2|P|X|  CC   |M|     PT      |       sequence number         |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                           timestamp                           |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |           synchronization source (SSRC) identifier            |
/// +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
/// |            contributing source (CSRC) identifiers             |
/// |                             ....                              |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpHeader {
    /// Padding flag - if set, packet contains padding octets.
    pub padding: bool,
    /// Extension flag - if set, header extension follows.
    pub extension: bool,
    /// CSRC count - number of CSRC identifiers.
    pub csrc_count: u8,
    /// Marker bit - interpretation depends on profile.
    pub marker: bool,
    /// Payload type - identifies the format of the RTP payload.
    pub payload_type: u8,
    /// Sequence number - increments by one for each RTP packet sent.
    pub sequence_number: u16,
    /// Timestamp - sampling instant of the first octet in the RTP data packet.
    pub timestamp: u32,
    /// SSRC - synchronization source identifier.
    pub ssrc: u32,
    /// CSRCs - contributing source identifiers.
    pub csrcs: Vec<u32>,
    /// Header extension (if present).
    pub extension_data: Option<RtpHeaderExtension>,
}

/// RTP header extension (RFC 3550 Section 5.3.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpHeaderExtension {
    /// Profile-defined extension type.
    pub profile: u16,
    /// Extension data.
    pub data: Bytes,
}

/// Complete RTP packet.
#[derive(Debug, Clone)]
pub struct RtpPacket {
    /// RTP header.
    pub header: RtpHeader,
    /// Payload data.
    pub payload: Bytes,
}

/// RTCP packet types (RFC 3550 Section 6.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtcpPacketType {
    /// Sender Report (SR).
    SenderReport = 200,
    /// Receiver Report (RR).
    ReceiverReport = 201,
    /// Source Description (SDES).
    SourceDescription = 202,
    /// Goodbye (BYE).
    Goodbye = 203,
    /// Application-defined (APP).
    ApplicationDefined = 204,
}

/// RTCP Sender Report (RFC 3550 Section 6.4.1).
#[derive(Debug, Clone)]
pub struct RtcpSenderReport {
    /// SSRC of sender.
    pub ssrc: u32,
    /// NTP timestamp (most significant 32 bits).
    pub ntp_timestamp_msw: u32,
    /// NTP timestamp (least significant 32 bits).
    pub ntp_timestamp_lsw: u32,
    /// RTP timestamp.
    pub rtp_timestamp: u32,
    /// Sender's packet count.
    pub sender_packet_count: u32,
    /// Sender's octet count.
    pub sender_octet_count: u32,
    /// Reception report blocks.
    pub report_blocks: Vec<RtcpReportBlock>,
}

/// RTCP Receiver Report (RFC 3550 Section 6.4.2).
#[derive(Debug, Clone)]
pub struct RtcpReceiverReport {
    /// SSRC of receiver.
    pub ssrc: u32,
    /// Reception report blocks.
    pub report_blocks: Vec<RtcpReportBlock>,
}

/// RTCP Report Block (RFC 3550 Section 6.4.1).
#[derive(Debug, Clone)]
pub struct RtcpReportBlock {
    /// SSRC of source.
    pub ssrc: u32,
    /// Fraction lost since last report.
    pub fraction_lost: u8,
    /// Cumulative number of packets lost.
    pub cumulative_packets_lost: i32,
    /// Extended highest sequence number received.
    pub extended_highest_sequence: u32,
    /// Interarrival jitter.
    pub interarrival_jitter: u32,
    /// Last SR timestamp.
    pub last_sr_timestamp: u32,
    /// Delay since last SR.
    pub delay_since_last_sr: u32,
}

/// RTP sequence number handler with rollover detection.
#[derive(Debug, Clone)]
pub struct SequenceNumberTracker {
    /// Highest sequence number seen.
    highest_seq: u16,
    /// Number of sequence number cycles.
    cycles: u32,
    /// Base sequence number.
    base_seq: u16,
    /// Number of packets received.
    received: u64,
    /// Expected prior.
    expected_prior: u64,
    /// Received prior.
    received_prior: u64,
}

impl SequenceNumberTracker {
    /// Creates a new sequence number tracker.
    #[must_use]
    pub fn new(initial_seq: u16) -> Self {
        Self {
            highest_seq: initial_seq,
            cycles: 0,
            base_seq: initial_seq,
            received: 0,
            expected_prior: 0,
            received_prior: 0,
        }
    }

    /// Updates with a new sequence number.
    pub fn update(&mut self, seq: u16) {
        let delta = seq.wrapping_sub(self.highest_seq);

        // Detect rollover
        if delta < 0x8000 {
            // In order or small jump forward
            if seq < self.highest_seq {
                self.cycles += 1;
            }
            self.highest_seq = seq;
        } else {
            // Old packet or very large jump - ignore for now
        }

        self.received += 1;
    }

    /// Gets the extended sequence number (with cycle count).
    #[must_use]
    pub const fn extended_sequence(&self) -> u32 {
        ((self.cycles as u32) << 16) | (self.highest_seq as u32)
    }

    /// Calculates the number of packets lost.
    #[must_use]
    pub fn packets_lost(&self) -> i64 {
        let expected = i64::from(self.extended_sequence() - u32::from(self.base_seq)) + 1;
        expected - self.received as i64
    }

    /// Calculates fraction lost since last report (8-bit fixed point).
    #[must_use]
    pub fn fraction_lost(&mut self) -> u8 {
        let expected = self.extended_sequence() as u64 - u64::from(self.base_seq);
        let expected_interval = expected.saturating_sub(self.expected_prior);
        let received_interval = self.received.saturating_sub(self.received_prior);

        self.expected_prior = expected;
        self.received_prior = self.received;

        if expected_interval == 0 || expected_interval <= received_interval {
            0
        } else {
            let lost = expected_interval - received_interval;
            ((lost << 8) / expected_interval).min(255) as u8
        }
    }
}

/// Jitter calculator (RFC 3550 Section 6.4.1).
#[derive(Debug, Clone)]
pub struct JitterCalculator {
    /// Interarrival jitter estimate.
    jitter: f64,
    /// Previous RTP timestamp.
    prev_timestamp: Option<u32>,
    /// Previous arrival time.
    prev_arrival: Option<Instant>,
    /// Clock rate (Hz).
    clock_rate: u32,
}

impl JitterCalculator {
    /// Creates a new jitter calculator.
    #[must_use]
    pub const fn new(clock_rate: u32) -> Self {
        Self {
            jitter: 0.0,
            prev_timestamp: None,
            prev_arrival: None,
            clock_rate,
        }
    }

    /// Updates jitter calculation with a new packet.
    pub fn update(&mut self, timestamp: u32, arrival_time: Instant) {
        if let (Some(prev_ts), Some(prev_arr)) = (self.prev_timestamp, self.prev_arrival) {
            let ts_diff = timestamp.wrapping_sub(prev_ts) as i64;
            let arr_diff = arrival_time.duration_since(prev_arr).as_secs_f64();
            let arr_diff_units = (arr_diff * f64::from(self.clock_rate)) as i64;

            let d = (arr_diff_units - ts_diff).abs() as f64;
            self.jitter += (d - self.jitter) / 16.0;
        }

        self.prev_timestamp = Some(timestamp);
        self.prev_arrival = Some(arrival_time);
    }

    /// Gets the current jitter estimate in timestamp units.
    #[must_use]
    pub const fn jitter(&self) -> u32 {
        self.jitter as u32
    }

    /// Gets the current jitter estimate as duration.
    #[must_use]
    pub fn jitter_duration(&self) -> Duration {
        Duration::from_secs_f64(self.jitter / f64::from(self.clock_rate))
    }
}

/// RTP session statistics.
#[derive(Debug, Clone)]
pub struct RtpStatistics {
    /// Sequence number tracker.
    pub seq_tracker: SequenceNumberTracker,
    /// Jitter calculator.
    pub jitter: JitterCalculator,
    /// Total packets received.
    pub packets_received: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Last packet timestamp.
    pub last_timestamp: u32,
    /// Last packet arrival time.
    pub last_arrival: Instant,
}

impl RtpStatistics {
    /// Creates new RTP statistics.
    #[must_use]
    pub fn new(initial_seq: u16, clock_rate: u32) -> Self {
        Self {
            seq_tracker: SequenceNumberTracker::new(initial_seq),
            jitter: JitterCalculator::new(clock_rate),
            packets_received: 0,
            bytes_received: 0,
            last_timestamp: 0,
            last_arrival: Instant::now(),
        }
    }

    /// Updates statistics with a new packet.
    pub fn update(&mut self, packet: &RtpPacket) {
        self.seq_tracker.update(packet.header.sequence_number);
        self.jitter.update(packet.header.timestamp, Instant::now());
        self.packets_received += 1;
        self.bytes_received += packet.payload.len() as u64;
        self.last_timestamp = packet.header.timestamp;
        self.last_arrival = Instant::now();
    }
}

/// RTP session manager.
#[derive(Debug)]
pub struct RtpSession {
    /// Local SSRC.
    pub ssrc: u32,
    /// Remote SSRCs and their statistics.
    pub remote_sources: HashMap<u32, RtpStatistics>,
    /// Next sequence number to send.
    pub next_seq: u16,
    /// Packets sent.
    pub packets_sent: u64,
    /// Bytes sent.
    pub bytes_sent: u64,
    /// Clock rate.
    pub clock_rate: u32,
}

impl RtpSession {
    /// Creates a new RTP session.
    #[must_use]
    pub fn new(ssrc: u32, clock_rate: u32) -> Self {
        Self {
            ssrc,
            remote_sources: HashMap::new(),
            next_seq: rand::random(),
            packets_sent: 0,
            bytes_sent: 0,
            clock_rate,
        }
    }

    /// Processes a received RTP packet.
    pub fn process_packet(&mut self, packet: &RtpPacket) {
        let ssrc = packet.header.ssrc;
        let stats = self
            .remote_sources
            .entry(ssrc)
            .or_insert_with(|| RtpStatistics::new(packet.header.sequence_number, self.clock_rate));
        stats.update(packet);
    }

    /// Gets the next sequence number and increments.
    pub fn next_sequence(&mut self) -> u16 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        seq
    }

    /// Records a sent packet.
    pub fn record_sent(&mut self, payload_size: usize) {
        self.packets_sent += 1;
        self.bytes_sent += payload_size as u64;
    }
}

impl RtpHeader {
    /// Minimum RTP header size (without CSRC or extensions).
    pub const MIN_SIZE: usize = 12;

    /// Parses an RTP header from bytes.
    pub fn parse(data: &[u8]) -> NetResult<(Self, usize)> {
        if data.len() < Self::MIN_SIZE {
            return Err(NetError::parse(0, "RTP header too short"));
        }

        let mut cursor = &data[..];

        // Byte 0: V(2), P(1), X(1), CC(4)
        let byte0 = cursor.get_u8();
        let version = (byte0 >> 6) & 0x03;
        if version != RTP_VERSION {
            return Err(NetError::protocol(format!(
                "Invalid RTP version: {version}"
            )));
        }

        let padding = (byte0 & 0x20) != 0;
        let extension = (byte0 & 0x10) != 0;
        let csrc_count = byte0 & 0x0F;

        // Byte 1: M(1), PT(7)
        let byte1 = cursor.get_u8();
        let marker = (byte1 & 0x80) != 0;
        let payload_type = byte1 & 0x7F;

        // Bytes 2-3: Sequence number
        let sequence_number = cursor.get_u16();

        // Bytes 4-7: Timestamp
        let timestamp = cursor.get_u32();

        // Bytes 8-11: SSRC
        let ssrc = cursor.get_u32();

        // CSRCs
        let mut csrcs = Vec::with_capacity(csrc_count as usize);
        for _ in 0..csrc_count {
            if cursor.len() < 4 {
                return Err(NetError::parse(0, "Not enough data for CSRC"));
            }
            csrcs.push(cursor.get_u32());
        }

        let mut header_size = Self::MIN_SIZE + (csrc_count as usize * 4);

        // Extension
        let extension_data = if extension {
            if cursor.len() < 4 {
                return Err(NetError::parse(0, "Not enough data for extension"));
            }

            let profile = cursor.get_u16();
            let length = cursor.get_u16() as usize * 4; // Length in 32-bit words

            if cursor.len() < length {
                return Err(NetError::parse(0, "Not enough data for extension data"));
            }

            let ext_data = cursor.copy_to_bytes(length);
            header_size += 4 + length;

            Some(RtpHeaderExtension {
                profile,
                data: ext_data,
            })
        } else {
            None
        };

        Ok((
            Self {
                padding,
                extension,
                csrc_count,
                marker,
                payload_type,
                sequence_number,
                timestamp,
                ssrc,
                csrcs,
                extension_data,
            },
            header_size,
        ))
    }

    /// Serializes the RTP header to bytes.
    pub fn serialize(&self, buf: &mut BytesMut) {
        // Byte 0
        let byte0 = (RTP_VERSION << 6)
            | (u8::from(self.padding) << 5)
            | (u8::from(self.extension) << 4)
            | (self.csrc_count & 0x0F);
        buf.put_u8(byte0);

        // Byte 1
        let byte1 = (u8::from(self.marker) << 7) | (self.payload_type & 0x7F);
        buf.put_u8(byte1);

        // Sequence number
        buf.put_u16(self.sequence_number);

        // Timestamp
        buf.put_u32(self.timestamp);

        // SSRC
        buf.put_u32(self.ssrc);

        // CSRCs
        for csrc in &self.csrcs {
            buf.put_u32(*csrc);
        }

        // Extension
        if let Some(ext) = &self.extension_data {
            buf.put_u16(ext.profile);
            buf.put_u16((ext.data.len() / 4) as u16);
            buf.put_slice(&ext.data);
        }
    }

    /// Calculates the header size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        let mut size = Self::MIN_SIZE + (self.csrc_count as usize * 4);
        if let Some(ext) = &self.extension_data {
            size += 4 + ext.data.len();
        }
        size
    }
}

impl RtpPacket {
    /// Parses an RTP packet from bytes.
    pub fn parse(data: Bytes) -> NetResult<Self> {
        let (header, header_size) = RtpHeader::parse(&data)?;

        if data.len() < header_size {
            return Err(NetError::parse(0, "Data shorter than header"));
        }

        let payload = data.slice(header_size..);

        Ok(Self { header, payload })
    }

    /// Serializes the RTP packet to bytes.
    pub fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(self.header.size() + self.payload.len());
        self.header.serialize(&mut buf);
        buf.put_slice(&self.payload);
        buf.freeze()
    }
}

impl RtcpSenderReport {
    /// Parses an RTCP Sender Report.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < 28 {
            return Err(NetError::parse(0, "RTCP SR too short"));
        }

        let mut cursor = &data[..];
        let ssrc = cursor.get_u32();
        let ntp_timestamp_msw = cursor.get_u32();
        let ntp_timestamp_lsw = cursor.get_u32();
        let rtp_timestamp = cursor.get_u32();
        let sender_packet_count = cursor.get_u32();
        let sender_octet_count = cursor.get_u32();

        let mut report_blocks = Vec::new();
        while cursor.len() >= 24 {
            report_blocks.push(RtcpReportBlock::parse(&mut cursor)?);
        }

        Ok(Self {
            ssrc,
            ntp_timestamp_msw,
            ntp_timestamp_lsw,
            rtp_timestamp,
            sender_packet_count,
            sender_octet_count,
            report_blocks,
        })
    }

    /// Serializes the RTCP Sender Report.
    pub fn serialize(&self, buf: &mut BytesMut) {
        buf.put_u32(self.ssrc);
        buf.put_u32(self.ntp_timestamp_msw);
        buf.put_u32(self.ntp_timestamp_lsw);
        buf.put_u32(self.rtp_timestamp);
        buf.put_u32(self.sender_packet_count);
        buf.put_u32(self.sender_octet_count);

        for block in &self.report_blocks {
            block.serialize(buf);
        }
    }
}

impl RtcpReportBlock {
    /// Parses an RTCP Report Block.
    fn parse(cursor: &mut &[u8]) -> NetResult<Self> {
        if cursor.len() < 24 {
            return Err(NetError::parse(0, "RTCP report block too short"));
        }

        let ssrc = cursor.get_u32();
        let byte = cursor.get_u8();
        let fraction_lost = byte;
        let cumulative_bytes = {
            let b1 = cursor.get_u8() as u32;
            let b2 = cursor.get_u8() as u32;
            let b3 = cursor.get_u8() as u32;
            (b1 << 16) | (b2 << 8) | b3
        };
        let cumulative_packets_lost =
            cumulative_bytes as i32 - if (byte & 0x80) != 0 { 0x1000000 } else { 0 };
        let extended_highest_sequence = cursor.get_u32();
        let interarrival_jitter = cursor.get_u32();
        let last_sr_timestamp = cursor.get_u32();
        let delay_since_last_sr = cursor.get_u32();

        Ok(Self {
            ssrc,
            fraction_lost,
            cumulative_packets_lost,
            extended_highest_sequence,
            interarrival_jitter,
            last_sr_timestamp,
            delay_since_last_sr,
        })
    }

    /// Serializes an RTCP Report Block.
    fn serialize(&self, buf: &mut BytesMut) {
        buf.put_u32(self.ssrc);
        buf.put_u8(self.fraction_lost);

        let cumulative = if self.cumulative_packets_lost < 0 {
            (self.cumulative_packets_lost + 0x1000000) as u32
        } else {
            self.cumulative_packets_lost as u32
        };
        buf.put_u8(((cumulative >> 16) & 0xFF) as u8);
        buf.put_u8(((cumulative >> 8) & 0xFF) as u8);
        buf.put_u8((cumulative & 0xFF) as u8);

        buf.put_u32(self.extended_highest_sequence);
        buf.put_u32(self.interarrival_jitter);
        buf.put_u32(self.last_sr_timestamp);
        buf.put_u32(self.delay_since_last_sr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtp_header_parse_basic() {
        let data = [
            0x80, 0x60, // V=2, P=0, X=0, CC=0, M=0, PT=96
            0x00, 0x01, // Seq = 1
            0x00, 0x00, 0x00, 0x64, // Timestamp = 100
            0x12, 0x34, 0x56, 0x78, // SSRC
        ];

        let (header, size) = RtpHeader::parse(&data).expect("should succeed in test");
        assert_eq!(size, 12);
        assert!(!header.padding);
        assert!(!header.extension);
        assert_eq!(header.csrc_count, 0);
        assert!(!header.marker);
        assert_eq!(header.payload_type, 96);
        assert_eq!(header.sequence_number, 1);
        assert_eq!(header.timestamp, 100);
        assert_eq!(header.ssrc, 0x12345678);
    }

    #[test]
    fn test_sequence_tracker() {
        let mut tracker = SequenceNumberTracker::new(100);

        tracker.update(101);
        tracker.update(102);
        assert_eq!(tracker.extended_sequence(), 102);

        // Rollover
        tracker.highest_seq = 65535;
        tracker.update(0);
        assert_eq!(tracker.cycles, 1);
    }

    #[test]
    fn test_jitter_calculator() {
        let mut jitter = JitterCalculator::new(90000);

        let start = Instant::now();
        jitter.update(1000, start);
        // Expected: 2000 at 11.11ms, but arrive late at 15ms
        jitter.update(2000, start + Duration::from_millis(15));
        jitter.update(3000, start + Duration::from_millis(25));

        assert!(jitter.jitter() > 0);
    }
}

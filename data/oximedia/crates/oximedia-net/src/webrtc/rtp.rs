//! RTP (Real-time Transport Protocol) implementation.
//!
//! This module implements RTP packet handling for WebRTC media streams.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::error::{NetError, NetResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};

/// RTP header extension.
#[derive(Debug, Clone)]
pub struct Extension {
    /// Extension ID.
    pub id: u8,
    /// Extension data.
    pub data: Bytes,
}

/// RTP packet.
#[derive(Debug, Clone)]
pub struct Packet {
    /// Version (always 2).
    pub version: u8,
    /// Padding flag.
    pub padding: bool,
    /// Extension flag.
    pub extension: bool,
    /// CSRC count.
    pub csrc_count: u8,
    /// Marker bit.
    pub marker: bool,
    /// Payload type.
    pub payload_type: u8,
    /// Sequence number.
    pub sequence_number: u16,
    /// Timestamp.
    pub timestamp: u32,
    /// SSRC (synchronization source).
    pub ssrc: u32,
    /// CSRC list.
    pub csrc: Vec<u32>,
    /// Extension profile.
    pub extension_profile: u16,
    /// Extensions.
    pub extensions: Vec<Extension>,
    /// Payload data.
    pub payload: Bytes,
}

impl Packet {
    /// Creates a new RTP packet.
    #[must_use]
    pub fn new(payload_type: u8, sequence_number: u16, timestamp: u32, ssrc: u32) -> Self {
        Self {
            version: 2,
            padding: false,
            extension: false,
            csrc_count: 0,
            marker: false,
            payload_type,
            sequence_number,
            timestamp,
            ssrc,
            csrc: Vec::new(),
            extension_profile: 0,
            extensions: Vec::new(),
            payload: Bytes::new(),
        }
    }

    /// Sets the marker bit.
    #[must_use]
    pub const fn with_marker(mut self) -> Self {
        self.marker = true;
        self
    }

    /// Sets the payload.
    #[must_use]
    pub fn with_payload(mut self, payload: impl Into<Bytes>) -> Self {
        self.payload = payload.into();
        self
    }

    /// Adds a CSRC.
    #[must_use]
    pub fn with_csrc(mut self, csrc: u32) -> Self {
        self.csrc.push(csrc);
        self.csrc_count = self.csrc.len() as u8;
        self
    }

    /// Encodes the packet to bytes.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        // Byte 0: V(2), P(1), X(1), CC(4)
        let byte0 = (self.version << 6)
            | (u8::from(self.padding) << 5)
            | (u8::from(self.extension) << 4)
            | (self.csrc_count & 0x0F);
        buf.put_u8(byte0);

        // Byte 1: M(1), PT(7)
        let byte1 = (u8::from(self.marker) << 7) | (self.payload_type & 0x7F);
        buf.put_u8(byte1);

        // Sequence number
        buf.put_u16(self.sequence_number);

        // Timestamp
        buf.put_u32(self.timestamp);

        // SSRC
        buf.put_u32(self.ssrc);

        // CSRC list
        for csrc in &self.csrc {
            buf.put_u32(*csrc);
        }

        // Extension (if present)
        if self.extension && !self.extensions.is_empty() {
            buf.put_u16(self.extension_profile);

            let mut ext_buf = BytesMut::new();
            for ext in &self.extensions {
                ext_buf.put_u8(ext.id);
                ext_buf.put_u8(ext.data.len() as u8);
                ext_buf.put(ext.data.clone());
            }

            // Pad to 4-byte boundary
            let padding = (4 - (ext_buf.len() % 4)) % 4;
            for _ in 0..padding {
                ext_buf.put_u8(0);
            }

            buf.put_u16((ext_buf.len() / 4) as u16);
            buf.put(ext_buf);
        }

        // Payload
        buf.put(self.payload.clone());

        buf.freeze()
    }

    /// Parses an RTP packet.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < 12 {
            return Err(NetError::parse(0, "RTP packet too short"));
        }

        let mut cursor = Bytes::copy_from_slice(data);

        // Parse header
        let byte0 = cursor.get_u8();
        let version = (byte0 >> 6) & 0x03;
        let padding = (byte0 & 0x20) != 0;
        let extension = (byte0 & 0x10) != 0;
        let csrc_count = byte0 & 0x0F;

        if version != 2 {
            return Err(NetError::parse(0, "Invalid RTP version"));
        }

        let byte1 = cursor.get_u8();
        let marker = (byte1 & 0x80) != 0;
        let payload_type = byte1 & 0x7F;

        let sequence_number = cursor.get_u16();
        let timestamp = cursor.get_u32();
        let ssrc = cursor.get_u32();

        // Parse CSRC list
        let mut csrc = Vec::new();
        for _ in 0..csrc_count {
            if cursor.remaining() < 4 {
                return Err(NetError::parse(0, "Incomplete CSRC list"));
            }
            csrc.push(cursor.get_u32());
        }

        // Parse extension
        let mut extension_profile = 0;
        let extensions = Vec::new();

        if extension {
            if cursor.remaining() < 4 {
                return Err(NetError::parse(0, "Incomplete extension header"));
            }

            extension_profile = cursor.get_u16();
            let ext_length = cursor.get_u16() as usize * 4;

            if cursor.remaining() < ext_length {
                return Err(NetError::parse(0, "Incomplete extension data"));
            }

            let _ext_data = cursor.copy_to_bytes(ext_length);
            // Parse individual extensions (simplified)
        }

        // Remaining data is payload
        let mut payload = cursor.copy_to_bytes(cursor.remaining());

        // Handle padding
        if padding {
            if payload.is_empty() {
                return Err(NetError::parse(0, "Padding flag set but no payload"));
            }
            let padding_length = payload[payload.len() - 1] as usize;
            if padding_length > payload.len() {
                return Err(NetError::parse(0, "Invalid padding length"));
            }
            payload.truncate(payload.len() - padding_length);
        }

        Ok(Self {
            version,
            padding,
            extension,
            csrc_count,
            marker,
            payload_type,
            sequence_number,
            timestamp,
            ssrc,
            csrc,
            extension_profile,
            extensions,
            payload,
        })
    }

    /// Gets the payload data.
    #[must_use]
    pub fn payload(&self) -> &Bytes {
        &self.payload
    }

    /// Gets the payload size.
    #[must_use]
    pub fn payload_size(&self) -> usize {
        self.payload.len()
    }
}

/// RTP session statistics.
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Packets sent.
    pub packets_sent: u64,
    /// Packets received.
    pub packets_received: u64,
    /// Bytes sent.
    pub bytes_sent: u64,
    /// Bytes received.
    pub bytes_received: u64,
    /// Packets lost.
    pub packets_lost: u64,
    /// Jitter.
    pub jitter: f64,
}

/// RTP session.
pub struct Session {
    /// SSRC.
    ssrc: u32,
    /// Next sequence number.
    next_sequence: u16,
    /// Statistics.
    stats: Statistics,
    /// Last received sequence number (for loss detection).
    last_seq: Option<u16>,
    /// Last received RTP timestamp (for jitter calculation).
    last_rtp_timestamp: Option<u32>,
    /// Last received wall-clock time in microseconds (for jitter calculation).
    last_arrival_time: Option<u64>,
    /// Jitter accumulator (RFC 3550 running estimate, scaled by 16).
    jitter_q4: f64,
}

impl Session {
    /// Creates a new RTP session.
    #[must_use]
    pub fn new(ssrc: u32) -> Self {
        Self {
            ssrc,
            next_sequence: 0,
            stats: Statistics::default(),
            last_seq: None,
            last_rtp_timestamp: None,
            last_arrival_time: None,
            jitter_q4: 0.0,
        }
    }

    /// Creates a packet.
    #[must_use]
    pub fn create_packet(
        &mut self,
        payload_type: u8,
        timestamp: u32,
        payload: impl Into<Bytes>,
    ) -> Packet {
        let seq = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);

        let packet = Packet::new(payload_type, seq, timestamp, self.ssrc).with_payload(payload);

        self.stats.packets_sent += 1;
        self.stats.bytes_sent += packet.payload.len() as u64;

        packet
    }

    /// Returns the current wall-clock time in microseconds.
    fn now_micros() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0)
    }

    /// Processes a received packet.
    ///
    /// Updates jitter using the RFC 3550 running-average algorithm and detects
    /// packet loss from gaps in sequence numbers.
    pub fn process_packet(&mut self, packet: &Packet) {
        self.stats.packets_received += 1;
        self.stats.bytes_received += packet.payload.len() as u64;

        // --- Packet loss detection ---
        // RFC 3550 §A.3: count missing sequence numbers as lost packets.
        if let Some(last_seq) = self.last_seq {
            // Compute forward distance in the 16-bit sequence number space.
            let expected: u16 = last_seq.wrapping_add(1);
            if packet.sequence_number != expected {
                // Handle wrap-around correctly: distance > 0x8000 means reordering/wraparound.
                let gap = packet.sequence_number.wrapping_sub(expected) as u32;
                if gap < 0x8000 {
                    // Normal forward gap – the intervening packets are lost.
                    self.stats.packets_lost += u64::from(gap);
                }
                // Negative gap (reordered/duplicate) is silently ignored.
            }
        }
        self.last_seq = Some(packet.sequence_number);

        // --- Jitter calculation (RFC 3550 §A.8) ---
        // J(i) = J(i-1) + (|D(i-1,i)| - J(i-1)) / 16
        // where D(i-1,i) = (R_i - R_{i-1}) - (S_i - S_{i-1})
        //       R = arrival time in the same clock units as the RTP timestamp
        //       S = RTP send timestamp
        let now = Self::now_micros();
        if let (Some(last_ts), Some(last_arrival)) =
            (self.last_rtp_timestamp, self.last_arrival_time)
        {
            // RTP timestamp difference (in RTP clock units – treat as microseconds for the
            // purpose of this calculation; real code would need the clock-rate).
            let rtp_diff = packet.timestamp.wrapping_sub(last_ts) as i64;
            // Arrival time difference in microseconds.
            let arrival_diff = now.wrapping_sub(last_arrival) as i64;
            // Inter-arrival timing difference.
            let d = (arrival_diff - rtp_diff).abs() as f64;
            // RFC 3550 running estimate.
            self.jitter_q4 += (d - self.jitter_q4) / 16.0;
            self.stats.jitter = self.jitter_q4;
        }
        self.last_rtp_timestamp = Some(packet.timestamp);
        self.last_arrival_time = Some(now);
    }

    /// Gets statistics.
    #[must_use]
    pub const fn stats(&self) -> &Statistics {
        &self.stats
    }

    /// Gets the SSRC.
    #[must_use]
    pub const fn ssrc(&self) -> u32 {
        self.ssrc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_new() {
        let packet = Packet::new(96, 1000, 48000, 12345);
        assert_eq!(packet.version, 2);
        assert_eq!(packet.payload_type, 96);
        assert_eq!(packet.sequence_number, 1000);
        assert_eq!(packet.timestamp, 48000);
        assert_eq!(packet.ssrc, 12345);
    }

    #[test]
    fn test_packet_encode_decode() {
        let packet = Packet::new(96, 1000, 48000, 12345).with_payload(vec![1, 2, 3, 4]);

        let encoded = packet.encode();
        let decoded = Packet::parse(&encoded).expect("should succeed in test");

        assert_eq!(decoded.version, packet.version);
        assert_eq!(decoded.payload_type, packet.payload_type);
        assert_eq!(decoded.sequence_number, packet.sequence_number);
        assert_eq!(decoded.timestamp, packet.timestamp);
        assert_eq!(decoded.ssrc, packet.ssrc);
        assert_eq!(decoded.payload, packet.payload);
    }

    #[test]
    fn test_packet_with_marker() {
        let packet = Packet::new(96, 1000, 48000, 12345).with_marker();

        assert!(packet.marker);
    }

    #[test]
    fn test_session() {
        let mut session = Session::new(12345);
        let packet = session.create_packet(96, 48000, vec![1, 2, 3, 4]);

        assert_eq!(packet.ssrc, 12345);
        assert_eq!(packet.sequence_number, 0);
        assert_eq!(session.stats().packets_sent, 1);

        let packet2 = session.create_packet(96, 48100, vec![5, 6, 7, 8]);
        assert_eq!(packet2.sequence_number, 1);
        assert_eq!(session.stats().packets_sent, 2);
    }

    #[test]
    fn test_parse_invalid_version() {
        let data = vec![0x00, 0x60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // Version 0
        assert!(Packet::parse(&data).is_err());
    }
}

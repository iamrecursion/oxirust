//! RTCP (RTP Control Protocol) implementation.
//!
//! This module implements RTCP packet handling for WebRTC media control.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::error::{NetError, NetResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};

/// RTCP packet type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    /// Sender Report.
    SR = 200,
    /// Receiver Report.
    RR = 201,
    /// Source Description.
    SDES = 202,
    /// Goodbye.
    BYE = 203,
    /// Application-defined.
    APP = 204,
    /// Transport layer feedback.
    RTPFB = 205,
    /// Payload-specific feedback.
    PSFB = 206,
}

impl PacketType {
    /// Parses from byte value.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            200 => Some(Self::SR),
            201 => Some(Self::RR),
            202 => Some(Self::SDES),
            203 => Some(Self::BYE),
            204 => Some(Self::APP),
            205 => Some(Self::RTPFB),
            206 => Some(Self::PSFB),
            _ => None,
        }
    }
}

/// RTCP Sender Report.
#[derive(Debug, Clone)]
pub struct SenderReport {
    /// SSRC of sender.
    pub ssrc: u32,
    /// NTP timestamp (most significant word).
    pub ntp_timestamp_msw: u32,
    /// NTP timestamp (least significant word).
    pub ntp_timestamp_lsw: u32,
    /// RTP timestamp.
    pub rtp_timestamp: u32,
    /// Sender packet count.
    pub sender_packet_count: u32,
    /// Sender octet count.
    pub sender_octet_count: u32,
    /// Report blocks.
    pub reports: Vec<ReportBlock>,
}

impl SenderReport {
    /// Creates a new sender report.
    #[must_use]
    pub const fn new(ssrc: u32) -> Self {
        Self {
            ssrc,
            ntp_timestamp_msw: 0,
            ntp_timestamp_lsw: 0,
            rtp_timestamp: 0,
            sender_packet_count: 0,
            sender_octet_count: 0,
            reports: Vec::new(),
        }
    }

    /// Sets NTP timestamp.
    #[must_use]
    pub const fn with_ntp_timestamp(mut self, msw: u32, lsw: u32) -> Self {
        self.ntp_timestamp_msw = msw;
        self.ntp_timestamp_lsw = lsw;
        self
    }

    /// Sets RTP timestamp.
    #[must_use]
    pub const fn with_rtp_timestamp(mut self, timestamp: u32) -> Self {
        self.rtp_timestamp = timestamp;
        self
    }

    /// Sets packet count.
    #[must_use]
    pub const fn with_packet_count(mut self, count: u32) -> Self {
        self.sender_packet_count = count;
        self
    }

    /// Sets octet count.
    #[must_use]
    pub const fn with_octet_count(mut self, count: u32) -> Self {
        self.sender_octet_count = count;
        self
    }

    /// Adds a report block.
    #[must_use]
    pub fn with_report(mut self, report: ReportBlock) -> Self {
        self.reports.push(report);
        self
    }

    /// Encodes the sender report.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        // RTCP header
        let rc = self.reports.len().min(31) as u8;
        buf.put_u8((2 << 6) | rc); // V=2, P=0, RC
        buf.put_u8(PacketType::SR as u8);

        // Length in 32-bit words - 1
        let length = 6 + self.reports.len() * 6;
        buf.put_u16(length as u16);

        // SR fields
        buf.put_u32(self.ssrc);
        buf.put_u32(self.ntp_timestamp_msw);
        buf.put_u32(self.ntp_timestamp_lsw);
        buf.put_u32(self.rtp_timestamp);
        buf.put_u32(self.sender_packet_count);
        buf.put_u32(self.sender_octet_count);

        // Report blocks
        for report in &self.reports {
            buf.put(report.encode());
        }

        buf.freeze()
    }

    /// Parses a sender report.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < 28 {
            return Err(NetError::parse(0, "SR packet too short"));
        }

        let mut cursor = Bytes::copy_from_slice(data);

        let byte0 = cursor.get_u8();
        let rc = byte0 & 0x1F;
        let _pt = cursor.get_u8();
        let _length = cursor.get_u16();

        let ssrc = cursor.get_u32();
        let ntp_timestamp_msw = cursor.get_u32();
        let ntp_timestamp_lsw = cursor.get_u32();
        let rtp_timestamp = cursor.get_u32();
        let sender_packet_count = cursor.get_u32();
        let sender_octet_count = cursor.get_u32();

        let mut reports = Vec::new();
        for _ in 0..rc {
            if cursor.remaining() >= 24 {
                reports.push(ReportBlock::parse(&mut cursor)?);
            }
        }

        Ok(Self {
            ssrc,
            ntp_timestamp_msw,
            ntp_timestamp_lsw,
            rtp_timestamp,
            sender_packet_count,
            sender_octet_count,
            reports,
        })
    }
}

/// RTCP Receiver Report.
#[derive(Debug, Clone)]
pub struct ReceiverReport {
    /// SSRC of receiver.
    pub ssrc: u32,
    /// Report blocks.
    pub reports: Vec<ReportBlock>,
}

impl ReceiverReport {
    /// Creates a new receiver report.
    #[must_use]
    pub const fn new(ssrc: u32) -> Self {
        Self {
            ssrc,
            reports: Vec::new(),
        }
    }

    /// Adds a report block.
    #[must_use]
    pub fn with_report(mut self, report: ReportBlock) -> Self {
        self.reports.push(report);
        self
    }

    /// Encodes the receiver report.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        // RTCP header
        let rc = self.reports.len().min(31) as u8;
        buf.put_u8((2 << 6) | rc); // V=2, P=0, RC
        buf.put_u8(PacketType::RR as u8);

        // Length in 32-bit words - 1
        let length = 1 + self.reports.len() * 6;
        buf.put_u16(length as u16);

        // RR fields
        buf.put_u32(self.ssrc);

        // Report blocks
        for report in &self.reports {
            buf.put(report.encode());
        }

        buf.freeze()
    }

    /// Parses a receiver report.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < 8 {
            return Err(NetError::parse(0, "RR packet too short"));
        }

        let mut cursor = Bytes::copy_from_slice(data);

        let byte0 = cursor.get_u8();
        let rc = byte0 & 0x1F;
        let _pt = cursor.get_u8();
        let _length = cursor.get_u16();

        let ssrc = cursor.get_u32();

        let mut reports = Vec::new();
        for _ in 0..rc {
            if cursor.remaining() >= 24 {
                reports.push(ReportBlock::parse(&mut cursor)?);
            }
        }

        Ok(Self { ssrc, reports })
    }
}

/// RTCP Report Block.
#[derive(Debug, Clone)]
pub struct ReportBlock {
    /// SSRC of source.
    pub ssrc: u32,
    /// Fraction lost.
    pub fraction_lost: u8,
    /// Cumulative packets lost.
    pub packets_lost: i32,
    /// Extended highest sequence number.
    pub extended_highest_seq: u32,
    /// Interarrival jitter.
    pub jitter: u32,
    /// Last SR timestamp.
    pub last_sr: u32,
    /// Delay since last SR.
    pub delay_since_last_sr: u32,
}

impl ReportBlock {
    /// Creates a new report block.
    #[must_use]
    pub const fn new(ssrc: u32) -> Self {
        Self {
            ssrc,
            fraction_lost: 0,
            packets_lost: 0,
            extended_highest_seq: 0,
            jitter: 0,
            last_sr: 0,
            delay_since_last_sr: 0,
        }
    }

    /// Encodes the report block.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        buf.put_u32(self.ssrc);
        buf.put_u8(self.fraction_lost);

        // Packets lost (24-bit signed)
        let packets_lost = self.packets_lost & 0x00FF_FFFF;
        buf.put_u8((packets_lost >> 16) as u8);
        buf.put_u8((packets_lost >> 8) as u8);
        buf.put_u8(packets_lost as u8);

        buf.put_u32(self.extended_highest_seq);
        buf.put_u32(self.jitter);
        buf.put_u32(self.last_sr);
        buf.put_u32(self.delay_since_last_sr);

        buf.freeze()
    }

    /// Parses a report block.
    pub fn parse(cursor: &mut Bytes) -> NetResult<Self> {
        if cursor.remaining() < 24 {
            return Err(NetError::parse(0, "Report block too short"));
        }

        let ssrc = cursor.get_u32();
        let fraction_lost = cursor.get_u8();

        // Packets lost (24-bit signed)
        let lost_hi = cursor.get_u8();
        let lost_mid = cursor.get_u8();
        let lost_lo = cursor.get_u8();
        let packets_lost =
            ((i32::from(lost_hi) << 16) | (i32::from(lost_mid) << 8) | i32::from(lost_lo)) as i32;

        let extended_highest_seq = cursor.get_u32();
        let jitter = cursor.get_u32();
        let last_sr = cursor.get_u32();
        let delay_since_last_sr = cursor.get_u32();

        Ok(Self {
            ssrc,
            fraction_lost,
            packets_lost,
            extended_highest_seq,
            jitter,
            last_sr,
            delay_since_last_sr,
        })
    }
}

/// RTCP packet.
#[derive(Debug, Clone)]
pub enum Packet {
    /// Sender report.
    SR(SenderReport),
    /// Receiver report.
    RR(ReceiverReport),
}

impl Packet {
    /// Encodes the packet.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        match self {
            Self::SR(sr) => sr.encode(),
            Self::RR(rr) => rr.encode(),
        }
    }

    /// Parses a packet.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < 4 {
            return Err(NetError::parse(0, "RTCP packet too short"));
        }

        let packet_type = data[1];

        match PacketType::from_u8(packet_type) {
            Some(PacketType::SR) => Ok(Self::SR(SenderReport::parse(data)?)),
            Some(PacketType::RR) => Ok(Self::RR(ReceiverReport::parse(data)?)),
            _ => Err(NetError::parse(0, "Unsupported RTCP packet type")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_type() {
        assert_eq!(PacketType::from_u8(200), Some(PacketType::SR));
        assert_eq!(PacketType::from_u8(201), Some(PacketType::RR));
    }

    #[test]
    fn test_sender_report() {
        let sr = SenderReport::new(12345)
            .with_ntp_timestamp(100, 200)
            .with_rtp_timestamp(48000)
            .with_packet_count(1000)
            .with_octet_count(50000);

        assert_eq!(sr.ssrc, 12345);
        assert_eq!(sr.ntp_timestamp_msw, 100);
        assert_eq!(sr.rtp_timestamp, 48000);
    }

    #[test]
    fn test_sender_report_encode_decode() {
        let sr = SenderReport::new(12345)
            .with_ntp_timestamp(100, 200)
            .with_rtp_timestamp(48000)
            .with_packet_count(1000)
            .with_octet_count(50000);

        let encoded = sr.encode();
        let decoded = SenderReport::parse(&encoded).expect("should succeed in test");

        assert_eq!(decoded.ssrc, sr.ssrc);
        assert_eq!(decoded.ntp_timestamp_msw, sr.ntp_timestamp_msw);
        assert_eq!(decoded.rtp_timestamp, sr.rtp_timestamp);
    }

    #[test]
    fn test_receiver_report() {
        let rr = ReceiverReport::new(12345);
        assert_eq!(rr.ssrc, 12345);
        assert!(rr.reports.is_empty());
    }

    #[test]
    fn test_receiver_report_encode_decode() {
        let rr = ReceiverReport::new(12345);
        let encoded = rr.encode();
        let decoded = ReceiverReport::parse(&encoded).expect("should succeed in test");

        assert_eq!(decoded.ssrc, rr.ssrc);
        assert_eq!(decoded.reports.len(), 0);
    }

    #[test]
    fn test_report_block() {
        let block = ReportBlock::new(54321);
        let encoded = block.encode();
        assert_eq!(encoded.len(), 24);
    }

    #[test]
    fn test_packet_parse_sr() {
        let sr = SenderReport::new(12345).with_packet_count(100);
        let encoded = sr.encode();

        let packet = Packet::parse(&encoded).expect("should succeed in test");
        match packet {
            Packet::SR(parsed_sr) => {
                assert_eq!(parsed_sr.ssrc, 12345);
            }
            _ => panic!("Expected SR packet"),
        }
    }
}

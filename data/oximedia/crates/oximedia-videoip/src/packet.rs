//! Packet format and serialization for video-over-IP protocol.

use crate::error::{VideoIpError, VideoIpResult};
use crate::types::StreamType;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::time::SystemTime;

/// Magic number for packet identification ("OXVP" = `OxiMedia` Video Protocol).
pub const MAGIC: u32 = 0x4F585650;

/// Current protocol version.
pub const VERSION: u8 = 1;

/// Maximum packet size (jumbo frames).
pub const MAX_PACKET_SIZE: usize = 9000;

/// Maximum payload size.
pub const MAX_PAYLOAD_SIZE: usize = MAX_PACKET_SIZE - PacketHeader::SIZE;

bitflags::bitflags! {
    /// Packet flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PacketFlags: u8 {
        /// Packet contains video data.
        const VIDEO = 0x01;
        /// Packet contains audio data.
        const AUDIO = 0x02;
        /// Packet contains metadata.
        const METADATA = 0x04;
        /// Packet is a keyframe/IDR.
        const KEYFRAME = 0x08;
        /// Packet is an FEC parity packet.
        const FEC = 0x10;
        /// Last packet in frame.
        const END_OF_FRAME = 0x20;
        /// First packet in frame.
        const START_OF_FRAME = 0x40;
    }
}

/// Packet header (16 bytes fixed size).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketHeader {
    /// Magic number for validation.
    pub magic: u32,
    /// Protocol version.
    pub version: u8,
    /// Packet flags.
    pub flags: PacketFlags,
    /// Sequence number (wraps at `u16::MAX`).
    pub sequence: u16,
    /// Timestamp in microseconds.
    pub timestamp: u64,
    /// Stream type identifier.
    pub stream_type: StreamType,
    /// Payload size.
    pub payload_size: u16,
}

impl PacketHeader {
    /// Size of the header in bytes.
    pub const SIZE: usize = 20;

    /// Creates a new packet header.
    #[must_use]
    pub const fn new(
        flags: PacketFlags,
        sequence: u16,
        timestamp: u64,
        stream_type: StreamType,
        payload_size: u16,
    ) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            flags,
            sequence,
            timestamp,
            stream_type,
            payload_size,
        }
    }

    /// Encodes the header into bytes.
    pub fn encode(&self, buf: &mut BytesMut) {
        buf.reserve(Self::SIZE);
        buf.put_u32(self.magic);
        buf.put_u8(self.version);
        buf.put_u8(self.flags.bits());
        buf.put_u16(self.sequence);
        buf.put_u64(self.timestamp);
        buf.put_u8(self.stream_type.to_id());
        buf.put_u8(0); // Reserved
        buf.put_u16(self.payload_size);
    }

    /// Decodes a header from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is invalid or malformed.
    pub fn decode(buf: &mut impl Buf) -> VideoIpResult<Self> {
        if buf.remaining() < Self::SIZE {
            return Err(VideoIpError::InvalidPacket(
                "insufficient data for header".to_string(),
            ));
        }

        let magic = buf.get_u32();
        if magic != MAGIC {
            return Err(VideoIpError::InvalidPacket(format!(
                "invalid magic: 0x{magic:08X}"
            )));
        }

        let version = buf.get_u8();
        if version != VERSION {
            return Err(VideoIpError::InvalidPacket(format!(
                "unsupported version: {version}"
            )));
        }

        let flags = PacketFlags::from_bits_truncate(buf.get_u8());
        let sequence = buf.get_u16();
        let timestamp = buf.get_u64();
        let stream_type = StreamType::from_id(buf.get_u8());
        let _reserved = buf.get_u8();
        let payload_size = buf.get_u16();

        Ok(Self {
            magic,
            version,
            flags,
            sequence,
            timestamp,
            stream_type,
            payload_size,
        })
    }

    /// Validates the header.
    ///
    /// # Errors
    ///
    /// Returns an error if the header is invalid.
    pub fn validate(&self) -> VideoIpResult<()> {
        if self.magic != MAGIC {
            return Err(VideoIpError::InvalidPacket(format!(
                "invalid magic: 0x{:08X}",
                self.magic
            )));
        }

        if self.version != VERSION {
            return Err(VideoIpError::InvalidPacket(format!(
                "unsupported version: {}",
                self.version
            )));
        }

        if usize::from(self.payload_size) > MAX_PAYLOAD_SIZE {
            return Err(VideoIpError::PacketTooLarge {
                size: usize::from(self.payload_size),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        Ok(())
    }
}

/// Complete packet with header and payload.
#[derive(Debug, Clone)]
pub struct Packet {
    /// Packet header.
    pub header: PacketHeader,
    /// Packet payload.
    pub payload: Bytes,
}

impl Packet {
    /// Creates a new packet.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload is too large.
    pub fn new(
        flags: PacketFlags,
        sequence: u16,
        timestamp: u64,
        stream_type: StreamType,
        payload: Bytes,
    ) -> VideoIpResult<Self> {
        if payload.len() > MAX_PAYLOAD_SIZE {
            return Err(VideoIpError::PacketTooLarge {
                size: payload.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        let header = PacketHeader::new(
            flags,
            sequence,
            timestamp,
            stream_type,
            payload.len() as u16,
        );

        Ok(Self { header, payload })
    }

    /// Encodes the packet into bytes.
    #[must_use]
    pub fn encode(&self) -> BytesMut {
        let mut buf = BytesMut::with_capacity(PacketHeader::SIZE + self.payload.len());
        self.header.encode(&mut buf);
        buf.put(self.payload.clone());
        buf
    }

    /// Decodes a packet from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is invalid or malformed.
    pub fn decode(mut buf: impl Buf) -> VideoIpResult<Self> {
        let header = PacketHeader::decode(&mut buf)?;
        header.validate()?;

        let payload_size = usize::from(header.payload_size);
        if buf.remaining() < payload_size {
            return Err(VideoIpError::InvalidPacket(format!(
                "insufficient payload data: expected {payload_size}, got {}",
                buf.remaining()
            )));
        }

        let payload = buf.copy_to_bytes(payload_size);

        Ok(Self { header, payload })
    }

    /// Returns the total packet size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        PacketHeader::SIZE + self.payload.len()
    }
}

/// Packet builder for easier packet construction.
pub struct PacketBuilder {
    flags: PacketFlags,
    sequence: u16,
    timestamp: u64,
    stream_type: StreamType,
}

impl PacketBuilder {
    /// Creates a new packet builder with the given sequence number.
    #[must_use]
    pub const fn new(sequence: u16) -> Self {
        Self {
            flags: PacketFlags::empty(),
            sequence,
            timestamp: 0,
            stream_type: StreamType::Program,
        }
    }

    /// Sets the timestamp to the current time.
    #[must_use]
    pub fn with_current_timestamp(mut self) -> Self {
        self.timestamp = current_timestamp_micros();
        self
    }

    /// Sets a specific timestamp.
    #[must_use]
    pub const fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Sets the stream type.
    #[must_use]
    pub const fn with_stream_type(mut self, stream_type: StreamType) -> Self {
        self.stream_type = stream_type;
        self
    }

    /// Marks this as a video packet.
    #[must_use]
    pub const fn video(mut self) -> Self {
        self.flags = self.flags.union(PacketFlags::VIDEO);
        self
    }

    /// Marks this as an audio packet.
    #[must_use]
    pub const fn audio(mut self) -> Self {
        self.flags = self.flags.union(PacketFlags::AUDIO);
        self
    }

    /// Marks this as a metadata packet.
    #[must_use]
    pub const fn metadata(mut self) -> Self {
        self.flags = self.flags.union(PacketFlags::METADATA);
        self
    }

    /// Marks this as a keyframe.
    #[must_use]
    pub const fn keyframe(mut self) -> Self {
        self.flags = self.flags.union(PacketFlags::KEYFRAME);
        self
    }

    /// Marks this as an FEC packet.
    #[must_use]
    pub const fn fec(mut self) -> Self {
        self.flags = self.flags.union(PacketFlags::FEC);
        self
    }

    /// Marks this as the start of a frame.
    #[must_use]
    pub const fn start_of_frame(mut self) -> Self {
        self.flags = self.flags.union(PacketFlags::START_OF_FRAME);
        self
    }

    /// Marks this as the end of a frame.
    #[must_use]
    pub const fn end_of_frame(mut self) -> Self {
        self.flags = self.flags.union(PacketFlags::END_OF_FRAME);
        self
    }

    /// Builds the packet with the given payload.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload is too large.
    pub fn build(self, payload: Bytes) -> VideoIpResult<Packet> {
        Packet::new(
            self.flags,
            self.sequence,
            self.timestamp,
            self.stream_type,
            payload,
        )
    }
}

/// Returns the current timestamp in microseconds since UNIX epoch.
#[must_use]
pub fn current_timestamp_micros() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_header_encode_decode() {
        let header = PacketHeader::new(
            PacketFlags::VIDEO | PacketFlags::KEYFRAME,
            123,
            456789,
            StreamType::Program,
            1000,
        );

        let mut buf = BytesMut::new();
        header.encode(&mut buf);

        assert_eq!(buf.len(), PacketHeader::SIZE);

        let decoded = PacketHeader::decode(&mut buf).expect("should succeed in test");
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_packet_header_validate() {
        let header = PacketHeader::new(PacketFlags::VIDEO, 0, 0, StreamType::Program, 1000);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_invalid_magic() {
        let mut header = PacketHeader::new(PacketFlags::VIDEO, 0, 0, StreamType::Program, 1000);
        header.magic = 0xDEADBEEF;
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_packet_too_large() {
        let payload = Bytes::from(vec![0u8; MAX_PAYLOAD_SIZE + 1]);
        let result = Packet::new(PacketFlags::VIDEO, 0, 0, StreamType::Program, payload);
        assert!(result.is_err());
    }

    #[test]
    fn test_packet_encode_decode() {
        let payload = Bytes::from_static(b"Hello, World!");
        let packet = Packet::new(
            PacketFlags::VIDEO | PacketFlags::KEYFRAME,
            42,
            123456789,
            StreamType::Program,
            payload.clone(),
        )
        .expect("should succeed in test");

        let encoded = packet.encode();
        let decoded = Packet::decode(&encoded[..]).expect("should succeed in test");

        assert_eq!(decoded.header.sequence, 42);
        assert_eq!(decoded.header.timestamp, 123456789);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_packet_builder() {
        let packet = PacketBuilder::new(10)
            .video()
            .keyframe()
            .with_timestamp(12345)
            .with_stream_type(StreamType::Preview)
            .build(Bytes::from_static(b"test"))
            .expect("should succeed in test");

        assert!(packet.header.flags.contains(PacketFlags::VIDEO));
        assert!(packet.header.flags.contains(PacketFlags::KEYFRAME));
        assert_eq!(packet.header.sequence, 10);
        assert_eq!(packet.header.timestamp, 12345);
        assert_eq!(packet.header.stream_type, StreamType::Preview);
    }

    #[test]
    fn test_packet_flags() {
        let flags = PacketFlags::VIDEO | PacketFlags::AUDIO;
        assert!(flags.contains(PacketFlags::VIDEO));
        assert!(flags.contains(PacketFlags::AUDIO));
        assert!(!flags.contains(PacketFlags::METADATA));
    }

    #[test]
    fn test_stream_type_roundtrip() {
        for i in 0..=255u8 {
            let stream_type = StreamType::from_id(i);
            assert_eq!(stream_type.to_id(), i);
        }
    }

    #[test]
    fn test_current_timestamp() {
        let ts = current_timestamp_micros();
        assert!(ts > 0);
    }
}

//! SRT packet definitions.
//!
//! This module defines the packet structures used in SRT protocol.

use crate::error::{NetError, NetResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};

/// SRT packet type flag in header.
pub const PACKET_TYPE_DATA: u32 = 0;
/// Control packet type flag.
pub const PACKET_TYPE_CONTROL: u32 = 1 << 31;

/// Packet position flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketFlags {
    /// Position in message (2 bits).
    pub position: PacketPosition,
    /// Order flag - must be delivered in order.
    pub order: bool,
    /// Key-based encryption flag.
    pub encryption: EncryptionFlag,
    /// Retransmitted packet flag.
    pub retransmitted: bool,
}

/// Packet position within a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PacketPosition {
    /// Middle packet of message.
    #[default]
    Middle = 0b00,
    /// Last packet of message.
    Last = 0b01,
    /// First packet of message.
    First = 0b10,
    /// Single packet message.
    Single = 0b11,
}

impl PacketPosition {
    /// Creates from 2-bit value.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Self::Middle,
            0b01 => Self::Last,
            0b10 => Self::First,
            _ => Self::Single,
        }
    }

    /// Returns true if this is the first packet (first or single).
    #[must_use]
    pub const fn is_first(&self) -> bool {
        matches!(self, Self::First | Self::Single)
    }

    /// Returns true if this is the last packet (last or single).
    #[must_use]
    pub const fn is_last(&self) -> bool {
        matches!(self, Self::Last | Self::Single)
    }
}

/// Encryption flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncryptionFlag {
    /// No encryption.
    #[default]
    None = 0b00,
    /// Even key.
    EvenKey = 0b01,
    /// Odd key.
    OddKey = 0b10,
}

impl EncryptionFlag {
    /// Creates from 2-bit value.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b01 => Self::EvenKey,
            0b10 => Self::OddKey,
            _ => Self::None,
        }
    }
}

impl Default for PacketFlags {
    fn default() -> Self {
        Self {
            position: PacketPosition::Single,
            order: false,
            encryption: EncryptionFlag::None,
            retransmitted: false,
        }
    }
}

impl PacketFlags {
    /// Creates from byte value.
    #[must_use]
    pub fn from_byte(b: u8) -> Self {
        Self {
            position: PacketPosition::from_bits(b >> 6),
            order: (b & 0x20) != 0,
            encryption: EncryptionFlag::from_bits((b >> 3) & 0b11),
            retransmitted: (b & 0x04) != 0,
        }
    }

    /// Converts to byte value.
    #[must_use]
    pub fn to_byte(&self) -> u8 {
        let mut b = (self.position as u8) << 6;
        if self.order {
            b |= 0x20;
        }
        b |= (self.encryption as u8) << 3;
        if self.retransmitted {
            b |= 0x04;
        }
        b
    }
}

/// SRT data packet.
#[derive(Debug, Clone)]
pub struct DataPacket {
    /// Packet sequence number.
    pub sequence_number: u32,
    /// Packet flags.
    pub flags: PacketFlags,
    /// Message number.
    pub message_number: u32,
    /// Timestamp (microseconds).
    pub timestamp: u32,
    /// Destination socket ID.
    pub dst_socket_id: u32,
    /// Payload data.
    pub payload: Bytes,
}

impl DataPacket {
    /// Creates a new data packet.
    #[must_use]
    pub fn new(sequence_number: u32, payload: Bytes) -> Self {
        Self {
            sequence_number,
            flags: PacketFlags::default(),
            message_number: 0,
            timestamp: 0,
            dst_socket_id: 0,
            payload,
        }
    }

    /// Sets the timestamp.
    #[must_use]
    pub const fn with_timestamp(mut self, timestamp: u32) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Sets the destination socket ID.
    #[must_use]
    pub const fn with_dst_socket(mut self, socket_id: u32) -> Self {
        self.dst_socket_id = socket_id;
        self
    }

    /// Sets the flags.
    #[must_use]
    pub const fn with_flags(mut self, flags: PacketFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Encodes the packet to bytes.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(16 + self.payload.len());

        // First 4 bytes: sequence number (bit 31 = 0 for data)
        buf.put_u32(self.sequence_number & 0x7FFF_FFFF);

        // Flags and message number
        let flags_byte = self.flags.to_byte();
        buf.put_u8(flags_byte);

        // Message number (lower 26 bits)
        let msg_bytes = self.message_number & 0x03FF_FFFF;
        buf.put_u8(((msg_bytes >> 16) & 0xFF) as u8);
        buf.put_u8(((msg_bytes >> 8) & 0xFF) as u8);
        buf.put_u8((msg_bytes & 0xFF) as u8);

        // Timestamp
        buf.put_u32(self.timestamp);

        // Destination socket ID
        buf.put_u32(self.dst_socket_id);

        // Payload
        buf.put_slice(&self.payload);

        buf.freeze()
    }

    /// Decodes a data packet from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is malformed.
    pub fn decode(data: &[u8]) -> NetResult<Self> {
        if data.len() < 16 {
            return Err(NetError::parse(0, "Data packet too short"));
        }

        let mut buf = data;
        let seq = buf.get_u32() & 0x7FFF_FFFF;

        let flags_byte = buf.get_u8();
        let flags = PacketFlags::from_byte(flags_byte);

        let msg_hi = buf.get_u8() as u32;
        let msg_mid = buf.get_u8() as u32;
        let msg_lo = buf.get_u8() as u32;
        let message_number = (msg_hi << 16) | (msg_mid << 8) | msg_lo;

        let timestamp = buf.get_u32();
        let dst_socket_id = buf.get_u32();

        let payload = Bytes::copy_from_slice(buf);

        Ok(Self {
            sequence_number: seq,
            flags,
            message_number,
            timestamp,
            dst_socket_id,
            payload,
        })
    }
}

/// Control packet types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ControlType {
    /// Handshake.
    Handshake = 0x0000,
    /// Keep-alive.
    Keepalive = 0x0001,
    /// ACK (acknowledgement).
    Ack = 0x0002,
    /// NAK (negative acknowledgement).
    Nak = 0x0003,
    /// Congestion warning.
    CongestionWarning = 0x0004,
    /// Shutdown.
    Shutdown = 0x0005,
    /// ACK-ACK.
    AckAck = 0x0006,
    /// Drop request.
    DropReq = 0x0007,
    /// Peer error.
    PeerError = 0x0008,
    /// User-defined.
    UserDefined = 0x7FFF,
}

impl ControlType {
    /// Creates from u16 value.
    #[must_use]
    pub const fn from_u16(v: u16) -> Option<Self> {
        match v {
            0x0000 => Some(Self::Handshake),
            0x0001 => Some(Self::Keepalive),
            0x0002 => Some(Self::Ack),
            0x0003 => Some(Self::Nak),
            0x0004 => Some(Self::CongestionWarning),
            0x0005 => Some(Self::Shutdown),
            0x0006 => Some(Self::AckAck),
            0x0007 => Some(Self::DropReq),
            0x0008 => Some(Self::PeerError),
            0x7FFF => Some(Self::UserDefined),
            _ => None,
        }
    }
}

/// Handshake extension types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum HandshakeExtension {
    /// SRT handshake request.
    HsReq = 1,
    /// SRT handshake response.
    HsRsp = 2,
    /// Key material request.
    KmReq = 3,
    /// Key material response.
    KmRsp = 4,
    /// Stream ID.
    Sid = 5,
    /// Congestion control.
    Congestion = 6,
    /// Filter.
    Filter = 7,
    /// Group.
    Group = 8,
}

/// Handshake information.
#[derive(Debug, Clone, Default)]
pub struct HandshakeInfo {
    /// Version (4 for SRT 1.0+).
    pub version: u32,
    /// Encryption field.
    pub encryption: u16,
    /// Extension field.
    pub extension: u16,
    /// Initial sequence number.
    pub initial_seq: u32,
    /// Maximum transmission unit size.
    pub mtu: u32,
    /// Maximum flow window size.
    pub flow_window: u32,
    /// Handshake type.
    pub handshake_type: i32,
    /// SRT socket ID.
    pub socket_id: u32,
    /// SYN cookie.
    pub syn_cookie: u32,
    /// Peer IP address (for rendez-vous).
    pub peer_ip: [u8; 16],
}

impl HandshakeInfo {
    /// Creates a new handshake info.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: 5, // SRT version
            mtu: 1500,
            flow_window: 8192,
            ..Default::default()
        }
    }

    /// Handshake type: waveahand (caller initial).
    pub const TYPE_WAVEAHAND: i32 = 0;
    /// Handshake type: induction (listener response).
    pub const TYPE_INDUCTION: i32 = 1;
    /// Handshake type: conclusion (final).
    pub const TYPE_CONCLUSION: i32 = -1;
    /// Handshake type: agreement (established).
    pub const TYPE_AGREEMENT: i32 = -2;

    /// Encodes the handshake info.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(48);

        buf.put_u32(self.version);
        buf.put_u16(self.encryption);
        buf.put_u16(self.extension);
        buf.put_u32(self.initial_seq);
        buf.put_u32(self.mtu);
        buf.put_u32(self.flow_window);
        buf.put_i32(self.handshake_type);
        buf.put_u32(self.socket_id);
        buf.put_u32(self.syn_cookie);
        buf.put_slice(&self.peer_ip);

        buf.freeze()
    }

    /// Decodes handshake info.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is malformed.
    pub fn decode(data: &[u8]) -> NetResult<Self> {
        if data.len() < 48 {
            return Err(NetError::parse(0, "Handshake too short"));
        }

        let mut buf = data;
        let version = buf.get_u32();
        let encryption = buf.get_u16();
        let extension = buf.get_u16();
        let initial_seq = buf.get_u32();
        let mtu = buf.get_u32();
        let flow_window = buf.get_u32();
        let handshake_type = buf.get_i32();
        let socket_id = buf.get_u32();
        let syn_cookie = buf.get_u32();

        let mut peer_ip = [0u8; 16];
        peer_ip.copy_from_slice(&buf[..16]);

        Ok(Self {
            version,
            encryption,
            extension,
            initial_seq,
            mtu,
            flow_window,
            handshake_type,
            socket_id,
            syn_cookie,
            peer_ip,
        })
    }
}

/// SRT control packet.
#[derive(Debug, Clone)]
pub struct ControlPacket {
    /// Control type.
    pub control_type: ControlType,
    /// Subtype.
    pub subtype: u16,
    /// Type-specific information.
    pub type_info: u32,
    /// Timestamp (microseconds).
    pub timestamp: u32,
    /// Destination socket ID.
    pub dst_socket_id: u32,
    /// Control information field.
    pub payload: Bytes,
}

impl ControlPacket {
    /// Creates a new control packet.
    #[must_use]
    pub fn new(control_type: ControlType) -> Self {
        Self {
            control_type,
            subtype: 0,
            type_info: 0,
            timestamp: 0,
            dst_socket_id: 0,
            payload: Bytes::new(),
        }
    }

    /// Creates a keepalive packet.
    #[must_use]
    pub fn keepalive(dst_socket_id: u32) -> Self {
        Self {
            control_type: ControlType::Keepalive,
            subtype: 0,
            type_info: 0,
            timestamp: 0,
            dst_socket_id,
            payload: Bytes::new(),
        }
    }

    /// Creates an ACK packet.
    #[must_use]
    pub fn ack(ack_number: u32, dst_socket_id: u32) -> Self {
        Self {
            control_type: ControlType::Ack,
            subtype: 0,
            type_info: ack_number,
            timestamp: 0,
            dst_socket_id,
            payload: Bytes::new(),
        }
    }

    /// Creates a NAK packet.
    #[must_use]
    pub fn nak(lost_packets: &[u32], dst_socket_id: u32) -> Self {
        let mut payload = BytesMut::with_capacity(lost_packets.len() * 4);
        for seq in lost_packets {
            payload.put_u32(*seq);
        }
        Self {
            control_type: ControlType::Nak,
            subtype: 0,
            type_info: 0,
            timestamp: 0,
            dst_socket_id,
            payload: payload.freeze(),
        }
    }

    /// Creates a shutdown packet.
    #[must_use]
    pub fn shutdown(dst_socket_id: u32) -> Self {
        Self {
            control_type: ControlType::Shutdown,
            subtype: 0,
            type_info: 0,
            timestamp: 0,
            dst_socket_id,
            payload: Bytes::new(),
        }
    }

    /// Creates a handshake packet.
    #[must_use]
    pub fn handshake(info: &HandshakeInfo, dst_socket_id: u32) -> Self {
        Self {
            control_type: ControlType::Handshake,
            subtype: 0,
            type_info: 0,
            timestamp: 0,
            dst_socket_id,
            payload: info.encode(),
        }
    }

    /// Sets the timestamp.
    #[must_use]
    pub const fn with_timestamp(mut self, timestamp: u32) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Encodes the control packet.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(16 + self.payload.len());

        // First 4 bytes: control type (bit 31 = 1 for control)
        let first =
            PACKET_TYPE_CONTROL | ((self.control_type as u32) << 16) | (self.subtype as u32);
        buf.put_u32(first);

        // Type-specific information
        buf.put_u32(self.type_info);

        // Timestamp
        buf.put_u32(self.timestamp);

        // Destination socket ID
        buf.put_u32(self.dst_socket_id);

        // Payload (Control Information Field)
        buf.put_slice(&self.payload);

        buf.freeze()
    }

    /// Decodes a control packet.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is malformed.
    pub fn decode(data: &[u8]) -> NetResult<Self> {
        if data.len() < 16 {
            return Err(NetError::parse(0, "Control packet too short"));
        }

        let mut buf = data;
        let first = buf.get_u32();

        if (first & PACKET_TYPE_CONTROL) == 0 {
            return Err(NetError::parse(0, "Not a control packet"));
        }

        let control_type_raw = ((first >> 16) & 0x7FFF) as u16;
        let control_type = ControlType::from_u16(control_type_raw).ok_or_else(|| {
            NetError::parse(0, format!("Unknown control type: {control_type_raw}"))
        })?;
        let subtype = (first & 0xFFFF) as u16;

        let type_info = buf.get_u32();
        let timestamp = buf.get_u32();
        let dst_socket_id = buf.get_u32();

        let payload = Bytes::copy_from_slice(buf);

        Ok(Self {
            control_type,
            subtype,
            type_info,
            timestamp,
            dst_socket_id,
            payload,
        })
    }
}

/// SRT packet (data or control).
#[derive(Debug, Clone)]
pub enum SrtPacket {
    /// Data packet.
    Data(DataPacket),
    /// Control packet.
    Control(ControlPacket),
}

impl SrtPacket {
    /// Decodes an SRT packet.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is malformed.
    pub fn decode(data: &[u8]) -> NetResult<Self> {
        if data.len() < 4 {
            return Err(NetError::parse(0, "Packet too short"));
        }

        let first = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);

        if (first & PACKET_TYPE_CONTROL) != 0 {
            Ok(Self::Control(ControlPacket::decode(data)?))
        } else {
            Ok(Self::Data(DataPacket::decode(data)?))
        }
    }

    /// Encodes the packet.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        match self {
            Self::Data(d) => d.encode(),
            Self::Control(c) => c.encode(),
        }
    }

    /// Returns true if this is a data packet.
    #[must_use]
    pub const fn is_data(&self) -> bool {
        matches!(self, Self::Data(_))
    }

    /// Returns true if this is a control packet.
    #[must_use]
    pub const fn is_control(&self) -> bool {
        matches!(self, Self::Control(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_flags() {
        let flags = PacketFlags {
            position: PacketPosition::First,
            order: true,
            encryption: EncryptionFlag::EvenKey,
            retransmitted: false,
        };

        let byte = flags.to_byte();
        let decoded = PacketFlags::from_byte(byte);

        assert_eq!(decoded.position, PacketPosition::First);
        assert!(decoded.order);
        assert_eq!(decoded.encryption, EncryptionFlag::EvenKey);
        assert!(!decoded.retransmitted);
    }

    #[test]
    fn test_data_packet_encode_decode() {
        let packet = DataPacket::new(12345, Bytes::from(vec![1, 2, 3, 4, 5]))
            .with_timestamp(1000)
            .with_dst_socket(42);

        let encoded = packet.encode();
        let decoded = DataPacket::decode(&encoded).expect("should succeed in test");

        assert_eq!(decoded.sequence_number, 12345);
        assert_eq!(decoded.timestamp, 1000);
        assert_eq!(decoded.dst_socket_id, 42);
        assert_eq!(decoded.payload, Bytes::from(vec![1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_control_packet_keepalive() {
        let packet = ControlPacket::keepalive(100);
        let encoded = packet.encode();
        let decoded = ControlPacket::decode(&encoded).expect("should succeed in test");

        assert_eq!(decoded.control_type, ControlType::Keepalive);
        assert_eq!(decoded.dst_socket_id, 100);
    }

    #[test]
    fn test_control_packet_ack() {
        let packet = ControlPacket::ack(50000, 200);
        let encoded = packet.encode();
        let decoded = ControlPacket::decode(&encoded).expect("should succeed in test");

        assert_eq!(decoded.control_type, ControlType::Ack);
        assert_eq!(decoded.type_info, 50000);
        assert_eq!(decoded.dst_socket_id, 200);
    }

    #[test]
    fn test_handshake_info() {
        let info = HandshakeInfo {
            version: 5,
            mtu: 1500,
            flow_window: 8192,
            handshake_type: HandshakeInfo::TYPE_INDUCTION,
            socket_id: 12345,
            syn_cookie: 0xDEADBEEF,
            ..Default::default()
        };

        let encoded = info.encode();
        let decoded = HandshakeInfo::decode(&encoded).expect("should succeed in test");

        assert_eq!(decoded.version, 5);
        assert_eq!(decoded.mtu, 1500);
        assert_eq!(decoded.flow_window, 8192);
        assert_eq!(decoded.handshake_type, HandshakeInfo::TYPE_INDUCTION);
        assert_eq!(decoded.socket_id, 12345);
        assert_eq!(decoded.syn_cookie, 0xDEADBEEF);
    }

    #[test]
    fn test_srt_packet_data() {
        let data = DataPacket::new(1, Bytes::from(vec![0xAB]));
        let packet = SrtPacket::Data(data);

        assert!(packet.is_data());
        assert!(!packet.is_control());

        let encoded = packet.encode();
        let decoded = SrtPacket::decode(&encoded).expect("should succeed in test");
        assert!(decoded.is_data());
    }

    #[test]
    fn test_srt_packet_control() {
        let ctrl = ControlPacket::shutdown(1);
        let packet = SrtPacket::Control(ctrl);

        assert!(!packet.is_data());
        assert!(packet.is_control());

        let encoded = packet.encode();
        let decoded = SrtPacket::decode(&encoded).expect("should succeed in test");
        assert!(decoded.is_control());
    }

    #[test]
    fn test_packet_position() {
        assert!(PacketPosition::First.is_first());
        assert!(!PacketPosition::First.is_last());
        assert!(PacketPosition::Single.is_first());
        assert!(PacketPosition::Single.is_last());
    }
}

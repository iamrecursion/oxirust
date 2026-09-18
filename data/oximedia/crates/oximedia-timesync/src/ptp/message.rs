//! PTP message types and serialization.

use super::{ClockIdentity, Domain, PortIdentity, PtpTimestamp};
use crate::error::{TimeSyncError, TimeSyncResult};
use bytes::{Buf, BufMut, BytesMut};

/// PTP message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Sync message
    Sync = 0x0,
    /// `Delay_Req` message
    DelayReq = 0x1,
    /// `Pdelay_Req` message
    PdelayReq = 0x2,
    /// `Pdelay_Resp` message
    PdelayResp = 0x3,
    /// `Follow_Up` message
    FollowUp = 0x8,
    /// `Delay_Resp` message
    DelayResp = 0x9,
    /// `Pdelay_Resp_Follow_Up` message
    PdelayRespFollowUp = 0xA,
    /// Announce message
    Announce = 0xB,
    /// Signaling message
    Signaling = 0xC,
    /// Management message
    Management = 0xD,
}

impl MessageType {
    /// Convert from u8
    pub fn from_u8(value: u8) -> TimeSyncResult<Self> {
        match value & 0x0F {
            0x0 => Ok(Self::Sync),
            0x1 => Ok(Self::DelayReq),
            0x2 => Ok(Self::PdelayReq),
            0x3 => Ok(Self::PdelayResp),
            0x8 => Ok(Self::FollowUp),
            0x9 => Ok(Self::DelayResp),
            0xA => Ok(Self::PdelayRespFollowUp),
            0xB => Ok(Self::Announce),
            0xC => Ok(Self::Signaling),
            0xD => Ok(Self::Management),
            _ => Err(TimeSyncError::InvalidPacket(format!(
                "Unknown message type: {value}"
            ))),
        }
    }
}

/// PTP message flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct Flags {
    /// Two-step flag
    pub two_step: bool,
    /// Unicast flag
    pub unicast: bool,
    /// PTP profile specific 1
    pub profile_specific_1: bool,
    /// PTP profile specific 2
    pub profile_specific_2: bool,
    /// Leap61
    pub leap61: bool,
    /// Leap59
    pub leap59: bool,
    /// Current UTC offset valid
    pub current_utc_offset_valid: bool,
    /// PTP timescale
    pub ptp_timescale: bool,
    /// Time traceable
    pub time_traceable: bool,
    /// Frequency traceable
    pub frequency_traceable: bool,
}

impl Flags {
    /// Encode flags to bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 2] {
        let mut bytes = [0u8; 2];
        if self.two_step {
            bytes[0] |= 0x02;
        }
        if self.unicast {
            bytes[0] |= 0x04;
        }
        if self.profile_specific_1 {
            bytes[0] |= 0x20;
        }
        if self.profile_specific_2 {
            bytes[0] |= 0x40;
        }
        if self.leap61 {
            bytes[1] |= 0x01;
        }
        if self.leap59 {
            bytes[1] |= 0x02;
        }
        if self.current_utc_offset_valid {
            bytes[1] |= 0x04;
        }
        if self.ptp_timescale {
            bytes[1] |= 0x08;
        }
        if self.time_traceable {
            bytes[1] |= 0x10;
        }
        if self.frequency_traceable {
            bytes[1] |= 0x20;
        }
        bytes
    }

    /// Decode flags from bytes
    #[must_use]
    pub fn from_bytes(bytes: [u8; 2]) -> Self {
        Self {
            two_step: (bytes[0] & 0x02) != 0,
            unicast: (bytes[0] & 0x04) != 0,
            profile_specific_1: (bytes[0] & 0x20) != 0,
            profile_specific_2: (bytes[0] & 0x40) != 0,
            leap61: (bytes[1] & 0x01) != 0,
            leap59: (bytes[1] & 0x02) != 0,
            current_utc_offset_valid: (bytes[1] & 0x04) != 0,
            ptp_timescale: (bytes[1] & 0x08) != 0,
            time_traceable: (bytes[1] & 0x10) != 0,
            frequency_traceable: (bytes[1] & 0x20) != 0,
        }
    }
}

/// PTP message header (common to all messages).
#[derive(Debug, Clone)]
pub struct Header {
    /// Message type
    pub message_type: MessageType,
    /// Version (should be 2 for `PTPv2`)
    pub version: u8,
    /// Message length
    pub message_length: u16,
    /// Domain number
    pub domain: Domain,
    /// Flags
    pub flags: Flags,
    /// Correction field (nanoseconds * 2^16)
    pub correction_field: i64,
    /// Source port identity
    pub source_port_identity: PortIdentity,
    /// Sequence ID
    pub sequence_id: u16,
    /// Control field (deprecated in `PTPv2`, but present for compatibility)
    pub control: u8,
    /// Log message interval
    pub log_message_interval: i8,
}

impl Header {
    /// Serialize header to bytes
    pub fn serialize(&self, buf: &mut BytesMut) -> TimeSyncResult<()> {
        // Byte 0: messageType and version
        buf.put_u8((self.message_type as u8) | ((self.version & 0x0F) << 4));

        // Byte 1: reserved
        buf.put_u8(0);

        // Bytes 2-3: messageLength
        buf.put_u16(self.message_length);

        // Byte 4: domainNumber
        buf.put_u8(self.domain.0);

        // Byte 5: reserved
        buf.put_u8(0);

        // Bytes 6-7: flagField
        let flags = self.flags.to_bytes();
        buf.put_slice(&flags);

        // Bytes 8-15: correctionField
        buf.put_i64(self.correction_field);

        // Bytes 16-19: reserved
        buf.put_u32(0);

        // Bytes 20-27: sourcePortIdentity (clockIdentity)
        buf.put_slice(&self.source_port_identity.clock_identity.0);

        // Bytes 28-29: sourcePortIdentity (portNumber)
        buf.put_u16(self.source_port_identity.port_number);

        // Bytes 30-31: sequenceId
        buf.put_u16(self.sequence_id);

        // Byte 32: control
        buf.put_u8(self.control);

        // Byte 33: logMessageInterval
        buf.put_i8(self.log_message_interval);

        Ok(())
    }

    /// Deserialize header from bytes
    pub fn deserialize(buf: &mut impl Buf) -> TimeSyncResult<Self> {
        if buf.remaining() < 34 {
            return Err(TimeSyncError::InvalidPacket(
                "Insufficient data for header".to_string(),
            ));
        }

        // Byte 0: messageType and version
        let byte0 = buf.get_u8();
        let message_type = MessageType::from_u8(byte0 & 0x0F)?;
        let version = (byte0 >> 4) & 0x0F;

        // Byte 1: reserved
        buf.get_u8();

        // Bytes 2-3: messageLength
        let message_length = buf.get_u16();

        // Byte 4: domainNumber
        let domain = Domain(buf.get_u8());

        // Byte 5: reserved
        buf.get_u8();

        // Bytes 6-7: flagField
        let flags = Flags::from_bytes([buf.get_u8(), buf.get_u8()]);

        // Bytes 8-15: correctionField
        let correction_field = buf.get_i64();

        // Bytes 16-19: reserved
        buf.get_u32();

        // Bytes 20-27: sourcePortIdentity (clockIdentity)
        let mut clock_id = [0u8; 8];
        buf.copy_to_slice(&mut clock_id);
        let clock_identity = ClockIdentity(clock_id);

        // Bytes 28-29: sourcePortIdentity (portNumber)
        let port_number = buf.get_u16();

        // Bytes 30-31: sequenceId
        let sequence_id = buf.get_u16();

        // Byte 32: control
        let control = buf.get_u8();

        // Byte 33: logMessageInterval
        let log_message_interval = buf.get_i8();

        Ok(Self {
            message_type,
            version,
            message_length,
            domain,
            flags,
            correction_field,
            source_port_identity: PortIdentity::new(clock_identity, port_number),
            sequence_id,
            control,
            log_message_interval,
        })
    }
}

// ---------------------------------------------------------------------------
// Zero-copy PTP message parsing
// ---------------------------------------------------------------------------

/// Minimum wire size of a PTP Sync message (header 34 bytes + timestamp 10 bytes).
const SYNC_MESSAGE_MIN_LEN: usize = 44;

/// A zero-copy view of a PTP Sync message that holds references into the
/// original byte slice, avoiding any allocation.
///
/// Field accessors decode each field on demand from the backing slice.
#[derive(Debug, Clone, Copy)]
pub struct SyncMessageView<'a> {
    /// The raw byte slice backing this view.  Must be ≥ 44 bytes.
    buf: &'a [u8],
}

impl<'a> SyncMessageView<'a> {
    /// Returns the message type nibble (low 4 bits of byte 0).
    #[must_use]
    pub fn message_type_raw(&self) -> u8 {
        self.buf[0] & 0x0F
    }

    /// Returns the PTP version (high 4 bits of byte 0, shifted down).
    #[must_use]
    pub fn version(&self) -> u8 {
        (self.buf[0] >> 4) & 0x0F
    }

    /// Returns the message length field (big-endian u16 at bytes 2–3).
    #[must_use]
    pub fn message_length(&self) -> u16 {
        u16::from_be_bytes([self.buf[2], self.buf[3]])
    }

    /// Returns the domain number (byte 4).
    #[must_use]
    pub fn domain(&self) -> u8 {
        self.buf[4]
    }

    /// Returns the raw 2-byte flags field (bytes 6–7).
    #[must_use]
    pub fn flags_raw(&self) -> [u8; 2] {
        [self.buf[6], self.buf[7]]
    }

    /// Returns the correction field as a signed 64-bit integer (bytes 8–15).
    #[must_use]
    pub fn correction_field(&self) -> i64 {
        i64::from_be_bytes(self.buf[8..16].try_into().unwrap_or([0u8; 8]))
    }

    /// Returns the clock identity bytes (bytes 20–27) as a reference into the
    /// backing buffer.
    #[must_use]
    pub fn clock_identity_bytes(&self) -> &'a [u8] {
        &self.buf[20..28]
    }

    /// Returns the port number (big-endian u16 at bytes 28–29).
    #[must_use]
    pub fn port_number(&self) -> u16 {
        u16::from_be_bytes([self.buf[28], self.buf[29]])
    }

    /// Returns the sequence ID (big-endian u16 at bytes 30–31).
    #[must_use]
    pub fn sequence_id(&self) -> u16 {
        u16::from_be_bytes([self.buf[30], self.buf[31]])
    }

    /// Returns the log message interval (signed byte 33).
    #[must_use]
    pub fn log_message_interval(&self) -> i8 {
        self.buf[33] as i8
    }

    /// Returns the origin timestamp seconds (6-byte big-endian at bytes 34–39).
    #[must_use]
    pub fn origin_seconds(&self) -> u64 {
        let hi = u16::from_be_bytes([self.buf[34], self.buf[35]]) as u64;
        let lo =
            u32::from_be_bytes([self.buf[36], self.buf[37], self.buf[38], self.buf[39]]) as u64;
        (hi << 32) | lo
    }

    /// Returns the origin timestamp nanoseconds (big-endian u32 at bytes 40–43).
    #[must_use]
    pub fn origin_nanoseconds(&self) -> u32 {
        u32::from_be_bytes([self.buf[40], self.buf[41], self.buf[42], self.buf[43]])
    }

    /// Returns the raw backing slice.
    #[must_use]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.buf
    }
}

/// PTP error type for zero-copy parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtpError {
    /// The buffer is too short to hold a valid message.
    BufferTooShort {
        /// Minimum number of bytes required.
        needed: usize,
        /// Actual number of bytes supplied.
        got: usize,
    },
    /// The message type field does not indicate a Sync message.
    NotASyncMessage {
        /// The actual message-type nibble found in the buffer.
        got: u8,
    },
}

impl std::fmt::Display for PtpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferTooShort { needed, got } => {
                write!(f, "buffer too short: need {needed} bytes, got {got}")
            }
            Self::NotASyncMessage { got } => {
                write!(f, "expected Sync (0x0), got message type 0x{got:x}")
            }
        }
    }
}

impl std::error::Error for PtpError {}

/// Parses a PTP Sync message from `buf` without any allocation, returning a
/// [`SyncMessageView`] that borrows directly from the input slice.
///
/// # Errors
///
/// Returns [`PtpError::BufferTooShort`] when `buf` is shorter than 44 bytes,
/// or [`PtpError::NotASyncMessage`] when the message-type nibble is not 0x0.
pub fn parse_sync_message_zerocopy(buf: &[u8]) -> Result<SyncMessageView<'_>, PtpError> {
    if buf.len() < SYNC_MESSAGE_MIN_LEN {
        return Err(PtpError::BufferTooShort {
            needed: SYNC_MESSAGE_MIN_LEN,
            got: buf.len(),
        });
    }
    let msg_type = buf[0] & 0x0F;
    if msg_type != 0x0 {
        return Err(PtpError::NotASyncMessage { got: msg_type });
    }
    Ok(SyncMessageView { buf })
}

/// Parses a batch of PTP Sync messages from a slice of byte slices.
///
/// Allocates the output `Vec` once with `bufs.len()` capacity and fills it
/// with the parse result for each buffer.  Each element is either a
/// [`SyncMessageView`] borrowing from the corresponding input slice, or a
/// [`PtpError`] describing why that buffer could not be parsed.
pub fn parse_sync_batch<'a>(bufs: &'a [&'a [u8]]) -> Vec<Result<SyncMessageView<'a>, PtpError>> {
    let mut out = Vec::with_capacity(bufs.len());
    for buf in bufs {
        out.push(parse_sync_message_zerocopy(buf));
    }
    out
}

/// PTP Sync message.
#[derive(Debug, Clone)]
pub struct SyncMessage {
    /// Header
    pub header: Header,
    /// Origin timestamp (when sync was sent)
    pub origin_timestamp: PtpTimestamp,
}

impl SyncMessage {
    /// Serialize to bytes
    pub fn serialize(&self) -> TimeSyncResult<BytesMut> {
        let mut buf = BytesMut::with_capacity(44);
        self.header.serialize(&mut buf)?;

        // Origin timestamp (10 bytes)
        buf.put_u16((self.origin_timestamp.seconds >> 32) as u16);
        buf.put_u32((self.origin_timestamp.seconds & 0xFFFF_FFFF) as u32);
        buf.put_u32(self.origin_timestamp.nanoseconds);

        Ok(buf)
    }

    /// Deserialize from bytes
    pub fn deserialize(mut buf: impl Buf) -> TimeSyncResult<Self> {
        let header = Header::deserialize(&mut buf)?;

        if buf.remaining() < 10 {
            return Err(TimeSyncError::InvalidPacket(
                "Insufficient data for Sync message".to_string(),
            ));
        }

        let seconds_hi = u64::from(buf.get_u16());
        let seconds_lo = u64::from(buf.get_u32());
        let seconds = (seconds_hi << 32) | seconds_lo;
        let nanoseconds = buf.get_u32();

        let origin_timestamp = PtpTimestamp::new(seconds, nanoseconds)?;

        Ok(Self {
            header,
            origin_timestamp,
        })
    }
}

/// PTP `Follow_Up` message.
#[derive(Debug, Clone)]
pub struct FollowUpMessage {
    /// Header
    pub header: Header,
    /// Precise origin timestamp
    pub precise_origin_timestamp: PtpTimestamp,
}

impl FollowUpMessage {
    /// Serialize to bytes
    pub fn serialize(&self) -> TimeSyncResult<BytesMut> {
        let mut buf = BytesMut::with_capacity(44);
        self.header.serialize(&mut buf)?;

        // Precise origin timestamp (10 bytes)
        buf.put_u16((self.precise_origin_timestamp.seconds >> 32) as u16);
        buf.put_u32((self.precise_origin_timestamp.seconds & 0xFFFF_FFFF) as u32);
        buf.put_u32(self.precise_origin_timestamp.nanoseconds);

        Ok(buf)
    }

    /// Deserialize from bytes
    pub fn deserialize(mut buf: impl Buf) -> TimeSyncResult<Self> {
        let header = Header::deserialize(&mut buf)?;

        if buf.remaining() < 10 {
            return Err(TimeSyncError::InvalidPacket(
                "Insufficient data for Follow_Up message".to_string(),
            ));
        }

        let seconds_hi = u64::from(buf.get_u16());
        let seconds_lo = u64::from(buf.get_u32());
        let seconds = (seconds_hi << 32) | seconds_lo;
        let nanoseconds = buf.get_u32();

        let precise_origin_timestamp = PtpTimestamp::new(seconds, nanoseconds)?;

        Ok(Self {
            header,
            precise_origin_timestamp,
        })
    }
}

/// PTP `Delay_Req` message.
#[derive(Debug, Clone)]
pub struct DelayReqMessage {
    /// Header
    pub header: Header,
    /// Origin timestamp
    pub origin_timestamp: PtpTimestamp,
}

impl DelayReqMessage {
    /// Serialize to bytes
    pub fn serialize(&self) -> TimeSyncResult<BytesMut> {
        let mut buf = BytesMut::with_capacity(44);
        self.header.serialize(&mut buf)?;

        // Origin timestamp (10 bytes)
        buf.put_u16((self.origin_timestamp.seconds >> 32) as u16);
        buf.put_u32((self.origin_timestamp.seconds & 0xFFFF_FFFF) as u32);
        buf.put_u32(self.origin_timestamp.nanoseconds);

        Ok(buf)
    }

    /// Deserialize from bytes
    pub fn deserialize(mut buf: impl Buf) -> TimeSyncResult<Self> {
        let header = Header::deserialize(&mut buf)?;

        if buf.remaining() < 10 {
            return Err(TimeSyncError::InvalidPacket(
                "Insufficient data for Delay_Req message".to_string(),
            ));
        }

        let seconds_hi = u64::from(buf.get_u16());
        let seconds_lo = u64::from(buf.get_u32());
        let seconds = (seconds_hi << 32) | seconds_lo;
        let nanoseconds = buf.get_u32();

        let origin_timestamp = PtpTimestamp::new(seconds, nanoseconds)?;

        Ok(Self {
            header,
            origin_timestamp,
        })
    }
}

/// PTP `Delay_Resp` message.
#[derive(Debug, Clone)]
pub struct DelayRespMessage {
    /// Header
    pub header: Header,
    /// Receive timestamp
    pub receive_timestamp: PtpTimestamp,
    /// Requesting port identity
    pub requesting_port_identity: PortIdentity,
}

impl DelayRespMessage {
    /// Serialize to bytes
    pub fn serialize(&self) -> TimeSyncResult<BytesMut> {
        let mut buf = BytesMut::with_capacity(54);
        self.header.serialize(&mut buf)?;

        // Receive timestamp (10 bytes)
        buf.put_u16((self.receive_timestamp.seconds >> 32) as u16);
        buf.put_u32((self.receive_timestamp.seconds & 0xFFFF_FFFF) as u32);
        buf.put_u32(self.receive_timestamp.nanoseconds);

        // Requesting port identity (10 bytes)
        buf.put_slice(&self.requesting_port_identity.clock_identity.0);
        buf.put_u16(self.requesting_port_identity.port_number);

        Ok(buf)
    }

    /// Deserialize from bytes
    pub fn deserialize(mut buf: impl Buf) -> TimeSyncResult<Self> {
        let header = Header::deserialize(&mut buf)?;

        if buf.remaining() < 20 {
            return Err(TimeSyncError::InvalidPacket(
                "Insufficient data for Delay_Resp message".to_string(),
            ));
        }

        let seconds_hi = u64::from(buf.get_u16());
        let seconds_lo = u64::from(buf.get_u32());
        let seconds = (seconds_hi << 32) | seconds_lo;
        let nanoseconds = buf.get_u32();

        let receive_timestamp = PtpTimestamp::new(seconds, nanoseconds)?;

        let mut clock_id = [0u8; 8];
        buf.copy_to_slice(&mut clock_id);
        let clock_identity = ClockIdentity(clock_id);
        let port_number = buf.get_u16();

        Ok(Self {
            header,
            receive_timestamp,
            requesting_port_identity: PortIdentity::new(clock_identity, port_number),
        })
    }
}

/// PTP Announce message.
#[derive(Debug, Clone)]
pub struct AnnounceMessage {
    /// Header
    pub header: Header,
    /// Origin timestamp
    pub origin_timestamp: PtpTimestamp,
    /// Current UTC offset
    pub current_utc_offset: i16,
    /// Grand master priority 1
    pub grandmaster_priority1: u8,
    /// Grand master clock quality
    pub grandmaster_clock_quality: ClockQuality,
    /// Grand master priority 2
    pub grandmaster_priority2: u8,
    /// Grand master identity
    pub grandmaster_identity: ClockIdentity,
    /// Steps removed
    pub steps_removed: u16,
    /// Time source
    pub time_source: u8,
}

/// Clock quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockQuality {
    /// Clock class
    pub clock_class: u8,
    /// Clock accuracy
    pub clock_accuracy: u8,
    /// Offset scaled log variance
    pub offset_scaled_log_variance: u16,
}

impl AnnounceMessage {
    /// Serialize to bytes
    pub fn serialize(&self) -> TimeSyncResult<BytesMut> {
        let mut buf = BytesMut::with_capacity(64);
        self.header.serialize(&mut buf)?;

        // Origin timestamp (10 bytes)
        buf.put_u16((self.origin_timestamp.seconds >> 32) as u16);
        buf.put_u32((self.origin_timestamp.seconds & 0xFFFF_FFFF) as u32);
        buf.put_u32(self.origin_timestamp.nanoseconds);

        // Current UTC offset
        buf.put_i16(self.current_utc_offset);

        // Reserved
        buf.put_u8(0);

        // Grandmaster priority 1
        buf.put_u8(self.grandmaster_priority1);

        // Grandmaster clock quality
        buf.put_u8(self.grandmaster_clock_quality.clock_class);
        buf.put_u8(self.grandmaster_clock_quality.clock_accuracy);
        buf.put_u16(self.grandmaster_clock_quality.offset_scaled_log_variance);

        // Grandmaster priority 2
        buf.put_u8(self.grandmaster_priority2);

        // Grandmaster identity
        buf.put_slice(&self.grandmaster_identity.0);

        // Steps removed
        buf.put_u16(self.steps_removed);

        // Time source
        buf.put_u8(self.time_source);

        Ok(buf)
    }

    /// Deserialize from bytes
    pub fn deserialize(mut buf: impl Buf) -> TimeSyncResult<Self> {
        let header = Header::deserialize(&mut buf)?;

        if buf.remaining() < 30 {
            return Err(TimeSyncError::InvalidPacket(
                "Insufficient data for Announce message".to_string(),
            ));
        }

        let seconds_hi = u64::from(buf.get_u16());
        let seconds_lo = u64::from(buf.get_u32());
        let seconds = (seconds_hi << 32) | seconds_lo;
        let nanoseconds = buf.get_u32();

        let origin_timestamp = PtpTimestamp::new(seconds, nanoseconds)?;

        let current_utc_offset = buf.get_i16();
        buf.get_u8(); // reserved

        let grandmaster_priority1 = buf.get_u8();

        let clock_class = buf.get_u8();
        let clock_accuracy = buf.get_u8();
        let offset_scaled_log_variance = buf.get_u16();

        let grandmaster_priority2 = buf.get_u8();

        let mut gm_id = [0u8; 8];
        buf.copy_to_slice(&mut gm_id);
        let grandmaster_identity = ClockIdentity(gm_id);

        let steps_removed = buf.get_u16();
        let time_source = buf.get_u8();

        Ok(Self {
            header,
            origin_timestamp,
            current_utc_offset,
            grandmaster_priority1,
            grandmaster_clock_quality: ClockQuality {
                clock_class,
                clock_accuracy,
                offset_scaled_log_variance,
            },
            grandmaster_priority2,
            grandmaster_identity,
            steps_removed,
            time_source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_type_conversion() {
        assert_eq!(
            MessageType::from_u8(0x0).expect("should succeed in test"),
            MessageType::Sync
        );
        assert_eq!(
            MessageType::from_u8(0x1).expect("should succeed in test"),
            MessageType::DelayReq
        );
        assert_eq!(
            MessageType::from_u8(0x8).expect("should succeed in test"),
            MessageType::FollowUp
        );
        assert!(MessageType::from_u8(0xFF).is_err());
    }

    #[test]
    fn test_flags_encoding() {
        let mut flags = Flags::default();
        flags.two_step = true;
        flags.ptp_timescale = true;

        let bytes = flags.to_bytes();
        assert_eq!(bytes[0] & 0x02, 0x02);
        assert_eq!(bytes[1] & 0x08, 0x08);

        let decoded = Flags::from_bytes(bytes);
        assert!(decoded.two_step);
        assert!(decoded.ptp_timescale);
    }

    #[test]
    fn test_sync_message_serialization() {
        let clock_id = ClockIdentity([1, 2, 3, 4, 5, 6, 7, 8]);
        let port_id = PortIdentity::new(clock_id, 1);

        let header = Header {
            message_type: MessageType::Sync,
            version: 2,
            message_length: 44,
            domain: Domain::DEFAULT,
            flags: Flags::default(),
            correction_field: 0,
            source_port_identity: port_id,
            sequence_id: 1,
            control: 0,
            log_message_interval: 0,
        };

        let origin_timestamp =
            PtpTimestamp::new(1000, 500_000_000).expect("should succeed in test");

        let sync = SyncMessage {
            header,
            origin_timestamp,
        };

        let serialized = sync.serialize().expect("should succeed in test");
        assert_eq!(serialized.len(), 44);

        let deserialized =
            SyncMessage::deserialize(&serialized[..]).expect("should succeed in test");
        assert_eq!(deserialized.origin_timestamp.seconds, 1000);
        assert_eq!(deserialized.origin_timestamp.nanoseconds, 500_000_000);
    }

    // -----------------------------------------------------------------------
    // Zero-copy Sync message parsing tests
    // -----------------------------------------------------------------------

    /// Builds a minimal 44-byte Sync wire frame with the given seconds and
    /// nanoseconds in the origin timestamp.
    fn make_sync_buf(seconds: u64, nanoseconds: u32) -> Vec<u8> {
        let clock_id = ClockIdentity([0xAA; 8]);
        let port_id = PortIdentity::new(clock_id, 1);
        let header = Header {
            message_type: MessageType::Sync,
            version: 2,
            message_length: 44,
            domain: Domain::DEFAULT,
            flags: Flags::default(),
            correction_field: 0,
            source_port_identity: port_id,
            sequence_id: 7,
            control: 0,
            log_message_interval: -7,
        };
        let sync = SyncMessage {
            header,
            origin_timestamp: PtpTimestamp::new(seconds, nanoseconds).expect("valid timestamp"),
        };
        sync.serialize().expect("serialise should succeed").to_vec()
    }

    #[test]
    fn test_zerocopy_parse_valid_sync() {
        let buf = make_sync_buf(12345, 678_000_000);
        let view = parse_sync_message_zerocopy(&buf).expect("parse should succeed");
        assert_eq!(view.origin_seconds(), 12345);
        assert_eq!(view.origin_nanoseconds(), 678_000_000);
        assert_eq!(view.sequence_id(), 7);
        assert_eq!(view.domain(), 0);
    }

    #[test]
    fn test_zerocopy_parse_buffer_too_short() {
        // 43 bytes is one byte short of the minimum.
        let buf = vec![0u8; 43];
        let result = parse_sync_message_zerocopy(&buf);
        assert!(matches!(
            result,
            Err(PtpError::BufferTooShort {
                needed: 44,
                got: 43
            })
        ));
    }

    #[test]
    fn test_zerocopy_parse_wrong_message_type() {
        let mut buf = make_sync_buf(0, 0);
        // Overwrite byte 0 to set message type = Announce (0x0B).
        buf[0] = (buf[0] & 0xF0) | 0x0B;
        let result = parse_sync_message_zerocopy(&buf);
        assert!(matches!(
            result,
            Err(PtpError::NotASyncMessage { got: 0x0B })
        ));
    }

    #[test]
    fn test_zerocopy_clock_identity_bytes_reference() {
        // Verify that clock_identity_bytes() points into the original buffer.
        let buf = make_sync_buf(0, 0);
        let view = parse_sync_message_zerocopy(&buf).expect("valid sync");
        let id_bytes = view.clock_identity_bytes();
        // The make_sync_buf sets clock identity to [0xAA; 8].
        assert_eq!(id_bytes, &[0xAA_u8; 8]);
        // Verify lifetime: id_bytes borrows from buf.
        assert_eq!(id_bytes.as_ptr(), buf[20..28].as_ptr());
    }

    #[test]
    fn test_parse_sync_batch() {
        let buf1 = make_sync_buf(100, 0);
        let buf2 = make_sync_buf(200, 500_000_000);
        let too_short = vec![0u8; 10];
        let bufs: &[&[u8]] = &[buf1.as_slice(), buf2.as_slice(), too_short.as_slice()];
        let results = parse_sync_batch(bufs);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert!(results[2].is_err());
        let v0 = results[0].as_ref().expect("first should succeed");
        assert_eq!(v0.origin_seconds(), 100);
        let v1 = results[1].as_ref().expect("second should succeed");
        assert_eq!(v1.origin_seconds(), 200);
        assert_eq!(v1.origin_nanoseconds(), 500_000_000);
    }

    #[test]
    fn test_zerocopy_roundtrip_matches_allocating_parse() {
        let buf = make_sync_buf(9_999_999, 123_456_789);
        let view = parse_sync_message_zerocopy(&buf).expect("zero-copy parse");
        let allocating = SyncMessage::deserialize(buf.as_slice()).expect("allocating parse");
        // Both parsers must agree on the origin timestamp.
        assert_eq!(view.origin_seconds(), allocating.origin_timestamp.seconds);
        assert_eq!(
            view.origin_nanoseconds(),
            allocating.origin_timestamp.nanoseconds
        );
        assert_eq!(view.sequence_id(), allocating.header.sequence_id);
    }
}

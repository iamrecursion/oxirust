//! Metadata support (timecode, ancillary data, closed captions).

use crate::error::{VideoIpError, VideoIpResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

/// Metadata packet type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MetadataType {
    /// Timecode (LTC/VITC).
    Timecode = 0,
    /// Closed captions (CEA-608/708).
    ClosedCaptions = 1,
    /// Active Format Description (AFD).
    Afd = 2,
    /// Custom metadata.
    Custom = 255,
}

impl MetadataType {
    /// Converts from a byte value.
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Timecode,
            1 => Self::ClosedCaptions,
            2 => Self::Afd,
            _ => Self::Custom,
        }
    }

    /// Converts to a byte value.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Timecode representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timecode {
    /// Hours (0-23).
    pub hours: u8,
    /// Minutes (0-59).
    pub minutes: u8,
    /// Seconds (0-59).
    pub seconds: u8,
    /// Frames (0-max frame rate).
    pub frames: u8,
    /// Drop frame flag.
    pub drop_frame: bool,
}

impl Timecode {
    /// Creates a new timecode.
    #[must_use]
    pub const fn new(hours: u8, minutes: u8, seconds: u8, frames: u8, drop_frame: bool) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
            drop_frame,
        }
    }

    /// Parses timecode from SMPTE 12M format.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid.
    pub fn from_bytes(data: &[u8]) -> VideoIpResult<Self> {
        if data.len() < 4 {
            return Err(VideoIpError::Metadata("invalid timecode data".to_string()));
        }

        let frames = Self::decode_bcd(data[0] & 0x3F);
        let drop_frame = (data[0] & 0x40) != 0;
        let seconds = Self::decode_bcd(data[1] & 0x7F);
        let minutes = Self::decode_bcd(data[2] & 0x7F);
        let hours = Self::decode_bcd(data[3] & 0x3F);

        Ok(Self::new(hours, minutes, seconds, frames, drop_frame))
    }

    /// Encodes timecode to SMPTE 12M format.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 4] {
        let mut data = [0u8; 4];
        data[0] = Self::encode_bcd(self.frames) | if self.drop_frame { 0x40 } else { 0 };
        data[1] = Self::encode_bcd(self.seconds);
        data[2] = Self::encode_bcd(self.minutes);
        data[3] = Self::encode_bcd(self.hours);
        data
    }

    /// Decodes a BCD (Binary-Coded Decimal) value.
    fn decode_bcd(value: u8) -> u8 {
        let high = (value >> 4) * 10;
        let low = value & 0x0F;
        high + low
    }

    /// Encodes a value to BCD.
    fn encode_bcd(value: u8) -> u8 {
        let high = (value / 10) << 4;
        let low = value % 10;
        high | low
    }

    /// Converts to total frames.
    #[must_use]
    pub fn to_frames(&self, fps: u8) -> u32 {
        let total_seconds =
            u32::from(self.hours) * 3600 + u32::from(self.minutes) * 60 + u32::from(self.seconds);
        total_seconds * u32::from(fps) + u32::from(self.frames)
    }

    /// Creates timecode from total frames.
    #[must_use]
    pub fn from_frames(total_frames: u32, fps: u8, drop_frame: bool) -> Self {
        let total_seconds = total_frames / u32::from(fps);
        let frames = (total_frames % u32::from(fps)) as u8;
        let hours = (total_seconds / 3600) as u8;
        let minutes = ((total_seconds % 3600) / 60) as u8;
        let seconds = (total_seconds % 60) as u8;

        Self::new(hours, minutes, seconds, frames, drop_frame)
    }
}

impl std::fmt::Display for Timecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let separator = if self.drop_frame { ';' } else { ':' };
        write!(
            f,
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, separator, self.frames
        )
    }
}

/// Closed caption data (CEA-608/708).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedCaptionData {
    /// Caption data bytes.
    pub data: Vec<u8>,
    /// Caption type (CEA-608 or CEA-708).
    pub caption_type: CaptionType,
}

/// Caption type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptionType {
    /// CEA-608 (Line 21).
    Cea608,
    /// CEA-708 (DTVCC).
    Cea708,
}

/// Active Format Description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Afd {
    /// Undefined.
    Undefined = 0,
    /// 4:3 full frame.
    Box4x3 = 8,
    /// 16:9 full frame.
    Box16x9 = 10,
    /// 14:9 center.
    Box14x9Center = 11,
    /// Letterbox 16:9.
    Letterbox16x9 = 13,
    /// Full frame.
    FullFrame = 15,
}

/// Metadata packet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataPacket {
    /// Metadata type.
    pub metadata_type: MetadataType,
    /// Metadata payload.
    pub payload: Vec<u8>,
}

impl MetadataPacket {
    /// Creates a new metadata packet.
    #[must_use]
    pub fn new(metadata_type: MetadataType, payload: Vec<u8>) -> Self {
        Self {
            metadata_type,
            payload,
        }
    }

    /// Creates a timecode metadata packet.
    #[must_use]
    pub fn timecode(tc: Timecode) -> Self {
        let data = tc.to_bytes();
        Self::new(MetadataType::Timecode, data.to_vec())
    }

    /// Creates a closed caption metadata packet.
    #[must_use]
    pub fn closed_caption(data: ClosedCaptionData) -> Self {
        Self::new(MetadataType::ClosedCaptions, data.data)
    }

    /// Encodes the metadata packet.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();
        buf.put_u8(self.metadata_type.to_u8());
        buf.put_u16(self.payload.len() as u16);
        buf.put(self.payload.as_slice());
        buf.freeze()
    }

    /// Decodes a metadata packet.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid.
    pub fn decode(mut data: &[u8]) -> VideoIpResult<Self> {
        if data.len() < 3 {
            return Err(VideoIpError::Metadata(
                "insufficient metadata data".to_string(),
            ));
        }

        let metadata_type = MetadataType::from_u8(data.get_u8());
        let payload_len = data.get_u16() as usize;

        if data.len() < payload_len {
            return Err(VideoIpError::Metadata(format!(
                "insufficient payload data: expected {payload_len}, got {}",
                data.len()
            )));
        }

        let payload = data.copy_to_bytes(payload_len);

        Ok(Self::new(metadata_type, payload.to_vec()))
    }

    /// Parses the payload as timecode.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload is not valid timecode data.
    pub fn as_timecode(&self) -> VideoIpResult<Timecode> {
        if self.metadata_type != MetadataType::Timecode {
            return Err(VideoIpError::Metadata("not a timecode packet".to_string()));
        }
        Timecode::from_bytes(&self.payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_creation() {
        let tc = Timecode::new(1, 23, 45, 12, false);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 23);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
        assert!(!tc.drop_frame);
    }

    #[test]
    fn test_timecode_display() {
        let tc = Timecode::new(1, 23, 45, 12, false);
        assert_eq!(tc.to_string(), "01:23:45:12");

        let tc_drop = Timecode::new(1, 23, 45, 12, true);
        assert_eq!(tc_drop.to_string(), "01:23:45;12");
    }

    #[test]
    fn test_timecode_encode_decode() {
        let tc = Timecode::new(12, 34, 56, 23, false);
        let bytes = tc.to_bytes();
        let decoded = Timecode::from_bytes(&bytes).expect("should succeed in test");
        assert_eq!(tc, decoded);
    }

    #[test]
    fn test_timecode_to_frames() {
        let tc = Timecode::new(0, 0, 1, 0, false);
        assert_eq!(tc.to_frames(30), 30);

        let tc2 = Timecode::new(0, 1, 0, 0, false);
        assert_eq!(tc2.to_frames(30), 1800);
    }

    #[test]
    fn test_timecode_from_frames() {
        let tc = Timecode::from_frames(90, 30, false);
        assert_eq!(tc.seconds, 3);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_metadata_type_conversion() {
        assert_eq!(MetadataType::Timecode.to_u8(), 0);
        assert_eq!(MetadataType::from_u8(0), MetadataType::Timecode);
        assert_eq!(MetadataType::from_u8(255), MetadataType::Custom);
    }

    #[test]
    fn test_metadata_packet_timecode() {
        let tc = Timecode::new(1, 2, 3, 4, false);
        let packet = MetadataPacket::timecode(tc);
        assert_eq!(packet.metadata_type, MetadataType::Timecode);

        let decoded_tc = packet.as_timecode().expect("should succeed in test");
        assert_eq!(decoded_tc, tc);
    }

    #[test]
    fn test_metadata_packet_encode_decode() {
        let tc = Timecode::new(5, 10, 15, 20, false);
        let packet = MetadataPacket::timecode(tc);

        let encoded = packet.encode();
        let decoded = MetadataPacket::decode(&encoded).expect("should succeed in test");

        assert_eq!(decoded.metadata_type, MetadataType::Timecode);
        assert_eq!(decoded.payload, packet.payload);
    }

    #[test]
    fn test_bcd_encoding() {
        assert_eq!(Timecode::encode_bcd(0), 0x00);
        assert_eq!(Timecode::encode_bcd(9), 0x09);
        assert_eq!(Timecode::encode_bcd(10), 0x10);
        assert_eq!(Timecode::encode_bcd(59), 0x59);
    }

    #[test]
    fn test_bcd_decoding() {
        assert_eq!(Timecode::decode_bcd(0x00), 0);
        assert_eq!(Timecode::decode_bcd(0x09), 9);
        assert_eq!(Timecode::decode_bcd(0x10), 10);
        assert_eq!(Timecode::decode_bcd(0x59), 59);
    }
}

//! SMPTE ST 2110-40 ancillary data over RTP.
//!
//! This module implements SMPTE ST 2110-40 which defines the transport of
//! ancillary data (ANC) over RTP for professional broadcast applications.
//! It supports VANC (Vertical Ancillary Data), HANC (Horizontal Ancillary Data),
//! closed captions (CEA-608/708), timecode (SMPTE 12M), and other metadata.

use crate::error::{NetError, NetResult};
use crate::smpte2110::rtp::{RtpHeader, RtpPacket};
use bytes::{Buf, BufMut, BytesMut};
use std::collections::HashMap;

/// RTP payload type for ST 2110-40 ancillary data (dynamic range).
pub const RTP_PAYLOAD_TYPE_ANC: u8 = 100;

/// Ancillary data type (DID - Data Identifier).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AncillaryDataType {
    /// Undefined/unknown.
    Undefined = 0x00,
    /// CEA-608/708 closed captions.
    ClosedCaptions = 0x61,
    /// SMPTE 12M timecode.
    Timecode = 0x60,
    /// Active Format Description (AFD).
    AFD = 0x41,
    /// Bar data.
    BarData = 0x42,
    /// SCTE-104 automation messages.
    SCTE104 = 0x43,
    /// OP-47 SDP data.
    OP47 = 0x45,
    /// Multi-packet ANC (continuation).
    MultiPacket = 0x62,
}

impl AncillaryDataType {
    /// Creates from DID value.
    #[must_use]
    pub fn from_did(did: u8) -> Self {
        match did {
            0x61 => Self::ClosedCaptions,
            0x60 => Self::Timecode,
            0x41 => Self::AFD,
            0x42 => Self::BarData,
            0x43 => Self::SCTE104,
            0x45 => Self::OP47,
            0x62 => Self::MultiPacket,
            _ => Self::Undefined,
        }
    }

    /// Gets the DID value.
    #[must_use]
    pub const fn as_did(self) -> u8 {
        self as u8
    }
}

/// Ancillary data location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AncillaryLocation {
    /// Vertical Ancillary Data (VANC).
    VANC {
        /// Line number in the frame.
        line_number: u16,
    },
    /// Horizontal Ancillary Data (HANC).
    HANC {
        /// Line number in the frame.
        line_number: u16,
        /// Horizontal offset.
        offset: u16,
    },
}

impl AncillaryLocation {
    /// Checks if this is VANC.
    #[must_use]
    pub const fn is_vanc(&self) -> bool {
        matches!(self, Self::VANC { .. })
    }

    /// Checks if this is HANC.
    #[must_use]
    pub const fn is_hanc(&self) -> bool {
        matches!(self, Self::HANC { .. })
    }

    /// Gets the line number.
    #[must_use]
    pub const fn line_number(&self) -> u16 {
        match self {
            Self::VANC { line_number } | Self::HANC { line_number, .. } => *line_number,
        }
    }
}

/// Ancillary data packet (SMPTE 291M format).
#[derive(Debug, Clone)]
pub struct AncillaryData {
    /// Data Identifier (DID).
    pub did: u8,
    /// Secondary Data Identifier (SDID).
    pub sdid: u8,
    /// Data Count (DC) - number of user data words.
    pub data_count: u8,
    /// User Data Words (UDW).
    pub user_data: Vec<u8>,
    /// Checksum.
    pub checksum: u16,
    /// Location in video frame.
    pub location: AncillaryLocation,
}

impl AncillaryData {
    /// Creates a new ancillary data packet.
    #[must_use]
    pub fn new(did: u8, sdid: u8, user_data: Vec<u8>, location: AncillaryLocation) -> Self {
        let data_count = user_data.len() as u8;
        let checksum = Self::calculate_checksum(did, sdid, data_count, &user_data);

        Self {
            did,
            sdid,
            data_count,
            user_data,
            checksum,
            location,
        }
    }

    /// Calculates the checksum (sum of DID + SDID + DC + UDW).
    fn calculate_checksum(did: u8, sdid: u8, data_count: u8, user_data: &[u8]) -> u16 {
        let mut sum = u16::from(did) + u16::from(sdid) + u16::from(data_count);

        for &byte in user_data {
            sum += u16::from(byte);
        }

        sum & 0x1FF // 9-bit checksum
    }

    /// Validates the checksum.
    #[must_use]
    pub fn validate_checksum(&self) -> bool {
        let calculated =
            Self::calculate_checksum(self.did, self.sdid, self.data_count, &self.user_data);
        calculated == self.checksum
    }

    /// Serializes to SMPTE 291M format.
    pub fn serialize(&self, buf: &mut BytesMut) {
        // ADF (Ancillary Data Flag) - 3 words
        buf.put_u16(0x0000);
        buf.put_u16(0x03FF);
        buf.put_u16(0x03FF);

        // DID, SDID, DC
        buf.put_u16(u16::from(self.did));
        buf.put_u16(u16::from(self.sdid));
        buf.put_u16(u16::from(self.data_count));

        // User Data Words
        for &byte in &self.user_data {
            buf.put_u16(u16::from(byte));
        }

        // Checksum
        buf.put_u16(self.checksum);
    }

    /// Parses from SMPTE 291M format.
    pub fn parse(data: &[u8], location: AncillaryLocation) -> NetResult<Self> {
        if data.len() < 12 {
            return Err(NetError::parse(0, "ANC data too short"));
        }

        let mut cursor = &data[..];

        // Check ADF (Ancillary Data Flag)
        let adf0 = cursor.get_u16();
        let adf1 = cursor.get_u16();
        let adf2 = cursor.get_u16();

        if adf0 != 0x0000 || adf1 != 0x03FF || adf2 != 0x03FF {
            return Err(NetError::protocol("Invalid ANC ADF"));
        }

        // DID, SDID, DC
        let did = (cursor.get_u16() & 0xFF) as u8;
        let sdid = (cursor.get_u16() & 0xFF) as u8;
        let data_count = (cursor.get_u16() & 0xFF) as u8;

        // User Data Words
        let mut user_data = Vec::with_capacity(data_count as usize);
        for _ in 0..data_count {
            if cursor.len() < 2 {
                return Err(NetError::parse(0, "Insufficient data for UDW"));
            }
            user_data.push((cursor.get_u16() & 0xFF) as u8);
        }

        // Checksum
        if cursor.len() < 2 {
            return Err(NetError::parse(0, "Missing checksum"));
        }
        let checksum = cursor.get_u16() & 0x1FF;

        let anc_data = Self {
            did,
            sdid,
            data_count,
            user_data,
            checksum,
            location,
        };

        // Validate checksum
        if !anc_data.validate_checksum() {
            return Err(NetError::protocol("ANC checksum mismatch"));
        }

        Ok(anc_data)
    }

    /// Gets the data type.
    #[must_use]
    pub fn data_type(&self) -> AncillaryDataType {
        AncillaryDataType::from_did(self.did)
    }

    /// Checks if this is CEA-608 caption data.
    #[must_use]
    pub fn is_cea608(&self) -> bool {
        self.did == 0x61 && self.sdid == 0x01
    }

    /// Checks if this is CEA-708 caption data.
    #[must_use]
    pub fn is_cea708(&self) -> bool {
        self.did == 0x61 && self.sdid == 0x02
    }

    /// Checks if this is timecode.
    #[must_use]
    pub fn is_timecode(&self) -> bool {
        self.did == 0x60
    }
}

/// Ancillary configuration.
#[derive(Debug, Clone)]
pub struct AncillaryConfig {
    /// Maximum packets per frame.
    pub max_packets_per_frame: usize,
    /// Field identification for interlaced content.
    pub field_id: bool,
}

impl Default for AncillaryConfig {
    fn default() -> Self {
        Self {
            max_packets_per_frame: 64,
            field_id: false,
        }
    }
}

/// RTP header extension for ancillary data (ST 2110-40).
#[derive(Debug, Clone, Copy)]
pub struct AncillaryHeaderExtension {
    /// Extended sequence number.
    pub extended_sequence: u16,
    /// Field identification.
    pub field_id: bool,
    /// Line number.
    pub line_number: u16,
    /// Horizontal offset.
    pub horizontal_offset: u16,
}

impl AncillaryHeaderExtension {
    /// Creates a new ancillary header extension.
    #[must_use]
    pub const fn new(field_id: bool, line_number: u16, horizontal_offset: u16) -> Self {
        Self {
            extended_sequence: 0,
            field_id,
            line_number,
            horizontal_offset,
        }
    }

    /// Serializes to bytes (8 bytes).
    pub fn serialize(&self, buf: &mut BytesMut) {
        buf.put_u16(self.extended_sequence);

        let line_and_field = (self.line_number & 0x7FFF) | (if self.field_id { 0x8000 } else { 0 });
        buf.put_u16(line_and_field);

        buf.put_u16(self.horizontal_offset);
        buf.put_u16(0); // Reserved
    }

    /// Parses from bytes.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < 8 {
            return Err(NetError::parse(0, "ANC header extension too short"));
        }

        let mut cursor = &data[..];
        let extended_sequence = cursor.get_u16();
        let line_and_field = cursor.get_u16();
        let horizontal_offset = cursor.get_u16();
        let _reserved = cursor.get_u16();

        let field_id = (line_and_field & 0x8000) != 0;
        let line_number = line_and_field & 0x7FFF;

        Ok(Self {
            extended_sequence,
            field_id,
            line_number,
            horizontal_offset,
        })
    }
}

/// Ancillary packet containing ANC data.
#[derive(Debug, Clone)]
pub struct AncillaryPacket {
    /// RTP header.
    pub header: RtpHeader,
    /// Ancillary header extension.
    pub anc_extension: AncillaryHeaderExtension,
    /// Ancillary data packets.
    pub anc_data: Vec<AncillaryData>,
}

impl AncillaryPacket {
    /// Creates a new ancillary packet.
    #[must_use]
    pub fn new(
        header: RtpHeader,
        anc_extension: AncillaryHeaderExtension,
        anc_data: Vec<AncillaryData>,
    ) -> Self {
        Self {
            header,
            anc_extension,
            anc_data,
        }
    }

    /// Parses from RTP packet.
    pub fn from_rtp(rtp_packet: &RtpPacket) -> NetResult<Self> {
        // Extract ancillary header extension
        let ext_data = rtp_packet
            .header
            .extension_data
            .as_ref()
            .ok_or_else(|| NetError::protocol("Missing ANC header extension"))?;

        let anc_extension = AncillaryHeaderExtension::parse(&ext_data.data)?;

        // Parse ancillary data packets from payload
        let mut anc_data = Vec::new();
        let mut offset = 0;

        while offset < rtp_packet.payload.len() {
            let location = if anc_extension.horizontal_offset > 0 {
                AncillaryLocation::HANC {
                    line_number: anc_extension.line_number,
                    offset: anc_extension.horizontal_offset,
                }
            } else {
                AncillaryLocation::VANC {
                    line_number: anc_extension.line_number,
                }
            };

            match AncillaryData::parse(&rtp_packet.payload[offset..], location) {
                Ok(anc) => {
                    let anc_size = 6 + 2 * (3 + anc.data_count as usize + 1); // ADF + DID/SDID/DC + UDW + CS
                    offset += anc_size;
                    anc_data.push(anc);
                }
                Err(_) => break, // No more valid ANC data
            }
        }

        Ok(Self {
            header: rtp_packet.header.clone(),
            anc_extension,
            anc_data,
        })
    }

    /// Converts to RTP packet.
    #[must_use]
    pub fn to_rtp(&self) -> RtpPacket {
        let mut payload = BytesMut::new();

        // Serialize all ANC data
        for anc in &self.anc_data {
            anc.serialize(&mut payload);
        }

        // Create RTP header extension
        let mut ext_data = BytesMut::with_capacity(8);
        self.anc_extension.serialize(&mut ext_data);

        let mut header = self.header.clone();
        header.extension = true;
        header.extension_data = Some(crate::smpte2110::rtp::RtpHeaderExtension {
            profile: 0x0200, // ST 2110-40 extension profile
            data: ext_data.freeze(),
        });

        RtpPacket {
            header,
            payload: payload.freeze(),
        }
    }
}

/// CEA-608 closed caption data (line 21).
#[derive(Debug, Clone, Copy)]
pub struct CEA608Data {
    /// Caption byte 1.
    pub cc_data1: u8,
    /// Caption byte 2.
    pub cc_data2: u8,
}

impl CEA608Data {
    /// Creates a new CEA-608 data packet.
    #[must_use]
    pub const fn new(cc_data1: u8, cc_data2: u8) -> Self {
        Self { cc_data1, cc_data2 }
    }

    /// Converts to ANC packet.
    #[must_use]
    pub fn to_anc(&self, line_number: u16) -> AncillaryData {
        let user_data = vec![self.cc_data1, self.cc_data2];
        let location = AncillaryLocation::VANC { line_number };

        AncillaryData::new(0x61, 0x01, user_data, location)
    }

    /// Parses from ANC packet.
    pub fn from_anc(anc: &AncillaryData) -> NetResult<Self> {
        if !anc.is_cea608() {
            return Err(NetError::protocol("Not a CEA-608 packet"));
        }

        if anc.user_data.len() != 2 {
            return Err(NetError::protocol("Invalid CEA-608 data length"));
        }

        Ok(Self {
            cc_data1: anc.user_data[0],
            cc_data2: anc.user_data[1],
        })
    }
}

/// CEA-708 closed caption data.
#[derive(Debug, Clone)]
pub struct CEA708Data {
    /// Caption data bytes.
    pub cc_data: Vec<u8>,
}

impl CEA708Data {
    /// Creates a new CEA-708 data packet.
    #[must_use]
    pub fn new(cc_data: Vec<u8>) -> Self {
        Self { cc_data }
    }

    /// Converts to ANC packet.
    #[must_use]
    pub fn to_anc(&self, line_number: u16) -> AncillaryData {
        let location = AncillaryLocation::VANC { line_number };
        AncillaryData::new(0x61, 0x02, self.cc_data.clone(), location)
    }

    /// Parses from ANC packet.
    pub fn from_anc(anc: &AncillaryData) -> NetResult<Self> {
        if !anc.is_cea708() {
            return Err(NetError::protocol("Not a CEA-708 packet"));
        }

        Ok(Self {
            cc_data: anc.user_data.clone(),
        })
    }
}

/// SMPTE 12M timecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timecode {
    /// Hours (0-23).
    pub hours: u8,
    /// Minutes (0-59).
    pub minutes: u8,
    /// Seconds (0-59).
    pub seconds: u8,
    /// Frames (0-29 or 0-24 depending on frame rate).
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

    /// Converts to ANC packet.
    #[must_use]
    pub fn to_anc(&self, line_number: u16) -> AncillaryData {
        let mut user_data = vec![0u8; 8];

        // Encode timecode (BCD format)
        user_data[0] = ((self.frames / 10) << 4) | (self.frames % 10);
        user_data[1] = ((self.seconds / 10) << 4) | (self.seconds % 10);
        user_data[2] = ((self.minutes / 10) << 4) | (self.minutes % 10);
        user_data[3] = ((self.hours / 10) << 4) | (self.hours % 10);

        // Drop frame flag
        if self.drop_frame {
            user_data[0] |= 0x40;
        }

        let location = AncillaryLocation::VANC { line_number };
        AncillaryData::new(0x60, 0x60, user_data, location)
    }

    /// Parses from ANC packet.
    pub fn from_anc(anc: &AncillaryData) -> NetResult<Self> {
        if !anc.is_timecode() {
            return Err(NetError::protocol("Not a timecode packet"));
        }

        if anc.user_data.len() < 4 {
            return Err(NetError::protocol("Invalid timecode data length"));
        }

        let frames_bcd = anc.user_data[0];
        let seconds_bcd = anc.user_data[1];
        let minutes_bcd = anc.user_data[2];
        let hours_bcd = anc.user_data[3];

        let frames = ((frames_bcd >> 4) & 0x03) * 10 + (frames_bcd & 0x0F);
        let seconds = ((seconds_bcd >> 4) & 0x07) * 10 + (seconds_bcd & 0x0F);
        let minutes = ((minutes_bcd >> 4) & 0x07) * 10 + (minutes_bcd & 0x0F);
        let hours = ((hours_bcd >> 4) & 0x03) * 10 + (hours_bcd & 0x0F);
        let drop_frame = (frames_bcd & 0x40) != 0;

        Ok(Self {
            hours,
            minutes,
            seconds,
            frames,
            drop_frame,
        })
    }

    /// Formats as string (HH:MM:SS:FF or HH:MM:SS;FF for drop frame).
    #[must_use]
    pub fn format(&self) -> String {
        let separator = if self.drop_frame { ';' } else { ':' };
        format!(
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, separator, self.frames
        )
    }
}

/// Active Format Description (AFD).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AFD {
    /// AFD code (4 bits).
    pub afd_code: u8,
    /// Aspect ratio.
    pub aspect_ratio: AspectRatio,
}

/// Aspect ratio for AFD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectRatio {
    /// 4:3
    Ratio4_3,
    /// 16:9
    Ratio16_9,
}

impl AFD {
    /// Creates a new AFD.
    #[must_use]
    pub const fn new(afd_code: u8, aspect_ratio: AspectRatio) -> Self {
        Self {
            afd_code,
            aspect_ratio,
        }
    }

    /// Converts to ANC packet.
    #[must_use]
    pub fn to_anc(&self, line_number: u16) -> AncillaryData {
        let ar_code = match self.aspect_ratio {
            AspectRatio::Ratio4_3 => 0,
            AspectRatio::Ratio16_9 => 1,
        };

        let user_data = vec![
            (ar_code << 7) | ((self.afd_code & 0x0F) << 3),
            0, // Reserved
        ];

        let location = AncillaryLocation::VANC { line_number };
        AncillaryData::new(0x41, 0x05, user_data, location)
    }
}

/// Ancillary encoder.
pub struct AncillaryEncoder {
    /// Configuration.
    config: AncillaryConfig,
    /// Current sequence number.
    sequence_number: u16,
    /// SSRC.
    ssrc: u32,
}

impl AncillaryEncoder {
    /// Creates a new ancillary encoder.
    #[must_use]
    pub fn new(config: AncillaryConfig, ssrc: u32) -> Self {
        Self {
            config,
            sequence_number: rand::random(),
            ssrc,
        }
    }

    /// Encodes ancillary data into RTP packet.
    pub fn encode(
        &mut self,
        anc_data: Vec<AncillaryData>,
        timestamp: u32,
    ) -> NetResult<AncillaryPacket> {
        if anc_data.is_empty() {
            return Err(NetError::protocol("No ancillary data to encode"));
        }

        // Use location from first packet
        let first_location = anc_data[0].location;
        let line_number = first_location.line_number();
        let horizontal_offset = match first_location {
            AncillaryLocation::HANC { offset, .. } => offset,
            AncillaryLocation::VANC { .. } => 0,
        };

        let anc_ext =
            AncillaryHeaderExtension::new(self.config.field_id, line_number, horizontal_offset);

        let header = RtpHeader {
            padding: false,
            extension: true,
            csrc_count: 0,
            marker: true,
            payload_type: RTP_PAYLOAD_TYPE_ANC,
            sequence_number: self.sequence_number,
            timestamp,
            ssrc: self.ssrc,
            csrcs: Vec::new(),
            extension_data: None,
        };

        self.sequence_number = self.sequence_number.wrapping_add(1);

        Ok(AncillaryPacket::new(header, anc_ext, anc_data))
    }

    /// Gets the configuration.
    #[must_use]
    pub const fn config(&self) -> &AncillaryConfig {
        &self.config
    }
}

/// Ancillary decoder.
pub struct AncillaryDecoder {
    /// Configuration.
    config: AncillaryConfig,
    /// Packet buffer.
    packet_buffer: HashMap<u32, Vec<AncillaryData>>,
}

impl AncillaryDecoder {
    /// Creates a new ancillary decoder.
    #[must_use]
    pub fn new(config: AncillaryConfig) -> Self {
        Self {
            config,
            packet_buffer: HashMap::new(),
        }
    }

    /// Processes an RTP packet.
    pub fn process_rtp_packet(&mut self, rtp_packet: &RtpPacket) -> NetResult<()> {
        let anc_packet = AncillaryPacket::from_rtp(rtp_packet)?;
        let timestamp = anc_packet.header.timestamp;

        self.packet_buffer
            .entry(timestamp)
            .or_insert_with(Vec::new)
            .extend(anc_packet.anc_data);

        Ok(())
    }

    /// Gets ancillary data for a timestamp.
    pub fn get_anc_data(&mut self, timestamp: u32) -> Option<Vec<AncillaryData>> {
        self.packet_buffer.remove(&timestamp)
    }

    /// Gets the configuration.
    #[must_use]
    pub const fn config(&self) -> &AncillaryConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ancillary_data_type() {
        assert_eq!(AncillaryDataType::from_did(0x61).as_did(), 0x61);
        assert_eq!(AncillaryDataType::Timecode.as_did(), 0x60);
    }

    #[test]
    fn test_ancillary_data() {
        let user_data = vec![0x12, 0x34, 0x56];
        let location = AncillaryLocation::VANC { line_number: 10 };

        let anc = AncillaryData::new(0x61, 0x01, user_data.clone(), location);

        assert_eq!(anc.did, 0x61);
        assert_eq!(anc.sdid, 0x01);
        assert_eq!(anc.data_count, 3);
        assert!(anc.validate_checksum());
    }

    #[test]
    fn test_cea608_data() {
        let cea608 = CEA608Data::new(0x80, 0x80);
        let anc = cea608.to_anc(21);

        assert!(anc.is_cea608());

        let parsed = CEA608Data::from_anc(&anc).expect("should succeed in test");
        assert_eq!(parsed.cc_data1, 0x80);
        assert_eq!(parsed.cc_data2, 0x80);
    }

    #[test]
    fn test_timecode() {
        let tc = Timecode::new(10, 30, 45, 12, false);
        assert_eq!(tc.format(), "10:30:45:12");

        let tc_drop = Timecode::new(10, 30, 45, 12, true);
        assert_eq!(tc_drop.format(), "10:30:45;12");

        let anc = tc.to_anc(10);
        assert!(anc.is_timecode());

        let parsed = Timecode::from_anc(&anc).expect("should succeed in test");
        assert_eq!(parsed, tc);
    }

    #[test]
    fn test_afd() {
        let afd = AFD::new(0x08, AspectRatio::Ratio16_9);
        let anc = afd.to_anc(11);

        assert_eq!(anc.did, 0x41);
        assert_eq!(anc.sdid, 0x05);
    }

    #[test]
    fn test_ancillary_encoder() {
        let config = AncillaryConfig::default();
        let mut encoder = AncillaryEncoder::new(config, 12345);

        let user_data = vec![0x12, 0x34];
        let location = AncillaryLocation::VANC { line_number: 10 };
        let anc = AncillaryData::new(0x61, 0x01, user_data, location);

        let packet = encoder
            .encode(vec![anc], 1000)
            .expect("should succeed in test");
        assert_eq!(packet.header.payload_type, RTP_PAYLOAD_TYPE_ANC);
        assert!(!packet.anc_data.is_empty());
    }
}

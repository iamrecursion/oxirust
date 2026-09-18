//! Program Specific Information (PSI) parsing.
//!
//! PSI tables describe the structure of the transport stream:
//! - PAT (Program Association Table) - maps programs to PMT PIDs
//! - PMT (Program Map Table) - maps elementary streams to PIDs
//! - SDT (Service Description Table) - service names and descriptions

use oximedia_core::{CodecId, OxiError, OxiResult};
use std::collections::HashMap;

/// CRC-32 polynomial for MPEG-2 PSI tables.
const CRC32_POLYNOMIAL: u32 = 0x04C1_1DB7;

/// Program Association Table (PAT) - Maps program numbers to PMT PIDs.
#[derive(Debug, Clone)]
pub struct ProgramAssociationTable {
    /// Transport stream ID.
    #[allow(dead_code)]
    pub transport_stream_id: u16,
    /// Version number.
    #[allow(dead_code)]
    pub version: u8,
    /// Map of program number to PMT PID.
    pub programs: HashMap<u16, u16>,
}

impl ProgramAssociationTable {
    /// Parses a PAT from section data.
    ///
    /// # Arguments
    ///
    /// * `data` - Section data (without pointer field)
    ///
    /// # Errors
    ///
    /// Returns an error if the PAT is malformed or CRC check fails.
    pub fn parse(data: &[u8]) -> OxiResult<Self> {
        if data.len() < 8 {
            return Err(OxiError::InvalidData("PAT too short".to_string()));
        }

        // Verify table ID
        if data[0] != 0x00 {
            return Err(OxiError::InvalidData(format!(
                "Invalid PAT table ID: expected 0x00, got 0x{:02X}",
                data[0]
            )));
        }

        let section_length = (((u16::from(data[1]) & 0x0F) << 8) | u16::from(data[2])) as usize;
        if data.len() < section_length + 3 {
            return Err(OxiError::InvalidData(format!(
                "PAT section too short: expected {}, got {}",
                section_length + 3,
                data.len()
            )));
        }

        // Verify CRC
        verify_crc32(&data[..section_length + 3])?;

        let transport_stream_id = (u16::from(data[3]) << 8) | u16::from(data[4]);
        let version = (data[5] >> 1) & 0x1F;

        // Parse program entries
        let mut programs = HashMap::new();
        let entries_end = section_length + 3 - 4; // Exclude CRC

        let mut offset = 8;
        while offset + 4 <= entries_end {
            let program_number = (u16::from(data[offset]) << 8) | u16::from(data[offset + 1]);
            let pmt_pid = (u16::from(data[offset + 2] & 0x1F) << 8) | u16::from(data[offset + 3]);

            if program_number != 0 {
                // program_number 0 is network PID, we skip it
                programs.insert(program_number, pmt_pid);
            }

            offset += 4;
        }

        Ok(Self {
            transport_stream_id,
            version,
            programs,
        })
    }
}

/// Stream type enumeration for PMT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    /// MPEG-2 Video (not supported - patent encumbered).
    Mpeg2Video,
    /// H.264/AVC (not supported - patent encumbered).
    H264,
    /// H.265/HEVC (not supported - patent encumbered).
    H265,
    /// AV1 video.
    Av1,
    /// VP9 video.
    Vp9,
    /// VP8 video.
    Vp8,
    /// MPEG-1 Audio Layer II (not supported - patent encumbered).
    Mpeg1Audio,
    /// AAC Audio (not supported - patent encumbered).
    AacAudio,
    /// Opus audio.
    Opus,
    /// FLAC audio.
    Flac,
    /// PCM audio.
    Pcm,
    /// Private stream (may contain various formats).
    PrivateStream,
    /// Unknown/unsupported stream type.
    Unknown(u8),
}

impl StreamType {
    /// Creates a `StreamType` from an 8-bit type value.
    #[must_use]
    pub const fn from_type_id(type_id: u8) -> Self {
        match type_id {
            0x01 | 0x02 => Self::Mpeg2Video,
            0x1B => Self::H264,
            0x24 => Self::H265,
            0x03 | 0x04 => Self::Mpeg1Audio,
            0x0F | 0x11 => Self::AacAudio,
            0x06 => Self::PrivateStream, // May contain various formats
            0x80 => Self::Pcm,           // User private, often PCM
            0x81 => Self::Opus,          // User private mapping for Opus
            0x82 => Self::Flac,          // User private mapping for FLAC
            0x83 => Self::Vp8,           // User private mapping for VP8
            0x84 => Self::Vp9,           // User private mapping for VP9
            0x85 => Self::Av1,           // User private mapping for AV1
            _ => Self::Unknown(type_id),
        }
    }

    /// Converts to `CodecId` if supported.
    #[must_use]
    pub const fn to_codec_id(self) -> Option<CodecId> {
        match self {
            Self::Av1 => Some(CodecId::Av1),
            Self::Vp9 => Some(CodecId::Vp9),
            Self::Vp8 => Some(CodecId::Vp8),
            Self::Opus => Some(CodecId::Opus),
            Self::Flac => Some(CodecId::Flac),
            Self::Pcm => Some(CodecId::Pcm),
            _ => None,
        }
    }

    /// Returns true if this stream type is patent-encumbered.
    #[must_use]
    pub const fn is_patent_encumbered(self) -> bool {
        matches!(
            self,
            Self::Mpeg2Video | Self::H264 | Self::H265 | Self::Mpeg1Audio | Self::AacAudio
        )
    }
}

/// Elementary stream information from PMT.
#[derive(Debug, Clone)]
pub struct ElementaryStreamInfo {
    /// Stream type.
    #[allow(dead_code)]
    pub stream_type: StreamType,
    /// Elementary stream PID.
    pub pid: u16,
    /// Codec ID, if supported.
    pub codec_id: Option<CodecId>,
    /// Descriptors (not parsed, stored as raw bytes).
    #[allow(dead_code)]
    pub descriptors: Vec<u8>,
}

/// Program Map Table (PMT) - Describes elementary streams in a program.
#[derive(Debug, Clone)]
pub struct ProgramMapTable {
    /// Program number.
    pub program_number: u16,
    /// Version number.
    #[allow(dead_code)]
    pub version: u8,
    /// PCR PID (PID containing Program Clock Reference).
    #[allow(dead_code)]
    pub pcr_pid: u16,
    /// Elementary stream information.
    pub streams: Vec<ElementaryStreamInfo>,
}

impl ProgramMapTable {
    /// Parses a PMT from section data.
    ///
    /// # Arguments
    ///
    /// * `data` - Section data (without pointer field)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The PMT is malformed
    /// - CRC check fails
    /// - Patent-encumbered codecs are detected
    pub fn parse(data: &[u8]) -> OxiResult<Self> {
        if data.len() < 12 {
            return Err(OxiError::InvalidData("PMT too short".to_string()));
        }

        // Verify table ID
        if data[0] != 0x02 {
            return Err(OxiError::InvalidData(format!(
                "Invalid PMT table ID: expected 0x02, got 0x{:02X}",
                data[0]
            )));
        }

        let section_length = (((u16::from(data[1]) & 0x0F) << 8) | u16::from(data[2])) as usize;
        if data.len() < section_length + 3 {
            return Err(OxiError::InvalidData(format!(
                "PMT section too short: expected {}, got {}",
                section_length + 3,
                data.len()
            )));
        }

        // Verify CRC
        verify_crc32(&data[..section_length + 3])?;

        let program_number = (u16::from(data[3]) << 8) | u16::from(data[4]);
        let version = (data[5] >> 1) & 0x1F;
        let pcr_pid = (u16::from(data[8] & 0x1F) << 8) | u16::from(data[9]);

        let program_info_length =
            (((u16::from(data[10]) & 0x0F) << 8) | u16::from(data[11])) as usize;

        let mut offset = 12 + program_info_length;
        let streams_end = section_length + 3 - 4; // Exclude CRC

        let mut streams = Vec::new();

        while offset + 5 <= streams_end {
            let stream_type_id = data[offset];
            let stream_type = StreamType::from_type_id(stream_type_id);

            // Check for patent-encumbered codecs
            if stream_type.is_patent_encumbered() {
                return Err(OxiError::PatentViolation(format!(
                    "Patent-encumbered stream type detected: {stream_type:?} (0x{stream_type_id:02X})"
                )));
            }

            let elementary_pid =
                (u16::from(data[offset + 1] & 0x1F) << 8) | u16::from(data[offset + 2]);

            let es_info_length = (((u16::from(data[offset + 3]) & 0x0F) << 8)
                | u16::from(data[offset + 4])) as usize;

            let descriptors = if es_info_length > 0 && offset + 5 + es_info_length <= streams_end {
                data[offset + 5..offset + 5 + es_info_length].to_vec()
            } else {
                Vec::new()
            };

            streams.push(ElementaryStreamInfo {
                stream_type,
                pid: elementary_pid,
                codec_id: stream_type.to_codec_id(),
                descriptors,
            });

            offset += 5 + es_info_length;
        }

        Ok(Self {
            program_number,
            version,
            pcr_pid,
            streams,
        })
    }
}

/// Service Description Table (SDT) entry.
#[derive(Debug, Clone)]
pub struct ServiceDescription {
    /// Service ID.
    #[allow(dead_code)]
    pub service_id: u16,
    /// Service name.
    #[allow(dead_code)]
    pub service_name: Option<String>,
    /// Service provider name.
    #[allow(dead_code)]
    pub provider_name: Option<String>,
}

/// Service Description Table (SDT) - Contains service/channel information.
#[derive(Debug, Clone)]
pub struct ServiceDescriptionTable {
    /// Transport stream ID.
    #[allow(dead_code)]
    pub transport_stream_id: u16,
    /// Version number.
    #[allow(dead_code)]
    pub version: u8,
    /// Service descriptions.
    #[allow(dead_code)]
    pub services: Vec<ServiceDescription>,
}

impl ServiceDescriptionTable {
    /// Parses an SDT from section data.
    ///
    /// # Arguments
    ///
    /// * `data` - Section data
    ///
    /// # Errors
    ///
    /// Returns an error if the SDT is malformed or CRC check fails.
    #[allow(dead_code)]
    pub fn parse(data: &[u8]) -> OxiResult<Self> {
        if data.len() < 11 {
            return Err(OxiError::InvalidData("SDT too short".to_string()));
        }

        // Verify table ID (0x42 for actual SDT)
        if data[0] != 0x42 {
            return Err(OxiError::InvalidData(format!(
                "Invalid SDT table ID: expected 0x42, got 0x{:02X}",
                data[0]
            )));
        }

        let section_length = (((u16::from(data[1]) & 0x0F) << 8) | u16::from(data[2])) as usize;
        if data.len() < section_length + 3 {
            return Err(OxiError::InvalidData("SDT section too short".to_string()));
        }

        // Verify CRC
        verify_crc32(&data[..section_length + 3])?;

        let transport_stream_id = (u16::from(data[3]) << 8) | u16::from(data[4]);
        let version = (data[5] >> 1) & 0x1F;

        let mut services = Vec::new();
        let mut offset = 11; // Skip header and original_network_id

        let services_end = section_length + 3 - 4; // Exclude CRC

        while offset + 5 <= services_end {
            let service_id = (u16::from(data[offset]) << 8) | u16::from(data[offset + 1]);
            let descriptors_loop_length = (((u16::from(data[offset + 3]) & 0x0F) << 8)
                | u16::from(data[offset + 4])) as usize;

            let mut service_name = None;
            let mut provider_name = None;

            // Parse descriptors (simplified - just extract service descriptor)
            let desc_end = offset + 5 + descriptors_loop_length;
            let mut desc_offset = offset + 5;

            while desc_offset + 2 <= desc_end {
                let descriptor_tag = data[desc_offset];
                let descriptor_length = data[desc_offset + 1] as usize;

                if descriptor_tag == 0x48 && desc_offset + 2 + descriptor_length <= desc_end {
                    // Service descriptor
                    let desc_data = &data[desc_offset + 2..desc_offset + 2 + descriptor_length];
                    if desc_data.len() >= 3 {
                        let provider_name_length = desc_data[1] as usize;
                        if desc_data.len() > 2 + provider_name_length {
                            provider_name =
                                String::from_utf8(desc_data[2..2 + provider_name_length].to_vec())
                                    .ok();

                            let service_name_length = desc_data[2 + provider_name_length] as usize;
                            if desc_data.len() >= 3 + provider_name_length + service_name_length {
                                service_name = String::from_utf8(
                                    desc_data[3 + provider_name_length
                                        ..3 + provider_name_length + service_name_length]
                                        .to_vec(),
                                )
                                .ok();
                            }
                        }
                    }
                }

                desc_offset += 2 + descriptor_length;
            }

            services.push(ServiceDescription {
                service_id,
                service_name,
                provider_name,
            });

            offset = desc_end;
        }

        Ok(Self {
            transport_stream_id,
            version,
            services,
        })
    }
}

/// Section assembler for PSI tables that span multiple TS packets.
#[derive(Debug, Clone)]
pub struct SectionAssembler {
    /// Accumulated section data.
    data: Vec<u8>,
    /// Expected section length (including header).
    expected_length: Option<usize>,
}

impl Default for SectionAssembler {
    fn default() -> Self {
        Self::new()
    }
}

impl SectionAssembler {
    /// Creates a new section assembler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            expected_length: None,
        }
    }

    /// Adds packet payload data to the section.
    ///
    /// Returns `Some(section_data)` when a complete section is assembled.
    ///
    /// # Arguments
    ///
    /// * `payload` - Packet payload data
    /// * `payload_unit_start` - Whether this is the start of a new section
    pub fn push(&mut self, payload: &[u8], payload_unit_start: bool) -> Option<Vec<u8>> {
        if payload_unit_start {
            // New section starts - reset
            self.data.clear();
            self.expected_length = None;

            if payload.is_empty() {
                return None;
            }

            // Skip pointer field
            let pointer = payload[0] as usize;
            if pointer + 1 >= payload.len() {
                return None;
            }

            self.data.extend_from_slice(&payload[pointer + 1..]);
        } else {
            // Continue existing section
            self.data.extend_from_slice(payload);
        }

        // Parse section length if we have enough data
        if self.expected_length.is_none() && self.data.len() >= 3 {
            let section_length =
                (((u16::from(self.data[1]) & 0x0F) << 8) | u16::from(self.data[2])) as usize;
            self.expected_length = Some(section_length + 3);
        }

        // Check if section is complete
        if let Some(expected) = self.expected_length {
            if self.data.len() >= expected {
                let section = self.data[..expected].to_vec();
                self.data.clear();
                self.expected_length = None;
                return Some(section);
            }
        }

        None
    }

    /// Resets the assembler state.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.data.clear();
        self.expected_length = None;
    }
}

/// Verifies the CRC-32 checksum of a PSI section.
///
/// # Arguments
///
/// * `data` - Complete section data including CRC
///
/// # Errors
///
/// Returns an error if the CRC check fails.
fn verify_crc32(data: &[u8]) -> OxiResult<()> {
    if data.len() < 4 {
        return Err(OxiError::InvalidData(
            "Section too short for CRC".to_string(),
        ));
    }

    let computed_crc = compute_crc32(data);
    if computed_crc != 0 {
        return Err(OxiError::InvalidData(format!(
            "CRC check failed: computed 0x{computed_crc:08X}"
        )));
    }

    Ok(())
}

/// Computes the CRC-32 checksum for MPEG-2 PSI tables.
///
/// The CRC should be 0 for valid data (including the CRC field).
fn compute_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;

    for &byte in data {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            if crc & 0x8000_0000 != 0 {
                crc = (crc << 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc <<= 1;
            }
        }
    }

    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_type_from_id() {
        assert_eq!(StreamType::from_type_id(0x1B), StreamType::H264);
        assert_eq!(StreamType::from_type_id(0x24), StreamType::H265);
        assert_eq!(StreamType::from_type_id(0x81), StreamType::Opus);
        assert_eq!(StreamType::from_type_id(0x85), StreamType::Av1);
    }

    #[test]
    fn test_stream_type_patent_check() {
        assert!(StreamType::H264.is_patent_encumbered());
        assert!(StreamType::H265.is_patent_encumbered());
        assert!(StreamType::AacAudio.is_patent_encumbered());
        assert!(!StreamType::Av1.is_patent_encumbered());
        assert!(!StreamType::Opus.is_patent_encumbered());
    }

    #[test]
    fn test_stream_type_to_codec_id() {
        assert_eq!(StreamType::Av1.to_codec_id(), Some(CodecId::Av1));
        assert_eq!(StreamType::Opus.to_codec_id(), Some(CodecId::Opus));
        assert_eq!(StreamType::H264.to_codec_id(), None);
    }

    #[test]
    fn test_crc32_computation() {
        // Test vector: empty data with appended CRC
        let test_data = vec![0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00, 0x00];
        let crc = compute_crc32(&test_data);

        // Append the CRC and verify it produces 0
        let mut data_with_crc = test_data;
        data_with_crc.extend_from_slice(&crc.to_be_bytes());
        assert_eq!(compute_crc32(&data_with_crc), 0);
    }

    #[test]
    fn test_section_assembler() {
        let mut assembler = SectionAssembler::new();

        // Simulate a section split across packets
        let section_data = vec![
            0x00, 0xB0, 0x0D, // Table ID and length
            0x00, 0x01, 0xC1, 0x00, 0x00, // Header
            0x00, 0x01, 0xE0, 0x20, // Data
            0x00, 0x00, 0x00, 0x00, // CRC placeholder
        ];

        // First packet with pointer field
        let mut packet1 = vec![0x00]; // Pointer = 0
        packet1.extend_from_slice(&section_data[..8]);

        let result = assembler.push(&packet1, true);
        assert!(result.is_none()); // Not complete yet

        // Second packet continues
        let packet2 = section_data[8..].to_vec();
        let result = assembler.push(&packet2, false);
        assert!(result.is_some()); // Should be complete now
    }
}

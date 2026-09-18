//! Profile 10 (AV1-based Dolby Vision) metadata structure support.
//!
//! Dolby Vision Profile 10 uses AV1 as the base codec and carries RPU metadata
//! in dedicated Metadata OBUs (Open Bitstream Units).  This module defines the
//! Profile 10-specific header, OBU framing, and metadata container structures.
//!
//! # Reference
//!
//! Dolby Vision Streams within the AV1 Video Coding Format — Specification
//! version 1.0.  This implementation covers metadata-only support.

use crate::{DolbyVisionError, DolbyVisionRpu, Profile, Result};

// ---------------------------------------------------------------------------
// Profile 10 constants
// ---------------------------------------------------------------------------

/// AV1 Metadata OBU type value used by Dolby Vision (type 16).
pub const AV1_METADATA_OBU_TYPE_DV: u8 = 16;

/// Dolby Vision IANA registered organization code in MDCV metadata.
pub const DV_IANA_ORG_CODE: u16 = 0x003C;

/// Profile 10 supports AV1 base layer with RPU carried in Metadata OBUs.
/// The signal is always HDR/PQ-based (no HLG for Profile 10).
pub const PROFILE10_SIGNAL: &str = "AV1/PQ";

// ---------------------------------------------------------------------------
// Profile 10 types
// ---------------------------------------------------------------------------

/// AV1 Metadata OBU header for Dolby Vision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1MetadataObuHeader {
    /// OBU type (should be `AV1_METADATA_OBU_TYPE_DV`).
    pub obu_type: u8,
    /// Whether the extension header is present.
    pub extension_flag: bool,
    /// Whether the size field is present (recommended `true`).
    pub has_size_field: bool,
    /// AV1 temporal layer ID (0–7).
    pub temporal_id: u8,
    /// AV1 spatial layer ID (0–3).
    pub spatial_id: u8,
}

impl Default for Av1MetadataObuHeader {
    fn default() -> Self {
        Self {
            obu_type: AV1_METADATA_OBU_TYPE_DV,
            extension_flag: false,
            has_size_field: true,
            temporal_id: 0,
            spatial_id: 0,
        }
    }
}

impl Av1MetadataObuHeader {
    /// Encode the OBU header as two bytes (extended AV1 format).
    ///
    /// Byte 0 layout: `obu_type[7:4] | has_extension[3] | has_size_field[2] | reserved[1:0]`
    /// (obu_type uses the full upper nibble; this is a non-standard extension to support
    /// Dolby Vision OBU type 16 which exceeds the 4-bit AV1 standard range).
    ///
    /// For the wire format we use one byte and encode the type in bits \[7:4\].
    #[must_use]
    pub fn encode_byte(&self) -> u8 {
        // Bits 7:4 = upper 4 bits of obu_type (covers 0-255 by using a 2-byte scheme,
        // but we store the low byte here for simplicity)
        let typ = (self.obu_type & 0x1F) << 3;
        let ext = if self.extension_flag { 1 << 2 } else { 0 };
        let sz = if self.has_size_field { 1 << 1 } else { 0 };
        typ | ext | sz
    }

    /// Decode an OBU header from a byte.
    #[must_use]
    pub fn from_byte(byte: u8) -> Self {
        Self {
            obu_type: (byte >> 3) & 0x1F,
            extension_flag: (byte >> 2) & 1 == 1,
            has_size_field: (byte >> 1) & 1 == 1,
            temporal_id: 0,
            spatial_id: 0,
        }
    }
}

/// Profile 10-specific RPU container embedded in an AV1 Metadata OBU.
#[derive(Debug, Clone)]
pub struct Profile10RpuContainer {
    /// AV1 Metadata OBU header.
    pub obu_header: Av1MetadataObuHeader,
    /// Organization code identifying the metadata type.
    pub iana_org_code: u16,
    /// Length of the embedded RPU payload in bytes.
    pub rpu_payload_size: u32,
    /// Raw RPU bytes (without NAL emulation prevention).
    pub rpu_payload: Vec<u8>,
}

impl Profile10RpuContainer {
    /// Create a new container for the given raw RPU payload.
    #[must_use]
    pub fn new(rpu_payload: Vec<u8>) -> Self {
        let size = rpu_payload.len() as u32;
        Self {
            obu_header: Av1MetadataObuHeader::default(),
            iana_org_code: DV_IANA_ORG_CODE,
            rpu_payload_size: size,
            rpu_payload,
        }
    }

    /// Serialize the container to bytes (simplified framing).
    ///
    /// Layout:
    /// 1. OBU header byte
    /// 2. IANA org code (2 bytes, big-endian)
    /// 3. RPU payload size (4 bytes, big-endian)
    /// 4. RPU payload bytes
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(7 + self.rpu_payload.len());
        out.push(self.obu_header.encode_byte());
        out.push((self.iana_org_code >> 8) as u8);
        out.push(self.iana_org_code as u8);
        out.push((self.rpu_payload_size >> 24) as u8);
        out.push((self.rpu_payload_size >> 16) as u8);
        out.push((self.rpu_payload_size >> 8) as u8);
        out.push(self.rpu_payload_size as u8);
        out.extend_from_slice(&self.rpu_payload);
        out
    }

    /// Deserialize a container from bytes.
    ///
    /// # Errors
    ///
    /// Returns error if the data is too short or the OBU type is wrong.
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 7 {
            return Err(DolbyVisionError::InvalidNalUnit(
                "Profile 10 OBU too short".to_string(),
            ));
        }
        let obu_header = Av1MetadataObuHeader::from_byte(data[0]);
        if obu_header.obu_type != AV1_METADATA_OBU_TYPE_DV {
            return Err(DolbyVisionError::InvalidNalUnit(format!(
                "Expected DV OBU type {}, got {}",
                AV1_METADATA_OBU_TYPE_DV, obu_header.obu_type
            )));
        }
        let iana_org_code = (u16::from(data[1]) << 8) | u16::from(data[2]);
        if iana_org_code != DV_IANA_ORG_CODE {
            return Err(DolbyVisionError::InvalidPayload(format!(
                "Unexpected IANA org code {:#06x}",
                iana_org_code
            )));
        }
        let rpu_payload_size = (u32::from(data[3]) << 24)
            | (u32::from(data[4]) << 16)
            | (u32::from(data[5]) << 8)
            | u32::from(data[6]);

        let expected_len = 7 + rpu_payload_size as usize;
        if data.len() < expected_len {
            return Err(DolbyVisionError::InvalidPayload(format!(
                "OBU payload truncated: need {} bytes, have {}",
                expected_len,
                data.len()
            )));
        }
        let rpu_payload = data[7..expected_len].to_vec();
        Ok(Self {
            obu_header,
            iana_org_code,
            rpu_payload_size,
            rpu_payload,
        })
    }

    /// Parse the contained RPU payload into a [`DolbyVisionRpu`].
    ///
    /// # Errors
    ///
    /// Returns error if RPU parsing fails.
    pub fn parse_rpu(&self) -> Result<DolbyVisionRpu> {
        let rpu = crate::parser::parse_rpu_bitstream(&self.rpu_payload)?;
        Ok(rpu)
    }
}

/// Profile 10 metadata block containing the frame-level DV information.
#[derive(Debug, Clone)]
pub struct Profile10FrameMetadata {
    /// The underlying RPU.
    pub rpu: DolbyVisionRpu,
    /// AV1 frame index within the current sequence.
    pub av1_frame_index: u64,
    /// Whether this frame is marked as a key frame in AV1.
    pub is_key_frame: bool,
    /// Whether the RPU was carried in an independent OBU (not dependent on
    /// a previous frame's prediction).
    pub is_independent_rpu: bool,
}

impl Profile10FrameMetadata {
    /// Create new Profile 10 frame metadata.
    #[must_use]
    pub fn new(av1_frame_index: u64, is_key_frame: bool) -> Self {
        Self {
            rpu: DolbyVisionRpu::new(Profile::Profile8), // Profile 10 uses Profile 8 base RPU
            av1_frame_index,
            is_key_frame,
            is_independent_rpu: is_key_frame,
        }
    }

    /// Returns `true` when the RPU is self-contained (useful at random-access points).
    #[must_use]
    pub fn is_self_contained(&self) -> bool {
        self.is_independent_rpu
    }
}

/// Profile 10 sequence-level configuration.
#[derive(Debug, Clone)]
pub struct Profile10SequenceConfig {
    /// AV1 sequence profile (0 = Main, 1 = High, 2 = Professional).
    pub av1_seq_profile: u8,
    /// Bit depth (10 for Profile 10).
    pub bit_depth: u8,
    /// Color primaries index.
    pub color_primaries: u8,
    /// Transfer characteristics.
    pub transfer_characteristics: u8,
    /// Matrix coefficients.
    pub matrix_coefficients: u8,
    /// Maximum content light level in nits.
    pub max_cll: u16,
    /// Maximum frame average light level in nits.
    pub max_fall: u16,
}

impl Default for Profile10SequenceConfig {
    fn default() -> Self {
        Self {
            av1_seq_profile: 0, // Main profile
            bit_depth: 10,
            color_primaries: 9,           // BT.2020
            transfer_characteristics: 16, // PQ (ST.2084)
            matrix_coefficients: 9,       // BT.2020 non-constant luminance
            max_cll: 1000,
            max_fall: 400,
        }
    }
}

impl Profile10SequenceConfig {
    /// Create configuration for a 10-bit PQ AV1 stream.
    #[must_use]
    pub fn pq_10bit() -> Self {
        Self::default()
    }

    /// Create configuration for a 12-bit PQ AV1 stream.
    #[must_use]
    pub fn pq_12bit() -> Self {
        Self {
            bit_depth: 12,
            ..Self::default()
        }
    }

    /// Returns `true` if the configuration uses PQ transfer function.
    #[must_use]
    pub fn is_pq(&self) -> bool {
        self.transfer_characteristics == 16
    }

    /// Returns `true` if the configuration uses BT.2020 primaries.
    #[must_use]
    pub fn is_bt2020(&self) -> bool {
        self.color_primaries == 9
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obu_header_default() {
        let h = Av1MetadataObuHeader::default();
        assert_eq!(h.obu_type, AV1_METADATA_OBU_TYPE_DV);
        assert!(h.has_size_field);
        assert!(!h.extension_flag);
    }

    #[test]
    fn test_obu_header_encode_decode_roundtrip() {
        let h = Av1MetadataObuHeader {
            obu_type: AV1_METADATA_OBU_TYPE_DV,
            extension_flag: false,
            has_size_field: true,
            temporal_id: 0,
            spatial_id: 0,
        };
        let byte = h.encode_byte();
        let decoded = Av1MetadataObuHeader::from_byte(byte);
        assert_eq!(decoded.obu_type, h.obu_type);
        assert_eq!(decoded.has_size_field, h.has_size_field);
        assert_eq!(decoded.extension_flag, h.extension_flag);
    }

    #[test]
    fn test_profile10_container_serialize_deserialize() {
        let payload = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let container = Profile10RpuContainer::new(payload.clone());
        let bytes = container.serialize();
        let decoded = Profile10RpuContainer::deserialize(&bytes).expect("deserialize should work");
        assert_eq!(decoded.rpu_payload, payload);
        assert_eq!(decoded.iana_org_code, DV_IANA_ORG_CODE);
    }

    #[test]
    fn test_profile10_container_too_short() {
        let result = Profile10RpuContainer::deserialize(&[0x00, 0x01]);
        assert!(result.is_err());
    }

    #[test]
    fn test_profile10_container_wrong_iana_code() {
        let mut bytes = vec![0x00u8; 7];
        bytes[0] = Av1MetadataObuHeader::default().encode_byte();
        bytes[1] = 0xFF; // Wrong org code
        bytes[2] = 0xFF;
        let result = Profile10RpuContainer::deserialize(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_profile10_frame_metadata_creation() {
        let meta = Profile10FrameMetadata::new(42, true);
        assert_eq!(meta.av1_frame_index, 42);
        assert!(meta.is_key_frame);
        assert!(meta.is_self_contained());
    }

    #[test]
    fn test_profile10_frame_non_key_not_self_contained() {
        let meta = Profile10FrameMetadata::new(100, false);
        assert!(!meta.is_key_frame);
        assert!(!meta.is_self_contained());
    }

    #[test]
    fn test_profile10_seq_config_default() {
        let cfg = Profile10SequenceConfig::default();
        assert_eq!(cfg.bit_depth, 10);
        assert!(cfg.is_pq());
        assert!(cfg.is_bt2020());
    }

    #[test]
    fn test_profile10_seq_config_12bit() {
        let cfg = Profile10SequenceConfig::pq_12bit();
        assert_eq!(cfg.bit_depth, 12);
        assert!(cfg.is_pq());
    }

    #[test]
    fn test_dv_iana_org_code_value() {
        assert_eq!(DV_IANA_ORG_CODE, 0x003C);
    }

    #[test]
    fn test_av1_metadata_obu_type_dv_value() {
        assert_eq!(AV1_METADATA_OBU_TYPE_DV, 16);
    }

    #[test]
    fn test_profile10_signal_string() {
        assert_eq!(PROFILE10_SIGNAL, "AV1/PQ");
    }

    #[test]
    fn test_container_payload_size_matches() {
        let payload = vec![0u8; 64];
        let container = Profile10RpuContainer::new(payload);
        assert_eq!(container.rpu_payload_size, 64);
    }
}

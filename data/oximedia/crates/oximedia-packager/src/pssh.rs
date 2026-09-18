// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! PSSH (Protection System Specific Header) box generation for DRM systems.
//!
//! This module implements ISOBMFF `pssh` box encoding and decoding for
//! Widevine, PlayReady, FairPlay, Marlin, and Common Encryption (CENC) DRM
//! systems.  Box layout follows ISO/IEC 14496-12 §8.1.1 and CENC (ISO 23001-7).

use crate::error::{PackagerError, PackagerResult};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

// ---------------------------------------------------------------------------
// System IDs
// ---------------------------------------------------------------------------

/// Widevine DRM system UUID.
pub const WIDEVINE_SYSTEM_ID: [u8; 16] = [
    0xed, 0xef, 0x8b, 0xa9, 0x79, 0xd6, 0x4a, 0xce, 0xa3, 0xc8, 0x27, 0xdc, 0xd5, 0x1d, 0x21, 0xed,
];

/// PlayReady DRM system UUID.
pub const PLAYREADY_SYSTEM_ID: [u8; 16] = [
    0x9a, 0x04, 0xf0, 0x79, 0x98, 0x40, 0x42, 0x86, 0xab, 0x92, 0xe6, 0x5b, 0xe0, 0x88, 0x5f, 0x95,
];

/// FairPlay DRM system UUID.
pub const FAIRPLAY_SYSTEM_ID: [u8; 16] = [
    0x94, 0xce, 0x86, 0xfb, 0x07, 0xff, 0x4f, 0x43, 0xad, 0xb8, 0x93, 0xd2, 0xfa, 0x96, 0x8c, 0xa2,
];

// ---------------------------------------------------------------------------
// DrmSystem
// ---------------------------------------------------------------------------

/// Identifies a content-protection (DRM) system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrmSystem {
    /// Google Widevine (UUID `edef8ba9-79d6-4ace-a3c8-27dcd51d21ed`).
    Widevine,
    /// Microsoft PlayReady (UUID `9a04f079-9840-4286-ab92-e65be0885f95`).
    PlayReady,
    /// Apple FairPlay Streaming (UUID `94ce86fb-07ff-4f43-adb8-93d2fa968ca2`).
    FairPlay,
    /// Marlin Broadband DRM.
    Marlin,
    /// ISO Common Encryption (generic).
    CommonEncryption,
}

impl DrmSystem {
    /// Returns the 16-byte system ID for this DRM.
    #[must_use]
    pub fn system_id(&self) -> [u8; 16] {
        match self {
            Self::Widevine => WIDEVINE_SYSTEM_ID,
            Self::PlayReady => PLAYREADY_SYSTEM_ID,
            Self::FairPlay => FAIRPLAY_SYSTEM_ID,
            Self::Marlin => [
                0x5e, 0x62, 0x9a, 0xf5, 0x38, 0xda, 0x40, 0x63, 0x89, 0x77, 0x97, 0xff, 0xbd, 0x9a,
                0xd3, 0x4a,
            ],
            Self::CommonEncryption => [
                0x10, 0x77, 0xef, 0xec, 0xc0, 0xb2, 0x4d, 0x02, 0xac, 0xe3, 0x3c, 0x1e, 0x52, 0xe2,
                0xfb, 0x4b,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// PsshBox
// ---------------------------------------------------------------------------

/// A fully parsed or constructed PSSH box.
///
/// PSSH layout (version 0):
/// ```text
/// 4  bytes  size (big-endian, includes header)
/// 4  bytes  "pssh"
/// 1  byte   version  (0 or 1)
/// 3  bytes  flags    (all zero)
/// 16 bytes  system_id
/// [v1 only: 4-byte key_id_count + N×16-byte key_ids]
/// 4  bytes  data_size
/// N  bytes  DRM-specific data
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsshBox {
    /// DRM system UUID.
    pub system_id: [u8; 16],
    /// Box version: 0 or 1 (v1 adds key-ID list).
    pub version: u8,
    /// Optional list of key IDs (version 1 only).
    pub key_ids: Vec<[u8; 16]>,
    /// DRM-system-specific data payload.
    pub data: Vec<u8>,
}

impl PsshBox {
    /// Construct a version-0 PSSH box (no key-ID list).
    #[must_use]
    pub fn new_v0(system_id: [u8; 16], data: Vec<u8>) -> Self {
        Self {
            system_id,
            version: 0,
            key_ids: Vec::new(),
            data,
        }
    }

    /// Construct a version-1 PSSH box (with key-ID list).
    #[must_use]
    pub fn new_v1(system_id: [u8; 16], key_ids: Vec<[u8; 16]>, data: Vec<u8>) -> Self {
        Self {
            system_id,
            version: 1,
            key_ids,
            data,
        }
    }

    /// Encode the box into its binary ISOBMFF representation.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        // Compute total size first so we can write it up front.
        // Layout:
        //   4  (size)
        //   4  ("pssh")
        //   4  (version + flags)
        //   16 (system_id)
        //   [v1: 4 + key_ids.len()*16]
        //   4  (data_size)
        //   N  (data)
        let v1_extra = if self.version >= 1 {
            4 + self.key_ids.len() * 16
        } else {
            0
        };
        let total = 4 + 4 + 4 + 16 + v1_extra + 4 + self.data.len();

        let mut buf: Vec<u8> = Vec::with_capacity(total);

        // Box size (big-endian u32)
        buf.extend_from_slice(&(total as u32).to_be_bytes());

        // fourcc
        buf.extend_from_slice(b"pssh");

        // version (1 byte) + flags (3 bytes, all zero)
        buf.push(self.version);
        buf.extend_from_slice(&[0u8; 3]);

        // system_id
        buf.extend_from_slice(&self.system_id);

        // key ID list (version 1 only)
        if self.version >= 1 {
            buf.extend_from_slice(&(self.key_ids.len() as u32).to_be_bytes());
            for kid in &self.key_ids {
                buf.extend_from_slice(kid);
            }
        }

        // data_size + data
        buf.extend_from_slice(&(self.data.len() as u32).to_be_bytes());
        buf.extend_from_slice(&self.data);

        buf
    }

    /// Parse a PSSH box from its binary representation.
    ///
    /// The slice must start at the beginning of the box (i.e. the 4-byte size
    /// field).
    pub fn decode(input: &[u8]) -> PackagerResult<Self> {
        if input.len() < 32 {
            return Err(PackagerError::DrmFailed(
                "PSSH box too short (need at least 32 bytes)".to_string(),
            ));
        }

        let declared_size = u32::from_be_bytes(
            input[0..4]
                .try_into()
                .map_err(|_| PackagerError::DrmFailed("size read failed".to_string()))?,
        ) as usize;

        if input.len() < declared_size {
            return Err(PackagerError::DrmFailed(format!(
                "Buffer too short: need {declared_size}, have {}",
                input.len()
            )));
        }

        if &input[4..8] != b"pssh" {
            return Err(PackagerError::DrmFailed(
                "Not a pssh box (wrong fourcc)".to_string(),
            ));
        }

        let version = input[8];
        // bytes 9-11 are flags (ignored)

        let mut system_id = [0u8; 16];
        system_id.copy_from_slice(&input[12..28]);

        let mut cursor = 28usize;

        // Version 1: read key IDs
        let mut key_ids: Vec<[u8; 16]> = Vec::new();
        if version >= 1 {
            if cursor + 4 > declared_size {
                return Err(PackagerError::DrmFailed(
                    "Truncated key_id count".to_string(),
                ));
            }
            let kid_count = u32::from_be_bytes(
                input[cursor..cursor + 4]
                    .try_into()
                    .map_err(|_| PackagerError::DrmFailed("kid count read failed".to_string()))?,
            ) as usize;
            cursor += 4;

            for _ in 0..kid_count {
                if cursor + 16 > declared_size {
                    return Err(PackagerError::DrmFailed("Truncated key ID".to_string()));
                }
                let mut kid = [0u8; 16];
                kid.copy_from_slice(&input[cursor..cursor + 16]);
                key_ids.push(kid);
                cursor += 16;
            }
        }

        // data_size
        if cursor + 4 > declared_size {
            return Err(PackagerError::DrmFailed("Truncated data_size".to_string()));
        }
        let data_size = u32::from_be_bytes(
            input[cursor..cursor + 4]
                .try_into()
                .map_err(|_| PackagerError::DrmFailed("data_size read failed".to_string()))?,
        ) as usize;
        cursor += 4;

        if cursor + data_size > declared_size {
            return Err(PackagerError::DrmFailed(format!(
                "Declared data_size {data_size} overflows box boundary"
            )));
        }

        let data = input[cursor..cursor + data_size].to_vec();

        Ok(Self {
            system_id,
            version,
            key_ids,
            data,
        })
    }
}

// ---------------------------------------------------------------------------
// Widevine PSSH builder
// ---------------------------------------------------------------------------

/// Encode a length-prefixed protobuf TLV field:  `[tag][varint-len][data]`.
///
/// Only single-byte tags and lengths < 128 are supported (sufficient for our
/// PSSH payloads).
fn proto_field(tag: u8, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + data.len());
    out.push(tag);
    // Minimal varint encoding for lengths < 128
    if data.len() < 128 {
        out.push(data.len() as u8);
    } else {
        // Two-byte varint (handles up to 16383 bytes)
        let len = data.len();
        out.push(((len & 0x7F) as u8) | 0x80);
        out.push(((len >> 7) & 0x7F) as u8);
    }
    out.extend_from_slice(data);
    out
}

/// Build a minimal Widevine PSSH box.
///
/// The payload is a hand-encoded Widevine `WidevineCencHeader` protobuf:
/// - field 2 (`key_ids`, wire-type 2): the 16-byte key ID
/// - field 4 (`content_id`, wire-type 2): the content ID bytes
///
/// Tags:  field_number << 3 | wire_type.  Wire-type 2 = length-delimited.
#[must_use]
pub fn build_widevine_pssh(key_id: &[u8; 16], content_id: &[u8]) -> PsshBox {
    // field 2, wire-type 2  =>  tag = (2 << 3) | 2 = 0x12
    let key_id_field = proto_field(0x12, key_id);
    // field 4, wire-type 2  =>  tag = (4 << 3) | 2 = 0x22
    let content_id_field = proto_field(0x22, content_id);

    let mut payload = Vec::with_capacity(key_id_field.len() + content_id_field.len());
    payload.extend_from_slice(&key_id_field);
    if !content_id.is_empty() {
        payload.extend_from_slice(&content_id_field);
    }

    PsshBox::new_v0(WIDEVINE_SYSTEM_ID, payload)
}

// ---------------------------------------------------------------------------
// PlayReady PSSH builder
// ---------------------------------------------------------------------------

/// Build a minimal PlayReady PSSH box.
///
/// The payload is a PlayReady Object (PRO) containing a single PlayReady
/// Record (type 1) whose value is a UTF-16LE-encoded `<WRMHEADER>` XML string.
#[must_use]
pub fn build_playready_pssh(key_id: &[u8; 16]) -> PsshBox {
    let key_id_b64 = BASE64.encode(key_id);

    // Build the WRM XML header (UTF-8 first, then re-encode as UTF-16LE below)
    let xml = format!(
        "<WRMHEADER xmlns=\"http://schemas.microsoft.com/DRM/2007/03/PlayReadyHeader\" \
         version=\"4.0.0.0\">\
         <DATA>\
         <PROTECTINFO>\
         <KEYLEN>16</KEYLEN>\
         </PROTECTINFO>\
         <KID>{key_id_b64}</KID>\
         </DATA>\
         </WRMHEADER>"
    );

    // Encode as UTF-16LE (PlayReady requirement)
    let xml_utf16: Vec<u8> = xml.encode_utf16().flat_map(|c| c.to_le_bytes()).collect();

    // PlayReady Record:
    //   2 bytes  record_type   (little-endian u16, 1 = rights management header)
    //   2 bytes  record_length (little-endian u16)
    //   N bytes  record_value  (UTF-16LE XML)
    let record_type: u16 = 1;
    let record_len = xml_utf16.len() as u16;
    let mut record: Vec<u8> = Vec::with_capacity(4 + xml_utf16.len());
    record.extend_from_slice(&record_type.to_le_bytes());
    record.extend_from_slice(&record_len.to_le_bytes());
    record.extend_from_slice(&xml_utf16);

    // PlayReady Object (PRO):
    //   4 bytes  pro_length       (little-endian u32, total PRO size including this field)
    //   2 bytes  record_count     (little-endian u16)
    //   N bytes  records
    let pro_length = (4 + 2 + record.len()) as u32;
    let record_count: u16 = 1;
    let mut pro: Vec<u8> = Vec::with_capacity(pro_length as usize);
    pro.extend_from_slice(&pro_length.to_le_bytes());
    pro.extend_from_slice(&record_count.to_le_bytes());
    pro.extend_from_slice(&record);

    PsshBox::new_v0(PLAYREADY_SYSTEM_ID, pro)
}

// ---------------------------------------------------------------------------
// FairPlay PSSH builder
// ---------------------------------------------------------------------------

/// Build a minimal FairPlay Streaming PSSH box.
///
/// FairPlay uses version 1 boxes with the key ID in the key-ID list.
/// The data payload contains the URI to the key server (as UTF-8 bytes).
#[must_use]
pub fn build_fairplay_pssh(key_id: &[u8; 16], server_uri: &str) -> PsshBox {
    PsshBox::new_v1(
        FAIRPLAY_SYSTEM_ID,
        vec![*key_id],
        server_uri.as_bytes().to_vec(),
    )
}

// ---------------------------------------------------------------------------
// CENC (Common Encryption) PSSH builder
// ---------------------------------------------------------------------------

/// Build a Common Encryption PSSH box (version 1) with multiple key IDs.
///
/// CENC PSSH boxes typically carry key IDs in the v1 key-ID list with an
/// empty data payload.  This is used for multi-key scenarios where each
/// track or quality level is encrypted with a different key.
#[must_use]
pub fn build_cenc_pssh(key_ids: &[[u8; 16]]) -> PsshBox {
    PsshBox::new_v1(
        DrmSystem::CommonEncryption.system_id(),
        key_ids.to_vec(),
        Vec::new(),
    )
}

// ---------------------------------------------------------------------------
// DrmSystem helpers
// ---------------------------------------------------------------------------

impl DrmSystem {
    /// Attempt to identify the DRM system from a 16-byte system ID.
    ///
    /// Returns `None` if the system ID is not recognised.
    #[must_use]
    pub fn from_system_id(id: &[u8; 16]) -> Option<Self> {
        if id == &WIDEVINE_SYSTEM_ID {
            Some(Self::Widevine)
        } else if id == &PLAYREADY_SYSTEM_ID {
            Some(Self::PlayReady)
        } else if id == &FAIRPLAY_SYSTEM_ID {
            Some(Self::FairPlay)
        } else {
            let marlin_id = Self::Marlin.system_id();
            let cenc_id = Self::CommonEncryption.system_id();
            if id == &marlin_id {
                Some(Self::Marlin)
            } else if id == &cenc_id {
                Some(Self::CommonEncryption)
            } else {
                None
            }
        }
    }

    /// Return a human-readable label for the DRM system.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Widevine => "Widevine",
            Self::PlayReady => "PlayReady",
            Self::FairPlay => "FairPlay",
            Self::Marlin => "Marlin",
            Self::CommonEncryption => "CENC",
        }
    }

    /// Return the system ID formatted as a UUID string (lowercase hex with dashes).
    #[must_use]
    pub fn uuid_string(&self) -> String {
        format_uuid(&self.system_id())
    }
}

impl std::fmt::Display for DrmSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// PsshBox helpers
// ---------------------------------------------------------------------------

impl PsshBox {
    /// Identify the DRM system of this box (if recognised).
    #[must_use]
    pub fn drm_system(&self) -> Option<DrmSystem> {
        DrmSystem::from_system_id(&self.system_id)
    }

    /// Encode the box and return it as a base64 string.
    ///
    /// Useful for embedding in DASH MPD `<cenc:pssh>` elements.
    #[must_use]
    pub fn to_base64(&self) -> String {
        BASE64.encode(self.encode())
    }

    /// Return the system ID as a lowercase hex string (no dashes).
    #[must_use]
    pub fn system_id_hex(&self) -> String {
        hex::encode(self.system_id)
    }

    /// Return the system ID formatted as a UUID string.
    #[must_use]
    pub fn system_id_uuid(&self) -> String {
        format_uuid(&self.system_id)
    }

    /// Return the total number of key IDs in this box.
    #[must_use]
    pub fn key_id_count(&self) -> usize {
        self.key_ids.len()
    }

    /// Scan a byte slice for all PSSH boxes and return them.
    ///
    /// This is useful for extracting PSSH boxes from an init segment that may
    /// contain multiple concatenated boxes.
    pub fn scan_all(data: &[u8]) -> Vec<PackagerResult<Self>> {
        let mut results = Vec::new();
        let mut offset = 0usize;

        while offset + 8 <= data.len() {
            let size_bytes: [u8; 4] = match data[offset..offset + 4].try_into() {
                Ok(b) => b,
                Err(_) => break,
            };
            let box_size = u32::from_be_bytes(size_bytes) as usize;
            if box_size < 8 || offset + box_size > data.len() {
                break;
            }
            if &data[offset + 4..offset + 8] == b"pssh" {
                results.push(Self::decode(&data[offset..offset + box_size]));
            }
            offset += box_size;
        }

        results
    }
}

// ---------------------------------------------------------------------------
// UUID formatting
// ---------------------------------------------------------------------------

/// Format 16 bytes as a UUID string: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
fn format_uuid(bytes: &[u8; 16]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- DrmSystem -----------------------------------------------------------

    #[test]
    fn test_drm_system_widevine_id() {
        let id = DrmSystem::Widevine.system_id();
        assert_eq!(&id, &WIDEVINE_SYSTEM_ID);
    }

    #[test]
    fn test_drm_system_playready_id() {
        let id = DrmSystem::PlayReady.system_id();
        assert_eq!(&id, &PLAYREADY_SYSTEM_ID);
    }

    #[test]
    fn test_drm_system_fairplay_id() {
        let id = DrmSystem::FairPlay.system_id();
        assert_eq!(&id, &FAIRPLAY_SYSTEM_ID);
    }

    #[test]
    fn test_drm_system_marlin_id_length() {
        let id = DrmSystem::Marlin.system_id();
        assert_eq!(id.len(), 16);
    }

    #[test]
    fn test_drm_system_common_enc_id_length() {
        let id = DrmSystem::CommonEncryption.system_id();
        assert_eq!(id.len(), 16);
    }

    // --- PsshBox v0 encode/decode round-trip ---------------------------------

    #[test]
    fn test_pssh_v0_encode_roundtrip() {
        let data = b"widevine-payload".to_vec();
        let pssh = PsshBox::new_v0(WIDEVINE_SYSTEM_ID, data.clone());
        let encoded = pssh.encode();
        let decoded = PsshBox::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.system_id, WIDEVINE_SYSTEM_ID);
        assert_eq!(decoded.version, 0);
        assert_eq!(decoded.data, data);
        assert!(decoded.key_ids.is_empty());
    }

    #[test]
    fn test_pssh_v0_encode_fourcc() {
        let pssh = PsshBox::new_v0(WIDEVINE_SYSTEM_ID, vec![]);
        let encoded = pssh.encode();
        assert_eq!(&encoded[4..8], b"pssh");
    }

    #[test]
    fn test_pssh_v0_encode_size_field() {
        let data = vec![0xAB; 32];
        let pssh = PsshBox::new_v0(WIDEVINE_SYSTEM_ID, data);
        let encoded = pssh.encode();
        let declared = u32::from_be_bytes(encoded[0..4].try_into().expect("4 bytes")) as usize;
        assert_eq!(declared, encoded.len());
    }

    #[test]
    fn test_pssh_v0_encode_version_byte() {
        let pssh = PsshBox::new_v0(WIDEVINE_SYSTEM_ID, vec![]);
        let encoded = pssh.encode();
        assert_eq!(encoded[8], 0); // version byte
        assert_eq!(&encoded[9..12], &[0u8; 3]); // flags
    }

    // --- PsshBox v1 encode/decode round-trip ---------------------------------

    #[test]
    fn test_pssh_v1_encode_roundtrip() {
        let key_id: [u8; 16] = [0x01; 16];
        let pssh = PsshBox::new_v1(WIDEVINE_SYSTEM_ID, vec![key_id], vec![0xDE, 0xAD]);
        let encoded = pssh.encode();
        let decoded = PsshBox::decode(&encoded).expect("decode v1 should succeed");
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.key_ids.len(), 1);
        assert_eq!(decoded.key_ids[0], key_id);
        assert_eq!(decoded.data, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_pssh_v1_multiple_key_ids() {
        let key_ids: Vec<[u8; 16]> = (0..3).map(|i| [i as u8; 16]).collect();
        let pssh = PsshBox::new_v1(PLAYREADY_SYSTEM_ID, key_ids.clone(), vec![]);
        let encoded = pssh.encode();
        let decoded = PsshBox::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.key_ids, key_ids);
    }

    // --- decode error cases --------------------------------------------------

    #[test]
    fn test_pssh_decode_too_short() {
        let result = PsshBox::decode(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pssh_decode_wrong_fourcc() {
        let mut buf = vec![0u8; 36];
        buf[0..4].copy_from_slice(&36u32.to_be_bytes());
        buf[4..8].copy_from_slice(b"moof"); // wrong
        let result = PsshBox::decode(&buf);
        assert!(result.is_err());
    }

    // --- Widevine builder ----------------------------------------------------

    #[test]
    fn test_build_widevine_pssh_system_id() {
        let key_id = [0xAA; 16];
        let pssh = build_widevine_pssh(&key_id, b"content-001");
        assert_eq!(pssh.system_id, WIDEVINE_SYSTEM_ID);
    }

    #[test]
    fn test_build_widevine_pssh_payload_non_empty() {
        let key_id = [0x11; 16];
        let pssh = build_widevine_pssh(&key_id, b"cid");
        assert!(!pssh.data.is_empty());
    }

    #[test]
    fn test_build_widevine_pssh_encodes_key_id_tag() {
        let key_id = [0x22; 16];
        let pssh = build_widevine_pssh(&key_id, b"");
        // First byte of data should be the key_ids field tag 0x12
        assert_eq!(pssh.data[0], 0x12);
    }

    #[test]
    fn test_build_widevine_pssh_encodes_content_id_tag() {
        let key_id = [0x33; 16];
        let pssh = build_widevine_pssh(&key_id, b"abc");
        // Should contain 0x22 tag for content_id
        assert!(pssh.data.contains(&0x22));
    }

    #[test]
    fn test_build_widevine_pssh_empty_content_id() {
        let key_id = [0x44; 16];
        let pssh = build_widevine_pssh(&key_id, b"");
        // Without content_id, the 0x22 tag should NOT appear
        assert!(!pssh.data.contains(&0x22));
    }

    #[test]
    fn test_build_widevine_pssh_roundtrip_encode() {
        let key_id = [0x55; 16];
        let pssh = build_widevine_pssh(&key_id, b"my-content");
        let encoded = pssh.encode();
        let decoded = PsshBox::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.system_id, WIDEVINE_SYSTEM_ID);
        assert_eq!(decoded.data, pssh.data);
    }

    // --- PlayReady builder ---------------------------------------------------

    #[test]
    fn test_build_playready_pssh_system_id() {
        let key_id = [0xBB; 16];
        let pssh = build_playready_pssh(&key_id);
        assert_eq!(pssh.system_id, PLAYREADY_SYSTEM_ID);
    }

    #[test]
    fn test_build_playready_pssh_pro_structure() {
        let key_id = [0xCC; 16];
        let pssh = build_playready_pssh(&key_id);
        // PRO starts with 4-byte length (LE u32)
        assert!(pssh.data.len() >= 6);
        let pro_len = u32::from_le_bytes(pssh.data[0..4].try_into().expect("4 bytes")) as usize;
        assert_eq!(pro_len, pssh.data.len());
    }

    #[test]
    fn test_build_playready_pssh_record_type() {
        let key_id = [0xDD; 16];
        let pssh = build_playready_pssh(&key_id);
        // After 4-byte PRO length + 2-byte record_count (=1), record starts at offset 6
        // record_type should be 1 (LE u16)
        let record_type = u16::from_le_bytes(pssh.data[6..8].try_into().expect("2 bytes"));
        assert_eq!(record_type, 1);
    }

    #[test]
    fn test_build_playready_pssh_contains_key_id_in_xml() {
        let key_id = [0xEE; 16];
        let pssh = build_playready_pssh(&key_id);
        // The record payload is UTF-16LE XML; extract record_length from offset 8
        let rec_len = u16::from_le_bytes(pssh.data[8..10].try_into().expect("2 bytes")) as usize;
        let xml_bytes = &pssh.data[10..10 + rec_len];
        // Convert UTF-16LE back to a string for assertion
        let xml_u16: Vec<u16> = xml_bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let xml = String::from_utf16_lossy(&xml_u16);
        assert!(xml.contains("WRMHEADER"));
        assert!(xml.contains("KID"));
    }

    #[test]
    fn test_build_playready_pssh_roundtrip_encode() {
        let key_id = [0xFF; 16];
        let pssh = build_playready_pssh(&key_id);
        let encoded = pssh.encode();
        let decoded = PsshBox::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.system_id, PLAYREADY_SYSTEM_ID);
        assert_eq!(decoded.data, pssh.data);
    }

    #[test]
    fn test_pssh_encode_contains_system_id_bytes() {
        let pssh = PsshBox::new_v0(FAIRPLAY_SYSTEM_ID, vec![]);
        let encoded = pssh.encode();
        assert_eq!(&encoded[12..28], &FAIRPLAY_SYSTEM_ID);
    }

    // --- FairPlay builder ---------------------------------------------------

    #[test]
    fn test_build_fairplay_pssh_system_id() {
        let key_id = [0xAA; 16];
        let pssh = build_fairplay_pssh(&key_id, "skd://example.com/key");
        assert_eq!(pssh.system_id, FAIRPLAY_SYSTEM_ID);
    }

    #[test]
    fn test_build_fairplay_pssh_version() {
        let key_id = [0xBB; 16];
        let pssh = build_fairplay_pssh(&key_id, "skd://example.com/key");
        assert_eq!(pssh.version, 1);
    }

    #[test]
    fn test_build_fairplay_pssh_key_id_present() {
        let key_id = [0xCC; 16];
        let pssh = build_fairplay_pssh(&key_id, "skd://example.com/key");
        assert_eq!(pssh.key_ids.len(), 1);
        assert_eq!(pssh.key_ids[0], key_id);
    }

    #[test]
    fn test_build_fairplay_pssh_roundtrip() {
        let key_id = [0xDD; 16];
        let pssh = build_fairplay_pssh(&key_id, "skd://example.com/key");
        let encoded = pssh.encode();
        let decoded = PsshBox::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.system_id, FAIRPLAY_SYSTEM_ID);
        assert_eq!(decoded.key_ids.len(), 1);
    }

    // --- CENC builder -------------------------------------------------------

    #[test]
    fn test_build_cenc_pssh_system_id() {
        let key_ids = vec![[0x11; 16], [0x22; 16]];
        let pssh = build_cenc_pssh(&key_ids);
        assert_eq!(pssh.system_id, DrmSystem::CommonEncryption.system_id());
    }

    #[test]
    fn test_build_cenc_pssh_version_1() {
        let key_ids = vec![[0x33; 16]];
        let pssh = build_cenc_pssh(&key_ids);
        assert_eq!(pssh.version, 1);
    }

    #[test]
    fn test_build_cenc_pssh_multiple_key_ids() {
        let key_ids = vec![[0x01; 16], [0x02; 16], [0x03; 16]];
        let pssh = build_cenc_pssh(&key_ids);
        assert_eq!(pssh.key_ids.len(), 3);
    }

    #[test]
    fn test_build_cenc_pssh_roundtrip() {
        let key_ids = vec![[0xAA; 16], [0xBB; 16]];
        let pssh = build_cenc_pssh(&key_ids);
        let encoded = pssh.encode();
        let decoded = PsshBox::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.key_ids, key_ids);
        assert!(decoded.data.is_empty());
    }

    // --- DRM system from_system_id ------------------------------------------

    #[test]
    fn test_drm_system_from_widevine_id() {
        let sys = DrmSystem::from_system_id(&WIDEVINE_SYSTEM_ID);
        assert_eq!(sys, Some(DrmSystem::Widevine));
    }

    #[test]
    fn test_drm_system_from_playready_id() {
        let sys = DrmSystem::from_system_id(&PLAYREADY_SYSTEM_ID);
        assert_eq!(sys, Some(DrmSystem::PlayReady));
    }

    #[test]
    fn test_drm_system_from_fairplay_id() {
        let sys = DrmSystem::from_system_id(&FAIRPLAY_SYSTEM_ID);
        assert_eq!(sys, Some(DrmSystem::FairPlay));
    }

    #[test]
    fn test_drm_system_from_unknown_id() {
        let sys = DrmSystem::from_system_id(&[0u8; 16]);
        assert!(sys.is_none());
    }

    // --- PsshBox helpers ----------------------------------------------------

    #[test]
    fn test_pssh_drm_system_accessor() {
        let pssh = PsshBox::new_v0(WIDEVINE_SYSTEM_ID, vec![]);
        assert_eq!(pssh.drm_system(), Some(DrmSystem::Widevine));
    }

    #[test]
    fn test_pssh_to_base64() {
        let pssh = PsshBox::new_v0(WIDEVINE_SYSTEM_ID, vec![0xAA]);
        let b64 = pssh.to_base64();
        assert!(!b64.is_empty());
        // Verify round-trip through base64
        let decoded_bytes = BASE64.decode(&b64).expect("base64 decode");
        let decoded_pssh = PsshBox::decode(&decoded_bytes).expect("pssh decode");
        assert_eq!(decoded_pssh.data, vec![0xAA]);
    }

    #[test]
    fn test_pssh_system_id_hex() {
        let pssh = PsshBox::new_v0(WIDEVINE_SYSTEM_ID, vec![]);
        let hex_str = pssh.system_id_hex();
        assert_eq!(hex_str, "edef8ba979d64acea3c827dcd51d21ed");
    }

    // --- proto_field edge cases ---------------------------------------------

    #[test]
    fn test_proto_field_large_data() {
        let data = vec![0xAB; 200]; // > 128 bytes triggers two-byte varint
        let field = proto_field(0x12, &data);
        assert_eq!(field[0], 0x12); // tag
                                    // Two-byte varint for 200: (200 & 0x7F) | 0x80 = 0xC8, (200 >> 7) = 1
        assert_eq!(field[1], 0xC8);
        assert_eq!(field[2], 0x01);
        assert_eq!(field.len(), 3 + 200);
    }

    #[test]
    fn test_decode_truncated_v1_key_ids() {
        // Build a v1 box that claims 100 key IDs but has no room
        let mut buf = vec![0u8; 40];
        let size: u32 = 40;
        buf[0..4].copy_from_slice(&size.to_be_bytes());
        buf[4..8].copy_from_slice(b"pssh");
        buf[8] = 1; // version 1
                    // system_id at 12..28 (zeros)
                    // key_id_count at 28..32 = 100
        buf[28..32].copy_from_slice(&100u32.to_be_bytes());
        let result = PsshBox::decode(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_truncated_data_size() {
        // v0 box that's too short for data_size field
        let mut buf = vec![0u8; 32];
        let size: u32 = 30; // declared size less than minimum needed
        buf[0..4].copy_from_slice(&size.to_be_bytes());
        buf[4..8].copy_from_slice(b"pssh");
        buf[8] = 0; // version 0
        let result = PsshBox::decode(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_data_overflow() {
        // v0 box where data_size extends past box boundary
        let mut buf = vec![0u8; 36];
        let size: u32 = 36;
        buf[0..4].copy_from_slice(&size.to_be_bytes());
        buf[4..8].copy_from_slice(b"pssh");
        buf[8] = 0; // version 0
                    // data_size at offset 28 = 100 (overflows)
        buf[28..32].copy_from_slice(&100u32.to_be_bytes());
        let result = PsshBox::decode(&buf);
        assert!(result.is_err());
    }
}

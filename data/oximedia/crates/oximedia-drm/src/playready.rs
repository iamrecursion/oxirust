//! PlayReady DRM implementation
//!
//! Supports Microsoft PlayReady DRM system.
//! Note: This is a partial implementation for educational purposes.

use crate::{DrmError, DrmSystem, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Cursor;
use uuid::Uuid;

// ──────────────────────────────────────────────────────────────────────────────
// PlayReady System ID constant (9a04f079-9840-4286-ab92-e65be0885f95)
// ──────────────────────────────────────────────────────────────────────────────

/// PlayReady DRM System ID as a 16-byte array (MPEG CENC / ISOBMFF PSSH).
///
/// UUID: `9a04f079-9840-4286-ab92-e65be0885f95`
pub const PLAYREADY_SYSTEM_ID: [u8; 16] = [
    0x9a, 0x04, 0xf0, 0x79, 0x98, 0x40, 0x42, 0x86, 0xab, 0x92, 0xe6, 0x5b, 0xe0, 0x88, 0x5f, 0x95,
];

// ──────────────────────────────────────────────────────────────────────────────
// PlayReadySimpleVersion — simplified 3-variant enum for PlayReadyHeaderObject
// ──────────────────────────────────────────────────────────────────────────────

/// Simplified PlayReady specification version selector used by
/// [`PlayReadyHeaderObject`].
///
/// Corresponds to the three generations of the PlayReady specification that
/// carry distinct header XML structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayReadySpecVersion {
    /// PlayReady 2.0 — original WRMHEADER v2 schema.
    V2_0,
    /// PlayReady 3.0 — v3 schema with enhanced policy fields.
    V3_0,
    /// PlayReady 4.0 — v4 schema, current generation.
    V4_0,
}

impl PlayReadySpecVersion {
    /// Map to the canonical WRMHEADER `version` attribute string.
    pub fn header_version_str(self) -> &'static str {
        match self {
            PlayReadySpecVersion::V2_0 => "4.0.0.0",
            PlayReadySpecVersion::V3_0 => "4.2.0.0",
            PlayReadySpecVersion::V4_0 => "4.3.0.0",
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PlayReadyHeaderObject — simplified API matching the DASH/HLS packaging spec
// ──────────────────────────────────────────────────────────────────────────────

/// High-level representation of a **PlayReady Header Object** (PRO), as
/// described in the Microsoft PlayReady Header Specification.
///
/// This type provides a cleaner, simpler surface than [`PlayReadyHeader`] and
/// uses raw `Vec<u8>` key IDs instead of [`Uuid`] to align with common
/// multi-DRM packaging workflows.
#[derive(Debug, Clone)]
pub struct PlayReadyHeaderObject {
    /// PlayReady specification version for this header.
    pub version: PlayReadySpecVersion,
    /// Raw 16-byte key identifier.
    pub key_id: Vec<u8>,
    /// License Acquisition URL (`LA_URL`).
    pub la_url: Option<String>,
    /// License UI URL (`LUI_URL`).
    pub lui_url: Option<String>,
    /// Domain service identifier (`DS_ID`).
    pub ds_id: Option<String>,
}

impl PlayReadyHeaderObject {
    /// Create a new `PlayReadyHeaderObject` using PlayReady 4.0 by default.
    ///
    /// Returns an error when `key_id` is not exactly 16 bytes.
    pub fn new(key_id: Vec<u8>, la_url: Option<String>) -> Result<Self> {
        if key_id.len() != 16 {
            return Err(DrmError::InvalidKey(format!(
                "PlayReady key_id must be 16 bytes, got {}",
                key_id.len()
            )));
        }
        Ok(Self {
            version: PlayReadySpecVersion::V4_0,
            key_id,
            la_url,
            lui_url: None,
            ds_id: None,
        })
    }

    /// Override the specification version.
    pub fn with_version(mut self, version: PlayReadySpecVersion) -> Self {
        self.version = version;
        self
    }

    /// Set the License UI URL.
    pub fn with_lui_url(mut self, lui_url: String) -> Self {
        self.lui_url = Some(lui_url);
        self
    }

    /// Set the Domain Service ID.
    pub fn with_ds_id(mut self, ds_id: String) -> Self {
        self.ds_id = Some(ds_id);
        self
    }

    /// Serialise to a PlayReady WRMHEADER XML string.
    ///
    /// The output follows the `WRMHEADER` schema used by DASH/HLS packagers.
    pub fn to_xml(&self) -> Result<String> {
        let mut writer = Writer::new(Cursor::new(Vec::new()));

        writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("utf-8"), None)))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        let mut wrmheader = BytesStart::new("WRMHEADER");
        wrmheader.push_attribute((
            "xmlns",
            "http://schemas.microsoft.com/DRM/2007/03/PlayReadyHeader",
        ));
        wrmheader.push_attribute(("version", self.version.header_version_str()));
        writer
            .write_event(Event::Start(wrmheader))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        writer
            .write_event(Event::Start(BytesStart::new("DATA")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // <PROTECTINFO>
        writer
            .write_event(Event::Start(BytesStart::new("PROTECTINFO")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::Start(BytesStart::new("ALGID")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::Text(BytesText::new("AESCTR")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("ALGID")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("PROTECTINFO")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // <KID> — base64-encoded raw 16-byte key ID
        writer
            .write_event(Event::Start(BytesStart::new("KID")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        let kid_b64 = STANDARD.encode(&self.key_id);
        writer
            .write_event(Event::Text(BytesText::new(&kid_b64)))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("KID")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        if let Some(ref la_url) = self.la_url {
            writer
                .write_event(Event::Start(BytesStart::new("LA_URL")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::Text(BytesText::new(la_url)))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::End(BytesEnd::new("LA_URL")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
        }

        if let Some(ref lui_url) = self.lui_url {
            writer
                .write_event(Event::Start(BytesStart::new("LUI_URL")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::Text(BytesText::new(lui_url)))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::End(BytesEnd::new("LUI_URL")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
        }

        if let Some(ref ds_id) = self.ds_id {
            writer
                .write_event(Event::Start(BytesStart::new("DS_ID")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::Text(BytesText::new(ds_id)))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::End(BytesEnd::new("DS_ID")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("DATA")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("WRMHEADER")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        let bytes = writer.into_inner().into_inner();
        String::from_utf8(bytes).map_err(|e| DrmError::XmlError(e.to_string()))
    }

    /// Parse a `PlayReadyHeaderObject` from a WRMHEADER XML string.
    ///
    /// Returns [`DrmError::XmlError`] on structural problems and
    /// [`DrmError::InvalidKey`] when the decoded `KID` is not 16 bytes.
    pub fn parse_xml(xml: &str) -> Result<Self> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut key_id: Option<Vec<u8>> = None;
        let mut la_url: Option<String> = None;
        let mut lui_url: Option<String> = None;
        let mut ds_id: Option<String> = None;
        let mut spec_version = PlayReadySpecVersion::V4_0;
        let mut current_element = String::new();
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    current_element = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if current_element == "WRMHEADER" {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"version" {
                                let v = String::from_utf8_lossy(&attr.value).to_string();
                                spec_version = match v.as_str() {
                                    "4.0.0.0" | "4.1.0.0" => PlayReadySpecVersion::V2_0,
                                    "4.2.0.0" => PlayReadySpecVersion::V3_0,
                                    _ => PlayReadySpecVersion::V4_0,
                                };
                            }
                        }
                    }
                }
                Ok(Event::Text(e)) => {
                    let text = String::from_utf8_lossy(e.as_ref());
                    match current_element.as_str() {
                        "KID" => {
                            let decoded = STANDARD
                                .decode(text.trim())
                                .map_err(DrmError::Base64Error)?;
                            if decoded.len() != 16 {
                                return Err(DrmError::InvalidKey(format!(
                                    "decoded KID must be 16 bytes, got {}",
                                    decoded.len()
                                )));
                            }
                            key_id = Some(decoded);
                        }
                        "LA_URL" => {
                            la_url = Some(text.to_string());
                        }
                        "LUI_URL" => {
                            lui_url = Some(text.to_string());
                        }
                        "DS_ID" => {
                            ds_id = Some(text.to_string());
                        }
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(DrmError::XmlError(e.to_string())),
                _ => {}
            }
            buf.clear();
        }

        let key_id = key_id.ok_or_else(|| {
            DrmError::XmlError("WRMHEADER is missing a <KID> element".to_string())
        })?;

        Ok(Self {
            version: spec_version,
            key_id,
            la_url,
            lui_url,
            ds_id,
        })
    }

    /// Wrap the serialised XML inside a full ISO-BMFF PSSH box for inclusion
    /// in DASH/CMAF/HLS streams.
    ///
    /// Layout:
    /// ```text
    /// 4 bytes  : box size (big-endian)
    /// 4 bytes  : 'pssh'
    /// 1 byte   : version = 0
    /// 3 bytes  : flags = 0
    /// 16 bytes : PLAYREADY_SYSTEM_ID
    /// 4 bytes  : data length
    /// N bytes  : PlayReady Header Object binary blob
    ///              (2-byte record count LE + per-record 2-byte type LE +
    ///               2-byte len LE + UTF-16LE XML)
    /// ```
    pub fn to_pssh_box(&self) -> Result<Vec<u8>> {
        let xml = self.to_xml()?;

        // Encode XML as UTF-16LE
        let utf16_bytes = encode_utf16_le(&xml);

        // PlayReady Header Object binary format:
        //   record_count       : u16 LE  (always 1)
        //   record[0].type     : u16 LE  (0x0001 = Rights Management Header)
        //   record[0].length   : u16 LE
        //   record[0].value    : utf16_bytes
        let record_type: u16 = 0x0001;
        let record_len: u16 = utf16_bytes.len() as u16;
        // Header Object size = 4 (object_size u32) + 2 (record_count) +
        //                      2 (type) + 2 (len) + utf16_bytes.len()
        let object_size: u32 = 4 + 2 + 2 + 2 + utf16_bytes.len() as u32;

        let mut pro = Vec::with_capacity(object_size as usize);
        pro.extend_from_slice(&object_size.to_le_bytes());
        pro.extend_from_slice(&1u16.to_le_bytes()); // record_count
        pro.extend_from_slice(&record_type.to_le_bytes());
        pro.extend_from_slice(&record_len.to_le_bytes());
        pro.extend_from_slice(&utf16_bytes);

        let total_size: u32 = (4 + 4 + 1 + 3 + 16 + 4 + pro.len()) as u32;
        let mut box_bytes = Vec::with_capacity(total_size as usize);
        box_bytes.extend_from_slice(&total_size.to_be_bytes());
        box_bytes.extend_from_slice(b"pssh");
        box_bytes.push(0u8);
        box_bytes.extend_from_slice(&[0u8; 3]);
        box_bytes.extend_from_slice(&PLAYREADY_SYSTEM_ID);
        box_bytes.extend_from_slice(&(pro.len() as u32).to_be_bytes());
        box_bytes.extend_from_slice(&pro);
        Ok(box_bytes)
    }
}

/// Encode a UTF-8 string to a UTF-16LE byte vector.
fn encode_utf16_le(s: &str) -> Vec<u8> {
    s.encode_utf16().flat_map(|c| c.to_le_bytes()).collect()
}

/// PlayReady header version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayReadyVersion {
    /// Version 4.0.0.0
    V4_0_0_0,
    /// Version 4.1.0.0
    V4_1_0_0,
    /// Version 4.2.0.0
    V4_2_0_0,
    /// Version 4.3.0.0
    V4_3_0_0,
}

impl PlayReadyVersion {
    /// Get version as string
    pub fn as_str(&self) -> &'static str {
        match self {
            PlayReadyVersion::V4_0_0_0 => "4.0.0.0",
            PlayReadyVersion::V4_1_0_0 => "4.1.0.0",
            PlayReadyVersion::V4_2_0_0 => "4.2.0.0",
            PlayReadyVersion::V4_3_0_0 => "4.3.0.0",
        }
    }
}

/// PlayReady header record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayReadyRecord {
    /// Record type
    pub record_type: String,
    /// Record value
    pub value: String,
}

impl PlayReadyRecord {
    /// Create a new PlayReady record
    pub fn new(record_type: String, value: String) -> Self {
        Self { record_type, value }
    }

    /// Create a rights management header record
    pub fn rights_management(content: String) -> Self {
        Self::new("RIGHTS_MANAGEMENT".to_string(), content)
    }
}

/// PlayReady header
#[derive(Debug, Clone)]
pub struct PlayReadyHeader {
    /// Version
    pub version: PlayReadyVersion,
    /// Key ID
    pub key_id: Uuid,
    /// License acquisition URL
    pub la_url: Option<String>,
    /// License UI URL
    pub lui_url: Option<String>,
    /// Domain service ID
    pub ds_id: Option<String>,
    /// Checksum (optional)
    pub checksum: Option<String>,
    /// Custom attributes
    pub custom_attributes: HashMap<String, String>,
}

impl PlayReadyHeader {
    /// Create a new PlayReady header
    pub fn new(key_id: Uuid) -> Self {
        Self {
            version: PlayReadyVersion::V4_3_0_0,
            key_id,
            la_url: None,
            lui_url: None,
            ds_id: None,
            checksum: None,
            custom_attributes: HashMap::new(),
        }
    }

    /// Set license acquisition URL
    pub fn with_la_url(mut self, url: String) -> Self {
        self.la_url = Some(url);
        self
    }

    /// Set license UI URL
    pub fn with_lui_url(mut self, url: String) -> Self {
        self.lui_url = Some(url);
        self
    }

    /// Set domain service ID
    pub fn with_ds_id(mut self, ds_id: String) -> Self {
        self.ds_id = Some(ds_id);
        self
    }

    /// Add custom attribute
    pub fn add_custom_attribute(&mut self, key: String, value: String) {
        self.custom_attributes.insert(key, value);
    }

    /// Generate PlayReady header XML
    pub fn to_xml(&self) -> Result<String> {
        let mut writer = Writer::new(Cursor::new(Vec::new()));

        // Write XML declaration
        writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("utf-8"), None)))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // <WRMHEADER>
        let mut wrmheader = BytesStart::new("WRMHEADER");
        wrmheader.push_attribute((
            "xmlns",
            "http://schemas.microsoft.com/DRM/2007/03/PlayReadyHeader",
        ));
        wrmheader.push_attribute(("version", self.version.as_str()));
        writer
            .write_event(Event::Start(wrmheader))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // <DATA>
        writer
            .write_event(Event::Start(BytesStart::new("DATA")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // <PROTECTINFO>
        writer
            .write_event(Event::Start(BytesStart::new("PROTECTINFO")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // <KEYLEN>
        writer
            .write_event(Event::Start(BytesStart::new("KEYLEN")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::Text(BytesText::new("16")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("KEYLEN")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // <ALGID>
        writer
            .write_event(Event::Start(BytesStart::new("ALGID")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::Text(BytesText::new("AESCTR")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("ALGID")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // </PROTECTINFO>
        writer
            .write_event(Event::End(BytesEnd::new("PROTECTINFO")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // <KID>
        writer
            .write_event(Event::Start(BytesStart::new("KID")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        let kid_b64 = STANDARD.encode(self.key_id.as_bytes());
        writer
            .write_event(Event::Text(BytesText::new(&kid_b64)))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("KID")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // <LA_URL> (if present)
        if let Some(ref la_url) = self.la_url {
            writer
                .write_event(Event::Start(BytesStart::new("LA_URL")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::Text(BytesText::new(la_url)))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::End(BytesEnd::new("LA_URL")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
        }

        // <LUI_URL> (if present)
        if let Some(ref lui_url) = self.lui_url {
            writer
                .write_event(Event::Start(BytesStart::new("LUI_URL")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::Text(BytesText::new(lui_url)))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::End(BytesEnd::new("LUI_URL")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
        }

        // <DS_ID> (if present)
        if let Some(ref ds_id) = self.ds_id {
            writer
                .write_event(Event::Start(BytesStart::new("DS_ID")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::Text(BytesText::new(ds_id)))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::End(BytesEnd::new("DS_ID")))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
        }

        // Custom attributes
        for (key, value) in &self.custom_attributes {
            writer
                .write_event(Event::Start(BytesStart::new(key.as_str())))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::Text(BytesText::new(value)))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
            writer
                .write_event(Event::End(BytesEnd::new(key.as_str())))
                .map_err(|e| DrmError::XmlError(e.to_string()))?;
        }

        // </DATA>
        writer
            .write_event(Event::End(BytesEnd::new("DATA")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        // </WRMHEADER>
        writer
            .write_event(Event::End(BytesEnd::new("WRMHEADER")))
            .map_err(|e| DrmError::XmlError(e.to_string()))?;

        let result = writer.into_inner().into_inner();
        String::from_utf8(result).map_err(|e| DrmError::XmlError(e.to_string()))
    }

    /// Parse PlayReady header from XML
    pub fn from_xml(xml: &str) -> Result<Self> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut key_id = None;
        let mut la_url = None;
        let mut lui_url = None;
        let mut ds_id = None;
        let mut version = PlayReadyVersion::V4_3_0_0;
        let mut current_element = String::new();
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    current_element = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    // Check for version in WRMHEADER
                    if current_element == "WRMHEADER" {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"version" {
                                let version_str = String::from_utf8_lossy(&attr.value).to_string();
                                version = match version_str.as_str() {
                                    "4.0.0.0" => PlayReadyVersion::V4_0_0_0,
                                    "4.1.0.0" => PlayReadyVersion::V4_1_0_0,
                                    "4.2.0.0" => PlayReadyVersion::V4_2_0_0,
                                    "4.3.0.0" => PlayReadyVersion::V4_3_0_0,
                                    _ => PlayReadyVersion::V4_3_0_0,
                                };
                            }
                        }
                    }
                }
                Ok(Event::Text(e)) => {
                    let text = String::from_utf8_lossy(e.as_ref());
                    match current_element.as_str() {
                        "KID" => {
                            let decoded = STANDARD
                                .decode(text.trim())
                                .map_err(|e| DrmError::Base64Error(e))?;
                            if decoded.len() == 16 {
                                let mut bytes = [0u8; 16];
                                bytes.copy_from_slice(&decoded);
                                key_id = Some(Uuid::from_bytes(bytes));
                            }
                        }
                        "LA_URL" => {
                            la_url = Some(text.to_string());
                        }
                        "LUI_URL" => {
                            lui_url = Some(text.to_string());
                        }
                        "DS_ID" => {
                            ds_id = Some(text.to_string());
                        }
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(DrmError::XmlError(e.to_string())),
                _ => {}
            }
            buf.clear();
        }

        let key_id =
            key_id.ok_or_else(|| DrmError::XmlError("Missing KID in header".to_string()))?;

        let mut header = PlayReadyHeader::new(key_id);
        header.version = version;
        header.la_url = la_url;
        header.lui_url = lui_url;
        header.ds_id = ds_id;

        Ok(header)
    }

    /// Get header as base64-encoded bytes
    pub fn to_base64(&self) -> Result<String> {
        let xml = self.to_xml()?;
        let utf16_bytes = encode_utf16(&xml);
        Ok(STANDARD.encode(&utf16_bytes))
    }
}

/// PlayReady PSSH data
#[derive(Debug, Clone)]
pub struct PlayReadyPsshData {
    /// PlayReady records
    pub records: Vec<PlayReadyRecord>,
}

impl PlayReadyPsshData {
    /// Create new PlayReady PSSH data
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Add a record
    pub fn add_record(&mut self, record: PlayReadyRecord) {
        self.records.push(record);
    }

    /// Add a PlayReady header as a record
    pub fn add_header(&mut self, header: &PlayReadyHeader) -> Result<()> {
        let xml = header.to_xml()?;
        let record = PlayReadyRecord::rights_management(xml);
        self.add_record(record);
        Ok(())
    }

    /// Serialize to bytes (simplified - would use proper binary format in production)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        // Simplified format: just concatenate records
        let mut result = Vec::new();

        // Write number of records
        result.extend_from_slice(&(self.records.len() as u32).to_le_bytes());

        for record in &self.records {
            // Write record type length and type
            let type_bytes = record.record_type.as_bytes();
            result.extend_from_slice(&(type_bytes.len() as u32).to_le_bytes());
            result.extend_from_slice(type_bytes);

            // Write value length and value
            let value_bytes = record.value.as_bytes();
            result.extend_from_slice(&(value_bytes.len() as u32).to_le_bytes());
            result.extend_from_slice(value_bytes);
        }

        Ok(result)
    }
}

impl Default for PlayReadyPsshData {
    fn default() -> Self {
        Self::new()
    }
}

/// PlayReady license challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayReadyLicenseChallenge {
    /// Challenge data (base64-encoded)
    pub challenge: String,
    /// Header (optional)
    pub header: Option<String>,
}

impl PlayReadyLicenseChallenge {
    /// Create a new license challenge
    pub fn new(challenge: Vec<u8>) -> Self {
        Self {
            challenge: STANDARD.encode(&challenge),
            header: None,
        }
    }

    /// Set header
    pub fn with_header(mut self, header: String) -> Self {
        self.header = Some(header);
        self
    }

    /// Get challenge bytes
    pub fn get_challenge(&self) -> Result<Vec<u8>> {
        STANDARD
            .decode(&self.challenge)
            .map_err(DrmError::Base64Error)
    }
}

/// PlayReady license
#[derive(Debug, Clone)]
pub struct PlayReadyLicense {
    /// License data
    pub data: Vec<u8>,
    /// Expiration time (optional)
    pub expiration: Option<u64>,
}

impl PlayReadyLicense {
    /// Create a new license
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            expiration: None,
        }
    }

    /// Set expiration time
    pub fn with_expiration(mut self, expiration: u64) -> Self {
        self.expiration = Some(expiration);
        self
    }
}

/// PlayReady license server (for testing/mocking)
pub struct PlayReadyLicenseServer {
    keys: HashMap<Uuid, Vec<u8>>,
    license_duration: u64,
}

impl PlayReadyLicenseServer {
    /// Create a new PlayReady license server
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            license_duration: 86400, // 24 hours
        }
    }

    /// Add a key
    pub fn add_key(&mut self, key_id: Uuid, key: Vec<u8>) {
        self.keys.insert(key_id, key);
    }

    /// Set license duration
    pub fn set_license_duration(&mut self, duration: u64) {
        self.license_duration = duration;
    }

    /// Process a license challenge
    pub fn process_challenge(
        &self,
        _challenge: &PlayReadyLicenseChallenge,
        key_id: Uuid,
    ) -> Result<PlayReadyLicense> {
        let key = self
            .keys
            .get(&key_id)
            .ok_or_else(|| DrmError::LicenseError(format!("Key not found: {}", key_id)))?;

        // In a real implementation, this would create a proper PlayReady license object
        // For now, just return the key
        Ok(PlayReadyLicense::new(key.clone()).with_expiration(self.license_duration))
    }

    /// Get number of keys
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
}

impl Default for PlayReadyLicenseServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a PlayReady PSSH box
pub fn create_playready_pssh(header: &PlayReadyHeader) -> Result<Vec<u8>> {
    use crate::cenc::PsshBox;

    let mut pssh_data = PlayReadyPsshData::new();
    pssh_data.add_header(header)?;

    let data = pssh_data.to_bytes()?;
    let pssh = PsshBox::new(DrmSystem::PlayReady.system_id(), data);
    pssh.to_bytes()
}

/// Encode string to UTF-16LE bytes
fn encode_utf16(s: &str) -> Vec<u8> {
    let mut result = Vec::new();
    for ch in s.encode_utf16() {
        result.extend_from_slice(&ch.to_le_bytes());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playready_version() {
        assert_eq!(PlayReadyVersion::V4_0_0_0.as_str(), "4.0.0.0");
        assert_eq!(PlayReadyVersion::V4_3_0_0.as_str(), "4.3.0.0");
    }

    #[test]
    fn test_playready_header() {
        let key_id = Uuid::new_v4();
        let header =
            PlayReadyHeader::new(key_id).with_la_url("https://license.example.com".to_string());

        assert_eq!(header.key_id, key_id);
        assert_eq!(
            header.la_url,
            Some("https://license.example.com".to_string())
        );
    }

    #[test]
    fn test_playready_header_xml() {
        let key_id = Uuid::new_v4();
        let header =
            PlayReadyHeader::new(key_id).with_la_url("https://license.example.com".to_string());

        let xml = header.to_xml().expect("operation should succeed");
        assert!(xml.contains("WRMHEADER"));
        assert!(xml.contains("KID"));
        assert!(xml.contains("LA_URL"));
        assert!(xml.contains("https://license.example.com"));

        let parsed = PlayReadyHeader::from_xml(&xml).expect("operation should succeed");
        assert_eq!(parsed.key_id, key_id);
        assert_eq!(parsed.la_url, header.la_url);
    }

    #[test]
    fn test_playready_record() {
        let record = PlayReadyRecord::rights_management("test content".to_string());
        assert_eq!(record.record_type, "RIGHTS_MANAGEMENT");
        assert_eq!(record.value, "test content");
    }

    #[test]
    fn test_playready_pssh_data() {
        let mut pssh_data = PlayReadyPsshData::new();
        let record = PlayReadyRecord::new("TEST".to_string(), "value".to_string());
        pssh_data.add_record(record);

        assert_eq!(pssh_data.records.len(), 1);

        let bytes = pssh_data.to_bytes().expect("operation should succeed");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_license_challenge() {
        let challenge_data = vec![1, 2, 3, 4, 5];
        let challenge = PlayReadyLicenseChallenge::new(challenge_data.clone());

        let decoded = challenge.get_challenge().expect("operation should succeed");
        assert_eq!(decoded, challenge_data);
    }

    #[test]
    fn test_license_server() {
        let mut server = PlayReadyLicenseServer::new();
        let key_id = Uuid::new_v4();
        let key = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        server.add_key(key_id, key.clone());
        assert_eq!(server.key_count(), 1);

        let challenge = PlayReadyLicenseChallenge::new(vec![1, 2, 3]);
        let license = server
            .process_challenge(&challenge, key_id)
            .expect("operation should succeed");

        assert_eq!(license.data, key);
    }

    #[test]
    fn test_encode_utf16() {
        let s = "test";
        let utf16 = encode_utf16(s);
        assert!(!utf16.is_empty());
        assert_eq!(utf16.len(), 8); // 4 chars * 2 bytes
    }

    // ── Tests for new PlayReadyHeaderObject / PLAYREADY_SYSTEM_ID ─────────────

    #[test]
    fn test_playready_system_id_bytes() {
        assert_eq!(PLAYREADY_SYSTEM_ID[0], 0x9a);
        assert_eq!(PLAYREADY_SYSTEM_ID[15], 0x95);
        assert_eq!(PLAYREADY_SYSTEM_ID.len(), 16);
    }

    #[test]
    fn test_playready_header_object_new_rejects_bad_key() {
        let short_key = vec![0u8; 8];
        let result = PlayReadyHeaderObject::new(short_key, None);
        assert!(
            result.is_err(),
            "should reject key_id shorter than 16 bytes"
        );
    }

    #[test]
    fn test_playready_header_object_to_xml_roundtrip() {
        let key_id: Vec<u8> = (0u8..16).collect();
        let obj =
            PlayReadyHeaderObject::new(key_id.clone(), Some("https://la.example.com".to_string()))
                .expect("valid 16-byte key_id");

        let xml = obj.to_xml().expect("to_xml should succeed");
        assert!(
            xml.contains("WRMHEADER"),
            "XML must contain WRMHEADER element"
        );
        assert!(xml.contains("LA_URL"), "XML must contain LA_URL element");

        let parsed = PlayReadyHeaderObject::parse_xml(&xml).expect("parse_xml should succeed");
        assert_eq!(parsed.key_id, key_id);
        assert_eq!(parsed.la_url.as_deref(), Some("https://la.example.com"));
    }

    #[test]
    fn test_playready_header_object_optional_fields() {
        let key_id: Vec<u8> = vec![0xabu8; 16];
        let obj = PlayReadyHeaderObject::new(key_id.clone(), None)
            .expect("valid 16-byte key_id")
            .with_version(PlayReadySpecVersion::V3_0)
            .with_lui_url("https://lui.example.com".to_string())
            .with_ds_id("domain-123".to_string());

        let xml = obj.to_xml().expect("to_xml should succeed");
        assert!(xml.contains("LUI_URL"), "XML must contain LUI_URL");
        assert!(xml.contains("DS_ID"), "XML must contain DS_ID");
        assert!(xml.contains("4.2.0.0"), "V3_0 maps to version 4.2.0.0");

        let parsed = PlayReadyHeaderObject::parse_xml(&xml).expect("parse_xml should succeed");
        assert_eq!(parsed.version, PlayReadySpecVersion::V3_0);
        assert_eq!(parsed.lui_url.as_deref(), Some("https://lui.example.com"));
        assert_eq!(parsed.ds_id.as_deref(), Some("domain-123"));
    }

    #[test]
    fn test_playready_header_object_to_pssh_box() {
        let key_id: Vec<u8> = (0u8..16).collect();
        let obj = PlayReadyHeaderObject::new(key_id, Some("https://la.example.com".to_string()))
            .expect("valid key_id");

        let pssh = obj.to_pssh_box().expect("to_pssh_box should succeed");
        // Verify PSSH box header
        assert!(pssh.len() >= 28, "PSSH box must be at least 28 bytes");
        assert_eq!(&pssh[4..8], b"pssh", "box type must be 'pssh'");
        assert_eq!(pssh[8], 0, "PSSH version must be 0");
        assert_eq!(&pssh[12..28], &PLAYREADY_SYSTEM_ID, "system_id mismatch");
    }

    #[test]
    fn test_playready_spec_version_header_strings() {
        assert_eq!(PlayReadySpecVersion::V2_0.header_version_str(), "4.0.0.0");
        assert_eq!(PlayReadySpecVersion::V3_0.header_version_str(), "4.2.0.0");
        assert_eq!(PlayReadySpecVersion::V4_0.header_version_str(), "4.3.0.0");
    }
}

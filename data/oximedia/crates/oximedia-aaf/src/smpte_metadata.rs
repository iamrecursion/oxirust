//! SMPTE metadata extensions for AAF
//!
//! Implements SMPTE UMID (Unique Material Identifier), SMPTE Universal Labels,
//! KLV (Key-Length-Value) encoding/decoding, and Reg-395 data model entries
//! per SMPTE ST 330, ST 336, and Reg-395.

use crate::{AafError, Result};

// ─── SMPTE UMID ──────────────────────────────────────────────────────────────

/// SMPTE UMID (Unique Material Identifier) — SMPTE ST 330.
///
/// A basic UMID consists of:
/// - 12-byte SMPTE label prefix (fixed)
/// - 1-byte UMID length (always 0x13 for 32-byte basic UMID)
/// - 1-byte instance type
/// - 1-byte material type
/// - 1-byte method flag
/// - 16-byte material number
/// - 4-byte instance number
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmpteUmid {
    /// 16-byte unique material number.
    pub material_number: [u8; 16],
    /// 4-byte instance number (distinguishes instances of the same material).
    pub instance_number: [u8; 4],
    /// UMID type byte (0x01 = MPEG-encoded material).
    pub umid_type: u8,
}

impl SmpteUmid {
    /// SMPTE UMID label prefix (first 12 bytes of the 32-byte basic UMID).
    ///
    /// Per SMPTE ST 330: `060A2B340101010101011300`
    const LABEL_PREFIX: [u8; 12] = [
        0x06, 0x0A, 0x2B, 0x34, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x13, 0x00,
    ];

    /// Generate a basic UMID from a 16-byte material number.
    ///
    /// Uses UMID type `0x01` (MPEG-encoded) and zero instance number.
    #[must_use]
    pub fn generate_basic(material: [u8; 16]) -> Self {
        Self {
            material_number: material,
            instance_number: [0u8; 4],
            umid_type: 0x01,
        }
    }

    /// Encode the UMID as a 64-character hex string in SMPTE notation.
    ///
    /// The 32-byte basic UMID is formatted as 4 groups of 8 hex pairs,
    /// separated by spaces (e.g. `060A2B34... 01011300... <material> <instance>`).
    ///
    /// The raw bytes are:
    /// - bytes  0-11: label prefix
    /// - byte  12: length (`0x13`)
    /// - byte  13: instance_type (`umid_type`)
    /// - byte  14: material type (`0x01`)
    /// - byte  15: method (`0x00`)
    /// - bytes 16-31: material number
    /// (The 4-byte instance_number is appended for the extended 32-byte form.)
    #[must_use]
    pub fn to_hex_string(&self) -> String {
        let mut bytes = [0u8; 32];

        // Bytes 0-11: label prefix
        bytes[..12].copy_from_slice(&Self::LABEL_PREFIX);
        // Byte 12: length of remainder (19 = 0x13)
        bytes[12] = 0x13;
        // Byte 13: instance type / UMID type
        bytes[13] = self.umid_type;
        // Byte 14: material type (generic = 0x01)
        bytes[14] = 0x01;
        // Byte 15: method (0x00 = no defined method)
        bytes[15] = 0x00;
        // Bytes 16-31: material number
        bytes[16..32].copy_from_slice(&self.material_number);

        // Format: 4 groups of 8 bytes = 4 groups of 16 hex chars
        // Group boundaries: 0-7, 8-15, 16-23, 24-31
        // We omit the extended instance number from the 64-char version
        // but append it separately as a suffix when needed.
        let hex: String = bytes.iter().map(|b| format!("{b:02X}")).collect();
        hex
    }

    /// Encode the UMID including the 4-byte instance number as a 72-char hex string.
    ///
    /// The extra 8 hex chars correspond to the `instance_number` field.
    #[must_use]
    pub fn to_full_hex_string(&self) -> String {
        let mut base = self.to_hex_string();
        let inst: String = self
            .instance_number
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect();
        base.push_str(&inst);
        base
    }
}

// ─── SMPTE Universal Label ────────────────────────────────────────────────────

/// A SMPTE 336 Universal Label (UL) — a 16-byte identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmpteLabel {
    /// 16-byte identifier key.
    pub identifier: [u8; 16],
}

impl SmpteLabel {
    /// Create a new SMPTE label.
    #[must_use]
    pub const fn new(identifier: [u8; 16]) -> Self {
        Self { identifier }
    }

    /// SMPTE UL for Picture Essence (`060E2B34.01010102.04010201.01000000`).
    pub const PICTURE_ESSENCE: Self = Self::new([
        0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01, 0x02, 0x04, 0x01, 0x02, 0x01, 0x01, 0x00, 0x00,
        0x00,
    ]);

    /// SMPTE UL for Sound Essence (`060E2B34.01010101.04020201.01000000`).
    pub const SOUND_ESSENCE: Self = Self::new([
        0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01, 0x01, 0x04, 0x02, 0x02, 0x01, 0x01, 0x00, 0x00,
        0x00,
    ]);

    /// SMPTE UL for Timecode Component (`060E2B34.02530101.0D010101.01030200`).
    pub const TIMECODE_COMPONENT: Self = Self::new([
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x03, 0x02,
        0x00,
    ]);

    /// Format the label as a SMPTE dot-notation string (`XXXXXXXX.XXXXXXXX.XXXXXXXX.XXXXXXXX`).
    #[must_use]
    pub fn to_dot_notation(&self) -> String {
        let bytes = &self.identifier;
        format!(
            "{:02x}{:02x}{:02x}{:02x}.{:02x}{:02x}{:02x}{:02x}.\
             {:02x}{:02x}{:02x}{:02x}.{:02x}{:02x}{:02x}{:02x}",
            bytes[0],
            bytes[1],
            bytes[2],
            bytes[3],
            bytes[4],
            bytes[5],
            bytes[6],
            bytes[7],
            bytes[8],
            bytes[9],
            bytes[10],
            bytes[11],
            bytes[12],
            bytes[13],
            bytes[14],
            bytes[15],
        )
    }
}

// ─── KLV Triplet ─────────────────────────────────────────────────────────────

/// A SMPTE 336 KLV (Key–Length–Value) triplet.
#[derive(Debug, Clone)]
pub struct KlvTriplet {
    /// 16-byte universal label key.
    pub key: SmpteLabel,
    /// Length of the value in bytes.
    pub length: u64,
    /// Value bytes.
    pub value: Vec<u8>,
}

impl KlvTriplet {
    /// Create a new KLV triplet.  `length` is set from `value.len()`.
    #[must_use]
    pub fn new(key: SmpteLabel, value: Vec<u8>) -> Self {
        let length = value.len() as u64;
        Self { key, length, value }
    }
}

/// Encode a `KlvTriplet` into a byte vector using BER length encoding.
///
/// BER length rules (SMPTE 336):
/// - length < 128  → 1 byte: the length itself
/// - otherwise     → `0x80 | n` followed by `n` big-endian bytes of the length
#[must_use]
pub fn encode_klv(triplet: &KlvTriplet) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + 9 + triplet.value.len());

    // Key (always 16 bytes)
    out.extend_from_slice(&triplet.key.identifier);

    // BER-encoded length
    encode_ber_length(&mut out, triplet.length);

    // Value
    out.extend_from_slice(&triplet.value);

    out
}

/// Encode a length value using BER short-form or long-form.
fn encode_ber_length(out: &mut Vec<u8>, length: u64) {
    if length < 128 {
        out.push(length as u8);
    } else {
        // Determine how many bytes the length occupies
        let n_bytes = ber_length_bytes_needed(length);
        out.push(0x80 | n_bytes as u8);
        // Write length big-endian, n_bytes wide
        for shift in (0..n_bytes).rev() {
            out.push(((length >> (shift * 8)) & 0xFF) as u8);
        }
    }
}

/// Number of bytes needed to represent `length` in big-endian.
fn ber_length_bytes_needed(length: u64) -> usize {
    if length <= 0xFF {
        1
    } else if length <= 0xFFFF {
        2
    } else if length <= 0xFF_FFFF {
        3
    } else if length <= 0xFFFF_FFFF {
        4
    } else if length <= 0xFF_FFFF_FFFF {
        5
    } else if length <= 0xFFFF_FFFF_FFFF {
        6
    } else if length <= 0xFF_FFFF_FFFF_FFFF {
        7
    } else {
        8
    }
}

/// Decode a `KlvTriplet` from raw bytes.
///
/// Returns `(triplet, bytes_consumed)` on success.
///
/// # Errors
///
/// Returns `AafError::ParseError` if:
/// - there are fewer than 17 bytes (16 key + at least 1 length byte)
/// - the BER length field indicates more bytes than available in `data`
/// - the value is truncated
pub fn decode_klv(data: &[u8]) -> Result<(KlvTriplet, usize)> {
    // Need at least 16 bytes for key + 1 byte for length
    if data.len() < 17 {
        return Err(AafError::ParseError(format!(
            "KLV data too short: {} bytes (need ≥ 17)",
            data.len()
        )));
    }

    // Parse the 16-byte key
    let mut key_bytes = [0u8; 16];
    key_bytes.copy_from_slice(&data[..16]);
    let key = SmpteLabel::new(key_bytes);

    // Parse BER length
    let (length, ber_bytes) = decode_ber_length(&data[16..])?;
    let header_len = 16 + ber_bytes;

    // Validate value boundaries
    let value_end = header_len
        .checked_add(length as usize)
        .ok_or_else(|| AafError::ParseError("KLV length overflow".to_string()))?;

    if value_end > data.len() {
        return Err(AafError::ParseError(format!(
            "KLV value truncated: need {} bytes, have {}",
            value_end,
            data.len()
        )));
    }

    let value = data[header_len..value_end].to_vec();
    let triplet = KlvTriplet { key, length, value };
    Ok((triplet, value_end))
}

/// Decode a BER length field from `data`.
///
/// Returns `(decoded_length, bytes_consumed)`.
fn decode_ber_length(data: &[u8]) -> Result<(u64, usize)> {
    if data.is_empty() {
        return Err(AafError::ParseError(
            "BER length field is empty".to_string(),
        ));
    }

    let first = data[0];
    if first < 0x80 {
        // Short form
        return Ok((u64::from(first), 1));
    }

    // Long form: lower 7 bits = number of subsequent length bytes
    let n = (first & 0x7F) as usize;
    if n == 0 {
        // Indefinite length — not supported in SMPTE 336
        return Err(AafError::ParseError(
            "Indefinite BER length not supported".to_string(),
        ));
    }
    if n > 8 {
        return Err(AafError::ParseError(format!(
            "BER length too large: {n} bytes"
        )));
    }
    if data.len() < 1 + n {
        return Err(AafError::ParseError(format!(
            "BER long-form length truncated: need {n} bytes, have {}",
            data.len() - 1
        )));
    }

    let mut length = 0u64;
    for &byte in &data[1..=n] {
        length = (length << 8) | u64::from(byte);
    }
    Ok((length, 1 + n))
}

// ─── UMID Parsing ─────────────────────────────────────────────────────────────

impl SmpteUmid {
    /// Parse a UMID from a 64-character hex string (32 bytes).
    ///
    /// The string must be exactly 64 hex characters (no separators).
    ///
    /// # Errors
    ///
    /// Returns `AafError::ParseError` if the string is not valid hex or wrong length.
    pub fn from_hex_string(hex: &str) -> Result<Self> {
        if hex.len() != 64 {
            return Err(AafError::ParseError(format!(
                "UMID hex string must be 64 chars, got {}",
                hex.len()
            )));
        }

        let bytes = hex_string_to_bytes(hex)?;
        if bytes.len() != 32 {
            return Err(AafError::ParseError(
                "Failed to decode 32 bytes from hex".to_string(),
            ));
        }

        // Validate SMPTE prefix (first 4 bytes)
        if bytes[0] != 0x06 || bytes[1] != 0x0A || bytes[2] != 0x2B || bytes[3] != 0x34 {
            return Err(AafError::ParseError(
                "Invalid SMPTE UMID prefix".to_string(),
            ));
        }

        let umid_type = bytes[13];
        let mut material_number = [0u8; 16];
        material_number.copy_from_slice(&bytes[16..32]);

        Ok(Self {
            material_number,
            instance_number: [0u8; 4],
            umid_type,
        })
    }

    /// Parse a UMID from a 72-character hex string (32 bytes + 4 instance bytes).
    pub fn from_full_hex_string(hex: &str) -> Result<Self> {
        if hex.len() != 72 {
            return Err(AafError::ParseError(format!(
                "Full UMID hex string must be 72 chars, got {}",
                hex.len()
            )));
        }

        let mut umid = Self::from_hex_string(&hex[..64])?;
        let inst_bytes = hex_string_to_bytes(&hex[64..72])?;
        if inst_bytes.len() >= 4 {
            umid.instance_number[0] = inst_bytes[0];
            umid.instance_number[1] = inst_bytes[1];
            umid.instance_number[2] = inst_bytes[2];
            umid.instance_number[3] = inst_bytes[3];
        }

        Ok(umid)
    }

    /// Check whether this UMID has a non-zero instance number.
    #[must_use]
    pub fn has_instance(&self) -> bool {
        self.instance_number != [0u8; 4]
    }

    /// Get the UMID type as a descriptive string.
    #[must_use]
    pub fn type_description(&self) -> &'static str {
        match self.umid_type {
            0x01 => "MPEG-encoded",
            0x02 => "SMPTE-controlled",
            0x03 => "ISO-controlled",
            _ => "Unknown",
        }
    }
}

/// Decode a hex string into bytes.
fn hex_string_to_bytes(hex: &str) -> Result<Vec<u8>> {
    let hex = hex.trim();
    if hex.len() % 2 != 0 {
        return Err(AafError::ParseError(
            "Hex string must have even length".to_string(),
        ));
    }

    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let mut i = 0;
    while i + 1 < hex.len() {
        let byte_str = &hex[i..i + 2];
        let byte = u8::from_str_radix(byte_str, 16)
            .map_err(|_| AafError::ParseError(format!("Invalid hex byte: '{byte_str}'")))?;
        bytes.push(byte);
        i += 2;
    }

    Ok(bytes)
}

// ─── SmpteLabel extensions ────────────────────────────────────────────────────

impl SmpteLabel {
    /// SMPTE UL for Data Essence (`060E2B34.01010101.04030100.00000000`).
    pub const DATA_ESSENCE: Self = Self::new([
        0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01, 0x01, 0x04, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ]);

    /// SMPTE UL for Descriptive Metadata (`060E2B34.02530101.0D010101.01040100`).
    pub const DESCRIPTIVE_METADATA: Self = Self::new([
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x04, 0x01,
        0x00,
    ]);

    /// Parse a SMPTE UL from a dot-notation string (`XXXXXXXX.XXXXXXXX.XXXXXXXX.XXXXXXXX`).
    pub fn from_dot_notation(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 4 {
            return Err(AafError::ParseError(format!(
                "Expected 4 dot-separated groups, got {}",
                parts.len()
            )));
        }

        let mut identifier = [0u8; 16];
        for (group_idx, part) in parts.iter().enumerate() {
            if part.len() != 8 {
                return Err(AafError::ParseError(format!(
                    "Group {} must be 8 hex chars, got {}",
                    group_idx,
                    part.len()
                )));
            }
            let group_bytes = hex_string_to_bytes(part)?;
            let offset = group_idx * 4;
            for (j, &b) in group_bytes.iter().enumerate() {
                identifier[offset + j] = b;
            }
        }

        Ok(Self { identifier })
    }

    /// Check whether this label matches the SMPTE UL prefix (`060E2B34`).
    #[must_use]
    pub fn is_smpte_ul(&self) -> bool {
        self.identifier[0] == 0x06
            && self.identifier[1] == 0x0E
            && self.identifier[2] == 0x2B
            && self.identifier[3] == 0x34
    }

    /// Return the category designator byte (byte 4).
    #[must_use]
    pub fn category_designator(&self) -> u8 {
        self.identifier[4]
    }

    /// Return the registry designator byte (byte 5).
    #[must_use]
    pub fn registry_designator(&self) -> u8 {
        self.identifier[5]
    }
}

// ─── KLV Batch Encoding ──────────────────────────────────────────────────────

/// Encode multiple KLV triplets into a single byte buffer.
#[must_use]
pub fn encode_klv_batch(triplets: &[KlvTriplet]) -> Vec<u8> {
    let total_size: usize = triplets
        .iter()
        .map(|t| 16 + ber_encoded_length_size(t.length) + t.value.len())
        .sum();

    let mut out = Vec::with_capacity(total_size);
    for triplet in triplets {
        out.extend(encode_klv(triplet));
    }
    out
}

/// Decode all KLV triplets from a contiguous byte buffer.
///
/// Stops when the remaining data is insufficient for another KLV header.
pub fn decode_klv_batch(data: &[u8]) -> Result<Vec<KlvTriplet>> {
    let mut triplets = Vec::new();
    let mut offset = 0;

    while offset + 17 <= data.len() {
        let (triplet, consumed) = decode_klv(&data[offset..])?;
        triplets.push(triplet);
        offset += consumed;
    }

    Ok(triplets)
}

/// Compute the byte size of a BER-encoded length field.
fn ber_encoded_length_size(length: u64) -> usize {
    if length < 128 {
        1
    } else {
        1 + ber_length_bytes_needed(length)
    }
}

// ─── Reg-395 Data Model ───────────────────────────────────────────────────────

/// A SMPTE Reg-395 data model entry.
///
/// Reg-395 defines the AAF/MXF metadata dictionary in terms of paths,
/// symbolic names, and type symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reg395DataModel {
    /// Hierarchical path of the property (e.g. `"/AAF/Header/ByteOrder"`).
    pub path: String,
    /// Human-readable property name.
    pub name: String,
    /// Type symbol as defined in Reg-395 (e.g. `"aafUInt16_t"`).
    pub type_sym: String,
}

impl Reg395DataModel {
    /// Create a new Reg-395 entry.
    #[must_use]
    pub fn new(
        path: impl Into<String>,
        name: impl Into<String>,
        type_sym: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            name: name.into(),
            type_sym: type_sym.into(),
        }
    }

    /// Well-known entry: `ByteOrder` property of `AAFHeader`.
    #[must_use]
    pub fn byte_order() -> Self {
        Self::new("/AAF/Header/ByteOrder", "ByteOrder", "aafUInt16_t")
    }

    /// Well-known entry: `ObjectModelVersion` property of `AAFHeader`.
    #[must_use]
    pub fn object_model_version() -> Self {
        Self::new(
            "/AAF/Header/ObjectModelVersion",
            "ObjectModelVersion",
            "aafUInt32_t",
        )
    }

    /// Check whether this entry's type is an integer type.
    #[must_use]
    pub fn is_integer_type(&self) -> bool {
        self.type_sym.contains("Int") || self.type_sym.contains("int")
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── UMID tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_umid_generate_basic() {
        let material = [0x01u8; 16];
        let umid = SmpteUmid::generate_basic(material);
        assert_eq!(umid.material_number, material);
        assert_eq!(umid.instance_number, [0u8; 4]);
        assert_eq!(umid.umid_type, 0x01);
    }

    #[test]
    fn test_umid_hex_string_length() {
        let material = [0xABu8; 16];
        let umid = SmpteUmid::generate_basic(material);
        let hex = umid.to_hex_string();
        // 32 bytes * 2 hex chars = 64 chars
        assert_eq!(hex.len(), 64, "hex={hex}");
    }

    #[test]
    fn test_umid_hex_string_prefix() {
        let umid = SmpteUmid::generate_basic([0u8; 16]);
        let hex = umid.to_hex_string();
        // First 4 bytes are 060A2B34
        assert!(
            hex.starts_with("060A2B34"),
            "Expected SMPTE prefix, got: {hex}"
        );
    }

    #[test]
    fn test_umid_full_hex_string_length() {
        let umid = SmpteUmid::generate_basic([0u8; 16]);
        let full = umid.to_full_hex_string();
        // 32 bytes + 4 instance bytes = 36 bytes * 2 = 72 chars
        assert_eq!(full.len(), 72, "full={full}");
    }

    #[test]
    fn test_umid_instance_number_in_full_hex() {
        let mut umid = SmpteUmid::generate_basic([0u8; 16]);
        umid.instance_number = [0xDE, 0xAD, 0xBE, 0xEF];
        let full = umid.to_full_hex_string();
        assert!(full.ends_with("DEADBEEF"), "full={full}");
    }

    // ── SmpteLabel tests ───────────────────────────────────────────────────

    #[test]
    fn test_picture_essence_label() {
        let label = SmpteLabel::PICTURE_ESSENCE;
        assert_eq!(label.identifier[0], 0x06);
        assert_eq!(label.identifier[1], 0x0E);
        assert_eq!(label.identifier[2], 0x2B);
        assert_eq!(label.identifier[3], 0x34);
    }

    #[test]
    fn test_sound_essence_label_differs_from_picture() {
        assert_ne!(
            SmpteLabel::PICTURE_ESSENCE.identifier,
            SmpteLabel::SOUND_ESSENCE.identifier
        );
    }

    #[test]
    fn test_timecode_component_label() {
        let label = SmpteLabel::TIMECODE_COMPONENT;
        assert_eq!(label.identifier[4], 0x02, "5th byte should be 0x02");
    }

    #[test]
    fn test_label_dot_notation_format() {
        let label = SmpteLabel::PICTURE_ESSENCE;
        let dot = label.to_dot_notation();
        let parts: Vec<&str> = dot.split('.').collect();
        assert_eq!(parts.len(), 4, "dot={dot}");
        for part in &parts {
            assert_eq!(part.len(), 8, "each group should be 8 hex chars");
        }
    }

    #[test]
    fn test_label_dot_notation_picture() {
        let label = SmpteLabel::PICTURE_ESSENCE;
        let dot = label.to_dot_notation();
        assert!(dot.starts_with("060e2b34"), "dot={dot}");
    }

    // ── KLV encoding tests ─────────────────────────────────────────────────

    #[test]
    fn test_encode_klv_short_value() {
        let key = SmpteLabel::PICTURE_ESSENCE;
        let value = vec![0xAA, 0xBB, 0xCC];
        let triplet = KlvTriplet::new(key, value.clone());
        let encoded = encode_klv(&triplet);
        // 16 (key) + 1 (length byte) + 3 (value)
        assert_eq!(encoded.len(), 20);
        assert_eq!(&encoded[..16], &key.identifier);
        assert_eq!(encoded[16], 3); // BER short-form length
        assert_eq!(&encoded[17..], &value);
    }

    #[test]
    fn test_encode_klv_empty_value() {
        let key = SmpteLabel::SOUND_ESSENCE;
        let triplet = KlvTriplet::new(key, vec![]);
        let encoded = encode_klv(&triplet);
        assert_eq!(encoded.len(), 17); // 16 + 1 length byte
        assert_eq!(encoded[16], 0);
    }

    #[test]
    fn test_encode_klv_ber_long_form() {
        // Value with 200 bytes → needs BER long form (length ≥ 128)
        let key = SmpteLabel::PICTURE_ESSENCE;
        let value = vec![0x42u8; 200];
        let triplet = KlvTriplet::new(key, value);
        let encoded = encode_klv(&triplet);
        // BER: 0x81 0xC8 → 2 bytes for length
        assert_eq!(encoded[16], 0x81, "Expected 0x81 for 1-byte long-form BER");
        assert_eq!(encoded[17], 200, "Length byte should be 200");
        assert_eq!(encoded.len(), 16 + 2 + 200);
    }

    #[test]
    fn test_klv_roundtrip_short() {
        let key = SmpteLabel::PICTURE_ESSENCE;
        let value = vec![1u8, 2, 3, 4, 5];
        let triplet = KlvTriplet::new(key, value.clone());
        let encoded = encode_klv(&triplet);

        let (decoded, consumed) = decode_klv(&encoded).expect("decode should succeed");
        assert_eq!(decoded.key.identifier, key.identifier);
        assert_eq!(decoded.value, value);
        assert_eq!(decoded.length, 5);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn test_klv_roundtrip_long_form() {
        let key = SmpteLabel::SOUND_ESSENCE;
        let value: Vec<u8> = (0..=255u8).cycle().take(300).collect();
        let triplet = KlvTriplet::new(key, value.clone());
        let encoded = encode_klv(&triplet);

        let (decoded, consumed) = decode_klv(&encoded).expect("decode long form");
        assert_eq!(decoded.value, value);
        assert_eq!(decoded.length, 300);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn test_klv_roundtrip_empty() {
        let key = SmpteLabel::TIMECODE_COMPONENT;
        let triplet = KlvTriplet::new(key, vec![]);
        let encoded = encode_klv(&triplet);
        let (decoded, _) = decode_klv(&encoded).expect("decode empty");
        assert!(decoded.value.is_empty());
        assert_eq!(decoded.length, 0);
    }

    #[test]
    fn test_decode_klv_too_short() {
        let data = [0u8; 10];
        let result = decode_klv(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_klv_truncated_value() {
        let key = SmpteLabel::PICTURE_ESSENCE;
        let value = vec![0xFFu8; 50];
        let triplet = KlvTriplet::new(key, value);
        let mut encoded = encode_klv(&triplet);
        // Truncate the value
        encoded.truncate(encoded.len() - 10);
        let result = decode_klv(&encoded);
        assert!(result.is_err(), "Should error on truncated value");
    }

    #[test]
    fn test_decode_klv_consecutive() {
        // Encode two KLVs back-to-back and decode them both
        let k1 = SmpteLabel::PICTURE_ESSENCE;
        let k2 = SmpteLabel::SOUND_ESSENCE;
        let v1 = vec![0x01, 0x02];
        let v2 = vec![0x03, 0x04, 0x05];
        let t1 = KlvTriplet::new(k1, v1.clone());
        let t2 = KlvTriplet::new(k2, v2.clone());
        let mut buf = encode_klv(&t1);
        buf.extend(encode_klv(&t2));

        let (d1, n1) = decode_klv(&buf).expect("first KLV");
        assert_eq!(d1.value, v1);
        let (d2, _) = decode_klv(&buf[n1..]).expect("second KLV");
        assert_eq!(d2.value, v2);
    }

    // ── Reg-395 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_reg395_new() {
        let entry = Reg395DataModel::new("/AAF/Mob/Name", "Name", "aafCharacter_t");
        assert_eq!(entry.path, "/AAF/Mob/Name");
        assert_eq!(entry.name, "Name");
        assert_eq!(entry.type_sym, "aafCharacter_t");
    }

    #[test]
    fn test_reg395_byte_order() {
        let entry = Reg395DataModel::byte_order();
        assert!(entry.path.contains("ByteOrder"));
        assert!(entry.is_integer_type(), "aafUInt16_t should be integer");
    }

    #[test]
    fn test_reg395_object_model_version() {
        let entry = Reg395DataModel::object_model_version();
        assert!(entry.type_sym.contains("UInt32"));
    }

    #[test]
    fn test_reg395_not_integer_type() {
        let entry = Reg395DataModel::new("/AAF/Mob/Name", "Name", "aafCharacter_t");
        assert!(!entry.is_integer_type());
    }

    #[test]
    fn test_reg395_equality() {
        let a = Reg395DataModel::byte_order();
        let b = Reg395DataModel::byte_order();
        assert_eq!(a, b);
    }

    // ── UMID parsing tests ────────────────────────────────────────────────

    #[test]
    fn test_umid_hex_roundtrip() {
        let material = [
            0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
            0x0B, 0x0C,
        ];
        let umid = SmpteUmid::generate_basic(material);
        let hex = umid.to_hex_string();
        let parsed = SmpteUmid::from_hex_string(&hex).expect("parse hex");
        assert_eq!(parsed.material_number, material);
        assert_eq!(parsed.umid_type, 0x01);
    }

    #[test]
    fn test_umid_full_hex_roundtrip() {
        let mut umid = SmpteUmid::generate_basic([0xAAu8; 16]);
        umid.instance_number = [0x11, 0x22, 0x33, 0x44];
        let full_hex = umid.to_full_hex_string();
        let parsed = SmpteUmid::from_full_hex_string(&full_hex).expect("parse full hex");
        assert_eq!(parsed.instance_number, [0x11, 0x22, 0x33, 0x44]);
    }

    #[test]
    fn test_umid_from_hex_string_wrong_length() {
        assert!(SmpteUmid::from_hex_string("ABCD").is_err());
    }

    #[test]
    fn test_umid_from_hex_string_invalid_hex() {
        let bad = "ZZZZ".to_string() + &"0".repeat(60);
        assert!(SmpteUmid::from_hex_string(&bad).is_err());
    }

    #[test]
    fn test_umid_has_instance() {
        let umid = SmpteUmid::generate_basic([0u8; 16]);
        assert!(!umid.has_instance());
        let mut umid2 = umid.clone();
        umid2.instance_number = [0, 0, 0, 1];
        assert!(umid2.has_instance());
    }

    #[test]
    fn test_umid_type_description() {
        let umid = SmpteUmid::generate_basic([0u8; 16]);
        assert_eq!(umid.type_description(), "MPEG-encoded");
    }

    // ── SmpteLabel extensions tests ────────────────────────────────────────

    #[test]
    fn test_label_dot_notation_roundtrip() {
        let label = SmpteLabel::PICTURE_ESSENCE;
        let dot = label.to_dot_notation();
        let parsed = SmpteLabel::from_dot_notation(&dot).expect("parse dot notation");
        assert_eq!(parsed.identifier, label.identifier);
    }

    #[test]
    fn test_label_from_dot_notation_wrong_groups() {
        assert!(SmpteLabel::from_dot_notation("060e2b34.01010102").is_err());
    }

    #[test]
    fn test_label_is_smpte_ul() {
        assert!(SmpteLabel::PICTURE_ESSENCE.is_smpte_ul());
        let custom = SmpteLabel::new([0x00u8; 16]);
        assert!(!custom.is_smpte_ul());
    }

    #[test]
    fn test_label_category_designator() {
        let label = SmpteLabel::PICTURE_ESSENCE;
        assert_eq!(label.category_designator(), 0x01);
    }

    #[test]
    fn test_data_essence_label() {
        assert!(SmpteLabel::DATA_ESSENCE.is_smpte_ul());
    }

    #[test]
    fn test_descriptive_metadata_label() {
        assert!(SmpteLabel::DESCRIPTIVE_METADATA.is_smpte_ul());
    }

    // ── KLV batch tests ───────────────────────────────────────────────────

    #[test]
    fn test_klv_batch_encode_decode() {
        let t1 = KlvTriplet::new(SmpteLabel::PICTURE_ESSENCE, vec![0x01, 0x02, 0x03]);
        let t2 = KlvTriplet::new(SmpteLabel::SOUND_ESSENCE, vec![0x04, 0x05]);
        let t3 = KlvTriplet::new(SmpteLabel::TIMECODE_COMPONENT, vec![0x06]);

        let encoded = encode_klv_batch(&[t1, t2, t3]);
        let decoded = decode_klv_batch(&encoded).expect("batch decode");

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].value, vec![0x01, 0x02, 0x03]);
        assert_eq!(decoded[1].value, vec![0x04, 0x05]);
        assert_eq!(decoded[2].value, vec![0x06]);
    }

    #[test]
    fn test_klv_batch_empty() {
        let encoded = encode_klv_batch(&[]);
        assert!(encoded.is_empty());
        let decoded = decode_klv_batch(&encoded).expect("empty batch");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_klv_batch_long_values() {
        let big_value: Vec<u8> = (0..=255u8).cycle().take(500).collect();
        let t = KlvTriplet::new(SmpteLabel::PICTURE_ESSENCE, big_value.clone());
        let encoded = encode_klv_batch(&[t]);
        let decoded = decode_klv_batch(&encoded).expect("long value batch");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].value, big_value);
    }
}

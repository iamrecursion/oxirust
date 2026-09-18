//! `ControlInformation` — the small, self-describing preamble that precedes the global header,
//! the header, the dictionary, and the triples section of every HDT file.
//!
//! # Wire format
//!
//! ```text
//! bytes    "$HDT"                          (4-byte cookie, not NUL-terminated)
//! byte     control type                    (Unknown=0, Global=1, Header=2, Dictionary=3,
//!                                            Triples=4, Index=5)
//! cstring  format IRI                      (NUL-terminated; may be empty)
//! cstring  "key1=value1;key2=value2;..."    (NUL-terminated; semicolon-terminated pairs)
//! CRC16    of every byte above (cookie through the properties' terminating NUL, inclusive)
//! ```
//!
//! This mirrors the reference `hdt-cpp` implementation
//! (`libhdt/src/hdt/ControlInformation.cpp`) exactly, including the CRC16 coverage.

use std::collections::BTreeMap;

use crate::crc::crc16;
use crate::error::{HdtError, Result};

/// The four-byte magic cookie that opens every [`ControlInformation`] block.
pub const COOKIE: &[u8; 4] = b"$HDT";

/// The kind of section a [`ControlInformation`] block introduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ControlInfoType {
    /// Not yet assigned (never written by this crate; recognized on read for robustness).
    Unknown = 0,
    /// The file-level control information at the very start of an HDT file.
    Global = 1,
    /// Introduces the plain-RDF header section.
    Header = 2,
    /// Introduces the dictionary section.
    Dictionary = 3,
    /// Introduces the triples section.
    Triples = 4,
    /// Introduces an auxiliary index (not produced by this crate).
    Index = 5,
}

impl ControlInfoType {
    /// Maps a raw control-type byte to a [`ControlInfoType`].
    ///
    /// # Errors
    ///
    /// Returns [`HdtError::UnexpectedControlType`] for any byte outside `0..=5`.
    pub fn from_byte(byte: u8, section: &'static str) -> Result<Self> {
        Ok(match byte {
            0 => Self::Unknown,
            1 => Self::Global,
            2 => Self::Header,
            3 => Self::Dictionary,
            4 => Self::Triples,
            5 => Self::Index,
            found => return Err(HdtError::UnexpectedControlType { section, found }),
        })
    }
}

/// A parsed `ControlInformation` block: a type, a format IRI, and a `key=value` property map.
#[derive(Debug, Clone, Default)]
pub struct ControlInformation {
    /// The section this control information introduces.
    pub control_type: Option<ControlInfoType>,
    /// The format IRI of the data that follows (e.g.
    /// `<http://purl.org/HDT/hdt#dictionaryFour>`).
    pub format: String,
    /// Free-form `key=value` properties (e.g. `numTriples`, `order`, `mapping`).
    pub properties: BTreeMap<String, String>,
}

impl ControlInformation {
    /// Creates an empty control-information block of the given `control_type` and `format`.
    #[must_use]
    pub fn new(control_type: ControlInfoType, format: impl Into<String>) -> Self {
        Self {
            control_type: Some(control_type),
            format: format.into(),
            properties: BTreeMap::new(),
        }
    }

    /// Sets a property, as a builder-style method.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Reads a property as a `u64`, returning `0` if it is absent or unparseable — matching the
    /// permissive behaviour of the reference implementation's `getUint`.
    #[must_use]
    pub fn property_u64(&self, key: &str) -> u64 {
        self.properties
            .get(key)
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    }

    /// Returns `true` if this block's `format` is an acceptable spelling of `expected`.
    ///
    /// The comparison ignores whether the IRI is wrapped in angle brackets: both spellings (e.g.
    /// `<http://purl.org/HDT/hdt#HDTv1>` and `http://purl.org/HDT/hdt#HDTv1`) are seen across HDT
    /// implementations in the wild, and neither this crate's own [`write`](Self::write) callers
    /// nor the wire format itself require one form over the other.
    #[must_use]
    pub fn format_matches(&self, expected: &str) -> bool {
        /// Strips a single pair of surrounding angle brackets, if present.
        fn bare(value: &str) -> &str {
            value
                .strip_prefix('<')
                .and_then(|value| value.strip_suffix('>'))
                .unwrap_or(value)
        }
        bare(&self.format) == bare(expected)
    }

    /// Serializes this block, appending it to `out`.
    pub fn write(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(COOKIE);
        out.push(self.control_type.map_or(0, |type_| type_ as u8));
        out.extend_from_slice(self.format.as_bytes());
        out.push(0);

        let mut properties = String::new();
        for (key, value) in &self.properties {
            properties.push_str(key);
            properties.push('=');
            properties.push_str(value);
            properties.push(';');
        }
        out.extend_from_slice(properties.as_bytes());
        out.push(0);

        let crc = crc16(&out[start..]);
        out.extend_from_slice(&crc.to_le_bytes());
    }

    /// Deserializes a control-information block starting at `*pos`, advancing it past the
    /// block (including its trailing CRC16).
    ///
    /// # Errors
    ///
    /// Returns [`HdtError::InvalidCookie`] if the four-byte magic cookie does not match,
    /// [`HdtError::UnexpectedEof`] if the input is truncated, and
    /// [`HdtError::CrcMismatch`] if the trailing CRC16 does not verify.
    pub fn read(data: &[u8], pos: &mut usize, section: &'static str) -> Result<Self> {
        let start = *pos;
        let cookie = data
            .get(*pos..*pos + 4)
            .ok_or(HdtError::UnexpectedEof { section })?;
        if cookie != COOKIE {
            let mut found = [0u8; 4];
            found.copy_from_slice(cookie);
            return Err(HdtError::InvalidCookie { found });
        }
        *pos += 4;

        let type_byte = *data.get(*pos).ok_or(HdtError::UnexpectedEof { section })?;
        *pos += 1;
        let control_type = ControlInfoType::from_byte(type_byte, section)?;

        let format = read_cstring(data, pos, section)?;
        let properties_raw = read_cstring(data, pos, section)?;

        let end = *pos;
        let computed = crc16(&data[start..end]);
        let crc_bytes = data
            .get(*pos..*pos + 2)
            .ok_or(HdtError::UnexpectedEof { section })?;
        let stored = u16::from_le_bytes([crc_bytes[0], crc_bytes[1]]);
        *pos += 2;
        if stored != computed {
            return Err(HdtError::CrcMismatch {
                kind: "CRC16",
                section,
                expected: u32::from(stored),
                actual: u32::from(computed),
            });
        }

        let mut properties = BTreeMap::new();
        for pair in properties_raw.split(';') {
            if let Some((key, value)) = pair.split_once('=') {
                properties.insert(key.to_owned(), value.to_owned());
            }
        }

        Ok(Self {
            control_type: Some(control_type),
            format,
            properties,
        })
    }
}

/// Reads a NUL-terminated string starting at `*pos`, advancing it past the terminator.
fn read_cstring(data: &[u8], pos: &mut usize, section: &'static str) -> Result<String> {
    let start = *pos;
    let mut end = start;
    loop {
        let byte = *data.get(end).ok_or(HdtError::UnexpectedEof { section })?;
        if byte == 0 {
            break;
        }
        end += 1;
    }
    let value = String::from_utf8_lossy(&data[start..end]).into_owned();
    *pos = end + 1;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_with_properties() -> Result<()> {
        let info = ControlInformation::new(ControlInfoType::Triples, "triplesBitmap")
            .with_property("numTriples", "42")
            .with_property("order", "1");
        let mut buffer = Vec::new();
        info.write(&mut buffer);
        let mut pos = 0;
        let decoded = ControlInformation::read(&buffer, &mut pos, "test")?;
        assert_eq!(pos, buffer.len());
        assert_eq!(decoded.control_type, Some(ControlInfoType::Triples));
        assert_eq!(decoded.format, "triplesBitmap");
        assert_eq!(decoded.property_u64("numTriples"), 42);
        assert_eq!(decoded.property_u64("order"), 1);
        assert_eq!(decoded.property_u64("missing"), 0);
        Ok(())
    }

    #[test]
    fn roundtrips_with_no_properties() -> Result<()> {
        let info = ControlInformation::new(ControlInfoType::Global, "hdt#HDTv1");
        let mut buffer = Vec::new();
        info.write(&mut buffer);
        let mut pos = 0;
        let decoded = ControlInformation::read(&buffer, &mut pos, "test")?;
        assert!(decoded.properties.is_empty());
        Ok(())
    }

    #[test]
    fn format_matches_ignores_angle_brackets() {
        let info =
            ControlInformation::new(ControlInfoType::Global, "<http://purl.org/HDT/hdt#HDTv1>");
        assert!(info.format_matches("<http://purl.org/HDT/hdt#HDTv1>"));
        assert!(info.format_matches("http://purl.org/HDT/hdt#HDTv1"));
        assert!(!info.format_matches("<http://purl.org/HDT/hdt#totally-different>"));
    }

    #[test]
    fn rejects_bad_cookie() {
        let buffer = b"NOPE".to_vec();
        let mut pos = 0;
        let error =
            ControlInformation::read(&buffer, &mut pos, "test").expect_err("expected an error");
        assert!(matches!(error, HdtError::InvalidCookie { .. }));
    }

    #[test]
    fn rejects_bad_crc() {
        let info = ControlInformation::new(ControlInfoType::Header, "ntriples");
        let mut buffer = Vec::new();
        info.write(&mut buffer);
        let last = buffer.len() - 1;
        buffer[last] ^= 0xFF;
        let mut pos = 0;
        let error =
            ControlInformation::read(&buffer, &mut pos, "test").expect_err("expected an error");
        assert!(matches!(error, HdtError::CrcMismatch { kind: "CRC16", .. }));
    }

    #[test]
    fn rejects_truncated_input() {
        let info = ControlInformation::new(ControlInfoType::Header, "ntriples");
        let mut buffer = Vec::new();
        info.write(&mut buffer);
        buffer.truncate(buffer.len() - 3);
        let mut pos = 0;
        let error =
            ControlInformation::read(&buffer, &mut pos, "test").expect_err("expected an error");
        assert!(matches!(error, HdtError::UnexpectedEof { .. }));
    }
}

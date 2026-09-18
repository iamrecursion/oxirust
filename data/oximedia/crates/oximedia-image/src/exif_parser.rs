//! Minimal EXIF metadata parser.
//!
//! Parses a subset of EXIF tags from in-memory byte slices without
//! external dependencies.  Supports both big-endian (Motorola) and
//! little-endian (Intel) IFD layouts.

#![allow(dead_code)]

use std::collections::HashMap;

/// Well-known EXIF tag identifiers (IFD tag codes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExifTag {
    /// Image width in pixels (tag 0x0100).
    ImageWidth,
    /// Image height in pixels (tag 0x0101).
    ImageHeight,
    /// Number of bits per sample component (tag 0x0102).
    BitsPerSample,
    /// Compression scheme (tag 0x0103).
    Compression,
    /// Image description / caption (tag 0x010E).
    ImageDescription,
    /// Camera make / manufacturer (tag 0x010F).
    Make,
    /// Camera model (tag 0x0110).
    Model,
    /// X resolution in DPI or DPC (tag 0x011A).
    XResolution,
    /// Y resolution in DPI or DPC (tag 0x011B).
    YResolution,
    /// Date/time of original capture (tag 0x9003).
    DateTimeOriginal,
    /// ISO speed rating (tag 0x8827).
    IsoSpeedRatings,
    /// Shutter speed as an APEX value (tag 0x9201).
    ShutterSpeedValue,
    /// Aperture as an APEX value (tag 0x9202).
    ApertureValue,
    /// Exposure time in seconds (tag 0x829A).
    ExposureTime,
    /// F-number (tag 0x829D).
    FNumber,
    /// Focal length in millimetres (tag 0x920A).
    FocalLength,
    /// Software used for capture or processing (tag 0x0131).
    Software,
    /// Artist / creator (tag 0x013B).
    Artist,
    /// Copyright notice (tag 0x8298).
    Copyright,
    /// GPS latitude (tag 0x0002 in GPS IFD).
    GpsLatitude,
    /// GPS longitude (tag 0x0004 in GPS IFD).
    GpsLongitude,
    /// An unrecognised tag identified only by its numeric code.
    Unknown(u16),
}

impl ExifTag {
    /// Convert a numeric IFD tag code to the corresponding [`ExifTag`] variant.
    #[must_use]
    pub fn from_code(code: u16) -> Self {
        match code {
            0x0100 => Self::ImageWidth,
            0x0101 => Self::ImageHeight,
            0x0102 => Self::BitsPerSample,
            0x0103 => Self::Compression,
            0x010E => Self::ImageDescription,
            0x010F => Self::Make,
            0x0110 => Self::Model,
            0x011A => Self::XResolution,
            0x011B => Self::YResolution,
            0x0131 => Self::Software,
            0x013B => Self::Artist,
            0x8298 => Self::Copyright,
            0x829A => Self::ExposureTime,
            0x829D => Self::FNumber,
            0x8827 => Self::IsoSpeedRatings,
            0x9003 => Self::DateTimeOriginal,
            0x9201 => Self::ShutterSpeedValue,
            0x9202 => Self::ApertureValue,
            0x920A => Self::FocalLength,
            other => Self::Unknown(other),
        }
    }

    /// Numeric IFD tag code for this variant.
    #[must_use]
    pub fn code(&self) -> u16 {
        match self {
            Self::ImageWidth => 0x0100,
            Self::ImageHeight => 0x0101,
            Self::BitsPerSample => 0x0102,
            Self::Compression => 0x0103,
            Self::ImageDescription => 0x010E,
            Self::Make => 0x010F,
            Self::Model => 0x0110,
            Self::XResolution => 0x011A,
            Self::YResolution => 0x011B,
            Self::Software => 0x0131,
            Self::Artist => 0x013B,
            Self::Copyright => 0x8298,
            Self::ExposureTime => 0x829A,
            Self::FNumber => 0x829D,
            Self::IsoSpeedRatings => 0x8827,
            Self::DateTimeOriginal => 0x9003,
            Self::ShutterSpeedValue => 0x9201,
            Self::ApertureValue => 0x9202,
            Self::FocalLength => 0x920A,
            Self::GpsLatitude => 0x0002,
            Self::GpsLongitude => 0x0004,
            Self::Unknown(c) => *c,
        }
    }
}

/// Typed value stored for one EXIF tag.
#[derive(Clone, Debug, PartialEq)]
pub enum ExifValue {
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Rational number stored as (numerator, denominator).
    Rational(u32, u32),
    /// ASCII string (NUL-terminated in the raw file; stored here as `String`).
    Ascii(String),
    /// List of rational values (e.g. GPS coordinates).
    RationalList(Vec<(u32, u32)>),
    /// Raw bytes for unrecognised or complex types.
    Raw(Vec<u8>),
}

impl ExifValue {
    /// Coerce the value to `f64` where semantically meaningful.
    ///
    /// Returns `None` for string and raw-byte variants.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::U16(v) => Some(f64::from(*v)),
            Self::U32(v) => Some(*v as f64),
            Self::Rational(n, d) => {
                if *d == 0 {
                    None
                } else {
                    Some(*n as f64 / *d as f64)
                }
            }
            _ => None,
        }
    }

    /// Return a string representation if the value is ASCII.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        if let Self::Ascii(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }
}

/// A single parsed EXIF entry (tag + value).
#[derive(Clone, Debug)]
pub struct ExifEntry {
    /// The tag identifier.
    pub tag: ExifTag,
    /// The parsed value.
    pub value: ExifValue,
}

impl ExifEntry {
    /// Construct an entry from a tag and value.
    #[must_use]
    pub fn new(tag: ExifTag, value: ExifValue) -> Self {
        Self { tag, value }
    }
}

/// Collection of parsed EXIF metadata for one image.
#[derive(Clone, Debug, Default)]
pub struct ExifData {
    entries: HashMap<u16, ExifEntry>,
}

impl ExifData {
    /// Create an empty [`ExifData`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite a tag entry.
    pub fn insert(&mut self, entry: ExifEntry) {
        self.entries.insert(entry.tag.code(), entry);
    }

    /// Look up a tag by its enum variant.
    #[must_use]
    pub fn get_tag(&self, tag: ExifTag) -> Option<&ExifEntry> {
        self.entries.get(&tag.code())
    }

    /// Look up a tag by its raw numeric code.
    #[must_use]
    pub fn get_by_code(&self, code: u16) -> Option<&ExifEntry> {
        self.entries.get(&code)
    }

    /// Return the number of parsed tags.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no tags have been parsed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &ExifEntry> {
        self.entries.values()
    }

    /// Convenience: return camera make string, if present.
    #[must_use]
    pub fn make(&self) -> Option<&str> {
        self.get_tag(ExifTag::Make)?.value.as_str()
    }

    /// Convenience: return camera model string, if present.
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.get_tag(ExifTag::Model)?.value.as_str()
    }

    /// Convenience: return ISO speed rating as u16, if present.
    #[must_use]
    pub fn iso(&self) -> Option<u16> {
        match self.get_tag(ExifTag::IsoSpeedRatings)?.value {
            ExifValue::U16(v) => Some(v),
            _ => None,
        }
    }

    /// Convenience: return focal length in mm as f64, if present.
    #[must_use]
    pub fn focal_length_mm(&self) -> Option<f64> {
        self.get_tag(ExifTag::FocalLength)?.value.as_f64()
    }
}

/// Minimal EXIF parser that builds [`ExifData`] from a byte slice.
///
/// Supports TIFF-style IFDs (the byte layout used inside JPEG APP1 segments
/// and standalone TIFF files).  This implementation handles a subset of
/// EXIF types sufficient for the most common camera metadata tags.
pub struct ExifParser;

impl ExifParser {
    /// Parse EXIF from a raw byte slice starting at the TIFF header offset.
    ///
    /// Returns an empty [`ExifData`] on any parse error rather than
    /// propagating errors, to allow best-effort metadata extraction.
    #[must_use]
    pub fn parse(data: &[u8]) -> ExifData {
        let mut exif = ExifData::new();
        if data.len() < 8 {
            return exif;
        }

        // Determine byte order.
        let big_endian = match &data[..2] {
            b"MM" => true,
            b"II" => false,
            _ => return exif,
        };

        // Verify TIFF magic (42).
        let magic = read_u16(&data[2..4], big_endian);
        if magic != 42 {
            return exif;
        }

        // Offset to first IFD.
        let ifd_offset = read_u32(&data[4..8], big_endian) as usize;
        Self::parse_ifd(data, ifd_offset, big_endian, &mut exif);
        exif
    }

    fn parse_ifd(data: &[u8], offset: usize, big_endian: bool, exif: &mut ExifData) {
        if offset + 2 > data.len() {
            return;
        }
        let count = read_u16(&data[offset..offset + 2], big_endian) as usize;
        let entries_start = offset + 2;

        for i in 0..count {
            let entry_start = entries_start + i * 12;
            if entry_start + 12 > data.len() {
                break;
            }
            let tag_code = read_u16(&data[entry_start..entry_start + 2], big_endian);
            let type_id = read_u16(&data[entry_start + 2..entry_start + 4], big_endian);
            let value_count = read_u32(&data[entry_start + 4..entry_start + 8], big_endian);
            let value_offset_raw = &data[entry_start + 8..entry_start + 12];

            let tag = ExifTag::from_code(tag_code);
            if let Some(value) =
                decode_value(data, type_id, value_count, value_offset_raw, big_endian)
            {
                exif.insert(ExifEntry::new(tag, value));
            }
        }
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn read_u16(bytes: &[u8], big_endian: bool) -> u16 {
    if big_endian {
        u16::from_be_bytes([bytes[0], bytes[1]])
    } else {
        u16::from_le_bytes([bytes[0], bytes[1]])
    }
}

fn read_u32(bytes: &[u8], big_endian: bool) -> u32 {
    if big_endian {
        u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    } else {
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

fn decode_value(
    data: &[u8],
    type_id: u16,
    count: u32,
    value_field: &[u8],
    big_endian: bool,
) -> Option<ExifValue> {
    match type_id {
        // SHORT (u16)
        3 => {
            let v = read_u16(value_field, big_endian);
            Some(ExifValue::U16(v))
        }
        // LONG (u32)
        4 => {
            let v = read_u32(value_field, big_endian);
            Some(ExifValue::U32(v))
        }
        // RATIONAL (two u32)
        5 => {
            let offset = read_u32(value_field, big_endian) as usize;
            if count == 1 {
                if offset + 8 > data.len() {
                    return None;
                }
                let n = read_u32(&data[offset..offset + 4], big_endian);
                let d = read_u32(&data[offset + 4..offset + 8], big_endian);
                Some(ExifValue::Rational(n, d))
            } else {
                // Defend against a `with_capacity` allocation bomb: the RATIONAL
                // list `count` (u32) is attacker-controlled. Bound the
                // reservation to what the remaining input can back (8 bytes per
                // entry); the loop's own `pos + 8 > data.len()` guard stops it.
                let cap = crate::limits::safe_capacity(
                    count as usize,
                    8,
                    data.len().saturating_sub(offset),
                );
                let mut list = Vec::with_capacity(cap);
                for i in 0..count as usize {
                    let pos = offset + i * 8;
                    if pos + 8 > data.len() {
                        break;
                    }
                    let n = read_u32(&data[pos..pos + 4], big_endian);
                    let d = read_u32(&data[pos + 4..pos + 8], big_endian);
                    list.push((n, d));
                }
                Some(ExifValue::RationalList(list))
            }
        }
        // ASCII
        2 => {
            let len = count as usize;
            let s = if len <= 4 {
                // Value fits in the 4-byte field.
                let raw = &value_field[..len.min(4)];
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                String::from_utf8_lossy(&raw[..end]).into_owned()
            } else {
                let offset = read_u32(value_field, big_endian) as usize;
                if offset + len > data.len() {
                    return None;
                }
                let raw = &data[offset..offset + len];
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                String::from_utf8_lossy(&raw[..end]).into_owned()
            };
            Some(ExifValue::Ascii(s))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(tag: ExifTag, value: ExifValue) -> ExifEntry {
        ExifEntry::new(tag, value)
    }

    #[test]
    fn test_exif_tag_from_code_known() {
        assert_eq!(ExifTag::from_code(0x010F), ExifTag::Make);
        assert_eq!(ExifTag::from_code(0x0110), ExifTag::Model);
        assert_eq!(ExifTag::from_code(0x8827), ExifTag::IsoSpeedRatings);
    }

    #[test]
    fn test_exif_tag_from_code_unknown() {
        assert_eq!(ExifTag::from_code(0xBEEF), ExifTag::Unknown(0xBEEF));
    }

    #[test]
    fn test_exif_tag_round_trip_code() {
        let tag = ExifTag::FocalLength;
        assert_eq!(ExifTag::from_code(tag.code()), tag);
    }

    #[test]
    fn test_exif_value_rational_as_f64() {
        let v = ExifValue::Rational(1, 100);
        assert!((v.as_f64().expect("should succeed in test") - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_exif_value_rational_div_zero() {
        let v = ExifValue::Rational(1, 0);
        assert!(v.as_f64().is_none());
    }

    #[test]
    fn test_exif_value_ascii_as_str() {
        let v = ExifValue::Ascii("Sony".to_string());
        assert_eq!(v.as_str(), Some("Sony"));
    }

    #[test]
    fn test_exif_data_insert_and_get() {
        let mut d = ExifData::new();
        d.insert(make_entry(
            ExifTag::Make,
            ExifValue::Ascii("Canon".to_string()),
        ));
        assert_eq!(
            d.get_tag(ExifTag::Make)
                .expect("should succeed in test")
                .value
                .as_str(),
            Some("Canon")
        );
    }

    #[test]
    fn test_exif_data_len() {
        let mut d = ExifData::new();
        assert_eq!(d.len(), 0);
        d.insert(make_entry(
            ExifTag::Model,
            ExifValue::Ascii("EOS R5".to_string()),
        ));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_exif_data_iso_convenience() {
        let mut d = ExifData::new();
        d.insert(make_entry(ExifTag::IsoSpeedRatings, ExifValue::U16(800)));
        assert_eq!(d.iso(), Some(800));
    }

    #[test]
    fn test_exif_data_make_and_model() {
        let mut d = ExifData::new();
        d.insert(make_entry(
            ExifTag::Make,
            ExifValue::Ascii("Nikon".to_string()),
        ));
        d.insert(make_entry(
            ExifTag::Model,
            ExifValue::Ascii("Z9".to_string()),
        ));
        assert_eq!(d.make(), Some("Nikon"));
        assert_eq!(d.model(), Some("Z9"));
    }

    #[test]
    fn test_exif_data_focal_length() {
        let mut d = ExifData::new();
        d.insert(make_entry(ExifTag::FocalLength, ExifValue::Rational(50, 1)));
        assert!((d.focal_length_mm().expect("should succeed in test") - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_exif_parser_empty_data_returns_empty() {
        let d = ExifParser::parse(&[]);
        assert!(d.is_empty());
    }

    #[test]
    fn test_exif_parser_wrong_magic_returns_empty() {
        let data = [0x00u8; 16];
        let d = ExifParser::parse(&data);
        assert!(d.is_empty());
    }

    #[test]
    fn test_exif_data_get_by_code() {
        let mut d = ExifData::new();
        d.insert(make_entry(
            ExifTag::Software,
            ExifValue::Ascii("Lightroom".to_string()),
        ));
        assert!(d.get_by_code(0x0131).is_some());
    }

    #[test]
    fn test_exif_data_iter_count() {
        let mut d = ExifData::new();
        d.insert(make_entry(
            ExifTag::Make,
            ExifValue::Ascii("Fuji".to_string()),
        ));
        d.insert(make_entry(
            ExifTag::Model,
            ExifValue::Ascii("X-T5".to_string()),
        ));
        assert_eq!(d.iter().count(), 2);
    }

    #[test]
    fn test_exif_value_u16_as_f64() {
        let v = ExifValue::U16(400);
        assert!((v.as_f64().expect("should succeed in test") - 400.0).abs() < f64::EPSILON);
    }
}

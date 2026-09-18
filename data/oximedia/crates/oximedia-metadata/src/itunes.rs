//! iTunes/MP4 metadata atom parsing and writing support.
//!
//! iTunes metadata is stored in MP4/M4A files using various atoms.
//!
//! # Common Atoms
//!
//! - **©nam**: Title
//! - **©ART**: Artist
//! - **©alb**: Album
//! - **aART**: Album Artist
//! - **©gen**: Genre
//! - **©day**: Year/Date
//! - **©cmt**: Comment
//! - **©wrt**: Composer
//! - **©too**: Encoder
//! - **cprt**: Copyright
//! - **covr**: Cover art

use crate::{Error, Metadata, MetadataFormat, MetadataValue, Picture, PictureType};
use std::io::{Cursor, Read};

// ---- iTunes atom data-type constants (well-known type indicators) ----

/// UTF-8 text data type.
const ITUNES_DATA_TYPE_UTF8: u32 = 1;

/// Integer data type (variable width: 1/2/4/8 bytes).
const ITUNES_DATA_TYPE_INTEGER: u32 = 21;

// ---- Well-known atom identifiers ----

/// Tempo (BPM). Stored as big-endian u16 inside a `data` sub-atom.
pub const ATOM_TEMPO: &str = "tmpo";

/// Compilation flag. Stored as a single byte (0 or 1).
pub const ATOM_COMPILATION: &str = "cpil";

/// Gapless playback flag. Stored as a single byte (0 or 1).
pub const ATOM_GAPLESS: &str = "pgap";

/// Disc number. Stored as 6-byte big-endian pair (disc, total).
pub const ATOM_DISC_NUMBER: &str = "disk";

/// Track number. Stored as 8-byte big-endian pair (track, total).
pub const ATOM_TRACK_NUMBER: &str = "trkn";

/// Media type. Stored as a single byte.
pub const ATOM_MEDIA_TYPE: &str = "stik";

/// Rating / advisory. Stored as a single byte (0=none, 1=explicit, 2=clean).
pub const ATOM_RATING: &str = "rtng";

/// Podcast flag. Stored as a single byte (0 or 1).
pub const ATOM_PODCAST: &str = "pcst";

/// Parse iTunes metadata from atom data.
///
/// # Errors
///
/// Returns an error if the data is not valid iTunes metadata.
pub fn parse(data: &[u8]) -> Result<Metadata, Error> {
    let mut metadata = Metadata::new(MetadataFormat::iTunes);
    let mut cursor = Cursor::new(data);

    while cursor.position() < data.len() as u64 {
        // Read atom size
        let size = read_u32_be(&mut cursor)?;
        if size < 8 {
            break;
        }

        // Read atom name
        let mut name_bytes = [0u8; 4];
        cursor
            .read_exact(&mut name_bytes)
            .map_err(|e| Error::ParseError(format!("Failed to read atom name: {e}")))?;

        let name = String::from_utf8_lossy(&name_bytes).to_string();

        // Read atom data
        let data_size = size.saturating_sub(8) as usize;
        let mut atom_data = vec![0u8; data_size];
        cursor
            .read_exact(&mut atom_data)
            .map_err(|e| Error::ParseError(format!("Failed to read atom data: {e}")))?;

        // Parse atom data
        let value = parse_atom(&name, &atom_data)?;
        metadata.insert(name, value);
    }

    Ok(metadata)
}

/// Parse a single atom.
fn parse_atom(name: &str, data: &[u8]) -> Result<MetadataValue, Error> {
    if data.len() < 8 {
        return Ok(MetadataValue::Binary(data.to_vec()));
    }

    // Parse the "data" sub-atom if present.
    // Format: 4-byte size + 4-byte type ("data") + 4-byte data-type + 4-byte locale + payload
    let (data_type, payload) = if data.len() >= 16 && &data[4..8] == b"data" {
        let dt = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        (Some(dt), &data[16..])
    } else if data.len() >= 8 && &data[0..4] == b"data" {
        // Simpler layout: "data" + 4 bytes flags/type, then payload at offset 8
        (None, &data[8..])
    } else {
        (None, data)
    };

    if payload.is_empty() {
        return Ok(MetadataValue::Binary(data.to_vec()));
    }

    // Cover art
    if name == "covr" {
        return parse_cover_art(payload);
    }

    // Integer / boolean atoms
    match name {
        ATOM_TEMPO => return parse_tempo_atom(payload),
        ATOM_COMPILATION | ATOM_GAPLESS | ATOM_PODCAST => return parse_boolean_atom(payload),
        ATOM_RATING | ATOM_MEDIA_TYPE => return parse_byte_integer_atom(payload),
        ATOM_TRACK_NUMBER => return parse_track_disc_atom(payload),
        ATOM_DISC_NUMBER => return parse_track_disc_atom(payload),
        _ => {}
    }

    // Well-known integer data type
    if data_type == Some(ITUNES_DATA_TYPE_INTEGER) {
        return parse_integer_atom(payload);
    }

    // Text atoms (UTF-8 or explicitly typed)
    if name.starts_with('\u{00A9}')
        || matches!(name, "aART" | "cprt")
        || data_type == Some(ITUNES_DATA_TYPE_UTF8)
    {
        return parse_text_atom(payload);
    }

    // Default: binary data
    Ok(MetadataValue::Binary(data.to_vec()))
}

/// Parse a tempo (BPM) atom -- big-endian u16.
fn parse_tempo_atom(data: &[u8]) -> Result<MetadataValue, Error> {
    if data.len() >= 2 {
        let bpm = u16::from_be_bytes([data[0], data[1]]);
        Ok(MetadataValue::Integer(i64::from(bpm)))
    } else if data.len() == 1 {
        Ok(MetadataValue::Integer(i64::from(data[0])))
    } else {
        Ok(MetadataValue::Integer(0))
    }
}

/// Parse a boolean atom (single byte: 0 = false, non-zero = true).
fn parse_boolean_atom(data: &[u8]) -> Result<MetadataValue, Error> {
    if data.is_empty() {
        return Ok(MetadataValue::Boolean(false));
    }
    Ok(MetadataValue::Boolean(data[0] != 0))
}

/// Parse a single-byte integer atom.
fn parse_byte_integer_atom(data: &[u8]) -> Result<MetadataValue, Error> {
    if data.is_empty() {
        return Ok(MetadataValue::Integer(0));
    }
    Ok(MetadataValue::Integer(i64::from(data[0])))
}

/// Parse track-number or disc-number atom (big-endian pairs).
fn parse_track_disc_atom(data: &[u8]) -> Result<MetadataValue, Error> {
    // Track: 8 bytes = 2 reserved + 2 track + 2 total + 2 reserved
    // Disc:  6 bytes = 2 reserved + 2 disc  + 2 total
    if data.len() >= 6 {
        let num = u16::from_be_bytes([data[2], data[3]]);
        let total = u16::from_be_bytes([data[4], data[5]]);
        if total > 0 {
            Ok(MetadataValue::Text(format!("{num}/{total}")))
        } else {
            Ok(MetadataValue::Integer(i64::from(num)))
        }
    } else {
        parse_integer_atom(data)
    }
}

/// Parse a generic integer atom (1/2/4/8 bytes, big-endian).
fn parse_integer_atom(data: &[u8]) -> Result<MetadataValue, Error> {
    let val = match data.len() {
        0 => 0i64,
        1 => i64::from(data[0]),
        2 => i64::from(i16::from_be_bytes([data[0], data[1]])),
        3 => i64::from(u32::from_be_bytes([0, data[0], data[1], data[2]])),
        4 => i64::from(i32::from_be_bytes([data[0], data[1], data[2], data[3]])),
        _ if data.len() >= 8 => i64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
        _ => {
            // Treat as big-endian unsigned, zero-extended
            let mut buf = [0u8; 8];
            let start = 8 - data.len();
            buf[start..].copy_from_slice(data);
            i64::from_be_bytes(buf)
        }
    };
    Ok(MetadataValue::Integer(val))
}

/// Parse text atom data.
fn parse_text_atom(data: &[u8]) -> Result<MetadataValue, Error> {
    let text = String::from_utf8(data.to_vec())
        .map_err(|e| Error::EncodingError(format!("Invalid UTF-8 in text atom: {e}")))?;
    Ok(MetadataValue::Text(text))
}

/// Parse cover art atom data.
fn parse_cover_art(data: &[u8]) -> Result<MetadataValue, Error> {
    // Detect MIME type from data
    let mime_type = detect_image_mime_type(data);

    let picture = Picture::new(
        mime_type.to_string(),
        PictureType::FrontCover,
        data.to_vec(),
    );

    Ok(MetadataValue::Picture(picture))
}

/// Detect image MIME type from data.
fn detect_image_mime_type(data: &[u8]) -> &'static str {
    if data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8 {
        "image/jpeg"
    } else if data.len() >= 8 && &data[0..8] == b"\x89PNG\r\n\x1a\n" {
        "image/png"
    } else if data.len() >= 2 && data[0] == 0x42 && data[1] == 0x4D {
        "image/bmp"
    } else {
        "application/octet-stream"
    }
}

/// Write iTunes metadata to atom data.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write(metadata: &Metadata) -> Result<Vec<u8>, Error> {
    let mut result = Vec::new();

    for (key, value) in metadata.fields() {
        let atom_data = write_atom(key, value)?;
        result.extend_from_slice(&atom_data);
    }

    Ok(result)
}

/// Write a single atom.
fn write_atom(name: &str, value: &MetadataValue) -> Result<Vec<u8>, Error> {
    if name.len() != 4 {
        return Err(Error::WriteError(format!(
            "Invalid atom name length: {name}"
        )));
    }

    let mut result = Vec::new();

    // Determine the data-type indicator and encode the payload.
    let (data_type_code, payload) = encode_atom_value(name, value)?;

    // Build the "data" sub-atom:
    //   4 bytes: sub-atom size (header + payload)
    //   4 bytes: "data"
    //   4 bytes: data-type indicator (big-endian)
    //   4 bytes: locale (0)
    //   N bytes: payload
    let sub_atom_size = (16 + payload.len()) as u32;
    let mut sub_atom = Vec::new();
    sub_atom.extend_from_slice(&sub_atom_size.to_be_bytes());
    sub_atom.extend_from_slice(b"data");
    sub_atom.extend_from_slice(&data_type_code.to_be_bytes());
    sub_atom.extend_from_slice(&[0u8; 4]); // locale
    sub_atom.extend_from_slice(&payload);

    // Write outer atom: size + name + sub-atom
    let outer_size = (8 + sub_atom.len()) as u32;
    result.extend_from_slice(&outer_size.to_be_bytes());
    result.extend_from_slice(name.as_bytes());
    result.extend_from_slice(&sub_atom);

    Ok(result)
}

/// Encode the value for a specific atom name.
///
/// Returns (data_type_code, encoded_payload).
fn encode_atom_value(name: &str, value: &MetadataValue) -> Result<(u32, Vec<u8>), Error> {
    match name {
        ATOM_TEMPO => {
            let bpm = match value {
                MetadataValue::Integer(i) => *i as u16,
                MetadataValue::Text(t) => t.parse::<u16>().unwrap_or(0),
                _ => 0,
            };
            Ok((ITUNES_DATA_TYPE_INTEGER, bpm.to_be_bytes().to_vec()))
        }
        ATOM_COMPILATION | ATOM_GAPLESS | ATOM_PODCAST => {
            let flag: u8 = match value {
                MetadataValue::Boolean(b) => u8::from(*b),
                MetadataValue::Integer(i) => u8::from(*i != 0),
                _ => 0,
            };
            Ok((ITUNES_DATA_TYPE_INTEGER, vec![flag]))
        }
        ATOM_RATING | ATOM_MEDIA_TYPE => {
            let byte_val: u8 = match value {
                MetadataValue::Integer(i) => *i as u8,
                _ => 0,
            };
            Ok((ITUNES_DATA_TYPE_INTEGER, vec![byte_val]))
        }
        _ => {
            // Default: text or binary
            match value {
                MetadataValue::Text(text) => Ok((ITUNES_DATA_TYPE_UTF8, text.as_bytes().to_vec())),
                MetadataValue::Integer(i) => {
                    Ok((ITUNES_DATA_TYPE_INTEGER, (*i as u32).to_be_bytes().to_vec()))
                }
                MetadataValue::Boolean(b) => Ok((ITUNES_DATA_TYPE_INTEGER, vec![u8::from(*b)])),
                MetadataValue::Picture(pic) => Ok((14, pic.data.clone())), // 14 = JPEG, 13 = PNG
                MetadataValue::Binary(data) => Ok((0, data.clone())),
                _ => Err(Error::WriteError(
                    "Unsupported value type for iTunes".to_string(),
                )),
            }
        }
    }
}

/// Read a 32-bit big-endian unsigned integer.
fn read_u32_be(cursor: &mut Cursor<&[u8]>) -> Result<u32, Error> {
    let mut bytes = [0u8; 4];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| Error::ParseError(format!("Failed to read u32: {e}")))?;
    Ok(u32::from_be_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_image_mime_type() {
        assert_eq!(
            detect_image_mime_type(&[0xFF, 0xD8, 0xFF, 0xE0]),
            "image/jpeg"
        );
        assert_eq!(
            detect_image_mime_type(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
            "image/png"
        );
        assert_eq!(detect_image_mime_type(&[0x42, 0x4D]), "image/bmp");
        assert_eq!(
            detect_image_mime_type(&[0x00, 0x00]),
            "application/octet-stream"
        );
    }

    #[test]
    fn test_itunes_text_atom_write() {
        let mut metadata = Metadata::new(MetadataFormat::iTunes);
        metadata.insert(
            "test".to_string(),
            MetadataValue::Text("Test Title".to_string()),
        );
        let data = write(&metadata).expect("Write failed");
        assert!(!data.is_empty());
    }

    // ------- Tempo / Compilation / Gapless tests -------

    /// Helper: build a minimal iTunes data atom for testing parse_atom().
    fn build_data_atom(data_type: u32, payload: &[u8]) -> Vec<u8> {
        let sub_size = (16 + payload.len()) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&sub_size.to_be_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_type.to_be_bytes());
        out.extend_from_slice(&[0u8; 4]); // locale
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn test_parse_tempo_atom() {
        let atom_data = build_data_atom(ITUNES_DATA_TYPE_INTEGER, &120u16.to_be_bytes());
        let value = parse_atom(ATOM_TEMPO, &atom_data).expect("parse should succeed");
        assert_eq!(value.as_integer(), Some(120));
    }

    #[test]
    fn test_parse_compilation_true() {
        let atom_data = build_data_atom(ITUNES_DATA_TYPE_INTEGER, &[1]);
        let value = parse_atom(ATOM_COMPILATION, &atom_data).expect("parse should succeed");
        assert_eq!(value.as_boolean(), Some(true));
    }

    #[test]
    fn test_parse_compilation_false() {
        let atom_data = build_data_atom(ITUNES_DATA_TYPE_INTEGER, &[0]);
        let value = parse_atom(ATOM_COMPILATION, &atom_data).expect("parse should succeed");
        assert_eq!(value.as_boolean(), Some(false));
    }

    #[test]
    fn test_parse_gapless_true() {
        let atom_data = build_data_atom(ITUNES_DATA_TYPE_INTEGER, &[1]);
        let value = parse_atom(ATOM_GAPLESS, &atom_data).expect("parse should succeed");
        assert_eq!(value.as_boolean(), Some(true));
    }

    #[test]
    fn test_parse_rating_atom() {
        let atom_data = build_data_atom(ITUNES_DATA_TYPE_INTEGER, &[1]); // explicit
        let value = parse_atom(ATOM_RATING, &atom_data).expect("parse should succeed");
        assert_eq!(value.as_integer(), Some(1));
    }

    #[test]
    fn test_parse_media_type_atom() {
        let atom_data = build_data_atom(ITUNES_DATA_TYPE_INTEGER, &[6]); // music video
        let value = parse_atom(ATOM_MEDIA_TYPE, &atom_data).expect("parse should succeed");
        assert_eq!(value.as_integer(), Some(6));
    }

    #[test]
    fn test_write_tempo_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::iTunes);
        metadata.insert(ATOM_TEMPO.to_string(), MetadataValue::Integer(140));

        let data = write(&metadata).expect("write should succeed");
        let parsed = parse(&data).expect("parse should succeed");

        assert_eq!(
            parsed.get(ATOM_TEMPO).and_then(|v| v.as_integer()),
            Some(140)
        );
    }

    #[test]
    fn test_write_compilation_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::iTunes);
        metadata.insert(ATOM_COMPILATION.to_string(), MetadataValue::Boolean(true));

        let data = write(&metadata).expect("write should succeed");
        let parsed = parse(&data).expect("parse should succeed");

        assert_eq!(
            parsed.get(ATOM_COMPILATION).and_then(|v| v.as_boolean()),
            Some(true)
        );
    }

    #[test]
    fn test_write_gapless_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::iTunes);
        metadata.insert(ATOM_GAPLESS.to_string(), MetadataValue::Boolean(true));

        let data = write(&metadata).expect("write should succeed");
        let parsed = parse(&data).expect("parse should succeed");

        assert_eq!(
            parsed.get(ATOM_GAPLESS).and_then(|v| v.as_boolean()),
            Some(true)
        );
    }

    #[test]
    fn test_write_gapless_false_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::iTunes);
        metadata.insert(ATOM_GAPLESS.to_string(), MetadataValue::Boolean(false));

        let data = write(&metadata).expect("write should succeed");
        let parsed = parse(&data).expect("parse should succeed");

        assert_eq!(
            parsed.get(ATOM_GAPLESS).and_then(|v| v.as_boolean()),
            Some(false)
        );
    }

    #[test]
    fn test_atom_constants() {
        assert_eq!(ATOM_TEMPO, "tmpo");
        assert_eq!(ATOM_COMPILATION, "cpil");
        assert_eq!(ATOM_GAPLESS, "pgap");
        assert_eq!(ATOM_DISC_NUMBER, "disk");
        assert_eq!(ATOM_TRACK_NUMBER, "trkn");
        assert_eq!(ATOM_MEDIA_TYPE, "stik");
        assert_eq!(ATOM_RATING, "rtng");
        assert_eq!(ATOM_PODCAST, "pcst");
    }

    #[test]
    fn test_parse_integer_atom_various_sizes() {
        // 1 byte
        assert_eq!(
            parse_integer_atom(&[42]).expect("ok").as_integer(),
            Some(42)
        );
        // 2 bytes
        assert_eq!(
            parse_integer_atom(&[0, 100]).expect("ok").as_integer(),
            Some(100)
        );
        // 4 bytes
        assert_eq!(
            parse_integer_atom(&1000i32.to_be_bytes())
                .expect("ok")
                .as_integer(),
            Some(1000)
        );
        // 0 bytes
        assert_eq!(parse_integer_atom(&[]).expect("ok").as_integer(), Some(0));
    }

    #[test]
    fn test_encode_atom_value_tempo() {
        let (dt, payload) =
            encode_atom_value(ATOM_TEMPO, &MetadataValue::Integer(128)).expect("ok");
        assert_eq!(dt, ITUNES_DATA_TYPE_INTEGER);
        assert_eq!(payload, 128u16.to_be_bytes());
    }

    #[test]
    fn test_encode_atom_value_text() {
        let (dt, payload) =
            encode_atom_value("test", &MetadataValue::Text("hello".to_string())).expect("ok");
        assert_eq!(dt, ITUNES_DATA_TYPE_UTF8);
        assert_eq!(payload, b"hello");
    }
}

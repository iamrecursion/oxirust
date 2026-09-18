//! Minimal AMF0 encoder for FLV `onMetaData` script tags.
//!
//! AMF0 is a binary serialisation format defined by Adobe.  Only the subset
//! required to write a well-formed FLV `onMetaData` script tag is implemented
//! here; full AMF0/AMF3 parsing is out of scope for the muxer.
//!
//! # Wire format overview
//!
//! | Marker byte | Type         | Payload                                   |
//! |-------------|--------------|-------------------------------------------|
//! | 0x00        | Number       | 8-byte big-endian IEEE 754 double          |
//! | 0x01        | Boolean      | 1-byte (0 = false, 1 = true)              |
//! | 0x02        | String       | 2-byte BE length + UTF-8 bytes            |
//! | 0x03        | Object       | property pairs + 0x00 0x00 0x09 terminator|
//! | 0x08        | ECMA Array   | 4-byte BE count + property pairs + terminator |

#![forbid(unsafe_code)]

// ============================================================================
// AMF0 marker constants
// ============================================================================

/// AMF0 Number type marker.
const AMF0_NUMBER: u8 = 0x00;
/// AMF0 Boolean type marker.
const AMF0_BOOLEAN: u8 = 0x01;
/// AMF0 String type marker.
const AMF0_STRING: u8 = 0x02;
/// AMF0 ECMA Array type marker.
const AMF0_ECMA_ARRAY: u8 = 0x08;
/// AMF0 object terminator (0x00 0x00 followed by 0x09 end-of-object marker).
const AMF0_OBJECT_END: [u8; 3] = [0x00, 0x00, 0x09];

// ============================================================================
// Low-level encoders
// ============================================================================

/// Appends an AMF0-encoded UTF-8 string (marker included).
fn push_amf0_string(out: &mut Vec<u8>, s: &str) {
    out.push(AMF0_STRING);
    push_amf0_utf8(out, s);
}

/// Appends a bare AMF0 UTF-8 key (2-byte BE length + bytes, no marker).
/// Used for property names inside objects/arrays.
fn push_amf0_utf8(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len() as u16;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
}

/// Appends an AMF0-encoded IEEE 754 double (marker included).
fn push_amf0_number(out: &mut Vec<u8>, v: f64) {
    out.push(AMF0_NUMBER);
    out.extend_from_slice(&v.to_be_bytes());
}

/// Appends an AMF0-encoded boolean (marker included).
fn push_amf0_boolean(out: &mut Vec<u8>, v: bool) {
    out.push(AMF0_BOOLEAN);
    out.push(u8::from(v));
}

/// Appends one ECMA-array property: bare UTF-8 key + typed value.
fn push_property_number(out: &mut Vec<u8>, key: &str, v: f64) {
    push_amf0_utf8(out, key);
    push_amf0_number(out, v);
}

/// Appends one ECMA-array boolean property: bare UTF-8 key + boolean value.
fn push_property_boolean(out: &mut Vec<u8>, key: &str, v: bool) {
    push_amf0_utf8(out, key);
    push_amf0_boolean(out, v);
}

// ============================================================================
// Public API
// ============================================================================

/// Writes an AMF0-encoded `onMetaData` body suitable for embedding in an FLV
/// type-18 (Script) tag.
///
/// The returned `Vec<u8>` encodes:
///
/// 1. An AMF0 String: `"onMetaData"`
/// 2. An AMF0 ECMA Array containing seven properties: `duration`, `width`,
///    `height`, `videodatarate`, `framerate`, `hasVideo`, `hasAudio`.
///
/// All numeric values default to `0.0` when unknown; boolean values are
/// encoded as AMF0 Booleans.
#[must_use]
pub fn write_on_metadata(
    duration: f64,
    width: f64,
    height: f64,
    video_data_rate: f64,
    frame_rate: f64,
    has_video: bool,
    has_audio: bool,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(128);

    // 1. AMF0 String: "onMetaData"
    push_amf0_string(&mut out, "onMetaData");

    // 2. AMF0 ECMA Array with 7 known properties.
    out.push(AMF0_ECMA_ARRAY);
    // 4-byte BE count (must match the number of properties that follow).
    out.extend_from_slice(&7u32.to_be_bytes());

    push_property_number(&mut out, "duration", duration);
    push_property_number(&mut out, "width", width);
    push_property_number(&mut out, "height", height);
    push_property_number(&mut out, "videodatarate", video_data_rate);
    push_property_number(&mut out, "framerate", frame_rate);
    push_property_boolean(&mut out, "hasVideo", has_video);
    push_property_boolean(&mut out, "hasAudio", has_audio);

    // Object terminator.
    out.extend_from_slice(&AMF0_OBJECT_END);

    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_on_metadata_starts_with_string_marker() {
        let buf = write_on_metadata(0.0, 1920.0, 1080.0, 0.0, 30.0, true, true);
        // First byte must be AMF0 String marker.
        assert_eq!(buf[0], AMF0_STRING);
        // Next two bytes: length of "onMetaData" = 10.
        assert_eq!(buf[1], 0x00);
        assert_eq!(buf[2], 0x0A);
        // Followed by the literal string "onMetaData".
        assert_eq!(&buf[3..13], b"onMetaData");
    }

    #[test]
    fn test_on_metadata_ecma_array_marker() {
        let buf = write_on_metadata(0.0, 0.0, 0.0, 0.0, 0.0, false, false);
        // After the 13-byte AMF0 String header, ECMA Array marker.
        assert_eq!(buf[13], AMF0_ECMA_ARRAY);
        // Count: 7 properties.
        assert_eq!(&buf[14..18], &7u32.to_be_bytes());
    }

    #[test]
    fn test_on_metadata_terminates_with_object_end() {
        let buf = write_on_metadata(1.5, 320.0, 240.0, 500.0, 25.0, true, false);
        let tail = &buf[buf.len() - 3..];
        assert_eq!(tail, &AMF0_OBJECT_END);
    }

    #[test]
    fn test_on_metadata_non_empty() {
        let buf = write_on_metadata(0.0, 0.0, 0.0, 0.0, 0.0, false, false);
        assert!(!buf.is_empty());
    }
}

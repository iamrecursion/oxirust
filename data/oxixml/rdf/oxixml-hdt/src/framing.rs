//! The small "metadata preamble + CRC8" / "payload + CRC32C" framing pattern that HDT reuses
//! for every packed data structure (dictionary section buffers, bitmaps, integer arrays).
//!
//! Per the specification: *"[the metadata] is always preceded by a metadata preamble ... [and]
//! followed by a CRC32 of the [payload]"*. This module factors that pattern out of
//! [`crate::bitarray`] and [`crate::bitmap`] so both encode/decode it identically.

use crate::crc::{crc8, crc32c};
use crate::error::{HdtError, Result};
use crate::vbyte::{read_vbyte, write_vbyte};

/// Appends a metadata preamble: a `format` byte, each of `fields` as a VByte, then a CRC8 of
/// everything just written.
pub fn write_metadata(out: &mut Vec<u8>, format: u8, fields: &[u64]) {
    let start = out.len();
    out.push(format);
    for &field in fields {
        write_vbyte(out, field);
    }
    out.push(crc8(&out[start..]));
}

/// Reads a metadata preamble written by [`write_metadata`]: a format byte followed by exactly
/// `num_fields` VByte-encoded fields and a verifying CRC8.
///
/// # Errors
///
/// Returns [`HdtError::UnexpectedEof`] if the input is exhausted early, or
/// [`HdtError::CrcMismatch`] if the trailing CRC8 does not match.
pub fn read_metadata(
    data: &[u8],
    pos: &mut usize,
    section: &'static str,
    num_fields: usize,
) -> Result<(u8, Vec<u64>)> {
    let start = *pos;
    let format = *data.get(*pos).ok_or(HdtError::UnexpectedEof { section })?;
    *pos += 1;

    let mut fields = Vec::with_capacity(num_fields);
    for _ in 0..num_fields {
        fields.push(read_vbyte(data, pos)?);
    }

    let end = *pos;
    let computed = u32::from(crc8(&data[start..end]));
    let stored = *data.get(*pos).ok_or(HdtError::UnexpectedEof { section })?;
    *pos += 1;
    if u32::from(stored) != computed {
        return Err(HdtError::CrcMismatch {
            kind: "CRC8",
            section,
            expected: u32::from(stored),
            actual: computed,
        });
    }

    Ok((format, fields))
}

/// Appends `payload` verbatim, followed by a little-endian CRC32C of `payload`.
pub fn write_payload(out: &mut Vec<u8>, payload: &[u8]) {
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc32c(payload).to_le_bytes());
}

/// Reads back a payload of exactly `len` bytes written by [`write_payload`], verifying its
/// trailing CRC32C.
///
/// # Errors
///
/// Returns [`HdtError::UnexpectedEof`] if fewer than `len + 4` bytes remain, or
/// [`HdtError::CrcMismatch`] if the trailing CRC32C does not match.
pub fn read_payload<'a>(
    data: &'a [u8],
    pos: &mut usize,
    section: &'static str,
    len: usize,
) -> Result<&'a [u8]> {
    let start = *pos;
    let end = start
        .checked_add(len)
        .ok_or(HdtError::UnexpectedEof { section })?;
    let payload = data
        .get(start..end)
        .ok_or(HdtError::UnexpectedEof { section })?;
    *pos = end;

    let computed = crc32c(payload);
    let crc_bytes = data
        .get(*pos..*pos + 4)
        .ok_or(HdtError::UnexpectedEof { section })?;
    let stored = u32::from_le_bytes(
        crc_bytes
            .try_into()
            .expect("slice of length 4 converts into [u8; 4]"),
    );
    *pos += 4;
    if stored != computed {
        return Err(HdtError::CrcMismatch {
            kind: "CRC32C",
            section,
            expected: stored,
            actual: computed,
        });
    }

    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_roundtrips() -> Result<()> {
        let mut buffer = Vec::new();
        write_metadata(&mut buffer, 1, &[42, 1000]);
        let mut pos = 0;
        let (format, fields) = read_metadata(&buffer, &mut pos, "test", 2)?;
        assert_eq!(format, 1);
        assert_eq!(fields, vec![42, 1000]);
        assert_eq!(pos, buffer.len());
        Ok(())
    }

    #[test]
    fn metadata_detects_corruption() {
        let mut buffer = Vec::new();
        write_metadata(&mut buffer, 1, &[42]);
        *buffer.last_mut().expect("non-empty") ^= 0xFF;
        let mut pos = 0;
        let error = read_metadata(&buffer, &mut pos, "test", 1).expect_err("expected an error");
        assert!(matches!(error, HdtError::CrcMismatch { kind: "CRC8", .. }));
    }

    #[test]
    fn payload_roundtrips() -> Result<()> {
        let mut buffer = Vec::new();
        write_payload(&mut buffer, b"hello, hdt");
        let mut pos = 0;
        let payload = read_payload(&buffer, &mut pos, "test", 10)?;
        assert_eq!(payload, b"hello, hdt");
        assert_eq!(pos, buffer.len());
        Ok(())
    }

    #[test]
    fn payload_detects_corruption() {
        let mut buffer = Vec::new();
        write_payload(&mut buffer, b"hello, hdt");
        buffer[0] ^= 0xFF;
        let mut pos = 0;
        let error = read_payload(&buffer, &mut pos, "test", 10).expect_err("expected an error");
        assert!(matches!(
            error,
            HdtError::CrcMismatch { kind: "CRC32C", .. }
        ));
    }
}

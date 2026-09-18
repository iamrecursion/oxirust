//! VByte variable-length integer encoding, as used throughout HDT for lengths and counts.
//!
//! Following [Williams and Zobel, *Compressing integers for fast file access*, 1999] (cited by
//! the HDT specification) and mirroring the reference `hdt-cpp` implementation
//! (`libhdt/src/libdcs/VByte.cpp`) byte-for-byte: a value is split into 7-bit groups,
//! least-significant group first. Every byte but the last carries a group in its low 7 bits
//! with the high bit clear (`0` = "more bytes follow"); the *last* byte carries the most
//! significant remaining bits with the high bit **set** (`1` = "this is the terminating
//! byte"). This is the mirror image of LEB128 (which sets the continuation bit on
//! non-terminal bytes) — do not confuse the two.
//!
//! ```text
//! 300 = 0b1_0010_1100
//!   byte 0: 0b0010_1100 = 0x2C  (low 7 bits,  high bit clear: more follows)
//!   byte 1: 0b1000_0010 = 0x82  (remaining 2 bits, high bit set: done)
//! ```

use crate::error::{HdtError, Result};

/// The section name reported in [`HdtError::UnexpectedEof`] by [`read_vbyte`].
const SECTION: &str = "vbyte";

/// Appends the VByte encoding of `value` to `out`.
pub fn write_vbyte(out: &mut Vec<u8>, value: u64) {
    let mut remaining = value;
    while remaining > 0x7F {
        out.push((remaining & 0x7F) as u8);
        remaining >>= 7;
    }
    out.push((remaining as u8) | 0x80);
}

/// Returns the number of bytes [`write_vbyte`] would emit for `value`, without allocating.
#[must_use]
pub const fn vbyte_len(value: u64) -> usize {
    let mut remaining = value;
    let mut len = 1;
    while remaining > 0x7F {
        remaining >>= 7;
        len += 1;
    }
    len
}

/// Reads a VByte-encoded integer from `data` starting at `*pos`, advancing `*pos` past it.
///
/// # Errors
///
/// Returns [`HdtError::UnexpectedEof`] if `data` is exhausted before a terminating byte (one
/// with the high bit set) is found, and [`HdtError::Malformed`] if the encoded value would
/// overflow a [`u64`] (more than 10 groups of 7 bits).
pub fn read_vbyte(data: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = *data
            .get(*pos)
            .ok_or(HdtError::UnexpectedEof { section: SECTION })?;
        *pos += 1;
        if shift >= 64 {
            return Err(HdtError::Malformed(
                "VByte-encoded integer overflows a 64-bit value".to_owned(),
            ));
        }
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 != 0 {
            return Ok(value);
        }
        shift += 7;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(value: u64) {
        let mut buffer = Vec::new();
        write_vbyte(&mut buffer, value);
        assert_eq!(buffer.len(), vbyte_len(value));
        let mut pos = 0;
        let decoded = read_vbyte(&buffer, &mut pos).expect("decode should succeed");
        assert_eq!(decoded, value);
        assert_eq!(pos, buffer.len());
    }

    #[test]
    fn roundtrips_edge_values() -> Result<()> {
        for value in [
            0u64,
            1,
            0x7F,
            0x80,
            300,
            u32::MAX as u64,
            u64::MAX,
            u64::MAX - 1,
        ] {
            roundtrip(value);
        }
        Ok(())
    }

    #[test]
    fn matches_hdt_cpp_worked_example() {
        // 300 = 0b1_0010_1100 -> [0b0010_1100, 0b1000_0010] = [0x2C, 0x82]
        let mut buffer = Vec::new();
        write_vbyte(&mut buffer, 300);
        assert_eq!(buffer, vec![0x2C, 0x82]);
    }

    #[test]
    fn single_byte_for_small_values() {
        let mut buffer = Vec::new();
        write_vbyte(&mut buffer, 42);
        assert_eq!(buffer, vec![42 | 0x80]);
    }

    #[test]
    fn truncated_input_is_unexpected_eof() {
        // High bit never set: the reader should run off the end of the buffer.
        let buffer = [0x01_u8, 0x02];
        let mut pos = 0;
        let error = read_vbyte(&buffer, &mut pos).expect_err("expected an error");
        assert!(matches!(
            error,
            HdtError::UnexpectedEof { section: "vbyte" }
        ));
    }

    #[test]
    fn sequential_reads_advance_position() -> Result<()> {
        let mut buffer = Vec::new();
        write_vbyte(&mut buffer, 1);
        write_vbyte(&mut buffer, 128);
        write_vbyte(&mut buffer, 16384);
        let mut pos = 0;
        assert_eq!(read_vbyte(&buffer, &mut pos)?, 1);
        assert_eq!(read_vbyte(&buffer, &mut pos)?, 128);
        assert_eq!(read_vbyte(&buffer, &mut pos)?, 16384);
        assert_eq!(pos, buffer.len());
        Ok(())
    }
}

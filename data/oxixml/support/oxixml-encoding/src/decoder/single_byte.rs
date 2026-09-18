//! Decoding for every single-byte encoding this crate supports: the original three
//! ([`Latin1`](crate::Encoding::Latin1), [`Windows1252`](crate::Encoding::Windows1252) and
//! [`Ascii`](crate::Encoding::Ascii)) plus the 26 legacy single-byte encodings dispatched through
//! [`crate::Encoding::is_legacy_single_byte_table`].
//!
//! None of these can ever have a sequence split across a chunk boundary (every byte decodes to
//! exactly one character on its own), so unlike [`super::utf8`] and [`super::utf16`] there is no
//! carried state at all — every function here is a pure, stateless function of its `input`.

use super::{DecodeError, DecodeErrorKind};
use crate::label::Encoding;
use crate::tables::{
    LATIN1_CODEPOINTS, WINDOWS_1252_CODEPOINTS, decode_single_byte, decode_single_byte_upper,
    legacy_single_byte_table,
};

/// Decode `input` through a 256-entry codepoint table. Infallible: every byte in `0x00..=0xFF`
/// has an entry in both [`LATIN1_CODEPOINTS`] and [`WINDOWS_1252_CODEPOINTS`].
fn decode_via_table(table: &[u32; 256], input: &[u8], dst: &mut String) -> usize {
    dst.reserve(input.len());
    for &byte in input {
        dst.push(decode_single_byte(table, byte));
    }
    input.len()
}

/// Strict ISO-8859-1 decode. Always succeeds.
pub(crate) fn decode_latin1(input: &[u8], dst: &mut String) -> Result<usize, DecodeError> {
    Ok(decode_via_table(&LATIN1_CODEPOINTS, input, dst))
}

/// Lossy ISO-8859-1 decode. Identical to [`decode_latin1`] since it is already infallible.
pub(crate) fn decode_latin1_lossy(input: &[u8], dst: &mut String) -> usize {
    decode_via_table(&LATIN1_CODEPOINTS, input, dst)
}

/// Strict windows-1252 decode. Always succeeds (the five nominally-unassigned bytes fall back to
/// their Latin-1 code point; see [`crate::tables::WINDOWS_1252_CODEPOINTS`]).
pub(crate) fn decode_windows1252(input: &[u8], dst: &mut String) -> Result<usize, DecodeError> {
    Ok(decode_via_table(&WINDOWS_1252_CODEPOINTS, input, dst))
}

/// Lossy windows-1252 decode. Identical to [`decode_windows1252`] since it is already infallible.
pub(crate) fn decode_windows1252_lossy(input: &[u8], dst: &mut String) -> usize {
    decode_via_table(&WINDOWS_1252_CODEPOINTS, input, dst)
}

/// Strict US-ASCII decode: bytes `0x80..=0xFF` are rejected.
pub(crate) fn decode_ascii(input: &[u8], dst: &mut String) -> Result<usize, DecodeError> {
    match input.iter().position(|&b| b >= 0x80) {
        None => {
            // The whole chunk is 7-bit clean, hence trivially valid UTF-8 too.
            if let Ok(s) = core::str::from_utf8(input) {
                dst.push_str(s);
            }
            Ok(input.len())
        }
        Some(bad_at) => {
            if let Ok(s) = core::str::from_utf8(&input[..bad_at]) {
                dst.push_str(s);
            }
            Err(DecodeError {
                offset: bad_at,
                length: 1,
                kind: DecodeErrorKind::DisallowedByte,
            })
        }
    }
}

/// Lossy US-ASCII decode: bytes `0x80..=0xFF` become `U+FFFD`, one replacement per byte.
pub(crate) fn decode_ascii_lossy(input: &[u8], dst: &mut String) -> usize {
    dst.reserve(input.len());
    for &byte in input {
        if byte < 0x80 {
            dst.push(byte as char);
        } else {
            dst.push(char::REPLACEMENT_CHARACTER);
        }
    }
    input.len()
}

/// Strict decode for one of the 26 legacy single-byte encodings backed by a 128-entry upper-half
/// table (see [`crate::Encoding::is_legacy_single_byte_table`]). Bytes `0x00..=0x7F` are always
/// ASCII; bytes `0x80..=0xFF` are looked up in `encoding`'s table, and a byte with no assigned
/// code point (per the WHATWG single-byte decoder algorithm, §9.1: "If codePoint is null, then
/// return error") is reported as [`DecodeErrorKind::DisallowedByte`].
pub(crate) fn decode_legacy_single_byte(
    encoding: Encoding,
    input: &[u8],
    dst: &mut String,
) -> Result<usize, DecodeError> {
    let table = legacy_single_byte_table(encoding);
    dst.reserve(input.len());
    for (i, &byte) in input.iter().enumerate() {
        if byte < 0x80 {
            dst.push(byte as char);
            continue;
        }
        match decode_single_byte_upper(table, byte) {
            Some(c) => dst.push(c),
            None => {
                return Err(DecodeError {
                    offset: i,
                    length: 1,
                    kind: DecodeErrorKind::DisallowedByte,
                });
            }
        }
    }
    Ok(input.len())
}

/// Lossy counterpart of [`decode_legacy_single_byte`]: an unassigned byte becomes `U+FFFD`.
pub(crate) fn decode_legacy_single_byte_lossy(
    encoding: Encoding,
    input: &[u8],
    dst: &mut String,
) -> usize {
    let table = legacy_single_byte_table(encoding);
    dst.reserve(input.len());
    for &byte in input {
        if byte < 0x80 {
            dst.push(byte as char);
        } else {
            dst.push(decode_single_byte_upper(table, byte).unwrap_or(char::REPLACEMENT_CHARACTER));
        }
    }
    input.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin1_decodes_every_byte_identically() -> Result<(), DecodeError> {
        let all_bytes: Vec<u8> = (0..=255).collect();
        let mut out = String::new();
        decode_latin1(&all_bytes, &mut out)?;
        let expected: String = (0..=255u32)
            .map(|b| char::from_u32(b).expect("valid"))
            .collect();
        assert_eq!(out, expected);
        Ok(())
    }

    #[test]
    fn windows1252_euro_sign() -> Result<(), DecodeError> {
        let mut out = String::new();
        decode_windows1252(&[0x80], &mut out)?;
        assert_eq!(out, "\u{20AC}");
        Ok(())
    }

    #[test]
    fn windows1252_unassigned_slots_fall_back_to_latin1() -> Result<(), DecodeError> {
        let mut out = String::new();
        decode_windows1252(&[0x81, 0x8D, 0x8F, 0x90, 0x9D], &mut out)?;
        assert_eq!(out, "\u{81}\u{8D}\u{8F}\u{90}\u{9D}");
        Ok(())
    }

    #[test]
    fn ascii_accepts_0x00_to_0x7f() -> Result<(), DecodeError> {
        let bytes: Vec<u8> = (0..=0x7Fu8).collect();
        let mut out = String::new();
        decode_ascii(&bytes, &mut out)?;
        assert_eq!(out.len(), 128);
        assert!(out.is_ascii());
        Ok(())
    }

    #[test]
    fn ascii_rejects_high_byte_with_offset() {
        let mut out = String::new();
        let err = decode_ascii(b"ok\x80rest", &mut out).expect_err("must fail");
        assert_eq!(out, "ok");
        assert_eq!(err.offset, 2);
        assert_eq!(err.kind, DecodeErrorKind::DisallowedByte);
    }

    #[test]
    fn ascii_lossy_replaces_every_high_byte() {
        let mut out = String::new();
        decode_ascii_lossy(b"a\xFFb\x80c", &mut out);
        assert_eq!(out, "a\u{FFFD}b\u{FFFD}c");
    }

    #[test]
    fn legacy_single_byte_decodes_ascii_range_identically_for_every_table() {
        let bytes: Vec<u8> = (0..0x80u8).collect();
        for &encoding in Encoding::ALL {
            if !encoding.is_legacy_single_byte_table() {
                continue;
            }
            let mut out = String::new();
            decode_legacy_single_byte(encoding, &bytes, &mut out)
                .unwrap_or_else(|e| panic!("{encoding}: {e}"));
            assert!(out.is_ascii(), "{encoding}");
            assert_eq!(out.len(), 128, "{encoding}");
        }
    }

    #[test]
    fn legacy_single_byte_windows_1250_letter_a_with_ogonek() -> Result<(), DecodeError> {
        // windows-1250 0xA5 == Ą (LATIN CAPITAL LETTER A WITH OGONEK), independently confirmed
        // against the system `iconv`'s CP1250 table while generating these tables.
        let mut out = String::new();
        decode_legacy_single_byte(Encoding::Windows1250, &[0xA5], &mut out)?;
        assert_eq!(out, "\u{0104}");
        Ok(())
    }

    #[test]
    fn legacy_single_byte_koi8_r_cyrillic_a() -> Result<(), DecodeError> {
        // KOI8-R's layout does not mirror Unicode's cyrillic block: 0xC1 == а (CYRILLIC SMALL
        // LETTER A, lowercase); the uppercase form (U+0410) is at 0xE1. Both independently
        // confirmed against the system `iconv`'s KOI8-R table while writing this test.
        let mut out = String::new();
        decode_legacy_single_byte(Encoding::Koi8R, &[0xC1, 0xE1], &mut out)?;
        assert_eq!(out, "\u{0430}\u{0410}");
        Ok(())
    }

    #[test]
    fn legacy_single_byte_rejects_unmapped_byte_with_offset() {
        // ISO-8859-6 (Arabic) leaves many bytes in 0xA0..=0xFF unassigned; 0xA1 is one of them.
        let mut out = String::new();
        let err = decode_legacy_single_byte(Encoding::Iso8859_6, b"ok\xA1rest", &mut out)
            .expect_err("must fail");
        assert_eq!(out, "ok");
        assert_eq!(err.offset, 2);
        assert_eq!(err.length, 1);
        assert_eq!(err.kind, DecodeErrorKind::DisallowedByte);
    }

    #[test]
    fn legacy_single_byte_lossy_replaces_unmapped_bytes() {
        let mut out = String::new();
        let consumed =
            decode_legacy_single_byte_lossy(Encoding::Iso8859_6, b"ok\xA1rest", &mut out);
        assert_eq!(consumed, b"ok\xA1rest".len());
        assert_eq!(out, "ok\u{FFFD}rest");
    }

    #[test]
    fn legacy_single_byte_lossy_never_fails_on_arbitrary_bytes() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        for &encoding in Encoding::ALL {
            if !encoding.is_legacy_single_byte_table() {
                continue;
            }
            let mut out = String::new();
            let consumed = decode_legacy_single_byte_lossy(encoding, &all_bytes, &mut out);
            assert_eq!(consumed, 256, "{encoding}");
            assert!(out.chars().count() >= 1, "{encoding}");
        }
    }
}

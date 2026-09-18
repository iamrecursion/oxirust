//! Encoding UTF-8 text (Rust's native string representation) into a legacy byte encoding.
//!
//! This is the direction an XML *writer* needs: given a `&str` (always valid UTF-8, since
//! that's what every Rust string is) and a target [`Encoding`], produce the bytes that encoding
//! would decode back to that text — or, if some character has no representation in the target
//! encoding, report exactly which one and where.

use crate::cjk::{big5, euc_kr, gb18030};
use crate::jis::{euc_jp, iso2022jp, shift_jis};
use crate::label::Encoding;
use crate::tables::{
    LATIN1_CODEPOINTS, WINDOWS_1252_CODEPOINTS, encode_single_byte, encode_single_byte_upper,
    legacy_single_byte_table,
};

/// An error reported by [`encode_from_utf8`]: `character` has no representation in the
/// requested target encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeError {
    /// The character that could not be represented in the target encoding.
    pub character: char,
    /// The byte offset of `character` within the source `&str` passed to
    /// [`encode_from_utf8`] (suitable for indexing/slicing that same `&str`).
    pub position: usize,
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "character {:?} (U+{:04X}) at byte offset {} has no representation in the target encoding",
            self.character, self.character as u32, self.position
        )
    }
}

impl std::error::Error for EncodeError {}

/// Encode `input` into `encoding`, appending the resulting bytes to `dst`.
///
/// [`Encoding::Utf8`] and the two UTF-16 variants can represent every Rust `char` (i.e. every
/// Unicode scalar value) and therefore never fail. [`Encoding::Gb18030`] is *almost* total in the
/// same sense — every scalar value except `U+E5E5` has a representation (see the type's own docs
/// for why that one code point is excluded) — but every other encoding, including
/// [`Encoding::Latin1`], [`Encoding::Windows1252`], [`Encoding::Ascii`], and the remaining legacy
/// multi-byte encodings ([`Encoding::Gbk`], [`Encoding::Big5`], [`Encoding::EucKr`], and the three
/// legacy Japanese ones), can only represent a subset of Unicode; the first character outside that
/// subset is reported as an [`EncodeError`]. Characters encoded before the failing one have
/// already been appended to `dst`, so partial output is available even after an error (useful
/// for producing a best-effort diagnostic, or for switching strategies — e.g. falling back to a
/// numeric character reference — and resuming from `error.position`).
///
/// # Errors
///
/// Returns [`EncodeError`] if `input` contains a character with no representation in `encoding`.
///
/// # Examples
///
/// ```
/// use oxixml_encoding::{encode_from_utf8, Encoding};
///
/// let mut dst = Vec::new();
/// encode_from_utf8("café", Encoding::Latin1, &mut dst).expect("café is representable in Latin-1");
/// assert_eq!(dst, b"caf\xE9");
///
/// let mut dst = Vec::new();
/// let err = encode_from_utf8("日本語", Encoding::Ascii, &mut dst).unwrap_err();
/// assert_eq!(err.character, '日');
/// assert_eq!(err.position, 0);
/// ```
pub fn encode_from_utf8(
    input: &str,
    encoding: Encoding,
    dst: &mut Vec<u8>,
) -> Result<(), EncodeError> {
    match encoding {
        Encoding::Utf8 => {
            dst.extend_from_slice(input.as_bytes());
            Ok(())
        }
        Encoding::Utf16Le => {
            encode_utf16(input, dst, false);
            Ok(())
        }
        Encoding::Utf16Be => {
            encode_utf16(input, dst, true);
            Ok(())
        }
        Encoding::Latin1 => encode_single_byte_str(input, dst, &LATIN1_CODEPOINTS),
        Encoding::Windows1252 => encode_single_byte_str(input, dst, &WINDOWS_1252_CODEPOINTS),
        Encoding::Ascii => encode_ascii(input, dst),
        Encoding::ShiftJis => shift_jis::encode(input, dst),
        Encoding::EucJp => euc_jp::encode(input, dst),
        Encoding::Iso2022Jp => iso2022jp::encode(input, dst),
        Encoding::Gbk => gb18030::encode(input, dst, true),
        Encoding::Gb18030 => gb18030::encode(input, dst, false),
        Encoding::Big5 => big5::encode(input, dst),
        Encoding::EucKr => euc_kr::encode(input, dst),
        // Every remaining variant is one of the 26 legacy single-byte encodings.
        other => encode_legacy_single_byte_str(input, dst, other),
    }
}

/// Encode `input` into one of the 26 legacy single-byte encodings backed by a 128-entry
/// upper-half table (see [`Encoding::is_legacy_single_byte_table`]).
fn encode_legacy_single_byte_str(
    input: &str,
    dst: &mut Vec<u8>,
    encoding: Encoding,
) -> Result<(), EncodeError> {
    let table = legacy_single_byte_table(encoding);
    dst.reserve(input.len());
    for (position, character) in input.char_indices() {
        match encode_single_byte_upper(table, character as u32) {
            Some(byte) => dst.push(byte),
            None => {
                return Err(EncodeError {
                    character,
                    position,
                });
            }
        }
    }
    Ok(())
}

/// Every Unicode scalar value is representable in UTF-16 (that is the entire point of the
/// surrogate mechanism), so this never fails.
fn encode_utf16(input: &str, dst: &mut Vec<u8>, big_endian: bool) {
    dst.reserve(input.len() * 2);
    for unit in input.encode_utf16() {
        let bytes = if big_endian {
            unit.to_be_bytes()
        } else {
            unit.to_le_bytes()
        };
        dst.extend_from_slice(&bytes);
    }
}

fn encode_single_byte_str(
    input: &str,
    dst: &mut Vec<u8>,
    table: &[u32; 256],
) -> Result<(), EncodeError> {
    dst.reserve(input.len());
    for (position, character) in input.char_indices() {
        match encode_single_byte(table, character as u32) {
            Some(byte) => dst.push(byte),
            None => {
                return Err(EncodeError {
                    character,
                    position,
                });
            }
        }
    }
    Ok(())
}

fn encode_ascii(input: &str, dst: &mut Vec<u8>) -> Result<(), EncodeError> {
    dst.reserve(input.len());
    for (position, character) in input.char_indices() {
        if character.is_ascii() {
            dst.push(character as u8);
        } else {
            return Err(EncodeError {
                character,
                position,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_target_is_a_plain_copy() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode_from_utf8("héllo→🎉", Encoding::Utf8, &mut dst)?;
        assert_eq!(dst, "héllo→🎉".as_bytes());
        Ok(())
    }

    #[test]
    fn utf16_targets_never_fail_even_for_astral_characters() -> Result<(), EncodeError> {
        let mut le = Vec::new();
        encode_from_utf8("🎉", Encoding::Utf16Le, &mut le)?;
        assert_eq!(
            le,
            "🎉"
                .encode_utf16()
                .flat_map(|u| u.to_le_bytes())
                .collect::<Vec<_>>()
        );

        let mut be = Vec::new();
        encode_from_utf8("🎉", Encoding::Utf16Be, &mut be)?;
        assert_eq!(
            be,
            "🎉"
                .encode_utf16()
                .flat_map(|u| u.to_be_bytes())
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn latin1_round_trips_within_0x00_to_0xff() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode_from_utf8("café", Encoding::Latin1, &mut dst)?;
        assert_eq!(dst, b"caf\xE9");
        Ok(())
    }

    #[test]
    fn latin1_rejects_out_of_range_character() {
        let mut dst = Vec::new();
        let err = encode_from_utf8("→", Encoding::Latin1, &mut dst).expect_err("must fail");
        assert_eq!(err.character, '→');
        assert_eq!(err.position, 0);
    }

    #[test]
    fn windows1252_encodes_smart_quotes() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode_from_utf8("\u{2018}hi\u{2019}", Encoding::Windows1252, &mut dst)?;
        assert_eq!(dst, [0x91, b'h', b'i', 0x92]);
        Ok(())
    }

    #[test]
    fn windows1252_rejects_character_outside_the_repertoire() {
        let mut dst = Vec::new();
        let err = encode_from_utf8("あ", Encoding::Windows1252, &mut dst).expect_err("must fail");
        assert_eq!(err.character, 'あ');
    }

    #[test]
    fn ascii_rejects_first_non_ascii_character_with_byte_position() {
        let mut dst = Vec::new();
        let err = encode_from_utf8("ok日本語", Encoding::Ascii, &mut dst).expect_err("must fail");
        assert_eq!(dst, b"ok");
        assert_eq!(err.character, '日');
        assert_eq!(err.position, 2);
    }

    #[test]
    fn error_display_is_human_readable() {
        let err = EncodeError {
            character: 'あ',
            position: 5,
        };
        assert_eq!(
            err.to_string(),
            "character 'あ' (U+3042) at byte offset 5 has no representation in the target encoding"
        );
    }

    #[test]
    fn legacy_single_byte_encodes_within_repertoire() -> Result<(), EncodeError> {
        // windows-1250 0xA5 == Ą (LATIN CAPITAL LETTER A WITH OGONEK).
        let mut dst = Vec::new();
        encode_from_utf8("\u{0104}", Encoding::Windows1250, &mut dst)?;
        assert_eq!(dst, vec![0xA5]);
        Ok(())
    }

    #[test]
    fn legacy_single_byte_rejects_character_outside_the_repertoire() {
        let mut dst = Vec::new();
        let err = encode_from_utf8("日", Encoding::Iso8859_2, &mut dst).expect_err("must fail");
        assert_eq!(err.character, '日');
        assert_eq!(err.position, 0);
    }

    #[test]
    fn every_legacy_single_byte_encoding_round_trips_its_own_mapped_bytes() {
        for &encoding in Encoding::ALL {
            if !encoding.is_legacy_single_byte_table() {
                continue;
            }
            let table = legacy_single_byte_table(encoding);
            for pointer in 0u8..128 {
                let codepoint = table[pointer as usize];
                if codepoint == crate::tables::UNMAPPED {
                    continue;
                }
                let Some(c) = char::from_u32(codepoint) else {
                    continue;
                };
                let mut dst = Vec::new();
                let text = c.to_string();
                encode_from_utf8(&text, encoding, &mut dst)
                    .unwrap_or_else(|e| panic!("{encoding}: {text:?}: {e}"));
                assert_eq!(dst, vec![0x80 + pointer], "{encoding}: {text:?}");
            }
        }
    }

    #[test]
    fn shift_jis_euc_jp_iso2022jp_are_routed_to_the_jis_module() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode_from_utf8("日", Encoding::ShiftJis, &mut dst)?;
        assert_eq!(dst, vec![0x93, 0xFA]);

        let mut dst = Vec::new();
        encode_from_utf8("日", Encoding::EucJp, &mut dst)?;
        assert_eq!(dst, vec![0xC6, 0xFC]);

        let mut dst = Vec::new();
        encode_from_utf8("日", Encoding::Iso2022Jp, &mut dst)?;
        assert_eq!(dst, vec![0x1B, 0x24, 0x42, 0x46, 0x7C, 0x1B, 0x28, 0x42]);
        Ok(())
    }

    #[test]
    fn gbk_gb18030_big5_euc_kr_are_routed_to_the_cjk_module() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode_from_utf8("中", Encoding::Gbk, &mut dst)?;
        assert_eq!(dst, vec![0xD6, 0xD0]);

        let mut dst = Vec::new();
        encode_from_utf8("中", Encoding::Gb18030, &mut dst)?;
        assert_eq!(dst, vec![0xD6, 0xD0]);

        let mut dst = Vec::new();
        encode_from_utf8("中", Encoding::Big5, &mut dst)?;
        assert_eq!(dst, vec![0xA4, 0xA4]);

        let mut dst = Vec::new();
        encode_from_utf8("한", Encoding::EucKr, &mut dst)?;
        assert_eq!(dst, vec![0xC7, 0xD1]);
        Ok(())
    }
}

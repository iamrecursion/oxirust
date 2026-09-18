//! Lookup tables for every encoding this crate supports beyond UTF-8/UTF-16.
//!
//! [`LATIN1_CODEPOINTS`] and [`WINDOWS_1252_CODEPOINTS`] are the two original 256-entry tables
//! (kept exactly as they were: full `0x00..=0xFF` coverage, since that predates the rest of this
//! module and downstream code in [`crate::decoder::single_byte`] and [`crate::encoder`] already
//! depends on their exact shape). Every other single-byte encoding is instead represented as a
//! 128-entry *upper-half* table (bytes `0x80..=0xFF` only, at index `byte - 0x80`) in
//! [`legacy_single_byte`], since bytes `0x00..=0x7F` are ASCII in all of them and tabulating that
//! identity half again would just be waste. See [`legacy_single_byte`] for the data sources.
//!
//! The three Japanese multi-byte encodings (Shift_JIS, EUC-JP, ISO-2022-JP) share the JIS X 0208
//! index ([`jis0208_decode`], [`jis0208_encode_full`], [`jis0208_encode_shift_jis`]); EUC-JP
//! additionally uses the JIS X 0212 index ([`jis0212_decode`]) for its `0x8F`-prefixed (SS3)
//! sequences, and ISO-2022-JP's encoder additionally uses [`iso2022jp_katakana`]. The actual
//! decoder/encoder state machines for those three live in [`crate::jis`]; this module only holds
//! their generated data and the small amount of lookup logic that is naturally table-shaped.
//!
//! The four CJK legacy encodings (GBK, gb18030, Big5, EUC-KR) follow the same split: their
//! generated indexes live here ([`gb18030_decode`], [`gb18030_encode`], [`gb18030_ranges`],
//! [`big5_decode`], [`big5_encode`], [`euc_kr_decode`], [`euc_kr_encode`]), and their
//! decoder/encoder state machines live in [`crate::cjk`].

mod big5_decode;
mod big5_encode;
mod euc_kr_decode;
mod euc_kr_encode;
mod gb18030_decode;
mod gb18030_encode;
mod gb18030_ranges;
mod iso2022jp_katakana;
mod jis0208_decode;
mod jis0208_encode_full;
mod jis0208_encode_shift_jis;
mod jis0212_decode;
mod legacy_single_byte;

pub(crate) use iso2022jp_katakana::ISO2022JP_KATAKANA;
pub(crate) use jis0208_decode::JIS0208_DECODE;
pub(crate) use jis0208_encode_full::JIS0208_ENCODE_FULL;
pub(crate) use jis0208_encode_shift_jis::JIS0208_ENCODE_SHIFT_JIS;
pub(crate) use jis0212_decode::JIS0212_DECODE;

use big5_decode::BIG5_DECODE;
use big5_encode::BIG5_ENCODE;
use euc_kr_decode::EUC_KR_DECODE;
use euc_kr_encode::EUC_KR_ENCODE;
use gb18030_decode::GB18030_DECODE;
use gb18030_encode::GB18030_ENCODE;
use gb18030_ranges::GB18030_RANGES;

use crate::label::Encoding;

/// Sentinel stored in a generated table entry to mean "this pointer/byte has no assigned code
/// point in this encoding". Never a valid Unicode scalar value (the maximum, `U+10FFFF`, is far
/// below this), so it can share the tables' plain `u32` element type without an `Option` wrapper
/// (which would not be usable in a `const` array initializer built by straight transcription).
pub(crate) const UNMAPPED: u32 = u32::MAX;

/// ISO-8859-1 (Latin-1) decode table: the identity mapping, byte `b` decodes to `U+00xx`.
///
/// Kept as an explicit table (rather than special-cased as `char::from(byte)`) so that the
/// decoder and encoder can share the exact same generic lookup/reverse-lookup code path used
/// for [`WINDOWS_1252_CODEPOINTS`].
pub(crate) const LATIN1_CODEPOINTS: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        table[i] = i as u32;
        i += 1;
    }
    table
};

/// windows-1252 decode table.
///
/// Bytes `0x00..=0x7F` and `0xA0..=0xFF` agree with [`LATIN1_CODEPOINTS`]; bytes `0x80..=0x9F`
/// are reassigned to the punctuation/typography block windows-1252 is best known for. The five
/// positions Microsoft left formally unassigned in that block (`0x81`, `0x8D`, `0x8F`, `0x90`,
/// `0x9D`) fall back to their Latin-1 (C1 control) code point, matching the "best fit" mapping
/// used by the reference `CP1252.TXT` table and by every other production windows-1252
/// implementation.
pub(crate) const WINDOWS_1252_CODEPOINTS: [u32; 256] = [
    0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x000A, 0x000B,
    0x000C, 0x000D, 0x000E, 0x000F, 0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017,
    0x0018, 0x0019, 0x001A, 0x001B, 0x001C, 0x001D, 0x001E, 0x001F, 0x0020, 0x0021, 0x0022, 0x0023,
    0x0024, 0x0025, 0x0026, 0x0027, 0x0028, 0x0029, 0x002A, 0x002B, 0x002C, 0x002D, 0x002E, 0x002F,
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037, 0x0038, 0x0039, 0x003A, 0x003B,
    0x003C, 0x003D, 0x003E, 0x003F, 0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
    0x0048, 0x0049, 0x004A, 0x004B, 0x004C, 0x004D, 0x004E, 0x004F, 0x0050, 0x0051, 0x0052, 0x0053,
    0x0054, 0x0055, 0x0056, 0x0057, 0x0058, 0x0059, 0x005A, 0x005B, 0x005C, 0x005D, 0x005E, 0x005F,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067, 0x0068, 0x0069, 0x006A, 0x006B,
    0x006C, 0x006D, 0x006E, 0x006F, 0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E, 0x007F,
    // 0x80..=0x9F: the windows-1252-specific block.
    0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039,
    0x0152, 0x008D, 0x017D, 0x008F, 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
    0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178,
    // 0xA0..=0xFF: identical to Latin-1.
    0x00A0, 0x00A1, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7, 0x00A8, 0x00A9, 0x00AA, 0x00AB,
    0x00AC, 0x00AD, 0x00AE, 0x00AF, 0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7,
    0x00B8, 0x00B9, 0x00BA, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF, 0x00C0, 0x00C1, 0x00C2, 0x00C3,
    0x00C4, 0x00C5, 0x00C6, 0x00C7, 0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF,
    0x00D0, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x00D7, 0x00D8, 0x00D9, 0x00DA, 0x00DB,
    0x00DC, 0x00DD, 0x00DE, 0x00DF, 0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7,
    0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF, 0x00F0, 0x00F1, 0x00F2, 0x00F3,
    0x00F4, 0x00F5, 0x00F6, 0x00F7, 0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x00FE, 0x00FF,
];

/// Convert a table entry (always a valid Unicode scalar value by construction, for the 256-entry
/// tables which have no [`UNMAPPED`] sentinel) to a [`char`], falling back to the replacement
/// character in the unreachable case that a future edit to one of the tables above introduces an
/// invalid entry.
#[inline]
pub(crate) fn codepoint_to_char(codepoint: u32) -> char {
    char::from_u32(codepoint).unwrap_or(char::REPLACEMENT_CHARACTER)
}

/// Decode a single byte through one of the two 256-entry tables above.
#[inline]
pub(crate) fn decode_single_byte(table: &[u32; 256], byte: u8) -> char {
    codepoint_to_char(table[byte as usize])
}

/// Reverse-lookup: find the byte that `table` decodes to `scalar`, if any.
///
/// Bytes `0x00..=0x7F` decode identically (as plain ASCII) in every encoding these tables back,
/// so that half is resolved in `O(1)` without touching the table. The upper half is found by a
/// linear scan over the remaining 128 entries, which is more than fast enough for the encoder's
/// use case (writing legacy-encoded XML is not a hot path, and non-ASCII characters are the
/// exception rather than the rule in most documents).
#[inline]
pub(crate) fn encode_single_byte(table: &[u32; 256], scalar: u32) -> Option<u8> {
    if scalar < 0x80 {
        return Some(scalar as u8);
    }
    table[0x80..=0xFF]
        .iter()
        .position(|&cp| cp == scalar)
        .map(|offset| (0x80 + offset) as u8)
}

/// Decode a byte `0x80..=0xFF` through one of the 128-entry upper-half tables in
/// [`legacy_single_byte`], returning [`None`] for a byte with no assigned code point.
///
/// # Panics
///
/// Panics (in a debug build; wraps in release) if `byte < 0x80` — every caller is expected to
/// have already special-cased the ASCII range `0x00..=0x7F` before reaching here, exactly as
/// [`encode_single_byte_upper`] special-cases it on the way back.
#[inline]
pub(crate) fn decode_single_byte_upper(table: &[u32; 128], byte: u8) -> Option<char> {
    let pointer = byte - 0x80;
    let codepoint = table[pointer as usize];
    if codepoint == UNMAPPED {
        None
    } else {
        Some(codepoint_to_char(codepoint))
    }
}

/// Reverse-lookup for a 128-entry upper-half table: find the byte `0x80..=0xFF` that decodes to
/// `scalar`, if any. See [`encode_single_byte`] for the rationale behind a linear scan here.
#[inline]
pub(crate) fn encode_single_byte_upper(table: &[u32; 128], scalar: u32) -> Option<u8> {
    if scalar < 0x80 {
        return Some(scalar as u8);
    }
    table
        .iter()
        .position(|&cp| cp == scalar)
        .map(|offset| (0x80 + offset) as u8)
}

/// The 128-entry upper-half decode table for one of the legacy single-byte encodings this crate
/// supports beyond [`Encoding::Latin1`]/[`Encoding::Windows1252`]/[`Encoding::Ascii`].
///
/// This `match` enumerates every [`Encoding`] variant explicitly instead of falling back to a
/// wildcard arm, so that adding a new variant to the enum without updating this function is a
/// compile error (a "non-exhaustive patterns" diagnostic naming the missing variant) rather than
/// a debug-only assertion that no-ops in release and leaves the panic below for a future caller
/// to hit at runtime. [`crate::decoder::DecoderKind::LegacySingleByte`] is the only caller, and
/// it is constructed only for the single-byte-table-backed variants (see `Decoder::new`, which
/// enumerates the same 26 variants for the same reason), so the panic itself stays unreachable
/// today; the exhaustiveness is what keeps it that way as the enum grows.
///
/// # Panics
///
/// Panics if `encoding` is not one of the 26 encodings backed by a table in
/// [`legacy_single_byte`]. This is a crate-internal invariant, not something a caller of the
/// public API can trigger.
pub(crate) fn legacy_single_byte_table(encoding: Encoding) -> &'static [u32; 128] {
    use legacy_single_byte::*;
    match encoding {
        Encoding::Ibm866 => &IBM866_UPPER,
        Encoding::Iso8859_2 => &ISO_8859_2_UPPER,
        Encoding::Iso8859_3 => &ISO_8859_3_UPPER,
        Encoding::Iso8859_4 => &ISO_8859_4_UPPER,
        Encoding::Iso8859_5 => &ISO_8859_5_UPPER,
        Encoding::Iso8859_6 => &ISO_8859_6_UPPER,
        Encoding::Iso8859_7 => &ISO_8859_7_UPPER,
        Encoding::Iso8859_8 => &ISO_8859_8_UPPER,
        Encoding::Iso8859_9 => &ISO_8859_9_UPPER,
        Encoding::Iso8859_10 => &ISO_8859_10_UPPER,
        Encoding::Iso8859_13 => &ISO_8859_13_UPPER,
        Encoding::Iso8859_14 => &ISO_8859_14_UPPER,
        Encoding::Iso8859_15 => &ISO_8859_15_UPPER,
        Encoding::Iso8859_16 => &ISO_8859_16_UPPER,
        Encoding::Koi8R => &KOI8_R_UPPER,
        Encoding::Koi8U => &KOI8_U_UPPER,
        Encoding::Macintosh => &MACINTOSH_UPPER,
        Encoding::Windows1250 => &WINDOWS_1250_UPPER,
        Encoding::Windows1251 => &WINDOWS_1251_UPPER,
        Encoding::Windows1253 => &WINDOWS_1253_UPPER,
        Encoding::Windows1254 => &WINDOWS_1254_UPPER,
        Encoding::Windows1255 => &WINDOWS_1255_UPPER,
        Encoding::Windows1256 => &WINDOWS_1256_UPPER,
        Encoding::Windows1257 => &WINDOWS_1257_UPPER,
        Encoding::Windows1258 => &WINDOWS_1258_UPPER,
        Encoding::XMacCyrillic => &X_MAC_CYRILLIC_UPPER,
        // Every one of these predates this table (handled by the 256-entry tables above) or is
        // multi-byte (handled entirely under `crate::jis`); named explicitly, rather than via a
        // wildcard, so the compiler rejects a new `Encoding` variant left out of this match.
        Encoding::Utf8
        | Encoding::Utf16Le
        | Encoding::Utf16Be
        | Encoding::Latin1
        | Encoding::Windows1252
        | Encoding::Ascii
        | Encoding::ShiftJis
        | Encoding::EucJp
        | Encoding::Iso2022Jp
        | Encoding::Gbk
        | Encoding::Gb18030
        | Encoding::Big5
        | Encoding::EucKr => unreachable!(
            "legacy_single_byte_table called for non-single-byte-table encoding {encoding:?}"
        ),
    }
}

/// The JIS X 0208 code point for `pointer`, or [`None`] if `pointer` is out of range or
/// unassigned. Shared by the Shift_JIS, EUC-JP, and ISO-2022-JP decoders (`crate::jis`), each of
/// which computes `pointer` from its own encoded bytes by its own arithmetic.
#[inline]
pub(crate) fn jis0208_decode(pointer: u32) -> Option<char> {
    let index = usize::try_from(pointer).ok()?;
    let codepoint = *JIS0208_DECODE.get(index)?;
    if codepoint == UNMAPPED {
        None
    } else {
        Some(codepoint_to_char(codepoint))
    }
}

/// The JIS X 0212 code point for `pointer`, or [`None`] if `pointer` is out of range or
/// unassigned. Used only by the EUC-JP decoder's `0x8F`-prefixed (SS3) sequences.
#[inline]
pub(crate) fn jis0212_decode(pointer: u32) -> Option<char> {
    let index = usize::try_from(pointer).ok()?;
    let codepoint = *JIS0212_DECODE.get(index)?;
    if codepoint == UNMAPPED {
        None
    } else {
        Some(codepoint_to_char(codepoint))
    }
}

/// The first (smallest) JIS X 0208 pointer for `scalar`, using the *unexcluded* index. Used by
/// the EUC-JP and ISO-2022-JP encoders.
#[inline]
pub(crate) fn jis0208_encode_full(scalar: u32) -> Option<u16> {
    JIS0208_ENCODE_FULL
        .binary_search_by_key(&scalar, |&(cp, _)| cp)
        .ok()
        .map(|i| JIS0208_ENCODE_FULL[i].1)
}

/// The first (smallest) JIS X 0208 pointer for `scalar` among entries outside the IBM-extension
/// range `8272..=8835`. Used only by the Shift_JIS encoder; see
/// [`jis0208_encode_shift_jis`](mod@jis0208_encode_shift_jis) for why.
#[inline]
pub(crate) fn jis0208_encode_shift_jis(scalar: u32) -> Option<u16> {
    JIS0208_ENCODE_SHIFT_JIS
        .binary_search_by_key(&scalar, |&(cp, _)| cp)
        .ok()
        .map(|i| JIS0208_ENCODE_SHIFT_JIS[i].1)
}

/// The replacement code point for a half-width katakana code point (`U+FF61..=U+FF9F`), per
/// [`ISO2022JP_KATAKANA`]. Used only by the ISO-2022-JP encoder, immediately before a JIS X 0208
/// pointer lookup of the *replacement*.
///
/// # Panics
///
/// Never: `ISO2022JP_KATAKANA` has exactly `0xFF9F - 0xFF61 + 1 == 63` entries, matching the
/// full range of `codepoint - 0xFF61` for a `codepoint` already known to be in that range.
#[inline]
pub(crate) fn iso2022jp_katakana_replacement(codepoint: u32) -> u32 {
    let index = (codepoint - 0xFF61) as usize;
    ISO2022JP_KATAKANA[index]
}

/// The code point for `pointer` in the GB18030/GBK 2-byte region, or [`None`] if `pointer` is
/// out of range or unassigned. Shared by the GBK and gb18030 decoders (`crate::cjk::gb18030`),
/// which are the same decoder per the WHATWG Encoding Standard ("GBK's decoder is gb18030's
/// decoder").
#[inline]
pub(crate) fn gb18030_decode_2byte(pointer: u32) -> Option<char> {
    let index = usize::try_from(pointer).ok()?;
    let codepoint = *GB18030_DECODE.get(index)?;
    if codepoint == UNMAPPED {
        None
    } else {
        Some(codepoint_to_char(codepoint))
    }
}

/// The first (smallest) GB18030/GBK 2-byte pointer for `scalar`, or [`None`] if `scalar` has no
/// 2-byte representation. Shared by the gb18030 and GBK encoders; a GBK encode additionally
/// falls back to `0x80` for U+20AC and errors instead of falling through to the 4-byte ranges
/// below on a miss here (`crate::cjk::gb18030::encode`).
#[inline]
pub(crate) fn gb18030_encode_2byte(scalar: u32) -> Option<u16> {
    GB18030_ENCODE
        .binary_search_by_key(&scalar, |&(cp, _)| cp)
        .ok()
        .map(|i| GB18030_ENCODE[i].1)
}

/// The gb18030 4-byte algorithmic ("ranges") code point for `pointer`, per the WHATWG "index
/// gb18030 ranges code point" algorithm. [`None`] if `pointer` falls in the gap between the BMP
/// and supplementary-plane ranges, is beyond `U+10FFFF`'s pointer, or (unreachable in practice,
/// since [`GB18030_RANGES`] is exhaustively range-tested) would compute a value outside the
/// Unicode scalar value space (e.g. a surrogate).
#[inline]
pub(crate) fn gb18030_ranges_decode(pointer: u32) -> Option<char> {
    if (pointer > 39419 && pointer < 189000) || pointer > 1_237_575 {
        return None;
    }
    if pointer == 7457 {
        // GB18030-2022 revision override: this pointer's linear-interpolation result was
        // reassigned without shifting every later range, so it is special-cased directly by the
        // spec rather than being a range boundary of its own.
        return Some('\u{E7C7}');
    }
    // `GB18030_RANGES` always has an entry at pointer 0, so `partition_point` here is always
    // `>= 1` for any `pointer` that reached this point (`pointer: u32` is never negative).
    let i = GB18030_RANGES.partition_point(|&(p, _)| p <= pointer) - 1;
    let (pointer_offset, codepoint_offset) = GB18030_RANGES[i];
    char::from_u32(codepoint_offset + (pointer - pointer_offset))
}

/// The gb18030 4-byte algorithmic ("ranges") pointer for `scalar`, per the WHATWG "index
/// gb18030 ranges pointer" algorithm. [`None`] only if `scalar` is below the first range's code
/// point (`U+0080`) -- unreachable in practice, since every caller has already tried an ASCII
/// and a 2-byte-index encode first, and gb18030 is total over every remaining Unicode scalar
/// value.
#[inline]
pub(crate) fn gb18030_ranges_encode(scalar: u32) -> Option<u32> {
    if scalar == 0xE7C7 {
        return Some(7457);
    }
    let i = GB18030_RANGES.partition_point(|&(_, cp)| cp <= scalar);
    if i == 0 {
        return None;
    }
    let (pointer_offset, codepoint_offset) = GB18030_RANGES[i - 1];
    Some(pointer_offset + (scalar - codepoint_offset))
}

/// The Big5 code point for `pointer`, or [`None`] if `pointer` is out of range or unassigned.
/// Does not cover the four pointers (1133, 1135, 1164, 1166) that decode to *two* code points --
/// a table entry can only hold one, so `crate::cjk::big5` special-cases those directly, exactly
/// as the WHATWG Big5 decoder algorithm does.
#[inline]
pub(crate) fn big5_decode(pointer: u32) -> Option<char> {
    let index = usize::try_from(pointer).ok()?;
    let codepoint = *BIG5_DECODE.get(index)?;
    if codepoint == UNMAPPED {
        None
    } else {
        Some(codepoint_to_char(codepoint))
    }
}

/// The Big5 pointer for `scalar`, or [`None`] if `scalar` has no Big5 representation. Per the
/// WHATWG "index Big5 pointer" algorithm: [`BIG5_ENCODE`] is generated already excluding the
/// low-pointer Hong Kong Supplementary Character Set duplicates and already preferring the
/// *last* pointer for the handful of code points the spec calls out by name.
#[inline]
pub(crate) fn big5_encode(scalar: u32) -> Option<u16> {
    BIG5_ENCODE
        .binary_search_by_key(&scalar, |&(cp, _)| cp)
        .ok()
        .map(|i| BIG5_ENCODE[i].1)
}

/// The code point for `pointer` in the EUC-KR index, or [`None`] if `pointer` is out of range or
/// unassigned.
#[inline]
pub(crate) fn euc_kr_decode(pointer: u32) -> Option<char> {
    let index = usize::try_from(pointer).ok()?;
    let codepoint = *EUC_KR_DECODE.get(index)?;
    if codepoint == UNMAPPED {
        None
    } else {
        Some(codepoint_to_char(codepoint))
    }
}

/// The first (smallest) EUC-KR pointer for `scalar`, or [`None`] if `scalar` has no EUC-KR
/// representation.
#[inline]
pub(crate) fn euc_kr_encode(scalar: u32) -> Option<u16> {
    EUC_KR_ENCODE
        .binary_search_by_key(&scalar, |&(cp, _)| cp)
        .ok()
        .map(|i| EUC_KR_ENCODE[i].1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin1_table_is_identity() {
        for b in 0..=255u8 {
            assert_eq!(LATIN1_CODEPOINTS[b as usize], b as u32);
        }
    }

    #[test]
    fn windows_1252_agrees_with_latin1_outside_c1_block() {
        for b in 0..0x80usize {
            assert_eq!(WINDOWS_1252_CODEPOINTS[b], LATIN1_CODEPOINTS[b]);
        }
        for b in 0xA0..=0xFFusize {
            assert_eq!(WINDOWS_1252_CODEPOINTS[b], LATIN1_CODEPOINTS[b]);
        }
    }

    #[test]
    fn windows_1252_euro_sign_at_0x80() {
        assert_eq!(WINDOWS_1252_CODEPOINTS[0x80], 0x20AC);
        assert_eq!(codepoint_to_char(WINDOWS_1252_CODEPOINTS[0x80]), '\u{20AC}');
    }

    #[test]
    fn windows_1252_unassigned_slots_fall_back_to_latin1() {
        for b in [0x81usize, 0x8D, 0x8F, 0x90, 0x9D] {
            assert_eq!(WINDOWS_1252_CODEPOINTS[b], b as u32, "byte 0x{b:02X}");
        }
    }

    #[test]
    fn every_256_table_entry_is_a_valid_scalar_value() {
        for &cp in LATIN1_CODEPOINTS
            .iter()
            .chain(WINDOWS_1252_CODEPOINTS.iter())
        {
            assert!(
                char::from_u32(cp).is_some(),
                "0x{cp:04X} is not a valid scalar value"
            );
        }
    }

    #[test]
    fn encode_single_byte_round_trips_windows_1252() {
        for b in 0..=255u8 {
            let scalar = WINDOWS_1252_CODEPOINTS[b as usize];
            assert_eq!(
                encode_single_byte(&WINDOWS_1252_CODEPOINTS, scalar),
                Some(b)
            );
        }
    }

    #[test]
    fn encode_single_byte_rejects_unmappable_scalar() {
        // U+3042 (HIRAGANA LETTER A) exists in neither table.
        assert_eq!(encode_single_byte(&WINDOWS_1252_CODEPOINTS, 0x3042), None);
        assert_eq!(encode_single_byte(&LATIN1_CODEPOINTS, 0x3042), None);
    }

    #[test]
    fn every_128_table_entry_is_unmapped_or_a_valid_scalar_value() {
        for &encoding in Encoding::ALL {
            if !encoding.is_legacy_single_byte_table() {
                continue;
            }
            let table = legacy_single_byte_table(encoding);
            for &cp in table.iter() {
                assert!(
                    cp == UNMAPPED || char::from_u32(cp).is_some(),
                    "{encoding}: 0x{cp:04X} is neither UNMAPPED nor a valid scalar value"
                );
            }
        }
    }

    // Regression test for a hand-synchronized invariant that used to be enforced only by a
    // catch-all `_ => unreachable!()` arm: `legacy_single_byte_table` now names every
    // `Encoding` variant explicitly, so a variant added to the enum without a corresponding
    // arm here fails the *build* (a "non-exhaustive patterns" error naming the missing
    // variant) rather than compiling clean and then panicking the first time some caller
    // reaches it with the new variant. This test pins the two halves of that partition — which
    // variants resolve to a table, and which do not — so a change to either list is caught here
    // even though the compile-time exhaustiveness itself has no runtime observable.
    #[test]
    fn legacy_single_byte_table_partitions_every_encoding_variant_without_panicking() {
        for &encoding in Encoding::ALL {
            if encoding.is_legacy_single_byte_table() {
                // Must not panic (this used to be the `unreachable!()` branch's complement).
                let _ = legacy_single_byte_table(encoding);
            }
        }
        // The complement — UTF-8/UTF-16LE/UTF-16BE, the original Latin-1/windows-1252/ASCII
        // trio, and the seven multi-byte encodings — is exactly what the exhaustive match's
        // non-table arm names; confirm none of `Encoding::ALL` silently drifted out of that
        // specific list without this test noticing.
        let non_table: Vec<Encoding> = Encoding::ALL
            .iter()
            .copied()
            .filter(|e| !e.is_legacy_single_byte_table())
            .collect();
        assert_eq!(
            non_table,
            vec![
                Encoding::Utf8,
                Encoding::Utf16Le,
                Encoding::Utf16Be,
                Encoding::Latin1,
                Encoding::Windows1252,
                Encoding::Ascii,
                Encoding::ShiftJis,
                Encoding::EucJp,
                Encoding::Iso2022Jp,
                Encoding::Gbk,
                Encoding::Gb18030,
                Encoding::Big5,
                Encoding::EucKr,
            ]
        );
    }

    #[test]
    fn upper_table_decode_encode_round_trips_for_every_mapped_byte() {
        for &encoding in Encoding::ALL {
            if !encoding.is_legacy_single_byte_table() {
                continue;
            }
            let table = legacy_single_byte_table(encoding);
            for pointer in 0u8..128 {
                if table[pointer as usize] == UNMAPPED {
                    continue;
                }
                let byte = 0x80 + pointer;
                let c = decode_single_byte_upper(table, byte)
                    .unwrap_or_else(|| panic!("{encoding}: byte 0x{byte:02X} must decode"));
                assert_eq!(
                    encode_single_byte_upper(table, c as u32),
                    Some(byte),
                    "{encoding}: round trip for byte 0x{byte:02X} (U+{:04X})",
                    c as u32
                );
            }
        }
    }

    #[test]
    fn decode_single_byte_upper_indexes_from_the_start_of_the_upper_half() {
        // Regression test: `decode_single_byte_upper` takes the *raw byte* (0x80..=0xFF), not an
        // already-subtracted pointer (0..=127) -- a mismatch here previously caused an
        // out-of-bounds panic on every byte at or above 0x80 + 128 = 0x100, and more subtly
        // decoded every byte through the wrong table slot (128 positions too high) for any byte
        // it didn't outright panic on.
        let table = legacy_single_byte_table(Encoding::Windows1250);
        // 0x80 must read pointer 0 (the very first upper-half entry), not panic or read pointer
        // 0x80 (out of bounds for a 128-entry table).
        assert_eq!(
            decode_single_byte_upper(table, 0x80),
            Some(codepoint_to_char(table[0]))
        );
        // 0xFF (the last byte) must read pointer 127 (the very last entry).
        assert_eq!(
            decode_single_byte_upper(table, 0xFF),
            Some(codepoint_to_char(table[127]))
        );
    }

    #[test]
    fn jis0208_decode_out_of_range_pointer_is_none() {
        assert_eq!(jis0208_decode(JIS0208_DECODE.len() as u32), None);
        assert_eq!(jis0208_decode(u32::MAX), None);
    }

    #[test]
    fn jis0212_decode_out_of_range_pointer_is_none() {
        assert_eq!(jis0212_decode(JIS0212_DECODE.len() as u32), None);
        assert_eq!(jis0212_decode(u32::MAX), None);
    }

    #[test]
    fn jis0208_hiragana_a_round_trips_at_well_known_pointer() {
        // U+3042 (HIRAGANA LETTER A, "あ") is pointer 283 -- EUC-JP 0xA4 0xA2, Shift_JIS 0x82
        // 0xA0, both well-known reference values independently confirmed against the system
        // `iconv`'s SHIFT_JIS/EUC-JP tables while generating these tables.
        assert_eq!(jis0208_decode(283), Some('\u{3042}'));
        assert_eq!(jis0208_encode_full(0x3042), Some(283));
        assert_eq!(jis0208_encode_shift_jis(0x3042), Some(283));
    }

    #[test]
    fn iso2022jp_katakana_table_covers_the_full_half_width_katakana_block() {
        assert_eq!(ISO2022JP_KATAKANA.len(), 63);
        // U+FF66 (halfwidth ヲ) -> U+30F2 (fullwidth ヲ), per the WHATWG table.
        assert_eq!(iso2022jp_katakana_replacement(0xFF66), 0x30F2);
    }
}

//! Shift_JIS decoder and encoder, per the
//! [WHATWG Encoding Standard §12.3](https://encoding.spec.whatwg.org/#shift_jis-decoder).

use crate::decoder::{DecodeError, DecodeErrorKind};
use crate::encoder::EncodeError;
use crate::tables::{jis0208_decode, jis0208_encode_shift_jis};

/// The private-use "EUDC" (End User Defined Characters) pointer range documented by the spec:
/// "This is interoperable legacy from Windows known as EUDC."
const EUDC_POINTER_RANGE: std::ops::RangeInclusive<i32> = 8836..=10715;
const EUDC_BASE: u32 = 0xE000 - 8836;

/// State carried between [`decode`]/[`decode_lossy`] calls: a single buffered lead byte, or
/// `0x00` if none is pending.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShiftJisState {
    leading: u8,
}

impl ShiftJisState {
    pub(crate) const fn new() -> Self {
        ShiftJisState { leading: 0 }
    }

    pub(crate) const fn has_pending(&self) -> bool {
        self.leading != 0
    }
}

/// Resolve a buffered `leading` byte plus the current `byte` to a decoded code point, per
/// WHATWG §12.3.1. Returns `None` for any combination that is not a valid two-byte sequence
/// (including one that decodes to an unassigned `index jis0208` pointer).
fn resolve_pair(leading: u8, byte: u8) -> Option<char> {
    let leading_offset: u32 = if leading < 0xA0 { 0x81 } else { 0xC1 };
    let offset: u32 = if byte < 0x7F { 0x40 } else { 0x41 };
    let in_range = (0x40..=0x7E).contains(&byte) || (0x80..=0xFC).contains(&byte);
    if !in_range {
        return None;
    }
    let pointer = (leading as i32 - leading_offset as i32) * 188 + (byte as i32 - offset as i32);
    if EUDC_POINTER_RANGE.contains(&pointer) {
        return char::from_u32(EUDC_BASE + pointer as u32);
    }
    if pointer < 0 {
        return None;
    }
    jis0208_decode(pointer as u32)
}

/// Decode the initial (no buffered lead byte) state for one byte, per WHATWG §12.3.1's final
/// four steps.
fn decode_initial(byte: u8) -> InitialOutcome {
    if byte <= 0x80 {
        return InitialOutcome::Emit(byte as char);
    }
    if (0xA1..=0xDF).contains(&byte) {
        // Half-width katakana: arithmetic, not a table lookup.
        let cp = 0xFF61u32 - 0xA1 + byte as u32;
        return InitialOutcome::Emit(char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER));
    }
    if (0x81..=0x9F).contains(&byte) || (0xE0..=0xFC).contains(&byte) {
        return InitialOutcome::BufferLeading;
    }
    InitialOutcome::Error
}

enum InitialOutcome {
    Emit(char),
    BufferLeading,
    Error,
}

/// Strict incremental Shift_JIS decode. See [`crate::decoder::Decoder::decode_to_string`].
pub(crate) fn decode(
    state: &mut ShiftJisState,
    input: &[u8],
    dst: &mut String,
    last: bool,
) -> Result<usize, DecodeError> {
    let mut idx = 0usize;
    while idx < input.len() {
        let byte = input[idx];
        if state.leading != 0 {
            let leading = state.leading;
            state.leading = 0;
            match resolve_pair(leading, byte) {
                Some(c) => {
                    dst.push(c);
                    idx += 1;
                }
                None => {
                    let leading_pos = idx.saturating_sub(1);
                    let length = if byte < 0x80 { 1 } else { 2 };
                    return Err(DecodeError {
                        offset: leading_pos,
                        length,
                        kind: DecodeErrorKind::MalformedSequence,
                    });
                }
            }
            continue;
        }
        match decode_initial(byte) {
            InitialOutcome::Emit(c) => {
                dst.push(c);
                idx += 1;
            }
            InitialOutcome::BufferLeading => {
                state.leading = byte;
                idx += 1;
            }
            InitialOutcome::Error => {
                return Err(DecodeError {
                    offset: idx,
                    length: 1,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
        }
    }
    if last && state.leading != 0 {
        state.leading = 0;
        return Err(DecodeError {
            offset: input.len().saturating_sub(1),
            length: 1,
            kind: DecodeErrorKind::TruncatedSequence,
        });
    }
    Ok(input.len())
}

/// Lossy incremental Shift_JIS decode: any malformed byte becomes `U+FFFD`. See
/// [`crate::decoder::Decoder::decode_to_string_lossy`].
pub(crate) fn decode_lossy(
    state: &mut ShiftJisState,
    input: &[u8],
    dst: &mut String,
    last: bool,
) -> usize {
    let mut idx = 0usize;
    while idx < input.len() {
        let byte = input[idx];
        if state.leading != 0 {
            let leading = state.leading;
            state.leading = 0;
            match resolve_pair(leading, byte) {
                Some(c) => {
                    dst.push(c);
                    idx += 1;
                }
                None => {
                    dst.push(char::REPLACEMENT_CHARACTER);
                    if byte < 0x80 {
                        // Restore: reprocess `byte` fresh, now with no buffered leading byte.
                    } else {
                        idx += 1;
                    }
                }
            }
            continue;
        }
        match decode_initial(byte) {
            InitialOutcome::Emit(c) => {
                dst.push(c);
                idx += 1;
            }
            InitialOutcome::BufferLeading => {
                state.leading = byte;
                idx += 1;
            }
            InitialOutcome::Error => {
                dst.push(char::REPLACEMENT_CHARACTER);
                idx += 1;
            }
        }
    }
    if last && state.leading != 0 {
        state.leading = 0;
        dst.push(char::REPLACEMENT_CHARACTER);
    }
    input.len()
}

/// Encode `input` (valid UTF-8 text) as Shift_JIS, per WHATWG §12.3.2.
pub(crate) fn encode(input: &str, dst: &mut Vec<u8>) -> Result<(), EncodeError> {
    for (position, character) in input.char_indices() {
        let codepoint = character as u32;
        if character.is_ascii() || codepoint == 0x80 {
            dst.push(codepoint as u8);
            continue;
        }
        if character == '\u{00A5}' {
            dst.push(0x5C);
            continue;
        }
        if character == '\u{203E}' {
            dst.push(0x7E);
            continue;
        }
        if (0xFF61..=0xFF9F).contains(&codepoint) {
            dst.push((codepoint - 0xFF61 + 0xA1) as u8);
            continue;
        }
        let lookup_codepoint = if character == '\u{2212}' {
            0xFF0Du32
        } else {
            codepoint
        };
        let Some(pointer) = jis0208_encode_shift_jis(lookup_codepoint) else {
            return Err(EncodeError {
                character,
                position,
            });
        };
        let pointer = pointer as u32;
        let leading = pointer / 188;
        let leading_offset: u32 = if leading < 0x1F { 0x81 } else { 0xC1 };
        let trailing = pointer % 188;
        let offset: u32 = if trailing < 0x3F { 0x40 } else { 0x41 };
        dst.push((leading + leading_offset) as u8);
        dst.push((trailing + offset) as u8);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(chunks: &[&[u8]]) -> Result<String, DecodeError> {
        let mut state = ShiftJisState::new();
        let mut out = String::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            decode(&mut state, chunk, &mut out, last)?;
        }
        Ok(out)
    }

    #[test]
    fn ascii_passthrough() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[b"Hello, world!"])?, "Hello, world!");
        Ok(())
    }

    #[test]
    fn byte_0x80_decodes_to_u_0080_not_an_error() -> Result<(), DecodeError> {
        // WHATWG §12.3.1: "If byte is an ASCII byte or 0x80, then return a code point whose
        // value is byte." -- 0x80 is a real (if unusual) success case, not an error.
        assert_eq!(decode_all(&[&[0x80]])?, "\u{0080}");
        Ok(())
    }

    #[test]
    fn byte_0xa0_is_a_decode_error() {
        // 0xA0 is none of: ASCII, 0x80, half-width katakana (0xA1..=0xDF), or a valid lead byte
        // (0x81..=0x9F, 0xE0..=0xFC) -- it is simply invalid on its own.
        let mut state = ShiftJisState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0xA0], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
        assert_eq!(err.offset, 0);
        assert_eq!(err.length, 1);
    }

    #[test]
    fn half_width_katakana_range() -> Result<(), DecodeError> {
        // 0xA1 = U+FF61 (halfwidth ideographic full stop), 0xDF = U+FF9F (halfwidth semi-voiced
        // sound mark) -- the two ends of the range.
        assert_eq!(decode_all(&[&[0xA1]])?, "\u{FF61}");
        assert_eq!(decode_all(&[&[0xDF]])?, "\u{FF9F}");
        Ok(())
    }

    #[test]
    fn well_known_kanji_bytes() -> Result<(), DecodeError> {
        // "日" (U+65E5) is a commonly-cited Shift_JIS reference value, independently confirmed
        // against the system `iconv`'s SHIFT_JIS table while generating these tables.
        assert_eq!(decode_all(&[&[0x93, 0xFA]])?, "\u{65E5}");
        // "あ" (U+3042), likewise.
        assert_eq!(decode_all(&[&[0x82, 0xA0]])?, "\u{3042}");
        Ok(())
    }

    #[test]
    fn lead_byte_with_ascii_trail_restores_the_trail_byte() {
        // 0x81 is a valid lead byte; 0x20 (space) is not a valid trail byte, but *is* ASCII, so
        // per spec it is "restored" (reprocessed fresh) rather than consumed as part of the
        // error. Lossy mode must therefore decode it as a literal space afterwards.
        let mut state = ShiftJisState::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0x81, 0x20], &mut out, true);
        assert_eq!(out, "\u{FFFD} ");
    }

    #[test]
    fn lead_byte_with_non_ascii_invalid_trail_consumes_both_bytes() {
        // 0x81 0xFF: 0xFF is neither in 0x40..=0x7E nor 0x80..=0xFC, and is not ASCII, so both
        // bytes are consumed as a single 2-byte malformed unit.
        let mut state = ShiftJisState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x81, 0xFF], &mut out, true).expect_err("must fail");
        assert_eq!(err.offset, 0);
        assert_eq!(err.length, 2);
    }

    #[test]
    fn truncated_lead_byte_at_true_end_of_stream() {
        let mut state = ShiftJisState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x82], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    }

    #[test]
    fn lead_byte_buffered_across_chunk_boundary() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[&[0x82], &[0xA0]])?, "\u{3042}");
        Ok(())
    }

    #[test]
    fn encode_ascii_and_yen_and_overline() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("a\u{00A5}\u{203E}", &mut dst)?;
        assert_eq!(dst, vec![b'a', 0x5C, 0x7E]);
        Ok(())
    }

    #[test]
    fn encode_well_known_kanji() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("日", &mut dst)?;
        assert_eq!(dst, vec![0x93, 0xFA]);
        let mut dst = Vec::new();
        encode("あ", &mut dst)?;
        assert_eq!(dst, vec![0x82, 0xA0]);
        Ok(())
    }

    #[test]
    fn encode_rejects_unmappable_character() {
        // U+1F600 (an emoji) has no Shift_JIS representation.
        let mut dst = Vec::new();
        let err = encode("😀", &mut dst).expect_err("must fail");
        assert_eq!(err.character, '😀');
    }

    #[test]
    fn round_trip_mixed_text_at_every_chunk_size() -> Result<(), Box<dyn std::error::Error>> {
        let text = "Hello, 世界! ネコ 日本語テスト123";
        let mut bytes = Vec::new();
        encode(text, &mut bytes)?;
        for chunk_size in 1..=3 {
            let mut state = ShiftJisState::new();
            let mut out = String::new();
            let mut offset = 0;
            while offset < bytes.len() {
                let end = (offset + chunk_size).min(bytes.len());
                let last = end == bytes.len();
                decode(&mut state, &bytes[offset..end], &mut out, last)?;
                offset = end;
            }
            assert_eq!(out, text, "chunk_size={chunk_size}");
        }
        Ok(())
    }

    #[test]
    fn lossy_decode_never_panics_on_arbitrary_bytes() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        let mut state = ShiftJisState::new();
        let mut out = String::new();
        let consumed = decode_lossy(&mut state, &all_bytes, &mut out, true);
        assert_eq!(consumed, 256);
    }
}

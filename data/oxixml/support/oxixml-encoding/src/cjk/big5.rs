//! Big5 decoder and encoder, per the
//! [WHATWG Encoding Standard §11.1](https://encoding.spec.whatwg.org/#big5-decoder).

use crate::decoder::{DecodeError, DecodeErrorKind};
use crate::encoder::EncodeError;
use crate::tables::{big5_decode, big5_encode};

/// State carried between [`decode`]/[`decode_lossy`] calls: a single buffered lead byte, or
/// `0x00` if none is pending.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Big5State {
    leading: u8,
}

impl Big5State {
    pub(crate) const fn new() -> Self {
        Big5State { leading: 0 }
    }

    pub(crate) const fn has_pending(&self) -> bool {
        self.leading != 0
    }
}

/// Resolve a buffered `leading` byte plus the current `byte` to one or two decoded code points,
/// per WHATWG §11.1.1. Returns `None` for any combination that is not a valid two-byte sequence
/// (including one that decodes to an unassigned pointer).
///
/// Four pointers (1133, 1135, 1164, 1166) decode to a base Latin letter *plus* a combining
/// diacritic rather than a single precomposed character, since [`crate::tables::big5_decode`]'s
/// underlying index can only hold one code point per pointer; those four are special-cased
/// directly here, exactly as the spec's own decoder algorithm does, ahead of the ordinary index
/// lookup.
fn resolve_pair(leading: u8, byte: u8) -> Option<(char, Option<char>)> {
    let offset: u32 = if byte < 0x7F { 0x40 } else { 0x62 };
    let in_range = (0x40..=0x7E).contains(&byte) || (0xA1..=0xFE).contains(&byte);
    if !in_range {
        return None;
    }
    let pointer = (leading as u32 - 0x81) * 157 + (byte as u32 - offset);
    match pointer {
        1133 => return Some(('\u{00CA}', Some('\u{0304}'))),
        1135 => return Some(('\u{00CA}', Some('\u{030C}'))),
        1164 => return Some(('\u{00EA}', Some('\u{0304}'))),
        1166 => return Some(('\u{00EA}', Some('\u{030C}'))),
        _ => {}
    }
    big5_decode(pointer).map(|c| (c, None))
}

/// Decode the initial (no buffered lead byte) state for one byte, per WHATWG §11.1.1's final
/// three steps.
fn decode_initial(byte: u8) -> InitialOutcome {
    if byte.is_ascii() {
        return InitialOutcome::Emit(byte as char);
    }
    if (0x81..=0xFE).contains(&byte) {
        return InitialOutcome::BufferLeading;
    }
    InitialOutcome::Error
}

enum InitialOutcome {
    Emit(char),
    BufferLeading,
    Error,
}

/// Strict incremental Big5 decode. See [`crate::decoder::Decoder::decode_to_string`].
pub(crate) fn decode(
    state: &mut Big5State,
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
                Some((c1, c2)) => {
                    dst.push(c1);
                    if let Some(c2) = c2 {
                        dst.push(c2);
                    }
                    idx += 1;
                }
                None => {
                    let leading_pos = idx.saturating_sub(1);
                    let length = if byte.is_ascii() { 1 } else { 2 };
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

/// Lossy incremental Big5 decode: any malformed byte becomes `U+FFFD`. See
/// [`crate::decoder::Decoder::decode_to_string_lossy`].
pub(crate) fn decode_lossy(
    state: &mut Big5State,
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
                Some((c1, c2)) => {
                    dst.push(c1);
                    if let Some(c2) = c2 {
                        dst.push(c2);
                    }
                    idx += 1;
                }
                None => {
                    dst.push(char::REPLACEMENT_CHARACTER);
                    if byte.is_ascii() {
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

/// Encode `input` (valid UTF-8 text) as Big5, per WHATWG §11.1.2.
pub(crate) fn encode(input: &str, dst: &mut Vec<u8>) -> Result<(), EncodeError> {
    for (position, character) in input.char_indices() {
        if character.is_ascii() {
            dst.push(character as u8);
            continue;
        }
        let Some(pointer) = big5_encode(character as u32) else {
            return Err(EncodeError {
                character,
                position,
            });
        };
        let pointer = pointer as u32;
        let leading = pointer / 157 + 0x81;
        let trailing = pointer % 157;
        let offset: u32 = if trailing < 0x3F { 0x40 } else { 0x62 };
        dst.push(leading as u8);
        dst.push((trailing + offset) as u8);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(chunks: &[&[u8]]) -> Result<String, DecodeError> {
        let mut state = Big5State::new();
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
    fn well_known_kanji_bytes() -> Result<(), DecodeError> {
        // "中" (U+4E2D), independently confirmed against the WHATWG index-big5.txt data (pointer
        // 5561) while writing this test.
        assert_eq!(decode_all(&[&[0xA4, 0xA4]])?, "\u{4E2D}");
        Ok(())
    }

    #[test]
    fn hkscs_extension_a_character() -> Result<(), DecodeError> {
        // Pointer 942 (U+43F0), the first entry in index-big5.txt: a Hong Kong Supplementary
        // Character Set / CJK Extension A ideograph, using a lead byte (0x87) below the plain
        // Big5 lead-byte range (0xA1..=0xF9) that only HKSCS uses.
        assert_eq!(decode_all(&[&[0x87, 0x40]])?, "\u{43F0}");
        Ok(())
    }

    #[test]
    fn two_code_point_pointers() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[&[0x88, 0x62]])?, "\u{00CA}\u{0304}");
        assert_eq!(decode_all(&[&[0x88, 0x64]])?, "\u{00CA}\u{030C}");
        assert_eq!(decode_all(&[&[0x88, 0xA3]])?, "\u{00EA}\u{0304}");
        assert_eq!(decode_all(&[&[0x88, 0xA5]])?, "\u{00EA}\u{030C}");
        Ok(())
    }

    #[test]
    fn lead_byte_buffered_across_chunk_boundary() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[&[0xA4], &[0xA4]])?, "\u{4E2D}");
        Ok(())
    }

    #[test]
    fn two_code_point_pointer_split_across_chunk_boundary() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[&[0x88], &[0x62]])?, "\u{00CA}\u{0304}");
        Ok(())
    }

    #[test]
    fn truncated_lead_byte_at_true_end_of_stream() {
        let mut state = Big5State::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0xA4], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    }

    #[test]
    fn lead_byte_with_ascii_trail_restores_the_trail_byte() {
        let mut state = Big5State::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0xA1, 0x20], &mut out, true);
        assert_eq!(out, "\u{FFFD} ");
    }

    #[test]
    fn lead_byte_with_non_ascii_invalid_trail_consumes_both_bytes() {
        // 0x80 is below Big5's trail ranges (0x40..=0x7E, 0xA1..=0xFE) and is not ASCII, so both
        // bytes are consumed as a single 2-byte malformed unit rather than restoring the trail
        // byte.
        let mut state = Big5State::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0xA1, 0x80], &mut out, true).expect_err("must fail");
        assert_eq!(err.offset, 0);
        assert_eq!(err.length, 2);
    }

    #[test]
    fn invalid_lead_byte_0x80_is_an_error() {
        let mut state = Big5State::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x80], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
        assert_eq!(err.offset, 0);
    }

    #[test]
    fn encode_well_known_kanji() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("中", &mut dst)?;
        assert_eq!(dst, vec![0xA4, 0xA4]);
        Ok(())
    }

    #[test]
    fn encode_rejects_unmappable_character() {
        let mut dst = Vec::new();
        let err = encode("😀", &mut dst).expect_err("must fail");
        assert_eq!(err.character, '😀');
    }

    #[test]
    fn encode_duplicate_code_points_prefer_the_documented_last_pointer() -> Result<(), EncodeError>
    {
        // U+5341 (十) is duplicated in the filtered Big5 index; per the WHATWG "index Big5
        // pointer" algorithm this code point (unlike almost every other duplicate) must resolve
        // to its *last* pointer. Round-tripping through this module's own decoder is the
        // behavior that actually matters; independently confirmed against index-big5.txt
        // (pointers 5287 and 5512 -- 5512 must be the one used) while writing this test.
        let mut dst = Vec::new();
        encode("十", &mut dst)?;
        let leading = 5512u32 / 157 + 0x81;
        let trailing = 5512u32 % 157;
        let offset: u32 = if trailing < 0x3F { 0x40 } else { 0x62 };
        assert_eq!(dst, vec![leading as u8, (trailing + offset) as u8]);
        Ok(())
    }

    #[test]
    fn round_trip_mixed_text_at_every_chunk_size() -> Result<(), Box<dyn std::error::Error>> {
        let text = "Hello, 世界! 中文測試123";
        let mut bytes = Vec::new();
        encode(text, &mut bytes)?;
        for chunk_size in 1..=3 {
            let mut state = Big5State::new();
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
        let mut state = Big5State::new();
        let mut out = String::new();
        let consumed = decode_lossy(&mut state, &all_bytes, &mut out, true);
        assert_eq!(consumed, 256);
    }

    #[test]
    fn lossy_decode_one_byte_at_a_time_never_panics() {
        let mut state = Big5State::new();
        let mut out = String::new();
        for b in 0u16..=255 {
            let last = b == 255;
            decode_lossy(&mut state, &[b as u8], &mut out, last);
        }
    }
}

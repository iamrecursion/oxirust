//! EUC-KR decoder and encoder, per the
//! [WHATWG Encoding Standard §13.1](https://encoding.spec.whatwg.org/#euc-kr-decoder).

use crate::decoder::{DecodeError, DecodeErrorKind};
use crate::encoder::EncodeError;
use crate::tables::{euc_kr_decode, euc_kr_encode};

/// State carried between [`decode`]/[`decode_lossy`] calls: a single buffered lead byte, or
/// `0x00` if none is pending.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EucKrState {
    leading: u8,
}

impl EucKrState {
    pub(crate) const fn new() -> Self {
        EucKrState { leading: 0 }
    }

    pub(crate) const fn has_pending(&self) -> bool {
        self.leading != 0
    }
}

/// Resolve a buffered `leading` byte plus the current `byte` to a decoded code point, per WHATWG
/// §13.1.1. Returns `None` for any combination that is not a valid two-byte sequence (including
/// one that decodes to an unassigned pointer, or a pointer past the end of
/// [`crate::tables::euc_kr_decode`]'s table -- the structural byte-range formula below admits a
/// slightly larger pointer space than the index actually assigns).
fn resolve_pair(leading: u8, byte: u8) -> Option<char> {
    if !(0x41..=0xFE).contains(&byte) {
        return None;
    }
    let pointer = (leading as u32 - 0x81) * 190 + (byte as u32 - 0x41);
    euc_kr_decode(pointer)
}

/// Decode the initial (no buffered lead byte) state for one byte, per WHATWG §13.1.1's final
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

/// Strict incremental EUC-KR decode. See [`crate::decoder::Decoder::decode_to_string`].
pub(crate) fn decode(
    state: &mut EucKrState,
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

/// Lossy incremental EUC-KR decode: any malformed byte becomes `U+FFFD`. See
/// [`crate::decoder::Decoder::decode_to_string_lossy`].
pub(crate) fn decode_lossy(
    state: &mut EucKrState,
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

/// Encode `input` (valid UTF-8 text) as EUC-KR, per WHATWG §13.1.2.
pub(crate) fn encode(input: &str, dst: &mut Vec<u8>) -> Result<(), EncodeError> {
    for (position, character) in input.char_indices() {
        if character.is_ascii() {
            dst.push(character as u8);
            continue;
        }
        let Some(pointer) = euc_kr_encode(character as u32) else {
            return Err(EncodeError {
                character,
                position,
            });
        };
        let pointer = pointer as u32;
        let leading = pointer / 190 + 0x81;
        let trailing = pointer % 190 + 0x41;
        dst.push(leading as u8);
        dst.push(trailing as u8);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(chunks: &[&[u8]]) -> Result<String, DecodeError> {
        let mut state = EucKrState::new();
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
    fn well_known_hangul_syllable() -> Result<(), DecodeError> {
        // "한" (U+D55C), independently confirmed against Python's built-in `euc_kr` codec while
        // writing this test.
        assert_eq!(decode_all(&[&[0xC7, 0xD1]])?, "\u{D55C}");
        Ok(())
    }

    #[test]
    fn unified_hangul_extension_character_at_pointer_zero() -> Result<(), DecodeError> {
        // Pointer 0 (the first entry in index-euc-kr.txt, U+AC02) is one of the Unified Hangul
        // Code / Windows-949 extension syllables not present in the older, smaller KS X 1001
        // repertoire -- independently confirmed against index-euc-kr.txt while writing this
        // test.
        assert_eq!(decode_all(&[&[0x81, 0x41]])?, "\u{AC02}");
        Ok(())
    }

    #[test]
    fn lead_byte_buffered_across_chunk_boundary() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[&[0xC7], &[0xD1]])?, "\u{D55C}");
        Ok(())
    }

    #[test]
    fn truncated_lead_byte_at_true_end_of_stream() {
        let mut state = EucKrState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0xC7], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    }

    #[test]
    fn lead_byte_with_ascii_trail_restores_the_trail_byte() {
        let mut state = EucKrState::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0x81, 0x20], &mut out, true);
        assert_eq!(out, "\u{FFFD} ");
    }

    #[test]
    fn lead_byte_with_non_ascii_invalid_trail_consumes_both_bytes() {
        // 0x80 is below EUC-KR's trail range (0x41..=0xFE) and is not ASCII, so both bytes are
        // consumed as a single 2-byte malformed unit rather than restoring the trail byte.
        let mut state = EucKrState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x81, 0x80], &mut out, true).expect_err("must fail");
        assert_eq!(err.offset, 0);
        assert_eq!(err.length, 2);
    }

    #[test]
    fn pointer_past_the_end_of_the_table_is_an_error_not_a_panic() {
        // 0xFE 0x41 computes structural pointer 23750 -- one past the table's highest actually
        // assigned pointer (23749) -- and 0xFE 0xFE computes the structural maximum, 23939. Both
        // must be reported as ordinary decode errors via the table's `.get()` bounds check, never
        // an out-of-bounds panic.
        for bytes in [[0xFEu8, 0x41], [0xFE, 0xFE]] {
            let mut state = EucKrState::new();
            let mut out = String::new();
            let err = decode(&mut state, &bytes, &mut out, true).expect_err("must fail");
            assert_eq!(err.kind, DecodeErrorKind::MalformedSequence, "{bytes:?}");
        }
    }

    #[test]
    fn invalid_lead_byte_0x80_is_an_error() {
        let mut state = EucKrState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x80], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
        assert_eq!(err.offset, 0);
    }

    #[test]
    fn encode_well_known_hangul_syllable() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("한", &mut dst)?;
        assert_eq!(dst, vec![0xC7, 0xD1]);
        Ok(())
    }

    #[test]
    fn encode_rejects_unmappable_character() {
        let mut dst = Vec::new();
        let err = encode("😀", &mut dst).expect_err("must fail");
        assert_eq!(err.character, '😀');
    }

    #[test]
    fn round_trip_mixed_text_at_every_chunk_size() -> Result<(), Box<dyn std::error::Error>> {
        let text = "Hello, 한국어 테스트123";
        let mut bytes = Vec::new();
        encode(text, &mut bytes)?;
        for chunk_size in 1..=3 {
            let mut state = EucKrState::new();
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
        let mut state = EucKrState::new();
        let mut out = String::new();
        let consumed = decode_lossy(&mut state, &all_bytes, &mut out, true);
        assert_eq!(consumed, 256);
    }

    #[test]
    fn lossy_decode_one_byte_at_a_time_never_panics() {
        let mut state = EucKrState::new();
        let mut out = String::new();
        for b in 0u16..=255 {
            let last = b == 255;
            decode_lossy(&mut state, &[b as u8], &mut out, last);
        }
    }
}

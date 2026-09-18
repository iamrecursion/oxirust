//! EUC-JP decoder and encoder, per the
//! [WHATWG Encoding Standard §12.1](https://encoding.spec.whatwg.org/#euc-jp-decoder).

use crate::decoder::{DecodeError, DecodeErrorKind};
use crate::encoder::EncodeError;
use crate::tables::{jis0208_decode, jis0208_encode_full, jis0212_decode};

/// State carried between [`decode`]/[`decode_lossy`] calls.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EucJpState {
    /// "EUC-JP leading": a buffered lead byte, or `0x00` if none is pending. While waiting for
    /// the final byte of a `0x8F`-prefixed (SS3 / JIS X 0212) sequence, this holds the *row*
    /// byte (the byte immediately after the `0x8F`), matching the spec's own reuse of the same
    /// field for that purpose.
    leading: u8,
    /// "EUC-JP jis0212": whether the pending 2-byte completion (once `leading` resolves) should
    /// be looked up in JIS X 0212 rather than JIS X 0208.
    jis0212: bool,
}

impl EucJpState {
    pub(crate) const fn new() -> Self {
        EucJpState {
            leading: 0,
            jis0212: false,
        }
    }

    pub(crate) const fn has_pending(&self) -> bool {
        self.leading != 0
    }
}

enum StepOutcome {
    Continue,
    Emit(char),
    /// The byte at `state.leading` alone is malformed; `byte` should be reprocessed fresh (not
    /// consumed) since it is itself a plain ASCII byte.
    ErrorRestoreCurrent,
    /// Both the buffered leading byte and `byte` are consumed as a single malformed unit.
    ErrorConsumeBoth,
    /// `byte` alone (no buffered state) is not a valid EUC-JP lead byte.
    ErrorHere,
}

fn step(state: &mut EucJpState, byte: u8) -> StepOutcome {
    if state.leading == 0x8E {
        state.leading = 0;
        if (0xA1..=0xDF).contains(&byte) {
            let cp = 0xFF61u32 - 0xA1 + byte as u32;
            return StepOutcome::Emit(char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER));
        }
        return if byte < 0x80 {
            StepOutcome::ErrorRestoreCurrent
        } else {
            StepOutcome::ErrorConsumeBoth
        };
    }
    if state.leading == 0x8F && (0xA1..=0xFE).contains(&byte) {
        state.jis0212 = true;
        state.leading = byte;
        return StepOutcome::Continue;
    }
    if state.leading != 0 {
        let leading = state.leading;
        state.leading = 0;
        let jis0212 = state.jis0212;
        state.jis0212 = false;
        let codepoint = if (0xA1..=0xFE).contains(&leading) && (0xA1..=0xFE).contains(&byte) {
            let pointer = (leading as u32 - 0xA1) * 94 + byte as u32 - 0xA1;
            if jis0212 {
                jis0212_decode(pointer)
            } else {
                jis0208_decode(pointer)
            }
        } else {
            None
        };
        return match codepoint {
            Some(c) => StepOutcome::Emit(c),
            None if byte < 0x80 => StepOutcome::ErrorRestoreCurrent,
            None => StepOutcome::ErrorConsumeBoth,
        };
    }
    if byte < 0x80 {
        return StepOutcome::Emit(byte as char);
    }
    if byte == 0x8E || byte == 0x8F || (0xA1..=0xFE).contains(&byte) {
        state.leading = byte;
        return StepOutcome::Continue;
    }
    StepOutcome::ErrorHere
}

/// Strict incremental EUC-JP decode. See [`crate::decoder::Decoder::decode_to_string`].
pub(crate) fn decode(
    state: &mut EucJpState,
    input: &[u8],
    dst: &mut String,
    last: bool,
) -> Result<usize, DecodeError> {
    let mut idx = 0usize;
    while idx < input.len() {
        let byte = input[idx];
        match step(state, byte) {
            StepOutcome::Continue => idx += 1,
            StepOutcome::Emit(c) => {
                dst.push(c);
                idx += 1;
            }
            StepOutcome::ErrorHere => {
                return Err(DecodeError {
                    offset: idx,
                    length: 1,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
            StepOutcome::ErrorRestoreCurrent => {
                return Err(DecodeError {
                    offset: idx.saturating_sub(1),
                    length: 1,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
            StepOutcome::ErrorConsumeBoth => {
                return Err(DecodeError {
                    offset: idx.saturating_sub(1),
                    length: 2,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
        }
    }
    if last && state.has_pending() {
        *state = EucJpState::new();
        return Err(DecodeError {
            offset: input.len().saturating_sub(1),
            length: 1,
            kind: DecodeErrorKind::TruncatedSequence,
        });
    }
    Ok(input.len())
}

/// Lossy incremental EUC-JP decode: any malformed byte becomes `U+FFFD`. See
/// [`crate::decoder::Decoder::decode_to_string_lossy`].
pub(crate) fn decode_lossy(
    state: &mut EucJpState,
    input: &[u8],
    dst: &mut String,
    last: bool,
) -> usize {
    let mut idx = 0usize;
    while idx < input.len() {
        let byte = input[idx];
        match step(state, byte) {
            StepOutcome::Continue => idx += 1,
            StepOutcome::Emit(c) => {
                dst.push(c);
                idx += 1;
            }
            StepOutcome::ErrorHere => {
                dst.push(char::REPLACEMENT_CHARACTER);
                idx += 1;
            }
            StepOutcome::ErrorRestoreCurrent => {
                dst.push(char::REPLACEMENT_CHARACTER);
                // `byte` is reprocessed fresh on the next iteration (no `idx` advance).
            }
            StepOutcome::ErrorConsumeBoth => {
                dst.push(char::REPLACEMENT_CHARACTER);
                idx += 1;
            }
        }
    }
    if last && state.has_pending() {
        *state = EucJpState::new();
        dst.push(char::REPLACEMENT_CHARACTER);
    }
    input.len()
}

/// Encode `input` (valid UTF-8 text) as EUC-JP, per WHATWG §12.1.2.
pub(crate) fn encode(input: &str, dst: &mut Vec<u8>) -> Result<(), EncodeError> {
    for (position, character) in input.char_indices() {
        let codepoint = character as u32;
        if character.is_ascii() {
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
            dst.push(0x8E);
            dst.push((codepoint - 0xFF61 + 0xA1) as u8);
            continue;
        }
        let lookup_codepoint = if character == '\u{2212}' {
            0xFF0Du32
        } else {
            codepoint
        };
        let Some(pointer) = jis0208_encode_full(lookup_codepoint) else {
            return Err(EncodeError {
                character,
                position,
            });
        };
        let pointer = pointer as u32;
        let leading = pointer / 94 + 0xA1;
        let trailing = pointer % 94 + 0xA1;
        dst.push(leading as u8);
        dst.push(trailing as u8);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(chunks: &[&[u8]]) -> Result<String, DecodeError> {
        let mut state = EucJpState::new();
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
    fn ss2_half_width_katakana() -> Result<(), DecodeError> {
        // 0x8E (SS2) + 0xB1 = halfwidth ｱ (U+FF71), independently confirmed against the system
        // `iconv`'s EUC-JP table while writing this test.
        assert_eq!(decode_all(&[&[0x8E, 0xB1]])?, "\u{FF71}");
        Ok(())
    }

    #[test]
    fn ss2_followed_by_invalid_byte_is_an_error() {
        let mut state = EucJpState::new();
        let mut out = String::new();
        // 0x8E must be followed by 0xA1..=0xDF; 0x20 is not, and is ASCII, so it is restored.
        let err = decode(&mut state, &[0x8E, 0x20], &mut out, true).expect_err("must fail");
        assert_eq!(err.offset, 0);
        assert_eq!(err.length, 1);
    }

    #[test]
    fn ss3_followed_by_byte_outside_0xa1_0xfe_is_restored() {
        // 0x8F must be followed by 0xA1..=0xFE (to become the JIS X 0212 row byte); 0x20 is not,
        // and is ASCII, so -- exactly like the analogous SS2 case above -- it is restored
        // (reprocessed fresh) rather than consumed as part of the error.
        let mut state = EucJpState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x8F, 0x20], &mut out, true).expect_err("must fail");
        assert_eq!(err.offset, 0);
        assert_eq!(err.length, 1);

        let mut out = String::new();
        decode_lossy(&mut EucJpState::new(), &[0x8F, 0x20], &mut out, true);
        assert_eq!(out, "\u{FFFD} ");
    }

    #[test]
    fn ss3_jis0212_three_byte_sequence() -> Result<(), DecodeError> {
        // 0x8F (SS3) + 0xA2 0xAF: pointer 108 in JIS X 0212, U+02D8 (BREVE) -- the first entry in
        // the index, since pointer 0 is unassigned there. Independently confirmed against the
        // system `iconv`'s EUC-JP table while writing this test.
        assert_eq!(decode_all(&[&[0x8F, 0xA2, 0xAF]])?, "\u{02D8}");
        Ok(())
    }

    #[test]
    fn ss3_followed_by_unassigned_pointer_is_an_error() {
        // Pointer 0 (leading=0xA1, trailing=0xA1) is unassigned in JIS X 0212.
        let mut state = EucJpState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x8F, 0xA1, 0xA1], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
    }

    #[test]
    fn well_known_kanji_and_kana_bytes() -> Result<(), DecodeError> {
        // "日" (U+65E5) and "あ" (U+3042), independently confirmed against the system `iconv`'s
        // EUC-JP table while generating these tables.
        assert_eq!(decode_all(&[&[0xC6, 0xFC]])?, "\u{65E5}");
        assert_eq!(decode_all(&[&[0xA4, 0xA2]])?, "\u{3042}");
        Ok(())
    }

    #[test]
    fn truncated_lead_byte_at_true_end_of_stream() {
        let mut state = EucJpState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0xA4], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    }

    #[test]
    fn lead_byte_buffered_across_chunk_boundary() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[&[0xA4], &[0xA2]])?, "\u{3042}");
        Ok(())
    }

    #[test]
    fn invalid_lead_byte_is_an_error() {
        // 0x80 is not ASCII, not 0x8E/0x8F, and not in 0xA1..=0xFE.
        let mut state = EucJpState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x80], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
        assert_eq!(err.offset, 0);
    }

    #[test]
    fn encode_ascii_yen_and_overline() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("a\u{00A5}\u{203E}", &mut dst)?;
        assert_eq!(dst, vec![b'a', 0x5C, 0x7E]);
        Ok(())
    }

    #[test]
    fn encode_well_known_kanji() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("日", &mut dst)?;
        assert_eq!(dst, vec![0xC6, 0xFC]);
        Ok(())
    }

    #[test]
    fn encode_half_width_katakana_uses_ss2() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("\u{FF71}", &mut dst)?;
        assert_eq!(dst, vec![0x8E, 0xB1]);
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
        let text = "Hello, 世界! ネコ 日本語テスト123 ｱｲｳ";
        let mut bytes = Vec::new();
        encode(text, &mut bytes)?;
        for chunk_size in 1..=3 {
            let mut state = EucJpState::new();
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
        let mut state = EucJpState::new();
        let mut out = String::new();
        let consumed = decode_lossy(&mut state, &all_bytes, &mut out, true);
        assert_eq!(consumed, 256);
    }
}

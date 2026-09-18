//! ISO-2022-JP decoder and encoder, per the
//! [WHATWG Encoding Standard §12.2](https://encoding.spec.whatwg.org/#iso-2022-jp-decoder).
//!
//! Unlike Shift_JIS and EUC-JP, ISO-2022-JP is a *stateful, 7-bit* encoding: which characters a
//! given byte represents depends on which of six "shift states" the stream is currently in, and
//! that state is itself changed mid-stream by three-byte `ESC`-prefixed escape sequences (never
//! by high-bit-set bytes, since none ever appear). The decoder below is a direct, byte-at-a-time
//! transliteration of the spec's own state machine; see [`Mode`] for the six states and [`step`]
//! for the transition table.

use crate::decoder::{DecodeError, DecodeErrorKind};
use crate::encoder::EncodeError;
use crate::tables::{iso2022jp_katakana_replacement, jis0208_decode, jis0208_encode_full};

/// The six states of "ISO-2022-JP decoder state" / "ISO-2022-JP decoder output state" (WHATWG
/// §12.2.1). `TrailingByte`, `EscapeStart`, and `Escape` are the three *pending* states: a
/// [`Iso2022JpState`] sitting in one of them is holding onto part of an incomplete sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Ascii,
    Roman,
    Katakana,
    LeadingByte,
    TrailingByte,
    EscapeStart,
    Escape,
}

impl Mode {
    const fn is_pending(self) -> bool {
        matches!(self, Mode::TrailingByte | Mode::EscapeStart | Mode::Escape)
    }
}

/// State carried between [`decode`]/[`decode_lossy`] calls.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Iso2022JpState {
    mode: Mode,
    /// "ISO-2022-JP decoder output state": the state an aborted escape sequence falls back to.
    output_mode: Mode,
    /// "ISO-2022-JP leading": reused both for an escape designator's lead byte (`0x24`/`0x28`)
    /// and for a JIS X 0208 row byte awaiting its trailing byte, exactly as the spec reuses a
    /// single field for both.
    leading: u8,
    /// "ISO-2022-JP output": true immediately after a *successful* escape-designator switch that
    /// has not yet been followed by any actually-decoded byte; used to reject two such switches
    /// in a row with nothing decoded between them.
    output: bool,
}

impl Iso2022JpState {
    pub(crate) const fn new() -> Self {
        Iso2022JpState {
            mode: Mode::Ascii,
            output_mode: Mode::Ascii,
            leading: 0,
            output: false,
        }
    }

    pub(crate) const fn has_pending(&self) -> bool {
        self.mode.is_pending()
    }
}

/// The result of feeding one byte to [`step`].
enum StepOutcome {
    /// Byte consumed; no output yet (mid-sequence).
    Continue,
    /// Byte consumed; one character decoded.
    Emit(char),
    /// This byte alone is a self-contained malformed unit (length 1, at the current position).
    ErrorAt,
    /// The span from the currently open unit's start through this byte (inclusive) is malformed;
    /// both ends are consumed, nothing is left to reprocess.
    ErrorSpan,
    /// Only the byte at the currently open unit's start is malformed (a lone/dangling byte); this
    /// byte is left unconsumed for the caller to reprocess.
    ErrorRestoreOne,
    /// Only the byte at the currently open unit's start (an `ESC` that began a 3-byte escape
    /// designator) is malformed; both the designator-lead byte immediately after it *and* this
    /// byte are left unconsumed for the caller to reprocess.
    ErrorRestoreTwo,
    /// `TrailingByte` state hit an unexpected `ESC`: report an error about the *old* dangling
    /// unit (whose start is carried in this variant), while this byte is consumed as the start of
    /// a brand new escape sequence (the caller's `unit_start` has already been updated to it).
    ErrorDanglingThenAdvance(Option<usize>),
}

/// Advance the state machine by one byte. `unit_start` tracks the position (within the *current*
/// call's `input`) at which the currently-pending unit began, or `None` if it began during an
/// earlier call (see the offset-clamping convention documented on [`DecodeError`]); it is updated
/// in place whenever a new unit opens.
fn step(
    state: &mut Iso2022JpState,
    unit_start: &mut Option<usize>,
    idx: usize,
    byte: u8,
) -> StepOutcome {
    match state.mode {
        Mode::Ascii => {
            if byte == 0x1B {
                *unit_start = Some(idx);
                state.mode = Mode::EscapeStart;
                StepOutcome::Continue
            } else if byte <= 0x7F && byte != 0x0E && byte != 0x0F {
                state.output = false;
                StepOutcome::Emit(byte as char)
            } else {
                state.output = false;
                StepOutcome::ErrorAt
            }
        }
        Mode::Roman => {
            if byte == 0x1B {
                *unit_start = Some(idx);
                state.mode = Mode::EscapeStart;
                StepOutcome::Continue
            } else if byte == 0x5C {
                state.output = false;
                StepOutcome::Emit('\u{00A5}')
            } else if byte == 0x7E {
                state.output = false;
                StepOutcome::Emit('\u{203E}')
            } else if byte <= 0x7F && !matches!(byte, 0x0E | 0x0F | 0x5C | 0x7E) {
                state.output = false;
                StepOutcome::Emit(byte as char)
            } else {
                state.output = false;
                StepOutcome::ErrorAt
            }
        }
        Mode::Katakana => {
            if byte == 0x1B {
                *unit_start = Some(idx);
                state.mode = Mode::EscapeStart;
                StepOutcome::Continue
            } else if (0x21..=0x5F).contains(&byte) {
                state.output = false;
                let cp = 0xFF61u32 - 0x21 + byte as u32;
                StepOutcome::Emit(char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER))
            } else {
                state.output = false;
                StepOutcome::ErrorAt
            }
        }
        Mode::LeadingByte => {
            if byte == 0x1B {
                *unit_start = Some(idx);
                state.mode = Mode::EscapeStart;
                StepOutcome::Continue
            } else if (0x21..=0x7E).contains(&byte) {
                state.output = false;
                state.leading = byte;
                *unit_start = Some(idx);
                state.mode = Mode::TrailingByte;
                StepOutcome::Continue
            } else {
                state.output = false;
                StepOutcome::ErrorAt
            }
        }
        Mode::TrailingByte => {
            if byte == 0x1B {
                let old_unit_start = *unit_start;
                *unit_start = Some(idx);
                state.mode = Mode::EscapeStart;
                StepOutcome::ErrorDanglingThenAdvance(old_unit_start)
            } else if (0x21..=0x7E).contains(&byte) {
                state.mode = Mode::LeadingByte;
                let pointer = (state.leading as u32 - 0x21) * 94 + byte as u32 - 0x21;
                match jis0208_decode(pointer) {
                    Some(c) => StepOutcome::Emit(c),
                    None => StepOutcome::ErrorSpan,
                }
            } else {
                state.mode = Mode::LeadingByte;
                StepOutcome::ErrorSpan
            }
        }
        Mode::EscapeStart => {
            if byte == 0x24 || byte == 0x28 {
                state.leading = byte;
                state.mode = Mode::Escape;
                StepOutcome::Continue
            } else {
                state.output = false;
                state.mode = state.output_mode;
                StepOutcome::ErrorRestoreOne
            }
        }
        Mode::Escape => {
            let leading = state.leading;
            state.leading = 0;
            let new_mode = match (leading, byte) {
                (0x28, 0x42) => Some(Mode::Ascii),
                (0x28, 0x4A) => Some(Mode::Roman),
                (0x28, 0x49) => Some(Mode::Katakana),
                (0x24, 0x40) | (0x24, 0x42) => Some(Mode::LeadingByte),
                _ => None,
            };
            match new_mode {
                Some(m) => {
                    state.mode = m;
                    state.output_mode = m;
                    let prev_output = state.output;
                    state.output = true;
                    if prev_output {
                        StepOutcome::ErrorSpan
                    } else {
                        StepOutcome::Continue
                    }
                }
                None => {
                    state.output = false;
                    state.mode = state.output_mode;
                    StepOutcome::ErrorRestoreTwo
                }
            }
        }
    }
}

/// Strict incremental ISO-2022-JP decode. See [`crate::decoder::Decoder::decode_to_string`].
pub(crate) fn decode(
    state: &mut Iso2022JpState,
    input: &[u8],
    dst: &mut String,
    last: bool,
) -> Result<usize, DecodeError> {
    let mut unit_start: Option<usize> = None;
    let mut idx = 0usize;
    while idx < input.len() {
        let byte = input[idx];
        match step(state, &mut unit_start, idx, byte) {
            StepOutcome::Continue => idx += 1,
            StepOutcome::Emit(c) => {
                dst.push(c);
                idx += 1;
            }
            StepOutcome::ErrorAt => {
                *state = Iso2022JpState::new();
                return Err(DecodeError {
                    offset: idx,
                    length: 1,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
            StepOutcome::ErrorSpan => {
                let start = unit_start.unwrap_or(idx);
                *state = Iso2022JpState::new();
                return Err(DecodeError {
                    offset: start,
                    length: idx - start + 1,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
            StepOutcome::ErrorRestoreOne | StepOutcome::ErrorRestoreTwo => {
                let start = unit_start.unwrap_or(idx);
                *state = Iso2022JpState::new();
                return Err(DecodeError {
                    offset: start,
                    length: 1,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
            StepOutcome::ErrorDanglingThenAdvance(old_start) => {
                let start = old_start.unwrap_or(idx);
                *state = Iso2022JpState::new();
                return Err(DecodeError {
                    offset: start,
                    length: 1,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
        }
    }
    if last && state.mode.is_pending() {
        let start = unit_start.unwrap_or(0);
        *state = Iso2022JpState::new();
        return Err(DecodeError {
            offset: start,
            length: 1,
            kind: DecodeErrorKind::TruncatedSequence,
        });
    }
    Ok(input.len())
}

/// Lossy incremental ISO-2022-JP decode: any malformed byte (or double no-op escape) becomes one
/// `U+FFFD`. See [`crate::decoder::Decoder::decode_to_string_lossy`].
pub(crate) fn decode_lossy(
    state: &mut Iso2022JpState,
    input: &[u8],
    dst: &mut String,
    last: bool,
) -> usize {
    let mut unit_start: Option<usize> = None;
    let mut idx = 0usize;
    while idx < input.len() {
        let byte = input[idx];
        match step(state, &mut unit_start, idx, byte) {
            StepOutcome::Continue => idx += 1,
            StepOutcome::Emit(c) => {
                dst.push(c);
                idx += 1;
            }
            StepOutcome::ErrorAt | StepOutcome::ErrorSpan => {
                dst.push(char::REPLACEMENT_CHARACTER);
                idx += 1;
            }
            StepOutcome::ErrorRestoreOne => {
                dst.push(char::REPLACEMENT_CHARACTER);
                // `byte` is reprocessed fresh on the next iteration (no `idx` advance).
            }
            StepOutcome::ErrorRestoreTwo => {
                dst.push(char::REPLACEMENT_CHARACTER);
                // Reprocess both the designator-lead byte and this one: rewind by one extra step
                // (saturating: if the designator-lead byte was itself from a previous call, there
                // is nothing earlier in `input` to rewind to, so just reprocess this byte alone).
                idx = idx.saturating_sub(1);
            }
            StepOutcome::ErrorDanglingThenAdvance(_) => {
                dst.push(char::REPLACEMENT_CHARACTER);
                idx += 1;
            }
        }
    }
    if last {
        if state.mode.is_pending() {
            dst.push(char::REPLACEMENT_CHARACTER);
        }
        *state = Iso2022JpState::new();
    }
    input.len()
}

/// The three states of "ISO-2022-JP encoder state" (WHATWG §12.2.2).
#[derive(Clone, Copy, PartialEq, Eq)]
enum EncMode {
    Ascii,
    Roman,
    Jis0208,
}

/// Encode `input` (valid UTF-8 text) as ISO-2022-JP, per WHATWG §12.2.2.
///
/// Unlike the other two Japanese encoders, this one is inherently stateful *within a single
/// call*: which escape sequences (if any) precede a given character's bytes depends on every
/// character encoded before it. Since [`encode`] always processes one complete `&str` per call,
/// that state is simply a local variable here — there is no need to expose it across calls the
/// way [`crate::decoder::Decoder`] exposes incremental decode state.
pub(crate) fn encode(input: &str, dst: &mut Vec<u8>) -> Result<(), EncodeError> {
    let mut mode = EncMode::Ascii;
    for (position, character) in input.char_indices() {
        let mut codepoint = character as u32;
        loop {
            if matches!(mode, EncMode::Ascii | EncMode::Roman)
                && matches!(codepoint, 0x0E | 0x0F | 0x1B)
            {
                // "This returns U+FFFD rather than codePoint to prevent attacks."
                return Err(EncodeError {
                    character: char::REPLACEMENT_CHARACTER,
                    position,
                });
            }
            if mode == EncMode::Ascii && codepoint < 0x80 {
                dst.push(codepoint as u8);
                break;
            }
            if mode == EncMode::Roman {
                if codepoint < 0x80 && codepoint != 0x5C && codepoint != 0x7E {
                    dst.push(codepoint as u8);
                    break;
                }
                if codepoint == 0x00A5 {
                    dst.push(0x5C);
                    break;
                }
                if codepoint == 0x203E {
                    dst.push(0x7E);
                    break;
                }
            }
            if codepoint < 0x80 && mode != EncMode::Ascii {
                dst.extend_from_slice(&[0x1B, 0x28, 0x42]);
                mode = EncMode::Ascii;
                continue;
            }
            if (codepoint == 0x00A5 || codepoint == 0x203E) && mode != EncMode::Roman {
                dst.extend_from_slice(&[0x1B, 0x28, 0x4A]);
                mode = EncMode::Roman;
                continue;
            }
            if codepoint == 0x2212 {
                codepoint = 0xFF0D;
            }
            if (0xFF61..=0xFF9F).contains(&codepoint) {
                codepoint = iso2022jp_katakana_replacement(codepoint);
            }
            let pointer = jis0208_encode_full(codepoint);
            let Some(pointer) = pointer else {
                if mode == EncMode::Jis0208 {
                    dst.extend_from_slice(&[0x1B, 0x28, 0x42]);
                    mode = EncMode::Ascii;
                    continue;
                }
                return Err(EncodeError {
                    character,
                    position,
                });
            };
            if mode != EncMode::Jis0208 {
                dst.extend_from_slice(&[0x1B, 0x24, 0x42]);
                mode = EncMode::Jis0208;
                continue;
            }
            let pointer = pointer as u32;
            let leading = pointer / 94 + 0x21;
            let trailing = pointer % 94 + 0x21;
            dst.push(leading as u8);
            dst.push(trailing as u8);
            break;
        }
    }
    if mode != EncMode::Ascii {
        dst.extend_from_slice(&[0x1B, 0x28, 0x42]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(chunks: &[&[u8]]) -> Result<String, DecodeError> {
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            decode(&mut state, chunk, &mut out, last)?;
        }
        Ok(out)
    }

    #[test]
    fn plain_ascii_needs_no_escapes() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[b"Hello, world!"])?, "Hello, world!");
        Ok(())
    }

    #[test]
    fn kanji_via_designator_and_reset() -> Result<(), DecodeError> {
        // ESC $ B (shift to jis0208), 0x46 0x7C (pointer (0x46-0x21)*94+(0x7C-0x21) = 3569 --
        // U+65E5 "日"), ESC ( B (shift back to ASCII), then "!". The full byte sequence is
        // independently confirmed against the system `iconv`'s ISO-2022-JP table (`printf '日' |
        // iconv -f UTF-8 -t ISO-2022-JP` emits exactly these bytes, sans the trailing "!").
        let bytes: &[u8] = &[0x1B, 0x24, 0x42, 0x46, 0x7C, 0x1B, 0x28, 0x42, b'!'];
        assert_eq!(decode_all(&[bytes])?, "日!");
        Ok(())
    }

    #[test]
    fn roman_yen_and_overline() -> Result<(), DecodeError> {
        // ESC ( J (shift to Roman), 0x5C (-> ¥), 0x7E (-> ‾), ESC ( B (back to ASCII).
        let bytes: &[u8] = &[0x1B, 0x28, 0x4A, 0x5C, 0x7E, 0x1B, 0x28, 0x42];
        assert_eq!(decode_all(&[bytes])?, "\u{00A5}\u{203E}");
        Ok(())
    }

    #[test]
    fn katakana_shift() -> Result<(), DecodeError> {
        // ESC ( I (shift to katakana), 0x21 -> U+FF61.
        let bytes: &[u8] = &[0x1B, 0x28, 0x49, 0x21];
        assert_eq!(decode_all(&[bytes])?, "\u{FF61}");
        Ok(())
    }

    #[test]
    fn unexpected_esc_designator_byte_is_rejected_strictly() {
        // ESC X: X (0x58) is neither 0x24 nor 0x28, so escape-start rejects it.
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x1B, b'X'], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
        assert_eq!(err.offset, 0);
        assert_eq!(err.length, 1);
    }

    #[test]
    fn unexpected_esc_designator_byte_is_restored_in_lossy_mode() {
        // Per spec, the offending byte after ESC is *not* consumed by escape-start on failure --
        // it is reprocessed fresh in the (unchanged) output state, i.e. plain ASCII here.
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0x1B, b'X'], &mut out, true);
        assert_eq!(out, "\u{FFFD}X");
    }

    #[test]
    fn unexpected_esc_mid_designator_restores_both_bytes_in_lossy_mode() {
        // ESC $ X: X (0x58) matches neither of the two valid jis0208 designators (0x40, 0x42).
        // Per spec both the designator-lead byte (0x24) and X are restored/reprocessed as plain
        // ASCII once the encoder falls back to its output state.
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0x1B, 0x24, b'X', b'A'], &mut out, true);
        assert_eq!(out, "\u{FFFD}$XA");
    }

    #[test]
    fn esc_while_awaiting_trailing_byte_reports_the_dangling_leading_byte() {
        // ESC $ B (shift to jis0208), 0x46 (a valid leading byte, now awaiting its trailing
        // byte), then ESC instead of a trailing byte: the dangling leading byte is the error: the
        // ESC itself is legitimately consumed to start a *new* escape sequence.
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        let err = decode(
            &mut state,
            &[0x1B, 0x24, 0x42, 0x46, 0x1B, 0x28, 0x42],
            &mut out,
            true,
        )
        .expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
        assert_eq!(err.length, 1);
    }

    #[test]
    fn esc_while_awaiting_trailing_byte_recovers_and_continues_in_lossy_mode() {
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        decode_lossy(
            &mut state,
            &[0x1B, 0x24, 0x42, 0x46, 0x1B, 0x28, 0x42, b'!'],
            &mut out,
            true,
        );
        // The dangling leading byte (0x46) errors out; the subsequent ESC ( B cleanly resets to
        // ASCII, and "!" decodes normally afterward.
        assert_eq!(out, "\u{FFFD}!");
    }

    #[test]
    fn invalid_byte_in_ascii_state_is_rejected() {
        // 0x80 (high bit set) never appears in valid ISO-2022-JP.
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x80], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
        assert_eq!(err.offset, 0);
        assert_eq!(err.length, 1);
    }

    #[test]
    fn truncated_escape_sequence_at_true_end_of_stream() {
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x1B], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);

        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x1B, 0x24], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);

        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        let err =
            decode(&mut state, &[0x1B, 0x24, 0x42, 0x46], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    }

    #[test]
    fn escape_sequence_split_across_every_chunk_boundary() -> Result<(), DecodeError> {
        let bytes: &[u8] = &[0x1B, 0x24, 0x42, 0x46, 0x7C, 0x1B, 0x28, 0x42];
        for split1 in 1..bytes.len() {
            for split2 in split1..bytes.len() {
                let chunks: [&[u8]; 3] =
                    [&bytes[..split1], &bytes[split1..split2], &bytes[split2..]];
                assert_eq!(
                    decode_all(&chunks)?,
                    "日",
                    "split1={split1} split2={split2}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn double_no_op_escape_designator_is_rejected() {
        // ESC ( J ESC ( J: two Roman-mode designators back to back with nothing decoded between
        // them is specifically disallowed (WHATWG's "ISO-2022-JP output" bookkeeping).
        let bytes: &[u8] = &[0x1B, 0x28, 0x4A, 0x1B, 0x28, 0x4A];
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        let err = decode(&mut state, bytes, &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
        assert_eq!(err.offset, 3);
        assert_eq!(err.length, 3);
    }

    #[test]
    fn real_character_between_two_designators_is_fine() {
        // ESC ( J (Roman) 0x41 ('A', a plain Roman-mode ASCII byte) ESC ( J (Roman again) --
        // legal, since a real byte was decoded between the two designator switches.
        let bytes: &[u8] = &[0x1B, 0x28, 0x4A, b'A', 0x1B, 0x28, 0x4A];
        assert_eq!(decode_all(&[bytes]).expect("must decode"), "A");
    }

    #[test]
    fn encode_ascii_only() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("Hello!", &mut dst)?;
        assert_eq!(dst, b"Hello!");
        Ok(())
    }

    #[test]
    fn encode_kanji_switches_to_jis0208_and_back() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("日!", &mut dst)?;
        assert_eq!(
            dst,
            vec![0x1B, 0x24, 0x42, 0x46, 0x7C, 0x1B, 0x28, 0x42, b'!']
        );
        Ok(())
    }

    #[test]
    fn encode_yen_and_overline_use_roman_designator() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("\u{00A5}\u{203E}", &mut dst)?;
        assert_eq!(dst, vec![0x1B, 0x28, 0x4A, 0x5C, 0x7E, 0x1B, 0x28, 0x42]);
        Ok(())
    }

    #[test]
    fn encode_literal_backslash_after_roman_switches_back_to_ascii_first() -> Result<(), EncodeError>
    {
        // A literal backslash cannot be represented directly in Roman mode (0x5C means yen
        // there), so the encoder must shift back to ASCII before emitting it.
        let mut dst = Vec::new();
        encode("\u{00A5}\\", &mut dst)?;
        assert_eq!(dst, vec![0x1B, 0x28, 0x4A, 0x5C, 0x1B, 0x28, 0x42, b'\\']);
        Ok(())
    }

    #[test]
    fn encode_rejects_forbidden_control_codes_with_u_fffd() {
        let mut dst = Vec::new();
        let err = encode("\u{001B}", &mut dst).expect_err("must fail");
        assert_eq!(err.character, char::REPLACEMENT_CHARACTER);
    }

    #[test]
    fn encode_rejects_unmappable_character() {
        let mut dst = Vec::new();
        let err = encode("😀", &mut dst).expect_err("must fail");
        assert_eq!(err.character, '😀');
    }

    #[test]
    fn round_trip_mixed_text_at_every_chunk_size() -> Result<(), Box<dyn std::error::Error>> {
        let text = "Hello, 世界! 日本語のテスト123 end.";
        let mut bytes = Vec::new();
        encode(text, &mut bytes)?;
        for chunk_size in 1..=3 {
            let mut state = Iso2022JpState::new();
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
    fn fresh_state_has_no_pending_sequence() {
        assert!(!Iso2022JpState::new().has_pending());
    }

    #[test]
    fn lossy_decode_never_panics_on_arbitrary_bytes() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        let consumed = decode_lossy(&mut state, &all_bytes, &mut out, true);
        assert_eq!(consumed, 256);
    }

    #[test]
    fn lossy_decode_byte_at_a_time_never_panics_and_consumes_everything() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        let mut state = Iso2022JpState::new();
        let mut out = String::new();
        let mut total = 0usize;
        for (i, &b) in all_bytes.iter().enumerate() {
            let last = i + 1 == all_bytes.len();
            total += decode_lossy(&mut state, &[b], &mut out, last);
        }
        assert_eq!(total, 256);
    }
}

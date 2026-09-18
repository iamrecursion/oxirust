//! GBK and gb18030 decoder and encoder, per the
//! [WHATWG Encoding Standard §10](https://encoding.spec.whatwg.org/#gb18030-decoder).
//!
//! Per the spec, "GBK's decoder is gb18030's decoder": the two share [`decode`]/[`decode_lossy`]
//! outright. Only the encoder differs, via the `is_gbk` flag on [`encode`]: GBK refuses to fall
//! back to the 4-byte algorithmic ranges gb18030 uses for code points outside its 2-byte
//! repertoire, and additionally special-cases U+20AC to the single byte `0x80`.

use crate::decoder::{DecodeError, DecodeErrorKind};
use crate::encoder::EncodeError;
use crate::tables::{
    gb18030_decode_2byte, gb18030_encode_2byte, gb18030_ranges_decode, gb18030_ranges_encode,
};

/// State carried between [`decode`]/[`decode_lossy`] calls: up to three buffered lead bytes of
/// an in-progress 2- or 4-byte sequence (`first`, then `second` and `third` once a 4-byte
/// sequence is confirmed), each `0x00` when not pending. Named after the WHATWG spec's own
/// "gb18030 first"/"gb18030 second"/"gb18030 third" state fields.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Gb18030State {
    first: u8,
    second: u8,
    third: u8,
}

impl Gb18030State {
    pub(crate) const fn new() -> Self {
        Gb18030State {
            first: 0,
            second: 0,
            third: 0,
        }
    }

    pub(crate) const fn has_pending(&self) -> bool {
        self.first != 0 || self.second != 0 || self.third != 0
    }
}

/// Outcome of feeding one byte through the gb18030 state machine ([`step`]).
enum StepOutcome {
    /// The byte extended an in-progress sequence; more bytes are needed.
    Continue,
    /// A complete sequence (of any length) resolved to a code point.
    Emit(char),
    /// No buffered state; `byte` alone is invalid (WHATWG's final "Return error" fallthrough).
    ErrorHere,
    /// A 2-byte sequence's trailing byte was invalid, but is itself ASCII, so it is restored (to
    /// be reprocessed fresh); the malformed unit is just the lead byte.
    ErrorRestoreCurrent,
    /// A 2-byte sequence's trailing byte was invalid and non-ASCII, so both bytes are consumed as
    /// the malformed unit.
    ErrorConsumeBoth,
    /// A 4-byte sequence's third byte (the one following the `0x30..=0x39` "second" byte) was not
    /// a valid lead byte (`0x81..=0xFE`); per spec this unconditionally restores the buffered
    /// `second` byte followed by the current byte (returned here as the tuple's single field),
    /// regardless of whether the current byte is ASCII. The malformed unit is just the original
    /// lead byte.
    ErrorRestoreOne(u8),
    /// A 4-byte sequence's fourth byte was not a digit (`0x30..=0x39`); per spec this
    /// unconditionally restores the buffered `second` and `third` bytes followed by the current
    /// byte. The malformed unit is just the original lead byte.
    ErrorRestoreTwo(u8, u8),
    /// A 4-byte sequence was structurally well-formed (all four bytes in their required ranges)
    /// but its linear pointer has no assigned code point in [`crate::tables::gb18030_ranges_decode`].
    /// All four bytes are consumed as the malformed unit.
    ErrorRangesUnassigned,
}

/// Feed one byte through the gb18030 decoder state machine, per WHATWG §10.2.1. Shared by strict
/// and lossy decoding, and by GBK (whose decoder is defined as gb18030's decoder verbatim).
fn step(state: &mut Gb18030State, byte: u8) -> StepOutcome {
    if state.third != 0 {
        let (first, second, third) = (state.first, state.second, state.third);
        state.first = 0;
        state.second = 0;
        state.third = 0;
        if !(0x30..=0x39).contains(&byte) {
            return StepOutcome::ErrorRestoreTwo(second, third);
        }
        let pointer = (first as u32 - 0x81) * (10 * 126 * 10)
            + (second as u32 - 0x30) * (10 * 126)
            + (third as u32 - 0x81) * 10
            + (byte as u32 - 0x30);
        return match gb18030_ranges_decode(pointer) {
            Some(c) => StepOutcome::Emit(c),
            None => StepOutcome::ErrorRangesUnassigned,
        };
    }
    if state.second != 0 {
        if (0x81..=0xFE).contains(&byte) {
            state.third = byte;
            return StepOutcome::Continue;
        }
        let second = state.second;
        state.first = 0;
        state.second = 0;
        return StepOutcome::ErrorRestoreOne(second);
    }
    if state.first != 0 {
        if (0x30..=0x39).contains(&byte) {
            state.second = byte;
            return StepOutcome::Continue;
        }
        let leading = state.first;
        state.first = 0;
        let offset: u32 = if byte < 0x7F { 0x40 } else { 0x41 };
        let in_range = (0x40..=0x7E).contains(&byte) || (0x80..=0xFE).contains(&byte);
        let codepoint = if in_range {
            gb18030_decode_2byte((leading as u32 - 0x81) * 190 + (byte as u32 - offset))
        } else {
            None
        };
        return match codepoint {
            Some(c) => StepOutcome::Emit(c),
            None if byte.is_ascii() => StepOutcome::ErrorRestoreCurrent,
            None => StepOutcome::ErrorConsumeBoth,
        };
    }
    if byte.is_ascii() {
        return StepOutcome::Emit(byte as char);
    }
    if byte == 0x80 {
        return StepOutcome::Emit('\u{20AC}');
    }
    if (0x81..=0xFE).contains(&byte) {
        state.first = byte;
        return StepOutcome::Continue;
    }
    StepOutcome::ErrorHere
}

/// Strict incremental GBK/gb18030 decode. See [`crate::decoder::Decoder::decode_to_string`].
pub(crate) fn decode(
    state: &mut Gb18030State,
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
            StepOutcome::ErrorRestoreOne(_) => {
                return Err(DecodeError {
                    offset: idx.saturating_sub(2),
                    length: 1,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
            StepOutcome::ErrorRestoreTwo(_, _) => {
                return Err(DecodeError {
                    offset: idx.saturating_sub(3),
                    length: 1,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
            StepOutcome::ErrorRangesUnassigned => {
                return Err(DecodeError {
                    offset: idx.saturating_sub(3),
                    length: 4,
                    kind: DecodeErrorKind::MalformedSequence,
                });
            }
        }
    }
    if last && state.has_pending() {
        *state = Gb18030State::new();
        return Err(DecodeError {
            offset: input.len().saturating_sub(1),
            length: 1,
            kind: DecodeErrorKind::TruncatedSequence,
        });
    }
    Ok(input.len())
}

/// Lossy incremental GBK/gb18030 decode: any malformed byte becomes `U+FFFD`. See
/// [`crate::decoder::Decoder::decode_to_string_lossy`].
///
/// Unlike the simpler Japanese multi-byte encodings, gb18030's "restore" steps can push back up
/// to two bytes that were consumed in a *previous* call (buffered in `state`), not just the
/// current byte -- `replay` is a small local queue (never persisted in `state`, since it is
/// always fully drained by the time this function returns) that those restored bytes are fed
/// back through [`step`] from, ahead of any unread `input`.
pub(crate) fn decode_lossy(
    state: &mut Gb18030State,
    input: &[u8],
    dst: &mut String,
    last: bool,
) -> usize {
    let mut idx = 0usize;
    let mut replay = [0u8; 3];
    let mut replay_len = 0usize;
    let mut replay_pos = 0usize;
    loop {
        let byte = if replay_pos < replay_len {
            let b = replay[replay_pos];
            replay_pos += 1;
            b
        } else if idx < input.len() {
            let b = input[idx];
            idx += 1;
            b
        } else {
            break;
        };
        match step(state, byte) {
            StepOutcome::Continue => {}
            StepOutcome::Emit(c) => dst.push(c),
            StepOutcome::ErrorHere => dst.push(char::REPLACEMENT_CHARACTER),
            StepOutcome::ErrorRestoreCurrent => {
                dst.push(char::REPLACEMENT_CHARACTER);
                replay[0] = byte;
                replay_len = 1;
                replay_pos = 0;
            }
            StepOutcome::ErrorConsumeBoth => dst.push(char::REPLACEMENT_CHARACTER),
            StepOutcome::ErrorRestoreOne(b0) => {
                dst.push(char::REPLACEMENT_CHARACTER);
                replay[0] = b0;
                replay[1] = byte;
                replay_len = 2;
                replay_pos = 0;
            }
            StepOutcome::ErrorRestoreTwo(b0, b1) => {
                dst.push(char::REPLACEMENT_CHARACTER);
                replay[0] = b0;
                replay[1] = b1;
                replay[2] = byte;
                replay_len = 3;
                replay_pos = 0;
            }
            StepOutcome::ErrorRangesUnassigned => dst.push(char::REPLACEMENT_CHARACTER),
        }
    }
    if last && state.has_pending() {
        *state = Gb18030State::new();
        dst.push(char::REPLACEMENT_CHARACTER);
    }
    input.len()
}

/// The 18 code points GB18030's encoder maps to fixed legacy byte pairs instead of consulting the
/// ordinary 2-byte index, per the WHATWG gb18030 encoder algorithm's own table: "This asymmetric
/// encoder table preserves compatibility with the GB18030-2005 standard." Checked *before* the
/// ordinary 2-byte lookup.
const PUA_ENCODE_OVERRIDE: [(u32, [u8; 2]); 18] = [
    (0xE78D, [0xA6, 0xD9]),
    (0xE78E, [0xA6, 0xDA]),
    (0xE78F, [0xA6, 0xDB]),
    (0xE790, [0xA6, 0xDC]),
    (0xE791, [0xA6, 0xDD]),
    (0xE792, [0xA6, 0xDE]),
    (0xE793, [0xA6, 0xDF]),
    (0xE794, [0xA6, 0xEC]),
    (0xE795, [0xA6, 0xED]),
    (0xE796, [0xA6, 0xF3]),
    (0xE81E, [0xFE, 0x59]),
    (0xE826, [0xFE, 0x61]),
    (0xE82B, [0xFE, 0x66]),
    (0xE82C, [0xFE, 0x67]),
    (0xE832, [0xFE, 0x6D]),
    (0xE843, [0xFE, 0x7E]),
    (0xE854, [0xFE, 0x90]),
    (0xE864, [0xFE, 0xA0]),
];

fn pua_encode_override(codepoint: u32) -> Option<[u8; 2]> {
    PUA_ENCODE_OVERRIDE
        .iter()
        .find(|&&(cp, _)| cp == codepoint)
        .map(|&(_, bytes)| bytes)
}

/// Encode `input` (valid UTF-8 text) as GBK (`is_gbk == true`) or gb18030 (`is_gbk == false`),
/// per WHATWG §10.1.2/§10.2.2.
///
/// gb18030's *encoding* direction is total over Unicode: every character except `U+E5E5` (which
/// the spec documents as unmappable, since index gb18030 maps its own byte pair to U+3000 instead
/// for compatibility with deployed content) has a representation. GBK is not total: it errors on
/// any character outside its 2-byte repertoire rather than falling back to gb18030's 4-byte
/// ranges.
///
/// Totality of encoding does not imply every character round-trips through encode-then-decode,
/// though: [`PUA_ENCODE_OVERRIDE`] lists 18 Private Use Area code points that this encoder
/// deliberately maps to a fixed legacy byte pair -- "This asymmetric encoder table preserves
/// compatibility with the GB18030-2005 standard" (WHATWG) -- which the *decoder* resolves back to
/// whatever code point the current index actually assigns that byte pair, not the original
/// character. This is spec-mandated, not a bug; see
/// [`crate::tables::gb18030_decode_2byte`]/[`crate::tables::gb18030_encode_2byte`] for the
/// ordinary (round-tripping) 2-byte path these 18 code points bypass.
pub(crate) fn encode(input: &str, dst: &mut Vec<u8>, is_gbk: bool) -> Result<(), EncodeError> {
    for (position, character) in input.char_indices() {
        let codepoint = character as u32;
        if character.is_ascii() {
            dst.push(codepoint as u8);
            continue;
        }
        if character == '\u{E5E5}' {
            return Err(EncodeError {
                character,
                position,
            });
        }
        if is_gbk && character == '\u{20AC}' {
            dst.push(0x80);
            continue;
        }
        if let Some(bytes) = pua_encode_override(codepoint) {
            dst.extend_from_slice(&bytes);
            continue;
        }
        if let Some(pointer) = gb18030_encode_2byte(codepoint) {
            let pointer = pointer as u32;
            let leading = pointer / 190 + 0x81;
            let trailing = pointer % 190;
            let offset: u32 = if trailing < 0x3F { 0x40 } else { 0x41 };
            dst.push(leading as u8);
            dst.push((trailing + offset) as u8);
            continue;
        }
        if is_gbk {
            return Err(EncodeError {
                character,
                position,
            });
        }
        let Some(mut pointer) = gb18030_ranges_encode(codepoint) else {
            return Err(EncodeError {
                character,
                position,
            });
        };
        let byte1 = pointer / (10 * 126 * 10);
        pointer %= 10 * 126 * 10;
        let byte2 = pointer / (10 * 126);
        pointer %= 10 * 126;
        let byte3 = pointer / 10;
        let byte4 = pointer % 10;
        dst.push((byte1 + 0x81) as u8);
        dst.push((byte2 + 0x30) as u8);
        dst.push((byte3 + 0x81) as u8);
        dst.push((byte4 + 0x30) as u8);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(chunks: &[&[u8]]) -> Result<String, DecodeError> {
        let mut state = Gb18030State::new();
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
    fn byte_0x80_decodes_to_euro_sign() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[&[0x80]])?, "\u{20AC}");
        Ok(())
    }

    #[test]
    fn well_known_2byte_kanji() -> Result<(), DecodeError> {
        // "中" (U+4E2D), independently confirmed against Python's built-in `gbk`/`gb18030`
        // codecs while writing this test.
        assert_eq!(decode_all(&[&[0xD6, 0xD0]])?, "\u{4E2D}");
        Ok(())
    }

    #[test]
    fn four_byte_supplementary_plane_character() -> Result<(), DecodeError> {
        // U+20000 (a CJK Extension B ideograph), independently confirmed against Python's
        // built-in `gb18030` codec while writing this test.
        assert_eq!(decode_all(&[&[0x95, 0x32, 0x82, 0x36]])?, "\u{20000}");
        Ok(())
    }

    #[test]
    fn four_byte_pointer_7457_special_case() -> Result<(), DecodeError> {
        // pointer 7457 = 0*(10*126*10) + 5*(10*126) + 115*10 + 7, i.e. first=0x81, second=0x35,
        // third=0x81+115=0xF4, fourth=0x30+7=0x37. WHATWG special-cases this exact pointer to
        // U+E7C7 rather than the range table's own linear interpolation (see
        // `crate::tables::gb18030_ranges_decode`).
        assert_eq!(decode_all(&[&[0x81, 0x35, 0xF4, 0x37]])?, "\u{E7C7}");
        Ok(())
    }

    #[test]
    fn lead_byte_buffered_across_chunk_boundary() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[&[0xD6], &[0xD0]])?, "\u{4E2D}");
        Ok(())
    }

    #[test]
    fn four_byte_sequence_split_across_every_boundary() -> Result<(), DecodeError> {
        let bytes = [0x95u8, 0x32, 0x82, 0x36];
        for split in 1..bytes.len() {
            let (a, b) = bytes.split_at(split);
            assert_eq!(decode_all(&[a, b])?, "\u{20000}", "split at {split}");
        }
        Ok(())
    }

    #[test]
    fn truncated_lead_byte_at_true_end_of_stream() {
        let mut state = Gb18030State::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0xD6], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    }

    #[test]
    fn truncated_four_byte_sequence_at_true_end_of_stream() {
        let mut state = Gb18030State::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0x95, 0x32, 0x82], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    }

    #[test]
    fn state_resets_after_a_strict_error_mid_four_byte_sequence() {
        // `Gb18030State` carries three fields (`first`/`second`/`third`); unlike the simpler
        // one-field Japanese multi-byte states, an error partway through a 4-byte sequence must
        // clear all three, not just the field the error was detected in.
        let mut state = Gb18030State::new();
        let mut out = String::new();
        // 0x81 0x30 0x81 is a well-formed 4-byte prefix (lead, digit, lead-range byte); feed it
        // with `last = false` so the sequence is still pending.
        decode(&mut state, &[0x81, 0x30, 0x81], &mut out, false)
            .expect("must decode (not yet complete)");
        assert!(state.has_pending());
        // The true end of stream arrives with the sequence still incomplete.
        let err = decode(&mut state, &[], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
        // Per the `decode_to_string` contract, carried state is reset to empty after an error, so
        // the same `Decoder` (here, the same `Gb18030State`) is safely reusable for a fresh call.
        assert!(!state.has_pending());
        let mut out2 = String::new();
        decode(&mut state, b"OK", &mut out2, true).expect("must decode after reset");
        assert_eq!(out2, "OK");
    }

    #[test]
    fn invalid_lead_byte_0xff_is_an_error() {
        let mut state = Gb18030State::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0xFF], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
        assert_eq!(err.offset, 0);
    }

    #[test]
    fn lead_byte_with_ascii_trail_restores_the_trail_byte() {
        // 0x81 is a valid lead byte; 0x20 (space) is not a valid trail byte for a 2-byte
        // sequence, but *is* ASCII, so per spec it is restored.
        let mut state = Gb18030State::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0x81, 0x20], &mut out, true);
        assert_eq!(out, "\u{FFFD} ");
    }

    #[test]
    fn four_byte_restore_of_two_bytes_replays_through_the_state_machine() {
        // 0x81 0x30 0x81 0x41: first=0x81 (lead), second=0x30 (digit, starts a 4-byte
        // sequence), third=0x81 (valid lead-range byte for the third position) -- so far this
        // looks like a well-formed 4-byte prefix. The fourth byte, 0x41, is not a digit
        // (0x30..=0x39), so per spec this unconditionally restores the buffered second (0x30)
        // and third (0x81) bytes plus the current byte (0x41) -- regardless of the current
        // byte's ASCII-ness. Reprocessed fresh: 0x30 is itself ASCII ('0'), then 0x81 0x41 forms
        // a brand new 2-byte sequence.
        let mut state = Gb18030State::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0x81, 0x30, 0x81, 0x41], &mut out, true);
        let mut expected = String::from("\u{FFFD}0");
        // pointer for lead 0x81 / trail 0x41 is (0x81-0x81)*190 + (0x41-0x40) == 1.
        let reprocessed = gb18030_decode_2byte(1);
        expected.push(reprocessed.expect("0x81 0x41 must be a valid 2-byte sequence"));
        assert_eq!(out, expected);
    }

    #[test]
    fn four_byte_restore_of_two_bytes_replays_correctly_when_split_across_a_chunk_boundary() {
        // Same byte sequence and expected output as
        // `four_byte_restore_of_two_bytes_replays_through_the_state_machine`, but with `second`
        // and `third` (0x30, 0x81) arriving in one call and the failing fourth byte (0x41)
        // arriving in the next: the replay queue exists precisely because those two bytes are no
        // longer available in `input` by the time the restore is discovered, so this is the case
        // an `idx`-rewind trick (sufficient for every other decoder in this crate) cannot express.
        let mut state = Gb18030State::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0x81, 0x30, 0x81], &mut out, false);
        assert!(state.has_pending());
        decode_lossy(&mut state, &[0x41], &mut out, true);

        let mut expected = String::from("\u{FFFD}0");
        let reprocessed = gb18030_decode_2byte(1);
        expected.push(reprocessed.expect("0x81 0x41 must be a valid 2-byte sequence"));
        assert_eq!(out, expected);
    }

    #[test]
    fn four_byte_restore_reported_as_strict_error_at_the_original_lead_byte() {
        let mut state = Gb18030State::new();
        let mut out = String::new();
        let err =
            decode(&mut state, &[0x81, 0x30, 0x81, 0x41], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
        assert_eq!(err.offset, 0);
        assert_eq!(err.length, 1);
    }

    #[test]
    fn encode_ascii_and_euro_sign_differ_between_gbk_and_gb18030()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut gbk = Vec::new();
        encode("a\u{20AC}", &mut gbk, true)?;
        assert_eq!(gbk, vec![b'a', 0x80]);

        let mut gb18030 = Vec::new();
        encode("a\u{20AC}", &mut gb18030, false)?;
        assert_eq!(gb18030, vec![b'a', 0xA2, 0xE3]);
        assert_ne!(gbk, gb18030);

        // Both must decode back to the same text through this module's own decoder.
        assert_eq!(decode_all(&[&gbk])?, "a\u{20AC}");
        assert_eq!(decode_all(&[&gb18030])?, "a\u{20AC}");
        Ok(())
    }

    #[test]
    fn encode_well_known_2byte_kanji() -> Result<(), EncodeError> {
        let mut dst = Vec::new();
        encode("中", &mut dst, false)?;
        assert_eq!(dst, vec![0xD6, 0xD0]);
        Ok(())
    }

    #[test]
    fn encode_supplementary_plane_character_uses_four_bytes_only_for_gb18030() {
        let mut dst = Vec::new();
        encode("\u{20000}", &mut dst, false).expect("gb18030 must encode every scalar value");
        assert_eq!(dst, vec![0x95, 0x32, 0x82, 0x36]);

        let mut dst = Vec::new();
        let err = encode("\u{20000}", &mut dst, true).expect_err("GBK must reject it");
        assert_eq!(err.character, '\u{20000}');
    }

    #[test]
    fn encode_rejects_u_e5e5_under_both_gbk_and_gb18030() {
        let mut dst = Vec::new();
        let err = encode("\u{E5E5}", &mut dst, false).expect_err("must fail");
        assert_eq!(err.character, '\u{E5E5}');
        let mut dst = Vec::new();
        let err = encode("\u{E5E5}", &mut dst, true).expect_err("must fail");
        assert_eq!(err.character, '\u{E5E5}');
    }

    #[test]
    fn encode_pua_override_table_produces_the_documented_legacy_bytes()
    -> Result<(), Box<dyn std::error::Error>> {
        // These 18 code points deliberately do *not* round-trip through encode-then-decode (see
        // `encode`'s doc comment and `gb18030_is_total_over_every_unicode_scalar_value`, which
        // asserts that non-round-tripping property exhaustively); this test only pins the
        // encoder's documented legacy byte pair for each one.
        for &(codepoint, bytes) in PUA_ENCODE_OVERRIDE.iter() {
            let character = char::from_u32(codepoint).expect("valid scalar value");
            let mut dst = Vec::new();
            encode(&character.to_string(), &mut dst, false)?;
            assert_eq!(dst, bytes.to_vec(), "U+{codepoint:04X}");
        }
        Ok(())
    }

    #[test]
    fn round_trip_mixed_text_at_every_chunk_size() -> Result<(), Box<dyn std::error::Error>> {
        let text = "Hello, 世界! 中文测试123 \u{20000}\u{20AC}";
        let mut bytes = Vec::new();
        encode(text, &mut bytes, false)?;
        for chunk_size in 1..=5 {
            let mut state = Gb18030State::new();
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
        let mut state = Gb18030State::new();
        let mut out = String::new();
        let consumed = decode_lossy(&mut state, &all_bytes, &mut out, true);
        assert_eq!(consumed, 256);
    }

    #[test]
    fn lossy_decode_one_byte_at_a_time_never_panics() {
        let mut state = Gb18030State::new();
        let mut out = String::new();
        for b in 0u16..=255 {
            let last = b == 255;
            decode_lossy(&mut state, &[b as u8], &mut out, last);
        }
    }

    #[test]
    fn gb18030_is_total_over_every_unicode_scalar_value() {
        // gb18030's *encoder* is documented as total over Unicode except for U+E5E5, but that is
        // a claim about encoding succeeding, not about encode-then-decode round-tripping to the
        // same character: `PUA_ENCODE_OVERRIDE`'s 18 code points are intentionally encoded to a
        // fixed legacy byte pair that the decoder resolves back to a *different* code point (see
        // `encode`'s own doc comment) -- both properties are asserted here, exhaustively, over
        // every scalar value. Reuses one `String`/`Vec<u8>`/`Gb18030State` per iteration to keep
        // the sweep well within budget.
        let mut checked = 0usize;
        let mut text = String::new();
        let mut bytes = Vec::new();
        let mut out = String::new();
        for cp in 0u32..=0x10FFFF {
            let Some(c) = char::from_u32(cp) else {
                continue; // surrogate range
            };
            if c == '\u{E5E5}' {
                continue; // the one documented, spec-mandated encode failure
            }
            text.clear();
            text.push(c);
            bytes.clear();
            encode(&text, &mut bytes, false).unwrap_or_else(|e| panic!("U+{cp:04X}: {e}"));
            let mut state = Gb18030State::new();
            out.clear();
            decode(&mut state, &bytes, &mut out, true)
                .unwrap_or_else(|e| panic!("U+{cp:04X}: {e}"));
            if pua_encode_override(cp).is_some() {
                // Deliberately does not round-trip; see `encode`'s doc comment.
                assert_ne!(
                    out, text,
                    "U+{cp:04X} was expected to be a non-round-tripping PUA override entry"
                );
            } else {
                assert_eq!(out, text, "U+{cp:04X} round trip");
            }
            checked += 1;
        }
        // U+0000..=U+10FFFF is 1,114,112 code points; minus 2,048 surrogates minus 1 (U+E5E5).
        assert_eq!(checked, 1_114_112 - 2_048 - 1);
    }

    #[test]
    fn gb18030_range_boundaries_round_trip() {
        // The code points immediately surrounding every range's start are exactly where an
        // off-by-one transcription error would show up; the broad sweep above uses a stride that
        // will not reliably land on them.
        for &cp in &[
            0x0080u32, 0x00A4, 0x00A5, 0x00A8, 0x00A9, 0xE7C7, 0xFFFF, 0x10000, 0x10001, 0x10FFFE,
            0x10FFFF,
        ] {
            let c = char::from_u32(cp).expect("valid scalar value");
            let mut bytes = Vec::new();
            encode(&c.to_string(), &mut bytes, false)
                .unwrap_or_else(|e| panic!("U+{cp:04X} must encode: {e}"));
            let decoded =
                decode_all(&[&bytes]).unwrap_or_else(|e| panic!("U+{cp:04X} must decode: {e}"));
            assert_eq!(decoded, c.to_string(), "U+{cp:04X}");
        }
    }
}

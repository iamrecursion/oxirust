//! Incremental UTF-16 (LE/BE) decoding.
//!
//! Two kinds of state can straddle a chunk boundary and must be carried between calls:
//!
//! 1. A lone byte at the end of an odd-length chunk, still waiting for its partner byte to form
//!    a complete 16-bit code unit ([`Utf16State::pending_byte`]).
//! 2. A high surrogate code unit (`0xD800..=0xDBFF`) that was itself assembled from a complete
//!    pair of bytes, but whose partner low surrogate has not arrived yet
//!    ([`Utf16State::pending_high_surrogate`]).

use super::{DecodeError, DecodeErrorKind};

/// State carried between [`decode`]/[`decode_lossy`] calls.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Utf16State {
    pending_byte: Option<u8>,
    pending_high_surrogate: Option<u16>,
}

impl Utf16State {
    pub(crate) const fn new() -> Self {
        Utf16State {
            pending_byte: None,
            pending_high_surrogate: None,
        }
    }

    pub(crate) const fn has_pending(&self) -> bool {
        self.pending_byte.is_some() || self.pending_high_surrogate.is_some()
    }
}

#[inline]
const fn combine_u16(first: u8, second: u8, big_endian: bool) -> u16 {
    if big_endian {
        u16::from_be_bytes([first, second])
    } else {
        u16::from_le_bytes([first, second])
    }
}

fn is_high_surrogate(unit: u16) -> bool {
    (0xD800..=0xDBFF).contains(&unit)
}

fn is_low_surrogate(unit: u16) -> bool {
    (0xDC00..=0xDFFF).contains(&unit)
}

/// Combine a pending high surrogate with `low` into the astral scalar value it encodes.
#[inline]
fn surrogate_pair_to_scalar(high: u16, low: u16) -> u32 {
    0x10000 + (((high as u32) - 0xD800) << 10) + ((low as u32) - 0xDC00)
}

/// Strict incremental UTF-16 decode. See [`super::Decoder::decode_to_string`].
pub(crate) fn decode(
    state: &mut Utf16State,
    input: &[u8],
    dst: &mut String,
    last: bool,
    big_endian: bool,
) -> Result<usize, DecodeError> {
    let mut idx = 0usize;

    if let Some(first) = state.pending_byte.take() {
        match input.first() {
            None => {
                state.pending_byte = Some(first);
                if last {
                    state.pending_byte = None;
                    return Err(DecodeError {
                        offset: 0,
                        length: 1,
                        kind: DecodeErrorKind::TruncatedSequence,
                    });
                }
                return Ok(0);
            }
            Some(&second) => {
                idx = 1;
                let unit = combine_u16(first, second, big_endian);
                process_unit(state, unit, dst, 0)?;
            }
        }
    }

    while idx < input.len() {
        if idx + 1 >= input.len() {
            state.pending_byte = Some(input[idx]);
            break;
        }
        let unit = combine_u16(input[idx], input[idx + 1], big_endian);
        process_unit(state, unit, dst, idx)?;
        idx += 2;
    }

    if last {
        if state.pending_byte.take().is_some() {
            return Err(DecodeError {
                offset: input.len().saturating_sub(1),
                length: 1,
                kind: DecodeErrorKind::TruncatedSequence,
            });
        }
        if state.pending_high_surrogate.take().is_some() {
            return Err(DecodeError {
                offset: input.len(),
                length: 0,
                kind: DecodeErrorKind::UnpairedSurrogate,
            });
        }
    }

    Ok(input.len())
}

/// Consume one already-assembled code unit: complete a pending surrogate pair, start a new
/// pending high surrogate, reject a stray low surrogate, or emit a plain BMP scalar value.
fn process_unit(
    state: &mut Utf16State,
    unit: u16,
    dst: &mut String,
    offset: usize,
) -> Result<(), DecodeError> {
    if let Some(high) = state.pending_high_surrogate.take() {
        if is_low_surrogate(unit) {
            let scalar = surrogate_pair_to_scalar(high, unit);
            if let Some(c) = char::from_u32(scalar) {
                dst.push(c);
            }
            return Ok(());
        }
        return Err(DecodeError {
            offset,
            length: 2,
            kind: DecodeErrorKind::UnpairedSurrogate,
        });
    }
    if is_high_surrogate(unit) {
        state.pending_high_surrogate = Some(unit);
        return Ok(());
    }
    if is_low_surrogate(unit) {
        return Err(DecodeError {
            offset,
            length: 2,
            kind: DecodeErrorKind::UnpairedSurrogate,
        });
    }
    if let Some(c) = char::from_u32(unit as u32) {
        dst.push(c);
    }
    Ok(())
}

/// Lossy incremental UTF-16 decode: unpaired surrogates and dangling trailing bytes become
/// `U+FFFD`. See [`super::Decoder::decode_to_string_lossy`].
pub(crate) fn decode_lossy(
    state: &mut Utf16State,
    input: &[u8],
    dst: &mut String,
    last: bool,
    big_endian: bool,
) -> usize {
    let mut idx = 0usize;

    if let Some(first) = state.pending_byte.take() {
        match input.first() {
            None => {
                if last {
                    dst.push(char::REPLACEMENT_CHARACTER);
                } else {
                    state.pending_byte = Some(first);
                }
                return input.len();
            }
            Some(&second) => {
                idx = 1;
                let unit = combine_u16(first, second, big_endian);
                process_unit_lossy(state, unit, dst);
            }
        }
    }

    while idx < input.len() {
        if idx + 1 >= input.len() {
            state.pending_byte = Some(input[idx]);
            break;
        }
        let unit = combine_u16(input[idx], input[idx + 1], big_endian);
        process_unit_lossy(state, unit, dst);
        idx += 2;
    }

    if last {
        if state.pending_byte.take().is_some() {
            dst.push(char::REPLACEMENT_CHARACTER);
        }
        if state.pending_high_surrogate.take().is_some() {
            dst.push(char::REPLACEMENT_CHARACTER);
        }
    }

    input.len()
}

/// Lossy counterpart of [`process_unit`]. An unpaired high surrogate emits its own `U+FFFD`, and
/// the code unit that broke the pairing is then re-evaluated on its own (recursing at most once,
/// since `state.pending_high_surrogate` is `None` on the second call).
fn process_unit_lossy(state: &mut Utf16State, unit: u16, dst: &mut String) {
    if let Some(high) = state.pending_high_surrogate.take() {
        if is_low_surrogate(unit) {
            let scalar = surrogate_pair_to_scalar(high, unit);
            dst.push(char::from_u32(scalar).unwrap_or(char::REPLACEMENT_CHARACTER));
            return;
        }
        dst.push(char::REPLACEMENT_CHARACTER);
        process_unit_lossy(state, unit, dst);
        return;
    }
    if is_high_surrogate(unit) {
        state.pending_high_surrogate = Some(unit);
        return;
    }
    if is_low_surrogate(unit) {
        dst.push(char::REPLACEMENT_CHARACTER);
        return;
    }
    dst.push(char::from_u32(unit as u32).unwrap_or(char::REPLACEMENT_CHARACTER));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(chunks: &[&[u8]], big_endian: bool) -> Result<String, DecodeError> {
        let mut state = Utf16State::new();
        let mut out = String::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            decode(&mut state, chunk, &mut out, last, big_endian)?;
        }
        Ok(out)
    }

    fn le_bytes(text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        for unit in text.encode_utf16() {
            out.extend_from_slice(&unit.to_le_bytes());
        }
        out
    }

    fn be_bytes(text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        for unit in text.encode_utf16() {
            out.extend_from_slice(&unit.to_be_bytes());
        }
        out
    }

    #[test]
    fn bmp_only_le_and_be() -> Result<(), DecodeError> {
        let text = "hello→ネコ";
        assert_eq!(decode_all(&[&le_bytes(text)], false)?, text);
        assert_eq!(decode_all(&[&be_bytes(text)], true)?, text);
        Ok(())
    }

    #[test]
    fn astral_surrogate_pair_le_and_be() -> Result<(), DecodeError> {
        let text = "🎉party🎊";
        assert_eq!(decode_all(&[&le_bytes(text)], false)?, text);
        assert_eq!(decode_all(&[&be_bytes(text)], true)?, text);
        Ok(())
    }

    #[test]
    fn surrogate_pair_split_across_every_byte_boundary() -> Result<(), DecodeError> {
        let bytes = le_bytes("🎉");
        for split in 1..bytes.len() {
            let (a, b) = bytes.split_at(split);
            assert_eq!(decode_all(&[a, b], false)?, "🎉", "split at {split}");
        }
        Ok(())
    }

    #[test]
    fn one_two_three_bytes_at_a_time() -> Result<(), DecodeError> {
        let text = "The quick brown 🦊 fox→ネコ";
        for &big_endian in &[false, true] {
            let bytes = if big_endian {
                be_bytes(text)
            } else {
                le_bytes(text)
            };
            for chunk_size in [1usize, 2, 3] {
                let mut state = Utf16State::new();
                let mut out = String::new();
                let mut offset = 0;
                while offset < bytes.len() {
                    let end = (offset + chunk_size).min(bytes.len());
                    let last = end == bytes.len();
                    decode(&mut state, &bytes[offset..end], &mut out, last, big_endian)?;
                    offset = end;
                }
                assert_eq!(out, text, "big_endian={big_endian} chunk_size={chunk_size}");
            }
        }
        Ok(())
    }

    #[test]
    fn unpaired_high_surrogate_is_an_error() {
        let mut state = Utf16State::new();
        let mut out = String::new();
        // 0xD800 (high surrogate) followed by 'x' (0x0078), little-endian.
        let bytes = [0x00, 0xD8, 0x78, 0x00];
        let err = decode(&mut state, &bytes, &mut out, true, false).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::UnpairedSurrogate);
    }

    #[test]
    fn unpaired_low_surrogate_is_an_error() {
        let mut state = Utf16State::new();
        let mut out = String::new();
        // 0xDC00 (low surrogate) on its own, little-endian.
        let bytes = [0x00, 0xDC];
        let err = decode(&mut state, &bytes, &mut out, true, false).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::UnpairedSurrogate);
    }

    #[test]
    fn high_surrogate_at_true_end_of_stream_is_truncated() {
        let mut state = Utf16State::new();
        let mut out = String::new();
        let bytes = [0x00, 0xD8]; // high surrogate, little-endian, then nothing.
        let err = decode(&mut state, &bytes, &mut out, true, false).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::UnpairedSurrogate);
    }

    #[test]
    fn odd_trailing_byte_at_true_end_of_stream_is_truncated() {
        let mut state = Utf16State::new();
        let mut out = String::new();
        let bytes = [0x41, 0x00, 0x42]; // "A" then a dangling lead byte.
        let err = decode(&mut state, &bytes, &mut out, true, false).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
        assert_eq!(out, "A");
    }

    #[test]
    fn odd_trailing_byte_not_at_end_is_buffered() -> Result<(), DecodeError> {
        let mut state = Utf16State::new();
        let mut out = String::new();
        decode(&mut state, &[0x41, 0x00, 0x42], &mut out, false, false)?;
        assert!(state.has_pending());
        decode(&mut state, &[0x00], &mut out, true, false)?;
        assert_eq!(out, "AB");
        Ok(())
    }

    #[test]
    fn lossy_replaces_unpaired_surrogate_and_keeps_going() {
        let mut state = Utf16State::new();
        let mut out = String::new();
        // high surrogate, then 'x' (0x0078), little-endian.
        let bytes = [0x00, 0xD8, 0x78, 0x00];
        decode_lossy(&mut state, &bytes, &mut out, true, false);
        assert_eq!(out, "\u{FFFD}x");
    }

    #[test]
    fn lossy_replaces_dangling_trailing_byte() {
        let mut state = Utf16State::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0x41, 0x00, 0x42], &mut out, true, false);
        assert_eq!(out, "A\u{FFFD}");
    }
}

//! Incremental UTF-8 decoding.
//!
//! UTF-8 validity is delegated to [`core::str::from_utf8`] — the standard library's validator
//! is exactly as strict as the Unicode standard requires (it rejects overlong encodings,
//! surrogate code points, and out-of-range scalar values) and exposes exactly the information
//! an incremental decoder needs through [`core::str::Utf8Error`]: `valid_up_to()` (the length of
//! the longest valid prefix) and `error_len()` (`Some(n)` for a definite `n`-byte malformed
//! sequence, or `None` when the bytes after the valid prefix are a *possibly*-valid sequence
//! that was simply cut short by the end of the slice).
//!
//! Chunk-boundary handling: at most 3 bytes of an incomplete trailing sequence are ever carried
//! between calls (a valid UTF-8 lead byte asserts at most 4 total bytes, so at most 3 can be
//! missing).

use super::{DecodeError, DecodeErrorKind};

/// State carried between [`decode`]/[`decode_lossy`] calls: the tail of an incomplete sequence
/// left over from the previous chunk.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Utf8State {
    buf: [u8; 3],
    buf_len: u8,
}

impl Utf8State {
    pub(crate) const fn new() -> Self {
        Utf8State {
            buf: [0; 3],
            buf_len: 0,
        }
    }

    pub(crate) const fn has_pending(&self) -> bool {
        self.buf_len != 0
    }
}

/// Outcome of validating one buffer (either `input` directly, or `carry ++ input`).
enum Outcome<'a> {
    /// The whole buffer was valid; nothing left over.
    Done,
    /// A valid prefix was pushed to `dst`; `tail` is an incomplete sequence to carry forward.
    Carry(&'a [u8]),
    /// A malformed or (at `last = true`) truncated sequence was found.
    Error(DecodeError),
}

/// Validate `buf` and push whatever is decodable to `dst`. `carry_len` is the number of leading
/// bytes of `buf` that came from a previous call (not from the `input` the caller is currently
/// reporting offsets against), used to translate `core::str::from_utf8`'s buffer-relative
/// offsets into input-relative ones.
fn validate<'a>(buf: &'a [u8], dst: &mut String, last: bool, carry_len: usize) -> Outcome<'a> {
    match core::str::from_utf8(buf) {
        Ok(s) => {
            dst.push_str(s);
            Outcome::Done
        }
        Err(e) => {
            let valid_up_to = e.valid_up_to();
            if valid_up_to > 0 {
                // Guaranteed valid UTF-8 by the contract of `Utf8Error::valid_up_to`: it is
                // defined as the length of the longest prefix of `buf` that is valid UTF-8.
                if let Ok(s) = core::str::from_utf8(&buf[..valid_up_to]) {
                    dst.push_str(s);
                }
            }
            match e.error_len() {
                Some(bad_len) => Outcome::Error(DecodeError {
                    offset: valid_up_to.saturating_sub(carry_len),
                    length: bad_len,
                    kind: DecodeErrorKind::MalformedSequence,
                }),
                None => {
                    let tail = &buf[valid_up_to..];
                    if last {
                        Outcome::Error(DecodeError {
                            offset: valid_up_to.saturating_sub(carry_len),
                            length: tail.len(),
                            kind: DecodeErrorKind::TruncatedSequence,
                        })
                    } else {
                        Outcome::Carry(tail)
                    }
                }
            }
        }
    }
}

/// Combine any carried-over bytes with `input` into an owned buffer. Returns `None` (and leaves
/// `input` usable directly, avoiding an allocation) when there was nothing to carry.
fn combined_with_carry(state: &mut Utf8State, input: &[u8]) -> Option<Vec<u8>> {
    let carry_len = state.buf_len as usize;
    if carry_len == 0 {
        return None;
    }
    let mut combined = Vec::with_capacity(carry_len + input.len());
    combined.extend_from_slice(&state.buf[..carry_len]);
    combined.extend_from_slice(input);
    state.buf_len = 0;
    Some(combined)
}

fn store_carry(state: &mut Utf8State, tail: &[u8]) {
    state.buf[..tail.len()].copy_from_slice(tail);
    state.buf_len = tail.len() as u8;
}

/// Strict incremental UTF-8 decode. See [`super::Decoder::decode_to_string`].
pub(crate) fn decode(
    state: &mut Utf8State,
    input: &[u8],
    dst: &mut String,
    last: bool,
) -> Result<usize, DecodeError> {
    let carry_len = state.buf_len as usize;
    let owned;
    let (buf, effective_carry_len): (&[u8], usize) = match combined_with_carry(state, input) {
        Some(combined) => {
            owned = combined;
            (&owned, carry_len)
        }
        None => (input, 0),
    };
    let outcome = validate(buf, dst, last, effective_carry_len);
    match outcome {
        Outcome::Done => Ok(input.len()),
        Outcome::Carry(tail) => {
            store_carry(state, tail);
            Ok(input.len())
        }
        Outcome::Error(err) => Err(err),
    }
}

/// Lossy incremental UTF-8 decode: malformed subsequences become `U+FFFD`. See
/// [`super::Decoder::decode_to_string_lossy`].
pub(crate) fn decode_lossy(
    state: &mut Utf8State,
    input: &[u8],
    dst: &mut String,
    last: bool,
) -> usize {
    let owned;
    let mut rest: &[u8] = match combined_with_carry(state, input) {
        Some(combined) => {
            owned = combined;
            &owned
        }
        None => input,
    };
    // Offsets are irrelevant in lossy mode (nothing is ever reported to the caller), so unlike
    // `validate` above this loop does not bother translating buffer-relative positions back to
    // `input`-relative ones.
    loop {
        match core::str::from_utf8(rest) {
            Ok(s) => {
                dst.push_str(s);
                break;
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to > 0 {
                    if let Ok(s) = core::str::from_utf8(&rest[..valid_up_to]) {
                        dst.push_str(s);
                    }
                }
                match e.error_len() {
                    Some(bad_len) => {
                        dst.push(char::REPLACEMENT_CHARACTER);
                        rest = &rest[valid_up_to + bad_len..];
                    }
                    None => {
                        let tail = &rest[valid_up_to..];
                        if last {
                            dst.push(char::REPLACEMENT_CHARACTER);
                        } else {
                            store_carry(state, tail);
                        }
                        break;
                    }
                }
            }
        }
    }
    input.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(chunks: &[&[u8]]) -> Result<String, DecodeError> {
        let mut state = Utf8State::new();
        let mut out = String::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            decode(&mut state, chunk, &mut out, last)?;
        }
        Ok(out)
    }

    #[test]
    fn ascii_single_chunk() -> Result<(), DecodeError> {
        assert_eq!(decode_all(&[b"hello world"])?, "hello world");
        Ok(())
    }

    #[test]
    fn multibyte_split_across_chunks() -> Result<(), DecodeError> {
        // "→" = E2 86 92, split at every possible byte boundary.
        assert_eq!(decode_all(&[&[0xE2], &[0x86], &[0x92]])?, "→");
        assert_eq!(decode_all(&[&[0xE2, 0x86], &[0x92]])?, "→");
        assert_eq!(decode_all(&[&[0xE2], &[0x86, 0x92]])?, "→");
        Ok(())
    }

    #[test]
    fn astral_split_across_chunks() -> Result<(), DecodeError> {
        // "🎉" = F0 9F 8E 89
        let bytes: &[u8] = "🎉".as_bytes();
        for split in 1..bytes.len() {
            let (a, b) = bytes.split_at(split);
            assert_eq!(decode_all(&[a, b])?, "🎉", "split at {split}");
        }
        Ok(())
    }

    #[test]
    fn invalid_continuation_byte_reports_offset() {
        let mut state = Utf8State::new();
        let mut out = String::new();
        // 0xC2 requires one continuation byte; 0x20 (space) is not one.
        let err = decode(&mut state, &[b'a', 0xC2, 0x20], &mut out, true).expect_err("must fail");
        assert_eq!(out, "a");
        assert_eq!(err.offset, 1);
        assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
    }

    #[test]
    fn truncated_sequence_at_true_end_of_stream_is_an_error() {
        let mut state = Utf8State::new();
        let mut out = String::new();
        let err = decode(&mut state, &[0xE2, 0x86], &mut out, true).expect_err("must fail");
        assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
        assert_eq!(err.offset, 0);
        assert_eq!(err.length, 2);
    }

    #[test]
    fn truncated_sequence_not_at_end_of_stream_is_buffered() -> Result<(), DecodeError> {
        let mut state = Utf8State::new();
        let mut out = String::new();
        decode(&mut state, &[0xE2, 0x86], &mut out, false)?;
        assert!(state.has_pending());
        assert_eq!(out, "");
        decode(&mut state, &[0x92], &mut out, true)?;
        assert_eq!(out, "→");
        Ok(())
    }

    #[test]
    fn lossy_replaces_malformed_bytes() {
        let mut state = Utf8State::new();
        let mut out = String::new();
        let consumed = decode_lossy(&mut state, &[b'a', 0xFF, b'b'], &mut out, true);
        assert_eq!(consumed, 3);
        assert_eq!(out, "a\u{FFFD}b");
    }

    #[test]
    fn lossy_handles_truncated_end_of_stream() {
        let mut state = Utf8State::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0xE2, 0x86], &mut out, true);
        assert_eq!(out, "\u{FFFD}");
    }

    #[test]
    fn lossy_carries_incomplete_sequence_across_chunks() {
        let mut state = Utf8State::new();
        let mut out = String::new();
        decode_lossy(&mut state, &[0xE2, 0x86], &mut out, false);
        assert_eq!(out, "");
        decode_lossy(&mut state, &[0x92], &mut out, true);
        assert_eq!(out, "→");
    }

    #[test]
    fn one_byte_at_a_time_matches_whole_buffer_decode() -> Result<(), DecodeError> {
        let text = "The quick brown fox 🦊 jumps über the lazy dog→ネコ";
        let bytes = text.as_bytes();
        for chunk_size in [1usize, 2, 3] {
            let mut state = Utf8State::new();
            let mut out = String::new();
            let mut offset = 0;
            while offset < bytes.len() {
                let end = (offset + chunk_size).min(bytes.len());
                let last = end == bytes.len();
                decode(&mut state, &bytes[offset..end], &mut out, last)?;
                offset = end;
            }
            assert_eq!(out, text, "chunk_size = {chunk_size}");
        }
        Ok(())
    }
}

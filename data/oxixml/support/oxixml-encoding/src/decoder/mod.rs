//! Incremental, chunk-safe decoding from bytes to [`String`]s.

mod single_byte;
mod utf16;
mod utf8;

use crate::cjk::{big5, euc_kr, gb18030};
use crate::jis::{euc_jp, iso2022jp, shift_jis};
use crate::label::Encoding;

/// An error reported by [`Decoder::decode_to_string`].
///
/// All positional information (`offset`, `length`) is expressed in bytes and is relative to the
/// `input` slice passed to the specific [`decode_to_string`](Decoder::decode_to_string) call
/// that returned this error — *not* to the logical start of the overall byte stream. A caller
/// tracking a running total offset across chunks should add the sum of the lengths of all
/// previously-consumed chunks to `offset` to obtain a stream-relative position.
///
/// When the malformed data spans a chunk boundary (part of it was buffered internally from a
/// previous call), `offset` is clamped to `0`: the problem became visible only once the new
/// bytes arrived, so `0` is the most useful position a caller can act on (for example, to slice
/// `input[..0]` off before the error and report the rest as unconsumed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeError {
    /// Byte offset within the `input` slice of the call that failed.
    pub offset: usize,
    /// The number of bytes that make up the malformed/incomplete unit, where known. This is a
    /// hint for error-recovery strategies (e.g. "skip `length` bytes and retry"); it may be `0`
    /// when no specific byte range is at fault (e.g. an unpaired surrogate detected only once a
    /// *following* code unit failed to complete the pair).
    pub length: usize,
    /// What kind of problem was found.
    pub kind: DecodeErrorKind,
}

/// The kind of problem a [`DecodeError`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeErrorKind {
    /// A byte sequence that can never be valid in this encoding (e.g. a UTF-8 continuation byte
    /// in a position where a lead byte was expected, or an overlong encoding).
    MalformedSequence,
    /// The input ended in the middle of a multi-byte sequence and the caller signaled `last =
    /// true`, so no further bytes will arrive to complete it.
    TruncatedSequence,
    /// A UTF-16 high surrogate was not followed by a low surrogate (or vice versa).
    UnpairedSurrogate,
    /// A byte outside the encoding's repertoire was found (e.g. `0x80..=0xFF` in
    /// [`Encoding::Ascii`]).
    DisallowedByte,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self.kind {
            DecodeErrorKind::MalformedSequence => "malformed byte sequence",
            DecodeErrorKind::TruncatedSequence => "truncated byte sequence at end of input",
            DecodeErrorKind::UnpairedSurrogate => "unpaired UTF-16 surrogate",
            DecodeErrorKind::DisallowedByte => "byte outside the encoding's repertoire",
        };
        write!(
            f,
            "{what} at offset {} (length {})",
            self.offset, self.length
        )
    }
}

impl std::error::Error for DecodeError {}

/// An incremental decoder for one [`Encoding`].
///
/// A `Decoder` is created via [`Encoding::new_decoder`] and is fed the byte stream in one or
/// more calls to [`decode_to_string`](Decoder::decode_to_string) (or its lossy counterpart,
/// [`decode_to_string_lossy`](Decoder::decode_to_string_lossy)). Chunks may be split at any
/// byte boundary — including in the middle of a multi-byte sequence or a UTF-16 surrogate pair
/// — and the decoder carries the necessary state between calls to produce output identical to
/// decoding the concatenation of all chunks in one call.
///
/// The final chunk (which may be empty) must be passed with `last = true` so the decoder can
/// detect a sequence left truncated at the true end of the stream.
///
/// # Examples
///
/// Feeding a UTF-8 string one byte at a time still reassembles multi-byte characters correctly:
///
/// ```
/// use oxixml_encoding::Encoding;
///
/// let text = "a→b🎉c"; // mixes ASCII, a 3-byte BMP arrow, and a 4-byte astral emoji.
/// let mut decoder = Encoding::Utf8.new_decoder();
/// let mut out = String::new();
/// let bytes = text.as_bytes();
/// for i in 0..bytes.len() {
///     let last = i + 1 == bytes.len();
///     decoder.decode_to_string(&bytes[i..=i], &mut out, last).expect("valid utf-8");
/// }
/// assert_eq!(out, text);
/// ```
#[derive(Debug, Clone)]
pub struct Decoder {
    kind: DecoderKind,
}

#[derive(Debug, Clone)]
enum DecoderKind {
    Utf8(utf8::Utf8State),
    Utf16Le(utf16::Utf16State),
    Utf16Be(utf16::Utf16State),
    Latin1,
    Windows1252,
    Ascii,
    /// Any of the 26 legacy single-byte encodings backed by a generated 128-entry upper-half
    /// table (see [`Encoding::is_legacy_single_byte_table`]). Stateless, like the three variants
    /// above, so the specific [`Encoding`] is carried directly rather than via a per-encoding
    /// unit-struct variant.
    LegacySingleByte(Encoding),
    ShiftJis(shift_jis::ShiftJisState),
    EucJp(euc_jp::EucJpState),
    Iso2022Jp(iso2022jp::Iso2022JpState),
    /// GBK and gb18030 share this state and decode identically; see [`Encoding::Gbk`] and
    /// [`Encoding::Gb18030`] for how the two differ (only in their encoder).
    Gbk(gb18030::Gb18030State),
    Gb18030(gb18030::Gb18030State),
    Big5(big5::Big5State),
    EucKr(euc_kr::EucKrState),
}

impl Decoder {
    /// Create a new decoder for `encoding`. Prefer [`Encoding::new_decoder`], which forwards
    /// here.
    #[must_use]
    pub(crate) const fn new(encoding: Encoding) -> Decoder {
        let kind = match encoding {
            Encoding::Utf8 => DecoderKind::Utf8(utf8::Utf8State::new()),
            Encoding::Utf16Le => DecoderKind::Utf16Le(utf16::Utf16State::new()),
            Encoding::Utf16Be => DecoderKind::Utf16Be(utf16::Utf16State::new()),
            Encoding::Latin1 => DecoderKind::Latin1,
            Encoding::Windows1252 => DecoderKind::Windows1252,
            Encoding::Ascii => DecoderKind::Ascii,
            Encoding::ShiftJis => DecoderKind::ShiftJis(shift_jis::ShiftJisState::new()),
            Encoding::EucJp => DecoderKind::EucJp(euc_jp::EucJpState::new()),
            Encoding::Iso2022Jp => DecoderKind::Iso2022Jp(iso2022jp::Iso2022JpState::new()),
            Encoding::Gbk => DecoderKind::Gbk(gb18030::Gb18030State::new()),
            Encoding::Gb18030 => DecoderKind::Gb18030(gb18030::Gb18030State::new()),
            Encoding::Big5 => DecoderKind::Big5(big5::Big5State::new()),
            Encoding::EucKr => DecoderKind::EucKr(euc_kr::EucKrState::new()),
            // Every remaining variant is one of the 26 legacy single-byte encodings; verified in
            // debug/test builds (this function must stay `const`, so a full `match` guard on
            // `is_legacy_single_byte_table` is not usable here without also enumerating all 26
            // variants a second time). `crate::tables::legacy_single_byte_table` enumerates
            // every `Encoding` variant explicitly rather than through a wildcard, so a future
            // variant left unhandled everywhere is still caught at compile time there, even if
            // this debug-only assertion is never exercised by a test.
            other => {
                debug_assert!(
                    other.is_legacy_single_byte_table(),
                    "Decoder::new: unhandled Encoding variant"
                );
                DecoderKind::LegacySingleByte(other)
            }
        };
        Decoder { kind }
    }

    /// The encoding this decoder was created for.
    #[must_use]
    pub const fn encoding(&self) -> Encoding {
        match &self.kind {
            DecoderKind::Utf8(_) => Encoding::Utf8,
            DecoderKind::Utf16Le(_) => Encoding::Utf16Le,
            DecoderKind::Utf16Be(_) => Encoding::Utf16Be,
            DecoderKind::Latin1 => Encoding::Latin1,
            DecoderKind::Windows1252 => Encoding::Windows1252,
            DecoderKind::Ascii => Encoding::Ascii,
            DecoderKind::LegacySingleByte(encoding) => *encoding,
            DecoderKind::ShiftJis(_) => Encoding::ShiftJis,
            DecoderKind::EucJp(_) => Encoding::EucJp,
            DecoderKind::Iso2022Jp(_) => Encoding::Iso2022Jp,
            DecoderKind::Gbk(_) => Encoding::Gbk,
            DecoderKind::Gb18030(_) => Encoding::Gb18030,
            DecoderKind::Big5(_) => Encoding::Big5,
            DecoderKind::EucKr(_) => Encoding::EucKr,
        }
    }

    /// Whether this decoder is currently holding onto part of an incomplete multi-byte sequence
    /// (or an unpaired UTF-16 high surrogate) carried over from a previous
    /// [`decode_to_string`](Decoder::decode_to_string) call.
    ///
    /// Useful for a caller that wants to detect a stream that ends mid-sequence *without*
    /// making a final `last = true` call (for example, when a network connection drops).
    #[must_use]
    pub const fn has_pending_state(&self) -> bool {
        match &self.kind {
            DecoderKind::Utf8(state) => state.has_pending(),
            DecoderKind::Utf16Le(state) | DecoderKind::Utf16Be(state) => state.has_pending(),
            DecoderKind::Latin1
            | DecoderKind::Windows1252
            | DecoderKind::Ascii
            | DecoderKind::LegacySingleByte(_) => false,
            DecoderKind::ShiftJis(state) => state.has_pending(),
            DecoderKind::EucJp(state) => state.has_pending(),
            DecoderKind::Iso2022Jp(state) => state.has_pending(),
            DecoderKind::Gbk(state) | DecoderKind::Gb18030(state) => state.has_pending(),
            DecoderKind::Big5(state) => state.has_pending(),
            DecoderKind::EucKr(state) => state.has_pending(),
        }
    }

    /// Decode as much of `input` as possible, appending the result to `dst`.
    ///
    /// `input` need not end on a character boundary: an incomplete trailing sequence is saved
    /// internally and completed by the start of the next call. Pass `last = true` for the final
    /// chunk of the stream (an empty final chunk is fine) so that a sequence left dangling at
    /// true end-of-input is reported as [`DecodeErrorKind::TruncatedSequence`] rather than
    /// silently held onto forever.
    ///
    /// Returns the number of bytes of `input` incorporated into the decode (either emitted as
    /// characters or buffered as a pending partial sequence). On success this is always
    /// `input.len()`. On error, see [`DecodeError`] for how `offset`/`length` are interpreted;
    /// any valid characters that appeared before the error have already been appended to `dst`,
    /// and the decoder's carried state is reset to empty so the `Decoder` remains usable for a
    /// fresh call (e.g. after the caller skips the offending bytes).
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if `input` (combined with any carried-over state) contains a
    /// byte sequence that is not valid in this decoder's encoding, or ends in a sequence that
    /// `last = true` asserts will never be completed.
    pub fn decode_to_string(
        &mut self,
        input: &[u8],
        dst: &mut String,
        last: bool,
    ) -> Result<usize, DecodeError> {
        match &mut self.kind {
            DecoderKind::Utf8(state) => utf8::decode(state, input, dst, last),
            DecoderKind::Utf16Le(state) => utf16::decode(state, input, dst, last, false),
            DecoderKind::Utf16Be(state) => utf16::decode(state, input, dst, last, true),
            DecoderKind::Latin1 => single_byte::decode_latin1(input, dst),
            DecoderKind::Windows1252 => single_byte::decode_windows1252(input, dst),
            DecoderKind::Ascii => single_byte::decode_ascii(input, dst),
            DecoderKind::LegacySingleByte(encoding) => {
                single_byte::decode_legacy_single_byte(*encoding, input, dst)
            }
            DecoderKind::ShiftJis(state) => shift_jis::decode(state, input, dst, last),
            DecoderKind::EucJp(state) => euc_jp::decode(state, input, dst, last),
            DecoderKind::Iso2022Jp(state) => iso2022jp::decode(state, input, dst, last),
            DecoderKind::Gbk(state) | DecoderKind::Gb18030(state) => {
                gb18030::decode(state, input, dst, last)
            }
            DecoderKind::Big5(state) => big5::decode(state, input, dst, last),
            DecoderKind::EucKr(state) => euc_kr::decode(state, input, dst, last),
        }
    }

    /// Like [`decode_to_string`](Decoder::decode_to_string), but never fails: any malformed or
    /// disallowed byte sequence is replaced by one `U+FFFD REPLACEMENT CHARACTER` per the
    /// [Unicode "maximal subpart" recovery algorithm][u5-22], and decoding continues from the
    /// byte immediately after the replaced sequence.
    ///
    /// Always consumes (and returns) the full length of `input`.
    ///
    /// [u5-22]: https://www.unicode.org/versions/Unicode15.0.0/ch03.pdf#page=22
    pub fn decode_to_string_lossy(&mut self, input: &[u8], dst: &mut String, last: bool) -> usize {
        match &mut self.kind {
            DecoderKind::Utf8(state) => utf8::decode_lossy(state, input, dst, last),
            DecoderKind::Utf16Le(state) => utf16::decode_lossy(state, input, dst, last, false),
            DecoderKind::Utf16Be(state) => utf16::decode_lossy(state, input, dst, last, true),
            DecoderKind::Latin1 => single_byte::decode_latin1_lossy(input, dst),
            DecoderKind::Windows1252 => single_byte::decode_windows1252_lossy(input, dst),
            DecoderKind::Ascii => single_byte::decode_ascii_lossy(input, dst),
            DecoderKind::LegacySingleByte(encoding) => {
                single_byte::decode_legacy_single_byte_lossy(*encoding, input, dst)
            }
            DecoderKind::ShiftJis(state) => shift_jis::decode_lossy(state, input, dst, last),
            DecoderKind::EucJp(state) => euc_jp::decode_lossy(state, input, dst, last),
            DecoderKind::Iso2022Jp(state) => iso2022jp::decode_lossy(state, input, dst, last),
            DecoderKind::Gbk(state) | DecoderKind::Gb18030(state) => {
                gb18030::decode_lossy(state, input, dst, last)
            }
            DecoderKind::Big5(state) => big5::decode_lossy(state, input, dst, last),
            DecoderKind::EucKr(state) => euc_kr::decode_lossy(state, input, dst, last),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_round_trips_for_every_variant() {
        for &enc in Encoding::ALL {
            assert_eq!(enc.new_decoder().encoding(), enc);
        }
    }

    #[test]
    fn fresh_decoder_has_no_pending_state() {
        for &enc in Encoding::ALL {
            assert!(!enc.new_decoder().has_pending_state(), "{enc}");
        }
    }

    #[test]
    fn shift_jis_euc_jp_iso2022jp_decoders_report_pending_state_mid_sequence()
    -> Result<(), DecodeError> {
        let mut decoder = Encoding::ShiftJis.new_decoder();
        let mut out = String::new();
        decoder.decode_to_string(&[0x82], &mut out, false)?;
        assert!(decoder.has_pending_state());
        decoder.decode_to_string(&[0xA0], &mut out, true)?;
        assert!(!decoder.has_pending_state());

        let mut decoder = Encoding::EucJp.new_decoder();
        let mut out = String::new();
        decoder.decode_to_string(&[0xA4], &mut out, false)?;
        assert!(decoder.has_pending_state());
        decoder.decode_to_string(&[0xA2], &mut out, true)?;
        assert!(!decoder.has_pending_state());

        let mut decoder = Encoding::Iso2022Jp.new_decoder();
        let mut out = String::new();
        decoder.decode_to_string(&[0x1B], &mut out, false)?;
        assert!(decoder.has_pending_state());
        decoder.decode_to_string(&[0x28, 0x42], &mut out, true)?;
        assert!(!decoder.has_pending_state());
        Ok(())
    }

    #[test]
    fn cjk_decoders_report_pending_state_mid_sequence() -> Result<(), DecodeError> {
        for (encoding, lead, trail) in [
            (Encoding::Gbk, 0xD6u8, 0xD0u8),
            (Encoding::Gb18030, 0xD6, 0xD0),
            (Encoding::Big5, 0xA4, 0xA4),
            (Encoding::EucKr, 0xC7, 0xD1),
        ] {
            let mut decoder = encoding.new_decoder();
            let mut out = String::new();
            decoder.decode_to_string(&[lead], &mut out, false)?;
            assert!(decoder.has_pending_state(), "{encoding}");
            decoder.decode_to_string(&[trail], &mut out, true)?;
            assert!(!decoder.has_pending_state(), "{encoding}");
        }
        Ok(())
    }

    #[test]
    fn legacy_single_byte_decoders_never_report_pending_state() {
        for &enc in Encoding::ALL {
            if !enc.is_legacy_single_byte_table() {
                continue;
            }
            let mut decoder = enc.new_decoder();
            let mut out = String::new();
            let _ = decoder.decode_to_string(&[0x80], &mut out, false);
            assert!(!decoder.has_pending_state(), "{enc}");
        }
    }

    #[test]
    fn utf8_decoder_reports_pending_state_mid_sequence() -> Result<(), DecodeError> {
        let mut decoder = Encoding::Utf8.new_decoder();
        let mut out = String::new();
        // 0xE2 0x86 0x92 is '→'; feed only the first byte.
        decoder.decode_to_string(&[0xE2], &mut out, false)?;
        assert!(decoder.has_pending_state());
        decoder.decode_to_string(&[0x86, 0x92], &mut out, true)?;
        assert!(!decoder.has_pending_state());
        assert_eq!(out, "→");
        Ok(())
    }

    #[test]
    fn decode_error_display_is_human_readable() {
        let err = DecodeError {
            offset: 3,
            length: 1,
            kind: DecodeErrorKind::DisallowedByte,
        };
        assert_eq!(
            err.to_string(),
            "byte outside the encoding's repertoire at offset 3 (length 1)"
        );
    }
}

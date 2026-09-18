//! Wrappers that decode byte data into strings.
//!
//! The [`Decoder`] plays the same role as the decoder in quick-xml 0.41: without
//! the `encoding` feature it is always a UTF-8 decoder; with the `encoding`
//! feature it carries the document encoding inferred from the XML declaration and
#![cfg_attr(
    feature = "encoding",
    doc = "is backed by the Pure-Rust [`oxixml-encoding`](oxixml_encoding) crate instead of \
`encoding_rs`."
)]
#![cfg_attr(
    not(feature = "encoding"),
    doc = "is backed by the Pure-Rust `oxixml-encoding` crate instead of `encoding_rs`."
)]

use std::borrow::Cow;
use std::str::Utf8Error;

#[cfg(feature = "encoding")]
use oxixml_encoding::Encoding;

/// Unicode "byte order mark" (`U+FEFF`) encoded as UTF-8.
pub(crate) const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];
/// Unicode "byte order mark" (`U+FEFF`) encoded as UTF-16 little-endian.
pub(crate) const UTF16_LE_BOM: &[u8] = &[0xFF, 0xFE];
/// Unicode "byte order mark" (`U+FEFF`) encoded as UTF-16 big-endian.
pub(crate) const UTF16_BE_BOM: &[u8] = &[0xFE, 0xFF];

/// An error raised while decoding (or encoding) byte data.
///
/// If the `encoding` feature is disabled, the only possible value is
/// [`EncodingError::Utf8`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodingError {
    /// Input was not valid UTF-8.
    Utf8(Utf8Error),
    /// Input did not adhere to the given encoding.
    #[cfg(feature = "encoding")]
    #[cfg_attr(docsrs, doc(cfg(feature = "encoding")))]
    Other(Encoding),
}

impl From<Utf8Error> for EncodingError {
    #[inline]
    fn from(e: Utf8Error) -> Self {
        Self::Utf8(e)
    }
}

impl std::error::Error for EncodingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Utf8(e) => Some(e),
            #[cfg(feature = "encoding")]
            Self::Other(_) => None,
        }
    }
}

impl std::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Utf8(e) => write!(f, "cannot decode input using UTF-8: {}", e),
            #[cfg(feature = "encoding")]
            Self::Other(encoding) => write!(f, "cannot decode input using {}", encoding.name()),
        }
    }
}

/// Decoder of byte slices into strings.
///
/// If the `encoding` feature is enabled, the encoding is taken from the
/// `encoding="..."` pseudo-attribute of the XML declaration, or assumed to be
/// UTF-8 if there is no declaration or the encoding is unknown.
///
/// If the `encoding` feature is disabled, the decoder is always a UTF-8 decoder
/// and any XML declaration is ignored.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Decoder {
    #[cfg(feature = "encoding")]
    pub(crate) encoding: Encoding,
}

impl Decoder {
    /// A UTF-8 decoder. This is the decoder used for manually-constructed events.
    pub(crate) const fn utf8() -> Self {
        Decoder {
            #[cfg(feature = "encoding")]
            encoding: Encoding::Utf8,
        }
    }

    /// A UTF-16 little-endian decoder, used only by the deserializer unit tests
    /// to check that decoding a non-UTF-8 source produces owned data.
    #[cfg(all(test, feature = "encoding", feature = "serialize"))]
    pub(crate) const fn utf16() -> Self {
        Decoder {
            encoding: Encoding::Utf16Le,
        }
    }

    /// Returns the encoding used by this decoder.
    ///
    /// This encoding is used by [`decode`](Self::decode).
    #[cfg(feature = "encoding")]
    #[cfg_attr(docsrs, doc(cfg(feature = "encoding")))]
    pub const fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Decodes the given bytes.
    ///
    /// Without the `encoding` feature this decodes a UTF-8 slice regardless of any
    /// XML declaration (and ignoring any byte-order mark). With the `encoding`
    /// feature it decodes using the encoding declared in the XML, or UTF-8
    /// otherwise.
    ///
    /// Returns an error in case of malformed sequences in `bytes`.
    pub fn decode<'b>(&self, bytes: &'b [u8]) -> Result<Cow<'b, str>, EncodingError> {
        #[cfg(not(feature = "encoding"))]
        {
            Ok(Cow::Borrowed(std::str::from_utf8(bytes)?))
        }

        #[cfg(feature = "encoding")]
        {
            decode(bytes, self.encoding)
        }
    }

    /// Like [`decode`](Self::decode) but writes into a pre-allocated buffer.
    pub fn decode_into(&self, bytes: &[u8], buf: &mut String) -> Result<(), EncodingError> {
        #[cfg(not(feature = "encoding"))]
        {
            buf.push_str(std::str::from_utf8(bytes)?);
        }

        #[cfg(feature = "encoding")]
        {
            decode_into(bytes, self.encoding, buf)?;
        }

        Ok(())
    }

    /// Decodes the `Cow` buffer, preserving the lifetime where possible.
    pub(crate) fn decode_cow<'b>(
        &self,
        bytes: &Cow<'b, [u8]>,
    ) -> Result<Cow<'b, str>, EncodingError> {
        match bytes {
            Cow::Borrowed(bytes) => self.decode(bytes),
            // Convert to owned, because otherwise the `Cow` would be bound with the
            // wrong lifetime.
            Cow::Owned(bytes) => Ok(self.decode(bytes)?.into_owned().into()),
        }
    }

    /// Decodes the `Cow` buffer and normalizes XML end-of-line sequences,
    /// preserving the lifetime where possible.
    pub(crate) fn content<'b>(
        &self,
        bytes: &Cow<'b, [u8]>,
        normalize_eol: impl Fn(&str) -> Cow<str>,
    ) -> Result<Cow<'b, str>, EncodingError> {
        match bytes {
            Cow::Borrowed(bytes) => {
                let text = self.decode(bytes)?;
                match normalize_eol(&text) {
                    // If the text is still borrowed after normalization, it is unchanged.
                    Cow::Borrowed(_) => Ok(text),
                    Cow::Owned(s) => Ok(Cow::Owned(s)),
                }
            }
            Cow::Owned(bytes) => {
                let text = self.decode(bytes)?;
                let text = normalize_eol(&text);
                Ok(text.into_owned().into())
            }
        }
    }
}

/// Decodes the provided bytes using the specified encoding.
///
/// Returns an error for malformed or non-representable sequences in `bytes`.
#[cfg(feature = "encoding")]
#[cfg_attr(docsrs, doc(cfg(feature = "encoding")))]
pub fn decode(bytes: &[u8], encoding: Encoding) -> Result<Cow<'_, str>, EncodingError> {
    if matches!(encoding, Encoding::Utf8) {
        // Like quick-xml 0.41, which routes UTF-8 through the same fallible
        // decoding path as every other encoding and therefore reports
        // `EncodingError::Other(UTF-8)` — not `EncodingError::Utf8(..)` — for
        // malformed input here. `decode_into` keeps the `Utf8` variant, again
        // like quick-xml.
        return std::str::from_utf8(bytes)
            .map(Cow::Borrowed)
            .map_err(|_| EncodingError::Other(Encoding::Utf8));
    }
    let mut out = String::new();
    let mut decoder = encoding.new_decoder();
    decoder
        .decode_to_string(bytes, &mut out, true)
        .map_err(|_| EncodingError::Other(encoding))?;
    Ok(Cow::Owned(out))
}

/// Like [`decode`] but writes into a pre-allocated buffer.
#[cfg(feature = "encoding")]
#[cfg_attr(docsrs, doc(cfg(feature = "encoding")))]
pub fn decode_into(
    bytes: &[u8],
    encoding: Encoding,
    buf: &mut String,
) -> Result<(), EncodingError> {
    if matches!(encoding, Encoding::Utf8) {
        buf.push_str(std::str::from_utf8(bytes)?);
        return Ok(());
    }
    let mut decoder = encoding.new_decoder();
    decoder
        .decode_to_string(bytes, buf, true)
        .map_err(|_| EncodingError::Other(encoding))?;
    Ok(())
}

/// Automatic detection of the XML encoding using the
/// [recommended algorithm](https://www.w3.org/TR/xml11/#sec-guessing).
///
/// Returns `Some` with a [`DetectedEncoding`] (which reports the BOM length, if
/// any) when the encoding is recognized, otherwise `None`.
///
/// The algorithm examines up to the first four bytes to determine the encoding.
pub fn detect_encoding(bytes: &[u8]) -> Option<DetectedEncoding> {
    match bytes {
        // with BOM
        _ if bytes.starts_with(UTF16_BE_BOM) => Some(DetectedEncoding::Utf16BeBom),
        _ if bytes.starts_with(UTF16_LE_BOM) => Some(DetectedEncoding::Utf16LeBom),
        _ if bytes.starts_with(UTF8_BOM) => Some(DetectedEncoding::Utf8Bom),

        // without BOM
        _ if bytes.starts_with(&[0x00, b'<', 0x00, b'?']) => Some(DetectedEncoding::Utf16BeLike),
        _ if bytes.starts_with(&[b'<', 0x00, b'?', 0x00]) => Some(DetectedEncoding::Utf16LeLike),
        _ if bytes.starts_with(b"<?xm") => Some(DetectedEncoding::AsciiCompatible),

        _ => None,
    }
}

/// Possible outcomes of start-of-document encoding detection.
///
/// See [`detect_encoding`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectedEncoding {
    /// Matches UTF-8 or some other ASCII-compatible encoding.
    AsciiCompatible,
    /// A UTF-8 BOM was seen.
    Utf8Bom,
    /// Matches UTF-16 LE (or a compatible 16-bit little-endian encoding).
    Utf16LeLike,
    /// A UTF-16 little-endian BOM was seen.
    Utf16LeBom,
    /// Matches UTF-16 BE (or a compatible 16-bit big-endian encoding).
    Utf16BeLike,
    /// A UTF-16 big-endian BOM was seen.
    Utf16BeBom,
}

impl DetectedEncoding {
    /// Returns the [`Encoding`] appropriate for the detected variant.
    #[cfg(feature = "encoding")]
    #[cfg_attr(docsrs, doc(cfg(feature = "encoding")))]
    pub const fn encoding(&self) -> Encoding {
        match self {
            DetectedEncoding::AsciiCompatible | DetectedEncoding::Utf8Bom => Encoding::Utf8,
            DetectedEncoding::Utf16LeLike | DetectedEncoding::Utf16LeBom => Encoding::Utf16Le,
            DetectedEncoding::Utf16BeLike | DetectedEncoding::Utf16BeBom => Encoding::Utf16Be,
        }
    }

    /// Length in bytes of the BOM that may need to be stripped from the input.
    pub const fn bom_len(&self) -> usize {
        match self {
            DetectedEncoding::Utf8Bom => 3,
            DetectedEncoding::Utf16LeBom | DetectedEncoding::Utf16BeBom => 2,
            DetectedEncoding::AsciiCompatible
            | DetectedEncoding::Utf16LeLike
            | DetectedEncoding::Utf16BeLike => 0,
        }
    }
}

/// Bytes read upfront so that [`DecodingReader::set_encoding`] can be called
/// before the main decode loop starts. Kept small — just enough for an XML
/// declaration — to limit the number of bytes decoded with a possibly wrong
/// initial encoding.
#[cfg(feature = "encoding")]
const PREFIX_CAP: usize = 64;

/// A reader wrapper that decodes a byte stream from any supported encoding into
/// UTF-8.
///
/// This mirrors `quick_xml::encoding::DecodingReader` of quick-xml 0.41, but is
/// backed by the Pure-Rust [`oxixml-encoding`](oxixml_encoding) crate instead of
/// `encoding_rs`.
///
/// The [`Reader`](crate::reader::Reader) scans for ASCII delimiters in the raw
/// byte stream and therefore only understands ASCII-compatible encodings. Wrap
/// the input in this reader to parse a document in an encoding that is not
/// ASCII-compatible, such as UTF-16.
///
/// On the first access the encoding is detected from the byte-order mark or from
/// the byte pattern of the XML declaration (see [`detect_encoding`]); it defaults
/// to UTF-8. Once the declaration has been parsed, the encoding it names can be
/// installed with [`set_encoding`](Self::set_encoding).
///
/// # Note on the declared encoding
///
/// What this reader hands to the parser is already UTF-8, but the `encoding`
/// pseudo-attribute of the transcoded XML declaration still says what the
/// *original* bytes were. With the `encoding` feature enabled the
/// [`Reader`](crate::reader::Reader) picks that name up for its own
/// [`Decoder`], exactly as quick-xml 0.41 does, and would then decode the
/// event payloads a second time. Documents encoded in UTF-16 normally rely on
/// the byte-order mark and carry no `encoding` pseudo-attribute, in which case
/// nothing special is needed; if the declaration does name the original
/// encoding, either build the reader without the `encoding` feature — the
/// payloads are UTF-8 by then anyway — or take the payload bytes of the events
/// as-is instead of running them through the reader's [`Decoder`] again.
///
/// # Examples
///
/// UTF-8 input passes through unchanged:
///
/// ```
/// use std::io::Read;
/// use oxixml_xml::encoding::DecodingReader;
///
/// let data = b"Hello, World!";
/// let mut reader = DecodingReader::new(&data[..]);
/// let mut buf = Vec::new();
/// reader.read_to_end(&mut buf).expect("read");
/// assert_eq!(buf, data);
/// ```
///
/// A UTF-16 document becomes readable by the [`Reader`](crate::reader::Reader):
///
/// ```
/// use oxixml_xml::encoding::DecodingReader;
/// use oxixml_xml::events::Event;
/// use oxixml_xml::reader::Reader;
///
/// let mut xml = vec![0xFF, 0xFE]; // UTF-16 LE BOM
/// for unit in "<?xml version='1.0' encoding='UTF-16'?><element/>".encode_utf16() {
///     xml.extend_from_slice(&unit.to_le_bytes());
/// }
/// let mut reader = Reader::from_reader(DecodingReader::new(xml.as_slice()));
///
/// let mut buf = Vec::new();
/// assert!(matches!(reader.read_event_into(&mut buf).expect("decl"), Event::Decl(_)));
/// buf.clear();
/// assert!(matches!(reader.read_event_into(&mut buf).expect("element"), Event::Empty(_)));
/// ```
#[cfg(feature = "encoding")]
#[cfg_attr(docsrs, doc(cfg(feature = "encoding")))]
pub struct DecodingReader<R> {
    /// The wrapped source of encoded bytes.
    inner: R,
    /// Incremental decoder for the current encoding.
    decoder: oxixml_encoding::Decoder,
    /// Whether the decoder has been told that the input is over. Feeding it more
    /// input afterwards is not allowed.
    finished: bool,
    /// Decoded UTF-8 output that has not been handed out yet.
    out: String,
    /// Start of the unconsumed part of [`Self::out`], in bytes.
    out_pos: usize,
    /// Bytes read upfront for encoding detection. `Some` until they are drained.
    prefix: Option<Box<Prefix>>,
    /// Whether the inner reader has reached the end of its input.
    inner_eof: bool,
}

/// The upfront-read prefix of a [`DecodingReader`].
#[cfg(feature = "encoding")]
struct Prefix {
    /// Bytes read upfront, still encoded.
    buf: [u8; PREFIX_CAP],
    /// Number of valid bytes in [`Self::buf`].
    len: usize,
    /// Whether encoding detection has already run.
    detected: bool,
}

#[cfg(feature = "encoding")]
impl<R: std::fmt::Debug> std::fmt::Debug for DecodingReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodingReader")
            .field("inner", &self.inner)
            .field("encoding", &self.decoder.encoding())
            .field("inner_eof", &self.inner_eof)
            .field("prefix_active", &self.prefix.is_some())
            .finish()
    }
}

#[cfg(feature = "encoding")]
impl<R> DecodingReader<R> {
    /// Creates a new decoding reader around `inner`.
    ///
    /// The encoding is auto-detected from the byte-order mark or from the XML
    /// declaration pattern on the first access, defaulting to UTF-8.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            decoder: Encoding::Utf8.new_decoder(),
            finished: false,
            out: String::new(),
            out_pos: 0,
            prefix: Some(Box::new(Prefix {
                buf: [0; PREFIX_CAP],
                len: 0,
                detected: false,
            })),
            inner_eof: false,
        }
    }

    /// Returns a reference to the underlying reader.
    pub const fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Returns a mutable reference to the underlying reader.
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Consumes this reader and returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Returns the encoding the decoder currently uses.
    ///
    /// Before the first read this is always UTF-8; afterwards it reflects the
    /// detected — or explicitly installed — encoding.
    pub const fn encoding(&self) -> Encoding {
        self.decoder.encoding()
    }

    /// Replaces the decoder with one for `encoding`, keeping the position in the
    /// stream.
    ///
    /// Call this right after the XML declaration has been parsed, that is while
    /// the upfront-read prefix is still being served. Once the prefix is drained
    /// the encoding can no longer be changed and the call is ignored, because
    /// the bytes that follow have already been fed to the current decoder.
    ///
    /// Returns whether the encoding was installed. Setting the encoding that is
    /// already in use always succeeds and changes nothing (replacing the decoder
    /// would discard the partial sequence it may hold).
    pub fn set_encoding(&mut self, encoding: Encoding) -> bool {
        if self.decoder.encoding() == encoding {
            return true;
        }
        if self.prefix.is_none() {
            return false;
        }
        self.decoder = encoding.new_decoder();
        self.finished = false;
        true
    }

    /// Decodes `bytes` into the output buffer, mapping a decoding failure to an
    /// [`io::Error`](std::io::Error) that carries the [`EncodingError`].
    fn decode_chunk(&mut self, bytes: &[u8], last: bool) -> std::io::Result<()> {
        let encoding = self.decoder.encoding();
        match self.decoder.decode_to_string(bytes, &mut self.out, last) {
            Ok(_) => Ok(()),
            Err(_) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                EncodingError::Other(encoding),
            )),
        }
    }
}

#[cfg(feature = "encoding")]
impl<R: std::io::BufRead> std::io::BufRead for DecodingReader<R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        // Fast path: serve what has already been decoded.
        if self.out_pos < self.out.len() {
            return Ok(&self.out.as_bytes()[self.out_pos..]);
        }
        self.out.clear();
        self.out_pos = 0;

        if let Some(mut prefix) = self.prefix.take() {
            if !prefix.detected {
                prefix.detected = true;
                while prefix.len < PREFIX_CAP {
                    match std::io::Read::read(&mut self.inner, &mut prefix.buf[prefix.len..]) {
                        Ok(0) => {
                            self.inner_eof = true;
                            break;
                        }
                        Ok(read) => prefix.len += read,
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(error) => {
                            self.prefix = Some(prefix);
                            return Err(error);
                        }
                    }
                }
                if let Some(detected) = detect_encoding(&prefix.buf[..prefix.len]) {
                    let bom_len = detected.bom_len();
                    if bom_len > 0 {
                        prefix.buf.copy_within(bom_len..prefix.len, 0);
                        prefix.len -= bom_len;
                    }
                    let encoding = detected.encoding();
                    if encoding != Encoding::Utf8 {
                        self.decoder = encoding.new_decoder();
                    }
                }
            }

            if prefix.len > 0 {
                let len = prefix.len;
                let mut bytes = [0; PREFIX_CAP];
                bytes[..len].copy_from_slice(&prefix.buf[..len]);
                prefix.len = 0;
                // Keep the prefix alive (empty) so that `set_encoding` still
                // works while the caller inspects the XML declaration that was
                // just decoded from it.
                self.prefix = Some(prefix);
                self.decode_chunk(&bytes[..len], false)?;
                if !self.out.is_empty() {
                    return Ok(self.out.as_bytes());
                }
                // Nothing was produced (for example a lone byte of a UTF-16 code
                // unit): drop the prefix and continue with the main path.
                self.prefix = None;
            }
        }

        if self.finished {
            return Ok(&[]);
        }

        loop {
            if self.inner_eof {
                self.finished = true;
                self.decode_chunk(&[], true)?;
                return Ok(self.out.as_bytes());
            }

            let chunk = {
                let available = self.inner.fill_buf()?;
                if available.is_empty() {
                    self.inner_eof = true;
                    continue;
                }
                available.to_vec()
            };
            self.inner.consume(chunk.len());
            self.decode_chunk(&chunk, false)?;
            if !self.out.is_empty() {
                return Ok(self.out.as_bytes());
            }
            // The decoder consumed everything into its internal state without
            // producing output; an empty slice would be read as end of input, so
            // more input has to be fed to it.
        }
    }

    fn consume(&mut self, amount: usize) {
        self.out_pos = (self.out_pos + amount).min(self.out.len());
    }
}

#[cfg(feature = "encoding")]
impl<R: std::io::BufRead> std::io::Read for DecodingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let available = std::io::BufRead::fill_buf(self)?;
        if available.is_empty() {
            return Ok(0);
        }
        let len = available.len().min(buf.len());
        buf[..len].copy_from_slice(&available[..len]);
        std::io::BufRead::consume(self, len);
        Ok(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_decode_borrows() -> Result<(), EncodingError> {
        let decoder = Decoder::utf8();
        assert_eq!(decoder.decode(b"hello")?, Cow::Borrowed("hello"));
        Ok(())
    }

    /// Malformed UTF-8 is rejected. Which variant is reported follows
    /// quick-xml 0.41: without the `encoding` feature `decode` is a plain
    /// `str::from_utf8`, with the feature it goes through the same fallible path
    /// as every other encoding and reports `Other(UTF-8)`.
    #[test]
    fn utf8_decode_rejects_invalid() {
        let decoder = Decoder::utf8();
        #[cfg(not(feature = "encoding"))]
        assert!(matches!(
            decoder.decode(&[0xFF, 0xFE]),
            Err(EncodingError::Utf8(_))
        ));
        #[cfg(feature = "encoding")]
        {
            assert_eq!(
                decoder.decode(&[0xFF, 0xFE]),
                Err(EncodingError::Other(Encoding::Utf8))
            );
            // `decode_into` keeps the `Utf8` variant, like quick-xml.
            let mut buf = String::new();
            assert!(matches!(
                decoder.decode_into(&[0xFF, 0xFE], &mut buf),
                Err(EncodingError::Utf8(_))
            ));
        }
    }

    #[test]
    fn detect() {
        assert_eq!(
            detect_encoding(b"\xEF\xBB\xBF<?xml").map(|d| d.bom_len()),
            Some(3)
        );
        assert_eq!(
            detect_encoding(b"<?xml version").map(|d| d.bom_len()),
            Some(0)
        );
        assert_eq!(detect_encoding(b"plain text"), None);
    }

    #[cfg(feature = "encoding")]
    #[test]
    fn non_utf8_decode() -> Result<(), EncodingError> {
        // Windows-1252 0xE9 == 'é'
        let decoder = Decoder {
            encoding: Encoding::Windows1252,
        };
        assert_eq!(decoder.decode(&[b'r', b'o', b's', 0xE9])?, "rosé");
        Ok(())
    }

    /// Regression: a byte-order mark is an authoritative signal of the actual
    /// UTF-16 byte order, so a bare, endianness-free `encoding="UTF-16"`
    /// declaration must not override it. Previously the reader's second
    /// (declaration-driven) refinement stage trusted
    /// `Encoding::for_label("UTF-16")`'s WHATWG default (`Utf16Le`) outright,
    /// silently flipping a big-endian document to little-endian mojibake —
    /// this is the same class of bug that was fixed in
    /// `oxixml_encoding::SniffResult::best_effort_encoding`, but reachable
    /// independently through this crate's own `Reader`/`EncodingRef` state
    /// machine.
    ///
    /// The document body is deliberately plain ASCII (not genuinely
    /// 2-bytes-per-character): only a byte-order mark makes it *sniff* as
    /// UTF-16. `Reader` only understands ASCII-compatible byte streams (see
    /// `DecodingReader`'s docs) unless wrapped by `DecodingReader`, so this is
    /// the only way to reach the buggy refinement step through the crate's
    /// public API without the unrelated, documented tokenizer limitation for
    /// genuine 2-bytes-per-character content masking the result.
    #[cfg(feature = "encoding")]
    #[test]
    fn bare_utf16_label_does_not_override_the_boms_byte_order() {
        use crate::events::Event;
        use crate::reader::Reader;

        for big_endian in [true, false] {
            let mut bytes = if big_endian {
                vec![0xFE, 0xFF]
            } else {
                vec![0xFF, 0xFE]
            };
            bytes.extend_from_slice(b"<?xml version=\"1.0\" encoding=\"UTF-16\"?><r>hi</r>");

            let mut reader = Reader::from_reader(bytes.as_slice());
            let expected = if big_endian {
                Encoding::Utf16Be
            } else {
                Encoding::Utf16Le
            };

            match reader.read_event() {
                Ok(Event::Decl(_)) => {}
                other => panic!("big_endian={big_endian}: expected Decl, got {other:?}"),
            }
            assert_eq!(
                reader.decoder().encoding(),
                expected,
                "big_endian={big_endian}: the BOM's byte order must survive the declaration",
            );

            // And it stays that way for the rest of the document.
            loop {
                match reader.read_event() {
                    Ok(Event::Eof) => break,
                    Ok(_) => {}
                    Err(e) => panic!("big_endian={big_endian}: unexpected error {e:?}"),
                }
            }
            assert_eq!(
                reader.decoder().encoding(),
                expected,
                "big_endian={big_endian}"
            );
        }
    }
}

#[cfg(all(test, feature = "encoding"))]
mod decoding_reader {
    use super::*;
    use std::io::{BufRead, Read};

    /// A `BufRead` source that hands out at most `chunk` bytes at a time, so
    /// that multi-byte sequences are split across `fill_buf` calls.
    struct Chunked<'a> {
        data: &'a [u8],
        pos: usize,
        chunk: usize,
    }

    impl std::io::Read for Chunked<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let available = self.fill_buf()?;
            let len = available.len().min(buf.len());
            buf[..len].copy_from_slice(&available[..len]);
            self.pos += len;
            Ok(len)
        }
    }

    impl std::io::BufRead for Chunked<'_> {
        fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
            let end = (self.pos + self.chunk).min(self.data.len());
            Ok(&self.data[self.pos..end])
        }

        fn consume(&mut self, amount: usize) {
            self.pos = (self.pos + amount).min(self.data.len());
        }
    }

    /// Encodes `text` as UTF-16, optionally with a byte-order mark.
    fn utf16(text: &str, little_endian: bool, bom: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        if bom {
            bytes.extend_from_slice(if little_endian {
                UTF16_LE_BOM
            } else {
                UTF16_BE_BOM
            });
        }
        for unit in text.encode_utf16() {
            if little_endian {
                bytes.extend_from_slice(&unit.to_le_bytes());
            } else {
                bytes.extend_from_slice(&unit.to_be_bytes());
            }
        }
        bytes
    }

    fn decode_all(bytes: &[u8], chunk: usize) -> std::io::Result<String> {
        let mut reader = DecodingReader::new(Chunked {
            data: bytes,
            pos: 0,
            chunk,
        });
        let mut out = Vec::new();
        reader.read_to_end(&mut out)?;
        String::from_utf8(out)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }

    #[test]
    fn utf8_passes_through() -> std::io::Result<()> {
        assert_eq!(decode_all(b"<a>text</a>", 3)?, "<a>text</a>");
        assert_eq!(decode_all(UTF8_BOM, 1)?, "");
        Ok(())
    }

    #[test]
    fn transcodes_utf16_at_every_chunk_size() -> std::io::Result<()> {
        let text = "<?xml version='1.0' encoding='UTF-16'?><doc>текст</doc>";
        for little_endian in [true, false] {
            for bom in [true, false] {
                for chunk in [1, 2, 3, 7, 64, 4096] {
                    let bytes = utf16(text, little_endian, bom);
                    assert_eq!(
                        decode_all(&bytes, chunk)?,
                        text,
                        "little_endian={little_endian} bom={bom} chunk={chunk}",
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn reports_malformed_input() {
        // A lone UTF-16 high surrogate at the end of the stream.
        let mut bytes = utf16("<a/>", true, true);
        bytes.extend_from_slice(&[0x00, 0xD8]);
        let error = decode_all(&bytes, 4).expect_err("a truncated sequence must fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn exposes_and_replaces_the_encoding() -> std::io::Result<()> {
        // The declaration fits into the upfront-read prefix; the non-ASCII byte
        // deliberately does not, so it is decoded with the installed encoding.
        let mut bytes = b"<?xml version='1.0' encoding='ISO-8859-1'?><a>".to_vec();
        bytes.extend_from_slice(&b"-".repeat(32));
        bytes.extend_from_slice(&[0xE9]);
        bytes.extend_from_slice(b"</a>");

        let mut reader = DecodingReader::new(&bytes[..]);
        assert_eq!(reader.encoding(), Encoding::Utf8);
        assert_eq!(reader.get_ref().len(), bytes.len());

        // Serve the prefix, which is what the XML declaration is parsed from...
        assert!(!reader.fill_buf()?.is_empty());
        // ... then install the declared encoding, still in time.
        assert!(reader.set_encoding(Encoding::Latin1));
        assert_eq!(reader.encoding(), Encoding::Latin1);

        let mut out = String::new();
        reader.read_to_string(&mut out)?;
        assert!(out.ends_with("\u{e9}</a>"), "{out}");
        assert!(reader.get_mut().is_empty());
        assert!(reader.into_inner().is_empty());
        Ok(())
    }

    #[test]
    fn refuses_to_change_the_encoding_too_late() -> std::io::Result<()> {
        let bytes =
            b"<a>0123456789012345678901234567890123456789012345678901234567890123456789</a>";
        let mut reader = DecodingReader::new(&bytes[..]);
        let mut out = String::new();
        reader.read_to_string(&mut out)?;
        assert_eq!(out, std::str::from_utf8(bytes).expect("utf-8"));
        // Setting the encoding that is already in use is a no-op that succeeds.
        assert!(reader.set_encoding(Encoding::Utf8));
        // Changing it once the prefix is gone is refused instead of panicking.
        assert!(!reader.set_encoding(Encoding::Utf16Le));
        Ok(())
    }

    #[test]
    fn parses_a_utf16_document_with_the_reader() -> crate::errors::Result<()> {
        use crate::events::Event;
        use crate::reader::Reader;

        // The document carries no `encoding` pseudo-attribute: the reader would
        // otherwise apply it to the *transcoded* (UTF-8) stream, see the note on
        // `DecodingReader`.
        let bytes = utf16("<?xml version='1.0'?><doc>hi</doc>", true, true);
        let mut reader = Reader::from_reader(DecodingReader::new(bytes.as_slice()));
        let mut buf = Vec::new();
        let mut texts = Vec::new();
        loop {
            buf.clear();
            match reader.read_event_into(&mut buf)? {
                Event::Eof => break,
                Event::Text(text) => texts.push(text.decode()?.into_owned()),
                _ => continue,
            }
        }
        assert_eq!(texts, vec!["hi".to_owned()]);
        Ok(())
    }
}

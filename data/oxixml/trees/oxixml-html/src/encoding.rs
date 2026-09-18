//! Determining the character encoding of a byte stream, and parsing from
//! bytes.
//!
//! This is the algorithm of
//! [HTML § 13.2.3.2](https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding):
//! a byte order mark wins outright, then any encoding the transport layer
//! declared, then the `<meta charset>` prescan, and finally the historical
//! default of `windows-1252`.
//!
//! The prescan's own numbered steps have no byte-count limit: HTML's mention
//! of "1024 bytes" is a recommendation for how much a network-backed
//! implementation should read ahead before giving up, not a cap the
//! algorithm enforces on itself — a comment or tag left open when that
//! budget is exhausted keeps consuming input for as long as it takes to
//! close, however far into the document that is (see `prescan`). Since
//! this crate always receives the complete document already in memory,
//! there is no read-ahead budget to bound in the first place, so it scans
//! the entire buffer it is given.
//!
//! The decoding itself is [`oxixml_encoding`]'s, which supports the encodings
//! an XML or HTML processor actually needs; a document in an encoding this
//! workspace does not implement is decoded as `windows-1252`, which is exactly
//! what a browser does with an unrecognised label.

use oxixml_encoding::Encoding;

use crate::parse::{ParseOptions, ParseOutput, parse_document_with};

/// How sure the sniffing algorithm is about the encoding it picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodingConfidence {
    /// A byte order mark or the transport layer said so; nothing can override
    /// it.
    Certain,
    /// A `<meta>` element or the default said so; a later `<meta>` could in
    /// principle change it.
    Tentative,
}

/// The outcome of [`sniff_encoding`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SniffedEncoding {
    /// The encoding to decode the stream with.
    pub encoding: Encoding,
    /// How the encoding was arrived at.
    pub confidence: EncodingConfidence,
    /// The length in bytes of the byte order mark, which the decoder must
    /// skip; zero when there was none.
    pub bom_length: usize,
}

/// Determines the encoding of an HTML byte stream.
///
/// `transport_charset` is the `charset` parameter of the `Content-Type` header,
/// when one is known.
///
/// # Examples
///
/// ```
/// use oxixml_encoding::Encoding;
/// use oxixml_html::{EncodingConfidence, sniff_encoding};
///
/// // A byte order mark is certain.
/// let sniffed = sniff_encoding(b"\xef\xbb\xbf<html>", None);
/// assert_eq!(sniffed.encoding, Encoding::Utf8);
/// assert_eq!(sniffed.confidence, EncodingConfidence::Certain);
/// assert_eq!(sniffed.bom_length, 3);
///
/// // A meta charset is tentative. Note that the Encoding Standard's label
/// // table — which HTML mandates — makes `iso-8859-1` an alias of
/// // windows-1252, unlike the XML label table `oxixml_encoding` exposes.
/// let sniffed = sniff_encoding(b"<meta charset=\"iso-8859-2\">", None);
/// assert_eq!(sniffed.confidence, EncodingConfidence::Tentative);
/// assert_eq!(
///     sniff_encoding(b"<meta charset=\"iso-8859-1\">", None).encoding,
///     Encoding::Windows1252
/// );
///
/// // And the historical default is windows-1252.
/// assert_eq!(sniff_encoding(b"<p>hi", None).encoding, Encoding::Windows1252);
/// ```
#[must_use]
pub fn sniff_encoding(bytes: &[u8], transport_charset: Option<&str>) -> SniffedEncoding {
    if let Some((encoding, bom_length)) = detect_bom(bytes) {
        return SniffedEncoding {
            encoding,
            confidence: EncodingConfidence::Certain,
            bom_length,
        };
    }
    if let Some(encoding) = transport_charset.and_then(encoding_for_label) {
        return SniffedEncoding {
            encoding,
            confidence: EncodingConfidence::Certain,
            bom_length: 0,
        };
    }
    let encoding = prescan(bytes).unwrap_or(Encoding::Windows1252);
    SniffedEncoding {
        encoding,
        confidence: EncodingConfidence::Tentative,
        bom_length: 0,
    }
}

/// Resolves a character encoding label the way HTML requires.
///
/// [HTML § 13.2.3.2](https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding)
/// resolves every label — the transport layer's `charset`, a `<meta charset>`,
/// the `charset` inside a `<meta http-equiv="content-type">` — through the
/// [Encoding Standard's "get an encoding"](https://encoding.spec.whatwg.org/#concept-encoding-get)
/// label table, which is deliberately *not* the same mapping an XML processor
/// uses. In particular the table assigns `iso-8859-1`, `latin1`, `ascii` and
/// `us-ascii` to **windows-1252**, because that is what the deployed web means
/// by them: a document labelled `iso-8859-1` that contains byte 0x93 wants a
/// left double quotation mark, not a C1 control.
///
/// [`oxixml_encoding`] is a general XML encoding crate and correctly keeps
/// ISO-8859-1 and windows-1252 apart, so this function applies the web's
/// aliasing on top of it before falling back to the general table.
fn encoding_for_label(label: &str) -> Option<Encoding> {
    /// The labels the Encoding Standard assigns to windows-1252.
    const WINDOWS_1252_LABELS: [&str; 17] = [
        "ansi_x3.4-1968",
        "ascii",
        "cp1252",
        "cp819",
        "csisolatin1",
        "ibm819",
        "iso-8859-1",
        "iso-ir-100",
        "iso8859-1",
        "iso88591",
        "iso_8859-1",
        "iso_8859-1:1987",
        "l1",
        "latin1",
        "us-ascii",
        "windows-1252",
        "x-cp1252",
    ];

    let trimmed = label.trim_matches(|ch: char| matches!(ch, '\t' | '\n' | '\u{c}' | '\r' | ' '));
    if WINDOWS_1252_LABELS
        .iter()
        .any(|known| known.eq_ignore_ascii_case(trimmed))
    {
        return Some(Encoding::Windows1252);
    }
    Encoding::for_label(trimmed)
}

fn detect_bom(bytes: &[u8]) -> Option<(Encoding, usize)> {
    match bytes {
        [0xef, 0xbb, 0xbf, ..] => Some((Encoding::Utf8, 3)),
        [0xfe, 0xff, ..] => Some((Encoding::Utf16Be, 2)),
        [0xff, 0xfe, ..] => Some((Encoding::Utf16Le, 2)),
        _ => None,
    }
}

/// Parses an HTML document from bytes, sniffing the encoding first.
///
/// # Examples
///
/// ```
/// use oxixml_html::parse_bytes;
///
/// // windows-1252 byte 0x93 is a left double quotation mark.
/// let parsed = parse_bytes(b"<p>\x93quoted\x94", None);
/// let document = parsed.document();
/// let body = document
///     .get_elements_by_tag_name_ns(document.root(), Some(oxixml_html::HTML_NAMESPACE), "p")
///     .next()
///     .expect("a p element");
/// assert_eq!(document.text_content(body).as_deref(), Some("\u{201c}quoted\u{201d}"));
/// ```
#[must_use]
pub fn parse_bytes(bytes: &[u8], transport_charset: Option<&str>) -> ParseOutput {
    parse_bytes_with(bytes, transport_charset, &ParseOptions::new())
}

/// Parses an HTML document from bytes with explicit options.
#[must_use]
pub fn parse_bytes_with(
    bytes: &[u8],
    transport_charset: Option<&str>,
    options: &ParseOptions,
) -> ParseOutput {
    let sniffed = sniff_encoding(bytes, transport_charset);
    let mut decoder = sniffed.encoding.new_decoder();
    let mut text = String::with_capacity(bytes.len());
    decoder.decode_to_string_lossy(&bytes[sniffed.bom_length..], &mut text, true);
    parse_document_with(&text, options)
}

// ------------------------------------------------------------------ prescan

/// "Prescan a byte stream to determine its encoding"
/// ([HTML § 13.2.3.2](https://html.spec.whatwg.org/multipage/parsing.html#prescan-a-byte-stream-to-determine-its-encoding)).
///
/// The algorithm's numbered steps carry no byte-count limit of their own: a
/// comment or tag that is still open keeps consuming bytes for as long as it
/// takes to close, and the scan for the *next* construct resumes right where
/// that left off, however far into the document that is. HTML's "user
/// agents are encouraged to only prescan the first 1024 bytes" is a
/// recommendation for how much a network-backed implementation should read
/// ahead before giving up, not a requirement the algorithm enforces on the
/// bytes it was actually given — html5lib-tests' `encoding/tests1.dat`
/// checks exactly this with a battery of blocks (`"2048 characters"`,
/// `"4096 characters"`, `"8192 characters"`, ...) that place a multi-kilobyte
/// comment before the `<meta charset>` under test, so that an implementation
/// which stopped dead at some fixed read-ahead size — 1024 bytes or any
/// other typical buffer size — would miss it. This crate always receives the
/// complete document already in memory, so it simply scans the whole buffer
/// it is given.
///
/// If, at any point, a construct (a comment, a tag, or one of its
/// attributes) is still open when the byte stream runs out, the *entire*
/// prescan aborts and reports no encoding, discarding anything a still-open
/// `<meta>` tag had tentatively found — see [`MetaOutcome::EndOfStream`] and
/// [`AttributeScan::EndOfStream`].
fn prescan(bytes: &[u8]) -> Option<Encoding> {
    let mut position = 0;
    while position < bytes.len() {
        let rest = &bytes[position..];
        if rest.starts_with(b"<!--") {
            position += 2;
            position = match find(bytes, position, b"-->") {
                Some(index) => index + 3,
                // The comment never closed: § 13.2.3.2 aborts the whole
                // prescan here, which coincides with just letting the outer
                // loop end below — there is nothing left to look at either
                // way.
                None => return None,
            };
        } else if starts_with_ignore_case(rest, b"<meta")
            && rest
                .get(5)
                .is_some_and(|&byte| is_space(byte) || byte == b'/')
        {
            position += 5;
            match prescan_meta(bytes, &mut position) {
                MetaOutcome::Found(encoding) => return Some(encoding),
                MetaOutcome::NotFound => {}
                MetaOutcome::EndOfStream => return None,
            }
        } else if rest.len() >= 2
            && rest[0] == b'<'
            && (rest[1].is_ascii_alphabetic()
                || (rest[1] == b'/' && rest.get(2).is_some_and(u8::is_ascii_alphabetic)))
        {
            position += 2;
            while position < bytes.len() && !is_space(bytes[position]) && bytes[position] != b'>' {
                position += 1;
            }
            loop {
                match get_attribute(bytes, &mut position) {
                    AttributeScan::Found(..) => {}
                    AttributeScan::Done => break,
                    AttributeScan::EndOfStream => return None,
                }
            }
        } else if rest.starts_with(b"<!") || rest.starts_with(b"</") || rest.starts_with(b"<?") {
            position = find(bytes, position, b">").map_or(bytes.len(), |index| index + 1);
        } else {
            position += 1;
        }
    }
    None
}

/// The outcome of scanning one `<meta>` element for a character encoding, in
/// [`prescan`]'s meta-element branch.
enum MetaOutcome {
    /// A usable encoding was found: a `charset` attribute that resolved to
    /// one, or a pragma-backed `content` attribute that did.
    Found(Encoding),
    /// The element closed (`>`) without yielding a usable encoding; the
    /// outer scan should keep looking at whatever comes after it.
    NotFound,
    /// The byte stream ended before the element closed. HTML § 13.2.3.2
    /// aborts the *entire* prescan in this case — see
    /// [`AttributeScan::EndOfStream`] — so the caller must not fall back on
    /// anything a partially-scanned attribute list had tentatively found.
    EndOfStream,
}

/// The attribute loop of a `<meta>` element found by the prescan.
fn prescan_meta(bytes: &[u8], position: &mut usize) -> MetaOutcome {
    let mut got_pragma = false;
    let mut need_pragma: Option<bool> = None;
    let mut charset: Option<Encoding> = None;

    loop {
        let (name, value) = match get_attribute(bytes, position) {
            AttributeScan::Found(name, value) => (name, value),
            AttributeScan::Done => break,
            AttributeScan::EndOfStream => return MetaOutcome::EndOfStream,
        };
        match name.as_slice() {
            b"http-equiv" if value == b"content-type" => got_pragma = true,
            b"content" if charset.is_none() => {
                if let Some(found) = extract_encoding_from_meta(&value) {
                    charset = Some(found);
                    need_pragma = Some(true);
                }
            }
            b"charset" => {
                charset = encoding_for_label(&String::from_utf8_lossy(&value));
                need_pragma = Some(false);
            }
            _ => {}
        }
    }

    let Some(need_pragma) = need_pragma else {
        return MetaOutcome::NotFound;
    };
    if need_pragma && !got_pragma {
        return MetaOutcome::NotFound;
    }
    match charset {
        // "If the encoding is UTF-16BE/LE, change it to UTF-8."
        Some(Encoding::Utf16Be | Encoding::Utf16Le) => MetaOutcome::Found(Encoding::Utf8),
        Some(other) => MetaOutcome::Found(other),
        None => MetaOutcome::NotFound,
    }
}

/// The outcome of [`get_attribute`], the prescan's "get an attribute"
/// algorithm.
enum AttributeScan {
    /// A `name=value` pair, both already ASCII-lowercased as the algorithm
    /// requires.
    Found(Vec<u8>, Vec<u8>),
    /// The byte at `position` was `>`: "abort the get an attribute
    /// algorithm. There isn't one." A normal, successful end to the
    /// attribute list — `position` is left pointing *at* the `>`, for the
    /// caller to consume.
    Done,
    /// `position` ran past the end of the byte stream before this
    /// attribute's name or value could be completed. HTML § 13.2.3.2: "If at
    /// any point during these steps ... the user agent ... runs out of
    /// bytes ... then abort the prescan a byte stream to determine its
    /// encoding algorithm" — every caller of [`get_attribute`] must
    /// propagate this as a total abort rather than use whatever was
    /// collected so far.
    EndOfStream,
}

/// "Get an attribute" of the prescan algorithm.
fn get_attribute(bytes: &[u8], position: &mut usize) -> AttributeScan {
    loop {
        match bytes.get(*position) {
            None => return AttributeScan::EndOfStream,
            Some(&byte) if is_space(byte) || byte == b'/' => *position += 1,
            Some(_) => break,
        }
    }
    if bytes[*position] == b'>' {
        return AttributeScan::Done;
    }

    let mut name = Vec::new();
    loop {
        let Some(&byte) = bytes.get(*position) else {
            return AttributeScan::EndOfStream;
        };
        match byte {
            b'=' if !name.is_empty() => {
                *position += 1;
                break;
            }
            b if is_space(b) => {
                // Spaces after the name: skip them and expect an `=`.
                *position += 1;
                loop {
                    match bytes.get(*position) {
                        None => return AttributeScan::EndOfStream,
                        Some(&byte) if is_space(byte) => *position += 1,
                        Some(_) => break,
                    }
                }
                if bytes[*position] != b'=' {
                    return AttributeScan::Found(name, Vec::new());
                }
                *position += 1;
                break;
            }
            b'/' | b'>' => return AttributeScan::Found(name, Vec::new()),
            b => {
                name.push(b.to_ascii_lowercase());
                *position += 1;
            }
        }
    }

    loop {
        match bytes.get(*position) {
            None => return AttributeScan::EndOfStream,
            Some(&byte) if is_space(byte) => *position += 1,
            Some(_) => break,
        }
    }

    let mut value = Vec::new();
    match bytes[*position] {
        quote @ (b'"' | b'\'') => {
            *position += 1;
            loop {
                let Some(&byte) = bytes.get(*position) else {
                    return AttributeScan::EndOfStream;
                };
                *position += 1;
                if byte == quote {
                    return AttributeScan::Found(name, value);
                }
                value.push(byte.to_ascii_lowercase());
            }
        }
        b'>' => AttributeScan::Found(name, value),
        first => {
            value.push(first.to_ascii_lowercase());
            *position += 1;
            loop {
                match bytes.get(*position) {
                    None => return AttributeScan::EndOfStream,
                    Some(&byte) if is_space(byte) || byte == b'>' => break,
                    Some(&byte) => {
                        value.push(byte.to_ascii_lowercase());
                        *position += 1;
                    }
                }
            }
            AttributeScan::Found(name, value)
        }
    }
}

/// "Extract a character encoding from a meta element" — finds `charset=` inside
/// a `content` attribute value.
fn extract_encoding_from_meta(value: &[u8]) -> Option<Encoding> {
    let mut position = 0;
    loop {
        let index = find(value, position, b"charset")?;
        position = index + 7;
        let mut cursor = position;
        while cursor < value.len() && is_space(value[cursor]) {
            cursor += 1;
        }
        if value.get(cursor) != Some(&b'=') {
            continue;
        }
        cursor += 1;
        while cursor < value.len() && is_space(value[cursor]) {
            cursor += 1;
        }
        let label = match value.get(cursor) {
            Some(&quote @ (b'"' | b'\'')) => {
                cursor += 1;
                let start = cursor;
                while cursor < value.len() && value[cursor] != quote {
                    cursor += 1;
                }
                if cursor >= value.len() {
                    return None;
                }
                &value[start..cursor]
            }
            Some(_) => {
                let start = cursor;
                while cursor < value.len() && !is_space(value[cursor]) && value[cursor] != b';' {
                    cursor += 1;
                }
                &value[start..cursor]
            }
            None => return None,
        };
        return encoding_for_label(&String::from_utf8_lossy(label));
    }
}

fn find(haystack: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if from >= haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|index| index + from)
}

fn starts_with_ignore_case(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.len() >= needle.len() && haystack[..needle.len()].eq_ignore_ascii_case(needle)
}

const fn is_space(byte: u8) -> bool {
    matches!(byte, b'\t' | b'\n' | b'\x0c' | b'\r' | b' ')
}

#[cfg(test)]
mod tests {
    use super::{Encoding, EncodingConfidence, prescan, sniff_encoding};

    #[test]
    fn meta_charset_is_found() {
        assert_eq!(
            prescan(b"<html><head><meta charset=utf-8></head>"),
            Some(Encoding::Utf8)
        );
        // The Encoding Standard's label table, which HTML mandates, makes
        // `iso-8859-1` an alias of windows-1252.
        assert_eq!(
            prescan(b"<meta charset='ISO-8859-1'>"),
            Some(Encoding::Windows1252)
        );
        assert_eq!(
            prescan(b"<meta charset=us-ascii>"),
            Some(Encoding::Windows1252)
        );
    }

    #[test]
    fn meta_content_type_needs_the_pragma() {
        assert_eq!(
            prescan(b"<meta http-equiv=\"content-type\" content=\"text/html; charset=utf-8\">"),
            Some(Encoding::Utf8)
        );
        // Without the pragma the content attribute is ignored.
        assert_eq!(
            prescan(b"<meta content=\"text/html; charset=utf-8\">"),
            None
        );
    }

    #[test]
    fn comments_and_other_tags_are_skipped() {
        assert_eq!(
            prescan(b"<!-- <meta charset=utf-8> --><meta charset=windows-1252>"),
            Some(Encoding::Windows1252)
        );
        assert_eq!(prescan(b"<!doctype html><p>nothing here"), None);
    }

    #[test]
    fn utf16_in_a_meta_becomes_utf8() {
        assert_eq!(prescan(b"<meta charset=utf-16>"), Some(Encoding::Utf8));
    }

    #[test]
    fn transport_beats_the_prescan() {
        let sniffed = sniff_encoding(b"<meta charset=utf-8>", Some("windows-1252"));
        assert_eq!(sniffed.encoding, Encoding::Windows1252);
        assert_eq!(sniffed.confidence, EncodingConfidence::Certain);
    }

    #[test]
    fn scanning_is_not_capped_at_one_kilobyte() {
        // HTML § 13.2.3.2's "1024 bytes" is a recommendation for how much a
        // network-backed implementation should read ahead, not a limit its
        // numbered steps enforce: nothing stops the scan from reaching a
        // `<meta>` well past that point once the bytes are actually
        // available, as they always are here.
        let mut input = vec![b' '; 1100];
        input.extend_from_slice(b"<meta charset=utf-8>");
        assert_eq!(prescan(&input), Some(Encoding::Utf8));
    }

    #[test]
    fn a_comment_left_open_past_a_kilobyte_does_not_hide_the_meta_after_it() {
        // Mirrors html5lib-tests' encoding/tests1.dat buffer-boundary blocks
        // ("2048 characters", "4096 characters", "8192 characters", ...): a
        // comment many kilobytes long must not stop the scan from reaching
        // the `<meta charset>` immediately after it closes.
        let mut input = b"<!-- ".to_vec();
        input.extend(std::iter::repeat_n(b'x', 4096));
        input.extend_from_slice(b" --><meta charset=iso-8859-2>");
        assert_eq!(prescan(&input), Some(Encoding::Iso8859_2));
    }

    #[test]
    fn an_unclosed_comment_aborts_the_scan() {
        let mut input = b"<!-- ".to_vec();
        input.extend(std::iter::repeat_n(b'x', 4096));
        // No "-->": the comment, and the prescan, never end.
        assert_eq!(prescan(&input), None);
    }

    #[test]
    fn an_unterminated_unquoted_attribute_value_aborts_the_whole_prescan() {
        // The byte stream ends mid-value, with no closing `>` in sight. HTML
        // § 13.2.3.2 aborts the *entire* prescan in this case — this must
        // not be conflated with the tag simply having no more attributes,
        // which would let a `charset` this malformed still win.
        assert_eq!(prescan(b"<meta charset=iso-8859-2"), None);
    }

    #[test]
    fn an_unterminated_quoted_attribute_value_aborts_the_whole_prescan() {
        assert_eq!(prescan(b"<meta charset=\"iso-8859-2"), None);
    }

    #[test]
    fn a_valueless_attribute_does_not_stop_a_later_charset_in_the_same_tag() {
        assert_eq!(
            prescan(b"<meta bogus charset=iso-8859-2>"),
            Some(Encoding::Iso8859_2)
        );
    }
}

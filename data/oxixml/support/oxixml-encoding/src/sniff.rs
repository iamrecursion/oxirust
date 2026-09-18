//! XML entity encoding sniffing, per
//! [XML 1.0 Appendix F: Autodetection of Character Encodings](https://www.w3.org/TR/xml/#sec-guessing).
//!
//! Appendix F describes a two-stage heuristic for figuring out how to start decoding an XML
//! entity that has no external encoding information (no `Content-Type` charset parameter, for
//! example):
//!
//! 1. Look at the entity's first few bytes. A byte-order mark, or one of a handful of
//!    fixed 2-or-4-byte patterns that `<?xm` (the start of every XML declaration) transforms
//!    into under different encodings, narrows things down to a coarse "encoding family" without
//!    decoding anything yet — see [`ByteOrderGuess`].
//! 2. If that family is ASCII-compatible (or UTF-16, whose ASCII range is trivially recovered),
//!    decode *just enough* of the prologue to search for a `<?xml ... encoding="..." ?>`
//!    pseudo-attribute and read the declared label out of it.
//!
//! [`sniff_xml_encoding`] performs both stages and returns everything discovered along the way
//! as a [`SniffResult`]; policy decisions (e.g. "trust the declared label over the byte-order
//! guess") are left to the caller, though [`SniffResult::best_effort_encoding`] implements the
//! reasonable default.

use crate::label::Encoding;

/// The number of bytes (for single-byte-scannable families) or code units (for UTF-16 families,
/// where the byte budget is doubled) of the entity's prologue that [`sniff_xml_encoding`] will
/// examine when searching for an `encoding="..."` pseudo-attribute.
///
/// An XML declaration is required to be the very first thing in the entity and is normally well
/// under 100 bytes, so this is generous headroom rather than a tight fit.
const SCAN_LIMIT_BYTES: usize = 1024;

/// The coarse encoding family implied purely by the first few bytes of an XML entity, before any
/// attempt to read an `encoding="..."` declaration.
///
/// This mirrors the byte-pattern table of XML 1.0 Appendix F.1 in full, including the UTF-32 and
/// EBCDIC patterns that this crate cannot itself decode ([`Encoding`] has no such variants) —
/// recognizing the pattern is still useful so a caller can produce a precise diagnostic ("this
/// looks like EBCDIC, which is unsupported") instead of a generic decode failure. Use
/// [`ByteOrderGuess::supported_encoding`] to get the [`Encoding`] this crate can actually decode
/// with, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrderGuess {
    /// `EF BB BF` (BOM) or `3C 3F 78 6D` (`<?xm` as ASCII/UTF-8/Latin-1/windows-1252/...).
    Utf8,
    /// `FF FE` (BOM) or `3C 00 3F 00` (`<?xm` as UTF-16, little-endian, no BOM).
    Utf16Le,
    /// `FE FF` (BOM) or `00 3C 00 3F` (`<?xm` as UTF-16, big-endian, no BOM).
    Utf16Be,
    /// `FF FE 00 00`: UTF-32, little-endian. Not decodable by this crate.
    Utf32Le,
    /// `00 00 FE FF`: UTF-32, big-endian. Not decodable by this crate.
    Utf32Be,
    /// `00 00 FF FE`: UTF-32 with an unusual 2-1-4-3 octet order. Not decodable by this crate.
    Utf32Order2143,
    /// `FE FF 00 00`: UTF-32 with an unusual 3-4-1-2 octet order. Not decodable by this crate.
    Utf32Order3412,
    /// `4C 6F A7 94` (`<?xm` in an EBCDIC code page). Not decodable by this crate.
    Ebcdic,
}

impl ByteOrderGuess {
    /// The [`Encoding`] this crate can actually decode with for this guess, if any.
    ///
    /// [`ByteOrderGuess::Utf8`] maps to [`Encoding::Utf8`] here as a *transport* guess only: at
    /// this stage all that's known is that the entity is ASCII-compatible, not specifically that
    /// it is UTF-8 rather than, say, windows-1252 — [`SniffResult::declared_encoding`] (read from
    /// the prologue) is what actually distinguishes those.
    #[must_use]
    pub const fn supported_encoding(self) -> Option<Encoding> {
        match self {
            ByteOrderGuess::Utf8 => Some(Encoding::Utf8),
            ByteOrderGuess::Utf16Le => Some(Encoding::Utf16Le),
            ByteOrderGuess::Utf16Be => Some(Encoding::Utf16Be),
            ByteOrderGuess::Utf32Le
            | ByteOrderGuess::Utf32Be
            | ByteOrderGuess::Utf32Order2143
            | ByteOrderGuess::Utf32Order3412
            | ByteOrderGuess::Ebcdic => None,
        }
    }
}

/// The result of [`sniff_xml_encoding`]: everything the Appendix F heuristic could determine
/// about an entity's encoding from its leading bytes and (if reachable) its XML declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SniffResult {
    /// The coarse family implied by the entity's first bytes, or [`None`] if none of the
    /// Appendix F patterns matched (which, per the spec, itself implies UTF-8 or an
    /// ASCII-compatible encoding: see [`SniffResult::best_effort_encoding`]).
    pub byte_order_guess: Option<ByteOrderGuess>,
    /// The number of leading bytes identified as a byte-order mark; `0` if none was present.
    /// These bytes are *not* included in [`declared_label`](SniffResult::declared_label)
    /// scanning and should be skipped before decoding the entity itself.
    pub bom_len: usize,
    /// The raw label text captured from an `encoding="..."` (or `encoding='...'`) pseudo-attribute
    /// in the XML declaration, if the prologue could be scanned and one was found.
    pub declared_label: Option<String>,
    /// [`declared_label`](SniffResult::declared_label) resolved through [`Encoding::for_label`],
    /// if it was present and recognized.
    pub declared_encoding: Option<Encoding>,
}

impl SniffResult {
    /// A reasonable single [`Encoding`] to start decoding with, combining everything this
    /// [`SniffResult`] discovered: the declared label wins if present and recognized, otherwise
    /// the byte-order guess if it maps to a supported encoding, otherwise the XML 1.0 default of
    /// UTF-8.
    ///
    /// # UTF-16
    ///
    /// One case is deliberately *not* decided by the declaration: a UTF-16 byte order. The label
    /// `UTF-16` carries no endianness of its own (XML 1.0 Appendix F and RFC 2781 both say the
    /// byte-order mark, or the Appendix F.1 four-byte pattern, is what determines it), while
    /// WHATWG's label table — which [`Encoding::for_label`] implements — resolves the bare label
    /// to [`Encoding::Utf16Le`] as an arbitrary default. Taking that default literally would
    /// decode every big-endian `<?xml ... encoding="UTF-16"?>` document as little-endian
    /// mojibake, even though the declaration could only be *read* in the first place by decoding
    /// the prologue big-endian. So whenever the leading bytes say UTF-16 — a BOM, or the exact
    /// `00 3C 00 3F` / `3C 00 3F 00` pattern that no ASCII-compatible encoding can produce —
    /// that byte order wins outright over whatever the declaration claims. See
    /// [`SniffResult::has_conflicting_declaration`] for detecting the contradictory case that
    /// XML 1.0 §4.3.3 makes a fatal error.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_encoding::{sniff_xml_encoding, Encoding};
    ///
    /// let bare = sniff_xml_encoding(b"<root/>");
    /// assert_eq!(bare.best_effort_encoding(), Encoding::Utf8);
    ///
    /// let declared = sniff_xml_encoding(b"<?xml version=\"1.0\" encoding=\"windows-1252\"?><r/>");
    /// assert_eq!(declared.best_effort_encoding(), Encoding::Windows1252);
    ///
    /// // A big-endian document whose declaration says only the endianness-free "UTF-16"
    /// // still decodes big-endian.
    /// let mut big_endian = vec![0xFE, 0xFF];
    /// for unit in "<?xml version=\"1.0\" encoding=\"UTF-16\"?><r/>".encode_utf16() {
    ///     big_endian.extend_from_slice(&unit.to_be_bytes());
    /// }
    /// assert_eq!(
    ///     sniff_xml_encoding(&big_endian).best_effort_encoding(),
    ///     Encoding::Utf16Be
    /// );
    /// ```
    #[must_use]
    pub fn best_effort_encoding(&self) -> Encoding {
        let byte_order_encoding = self
            .byte_order_guess
            .and_then(ByteOrderGuess::supported_encoding);
        // A UTF-16 byte order comes from a BOM or from the Appendix F.1 four-byte pattern, both
        // of which are exact and neither of which any ASCII-compatible encoding can produce. It
        // is therefore authoritative: the declaration -- which was only legible at all because
        // the prologue was decoded as UTF-16 with this very byte order -- may confirm the family
        // but never flip the byte order, let alone name another family.
        if let Some(from_bytes @ (Encoding::Utf16Le | Encoding::Utf16Be)) = byte_order_encoding {
            return from_bytes;
        }
        if let Some(encoding) = self.declared_encoding {
            return encoding;
        }
        if let Some(encoding) = byte_order_encoding {
            return encoding;
        }
        Encoding::Utf8
    }

    /// Whether the declared label contradicts the byte-order evidence in a way XML 1.0 §4.3.3
    /// makes a fatal error ("it is a fatal error for an entity ... to be presented to the XML
    /// processor in an encoding other than that named in the declaration").
    ///
    /// This is `true` only when both signals are present, both are decodable by this crate, and
    /// they disagree — including the endianness-only disagreement that
    /// [`best_effort_encoding`](SniffResult::best_effort_encoding) silently resolves in favour of
    /// the bytes, e.g. a big-endian entity declaring `encoding="UTF-16LE"`. An entity carrying no
    /// declaration, or one whose declaration merely refines the coarse ASCII-compatible guess
    /// (`windows-1252`, `Shift_JIS`, ...), is not a conflict.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_encoding::sniff_xml_encoding;
    ///
    /// // A big-endian BOM with a matching-family, endianness-free label: no conflict.
    /// let mut agreeing = vec![0xFE, 0xFF];
    /// for unit in "<?xml version=\"1.0\" encoding=\"UTF-16\"?><r/>".encode_utf16() {
    ///     agreeing.extend_from_slice(&unit.to_be_bytes());
    /// }
    /// assert!(!sniff_xml_encoding(&agreeing).has_conflicting_declaration());
    ///
    /// // The same bytes explicitly declaring the *other* byte order: a conflict.
    /// let mut disagreeing = vec![0xFE, 0xFF];
    /// for unit in "<?xml version=\"1.0\" encoding=\"UTF-16LE\"?><r/>".encode_utf16() {
    ///     disagreeing.extend_from_slice(&unit.to_be_bytes());
    /// }
    /// assert!(sniff_xml_encoding(&disagreeing).has_conflicting_declaration());
    /// ```
    #[must_use]
    pub fn has_conflicting_declaration(&self) -> bool {
        let (Some(declared), Some(from_bytes)) = (
            self.declared_encoding,
            self.byte_order_guess
                .and_then(ByteOrderGuess::supported_encoding),
        ) else {
            return false;
        };
        match from_bytes {
            // The UTF-16 guesses are exact (a BOM or the Appendix F.1 pattern pins the byte
            // order), so anything else declared -- another byte order, or another family
            // entirely -- contradicts them. The bare label `UTF-16` is exempt: it names the
            // family without committing to a byte order, so it agrees with either.
            Encoding::Utf16Le | Encoding::Utf16Be => {
                if self
                    .declared_label
                    .as_deref()
                    .is_some_and(|label| label.eq_ignore_ascii_case("utf-16"))
                {
                    false
                } else {
                    declared != from_bytes
                }
            }
            // `ByteOrderGuess::Utf8` only means "ASCII-compatible" (see
            // `ByteOrderGuess::supported_encoding`), so a declaration naming any other
            // ASCII-compatible encoding refines rather than contradicts it. Only a UTF-16
            // declaration, which is not ASCII-compatible at all, is a genuine conflict.
            _ => matches!(declared, Encoding::Utf16Le | Encoding::Utf16Be),
        }
    }
}

/// Sniff the encoding of an XML entity from its leading bytes, per
/// [XML 1.0 Appendix F](https://www.w3.org/TR/xml/#sec-guessing).
///
/// See the module-level documentation for the two-stage algorithm this implements, and
/// [`SniffResult`] for how to interpret the result.
///
/// # Examples
///
/// ```
/// use oxixml_encoding::{sniff_xml_encoding, ByteOrderGuess, Encoding};
///
/// // Appendix F pattern: "<?xm" as UTF-16BE with no BOM.
/// let utf16be_prologue: &[u8] = &[0x00, b'<', 0x00, b'?', 0x00, b'x', 0x00, b'm'];
/// let result = sniff_xml_encoding(utf16be_prologue);
/// assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf16Be));
/// assert_eq!(result.bom_len, 0);
///
/// // A UTF-8 BOM plus an explicit (if redundant) encoding declaration.
/// let mut with_bom = vec![0xEF, 0xBB, 0xBF];
/// with_bom.extend_from_slice(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?><doc/>");
/// let result = sniff_xml_encoding(&with_bom);
/// assert_eq!(result.bom_len, 3);
/// assert_eq!(result.declared_encoding, Some(Encoding::Utf8));
/// ```
#[must_use]
pub fn sniff_xml_encoding(bytes: &[u8]) -> SniffResult {
    let (guess, bom_len) = guess_and_bom_len(bytes);
    let window = &bytes[bom_len..];
    let (declared_label, declared_encoding) = match guess {
        // UTF-32 and EBCDIC prologues are out of scope for this crate's ASCII/UTF-16 scanner.
        Some(
            ByteOrderGuess::Utf32Le
            | ByteOrderGuess::Utf32Be
            | ByteOrderGuess::Utf32Order2143
            | ByteOrderGuess::Utf32Order3412
            | ByteOrderGuess::Ebcdic,
        ) => (None, None),
        // No recognizable pattern at all: per Appendix F, fall back to assuming an
        // ASCII-compatible encoding and scan for a declaration anyway.
        Some(ByteOrderGuess::Utf8) | None => scan_declaration(window, ByteOrderGuess::Utf8),
        Some(g @ (ByteOrderGuess::Utf16Le | ByteOrderGuess::Utf16Be)) => {
            scan_declaration(window, g)
        }
    };
    SniffResult {
        byte_order_guess: guess,
        bom_len,
        declared_label,
        declared_encoding,
    }
}

/// The full XML 1.0 Appendix F.1 heuristic table, in the precedence order the spec lists it:
/// the four-byte UCS-4/UTF-16-without-BOM/UTF-8/EBCDIC patterns are checked first (since some of
/// them share their first two bytes with a plain UTF-16 byte-order mark, e.g. `FF FE 00 00`
/// starts with the same two bytes as the `FF FE` UTF-16LE BOM), and only once none of those match
/// does a bare two- or three-byte BOM get considered.
///
/// Returns the guessed family together with the number of leading bytes that are a byte-order
/// mark and should be skipped before decoding (`0` for the no-BOM four-byte patterns, since
/// those four bytes are themselves the start of `<?xm` and must be decoded, not skipped).
fn guess_and_bom_len(bytes: &[u8]) -> (Option<ByteOrderGuess>, usize) {
    if bytes.len() >= 4 {
        let guess = match [bytes[0], bytes[1], bytes[2], bytes[3]] {
            [0x00, 0x00, 0xFE, 0xFF] => Some(ByteOrderGuess::Utf32Be),
            [0xFF, 0xFE, 0x00, 0x00] => Some(ByteOrderGuess::Utf32Le),
            [0x00, 0x00, 0xFF, 0xFE] => Some(ByteOrderGuess::Utf32Order2143),
            [0xFE, 0xFF, 0x00, 0x00] => Some(ByteOrderGuess::Utf32Order3412),
            [0x00, 0x3C, 0x00, 0x3F] => Some(ByteOrderGuess::Utf16Be),
            [0x3C, 0x00, 0x3F, 0x00] => Some(ByteOrderGuess::Utf16Le),
            [0x3C, 0x3F, 0x78, 0x6D] => Some(ByteOrderGuess::Utf8),
            [0x4C, 0x6F, 0xA7, 0x94] => Some(ByteOrderGuess::Ebcdic),
            _ => None,
        };
        if let Some(guess) = guess {
            return (Some(guess), 0);
        }
    }
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        (Some(ByteOrderGuess::Utf8), 3)
    } else if bytes.starts_with(&[0xFE, 0xFF]) {
        (Some(ByteOrderGuess::Utf16Be), 2)
    } else if bytes.starts_with(&[0xFF, 0xFE]) {
        (Some(ByteOrderGuess::Utf16Le), 2)
    } else {
        (None, 0)
    }
}

/// Scan `window` (the entity's bytes with any BOM already stripped) for an `encoding="..."`
/// pseudo-attribute, decoding just enough of it to do so given `guess`'s byte width.
fn scan_declaration(window: &[u8], guess: ByteOrderGuess) -> (Option<String>, Option<Encoding>) {
    let logical_ascii_bytes: Vec<u8> = match guess {
        ByteOrderGuess::Utf16Le => decode_prologue_utf16(window, false),
        ByteOrderGuess::Utf16Be => decode_prologue_utf16(window, true),
        _ => {
            let limit = window.len().min(SCAN_LIMIT_BYTES);
            window[..limit].to_vec()
        }
    };
    parse_encoding_pseudo_attribute(&logical_ascii_bytes)
}

/// Decode a bounded prefix of `window` as UTF-16, lossily (a sniffer must never itself fail on
/// malformed input), producing a plain byte buffer suitable for the same ASCII-oriented scan
/// used for single-byte encodings. This works because the XML declaration itself is required to
/// be pure ASCII, so its UTF-8 re-encoding (which is what `String` gives us) is byte-identical to
/// the ASCII text originally carried one-per-UTF-16-code-unit.
fn decode_prologue_utf16(window: &[u8], big_endian: bool) -> Vec<u8> {
    let limit = window.len().min(SCAN_LIMIT_BYTES * 2);
    let encoding = if big_endian {
        Encoding::Utf16Be
    } else {
        Encoding::Utf16Le
    };
    let mut decoder = encoding.new_decoder();
    let mut text = String::new();
    decoder.decode_to_string_lossy(&window[..limit], &mut text, true);
    text.into_bytes()
}

/// Locate `needle` in `haystack`, if present.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Parse `<?xml ... encoding="LABEL" ... ?>` (or with single quotes) out of `bytes`, which is
/// assumed to already be a plain ASCII-compatible byte sequence (see [`decode_prologue_utf16`]
/// and the direct single-byte path in [`scan_declaration`]).
fn parse_encoding_pseudo_attribute(bytes: &[u8]) -> (Option<String>, Option<Encoding>) {
    if !bytes.starts_with(b"<?xml") {
        return (None, None);
    }
    let declaration_end = find_subslice(bytes, b"?>").unwrap_or(bytes.len());
    let declaration = &bytes[..declaration_end];

    let Some(keyword_at) = find_subslice(declaration, b"encoding") else {
        return (None, None);
    };
    let mut i = keyword_at + b"encoding".len();

    i += skip_ascii_whitespace(&declaration[i..]);
    if declaration.get(i) != Some(&b'=') {
        return (None, None);
    }
    i += 1;
    i += skip_ascii_whitespace(&declaration[i..]);

    let Some(&quote) = declaration.get(i) else {
        return (None, None);
    };
    if quote != b'"' && quote != b'\'' {
        return (None, None);
    }
    i += 1;
    let value_start = i;
    while declaration.get(i).is_some_and(|&b| b != quote) {
        i += 1;
    }
    if i >= declaration.len() {
        // Unterminated quote: the declaration was cut off before we could find the closing one.
        return (None, None);
    }
    let raw_label = &declaration[value_start..i];

    // `EncName ::= [A-Za-z] ([A-Za-z0-9._] | '-')*` (XML 1.0 §4.3.3): reject anything that
    // couldn't possibly be a valid encoding name rather than propagate garbage.
    if raw_label.is_empty()
        || !raw_label[0].is_ascii_alphabetic()
        || !raw_label
            .iter()
            .all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return (None, None);
    }

    let Ok(label) = core::str::from_utf8(raw_label) else {
        return (None, None);
    };
    let label = label.to_string();
    let encoding = Encoding::for_label(&label);
    (Some(label), encoding)
}

fn skip_ascii_whitespace(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_bom_no_declaration_defaults_to_utf8() {
        let result = sniff_xml_encoding(b"<root/>");
        assert_eq!(result.byte_order_guess, None);
        assert_eq!(result.bom_len, 0);
        assert_eq!(result.declared_label, None);
        assert_eq!(result.best_effort_encoding(), Encoding::Utf8);
    }

    #[test]
    fn utf8_bom_alone() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(b"<root/>");
        let result = sniff_xml_encoding(&bytes);
        assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf8));
        assert_eq!(result.bom_len, 3);
        assert_eq!(result.declared_label, None);
        assert_eq!(result.best_effort_encoding(), Encoding::Utf8);
    }

    #[test]
    fn utf8_bom_with_redundant_declaration() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?><root/>");
        let result = sniff_xml_encoding(&bytes);
        assert_eq!(result.bom_len, 3);
        assert_eq!(result.declared_label.as_deref(), Some("UTF-8"));
        assert_eq!(result.declared_encoding, Some(Encoding::Utf8));
    }

    #[test]
    fn ascii_prologue_with_windows1252_declaration() {
        let xml = b"<?xml version=\"1.0\" encoding=\"windows-1252\"?><root/>";
        let result = sniff_xml_encoding(xml);
        assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf8));
        assert_eq!(result.declared_label.as_deref(), Some("windows-1252"));
        assert_eq!(result.declared_encoding, Some(Encoding::Windows1252));
        assert_eq!(result.best_effort_encoding(), Encoding::Windows1252);
    }

    #[test]
    fn single_quoted_declaration_is_recognized() {
        let xml = b"<?xml version='1.0' encoding='ISO-8859-1'?><root/>";
        let result = sniff_xml_encoding(xml);
        assert_eq!(result.declared_label.as_deref(), Some("ISO-8859-1"));
        assert_eq!(result.declared_encoding, Some(Encoding::Latin1));
    }

    #[test]
    fn utf16le_bom_with_declaration() {
        let mut bytes = vec![0xFF, 0xFE];
        let text = "<?xml version=\"1.0\" encoding=\"UTF-16\"?><root/>";
        for unit in text.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        let result = sniff_xml_encoding(&bytes);
        assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf16Le));
        assert_eq!(result.bom_len, 2);
        assert_eq!(result.declared_label.as_deref(), Some("UTF-16"));
        assert_eq!(result.declared_encoding, Some(Encoding::Utf16Le));
    }

    #[test]
    fn utf16be_no_bom_pattern_detected_from_first_four_bytes() {
        let text = "<?xml version=\"1.0\" encoding=\"UTF-16BE\"?><root/>";
        let mut bytes = Vec::new();
        for unit in text.encode_utf16() {
            bytes.extend_from_slice(&unit.to_be_bytes());
        }
        let result = sniff_xml_encoding(&bytes);
        assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf16Be));
        assert_eq!(result.bom_len, 0);
        assert_eq!(result.declared_label.as_deref(), Some("UTF-16BE"));
        assert_eq!(result.declared_encoding, Some(Encoding::Utf16Be));
    }

    #[test]
    fn utf16le_no_bom_pattern_detected_from_first_four_bytes() {
        let text = "<?xml version=\"1.0\"?><root/>";
        let mut bytes = Vec::new();
        for unit in text.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        let result = sniff_xml_encoding(&bytes);
        assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf16Le));
        assert_eq!(result.declared_label, None);
        assert_eq!(result.best_effort_encoding(), Encoding::Utf16Le);
    }

    #[test]
    fn ascii_compatible_four_byte_pattern_3c_3f_78_6d() {
        let result = sniff_xml_encoding(b"<?xml version=\"1.0\"?><root/>");
        assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf8));
    }

    #[test]
    fn utf32_be_pattern_is_recognized_but_unsupported() {
        let bytes = [0x00, 0x00, 0xFE, 0xFF, b'<', b'?', b'x', b'm', b'l'];
        let result = sniff_xml_encoding(&bytes);
        assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf32Be));
        assert_eq!(
            result
                .byte_order_guess
                .and_then(ByteOrderGuess::supported_encoding),
            None
        );
        assert_eq!(result.declared_label, None);
        // The overall best-effort fallback is still well-defined (UTF-8), even though it will
        // almost certainly be wrong for genuine UTF-32 content; callers that care should check
        // `byte_order_guess` explicitly and reject/report unsupported families themselves.
        assert_eq!(result.best_effort_encoding(), Encoding::Utf8);
    }

    #[test]
    fn utf32_le_pattern_is_recognized() {
        let bytes = [0xFF, 0xFE, 0x00, 0x00];
        let result = sniff_xml_encoding(&bytes);
        assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf32Le));
    }

    #[test]
    fn utf32_unusual_order_patterns_are_recognized() {
        assert_eq!(
            sniff_xml_encoding(&[0x00, 0x00, 0xFF, 0xFE]).byte_order_guess,
            Some(ByteOrderGuess::Utf32Order2143)
        );
        assert_eq!(
            sniff_xml_encoding(&[0xFE, 0xFF, 0x00, 0x00]).byte_order_guess,
            Some(ByteOrderGuess::Utf32Order3412)
        );
    }

    #[test]
    fn ebcdic_pattern_is_recognized_but_unsupported() {
        let bytes = [0x4C, 0x6F, 0xA7, 0x94];
        let result = sniff_xml_encoding(&bytes);
        assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Ebcdic));
        assert_eq!(result.declared_label, None);
    }

    #[test]
    fn declaration_without_encoding_pseudo_attribute() {
        let result = sniff_xml_encoding(b"<?xml version=\"1.0\"?><root/>");
        assert_eq!(result.declared_label, None);
        assert_eq!(result.declared_encoding, None);
    }

    #[test]
    fn unrecognized_declared_label_is_still_captured() {
        // "windows-874" (Thai) is outside this crate's current scope.
        let xml = b"<?xml version=\"1.0\" encoding=\"windows-874\"?><root/>";
        let result = sniff_xml_encoding(xml);
        assert_eq!(result.declared_label.as_deref(), Some("windows-874"));
        assert_eq!(result.declared_encoding, None);
        // Falls back to the byte-order guess (ASCII-compatible => UTF-8) since the declared
        // label, though present, was not one this crate can decode.
        assert_eq!(result.best_effort_encoding(), Encoding::Utf8);
    }

    #[test]
    fn empty_input_does_not_panic() {
        let result = sniff_xml_encoding(b"");
        assert_eq!(result.byte_order_guess, None);
        assert_eq!(result.declared_label, None);
        assert_eq!(result.best_effort_encoding(), Encoding::Utf8);
    }

    #[test]
    fn truncated_declaration_without_closing_quote_does_not_panic() {
        let result = sniff_xml_encoding(b"<?xml encoding=\"UTF-8");
        assert_eq!(result.declared_label, None);
    }

    #[test]
    fn content_not_starting_with_xml_declaration_has_no_declared_label() {
        let result = sniff_xml_encoding(b"   <!-- comment --><root/>");
        assert_eq!(result.declared_label, None);
    }

    /// Appendix F sniffing needs no encoding-specific logic beyond [`Encoding::for_label`]: an
    /// `<?xml ... encoding="LABEL" ?>` declaration is required (XML 1.0 §2.8/§4.3.3) to be pure
    /// ASCII, so it is byte-identical whether the rest of the entity is windows-1250, Shift_JIS,
    /// EUC-JP, ISO-2022-JP, GBK, gb18030, Big5, or EUC-KR -- all of these pass the same `<?xm` ->
    /// ASCII-compatible Appendix F.1 byte-pattern check as UTF-8 does. This test is the evidence
    /// for that claim, one encoding from each newly-added family.
    #[test]
    fn appendix_f_recognizes_every_newly_added_encoding_family_by_label() {
        for (label, expected) in [
            ("windows-1250", Encoding::Windows1250),
            ("KOI8-R", Encoding::Koi8R),
            ("ISO-8859-9", Encoding::Iso8859_9),
            ("Shift_JIS", Encoding::ShiftJis),
            ("EUC-JP", Encoding::EucJp),
            ("ISO-2022-JP", Encoding::Iso2022Jp),
            ("GBK", Encoding::Gbk),
            ("gb18030", Encoding::Gb18030),
            ("Big5", Encoding::Big5),
            ("EUC-KR", Encoding::EucKr),
        ] {
            let xml = format!("<?xml version=\"1.0\" encoding=\"{label}\"?><root/>");
            let result = sniff_xml_encoding(xml.as_bytes());
            assert_eq!(
                result.byte_order_guess,
                Some(ByteOrderGuess::Utf8),
                "label={label}"
            );
            assert_eq!(
                result.declared_label.as_deref(),
                Some(label),
                "label={label}"
            );
            assert_eq!(result.declared_encoding, Some(expected), "label={label}");
            assert_eq!(result.best_effort_encoding(), expected, "label={label}");
        }
    }

    /// Encodes `text` as UTF-16 with the given byte order, optionally prefixed by a BOM.
    fn utf16_document(text: &str, big_endian: bool, bom: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        if bom {
            bytes.extend_from_slice(if big_endian {
                &[0xFE, 0xFF]
            } else {
                &[0xFF, 0xFE]
            });
        }
        for unit in text.encode_utf16() {
            if big_endian {
                bytes.extend_from_slice(&unit.to_be_bytes());
            } else {
                bytes.extend_from_slice(&unit.to_le_bytes());
            }
        }
        bytes
    }

    /// Regression: the label `UTF-16` carries no endianness (XML 1.0 Appendix F / RFC 2781), so
    /// it must never override the byte order established by the BOM or by the Appendix F.1
    /// four-byte pattern. `Encoding::for_label("UTF-16")` resolves to `Utf16Le` (WHATWG's
    /// arbitrary default), which used to win outright and decode every big-endian document as
    /// byte-swapped mojibake.
    #[test]
    fn bare_utf16_label_does_not_override_big_endian_byte_order() {
        let text = "<?xml version=\"1.0\" encoding=\"UTF-16\"?><root>hi</root>";
        for bom in [true, false] {
            let bytes = utf16_document(text, true, bom);
            let result = sniff_xml_encoding(&bytes);
            assert_eq!(
                result.byte_order_guess,
                Some(ByteOrderGuess::Utf16Be),
                "bom={bom}"
            );
            assert_eq!(
                result.declared_label.as_deref(),
                Some("UTF-16"),
                "bom={bom}"
            );
            assert_eq!(
                result.best_effort_encoding(),
                Encoding::Utf16Be,
                "bom={bom}"
            );
            assert!(!result.has_conflicting_declaration(), "bom={bom}");

            // And the whole document really does decode back to its original text.
            let mut decoder = result.best_effort_encoding().new_decoder();
            let mut decoded = String::new();
            decoder.decode_to_string_lossy(&bytes[result.bom_len..], &mut decoded, true);
            assert_eq!(decoded, text, "bom={bom}");
        }
    }

    /// The little-endian mirror of the above: the same bare label must keep little-endian
    /// little-endian.
    #[test]
    fn bare_utf16_label_keeps_little_endian_byte_order() {
        let text = "<?xml version=\"1.0\" encoding=\"UTF-16\"?><root>hi</root>";
        for bom in [true, false] {
            let bytes = utf16_document(text, false, bom);
            let result = sniff_xml_encoding(&bytes);
            assert_eq!(
                result.byte_order_guess,
                Some(ByteOrderGuess::Utf16Le),
                "bom={bom}"
            );
            assert_eq!(
                result.best_effort_encoding(),
                Encoding::Utf16Le,
                "bom={bom}"
            );
            assert!(!result.has_conflicting_declaration(), "bom={bom}");
        }
    }

    /// An *explicit* endianness in the declaration that contradicts the bytes is a fatal error
    /// per XML 1.0 §4.3.3; best-effort decoding still follows the bytes (the declaration could
    /// only be read at all because the bytes were decoded that way), and the contradiction is
    /// reported through `has_conflicting_declaration`.
    #[test]
    fn explicitly_wrong_endianness_declaration_follows_the_bytes_and_is_flagged() {
        let bytes = utf16_document(
            "<?xml version=\"1.0\" encoding=\"UTF-16LE\"?><root/>",
            true,
            true,
        );
        let result = sniff_xml_encoding(&bytes);
        assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf16Be));
        assert_eq!(result.declared_encoding, Some(Encoding::Utf16Le));
        assert_eq!(result.best_effort_encoding(), Encoding::Utf16Be);
        assert!(result.has_conflicting_declaration());
    }

    /// A UTF-16 byte order plus a declaration naming an entirely different family is likewise a
    /// conflict; the (non-ASCII-compatible) bytes still win, since a single-byte decoder cannot
    /// possibly be right about a document whose prologue was only legible as UTF-16.
    #[test]
    fn utf16_bytes_with_single_byte_declaration_follow_the_bytes() {
        let bytes = utf16_document(
            "<?xml version=\"1.0\" encoding=\"windows-1252\"?><root/>",
            true,
            true,
        );
        let result = sniff_xml_encoding(&bytes);
        assert_eq!(result.declared_encoding, Some(Encoding::Windows1252));
        assert_eq!(result.best_effort_encoding(), Encoding::Utf16Be);
        assert!(result.has_conflicting_declaration());
    }

    /// A declaration that merely refines the coarse "ASCII-compatible" guess is not a conflict,
    /// but claiming UTF-16 from ASCII-compatible bytes is.
    #[test]
    fn ascii_compatible_declarations_conflict_only_with_utf16() {
        let refined = sniff_xml_encoding(b"<?xml version=\"1.0\" encoding=\"Shift_JIS\"?><r/>");
        assert!(!refined.has_conflicting_declaration());
        assert_eq!(refined.best_effort_encoding(), Encoding::ShiftJis);

        let contradictory = sniff_xml_encoding(b"<?xml version=\"1.0\" encoding=\"UTF-16\"?><r/>");
        assert!(contradictory.has_conflicting_declaration());

        let no_declaration = sniff_xml_encoding(b"<root/>");
        assert!(!no_declaration.has_conflicting_declaration());
    }

    #[test]
    fn every_supported_encodings_name_is_recognized_when_declared() {
        for &encoding in Encoding::ALL {
            let xml = format!(
                "<?xml version=\"1.0\" encoding=\"{}\"?><root/>",
                encoding.name()
            );
            let result = sniff_xml_encoding(xml.as_bytes());
            assert_eq!(result.declared_encoding, Some(encoding), "{encoding}");
        }
    }
}

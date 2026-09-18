//! Holds the incremental state of the pull [`Reader`](crate::reader::Reader).
//!
//! The [`ReaderState`] is independent of the way bytes are obtained: the various
//! [`XmlSource`](crate::reader::XmlSource) implementations feed raw slices into
//! it and it turns them into [`Event`]s, tracking open elements, byte offsets and
//! the (optional) detected encoding along the way.

use std::fmt::Debug;

use crate::encoding::Decoder;
use crate::errors::{Error, IllFormedError, Result, WellFormedError};
use crate::events::{BytesCData, BytesDecl, BytesEnd, BytesPI, BytesStart, BytesText, Event};
use crate::parser::{DtdParser, Parser, PiParser};
#[cfg(feature = "encoding")]
use crate::reader::EncodingRef;
use crate::reader::{BangType, Config, ParseState};
use crate::utils::{Bytes, is_whitespace, name_len};
use crate::wellformed::WellFormed;
#[cfg(feature = "encoding")]
use oxixml_encoding::Encoding;

/// A struct that holds a current reader state and a parser configuration.
/// It is independent on a way of reading data: the reader feed data into it and
/// get back produced [`Event`]s.
#[derive(Clone)]
pub(super) struct ReaderState {
    /// Number of bytes read from the source of data since the reader was created
    pub offset: u64,
    /// A snapshot of an `offset` of the last error returned. It can be less than
    /// `offset`, because some errors conveniently report at earlier position,
    /// and changing `offset` is not possible, because `Error::IllFormed` errors
    /// are recoverable.
    pub last_error_offset: u64,
    /// Defines how to process next byte
    pub state: ParseState,
    /// User-defined settings that affect parsing
    pub config: Config,
    /// All currently Started elements which didn't have a matching
    /// End element yet.
    ///
    /// For an XML
    ///
    /// ```xml
    /// <root><one/><inner attr="value">|<tag></inner></root>
    /// ```
    /// when cursor at the `|` position buffer contains:
    ///
    /// ```text
    /// rootinner
    /// ^   ^
    /// ```
    ///
    /// The `^` symbols shows which positions stored in the [`Self::opened_starts`]
    /// (0 and 4 in that case).
    opened_buffer: Vec<u8>,
    /// Opened name start indexes into [`Self::opened_buffer`]. See documentation
    /// for that field for details
    opened_starts: Vec<usize>,

    /// Incremental state of the optional well-formedness checks, driven only when
    /// [`Config::check_well_formedness`](crate::reader::Config::check_well_formedness)
    /// is enabled.
    well_formed: WellFormed,
    /// Whether the end-of-input well-formedness constraints have already been
    /// applied once. See [`Self::check_well_formedness`] for why they must not be
    /// applied twice.
    eof_checked: bool,

    #[cfg(feature = "encoding")]
    /// Reference to the encoding used to read an XML
    pub encoding: EncodingRef,
}

impl ReaderState {
    /// Trims end whitespaces from `bytes`, if required, and returns a text event.
    ///
    /// # Parameters
    /// - `bytes`: data from the start of stream to the first `<` or from `>` to `<`
    pub fn emit_text<'b>(&mut self, bytes: &'b [u8]) -> BytesText<'b> {
        let mut content = bytes;

        if self.config.trim_text_end {
            // Skip the ending '<'
            let len = bytes
                .iter()
                .rposition(|&b| !is_whitespace(b))
                .map_or(0, |p| p + 1);
            content = &bytes[..len];
        }
        BytesText::wrap(content, self.decoder())
    }

    /// Returns `Comment`, `CData` or `DocType` event.
    ///
    /// `buf` contains data between `<` and `>`:
    /// - CDATA: `<![CDATA[...]]`
    /// - Comment: `<!--...--`
    /// - Doctype (uppercase): `<!D...`
    /// - Doctype (lowercase): `<!d...`
    pub fn emit_bang<'b>(&mut self, bang_type: BangType, buf: &'b [u8]) -> Result<Event<'b>> {
        debug_assert!(
            buf.starts_with(b"<!"),
            "CDATA, comment or DOCTYPE must start from '<!':\n{:?}",
            Bytes(buf)
        );
        debug_assert!(
            buf.ends_with(b">"),
            "CDATA, comment or DOCTYPE must end with '>':\n{:?}",
            Bytes(buf)
        );

        let uncased_starts_with = |string: &[u8], prefix: &[u8]| {
            string.len() >= prefix.len() && string[..prefix.len()].eq_ignore_ascii_case(prefix)
        };

        let len = buf.len();
        match bang_type {
            BangType::Comment if buf.starts_with(b"<!--") => {
                debug_assert!(
                    buf.ends_with(b"-->"),
                    "comment must end with '-->':\n{:?}",
                    Bytes(buf)
                );
                if self.config.check_comments {
                    // search if '--' not in comments
                    let mut haystack = &buf[4..len - 3];
                    let mut off = 0;
                    while let Some(p) = oxixml_support::bytes::memchr(b'-', haystack) {
                        off += p + 1;
                        // if next byte after `-` is also `-`, return an error
                        if buf[4 + off] == b'-' {
                            // Explanation of the magic:
                            //
                            // - `self.offset` just after `>`,
                            // - `buf` contains `<!-- con--tent --`
                            // - `p` is counted from byte after `<!--`
                            //
                            // <!-- con--tent -->:
                            // ~~~~~~~~~~~~~~~~~~: - buf
                            //  :  ===========   : - zone of search (possible values of `p`)
                            //  :  |---p         : - p is counted from | (| is 0)
                            //  :  :   :         ^ - self.offset
                            //  ^  :   :           - self.offset - len
                            //     ^   :           - self.offset - len + 4
                            //         ^           - self.offset - len + 4 + p
                            self.last_error_offset = self.offset - len as u64 + 4 + p as u64;
                            return Err(Error::IllFormed(IllFormedError::DoubleHyphenInComment));
                        }
                        // Continue search after single `-` (+1 to skip it)
                        haystack = &haystack[p + 1..];
                    }
                }
                Ok(Event::Comment(BytesText::wrap(
                    // Cut of `<!--` and `-->` from start and end
                    &buf[4..len - 3],
                    self.decoder(),
                )))
            }
            // XML requires uppercase only:
            // https://www.w3.org/TR/xml11/#sec-cdata-sect
            // Even HTML5 required uppercase only:
            // https://html.spec.whatwg.org/multipage/parsing.html#markup-declaration-open-state
            BangType::CData if buf.starts_with(b"<![CDATA[") => {
                debug_assert!(
                    buf.ends_with(b"]]>"),
                    "CDATA must end with ']]>':\n{:?}",
                    Bytes(buf)
                );
                Ok(Event::CData(BytesCData::wrap(
                    // Cut of `<![CDATA[` and `]]>` from start and end
                    &buf[9..len - 3],
                    self.decoder(),
                )))
            }
            // XML requires uppercase only, but we will check that on validation stage:
            // https://www.w3.org/TR/xml11/#sec-prolog-dtd
            // HTML5 allows mixed case for doctype declarations:
            // https://html.spec.whatwg.org/multipage/parsing.html#markup-declaration-open-state
            BangType::DocType(DtdParser::Finished) if uncased_starts_with(buf, b"<!DOCTYPE") => {
                // XML requires the keyword to be upper-case; the permissive
                // default path also accepts the HTML spelling.
                if self.config.check_well_formedness && !buf.starts_with(b"<!DOCTYPE") {
                    self.last_error_offset = self.offset - len as u64;
                    return Err(Error::NotWellFormed(WellFormedError::InvalidDoctype(
                        "the `DOCTYPE` keyword must be upper-case".to_owned(),
                    )));
                }
                match buf[9..len - 1].iter().position(|&b| !is_whitespace(b)) {
                    Some(start) => Ok(Event::DocType(BytesText::wrap(
                        // Cut of `<!DOCTYPE` and any number of spaces from start and `>` from the end
                        &buf[9 + start..len - 1],
                        self.decoder(),
                    ))),
                    None => {
                        // Because we here, we at least read `<!DOCTYPE>` and offset after `>`.
                        // We want report error at place where name is expected - this is just
                        // before `>`
                        self.last_error_offset = self.offset - 1;
                        Err(Error::IllFormed(IllFormedError::MissingDoctypeName))
                    }
                }
            }
            _ => {
                // <!....>
                // ~~~~~~~- `buf` contains that and `self.offset` is after `>`.
                // ^------- We report error at that position, so we need to subtract buf len
                self.last_error_offset = self.offset - len as u64;
                Err(Error::Syntax(bang_type.to_err()))
            }
        }
    }

    /// Wraps content of `buf` into the [`Event::End`] event. Does the check that
    /// end name matches the last opened start name if `self.config.check_end_names` is set.
    ///
    /// `buf` contains data between `<` and up to, including, `>`, for example `</tag>`.
    pub fn emit_end<'b>(&mut self, buf: &'b [u8]) -> Result<Event<'b>> {
        debug_assert!(
            buf.starts_with(b"</"),
            "end tag must start from '</':\n{:?}",
            Bytes(buf)
        );
        debug_assert!(
            buf.ends_with(b">"),
            "end tag must end with '>':\n{:?}",
            Bytes(buf)
        );

        // Strip the `</` and `>` characters. `content` contains data between `</` and `>`
        let content = &buf[2..buf.len() - 1];
        // XML standard permits whitespaces after the markup name in closing tags.
        // Let's strip them from the buffer before comparing tag names.
        let name = if self.config.trim_markup_names_in_closing_tags {
            if let Some(pos_end_name) = content.iter().rposition(|&b| !is_whitespace(b)) {
                &content[..pos_end_name + 1]
            } else {
                content
            }
        } else {
            content
        };

        let decoder = self.decoder();

        // Get the index in self.opened_buffer of the name of the last opened tag
        match self.opened_starts.pop() {
            Some(start) => {
                if self.config.check_end_names {
                    let expected = &self.opened_buffer[start..];
                    if name != expected {
                        let expected = decoder.decode(expected).unwrap_or_default().into_owned();
                        // #513: In order to allow error recovery we should drop content of the buffer
                        self.opened_buffer.truncate(start);

                        // Report error at start of the end tag at `<` character
                        self.last_error_offset = self.offset - buf.len() as u64;
                        return Err(Error::IllFormed(IllFormedError::MismatchedEndTag {
                            expected,
                            found: decoder.decode(name).unwrap_or_default().into_owned(),
                        }));
                    }
                }

                self.opened_buffer.truncate(start);
            }
            None => {
                if !self.config.allow_unmatched_ends {
                    // Report error at start of the end tag at `<` character
                    self.last_error_offset = self.offset - buf.len() as u64;
                    return Err(Error::IllFormed(IllFormedError::UnmatchedEndTag(
                        decoder.decode(name).unwrap_or_default().into_owned(),
                    )));
                }
            }
        }

        Ok(Event::End(BytesEnd::wrap(name.into())))
    }

    /// `buf` contains data between `<` and `>` and the first byte is `?`.
    /// `self.offset` already after the `>`
    ///
    /// Returns `Decl` or `PI` event
    pub fn emit_question_mark<'b>(&mut self, buf: &'b [u8]) -> Result<Event<'b>> {
        debug_assert!(
            buf.starts_with(b"<?"),
            "processing instruction or XML declaration must start from '<?':\n{:?}",
            Bytes(buf)
        );
        debug_assert!(
            buf.ends_with(b"?>"),
            "processing instruction or XML declaration must end with '?>':\n{:?}",
            Bytes(buf)
        );

        let len = buf.len();
        // We accept at least <??>
        //                    ~~~~ - len = 4
        if len > 3 {
            // Cut of `<?` and `?>` from start and end
            let content = &buf[2..len - 2];
            let len = content.len();

            if content.starts_with(b"xml") && (len == 3 || is_whitespace(content[3])) {
                let event = BytesDecl::from_start(BytesStart::wrap(content, 3, self.decoder()));

                // Try getting encoding from the declaration event
                #[cfg(feature = "encoding")]
                if self.encoding.can_be_refined() {
                    if let Some(declared) = event.encoder() {
                        self.encoding = EncodingRef::XmlDetected(Self::resolve_declared_encoding(
                            self.encoding,
                            declared,
                        ));
                    }
                }

                Ok(Event::Decl(event))
            } else {
                Ok(Event::PI(BytesPI::wrap(
                    content,
                    name_len(content),
                    self.decoder(),
                )))
            }
        } else {
            // <?...?>
            // ~~~~~~~- `buf` contains that and `self.offset` is after `>`.
            // ^------- We report error at that position, so we need to subtract buf len
            self.last_error_offset = self.offset - len as u64;
            Err(Error::Syntax(PiParser(false).eof_error(buf)))
        }
    }

    /// Resolves what a just-parsed `encoding="..."` declaration should
    /// actually refine the reader's encoding to, given what was already
    /// established (`current`) from a byte-order mark or the XML 1.0
    /// Appendix F.1 four-byte pattern.
    ///
    /// A BOM or that four-byte pattern is an exact, authoritative signal of
    /// the UTF-16 byte order actually used to decode the input far enough to
    /// make the declaration itself legible in the first place. The bare label
    /// `UTF-16` inside it carries no endianness of its own (XML 1.0 Appendix F,
    /// RFC 2781): [`BytesDecl::encoder`](crate::events::BytesDecl::encoder)
    /// resolves it to [`Encoding::Utf16Le`] as an arbitrary default, mirroring
    /// the WHATWG ["get an encoding"] algorithm. Taking that default literally
    /// here would silently flip a big-endian document to little-endian
    /// mojibake, so whenever the bytes already pinned a UTF-16 byte order, it
    /// wins over whatever the declaration names — it can be *confirmed* by an
    /// agreeing declaration, but never *overridden* by one, exactly as
    /// [`oxixml_encoding::SniffResult::best_effort_encoding`] resolves the
    /// same ambiguity for the standalone sniffer.
    ///
    /// ["get an encoding"]: https://encoding.spec.whatwg.org/#concept-encoding-get
    /// [`oxixml_encoding::SniffResult::best_effort_encoding`]: oxixml_encoding::SniffResult::best_effort_encoding
    #[cfg(feature = "encoding")]
    fn resolve_declared_encoding(current: EncodingRef, declared: Encoding) -> Encoding {
        match (current, declared) {
            (
                EncodingRef::BomDetected(from_bytes @ (Encoding::Utf16Le | Encoding::Utf16Be)),
                Encoding::Utf16Le | Encoding::Utf16Be,
            ) => from_bytes,
            (_, declared) => declared,
        }
    }

    /// Converts content of a tag to a `Start` or an `Empty` event
    ///
    /// # Parameters
    /// - `content`: Content of a tag between `<` and `>`
    pub fn emit_start<'b>(&mut self, content: &'b [u8]) -> Event<'b> {
        debug_assert!(
            content.starts_with(b"<"),
            "start or empty tag must start from '<':\n{:?}",
            Bytes(content)
        );
        debug_assert!(
            content.ends_with(b">"),
            "start or empty tag must end with '>':\n{:?}",
            Bytes(content)
        );

        // strip `<`
        let content = &content[1..];
        if let Some(content) = content.strip_suffix(b"/>") {
            // This is self-closed tag `<something/>`
            let event = BytesStart::wrap(content, name_len(content), self.decoder());

            if self.config.expand_empty_elements {
                self.state = ParseState::InsideEmpty;
                self.opened_starts.push(self.opened_buffer.len());
                self.opened_buffer.extend(event.name().as_ref());
                Event::Start(event)
            } else {
                Event::Empty(event)
            }
        } else {
            // strip `>`
            let content = &content[..content.len() - 1];
            let event = BytesStart::wrap(content, name_len(content), self.decoder());

            // #514: Always store names event when .check_end_names == false,
            // because checks can be temporary disabled and when they would be
            // enabled, we should have that information
            self.opened_starts.push(self.opened_buffer.len());
            self.opened_buffer.extend(event.name().as_ref());
            Event::Start(event)
        }
    }

    #[inline]
    pub fn close_expanded_empty(&mut self) -> BytesEnd<'static> {
        self.state = ParseState::InsideText;
        let start = self
            .opened_starts
            .pop()
            .expect("`InsideEmpty` state is only entered after pushing an opened start");
        let name = self.opened_buffer.split_off(start);
        BytesEnd::wrap(name.into())
    }

    /// Returns the replacement text of an internal general entity declared by
    /// the document type declaration of the document being read.
    #[inline]
    pub fn declared_entity(&self, name: &str) -> Option<&str> {
        self.well_formed.declared_entity(name)
    }

    /// Switches the well-formedness checker to namespace-aware mode.
    #[inline]
    pub fn enable_namespace_constraints(&mut self) {
        self.well_formed.set_namespace_aware();
    }

    /// Applies the optional, conformance-grade well-formedness checks to `event`.
    ///
    /// See [`Config::check_well_formedness`](crate::reader::Config::check_well_formedness)
    /// for the list of constraints that are enforced.
    ///
    /// # End of input is checked exactly once
    ///
    /// [`Error::IllFormed`] is a *recoverable* error in this crate: the reader deliberately
    /// leaves its state untouched so that a caller may keep reading (quick-xml issue #513), and
    /// several callers in this workspace do. End of input, however, is a fixed point —
    /// [`ParseState::Done`] answers [`Event::Eof`] to every subsequent read — so an error
    /// synthesized here would be synthesized again, identically, on every further read. Before
    /// the guard below, `<a>` with these checks enabled produced `MissingEndTag` for ever: a
    /// caller looping until `Eof` never terminated, and one accumulating a diagnostic per error
    /// (as a linter does) allocated without bound until the process was killed. The
    /// end-of-input constraints therefore run once and are then satisfied by construction, so
    /// that reading past them always reaches `Eof`.
    pub fn check_well_formedness(&mut self, event: &Event<'_>) -> Result<()> {
        let decoder = self.decoder();
        if matches!(event, Event::Eof) {
            if self.eof_checked {
                return Ok(());
            }
            self.eof_checked = true;
            // An unclosed element is reported through the pre-existing error type so
            // that the diagnostics stay familiar to `quick-xml` users. `opened_buffer` holds
            // every still-open element's name run together (see its own documentation), so only
            // the *innermost* one — the last start index — slices to the end of the buffer;
            // taking the first index reported `<a><b>` as a missing `</ab>`.
            if let Some(&start) = self.opened_starts.last() {
                let name = decoder
                    .decode(&self.opened_buffer[start..])
                    .unwrap_or_default()
                    .into_owned();
                self.last_error_offset = self.offset;
                return Err(Error::IllFormed(IllFormedError::MissingEndTag(name)));
            }
        }
        match self.well_formed.check(event, decoder) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.last_error_offset = self.offset;
                Err(error)
            }
        }
    }

    /// Get the decoder, used to decode bytes, read by this reader, to the strings.
    ///
    /// If [`encoding`] feature is enabled, the used encoding may change after
    /// parsing the XML declaration, otherwise encoding is fixed to UTF-8.
    ///
    /// If [`encoding`] feature is enabled and no encoding is specified in declaration,
    /// defaults to UTF-8.
    ///
    /// [`encoding`]: ../../index.html#encoding
    pub const fn decoder(&self) -> Decoder {
        Decoder {
            #[cfg(feature = "encoding")]
            encoding: self.encoding.encoding(),
        }
    }
}

impl Default for ReaderState {
    fn default() -> Self {
        Self {
            offset: 0,
            last_error_offset: 0,
            state: ParseState::Init,
            config: Config::default(),
            opened_buffer: Vec::new(),
            opened_starts: Vec::new(),
            well_formed: WellFormed::default(),
            eof_checked: false,

            #[cfg(feature = "encoding")]
            encoding: EncodingRef::Implicit(oxixml_encoding::Encoding::Utf8),
        }
    }
}

impl Debug for ReaderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("ReaderState");

        d.field("offset", &self.offset);
        d.field("last_error_offset", &self.last_error_offset);
        d.field("state", &self.state);
        d.field("config", &self.config);
        d.field("opened_buffer", &Bytes(&self.opened_buffer));
        d.field("opened_starts", &self.opened_starts);

        #[cfg(feature = "encoding")]
        d.field("encoding", &self.encoding);

        d.finish()
    }
}

#[cfg(all(test, feature = "encoding"))]
mod tests {
    use super::*;
    use crate::encoding::detect_encoding;

    /// White-box companion to
    /// `encoding::tests::bare_utf16_label_does_not_override_the_boms_byte_order`,
    /// covering the no-BOM XML 1.0 Appendix F.1 four-byte pattern (`00 3C 00 3F`
    /// / `3C 00 3F 00`) instead of an actual byte-order mark.
    ///
    /// That pattern cannot be exercised end-to-end through the public
    /// `Reader` API the way a BOM can: a BOM's bytes are *stripped* before
    /// tokenizing resumes, so an otherwise-plain-ASCII body can follow it, but
    /// the four-byte pattern's bytes are *not* stripped -- they are the first
    /// two characters (`<?`) of the declaration itself -- so anything past
    /// them would have to genuinely continue as 2-bytes-per-character UTF-16,
    /// which hits the same pre-existing, unrelated tokenizer limitation that
    /// makes `DecodingReader` mandatory for non-ASCII-compatible input (see
    /// its docs), masking the very refinement step this test targets before
    /// it ever runs. Driving `ReaderState::emit_question_mark` directly
    /// exercises that exact step, since both detection routes funnel into the
    /// identical `EncodingRef::BomDetected(Encoding::Utf16Be | Utf16Le)` state.
    #[test]
    fn pattern_detected_utf16_byte_order_survives_bare_declared_label() {
        for big_endian in [true, false] {
            // The exact bytes `ParseState::Init` would have seen at the start
            // of the (BOM-less) input.
            let pattern: &[u8] = if big_endian {
                b"\x00<\x00?"
            } else {
                b"<\x00?\x00"
            };
            let detected =
                detect_encoding(pattern).expect("Appendix F.1 four-byte pattern recognized");
            assert_eq!(detected.bom_len(), 0, "the pattern bytes are not a BOM");
            let expected = if big_endian {
                Encoding::Utf16Be
            } else {
                Encoding::Utf16Le
            };
            assert_eq!(detected.encoding(), expected, "big_endian={big_endian}");

            // Simulate the first (BOM/pattern-driven) refinement stage that
            // `ParseState::Init` performs, then drive the second
            // (declaration-driven) stage directly with a bare "UTF-16" label.
            let mut state = ReaderState {
                encoding: EncodingRef::BomDetected(detected.encoding()),
                ..ReaderState::default()
            };

            let event = state
                .emit_question_mark(b"<?xml version=\"1.0\" encoding=\"UTF-16\"?>")
                .expect("well-formed declaration");
            assert!(
                matches!(event, Event::Decl(_)),
                "big_endian={big_endian}: expected a Decl event, got {event:?}",
            );
            assert_eq!(
                state.encoding.encoding(),
                expected,
                "big_endian={big_endian}: the pattern's byte order must survive the bare label",
            );
        }
    }
}

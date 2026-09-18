//! High-level interface for the pull-based XML parser.
//!
//! The central type is [`Reader`], a low-level, encoding-agnostic, pull-based
//! XML event reader that mirrors the quick-xml 0.41 `Reader`. It consumes bytes
//! and streams XML [`Event`]s, borrowing their payload
//! either from the input itself (for the `&[u8]`/`&str` sources, giving a
//! zero-copy path) or from a caller-provided `Vec<u8>` buffer (for arbitrary
//! [`BufRead`](std::io::BufRead) sources).
//!
//! Reading is configured through [`Config`], retrieved with [`Reader::config()`]
//! and mutated through [`Reader::config_mut()`].
//!
//! This reader does not manage namespace declarations and is not able to resolve
//! prefixes. For namespace-aware reading, use [`NsReader`], which wraps this
//! [`Reader`] and resolves prefixes to namespaces.
//!
//! # Example
//!
//! ```
//! use oxixml_xml::events::Event;
//! use oxixml_xml::reader::Reader;
//!
//! let mut reader = Reader::from_str("<tag>text</tag>");
//! reader.config_mut().trim_text(true);
//!
//! assert!(matches!(reader.read_event().expect("start"), Event::Start(_)));
//! assert!(matches!(reader.read_event().expect("text"), Event::Text(_)));
//! assert!(matches!(reader.read_event().expect("end"), Event::End(_)));
//! assert!(matches!(reader.read_event().expect("eof"), Event::Eof));
//! ```

use std::io;
use std::ops::Range;

use crate::encoding::Decoder;
#[cfg(feature = "encoding")]
use crate::encoding::DetectedEncoding;
use crate::errors::{Error, IllFormedError, SyntaxError};
use crate::events::{BytesRef, Event};
use crate::parser::{DtdParser, ElementParser, Parser, PiParser};
use crate::reader::state::ReaderState;
#[cfg(feature = "encoding")]
use oxixml_encoding::Encoding;

/// A parser configuration.
///
/// The current configuration can be retrieved with [`Reader::config()`] and
/// changed through [`Reader::config_mut()`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde-types", derive(serde::Deserialize, serde::Serialize))]
#[non_exhaustive]
pub struct Config {
    /// Whether a lone ampersand (without a paired semicolon) is allowed in text
    /// content. Unless enabled, a dangling ampersand yields
    /// [`Error::IllFormed(UnclosedReference)`].
    ///
    /// Default: `false`.
    ///
    /// [`Error::IllFormed(UnclosedReference)`]: crate::errors::IllFormedError::UnclosedReference
    pub allow_dangling_amp: bool,

    /// Whether unmatched closing tag names are allowed. Unless enabled, a dangling
    /// end tag yields [`Error::IllFormed(UnmatchedEndTag)`].
    ///
    /// When set to `true`, it won't check if a closing tag has a corresponding
    /// opening tag at all. For example, `<a></a></b>` will be permitted.
    ///
    /// Default: `false`.
    ///
    /// [`Error::IllFormed(UnmatchedEndTag)`]: crate::errors::IllFormedError::UnmatchedEndTag
    pub allow_unmatched_ends: bool,

    /// Whether comments are validated for the forbidden `--` sequence. If enabled,
    /// an invalid comment yields [`Error::IllFormed(DoubleHyphenInComment)`].
    ///
    /// Default: `false`.
    ///
    /// [`Error::IllFormed(DoubleHyphenInComment)`]: crate::errors::IllFormedError::DoubleHyphenInComment
    pub check_comments: bool,

    /// Whether mismatched closing tag names are detected. If enabled, a mismatch
    /// yields [`Error::IllFormed(MismatchedEndTag)`].
    ///
    /// When set to `false`, it won't check if a closing tag matches the
    /// corresponding opening tag. For example, `<mytag></different_tag>` will be
    /// permitted.
    ///
    /// Default: `true`.
    ///
    /// [`Error::IllFormed(MismatchedEndTag)`]: crate::errors::IllFormedError::MismatchedEndTag
    pub check_end_names: bool,

    /// Whether the full set of XML [well-formedness constraints] is enforced.
    ///
    /// The default parsing path of this crate is deliberately as permissive as
    /// `quick-xml` 0.41, which only detects a handful of problems. Setting this
    /// flag turns the reader into a conforming **non-validating** XML processor:
    /// the `Char` and `Name` productions, the document structure of section 2.1,
    /// the grammar of the XML declaration and of the (internal) document type
    /// declaration, and the entity-related constraints are all enforced, and
    /// violations are reported as [`Error::NotWellFormed`].
    ///
    /// *Validity* constraints are never enforced — that would require reading the
    /// external DTD subset, which this crate does not do.
    ///
    /// This check assumes that no white space is trimmed away, so it should not
    /// be combined with [`trim_text_start`](Self::trim_text_start).
    ///
    /// Default: `false`.
    ///
    /// [well-formedness constraints]: https://www.w3.org/TR/xml/#dt-wellformed
    /// [`Error::NotWellFormed`]: crate::errors::Error::NotWellFormed
    pub check_well_formedness: bool,

    /// Whether empty elements are split into a [`Start`] and an [`End`] event.
    ///
    /// When `true`, every [`Empty`] event from a self-closing tag like `<tag/>` is
    /// expanded into a [`Start`] followed by an [`End`].
    ///
    /// Default: `false`.
    ///
    /// [`Empty`]: crate::events::Event::Empty
    /// [`Start`]: crate::events::Event::Start
    /// [`End`]: crate::events::Event::End
    pub expand_empty_elements: bool,

    /// Whether trailing white space after the markup name is trimmed in closing
    /// tags (`</a >`).
    ///
    /// Default: `true`.
    pub trim_markup_names_in_closing_tags: bool,

    /// Whether white space before character data is removed. If, after trimming, a
    /// [`Text`](crate::events::Event::Text) event is empty it is not emitted.
    ///
    /// Default: `false`.
    pub trim_text_start: bool,

    /// Whether white space after character data is removed. If, after trimming, a
    /// [`Text`](crate::events::Event::Text) event is empty it is not emitted.
    ///
    /// Default: `false`.
    pub trim_text_end: bool,

    /// Maximum number of open (unclosed) elements a
    /// [`NsReader`] will accept while nesting namespace
    /// scopes, before returning
    /// [`Error::Namespace(NamespaceError::TooDeeplyNested)`](crate::errors::Error::Namespace)
    /// instead of pushing another scope.
    ///
    /// This guards [`NamespaceResolver`](crate::name::NamespaceResolver)'s
    /// scope stack against unbounded growth from pathologically (or
    /// maliciously) deep documents. Only [`NsReader`]
    /// enforces it; the namespace-agnostic [`Reader`] does not track nesting
    /// depth and is unaffected by this setting.
    ///
    /// Default: [`DEFAULT_MAX_ELEMENT_DEPTH`]. Raise it if you need to read
    /// documents nested deeper than that (there is no arithmetic-overflow
    /// reason to keep it low — the underlying counter is a `usize` — only a
    /// resource-usage one), or set it to `usize::MAX` to disable the check
    /// entirely.
    pub max_element_depth: usize,
}

/// Default value of [`Config::max_element_depth`].
pub const DEFAULT_MAX_ELEMENT_DEPTH: usize = 1024;

impl Config {
    /// Sets both [`trim_text_start`](Self::trim_text_start) and
    /// [`trim_text_end`](Self::trim_text_end) to the same value.
    #[inline]
    pub fn trim_text(&mut self, trim: bool) {
        self.trim_text_start = trim;
        self.trim_text_end = trim;
    }

    /// Turns on or off all well-formedness checks, namely
    /// [`check_comments`](Self::check_comments),
    /// [`check_end_names`](Self::check_end_names) and
    /// [`check_well_formedness`](Self::check_well_formedness).
    #[inline]
    pub fn enable_all_checks(&mut self, enable: bool) {
        self.check_comments = enable;
        self.check_end_names = enable;
        self.check_well_formedness = enable;
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            allow_dangling_amp: false,
            allow_unmatched_ends: false,
            check_comments: false,
            check_end_names: true,
            check_well_formedness: false,
            expand_empty_elements: false,
            trim_markup_names_in_closing_tags: true,
            trim_text_start: false,
            trim_text_end: false,
            max_element_depth: DEFAULT_MAX_ELEMENT_DEPTH,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

macro_rules! read_event_impl {
    (
        $self:ident, $buf:ident,
        $reader:expr,
        $read_until_close:ident
        $(, $await:ident)?
    ) => {{
        let event = loop {
            break match $self.state.state {
                ParseState::Init => {
                    // If encoding set explicitly, we not need to detect it. For example,
                    // explicit UTF-8 set automatically if Reader was created using `from_str`.
                    // But we still need to remove BOM for consistency with no encoding
                    // feature enabled path
                    #[cfg(feature = "encoding")]
                    if let Some(encoding) = $reader.detect_encoding() $(.$await)? ? {
                        if $self.state.encoding.can_be_refined() {
                            $self.state.encoding =
                                crate::reader::EncodingRef::BomDetected(encoding.encoding());
                        }
                    }

                    // Removes UTF-8 BOM if it is present
                    #[cfg(not(feature = "encoding"))]
                    $reader.remove_utf8_bom() $(.$await)? ?;

                    $self.state.state = ParseState::InsideText;
                    continue;
                }
                ParseState::InsideRef => {
                    let start = $self.state.offset;
                    match $reader.read_ref($buf, &mut $self.state.offset) $(.$await)? {
                        // Emit reference, go to InsideText state
                        ReadRefResult::Ref(bytes) => {
                            $self.state.state = ParseState::InsideText;
                            // +1 to skip start `&`
                            // -1 to skip end `;`
                            Ok(Event::GeneralRef(BytesRef::wrap(
                                &bytes[1..bytes.len() - 1],
                                $self.decoder(),
                            )))
                        }
                        // Go to Done state
                        ReadRefResult::UpToEof(bytes) if $self.state.config.allow_dangling_amp => {
                            $self.state.state = ParseState::Done;
                            Ok(Event::Text($self.state.emit_text(bytes)))
                        }
                        ReadRefResult::UpToEof(_) => {
                            $self.state.state = ParseState::Done;
                            $self.state.last_error_offset = start;
                            Err(Error::IllFormed(IllFormedError::UnclosedReference))
                        }
                        // Do not change state, stay in InsideRef
                        ReadRefResult::UpToRef(bytes) if $self.state.config.allow_dangling_amp => {
                            Ok(Event::Text($self.state.emit_text(bytes)))
                        }
                        ReadRefResult::UpToRef(_) => {
                            $self.state.last_error_offset = start;
                            Err(Error::IllFormed(IllFormedError::UnclosedReference))
                        }
                        // Go to InsideMarkup state
                        ReadRefResult::UpToMarkup(bytes)
                            if $self.state.config.allow_dangling_amp =>
                        {
                            $self.state.state = ParseState::InsideMarkup;
                            Ok(Event::Text($self.state.emit_text(bytes)))
                        }
                        ReadRefResult::UpToMarkup(_) => {
                            $self.state.state = ParseState::InsideMarkup;
                            $self.state.last_error_offset = start;
                            Err(Error::IllFormed(IllFormedError::UnclosedReference))
                        }
                        ReadRefResult::Err(e) => Err(Error::from(e)),
                    }
                }
                ParseState::InsideText => {
                    if $self.state.config.trim_text_start {
                        $reader.skip_whitespace(&mut $self.state.offset) $(.$await)? ?;
                    }

                    match $reader.read_text(
                        $buf,
                        &mut $self.state.offset,
                        $self.state.config.trim_text_end,
                    ) $(.$await)? {
                        ReadTextResult::Markup(buf) => {
                            $self.state.state = ParseState::InsideMarkup;
                            // Pass `buf` to the next iteration of parsing loop
                            $buf = buf;
                            continue;
                        }
                        ReadTextResult::Ref(buf) => {
                            $self.state.state = ParseState::InsideRef;
                            // Pass `buf` to the next iteration of parsing loop
                            $buf = buf;
                            continue;
                        }
                        ReadTextResult::UpToMarkup(bytes) => {
                            $self.state.state = ParseState::InsideMarkup;
                            // An all-white-space run before markup cannot reach
                            // this arm while `trim_text_end` is set: `read_text`
                            // reports it as `Markup` instead, so no empty `Text`
                            // event is ever produced here — as
                            // `Config::trim_text_end` documents, and unlike
                            // quick-xml 0.41, which leaves this case open.
                            debug_assert!(
                                !$self.state.config.trim_text_end
                                    || !bytes.iter().all(|b| crate::utils::is_whitespace(*b)),
                                "an all-white-space run must be reported as `Markup`",
                            );
                            Ok(Event::Text($self.state.emit_text(bytes)))
                        }
                        ReadTextResult::UpToRef(bytes) => {
                            $self.state.state = ParseState::InsideRef;
                            // As for `UpToMarkup`, an all-white-space run cannot
                            // reach this arm while `trim_text_end` is set, so the
                            // emitted event is never empty.
                            debug_assert!(
                                !$self.state.config.trim_text_end
                                    || !bytes.iter().all(|b| crate::utils::is_whitespace(*b)),
                                "an all-white-space run must be reported as `Ref`",
                            );
                            Ok(Event::Text($self.state.emit_text(bytes)))
                        }
                        ReadTextResult::UpToEof(bytes) => {
                            $self.state.state = ParseState::Done;
                            // Trim bytes from end if required
                            let event = $self.state.emit_text(bytes);
                            if event.is_empty() {
                                Ok(Event::Eof)
                            } else {
                                Ok(Event::Text(event))
                            }
                        }
                        ReadTextResult::Err(e) => Err(Error::from(e)),
                    }
                }
                // Go to InsideText state in next two arms
                ParseState::InsideMarkup => $self.$read_until_close($buf) $(.$await)?,
                ParseState::InsideEmpty => Ok(Event::End($self.state.close_expanded_empty())),
                ParseState::Done => Ok(Event::Eof),
            };
        };
        let event = match event {
            Ok(ev) if $self.state.config.check_well_formedness => {
                match $self.state.check_well_formedness(&ev) {
                    Ok(()) => Ok(ev),
                    Err(err) => Err(err),
                }
            }
            other => other,
        };
        match event {
            // #513: In case of ill-formed errors we already consume the wrong data
            // and change the state. We can continue parsing if we wish
            Err(Error::IllFormed(_)) => {}
            Err(_) | Ok(Event::Eof) => $self.state.state = ParseState::Done,
            _ => {}
        }
        event
    }};
}

/// Read bytes up to the `>` and skip it. This macro is expected to be used after
/// seeing the `<` symbol and skipping it. It inspects the next (current) symbol
/// and returns an appropriate [`Event`]:
///
/// |Symbol |Event
/// |-------|-------------------------------------
/// |`!`    |`Comment`, `CData` or `DocType`
/// |`/`    |`End`
/// |`?`    |`PI`
/// |_other_|`Start` or `Empty`
///
/// Moves parser to the `InsideText` state.
macro_rules! read_until_close {
    (
        $self:ident, $buf:ident,
        $reader:expr
        $(, $await:ident)?
    ) => {{
        $self.state.state = ParseState::InsideText;

        let start = $self.state.offset;
        match $reader.peek_one() $(.$await)? {
            // `<!` - comment, CDATA or DOCTYPE declaration
            Ok(Some(b'!')) => {
                match $reader.read_bang_element($buf, &mut $self.state.offset) $(.$await)? {
                    Ok((bang_type, bytes)) => $self.state.emit_bang(bang_type, bytes),
                    Err(e) => {
                        // We want to report error at `<`
                        $self.state.last_error_offset = start;
                        Err(e)
                    }
                }
            }
            // `</` - closing tag
            // #776: We parse using ElementParser which allows us to have attributes
            // in close tags. While such tags are not allowed by the specification,
            // we anyway allow to parse them because:
            // - we do not check constraints during parsing. This is performed by the
            //   optional validate step which user should call manually
            // - if we just look for `>` we will parse `</tag attr=">" >` as end tag
            //   `</tag attr=">` and text `" >` which probably no one existing parser
            //   does. This is malformed XML, however it is tolerated by some parsers
            //   (e.g. the one used by Adobe Flash) and such documents do exist in the wild.
            Ok(Some(b'/')) => {
                match $reader.read_with(ElementParser::Outside, $buf, &mut $self.state.offset) $(.$await)? {
                    Ok(bytes) => $self.state.emit_end(bytes),
                    Err(e) => {
                        // We want to report error at `<`
                        $self.state.last_error_offset = start;
                        Err(e)
                    }
                }
            }
            // `<?` - processing instruction
            Ok(Some(b'?')) => {
                match $reader.read_with(PiParser(false), $buf, &mut $self.state.offset) $(.$await)? {
                    Ok(bytes) => $self.state.emit_question_mark(bytes),
                    Err(e) => {
                        // We want to report error at `<`
                        $self.state.last_error_offset = start;
                        Err(e)
                    }
                }
            }
            // `<...` - opening or self-closed tag
            Ok(Some(_)) => {
                match $reader.read_with(ElementParser::Outside, $buf, &mut $self.state.offset) $(.$await)? {
                    Ok(bytes) => Ok($self.state.emit_start(bytes)),
                    Err(e) => {
                        // We want to report error at `<`
                        $self.state.last_error_offset = start;
                        Err(e)
                    }
                }
            }
            // `<` - syntax error, tag not closed
            Ok(None) => {
                // We want to report error at `<`
                $self.state.last_error_offset = start;
                Err(Error::Syntax(SyntaxError::UnclosedTag))
            }
            Err(e) => Err(Error::from(e)),
        }
    }};
}

/// Generalization of `read_to_end` method for buffered and borrowed readers
macro_rules! read_to_end {
    (
        // $self: &mut Reader
        $self:expr, $end:expr, $buf:expr,
        $read_event:ident,
        // Code block that performs clearing of internal buffer after read of each event
        $clear:block
        $(, $await:ident)?
    ) => {{
        // Because we take position after the event before the End event,
        // it is important that this position indicates beginning of the End event.
        // If between last event and the End event would be only spaces, then we
        // take position before the spaces, but spaces would be skipped without
        // generating event if `trim_text_start` is set to `true`. To prevent that
        // we temporary disable start text trimming.
        //
        // We also cannot take position after getting End event, because if
        // `trim_markup_names_in_closing_tags` is set to `true` (which is the default),
        // we do not known the real size of the End event that it is occupies in
        // the source and cannot correct the position after the End event.
        // So, we in any case should tweak parser configuration.
        let config = $self.config_mut();
        let trim = config.trim_text_start;
        config.trim_text_start = false;

        let start = $self.buffer_position();
        let mut depth = 0;
        loop {
            $clear
            let end = $self.buffer_position();
            match $self.$read_event($buf) $(.$await)? {
                Err(e) => {
                    $self.config_mut().trim_text_start = trim;
                    return Err(e);
                }

                Ok(Event::Start(e)) if e.name() == $end => depth += 1,
                Ok(Event::End(e)) if e.name() == $end => {
                    if depth == 0 {
                        $self.config_mut().trim_text_start = trim;
                        break start..end;
                    }
                    depth -= 1;
                }
                Ok(Event::Eof) => {
                    $self.config_mut().trim_text_start = trim;
                    return Err(Error::missed_end($end, $self.decoder()));
                }
                _ => (),
            }
        }
    }};
}

#[cfg(feature = "async-tokio")]
mod async_tokio;
mod buffered_reader;
#[cfg(test)]
mod check_tests;
mod ns_reader;
mod slice_reader;
mod state;

pub use ns_reader::NsReader;

/// Range of input in bytes, that corresponds to some piece of XML
pub type Span = Range<u64>;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Possible reader states. The state transition diagram (`true` and `false` shows
/// value of [`Config::expand_empty_elements`] option):
///
/// ```mermaid
/// flowchart LR
///   subgraph _
///     direction LR
///
///     Init         -- "(no event)"\n                                       --> InsideMarkup
///     InsideMarkup -- Decl, DocType, PI\nComment, CData\nStart, Empty, End --> InsideText
///     InsideText   -- "#lt;false#gt;\n(no event)"\nText                    --> InsideMarkup
///     InsideRef    -- "(no event)"\nGeneralRef                             --> InsideText
///   end
///   InsideText     -- "#lt;true#gt;"\nStart --> InsideEmpty
///   InsideEmpty    -- End                   --> InsideText
///   _ -. Eof .-> Done
/// ```
#[derive(Clone, Debug)]
pub(crate) enum ParseState {
    /// Initial state in which reader stay after creation. Transition from that
    /// state could produce a `Text`, `Decl`, `Comment` or `Start` event. The next
    /// state is always `InsideText`. The reader will never return to this state.
    Init,
    /// State after seeing the `&` symbol in textual content. Depending on the next
    /// symbol all other events could be generated.
    ///
    /// After generating one event the reader moves to the `InsideText` state.
    InsideRef,
    /// State after seeing the `<` symbol. Depending on the next symbol all other
    /// events could be generated.
    ///
    /// After generating one event the reader moves to the `InsideText` state.
    InsideMarkup,
    /// State in which reader searches the `<` symbol of a markup. All bytes before
    /// that symbol will be returned in the [`Event::Text`]
    /// event. After that the reader moves to the `InsideMarkup` state.
    InsideText,
    /// This state is used only if option [`expand_empty_elements`] is set to `true`.
    /// Reader enters to this state when it is in a `InsideText` state and emits an
    /// [`Event::Start`](crate::events::Event::Start) event. The next event emitted
    /// will be an [`Event::End`](crate::events::Event::End), after which reader
    /// returned to the `InsideText` state.
    ///
    /// [`expand_empty_elements`]: Config::expand_empty_elements
    InsideEmpty,
    /// Reader enters this state when `Eof` event generated or an error occurred.
    /// This is the last state, the reader stay in it forever.
    Done,
}

/// A reference to an encoding together with information about how it was retrieved.
///
/// The state transition diagram:
///
/// ```mermaid
/// flowchart LR
///   Implicit    -- from_str       --> Explicit
///   Implicit    -- BOM            --> BomDetected
///   Implicit    -- "encoding=..." --> XmlDetected
///   BomDetected -- "encoding=..." --> XmlDetected
/// ```
#[cfg(feature = "encoding")]
#[derive(Clone, Copy, Debug)]
pub(crate) enum EncodingRef {
    /// Encoding was implicitly assumed to have a specified value. It can be refined
    /// using BOM or by the XML declaration event (`<?xml encoding=... ?>`)
    Implicit(Encoding),
    /// Encoding was explicitly set to the desired value. It cannot be changed
    /// nor by BOM, nor by parsing XML declaration (`<?xml encoding=... ?>`)
    Explicit(Encoding),
    /// Encoding was detected from a byte order mark (BOM) or by the first bytes
    /// of the content. It can be refined by the XML declaration event
    /// (`<?xml encoding=... ?>`)
    BomDetected(Encoding),
    /// Encoding was detected using XML declaration event (`<?xml encoding=... ?>`).
    /// It can no longer change.
    XmlDetected(Encoding),
}

#[cfg(feature = "encoding")]
impl EncodingRef {
    #[inline]
    const fn encoding(&self) -> Encoding {
        match self {
            Self::Implicit(e) | Self::Explicit(e) | Self::BomDetected(e) | Self::XmlDetected(e) => {
                *e
            }
        }
    }
    #[inline]
    const fn can_be_refined(&self) -> bool {
        match self {
            Self::Implicit(_) | Self::BomDetected(_) => true,
            Self::Explicit(_) | Self::XmlDetected(_) => false,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A direct stream to the underlying [`Reader`]s reader which updates
/// [`Reader::buffer_position()`] when read from it.
#[derive(Debug)]
#[must_use = "streams do nothing unless read or polled"]
pub struct BinaryStream<'r, R> {
    inner: &'r mut R,
    offset: &'r mut u64,
}

impl<'r, R> BinaryStream<'r, R> {
    /// Returns current position in bytes in the original source.
    #[inline]
    pub const fn offset(&self) -> u64 {
        *self.offset
    }

    /// Gets a reference to the underlying reader.
    #[inline]
    pub const fn get_ref(&self) -> &R {
        self.inner
    }

    /// Gets a mutable reference to the underlying reader.
    ///
    /// Avoid read from this reader because this will not update reader's position
    /// and will lead to incorrect positions of errors. Read from this stream instead.
    #[inline]
    pub fn get_mut(&mut self) -> &mut R {
        self.inner
    }
}

impl<R> io::Read for BinaryStream<'_, R>
where
    R: io::Read,
{
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let amt = self.inner.read(buf)?;
        *self.offset += amt as u64;
        Ok(amt)
    }
}

impl<R> io::BufRead for BinaryStream<'_, R>
where
    R: io::BufRead,
{
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        self.inner.consume(amt);
        *self.offset += amt as u64;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A low level encoding-agnostic XML event reader.
///
/// Consumes bytes and streams XML [`Event`]s.
///
/// This reader does not manage namespace declarations and is not able to resolve
/// prefixes. For namespace-aware reading, use [`NsReader`].
///
/// # Examples
///
/// ```
/// use oxixml_xml::events::Event;
/// use oxixml_xml::reader::Reader;
///
/// let xml = r#"<tag1 att1 = "test">
///                 <tag2><!--Test comment-->Test</tag2>
///                 <tag2>Test 2</tag2>
///              </tag1>"#;
/// let mut reader = Reader::from_str(xml);
/// reader.config_mut().trim_text(true);
///
/// let mut count = 0;
/// let mut txt = Vec::new();
/// let mut buf = Vec::new();
///
/// // The `Reader` does not implement `Iterator` because it outputs borrowed data (`Cow`s)
/// loop {
///     // NOTE: this is the generic case when we don't know about the input BufRead.
///     // when the input is a &str or a &[u8], we don't actually need to use another
///     // buffer, we could directly call `reader.read_event()`
///     match reader.read_event_into(&mut buf) {
///         Err(e) => panic!("Error at position {}: {:?}", reader.error_position(), e),
///         // exits the loop when reaching end of file
///         Ok(Event::Eof) => break,
///
///         Ok(Event::Start(e)) => {
///             match e.name().as_ref() {
///                 b"tag2" => count += 1,
///                 _ => (),
///             }
///         }
///         Ok(Event::Text(e)) => txt.push(e.decode().expect("utf-8").into_owned()),
///
///         // There are several other `Event`s we do not consider here
///         _ => (),
///     }
///     // if we don't keep a borrow elsewhere, we can clear the buffer to keep memory usage low
///     buf.clear();
/// }
/// assert_eq!(count, 2);
/// ```
#[derive(Debug, Clone)]
pub struct Reader<R> {
    /// Source of data for parse
    reader: R,
    /// Configuration and current parse state
    state: ReaderState,
}

/// Builder methods
impl<R> Reader<R> {
    /// Creates a `Reader` that reads from a given reader.
    pub fn from_reader(reader: R) -> Self {
        Self {
            reader,
            state: ReaderState::default(),
        }
    }

    /// Returns reference to the parser configuration
    pub const fn config(&self) -> &Config {
        &self.state.config
    }

    /// Returns mutable reference to the parser configuration
    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.state.config
    }

    /// Returns the replacement text of an internal general entity declared by the
    /// document type declaration of the document being read. Only populated while
    /// [`Config::check_well_formedness`] is enabled.
    #[inline]
    pub(super) fn declared_entity(&self, name: &str) -> Option<&str> {
        self.state.declared_entity(name)
    }

    /// Tells the optional well-formedness checker that this reader is driven by
    /// an [`NsReader`], so that the extra name constraints of Namespaces in XML
    /// apply.
    #[inline]
    pub(super) fn enable_namespace_constraints(&mut self) {
        self.state.enable_namespace_constraints();
    }
}

/// Getters
impl<R> Reader<R> {
    /// Consumes `Reader` returning the underlying reader.
    ///
    /// Can be used to compute line and column of a parsing error position.
    pub fn into_inner(self) -> R {
        self.reader
    }

    /// Gets a reference to the underlying reader.
    pub const fn get_ref(&self) -> &R {
        &self.reader
    }

    /// Gets a mutable reference to the underlying reader.
    ///
    /// Avoid read from this reader because this will not update reader's position
    /// and will lead to incorrect positions of errors. If you want to read, use
    /// [`stream()`] instead.
    ///
    /// [`stream()`]: Self::stream
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Gets the byte position in the input data just after the last emitted event
    /// (i.e. this is position where data of last event ends).
    ///
    /// Note, that for text events which is originally ended with whitespace characters
    /// (` `, `\t`, `\r`, and `\n`) if [`Config::trim_text_end`] is set this is position
    /// before trim, not the position of the last byte of the
    /// [`Event::Text`] content.
    pub const fn buffer_position(&self) -> u64 {
        self.state.offset
    }

    /// Gets the last error byte position in the input data. If there is no errors
    /// yet, returns `0`.
    ///
    /// Unlike `buffer_position` it will point to the place where it is rational
    /// to report error to the end user. For example, all
    /// [`SyntaxError`]s are reported when the parser
    /// sees EOF inside of some kind of markup. The `buffer_position()` will point
    /// to the last byte of input which is not very useful. `error_position()` will
    /// point to the start of corresponding markup element (i. e. to the `<` character).
    ///
    /// This position is always `<= buffer_position()`.
    pub const fn error_position(&self) -> u64 {
        self.state.last_error_offset
    }

    /// Get the decoder, used to decode bytes, read by this reader, to the strings.
    ///
    /// If [`encoding`] feature is enabled, the used encoding may change after
    /// parsing the XML declaration, otherwise encoding is fixed to UTF-8.
    ///
    /// If [`encoding`] feature is enabled and no encoding is specified in declaration,
    /// defaults to UTF-8.
    ///
    /// [`encoding`]: ../index.html#encoding
    #[inline]
    pub const fn decoder(&self) -> Decoder {
        self.state.decoder()
    }

    /// Get the direct access to the underlying reader, but tracks the amount of
    /// read data and update [`Reader::buffer_position()`] accordingly.
    ///
    /// Note, that this method gives you access to the internal reader and read
    /// data will not be returned in any subsequent events read by `read_event`
    /// family of methods.
    ///
    /// # Example
    ///
    /// This example demonstrates how to read stream raw bytes from an XML document.
    ///
    /// ```
    /// use std::io::Read;
    /// use oxixml_xml::events::{BytesEnd, BytesStart, Event};
    /// use oxixml_xml::reader::Reader;
    ///
    /// let mut reader = Reader::from_str("<tag>binary << data&></tag>");
    /// //                                 ^    ^               ^     ^
    /// //                                 0    5              21    27
    ///
    /// assert_eq!(
    ///     (reader.read_event().expect("start"), reader.buffer_position()),
    ///     (Event::Start(BytesStart::new("tag")), 5)
    /// );
    ///
    /// // Reading from the stream() advances position
    /// let mut inner = reader.stream();
    ///
    /// // Read binary data. We must know its size
    /// let mut binary = [0u8; 16];
    /// inner.read_exact(&mut binary).expect("16 bytes");
    /// assert_eq!(&binary, b"binary << data&>");
    /// assert_eq!(inner.offset(), 21);
    /// assert_eq!(reader.buffer_position(), 21);
    ///
    /// assert_eq!(
    ///     (reader.read_event().expect("end"), reader.buffer_position()),
    ///     (Event::End(BytesEnd::new("tag")), 27)
    /// );
    ///
    /// assert_eq!(reader.read_event().expect("eof"), Event::Eof);
    /// ```
    #[inline]
    pub fn stream(&mut self) -> BinaryStream<'_, R> {
        BinaryStream {
            inner: &mut self.reader,
            offset: &mut self.state.offset,
        }
    }
}

/// Private sync reading methods
impl<R> Reader<R> {
    /// Read text into the given buffer, and return an event that borrows from
    /// either that buffer or from the input itself, based on the type of the
    /// reader.
    fn read_event_impl<'i, B>(&mut self, mut buf: B) -> Result<Event<'i>, Error>
    where
        R: XmlSource<'i, B>,
    {
        read_event_impl!(self, buf, self.reader, read_until_close)
    }

    /// Private function to read until `>` is found. This function expects that
    /// it was called just after encounter a `<` symbol.
    fn read_until_close<'i, B>(&mut self, buf: B) -> Result<Event<'i>, Error>
    where
        R: XmlSource<'i, B>,
    {
        read_until_close!(self, buf, self.reader)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Result of an attempt to read XML textual data from the source.
#[derive(Debug)]
pub(crate) enum ReadTextResult<'r, B> {
    /// Start of markup (`<` character) was found in the first byte. `<` was consumed.
    /// Contains buffer that should be returned back to the next iteration cycle
    /// to satisfy borrow checker requirements.
    Markup(B),
    /// Start of reference (`&` character) was found in the first byte.
    /// `&` was not consumed.
    /// Contains buffer that should be returned back to the next iteration cycle
    /// to satisfy borrow checker requirements.
    Ref(B),
    /// Contains text block up to start of markup (`<` character). `<` was consumed.
    UpToMarkup(&'r [u8]),
    /// Contains text block up to start of reference (`&` character).
    /// `&` was not consumed.
    UpToRef(&'r [u8]),
    /// Contains text block up to EOF, neither start of markup (`<` character)
    /// or start of reference (`&` character) was found.
    UpToEof(&'r [u8]),
    /// IO error occurred.
    Err(io::Error),
}

/// Result of an attempt to read general reference from the reader.
#[derive(Debug)]
pub(crate) enum ReadRefResult<'r> {
    /// Contains text block up to end of reference (`;` character).
    /// Result includes start `&`, but not end `;`.
    Ref(&'r [u8]),
    /// Contains text block up to EOF. Neither end of reference (`;`), start of
    /// another reference (`&`) or start of markup (`<`) characters was found.
    /// Result includes start `&`.
    UpToEof(&'r [u8]),
    /// Contains text block up to next possible reference (`&` character).
    /// Result includes start `&`.
    UpToRef(&'r [u8]),
    /// Contains text block up to start of markup (`<` character).
    /// Result includes start `&`.
    UpToMarkup(&'r [u8]),
    /// IO error occurred.
    Err(io::Error),
}

/// Represents an input for a reader that can return borrowed data.
///
/// There are two implementors of this trait: a generic one that reads data from
/// `Self`, copies some part of it into a provided buffer of type `B` and then
/// returns data that borrows from that buffer.
///
/// The other implementor is for `&[u8]` and instead of copying data returns
/// borrowed data from `Self` instead. This implementation allows zero-copy
/// deserialization.
///
/// # Parameters
/// - `'r`: lifetime of a buffer from which events will borrow
/// - `B`: a type of a buffer that can be used to store data read from `Self` and
///   from which events can borrow
pub(crate) trait XmlSource<'r, B> {
    /// Removes UTF-8 BOM if it is present
    #[cfg(not(feature = "encoding"))]
    fn remove_utf8_bom(&mut self) -> io::Result<()>;

    /// Determines encoding from the start of input and removes BOM if it is present
    #[cfg(feature = "encoding")]
    fn detect_encoding(&mut self) -> io::Result<Option<DetectedEncoding>>;

    /// Read input until start of markup (the `<`) is found, start of general entity
    /// reference (the `&`) is found or end of input is reached.
    ///
    /// # Parameters
    /// - `buf`: Buffer that could be filled from an input (`Self`) and
    ///   from which events could borrow their data
    /// - `position`: Will be increased by amount of bytes consumed
    /// - `trim_whitespace_only`: When set, a text run that consists solely of
    ///   white space and is followed by markup is dropped and
    ///   [`ReadTextResult::Markup`] — respectively [`ReadTextResult::Ref`] —
    ///   is returned instead of an empty [`ReadTextResult::UpToMarkup`] or
    ///   [`ReadTextResult::UpToRef`]. Set from
    ///   [`Config::trim_text_end`](crate::reader::Config::trim_text_end), which
    ///   would trim such a run to nothing anyway.
    fn read_text(
        &mut self,
        buf: B,
        position: &mut u64,
        trim_whitespace_only: bool,
    ) -> ReadTextResult<'r, B>;

    /// Read input until end of general reference (the `;`) is found, start of
    /// another general reference (the `&`) is found or end of input is reached.
    ///
    /// This method must be called when current character is `&`.
    ///
    /// # Parameters
    /// - `buf`: Buffer that could be filled from an input (`Self`) and
    ///   from which events could borrow their data
    /// - `position`: Will be increased by amount of bytes consumed
    fn read_ref(&mut self, buf: B, position: &mut u64) -> ReadRefResult<'r>;

    /// Read input until processing instruction is finished.
    ///
    /// This method expect that start sequence of a parser already was read.
    ///
    /// Returns a slice of data read up to the end of the thing being parsed.
    /// The end of thing and the returned content is determined by the used parser.
    ///
    /// If input (`Self`) is exhausted and no bytes was read, or if the specified
    /// parser could not find the ending sequence of the thing, returns `SyntaxError`.
    ///
    /// # Parameters
    /// - `buf`: Buffer that could be filled from an input (`Self`) and
    ///   from which events could borrow their data
    /// - `position`: Will be increased by amount of bytes consumed
    ///
    /// A `P` type parameter is used to preserve state between calls to the underlying
    /// reader which provides bytes fed into the parser.
    fn read_with<P>(&mut self, parser: P, buf: B, position: &mut u64) -> Result<&'r [u8], Error>
    where
        P: Parser;

    /// Read input until comment or CDATA is finished.
    ///
    /// This method expect that `<` already was read.
    ///
    /// Returns a slice of data read up to end of comment or CDATA (`>`),
    /// which does not include into result.
    ///
    /// If input (`Self`) is exhausted and nothing was read, returns `None`.
    ///
    /// # Parameters
    /// - `buf`: Buffer that could be filled from an input (`Self`) and
    ///   from which events could borrow their data
    /// - `position`: Will be increased by amount of bytes consumed
    fn read_bang_element(
        &mut self,
        buf: B,
        position: &mut u64,
    ) -> Result<(BangType, &'r [u8]), Error>;

    /// Consume and discard all the whitespace until the next non-whitespace
    /// character or EOF.
    ///
    /// # Parameters
    /// - `position`: Will be increased by amount of bytes consumed
    fn skip_whitespace(&mut self, position: &mut u64) -> io::Result<()>;

    /// Return one character without consuming it, so that future `read_*` calls
    /// will still include it. On EOF, return `None`.
    fn peek_one(&mut self) -> io::Result<Option<u8>>;
}

/// Possible elements started with `<!`
#[derive(Debug, PartialEq)]
pub(crate) enum BangType {
    /// `<![CDATA[...]]>`
    CData,
    /// `<!--...-->`
    Comment,
    /// `<!DOCTYPE...>`.
    DocType(DtdParser),
}
impl BangType {
    #[inline(always)]
    const fn new(byte: Option<u8>) -> Result<Self, SyntaxError> {
        Ok(match byte {
            Some(b'[') => Self::CData,
            Some(b'-') => Self::Comment,
            Some(b'D') | Some(b'd') => Self::DocType(DtdParser::BeforeInternalSubset(0)),
            _ => return Err(SyntaxError::InvalidBangMarkup),
        })
    }

    /// If element is finished, returns its content up to `>` symbol and
    /// an index of this symbol, otherwise returns `None`
    ///
    /// # Parameters
    /// - `buf`: buffer with data consumed on previous iterations
    /// - `chunk`: data read on current iteration and not yet consumed from reader
    #[inline(always)]
    fn feed(&mut self, buf: &[u8], chunk: &[u8]) -> Option<usize> {
        match self {
            Self::Comment => {
                for i in crate::utils::memchr_iter(b'>', chunk) {
                    // Need to read at least 6 symbols (`!---->`) for properly finished comment
                    // <!----> - XML comment
                    // 0123456 - i
                    if buf.len() + i > 5 {
                        if chunk[..i].ends_with(b"--") {
                            // We cannot strip last `--` from the buffer because we need it in case of
                            // check_comments enabled option. XML standard requires that comment
                            // will not end with `--->` sequence because this is a special case of
                            // `--` in the comment (https://www.w3.org/TR/xml11/#sec-comments)
                            return Some(i);
                        }
                        // End sequence `-|->` was splitted at |
                        //        buf --/   \-- chunk
                        if i == 1 && buf.ends_with(b"-") && chunk[0] == b'-' {
                            return Some(i);
                        }
                        // End sequence `--|>` was splitted at |
                        //         buf --/   \-- chunk
                        if i == 0 && buf.ends_with(b"--") {
                            return Some(i);
                        }
                    }
                }
            }
            Self::CData => {
                for i in crate::utils::memchr_iter(b'>', chunk) {
                    if chunk[..i].ends_with(b"]]") {
                        return Some(i);
                    }
                    // End sequence `]|]>` was splitted at |
                    //        buf --/   \-- chunk
                    if i == 1 && buf.ends_with(b"]") && chunk[0] == b']' {
                        return Some(i);
                    }
                    // End sequence `]]|>` was splitted at |
                    //         buf --/   \-- chunk
                    if i == 0 && buf.ends_with(b"]]") {
                        return Some(i);
                    }
                }
            }
            Self::DocType(parser) => return parser.feed(buf, chunk),
        }
        None
    }
    #[inline]
    const fn to_err(&self) -> SyntaxError {
        match self {
            Self::CData => SyntaxError::UnclosedCData,
            Self::Comment => SyntaxError::UnclosedComment,
            Self::DocType(_) => SyntaxError::UnclosedDoctype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let config = Config::default();
        assert!(!config.allow_dangling_amp);
        assert!(!config.allow_unmatched_ends);
        assert!(!config.check_comments);
        assert!(config.check_end_names);
        assert!(!config.expand_empty_elements);
        assert!(config.trim_markup_names_in_closing_tags);
        assert!(!config.trim_text_start);
        assert!(!config.trim_text_end);
        assert_eq!(config.max_element_depth, DEFAULT_MAX_ELEMENT_DEPTH);
    }

    #[test]
    fn trim_text_sets_both() {
        let mut config = Config::default();
        config.trim_text(true);
        assert!(config.trim_text_start);
        assert!(config.trim_text_end);
    }

    #[test]
    fn enable_all_checks_toggles() {
        let mut config = Config::default();
        config.enable_all_checks(true);
        assert!(config.check_comments);
        assert!(config.check_end_names);
        config.enable_all_checks(false);
        assert!(!config.check_comments);
        assert!(!config.check_end_names);
    }
}

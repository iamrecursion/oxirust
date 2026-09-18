//! The HTML tokenizer of
//! [HTML § 13.2.5](https://html.spec.whatwg.org/multipage/parsing.html#tokenization).
//!
//! The tokenizer is a pull parser: [`Tokenizer::next_token`] runs the state
//! machine until at least one token is ready and returns it. That shape is what
//! the standard's parser needs, because tree construction feeds information back
//! into tokenization — a `<title>` start tag switches the tokenizer to RCDATA, a
//! `<script>` start tag to script data, and `<![CDATA[` is only recognised while
//! the adjusted current node is a foreign element.
//!
//! # Input preprocessing
//!
//! [`Tokenizer::new`] applies the preprocessing of
//! [HTML § 13.2.3.5](https://html.spec.whatwg.org/multipage/parsing.html#preprocessing-the-input-stream):
//! every `CRLF` pair and every lone `CR` becomes a single `LF`. U+0000 is *not*
//! removed — the standard handles it state by state — and neither are
//! noncharacters, which are only parse errors.
//!
//! # Resumable input
//!
//! [`Tokenizer::new`] is handed the whole input at once, so its state machine
//! may treat "no more characters" as end of file. The same state machine also
//! runs the crate's streaming parser
//! ([`StreamingParser`](crate::StreamingParser)), where it is created by
//! [`Tokenizer::streaming`] over an input that is *extended* between steps
//! and only later closed by [`Tokenizer::close_input`].
//!
//! Until the input is closed, running out of characters means "not yet"
//! rather than "never": [`Tokenizer::consume`] then sets the `suspended` flag
//! instead of yielding an end-of-file decision, and the state handler returns
//! at once, leaving the state, the position and the half-built token exactly
//! as they were so that the very same step runs again — and this time reaches
//! a different conclusion — once the next chunk arrives. The handful of states
//! that decide by looking *ahead* rather than by consuming (the markup
//! declaration `--` / `DOCTYPE` / `[CDATA[` sniff, the DOCTYPE `PUBLIC` /
//! `SYSTEM` keywords, the named character reference table, the numeric
//! character reference start) ask for their lookahead up front with
//! [`Tokenizer::needs_lookahead`].
//!
//! A tokenizer created by [`Tokenizer::new`] is closed from the start, so none
//! of that applies to it: it can never suspend, and the whole mechanism is
//! inert on the one-shot path.
//!
//! # Example
//!
//! ```
//! use oxixml_html::{Token, Tokenizer};
//!
//! let mut tokenizer = Tokenizer::new("<p class=greeting>hi &amp; bye");
//! assert!(matches!(tokenizer.next_token(), Token::StartTag(tag)
//!     if tag.name == "p" && tag.attribute("class") == Some("greeting")));
//! assert_eq!(tokenizer.next_token(), Token::Characters("hi ".into()));
//! assert_eq!(tokenizer.next_token(), Token::Characters("&".into()));
//! assert_eq!(tokenizer.next_token(), Token::Characters(" bye".into()));
//! assert_eq!(tokenizer.next_token(), Token::Eof);
//! ```

mod attribute;
mod charref;
mod comment;
mod data;
mod doctype;
mod pi;
mod script;

use std::collections::VecDeque;

use oxixml_support::pos::TextPosition;

use crate::error::{ParseError, codes};
use crate::token::{Attribute, Doctype, ProcessingInstruction, Tag, Token};

/// Which "content model" the tokenizer is reading.
///
/// The tree construction stage switches the tokenizer between these as it
/// encounters the start tags that change how the following text is read; see
/// the "generic raw text element parsing algorithm" and the "generic RCDATA
/// element parsing algorithm" of
/// [HTML § 13.2.6.2](https://html.spec.whatwg.org/multipage/parsing.html#parsing-elements-that-contain-only-text).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ContentModel {
    /// Ordinary markup: tags, comments and character references are recognised.
    #[default]
    Data,
    /// `<title>`, `<textarea>`: character references are recognised, other
    /// markup is not, and only the matching end tag closes the element.
    Rcdata,
    /// `<style>`, `<xmp>`, `<iframe>`, `<noembed>`, `<noframes>`: neither
    /// character references nor markup are recognised.
    RawText,
    /// `<script>`: like raw text, but with the extra `<!--` / `<script>`
    /// escaping states the standard defines for it.
    ScriptData,
    /// `<plaintext>`: everything to the end of the file is text.
    PlainText,
    /// The interior of a `<![CDATA[…]]>` section, which
    /// [HTML § 13.2.5.69](https://html.spec.whatwg.org/multipage/parsing.html#cdata-section-state)
    /// only reaches while the adjusted current node is a foreign element.
    ///
    /// It is not a content model an element selects; it is exposed so that a
    /// caller resuming a tokenizer *inside* a CDATA section — which is what the
    /// html5lib tokenizer suite's `CDATA section state` initial state does —
    /// can name it.
    CdataSection,
}

/// The tokenizer states of
/// [HTML § 13.2.5](https://html.spec.whatwg.org/multipage/parsing.html#tokenization).
///
/// Kept private: callers steer the tokenizer through
/// [`Tokenizer::set_content_model`] rather than by naming states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum State {
    Data,
    Rcdata,
    RawText,
    ScriptData,
    PlainText,
    TagOpen,
    EndTagOpen,
    TagName,
    RcdataLessThanSign,
    RcdataEndTagOpen,
    RcdataEndTagName,
    RawTextLessThanSign,
    RawTextEndTagOpen,
    RawTextEndTagName,
    ScriptDataLessThanSign,
    ScriptDataEndTagOpen,
    ScriptDataEndTagName,
    ScriptDataEscapeStart,
    ScriptDataEscapeStartDash,
    ScriptDataEscaped,
    ScriptDataEscapedDash,
    ScriptDataEscapedDashDash,
    ScriptDataEscapedLessThanSign,
    ScriptDataEscapedEndTagOpen,
    ScriptDataEscapedEndTagName,
    ScriptDataDoubleEscapeStart,
    ScriptDataDoubleEscaped,
    ScriptDataDoubleEscapedDash,
    ScriptDataDoubleEscapedDashDash,
    ScriptDataDoubleEscapedLessThanSign,
    ScriptDataDoubleEscapeEnd,
    BeforeAttributeName,
    AttributeName,
    AfterAttributeName,
    BeforeAttributeValue,
    AttributeValueDoubleQuoted,
    AttributeValueSingleQuoted,
    AttributeValueUnquoted,
    AfterAttributeValueQuoted,
    SelfClosingStartTag,
    BogusComment,
    MarkupDeclarationOpen,
    CommentStart,
    CommentStartDash,
    Comment,
    CommentLessThanSign,
    CommentLessThanSignBang,
    CommentLessThanSignBangDash,
    CommentLessThanSignBangDashDash,
    CommentEndDash,
    CommentEnd,
    CommentEndBang,
    Doctype,
    BeforeDoctypeName,
    DoctypeName,
    AfterDoctypeName,
    AfterDoctypePublicKeyword,
    BeforeDoctypePublicIdentifier,
    DoctypePublicIdentifierDoubleQuoted,
    DoctypePublicIdentifierSingleQuoted,
    AfterDoctypePublicIdentifier,
    BetweenDoctypePublicAndSystemIdentifiers,
    AfterDoctypeSystemKeyword,
    BeforeDoctypeSystemIdentifier,
    DoctypeSystemIdentifierDoubleQuoted,
    DoctypeSystemIdentifierSingleQuoted,
    AfterDoctypeSystemIdentifier,
    BogusDoctype,
    CdataSection,
    CdataSectionBracket,
    CdataSectionEnd,
    ProcessingInstructionOpen,
    ProcessingInstructionTarget,
    AfterProcessingInstructionTarget,
    ProcessingInstructionData,
    ProcessingInstructionQuestionable,
    CharacterReference,
    NamedCharacterReference,
    AmbiguousAmpersand,
    NumericCharacterReference,
    HexadecimalCharacterReferenceStart,
    DecimalCharacterReferenceStart,
    HexadecimalCharacterReference,
    DecimalCharacterReference,
    NumericCharacterReferenceEnd,
}

impl State {
    /// The three states in which a character reference is expanded into an
    /// attribute value rather than into character tokens.
    const fn is_attribute_value(self) -> bool {
        matches!(
            self,
            Self::AttributeValueDoubleQuoted
                | Self::AttributeValueSingleQuoted
                | Self::AttributeValueUnquoted
        )
    }
}

/// Whether the tag under construction is a start or an end tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagKind {
    Start,
    End,
}

/// The HTML tokenizer.
///
/// See the [crate documentation](crate) for the overall shape and an example.
#[derive(Debug)]
pub struct Tokenizer<'a> {
    /// The preprocessed input, owned when preprocessing had to rewrite it or
    /// when the input arrives in chunks.
    source: Source<'a>,
    /// The byte offset of the next character to consume, relative to the
    /// start of [`Self::source`] — which is not the start of the stream once
    /// [`Self::discard_consumed`] has thrown a consumed prefix away.
    pos: usize,
    /// The position of the first byte still held in [`Self::source`]: the
    /// origin every reported position is measured from. It stays at the very
    /// start of the stream unless a consumed prefix has been discarded.
    base: TextPosition,
    /// The offsets, relative to [`Self::source`], at which each line *after*
    /// the first one held in it starts, for position reporting.
    line_starts: Vec<usize>,
    state: State,
    return_state: State,
    queue: VecDeque<Token>,
    tag_kind: TagKind,
    tag: Tag,
    attribute: Option<Attribute>,
    comment: String,
    doctype: Doctype,
    processing_instruction: ProcessingInstruction,
    temp_buffer: String,
    character_reference_code: u32,
    last_start_tag: Option<String>,
    in_foreign_content: bool,
    /// Whether the non-standard `<?…?>` extension is enabled; see
    /// [`Tokenizer::set_processing_instructions`].
    processing_instructions: bool,
    errors: Vec<ParseError>,
    at_eof: bool,
    /// Whether the input is complete. When it is not, running out of
    /// characters suspends the state machine instead of ending the document;
    /// see the module documentation.
    closed: bool,
    /// Set by [`Self::consume`] and [`Self::needs_lookahead`] when the state
    /// machine wants characters the still-open input has not delivered yet.
    /// The step is abandoned and retried from the top when it has.
    suspended: bool,
    /// A `CR` at the very end of a chunk, held back because the `CRLF` it may
    /// be the first half of can only be recognised once the next chunk
    /// arrives; see [`Self::feed`].
    pending_cr: bool,
}

/// The preprocessed input: borrowed when newline normalisation was a no-op,
/// which is the common case for documents that use `LF` line breaks.
#[derive(Debug)]
enum Source<'a> {
    Borrowed(&'a str),
    Owned(String),
}

impl Source<'_> {
    fn as_str(&self) -> &str {
        match self {
            Self::Borrowed(text) => text,
            Self::Owned(text) => text.as_str(),
        }
    }

    /// Appends more preprocessed input, taking ownership of a borrowed source
    /// if it is still borrowed.
    fn push_str(&mut self, text: &str) {
        match self {
            Self::Owned(buffer) => buffer.push_str(text),
            Self::Borrowed(existing) => {
                let mut buffer = String::with_capacity(existing.len() + text.len());
                buffer.push_str(existing);
                buffer.push_str(text);
                *self = Self::Owned(buffer);
            }
        }
    }
}

/// Normalises `CRLF` and lone `CR` to `LF`, borrowing when there is nothing to
/// do.
fn preprocess(input: &str) -> Source<'_> {
    if !input.contains('\r') {
        return Source::Borrowed(input);
    }
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            out.push('\n');
        } else {
            out.push(ch);
        }
    }
    Source::Owned(out)
}

/// The offsets, relative to `text`, at which each line after the first one
/// starts — that is, one past each `LF`. `base` shifts them, so that a slice
/// appended to an existing buffer contributes offsets into that buffer.
fn line_starts_of(text: &str, base: usize) -> impl Iterator<Item = usize> + '_ {
    text.bytes()
        .enumerate()
        .filter(|&(_, byte)| byte == b'\n')
        .map(move |(index, _)| base + index + 1)
}

impl<'a> Tokenizer<'a> {
    /// Creates a tokenizer over `input`, applying the newline preprocessing of
    /// [HTML § 13.2.3.5](https://html.spec.whatwg.org/multipage/parsing.html#preprocessing-the-input-stream).
    #[must_use]
    pub fn new(input: &'a str) -> Self {
        let source = preprocess(input);
        let line_starts = line_starts_of(source.as_str(), 0).collect();
        Self {
            source,
            line_starts,
            closed: true,
            ..Self::empty()
        }
    }

    /// Creates a tokenizer whose input arrives later, in chunks, through
    /// [`Self::feed`], and is declared complete by [`Self::close_input`].
    ///
    /// See the module documentation for what changes: until the input is
    /// closed, running out of characters suspends the state machine rather
    /// than ending the document, and [`Self::try_next_token`] reports that as
    /// `None`.
    pub(crate) fn streaming() -> Self {
        Self {
            source: Source::Owned(String::new()),
            line_starts: Vec::new(),
            closed: false,
            ..Self::empty()
        }
    }

    /// Everything that does not depend on how the input is supplied.
    fn empty() -> Self {
        Self {
            source: Source::Borrowed(""),
            pos: 0,
            base: TextPosition::start(),
            line_starts: Vec::new(),
            state: State::Data,
            return_state: State::Data,
            queue: VecDeque::new(),
            tag_kind: TagKind::Start,
            tag: Tag::default(),
            attribute: None,
            comment: String::new(),
            doctype: Doctype::default(),
            processing_instruction: ProcessingInstruction::default(),
            temp_buffer: String::new(),
            character_reference_code: 0,
            last_start_tag: None,
            in_foreign_content: false,
            processing_instructions: false,
            errors: Vec::new(),
            at_eof: false,
            closed: false,
            suspended: false,
            pending_cr: false,
        }
    }

    /// Appends another chunk of input to a tokenizer created by
    /// [`Self::streaming`], applying the newline preprocessing of
    /// [HTML § 13.2.3.5](https://html.spec.whatwg.org/multipage/parsing.html#preprocessing-the-input-stream)
    /// across the chunk boundary.
    ///
    /// A `CR` at the very end of a chunk is held back rather than normalised
    /// straight away: whether it is a lone `CR` (one `LF`) or the first half
    /// of a `CRLF` (still one `LF`, but swallowing the `LF` that follows) is
    /// not decidable until the next chunk arrives, or until
    /// [`Self::close_input`] says there is none.
    pub(crate) fn feed(&mut self, chunk: &str) {
        debug_assert!(
            !self.closed,
            "Tokenizer::feed after Tokenizer::close_input: the state machine has already been \
             told the input ended, so anything appended now would be tokenized after an \
             end-of-file decision had been taken over it"
        );
        if chunk.is_empty() {
            // An empty chunk is not the end of the stream, so a held-back `CR`
            // stays held back: the `LF` that would pair with it may still be
            // the first character of the next chunk.
            return;
        }
        let mut normalised = String::with_capacity(chunk.len() + 1);
        let mut chars = chunk.chars().peekable();
        if std::mem::take(&mut self.pending_cr) {
            normalised.push('\n');
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
        }
        while let Some(ch) = chars.next() {
            if ch == '\r' {
                match chars.peek() {
                    Some('\n') => {
                        chars.next();
                        normalised.push('\n');
                    }
                    Some(_) => normalised.push('\n'),
                    None => self.pending_cr = true,
                }
            } else {
                normalised.push(ch);
            }
        }
        self.append(&normalised);
    }

    /// Declares the input complete, so that running out of characters means
    /// end of file again.
    pub(crate) fn close_input(&mut self) {
        if std::mem::take(&mut self.pending_cr) {
            // A stream that ends on a `CR` ends on the `LF` it normalises to.
            self.append("\n");
        }
        self.closed = true;
    }

    /// Appends already-preprocessed text and extends the line index over it.
    fn append(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let base = self.source.as_str().len();
        self.line_starts.extend(line_starts_of(text, base));
        self.source.push_str(text);
    }

    /// Throws away the part of the buffered input the state machine has
    /// already consumed, so that a stream fed in chunks does not retain the
    /// whole document.
    ///
    /// Only safe *between* tokens: within a step the state handlers reach back
    /// over the character they just consumed ([`Self::unconsume`]), and
    /// [`Self::rest`] is sliced against the buffer. [`Self::position`] keeps
    /// reporting stream-absolute line, column and offset regardless, because
    /// the discarded prefix's own position is remembered as [`Self::base`].
    pub(crate) fn discard_consumed(&mut self) {
        // Nothing to do for a borrowed one-shot input: it is the caller's
        // buffer, and the whole of it outlives the parse anyway.
        if self.pos == 0 || !matches!(self.source, Source::Owned(_)) {
            return;
        }
        self.base = self.position_at(self.pos);
        if let Source::Owned(buffer) = &mut self.source {
            buffer.drain(..self.pos);
        }
        let consumed = self.pos;
        self.pos = 0;
        self.line_starts
            .retain_mut(|start| match start.checked_sub(consumed) {
                Some(remaining) if remaining > 0 => {
                    *start = remaining;
                    true
                }
                _ => false,
            });
    }

    /// The number of bytes of decoded input the tokenizer is still holding
    /// on to: everything fed in but not yet consumed by the state machine.
    pub(crate) fn buffered_bytes(&self) -> usize {
        self.source.as_str().len() - self.pos
    }

    /// Switches the tokenizer to a different content model.
    ///
    /// The tree construction stage calls this after emitting the start tag of an
    /// element whose content is not ordinary markup.
    pub fn set_content_model(&mut self, model: ContentModel) {
        self.state = match model {
            ContentModel::Data => State::Data,
            ContentModel::Rcdata => State::Rcdata,
            ContentModel::RawText => State::RawText,
            ContentModel::ScriptData => State::ScriptData,
            ContentModel::PlainText => State::PlainText,
            ContentModel::CdataSection => State::CdataSection,
        };
    }

    /// Enables this crate's non-standard `<?…?>` processing instruction
    /// extension, which is **off** by default.
    ///
    /// [HTML § 13.2.5.6](https://html.spec.whatwg.org/multipage/parsing.html#tag-open-state)
    /// makes `<?` an `unexpected-question-mark-instead-of-tag-name` parse error
    /// that is reconsumed in the bogus comment state, so `<?php ?>` in an HTML
    /// document is a comment whose data is `?php ?`. That is what this
    /// tokenizer does by default, and what the html5lib tokenizer and tree
    /// construction suites assert.
    ///
    /// Turning the extension on instead tokenizes a restricted XML-style
    /// processing instruction — the target must match `[A-Za-z_][A-Za-z0-9_-]*`
    /// and must be neither `xml` nor `xml-stylesheet` — so that a template
    /// pipeline that round-trips `<?php … ?>` through this parser keeps it as a
    /// [`Token::ProcessingInstruction`] rather than flattening it to a comment.
    /// Anything the restrictions reject falls back to the bogus comment state
    /// with the leading `?` restored, exactly as the standard would have done.
    pub const fn set_processing_instructions(&mut self, enabled: bool) {
        self.processing_instructions = enabled;
    }

    /// Sets the name of the "most recently emitted start tag", which is what
    /// decides whether an end tag inside RCDATA, RAWTEXT or script data closes
    /// the element or is just text.
    ///
    /// The tokenizer maintains this itself while it runs; a caller only needs
    /// it when starting a tokenizer *in* one of those content models, as the
    /// fragment parsing algorithm does.
    pub fn set_last_start_tag(&mut self, name: &str) {
        self.last_start_tag = Some(name.to_owned());
    }

    /// Tells the tokenizer whether the adjusted current node is a foreign
    /// (MathML or SVG) element, which is the only situation in which
    /// `<![CDATA[` starts a CDATA section rather than a bogus comment.
    pub const fn set_in_foreign_content(&mut self, foreign: bool) {
        self.in_foreign_content = foreign;
    }

    /// The parse errors recorded so far.
    #[must_use]
    pub fn errors(&self) -> &[ParseError] {
        &self.errors
    }

    /// Takes ownership of the parse errors recorded so far, leaving the
    /// tokenizer's list empty.
    #[must_use]
    pub fn take_errors(&mut self) -> Vec<ParseError> {
        std::mem::take(&mut self.errors)
    }

    /// The current position in the preprocessed input stream.
    #[must_use]
    pub fn position(&self) -> TextPosition {
        self.position_at(self.pos)
    }

    /// Runs the state machine until a token is ready, and returns it.
    ///
    /// Once [`Token::Eof`] has been returned every further call returns
    /// [`Token::Eof`] again.
    pub fn next_token(&mut self) -> Token {
        // `Tokenizer::new` closes the input, and only a closed tokenizer is
        // reachable through the public API, so the state machine cannot
        // suspend here.
        self.try_next_token().unwrap_or(Token::Eof)
    }

    /// Runs the state machine until a token is ready, and returns it, or
    /// returns `None` when the input ran out and has not been closed yet.
    ///
    /// After `None` the tokenizer is exactly where it was before the step
    /// that ran out: call [`Self::feed`] (or [`Self::close_input`]) and then
    /// call this again.
    pub(crate) fn try_next_token(&mut self) -> Option<Token> {
        loop {
            if let Some(token) = self.queue.pop_front() {
                return Some(token);
            }
            if self.at_eof {
                return Some(Token::Eof);
            }
            self.suspended = false;
            self.step();
            if self.suspended {
                return None;
            }
        }
    }

    // ------------------------------------------------------------- internals

    /// The whole preprocessed input.
    fn input(&self) -> &str {
        self.source.as_str()
    }

    /// The not-yet-consumed part of the input.
    fn rest(&self) -> &str {
        &self.input()[self.pos..]
    }

    /// The stream-absolute position of a buffer-relative offset.
    ///
    /// `line_starts` holds only the lines that start *inside* the buffer, so
    /// an offset on the buffer's first line continues the line
    /// [`Self::base`] was left on and has to pick up that line's column
    /// count; every other offset starts its column count from a line start
    /// the buffer still holds.
    fn position_at(&self, offset: usize) -> TextPosition {
        let lines_before = self.line_starts.partition_point(|&start| start <= offset);
        // `TextPosition` is 0-based (see `oxixml_support::pos`); the 1-based rendering is applied
        // by `Display` alone.
        let (line_start, base_column) = match lines_before.checked_sub(1) {
            Some(index) => (self.line_starts[index], 0),
            None => (0, self.base.column),
        };
        let column = base_column + self.input()[line_start..offset].chars().count() as u64;
        TextPosition {
            line: self.base.line + lines_before as u64,
            column,
            offset: self.base.offset + offset as u64,
        }
    }

    fn error(&mut self, code: &'static str) {
        let position = self.position();
        self.errors.push(ParseError::new(position, code));
    }

    /// Consumes and returns the next input character, or `None` at end of file.
    ///
    /// On a still-open streaming input, "no next character" is not end of file
    /// but "not yet": the `suspended` flag is raised, and the state handler
    /// must return without taking the end-of-file branch — every `None` arm in
    /// this module's state handlers starts by checking for that. See the
    /// module documentation.
    fn consume(&mut self) -> Option<char> {
        let Some(ch) = self.rest().chars().next() else {
            self.suspended = !self.closed;
            return None;
        };
        self.pos += ch.len_utf8();
        Some(ch)
    }

    /// Suspends the state machine when the input holds fewer than `bytes`
    /// unconsumed bytes and more may still arrive, for the states that decide
    /// by looking ahead rather than by consuming.
    ///
    /// Returns `true` when the caller must return from the step immediately;
    /// on a closed input it is always `false`, since what is there is all
    /// there will ever be.
    fn needs_lookahead(&mut self, bytes: usize) -> bool {
        if self.closed || self.rest().len() >= bytes {
            return false;
        }
        self.suspended = true;
        true
    }

    /// Puts back the character most recently returned by [`Self::consume`].
    fn unconsume(&mut self, ch: char) {
        self.pos -= ch.len_utf8();
    }

    /// Consumes the longest run of characters that contains none of `stops`.
    fn take_run(&mut self, stops: &[char]) -> Option<String> {
        let rest = self.rest();
        let end = rest
            .find(|ch: char| stops.contains(&ch))
            .unwrap_or(rest.len());
        if end == 0 {
            return None;
        }
        let run = rest[..end].to_owned();
        self.pos += end;
        Some(run)
    }

    /// Consumes `text` if the input continues with it, matching ASCII
    /// case-insensitively when `ignore_case` is set.
    fn consume_if_match(&mut self, text: &str, ignore_case: bool) -> bool {
        let rest = self.rest();
        let matched = if ignore_case {
            rest.len() >= text.len()
                && rest.is_char_boundary(text.len())
                && rest[..text.len()].eq_ignore_ascii_case(text)
        } else {
            rest.starts_with(text)
        };
        if matched {
            self.pos += text.len();
        }
        matched
    }

    fn emit(&mut self, token: Token) {
        if let Token::StartTag(tag) = &token {
            self.last_start_tag = Some(tag.name.clone());
        }
        self.queue.push_back(token);
    }

    /// Emits a character token, merging it into the previous one when the
    /// previous queued token is also a character token.
    fn emit_char(&mut self, ch: char) {
        match self.queue.back_mut() {
            Some(Token::Characters(text)) => text.push(ch),
            _ => self.queue.push_back(Token::Characters(ch.to_string())),
        }
    }

    fn emit_str(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match self.queue.back_mut() {
            Some(Token::Characters(buffer)) => buffer.push_str(text),
            _ => self.queue.push_back(Token::Characters(text.to_owned())),
        }
    }

    fn emit_eof(&mut self) {
        self.at_eof = true;
        self.queue.push_back(Token::Eof);
    }

    // ------------------------------------------------------------------ tags

    fn start_tag(&mut self, kind: TagKind, first: char) {
        self.tag_kind = kind;
        self.tag = Tag::default();
        self.tag.name.push(first);
        self.attribute = None;
    }

    /// Finishes the attribute under construction, dropping it when its name
    /// duplicates one already on the tag.
    fn finish_attribute(&mut self) {
        if let Some(attribute) = self.attribute.take() {
            if self.tag.has_attribute(&attribute.name) {
                self.error(codes::DUPLICATE_ATTRIBUTE);
            } else {
                self.tag.attributes.push(attribute);
            }
        }
    }

    fn attribute_value_mut(&mut self) -> &mut String {
        &mut self.attribute.get_or_insert_with(Attribute::default).value
    }

    fn emit_current_tag(&mut self) {
        self.finish_attribute();
        let tag = std::mem::take(&mut self.tag);
        match self.tag_kind {
            TagKind::Start => self.emit(Token::StartTag(tag)),
            TagKind::End => {
                if !tag.attributes.is_empty() {
                    self.error(codes::END_TAG_WITH_ATTRIBUTES);
                }
                if tag.self_closing {
                    self.error(codes::END_TAG_WITH_TRAILING_SOLIDUS);
                }
                self.emit(Token::EndTag(tag));
            }
        }
    }

    /// The "appropriate end tag token" test: an end tag whose name matches the
    /// most recently emitted start tag.
    fn is_appropriate_end_tag(&self) -> bool {
        self.tag_kind == TagKind::End
            && self
                .last_start_tag
                .as_ref()
                .is_some_and(|name| *name == self.tag.name)
    }

    fn emit_comment(&mut self) {
        let data = std::mem::take(&mut self.comment);
        self.emit(Token::Comment(data));
    }

    fn emit_processing_instruction(&mut self) {
        let pi = std::mem::take(&mut self.processing_instruction);
        self.emit(Token::ProcessingInstruction(pi));
    }

    /// "Convert the temporary buffer to a comment": a processing instruction
    /// with an invalid target becomes a bogus comment whose data is `?`
    /// followed by whatever had been consumed as the target.
    fn temp_buffer_to_comment(&mut self) {
        self.comment.clear();
        self.comment.push('?');
        let buffer = std::mem::take(&mut self.temp_buffer);
        self.comment.push_str(&buffer);
    }

    fn emit_doctype(&mut self) {
        let doctype = std::mem::take(&mut self.doctype);
        self.emit(Token::Doctype(doctype));
    }

    /// Runs exactly one state handler.
    fn step(&mut self) {
        match self.state {
            State::Data
            | State::Rcdata
            | State::RawText
            | State::ScriptData
            | State::PlainText
            | State::TagOpen
            | State::EndTagOpen
            | State::TagName
            | State::RcdataLessThanSign
            | State::RcdataEndTagOpen
            | State::RcdataEndTagName
            | State::RawTextLessThanSign
            | State::RawTextEndTagOpen
            | State::RawTextEndTagName
            | State::ScriptDataLessThanSign
            | State::ScriptDataEndTagOpen
            | State::ScriptDataEndTagName => self.step_data(),
            State::ScriptDataEscapeStart
            | State::ScriptDataEscapeStartDash
            | State::ScriptDataEscaped
            | State::ScriptDataEscapedDash
            | State::ScriptDataEscapedDashDash
            | State::ScriptDataEscapedLessThanSign
            | State::ScriptDataEscapedEndTagOpen
            | State::ScriptDataEscapedEndTagName
            | State::ScriptDataDoubleEscapeStart
            | State::ScriptDataDoubleEscaped
            | State::ScriptDataDoubleEscapedDash
            | State::ScriptDataDoubleEscapedDashDash
            | State::ScriptDataDoubleEscapedLessThanSign
            | State::ScriptDataDoubleEscapeEnd => self.step_script(),
            State::BeforeAttributeName
            | State::AttributeName
            | State::AfterAttributeName
            | State::BeforeAttributeValue
            | State::AttributeValueDoubleQuoted
            | State::AttributeValueSingleQuoted
            | State::AttributeValueUnquoted
            | State::AfterAttributeValueQuoted
            | State::SelfClosingStartTag => self.step_attribute(),
            State::BogusComment
            | State::MarkupDeclarationOpen
            | State::CommentStart
            | State::CommentStartDash
            | State::Comment
            | State::CommentLessThanSign
            | State::CommentLessThanSignBang
            | State::CommentLessThanSignBangDash
            | State::CommentLessThanSignBangDashDash
            | State::CommentEndDash
            | State::CommentEnd
            | State::CommentEndBang
            | State::CdataSection
            | State::CdataSectionBracket
            | State::CdataSectionEnd => self.step_comment(),
            State::ProcessingInstructionOpen
            | State::ProcessingInstructionTarget
            | State::AfterProcessingInstructionTarget
            | State::ProcessingInstructionData
            | State::ProcessingInstructionQuestionable => self.step_processing_instruction(),
            State::Doctype
            | State::BeforeDoctypeName
            | State::DoctypeName
            | State::AfterDoctypeName
            | State::AfterDoctypePublicKeyword
            | State::BeforeDoctypePublicIdentifier
            | State::DoctypePublicIdentifierDoubleQuoted
            | State::DoctypePublicIdentifierSingleQuoted
            | State::AfterDoctypePublicIdentifier
            | State::BetweenDoctypePublicAndSystemIdentifiers
            | State::AfterDoctypeSystemKeyword
            | State::BeforeDoctypeSystemIdentifier
            | State::DoctypeSystemIdentifierDoubleQuoted
            | State::DoctypeSystemIdentifierSingleQuoted
            | State::AfterDoctypeSystemIdentifier
            | State::BogusDoctype => self.step_doctype(),
            State::CharacterReference
            | State::NamedCharacterReference
            | State::AmbiguousAmpersand
            | State::NumericCharacterReference
            | State::HexadecimalCharacterReferenceStart
            | State::DecimalCharacterReferenceStart
            | State::HexadecimalCharacterReference
            | State::DecimalCharacterReference
            | State::NumericCharacterReferenceEnd => self.step_charref(),
        }
    }
}

/// The white space characters the tokenizer treats as such: tab, line feed,
/// form feed and space.
pub(crate) const fn is_html_whitespace(ch: char) -> bool {
    matches!(ch, '\t' | '\n' | '\u{c}' | ' ')
}

#[cfg(test)]
mod tests {
    use super::{ContentModel, Tokenizer};
    use crate::token::Token;

    fn tokens(input: &str) -> Vec<Token> {
        let mut tokenizer = Tokenizer::new(input);
        let mut out = Vec::new();
        loop {
            let token = tokenizer.next_token();
            let stop = token == Token::Eof;
            out.push(token);
            if stop {
                return out;
            }
        }
    }

    #[test]
    fn newlines_are_normalised() {
        let collected = tokens("a\r\nb\rc");
        assert_eq!(collected[0], Token::Characters("a\nb\nc".into()));
    }

    #[test]
    fn plain_text_swallows_everything() {
        let mut tokenizer = Tokenizer::new("<a>&amp;");
        tokenizer.set_content_model(ContentModel::PlainText);
        assert_eq!(tokenizer.next_token(), Token::Characters("<a>&amp;".into()));
        assert_eq!(tokenizer.next_token(), Token::Eof);
    }

    #[test]
    fn positions_track_lines() {
        let mut tokenizer = Tokenizer::new("a\nb");
        let _ = tokenizer.next_token();
        // 0-based: the second line is line 1.
        assert_eq!(tokenizer.position().line, 1);
    }
}

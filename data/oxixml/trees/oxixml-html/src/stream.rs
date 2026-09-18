//! Parsing a document that arrives in pieces.
//!
//! [`parse_document`](crate::parse_document) and its siblings need the whole
//! input up front. [`StreamingParser`] does not: it takes the document a chunk
//! at a time — as it comes off a socket, a decompressor or a file reader — and
//! runs tokenization *and* tree construction over each chunk as it arrives, so
//! that the [`Document`](oxixml_dom::Document) is already most of the way
//! built by the time the last byte turns up.
//!
//! It is the same state machine, not a second one. A chunk boundary is
//! invisible to it: a tag, a comment, a character reference, a `<!--`, a
//! `<![CDATA[` or a multi-byte character split across two `write` calls parses
//! exactly as it would have if the two chunks had been one, and the resulting
//! tree is identical to what the one-shot entry points produce for the same
//! bytes. Feeding a document one byte at a time is a legitimate — if slow —
//! way to parse it.
//!
//! # Example
//!
//! ```
//! use oxixml_html::{StreamingParser, dump_tree, parse_document};
//!
//! let mut parser = StreamingParser::new();
//! // The `<p` and the `>` arrive in different chunks, as does the `&am` of
//! // `&amp;` and the two halves of the `é`.
//! parser.write(b"<!DOCTYPE html><p");
//! parser.write(b" class=x>caf\xc3");
//! parser.write(b"\xa9 &am");
//! parser.write(b"p; cream");
//! let parsed = parser.finish();
//!
//! assert_eq!(
//!     dump_tree(&parsed),
//!     dump_tree(&parse_document("<!DOCTYPE html><p class=x>caf\u{e9} &amp; cream"))
//! );
//! ```
//!
//! # What it buffers
//!
//! Tokens are never accumulated: each one is handed to tree construction the
//! moment it is produced, and the nodes it creates are attached to the
//! document straight away. Input text is only retained until the tokenizer has
//! consumed it — after every `write` the consumed prefix is discarded — so
//! what stays in memory is the unfinished tail of the current construct, plus
//! whatever a still-open element holds, plus the document itself.
//! [`StreamingParser::buffered_bytes`] reports the first of those.
//!
//! # Encodings
//!
//! [`StreamingParser::new`] reads UTF-8, [`StreamingParser::with_encoding`]
//! reads whatever encoding the caller already knows, and
//! [`StreamingParser::sniffing`] runs the same byte order mark and
//! `<meta charset>` prescan that [`parse_bytes`](crate::parse_bytes) does —
//! see that constructor for the one way in which a stream differs from a
//! buffer there.

use oxixml_dom::NodeId;
use oxixml_encoding::{Decoder, Encoding};

use crate::encoding::sniff_encoding;
use crate::ns::Namespace;
use crate::parse::{
    ParseOptions, ParseOutput, begin_fragment, configure, finish_document, finish_fragment,
};
use crate::tree::TreeBuilder;

/// An HTML parser that is fed its input in chunks.
///
/// See the [module documentation](self) for the guarantees, the buffering
/// behaviour and an example.
pub struct StreamingParser {
    builder: TreeBuilder<'static>,
    input: Input,
    /// The `html` element a fragment parse builds into, and `None` for a
    /// document parse; see [`Self::fragment`].
    fragment_root: Option<NodeId>,
}

/// How incoming bytes become characters.
enum Input {
    /// The encoding is settled and every chunk goes straight through this
    /// decoder.
    Decoding(Decoder),
    /// The encoding is not settled yet: bytes are held here until there are
    /// enough of them to prescan, or until the stream ends.
    Sniffing {
        buffer: Vec<u8>,
        transport_charset: Option<String>,
    },
}

impl StreamingParser {
    /// How many bytes [`Self::sniffing`] buffers before it commits to an
    /// encoding.
    ///
    /// This is the read-ahead
    /// [HTML § 13.2.3.2](https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding)
    /// recommends for exactly this situation — an implementation that cannot
    /// see the whole byte stream at once and must start decoding before it
    /// ends.
    pub const PRESCAN_LIMIT: usize = 1024;

    /// A streaming parser for a whole document encoded in UTF-8.
    ///
    /// Bytes that are not valid UTF-8 are replaced by `U+FFFD`, one per
    /// maximal ill-formed subsequence, which is what the Encoding Standard's
    /// decoder does and what [`parse_bytes`](crate::parse_bytes) would do for
    /// the same input.
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(&ParseOptions::new())
    }

    /// A streaming parser for a whole UTF-8 document, with explicit options.
    #[must_use]
    pub fn with_options(options: &ParseOptions) -> Self {
        Self::for_document(Input::Decoding(Encoding::Utf8.new_decoder()), options)
    }

    /// A streaming parser for a whole document in a known encoding.
    ///
    /// Use this when the encoding is settled before the first byte arrives —
    /// a `Content-Type` charset, a byte order mark the caller has already
    /// stripped, or a format that fixes it.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_encoding::Encoding;
    /// use oxixml_html::{HTML_NAMESPACE, ParseOptions, StreamingParser};
    ///
    /// let mut parser =
    ///     StreamingParser::with_encoding(Encoding::Windows1252, &ParseOptions::new());
    /// // windows-1252 byte 0x93 is a left double quotation mark.
    /// parser.write(b"<p>\x93quo");
    /// parser.write(b"ted\x94");
    /// let parsed = parser.finish();
    ///
    /// let document = parsed.document();
    /// let paragraph = document
    ///     .get_elements_by_tag_name_ns(document.root(), Some(HTML_NAMESPACE), "p")
    ///     .next()
    ///     .expect("a p element");
    /// assert_eq!(
    ///     document.text_content(paragraph).as_deref(),
    ///     Some("\u{201c}quoted\u{201d}")
    /// );
    /// ```
    #[must_use]
    pub fn with_encoding(encoding: Encoding, options: &ParseOptions) -> Self {
        Self::for_document(Input::Decoding(encoding.new_decoder()), options)
    }

    /// A streaming parser that determines the encoding itself, from a byte
    /// order mark, the transport layer's `charset` and the `<meta charset>`
    /// prescan of
    /// [HTML § 13.2.3.2](https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding).
    ///
    /// Sniffing needs to look at the start of the stream before it can decode
    /// any of it, so the first [`Self::PRESCAN_LIMIT`] bytes are buffered
    /// rather than parsed; once that many have arrived (or the stream ends
    /// first, whichever comes sooner) the encoding is chosen and parsing
    /// begins.
    ///
    /// **The one difference from [`parse_bytes`](crate::parse_bytes).** That
    /// function is handed the whole document, so its prescan has no read-ahead
    /// budget to respect and finds a `<meta charset>` however far into the
    /// file it is. A stream has to commit earlier: a `<meta charset>` that
    /// only appears after the first [`Self::PRESCAN_LIMIT`] bytes is missed
    /// here, and the default encoding is used instead. Nor is the encoding
    /// ever revised afterwards: HTML's "change the encoding" step restarts the
    /// parse from the beginning with the new encoding, which a stream whose
    /// earlier bytes are gone cannot do. Pass the encoding to
    /// [`Self::with_encoding`] when it is known from elsewhere, or use
    /// [`parse_bytes`](crate::parse_bytes) when the whole document is in
    /// memory anyway.
    #[must_use]
    pub fn sniffing(transport_charset: Option<&str>, options: &ParseOptions) -> Self {
        Self::for_document(
            Input::Sniffing {
                buffer: Vec::new(),
                transport_charset: transport_charset.map(str::to_owned),
            },
            options,
        )
    }

    /// A streaming parser for an HTML fragment in the context of an element,
    /// as [`parse_fragment_with`](crate::parse_fragment_with) is for a
    /// complete one. The input is UTF-8.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_html::{Namespace, ParseOptions, StreamingParser, dump_tree};
    ///
    /// let mut parser =
    ///     StreamingParser::fragment(Namespace::Html, "tr", &ParseOptions::new());
    /// parser.write(b"<td>ce");
    /// parser.write(b"ll");
    /// assert_eq!(dump_tree(&parser.finish()), "| <td>\n|   \"cell\"\n");
    /// ```
    #[must_use]
    pub fn fragment(namespace: Namespace, context: &str, options: &ParseOptions) -> Self {
        let mut builder = TreeBuilder::streaming(options.scripting(), options.max_depth());
        configure(&mut builder, options);
        let root = begin_fragment(&mut builder, namespace, context, options);
        Self {
            builder,
            input: Input::Decoding(Encoding::Utf8.new_decoder()),
            fragment_root: Some(root),
        }
    }

    /// The shared part of every whole-document constructor.
    fn for_document(input: Input, options: &ParseOptions) -> Self {
        let mut builder = TreeBuilder::streaming(options.scripting(), options.max_depth());
        configure(&mut builder, options);
        Self {
            builder,
            input,
            fragment_root: None,
        }
    }

    /// Feeds the next chunk of the document.
    ///
    /// The chunk may end anywhere at all — in the middle of a tag, of a
    /// character reference, of a comment, or of a multi-byte character. What
    /// cannot be decided yet is held until the next chunk (or until
    /// [`Self::finish`]) settles it.
    ///
    /// There is no "feeding after the end" to worry about: only
    /// [`Self::finish`] ends the document, and it consumes the parser.
    pub fn write(&mut self, chunk: &[u8]) {
        match &mut self.input {
            Input::Decoding(decoder) => {
                let mut text = String::with_capacity(chunk.len());
                decoder.decode_to_string_lossy(chunk, &mut text, false);
                self.builder.tokenizer.feed(&text);
            }
            Input::Sniffing {
                buffer,
                transport_charset,
            } => {
                buffer.extend_from_slice(chunk);
                if buffer.len() < Self::PRESCAN_LIMIT {
                    return;
                }
                let buffered = std::mem::take(buffer);
                let transport = transport_charset.take();
                self.settle(&buffered, transport.as_deref());
            }
        }
        self.pump();
    }

    /// Feeds the next chunk of the document as text that has already been
    /// decoded.
    ///
    /// This is the same as [`Self::write`] except that no decoding step is
    /// involved. A parser that is still [sniffing](Self::sniffing) has no
    /// decoded text to append to yet, so for that one case the text is fed
    /// through the decoder as UTF-8 bytes, which is what the prescan then
    /// sees. The two may otherwise be interleaved freely, as long as a
    /// multi-byte character is not split across one of each.
    pub fn write_str(&mut self, chunk: &str) {
        match &self.input {
            Input::Decoding(_) => {
                self.builder.tokenizer.feed(chunk);
                self.pump();
            }
            Input::Sniffing { .. } => self.write(chunk.as_bytes()),
        }
    }

    /// The document as it stands.
    ///
    /// A streaming parse attaches each node as soon as the token that creates
    /// it comes out of the tokenizer, so this is not an empty shell that fills
    /// up at the end: everything the chunks written so far have produced is
    /// already in it, and only the elements still open when the last chunk ran
    /// out are missing their remaining children.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_html::{HTML_NAMESPACE, StreamingParser};
    ///
    /// let mut parser = StreamingParser::new();
    /// parser.write(b"<p>first</p><p>second");
    ///
    /// // Both paragraphs are in the tree already, before `finish`.
    /// let document = parser.document();
    /// assert_eq!(
    ///     document
    ///         .get_elements_by_tag_name_ns(document.root(), Some(HTML_NAMESPACE), "p")
    ///         .count(),
    ///     2
    /// );
    /// ```
    #[must_use]
    pub const fn document(&self) -> &oxixml_dom::Document {
        &self.builder.document
    }

    /// The number of bytes of decoded input the parser is still holding: the
    /// tail of the stream the tokenizer has not consumed yet, plus anything
    /// buffered for the encoding prescan.
    ///
    /// This is what a caller applying backpressure wants to watch. It stays
    /// small — a few bytes between constructs, growing only while a single
    /// construct (a very long comment, a very long run of text) is still
    /// unfinished.
    #[must_use]
    pub fn buffered_bytes(&self) -> usize {
        let sniffing = match &self.input {
            Input::Decoding(_) => 0,
            Input::Sniffing { buffer, .. } => buffer.len(),
        };
        sniffing + self.builder.tokenizer.buffered_bytes()
    }

    /// Signals the end of the document and returns the parse result, running
    /// the same end-of-file tree construction steps the one-shot entry points
    /// run.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_html::{StreamingParser, dump_tree, parse_document};
    ///
    /// // A stream with nothing in it is still a document.
    /// assert_eq!(
    ///     dump_tree(&StreamingParser::new().finish()),
    ///     dump_tree(&parse_document(""))
    /// );
    /// ```
    #[must_use]
    pub fn finish(mut self) -> ParseOutput {
        if let Input::Sniffing {
            buffer,
            transport_charset,
        } = &mut self.input
        {
            let buffered = std::mem::take(buffer);
            let transport = transport_charset.take();
            self.settle(&buffered, transport.as_deref());
        }
        if let Input::Decoding(decoder) = &mut self.input {
            let mut text = String::new();
            decoder.decode_to_string_lossy(&[], &mut text, true);
            self.builder.tokenizer.feed(&text);
        }
        self.builder.tokenizer.close_input();
        let finished = self.builder.pump();
        debug_assert!(
            finished,
            "the tokenizer's input was closed on the line above, so the pump cannot stop short \
             of the end-of-file token"
        );
        match self.fragment_root {
            Some(root) => finish_fragment(self.builder, root),
            None => finish_document(self.builder),
        }
    }

    /// Chooses an encoding for the bytes buffered so far, switches to
    /// decoding, and feeds those bytes through it.
    fn settle(&mut self, buffered: &[u8], transport_charset: Option<&str>) {
        let sniffed = sniff_encoding(buffered, transport_charset);
        let mut decoder = sniffed.encoding.new_decoder();
        let mut text = String::with_capacity(buffered.len());
        decoder.decode_to_string_lossy(&buffered[sniffed.bom_length..], &mut text, false);
        self.builder.tokenizer.feed(&text);
        self.input = Input::Decoding(decoder);
    }

    /// Runs tokenization and tree construction over everything fed in so far,
    /// then releases the input it consumed.
    fn pump(&mut self) {
        let _ = self.builder.pump();
        self.builder.tokenizer.discard_consumed();
    }
}

impl Default for StreamingParser {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StreamingParser {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StreamingParser")
            .field("buffered_bytes", &self.buffered_bytes())
            .field("is_fragment", &self.fragment_root.is_some())
            .finish()
    }
}

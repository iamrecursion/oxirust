//! The [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) streaming parser.

#[cfg(feature = "async-tokio")]
use std::io::Cursor;
use std::io::{BufRead, BufReader, Read};
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, AsyncReadExt};

use oxixml_model::{BlankNode, GraphName, Literal, NamedNode, NamedOrBlankNode, Quad, Term};
use oxixml_qname::is_ncname;
use oxixml_xml::XmlVersion;
use oxixml_xml::escape::resolve_predefined_entity;
use oxixml_xml::events::{BytesStart, Event};
use oxixml_xml::name::{Namespace, QName, ResolveResult};
use oxixml_xml::reader::NsReader;

use crate::error::{TriXParseError, TriXSyntaxError};

/// The [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) XML namespace.
pub const TRIX_NAMESPACE: &str = "http://www.w3.org/2004/03/trix/trix-1/";

const ELEM_ROOT: &[u8] = b"TriX";
const ELEM_GRAPH: &[u8] = b"graph";
const ELEM_TRIPLE: &[u8] = b"triple";
const ELEM_URI: &[u8] = b"uri";
const ELEM_ID: &[u8] = b"id";
const ELEM_PLAIN_LITERAL: &[u8] = b"plainLiteral";
const ELEM_TYPED_LITERAL: &[u8] = b"typedLiteral";
const ATTR_LANG: &[u8] = b"xml:lang";
const ATTR_DATATYPE: &[u8] = b"datatype";

/// A [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) streaming parser.
///
/// Builds either a [`SliceTriXParser`] (from an in-memory buffer) or a
/// [`ReaderTriXParser`] (from a [`Read`] implementation), both of which implement
/// [`Iterator<Item = Result<Quad, TriXParseError>>`](Iterator).
///
/// # Example
///
/// ```
/// use oxixml_trix::TriXParser;
///
/// let file = r#"<?xml version="1.0" encoding="utf-8"?>
/// <TriX xmlns="http://www.w3.org/2004/03/trix/trix-1/">
///   <graph>
///     <uri>http://example.com/graph1</uri>
///     <triple>
///       <uri>http://example.com/s</uri>
///       <uri>http://example.com/p</uri>
///       <plainLiteral xml:lang="en">hello</plainLiteral>
///     </triple>
///   </graph>
/// </TriX>"#;
///
/// let quads = TriXParser::new()
///     .for_slice(file)
///     .collect::<Result<Vec<_>, _>>()?;
/// assert_eq!(quads.len(), 1);
/// assert_eq!(quads[0].graph_name.to_string(), "<http://example.com/graph1>");
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default, Clone, Copy)]
#[must_use]
pub struct TriXParser {
    _private: (),
}

impl TriXParser {
    /// Builds a new [`TriXParser`] with the default configuration.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses a TriX document from a [`Read`] implementation.
    pub fn for_reader<R: Read>(self, reader: R) -> ReaderTriXParser<R> {
        ReaderTriXParser {
            inner: InternalTriXParser::new(BufReader::new(reader)),
        }
    }

    /// Parses a TriX document from a [`AsyncRead`] implementation.
    ///
    /// The asynchronous counterpart of [`for_reader`](Self::for_reader), taking the same
    /// (non-`async`) shape so umbrella crates can forward to it uniformly.
    ///
    /// <div class="warning">
    ///
    /// Unlike the Turtle-family and RDF/XML async parsers, this one is **not** incrementally
    /// streaming: the TriX state machine is written against [`BufRead`] and reads several XML
    /// events per term, so it cannot be suspended mid-term. The document is therefore drained
    /// from the reader on the first [`next`](TokioAsyncReaderTriXParser::next) call — awaiting
    /// only the I/O — and then parsed by the very same core the blocking path uses, giving
    /// identical quads and errors at the cost of buffering the input.</div>
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_reader<R: AsyncRead + Unpin>(
        self,
        reader: R,
    ) -> TokioAsyncReaderTriXParser<R> {
        TokioAsyncReaderTriXParser {
            reader: Some(reader),
            inner: None,
        }
    }

    /// Parses a TriX document from a byte slice (or `&str`).
    pub fn for_slice(self, slice: &(impl AsRef<[u8]> + ?Sized)) -> SliceTriXParser<'_> {
        SliceTriXParser {
            inner: InternalTriXParser::new(slice.as_ref()),
        }
    }
}

/// Parses a TriX document from a [`Read`] implementation.
///
/// Can be built using [`TriXParser::for_reader`].
#[must_use]
pub struct ReaderTriXParser<R: Read> {
    inner: InternalTriXParser<BufReader<R>>,
}

impl<R: Read> Iterator for ReaderTriXParser<R> {
    type Item = Result<Quad, TriXParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_quad().transpose()
    }
}

/// Parses a TriX document from a [`AsyncRead`] implementation.
///
/// Can be built using [`TriXParser::for_tokio_async_reader`]. See that method for why the
/// document is buffered rather than streamed.
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncReaderTriXParser<R: AsyncRead + Unpin> {
    /// The source, taken when the document is drained. `None` afterwards.
    reader: Option<R>,
    /// The running parse state, created once the document has been drained.
    inner: Option<InternalTriXParser<Cursor<Vec<u8>>>>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderTriXParser<R> {
    /// Reads the next quad or returns `None` if the document is finished.
    pub async fn next(&mut self) -> Option<Result<Quad, TriXParseError>> {
        if let Some(mut reader) = self.reader.take() {
            let mut buffer = Vec::new();
            if let Err(error) = reader.read_to_end(&mut buffer).await {
                return Some(Err(TriXParseError::Io(error)));
            }
            self.inner = Some(InternalTriXParser::new(Cursor::new(buffer)));
        }
        self.inner.as_mut()?.next_quad().transpose()
    }
}

/// Parses a TriX document from a byte slice.
///
/// Can be built using [`TriXParser::for_slice`].
#[must_use]
pub struct SliceTriXParser<'a> {
    inner: InternalTriXParser<&'a [u8]>,
}

impl Iterator for SliceTriXParser<'_> {
    type Item = Result<Quad, TriXParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_quad().transpose()
    }
}

/// A resolved RDF term read from a single TriX term element (`<uri>`, `<id>`,
/// `<plainLiteral>` or `<typedLiteral>`).
enum ParsedTerm {
    Named(NamedNode),
    Blank(BlankNode),
    Literal(Literal),
}

impl ParsedTerm {
    fn into_term(self) -> Term {
        match self {
            Self::Named(n) => n.into(),
            Self::Blank(b) => b.into(),
            Self::Literal(l) => l.into(),
        }
    }

    fn into_named_or_blank(self) -> Option<NamedOrBlankNode> {
        match self {
            Self::Named(n) => Some(n.into()),
            Self::Blank(b) => Some(b.into()),
            Self::Literal(_) => None,
        }
    }

    fn into_named(self) -> Option<NamedNode> {
        match self {
            Self::Named(n) => Some(n),
            Self::Blank(_) | Self::Literal(_) => None,
        }
    }
}

/// A frame in the (bounded-depth) TriX grammar stack: `TriX` > `graph` > `triple`.
enum Frame {
    /// Inside the root `<TriX>` element, waiting for `<graph>` children.
    Root,
    /// Inside a `<graph>` element. `name_locked` becomes `true` once either a name
    /// element (`<uri>`/`<id>`) or the first `<triple>` has been read, at which
    /// point a later name element is rejected.
    Graph { name: GraphName, name_locked: bool },
    /// Inside a `<triple>` element, having already read `terms.len()` (0 to 3)
    /// child term elements.
    Triple {
        graph_name: GraphName,
        terms: Vec<ParsedTerm>,
    },
}

/// A cheap, owned summary of `stack.last()`, computed once per step to avoid
/// holding a borrow of the stack across the mutations that follow.
#[derive(Clone, Copy)]
enum StackTop {
    /// Nothing has been read yet: the next element must be the document root.
    Empty,
    Root,
    Graph {
        name_locked: bool,
    },
    Triple,
}

fn stack_top_kind(stack: &[Frame]) -> StackTop {
    match stack.last() {
        None => StackTop::Empty,
        Some(Frame::Root) => StackTop::Root,
        Some(Frame::Graph { name_locked, .. }) => StackTop::Graph {
            name_locked: *name_locked,
        },
        Some(Frame::Triple { .. }) => StackTop::Triple,
    }
}

/// The shared streaming engine, generic over any [`BufRead`] source (both `&[u8]`
/// and `BufReader<R>` implement it, so slice- and reader-based parsing share one
/// implementation).
struct InternalTriXParser<R: BufRead> {
    reader: NsReader<R>,
    buffer: Vec<u8>,
    stack: Vec<Frame>,
    xml_version: XmlVersion,
    started: bool,
    finished: bool,
    pending: Option<Quad>,
}

impl<R: BufRead> InternalTriXParser<R> {
    fn new(inner: R) -> Self {
        let mut reader = NsReader::from_reader(inner);
        // Keep `Start`/`End` and `Empty` distinguishable: the engine relies on
        // `Empty` never producing a matching `End` event.
        reader.config_mut().expand_empty_elements = false;
        Self {
            reader,
            buffer: Vec::new(),
            stack: Vec::new(),
            xml_version: XmlVersion::Implicit1_0,
            started: false,
            finished: false,
            pending: None,
        }
    }

    /// Drives the state machine forward until a [`Quad`] is produced or the
    /// document is exhausted.
    fn next_quad(&mut self) -> Result<Option<Quad>, TriXParseError> {
        loop {
            if let Some(quad) = self.pending.take() {
                return Ok(Some(quad));
            }
            if self.finished {
                return Ok(None);
            }
            self.step()?;
        }
    }

    fn position(&self) -> u64 {
        self.reader.buffer_position()
    }

    fn err(&self, message: impl Into<String>) -> TriXParseError {
        TriXSyntaxError::msg(self.position(), message).into()
    }

    fn xml_err(&self, error: oxixml_xml::errors::Error) -> TriXParseError {
        if let oxixml_xml::errors::Error::Io(io_error) = &error {
            return TriXParseError::Io(std::io::Error::new(io_error.kind(), io_error.to_string()));
        }
        TriXSyntaxError::xml(self.reader.error_position(), error.to_string()).into()
    }

    fn wrap_leaf_error(&self, error: LeafTextError) -> TriXParseError {
        match error {
            LeafTextError::Xml(e) => self.xml_err(e),
            LeafTextError::Message(position, msg) => TriXSyntaxError::msg(position, msg).into(),
        }
    }

    fn on_end(&mut self) -> Result<(), TriXParseError> {
        match self.stack.pop() {
            None => Err(self.err("unexpected closing tag")),
            Some(Frame::Root | Frame::Graph { .. }) => Ok(()),
            Some(Frame::Triple { graph_name, terms }) => {
                let quad = build_quad(graph_name, terms, self.position())?;
                self.pending = Some(quad);
                Ok(())
            }
        }
    }

    fn on_eof(&mut self) -> Result<(), TriXParseError> {
        // Set unconditionally, before any of the checks below can return early: the
        // parser is done reading input either way (there is no more input to read),
        // so every path -- success or error -- must leave `finished` set. Otherwise
        // the next `next_quad()` call would re-enter `step()`, read `Event::Eof`
        // again, and call `on_eof()` again, forever re-producing the same `Err` and
        // never reaching the `if self.finished { return Ok(None); }` fast path.
        self.finished = true;
        if !self.started {
            return Err(self.err("missing document element: expected the root <TriX> element"));
        }
        if !self.stack.is_empty() {
            return Err(self.err("unexpected end of file: an element was not closed"));
        }
        Ok(())
    }

    /// Reads and processes exactly one XML event, updating `self.stack` and
    /// possibly setting `self.pending` or `self.finished`.
    ///
    /// Any data borrowed from `self.buffer` (the just-read event) is fully
    /// converted to owned values *before* this function ever calls another method
    /// on `self`, and before `self.buffer` is reused by [`read_leaf_text`]: the
    /// underlying XML events are not `'static`, so mixing a live borrow of them
    /// with a fresh `&mut self`/`&self` call would not borrow-check.
    fn step(&mut self) -> Result<(), TriXParseError> {
        self.buffer.clear();
        // The two conversions below deliberately reference `self.reader` (a field
        // projection) rather than call `self.xml_err`/`self.err` (which would need
        // to borrow the whole of `self`, including `self.buffer`, conflicting with
        // the `&mut self.buffer` borrow that is still needed for `event`).
        let event = self
            .reader
            .read_event_into(&mut self.buffer)
            .map_err(|e| xml_parse_error(&self.reader, e))?;
        let (start, is_empty) = match event {
            Event::Decl(decl) => {
                self.xml_version = decl.xml_version().map_err(|e| self.xml_err(e))?;
                return Ok(());
            }
            Event::Comment(_) | Event::PI(_) => return Ok(()),
            Event::DocType(_) => {
                return Err(self.err("a <!DOCTYPE> declaration is not allowed in a TriX document"));
            }
            Event::CData(_) => {
                return Err(self.err("unexpected CDATA section outside of a term element"));
            }
            Event::Text(text) => {
                let has_content = text
                    .decode()
                    .map_err(|e| syntax_error(&self.reader, e.to_string()))?
                    .chars()
                    .any(|c| !c.is_whitespace());
                return if has_content {
                    Err(self.err("unexpected text content outside of a term element"))
                } else {
                    Ok(())
                };
            }
            Event::GeneralRef(reference) => {
                let is_whitespace = reference
                    .resolve_char_ref()
                    .map_err(|e| self.err(e.to_string()))?
                    .is_some_and(char::is_whitespace);
                return if is_whitespace {
                    Ok(())
                } else {
                    Err(self.err("unexpected character reference outside of a term element"))
                };
            }
            Event::End(_) => return self.on_end(),
            Event::Eof => return self.on_eof(),
            Event::Start(start) => (start, false),
            Event::Empty(start) => (start, true),
        };

        // Fully consume `start` into owned data before doing anything else: no
        // method is called on `self` until every piece of information taken from
        // this element (its resolved name and every attribute we might need) has
        // been copied out.
        let qname_bytes = start.name().as_ref().to_vec();
        let (resolved, local) = self.reader.resolve_element(QName(qname_bytes.as_slice()));
        let in_trix_namespace = is_trix_namespace(&resolved);
        let local_name = local.as_ref().to_vec();
        let lang_attr_result = extract_attr_value(&start, ATTR_LANG, &self.reader);
        let datatype_attr_result = extract_attr_value(&start, ATTR_DATATYPE, &self.reader);
        // `start` (and everything borrowed from `self.buffer`) is no longer used
        // beyond this point.

        let lang_attr = lang_attr_result.map_err(|e| self.xml_err(e))?;
        let datatype_attr = datatype_attr_result.map_err(|e| self.xml_err(e))?;
        if !in_trix_namespace {
            return Err(self.err(format!(
                "element `{}` is not in the TriX namespace `{TRIX_NAMESPACE}`",
                String::from_utf8_lossy(&qname_bytes),
            )));
        }

        match stack_top_kind(&self.stack) {
            StackTop::Empty => {
                if local_name != ELEM_ROOT {
                    return Err(self.err(format!(
                        "expected the root <TriX> element, found <{}>",
                        String::from_utf8_lossy(&local_name),
                    )));
                }
                self.started = true;
                if !is_empty {
                    self.stack.push(Frame::Root);
                }
                Ok(())
            }
            StackTop::Root => {
                if local_name != ELEM_GRAPH {
                    return Err(self.err(format!(
                        "expected a <graph> element inside <TriX>, found <{}>",
                        String::from_utf8_lossy(&local_name),
                    )));
                }
                if !is_empty {
                    self.stack.push(Frame::Graph {
                        name: GraphName::DefaultGraph,
                        name_locked: false,
                    });
                }
                Ok(())
            }
            StackTop::Graph { name_locked } => {
                self.on_graph_child(local_name, is_empty, name_locked)
            }
            StackTop::Triple => {
                self.on_triple_child(local_name, is_empty, lang_attr, datatype_attr)
            }
        }
    }

    fn on_graph_child(
        &mut self,
        local_name: Vec<u8>,
        is_empty: bool,
        name_locked: bool,
    ) -> Result<(), TriXParseError> {
        match local_name.as_slice() {
            ELEM_URI | ELEM_ID => {
                if name_locked {
                    return Err(self.err(
                        "a graph name element (<uri> or <id>) must be the first child of <graph>",
                    ));
                }
                let kind = if local_name == ELEM_URI {
                    TermElemKind::Uri
                } else {
                    TermElemKind::Id
                };
                // A self-closing `<uri/>`/`<id/>` never yields a matching `End` event (the reader
                // runs with `expand_empty_elements = false`), so `read_leaf_text` must not be
                // called for it: it would swallow the following sibling's events instead.
                let value = if is_empty {
                    String::new()
                } else {
                    read_leaf_text(&mut self.reader, &mut self.buffer, self.xml_version)
                        .map_err(|e| self.wrap_leaf_error(e))?
                };
                let term = build_term(kind, value, self.position())?;
                let name: GraphName = term
                    .into_named_or_blank()
                    .ok_or_else(|| {
                        self.err("a graph name must be a <uri> or an <id>, not a literal")
                    })?
                    .into();
                if let Some(Frame::Graph {
                    name: slot,
                    name_locked: locked,
                }) = self.stack.last_mut()
                {
                    *slot = name;
                    *locked = true;
                }
                Ok(())
            }
            ELEM_TRIPLE => {
                if is_empty {
                    return Err(
                        self.err("a <triple> element must contain exactly 3 child terms, found 0")
                    );
                }
                let graph_name = match self.stack.last_mut() {
                    Some(Frame::Graph {
                        name,
                        name_locked: locked,
                    }) => {
                        *locked = true;
                        name.clone()
                    }
                    _ => GraphName::DefaultGraph,
                };
                self.stack.push(Frame::Triple {
                    graph_name,
                    terms: Vec::with_capacity(3),
                });
                Ok(())
            }
            _ => Err(self.err(format!(
                "expected a graph name element (<uri>, <id>) or a <triple>, found <{}>",
                String::from_utf8_lossy(&local_name),
            ))),
        }
    }

    fn on_triple_child(
        &mut self,
        local_name: Vec<u8>,
        is_empty: bool,
        lang_attr: Option<String>,
        datatype_attr: Option<String>,
    ) -> Result<(), TriXParseError> {
        let kind = match local_name.as_slice() {
            ELEM_URI => TermElemKind::Uri,
            ELEM_ID => TermElemKind::Id,
            ELEM_PLAIN_LITERAL => TermElemKind::PlainLiteral(lang_attr),
            ELEM_TYPED_LITERAL => {
                let datatype = datatype_attr.ok_or_else(|| {
                    self.err("a <typedLiteral> element requires a `datatype` attribute")
                })?;
                TermElemKind::TypedLiteral(datatype)
            }
            _ => {
                return Err(self.err(format!(
                    "expected a term element (<uri>, <id>, <plainLiteral> or <typedLiteral>) inside <triple>, found <{}>",
                    String::from_utf8_lossy(&local_name),
                )));
            }
        };
        // A self-closing term element (`<plainLiteral/>`, `<uri/>`, ...) is a legitimate,
        // spec-legal XML spelling of an element with empty content. It produces a single `Empty`
        // event with no matching `End` (`expand_empty_elements` is off), so `read_leaf_text` —
        // which loops until it sees an `End` — must be skipped: otherwise it consumes the
        // following sibling's events and desynchronizes the whole parse.
        let value = if is_empty {
            String::new()
        } else {
            read_leaf_text(&mut self.reader, &mut self.buffer, self.xml_version)
                .map_err(|e| self.wrap_leaf_error(e))?
        };
        let term = build_term(kind, value, self.position())?;
        match self.stack.last_mut() {
            Some(Frame::Triple { terms, .. }) => {
                if terms.len() >= 3 {
                    return Err(self.err(
                        "a <triple> element must contain exactly 3 child terms, found more than 3",
                    ));
                }
                terms.push(term);
                Ok(())
            }
            _ => Err(self.err("internal error: term element read outside of a <triple>")),
        }
    }
}

/// The kind of RDF term a `<uri>`/`<id>`/`<plainLiteral>`/`<typedLiteral>` element
/// represents, together with any attribute already extracted from its start tag.
enum TermElemKind {
    Uri,
    Id,
    PlainLiteral(Option<String>),
    TypedLiteral(String),
}

/// Converts an [`oxixml_xml::errors::Error`] into a [`TriXParseError`], reading the reported
/// position off `reader`.
///
/// A free function (rather than a method taking `&self`) so that it can be called from inside a
/// closure that must only capture the `reader` field, not the whole parser: at the call sites
/// that need it, `self.buffer` is still mutably borrowed by the very expression being mapped.
fn xml_parse_error<R: BufRead>(
    reader: &NsReader<R>,
    error: oxixml_xml::errors::Error,
) -> TriXParseError {
    if let oxixml_xml::errors::Error::Io(io_error) = &error {
        return TriXParseError::Io(std::io::Error::new(io_error.kind(), io_error.to_string()));
    }
    TriXSyntaxError::xml(reader.error_position(), error.to_string()).into()
}

/// Builds a free-form [`TriXParseError`] at `reader`'s current buffer position.
///
/// See [`xml_parse_error`] for why this is a free function.
fn syntax_error<R: BufRead>(reader: &NsReader<R>, message: impl Into<String>) -> TriXParseError {
    TriXSyntaxError::msg(reader.buffer_position(), message).into()
}

fn is_trix_namespace(resolved: &ResolveResult<'_>) -> bool {
    matches!(resolved, ResolveResult::Bound(Namespace(ns)) if *ns == TRIX_NAMESPACE.as_bytes())
}

/// Reads a (possibly absent) attribute value, normalizing entities. Returns
/// `Ok(None)` if the attribute is not present.
fn extract_attr_value<R: BufRead>(
    start: &BytesStart<'_>,
    name: &[u8],
    reader: &NsReader<R>,
) -> oxixml_xml::errors::Result<Option<String>> {
    let Some(attr) = start.try_get_attribute(name)? else {
        return Ok(None);
    };
    let value = attr.decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())?;
    Ok(Some(value.into_owned()))
}

/// An error while accumulating a leaf term element's text content, not yet tied to
/// a [`TriXSyntaxError`] position (the XML variant carries its own).
enum LeafTextError {
    Xml(oxixml_xml::errors::Error),
    Message(u64, String),
}

impl From<oxixml_xml::errors::Error> for LeafTextError {
    fn from(error: oxixml_xml::errors::Error) -> Self {
        Self::Xml(error)
    }
}

impl From<oxixml_xml::encoding::EncodingError> for LeafTextError {
    fn from(error: oxixml_xml::encoding::EncodingError) -> Self {
        Self::Xml(error.into())
    }
}

/// Reads the text content of a leaf term element (`<uri>`, `<id>`,
/// `<plainLiteral>` or `<typedLiteral>`) after its `Start` event has been
/// consumed, resolving character references and predefined entities, and
/// rejecting any nested child element.
///
/// The caller must have fully extracted any data it needs from the element's
/// `Start` event *before* calling this function, since it repeatedly clears and
/// refills `buffer`.
fn read_leaf_text<R: BufRead>(
    reader: &mut NsReader<R>,
    buffer: &mut Vec<u8>,
    xml_version: XmlVersion,
) -> Result<String, LeafTextError> {
    let mut out = String::new();
    loop {
        buffer.clear();
        match reader.read_event_into(buffer)? {
            Event::Text(text) => out.push_str(&text.xml_content(xml_version)?),
            Event::CData(cdata) => out.push_str(&cdata.xml_content(xml_version)?),
            Event::GeneralRef(reference) => {
                if let Some(ch) = reference.resolve_char_ref()? {
                    out.push(ch);
                } else {
                    let name = reference.decode()?;
                    match resolve_predefined_entity(&name) {
                        Some(resolved) => out.push_str(resolved),
                        None => {
                            return Err(LeafTextError::Message(
                                reader.buffer_position(),
                                format!("unknown entity reference `&{name};`"),
                            ));
                        }
                    }
                }
            }
            Event::Comment(_) | Event::PI(_) => {}
            Event::End(_) => return Ok(out),
            Event::Start(_) | Event::Empty(_) => {
                return Err(LeafTextError::Message(
                    reader.buffer_position(),
                    "a TriX term element must not contain nested elements".to_owned(),
                ));
            }
            Event::Eof => {
                return Err(LeafTextError::Message(
                    reader.buffer_position(),
                    "unexpected end of file inside a term element".to_owned(),
                ));
            }
            Event::Decl(_) | Event::DocType(_) => {
                return Err(LeafTextError::Message(
                    reader.buffer_position(),
                    "unexpected declaration inside a term element".to_owned(),
                ));
            }
        }
    }
}

/// Builds a [`ParsedTerm`] from a resolved [`TermElemKind`] and its (already
/// unescaped) text content.
///
/// Blank node ids are scoped per document: the same `<id>` string always maps to
/// an equal [`BlankNode`], because the id's literal text is used directly as the
/// blank node's internal label. This holds for one parse ([`SliceTriXParser`] or
/// [`ReaderTriXParser`]) at a time; merging quads parsed from multiple, separate
/// TriX documents is the caller's responsibility, same as for every other RDF
/// syntax in this workspace.
fn build_term(
    kind: TermElemKind,
    value: String,
    position: u64,
) -> Result<ParsedTerm, TriXParseError> {
    match kind {
        TermElemKind::Uri => {
            let node = NamedNode::new(value.clone())
                .map_err(|e| TriXSyntaxError::invalid_iri(position, value, e))?;
            Ok(ParsedTerm::Named(node))
        }
        TermElemKind::Id => {
            if !is_ncname(&value) {
                return Err(TriXSyntaxError::msg(
                    position,
                    format!("invalid blank node id `{value}`: must be a valid XML NCName"),
                )
                .into());
            }
            Ok(ParsedTerm::Blank(BlankNode::new_unchecked(value)))
        }
        TermElemKind::PlainLiteral(lang) => {
            let literal = match lang {
                Some(lang) => Literal::new_language_tagged_literal(value, lang.clone())
                    .map_err(|e| TriXSyntaxError::invalid_language_tag(position, lang, e))?,
                None => Literal::new_simple_literal(value),
            };
            Ok(ParsedTerm::Literal(literal))
        }
        TermElemKind::TypedLiteral(datatype) => {
            let datatype_node = NamedNode::new(datatype.clone())
                .map_err(|e| TriXSyntaxError::invalid_iri(position, datatype, e))?;
            Ok(ParsedTerm::Literal(Literal::new_typed_literal(
                value,
                datatype_node,
            )))
        }
    }
}

/// Builds the final [`Quad`] once a `<triple>` element has been fully read,
/// validating that it had exactly 3 child terms in valid subject/predicate/object
/// roles.
fn build_quad(
    graph_name: GraphName,
    mut terms: Vec<ParsedTerm>,
    position: u64,
) -> Result<Quad, TriXParseError> {
    if terms.len() != 3 {
        return Err(TriXSyntaxError::msg(
            position,
            format!(
                "a <triple> element must contain exactly 3 child terms, found {}",
                terms.len()
            ),
        )
        .into());
    }
    let object = terms.pop().expect("length checked above").into_term();
    let predicate = terms
        .pop()
        .expect("length checked above")
        .into_named()
        .ok_or_else(|| TriXSyntaxError::msg(position, "a triple's predicate must be a <uri>"))?;
    let subject = terms
        .pop()
        .expect("length checked above")
        .into_named_or_blank()
        .ok_or_else(|| {
            TriXSyntaxError::msg(
                position,
                "a triple's subject must be a <uri> or an <id>, not a literal",
            )
        })?;
    Ok(Quad::new(subject, predicate, object, graph_name))
}

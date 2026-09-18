//! The [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) serializer.

use std::io::{self, Write};
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncWrite, AsyncWriteExt};

use oxixml_model::vocab::xsd;
use oxixml_model::{
    GraphName, GraphNameRef, LiteralRef, NamedNodeRef, NamedOrBlankNodeRef, QuadRef, TermRef,
};
use oxixml_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use oxixml_xml::writer::Writer;

use crate::parser::TRIX_NAMESPACE;

/// A [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) serializer.
///
/// # Example
///
/// ```
/// use oxixml_model::{NamedNode, Quad};
/// use oxixml_trix::TriXSerializer;
///
/// let mut serializer = TriXSerializer::new().for_writer(Vec::new());
/// serializer.serialize_quad(&Quad::new(
///     NamedNode::new("http://example.com/s")?,
///     NamedNode::new("http://example.com/p")?,
///     NamedNode::new("http://example.com/o")?,
///     NamedNode::new("http://example.com/g")?,
/// ))?;
/// let bytes = serializer.finish()?;
/// assert!(std::str::from_utf8(&bytes)?.contains("<graph>"));
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default, Clone, Copy)]
#[must_use]
pub struct TriXSerializer {
    _private: (),
}

impl TriXSerializer {
    /// Builds a new [`TriXSerializer`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Serializes a TriX document to a [`Write`] implementation.
    ///
    /// This writer does unbuffered writes.
    pub fn for_writer<W: Write>(self, writer: W) -> WriterTriXSerializer<W> {
        WriterTriXSerializer {
            writer: Writer::new_with_indent(writer, b' ', 2),
            started: false,
            current_graph: None,
            graph_open: false,
        }
    }

    /// Serializes a TriX document to a [`AsyncWrite`] implementation.
    ///
    /// The asynchronous counterpart of [`for_writer`](Self::for_writer). It renders each quad
    /// with the very same event stream as the blocking path, into a small scratch buffer that is
    /// then flushed to the sink, so the bytes produced are identical.
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_writer<W: AsyncWrite + Unpin>(
        self,
        writer: W,
    ) -> TokioAsyncWriterTriXSerializer<W> {
        TokioAsyncWriterTriXSerializer {
            inner: self.for_writer(Vec::new()),
            writer,
        }
    }
}

/// Serializes a TriX document to a [`AsyncWrite`] implementation.
///
/// Can be built using [`TriXSerializer::for_tokio_async_writer`]. Quads are grouped by graph
/// exactly as in [`WriterTriXSerializer`].
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncWriterTriXSerializer<W: AsyncWrite + Unpin> {
    /// The blocking serializer, writing into an in-memory scratch buffer.
    inner: WriterTriXSerializer<Vec<u8>>,
    writer: W,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterTriXSerializer<W> {
    /// Serializes an extra quad.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying writer fails.
    pub async fn serialize_quad<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> io::Result<()> {
        self.inner.serialize_quad(quad)?;
        self.flush_buffer().await
    }

    /// Ends the write process and returns the underlying [`AsyncWrite`] implementation.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying writer fails.
    pub async fn finish(mut self) -> io::Result<W> {
        self.flush_buffer().await?;
        let buffer = self.inner.finish()?;
        self.writer.write_all(&buffer).await?;
        Ok(self.writer)
    }

    /// Moves everything the blocking serializer has rendered so far out to the async sink.
    async fn flush_buffer(&mut self) -> io::Result<()> {
        let buffer = self.inner.take_buffer();
        if !buffer.is_empty() {
            self.writer.write_all(&buffer).await?;
        }
        Ok(())
    }
}

/// Serializes a TriX document to a [`Write`] implementation.
///
/// Can be built using [`TriXSerializer::for_writer`]. Quads are grouped by graph as they
/// arrive: consecutive quads sharing a [`graph_name`](oxixml_model::Quad::graph_name) are written under a
/// single `<graph>` element, while a change of graph closes the current `<graph>` element and
/// opens a new one (even if that graph name was already seen earlier — TriX allows a dataset's
/// graph to be split across several `<graph>` elements, each contributing triples to it).
#[must_use]
pub struct WriterTriXSerializer<W: Write> {
    writer: Writer<W>,
    started: bool,
    current_graph: Option<GraphName>,
    graph_open: bool,
}

impl<W: Write> WriterTriXSerializer<W> {
    /// Serializes an extra quad.
    pub fn serialize_quad<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> io::Result<()> {
        let quad = quad.into();
        self.ensure_started()?;
        if self.current_graph.as_ref().map(GraphName::as_ref) != Some(quad.graph_name) {
            self.close_graph()?;
            self.open_graph(quad.graph_name)?;
        }
        self.writer
            .write_event(Event::Start(BytesStart::new("triple")))?;
        write_named_or_blank(&mut self.writer, quad.subject)?;
        write_named(&mut self.writer, quad.predicate)?;
        write_term(&mut self.writer, quad.object)?;
        self.writer.write_event(Event::End(BytesEnd::new("triple")))
    }

    /// Ends the write process and returns the underlying [`Write`] implementation.
    pub fn finish(mut self) -> io::Result<W> {
        self.ensure_started()?;
        self.close_graph()?;
        self.writer.write_event(Event::End(BytesEnd::new("TriX")))?;
        Ok(self.writer.into_inner())
    }

    fn ensure_started(&mut self) -> io::Result<()> {
        if self.started {
            return Ok(());
        }
        self.writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;
        let root = BytesStart::new("TriX").with_attributes([("xmlns", TRIX_NAMESPACE)]);
        self.writer.write_event(Event::Start(root))?;
        self.started = true;
        Ok(())
    }

    fn open_graph(&mut self, name: GraphNameRef<'_>) -> io::Result<()> {
        self.writer
            .write_event(Event::Start(BytesStart::new("graph")))?;
        write_graph_name(&mut self.writer, name)?;
        self.current_graph = Some(name.into_owned());
        self.graph_open = true;
        Ok(())
    }

    fn close_graph(&mut self) -> io::Result<()> {
        if self.graph_open {
            self.writer
                .write_event(Event::End(BytesEnd::new("graph")))?;
            self.graph_open = false;
        }
        Ok(())
    }
}

#[cfg(feature = "async-tokio")]
impl WriterTriXSerializer<Vec<u8>> {
    /// Empties the in-memory scratch buffer, returning everything rendered so far.
    ///
    /// Used by [`TokioAsyncWriterTriXSerializer`] to move rendered bytes out to its async sink
    /// after each quad, so that memory use stays bounded rather than accumulating the whole
    /// document.
    fn take_buffer(&mut self) -> Vec<u8> {
        std::mem::take(self.writer.get_mut())
    }
}

fn write_named_or_blank<W: Write>(
    writer: &mut Writer<W>,
    node: NamedOrBlankNodeRef<'_>,
) -> io::Result<()> {
    match node {
        NamedOrBlankNodeRef::NamedNode(n) => write_uri(writer, n.as_str()),
        NamedOrBlankNodeRef::BlankNode(b) => write_id(writer, b.as_str()),
    }
}

fn write_named<W: Write>(writer: &mut Writer<W>, node: NamedNodeRef<'_>) -> io::Result<()> {
    write_uri(writer, node.as_str())
}

fn write_term<W: Write>(writer: &mut Writer<W>, term: TermRef<'_>) -> io::Result<()> {
    match term {
        TermRef::NamedNode(n) => write_uri(writer, n.as_str()),
        TermRef::BlankNode(b) => write_id(writer, b.as_str()),
        TermRef::Literal(lit) => write_literal(writer, lit),
        // `Term::Triple` only exists when `oxixml-model` is built with its `rdf-12` feature, which
        // another crate of the build may switch on. TriX has no element for an RDF 1.2 triple
        // term, so the quad is rejected rather than silently written out as something else.
        #[allow(
            unreachable_patterns,
            reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
        )]
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "TriX has no syntax for an RDF 1.2 triple term",
        )),
    }
}

fn write_graph_name<W: Write>(writer: &mut Writer<W>, name: GraphNameRef<'_>) -> io::Result<()> {
    match name {
        GraphNameRef::NamedNode(n) => write_uri(writer, n.as_str()),
        GraphNameRef::BlankNode(b) => write_id(writer, b.as_str()),
        GraphNameRef::DefaultGraph => Ok(()),
    }
}

fn write_uri<W: Write>(writer: &mut Writer<W>, value: &str) -> io::Result<()> {
    write_text_element(writer, "uri", None, value)
}

fn write_id<W: Write>(writer: &mut Writer<W>, value: &str) -> io::Result<()> {
    write_text_element(writer, "id", None, value)
}

fn write_literal<W: Write>(writer: &mut Writer<W>, literal: LiteralRef<'_>) -> io::Result<()> {
    if let Some(lang) = literal.language() {
        write_text_element(
            writer,
            "plainLiteral",
            Some(("xml:lang", lang)),
            literal.value(),
        )
    } else if literal.datatype().as_str() == xsd::STRING.as_str() {
        write_text_element(writer, "plainLiteral", None, literal.value())
    } else {
        write_text_element(
            writer,
            "typedLiteral",
            Some(("datatype", literal.datatype().as_str())),
            literal.value(),
        )
    }
}

/// Writes `<name attr_name="attr_value">value</name>` (or without the attribute if `attr` is
/// `None`). Shared by [`write_uri`], [`write_id`] and [`write_literal`], so `value` here may be
/// a URI, a blank node id, or a literal's lexical value; `attr_value` (when present) is a
/// `<plainLiteral>`'s language tag or a `<typedLiteral>`'s datatype IRI, the only two attributes
/// this serializer ever writes.
fn write_text_element<W: Write>(
    writer: &mut Writer<W>,
    name: &str,
    attr: Option<(&str, &str)>,
    value: &str,
) -> io::Result<()> {
    let mut start = BytesStart::new(name);
    if let Some((attr_name, attr_value)) = attr {
        push_escaped_attribute(&mut start, attr_name, attr_value)?;
    }
    writer.write_event(Event::Start(start))?;
    write_text_content(writer, name, value)?;
    writer.write_event(Event::End(BytesEnd::new(name)))
}

/// Writes `value` as `name`'s element text content, escaping an embedded carriage return and
/// rejecting any code point the XML `Char` production forbids outright.
///
/// # Carriage returns
///
/// XML's end-of-line handling ([XML 1.0 §2.11]) requires every conformant parser to silently
/// rewrite a raw, unescaped `\r` (and `\r\n`) in text content to `\n`; a character reference is
/// explicitly exempt from that rewrite. So a value containing a lone `\r` is written as two
/// [`Event::Text`] runs around a `&#13;` reference (itself carried by an `Event::Text` built with
/// [`BytesText::from_escaped`], which -- unlike [`BytesText::new`] -- writes its argument
/// verbatim instead of re-escaping it) rather than as one run that a re-parse would silently turn
/// into `\n`.
///
/// # Code points XML cannot represent
///
/// The C0 controls other than tab/LF/CR, and the two byte-order-mark-adjacent noncharacters
/// U+FFFE/U+FFFF, are outside the XML `Char` production: there is no legal way to write them,
/// escaped or not, because a character reference still expands back to the same forbidden code
/// point on re-parse. Rather than silently emit a document that is not well-formed XML, such a
/// value is rejected with [`io::ErrorKind::InvalidInput`].
///
/// [XML 1.0 §2.11]: https://www.w3.org/TR/xml/#sec-line-ends
fn write_text_content<W: Write>(writer: &mut Writer<W>, name: &str, value: &str) -> io::Result<()> {
    reject_illegal_xml_char(name, value)?;
    let mut parts = value.split('\r');
    if let Some(first) = parts.next() {
        writer.write_event(Event::Text(BytesText::new(first)))?;
    }
    for part in parts {
        writer.write_event(Event::Text(BytesText::from_escaped("&#13;")))?;
        writer.write_event(Event::Text(BytesText::new(part)))?;
    }
    Ok(())
}

/// Returns the first code point in `value` that the XML 1.0 `Char` production forbids, if any:
/// `Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]` ([XML 1.0
/// §2.2]). Shared by [`reject_illegal_xml_char`] and [`push_escaped_attribute`], which each wrap
/// it in a message tailored to whether `value` is element text or an attribute value.
///
/// [XML 1.0 §2.2]: https://www.w3.org/TR/xml/#charsets
fn first_illegal_char(value: &str) -> Option<char> {
    value.chars().find(|c| !is_xml_char(*c))
}

/// Returns an error naming the first code point in `value` that the XML `Char` production
/// forbids, if any; see [`first_illegal_char`].
fn reject_illegal_xml_char(element: &str, value: &str) -> io::Result<()> {
    if let Some(c) = first_illegal_char(value) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "a TriX <{element}> value contains U+{:04X}, which the XML `Char` production forbids and cannot be serialized",
                c as u32
            ),
        ));
    }
    Ok(())
}

/// Returns whether `c` is allowed by the XML 1.0 `Char` production; see
/// [`first_illegal_char`].
fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r')
        || ('\u{20}'..='\u{D7FF}').contains(&c)
        || ('\u{E000}'..='\u{FFFD}').contains(&c)
        || ('\u{10000}'..='\u{10FFFF}').contains(&c)
}

/// Adds `name="value"` to `start`, escaping `value` for attribute syntax and rejecting any code
/// point the XML `Char` production forbids outright -- the attribute-value counterpart of
/// [`write_text_content`], applied to a `<plainLiteral>`'s `xml:lang` or a `<typedLiteral>`'s
/// `datatype` (see [`write_text_element`]).
///
/// `oxixml_xml`'s own attribute constructor escapes `<`, `>`, `&`, `'` and `"` but, like
/// [`BytesText::new`], leaves tab/CR/LF alone; for an *attribute* value that is a worse defect
/// than for text content, because XML attribute-value normalization ([XML 1.0
/// §3.3.3](https://www.w3.org/TR/xml/#AVNormalize)) collapses a raw, unescaped one of the three
/// to a single space on every conformant re-parse, silently changing the value rather than only
/// its line-ending spelling. A character reference is exempt from that normalization the same
/// way it is exempt from end-of-line handling, so `value` is escaped by hand into a plain
/// `String` and pushed through `oxixml_xml`'s raw-bytes attribute constructor (bypassing its own
/// escaping, which would otherwise re-escape the `&` these character references start with).
fn push_escaped_attribute(start: &mut BytesStart<'_>, name: &str, value: &str) -> io::Result<()> {
    if let Some(c) = first_illegal_char(value) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "a TriX {name} attribute value contains U+{:04X}, which the XML `Char` production forbids and cannot be serialized",
                c as u32
            ),
        ));
    }
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '&' => escaped.push_str("&amp;"),
            '\'' => escaped.push_str("&apos;"),
            '"' => escaped.push_str("&quot;"),
            '\t' => escaped.push_str("&#9;"),
            '\n' => escaped.push_str("&#10;"),
            '\r' => escaped.push_str("&#13;"),
            _ => escaped.push(ch),
        }
    }
    start.push_attribute((name.as_bytes(), escaped.as_bytes()));
    Ok(())
}

//! The [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/) streaming serializer.

use crate::utils::{is_name_char, is_name_start_char};
use oxixml_iri::{Iri, IriParseError};
use oxixml_model::vocab::{rdf, xsd};
use oxixml_model::{NamedNodeRef, NamedOrBlankNode, NamedOrBlankNodeRef, TermRef, TripleRef};
use oxixml_xml::Writer;
use oxixml_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io;
use std::io::Write;
#[cfg(feature = "async-tokio")]
use std::sync::Arc;
#[cfg(feature = "async-tokio")]
use tokio::io::AsyncWrite;

/// A [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/) serializer.
///
/// Triples are written in streaming: consecutive triples that share a subject are grouped into a
/// single `rdf:Description` (or into a typed node element when the first of them is an
/// `rdf:type`), so it is worth sorting the input by subject before serializing it.
///
/// ```
/// use oxixml_model::vocab::rdf;
/// use oxixml_model::{Literal, NamedNode, Triple};
/// use oxixml_rdfxml::RdfXmlSerializer;
///
/// let mut serializer = RdfXmlSerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .for_writer(Vec::new());
/// serializer.serialize_triple(&Triple::new(
///     NamedNode::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNode::new("http://schema.org/Person")?,
/// ))?;
/// serializer.serialize_triple(&Triple::new(
///     NamedNode::new("http://example.com#me")?,
///     NamedNode::new("http://schema.org/name")?,
///     Literal::new_language_tagged_literal_unchecked("Foo Bar", "en"),
/// ))?;
/// assert_eq!(
///     b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns:schema=\"http://schema.org/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" xmlns:its=\"http://www.w3.org/2005/11/its\">\n\t<schema:Person rdf:about=\"http://example.com#me\">\n\t\t<schema:name xml:lang=\"en\">Foo Bar</schema:name>\n\t</schema:Person>\n</rdf:RDF>",
///     serializer.finish()?.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default, Clone)]
#[must_use]
pub struct RdfXmlSerializer {
    prefixes: BTreeMap<String, String>,
    base_iri: Option<Iri<String>>,
}

impl RdfXmlSerializer {
    /// Builds a new [`RdfXmlSerializer`].
    #[inline]
    pub fn new() -> Self {
        Self {
            prefixes: BTreeMap::new(),
            base_iri: None,
        }
    }

    /// Adds a `xmlns:` prefix declaration to the `rdf:RDF` element.
    ///
    /// The `rdf` and `its` prefixes are reserved and always bound to their standard IRIs; so is
    /// the internal `oxprefix` prefix, used for IRIs no declared prefix covers.
    ///
    /// # Errors
    ///
    /// Returns an [`IriParseError`] if `prefix_iri` is not a valid absolute IRI.
    #[inline]
    pub fn with_prefix(
        mut self,
        prefix_name: impl Into<String>,
        prefix_iri: impl Into<String>,
    ) -> Result<Self, IriParseError> {
        let prefix_name = prefix_name.into();
        if prefix_name == "oxprefix" {
            return Ok(self); // It is reserved
        }
        self.prefixes
            .insert(prefix_name, Iri::parse(prefix_iri.into())?.into_inner());
        Ok(self)
    }

    /// Sets the `xml:base` of the `rdf:RDF` element; IRIs are then written relative to it.
    ///
    /// ```
    /// use oxixml_model::{NamedNode, Triple};
    /// use oxixml_rdfxml::RdfXmlSerializer;
    ///
    /// let mut serializer = RdfXmlSerializer::new()
    ///     .with_base_iri("http://example.com")?
    ///     .with_prefix("ex", "http://example.com/ns#")?
    ///     .for_writer(Vec::new());
    /// serializer.serialize_triple(&Triple::new(
    ///     NamedNode::new("http://example.com#me")?,
    ///     NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
    ///     NamedNode::new("http://example.com/ns#Person")?,
    /// ))?;
    /// serializer.serialize_triple(&Triple::new(
    ///     NamedNode::new("http://example.com#me")?,
    ///     NamedNode::new("http://example.com/ns#parent")?,
    ///     NamedNode::new("http://example.com#other")?,
    /// ))?;
    /// assert_eq!(
    ///     b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xml:base=\"http://example.com\" xmlns:ex=\"http://example.com/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" xmlns:its=\"http://www.w3.org/2005/11/its\">\n\t<ex:Person rdf:about=\"#me\">\n\t\t<ex:parent rdf:resource=\"#other\"/>\n\t</ex:Person>\n</rdf:RDF>",
    ///     serializer.finish()?.as_slice()
    /// );
    /// # Result::<_,Box<dyn std::error::Error>>::Ok(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`IriParseError`] if `base_iri` is not a valid absolute IRI.
    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        self.base_iri = Some(Iri::parse(base_iri.into())?);
        Ok(self)
    }

    /// Serializes a RDF/XML file to a [`Write`] implementation.
    ///
    /// This writer does unbuffered writes.
    ///
    /// ```
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_model::{Literal, NamedNode, Triple};
    /// use oxixml_rdfxml::RdfXmlSerializer;
    ///
    /// let mut serializer = RdfXmlSerializer::new()
    ///     .with_prefix("schema", "http://schema.org/")?
    ///     .for_writer(Vec::new());
    /// serializer.serialize_triple(&Triple::new(
    ///     NamedNode::new("http://example.com#me")?,
    ///     rdf::TYPE,
    ///     NamedNode::new("http://schema.org/Person")?,
    /// ))?;
    /// assert_eq!(
    ///     b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns:schema=\"http://schema.org/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" xmlns:its=\"http://www.w3.org/2005/11/its\">\n\t<schema:Person rdf:about=\"http://example.com#me\">\n\t</schema:Person>\n</rdf:RDF>",
    ///     serializer.finish()?.as_slice()
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_writer<W: Write>(self, writer: W) -> WriterRdfXmlSerializer<W> {
        WriterRdfXmlSerializer {
            writer: Writer::new_with_indent(writer, b'\t', 1),
            inner: self.inner_writer(),
        }
    }

    /// Writes an RDF/XML file to a [`AsyncWrite`] implementation.
    ///
    /// This is the asynchronous counterpart of [`for_writer`](Self::for_writer): it emits exactly
    /// the same event stream, awaiting each write instead of blocking on it.
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_writer<W: AsyncWrite + Unpin>(
        self,
        writer: W,
    ) -> TokioAsyncWriterRdfXmlSerializer<W> {
        TokioAsyncWriterRdfXmlSerializer {
            writer: Writer::new_with_indent(writer, b'\t', 1),
            inner: self.inner_writer(),
        }
    }

    /// Normalizes the prefix map and builds the shared serialization state.
    fn inner_writer(mut self) -> InnerRdfXmlWriter {
        // Makes sure 'rdf' and 'its' are the proper prefixes, by first removing them
        self.prefixes.remove("rdf");
        self.prefixes.remove("its");
        let custom_default_prefix = self.prefixes.contains_key("");
        // The serializer wants to have the URL first, we swap
        let mut prefixes = self
            .prefixes
            .into_iter()
            .map(|(key, value)| (value, key))
            .collect::<BTreeMap<_, _>>();
        prefixes.insert(
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".into(),
            "rdf".into(),
        );
        prefixes.insert("http://www.w3.org/2005/11/its".into(), "its".into());
        InnerRdfXmlWriter {
            current_subject: None,
            current_resource_tag: None,
            custom_default_prefix,
            prefixes_by_iri: prefixes,
            base_iri: self.base_iri,
        }
    }
}

/// Serializes a RDF/XML file to a [`Write`] implementation.
///
/// Can be built using [`RdfXmlSerializer::for_writer`].
///
/// ```
/// use oxixml_model::vocab::rdf;
/// use oxixml_model::{Literal, NamedNode, Triple};
/// use oxixml_rdfxml::RdfXmlSerializer;
///
/// let mut serializer = RdfXmlSerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .for_writer(Vec::new());
/// serializer.serialize_triple(&Triple::new(
///     NamedNode::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNode::new("http://schema.org/Person")?,
/// ))?;
/// serializer.serialize_triple(&Triple::new(
///     NamedNode::new("http://example.com#me")?,
///     NamedNode::new("http://schema.org/name")?,
///     Literal::new_language_tagged_literal_unchecked("Foo Bar", "en"),
/// ))?;
/// assert_eq!(
///     b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns:schema=\"http://schema.org/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" xmlns:its=\"http://www.w3.org/2005/11/its\">\n\t<schema:Person rdf:about=\"http://example.com#me\">\n\t\t<schema:name xml:lang=\"en\">Foo Bar</schema:name>\n\t</schema:Person>\n</rdf:RDF>",
///     serializer.finish()?.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct WriterRdfXmlSerializer<W: Write> {
    writer: Writer<W>,
    inner: InnerRdfXmlWriter,
}

impl<W: Write> WriterRdfXmlSerializer<W> {
    /// Serializes an extra triple.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the underlying writer fails, or if the triple uses an RDF/XML
    /// reserved syntax term as its predicate.
    pub fn serialize_triple<'a>(&mut self, t: impl Into<TripleRef<'a>>) -> io::Result<()> {
        let mut buffer = Vec::new();
        self.inner.serialize_triple(t, &mut buffer)?;
        self.flush_buffer(&mut buffer)
    }

    /// Ends the write process and returns the underlying [`Write`].
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the underlying writer fails.
    pub fn finish(mut self) -> io::Result<W> {
        let mut buffer = Vec::new();
        self.inner.finish(&mut buffer);
        self.flush_buffer(&mut buffer)?;
        Ok(self.writer.into_inner())
    }

    /// Writes the events accumulated for one triple.
    fn flush_buffer(&mut self, buffer: &mut Vec<Event<'_>>) -> io::Result<()> {
        for event in buffer.drain(0..) {
            self.writer.write_event(event)?;
        }
        Ok(())
    }
}

/// RDF/XML syntax terms that may not be used as a predicate nor as a typed node element.
const RESERVED_SYNTAX_TERMS: [&str; 9] = [
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#Description",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#li",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#RDF",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#ID",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#about",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#parseType",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#resource",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#nodeID",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#datatype",
];

/// The reader-agnostic serialization state: which subject is open and which prefixes are bound.
/// Writes an RDF/XML file to a [`AsyncWrite`] implementation.
///
/// Can be built using [`RdfXmlSerializer::for_tokio_async_writer`].
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncWriterRdfXmlSerializer<W: AsyncWrite + Unpin> {
    writer: Writer<W>,
    inner: InnerRdfXmlWriter,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterRdfXmlSerializer<W> {
    /// Serializes an extra triple.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the underlying writer fails, or if the triple uses an RDF/XML
    /// reserved syntax term as its predicate.
    pub async fn serialize_triple<'a>(&mut self, t: impl Into<TripleRef<'a>>) -> io::Result<()> {
        let mut buffer = Vec::new();
        self.inner.serialize_triple(t, &mut buffer)?;
        self.flush_buffer(&mut buffer).await
    }

    /// Ends the write process and returns the underlying [`AsyncWrite`].
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the underlying writer fails.
    pub async fn finish(mut self) -> io::Result<W> {
        let mut buffer = Vec::new();
        self.inner.finish(&mut buffer);
        self.flush_buffer(&mut buffer).await?;
        Ok(self.writer.into_inner())
    }

    /// Writes the events accumulated for one triple.
    async fn flush_buffer(&mut self, buffer: &mut Vec<Event<'_>>) -> io::Result<()> {
        for event in buffer.drain(0..) {
            self.writer
                .write_event_async(event)
                .await
                .map_err(map_xml_err)?;
        }
        Ok(())
    }
}

/// Maps an `oxixml-xml` writer error back to [`io::Error`].
///
/// The synchronous [`Writer::write_event`] already returns [`io::Error`]; the asynchronous
/// counterpart returns the richer [`oxixml_xml::Error`], so the I/O variant is unwrapped back out
/// (recovering the original error whenever this is the only live reference) and anything else is
/// wrapped with [`io::Error::other`].
#[cfg(feature = "async-tokio")]
fn map_xml_err(error: oxixml_xml::Error) -> io::Error {
    if let oxixml_xml::Error::Io(error) = error {
        Arc::try_unwrap(error).unwrap_or_else(|error| io::Error::new(error.kind(), error))
    } else {
        io::Error::other(error)
    }
}

struct InnerRdfXmlWriter {
    current_subject: Option<NamedOrBlankNode>,
    current_resource_tag: Option<String>,
    custom_default_prefix: bool,
    prefixes_by_iri: BTreeMap<String, String>,
    base_iri: Option<Iri<String>>,
}

impl InnerRdfXmlWriter {
    /// Turns one triple into XML events.
    fn serialize_triple<'a>(
        &mut self,
        triple: impl Into<TripleRef<'a>>,
        output: &mut Vec<Event<'a>>,
    ) -> io::Result<()> {
        if self.current_subject.is_none() {
            self.write_start(output);
        }

        let triple = triple.into();
        // We open a new rdf:Description if useful
        if self.current_subject.as_ref().map(NamedOrBlankNode::as_ref) != Some(triple.subject) {
            if self.current_subject.is_some() {
                output.push(Event::End(
                    self.current_resource_tag
                        .take()
                        .map_or_else(|| BytesEnd::new("rdf:Description"), BytesEnd::new),
                ));
            }
            self.current_subject = Some(triple.subject.into_owned());

            let (mut description_start, with_type_tag) = if triple.predicate == rdf::TYPE {
                if let TermRef::NamedNode(t) = triple.object {
                    if RESERVED_SYNTAX_TERMS.contains(&t.as_str()) {
                        (BytesStart::new("rdf:Description"), false)
                    } else {
                        let (prop_qname, prop_xmlns) = self.uri_to_qname_and_xmlns(t);
                        let mut description_start = BytesStart::new(prop_qname.clone());
                        push_optional_xmlns(&mut description_start, prop_xmlns)?;
                        self.current_resource_tag = Some(prop_qname.into_owned());
                        (description_start, true)
                    }
                } else {
                    (BytesStart::new("rdf:Description"), false)
                }
            } else {
                (BytesStart::new("rdf:Description"), false)
            };
            let (attr_name, attr_value) = match triple.subject {
                NamedOrBlankNodeRef::NamedNode(node) => {
                    ("rdf:about", relative_iri(node.as_str(), &self.base_iri))
                }
                NamedOrBlankNodeRef::BlankNode(node) => {
                    ("rdf:nodeID", Cow::Borrowed(node.as_str()))
                }
            };
            push_escaped_attribute(&mut description_start, attr_name, &attr_value)?;
            output.push(Event::Start(description_start));
            if with_type_tag {
                return Ok(()); // No need for a value
            }
        }
        self.write_predicate_object(triple.predicate, triple.object, output)
    }

    /// Writes one property element.
    fn write_predicate_object<'a>(
        &mut self,
        predicate: NamedNodeRef<'a>,
        object: TermRef<'a>,
        output: &mut Vec<Event<'a>>,
    ) -> io::Result<()> {
        if RESERVED_SYNTAX_TERMS.contains(&predicate.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "RDF/XML reserved syntax term is not allowed as a predicate",
            ));
        }

        let (prop_qname, prop_xmlns) = self.uri_to_qname_and_xmlns(predicate);
        let mut property_open = BytesStart::new(prop_qname.clone());
        push_optional_xmlns(&mut property_open, prop_xmlns)?;
        match object {
            TermRef::NamedNode(node) => {
                let resource = relative_iri(node.as_str(), &self.base_iri);
                push_escaped_attribute(&mut property_open, "rdf:resource", &resource)?;
                output.push(Event::Empty(property_open));
            }
            TermRef::BlankNode(node) => {
                push_escaped_attribute(&mut property_open, "rdf:nodeID", node.as_str())?;
                output.push(Event::Empty(property_open));
            }
            TermRef::Literal(literal) => {
                if let Some(language) = literal.language() {
                    push_escaped_attribute(&mut property_open, "xml:lang", language)?;
                } else if literal.datatype() != xsd::STRING {
                    let datatype = relative_iri(literal.datatype().as_str(), &self.base_iri);
                    push_escaped_attribute(&mut property_open, "rdf:datatype", &datatype)?;
                }
                output.push(Event::Start(property_open));
                push_literal_text(output, literal.value())?;
                output.push(Event::End(BytesEnd::new(prop_qname)));
            }
            // `Term::Triple` only exists when `oxixml-model` is built with its `rdf-12` feature,
            // which another crate of the build may switch on. This serializer implements the
            // RDF/XML 1.1 grammar, which has no production for a triple term (RDF/XML 1.2 writes
            // them as `rdf:parseType="Triple"`), so the triple is rejected rather than dropped.
            #[allow(
                unreachable_patterns,
                reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
            )]
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "RDF/XML 1.1 has no syntax for an RDF 1.2 triple term",
                ));
            }
        }
        Ok(())
    }

    /// Writes the XML declaration and the opening `rdf:RDF` tag.
    fn write_start(&self, output: &mut Vec<Event<'_>>) {
        output.push(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)));
        let mut rdf_open = BytesStart::new("rdf:RDF");
        if let Some(base_iri) = &self.base_iri {
            rdf_open.push_attribute(("xml:base", base_iri.as_str()));
        }
        for (prefix_value, prefix_name) in &self.prefixes_by_iri {
            rdf_open.push_attribute((
                if prefix_name.is_empty() {
                    "xmlns".into()
                } else {
                    format!("xmlns:{prefix_name}")
                }
                .as_str(),
                prefix_value.as_str(),
            ));
        }
        output.push(Event::Start(rdf_open));
    }

    /// Closes the last `rdf:Description` and the `rdf:RDF` element.
    fn finish(&mut self, output: &mut Vec<Event<'static>>) {
        if self.current_subject.is_some() {
            output.push(Event::End(
                self.current_resource_tag
                    .take()
                    .map_or_else(|| BytesEnd::new("rdf:Description"), BytesEnd::new),
            ));
        } else {
            self.write_start(output);
        }
        output.push(Event::End(BytesEnd::new("rdf:RDF")));
    }

    /// Turns an IRI into a QName, adding an inline `xmlns` declaration when needed.
    fn uri_to_qname_and_xmlns<'a>(
        &self,
        uri: NamedNodeRef<'a>,
    ) -> (Cow<'a, str>, Option<(&'a str, &'a str)>) {
        let (prop_prefix, prop_value) = split_iri(uri.as_str());
        if let Some(prop_prefix) = self.prefixes_by_iri.get(prop_prefix) {
            (
                if prop_prefix.is_empty() {
                    Cow::Borrowed(prop_value)
                } else {
                    Cow::Owned(format!("{prop_prefix}:{prop_value}"))
                },
                None,
            )
        } else if prop_prefix == "http://www.w3.org/2000/xmlns/" {
            (Cow::Owned(format!("xmlns:{prop_value}")), None)
        } else if !prop_value.is_empty() && !self.custom_default_prefix {
            (Cow::Borrowed(prop_value), Some(("xmlns", prop_prefix)))
        } else {
            (
                Cow::Owned(format!("oxprefix:{prop_value}")),
                Some(("xmlns:oxprefix", prop_prefix)),
            )
        }
    }
}

/// Pushes a literal's value as element text content, escaping an embedded carriage return and
/// rejecting any code point the XML `Char` production forbids outright.
///
/// # Carriage returns
///
/// XML's end-of-line handling ([XML 1.0 §2.11]) requires every conformant parser to silently
/// rewrite a raw, unescaped `\r` (and `\r\n`) in text content to `\n`; a character reference is
/// explicitly exempt from that rewrite. So a literal value containing a lone `\r` is written as
/// two [`Event::Text`] runs around a `&#13;` reference (itself carried by a `Event::Text` built
/// with [`BytesText::from_escaped`], which -- unlike [`BytesText::new`] -- writes its argument
/// verbatim instead of re-escaping it) rather than as one run that a re-parse would silently
/// turn into `\n`.
///
/// # Code points XML cannot represent
///
/// The C0 controls other than tab/LF/CR, and the two byte-order-mark-adjacent noncharacters
/// U+FFFE/U+FFFF, are outside the XML `Char` production: there is no legal way to write them,
/// escaped or not, because a character reference still expands back to the same forbidden code
/// point on re-parse. Rather than silently emit a document that is not well-formed XML, such a
/// literal is rejected with [`io::ErrorKind::InvalidInput`].
///
/// [XML 1.0 §2.11]: https://www.w3.org/TR/xml/#sec-line-ends
fn push_literal_text<'a>(output: &mut Vec<Event<'a>>, value: &'a str) -> io::Result<()> {
    reject_illegal_xml_char(value)?;
    let mut parts = value.split('\r');
    if let Some(first) = parts.next() {
        output.push(Event::Text(BytesText::new(first)));
    }
    for part in parts {
        output.push(Event::Text(BytesText::from_escaped("&#13;")));
        output.push(Event::Text(BytesText::new(part)));
    }
    Ok(())
}

/// Returns an error naming the first code point in `value` that the XML 1.0 `Char` production
/// forbids, if any: `Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] |
/// [#x10000-#x10FFFF]` ([XML 1.0 §2.2]).
///
/// [XML 1.0 §2.2]: https://www.w3.org/TR/xml/#charsets
fn reject_illegal_xml_char(value: &str) -> io::Result<()> {
    if let Some(c) = value.chars().find(|c| !is_xml_char(*c)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "literal value contains U+{:04X}, which the XML `Char` production forbids and cannot be serialized as RDF/XML",
                c as u32
            ),
        ));
    }
    Ok(())
}

/// Returns whether `c` is allowed by the XML 1.0 `Char` production; see
/// [`reject_illegal_xml_char`].
fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r')
        || ('\u{20}'..='\u{D7FF}').contains(&c)
        || ('\u{E000}'..='\u{FFFD}').contains(&c)
        || ('\u{10000}'..='\u{10FFFF}').contains(&c)
}

/// Adds `name="value"` to `start`, escaping `value` for attribute syntax and rejecting any code
/// point the XML `Char` production forbids outright (see [`push_literal_text`]).
///
/// `value` here is always an IRI, a blank node label, a language tag, or a namespace IRI pulled
/// out of one of those for an inline `xmlns` declaration -- never itself XML-validated by
/// `oxixml-model` before an `_unchecked` constructor, exactly like a literal's value.
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
    reject_illegal_xml_char(value)?;
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

/// Adds the `("xmlns"` or `"xmlns:oxprefix", namespace_iri)` attribute
/// [`uri_to_qname_and_xmlns`](InnerRdfXmlWriter::uri_to_qname_and_xmlns) returns, if any, to
/// `start` -- escaping `namespace_iri` exactly like [`push_escaped_attribute`]'s other callers.
/// `namespace_iri` is a slice of a [`NamedNode`]'s own IRI text (the predicate's, or an
/// `rdf:type` object's, when it is written as a typed node element name instead), so it carries
/// exactly the same "not necessarily validated by `oxixml-model`" risk as `rdf:about`,
/// `rdf:resource`, and the rest.
fn push_optional_xmlns(start: &mut BytesStart<'_>, xmlns: Option<(&str, &str)>) -> io::Result<()> {
    if let Some((name, namespace_iri)) = xmlns {
        push_escaped_attribute(start, name, namespace_iri)?;
    }
    Ok(())
}

/// Splits an IRI into the longest namespace prefix and the trailing XML `Name`.
fn split_iri(iri: &str) -> (&str, &str) {
    if let Some(position_base) = iri.rfind(|c| !is_name_char(c) || c == ':') {
        if let Some(position_add) = iri[position_base..].find(|c| is_name_start_char(c) && c != ':')
        {
            (
                &iri[..position_base + position_add],
                &iri[position_base + position_add..],
            )
        } else {
            (iri, "")
        }
    } else {
        (iri, "")
    }
}

/// Relativizes `iri` against `base_iri` when possible.
fn relative_iri<'a>(iri: &'a str, base_iri: &Option<Iri<String>>) -> Cow<'a, str> {
    if let Some(base_iri) = base_iri {
        if let Ok(relative) = base_iri.relativize(&Iri::parse_unchecked(iri)) {
            return relative.into_inner().into();
        }
    }
    iri.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_model::{BlankNode, Literal, NamedNode, Term, Triple};
    use std::error::Error;

    #[test]
    fn test_split_iri() {
        assert_eq!(
            split_iri("http://schema.org/Person"),
            ("http://schema.org/", "Person")
        );
        assert_eq!(split_iri("http://schema.org/"), ("http://schema.org/", ""));
        assert_eq!(
            split_iri("http://schema.org#foo"),
            ("http://schema.org#", "foo")
        );
        assert_eq!(split_iri("urn:isbn:foo"), ("urn:isbn:", "foo"));
    }

    #[test]
    fn test_custom_rdf_ns() -> Result<(), Box<dyn Error>> {
        let output = RdfXmlSerializer::new()
            .with_prefix("rdf", "http://example.com/")?
            .for_writer(Vec::new())
            .finish()?;
        assert_eq!(
            String::from_utf8(output)?,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" xmlns:its=\"http://www.w3.org/2005/11/its\">\n</rdf:RDF>"
        );
        Ok(())
    }

    #[test]
    fn test_custom_empty_ns() -> Result<(), Box<dyn Error>> {
        let mut serializer = RdfXmlSerializer::new()
            .with_prefix("", "http://example.com/")?
            .for_writer(Vec::new());
        serializer.serialize_triple(&Triple::new(
            NamedNode::new("http://example.com/s")?,
            rdf::TYPE,
            NamedNode::new("http://example.org/o")?,
        ))?;
        serializer.serialize_triple(&Triple::new(
            NamedNode::new("http://example.com/s")?,
            NamedNode::new("http://example.com/p")?,
            NamedNode::new("http://example.com/o2")?,
        ))?;
        let output = serializer.finish()?;
        assert_eq!(
            String::from_utf8_lossy(&output),
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns=\"http://example.com/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" xmlns:its=\"http://www.w3.org/2005/11/its\">\n\t<oxprefix:o xmlns:oxprefix=\"http://example.org/\" rdf:about=\"http://example.com/s\">\n\t\t<p rdf:resource=\"http://example.com/o2\"/>\n\t</oxprefix:o>\n</rdf:RDF>"
        );
        Ok(())
    }

    #[test]
    fn reserved_predicate_is_rejected() -> Result<(), Box<dyn Error>> {
        let mut serializer = RdfXmlSerializer::new().for_writer(Vec::new());
        let error = serializer
            .serialize_triple(&Triple::new(
                NamedNode::new("http://example.com/s")?,
                NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#li")?,
                NamedNode::new("http://example.com/o")?,
            ))
            .expect_err("reserved term");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        Ok(())
    }

    #[test]
    fn typed_literal_uses_rdf_datatype() -> Result<(), Box<dyn Error>> {
        let mut serializer = RdfXmlSerializer::new().for_writer(Vec::new());
        serializer.serialize_triple(&Triple::new(
            NamedNode::new("http://example.com/s")?,
            NamedNode::new("http://example.com/p")?,
            Literal::new_typed_literal("1", NamedNode::new("http://example.com/dt")?),
        ))?;
        let output = String::from_utf8(serializer.finish()?)?;
        assert!(
            output.contains("rdf:datatype=\"http://example.com/dt\""),
            "{output}"
        );
        Ok(())
    }

    /// Regression test: a literal's embedded `\r` used to be written as a raw, unescaped byte.
    /// XML mandates that a conformant parser silently rewrite that to `\n` on re-parse (XML 1.0
    /// end-of-line handling), so the value did not round-trip. It must now survive as a `&#13;`
    /// character reference, which end-of-line handling does not touch.
    #[test]
    fn literal_with_carriage_return_round_trips() -> Result<(), Box<dyn Error>> {
        let value = "a\tb\nc\rd";
        let mut serializer = RdfXmlSerializer::new().for_writer(Vec::new());
        serializer.serialize_triple(&Triple::new(
            NamedNode::new("http://example.com/s")?,
            NamedNode::new("http://example.com/p")?,
            Literal::new_simple_literal(value),
        ))?;
        let output = serializer.finish()?;
        assert!(
            String::from_utf8_lossy(&output).contains("&#13;"),
            "expected an escaped carriage return in {}",
            String::from_utf8_lossy(&output)
        );
        let triples = crate::RdfXmlParser::new()
            .for_slice(&output)
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|e| {
                panic!(
                    "re-parsing the serialized document failed: {e}\n---\n{}",
                    String::from_utf8_lossy(&output)
                )
            });
        assert_eq!(triples.len(), 1);
        let Term::Literal(literal) = &triples[0].object else {
            panic!("expected a literal object, got {:?}", triples[0].object);
        };
        assert_eq!(literal.value(), value);
        Ok(())
    }

    /// Regression test: code points the XML `Char` production forbids (here, NUL and the
    /// noncharacter U+FFFE) must be rejected rather than written as non-well-formed XML.
    #[test]
    fn literal_with_illegal_xml_char_is_rejected() -> Result<(), Box<dyn Error>> {
        for value in ["\u{0}", "\u{fffe}", "\u{b}"] {
            let mut serializer = RdfXmlSerializer::new().for_writer(Vec::new());
            let error = serializer
                .serialize_triple(&Triple::new(
                    NamedNode::new("http://example.com/s")?,
                    NamedNode::new("http://example.com/p")?,
                    Literal::new_simple_literal(value),
                ))
                .expect_err("illegal XML character");
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        }
        Ok(())
    }

    /// Regression test: `rdf:about`/`rdf:resource`/`rdf:nodeID`/`xml:lang`/`rdf:datatype` are
    /// attribute values carrying an IRI, a blank node label or a language tag -- the same class
    /// of "arbitrary text this crate did not itself validate" as a literal's value, reachable
    /// through an `_unchecked` constructor. They used to go through `oxixml_xml`'s own attribute
    /// escaping, which (like `BytesText::new`) never escapes tab/CR/LF; for an *attribute* value
    /// that is worse than for text content, since attribute-value normalization collapses a raw
    /// one of the three to a single space on re-parse rather than only rewriting `\r` to `\n`.
    ///
    /// Unlike a literal's value, every one of these fields has its own grammar (IRI syntax,
    /// `NCName`, BCP 47) that a bare `\r` violates regardless of how the XML layer carries it, so
    /// re-parsing this output back into equal terms is not the right check (the same reasoning
    /// `oxixml-trix`'s equivalent test documents for `<uri>`/`<id>`). This asserts the byte-level
    /// fix instead: the serialized document must carry `&#13;`, not a raw carriage return.
    #[test]
    fn attribute_values_escape_carriage_return_instead_of_corrupting() -> Result<(), Box<dyn Error>>
    {
        let mut serializer = RdfXmlSerializer::new().for_writer(Vec::new());
        // subject (rdf:nodeID), object (rdf:resource).
        serializer.serialize_triple(&Triple::new(
            BlankNode::new_unchecked("s\r1"),
            NamedNode::new("http://example.com/p")?,
            NamedNode::new_unchecked("http://example.com/o\r1"),
        ))?;
        // object (rdf:nodeID on a fresh subject, so a new rdf:Description is opened).
        serializer.serialize_triple(&Triple::new(
            NamedNode::new("http://example.com/s2")?,
            NamedNode::new("http://example.com/p2")?,
            BlankNode::new_unchecked("o\r2"),
        ))?;
        // xml:lang and rdf:datatype.
        serializer.serialize_triple(&Triple::new(
            NamedNode::new("http://example.com/s3")?,
            NamedNode::new("http://example.com/p3")?,
            Literal::new_language_tagged_literal_unchecked("hello", "en\r"),
        ))?;
        serializer.serialize_triple(&Triple::new(
            NamedNode::new("http://example.com/s4")?,
            NamedNode::new("http://example.com/p4")?,
            Literal::new_typed_literal("1", NamedNode::new_unchecked("http://example.com/dt\r1")),
        ))?;
        let output = serializer.finish()?;
        let text = String::from_utf8_lossy(&output);
        assert!(
            !text.contains('\r'),
            "a raw carriage return leaked through: {text}"
        );
        assert_eq!(
            text.matches("&#13;").count(),
            5,
            "expected one escaped carriage return per attribute value: {text}"
        );
        Ok(())
    }

    /// Regression test: the same illegal-code-point rejection as
    /// [`literal_with_illegal_xml_char_is_rejected`], but for an attribute value (`rdf:about`).
    #[test]
    fn attribute_value_with_illegal_xml_char_is_rejected() -> Result<(), Box<dyn Error>> {
        let mut serializer = RdfXmlSerializer::new().for_writer(Vec::new());
        let error = serializer
            .serialize_triple(&Triple::new(
                NamedNode::new_unchecked("http://example.com/s\u{0}"),
                NamedNode::new("http://example.com/p")?,
                NamedNode::new("http://example.com/o")?,
            ))
            .expect_err("illegal XML character in rdf:about");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        Ok(())
    }

    /// Regression test: `uri_to_qname_and_xmlns` injects an inline `xmlns` declaration whose
    /// value is a slice of the predicate's own IRI text (the namespace half `split_iri` finds),
    /// reachable through `NamedNode::new_unchecked` exactly like `rdf:about`/`rdf:resource`; it
    /// used to go through the raw, unescaping `push_attribute` instead of
    /// [`push_escaped_attribute`]. A `\r` in it must now come out as `&#13;` rather than
    /// silently corrupting on re-parse, the same as every other attribute value in this file.
    #[test]
    fn predicate_xmlns_declaration_escapes_carriage_return() -> Result<(), Box<dyn Error>> {
        let mut serializer = RdfXmlSerializer::new().for_writer(Vec::new());
        serializer.serialize_triple(&Triple::new(
            NamedNode::new("http://example.com/s")?,
            NamedNode::new_unchecked("http://example.com/p\r/pred"),
            NamedNode::new("http://example.com/o")?,
        ))?;
        let output = serializer.finish()?;
        let text = String::from_utf8_lossy(&output);
        assert!(
            !text.contains('\r'),
            "a raw carriage return leaked through the predicate's xmlns declaration: {text}"
        );
        assert!(text.contains("&#13;"), "{text}");
        Ok(())
    }

    /// Same defect as [`predicate_xmlns_declaration_escapes_carriage_return`], reached through
    /// the other call site: an `rdf:type` object written as a typed node element name (instead
    /// of an `rdf:Description` with a plain `rdf:type` property) carries its own inline `xmlns`
    /// declaration the same way a predicate does.
    #[test]
    fn typed_node_xmlns_declaration_escapes_carriage_return() -> Result<(), Box<dyn Error>> {
        let mut serializer = RdfXmlSerializer::new().for_writer(Vec::new());
        serializer.serialize_triple(&Triple::new(
            NamedNode::new("http://example.com/s")?,
            rdf::TYPE,
            NamedNode::new_unchecked("http://example.com/T\r/Type"),
        ))?;
        let output = serializer.finish()?;
        let text = String::from_utf8_lossy(&output);
        assert!(
            !text.contains('\r'),
            "a raw carriage return leaked through the typed node's xmlns declaration: {text}"
        );
        assert!(text.contains("&#13;"), "{text}");
        Ok(())
    }

    /// Regression test: the same illegal-code-point rejection as
    /// [`attribute_value_with_illegal_xml_char_is_rejected`], but for the predicate's inline
    /// `xmlns` declaration (see [`predicate_xmlns_declaration_escapes_carriage_return`]).
    #[test]
    fn predicate_xmlns_declaration_rejects_illegal_xml_char() -> Result<(), Box<dyn Error>> {
        let mut serializer = RdfXmlSerializer::new().for_writer(Vec::new());
        let error = serializer
            .serialize_triple(&Triple::new(
                NamedNode::new("http://example.com/s")?,
                NamedNode::new_unchecked("http://example.com/p\u{0}/pred"),
                NamedNode::new("http://example.com/o")?,
            ))
            .expect_err("illegal XML character in the predicate's xmlns declaration");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        Ok(())
    }
}

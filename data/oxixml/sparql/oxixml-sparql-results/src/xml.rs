//! Implementation of the [SPARQL Query Results XML Format](https://www.w3.org/TR/rdf-sparql-XMLres/).

use crate::error::{QueryResultsParseError, QueryResultsSyntaxError};
#[cfg(feature = "rdf-12")]
use oxixml_model::BaseDirection;
use oxixml_model::vocab::{rdf, xsd};
use oxixml_model::{BlankNode, Literal, NamedNode, OxString, Term, Variable};
#[cfg(feature = "rdf-12")]
use oxixml_model::{NamedOrBlankNode, Triple};
use oxixml_xml::errors::Error;
use oxixml_xml::escape::{EscapeError, resolve_xml_entity};
use oxixml_xml::events::{BytesDecl, BytesEnd, BytesRef, BytesStart, BytesText, Event};
use oxixml_xml::reader::Reader;
use oxixml_xml::writer::Writer;
use oxixml_xml::{Decoder, XmlVersion};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{self, BufReader, Read, Write};
use std::mem::take;
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, AsyncWrite, BufReader as AsyncBufReader};

/// Writes a boolean SPARQL query result (from an `ASK` query) as SPARQL Results XML.
pub fn write_boolean_xml_result<W: Write>(writer: W, value: bool) -> io::Result<W> {
    let mut writer = Writer::new(writer);
    for event in inner_write_boolean_xml_result(value) {
        writer.write_event(event)?;
    }
    Ok(writer.into_inner())
}

/// Writes a boolean SPARQL query result (from an `ASK` query) as SPARQL Results XML to an
/// asynchronous sink.
#[cfg(feature = "async-tokio")]
pub async fn tokio_async_write_boolean_xml_result<W: AsyncWrite + Unpin>(
    writer: W,
    value: bool,
) -> io::Result<W> {
    let mut writer = Writer::new(writer);
    for event in inner_write_boolean_xml_result(value) {
        writer
            .write_event_async(event)
            .await
            .map_err(map_xml_write_err)?;
    }
    Ok(writer.into_inner())
}

/// Maps an `oxixml-xml` writer error back to [`io::Error`].
///
/// The synchronous [`Writer::write_event`] already returns [`io::Error`]; the asynchronous
/// counterpart returns the richer [`oxixml_xml::Error`], so the I/O variant is unwrapped back out
/// (recovering the original error whenever this is the only live reference) and anything else is
/// wrapped with [`io::Error::other`].
#[cfg(feature = "async-tokio")]
fn map_xml_write_err(error: Error) -> io::Error {
    if let Error::Io(error) = error {
        std::sync::Arc::try_unwrap(error)
            .unwrap_or_else(|error| io::Error::new(error.kind(), error))
    } else {
        io::Error::other(error)
    }
}

fn inner_write_boolean_xml_result(value: bool) -> [Event<'static>; 8] {
    [
        Event::Decl(BytesDecl::new("1.0", None, None)),
        Event::Start(
            BytesStart::new("sparql")
                .with_attributes([("xmlns", "http://www.w3.org/2005/sparql-results#")]),
        ),
        Event::Start(BytesStart::new("head")),
        Event::End(BytesEnd::new("head")),
        Event::Start(BytesStart::new("boolean")),
        Event::Text(BytesText::new(if value { "true" } else { "false" })),
        Event::End(BytesEnd::new("boolean")),
        Event::End(BytesEnd::new("sparql")),
    ]
}

/// Serializes [`QuerySolution`](crate::QuerySolution)s to a [`Write`] implementation as SPARQL
/// Results XML.
pub struct WriterXmlSolutionsSerializer<W: Write> {
    inner: InnerXmlSolutionsSerializer,
    writer: Writer<W>,
}

impl<W: Write> WriterXmlSolutionsSerializer<W> {
    pub(crate) fn start(writer: W, variables: &[Variable]) -> io::Result<Self> {
        let mut writer = Writer::new(writer);
        let mut buffer = Vec::with_capacity(48);
        let inner = InnerXmlSolutionsSerializer::start(&mut buffer, variables);
        Self::do_write(&mut writer, buffer)?;
        Ok(Self { inner, writer })
    }

    pub(crate) fn serialize<'a>(
        &mut self,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) -> io::Result<()> {
        let mut buffer = Vec::with_capacity(48);
        self.inner.write(&mut buffer, solution);
        Self::do_write(&mut self.writer, buffer)
    }

    pub(crate) fn finish(mut self) -> io::Result<W> {
        let mut buffer = Vec::with_capacity(4);
        self.inner.finish(&mut buffer);
        Self::do_write(&mut self.writer, buffer)?;
        Ok(self.writer.into_inner())
    }

    fn do_write(writer: &mut Writer<W>, output: Vec<Event<'_>>) -> io::Result<()> {
        for event in output {
            writer.write_event(event)?;
        }
        Ok(())
    }
}

struct InnerXmlSolutionsSerializer;

impl InnerXmlSolutionsSerializer {
    fn start<'a>(output: &mut Vec<Event<'a>>, variables: &'a [Variable]) -> Self {
        output.push(Event::Decl(BytesDecl::new("1.0", None, None)));
        output.push(Event::Start(BytesStart::new("sparql").with_attributes([(
            "xmlns",
            "http://www.w3.org/2005/sparql-results#",
        )])));
        output.push(Event::Start(BytesStart::new("head")));
        for variable in variables {
            output.push(Event::Empty(
                BytesStart::new("variable").with_attributes([("name", variable.as_str())]),
            ));
        }
        output.push(Event::End(BytesEnd::new("head")));
        output.push(Event::Start(BytesStart::new("results")));
        Self {}
    }

    fn write<'a>(
        &self,
        output: &mut Vec<Event<'a>>,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) {
        output.push(Event::Start(BytesStart::new("result")));
        for (variable, value) in solution {
            // The `<binding>` wrapper is pushed speculatively and rolled back if `write_xml_term`
            // reports that `value` has no representation in this build configuration: per the
            // SPARQL Query Results XML Format, a `<binding>` element must contain exactly one of
            // uri/literal/bnode/triple, so an empty wrapper would be invalid XML.
            let binding_start = output.len();
            output.push(Event::Start(
                BytesStart::new("binding").with_attributes([("name", variable.as_str())]),
            ));
            if write_xml_term(output, value) {
                output.push(Event::End(BytesEnd::new("binding")));
            } else {
                output.truncate(binding_start);
            }
        }
        output.push(Event::End(BytesEnd::new("result")));
    }

    fn finish(self, output: &mut Vec<Event<'_>>) {
        output.push(Event::End(BytesEnd::new("results")));
        output.push(Event::End(BytesEnd::new("sparql")));
    }
}

/// Writes `term` as a SPARQL Results XML term element.
///
/// Returns `true` if a value was written, or `false` if `term` has no representation in this
/// build configuration (see the fallback arm below), in which case the caller must not emit an
/// empty `<binding>` wrapper.
fn write_xml_term<'a>(output: &mut Vec<Event<'a>>, term: &'a Term) -> bool {
    match term {
        Term::NamedNode(uri) => {
            write_xml_named_node(output, uri);
            true
        }
        Term::BlankNode(bnode) => {
            write_xml_blank_node(output, bnode);
            true
        }
        Term::Literal(literal) => {
            let mut start = BytesStart::new("literal");
            if let Some(language) = literal.language() {
                start.push_attribute(("xml:lang", language));
                #[cfg(feature = "rdf-12")]
                if let Some(direction) = literal.direction() {
                    start.push_attribute((
                        "its:dir",
                        match direction {
                            BaseDirection::Ltr => "ltr",
                            BaseDirection::Rtl => "rtl",
                        },
                    ));
                    start.push_attribute(("xmlns:its", "http://www.w3.org/2005/11/its"));
                    start.push_attribute(("its:version", "2.0"));
                }
            } else if literal.datatype() != xsd::STRING {
                start.push_attribute(("datatype", literal.datatype().as_str()));
            }
            output.push(Event::Start(start));
            output.push(Event::Text(BytesText::from_escaped(
                escape_including_bound_whitespaces(literal.value()),
            )));
            output.push(Event::End(BytesEnd::new("literal")));
            true
        }
        #[cfg(feature = "rdf-12")]
        Term::Triple(triple) => {
            output.push(Event::Start(BytesStart::new("triple")));
            output.push(Event::Start(BytesStart::new("subject")));
            match &triple.subject {
                NamedOrBlankNode::NamedNode(uri) => write_xml_named_node(output, uri),
                NamedOrBlankNode::BlankNode(bnode) => write_xml_blank_node(output, bnode),
            }
            output.push(Event::End(BytesEnd::new("subject")));
            output.push(Event::Start(BytesStart::new("predicate")));
            write_xml_named_node(output, &triple.predicate);
            output.push(Event::End(BytesEnd::new("predicate")));
            output.push(Event::Start(BytesStart::new("object")));
            write_xml_term(output, &triple.object);
            output.push(Event::End(BytesEnd::new("object")));
            output.push(Event::End(BytesEnd::new("triple")));
            true
        }
        // `Term::Triple` exists as soon as `oxixml-model` is built with its own `rdf-12` feature,
        // which any other crate of the build may switch on through Cargo feature unification even
        // when this crate's `rdf-12` feature is off. Without this arm the match would then stop
        // being exhaustive and the crate would fail to compile. Per the SPARQL Query Results XML
        // Format a `<binding>` element must contain exactly one of uri/literal/bnode/triple, so
        // report "wrote nothing" to the caller instead of emitting an empty term element.
        #[allow(
            unreachable_patterns,
            reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build without this crate's own rdf-12"
        )]
        _ => false,
    }
}

fn write_xml_named_node<'a>(output: &mut Vec<Event<'a>>, uri: &'a NamedNode) {
    output.push(Event::Start(BytesStart::new("uri")));
    output.push(Event::Text(BytesText::new(uri.as_str())));
    output.push(Event::End(BytesEnd::new("uri")));
}

fn write_xml_blank_node<'a>(output: &mut Vec<Event<'a>>, bnode: &'a BlankNode) {
    output.push(Event::Start(BytesStart::new("bnode")));
    output.push(Event::Text(BytesText::new(bnode.as_str())));
    output.push(Event::End(BytesEnd::new("bnode")));
}

/// Serializes [`QuerySolution`](crate::QuerySolution)s to an [`AsyncWrite`] implementation as
/// SPARQL Results XML.
#[cfg(feature = "async-tokio")]
pub struct TokioAsyncWriterXmlSolutionsSerializer<W: AsyncWrite + Unpin> {
    inner: InnerXmlSolutionsSerializer,
    writer: Writer<W>,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterXmlSolutionsSerializer<W> {
    pub(crate) async fn start(writer: W, variables: &[Variable]) -> io::Result<Self> {
        let mut writer = Writer::new(writer);
        let mut buffer = Vec::with_capacity(48);
        let inner = InnerXmlSolutionsSerializer::start(&mut buffer, variables);
        Self::do_write(&mut writer, buffer).await?;
        Ok(Self { inner, writer })
    }

    pub(crate) async fn serialize<'a>(
        &mut self,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) -> io::Result<()> {
        let mut buffer = Vec::with_capacity(48);
        self.inner.write(&mut buffer, solution);
        Self::do_write(&mut self.writer, buffer).await
    }

    pub(crate) async fn finish(mut self) -> io::Result<W> {
        let mut buffer = Vec::with_capacity(4);
        self.inner.finish(&mut buffer);
        Self::do_write(&mut self.writer, buffer).await?;
        Ok(self.writer.into_inner())
    }

    async fn do_write(writer: &mut Writer<W>, output: Vec<Event<'_>>) -> io::Result<()> {
        for event in output {
            writer
                .write_event_async(event)
                .await
                .map_err(map_xml_write_err)?;
        }
        Ok(())
    }
}

/// The result of reading a SPARQL Results XML document from a [`Read`] implementation.
#[allow(clippy::large_enum_variant)]
pub enum ReaderXmlQueryResultsParserOutput<R: Read> {
    /// A set of solutions with their declared variables.
    Solutions {
        /// The declared variables, in the order they were declared.
        variables: Vec<Variable>,
        /// The streaming parser producing the solutions.
        solutions: ReaderXmlSolutionsParser<R>,
    },
    /// A boolean, from an `ASK` query.
    Boolean(bool),
}

impl<R: Read> ReaderXmlQueryResultsParserOutput<R> {
    pub(crate) fn read(reader: R) -> Result<Self, QueryResultsParseError> {
        let mut reader = Reader::from_reader(BufReader::new(reader));
        reader.config_mut().expand_empty_elements = true;
        let mut reader_buffer = Vec::new();
        let mut inner = XmlInnerQueryResultsParser {
            state: ResultsState::Start,
            variables: Vec::new(),
            decoder: reader.decoder(),
            text_buffer: String::new(),
            xml_version: XmlVersion::Implicit1_0,
        };
        loop {
            reader_buffer.clear();
            let event = reader.read_event_into(&mut reader_buffer)?;
            if let Some(result) = inner.read_event(event)? {
                return Ok(match result {
                    XmlInnerQueryResults::Solutions {
                        variables,
                        solutions,
                    } => Self::Solutions {
                        variables,
                        solutions: ReaderXmlSolutionsParser {
                            reader,
                            inner: solutions,
                            reader_buffer,
                        },
                    },
                    XmlInnerQueryResults::Boolean(value) => Self::Boolean(value),
                });
            }
        }
    }
}

/// A streaming parser of a set of solutions read from a [`Read`] implementation.
pub struct ReaderXmlSolutionsParser<R: Read> {
    reader: Reader<BufReader<R>>,
    inner: XmlInnerSolutionsParser,
    reader_buffer: Vec<u8>,
}

impl<R: Read> ReaderXmlSolutionsParser<R> {
    pub(crate) fn parse_next(
        &mut self,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsParseError> {
        loop {
            self.reader_buffer.clear();
            let event = self.reader.read_event_into(&mut self.reader_buffer)?;
            if event == Event::Eof {
                return Ok(None);
            }
            if let Some(solution) = self.inner.read_event(event)? {
                return Ok(Some(solution));
            }
        }
    }
}

/// The result of reading a SPARQL Results XML document from an [`AsyncRead`] implementation.
#[cfg(feature = "async-tokio")]
#[allow(clippy::large_enum_variant)]
pub enum TokioAsyncReaderXmlQueryResultsParserOutput<R: AsyncRead + Unpin> {
    /// A set of solutions with their declared variables.
    Solutions {
        /// The declared variables, in the order they were declared.
        variables: Vec<Variable>,
        /// The streaming parser producing the solutions.
        solutions: TokioAsyncReaderXmlSolutionsParser<R>,
    },
    /// A boolean, from an `ASK` query.
    Boolean(bool),
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderXmlQueryResultsParserOutput<R> {
    pub(crate) async fn read(reader: R) -> Result<Self, QueryResultsParseError> {
        let mut reader = Reader::from_reader(AsyncBufReader::new(reader));
        reader.config_mut().expand_empty_elements = true;
        let mut reader_buffer = Vec::new();
        let mut inner = XmlInnerQueryResultsParser {
            state: ResultsState::Start,
            variables: Vec::new(),
            decoder: reader.decoder(),
            text_buffer: String::new(),
            xml_version: XmlVersion::Implicit1_0,
        };
        loop {
            reader_buffer.clear();
            let event = reader.read_event_into_async(&mut reader_buffer).await?;
            if let Some(result) = inner.read_event(event)? {
                return Ok(match result {
                    XmlInnerQueryResults::Solutions {
                        variables,
                        solutions,
                    } => Self::Solutions {
                        variables,
                        solutions: TokioAsyncReaderXmlSolutionsParser {
                            reader,
                            inner: solutions,
                            reader_buffer,
                        },
                    },
                    XmlInnerQueryResults::Boolean(value) => Self::Boolean(value),
                });
            }
        }
    }
}

/// A streaming parser of a set of solutions read from an [`AsyncRead`] implementation.
#[cfg(feature = "async-tokio")]
pub struct TokioAsyncReaderXmlSolutionsParser<R: AsyncRead + Unpin> {
    reader: Reader<AsyncBufReader<R>>,
    inner: XmlInnerSolutionsParser,
    reader_buffer: Vec<u8>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderXmlSolutionsParser<R> {
    pub(crate) async fn parse_next(
        &mut self,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsParseError> {
        loop {
            self.reader_buffer.clear();
            let event = self
                .reader
                .read_event_into_async(&mut self.reader_buffer)
                .await?;
            if event == Event::Eof {
                return Ok(None);
            }
            if let Some(solution) = self.inner.read_event(event)? {
                return Ok(Some(solution));
            }
        }
    }
}

/// The result of reading a SPARQL Results XML document from an in-memory byte slice.
#[allow(clippy::large_enum_variant)]
pub enum SliceXmlQueryResultsParserOutput<'a> {
    /// A set of solutions with their declared variables.
    Solutions {
        /// The declared variables, in the order they were declared.
        variables: Vec<Variable>,
        /// The streaming parser producing the solutions.
        solutions: SliceXmlSolutionsParser<'a>,
    },
    /// A boolean, from an `ASK` query.
    Boolean(bool),
}

impl<'a> SliceXmlQueryResultsParserOutput<'a> {
    pub(crate) fn read(slice: &'a [u8]) -> Result<Self, QueryResultsSyntaxError> {
        Self::do_read(slice).map_err(|e| match e {
            QueryResultsParseError::Syntax(e) => e,
            QueryResultsParseError::Io(e) => {
                QueryResultsSyntaxError::msg(format!("Unexpected I/O error on a slice: {e}"))
            }
        })
    }

    fn do_read(slice: &'a [u8]) -> Result<Self, QueryResultsParseError> {
        let mut reader = Reader::from_reader(slice);
        reader.config_mut().expand_empty_elements = true;
        let mut reader_buffer = Vec::new();
        let mut inner = XmlInnerQueryResultsParser {
            state: ResultsState::Start,
            variables: Vec::new(),
            decoder: reader.decoder(),
            text_buffer: String::new(),
            xml_version: XmlVersion::Implicit1_0,
        };
        loop {
            reader_buffer.clear();
            let event = reader.read_event_into(&mut reader_buffer)?;
            if let Some(result) = inner.read_event(event)? {
                return Ok(match result {
                    XmlInnerQueryResults::Solutions {
                        variables,
                        solutions,
                    } => Self::Solutions {
                        variables,
                        solutions: SliceXmlSolutionsParser {
                            reader,
                            inner: solutions,
                            reader_buffer,
                        },
                    },
                    XmlInnerQueryResults::Boolean(value) => Self::Boolean(value),
                });
            }
        }
    }
}

/// A streaming parser of a set of solutions read from an in-memory byte slice.
pub struct SliceXmlSolutionsParser<'a> {
    reader: Reader<&'a [u8]>,
    inner: XmlInnerSolutionsParser,
    reader_buffer: Vec<u8>,
}

impl SliceXmlSolutionsParser<'_> {
    pub(crate) fn parse_next(
        &mut self,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsSyntaxError> {
        self.do_parse_next().map_err(|e| match e {
            QueryResultsParseError::Syntax(e) => e,
            QueryResultsParseError::Io(e) => {
                QueryResultsSyntaxError::msg(format!("Unexpected I/O error on a slice: {e}"))
            }
        })
    }

    fn do_parse_next(&mut self) -> Result<Option<Vec<Option<Term>>>, QueryResultsParseError> {
        loop {
            self.reader_buffer.clear();
            let event = self.reader.read_event_into(&mut self.reader_buffer)?;
            if event == Event::Eof {
                return Ok(None);
            }
            if let Some(solution) = self.inner.read_event(event)? {
                return Ok(Some(solution));
            }
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum XmlInnerQueryResults {
    Solutions {
        variables: Vec<Variable>,
        solutions: XmlInnerSolutionsParser,
    },
    Boolean(bool),
}

#[derive(Clone, Copy)]
enum ResultsState {
    Start,
    Sparql,
    Head,
    AfterHead,
    Boolean,
}

struct XmlInnerQueryResultsParser {
    state: ResultsState,
    variables: Vec<Variable>,
    decoder: Decoder,
    text_buffer: String,
    xml_version: XmlVersion,
}

impl XmlInnerQueryResultsParser {
    fn read_event(
        &mut self,
        event: Event<'_>,
    ) -> Result<Option<XmlInnerQueryResults>, QueryResultsParseError> {
        match event {
            Event::Start(event) => match self.state {
                ResultsState::Start => {
                    if event.local_name().as_ref() == b"sparql" {
                        self.state = ResultsState::Sparql;
                        Ok(None)
                    } else {
                        Err(QueryResultsSyntaxError::msg(format!(
                            "Expecting <sparql> tag, found <{}>",
                            self.decoder.decode(event.name().as_ref())?
                        ))
                        .into())
                    }
                }
                ResultsState::Sparql => {
                    if event.local_name().as_ref() == b"head" {
                        self.state = ResultsState::Head;
                        Ok(None)
                    } else {
                        Err(QueryResultsSyntaxError::msg(format!(
                            "Expecting <head> tag, found <{}>",
                            self.decoder.decode(event.name().as_ref())?
                        ))
                        .into())
                    }
                }
                ResultsState::Head => {
                    if event.local_name().as_ref() == b"variable" {
                        let name = event
                            .attributes()
                            .filter_map(Result::ok)
                            .find(|attr| attr.key.local_name().as_ref() == b"name")
                            .ok_or_else(|| {
                                QueryResultsSyntaxError::msg(
                                    "No name attribute found for the <variable> tag",
                                )
                            })?;
                        let name = name.decoded_and_normalized_value(self.xml_version, self.decoder)?;
                        let variable = Variable::new(OxString::new_owned(&name)).map_err(|e| {
                            QueryResultsSyntaxError::msg(format!("Invalid variable name: {e}"))
                        })?;
                        if self.variables.contains(&variable) {
                            return Err(QueryResultsSyntaxError::msg(format!(
                                "The variable {variable} is declared twice"
                            ))
                            .into());
                        }
                        self.variables.push(variable);
                        Ok(None)
                    } else if event.local_name().as_ref() == b"link" {
                        Ok(None)
                    } else {
                        Err(QueryResultsSyntaxError::msg(format!(
                            "Expecting <variable> or <link> tag, found <{}>",
                            self.decoder.decode(event.name().as_ref())?
                        ))
                        .into())
                    }
                }
                ResultsState::AfterHead => {
                    if event.local_name().as_ref() == b"boolean" {
                        self.state = ResultsState::Boolean;
                        Ok(None)
                    } else if event.local_name().as_ref() == b"results" {
                        let mut mapping = HashMap::new();
                        for (i, var) in self.variables.iter().enumerate() {
                            mapping.insert(var.clone().into_string(), i);
                        }
                        Ok(Some(XmlInnerQueryResults::Solutions {
                            variables: take(&mut self.variables),
                            solutions: XmlInnerSolutionsParser {
                                decoder: self.decoder,
                                mapping,
                                state_stack: vec![State::Start, State::Start],
                                new_bindings: Vec::new(),
                                current_var: None,
                                term: None,
                                lang: None,
                                #[cfg(feature = "rdf-12")]
                                direction: None,
                                datatype: None,
                                subject_stack: Vec::new(),
                                predicate_stack: Vec::new(),
                                object_stack: Vec::new(),
                                text_buffer: String::new(),
                                xml_version: self.xml_version,
                                result_had_error: false,
                            },
                        }))
                    } else if event.local_name().as_ref() != b"link"
                        && event.local_name().as_ref() != b"results"
                        && event.local_name().as_ref() != b"boolean"
                    {
                        Err(QueryResultsSyntaxError::msg(format!(
                            "Expecting sparql tag, found <{}>",
                            self.decoder.decode(event.name().as_ref())?
                        ))
                        .into())
                    } else {
                        Ok(None)
                    }
                }
                ResultsState::Boolean => Err(QueryResultsSyntaxError::msg(format!(
                    "Unexpected tag inside of <boolean> tag: <{}>",
                    self.decoder.decode(event.name().as_ref())?
                ))
                .into()),
            },
            Event::Text(event) => {
                self.text_buffer.push_str(&event.xml_content(self.xml_version)?);
                Ok(None)
            }
            Event::GeneralRef(event) => {
                decode_xml_entity(&event, &mut self.text_buffer, self.xml_version)?;
                Ok(None)
            }
            Event::End(event) => {
                let value = take(&mut self.text_buffer);
                let value = value.trim_matches(|c| matches!(c, '\t' | '\n' | '\r' | ' '));
                match self.state {
                    ResultsState::Boolean => {
                        if value == "true" {
                            Ok(Some(XmlInnerQueryResults::Boolean(true)))
                        } else if value == "false" {
                            Ok(Some(XmlInnerQueryResults::Boolean(false)))
                        } else {
                            Err(QueryResultsSyntaxError::msg(format!(
                                "Unexpected boolean value. Found '{value}'"
                            ))
                            .into())
                        }
                    }
                    ResultsState::Head => {
                        if event.local_name().as_ref() == b"head" {
                            self.state = ResultsState::AfterHead;
                        }
                        Ok(None)
                    }
                    _ => {
                        if value.is_empty() {
                            Err(QueryResultsSyntaxError::msg(
                                "Unexpected early file end. All results file must have a <head> and a <result> or <boolean> tag",
                            )
                            .into())
                        } else {
                            Err(QueryResultsSyntaxError::msg(format!(
                                "Unexpected textual value found: '{value}'"
                            ))
                            .into())
                        }
                    }
                }
            }
            Event::Eof => Err(QueryResultsSyntaxError::msg(
                "Unexpected early file end. All results file must have a <head> and a <result> or <boolean> tag",
            )
            .into()),
            Event::Decl(event) => {
                self.xml_version = event.xml_version()?;
                Ok(None)
            }
            Event::Comment(_) | Event::PI(_) | Event::DocType(_) => Ok(None),
            Event::Empty(_) => unreachable!("Empty events are expanded"),
            Event::CData(_) => Err(QueryResultsSyntaxError::msg(
                "<![CDATA[...]]> are not supported in SPARQL XML results",
            )
            .into()),
        }
    }
}

enum State {
    Start,
    Result,
    Binding,
    Uri,
    BNode,
    Literal,
    Triple,
    Subject,
    Predicate,
    Object,
    /// An element rejected while parsing the currently open `<result>` (see
    /// [`XmlInnerSolutionsParser::skip_rejected_element`]): its own end tag has not been read
    /// yet, and it may hold further nested elements of its own, so its whole subtree is
    /// discarded one end tag at a time rather than interpreted.
    Skip,
}

struct XmlInnerSolutionsParser {
    decoder: Decoder,
    mapping: HashMap<OxString, usize>,
    state_stack: Vec<State>,
    new_bindings: Vec<Option<Term>>,
    current_var: Option<OxString>,
    term: Option<Term>,
    lang: Option<OxString>,
    #[cfg(feature = "rdf-12")]
    direction: Option<BaseDirection>,
    datatype: Option<NamedNode>,
    subject_stack: Vec<Term>,
    predicate_stack: Vec<Term>,
    object_stack: Vec<Term>,
    text_buffer: String,
    xml_version: XmlVersion,
    /// Set when something inside the currently open `<result>` was rejected: a `<binding>` with
    /// a missing `name` attribute or an undeclared variable, a second value for one binding, an
    /// unexpected child element, or an invalid attribute on one of `<literal>`'s own.
    ///
    /// A rejected element reports its own error immediately, but the surrounding `<result>`
    /// element itself is still open on `state_stack` and its closing tag has not been read yet:
    /// without this flag, that `</result>` would later close normally and hand back whatever
    /// `new_bindings` happens to hold (a spurious, incomplete solution) instead of the caller
    /// ever getting a chance to see this `<result>` as the failure it was. Checked and cleared by
    /// `</result>`'s own handling, so it never leaks into the next, unrelated `<result>`.
    result_had_error: bool,
}

impl XmlInnerSolutionsParser {
    /// Reports `self.result_had_error` and pushes [`State::Skip`] in place of whatever frame the
    /// element being rejected would otherwise have gotten.
    ///
    /// Call this immediately before failing on an `Event::Start` from within the currently open
    /// `<result>`: the rejected element's `Event::Start` has already been consumed by the
    /// underlying XML reader, and it still has its own matching end tag later in the stream (and
    /// possibly further nested elements of its own) — without a frame of its own, `state_stack`
    /// would fall one frame short of the real element nesting, and every end tag from there on
    /// would pop the *wrong* frame instead of the one it actually closes.
    fn skip_rejected_element(&mut self) {
        self.result_had_error = true;
        self.state_stack.push(State::Skip);
    }

    /// Parses a `<literal>` start tag's `xml:lang`, `datatype` and (`rdf-12` only) `its:dir`
    /// attributes into `self.lang`/`self.datatype`/`self.direction`.
    ///
    /// Pulled out of [`Self::read_event`] so that every attribute-level failure (malformed
    /// attribute syntax, an invalid `datatype` IRI, an `its:dir` value that is neither `ltr` nor
    /// `rtl`, …) is caught by the *same* caller-side check, rather than needing
    /// [`Self::skip_rejected_element`] repeated at each one: the caller already pushes
    /// [`State::Literal`] before calling this, so all this needs to report on failure is the
    /// error itself.
    fn parse_literal_attributes(
        &mut self,
        event: &BytesStart<'_>,
    ) -> Result<(), QueryResultsParseError> {
        for attr in event.attributes() {
            let attr = attr.map_err(Error::from)?;
            if attr.key.as_ref() == b"xml:lang" {
                self.lang = Some(OxString::new_owned(
                    &attr.decoded_and_normalized_value(self.xml_version, self.decoder)?,
                ));
            } else if attr.key.local_name().as_ref() == b"datatype" {
                let iri = attr.decoded_and_normalized_value(self.xml_version, self.decoder)?;
                self.datatype = Some(NamedNode::new(OxString::new_owned(&iri)).map_err(|e| {
                    QueryResultsSyntaxError::msg(format!("Invalid datatype IRI '{iri}': {e}"))
                })?);
            }
            #[cfg(feature = "rdf-12")]
            if attr.key.as_ref() == b"its:dir" {
                let value = attr.decoded_and_normalized_value(self.xml_version, self.decoder)?;
                self.direction = Some(match value.as_ref() {
                    "ltr" => BaseDirection::Ltr,
                    "rtl" => BaseDirection::Rtl,
                    _ => {
                        return Err(QueryResultsSyntaxError::msg(format!(
                            "Invalid its:dir value '{value}', expecting 'ltr' or 'rtl'"
                        ))
                        .into());
                    }
                });
            }
        }
        Ok(())
    }

    fn read_event(
        &mut self,
        event: Event<'_>,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsParseError> {
        match event {
            Event::Start(event) => match self.state_stack.last().ok_or_else(|| {
                QueryResultsSyntaxError::msg("Extra XML is not allowed at the end of the document")
            })? {
                State::Start => {
                    if event.local_name().as_ref() == b"result" {
                        self.new_bindings = vec![None; self.mapping.len()];
                        self.state_stack.push(State::Result);
                        Ok(None)
                    } else {
                        Err(QueryResultsSyntaxError::msg(format!(
                            "Expecting <result>, found <{}>",
                            self.decoder.decode(event.name().as_ref())?
                        ))
                        .into())
                    }
                }
                State::Result => {
                    if event.local_name().as_ref() == b"binding" {
                        let Some(attr) = event
                            .attributes()
                            .filter_map(Result::ok)
                            .find(|attr| attr.key.local_name().as_ref() == b"name")
                        else {
                            self.skip_rejected_element();
                            return Err(QueryResultsSyntaxError::msg(
                                "No name attribute found for the <binding> tag",
                            )
                            .into());
                        };
                        self.current_var = Some(OxString::new_owned(
                            &attr.decoded_and_normalized_value(self.xml_version, self.decoder)?,
                        ));
                        self.state_stack.push(State::Binding);
                        Ok(None)
                    } else {
                        self.skip_rejected_element();
                        Err(QueryResultsSyntaxError::msg(format!(
                            "Expecting <binding>, found <{}>",
                            self.decoder.decode(event.name().as_ref())?
                        ))
                        .into())
                    }
                }
                State::Binding | State::Subject | State::Predicate | State::Object => {
                    if self.term.is_some() {
                        self.skip_rejected_element();
                        return Err(QueryResultsSyntaxError::msg(
                            "There is already a value for the current binding",
                        )
                        .into());
                    }
                    if event.local_name().as_ref() == b"uri" {
                        self.state_stack.push(State::Uri);
                        Ok(None)
                    } else if event.local_name().as_ref() == b"bnode" {
                        self.state_stack.push(State::BNode);
                        Ok(None)
                    } else if event.local_name().as_ref() == b"literal" {
                        // Pushed unconditionally, *before* the fallible attribute parsing below:
                        // `<literal>`'s own end tag is coming regardless of whether its attributes
                        // turn out to be well-formed, so `state_stack` needs a frame for it either
                        // way. `result_had_error` (set on the error path below) makes sure the
                        // enclosing `<result>` still gets discarded if they are not.
                        self.state_stack.push(State::Literal);
                        if let Err(e) = self.parse_literal_attributes(&event) {
                            self.result_had_error = true;
                            return Err(e);
                        }
                        Ok(None)
                    } else if event.local_name().as_ref() == b"triple" {
                        self.state_stack.push(State::Triple);
                        Ok(None)
                    } else {
                        self.skip_rejected_element();
                        Err(QueryResultsSyntaxError::msg(format!(
                            "Expecting <uri>, <bnode> or <literal> found <{}>",
                            self.decoder.decode(event.name().as_ref())?
                        ))
                        .into())
                    }
                }
                State::Triple => {
                    if event.local_name().as_ref() == b"subject" {
                        self.state_stack.push(State::Subject);
                        Ok(None)
                    } else if event.local_name().as_ref() == b"predicate" {
                        self.state_stack.push(State::Predicate);
                        Ok(None)
                    } else if event.local_name().as_ref() == b"object" {
                        self.state_stack.push(State::Object);
                        Ok(None)
                    } else {
                        self.skip_rejected_element();
                        Err(QueryResultsSyntaxError::msg(format!(
                            "Expecting <subject>, <predicate> or <object> found <{}>",
                            self.decoder.decode(event.name().as_ref())?
                        ))
                        .into())
                    }
                }
                State::Uri => {
                    self.skip_rejected_element();
                    Err(QueryResultsSyntaxError::msg(format!(
                        "<uri> must only contain a string, found <{}>",
                        self.decoder.decode(event.name().as_ref())?
                    ))
                    .into())
                }
                State::BNode => {
                    self.skip_rejected_element();
                    Err(QueryResultsSyntaxError::msg(format!(
                        "<bnode> must only contain a string, found <{}>",
                        self.decoder.decode(event.name().as_ref())?
                    ))
                    .into())
                }
                State::Literal => {
                    self.skip_rejected_element();
                    Err(QueryResultsSyntaxError::msg(format!(
                        "<literal> must only contain a string, found <{}>",
                        self.decoder.decode(event.name().as_ref())?
                    ))
                    .into())
                }
                State::Skip => {
                    // Already discarding a rejected element's content: whatever nested element
                    // appears next is part of what is being thrown away, so it gets a skip frame
                    // of its own instead of being checked against any expected tag name.
                    self.state_stack.push(State::Skip);
                    Ok(None)
                }
            },
            Event::Text(event) => {
                self.text_buffer
                    .push_str(&event.xml_content(self.xml_version)?);
                Ok(None)
            }
            Event::End(_) => {
                let value = take(&mut self.text_buffer);
                let value = OxString::new_owned(
                    value.trim_matches(|c| matches!(c, '\t' | '\n' | '\r' | ' ')),
                );
                match self.state_stack.pop().ok_or_else(|| {
                    QueryResultsSyntaxError::msg(
                        "Extra XML is not allowed at the end of the document",
                    )
                })? {
                    State::Start => Ok(None),
                    State::Result => {
                        if take(&mut self.result_had_error) {
                            // This `<result>` already reported its own error, at the malformed
                            // `<binding>` itself: closing it normally here would otherwise hand
                            // back whatever partial `new_bindings` it built up as if it were a
                            // genuine (if unusually sparse) solution. `new_bindings` is
                            // reinitialized from scratch the next time a `<result>` opens (see
                            // `State::Start` above), so there is nothing to clean up here beyond
                            // not returning it.
                            Ok(None)
                        } else {
                            Ok(Some(take(&mut self.new_bindings)))
                        }
                    }
                    State::Binding => {
                        let Some(var) = &self.current_var else {
                            // Clear the scratch fields a nested `<uri>`/`<bnode>`/`<literal>`/
                            // `<triple>` may have already populated: leaving them set would make
                            // the very next binding's opening check (`self.term.is_some()`) find
                            // this discarded value and spuriously reject an otherwise well-formed
                            // `<result>`.
                            self.clear_binding_scratch();
                            self.result_had_error = true;
                            return Err(QueryResultsSyntaxError::msg(
                                "No name found for <binding> tag",
                            )
                            .into());
                        };
                        let Some(index) = self.mapping.get(var.as_str()) else {
                            let var = var.clone();
                            self.clear_binding_scratch();
                            self.result_had_error = true;
                            return Err(QueryResultsSyntaxError::msg(format!(
                                "The variable '{var}' is used in a binding but not declared in the variables list"
                            ))
                            .into());
                        };
                        self.new_bindings[*index] = self.term.take();
                        Ok(None)
                    }
                    State::Subject => {
                        if let Some(subject) = self.term.take() {
                            self.subject_stack.push(subject);
                        }
                        Ok(None)
                    }
                    State::Predicate => {
                        if let Some(predicate) = self.term.take() {
                            self.predicate_stack.push(predicate);
                        }
                        Ok(None)
                    }
                    State::Object => {
                        if let Some(object) = self.term.take() {
                            self.object_stack.push(object);
                        }
                        Ok(None)
                    }
                    State::Uri => match NamedNode::new(value.clone()) {
                        Ok(node) => {
                            self.term = Some(node.into());
                            Ok(None)
                        }
                        Err(e) => {
                            // Unlike an `Event::Start` rejection, `state_stack` is already back in
                            // sync (the pop above already happened unconditionally): the *only*
                            // thing left to signal is that the enclosing `<result>` must not hand
                            // back `new_bindings` as if this value had parsed, once its own
                            // `</result>` is reached.
                            self.result_had_error = true;
                            Err(QueryResultsSyntaxError::msg(format!(
                                "Invalid IRI value '{value}': {e}"
                            ))
                            .into())
                        }
                    },
                    State::BNode => {
                        let bnode = if value.as_str().is_empty() {
                            Ok(BlankNode::default())
                        } else {
                            BlankNode::new(value.clone())
                        };
                        match bnode {
                            Ok(bnode) => {
                                self.term = Some(bnode.into());
                                Ok(None)
                            }
                            Err(e) => {
                                self.result_had_error = true;
                                Err(QueryResultsSyntaxError::msg(format!(
                                    "Invalid blank node value '{value}': {e}"
                                ))
                                .into())
                            }
                        }
                    }
                    State::Literal => match build_literal(
                        value,
                        self.lang.take(),
                        #[cfg(feature = "rdf-12")]
                        self.direction.take(),
                        self.datatype.take(),
                    ) {
                        Ok(literal) => {
                            self.term = Some(literal.into());
                            Ok(None)
                        }
                        Err(e) => {
                            self.result_had_error = true;
                            Err(e.into())
                        }
                    },
                    State::Triple => {
                        #[cfg(feature = "rdf-12")]
                        {
                            if let (Some(subject), Some(predicate), Some(object)) = (
                                self.subject_stack.pop(),
                                self.predicate_stack.pop(),
                                self.object_stack.pop(),
                            ) {
                                self.term = Some(
                                    Triple::new(
                                        match subject {
                                            Term::NamedNode(subject) => {
                                                NamedOrBlankNode::from(subject)
                                            }
                                            Term::BlankNode(subject) => {
                                                NamedOrBlankNode::from(subject)
                                            }
                                            Term::Triple(_) => {
                                                self.result_had_error = true;
                                                return Err(QueryResultsSyntaxError::msg(
                                                    "The <subject> value cannot be a <triple>",
                                                )
                                                .into());
                                            }
                                            Term::Literal(_) => {
                                                self.result_had_error = true;
                                                return Err(QueryResultsSyntaxError::msg(
                                                    "The <subject> value cannot be a <literal>",
                                                )
                                                .into());
                                            }
                                        },
                                        if let Term::NamedNode(predicate) = predicate {
                                            predicate
                                        } else {
                                            self.result_had_error = true;
                                            return Err(QueryResultsSyntaxError::msg(
                                                "The <predicate> value must be an <uri>",
                                            )
                                            .into());
                                        },
                                        object,
                                    )
                                    .into(),
                                );
                                Ok(None)
                            } else {
                                self.result_had_error = true;
                                Err(QueryResultsSyntaxError::msg(
                                    "A <triple> must contain a <subject>, a <predicate> and an <object>",
                                )
                                .into())
                            }
                        }
                        #[cfg(not(feature = "rdf-12"))]
                        {
                            self.result_had_error = true;
                            Err(QueryResultsSyntaxError::msg(
                                "The <triple> tag is only supported when the `rdf-12` feature is enabled",
                            )
                            .into())
                        }
                    }
                    State::Skip => Ok(None),
                }
            }
            Event::Decl(event) => {
                self.xml_version = event.xml_version()?;
                Ok(None)
            }
            Event::Eof | Event::Comment(_) | Event::PI(_) | Event::DocType(_) => Ok(None),
            Event::GeneralRef(event) => {
                decode_xml_entity(&event, &mut self.text_buffer, self.xml_version)?;
                Ok(None)
            }
            Event::Empty(_) => unreachable!("Empty events are expanded"),
            Event::CData(_) => Err(QueryResultsSyntaxError::msg(
                "<![CDATA[...]]> are not supported in SPARQL XML results",
            )
            .into()),
        }
    }

    /// Resets the per-binding scratch fields to their initial, empty state.
    ///
    /// Called when a `<binding>` element is rejected (an undeclared variable, or a missing
    /// `name` attribute) after a nested `<uri>`/`<bnode>`/`<literal>`/`<triple>` value may
    /// already have been parsed into them: the `state_stack` pop that already happened keeps the
    /// element nesting itself in sync with the real XML, but without this reset the discarded
    /// value would leak into the next `<result>` and make its own, otherwise well-formed,
    /// bindings spuriously fail the `self.term.is_some()` check.
    fn clear_binding_scratch(&mut self) {
        self.term = None;
        self.lang = None;
        #[cfg(feature = "rdf-12")]
        {
            self.direction = None;
        }
        self.datatype = None;
    }
}

fn build_literal(
    value: OxString,
    lang: Option<OxString>,
    #[cfg(feature = "rdf-12")] direction: Option<BaseDirection>,
    datatype: Option<NamedNode>,
) -> Result<Literal, QueryResultsSyntaxError> {
    if let Some(lang) = lang {
        #[cfg(feature = "rdf-12")]
        if let Some(direction) = direction {
            if let Some(datatype) = datatype {
                if datatype.as_ref() != rdf::DIR_LANG_STRING {
                    return Err(QueryResultsSyntaxError::msg(format!(
                        "its:dir value '{direction}' provided with the datatype {datatype}"
                    )));
                }
            }
            return Literal::new_directional_language_tagged_literal(
                value,
                lang.clone(),
                direction,
            )
            .map_err(|e| {
                QueryResultsSyntaxError::msg(format!("Invalid xml:lang value '{lang}': {e}"))
            });
        }
        if let Some(datatype) = datatype {
            if datatype.as_ref() != rdf::LANG_STRING {
                return Err(QueryResultsSyntaxError::msg(format!(
                    "xml:lang value '{lang}' provided with the datatype {datatype}"
                )));
            }
        }
        Literal::new_language_tagged_literal(value, lang.clone()).map_err(|e| {
            QueryResultsSyntaxError::msg(format!("Invalid xml:lang value '{lang}': {e}"))
        })
    } else {
        #[cfg(feature = "rdf-12")]
        if direction.is_some() {
            return Err(QueryResultsSyntaxError::msg(
                "its:dir can only be present alongside xml:lang",
            ));
        }
        Ok(if let Some(datatype) = datatype {
            Literal::new_typed_literal(value, datatype)
        } else {
            Literal::new_simple_literal(value)
        })
    }
}

/// Escapes characters to avoid parsing issues (`<`, `>`, `&`...) or normalization (`\r`,
/// whitespace at the beginning or end...).
fn escape_including_bound_whitespaces(value: &str) -> Cow<'_, str> {
    let mut escaped = None;
    let mut previous_index = 0;
    let mut push_escaped = |escape: &'static str, c: char, i: usize| {
        let buf = escaped.get_or_insert_with(|| String::with_capacity(value.len()));
        buf.push_str(&value[previous_index..i]);
        buf.push_str(escape);
        previous_index = i + c.len_utf8();
    };

    for (i, c) in value.char_indices() {
        match c {
            '<' => push_escaped("&lt;", c, i),
            '>' => push_escaped("&gt;", c, i),
            '\'' => push_escaped("&apos;", c, i),
            '&' => push_escaped("&amp;", c, i),
            '"' => push_escaped("&quot;", c, i),
            '\r' => push_escaped("&#13;", c, i),
            '\u{85}' => push_escaped("&#133;", c, i),
            '\u{2028}' => push_escaped("&#8232;", c, i),
            '\t' if i == 0 || i == value.len() - 1 => push_escaped("&#9;", c, i),
            '\n' if i == 0 || i == value.len() - 1 => push_escaped("&#10;", c, i),
            ' ' if i == 0 || i == value.len() - 1 => push_escaped("&#32;", c, i),
            _ => (),
        }
    }
    if let Some(mut escaped) = escaped {
        escaped.push_str(&value[previous_index..]);
        escaped.into()
    } else {
        value.into()
    }
}

fn decode_xml_entity(
    event: &BytesRef<'_>,
    buffer: &mut String,
    xml_version: XmlVersion,
) -> Result<(), Error> {
    if let Some(char_ref) = event.resolve_char_ref()? {
        buffer.push(char_ref);
        return Ok(());
    }
    let reference = event.xml_content(xml_version)?;
    let Some(value) = resolve_xml_entity(&reference) else {
        return Err(EscapeError::UnrecognizedEntity(0..event.len(), reference.into_owned()).into());
    };
    buffer.push_str(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::format::QueryResultsFormat;
    use crate::parser::{
        QueryResultsParser, ReaderQueryResultsParserOutput, SliceQueryResultsParserOutput,
    };

    /// One malformed `<binding>` in a streamed `<result>` must not silently discard every
    /// solution after it: before this fix, the undeclared-variable error path left `self.term`
    /// set, so the *next*, otherwise well-formed `<result>`'s opening check
    /// (`self.term.is_some()`) spuriously rejected it with "There is already a value for the
    /// current binding" instead of ever getting a chance to parse — and the stream never
    /// recovered from there (see the finding this regression test is for).
    fn assert_bad_binding_does_not_poison_later_results(
        xml: &str,
        bad_binding_error_substring: &str,
    ) {
        let ReaderQueryResultsParserOutput::Solutions(mut solutions) =
            QueryResultsParser::from_format(QueryResultsFormat::Xml)
                .for_reader(xml.as_bytes())
                .expect("the head must parse fine")
        else {
            panic!("expected solutions");
        };
        let first = solutions.next().expect("a first item");
        assert!(first.is_ok(), "the first, well-formed solution: {first:?}");
        let second = solutions.next().expect("a second item");
        let message = second
            .expect_err("the malformed binding must fail")
            .to_string();
        assert!(
            message.contains(bad_binding_error_substring),
            "must fail because of the malformed binding itself, not stale scratch state: {message}"
        );
        let third = solutions.next().expect("a third item");
        assert!(
            third.is_ok(),
            "the third, well-formed solution must still parse: {third:?}"
        );
        assert!(solutions.next().is_none(), "exactly three results");

        // The slice-based reader must behave identically.
        let SliceQueryResultsParserOutput::Solutions(mut solutions) =
            QueryResultsParser::from_format(QueryResultsFormat::Xml)
                .for_slice(xml.as_bytes())
                .expect("the head must parse fine")
        else {
            panic!("expected solutions");
        };
        assert!(solutions.next().expect("a first item").is_ok());
        assert!(solutions.next().expect("a second item").is_err());
        assert!(
            solutions.next().expect("a third item").is_ok(),
            "the third, well-formed solution must still parse (slice)"
        );
    }

    #[test]
    fn undeclared_variable_binding_does_not_poison_the_next_result() {
        assert_bad_binding_does_not_poison_later_results(
            r#"<?xml version="1.0"?>
            <sparql><head><variable name="x"/></head><results>
            <result><binding name="x"><uri>http://example.com/1</uri></binding></result>
            <result><binding name="y"><uri>http://example.com/bad</uri></binding></result>
            <result><binding name="x"><uri>http://example.com/2</uri></binding></result>
            </results></sparql>"#,
            "not declared",
        );
    }

    /// The sibling of `undeclared_variable_binding_does_not_poison_the_next_result` above: a
    /// `<binding>` missing its `name` attribute entirely is rejected at the *opening* tag, before
    /// `state_stack` would otherwise gain a frame for it. Without pushing a frame for it anyway,
    /// `state_stack` falls one frame short of the real element nesting for the rest of the
    /// document, so the next `</...>` end tag pops the *wrong* frame — not a panic, but every
    /// element from there on is checked against the wrong expected tag name.
    #[test]
    fn missing_name_attribute_binding_does_not_poison_the_next_result() {
        assert_bad_binding_does_not_poison_later_results(
            r#"<?xml version="1.0"?>
            <sparql><head><variable name="x"/></head><results>
            <result><binding name="x"><uri>http://example.com/1</uri></binding></result>
            <result><binding><uri>http://example.com/bad</uri></binding></result>
            <result><binding name="x"><uri>http://example.com/2</uri></binding></result>
            </results></sparql>"#,
            "No name attribute",
        );
    }

    /// A different failure shape than the two tests above: `state_stack` is never at risk here
    /// (the pop for `</uri>` already happened unconditionally before this arm runs), but before
    /// this fix the failure did not set `result_had_error`, so the enclosing `</result>` — unaware
    /// anything had gone wrong — handed back `new_bindings` as if this (silently unbound) value had
    /// parsed, in addition to the `Err` already reported for the bad IRI itself.
    #[test]
    fn invalid_uri_value_does_not_poison_the_next_result() {
        assert_bad_binding_does_not_poison_later_results(
            r#"<?xml version="1.0"?>
            <sparql><head><variable name="x"/></head><results>
            <result><binding name="x"><uri>http://example.com/1</uri></binding></result>
            <result><binding name="x"><uri>bad uri with spaces</uri></binding></result>
            <result><binding name="x"><uri>http://example.com/2</uri></binding></result>
            </results></sparql>"#,
            "Invalid IRI value",
        );
    }

    /// The same class as [`invalid_uri_value_does_not_poison_the_next_result`], for
    /// [`XmlInnerSolutionsParser::parse_literal_attributes`]'s own error path (an invalid
    /// `datatype` IRI attribute on a `<literal>` tag) rather than [`State::Uri`]'s.
    #[test]
    fn invalid_literal_datatype_does_not_poison_the_next_result() {
        assert_bad_binding_does_not_poison_later_results(
            r#"<?xml version="1.0"?>
            <sparql><head><variable name="x"/></head><results>
            <result><binding name="x"><literal>ok</literal></binding></result>
            <result><binding name="x"><literal datatype="not a uri">bad</literal></binding></result>
            <result><binding name="x"><literal>ok again</literal></binding></result>
            </results></sparql>"#,
            "Invalid datatype IRI",
        );
    }

    /// A second value nested inside one binding: `state_stack` desyncs the same way a missing
    /// `name` attribute does (this rejection also happens on an `Event::Start`, before any frame
    /// is pushed for the second, unwanted value), just one production over.
    #[test]
    fn second_value_in_a_binding_does_not_poison_the_next_result() {
        assert_bad_binding_does_not_poison_later_results(
            r#"<?xml version="1.0"?>
            <sparql><head><variable name="x"/></head><results>
            <result><binding name="x"><uri>http://example.com/1</uri></binding></result>
            <result><binding name="x"><uri>http://example.com/bad1</uri><uri>http://example.com/bad2</uri></binding></result>
            <result><binding name="x"><uri>http://example.com/2</uri></binding></result>
            </results></sparql>"#,
            "already a value",
        );
    }
}

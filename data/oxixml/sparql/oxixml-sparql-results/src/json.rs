//! Implementation of the [SPARQL Query Results JSON Format](https://www.w3.org/TR/sparql11-results-json/).

use crate::error::{QueryResultsParseError, QueryResultsSyntaxError};
use oxixml_json::{JsonEvent, ReaderJsonParser, SliceJsonParser, WriterJsonSerializer};
#[cfg(feature = "async-tokio")]
use oxixml_json::{TokioAsyncReaderJsonParser, TokioAsyncWriterJsonSerializer};
#[cfg(feature = "rdf-12")]
use oxixml_model::BaseDirection;
use oxixml_model::vocab::{rdf, xsd};
use oxixml_model::{BlankNode, Literal, NamedNode, OxString, Term, Variable};
#[cfg(feature = "rdf-12")]
use oxixml_model::{NamedOrBlankNode, Triple};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::mem::take;
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, AsyncWrite};

/// Writes a boolean SPARQL query result (from an `ASK` query) as SPARQL Results JSON.
pub fn write_boolean_json_result<W: Write>(writer: W, value: bool) -> io::Result<W> {
    let mut serializer = WriterJsonSerializer::new(writer);
    for event in inner_write_boolean_json_result(value) {
        serializer.serialize_event(event)?;
    }
    serializer.finish()
}

/// Writes a boolean SPARQL query result (from an `ASK` query) as SPARQL Results JSON to an
/// asynchronous sink.
#[cfg(feature = "async-tokio")]
pub async fn tokio_async_write_boolean_json_result<W: AsyncWrite + Unpin>(
    writer: W,
    value: bool,
) -> io::Result<W> {
    let mut serializer = TokioAsyncWriterJsonSerializer::new(writer);
    for event in inner_write_boolean_json_result(value) {
        serializer.serialize_event(event).await?;
    }
    serializer.finish()
}

fn inner_write_boolean_json_result(value: bool) -> [JsonEvent<'static>; 7] {
    [
        JsonEvent::StartObject,
        JsonEvent::ObjectKey("head".into()),
        JsonEvent::StartObject,
        JsonEvent::EndObject,
        JsonEvent::ObjectKey("boolean".into()),
        JsonEvent::Boolean(value),
        JsonEvent::EndObject,
    ]
}

/// Serializes [`QuerySolution`](crate::QuerySolution)s to a [`Write`] implementation as SPARQL
/// Results JSON.
pub struct WriterJsonSolutionsSerializer<W: Write> {
    inner: InnerJsonSolutionsSerializer,
    serializer: WriterJsonSerializer<W>,
}

impl<W: Write> WriterJsonSolutionsSerializer<W> {
    pub(crate) fn start(writer: W, variables: &[Variable]) -> io::Result<Self> {
        let mut serializer = WriterJsonSerializer::new(writer);
        let mut buffer = Vec::with_capacity(48);
        let inner = InnerJsonSolutionsSerializer::start(&mut buffer, variables);
        Self::do_write(&mut serializer, buffer)?;
        Ok(Self { inner, serializer })
    }

    pub(crate) fn serialize<'a>(
        &mut self,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) -> io::Result<()> {
        let mut buffer = Vec::with_capacity(48);
        self.inner.write(&mut buffer, solution);
        Self::do_write(&mut self.serializer, buffer)
    }

    pub(crate) fn finish(mut self) -> io::Result<W> {
        let mut buffer = Vec::with_capacity(4);
        self.inner.finish(&mut buffer);
        Self::do_write(&mut self.serializer, buffer)?;
        self.serializer.finish()
    }

    fn do_write(
        serializer: &mut WriterJsonSerializer<W>,
        output: Vec<JsonEvent<'_>>,
    ) -> io::Result<()> {
        for event in output {
            serializer.serialize_event(event)?;
        }
        Ok(())
    }
}

struct InnerJsonSolutionsSerializer;

impl InnerJsonSolutionsSerializer {
    fn start<'a>(output: &mut Vec<JsonEvent<'a>>, variables: &'a [Variable]) -> Self {
        output.push(JsonEvent::StartObject);
        output.push(JsonEvent::ObjectKey("head".into()));
        output.push(JsonEvent::StartObject);
        output.push(JsonEvent::ObjectKey("vars".into()));
        output.push(JsonEvent::StartArray);
        for variable in variables {
            output.push(JsonEvent::String(variable.as_str().into()));
        }
        output.push(JsonEvent::EndArray);
        output.push(JsonEvent::EndObject);
        output.push(JsonEvent::ObjectKey("results".into()));
        output.push(JsonEvent::StartObject);
        output.push(JsonEvent::ObjectKey("bindings".into()));
        output.push(JsonEvent::StartArray);
        Self {}
    }

    fn write<'a>(
        &self,
        output: &mut Vec<JsonEvent<'a>>,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) {
        output.push(JsonEvent::StartObject);
        for (variable, value) in solution {
            // The key is pushed speculatively and rolled back if `write_json_term` reports that
            // `value` has no representation in this build configuration: per the SPARQL Query
            // Results JSON Format, an unbound variable is represented by omitting its key from the
            // binding object entirely, so a dangling key with no value would be invalid JSON.
            let key_position = output.len();
            output.push(JsonEvent::ObjectKey(variable.as_str().into()));
            if !write_json_term(output, value) {
                output.truncate(key_position);
            }
        }
        output.push(JsonEvent::EndObject);
    }

    fn finish(self, output: &mut Vec<JsonEvent<'_>>) {
        output.push(JsonEvent::EndArray);
        output.push(JsonEvent::EndObject);
        output.push(JsonEvent::EndObject);
    }
}

/// Writes `term` as a SPARQL Results JSON term object.
///
/// Returns `true` if a value was written, or `false` if `term` has no representation in this
/// build configuration (see the fallback arm below), in which case the caller must not emit a
/// dangling object key.
fn write_json_term<'a>(output: &mut Vec<JsonEvent<'a>>, term: &'a Term) -> bool {
    match term {
        Term::NamedNode(uri) => {
            write_json_named_node(output, uri);
            true
        }
        Term::BlankNode(bnode) => {
            write_json_blank_node(output, bnode);
            true
        }
        Term::Literal(literal) => {
            output.push(JsonEvent::StartObject);
            output.push(JsonEvent::ObjectKey("type".into()));
            output.push(JsonEvent::String("literal".into()));
            output.push(JsonEvent::ObjectKey("value".into()));
            output.push(JsonEvent::String(literal.value().into()));
            if let Some(language) = literal.language() {
                output.push(JsonEvent::ObjectKey("xml:lang".into()));
                output.push(JsonEvent::String(language.into()));
                #[cfg(feature = "rdf-12")]
                if let Some(direction) = literal.direction() {
                    output.push(JsonEvent::ObjectKey("its:dir".into()));
                    output.push(JsonEvent::String(
                        match direction {
                            BaseDirection::Ltr => "ltr",
                            BaseDirection::Rtl => "rtl",
                        }
                        .into(),
                    ));
                }
            } else if literal.datatype() != xsd::STRING {
                output.push(JsonEvent::ObjectKey("datatype".into()));
                output.push(JsonEvent::String(literal.datatype().as_str().into()));
            }
            output.push(JsonEvent::EndObject);
            true
        }
        #[cfg(feature = "rdf-12")]
        Term::Triple(triple) => {
            output.push(JsonEvent::StartObject);
            output.push(JsonEvent::ObjectKey("type".into()));
            output.push(JsonEvent::String("triple".into()));
            output.push(JsonEvent::ObjectKey("value".into()));
            output.push(JsonEvent::StartObject);
            output.push(JsonEvent::ObjectKey("subject".into()));
            match &triple.subject {
                NamedOrBlankNode::NamedNode(uri) => write_json_named_node(output, uri),
                NamedOrBlankNode::BlankNode(bnode) => write_json_blank_node(output, bnode),
            }
            output.push(JsonEvent::ObjectKey("predicate".into()));
            write_json_named_node(output, &triple.predicate);
            output.push(JsonEvent::ObjectKey("object".into()));
            write_json_term(output, &triple.object);
            output.push(JsonEvent::EndObject);
            output.push(JsonEvent::EndObject);
            true
        }
        // `Term::Triple` exists as soon as `oxixml-model` is built with its own `rdf-12` feature,
        // which any other crate of the build may switch on through Cargo feature unification even
        // when this crate's `rdf-12` feature is off. Without this arm the match would then stop
        // being exhaustive and the crate would fail to compile. Per the SPARQL Query Results JSON
        // Format, an unbound variable is represented by omitting its key entirely from the binding
        // object, so report "wrote nothing" to the caller instead of emitting a dangling key.
        #[allow(
            unreachable_patterns,
            reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build without this crate's own rdf-12"
        )]
        _ => false,
    }
}

fn write_json_named_node<'a>(output: &mut Vec<JsonEvent<'a>>, uri: &'a NamedNode) {
    output.push(JsonEvent::StartObject);
    output.push(JsonEvent::ObjectKey("type".into()));
    output.push(JsonEvent::String("uri".into()));
    output.push(JsonEvent::ObjectKey("value".into()));
    output.push(JsonEvent::String(uri.as_str().into()));
    output.push(JsonEvent::EndObject);
}

fn write_json_blank_node<'a>(output: &mut Vec<JsonEvent<'a>>, bnode: &'a BlankNode) {
    output.push(JsonEvent::StartObject);
    output.push(JsonEvent::ObjectKey("type".into()));
    output.push(JsonEvent::String("bnode".into()));
    output.push(JsonEvent::ObjectKey("value".into()));
    output.push(JsonEvent::String(bnode.as_str().into()));
    output.push(JsonEvent::EndObject);
}

/// Serializes [`QuerySolution`](crate::QuerySolution)s to an [`AsyncWrite`] implementation as
/// SPARQL Results JSON.
#[cfg(feature = "async-tokio")]
pub struct TokioAsyncWriterJsonSolutionsSerializer<W: AsyncWrite + Unpin> {
    inner: InnerJsonSolutionsSerializer,
    serializer: TokioAsyncWriterJsonSerializer<W>,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterJsonSolutionsSerializer<W> {
    pub(crate) async fn start(writer: W, variables: &[Variable]) -> io::Result<Self> {
        let mut serializer = TokioAsyncWriterJsonSerializer::new(writer);
        let mut buffer = Vec::with_capacity(48);
        let inner = InnerJsonSolutionsSerializer::start(&mut buffer, variables);
        Self::do_write(&mut serializer, buffer).await?;
        Ok(Self { inner, serializer })
    }

    pub(crate) async fn serialize<'a>(
        &mut self,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) -> io::Result<()> {
        let mut buffer = Vec::with_capacity(48);
        self.inner.write(&mut buffer, solution);
        Self::do_write(&mut self.serializer, buffer).await
    }

    pub(crate) async fn finish(mut self) -> io::Result<W> {
        let mut buffer = Vec::with_capacity(4);
        self.inner.finish(&mut buffer);
        Self::do_write(&mut self.serializer, buffer).await?;
        self.serializer.finish()
    }

    async fn do_write(
        serializer: &mut TokioAsyncWriterJsonSerializer<W>,
        output: Vec<JsonEvent<'_>>,
    ) -> io::Result<()> {
        for event in output {
            serializer.serialize_event(event).await?;
        }
        Ok(())
    }
}

/// The result of reading a SPARQL Results JSON document from a [`Read`] implementation.
#[allow(clippy::large_enum_variant)]
pub enum ReaderJsonQueryResultsParserOutput<R: Read> {
    /// A set of solutions with their declared variables.
    Solutions {
        /// The declared variables, in the order they were declared.
        variables: Vec<Variable>,
        /// The streaming parser producing the solutions.
        solutions: ReaderJsonSolutionsParser<R>,
    },
    /// A boolean, from an `ASK` query.
    Boolean(bool),
}

impl<R: Read> ReaderJsonQueryResultsParserOutput<R> {
    pub(crate) fn read(reader: R) -> Result<Self, QueryResultsParseError> {
        let mut json_parser = ReaderJsonParser::new(reader);
        let mut inner = JsonInnerReader::new();
        loop {
            if let Some(result) = inner.read_event(json_parser.parse_next()?)? {
                return match result {
                    JsonInnerQueryResults::Solutions {
                        variables,
                        solutions,
                    } => Ok(Self::Solutions {
                        variables,
                        solutions: ReaderJsonSolutionsParser {
                            inner: solutions,
                            json_parser,
                        },
                    }),
                    JsonInnerQueryResults::Boolean(value) => Ok(Self::Boolean(value)),
                };
            }
        }
    }
}

/// A streaming parser of a set of solutions read from a [`Read`] implementation.
pub struct ReaderJsonSolutionsParser<R: Read> {
    inner: JsonInnerSolutions,
    json_parser: ReaderJsonParser<R>,
}

impl<R: Read> ReaderJsonSolutionsParser<R> {
    pub(crate) fn parse_next(
        &mut self,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsParseError> {
        match &mut self.inner {
            JsonInnerSolutions::Reader(reader) => {
                if reader.is_poisoned() {
                    return Ok(None);
                }
                loop {
                    let event = self.json_parser.parse_next()?;
                    if event == JsonEvent::Eof {
                        return Ok(None);
                    }
                    if let Some(result) = reader.parse_event(event)? {
                        return Ok(Some(result));
                    }
                }
            }
            JsonInnerSolutions::Iterator(iter) => Ok(iter.next()?),
        }
    }
}

/// The result of reading a SPARQL Results JSON document from an [`AsyncRead`] implementation.
#[cfg(feature = "async-tokio")]
#[allow(clippy::large_enum_variant)]
pub enum TokioAsyncReaderJsonQueryResultsParserOutput<R: AsyncRead + Unpin> {
    /// A set of solutions with their declared variables.
    Solutions {
        /// The declared variables, in the order they were declared.
        variables: Vec<Variable>,
        /// The streaming parser producing the solutions.
        solutions: TokioAsyncReaderJsonSolutionsParser<R>,
    },
    /// A boolean, from an `ASK` query.
    Boolean(bool),
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderJsonQueryResultsParserOutput<R> {
    pub(crate) async fn read(reader: R) -> Result<Self, QueryResultsParseError> {
        let mut json_parser = TokioAsyncReaderJsonParser::new(reader);
        let mut inner = JsonInnerReader::new();
        loop {
            if let Some(result) = inner.read_event(json_parser.parse_next().await?)? {
                return match result {
                    JsonInnerQueryResults::Solutions {
                        variables,
                        solutions,
                    } => Ok(Self::Solutions {
                        variables,
                        solutions: TokioAsyncReaderJsonSolutionsParser {
                            inner: solutions,
                            json_parser,
                        },
                    }),
                    JsonInnerQueryResults::Boolean(value) => Ok(Self::Boolean(value)),
                };
            }
        }
    }
}

/// A streaming parser of a set of solutions read from an [`AsyncRead`] implementation.
#[cfg(feature = "async-tokio")]
pub struct TokioAsyncReaderJsonSolutionsParser<R: AsyncRead + Unpin> {
    inner: JsonInnerSolutions,
    json_parser: TokioAsyncReaderJsonParser<R>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderJsonSolutionsParser<R> {
    pub(crate) async fn parse_next(
        &mut self,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsParseError> {
        match &mut self.inner {
            JsonInnerSolutions::Reader(reader) => {
                if reader.is_poisoned() {
                    return Ok(None);
                }
                loop {
                    let event = self.json_parser.parse_next().await?;
                    if event == JsonEvent::Eof {
                        return Ok(None);
                    }
                    if let Some(result) = reader.parse_event(event)? {
                        return Ok(Some(result));
                    }
                }
            }
            JsonInnerSolutions::Iterator(iter) => Ok(iter.next()?),
        }
    }
}

/// The result of reading a SPARQL Results JSON document from an in-memory byte slice.
#[allow(clippy::large_enum_variant)]
pub enum SliceJsonQueryResultsParserOutput<'a> {
    /// A set of solutions with their declared variables.
    Solutions {
        /// The declared variables, in the order they were declared.
        variables: Vec<Variable>,
        /// The streaming parser producing the solutions.
        solutions: SliceJsonSolutionsParser<'a>,
    },
    /// A boolean, from an `ASK` query.
    Boolean(bool),
}

impl<'a> SliceJsonQueryResultsParserOutput<'a> {
    pub(crate) fn read(slice: &'a [u8]) -> Result<Self, QueryResultsSyntaxError> {
        let mut json_parser = SliceJsonParser::new(slice);
        let mut inner = JsonInnerReader::new();
        loop {
            if let Some(result) = inner.read_event(json_parser.parse_next()?)? {
                return match result {
                    JsonInnerQueryResults::Solutions {
                        variables,
                        solutions,
                    } => Ok(Self::Solutions {
                        variables,
                        solutions: SliceJsonSolutionsParser {
                            inner: solutions,
                            json_parser,
                        },
                    }),
                    JsonInnerQueryResults::Boolean(value) => Ok(Self::Boolean(value)),
                };
            }
        }
    }
}

/// A streaming parser of a set of solutions read from an in-memory byte slice.
pub struct SliceJsonSolutionsParser<'a> {
    inner: JsonInnerSolutions,
    json_parser: SliceJsonParser<'a>,
}

impl SliceJsonSolutionsParser<'_> {
    pub(crate) fn parse_next(
        &mut self,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsSyntaxError> {
        match &mut self.inner {
            JsonInnerSolutions::Reader(reader) => {
                if reader.is_poisoned() {
                    return Ok(None);
                }
                loop {
                    let event = self.json_parser.parse_next()?;
                    if event == JsonEvent::Eof {
                        return Ok(None);
                    }
                    if let Some(result) = reader.parse_event(event)? {
                        return Ok(Some(result));
                    }
                }
            }
            JsonInnerSolutions::Iterator(iter) => iter.next(),
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum JsonInnerQueryResults {
    Solutions {
        variables: Vec<Variable>,
        solutions: JsonInnerSolutions,
    },
    Boolean(bool),
}

#[allow(clippy::large_enum_variant)]
enum JsonInnerSolutions {
    Reader(JsonInnerSolutionsParser),
    Iterator(JsonBufferedSolutionsIterator),
}

struct JsonInnerReader {
    state: JsonInnerReaderState,
    variables: Vec<Variable>,
    current_solution_variables: Vec<OxString>,
    current_solution_values: Vec<Term>,
    solutions: Vec<(Vec<OxString>, Vec<Term>)>,
    vars_read: bool,
    solutions_read: bool,
}

#[allow(clippy::large_enum_variant)]
enum JsonInnerReaderState {
    Start,
    InRootObject,
    BeforeHead,
    InHead,
    BeforeVars,
    InVars,
    BeforeLinks,
    InLinks,
    BeforeResults,
    InResults,
    BeforeBindings,
    BeforeSolution,
    BetweenSolutionTerms,
    Term {
        reader: JsonInnerTermReader,
        variable: OxString,
    },
    AfterBindings,
    BeforeBoolean,
    Ignore {
        level: usize,
        after: JsonInnerReaderStateAfterIgnore,
    },
}

#[derive(Clone, Copy)]
enum JsonInnerReaderStateAfterIgnore {
    InRootObject,
    InHead,
    InResults,
    AfterBindings,
}

impl JsonInnerReader {
    fn new() -> Self {
        Self {
            state: JsonInnerReaderState::Start,
            variables: Vec::new(),
            current_solution_variables: Vec::new(),
            current_solution_values: Vec::new(),
            solutions: Vec::new(),
            vars_read: false,
            solutions_read: false,
        }
    }

    fn read_event(
        &mut self,
        event: JsonEvent<'_>,
    ) -> Result<Option<JsonInnerQueryResults>, QueryResultsSyntaxError> {
        match &mut self.state {
            JsonInnerReaderState::Start => {
                if event == JsonEvent::StartObject {
                    self.state = JsonInnerReaderState::InRootObject;
                    Ok(None)
                } else {
                    Err(QueryResultsSyntaxError::msg(
                        "SPARQL JSON results must be an object",
                    ))
                }
            }
            JsonInnerReaderState::InRootObject => match event {
                JsonEvent::ObjectKey(key) => match key.as_ref() {
                    "head" => {
                        self.state = JsonInnerReaderState::BeforeHead;
                        Ok(None)
                    }
                    "results" => {
                        self.state = JsonInnerReaderState::BeforeResults;
                        Ok(None)
                    }
                    "boolean" => {
                        self.state = JsonInnerReaderState::BeforeBoolean;
                        Ok(None)
                    }
                    _ => {
                        self.state = JsonInnerReaderState::Ignore {
                            level: 0,
                            after: JsonInnerReaderStateAfterIgnore::InRootObject,
                        };
                        Ok(None)
                    }
                },
                JsonEvent::EndObject => Err(QueryResultsSyntaxError::msg(
                    "SPARQL JSON results must contain a 'boolean' or a 'results' key",
                )),
                _ => unreachable!("the JSON grammar guarantees the root object's members"),
            },
            JsonInnerReaderState::BeforeHead => {
                if event == JsonEvent::StartObject {
                    self.state = JsonInnerReaderState::InHead;
                    Ok(None)
                } else {
                    Err(QueryResultsSyntaxError::msg(
                        "SPARQL JSON results head must be an object",
                    ))
                }
            }
            JsonInnerReaderState::InHead => match event {
                JsonEvent::ObjectKey(key) => match key.as_ref() {
                    "vars" => {
                        self.state = JsonInnerReaderState::BeforeVars;
                        self.vars_read = true;
                        Ok(None)
                    }
                    "links" => {
                        self.state = JsonInnerReaderState::BeforeLinks;
                        Ok(None)
                    }
                    _ => {
                        self.state = JsonInnerReaderState::Ignore {
                            level: 0,
                            after: JsonInnerReaderStateAfterIgnore::InHead,
                        };
                        Ok(None)
                    }
                },
                JsonEvent::EndObject => {
                    self.state = JsonInnerReaderState::InRootObject;
                    Ok(None)
                }
                _ => unreachable!("the JSON grammar guarantees the head object's members"),
            },
            JsonInnerReaderState::BeforeVars => {
                if event == JsonEvent::StartArray {
                    self.state = JsonInnerReaderState::InVars;
                    Ok(None)
                } else {
                    Err(QueryResultsSyntaxError::msg(
                        "SPARQL JSON results vars must be an array",
                    ))
                }
            }
            JsonInnerReaderState::InVars => match event {
                JsonEvent::String(variable) => {
                    match Variable::new(OxString::new_owned(&variable)) {
                        Ok(var) => {
                            if self.variables.contains(&var) {
                                return Err(QueryResultsSyntaxError::msg(format!(
                                    "The variable {var} is declared twice"
                                )));
                            }
                            self.variables.push(var);
                            Ok(None)
                        }
                        Err(e) => Err(QueryResultsSyntaxError::msg(format!(
                            "Invalid variable name '{variable}': {e}"
                        ))),
                    }
                }
                JsonEvent::EndArray => {
                    if self.solutions_read {
                        let mut mapping = HashMap::new();
                        for (i, var) in self.variables.iter().enumerate() {
                            mapping.insert(var.clone().into_string(), i);
                        }
                        Ok(Some(JsonInnerQueryResults::Solutions {
                            variables: take(&mut self.variables),
                            solutions: JsonInnerSolutions::Iterator(
                                JsonBufferedSolutionsIterator {
                                    mapping,
                                    bindings: take(&mut self.solutions).into_iter(),
                                    poisoned: false,
                                },
                            ),
                        }))
                    } else {
                        self.state = JsonInnerReaderState::InHead;
                        Ok(None)
                    }
                }
                _ => Err(QueryResultsSyntaxError::msg(
                    "Variables name in the vars array must be strings",
                )),
            },
            JsonInnerReaderState::BeforeLinks => {
                if event == JsonEvent::StartArray {
                    self.state = JsonInnerReaderState::InLinks;
                    Ok(None)
                } else {
                    Err(QueryResultsSyntaxError::msg(
                        "SPARQL JSON results links must be an array",
                    ))
                }
            }
            JsonInnerReaderState::InLinks => match event {
                JsonEvent::String(_) => Ok(None),
                JsonEvent::EndArray => {
                    self.state = JsonInnerReaderState::InHead;
                    Ok(None)
                }
                _ => Err(QueryResultsSyntaxError::msg(
                    "Links in the links array must be strings",
                )),
            },
            JsonInnerReaderState::BeforeResults => {
                if event == JsonEvent::StartObject {
                    self.state = JsonInnerReaderState::InResults;
                    Ok(None)
                } else {
                    Err(QueryResultsSyntaxError::msg(
                        "SPARQL JSON results result must be an object",
                    ))
                }
            }
            JsonInnerReaderState::InResults => match event {
                JsonEvent::ObjectKey(key) => {
                    if key == "bindings" {
                        self.state = JsonInnerReaderState::BeforeBindings;
                        Ok(None)
                    } else {
                        self.state = JsonInnerReaderState::Ignore {
                            level: 0,
                            after: JsonInnerReaderStateAfterIgnore::InResults,
                        };
                        Ok(None)
                    }
                }
                JsonEvent::EndObject => Err(QueryResultsSyntaxError::msg(
                    "The results object must contains a 'bindings' key",
                )),
                _ => unreachable!("the JSON grammar guarantees the results object's members"),
            },
            JsonInnerReaderState::BeforeBindings => {
                if event == JsonEvent::StartArray {
                    self.solutions_read = true;
                    if self.vars_read {
                        let mut mapping = HashMap::new();
                        for (i, var) in self.variables.iter().enumerate() {
                            mapping.insert(var.clone().into_string(), i);
                        }
                        Ok(Some(JsonInnerQueryResults::Solutions {
                            variables: take(&mut self.variables),
                            solutions: JsonInnerSolutions::Reader(JsonInnerSolutionsParser {
                                state: JsonInnerSolutionsParserState::BeforeSolution,
                                mapping,
                                new_bindings: Vec::new(),
                                poisoned: false,
                            }),
                        }))
                    } else {
                        self.state = JsonInnerReaderState::BeforeSolution;
                        Ok(None)
                    }
                } else {
                    Err(QueryResultsSyntaxError::msg(
                        "SPARQL JSON results bindings must be an array",
                    ))
                }
            }
            JsonInnerReaderState::BeforeSolution => match event {
                JsonEvent::StartObject => {
                    self.state = JsonInnerReaderState::BetweenSolutionTerms;
                    Ok(None)
                }
                JsonEvent::EndArray => {
                    self.state = JsonInnerReaderState::AfterBindings;
                    Ok(None)
                }
                _ => Err(QueryResultsSyntaxError::msg(
                    "Expecting a new solution object",
                )),
            },
            JsonInnerReaderState::BetweenSolutionTerms => match event {
                JsonEvent::ObjectKey(key) => {
                    self.state = JsonInnerReaderState::Term {
                        reader: JsonInnerTermReader::default(),
                        variable: OxString::new_owned(&key),
                    };
                    Ok(None)
                }
                JsonEvent::EndObject => {
                    self.state = JsonInnerReaderState::BeforeSolution;
                    self.solutions.push((
                        take(&mut self.current_solution_variables),
                        take(&mut self.current_solution_values),
                    ));
                    Ok(None)
                }
                _ => unreachable!("the JSON grammar guarantees the solution object's members"),
            },
            JsonInnerReaderState::Term { reader, variable } => {
                let result = reader.read_event(event);
                if let Some(term) = result? {
                    self.current_solution_variables.push(take(variable));
                    self.current_solution_values.push(term);
                    self.state = JsonInnerReaderState::BetweenSolutionTerms;
                }
                Ok(None)
            }
            JsonInnerReaderState::AfterBindings => {
                if event == JsonEvent::EndObject {
                    self.state = JsonInnerReaderState::InRootObject;
                } else {
                    self.state = JsonInnerReaderState::Ignore {
                        level: 0,
                        after: JsonInnerReaderStateAfterIgnore::AfterBindings,
                    };
                }
                Ok(None)
            }
            JsonInnerReaderState::BeforeBoolean => {
                if let JsonEvent::Boolean(v) = event {
                    Ok(Some(JsonInnerQueryResults::Boolean(v)))
                } else {
                    Err(QueryResultsSyntaxError::msg("Unexpected boolean value"))
                }
            }
            JsonInnerReaderState::Ignore { level, after } => {
                let new_level = match event {
                    JsonEvent::StartArray | JsonEvent::StartObject => *level + 1,
                    JsonEvent::EndArray | JsonEvent::EndObject => *level - 1,
                    JsonEvent::String(_)
                    | JsonEvent::Number(_)
                    | JsonEvent::Boolean(_)
                    | JsonEvent::Null
                    | JsonEvent::ObjectKey(_)
                    | JsonEvent::Eof => *level,
                };
                self.state = if new_level == 0 {
                    match after {
                        JsonInnerReaderStateAfterIgnore::InRootObject => {
                            JsonInnerReaderState::InRootObject
                        }
                        JsonInnerReaderStateAfterIgnore::InHead => JsonInnerReaderState::InHead,
                        JsonInnerReaderStateAfterIgnore::InResults => {
                            JsonInnerReaderState::InResults
                        }
                        JsonInnerReaderStateAfterIgnore::AfterBindings => {
                            JsonInnerReaderState::AfterBindings
                        }
                    }
                } else {
                    JsonInnerReaderState::Ignore {
                        level: new_level,
                        after: *after,
                    }
                };
                Ok(None)
            }
        }
    }
}

struct JsonInnerSolutionsParser {
    state: JsonInnerSolutionsParserState,
    mapping: HashMap<OxString, usize>,
    new_bindings: Vec<Option<Term>>,
    /// Set once [`Self::parse_event_inner`] has returned an error.
    ///
    /// The JSON grammar arms below assume the tokenizer position matches `self.state`; once an
    /// error is returned, the tokenizer has already moved past the token that caused it (a
    /// variable name whose value has not been read yet, for example), so `self.state` no longer
    /// describes what comes next. Rather than feed the following tokens into that stale state —
    /// which can hit an `unreachable!()` for a perfectly well-formed later solution — every call
    /// after the first error is fused to `Ok(None)`, exactly like a [`FusedIterator`](
    /// std::iter::FusedIterator) that has already returned its terminal value.
    poisoned: bool,
}

#[allow(clippy::large_enum_variant)]
enum JsonInnerSolutionsParserState {
    BeforeSolution,
    BetweenSolutionTerms,
    Term {
        reader: JsonInnerTermReader,
        key: usize,
    },
    AfterEnd,
}

impl JsonInnerSolutionsParser {
    /// Whether a previous call already failed: from then on every call is a fused no-op, since
    /// the underlying JSON tokenizer is left positioned mid-value with no way to resynchronize
    /// `self.state` to it.
    fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    fn parse_event(
        &mut self,
        event: JsonEvent<'_>,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsSyntaxError> {
        if self.poisoned {
            return Ok(None);
        }
        let result = self.parse_event_inner(event);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn parse_event_inner(
        &mut self,
        event: JsonEvent<'_>,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsSyntaxError> {
        match &mut self.state {
            JsonInnerSolutionsParserState::BeforeSolution => match event {
                JsonEvent::StartObject => {
                    self.state = JsonInnerSolutionsParserState::BetweenSolutionTerms;
                    self.new_bindings = vec![None; self.mapping.len()];
                    Ok(None)
                }
                JsonEvent::EndArray => {
                    self.state = JsonInnerSolutionsParserState::AfterEnd;
                    Ok(None)
                }
                _ => Err(QueryResultsSyntaxError::msg(
                    "Expecting a new solution object",
                )),
            },
            JsonInnerSolutionsParserState::BetweenSolutionTerms => match event {
                JsonEvent::ObjectKey(key) => {
                    let key = *self.mapping.get(key.as_ref()).ok_or_else(|| {
                        QueryResultsSyntaxError::msg(format!(
                            "The variable {key} has not been defined in the header"
                        ))
                    })?;
                    self.state = JsonInnerSolutionsParserState::Term {
                        reader: JsonInnerTermReader::default(),
                        key,
                    };
                    Ok(None)
                }
                JsonEvent::EndObject => {
                    self.state = JsonInnerSolutionsParserState::BeforeSolution;
                    Ok(Some(take(&mut self.new_bindings)))
                }
                _ => unreachable!("the JSON grammar guarantees the solution object's members"),
            },
            JsonInnerSolutionsParserState::Term { reader, key } => {
                let result = reader.read_event(event);
                if let Some(term) = result? {
                    self.new_bindings[*key] = Some(term);
                    self.state = JsonInnerSolutionsParserState::BetweenSolutionTerms;
                }
                Ok(None)
            }
            JsonInnerSolutionsParserState::AfterEnd => {
                if event == JsonEvent::EndObject {
                    Ok(None)
                } else {
                    Err(QueryResultsSyntaxError::msg(
                        "Unexpected JSON after the end of the bindings array",
                    ))
                }
            }
        }
    }
}

/// The deepest a chain of RDF-1.2 embedded triple terms (`{"type":"triple","value":{...,
/// "object":<nested triple term>}}`) may nest.
///
/// [`JsonInnerTermReader::read_event`] parses nested triple terms iteratively — see
/// [`JsonInnerTermReader::frames`] — so this bound is not needed to keep *this parser's* call
/// stack shallow. It exists because the [`Term::Triple`] value the parse produces is still walked
/// by natively recursive `Clone`/`Hash`/`PartialEq` implementations and compiler-generated `Drop`
/// glue in `oxixml-model`, which this crate cannot itself guard: capping the depth here is the
/// only lever available to keep those *later* operations within a safe native stack depth.
/// Matches the workspace-wide ceiling for structural/RDF document nesting.
///
/// Not exposed as `pub`: the `json` module itself is private (SPARQL results callers only ever
/// see the format-agnostic [`crate::QueryResultsParser`]), so re-declaring this one constant
/// `pub` would not actually make it reachable from outside the crate.
#[cfg(feature = "rdf-12")]
const MAX_TRIPLE_TERM_DEPTH: usize = 256;

#[derive(Default)]
struct JsonInnerTermReader {
    state: JsonInnerTermReaderState,
    term_type: Option<TermType>,
    value: Option<OxString>,
    lang: Option<OxString>,
    #[cfg(feature = "rdf-12")]
    direction: Option<BaseDirection>,
    datatype: Option<NamedNode>,
    #[cfg(feature = "rdf-12")]
    subject: Option<Term>,
    #[cfg(feature = "rdf-12")]
    predicate: Option<Term>,
    #[cfg(feature = "rdf-12")]
    object: Option<Term>,
    /// One saved frame per RDF-1.2 triple term this reader is currently suspended inside of while
    /// it parses a nested `"subject"`/`"predicate"`/`"object"` value, innermost last.
    ///
    /// This single reader's scratch fields above are reused for every nesting level in turn:
    /// entering a nested value saves the enclosing triple's fields into a [`Frame`] (see
    /// [`JsonInnerTermReader::read_event`]'s `InValue` arm) and resets `state` to
    /// [`JsonInnerTermReaderState::Start`] to parse the nested term from scratch; completing a
    /// term pops its frame, if any, and restores it. `read_event` therefore costs one native call
    /// per JSON token regardless of nesting depth — never one native call per nesting level — so
    /// [`MAX_TRIPLE_TERM_DEPTH`] only has to bound the size of the resulting term tree, not this
    /// parser's own recursion.
    #[cfg(feature = "rdf-12")]
    frames: Vec<Frame>,
}

/// A suspended, enclosing RDF-1.2 triple term's scratch state, saved by
/// [`JsonInnerTermReader::read_event`] while a nested `"subject"`/`"predicate"`/`"object"` value
/// is parsed using the same reader's fields, and restored once that nested value completes.
#[cfg(feature = "rdf-12")]
struct Frame {
    /// Which of the enclosing triple's slots the nested term being parsed will fill in once it
    /// completes.
    slot: TripleSlot,
    term_type: Option<TermType>,
    value: Option<OxString>,
    lang: Option<OxString>,
    direction: Option<BaseDirection>,
    datatype: Option<NamedNode>,
    subject: Option<Term>,
    predicate: Option<Term>,
    object: Option<Term>,
}

/// Which slot of an RDF-1.2 triple term a nested value belongs to.
#[cfg(feature = "rdf-12")]
enum TripleSlot {
    Subject,
    Predicate,
    Object,
}

#[derive(Default)]
enum JsonInnerTermReaderState {
    #[default]
    Start,
    Middle,
    TermType,
    Value,
    Lang,
    #[cfg(feature = "rdf-12")]
    BaseDirection,
    Datatype,
    #[cfg(feature = "rdf-12")]
    InValue,
}

enum TermType {
    Uri,
    BNode,
    Literal,
    #[cfg(feature = "rdf-12")]
    Triple,
}

impl JsonInnerTermReader {
    /// Builds the [`Term`] described by this reader's scratch fields once its enclosing JSON
    /// object has fully closed (an `EndObject` event in [`JsonInnerTermReaderState::Middle`]).
    ///
    /// Used identically for the top-level term of a binding and, when RDF-1.2 triple terms are
    /// nested, for each inner `"subject"`/`"predicate"`/`"object"` value in turn: see
    /// [`JsonInnerTermReader::frames`] for how the same scratch fields are reused, rather than
    /// recursed into, at every nesting level.
    fn finish_term(&mut self) -> Result<Term, QueryResultsSyntaxError> {
        match self.term_type.take() {
            None => Err(QueryResultsSyntaxError::msg(
                "Term serialization must have a 'type' key",
            )),
            Some(TermType::Uri) => Ok(NamedNode::new(self.value.take().ok_or_else(|| {
                QueryResultsSyntaxError::msg("uri serialization must have a 'value' key")
            })?)
            .map_err(|e| QueryResultsSyntaxError::msg(format!("Invalid uri value: {e}")))?
            .into()),
            Some(TermType::BNode) => Ok(BlankNode::new(self.value.take().ok_or_else(|| {
                QueryResultsSyntaxError::msg("bnode serialization must have a 'value' key")
            })?)
            .map_err(|e| QueryResultsSyntaxError::msg(format!("Invalid bnode value: {e}")))?
            .into()),
            Some(TermType::Literal) => {
                let value = self.value.take().ok_or_else(|| {
                    QueryResultsSyntaxError::msg("literal serialization must have a 'value' key")
                })?;
                Ok(if let Some(lang) = self.lang.take() {
                    #[cfg(feature = "rdf-12")]
                    if let Some(direction) = self.direction.take() {
                        if let Some(datatype) = &self.datatype {
                            if datatype.as_ref() != rdf::DIR_LANG_STRING {
                                return Err(QueryResultsSyntaxError::msg(format!(
                                    "xml:lang value '{lang}' and its:dir value '{direction}' provided with the datatype {datatype}"
                                )));
                            }
                        }
                        return Ok(Literal::new_directional_language_tagged_literal(
                            value,
                            lang.clone(),
                            direction,
                        )
                        .map_err(|e| {
                            QueryResultsSyntaxError::msg(format!(
                                "Invalid xml:lang value '{lang}': {e}"
                            ))
                        })?
                        .into());
                    }
                    if let Some(datatype) = &self.datatype {
                        if datatype.as_ref() != rdf::LANG_STRING {
                            return Err(QueryResultsSyntaxError::msg(format!(
                                "xml:lang value '{lang}' provided with the datatype {datatype}"
                            )));
                        }
                    }
                    Literal::new_language_tagged_literal(value, lang.clone()).map_err(|e| {
                        QueryResultsSyntaxError::msg(format!(
                            "Invalid xml:lang value '{lang}': {e}"
                        ))
                    })?
                } else {
                    #[cfg(feature = "rdf-12")]
                    if self.direction.take().is_some() {
                        return Err(QueryResultsSyntaxError::msg(
                            "its:dir can only be present alongside xml:lang",
                        ));
                    }
                    if let Some(datatype) = self.datatype.take() {
                        Literal::new_typed_literal(value, datatype)
                    } else {
                        Literal::new_simple_literal(value)
                    }
                }
                .into())
            }
            #[cfg(feature = "rdf-12")]
            Some(TermType::Triple) => Ok(Triple::new(
                match self.subject.take().ok_or_else(|| {
                    QueryResultsSyntaxError::msg("triple serialization must have a 'subject' key")
                })? {
                    Term::NamedNode(subject) => NamedOrBlankNode::from(subject),
                    Term::BlankNode(subject) => NamedOrBlankNode::from(subject),
                    Term::Triple(_) => {
                        return Err(QueryResultsSyntaxError::msg(
                            "The 'subject' value cannot be a triple term",
                        ));
                    }
                    Term::Literal(_) => {
                        return Err(QueryResultsSyntaxError::msg(
                            "The 'subject' value cannot be a literal",
                        ));
                    }
                },
                if let Term::NamedNode(predicate) = self.predicate.take().ok_or_else(|| {
                    QueryResultsSyntaxError::msg("triple serialization must have a 'predicate' key")
                })? {
                    predicate
                } else {
                    return Err(QueryResultsSyntaxError::msg(
                        "The 'predicate' value must be a uri",
                    ));
                },
                self.object.take().ok_or_else(|| {
                    QueryResultsSyntaxError::msg("triple serialization must have a 'object' key")
                })?,
            )
            .into()),
        }
    }

    fn read_event(
        &mut self,
        event: JsonEvent<'_>,
    ) -> Result<Option<Term>, QueryResultsSyntaxError> {
        match &mut self.state {
            JsonInnerTermReaderState::Start => {
                if event == JsonEvent::StartObject {
                    self.state = JsonInnerTermReaderState::Middle;
                    Ok(None)
                } else {
                    Err(QueryResultsSyntaxError::msg(
                        "RDF terms must be encoded using objects",
                    ))
                }
            }
            JsonInnerTermReaderState::Middle => match event {
                JsonEvent::ObjectKey(object_key) => {
                    self.state = match object_key.as_ref() {
                        "type" => JsonInnerTermReaderState::TermType,
                        "value" => JsonInnerTermReaderState::Value,
                        "datatype" => JsonInnerTermReaderState::Datatype,
                        "xml:lang" => JsonInnerTermReaderState::Lang,
                        #[cfg(feature = "rdf-12")]
                        "its:dir" => JsonInnerTermReaderState::BaseDirection,
                        _ => {
                            return Err(QueryResultsSyntaxError::msg(format!(
                                "Unsupported term key: {object_key}"
                            )));
                        }
                    };
                    Ok(None)
                }
                JsonEvent::EndObject => {
                    let term_result = self.finish_term();
                    #[cfg(feature = "rdf-12")]
                    if let Some(frame) = self.frames.pop() {
                        self.term_type = frame.term_type;
                        self.value = frame.value;
                        self.lang = frame.lang;
                        self.direction = frame.direction;
                        self.datatype = frame.datatype;
                        self.subject = frame.subject;
                        self.predicate = frame.predicate;
                        self.object = frame.object;
                        return match term_result {
                            Ok(term) => {
                                match frame.slot {
                                    TripleSlot::Subject => self.subject = Some(term),
                                    TripleSlot::Predicate => self.predicate = Some(term),
                                    TripleSlot::Object => self.object = Some(term),
                                }
                                self.state = JsonInnerTermReaderState::InValue;
                                Ok(None)
                            }
                            Err(e) => Err(e),
                        };
                    }
                    self.state = JsonInnerTermReaderState::Start;
                    term_result.map(Some)
                }
                _ => unreachable!("the JSON grammar guarantees the term object's members"),
            },
            JsonInnerTermReaderState::TermType => {
                self.state = JsonInnerTermReaderState::Middle;
                if let JsonEvent::String(value) = event {
                    match value.as_ref() {
                        "uri" => {
                            self.term_type = Some(TermType::Uri);
                            Ok(None)
                        }
                        "bnode" => {
                            self.term_type = Some(TermType::BNode);
                            Ok(None)
                        }
                        "literal" | "typed-literal" => {
                            self.term_type = Some(TermType::Literal);
                            Ok(None)
                        }
                        #[cfg(feature = "rdf-12")]
                        "triple" => {
                            self.term_type = Some(TermType::Triple);
                            Ok(None)
                        }
                        _ => Err(QueryResultsSyntaxError::msg(format!(
                            "Unexpected term type: '{value}'"
                        ))),
                    }
                } else {
                    Err(QueryResultsSyntaxError::msg("Term type must be a string"))
                }
            }
            JsonInnerTermReaderState::Value => match event {
                JsonEvent::String(value) => {
                    self.value = Some(OxString::new_owned(&value));
                    self.state = JsonInnerTermReaderState::Middle;
                    Ok(None)
                }
                #[cfg(feature = "rdf-12")]
                JsonEvent::StartObject => {
                    self.state = JsonInnerTermReaderState::InValue;
                    Ok(None)
                }
                _ => {
                    self.state = JsonInnerTermReaderState::Middle;
                    Err(QueryResultsSyntaxError::msg("Term value must be a string"))
                }
            },
            JsonInnerTermReaderState::Lang => {
                self.state = JsonInnerTermReaderState::Middle;
                if let JsonEvent::String(value) = event {
                    self.lang = Some(OxString::new_owned(&value));
                    Ok(None)
                } else {
                    Err(QueryResultsSyntaxError::msg("Term lang must be strings"))
                }
            }
            #[cfg(feature = "rdf-12")]
            JsonInnerTermReaderState::BaseDirection => {
                self.state = JsonInnerTermReaderState::Middle;
                if let JsonEvent::String(value) = event {
                    self.direction = Some(match value.as_ref() {
                        "ltr" => BaseDirection::Ltr,
                        "rtl" => BaseDirection::Rtl,
                        _ => {
                            return Err(QueryResultsSyntaxError::msg(format!(
                                "Invalid its:dir value '{value}', expecting 'ltr' or 'rtl'"
                            )));
                        }
                    });
                    Ok(None)
                } else {
                    Err(QueryResultsSyntaxError::msg(
                        "Term base directions must be strings",
                    ))
                }
            }
            JsonInnerTermReaderState::Datatype => {
                self.state = JsonInnerTermReaderState::Middle;
                if let JsonEvent::String(value) = event {
                    match NamedNode::new(OxString::new_owned(&value)) {
                        Ok(datatype) => {
                            self.datatype = Some(datatype);
                            Ok(None)
                        }
                        Err(e) => Err(QueryResultsSyntaxError::msg(format!(
                            "Invalid datatype: {e}"
                        ))),
                    }
                } else {
                    Err(QueryResultsSyntaxError::msg("Term lang must be strings"))
                }
            }
            #[cfg(feature = "rdf-12")]
            JsonInnerTermReaderState::InValue => match event {
                JsonEvent::ObjectKey(object_key) => {
                    let slot = match object_key.as_ref() {
                        "subject" => TripleSlot::Subject,
                        "predicate" => TripleSlot::Predicate,
                        "object" => TripleSlot::Object,
                        _ => {
                            return Err(QueryResultsSyntaxError::msg(format!(
                                "Unsupported value key: {object_key}"
                            )));
                        }
                    };
                    // Bounds the *result*, not this parser's own call stack: `read_event` never
                    // recurses (see `Self::frames`), so this only has to keep the produced
                    // `Term::Triple` shallow enough for `oxixml-model`'s native Clone/Hash/Drop.
                    if self.frames.len() >= MAX_TRIPLE_TERM_DEPTH {
                        return Err(QueryResultsSyntaxError::msg(format!(
                            "RDF-1.2 triple terms are nested more than {MAX_TRIPLE_TERM_DEPTH} levels deep"
                        )));
                    }
                    self.frames.push(Frame {
                        slot,
                        term_type: self.term_type.take(),
                        value: self.value.take(),
                        lang: self.lang.take(),
                        direction: self.direction.take(),
                        datatype: self.datatype.take(),
                        subject: self.subject.take(),
                        predicate: self.predicate.take(),
                        object: self.object.take(),
                    });
                    // Parse the nested term from scratch, reusing this same reader's fields.
                    self.state = JsonInnerTermReaderState::Start;
                    Ok(None)
                }
                JsonEvent::EndObject => {
                    self.state = JsonInnerTermReaderState::Middle;
                    Ok(None)
                }
                _ => unreachable!("the JSON grammar guarantees the value object's members"),
            },
        }
    }
}

/// An iterator over solutions that were fully buffered because the SPARQL JSON `results`
/// key was read before the `head` key (a legal but unusual key ordering).
struct JsonBufferedSolutionsIterator {
    mapping: HashMap<OxString, usize>,
    bindings: std::vec::IntoIter<(Vec<OxString>, Vec<Term>)>,
    /// Set once [`Self::next`] has returned an error, so this iterator is fused exactly like the
    /// streaming [`JsonInnerSolutionsParser`]: a caller sees the very same "stop after the first
    /// `Err`" contract regardless of which key (`head` or `results`) happened to come first in
    /// the document.
    poisoned: bool,
}

impl JsonBufferedSolutionsIterator {
    fn next(&mut self) -> Result<Option<Vec<Option<Term>>>, QueryResultsSyntaxError> {
        if self.poisoned {
            return Ok(None);
        }
        let result = self.next_inner();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn next_inner(&mut self) -> Result<Option<Vec<Option<Term>>>, QueryResultsSyntaxError> {
        let Some((variables, values)) = self.bindings.next() else {
            return Ok(None);
        };
        let mut new_bindings = vec![None; self.mapping.len()];
        for (variable, value) in variables.into_iter().zip(values) {
            let k = *self.mapping.get(&variable).ok_or_else(|| {
                QueryResultsSyntaxError::msg(format!(
                    "The variable {variable} has not been defined in the header"
                ))
            })?;
            new_bindings[k] = Some(value);
        }
        Ok(Some(new_bindings))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rdf-12")]
    use super::MAX_TRIPLE_TERM_DEPTH;
    use crate::format::QueryResultsFormat;
    use crate::parser::{QueryResultsParser, ReaderQueryResultsParserOutput};

    /// After the streaming JSON solutions reader returns an error, it must behave like a fused
    /// iterator (every further call returns `None`) instead of feeding the next raw token into
    /// state left over from the error, which used to hit `unreachable!()` for the perfectly
    /// well-formed solution after the bad one.
    fn assert_fused_after_error(json: &str) {
        let ReaderQueryResultsParserOutput::Solutions(mut solutions) =
            QueryResultsParser::from_format(QueryResultsFormat::Json)
                .for_reader(json.as_bytes())
                .expect("the head must parse fine")
        else {
            panic!("expected solutions");
        };
        let first = solutions.next().expect("a first item");
        assert!(first.is_ok(), "the first, well-formed solution: {first:?}");
        let second = solutions.next().expect("a second item");
        assert!(second.is_err(), "the malformed solution must fail");
        // Fused: every call from here on returns `None`, never `Some(Ok(_))` (recovering into the
        // third, otherwise well-formed solution) nor a panic.
        assert!(solutions.next().is_none(), "must be fused after the error");
        assert!(solutions.next().is_none(), "must stay fused");

        // The slice-based reader must behave identically.
        let crate::parser::SliceQueryResultsParserOutput::Solutions(mut solutions) =
            QueryResultsParser::from_format(QueryResultsFormat::Json)
                .for_slice(json)
                .expect("the head must parse fine")
        else {
            panic!("expected solutions");
        };
        assert!(solutions.next().expect("a first item").is_ok());
        assert!(solutions.next().expect("a second item").is_err());
        assert!(
            solutions.next().is_none(),
            "must be fused after the error (slice)"
        );
        assert!(solutions.next().is_none(), "must stay fused (slice)");
    }

    #[test]
    fn solutions_iterator_is_fused_after_an_undeclared_variable_error() {
        assert_fused_after_error(
            r#"{"head":{"vars":["x"]},"results":{"bindings":[
                {"x":{"type":"uri","value":"http://example.com/1"}},
                {"y":{"type":"uri","value":"http://example.com/bad"}},
                {"x":{"type":"uri","value":"http://example.com/2"}}
            ]}}"#,
        );
    }

    /// The same fused contract, but for the *buffered* solutions iterator used when `results`
    /// happens to come before `head` in the document (a legal but unusual key ordering, see
    /// [`JsonBufferedSolutionsIterator`]): a caller must see the same "stop after the first
    /// error" behavior regardless of which internal representation happened to answer `.next()`.
    #[test]
    fn buffered_solutions_iterator_is_fused_after_an_undeclared_variable_error() {
        let json = r#"{"results":{"bindings":[
                {"x":{"type":"uri","value":"http://example.com/1"}},
                {"y":{"type":"uri","value":"http://example.com/bad"}},
                {"x":{"type":"uri","value":"http://example.com/2"}}
            ]},"head":{"vars":["x"]}}"#;
        let ReaderQueryResultsParserOutput::Solutions(mut solutions) =
            QueryResultsParser::from_format(QueryResultsFormat::Json)
                .for_reader(json.as_bytes())
                .expect("the head must parse fine")
        else {
            panic!("expected solutions");
        };
        assert!(solutions.next().expect("a first item").is_ok());
        assert!(solutions.next().expect("a second item").is_err());
        assert!(solutions.next().is_none(), "must be fused after the error");
        assert!(solutions.next().is_none(), "must stay fused");
    }

    /// The same fused contract for a malformed *term* (an unsupported key inside the term
    /// object), rather than an undeclared variable: `JsonInnerTermReaderState::Middle` used to
    /// leave its state unmoved on this error, so the very next token (the bad key's own value)
    /// hit a second, different `unreachable!()` one level down.
    #[test]
    fn solutions_iterator_is_fused_after_a_malformed_term_error() {
        assert_fused_after_error(
            r#"{"head":{"vars":["x"]},"results":{"bindings":[
                {"x":{"type":"uri","value":"http://example.com/1"}},
                {"x":{"unsupported-key":"z","type":"uri","value":"http://example.com/bad"}},
                {"x":{"type":"uri","value":"http://example.com/2"}}
            ]}}"#,
        );
    }

    /// Builds a SPARQL JSON results document with a single `?x` binding holding a chain of
    /// `depth` nested RDF-1.2 triple terms (`{"type":"triple","value":{...,"object":<nested>}}`),
    /// bottoming out in a plain `uri` term.
    ///
    /// Each level opens two braces that are not closed until the level itself does (one for the
    /// `{"type":"triple",...}` term object, one for its `"value"` object) — the trailing
    /// `"}".repeat(depth * 2)` below closes exactly those, innermost first.
    #[cfg(feature = "rdf-12")]
    fn nested_triple_json(depth: usize) -> String {
        let mut json = String::from(r#"{"head":{"vars":["x"]},"results":{"bindings":[{"x":"#);
        for _ in 0..depth {
            json.push_str(r#"{"type":"triple","value":{"subject":{"type":"uri","value":"http://e/s"},"predicate":{"type":"uri","value":"http://e/p"},"object":"#);
        }
        json.push_str(r#"{"type":"uri","value":"http://e/o"}"#);
        json.push_str(&"}".repeat(depth * 2));
        json.push_str("}]}}");
        json
    }

    /// A chain of deeply nested RDF-1.2 triple terms must be rejected with a bounded syntax
    /// error rather than crashing the process via unbounded (native-call) recursion.
    #[cfg(feature = "rdf-12")]
    #[test]
    fn deeply_nested_triple_terms_are_rejected_without_overflowing_the_stack() {
        let json = nested_triple_json(5_000);

        let output =
            QueryResultsParser::from_format(QueryResultsFormat::Json).for_reader(json.as_bytes());
        let ReaderQueryResultsParserOutput::Solutions(solutions) =
            output.expect("the head must parse fine")
        else {
            panic!("expected solutions");
        };
        // Draining the (bounded) solutions must never crash; the deeply nested term is expected
        // to be cleanly rejected once the depth ceiling is hit.
        let mut saw_err = false;
        for result in solutions {
            if result.is_err() {
                saw_err = true;
                break;
            }
        }
        assert!(
            saw_err,
            "a chain this deep must be rejected, not silently accepted"
        );
    }

    /// A chain nested one level past [`MAX_TRIPLE_TERM_DEPTH`] must fail with the documented,
    /// bounded syntax error naming the limit — not any other error, and not a crash — pinning the
    /// exact boundary rather than merely "some sufficiently long chain fails".
    #[cfg(feature = "rdf-12")]
    #[test]
    fn triple_term_nesting_one_past_the_limit_is_rejected_with_the_documented_error() {
        let json = nested_triple_json(MAX_TRIPLE_TERM_DEPTH + 1);
        let ReaderQueryResultsParserOutput::Solutions(mut solutions) =
            QueryResultsParser::from_format(QueryResultsFormat::Json)
                .for_reader(json.as_bytes())
                .expect("the head must parse fine")
        else {
            panic!("expected solutions");
        };
        let err = solutions
            .next()
            .expect("one item")
            .expect_err("nesting one past the limit must be rejected");
        let message = err.to_string();
        assert!(
            message.contains(&MAX_TRIPLE_TERM_DEPTH.to_string()),
            "the error must name the configured limit: {message}"
        );
    }

    /// A nesting depth comfortably inside the ceiling must still parse and produce the expected
    /// triple term (proving the cap does not reject reasonable RDF-1.2 documents).
    #[cfg(feature = "rdf-12")]
    #[test]
    fn reasonably_nested_triple_terms_still_parse() {
        let json = nested_triple_json(10);

        let ReaderQueryResultsParserOutput::Solutions(mut solutions) =
            QueryResultsParser::from_format(QueryResultsFormat::Json)
                .for_reader(json.as_bytes())
                .expect("the head must parse fine")
        else {
            panic!("expected solutions");
        };
        let solution = solutions
            .next()
            .expect("one solution")
            .expect("it must parse successfully");
        assert_eq!(solution.len(), 1);
    }

    /// A chain of exactly [`MAX_TRIPLE_TERM_DEPTH`] levels — right at the boundary, not just
    /// comfortably under it — must still parse, and the resulting value must be droppable without
    /// crashing (the depth cap exists precisely to keep `oxixml-model`'s native `Drop`/`Clone`
    /// glue for [`Term::Triple`] within a safe stack depth, so this is the property that matters).
    #[cfg(feature = "rdf-12")]
    #[test]
    fn triple_term_nesting_exactly_at_the_limit_parses_and_drops_cleanly() {
        let json = nested_triple_json(MAX_TRIPLE_TERM_DEPTH);
        let ReaderQueryResultsParserOutput::Solutions(mut solutions) =
            QueryResultsParser::from_format(QueryResultsFormat::Json)
                .for_reader(json.as_bytes())
                .expect("the head must parse fine")
        else {
            panic!("expected solutions");
        };
        let solution = solutions
            .next()
            .expect("one solution")
            .expect("nesting exactly at the limit must still parse");
        assert_eq!(solution.len(), 1);
        drop(solution);
    }
}

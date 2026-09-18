//! Implementation of the
//! [SPARQL 1.1 Query Results CSV and TSV Formats](https://www.w3.org/TR/sparql11-results-csv-tsv/).

use crate::error::{QueryResultsParseError, QueryResultsSyntaxError, TextPosition};
use oxixml_model::vocab::xsd;
#[cfg(feature = "rdf-12")]
use oxixml_model::{BaseDirection, NamedOrBlankNode};
use oxixml_model::{BlankNode, NamedNode, NamedNodeRef, Term, Variable};
use oxixml_support::bytes::memchr;
use std::io::{self, Read, Write};
use std::str::{self, FromStr};
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// The maximum size, in bytes, that the internal TSV line buffer is allowed to grow to before
/// an [`io::ErrorKind::OutOfMemory`] error is returned. Guards against unbounded memory growth
/// on adversarial input that never contains a line feed.
const MAX_BUFFER_SIZE: usize = 4096 * 4096;

/// Writes a boolean SPARQL query result (from an `ASK` query) as SPARQL Results CSV/TSV.
///
/// Both formats share the same (non-standardized) boolean serialization: the bare literal
/// `true` or `false`.
pub fn write_boolean_csv_result<W: Write>(mut writer: W, value: bool) -> io::Result<W> {
    writer.write_all(if value { b"true" } else { b"false" })?;
    Ok(writer)
}

/// Writes a boolean SPARQL query result (from an `ASK` query) as SPARQL Results CSV/TSV to an
/// asynchronous sink.
#[cfg(feature = "async-tokio")]
pub async fn tokio_async_write_boolean_csv_result<W: AsyncWrite + Unpin>(
    mut writer: W,
    value: bool,
) -> io::Result<W> {
    writer
        .write_all(if value { b"true" } else { b"false" })
        .await?;
    Ok(writer)
}

/// Serializes [`QuerySolution`](crate::QuerySolution)s to a [`Write`] implementation as SPARQL
/// Results CSV.
pub struct WriterCsvSolutionsSerializer<W: Write> {
    inner: InnerCsvSolutionsSerializer,
    writer: W,
    buffer: String,
}

impl<W: Write> WriterCsvSolutionsSerializer<W> {
    pub(crate) fn start(mut writer: W, variables: Vec<Variable>) -> io::Result<Self> {
        let mut buffer = String::new();
        let inner = InnerCsvSolutionsSerializer::start(&mut buffer, variables);
        writer.write_all(buffer.as_bytes())?;
        buffer.clear();
        Ok(Self {
            inner,
            writer,
            buffer,
        })
    }

    pub(crate) fn serialize<'a>(
        &mut self,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) -> io::Result<()> {
        self.inner.write(&mut self.buffer, solution);
        self.writer.write_all(self.buffer.as_bytes())?;
        self.buffer.clear();
        Ok(())
    }

    pub(crate) fn finish(self) -> W {
        self.writer
    }
}

struct InnerCsvSolutionsSerializer {
    variables: Vec<Variable>,
}

impl InnerCsvSolutionsSerializer {
    fn start(output: &mut String, variables: Vec<Variable>) -> Self {
        let mut start_vars = true;
        for variable in &variables {
            if start_vars {
                start_vars = false;
            } else {
                output.push(',');
            }
            output.push_str(variable.as_str());
        }
        output.push_str("\r\n");
        Self { variables }
    }

    fn write<'a>(
        &self,
        output: &mut String,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) {
        let mut values = vec![None; self.variables.len()];
        for (variable, value) in solution {
            if let Some(position) = self.variables.iter().position(|v| v == variable) {
                values[position] = Some(value);
            }
        }
        let mut start_binding = true;
        for value in values {
            if start_binding {
                start_binding = false;
            } else {
                output.push(',');
            }
            if let Some(value) = value {
                write_csv_term(output, value);
            }
        }
        output.push_str("\r\n");
    }
}

fn write_csv_term(output: &mut String, term: &Term) {
    match term {
        Term::NamedNode(uri) => write_csv_named_node(output, uri),
        Term::BlankNode(bnode) => write_csv_blank_node(output, bnode),
        Term::Literal(literal) => write_escaped_csv_string(output, literal.value()),
        #[cfg(feature = "rdf-12")]
        Term::Triple(triple) => {
            match &triple.subject {
                NamedOrBlankNode::NamedNode(uri) => write_csv_named_node(output, uri),
                NamedOrBlankNode::BlankNode(bnode) => write_csv_blank_node(output, bnode),
            }
            output.push(' ');
            write_csv_named_node(output, &triple.predicate);
            output.push(' ');
            write_csv_term(output, &triple.object);
        }
        // `Term::Triple` exists as soon as `oxixml-model` is built with its own `rdf-12` feature,
        // which any other crate of the build may switch on through Cargo feature unification even
        // when this crate's `rdf-12` feature is off. Without this arm the match would then stop
        // being exhaustive and the crate would fail to compile. `write_csv_term` is a pure
        // appender with no return value, and the caller (`write`, above) already pushes the field
        // separator unconditionally and represents "nothing here" as an empty cell, so a silent
        // no-op is the correct, spec-consistent fallback.
        #[allow(
            unreachable_patterns,
            reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build without this crate's own rdf-12"
        )]
        _ => {}
    }
}

fn write_csv_named_node(output: &mut String, uri: &NamedNode) {
    output.push_str(uri.as_str());
}

fn write_csv_blank_node(output: &mut String, bnode: &BlankNode) {
    output.push_str("_:");
    output.push_str(bnode.as_str());
}

fn write_escaped_csv_string(output: &mut String, s: &str) {
    if s.bytes().any(|c| matches!(c, b'"' | b',' | b'\n' | b'\r')) {
        output.push('"');
        for c in s.chars() {
            if c == '"' {
                output.push('"');
                output.push('"');
            } else {
                output.push(c);
            }
        }
        output.push('"');
    } else {
        output.push_str(s);
    }
}

/// Serializes [`QuerySolution`](crate::QuerySolution)s to an [`AsyncWrite`] implementation as
/// SPARQL Results CSV.
#[cfg(feature = "async-tokio")]
pub struct TokioAsyncWriterCsvSolutionsSerializer<W: AsyncWrite + Unpin> {
    inner: InnerCsvSolutionsSerializer,
    writer: W,
    buffer: String,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterCsvSolutionsSerializer<W> {
    pub(crate) async fn start(mut writer: W, variables: Vec<Variable>) -> io::Result<Self> {
        let mut buffer = String::new();
        let inner = InnerCsvSolutionsSerializer::start(&mut buffer, variables);
        writer.write_all(buffer.as_bytes()).await?;
        buffer.clear();
        Ok(Self {
            inner,
            writer,
            buffer,
        })
    }

    pub(crate) async fn serialize<'a>(
        &mut self,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) -> io::Result<()> {
        self.inner.write(&mut self.buffer, solution);
        self.writer.write_all(self.buffer.as_bytes()).await?;
        self.buffer.clear();
        Ok(())
    }

    pub(crate) fn finish(self) -> W {
        self.writer
    }
}

/// Serializes [`QuerySolution`](crate::QuerySolution)s to a [`Write`] implementation as SPARQL
/// Results TSV.
pub struct WriterTsvSolutionsSerializer<W: Write> {
    inner: InnerTsvSolutionsSerializer,
    writer: W,
    buffer: String,
}

impl<W: Write> WriterTsvSolutionsSerializer<W> {
    pub(crate) fn start(mut writer: W, variables: Vec<Variable>) -> io::Result<Self> {
        let mut buffer = String::new();
        let inner = InnerTsvSolutionsSerializer::start(&mut buffer, variables);
        writer.write_all(buffer.as_bytes())?;
        buffer.clear();
        Ok(Self {
            inner,
            writer,
            buffer,
        })
    }

    pub(crate) fn serialize<'a>(
        &mut self,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) -> io::Result<()> {
        self.inner.write(&mut self.buffer, solution);
        self.writer.write_all(self.buffer.as_bytes())?;
        self.buffer.clear();
        Ok(())
    }

    pub(crate) fn finish(self) -> W {
        self.writer
    }
}

struct InnerTsvSolutionsSerializer {
    variables: Vec<Variable>,
}

impl InnerTsvSolutionsSerializer {
    fn start(output: &mut String, variables: Vec<Variable>) -> Self {
        let mut start_vars = true;
        for variable in &variables {
            if start_vars {
                start_vars = false;
            } else {
                output.push('\t');
            }
            output.push('?');
            output.push_str(variable.as_str());
        }
        output.push('\n');
        Self { variables }
    }

    fn write<'a>(
        &self,
        output: &mut String,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) {
        let mut values = vec![None; self.variables.len()];
        for (variable, value) in solution {
            if let Some(position) = self.variables.iter().position(|v| v == variable) {
                values[position] = Some(value);
            }
        }
        let mut start_binding = true;
        for value in values {
            if start_binding {
                start_binding = false;
            } else {
                output.push('\t');
            }
            if let Some(value) = value {
                write_tsv_term(output, value);
            }
        }
        output.push('\n');
    }
}

fn write_tsv_term(output: &mut String, term: &Term) {
    match term {
        Term::NamedNode(node) => write_tsv_named_node(output, node),
        Term::BlankNode(node) => write_tsv_blank_node(output, node),
        Term::Literal(literal) => {
            let value = literal.value();
            if let Some(language) = literal.language() {
                write_tsv_quoted_str(output, value);
                output.push('@');
                output.push_str(language);
                #[cfg(feature = "rdf-12")]
                if let Some(direction) = literal.direction() {
                    output.push_str(match direction {
                        BaseDirection::Ltr => "--ltr",
                        BaseDirection::Rtl => "--rtl",
                    });
                }
            } else {
                let datatype = literal.datatype();
                if datatype == xsd::BOOLEAN && is_turtle_boolean(value)
                    || datatype == xsd::INTEGER && is_turtle_integer(value)
                    || datatype == xsd::DECIMAL && is_turtle_decimal(value)
                    || datatype == xsd::DOUBLE && is_turtle_double(value)
                {
                    output.push_str(value);
                } else {
                    write_tsv_quoted_str(output, value);
                    if datatype != xsd::STRING {
                        output.push_str("^^");
                        write_tsv_named_node(output, datatype);
                    }
                }
            }
        }
        #[cfg(feature = "rdf-12")]
        Term::Triple(triple) => {
            output.push_str("<<( ");
            match &triple.subject {
                NamedOrBlankNode::NamedNode(node) => write_tsv_named_node(output, node),
                NamedOrBlankNode::BlankNode(node) => write_tsv_blank_node(output, node),
            }
            output.push(' ');
            write_tsv_named_node(output, &triple.predicate);
            output.push(' ');
            write_tsv_term(output, &triple.object);
            output.push_str(" )>>");
        }
        // `Term::Triple` exists as soon as `oxixml-model` is built with its own `rdf-12` feature,
        // which any other crate of the build may switch on through Cargo feature unification even
        // when this crate's `rdf-12` feature is off. Without this arm the match would then stop
        // being exhaustive and the crate would fail to compile. `write_tsv_term` is a pure
        // appender with no return value, and the caller (`write`, above) already pushes the field
        // separator unconditionally and represents "nothing here" as an empty cell, so a silent
        // no-op is the correct, spec-consistent fallback.
        #[allow(
            unreachable_patterns,
            reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build without this crate's own rdf-12"
        )]
        _ => {}
    }
}

fn write_tsv_named_node<'a>(output: &mut String, node: impl Into<NamedNodeRef<'a>>) {
    output.push('<');
    output.push_str(node.into().as_str());
    output.push('>');
}

fn write_tsv_blank_node(output: &mut String, node: &BlankNode) {
    output.push_str("_:");
    output.push_str(node.as_str());
}

fn write_tsv_quoted_str(output: &mut String, string: &str) {
    output.push('"');
    for c in string.chars() {
        match c {
            '\t' => output.push_str("\\t"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            _ => output.push(c),
        }
    }
    output.push('"');
}

fn is_turtle_boolean(value: &str) -> bool {
    matches!(value, "true" | "false")
}

fn is_turtle_integer(value: &str) -> bool {
    // [19]  INTEGER  ::=  [+-]? [0-9]+
    let mut value = value.as_bytes();
    if let Some(v) = value.strip_prefix(b"+") {
        value = v;
    } else if let Some(v) = value.strip_prefix(b"-") {
        value = v;
    }
    !value.is_empty() && value.iter().all(u8::is_ascii_digit)
}

fn is_turtle_decimal(value: &str) -> bool {
    // [20]  DECIMAL  ::=  [+-]? [0-9]* '.' [0-9]+
    let mut value = value.as_bytes();
    if let Some(v) = value.strip_prefix(b"+") {
        value = v;
    } else if let Some(v) = value.strip_prefix(b"-") {
        value = v;
    }
    while value.first().is_some_and(u8::is_ascii_digit) {
        value = &value[1..];
    }
    let Some(value) = value.strip_prefix(b".") else {
        return false;
    };
    !value.is_empty() && value.iter().all(u8::is_ascii_digit)
}

fn is_turtle_double(value: &str) -> bool {
    // [21]    DOUBLE    ::=  [+-]? ([0-9]+ '.' [0-9]* EXPONENT | '.' [0-9]+ EXPONENT | [0-9]+ EXPONENT)
    // [154s]  EXPONENT  ::=  [eE] [+-]? [0-9]+
    let mut value = value.as_bytes();
    if let Some(v) = value.strip_prefix(b"+") {
        value = v;
    } else if let Some(v) = value.strip_prefix(b"-") {
        value = v;
    }
    let mut with_before = false;
    while value.first().is_some_and(u8::is_ascii_digit) {
        value = &value[1..];
        with_before = true;
    }
    let mut with_after = false;
    if let Some(v) = value.strip_prefix(b".") {
        value = v;
        while value.first().is_some_and(u8::is_ascii_digit) {
            value = &value[1..];
            with_after = true;
        }
    }
    if let Some(v) = value.strip_prefix(b"e") {
        value = v;
    } else if let Some(v) = value.strip_prefix(b"E") {
        value = v;
    } else {
        return false;
    }
    if let Some(v) = value.strip_prefix(b"+") {
        value = v;
    } else if let Some(v) = value.strip_prefix(b"-") {
        value = v;
    }
    (with_before || with_after) && !value.is_empty() && value.iter().all(u8::is_ascii_digit)
}

/// Serializes [`QuerySolution`](crate::QuerySolution)s to an [`AsyncWrite`] implementation as
/// SPARQL Results TSV.
#[cfg(feature = "async-tokio")]
pub struct TokioAsyncWriterTsvSolutionsSerializer<W: AsyncWrite + Unpin> {
    inner: InnerTsvSolutionsSerializer,
    writer: W,
    buffer: String,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterTsvSolutionsSerializer<W> {
    pub(crate) async fn start(mut writer: W, variables: Vec<Variable>) -> io::Result<Self> {
        let mut buffer = String::new();
        let inner = InnerTsvSolutionsSerializer::start(&mut buffer, variables);
        writer.write_all(buffer.as_bytes()).await?;
        buffer.clear();
        Ok(Self {
            inner,
            writer,
            buffer,
        })
    }

    pub(crate) async fn serialize<'a>(
        &mut self,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) -> io::Result<()> {
        self.inner.write(&mut self.buffer, solution);
        self.writer.write_all(self.buffer.as_bytes()).await?;
        self.buffer.clear();
        Ok(())
    }

    pub(crate) fn finish(self) -> W {
        self.writer
    }
}

/// The result of reading a SPARQL Results TSV document from a [`Read`] implementation.
pub enum ReaderTsvQueryResultsParserOutput<R: Read> {
    /// A set of solutions with their declared variables.
    Solutions {
        /// The declared variables, in the order they were declared.
        variables: Vec<Variable>,
        /// The streaming parser producing the solutions.
        solutions: ReaderTsvSolutionsParser<R>,
    },
    /// A boolean, from an `ASK` query.
    Boolean(bool),
}

impl<R: Read> ReaderTsvQueryResultsParserOutput<R> {
    pub(crate) fn read(mut reader: R) -> Result<Self, QueryResultsParseError> {
        let mut line_reader = LineReader::new();
        let mut buffer = Vec::new();
        let line = line_reader.next_line_from_reader(&mut buffer, &mut reader)?;
        Ok(match inner_read_first_line(line_reader, line)? {
            TsvInnerQueryResults::Solutions {
                variables,
                solutions,
            } => Self::Solutions {
                variables,
                solutions: ReaderTsvSolutionsParser {
                    reader,
                    inner: solutions,
                    buffer,
                },
            },
            TsvInnerQueryResults::Boolean(value) => Self::Boolean(value),
        })
    }
}

/// A streaming parser of a set of solutions read from a [`Read`] implementation.
pub struct ReaderTsvSolutionsParser<R: Read> {
    reader: R,
    inner: TsvInnerSolutionsParser,
    buffer: Vec<u8>,
}

impl<R: Read> ReaderTsvSolutionsParser<R> {
    pub(crate) fn parse_next(
        &mut self,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsParseError> {
        let line = self
            .inner
            .line_reader
            .next_line_from_reader(&mut self.buffer, &mut self.reader)?;
        Ok(self.inner.parse_next(line)?)
    }
}

/// The result of reading a SPARQL Results TSV document from an [`AsyncRead`] implementation.
#[cfg(feature = "async-tokio")]
pub enum TokioAsyncReaderTsvQueryResultsParserOutput<R: AsyncRead + Unpin> {
    /// A set of solutions with their declared variables.
    Solutions {
        /// The declared variables, in the order they were declared.
        variables: Vec<Variable>,
        /// The streaming parser producing the solutions.
        solutions: TokioAsyncReaderTsvSolutionsParser<R>,
    },
    /// A boolean, from an `ASK` query.
    Boolean(bool),
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderTsvQueryResultsParserOutput<R> {
    pub(crate) async fn read(mut reader: R) -> Result<Self, QueryResultsParseError> {
        let mut line_reader = LineReader::new();
        let mut buffer = Vec::new();
        let line = line_reader
            .next_line_from_tokio_async_read(&mut buffer, &mut reader)
            .await?;
        Ok(match inner_read_first_line(line_reader, line)? {
            TsvInnerQueryResults::Solutions {
                variables,
                solutions,
            } => Self::Solutions {
                variables,
                solutions: TokioAsyncReaderTsvSolutionsParser {
                    reader,
                    inner: solutions,
                    buffer,
                },
            },
            TsvInnerQueryResults::Boolean(value) => Self::Boolean(value),
        })
    }
}

/// A streaming parser of a set of solutions read from an [`AsyncRead`] implementation.
#[cfg(feature = "async-tokio")]
pub struct TokioAsyncReaderTsvSolutionsParser<R: AsyncRead + Unpin> {
    reader: R,
    inner: TsvInnerSolutionsParser,
    buffer: Vec<u8>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderTsvSolutionsParser<R> {
    pub(crate) async fn parse_next(
        &mut self,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsParseError> {
        let line = self
            .inner
            .line_reader
            .next_line_from_tokio_async_read(&mut self.buffer, &mut self.reader)
            .await?;
        Ok(self.inner.parse_next(line)?)
    }
}

/// The result of reading a SPARQL Results TSV document from an in-memory byte slice.
pub enum SliceTsvQueryResultsParserOutput<'a> {
    /// A set of solutions with their declared variables.
    Solutions {
        /// The declared variables, in the order they were declared.
        variables: Vec<Variable>,
        /// The streaming parser producing the solutions.
        solutions: SliceTsvSolutionsParser<'a>,
    },
    /// A boolean, from an `ASK` query.
    Boolean(bool),
}

impl<'a> SliceTsvQueryResultsParserOutput<'a> {
    pub(crate) fn read(slice: &'a [u8]) -> Result<Self, QueryResultsSyntaxError> {
        let mut reader = LineReader::new();
        let line = reader.next_line_from_slice(slice)?;
        Ok(match inner_read_first_line(reader, line)? {
            TsvInnerQueryResults::Solutions {
                variables,
                solutions,
            } => Self::Solutions {
                variables,
                solutions: SliceTsvSolutionsParser {
                    slice,
                    inner: solutions,
                },
            },
            TsvInnerQueryResults::Boolean(value) => Self::Boolean(value),
        })
    }
}

/// A streaming parser of a set of solutions read from an in-memory byte slice.
pub struct SliceTsvSolutionsParser<'a> {
    slice: &'a [u8],
    inner: TsvInnerSolutionsParser,
}

impl SliceTsvSolutionsParser<'_> {
    pub(crate) fn parse_next(
        &mut self,
    ) -> Result<Option<Vec<Option<Term>>>, QueryResultsSyntaxError> {
        let line = self.inner.line_reader.next_line_from_slice(self.slice)?;
        self.inner.parse_next(line)
    }
}

enum TsvInnerQueryResults {
    Solutions {
        variables: Vec<Variable>,
        solutions: TsvInnerSolutionsParser,
    },
    Boolean(bool),
}

fn inner_read_first_line(
    reader: LineReader,
    line: &str,
) -> Result<TsvInnerQueryResults, QueryResultsSyntaxError> {
    let line = line.trim_matches(|c| matches!(c, ' ' | '\r' | '\n'));
    if line.eq_ignore_ascii_case("true") {
        return Ok(TsvInnerQueryResults::Boolean(true));
    }
    if line.eq_ignore_ascii_case("false") {
        return Ok(TsvInnerQueryResults::Boolean(false));
    }
    let mut variables = Vec::new();
    if !line.is_empty() {
        for v in line.split('\t') {
            let v = v.trim();
            if v.is_empty() {
                return Err(QueryResultsSyntaxError::msg(
                    "Empty column on the first row. The first row should be a list of variables like ?foo or $bar",
                ));
            }
            let variable = Variable::from_str(v).map_err(|e| {
                QueryResultsSyntaxError::msg(format!("Invalid variable declaration '{v}': {e}"))
            })?;
            if variables.contains(&variable) {
                return Err(QueryResultsSyntaxError::msg(format!(
                    "The variable {variable} is declared twice"
                )));
            }
            variables.push(variable);
        }
    }
    let column_len = variables.len();
    Ok(TsvInnerQueryResults::Solutions {
        variables,
        solutions: TsvInnerSolutionsParser {
            line_reader: reader,
            column_len,
        },
    })
}

struct TsvInnerSolutionsParser {
    line_reader: LineReader,
    column_len: usize,
}

impl TsvInnerSolutionsParser {
    fn parse_next(&self, line: &str) -> Result<Option<Vec<Option<Term>>>, QueryResultsSyntaxError> {
        if line.is_empty() {
            return Ok(None); // EOF
        }
        let elements = line
            .split('\t')
            .enumerate()
            .map(|(i, v)| {
                let v = v.trim();
                if v.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(Term::from_str(v).map_err(|e| {
                        let start_position_char = line
                            .split('\t')
                            .take(i)
                            .map(|c| c.chars().count() + 1)
                            .sum::<usize>();
                        let start_position_bytes =
                            line.split('\t').take(i).map(|c| c.len() + 1).sum::<usize>();
                        QueryResultsSyntaxError::term(
                            e,
                            v.into(),
                            TextPosition::new(
                                self.line_reader.line_count.saturating_sub(1),
                                start_position_char as u64,
                                self.line_reader.last_line_start + start_position_bytes as u64,
                            )
                                ..TextPosition::new(
                                    self.line_reader.line_count.saturating_sub(1),
                                    (start_position_char + v.chars().count()) as u64,
                                    self.line_reader.last_line_start
                                        + (start_position_bytes + v.len()) as u64,
                                ),
                        )
                    })?))
                }
            })
            .collect::<Result<Vec<_>, QueryResultsSyntaxError>>()?;
        if elements.len() == self.column_len {
            Ok(Some(elements))
        } else if self.column_len == 0 && elements == [None] {
            Ok(Some(Vec::new())) // Zero columns case
        } else {
            Err(QueryResultsSyntaxError::located_message(
                format!(
                    "This TSV files has {} columns but we found a row on line {} with {} columns: {}",
                    self.column_len,
                    self.line_reader.line_count,
                    elements.len(),
                    line
                ),
                TextPosition::new(
                    self.line_reader.line_count.saturating_sub(1),
                    0,
                    self.line_reader.last_line_start,
                )
                    ..TextPosition::new(
                        self.line_reader.line_count.saturating_sub(1),
                        line.chars().count() as u64,
                        self.line_reader.last_line_end,
                    ),
            ))
        }
    }
}

/// Incrementally reads `\n`-terminated lines, tracking line/byte-offset bookkeeping for
/// diagnostics.
struct LineReader {
    buffer_start: usize,
    buffer_end: usize,
    line_count: u64,
    last_line_start: u64,
    last_line_end: u64,
}

impl LineReader {
    fn new() -> Self {
        Self {
            buffer_start: 0,
            buffer_end: 0,
            line_count: 0,
            last_line_start: 0,
            last_line_end: 0,
        }
    }

    fn next_line_from_reader<'a>(
        &mut self,
        buffer: &'a mut Vec<u8>,
        reader: &mut impl Read,
    ) -> Result<&'a str, QueryResultsParseError> {
        let line_end = loop {
            if let Some(eol) = memchr(b'\n', &buffer[self.buffer_start..self.buffer_end]) {
                break self.buffer_start + eol + 1;
            }
            if self.buffer_start > 0 {
                buffer.copy_within(self.buffer_start..self.buffer_end, 0);
                self.buffer_end -= self.buffer_start;
                self.buffer_start = 0;
            }
            if self.buffer_end + 1024 > buffer.len() {
                if self.buffer_end + 1024 > MAX_BUFFER_SIZE {
                    return Err(io::Error::new(
                        io::ErrorKind::OutOfMemory,
                        format!("Reached the buffer maximal size of {MAX_BUFFER_SIZE}"),
                    )
                    .into());
                }
                buffer.resize(self.buffer_end + 1024, b'\0');
            }
            let read = reader.read(&mut buffer[self.buffer_end..])?;
            if read == 0 {
                break self.buffer_end;
            }
            self.buffer_end += read;
        };
        let result = str::from_utf8(&buffer[self.buffer_start..line_end]).map_err(|e| {
            QueryResultsSyntaxError::msg(format!("Invalid UTF-8 in the TSV file: {e}")).into()
        });
        self.line_count += 1;
        self.last_line_start = self.last_line_end;
        self.last_line_end += (line_end - self.buffer_start) as u64;
        self.buffer_start = line_end;
        result
    }

    /// Reads one line from an asynchronous reader into `buffer`.
    ///
    /// The asynchronous mirror of [`next_line_from_reader`](Self::next_line_from_reader),
    /// applying exactly the same compaction, growth and maximal-size policy.
    #[cfg(feature = "async-tokio")]
    async fn next_line_from_tokio_async_read<'a>(
        &mut self,
        buffer: &'a mut Vec<u8>,
        reader: &mut (impl AsyncRead + Unpin),
    ) -> Result<&'a str, QueryResultsParseError> {
        let line_end = loop {
            if let Some(eol) = memchr(b'\n', &buffer[self.buffer_start..self.buffer_end]) {
                break self.buffer_start + eol + 1;
            }
            if self.buffer_start > 0 {
                buffer.copy_within(self.buffer_start..self.buffer_end, 0);
                self.buffer_end -= self.buffer_start;
                self.buffer_start = 0;
            }
            if self.buffer_end + 1024 > buffer.len() {
                if self.buffer_end + 1024 > MAX_BUFFER_SIZE {
                    return Err(io::Error::new(
                        io::ErrorKind::OutOfMemory,
                        format!("Reached the buffer maximal size of {MAX_BUFFER_SIZE}"),
                    )
                    .into());
                }
                buffer.resize(self.buffer_end + 1024, b'\0');
            }
            let read = reader.read(&mut buffer[self.buffer_end..]).await?;
            if read == 0 {
                break self.buffer_end;
            }
            self.buffer_end += read;
        };
        let result = str::from_utf8(&buffer[self.buffer_start..line_end]).map_err(|e| {
            QueryResultsSyntaxError::msg(format!("Invalid UTF-8 in the TSV file: {e}")).into()
        });
        self.line_count += 1;
        self.last_line_start = self.last_line_end;
        self.last_line_end += (line_end - self.buffer_start) as u64;
        self.buffer_start = line_end;
        result
    }

    fn next_line_from_slice<'a>(
        &mut self,
        slice: &'a [u8],
    ) -> Result<&'a str, QueryResultsSyntaxError> {
        let line_end = memchr(b'\n', &slice[self.buffer_start..])
            .map_or_else(|| slice.len(), |eol| self.buffer_start + eol + 1);
        let result = str::from_utf8(&slice[self.buffer_start..line_end]).map_err(|e| {
            QueryResultsSyntaxError::msg(format!("Invalid UTF-8 in the TSV file: {e}"))
        });
        self.line_count += 1;
        self.last_line_start = self.last_line_end;
        self.last_line_end += (line_end - self.buffer_start) as u64;
        self.buffer_start = line_end;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_model::Literal;
    use std::error::Error;

    fn build_example() -> (Vec<Variable>, Vec<Vec<Option<Term>>>) {
        (
            vec![
                Variable::new_unchecked("x"),
                Variable::new_unchecked("literal"),
            ],
            vec![
                vec![
                    Some(NamedNode::new_unchecked("http://example/x").into()),
                    Some(Literal::new_simple_literal("String").into()),
                ],
                vec![
                    Some(NamedNode::new_unchecked("http://example/x").into()),
                    Some(Literal::new_simple_literal("String-with-dquote\"").into()),
                ],
                vec![
                    Some(BlankNode::new_unchecked("b0").into()),
                    Some(Literal::new_simple_literal("Blank node").into()),
                ],
                vec![
                    None,
                    Some(Literal::new_simple_literal("Missing 'x'").into()),
                ],
                vec![None, None],
                vec![
                    Some(NamedNode::new_unchecked("http://example/x").into()),
                    None,
                ],
                vec![
                    Some(BlankNode::new_unchecked("b1").into()),
                    Some(
                        Literal::new_language_tagged_literal_unchecked("String-with-lang", "en")
                            .into(),
                    ),
                ],
                vec![
                    Some(BlankNode::new_unchecked("b1").into()),
                    Some(Literal::new_typed_literal("123", xsd::INTEGER).into()),
                ],
                vec![
                    None,
                    Some(Literal::new_simple_literal("escape,\t\r\n").into()),
                ],
            ],
        )
    }

    #[test]
    fn test_csv_serialization() {
        let (variables, solutions) = build_example();
        let mut buffer = String::new();
        let serializer = InnerCsvSolutionsSerializer::start(&mut buffer, variables.clone());
        for solution in solutions {
            serializer.write(
                &mut buffer,
                variables
                    .iter()
                    .zip(&solution)
                    .filter_map(|(v, s)| s.as_ref().map(|s| (v, s))),
            );
        }
        let expected = "x,literal\r\nhttp://example/x,String\r\nhttp://example/x,\"String-with-dquote\"\"\"\r\n_:b0,Blank node\r\n,Missing 'x'\r\n,\r\nhttp://example/x,\r\n_:b1,String-with-lang\r\n_:b1,123\r\n,\"escape,\t\r\n\"\r\n".to_owned();
        assert_eq!(buffer, expected);
    }

    #[test]
    fn test_tsv_roundtrip() -> Result<(), Box<dyn Error>> {
        let (variables, solutions) = build_example();

        let mut buffer = String::new();
        let serializer = InnerTsvSolutionsSerializer::start(&mut buffer, variables.clone());
        for solution in &solutions {
            serializer.write(
                &mut buffer,
                variables
                    .iter()
                    .zip(solution)
                    .filter_map(|(v, s)| s.as_ref().map(|s| (v, s))),
            );
        }
        let expected = "?x\t?literal\n<http://example/x>\t\"String\"\n<http://example/x>\t\"String-with-dquote\\\"\"\n_:b0\t\"Blank node\"\n\t\"Missing 'x'\"\n\t\n<http://example/x>\t\n_:b1\t\"String-with-lang\"@en\n_:b1\t123\n\t\"escape,\\t\\r\\n\"\n".to_owned();
        assert_eq!(buffer, expected);

        if let SliceTsvQueryResultsParserOutput::Solutions {
            solutions: mut solutions_iter,
            variables: actual_variables,
        } = SliceTsvQueryResultsParserOutput::read(buffer.as_bytes())?
        {
            assert_eq!(actual_variables.as_slice(), variables.as_slice());
            let mut rows = Vec::new();
            while let Some(row) = solutions_iter.parse_next()? {
                rows.push(row);
            }
            assert_eq!(rows, solutions);
        } else {
            unreachable!("solutions were serialized above")
        }

        Ok(())
    }

    #[test]
    fn test_bad_tsv() {
        let mut bad_tsvs = vec![
            "?",
            "?p",
            "?p?o",
            "?p\n<",
            "?p\n_",
            "?p\n_:",
            "?p\n\"",
            "?p\n<<",
            "?p\n<<(",
            "?p\n1\t2\n",
            "?p\n\n",
        ];
        let a_lot_of_strings = format!("?p\n{}\n", "<".repeat(100_000));
        bad_tsvs.push(&a_lot_of_strings);
        for bad_tsv in bad_tsvs {
            if let Ok(ReaderTsvQueryResultsParserOutput::Solutions { mut solutions, .. }) =
                ReaderTsvQueryResultsParserOutput::read(bad_tsv.as_bytes())
            {
                while let Ok(Some(_)) = solutions.parse_next() {}
            }
        }
    }

    #[test]
    fn test_no_columns_csv_serialization() {
        let mut buffer = String::new();
        let serializer = InnerCsvSolutionsSerializer::start(&mut buffer, Vec::new());
        serializer.write(&mut buffer, []);
        assert_eq!(buffer, "\r\n\r\n");
    }

    #[test]
    fn test_no_columns_tsv_serialization() {
        let mut buffer = String::new();
        let serializer = InnerTsvSolutionsSerializer::start(&mut buffer, Vec::new());
        serializer.write(&mut buffer, []);
        assert_eq!(buffer, "\n\n");
    }

    #[test]
    fn test_no_columns_tsv_parsing() -> io::Result<()> {
        if let ReaderTsvQueryResultsParserOutput::Solutions {
            mut solutions,
            variables,
        } = ReaderTsvQueryResultsParserOutput::read(b"\n\n".as_slice())?
        {
            assert_eq!(variables, Vec::<Variable>::new());
            assert_eq!(solutions.parse_next()?, Some(Vec::new()));
            assert_eq!(solutions.parse_next()?, None);
        } else {
            unreachable!("zero variables still produces a Solutions output")
        }
        Ok(())
    }

    #[test]
    fn test_no_results_csv_serialization() {
        let mut buffer = String::new();
        InnerCsvSolutionsSerializer::start(&mut buffer, vec![Variable::new_unchecked("a")]);
        assert_eq!(buffer, "a\r\n");
    }

    #[test]
    fn test_no_results_tsv_serialization() {
        let mut buffer = String::new();
        InnerTsvSolutionsSerializer::start(&mut buffer, vec![Variable::new_unchecked("a")]);
        assert_eq!(buffer, "?a\n");
    }

    #[test]
    fn test_no_results_tsv_parsing() -> io::Result<()> {
        if let ReaderTsvQueryResultsParserOutput::Solutions {
            mut solutions,
            variables,
        } = ReaderTsvQueryResultsParserOutput::read(b"?a\n".as_slice())?
        {
            assert_eq!(variables, vec![Variable::new_unchecked("a")]);
            assert_eq!(solutions.parse_next()?, None);
        } else {
            unreachable!("a header-only TSV still produces a Solutions output")
        }
        Ok(())
    }
}

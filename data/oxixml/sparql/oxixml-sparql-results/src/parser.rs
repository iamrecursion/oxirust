//! [`QueryResultsParser`]: parsing entry point for the four SPARQL query results formats.

use crate::csv::{
    ReaderTsvQueryResultsParserOutput, ReaderTsvSolutionsParser, SliceTsvQueryResultsParserOutput,
    SliceTsvSolutionsParser,
};
#[cfg(feature = "async-tokio")]
use crate::csv::{TokioAsyncReaderTsvQueryResultsParserOutput, TokioAsyncReaderTsvSolutionsParser};
use crate::error::{QueryResultsParseError, QueryResultsSyntaxError};
use crate::format::QueryResultsFormat;
use crate::json::{
    ReaderJsonQueryResultsParserOutput, ReaderJsonSolutionsParser,
    SliceJsonQueryResultsParserOutput, SliceJsonSolutionsParser,
};
#[cfg(feature = "async-tokio")]
use crate::json::{
    TokioAsyncReaderJsonQueryResultsParserOutput, TokioAsyncReaderJsonSolutionsParser,
};
use crate::solution::QuerySolution;
use crate::xml::{
    ReaderXmlQueryResultsParserOutput, ReaderXmlSolutionsParser, SliceXmlQueryResultsParserOutput,
    SliceXmlSolutionsParser,
};
#[cfg(feature = "async-tokio")]
use crate::xml::{TokioAsyncReaderXmlQueryResultsParserOutput, TokioAsyncReaderXmlSolutionsParser};
use oxixml_model::Variable;
use std::io::Read;
use std::sync::Arc;
#[cfg(feature = "async-tokio")]
use tokio::io::AsyncRead;

/// Parsers for [SPARQL query](https://www.w3.org/TR/sparql11-query/) results serialization
/// formats.
///
/// It currently supports the following formats:
/// * [SPARQL Query Results XML Format](https://www.w3.org/TR/rdf-sparql-XMLres/)
///   ([`QueryResultsFormat::Xml`]).
/// * [SPARQL Query Results JSON Format](https://www.w3.org/TR/sparql11-results-json/)
///   ([`QueryResultsFormat::Json`]).
/// * [SPARQL Query Results TSV Format](https://www.w3.org/TR/sparql11-results-csv-tsv/)
///   ([`QueryResultsFormat::Tsv`]).
///
/// The CSV format is not supported for parsing: it is a *lossy* serialization (it does not
/// distinguish IRIs from strings, nor carry datatypes or language tags) and therefore cannot be
/// turned back into a proper [`QuerySolution`].
///
/// Example in JSON (the API is the same for XML and TSV):
/// ```
/// use oxixml_model::{Literal, Variable};
/// use oxixml_sparql_results::{QueryResultsFormat, QueryResultsParser, ReaderQueryResultsParserOutput};
///
/// let json_parser = QueryResultsParser::from_format(QueryResultsFormat::Json);
/// // boolean
/// if let ReaderQueryResultsParserOutput::Boolean(v) = json_parser
///     .clone()
///     .for_reader(br#"{"boolean":true}"#.as_slice())?
/// {
///     assert!(v);
/// }
/// // solutions
/// if let ReaderQueryResultsParserOutput::Solutions(solutions) = json_parser.for_reader(
///     br#"{"head":{"vars":["foo","bar"]},"results":{"bindings":[{"foo":{"type":"literal","value":"test"}}]}}"#
///         .as_slice(),
/// )? {
///     assert_eq!(solutions.variables(), &[Variable::new("foo")?, Variable::new("bar")?]);
///     for solution in solutions {
///         assert_eq!(
///             solution?.iter().collect::<Vec<_>>(),
///             vec![(&Variable::new("foo")?, &Literal::from("test").into())]
///         );
///     }
/// }
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
#[derive(Clone)]
pub struct QueryResultsParser {
    format: QueryResultsFormat,
}

impl QueryResultsParser {
    /// Builds a parser for the given format.
    #[inline]
    pub const fn from_format(format: QueryResultsFormat) -> Self {
        Self { format }
    }

    /// Reads a result file from a [`Read`] implementation.
    ///
    /// Reads are automatically buffered.
    pub fn for_reader<R: Read>(
        self,
        reader: R,
    ) -> Result<ReaderQueryResultsParserOutput<R>, QueryResultsParseError> {
        Ok(match self.format {
            QueryResultsFormat::Xml => match ReaderXmlQueryResultsParserOutput::read(reader)? {
                ReaderXmlQueryResultsParserOutput::Boolean(r) => {
                    ReaderQueryResultsParserOutput::Boolean(r)
                }
                ReaderXmlQueryResultsParserOutput::Solutions {
                    solutions,
                    variables,
                } => ReaderQueryResultsParserOutput::Solutions(ReaderSolutionsParser {
                    variables: variables.into(),
                    solutions: ReaderSolutionsParserKind::Xml(solutions),
                }),
            },
            QueryResultsFormat::Json => match ReaderJsonQueryResultsParserOutput::read(reader)? {
                ReaderJsonQueryResultsParserOutput::Boolean(r) => {
                    ReaderQueryResultsParserOutput::Boolean(r)
                }
                ReaderJsonQueryResultsParserOutput::Solutions {
                    solutions,
                    variables,
                } => ReaderQueryResultsParserOutput::Solutions(ReaderSolutionsParser {
                    variables: variables.into(),
                    solutions: ReaderSolutionsParserKind::Json(solutions),
                }),
            },
            QueryResultsFormat::Csv => {
                return Err(QueryResultsSyntaxError::msg(
                    "CSV SPARQL results syntax is lossy and can't be parsed to a proper RDF representation",
                )
                .into());
            }
            QueryResultsFormat::Tsv => match ReaderTsvQueryResultsParserOutput::read(reader)? {
                ReaderTsvQueryResultsParserOutput::Boolean(r) => {
                    ReaderQueryResultsParserOutput::Boolean(r)
                }
                ReaderTsvQueryResultsParserOutput::Solutions {
                    solutions,
                    variables,
                } => ReaderQueryResultsParserOutput::Solutions(ReaderSolutionsParser {
                    variables: variables.into(),
                    solutions: ReaderSolutionsParserKind::Tsv(solutions),
                }),
            },
        })
    }

    /// Reads a result file from a [`AsyncRead`] implementation.
    ///
    /// This is the asynchronous counterpart of [`for_reader`](Self::for_reader): it drives the
    /// very same per-format incremental cores, awaiting each chunk instead of blocking on it.
    ///
    /// Reads are automatically buffered.
    ///
    /// # Errors
    ///
    /// Returns an error on malformed input, or if the format is
    /// [`QueryResultsFormat::Csv`] (which is lossy and cannot be parsed back).
    #[cfg(feature = "async-tokio")]
    pub async fn for_tokio_async_reader<R: AsyncRead + Unpin>(
        self,
        reader: R,
    ) -> Result<TokioAsyncReaderQueryResultsParserOutput<R>, QueryResultsParseError> {
        Ok(match self.format {
            QueryResultsFormat::Xml => {
                match TokioAsyncReaderXmlQueryResultsParserOutput::read(reader).await? {
                    TokioAsyncReaderXmlQueryResultsParserOutput::Boolean(r) => {
                        TokioAsyncReaderQueryResultsParserOutput::Boolean(r)
                    }
                    TokioAsyncReaderXmlQueryResultsParserOutput::Solutions {
                        solutions,
                        variables,
                    } => TokioAsyncReaderQueryResultsParserOutput::Solutions(
                        TokioAsyncReaderSolutionsParser {
                            variables: variables.into(),
                            solutions: TokioAsyncReaderSolutionsParserKind::Xml(solutions),
                        },
                    ),
                }
            }
            QueryResultsFormat::Json => {
                match TokioAsyncReaderJsonQueryResultsParserOutput::read(reader).await? {
                    TokioAsyncReaderJsonQueryResultsParserOutput::Boolean(r) => {
                        TokioAsyncReaderQueryResultsParserOutput::Boolean(r)
                    }
                    TokioAsyncReaderJsonQueryResultsParserOutput::Solutions {
                        solutions,
                        variables,
                    } => TokioAsyncReaderQueryResultsParserOutput::Solutions(
                        TokioAsyncReaderSolutionsParser {
                            variables: variables.into(),
                            solutions: TokioAsyncReaderSolutionsParserKind::Json(solutions),
                        },
                    ),
                }
            }
            QueryResultsFormat::Csv => {
                return Err(QueryResultsSyntaxError::msg(
                    "CSV SPARQL results syntax is lossy and can't be parsed to a proper RDF representation",
                )
                .into());
            }
            QueryResultsFormat::Tsv => {
                match TokioAsyncReaderTsvQueryResultsParserOutput::read(reader).await? {
                    TokioAsyncReaderTsvQueryResultsParserOutput::Boolean(r) => {
                        TokioAsyncReaderQueryResultsParserOutput::Boolean(r)
                    }
                    TokioAsyncReaderTsvQueryResultsParserOutput::Solutions {
                        solutions,
                        variables,
                    } => TokioAsyncReaderQueryResultsParserOutput::Solutions(
                        TokioAsyncReaderSolutionsParser {
                            variables: variables.into(),
                            solutions: TokioAsyncReaderSolutionsParserKind::Tsv(solutions),
                        },
                    ),
                }
            }
        })
    }

    /// Reads a result file from an in-memory byte slice.
    pub fn for_slice(
        self,
        slice: &(impl AsRef<[u8]> + ?Sized),
    ) -> Result<SliceQueryResultsParserOutput<'_>, QueryResultsSyntaxError> {
        Ok(match self.format {
            QueryResultsFormat::Xml => {
                match SliceXmlQueryResultsParserOutput::read(slice.as_ref())? {
                    SliceXmlQueryResultsParserOutput::Boolean(r) => {
                        SliceQueryResultsParserOutput::Boolean(r)
                    }
                    SliceXmlQueryResultsParserOutput::Solutions {
                        solutions,
                        variables,
                    } => SliceQueryResultsParserOutput::Solutions(SliceSolutionsParser {
                        variables: variables.into(),
                        solutions: SliceSolutionsParserKind::Xml(solutions),
                    }),
                }
            }
            QueryResultsFormat::Json => {
                match SliceJsonQueryResultsParserOutput::read(slice.as_ref())? {
                    SliceJsonQueryResultsParserOutput::Boolean(r) => {
                        SliceQueryResultsParserOutput::Boolean(r)
                    }
                    SliceJsonQueryResultsParserOutput::Solutions {
                        solutions,
                        variables,
                    } => SliceQueryResultsParserOutput::Solutions(SliceSolutionsParser {
                        variables: variables.into(),
                        solutions: SliceSolutionsParserKind::Json(solutions),
                    }),
                }
            }
            QueryResultsFormat::Csv => {
                return Err(QueryResultsSyntaxError::msg(
                    "CSV SPARQL results syntax is lossy and can't be parsed to a proper RDF representation",
                ));
            }
            QueryResultsFormat::Tsv => {
                match SliceTsvQueryResultsParserOutput::read(slice.as_ref())? {
                    SliceTsvQueryResultsParserOutput::Boolean(r) => {
                        SliceQueryResultsParserOutput::Boolean(r)
                    }
                    SliceTsvQueryResultsParserOutput::Solutions {
                        solutions,
                        variables,
                    } => SliceQueryResultsParserOutput::Solutions(SliceSolutionsParser {
                        variables: variables.into(),
                        solutions: SliceSolutionsParserKind::Tsv(solutions),
                    }),
                }
            }
        })
    }
}

impl From<QueryResultsFormat> for QueryResultsParser {
    #[inline]
    fn from(format: QueryResultsFormat) -> Self {
        Self::from_format(format)
    }
}

/// The result of reading a result file from a [`Read`] implementation with
/// [`QueryResultsParser::for_reader`].
///
/// It is either a read boolean ([`bool`], from an `ASK` query) or a streaming reader of a set of
/// solutions ([`ReaderSolutionsParser`], from a `SELECT` query).
#[allow(clippy::large_enum_variant)]
pub enum ReaderQueryResultsParserOutput<R: Read> {
    /// The streaming solutions of a `SELECT` query, with their declared variables.
    Solutions(ReaderSolutionsParser<R>),
    /// The boolean result of an `ASK` query.
    Boolean(bool),
}

/// A streaming parser of a set of [`QuerySolution`] solutions read from a [`Read`]
/// implementation.
///
/// It implements the [`Iterator`] API to iterate over the solutions.
pub struct ReaderSolutionsParser<R: Read> {
    variables: Arc<[Variable]>,
    solutions: ReaderSolutionsParserKind<R>,
}

enum ReaderSolutionsParserKind<R: Read> {
    Xml(ReaderXmlSolutionsParser<R>),
    Json(ReaderJsonSolutionsParser<R>),
    Tsv(ReaderTsvSolutionsParser<R>),
}

impl<R: Read> ReaderSolutionsParser<R> {
    /// Ordered list of the declared variables at the beginning of the results.
    #[inline]
    #[must_use]
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }
}

impl<R: Read> Iterator for ReaderSolutionsParser<R> {
    type Item = Result<QuerySolution, QueryResultsParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(
            match &mut self.solutions {
                ReaderSolutionsParserKind::Xml(reader) => reader.parse_next(),
                ReaderSolutionsParserKind::Json(reader) => reader.parse_next(),
                ReaderSolutionsParserKind::Tsv(reader) => reader.parse_next(),
            }
            .transpose()?
            .map(|values| (Arc::clone(&self.variables), values).into()),
        )
    }
}

/// The result of reading a result file from a [`AsyncRead`] implementation with
/// [`QueryResultsParser::for_tokio_async_reader`].
///
/// It is either a read boolean ([`bool`], from an `ASK` query) or a streaming reader of a set of
/// solutions ([`TokioAsyncReaderSolutionsParser`], from a `SELECT` query).
#[cfg(feature = "async-tokio")]
#[allow(clippy::large_enum_variant)]
pub enum TokioAsyncReaderQueryResultsParserOutput<R: AsyncRead + Unpin> {
    /// The streaming solutions of a `SELECT` query, with their declared variables.
    Solutions(TokioAsyncReaderSolutionsParser<R>),
    /// The boolean result of an `ASK` query.
    Boolean(bool),
}

/// A streaming parser of a set of [`QuerySolution`] solutions read from a [`AsyncRead`]
/// implementation.
///
/// Call [`next`](Self::next) in a loop to iterate over the solutions.
#[cfg(feature = "async-tokio")]
pub struct TokioAsyncReaderSolutionsParser<R: AsyncRead + Unpin> {
    variables: Arc<[Variable]>,
    solutions: TokioAsyncReaderSolutionsParserKind<R>,
}

#[cfg(feature = "async-tokio")]
enum TokioAsyncReaderSolutionsParserKind<R: AsyncRead + Unpin> {
    Xml(TokioAsyncReaderXmlSolutionsParser<R>),
    Json(TokioAsyncReaderJsonSolutionsParser<R>),
    Tsv(TokioAsyncReaderTsvSolutionsParser<R>),
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderSolutionsParser<R> {
    /// Ordered list of the declared variables at the beginning of the results.
    #[inline]
    #[must_use]
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// Reads the next solution or returns `None` if the file is finished.
    pub async fn next(&mut self) -> Option<Result<QuerySolution, QueryResultsParseError>> {
        Some(
            match &mut self.solutions {
                TokioAsyncReaderSolutionsParserKind::Xml(reader) => reader.parse_next().await,
                TokioAsyncReaderSolutionsParserKind::Json(reader) => reader.parse_next().await,
                TokioAsyncReaderSolutionsParserKind::Tsv(reader) => reader.parse_next().await,
            }
            .transpose()?
            .map(|values| (Arc::clone(&self.variables), values).into()),
        )
    }
}

/// The result of reading a result file from an in-memory byte slice with
/// [`QueryResultsParser::for_slice`].
///
/// It is either a read boolean ([`bool`], from an `ASK` query) or a streaming reader of a set of
/// solutions ([`SliceSolutionsParser`], from a `SELECT` query).
#[allow(clippy::large_enum_variant)]
pub enum SliceQueryResultsParserOutput<'a> {
    /// The streaming solutions of a `SELECT` query, with their declared variables.
    Solutions(SliceSolutionsParser<'a>),
    /// The boolean result of an `ASK` query.
    Boolean(bool),
}

/// A streaming parser of a set of [`QuerySolution`] solutions read from an in-memory byte slice.
///
/// It implements the [`Iterator`] API to iterate over the solutions.
pub struct SliceSolutionsParser<'a> {
    variables: Arc<[Variable]>,
    solutions: SliceSolutionsParserKind<'a>,
}

#[allow(clippy::large_enum_variant)]
enum SliceSolutionsParserKind<'a> {
    Xml(SliceXmlSolutionsParser<'a>),
    Json(SliceJsonSolutionsParser<'a>),
    Tsv(SliceTsvSolutionsParser<'a>),
}

impl SliceSolutionsParser<'_> {
    /// Ordered list of the declared variables at the beginning of the results.
    #[inline]
    #[must_use]
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }
}

impl Iterator for SliceSolutionsParser<'_> {
    type Item = Result<QuerySolution, QueryResultsSyntaxError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(
            match &mut self.solutions {
                SliceSolutionsParserKind::Xml(reader) => reader.parse_next(),
                SliceSolutionsParserKind::Json(reader) => reader.parse_next(),
                SliceSolutionsParserKind::Tsv(reader) => reader.parse_next(),
            }
            .transpose()?
            .map(|values| (Arc::clone(&self.variables), values).into()),
        )
    }
}

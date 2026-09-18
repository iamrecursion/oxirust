//! [`QueryResultsSerializer`]: serialization entry point for the four SPARQL query results
//! formats.

#[cfg(feature = "async-tokio")]
use crate::csv::{
    TokioAsyncWriterCsvSolutionsSerializer, TokioAsyncWriterTsvSolutionsSerializer,
    tokio_async_write_boolean_csv_result,
};
use crate::csv::{
    WriterCsvSolutionsSerializer, WriterTsvSolutionsSerializer, write_boolean_csv_result,
};
use crate::format::QueryResultsFormat;
#[cfg(feature = "async-tokio")]
use crate::json::{TokioAsyncWriterJsonSolutionsSerializer, tokio_async_write_boolean_json_result};
use crate::json::{WriterJsonSolutionsSerializer, write_boolean_json_result};
#[cfg(feature = "async-tokio")]
use crate::xml::{TokioAsyncWriterXmlSolutionsSerializer, tokio_async_write_boolean_xml_result};
use crate::xml::{WriterXmlSolutionsSerializer, write_boolean_xml_result};
use oxixml_model::{Term, Variable};
use std::io::{self, Write};
#[cfg(feature = "async-tokio")]
use tokio::io::AsyncWrite;

/// A serializer for [SPARQL query](https://www.w3.org/TR/sparql11-query/) results serialization
/// formats.
///
/// It currently supports the following formats:
/// * [SPARQL Query Results XML Format](https://www.w3.org/TR/rdf-sparql-XMLres/)
///   ([`QueryResultsFormat::Xml`]).
/// * [SPARQL Query Results JSON Format](https://www.w3.org/TR/sparql11-results-json/)
///   ([`QueryResultsFormat::Json`]).
/// * [SPARQL Query Results CSV Format](https://www.w3.org/TR/sparql11-results-csv-tsv/)
///   ([`QueryResultsFormat::Csv`]).
/// * [SPARQL Query Results TSV Format](https://www.w3.org/TR/sparql11-results-csv-tsv/)
///   ([`QueryResultsFormat::Tsv`]).
///
/// Example in JSON (the API is the same for XML, CSV and TSV):
/// ```
/// use oxixml_model::{Literal, Variable};
/// use oxixml_sparql_results::{QueryResultsFormat, QueryResultsSerializer};
/// use std::iter::once;
///
/// let json_serializer = QueryResultsSerializer::from_format(QueryResultsFormat::Json);
///
/// // boolean
/// let mut buffer = Vec::new();
/// json_serializer.clone().serialize_boolean_to_writer(&mut buffer, true)?;
/// assert_eq!(buffer, br#"{"head":{},"boolean":true}"#);
///
/// // solutions
/// let mut buffer = Vec::new();
/// let mut serializer = json_serializer
///     .serialize_solutions_to_writer(&mut buffer, vec![Variable::new("foo")?, Variable::new("bar")?])?;
/// serializer.serialize(once((&Variable::new("foo")?, &Literal::new_simple_literal("test").into())))?;
/// serializer.finish()?;
/// assert_eq!(
///     buffer,
///     br#"{"head":{"vars":["foo","bar"]},"results":{"bindings":[{"foo":{"type":"literal","value":"test"}}]}}"#
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
#[derive(Clone)]
pub struct QueryResultsSerializer {
    format: QueryResultsFormat,
}

impl QueryResultsSerializer {
    /// Builds a serializer for the given format.
    #[inline]
    pub const fn from_format(format: QueryResultsFormat) -> Self {
        Self { format }
    }

    /// Writes a boolean query result (from an `ASK` query) into the given [`Write`]
    /// implementation.
    pub fn serialize_boolean_to_writer<W: Write>(self, writer: W, value: bool) -> io::Result<W> {
        match self.format {
            QueryResultsFormat::Xml => write_boolean_xml_result(writer, value),
            QueryResultsFormat::Json => write_boolean_json_result(writer, value),
            QueryResultsFormat::Csv | QueryResultsFormat::Tsv => {
                write_boolean_csv_result(writer, value)
            }
        }
    }

    /// Writes a boolean query result (from an `ASK` query) into the given [`AsyncWrite`]
    /// implementation.
    ///
    /// This is the asynchronous counterpart of
    /// [`serialize_boolean_to_writer`](Self::serialize_boolean_to_writer).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying writer fails.
    #[cfg(feature = "async-tokio")]
    pub async fn serialize_boolean_to_tokio_async_write<W: AsyncWrite + Unpin>(
        self,
        writer: W,
        value: bool,
    ) -> io::Result<W> {
        match self.format {
            QueryResultsFormat::Xml => tokio_async_write_boolean_xml_result(writer, value).await,
            QueryResultsFormat::Json => tokio_async_write_boolean_json_result(writer, value).await,
            QueryResultsFormat::Csv | QueryResultsFormat::Tsv => {
                tokio_async_write_boolean_csv_result(writer, value).await
            }
        }
    }

    /// Returns a [`WriterSolutionsSerializer`] allowing writing query solutions into the given
    /// [`Write`] implementation.
    ///
    /// <div class="warning">
    ///
    /// Do not forget to run the [`finish`](WriterSolutionsSerializer::finish()) method to
    /// properly write the last bytes of the file.</div>
    ///
    /// <div class="warning">
    ///
    /// This writer does unbuffered writes. You might want to use
    /// [`BufWriter`](io::BufWriter) to avoid that.</div>
    pub fn serialize_solutions_to_writer<W: Write>(
        self,
        writer: W,
        variables: Vec<Variable>,
    ) -> io::Result<WriterSolutionsSerializer<W>> {
        Ok(WriterSolutionsSerializer {
            formatter: match self.format {
                QueryResultsFormat::Xml => WriterSolutionsSerializerKind::Xml(
                    WriterXmlSolutionsSerializer::start(writer, &variables)?,
                ),
                QueryResultsFormat::Json => WriterSolutionsSerializerKind::Json(
                    WriterJsonSolutionsSerializer::start(writer, &variables)?,
                ),
                QueryResultsFormat::Csv => WriterSolutionsSerializerKind::Csv(
                    WriterCsvSolutionsSerializer::start(writer, variables)?,
                ),
                QueryResultsFormat::Tsv => WriterSolutionsSerializerKind::Tsv(
                    WriterTsvSolutionsSerializer::start(writer, variables)?,
                ),
            },
        })
    }

    /// Returns a [`TokioAsyncWriterSolutionsSerializer`] allowing writing query solutions into
    /// the given [`AsyncWrite`] implementation.
    ///
    /// This is the asynchronous counterpart of
    /// [`serialize_solutions_to_writer`](Self::serialize_solutions_to_writer).
    ///
    /// <div class="warning">
    ///
    /// Do not forget to run the
    /// [`finish`](TokioAsyncWriterSolutionsSerializer::finish()) method to properly write the
    /// last bytes of the file.</div>
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying writer fails.
    #[cfg(feature = "async-tokio")]
    pub async fn serialize_solutions_to_tokio_async_write<W: AsyncWrite + Unpin>(
        self,
        writer: W,
        variables: Vec<Variable>,
    ) -> io::Result<TokioAsyncWriterSolutionsSerializer<W>> {
        Ok(TokioAsyncWriterSolutionsSerializer {
            formatter: match self.format {
                QueryResultsFormat::Xml => TokioAsyncWriterSolutionsSerializerKind::Xml(
                    TokioAsyncWriterXmlSolutionsSerializer::start(writer, &variables).await?,
                ),
                QueryResultsFormat::Json => TokioAsyncWriterSolutionsSerializerKind::Json(
                    TokioAsyncWriterJsonSolutionsSerializer::start(writer, &variables).await?,
                ),
                QueryResultsFormat::Csv => TokioAsyncWriterSolutionsSerializerKind::Csv(
                    TokioAsyncWriterCsvSolutionsSerializer::start(writer, variables).await?,
                ),
                QueryResultsFormat::Tsv => TokioAsyncWriterSolutionsSerializerKind::Tsv(
                    TokioAsyncWriterTsvSolutionsSerializer::start(writer, variables).await?,
                ),
            },
        })
    }
}

/// Allows writing query results into an [`AsyncWrite`] implementation.
///
/// Built using [`QueryResultsSerializer::serialize_solutions_to_tokio_async_write`].
///
/// <div class="warning">
///
/// Do not forget to run the [`finish`](Self::finish()) method to properly write the last bytes
/// of the file.</div>
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncWriterSolutionsSerializer<W: AsyncWrite + Unpin> {
    formatter: TokioAsyncWriterSolutionsSerializerKind<W>,
}

#[cfg(feature = "async-tokio")]
enum TokioAsyncWriterSolutionsSerializerKind<W: AsyncWrite + Unpin> {
    Xml(TokioAsyncWriterXmlSolutionsSerializer<W>),
    Json(TokioAsyncWriterJsonSolutionsSerializer<W>),
    Csv(TokioAsyncWriterCsvSolutionsSerializer<W>),
    Tsv(TokioAsyncWriterTsvSolutionsSerializer<W>),
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterSolutionsSerializer<W> {
    /// Writes a solution.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying writer fails.
    pub async fn serialize<'a>(
        &mut self,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) -> io::Result<()> {
        match &mut self.formatter {
            TokioAsyncWriterSolutionsSerializerKind::Xml(writer) => {
                writer.serialize(solution).await
            }
            TokioAsyncWriterSolutionsSerializerKind::Json(writer) => {
                writer.serialize(solution).await
            }
            TokioAsyncWriterSolutionsSerializerKind::Csv(writer) => {
                writer.serialize(solution).await
            }
            TokioAsyncWriterSolutionsSerializerKind::Tsv(writer) => {
                writer.serialize(solution).await
            }
        }
    }

    /// Writes the last bytes of the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying writer fails.
    pub async fn finish(self) -> io::Result<W> {
        match self.formatter {
            TokioAsyncWriterSolutionsSerializerKind::Xml(serializer) => serializer.finish().await,
            TokioAsyncWriterSolutionsSerializerKind::Json(serializer) => serializer.finish().await,
            TokioAsyncWriterSolutionsSerializerKind::Csv(serializer) => Ok(serializer.finish()),
            TokioAsyncWriterSolutionsSerializerKind::Tsv(serializer) => Ok(serializer.finish()),
        }
    }
}

impl From<QueryResultsFormat> for QueryResultsSerializer {
    #[inline]
    fn from(format: QueryResultsFormat) -> Self {
        Self::from_format(format)
    }
}

/// Allows writing query results into a [`Write`] implementation.
///
/// Built using [`QueryResultsSerializer::serialize_solutions_to_writer`].
///
/// <div class="warning">
///
/// Do not forget to run the [`finish`](Self::finish()) method to properly write the last bytes
/// of the file.</div>
///
/// <div class="warning">
///
/// This writer does unbuffered writes. You might want to use [`BufWriter`](io::BufWriter) to
/// avoid that.</div>
#[must_use]
pub struct WriterSolutionsSerializer<W: Write> {
    formatter: WriterSolutionsSerializerKind<W>,
}

enum WriterSolutionsSerializerKind<W: Write> {
    Xml(WriterXmlSolutionsSerializer<W>),
    Json(WriterJsonSolutionsSerializer<W>),
    Csv(WriterCsvSolutionsSerializer<W>),
    Tsv(WriterTsvSolutionsSerializer<W>),
}

impl<W: Write> WriterSolutionsSerializer<W> {
    /// Writes a solution.
    pub fn serialize<'a>(
        &mut self,
        solution: impl IntoIterator<Item = (&'a Variable, &'a Term)>,
    ) -> io::Result<()> {
        match &mut self.formatter {
            WriterSolutionsSerializerKind::Xml(writer) => writer.serialize(solution),
            WriterSolutionsSerializerKind::Json(writer) => writer.serialize(solution),
            WriterSolutionsSerializerKind::Csv(writer) => writer.serialize(solution),
            WriterSolutionsSerializerKind::Tsv(writer) => writer.serialize(solution),
        }
    }

    /// Writes the last bytes of the file.
    pub fn finish(self) -> io::Result<W> {
        match self.formatter {
            WriterSolutionsSerializerKind::Xml(serializer) => serializer.finish(),
            WriterSolutionsSerializerKind::Json(serializer) => serializer.finish(),
            WriterSolutionsSerializerKind::Csv(serializer) => Ok(serializer.finish()),
            WriterSolutionsSerializerKind::Tsv(serializer) => Ok(serializer.finish()),
        }
    }
}

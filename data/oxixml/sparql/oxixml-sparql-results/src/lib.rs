//! `oxixml-sparql-results` — SPARQL 1.1 Query Results I/O (XML/JSON/CSV/TSV; replaces
//! `sparesults`).
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate mirrors `sparesults`' public
//! API (module names, type names and method signatures) so that code written against it ports
//! over by changing only the `use` path, using [`oxixml-model`](oxixml_model) terms rather than
//! `oxrdf` ones.
//!
//! # Formats
//!
//! [`QueryResultsFormat`] enumerates the four SPARQL 1.1 query results serializations:
//!
//! * [SPARQL Query Results XML Format](https://www.w3.org/TR/rdf-sparql-XMLres/)
//!   ([`QueryResultsFormat::Xml`]), built on [`oxixml-xml`](oxixml_xml).
//! * [SPARQL Query Results JSON Format](https://www.w3.org/TR/sparql11-results-json/)
//!   ([`QueryResultsFormat::Json`]), built on [`oxixml-json`](oxixml_json).
//! * [SPARQL Query Results CSV Format](https://www.w3.org/TR/sparql11-results-csv-tsv/)
//!   ([`QueryResultsFormat::Csv`]) — write-only, since the format is lossy (it cannot
//!   distinguish IRIs from plain strings, nor carry datatypes or language tags).
//! * [SPARQL Query Results TSV Format](https://www.w3.org/TR/sparql11-results-csv-tsv/)
//!   ([`QueryResultsFormat::Tsv`]) — a Turtle-ish, lossless term syntax.
//!
//! # Parsing
//!
//! [`QueryResultsParser`] reads either a boolean (from an `ASK` query) or a streaming set of
//! [`QuerySolution`]s (from a `SELECT` query) from a [`std::io::Read`] implementation
//! ([`QueryResultsParser::for_reader`]) or an in-memory byte slice
//! ([`QueryResultsParser::for_slice`]).
//!
//! # Serialization
//!
//! [`QueryResultsSerializer`] writes a boolean
//! ([`QueryResultsSerializer::serialize_boolean_to_writer`]) or a streaming set of solutions
//! ([`QueryResultsSerializer::serialize_solutions_to_writer`]) to a [`std::io::Write`]
//! implementation.
//!
//! # RDF 1.2 / RDF-star
//!
//! The `rdf-12` feature (off by default) enables embedded triple terms and directional
//! language-tagged strings across all four formats, mirroring `sparesults`' `sparql-12` feature
//! and enabling [`oxixml-model`](oxixml_model)'s own `rdf-12` feature.
//!
//! # Asynchronous I/O
//!
//! The `async-tokio` feature (off by default) adds a `tokio`-based mirror of the whole API,
//! matching `sparesults`' feature of the same name:
#![cfg_attr(
    feature = "async-tokio",
    doc = "[`QueryResultsParser::for_tokio_async_reader`], \
[`QueryResultsSerializer::serialize_boolean_to_tokio_async_write`] and \
[`QueryResultsSerializer::serialize_solutions_to_tokio_async_write`], yielding \
[`TokioAsyncReaderQueryResultsParserOutput`], [`TokioAsyncReaderSolutionsParser`] and \
[`TokioAsyncWriterSolutionsSerializer`]."
)]
#![cfg_attr(
    not(feature = "async-tokio"),
    doc = "`QueryResultsParser::for_tokio_async_reader`, \
`QueryResultsSerializer::serialize_boolean_to_tokio_async_write` and \
`QueryResultsSerializer::serialize_solutions_to_tokio_async_write`, yielding \
`TokioAsyncReaderQueryResultsParserOutput`, `TokioAsyncReaderSolutionsParser` and \
`TokioAsyncWriterSolutionsSerializer`."
)]
//! Every one of them drives the very same per-format
//! incremental core as its blocking counterpart — only the I/O is awaited — so the bytes written
//! and the solutions parsed are identical. Keeping it off by default keeps the default build
//! third-party-free.
//!
//! # Example
//!
//! ```
//! use oxixml_model::{Literal, Variable};
//! use oxixml_sparql_results::{
//!     QueryResultsFormat, QueryResultsParser, QueryResultsSerializer, ReaderQueryResultsParserOutput,
//! };
//! use std::iter::once;
//!
//! // Serialize one solution as SPARQL Results JSON.
//! let serializer = QueryResultsSerializer::from_format(QueryResultsFormat::Json);
//! let mut buffer = Vec::new();
//! let mut writer = serializer.serialize_solutions_to_writer(&mut buffer, vec![Variable::new("x")?])?;
//! writer.serialize(once((&Variable::new("x")?, &Literal::from(1).into())))?;
//! writer.finish()?;
//!
//! // ... and read it back.
//! let parser = QueryResultsParser::from_format(QueryResultsFormat::Json);
//! if let ReaderQueryResultsParserOutput::Solutions(solutions) = parser.for_reader(buffer.as_slice())? {
//!     for solution in solutions {
//!         assert_eq!(solution?.get("x"), Some(&Literal::from(1).into()));
//!     }
//! }
//! # Result::<_, Box<dyn std::error::Error>>::Ok(())
//! ```

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod csv;
mod error;
mod format;
mod json;
mod parser;
mod serializer;
mod solution;
mod xml;

pub use crate::error::{QueryResultsParseError, QueryResultsSyntaxError, TextPosition};
pub use crate::format::QueryResultsFormat;
pub use crate::parser::{
    QueryResultsParser, ReaderQueryResultsParserOutput, ReaderSolutionsParser,
    SliceQueryResultsParserOutput, SliceSolutionsParser,
};
#[cfg(feature = "async-tokio")]
pub use crate::parser::{
    TokioAsyncReaderQueryResultsParserOutput, TokioAsyncReaderSolutionsParser,
};
#[cfg(feature = "async-tokio")]
pub use crate::serializer::TokioAsyncWriterSolutionsSerializer;
pub use crate::serializer::{QueryResultsSerializer, WriterSolutionsSerializer};
pub use crate::solution::{QuerySolution, VariableSolutionIndex};

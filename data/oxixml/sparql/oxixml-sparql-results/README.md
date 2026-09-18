# oxixml-sparql-results

SPARQL 1.1 Query Results I/O — XML, JSON, CSV and TSV — a pure-Rust replacement for
[`sparesults`](https://crates.io/crates/sparesults), for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace.

This crate mirrors `sparesults`' public API (module names, type names, method signatures) so
code written against it ports over by changing only the `use` path, using `oxixml-model` terms
rather than `oxrdf` ones. `QueryResultsFormat` enumerates the four SPARQL 1.1 query results
serializations: [XML](https://www.w3.org/TR/rdf-sparql-XMLres/) and
[JSON](https://www.w3.org/TR/sparql11-results-json/) (read/write, built on `oxixml-xml` /
`oxixml-json`), and [CSV/TSV](https://www.w3.org/TR/sparql11-results-csv-tsv/) (CSV is
write-only — the format is lossy and cannot distinguish IRIs from plain strings, nor carry
datatypes or language tags; TSV is a lossless, Turtle-ish term syntax). `QueryResultsParser`
reads a boolean (`ASK`) or a streaming set of `QuerySolution`s (`SELECT`) from a `Read` or a byte
slice; `QueryResultsSerializer` writes the same back out.

## Cargo features

* `rdf-12` (off by default) — embedded triple terms and directional language-tagged strings
  across all four formats, mirroring `sparesults`' `sparql-12` feature.
* `async-tokio` (off by default) — a `tokio`-based mirror of the whole reader/writer API
  (`QueryResultsParser::for_tokio_async_reader`,
  `QueryResultsSerializer::serialize_boolean_to_tokio_async_write` and
  `QueryResultsSerializer::serialize_solutions_to_tokio_async_write`), mirroring `sparesults`'
  `async-tokio` feature. Every asynchronous entry point drives the same per-format incremental
  core as its blocking counterpart, so the bytes written and solutions parsed are identical; off
  by default keeps the crate third-party-free in its default configuration.

## Install

```toml
[dependencies]
oxixml-sparql-results = "0.1.2"
```

## Usage

```rust
use oxixml_model::{Literal, Variable};
use oxixml_sparql_results::{
    QueryResultsFormat, QueryResultsParser, QueryResultsSerializer, ReaderQueryResultsParserOutput,
};
use std::iter::once;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Serialize one solution as SPARQL Results JSON.
    let serializer = QueryResultsSerializer::from_format(QueryResultsFormat::Json);
    let mut buffer = Vec::new();
    let mut writer =
        serializer.serialize_solutions_to_writer(&mut buffer, vec![Variable::new("x")?])?;
    writer.serialize(once((&Variable::new("x")?, &Literal::from(1).into())))?;
    writer.finish()?;

    // ... and read it back.
    let parser = QueryResultsParser::from_format(QueryResultsFormat::Json);
    if let ReaderQueryResultsParserOutput::Solutions(solutions) = parser.for_reader(buffer.as_slice())? {
        for solution in solutions {
            assert_eq!(solution?.get("x"), Some(&Literal::from(1).into()));
        }
    }
    Ok(())
}
```

## Conformance

`tests/results_suite.rs` runs every `.srx` and `.srj` fixture in the W3C `rdf-tests/sparql`
corpus (via `OXIXML_TESTSUITE_DIR`) through this crate: every fixture the parser accepts must
re-serialize and re-parse to the exact same solutions or boolean. The suite requires a 100%
round-trip rate over whatever it attempts, and currently passes both the XML and JSON corpora.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

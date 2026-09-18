# oxixml-sparql-syntax

SPARQL 1.1 query and update syntax to algebra — a pure-Rust, dependency-light replacement for
[`spargebra`](https://crates.io/crates/spargebra) — for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace.

A hand-written recursive-descent parser reads the full
[SPARQL 1.1 Query](https://www.w3.org/TR/sparql11-query/) and
[SPARQL 1.1 Update](https://www.w3.org/TR/sparql11-update/) grammars and translates them into a
SPARQL algebra (`crate::algebra`) that round-trips back to SPARQL text and to the
[SSE](https://jena.apache.org/documentation/notes/sse.html) s-expression form. RDF terms are
reused from `oxixml-model`, exactly like `spargebra` reuses `oxrdf`.

## Cargo features

* `sparql-12` (off by default) — [SPARQL 1.2](https://www.w3.org/TR/sparql12-query/) / SPARQL-star:
  triple terms, reified triples, reifiers and annotation blocks, the `VERSION` declaration,
  directional language-tagged strings, and the `TRIPLE`/`SUBJECT`/`LANGDIR`/`hasLANG`/… built-ins.

## Install

```toml
[dependencies]
oxixml-sparql-syntax = "0.1.2"
# or, for SPARQL 1.2 / SPARQL-star:
# oxixml-sparql-syntax = { version = "0.1.2", features = ["sparql-12"] }
```

## Usage

```rust
use oxixml_sparql_syntax::{SparqlParser, SparqlSyntaxError};

fn main() -> Result<(), SparqlSyntaxError> {
    let query = SparqlParser::new().parse_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o . }")?;
    assert_eq!(query.to_string(), "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }");
    assert_eq!(query.to_sse(), "(project (?s ?p ?o) (bgp (triple ?s ?p ?o)))");
    Ok(())
}
```

## Conformance

`tests/suite.rs` runs the W3C `rdf-tests/sparql` syntax corpus (via `OXIXML_TESTSUITE_DIR`),
checking both accept/reject decisions and algebra round-trip stability. Current results:

| suite | positive | negative | round-trips |
|---|---|---|---|
| SPARQL 1.0 | 149 / 149 | 50 / 50 | 149 / 149 |
| SPARQL 1.1 | 108 / 108 | 61 / 61 | 108 / 108 |
| SPARQL 1.2 (`sparql-12`) | 126 / 126 | 72 / 72 | 126 / 126 |

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

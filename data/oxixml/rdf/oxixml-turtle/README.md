# oxixml-turtle

Streaming parsers and serializers for the RDF Turtle family of syntaxes, for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust foundation for
XML and RDF processing.

Covers [N-Triples](https://www.w3.org/TR/n-triples/), [N-Quads](https://www.w3.org/TR/n-quads/),
[Turtle](https://www.w3.org/TR/turtle/), [TriG](https://www.w3.org/TR/trig/) and
[Notation3](https://w3c.github.io/N3/spec/). Every format exposes a builder-style parser with
`for_slice`/`for_reader` iterators yielding `Triple`/`Quad` results, a low-level incremental push
API for streaming/`async` use, and syntax errors carrying precise line/column/offset positions.
The off-by-default `rdf-12` feature adds RDF 1.2 (RDF-star) quoted/triple terms and directional
language-tagged strings.

## Install

```toml
[dependencies]
oxixml-turtle = "0.1.2"
# with RDF 1.2 (RDF-star) support:
# oxixml-turtle = { version = "0.1.2", features = ["rdf-12"] }
```

## Example

```rust
use oxixml_turtle::TurtleParser;
use oxixml_model::vocab::rdf;
use oxixml_model::NamedNode;

let data = br#"@base <http://example.com/> .
@prefix schema: <http://schema.org/> .
<foo> a schema:Person ;
    schema:name "Foo" ."#;

let schema_person = NamedNode::new("http://schema.org/Person")?;
let mut count = 0;
for triple in TurtleParser::new().for_slice(data) {
    let triple = triple?;
    if triple.predicate == rdf::TYPE {
        assert!(triple.object == schema_person);
    }
    count += 1;
}
assert_eq!(count, 2);
# Result::<_, Box<dyn std::error::Error>>::Ok(())
```

## Replaces

The `std`-only, dependency-light replacement for [`oxttl`](https://crates.io/crates/oxttl): its
public API is mirrored so code written against `oxttl` ports over by changing only the `use` path.

## Conformance

Passes the complete W3C RDF 1.1 test suite (N-Triples, N-Quads, Turtle, TriG). Against the RDF 1.2
(RDF-star) suites: N-Triples 70/70, N-Quads 68/68, Turtle 96/96, TriG 60/60 — all passing (run via
`tests/rdf_tests.rs`, gated on the `OXIXML_TESTSUITE_DIR` environment variable).

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

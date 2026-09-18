# oxixml-jsonld

A complete [JSON-LD 1.1](https://www.w3.org/TR/json-ld11/) processor for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust foundation for
XML and RDF processing.

Two layers: a streaming RDF interface (`JsonLdParser`/`JsonLdSerializer`, the *toRdf*/*fromRdf*
pipelines) mirroring `oxjsonld`'s API, and the full [JSON-LD 1.1
API](https://www.w3.org/TR/json-ld11-api/) — `expand`, `compact`, `flatten`, `to_rdf`, `from_rdf`
— over an internal, `serde`-free JSON tree parsed via `oxixml-json` events. Context processing
(term definitions, `@vocab`, `@base`, protected terms, scoped and remote contexts via a
`DocumentLoader` hook) is implemented in full. The default loader performs **no** network or file
access; a document is only fetched if the caller supplies a loader. [JSON-LD
Framing](https://www.w3.org/TR/json-ld11-framing/) is not implemented — see the crate
documentation for the precise gap.

## Install

```sh
cargo add oxixml-jsonld@0.1.2
```

## Example

```rust
use oxixml_jsonld::{JsonLdParser};
use oxixml_model::vocab::rdf;
use oxixml_model::{NamedNode, Term};

let file = r#"{
    "@context": {"name": "http://schema.org/name"},
    "@id": "http://example.com/foo",
    "@type": "http://schema.org/Person",
    "name": "Foo"
}"#;

let person = NamedNode::new("http://schema.org/Person")?;
let mut has_type = false;
for quad in JsonLdParser::new().for_slice(file.as_bytes()) {
    let quad = quad?;
    if quad.predicate == rdf::TYPE && quad.object == Term::from(person.clone()) {
        has_type = true;
    }
}
assert!(has_type);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Replaces

The Pure-Rust replacement for [`oxjsonld`](https://crates.io/crates/oxjsonld).

## Conformance

Passes every entry it runs of the W3C JSON-LD 1.1 [API](https://w3c.github.io/json-ld-api/tests/)
and [Streaming](https://w3c.github.io/json-ld-streaming/tests/) test suites: expansion 376/376,
compaction 244/244, flattening 55/55, toRdf 455/455, fromRdf 52/52, streaming toRdf 471/471 (run
via `tests/jsonld_suite.rs`, gated on the `OXIXML_TESTSUITE_DIR` environment variable). Entries
targeting JSON-LD 1.0 or HTML input, and one generalized-RDF case a `Quad` cannot represent, are
not run.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

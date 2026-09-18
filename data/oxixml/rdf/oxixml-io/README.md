# oxixml-io

Unified RDF format detection, parsing, and serialization for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust foundation for
XML and RDF processing.

Wires together every RDF syntax this workspace implements behind one API: N-Triples, N-Quads,
Turtle, TriG, N3 (via `oxixml-turtle`), RDF/XML (via `oxixml-rdfxml`), JSON-LD (via
`oxixml-jsonld`), and TriX (via `oxixml-trix`, an OxiXML extension beyond `oxrdfio`'s format set).
`RdfParser::from_format`/`RdfSerializer::from_format` build iterator-based parsers and writers over
`Quad`s; `RdfFormat::from_extension`/`from_media_type` recognize file extensions and IANA media
types the same way `oxrdfio` does. The off-by-default `rdf-12` feature forwards RDF 1.2 (RDF-star)
support to the Turtle-family backends; RDF/XML, TriX and JSON-LD report an error for a triple term
since those syntaxes cannot express one.

## Install

```sh
cargo add oxixml-io@0.1.2
# with RDF 1.2 (RDF-star) support:
cargo add oxixml-io@0.1.2 --features rdf-12
```

## Example

```rust
use oxixml_io::{RdfFormat, RdfParser, RdfSerializer};

// Detect the format from a file extension, then parse.
let format = RdfFormat::from_extension("ttl").expect("known extension");
let turtle = r#"@prefix schema: <http://schema.org/> .
_:alice a schema:Person ;
    schema:name "Alice" ;
    schema:knows _:bob .
_:bob a schema:Person ;
    schema:name "Bob" ."#;

let quads = RdfParser::from_format(format)
    .rename_blank_nodes()
    .for_reader(turtle.as_bytes())
    .collect::<Result<Vec<_>, _>>()?;
assert_eq!(quads.len(), 5);

// Re-serialize the same quads as N-Quads (a dataset format, so every quad round-trips).
let mut serializer = RdfSerializer::from_format(RdfFormat::NQuads).for_writer(Vec::new());
for quad in &quads {
    serializer.serialize_quad(quad)?;
}
let nquads = String::from_utf8(serializer.finish()?)?;
assert_eq!(nquads.lines().count(), 5);
# Result::<(), Box<dyn std::error::Error>>::Ok(())
```

## Replaces

The dependency-light replacement for [`oxrdfio`](https://crates.io/crates/oxrdfio): its public API
— the `RdfFormat` enum, the `RdfParser`/`RdfSerializer` builders, and the unified error types — is
mirrored so code written against `oxrdfio` ports over by changing only the `use` paths, plus one
OxiXML extension: `RdfFormat::TriX`.

## Conformance

Inherits the conformance of its per-format backend crates (`oxixml-turtle`, `oxixml-rdfxml`,
`oxixml-jsonld`, `oxixml-trix`); see each crate's own README for figures. This crate's own
`tests/conversion.rs` covers cross-format detection, parsing, and round-tripping.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

# oxixml-rdfxml

A streaming [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/) parser and serializer for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust foundation for
XML and RDF processing.

`RdfXmlParser` supports the full RDF/XML 1.1 grammar: an optional `rdf:RDF` document element, node
and property elements, `rdf:about`/`rdf:ID`/`rdf:nodeID`/`rdf:resource`, property attributes,
`rdf:parseType` of `Literal`, `Resource` and `Collection`, containers, reification, `xml:lang` and
`xml:base` scoping, `rdf:datatype`, internal-DTD entities, and rejection of withdrawn terms.
`RdfXmlSerializer` writes striped RDF/XML, grouping consecutive same-subject triples into one
`rdf:Description` (or a typed node element when the group starts with `rdf:type`).

## Install

```toml
[dependencies]
oxixml-rdfxml = "0.1.2"
```

## Example

```rust
use oxixml_model::vocab::rdf;
use oxixml_model::NamedNode;
use oxixml_rdfxml::{RdfXmlParser, RdfXmlSerializer};

let file = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
 <schema:Person rdf:about="http://example.com/foo" schema:name="Foo" />
</rdf:RDF>"#;

let triples = RdfXmlParser::new()
    .for_slice(file)
    .collect::<Result<Vec<_>, _>>()?;
assert_eq!(triples.len(), 2);
let person = NamedNode::new("http://schema.org/Person")?;
assert!(triples.iter().any(|t| t.predicate == rdf::TYPE && t.object == person));

let mut serializer = RdfXmlSerializer::new()
    .with_prefix("schema", "http://schema.org/")?
    .for_writer(Vec::new());
for triple in &triples {
    serializer.serialize_triple(triple)?;
}
let out = String::from_utf8(serializer.finish()?)?;
assert!(out.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
# Result::<_, Box<dyn std::error::Error>>::Ok(())
```

## Replaces

The dependency-light replacement for [`oxrdfxml`](https://crates.io/crates/oxrdfxml): its public
API is mirrored so code written against `oxrdfxml` ports over by changing only the `use` paths.
The XML layer is `oxixml-xml` rather than `quick-xml`, and the data model is `oxixml-model` rather
than `oxrdf`.

## Conformance

166/166 passing on the W3C [RDF 1.1 RDF/XML syntax test
suite](https://w3c.github.io/rdf-tests/rdf/rdf11/rdf-xml/) (run via `tests/rdf_tests.rs`, gated on
the `OXIXML_TESTSUITE_DIR` environment variable).

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

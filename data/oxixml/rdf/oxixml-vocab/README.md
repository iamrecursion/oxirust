# oxixml-vocab

Standard RDF vocabulary constants for the [OxiXML](https://github.com/cool-japan/oxixml)
workspace — the COOLJAPAN Pure-Rust foundation for XML and RDF processing.

Every well-known RDF/RDFS/OWL/SHACL/etc. term used across the ecosystem is exposed as a `'static`
`NamedNodeRef` constant, built with `NamedNodeRef::new_const_unchecked` so that using a term
allocates nothing. Covered vocabularies: RDF 1.1/1.2, RDFS, XSD, OWL 2, SKOS, DCMI Terms, DCMI
Type, DCAT 3, FOAF, PROV-O, SHACL, OGC GeoSPARQL, W3C Basic Geo (WGS84), VoID, a curated
schema.org subset, and OWL-Time. Each module exposes a `NAMESPACE` constant, one term constant
per property/class, and an `ALL` slice for introspection.

## Install

```toml
[dependencies]
oxixml-vocab = "0.1.2"
```

## Example

```rust
use oxixml_model::Literal;
use oxixml_vocab::{rdf, xsd};

// Build an `xsd:integer`-typed literal using a vocabulary constant.
let one = Literal::new_typed_literal("1", xsd::INTEGER);
assert_eq!(one.datatype().as_str(), xsd::INTEGER.as_str());

// Namespaces are always absolute IRIs ending in `#` or `/`.
assert!(rdf::NAMESPACE.ends_with('#'));
assert_eq!(rdf::TYPE.as_str(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
```

## Replaces

No direct upstream equivalent — this crate is specific to the OxiXML/COOLJAPAN ecosystem.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

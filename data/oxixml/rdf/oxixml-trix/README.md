# oxixml-trix

A [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) (Triples in XML) parser and serializer for
the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust foundation
for XML and RDF processing.

TriX is a simple XML syntax for RDF datasets: a `<TriX>` document element contains zero or more
`<graph>` elements, each optionally named, followed by `<triple>` elements whose three term
children (`<uri>`, `<id>`, `<plainLiteral>`, `<typedLiteral>`) map directly onto
`oxixml-model`'s `NamedNode`/`BlankNode`/`Literal`. `TriXParser` is a streaming, pull-based
iterator of `Quad`s built on `oxixml-xml`'s namespace-aware reader; `TriXSerializer` writes quads
back out, grouping consecutive same-graph quads under one `<graph>` element.

## Install

```toml
[dependencies]
oxixml-trix = "0.1.2"
```

## Example

```rust
use oxixml_model::{Literal, NamedNode, Quad};
use oxixml_trix::{TriXParser, TriXSerializer};

let quad = Quad::new(
    NamedNode::new("http://example.com/s")?,
    NamedNode::new("http://example.com/p")?,
    Literal::new_simple_literal("o"),
    NamedNode::new("http://example.com/g")?,
);

let mut serializer = TriXSerializer::new().for_writer(Vec::new());
serializer.serialize_quad(&quad)?;
let bytes = serializer.finish()?;

let parsed = TriXParser::new()
    .for_slice(&bytes)
    .collect::<Result<Vec<_>, _>>()?;
assert_eq!(parsed, vec![quad]);
# Result::<_, Box<dyn std::error::Error>>::Ok(())
```

## Replaces

No direct upstream equivalent — TriX support is unique to OxiXML within the COOLJAPAN ecosystem.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

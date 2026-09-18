# oxixml-model

RDF 1.1/1.2 data model for the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
COOLJAPAN Pure-Rust foundation for XML and RDF processing.

`oxixml-model` provides the owned/borrowed term types every other OxiXML RDF crate builds on:
`NamedNode`, `BlankNode`, `Literal`, `Variable`, the `Term`/`Triple`/`Quad` unions (plus their
`*Ref` borrows), and the in-memory `Graph`/`Dataset` containers with triple-pattern queries and
blank-node canonicalization. Two off-by-default features extend it: `rdf-12` adds RDF 1.2
(RDF-star) embedded triple terms and directional language-tagged strings, and `rdfc-10` adds the
standard RDFC-1.0 dataset-canonicalization algorithm (SHA-256/384).

## Install

```sh
cargo add oxixml-model@0.1.2
# with RDF 1.2 (RDF-star) support:
cargo add oxixml-model@0.1.2 --features rdf-12
```

## Example

```rust
use oxixml_model::{Graph, NamedNodeRef, TripleRef};

let mut graph = Graph::new();
let ex = NamedNodeRef::new("http://example.com/foo")?;
let triple = TripleRef::new(ex, ex, ex);
graph.insert(triple);

assert!(graph.contains(triple));
assert_eq!(
    graph.to_string(),
    "<http://example.com/foo> <http://example.com/foo> <http://example.com/foo> .\n"
);
# Result::<_, oxixml_model::IriParseError>::Ok(())
```

## Replaces

The `std`-only, dependency-light replacement for [`oxrdf`](https://crates.io/crates/oxrdf): its
public API is mirrored so code written against `oxrdf` ports over by changing only the `use` path,
with three documented deviations noted in the crate documentation.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

# oxixml-hdt

HDT (Header-Dictionary-Triples) binary RDF for the [OxiXML](https://github.com/cool-japan/oxixml)
workspace — the COOLJAPAN Pure-Rust foundation for XML and RDF processing.

Implements the [HDT binary format](http://www.rdfhdt.org/hdt-binary-format/): a compact,
self-indexed binary encoding for large, static RDF datasets, built from a CRC16-guarded control-info
preamble, a block-compressed four-section dictionary mapping every distinct RDF term to a compact
integer ID, and a bitmap-indexed triples section encoded purely in dictionary IDs. `Hdt` ties these
together into a single, queryable, fully in-memory dataset; `HdtWriter` serializes RDF triples the
other way. This crate implements the default single-order (SPO) `BitmapTriples` layout and the
default `FourSectionDictionary` — the combination every mainstream HDT tool writes and reads; see
the crate documentation for the precise scope boundary (no secondary sort-order indexes or
`.hdt.index` companion file).

## Install

```sh
cargo add oxixml-hdt@0.1.2
```

## Example

```rust
use oxixml_hdt::{Hdt, HdtWriter};
use oxixml_model::{NamedNode, Triple};

let ex = |s: &str| NamedNode::new(format!("http://example.com/{s}")).expect("valid iri");
let triples = vec![
    Triple::new(ex("alice"), ex("knows"), ex("bob")),
    Triple::new(ex("bob"), ex("knows"), ex("alice")),
];

// Write.
let mut bytes = Vec::new();
HdtWriter::new().write(&mut bytes, triples.clone())?;

// Read back and query.
let hdt = Hdt::from_bytes(&bytes)?;
assert_eq!(hdt.len(), 2);

let alice = ex("alice");
let matches: Vec<Triple> = hdt
    .triples_matching(Some(alice.as_ref().into()), None, None)
    .collect::<Result<_, _>>()?;
assert_eq!(matches, vec![Triple::new(ex("alice"), ex("knows"), ex("bob"))]);
# Result::<(), Box<dyn std::error::Error>>::Ok(())
```

## Replaces

No direct upstream equivalent — HDT support is unique to OxiXML within the COOLJAPAN ecosystem.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

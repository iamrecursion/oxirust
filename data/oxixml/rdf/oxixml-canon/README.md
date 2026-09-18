# oxixml-canon

[RDF Dataset Canonicalization (RDFC-1.0)](https://www.w3.org/TR/rdf-canon/) for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust foundation for
XML and RDF processing.

RDFC-1.0 (a.k.a. URDNA2015) deterministically relabels the blank nodes of an RDF dataset so that
any two isomorphic datasets produce byte-for-byte identical canonical N-Quads — the basis for
dataset diffing, digital signatures, and content-addressed storage. `canonicalize` takes a
borrowed `Dataset`; `canonicalize_quads` is the streaming entry point over any
`IntoIterator<Item = Quad>`. The `*_with` variants accept `CanonicalizationOptions` to select the
hash algorithm (SHA-256 by default, SHA-384 per the spec) and a `WorkBudget` bounding the
factorial-worst-case `hash-n-degree-quads` step against adversarial input, per the spec's own
security considerations. No external dependencies are pulled in; hashing comes from
`oxixml-support`.

## Install

```sh
cargo add oxixml-canon@0.1.2
```

## Example

```rust
use oxixml_canon::canonicalize;
use oxixml_model::{BlankNode, Dataset, GraphNameRef, NamedNode, QuadRef};

let p = NamedNode::new("http://example.com/p")?;
let q = NamedNode::new("http://example.com/q")?;

// Two blank nodes, distinguishable only by structure.
let mut dataset = Dataset::new();
let a = BlankNode::default();
let b = BlankNode::default();
dataset.insert(QuadRef::new(&p, &q, &a, GraphNameRef::DefaultGraph));
dataset.insert(QuadRef::new(&a, &p, &b, GraphNameRef::DefaultGraph));

let result = canonicalize(&dataset)?;

// The blank nodes are relabeled to stable `c14n` identifiers.
assert_eq!(result.issued_map().len(), 2);
assert!(result.canonical_n_quads().contains("_:c14n0"));
# Result::<_, Box<dyn std::error::Error>>::Ok(())
```

## Replaces

No direct upstream equivalent — this crate implements the standard algorithm the COOLJAPAN
ecosystem uses wherever dataset canonicalization is needed.

## Conformance

86/86 passing on the [rdf-canon RDFC-1.0 test suite](https://w3c.github.io/rdf-canon/tests/) (run
via `tests/rdf_canon.rs`, gated on the `OXIXML_TESTSUITE_DIR` environment variable).

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

# oxixml-infoset

The [XPath and XQuery Data Model (XDM) 3.1](https://www.w3.org/TR/xpath-datamodel-31/), in pure
Rust, for the [OxiXML](https://github.com/cool-japan/oxixml) workspace.

This is the typed abstraction that XPath 3.1, XQuery 3.1, XSLT 3.0 and the schema validators all
evaluate over — what Xalan and Saxon call the *node tree and sequence model*. `NodeKind`
enumerates the seven node kinds; `XdmNode` is the accessor interface an engine is written
against (`node_kind`, `string_value`, `typed_value`, `parent`, `children`, …), implemented for
`NodeRef`, a cheap handle over a pluggable `XdmStore` backend. A node's `node_name` keeps the
prefix it was *written* with, even though the prefix plays no part in equality or a name test.
`AtomicValue` carries an XML Schema value together with its type, language and collation; `Item`
is a node, an atomic value, an `XdmMap` or an `XdmArray`; `Sequence` is a flat, ordered collection
that structurally cannot nest — the central axiom of the model. `XdmMap` is a persistent hash
array-mapped trie: cloning one is a reference-count bump, not a copy, so the pattern a query runs
constantly — derive one new map per key via `map:put`/`map:remove` — touches `O(log₃₂ n)` nodes
instead of the whole map. All thirteen XPath `Axis` variants are computed by `axis_nodes`, each
in the order the spec requires, with `compare_document_order`/`sort_document_order` and
`deep_equal` (`fn:deep-equal`) alongside them. This crate models *data*, never evaluation: there
is no expression tree or function library here.

## Install

```toml
[dependencies]
oxixml-infoset = "0.1.2"
```

## Usage

```rust
use oxixml_infoset::{NodeKind, Tree, XdmError, XdmNode};

fn main() -> Result<(), XdmError> {
    let tree = Tree::parse_str(r#"<catalog><book id="b1">Dune</book></catalog>"#)?;

    let root = tree.root();
    assert_eq!(root.node_kind(), NodeKind::Document);

    let catalog = tree.document_element().expect("a document element");
    assert_eq!(catalog.node_name().map(|n| n.to_string()).as_deref(), Some("catalog"));
    assert_eq!(catalog.string_value(), "Dune");

    let book = catalog.children().first().cloned().expect("one child");
    assert_eq!(book.attributes().len(), 1);
    Ok(())
}
```

## Conformance

`tests/conformance.rs` builds a `Tree` from every well-formed document in the W3C XML
Conformance Test Suite (via `OXIXML_XML_SUITES_DIR`) and asserts the XDM data-model axioms hold
for every node: kind/name/string-value consistency, exactly one document-order axis partition,
and stable round-tripping through re-parse.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

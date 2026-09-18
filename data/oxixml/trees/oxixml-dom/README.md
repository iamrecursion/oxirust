# oxixml-dom

A [DOM Level 3 Core](https://www.w3.org/TR/DOM-Level-3-Core/)-class mutable document tree over
an arena, with an XML builder and serializer, for the [OxiXML](https://github.com/cool-japan/oxixml)
workspace.

A `Document` owns one `Vec` of nodes and hands out `NodeId` handles into it: there are no
reference cycles or reference counting (dropping a document is a single deallocation), node
identifiers stay valid across mutations, there is no `unsafe` code anywhere, and the whole tree
is `Send`. It is the in-memory document model that canonicalization, XSLT, schema validation and
XInclude all build on — roughly the role Apache Xerces' DOM plays, redesigned around index
handles instead of reference-counted nodes.

## Install

```toml
[dependencies]
oxixml-dom = "0.1.2"
```

## Usage

```rust
use oxixml_dom::Document;

fn main() -> Result<(), oxixml_dom::DomError> {
    let mut doc = Document::new();
    let root = doc.create_element("catalog")?;
    doc.append_child(doc.root(), root)?;
    doc.set_attribute(root, "count", "0")?;

    assert_eq!(doc.to_xml(), "<catalog count=\"0\"/>");
    Ok(())
}
```

`build::parse_str` and `build::parse_reader` build a `Document` from XML text instead, resolving
namespaces with `NsReader` semantics; see the [crate documentation](https://docs.rs/oxixml-dom)
for the full round-trip example.

## Conformance

`tests/xmlts_roundtrip.rs` round-trips every *valid* document of the W3C XML Conformance Test
Suite through the tree (parse → serialize → re-parse → structural equality, then a further
byte-identical re-serialization). As of this release: **383 documents round-trip**; 3 more are
skipped for using a non-UTF-8 encoding and 8 for needing an external-DTD retrieval that a
non-validating processor is not required to perform.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

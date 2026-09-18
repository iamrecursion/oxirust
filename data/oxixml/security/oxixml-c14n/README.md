# oxixml-c14n

Canonical XML 1.0, 1.1, 2.0 and Exclusive XML Canonicalization, in Pure Rust.

Two XML documents can carry exactly the same information and still differ byte for byte:
attributes reorder, empty elements collapse, entities expand, namespace declarations move.
*Canonicalization* maps every such spelling onto one, so a digital signature can be computed
over meaning rather than over bytes. `oxixml-c14n` implements every canonicalization algorithm
the XML Security stack uses — Canonical XML 1.0/1.1, Exclusive C14N 1.0, and Canonical XML 2.0 —
reading an `oxixml_dom::Document` and a `NodeSet` (an XPath node-set, so a `<ds:Signature>`
transform can select an arbitrary document subset) and writing canonical UTF-8 octets.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-c14n = "0.1.2"
```

## Example

```rust
use oxixml_c14n::{exclusive_c14n, InclusiveNamespaces, NodeSet};
use oxixml_dom::build;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str(
        r#"<?xml version="1.0"?>
           <env:Envelope xmlns:env="urn:env" xmlns:unused="urn:unused" xml:lang="ja">
             <p:Payload xmlns:p="urn:payload" b="2" a="1">   text   </p:Payload>
           </env:Envelope>"#,
    )?;

    let payload = doc
        .get_elements_by_tag_name_ns(doc.root(), Some("urn:payload"), "Payload")
        .next()
        .expect("the payload element");
    let subset = NodeSet::subtree(&doc, payload);

    // Exclusive canonicalization keeps only the namespace the element visibly uses —
    // the shape almost every XML Signature needs.
    assert_eq!(
        exclusive_c14n(&doc, &subset, &InclusiveNamespaces::none(), false)?,
        br#"<p:Payload xmlns:p="urn:payload" a="1" b="2">   text   </p:Payload>"#,
    );
    Ok(())
}
```

## What it replaces

`oxixml-c14n` replaces the C14N modules of libxml2/xmlsec1 (C) or Apache Santuario (Java).

## Conformance

Checked against the W3C Canonical XML 2.0 test vectors via `tests/w3c_c14n2_testcases.rs`, and
against the `<ds:SignedInfo>` elements of the 45 W3C XML Signature 1.1 interoperability documents
via `tests/xmldsig_interop.rs`, whenever `OXIXML_XML_SUITES_DIR` is set. Last measured:
**30/30** Canonical XML 2.0 vectors and **45/45** `SignedInfo` elements canonicalize correctly.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

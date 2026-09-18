# oxixml-schema

W3C XML Schema (XSD 1.0/1.1) assembly and validation, in Pure Rust.

`oxixml-schema` is the Pure-Rust equivalent of the Apache Xerces schema engine. `SchemaSet`
collects schema documents — added directly, read from disk, or pulled in transitively through
`xs:include`/`xs:import`/`xs:redefine`/`xs:override` — then `compile()`s them into an immutable
`Schema`: every `QName` reference resolved, every constraining facet merged, every complex
type's effective content model turned into a finite automaton, and the schema component
constraints (including Unique Particle Attribution) checked. `Schema::validate_document`/
`validate_str` then assess an instance, honoring `xsi:type`, `xsi:nil`, substitution groups,
wildcards, identity constraints, and — because XSD 1.1 §3.13 defines them in terms of XPath
2.0 — `xs:assert`, the `assertion` facet, and `xs:alternative`, evaluated with the sibling
`oxixml-xpath` engine.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-schema = "0.1.2"
```

## Example

```rust
use oxixml_schema::SchemaSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut set = SchemaSet::new();
    set.add_document(
        None,
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="order">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="item" maxOccurs="unbounded">
                     <xs:complexType>
                       <xs:attribute name="sku" use="required">
                         <xs:simpleType>
                           <xs:restriction base="xs:string">
                             <xs:pattern value="[A-Z]{2}-[0-9]{4}"/>
                           </xs:restriction>
                         </xs:simpleType>
                       </xs:attribute>
                       <xs:attribute name="qty" type="xs:positiveInteger" default="1"/>
                     </xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    )?;
    let schema = set.compile()?;

    assert!(schema.validate_str(r#"<order><item sku="AB-1234" qty="3"/></order>"#).valid);

    // A bad `sku` breaks the pattern facet…
    let bad = schema.validate_str(r#"<order><item sku="oops"/></order>"#);
    assert_eq!(bad.first_error().map(|error| error.code()), Some("cvc-pattern-valid"));

    // …and an empty order does not satisfy `minOccurs="1"` on `item`.
    assert!(!schema.validate_str("<order/>").valid);
    Ok(())
}
```

## What it replaces

`oxixml-schema` replaces Apache Xerces (Java) or Saxon-EE for W3C XML Schema (XSD) validation;
there is no comparably complete Pure-Rust XSD 1.0/1.1 engine.

## Conformance

Gated against the [W3C XML Schema Test Suite](https://www.w3.org/XML/2004/xml-schema-test-suite/)
via `tests/xsts.rs` whenever `OXIXML_XML_SUITES_DIR` is set, in two full passes (XSD 1.0 and XSD
1.1). Last measured: **99.7%** of 39,131 XSD 1.0 assertions (39,007 passed) and **99.6%** of
41,427 assertions across the whole XSD 1.1 pass (41,275 passed) — **99.1%** (2,247 of 2,267)
over the 1.1-only Saxonica/IBM/Oracle contributions specifically, where the 1.1-only machinery
lives. Documented gaps are listed in the crate's rustdoc.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

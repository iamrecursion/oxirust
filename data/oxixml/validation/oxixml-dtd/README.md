# oxixml-dtd

XML 1.0 DTD parsing and validation, in Pure Rust.

`oxixml-dtd` implements the half of the XML 1.0 Recommendation that a validating processor
needs and a well-formedness-only parser does not: the grammar of the internal and external DTD
subsets, compiled content-model automata, attribute defaulting/typing, ID/IDREF resolution, and
entity expansion. Two hardening measures are on by default: an `ExpansionLimits` budget that
turns a "billion laughs" entity bomb into a bounded error in milliseconds instead of exhausting
memory, and no implicit external access — external entities are fetched only if the caller
installs an `EntityResolver`.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-dtd = "0.1.2"
```

## Example

```rust
use oxixml_dtd::{DtdParser, ValidationOptions, Validator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dtd = DtdParser::new().parse_internal_subset(
        r#"<!ELEMENT book (title, chapter+)>
           <!ELEMENT title (#PCDATA)>
           <!ELEMENT chapter (#PCDATA)>
           <!ATTLIST chapter id ID #REQUIRED lang NMTOKEN "en">"#,
    )?;
    assert_eq!(
        dtd.element("book").map(|d| d.content().to_string()).as_deref(),
        Some("(title,chapter+)")
    );

    let mut validator = Validator::new(&dtd, ValidationOptions::default());
    validator.start_element("book", &[]);
    validator.start_element("title", &[]);
    validator.characters("Moby-Dick");
    validator.end_element("title");
    let defaults = validator.start_element("chapter", &[("id", "c1")]);
    assert_eq!(defaults, vec![("lang".to_owned(), "en".to_owned())]);
    validator.end_element("chapter");
    validator.end_element("book");
    assert!(validator.finish().is_empty());
    Ok(())
}
```

## What it replaces

There is no widely used Pure-Rust DTD validator. `oxixml-dtd` replaces reaching for Apache
Xerces or libxml2 to parse and validate the DTD half of XML 1.0.

## Conformance

Gated against the [W3C XML Conformance Test Suite](https://www.w3.org/XML/Test/) (`xmlconf`) via
`tests/w3c_xmlconf.rs` whenever `OXIXML_XML_SUITES_DIR` points at a checkout. Last measured:
**933/933** applicable DTD cases passed, with 0 known divergences.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

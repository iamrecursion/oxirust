# oxixml-schematron

ISO/IEC 19757-3 Schematron rule-based validation, in Pure Rust, on the `oxixml-xpath` engine.

Schematron describes a schema as a set of *patterns*, each a set of *rules* that fire on the
nodes matching a *context*, each asserting or reporting the truth of an XPath *test* — a
different shape from a grammar-based language like DTD, XSD or RELAX NG. `Schematron::compile`
resolves `sch:phase` activation, `sch:let` variables, abstract-pattern instantiation, and
`sch:rule`/`sch:extends` inheritance into a `Compiled` schema, which validates any number of
documents and produces a structured `SchematronReport` with an SVRL serialization. Every XPath
expression in a schema compiles under `oxixml-xpath` at the language level the schema's
`queryBinding` selects (1.0, 2.0 or 3.0).

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-schematron = "0.1.2"
```

## Example

```rust
use oxixml_schematron::Schematron;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = r#"
        <schema xmlns="http://purl.oclc.org/dsdl/schematron">
          <ns prefix="c" uri="urn:example:catalog"/>
          <pattern id="prices">
            <rule context="c:book">
              <assert test="c:price" id="has-price">every book needs a price</assert>
            </rule>
          </pattern>
        </schema>
    "#;

    let compiled = Schematron::compile(schema, None)?;

    let good = r#"<c:catalog xmlns:c="urn:example:catalog">
                    <c:book><c:price>9.99</c:price></c:book>
                  </c:catalog>"#;
    assert!(compiled.validate(good)?.is_valid());

    let bad = r#"<c:catalog xmlns:c="urn:example:catalog"><c:book/></c:catalog>"#;
    let report = compiled.validate(bad)?;
    let failure = report.failed_asserts().next().expect("one failure");
    assert_eq!(failure.id.as_deref(), Some("has-price"));
    Ok(())
}
```

## What it replaces

`oxixml-schematron` replaces XSLT-hosted Schematron implementations (the ISO Schematron
skeleton, `probatron`) for schemas whose `let`/`test`/`context` are XPath — which covers the
overwhelming majority of real-world Schematron schemas. Running an embedded XSLT sequence
constructor under the `xslt` query binding is a documented gap; see the crate's rustdoc.

## Conformance

`tests/conformance.rs` compiles every real-world `.sch` schema in a reference corpus whenever
`OXIXML_XML_SUITES_DIR` is set. Last measured: **307 of 345** schemas compiled; the residue is
almost entirely XSD-to-Schematron-generated schemas that lean on `xs:anySimpleType` casts or
cross-file rule references outside plain ISO Schematron authoring.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

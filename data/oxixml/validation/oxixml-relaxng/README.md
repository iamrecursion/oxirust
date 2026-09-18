# oxixml-relaxng

A complete RELAX NG validator, in Pure Rust — both the XML syntax and the compact syntax.

Compiling a schema runs the full pipeline the specification prescribes: parse into the RELAX NG
data model (from XML, or from the compact syntax); check the full syntax; apply the twenty-one
simplification rules (`include`/`externalRef` expansion, `combine`, grammar flattening,
`notAllowed`/`empty` propagation); verify the restrictions of §7; compile into a hash-consed
pattern arena. Validation uses James Clark's derivative algorithm and recovers from errors, so
one run reports every problem it can rather than only the first. Two datatype libraries are
built in: the empty-URI library (`string`/`token`) and the full W3C XML Schema datatype library,
backed by `oxixml-xsd` and `oxixml-regex`.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-relaxng = "0.1.2"
```

## Example

```rust
use oxixml_relaxng::RelaxNg;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = RelaxNg::from_compact(
        r#"
        element addressBook {
          element card {
            element name { text },
            element email { text }
          }*
        }
        "#,
    )?;

    assert!(schema.validate(
        r#"<addressBook>
             <card><name>John Smith</name><email>js@example.com</email></card>
           </addressBook>"#
    ).is_ok());

    // A missing `email` element is reported, with the name that was expected.
    let errors = schema
        .validate("<addressBook><card><name>John Smith</name></card></addressBook>")
        .expect_err("the card is incomplete");
    assert!(errors[0].to_string().contains("email"));
    Ok(())
}
```

## What it replaces

`oxixml-relaxng` replaces James Clark's Jing (Java) or libxml2's RELAX NG module for validating
against RELAX NG schemas in either syntax.

## Conformance

Gated against James Clark's own [RELAX NG specification test suite](https://github.com/relaxng/jing-trang)
via `tests/spectest.rs` (XML syntax) and `tests/compacttest.rs` (compact syntax), whenever
`OXIXML_XML_SUITES_DIR` is set. Last measured: **385/385** schemas judged correctly, **289/289**
valid instances accepted and **291/291** invalid instances rejected on the specification suite;
**31/31** compact-syntax documents correctly rejected and **56/56** XML/compact syntax pairs
agree on the compact-syntax suite.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

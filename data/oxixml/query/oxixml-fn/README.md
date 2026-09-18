# oxixml-fn

The W3C *XPath and XQuery Functions and Operators 3.1* library, in Pure Rust.

`oxixml-fn` is the shared function and operator substrate that `oxixml-xpath`, `oxixml-xquery`
and `oxixml-xslt` all evaluate on top of: it implements F&O 3.1 over the XDM values of
`oxixml-infoset`. There is no parser and no expression tree here — a function library is exactly
what this is. Each area of the specification is a module (`strings`, `numeric`, `datetime`,
`nodes`, `sequences`, `higher_order`, `maps`, `arrays`, `qnames`, `json`, ...), and each function
is a plain Rust function with the types its signature implies, so the library is usable directly
without going through a dynamic argument list. The JSON functions (`fn:parse-json`,
`fn:json-to-xml`, `fn:xml-to-json`, with the `fallback` option) live here too, since they are
pure functions over strings and XDM values with no I/O of their own; `fn:doc`, `fn:json-doc` and
the rest of the resource-retrieval functions stay with the evaluators that own an I/O policy.
Every failure is an `FnError` carrying the qualified error name (`err:FORG0001`, `err:FOCH0002`,
...) a conforming processor would raise.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-fn = "0.1.2"
```

## Example

```rust
use oxixml_fn::{numeric, strings, FnContext};
use oxixml_infoset::{AtomicType, AtomicValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = FnContext::new();

    assert_eq!(strings::substring("metadata", 4.0, Some(3.0)), "ada");
    assert!(strings::contains(&context, "tattoo", "tat", None)?);

    let sum = numeric::add(
        &AtomicValue::integer(1_i64),
        &AtomicValue::parse(AtomicType::Decimal, "2.5")?,
    )?;
    assert_eq!(sum.string_value(), "3.5");
    Ok(())
}
```

## What it replaces

There is no standalone Pure-Rust F&O 3.1 library. `oxixml-fn` replaces the function-library half
of Saxon or Apache Xalan.

## Conformance

`tests/conformance.rs` checks every function name and arity this crate registers against the
names the W3C *QT3* (FOTS) suite uses, whenever `OXIXML_XML_SUITES_DIR` is set — this crate has
no expression evaluator of its own to run the suite's inputs through, so full expression-level
conformance is exercised transitively by the QT3 runs in the consuming `oxixml-xpath` and
`oxixml-xquery` crates.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

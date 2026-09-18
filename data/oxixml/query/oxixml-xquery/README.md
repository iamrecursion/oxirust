# oxixml-xquery

XQuery 3.1, in Pure Rust.

`oxixml-xquery` is the query language Saxon occupies elsewhere: the whole of XQuery 3.1 — FLWOR
expressions, node constructors, the query prolog, library modules, `typeswitch`, `switch`,
`try`/`catch` — evaluated over the XDM of `oxixml-infoset`, with the function library of
`oxixml-fn` behind it. XQuery is treated as a superset of XPath: any byte range of an expression
that is pure XPath is compiled and evaluated by the sibling `oxixml-xpath` crate directly, so it
behaves exactly as that engine's own conformance suite says it does, with no second
implementation to drift. Node constructors follow XQuery 3.1 §3.9's namespace-fixup and
`xml:base`-inheritance rules closely, including nested direct constructors and per-module base
URIs, construction modes and boundary-space policies.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-xquery = "0.1.2"
```

## Example

```rust
use oxixml_xquery::{DynamicQueryContext, XQuery};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A FLWOR expression: grouping, with the non-grouping variables collected per group.
    let grouped = XQuery::compile_default(
        "string-join(
           for $n in 1 to 6
           group by $parity := $n mod 2
           order by $parity
           return concat($parity, ':', count($n)),
         ' ')",
    )?;
    assert_eq!(
        grouped.execute_to_string(&DynamicQueryContext::new())?,
        "0:3 1:3"
    );
    Ok(())
}
```

## What it replaces

`oxixml-xquery` replaces Saxon's XQuery 3.1 engine. The XQuery Update Facility, schema
awareness and Full Text are documented gaps; see the crate's rustdoc.

## Conformance

Gated against the W3C *QT3* test suite (FOTS), compiling every case as XQuery 3.1, via
`tests/qt3.rs` whenever `OXIXML_XML_SUITES_DIR` is set. Last measured: **99.7%** of the 3,092
attempted XQuery-only cases (FLWOR, constructors, prolog) and **99.4%** of the 29,407 cases
attempted across the whole suite.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

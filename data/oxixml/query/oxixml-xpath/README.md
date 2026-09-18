# oxixml-xpath

XPath 1.0 through 3.1, in Pure Rust.

`oxixml-xpath` is the query engine that Apache Xalan and Saxon occupy elsewhere: a lexer and
recursive-descent parser for the whole XPath 3.1 grammar, and an evaluator over the XDM of
`oxixml-infoset` with the function library of `oxixml-fn` behind it. `XPathVersion` is not a
hint — it selects both what parses and what it means: XPath 1.0 gives the four-type language of
node-sets/booleans/numbers/strings, XPath 2.0+ gives XDM sequences and typed values, 3.0 adds
function items, and 3.1 adds maps and arrays. A few functions need more than a plain Rust
signature can carry and are implemented here directly instead of in `oxixml-fn`: the JSON
functions' `fallback` option (`fn:parse-json`, `fn:json-to-xml`, `fn:json-doc`) and
`fn:format-number`'s EQName decimal-format-name lookup, which needs the static context. A host
language embedding this engine can see the call site's live focus through the exported
`calling_focus` — which is how `oxixml-xslt` implements `fn:key` and `fn:id`.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-xpath = "0.1.2"
```

## Example

```rust
use oxixml_infoset::Tree;
use oxixml_xpath::{StaticContext, XPath, XPathVersion};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tree = Tree::parse_str(
        r#"<catalog>
             <book year="1965"><title>Dune</title><price>9.99</price></book>
             <book year="1815"><title>Emma</title><price>5.50</price></book>
           </catalog>"#,
    )?;
    let root = tree.root();

    // Paths, predicates and the function library.
    let titles = XPath::compile(
        "//book[@year < 1900]/title/string()",
        XPathVersion::XPath31,
        StaticContext::new(),
    )?;
    assert_eq!(titles.evaluate_to_string(root.clone())?, "Emma");

    // Sequences, `for`, and the arrow operator.
    let total = XPath::compile(
        "sum(for $b in //book return xs:decimal($b/price))",
        XPathVersion::XPath31,
        StaticContext::new(),
    )?;
    assert_eq!(total.evaluate_to_string(root.clone())?, "15.49");
    Ok(())
}
```

## What it replaces

`oxixml-xpath` replaces Apache Xalan's XPath engine, or Saxon's, for evaluating XPath 1.0–3.1
expressions.

## Conformance

Gated against the W3C *QT3* test suite (FOTS) via `tests/qt3.rs`, filtered to the expression
tests this engine's dependencies satisfy, whenever `OXIXML_XML_SUITES_DIR` is set. Last measured:
**92.9%** overall — 20,203 cases attempted, 11,618 skipped as needing schema awareness, XSLT,
serialization or an XQuery-only construct — **97.0%** of the operator/production tests (`op/`,
`prod/`) and **88.2%** of the function tests (`fn/`, `math/`, `map/`, `array/`). See the crate's
rustdoc for the shortfall breakdown.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

# oxixml-xslt

An XSLT 1.0 processor with the essentials of XSLT 2.0 and 3.0, in Pure Rust.

`oxixml-xslt` occupies the place Apache Xalan and Saxon occupy elsewhere: it compiles a
stylesheet into template rules, patterns and instructions, and runs them over the XDM of
`oxixml-infoset` with `oxixml-xpath` as its expression engine and `oxixml-fn` as its function
library. XSLT 1.0 is complete; XSLT 2.0/3.0 essentials — `xsl:for-each-group`, `xsl:function`,
`xsl:analyze-string`, `xsl:try`/`xsl:catch`, `xsl:iterate`, `xsl:context-item`, match patterns
with `except`/`intersect`, and more — are implemented too. Streaming, schema-awareness and
packaging are documented gaps, and a stylesheet that reaches one is rejected with the crate's own
`OXIXML0001` rather than silently mishandled. Embedding is configurable: `StaticOptions` can
switch on strict `XTDE0160` rejection of `version="1.0"` behaviour instead of the permissive
default, `Parameters` can raise the template-instantiation recursion ceiling for a caller running
on a larger stack, and `CompiledStylesheet::transform_optional` runs with no source document at
all, for a named-template entry point that needs none.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-xslt = "0.1.2"
```

## Example

```rust
use oxixml_infoset::Tree;
use oxixml_xslt::Stylesheet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stylesheet = Stylesheet::compile_str(
        r#"<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:output method="xml" omit-xml-declaration="yes"/>
             <xsl:template match="/catalog"><titles count="{count(book)}"><xsl:apply-templates
                 select="book"><xsl:sort select="@year" data-type="number"/>
               </xsl:apply-templates></titles></xsl:template>
             <xsl:template match="book"><title year="{@year}"><xsl:value-of
               select="title"/></title></xsl:template>
           </xsl:stylesheet>"#,
    )?;

    let source = Tree::parse_str(
        r#"<catalog>
             <book year="1965"><title>Dune</title></book>
             <book year="1815"><title>Emma</title></book>
           </catalog>"#,
    )?;

    let result = stylesheet.transform(&source.root())?;
    assert_eq!(
        result.as_str(),
        r#"<titles count="2"><title year="1815">Emma</title><title year="1965">Dune</title></titles>"#
    );
    Ok(())
}
```

## What it replaces

`oxixml-xslt` replaces Apache Xalan or Saxon for running XSLT 1.0–3.0 stylesheets.

## Conformance

Gated against the W3C *XSLT 3.0* test suite (`xslt30-test`), filtered by an allowlist to the
test sets whose features this processor implements, via `tests/conformance.rs` whenever
`OXIXML_XML_SUITES_DIR` is set. A bare percentage means little without stating the allowlist it
was measured over, so the exact pass rate is tracked in the workspace `CHANGELOG.md` rather than
frozen here; as of this release it is **99.9%** (756 of 757 cases attempted) over the 72-set
core-language allowlist, which now includes `xsl:context-item` (the one residual failure is
`xsl:function cache="yes"` memoization, an honest timeout). The crate's rustdoc lists what is
implemented and what is a documented gap.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

# oxixml-html

A pure-Rust [HTML5](https://html.spec.whatwg.org/multipage/parsing.html) tokenizer and
tree-builder producing an `oxixml-dom` tree, for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace.

This is what lets the RDF side of the workspace ingest the web: RDFa, microdata and
`<link rel="alternate">` discovery all need a parser that reads real-world tag soup exactly the
way a browser does, then hands back the same `Document` that canonicalization, XPath and XSLT
already operate on. It implements all eighty-odd tokenizer states and all twenty-one tree
construction insertion modes of the HTML standard — the adoption agency algorithm, foster
parenting, active formatting elements, `<template>` contents, MathML/SVG foreign content — plus
encoding sniffing (including the `<meta charset>` prescan) and an HTML fragment serializer.
Because XML is stricter than HTML, the handful of constructs an XML DOM cannot represent (e.g.
`&#1;`, `<!--a--b-->`) are rewritten with a recorded parse error; the original spelling is kept
alongside for round-tripping, and well-formed HTML is never rewritten at all.

## Install

```toml
[dependencies]
oxixml-html = "0.1.2"
```

## Usage

```rust
use oxixml_html::{QuirksMode, parse_document};

fn main() {
    let parsed = parse_document("<!DOCTYPE html><p>Hello<p>World");
    let document = parsed.document();

    // The parser inserted <html>, <head> and <body>, and closed the first <p>.
    assert_eq!(parsed.quirks_mode(), QuirksMode::NoQuirks);

    // It is an ordinary oxixml-dom document, so every DOM operation works.
    let body = document
        .get_elements_by_tag_name_ns(document.root(), Some(oxixml_html::HTML_NAMESPACE), "body")
        .next()
        .expect("a body element");
    assert_eq!(document.text_content(body).as_deref(), Some("HelloWorld"));
}
```

## Conformance

Measured against the html5lib-tests corpus (via `OXIXML_HTML5LIB_TESTS_DIR` or
`OXIXML_XML_SUITES_DIR`):

| suite | result |
|---|---|
| tokenizer (13 files, all six initial states) | **7032 / 7032** |
| tree construction (57 files) | **1792 / 1792** |
| encoding (3 files, runnable blocks) | **82 / 82** |

Tree construction is a clean 100%, including the four `<selectedcontent>` blocks that used to be
a known gap: HTML's "clone an option into a selectedcontent" algorithm runs alongside tree
construction proper, driven by the `option`/`selectedcontent` elements' own insertion, connection
and pop reactions, and this crate now reproduces it (`src/tree/select.rs`). The suite's
`scripted/` blocks are counted as skips, not runnable cases: their expected tree comes from
executing a script (`document.write`, or a script-driven DOM mutation), which needs a JavaScript
engine this parser deliberately does not have.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

# oxixml-xinclude

XInclude 1.0 (Second Edition) processing over `oxixml-dom`, in Pure Rust.

`process_document` walks an already-parsed DOM document and replaces every `xi:include` element
with the content it points to: `parse="xml"` resources are parsed, recursively
XInclude-processed, optionally narrowed with `xpointer` (the `element()` scheme plus minimal
`xmlns()`/`xpointer(id(...))` support), and grafted in with base-URI and `xml:lang` fixup
applied; `parse="text"` resources are decoded and spliced in as a text node; `xi:fallback` runs
when a resource error occurs; inclusion loops and self-references are detected per the
specification. Resource fetching is entirely pluggable through the `ResourceLoader` trait — this
crate never touches the filesystem or the network on its own, so it is SSRF-safe by default.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-xinclude = "0.1.2"
```

## Example

```rust
use oxixml_dom::build;
use oxixml_xinclude::{MapLoader, process_document};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = build::parse_str(
        r#"<doc xmlns:xi="http://www.w3.org/2001/XInclude">
             <xi:include href="https://example.com/greeting.xml"/>
           </doc>"#,
    )?;

    let loader =
        MapLoader::new().with_text("https://example.com/greeting.xml", "<p>hello</p>");
    process_document(&mut doc, "https://example.com/index.xml", &loader)?;

    let root = doc.document_element().expect("document element");
    let grafted = doc
        .get_elements_by_tag_name(root, "p")
        .next()
        .expect("the xi:include was replaced by the fetched <p>");
    assert_eq!(doc.text_content(grafted).as_deref(), Some("hello"));
    Ok(())
}
```

## What it replaces

There is no widely used Pure-Rust XInclude processor. `oxixml-xinclude` replaces the XInclude
module of libxml2.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

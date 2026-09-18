# oxixml-quickxml-compat

A drop-in [`quick-xml`](https://docs.rs/quick-xml) compatibility shim over
[`oxixml-xml`](https://docs.rs/oxixml-xml). Every module path, type name and
public method signature mirrors the `quick-xml` 0.41 public API exactly, so
downstream code that depends on `quick_xml::*` keeps compiling and behaving
the same way once the dependency is repointed at this crate under the
`quick_xml` package alias — no source changes required.

## Install

```toml
[dependencies]
quick_xml = { package = "oxixml-quickxml-compat", version = "0.1.2" }
```

## Usage

Reading, ported verbatim from the `quick-xml` 0.41 README (only the `use`
paths are rewritten to import this crate under the `quick_xml` alias):

```rust
use oxixml_quickxml_compat as quick_xml;
use quick_xml::events::Event;
use quick_xml::reader::Reader;

let xml = r#"<tag1 att1 = "test">
                <tag2><!--Test comment-->Test</tag2>
                <tag2>Test 2</tag2>
             </tag1>"#;
let mut reader = Reader::from_str(xml);
reader.config_mut().trim_text(true);

let mut count = 0;
let mut txt = Vec::new();
let mut buf = Vec::new();

loop {
    match reader.read_event_into(&mut buf) {
        Err(e) => panic!("Error at position {}: {:?}", reader.error_position(), e),
        Ok(Event::Eof) => break,
        Ok(Event::Start(e)) => {
            match e.name().as_ref() {
                b"tag1" => println!(
                    "attributes values: {:?}",
                    e.attributes().map(|a| a.expect("attr").value).collect::<Vec<_>>()
                ),
                b"tag2" => count += 1,
                _ => (),
            }
        }
        Ok(Event::Text(e)) => txt.push(e.decode().expect("utf-8").into_owned()),
        _ => (),
    }
    buf.clear();
}
assert_eq!(count, 2);
assert_eq!(txt, vec!["Test".to_string(), "Test 2".to_string()]);
```

## Features

`arbitrary`, `async-tokio`, `encoding`, `escape-html`, `overlapped-lists`,
`serialize` and `serde-types` all forward 1:1 to the identically-named
`oxixml-xml` features and are off by default.

`arbitrary` forwards to `oxixml-xml`'s own `arbitrary` feature, which mirrors
`quick-xml` 0.41's optional `arbitrary` dependency: `Arbitrary` impls on all
the event types (`BytesStart`, `BytesEnd`, `BytesText`, `BytesCData`,
`BytesDecl`, `BytesPI`, `BytesRef` and `Event`), with the same validation
`quick-xml` applies. A fuzz target that depends on `quick-xml/arbitrary`
therefore builds against this shim unchanged, once repointed at it, by
enabling this crate's `arbitrary` feature instead. Every `quick-xml` 0.41
feature — including `encoding::DecodingReader` and `serde_helpers` — is now
mirrored. This crate **replaces** [`quick-xml`](https://docs.rs/quick-xml)
itself, as a path-compatible alias (see `Install`); the underlying engine is
the Pure-Rust `oxixml-xml`.

## Testing

No numeric conformance gate of its own — the test suite (`quickxml_doc_examples.rs`,
`quickxml_readme_examples.rs`, `serde_examples.rs`, `async_examples.rs`) instead
replicates `quick-xml` 0.41's own doc and README examples verbatim, to confirm
source-level compatibility under the `quick_xml` alias.

## License

Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

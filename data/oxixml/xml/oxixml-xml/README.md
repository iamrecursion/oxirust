# oxixml-xml

A Pure-Rust XML 1.0/1.1 pull parser and writer that mirrors the **quick-xml
0.41** public API (module paths, type names and method signatures) closely
enough that most downstream code can switch over with a one-line dependency
change. It provides the zero-copy `Event` model, a namespace-aware `NsReader`,
an events-based `Writer`, escaping utilities, and — behind opt-in features — a
serde (de)serializer/serializer, `tokio` async I/O, and non-UTF-8 decoding.
Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the
COOLJAPAN Pure-Rust foundation for XML and RDF processing.

## Install

```toml
[dependencies]
oxixml-xml = "0.1.2"

# with optional features:
# oxixml-xml = { version = "0.1.2", features = ["serialize", "encoding"] }
```

## Usage

```rust
use oxixml_xml::escape::{escape, unescape};
use oxixml_xml::events::{BytesStart, Event};
use oxixml_xml::name::QName;

// Build a start tag and inspect its name and attributes.
let start = BytesStart::new("greeting").with_attributes([("lang", "en")]);
assert_eq!(start.name(), QName(b"greeting"));
let (key, value) = {
    let attr = start.attributes().next().expect("one attribute").expect("valid");
    (attr.key.as_ref().to_vec(), attr.value.to_vec())
};
assert_eq!((key.as_slice(), value.as_slice()), (&b"lang"[..], &b"en"[..]));

// Wrap it in an event; events borrow their payload bytes.
let event = Event::Start(start);
assert_eq!(&*event, b"greeting lang=\"en\"".as_ref());

// Escape and unescape round-trip.
assert_eq!(escape("a < b & c"), "a &lt; b &amp; c");
assert_eq!(unescape("a &lt; b &amp; c").expect("valid"), "a < b & c");
```

For reading a document event-by-event, use `oxixml_xml::reader::Reader` (or
the namespace-aware `NsReader`); for writing, `oxixml_xml::writer::Writer` and
its `create_element` builder. See the crate documentation for full examples.

## Features

`arbitrary` (`arbitrary::Arbitrary` impls on the event types, for fuzz
targets), `async-tokio` (async I/O via `tokio`), `encoding` (non-UTF-8 input,
backed by the Pure-Rust `oxixml-encoding` crate instead of `encoding_rs`),
`escape-html` (HTML5 named entity recognition), `overlapped-lists` and
`serialize` (serde (de)serialization through the `de`/`se` modules), and
`serde-types` (serde impls on a few event types, plus the `serde_helpers`
module). All are off by default.

`arbitrary` mirrors `quick-xml` 0.41's optional `arbitrary` dependency
exactly: `BytesStart`, `BytesEnd`, `BytesText`, `BytesCData`, `BytesDecl`,
`BytesPI`, `BytesRef` and `Event` all implement `arbitrary::Arbitrary`, with
the same validation `quick-xml` applies — `BytesStart` requires an
all-alphanumeric, non-empty name; `BytesText` requires an all-alphanumeric
string but allows it to be empty; every other type accepts any string. A fuzz
harness can call `Event::arbitrary` directly on raw fuzzer bytes instead of
hand-rolling a generator. Off by default, so the crate stays dependency-free
(see `Replaces`) unless explicitly opted into.

## Replaces

[`quick-xml`](https://docs.rs/quick-xml) — `oxixml-xml` is a Pure-Rust,
`unsafe`-free replacement with the same 0.41-era API surface.

## Conformance

Gated on the W3C XML Conformance Test Suite (`xmlts`), restricted to the
subset applicable to a non-validating, non-external-entity-reading processor:
**984/988** `not-wf` cases correctly rejected (4 cases are intentional
exceptions, tracked in an allowlist) and **478/478** `valid`/`invalid`
documents parsed successfully. XML 1.1 and Namespaces in XML 1.1 cases are
included. See `tests/xmlts.rs` for the exact skip policy and allowlist
rationale.

## License

Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

# oxixml-json

Streaming JSON event parser and writer for the [OxiXML](https://github.com/cool-japan/oxixml)
workspace — the COOLJAPAN Pure-Rust foundation for XML and RDF processing.

`oxixml-json` turns a JSON document into (or back from) a flat stream of `JsonEvent`s —
`StartObject`, `ObjectKey`, `String`, `Number`, ..., `EndObject` — rather than building an
in-memory tree, which makes it a good foundation for JSON-LD, SPARQL-results-JSON, or any other
JSON-based format processor without paying for a DOM you only walk once. It targets
[RFC 8259](https://www.rfc-editor.org/rfc/rfc8259) exactly: every number form the grammar allows
(and only those), every string escape including `\uXXXX` surrogate pairs, and UTF-8 validation of
the input. `SliceJsonParser` is zero-copy where no escape processing is needed; `ReaderJsonParser`
and `LowLevelJsonParser` cover `std::io::Read` sources and hand-fed buffers respectively.

## Install

```sh
cargo add oxixml-json@0.1.2
```

## Example

```rust
use oxixml_json::{JsonEvent, SliceJsonParser, WriterJsonSerializer};

// Parse.
let mut parser = SliceJsonParser::new(br#"{"answer": 42, "ok": true}"#);
assert_eq!(parser.parse_next()?, JsonEvent::StartObject);
assert_eq!(parser.parse_next()?, JsonEvent::ObjectKey("answer".into()));
assert_eq!(parser.parse_next()?, JsonEvent::Number("42".into()));
assert_eq!(parser.parse_next()?, JsonEvent::ObjectKey("ok".into()));
assert_eq!(parser.parse_next()?, JsonEvent::Boolean(true));
assert_eq!(parser.parse_next()?, JsonEvent::EndObject);
assert_eq!(parser.parse_next()?, JsonEvent::Eof);

// And write the same events back out.
let mut writer = WriterJsonSerializer::new(Vec::new());
writer.serialize_event(JsonEvent::StartObject)?;
writer.serialize_event(JsonEvent::ObjectKey("answer".into()))?;
writer.serialize_event(JsonEvent::Number("42".into()))?;
writer.serialize_event(JsonEvent::ObjectKey("ok".into()))?;
writer.serialize_event(JsonEvent::Boolean(true))?;
writer.serialize_event(JsonEvent::EndObject)?;
assert_eq!(writer.finish()?, br#"{"answer":42,"ok":true}"#);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Replaces

The Pure-Rust replacement for
[`json-event-parser`](https://crates.io/crates/json-event-parser).

## Conformance

Validated against an inline suite of RFC 8259 accept/reject vectors (`tests/conformance.rs`)
covering number forms, string escapes, surrogate pairs, and malformed-input rejection.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

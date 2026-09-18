# oxixml-encoding

Character-encoding decoders and encoders for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust foundation for
XML and RDF processing. It plays the role that [`encoding_rs`](https://docs.rs/encoding_rs) plays
in other Rust XML stacks, written from scratch against `std` alone: incremental, chunk-safe
decoding (a partial multi-byte sequence at a chunk boundary is carried over correctly), the inverse
encoding direction, byte-order-mark detection, and the XML 1.0 Appendix F encoding-sniffing
heuristic for `<?xml ... encoding="..."?>` prologues.

## Install

```toml
[dependencies]
oxixml-encoding = "0.1.2"
```

## Usage

```rust
use oxixml_encoding::{sniff_xml_encoding, Encoding};

// A UTF-8 document that self-declares its own encoding.
let xml = b"<?xml version=\"1.0\" encoding=\"UTF-8\"?><root/>";
let sniffed = sniff_xml_encoding(xml);
assert_eq!(sniffed.declared_encoding, Some(Encoding::Utf8));

// Decode a chunk of UTF-8 incrementally, one byte at a time, and get back the original text.
let mut decoder = Encoding::Utf8.new_decoder();
let mut out = String::new();
for (i, byte) in "héllo".as_bytes().iter().enumerate() {
    let last = i + 1 == "héllo".len();
    decoder
        .decode_to_string(std::slice::from_ref(byte), &mut out, last)
        .expect("valid UTF-8 fed one byte at a time");
}
assert_eq!(out, "héllo");
```

## Supported encodings

`UTF-8`, `UTF-16` (both byte orders), `ISO-8859-1`/`windows-1252`, `US-ASCII`, the rest of the
WHATWG Encoding Standard's legacy single-byte encodings (the full `ISO-8859-*`, `windows-125x`,
`KOI8-*`, `IBM866`, `macintosh`, `x-mac-cyrillic` families), the three legacy Japanese multi-byte
encodings (`Shift_JIS`, `EUC-JP`, `ISO-2022-JP`), and the legacy CJK multi-byte encodings (`GBK`,
`gb18030`, `Big5`, `EUC-KR`) prioritized for COOLJAPAN. `gb18030`'s encoder is total over Unicode:
every scalar value except the single documented exception, `U+E5E5`, has a representation (this
does not imply every character round-trips through encode-then-decode -- 18 Private Use Area code
points are deliberately encoded to a fixed legacy byte pair for GB18030-2005 compatibility; see
`oxixml_encoding`'s own rustdoc for the full list). `UTF-32` and `EBCDIC` prologues are recognized
by `sniff_xml_encoding` but are out of scope for decoding.

## What it replaces

A from-scratch, dependency-free alternative to `encoding_rs` sized for what an XML processor
actually needs, plus the XML-specific BOM detection and prologue-sniffing that `encoding_rs` itself
does not provide.

## Testing

Every supported encoding is round-tripped through a "chunk-boundary torture test" — encode, then
feed the bytes back through an incremental decoder one, two, and three bytes at a time — and the
multi-byte encodings have dedicated coverage for their lead/trail byte error ranges, GBK/gb18030's
2-/4-byte state machine, Big5's two-code-point decode outputs, and ISO-2022-JP's escape-sequence
state machine.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

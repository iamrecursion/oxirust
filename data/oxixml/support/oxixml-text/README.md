# oxixml-text

Lexical/text codecs shared across the [OxiXML](https://github.com/cool-japan/oxixml) workspace —
the COOLJAPAN Pure-Rust foundation for XML and RDF processing. It gathers the small, spec-exact
text primitives that XML, XSD, and IRI processing all need and would otherwise be re-implemented
(subtly differently) throughout the stack: RFC 4648 base64 and hex, `AsciiSet`-parameterized
percent-encoding for IRIs, and `f64`/`f32` string formatting for both ECMAScript's
`Number::toString(10)` and the XSD 1.1 canonical mapping.

## Install

```toml
[dependencies]
oxixml-text = "0.1.2"
```

## Usage

```rust
use oxixml_text::{base64, hex, num, percent, ws};

// Base64: round-trip through the standard alphabet.
let encoded = base64::base64_encode(b"OxiXML");
assert_eq!(encoded, "T3hpWE1M");
assert_eq!(base64::base64_decode(&encoded).expect("valid base64"), b"OxiXML");

// Hex: xsd:hexBinary uses uppercase; decoding accepts either case.
assert_eq!(hex::encode_upper(&[0xDE, 0xAD, 0xBE, 0xEF]), "DEADBEEF");
assert_eq!(hex::decode("deadbeef").expect("valid hex"), vec![0xDE, 0xAD, 0xBE, 0xEF]);

// Percent-encoding an IRI path component.
assert_eq!(percent::percent_encode(b"a b/c", &percent::PATH_SEGMENT), "a%20b%2Fc");

// ECMAScript-style vs. XSD-canonical number formatting diverge on purpose.
assert_eq!(num::ecmascript_double(0.1), "0.1");
assert_eq!(num::xsd_canonical_double(0.1), "1.0E-1");

// Whitespace collapse implements the XSD `collapse` facet.
assert_eq!(ws::collapse("  a\t b  \n c  "), "a b c");
```

## What it replaces

Covers the ground of the `base64`, `hex`/`data-encoding`, and `percent-encoding` crates in one
place, plus the ECMAScript and XSD 1.1 number-formatting algorithms that none of those crates (nor
Rust's own `{}`/`{:e}` formatters) implement.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

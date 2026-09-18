# oxixml-support

Zero-dependency substrate for the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
COOLJAPAN Pure-Rust foundation for XML and RDF processing. It depends on nothing but `std` and
sits at the very bottom of the dependency graph, gathering the small, sharp primitives that would
otherwise be re-implemented (subtly differently) in a dozen parsers: `memchr`-class SWAR byte
scanning, line/column position tracking, deterministic PRNGs, FIPS 180-4 SHA/HMAC digests, and a
minimal unsigned bigint that backs RSA signature verification in `oxixml-dsig`.

## Install

```toml
[dependencies]
oxixml-support = "0.1.2"
```

## Usage

```rust
use oxixml_support::bytes::memchr;
use oxixml_support::hash::sha256;
use oxixml_support::pos::LineColTracker;

// Byte scanning without `unsafe`.
assert_eq!(memchr(b'/', b"path/to/thing"), Some(4));

// A SHA-256 digest matching the FIPS 180-4 "abc" test vector.
let digest = sha256(b"abc");
assert_eq!(digest[0], 0xba);
assert_eq!(digest[1], 0x78);

// Track line and column through mixed line endings.
let mut tracker = LineColTracker::new();
tracker.advance(b"first\r\nsecond");
let position = tracker.position();
assert_eq!(position.line, 2);
assert_eq!(position.column, 7);
```

## What it replaces

An in-tree, `unsafe`-free stand-in for the pieces of `memchr`, `sha1`/`sha2`/`hmac`, and
`rand_pcg` that an XML/RDF stack needs, so the rest of OxiXML has zero external dependencies at
this layer.

## Conformance

The SHA-1/SHA-2 digests are validated against the official NIST SHA test vectors (`""`, `"abc"`,
the two-block message, and one million `'a'`s); HMAC is validated against RFC 2202 (HMAC-SHA-1)
and RFC 4231 (HMAC-SHA-256/384/512).

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

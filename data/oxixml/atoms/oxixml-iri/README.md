# oxixml-iri

RFC 3987 IRI/URI parsing, resolution, and validation — the atom that gives every other layer
of [OxiXML](https://github.com/cool-japan/oxixml) (RDF terms, XML base URIs, catalogs, XInclude)
a single, spec-exact notion of an absolute or relative internationalized reference.

`Iri<T>` and `IriRef<T>` parse and validate the full
[RFC 3987](https://www.rfc-editor.org/rfc/rfc3987) grammar — scheme, authority (including
`IP-literal`/`IPv6address`/`IPvFuture` hosts), the five path forms, query, and fragment,
including the Unicode `ucschar`/`iprivate` ranges that distinguish an IRI from a plain
[RFC 3986](https://www.rfc-editor.org/rfc/rfc3986) URI. `Iri` additionally guarantees its value
is absolute; `IriRef` allows relative references too. Both are generic over their text storage,
so you can parse zero-copy from a borrowed `&str` or own a `String`. On top of parsing, the
crate implements strict [RFC 3986 §5](https://www.rfc-editor.org/rfc/rfc3986#section-5)
reference resolution (`resolve`) and its practical inverse, `relativize`. An opt-in `serde`
feature (off by default) adds `Serialize`/`Deserialize` for both types, encoding to the plain
string form and turning a parse failure into a `serde::de::Error` — mirroring `oxiri`'s own
`serde` feature exactly.

## Install

```sh
cargo add oxixml-iri@0.1.2
```

## Usage

```rust
use oxixml_iri::{Iri, IriRef};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse and validate an absolute base IRI.
    let base = Iri::parse("http://example.com/dir/file?query#frag")?;
    assert_eq!(base.scheme(), "http");
    assert_eq!(base.authority(), Some("example.com"));
    assert_eq!(base.path(), "/dir/file");

    // Resolve a relative reference against it (RFC 3986 §5).
    let resolved = base.resolve(&IriRef::parse("../other?q2")?)?;
    assert_eq!(resolved.as_str(), "http://example.com/other?q2");

    // Relativizing is the practical inverse of resolving.
    let relative = base.relativize(&resolved)?;
    assert_eq!(base.resolve(&relative)?, resolved);

    // Internationalized (non-ASCII) IRIs are accepted, per RFC 3987.
    assert!(Iri::parse("http://example.com/日本語").is_ok());

    // Malformed input is rejected with a descriptive error.
    assert!(Iri::parse("http://example.com/a b").is_err()); // raw space
    Ok(())
}
```

## Replaces

Pure-Rust, `std`-only drop-in replacement for [`oxiri`](https://crates.io/crates/oxiri): it
mirrors `oxiri`'s public API shape (`Iri`, `IriParseError`), so code written against it typically
ports over by changing only the `use` path.

## Not included: normalization

Parsing is deliberately **non-normalizing** and comparison is codepoint-exact. No RFC 3986
§6.2.2 syntax-based normalization (scheme/host case folding, percent-triplet case folding,
unreserved-character decoding, dot-segment removal outside `resolve`) and no RFC 3987 §3.1
IDNA/ToASCII host conversion are applied — `HTTP://EX.COM/A`, `%2f`/`%2F` and `例.example` /
`xn--fsq.example` all stay distinct. That matches RDF's exact-string IRI term identity, but
differs from URL libraries that normalize: normalize explicitly beforehand if RFC 3986
equivalence is what you need.

## Conformance

Verified against the RFC 3986 §5.4 "Normal" and "Abnormal" reference-resolution example tables,
transcribed verbatim from the RFC text and round-tripped through `relativize`, in addition to the
crate's own parser/resolver unit and integration tests.

## License

Licensed under the Apache License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>).
See the `LICENSE` file at the root of the [OxiXML](https://github.com/cool-japan/oxixml)
workspace.

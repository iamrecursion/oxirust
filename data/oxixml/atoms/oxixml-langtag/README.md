# oxixml-langtag

[BCP 47](https://www.rfc-editor.org/rfc/rfc5646) (RFC 5646) language-tag parsing,
well-formedness checking, and case normalization — the atom
[OxiXML](https://github.com/cool-japan/oxixml) uses anywhere an `xml:lang` value or an
RDF language-tagged literal needs to be validated.

`LanguageTag::parse` checks conformance to the RFC 5646 §2.1 `Language-Tag` ABNF grammar:
primary/extended language, script, region, variants (rejecting a duplicate variant), extensions
(rejecting a duplicate singleton), `privateuse`, and the 26 fixed `grandfathered` tags. This is
RFC 5646's **well-formedness** check (§2.2.9); it does not consult the IANA Language Subtag
Registry, so — exactly like `oxilangtag` — it cannot tell whether a well-formed tag's subtags are
actually *assigned* (that stronger check is *validity*, out of scope here).
`LanguageTag::parse_and_normalize` additionally applies the §2.1.1 case-normalization algorithm
(language/extlang/variant/extension/privateuse lowercase, script Titlecase, region UPPERCASE).

The crate is `#![no_std]` (with `alloc`). The default-on `std` feature adds only the
`std::error::Error` implementation for `LanguageTagParseError`, so embedded targets can use
`default-features = false` — mirroring `oxilangtag`.

## Install

```sh
cargo add oxixml-langtag@0.1.2
```

## Usage

```rust
use oxixml_langtag::LanguageTag;

fn main() -> Result<(), oxixml_langtag::LanguageTagParseError> {
    // Well-formedness checking, preserving the original casing.
    let tag = LanguageTag::parse("zh-cmn-Hans-CN")?;
    assert_eq!(tag.primary_language(), "zh");
    assert_eq!(tag.extended_language(), Some("cmn"));
    assert_eq!(tag.script(), Some("Hans"));
    assert_eq!(tag.region(), Some("CN"));

    // Case normalization per RFC 5646 §2.1.1: language/extlang/variant/extension/privateuse
    // lowercase, script Titlecase, region UPPERCASE.
    let normalized = LanguageTag::parse_and_normalize("mn-cyrl-mn")?;
    assert_eq!(normalized.as_str(), "mn-Cyrl-MN");

    // Duplicate variants and duplicate extension singletons are rejected.
    assert!(LanguageTag::parse("de-1996-1996").is_err());
    Ok(())
}
```

## Replaces

Pure-Rust drop-in replacement for [`oxilangtag`](https://crates.io/crates/oxilangtag): it mirrors
`oxilangtag`'s public API shape (`LanguageTag`, `LanguageTagParseError`).

## Conformance

Exercised against RFC 5646 Appendix A's worked examples, the fixed grandfathered-tag list, the
well-formedness edge cases from §2.1 (duplicate variants/extensions), and the §2.1.1
case-normalization algorithm, transcribed directly from the RFC text.

## License

Licensed under the Apache License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>).
See the `LICENSE` file at the root of the [OxiXML](https://github.com/cool-japan/oxixml)
workspace.

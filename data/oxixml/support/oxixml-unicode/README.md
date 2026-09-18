# oxixml-unicode

Unicode Character Database (UCD) tables and algorithms for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust foundation for
XML and RDF processing. This `std`-only crate packages a compact, sorted copy of the UCD together
with the general-category, block, script, case-folding, and normalization algorithms every
downstream XML/XSD/XPath parser and validator needs to agree on exactly. XML's own
character-class productions (`Char`, `NameStartChar`, `NameChar`, white space) come straight from
the XML 1.0/1.1 grammars, not the UCD.

## Install

```toml
[dependencies]
oxixml-unicode = "0.1.2"
```

## Usage

```rust
use oxixml_unicode::{gc, norm, xml, case};
use oxixml_unicode::gc::GeneralCategory;

// XML name productions (identical for XML 1.0 5e and XML 1.1).
assert!(xml::is_name_start_char('x'));
assert!(!xml::is_name_start_char('-'));
assert!(xml::is_name_char('-'));

// Unicode general categories and the regex one-letter groups.
assert_eq!(gc::general_category('A'), GeneralCategory::UppercaseLetter);
assert!(gc::matches_group('A', 'L'));

// Case folding for caseless comparison.
assert_eq!(case::fold_str("Straße"), "strasse");

// Canonical composition and decomposition (UAX #15).
assert_eq!(norm::nfc("A\u{030A}"), "\u{00C5}"); // A + ring  -> Å
assert_eq!(norm::nfd("\u{00C5}"), "A\u{030A}"); // Å         -> A + ring
assert!(norm::is_nfc("\u{00C5}"));
```

## What it replaces

Covers the ground of `unicode-normalization` plus general-category/script/block property lookups,
generated from a single UCD snapshot (`oxixml_unicode::UNICODE_VERSION`, currently `"17.0.0"`), so
every OxiXML crate agrees on the same Unicode data instead of pulling in several overlapping
crates.

## Conformance

Normalization is checked against the official Unicode `NormalizationTest.txt`: the default test
run validates an embedded every-10th-line subset generated from the same UCD version as the
crate's tables, and the full file can be run locally by pointing the `OXIXML_UCD_NORMTEST`
environment variable at a copy of it. XML character-class productions are cross-checked against
reference implementations written directly from the XML 1.0 (Fifth Edition) and XML 1.1 grammars.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

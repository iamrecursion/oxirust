//! `oxixml-unicode` — Unicode property data, character classes, and
//! normalization for the whole [OxiXML](https://github.com/cool-japan/oxixml)
//! workspace.
//!
//! Part of the COOLJAPAN Pure-Rust foundation for all XML and RDF processing.
//! This crate is `std`-only and has **no external dependencies**: it packages a
//! compact, sorted copy of the [Unicode Character Database][ucd] (UCD) together
//! with the algorithms every downstream parser, validator, and query engine
//! needs to agree on exactly:
//!
//! * [`xml`] — the character-class productions of the XML 1.0 (Fifth Edition)
//!   and XML 1.1 grammars: `Char`, `RestrictedChar`, `NameStartChar`,
//!   `NameChar`, and XML white space. These come straight from the grammar
//!   productions, not from the UCD.
//! * [`gc`] — the 30 Unicode [`gc::GeneralCategory`] values, a
//!   [`gc::general_category`] lookup, and [`gc::matches_group`] for the
//!   one-letter groups used by `\p{L}`-style regular expressions.
//! * [`blocks`] — [`blocks::block`], the Unicode block containing a character,
//!   with UAX #44-LM3-style loose name matching for the XSD `\p{IsBasicLatin}`
//!   regular-expression forms.
//! * [`scripts`] — the [`scripts::Script`] property and the
//!   [`scripts::script`] lookup.
//! * [`case`] — simple upper/lower/title mappings, the full case mappings of
//!   `SpecialCasing.txt` ([`case::full_uppercase`], [`case::full_lowercase`],
//!   [`case::full_titlecase`], and their [`case::CaseLocale`]-tailored
//!   variants for Lithuanian, Turkish, and Azerbaijani), and Unicode case
//!   folding ([`case::simple_fold`] and [`case::fold_str`]).
//! * [`norm`] — the four Unicode normalization forms
//!   ([`norm::nfc`], [`norm::nfd`], [`norm::nfkc`], [`norm::nfkd`]) plus the
//!   [`norm::is_nfc`] quick check, implemented per [UAX #15][uax15].
//!
//! All tables are generated offline from the UCD; the crate embeds the version
//! string in [`UNICODE_VERSION`].
//!
//! [ucd]: https://www.unicode.org/reports/tr44/
//! [uax15]: https://www.unicode.org/reports/tr15/
//!
//! # Example
//!
//! ```
//! use oxixml_unicode::{gc, norm, xml, case};
//! use oxixml_unicode::gc::GeneralCategory;
//!
//! // XML name productions (identical for XML 1.0 5e and XML 1.1).
//! assert!(xml::is_name_start_char('x'));
//! assert!(!xml::is_name_start_char('-'));
//! assert!(xml::is_name_char('-'));
//!
//! // Unicode general categories and the regex one-letter groups.
//! assert_eq!(gc::general_category('A'), GeneralCategory::UppercaseLetter);
//! assert!(gc::matches_group('A', 'L'));
//!
//! // Case folding for caseless comparison.
//! assert_eq!(case::fold_str("Straße"), "strasse");
//!
//! // Canonical composition and decomposition (UAX #15).
//! assert_eq!(norm::nfc("A\u{030A}"), "\u{00C5}"); // A + ring  -> Å
//! assert_eq!(norm::nfd("\u{00C5}"), "A\u{030A}"); // Å         -> A + ring
//! assert!(norm::is_nfc("\u{00C5}"));
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod blocks;
pub mod case;
pub mod gc;
pub mod norm;
pub mod scripts;
pub mod xml;

mod tables;

/// The version of the Unicode Character Database this crate's tables were
/// generated from, as an `"X.Y.Z"` string (for example `"17.0.0"`).
pub const UNICODE_VERSION: &str = "17.0.0";

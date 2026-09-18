//! `oxixml-langtag` — BCP 47 ([RFC 5646](https://www.rfc-editor.org/rfc/rfc5646)) language-tag
//! parsing, well-formedness checking, and case normalization. Replaces
//! [`oxilangtag`](https://crates.io/crates/oxilangtag) with the same public API shape.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing.
//!
//! # Scope
//!
//! [`LanguageTag::parse`] and [`LanguageTag::parse_and_normalize`] check **well-formedness**
//! ([RFC 5646 §2.2.9](https://www.rfc-editor.org/rfc/rfc5646#section-2.2.9)): conformance to the
//! `Language-Tag` ABNF grammar of [RFC 5646 §2.1](https://www.rfc-editor.org/rfc/rfc5646#section-2.1),
//! covering:
//!
//! * `language` — a primary language subtag (2-3 letters, optionally followed by up to three
//!   extended-language subtags; or 4 letters, reserved; or 5-8 letters, a registered subtag);
//! * `script` — exactly 4 letters;
//! * `region` — exactly 2 letters, or exactly 3 digits;
//! * `variant` — 5-8 alphanumerics, or a digit followed by 3 alphanumerics; repeatable, but
//!   never with the same variant twice;
//! * `extension` — a singleton letter/digit (any but `x`/`X`) followed by one or more 2-8
//!   alphanumeric value subtags; repeatable, but never with the same singleton twice;
//! * `privateuse` — the introducer `x`/`X` followed by one or more 1-8 alphanumeric value
//!   subtags, either as the tail of a `langtag` or as a standalone tag (`x-whatever`);
//! * the 26 fixed `grandfathered` tags (e.g. `i-klingon`, `zh-min-nan`, `sgn-BE-FR`).
//!
//! This crate does **not** consult the IANA Language Subtag Registry, so it cannot tell whether
//! a well-formed tag's subtags are actually *assigned* — that additional check is what RFC 5646
//! calls **validity**, and is out of scope here, exactly as it is for `oxilangtag`.
//!
//! Beyond bare syntax, [`LanguageTag::parse`] also rejects a duplicate variant subtag or a
//! duplicate extension singleton (e.g. `ar-a-aaa-b-bbb-a-ccc`, RFC 5646 Appendix A's own example
//! of an invalid tag) — RFC 5646 §2.2.9 lists both as exclusions from *validity*, but since a
//! tag with either has no unambiguous reading, this crate treats them as parse failures rather
//! than silently accepting the first (or last) occurrence.
//!
//! # `no_std`
//!
//! The crate is `#![no_std]` and needs only `alloc` (owned tags are backed by
//! [`alloc::string::String`]). The default-on `std` feature adds nothing but the
//! [`std::error::Error`] implementation for [`LanguageTagParseError`], so embedded targets can
//! depend on it with `default-features = false` — exactly as `oxilangtag` allows.
//!
//! # Example
//!
//! ```
//! use oxixml_langtag::LanguageTag;
//!
//! // Well-formedness checking, preserving the original casing.
//! let tag = LanguageTag::parse("zh-cmn-Hans-CN")?;
//! assert_eq!(tag.primary_language(), "zh");
//! assert_eq!(tag.extended_language(), Some("cmn"));
//! assert_eq!(tag.script(), Some("Hans"));
//! assert_eq!(tag.region(), Some("CN"));
//!
//! // Case normalization per RFC 5646 §2.1.1: language/extlang/variant/extension/privateuse
//! // lowercase, script Titlecase, region UPPERCASE.
//! let normalized = LanguageTag::parse_and_normalize("mn-cyrl-mn")?;
//! assert_eq!(normalized.as_str(), "mn-Cyrl-MN");
//!
//! // Grandfathered tags normalize to their one fixed canonical spelling.
//! let grandfathered = LanguageTag::parse_and_normalize("SGN-be-FR")?;
//! assert_eq!(grandfathered.as_str(), "sgn-BE-FR");
//!
//! // Duplicate variants and duplicate extension singletons are rejected.
//! assert!(LanguageTag::parse("de-1996-1996").is_err());
//! assert!(LanguageTag::parse("ar-a-aaa-b-bbb-a-ccc").is_err());
//! # Ok::<(), oxixml_langtag::LanguageTagParseError>(())
//! ```

#![no_std]
#![deny(missing_docs)]
#![deny(unsafe_code)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod error;
mod extension_iter;
mod grandfathered;
mod parse;
mod positions;
mod subtag_iter;
mod tag;
mod tag_ops;

pub use error::LanguageTagParseError;
pub use tag::LanguageTag;

//! `oxixml-iri` — RFC 3987 IRI/URI parsing, resolution, and validation.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate is the Pure-Rust,
//! `std`-only replacement for the [`oxiri`](https://crates.io/crates/oxiri) crate, mirroring
//! its public API shape so that code written against `oxiri::{Iri, IriParseError}` ports over
//! by changing only the `use` path.
//!
//! # What this crate validates
//!
//! [`Iri`] and [`IriRef`] parse and validate against the full
//! [RFC 3987](https://www.rfc-editor.org/rfc/rfc3987) `IRI`/`IRI-reference` grammar: `scheme`,
//! `iauthority` (`iuserinfo`, `ihost` — including `IP-literal` forms, `IPv6address` and
//! `IPvFuture` — and `port`), the five `ipath-*` variants, `iquery`, and `ifragment`, including
//! the Unicode `ucschar`/`iprivate` ranges that distinguish an IRI from a plain
//! [RFC 3986](https://www.rfc-editor.org/rfc/rfc3986) URI. [`Iri`] additionally guarantees its
//! value is *absolute* (has a `scheme`); [`IriRef`] allows relative references too.
//!
//! # What this crate computes
//!
//! * [`Iri::resolve`] / [`IriRef::resolve`] implement [RFC 3986 §5](https://www.rfc-editor.org/rfc/rfc3986#section-5)
//!   relative-reference resolution — the *strict* algorithm (§5.3), with no backward-compatible
//!   leniency for malformed base IRIs.
//! * [`Iri::relativize`] computes a reasonably short [`IriRef`] that resolves back to a given
//!   target IRI when resolved against a given base — the practical inverse of `resolve`.
//!
//! # What this crate does *not* do
//!
//! Parsing is **non-normalizing**, and comparison is codepoint-exact — an [`Iri`] compares
//! equal to another only when the two strings are byte-for-byte identical. Concretely:
//!
//! * No [RFC 3986 §6.2.2](https://www.rfc-editor.org/rfc/rfc3986#section-6.2.2) syntax-based
//!   normalization. The `scheme` and `ihost` keep their original case (`HTTP://EX.COM/A` stays
//!   `HTTP://EX.COM/A`), percent-encoding triplets keep their original case (`%2f` and `%2F`
//!   stay distinct), unreserved characters are never percent-decoded (`%7E` stays `%7E`, it
//!   does not become `~`), and `remove_dot_segments` runs only inside [`Iri::resolve`] —
//!   parsing `http://ex.com/a/./b` on its own leaves the `.` segment in place.
//! * No [RFC 3987 §3.1](https://www.rfc-editor.org/rfc/rfc3987#section-3.1) IDNA/ToASCII
//!   conversion of `ihost`: a Unicode host is validated but never Punycode-encoded, and an
//!   `xn--` host is never decoded. The two spellings of one domain therefore stay distinct.
//! * No scheme-based normalization (default ports such as `:80` for `http` are kept, an empty
//!   path is not rewritten to `/`) and no protocol-based normalization.
//!
//! That is deliberate: RDF defines IRI term identity as *exact string* equality
//! ([RDF 1.1 Concepts §3.1](https://www.w3.org/TR/rdf11-concepts/#section-IRIs)), so
//! normalizing would silently merge terms that RDF, SPARQL and XML consider different. But it
//! does differ from a typical URL library — code ported from one that normalizes will observe
//! different equality, hashing and deduplication results, and must normalize explicitly
//! beforehand if RFC 3986 equivalence is what it needs.
//!
//! # Example
//!
//! ```
//! use oxixml_iri::{Iri, IriRef};
//!
//! // Parse and validate an absolute base IRI.
//! let base = Iri::parse("http://example.com/dir/file?query#frag")?;
//! assert_eq!(base.scheme(), "http");
//! assert_eq!(base.authority(), Some("example.com"));
//! assert_eq!(base.path(), "/dir/file");
//!
//! // Resolve a relative reference against it (RFC 3986 §5).
//! let resolved = base.resolve(&IriRef::parse("../other?q2")?)?;
//! assert_eq!(resolved.as_str(), "http://example.com/other?q2");
//!
//! // Relativizing is the practical inverse of resolving.
//! let relative = base.relativize(&resolved)?;
//! assert_eq!(base.resolve(&relative)?, resolved);
//!
//! // Internationalized (non-ASCII) IRIs are accepted, per RFC 3987.
//! assert!(Iri::parse("http://example.com/\u{65e5}\u{672c}\u{8a9e}").is_ok());
//!
//! // Malformed input is rejected with a descriptive error.
//! assert!(Iri::parse("http://example.com/a b").is_err()); // raw space
//! assert!(Iri::parse("not a scheme://host/").is_err()); // space in scheme
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod error;
mod iri;
mod iri_ref;
mod positions;
mod resolve;
mod validate;

pub use error::{IriParseError, IriRelativizeError};
pub use iri::Iri;
pub use iri_ref::IriRef;
pub use resolve::OutputBuffer;

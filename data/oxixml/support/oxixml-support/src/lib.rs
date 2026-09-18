//! `oxixml-support` — the zero-dependency substrate every other OxiXML crate builds on.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate deliberately depends on
//! **nothing but `std`**: it sits at the very bottom of the dependency graph so that every
//! other crate can rely on it without risking a cycle.
//!
//! It gathers the small, sharp, well-tested primitives that would otherwise be re-implemented
//! (subtly differently) in a dozen parsers:
//!
//! * [`bytes`] — `memchr`-class byte scanning using safe word-at-a-time (SWAR) techniques,
//!   with no `unsafe` and no external crates, plus [`ct_eq`](bytes::ct_eq) for comparing a
//!   secret or forgeable value without leaking where it differs.
//! * [`pos`] — line/column/offset bookkeeping ([`TextPosition`](pos::TextPosition) and
//!   [`LineColTracker`](pos::LineColTracker)) that understands `LF`, `CRLF` and lone `CR`
//!   line breaks.
//! * [`rng`] — deterministic, non-cryptographic pseudo-random generators
//!   ([`SplitMix64`](rng::SplitMix64) and [`Pcg32`](rng::Pcg32)) used for property testing
//!   and reproducible fuzzing seeds.
//! * [`hash`] — streaming SHA-1 / SHA-2 digests and HMAC implemented directly from
//!   [FIPS 180-4](https://doi.org/10.6028/NIST.FIPS.180-4), validated against the official
//!   NIST and RFC test vectors.
//! * [`bigint`] — a minimal arbitrary-precision **unsigned** integer that exists to back RSA
//!   signature verification in `oxixml-dsig`.
//! * [`ec`] — arithmetic on the NIST prime curves P-256 and P-384 ([`Curve`](ec::Curve),
//!   [`AffinePoint`](ec::AffinePoint) and the SEC 1 point encoding), the foundation the
//!   elliptic-curve signature and key-agreement algorithms are built on.
//!
//! # Example
//!
//! ```
//! use oxixml_support::bytes::memchr;
//! use oxixml_support::hash::sha256;
//! use oxixml_support::pos::LineColTracker;
//!
//! // Byte scanning without `unsafe`.
//! assert_eq!(memchr(b'/', b"path/to/thing"), Some(4));
//!
//! // A SHA-256 digest matching the FIPS 180-4 "abc" test vector.
//! let digest = sha256(b"abc");
//! assert_eq!(digest[0], 0xba);
//! assert_eq!(digest[1], 0x78);
//!
//! // Track line and column through mixed line endings. Line and column are 0-based in
//! // storage; only `Display` renders the human-facing 1-based numbering.
//! let mut tracker = LineColTracker::new();
//! tracker.advance(b"first\r\nsecond");
//! let position = tracker.position();
//! assert_eq!(position.line, 1);
//! assert_eq!(position.column, 6);
//! assert_eq!(position.to_string(), "line 2, column 7");
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod bigint;
pub mod bytes;
pub mod ec;
pub mod hash;
pub mod pos;
pub mod rng;

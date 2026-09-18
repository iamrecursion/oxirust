//! `oxixml-text` — lexical/text codecs shared across the OxiXML stack.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate gathers the small,
//! spec-exact text primitives that XML, XSD, and IRI processing all need and that would
//! otherwise be re-implemented (subtly differently) throughout the stack:
//!
//! * [`base64`] — RFC 4648 base64, with standard and URL-safe alphabets, optional padding,
//!   and both a strict decoder and a "forgiving" decoder that skips embedded XML whitespace
//!   as required by the `xsd:base64Binary` lexical space.
//! * [`hex`] — RFC 4648 base16 (hex), case-insensitive decoding, for `xsd:hexBinary`.
//! * [`percent`] — percent-encoding/decoding parameterized by [`percent::AsciiSet`]s for the
//!   ASCII byte classes an IRI implementation needs (unreserved, path, query, fragment,
//!   userinfo), with non-ASCII bytes always percent-encoded as UTF-8.
//! * [`num`] — `f64`/`f32` to string formatting matching two different specifications:
//!   the ECMAScript `Number::toString(10)` algorithm and the XSD 1.1 canonical mapping for
//!   `double`/`float`.
//! * [`ws`] — the three XML Schema whitespace facets (`preserve` is a no-op; `replace` and
//!   `collapse` are provided).
//!
//! # Example
//!
//! ```
//! use oxixml_text::{base64, hex, num, percent, ws};
//!
//! // Base64: round-trip through the standard alphabet.
//! let encoded = base64::base64_encode(b"OxiXML");
//! assert_eq!(encoded, "T3hpWE1M");
//! assert_eq!(base64::base64_decode(&encoded).expect("valid base64"), b"OxiXML");
//!
//! // Hex: xsd:hexBinary uses uppercase; decoding accepts either case.
//! assert_eq!(hex::encode_upper(&[0xDE, 0xAD, 0xBE, 0xEF]), "DEADBEEF");
//! assert_eq!(hex::decode("deadbeef").expect("valid hex"), vec![0xDE, 0xAD, 0xBE, 0xEF]);
//!
//! // Percent-encoding an IRI path component.
//! assert_eq!(percent::percent_encode(b"a b/c", &percent::PATH_SEGMENT), "a%20b%2Fc");
//!
//! // ECMAScript-style vs. XSD-canonical number formatting diverge on purpose.
//! assert_eq!(num::ecmascript_double(0.1), "0.1");
//! assert_eq!(num::xsd_canonical_double(0.1), "1.0E-1");
//!
//! // Whitespace collapse implements the XSD `collapse` facet.
//! assert_eq!(ws::collapse("  a\t b  \n c  "), "a b c");
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod base64;
pub mod hex;
pub mod num;
pub mod percent;
pub mod ws;

//! `oxixml-encoding` — Character-encoding decoders and encoders for XML processing.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing.
//!
//! This crate provides the character-encoding subset that an XML processor actually needs:
//! it plays the role that [`encoding_rs`](https://docs.rs/encoding_rs) plays in other Rust XML
//! stacks, but is written from scratch against `std` alone, with no external dependencies.
//!
//! # What's here
//!
//! * [`Encoding`] — the core set of encodings an XML parser must recognize (`UTF-8`,
//!   `UTF-16LE`, `UTF-16BE`, `ISO-8859-1`, `windows-1252`, `US-ASCII`), extended with the rest of
//!   the [WHATWG Encoding Standard](https://encoding.spec.whatwg.org/)'s legacy single-byte
//!   encodings, its three legacy Japanese multi-byte encodings (Shift_JIS, EUC-JP, ISO-2022-JP),
//!   and its legacy CJK multi-byte encodings (GBK, gb18030, Big5, EUC-KR) — see
//!   [`Encoding::ALL`] for the complete, up-to-date list — plus [`Encoding::for_label`] to
//!   resolve an IANA/WHATWG charset label (as found in an `encoding="..."` declaration or a
//!   `Content-Type` header) to one of them.
//! * [`Decoder`] — an incremental, chunk-safe decoder: bytes can be fed in arbitrarily small
//!   pieces (even one byte at a time) and a partial multi-byte sequence at a chunk boundary is
//!   carried over correctly. Both a strict mode (which reports a [`DecodeError`] with a byte
//!   offset) and a lossy mode (which substitutes `U+FFFD REPLACEMENT CHARACTER`) are provided.
//! * [`encode_from_utf8`] — the inverse direction, used when *writing* XML in a legacy
//!   encoding; reports the first unmappable character and its position.
//! * [`detect_bom`] — recognizes the three byte-order marks relevant to XML.
//! * [`sniff_xml_encoding`] — implements the heuristic described in
//!   [XML 1.0 Appendix F](https://www.w3.org/TR/xml/#sec-guessing): inspect the first few bytes
//!   of an entity, then (if the bytes look ASCII-compatible or UTF-16) scan the prologue for an
//!   `encoding="..."` pseudo-attribute.
//!
//! # Example
//!
//! ```
//! use oxixml_encoding::{sniff_xml_encoding, Encoding};
//!
//! // A UTF-8 document that self-declares its own encoding.
//! let xml = b"<?xml version=\"1.0\" encoding=\"UTF-8\"?><root/>";
//! let sniffed = sniff_xml_encoding(xml);
//! assert_eq!(sniffed.declared_encoding, Some(Encoding::Utf8));
//!
//! // Decode a chunk of UTF-8 incrementally, one byte at a time, and get back the original text.
//! let mut decoder = Encoding::Utf8.new_decoder();
//! let mut out = String::new();
//! for (i, byte) in "héllo".as_bytes().iter().enumerate() {
//!     let last = i + 1 == "héllo".len();
//!     decoder
//!         .decode_to_string(std::slice::from_ref(byte), &mut out, last)
//!         .expect("valid UTF-8 fed one byte at a time");
//! }
//! assert_eq!(out, "héllo");
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod bom;
mod cjk;
mod decoder;
mod encoder;
mod jis;
mod label;
mod sniff;
mod tables;

pub use bom::detect_bom;
pub use decoder::{DecodeError, DecodeErrorKind, Decoder};
pub use encoder::{EncodeError, encode_from_utf8};
pub use label::Encoding;
pub use sniff::{ByteOrderGuess, SniffResult, sniff_xml_encoding};

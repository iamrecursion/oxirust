//! `oxixml-quickxml-compat` — drop-in [`quick-xml`](https://docs.rs/quick-xml)
//! compatibility shim over [`oxixml-xml`](https://docs.rs/oxixml-xml).
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
//! COOLJAPAN Pure-Rust foundation for all XML and RDF processing.
//!
//! # Usage
//!
//! Downstream crates switch to the Pure-Rust `oxixml-xml` engine with a
//! one-line dependency change, by aliasing this crate's package name to
//! `quick_xml` in their `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! quick_xml = { package = "oxixml-quickxml-compat", version = "0.1.2" }
//! ```
//!
//! Every module path, type name and public method signature below mirrors the
//! `quick-xml` 0.41 public API — including its optional `arbitrary` feature
//! (see `README.md`) — so, with one exception, no source changes are required
//! beyond that `Cargo.toml` edit: everywhere the downstream crate wrote
//! `quick_xml::reader::Reader`, `quick_xml::events::Event` and so on keeps
//! compiling and behaving the same way, backed by [`oxixml_xml`] underneath.
//! The exception is [`Error`], which reports an additional `NotWellFormed`
//! variant beyond quick-xml 0.41's seven: an exhaustive match with no
//! wildcard arm needs one added, a one-time edit that [`Error`] being
//! `#[non_exhaustive]` will not ask for again on a future release.
//!
//! * [`events`] — the [`Event`](events::Event) enum and its `Bytes*` payloads
//!   ([`BytesStart`](events::BytesStart), [`BytesText`](events::BytesText), …),
//!   plus the [`attributes`](events::attributes) iterator.
//! * [`name`] — [`QName`](name::QName), [`LocalName`](name::LocalName),
//!   [`Prefix`](name::Prefix), [`Namespace`](name::Namespace) and the
//!   [`NamespaceResolver`](name::NamespaceResolver).
//! * [`escape`] — [`escape`](escape::escape) / [`unescape`](escape::unescape)
//!   and friends, including attribute-value normalization.
//! * [`encoding`] — the [`Decoder`] (UTF-8 always; non-UTF-8 behind the
//!   `encoding` feature), plus the `DecodingReader` transcoding wrapper that the
//!   `encoding` feature adds.
//! * [`errors`] — [`Error`], [`Result`], [`SyntaxError`](errors::SyntaxError)
//!   and [`IllFormedError`](errors::IllFormedError).
//! * [`reader`] — the pull-based [`Reader`] and its
//!   [`Config`](reader::Config), plus the namespace-aware
//!   [`NsReader`].
//! * [`writer`] — the events-based [`Writer`] and its
//!   [`ElementWriter`] builder.
//! * [`parser`] — the low-level incremental parsers that the [`Reader`] drives
//!   ([`ElementParser`](parser::ElementParser),
//!   [`CommentParser`](parser::CommentParser), [`PiParser`](parser::PiParser)).
//!
//! With the `serde-types` feature enabled, the crate additionally provides the
#![cfg_attr(
    feature = "serde-types",
    doc = "[`serde_helpers`] module (`text_content` and the \
`impl_deserialize_for_internally_tagged_enum!` family of macros)."
)]
#![cfg_attr(
    not(feature = "serde-types"),
    doc = "`serde_helpers` module (`text_content` and the \
`impl_deserialize_for_internally_tagged_enum!` family of macros)."
)]
//!
//! With the `serialize` feature enabled, the crate additionally provides the
#![cfg_attr(
    feature = "serialize",
    doc = "serde deserializer in [`de`] ([`de::from_str`], [`de::from_reader`] and \
[`de::Deserializer`]) and the matching serializer in [`se`] ([`se::to_string`], \
[`se::to_string_with_root`], [`se::to_writer`] and [`se::Serializer`])."
)]
#![cfg_attr(
    not(feature = "serialize"),
    doc = "serde deserializer in `de` (`from_str`, `from_reader` and `Deserializer`) \
and the matching serializer in `se` (`to_string`, `to_string_with_root`, `to_writer` \
and `Serializer`)."
)]
//!
//! # Examples
//!
//! Reading, ported verbatim from the `quick-xml` 0.41 README (only the `use`
//! paths are rewritten to import this crate under the `quick_xml` alias):
//!
//! ```
//! use oxixml_quickxml_compat as quick_xml;
//! use quick_xml::events::Event;
//! use quick_xml::reader::Reader;
//!
//! let xml = r#"<tag1 att1 = "test">
//!                 <tag2><!--Test comment-->Test</tag2>
//!                 <tag2>Test 2</tag2>
//!              </tag1>"#;
//! let mut reader = Reader::from_str(xml);
//! reader.config_mut().trim_text(true);
//!
//! let mut count = 0;
//! let mut txt = Vec::new();
//! let mut buf = Vec::new();
//!
//! loop {
//!     match reader.read_event_into(&mut buf) {
//!         Err(e) => panic!("Error at position {}: {:?}", reader.error_position(), e),
//!         Ok(Event::Eof) => break,
//!         Ok(Event::Start(e)) => {
//!             match e.name().as_ref() {
//!                 b"tag1" => println!(
//!                     "attributes values: {:?}",
//!                     e.attributes().map(|a| a.expect("attr").value).collect::<Vec<_>>()
//!                 ),
//!                 b"tag2" => count += 1,
//!                 _ => (),
//!             }
//!         }
//!         Ok(Event::Text(e)) => txt.push(e.decode().expect("utf-8").into_owned()),
//!         _ => (),
//!     }
//!     buf.clear();
//! }
//! assert_eq!(count, 2);
//! assert_eq!(txt, vec!["Test".to_string(), "Test 2".to_string()]);
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

/// Wrappers that decode byte data into strings.
///
/// Mirrors the `quick_xml::encoding` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::encoding`] at the same path.
pub mod encoding {
    pub use oxixml_xml::encoding::*;
}

/// Error management for the crate.
///
/// Mirrors the `quick_xml::errors` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::errors`] at the same path. With the `serialize` feature
/// enabled, also re-exports the nested `serialize` module
#[cfg_attr(feature = "serialize", doc = "([`DeError`], [`SeError`]).")]
#[cfg_attr(not(feature = "serialize"), doc = "(`DeError`, `SeError`).")]
pub mod errors {
    pub use oxixml_xml::errors::*;
}

/// Manage XML character escapes.
///
/// Mirrors the `quick_xml::escape` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::escape`] at the same path.
pub mod escape {
    pub use oxixml_xml::escape::*;
}

/// Zero-copy XML events used throughout this library.
///
/// Mirrors the `quick_xml::events` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::events`] at the same path, including the nested
/// [`attributes`](events::attributes) module.
pub mod events {
    pub use oxixml_xml::events::*;

    /// Xml attributes iterator.
    ///
    /// Mirrors the `quick_xml::events::attributes` module of quick-xml 0.41;
    /// re-exports [`oxixml_xml::events::attributes`] at the same path.
    pub mod attributes {
        pub use oxixml_xml::events::attributes::*;
    }
}

/// Xml name handling.
///
/// Mirrors the `quick_xml::name` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::name`] at the same path.
pub mod name {
    pub use oxixml_xml::name::*;
}

/// Low level parsers of different XML pieces.
///
/// Mirrors the `quick_xml::parser` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::parser`] at the same path.
pub mod parser {
    pub use oxixml_xml::parser::*;
}

/// High-level interface for the pull-based XML parser.
///
/// Mirrors the `quick_xml::reader` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::reader`] at the same path, including [`Reader`]
/// and [`NsReader`].
pub mod reader {
    pub use oxixml_xml::reader::*;
}

/// High-level interface for an events-based XML emitter.
///
/// Mirrors the `quick_xml::writer` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::writer`] at the same path, including
/// [`Writer`] and [`ElementWriter`].
pub mod writer {
    pub use oxixml_xml::writer::*;
}

/// Serde `Deserializer` module.
///
/// Mirrors the `quick_xml::de` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::de`] at the same path. Enabled by the `serialize` feature.
#[cfg(feature = "serialize")]
#[cfg_attr(docsrs, doc(cfg(feature = "serialize")))]
pub mod de {
    pub use oxixml_xml::de::*;
}

/// Serde `Serializer` module.
///
/// Mirrors the `quick_xml::se` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::se`] at the same path. Enabled by the `serialize` feature.
#[cfg(feature = "serialize")]
#[cfg_attr(docsrs, doc(cfg(feature = "serialize")))]
pub mod se {
    pub use oxixml_xml::se::*;
}

/// Helper functions and macros that glue XML to a serde content model.
///
/// Mirrors the `quick_xml::serde_helpers` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::serde_helpers`] at the same path, so
/// `#[serde(with = "quick_xml::serde_helpers::text_content")]` keeps working.
/// Enabled by the `serde-types` feature.
#[cfg(feature = "serde-types")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde-types")))]
pub mod serde_helpers {
    pub use oxixml_xml::serde_helpers::*;
}

/// Not an official API; exposed for integration tests.
///
/// Mirrors the `quick_xml::utils` module of quick-xml 0.41; re-exports
/// [`oxixml_xml::utils`] at the same path.
#[doc(hidden)]
pub mod utils {
    pub use oxixml_xml::utils::*;
}

// Top-level re-exports mirroring quick-xml 0.41.
pub use oxixml_xml::Decoder;
#[cfg(feature = "serialize")]
#[cfg_attr(docsrs, doc(cfg(feature = "serialize")))]
pub use oxixml_xml::{DeError, SeError};
pub use oxixml_xml::{ElementWriter, XmlVersion};
pub use oxixml_xml::{Error, NsReader, Reader, Result, Writer};
// The macros of `serde_helpers`, which quick-xml exports at the crate root.
#[cfg(feature = "serde-types")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde-types")))]
pub use oxixml_xml::{
    deserialize_match, deserialize_variant, impl_deserialize_for_internally_tagged_enum,
};

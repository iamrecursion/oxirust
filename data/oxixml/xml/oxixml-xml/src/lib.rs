//! `oxixml-xml` — XML 1.0/1.1 pull parser and writer, quick-xml 0.41-compatible.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
//! COOLJAPAN Pure-Rust foundation for all XML and RDF processing. This crate is a
//! drop-in replacement for [`quick-xml`](https://docs.rs/quick-xml): it mirrors
//! the 0.41 public API (module paths, type names and method signatures) so that
//! downstream projects can switch over with a one-line dependency change.
//!
//! It provides the zero-copy event model, its supporting vocabulary, and the
//! pull-based [`Reader`]:
//!
//! * [`events`] — the [`Event`](events::Event) enum and its `Bytes*` payloads
//!   ([`BytesStart`](events::BytesStart), [`BytesText`](events::BytesText), …),
//!   plus the [`attributes`](events::attributes) iterator.
//! * [`name`] — [`QName`](name::QName), [`LocalName`](name::LocalName),
//!   [`Prefix`](name::Prefix), [`Namespace`](name::Namespace) and the
//!   [`NamespaceResolver`](name::NamespaceResolver).
//! * [`escape`] — [`escape`](escape::escape) / [`unescape`](escape::unescape) and
//!   friends, including attribute-value normalization.
//! * [`encoding`] — the [`Decoder`] (UTF-8 always; non-UTF-8 behind the
//!   `encoding` feature, backed by `oxixml-encoding`).
//! * [`errors`] — [`Error`], [`Result`], [`SyntaxError`](errors::SyntaxError) and
//!   [`IllFormedError`](errors::IllFormedError).
//! * [`reader`] — the pull-based [`Reader`] and its
//!   [`Config`](reader::Config), plus the namespace-aware
//!   [`NsReader`] that resolves prefixes to namespaces.
//! * [`writer`] — the events-based [`Writer`] and its
//!   [`ElementWriter`] builder.
//! * [`parser`] — the low-level incremental parsers that the [`Reader`] drives
//!   ([`ElementParser`](parser::ElementParser), [`CommentParser`](parser::CommentParser),
//!   [`PiParser`](parser::PiParser)).
//!
//! # Conformance
//!
//! Parsing is `quick-xml`-compatible — that is, deliberately permissive — by
//! default. Setting [`Config::check_well_formedness`](reader::Config::check_well_formedness)
//! (or [`Config::enable_all_checks`](reader::Config::enable_all_checks)) turns the
//! reader into a conforming **non-validating** XML processor: the `Char` and
//! `Name` productions, the document structure, the grammar of the XML declaration
//! and of the internal DTD subset, and the entity-related well-formedness
//! constraints are then all enforced, and violations surface as
//! [`Error::NotWellFormed`]. Driven through an
//! [`NsReader`], the extra `QName` and prefix-binding
//! constraints of Namespaces in XML are enforced as well.
//!
//! ```
//! use oxixml_xml::reader::Reader;
//!
//! let mut reader = Reader::from_str("<a/><b/>");
//! reader.config_mut().check_well_formedness = true;
//! let _ = reader.read_event().expect("first root element");
//! assert!(reader.read_event().is_err(), "a document has a single root element");
//! ```
//!
//! With the `serialize` feature enabled, the crate additionally provides the
#![cfg_attr(
    feature = "serialize",
    doc = "serde deserializer in [`de`] ([`de::from_str`], [`de::from_reader`] and \
[`de::Deserializer`]) and the matching serializer in [`se`] ([`se::to_string`], \
[`se::to_string_with_root`], [`se::to_writer`] and [`se::Serializer`]),"
)]
#![cfg_attr(
    not(feature = "serialize"),
    doc = "serde deserializer in `de` (`from_str`, `from_reader` and `Deserializer`) \
and the matching serializer in `se` (`to_string`, `to_string_with_root`, `to_writer` \
and `Serializer`),"
)]
//! which mirror the quick-xml 0.41 mapping rules
//! (`@field` for attributes, `$text` and `$value` special names, sequences of
//! consecutive elements, and so on).
//!
//! # Example
//!
//! ```
//! use oxixml_xml::escape::{escape, unescape};
//! use oxixml_xml::events::{BytesStart, Event};
//! use oxixml_xml::name::QName;
//!
//! // Build a start tag and inspect its name and attributes.
//! let start = BytesStart::new("greeting").with_attributes([("lang", "en")]);
//! assert_eq!(start.name(), QName(b"greeting"));
//! let (key, value) = {
//!     let attr = start.attributes().next().expect("one attribute").expect("valid");
//!     (attr.key.as_ref().to_vec(), attr.value.to_vec())
//! };
//! assert_eq!((key.as_slice(), value.as_slice()), (&b"lang"[..], &b"en"[..]));
//!
//! // Wrap it in an event; events borrow their payload bytes.
//! let event = Event::Start(start);
//! assert_eq!(&*event, b"greeting lang=\"en\"".as_ref());
//!
//! // Escape and unescape round-trip.
//! assert_eq!(escape("a < b & c"), "a &lt; b &amp; c");
//! assert_eq!(unescape("a &lt; b &amp; c").expect("valid"), "a < b & c");
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

#[cfg(feature = "serialize")]
#[cfg_attr(docsrs, doc(cfg(feature = "serialize")))]
pub mod de;
pub mod encoding;
pub mod errors;
pub mod escape;
pub mod events;
pub mod name;
pub mod parser;
pub mod reader;
#[cfg(feature = "serialize")]
#[cfg_attr(docsrs, doc(cfg(feature = "serialize")))]
pub mod se;
#[cfg(feature = "serde-types")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde-types")))]
pub mod serde_helpers;
/// Not an official API; exposed for integration tests.
#[doc(hidden)]
pub mod utils;
mod wellformed;
pub mod writer;

use std::borrow::Cow;

// Re-exports mirroring quick-xml 0.41.
pub use crate::encoding::Decoder;
#[cfg(feature = "serialize")]
#[cfg_attr(docsrs, doc(cfg(feature = "serialize")))]
pub use crate::errors::serialize::{DeError, SeError};
pub use crate::errors::{Error, Result};
pub use crate::reader::{NsReader, Reader};
pub use crate::writer::{ElementWriter, Writer};

/// Version of the XML standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XmlVersion {
    /// No XML declaration was present, so [version 1.0] is assumed (the default).
    ///
    /// [version 1.0]: https://www.w3.org/TR/xml/
    Implicit1_0,
    /// The XML version was explicitly declared as [`1.0`](https://www.w3.org/TR/xml/).
    Explicit1_0,
    /// The XML version was explicitly declared as [`1.1`](https://www.w3.org/TR/xml11/).
    Explicit1_1,
}

impl XmlVersion {
    /// Normalizes an attribute value according to the rules of this XML version.
    pub(crate) fn normalize_attribute_value<'input, 'entity, F>(
        &self,
        value: &'input str,
        depth: usize,
        resolve_entity: F,
    ) -> std::result::Result<Cow<'input, str>, escape::EscapeError>
    where
        // The lifetime of the output comes from a capture or is `'static`.
        F: FnMut(&str) -> Option<&'entity str>,
    {
        match self {
            Self::Explicit1_1 => {
                escape::normalize_xml11_attribute_value(value, depth, resolve_entity)
            }
            _ => escape::normalize_xml10_attribute_value(value, depth, resolve_entity),
        }
    }
}

impl Default for XmlVersion {
    #[inline]
    fn default() -> Self {
        Self::Implicit1_0
    }
}

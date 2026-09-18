//! `oxixml-qname` — XML names, NCName/QName, and namespace-URI handling.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate provides the name-shaped
//! vocabulary that every other layer (SAX/DOM events, the schema validators, XPath/XQuery,
//! canonicalization) builds identity on top of:
//!
//! * [`is_name`], [`is_ncname`], [`is_nmtoken`], [`is_qname`] — the boolean character-class
//!   predicates from [Namespaces in XML 1.0/1.1](https://www.w3.org/TR/xml-names/), built
//!   directly on [`oxixml_unicode::xml`]'s hand-written `NameStartChar`/`NameChar` productions
//!   (which are themselves shared, byte-for-byte, between XML 1.0 Fifth Edition and XML 1.1).
//! * [`QName`] — a *lexical*, unresolved qualified name (`prefix:local` or just `local`),
//!   generic over its text storage the same way the sibling `oxixml-iri` crate's `Iri<T>` is
//!   generic over its: pass `QName<&str>` to borrow zero-copy from a buffer you already own, or
//!   `QName<String>` (via
//!   [`FromStr`](std::str::FromStr)) to own one. Either way there is a single allocation (or
//!   none, when borrowing) — `QName` remembers only the byte offset of the separating colon
//!   rather than splitting into two owned buffers.
//! * [`ExpandedName`] / [`ExpandedNameRef`] — the *resolved* namespace-URI-plus-local-name pair
//!   that is the universal name identity once a `QName` has been resolved against a namespace
//!   context. [`Eq`]/[`std::hash::Hash`] are defined on `(namespace, local)`, independent of
//!   whichever prefix happened to be used to spell the name in a particular document. Either
//!   type can additionally *retain* one such prefix as a pure annotation — invisible to
//!   [`Eq`]/[`std::hash::Hash`] — via `with_prefix`/`prefix`, for callers (e.g. `fn:name()` on a
//!   computed constructor) that need to recover it later.
//! * [`NamespaceStack`] — the scoped `prefix -> namespace-name` binding table a streaming parser
//!   pushes and pops one element at a time, implementing the reserved-prefix rules and the
//!   default-vs-no-namespace distinction of Namespaces in XML 1.0/1.1 §3 exactly, including the
//!   1.1-only "undeclare with an empty URI" relaxation.
//! * [`XML_NAMESPACE`] / [`XMLNS_NAMESPACE`] — the two namespace names fixed by the
//!   specification itself.
//!
//! # Example
//!
//! ```
//! use oxixml_qname::{is_ncname, is_qname, NameContext, NamespaceStack, NamespacesVersion, QName};
//!
//! // Character-class predicates.
//! assert!(is_ncname("book"));
//! assert!(!is_ncname("book:title")); // a colon is not allowed inside an NCName
//! assert!(is_qname("book:title"));
//!
//! // Lexical QName parsing: zero-copy over a borrowed `&str`...
//! let qname = QName::parse("book:title")?;
//! assert_eq!(qname.prefix(), Some("book"));
//! assert_eq!(qname.local_part(), "title");
//!
//! // ...or owned, via `FromStr`.
//! let owned: QName<String> = "title".parse()?;
//! assert_eq!(owned.prefix(), None);
//!
//! // A namespace stack resolves prefixes in scope, one element at a time.
//! let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_1);
//! namespaces.push_scope();
//! namespaces.declare(Some("book"), "urn:example:book")?;
//! let element_name = namespaces.resolve_qname("book:title", NameContext::Element)?;
//! assert_eq!(element_name.namespace(), Some("urn:example:book"));
//!
//! // Unprefixed *attributes* never inherit the default namespace (unlike elements).
//! namespaces.declare(None, "urn:example:default")?;
//! let attr_name = namespaces.resolve_qname("id", NameContext::Attribute)?;
//! assert_eq!(attr_name.namespace(), None);
//! namespaces.pop_scope();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod constants;
mod error;
mod expanded_name;
mod namespace_stack;
mod qname;
mod validate;

pub use constants::{XML_NAMESPACE, XMLNS_NAMESPACE};
pub use error::{NamespaceDeclError, QNameParseError, ResolveQNameError};
pub use expanded_name::{ExpandedName, ExpandedNameRef};
pub use namespace_stack::{NameContext, NamespaceStack, NamespacesVersion};
pub use qname::QName;
pub use validate::{is_name, is_ncname, is_nmtoken, is_qname};

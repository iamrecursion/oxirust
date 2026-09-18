//! `oxixml-xinclude` — an [XInclude 1.0 (Second Edition)](https://www.w3.org/TR/xinclude/)
//! processor over [`oxixml-dom`](oxixml_dom).
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing.
//!
//! # What this crate does
//!
//! [`process_document`] walks an already-parsed [`Document`](oxixml_dom::Document) and replaces
//! every `xi:include` element it finds with the content it points to, in place:
//!
//! * **`parse="xml"` (the default)** — the referenced resource is parsed as XML, recursively
//!   XInclude-processed itself, optionally narrowed down with the `xpointer` attribute (the
//!   `element()` scheme, both its shorthand-`ID` and child-sequence forms — see the [`xpointer`
//!   module docs](https://www.w3.org/TR/xptr-element/) — plus minimal support for `xmlns()` and
//!   `xpointer(id(...))`), and grafted into the tree with base-URI and `xml:lang`
//!   [fixup](https://www.w3.org/TR/xinclude/#base_uri_fixup) applied so that the moved content
//!   keeps resolving relative references and inheriting language exactly as it did in its
//!   original document.
//! * **`parse="text"`** — the resource is decoded (BOM, then the `encoding` attribute, then
//!   UTF-8, via [`oxixml-encoding`](oxixml_encoding)) and spliced in as a single text node.
//! * **`xi:fallback`** — a resource error (the resource cannot be fetched, or an `xpointer`
//!   selects nothing) runs the sibling `xi:fallback` element's content instead of failing the
//!   whole document; a *fatal* error (a malformed `xi:include`, non-well-formed included XML, an
//!   inclusion loop, …) still aborts processing, since XInclude 1.0 draws that line itself — see
//!   [`XIncludeError`].
//! * **Loop detection** — an inclusion chain may never revisit the same `(location, xpointer)`
//!   pair, per [XInclude 1.0 §4.2.7](https://www.w3.org/TR/xinclude/#loops).
//! * **`href=""` self-reference** — an empty (or absent) `href` refers to the including
//!   document itself, exactly as the specification defines it.
//!
//! Resource fetching is entirely pluggable through the [`ResourceLoader`] trait: this crate
//! never touches the filesystem or the network on its own. [`NullLoader`] (the safe, SSRF-proof
//! default posture) resolves nothing; [`MapLoader`] serves a fixed, in-memory table, handy for
//! tests and for embedding a small trusted resource set.
//!
//! # Example
//!
//! ```
//! use oxixml_dom::build;
//! use oxixml_xinclude::{MapLoader, process_document};
//!
//! let mut doc = build::parse_str(
//!     r#"<doc xmlns:xi="http://www.w3.org/2001/XInclude">
//!          <xi:include href="https://example.com/greeting.xml"/>
//!        </doc>"#,
//! )?;
//!
//! let loader =
//!     MapLoader::new().with_text("https://example.com/greeting.xml", "<p>hello</p>");
//! process_document(&mut doc, "https://example.com/index.xml", &loader)?;
//!
//! let root = doc.document_element().expect("document element");
//! let grafted = doc
//!     .get_elements_by_tag_name(root, "p")
//!     .next()
//!     .expect("the xi:include was replaced by the fetched <p>");
//! assert_eq!(doc.text_content(grafted).as_deref(), Some("hello"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod error;
mod fixup;
mod loader;
mod names;
mod options;
mod process;
mod xpointer;

pub use error::XIncludeError;
pub use loader::{LoadError, MapLoader, NullLoader, ResourceLoader};
pub use names::XINCLUDE_NAMESPACE;
pub use options::XIncludeOptions;
pub use process::{process_document, process_document_with};

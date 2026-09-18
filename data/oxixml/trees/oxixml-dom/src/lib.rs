//! `oxixml-dom` — a DOM Level 3 Core-class document tree over an arena.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate is the in-memory document
//! model that canonicalization, XSLT, schema validation and XInclude all build on.
//!
//! # Design
//!
//! A [`Document`] owns one `Vec` of nodes and hands out [`NodeId`] handles into it. Parents,
//! children and siblings are plain indices, so:
//!
//! * there are no reference cycles and no reference counting — dropping a document is a single
//!   `Vec` deallocation, and no recursion;
//! * a node identifier stays valid for the lifetime of the document, even across mutations —
//!   [`Document::remove_child`] *detaches* a node rather than destroying it, so the returned
//!   identifier can be re-inserted somewhere else;
//! * there is no `unsafe` code anywhere, and the whole tree is `Send`.
//!
//! Attribute nodes live in the same arena but are not part of the child tree: they hang off
//! their element's attribute list, exactly as
//! [DOM Level 3 Core](https://www.w3.org/TR/DOM-Level-3-Core/) specifies. Namespace
//! declarations are *ordinary attributes* in the `http://www.w3.org/2000/xmlns/` namespace, so
//! nothing has to be kept in sync and a parsed document re-serializes its declarations
//! verbatim.
//!
//! # What is here
//!
//! * **Node factories and mutation** — [`Document::create_element_ns`],
//!   [`Document::append_child`], [`Document::insert_before`], [`Document::replace_child`],
//!   [`Document::remove_child`], [`Document::set_attribute_ns`], [`Document::clone_node`],
//!   [`Document::import_node`] and [`Document::normalize`], each performing the
//!   well-formedness checks of DOM Core and reporting them as a [`DomError`] named after the
//!   corresponding `DOMException`.
//! * **Navigation** — [`Document::parent`], [`Document::children`], [`Document::attributes`],
//!   [`Document::ancestors`], [`Document::descendants`], and the XPath [`Document::following`]
//!   and [`Document::preceding`] axes, plus a lazily maintained
//!   [document-order index](Document::document_order).
//! * **Lookups** — [`Document::get_element_by_id`] (honouring `xml:id` and `<!ATTLIST … ID …>`
//!   declarations), [`Document::get_elements_by_tag_name`] and
//!   [`Document::get_elements_by_tag_name_ns`].
//! * **Building** — [`build::parse_str`] and [`build::parse_reader`] turn an
//!   [`oxixml_xml`] event stream into a tree, resolving namespaces with `NsReader` semantics;
//!   [`build::DocumentBuilder`] exposes the same state machine to any other event source.
//! * **Serializing** — [`Document::to_xml`], [`Document::write_xml`] and
//!   [`Document::write_xml_io`], with correct escaping, empty-element folding, optional pretty
//!   indentation and DOM namespace fixup.
//!
//! # Example
//!
//! ```
//! use oxixml_dom::{Document, NodeKind, SerializeOptions, build};
//!
//! // Build a tree from XML …
//! let mut doc = build::parse_str(
//!     r#"<library xmlns="urn:example:library">
//!          <book id="b1"><title>Kokoro</title></book>
//!          <book id="b2"><title>Rashomon</title></book>
//!        </library>"#,
//! )?;
//!
//! let library = doc.document_element().expect("a document element");
//! let titles: Vec<String> = doc
//!     .get_elements_by_tag_name_ns(library, Some("urn:example:library"), "title")
//!     .filter_map(|node| doc.text_content(node))
//!     .collect();
//! assert_eq!(titles, ["Kokoro", "Rashomon"]);
//!
//! // … mutate it …
//! let book = doc.create_element_ns(Some("urn:example:library"), "book")?;
//! doc.set_attribute(book, "id", "b3")?;
//! doc.append_child(library, book)?;
//! assert_eq!(doc.get_element_by_id("b3"), None); // `id` is not an ID without a declaration
//! doc.declare_id_attribute(Some("book"), "id");
//! assert_eq!(doc.get_element_by_id("b3"), Some(book));
//!
//! // … and write it back out.
//! let xml = doc.to_xml_with(&SerializeOptions::pretty());
//! assert!(xml.contains("<book id=\"b3\"/>"));
//!
//! // A round-trip through the serializer preserves the tree.
//! let reparsed = build::parse_str(&doc.to_xml())?;
//! assert!(doc.structural_eq(&reparsed), "{:?}", doc.structural_diff(&reparsed));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod build;
mod document;
mod dtd;
mod entity;
mod error;
pub mod external;
pub mod iter;
mod names;
mod node;
mod serialize;

pub use build::{BuilderOptions, DocumentBuilder, EntityPolicy};
pub use document::Document;
pub use error::{
    BuildError, DomError, DomResult, HierarchyRequest, NamespaceViolation, NotFound, SerializeError,
};
pub use names::{
    XML_NAMESPACE, XMLNS_NAMESPACE, is_xml_char, is_xml11_char, is_xml11_restricted_char,
};
pub use node::{
    AttrData, DocumentTypeData, ElementData, NodeId, NodeKind, ProcessingInstructionData,
};
pub use serialize::{SerializeOptions, XmlDeclaration};

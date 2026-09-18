//! `oxixml-infoset` — the XPath and XQuery Data Model (XDM), in Pure Rust.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate is the typed abstraction that
//! [XPath 3.1](https://www.w3.org/TR/xpath-31/), [XQuery 3.1](https://www.w3.org/TR/xquery-31/),
//! [XSLT 3.0](https://www.w3.org/TR/xslt-30/) and the schema validators all evaluate over: what
//! Xalan and Saxon call the *node tree and sequence model*, specified by
//! [XQuery and XPath Data Model 3.1](https://www.w3.org/TR/xpath-datamodel-31/).
//!
//! # The two halves of the model
//!
//! **Nodes.** [`NodeKind`] enumerates the seven node kinds. [`XdmNode`] is the accessor
//! interface an engine is written against — [`node_kind`](XdmNode::node_kind),
//! [`node_name`](XdmNode::node_name), [`string_value`](XdmNode::string_value),
//! [`typed_value`](XdmNode::typed_value), [`parent`](XdmNode::parent),
//! [`children`](XdmNode::children), [`attributes`](XdmNode::attributes),
//! [`namespaces`](XdmNode::namespaces), [`base_uri`](XdmNode::base_uri),
//! [`document_order_key`](XdmNode::document_order_key), [`is_id`](XdmNode::is_id) and
//! [`is_idref`](XdmNode::is_idref). It is implemented for [`NodeRef`], a cheap handle that
//! forwards to a pluggable [`XdmStore`] backend, so the same navigation code drives the
//! in-memory [`Tree`] this crate ships, a DOM, or a future streaming window without changing a
//! signature.
//!
//! **Values.** [`AtomicValue`] carries an XML Schema value from [`oxixml_xsd`] together with its
//! [`AtomicType`] label and the language and collation in force where it was produced. [`Item`]
//! is a node, an atomic value, an [`XdmMap`] or an [`XdmArray`]; [`Sequence`] is a flat, ordered
//! collection of items that *structurally* cannot nest, which is the central axiom of the model.
//!
//! # Navigation
//!
//! All thirteen [`Axis`] variants are computed over any [`XdmNode`] by [`axis_nodes`], each in
//! its own order — document order for the eight forward axes, reverse document order for the
//! five reverse ones — with no duplicates. [`compare_document_order`] and
//! [`sort_document_order`] provide the normalization every path expression applies, and
//! [`deep_equal`] implements `fn:deep-equal`.
//!
//! # Example
//!
//! ```
//! use oxixml_infoset::{
//!     Axis, NodeKind, Sequence, Tree, XdmNode, axis_nodes, deep_equal, sort_document_order,
//! };
//!
//! let tree = Tree::parse_str(
//!     r#"<catalog xmlns:dc="urn:dc"><book dc:id="b1">Dune</book><book>Emma</book></catalog>"#,
//! )?;
//!
//! // Navigate: the document element, its children, an attribute.
//! let catalog = tree.document_element().expect("a document element");
//! assert_eq!(catalog.node_kind(), NodeKind::Element);
//! assert_eq!(catalog.string_value(), "DuneEmma");
//!
//! let books = axis_nodes(&catalog, Axis::Child);
//! assert_eq!(books.len(), 2);
//!
//! let id = books[0].attributes().first().cloned().expect("one attribute");
//! assert_eq!(id.node_name().map(|name| name.to_string()).as_deref(), Some("{urn:dc}id"));
//! assert_eq!(id.string_value(), "b1");
//!
//! // Atomize: an unvalidated attribute yields one xs:untypedAtomic value.
//! let atomized = id.typed_value()?;
//! assert_eq!(atomized.len(), 1);
//! assert!(atomized.effective_boolean_value()?);
//!
//! // Reverse axes come back in reverse document order.
//! let ancestors = axis_nodes(&books[1], Axis::Ancestor);
//! assert_eq!(ancestors, vec![catalog.clone(), tree.root()]);
//!
//! // Node sets normalize to document order without duplicates.
//! let mut jumbled = vec![books[1].clone(), books[0].clone(), books[1].clone()];
//! sort_document_order(&mut jumbled);
//! assert_eq!(jumbled, books);
//!
//! // fn:deep-equal ignores comments and attribute order.
//! let other = Tree::parse_str("<book>Emma</book>")?;
//! assert!(deep_equal(
//!     &Sequence::from(books[1].clone()),
//!     &Sequence::from(other.document_element().expect("a document element")),
//! )?);
//! # Ok::<(), oxixml_infoset::XdmError>(())
//! ```
//!
//! # Scope
//!
//! This crate models *data*, never evaluation: there is no expression tree, no dynamic context
//! and no function library here. It also stops at the built-in types — a schema-aware processor
//! validates against its own type registry and hands the results down through [`Annotation`],
//! which is how a typed value, a type name, `nilled`, `is-id` and `is-idrefs` reach a node.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod array;
mod atomic;
mod axis;
mod deep_equal;
mod error;
mod function;
mod item;
mod map;
mod node;
mod order;
mod sequence;
mod tree;

pub use array::XdmArray;
pub use atomic::{
    AtomicData, AtomicType, AtomicValue, BigDecimal, BigInteger, CODEPOINT_COLLATION,
    HTML_ASCII_CASE_INSENSITIVE_COLLATION, QNameValue, XSD_NAMESPACE, atomic_equal,
    atomic_equal_in_context, atomic_equal_with_collation, compare_atomic,
    compare_atomic_in_context, compare_atomic_with_collation, compare_strings, same_key,
    same_key_hash,
};
pub use axis::{
    Axis, ancestor_or_self, ancestors, axis_nodes, descendant_or_self, descendants, following,
    following_siblings, preceding, preceding_siblings,
};
pub use deep_equal::{deep_equal, deep_equal_items, deep_equal_with_collation};
pub use error::{ErrorCode, XdmError};
pub use function::{FunctionBody, XdmFunction};
pub use item::Item;
pub use map::{DuplicateKeys, XdmMap};
pub use node::{DocumentOrderKey, NodeId, NodeKind, NodeRef, XdmNode, XdmStore};
pub use order::{
    compare_document_order, deduplicate_nodes, is_ancestor_of, is_descendant_of,
    sort_document_order,
};
pub use sequence::Sequence;
/// Optional retrieval of external parsed general entities: the resolver hook
/// [`Tree::parse_str_with_resolver`] takes, and the file-backed resolver most callers want.
///
/// Every item is also re-exported at the crate root, so
/// `use oxixml_infoset::ExternalEntityResolver;` and
/// `use oxixml_infoset::external::ExternalEntityResolver;` name the same trait.
pub use tree::external;
pub use tree::{
    Annotation, DEFAULT_MAX_ENTITY_BYTES, EXTERNAL_SUBSET_NAME, ExternalEntity,
    ExternalEntityError, ExternalEntityResolver, FileEntityResolver, Notation, ParseOptions,
    SourcePosition, Tree, TreeBuilder, UnparsedEntity, next_tree_id,
};

/// Re-export of [`oxixml_qname::ExpandedName`], the `(namespace, local name)` pair that is a
/// node's name and an `xs:QName`'s value.
pub use oxixml_qname::ExpandedName;
/// Re-export of [`oxixml_qname::ExpandedNameRef`], the borrowed form of [`ExpandedName`].
pub use oxixml_qname::ExpandedNameRef;
/// Re-export of [`oxixml_qname::NamespacesVersion`], which selects the namespace rules
/// [`ParseOptions`] applies.
pub use oxixml_qname::NamespacesVersion;

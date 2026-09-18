//! The node half of the data model: [`NodeKind`], the pluggable [`XdmStore`] backend trait, the
//! [`NodeRef`] handle, and the [`XdmNode`] accessor trait implemented for it.

use crate::error::XdmError;
use crate::sequence::Sequence;
use crate::tree::{Notation, SourcePosition, UnparsedEntity};
use oxixml_qname::ExpandedName;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The seven kinds of node in the XDM.
///
/// [XDM 3.1 §6](https://www.w3.org/TR/xpath-datamodel-31/#Node) defines exactly these; there is
/// no "entity reference", "CDATA section" or "document type" node — those Infoset and DOM
/// concepts are resolved away before a tree reaches this layer.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::NodeKind;
///
/// // The kind test name used in an XPath node test.
/// assert_eq!(NodeKind::ProcessingInstruction.kind_test_name(), "processing-instruction");
/// // Attribute and namespace nodes are not children of anything.
/// assert!(!NodeKind::Attribute.is_child_kind());
/// assert!(NodeKind::Text.is_child_kind());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NodeKind {
    /// The root of a tree that represents a whole XML document.
    Document,
    /// An element.
    Element,
    /// An attribute. Attributes are *not* children of their element.
    Attribute,
    /// Character data. Adjacent text nodes never occur in a well-formed data-model instance.
    Text,
    /// A comment.
    Comment,
    /// A processing instruction. Its node name is the target; its string value is the content.
    ProcessingInstruction,
    /// An in-scope namespace binding of an element. Namespace nodes are *not* children of their
    /// element.
    Namespace,
}

impl NodeKind {
    /// The name this kind is spelled with in an XPath kind test, without the parentheses.
    #[must_use]
    pub const fn kind_test_name(self) -> &'static str {
        match self {
            Self::Document => "document-node",
            Self::Element => "element",
            Self::Attribute => "attribute",
            Self::Text => "text",
            Self::Comment => "comment",
            Self::ProcessingInstruction => "processing-instruction",
            Self::Namespace => "namespace-node",
        }
    }

    /// Whether nodes of this kind appear on the `child` axis of their parent.
    ///
    /// Attribute and namespace nodes have a parent but are not their parent's children — they
    /// live on the `attribute` and `namespace` axes instead.
    #[must_use]
    pub const fn is_child_kind(self) -> bool {
        !matches!(self, Self::Attribute | Self::Namespace | Self::Document)
    }
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.kind_test_name())
    }
}

/// A backend-defined identifier for one node inside one [`XdmStore`].
///
/// The numeric value is meaningful only to the store that issued it; never compare `NodeId`s
/// that came from different stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u64);

/// A total order on nodes that realizes [document order](https://www.w3.org/TR/xpath-datamodel-31/#document-order).
///
/// Within a tree, `position` increases through the document-order traversal: an element, then
/// its namespace nodes, then its attribute nodes, then its children, recursively. Across trees,
/// `tree` provides the "stable but implementation-dependent" ordering the specification allows
/// between nodes in different trees.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{DocumentOrderKey, Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a><b/><c/></a>")?;
/// let root = tree.root();
/// let element = root.children().first().cloned().expect("the document element");
/// assert!(root.document_order_key() < element.document_order_key());
///
/// // Keys are comparable across trees, and the order is stable within a process.
/// let other = Tree::parse_str("<z/>")?;
/// let first: DocumentOrderKey = tree.root().document_order_key();
/// let second = other.root().document_order_key();
/// assert_eq!(first < second, first < second);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DocumentOrderKey {
    /// Identifies the tree the node belongs to. Distinct trees always have distinct values.
    pub tree: u64,
    /// The node's position in its tree's document-order traversal.
    pub position: u64,
}

/// A pluggable backend that stores an XDM tree.
///
/// This is the trait a *tree representation* implements; consumers navigate through the
/// [`NodeRef`] handle and the [`XdmNode`] trait instead. Splitting the two lets a DOM, a
/// streaming window, a lazily-materialized database cursor, or the in-memory [`Tree`](crate::Tree)
/// this crate ships all present the same node interface without any of them appearing in the
/// signatures the query engine is written against.
///
/// # Invariants an implementation must uphold
///
/// * [`document_order_key`](XdmStore::document_order_key) is injective within the store, and
///   its `tree` component is unique to this store's trees process-wide (use
///   [`next_tree_id`](crate::next_tree_id)).
/// * The key order is exactly document order: a node precedes its namespace nodes, which
///   precede its attribute nodes, which precede its children.
/// * [`parent`](XdmStore::parent) is the inverse of
///   [`children`](XdmStore::children)/[`attributes`](XdmStore::attributes)/[`namespaces`](XdmStore::namespaces).
/// * [`children`](XdmStore::children) never contains two adjacent text nodes, and never
///   contains an attribute or namespace node.
///
/// # Examples
///
/// See the source of [`Tree`](crate::Tree) for a complete implementation; every method is
/// mechanical once the arena is laid out in document order.
pub trait XdmStore: fmt::Debug + Send + Sync {
    /// The node's kind.
    fn node_kind(&self, node: NodeId) -> NodeKind;

    /// The node's name, or `None` for the kinds that have none.
    ///
    /// A store that knows the prefix the name was *written* with must retain it on the
    /// [`ExpandedName`] (see [`XdmNode::node_name`]); the prefix takes no part in equality, so
    /// carrying it costs the name tests nothing.
    fn node_name(&self, node: NodeId) -> Option<ExpandedName>;

    /// The node's string value.
    fn string_value(&self, node: NodeId) -> String;

    /// The node's typed value.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foty0012`](crate::ErrorCode::Foty0012) when the node has no typed value.
    fn typed_value(&self, node: NodeId) -> Result<Sequence, XdmError>;

    /// The node's type annotation, or `None` when the store is not schema-aware.
    fn type_name(&self, node: NodeId) -> Option<ExpandedName>;

    /// Whether the node is a nilled element, or `None` for kinds where the property does not
    /// apply.
    fn nilled(&self, node: NodeId) -> Option<bool>;

    /// The node's parent, or `None` for the root of a tree.
    fn parent(&self, node: NodeId) -> Option<NodeId>;

    /// The node's children, in document order. Never includes attribute or namespace nodes.
    fn children(&self, node: NodeId) -> Vec<NodeId>;

    /// The node's attribute nodes, in document order. Empty for every kind but `Element`.
    fn attributes(&self, node: NodeId) -> Vec<NodeId>;

    /// The node's namespace nodes, in document order. Empty for every kind but `Element`.
    fn namespaces(&self, node: NodeId) -> Vec<NodeId>;

    /// The node's base URI, per [XDM 3.1 §5.2](https://www.w3.org/TR/xpath-datamodel-31/#dm-base-uri).
    fn base_uri(&self, node: NodeId) -> Option<String>;

    /// The absolute URI of the document the node belongs to, if it has one.
    fn document_uri(&self, node: NodeId) -> Option<String>;

    /// The node's position in document order.
    fn document_order_key(&self, node: NodeId) -> DocumentOrderKey;

    /// Whether this node is an ID, per [XDM 3.1 §5.6](https://www.w3.org/TR/xpath-datamodel-31/#dm-is-id).
    fn is_id(&self, node: NodeId) -> bool;

    /// Whether this node is an IDREF or IDREFS, per
    /// [XDM 3.1 §5.7](https://www.w3.org/TR/xpath-datamodel-31/#dm-is-idrefs).
    fn is_idref(&self, node: NodeId) -> bool;

    /// Whether a text node's content must be serialized *without* escaping.
    ///
    /// This is the data-model side of XSLT's
    /// [`disable-output-escaping`](https://www.w3.org/TR/xslt-30/#disable-output-escaping)
    /// attribute: a store that materializes the result of a transformation records the flag on
    /// the text nodes it builds so that a later copy, a temporary tree, or the serializer can
    /// still tell that `&lt;` was meant to reach the output as `<`. Every other store — and
    /// every non-text node — answers `false`, which is why the default implementation says so
    /// and no existing implementor has to change.
    ///
    /// Marked text is not otherwise special: it compares, sorts, and reports its string value
    /// exactly like any other text node. Only serialization consults the flag, and a serializer
    /// that cannot honour it (one writing to a data model rather than to characters) is free to
    /// ignore it.
    fn is_unescaped_text(&self, _node: NodeId) -> bool {
        false
    }

    /// The root of the tree `node` belongs to.
    ///
    /// The default implementation walks the parent chain, which is correct for every store;
    /// override it if the root is directly available.
    fn root(&self, node: NodeId) -> NodeId {
        let mut current = node;
        while let Some(parent) = self.parent(current) {
            current = parent;
        }
        current
    }

    /// The unparsed (`NDATA`) general entity `name`, when `node` is a document node whose DTD
    /// declared one.
    ///
    /// [XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode) makes
    /// `unparsed-entities` an accessor of the *document* node, so an implementation returns
    /// `None` for every other kind of node — which is what lets `fn:unparsed-entity-uri`
    /// distinguish "no such entity" from "not rooted at a document".
    ///
    /// The default implementation reports a store that retains no DTD.
    fn unparsed_entity(&self, node: NodeId, name: &str) -> Option<UnparsedEntity> {
        let _ = (node, name);
        None
    }

    /// The names of every unparsed entity `node` carries, in a stable order.
    ///
    /// The default implementation reports a store that retains no DTD.
    fn unparsed_entity_names(&self, node: NodeId) -> Vec<String> {
        let _ = node;
        Vec::new()
    }

    /// The notation `name`, when `node` is a document node whose DTD declared one.
    ///
    /// The default implementation reports a store that retains no DTD.
    fn notation(&self, node: NodeId, name: &str) -> Option<Notation> {
        let _ = (node, name);
        None
    }

    /// The names of every notation `node` carries, in a stable order.
    ///
    /// The default implementation reports a store that retains no DTD.
    fn notation_names(&self, node: NodeId) -> Vec<String> {
        let _ = node;
        Vec::new()
    }

    /// Where `node` began in the text the tree was parsed from.
    ///
    /// This is diagnostic information rather than a property of the data model: a host language
    /// that reports where in a source module something went wrong needs it, and nothing else does.
    /// A store therefore only ever answers `Some` when it was explicitly asked to keep the
    /// coordinates — [`ParseOptions::record_source_positions`](crate::ParseOptions::record_source_positions)
    /// is how [`Tree`](crate::Tree) is asked — and a store built from anything but source text
    /// (the result of a transformation, a constructed sequence) has none to give. That is why the
    /// default implementation says `None` and no existing implementor has to change.
    fn source_position(&self, node: NodeId) -> Option<SourcePosition> {
        let _ = node;
        None
    }
}

/// A handle to one node of one [`XdmStore`]: the value an XPath engine passes around.
///
/// `NodeRef` is cheap to clone (one [`Arc`] bump) and has *identity* semantics: two handles are
/// equal exactly when they denote the same node of the same store, which is what the `is`
/// operator and the `union`/`intersect`/`except` de-duplication rules need. Ordering is
/// [document order](DocumentOrderKey), so a `Vec<NodeRef>` can simply be sorted.
///
/// The accessors live on the [`XdmNode`] trait, which must be in scope to call them.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{NodeKind, Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a><b/></a>")?;
/// let a = tree.document_element().expect("a document element");
/// assert_eq!(a.node_kind(), NodeKind::Element);
///
/// // Cloning a handle does not clone the node.
/// assert_eq!(a.clone(), a);
/// // Different nodes are never equal, even with the same name.
/// assert_ne!(a.clone(), tree.root());
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Clone)]
pub struct NodeRef {
    store: Arc<dyn XdmStore>,
    id: NodeId,
}

impl NodeRef {
    /// Builds a handle to `id` inside `store`.
    ///
    /// Backends call this to hand out their nodes; consumers get handles from a tree's root or
    /// by navigating.
    #[must_use]
    pub fn new(store: Arc<dyn XdmStore>, id: NodeId) -> Self {
        Self { store, id }
    }

    /// The store this node lives in.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn XdmStore> {
        &self.store
    }

    /// The node's store-local identifier.
    #[must_use]
    pub const fn id(&self) -> NodeId {
        self.id
    }

    /// Whether the two handles denote the same node — the XPath `is` operator.
    #[must_use]
    pub fn is_same_node(&self, other: &Self) -> bool {
        self == other
    }

    /// Rewraps `id` from the same store as this node.
    #[must_use]
    pub(crate) fn sibling_handle(&self, id: NodeId) -> Self {
        Self {
            store: Arc::clone(&self.store),
            id,
        }
    }

    fn store_address(&self) -> *const () {
        Arc::as_ptr(&self.store).cast::<()>()
    }
}

impl PartialEq for NodeRef {
    /// Node identity: the same node of the same store.
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && std::ptr::eq(self.store_address(), other.store_address())
    }
}

impl Eq for NodeRef {}

impl Hash for NodeRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        (self.store_address() as usize).hash(state);
    }
}

impl PartialOrd for NodeRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeRef {
    /// Document order. Nodes in different trees are ordered by their trees' identifiers, which
    /// is the "stable but implementation-dependent" order the specification permits.
    fn cmp(&self, other: &Self) -> Ordering {
        self.document_order_key().cmp(&other.document_order_key())
    }
}

impl fmt::Debug for NodeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = self.document_order_key();
        f.debug_struct("NodeRef")
            .field("kind", &self.node_kind())
            .field("name", &self.node_name().map(|name| name.to_string()))
            .field("tree", &key.tree)
            .field("position", &key.position)
            .finish()
    }
}

/// The accessors of the XDM node interface.
///
/// This is the trait an XPath, XQuery, XSLT or schema implementation is written against. It is
/// implemented for [`NodeRef`], which forwards to whatever [`XdmStore`] backs the node, so the
/// same code navigates an in-memory tree, a DOM, or a streaming window unchanged.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{NodeKind, Tree, XdmNode};
///
/// let tree = Tree::parse_str(r#"<doc xml:base="http://example.com/"><p id="x">hi</p></doc>"#)?;
/// let doc = tree.document_element().expect("a document element");
/// let paragraph = doc.children().first().cloned().expect("one child");
///
/// assert_eq!(paragraph.node_name().map(|n| n.to_string()).as_deref(), Some("p"));
/// assert_eq!(paragraph.string_value(), "hi");
/// assert_eq!(paragraph.base_uri().as_deref(), Some("http://example.com/"));
/// assert_eq!(paragraph.parent(), Some(doc));
/// assert_eq!(paragraph.attributes().len(), 1);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
pub trait XdmNode {
    /// The node's kind.
    fn node_kind(&self) -> NodeKind;

    /// The node's name.
    ///
    /// `None` for document, text and comment nodes. A processing instruction's name is its
    /// target (in no namespace); a namespace node's name is its prefix (in no namespace), or
    /// `None` for a default-namespace binding.
    ///
    /// [XDM 3.1 §5.4](https://www.w3.org/TR/xpath-datamodel-31/#dm-node-name) makes a name a
    /// *triple*, and the returned [`ExpandedName`] carries the third component — the prefix the
    /// name was written with — whenever the tree was built from one:
    /// [`Tree::parse_str`](crate::Tree::parse_str) records it, and
    /// [`TreeBuilder`](crate::TreeBuilder) records whatever its caller supplied through
    /// [`ExpandedName::with_prefix`]. That is what lets `fn:name`, `fn:node-name` and
    /// `fn:prefix-from-QName` report the document's own spelling; it never affects equality,
    /// hashing or a name test.
    fn node_name(&self) -> Option<ExpandedName>;

    /// The node's [string value](https://www.w3.org/TR/xpath-datamodel-31/#dm-string-value).
    fn string_value(&self) -> String;

    /// The node's [typed value](https://www.w3.org/TR/xpath-datamodel-31/#dm-typed-value).
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foty0012`](crate::ErrorCode::Foty0012) if the node has no typed value —
    /// which, for a schema-validated tree, means an element with element-only content or a
    /// nilled element.
    fn typed_value(&self) -> Result<Sequence, XdmError>;

    /// The node's type annotation, or `None` when the tree is not schema-aware.
    fn type_name(&self) -> Option<ExpandedName>;

    /// Whether the node is a nilled element (`xsi:nil="true"` and schema-valid).
    fn nilled(&self) -> Option<bool>;

    /// The node's parent.
    fn parent(&self) -> Option<NodeRef>;

    /// The node's children, in document order.
    fn children(&self) -> Vec<NodeRef>;

    /// The node's attributes, in document order.
    fn attributes(&self) -> Vec<NodeRef>;

    /// The node's in-scope namespace bindings, in document order.
    fn namespaces(&self) -> Vec<NodeRef>;

    /// The node's base URI.
    fn base_uri(&self) -> Option<String>;

    /// The URI of the document the node came from.
    fn document_uri(&self) -> Option<String>;

    /// The node's position in document order.
    fn document_order_key(&self) -> DocumentOrderKey;

    /// Whether the node is an ID.
    fn is_id(&self) -> bool;

    /// Whether the node is an IDREF or IDREFS.
    fn is_idref(&self) -> bool;

    /// The root of the node's tree.
    fn root(&self) -> NodeRef;

    /// The unparsed (`NDATA`) general entity `name` declared by this document's DTD.
    ///
    /// `unparsed-entities` is a property of the *document* node
    /// ([XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode)), so this is
    /// `None` on every other kind of node — call it on
    /// [`root`](XdmNode::root) to ask the question `fn:unparsed-entity-uri` asks.
    fn unparsed_entity(&self, name: &str) -> Option<UnparsedEntity>;

    /// The names of every unparsed entity this document declares, in a stable order.
    fn unparsed_entity_names(&self) -> Vec<String>;

    /// The notation `name` declared by this document's DTD, on the same terms as
    /// [`unparsed_entity`](XdmNode::unparsed_entity).
    fn notation(&self, name: &str) -> Option<Notation>;

    /// The names of every notation this document declares, in a stable order.
    fn notation_names(&self) -> Vec<String>;

    /// Where this node began in the text its tree was parsed from, when the tree was asked to
    /// record source positions.
    ///
    /// `None` whenever the coordinates were not kept: for a tree parsed without
    /// [`ParseOptions::record_source_positions`](crate::ParseOptions::record_source_positions),
    /// for a tree that was built rather than parsed, and for the node kinds a parse records
    /// nothing for — see [`SourcePosition`].
    fn source_position(&self) -> Option<SourcePosition>;
}

impl XdmNode for NodeRef {
    fn node_kind(&self) -> NodeKind {
        self.store.node_kind(self.id)
    }

    fn node_name(&self) -> Option<ExpandedName> {
        self.store.node_name(self.id)
    }

    fn string_value(&self) -> String {
        self.store.string_value(self.id)
    }

    fn typed_value(&self) -> Result<Sequence, XdmError> {
        self.store.typed_value(self.id)
    }

    fn type_name(&self) -> Option<ExpandedName> {
        self.store.type_name(self.id)
    }

    fn nilled(&self) -> Option<bool> {
        self.store.nilled(self.id)
    }

    fn parent(&self) -> Option<NodeRef> {
        self.store.parent(self.id).map(|id| self.sibling_handle(id))
    }

    fn children(&self) -> Vec<NodeRef> {
        self.store
            .children(self.id)
            .into_iter()
            .map(|id| self.sibling_handle(id))
            .collect()
    }

    fn attributes(&self) -> Vec<NodeRef> {
        self.store
            .attributes(self.id)
            .into_iter()
            .map(|id| self.sibling_handle(id))
            .collect()
    }

    fn namespaces(&self) -> Vec<NodeRef> {
        self.store
            .namespaces(self.id)
            .into_iter()
            .map(|id| self.sibling_handle(id))
            .collect()
    }

    fn base_uri(&self) -> Option<String> {
        self.store.base_uri(self.id)
    }

    fn document_uri(&self) -> Option<String> {
        self.store.document_uri(self.id)
    }

    fn document_order_key(&self) -> DocumentOrderKey {
        self.store.document_order_key(self.id)
    }

    fn is_id(&self) -> bool {
        self.store.is_id(self.id)
    }

    fn is_idref(&self) -> bool {
        self.store.is_idref(self.id)
    }

    fn root(&self) -> NodeRef {
        self.sibling_handle(self.store.root(self.id))
    }

    fn unparsed_entity(&self, name: &str) -> Option<UnparsedEntity> {
        self.store.unparsed_entity(self.id, name)
    }

    fn unparsed_entity_names(&self) -> Vec<String> {
        self.store.unparsed_entity_names(self.id)
    }

    fn notation(&self, name: &str) -> Option<Notation> {
        self.store.notation(self.id, name)
    }

    fn notation_names(&self) -> Vec<String> {
        self.store.notation_names(self.id)
    }

    fn source_position(&self) -> Option<SourcePosition> {
        self.store.source_position(self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_test_names_and_child_kinds() {
        assert_eq!(NodeKind::Document.kind_test_name(), "document-node");
        assert_eq!(NodeKind::Namespace.to_string(), "namespace-node");
        assert!(NodeKind::Element.is_child_kind());
        assert!(NodeKind::Comment.is_child_kind());
        assert!(!NodeKind::Attribute.is_child_kind());
        assert!(!NodeKind::Namespace.is_child_kind());
        assert!(!NodeKind::Document.is_child_kind());
    }

    #[test]
    fn document_order_keys_sort_by_tree_then_position() {
        let first = DocumentOrderKey {
            tree: 1,
            position: 9,
        };
        let second = DocumentOrderKey {
            tree: 2,
            position: 0,
        };
        assert!(first < second);
        assert!(
            DocumentOrderKey {
                tree: 1,
                position: 0
            } < first
        );
    }
}

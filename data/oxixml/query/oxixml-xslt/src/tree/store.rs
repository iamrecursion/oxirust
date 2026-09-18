//! The arena and the [`XdmStore`] implementation behind a result tree.

use oxixml_infoset::{
    DocumentOrderKey, ExpandedName, NodeId, NodeKind, NodeRef, Notation, Sequence, UnparsedEntity,
    XdmError, XdmStore, next_tree_id,
};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// The DTD declarations a result tree carries on its document node.
///
/// A result tree normally has none — nothing a stylesheet constructs declares an entity. The
/// exception is the stripped copy of a *source* document that `xsl:strip-space` produces: it is
/// still the same document, so [XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode)'s
/// `[unparsed entities]` and `[notations]` properties have to survive the copy, or
/// `fn:unparsed-entity-uri` would answer differently for a stylesheet that strips whitespace.
#[derive(Debug, Clone, Default)]
pub(crate) struct Declarations {
    pub(crate) unparsed_entities: BTreeMap<String, UnparsedEntity>,
    pub(crate) notations: BTreeMap<String, Notation>,
}

impl Declarations {
    /// Whether nothing at all was declared, which is the common case.
    fn is_empty(&self) -> bool {
        self.unparsed_entities.is_empty() && self.notations.is_empty()
    }
}

/// One node of a result tree, laid out in document order.
#[derive(Debug, Clone)]
pub(crate) struct ResultNode {
    pub(crate) kind: NodeKind,
    pub(crate) name: Option<ExpandedName>,
    /// The prefix the serializer should use, when the stylesheet asked for a particular one.
    pub(crate) prefix: Option<String>,
    pub(crate) value: String,
    /// Set on a text node produced with `disable-output-escaping="yes"`.
    pub(crate) unescaped: bool,
    pub(crate) parent: Option<usize>,
    pub(crate) children: Vec<usize>,
    pub(crate) attributes: Vec<usize>,
    pub(crate) namespaces: Vec<usize>,
    pub(crate) base_uri: Option<String>,
    /// Set on an attribute whose source counterpart was typed as an ID.
    pub(crate) typed_id: bool,
    /// Set on an attribute whose source counterpart was typed as an IDREF or IDREFS.
    pub(crate) typed_idref: bool,
}

impl ResultNode {
    pub(crate) fn new(kind: NodeKind) -> Self {
        Self {
            kind,
            name: None,
            prefix: None,
            value: String::new(),
            unescaped: false,
            parent: None,
            children: Vec::new(),
            attributes: Vec::new(),
            namespaces: Vec::new(),
            base_uri: None,
            typed_id: false,
            typed_idref: false,
        }
    }
}

/// The arena a [`ResultTree`](super::ResultTree) hands [`NodeRef`]s into.
pub(crate) struct ResultStore {
    nodes: Vec<ResultNode>,
    tree: u64,
    document_uri: Option<String>,
    declarations: Declarations,
}

impl fmt::Debug for ResultStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResultStore")
            .field("nodes", &self.nodes.len())
            .field("tree", &self.tree)
            .finish()
    }
}

impl ResultStore {
    pub(crate) fn new(
        nodes: Vec<ResultNode>,
        document_uri: Option<String>,
        declarations: Declarations,
    ) -> Self {
        Self {
            nodes,
            tree: next_tree_id(),
            document_uri,
            declarations,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn node(&self, id: NodeId) -> Option<&ResultNode> {
        usize::try_from(id.0).ok().and_then(|id| self.nodes.get(id))
    }

    pub(crate) fn raw(&self, index: usize) -> Option<&ResultNode> {
        self.nodes.get(index)
    }

    fn get(&self, id: NodeId) -> Option<&ResultNode> {
        self.node(id)
    }
}

fn to_id(index: usize) -> NodeId {
    NodeId(index as u64)
}

impl XdmStore for ResultStore {
    fn node_kind(&self, node: NodeId) -> NodeKind {
        self.get(node).map_or(NodeKind::Document, |data| data.kind)
    }

    fn node_name(&self, node: NodeId) -> Option<ExpandedName> {
        self.get(node).and_then(|data| data.name.clone())
    }

    fn string_value(&self, node: NodeId) -> String {
        let Some(data) = self.get(node) else {
            return String::new();
        };
        match data.kind {
            NodeKind::Document | NodeKind::Element => {
                let mut out = String::new();
                self.append_text(node, &mut out);
                out
            }
            _ => data.value.clone(),
        }
    }

    fn typed_value(&self, node: NodeId) -> Result<Sequence, XdmError> {
        Ok(Sequence::from(oxixml_infoset::AtomicValue::untyped_atomic(
            self.string_value(node),
        )))
    }

    fn type_name(&self, _node: NodeId) -> Option<ExpandedName> {
        None
    }

    fn nilled(&self, node: NodeId) -> Option<bool> {
        (self.node_kind(node) == NodeKind::Element).then_some(false)
    }

    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.get(node).and_then(|data| data.parent).map(to_id)
    }

    fn children(&self, node: NodeId) -> Vec<NodeId> {
        self.get(node)
            .map(|data| data.children.iter().copied().map(to_id).collect())
            .unwrap_or_default()
    }

    fn attributes(&self, node: NodeId) -> Vec<NodeId> {
        self.get(node)
            .map(|data| data.attributes.iter().copied().map(to_id).collect())
            .unwrap_or_default()
    }

    fn namespaces(&self, node: NodeId) -> Vec<NodeId> {
        self.get(node)
            .map(|data| data.namespaces.iter().copied().map(to_id).collect())
            .unwrap_or_default()
    }

    fn base_uri(&self, node: NodeId) -> Option<String> {
        self.get(node).and_then(|data| data.base_uri.clone())
    }

    fn document_uri(&self, node: NodeId) -> Option<String> {
        (node == NodeId(0))
            .then(|| self.document_uri.clone())
            .flatten()
    }

    fn document_order_key(&self, node: NodeId) -> DocumentOrderKey {
        DocumentOrderKey {
            tree: self.tree,
            position: node.0,
        }
    }

    fn is_unescaped_text(&self, node: NodeId) -> bool {
        self.get(node)
            .is_some_and(|data| data.kind == NodeKind::Text && data.unescaped)
    }

    /// Whether the node is an ID.
    ///
    /// A tree a stylesheet *constructs* has no DTD and no schema, so the only ID it can carry is
    /// the one [xml:id](https://www.w3.org/TR/xml-id/) defines: "an `xml:id` attribute … is an
    /// ID", which holds wherever the attribute occurs and therefore in a tree a stylesheet built
    /// as much as in one a parser did. `key-077` builds a lookup table of `<code xml:id="…">`
    /// elements in a variable and reads it back with `id('d', $lookup)`.
    ///
    /// A tree this crate builds is not always a constructed one, though: whitespace stripping
    /// ([XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#stripping)) re-materializes a *source*
    /// document, and the result is the same document with some text nodes removed — the DTD that
    /// declared an attribute `ID` still did so. The typing is therefore carried across that copy
    /// and answered here, exactly as the unparsed entities and notations beside it are; without
    /// it a stylesheet that strips whitespace loses every `fn:id` and every `id(…)` pattern over
    /// a DTD-declared identifier (`id-028` … `id-031`).
    fn is_id(&self, node: NodeId) -> bool {
        self.get(node).is_some_and(|data| {
            data.kind == NodeKind::Attribute
                && (data.typed_id
                    || data.name.as_ref().is_some_and(|name| {
                        name.namespace() == Some(oxixml_qname::XML_NAMESPACE)
                            && name.local_name() == "id"
                    }))
        })
    }

    fn is_idref(&self, node: NodeId) -> bool {
        self.get(node)
            .is_some_and(|data| data.kind == NodeKind::Attribute && data.typed_idref)
    }

    fn unparsed_entity(&self, node: NodeId, name: &str) -> Option<UnparsedEntity> {
        self.on_a_document_node(node)?
            .unparsed_entities
            .get(name)
            .cloned()
    }

    fn unparsed_entity_names(&self, node: NodeId) -> Vec<String> {
        self.on_a_document_node(node)
            .map(|declarations| declarations.unparsed_entities.keys().cloned().collect())
            .unwrap_or_default()
    }

    fn notation(&self, node: NodeId, name: &str) -> Option<Notation> {
        self.on_a_document_node(node)?.notations.get(name).cloned()
    }

    fn notation_names(&self, node: NodeId) -> Vec<String> {
        self.on_a_document_node(node)
            .map(|declarations| declarations.notations.keys().cloned().collect())
            .unwrap_or_default()
    }
}

impl ResultStore {
    /// The DTD declarations, when `node` is the document node they belong to.
    ///
    /// XDM 3.1 makes `[unparsed entities]` and `[notations]` properties of a *document* node, so
    /// an element or a text node reports none of its own; a caller reaches them through
    /// `fn:root`, which is exactly the `XTDE1370`/`XTDE1380` condition of the accessors.
    fn on_a_document_node(&self, node: NodeId) -> Option<&Declarations> {
        (!self.declarations.is_empty() && self.node_kind(node) == NodeKind::Document)
            .then_some(&self.declarations)
    }

    fn append_text(&self, node: NodeId, out: &mut String) {
        let Some(data) = self.get(node) else {
            return;
        };
        match data.kind {
            NodeKind::Text => out.push_str(&data.value),
            NodeKind::Document | NodeKind::Element => {
                for child in &data.children {
                    self.append_text(to_id(*child), out);
                }
            }
            _ => {}
        }
    }
}

/// A materialized result tree: the output of a sequence constructor.
///
/// It is a full [`XdmStore`], so a variable bound to one can be navigated by XPath exactly like
/// a source document.
#[derive(Debug, Clone)]
pub struct ResultTree {
    pub(crate) store: Arc<ResultStore>,
    /// The text a destination whose serialization method builds *no* tree already produced.
    ///
    /// [XSLT 3.0 §26](https://www.w3.org/TR/xslt-30/#serialization) gives `build-tree` the
    /// default `no` for the `json` and `adaptive` methods: what such a destination produces is
    /// a *raw result*, a sequence that may hold a map or an array, and there is no tree it could
    /// be normalized into. The destination is still one of the transformation's result trees as
    /// far as every caller is concerned, so it is carried as one — with an empty document node
    /// standing for the tree, and the serialization it really produced held here for
    /// [`crate::serialize::serialize`] to hand back unchanged.
    raw_serialization: Option<String>,
}

impl ResultTree {
    pub(crate) fn from_store(store: ResultStore) -> Self {
        Self {
            store: Arc::new(store),
            raw_serialization: None,
        }
    }

    /// Records the serialization a method that builds no tree produced — see
    /// [`ResultTree::raw_serialization`].
    #[must_use]
    pub(crate) fn with_raw_serialization(mut self, text: String) -> Self {
        self.raw_serialization = Some(text);
        self
    }

    /// The serialization a method that builds no tree produced, if this is such a destination.
    pub(crate) fn raw_serialization(&self) -> Option<&str> {
        self.raw_serialization.as_deref()
    }

    /// The document node at the root of the tree.
    #[must_use]
    pub fn root(&self) -> NodeRef {
        NodeRef::new(Arc::clone(&self.store) as Arc<dyn XdmStore>, NodeId(0))
    }

    /// A handle to the node at `index` in the arena.
    #[must_use]
    pub(crate) fn node_at(&self, index: usize) -> Option<NodeRef> {
        (index < self.store.len())
            .then(|| NodeRef::new(Arc::clone(&self.store) as Arc<dyn XdmStore>, to_id(index)))
    }

    /// The number of nodes in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Always `false`: a result tree always has its document node.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The arena entry at `index`, for the serializer, which needs the prefix hints and the
    /// output-escaping flags that the XDM view does not model.
    pub(crate) fn raw(&self, index: usize) -> Option<&ResultNode> {
        self.store.raw(index)
    }

    /// The children of the document node.
    pub(crate) fn root_children(&self) -> Vec<usize> {
        self.store
            .raw(0)
            .map(|data| data.children.clone())
            .unwrap_or_default()
    }
}

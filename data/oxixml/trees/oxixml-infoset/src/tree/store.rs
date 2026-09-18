//! The arena that backs [`Tree`](crate::Tree), and its [`XdmStore`] implementation.

use super::dtd::{Notation, UnparsedEntity};
use super::position::SourcePosition;
use crate::atomic::AtomicValue;
use crate::error::XdmError;
use crate::node::{DocumentOrderKey, NodeId, NodeKind, XdmStore};
use crate::sequence::Sequence;
use oxixml_qname::ExpandedName;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Hands out process-unique tree identifiers.
///
/// Every [`XdmStore`] must give its trees an identifier no other store uses, so that
/// [`DocumentOrderKey`] can order nodes from unrelated trees consistently. Call this once per
/// tree you create.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::next_tree_id;
///
/// assert_ne!(next_tree_id(), next_tree_id());
/// ```
#[must_use]
pub fn next_tree_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// One node of the arena. Fields are laid out in document order by construction, so a node's
/// index *is* its document-order position.
#[derive(Debug, Clone)]
pub(crate) struct NodeData {
    pub(crate) kind: NodeKind,
    pub(crate) name: Option<ExpandedName>,
    pub(crate) parent: Option<usize>,
    pub(crate) children: Vec<usize>,
    pub(crate) attributes: Vec<usize>,
    pub(crate) namespaces: Vec<usize>,
    /// Character data for a text node, the normalized value for an attribute, the content for a
    /// comment or processing instruction, the namespace name for a namespace node. Unused for
    /// document and element nodes, whose string value is computed from their descendants.
    pub(crate) value: String,
    pub(crate) base_uri: Option<String>,
    pub(crate) type_name: Option<ExpandedName>,
    pub(crate) typed_value: Option<Sequence>,
    pub(crate) nilled: Option<bool>,
    pub(crate) is_id: bool,
    pub(crate) is_idref: bool,
}

impl NodeData {
    pub(crate) fn new(kind: NodeKind) -> Self {
        Self {
            kind,
            name: None,
            parent: None,
            children: Vec::new(),
            attributes: Vec::new(),
            namespaces: Vec::new(),
            value: String::new(),
            base_uri: None,
            type_name: None,
            typed_value: None,
            nilled: None,
            is_id: false,
            is_idref: false,
        }
    }
}

/// The in-memory arena an [`crate::Tree`] wraps.
#[derive(Debug)]
pub(crate) struct TreeStore {
    tree_id: u64,
    nodes: Vec<NodeData>,
    document_uri: Option<String>,
    /// The unparsed entities the document's DTD declared, keyed on name. Empty for a tree with
    /// no document node, which has nothing to carry the property.
    unparsed_entities: BTreeMap<String, UnparsedEntity>,
    /// The notations the document's DTD declared, keyed on name.
    notations: BTreeMap<String, Notation>,
    /// Where each node began in the source text, keyed on the node's index.
    ///
    /// A side table rather than a field of [`NodeData`]: recording positions is opt-in and rare,
    /// and a tree that does not record them must not pay a machine word per node for the ones it
    /// does not have. The map is empty for every tree built without
    /// [`ParseOptions::record_source_positions`](crate::ParseOptions::record_source_positions).
    positions: BTreeMap<usize, SourcePosition>,
}

impl TreeStore {
    pub(crate) fn new(
        nodes: Vec<NodeData>,
        document_uri: Option<String>,
        unparsed_entities: BTreeMap<String, UnparsedEntity>,
        notations: BTreeMap<String, Notation>,
        positions: BTreeMap<usize, SourcePosition>,
    ) -> Self {
        Self {
            tree_id: next_tree_id(),
            nodes,
            document_uri,
            unparsed_entities,
            notations,
            positions,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn document_uri_str(&self) -> Option<&str> {
        self.document_uri.as_deref()
    }

    pub(crate) const fn unparsed_entity_map(&self) -> &BTreeMap<String, UnparsedEntity> {
        &self.unparsed_entities
    }

    pub(crate) const fn notation_map(&self) -> &BTreeMap<String, Notation> {
        &self.notations
    }

    /// Whether `node` is the document node these declarations belong to.
    ///
    /// The DTD is a property of the document information item, so a query that reaches it from
    /// somewhere else in the tree has to climb to the root first — which is exactly what
    /// `fn:unparsed-entity-uri` is specified to do.
    fn is_document_node(&self, node: NodeId) -> bool {
        self.node(node)
            .is_some_and(|data| data.kind == NodeKind::Document)
    }

    fn node(&self, id: NodeId) -> Option<&NodeData> {
        usize::try_from(id.0).ok().and_then(|id| self.nodes.get(id))
    }

    /// Concatenates the string values of every text node in the subtree rooted at `id`, in
    /// document order — the string value of a document or element node.
    fn text_content(&self, id: usize) -> String {
        let mut out = String::new();
        self.push_text_content(id, &mut out);
        out
    }

    /// Appends the string value of the subtree rooted at `id` to `out`.
    ///
    /// Deliberately iterative: recursing once per tree level would overflow the
    /// stack — an uncatchable process abort — on a merely deeply nested document.
    fn push_text_content(&self, id: usize, out: &mut String) {
        // Each entry is the not-yet-visited tail of one level's child list, kept
        // in reverse so that `pop` yields the next node in document order.
        let Some(node) = self.nodes.get(id) else {
            return;
        };
        let mut stack: Vec<Vec<usize>> = Vec::new();
        let mut pending: Vec<usize> = node.children.clone();
        pending.reverse();
        stack.push(pending);

        while let Some(level) = stack.last_mut() {
            let Some(child) = level.pop() else {
                stack.pop();
                continue;
            };
            let Some(data) = self.nodes.get(child) else {
                continue;
            };
            match data.kind {
                NodeKind::Text => out.push_str(&data.value),
                NodeKind::Element if !data.children.is_empty() => {
                    let mut grandchildren = data.children.clone();
                    grandchildren.reverse();
                    stack.push(grandchildren);
                }
                _ => {}
            }
        }
    }

    fn ids(indices: &[usize]) -> Vec<NodeId> {
        indices.iter().map(|index| NodeId(*index as u64)).collect()
    }
}

impl XdmStore for TreeStore {
    fn node_kind(&self, node: NodeId) -> NodeKind {
        // Every identifier a caller can obtain names a node of this store; the fallback only
        // covers a `NodeId` invented elsewhere, and a document node is the least surprising
        // answer for a tree that has one.
        self.node(node).map_or(NodeKind::Document, |data| data.kind)
    }

    fn node_name(&self, node: NodeId) -> Option<ExpandedName> {
        self.node(node).and_then(|data| data.name.clone())
    }

    fn string_value(&self, node: NodeId) -> String {
        let Some(data) = self.node(node) else {
            return String::new();
        };
        match data.kind {
            NodeKind::Document | NodeKind::Element => {
                usize::try_from(node.0).map_or_else(|_| String::new(), |id| self.text_content(id))
            }
            _ => data.value.clone(),
        }
    }

    fn typed_value(&self, node: NodeId) -> Result<Sequence, XdmError> {
        let Some(data) = self.node(node) else {
            return Ok(Sequence::empty());
        };
        if let Some(typed) = &data.typed_value {
            return Ok(typed.clone());
        }
        let string_value = self.string_value(node);
        Ok(match data.kind {
            // XDM 3.1 §6: an untyped element, an attribute of type xs:untypedAtomic, a text
            // node and a document node all atomize to a single xs:untypedAtomic value.
            NodeKind::Document | NodeKind::Element | NodeKind::Attribute | NodeKind::Text => {
                Sequence::from(AtomicValue::untyped_atomic(string_value))
            }
            // Comments, processing instructions and namespace nodes atomize to xs:string.
            NodeKind::Comment | NodeKind::ProcessingInstruction | NodeKind::Namespace => {
                Sequence::from(AtomicValue::string(string_value))
            }
        })
    }

    fn type_name(&self, node: NodeId) -> Option<ExpandedName> {
        self.node(node).and_then(|data| data.type_name.clone())
    }

    fn nilled(&self, node: NodeId) -> Option<bool> {
        let data = self.node(node)?;
        match data.kind {
            NodeKind::Element => Some(data.nilled.unwrap_or(false)),
            _ => None,
        }
    }

    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.node(node)
            .and_then(|data| data.parent)
            .map(|index| NodeId(index as u64))
    }

    fn children(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node)
            .map_or_else(Vec::new, |data| Self::ids(&data.children))
    }

    fn attributes(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node)
            .map_or_else(Vec::new, |data| Self::ids(&data.attributes))
    }

    fn namespaces(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node)
            .map_or_else(Vec::new, |data| Self::ids(&data.namespaces))
    }

    fn base_uri(&self, node: NodeId) -> Option<String> {
        self.node(node).and_then(|data| data.base_uri.clone())
    }

    fn document_uri(&self, node: NodeId) -> Option<String> {
        // Only the document node itself carries a document URI, per XDM 3.1 §6.1.
        match self.node(node)?.kind {
            NodeKind::Document => self.document_uri.clone(),
            _ => None,
        }
    }

    fn document_order_key(&self, node: NodeId) -> DocumentOrderKey {
        DocumentOrderKey {
            tree: self.tree_id,
            position: node.0,
        }
    }

    fn is_id(&self, node: NodeId) -> bool {
        self.node(node).is_some_and(|data| data.is_id)
    }

    fn is_idref(&self, node: NodeId) -> bool {
        self.node(node).is_some_and(|data| data.is_idref)
    }

    fn root(&self, _node: NodeId) -> NodeId {
        NodeId(0)
    }

    fn unparsed_entity(&self, node: NodeId, name: &str) -> Option<UnparsedEntity> {
        self.is_document_node(node)
            .then(|| self.unparsed_entities.get(name).cloned())
            .flatten()
    }

    fn unparsed_entity_names(&self, node: NodeId) -> Vec<String> {
        match self.is_document_node(node) {
            true => self.unparsed_entities.keys().cloned().collect(),
            false => Vec::new(),
        }
    }

    fn notation(&self, node: NodeId, name: &str) -> Option<Notation> {
        self.is_document_node(node)
            .then(|| self.notations.get(name).cloned())
            .flatten()
    }

    fn notation_names(&self, node: NodeId) -> Vec<String> {
        match self.is_document_node(node) {
            true => self.notations.keys().cloned().collect(),
            false => Vec::new(),
        }
    }

    fn source_position(&self, node: NodeId) -> Option<SourcePosition> {
        let index = usize::try_from(node.0).ok()?;
        self.positions.get(&index).copied()
    }
}

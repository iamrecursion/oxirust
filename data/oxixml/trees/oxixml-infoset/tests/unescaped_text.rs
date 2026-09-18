//! The `is_unescaped_text` property of a text node.
//!
//! XSLT's [`disable-output-escaping`](https://www.w3.org/TR/xslt-30/#disable-output-escaping)
//! asks the serializer to write a text node's characters through untouched, so that `&lt;`
//! reaches the output as `<`. A processor that materializes its result as a tree has to record
//! that somewhere, because the node may be copied — into another result tree, into a temporary
//! tree, through a `text()` template — long before anything is serialized, and a copy that
//! forgets the flag silently escapes what the stylesheet asked not to escape.
//!
//! [`XdmStore::is_unescaped_text`] is where it is recorded. It is defaulted to `false`, so a
//! store that has no such notion — every store built from a parsed document — implements
//! nothing; only a store that materializes transformation results overrides it.

use std::sync::Arc;

use oxixml_infoset::{
    AtomicValue, DocumentOrderKey, ExpandedName, NodeId, NodeKind, NodeRef, Sequence, Tree,
    XdmError, XdmNode, XdmStore, next_tree_id,
};

/// A store holding one document node with two text children, the second of which is marked.
#[derive(Debug)]
struct ResultStore {
    tree: u64,
    /// `(the characters, whether they must reach the output unescaped)`, in document order.
    texts: Vec<(String, bool)>,
}

impl ResultStore {
    /// The document node is index `0`; text `n` is index `n + 1`.
    const DOCUMENT: NodeId = NodeId(0);

    fn new(texts: &[(&str, bool)]) -> Arc<Self> {
        Arc::new(Self {
            tree: next_tree_id(),
            texts: texts
                .iter()
                .map(|&(text, unescaped)| (text.to_owned(), unescaped))
                .collect(),
        })
    }

    fn text_index(node: NodeId) -> Option<usize> {
        usize::try_from(node.0).ok()?.checked_sub(1)
    }
}

impl XdmStore for ResultStore {
    fn node_kind(&self, node: NodeId) -> NodeKind {
        match Self::text_index(node) {
            Some(_) => NodeKind::Text,
            None => NodeKind::Document,
        }
    }

    fn node_name(&self, _node: NodeId) -> Option<ExpandedName> {
        None
    }

    fn string_value(&self, node: NodeId) -> String {
        match Self::text_index(node).and_then(|index| self.texts.get(index)) {
            Some((text, _)) => text.clone(),
            None => self.texts.iter().map(|(text, _)| text.as_str()).collect(),
        }
    }

    fn typed_value(&self, node: NodeId) -> Result<Sequence, XdmError> {
        // XDM 3.1 §6: a text node and a document node both atomize to one `xs:untypedAtomic`.
        Ok(Sequence::from(AtomicValue::untyped_atomic(
            self.string_value(node),
        )))
    }

    fn type_name(&self, _node: NodeId) -> Option<ExpandedName> {
        None
    }

    fn nilled(&self, _node: NodeId) -> Option<bool> {
        None
    }

    fn parent(&self, node: NodeId) -> Option<NodeId> {
        Self::text_index(node).map(|_| Self::DOCUMENT)
    }

    fn children(&self, node: NodeId) -> Vec<NodeId> {
        match Self::text_index(node) {
            Some(_) => Vec::new(),
            None => (1..=self.texts.len())
                .filter_map(|index| u64::try_from(index).ok())
                .map(NodeId)
                .collect(),
        }
    }

    fn attributes(&self, _node: NodeId) -> Vec<NodeId> {
        Vec::new()
    }

    fn namespaces(&self, _node: NodeId) -> Vec<NodeId> {
        Vec::new()
    }

    fn base_uri(&self, _node: NodeId) -> Option<String> {
        None
    }

    fn document_uri(&self, _node: NodeId) -> Option<String> {
        None
    }

    fn document_order_key(&self, node: NodeId) -> DocumentOrderKey {
        DocumentOrderKey {
            tree: self.tree,
            position: node.0,
        }
    }

    fn is_id(&self, _node: NodeId) -> bool {
        false
    }

    fn is_idref(&self, _node: NodeId) -> bool {
        false
    }

    fn is_unescaped_text(&self, node: NodeId) -> bool {
        Self::text_index(node)
            .and_then(|index| self.texts.get(index))
            .is_some_and(|&(_, unescaped)| unescaped)
    }
}

#[test]
fn a_store_that_records_the_flag_can_report_it_back() {
    let store = ResultStore::new(&[("escaped", false), ("&lt;raw&gt;", true)]);
    let document = NodeRef::new(store.clone(), ResultStore::DOCUMENT);
    let children = document.children();
    assert_eq!(children.len(), 2);

    // The property is read back through the store the handle carries, which is the only way a
    // consumer holding a `NodeRef` can reach it: `Arc<dyn XdmStore>` cannot be downcast.
    let flags: Vec<bool> = children
        .iter()
        .map(|child| child.store().is_unescaped_text(child.id()))
        .collect();
    assert_eq!(flags, [false, true]);

    // Nothing else about the node changes: the flag is a serialization property only.
    assert_eq!(children[1].string_value(), "&lt;raw&gt;");
    assert_eq!(children[1].node_kind(), NodeKind::Text);
    assert!(!document.store().is_unescaped_text(document.id()));
}

#[test]
fn every_other_store_answers_false_without_implementing_anything() -> Result<(), XdmError> {
    // The method is defaulted, so a store built from a parsed document — which has no notion of
    // disabled output escaping — keeps compiling and answers `false` for every node.
    let tree = Tree::parse_str("<r>text<!--c--><b a='1'/></r>")?;
    let root = tree.root();
    let mut stack = vec![root];
    let mut seen = 0_usize;
    while let Some(node) = stack.pop() {
        seen += 1;
        assert!(
            !node.store().is_unescaped_text(node.id()),
            "{:?} must not be marked",
            node.node_kind()
        );
        stack.extend(node.children());
        stack.extend(node.attributes());
    }
    assert!(seen >= 5, "the walk must have reached every node");
    Ok(())
}

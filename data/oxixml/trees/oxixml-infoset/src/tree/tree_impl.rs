//! [`Tree`] — an in-memory XDM tree and the [`XdmStore`](crate::XdmStore) it is backed by.

use super::dtd::{Notation, UnparsedEntity};
use super::store::TreeStore;
use crate::node::{NodeId, NodeKind, NodeRef, XdmNode};
use std::sync::Arc;

/// A complete, immutable XDM tree held in a single arena.
///
/// This is the reference [`XdmStore`](crate::XdmStore) implementation: it is what
/// [`Tree::parse_str`] and [`TreeBuilder`](crate::TreeBuilder) produce, and what a query engine
/// evaluates over when nothing more exotic is plugged in. Cloning a `Tree` shares the arena.
///
/// The arena is laid out in document order, so [`DocumentOrderKey`](crate::DocumentOrderKey)
/// comparison is an integer comparison and [`Tree::nodes_in_document_order`] is a slice scan.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{NodeKind, Tree, XdmNode};
///
/// let tree = Tree::parse_str(r#"<catalog><book id="b1">Dune</book></catalog>"#)?;
///
/// let root = tree.root();
/// assert_eq!(root.node_kind(), NodeKind::Document);
///
/// let catalog = tree.document_element().expect("a document element");
/// assert_eq!(catalog.node_name().map(|name| name.to_string()).as_deref(), Some("catalog"));
/// assert_eq!(catalog.string_value(), "Dune");
///
/// let book = catalog.children().first().cloned().expect("one child");
/// assert_eq!(book.attributes().len(), 1);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Tree {
    store: Arc<TreeStore>,
}

impl Tree {
    pub(crate) fn from_store(store: Arc<TreeStore>) -> Self {
        Self { store }
    }

    /// The document node at the root of the tree.
    #[must_use]
    pub fn root(&self) -> NodeRef {
        NodeRef::new(self.erased_store(), NodeId(0))
    }

    /// The tree's single element child — the *document element* — if it has one.
    ///
    /// A tree built by [`TreeBuilder::new_element_rooted`](crate::TreeBuilder::new_element_rooted)
    /// *is* its element, so that is what this returns for one.
    #[must_use]
    pub fn document_element(&self) -> Option<NodeRef> {
        let root = self.root();
        if root.node_kind() == NodeKind::Element {
            return Some(root);
        }
        root.children()
            .into_iter()
            .find(|child| child.node_kind() == NodeKind::Element)
    }

    /// The absolute URI the document was retrieved from, if one was supplied.
    ///
    /// An [element-rooted](crate::TreeBuilder::new_element_rooted) tree never has one: a
    /// `document-uri` belongs to a document node, and such a tree has none.
    #[must_use]
    pub fn document_uri(&self) -> Option<&str> {
        self.store.document_uri_str()
    }

    /// The unparsed (`NDATA`) general entity `name`, if the document's DTD declared one.
    ///
    /// Unlike [`XdmNode::unparsed_entity`](crate::XdmNode::unparsed_entity), which answers for
    /// one *node* and is therefore `None` anywhere but the document node, this is a property of
    /// the whole tree and answers wherever the declaration exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::Tree;
    ///
    /// let tree = Tree::parse_str(
    ///     r#"<!DOCTYPE r [<!NOTATION jpg SYSTEM "image/jpeg">
    ///                     <!ENTITY photo SYSTEM "p.jpg" NDATA jpg>]><r/>"#,
    /// )?;
    /// assert_eq!(tree.unparsed_entity_names(), ["photo"]);
    /// assert_eq!(
    ///     tree.unparsed_entity("photo").map(|entity| entity.notation.clone()),
    ///     Some("jpg".to_owned())
    /// );
    /// assert!(tree.unparsed_entity("absent").is_none());
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    #[must_use]
    pub fn unparsed_entity(&self, name: &str) -> Option<&UnparsedEntity> {
        self.store.unparsed_entity_map().get(name)
    }

    /// The names of every unparsed entity the document declares, in ascending order.
    #[must_use]
    pub fn unparsed_entity_names(&self) -> Vec<&str> {
        self.store
            .unparsed_entity_map()
            .keys()
            .map(String::as_str)
            .collect()
    }

    /// The notation `name`, if the document's DTD declared one.
    #[must_use]
    pub fn notation(&self, name: &str) -> Option<&Notation> {
        self.store.notation_map().get(name)
    }

    /// The names of every notation the document declares, in ascending order.
    #[must_use]
    pub fn notation_names(&self) -> Vec<&str> {
        self.store
            .notation_map()
            .keys()
            .map(String::as_str)
            .collect()
    }

    /// The number of nodes in the tree, counting attribute and namespace nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Always `false`: a tree always contains at least its root node.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Every node of the tree, in document order.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{NodeKind, Tree, XdmNode};
    ///
    /// let tree = Tree::parse_str(r#"<a k="v"><b/></a>"#)?;
    /// let kinds: Vec<_> = tree
    ///     .nodes_in_document_order()
    ///     .iter()
    ///     .map(XdmNode::node_kind)
    ///     .collect();
    ///
    /// // Document, <a>, <a>'s `xml` namespace node, @k, <b>, <b>'s `xml` namespace node.
    /// assert_eq!(
    ///     kinds,
    ///     [
    ///         NodeKind::Document,
    ///         NodeKind::Element,
    ///         NodeKind::Namespace,
    ///         NodeKind::Attribute,
    ///         NodeKind::Element,
    ///         NodeKind::Namespace,
    ///     ]
    /// );
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    #[must_use]
    pub fn nodes_in_document_order(&self) -> Vec<NodeRef> {
        let store = self.erased_store();
        (0..self.store.len())
            .map(|index| NodeRef::new(Arc::clone(&store), NodeId(index as u64)))
            .collect()
    }

    /// A handle to the node with the given store-local identifier, if it exists.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<NodeRef> {
        (usize::try_from(id.0).ok()? < self.store.len())
            .then(|| NodeRef::new(self.erased_store(), id))
    }

    fn erased_store(&self) -> Arc<dyn crate::node::XdmStore> {
        Arc::clone(&self.store) as Arc<dyn crate::node::XdmStore>
    }
}
